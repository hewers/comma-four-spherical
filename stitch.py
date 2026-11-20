import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import re
import signal
import struct
import subprocess as sp
import sys
import tempfile

import capnp
import cv2
import imageio_ffmpeg
import numpy as np
import zstandard

FF = imageio_ffmpeg.get_ffmpeg_exe()
SCHEMA = capnp.load(str(Path(__file__).parent / 'schemas/log.capnp'))
CAM = {'e': 'wideRoadEncodeIdx', 'd': 'cabinEncodeIdx', 'f': 'narrowRoadEncodeIdx', 'q': 'qNarrowRoadEncodeIdx'}


def log_events(path):
    # Cap'n Proto stream framing: padded segment table followed by words.
    with path.open('rb') as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError('truncated rlog header')
            count, first = struct.unpack('<II', header)
            count += 1
            if count > 512:
                raise ValueError('invalid rlog segment table')
            table_size = 4 * (count - 1 + (count % 2 == 0))
            table = stream.read(table_size)
            if len(table) != table_size:
                raise ValueError('truncated rlog segment table')
            size = 8 * (first + sum(struct.unpack_from(f'<{count-1}I', table)))
            if size > 64 * 1024 * 1024:
                raise ValueError('oversized rlog message')
            body = stream.read(size)
            if len(body) != size:
                raise ValueError('truncated rlog message')
            with SCHEMA.Event.from_bytes(header + table + body) as event:
                yield event


def timing(segment):
    times = {k: [] for k in CAM}
    offsets = []
    fallback = []
    for event in log_events(Path('rlogs', segment.name, 'rlog.zst')):
        try:
            kind = event.which()
        except capnp.KjException:
            continue  # Newer, unrelated event discriminant.
        if kind == 'initData' and event.initData.wallTimeNanos > 10**18:
            fallback.append(event.initData.wallTimeNanos - event.logMonoTime)
        if kind == 'clocks' and event.clocks.wallTimeNanos > 10**18:
            offsets.append(event.clocks.wallTimeNanos - event.logMonoTime)
        for key, name in CAM.items():
            if kind == name:
                idx = getattr(event, name)
                if idx.timestampSof:
                    times[key].append((idx.segmentId, idx.timestampSof))
    offsets = offsets or fallback
    if not offsets or any(not times[k] for k in 'edf'):
        raise ValueError(f'{segment}: missing wall clock or camera timing')
    return {k: np.array([t for _, t in sorted(v)], dtype=np.int64) for k, v in times.items()}, sorted(offsets)[len(offsets)//2]


def unit(r):
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def frame_basis(cam):
    # Trace at least two parallel physical lines per axis. Each line is a
    # great circle after unprojection; its plane normal constrains the VP.
    def plane(points):
        p = (np.asarray(points) - cam['center']) / cam['focal']
        if cam['model'] == 'equidistant':
            a = np.linalg.norm(p, axis=1)
            rays = np.column_stack((p * np.sinc(a[:, None]/np.pi), np.cos(a)))
        else:
            rays = unit(np.column_stack((p, np.ones(len(p)))))
        return np.linalg.svd(rays, full_matrices=True)[2][-1]

    def vp(axis):
        normals = [plane(line) for line in cam[axis]]
        if len(normals) < 2 or np.linalg.matrix_rank(normals, tol=1e-3) < 2:
            raise ValueError(f'Need distinct {axis} lines')
        return np.linalg.svd(normals)[2][-1]
    if 'horizon' in cam:
        y = plane(cam['horizon'])
        directions = [unit(np.cross(y, plane(line))) for line in cam['longitudinal']]
        z = unit(sum(d if d[2] >= 0 else -d for d in directions))
    else:
        z, y = vp('longitudinal'), vp('vertical')
    z *= (1 if z[2] >= 0 else -1) * cam.get('facing', 1)
    y *= 1 if y[1] >= 0 else -1
    y = unit(y - z * (z @ y))
    return np.column_stack((np.cross(y, z), y, z))


def maps(cam, sphere):
    x, y, z = (sphere @ np.asarray(cam['basis'], dtype=sphere.dtype)).transpose(2, 0, 1)
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    if cam['model'] == 'equidistant':
        scale = np.divide(theta, r, out=np.ones_like(r), where=r > 1e-8)
        valid = theta <= np.deg2rad(cam['max_angle'])
    else:
        valid = z > 1e-6
        scale = 1 / np.maximum(z, 1e-6)
    u, v = (x*scale*cam['focal'] + cam['center'][0], y*scale*cam['focal'] + cam['center'][1])
    w, h = cam['size']
    valid &= (u >= 0) & (u < w-1) & (v >= 0) & (v < h-1)
    return tuple(np.where(valid, p, -1).astype('float32') for p in (u, v)), valid


def ff(*args):
    return [FF, '-hide_banner', '-loglevel', 'error', '-nostdin', *map(str, args)]


def render(seg, calibration, path, width, limit):
    missing = [f'{k}camera.hevc' for k in 'edf' if not (seg/f'{k}camera.hevc').is_file()]
    if missing:
        raise ValueError('missing ' + ', '.join(missing))
    times, offset = timing(seg)
    h = width//2
    lon, lat = np.meshgrid(((np.arange(width, dtype=np.float32)+.5)/width-.5)*2*np.pi,
                           ((np.arange(h, dtype=np.float32)+.5)/h-.5)*np.pi)
    sphere = np.stack((np.sin(lon)*np.cos(lat), np.sin(lat), np.cos(lon)*np.cos(lat)), axis=-1)
    start = max(times[k][0] for k in 'edf')
    end = min(times[k][-1] for k in 'edf')
    ticks = np.arange(start, end+1, 50_000_000, dtype=np.int64)
    if limit:
        ticks = ticks[:limit]
    if not len(ticks):
        raise ValueError(f'{seg}: no common camera time')
    audio_path = seg/'qcamera.ts'
    # A TS audio/video offset is preserved by FFmpeg. The first qRoad
    # capture time anchors that container to the high-quality streams.
    audio = []
    if audio_path.exists() and len(times['q']):
        probe = sp.run([FF, '-hide_banner', '-i', str(audio_path)], capture_output=True, text=True).stderr
        if 'Audio:' in probe:
            shift = (int(times['q'][0])-int(start))/1e9
            audio = ['-itsoffset', str(shift), '-i', audio_path]
    if not audio:
        print(f'{seg}: no usable qcamera audio, writing silence', flush=True)
        audio = ['-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono']
    readers = {}
    writer = None
    try:
        for k in 'edf':
            readers[k] = imageio_ffmpeg.read_frames(str(seg/f'{k}camera.hevc'), input_params=['-threads','1'])
        warps = {}
        for k, reader in readers.items():
            meta = next(reader)
            if tuple(calibration[k]['size']) != meta['size']:
                raise ValueError(f'{seg}/{k}: calibration image size mismatch: {meta["size"]}')
            warps[k] = maps(calibration[k], sphere)
        covered = np.logical_or.reduce([mask for _, mask in warps.values()])
        external = warps['e'][1]
        ey = np.flatnonzero(external.any(axis=1))
        ex = np.flatnonzero(external.any(axis=0))
        if not len(ex):
            raise ValueError('empty external camera coverage')
        stamp_w = max(1, (int(ex[-1])-int(ex[0])+1)//2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        sample = '2000-00-00T00:00:00.000Z'
        scale = stamp_w / cv2.getTextSize(sample, font, 1, 1)[0][0]
        thickness = max(1, round(2*scale))
        (tw, th), baseline = cv2.getTextSize(sample, font, scale, thickness)
        pad = max(2, round(3*scale))
        stamp_h = th + baseline + 2*pad
        x = (int(ex[0])+int(ex[-1])+1-tw)//2
        stamp = next(((x, y+pad+th) for y in range(int(ey[-1])+1+pad, h-stamp_h+1, pad)
                      if not covered[y:y+stamp_h, x-pad:x+tw+pad].any()), None)
        del external
        if stamp is None:
            raise ValueError('no uncovered area below external camera for timestamp')
        # Use only the bounding rectangle of each camera, and write valid
        # pixels directly into the output without a full-size warped copy.
        regions = {}
        for k, (xy, mask) in warps.items():
            regions[k] = []
            spans = ((0, width//2), (width//2, width)) if k == 'd' else ((0, width),)
            for left, right in spans:
                sub = mask[:, left:right]
                xs = np.flatnonzero(sub.any(axis=0))
                if not len(xs):
                    continue
                ys = np.flatnonzero(sub.any(axis=1))
                roi = np.s_[ys[0]:ys[-1]+1, left+xs[0]:left+xs[-1]+1]
                regions[k].append((roi, cv2.convertMaps(*(p[roi] for p in xy), cv2.CV_16SC2)))
            if not regions[k]:
                raise ValueError(f'{seg}/{k}: empty camera coverage')
        del warps, covered, sphere, lon, lat, xy, mask, sub
        writer = sp.Popen(ff('-y', '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{h}', '-r','20','-i','pipe:0', *audio,
                            '-map','0:v', *(['-map','1:a','-c:a','aac','-af','aresample=async=1:first_pts=0,apad'] if audio else []),
                            '-c:v','libx264','-preset','veryfast','-crf','19','-threads','4','-pix_fmt','yuv420p','-t',str(len(ticks)/20), path), stdin=sp.PIPE)
        positions = dict.fromkeys('edf', -1)
        frames = {}
        out = np.empty((h,width,3), np.uint8)
        for t in ticks:
            out.fill(0)
            for k in 'edf':
                ts = times[k]
                i = int(np.clip(np.searchsorted(ts,t),0,len(ts)-1))
                if i and abs(t-ts[i-1]) <= abs(t-ts[i]):
                    i -= 1
                while positions[k] < i:
                    frames[k] = next(readers[k])
                    positions[k] += 1
                w, ch = calibration[k]['size']
                image = np.frombuffer(frames[k],np.uint8).reshape(ch,w,3)
                for roi, xy in regions[k]:
                    cv2.remap(image, *xy, cv2.INTER_LINEAR, dst=out[roi], borderMode=cv2.BORDER_TRANSPARENT)
            text = dt.datetime.fromtimestamp((int(t)+offset)/1e9,dt.UTC).isoformat(timespec='milliseconds').replace('+00:00','Z')
            cv2.putText(out, text, stamp, font, scale, (255,255,255), thickness, cv2.LINE_AA)
            writer.stdin.write(memoryview(out).cast('B'))
        writer.stdin.close()
        if writer.wait():
            raise RuntimeError('FFmpeg encoding failed')
    finally:
        for reader in readers.values():
            reader.close()
        if writer and writer.poll() is None:
            writer.kill()
            writer.wait()


def main():
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--solve', action='store_true', help='fit rotations from annotated parallel lines and print calibration')
    parser.add_argument('--drive', help='route prefix, without final segment number')
    parser.add_argument('--segment', help='render just this segment')
    parser.add_argument('--frames', type=int, help='limit frames per segment for previews')
    parser.add_argument('--width', type=int, default=4096)
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=1, help='maximum parallel drives')
    parser.add_argument('--resume', action='store_true', help='skip already published drives in this output directory')
    parser.add_argument('--output', type=Path, default=Path('videos'))
    a = parser.parse_args()
    cfg = json.loads(a.calibration.read_text())
    if a.solve:
        for cam in cfg.values():
            cam['basis'] = frame_basis(cam).tolist()
        print(json.dumps(cfg, indent=2))
        return
    if a.width < 320 or a.width % 4 or (a.frames is not None and a.frames < 1):
        parser.error('width must be a multiple of four and at least 320; frames must be positive')
    routes = {}
    for p in Path('.').iterdir():
        m = re.fullmatch(r'([\da-f]+--[\da-f]+)--(\d+)', p.name)
        if p.is_dir() and m and (not a.drive or m[1] == a.drive) and (not a.segment or p.name == a.segment):
            routes.setdefault(m[1], []).append((int(m[2]), p))
    if not routes:
        parser.error('no matching segments')
    a.output.mkdir(parents=True, exist_ok=True)
    if a.jobs > 1 and len(routes) > 1:
        def worker(route):
            args = ['--drive', route, '--width', str(a.width), '--output', str(a.output)]
            if a.frames:
                args += ['--frames', str(a.frames)]
            if a.resume:
                args += ['--resume']
            result = sp.run([sys.executable, __file__, str(a.calibration), *args])
            if result.returncode:
                print(f'FAILED drive {route}: exit {result.returncode}', flush=True)
            return result.returncode
        ordered = sorted(routes, key=lambda route: -len(routes[route]))
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            failures = sum(code != 0 for code in pool.map(worker, ordered))
        print(f'Batch finished: {failures} drive process failures; see skipped segments above.', flush=True)
        return
    for route, segments in sorted(routes.items()):
        dest = a.output / f'{a.segment or route}.mp4'
        if a.resume and dest.is_file():
            continue
        with tempfile.TemporaryDirectory(dir=a.output) as tmp:
            tmp = Path(tmp)
            parts = []
            for number, seg in sorted(segments):
                print(f'Rendering {seg}', flush=True)
                part = tmp/f'{number}.mp4'
                try:
                    render(seg, cfg, part, a.width, a.frames)
                except (OSError, ValueError, RuntimeError, StopIteration, capnp.KjException, zstandard.ZstdError) as exc:
                    if a.segment:
                        raise
                    print(f'Skipping {seg}: {type(exc).__name__}: {exc}', flush=True)
                    continue
                parts.append(part.name)
            if not parts:
                print(f'No renderable segments for {route}', flush=True)
                continue
            manifest = tmp/'concat.txt'
            manifest.write_text(''.join(f"file '{p}'\n" for p in parts))
            assembled = tmp/'assembled.mp4'
            sp.run(ff('-f','concat','-safe','1','-i',manifest,'-c','copy','-movflags','+faststart',assembled),check=True)
            for name in parts:
                (tmp/name).unlink()
            tagged = tmp/'spherical.mp4'
            sp.run([sys.executable, '-m', 'spatialmedia', '-i', str(assembled), str(tagged)], check=True)
            tagged.replace(dest)
            print(f'Published {dest}', flush=True)


if __name__ == '__main__':
    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    main()
