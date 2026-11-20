import argparse
import datetime as dt
import json
from pathlib import Path
import re
import struct
import subprocess as sp
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
    def vp(axis):
        normals = []
        for line in cam[axis]:
            p = (np.asarray(line) - cam['center']) / cam['focal']
            if cam['model'] == 'equidistant':
                a = np.linalg.norm(p, axis=1)
                rays = np.column_stack((p * np.sinc(a[:, None]/np.pi), np.cos(a)))
            else:
                rays = unit(np.column_stack((p, np.ones(len(p)))))
            normals.append(np.linalg.svd(rays, full_matrices=True)[2][-1])
        if len(normals) < 2 or np.linalg.matrix_rank(normals, tol=1e-3) < 2:
            raise ValueError(f'Need distinct {axis} lines')
        return np.linalg.svd(normals)[2][-1]
    z, y = vp('longitudinal'), vp('vertical')
    z *= (1 if z[2] >= 0 else -1) * cam.get('facing', 1)
    y *= 1 if y[1] >= 0 else -1
    y = unit(y - z * (z @ y))
    return np.column_stack((np.cross(y, z), y, z))


def maps(cam, sphere):
    x, y, z = (sphere @ np.asarray(cam['basis'])).transpose(2, 0, 1)
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
    lon, lat = np.meshgrid(((np.arange(width)+.5)/width-.5)*2*np.pi,
                           ((np.arange(h)+.5)/h-.5)*np.pi)
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
        stamp_w = 310
        stamp = next(((x, y) for y in range(h-26, -1, -8) for x in (10, width-stamp_w-10, (width-stamp_w)//2)
                      if not covered[y:y+26, x:x+stamp_w].any()), None)
        if stamp is None:
            raise ValueError('no uncovered sphere area for timestamp')
        writer = sp.Popen(ff('-y', '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{h}', '-r','20','-i','pipe:0', *audio,
                            '-map','0:v', *(['-map','1:a','-c:a','aac','-af','aresample=async=1:first_pts=0,apad'] if audio else []),
                            '-c:v','libx264','-preset','fast','-threads','4','-pix_fmt','yuv420p','-t',str(len(ticks)/20), path), stdin=sp.PIPE)
        positions = dict.fromkeys('edf', -1)
        frames = {}
        for t in ticks:
            out = np.zeros((h,width,3), np.uint8)
            for k in 'edf':
                ts = times[k]
                i = int(np.clip(np.searchsorted(ts,t),0,len(ts)-1))
                if i and abs(t-ts[i-1]) <= abs(t-ts[i]):
                    i -= 1
                while positions[k] < i:
                    data = next(readers[k])
                    positions[k] += 1
                if k not in frames or frames[k][0] != i:
                    w, ch = calibration[k]['size']
                    xy, mask = warps[k]
                    frames[k] = (i, cv2.remap(np.frombuffer(data,np.uint8).reshape(ch,w,3), *xy, cv2.INTER_LINEAR))
                mask = warps[k][1]
                np.copyto(out[:h], frames[k][1], where=mask[:, :, None])
            text = dt.datetime.fromtimestamp((int(t)+offset)/1e9,dt.UTC).isoformat(timespec='milliseconds').replace('+00:00','Z')
            cv2.putText(out,text,(stamp[0],stamp[1]+20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            writer.stdin.write(out.tobytes())
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
    parser = argparse.ArgumentParser()
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--solve', action='store_true', help='fit rotations from annotated parallel lines and print calibration')
    parser.add_argument('--drive', help='route prefix, without final segment number')
    parser.add_argument('--segment', help='render just this segment')
    parser.add_argument('--frames', type=int, help='limit frames per segment for previews')
    parser.add_argument('--width', type=int, default=4096)
    parser.add_argument('--output', type=Path, default=Path('videos'))
    a = parser.parse_args()
    cfg = json.loads(a.calibration.read_text())
    if a.solve:
        for cam in cfg.values():
            if 'reference' not in cam:
                cam['basis'] = frame_basis(cam).tolist()
        for cam in cfg.values():
            if 'reference' in cam:
                cam['basis'] = (np.diag([-1, 1, -1]) @ cfg[cam['reference']]['basis']).tolist()
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
    for route, segments in sorted(routes.items()):
        dest = a.output / f'{a.segment or route}.mp4'
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
                continue
            manifest = tmp/'concat.txt'
            manifest.write_text(''.join(f"file '{p}'\n" for p in parts))
            assembled = tmp/'assembled.mp4'
            sp.run(ff('-f','concat','-safe','1','-i',manifest,'-c','copy','-movflags','+faststart',assembled),check=True)
            assembled.replace(dest)


if __name__ == '__main__':
    main()
