"""Rank steady highway windows for manual multi-camera calibration."""
import argparse
import json
from pathlib import Path

import capnp
import cv2
import numpy as np
import zstandard

from motion_alignment import images
from stitch import log_events, timing


def windows(path):
    motion, poses = [], []
    for event in log_events(path):
        try:
            kind = event.which()
        except capnp.KjException:
            continue
        if kind == 'cameraOdometry':
            o = event.cameraOdometry
            if len(o.trans) == len(o.rot) == 3 and o.timestampEof:
                motion.append([o.timestampEof, o.trans[0], *o.rot])
        elif kind == 'extrinsicsCalibration':
            c = event.extrinsicsCalibration
            if c.calStatus == 'calibrated' and len(c.rpyCalib) == 3:
                poses.append([event.logMonoTime, *c.rpyCalib])
    if len(motion) < 50:
        return []
    rows = np.asarray(motion)
    prior = np.asarray(poses)
    results = []
    start = rows[0, 0]
    for left in np.arange(start, rows[-1, 0]-3e9, 5e9):
        sel = rows[(rows[:, 0] >= left) & (rows[:, 0] < left+5e9)]
        if len(sel) < 65 or sel[-1, 0]-sel[0, 0] < 4e9:
            continue
        speed = np.median(sel[:, 1])
        p90 = np.quantile(np.abs(sel[:, 2:]), .9, axis=0)
        if speed < 23 or np.ptp(sel[:, 1]) > 4 or max(p90) > .025:
            continue
        mid = int(sel[len(sel)//2, 0])
        attitude = None
        if len(prior):
            i = np.argmin(np.abs(prior[:, 0]-mid))
            if abs(prior[i, 0]-mid) < 10e9:
                attitude = np.degrees(prior[i, 1:]).tolist()
        results.append(dict(segment=path.parent.name, mono_ns=mid, speed_mps=float(speed),
                            rate_p90_deg_s=np.degrees(p90).tolist(), road_calib_rpy_deg=attitude,
                            score=float(np.linalg.norm(p90))))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--output', type=Path, default=Path('previews/highway-survey'))
    args = ap.parse_args()
    if args.stride < 1:
        ap.error('stride must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    candidates = []
    for path in sorted(Path('rlogs').glob('*/rlog.zst'))[::args.stride]:
        seg = Path(path.parent.name)
        if not all((seg/f'{k}camera.hevc').is_file() for k in 'ed'):
            continue
        try:
            found = windows(path)
        except (OSError, ValueError, capnp.KjException, zstandard.ZstdError):
            continue
        candidates.extend(found)
        print(seg, len(found), flush=True)
    candidates.sort(key=lambda x: x['score'])
    (args.output/'windows.json').write_text(json.dumps(candidates, indent=2)+'\n')
    used = set()
    exported = []
    for row in candidates:
        seg = Path(row['segment'])
        route = seg.name.rsplit('--', 1)[0]
        if route in used:
            continue
        try:
            ts, _ = timing(seg)
            frames = [images(seg/f'{k}camera.hevc', [int(np.argmin(np.abs(ts[k]-row['mono_ns'])))])[0] for k in 'ed']
        except (OSError, ValueError, capnp.KjException, zstandard.ZstdError):
            continue
        for k, frame in zip('ed', frames):
            cv2.imwrite(str(args.output/f'{seg.name}-{k}.jpg'), cv2.resize(frame, (672, 380)))
        used.add(route)
        exported.append(row)
        if len(used) == 8:
            break
    (args.output/'exported.json').write_text(json.dumps(exported, indent=2)+'\n')
    print(f'{len(candidates)} steady windows; exported paired frames from {len(used)} drives', flush=True)


if __name__ == '__main__':
    main()
