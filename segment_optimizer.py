"""Bounded cabin rotation fit over a whole highway segment, using static flow.

The cross-camera residual is angular: the forward travel direction inferred
from external-camera flow should lie in each cabin feature's motion plane.
No RGB residual is claimed where there is no shared field of view.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration_survey import rays
from highway_survey import windows
from rear_window_survey import WINDOWS
from stitch import imageio_ffmpeg, timing, unit


def frames(path, wanted):
    reader = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '1'])
    try:
        if next(reader)['size'] != (1344, 760):
            raise ValueError('unsupported image size')
        last = max(wanted)
        out = {}
        for i, frame in enumerate(reader):
            if i in wanted:
                im = np.frombuffer(frame, np.uint8).reshape(760, 1344, 3)
                out[i] = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if i >= last:
                break
        return out
    finally:
        reader.close()


def normals(a, b, mask, focal):
    p = cv2.goodFeaturesToTrack(a, 120, .012, 10, mask=mask)
    if p is None:
        return np.empty((0, 3))
    opts = dict(winSize=(17, 17), maxLevel=3)
    q, status, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **opts)
    if q is None:
        return np.empty((0, 3))
    back, valid, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **opts)
    if back is None:
        return np.empty((0, 3))
    p, q = p[:, 0], q[:, 0]
    movement = np.linalg.norm(q-p, axis=1)
    xy = np.clip(q.round().astype(int), [0, 0], [1343, 759])
    keep = ((status[:, 0] > 0) & (valid[:, 0] > 0) & (np.linalg.norm(back[:, 0]-p, axis=1) < .8)
            & (movement > 2) & (movement < 70) & (mask[xy[:, 1], xy[:, 0]] != 0))
    n = np.cross(rays(p[keep], focal, True), rays(q[keep], focal, True))
    return unit(n)


def travel_direction(n, initial):
    if len(n) < 20:
        return None
    # Robust, tightly bounded estimate; each block can have a slightly
    # different heading or grade, but not an arbitrary motion direction.
    def residual(x):
        v = Rotation.from_rotvec(x).apply(initial)
        return np.r_[n @ v, .25*x]
    fit = least_squares(residual, np.zeros(3), bounds=(-.045, .045), loss='soft_l1', f_scale=.035)
    v = Rotation.from_rotvec(fit.x).apply(initial)
    if np.median(np.abs(n @ v)) > .12:
        return None
    return v


def collect(segment, cfg):
    log = Path('rlogs', segment.name, 'rlog.zst')
    blocks = windows(log)
    if not blocks:
        return []
    times, _ = timing(segment)
    schedule = []
    for block, row in enumerate(blocks):
        for offset in (-1_500_000_000, -500_000_000, 500_000_000, 1_500_000_000):
            start = row['mono_ns'] + offset
            ticks = [start, start+250_000_000]
            indices = {k: [int(np.argmin(np.abs(times[k]-t))) for t in ticks] for k in 'ed'}
            if all(abs(int(times[k][i])-t) < 40_000_000 for k in 'ed' for i, t in zip(indices[k], ticks)):
                schedule.append((block, indices))
    if not schedule:
        return []
    # Store only reduced grayscale samples, not the full video. 80 pairs
    # per typical segment is under 170 MiB per camera at this geometry.
    decoded = {k: frames(segment/f'{k}camera.hevc', {i for _, row in schedule for i in row[k]}) for k in 'ed'}
    wide = np.zeros((760, 1344), np.uint8)
    wide[215:505, 90:-90] = 255
    masks = {k: cv2.fillConvexPoly(np.zeros_like(wide), 2*np.asarray(poly, np.int32), 255)
             for k, poly in WINDOWS.items()}
    be = np.array(cfg['e']['basis'])
    records = []
    for block, idx in schedule:
        if any(i not in decoded[k] for k in 'ed' for i in idx[k]):
            continue
        e = normals(*(decoded['e'][i] for i in idx['e']), wide, cfg['e']['focal'])
        v = travel_direction(e @ be, np.array([0., 0., 1.]))
        if v is None:
            continue
        for side, mask in masks.items():
            n = normals(*(decoded['d'][i] for i in idx['d']), mask, cfg['d']['focal'])
            if len(n) >= 5:
                records.append((block, side, n, v))
    return records


def optimize(records, seed):
    def errors(x, rows):
        b = seed @ Rotation.from_rotvec([0., x[0], 0.]).as_matrix()
        return [(side, n @ (b @ v)) for _, side, n, v in rows]

    def objective(x, rows):
        groups = errors(x, rows)
        return np.concatenate([r/np.sqrt(len(r)) for _, r in groups] + [1.8*x])

    train, test = ([r for r in records if (r[0] % 3 == 0) == test] for test in (False, True))
    if len({r[0] for r in train}) < 3 or len({r[0] for r in test}) < 2:
        raise ValueError('insufficient independent time blocks')
    # Only yaw changes: rotation about world down preserves the horizon.
    x = np.zeros(1)
    trace = []
    bound = np.radians(4.)
    step = np.radians(.25)
    for _ in range(20):
        lo, hi = np.maximum(-bound, x-step), np.minimum(bound, x+step)
        target = least_squares(objective, x, args=(train,), bounds=(lo, hi),
                               loss='soft_l1', f_scale=.025).x
        trace.append(dict(delta_deg=np.degrees(target).tolist(), step_deg=float(np.degrees(np.linalg.norm(target-x)))))
        if np.linalg.norm(target-x) < 1e-5:
            x = target
            break
        x = target

    def rms(rows, side=None):
        values = []
        for delta in (np.zeros(1), x):
            groups = [r for k, r in errors(delta, rows) if side is None or k == side]
            values.append(float(np.sqrt(np.mean([np.mean(np.degrees(np.arcsin(np.clip(r, -1, 1)))**2)
                                                 for r in groups]))) if groups else None)
        return values
    return seed @ Rotation.from_rotvec([0., x[0], 0.]).as_matrix(), dict(steps=trace,
           training_rms_deg=rms(train), heldout_rms_deg=rms(test),
           heldout_by_window={k: rms(test, k) for k in WINDOWS},
           training_groups=len(train), heldout_groups=len(test))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('segment', type=Path)
    parser.add_argument('--calibration', type=Path, default=Path('calibration-best.json'))
    parser.add_argument('--output', type=Path, default=Path('previews/segment-optimizer'))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cfg = json.loads(args.calibration.read_text())
    records = collect(args.segment, cfg)
    basis, report = optimize(records, np.array(cfg['d']['basis']))
    cfg['d']['basis'] = basis.tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'candidate.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'steps'}, indent=2))


if __name__ == '__main__':
    main()
