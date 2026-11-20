"""Collect per-camera feature tracks with gyro rotation and pose velocity for rig calibration.

For sampled instants in every segment (turns, pitch/roll events and straight
driving) each camera contributes short feature tracks between two of its own
frames. The gyro, integrated between that camera's own exposure times, gives
the device rotation; deviceMotion gives the direction of travel in the device
frame. No shared scenery between cameras is needed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess as sp

import capnp
import cv2
import imageio_ffmpeg
import numpy as np
from scipy.spatial.transform import Rotation
import zstandard

from stitch import log_events

FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = Path('rig/tracks')
ENC = {'e': 'wideRoadEncodeIdx', 'd': 'cabinEncodeIdx', 'f': 'narrowRoadEncodeIdx'}
W, H = 1344, 760


def device(v):
    # Sensor axes to openpilot device axes (forward, right, down), as in locationd.
    return [-v[2], -v[1], -v[0]]


def parse(seg):
    gyro, acc, dm, calib = [], [], [], None
    enc = {k: {} for k in ENC}
    for e in log_events(Path('rlogs', seg, 'rlog.zst')):
        try:
            kind = e.which()
        except capnp.KjException:
            continue
        if kind == 'gyroscope':
            g = e.gyroscope
            if g.which() == 'gyroUncalibrated' and len(g.gyroUncalibrated.v) == 3:
                gyro.append([g.timestamp, *device(g.gyroUncalibrated.v)])
        elif kind == 'accelerometer':
            a = e.accelerometer
            if a.which() == 'acceleration' and len(a.acceleration.v) == 3:
                acc.append([a.timestamp, *device(a.acceleration.v)])
        elif kind == 'deviceMotion':
            p = e.deviceMotion
            if p.inputsOK and p.sensorsOK and p.posenetOK and p.velocityDevice.valid and p.angularVelocityDevice.valid:
                v, w = p.velocityDevice, p.angularVelocityDevice
                dm.append([p.timestamp, v.x, v.y, v.z, w.x, w.y, w.z])
        elif kind == 'extrinsicsCalibration':
            c = e.extrinsicsCalibration
            if str(c.calStatus) == 'calibrated' and len(c.rpyCalib) == 3 and len(c.wideFromDeviceEuler) == 3:
                calib = [*c.rpyCalib, *c.wideFromDeviceEuler]
        else:
            for k, name in ENC.items():
                if kind == name:
                    i = getattr(e, name)
                    if i.timestampSof and i.timestampEof:
                        enc[k][i.segmentId] = (i.timestampSof + i.timestampEof)/2
    return np.array(gyro, float), np.array(acc, float), np.array(dm, float), calib, enc


def pick(times, score, ok, count, gap=1.5e9):
    chosen = []
    for i in np.argsort(-score):
        if not ok[i] or len(chosen) == count:
            continue
        if all(abs(times[i]-times[j]) > gap for j in chosen):
            chosen.append(i)
    return chosen


def decode(path, indices):
    expr = '+'.join(f'eq(n\\,{i})' for i in indices)
    out = sp.run([FF, '-v', 'error', '-threads', '1', '-i', str(path), '-vf', f"select='{expr}'",
                  '-fps_mode', 'passthrough', '-frames:v', str(len(indices)), '-pix_fmt', 'gray',
                  '-f', 'rawvideo', '-'], capture_output=True).stdout
    if len(out) != len(indices)*W*H:
        raise ValueError('short decode')
    return dict(zip(indices, np.frombuffer(out, np.uint8).reshape(-1, H, W)))


def track(a, b, mask, count):
    p = cv2.goodFeaturesToTrack(a, count, .01, 9, mask=mask)
    if p is None:
        return np.empty((0, 4), np.float32)
    opts = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    q, s0, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **opts)
    r, s1, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **opts)
    p, q, r = p[:, 0], q[:, 0], r[:, 0]
    move = np.linalg.norm(q-p, axis=1)
    xy = np.clip(q.round().astype(int), 0, [W-1, H-1])
    keep = ((s0[:, 0] > 0) & (s1[:, 0] > 0) & (np.linalg.norm(r-p, axis=1) < .5) & (move > 1.5) & (move < 160)
            & (mask[xy[:, 1], xy[:, 0]] > 0))
    return np.hstack([p[keep], q[keep]]).astype(np.float32)


def work(seg):
    try:
        gyro, acc, dm, calib, enc = parse(seg)
        if len(gyro) < 2000 or len(dm) < 300 or any(len(enc[k]) < 100 for k in ENC):
            return seg, 'insufficient log data'
        gyro = gyro[np.argsort(gyro[:, 0])]
        dm = dm[np.argsort(dm[:, 0])]
        raw_at_dm = np.column_stack([np.interp(dm[:, 0], gyro[:, 0], gyro[:, i]) for i in (1, 2, 3)])
        bias = np.median(raw_at_dm - dm[:, 4:7], axis=0)
        corr = [float(np.corrcoef(raw_at_dm[:, i], dm[:, 4+i])[0, 1]) for i in range(3)]
        rate = lambda t: np.array([np.interp(t, gyro[:, 0], gyro[:, i]) for i in (1, 2, 3)]) - bias

        def rotation(t0, t1):
            inside = gyro[(gyro[:, 0] > t0) & (gyro[:, 0] < t1), 0]
            knots = np.r_[t0, inside, t1]
            total = Rotation.identity()
            for a, b in zip(knots, knots[1:]):
                total = total * Rotation.from_rotvec(rate((a+b)/2)*(b-a)/1e9)
            return total.as_rotvec()

        t, speed = dm[:, 0], np.linalg.norm(dm[:, 1:4], axis=1)
        w = dm[:, 4:7]
        inner = (t > t[0]+1e9) & (t < t[-1]-1e9)
        samples = []
        for cat, idx in ((1, pick(t, np.abs(w[:, 2]), inner & (speed > 2) & (np.abs(w[:, 2]) > .05), 3)),
                         (2, pick(t, np.hypot(w[:, 0], w[:, 1]), inner & (speed > 2) & (np.hypot(w[:, 0], w[:, 1]) > .03), 2)),
                         (0, pick(t, -np.linalg.norm(w, axis=1), inner & (speed > 5) & (np.linalg.norm(w, axis=1) < .012), 3, 6e9))):
            samples += [(cat, t[i], speed[i]) for i in idx if all(abs(t[i]-s[1]) > 1e9 for s in samples)]
        if not samples:
            return seg, 'no usable motion'
        result = dict(gyro_bias=bias, gyro_dm_corr=corr, calib=np.array(calib if calib else [np.nan]*6),
                      gravity=-np.median(acc[:, 1:], axis=0) if len(acc) else np.full(3, np.nan))
        for k in ENC:
            mids = np.array([enc[k].get(i, np.nan) for i in range(max(enc[k])+1)])
            mask = cv2.imread(f'rig/masks/{k}.png', 0)
            plan = []
            for cat, ts, v in samples:
                i0 = int(np.nanargmin(np.abs(mids-ts)))
                gap = 1 if (k == 'd' and v > 14) else 2
                if i0+gap < len(mids) and np.isfinite(mids[[i0, i0+gap]]).all() and abs(mids[i0]-ts) < 6e7:
                    plan.append((cat, i0, i0+gap))
            frames = decode(Path(seg, f'{k}camera.hevc'), sorted({i for _, a, b in plan for i in (a, b)})) if plan else {}
            pairs, tracks = [], []
            for cat, i0, i1 in plan:
                found = track(frames[i0], frames[i1], mask, 300 if k == 'd' else 350)
                if len(found) < 8:
                    continue
                t0, t1 = mids[i0], mids[i1]
                vel = [np.interp((t0+t1)/2, t, dm[:, i]) for i in (1, 2, 3)]
                tracks.append(np.column_stack([np.full(len(found), len(pairs), np.float32), found]))
                pairs.append([cat, i0, i1, (t1-t0)/1e9, *rotation(t0, t1), *rate(t0), *rate(t1), *vel])
            result[f'{k}_pairs'] = np.array(pairs, float).reshape(-1, 16)
            result[f'{k}_tracks'] = np.concatenate(tracks).astype(np.float32) if tracks else np.empty((0, 5), np.float32)
        np.savez_compressed(OUT/f'{seg}.npz', **result)
        return seg, ' '.join(f"{k}:{len(result[f'{k}_pairs'])}/{len(result[f'{k}_tracks'])}" for k in ENC)
    except (OSError, ValueError, capnp.KjException, zstandard.ZstdError, cv2.error) as exc:
        return seg, f'{type(exc).__name__}: {str(exc)[:80]}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    segs = a.only or sorted(p.parent.name for p in Path('rlogs').glob('*/rlog.zst')
                            if all(Path(p.parent.name, f'{k}camera.hevc').is_file() for k in ENC))[::a.stride]
    segs = [s for s in segs if not (OUT/f'{s}.npz').exists()]
    with ProcessPoolExecutor(a.jobs) as pool:
        for seg, note in pool.map(work, segs):
            print(seg, note, flush=True)


if __name__ == '__main__':
    main()
