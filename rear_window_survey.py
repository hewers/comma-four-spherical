"""Estimate the cabin's rearward vanishing direction from window optical flow."""
import argparse
import json
from pathlib import Path

import capnp
import cv2
import numpy as np

from calibration_survey import rays
from stitch import imageio_ffmpeg, log_events, unit

# Pixel polygons at 672x380. Narrow rear-glass band excludes the central
# headrests as much as possible; forward/backward tracking rejects occlusions.
WINDOWS = {
    'rear': [[299, 100], [395, 100], [392, 116], [304, 116]],
    'left': [[112, 128], [195, 111], [177, 158], [67, 224]],
    'right': [[508, 124], [561, 147], [609, 206], [600, 237], [520, 174]],
}


def straight_motion(segment):
    speed, rotation = [], []
    path = Path('rlogs', segment.name, 'rlog.zst')
    for event in log_events(path):
        try:
            kind = event.which()
        except capnp.KjException:
            continue
        if kind == 'cameraOdometry':
            odom = event.cameraOdometry
            if len(odom.trans) == 3 and len(odom.rot) == 3:
                speed.append(odom.trans[0])
                rotation.append(np.linalg.norm(list(odom.rot)))
            if len(speed) == 12:
                break
    return len(speed) >= 8 and np.median(speed) > 12 and np.median(rotation) < .02


def tracks(path, masks, focal):
    reader = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '1'])
    try:
        w, h = next(reader)['size']
        if (w, h) != (1344, 760):
            return {}
        images = []
        for i in range(13):
            rgb = next(reader)
            if i % 6 == 0:
                im = np.frombuffer(rgb, np.uint8).reshape(h, w, 3)
                images.append(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))
    finally:
        reader.close()
    result = {}
    for label, mask in masks.items():
        chunks = []
        for a, b in zip(images, images[1:]):
            p = cv2.goodFeaturesToTrack(a, 90, .012, 9, mask=mask)
            if p is None:
                continue
            opts = dict(winSize=(17, 17), maxLevel=3)
            q, good, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **opts)
            if q is None:
                continue
            back, good_back, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **opts)
            if back is None:
                continue
            p, q = p[:, 0], q[:, 0]
            flow = np.linalg.norm(q-p, axis=1)
            ij = np.clip(q.round().astype(int), [0, 0], [w-1, h-1])
            keep = (good[:, 0].astype(bool) & good_back[:, 0].astype(bool) &
                    (np.linalg.norm(back[:, 0]-p, axis=1) < 1) & (flow > 2) & (flow < 90) &
                    (mask[ij[:, 1], ij[:, 0]] != 0))
            if np.any(keep):
                cross = np.cross(rays(p[keep], focal, True), rays(q[keep], focal, True))
                chunks.append(unit(cross))
        if chunks:
            result[label] = np.concatenate(chunks)
    return result


def fit(data, old, down):
    # Restrict the motion VP to the independently estimated horizon plane.
    # Sweep azimuth to avoid a least-squares fit dominated by moving traffic.
    side = unit(np.cross(down, old))
    rear = unit(np.cross(side, down))
    sweep = np.radians(np.linspace(-15, 15, 301))
    grid = np.cos(sweep[:, None])*rear + np.sin(sweep[:, None])*side
    scores = []
    for label in WINDOWS:
        groups = [t[label] for t in data if label in t and len(t[label]) >= 3]
        if len(groups) < 5:
            continue
        z = np.concatenate(groups) @ grid.T
        # Report each window independently so the larger side ROIs cannot
        # make a biased rear estimate look like agreement.
        loss = np.abs(z).clip(0, np.sin(np.radians(8)))
        per_scene = np.split(loss, np.cumsum([len(g) for g in groups])[:-1])
        costs = np.mean([np.median(a, axis=0) for a in per_scene], axis=0)
        i = int(np.argmin(costs))
        scores.append(dict(window=label, scenes=len(groups), tracks=len(z),
                           heading_delta_deg=float(np.degrees(sweep[i])),
                           baseline_residual_deg=float(np.degrees(np.arcsin(costs[150]))),
                           fitted_residual_deg=float(np.degrees(np.arcsin(costs[i])))))
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--output', type=Path, default=Path('previews/rear-window-fit.json'))
    args = ap.parse_args()
    if args.stride < 1:
        ap.error('stride must be positive')
    cv2.setNumThreads(1)
    cam = json.loads(Path('calibration.json').read_text())['d']
    basis = np.array(cam['basis'])
    masks = {k: cv2.fillConvexPoly(np.zeros((760, 1344), np.uint8), 2*np.array(v, np.int32), 255)
             for k, v in WINDOWS.items()}
    data = []
    for path in sorted(Path('.').glob('*--*/dcamera.hevc'))[::args.stride]:
        try:
            if not straight_motion(path.parent):
                continue
            obs = tracks(path, masks, cam['focal'])
        except (StopIteration, OSError, ValueError, cv2.error, capnp.KjException):
            continue
        if obs:
            data.append(obs)
        print(path.parent, {k: len(v) for k, v in obs.items()}, flush=True)
    scores = fit(data, -basis[:, 2], basis[:, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2)+'\n')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
