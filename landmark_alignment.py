"""Test direct wide/cabin alignment using landmarks seen before and after passing."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration_survey import first_image, rays
from rear_window_survey import WINDOWS, straight_motion
from stitch import imageio_ffmpeg, unit


def cabin_frames(path):
    g = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '1'])
    try:
        w, h = next(g)['size']
        if (w, h) != (1344, 760):
            return
        for i, rgb in enumerate(g):
            if i in (20, 40, 80, 120):
                yield i, cv2.cvtColor(np.frombuffer(rgb, np.uint8).reshape(h, w, 3), cv2.COLOR_RGB2GRAY)
            if i >= 120:
                break
    finally:
        g.close()


def observations(seg, sift, matcher, masks, cfg):
    ext = first_image(seg, 'e')
    if ext.shape != masks['e'].shape:
        return []
    kp, dp = sift.detectAndCompute(ext, masks['e'])
    if dp is None:
        return []
    results = []
    for frame, cabin in cabin_frames(seg/'dcamera.hevc'):
        kq, dq = sift.detectAndCompute(cabin, masks['d'])
        if dq is None or len(dq) < 2:
            continue
        pairs = [(m.queryIdx, m.trainIdx) for pair in matcher.knnMatch(dp, dq, k=2)
                 if len(pair) == 2 for m, n in [pair] if m.distance < .68*n.distance]
        if len(pairs) < 8:
            continue
        a = np.float32([kp[i].pt for i, _ in pairs])
        b = np.float32([kq[j].pt for _, j in pairs])
        ae = rays(a, cfg['e']['focal'], True) @ np.asarray(cfg['e']['basis'])
        bd = rays(b, cfg['d']['focal'], True) @ np.asarray(cfg['d']['basis'])
        # For straight motion the lateral/down plane through a static
        # feature is shared by both rays even as their longitudinal angles
        # change. Mask out rays near the travel direction (degenerate).
        valid = (np.linalg.norm(ae[:, :2], axis=1) > .18) & (np.linalg.norm(bd[:, :2], axis=1) > .18)
        ar, br = ae[valid], bd[valid]
        n = unit(np.cross(ar, [0, 0, 1]))
        radial_agreement = (ar[:, :2]*br[:, :2]).sum(axis=1) > 0
        keep = radial_agreement & (np.abs(np.sum(n*br, axis=1)) < np.sin(np.radians(12)))
        if np.count_nonzero(keep) < 6:
            continue
        results.append((frame, a[valid][keep], b[valid][keep], n[keep], br[keep]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('previews/landmark-fit'))
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path('calibration.json').read_text())
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=2400, contrastThreshold=.015)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    e = np.zeros((760, 1344), np.uint8)
    e[205:516, 45:-45] = 255
    d = np.zeros_like(e)
    for window in WINDOWS.values():
        cv2.fillConvexPoly(d, 2*np.asarray(window, np.int32), 255)
    data = []
    audit = []
    paths = sorted(p.parent for p in Path('.').glob('*--*/dcamera.hevc') if (p.parent/'ecamera.hevc').exists())
    routes = sorted({p.name.rsplit('--', 1)[0] for p in paths})
    heldout = set(routes[::4])
    for seg in paths[::args.stride]:
        try:
            if not straight_motion(seg):
                continue
            found = observations(seg, sift, matcher, {'e': e, 'd': d}, cfg)
        except (OSError, RuntimeError, StopIteration, ValueError, cv2.error) as exc:
            print(seg, type(exc).__name__, flush=True)
            continue
        print(seg, sum(len(a) for _, a, *_ in found), flush=True)
        if found:
            hold = seg.name.rsplit('--', 1)[0] in heldout
            for frame, a, b, n, r in found:
                data.append((hold, n, r))
                audit.append(dict(segment=seg.name, cabin_frame=frame, wide_xy=a.tolist(), cabin_xy=b.tolist()))
    (args.output/'matches.json').write_text(json.dumps(audit)+'\n')
    train = [(n, r) for hold, n, r in data if not hold]
    test = [(n, r) for hold, n, r in data if hold]
    if len(train) < 10 or len(test) < 5:
        print(f'Insufficient independent evidence: {len(train)} training and {len(test)} held-out image pairs')
        return
    def residual(x, groups):
        r = Rotation.from_rotvec(x).as_matrix()
        return np.concatenate([np.sum(n*(v @ r.T), axis=1) for n, v in groups])
    x = least_squares(lambda x: residual(x, train), np.zeros(3), loss='soft_l1',
                      f_scale=.025, bounds=(-.17, .17)).x
    for title, groups in [('train', train), ('held out', test)]:
        errors = [float(np.degrees(np.arcsin(np.median(np.abs(residual(v, groups)))))) for v in (np.zeros(3), x)]
        print(title, len(groups), 'pairs; median residual before/after:', errors)
    print('candidate correction (degrees, world right/down/forward axes):', np.degrees(x))


if __name__ == '__main__':
    main()
