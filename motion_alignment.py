"""Estimate wide-to-cabin rotation from synchronized camera motion, not shared scenery."""
import argparse
import json
from pathlib import Path

import capnp
import cv2
import zstandard
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration_survey import rays
from rear_window_survey import WINDOWS
from stitch import imageio_ffmpeg, timing


def images(path, indices):
    reader = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '1'])
    try:
        size = next(reader)['size']
        if size != (1344, 760):
            raise ValueError('unsupported frame size')
        result = []
        for i, frame in enumerate(reader):
            if i in indices:
                result.append(cv2.cvtColor(np.frombuffer(frame, np.uint8).reshape(760, 1344, 3), cv2.COLOR_RGB2GRAY))
            if i >= indices[-1]:
                break
        if len(result) != len(indices):
            raise ValueError('not enough frames')
        return result
    finally:
        reader.close()


def estimate(im, mask, focal):
    detector = cv2.SIFT_create(nfeatures=1600, contrastThreshold=.018)
    k0, f0 = detector.detectAndCompute(im[0], mask)
    k1, f1 = detector.detectAndCompute(im[1], mask)
    if f0 is None or f1 is None or len(f1) < 2:
        return None
    pairs = [m for pair in cv2.BFMatcher(cv2.NORM_L2).knnMatch(f0, f1, k=2)
             if len(pair) == 2 for m, n in [pair] if m.distance < .73*n.distance]
    if len(pairs) < 18:
        return None
    a = np.array([k0[m.queryIdx].pt for m in pairs])
    b = np.array([k1[m.trainIdx].pt for m in pairs])
    moving = np.linalg.norm(a-b, axis=1) > 2
    a, b = a[moving], b[moving]
    if len(a) < 18:
        return None
    u, v = [rays(p, focal, True) for p in (a, b)]
    keep = (u[:, 2] > .28) & (v[:, 2] > .28)
    u, v = [p[keep, :2] / p[keep, 2:3] for p in (u, v)]
    if len(u) < 18:
        return None
    e, inliers = cv2.findEssentialMat(u, v, np.eye(3), method=cv2.RANSAC, prob=.999, threshold=.003)
    if e is None or e.shape != (3, 3):
        return None
    count, r, t, _ = cv2.recoverPose(e, u, v, np.eye(3), mask=inliers)
    if count < 15 or count < .25*len(u):
        return None
    angle = Rotation.from_matrix(r).as_rotvec()
    if np.linalg.norm(angle) > .4:
        return None
    return angle, t[:, 0], int(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('previews/motion-fit.json'))
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    cfg = json.loads(Path('calibration.json').read_text())
    cv2.setNumThreads(1)
    wide = np.zeros((760, 1344), np.uint8)
    wide[130:515, 80:-80] = 255
    cabin = np.zeros_like(wide)
    for region in WINDOWS.values():
        cv2.fillConvexPoly(cabin, 2*np.asarray(region, np.int32), 255)
    paths = sorted(p.parent for p in Path('.').glob('*--*/dcamera.hevc') if (p.parent/'ecamera.hevc').exists())
    routes = sorted({p.name.rsplit('--', 1)[0] for p in paths})
    hold = set(routes[::4])
    records = []
    for seg in paths[::args.stride]:
        try:
            # Use capture time, not ordinal equality, to pair motion intervals.
            ts, _ = timing(seg)
            start = max(ts[k][0] for k in 'ed')
            ticks = (start, start+850_000_000)
            motion = []
            for k, mask in [('e', wide), ('d', cabin)]:
                index = [int(np.argmin(np.abs(ts[k]-t))) for t in ticks]
                if any(abs(int(ts[k][i])-t) > 55_000_000 for i, t in zip(index, ticks)):
                    break
                est = estimate(images(seg/f'{k}camera.hevc', index), mask, cfg[k]['focal'])
                if est is None:
                    break
                motion.append(est)
            if len(motion) != 2:
                continue
        except (OSError, ValueError, StopIteration, cv2.error, capnp.KjException, zstandard.ZstdError) as exc:
            print(seg, type(exc).__name__, flush=True)
            continue
        (we, te, ne), (wd, td, nd) = motion
        # Estimate the same rigid-body turn; weak rotations contribute little.
        if abs(np.linalg.norm(we)-np.linalg.norm(wd)) > .055:
            continue
        records.append(dict(segment=seg.name, heldout=seg.name.rsplit('--', 1)[0] in hold,
                            we=we.tolist(), wd=wd.tolist(), te=te.tolist(), td=td.tolist(), counts=[ne, nd]))
        print(seg, ne, nd, flush=True)
    seed = np.asarray(cfg['d']['basis']) @ np.asarray(cfg['e']['basis']).T
    train, test = ([r for r in records if r['heldout'] == h] for h in (False, True))
    def residual(x, rows):
        r = Rotation.from_rotvec(x).as_matrix() @ seed
        t = np.array([r @ row['te'] - row['td'] for row in rows])
        w = np.array([r @ row['we'] - row['wd'] for row in rows])
        return t, w
    report = dict(records=records, training=len(train), heldout=len(test))
    if len(train) >= 10 and len(test) >= 5:
        def cost(x):
            t, w = residual(x, train)
            return np.r_[t.ravel(), 4*w.ravel()]
        x = least_squares(cost, np.zeros(3), bounds=(-.17,.17), loss='soft_l1', f_scale=.07).x
        report['delta_deg'] = np.degrees(x).tolist()
        report['heldout_translation_deg'] = [float(np.degrees(np.median(2*np.arcsin(np.clip(np.linalg.norm(residual(v,test)[0], axis=1)/2, 0, 1))))) for v in (np.zeros(3), x)]
        report['heldout_rotation_vector_error_deg'] = [float(np.degrees(np.median(np.linalg.norm(residual(v,test)[1], axis=1)))) for v in (np.zeros(3), x)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print({k: v for k, v in report.items() if k != 'records'}, flush=True)


if __name__ == '__main__':
    main()
