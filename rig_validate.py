"""Independent checks of a rig calibration that the track fit never saw."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration_survey import match, rays


def world(cfg, k, px):
    c = cfg[k]
    if c['model'] == 'equidistant':
        r = rays(px, c['focal'], True, c.get('center', (672, 380)), c.get('radial', (0., 0.)))
    else:
        p = (px - np.asarray(c.get('center', (672, 380))))/c['focal']
        r = np.column_stack((p, np.ones(len(p))))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
    return r @ np.asarray(c['basis'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('calibrations', nargs='+', type=Path)
    ap.add_argument('--segments', type=int, default=60)
    a = ap.parse_args()
    cv2.setNumThreads(2)
    # 1. Wide/narrow overlap: SIFT matches on simultaneous frames from held-out routes.
    files = sorted(Path('rig/tracks').glob('*.npz'))
    routes = sorted({f.stem.rsplit('--', 1)[0] for f in files})
    held = [f.stem for f in files if f.stem.rsplit('--', 1)[0] in set(routes[1::4])]
    sift, matcher = cv2.SIFT_create(nfeatures=1500), cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for seg in held[::max(1, len(held)//a.segments)]:
        try:
            found = match(Path(seg), sift, matcher)
        except (OSError, ValueError, StopIteration, cv2.error):
            continue
        if found is not None:
            pairs.append(found)
    wide = np.concatenate([p[0] for p in pairs]).astype(float)
    narrow = np.concatenate([p[1] for p in pairs]).astype(float)
    radius = np.linalg.norm(narrow - [672, 380], axis=1)
    print(f'wide/narrow overlap: {len(pairs)} held-out segments, {len(wide)} matches')
    for path in a.calibrations:
        cfg = json.loads(path.read_text())
        err = np.degrees(np.arccos(np.clip(np.sum(world(cfg, 'e', wide)*world(cfg, 'f', narrow), axis=1), -1, 1)))
        print(f'  {str(path):45s} median {np.median(err):.3f} deg, 90th pct {np.quantile(err, .9):.3f}; '
              f'centre {np.median(err[radius < 250]):.3f}, edge of narrow {np.median(err[radius > 500]):.3f}')

    # 2. Gravity: accelerometer mean versus the axis the car actually yaws about in turns.
    turns, down = [], []
    for f in files:
        z = np.load(f)
        if np.isfinite(z['gravity']).all():
            down.append(z['gravity']/np.linalg.norm(z['gravity']))
        p = z['e_pairs']
        w = p[p[:, 0] == 1, 7:10]
        turns.extend(w[np.linalg.norm(w, axis=1) > .1])
    turns = np.array(turns)
    axis = np.linalg.svd(turns, full_matrices=False)[2][0]
    axis *= np.sign(axis[2])
    g = np.median(down, axis=0); g /= np.linalg.norm(g)
    ang = lambda v: (np.degrees(np.arctan2(v[1], v[2])), np.degrees(np.arctan2(-v[0], v[2])))
    print(f'gravity in device frame: accelerometer roll/pitch {ang(g)[0]:.2f}/{ang(g)[1]:.2f} deg; '
          f'yaw axis over {len(turns)} turns roll/pitch {ang(axis)[0]:.2f}/{ang(axis)[1]:.2f} deg; '
          f'difference {np.degrees(np.arccos(np.clip(axis @ g, -1, 1))):.2f} deg')
    left, right = turns[turns[:, 2] < 0], turns[turns[:, 2] > 0]
    for name, part in (('left turns', left), ('right turns', right)):
        v = np.linalg.svd(part, full_matrices=False)[2][0]; v *= np.sign(v[2])
        print(f'  {name}: {len(part)}, roll/pitch {ang(v)[0]:.2f}/{ang(v)[1]:.2f}')


if __name__ == '__main__':
    main()
