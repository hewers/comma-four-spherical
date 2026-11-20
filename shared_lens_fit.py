"""Fit identical wide/cabin fisheye intrinsics and independent rotations."""
import json
from pathlib import Path
import argparse

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration_survey import rays
from stitch import unit


def plane(points):
    return np.linalg.svd(points, full_matrices=False)[2][-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('previews/shared-lens-fit'))
    a = ap.parse_args()
    cfg = json.loads(Path('calibration-best.json').read_text())
    scenes = json.loads(Path('highway-scenes.json').read_text())
    barrier = json.loads(Path('barrier-reference.json').read_text())
    seeds = {k: np.asarray(cfg[k]['basis']) for k in 'ed'}
    # Focal, principal point, and radial coefficients are shared exactly.
    prior = np.array([433., 672., 380., 0., 0., *([0.]*6)])
    low = np.array([390., 630., 345., -.03, -.002, *([-.14]*6)])
    high = np.array([485., 714., 415., .10, .025, *([.14]*6)])
    scales = np.array([30., 24., 20., .05, .012, *([.07]*6)])

    def basis(x, k):
        i = 5 if k == 'e' else 8
        return seeds[k] @ Rotation.from_rotvec(x[i:i+3]).as_matrix()

    def world(x, k, points):
        return rays(2*np.asarray(points), x[0], True, x[1:3], x[3:5]) @ basis(x, k)

    def scene_errors(x, scene):
        horizon, heading, straight = [], [], []
        directions = []
        for k in 'ed':
            if 'horizon' in scene[k]:
                horizon.extend(world(x, k, scene[k]['horizon'])[:, 1])
            lines = [world(x, k, p) for p in scene[k]['longitudinal']]
            normals = [plane(line) for line in lines]
            for line, normal in zip(lines, normals):
                straight.extend(line @ normal)
            if k == 'e' and len(normals) >= 2:
                v = unit(np.cross(*normals[:2]))
                v *= 1 if v[2] >= 0 else -1
                directions.append(v)
                heading.extend([.35*v[0], .35*v[1]])
            elif directions:
                heading.extend(normal @ directions[0] for normal in normals)
        return np.asarray(horizon), np.asarray(heading), np.asarray(straight)

    def objective(x, group):
        chunks = []
        for scene in group:
            for weight, err in zip((1., 1., .6), scene_errors(x, scene)):
                if len(err):
                    chunks.append(weight*err/np.sqrt(len(err)))
        return np.concatenate(chunks + [.004*(x-prior)/scales])

    def fit(group):
        x = prior.copy()
        # Recenter the trust bounds after each small fit. Maximum rotation
        # increment is 0.25 degrees per axis per outer iteration.
        step = np.array([4., 4., 4., .008, .002, *([np.radians(.25)]*6)])
        for _ in range(36):
            nxt = least_squares(objective, x, args=(group,), loss='soft_l1', f_scale=.012,
                                x_scale=scales, bounds=(np.maximum(low, x-step), np.minimum(high, x+step))).x
            if np.linalg.norm((nxt-x)/scales) < 1e-4:
                return nxt
            x = nxt
        return x

    def score(x, scene):
        return [float(np.sqrt(np.mean(np.degrees(np.arcsin(np.clip(e, -1, 1)))**2))) if len(e) else None
                for e in scene_errors(x, scene)]

    # Hold out both the annotated scene and the barrier if from the same route.
    report = []
    for name, scene in scenes.items():
        route = name.rsplit('--', 1)[0]
        training = [s for n, s in scenes.items() if n.rsplit('--', 1)[0] != route]
        if barrier['segment'].rsplit('--', 1)[0] != route:
            training.append(barrier)
        x = fit(training)
        report.append(dict(segment=name, shared_seed_deg=score(prior, scene), candidate_deg=score(x, scene)))
    x = fit(list(scenes.values())+[barrier])
    old_e = seeds['e']
    for k in 'ed':
        cfg[k].update(focal=float(x[0]), center=x[1:3].tolist(), radial=x[3:5].tolist(), basis=basis(x,k).tolist())
    cfg['f']['basis'] = (np.asarray(cfg['f']['basis']) @ old_e.T @ basis(x,'e')).tolist()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'candidate.json').write_text(json.dumps(cfg, indent=2)+'\n')
    result = dict(metric_order=['horizon', 'heading', 'straightness'], shared_lens=x[:5].tolist(),
                  rotation_deltas_deg=np.degrees(x[5:]).tolist(), heldout=report,
                  barrier_shared_seed_deg=score(prior, barrier), barrier_fitted_deg=score(x, barrier))
    (a.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
