"""Fit fixed camera rotations from reviewed steady-highway geometry."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibration_survey import rays, rotation_median, scene_basis


def angular(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def errors(cfg, scene, bases):
    views = {}
    for k in 'ed':
        cam, annotation, basis = cfg[k], scene[k], bases[k]
        h = rays(2*np.asarray(annotation['horizon']), cam['focal'], True) @ basis
        view = scene_basis(cam, annotation, cam['focal'])
        views[k] = (np.degrees(np.arcsin(np.clip(h[:, 1], -1, 1))), basis.T @ view[:, 2])
    return dict(wide_horizon_rms_deg=float(np.sqrt(np.mean(views['e'][0]**2))),
                cabin_horizon_rms_deg=float(np.sqrt(np.mean(views['d'][0]**2))),
                heading_error_deg=angular(views['e'][1], views['d'][1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('annotations', type=Path)
    ap.add_argument('--output', type=Path, default=Path('previews/highway-fit'))
    a = ap.parse_args()
    scenes = json.loads(a.annotations.read_text())
    if len(scenes) < 3:
        ap.error('provide at least three reviewed scenes from distinct drives')
    routes = [s.rsplit('--', 1)[0] for s in scenes]
    if len(set(routes)) != len(routes):
        ap.error('only one scene per drive is supported for independent validation')
    cfg = json.loads(Path('calibration.json').read_text())
    old = {k: np.array(cfg[k]['basis']) for k in 'edf'}
    pairs = [(scene_basis(cfg['e'], s['e'], cfg['e']['focal']), scene_basis(cfg['d'], s['d'], cfg['d']['focal']))
             for s in scenes.values()]

    def fit(pairs):
        e = rotation_median([e for e, _ in pairs], old['e'])
        d_from_e = rotation_median([d @ e.T for e, d in pairs], old['d'] @ old['e'].T)
        return {'e': e, 'd': d_from_e @ e, 'f': old['f'] @ old['e'].T @ e}

    report = []
    for i, (name, scene) in enumerate(scenes.items()):
        report.append(dict(segment=name, before=errors(cfg, scene, old),
                           after=errors(cfg, scene, fit(pairs[:i]+pairs[i+1:]))))
    for k, basis in fit(pairs).items():
        cfg[k]['basis'] = basis.tolist()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'candidate.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (a.output/'heldout.json').write_text(json.dumps(report, indent=2)+'\n')
    for key in report[0]['before']:
        print(key, *(round(float(np.median([r[s][key] for r in report])), 3) for s in ('before', 'after')))


if __name__ == '__main__':
    main()
