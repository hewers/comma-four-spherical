"""Read-only forward-camera calibration survey; never edits live calibration."""
import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from stitch import frame_basis, imageio_ffmpeg, unit


def first_image(seg, key):
    with_frames = imageio_ffmpeg.read_frames(str(seg/f'{key}camera.hevc'), input_params=['-threads', '1'])
    try:
        w, h = next(with_frames)['size']
        rgb = np.frombuffer(next(with_frames), np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    finally:
        with_frames.close()


def match(seg, sift, matcher):
    # Enlarge the wide camera's middle to about the narrow camera's scale.
    wide, narrow = (first_image(seg, key) for key in 'ef')
    h, w = wide.shape
    if wide.shape != narrow.shape or (w, h) != (1344, 760):
        raise ValueError('unexpected image geometry')
    x, y = w//2-300, h//2-180
    crop = cv2.resize(wide[y:y+360, x:x+600], None, fx=2, fy=2)
    ka, da = sift.detectAndCompute(crop, None)
    kb, db = sift.detectAndCompute(narrow, None)
    if da is None or db is None or len(db) < 2:
        return None
    pairs = [m for pair in matcher.knnMatch(da, db, k=2) if len(pair) == 2
             for m, n in [pair] if m.distance < .72*n.distance]
    if len(pairs) < 20:
        return None
    a = np.float32([ka[m.queryIdx].pt for m in pairs])/2 + [x, y]
    b = np.float32([kb[m.trainIdx].pt for m in pairs])
    # A homography rejects descriptor mismatches and selects a dominant
    # depth plane; the ray fit below then uses a robust angular loss.
    _, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 4.)
    if mask is None or mask.sum() < 20:
        return None
    keep = mask.ravel().astype(bool)
    return a[keep], b[keep]


def rays(points, focal, equidistant, center=(672, 380), radial=(0., 0.)):
    p = (points-center)/focal
    if equidistant:
        r = np.linalg.norm(p, axis=1)
        t = r * (1 + radial[0]*r*r + radial[1]*r**4)
        s = np.divide(np.sin(t), r, out=np.ones_like(r), where=r > 0)
        return np.column_stack((p*s[:, None], np.cos(t)))
    return unit(np.column_stack((p, np.ones(len(p)))))


def fit(matches, cfg):
    e, f = cfg['e'], cfg['f']
    seed = np.array(e['basis']) @ np.array(f['basis']).T
    train = [data for _, data, heldout in matches if not heldout]
    if len(train) < 3:
        raise ValueError('too few training segments with reliable matches')
    a, b = map(np.concatenate, zip(*train))

    def error(x, wide, narrow):
        r = Rotation.from_rotvec(x[:3]).as_matrix() @ seed
        return rays(wide, x[3], True) - rays(narrow, f['focal'], False) @ r.T

    initial = np.r_[0., 0., 0., e['focal']]
    result = least_squares(lambda x: error(x, a, b).ravel(), initial, loss='soft_l1',
                           f_scale=.002, bounds=([-.2]*3+[350], [.2]*3+[500]))
    r = Rotation.from_rotvec(result.x[:3]).as_matrix() @ seed
    # Preserve the wide camera's surveyed horizon; transfer its orientation
    # to the narrow camera using real shared scenery, rather than separate VPs.
    e['focal'] = float(result.x[3])
    e['basis'] = pooled_basis(e, 'e', result.x[3]).tolist()
    f['basis'] = (r.T @ np.asarray(e['basis'])).tolist()
    scores = []
    for name, (wide, narrow), heldout in matches:
        before = np.linalg.norm(error(initial, wide, narrow), axis=1)
        after = np.linalg.norm(error(result.x, wide, narrow), axis=1)
        scores.append(dict(segment=name, heldout=heldout, matches=len(wide),
                           before_deg=float(np.degrees(np.median(before))),
                           after_deg=float(np.degrees(np.median(after)))))
    return scores


def scene_basis(cam, annotation, focal):
    points = {field: (2*np.asarray(values)).tolist() if field == 'horizon'
              else [(2*np.asarray(line)).tolist() for line in values]
              for field, values in annotation.items()}
    return frame_basis({**cam, **points, 'focal': focal})


def rotation_median(rotations, seed):
    delta = Rotation.from_matrix(np.asarray(rotations) @ seed.T).as_rotvec()
    return Rotation.from_rotvec(np.median(delta, axis=0)).as_matrix() @ seed


def pooled_basis(cam, key, focal):
    scenes = json.loads(Path('calibration-scenes.json').read_text())
    return rotation_median([scene_basis(cam, s[key], focal) for s in scenes.values()], np.asarray(cam['basis']))


def cabin_fit(cfg, baseline):
    scenes = json.loads(Path('calibration-scenes.json').read_text())
    seed = np.asarray(baseline['d']['basis']) @ np.asarray(baseline['e']['basis']).T
    paired = [(scene_basis(cfg['e'], s['e'], cfg['e']['focal']),
               scene_basis(cfg['d'], s['d'], cfg['d']['focal'])) for s in scenes.values()]
    relative = [d @ e.T for e, d in paired]
    candidate = rotation_median(relative, seed) @ np.asarray(cfg['e']['basis'])
    # Leave each scene out in turn. Compare the disagreement of its
    # observed horizon normals after mapping them to common coordinates.
    def discrepancy(e, d, be, bd):
        a, b = be.T @ e[:, 1], bd.T @ d[:, 1]
        return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
    scores = []
    for i, (name, scene) in enumerate(scenes.items()):
        e, d = paired[i]
        be = np.asarray(cfg['e']['basis'])
        bd = rotation_median(relative[:i]+relative[i+1:], seed) @ be
        old_e = scene_basis(baseline['e'], scene['e'], baseline['e']['focal'])
        before = discrepancy(old_e, d, np.asarray(baseline['e']['basis']), np.asarray(baseline['d']['basis']))
        scores.append(dict(segment=name, before_deg=before, after_deg=discrepancy(e, d, be, bd)))
    return candidate, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('previews/calibration-survey'))
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    segments = sorted(p for p in Path('.').iterdir() if p.is_dir() and re.fullmatch(r'[\da-f]+--[\da-f]+--\d+', p.name)
                      and all((p/f'{k}camera.hevc').is_file() for k in 'ef'))
    routes = sorted({p.name.rsplit('--', 1)[0] for p in segments})
    heldout = set(routes[::5])
    matches = []
    for seg in segments[::args.stride]:
        try:
            data = match(seg, sift, matcher)
            if data is not None:
                # Cap the influence of any one frame.
                a, b = data
                indices = np.linspace(0, len(a)-1, min(120, len(a)), dtype=int)
                matches.append((seg.name, (a[indices], b[indices]), seg.name.rsplit('--', 1)[0] in heldout))
            print(seg, len(data[0]) if data is not None else 0, flush=True)
        except (OSError, ValueError, StopIteration, cv2.error) as exc:
            print(seg, type(exc).__name__, str(exc).split('\n')[0], flush=True)
    cfg = json.loads(Path('calibration.json').read_text())
    baseline = json.loads(json.dumps(cfg))
    scores = fit(matches, cfg)
    cabin, cabin_scores = cabin_fit(cfg, baseline)
    cfg['d']['basis'] = cabin.tolist()
    (args.output/'cabin-residuals.json').write_text(json.dumps(cabin_scores, indent=2)+'\n')
    (args.output/'candidate.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (args.output/'residuals.json').write_text(json.dumps(scores, indent=2)+'\n')
    heldout_scores = [s for s in scores if s['heldout']]
    def improvement(rows):
        return float(1 - np.median([s['after_deg'] for s in rows]) /
                     max(1e-9, np.median([s['before_deg'] for s in rows]))) if rows else None
    forward_gain = improvement(heldout_scores)
    cabin_gain = improvement(cabin_scores)
    passes = len(heldout_scores) >= 10 and forward_gain >= .25 and cabin_gain >= .25
    summary = dict(heldout_forward_segments=len(heldout_scores),
                   forward_improvement=forward_gain, cabin_improvement=cabin_gain,
                   passes_numeric_gate=passes)
    (args.output/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(summary, flush=True)
    for held in (False, True):
        group = [s for s in scores if s['heldout'] == held]
        if group:
            print('held out' if held else 'training', len(group), 'segments; median degrees before/after:',
                  *(np.median([s[k] for s in group]) for k in ('before_deg', 'after_deg')), flush=True)


if __name__ == '__main__':
    main()
