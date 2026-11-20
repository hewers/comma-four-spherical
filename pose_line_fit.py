"""Fit camera-to-device rotations using logged pose, gravity, and road lines."""
import argparse
import json
from pathlib import Path

import capnp
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration_survey import rays
from stitch import log_events, timing, unit

# Openpilot device axes are forward/right/down; image axes are right/down/forward.
VIEW = np.array([[0,1,0],[0,0,1],[1,0,0]])


def xyz(m):
    return np.array([m.x, m.y, m.z])


def observations(seg, target):
    poses, fixes, imu = [], [], []
    for e in log_events(Path('rlogs', seg, 'rlog.zst')):
        try:
            kind = e.which()
        except capnp.KjException:
            continue
        if kind == 'deviceMotion':
            p = e.deviceMotion
            if abs(int(p.timestamp)-target) > 700_000_000 or not (p.inputsOK and p.sensorsOK and p.posenetOK):
                continue
            if not all(m.valid for m in (p.orientationNED, p.velocityDevice, p.angularVelocityDevice)):
                continue
            if np.linalg.norm(xyz(p.angularVelocityDevice)) > .04 or np.linalg.norm(xyz(p.velocityDevice)) < 17:
                continue
            down = Rotation.from_euler('xyz', xyz(p.orientationNED)).inv().apply([0,0,1])
            poses.append((down, unit(xyz(p.velocityDevice))))
        elif kind == 'gpsLocationExternal' and abs(int(e.logMonoTime)-target) < 1_500_000_000:
            g = e.gpsLocationExternal
            if g.hasFix and g.speed > 15:
                fixes.append((g.speed, g.bearingDeg))
        elif kind == 'accelerometer':
            p = e.accelerometer
            if abs(int(p.timestamp)-target) < 700_000_000 and p.which() == 'acceleration' and len(p.acceleration.v) == 3:
                # Match upstream locationd: device specific force = [-z,-y,-x].
                v = np.asarray(p.acceleration.v)
                if 9 < np.linalg.norm(v) < 10.7:
                    imu.append(unit(v[::-1]))
        if e.logMonoTime > target+2_000_000_000:
            break
    if len(poses) < 4:
        raise ValueError('insufficient valid low-rate device pose')
    down, forward = (unit(v) for v in np.median(poses, axis=0))
    raw = unit(np.median(imu, axis=0)) if imu else None
    gap = float(np.degrees(np.arccos(np.clip(raw @ down,-1,1)))) if raw is not None else None
    if gap is not None and gap > 5:
        raise ValueError(f'pose gravity inconsistent with low-dynamics IMU: {gap:.1f} deg')
    course_spread = None
    if fixes:
        courses = np.unwrap(np.radians([b for _, b in fixes]))
        course_spread = float(np.degrees(np.ptp(courses)))
        if course_spread > 3:
            raise ValueError('GNSS course changes too much')
    return down, forward, dict(pose_samples=len(poses), imu_gravity_gap_deg=gap,
                              gnss_fixes=len(fixes), gnss_course_spread_deg=course_spread)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('previews/pose-line-fit'))
    args = ap.parse_args()
    cfg = json.loads(Path('calibration-best.json').read_text())
    annotations = json.loads(Path('highway-scenes.json').read_text())
    barrier = json.loads(Path('barrier-reference.json').read_text())
    annotations[barrier['segment']] = barrier
    # Capture clock times of the two exported annotated mid-window frames.
    midpoints = {'000000da--fafd43926e--50': 185273010658465,
                 '000000ea--f23abe10b0--10': 10859921244605}
    rows = []
    for name, scene in annotations.items():
        ts, _ = timing(Path(name))
        target = int(ts['e'][scene['e'].get('frame', 0)])
        target = midpoints.get(name, target)
        try:
            d, f, meta = observations(name, target)
        except ValueError as exc:
            print(name, exc, flush=True)
            continue
        rows.append((name, scene, d, f, meta))
    if len({r[0].rsplit('--',1)[0] for r in rows}) < 3:
        raise SystemExit('Need at least three independent routes with valid pose and annotations')
    down = unit(np.median([r[2] for r in rows], axis=0))
    fwd = unit(np.median([r[3] for r in rows], axis=0))
    fwd = unit(fwd - down*(down @ fwd))
    level = np.column_stack((np.cross(down, fwd), down, fwd))
    seed = {k: np.asarray(cfg[k]['basis']) @ level.T for k in 'ed'}
    initial = np.array([433.,672.,380.,0.,0.,*([0.]*6)])
    scales = np.array([25.,15.,15.,.035,.008,*([.07]*6)])
    lower = [390,640,350,-.03,-.002,*([-.14]*6)]
    upper = [485,704,410,.08,.02,*([.14]*6)]

    def basis(x,k):
        i = 5 if k == 'e' else 8
        return Rotation.from_rotvec(x[i:i+3]).as_matrix() @ seed[k]

    def errs(x, row):
        _, scene, gravity, travel, _ = row
        result = []
        for k in 'ed':
            cam = basis(x,k)
            def unproject(p):
                return rays(2*np.asarray(p), x[0], True, x[1:3], x[3:5])
            skyline = unproject(scene[k]['horizon']) @ cam @ gravity if 'horizon' in scene[k] else np.empty(0)
            heading, straight = [], []
            for trace in scene[k]['longitudinal']:
                vectors = unproject(trace)
                n = np.linalg.svd(vectors, full_matrices=False)[2][-1]
                heading.append(n @ cam @ travel)
                straight.extend(vectors @ n)
            result.extend((skyline, np.array(heading), np.array(straight)))
        return result

    def fit(rows):
        def loss(x):
            values = [w*v/np.sqrt(len(v)) for row in rows for w,v in zip((1.,1.,.6)*2,errs(x,row)) if len(v)]
            return np.concatenate(values + [.007*(x-initial)/scales])
        return least_squares(loss, initial, bounds=(lower,upper), x_scale=scales, loss='soft_l1', f_scale=.01).x

    def score(x,row):
        return [float(np.degrees(np.sqrt(np.mean(np.arcsin(np.clip(v,-1,1))**2)))) if len(v) else None for v in errs(x,row)]

    held = []
    for row in rows:
        route = row[0].rsplit('--',1)[0]
        x = fit([r for r in rows if r[0].rsplit('--',1)[0] != route])
        held.append(dict(segment=row[0], evidence=row[4], before=score(initial,row), after=score(x,row)))
    x = fit(rows)
    e0 = np.asarray(cfg['e']['basis'])
    for k in 'ed':
        cfg[k].update(focal=float(x[0]), center=x[1:3].tolist(), radial=x[3:5].tolist(), basis=(basis(x,k)@level).tolist())
    cfg['f']['basis'] = (np.asarray(cfg['f']['basis']) @ e0.T @ np.asarray(cfg['e']['basis'])).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'candidate.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result = dict(metrics=['wide_horizon','wide_heading','wide_straightness','cabin_horizon','cabin_heading','cabin_straightness'],
                  parameters=x.tolist(), heldout=held)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
