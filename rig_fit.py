"""Joint rig calibration from feature tracks, gyro rotation and pose velocity.

Every track (u -> v) in a camera must satisfy the epipolar constraint for the
motion of the device between the two exposures: rotation D from the gyro and
translation direction t from deviceMotion, both in the device frame. With
C_k the rotation from device to camera k, the residual is the angular
distance of the second ray from the epipolar great circle of the first:

    a = C_k^T u,  b = D C_k^T v,  r = b . (t x a) / |t x a|

Unknowns: C_e, C_d, C_f; one fisheye model shared by wide and cabin
(focal, centre, two radial terms); narrow focal; and optionally per-camera
time offsets, sensor readout time and a constant correction of the logged
travel direction. Whole routes are held out for validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

W, H = 1344, 760
VIEW = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])     # camera (right, down, forward) from device (forward, right, down)
BACK = np.array([[0., -1, 0], [0, 0, 1], [-1, 0, 0]])   # the same looking backwards
NAMES = ['f', 'cx', 'cy', 'k1', 'k2', 'f_narrow',
         'e_rx', 'e_ry', 'e_rz', 'd_rx', 'd_ry', 'd_rz', 'f_rx', 'f_ry', 'f_rz',
         'td_e', 'td_d', 'td_f', 'readout', 'vel_pitch', 'vel_yaw']
X0 = np.array([425.25, 672, 380, 0, 0, 1141.5, *[0]*9, 0, 0, 0, 0, 0, 0], float)
SCALE = np.array([20, 10, 10, .03, .01, 30, *[.02]*9, .01, .01, .01, .01, .01, .01])
LOW = np.array([330, 600, 320, -.2, -.1, 1000, *[-.5]*9, -.05, -.05, -.05, -.08, -.1, -.1])
HIGH = np.array([560, 744, 440, .3, .1, 1300, *[.5]*9, .05, .05, .05, .08, .1, .1])
STAGES = {'rotation': range(6, 15), 'focal': [0, 3, 5, *range(6, 15)],
          'intrinsics': range(0, 15), 'timing': range(0, 21)}


def load(max_per_pair, split='quarter', seed=0):
    rng = np.random.default_rng(seed)
    files = sorted(Path('rig/tracks').glob('*.npz'))
    routes = sorted({f.stem.rsplit('--', 1)[0] for f in files})
    held = {'quarter': set(routes[1::4]), 'odd': set(routes[::2]), 'even': set(routes[1::2]), 'none': set()}[split]
    data = {k: dict(u=[], v=[], D=[], t=[], w0=[], w1=[], cat=[], held=[], pair=[], speed=[]) for k in 'edf'}
    gravity, forward, calib = [], [], []
    count = dict.fromkeys('edf', 0)
    for f in files:
        z = np.load(f)
        if np.isfinite(z['gravity']).all():
            gravity.append(z['gravity']/np.linalg.norm(z['gravity']))
        if np.isfinite(z['calib']).all():
            calib.append(z['calib'])
        for k in 'edf':
            pairs, tracks = z[f'{k}_pairs'], z[f'{k}_tracks']
            for i, p in enumerate(pairs):
                rows = tracks[tracks[:, 0] == i]
                if len(rows) > max_per_pair:
                    rows = rows[rng.choice(len(rows), max_per_pair, replace=False)]
                if not len(rows):
                    continue
                d = data[k]
                rot = Rotation.from_rotvec(p[4:7])
                vel = p[13:16]
                speed = np.linalg.norm(vel)
                # Velocity is sampled mid-interval; express it in the device frame of the first exposure.
                t = Rotation.from_rotvec(p[4:7]/2).apply(vel/speed)
                if k == 'e' and p[0] == 0 and speed > 15:
                    forward.append(vel/speed)
                n = len(rows)
                d['u'].append(rows[:, 1:3]); d['v'].append(rows[:, 3:5])
                d['pair'].append(np.full(n, count[k]))
                d['D'].append(rot.as_matrix()); d['t'].append(t); d['w0'].append(p[7:10]); d['w1'].append(p[10:13])
                d['cat'].append(p[0]); d['speed'].append(speed)
                d['held'].append(f.stem.rsplit('--', 1)[0] in held)
                count[k] += 1
    for d in data.values():
        for key in ('u', 'v', 'pair'):
            d[key] = np.concatenate(d[key])
        for key in ('D', 't', 'w0', 'w1', 'cat', 'held', 'speed'):
            d[key] = np.array(d[key])
    overlap = None
    if Path('rig/overlap.npz').exists():
        z = np.load('rig/overlap.npz')
        route = np.array([s.rsplit('--', 1)[0] for s in z['segment']])
        overlap = dict(wide=z['wide'].astype(float), narrow=z['narrow'].astype(float), held=np.isin(route, sorted(held)))
    return data, np.array(gravity), np.array(forward), np.array(calib), len(files), sorted(held), overlap


def unproject(px, x, cam):
    if cam == 'f':
        p = (px - [W/2, H/2])/x[5]
        r = np.column_stack((p, np.ones(len(p))))
        return r/np.linalg.norm(r, axis=1, keepdims=True)
    p = (px - x[1:3])/x[0]
    rho = np.linalg.norm(p, axis=1)
    theta = rho*(1 + x[3]*rho**2 + x[4]*rho**4)
    s = np.divide(np.sin(theta), rho, out=np.ones_like(rho), where=rho > 1e-9)
    return np.column_stack((p*s[:, None], np.cos(theta)))


def rotation(x, cam, init):
    i = {'e': 6, 'd': 9, 'f': 12}[cam]
    return Rotation.from_rotvec(x[i:i+3]).as_matrix() @ init[cam]


def residual(x, d, cam, init, sel=None):
    """Angular epipolar residual (radians) for the tracks of one camera."""
    u, v, pair = (d[k] if sel is None else d[k][sel] for k in ('u', 'v', 'pair'))
    C = rotation(x, cam, init)
    td = x[15 + 'edf'.index(cam)]
    a = unproject(u, x, cam) @ C
    b = unproject(v, x, cam) @ C
    # Capture-time correction: offset plus rolling-shutter row delay.
    ta = (td + x[18]*(u[:, 1]/H - .5))[:, None]
    tb = (td + x[18]*(v[:, 1]/H - .5))[:, None]
    a = a + np.cross(d['w0'][pair]*ta, a)
    b = b + np.cross(d['w1'][pair]*tb, b)
    b = np.einsum('nij,nj->ni', d['D'][pair], b)
    t = d['t'][pair]
    t = t + np.cross(np.array([0, x[19], x[20]]), t)
    n = np.cross(t, a)
    norm = np.linalg.norm(n, axis=1)
    r = np.einsum('ni,ni->n', n, b)/np.maximum(norm, 1e-9)
    r[norm < np.sin(np.radians(4))] = 0     # epipolar line undefined next to the epipole
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-per-pair', type=int, default=80)
    ap.add_argument('--output', type=Path, default=Path('rig'))
    ap.add_argument('--split', choices=['quarter', 'odd', 'even', 'none'], default='quarter',
                    help='quarter: hold out every 4th route; odd/even: fit on half the routes for a stability check; none: use everything')
    ap.add_argument('--overlap-weight', type=float, default=1., help='0 keeps the wide/narrow matches for validation only')
    ap.add_argument('--scale', type=float, default=.0012, help='robust loss scale in radians')
    a = ap.parse_args()
    data, gravity, forward, calib, nfiles, held_routes, overlap = load(a.max_per_pair, a.split)
    train = {k: np.flatnonzero(~d['held'][d['pair']]) for k, d in data.items()}
    test = {k: np.flatnonzero(d['held'][d['pair']]) for k, d in data.items()}
    print(f'{nfiles} segments;', {k: (len(d['cat']), len(d['u'])) for k, d in data.items()}, 'pairs/tracks;',
          f'held-out routes: {len(held_routes)}', flush=True)
    init = {'e': VIEW, 'f': VIEW, 'd': BACK}
    rng = np.random.default_rng(1)

    def overlap_error(x, held):
        # The same distant scenery seen by wide and narrow at the same instant, compared in device axes.
        sel = overlap['held'] == held
        return (unproject(overlap['wide'][sel], x, 'e') @ rotation(x, 'e', init)
                - unproject(overlap['narrow'][sel], x, 'f') @ rotation(x, 'f', init))

    def cost(x):
        out = [residual(x, data[k], k, init, train[k]) for k in 'edf']
        if overlap is not None and a.overlap_weight:
            out.append(a.overlap_weight*overlap_error(x, False).ravel())
        return np.concatenate(out)

    # Coarse search for the cabin camera: nothing is assumed beyond "roughly backwards".
    sub = {'d': rng.choice(train['d'], min(40000, len(train['d'])), replace=False)}
    best = (np.inf, None)
    for pitch in range(-40, 21, 5):
        for yaw in range(-25, 26, 5):
            for roll in range(-15, 16, 5):
                trial = Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix() @ BACK
                init['d'] = trial
                c = np.median(np.abs(residual(X0, data['d'], 'd', init, sub['d'])))
                if c < best[0]:
                    best = (c, (pitch, yaw, roll), trial)
    init['d'] = best[2]
    print('cabin coarse search: pitch/yaw/roll about camera axes', best[1], 'median residual deg', round(float(np.degrees(best[0])), 3), flush=True)

    def summary(x):
        rows = {}
        for k in 'edf':
            for label, idx in (('train', train[k]), ('held', test[k])):
                r = np.abs(residual(x, data[k], k, init, idx))
                cat = data[k]['cat'][data[k]['pair'][idx]]
                rows[f'{k}-{label}'] = [round(float(np.degrees(np.median(r[cat == c]))), 4) if np.any(cat == c) else None
                                        for c in (0, 1, 2)] + [int(len(r))]
        if overlap is not None:
            for label, flag in (('train', False), ('held', True)):
                err = np.linalg.norm(overlap_error(x, flag), axis=1)
                rows[f'overlap-{label}'] = [round(float(np.degrees(np.median(err))), 4) if len(err) else None, int(len(err))]
        return rows

    report = dict(columns='median |residual| deg for straight, yaw-turn, pitch/roll pairs; track count. overlap: median wide/narrow angular error deg',
                  stages={'nominal': summary(X0)})
    x = X0.copy()
    for stage, free in STAGES.items():
        free = np.array(list(free))
        def fun(p):
            y = x.copy(); y[free] = p
            return cost(y)
        fit = least_squares(fun, x[free], bounds=(LOW[free], HIGH[free]), x_scale=SCALE[free],
                            loss='cauchy', f_scale=a.scale, max_nfev=150)
        x[free] = fit.x
        report['stages'][stage] = summary(x)
        print(stage, 'status', fit.status, 'nfev', fit.nfev, json.dumps(report['stages'][stage]), flush=True)
        print('   ', {n: round(float(v), 5) for n, v in zip(NAMES, x)}, flush=True)

    # Parameter uncertainty from the Gauss-Newton approximation with a robust residual scale.
    free = np.arange(len(x))
    r0 = cost(x)
    J = np.empty((len(r0), len(x)))
    for i in free:
        step = SCALE[i]*1e-3
        y = x.copy(); y[i] += step
        J[:, i] = (cost(y) - r0)/step
    sigma = 1.4826*np.median(np.abs(r0))
    weight = 1/(1 + (r0/a.scale)**2)
    cov = sigma**2*np.linalg.pinv((J*weight[:, None]).T @ J)
    std = np.sqrt(np.diag(cov))
    at_bound = [n for n, v, lo, hi in zip(NAMES, x, LOW, HIGH) if min(v-lo, hi-v) < 1e-6*(hi-lo)]

    down = np.median(gravity, axis=0); down /= np.linalg.norm(down)
    fwd = np.median(forward, axis=0); fwd /= np.linalg.norm(fwd)
    fwd = fwd + np.cross(np.array([0, x[19], x[20]]), fwd)
    fwd -= down*(fwd @ down); fwd /= np.linalg.norm(fwd)
    level = np.column_stack((np.cross(down, fwd), down, fwd))      # world (right, down, forward) in device axes
    C = {k: rotation(x, k, init) for k in 'edf'}
    fish = dict(model='equidistant', focal=float(x[0]), center=[float(x[1]), float(x[2])], radial=[float(x[3]), float(x[4])],
                size=[W, H], max_angle=92)   # image circle fades beyond ~92 degrees
    cfg = {'e': {**fish, 'basis': (C['e'] @ level).tolist()}, 'd': {**fish, 'basis': (C['d'] @ level).tolist()},
           'f': dict(model='pinhole', focal=float(x[5]), center=[W/2, H/2], size=[W, H], basis=(C['f'] @ level).tolist())}
    for k in 'edf':
        cfg[k]['camera_from_device'] = C[k].tolist()
    cfg['device_from_vehicle'] = level.tolist()
    cfg['timing'] = dict(camera_time_offset_s={k: float(x[15+i]) for i, k in enumerate('edf')}, readout_s=float(x[18]))
    cfg['frames'] = dict(
        cameras='e = wide road (ecamera), d = cabin (dcamera), f = narrow road (fcamera); all 1344x760',
        image='pixel (u, v): u right, v down, integer coordinates at pixel centres',
        camera='x right, y down, z along the optical axis',
        device='openpilot device frame: x forward, y right, z down; IMU sample v maps to (-v[2], -v[1], -v[0])',
        vehicle='x right, y down (mean gravity), z forward (mean direction of travel, horizontal)',
        equidistant='p = ((u-cx)/f, (v-cy)/f); rho = |p|; theta = rho*(1 + k1*rho^2 + k2*rho^4); '
                    'ray = (p/rho*sin(theta), cos(theta)); valid to max_angle degrees',
        pinhole='ray = normalize((u-cx)/f, (v-cy)/f, 1)',
        camera_from_device='ray_camera = camera_from_device @ v_device',
        device_from_vehicle='v_device = device_from_vehicle @ v_vehicle',
        basis='camera_from_vehicle = camera_from_device @ device_from_vehicle; ray_camera = basis @ v_vehicle',
        equirectangular='lon = atan2(x, z), lat = asin(y) of a unit vehicle vector; column = (lon/2pi + 0.5)*W, row = (lat/pi + 0.5)*H',
        timing='gyro time of image row v = (SOF+EOF)/2 + camera_time_offset_s + readout_s*(v/760 - 0.5)',
        translation='not estimated; the three lenses are within a few centimetres of each other')
    euler = lambda m: np.degrees(Rotation.from_matrix(m).as_euler('xyz')).round(3).tolist()
    logged = np.median(calib, axis=0)
    report.update(
        parameters={n: float(v) for n, v in zip(NAMES, x)},
        std={n: float(v) for n, v in zip(NAMES, std)}, at_bound=at_bound,
        robust_sigma_deg=float(np.degrees(sigma)),
        device_down=down.tolist(), device_forward=fwd.tolist(),
        checks=dict(
            wide_from_device_fit_rpy_deg=euler(VIEW.T @ C['e']),
            wide_from_device_logged_rpy_deg=np.degrees(logged[3:]).round(3).tolist(),
            narrow_from_device_fit_rpy_deg=euler(VIEW.T @ C['f']),
            forward_in_device_deg=dict(pitch=float(np.degrees(np.arctan2(-np.median(forward, axis=0)[2], np.median(forward, axis=0)[0]))),
                                       yaw=float(np.degrees(np.arctan2(np.median(forward, axis=0)[1], np.median(forward, axis=0)[0])))),
            logged_rpy_calib_deg=np.degrees(logged[:3]).round(3).tolist(),
            forward_vs_down_deg=float(np.degrees(np.arccos(np.median(forward, axis=0) @ down/np.linalg.norm(np.median(forward, axis=0))))),
            gravity_spread_deg=float(np.degrees(np.median(np.arccos(np.clip(gravity @ down, -1, 1))))),
            cabin_vs_wide_flip_rotvec_deg=np.degrees(Rotation.from_matrix(C['d'] @ C['e'].T @ np.diag([-1., 1, -1])).as_rotvec()).round(3).tolist(),
            cabin_axis_world_lon_lat_deg=[float(np.degrees(np.arctan2(*(np.array([0, 0, 1.]) @ C['d'] @ level)[[0, 2]]))),
                                          float(np.degrees(np.arcsin((np.array([0, 0, 1.]) @ C['d'] @ level)[1])))],
            wide_axis_world_lon_lat_deg=[float(np.degrees(np.arctan2(*(np.array([0, 0, 1.]) @ C['e'] @ level)[[0, 2]]))),
                                         float(np.degrees(np.arcsin((np.array([0, 0, 1.]) @ C['e'] @ level)[1])))],
        ))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'calibration.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('parameters', 'std', 'at_bound', 'robust_sigma_deg', 'checks')}, indent=1))


if __name__ == '__main__':
    main()
