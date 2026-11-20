"""Audit IMU and camera timing before attempting nonoverlapping rig calibration."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import capnp
import numpy as np
import zstandard

from stitch import log_events


def inspect(path):
    gyro, accel, cameras = [], [], defaultdict(dict)
    sensors = defaultdict(set)
    for e in log_events(path):
        try:
            k = e.which()
        except capnp.KjException:
            continue
        if k in ('gyroscope', 'accelerometer'):
            event = getattr(e, k)
            which = event.which()
            if which not in ('gyro', 'gyroUncalibrated', 'acceleration'):
                continue
            v = getattr(event, which).v
            if event.timestamp > 0 and len(v) == 3:
                (gyro if k == 'gyroscope' else accel).append((event.timestamp, *v))
        elif k in ('wideRoadCameraState', 'cabinCameraState', 'narrowRoadCameraState'):
            frame = getattr(e, k)
            sensors[k].add(str(frame.sensor))
            if frame.timestampSof and frame.timestampEof:
                cameras[k][frame.frameId] = (frame.timestampSof + frame.timestampEof)//2
    if len(gyro) < 20:
        return None, np.empty((0, 3))
    g = np.asarray(gyro)
    g = g[np.argsort(g[:, 0])]
    t = (g[:, 0] - g[0, 0])/1e9
    dt = np.diff(t)
    good = (dt > 0) & (dt < .05)
    rate = 1/float(np.median(dt[good])) if np.any(good) else None
    # Approximate excitation statistics, not a rotation trajectory. Remove
    # one median bias estimate and integrate adjacent samples in one-second bins.
    bias = np.median(g[:, 1:], axis=0)
    omega = (g[:-1, 1:]+g[1:, 1:])/2-bias
    bins = np.floor(t[:-1]).astype(int)
    count = np.bincount(bins[good], weights=dt[good])
    integrals = np.column_stack([np.bincount(bins[good], weights=dt[good]*x[good], minlength=len(count))
                                for x in omega.T])[count > .8]
    report = dict(segment=path.parent.name, gyro_hz=rate, gyro_samples=len(gyro), accel_samples=len(accel),
                  sensors={k: sorted(v) for k, v in sensors.items()},
                  one_second_rotation_p95_deg=float(np.degrees(np.quantile(np.linalg.norm(integrals, axis=1), .95))) if len(integrals) else None)
    wide = cameras['wideRoadCameraState']
    for key in ('cabinCameraState', 'narrowRoadCameraState'):
        other = cameras[key]
        if wide.keys() & other.keys():
            delta = np.array([other[i]-wide[i] for i in wide.keys() & other.keys()])/1e6
            report[key+'_minus_wide_ms'] = np.quantile(delta, [.05, .5, .95]).tolist()
    # Check whether the timestamp domains overlap, without assuming a
    # device-to-sensor axis transform or shutter readout direction.
    report['gyro_camera_time_overlap'] = bool(wide and min(wide.values()) < g[-1, 0] and max(wide.values()) > g[0, 0])
    return report, integrals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('previews/rig-audit.json'))
    ap.add_argument('--per-route', type=int, default=2)
    args = ap.parse_args()
    if args.per_route < 1:
        ap.error('per-route must be positive')
    routes = defaultdict(list)
    for p in Path('rlogs').glob('*/rlog.zst'):
        if all(Path(p.parent.name, f'{k}camera.hevc').is_file() for k in 'ed'):
            routes[p.parent.name.rsplit('--', 1)[0]].append(p)
    reports, rotations = [], []
    for route, paths in sorted(routes.items()):
        paths.sort(key=lambda p: int(p.parent.name.rsplit('--', 1)[1]))
        for i in np.unique(np.linspace(0, len(paths)-1, min(args.per_route, len(paths))).astype(int)):
            try:
                report, v = inspect(paths[i])
            except (OSError, ValueError, zstandard.ZstdError, capnp.KjException) as exc:
                print(paths[i], type(exc).__name__, flush=True)
                continue
            if report:
                reports.append(report)
                rotations.extend(v)
                print(report['segment'], report['gyro_hz'], report['one_second_rotation_p95_deg'], flush=True)
    sv = np.linalg.svd(rotations, compute_uv=False) if rotations else np.zeros(3)
    result = dict(sampling='evenly spaced segments per route', segments=reports,
                  rotation_excitation_relative_singular_values=(sv/sv[0]).tolist() if sv[0] else [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{len(reports)} segments, {len(rotations)} one-second bins, relative excitation {result["rotation_excitation_relative_singular_values"]}')


if __name__ == '__main__':
    main()
