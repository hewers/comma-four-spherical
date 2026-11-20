# Rig calibration

`rig/calibration.json` is the calibration to render with. It was fitted from measured data only: feature tracks in each camera, gyro rotation, pose velocity, and wide/narrow feature matches. Nothing in it is hand-traced.

## Method

Each camera is calibrated against the device (IMU) frame, so the cameras never need to see the same scenery.

1. `rig_masks.py` finds, for each camera, the pixels that change while driving (windshield, side and rear windows) and removes dashboard, pillars, roof and occupants. Masks are in `rig/masks/`.
2. `rig_collect.py` samples yaw turns, pitch/roll events and straight driving in every segment. For each sample and camera it tracks features between two frames 50–100 ms apart, integrates the bias-corrected gyro between that camera's own exposure times, and records the direction of travel from `deviceMotion.velocityDevice`.
3. `rig_overlap.py` matches SIFT features between simultaneous wide and narrow frames.
4. `rig_fit.py` solves jointly for:
   - the rotation of each camera relative to the device;
   - one fisheye model shared by wide and cabin (focal length, centre, two radial terms) and the narrow focal length;
   - a time offset per camera, the sensor readout time, and a constant correction to the logged travel direction.

   The residual for a track is the angle between its second ray and the epipolar great circle predicted by the gyro rotation and travel direction. Straight driving fixes the travel axis in each camera, turns fix the rotation about it, and the wide/narrow matches tie the two forward cameras together. The cabin camera starts from a coarse search over "roughly backwards", not from an assumed 180° flip.
5. The output sphere is levelled with the median accelerometer gravity and pointed along the median travel direction.

## Result (594 segments, about 3,400 frame pairs and 160,000–195,000 tracks per camera, 27,000 wide/narrow matches)

| Quantity | Fitted | Nominal (openpilot / clipper) |
|---|---|---|
| Fisheye focal length | 455.0 px | 425.25 px |
| Fisheye centre | (668.5, 376.0) | (672, 380) |
| Fisheye radial terms k1, k2 | −0.0024, 0.0080 | 0, 0 |
| Narrow focal length | 1177.5 px | 1141.5 px |
| Cabin relative to a 180° flip of wide | pitch 15.7°, yaw 0.36°, roll −0.66° | 0 (clipper shifts the image instead) |
| Wide axis in the levelled frame | 1.5° left, 3.7° up | — |
| Cabin axis in the levelled frame | 178.7°, 19.4° down | — |
| Gyro lag behind the cameras | 19–20 ms | — |
| Sensor readout time | 25 ms | — |

Angle from the optical axis at 672 px radius is 87.4°, so each fisheye covers slightly less than a hemisphere horizontally. The image circle fades beyond about 92°.

## Validation

- **Held-out routes (9 of 35).** Median epipolar residual, nominal → fitted, in degrees:

  | Camera | Straight | Yaw turns | Pitch/roll |
  |---|---|---|---|
  | Wide | 0.020 → 0.017 | 0.032 → 0.018 | 0.034 → 0.017 |
  | Cabin | 0.034 → 0.023 | 0.127 → 0.039 | 0.048 → 0.029 |
  | Narrow | 0.009 → 0.007 | 0.021 → 0.010 | 0.030 → 0.009 |

  Wide/narrow matches on held-out routes: 0.60° → 0.055°.
- **Independent halves.** Fitting odd and even routes separately gives focal lengths within 1.9 px, centres within 1.1 px, and cabin-versus-wide rotation within 0.25°. Absolute rotations agree within 0.36°.
- **Gravity.** The accelerometer mean and the axis the car yaws about in 788 turns agree to 0.4° in roll. In pitch the accelerometer is perpendicular to the travel direction to 0.06°; the turn axis is 1.2° off, so the accelerometer is used.
- **Visual.** On flat highway the horizon lies on the equator in the wide view and in both cabin side windows; sequoia trunks follow the meridians; the barrier at `2026-11-15T21:47:49.963Z` continues across the driver-side seam. Stills: `uv run rig_samples.py rig/calibration.json --grid --output previews/rig/samples-grid`.

`uv run rig_validate.py rig/calibration.json` repeats the overlap and gravity checks.

## What calibration cannot fix

- **Gaps beside the A-pillars.** The sensor crops each fisheye at about 87° horizontally, and the cabin camera is pitched 19° down, so a lens-shaped hole about 7° wide remains at the horizon near ±90°. Where the fields do overlap (thin wedges above and below the hole) wide and cabin are averaged 50/50.
- **Parallax.** The cameras are a few centimetres apart. Distant scenery lines up; the A-pillars, mirrors and dashboard are seen from slightly different positions.
- **Level is car-fixed.** The sphere is levelled for the car's average attitude. On a grade or banked curve the horizon tilts with the car.
- **Timing.** Cabin frames are exposed about 25 ms after wide frames; the renderer uses the nearest frame and does not interpolate.

## Earlier attempts

Every earlier candidate (`calibration.json`, `calibration-best.json`, `calibration-shared.json`, the pose-line fit) is superseded. Two problems affected all of them:

- `stitch.py` applied each camera rotation inverted relative to the fitters. Horizons that fitted to within 1° were drawn 3–5° off, and the cabin camera was drawn about 19° off in yaw. Fixed in `maps()`.
- The cabin and lens fits rested on a few lines traced by eye on half-resolution frames. The focal length ran to its bound and the cabin yaw came out 9° from straight back.

## Community reference

[nelsonjchen/op-replay-clipper](https://github.com/nelsonjchen/op-replay-clipper) (read from the Software Heritage snapshot of 2026-04-26; the GitHub repository is currently blocked) renders 360° with FFmpeg `v360=dfisheye:equirect:ih_fov=195:iv_fov=122` after padding the driver image by 290 px and centre-cropping. That is the nominal equidistant model (567 px at 1928 wide, 425.25 px here), cameras exactly back to back, and a fixed vertical shift standing in for the cabin pitch. At this sensor size the shift is about 19°, which agrees with the fitted cabin pitch. Its forward-upon-wide mode uses the logged `wideFromDeviceEuler` with nominal focal lengths.

## To refit

```sh
uv run rig_masks.py
OPENBLAS_NUM_THREADS=1 uv run rig_collect.py --jobs 12
OPENBLAS_NUM_THREADS=1 uv run rig_overlap.py
uv run rig_fit.py --split quarter --output previews/rig/fit-quarter   # held-out report
uv run rig_fit.py --split none --output rig                           # production fit
```
