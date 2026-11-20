# Rig calibration summary

Calibration of the three cameras of one comma four (`mici`, OS04C10 sensors, openpilot 0.11.3) from its own drive logs. The machine-readable file is `rig/calibration.json`, which holds these values at full precision. Method, validation detail and refit commands are in `RIG_CALIBRATION.md`.

The fit uses measured data only: about 550,000 feature tracks across 594 segments, gyro rotation, pose velocity, and 27,000 wide/narrow feature matches. Nothing is hand-traced. The published values come from a fit on all 594 segments with none held out; held-out fits were used only to validate.

## Frames

- **Image:** pixel (u, v), u right, v down, 1344×760, integer coordinates at pixel centres.
- **Camera:** x right, y down, z along the optical axis.
- **Device:** openpilot device frame, x forward, y right, z down. An IMU sample `v` maps to `(-v[2], -v[1], -v[0])`.
- **Vehicle (the sphere):** x right, y down (mean gravity), z forward (mean direction of travel, horizontal).
- **Equirectangular:** `lon = atan2(x, z)`, `lat = asin(y)`; column = `(lon/2π + 0.5)·W`, row = `(lat/π + 0.5)·H`.

## Intrinsics

| Camera | Model | Focal (px) | Centre (px) | Radial k1, k2 |
|---|---|---|---|---|
| Wide (`ecamera`) and cabin (`dcamera`), shared | Equidistant fisheye | 454.99 | (668.53, 375.98) | −0.00240, 0.00796 |
| Narrow (`fcamera`) | Pinhole | 1177.53 | (672, 380), assumed | — |

- **Fisheye unprojection:** `p = ((u−cx)/f, (v−cy)/f)`, `ρ = |p|`, `θ = ρ(1 + k1ρ² + k2ρ⁴)`, `ray = (p/ρ·sinθ, cosθ)`. Use it out to 92° from the axis; the image circle fades beyond that.
- **Pinhole unprojection:** `ray = normalize((u−cx)/f, (v−cy)/f, 1)`.
- **Nominal values for comparison:** openpilot and the community clipper use 425.25 px for the fisheyes and 1141.5 px for the narrow camera, both centred at (672, 380) with no radial terms.

## Extrinsics

`ray_camera = camera_from_device @ v_device`. Rotation only; the lenses are within a few centimetres of each other and translation is not estimated.

```
wide (e) camera_from_device        optical axis: 1.47° left of travel, 3.72° above horizontal
  [-0.005551,  0.999936, -0.009815]
  [ 0.015727,  0.009901,  0.999827]
  [ 0.999861,  0.005396, -0.015781]

cabin (d) camera_from_device       optical axis: 178.66°, 19.44° below horizontal
  [ 0.001190, -0.999755,  0.022088]
  [ 0.285988,  0.021506,  0.957992]
  [-0.958232,  0.005177,  0.285944]

narrow (f) camera_from_device      optical axis: 1.49° left of travel, 3.33° above horizontal
  [-0.005502,  0.999930, -0.010479]
  [ 0.008989,  0.010529,  0.999904]
  [ 0.999944,  0.005407, -0.009046]

device_from_vehicle                v_device = device_from_vehicle @ v_vehicle
  [-0.031804, -0.049363,  0.998274]
  [ 0.998373,  0.045720,  0.034068]
  [-0.047323,  0.997734,  0.047829]
```

- **Camera to sphere:** `camera_from_vehicle = camera_from_device @ device_from_vehicle`, stored as `basis` in the file.
- **Cabin versus wide:** the cabin camera is the wide camera turned 180° about vertical, then pitched 15.7° down, with 0.36° yaw and −0.66° roll.
- **Device mount:** `device_from_vehicle` is specific to this installation (how the device sits on this windshield). The camera-from-device rotations and intrinsics belong to the device itself.

## Timing

- **Gyro lag:** row `v` of an image is at gyro time `(SOF+EOF)/2 + offset + 0.0254·(v/760 − 0.5)` seconds, with offset 0.0196 (wide), 0.0190 (cabin), 0.0203 (narrow).
- **Cabin phase:** cabin frames are exposed about 25 ms after wide frames with the same frame ID.

## Accuracy

- **Held-out routes (9 of 35):** wide/narrow mismatch went from 0.60° to 0.055°. Cabin epipolar residual in turns went from 0.127° to 0.039°.
- **Independent halves of the routes:** focal length agrees within 1.9 px, cabin-versus-wide rotation within 0.25°, absolute rotations within 0.36°.
- **Level:** accelerometer gravity and the axis the car yaws about in 788 turns agree to 0.4° in roll.
- **Visual:** on flat highway the horizon sits on the equator in the wide view and both cabin side windows; tree trunks run along the meridians.

## Limits

- **Gaps at the A-pillars:** a lens-shaped hole about 7° wide remains at the horizon near ±90° on each side. The sensor crops each fisheye at about 87°, so neither camera sees there.
- **Parallax:** distant scenery lines up; pillars, mirrors and dashboard are seen from slightly different positions.
- **Level is car-fixed:** on a grade or banked curve the horizon tilts with the car.
- **One device:** these values are for this unit. Another comma four should be close in intrinsics and cabin tilt, but its centre, small rotations and mount will differ.
