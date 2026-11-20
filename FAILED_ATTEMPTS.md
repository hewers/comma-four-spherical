# Earlier calibration attempts

Kept as a record of what did not work and why. The current calibration is `rig/calibration.json`; the method that produced it is in `RIG_CALIBRATION.md`, and the scripts behind these attempts have been removed from the repository.

| Attempt | Idea | Why it failed |
|---|---|---|
| Hand-traced road lines and poles, single frame | Fit a vanishing point per axis from two traced lines per camera, assume the cabin camera looks 180° backwards | The traces were made by eye on half-resolution frames and carried about 1° of uncertainty per line. Two lines per axis is the minimum for a vanishing point, so nothing averaged out. The 180° assumption was simply wrong: the cabin camera is pitched about 16° down relative to that flip. |
| Side-window horizon and shoulder line | Trace the horizon through both cabin windows plus a white shoulder line, and the distant skyline in the forward views | A skyline is not the horizon. Terrain, grade and the approximate lens model all move it, and the three windows disagreed with each other by 1–2°. Pooling them gave a cabin rotation whose held-out horizon error got worse, not better. |
| Festoon of SIFT matches between wide and narrow | Fit relative rotation and wide focal length from thousands of feature matches on held-out routes | This part worked, and its wide-to-narrow result (0.06°) survived into the final calibration. It cannot see the cabin camera at all, and on its own it left the cabin alignment at the 180° assumption. |
| Cabin yaw from window optical flow | Sweep the rear/side window motion vanishing point around the side-window horizon | The rear glass is usually overexposed or blocked, and the left and right windows disagreed: both fits ran to the ±15° sweep bound in opposite directions. Averaging disagreeing windows would have made a biased estimate look precise. |
| Delayed landmark matching | Match scenery seen through the windshield, then again seconds later through a cabin window | Only 25 image pairs were found even before holding out routes, and none survived the held-out split. Tree texture repeat and moving vehicles produced matches that satisfied the geometric test without being the same object. |
| Motion-only rigid-body fit | Estimate rotation and translation independently in each camera and find the rotation that makes them agree | Planar highway scenery and the camera lever arm make the translation direction poorly conditioned, and only 5 training plus 1 held-out segment produced usable estimates. |
| Pose, gravity and lane lines | Use fused `deviceMotion`, GNSS course and IMU gravity to level the vehicle, then fit shared lens and rotations from traced lane and barrier lines | The vehicle reference from pose and gravity was good (it is still used to level the sphere), but the fit still rested on a handful of traced lines. Cabin heading did not improve, and the focal length ran to its bound. |
| Shared-lens parametric fit | Independent rotations with one shared fisheye model, using traced highways and one barrier scene | Closer, but only four scenes, and the focal length again sat on its bound. Held-out horizon error improved slightly while cabin heading got worse. |
| Direct cabin yaw over a whole segment | Minimize cross-camera motion-plane residuals over all steady windows in one highway segment, 0.25° steps | The barrier trace disagreed by 2.6°, and a bounded yaw-only correction recovered only 0.1° of it. A pure yaw error is not what the data shows. |
| Community clipper recipe | Adopt the op-replay-clipper 360 filters: nominal focal length, cameras back to back, fixed vertical crop shift | It is a display recipe, not a calibration: the shift stands in for cabin pitch but leaves the relative rotations and the lens model unmodelled. It matched the cabin pitch to within a fraction of a degree, which was a useful independent check, but it was never a candidate. |

## One real bug, found late

`stitch.py` applied every camera rotation inverted relative to every fitter. Horizons that fitted to within 1° were rendered 3–5° off, and the cabin camera was drawn about 19° off in yaw. This made several of the failures above look worse than their underlying data, and it is why the A-pillar alignment looked lopsided. Fixed in `maps()`; the fitters and the renderer now agree on `ray = basis.T @ world`.

## What finally worked

Calibrate each camera directly against the IMU/device frame from feature tracks, gyro rotation and pose velocity, so the cameras never need to see the same scenery. See `RIG_CALIBRATION.md`.
