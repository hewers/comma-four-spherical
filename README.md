# Openpilot drive stitching

From the repository root:

```sh
uv sync
OPENBLAS_CORETYPE=ARMV8 uv run stitch.py calibration-lines.json --solve > calibration.json
uv run stitch.py calibration.json --output videos-4k --jobs 3 --resume
```

The `OPENBLAS_CORETYPE` override is for the local ARM host. Use `--drive ROUTE` to render a single route or `--segment SEGMENT --frames 40 --output previews` for a short check. The batch sorts each route numerically by segment, skips unreadable or incomplete segments, and publishes `ROUTE.mp4` in the output directory after the route finishes. Check the render log for gaps. No frames are fabricated for missing segments.

Rendering is serial within each drive (`--jobs` caps concurrent drives): rlogs are decompressed one message at a time, camera frames are piped from FFmpeg and composited one output frame at a time, and each encoded segment goes directly to disk. Decoders and frame buffers are released after each segment. Temporary segment MP4s are deleted after concatenation. Allow disk space for two copies of each active drive during concatenation or metadata injection. `--resume` skips already published drives; use a fresh output directory when changing calibration.

`calibration-lines.json` contains manually traced road edges and distant horizons from `000000ea--f23abe10b0--5`. For the cabin camera the horizon is traced through both side windows and the white shoulder marking through the right side of the image fixes heading. After lens unprojection, the solver fits a horizon plane and intersects it with each road-line plane to estimate the longitudinal vanishing direction. The horizon normal sets vertical; all cameras are rotated into the same road-relative frame. Intrinsics are based on the openpilot OS04C10 defaults, with an approximate equidistant fisheye. This is a single-scene calibration, not a distortion or parallax correction, and assumes the mount did not move between drives.

The default output is 4096×2048 (2:1 equirectangular), with a UTC ISO 8601 timestamp in an uncovered region. The final MP4 is tagged as monoscopic spherical equirectangular using Google's Spatial Media tool. If the headset player ignores the tag, select monoscopic 360° mode manually. The narrow road image takes priority in the overlap. Areas outside every camera are black; camera translation/parallax and occlusions are not corrected.

Frame timestamps come from the rlog encoder indices and are placed on the UTC clock using rlog clock events (or init data when needed). Sampled `dcamera.hevc` is video only. Audio is AAC in `qcamera.ts`; the qRoad encode timestamp anchors it to the camera timeline. Segments without usable audio get silence. The rlog `Microphone` event in the upstream schema contains sound levels, not waveform samples.

Schemas are adapted from commaai/openpilot revision `ff300ecf785152db7a518ffd959871dab6b6e6d1`; unused vehicle type references are stubbed for standalone log reading.
