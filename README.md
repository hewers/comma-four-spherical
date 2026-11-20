# Openpilot drive stitching

From the repository root:

```sh
uv sync
OPENBLAS_CORETYPE=ARMV8 uv run stitch.py calibration-lines.json --solve > calibration.json
uv run stitch.py calibration.json --width 1280
```

The `OPENBLAS_CORETYPE` override is for the local ARM host. Use `--drive ROUTE` to render a single route or `--segment SEGMENT --frames 40 --output previews` for a short check. The batch sorts each route numerically by segment, skips unreadable or incomplete segments, and publishes `videos/ROUTE.mp4` after the route finishes. Check the render log for gaps. No frames are fabricated for missing segments.

`calibration-lines.json` contains manually traced lane/shoulder lines and poles from `000000ea--f23abe10b0--5`. The solver estimates vanishing directions after lens unprojection and stores a fixed rotation per camera. Intrinsics are based on the openpilot OS04C10 defaults; the fisheye is approximated as equidistant. The cabin rotation **assumes** a 180-degree yaw relative to the wide camera; this is not an independent rear calibration. To fit it independently, remove `reference` and provide measured `longitudinal` and `vertical` parallel-line groups in cabin pixel coordinates. These rotations assume the mount did not move between drives.

The output is an equirectangular image with a separate UTC ISO 8601 strip below it (ordinary MP4, no spherical player metadata). The narrow road image takes priority in the overlap. Areas outside every camera are black; camera translation/parallax and occlusions are not corrected.

Frame timestamps come from the rlog encoder indices and are placed on the UTC clock using rlog clock events (or init data when needed). Sampled `dcamera.hevc` is video only. Audio is AAC in `qcamera.ts`; the qRoad encode timestamp anchors it to the camera timeline. Segments without usable audio get silence. The rlog `Microphone` event in the upstream schema contains sound levels, not waveform samples.

Schemas are adapted from commaai/openpilot revision `ff300ecf785152db7a518ffd959871dab6b6e6d1`; unused vehicle type references are stubbed for standalone log reading.
