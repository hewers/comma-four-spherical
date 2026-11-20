# Multi-drive calibration investigation

Run `OPENBLAS_NUM_THREADS=1 nice -n 12 uv run calibration_survey.py` from the repository root while rendering. It samples the first frame of every segment with both forward sources, finds SIFT correspondences, rejects outliers with a homography, and fits relative rotation and wide focal length with robust angular loss. Every fifth route is held out wholesale for validation. Homographies and the ray fit favor dominant scenery and can still be biased by road-plane parallax, so inspect previews as well as residuals.

`calibration-scenes.json` contains manual half-resolution road and distant skyline traces for three paired wide/cabin frames. The solver pools the wide-camera reference and estimates cabin-to-wide rotation from paired horizon normals and road direction. Cabin validation leaves each scene out in turn. A skyline is only a proxy for level: terrain, roll/grade, and the approximate lens model can move it. The narrow overlap fit does not validate wide lens distortion at the edges.

`OPENBLAS_NUM_THREADS=1 uv run rear_window_survey.py` checks whether scenery tracked through the rear and both side windows supports a common cabin vanishing point on the side-window horizon. It filters for fast, low-rotation intervals using rlog camera odometry, rejects inconsistent optical flow, and reports independent window fits in `previews/rear-window-fit.json`. Rear glass is often overexposed or blocked; the glass edge, headrests, and luggage are not horizon observations. Disagreeing window fits are a reason to retain the current cabin calibration, not to average them blindly.

Outputs go to `previews/calibration-survey/`. The report requires at least 10 held-out forward segments and 25% improvement in both held-out forward and cabin median errors before recommending a visual review. The survey never changes live calibration or rerenders automatically. If the candidate passes and a multi-drive visual check confirms an improvement, render it into a new output directory so existing drives are not overwritten:

```sh
uv run stitch.py previews/calibration-survey/candidate.json --output videos-4k-refined --jobs 3 --resume
```
