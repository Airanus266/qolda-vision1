# Camera and glove study protocol

Status: implemented collection/analysis tools, no supplied hardware or recruited-person results.

## Recruitment and recording

Invite 10–20 consenting volunteers only if practical; do not invent enrollment. Use P01, P02, etc. consistently across all sessions for one person. Do not collect names, photos, health labels or patient data. Explain what numerical movement data is saved, where it goes and how to request deletion. Minors need appropriate guardian/institutional permission. Participants may stop at any time; use comfortable voluntary movements only.

For each participant record the four known poses (Open palm, Fist, Pinch, Point), static-hold and open-close tasks in multiple independent sessions. Suggested plan: 10-second poses, 20-second static holds and 30-second comfortable cycles. Vary one condition at a time: normal/dim/backlit, frontal/side/occluded, hand and measured distance. Keep device and camera configuration in a separate study log. No recording plan here asserts that recordings already exist.

Enable camera in Motion studio, then Method → Research capture. Set IDs, task and conditions; start, stop, export. Do not change the instructed pose during a labeled recording. Calibration references are separate calibration-open/closed recordings, not labeled test examples. Research capture saves every inference attempt (cap 9,000), including failures. Dropped frames before an inference attempt are not counted. No image/video is exported.

## Reliability and movement

Run `python research/analysis.py captures/*.json`, then `python research/condition_plots.py research/results/capture-analysis.json`.

Failure rate = missing-hand attempts / all recorded attempts; it measures detection availability, not landmark correctness. Per-condition plots use per-recording rates and show sample counts. They cannot establish population differences without independent people and balanced conditions.

Jitter is RMS mean-landmark displacement between adjacent valid static-hold frames in normalized screen coordinates. It includes real motion, is frame-rate dependent and is not sensor noise alone. No jitter is reported for movement tasks.

Movement uses normalized mean non-thumb flexion. It resamples at median interval, applies Savitzky–Golay (window up to 11, order 3) and finite differences. Outputs: p95–p05 amplitude, mean absolute speed, RMS jerk, cycle-period coefficient of variation and peak-amplitude SD. Jerk is sampling/smoothing dependent. Long gaps invalidate the calculation; do not silently fill them. These are exploratory descriptors, not clinical scores or range of motion in degrees.

## Participant-held-out camera models

`python research/camera_benchmark.py captures/*.json`

Requires at least 3 independent IDs, all four labels per ID; 10–20 people would be stronger but are not fabricated. It samples up to 75 examples per person/class. The same world/screen complete cases and held-participant splits are used for centroid, kNN and MLP. Scaling is training-only. It compares world vs screen and removes each of the six features separately. Calibration ablation requires separate open/closed references for every person. Using test-person reference poses makes this **personalized calibration**, not zero-shot generalization. Frames within sessions are correlated; participant count is the meaningful unit. Do not choose hyperparameters on the reported test folds and claim a fresh validation.

## Glove recording and alignment

No glove is required for the public benchmark. Real agreement experiments require actual hardware. The supplied serial recorder expects one JSON line per reading:

```json
{"values":{"thumb":0.2,"index":0.3,"ring":0.1,"little":0.4}}
```

These are normalized flexion proxies, 0 open to 1 closed. The current contract excludes the middle finger. Raw ADC readings, servo commands and internal model predictions are not ground-truth joint angles. Adapt the actual firmware explicitly to this contract; the recorder does not control actuators.

On the **same computer** as the browser, start:

```sh
python research/glove_capture.py --port COM3 --baud 115200 --seconds 60 --output glove.json
```

Record camera observations during the overlapping interval. Compare:

```sh
python research/compare_glove.py camera.json glove.json --tolerance-ms 50 --offset-ms 0 > agreement.json
```

Camera timestamps are host epoch at inference start; glove timestamps are host epoch at serial receipt. Pairing is greedy nearest-time, within tolerance, without sample reuse. This is receipt-time alignment, **not hardware synchronization**. Video exposure timing, USB/serial latency and clock changes remain sources of error. Report paired count, tolerance, offset and unmatched observations. Fix an offset using a separate calibration sequence, never optimize it on test results. Split discontinuous intervals and avoid clock changes.

Outputs are per-finger proxy MAE, RMSE, bias and correlation. Camera thumb geometry and flex-sensor responses need not represent the same quantity. To claim degree error, independently calibrate each sensor against a physical angle reference, match anatomical joints and reserve a separate evaluation sequence. No such calibration is included or claimed.

## Unknowns

The public experiment excludes UCI class 5 from both learning and calibration and reports detection metrics on eligible test IDs. The browser's optional threshold uses only known examples from a separate session. Test novel poses in separately labeled Unknown recordings; do not use unknown test labels to tune a threshold. A rejected pose is not a diagnosis and accepting a pose does not establish reliability.
