# Model card — release 0.3

Two separate systems are supplied. Their performance claims are not interchangeable.

## Public motion-capture benchmark

A PyTorch 18→64→32→5 MLP, kNN5 and nearest centroid consume permutation-invariant features of unordered Vicon markers. Data: UCI405, CC BY 4.0; 5,050 sampled records, 14 distinct IDs (source describes 12 users; discrepancy unresolved). Every test ID and validation ID are excluded from training and scaler fitting. See README, saved split manifests and three-page report for actual results and limitations.

Checkpoints include state_dict, scaler means/scales, classes and held/training IDs. These are per-fold evaluation models, not a model trained on all data for deployment. They cannot process MediaPipe's six-feature vectors. No clinical validation or original participant recruitment.

## Browser system

Google MediaPipe Hand Landmarker, pinned Tasks Vision 0.10.18 and hand_landmarker float16/1 model, is third-party pretrained perception. Six geometric features feed a personal kNN or centroid. Four known labels: Open palm, Fist, Pinch, Point. Optional distance rejection uses a held-out calibration session; recording/import invalidates it. Neighbor agreement is not a calibrated probability.

Browser evaluation holds out capture sessions, not automatically people. A separate Python camera runner groups by explicit participant IDs and supports six-feature, world/screen and personalized-calibration ablations. No actual webcam dataset or measured webcam accuracy is supplied. Tracking, calibration and movement descriptors require target-device validation.

## Intended use and limits

Educational hand-posture and motion experiments. No diagnosis, prescribed exercises, treatment or assistive-control safety claims. Occlusion, uncommon poses, lighting and motor impairments can cause errors. Public Vicon results do not establish MediaPipe performance. Movement amplitude/jerk are normalized numerical descriptors, not validated rehabilitation scores. Report sensor, split unit, provenance, missing classes and limitations whenever sharing results.
