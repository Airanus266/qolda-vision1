# Experiment record — fill only after collecting real observations

Status: **not conducted**. No participant results or real-user accuracy are provided with this repository.

## Question
Can a personal gesture classifier distinguish open palm, fist, pinch and point on a separate capture session? A later question is whether it generalizes to entirely new participants.

## Protocol
1. Obtain voluntary permission from participants. Begin with healthy adults; do not recruit patients or claim a clinical study.
2. Record device/browser, webcam position and lighting. Use anonymous participant codes.
3. Collect all four classes in each session, with a few independent recordings per class. Vary the pose slightly; do not record thousands of identical frames.
4. Use at least two separate sessions. For person-level evaluation, group all recordings from the same participant under one consistent group ID.
5. Export the CSV and preserve the raw session boundaries. Exclude synthetic data.
6. Run `node scripts/benchmark.mjs dataset.csv`, save the resulting JSON, and record failures as well as successes.
7. Manually count a small set of open/close cycles on a separate run and compare against the counter. Report false counts, missed counts and hand-loss conditions.

## Fill in
- Collection date:
- Device and browser:
- Number of consenting participants:
- Independent sessions per participant:
- Class sample counts:
- Split unit (session or participant):
- Camera placement and lighting:
- kNN accuracy / macro F1:
- Centroid accuracy:
- Confusion matrix:
- Manual cycle count / automatic cycle count:
- Main observed errors:
- What changed after inspecting errors:
- Limits of this experiment:

Do not replace blanks with invented data. Unit tests and simulation are not participant experiments.
