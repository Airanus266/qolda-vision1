# Validation record — release 0.3

Date: 29 October 2026. Software correctness and scientific performance are separate.

## Completed

- `npm run check`: JS syntax, UI references and shipped local HTML assets passed.
- `npm test`: 20 tests passed. Covers geometry, temporal counting, tracking-loss reset, provenance, session exclusion, imports/exports and held-session unknown rejection.
- `python -m unittest discover -s research -p 'test_*.py'`: 7 tests passed. Covers point-cloud permutation invariance, motion descriptors, gap rejection, timestamp pairing, agreement, detection-failure denominator and raw-ADC rejection.
- Public-data training: all 14 held-ID folds completed for three models and three feature variants. Actual per-fold outputs, weights and predictions retained.
- `python research/verify_results.py`: verified original CSV hash, sampled indices, disjoint split manifests, metrics from predictions, summaries and predictions reproduced from all 14 saved MLP checkpoints.
- Browser inspection: release 0.3 initial view and new research-capture form rendered; empty export and recording without a camera rejected; unknown-rejection calibration without data rejected. Synthetic demonstration and basic session controls had been inspected in the prior release. These checks do not test a real webcam.
- Three-page report rendered and visually checked. Video file duration is 60 seconds; it is a synthetic walkthrough plus measured public-data results.
- Self-contained Colab notebook contains the reviewed benchmark and plotting code. Those scripts were executed locally; hosted Colab execution was not independently tested.

## Pending external validation

- Actual webcam acquisition, MediaPipe inference and capture accuracy on the owner's device.
- Real camera export followed by movement, condition and participant-level analyses. Those paths are supplied but not validated on a recruited camera dataset.
- Physical glove firmware adapter, latency measurement, calibration and real camera–glove agreement.
- Consented multi-person camera recordings; actual lighting/angle reliability measurements.
- Repeated seeds, more datasets and independent replication.
- GitHub Actions execution after upload; the workflow is supplied under `.github/workflows/quality.yml`, but no green remote run is claimed.

No therapeutic result, clinical validity, hardware study or original participant recruitment is claimed. Public motion-capture accuracy cannot be presented as camera accuracy.
