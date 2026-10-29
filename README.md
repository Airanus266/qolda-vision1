QOLDA Vision — Hand Motion Research Lab
A hand-motion research toolkit combining a browser camera lab, a trained PyTorch benchmark, and camera–glove comparison tools.
Benchmark results
Model	Mean held-user-ID accuracy ± SD	Macro F1
Nearest centroid	75.53% ± 16.21 pp	0.712
kNN, k=5	80.21% ± 14.87 pp	0.774
PyTorch MLP	86.83% ± 15.28 pp	0.847



Measured on 5,050 sampled records from UCI Motion Capture Hand Postures, using identical held-ID splits and training-only standardization. These results concern public Vicon motion-capture data; webcam and QOLDA-glove performance require separate measurements. The source describes 12 users, while the CSV contains 14 unique IDs; that discrepancy remains unresolved. SD describes variation across ID folds. The experiment uses one seed and fixed hyperparameters.

[Technical report](docs/QOLDA_Research_Report.pdf) · [Full results](research/results/TABLES.md) · [Notebook](QOLDA_Research_Colab.ipynb) · [60-second walkthrough](docs/assets/walkthrough-60s.mp4) · [Study protocol](docs/RESEARCH_PROTOCOL.md)
Features
- Browser-local MediaPipe hand tracking, geometric features, personal calibration and repetition counting.
- Personal kNN gesture learning, held-session evaluation and optional unknown-gesture rejection.
- PyTorch MLP compared with kNN and centroid on the same public-data splits.
- Feature ablations, unknown-class evaluation and synthetic marker-dropout stress tests.
- Camera recording with landmarks, missing-hand observations and lighting/view metadata.
- Movement amplitude, speed, RMS jerk and cycle-variability analysis.
- Serial glove capture and timestamp-based proxy agreement analysis.
Run the browser app
Extract the project and double-click START_WINDOWS.bat, or run this command from the project folder:
python -m http.server 8000 --directory dist
Open http://localhost:8000 and keep the terminal running. A recent browser and Python 3 are required. Camera mode needs internet access to download MediaPipe assets and camera permission. No API key or npm install is needed. Simulation is explicitly labeled; the walkthrough uses synthetic animation alongside measured public-data results.

Motion studio displays landmarks and movement descriptors. Gesture learning collects camera examples and evaluates whole held-out sessions. Method & evidence → Research capture records anonymous IDs, task and conditions, then exports numerical observations. Export before starting a new recording or reloading the page.
Run the notebook
Upload QOLDA_Research_Colab.ipynb to Google Colab and select Runtime → Run all. The self-contained notebook installs dependencies, downloads the public dataset, trains the models and displays the result table and figure. No repository URL or personal-data upload is required.

All four code cells were executed successfully in a fresh local IPython namespace on 29 October 2026, including a new data download, training and rendered outputs. Results matched the supplied benchmark. [Executed notebook](research/results/QOLDA_Research_Colab.executed.ipynb) · [Execution record](docs/NOTEBOOK_EXECUTION.json). Hosted Google Colab execution requires account access and remains unverified.
Reproduce from Python
Python 3.12 is recommended. Use a virtual environment.
python -m pip install -r research/requirements.txt
python research/benchmark.py
python research/plot_results.py
python research/verify_results.py
The benchmark samples up to 75 records per class per ID, with seed 42. In each of 14 folds, one ID is test, the next cyclic ID is validation, and the remaining IDs are training. Models use 18 permutation-invariant descriptors of unordered markers. The 18→64→32→5 MLP trains for 50 epochs with Adam. Raw data is downloaded separately; source hash, sample indices, splits, predictions and per-fold checkpoints are included. Rerunning overwrites research/results/.

The public-data MLP is a separate experiment from the browser's six-feature personal classifier. The browser uses Google's pretrained MediaPipe Hand Landmarker.

Removing marker count lowers MLP accuracy to 80.09%; removing pair-distance descriptors gives 87.25%. With class 5 withheld from learning and threshold calibration, unknown recall averages 57.44%, known false rejection 11.25%, and AUROC 0.937 across eligible folds. Full methods and limitations are in the report.
Camera and glove studies
The [study protocol](docs/RESEARCH_PROTOCOL.md) explains recording, participant splits, condition analysis and hardware adaptation.
python research/analysis.py captures/*.json
python research/condition_plots.py research/results/capture-analysis.json
python research/camera_benchmark.py captures/*.json
python research/glove_capture.py --port COM3 --seconds 60 --output glove.json
python research/compare_glove.py camera.json glove.json
Camera and glove recordings are not supplied. The comparison uses host receipt timestamps and normalized sensor proxies; degree-level agreement requires independent physical calibration. Real lighting reliability, multi-person webcam performance and clinical outcomes remain to be measured.
Checks and source map
