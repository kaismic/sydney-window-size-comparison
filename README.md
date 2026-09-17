# Sydney window size comparison

This project compares 20, 40, …, 240-second windows for bus, car, and train classification. Every step is half its window size. The study uses seven leave-one-participant-out (LOPO) folds and seeds 42, 43, and 44, for 252 fixed XGBoost fits. It reports pooled held-out window-level macro F1, balanced accuracy, per-class scores, and paired macro F1 differences against the 60-second v2 setting.

## Data and reproducibility

- The adjacent `../all-tmd-v2` checkout must be at commit `6f8e77d7d179fde3fa664951957f63b51d634128`. The project imports that checkout's ingestion, feature extraction, LOPO manifest builder, model construction, and metric functions.
- `manifests/sydney-166.json` is an exact copy of v2's frozen Sydney snapshot. Only its bus, car, and train sessions with snapshot duration at least 240 seconds are candidates (149 before event-duration and feature-quality checks). No maximum duration applies. Tram, metro, later uploads, and explicitly non-`AU` sessions are excluded. Legacy sessions without a country label must be present in this exact snapshot.
- The committed `model-lock.json` is v2's NOR-selected fixed parameter lock. No NOR rows are loaded for this study. Its digest, features, and dependency versions are checked before a run.
- The same quality-surviving session intersection must produce at least one valid window at all 12 sizes and retain all seven participants. `prepare` freezes that cohort and a single LOPO manifest. Changed inputs or cohort selection are rejected on rerun.
- The completed local preparation retained all 149 candidates: 38 bus, 66 car, and 45 train sessions. The frozen cohort ID digest is `f4d3f3277bc6c3fbcdd7539ccc8a5aa05de6933a2737886cca6e4d8e4528a2dd`.
- The copied `requirements.lock` pins v2's direct dependency versions. Runtime packages are checked against the model lock. Python 3.12 or newer is required.

## Run locally

From this directory on PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
$env:ALL_TMD_DATA_DIR = 'D:\tmd-data'  # directory containing downloaded_sessions
.\.venv\Scripts\python.exe -m sydney_window_comparison verify
.\.venv\Scripts\python.exe -m sydney_window_comparison prepare
.\.venv\Scripts\python.exe -m sydney_window_comparison run
.\.venv\Scripts\python.exe -m sydney_window_comparison report
```

`--data-dir`, `--work-dir`, `--results-dir`, and `--n-jobs` override defaults. If `ALL_TMD_DATA_DIR` is unset, the command reads that one setting from `../all-tmd-v2/.env`. `all` runs prepare, run, and report in order. Preparation caches the frozen event data and each window's features under `work/`; completed model runs resume by window and seed under `results/runs/`. An incomplete run directory stops execution for review.

`results/window-curve.csv`, `per-class.csv`, `paired-vs-60s.csv`, `window-curve.png`, and `study-summary.json` are compact deliverables. The paired 95% intervals resample seeds, participants, and sessions with the same draws for each window and the 60-second baseline. Scores are pooled over windows, so window counts differ by size even though session membership is fixed.

Run tests with `.\.venv\Scripts\python.exe -m pytest`.
