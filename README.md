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

`results/window-curve.csv`, `absolute-macro-f1.csv`, `per-class.csv`, `paired-vs-60s.csv`, `window-curve.png`, `absolute-macro-f1.png`, `window-curve-combined.png`, `paired-vs-60s.png`, and `study-summary.json` are compact deliverables. The absolute and paired 95% intervals use 5,000 bootstrap draws that resample seeds, participants, and sessions; paired draws are shared between each window and the 60-second baseline. Scores are pooled over windows, so window counts differ by size even though session membership is fixed.

`results/window-curve.png` plots macro F1 against window size without intervals; `results/absolute-macro-f1.png` plots the same absolute means with their bootstrap intervals. `results/window-curve-combined.png` places those plots in one side-by-side figure, with the window curve on the left. Balanced accuracy remains in the CSV and JSON results. The detailed, standalone LaTeX report is `reports/report-window-size-comparison.tex`. To compile it with `pdflatex`, run `cd reports` followed by `pdflatex report-window-size-comparison.tex` twice for references. Its content is in `reports/window-size-comparison-section.tex`, which starts with `\subsection{Comprehensive Window Length Comparison}` and can be inserted under the Results section of `../../reports/report-8.tex`. When compiling that parent report from its `reports/` directory, add `\graphicspath{{../ml/sydney-window-size-comparison/results/}}` and `\input{../ml/sydney-window-size-comparison/reports/window-size-comparison-section.tex}` at the intended location. The parent already loads `graphicx`.

`reports/report-window-size-comparison-independent.tex` is a separate report focused on the full window-length grid. It includes its own methods, results, and limitations, with no input from the parent report and no baseline-relative evaluation. It includes `results/window-curve.png`, so keep that image at its relative path. Compile the report from `reports/` with `pdflatex report-window-size-comparison-independent.tex` twice for references.

## Completed comparison

All 36 size-and-seed runs and 252 LOPO fits completed on the frozen 149-session cohort. The highest mean macro F1 was **0.693 at 120 seconds**, compared with **0.643 at 60 seconds**; balanced accuracy was 0.691 and 0.643, respectively. The mean paired macro F1 difference was +0.050, with a participant/session/seed bootstrap 95% interval of **−0.008 to +0.087**. This interval includes zero. The 20-second setting scored 0.596 macro F1, with a paired difference of −0.047 (95% interval −0.079 to −0.005) versus 60 seconds. Bus F1 rose from 0.426 at 60 seconds to 0.540 at 120 seconds; the class-level tables show the full breakdown.

Run tests with `.\.venv\Scripts\python.exe -m pytest`.
