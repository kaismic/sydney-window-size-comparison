"""Summarize complete pooled-window LOPO predictions and paired size differences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from all_tmd.study import canonical_digest
from all_tmd.study_runner import classification_metrics, load_model_lock

from .study import COHORT, LABELS, LOPO, MODEL_LOCK, SEEDS, STUDY, WINDOWS, _load_frozen


def _macro_f1(matrix: np.ndarray) -> float:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    denominator = support + predicted
    return float(np.mean(np.divide(2 * diagonal, denominator,
                                   out=np.zeros(3, dtype=float), where=denominator != 0)))


def _per_session_matrices(predictions: pd.DataFrame, sessions: list[str]) -> np.ndarray:
    grouped = {str(key): group for key, group in predictions.groupby("session_id")}
    if set(grouped) != set(sessions):
        raise ValueError("predictions have missing or unexpected sessions")
    return np.stack([
        confusion_matrix(grouped[s]["label"], grouped[s]["prediction"], labels=[0, 1, 2])
        for s in sessions
    ])


def paired_interval(
    candidate: dict[int, pd.DataFrame],
    baseline: dict[int, pd.DataFrame],
    participant_sessions: dict[str, list[str]],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Resample seeds, then participants and their sessions using shared draws."""
    sessions = [s for ids in participant_sessions.values() for s in ids]
    indices = {s: i for i, s in enumerate(sessions)}
    by_participant = [np.array([indices[s] for s in ids]) for ids in participant_sessions.values()]
    candidate_matrices = np.stack([_per_session_matrices(candidate[s], sessions) for s in SEEDS])
    baseline_matrices = np.stack([_per_session_matrices(baseline[s], sessions) for s in SEEDS])
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations)
    for iteration in range(iterations):
        selected_seeds = rng.integers(0, len(SEEDS), len(SEEDS))
        chosen_participants = rng.integers(0, len(by_participant), len(by_participant))
        sampled = np.concatenate([
            rng.choice(by_participant[p], size=len(by_participant[p]), replace=True)
            for p in chosen_participants
        ])
        differences[iteration] = float(np.mean([
            _macro_f1(candidate_matrices[seed_index, sampled].sum(axis=0))
            - _macro_f1(baseline_matrices[seed_index, sampled].sum(axis=0))
            for seed_index in selected_seeds
        ]))
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(lower), float(upper)


def _completed_runs(results_dir: Path, cohort: dict[str, Any], lopo: dict[str, Any]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], pd.DataFrame]]:
    runs = {}
    predictions = {}
    expected_ids = set(cohort["session_ids"])
    for window in WINDOWS:
        for seed in SEEDS:
            folder = results_dir / "runs" / f"window-{window:03d}" / f"seed-{seed}"
            if not (folder / "_SUCCESS").exists():
                raise FileNotFoundError(f"incomplete study run: {folder}")
            run = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            identity = run["identity"]
            if (identity["window_seconds"], identity["step_seconds"], identity["seed"]) != (window, window // 2, seed):
                raise ValueError(f"wrong study grid identity: {folder}")
            if identity["cohort_digest"] != cohort["session_id_digest"] or identity["lopo_digest"] != lopo["manifest_digest"]:
                raise ValueError(f"run has different frozen cohort or split: {folder}")
            if identity["model_lock_digest"] != cohort["model_lock_digest"] or identity["v2_commit"] != STUDY["v2_commit"]:
                raise ValueError(f"run has different model or code: {folder}")
            if len(run["folds"]) != 7 or {f["held_out_participant_id"] for f in run["folds"]} != set(cohort["participants"]):
                raise ValueError(f"run does not have seven LOPO folds: {folder}")
            frame = pd.read_parquet(folder / "predictions.parquet")
            if set(frame["session_id"].astype(str)) != expected_ids:
                raise ValueError(f"run has different evaluated sessions: {folder}")
            if classification_metrics(frame, LABELS) != run["metrics"]:
                raise ValueError(f"recorded metrics differ from predictions: {folder}")
            runs[(window, seed)] = run
            predictions[(window, seed)] = frame
    return runs, predictions


def report(results_dir: Path) -> dict[str, Any]:
    cohort, lopo = _load_frozen()
    lock_raw = json.loads(MODEL_LOCK.read_text(encoding="utf-8"))
    lock = load_model_lock(MODEL_LOCK, lock_raw["feature_names"])
    if lock["lock_digest"] != cohort["model_lock_digest"]:
        raise ValueError("model lock differs from frozen cohort")
    runs, predictions = _completed_runs(results_dir, cohort, lopo)
    baseline_window = STUDY["baseline_window_seconds"]
    baseline = {seed: predictions[(baseline_window, seed)] for seed in SEEDS}
    snapshot = json.loads((COHORT.parent / "sydney-166.json").read_text(encoding="utf-8"))
    participant_by_session = {row["session_id"]: row["participant_id"] for row in snapshot["sessions"]}
    participant_sessions = {
        participant: [session for session in cohort["session_ids"] if participant_by_session[session] == participant]
        for participant in cohort["participants"]
    }
    curve = []
    per_class = []
    paired = []
    for window in WINDOWS:
        window_runs = [runs[(window, seed)] for seed in SEEDS]
        scores = [run["metrics"]["macro_f1"] for run in window_runs]
        balanced = [run["metrics"]["balanced_accuracy"] for run in window_runs]
        curve.append({
            "window_seconds": window,
            "step_seconds": window // 2,
            "macro_f1_mean": float(np.mean(scores)),
            "macro_f1_sd": float(np.std(scores, ddof=1)),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_sd": float(np.std(balanced, ddof=1)),
            "windows_mean": float(np.mean([run["metrics"]["rows"] for run in window_runs])),
            "sessions": len(cohort["session_ids"]),
        })
        for mode in LABELS:
            per_class.append({
                "window_seconds": window,
                "mode": mode,
                "f1_mean": float(np.mean([run["metrics"]["per_class"][mode]["f1"] for run in window_runs])),
                "recall_mean": float(np.mean([run["metrics"]["per_class"][mode]["recall"] for run in window_runs])),
                "support_mean": float(np.mean([run["metrics"]["per_class"][mode]["support"] for run in window_runs])),
            })
        differences = [runs[(window, seed)]["metrics"]["macro_f1"] - runs[(baseline_window, seed)]["metrics"]["macro_f1"] for seed in SEEDS]
        lower, upper = (0.0, 0.0) if window == baseline_window else paired_interval(
            {seed: predictions[(window, seed)] for seed in SEEDS}, baseline,
            participant_sessions, STUDY["bootstrap_iterations"], STUDY["analysis_seed"] + window,
        )
        paired.append({"window_seconds": window, "baseline_window_seconds": baseline_window,
                       "macro_f1_difference_mean": float(np.mean(differences)),
                       "ci_95_lower": lower, "ci_95_upper": upper})
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve).to_csv(results_dir / "window-curve.csv", index=False)
    pd.DataFrame(per_class).to_csv(results_dir / "per-class.csv", index=False)
    pd.DataFrame(paired).to_csv(results_dir / "paired-vs-60s.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row["window_seconds"] for row in curve], [row["macro_f1_mean"] for row in curve], marker="o", label="Macro F1")
    axis.plot([row["window_seconds"] for row in curve], [row["balanced_accuracy_mean"] for row in curve], marker="s", label="Balanced accuracy")
    axis.axvline(baseline_window, color="gray", linestyle="--", linewidth=1)
    axis.set(xlabel="Window size (seconds)", ylabel="Pooled held-out window score", xticks=list(WINDOWS), ylim=(0, 1))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "window-curve.png", dpi=180)
    plt.close(figure)
    summary = {
        "schema_version": 1,
        "complete": True,
        "run_count": len(runs),
        "fit_count": sum(len(run["folds"]) for run in runs.values()),
        "participant_count": len(participant_sessions),
        "session_count": len(cohort["session_ids"]),
        "source_snapshot_digest": STUDY["source_snapshot_digest"],
        "cohort_digest": cohort["session_id_digest"],
        "lopo_digest": lopo["manifest_digest"],
        "model_lock_digest": lock["lock_digest"],
        "v2_commit": STUDY["v2_commit"],
        "grid_digest": canonical_digest([run["identity"] for run in runs.values()]),
        "best_macro_f1_window_seconds": max(curve, key=lambda row: row["macro_f1_mean"])["window_seconds"],
        "curve": curve,
        "paired_vs_60s": paired,
        "bootstrap_iterations": STUDY["bootstrap_iterations"],
    }
    (results_dir / "study-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("run_count", "fit_count", "session_count", "best_macro_f1_window_seconds")}, indent=2))
    return summary
