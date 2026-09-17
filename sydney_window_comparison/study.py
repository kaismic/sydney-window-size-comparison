"""Run the pinned v2 collector pipeline across a fixed Sydney LOPO cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
V2 = PROJECT.parent / "all-tmd-v2"
sys.path.insert(0, str(V2))

from all_tmd.config import PipelineConfig, SourceConfig  # noqa: E402
from all_tmd.ingest import ingest_collector  # noqa: E402
from all_tmd.mlflow_utils import dataset_digest  # noqa: E402
from all_tmd.study import canonical_digest, create_lopo_manifest, verify_snapshot, verify_snapshot_files  # noqa: E402
from all_tmd.study_runner import (  # noqa: E402
    _fit_fixed_model,
    _prediction_frame,
    classification_metrics,
    load_model_lock,
    read_feature_dataset,
)
from all_tmd.windowing import _build_source_features, feature_output_dir  # noqa: E402

STUDY = json.loads((PROJECT / "study.json").read_text(encoding="utf-8"))
WINDOWS = tuple(STUDY["windows_seconds"])
SEEDS = tuple(STUDY["seeds"])
LABELS = STUDY["labels"]
SNAPSHOT = PROJECT / "manifests" / "sydney-166.json"
COHORT = PROJECT / "manifests" / "cohort.json"
LOPO = PROJECT / "manifests" / "sydney-lopo.json"
MODEL_LOCK = PROJECT / "model-lock.json"


def validate_plan() -> None:
    if WINDOWS != tuple(range(20, 241, 20)) or SEEDS != (42, 43, 44):
        raise ValueError("study grid differs from the frozen 12-size, three-seed plan")
    if STUDY["minimum_session_seconds"] != max(WINDOWS) or STUDY["maximum_session_seconds"] is not None:
        raise ValueError("session duration must be at least the largest window, with no maximum")
    if LABELS != {"bus": 0, "car": 1, "train": 2}:
        raise ValueError("only bus, car, and train are allowed")
    if STUDY["baseline_window_seconds"] != 60:
        raise ValueError("the v2 baseline window is 60 seconds")


def verify_inputs(data_dir: Path) -> dict[str, Any]:
    validate_plan()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=V2, text=True).strip()
    if commit != STUDY["v2_commit"]:
        raise ValueError(f"all-tmd-v2 checkout changed: expected {STUDY['v2_commit']}, got {commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=V2, text=True).strip():
        raise ValueError("all-tmd-v2 checkout has uncommitted changes")
    original = V2 / "manifests" / "sydney-166.json"
    if SNAPSHOT.read_bytes() != original.read_bytes():
        raise ValueError("the local snapshot differs from pinned all-tmd-v2")
    summary = verify_snapshot(SNAPSHOT)
    if summary["snapshot_digest"] != STUDY["source_snapshot_digest"]:
        raise ValueError("Sydney snapshot ID digest changed")
    verify_snapshot_files(SNAPSHOT, data_dir / "downloaded_sessions")
    return summary


def candidate_sessions() -> dict[str, dict[str, Any]]:
    rows = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["sessions"]
    selected = {
        row["session_id"]: row
        for row in rows
        if row["vehicle_type"] in LABELS
        and float(row["duration_seconds"]) >= STUDY["minimum_session_seconds"]
    }
    if len(selected) != 149:
        raise ValueError(f"expected 149 mode/duration candidates; found {len(selected)}")
    return selected


def common_cohort(candidate_ids: set[str], window_session_ids: dict[int, set[str]]) -> set[str]:
    if set(window_session_ids) != set(WINDOWS):
        raise ValueError("all 12 window sizes are required before freezing the cohort")
    common = candidate_ids.copy()
    for window in WINDOWS:
        common &= window_session_ids[window]
    return common


def configuration(data_dir: Path, work_dir: Path, window_seconds: int = 60, n_jobs: int = -1) -> PipelineConfig:
    if window_seconds not in WINDOWS:
        raise ValueError(f"unsupported window size: {window_seconds}")
    source = PipelineConfig.from_files(V2 / "model.config.yaml", V2 / "study-trial.json")
    trial_raw = json.loads(json.dumps(source.trial.raw))
    trial_raw["features"].update(
        default_window_seconds=window_seconds,
        default_step_seconds=window_seconds // 2,
        context_windows_seconds=[window_seconds],
    )
    feature_config = replace(
        source.trial.features,
        default_window_seconds=window_seconds,
        default_step_seconds=window_seconds // 2,
        context_windows_seconds=(window_seconds,),
    )
    return replace(
        source,
        sources=replace(
            source.sources,
            collector=SourceConfig(
                input_path=data_dir / "downloaded_sessions",
                include_globs=("**/*.json", "**/*.json.gz"),
            ),
        ),
        dataset=replace(
            source.dataset,
            work_dir=work_dir,
            minimum_trip_seconds=STUDY["minimum_session_seconds"],
            maximum_trip_seconds=None,
            snapshot_manifest=SNAPSHOT,
        ),
        training=replace(source.training, n_jobs=n_jobs),
        mlflow=replace(source.mlflow, enabled=False),
        trial=replace(source.trial, features=feature_config, raw=trial_raw),
    )


def _freeze(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"frozen manifest differs from the current input: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _event_inventory(event_dir: Path, candidates: dict[str, dict[str, Any]]) -> set[str]:
    observed: dict[str, tuple[str, str]] = {}
    for part in sorted(event_dir.glob("part-*.parquet")):
        frame = pd.read_parquet(part, columns=["session_id", "participant_id", "vehicle_type"])
        for session_id, group in frame.groupby("session_id", sort=False):
            values = set(zip(group["participant_id"].astype(str), group["vehicle_type"].astype(str)))
            if len(values) != 1:
                raise ValueError(f"mixed participant or mode in session {session_id}")
            pair = next(iter(values))
            if session_id in observed and observed[session_id] != pair:
                raise ValueError(f"conflicting session metadata: {session_id}")
            observed[str(session_id)] = pair
    for session_id, (participant, mode) in observed.items():
        if session_id not in candidates:
            continue
        row = candidates[session_id]
        if (participant, mode) != (row["participant_id"], row["vehicle_type"]):
            raise ValueError(f"session metadata differs from frozen snapshot: {session_id}")
    return set(observed)


def prepare(data_dir: Path, work_dir: Path, n_jobs: int = -1) -> dict[str, Any]:
    verify_inputs(data_dir)
    candidates = candidate_sessions()
    config = configuration(data_dir, work_dir, n_jobs=n_jobs)
    model = load_model_lock(MODEL_LOCK, config.trial.feature_names, validate_dependencies=True)
    event_dir = ingest_collector(config)
    observed = _event_inventory(event_dir, candidates)
    missing = set(candidates) - observed
    if missing:
        print(f"{len(missing)} snapshot candidates did not meet the effective 240-second event duration")
    available_by_window: dict[int, set[str]] = {}
    window_counts: dict[str, int] = {}
    window_sessions: dict[str, int] = {}
    for window in WINDOWS:
        current = configuration(data_dir, work_dir, window, n_jobs)
        path = _build_source_features(current, "collector", incremental=False)
        frame = read_feature_dataset(path)
        ids = set(frame["session_id"].astype(str)) & set(candidates)
        available_by_window[window] = ids
        window_counts[str(window)] = int(frame[frame["session_id"].isin(candidates)].shape[0])
        window_sessions[str(window)] = len(ids)
        print(f"window={window}s eligible_sessions={len(ids)} windows={window_counts[str(window)]}", flush=True)
    common = common_cohort(set(candidates) & observed, available_by_window)
    reference = read_feature_dataset(feature_output_dir(config, "collector"))
    reference = reference.loc[reference["session_id"].astype(str).isin(common)].copy()
    participants = sorted(reference["participant_id"].astype(str).unique())
    if len(participants) != 7:
        raise ValueError(f"common cohort lost a participant: {participants}")
    cohort = {
        "schema_version": 1,
        "source_snapshot_digest": STUDY["source_snapshot_digest"],
        "model_lock_digest": model["lock_digest"],
        "v2_commit": STUDY["v2_commit"],
        "minimum_session_seconds": 240,
        "maximum_session_seconds": None,
        "windows_seconds": list(WINDOWS),
        "candidate_session_count": len(candidates),
        "session_ids": sorted(common),
        "session_id_digest": canonical_digest(sorted(common)),
        "participants": participants,
        "mode_counts": dict(sorted(Counter(candidates[s]["vehicle_type"] for s in common).items())),
        "pre_intersection_window_counts": window_counts,
        "pre_intersection_session_counts": window_sessions,
    }
    _freeze(COHORT, cohort)
    lopo = create_lopo_manifest(reference)
    if lopo["evaluation_session_ids"] != cohort["session_ids"]:
        raise ValueError("LOPO evaluation session IDs differ from common cohort")
    _freeze(LOPO, lopo)
    print(f"frozen_common_sessions={len(common)} participants={len(participants)}")
    return cohort


def _load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    lopo = json.loads(LOPO.read_text(encoding="utf-8"))
    if canonical_digest(cohort["session_ids"]) != cohort["session_id_digest"]:
        raise ValueError("cohort digest is invalid")
    if canonical_digest({k: v for k, v in lopo.items() if k != "manifest_digest"}) != lopo["manifest_digest"]:
        raise ValueError("LOPO digest is invalid")
    if lopo["evaluation_session_ids"] != cohort["session_ids"] or len(lopo["folds"]) != 7:
        raise ValueError("LOPO manifest differs from frozen cohort")
    return cohort, lopo


def run(data_dir: Path, work_dir: Path, results_dir: Path, n_jobs: int = -1) -> None:
    verify_inputs(data_dir)
    cohort, lopo = _load_frozen()
    candidate_ids = set(cohort["session_ids"])
    base_config = configuration(data_dir, work_dir, n_jobs=n_jobs)
    lock = load_model_lock(MODEL_LOCK, base_config.trial.feature_names, validate_dependencies=True)
    if lock["lock_digest"] != cohort["model_lock_digest"]:
        raise ValueError("model lock differs from frozen cohort")
    for window in WINDOWS:
        config = configuration(data_dir, work_dir, window, n_jobs)
        frame = read_feature_dataset(feature_output_dir(config, "collector"))
        frame = frame.loc[frame["session_id"].astype(str).isin(candidate_ids)].copy()
        if set(frame["domain"].astype(str)) != {"collector"}:
            raise ValueError("non-Sydney source features entered the study")
        if set(frame["session_id"].astype(str)) != candidate_ids:
            raise ValueError(f"window {window} has lost a frozen cohort session")
        if create_lopo_manifest(frame)["manifest_digest"] != lopo["manifest_digest"]:
            raise ValueError(f"window {window} does not match the frozen LOPO split")
        feature_digest = dataset_digest(frame, config.trial.feature_names)
        for seed in SEEDS:
            destination = results_dir / "runs" / f"window-{window:03d}" / f"seed-{seed}"
            identity = {
                "window_seconds": window,
                "step_seconds": window // 2,
                "seed": seed,
                "cohort_digest": cohort["session_id_digest"],
                "lopo_digest": lopo["manifest_digest"],
                "model_lock_digest": lock["lock_digest"],
                "feature_digest": feature_digest,
                "v2_commit": STUDY["v2_commit"],
            }
            if (destination / "_SUCCESS").exists():
                report = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                if report["identity"] != identity:
                    raise ValueError(f"completed run has different inputs: {destination}")
                print(f"already complete: window={window} seed={seed}", flush=True)
                continue
            if destination.exists():
                raise ValueError(f"incomplete run requires review: {destination}")
            destination.mkdir(parents=True)
            fold_reports = []
            predictions = []
            for fold in lopo["folds"]:
                held_out = fold["held_out_participant_id"]
                train_ids = set(fold["training_pool_session_ids"])
                test_ids = set(fold["test_session_ids"])
                train = frame.loc[frame["session_id"].astype(str).isin(train_ids)]
                test = frame.loc[frame["session_id"].astype(str).isin(test_ids)]
                if set(train["participant_id"].astype(str)) & {held_out}:
                    raise ValueError("held-out participant leaked into training")
                if set(test["participant_id"].astype(str)) != {held_out}:
                    raise ValueError("LOPO test rows contain another participant")
                if set(train["label"].astype(int)) != {0, 1, 2} or test.empty:
                    raise ValueError(f"invalid train/test class coverage for {held_out}")
                model = _fit_fixed_model(config, lock["params"], train, seed)
                prediction = _prediction_frame(model, test, config.trial.feature_names)
                predictions.append(prediction)
                fold_reports.append({
                    "held_out_participant_id": held_out,
                    "training_session_digest": fold["training_pool_digest"],
                    "test_session_digest": fold["test_digest"],
                    "training_windows": len(train),
                    "test_windows": len(test),
                    "metrics": classification_metrics(prediction, LABELS),
                })
                print(f"window={window} seed={seed} fold={held_out} test_windows={len(test)}", flush=True)
            pooled = pd.concat(predictions, ignore_index=True)
            if set(pooled["session_id"].astype(str)) != candidate_ids:
                raise ValueError("pooled predictions omit frozen cohort sessions")
            report = {"schema_version": 1, "identity": identity, "folds": fold_reports,
                      "metrics": classification_metrics(pooled, LABELS)}
            pooled.to_parquet(destination / "predictions.parquet", index=False)
            (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (destination / "_SUCCESS").write_text("", encoding="utf-8")


def _data_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    environment = os.environ.get("ALL_TMD_DATA_DIR")
    if environment:
        return Path(environment).expanduser().resolve()
    env_path = V2 / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ALL_TMD_DATA_DIR="):
                return Path(line.split("=", 1)[1].strip().strip('"')).expanduser().resolve()
    raise ValueError("set ALL_TMD_DATA_DIR or pass --data-dir")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "prepare", "run", "report", "all"))
    parser.add_argument("--data-dir", help="parent of downloaded_sessions")
    parser.add_argument("--work-dir", type=Path, default=PROJECT / "work")
    parser.add_argument("--results-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    data_dir = _data_dir(args.data_dir)
    if args.command == "verify":
        summary = verify_inputs(data_dir)
        print(json.dumps({
            "snapshot_digest": summary["snapshot_digest"],
            "snapshot_sessions": summary["session_count"],
            "three_class_sessions": summary["eligible_session_count"],
            "duration_candidates": len(candidate_sessions()),
            "participants": summary["participant_count"],
        }, indent=2))
    if args.command in ("prepare", "all"):
        prepare(data_dir, args.work_dir, args.n_jobs)
    if args.command in ("run", "all"):
        run(data_dir, args.work_dir, args.results_dir, args.n_jobs)
    if args.command in ("report", "all"):
        from .report import report
        report(args.results_dir)


if __name__ == "__main__":
    main()
