import json
import numpy as np
import pandas as pd
import pytest

from sydney_window_comparison import study
from sydney_window_comparison.report import _macro_f1, paired_interval
from all_tmd.ingest import _is_sydney_collector_session
from all_tmd.ingest import _duration_filter
from all_tmd.study import create_lopo_manifest
from all_tmd.study_runner import load_model_lock


def test_frozen_candidate_scope_and_window_contract(tmp_path):
    candidates = study.candidate_sessions()
    assert len(candidates) == 149
    assert {row["vehicle_type"] for row in candidates.values()} == {"bus", "car", "train"}
    assert min(row["duration_seconds"] for row in candidates.values()) >= 240
    assert max(row["duration_seconds"] for row in candidates.values()) > 7200
    assert {row["participant_id"] for row in candidates.values()} == {
        "participant_001", "participant_002", "participant_004", "participant_005",
        "participant_006", "participant_008", "participant_009",
    }
    for window in study.WINDOWS:
        config = study.configuration(tmp_path, tmp_path / "work", window)
        assert config.dataset.minimum_trip_seconds == 240
        assert config.dataset.maximum_trip_seconds is None
        assert config.trial.features.default_step_seconds == window // 2
        assert config.trial.features.context_windows_seconds == (window,)
        assert config.trial.feature_names == json.loads(study.MODEL_LOCK.read_text())["feature_names"]


def test_country_sidecar_filter(tmp_path):
    raw = tmp_path / "raw" / "participant_001" / "device"
    raw.mkdir(parents=True)
    payload = raw / "abc.json.gz"
    payload.touch()
    sidecar = raw / "abc.json.gz.metadata.json"
    sidecar.write_text(json.dumps({"participant_id": "participant_001", "collection_country_code": "KR"}))
    assert not _is_sydney_collector_session(payload)
    sidecar.write_text(json.dumps({"participant_id": "participant_001", "collection_country_code": "AU"}))
    assert _is_sydney_collector_session(payload)
    sidecar.write_text(json.dumps({"participant_id": "participant_001"}))
    assert _is_sydney_collector_session(payload)


def test_effective_duration_boundary_and_no_maximum(tmp_path):
    config = study.configuration(tmp_path, tmp_path / "work")
    frame = pd.DataFrame([
        {"session_id": "short", "timestamp_ms": 0},
        {"session_id": "short", "timestamp_ms": 239_999},
        {"session_id": "boundary", "timestamp_ms": 0},
        {"session_id": "boundary", "timestamp_ms": 240_000},
        {"session_id": "long", "timestamp_ms": 0},
        {"session_id": "long", "timestamp_ms": 10_000_000},
    ])
    kept = _duration_filter(frame, config)
    assert set(kept["session_id"]) == {"boundary", "long"}


def test_event_inventory_rejects_snapshot_metadata_mismatch(tmp_path):
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    pd.DataFrame([{"session_id": "a", "participant_id": "wrong", "vehicle_type": "bus"}]).to_parquet(event_dir / "part-000000.parquet")
    candidates = {"a": {"participant_id": "participant_001", "vehicle_type": "bus"}}
    with pytest.raises(ValueError, match="differs"):
        study._event_inventory(event_dir, candidates)


def test_common_intersection_requires_all_sizes():
    available = {size: {"a", "b"} for size in study.WINDOWS}
    available[240] = {"a"}
    assert study.common_cohort({"a", "b", "c"}, available) == {"a"}
    with pytest.raises(ValueError, match="all 12"):
        study.common_cohort({"a"}, {20: {"a"}})


def test_frozen_lopo_separates_held_out_participants():
    frame = pd.DataFrame([
        {"participant_id": f"participant_{i:03d}", "session_id": f"session-{i}",
         "vehicle_type": ("bus", "car", "train")[i % 3],
         "window_start_ms": 0, "window_end_ms": 20_000}
        for i in range(1, 8)
    ])
    manifest = create_lopo_manifest(frame)
    assert len(manifest["folds"]) == 7
    for fold in manifest["folds"]:
        assert not set(fold["test_session_ids"]) & set(fold["training_pool_session_ids"])
        assert len(fold["test_session_ids"]) == 1
        assert len(fold["training_pool_session_ids"]) == 6


def test_lock_digest_rejects_modified_parameters(tmp_path):
    lock = json.loads(study.MODEL_LOCK.read_text())
    lock["params"]["n_estimators"] = 1
    bad = tmp_path / "model-lock.json"
    bad.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="digest"):
        load_model_lock(bad, lock["feature_names"])


def test_manifest_rerun_rejects_change(tmp_path):
    path = tmp_path / "cohort.json"
    study._freeze(path, {"session_ids": ["a"]})
    study._freeze(path, {"session_ids": ["a"]})
    with pytest.raises(ValueError, match="differs"):
        study._freeze(path, {"session_ids": ["b"]})


def test_paired_interval_identical_predictions_is_zero():
    rows = pd.DataFrame([
        {"session_id": "a", "label": 0, "prediction": 0},
        {"session_id": "b", "label": 1, "prediction": 2},
        {"session_id": "c", "label": 2, "prediction": 2},
    ])
    by_seed = {seed: rows for seed in study.SEEDS}
    low, high = paired_interval(by_seed, by_seed, {"p1": ["a", "b"], "p2": ["c"]}, 25, 42)
    assert (low, high) == (0, 0)
    assert _macro_f1(np.eye(3, dtype=int)) == 1
