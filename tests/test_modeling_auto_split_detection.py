"""Auto detection admits complete source memberships before any derivation."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from marvis.agent.modeling_setup import ModelingSetupError, build_modeling_proposal
from marvis.agent.sample_setup import detect_setup
from tests.test_modeling_existing_split_contract import _frame
from tests.test_modeling_prepare import _runtime


def _dataset_bindings(registry, task_id):
    return {(row.id, row.role, row.row_count, row.content_hash) for row in registry.list_for_task(task_id)}


def _register_source(root, frame):
    backend, registry = _runtime(root)
    source_dir = root / "source"
    source_dir.mkdir()
    path = source_dir / "sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_from_upload("task-1", path, role="sample", seed=0)
    registry.authenticate_dataset_binding(dataset.id, expected_task_id=dataset.task_id,
                                           expected_content_hash=dataset.content_hash)
    return backend, registry, dataset


@pytest.mark.parametrize("dtype", ["str", "string", "category", "object"])
def test_auto_detect_nondefault_string_partition_preserves_authenticated_source(tmp_path, dtype):
    source = _frame("cohort", ("Development", "holdout", "time_oot"))
    source["cohort"] = source["cohort"].astype(dtype)
    backend, registry, dataset = _register_source(tmp_path, source)
    path = registry.resolve_verified_path(dataset.id)
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before_datasets = _dataset_bindings(registry, dataset.task_id)
    proposal = build_modeling_proposal(registry, backend, dataset.task_id, tmp_path / "source",
                                      target_col="y", anchor_id=dataset.id, recipes=["lr"], n_trials=1)
    assert proposal.dataset_id == dataset.id
    assert proposal.split_col == "cohort"
    assert proposal.split_values == {"train": "Development", "test": "holdout", "oot": "time_oot"}
    assert proposal.counts == {"train": 360, "test": 120, "oot": 120}
    assert proposal.holdout_values == ["time_oot"]
    assert _dataset_bindings(registry, dataset.task_id) == before_datasets
    result = backend.read_frame(registry.resolve_verified_path(proposal.dataset_id))
    for label in proposal.split_values.values():
        assert result.loc[result.cohort == label, "signal"].tolist() == source.loc[source.cohort == label, "signal"].tolist()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash


@pytest.mark.parametrize("variant", ["unknown", "null", "duplicate_role", "multiple", "named_invalid"])
def test_suspicious_or_ambiguous_auto_partition_refuses_before_derivation(tmp_path, variant):
    source = _frame("cohort", ("Development", "holdout", "time_oot"))
    if variant == "multiple":
        source["alternative"] = source.cohort.iloc[::-1].to_numpy()
    elif variant == "named_invalid":
        source["cohort"] = "unclassified"
    else:
        source.loc[599, "cohort"] = {"unknown": "future", "null": None, "duplicate_role": "dev"}[variant]
    backend, registry, dataset = _register_source(tmp_path, source)
    before_datasets = _dataset_bindings(registry, dataset.task_id)
    before_files = set(tmp_path.rglob("*.parquet"))
    with pytest.raises(ModelingSetupError, match="切分列"):
        build_modeling_proposal(registry, backend, dataset.task_id, tmp_path / "source",
                                target_col="y", anchor_id=dataset.id, recipes=["lr"], n_trials=1)
    assert _dataset_bindings(registry, dataset.task_id) == before_datasets
    assert set(tmp_path.rglob("*.parquet")) == before_files


def test_explicit_binding_resolves_multiple_auto_candidates_without_memory_guess(tmp_path):
    source = _frame("cohort", ("Development", "holdout", "time_oot"))
    source["alternative"] = source.cohort.iloc[::-1].to_numpy()
    backend, registry, dataset = _register_source(tmp_path, source)
    path = registry.resolve_verified_path(dataset.id)
    ambiguous = detect_setup(backend, path, configured_target="y", field_hints={"split_col": "cohort"})
    assert ambiguous.split_col is None and ambiguous.split_error
    assert "split_col" not in ambiguous.memory_matched_fields
    proposal = build_modeling_proposal(registry, backend, dataset.task_id, tmp_path / "source",
                                      target_col="y", split_col="cohort", anchor_id=dataset.id, recipes=["lr"], n_trials=1)
    assert proposal.dataset_id == dataset.id
    assert proposal.counts == {"train": 360, "test": 120, "oot": 120}


def test_auto_partition_checks_unknown_member_outside_dtype_probe(tmp_path, monkeypatch):
    source = pd.DataFrame({"signal": range(70008), "y": [0, 1] * 35004,
                           "cohort": ["Development"] * 40000 + ["holdout"] * 15000 + ["time_oot"] * 15008})
    source.loc[69000, "cohort"] = "future"
    backend, registry, dataset = _register_source(tmp_path, source)
    path = registry.resolve_verified_path(dataset.id)
    monkeypatch.setattr(backend, "sample_rows", lambda *_args, **_kwargs: source.iloc[:4000].copy())
    detected = detect_setup(backend, path, configured_target="y")
    assert detected.split_col is None and detected.split_error
    before_datasets = _dataset_bindings(registry, dataset.task_id)
    with pytest.raises(ModelingSetupError, match="完整成员"):
        build_modeling_proposal(registry, backend, dataset.task_id, tmp_path / "source",
                                target_col="y", anchor_id=dataset.id, recipes=["lr"], n_trials=1)
    assert _dataset_bindings(registry, dataset.task_id) == before_datasets
