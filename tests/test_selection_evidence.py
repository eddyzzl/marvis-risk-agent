"""Real tools preserve the selection basis without promoting CV independence."""

import base64
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Event, current_thread
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marvis.feature.errors import FeatureError
from marvis.packs.modeling.selection_evidence import KIND, load_selection_evidence
from marvis.plugins.manifest import ToolRef
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_feature_pack import _runtime


def _source(tmp_path, *, partitions=("train", "test", "oot"), target_type="binary"):
    runner, registry, repo, _ = _runtime(tmp_path)
    rng = np.random.default_rng(21)
    count = 40 * len(partitions)
    frame = pd.DataFrame({
        "split": np.repeat(partitions, 40),
        "x": rng.normal(size=count), "z": rng.normal(size=count),
        "y": np.arange(count) % 2 if target_type == "binary" else rng.normal(size=count),
    })
    path = tmp_path / "selection.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")
    return runner, registry, repo, dataset


def _run(runner, dataset, tool="screen_features", **kwargs):
    return runner.invoke(ToolRef("modeling", tool), {
        "dataset_id": dataset.id, "features": ["x", "z"], "target_col": "y",
        "split_col": "split", **({"seed": 3} if tool == "select_features" else {}), **kwargs,
    }, task_id="task-feature")


def _mask(payload, role):
    raw = base64.b64decode(payload["memberships"][role]["membership"])
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little", count=payload["row_count"]).astype(bool)


def test_screen_receipt_distinguishes_fit_labels_from_exposed_diagnostics(tmp_path):
    runner, registry, _, dataset = _source(tmp_path)
    result = _run(runner, dataset, leakage_ks=1.0)
    assert result.ok, result.error
    payload = load_selection_evidence(registry, "task-feature", result.output["selection_evidence_ref"])
    assert np.array_equal(np.flatnonzero(_mask(payload, "fit")), np.arange(40))
    assert np.array_equal(np.flatnonzero(_mask(payload, "label_diagnostics")), np.arange(80))
    assert np.array_equal(np.flatnonzero(_mask(payload, "value_diagnostics")), np.r_[0:40, 80:120])
    assert payload["selected"] == result.output["selected"]
    assert payload["candidates"] == ["x", "z"]
    assert payload["parameters"]["batch_size"] == 16
    assert payload["parameters"]["leakage_ks"] == 1.0
    assert payload["independent_inner_validation"] == "not_established"
    assert payload["upstream_selection"] == payload["historical_availability"] == "unknown"
    assert set(result.output["selection_evidence_ref"]) == {"artifact_id", "content_hash"}


def test_nonstandard_screen_records_actual_legacy_fallback_members(tmp_path):
    runner, registry, _, dataset = _source(tmp_path, partitions=("development", "test", "oot"))
    result = _run(runner, dataset, leakage_ks=1.0)
    assert result.ok, result.error
    payload = load_selection_evidence(registry, "task-feature", result.output["selection_evidence_ref"])
    assert np.array_equal(np.flatnonzero(_mask(payload, "fit")), np.arange(80))
    assert not _mask(payload, "label_diagnostics").any()


def test_numeric_selection_records_original_positions_and_no_diagnostic_members(tmp_path):
    runner, registry, _, dataset = _source(tmp_path, partitions=(1, 0, 2))
    result = _run(runner, dataset, "select_features", split_value=0.0, holdout_values=["1", "2"], iv_min=0.0)
    assert result.ok, result.error
    payload = load_selection_evidence(registry, "task-feature", result.output["selection_evidence_ref"])
    assert np.array_equal(np.flatnonzero(_mask(payload, "fit")), np.arange(40, 80))
    assert not _mask(payload, "label_diagnostics").any()
    assert not _mask(payload, "value_diagnostics").any()
    assert payload["parameters"]["multivariate_sample_rows"] == 50_000


def test_continuous_screen_has_native_training_members_without_binary_diagnostics(tmp_path):
    runner, registry, _, dataset = _source(tmp_path, target_type="continuous")
    result = _run(runner, dataset, target_type="continuous")
    assert result.ok, result.error
    payload = load_selection_evidence(registry, "task-feature", result.output["selection_evidence_ref"])
    assert np.array_equal(np.flatnonzero(_mask(payload, "fit")), np.arange(40))
    assert not _mask(payload, "label_diagnostics").any()


@pytest.mark.parametrize("oot_varies", [False, True])
def test_auxiliary_oot_categorical_hint_is_not_certified_by_empty_core_diagnostics(tmp_path, oot_varies):
    runner, registry, _, _ = _source(tmp_path)
    frame = pd.DataFrame({
        "x": [1.0] * 120, "y": [0, 1] * 60,
        "region_code": [0] * 40 + [1] * 40 + (list(range(40)) if oot_varies else [1] * 40),
        "split": ["train"] * 40 + ["test"] * 40 + ["oot"] * 40,
    })
    path = tmp_path / "hints.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")
    result = _run(runner, dataset, features=["x"])
    assert result.ok, result.error
    assert bool(result.output.get("suspected_categorical")) is not oot_varies
    payload = load_selection_evidence(registry, "task-feature", result.output["selection_evidence_ref"])
    assert not _mask(payload, "value_diagnostics").any()
    assert not _mask(payload, "label_diagnostics").any()
    assert payload["auxiliary_diagnostics"]["assurance"] == "unknown"
    assert "categorical_hints" in payload["auxiliary_diagnostics"]["excluded"]
    assert payload["independent_inner_validation"] == "not_established"


@pytest.mark.parametrize("change", ["reference", "receipt", "source", "task"])
def test_changed_selection_binding_cannot_be_loaded(tmp_path, change):
    runner, registry, repo, dataset = _source(tmp_path)
    result = _run(runner, dataset, "select_features", iv_min=0.0)
    assert result.ok, result.error
    ref = dict(result.output["selection_evidence_ref"])
    task_id = "task-feature"
    if change == "reference":
        ref["content_hash"] = "a" * 64
    elif change == "receipt":
        record = TaskArtifactRepository(repo.db_path).get_for_task(task_id, ref["artifact_id"])
        path = registry.datasets_root.parent / record["path"]
        path.write_text(path.read_text().replace('"independent_inner_validation": "not_established"', '"independent_inner_validation": "verified"'))
    elif change == "source":
        path = registry.resolve_path(dataset.id)
        path.chmod(0o600)
        path.write_bytes(b"changed source")
    else:
        task_id = "another-task"
    with pytest.raises((FeatureError, ValueError, RuntimeError)):
        load_selection_evidence(registry, task_id, ref)


def test_source_replacement_during_selection_publishes_no_receipt(tmp_path, monkeypatch):
    from marvis.packs.modeling import feature_tools

    runner, registry, repo, dataset = _source(tmp_path)
    original = feature_tools.select_features

    @wraps(original)
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = registry.resolve_path(dataset.id)
        path.chmod(0o600)
        path.write_bytes(b"concurrent replacement")
        return result

    monkeypatch.setattr(feature_tools, "select_features", changed)
    ctx = SimpleNamespace(task_id="task-feature", datasets_root=registry.datasets_root,
                          workspace=registry.datasets_root.parent, seed=3)
    with pytest.raises(RuntimeError, match="changed"):
        feature_tools.tool_select_features({
            "dataset_id": dataset.id, "features": ["x", "z"], "target_col": "y",
            "split_col": "split", "iv_min": 0.0,
        }, ctx)
    assert not [r for r in TaskArtifactRepository(repo.db_path).list_for_task("task-feature") if r["kind"] == KIND]


def test_column_batches_share_one_private_source_snapshot(tmp_path, monkeypatch):
    from marvis.data import authenticated_snapshot
    from marvis.packs.modeling import feature_tools

    runner, registry, _, dataset = _source(tmp_path)
    original = authenticated_snapshot.tempfile.TemporaryFile
    snapshots = []

    def created(*args, **kwargs):
        value = original(*args, **kwargs)
        snapshots.append(value)
        return value

    monkeypatch.setattr(authenticated_snapshot.tempfile, "TemporaryFile", created)
    ctx = SimpleNamespace(task_id="task-feature", datasets_root=registry.datasets_root,
                          workspace=registry.datasets_root.parent, seed=3)
    result = feature_tools.tool_screen_features({
        "dataset_id": dataset.id, "features": ["x", "z"], "target_col": "y",
        "split_col": "split", "batch_size": 1, "leakage_ks": 1.0,
    }, ctx)
    assert len(snapshots) == 1
    assert snapshots[0].closed
    assert load_selection_evidence(registry, "task-feature", result["selection_evidence_ref"])["row_count"] == 120


def test_exact_selection_replay_reuses_receipt_and_cannot_repair_tampering(tmp_path):
    runner, registry, repo, dataset = _source(tmp_path)
    first = _run(runner, dataset, "select_features", iv_min=0.0)
    second = _run(runner, dataset, "select_features", iv_min=0.0)
    assert first.ok and second.ok
    assert first.output["selection_evidence_ref"] == second.output["selection_evidence_ref"]
    records = [r for r in TaskArtifactRepository(repo.db_path).list_for_task("task-feature") if r["kind"] == KIND]
    assert len(records) == 1
    path = registry.datasets_root.parent / records[0]["path"]
    path.write_bytes(b"tampered receipt")
    repeated = _run(runner, dataset, "select_features", iv_min=0.0)
    assert not repeated.ok
    assert path.read_bytes() == b"tampered receipt"


@pytest.mark.parametrize("first_fails", [False, True])
def test_concurrent_selection_publication_preserves_committed_receipt(tmp_path, monkeypatch, first_fails):
    from marvis.artifacts import ArtifactUnitOfWork
    from marvis.packs.modeling import feature_tools, selection_evidence

    _, registry, repo, dataset = _source(tmp_path)
    ctx = SimpleNamespace(task_id="task-feature", datasets_root=registry.datasets_root,
                          workspace=registry.datasets_root.parent, seed=3)
    inputs = {"dataset_id": dataset.id, "features": ["x", "z"], "target_col": "y",
              "split_col": "split", "iv_min": 0.0, "seed": 3}
    promoted, contender, release = Event(), Event(), Event()
    finalize = ArtifactUnitOfWork.finalize_with_connection
    acquire = selection_evidence.FileLock.acquire

    def observe_acquire(self, *args, **kwargs):
        if current_thread().name.startswith("contender"):
            contender.set()
        return acquire(self, *args, **kwargs)

    def pause_after_promotion(self, factory, callback):
        if current_thread().name.startswith("publisher"):
            self.promote_all()
            promoted.set()
            assert release.wait(10)
            if first_fails:
                raise RuntimeError("injected registration failure")
        return finalize(self, factory, callback)

    monkeypatch.setattr(selection_evidence.FileLock, "acquire", observe_acquire)
    monkeypatch.setattr(ArtifactUnitOfWork, "finalize_with_connection", pause_after_promotion)
    with ThreadPoolExecutor(1, thread_name_prefix="publisher") as first_pool, \
            ThreadPoolExecutor(1, thread_name_prefix="contender") as second_pool:
        first = first_pool.submit(feature_tools.tool_select_features, inputs, ctx)
        try:
            assert promoted.wait(10)
            second = second_pool.submit(feature_tools.tool_select_features, inputs, ctx)
            assert contender.wait(10)
            assert not second.done()
        finally:
            release.set()
        if first_fails:
            with pytest.raises(RuntimeError, match="injected registration failure"):
                first.result(timeout=10)
        else:
            first_result = first.result(timeout=10)
        second_result = second.result(timeout=10)
    reference = second_result["selection_evidence_ref"]
    assert load_selection_evidence(registry, "task-feature", reference)["selected"] == second_result["selected"]
    if not first_fails:
        assert first_result["selection_evidence_ref"] == reference
    records = [r for r in TaskArtifactRepository(repo.db_path).list_for_task("task-feature") if r["kind"] == KIND]
    assert len(records) == 1
    assert not list((registry.datasets_root / "task-feature" / "selection").rglob("*.bak"))
    assert not list((registry.datasets_root / "task-feature" / "selection").rglob(".staging"))
