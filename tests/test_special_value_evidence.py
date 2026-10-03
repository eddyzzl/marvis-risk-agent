"""Native masking preserves lineage and publishes dataset + receipt atomically."""

from types import SimpleNamespace

import pytest

from marvis.data.preprocessing_evidence import KIND, load_preprocessing_state
from marvis.feature.preprocessing import sidecar_path, write_preprocessing_chain
from marvis.packs.modeling import special_value_tools
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_modeling_special_values import _runtime, _sample, _sentinels


def _fixture(tmp_path):
    _, registry, settings, task = _runtime(tmp_path)
    dataset = _sample(registry, tmp_path, task.id)
    ctx = SimpleNamespace(task_id=task.id, datasets_root=settings.datasets_dir,
                          workspace=settings.workspace, seed=17)
    inputs = {"dataset_id": dataset.id, "features": ["x1", "x2"],
              "sentinel_columns": _sentinels(),
              "decisions": {"x1": {"action": "mask"}, "x2": {"action": "drop"}}}
    return registry, settings, task, dataset, ctx, inputs


@pytest.mark.parametrize("changed", ["source", "sidecar"])
def test_mask_evidence_rejects_changed_source_or_parameters(tmp_path, changed):
    registry, settings, task, dataset, ctx, inputs = _fixture(tmp_path)
    result = special_value_tools.tool_resolve_special_values(inputs, ctx)
    result_id = result["result_dataset_id"]
    assert load_preprocessing_state(registry, result_id).assurance == "row_local"
    path = (registry.resolve_path(dataset.id) if changed == "source"
            else sidecar_path(registry.resolve_path(result_id)))
    path.chmod(0o600)
    path.write_bytes(b"changed evidence")
    with pytest.raises((ValueError, RuntimeError)):
        load_preprocessing_state(registry, result_id)


def test_mask_does_not_upgrade_legacy_parent_parameters(tmp_path):
    registry, _, _, dataset, ctx, inputs = _fixture(tmp_path)
    write_preprocessing_chain(sidecar_path(registry.resolve_path(dataset.id)), [
        {"kind": "impute", "columns": ["x2"], "params": {"x2": 0.0}},
    ])
    result = special_value_tools.tool_resolve_special_values(inputs, ctx)
    state = load_preprocessing_state(registry, result["result_dataset_id"])
    assert state.artifact_id is not None
    assert state.assurance == "unknown"
    assert len(state.steps) == 2


@pytest.mark.parametrize("failure", ["receipt_registration", "source_during_transform"])
def test_mask_failure_rolls_back_all_native_artifacts(tmp_path, monkeypatch, failure):
    registry, settings, task, dataset, ctx, inputs = _fixture(tmp_path)
    before = {d.id for d in registry.list_for_task(task.id)}
    if failure == "receipt_registration":
        original = special_value_tools.register_preprocessing_evidence

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected receipt registration failure")

        monkeypatch.setattr(special_value_tools, "register_preprocessing_evidence", fail)
    else:
        original = special_value_tools._stream_mask_to_parquet

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            path = registry.resolve_path(dataset.id)
            path.chmod(0o600)
            path.write_bytes(b"source replaced during streaming")

        monkeypatch.setattr(special_value_tools, "_stream_mask_to_parquet", fail)
    with pytest.raises((ValueError, RuntimeError)):
        special_value_tools.tool_resolve_special_values(inputs, ctx)
    assert {d.id for d in registry.list_for_task(task.id)} == before
    assert not [r for r in TaskArtifactRepository(settings.db_path).list_for_task(task.id) if r["kind"] == KIND]
    destination = settings.datasets_dir / task.id / "modeling"
    assert not list(destination.rglob("*special_values*"))
    assert not list(destination.rglob("*.bak"))
    assert not list(destination.rglob(".staging"))
