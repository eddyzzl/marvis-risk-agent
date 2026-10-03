from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from marvis.data.dataset_identity import dataset_identity_equal
from marvis.data.workspace import DataWorkspaceDraft
from marvis.db import PluginRepository
from marvis.packs.modeling import monitor_tools
from marvis.packs.modeling import scoring
from marvis.packs.modeling._runtime import _artifact_base_dir
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.monitor_binding import (
    capture_monitoring_binding, validate_monitoring_binding,
)
from marvis.plugins.manifest import ToolRef
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.modeling import ModelingRepository

from test_modeling_monitor import _train_lr_experiment
from test_modeling_pack import _runtime


@pytest.fixture
def scenario(tmp_path):
    runner, _, registry, _, settings, task = _runtime(tmp_path)
    trained, frame = _train_lr_experiment(runner, registry, tmp_path, task)
    experiment_id = trained.output["experiment_id"]
    selected = runner.invoke(ToolRef("modeling", "select_experiment"), {
        "experiment_ids": [experiment_id], "selected_experiment_id": experiment_id,
        "target_type": "binary", "refit_on_train_plus_test": False,
    }, task_id=task.id)
    assert selected.ok, selected.error
    path = tmp_path / "monitor-period.parquet"
    frame.drop(columns=["split"]).to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="monitoring.input")
    binding = capture_monitoring_binding(
        settings, task.id, experiment_id, dataset.id, dataset.content_hash,
    )
    return SimpleNamespace(
        runner=runner, registry=registry, settings=settings, task=task, dataset=dataset,
        experiment_id=experiment_id, binding=binding,
        ctx=SimpleNamespace(workspace=settings.workspace, datasets_root=settings.datasets_dir,
                            task_id=task.id),
    )


def _score(scenario, **extra):
    s = scenario
    return s.runner.invoke(ToolRef("modeling", "score_dataset"), {
        "experiment_id": s.experiment_id, "dataset_id": s.dataset.id,
        "monitoring_binding": s.binding, **extra,
    }, task_id=s.task.id)


def test_binding_reads_are_pure_and_both_monitor_paths_execute(scenario):
    s = scenario
    with sqlite3.connect(s.settings.db_path) as conn:
        before = list(conn.iterdump())
    validate_monitoring_binding(s.settings, s.task.id, s.binding)
    json.dumps(s.binding, allow_nan=False)
    with sqlite3.connect(s.settings.db_path) as conn:
        assert list(conn.iterdump()) == before
    assert dataset_identity_equal(s.registry.get(s.dataset.id), s.dataset)
    scored = _score(s)
    assert scored.ok, scored.error
    for dataset_input in ({"dataset_id": s.dataset.id}, {
        "scored_dataset_id": scored.output["result_dataset_id"],
        "score_col": scored.output["score_col"],
    }):
        result = s.runner.invoke(ToolRef("modeling", "monitor_run"), {
            "experiment_id": s.experiment_id, "monitoring_binding": s.binding,
            "target_col": "y", **dataset_input,
        }, task_id=s.task.id)
        assert result.ok, result.error
        assert result.output["row_count"] == s.dataset.row_count
        assert len(result.output["checks"]) == 4
        assert result.output["label_mode"] == "declared_label_column"
        assert result.output["label_maturity_assurance"] == "unknown"
        assert result.output["business_acceptance"] == "not_established"
    audits = PluginRepository(s.settings.db_path).list_audit(kind="modeling.monitor.run")
    assert len(audits) == 2
    for row in audits:
        assert row["detail"]["baseline_sha256"] == s.binding["baseline_sha256"]
        assert row["detail"]["source_dataset_id"] == s.dataset.id
        assert row["detail"]["source_content_hash"] == s.dataset.content_hash


@pytest.mark.parametrize("mutation", ["status", "model_bytes", "baseline", "metrics", "dataset_bytes", "source_path"])
def test_binding_rejects_each_identity_mutation_before_publish(scenario, mutation):
    s = scenario
    if mutation == "status":
        ExperimentStore(s.settings.db_path).set_status(s.experiment_id, "trained")
    elif mutation == "model_bytes":
        artifact = ModelingRepository(s.settings.db_path).get_model_artifact(s.binding["artifact_id"])
        path = _artifact_base_dir(s.settings, s.task.id) / artifact.model_path
        path.write_bytes(path.read_bytes() + b"changed")
    elif mutation in {"baseline", "metrics", "source_path"}:
        with sqlite3.connect(s.settings.db_path) as conn:
            if mutation == "baseline":
                baseline = json.loads(conn.execute(
                    "SELECT baseline_distributions_json FROM model_artifacts WHERE id=?",
                    (s.binding["artifact_id"],),
                ).fetchone()[0])
                baseline["score_edges"][1] += 0.01
                conn.execute("UPDATE model_artifacts SET baseline_distributions_json=? WHERE id=?",
                             (json.dumps(baseline), s.binding["artifact_id"]))
            elif mutation == "metrics":
                conn.execute("UPDATE experiments SET metrics_json=json_set(metrics_json,'$.train_auc',0.1) WHERE id=?", (s.experiment_id,))
            else:
                original = s.registry.resolve_path(s.dataset.id)
                copy = original.with_name("same-bytes.parquet")
                copy.write_bytes(original.read_bytes())
                conn.execute("UPDATE datasets SET source_path=? WHERE id=?", (str(copy.relative_to(s.settings.datasets_dir)), s.dataset.id))
    else:
        path = s.registry.resolve_path(s.dataset.id)
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ModelingError):
        validate_monitoring_binding(s.settings, s.task.id, s.binding)
    before = DatasetRepository(s.settings.db_path).list_datasets(s.task.id)
    result = _score(s)
    assert not result.ok
    assert DatasetRepository(s.settings.db_path).list_datasets(s.task.id) == before


def test_unrelated_or_legacy_scored_child_cannot_claim_frozen_binding(scenario):
    s = scenario
    legacy = s.runner.invoke(ToolRef("modeling", "score_dataset"), {
        "experiment_id": s.experiment_id, "dataset_id": s.dataset.id,
    }, task_id=s.task.id)
    assert legacy.ok, legacy.error
    with pytest.raises(ModelingError, match="scored_source_mismatch"):
        validate_monitoring_binding(s.settings, s.task.id, s.binding,
                                    scored_dataset_id=legacy.output["result_dataset_id"])
    scored = _score(s)
    assert scored.ok, scored.error
    wrong_column = s.runner.invoke(ToolRef("modeling", "monitor_run"), {
        "experiment_id": s.experiment_id, "monitoring_binding": s.binding,
        "scored_dataset_id": scored.output["result_dataset_id"], "score_col": "x1",
    }, task_id=s.task.id)
    assert not wrong_column.ok
    assert "scored_column_mismatch" in str(wrong_column.error)


@pytest.mark.parametrize("mutation", ["workspace", "model"])
def test_change_during_scoring_publishes_no_child(scenario, monkeypatch, mutation):
    s = scenario
    repo = DataWorkspaceRepository(s.settings.db_path)
    snapshot = repo.save_initial_binding(s.task.id, DataWorkspaceDraft(
        active_dataset_id=s.dataset.id, active_dataset_content_hash=s.dataset.content_hash,
    ), expected_revision=0)
    s.binding["workspace_binding"] = {name: getattr(snapshot, name) for name in (
        "revision", "analysis_generation", "active_dataset_id", "active_dataset_content_hash",
    )}
    original_score = monitor_tools._ModelArtifactScorer.score

    def mutate_workspace(self, frame):
        result = original_score(self, frame)
        if mutation == "workspace":
            changed = repo.save(s.task.id, DataWorkspaceDraft(
                active_dataset_id=s.dataset.id, active_dataset_content_hash=s.dataset.content_hash,
                selected_field="x1",
            ), expected_revision=snapshot.revision)
            assert changed.revision > snapshot.revision
        else:
            model_path = _artifact_base_dir(s.settings, s.task.id) / self.artifact.model_path
            model_path.write_bytes(model_path.read_bytes() + b"changed after scoring")
        return result

    monkeypatch.setattr(monitor_tools._ModelArtifactScorer, "score", mutate_workspace)
    before = DatasetRepository(s.settings.db_path).list_datasets(s.task.id)
    with pytest.raises(ModelingError, match=("workspace_changed" if mutation == "workspace" else "integrity_failed")):
        monitor_tools.tool_score_dataset({
            "experiment_id": s.experiment_id, "dataset_id": s.dataset.id,
            "monitoring_binding": s.binding,
        }, s.ctx)
    assert DatasetRepository(s.settings.db_path).list_datasets(s.task.id) == before
    assert not list((s.settings.datasets_dir / s.task.id / "modeling").glob("scored_*.parquet"))


def test_binding_cannot_be_reused_with_another_input_or_task(scenario):
    s = scenario
    other = deepcopy(s.binding)
    other["dataset"]["id"] = "missing"
    assert not _score(s, monitoring_binding=other).ok
    with pytest.raises(ModelingError):
        validate_monitoring_binding(s.settings, "another-task", s.binding)


def test_scoring_loads_frozen_bytes_despite_transient_source_replacement(scenario, monkeypatch):
    s = scenario
    original_loader = scoring.load_model
    seen = []

    def replace_around_model_load(artifact, *, base_dir):
        live = _artifact_base_dir(s.settings, s.task.id) / artifact.model_path
        original = live.read_bytes()
        live.write_bytes(b"not a model while deserialization runs")
        try:
            seen.append(base_dir)
            assert base_dir != live.parent
            assert (base_dir / artifact.model_path).read_bytes() == original
            return original_loader(artifact, base_dir=base_dir)
        finally:
            live.write_bytes(original)

    monkeypatch.setattr(scoring, "load_model", replace_around_model_load)
    result = monitor_tools.tool_score_dataset({
        "experiment_id": s.experiment_id, "dataset_id": s.dataset.id,
        "monitoring_binding": s.binding,
    }, s.ctx)
    assert result["row_count"] == s.dataset.row_count
    assert seen and not seen[0].exists()
    validate_monitoring_binding(s.settings, s.task.id, s.binding,
                                scored_dataset_id=result["result_dataset_id"])
