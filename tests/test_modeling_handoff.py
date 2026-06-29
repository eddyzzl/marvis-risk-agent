import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, ModelingRepository, PluginRepository, TaskRepository, init_db
import marvis.db as db_module
from marvis.domain import TASK_TYPE_VALIDATION, TaskCreate, TaskStatus
from marvis.notebook_contract import precheck_notebook_contract
from marvis.packs.modeling.artifact import save_model
from marvis.packs.modeling.contracts import ModelMetrics, TrainConfig, TrainResult
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.handoff import (
    create_challenger_backtest_task,
    handoff_to_validation,
    mark_validated_from_validation_task,
)
import marvis.packs.modeling.tools as modeling_tools
from marvis.settings import build_settings


def _metrics() -> ModelMetrics:
    return ModelMetrics(
        train_ks=0.40,
        test_ks=0.34,
        oot_ks=0.31,
        train_auc=0.78,
        test_auc=0.73,
        oot_auc=0.70,
        psi_test_vs_train=0.02,
        psi_oot_vs_train=0.04,
        overfit_train_test_gap=0.05,
        overfit_train_oot_gap=0.09,
        overfit_flag=False,
    )


def _config(dataset_id: str) -> TrainConfig:
    return TrainConfig(
        dataset_id=dataset_id,
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"time_col": "apply_month"},
        seed=19,
        early_stopping_rounds=None,
    )


def _create_source_task(repo: TaskRepository, source_dir: Path):
    return repo.create_task(
        TaskCreate(
            model_name="贷前建模样例",
            model_version="dev",
            validator="建模平台",
            source_dir=str(source_dir),
            algorithm="lr",
            run_mode="agent",
            target_col="y",
            score_col="pred",
            split_col="split",
            time_col="apply_month",
            feature_columns=["x1", "x2"],
        )
    )


def _seed_experiment(tmp_path: Path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_task = _create_source_task(TaskRepository(settings.db_path), source_dir)
    frame = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
        "x2": [0.3, 0.4, 0.7, 0.6, 0.35, 0.75],
        "y": [0, 0, 1, 1, 0, 1],
        "split": ["train", "train", "test", "test", "oot", "oot"],
        "apply_month": ["2026-01", "2026-01", "2026-02", "2026-02", "2026-03", "2026-03"],
    })
    upload_path = tmp_path / "sample.parquet"
    frame.to_parquet(upload_path, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(upload_path, task_id=source_task.id, role="modeling_sample")
    model = LogisticRegression().fit(frame[["x1", "x2"]], frame["y"])
    model_dir = settings.tasks_dir / source_task.id / "modeling_artifacts"
    artifact = save_model(
        model,
        "lr",
        model_dir,
        feature_list=("x1", "x2"),
        params={"C": 1.0},
    )
    store = ExperimentStore(settings.db_path)
    experiment_id = store.create(source_task.id, "lr", _config(dataset.id))
    store.attach_result(
        experiment_id,
        TrainResult(
            artifact=artifact,
            metrics=_metrics(),
            feature_importance=(("x1", 0.8), ("x2", 0.2)),
            experiment_id="",
        ),
    )
    stored_artifact = ModelingRepository(settings.db_path).get_model_artifact(artifact.id)
    assert stored_artifact is not None
    return settings, store, source_task, dataset, stored_artifact


def test_handoff_to_validation_exports_pmml_and_creates_v1_task(tmp_path):
    settings, store, source_task, dataset, artifact = _seed_experiment(tmp_path)

    validation_task_id = handoff_to_validation(
        store,
        artifact,
        sample_dataset_id=dataset.id,
        settings=settings,
    )

    validation_task = TaskRepository(settings.db_path).get_task(validation_task_id)
    material_dir = Path(validation_task.source_dir)
    persisted_artifact = ModelingRepository(settings.db_path).get_model_artifact(artifact.id)
    assert persisted_artifact is not None
    assert validation_task.task_type == TASK_TYPE_VALIDATION
    assert validation_task.model_name == source_task.model_name
    assert validation_task.model_version == artifact.id
    assert validation_task.algorithm == "lr"
    assert validation_task.run_mode == "agent"
    assert validation_task.target_col == "y"
    assert validation_task.split_col == "split"
    assert validation_task.time_col == "apply_month"
    assert validation_task.feature_columns == ["x1", "x2"]
    assert validation_task.sample_path == "sample.parquet"
    assert validation_task.pmml_path == "model.pmml"
    assert validation_task.notebook_path == "scoring_notebook.ipynb"
    assert validation_task.dictionary_path == "dictionary.csv"
    assert (material_dir / "sample.parquet").exists()
    assert (material_dir / "model.pmml").exists()
    assert (material_dir / "dictionary.csv").exists()
    assert not (material_dir.parent / ".staging").exists()
    precheck_notebook_contract(material_dir / "scoring_notebook.ipynb")
    notebook = nbformat.read(material_dir / "scoring_notebook.ipynb", as_version=4)
    source = notebook.cells[0].source
    assert "Path('model.joblib')" in source or 'Path("model.joblib")' in source
    assert "Path('\\\"model.joblib\\\"')" not in source
    assert 'Path(\'"model.joblib"\')' not in source
    assert persisted_artifact.pmml_path == f"{artifact.id}.pmml"
    assert store.get(artifact.experiment_id).status == "handed_off"
    audit = db_module.PluginRepository(settings.db_path).list_audit(
        kind="modeling.validation_handoff.create",
    )[0]
    assert audit["target_ref"] == validation_task_id
    assert audit["detail"]["experiment_id"] == artifact.experiment_id
    assert audit["detail"]["artifact_id"] == artifact.id


def test_handoff_to_validation_rolls_back_task_status_and_materials_when_audit_fails(
    tmp_path,
    monkeypatch,
):
    settings, store, source_task, dataset, artifact = _seed_experiment(tmp_path)
    original_write_audit = db_module._write_audit_row

    def fail_handoff_audit(conn, *args, **kwargs):
        if kwargs.get("kind") == "modeling.validation_handoff.create":
            raise RuntimeError("audit down")
        return original_write_audit(conn, *args, **kwargs)

    monkeypatch.setattr(db_module, "_write_audit_row", fail_handoff_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        handoff_to_validation(
            store,
            artifact,
            sample_dataset_id=dataset.id,
            settings=settings,
        )

    task_repo = TaskRepository(settings.db_path)
    assert [task.id for task in task_repo.list_tasks()] == [source_task.id]
    assert store.get(artifact.experiment_id).status == "trained"
    assert not (
        settings.tasks_dir
        / source_task.id
        / "validation_handoff"
        / artifact.id
    ).exists()
    assert not (
        settings.tasks_dir
        / source_task.id
        / "validation_handoff"
        / ".staging"
    ).exists()


def test_create_challenger_backtest_task_writes_materials_task_and_audit(tmp_path):
    settings, store, source_task, dataset, artifact = _seed_experiment(tmp_path)

    result = create_challenger_backtest_task(
        store,
        artifact,
        sample_dataset_id=dataset.id,
        settings=settings,
        selection_policy_decision={"status": "ready"},
        monitoring_policy={"policy_version": "model_monitoring_v1", "status": "pass"},
        challenger_comparison={
            "status": "warn",
            "recommendation": "需复核 Champion 差异",
            "champion": {"label": "current_champion"},
            "summary": {"comparable_metric_count": 2, "metric_count": 2, "declined_count": 1},
        },
    )

    task = TaskRepository(settings.db_path).get_task(result["task_id"])
    material_dir = Path(task.source_dir)
    assert task.task_type == TASK_TYPE_VALIDATION
    assert task.model_name == source_task.model_name
    assert task.model_version == f"{artifact.id}-challenger-backtest"
    assert task.validator == "MARVIS Challenger Backtest"
    assert task.notebook_path == "scoring_notebook.ipynb"
    assert task.sample_path == "sample.parquet"
    assert task.pmml_path == "model.pmml"
    assert task.dictionary_path == "dictionary.csv"
    assert task.report_values_revision == 0
    assert (material_dir / "sample.parquet").exists()
    assert (material_dir / "model.pmml").exists()
    assert (material_dir / "model.joblib").exists()
    assert (material_dir / "dictionary.csv").exists()
    assert Path(result["package_path"]).exists()
    assert Path(result["markdown_path"]).exists()
    payload = json.loads(Path(result["package_path"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "modeling_challenger_backtest"
    assert payload["experiment_id"] == artifact.experiment_id
    assert payload["selection_policy_decision"]["status"] == "ready"
    assert payload["monitoring_policy"]["status"] == "pass"
    assert payload["challenger_comparison"]["status"] == "warn"
    assert "compare selected model" in payload["recommended_checks"][0]
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert "# Challenger / Backtest 任务包" in markdown
    assert "建议检查" in markdown
    assert "监控策略" in markdown
    assert "Champion对比" in markdown
    precheck_notebook_contract(material_dir / "scoring_notebook.ipynb")
    audit = db_module.PluginRepository(settings.db_path).list_audit(
        kind="modeling.challenger_backtest.create",
    )[0]
    assert audit["target_ref"] == result["task_id"]
    assert audit["detail"]["artifact_id"] == artifact.id
    assert store.get(artifact.experiment_id).status == "trained"


def test_create_challenger_backtest_task_rolls_back_materials_when_audit_fails(
    tmp_path,
    monkeypatch,
):
    settings, store, source_task, dataset, artifact = _seed_experiment(tmp_path)
    original_write_audit = db_module._write_audit_row

    def fail_challenger_audit(conn, *args, **kwargs):
        if kwargs.get("kind") == "modeling.challenger_backtest.create":
            raise RuntimeError("audit down")
        return original_write_audit(conn, *args, **kwargs)

    monkeypatch.setattr(db_module, "_write_audit_row", fail_challenger_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        create_challenger_backtest_task(
            store,
            artifact,
            sample_dataset_id=dataset.id,
            settings=settings,
        )

    task_repo = TaskRepository(settings.db_path)
    assert [task.id for task in task_repo.list_tasks()] == [source_task.id]
    assert not (
        settings.tasks_dir
        / source_task.id
        / "challenger_backtest"
        / artifact.id
    ).exists()
    assert not (
        settings.tasks_dir
        / source_task.id
        / "challenger_backtest"
        / ".staging"
    ).exists()


def test_export_pmml_meta_failure_does_not_persist_success_state(tmp_path, monkeypatch):
    settings, _store, source_task, _dataset, artifact = _seed_experiment(tmp_path)

    def fail_meta(*args, **kwargs):
        raise RuntimeError("meta down")

    monkeypatch.setattr(modeling_tools, "persist_model_meta", fail_meta)
    ctx = SimpleNamespace(
        task_id=source_task.id,
        workspace=settings.workspace,
        datasets_root=settings.datasets_dir,
        seed=0,
    )

    with pytest.raises(RuntimeError, match="meta down"):
        modeling_tools.tool_export_pmml({"artifact_id": artifact.id}, ctx)

    stored = ModelingRepository(settings.db_path).get_model_artifact(artifact.id)
    assert stored is not None
    assert stored.pmml_path is None
    assert PluginRepository(settings.db_path).list_audit(kind="modeling.artifact.pmml") == []
    assert not list((settings.tasks_dir / source_task.id / "modeling_artifacts").glob("*.pmml"))


def test_mark_validated_from_validation_task_updates_completed_experiment(tmp_path):
    settings, store, _, dataset, artifact = _seed_experiment(tmp_path)
    validation_task_id = handoff_to_validation(
        store,
        artifact,
        sample_dataset_id=dataset.id,
        settings=settings,
    )
    task_repo = TaskRepository(settings.db_path)
    task_repo.update_status(
        validation_task_id,
        TaskStatus.SCANNED,
        "source scanned",
        expected=TaskStatus.CREATED,
    )
    task_repo.update_status(
        validation_task_id,
        TaskStatus.RUNNING,
        "notebook running",
        expected=TaskStatus.SCANNED,
    )
    task_repo.update_status(
        validation_task_id,
        TaskStatus.EXECUTED,
        "notebook executed",
        expected=TaskStatus.RUNNING,
    )
    task_repo.update_status(
        validation_task_id,
        TaskStatus.COMPUTING_METRICS,
        "computing metrics",
        expected=TaskStatus.EXECUTED,
    )
    task_repo.update_status(
        validation_task_id,
        TaskStatus.WRITING_ARTIFACTS,
        "writing artifacts",
        expected=TaskStatus.COMPUTING_METRICS,
    )
    task_repo.update_status(
        validation_task_id,
        TaskStatus.SUCCEEDED,
        "pipeline succeeded",
        expected=TaskStatus.WRITING_ARTIFACTS,
    )

    did_update = mark_validated_from_validation_task(
        store,
        artifact,
        validation_task_id=validation_task_id,
        settings=settings,
    )

    assert did_update is True
    assert store.get(artifact.experiment_id).status == "validated"
