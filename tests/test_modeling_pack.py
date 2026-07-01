import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pandas as pd
import pytest
from openpyxl import load_workbook

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, ModelingRepository, PluginRepository, TaskRepository, init_db
import marvis.repositories.datasets as dataset_repo_module
import marvis.repositories.modeling as modeling_repo_module
from marvis.domain import TaskCreate
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.defaults import DEFAULT_RANDOM_SEED
from marvis.packs.modeling import tools as modeling_tools
from marvis.packs.modeling.tools import _ModelArtifactScorer, _effective_seed
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="建模能力包样例",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            target_col="y",
            split_col="split",
            time_col="apply_month",
            feature_columns=["x1", "x2"],
        )
    )
    return runner, plugin_registry, registry, backend, settings, task


def _register_modeling_sample(registry, tmp_path, task_id: str):
    rows = 240
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)],
        "income": [3500 + (((i * 37) % 101) * 38) + (((i * 17) % 89) * 9) for i in range(rows)],
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
        "apply_month": [f"2026-{(i % 6) + 1:02d}" for i in range(rows)],
        "approved": [1] * rows,
    })
    path = tmp_path / "modeling_sample.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="modeling_sample")


def test_modeling_manifest_registers_expected_tools(tmp_path):
    _runner, plugin_registry, _registry, _backend, _settings, _task = _runtime(tmp_path)

    manifest = plugin_registry.get("modeling")
    tool_names = {tool.name for tool in manifest.tools}
    train_tool = next(tool for tool in manifest.tools if tool.name == "train_model")
    choose_tool = next(tool for tool in manifest.tools if tool.name == "choose_modeling_spec")
    configure_tool = next(tool for tool in manifest.tools if tool.name == "configure_tuning")
    tune_tool = next(tool for tool in manifest.tools if tool.name == "tune_hyperparameters")
    calibrate_tool = next(tool for tool in manifest.tools if tool.name == "calibrate_model")
    export_tool = next(tool for tool in manifest.tools if tool.name == "export_pmml")
    handoff_tool = next(tool for tool in manifest.tools if tool.name == "handoff_to_validation")
    post_training_tool = next(tool for tool in manifest.tools if tool.name == "post_training_action")
    reject_tool = next(tool for tool in manifest.tools if tool.name == "reject_inference")
    select_tool = next(tool for tool in manifest.tools if tool.name == "select_features")
    select_experiment_tool = next(tool for tool in manifest.tools if tool.name == "select_experiment")
    report_tool = next(tool for tool in manifest.tools if tool.name == "generate_model_report")

    assert tool_names == {
        "check_data_quality",
        "modeling_readiness",
        "reject_inference",
        "prepare_modeling_frame",
        "make_split",
        "screen_features",
        "select_features",
        "choose_modeling_spec",
        "configure_tuning",
        "tune_hyperparameters",
        "train_model",
        "train_models",
        "compare_experiments",
        "select_experiment",
        "calibrate_model",
        "export_pmml",
        "handoff_to_validation",
        "post_training_action",
        "generate_model_report",
        "generate_model_reports",
    }
    assert train_tool.determinism == "stochastic"
    assert reject_tool.determinism == "deterministic"
    assert "seed" not in reject_tool.input_schema["properties"]
    assert {"read:dataset", "write:dataset"} <= set(reject_tool.side_effects)
    assert {"write:model", "write:dataset"} <= set(train_tool.side_effects)
    assert "monotone_constraints" in train_tool.input_schema["properties"]
    assert choose_tool.determinism == "deterministic"
    assert "metric_policy" in choose_tool.output_schema["required"]
    assert "params" in configure_tool.input_schema["properties"]
    assert "params" in configure_tool.output_schema["required"]
    assert "params" in tune_tool.input_schema["properties"]
    assert "oot_stability_penalty" not in tune_tool.input_schema["properties"]
    assert "OOT metrics are reported but not used" in tune_tool.summary
    assert calibrate_tool.determinism == "deterministic"
    assert {"read:model", "read:dataset", "write:model"} <= set(calibrate_tool.side_effects)
    assert select_tool.input_schema["properties"]["space"]["enum"] == ["raw", "woe"]
    assert "warnings" in select_tool.output_schema["required"]
    assert "selection_policy" in select_experiment_tool.input_schema["properties"]
    assert "selected_experiment_id" in select_experiment_tool.output_schema["required"]
    assert "policy_decision" in select_experiment_tool.output_schema["required"]
    assert "write:experiment" in select_experiment_tool.side_effects
    assert "LR modeling artifact" not in export_tool.summary
    assert "lr/lgb/xgb/scorecard" in export_tool.summary
    assert "write:task" in handoff_tool.side_effects
    assert "write:task" in post_training_tool.side_effects
    assert "actions" in post_training_tool.output_schema["required"]
    assert "create_challenger_backtest" in (
        post_training_tool.input_schema["properties"]["actions"]["items"]["enum"]
    )
    assert "model_id" in report_tool.input_schema["properties"]
    assert "llm" in report_tool.side_effects
    assert "selection_policy_decision" in post_training_tool.input_schema["properties"]
    assert "monitoring_policy" in post_training_tool.input_schema["properties"]
    assert "champion_reference" in post_training_tool.input_schema["properties"]
    assert "challenger_task_id" in post_training_tool.output_schema["required"]
    assert "challenger_package_markdown_path" in post_training_tool.output_schema["required"]
    assert "approval_package_path" in post_training_tool.output_schema["required"]
    assert "approval_package_markdown_path" in post_training_tool.output_schema["required"]
    assert "monitoring_policy_path" in post_training_tool.output_schema["required"]
    assert "monitoring_policy_markdown_path" in post_training_tool.output_schema["required"]
    assert "monitoring_policy" in post_training_tool.output_schema["required"]
    assert "model_card_path" in post_training_tool.output_schema["required"]
    assert "model_card_markdown_path" in post_training_tool.output_schema["required"]
    assert "model_card" in post_training_tool.output_schema["required"]
    assert "challenger_comparison_path" in post_training_tool.output_schema["required"]
    assert "challenger_comparison_markdown_path" in post_training_tool.output_schema["required"]
    assert "challenger_comparison" in post_training_tool.output_schema["required"]


def test_modeling_tool_seed_fallback_uses_shared_default():
    class Ctx:
        seed = None

    assert _effective_seed({}, Ctx()) == DEFAULT_RANDOM_SEED
    assert _effective_seed({"seed": 0}, Ctx()) == 0
    Ctx.seed = 17
    assert _effective_seed({}, Ctx()) == 17


def test_reject_inference_tool_registers_augmented_dataset(tmp_path):
    runner, _plugin_registry, registry, backend, settings, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.9, 0.8],
        "bad": [0, 1, None, None],
        "decision": ["approved", "approved", "rejected", "rejected"],
    })
    path = tmp_path / "reject_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

    result = runner.invoke(
        ToolRef("modeling", "reject_inference"),
        {
            "dataset_id": dataset.id,
            "target_col": "bad",
            "decision_col": "decision",
            "score_col": "score",
            "reject_bad_rate": 0.5,
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    assert result.output["target_col"] == "__reject_inference_target__"
    assert result.output["sample_weight_col"] == "__reject_inference_weight__"
    augmented_path = registry.resolve_path(result.output["result_dataset_id"])
    augmented = backend.read_frame(augmented_path)
    assert augmented_path.parent.name == "modeling"
    assert not (augmented_path.parent / ".staging").exists()
    assert "__reject_inference_source__" in augmented.columns
    assert augmented["__reject_inference_source__"].tolist().count("rejected_inferred") == 2
    assert result.output["diagnostics"]["rejected_rows"] == 2
    audit = PluginRepository(settings.db_path).list_audit(
        kind="modeling.reject_inference.created",
    )[0]
    assert audit["target_ref"] == result.output["result_dataset_id"]
    assert audit["detail"]["source_dataset_id"] == dataset.id


def test_reject_inference_audit_failure_rolls_back_dataset_and_file(tmp_path, monkeypatch):
    _runner, _plugin_registry, registry, _backend, settings, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.9, 0.8],
        "bad": [0, 1, None, None],
        "decision": ["approved", "approved", "rejected", "rejected"],
    })
    path = tmp_path / "reject_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")
    original_write_audit = dataset_repo_module._write_audit_row

    def fail_reject_inference_audit(conn, *args, **kwargs):
        if kwargs.get("kind") == "modeling.reject_inference.created":
            raise RuntimeError("reject audit down")
        return original_write_audit(conn, *args, **kwargs)

    monkeypatch.setattr(dataset_repo_module, "_write_audit_row", fail_reject_inference_audit)

    with pytest.raises(RuntimeError, match="reject audit down"):
        modeling_tools.tool_reject_inference(
            {
                "dataset_id": dataset.id,
                "target_col": "bad",
                "decision_col": "decision",
                "score_col": "score",
                "reject_bad_rate": 0.5,
            },
            SimpleNamespace(
                workspace=settings.workspace,
                datasets_root=settings.datasets_dir,
                task_id=task.id,
                seed=0,
            ),
        )

    output_dir = registry._root / task.id / "modeling"
    assert [stored.id for stored in registry.list_for_task(task.id)] == [dataset.id]
    assert not list(output_dir.glob("reject_inference_*.parquet"))
    assert not (output_dir / ".staging").exists()


def test_calibrate_model_records_diagnostics_and_report_sheet(tmp_path):
    runner, _plugin_registry, registry, _backend, settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)

    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"max_iter": 200},
            "seed": 23,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error

    calibrated = runner.invoke(
        ToolRef("modeling", "calibrate_model"),
        {
            "artifact_id": trained.output["artifact_id"],
            "dataset_id": dataset.id,
            "method": "sigmoid",
            "split": "test",
            "min_samples": 20,
            "n_bins": 5,
        },
        task_id=task.id,
    )

    assert calibrated.ok is True, calibrated.error
    assert Path(calibrated.output["calibration_path"]).exists()
    assert calibrated.output["sample_count"] == 60
    assert calibrated.output["pmml_includes_calibration"] is False
    assert {row["score_type"] for row in calibrated.output["reliability_curve"]} == {"raw", "calibrated"}
    artifact = ModelingRepository(settings.db_path).get_model_artifact(trained.output["artifact_id"])
    assert artifact is not None
    assert artifact.params["calibration"]["method"] == "sigmoid"
    assert artifact.params["calibration"]["sample_count"] == 60
    base_dir = Path(settings.tasks_dir) / task.id / "modeling_artifacts"
    assert not (base_dir / ".staging").exists()
    meta = json.loads((base_dir / f"{artifact.id}.model_meta.json").read_text(encoding="utf-8"))
    assert meta["params"]["calibration"]["path"] == f"{artifact.id}.calibration.sigmoid.joblib"
    calibration_audit = PluginRepository(settings.db_path).list_audit(
        kind="modeling.artifact.calibrate",
    )[0]
    assert calibration_audit["target_ref"] == artifact.id
    assert calibration_audit["detail"]["sample_count"] == 60

    frame = pd.read_parquet(registry.resolve_path(dataset.id))
    scorer = _ModelArtifactScorer(artifact, base_dir=base_dir)
    raw_scores = scorer.score(frame, use_calibration=False)
    calibrated_scores = scorer.score(frame)
    assert all(0.0 <= score <= 1.0 for score in calibrated_scores)
    assert any(abs(raw - calibrated) > 1e-9 for raw, calibrated in zip(raw_scores, calibrated_scores, strict=False))

    report = runner.invoke(
        ToolRef("modeling", "generate_model_report"),
        {
            "experiment_id": trained.output["experiment_id"],
            "dataset_id": dataset.id,
        },
        task_id=task.id,
    )

    assert report.ok is True, report.error
    assert report.output["calibration"][0]["method"] == "sigmoid"
    report_audit = PluginRepository(settings.db_path).list_audit(
        kind="modeling.report.generated",
    )[0]
    assert report_audit["target_ref"] == trained.output["experiment_id"]
    assert report_audit["detail"]["artifact_id"] == artifact.id
    workbook = load_workbook(report.output["report_path"])
    assert workbook["概率校准"]["A1"].value == "score_type"
    assert workbook["概率校准"]["B2"].value == "sigmoid"
    headers = [cell.value for cell in workbook["概率校准"][1]]
    assert {"bin", "avg_predicted_pd", "observed_bad_rate"} <= set(headers)

    handed_off = runner.invoke(
        ToolRef("modeling", "handoff_to_validation"),
        {
            "experiment_id": trained.output["experiment_id"],
            "sample_dataset_id": dataset.id,
        },
        task_id=task.id,
    )

    assert handed_off.ok is True, handed_off.error
    handoff_audit = PluginRepository(settings.db_path).list_audit(
        kind="modeling.validation_handoff.create",
    )[0]
    assert handoff_audit["target_ref"] == handed_off.output["validation_task_id"]
    validation_task = TaskRepository(settings.db_path).get_task(handed_off.output["validation_task_id"])
    material_dir = Path(validation_task.source_dir)
    notebook_text = (material_dir / "scoring_notebook.ipynb").read_text(encoding="utf-8")
    assert (material_dir / "calibration.joblib").exists()
    assert "RMC_CALIBRATION_FILENAME" in notebook_text
    assert "RMC_SCORE_VERSION" in notebook_text
    assert "_rmc_apply_calibration" in notebook_text
    notebook = nbformat.read(material_dir / "scoring_notebook.ipynb", as_version=4)
    namespace = {"RMC_SAMPLE_PATH": str(material_dir / "sample.parquet")}
    cwd = Path.cwd()
    try:
        os.chdir(material_dir)
        exec(notebook.cells[0].source, namespace)  # noqa: S102 - execute generated local notebook source under test
    finally:
        os.chdir(cwd)
    notebook_scores = namespace["RMC_SCORE_FN"](namespace["RMC_SAMPLE_DF"])
    assert all(0.0 <= float(score) <= 1.0 for score in notebook_scores)
    assert any(
        abs(float(left) - float(right)) > 1e-9
        for left, right in zip(notebook_scores, raw_scores, strict=False)
    )

    post_training = runner.invoke(
        ToolRef("modeling", "post_training_action"),
        {
            "experiment_id": trained.output["experiment_id"],
            "sample_dataset_id": dataset.id,
            "actions": ["export_pmml"],
        },
        task_id=task.id,
    )
    assert post_training.ok is True, post_training.error
    capabilities = post_training.output["capabilities"]
    assert capabilities["calibrated"] is True
    assert capabilities["pmml_includes_calibration"] is False
    assert capabilities["calibration"]["method"] == "sigmoid"
    assert any("PMML" in item and "校准" in item for item in capabilities["limitations"])
    assert post_training.output["model_card"]["delivery"]["pmml_includes_calibration"] is False
    assert any("PMML" in item and "校准" in item for item in post_training.output["model_card"]["limitations"])
    assert "PMML" in Path(post_training.output["model_card_markdown_path"]).read_text(encoding="utf-8")
    assert "校准" in Path(post_training.output["approval_package_markdown_path"]).read_text(encoding="utf-8")


def test_calibrate_model_rolls_back_files_and_meta_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner, _plugin_registry, registry, _backend, settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)

    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"max_iter": 200},
            "seed": 23,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error
    artifact_id = trained.output["artifact_id"]
    base_dir = Path(settings.tasks_dir) / task.id / "modeling_artifacts"
    artifact_meta_path = base_dir / f"{artifact_id}.model_meta.json"
    generic_meta_path = base_dir / "model_meta.json"
    calibration_path = base_dir / f"{artifact_id}.calibration.sigmoid.joblib"
    original_artifact_meta = artifact_meta_path.read_text(encoding="utf-8")
    original_generic_meta = generic_meta_path.read_text(encoding="utf-8")
    original_artifact = ModelingRepository(settings.db_path).get_model_artifact(artifact_id)
    assert original_artifact is not None
    original_params = original_artifact.params

    original_write_audit = modeling_repo_module._write_audit_row

    def fail_calibration_audit(conn, *args, **kwargs):
        if kwargs.get("kind") == "modeling.artifact.calibrate":
            raise RuntimeError("audit down")
        return original_write_audit(conn, *args, **kwargs)

    monkeypatch.setattr(modeling_repo_module, "_write_audit_row", fail_calibration_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        modeling_tools.tool_calibrate_model(
            {
                "artifact_id": artifact_id,
                "dataset_id": dataset.id,
                "method": "sigmoid",
                "split": "test",
                "min_samples": 20,
                "n_bins": 5,
            },
            SimpleNamespace(
                task_id=task.id,
                datasets_root=settings.datasets_dir,
                workspace=settings.workspace,
                seed=None,
            ),
        )

    assert not calibration_path.exists()
    assert not (base_dir / ".staging").exists()
    assert artifact_meta_path.read_text(encoding="utf-8") == original_artifact_meta
    assert generic_meta_path.read_text(encoding="utf-8") == original_generic_meta
    artifact = ModelingRepository(settings.db_path).get_model_artifact(artifact_id)
    assert artifact.params == original_params
    assert PluginRepository(settings.db_path).list_audit(kind="modeling.artifact.calibrate") == []


def test_select_features_supports_woe_space_on_train_split(tmp_path):
    runner, _plugin_registry, registry, _backend, _settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)

    selected = runner.invoke(
        ToolRef("modeling", "select_features"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "target_col": "y",
            "space": "woe",
            "split_col": "split",
            "split_value": "train",
            "iv_min": 0.0,
            "corr_max": 0.99,
            "vif_max": 1000,
            "scorecard_max_bins": 4,
            "enforce_monotonic": True,
            "sign_check": True,
        },
        task_id=task.id,
    )

    assert selected.ok is True, selected.error
    assert set(selected.output["selected"]) <= {"x1", "x2"}
    assert isinstance(selected.output["warnings"], list)
    for feature in ("x1", "x2"):
        assert selected.output["scores"][feature]["space"] == "woe"
        assert selected.output["scores"][feature]["bin_count"] >= 1
        assert selected.output["scores"][feature]["monotonic_direction"] in {"increasing", "decreasing"}


def test_train_model_applies_top_level_monotone_constraints_for_tree_models(tmp_path):
    runner, _plugin_registry, registry, _backend, settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)

    recipes = {
        "lgb": ({"num_boost_round": 2, "num_leaves": 4}, [1, -1]),
        "xgb": ({"num_boost_round": 2, "max_depth": 2}, "(1,-1)"),
    }
    for recipe, (params, expected_constraints) in recipes.items():
        trained = runner.invoke(
            ToolRef("modeling", "train_model"),
            {
                "dataset_id": dataset.id,
                "recipe": recipe,
                "features": ["x1", "x2"],
                "target_col": "y",
                "split_col": "split",
                "split_values": {"train": "train", "test": "test", "oot": "oot"},
                "params": params,
                "monotone_constraints": {"x1": 1, "x2": -1},
                "seed": 23,
            },
            task_id=task.id,
        )

        assert trained.ok is True, trained.error
        artifact = ModelingRepository(settings.db_path).get_model_artifact(trained.output["artifact_id"])
        assert artifact is not None
        assert artifact.params["monotone_constraints"] == expected_constraints


def test_modeling_pack_tools_round_trip_via_runner(tmp_path):
    runner, _plugin_registry, registry, _backend, settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)

    quality = runner.invoke(
        ToolRef("modeling", "check_data_quality"),
        {"dataset_id": dataset.id, "target_col": "y"},
        task_id=task.id,
    )
    readiness = runner.invoke(
        ToolRef("modeling", "modeling_readiness"),
        {"dataset_id": dataset.id, "target_col": "y", "split_col": "split"},
        task_id=task.id,
    )
    prepared = runner.invoke(
        ToolRef("modeling", "prepare_modeling_frame"),
        {
            "dataset_id": dataset.id,
            "target_col": "y",
            "feature_cols": ["x1", "x2"],
            "split_col": "split",
            "seed": 11,
        },
        task_id=task.id,
    )

    assert quality.ok is True, quality.error
    assert readiness.ok is True, readiness.error
    assert prepared.ok is True, prepared.error
    assert "result_dataset_id" in prepared.output
    assert prepared.output["split_counts"] == {"oot": 40, "test": 60, "train": 140}
    prepared_audit = PluginRepository(settings.db_path).list_audit(
        kind="modeling.dataset.derived",
    )[0]
    assert prepared_audit["target_ref"] == prepared.output["result_dataset_id"]
    assert prepared_audit["detail"]["tool"] == "prepare_modeling_frame"

    selected = runner.invoke(
        ToolRef("modeling", "select_features"),
        {
            "dataset_id": prepared.output["result_dataset_id"],
            "features": ["x1", "x2"],
            "target_col": "y",
            "iv_min": 0.0,
            "corr_max": 0.99,
            "top_k": 2,
        },
        task_id=task.id,
    )

    assert selected.ok is True, selected.error
    assert set(selected.output["selected"]) <= {"x1", "x2"}

    train_outputs = {}
    for recipe, params in {
        "lgb": {"num_boost_round": 2, "learning_rate": 0.1, "num_leaves": 4},
        "xgb": {"num_boost_round": 2, "max_depth": 2, "eta": 0.1},
        "lr": {"max_iter": 200},
        "scorecard": {"scorecard_max_bins": 3, "max_iter": 200},
    }.items():
        result = runner.invoke(
            ToolRef("modeling", "train_model"),
            {
                "dataset_id": prepared.output["result_dataset_id"],
                "recipe": recipe,
                "features": ["x1", "x2"],
                "target_col": "y",
                "split_col": "split",
                "split_values": {"train": "train", "test": "test", "oot": "oot"},
                "params": params,
                "seed": 23,
            },
            task_id=task.id,
        )
        assert result.ok is True, result.error
        assert result.output["metrics"]["overfit_flag"] in {True, False}
        assert result.output["artifact_id"]
        train_outputs[recipe] = result.output

    repeated_lr = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": prepared.output["result_dataset_id"],
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"max_iter": 200},
            "seed": 23,
        },
        task_id=task.id,
    )

    assert repeated_lr.ok is True, repeated_lr.error
    assert repeated_lr.output["metrics"]["test_auc"] == train_outputs["lr"]["metrics"]["test_auc"]

    compared = runner.invoke(
        ToolRef("modeling", "compare_experiments"),
        {"experiment_ids": [output["experiment_id"] for output in train_outputs.values()]},
        task_id=task.id,
    )
    exported_by_recipe = {}
    for recipe, output in train_outputs.items():
        exported_by_recipe[recipe] = runner.invoke(
            ToolRef("modeling", "export_pmml"),
            {"artifact_id": output["artifact_id"]},
            task_id=task.id,
        )
    handed_off = runner.invoke(
        ToolRef("modeling", "handoff_to_validation"),
        {
            "experiment_id": train_outputs["lr"]["experiment_id"],
            "sample_dataset_id": prepared.output["result_dataset_id"],
        },
        task_id=task.id,
    )
    scorecard_handed_off = runner.invoke(
        ToolRef("modeling", "handoff_to_validation"),
        {
            "experiment_id": train_outputs["scorecard"]["experiment_id"],
            "sample_dataset_id": prepared.output["result_dataset_id"],
        },
        task_id=task.id,
    )

    assert compared.ok is True, compared.error
    scorecard_artifact = ModelingRepository(settings.db_path).get_model_artifact(
        train_outputs["scorecard"]["artifact_id"]
    )
    assert scorecard_artifact is not None
    assert scorecard_artifact.scorecard_table
    assert {"feature", "bin_label", "points"} <= set(scorecard_artifact.scorecard_table[0])
    assert [row["recipe"] for row in compared.output["experiments"]] == [
        "lgb",
        "xgb",
        "lr",
        "scorecard",
    ]
    rows_by_recipe = {row["recipe"]: row for row in compared.output["experiments"]}
    for recipe in ("lgb", "xgb", "lr", "scorecard"):
        assert rows_by_recipe[recipe]["capabilities"]["pmml_supported"] is True
        assert rows_by_recipe[recipe]["capabilities"]["handoff_supported"] is True
        assert rows_by_recipe[recipe]["capabilities"]["reason"] is None
        assert exported_by_recipe[recipe].ok is True, exported_by_recipe[recipe].error
        assert Path(exported_by_recipe[recipe].output["pmml_path"]).exists()
        meta_path = (
            Path(settings.tasks_dir)
            / task.id
            / "modeling_artifacts"
            / f"{train_outputs[recipe]['artifact_id']}.model_meta.json"
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["pmml_path"] == f"{train_outputs[recipe]['artifact_id']}.pmml"
    pmml_audits = PluginRepository(settings.db_path).list_audit(kind="modeling.artifact.pmml")
    assert {audit["target_ref"] for audit in pmml_audits} >= {
        output["artifact_id"] for output in train_outputs.values()
    }
    assert handed_off.ok is True, handed_off.error
    assert scorecard_handed_off.ok is True, scorecard_handed_off.error
    validation_task = TaskRepository(settings.db_path).get_task(
        handed_off.output["validation_task_id"]
    )
    scorecard_validation_task = TaskRepository(settings.db_path).get_task(
        scorecard_handed_off.output["validation_task_id"]
    )
    assert validation_task.task_type == "validation"
    assert validation_task.pmml_path == "model.pmml"
    assert scorecard_validation_task.task_type == "validation"
    assert scorecard_validation_task.algorithm == "scorecard"
    assert scorecard_validation_task.pmml_path == "model.pmml"


def test_modeling_pack_trains_income_regression_scenario_via_runner(tmp_path):
    runner, _plugin_registry, registry, _backend, _settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)

    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "lgb_regressor",
            "features": ["x1", "x2"],
            "target_col": "income",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"num_boost_round": 4, "learning_rate": 0.1, "num_leaves": 4},
            "seed": 29,
            "scenario": "income",
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    assert result.output["artifact_id"]
    assert result.output["metrics"]["test_ks"] is None
    assert result.output["metrics"]["test_auc"] is None
    assert result.output["metrics"]["test_rmse"] > 0
    assert result.output["metrics"]["test_mae"] > 0

    compared = runner.invoke(
        ToolRef("modeling", "compare_experiments"),
        {"experiment_ids": [result.output["experiment_id"]]},
        task_id=task.id,
    )

    assert compared.ok is True, compared.error
    row = compared.output["experiments"][0]
    assert row["recipe"] == "lgb_regressor"
    assert row["test_ks"] is None
    assert row["test_rmse"] == result.output["metrics"]["test_rmse"]


def _register_modeling_sample_with_nan_train_label(registry, tmp_path, task_id: str):
    rows = 240
    y = [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)]
    target = [float(value) for value in y]
    target[0] = float("nan")  # one missing label in the train split
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": target,
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
    })
    path = tmp_path / "modeling_nan_sample.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="modeling_sample")


def test_train_model_gates_nan_train_label(tmp_path):
    runner, _plugin_registry, registry, _backend, _settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample_with_nan_train_label(registry, tmp_path, task.id)
    base_inputs = {
        "dataset_id": dataset.id,
        "recipe": "lr",
        "features": ["x1", "x2"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "params": {"max_iter": 200},
        "seed": 23,
    }

    blocked = runner.invoke(ToolRef("modeling", "train_model"), dict(base_inputs), task_id=task.id)
    assert blocked.ok is False
    assert blocked.error_kind == "nan_label_not_confirmed"
    assert blocked.error_detail["by_split"]["train"]["n_nan"] == 1

    confirmed = runner.invoke(
        ToolRef("modeling", "train_model"),
        {**base_inputs, "drop_nan_labels": True},
        task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["nan_labels_dropped"] == 1


def test_train_model_allows_scoring_only_oot(tmp_path):
    runner, _plugin_registry, registry, _backend, _settings, task = _runtime(tmp_path)
    rows = 240
    y = [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)]
    target = [float(value) for value in y]
    for i in range(200, 240):  # OOT split has no labels at all (scoring only)
        target[i] = float("nan")
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": target,
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
    })
    path = tmp_path / "modeling_oot_scoring.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"max_iter": 200},
            "seed": 23,
        },
        task_id=task.id,
    )
    # Scoring-only OOT is legitimate: no gate, OOT label metrics unavailable, no rows dropped.
    assert result.ok is True, result.error
    assert result.output["nan_labels_dropped"] == 0
    assert result.output["metrics"]["oot_ks"] is None
    assert result.output["metrics"]["oot_auc"] is None


def test_tune_skips_random_search_for_non_lgb_recipe():
    """The LightGBM random search only runs for the lgb recipe; other recipes
    (lr/scorecard/xgb/catboost) train with their own defaults, so tuning returns empty
    params (G2 recipe-aware tune). The skip path runs before touching the runtime."""
    from marvis.packs.modeling.tools import tool_tune_hyperparameters

    for recipe in ("lr", "scorecard", "xgb", "catboost"):
        out = tool_tune_hyperparameters(
            {
                "recipe": recipe,
                "dataset_id": "x",
                "features": ["a"],
                "target_col": "y",
                "split_col": "split",
                "split_values": {},
            },
            ctx=None,
        )
        assert out == {"best_params": {}, "best_metrics": {}, "n_trials": 0, "trials": []}


def test_choose_modeling_spec_validates_family_and_sample_weight_controls():
    from marvis.packs.modeling.errors import ModelingError
    from marvis.packs.modeling.tools import tool_choose_modeling_spec

    out = tool_choose_modeling_spec(
        {
            "target_col": "bad",
            "features": ["x1", "weight", "x2"],
            "recipe": "lgb",
            "recipes": ["lgb", "catboost"],
            "target_type": "binary",
            "sample_weight_col": "weight",
            "sample_weight_candidates": ["weight", "sample_weight"],
            "sample_weight_diagnostics": [
                {
                    "column": "weight",
                    "valid": True,
                    "missing_rate": 0.0,
                    "min": 1.0,
                    "max": 2.0,
                    "mean": 1.4,
                    "reason": "",
                }
            ],
            "n_trials": 9,
            "params": {"learning_rate": 0.03},
            "seed": 17,
        },
        SimpleNamespace(seed=None),
    )

    assert out["recipe"] == "lgb"
    assert out["recipes"] == ["lgb", "catboost"]
    assert out["target_type"] == "binary"
    assert out["sample_weight_col"] == "weight"
    assert out["sample_weight_diagnostics"][0]["column"] == "weight"
    assert out["sample_weight_diagnostics"][0]["valid"] is True
    assert out["feature_cols"] == ["x1", "x2"]
    assert out["params"]["sample_weight_col"] == "weight"
    assert out["params"]["learning_rate"] == 0.03
    assert "lgb" in out["eligible_algorithms"]
    assert any(item["recipe"] == "lgb_regressor" for item in out["disabled_algorithms"])

    with pytest.raises(ModelingError, match="cannot be mixed"):
        tool_choose_modeling_spec(
            {"recipe": "lgb", "recipes": ["lgb", "lgb_regressor"], "seed": 17},
            SimpleNamespace(seed=None),
        )


def test_configure_tuning_preserves_manual_controls_and_disables_non_lgb():
    from marvis.packs.modeling.tools import tool_configure_tuning

    lgb = tool_configure_tuning(
        {
            "recipe": "lgb",
            "target_type": "binary",
            "n_trials": 7,
            "seed": 13,
            "sample_weight_col": "weight",
            "params": {"learning_rate": 0.03, "monotone_constraints": {"x1": 1, "x2": -1}},
        },
        SimpleNamespace(seed=None),
    )
    assert lgb["tune_enabled"] is True
    assert lgb["n_trials"] == 7
    assert lgb["sample_weight_col"] == "weight"
    assert lgb["params"]["sample_weight_col"] == "weight"
    assert lgb["params"]["learning_rate"] == 0.03
    assert lgb["params"]["monotone_constraints"] == {"x1": 1, "x2": -1}

    lr = tool_configure_tuning({"recipe": "lr", "seed": 13, "n_trials": 7}, SimpleNamespace(seed=None))
    assert lr["tune_enabled"] is False
    assert lr["n_trials"] == 0
    assert lr["params"] == {}


def test_train_models_trains_each_recipe_and_picks_best(tmp_path):
    """G2 multi-algorithm: train_models trains every requested recipe and returns
    all experiments + the best by OOT KS. lgb consumes the tuned params; lr trains
    with its defaults."""
    runner, _pr, registry, _backend, _settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)
    prepared = runner.invoke(
        ToolRef("modeling", "prepare_modeling_frame"),
        {"dataset_id": dataset.id, "target_col": "y", "feature_cols": ["x1", "x2"], "split_col": "split", "seed": 11},
        task_id=task.id,
    )
    assert prepared.ok is True, prepared.error

    result = runner.invoke(
        ToolRef("modeling", "train_models"),
        {
            "dataset_id": prepared.output["result_dataset_id"],
            "recipes": ["lgb", "lr"],
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"num_boost_round": 2, "learning_rate": 0.1, "num_leaves": 4},
            "seed": 23,
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    out = result.output
    assert {exp["recipe"] for exp in out["experiments"]} == {"lgb", "lr"}
    assert len(out["experiment_ids"]) == 2
    assert out["best_experiment_id"] in out["experiment_ids"]
    assert out["best_recipe"] in {"lgb", "lr"}
    assert out["target_type"] == "binary"
    assert out["selection_metric"] == "oot_ks"


def test_train_models_supports_catboost_and_sample_weight_col(tmp_path):
    runner, _pr, registry, backend, settings, task = _runtime(tmp_path)
    rows = 180
    frame = pd.DataFrame({
        "x1": [((i * 19) % 97) / 100 for i in range(rows)],
        "x2": [((i * 23) % 83) / 100 for i in range(rows)],
        "weight": [2.0 if i % 5 == 0 else 1.0 for i in range(rows)],
        "y": [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)],
        "split": ["train"] * 100 + ["test"] * 50 + ["oot"] * 30,
    })
    path = tmp_path / "weighted_modeling_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")
    prepared = runner.invoke(
        ToolRef("modeling", "make_split"),
        {
            "dataset_id": dataset.id,
            "target_col": "y",
            "feature_cols": ["x1", "x2"],
            "split_col": "split",
            "passthrough_cols": ["weight"],
            "seed": 11,
        },
        task_id=task.id,
    )
    assert prepared.ok is True, prepared.error
    prepared_frame = backend.read_frame(registry.resolve_path(prepared.output["result_dataset_id"]))
    assert "weight" in prepared_frame.columns
    assert prepared.output["feature_cols"] == ["x1", "x2"]

    trained = runner.invoke(
        ToolRef("modeling", "train_models"),
        {
            "dataset_id": prepared.output["result_dataset_id"],
            "recipes": ["lgb", "catboost"],
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"num_boost_round": 2, "learning_rate": 0.1, "num_leaves": 4},
            "sample_weight_col": "weight",
            "seed": 23,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error
    assert {exp["recipe"] for exp in trained.output["experiments"]} == {"lgb", "catboost"}

    store = ExperimentStore(settings.db_path)
    modeling_repo = ModelingRepository(settings.db_path)
    artifacts = {}
    for experiment_id in trained.output["experiment_ids"]:
        experiment = store.get(experiment_id)
        artifact = modeling_repo.get_model_artifact(experiment.artifact_id)
        assert artifact is not None
        artifacts[artifact.algorithm] = artifact
        assert artifact.model_path.endswith(".pkl")
        assert artifact.params["sample_weight_col"] == "weight"

    assert set(artifacts) == {"lgb", "catboost"}
    compared = runner.invoke(
        ToolRef("modeling", "compare_experiments"),
        {"experiment_ids": trained.output["experiment_ids"]},
        task_id=task.id,
    )
    assert compared.ok is True, compared.error
    catboost_row = next(row for row in compared.output["experiments"] if row["recipe"] == "catboost")
    assert catboost_row["capabilities"]["native_model_supported"] is True
    assert catboost_row["capabilities"]["pmml_supported"] is False
    assert "sklearn2pmml" in catboost_row["capabilities"]["reason"]
    catboost_experiment_id = next(
        exp_id
        for exp_id in trained.output["experiment_ids"]
        if store.get(exp_id).recipe_id == "catboost"
    )
    lgb_experiment_id = next(
        exp_id
        for exp_id in trained.output["experiment_ids"]
        if store.get(exp_id).recipe_id == "lgb"
    )
    prior_selected = runner.invoke(
        ToolRef("modeling", "select_experiment"),
        {
            "experiment_ids": trained.output["experiment_ids"],
            "selected_experiment_id": lgb_experiment_id,
            "target_type": "binary",
        },
        task_id=task.id,
    )
    assert prior_selected.ok is True, prior_selected.error
    assert store.get(lgb_experiment_id).status == "selected"
    selected = runner.invoke(
        ToolRef("modeling", "select_experiment"),
        {
            "experiment_ids": trained.output["experiment_ids"],
            "selected_experiment_id": catboost_experiment_id,
            "target_type": "binary",
        },
        task_id=task.id,
    )
    assert selected.ok is True, selected.error
    assert selected.output["selected_experiment_id"] == catboost_experiment_id
    assert selected.output["artifact_id"] == artifacts["catboost"].id
    assert selected.output["selection_metric"] == "manual"
    assert selected.output["policy_decision"]["status"] == "not_requested"
    assert store.get(catboost_experiment_id).status == "selected"
    blocked = runner.invoke(
        ToolRef("modeling", "select_experiment"),
        {
            "experiment_ids": trained.output["experiment_ids"],
            "selected_experiment_id": catboost_experiment_id,
            "target_type": "binary",
            "selection_policy": {"require_pmml": True, "require_handoff": True},
        },
        task_id=task.id,
    )
    assert blocked.ok is False
    assert "violates selection_policy" in blocked.error
    assert "require_pmml" in blocked.error
    assert "require_handoff" in blocked.error
    missing_reason = runner.invoke(
        ToolRef("modeling", "select_experiment"),
        {
            "experiment_ids": trained.output["experiment_ids"],
            "selected_experiment_id": catboost_experiment_id,
            "target_type": "binary",
            "selection_policy": {
                "require_pmml": True,
                "require_handoff": True,
                "allow_policy_override": True,
            },
        },
        task_id=task.id,
    )
    assert missing_reason.ok is False
    assert "override_reason_required" in missing_reason.error
    overridden = runner.invoke(
        ToolRef("modeling", "select_experiment"),
        {
            "experiment_ids": trained.output["experiment_ids"],
            "selected_experiment_id": catboost_experiment_id,
            "target_type": "binary",
            "selection_policy": {
                "require_pmml": True,
                "require_handoff": True,
                "allow_policy_override": True,
                "override_reason": "业务方本轮只验收原生 CatBoost pkl,不做 V1 验证移交。",
            },
        },
        task_id=task.id,
    )
    assert overridden.ok is True, overridden.error
    assert overridden.output["selected_experiment_id"] == catboost_experiment_id
    assert overridden.output["policy_decision"]["status"] == "overridden"
    assert overridden.output["policy_decision"]["override_reason"].startswith("业务方本轮只验收")
    assert {item["code"] for item in overridden.output["policy_decision"]["violations"]} == {
        "require_pmml",
        "require_handoff",
    }
    post_training = runner.invoke(
        ToolRef("modeling", "post_training_action"),
        {
            "experiment_id": selected.output["selected_experiment_id"],
            "sample_dataset_id": prepared.output["result_dataset_id"],
            "actions": ["export_pmml", "handoff_to_validation", "create_challenger_backtest"],
            "selection_policy_decision": overridden.output["policy_decision"],
            "monitoring_policy": {
                "owner": "model_governance",
                "thresholds": {"oot_ks": {"warn": 100.0, "fail": 99.0}},
            },
        },
        task_id=task.id,
    )
    assert post_training.ok is True, post_training.error
    assert post_training.output["artifact_id"] == artifacts["catboost"].id
    assert post_training.output["capabilities"]["native_model_supported"] is True
    assert {item["status"] for item in post_training.output["actions"]} == {"skipped"}
    assert post_training.output["challenger_task_id"] == ""
    assert post_training.output["challenger_package_markdown_path"] == ""
    assert post_training.output["monitoring_policy"]["owner"] == "model_governance"
    assert post_training.output["monitoring_policy"]["schema_version"] == 1
    assert post_training.output["monitoring_policy"]["policy_version"] == "model_monitoring_v1"
    assert post_training.output["model_card"]["card_version"] == "model_card_v1"
    assert post_training.output["model_card"]["artifact_id"] == artifacts["catboost"].id
    assert post_training.output["model_card"]["delivery"]["export_pmml_status"] == "skipped"
    assert post_training.output["challenger_comparison"]["status"] in {"pass", "warn"}
    assert post_training.output["challenger_comparison"]["champion"]["label"] == "previous_selected_experiment"
    assert post_training.output["challenger_comparison"]["champion"]["experiment_id"] == lgb_experiment_id
    assert post_training.output["challenger_comparison"]["summary"]["comparable_metric_count"] >= 1
    assert {item["metric"] for item in post_training.output["monitoring_policy"]["checks"]} >= {
        "oot_ks",
        "psi_oot_vs_train",
    }
    assert "oot_rmse" not in {
        item["metric"] for item in post_training.output["monitoring_policy"]["checks"]
    }
    monitoring_policy = Path(post_training.output["monitoring_policy_path"])
    monitoring_markdown = Path(post_training.output["monitoring_policy_markdown_path"])
    assert monitoring_policy.exists()
    assert monitoring_markdown.exists()
    monitoring_payload = json.loads(monitoring_policy.read_text(encoding="utf-8"))
    assert monitoring_payload["owner"] == "model_governance"
    assert "# 模型监控策略" in monitoring_markdown.read_text(encoding="utf-8")
    model_card_path = Path(post_training.output["model_card_path"])
    model_card_markdown = Path(post_training.output["model_card_markdown_path"])
    assert model_card_path.exists()
    assert model_card_markdown.exists()
    model_card_payload = json.loads(model_card_path.read_text(encoding="utf-8"))
    assert model_card_payload["card_version"] == "model_card_v1"
    assert model_card_payload["governance"]["monitoring_status"] == post_training.output["monitoring_policy"]["status"]
    assert "# 模型卡" in model_card_markdown.read_text(encoding="utf-8")
    comparison_path = Path(post_training.output["challenger_comparison_path"])
    comparison_markdown = Path(post_training.output["challenger_comparison_markdown_path"])
    assert comparison_path.exists()
    assert comparison_markdown.exists()
    comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison_payload["comparison_version"] == "champion_challenger_v1"
    assert comparison_payload["champion"]["experiment_id"] == lgb_experiment_id
    assert "# Champion / Challenger 对比" in comparison_markdown.read_text(encoding="utf-8")
    approval_package = Path(post_training.output["approval_package_path"])
    assert approval_package.exists()
    approval_payload = json.loads(approval_package.read_text(encoding="utf-8"))
    assert approval_payload["schema_version"] == 1
    assert approval_payload["experiment_id"] == catboost_experiment_id
    assert approval_payload["artifact_id"] == artifacts["catboost"].id
    assert approval_payload["selection_policy_decision"]["status"] == "overridden"
    assert approval_payload["selection_policy_decision"]["override_reason"].startswith("业务方本轮只验收")
    assert approval_payload["monitoring_policy"]["owner"] == "model_governance"
    assert approval_payload["model_card"]["card_version"] == "model_card_v1"
    assert approval_payload["artifacts"]["model_card_markdown_path"].endswith(".model_card.md")
    assert approval_payload["challenger_comparison"]["champion"]["experiment_id"] == lgb_experiment_id
    assert {item["status"] for item in approval_payload["delivery_actions"]} == {"skipped"}
    assert approval_payload["artifacts"]["challenger_task_id"] == ""
    assert approval_payload["artifacts"]["challenger_comparison_markdown_path"].endswith(
        ".champion_comparison.md"
    )
    assert approval_payload["feature_count"] == 2
    approval_markdown = Path(post_training.output["approval_package_markdown_path"])
    assert approval_markdown.exists()
    markdown_text = approval_markdown.read_text(encoding="utf-8")
    assert "# 模型审批包" in markdown_text
    assert "业务方本轮只验收" in markdown_text
    assert "require_pmml" in markdown_text
    assert "Challenger/Backtest任务" in markdown_text
    assert "模型卡" in markdown_text
    assert "## 监控策略" in markdown_text
    assert "## Champion对比" in markdown_text
    assert "## 入模特征" in markdown_text


def test_pick_best_experiment_is_target_type_aware():
    from marvis.packs.modeling.tools import _pick_best_experiment

    regression = [
        {"experiment_id": "reg-a", "recipe": "lgb_regressor", "metrics": {"oot_rmse": 2.5, "test_rmse": 2.0}},
        {"experiment_id": "reg-b", "recipe": "lgb_regressor", "metrics": {"oot_rmse": 1.8, "test_rmse": 2.2}},
    ]
    best, metric = _pick_best_experiment(regression, target_type="continuous")
    assert best["experiment_id"] == "reg-b"
    assert metric == "oot_rmse"

    multiclass = [
        {"experiment_id": "mc-a", "recipe": "lgb_multiclass", "metrics": {"oot_macro_auc": 0.71}},
        {"experiment_id": "mc-b", "recipe": "lgb_multiclass", "metrics": {"oot_macro_auc": 0.82}},
    ]
    best, metric = _pick_best_experiment(multiclass, target_type="multiclass")
    assert best["experiment_id"] == "mc-b"
    assert metric == "oot_macro_auc"


def test_pick_best_comparison_row_prefers_delivery_ready_candidate():
    from marvis.packs.modeling.tools import _pick_best_comparison_row

    rows = [
        {
            "id": "catboost-best",
            "recipe": "catboost",
            "oot_ks": 0.52,
            "capabilities": {"pmml_supported": False, "handoff_supported": False},
        },
        {
            "id": "lgb-deliverable",
            "recipe": "lgb",
            "oot_ks": 0.48,
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
        },
    ]

    best, metric = _pick_best_comparison_row(rows, target_type="binary")

    assert best["id"] == "lgb-deliverable"
    assert metric == "oot_ks"


def test_policy_selection_prefers_compliant_scorecard_candidate():
    from marvis.packs.modeling.tools import _pick_best_comparison_row_with_policy

    rows = [
        {
            "id": "lgb-higher-ks",
            "recipe": "lgb",
            "oot_ks": 0.55,
            "psi_oot_vs_train": 0.04,
            "feature_count": 18,
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
            "model_params": {"monotone_constraints": [1, 0, -1]},
        },
        {
            "id": "scorecard-compliant",
            "recipe": "scorecard",
            "oot_ks": 0.49,
            "psi_oot_vs_train": 0.03,
            "feature_count": 12,
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
            "scorecard_table": [{"feature": "x1", "monotonic_direction": "increasing"}],
        },
        {
            "id": "scorecard-partial-monotonic",
            "recipe": "scorecard",
            "oot_ks": 0.53,
            "psi_oot_vs_train": 0.03,
            "feature_count": 12,
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
            "scorecard_table": [
                {"feature": "x1", "monotonic_direction": "increasing"},
                {"feature": "x2"},
            ],
        },
        {
            "id": "scorecard-no-monotonic",
            "recipe": "scorecard",
            "oot_ks": 0.51,
            "psi_oot_vs_train": 0.03,
            "feature_count": 12,
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
            "scorecard_table": [{"feature": "x2"}],
        },
    ]

    best, metric, decision = _pick_best_comparison_row_with_policy(
        rows,
        target_type="binary",
        policy={
            "require_pmml": True,
            "require_handoff": True,
            "require_monotonicity": True,
            "prefer_scorecard": True,
            "allow_policy_override": False,
            "override_reason": "",
        },
    )

    assert best["id"] == "scorecard-compliant"
    assert metric == "oot_ks"
    assert decision["status"] == "accepted"
    assert decision["policy_candidate_count"] == 2
    assert decision["selected_by_preference"] is True


def test_selection_policy_rejects_partial_scorecard_monotonicity():
    from marvis.packs.modeling.tools import _selection_policy_decision

    decision = _selection_policy_decision(
        {
            "id": "scorecard-partial-monotonic",
            "recipe": "scorecard",
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
            "scorecard_table": [
                {"feature": "x1", "monotonic_direction": "increasing"},
                {"feature": "x2"},
            ],
        },
        {"require_monotonicity": True},
        explicit=False,
    )

    assert decision["status"] == "blocked"
    assert decision["profile"]["monotonicity_declared"] is False
    assert decision["profile"]["monotonicity_coverage"] == "partial"
    assert decision["profile"]["monotonicity_missing_features"] == ["x2"]
    assert decision["violations"][0]["code"] == "require_monotonicity"
    assert "x2" in decision["violations"][0]["message"]


def test_selection_policy_rejects_zero_monotone_constraints():
    from marvis.packs.modeling.tools import _selection_policy_decision

    decision = _selection_policy_decision(
        {
            "id": "lgb-zero-constraints",
            "recipe": "lgb",
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
            "model_params": {"monotone_constraints": [0, 0, 0]},
        },
        {"require_monotonicity": True},
        explicit=False,
    )

    assert decision["status"] == "blocked"
    assert decision["profile"]["monotonicity_declared"] is False
    assert decision["violations"][0]["code"] == "require_monotonicity"


def test_selection_policy_rejects_missing_feature_and_psi_evidence():
    from marvis.packs.modeling.tools import _selection_policy_decision

    policy = {
        "max_feature_count": 30,
        "max_oot_psi": 0.15,
        "allow_policy_override": False,
        "override_reason": "",
    }
    decision = _selection_policy_decision(
        {
            "id": "missing-policy-evidence",
            "recipe": "lgb",
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
        },
        policy,
        explicit=False,
    )

    assert decision["status"] == "blocked"
    assert {item["code"] for item in decision["violations"]} == {
        "max_feature_count_missing",
        "max_oot_psi_missing",
    }

    overridden = _selection_policy_decision(
        {
            "id": "missing-policy-evidence",
            "recipe": "lgb",
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
        },
        {
            **policy,
            "allow_policy_override": True,
            "override_reason": "历史候选缺少早期证据，本轮人工复核后临时放行。",
        },
        explicit=True,
    )

    assert overridden["status"] == "overridden"
    assert overridden["override_reason"].startswith("历史候选缺少早期证据")


def test_selection_policy_prefers_weighted_oot_psi_when_available():
    from marvis.packs.modeling.tools import _selection_policy_decision

    decision = _selection_policy_decision(
        {
            "id": "weighted-drift",
            "recipe": "lgb",
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
            "psi_oot_vs_train": 0.04,
            "weighted_psi_oot_vs_train": 0.22,
        },
        {"max_oot_psi": 0.15},
        explicit=False,
    )

    assert decision["status"] == "blocked"
    assert decision["profile"]["psi_oot_vs_train"] == 0.04
    assert decision["profile"]["weighted_psi_oot_vs_train"] == 0.22
    assert decision["profile"]["policy_psi_oot_vs_train"] == 0.22
    assert decision["profile"]["policy_psi_source"] == "weighted_psi_oot_vs_train"
    assert decision["violations"][0]["code"] == "max_oot_psi"
    assert "加权 OOT PSI" in decision["violations"][0]["message"]


def test_selection_policy_uses_weighted_oot_psi_when_raw_missing():
    from marvis.packs.modeling.tools import _selection_policy_decision

    decision = _selection_policy_decision(
        {
            "id": "weighted-only",
            "recipe": "lgb",
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
            "weighted_psi_oot_vs_train": 0.08,
        },
        {"max_oot_psi": 0.15},
        explicit=False,
    )

    assert decision["status"] == "accepted"
    assert decision["profile"]["policy_psi_oot_vs_train"] == 0.08
    assert decision["profile"]["policy_psi_source"] == "weighted_psi_oot_vs_train"
    assert decision["violations"] == []


def test_selection_policy_metric_thresholds_pick_compliant_business_candidate():
    from marvis.packs.modeling.tools import _pick_best_comparison_row_with_policy

    rows = [
        {
            "id": "high-test-low-oot",
            "recipe": "lgb",
            "test_ks": 0.45,
            "oot_ks": 0.29,
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
        },
        {
            "id": "business-compliant",
            "recipe": "lgb",
            "test_ks": 0.39,
            "oot_ks": 0.33,
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
        },
    ]

    best, metric, decision = _pick_best_comparison_row_with_policy(
        rows,
        target_type="binary",
        policy={"min_oot_ks": 0.30},
    )

    assert best["id"] == "business-compliant"
    assert metric == "oot_ks"
    assert decision["status"] == "accepted"
    assert decision["policy"]["metric_thresholds"] == {"oot_ks": {"min": 0.30}}
    assert decision["policy_candidate_count"] == 1


def test_selection_policy_metric_thresholds_block_missing_or_weak_metrics():
    from marvis.packs.modeling.tools import _selection_policy_decision

    weak_regression = _selection_policy_decision(
        {
            "id": "weak-regressor",
            "recipe": "lgb_regressor",
            "oot_rmse": 2.1,
            "capabilities": {"pmml_supported": False, "handoff_supported": False},
        },
        {"metric_thresholds": {"oot_rmse": {"max": 1.8}}},
        explicit=False,
    )

    assert weak_regression["status"] == "blocked"
    assert weak_regression["violations"][0]["code"] == "metric_max_threshold"
    assert weak_regression["violations"][0]["metric"] == "oot_rmse"

    missing_metric = _selection_policy_decision(
        {
            "id": "missing-oot-ks",
            "recipe": "lgb",
            "test_ks": 0.42,
            "capabilities": {"pmml_supported": True, "handoff_supported": True},
        },
        {"min_oot_ks": 0.30},
        explicit=False,
    )

    assert missing_metric["status"] == "blocked"
    assert missing_metric["violations"][0]["code"] == "metric_threshold_missing"
    assert missing_metric["violations"][0]["metric"] == "oot_ks"


def test_selection_policy_string_false_is_not_enabled():
    from marvis.packs.modeling.tools import _normalize_selection_policy

    policy = _normalize_selection_policy({
        "require_pmml": "false",
        "require_handoff": "0",
        "prefer_scorecard": "yes",
    })

    assert policy["require_pmml"] is False
    assert policy["require_handoff"] is False
    assert policy["prefer_scorecard"] is True


def test_selection_policy_normalizes_string_thresholds_and_rejects_nonfinite():
    from marvis.packs.modeling.tools import _normalize_selection_policy

    policy = _normalize_selection_policy({
        "max_feature_count": "30",
        "max_oot_psi": "0.15",
    })

    assert policy["max_feature_count"] == 30
    assert policy["max_oot_psi"] == 0.15
    assert "metric_thresholds" not in policy

    ignored = _normalize_selection_policy({
        "max_feature_count": "0",
        "max_oot_psi": "inf",
        "min_oot_ks": "nan",
        "metric_thresholds": {
            "oot_rmse": {"max": "bad"},
            "unsafe metric": {"min": 0.2},
        },
    })

    assert "max_feature_count" not in ignored
    assert "max_oot_psi" not in ignored
    assert "metric_thresholds" not in ignored

    thresholds = _normalize_selection_policy({
        "min_oot_ks": "0.31",
        "max_oot_rmse": "1.8",
        "metric_thresholds": {
            "oot_auc": {"min": "0.72"},
            "oot_logloss": {"max": 0.45},
        },
    })

    assert thresholds["metric_thresholds"] == {
        "oot_auc": {"min": 0.72},
        "oot_ks": {"min": 0.31},
        "oot_logloss": {"max": 0.45},
        "oot_rmse": {"max": 1.8},
    }


def test_approval_package_markdown_surfaces_configured_selection_policy_thresholds():
    from marvis.packs.modeling.tools import _approval_package_markdown

    markdown = _approval_package_markdown({
        "experiment_id": "exp-1",
        "artifact_id": "art-1",
        "algorithm": "lgb",
        "target_type": "binary",
        "target_col": "bad",
        "sample_dataset_id": "sample-1",
        "feature_count": 12,
        "sample_weight_col": "",
        "metrics": {"oot_ks": 0.34, "psi_oot_vs_train": 0.08},
        "selection_policy_decision": {
            "status": "accepted",
            "override_reason": "",
            "policy": {
                "require_pmml": True,
                "require_handoff": True,
                "max_feature_count": 30,
                "max_oot_psi": 0.15,
                "metric_thresholds": {
                    "oot_ks": {"min": 0.30},
                    "oot_rmse": {"max": 1.80},
                },
            },
            "violations": [],
        },
        "capabilities": {"pmml_supported": True, "handoff_supported": True},
        "monitoring_policy": {},
        "challenger_comparison": {},
        "delivery_actions": [],
        "artifacts": {},
        "features": ["x1", "x2"],
        "training": {},
    })

    assert "## 策略执行" in markdown
    assert "| 策略要求 | 配置 |" in markdown
    assert "| 要求 PMML | 是 |" in markdown
    assert "| 要求验证移交 | 是 |" in markdown
    assert "| 最大特征数 | 30 |" in markdown
    assert "| 最大 OOT PSI | 0.15 |" in markdown
    assert "| 指标 oot_ks | >= 0.3 |" in markdown
    assert "| 指标 oot_rmse | <= 1.8 |" in markdown


def test_make_split_tool_returns_sample_analysis_with_channel_distribution(tmp_path):
    """MODELING G1: make_split applies a channel/time rule set and returns a derived
    dataset plus a JSON-safe sample analysis (per-split counts + per-split x channel/month
    distribution table)."""
    runner, _pr, registry, _backend, _settings, task = _runtime(tmp_path)
    rows = 240
    channels = ["A", "B", "C"]
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)],
        "channel": [channels[i % 3] for i in range(rows)],
        "apply_month": [f"2026-{(i % 6) + 1:02d}" for i in range(rows)],
    })
    path = tmp_path / "make_split_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

    result = runner.invoke(
        ToolRef("modeling", "make_split"),
        {
            "dataset_id": dataset.id,
            "target_col": "y",
            "feature_cols": ["x1", "x2"],
            "split_config": {
                "rules": [
                    {"when": [{"col": "channel", "op": "eq", "val": "A"}], "assign": "train"},
                    {"when": [{"col": "channel", "op": "eq", "val": "C"}], "assign": "oot"},
                ],
                "test_size": 0.3,
            },
            "seed": 7,
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    out = result.output
    assert out["result_dataset_id"]
    analysis = out["sample_analysis"]
    assert analysis["total_rows"] == rows
    assert sum(analysis["split_counts"].values()) == rows
    assert set(analysis["split_counts"]) <= {"train", "test", "oot"}
    # channel + apply_month are both detected as group columns
    distributions = analysis["group_distributions"]
    assert set(distributions) == {"channel", "apply_month"}
    # channel A is wholly train, channel C is wholly oot (frozen by the rules)
    assert set(distributions["channel"].get("train", {})) == {"A"} or "A" in distributions["channel"].get("train", {})
    channel_train = distributions["channel"].get("train", {})
    channel_oot = distributions["channel"].get("oot", {})
    assert channel_train.get("A", 0) > 0 and "C" not in channel_train
    assert channel_oot.get("C", 0) > 0 and "A" not in channel_oot
    # JSON-safe: payload serializes under strict JSON (no NaN/Infinity tokens)
    import json

    json.dumps(analysis, allow_nan=False)


def test_continuous_screen_drops_constant_and_all_missing(tmp_path):
    """The modeling-pack non-binary screen mirrors the feature-pack one: constant and all-NaN
    columns land in `unusable`, not `selected`."""
    runner, _pr, registry, _backend, _settings, task = _runtime(tmp_path)
    rows = 60
    frame = pd.DataFrame({
        "good1": [i / rows for i in range(rows)],
        "good2": [(rows - i) / rows for i in range(rows)],
        "const": [3.0] * rows,
        "allnan": [float("nan")] * rows,
        "income": [1000.0 + i for i in range(rows)],
        "split": ["train"] * 36 + ["test"] * 12 + ["oot"] * 12,
    })
    path = tmp_path / "screen_unusable.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

    screened = runner.invoke(
        ToolRef("modeling", "screen_features"),
        {
            "dataset_id": dataset.id,
            "features": ["good1", "good2", "const", "allnan"],
            "target_col": "income",
            "split_col": "split",
            "target_type": "continuous",
        },
        task_id=task.id,
    )
    assert screened.ok is True, screened.error
    assert set(screened.output["selected"]) == {"good1", "good2"}
    reasons = {row[0]: row[1] for row in screened.output["unusable"]}
    assert reasons == {"const": "constant", "allnan": "high_missing"}

    capped = runner.invoke(
        ToolRef("modeling", "screen_features"),
        {
            "dataset_id": dataset.id,
            "features": ["good1", "good2", "const", "allnan"],
            "target_col": "income",
            "split_col": "split",
            "target_type": "continuous",
            "top_k": 1,
        },
        task_id=task.id,
    )
    assert capped.ok is True, capped.error
    assert len(capped.output["selected"]) == 1
    assert [row[0] for row in capped.output["ranked"]] == ["good1", "good2"]


def test_continuous_screen_uses_dev_rows_for_usability_stats(tmp_path):
    runner, _pr, registry, _backend, _settings, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "good_dev_missing_oot": [1.0, 2.0, 3.0, 4.0, None, None],
        "bad_dev_constant": [5.0, 5.0, 5.0, 5.0, 6.0, 7.0],
        "income": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
        "split": ["train", "train", "test", "test", "oot", "oot"],
    })
    path = tmp_path / "screen_holdout.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

    screened = runner.invoke(
        ToolRef("modeling", "screen_features"),
        {
            "dataset_id": dataset.id,
            "features": ["good_dev_missing_oot", "bad_dev_constant"],
            "target_col": "income",
            "split_col": "split",
            "target_type": "continuous",
        },
        task_id=task.id,
    )

    assert screened.ok is True, screened.error
    assert screened.output["selected"] == ["good_dev_missing_oot"]
    assert {row[0]: row[1] for row in screened.output["unusable"]} == {"bad_dev_constant": "constant"}
    assert screened.output["scores"]["good_dev_missing_oot"]["missing_rate"] == 0.0


def test_continuous_screen_then_train_models_regression_end_to_end(tmp_path):
    """End-to-end regression unblock: a continuous screen (target_type='continuous')
    does NOT crash and keeps every candidate, and train_models with lgb_regressor +
    target_type='continuous' yields RMSE/MAE/R2 (no KS/AUC)."""
    runner, _pr, registry, _backend, _settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)

    screened = runner.invoke(
        ToolRef("modeling", "screen_features"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "target_col": "income",
            "split_col": "split",
            "target_type": "continuous",
        },
        task_id=task.id,
    )
    assert screened.ok is True, screened.error
    assert set(screened.output["selected"]) == {"x1", "x2"}
    assert screened.output["leakage"] == []
    assert screened.output["note"] == "非二分类目标：跳过泄漏KS筛选，已剔除常量/高缺失列"

    trained = runner.invoke(
        ToolRef("modeling", "train_models"),
        {
            "dataset_id": dataset.id,
            "recipes": ["lgb_regressor"],
            "features": screened.output["selected"],
            "target_col": "income",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {},
            "seed": 23,
            "target_type": "continuous",
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error
    out = trained.output
    assert out["best_recipe"] == "lgb_regressor"
    best = next(
        exp for exp in out["experiments"] if exp["experiment_id"] == out["best_experiment_id"]
    )
    metrics = best["metrics"]
    assert metrics["test_rmse"] is not None
    assert metrics["test_rmse"] > 0
    assert metrics["test_mae"] is not None
    assert metrics["test_ks"] is None
    assert metrics["test_auc"] is None
