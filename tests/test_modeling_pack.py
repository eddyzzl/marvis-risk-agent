import sys
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
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
    handoff_tool = next(tool for tool in manifest.tools if tool.name == "handoff_to_validation")
    report_tool = next(tool for tool in manifest.tools if tool.name == "generate_model_report")

    assert tool_names == {
        "check_data_quality",
        "modeling_readiness",
        "prepare_modeling_frame",
        "select_features",
        "train_model",
        "compare_experiments",
        "export_pmml",
        "handoff_to_validation",
        "generate_model_report",
    }
    assert "reject_inference" not in tool_names
    assert train_tool.determinism == "stochastic"
    assert {"write:model", "write:dataset"} <= set(train_tool.side_effects)
    assert "write:task" in handoff_tool.side_effects
    assert "model_id" in report_tool.input_schema["properties"]
    assert "llm" in report_tool.side_effects


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
    exported = runner.invoke(
        ToolRef("modeling", "export_pmml"),
        {"artifact_id": train_outputs["lr"]["artifact_id"]},
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

    assert compared.ok is True, compared.error
    assert [row["recipe"] for row in compared.output["experiments"]] == [
        "lgb",
        "xgb",
        "lr",
        "scorecard",
    ]
    assert exported.ok is True, exported.error
    assert Path(exported.output["pmml_path"]).exists()
    assert handed_off.ok is True, handed_off.error
    validation_task = TaskRepository(settings.db_path).get_task(
        handed_off.output["validation_task_id"]
    )
    assert validation_task.task_type == "validation"
    assert validation_task.pmml_path == "model.pmml"


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
