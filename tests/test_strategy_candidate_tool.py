from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.agent.renderers import render_tool_output
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


def _runtime(tmp_path: Path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(
        plugin_registry,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    task_repo = TaskRepository(settings.db_path)
    task = task_repo.create_task(
        TaskCreate(
            model_name="candidate-tool",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    other_task = task_repo.create_task(
        TaskCreate(
            model_name="other-task",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "other-source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "score": [100, 200, 300, 400, 500, 600, 700, 800],
            "income": [10, 20, 30, 40, 50, 60, 70, 80],
            "bad": [0, 0, 0, 1, 0, 1, 1, 1],
            "ead": [100, 200, 300, 400, 500, 600, 700, 800],
            "pd_12m": [0.01, 0.02, 0.03, 0.04, 0.10, 0.12, 0.14, 0.16],
            "utilization": [0.5] * 8,
            "unused": list("abcdefgh"),
        }
    )
    source = tmp_path / "candidate.parquet"
    frame.to_parquet(source, index=False)
    dataset = registry.register_existing(
        source,
        task_id=task.id,
        role="strategy_sample",
    )
    return runner, task, other_task, dataset


def _limit_inputs(dataset_id: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "target_col": "bad",
        "strategy_type": "limit",
        "candidate_design": {
            "method": "score_band_limit",
            "score_col": "score",
            "n_bands": 2,
            "limit_grid": [1000, 2000, 4000],
            "max_expected_loss_per_account": 100,
        },
        "economics_inputs": {
            "pd_col": "pd_12m",
            "lgd_value": 0.5,
            "utilization_col": "utilization",
        },
        "candidate_policy_version": "strategy.candidate_policy.v1",
    }


def test_candidate_tool_is_task_owned_minimal_and_deterministic(tmp_path: Path) -> None:
    runner, task, other_task, dataset = _runtime(tmp_path)
    inputs = _limit_inputs(dataset.id)

    first = runner.invoke(
        ToolRef("strategy", "design_strategy_candidate"),
        inputs,
        task_id=task.id,
    )
    repeated = runner.invoke(
        ToolRef("strategy", "design_strategy_candidate"),
        inputs,
        task_id=task.id,
    )

    assert first.ok, first.error
    assert repeated.ok, repeated.error
    assert first.output == repeated.output
    assert first.output["schema_version"] == "strategy.candidate_tool.v1"
    assert first.output["source_dataset_content_hash"] == (
        first.output["source_evidence"]["dataset_content_hash"]
    )
    assert first.output["source_evidence"]["columns"] == [
        "bad",
        "score",
        "pd_12m",
        "utilization",
    ]
    assert "unused" not in first.output["source_evidence"]["columns"]
    assert first.output["strategy_spec"]["strategy_type"] == "limit"

    cross_task = runner.invoke(
        ToolRef("strategy", "design_strategy_candidate"),
        inputs,
        task_id=other_task.id,
    )
    assert cross_task.ok is False
    assert "dataset not found" in (cross_task.error or "")


def test_candidate_renderer_surfaces_evidence_and_adoption_boundary(
    tmp_path: Path,
) -> None:
    runner, task, _other_task, dataset = _runtime(tmp_path)
    result = runner.invoke(
        ToolRef("strategy", "design_strategy_candidate"),
        _limit_inputs(dataset.id),
        task_id=task.id,
    )

    assert result.ok, result.error
    text, tables = render_tool_output("design_strategy_candidate", result.output)

    assert "确定性生成" in text
    assert "尚未采纳" in text
    assert "采纳仍需人工确认" in text
    assert result.output["source_dataset_content_hash"][:12] in text
    assert [table["title"] for table in tables] == ["确定性候选分箱与动作"]


def test_candidate_tool_schema_rejects_caller_results_and_fixed_pricing_pd(
    tmp_path: Path,
) -> None:
    runner, task, _other_task, dataset = _runtime(tmp_path)
    injected = runner.invoke(
        ToolRef("strategy", "design_strategy_candidate"),
        {**_limit_inputs(dataset.id), "strategy_spec": {}},
        task_id=task.id,
    )
    assert injected.ok is False
    assert injected.error_kind == "schema"

    fixed_pd = runner.invoke(
        ToolRef("strategy", "design_strategy_candidate"),
        {
            "dataset_id": dataset.id,
            "target_col": "bad",
            "strategy_type": "pricing",
            "candidate_design": {
                "method": "score_band_pricing",
                "score_col": "score",
                "rate_grid": [0.1, 0.2],
            },
            "economics_inputs": {
                "ead_col": "ead",
                "pd_value": 0.1,
                "lgd_value": 0.5,
                "funding_rate_value": 0.02,
                "term_months_value": 12,
                "operating_cost_per_loan_value": 0,
            },
            "candidate_policy_version": "strategy.candidate_policy.v1",
        },
        task_id=task.id,
    )
    assert fixed_pd.ok is False
    assert fixed_pd.error_kind == "schema"


def test_segmentation_candidate_threads_nullable_economics_through_backtest(
    tmp_path: Path,
) -> None:
    runner, task, _other_task, dataset = _runtime(tmp_path)
    designed = runner.invoke(
        ToolRef("strategy", "design_strategy_candidate"),
        {
            "dataset_id": dataset.id,
            "target_col": "bad",
            "strategy_type": "segmentation",
            "candidate_design": {
                "method": "single_variable_segmentation",
                "feature_col": "income",
                "n_bands": 3,
            },
            "candidate_policy_version": "strategy.candidate_policy.v1",
        },
        task_id=task.id,
    )
    assert designed.ok, designed.error
    assert designed.output["economics_inputs"] is None

    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_spec": designed.output["strategy_spec"]},
        task_id=task.id,
    )
    assert built.ok, built.error
    backtested = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {
            "dataset_id": dataset.id,
            "strategy_id": built.output["strategy_id"],
            "target_col": "bad",
            "economics_inputs": None,
        },
        task_id=task.id,
    )
    assert backtested.ok, backtested.error
    assert backtested.output["strategy_type"] == "segmentation"
