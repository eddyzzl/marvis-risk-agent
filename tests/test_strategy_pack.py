import sys
from pathlib import Path

import pandas as pd
import pytest

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
            model_name="策略能力包样例",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            target_col="bad",
            score_col="score",
            split_col="split",
            time_col="month",
            feature_columns=["score", "segment"],
        )
    )
    return runner, plugin_registry, registry, task


def _register_strategy_sample(registry, tmp_path, task_id: str):
    frame = pd.DataFrame({
        "customer_id": ["A", "A", "B", "B", "C", "C"],
        "month": ["2026-01", "2026-02", "2026-01", "2026-02", "2026-01", "2026-02"],
        "status": ["C", "M1", "C", "C", "M3+", "M3+"],
        "cohort": ["202601", "202601", "202602", "202602", "202603", "202603"],
        "mob": [0, 1, 0, 1, 0, 1],
        "bad": [1, 1, 0, 0, 1, 1],
        "score": [580, 620, 730, 760, 590, 800],
        "ead": [1000.0, 2000.0, 1000.0, 500.0, 1000.0, 800.0],
        "pd": [0.20, 0.05, 0.02, 0.10, 0.15, 0.03],
        "segment": ["A", "A", "B", "B", "A", "B"],
    })
    path = tmp_path / "strategy_sample.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def test_strategy_manifest_registers_expected_tools(tmp_path):
    _runner, plugin_registry, _registry, _task = _runtime(tmp_path)

    manifest = plugin_registry.get("strategy")
    tool_names = {tool.name for tool in manifest.tools}
    build_tool = next(tool for tool in manifest.tools if tool.name == "build_strategy")
    backtest_tool = next(tool for tool in manifest.tools if tool.name == "backtest_strategy")

    assert tool_names == {
        "vintage_curve",
        "roll_rate_matrix",
        "profit_calc",
        "build_strategy",
        "backtest_strategy",
        "tradeoff_view",
        "design_cutoff_bands",
        "compare_strategies",
        "adopt_strategy",
        "render_strategy_doc",
        "mine_rules",
        "evaluate_rule_set",
        "select_rule_set",
        "limit_pricing_matrix",
        "render_challenger_report",
        "run_strategy_monitoring",
        "render_monitoring_report",
    }
    assert build_tool.determinism == "deterministic"
    assert "write:strategy" in build_tool.side_effects
    assert "write:backtest" in backtest_tool.side_effects


@pytest.mark.slow
def test_strategy_pack_tools_round_trip_via_runner(tmp_path):
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    dataset = _register_strategy_sample(registry, tmp_path, task.id)
    params = {
        "annual_rate": 0.12,
        "funding_rate": 0.03,
        "lgd": 0.5,
        "operating_cost_per_loan": 10.0,
        "term_months": 6,
    }

    vintage = runner.invoke(
        ToolRef("strategy", "vintage_curve"),
        {
            "dataset_id": dataset.id,
            "cohort_col": "cohort",
            "mob_col": "mob",
            "bad_col": "bad",
            "mob_max": 2,
        },
        task_id=task.id,
    )
    roll = runner.invoke(
        ToolRef("strategy", "roll_rate_matrix"),
        {
            "dataset_id": dataset.id,
            "id_col": "customer_id",
            "time_col": "month",
            "status_col": "status",
            "states": ["C", "M1", "M3+"],
        },
        task_id=task.id,
    )
    profit = runner.invoke(
        ToolRef("strategy", "profit_calc"),
        {
            "dataset_id": dataset.id,
            "segment_col": "segment",
            "ead_col": "ead",
            "pd_col": "pd",
            "params": params,
        },
        task_id=task.id,
    )
    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {
            "strategy_type": "approval",
            "rules": [{"condition": "score < 600", "decision": "reject"}],
            "score_col": "score",
            "default_decision": "approve",
            "description": "reject low scores",
        },
        task_id=task.id,
    )

    assert vintage.ok is True, vintage.error
    assert vintage.output["cohorts"] == ["2026-01", "2026-02", "2026-03"]
    assert vintage.output["summary"]["trend"] in {"deteriorating", "stable", "improving"}
    assert roll.ok is True, roll.error
    assert roll.output["base_counts"] == {"C": 2, "M1": 0, "M3+": 1}
    assert profit.ok is True, profit.error
    assert {row["segment"] for row in profit.output["results"]} == {"A", "B"}
    assert built.ok is True, built.error
    assert built.output["strategy_id"]

    backtest = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {
            "dataset_id": dataset.id,
            "strategy_id": built.output["strategy_id"],
            "target_col": "bad",
            "profit_params": params,
            "ead_col": "ead",
            "pd_col": "pd",
        },
        task_id=task.id,
    )
    tradeoff = runner.invoke(
        ToolRef("strategy", "tradeoff_view"),
        {
            "dataset_id": dataset.id,
            "score_col": "score",
            "target_col": "bad",
            "cutoffs": [600, 700],
            "profit_params": params,
            "ead_col": "ead",
            "pd_col": "pd",
            "max_bad_rate": 0.7,
            "objective": "max_profit",
        },
        task_id=task.id,
    )

    assert backtest.ok is True, backtest.error
    assert backtest.output["backtest_id"]
    assert backtest.output["approval_rate"] == pytest.approx(4 / 6)
    assert "by_segment" in backtest.output
    strategy_audits = PluginRepository(
        build_settings(tmp_path / "workspace").db_path
    ).list_audit()
    assert any(
        audit["kind"] == "strategy.create"
        and audit["target_ref"] == built.output["strategy_id"]
        for audit in strategy_audits
    )
    assert any(
        audit["kind"] == "strategy.backtest"
        and audit["target_ref"] == backtest.output["backtest_id"]
        for audit in strategy_audits
    )
    assert tradeoff.ok is True, tradeoff.error
    assert [point["cutoff"] for point in tradeoff.output["points"]] == [600.0, 700.0]
    assert tradeoff.output["recommended"]["cutoff"] in {600.0, 700.0}


def test_roll_rate_matrix_tool_surfaces_balance_weighting_and_warnings(tmp_path):
    # DOM-8: balance_col weights transitions; a missing-month gap for one id
    # surfaces as a data_quality_warnings entry through the tool boundary.
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "customer_id": ["A", "A", "B", "B"],
        "month": ["202601", "202603", "202601", "202602"],
        "status": ["C", "M1", "C", "C"],
        "balance": [100.0, 100.0, 300.0, 300.0],
    })
    path = tmp_path / "roll_rate_balance_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="strategy_sample")

    result = runner.invoke(
        ToolRef("strategy", "roll_rate_matrix"),
        {
            "dataset_id": dataset.id,
            "id_col": "customer_id",
            "time_col": "month",
            "status_col": "status",
            "states": ["C", "M1"],
            "balance_col": "balance",
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    assert result.output["base_counts"] == {"C": 400.0, "M1": 0.0}
    assert len(result.output["data_quality_warnings"]) == 1
    assert result.output["data_quality_warnings"][0]["id"] == "A"


def _register_strategy_sample_with_nan_label(registry, tmp_path, task_id: str):
    frame = pd.DataFrame({
        "bad": [1.0, 0.0, float("nan"), 0.0, 1.0, 0.0],
        "score": [580, 620, 730, 760, 590, 800],
    })
    path = tmp_path / "strategy_nan_sample.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def test_tradeoff_view_gates_nan_label(tmp_path):
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    dataset = _register_strategy_sample_with_nan_label(registry, tmp_path, task.id)
    base_inputs = {
        "dataset_id": dataset.id,
        "score_col": "score",
        "target_col": "bad",
        "cutoffs": [600, 700],
    }

    blocked = runner.invoke(ToolRef("strategy", "tradeoff_view"), dict(base_inputs), task_id=task.id)
    assert blocked.ok is False
    assert blocked.error_kind == "nan_label_not_confirmed"
    assert blocked.error_detail["n_nan"] == 1

    confirmed = runner.invoke(
        ToolRef("strategy", "tradeoff_view"),
        {**base_inputs, "drop_nan_labels": True},
        task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["nan_labels_dropped"] == 1


def test_vintage_curve_gates_nan_label(tmp_path):
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "cohort": ["202601", "202601", "202602"],
        "mob": [0, 1, 0],
        "bad": [0.0, float("nan"), 1.0],
    })
    path = tmp_path / "vintage_nan_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="strategy_sample")
    base_inputs = {
        "dataset_id": dataset.id,
        "cohort_col": "cohort",
        "mob_col": "mob",
        "bad_col": "bad",
    }

    blocked = runner.invoke(ToolRef("strategy", "vintage_curve"), dict(base_inputs), task_id=task.id)
    assert blocked.ok is False
    assert blocked.error_kind == "nan_label_not_confirmed"
    assert blocked.error_detail["n_nan"] == 1

    confirmed = runner.invoke(
        ToolRef("strategy", "vintage_curve"),
        {**base_inputs, "drop_nan_labels": True},
        task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["nan_labels_dropped"] == 1
