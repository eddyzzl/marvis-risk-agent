"""Trusted challenger presentation and compare rendering.

The report accepts a persisted challenger-backtest receipt, reloads both
task-owned strategies and the bound dataset, and deterministically recomputes
the displayed evidence. Caller-supplied metrics or lifecycle claims are not an
input contract. With no champion it remains a no-artifact no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.agent.renderers import render_tool_output
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, StrategyRepository, init_db
from marvis.db_schema import connect
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
    load_builtin_packs(plugin_registry, Path(__file__).parents[1] / "marvis" / "packs")
    runner = ToolRunner(
        ToolRegistry(plugin_registry), plugin_repo, python_executable=sys.executable,
        datasets_root=settings.datasets_dir, workspace=settings.workspace,
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir
    )
    return runner, registry, settings


def _build_strategy(runner, *, threshold=2, task_id="task-1"):
    return runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval", "rules": [{"condition": f"score < {threshold}", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id=task_id,
    ).output["strategy_id"]


def _dataset(registry, tmp_path):
    path = tmp_path / "challenger.parquet"
    pd.DataFrame(
        {
            "score": [0, 1, 2, 3, 4, 5],
            "bad": [1, 1, 0, 0, 0, 0],
        }
    ).to_parquet(path, index=False)
    return registry.register_existing(
        path,
        task_id="task-1",
        role="strategy_sample",
    )


def _trusted_evidence(runner, registry, tmp_path):
    champion_id = _build_strategy(runner, threshold=2)
    strategy_id = _build_strategy(runner, threshold=3)
    dataset = _dataset(registry, tmp_path)
    champion_bt = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {
            "dataset_id": dataset.id,
            "strategy_id": champion_id,
            "target_col": "bad",
        },
        task_id="task-1",
    )
    assert champion_bt.ok is True, champion_bt.error
    challenger_bt = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {
            "dataset_id": dataset.id,
            "strategy_id": strategy_id,
            "baseline_strategy_id": champion_id,
            "target_col": "bad",
        },
        task_id="task-1",
    )
    assert challenger_bt.ok is True, challenger_bt.error
    return strategy_id, champion_id, champion_bt.output, challenger_bt.output


def test_challenger_report_recomputes_task_owned_persisted_evidence(tmp_path):
    runner, registry, settings = _runtime(tmp_path)
    strategy_id, champion_id, champion_bt, challenger_bt = _trusted_evidence(
        runner, registry, tmp_path
    )

    result = runner.invoke(
        ToolRef("strategy", "render_challenger_report"),
        {
            "strategy_id": strategy_id,
            "champion_strategy_id": champion_id,
            "challenger_backtest": challenger_bt,
        },
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["status"] == "rendered"
    md = result.output["report_md"]
    assert f"{float(challenger_bt['approval_rate']):.4f}" in md
    assert f"{float(champion_bt['approval_rate']):.4f}" in md
    assert "未采纳（仍以基线为准）" in md
    assert "挑战者审批率较基线" in md
    assert [a["kind"] for a in result.output["artifacts"]] == ["challenger_report_md"]

    strategies = StrategyRepository(settings.db_path)
    assert [a["kind"] for a in strategies.list_strategy_artifacts(strategy_id)] == ["challenger_report_md"]
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT detail_json FROM audit WHERE kind='strategy.artifact' "
            "AND detail_json LIKE '%challenger_report_md%'"
        ).fetchall()
    assert len(rows) == 1


def test_challenger_report_rejects_caller_fabricated_metrics_and_status(tmp_path):
    runner, registry, _settings = _runtime(tmp_path)
    strategy_id, champion_id, _champion_bt, challenger_bt = _trusted_evidence(
        runner, registry, tmp_path
    )
    result = runner.invoke(
        ToolRef("strategy", "render_challenger_report"),
        {
            "strategy_id": strategy_id,
            "champion_strategy_id": champion_id,
            "challenger_backtest": challenger_bt,
            "compare": {"deltas": {"expected_profit": 999999}},
            "champion_backtest": {"approval_rate": 1.0},
            "adopted": True,
        },
        task_id="task-1",
    )
    assert result.ok is False
    assert result.error_kind == "schema"


@pytest.mark.slow
def test_challenger_report_no_champion_degrades_to_no_op(tmp_path):
    runner, _registry, settings = _runtime(tmp_path)
    strategy_id = _build_strategy(runner)

    result = runner.invoke(
        ToolRef("strategy", "render_challenger_report"),
        {"strategy_id": strategy_id},
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["status"] == "no_baseline"
    assert result.output["artifacts"] == []
    assert "未提供基线" in result.output["report_md"]
    strategies = StrategyRepository(settings.db_path)
    assert strategies.list_strategy_artifacts(strategy_id) == []


def test_challenger_report_rejects_cross_task_champion_before_writing(tmp_path):
    runner, _registry, settings = _runtime(tmp_path)
    strategy_id = _build_strategy(runner, task_id="task-1")
    foreign_champion_id = _build_strategy(runner, threshold=3, task_id="task-2")
    result = runner.invoke(
        ToolRef("strategy", "render_challenger_report"),
        {
            "strategy_id": strategy_id,
            "champion_strategy_id": foreign_champion_id,
            "challenger_backtest": {"backtest_id": "not-used"},
        },
        task_id="task-1",
    )
    assert result.ok is False
    # The ownership boundary deliberately does not reveal that the foreign id
    # exists; it is indistinguishable from an unknown strategy to this task.
    assert "strategy not found" in str(result.error)
    assert StrategyRepository(settings.db_path).list_strategy_artifacts(strategy_id) == []


def test_compare_renderer_uses_matrix_heat_and_conclusion_line():
    out = {
        "matrix_2x2": {
            "both_approve": {"count": 10, "bad_rate": 0.05},
            "only_new": {"count": 3, "bad_rate": 0.10},
            "only_baseline": {"count": 2, "bad_rate": 0.20},
            "both_decline": {"count": 5, "bad_rate": 0.40},
        },
        "deltas": {"approval_rate": 0.02, "approved_bad_rate": -0.01, "expected_profit": 15.0},
        "summary_text": "新策略审批率较基线上升2.0pp。",
        "red_flags": [],
    }
    text, tables = render_tool_output("compare_strategies", out)

    # Templated conclusion, numbers from deltas (INV-1 presentation only).
    assert "结论：挑战者在通过率上升 2.0pp" in text
    assert "通过客群坏率下降 1.00pp" in text
    # 2x2 swap uses matrix-heat column specs; cell heat = each cell's bad_rate.
    heat = tables[0]
    assert heat["column_specs"] == [{"kind": "text"}, {"kind": "matrix-heat"}, {"kind": "matrix-heat"}]
    assert heat["rows"][0][1] == 0.05  # both_approve bad rate as heat value
    assert heat["rows"][0][2] == 0.10  # only_new bad rate as heat value
    # side-by-side key metrics with a direction column.
    metrics = tables[1]
    assert metrics["columns"] == ["指标", "挑战者−基线", "方向"]
