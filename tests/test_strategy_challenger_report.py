"""S6 Commit 3: challenger presentation — report tool + compare renderer.

The report is assembled purely from the passed-in compare + backtest tool outputs
(report follows tool output: change the numbers in, the numbers in the report change).
With no champion the report degrades to a no-op (no artifact) mirroring
compare_strategies' own no-baseline degradation.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def _build_strategy(runner):
    return runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval", "rules": [{"condition": "score < 2", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id="task-1",
    ).output["strategy_id"]


def test_challenger_report_numbers_follow_tool_output(tmp_path):
    runner, _registry, settings = _runtime(tmp_path)
    strategy_id = _build_strategy(runner)
    compare = {
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
    challenger_bt = {"approval_rate": 0.62, "approved_bad_rate": 0.04, "expected_profit": 115.0}
    champion_bt = {"approval_rate": 0.60, "approved_bad_rate": 0.05, "expected_profit": 100.0}

    result = runner.invoke(
        ToolRef("strategy", "render_challenger_report"),
        {
            "strategy_id": strategy_id,
            "champion_strategy_id": "champion-1",
            "compare": compare,
            "challenger_backtest": challenger_bt,
            "champion_backtest": champion_bt,
            "adopted": True,
        },
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["status"] == "rendered"
    md = result.output["report_md"]
    # Numbers come straight from the passed-in tool outputs.
    assert "0.6200" in md and "0.6000" in md  # challenger vs champion approval
    assert "115.0000" in md and "100.0000" in md  # expected profit both sides
    assert "0.0200" in md  # approval delta
    assert "已采纳挑战者" in md
    assert compare["summary_text"] in md
    assert [a["kind"] for a in result.output["artifacts"]] == ["challenger_report_md"]

    strategies = StrategyRepository(settings.db_path)
    assert [a["kind"] for a in strategies.list_strategy_artifacts(strategy_id)] == ["challenger_report_md"]
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT detail_json FROM audit WHERE kind='strategy.artifact' "
            "AND detail_json LIKE '%challenger_report_md%'"
        ).fetchall()
    assert len(rows) == 1


def test_challenger_report_changes_when_compare_numbers_change(tmp_path):
    runner, _registry, _settings = _runtime(tmp_path)
    strategy_id = _build_strategy(runner)

    def _render(profit_delta):
        return runner.invoke(
            ToolRef("strategy", "render_challenger_report"),
            {
                "strategy_id": strategy_id,
                "champion_strategy_id": "champion-1",
                "compare": {
                    "deltas": {"approval_rate": 0.0, "approved_bad_rate": 0.0, "expected_profit": profit_delta},
                    "summary_text": "x",
                    "red_flags": [],
                },
                "challenger_backtest": {"expected_profit": profit_delta},
                "champion_backtest": {"expected_profit": 0.0},
            },
            task_id="task-1",
        ).output["report_md"]

    assert "42.0000" in _render(42.0)
    assert "99.0000" in _render(99.0)


def test_challenger_report_no_champion_degrades_to_no_op(tmp_path):
    runner, _registry, settings = _runtime(tmp_path)
    strategy_id = _build_strategy(runner)

    result = runner.invoke(
        ToolRef("strategy", "render_challenger_report"),
        {"strategy_id": strategy_id, "compare": {}, "challenger_backtest": {}},
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["status"] == "no_baseline"
    assert result.output["artifacts"] == []
    assert "未提供基线" in result.output["report_md"]
    strategies = StrategyRepository(settings.db_path)
    assert strategies.list_strategy_artifacts(strategy_id) == []


def test_challenger_report_degrades_when_compare_itself_degraded(tmp_path):
    runner, _registry, _settings = _runtime(tmp_path)
    strategy_id = _build_strategy(runner)
    # compare_strategies' own no-baseline no-op carries this summary_text.
    degraded_compare = {"summary_text": "未提供基线策略，跳过对比。", "deltas": {}, "red_flags": []}

    result = runner.invoke(
        ToolRef("strategy", "render_challenger_report"),
        {"strategy_id": strategy_id, "champion_strategy_id": "champion-1", "compare": degraded_compare},
        task_id="task-1",
    )
    assert result.output["status"] == "no_baseline"
    assert result.output["artifacts"] == []


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
