import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy import build_strategy
from marvis.packs.strategy.bands import design_cutoff_bands
from marvis.packs.strategy.compare import compare_strategies
from marvis.packs.strategy.tradeoff import tradeoff_feasible_flags, tradeoff_view
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


# ---------------------------------------------------------------------------
# design_cutoff_bands core: hand-computed values on a tiny 6-row set.
# ---------------------------------------------------------------------------
def _hand_frame() -> pd.DataFrame:
    # higher_is_better: high score = good; the two lowest-score rows are bad.
    return pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 600],
        "bad":   [1,   1,   0,   0,   0,   0],
    })


def test_design_cutoff_bands_hand_computed_values():
    res = design_cutoff_bands(
        _hand_frame(),
        score_col="score",
        target_col="bad",
        score_direction="higher_is_better",
        band_edges=[100, 300, 500, 600],  # bands [100,300) [300,500) [500,600]
        objective="max_approval",
        max_bad_rate=0.3,
    )
    bands = res.bands
    assert [b.count for b in bands] == [2, 2, 2]
    assert [round(b.pop_pct, 4) for b in bands] == [0.3333, 0.3333, 0.3333]
    assert [round(b.bad_rate, 4) for b in bands] == [1.0, 0.0, 0.0]
    # approval order (higher_is_better) is top-down: band2 -> band1 -> band0.
    assert [round(b.cum_approval_rate, 4) for b in bands] == [1.0, 0.6667, 0.3333]
    assert [round(b.cum_bad_rate, 4) for b in bands] == [0.3333, 0.0, 0.0]
    # max_bad_rate=0.3 forbids approving band0 (would push cum bad to 0.333),
    # so the feasible cut approves the two clean top bands only.
    assert [b.decision for b in bands] == ["decline", "approve", "approve"]
    assert res.recommended_rules == ({"condition": "score < 300", "decision": "reject"},)
    assert res.red_flags == ()


def test_design_cutoff_bands_is_deterministic():
    frame = _hand_frame()
    kwargs = dict(
        score_col="score", target_col="bad", score_direction="higher_is_better",
        band_edges=[100, 300, 500, 600], objective="max_profit",
    )
    a = design_cutoff_bands(frame, **kwargs)
    b = design_cutoff_bands(frame, **kwargs)
    assert [x.__dict__ for x in a.bands] == [x.__dict__ for x in b.bands]
    assert a.band_edges == b.band_edges


def test_design_cutoff_bands_manual_band_edges_override():
    frame = _hand_frame()
    res = design_cutoff_bands(
        frame, score_col="score", target_col="bad",
        score_direction="higher_is_better", band_edges=[100, 350, 600],
    )
    assert res.band_edges == (100.0, 350.0, 600.0)
    assert len(res.bands) == 2
    # [100,350) -> rows 100,200,300 -> 2 bad of 3; [350,600] -> 3 rows, 0 bad.
    assert [b.count for b in res.bands] == [3, 3]
    assert round(res.bands[0].bad_rate, 4) == round(2 / 3, 4)


def test_design_cutoff_bands_flags_nonmonotonic_bad_rate():
    frame = pd.DataFrame({
        "score": [100, 150, 300, 350, 500, 550],
        "bad":   [0,   0,   1,   1,   0,   1],
    })
    res = design_cutoff_bands(
        frame, score_col="score", target_col="bad",
        score_direction="higher_is_better", band_edges=[100, 300, 500, 550],
        objective="max_profit",
    )
    codes = {f.code for f in res.red_flags}
    assert "nonmonotonic_bad_rate" in codes


def test_design_cutoff_bands_flags_sparse_band():
    frame = pd.DataFrame({"score": list(range(1, 101)), "bad": [0] * 100})
    res = design_cutoff_bands(
        frame, score_col="score", target_col="bad",
        score_direction="higher_is_better", band_edges=[1, 50, 100, 101],
    )
    # top band [100,101] holds a single row -> 1% < 2%.
    assert res.bands[-1].count == 1
    assert any(f.code == "sparse_band" for f in res.red_flags)


def test_design_cutoff_bands_flags_infeasible_constraints():
    frame = pd.DataFrame({"score": [100, 200, 300, 400], "bad": [1, 1, 1, 1]})
    res = design_cutoff_bands(
        frame, score_col="score", target_col="bad",
        score_direction="higher_is_better", band_edges=[100, 200, 300, 400],
        objective="max_approval", max_bad_rate=0.1,
    )
    assert any(f.code == "infeasible_constraints" and f.level == "red" for f in res.red_flags)


def test_design_cutoff_bands_higher_is_riskier_order_and_rule():
    # higher_is_riskier: low score = good. bad concentrated in high scores.
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 600],
        "bad":   [0,   0,   0,   0,   1,   1],
    })
    res = design_cutoff_bands(
        frame, score_col="score", target_col="bad",
        score_direction="higher_is_riskier", band_edges=[100, 300, 500, 600],
        objective="max_approval", max_bad_rate=0.3,
    )
    # approval order is bottom-up: band0 -> band1 -> band2.
    assert [round(b.cum_approval_rate, 4) for b in res.bands] == [0.3333, 0.6667, 1.0]
    assert [b.decision for b in res.bands] == ["approve", "approve", "decline"]
    assert res.recommended_rules == ({"condition": "score >= 500", "decision": "reject"},)


# ---------------------------------------------------------------------------
# tradeoff feasibility upgrade.
# ---------------------------------------------------------------------------
def test_tradeoff_feasible_flags_filter():
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 600],
        "bad":   [1,   1,   0,   0,   0,   0],
    })
    points = tradeoff_view(
        frame, score_col="score", target_col="bad",
        cutoffs=[300, 500], score_direction="higher_is_better",
    )
    # cutoff 300 approves scores>=300 (4 rows, 0 bad); cutoff 500 approves 2 rows.
    flags = tradeoff_feasible_flags(points, max_bad_rate=0.0, min_approval_rate=0.5)
    # both have bad_rate 0 (feasible on bad); approval 4/6=.667 and 2/6=.333.
    assert flags == [True, False]


def test_tradeoff_view_all_infeasible_returns_none_recommended():
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400],
        "bad":   [1,   1,   1,   1],
    })
    points = tradeoff_view(
        frame, score_col="score", target_col="bad",
        cutoffs=[200, 300], score_direction="higher_is_better",
    )
    flags = tradeoff_feasible_flags(points, max_bad_rate=0.1)
    assert not any(flags)


# ---------------------------------------------------------------------------
# compare_strategies 2x2 + swap_in_worse.
# ---------------------------------------------------------------------------
def test_compare_strategies_matrix_hand_computed():
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 600],
        "bad":   [1,   0,   1,   0,   0,   0],
    })
    # new: reject score < 250 (approves 300..600 -> 4 rows)
    new = build_strategy(
        "approval", [{"condition": "score < 250", "decision": "reject"}],
        score_col="score", default_decision="approve", description="new",
    )
    # baseline: reject score < 450 (approves 500,600 -> 2 rows)
    baseline = build_strategy(
        "approval", [{"condition": "score < 450", "decision": "reject"}],
        score_col="score", default_decision="approve", description="base",
    )
    res = compare_strategies(frame, new, baseline, target_col="bad")
    m = res.matrix_2x2
    # both approve: score>=250 AND score>=450 -> 500,600 -> 2 rows, 0 bad.
    assert m["both_approve"].count == 2
    assert m["both_approve"].bad_rate == 0.0
    # only new: approved by new not baseline -> 300,400 -> 2 rows, 1 bad (300).
    assert m["only_new"].count == 2
    assert round(m["only_new"].bad_rate, 4) == 0.5
    # only baseline: none (baseline stricter) -> 0.
    assert m["only_baseline"].count == 0
    # both decline: score<250 -> 100,200 -> 2 rows.
    assert m["both_decline"].count == 2
    # approval delta: new 4/6 - base 2/6 = 2/6.
    assert round(res.deltas["approval_rate"], 4) == round(2 / 6, 4)


def test_compare_strategies_swap_in_worse_triggers_when_swapin_riskier():
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400],
        "bad":   [0,   1,   0,   0],
    })
    # new approves the low/mid band (reject high scores); single-direction rule.
    new = build_strategy(
        "approval", [{"condition": "score >= 350", "decision": "reject"}],
        score_col="score", default_decision="approve", description="new",
    )
    # baseline approves only the top score (reject the rest); single-direction rule.
    baseline = build_strategy(
        "approval", [{"condition": "score < 350", "decision": "reject"}],
        score_col="score", default_decision="approve", description="base",
    )
    res = compare_strategies(frame, new, baseline, target_col="bad")
    m = res.matrix_2x2
    # only_new: 100,200(bad),300 -> swap-in bad_rate 1/3; only_baseline: 400 -> 0.
    assert m["only_new"].count == 3
    assert m["only_baseline"].count == 1
    assert round(m["only_new"].bad_rate, 4) == round(1 / 3, 4)
    assert round(m["only_baseline"].bad_rate, 4) == 0.0
    assert "swap_in_worse" in {f["code"] for f in res.red_flags}


def test_compare_strategies_empty_swap_cell_is_none_not_zero():
    # DOM-11: an empty swap cell (no rows fall in it) has no defined bad rate --
    # it must be None, not the misleading 0.0, and must not crash swap_in_worse.
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400],
        "bad":   [0,   1,   0,   0],
    })
    # new and baseline agree on every row -> only_new/only_baseline are both empty.
    same = build_strategy(
        "approval", [{"condition": "score < 250", "decision": "reject"}],
        score_col="score", default_decision="approve", description="same",
    )
    res = compare_strategies(frame, same, same, target_col="bad")
    m = res.matrix_2x2
    assert m["only_new"].count == 0
    assert m["only_new"].bad_rate is None
    assert m["only_baseline"].count == 0
    assert m["only_baseline"].bad_rate is None
    # empty cells must not spuriously trigger swap_in_worse (no > comparison on None).
    assert "swap_in_worse" not in {f["code"] for f in res.red_flags}


# ---------------------------------------------------------------------------
# Tool boundary: nan gate + direction conflict via the runner.
# ---------------------------------------------------------------------------
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
            model_name="S2 策略开发",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            target_col="bad",
            score_col="score",
        )
    )
    return runner, registry, task


def _register(registry, tmp_path, frame: pd.DataFrame, name: str, task_id: str):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


@pytest.mark.slow
def test_tool_design_cutoff_bands_gates_nan_label(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 600],
        "bad":   [1.0, 0.0, float("nan"), 0.0, 0.0, 0.0],
    })
    dataset = _register(registry, tmp_path, frame, "nan_bands", task.id)
    base = {
        "dataset_id": dataset.id, "score_col": "score", "target_col": "bad",
        "score_direction": "higher_is_better", "band_edges": [100, 300, 600],
    }
    blocked = runner.invoke(ToolRef("strategy", "design_cutoff_bands"), dict(base), task_id=task.id)
    assert blocked.ok is False
    assert blocked.error_kind == "nan_label_not_confirmed"

    confirmed = runner.invoke(
        ToolRef("strategy", "design_cutoff_bands"),
        {**base, "drop_nan_labels": True},
        task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["nan_labels_dropped"] == 1
    assert any(f["code"] == "nan_labels_dropped" for f in confirmed.output["red_flags"])


@pytest.mark.slow
def test_tool_design_cutoff_bands_direction_conflict_gate(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    # Declare higher_is_better but make high scores clearly riskier (positive corr).
    n = 60
    scores = list(range(1, n + 1))
    bad = [0] * (n // 2) + [1] * (n - n // 2)
    frame = pd.DataFrame({"score": scores, "bad": bad})
    dataset = _register(registry, tmp_path, frame, "conflict_bands", task.id)
    base = {
        "dataset_id": dataset.id, "score_col": "score", "target_col": "bad",
        "score_direction": "higher_is_better", "n_bands": 4,
    }
    blocked = runner.invoke(ToolRef("strategy", "design_cutoff_bands"), dict(base), task_id=task.id)
    assert blocked.ok is False
    assert blocked.error_kind == "score_direction_conflict"

    confirmed = runner.invoke(
        ToolRef("strategy", "design_cutoff_bands"),
        {**base, "confirm_direction_conflict": True},
        task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert any(f["code"] == "direction_conflict" for f in confirmed.output["red_flags"])


@pytest.mark.slow
def test_tool_compare_strategies_round_trip(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 600],
        "bad":   [1,   0,   1,   0,   0,   0],
    })
    dataset = _register(registry, tmp_path, frame, "cmp", task.id)
    new = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval",
         "rules": [{"condition": "score < 250", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id=task.id,
    )
    base = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval",
         "rules": [{"condition": "score < 450", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id=task.id,
    )
    cmp = runner.invoke(
        ToolRef("strategy", "compare_strategies"),
        {"dataset_id": dataset.id, "target_col": "bad",
         "strategy_id": new.output["strategy_id"],
         "baseline_strategy_id": base.output["strategy_id"]},
        task_id=task.id,
    )
    assert cmp.ok is True, cmp.error
    assert cmp.output["matrix_2x2"]["both_approve"]["count"] == 2
    assert "summary_text" in cmp.output
    assert isinstance(cmp.output["red_flags"], list)
    # No NaN labels in this frame -> full coverage (DOM-11).
    assert cmp.output["label_coverage"] == 1.0
    _ = PluginRepository  # keep import used across slow/fast paths


@pytest.mark.slow
def test_tool_backtest_and_compare_strategies_label_coverage_hand_computed(tmp_path):
    # DOM-11: label_coverage = labeled rows / total rows under drop_nan_labels
    # semantics. 6 rows, 2 NaN labels -> coverage = 4/6.
    runner, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 600],
        "bad":   [1.0, 0.0, float("nan"), 0.0, 0.0, float("nan")],
    })
    dataset = _register(registry, tmp_path, frame, "coverage", task.id)
    new = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval",
         "rules": [{"condition": "score < 250", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id=task.id,
    )
    base = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval",
         "rules": [{"condition": "score < 450", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id=task.id,
    )

    backtest = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {"dataset_id": dataset.id, "strategy_id": new.output["strategy_id"],
         "target_col": "bad", "drop_nan_labels": True},
        task_id=task.id,
    )
    assert backtest.ok is True, backtest.error
    assert backtest.output["nan_labels_dropped"] == 2
    assert round(backtest.output["label_coverage"], 4) == round(4 / 6, 4)

    cmp = runner.invoke(
        ToolRef("strategy", "compare_strategies"),
        {"dataset_id": dataset.id, "target_col": "bad",
         "strategy_id": new.output["strategy_id"],
         "baseline_strategy_id": base.output["strategy_id"],
         "drop_nan_labels": True},
        task_id=task.id,
    )
    assert cmp.ok is True, cmp.error
    assert cmp.output["nan_labels_dropped"] == 2
    assert round(cmp.output["label_coverage"], 4) == round(4 / 6, 4)


# ---------------------------------------------------------------------------
# Deliverables + doc cores (fast).
# ---------------------------------------------------------------------------
def test_decision_table_csv_content():
    from marvis.packs.strategy.deliverables import decision_table_csv

    rules = [{"condition": "score < 300", "decision": "reject", "value": None}]
    bands = [
        {"lo": 100, "hi": 300, "pop_pct": 0.3333, "bad_rate": 1.0, "expected_profit": -50.0},
    ]
    csv_text = decision_table_csv(rules, bands)
    lines = csv_text.strip().splitlines()
    assert lines[0] == "序号,条件,决策,取值,band区间,样本占比,坏率,预期利润"
    assert "score < 300" in lines[1]
    assert "[100,300)" in lines[1]
    assert "33.33%" in lines[1]
    assert "100.00%" in lines[1]
    assert "-50.00" in lines[1]


def test_build_monitoring_plan_shape():
    from marvis.packs.strategy.deliverables import build_monitoring_plan

    plan = build_monitoring_plan(
        strategy_id="s-1", version=2, approved_bad_rate=0.04, approval_rate=0.7
    )
    assert plan["strategy_id"] == "s-1"
    assert plan["version"] == 2
    thresholds = plan["thresholds"]
    assert set(thresholds) == {"approved_bad_rate", "approval_rate"}
    assert thresholds["approved_bad_rate"]["direction"] == "max"
    assert round(thresholds["approved_bad_rate"]["warn"], 4) == 0.06
    assert thresholds["approval_rate"]["direction"] == "min"


def test_render_strategy_doc_markdown_sections():
    from marvis.packs.strategy.doc import render_strategy_doc_markdown

    markdown, sections = render_strategy_doc_markdown(
        strategy={
            "id": "s-1",
            "strategy_type": "approval",
            "rules": [{"condition": "score < 600", "decision": "reject", "value": None}],
            "default_decision": "approve",
        },
        meta={"version": 2, "status": "adopted", "parent_strategy_id": "s-0",
              "adopted_at": "2026-06-20T00:00:00Z", "adoption_reason": "committee"},
        backtests=[{"approval_rate": 0.7, "approved_bad_rate": 0.04, "rejected_bad_rate": 0.2,
                    "expected_profit": 100.0, "swap_in_count": 5, "swap_out_count": 3,
                    "swap_in_bad_rate": 0.1, "swap_out_bad_rate": 0.02}],
        artifacts=[{"kind": "monitoring_plan_json", "path": "workspace/tasks/t/strategy/mon.json"}],
        band_stats=[{"lo": 100, "hi": 600, "pop_pct": 0.5, "bad_rate": 0.2,
                     "cum_approval_rate": 0.5, "cum_bad_rate": 0.2, "decision": "approve"}],
        red_flags=[{"level": "amber", "code": "sparse_band", "message": "示例"}],
    )
    assert sections == ["策略概览", "规则清单", "回测摘要", "分数带", "红旗与处置记录", "监控计划摘要"]
    for heading in ("## 策略概览", "## 规则清单", "## 回测摘要", "## 分数带",
                    "## 红旗与处置记录", "## 监控计划摘要"):
        assert heading in markdown
    assert "v2" in markdown
    assert "已采纳" in markdown
    assert "s-0" in markdown  # lineage parent
    assert "sparse_band" in markdown


# ---------------------------------------------------------------------------
# Renderer table structure (fast).
# ---------------------------------------------------------------------------
def test_render_design_cutoff_bands_tables():
    from marvis.agent.renderers import _render_design_cutoff_bands

    text, tables = _render_design_cutoff_bands({
        "bands": [
            {"lo": 100, "hi": 300, "pop_pct": 0.5, "bad_rate": 0.2,
             "cum_approval_rate": 0.5, "cum_bad_rate": 0.2, "decision": "decline"},
            {"lo": 300, "hi": 600, "pop_pct": 0.5, "bad_rate": 0.0,
             "cum_approval_rate": 0.5, "cum_bad_rate": 0.0, "decision": "approve"},
        ],
        "recommended_rules": [{"condition": "score < 300", "decision": "reject"}],
        "red_flags": [{"level": "red", "code": "infeasible_constraints", "message": "x"}],
    })
    assert "分数带设计完成" in text
    assert "红项" in text  # red flag surfaced
    assert tables[0]["columns"] == ["band 区间", "样本占比", "坏率", "累计审批率", "累计坏率", "决策"]
    assert len(tables[0]["rows"]) == 2
    assert tables[-1]["title"] == "红旗清单"


def test_render_compare_strategies_matrix_table():
    from marvis.agent.renderers import _render_compare_strategies

    text, tables = _render_compare_strategies({
        "matrix_2x2": {
            "both_approve": {"count": 2, "bad_rate": 0.0},
            "only_new": {"count": 2, "bad_rate": 0.5},
            "only_baseline": {"count": 0, "bad_rate": 0.0},
            "both_decline": {"count": 2, "bad_rate": 1.0},
        },
        "deltas": {"approval_rate": 0.33, "approved_bad_rate": 0.1, "expected_profit": -5.0},
        "summary_text": "示例摘要",
        "red_flags": [{"level": "red", "code": "swap_in_worse", "message": "x"}],
    })
    assert "策略对比完成" in text
    # S6: templated Chinese conclusion line, numbers straight from the deltas.
    assert "结论：挑战者在通过率" in text
    # swap 2x2 is now a matrix-heat card (each cell's bad_rate colors the heat chip).
    assert tables[0]["columns"] == ["", "基线通过", "基线拒绝"]
    assert len(tables[0]["rows"]) == 2
    assert tables[0]["column_specs"] == [
        {"kind": "text"}, {"kind": "matrix-heat"}, {"kind": "matrix-heat"}
    ]
    assert tables[0]["rows"][0][1] == 0.0   # both_approve bad_rate as heat value
    assert tables[0]["rows"][0][2] == 0.5   # only_new bad_rate as heat value
    assert tables[1]["title"] == "关键指标并排（挑战者 vs 基线）"


def test_render_compare_strategies_empty_swap_cell_renders_na():
    # DOM-11: an empty swap cell's bad_rate is None from the core; the heat chip
    # falls back to 0.0 heat (no color signal) without raising, per _heat_cell.
    from marvis.agent.renderers import _render_compare_strategies

    text, tables = _render_compare_strategies({
        "matrix_2x2": {
            "both_approve": {"count": 2, "bad_rate": 0.0},
            "only_new": {"count": 0, "bad_rate": None},
            "only_baseline": {"count": 0, "bad_rate": None},
            "both_decline": {"count": 2, "bad_rate": 1.0},
        },
        "deltas": {"approval_rate": 0.0, "approved_bad_rate": 0.0, "expected_profit": 0.0},
        "summary_text": "示例摘要",
        "label_coverage": 0.8,
        "red_flags": [],
    })
    assert tables[0]["rows"][0][2] == 0.0  # only_new None -> heat falls back to 0.0
    assert "标签覆盖率 80.0%" in text


def test_render_backtest_strategy_label_coverage_and_na_swap():
    from marvis.agent.renderers import _render_backtest_strategy

    text, tables = _render_backtest_strategy({
        "approval_rate": 1.0,
        "approved_count": 2,
        "approved_bad_rate": 0.5,
        "rejected_bad_rate": 0.0,
        "expected_profit": 0.0,
        "swap_in_count": 0,
        "swap_out_count": 0,
        "swap_in_bad_rate": None,
        "swap_out_bad_rate": None,
        "label_coverage": 0.9,
        "by_segment": [],
    })
    assert "标签覆盖率 90.0%" in text
    rows = {row[0]: row[1] for row in tables[0]["rows"]}
    assert rows["标签覆盖率"] == "90.0%"


def test_render_adopt_strategy_table():
    from marvis.agent.renderers import _render_adopt_strategy

    text, tables = _render_adopt_strategy({
        "strategy_id": "s-1", "version": 2, "status": "adopted",
        "retired_strategy_ids": ["s-0"],
        "artifacts": [
            {"kind": "decision_table_csv", "path": "a.csv"},
            {"kind": "monitoring_plan_json", "path": "b.json"},
        ],
    })
    assert "策略已采纳" in text
    assert tables[0]["title"] == "交付物"
    assert len(tables[0]["rows"]) == 2
    assert tables[1]["title"] == "退役策略"


def test_render_strategy_doc_table():
    from marvis.agent.renderers import _render_strategy_doc

    text, tables = _render_strategy_doc({
        "doc_path": "workspace/tasks/t/strategy/doc.md",
        "sections": ["策略概览", "规则清单"],
    })
    assert "策略文档已生成" in text
    assert tables[0]["columns"] == ["#", "章节"]
    assert len(tables[0]["rows"]) == 2


# ---------------------------------------------------------------------------
# adopt + doc via the runner (slow).
# ---------------------------------------------------------------------------
def _build_and_backtest(runner, dataset, task, condition="score < 250"):
    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval",
         "rules": [{"condition": condition, "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id=task.id,
    )
    backtest = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {"dataset_id": dataset.id, "strategy_id": built.output["strategy_id"],
         "target_col": "bad"},
        task_id=task.id,
    )
    return built.output["strategy_id"], backtest.output["backtest_id"]


@pytest.mark.slow
def test_tool_adopt_strategy_rejects_foreign_backtest(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({"score": [100, 200, 300, 400, 500, 600], "bad": [1, 0, 1, 0, 0, 0]})
    dataset = _register(registry, tmp_path, frame, "adopt_foreign", task.id)
    sid_a, _ = _build_and_backtest(runner, dataset, task, "score < 250")
    _, bid_b = _build_and_backtest(runner, dataset, task, "score < 450")
    adopt = runner.invoke(
        ToolRef("strategy", "adopt_strategy"),
        {"strategy_id": sid_a, "backtest_id": bid_b, "adoption_reason": "x"},
        task_id=task.id,
    )
    assert adopt.ok is False
    assert "does not belong" in (adopt.error or "")


@pytest.mark.slow
def test_tool_adopt_strategy_lands_and_registers_deliverables(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({"score": [100, 200, 300, 400, 500, 600], "bad": [1, 0, 1, 0, 0, 0]})
    dataset = _register(registry, tmp_path, frame, "adopt_ok", task.id)
    sid, bid = _build_and_backtest(runner, dataset, task, "score < 250")
    band_stats = {
        "bands": [
            {"lo": 100, "hi": 300, "pop_pct": 0.3333, "bad_rate": 0.5, "expected_profit": 0.0,
             "cum_approval_rate": 1.0, "cum_bad_rate": 0.3, "decision": "decline"},
        ]
    }
    adopt = runner.invoke(
        ToolRef("strategy", "adopt_strategy"),
        {"strategy_id": sid, "backtest_id": bid, "adoption_reason": "approved",
         "band_stats": band_stats},
        task_id=task.id,
    )
    assert adopt.ok is True, adopt.error
    assert adopt.output["status"] == "adopted"
    assert adopt.output["version"] == 1
    kinds = {a["kind"] for a in adopt.output["artifacts"]}
    assert kinds == {"decision_table_csv", "monitoring_plan_json"}
    for artifact in adopt.output["artifacts"]:
        assert Path(artifact["path"]).exists()
    csv_path = next(a["path"] for a in adopt.output["artifacts"] if a["kind"] == "decision_table_csv")
    csv_text = Path(csv_path).read_text(encoding="utf-8")
    assert "序号,条件,决策" in csv_text
    assert "score < 250" in csv_text

    settings = build_settings(tmp_path / "workspace")
    audits = PluginRepository(settings.db_path).list_audit()
    kinds_audit = {a["kind"] for a in audits}
    assert "strategy.adopt" in kinds_audit
    assert "strategy.artifact" in kinds_audit

    # render doc reads persisted results and registers a third artifact.
    doc = runner.invoke(
        ToolRef("strategy", "render_strategy_doc"),
        {"strategy_id": sid, "band_stats": band_stats},
        task_id=task.id,
    )
    assert doc.ok is True, doc.error
    assert Path(doc.output["doc_path"]).exists()
    assert doc.output["sections"][0] == "策略概览"
    md = Path(doc.output["doc_path"]).read_text(encoding="utf-8")
    assert "已采纳" in md


@pytest.mark.slow
def test_tool_adopt_strategy_double_adopt_conflicts(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({"score": [100, 200, 300, 400, 500, 600], "bad": [1, 0, 1, 0, 0, 0]})
    dataset = _register(registry, tmp_path, frame, "adopt_twice", task.id)
    sid, bid = _build_and_backtest(runner, dataset, task, "score < 250")
    first = runner.invoke(
        ToolRef("strategy", "adopt_strategy"),
        {"strategy_id": sid, "backtest_id": bid, "adoption_reason": "first"},
        task_id=task.id,
    )
    assert first.ok is True, first.error
    second = runner.invoke(
        ToolRef("strategy", "adopt_strategy"),
        {"strategy_id": sid, "backtest_id": bid, "adoption_reason": "again"},
        task_id=task.id,
    )
    assert second.ok is False


# ---------------------------------------------------------------------------
# LT-11: recommendations carry evidence (B.1) + tradeoff alternatives (B.2).
# All numbers come from fields the tool output already carries -- no new compute.
# ---------------------------------------------------------------------------
def test_render_tradeoff_view_recommendation_carries_evidence_and_alternatives():
    from marvis.agent.renderers import _render_tradeoff_view

    text, tables = _render_tradeoff_view({
        "score_direction": "higher_is_better",
        "recommended": {"cutoff": 500, "approval_rate": 0.33, "bad_rate": 0.0,
                        "expected_profit": 120.0, "feasible": True},
        "points": [
            {"cutoff": 300, "approval_rate": 0.67, "bad_rate": 0.0,
             "expected_profit": 100.0, "feasible": True},
            {"cutoff": 500, "approval_rate": 0.33, "bad_rate": 0.0,
             "expected_profit": 120.0, "feasible": True},
            {"cutoff": 200, "approval_rate": 0.83, "bad_rate": 0.2,
             "expected_profit": 80.0, "feasible": False},
        ],
        "red_flags": [],
    })
    # B.1: the推荐 line cites its evidence -- feasible-point count + advantage over次优.
    assert "推荐 cutoff `500`" in text
    assert "依据：满足约束的可行点（共 2/3 个 cutoff 可行）" in text
    assert "预期利润较次优 cutoff `300` 高 +20.0000" in text
    # points table gains a 推荐 marker column; the recommended row is ★.
    points_table = next(t for t in tables if t["title"] == "cutoff 权衡点")
    assert points_table["columns"] == ["推荐", "cutoff", "审批率", "坏率", "预期利润", "可行"]
    reco_row = next(row for row in points_table["rows"] if row[1] == "500")
    assert reco_row[0] == "★"
    # B.2: a top-2 feasible备选 table shows what推荐 gives up (profit delta vs推荐).
    alt_table = next(t for t in tables if t["title"].startswith("次优可行 cutoff"))
    assert alt_table["columns"][-1] == "与推荐预期利润差"
    assert alt_table["rows"][0][0] == "300"
    assert alt_table["rows"][0][-1] == "-20.0000"


def test_render_tradeoff_view_without_recommendation_has_no_evidence():
    from marvis.agent.renderers import _render_tradeoff_view

    text, tables = _render_tradeoff_view({
        "score_direction": "higher_is_better",
        "recommended": None,
        "points": [{"cutoff": 300, "approval_rate": 0.5, "bad_rate": 0.3,
                    "expected_profit": 10.0, "feasible": False}],
    })
    assert "策略权衡视图完成" in text
    assert "依据" not in text
    assert not any(t["title"].startswith("次优可行 cutoff") for t in tables)


def test_render_design_cutoff_bands_recommendation_carries_evidence():
    from marvis.agent.renderers import _render_design_cutoff_bands

    text, _ = _render_design_cutoff_bands({
        "bands": [
            {"lo": 100, "hi": 300, "pop_pct": 0.5, "bad_rate": 0.2,
             "cum_approval_rate": 0.5, "cum_bad_rate": 0.2, "decision": "decline"},
            {"lo": 300, "hi": 600, "pop_pct": 0.5, "bad_rate": 0.0,
             "cum_approval_rate": 0.5, "cum_bad_rate": 0.0, "decision": "approve"},
        ],
        "recommended_rules": [{"condition": "score < 300", "decision": "reject"}],
        "red_flags": [],
    })
    # B.1: the推荐切法 cites its evidence -- the通过客群 cumulative bad/approval at the cut.
    assert "推荐切法 `score < 300`" in text
    assert "依据：通过客群累计坏率 0.0%，累计审批率 50.0%，满足约束" in text


def test_render_train_models_champion_carries_evidence_vs_runner_up():
    from marvis.agent.renderers import _render_train_models

    text, _ = _render_train_models({
        "target_type": "binary",
        "best_experiment_id": "exp-1", "best_recipe": "lgb",
        "experiments": [
            {"experiment_id": "exp-1", "recipe": "lgb", "metrics": {"oot_ks": 0.43}},
            {"experiment_id": "exp-2", "recipe": "xgb", "metrics": {"oot_ks": 0.39}},
        ],
    })
    # B.1/B.2: champion line cites the selection metric value + gap to the runner-up.
    assert "最优 **lgb**" in text
    assert "依据：按 OOT KS=0.4300" in text
    assert "较次优 xgb（0.3900）高 0.0400" in text
