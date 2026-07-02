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
    _ = PluginRepository  # keep import used across slow/fast paths
