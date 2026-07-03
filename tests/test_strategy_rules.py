"""S4 Commit 1: rule mining + rule-set evaluation core and tool boundary.

Covers the spec's Commit-1 test checklist:
* tree channel + univariate channel each surface at least one rule, asserted
  field-by-field on an 8-row hand set;
* determinism (INV-1): two identical mine_rules calls return byte-equal dicts;
* round-trip: a mined condition fed to build_strategy hits the exact same rows
  (the mine/evaluate/build shared-evaluator contract);
* the five red_flags (suspect_leakage, low_support, rule_shadowed, high_overlap,
  nan_labels_dropped) each fire on a targeted case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.rules import evaluate_rule_set, mine_rules
from marvis.packs.strategy.strategy import (
    apply_strategy,
    build_strategy,
    evaluate_condition_mask,
)
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


# ---------------------------------------------------------------------------
# Hand-computed 8-row set: f1 cleanly separates bad (bottom 3), f2 is partial.
# ---------------------------------------------------------------------------
def _hand_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "f1":  [10, 20, 30, 40, 50, 60, 70, 80],
        "f2":  [1,  1,  0,  0,  1,  0,  0,  0],
        "bad": [1,  1,  1,  0,  0,  0,  0,  0],
    })


def _mine(**overrides):
    kwargs = dict(
        feature_cols=["f1", "f2"], target_col="bad", max_depth=2,
        min_support=0.1, min_lift=1.2, top_k=10,
    )
    kwargs.update(overrides)
    return mine_rules(_hand_frame(), **kwargs)


def test_mine_rules_surfaces_tree_and_univariate_channels():
    rules = _mine()
    sources = {rule.source for rule in rules}
    assert "tree" in sources
    assert "univariate" in sources
    # base bad rate is 3/8 = 0.375; the f1 low band (3 rows, all bad) has lift
    # 1.0/0.375 = 2.667 and support 3/8 = 0.375 on BOTH channels.
    tree = next(rule for rule in rules if rule.source == "tree")
    assert tree.hit_count == 3
    assert round(tree.hit_bad_rate, 4) == 1.0
    assert round(tree.support, 4) == 0.375
    assert round(tree.lift, 4) == round(1.0 / 0.375, 4)
    assert "f1" in tree.condition
    univariate = next(rule for rule in rules if rule.source == "univariate")
    assert univariate.hit_count >= 1
    assert univariate.lift >= 1.2


def test_mine_rules_is_deterministic_dict_equal():
    a = [rule.as_dict() for rule in _mine()]
    b = [rule.as_dict() for rule in _mine()]
    assert a == b
    assert a  # non-empty


def test_mine_rules_condition_round_trips_through_build_strategy():
    frame = _hand_frame()
    for rule in _mine():
        mine_mask = evaluate_condition_mask(frame, rule.condition).to_numpy(dtype=bool)
        strategy = build_strategy(
            "approval",
            [{"condition": rule.condition, "decision": "reject"}],
            score_col=None, default_decision="approve",
        )
        build_mask = (apply_strategy(frame, strategy) == "reject").to_numpy(dtype=bool)
        assert (mine_mask == build_mask).all(), rule.condition


def test_evaluate_rule_set_waterfall_first_match_wins():
    frame = _hand_frame()
    # Two overlapping low-f1 rules; the second should show zero incremental hits.
    result = evaluate_rule_set(
        frame,
        [{"condition": "f1 < 35"}, {"condition": "f1 < 31"}],
        target_col="bad",
    )
    waterfall = result["waterfall"]
    assert waterfall[0]["incremental_hits"] == 3
    assert waterfall[1]["incremental_hits"] == 0  # fully shadowed
    assert round(waterfall[0]["cum_reject_rate"], 4) == 0.375
    assert round(result["residual"]["approval_rate"], 4) == 0.625
    assert round(result["residual"]["bad_rate"], 4) == 0.0
    assert round(result["combined"]["rejected_bad_rate"], 4) == 1.0
    # overlap matrix: 3∩3 over 3∪3 == 1.0 (rule2 ⊂ rule1).
    assert result["overlap_matrix"][0][1] == 1.0


# ---------------------------------------------------------------------------
# Tool boundary via the real runner.
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
            model_name="S4 规则策略", model_version="dev", validator="qa",
            source_dir=str(tmp_path / "source"), algorithm="lr", run_mode="agent",
            task_type="strategy", target_col="bad",
        )
    )
    return runner, registry, task


def _register(registry, tmp_path, frame, name, task_id):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


@pytest.mark.slow
def test_tool_mine_rules_flags_suspect_leakage(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    # A feature that IS the label -> a single rule with lift far above 10 and
    # hit bad rate 1.0 -> suspect_leakage.
    frame = pd.DataFrame({
        "leak": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 5,
        "bad":  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 5,
    })
    dataset = _register(registry, tmp_path, frame, "leak", task.id)
    out = runner.invoke(
        ToolRef("strategy", "mine_rules"),
        {"dataset_id": dataset.id, "target_col": "bad", "feature_cols": ["leak"],
         "min_lift": 1.2, "min_support": 0.05},
        task_id=task.id,
    )
    assert out.ok is True, out.error
    assert out.output["candidate_rules"]
    assert any(f["code"] == "suspect_leakage" and f["level"] == "red" for f in out.output["red_flags"])


@pytest.mark.slow
def test_tool_mine_rules_gates_nan_labels(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "f1":  [10, 20, 30, 40, 50, 60, 70, 80],
        "bad": [1.0, 1.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    dataset = _register(registry, tmp_path, frame, "nan", task.id)
    base = {"dataset_id": dataset.id, "target_col": "bad", "feature_cols": ["f1"],
            "min_lift": 1.0, "min_support": 0.05}
    blocked = runner.invoke(ToolRef("strategy", "mine_rules"), dict(base), task_id=task.id)
    assert blocked.ok is False
    assert blocked.error_kind == "nan_label_not_confirmed"
    confirmed = runner.invoke(
        ToolRef("strategy", "mine_rules"), {**base, "drop_nan_labels": True}, task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["nan_labels_dropped"] == 1
    assert any(f["code"] == "nan_labels_dropped" for f in confirmed.output["red_flags"])


@pytest.mark.slow
def test_tool_mine_rules_flags_low_support(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    # A tiny bad pocket: mine at a loose min_support (0.001) so a sub-2% rule is
    # INCLUDED, then the tool's fixed 2% low-support floor flags it (the mining
    # min_support filter and the display floor are deliberately independent).
    n = 300
    f1 = list(range(n))
    bad = [1 if i < 3 else 0 for i in range(n)]  # 3/300 = 1% base
    frame = pd.DataFrame({"f1": f1, "bad": bad})
    dataset = _register(registry, tmp_path, frame, "lowsup", task.id)
    out = runner.invoke(
        ToolRef("strategy", "mine_rules"),
        {"dataset_id": dataset.id, "target_col": "bad", "feature_cols": ["f1"],
         "min_support": 0.001, "min_lift": 1.2},
        task_id=task.id,
    )
    assert out.ok is True, out.error
    supports = [r["support"] for r in out.output["candidate_rules"]]
    assert any(s < 0.02 for s in supports)  # a sub-2% support rule got in
    assert any(f["code"] == "low_support" and f["level"] == "amber" for f in out.output["red_flags"])


@pytest.mark.slow
def test_tool_evaluate_rule_set_flags_shadowed_and_overlap(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "f1":  [10, 20, 30, 40, 50, 60, 70, 80],
        "bad": [1,  1,  1,  0,  0,  0,  0,  0],
    })
    dataset = _register(registry, tmp_path, frame, "eval", task.id)
    out = runner.invoke(
        ToolRef("strategy", "evaluate_rule_set"),
        {"dataset_id": dataset.id, "target_col": "bad",
         "rules": [{"condition": "f1 < 35"}, {"condition": "f1 < 31"}]},
        task_id=task.id,
    )
    assert out.ok is True, out.error
    codes = {f["code"] for f in out.output["red_flags"]}
    assert "rule_shadowed" in codes  # rule 2 has zero incremental hits
    assert "high_overlap" in codes   # rule2 ⊂ rule1 -> Jaccard 1.0 > 0.8


@pytest.mark.slow
def test_tool_select_rule_set_applies_selection(tmp_path):
    runner, registry, task = _runtime(tmp_path)
    candidates = [
        {"rule_id": "rule_1", "condition": "f1 < 31"},
        {"rule_id": "rule_2", "condition": "f2 >= 1"},
        {"rule_id": "rule_3", "condition": "f1 >= 70"},
    ]
    picked = runner.invoke(
        ToolRef("strategy", "select_rule_set"),
        {"candidate_rules": candidates, "selection": [1, 3]},
        task_id=task.id,
    )
    assert picked.ok is True, picked.error
    assert [r["condition"] for r in picked.output["selected_rules"]] == ["f1 < 31", "f1 >= 70"]
    assert picked.output["selected_count"] == 2
    # None selection -> keep all.
    keep_all = runner.invoke(
        ToolRef("strategy", "select_rule_set"),
        {"candidate_rules": candidates, "selection": None},
        task_id=task.id,
    )
    assert keep_all.output["selected_count"] == 3
