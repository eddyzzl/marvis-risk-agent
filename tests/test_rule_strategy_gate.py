"""S4 Commit 2: rule-set gate reply parsing + the three S4 renderers.

Gate parser (_parse_rule_selection_instruction): 「选 …」/「去掉 …」/「全选」 map to
ordered 1-based index lists; unrelated/question/out-of-range replies return None
(fall through to the LLM router), with a typed range guard on indices.
Renderers turn tool output into (text, tables) for the driver messages.
"""

from __future__ import annotations

from marvis.agent.plan_driver import _parse_rule_selection_instruction as parse_selection
from marvis.agent.renderers import (
    _render_evaluate_rule_set,
    _render_mine_rules,
    _render_select_rule_set,
)


# ---------------------------------------------------------------------------
# gate reply parser
# ---------------------------------------------------------------------------
def test_parse_selection_keep_explicit_indices_in_order():
    assert parse_selection("选 1,3,5", 6) == [1, 3, 5]
    assert parse_selection("保留 1 3 5", 6) == [1, 3, 5]
    assert parse_selection("pick 1 3", 5) == [1, 3]


def test_parse_selection_select_all():
    assert parse_selection("全选", 4) == [1, 2, 3, 4]
    assert parse_selection("都要", 3) == [1, 2, 3]
    assert parse_selection("all", 2) == [1, 2]


def test_parse_selection_drop_indices():
    assert parse_selection("去掉 2", 4) == [1, 3, 4]
    assert parse_selection("去除 2 4", 5) == [1, 3, 5]
    assert parse_selection("drop 2", 3) == [1, 3]


def test_parse_selection_reorders_by_user_order():
    assert parse_selection("选第 3 条和第 1 条", 5) == [3, 1]


def test_parse_selection_bare_index_list_is_a_keep():
    assert parse_selection("1,3,5", 6) == [1, 3, 5]
    assert parse_selection("1 3 5", 6) == [1, 3, 5]


def test_parse_selection_returns_none_for_non_selection_replies():
    assert parse_selection("这些规则怎么样？", 5) is None
    assert parse_selection("确认", 5) is None
    assert parse_selection("阈值放宽到 0.1", 5) is None


def test_parse_selection_drops_out_of_range_indices_typed_guard():
    # index 9 is out of [1, 5] -> dropped; an all-out-of-range keep -> None.
    assert parse_selection("选 9", 5) is None
    assert parse_selection("选 2 9", 5) == [2]
    # a drop with only out-of-range indices is not actionable -> None.
    assert parse_selection("去掉 9", 3) is None


def test_parse_selection_no_candidates_returns_none():
    assert parse_selection("全选", 0) is None


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------
def test_render_mine_rules_table_and_flags():
    text, tables = _render_mine_rules({
        "candidate_rules": [
            {"rule_id": "rule_1", "condition": "f1 < 30", "support": 0.3,
             "hit_bad_rate": 0.8, "lift": 2.5, "source": "tree"},
            {"rule_id": "rule_2", "condition": "f2 >= 10", "support": 0.2,
             "hit_bad_rate": 0.95, "lift": 12.0, "source": "univariate"},
        ],
        "n_rows": 100,
        "red_flags": [{"level": "red", "code": "suspect_leakage", "message": "x"}],
    })
    assert "规则挖掘完成" in text
    assert "选 1,3,5" in text  # the gate prompt guides the reply syntax
    assert "红旗" in text
    assert tables[0]["columns"] == ["#", "规则", "支持度", "命中坏率", "lift", "来源"]
    assert len(tables[0]["rows"]) == 2
    assert tables[-1]["title"] == "红旗清单"


def test_render_select_rule_set_table():
    text, tables = _render_select_rule_set({
        "selected_rules": [
            {"condition": "f1 < 30", "decision": "reject", "lift": 2.5, "source": "tree"},
        ],
        "selected_count": 1,
        "candidate_count": 3,
    })
    assert "规则集已选定" in text
    assert tables[0]["title"] == "已选规则（按顺序命中）"
    assert len(tables[0]["rows"]) == 1


def test_render_evaluate_rule_set_waterfall_and_overlap():
    text, tables = _render_evaluate_rule_set({
        "decision": "reject",
        "waterfall": [
            {"rule_id": "rule_1", "incremental_hits": 3, "incremental_bad_rate": 1.0,
             "cum_reject_rate": 0.3, "cum_reject_bad_rate": 1.0},
            {"rule_id": "rule_2", "incremental_hits": 0, "incremental_bad_rate": 0.0,
             "cum_reject_rate": 0.3, "cum_reject_bad_rate": 1.0},
        ],
        "overlap_matrix": [[1.0, 0.9], [0.9, 1.0]],
        "residual": {"approval_rate": 0.7, "bad_rate": 0.0},
        "combined": {"reject_rate": 0.3, "rejected_bad_rate": 1.0, "approved_bad_rate": 0.0},
        "red_flags": [
            {"level": "amber", "code": "rule_shadowed", "message": "x"},
            {"level": "amber", "code": "high_overlap", "message": "y"},
        ],
    })
    assert "规则集评估完成" in text
    assert "告警" in text
    titles = [t["title"] for t in tables]
    assert "命中瀑布（按顺序，首个命中生效）" in titles
    assert "规则重叠矩阵（共同命中占比）" in titles
    assert tables[-1]["title"] == "红旗清单"
