import pandas as pd
import pytest

import marvis.packs.strategy as strategy_pack
from marvis.data.errors import ScoreDirectionConflictError
from marvis.packs.strategy import (
    StrategyError,
    apply_strategy,
    build_strategy,
    build_strategy_from_spec,
)
from marvis.packs.strategy.strategy import infer_strategy_rule_direction


def test_build_strategy_applies_rules_in_order_with_default_decision():
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score < 600", "decision": "reject"},
            {"condition": "score >= 720", "decision": "approve"},
        ],
        score_col="score",
        default_decision="approve",
        description="baseline cutoff",
    )
    frame = pd.DataFrame({"score": [580, 650, 750]})

    decisions = apply_strategy(frame, strategy)

    assert strategy.strategy_type == "approval"
    assert strategy.score_col == "score"
    assert strategy.description == "baseline cutoff"
    assert decisions.tolist() == ["reject", "approve", "approve"]


def test_strategy_id_excludes_display_and_observation_metadata() -> None:
    first = build_strategy(
        "approval",
        [
            {
                "condition": "score < 600",
                "decision": "reject",
                "support": 0.2,
                "lift": 1.8,
                "source": "tree",
            }
        ],
        score_col="score",
        default_decision="approve",
        description="first display copy",
    )
    renamed = build_strategy(
        "approval",
        [
            {
                "condition": "score < 600",
                "decision": "reject",
                "support": 0.8,
                "lift": 9.9,
                "source": "manual",
            }
        ],
        score_col="score",
        default_decision="approve",
        description="renamed display copy",
    )
    changed = build_strategy(
        "approval",
        [{"condition": "score < 601", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )

    assert first.id == renamed.id
    assert first.rules[0].rule_id == renamed.rules[0].rule_id
    assert changed.id != first.id
    assert changed.rules[0].rule_id != first.rules[0].rule_id


def test_explicit_priority_reordering_keeps_rule_identity_bound_to_condition() -> None:
    strategy = build_strategy(
        "approval",
        [
            {
                "condition": "score >= 800",
                "decision": "approve",
                "rule_id": "high-score",
                "priority": 20,
                "reason_code": "HIGH",
            },
            {
                "condition": "score < 600",
                "decision": "reject",
                "rule_id": "low-score",
                "priority": 10,
                "reason_code": "LOW",
            },
        ],
        score_col="score",
        default_decision="review",
    )

    assert [
        (rule.condition, rule.rule_id, rule.priority, rule.reason_code)
        for rule in strategy.rules
    ] == [
        ("score < 600", "low-score", 10, "LOW"),
        ("score >= 800", "high-score", 20, "HIGH"),
    ]
    assert [rule.rule_id for rule in strategy.spec.rules] == [
        "low-score",
        "high-score",
    ]


def test_generated_rule_ids_are_stable_when_explicit_priorities_reorder_input() -> None:
    rules = [
        {
            "condition": "score < 600",
            "decision": "reject",
            "priority": 10,
        },
        {
            "condition": "score >= 750",
            "decision": "approve",
            "priority": 20,
        },
    ]

    forward = build_strategy(
        "approval",
        rules,
        score_col="score",
        default_decision="review",
    )
    reversed_input = build_strategy(
        "approval",
        list(reversed(rules)),
        score_col="score",
        default_decision="review",
    )

    assert [rule.rule_id for rule in forward.rules] == [
        rule.rule_id for rule in reversed_input.rules
    ]
    assert forward.id == reversed_input.id


def test_duplicate_semantic_rules_require_explicit_unique_ids() -> None:
    duplicate = {"condition": "score < 600", "decision": "reject"}

    with pytest.raises(StrategyError, match="duplicate rule_id"):
        build_strategy(
            "approval",
            [duplicate, duplicate],
            score_col="score",
            default_decision="approve",
        )


def test_build_strategy_from_spec_executes_nested_tree_and_n_of_k_rules() -> None:
    strategy = build_strategy_from_spec(
        {
            "schema_version": "strategy.dsl.v1",
            "strategy_type": "approval",
            "match_policy": "first_match",
            "default_action": {"type": "approval", "value": "approve"},
            "rules": [
                {
                    "rule_id": "tree-high-risk",
                    "priority": 10,
                    "condition": {
                        "op": "and",
                        "args": [
                            {
                                "op": "between",
                                "field": "score",
                                "lower": 500,
                                "upper": 620,
                                "include_lower": True,
                                "include_upper": False,
                            },
                            {"op": "is_null", "field": "income"},
                        ],
                    },
                    "action": {
                        "type": "reject",
                        "reason_code": "TREE_HIGH_RISK",
                    },
                },
                {
                    "rule_id": "vote-two-of-three",
                    "priority": 20,
                    "condition": {
                        "op": "n_of_k",
                        "n": 2,
                        "args": [
                            {
                                "op": "compare",
                                "field": "late_count",
                                "operator": ">=",
                                "value": 3,
                            },
                            {
                                "op": "compare",
                                "field": "utilization",
                                "operator": ">",
                                "value": 0.9,
                            },
                            {"op": "is_null", "field": "income"},
                        ],
                    },
                    "action": {
                        "type": "reject",
                        "reason_code": "MULTI_SIGNAL",
                    },
                },
            ],
            "metadata": {"lineage": {"source_artifact": "tree-1"}},
        },
        score_col="score",
        description="nested strategy",
    )
    frame = pd.DataFrame(
        [
            {
                "score": 550,
                "income": None,
                "late_count": 0,
                "utilization": 0.1,
            },
            {
                "score": 700,
                "income": None,
                "late_count": 3,
                "utilization": 0.2,
            },
            {
                "score": 700,
                "income": 5000,
                "late_count": 3,
                "utilization": 0.95,
            },
            {
                "score": 700,
                "income": 5000,
                "late_count": 0,
                "utilization": 0.2,
            },
        ]
    )

    assert apply_strategy(frame, strategy).tolist() == [
        "reject",
        "reject",
        "reject",
        "approve",
    ]
    assert [rule.rule_id for rule in strategy.rules] == [
        "tree-high-risk",
        "vote-two-of-three",
    ]


def test_canonical_rule_projection_round_trips_through_legacy_build_surface() -> None:
    original = build_strategy_from_spec(
        {
            "strategy_type": "approval",
            "default_action": {"type": "review"},
            "rules": [
                {
                    "rule_id": "approve-prime",
                    "priority": 10,
                    "condition": {
                        "op": "and",
                        "args": [
                            {
                                "op": "compare",
                                "field": "score",
                                "operator": ">=",
                                "value": 700,
                            },
                            {"op": "is_not_null", "field": "income"},
                        ],
                    },
                    "action": {"type": "approval", "reason_code": "PRIME"},
                }
            ],
        },
        score_col="score",
    )
    rebuilt = build_strategy(
        original.strategy_type,
        [
            {
                "condition": rule.condition,
                "decision": rule.decision,
                "value": rule.value,
                "rule_id": rule.rule_id,
                "priority": rule.priority,
                "reason_code": rule.reason_code,
            }
            for rule in original.rules
        ],
        score_col=original.score_col,
        default_decision=original.default_decision,
    )
    frame = pd.DataFrame(
        {"score": [650, 750, 750], "income": [1000, None, 1000]}
    )

    assert apply_strategy(frame, rebuilt).tolist() == apply_strategy(
        frame, original
    ).tolist()
    assert rebuilt.spec.rules[0].condition == original.spec.rules[0].condition


def test_apply_strategy_uses_first_matching_rule():
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score >= 600", "decision": "approve"},
            {"condition": "score >= 700", "decision": "reject"},
        ],
        score_col="score",
        default_decision="reject",
    )

    decisions = apply_strategy(pd.DataFrame({"score": [750]}), strategy)

    assert decisions.tolist() == ["approve"]


def test_approval_strategy_supports_explicit_review_band() -> None:
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score < 600", "decision": "reject"},
            {"condition": "score < 680", "decision": "review"},
        ],
        score_col="score",
        default_decision="approve",
    )

    assert apply_strategy(
        pd.DataFrame({"score": [580, 650, 720]}), strategy
    ).tolist() == ["reject", "review", "approve"]


def test_apply_strategy_supports_in_conditions_and_rule_values():
    strategy = build_strategy(
        "segmentation",
        [
            {"condition": "grade in ['A', 'B']", "decision": "segment", "value": "prime"},
            {"condition": "grade == 'C' or score >= 700", "decision": "segment", "value": "watch"},
        ],
        score_col="score",
        default_decision="other",
    )
    frame = pd.DataFrame({"grade": ["A", "C", "D"], "score": [610, 650, 720]})

    assert apply_strategy(frame, strategy).tolist() == ["prime", "watch", "watch"]


def test_build_strategy_rejects_decision_mismatch_and_unknown_type():
    with pytest.raises(StrategyError, match="decision"):
        build_strategy(
            "approval",
            [{"condition": "score < 600", "decision": "limit", "value": 1000}],
            score_col="score",
            default_decision="approve",
        )


@pytest.mark.parametrize("priority", [1.5, "10", True])
def test_build_strategy_rejects_non_integer_priority(priority) -> None:
    with pytest.raises(StrategyError, match="priority must be an integer"):
        build_strategy(
            "approval",
            [
                {
                    "condition": "score < 600",
                    "decision": "reject",
                    "priority": priority,
                }
            ],
            score_col="score",
            default_decision="approve",
        )
    with pytest.raises(StrategyError, match="strategy_type"):
        build_strategy(
            "unknown",
            [{"condition": "score < 600", "decision": "reject"}],
            score_col="score",
            default_decision="approve",
        )


def test_safe_condition_rejects_calls_attributes_and_unknown_columns():
    with pytest.raises(StrategyError, match="unsupported condition"):
        build_strategy(
            "approval",
            [{"condition": "__import__('os').system('touch /tmp/marvis_pwned') == 0", "decision": "reject"}],
            score_col="score",
            default_decision="approve",
        )
    with pytest.raises(StrategyError, match="unsupported condition"):
        build_strategy(
            "approval",
            [{"condition": "score.__class__ == int", "decision": "reject"}],
            score_col="score",
            default_decision="approve",
        )

    strategy = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )
    with pytest.raises(StrategyError, match="unknown field"):
        apply_strategy(pd.DataFrame({"model_score": [500]}), strategy)


def test_strategy_package_exports_build_surface():
    assert strategy_pack.StrategyError is StrategyError
    assert strategy_pack.build_strategy is build_strategy
    assert strategy_pack.apply_strategy is apply_strategy


def test_build_strategy_raises_on_inconsistent_rule_directions():
    """Two rules land on the SAME decision via opposite comparison styles against
    score_col -- no single score direction can explain "always reject" whether the
    score is low or high, unlike a coherent banded cutoff strategy."""
    with pytest.raises(ScoreDirectionConflictError, match="score_direction conflict"):
        build_strategy(
            "approval",
            [
                {"condition": "score < 500", "decision": "reject"},
                {"condition": "score >= 900", "decision": "reject"},
            ],
            score_col="score",
            default_decision="approve",
        )


def test_build_strategy_single_direction_rules_pass_and_report_inferred_direction():
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score >= 600", "decision": "approve"},
            {"condition": "score >= 750", "decision": "approve"},
        ],
        score_col="score",
        default_decision="reject",
    )

    assert infer_strategy_rule_direction(list(strategy.rules), strategy.score_col) == "higher_is_better"


def test_build_strategy_banded_cutoff_does_not_raise_despite_opposite_operator_styles():
    """score < 600 -> reject and score >= 720 -> approve use opposite comparison
    styles but agree that a higher score is better (reject the low band, approve the
    high band) -- a coherent monotonic cutoff strategy, not a direction conflict."""
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score < 600", "decision": "reject"},
            {"condition": "score >= 720", "decision": "approve"},
        ],
        score_col="score",
        default_decision="approve",
    )

    assert strategy.score_col == "score"
