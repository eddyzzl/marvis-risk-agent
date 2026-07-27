from __future__ import annotations

import pandas as pd
import pytest

from marvis.packs.strategy.contracts import Strategy, StrategyRule
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_strategy_rows
from marvis.packs.strategy.legacy_adapter import (
    legacy_condition_to_expression,
    legacy_rule_to_dsl,
    legacy_strategy_to_spec,
)
from marvis.packs.strategy.strategy import apply_strategy, build_strategy


def test_legacy_strategy_adapter_is_row_equivalent_for_current_boolean_surface() -> None:
    legacy = build_strategy(
        "approval",
        [
            {
                "condition": "score >= 700 and region in ['north', 'east']",
                "decision": "approve",
            },
            {
                "condition": "score < 650 or region not in ['north', 'east']",
                "decision": "reject",
            },
        ],
        score_col=None,
        default_decision="review",
        description="legacy policy",
    )
    frame = pd.DataFrame(
        [
            {"score": "720", "region": "north"},
            {"score": 640, "region": "north"},
            {"score": 680, "region": "west"},
            {"score": 680, "region": "east"},
        ]
    )

    spec = legacy_strategy_to_spec(legacy)
    old_values = apply_strategy(frame, legacy).tolist()
    new_values = [result.action.value for result in evaluate_strategy_rows(frame, spec)]

    assert new_values == old_values == ["approve", "reject", "reject", "review"]
    assert [result.matched_rule_id for result in evaluate_strategy_rows(frame, spec)] == [
        spec.rules[0].rule_id,
        spec.rules[1].rule_id,
        spec.rules[1].rule_id,
        None,
    ]
    assert spec.metadata["lineage"] == {
        "source": "legacy_strategy",
        "strategy_id": legacy.id,
    }


def test_legacy_condition_adapter_preserves_parenthesized_and_or_ast() -> None:
    expression = legacy_condition_to_expression(
        "score >= 700 and (region == 'east' or region == 'north')"
    )

    assert expression == {
        "op": "and",
        "args": [
            {
                "op": "compare",
                "field": "score",
                "operator": ">=",
                "value": 700,
                "missing": "no_match",
            },
            {
                "op": "or",
                "args": [
                    {
                        "op": "compare",
                        "field": "region",
                        "operator": "==",
                        "value": "east",
                        "missing": "no_match",
                    },
                    {
                        "op": "compare",
                        "field": "region",
                        "operator": "==",
                        "value": "north",
                        "missing": "no_match",
                    },
                ],
            },
        ],
    }


@pytest.mark.parametrize(
    ("decision", "value", "action_type", "action_value"),
    [
        ("approve", None, "approval", "approve"),
        ("reject", None, "reject", "reject"),
        ("review", None, "review", "review"),
        ("limit", 5000, "limit", 5000),
        ("price", 0.12, "pricing", 0.12),
        ("segment", "prime", "segment", "prime"),
    ],
)
def test_legacy_decisions_map_to_typed_actions(
    decision: str, value: object, action_type: str, action_value: object
) -> None:
    converted = legacy_rule_to_dsl(
        StrategyRule(condition="score >= 700", decision=decision, value=value),
        priority=10,
        ordinal=0,
    )

    assert converted.action.type == action_type
    assert converted.action.value == action_value
    assert converted.action.stop is True


def test_generated_legacy_rule_ids_are_stable_and_content_sensitive() -> None:
    rule = StrategyRule(condition="score >= 700", decision="approve", value=None)
    first = legacy_rule_to_dsl(rule, priority=10, ordinal=0)
    second = legacy_rule_to_dsl(rule, priority=10, ordinal=0)
    moved = legacy_rule_to_dsl(rule, priority=20, ordinal=1)
    changed = legacy_rule_to_dsl(
        StrategyRule(condition="score >= 701", decision="approve", value=None),
        priority=10,
        ordinal=0,
    )

    assert first.rule_id == second.rule_id
    assert first.rule_id == moved.rule_id
    assert first.rule_id != changed.rule_id
    assert first.rule_id.startswith("rule-legacy-")


@pytest.mark.parametrize(
    "condition",
    [
        "600 < score < 800",
        "not score >= 700",
        "score + 1 >= 700",
        "is_good(score)",
        "score in allowed_regions",
    ],
)
def test_legacy_adapter_rejects_expressions_outside_current_evaluator_surface(
    condition: str,
) -> None:
    with pytest.raises(StrategyError):
        legacy_condition_to_expression(condition)


def test_legacy_missing_values_remain_false_for_comparisons() -> None:
    legacy = build_strategy(
        "approval",
        [{"condition": "score >= 700", "decision": "approve"}],
        score_col=None,
        default_decision="reject",
    )
    frame = pd.DataFrame([{"score": None}, {"score": 700}])
    spec = legacy_strategy_to_spec(legacy)

    assert apply_strategy(frame, legacy).tolist() == ["reject", "approve"]
    assert [item.action.value for item in evaluate_strategy_rows(frame, spec)] == [
        "reject",
        "approve",
    ]


@pytest.mark.parametrize("operator", ["!=", "not in"])
def test_legacy_adapter_preserves_pandas_negative_comparison_nan_behavior(
    operator: str,
) -> None:
    condition = (
        "region != 'blocked'" if operator == "!=" else "region not in ['blocked']"
    )
    legacy = build_strategy(
        "approval",
        [{"condition": condition, "decision": "approve"}],
        score_col=None,
        default_decision="reject",
    )
    frame = pd.DataFrame([{"region": None}, {"region": "blocked"}])
    spec = legacy_strategy_to_spec(legacy)

    assert spec.rules[0].condition["missing"] == "match"
    assert apply_strategy(frame, legacy).tolist() == ["approve", "reject"]
    assert [item.action.value for item in evaluate_strategy_rows(frame, spec)] == [
        "approve",
        "reject",
    ]


def test_historical_duplicate_legacy_rules_remain_readable_and_row_equivalent() -> None:
    legacy = Strategy(
        id="legacy-duplicates",
        strategy_type="approval",
        rules=(
            StrategyRule("score < 600", "reject", None),
            StrategyRule("score < 600", "reject", None),
        ),
        score_col="score",
        default_decision="approve",
        description="historical",
    )

    spec = legacy_strategy_to_spec(legacy)

    assert len({rule.rule_id for rule in spec.rules}) == 2
    assert spec.rules[1].rule_id.endswith("-duplicate-2")
    assert apply_strategy(
        pd.DataFrame({"score": [500, 700]}), legacy
    ).tolist() == ["reject", "approve"]
