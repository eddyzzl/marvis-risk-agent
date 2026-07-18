from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import (
    FrameEvaluation,
    evaluate_expression_frame,
    evaluate_strategy_frame,
    evaluate_strategy_rows,
)


def _rule(
    rule_id: str,
    priority: int,
    condition: dict,
    value: object,
    *,
    action_type: str = "segment",
) -> StrategyRuleSpec:
    return StrategyRuleSpec(
        rule_id=rule_id,
        priority=priority,
        condition=condition,
        action=StrategyAction(type=action_type, value=value),
    )


def _nested_spec() -> StrategySpec:
    return StrategySpec(
        strategy_type="segmentation",
        default_action=StrategyAction(type="segment", value="standard"),
        rules=(
            # Deliberately supplied out of order: first-match is defined by priority.
            _rule(
                "two-signals",
                20,
                {
                    "op": "n_of_k",
                    "n": 2,
                    "args": [
                        {
                            "op": "compare",
                            "field": "income",
                            "operator": ">=",
                            "value": 10000,
                        },
                        {"op": "is_not_null", "field": "phone"},
                        {
                            "op": "not",
                            "arg": {
                                "op": "compare",
                                "field": "region",
                                "operator": "in",
                                "value": ["blocked"],
                            },
                        },
                    ],
                },
                "review",
            ),
            _rule(
                "prime",
                10,
                {
                    "op": "and",
                    "args": [
                        {
                            "op": "between",
                            "field": "score",
                            "lower": 700,
                            "upper": 850,
                            "include_lower": True,
                            "include_upper": False,
                        },
                        {
                            "op": "or",
                            "args": [
                                {
                                    "op": "compare",
                                    "field": "region",
                                    "operator": "==",
                                    "value": "north",
                                },
                                {"op": "is_null", "field": "region"},
                            ],
                        },
                    ],
                },
                "prime",
            ),
        ),
    )


def test_vectorized_strategy_evaluator_matches_row_golden_for_nested_dsl() -> None:
    frame = pd.DataFrame(
        [
            {"score": "700", "income": "9000", "phone": "1", "region": "north"},
            {"score": 849, "income": 12000, "phone": None, "region": np.nan},
            {"score": 850, "income": "12000", "phone": "1", "region": "blocked"},
            {"score": np.nan, "income": 12000, "phone": "1", "region": "south"},
            {"score": 650, "income": 100, "phone": None, "region": "blocked"},
        ],
        index=pd.Index([10, 10, 30, 40, 50], name="application_id"),
    )
    spec = _nested_spec()

    expected = evaluate_strategy_rows(frame, spec)
    actual = evaluate_strategy_frame(frame, spec)

    assert isinstance(actual, FrameEvaluation)
    assert actual.decisions.index.equals(frame.index)
    assert actual.matched_rule_id.index.equals(frame.index)
    assert actual.decisions.tolist() == [item.action.value for item in expected]
    assert actual.matched_rule_id.tolist() == [item.matched_rule_id for item in expected]
    assert actual.decisions.tolist() == [
        "prime",
        "prime",
        "review",
        "review",
        "standard",
    ]
    assert actual.matched_rule_id.tolist() == [
        "prime",
        "prime",
        "two-signals",
        "two-signals",
        None,
    ]
    assert actual.to_frame().columns.tolist() == [
        "decision",
        "action_type",
        "matched_rule_id",
        "reason_code",
    ]
    assert actual.action_type.tolist() == ["segment"] * len(frame)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            {"op": "compare", "field": "score", "operator": ">=", "value": 700},
            [True, False, False],
        ),
        (
            {
                "op": "compare",
                "field": "score",
                "operator": ">=",
                "value": 700,
                "missing": "match",
            },
            [True, False, True],
        ),
        (
            {
                "op": "between",
                "field": "score",
                "lower": 699,
                "upper": 800,
                "include_lower": False,
                "include_upper": False,
            },
            [True, False, False],
        ),
        ({"op": "is_null", "field": "score"}, [False, False, True]),
        ({"op": "is_not_null", "field": "score"}, [True, True, False]),
    ],
)
def test_vectorized_atomic_expressions_coerce_numeric_strings_and_handle_nan(
    expression: dict, expected: list[bool]
) -> None:
    frame = pd.DataFrame({"score": ["700", "650", np.nan]})

    assert evaluate_expression_frame(frame, expression).tolist() == expected


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("<", 3, [True, False, False]),
        ("<=", 2, [True, False, False]),
        (">", 2, [False, True, False]),
        (">=", 3, [False, True, False]),
        ("==", 2, [True, False, False]),
        ("!=", 2, [False, True, False]),
        ("in", [2, 4], [True, False, False]),
        ("not_in", [2, 4], [False, True, False]),
    ],
)
def test_vectorized_compare_supports_every_dsl_operator(
    operator: str, value: object, expected: list[bool]
) -> None:
    frame = pd.DataFrame({"score": ["2", "3", np.nan]})
    expression = {
        "op": "compare",
        "field": "score",
        "operator": operator,
        "value": value,
    }

    assert evaluate_expression_frame(frame, expression).tolist() == expected


def test_vectorized_missing_error_respects_boolean_and_first_match_short_circuit() -> None:
    # The error branch is inactive for every row, matching the row evaluator.
    expression = {
        "op": "or",
        "args": [
            {"op": "compare", "field": "kind", "operator": "==", "value": "known"},
            {
                "op": "compare",
                "field": "score",
                "operator": ">=",
                "value": 700,
                "missing": "error",
            },
        ],
    }
    frame = pd.DataFrame({"kind": ["known", "known"], "score": [np.nan, np.nan]})
    assert evaluate_expression_frame(frame, expression).tolist() == [True, True]

    spec = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="reject"),
        rules=(
            _rule(
                "known",
                1,
                {"op": "is_not_null", "field": "kind"},
                "approve",
                action_type="approval",
            ),
            _rule(
                "would-error",
                2,
                {
                    "op": "compare",
                    "field": "score",
                    "operator": ">=",
                    "value": 700,
                    "missing": "error",
                },
                "approve",
                action_type="approval",
            ),
        ),
    )
    assert evaluate_strategy_frame(frame, spec).decisions.tolist() == ["approve", "approve"]


def test_vectorized_missing_error_raises_when_atom_is_active() -> None:
    frame = pd.DataFrame({"score": [700, np.nan]})
    expression = {
        "op": "compare",
        "field": "score",
        "operator": ">=",
        "value": 700,
        "missing": "error",
    }

    with pytest.raises(StrategyError, match="field value is missing"):
        evaluate_expression_frame(frame, expression)


def test_vectorized_evaluator_validates_all_fields_before_short_circuit() -> None:
    frame = pd.DataFrame({"known": [1, 1]})
    hidden_in_or = {
        "op": "or",
        "args": [
            {"op": "compare", "field": "known", "operator": "==", "value": 1},
            {"op": "compare", "field": "missing", "operator": "==", "value": 1},
        ],
    }
    with pytest.raises(StrategyError, match="unknown field: missing"):
        evaluate_expression_frame(frame, hidden_in_or)

    spec = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="reject"),
        rules=(
            _rule(
                "first",
                1,
                {"op": "is_not_null", "field": "known"},
                "approve",
                action_type="approval",
            ),
            _rule(
                "hidden-later",
                2,
                {"op": "is_not_null", "field": "missing"},
                "approve",
                action_type="approval",
            ),
        ),
    )
    with pytest.raises(StrategyError, match="unknown field: missing"):
        evaluate_strategy_frame(frame, spec)


def test_vectorized_evaluator_supports_empty_frames_after_schema_validation() -> None:
    frame = pd.DataFrame(columns=["score"])
    spec = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="reject"),
        rules=(
            _rule(
                "approve",
                1,
                {"op": "compare", "field": "score", "operator": ">=", "value": 700},
                "approve",
                action_type="approval",
            ),
        ),
    )

    result = evaluate_strategy_frame(frame, spec)

    assert result.decisions.empty
    assert result.matched_rule_id.empty
    assert result.decisions.dtype == object
    assert result.matched_rule_id.dtype == object
