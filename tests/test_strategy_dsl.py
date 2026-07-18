from __future__ import annotations

import math

import pytest

from marvis.packs.strategy.dsl import (
    STRATEGY_DSL_SCHEMA_VERSION,
    StrategyAction,
    StrategyRuleSpec,
    StrategySpec,
    canonical_strategy_json,
    canonicalize_expression,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression, evaluate_strategy_row


def _approval(value: str = "approve", *, stop: bool = True) -> StrategyAction:
    return StrategyAction(type="approval", value=value, stop=stop)


def _rule(
    rule_id: str,
    priority: int,
    condition: dict,
    *,
    action: StrategyAction | None = None,
) -> StrategyRuleSpec:
    return StrategyRuleSpec(
        rule_id=rule_id,
        priority=priority,
        condition=condition,
        action=action or _approval(),
    )


def _with_default_missing(expression: dict) -> dict:
    result = dict(expression)
    if result["op"] in {"compare", "between"}:
        result.setdefault("missing", "no_match")
    if "args" in result:
        result["args"] = [_with_default_missing(item) for item in result["args"]]
    if "arg" in result:
        result["arg"] = _with_default_missing(result["arg"])
    return result


def test_expression_canonicalizer_covers_all_v1_expression_types() -> None:
    expressions = [
        {"op": "compare", "field": "score", "operator": ">=", "value": 700},
        {"op": "compare", "field": "score", "operator": "<", "value": 800},
        {"op": "compare", "field": "grade", "operator": "==", "value": "A"},
        {"op": "compare", "field": "grade", "operator": "!=", "value": "D"},
        {
            "op": "compare",
            "field": "region",
            "operator": "in",
            "value": ["east", "north"],
        },
        {
            "op": "compare",
            "field": "region",
            "operator": "not_in",
            "value": ["blocked"],
        },
        {
            "op": "between",
            "field": "age",
            "lower": 18,
            "upper": 65,
            "include_lower": True,
            "include_upper": False,
        },
        {"op": "is_null", "field": "income"},
        {"op": "is_not_null", "field": "income"},
        {
            "op": "and",
            "args": [
                {"op": "compare", "field": "score", "operator": ">", "value": 600},
                {"op": "compare", "field": "score", "operator": "<=", "value": 900},
            ],
        },
        {
            "op": "or",
            "args": [
                {"op": "is_null", "field": "income"},
                {"op": "compare", "field": "income", "operator": ">", "value": 0},
            ],
        },
        {
            "op": "not",
            "arg": {"op": "compare", "field": "grade", "operator": "==", "value": "D"},
        },
        {
            "op": "n_of_k",
            "n": 2,
            "args": [
                {"op": "compare", "field": "a", "operator": "==", "value": 1},
                {"op": "compare", "field": "b", "operator": "==", "value": 1},
                {"op": "compare", "field": "c", "operator": "==", "value": 1},
            ],
        },
    ]

    assert [canonicalize_expression(item) for item in expressions] == [
        _with_default_missing(item) for item in expressions
    ]


def test_expression_evaluator_has_explicit_boundary_null_boolean_and_n_of_k_semantics() -> None:
    row = {"score": "700", "age": 65, "income": None, "a": 1, "b": 0, "c": 1}

    assert evaluate_expression(
        row, {"op": "compare", "field": "score", "operator": ">=", "value": 700}
    )
    assert evaluate_expression(
        row,
        {
            "op": "between",
            "field": "age",
            "lower": 18,
            "upper": 65,
            "include_lower": True,
            "include_upper": True,
        },
    )
    assert not evaluate_expression(
        row,
        {
            "op": "between",
            "field": "age",
            "lower": 18,
            "upper": 65,
            "include_lower": True,
            "include_upper": False,
        },
    )
    assert evaluate_expression(row, {"op": "is_null", "field": "income"})
    assert evaluate_expression(
        row,
        {
            "op": "and",
            "args": [
                {"op": "is_null", "field": "income"},
                {
                    "op": "not",
                    "arg": {
                        "op": "compare",
                        "field": "score",
                        "operator": "<",
                        "value": 700,
                    },
                },
            ],
        },
    )
    assert evaluate_expression(
        row,
        {
            "op": "n_of_k",
            "n": 2,
            "args": [
                {"op": "compare", "field": "a", "operator": "==", "value": 1},
                {"op": "compare", "field": "b", "operator": "==", "value": 1},
                {"op": "compare", "field": "c", "operator": "==", "value": 1},
            ],
        },
    )


@pytest.mark.parametrize(
    "expression",
    [
        {"op": "compare", "field": "missing", "operator": "!=", "value": 1},
        {"op": "compare", "field": "missing", "operator": "not_in", "value": [1]},
        {"op": "between", "field": "missing", "lower": 0, "upper": 1},
        {"op": "is_null", "field": "missing"},
        {"op": "is_not_null", "field": "missing"},
    ],
)
def test_unknown_field_fails_closed(expression: dict) -> None:
    with pytest.raises(StrategyError, match="unknown field: missing"):
        evaluate_expression({}, expression)


def test_unknown_field_is_not_hidden_by_boolean_or_first_match_short_circuit() -> None:
    with pytest.raises(StrategyError, match="unknown field: missing"):
        evaluate_expression(
            {"known": 1},
            {
                "op": "or",
                "args": [
                    {"op": "compare", "field": "known", "operator": "==", "value": 1},
                    {"op": "compare", "field": "missing", "operator": "==", "value": 1},
                ],
            },
        )

    spec = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="reject"),
        rules=(
            _rule("first", 1, {"op": "is_not_null", "field": "known"}),
            _rule("later", 2, {"op": "is_not_null", "field": "missing"}),
        ),
    )
    with pytest.raises(StrategyError, match="unknown field: missing"):
        evaluate_strategy_row({"known": 1}, spec)


@pytest.mark.parametrize("missing_value", [None, float("nan")])
def test_missing_value_never_matches_compare_or_between(missing_value: object) -> None:
    row = {"score": missing_value}
    assert not evaluate_expression(
        row, {"op": "compare", "field": "score", "operator": "!=", "value": 1}
    )
    assert not evaluate_expression(
        row, {"op": "compare", "field": "score", "operator": "not_in", "value": [1]}
    )
    assert not evaluate_expression(
        row, {"op": "between", "field": "score", "lower": 0, "upper": 1}
    )
    assert evaluate_expression(
        row,
        {
            "op": "compare",
            "field": "score",
            "operator": ">=",
            "value": 700,
            "missing": "match",
        },
    )
    assert evaluate_expression(
        row,
        {
            "op": "between",
            "field": "score",
            "lower": 0,
            "upper": 1,
            "missing": "match",
        },
    )
    with pytest.raises(StrategyError, match="field value is missing"):
        evaluate_expression(
            row,
            {
                "op": "compare",
                "field": "score",
                "operator": ">=",
                "value": 700,
                "missing": "error",
            },
        )


def test_typed_actions_are_canonical_and_value_actions_require_a_value() -> None:
    assert StrategyAction.from_dict({"type": "approval"}).to_dict() == {
        "type": "approval",
        "value": "approve",
        "reason_code": None,
        "stop": True,
    }
    assert StrategyAction.from_dict({"type": "reject", "reason_code": "POLICY"}).to_dict() == {
        "type": "reject",
        "value": "reject",
        "reason_code": "POLICY",
        "stop": True,
    }
    assert StrategyAction.from_dict({"type": "review"}).to_dict() == {
        "type": "review",
        "value": "review",
        "reason_code": None,
        "stop": True,
    }
    assert StrategyAction(
        type="limit", value=5000, reason_code="LIMIT_TIER"
    ).to_dict() == {
        "type": "limit",
        "value": 5000,
        "reason_code": "LIMIT_TIER",
        "stop": True,
    }
    assert StrategyAction(type="pricing", value=0.12).type == "pricing"
    assert StrategyAction(type="segment", value="prime").type == "segment"

    with pytest.raises(StrategyError, match="requires a value"):
        StrategyAction(type="limit")
    with pytest.raises(StrategyError, match="unsupported action type"):
        StrategyAction(type="manual_review", value="manual")


def test_strategy_spec_orders_first_match_rules_by_unique_priority() -> None:
    spec = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="reject"),
        rules=(
            _rule(
                "later",
                20,
                {"op": "compare", "field": "score", "operator": ">=", "value": 600},
                action=StrategyAction(type="approval", reason_code="LATER"),
            ),
            _rule(
                "first",
                10,
                {"op": "compare", "field": "score", "operator": ">=", "value": 600},
                action=StrategyAction(type="approval", reason_code="FIRST"),
            ),
        ),
        metadata={"lineage": {"source_artifact": "dataset-1"}},
    )

    assert spec.schema_version == STRATEGY_DSL_SCHEMA_VERSION
    assert spec.match_policy == "first_match"
    assert [rule.rule_id for rule in spec.rules] == ["first", "later"]
    result = evaluate_strategy_row({"score": 700}, spec)
    assert result.matched_rule_id == "first"
    assert result.action.value == "approve"
    assert result.action.reason_code == "FIRST"
    assert result.action.stop is True


def test_dsl_v1_rejects_non_terminal_action_at_the_action_boundary() -> None:
    with pytest.raises(StrategyError, match="supports only stop=true"):
        StrategyAction(type="reject", stop=False)


def test_strategy_row_uses_typed_default_action_when_no_rule_matches() -> None:
    spec = StrategySpec(
        strategy_type="pricing",
        default_action=StrategyAction(type="pricing", value=0.18, reason_code="BASE"),
        rules=(
            _rule(
                "prime-price",
                10,
                {"op": "compare", "field": "score", "operator": ">=", "value": 800},
                action=StrategyAction(type="pricing", value=0.08, reason_code="PRIME"),
            ),
        ),
    )

    result = evaluate_strategy_row({"score": 700}, spec)
    assert result.to_dict() == {
        "matched_rule_id": None,
        "action": {
            "type": "pricing",
            "value": 0.18,
            "reason_code": "BASE",
            "stop": True,
        },
    }


def test_canonical_hash_excludes_all_nonexecuting_metadata() -> None:
    base = {
        "schema_version": STRATEGY_DSL_SCHEMA_VERSION,
        "strategy_type": "approval",
        "match_policy": "first_match",
        "default_action": {"type": "reject"},
        "rules": [
            {
                "rule_id": "approve-prime",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "score",
                    "operator": ">=",
                    "value": 750,
                },
                "action": {"type": "approval", "value": "approve"},
            }
        ],
        "metadata": {
            "lineage": {"source_artifact": "dataset-1", "generation": {"seed": 7}},
            "display": {"title": "Champion", "color": "green"},
            "description": "shown to users",
        },
    }
    renamed = {
        **base,
        "metadata": {
            **base["metadata"],
            "display": {"title": "Renamed", "color": "blue"},
            "description": "new copy",
        },
    }
    new_lineage = {
        **base,
        "metadata": {
            **base["metadata"],
            "lineage": {"source_artifact": "dataset-2", "generation": {"seed": 7}},
        },
    }

    assert strategy_spec_hash(base) == strategy_spec_hash(renamed)
    assert strategy_spec_hash(base) == strategy_spec_hash(new_lineage)
    assert '"title":"Champion"' in canonical_strategy_json(base, include_display_metadata=True)
    assert '"title":"Champion"' not in canonical_strategy_json(
        base, include_display_metadata=False
    )
    assert '"lineage"' not in canonical_strategy_json(
        base, include_display_metadata=False
    )


@pytest.mark.parametrize(
    ("action_type", "wrong_value", "expected"),
    [
        ("approval", "reject", "approve"),
        ("reject", "approve", "reject"),
        ("review", "approve", "review"),
    ],
)
def test_fixed_decision_actions_reject_contradictory_values(
    action_type: str, wrong_value: str, expected: str
) -> None:
    with pytest.raises(
        StrategyError,
        match=rf"action {action_type} value must be '{expected}'",
    ):
        StrategyAction(type=action_type, value=wrong_value)


def test_legacy_output_alias_is_explicit_and_cannot_impersonate_another_action() -> None:
    action = StrategyAction(type="approval", output_value="pass")

    assert action.value == "approve"
    assert action.decision_value == "pass"
    assert action.to_dict()["output_value"] == "pass"
    with pytest.raises(StrategyError, match="output_value contradicts"):
        StrategyAction(type="approval", output_value="reject")


@pytest.mark.parametrize(
    ("action_type", "value", "message"),
    [
        ("limit", -1, "non-negative number"),
        ("limit", "1000", "non-negative number"),
        ("pricing", -0.01, "annual decimal rate"),
        ("pricing", 1.01, "annual decimal rate"),
        ("pricing", "0.12", "annual decimal rate"),
        ("segment", "", "non-empty scalar id"),
        ("segment", [], "non-empty scalar id"),
    ],
)
def test_value_actions_enforce_typed_domain_values(
    action_type: str, value, message: str
) -> None:
    with pytest.raises(StrategyError, match=message):
        StrategyAction(type=action_type, value=value)


def test_spec_round_trip_produces_canonical_json_ready_payload() -> None:
    payload = {
        "schema_version": STRATEGY_DSL_SCHEMA_VERSION,
        "strategy_type": "segmentation",
        "match_policy": "first_match",
        "default_action": {"type": "segment", "value": "other"},
        "rules": [
            {
                "rule_id": "prime",
                "priority": 1,
                "condition": {"op": "compare", "field": "score", "operator": ">", "value": 800},
                "action": {"type": "segment", "value": "prime"},
            }
        ],
        "metadata": {"lineage": {"source_artifact": "scores.csv"}},
    }

    parsed = parse_strategy_spec(payload)
    canonical = parsed.to_dict()
    assert canonical["schema_version"] == STRATEGY_DSL_SCHEMA_VERSION
    assert canonical["rules"][0]["action"]["reason_code"] is None
    assert parse_strategy_spec(canonical) == parsed


def test_invalid_dsl_fails_closed() -> None:
    with pytest.raises(StrategyError, match="unsupported comparison operator"):
        canonicalize_expression(
            {"op": "compare", "field": "score", "operator": "contains", "value": 7}
        )
    with pytest.raises(StrategyError, match="requires a list value"):
        canonicalize_expression(
            {"op": "compare", "field": "region", "operator": "in", "value": "east"}
        )
    with pytest.raises(StrategyError, match="n must be between"):
        canonicalize_expression(
            {
                "op": "n_of_k",
                "n": 2,
                "args": [{"op": "is_null", "field": "income"}],
            }
        )
    with pytest.raises(StrategyError, match="finite JSON number"):
        canonicalize_expression(
            {"op": "compare", "field": "score", "operator": ">", "value": math.inf}
        )
    with pytest.raises(StrategyError, match="duplicate rule_id"):
        StrategySpec(
            strategy_type="approval",
            default_action=StrategyAction(type="reject"),
            rules=(
                _rule("same", 1, {"op": "is_null", "field": "x"}),
                _rule("same", 2, {"op": "is_null", "field": "y"}),
            ),
        )
    with pytest.raises(StrategyError, match="duplicate rule priority"):
        StrategySpec(
            strategy_type="approval",
            default_action=StrategyAction(type="reject"),
            rules=(
                _rule("one", 1, {"op": "is_null", "field": "x"}),
                _rule("two", 1, {"op": "is_null", "field": "y"}),
            ),
        )


def test_strategy_spec_rejects_action_types_from_another_strategy_channel() -> None:
    with pytest.raises(StrategyError, match="not allowed for pricing: reject"):
        StrategySpec(
            strategy_type="pricing",
            default_action=StrategyAction(type="pricing", value=0.18),
            rules=(
                _rule(
                    "wrong-channel",
                    1,
                    {"op": "is_null", "field": "score"},
                    action=StrategyAction(type="reject"),
                ),
            ),
        )
