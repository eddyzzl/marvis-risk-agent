from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from marvis.packs.strategy.contracts import Strategy, StrategyRule
from marvis.packs.strategy.dsl import (
    StrategyAction,
    StrategyRuleSpec,
    StrategySpec,
    canonicalize_expression,
)
from marvis.packs.strategy.errors import StrategyError


_COMPARISON_OPERATORS: tuple[tuple[type[ast.cmpop], str], ...] = (
    (ast.Lt, "<"),
    (ast.LtE, "<="),
    (ast.Gt, ">"),
    (ast.GtE, ">="),
    (ast.Eq, "=="),
    (ast.NotEq, "!="),
    (ast.In, "in"),
    (ast.NotIn, "not_in"),
)

_LEGACY_ACTION_TYPES = {
    "approve": "approval",
    # Historical strategy bands and imported reports used ``decline`` as the
    # row-level label for the same governed reject action.  Keep the label in
    # ``output_value`` while mapping the action semantics to ``reject``.
    "decline": "reject",
    "reject": "reject",
    "review": "review",
    "limit": "limit",
    "price": "pricing",
    "segment": "segment",
}

_FALLBACK_ACTION_TYPES = {
    "approval": "approval",
    "reject": "reject",
    "limit": "limit",
    "pricing": "pricing",
    "segmentation": "segment",
}


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        if not isinstance(node.operand, ast.Constant) or not isinstance(
            node.operand.value, int | float
        ):
            raise StrategyError("unsupported legacy condition literal")
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return [_literal_value(item) for item in node.elts]
    raise StrategyError("unsupported legacy condition literal")


def _comparison_operator(operator: ast.cmpop) -> str:
    for operator_type, value in _COMPARISON_OPERATORS:
        if isinstance(operator, operator_type):
            return value
    raise StrategyError("unsupported legacy condition comparison")


def _convert_node(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            op = "and"
        elif isinstance(node.op, ast.Or):
            op = "or"
        else:
            raise StrategyError("unsupported legacy condition operator")
        return {
            "op": op,
            "args": [_convert_node(value) for value in node.values],
        }
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise StrategyError("unsupported legacy condition comparison")
        if not isinstance(node.left, ast.Name):
            raise StrategyError("unsupported legacy condition expression")
        operator = _comparison_operator(node.ops[0])
        # pandas considers NaN unequal to ordinary values and outside every
        # membership set. Preserve that historical behavior only in migrated
        # legacy rules; newly authored DSL expressions default to no_match.
        missing = "match" if operator in {"!=", "not_in"} else "no_match"
        return canonicalize_expression(
            {
                "op": "compare",
                "field": node.left.id,
                "operator": operator,
                "value": _literal_value(node.comparators[0]),
                "missing": missing,
            }
        )
    raise StrategyError("unsupported legacy condition expression")


def legacy_condition_to_expression(condition: str) -> dict[str, Any]:
    """Translate a legacy condition or canonical JSON projection into DSL v1.

    Canonical strategies retain ``Strategy.rules`` for old report/version surfaces.
    Their conditions are JSON-encoded typed expressions; accepting that projection
    here makes it a lossless round trip instead of a display-only dead end.
    """

    if not isinstance(condition, str) or not condition.strip():
        raise StrategyError("legacy condition must be a non-empty string")
    try:
        canonical_payload = json.loads(condition)
    except (TypeError, ValueError, json.JSONDecodeError):
        canonical_payload = None
    if isinstance(canonical_payload, Mapping):
        return canonicalize_expression(canonical_payload)
    try:
        expression = ast.parse(condition, mode="eval")
    except SyntaxError as exc:
        raise StrategyError(f"invalid legacy condition: {condition}") from exc
    return canonicalize_expression(_convert_node(expression.body))


def _legacy_action(
    decision: object,
    value: Any,
    *,
    strategy_type: str | None = None,
    reason_code: str | None = None,
) -> StrategyAction:
    decision_text = str(decision)
    action_type = _LEGACY_ACTION_TYPES.get(decision_text)
    if action_type is None and strategy_type is not None:
        action_type = _FALLBACK_ACTION_TYPES.get(strategy_type)
    if action_type is None:
        raise StrategyError(f"unsupported legacy decision: {decision_text}")
    # The typed value remains canonical for approval/reject/review semantics. Old
    # strategies may still emit custom decision labels (most commonly default
    # ``pass``/``manual``); preserve those explicitly as output_value so metrics use
    # the action type while row-level compatibility remains exact.
    legacy_output = decision_text if value is None else value
    fixed_value = StrategyAction._DEFAULT_VALUES.get(action_type)
    action_value = fixed_value if fixed_value is not None else legacy_output
    output_value = (
        legacy_output
        if fixed_value is not None and legacy_output != fixed_value
        else None
    )
    return StrategyAction(
        type=action_type,
        value=action_value,
        reason_code=reason_code,
        stop=True,
        output_value=output_value,
    )


def _stable_legacy_rule_id(
    *,
    condition: Mapping[str, Any],
    action: StrategyAction,
) -> str:
    payload = {
        "condition": condition,
        "action": action.to_dict(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"rule-legacy-{hashlib.sha256(encoded).hexdigest()[:16]}"


def legacy_rule_to_dsl(
    rule: StrategyRule,
    *,
    priority: int,
    ordinal: int,
    rule_id: str | None = None,
) -> StrategyRuleSpec:
    if not isinstance(rule, StrategyRule):
        raise StrategyError("legacy rule must be a StrategyRule")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise StrategyError("legacy rule ordinal must be a non-negative integer")
    condition = legacy_condition_to_expression(rule.condition)
    action = _legacy_action(
        rule.decision,
        rule.value,
        reason_code=rule.reason_code,
    )
    stable_id = rule_id or _stable_legacy_rule_id(
        condition=condition,
        action=action,
    )
    return StrategyRuleSpec(
        rule_id=stable_id,
        priority=priority,
        condition=condition,
        action=action,
    )


def legacy_strategy_to_spec(
    strategy: Strategy,
    *,
    metadata: Mapping[str, Any] | None = None,
    duplicate_rule_ids: str = "suffix",
) -> StrategySpec:
    """Convert a persisted legacy Strategy without changing row decisions."""

    if not isinstance(strategy, Strategy):
        raise StrategyError("legacy strategy must be a Strategy")
    metadata_payload = dict(metadata or {})
    lineage = metadata_payload.get("lineage", {})
    if not isinstance(lineage, Mapping):
        raise StrategyError("strategy metadata lineage must be an object")
    normalized_lineage = dict(lineage)
    normalized_lineage.setdefault("source", "legacy_strategy")
    if strategy.id:
        normalized_lineage.setdefault("strategy_id", strategy.id)
    metadata_payload["lineage"] = normalized_lineage
    if strategy.description and "description" not in metadata_payload:
        metadata_payload["description"] = strategy.description
    default_action = _legacy_action(
        strategy.default_decision,
        strategy.default_decision,
        strategy_type=strategy.strategy_type,
    )
    rules = tuple(
        legacy_rule_to_dsl(
            rule,
            priority=(
                rule.priority
                if rule.priority is not None
                else (ordinal + 1) * 10
            ),
            ordinal=ordinal,
            rule_id=rule.rule_id,
        )
        for ordinal, rule in enumerate(strategy.rules)
    )
    if duplicate_rule_ids not in {"suffix", "error"}:
        raise StrategyError("duplicate_rule_ids must be suffix or error")
    seen: dict[str, int] = {}
    normalized_rules: list[StrategyRuleSpec] = []
    for rule in rules:
        occurrence = seen.get(rule.rule_id, 0) + 1
        seen[rule.rule_id] = occurrence
        if occurrence == 1:
            normalized_rules.append(rule)
            continue
        if duplicate_rule_ids == "error":
            raise StrategyError(f"duplicate rule_id: {rule.rule_id}")
        normalized_rules.append(
            replace(rule, rule_id=f"{rule.rule_id}-duplicate-{occurrence}")
        )
    return StrategySpec(
        strategy_type=strategy.strategy_type,
        default_action=default_action,
        rules=tuple(normalized_rules),
        metadata=metadata_payload,
    )


__all__ = [
    "legacy_condition_to_expression",
    "legacy_rule_to_dsl",
    "legacy_strategy_to_spec",
]
