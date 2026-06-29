from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

import pandas as pd

from marvis.packs.strategy.contracts import Strategy, StrategyRule
from marvis.packs.strategy.errors import StrategyError


_ALLOWED_DECISIONS = {
    "approval": {"approve", "reject"},
    "limit": {"limit"},
    "pricing": {"price"},
    "reject": {"reject"},
    "segmentation": {"segment"},
}


def build_strategy(
    strategy_type: str,
    rules: list[dict],
    *,
    score_col: str | None,
    default_decision,
    description: str = "",
) -> Strategy:
    if strategy_type not in _ALLOWED_DECISIONS:
        raise StrategyError(f"unsupported strategy_type: {strategy_type}")

    parsed_rules = []
    for rule in rules:
        condition = str(rule["condition"])
        _parse_condition(condition)
        decision = str(rule["decision"])
        value = rule.get("value")
        _validate_decision(strategy_type, decision, value)
        parsed_rules.append(
            StrategyRule(
                condition=condition,
                decision=decision,
                value=value,
            )
        )

    return Strategy(
        id=_strategy_id(
            strategy_type=strategy_type,
            rules=rules,
            score_col=score_col,
            default_decision=default_decision,
            description=description,
        ),
        strategy_type=strategy_type,
        rules=tuple(parsed_rules),
        score_col=score_col,
        default_decision=default_decision,
        description=description,
    )


def apply_strategy(df: pd.DataFrame, strategy: Strategy) -> pd.Series:
    decisions = pd.Series([strategy.default_decision] * len(df), index=df.index, dtype="object")
    assigned = pd.Series(False, index=df.index)
    for rule in strategy.rules:
        mask = _safe_eval_condition(df, rule.condition) & ~assigned
        decisions.loc[mask] = rule.value if rule.value is not None else rule.decision
        assigned |= mask
    return decisions


def _safe_eval_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    expression = _parse_condition(condition)
    mask = _eval_node(expression.body, df)
    if not isinstance(mask, pd.Series):
        raise StrategyError("condition must evaluate to a boolean mask")
    return mask.fillna(False).astype(bool)


def _parse_condition(condition: str) -> ast.Expression:
    try:
        expression = ast.parse(condition, mode="eval")
    except SyntaxError as exc:
        raise StrategyError(f"invalid condition: {condition}") from exc
    _validate_condition_ast(expression.body)
    return expression


def _validate_condition_ast(node: ast.AST) -> None:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.And | ast.Or):
            raise StrategyError("unsupported condition operator")
        for value in node.values:
            _validate_condition_ast(value)
        return
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise StrategyError("unsupported condition comparison")
        if not isinstance(node.left, ast.Name):
            raise StrategyError("unsupported condition expression")
        if not isinstance(
            node.ops[0],
            ast.Lt | ast.LtE | ast.Gt | ast.GtE | ast.Eq | ast.NotEq | ast.In | ast.NotIn,
        ):
            raise StrategyError("unsupported condition comparison")
        _validate_literal(node.comparators[0])
        return
    raise StrategyError("unsupported condition expression")


def _validate_literal(node: ast.AST) -> None:
    if isinstance(node, ast.Constant):
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int | float):
            return
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        for element in node.elts:
            _validate_literal(element)
        return
    raise StrategyError("unsupported condition literal")


def _eval_node(node: ast.AST, df: pd.DataFrame) -> pd.Series:
    if isinstance(node, ast.BoolOp):
        masks = [_eval_node(value, df) for value in node.values]
        result = masks[0]
        for mask in masks[1:]:
            if isinstance(node.op, ast.And):
                result = result & mask
            else:
                result = result | mask
        return result
    if isinstance(node, ast.Compare):
        field = node.left.id
        if field not in df.columns:
            raise StrategyError(f"unknown field: {field}")
        return _eval_comparison(df[field], node.ops[0], _literal_value(node.comparators[0]))
    raise StrategyError("unsupported condition expression")


def _eval_comparison(series: pd.Series, op: ast.cmpop, value: Any) -> pd.Series:
    try:
        if _numeric_literal(value):
            series = _coerce_numeric_series(series)
        if isinstance(op, ast.Lt):
            return series < value
        if isinstance(op, ast.LtE):
            return series <= value
        if isinstance(op, ast.Gt):
            return series > value
        if isinstance(op, ast.GtE):
            return series >= value
        if isinstance(op, ast.Eq):
            return series == value
        if isinstance(op, ast.NotEq):
            return series != value
        if isinstance(op, ast.In):
            values = _membership_values(value)
            if values and all(_numeric_literal(item) for item in values):
                series = _coerce_numeric_series(series)
            return series.isin(values)
        if isinstance(op, ast.NotIn):
            values = _membership_values(value)
            if values and all(_numeric_literal(item) for item in values):
                series = _coerce_numeric_series(series)
            return ~series.isin(values)
    except TypeError as exc:
        raise StrategyError("condition comparison failed") from exc
    raise StrategyError("unsupported condition comparison")


def _numeric_literal(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    failed = series.notna() & numeric.isna()
    if bool(failed.any()):
        raise StrategyError("condition comparison failed: field contains non-numeric values")
    return numeric


def _literal_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _literal_value(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return [_literal_value(element) for element in node.elts]
    raise StrategyError("unsupported condition literal")


def _membership_values(value) -> list:
    if not isinstance(value, list | tuple | set):
        raise StrategyError("in condition requires a list, tuple, or set")
    return list(value)


def _validate_decision(strategy_type: str, decision: str, value) -> None:
    allowed = _ALLOWED_DECISIONS[strategy_type]
    if decision not in allowed:
        raise StrategyError(f"decision {decision} is not allowed for {strategy_type}")
    if decision in {"limit", "price", "segment"} and value is None:
        raise StrategyError(f"decision {decision} requires a value")


def _strategy_id(
    *,
    strategy_type: str,
    rules: list[dict],
    score_col: str | None,
    default_decision,
    description: str,
) -> str:
    payload = {
        "default_decision": default_decision,
        "description": description,
        "rules": rules,
        "score_col": score_col,
        "strategy_type": strategy_type,
    }
    digest = hashlib.sha256(json.dumps(payload, default=str, sort_keys=True).encode("utf-8")).hexdigest()
    return f"strategy-{digest[:12]}"


__all__ = ["apply_strategy", "build_strategy"]
