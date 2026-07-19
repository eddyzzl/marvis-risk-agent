from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.dsl import (
    StrategyAction,
    StrategySpec,
    canonicalize_expression,
    parse_strategy_spec,
)
from marvis.packs.strategy.errors import StrategyError


@dataclass(frozen=True)
class RowEvaluation:
    matched_rule_id: str | None
    action: StrategyAction

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_rule_id": self.matched_rule_id,
            "action": self.action.to_dict(),
        }


@dataclass(frozen=True)
class FrameEvaluation:
    """Vectorized first-match decisions aligned to the input DataFrame index."""

    decisions: pd.Series
    matched_rule_id: pd.Series
    action_type: pd.Series
    reason_code: pd.Series

    @property
    def matched_rule_ids(self) -> pd.Series:
        """Plural compatibility alias for callers that treat the result as a batch."""

        return self.matched_rule_id

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "decision": self.decisions,
                "action_type": self.action_type,
                "matched_rule_id": self.matched_rule_id,
                "reason_code": self.reason_code,
            },
            index=self.decisions.index,
        )


def _lookup(row: Mapping[str, Any], field: str) -> Any:
    try:
        if field not in row:
            raise StrategyError(f"unknown field: {field}")
        return row[field]
    except (KeyError, TypeError):
        raise StrategyError(f"unknown field: {field}") from None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        marker = pd.isna(value)
        return bool(marker) if isinstance(marker, bool | Integral) else False
    except (TypeError, ValueError):
        return False


def _numeric_literal(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _boolean_literal(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _strict_equal(value: Any, expected: Any) -> bool:
    """Compare JSON scalar values without the legacy numeric-string coercion."""

    if _boolean_literal(expected):
        return _boolean_literal(value) and bool(value) is bool(expected)
    if _numeric_literal(expected):
        return _numeric_literal(value) and bool(value == expected)
    if isinstance(expected, str):
        return isinstance(value, str) and value == expected
    return value is expected


def _coerce_numeric(value: Any) -> int | float:
    if _is_missing(value):
        raise StrategyError("condition comparison failed: field value is missing")
    try:
        numeric = pd.to_numeric(value, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise StrategyError("condition comparison failed") from exc
    if _is_missing(numeric):
        raise StrategyError(
            "condition comparison failed: field contains a non-numeric value"
        )
    if isinstance(numeric, Integral):
        return int(numeric)
    return float(numeric)


def _missing_result(value: Any, policy: str) -> bool | None:
    if not _is_missing(value):
        return None
    if policy == "no_match":
        return False
    if policy == "match":
        return True
    if policy == "error":
        raise StrategyError("condition comparison failed: field value is missing")
    raise StrategyError(f"unsupported missing policy: {policy}")


def _compare(
    value: Any,
    operator: str,
    expected: Any,
    *,
    missing: str,
    coercion: str = "auto",
) -> bool:
    missing_result = _missing_result(value, missing)
    if missing_result is not None:
        return missing_result
    if coercion == "strict":
        if operator in {"in", "not_in"}:
            matched = any(_strict_equal(value, item) for item in expected)
            return matched if operator == "in" else not matched
        matched = _strict_equal(value, expected)
        return matched if operator == "==" else not matched
    try:
        if operator in {"in", "not_in"}:
            candidates = list(expected)
            if candidates and all(_numeric_literal(item) for item in candidates):
                value = _coerce_numeric(value)
            matched = value in candidates
            return matched if operator == "in" else not matched
        if _numeric_literal(expected):
            value = _coerce_numeric(value)
        if operator == "<":
            return bool(value < expected)
        if operator == "<=":
            return bool(value <= expected)
        if operator == ">":
            return bool(value > expected)
        if operator == ">=":
            return bool(value >= expected)
        if operator == "==":
            return bool(value == expected)
        if operator == "!=":
            return bool(value != expected)
    except (TypeError, ValueError) as exc:
        raise StrategyError("condition comparison failed") from exc
    raise StrategyError(f"unsupported comparison operator: {operator}")


def _between(value: Any, expression: Mapping[str, Any]) -> bool:
    missing_result = _missing_result(value, expression["missing"])
    if missing_result is not None:
        return missing_result
    lower = expression["lower"]
    upper = expression["upper"]
    if _numeric_literal(lower) and _numeric_literal(upper):
        value = _coerce_numeric(value)
    try:
        lower_match = value >= lower if expression["include_lower"] else value > lower
        upper_match = value <= upper if expression["include_upper"] else value < upper
        return bool(lower_match and upper_match)
    except (TypeError, ValueError) as exc:
        raise StrategyError("condition comparison failed") from exc


def _evaluate_canonical(row: Mapping[str, Any], expression: Mapping[str, Any]) -> bool:
    op = expression["op"]
    if op == "compare":
        value = _lookup(row, expression["field"])
        return _compare(
            value,
            expression["operator"],
            expression["value"],
            missing=expression["missing"],
            coercion=expression.get("coercion", "auto"),
        )
    if op == "between":
        value = _lookup(row, expression["field"])
        return _between(value, expression)
    if op == "is_null":
        value = _lookup(row, expression["field"])
        return _is_missing(value)
    if op == "is_not_null":
        value = _lookup(row, expression["field"])
        return not _is_missing(value)
    if op == "and":
        return all(_evaluate_canonical(row, arg) for arg in expression["args"])
    if op == "or":
        return any(_evaluate_canonical(row, arg) for arg in expression["args"])
    if op == "not":
        return not _evaluate_canonical(row, expression["arg"])
    if op == "n_of_k":
        required = expression["n"]
        matched = 0
        for arg in expression["args"]:
            matched += int(_evaluate_canonical(row, arg))
            if matched >= required:
                return True
        return False
    raise StrategyError(f"unsupported expression op: {op}")


def _expression_fields(expression: Mapping[str, Any]) -> tuple[str, ...]:
    op = expression["op"]
    if op in {"compare", "between", "is_null", "is_not_null"}:
        return (expression["field"],)
    if op in {"and", "or", "n_of_k"}:
        return tuple(
            field
            for argument in expression["args"]
            for field in _expression_fields(argument)
        )
    if op == "not":
        return _expression_fields(expression["arg"])
    raise StrategyError(f"unsupported expression op: {op}")


def _assert_known_fields(row: Mapping[str, Any], expression: Mapping[str, Any]) -> None:
    for field in _expression_fields(expression):
        _lookup(row, field)


def _assert_known_frame_fields(
    frame: pd.DataFrame, expression: Mapping[str, Any]
) -> None:
    for field in _expression_fields(expression):
        if field not in frame.columns:
            raise StrategyError(f"unknown field: {field}")


def _frame_column(frame: pd.DataFrame, field: str) -> pd.Series:
    values = frame[field]
    if not isinstance(values, pd.Series):
        raise StrategyError(f"strategy field must identify one column: {field}")
    return values


def _empty_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=frame.index, dtype=bool)


def _active_values(
    frame: pd.DataFrame, field: str, active: pd.Series
) -> tuple[pd.Series, pd.Series]:
    values = _frame_column(frame, field).loc[active]
    return values, values.isna()


def _apply_frame_missing_policy(
    result: pd.Series,
    missing_mask: pd.Series,
    *,
    policy: str,
) -> None:
    if not bool(missing_mask.any()):
        return
    if policy == "no_match":
        return
    if policy == "match":
        result.loc[missing_mask.index[missing_mask]] = True
        return
    if policy == "error":
        raise StrategyError("condition comparison failed: field value is missing")
    raise StrategyError(f"unsupported missing policy: {policy}")


def _coerce_numeric_series(values: pd.Series) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise StrategyError("condition comparison failed") from exc
    if bool(numeric.isna().any()):
        raise StrategyError(
            "condition comparison failed: field contains a non-numeric value"
        )
    return numeric


def _strict_equal_series(values: pd.Series, expected: Any) -> pd.Series:
    return values.map(lambda value: _strict_equal(value, expected)).astype(bool)


def _compare_series(
    values: pd.Series,
    operator: str,
    expected: Any,
    *,
    coercion: str = "auto",
) -> pd.Series:
    if coercion == "strict":
        if operator in {"in", "not_in"}:
            matched = pd.Series(False, index=values.index, dtype=bool)
            for candidate in expected:
                matched |= _strict_equal_series(values, candidate)
            return matched if operator == "in" else ~matched
        matched = _strict_equal_series(values, expected)
        return matched if operator == "==" else ~matched
    try:
        if operator in {"in", "not_in"}:
            candidates = list(expected)
            if candidates and all(_numeric_literal(item) for item in candidates):
                values = _coerce_numeric_series(values)
            matched = values.isin(candidates)
            return matched if operator == "in" else ~matched
        if _numeric_literal(expected):
            values = _coerce_numeric_series(values)
        if operator == "<":
            return values < expected
        if operator == "<=":
            return values <= expected
        if operator == ">":
            return values > expected
        if operator == ">=":
            return values >= expected
        if operator == "==":
            return values == expected
        if operator == "!=":
            return values != expected
    except StrategyError:
        raise
    except (TypeError, ValueError) as exc:
        raise StrategyError("condition comparison failed") from exc
    raise StrategyError(f"unsupported comparison operator: {operator}")


def _evaluate_frame_compare(
    frame: pd.DataFrame,
    expression: Mapping[str, Any],
    active: pd.Series,
) -> pd.Series:
    result = _empty_mask(frame)
    values, missing_mask = _active_values(frame, expression["field"], active)
    _apply_frame_missing_policy(
        result,
        missing_mask,
        policy=expression["missing"],
    )
    present_values = values.loc[~missing_mask]
    if not present_values.empty:
        comparison = _compare_series(
            present_values,
            expression["operator"],
            expression["value"],
            coercion=expression.get("coercion", "auto"),
        )
        result.loc[present_values.index] = comparison.fillna(False).astype(bool)
    return result


def _evaluate_frame_between(
    frame: pd.DataFrame,
    expression: Mapping[str, Any],
    active: pd.Series,
) -> pd.Series:
    result = _empty_mask(frame)
    values, missing_mask = _active_values(frame, expression["field"], active)
    _apply_frame_missing_policy(
        result,
        missing_mask,
        policy=expression["missing"],
    )
    present_values = values.loc[~missing_mask]
    if present_values.empty:
        return result
    lower = expression["lower"]
    upper = expression["upper"]
    if _numeric_literal(lower) and _numeric_literal(upper):
        present_values = _coerce_numeric_series(present_values)
    try:
        lower_match = (
            present_values >= lower
            if expression["include_lower"]
            else present_values > lower
        )
        upper_match = (
            present_values <= upper
            if expression["include_upper"]
            else present_values < upper
        )
        result.loc[present_values.index] = (
            (lower_match & upper_match).fillna(False).astype(bool)
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError("condition comparison failed") from exc
    return result


def _evaluate_frame_canonical(
    frame: pd.DataFrame,
    expression: Mapping[str, Any],
    active: pd.Series,
) -> pd.Series:
    """Evaluate canonical IR while preserving per-row short-circuit semantics."""

    op = expression["op"]
    if op == "compare":
        return _evaluate_frame_compare(frame, expression, active)
    if op == "between":
        return _evaluate_frame_between(frame, expression, active)
    if op in {"is_null", "is_not_null"}:
        result = _empty_mask(frame)
        values = _frame_column(frame, expression["field"]).loc[active]
        matched = values.isna()
        if op == "is_not_null":
            matched = ~matched
        result.loc[values.index] = matched.astype(bool)
        return result
    if op == "and":
        remaining = active.copy()
        for argument in expression["args"]:
            if not bool(remaining.any()):
                break
            remaining &= _evaluate_frame_canonical(frame, argument, remaining)
        return remaining
    if op == "or":
        matched = _empty_mask(frame)
        remaining = active.copy()
        for argument in expression["args"]:
            if not bool(remaining.any()):
                break
            argument_match = _evaluate_frame_canonical(frame, argument, remaining)
            matched |= argument_match
            remaining &= ~argument_match
        return matched
    if op == "not":
        return active & ~_evaluate_frame_canonical(frame, expression["arg"], active)
    if op == "n_of_k":
        matched = _empty_mask(frame)
        remaining = active.copy()
        counts = pd.Series(0, index=frame.index, dtype="int64")
        for argument in expression["args"]:
            if not bool(remaining.any()):
                break
            argument_match = _evaluate_frame_canonical(frame, argument, remaining)
            counts.loc[argument_match] += 1
            newly_matched = remaining & (counts >= expression["n"])
            matched |= newly_matched
            remaining &= ~newly_matched
        return matched
    raise StrategyError(f"unsupported expression op: {op}")


def evaluate_expression(row: Mapping[str, Any], expression: Mapping[str, Any]) -> bool:
    """Evaluate one validated expression against one row.

    Unknown columns fail closed just like the legacy MARVIS evaluator.  Present but
    missing atomic values follow the expression's explicit missing policy (default
    ``no_match``). Numeric literals use the legacy evaluator's numeric-string
    coercion so migration does not silently turn values such as ``"700"`` into a
    different decision.
    """

    if not isinstance(row, Mapping) and not isinstance(row, pd.Series):
        raise StrategyError("strategy row must be a mapping")
    canonical = canonicalize_expression(expression)
    _assert_known_fields(row, canonical)
    return _evaluate_canonical(row, canonical)


def evaluate_expression_frame(
    frame: pd.DataFrame, expression: Mapping[str, Any]
) -> pd.Series:
    """Evaluate one canonical expression as an index-aligned boolean mask.

    Column references are validated for the complete expression before evaluation,
    so a missing field cannot be hidden by boolean short-circuiting. Runtime value
    errors still honor the row evaluator's short-circuit behavior.
    """

    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("strategy rows must be a DataFrame")
    canonical = canonicalize_expression(expression)
    _assert_known_frame_fields(frame, canonical)
    working = frame.reset_index(drop=True)
    active = pd.Series(True, index=working.index, dtype=bool)
    result = _evaluate_frame_canonical(working, canonical, active)
    return pd.Series(
        result.to_numpy(dtype=bool, copy=False),
        index=frame.index,
        dtype=bool,
        name="matched",
    )


def evaluate_strategy_row(
    row: Mapping[str, Any], spec: StrategySpec | Mapping[str, Any]
) -> RowEvaluation:
    """Return the first matching stable rule id and its typed action for one row."""

    if not isinstance(row, Mapping) and not isinstance(row, pd.Series):
        raise StrategyError("strategy row must be a mapping")
    parsed = parse_strategy_spec(spec)
    for rule in parsed.rules:
        _assert_known_fields(row, rule.condition)
    for rule in parsed.rules:
        if _evaluate_canonical(row, rule.condition):
            return RowEvaluation(matched_rule_id=rule.rule_id, action=rule.action)
    return RowEvaluation(matched_rule_id=None, action=parsed.default_action)


def _filled_object_array(size: int, value: Any) -> np.ndarray:
    values = np.empty(size, dtype=object)
    values.fill(value)
    return values


def evaluate_strategy_frame(
    frame: pd.DataFrame,
    spec: StrategySpec | Mapping[str, Any],
) -> FrameEvaluation:
    """Vectorized first-match evaluation for large apply/backtest DataFrames.

    The function executes the same canonical IR and priority ordering as
    :func:`evaluate_strategy_row`. It returns action values as decisions plus the
    stable id of the matching rule; default decisions have no matched rule id.
    """

    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("strategy rows must be a DataFrame")
    parsed = parse_strategy_spec(spec)
    # Validate the complete schema up front. This intentionally happens before
    # checking frame length or applying first-match short-circuiting.
    for rule in parsed.rules:
        _assert_known_frame_fields(frame, rule.condition)

    working = frame.reset_index(drop=True)
    decisions = _filled_object_array(len(working), parsed.default_action.decision_value)
    matched_rule_ids = _filled_object_array(len(working), None)
    action_types = _filled_object_array(len(working), parsed.default_action.type)
    reason_codes = _filled_object_array(len(working), parsed.default_action.reason_code)
    remaining = pd.Series(True, index=working.index, dtype=bool)

    for rule in parsed.rules:
        if not bool(remaining.any()):
            break
        rule_matches = _evaluate_frame_canonical(
            working,
            rule.condition,
            remaining,
        )
        positions = np.flatnonzero(rule_matches.to_numpy(dtype=bool, copy=False))
        if positions.size:
            decisions[positions] = _filled_object_array(
                int(positions.size), rule.action.decision_value
            )
            matched_rule_ids[positions] = rule.rule_id
            action_types[positions] = rule.action.type
            reason_codes[positions] = rule.action.reason_code
            remaining &= ~rule_matches

    return FrameEvaluation(
        decisions=pd.Series(
            decisions,
            index=frame.index,
            dtype="object",
            name="decision",
        ),
        matched_rule_id=pd.Series(
            matched_rule_ids,
            index=frame.index,
            dtype="object",
            name="matched_rule_id",
        ),
        action_type=pd.Series(
            action_types,
            index=frame.index,
            dtype="object",
            name="action_type",
        ),
        reason_code=pd.Series(
            reason_codes,
            index=frame.index,
            dtype="object",
            name="reason_code",
        ),
    )


def evaluate_strategy_rows(
    rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    spec: StrategySpec | Mapping[str, Any],
) -> tuple[RowEvaluation, ...]:
    """Evaluate mappings row by row as the canonical correctness reference."""

    parsed = parse_strategy_spec(spec)
    if isinstance(rows, pd.DataFrame):
        iterable: Iterable[Mapping[str, Any]] = (row for _, row in rows.iterrows())
    else:
        iterable = rows
    return tuple(evaluate_strategy_row(row, parsed) for row in iterable)


__all__ = [
    "FrameEvaluation",
    "RowEvaluation",
    "evaluate_expression",
    "evaluate_expression_frame",
    "evaluate_strategy_frame",
    "evaluate_strategy_row",
    "evaluate_strategy_rows",
]
