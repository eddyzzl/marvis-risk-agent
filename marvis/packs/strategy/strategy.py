from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Literal

import pandas as pd

from marvis.data.errors import ScoreDirectionConflictError
from marvis.packs.strategy.contracts import Strategy, StrategyRule
from marvis.packs.strategy.errors import StrategyError


# S1a: a rule-evaluation consumer has no "declared" score_direction to check against
# (the direction lives implicitly in each rule's comparison operator) -- these two
# flags are the two possible per-rule directions, distinct from the enum in
# marvis.data.direction (which describes a *declared* direction, not an *inferred* one).
_RuleDirectionFlag = Literal["gte_style", "lte_style"]


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

    if score_col:
        _raise_on_inconsistent_rule_directions(parsed_rules, score_col)

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


def infer_strategy_rule_direction(rules: list[StrategyRule], score_col: str | None) -> str | None:
    """S1a: best-effort direction implied by a strategy's rules' comparison operators
    against ``score_col`` (not a declared direction -- see module docstring note on
    _RuleDirectionFlag). Returns None when score_col is falsy, no rule references it
    via a simple top-level comparison, or the operator styles disagree (ambiguous,
    including the legitimate case where opposite styles agree with each other via
    opposite decisions -- see _raise_on_inconsistent_rule_directions). Exposed
    separately from build_strategy so the tool layer can surface it in a response
    dict without adding a field to the Strategy dataclass (this is a self-check
    byproduct, not part of the persisted contract -- see spec §2.3).
    """
    if not score_col:
        return None
    flags = _rule_direction_flags(rules, score_col)
    distinct = {flag for _, flag in flags}
    if len(distinct) != 1:
        return None
    return "higher_is_better" if distinct == {"gte_style"} else "higher_is_riskier"


def _raise_on_inconsistent_rule_directions(rules: list[StrategyRule], score_col: str) -> None:
    """Flag rules whose comparison operators against score_col disagree in a way that
    cannot be explained by a coherent single-direction strategy.

    The spec's naive check (any two distinct operator styles == conflict) has a false
    positive: a common, coherent "banded cutoff" strategy like
    ``score < 600 -> reject`` + ``score >= 720 -> approve`` uses opposite operator
    styles but agrees that higher score is better (the low band is explicitly
    rejected, the high band explicitly approved) -- opposite styles paired with
    opposite decisions is exactly what a monotonic cutoff strategy looks like, not a
    contradiction. The real contradiction is opposite styles that land on the SAME
    decision (e.g. ``score < 500 -> reject`` + ``score >= 900 -> reject`` with nothing
    ever approved), which no single score direction can explain. Same-style rules
    with different decisions (e.g. two ``>=`` rules, one approve one reject) are a
    rule-ordering pattern (first match wins), not a direction question, and are left
    alone here -- see test_apply_strategy_uses_first_matching_rule.
    """
    flags = _rule_direction_flags(rules, score_col)
    by_style: dict[_RuleDirectionFlag, set] = {"gte_style": set(), "lte_style": set()}
    for rule, style in flags:
        by_style[style].add(rule.decision)
    conflicting_decisions = by_style["gte_style"] & by_style["lte_style"]
    if conflicting_decisions:
        raise ScoreDirectionConflictError(
            tool="build_strategy",
            score_col=score_col,
            reason="rules reference score_col with inconsistent comparison direction",
            conflicting_rules=[
                rule.condition
                for rule, style in flags
                if rule.decision in conflicting_decisions
            ],
        )


def _rule_direction_flags(
    rules: list[StrategyRule], score_col: str
) -> list[tuple[StrategyRule, _RuleDirectionFlag]]:
    flags: list[tuple[StrategyRule, _RuleDirectionFlag]] = []
    for rule in rules:
        direction = _infer_condition_direction(rule.condition, score_col)
        if direction is not None:
            flags.append((rule, direction))
    return flags


def _infer_condition_direction(condition: str, score_col: str) -> _RuleDirectionFlag | None:
    expression = _parse_condition(condition)
    for clause in _flatten_top_level_compares(expression.body):
        if not (isinstance(clause, ast.Compare) and isinstance(clause.left, ast.Name)):
            continue
        if clause.left.id != score_col:
            continue
        op = clause.ops[0]
        if isinstance(op, ast.GtE | ast.Gt):
            return "gte_style"
        if isinstance(op, ast.LtE | ast.Lt):
            return "lte_style"
        return None  # Eq/NotEq/In/NotIn carry no direction
    return None


def _flatten_top_level_compares(node: ast.AST) -> list[ast.AST]:
    """Expand the direct children of a top-level BoolOp into their leaf clauses,
    without recursing into nested BoolOps -- direction inference only looks at
    clauses that directly compare score_col, not at the overall boolean structure
    (see spec §2.3: complex boolean combinations have no well-defined "direction")."""
    if isinstance(node, ast.BoolOp):
        clauses: list[ast.AST] = []
        for value in node.values:
            clauses.extend(_flatten_top_level_compares(value))
        return clauses
    return [node]


def apply_strategy(df: pd.DataFrame, strategy: Strategy) -> pd.Series:
    decisions = pd.Series([strategy.default_decision] * len(df), index=df.index, dtype="object")
    assigned = pd.Series(False, index=df.index)
    for rule in strategy.rules:
        mask = _safe_eval_condition(df, rule.condition) & ~assigned
        decisions.loc[mask] = rule.value if rule.value is not None else rule.decision
        assigned |= mask
    return decisions


def evaluate_condition_mask(df: pd.DataFrame, condition: str) -> pd.Series:
    """Boolean hit mask for one strategy condition string against ``df``.

    S4: the single, shared condition-evaluation entry point. build_strategy
    (via apply_strategy), rule mining (rules.mine_rules) and rule-set evaluation
    (rules.evaluate_rule_set) all resolve a condition's per-row hits through this
    one function, so a condition string produced by any of the three lands on the
    exact same hit set -- there is never a second, drifting evaluator. It is the
    same validated ``_safe_eval_condition`` apply_strategy already uses; exposed
    publicly only so mine/evaluate reuse it by import instead of re-implementing.
    """
    return _safe_eval_condition(df, condition)


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


__all__ = ["apply_strategy", "build_strategy", "evaluate_condition_mask", "infer_strategy_rule_direction"]
