from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, ClassVar

from marvis.packs.strategy.errors import StrategyError


STRATEGY_DSL_SCHEMA_VERSION = "strategy.dsl.v1"
FIRST_MATCH_POLICY = "first_match"

_COMPARISON_OPERATORS = frozenset({"<", "<=", ">", ">=", "==", "!=", "in", "not_in"})
_COMPARISON_COERCIONS = frozenset({"auto", "strict"})
_MISSING_POLICIES = frozenset({"no_match", "match", "error"})
_ACTION_TYPES = frozenset(
    {"approval", "reject", "review", "limit", "pricing", "segment"}
)
_VALUE_ACTION_TYPES = frozenset({"limit", "pricing", "segment"})
_STRATEGY_TYPES = frozenset({"approval", "reject", "limit", "pricing", "segmentation"})
_STRATEGY_ACTION_TYPES = {
    "approval": frozenset({"approval", "reject", "review"}),
    "reject": frozenset({"approval", "reject", "review"}),
    "limit": frozenset({"limit"}),
    "pricing": frozenset({"pricing"}),
    "segmentation": frozenset({"segment"}),
}


def _object(payload: Mapping[str, Any] | object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise StrategyError(f"{name} must be an object")
    for key in payload:
        if not isinstance(key, str):
            raise StrategyError(f"{name} keys must be strings")
    return payload


def _only_keys(payload: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise StrategyError(f"{name} has unsupported fields: {', '.join(unexpected)}")


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _canonical_json_value(value: Any, *, path: str = "value") -> Any:
    """Return an immutable-input-safe JSON value and reject lossy serialization.

    Strategy hashes must not depend on ``default=str`` or object identity.  Numeric
    scalar types (including numpy integer/float scalars) are normalized to Python
    JSON numbers, while NaN/Infinity and arbitrary Python objects fail closed.
    """

    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise StrategyError(f"{path} must be a finite JSON number")
        return normalized
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrategyError(f"{path} keys must be strings")
            result[key] = _canonical_json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise StrategyError(f"{path} must be JSON serializable")


def _field_name(payload: Mapping[str, Any]) -> str:
    value = payload.get("field")
    if not isinstance(value, str) or not value.strip():
        raise StrategyError("expression field must be a non-empty string")
    # Column names are data identifiers, not display copy. Preserve significant
    # leading/trailing whitespace rather than silently pointing at another field.
    return value


def _missing_policy(payload: Mapping[str, Any]) -> str:
    policy = payload.get("missing", "no_match")
    if not isinstance(policy, str) or policy not in _MISSING_POLICIES:
        allowed = ", ".join(sorted(_MISSING_POLICIES))
        raise StrategyError(f"expression missing policy must be one of: {allowed}")
    return policy


def canonicalize_expression(expression: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one Strategy DSL v1 typed expression.

    The output contains only schema-owned keys and JSON-native values, making it
    safe to persist, hash, render, and execute without evaluating free-form code.
    """

    payload = _object(expression, name="expression")
    op = _nonempty_string(payload.get("op"), name="expression op")

    if op == "compare":
        _only_keys(
            payload,
            {"op", "field", "operator", "value", "missing", "coercion"},
            name="compare expression",
        )
        field_name = _field_name(payload)
        missing = _missing_policy(payload)
        operator = _nonempty_string(payload.get("operator"), name="comparison operator")
        if operator not in _COMPARISON_OPERATORS:
            raise StrategyError(f"unsupported comparison operator: {operator}")
        if "value" not in payload:
            raise StrategyError("compare expression requires value")
        value = _canonical_json_value(payload["value"], path="compare value")
        if operator in {"in", "not_in"} and not isinstance(value, list):
            raise StrategyError(f"comparison operator {operator} requires a list value")
        if operator not in {"in", "not_in"} and isinstance(value, dict | list):
            raise StrategyError(
                f"comparison operator {operator} requires a scalar value"
            )
        coercion = payload.get("coercion", "auto")
        if not isinstance(coercion, str) or coercion not in _COMPARISON_COERCIONS:
            allowed = ", ".join(sorted(_COMPARISON_COERCIONS))
            raise StrategyError(f"comparison coercion must be one of: {allowed}")
        if coercion == "strict" and operator not in {"==", "!=", "in", "not_in"}:
            raise StrategyError(
                "strict comparison coercion supports only equality and membership operators"
            )
        result = {
            "op": op,
            "field": field_name,
            "operator": operator,
            "value": value,
            "missing": missing,
        }
        # Preserve the historical canonical form and hashes for ordinary DSL.
        # Strict is opt-in for type-preserving categorical evidence.
        if coercion == "strict":
            result["coercion"] = "strict"
        return result

    if op == "between":
        _only_keys(
            payload,
            {
                "op",
                "field",
                "lower",
                "upper",
                "include_lower",
                "include_upper",
                "missing",
            },
            name="between expression",
        )
        field_name = _field_name(payload)
        missing = _missing_policy(payload)
        if "lower" not in payload or "upper" not in payload:
            raise StrategyError("between expression requires lower and upper")
        lower = _canonical_json_value(payload["lower"], path="between lower")
        upper = _canonical_json_value(payload["upper"], path="between upper")
        if isinstance(lower, dict | list) or isinstance(upper, dict | list):
            raise StrategyError("between bounds must be scalar values")
        include_lower = payload.get("include_lower", True)
        include_upper = payload.get("include_upper", True)
        if not isinstance(include_lower, bool) or not isinstance(include_upper, bool):
            raise StrategyError("between inclusion flags must be booleans")
        return {
            "op": op,
            "field": field_name,
            "lower": lower,
            "upper": upper,
            "include_lower": include_lower,
            "include_upper": include_upper,
            "missing": missing,
        }

    if op in {"is_null", "is_not_null"}:
        _only_keys(payload, {"op", "field"}, name=f"{op} expression")
        return {"op": op, "field": _field_name(payload)}

    if op in {"and", "or", "n_of_k"}:
        allowed = {"op", "args", "n"} if op == "n_of_k" else {"op", "args"}
        _only_keys(payload, allowed, name=f"{op} expression")
        args_value = payload.get("args")
        if not isinstance(args_value, Sequence) or isinstance(
            args_value, str | bytes | bytearray
        ):
            raise StrategyError(f"{op} expression args must be a list")
        if not args_value:
            raise StrategyError(f"{op} expression requires at least one argument")
        args = [
            canonicalize_expression(_object(arg, name=f"{op} argument"))
            for arg in args_value
        ]
        if op != "n_of_k":
            return {"op": op, "args": args}
        n = payload.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            raise StrategyError("n_of_k n must be an integer")
        if n < 1 or n > len(args):
            raise StrategyError(f"n_of_k n must be between 1 and {len(args)}")
        return {"op": op, "n": n, "args": args}

    if op == "not":
        _only_keys(payload, {"op", "arg"}, name="not expression")
        if "arg" not in payload:
            raise StrategyError("not expression requires arg")
        return {
            "op": op,
            "arg": canonicalize_expression(
                _object(payload["arg"], name="not argument")
            ),
        }

    raise StrategyError(f"unsupported expression op: {op}")


@dataclass(frozen=True)
class StrategyAction:
    type: str
    value: Any = None
    reason_code: str | None = None
    stop: bool = True
    output_value: Any = None

    _DEFAULT_VALUES: ClassVar[dict[str, str]] = {
        "approval": "approve",
        "reject": "reject",
        "review": "review",
    }

    def __post_init__(self) -> None:
        action_type = _nonempty_string(self.type, name="action type")
        if action_type not in _ACTION_TYPES:
            raise StrategyError(f"unsupported action type: {action_type}")
        value = self.value
        fixed_value = self._DEFAULT_VALUES.get(action_type)
        if value is None and fixed_value is not None:
            value = fixed_value
        elif fixed_value is not None and value != fixed_value:
            raise StrategyError(f"action {action_type} value must be {fixed_value!r}")
        if value is None and action_type in _VALUE_ACTION_TYPES:
            raise StrategyError(f"action {action_type} requires a value")
        value = _canonical_json_value(value, path="action value")
        if action_type == "limit" and (
            not isinstance(value, int | float) or isinstance(value, bool) or value < 0
        ):
            raise StrategyError("action limit value must be a non-negative number")
        if action_type == "pricing" and (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not 0 <= value <= 1
        ):
            raise StrategyError(
                "action pricing value must be an annual decimal rate between 0 and 1"
            )
        if action_type == "segment" and (
            isinstance(value, bool)
            or not isinstance(value, str | int | float)
            or (isinstance(value, str) and not value.strip())
        ):
            raise StrategyError("action segment value must be a non-empty scalar id")
        output_value = self.output_value
        if output_value is not None:
            output_value = _canonical_json_value(
                output_value, path="action output_value"
            )
            if (
                action_type in self._DEFAULT_VALUES
                and output_value in set(self._DEFAULT_VALUES.values())
                and output_value != value
            ):
                raise StrategyError(
                    f"action {action_type} output_value contradicts its typed value"
                )
        reason_code = self.reason_code
        if reason_code is not None:
            reason_code = _nonempty_string(reason_code, name="action reason_code")
        if not isinstance(self.stop, bool):
            raise StrategyError("action stop must be a boolean")
        if not self.stop:
            raise StrategyError("strategy.dsl.v1 first_match supports only stop=true")
        object.__setattr__(self, "type", action_type)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "output_value", output_value)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StrategyAction:
        value = _object(payload, name="action")
        _only_keys(
            value,
            {"type", "value", "reason_code", "stop", "output_value"},
            name="action",
        )
        return cls(
            type=value.get("type"),
            value=value.get("value"),
            reason_code=value.get("reason_code"),
            stop=value.get("stop", True),
            output_value=value.get("output_value"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "value": _canonical_json_value(self.value, path="action value"),
            "reason_code": self.reason_code,
            "stop": self.stop,
        }
        if self.output_value is not None:
            payload["output_value"] = _canonical_json_value(
                self.output_value, path="action output_value"
            )
        return payload

    @property
    def decision_value(self) -> Any:
        return self.value if self.output_value is None else self.output_value


@dataclass(frozen=True)
class StrategyRuleSpec:
    rule_id: str
    priority: int
    condition: Mapping[str, Any]
    action: StrategyAction

    def __post_init__(self) -> None:
        rule_id = _nonempty_string(self.rule_id, name="rule_id")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise StrategyError("rule priority must be an integer")
        if self.priority < 0:
            raise StrategyError("rule priority must be >= 0")
        condition = canonicalize_expression(self.condition)
        action = self.action
        if isinstance(action, Mapping):
            action = StrategyAction.from_dict(action)
        if not isinstance(action, StrategyAction):
            raise StrategyError("rule action must be a StrategyAction")
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "action", action)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StrategyRuleSpec:
        value = _object(payload, name="rule")
        _only_keys(value, {"rule_id", "priority", "condition", "action"}, name="rule")
        return cls(
            rule_id=value.get("rule_id"),
            priority=value.get("priority"),
            condition=_object(value.get("condition"), name="rule condition"),
            action=StrategyAction.from_dict(
                _object(value.get("action"), name="rule action")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "priority": self.priority,
            "condition": canonicalize_expression(self.condition),
            "action": self.action.to_dict(),
        }


@dataclass(frozen=True)
class StrategySpec:
    strategy_type: str
    default_action: StrategyAction
    rules: tuple[StrategyRuleSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = STRATEGY_DSL_SCHEMA_VERSION
    match_policy: str = FIRST_MATCH_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != STRATEGY_DSL_SCHEMA_VERSION:
            raise StrategyError(
                f"unsupported strategy schema_version: {self.schema_version}"
            )
        strategy_type = _nonempty_string(self.strategy_type, name="strategy_type")
        if strategy_type not in _STRATEGY_TYPES:
            raise StrategyError(f"unsupported strategy_type: {strategy_type}")
        if self.match_policy != FIRST_MATCH_POLICY:
            raise StrategyError(
                f"unsupported strategy match_policy: {self.match_policy}"
            )
        default_action = self.default_action
        if isinstance(default_action, Mapping):
            default_action = StrategyAction.from_dict(default_action)
        if not isinstance(default_action, StrategyAction):
            raise StrategyError("default_action must be a StrategyAction")
        rules = tuple(
            rule
            if isinstance(rule, StrategyRuleSpec)
            else StrategyRuleSpec.from_dict(rule)
            for rule in self.rules
        )
        rules = tuple(sorted(rules, key=lambda rule: rule.priority))
        rule_ids = [rule.rule_id for rule in rules]
        duplicate_ids = sorted(
            {rule_id for rule_id in rule_ids if rule_ids.count(rule_id) > 1}
        )
        if duplicate_ids:
            raise StrategyError(f"duplicate rule_id: {', '.join(duplicate_ids)}")
        priorities = [rule.priority for rule in rules]
        duplicate_priorities = sorted(
            {priority for priority in priorities if priorities.count(priority) > 1}
        )
        if duplicate_priorities:
            rendered = ", ".join(str(priority) for priority in duplicate_priorities)
            raise StrategyError(f"duplicate rule priority: {rendered}")
        allowed_action_types = _STRATEGY_ACTION_TYPES[strategy_type]
        incompatible = sorted(
            {
                action.type
                for action in (default_action, *(rule.action for rule in rules))
                if action.type not in allowed_action_types
            }
        )
        if incompatible:
            rendered = ", ".join(incompatible)
            raise StrategyError(
                f"action type is not allowed for {strategy_type}: {rendered}"
            )
        metadata = _canonical_json_value(
            _object(self.metadata, name="strategy metadata"), path="metadata"
        )
        lineage = metadata.get("lineage", {})
        if not isinstance(lineage, dict):
            raise StrategyError("strategy metadata lineage must be an object")
        metadata["lineage"] = lineage
        object.__setattr__(self, "strategy_type", strategy_type)
        object.__setattr__(self, "default_action", default_action)
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StrategySpec:
        value = _object(payload, name="strategy spec")
        _only_keys(
            value,
            {
                "schema_version",
                "strategy_type",
                "match_policy",
                "default_action",
                "rules",
                "metadata",
            },
            name="strategy spec",
        )
        rules_value = value.get("rules", [])
        if not isinstance(rules_value, Sequence) or isinstance(
            rules_value, str | bytes | bytearray
        ):
            raise StrategyError("strategy rules must be a list")
        return cls(
            schema_version=value.get("schema_version", STRATEGY_DSL_SCHEMA_VERSION),
            strategy_type=value.get("strategy_type"),
            match_policy=value.get("match_policy", FIRST_MATCH_POLICY),
            default_action=StrategyAction.from_dict(
                _object(value.get("default_action"), name="strategy default_action")
            ),
            rules=tuple(
                StrategyRuleSpec.from_dict(_object(rule, name="strategy rule"))
                for rule in rules_value
            ),
            metadata=_object(value.get("metadata", {}), name="strategy metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_type": self.strategy_type,
            "match_policy": self.match_policy,
            "default_action": self.default_action.to_dict(),
            "rules": [rule.to_dict() for rule in self.rules],
            "metadata": _canonical_json_value(self.metadata, path="metadata"),
        }


def parse_strategy_spec(payload: StrategySpec | Mapping[str, Any]) -> StrategySpec:
    if isinstance(payload, StrategySpec):
        return payload
    return StrategySpec.from_dict(payload)


def _without_nonsemantic_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    # Metadata is persisted evidence and presentation context, never executable
    # strategy semantics.  Keeping lineage, source task ids, or display copy in the
    # effect hash would make an identical rule definition appear to be a different
    # strategy merely because it was copied, renamed, or versioned.
    payload["metadata"] = {}
    return payload


def canonical_strategy_json(
    spec: StrategySpec | Mapping[str, Any], *, include_display_metadata: bool = True
) -> str:
    payload = parse_strategy_spec(spec).to_dict()
    if not include_display_metadata:
        payload = _without_nonsemantic_metadata(payload)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def strategy_spec_hash(spec: StrategySpec | Mapping[str, Any]) -> str:
    canonical = canonical_strategy_json(spec, include_display_metadata=False).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "FIRST_MATCH_POLICY",
    "STRATEGY_DSL_SCHEMA_VERSION",
    "StrategyAction",
    "StrategyRuleSpec",
    "StrategySpec",
    "canonical_strategy_json",
    "canonicalize_expression",
    "parse_strategy_spec",
    "strategy_spec_hash",
]
