"""Shared deterministic projection helpers for Strategy apply workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
import re

import pandas as pd

from marvis.packs.strategy.errors import StrategyError


DEFAULT_STRATEGY_APPLY_PREFIX = "strategy_"
DEFAULT_POOL_APPLY_PREFIX = "strategy_pool_"

_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_BASE_OUTPUT_SUFFIXES = (
    ("action", "action"),
    ("value", "value"),
    ("value_type", "value_type"),
    ("rule_id", "rule_id"),
)
_ENTRY_OUTPUT_SUFFIX = ("entry_id", "entry_id")
_REASON_OUTPUT_SUFFIX = ("reason_code", "reason_code")


@dataclass(frozen=True)
class ApplyOutputColumns:
    """Resolved output names, with optional Pool entry lineage."""

    action: str
    value: str
    value_type: str
    rule_id: str
    reason_code: str
    entry_id: str | None = None

    def as_dict(self) -> dict[str, str]:
        values = {
            "action": self.action,
            "value": self.value,
            "value_type": self.value_type,
            "rule_id": self.rule_id,
        }
        if self.entry_id is not None:
            values["entry_id"] = self.entry_id
        values["reason_code"] = self.reason_code
        return values


@dataclass(frozen=True)
class SerializedDecisionValues:
    """Parquet-safe decision values plus their exact JSON scalar types."""

    values: pd.Series = field(compare=False, repr=False)
    value_types: pd.Series = field(compare=False, repr=False)


def resolve_apply_output_columns(
    source_columns: Iterable[object],
    *,
    output_prefix: str | None = None,
    output_columns: Mapping[str, object] | None = None,
    default_prefix: str = DEFAULT_STRATEGY_APPLY_PREFIX,
    include_entry_id: bool = False,
) -> ApplyOutputColumns:
    """Resolve safe, non-overwriting, case-insensitively unique output names."""

    if output_prefix is not None and output_columns is not None:
        raise StrategyError(
            "strategy apply accepts output_prefix or output_columns, not both"
        )
    suffixes = [
        *_BASE_OUTPUT_SUFFIXES,
        *([_ENTRY_OUTPUT_SUFFIX] if include_entry_id else []),
        _REASON_OUTPUT_SUFFIX,
    ]
    if output_columns is None:
        prefix = default_prefix if output_prefix is None else output_prefix
        _require_safe_output_name(prefix, name="output_prefix", is_prefix=True)
        resolved = {key: f"{prefix}{suffix}" for key, suffix in suffixes}
    else:
        if not isinstance(output_columns, Mapping):
            raise StrategyError("output_columns must be an object")
        allowed = {key for key, _ in suffixes}
        unsupported = sorted(str(key) for key in set(output_columns) - allowed)
        if unsupported:
            raise StrategyError(
                "output_columns has unsupported fields: " + ", ".join(unsupported)
            )
        _require_safe_output_name(
            default_prefix,
            name="default_prefix",
            is_prefix=True,
        )
        resolved = {}
        for key, suffix in suffixes:
            value = output_columns.get(key)
            if value is None:
                resolved[key] = f"{default_prefix}{suffix}"
            elif not isinstance(value, str):
                raise StrategyError(f"output_columns.{key} must be a string")
            else:
                resolved[key] = value

    for key, column in resolved.items():
        _require_safe_output_name(column, name=f"output_columns.{key}")
    normalized = [column.casefold() for column in resolved.values()]
    if len(set(normalized)) != len(normalized):
        raise StrategyError(
            "strategy output column names must be case-insensitively unique"
        )
    source_names = {str(column).casefold() for column in source_columns}
    collisions = sorted(
        column for column in resolved.values() if column.casefold() in source_names
    )
    if collisions:
        raise StrategyError(
            "strategy output columns already exist (case-insensitive): "
            + ", ".join(collisions)
        )
    return ApplyOutputColumns(
        action=resolved["action"],
        value=resolved["value"],
        value_type=resolved["value_type"],
        rule_id=resolved["rule_id"],
        entry_id=resolved.get("entry_id"),
        reason_code=resolved["reason_code"],
    )


def serialize_strategy_decisions(
    decisions: pd.Series,
    *,
    strategy_type: str,
) -> SerializedDecisionValues:
    """Serialize heterogeneous JSON decisions without losing their exact type."""

    if not isinstance(decisions, pd.Series):
        raise StrategyError("strategy decisions must be a Series")
    decision_values = decisions.tolist()
    value_types = [_strategy_value_type(value) for value in decision_values]
    numeric_storage = strategy_type in {"limit", "pricing"} and all(
        value_type in {"integer", "number"} for value_type in value_types
    )
    values = [
        _strategy_storage_value(
            value,
            value_type=value_type,
            numeric_storage=numeric_storage,
        )
        for value, value_type in zip(
            decision_values,
            value_types,
            strict=True,
        )
    ]
    return SerializedDecisionValues(
        values=pd.Series(values, index=decisions.index, dtype="object"),
        value_types=pd.Series(value_types, index=decisions.index, dtype="object"),
    )


def deterministic_string_counts(values: pd.Series) -> dict[str, int]:
    """Return string-keyed, key-sorted counts including missing values."""

    if not isinstance(values, pd.Series):
        raise StrategyError("strategy count values must be a Series")
    counts = values.value_counts(dropna=False).to_dict()
    return {
        str(key): int(counts[key])
        for key in sorted(counts, key=lambda item: str(item))
    }


def deterministic_rule_counts(values: pd.Series) -> dict[str, int]:
    """Return deterministic counts for non-default rule or entry ids."""

    if not isinstance(values, pd.Series):
        raise StrategyError("strategy rule count values must be a Series")
    return deterministic_string_counts(values.loc[values.notna()].map(str))


def _require_safe_output_name(
    value: object,
    *,
    name: str,
    is_prefix: bool = False,
) -> None:
    limit = 48 if is_prefix else 64
    if not isinstance(value, str) or not value or len(value) > limit:
        raise StrategyError(f"{name} must be a non-empty safe identifier")
    if _SAFE_OUTPUT_NAME.fullmatch(value) is None:
        raise StrategyError(
            f"{name} must contain only ASCII letters, digits, and underscores "
            "and cannot start with a digit"
        )


def _strategy_storage_value(
    value: object,
    *,
    value_type: str,
    numeric_storage: bool,
) -> object:
    if numeric_storage:
        return value
    if value_type == "string":
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strategy_value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    raise StrategyError("strategy decision value must be JSON serializable")


__all__ = [
    "ApplyOutputColumns",
    "DEFAULT_POOL_APPLY_PREFIX",
    "DEFAULT_STRATEGY_APPLY_PREFIX",
    "SerializedDecisionValues",
    "deterministic_rule_counts",
    "deterministic_string_counts",
    "resolve_apply_output_columns",
    "serialize_strategy_decisions",
]
