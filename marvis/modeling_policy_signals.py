from __future__ import annotations

import re
from typing import Any


_MONOTONIC_CONSTRAINT_KEYS = ("monotonic_constraints", "monotone_constraints", "monotonic_directions")
_PARAM_CONTAINER_KEYS = ("params", "model_params", "fixed_params")


def monotonic_policy_profile(item: dict | None, scorecard_rows: list | None = None) -> dict:
    source = item if isinstance(item, dict) else {}
    rows = scorecard_rows if isinstance(scorecard_rows, list) else []
    scorecard_features = _scorecard_features(rows)
    row_direction_features = _scorecard_direction_features(rows)
    constraint_signal, constraint_features = _constraint_signal(source)
    constrained_features = sorted(set(row_direction_features) | set(constraint_features))

    if scorecard_features:
        missing = sorted(set(scorecard_features) - set(constrained_features))
        if not missing:
            coverage = "all"
            declared = True
        elif constrained_features:
            coverage = "partial"
            declared = False
        else:
            coverage = "none"
            declared = False
    elif constraint_signal:
        missing = []
        coverage = "declared"
        declared = True
    else:
        missing = []
        coverage = "none"
        declared = False

    return {
        "monotonicity_declared": declared,
        "monotonicity_coverage": coverage,
        "monotonicity_missing_features": missing,
        "monotonicity_constrained_features": constrained_features,
    }


def has_monotonic_policy(item: dict | None, scorecard_rows: list | None = None) -> bool:
    return bool(monotonic_policy_profile(item, scorecard_rows).get("monotonicity_declared"))


def _scorecard_features(rows: list) -> list[str]:
    features: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature") or "").strip()
        if feature and not feature.startswith("__"):
            features.add(feature)
    return sorted(features)


def _scorecard_direction_features(rows: list) -> list[str]:
    features: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature") or "").strip()
        if not feature or feature.startswith("__"):
            continue
        if _value_has_signal(row.get("monotonic_direction")):
            features.add(feature)
    return sorted(features)


def _constraint_signal(item: dict) -> tuple[bool, list[str]]:
    for key in _MONOTONIC_CONSTRAINT_KEYS:
        if key not in item:
            continue
        has_signal, features = _value_signal(item.get(key))
        if has_signal:
            return True, features
    for container_key in _PARAM_CONTAINER_KEYS:
        value = item.get(container_key)
        if isinstance(value, dict):
            has_signal, features = _constraint_signal(value)
            if has_signal:
                return True, features
    return False, []


def _value_signal(value: Any) -> tuple[bool, list[str]]:
    if isinstance(value, dict):
        features = [
            str(feature)
            for feature, raw in value.items()
            if str(feature).strip() and _value_has_signal(raw)
        ]
        return bool(features), sorted(features)
    if isinstance(value, (list, tuple, set)):
        return any(_value_has_signal(item) for item in value), []
    return _value_has_signal(value), []


def _value_has_signal(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return _string_has_signal(value)
    return value not in (None, "")


def _string_has_signal(value: str) -> bool:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"", "0", "0.0", "false", "none", "null", "no", "off", "[]", "()", "{}"}:
        return False
    tokens = [token for token in re.split(r"[\s,]+", text.strip("()[]{}")) if token]
    if not tokens:
        return False
    numbers: list[float] = []
    for token in tokens:
        try:
            numbers.append(float(token))
        except ValueError:
            return True
    return any(number != 0.0 for number in numbers)
