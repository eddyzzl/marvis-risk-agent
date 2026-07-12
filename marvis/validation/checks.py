from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.validation.input_contracts import JsonScalar


def validate_binary_target(sample: pd.DataFrame, target_col: str) -> None:
    if target_col not in sample.columns:
        raise ValueError(f"target column {target_col!r} is missing")
    target = sample[target_col]
    invalid_mask = target.isna() | ~target.isin([0, 1, False, True, "0", "1"])
    if not invalid_mask.any():
        return
    found = ", ".join(sorted(str(value) for value in set(target[invalid_mask].dropna())))
    if not found:
        found = "missing"
    raise ValueError(
        f"binary target {target_col!r} must contain only 0/1 values; found: {found}"
    )


def binary_target_series(sample: pd.DataFrame, target_col: str) -> pd.Series:
    validate_binary_target(sample, target_col)
    return pd.to_numeric(sample[target_col], errors="raise").astype(int)


def validate_required_splits(
    sample: pd.DataFrame,
    *,
    split_col: str,
    split_values: dict[str, JsonScalar],
    required: tuple[str, ...] = ("train", "test", "oot"),
) -> None:
    if split_col not in sample.columns:
        raise ValueError(f"split column {split_col!r} is missing")
    missing_keys = [key for key in required if key not in split_values]
    if missing_keys:
        raise ValueError(f"split_values missing required keys: {', '.join(missing_keys)}")
    missing_splits = [
        key
        for key in required
        if not _typed_scalar_mask(sample[split_col], split_values[key]).any()
    ]
    if missing_splits:
        raise ValueError(
            "required sample split has no rows: " + ", ".join(missing_splits)
        )


def _typed_scalar_mask(values: pd.Series, expected: JsonScalar) -> pd.Series:
    expected_value = expected.item() if isinstance(expected, np.generic) else expected

    def matches(raw: object) -> bool:
        value = raw.item() if isinstance(raw, np.generic) else raw
        if type(value) is not type(expected_value):
            return False
        if value is None:
            return expected_value is None
        try:
            equal = value == expected_value
        except (TypeError, ValueError):
            return False
        return isinstance(equal, (bool, np.bool_)) and bool(equal)

    return values.map(matches).astype(bool)


def finite_score_series(
    scores,
    *,
    index,
    label: str,
) -> pd.Series:
    raw = pd.Series(scores, index=index)
    numeric = pd.to_numeric(raw, errors="coerce")
    finite_mask = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=raw.index)
    invalid_mask = raw.isna() | numeric.isna() | ~finite_mask
    if invalid_mask.any():
        rows = raw.index[invalid_mask].tolist()[:10]
        raise ValueError(f"{label} returned non-finite scores at rows: {rows}")
    return numeric.astype(float)
