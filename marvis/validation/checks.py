from __future__ import annotations

import numpy as np
import pandas as pd


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
    split_values: dict[str, str],
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
        if sample[sample[split_col] == split_values[key]].empty
    ]
    if missing_splits:
        raise ValueError(
            "required sample split has no rows: " + ", ".join(missing_splits)
        )


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
