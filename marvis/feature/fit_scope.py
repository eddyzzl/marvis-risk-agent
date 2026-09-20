"""Explicit fitting membership shared by supervised and statistical transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.feature.errors import FeatureError, FitRequiresSplitError


_EVALUATION_VALUES = frozenset({"test", "oot", "validation", "valid", "holdout"})


def _names(value, *, default, name):
    values = default if value is None else value
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in values
    ):
        raise FeatureError(f"{name} must contain explicit partition names")
    return tuple(dict.fromkeys(values))


def fit_membership(
    frame: pd.DataFrame, inputs: dict, *, tool: str, dataset_id: str,
) -> tuple[np.ndarray, str]:
    """Select declared training rows; an excluded holdout list cannot add rows.

    Full-pool exploration still requires an explicit boolean acknowledgement and
    is labelled full. Missing or unrecognised partition identity is never train.
    """
    allow_full = inputs.get("allow_full_fit", False)
    if not isinstance(allow_full, bool):
        raise FeatureError("allow_full_fit must be a boolean")
    split_col = inputs.get("split_col")
    if not split_col:
        if allow_full:
            return np.ones(len(frame), dtype=bool), "full"
        raise FitRequiresSplitError(tool=tool, dataset_id=dataset_id)
    if not isinstance(split_col, str) or split_col not in frame:
        raise FeatureError("split partition column is missing")
    train = _names(inputs.get("train_values"), default=("train",), name="train_values")
    holdout = _names(inputs.get("holdout_values"), default=("test", "oot"), name="holdout_values")
    if not train:
        raise FeatureError("train_values requires a training partition")
    if any(value.casefold() in _EVALUATION_VALUES for value in train):
        raise FeatureError("evaluation partition cannot be used for fitting")
    if set(train).intersection(holdout):
        raise FeatureError("training and holdout partition identities overlap")
    values = frame[split_col]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise FeatureError("split partition identity is missing")
    labels = values.astype(str)
    recognised = labels.isin((*train, *holdout)) | labels.str.casefold().isin(_EVALUATION_VALUES)
    if not recognised.all():
        raise FeatureError("unknown partition identity; declare train_values or holdout_values explicitly")
    mask = labels.isin(train).to_numpy(dtype=bool)
    if not mask.any():
        raise FeatureError(f"{tool} fit frame is empty after selecting training partition")
    return mask, "train"
