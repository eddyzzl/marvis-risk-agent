from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.feature.binning import assign_bins
from marvis.feature.contracts import WOEResult
from marvis.feature.errors import FeatureError


def onehot_encode(
    df: pd.DataFrame,
    columns: list[str],
    *,
    max_categories: int = 50,
    handle_unknown: str = "ignore",
) -> tuple[pd.DataFrame, dict[str, list[object]]]:
    if handle_unknown not in {"ignore", "error"}:
        raise FeatureError("handle_unknown must be 'ignore' or 'error'")
    mapping = {}
    dummy_frames = []
    for column in columns:
        categories = _ordered_categories(df[column])
        if len(categories) > max_categories:
            raise FeatureError(f"{column} has too many categories")
        mapping[column] = categories
        dummy_frames.append(_dummy_frame(df[column], column, categories))
    encoded = df.drop(columns=columns)
    if dummy_frames:
        encoded = pd.concat([encoded, *dummy_frames], axis=1)
    return encoded, mapping


def label_encode(series: pd.Series) -> tuple[pd.Series, dict[object, int]]:
    categories = _ordered_categories(series)
    mapping = {category: index for index, category in enumerate(categories)}
    encoded = series.map(mapping).fillna(-1).astype(int)
    return encoded.rename(series.name), mapping


def woe_encode(df: pd.DataFrame, feature: str, woe: WOEResult) -> pd.Series:
    if feature not in df.columns:
        raise FeatureError(f"{feature} does not exist")
    edges = np.asarray(woe.edges, dtype=float)
    if len(woe.woe_by_bin) != edges.size - 1:
        raise FeatureError("woe_by_bin length must match edges")
    assigned = assign_bins(df[feature].to_numpy(dtype=float), edges)
    out = np.zeros(assigned.shape, dtype=float)
    for index, bin_index in enumerate(assigned):
        if bin_index >= 0:
            out[index] = float(woe.woe_by_bin[int(bin_index)])
        else:
            out[index] = float(woe.na_woe if woe.na_woe is not None else 0.0)
    return pd.Series(out, index=df.index, name=f"{feature}_woe")


def _ordered_categories(series: pd.Series) -> list[object]:
    return [value for value in pd.unique(series.dropna())]


def _dummy_frame(series: pd.Series, column: str, categories: list[object]) -> pd.DataFrame:
    data = {
        f"{column}_{category}": (series == category).astype(int)
        for category in categories
    }
    return pd.DataFrame(data, index=series.index)


__all__ = ["label_encode", "onehot_encode", "woe_encode"]
