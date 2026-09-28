"""Fit once and replay population-dependent derived features on any batch size."""

from __future__ import annotations

import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from marvis.feature.errors import FeatureError


AGGREGATIONS = frozenset({"mean", "max", "min", "std", "sum", "count"})


def _group_key(value):
    if pd.isna(value):
        return '["null"]'
    if isinstance(value, (str, np.str_)):
        return json.dumps(["text", str(value)], ensure_ascii=False)
    if isinstance(value, (bool, int, np.integer, np.bool_)):
        # Do not round 64-bit customer/category keys through binary64.
        return json.dumps(["number", str(int(value))])
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        number = float(value)
        return json.dumps(
            ["number", str(int(number)) if number.is_integer() else repr(number)]
        )
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC")
        return json.dumps(["timestamp", stamp.isoformat()])
    if isinstance(value, date):
        return json.dumps(["date", value.isoformat()])
    raise FeatureError(
        "aggregate group keys must be scalar text, numbers, dates or null"
    )


def fit_aggregate_parameters(
    frame,
    group_col,
    value_col,
    aggs,
    *,
    fit_mask=None,
    min_group_size=30,
    target_col=None,
):
    if not {group_col, value_col}.issubset(frame.columns):
        raise FeatureError("aggregate source columns are missing")
    if not aggs or any(agg not in AGGREGATIONS for agg in aggs):
        raise FeatureError("unsupported or empty aggregate functions")
    if value_col == target_col or group_col == target_col:
        raise FeatureError("aggregate_feature cannot use the target column")
    if isinstance(min_group_size, bool) or int(min_group_size) < 1:
        raise FeatureError("min_group_size must be positive")
    fit = frame if fit_mask is None else frame.loc[fit_mask]
    if fit.empty:
        raise FeatureError(
            "aggregate_feature fit frame is empty after excluding holdout rows"
        )
    values = pd.to_numeric(fit[value_col], errors="raise")
    keyed = pd.DataFrame({"key": fit[group_col].map(_group_key), "value": values})
    grouped = keyed.groupby("key", sort=False)["value"]
    stats = grouped.agg(aggs)
    defaults = values.agg(aggs)
    for agg in aggs:
        stats.loc[grouped.size() < int(min_group_size), agg] = defaults[agg]
        stats[agg] = stats[agg].fillna(defaults[agg])
    return {
        "group": group_col,
        "value": value_col,
        "aggs": list(aggs),
        "mapping": {
            str(key): {agg: float(row[agg]) for agg in aggs}
            for key, row in stats.iterrows()
        },
        "defaults": {agg: float(defaults[agg]) for agg in aggs},
    }


def apply_aggregate_parameters(frame, params):
    group = params["group"]
    if group not in frame:
        raise FeatureError("aggregate source group is missing")
    keys = frame[group].map(_group_key)
    out = frame.copy()
    columns = []
    for agg in params["aggs"]:
        column = f"{params['value']}_by_{group}_{agg}"
        if column in out:
            raise FeatureError(f"derived column already exists: {column}")
        default = params["defaults"][agg]
        out[column] = keys.map(
            lambda key: params["mapping"].get(key, {}).get(agg, default)
        )
        columns.append(column)
    return out, columns


def fit_rank_parameters(frame, column, fit_mask):
    values = pd.to_numeric(frame.loc[fit_mask, column], errors="raise")
    values = values[np.isfinite(values)]
    if values.empty:
        raise FeatureError("rank requires finite training values")
    ranks = pd.DataFrame({"value": values, "rank": values.rank(pct=True)})
    distinct = ranks.groupby("value", sort=True)["rank"].first()
    return {
        "column": column,
        "values": distinct.index.tolist(),
        "ranks": distinct.tolist(),
    }


def apply_rank_parameters(frame, params):
    column = params["column"]
    output = f"{column}__rank"
    if column not in frame or output in frame:
        raise FeatureError("rank source missing or derived column already exists")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    ranked = np.interp(values, params["values"], params["ranks"], left=0.0, right=1.0)
    ranked[~np.isfinite(values)] = np.nan
    return frame.assign(**{output: ranked}), [output]
