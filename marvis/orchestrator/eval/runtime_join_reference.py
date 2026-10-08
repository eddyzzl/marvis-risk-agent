"""Independent ordinary left-JOIN calculation over frozen source rows.

Uses Python lookup/grouping, never the production SQL JOIN or receipt verifier.
Physical row membership is computed from keys, not accepted from the receipt.
"""
from collections import defaultdict
from datetime import datetime
import hashlib
import math
import numbers
import re

import pandas as pd

from .runtime_archive_reader import _Unsupported


def _missing(value):
    result = pd.isna(value)
    if not isinstance(result, (bool, type(pd.isna(1)))):
        raise _Unsupported("nested JOIN values require a separate reference")
    return bool(result)


def _key(value, rule, side):
    if _missing(value):
        return None
    if isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        if not math.isfinite(value):
            raise _Unsupported("nonfinite JOIN keys")
        text = str(int(value)) if value == int(value) else str(value)
    else:
        text = str(value).strip()
    text = re.sub(r"\.0+$", "", text) if re.fullmatch(r"-?[0-9]+\.0+", text) else text
    if not text:
        return None
    method = rule["match_method"]
    if method == "exact":
        return text
    if method == "exact_lower":
        return text.lower()
    if method == "date":
        for pattern in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
                        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(text, pattern).date().isoformat()
            except ValueError:
                continue
        return None
    if method in {"hash:md5", "hash:sha256"}:
        if rule["transform_side"] in {side, "both"}:
            return hashlib.new(method.split(":")[1], text.encode()).hexdigest()
        return text.lower()
    raise _Unsupported("JOIN key transform requires a separate reference")


def join_reference(anchor, features, specs):
    """Return expected rows and physical memberships for every sequential join."""
    if len(features) != len(specs) or not specs:
        raise ValueError("JOIN source and rule count mismatch")
    current = anchor.copy()
    stages = []
    for feature, spec in zip(features, specs, strict=True):
        rules, strategy = spec["keys"], spec["dedup_strategy"]
        if not rules or strategy not in {None, "abort", "first", "last", "agg_mean", "agg_max"}:
            raise ValueError("invalid JOIN rules")
        if any(rule["anchor_col"] not in current or rule["feature_col"] not in feature
               or rule["transform_side"] not in {"both", "anchor", "feature"} for rule in rules):
            raise ValueError("JOIN keys are not declared source columns")
        if not current.columns.is_unique or not feature.columns.is_unique:
            raise ValueError("duplicate JOIN column names")
        mapping, occupied = {}, set(current.columns)
        for name in sorted(set(feature.columns) - {r["feature_col"] for r in rules}):
            alias, suffix = name, 2
            if alias in occupied:
                alias = f"feature_{name}"
                while alias in occupied:
                    alias, suffix = f"feature_{name}_{suffix}", suffix + 1
            mapping[name] = alias
            occupied.add(alias)
        if len(current) * len(occupied) > 2_000_000:
            raise _Unsupported("JOIN reference output cell limit")
        groups = defaultdict(list)
        rows = feature.to_dict("records")
        for index, row in enumerate(rows):
            key = tuple(_key(row[r["feature_col"]], r, "feature") for r in rules)
            if None not in key:
                groups[key].append(index)
        if strategy in {None, "abort"} and any(len(group) > 1 for group in groups.values()):
            raise ValueError("nonunique JOIN feature keys require a dedup decision")
        values, members, matched = [], [], 0
        for index, row in enumerate(current.to_dict("records")):
            key = tuple(_key(row[r["anchor_col"]], r, "anchor") for r in rules)
            selected = list(groups.get(key, [])) if None not in key else []
            if selected and strategy in {"first", "last"}:
                if "file_row_number" in feature:
                    # Native compatibility policy orders all columns for this reserved name.
                    # Keep NULL last for either sort direction, as in the SQL default.
                    candidates = feature.iloc[selected].copy()
                    ordinal = "__reference_ordinal"
                    while ordinal in candidates:
                        ordinal += "_"
                    candidates[ordinal] = selected
                    candidates = candidates.sort_values([*sorted(feature.columns), ordinal],
                        ascending=strategy == "first", na_position="last", kind="stable")
                    selected = [int(candidates[ordinal].iloc[0])]
                else:
                    selected = [selected[0 if strategy == "first" else -1]]
            added = {}
            for name, alias in mapping.items():
                nonnull = [rows[i][name] for i in selected if not _missing(rows[i][name])]
                if not nonnull:
                    added[alias] = None
                elif strategy == "agg_mean" and pd.api.types.is_numeric_dtype(feature[name]) and not pd.api.types.is_bool_dtype(feature[name]):
                    added[alias] = math.fsum(nonnull) / len(nonnull)
                elif strategy in {"agg_mean", "agg_max"}:
                    added[alias] = max(nonnull)
                else:
                    added[alias] = rows[selected[0]][name]
            values.append({**row, **added})
            members.append({"output_row": index, "left_row": index, "right_rows": selected or None})
            matched += bool(selected)
        current = pd.DataFrame(values, columns=[*current.columns, *mapping.values()])
        stages.append({"frame": current.copy(), "members": members, "mapping": mapping,
                       "matched_rows": matched, "match_rate": matched / len(current) if len(current) else 0.0})
    return current, stages
