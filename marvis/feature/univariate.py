from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from marvis.feature.binning import (
    assign_bins,
    chimerge_edges,
    degraded_bin_diagnostic,
    equal_frequency_edges,
    equal_width_edges,
    tree_edges,
)
from marvis.feature.errors import FeatureError
from marvis.feature.iv import _smoothed_woe_iv
from marvis.feature.metrics import feature_auc, feature_ks


SCHEMA_VERSION = "univariate-analysis-result.v1"
_NUMERIC_METHODS = ("equal_frequency", "equal_width", "chimerge", "tree")
_ALL_METHODS = (*_NUMERIC_METHODS, "categorical")
_METHOD_ORDER = {method: index for index, method in enumerate(_ALL_METHODS)}
_MAX_SAFE_JSON_INTEGER = 2**53 - 1


def analyze_univariate(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    methods: Sequence[str] | None = None,
    feature_types: Mapping[str, str] | None = None,
    bin_count: int = 10,
    sentinel_values: Mapping[str, Sequence[object]] | None = None,
    loan_amount: str | None = None,
    overdue_amount: str | None = None,
    max_rows: int = 1_000_000,
    max_features: int = 50,
    max_bins: int = 20,
    max_categories: int = 100,
    min_bin_pct: float = 0.02,
    smoothing: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute bounded, deterministic single-feature strategy evidence.

    This kernel never executes user code and never substitutes another binning
    method when the requested method is unavailable.  Such cases are returned as
    typed evidence so callers can ask the user for a different method or data.
    Numeric intervals use ``[lower, upper)`` semantics, exactly matching
    :func:`marvis.feature.binning.assign_bins`.
    """

    _validate_request(
        frame,
        features=features,
        target=target,
        methods=methods,
        feature_types=feature_types,
        bin_count=bin_count,
        max_rows=max_rows,
        max_features=max_features,
        max_bins=max_bins,
        max_categories=max_categories,
        min_bin_pct=min_bin_pct,
        smoothing=smoothing,
        seed=seed,
        loan_amount=loan_amount,
        overdue_amount=overdue_amount,
    )
    feature_names = tuple(features)
    target_values = _binary_target(frame[target], target)
    loan_values = _amount_values(frame, loan_amount)
    overdue_values = _amount_values(frame, overdue_amount)
    if sentinel_values is not None and not isinstance(sentinel_values, Mapping):
        raise FeatureError("sentinel_values must be a mapping")
    sentinels = sentinel_values or {}
    if any(not isinstance(key, str) for key in sentinels):
        raise FeatureError("sentinel_values keys must be feature names")
    unknown_sentinel_features = sorted(set(sentinels) - set(feature_names))
    if unknown_sentinel_features:
        raise FeatureError(
            "sentinel_values contains unrequested features: "
            + ", ".join(unknown_sentinel_features)
        )
    requested_methods = _normalize_methods(methods)
    type_overrides = dict(feature_types or {})

    feature_results: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    for feature in feature_names:
        resolved_type = type_overrides.get(feature) or _infer_feature_type(
            frame[feature]
        )
        feature_sentinels = _normalize_sentinels(
            sentinels.get(feature, ()),
            feature,
            max_values=(max_bins if resolved_type == "numeric" else max_categories),
        )
        if requested_methods is None:
            method_names = (
                _NUMERIC_METHODS if resolved_type == "numeric" else ("categorical",)
            )
        elif resolved_type == "categorical" and "categorical" not in requested_methods:
            # Public Candidate Lab requests select numeric binning methods only;
            # categorical fields always retain their fixed equal-value method.
            method_names = ("categorical",)
        else:
            method_names = requested_methods
        method_results = []
        for method in method_names:
            result = _analyze_method(
                frame[feature],
                target_values,
                feature=feature,
                feature_type=resolved_type,
                method=method,
                bin_count=bin_count,
                sentinels=feature_sentinels,
                loan_values=loan_values,
                overdue_values=overdue_values,
                max_categories=max_categories,
                min_bin_pct=min_bin_pct,
                smoothing=smoothing,
                seed=seed,
            )
            method_results.append(result)
            if result["status"] == "available":
                rankings.append(
                    {
                        "feature": feature,
                        "method": method,
                        "iv": result["metrics"]["iv"],
                        "ks": result["metrics"]["ks"],
                        "auc": result["metrics"]["auc"],
                    }
                )
        feature_results.append(
            {
                "feature": feature,
                "feature_type": resolved_type,
                "row_count": int(len(frame)),
                "missing_rate": _finite_float(float(frame[feature].isna().mean())),
                "sentinel_values": [_json_scalar(value) for value in feature_sentinels],
                "methods": method_results,
            }
        )

    rankings.sort(
        key=lambda item: (
            -item["iv"],
            -item["ks"],
            item["feature"],
            _METHOD_ORDER[item["method"]],
        )
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "target_definition": {"good": 0, "bad": 1},
        "row_count": int(len(frame)),
        "feature_count": len(feature_names),
        "parameters": {
            "bin_count": int(bin_count),
            "smoothing": float(smoothing),
            "min_bin_pct": float(min_bin_pct),
            "seed": int(seed),
            "loan_amount": loan_amount,
            "overdue_amount": overdue_amount,
        },
        "features": feature_results,
        "rankings": rankings,
        "resource_budget": {
            "max_rows": int(max_rows),
            "max_features": int(max_features),
            "max_bins": int(max_bins),
            "max_categories": int(max_categories),
            "rows_used": int(len(frame)),
            "features_used": len(feature_names),
            "method_runs": sum(len(item["methods"]) for item in feature_results),
            "truncated": False,
        },
    }
    _assert_finite_json(payload)
    return payload


def _analyze_method(
    series: pd.Series,
    target: np.ndarray,
    *,
    feature: str,
    feature_type: str,
    method: str,
    bin_count: int,
    sentinels: tuple[object, ...],
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    max_categories: int,
    min_bin_pct: float,
    smoothing: float,
    seed: int,
) -> dict[str, Any]:
    if (feature_type == "numeric") != (method in _NUMERIC_METHODS):
        expected = "numeric" if method in _NUMERIC_METHODS else "categorical"
        return _unavailable(
            method, "incompatible_feature_type", expected=expected, actual=feature_type
        )
    if feature_type == "categorical":
        return _categorical_result(
            series,
            target,
            feature=feature,
            method=method,
            sentinels=sentinels,
            loan_values=loan_values,
            overdue_values=overdue_values,
            max_categories=max_categories,
            min_bin_pct=min_bin_pct,
            smoothing=smoothing,
        )
    return _numeric_result(
        series,
        target,
        feature=feature,
        method=method,
        bin_count=bin_count,
        sentinels=sentinels,
        loan_values=loan_values,
        overdue_values=overdue_values,
        smoothing=smoothing,
        min_bin_pct=min_bin_pct,
        seed=seed,
    )


def _numeric_result(
    series: pd.Series,
    target: np.ndarray,
    *,
    feature: str,
    method: str,
    bin_count: int,
    sentinels: tuple[object, ...],
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    smoothing: float,
    min_bin_pct: float,
    seed: int,
) -> dict[str, Any]:
    if _contains_unsafe_integer(series):
        return _unavailable(method, "unsafe_numeric_precision")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    invalid = series.notna().to_numpy() & ~np.isfinite(values)
    if np.any(invalid):
        return _unavailable(method, "non_numeric_values", count=int(np.sum(invalid)))
    try:
        sentinel_numbers = tuple(float(value) for value in sentinels)
    except (TypeError, ValueError) as exc:
        raise FeatureError(
            f"sentinel values for numeric feature {feature} must be numeric"
        ) from exc
    if any(not math.isfinite(value) for value in sentinel_numbers):
        raise FeatureError(
            f"sentinel values for numeric feature {feature} must be finite"
        )
    if len(set(sentinel_numbers)) != len(sentinel_numbers):
        raise FeatureError(
            f"sentinel values for numeric feature {feature} must be unique"
        )
    sentinel_mask = np.isin(values, sentinel_numbers) & np.isfinite(values)
    regular_mask = np.isfinite(values) & ~sentinel_mask
    regular_values = values[regular_mask]
    if regular_values.size == 0:
        return _unavailable(method, "no_regular_values")
    unique_count = int(np.unique(regular_values).size)
    supervised = method in {"chimerge", "tree"}
    if supervised and (unique_count < 2 or np.unique(target[regular_mask]).size < 2):
        return _unavailable(
            method,
            "insufficient_supervised_variation",
            unique_values=unique_count,
            target_classes=int(np.unique(target[regular_mask]).size),
        )
    try:
        if method == "equal_frequency":
            edges = equal_frequency_edges(
                regular_values, bin_count, min_bin_pct=min_bin_pct
            )
        elif method == "equal_width":
            edges = equal_width_edges(regular_values, bin_count)
        elif method == "chimerge":
            edges = chimerge_edges(
                regular_values,
                target[regular_mask],
                max_bins=bin_count,
                min_bin_pct=min_bin_pct,
            )
        elif method == "tree":
            edges = tree_edges(
                regular_values,
                target[regular_mask],
                max_bins=bin_count,
                min_samples_leaf=(min_bin_pct if min_bin_pct > 0 else 1),
                seed=seed,
            )
        else:  # guarded by request validation
            raise AssertionError(method)
    except (FeatureError, ValueError) as exc:
        return _unavailable(method, "method_failed", detail=str(exc)[:240])

    assigned = np.full(len(values), -999, dtype=int)
    assigned[regular_mask] = assign_bins(regular_values, edges)
    groups: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        groups.append(
            {
                "id": f"regular:{index}",
                "kind": "numeric_interval",
                "mask": assigned == index,
                "condition": _numeric_condition(
                    feature, edges, index, sentinel_numbers
                ),
                "lower": _finite_bound(edges[index]),
                "upper": _finite_bound(edges[index + 1]),
                "include_lower": True,
                "include_upper": False,
            }
        )
    for index, sentinel in enumerate(sentinel_numbers):
        groups.append(
            {
                "id": f"sentinel:{index}",
                "kind": "sentinel",
                "mask": np.isfinite(values) & (values == sentinel),
                "condition": _compare(feature, "==", _json_scalar(sentinel)),
                "value": _json_scalar(sentinel),
            }
        )
    missing_mask = ~np.isfinite(values) & ~invalid
    if np.any(missing_mask):
        groups.append(
            {
                "id": "missing",
                "kind": "missing",
                "mask": missing_mask,
                "condition": {"op": "is_null", "field": feature},
            }
        )
    evidence = []
    diagnostic = degraded_bin_diagnostic(edges, bin_count, feature=feature)
    if diagnostic is not None:
        evidence.append(diagnostic)
    bin_shares = [
        float(np.mean(assigned[regular_mask] == index))
        for index in range(len(edges) - 1)
    ]
    if min_bin_pct > 0 and any(0 < share < min_bin_pct for share in bin_shares):
        evidence.append(
            {
                "kind": "min_bin_pct_not_achieved",
                "severity": "red_flag",
                "requested_min_bin_pct": min_bin_pct,
                "smallest_nonempty_bin_pct": min(
                    share for share in bin_shares if share > 0
                ),
                "message": "该方法无法在当前重复值/边界约束下满足最小箱占比。",
            }
        )
    return _available_result(
        method,
        groups,
        target,
        risk_direction=_numeric_risk_direction(groups, target),
        loan_values=loan_values,
        overdue_values=overdue_values,
        smoothing=smoothing,
        evidence=evidence,
    )


def _categorical_result(
    series: pd.Series,
    target: np.ndarray,
    *,
    feature: str,
    method: str,
    sentinels: tuple[object, ...],
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    max_categories: int,
    min_bin_pct: float,
    smoothing: float,
) -> dict[str, Any]:
    if _contains_unsafe_integer(series):
        return _unavailable(method, "unsafe_numeric_precision")
    values = series.to_numpy(dtype=object)
    missing = pd.isna(values)
    try:
        normalized_values = [
            None if is_na else _json_scalar(value)
            for value, is_na in zip(values, missing)
        ]
    except FeatureError:
        return _unavailable(method, "non_json_category")
    sentinel_keys = {_category_key(value) for value in sentinels}
    category_by_key = {
        _category_key(value): value
        for value in normalized_values
        if value is not None and _category_key(value) not in sentinel_keys
    }
    categories = [category_by_key[key] for key in sorted(category_by_key)]
    if len(categories) + len(sentinels) > max_categories:
        return _unavailable(
            method,
            "category_budget_exceeded",
            category_count=len(categories) + len(sentinels),
            max_categories=max_categories,
        )
    groups: list[dict[str, Any]] = []
    for index, category in enumerate(categories):
        category_mask = np.array(
            [
                value is not None and _category_key(value) == _category_key(category)
                for value in normalized_values
            ],
            dtype=bool,
        )
        groups.append(
            {
                "id": f"category:{index}",
                "kind": "category",
                "mask": category_mask,
                "condition": _compare(
                    feature,
                    "==",
                    category,
                    coercion="strict",
                ),
                "value": category,
            }
        )
    for index, sentinel in enumerate(sorted(sentinels, key=_category_key)):
        sentinel_mask = np.array(
            [
                value is not None and _category_key(value) == _category_key(sentinel)
                for value in normalized_values
            ],
            dtype=bool,
        )
        groups.append(
            {
                "id": f"sentinel:{index}",
                "kind": "sentinel",
                "mask": sentinel_mask,
                "condition": _compare(
                    feature,
                    "==",
                    sentinel,
                    coercion="strict",
                ),
                "value": sentinel,
            }
        )
    if np.any(missing):
        groups.append(
            {
                "id": "missing",
                "kind": "missing",
                "mask": missing,
                "condition": {"op": "is_null", "field": feature},
            }
        )
    if not groups:
        return _unavailable(method, "no_categories")
    result = _available_result(
        method,
        groups,
        target,
        risk_direction="unordered",
        loan_values=loan_values,
        overdue_values=overdue_values,
        smoothing=smoothing,
        evidence=[],
    )
    small_shares = [
        row["share"]
        for row in result["bins"]
        if row["count"] and row["share"] < min_bin_pct
    ]
    if min_bin_pct > 0 and small_shares:
        result["evidence"].append(
            {
                "kind": "min_bin_pct_not_achieved",
                "severity": "red_flag",
                "requested_min_bin_pct": min_bin_pct,
                "smallest_nonempty_bin_pct": min(small_shares),
                "message": "类别等值箱不自动合并，存在低于最小箱占比的类别。",
            }
        )
    return result


def _available_result(
    method: str,
    groups: list[dict[str, Any]],
    target: np.ndarray,
    *,
    risk_direction: str,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    smoothing: float,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    total_bad = int(np.sum(target == 1))
    total_good = int(np.sum(target == 0))
    group_count = len(groups)
    base_bad_rate = total_bad / len(target)
    total_iv = 0.0
    bin_rows = []
    score = np.zeros(len(target), dtype=float)
    for position, group in enumerate(groups):
        mask = np.asarray(group.pop("mask"), dtype=bool)
        count = int(np.sum(mask))
        bad = int(np.sum(target[mask] == 1))
        good = count - bad
        woe, contribution = _smoothed_woe_iv(
            bad,
            good,
            total_bad,
            total_good,
            group_count,
            smoothing=smoothing,
        )
        total_iv += contribution
        bad_rate = bad / count if count else None
        if bad_rate is not None:
            score[mask] = bad_rate
        bin_rows.append(
            {
                "index": position,
                **group,
                "count": count,
                "share": count / len(target),
                "good": good,
                "bad": bad,
                "bad_rate": bad_rate,
                "woe": woe,
                "iv_contribution": contribution,
                "lift": (bad_rate / base_bad_rate) if bad_rate is not None else None,
                "cumulative_ks": 0.0,
                "amount_metrics": _amount_metrics(mask, loan_values, overdue_values),
            }
        )
    _assign_cumulative_ks(bin_rows, total_bad=total_bad, total_good=total_good)
    return {
        "method": method,
        "requested_method": method,
        "actual_method": method,
        "status": "available",
        "evidence": evidence,
        "metrics": {
            "iv": total_iv,
            "ks": feature_ks(score, target),
            "auc": feature_auc(score, target),
            "risk_direction": risk_direction,
            "missing_rate": next(
                (row["share"] for row in bin_rows if row["kind"] == "missing"), 0.0
            ),
            "amount_metrics": _amount_metrics(
                np.ones(len(target), dtype=bool), loan_values, overdue_values
            ),
        },
        "bins": bin_rows,
    }


def _assign_cumulative_ks(
    bin_rows: list[dict[str, Any]],
    *,
    total_bad: int,
    total_good: int,
) -> None:
    """Attach KS values in the same ascending risk-score order as feature_ks."""

    by_score: dict[float, list[dict[str, Any]]] = {}
    for row in bin_rows:
        bad_rate = row["bad_rate"]
        if row["count"] and bad_rate is not None:
            by_score.setdefault(float(bad_rate), []).append(row)
    cumulative_bad = 0
    cumulative_good = 0
    for score in sorted(by_score):
        tied = by_score[score]
        cumulative_bad += sum(int(row["bad"]) for row in tied)
        cumulative_good += sum(int(row["good"]) for row in tied)
        ks = abs(cumulative_bad / total_bad - cumulative_good / total_good)
        for row in tied:
            row["cumulative_ks"] = ks


def _amount_metrics(
    mask: np.ndarray,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
) -> dict[str, Any]:
    loan = _amount_measure(mask, loan_values, "loan_amount")
    overdue = _amount_measure(mask, overdue_values, "overdue_amount")
    if loan_values is None or overdue_values is None:
        rate = {"status": "unavailable", "reason": "amount_column_not_configured"}
    else:
        paired = mask & np.isfinite(loan_values) & np.isfinite(overdue_values)
        if not np.any(mask):
            rate = {"status": "not_applicable", "reason": "empty_bin"}
        elif not np.any(paired):
            rate = {"status": "unavailable", "reason": "no_paired_amounts"}
        else:
            denominator = float(np.sum(loan_values[paired]))
            if denominator == 0:
                rate = {"status": "not_applicable", "reason": "zero_loan_amount"}
            else:
                rate = {
                    "status": "available",
                    "value": float(np.sum(overdue_values[paired]) / denominator),
                    "paired_count": int(np.sum(paired)),
                }
    return {"loan_amount": loan, "overdue_amount": overdue, "overdue_rate": rate}


def _amount_measure(
    mask: np.ndarray, values: np.ndarray | None, name: str
) -> dict[str, Any]:
    if values is None:
        return {"status": "unavailable", "reason": f"{name}_not_configured"}
    covered = mask & np.isfinite(values)
    selected_count = int(np.sum(mask))
    if selected_count == 0:
        return {
            "status": "available",
            "sum": 0.0,
            "covered_count": 0,
            "coverage_rate": 1.0,
        }
    if not np.any(covered):
        return {
            "status": "unavailable",
            "reason": "no_covered_rows",
            "coverage_rate": 0.0,
        }
    return {
        "status": "available",
        "sum": float(np.sum(values[covered])),
        "covered_count": int(np.sum(covered)),
        "coverage_rate": float(np.sum(covered) / selected_count)
        if selected_count
        else 0.0,
    }


def _numeric_risk_direction(groups: list[dict[str, Any]], target: np.ndarray) -> str:
    rates = []
    for group in groups:
        if group["kind"] != "numeric_interval":
            continue
        mask = group["mask"]
        if np.any(mask):
            rates.append(float(np.mean(target[mask] == 1)))
    if len(rates) <= 1 or all(value == rates[0] for value in rates):
        return "flat"
    increasing = all(left <= right for left, right in zip(rates, rates[1:]))
    decreasing = all(left >= right for left, right in zip(rates, rates[1:]))
    if increasing:
        return "increasing"
    if decreasing:
        return "decreasing"
    return "non_monotonic"


def _numeric_condition(
    feature: str, edges: np.ndarray, index: int, sentinels: tuple[float, ...]
) -> dict[str, Any]:
    args = []
    lower = float(edges[index])
    upper = float(edges[index + 1])
    if math.isfinite(lower):
        args.append(_compare(feature, ">=", lower))
    if math.isfinite(upper):
        args.append(_compare(feature, "<", upper))
    if sentinels:
        args.append(
            {
                "op": "compare",
                "field": feature,
                "operator": "not_in",
                "value": [_json_scalar(value) for value in sentinels],
                "missing": "no_match",
            }
        )
    if len(args) == 1:
        return args[0]
    if not args:
        return {"op": "is_not_null", "field": feature}
    return {"op": "and", "args": args}


def _compare(
    feature: str,
    operator: str,
    value: object,
    *,
    coercion: str = "auto",
) -> dict[str, Any]:
    result = {
        "op": "compare",
        "field": feature,
        "operator": operator,
        "value": value,
        "missing": "no_match",
    }
    if coercion == "strict":
        result["coercion"] = "strict"
    return result


def _unavailable(method: str, kind: str, **detail: Any) -> dict[str, Any]:
    return {
        "method": method,
        "requested_method": method,
        "actual_method": None,
        "status": "unavailable",
        "evidence": {"kind": kind, **detail},
        "metrics": None,
        "bins": [],
    }


def _validate_request(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    methods: Sequence[str] | None,
    feature_types: Mapping[str, str] | None,
    bin_count: int,
    max_rows: int,
    max_features: int,
    max_bins: int,
    max_categories: int,
    min_bin_pct: float,
    smoothing: float,
    loan_amount: str | None,
    overdue_amount: str | None,
    seed: int,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise FeatureError("frame must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise FeatureError("frame columns must be unique")
    if isinstance(features, str) or not isinstance(features, Sequence):
        raise FeatureError("features must be a sequence of column names")
    if not isinstance(target, str) or not target:
        raise FeatureError("target must be a non-empty column name")
    if feature_types is not None and not isinstance(feature_types, Mapping):
        raise FeatureError("feature_types must be a mapping")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2_147_483_647
    ):
        raise FeatureError("seed must be an integer between 0 and 2147483647")
    for name, label in (
        (loan_amount, "loan_amount"),
        (overdue_amount, "overdue_amount"),
    ):
        if name is not None and (not isinstance(name, str) or not name):
            raise FeatureError(f"{label} must be a non-empty column name or null")
    if (
        not isinstance(max_features, int)
        or isinstance(max_features, bool)
        or not 1 <= max_features <= 50
    ):
        raise FeatureError("max_features must be between 1 and 50")
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise FeatureError("max_rows must be a positive integer")
    if (
        not isinstance(max_bins, int)
        or isinstance(max_bins, bool)
        or not 3 <= max_bins <= 20
    ):
        raise FeatureError("max_bins must be between 3 and 20")
    if not 1 <= len(features) <= max_features:
        raise FeatureError(
            "features must contain 1 to max_features unique columns (maximum 50)"
        )
    if any(not isinstance(item, str) or not item for item in features):
        raise FeatureError("features must contain unique non-empty column names")
    if len(set(features)) != len(features):
        raise FeatureError("features must contain unique non-empty column names")
    if not 1 <= len(frame) <= max_rows:
        raise FeatureError("frame row count exceeds the configured resource budget")
    if (
        not isinstance(max_categories, int)
        or isinstance(max_categories, bool)
        or max_categories < 1
    ):
        raise FeatureError("max_categories must be a positive integer")
    if (
        not isinstance(bin_count, int)
        or isinstance(bin_count, bool)
        or not 3 <= bin_count <= max_bins <= 20
    ):
        raise FeatureError("bin_count must be between 3 and max_bins (maximum 20)")
    if (
        not isinstance(smoothing, int | float)
        or isinstance(smoothing, bool)
        or not math.isfinite(float(smoothing))
        or smoothing <= 0
    ):
        raise FeatureError("smoothing must be a positive finite number")
    if (
        not isinstance(min_bin_pct, int | float)
        or isinstance(min_bin_pct, bool)
        or not math.isfinite(float(min_bin_pct))
        or not 0 <= min_bin_pct <= 0.5
    ):
        raise FeatureError("min_bin_pct must be between 0 and 0.5")
    required = [
        target,
        *features,
        *[name for name in (loan_amount, overdue_amount) if name],
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise FeatureError(f"unknown columns: {', '.join(missing)}")
    if target in features:
        raise FeatureError("target cannot also be a feature")
    _normalize_methods(methods)
    for feature, kind in (feature_types or {}).items():
        if feature not in features or kind not in {"numeric", "categorical"}:
            raise FeatureError(
                "feature_types must map requested features to numeric or categorical"
            )


def _normalize_methods(methods: Sequence[str] | None) -> tuple[str, ...] | None:
    if methods is None:
        return None
    if isinstance(methods, str) or not methods:
        raise FeatureError("methods must be a non-empty sequence")
    normalized = tuple(methods)
    if any(not isinstance(method, str) for method in normalized):
        raise FeatureError(
            f"methods must be unique and selected from: {', '.join(_ALL_METHODS)}"
        )
    if len(set(normalized)) != len(normalized) or any(
        method not in _ALL_METHODS for method in normalized
    ):
        raise FeatureError(
            f"methods must be unique and selected from: {', '.join(_ALL_METHODS)}"
        )
    return tuple(sorted(normalized, key=_METHOD_ORDER.__getitem__))


def _binary_target(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, [0, 1])):
        raise FeatureError(f"target {name} must contain only binary 0/1 values")
    if np.unique(values).size != 2:
        raise FeatureError(f"target {name} must contain both good and bad classes")
    return values.astype(int)


def _amount_values(frame: pd.DataFrame, column: str | None) -> np.ndarray | None:
    if column is None:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    invalid = frame[column].notna().to_numpy() & ~np.isfinite(values)
    if np.any(invalid) or np.any(values[np.isfinite(values)] < 0):
        raise FeatureError(
            f"amount column {column} must contain non-negative finite numbers or null"
        )
    return values


def _infer_feature_type(series: pd.Series) -> str:
    numeric = pd.api.types.is_numeric_dtype(
        series.dtype
    ) and not pd.api.types.is_bool_dtype(series.dtype)
    return "numeric" if numeric else "categorical"


def _contains_unsafe_integer(series: pd.Series) -> bool:
    for value in series.array:
        if _is_missing_scalar(value):
            continue
        if (
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and abs(int(value)) > _MAX_SAFE_JSON_INTEGER
        ):
            return True
    return False


def _is_missing_scalar(value: object) -> bool:
    try:
        marker = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(marker) if isinstance(marker, (bool, np.bool_)) else False


def _normalize_sentinels(
    values: Sequence[object], feature: str, *, max_values: int
) -> tuple[object, ...]:
    if isinstance(values, str | bytes):
        raise FeatureError(f"sentinel values for {feature} must be a sequence")
    if not isinstance(values, Sequence):
        raise FeatureError(f"sentinel values for {feature} must be a sequence")
    if len(values) > max_values:
        raise FeatureError(
            f"sentinel values for {feature} exceed the configured bin/category budget"
        )
    normalized = []
    for value in values:
        try:
            is_null = value is None or bool(pd.isna(value))
        except (TypeError, ValueError):
            is_null = False
        if is_null:
            raise FeatureError(f"sentinel values for {feature} cannot be null")
        normalized.append(_json_scalar(value))
    if len({_category_key(value) for value in normalized}) != len(normalized):
        raise FeatureError(f"sentinel values for {feature} must be unique")
    return tuple(sorted(normalized, key=_category_key))


def _finite_bound(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _category_key(value: object) -> tuple[str, str]:
    if isinstance(value, (bool, np.bool_)):
        return ("boolean", "true" if bool(value) else "false")
    if isinstance(value, (int, np.integer)):
        return ("number", str(int(value)))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == 0 or (
            number.is_integer() and abs(number) <= _MAX_SAFE_JSON_INTEGER
        ):
            return ("number", str(int(number)))
        return ("number", repr(number))
    return (type(value).__name__, repr(value))


def _json_scalar(value: object) -> str | int | float | bool:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        result = int(value)
        if abs(result) > _MAX_SAFE_JSON_INTEGER:
            raise FeatureError("integer value exceeds exact JSON numeric precision")
        return result
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise FeatureError("values must be finite JSON scalars")
        return result
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    raise FeatureError("values must be finite JSON scalars")


def _finite_float(value: float) -> float:
    if not math.isfinite(value):
        raise FeatureError("computed metric is not finite")
    return float(value)


def _assert_finite_json(value: Any, *, path: str = "result") -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FeatureError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_finite_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FeatureError(f"{path} contains a non-string key")
            _assert_finite_json(item, path=f"{path}.{key}")
        return
    raise FeatureError(f"{path} contains a non-JSON value")


__all__ = ["SCHEMA_VERSION", "analyze_univariate"]
