from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd

from marvis.feature.errors import FeatureError
from marvis.feature.metrics import feature_metrics


ALLOWED_CROSS_OPS = {"add", "sub", "mul", "div", "ratio"}
ALLOWED_AGGS = {"mean", "max", "min", "std", "sum", "count"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
CROSS_SYS = (
    "你基于特征的业务含义推荐值得交叉的特征对和运算，给出理由。"
    "你不计算任何 IV/KS/指标，那些由平台算。"
    "只输出特征对、运算和理由的 JSON。"
)


@dataclass(frozen=True)
class CrossRecommendation:
    col_a: str
    col_b: str
    ops: tuple[str, ...]
    rationale: str
    confidence: str


def cross_arithmetic(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    ops: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    _assert_columns(df, [col_a, col_b])
    if not ops:
        raise FeatureError("cross ops must not be empty")
    invalid = [op for op in ops if op not in ALLOWED_CROSS_OPS]
    if invalid:
        raise FeatureError(f"unsupported cross ops: {', '.join(invalid)}")

    left = pd.to_numeric(df[col_a], errors="raise")
    right = pd.to_numeric(df[col_b], errors="raise")
    out = {}
    denominator = right.replace(0, np.nan)
    if "add" in ops:
        out[f"{col_a}_add_{col_b}"] = left + right
    if "sub" in ops:
        out[f"{col_a}_sub_{col_b}"] = left - right
    if "mul" in ops:
        out[f"{col_a}_mul_{col_b}"] = left * right
    if "div" in ops:
        out[f"{col_a}_div_{col_b}"] = left / denominator
    if "ratio" in ops:
        out[f"{col_a}_ratio_{col_b}"] = left / denominator

    _assert_no_conflicts(df, out)
    return df.assign(**out), list(out.keys())


def aggregate_feature(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    aggs: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    _assert_columns(df, [group_col, value_col])
    if not aggs:
        raise FeatureError("aggregate functions must not be empty")
    invalid = [agg for agg in aggs if agg not in ALLOWED_AGGS]
    if invalid:
        raise FeatureError(f"unsupported aggregate functions: {', '.join(invalid)}")
    grouped = (
        df.groupby(group_col, dropna=False)[value_col]
        .agg(aggs)
        .rename(columns={agg: f"{value_col}_by_{group_col}_{agg}" for agg in aggs})
        .reset_index()
    )
    new_cols = [f"{value_col}_by_{group_col}_{agg}" for agg in aggs]
    _assert_no_conflicts(df, {col: None for col in new_cols})
    merged = df.merge(grouped, on=group_col, how="left", sort=False)
    if len(merged) != len(df):
        raise FeatureError("aggregate join expanded row count")
    return merged, new_cols


def derive_batch(df: pd.DataFrame, recipe: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    new_cols = []
    for item in recipe:
        kind = item.get("kind")
        if kind == "cross":
            out, cols = cross_arithmetic(out, str(item["a"]), str(item["b"]), list(item["ops"]))
        elif kind == "agg":
            out, cols = aggregate_feature(
                out,
                str(item["group"]),
                str(item["value"]),
                list(item["aggs"]),
            )
        elif kind == "ratio":
            out, cols = cross_arithmetic(out, str(item["num"]), str(item["den"]), ["ratio"])
        else:
            raise FeatureError(f"unsupported derive recipe kind: {kind}")
        repeated = set(new_cols).intersection(cols)
        if repeated:
            raise FeatureError(f"duplicate derived columns: {', '.join(sorted(repeated))}")
        new_cols.extend(cols)
    return out, new_cols


def recommend_feature_crosses(
    feature_dictionary: dict,
    existing_metrics: dict,
    *,
    llm_factory,
    max_candidates: int = 30,
) -> list[CrossRecommendation]:
    try:
        raw = llm_factory().complete(
            system_prompt=CROSS_SYS,
            user_prompt=build_cross_prompt(feature_dictionary, existing_metrics, max_candidates),
            response_format={"type": "json_object"},
            stream=False,
        )
        recs = _parse_recommendations(str(raw))
    except Exception:
        return []
    valid_features = set(feature_dictionary)
    valid = [
        rec
        for rec in recs
        if rec.col_a in valid_features and rec.col_b in valid_features and rec.col_a != rec.col_b
    ]
    return valid[:max_candidates]


def evaluate_crosses(
    df: pd.DataFrame,
    target: np.ndarray,
    recommendations: list[CrossRecommendation],
    *,
    selected_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    rec_by_pair = {(rec.col_a, rec.col_b): rec for rec in recommendations}
    pairs = selected_pairs or list(rec_by_pair)
    results = []
    for pair in pairs:
        if pair not in rec_by_pair:
            raise FeatureError(f"selected pair not found in recommendations: {pair[0]}, {pair[1]}")
        rec = rec_by_pair[pair]
        out, cols = cross_arithmetic(out, rec.col_a, rec.col_b, list(rec.ops))
        for col in cols:
            metrics = feature_metrics(out[col].to_numpy(dtype=float), target, feature=col)
            results.append({
                "new_col": col,
                "iv": metrics.iv,
                "ks": metrics.ks,
                "from": (rec.col_a, rec.col_b),
                "op": _op_from_col(col, rec.col_a, rec.col_b),
            })
    return out, results


def build_cross_prompt(feature_dictionary: dict, existing_metrics: dict, max_candidates: int) -> str:
    return json.dumps(
        {
            "instruction": (
                "Recommend feature crosses only. Do not calculate or output new IV, KS, "
                "AUC, PSI, lift, or any derived metric."
            ),
            "max_candidates": int(max_candidates),
            "feature_dictionary": feature_dictionary,
            "existing_metrics": existing_metrics,
            "output_schema": {
                "recommendations": [
                    {
                        "col_a": "existing feature name",
                        "col_b": "existing feature name",
                        "ops": ["ratio"],
                        "rationale": "business reason only",
                        "confidence": "high|medium|low",
                    }
                ]
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _parse_recommendations(raw: str) -> list[CrossRecommendation]:
    data = json.loads(raw)
    items = data if isinstance(data, list) else data.get("recommendations") or data.get("candidates") or []
    if not isinstance(items, list):
        return []
    recs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ops = _parse_ops(item.get("ops"))
        if not ops:
            continue
        confidence = str(item.get("confidence") or "low").lower()
        recs.append(
            CrossRecommendation(
                col_a=str(item.get("col_a") or item.get("a") or ""),
                col_b=str(item.get("col_b") or item.get("b") or ""),
                ops=tuple(ops),
                rationale=str(item.get("rationale") or item.get("reason") or ""),
                confidence=confidence if confidence in CONFIDENCE_LEVELS else "low",
            )
        )
    return recs


def _parse_ops(raw: Any) -> list[str]:
    values = [raw] if isinstance(raw, str) else list(raw or [])
    ops = []
    for value in values:
        op = str(value).lower()
        if op in ALLOWED_CROSS_OPS and op not in ops:
            ops.append(op)
    return ops


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureError(f"missing columns: {', '.join(missing)}")


def _assert_no_conflicts(df: pd.DataFrame, new_values: dict) -> None:
    conflicts = [col for col in new_values if col in df.columns]
    if conflicts:
        raise FeatureError(f"derived columns already exist: {', '.join(conflicts)}")


def _op_from_col(col: str, col_a: str, col_b: str) -> str:
    prefix = f"{col_a}_"
    suffix = f"_{col_b}"
    if col.startswith(prefix) and col.endswith(suffix):
        return col[len(prefix):-len(suffix)]
    return ""


__all__ = [
    "CROSS_SYS",
    "CrossRecommendation",
    "aggregate_feature",
    "build_cross_prompt",
    "cross_arithmetic",
    "derive_batch",
    "evaluate_crosses",
    "recommend_feature_crosses",
]
