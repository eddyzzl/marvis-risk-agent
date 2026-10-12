"""Frozen candidate bucket distributions on governed independent partitions.

The tool boundary authenticates source artifacts, model dependencies and physical
membership. This kernel receives row-aligned assignments; it cannot select,
adopt, retrain or promote any candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np
import pandas as pd

from marvis.packs.strategy.errors import StrategyError
from marvis.feature.metrics import feature_auc, feature_ks
from marvis.validation.time_periods import month_key_series

SCHEMA = "strategy.asset-independent-stability.v1"
FREEZE_SCHEMA = "strategy.asset-stability-freeze.v1"
MAX_ROWS = 1_000_000
MAX_BUCKETS = 4096
MAX_MONTHS = 240
MAX_METRIC_ROWS = 50_000
EPSILON = 1e-6


def canonical(value: Mapping) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise StrategyError("asset stability JSON is invalid") from exc


def seal(body: Mapping, *, prefix: str) -> dict:
    digest = hashlib.sha256(canonical(body)).hexdigest()
    return {**body, "content_hash": digest, "stability_id": f"{prefix}-{digest[:24]}"}


def validate_sealed(value: object, *, schema: str, prefix: str) -> dict:
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise StrategyError("asset stability schema is invalid")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"content_hash", "stability_id"}
    }
    expected = seal(body, prefix=prefix)
    if dict(value) != expected:
        raise StrategyError("asset stability identity or content hash changed")
    return expected


def measure_asset_stability(
    *,
    freeze: Mapping,
    frozen_ref: Mapping,
    buckets: np.ndarray,
    bucket_ids: list[str],
    labels: np.ndarray,
    months: pd.Series,
    masks: Mapping[str, np.ndarray],
    scores: np.ndarray | None = None,
) -> dict:
    """Count fixed buckets without fitting anything on validation/OOT rows."""
    freeze = validate_sealed(freeze, schema=FREEZE_SCHEMA, prefix="asset-freeze")
    buckets = np.asarray(buckets)
    labels = np.asarray(labels, dtype=np.float64)
    if (
        buckets.ndim != 1
        or len(buckets) != len(labels)
        or not 0 < len(buckets) <= MAX_ROWS
    ):
        raise StrategyError("asset stability row universe is invalid or exceeds budget")
    if (
        not bucket_ids
        or len(bucket_ids) > MAX_BUCKETS
        or len(set(bucket_ids)) != len(bucket_ids)
    ):
        raise StrategyError("asset stability bucket definitions are invalid")
    if not np.isin(buckets, bucket_ids).all():
        raise StrategyError("asset stability assignments have unknown buckets")
    if np.isinf(labels).any() or not np.isin(labels[np.isfinite(labels)], [0, 1]).all():
        raise StrategyError("asset stability target must use normalized bad=1 polarity")
    keys = month_key_series(months, column_name="asset observation time")
    if len(keys) != len(buckets) or keys.isna().any():
        raise StrategyError(
            "asset stability requires known observation time for every selected row"
        )
    if len(set(keys.tolist())) > MAX_MONTHS:
        raise StrategyError("asset stability observation periods exceed budget")
    selected = ["development", *freeze["partitions"]]
    normalized = {}
    for partition in selected:
        raw = np.asarray(masks.get(partition))
        if raw.dtype != np.dtype(bool) or raw.shape != buckets.shape or not raw.any():
            raise StrategyError("asset stability membership is missing or empty")
        normalized[partition] = raw
    total = np.zeros(len(buckets), dtype=np.int8)
    for mask in normalized.values():
        total += mask
    if np.any(total > 1):
        raise StrategyError("asset stability partitions overlap")
    metric_rows = len(bucket_ids) * (
        len(selected)
        + sum(len(set(keys[normalized[p]].tolist())) for p in freeze["partitions"])
    )
    if metric_rows > MAX_METRIC_ROWS:
        raise StrategyError(
            "asset stability exact monthly bucket evidence exceeds metric row budget"
        )
    codes = pd.Categorical(buckets, categories=bucket_ids).codes
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if (
            scores.shape != buckets.shape
            or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
        ):
            raise StrategyError(
                "asset stability authenticated bad-probability score universe is invalid"
            )

    def counts(mask):
        n = int(mask.sum())
        count_values = np.bincount(codes[mask], minlength=len(bucket_ids))
        labeled = mask & np.isfinite(labels)
        labeled_values = np.bincount(codes[labeled], minlength=len(bucket_ids))
        bad_values = np.bincount(
            codes[labeled], weights=labels[labeled], minlength=len(bucket_ids)
        )
        rows = []
        for ordinal, bucket in enumerate(bucket_ids):
            count = int(count_values[ordinal])
            labeled_count = int(labeled_values[ordinal])
            bad = int(bad_values[ordinal])
            rows.append(
                {
                    "bucket_id": bucket,
                    "count": count,
                    "share": count / n,
                    "labeled_count": labeled_count,
                    "label_coverage": None if count == 0 else labeled_count / count,
                    "bad_count": bad,
                    "bad_rate": None if labeled_count == 0 else bad / labeled_count,
                }
            )
        performance = {
            "status": "not_applicable",
            "reason": "source_is_bucket_rule_without_probability_score",
            "auc": None,
            "ks": None,
        }
        if scores is not None:
            if set(labels[labeled].tolist()) == {0.0, 1.0}:
                performance = {
                    "status": "available",
                    "reason": None,
                    "auc": float(feature_auc(scores[labeled], labels[labeled])),
                    "ks": float(feature_ks(scores[labeled], labels[labeled])),
                }
            else:
                performance = {
                    "status": "unavailable",
                    "reason": "independent_partition_lacks_two_labeled_classes",
                    "auc": None,
                    "ks": None,
                }
        return {
            "sample_count": n,
            "labeled_count": sum(row["labeled_count"] for row in rows),
            "bad_count": sum(row["bad_count"] for row in rows),
            "buckets": rows,
            "performance": performance,
        }

    baseline = counts(normalized["development"])
    expected = np.asarray([row["share"] for row in baseline["buckets"]])

    def independent(mask):
        result = counts(mask)
        actual = np.asarray([row["share"] for row in result["buckets"]])
        # Fixed epsilon smoothing, renormalized over the same development buckets.
        p = np.maximum(expected, EPSILON)
        p /= p.sum()
        q = np.maximum(actual, EPSILON)
        q /= q.sum()
        contributions = (q - p) * np.log(q / p)
        for row, contribution in zip(result["buckets"], contributions, strict=True):
            row["psi_contribution"] = float(contribution)
        result["psi_vs_development"] = float(contributions.sum())
        result["small_sample"] = result["sample_count"] < 30
        return result

    partitions = []
    for partition in freeze["partitions"]:
        mask = normalized[partition]
        monthly = [
            {"month": str(month), **independent(mask & (keys.to_numpy() == month))}
            for month in sorted(set(keys[mask].tolist()))
        ]
        partitions.append(
            {"partition": partition, **independent(mask), "monthly": monthly}
        )
    body = {
        "schema_version": SCHEMA,
        "frozen_ref": dict(frozen_ref),
        "freeze": freeze,
        "baseline": baseline,
        "partitions": partitions,
        "metric_contract": {
            "baseline": "risk/development",
            "bucket_policy": "frozen_source_conditions",
            "psi_epsilon": EPSILON,
            "psi_smoothing": "clip_then_normalize",
            "bad_value": 1,
            "row_alignment": "authenticated_raw_row_ordinal",
        },
        "lifecycle": {
            "validation_status": "independent_evidence_measured",
            "candidate_promoted": False,
            "selection_performed": False,
            "deployment_performed": False,
        },
        "conservation": {
            "fixed_buckets": True,
            "partition_disjointness": True,
            "all_counts_conserved": all(
                sum(row["count"] for row in part["buckets"]) == part["sample_count"]
                for part in partitions
            ),
        },
    }
    return seal(body, prefix="asset-stability")


def validate_evidence(value: object) -> dict:
    evidence = validate_sealed(value, schema=SCHEMA, prefix="asset-stability")
    freeze = validate_sealed(
        evidence["freeze"], schema=FREEZE_SCHEMA, prefix="asset-freeze"
    )
    if [item["partition"] for item in evidence["partitions"]] != freeze["partitions"]:
        raise StrategyError("asset stability independent partitions changed")
    for group in [
        evidence["baseline"],
        *evidence["partitions"],
        *(month for part in evidence["partitions"] for month in part["monthly"]),
    ]:
        if sum(row["count"] for row in group["buckets"]) != group["sample_count"]:
            raise StrategyError("asset stability counts are not conserved")
        for row in group["buckets"]:
            if not 0 <= row["bad_count"] <= row["labeled_count"] <= row["count"]:
                raise StrategyError("asset stability label counts are invalid")
        if "psi_vs_development" in group and not math.isclose(
            sum(row["psi_contribution"] for row in group["buckets"]),
            group["psi_vs_development"],
            abs_tol=1e-12,
        ):
            raise StrategyError("asset stability PSI is not conserved")
    return evidence
