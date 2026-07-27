"""Deterministic monthly stability evidence for one strategy candidate.

The persistence/tool boundary authenticates the source candidate or exact Pool
revision and binds the development sample.  This module receives only those
already-trusted references, the bound development rows, and the deterministic
hit mask.  It computes aggregate hit/not-hit stability evidence and never
creates, adopts, promotes, or deploys a strategy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_execution import (
    StrategyRiskDevelopmentRef,
)
from marvis.validation.binning import compute_psi
from marvis.validation.time_periods import month_key_series


CANDIDATE_STABILITY_SCHEMA_VERSION = "strategy.candidate-stability.v1"
CANDIDATE_STABILITY_PRODUCER_VERSION = "marvis.strategy.candidate-stability/1"
CANDIDATE_STABILITY_MAX_ROWS = 1_000_000
CANDIDATE_STABILITY_MAX_MONTHS = 240
CANDIDATE_STABILITY_MIN_MONTH_ROWS = 30

_MAX_SAFE_JSON_INTEGER = 2**53 - 1
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_STABILITY_ID_RE = re.compile(r"^candidate-stability-[0-9a-f]{24}$")
_MONTH_RE = re.compile(r"^\d{6}$")
_BASIS_TO_SOURCE_KIND = {
    "asset_rule_hit": "univariate_asset",
    "pool_entry_incremental_first_match": "pool_entry",
}
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "stability_id",
        "basis",
        "identity",
        "source_ref",
        "sample_design_ref",
        "bindings",
        "lifecycle",
        "baseline",
        "summary",
        "monthly",
        "red_flags",
        "conservation",
        "content_hash",
    }
)
_BODY_FIELDS = _TOP_LEVEL_FIELDS - {"stability_id", "content_hash"}
_IDENTITY_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
    }
)
_ASSET_SOURCE_FIELDS = frozenset(
    {
        "source_kind",
        "artifact_id",
        "artifact_content_hash",
        "asset_id",
        "asset_hash",
        "rule_id",
    }
)
_POOL_SOURCE_FIELDS = frozenset(
    {
        "source_kind",
        "artifact_id",
        "artifact_content_hash",
        "pool_id",
        "revision",
        "revision_id",
        "snapshot_hash",
        "entry_id",
        "rule_id",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "month_col",
        "target_col",
        "target_bad_value",
        "minimum_month_rows",
    }
)
_LIFECYCLE = {
    "candidate_stage": "development",
    "observation_stage": "backtested",
    "validation_status": "unvalidated",
    "not_created_strategy": True,
    "not_adopted": True,
    "not_deployed": True,
}
_METRIC_FIELDS = frozenset(
    {
        "sample_count",
        "hit_count",
        "not_hit_count",
        "hit_share",
        "not_hit_share",
        "labeled_count",
        "label_coverage",
        "hit_labeled_count",
        "hit_bad_count",
        "hit_bad_rate",
        "psi_vs_development",
    }
)
_MONTHLY_FIELDS = _METRIC_FIELDS | {"month"}
_SUMMARY_FIELDS = frozenset(
    {
        "population_count",
        "month_count",
        "max_psi",
        "max_psi_month",
        "insufficient_month_count",
    }
)
_RED_FLAG_FIELDS = frozenset(
    {"kind", "month", "observed_rows", "minimum_rows"}
)
_CONSERVATION = {
    "monthly_count_rolls_to_development": True,
    "monthly_hits_roll_to_development": True,
    "monthly_labels_roll_to_development": True,
    "monthly_hit_labels_roll_to_development": True,
    "monthly_hit_bads_roll_to_development": True,
}


class CandidateStabilityError(StrategyError):
    """Candidate stability inputs or immutable evidence failed closed."""


def build_candidate_stability_artifact(
    *,
    frame: pd.DataFrame,
    month_col: str,
    target_col: str,
    hit_mask: object,
    basis: str,
    identity: Mapping[str, Any],
    source_ref: Mapping[str, Any],
    sample_design_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Build aggregate monthly stability evidence on the bound development sample.

    ``hit_mask`` is the direct rule hit for a univariate asset, or the exact
    first-match incremental hit for one Pool entry.  The complete development
    population is always the PSI baseline; no month is promoted to a baseline.
    """

    working = _development_frame(
        frame,
        month_col=month_col,
        target_col=target_col,
    )
    normalized_basis = _basis(basis)
    normalized_identity = _identity(identity)
    normalized_source = _source_ref(source_ref, basis=normalized_basis)
    normalized_sample_ref = _sample_design_ref(sample_design_ref)
    bindings = _bindings(
        {
            "month_col": month_col,
            "target_col": target_col,
            "target_bad_value": 1,
            "minimum_month_rows": CANDIDATE_STABILITY_MIN_MONTH_ROWS,
        }
    )
    hits = _hit_array(hit_mask, frame=working)
    periods = _month_keys(working, month_col=bindings["month_col"])
    targets = _target_array(working, target_col=bindings["target_col"])

    month_values = sorted(str(value) for value in periods.unique().tolist())
    if len(month_values) > CANDIDATE_STABILITY_MAX_MONTHS:
        raise CandidateStabilityError(
            "candidate stability exceeds the 240-month limit"
        )

    all_rows = np.ones(len(working), dtype=bool)
    baseline = _metric_slice(
        all_rows,
        hits=hits,
        targets=targets,
        baseline_distribution=None,
    )
    baseline_distribution = _distribution(baseline)
    monthly = [
        {
            "month": month,
            **_metric_slice(
                periods.eq(month).to_numpy(dtype=bool),
                hits=hits,
                targets=targets,
                baseline_distribution=baseline_distribution,
            ),
        }
        for month in month_values
    ]
    red_flags = _expected_red_flags(monthly)
    max_row = max(monthly, key=lambda row: (float(row["psi_vs_development"]), -int(row["month"])))
    body = {
        "schema_version": CANDIDATE_STABILITY_SCHEMA_VERSION,
        "producer_version": CANDIDATE_STABILITY_PRODUCER_VERSION,
        "basis": normalized_basis,
        "identity": normalized_identity,
        "source_ref": normalized_source,
        "sample_design_ref": normalized_sample_ref,
        "bindings": bindings,
        "lifecycle": dict(_LIFECYCLE),
        "baseline": baseline,
        "summary": {
            "population_count": len(working),
            "month_count": len(monthly),
            "max_psi": float(max_row["psi_vs_development"]),
            "max_psi_month": max_row["month"],
            "insufficient_month_count": len(red_flags),
        },
        "monthly": monthly,
        "red_flags": red_flags,
        "conservation": dict(_CONSERVATION),
    }
    stability_id = "candidate-stability-" + _sha256(_canonical_json(body))[:24]
    document = {**body, "stability_id": stability_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    return validate_candidate_stability_artifact(document)


def validate_candidate_stability_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact fields, hashes, derived metrics, ordering, and rollups."""

    if not isinstance(payload, Mapping):
        raise CandidateStabilityError(
            "candidate stability artifact must be an object"
        )
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "candidate stability artifact")
    normalized_json = _json_value(dict(payload), "candidate stability artifact")
    if not isinstance(normalized_json, dict):
        raise CandidateStabilityError(
            "candidate stability artifact must be an object"
        )

    stability_id = _text(normalized_json["stability_id"], "stability_id")
    if _STABILITY_ID_RE.fullmatch(stability_id) is None:
        raise CandidateStabilityError("stability_id has an invalid format")
    supplied_hash = _hash(normalized_json["content_hash"], "content_hash")
    body = _normalize_body(
        {
            key: normalized_json[key]
            for key in normalized_json
            if key not in {"stability_id", "content_hash"}
        }
    )
    expected_id = "candidate-stability-" + _sha256(_canonical_json(body))[:24]
    if not hmac.compare_digest(stability_id, expected_id):
        raise CandidateStabilityError(
            "stability_id does not match canonical stability evidence"
        )
    without_hash = {**body, "stability_id": stability_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise CandidateStabilityError(
            "content_hash does not match canonical stability evidence"
        )
    return {**without_hash, "content_hash": supplied_hash}


def canonical_candidate_stability_artifact_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole byte-stable JSON representation for one artifact."""

    return _canonical_json(validate_candidate_stability_artifact(payload))


def candidate_stability_artifact_content_hash(
    payload: Mapping[str, Any],
) -> str:
    """Return the verified embedded content hash."""

    return validate_candidate_stability_artifact(payload)["content_hash"]


def _normalize_body(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(value, _BODY_FIELDS, "candidate stability body")
    if value["schema_version"] != CANDIDATE_STABILITY_SCHEMA_VERSION:
        raise CandidateStabilityError("candidate stability schema_version is invalid")
    if value["producer_version"] != CANDIDATE_STABILITY_PRODUCER_VERSION:
        raise CandidateStabilityError(
            "candidate stability producer_version is invalid"
        )
    basis = _basis(value["basis"])
    identity = _identity(value["identity"])
    source = _source_ref(value["source_ref"], basis=basis)
    sample_ref = _sample_design_ref(value["sample_design_ref"])
    bindings = _bindings(value["bindings"])
    if value["lifecycle"] != _LIFECYCLE:
        raise CandidateStabilityError(
            "candidate stability lifecycle must remain development/backtested/"
            "unvalidated and non-mutating"
        )
    baseline = _metric_row(value["baseline"], name="baseline", baseline=True)
    monthly = _monthly_rows(value["monthly"], baseline=baseline)
    _require_monthly_rollup(monthly, baseline=baseline)
    red_flags = _red_flags(value["red_flags"])
    expected_flags = _expected_red_flags(monthly)
    if red_flags != expected_flags:
        raise CandidateStabilityError(
            "candidate stability red_flags do not match monthly row counts"
        )
    summary = _summary(
        value["summary"],
        baseline=baseline,
        monthly=monthly,
        red_flags=red_flags,
    )
    if value["conservation"] != _CONSERVATION:
        raise CandidateStabilityError(
            "candidate stability conservation checks must all pass"
        )
    return {
        "schema_version": CANDIDATE_STABILITY_SCHEMA_VERSION,
        "producer_version": CANDIDATE_STABILITY_PRODUCER_VERSION,
        "basis": basis,
        "identity": identity,
        "source_ref": source,
        "sample_design_ref": sample_ref,
        "bindings": bindings,
        "lifecycle": dict(_LIFECYCLE),
        "baseline": baseline,
        "summary": summary,
        "monthly": monthly,
        "red_flags": red_flags,
        "conservation": dict(_CONSERVATION),
    }


def _development_frame(
    value: object,
    *,
    month_col: object,
    target_col: object,
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise CandidateStabilityError("frame must be a pandas DataFrame")
    if value.empty:
        raise CandidateStabilityError(
            "candidate stability development sample must not be empty"
        )
    if len(value) > CANDIDATE_STABILITY_MAX_ROWS:
        raise CandidateStabilityError(
            "candidate stability exceeds the 1,000,000-row limit"
        )
    if not value.columns.is_unique:
        raise CandidateStabilityError("frame columns must be unique")
    month_name = _text(month_col, "month_col")
    target_name = _text(target_col, "target_col")
    if month_name == target_name:
        raise CandidateStabilityError("month_col and target_col must be different")
    missing = [
        name for name in (month_name, target_name) if name not in value.columns
    ]
    if missing:
        raise CandidateStabilityError(
            "frame is missing required columns: " + ", ".join(missing)
        )
    return value


def _month_keys(frame: pd.DataFrame, *, month_col: str) -> pd.Series:
    try:
        periods = month_key_series(frame[month_col], column_name=month_col)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandidateStabilityError(
            f"candidate stability month_col is invalid: {exc}"
        ) from exc
    if len(periods) != len(frame):
        raise CandidateStabilityError(
            "candidate stability month keys do not match the development sample"
        )
    return periods.reset_index(drop=True)


def _target_array(frame: pd.DataFrame, *, target_col: str) -> np.ndarray:
    source = frame[target_col].reset_index(drop=True)
    numeric = pd.to_numeric(source, errors="coerce")
    invalid = source.notna() & numeric.isna()
    if bool(invalid.any()):
        raise CandidateStabilityError(
            "target_col must contain only binary 0/1 values or null"
        )
    values = numeric.to_numpy(dtype=float)
    non_missing = source.notna().to_numpy(dtype=bool)
    if np.any(non_missing & ~np.isfinite(values)):
        raise CandidateStabilityError(
            "target_col must contain only binary 0/1 values or null"
        )
    labelled = np.isfinite(values)
    if np.any(~np.isin(values[labelled], [0.0, 1.0])):
        raise CandidateStabilityError(
            "target_col must contain only binary 0/1 values or null"
        )
    return values


def _hit_array(value: object, *, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(value, pd.Series):
        if not value.index.equals(frame.index):
            raise CandidateStabilityError(
                "hit_mask Series index must exactly match the frame index"
            )
        raw = value.array
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise CandidateStabilityError("hit_mask must be one-dimensional")
        raw = value
    elif isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        raw = value
    else:
        raise CandidateStabilityError("hit_mask must be a boolean sequence")
    if len(raw) != len(frame):
        raise CandidateStabilityError(
            "hit_mask length must match the development sample"
        )
    values = list(raw)
    if any(
        item is None
        or item is pd.NA
        or (isinstance(item, float | np.floating) and math.isnan(float(item)))
        for item in values
    ):
        raise CandidateStabilityError("hit_mask must not contain null values")
    if any(not isinstance(item, bool | np.bool_) for item in values):
        raise CandidateStabilityError("hit_mask must contain only booleans")
    return np.asarray(values, dtype=bool)


def _metric_slice(
    selected: np.ndarray,
    *,
    hits: np.ndarray,
    targets: np.ndarray,
    baseline_distribution: np.ndarray | None,
) -> dict[str, Any]:
    sample_count = int(selected.sum())
    selected_hits = selected & hits
    hit_count = int(selected_hits.sum())
    not_hit_count = sample_count - hit_count
    selected_labelled = selected & np.isfinite(targets)
    labeled_count = int(selected_labelled.sum())
    hit_labelled = selected_hits & np.isfinite(targets)
    hit_labeled_count = int(hit_labelled.sum())
    hit_bad_count = int(np.sum(targets[hit_labelled] == 1.0))
    distribution = np.asarray(
        [_ratio(not_hit_count, sample_count), _ratio(hit_count, sample_count)],
        dtype=float,
    )
    return {
        "sample_count": sample_count,
        "hit_count": hit_count,
        "not_hit_count": not_hit_count,
        "hit_share": float(distribution[1]),
        "not_hit_share": float(distribution[0]),
        "labeled_count": labeled_count,
        "label_coverage": _ratio(labeled_count, sample_count),
        "hit_labeled_count": hit_labeled_count,
        "hit_bad_count": hit_bad_count,
        "hit_bad_rate": (
            None
            if hit_labeled_count == 0
            else _ratio(hit_bad_count, hit_labeled_count)
        ),
        "psi_vs_development": (
            0.0
            if baseline_distribution is None
            else float(compute_psi(baseline_distribution, distribution))
        ),
    }


def _metric_row(
    value: object,
    *,
    name: str,
    baseline: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateStabilityError(f"{name} must be an object")
    _exact_fields(value, _METRIC_FIELDS, name)
    sample_count = _positive_int(value["sample_count"], f"{name}.sample_count")
    hit_count = _count(value["hit_count"], f"{name}.hit_count", sample_count)
    not_hit_count = _count(
        value["not_hit_count"], f"{name}.not_hit_count", sample_count
    )
    if hit_count + not_hit_count != sample_count:
        raise CandidateStabilityError(
            f"{name} hit/not-hit counts do not cover the sample"
        )
    hit_share = _rate(value["hit_share"], f"{name}.hit_share")
    not_hit_share = _rate(value["not_hit_share"], f"{name}.not_hit_share")
    _same_number(hit_share, _ratio(hit_count, sample_count), f"{name}.hit_share")
    _same_number(
        not_hit_share,
        _ratio(not_hit_count, sample_count),
        f"{name}.not_hit_share",
    )
    labeled_count = _count(
        value["labeled_count"], f"{name}.labeled_count", sample_count
    )
    label_coverage = _rate(value["label_coverage"], f"{name}.label_coverage")
    _same_number(
        label_coverage,
        _ratio(labeled_count, sample_count),
        f"{name}.label_coverage",
    )
    hit_labeled_count = _count(
        value["hit_labeled_count"],
        f"{name}.hit_labeled_count",
        min(hit_count, labeled_count),
    )
    hit_bad_count = _count(
        value["hit_bad_count"],
        f"{name}.hit_bad_count",
        hit_labeled_count,
    )
    hit_bad_rate = _optional_rate(
        value["hit_bad_rate"], f"{name}.hit_bad_rate"
    )
    expected_bad_rate = (
        None
        if hit_labeled_count == 0
        else _ratio(hit_bad_count, hit_labeled_count)
    )
    if expected_bad_rate is None:
        if hit_bad_rate is not None:
            raise CandidateStabilityError(
                f"{name}.hit_bad_rate must be null without labelled hits"
            )
    elif hit_bad_rate is None:
        raise CandidateStabilityError(
            f"{name}.hit_bad_rate is required with labelled hits"
        )
    else:
        _same_number(hit_bad_rate, expected_bad_rate, f"{name}.hit_bad_rate")
    psi = _non_negative_number(
        value["psi_vs_development"], f"{name}.psi_vs_development"
    )
    if baseline and psi != 0.0:
        raise CandidateStabilityError(
            "baseline.psi_vs_development must be zero"
        )
    return {
        "sample_count": sample_count,
        "hit_count": hit_count,
        "not_hit_count": not_hit_count,
        "hit_share": hit_share,
        "not_hit_share": not_hit_share,
        "labeled_count": labeled_count,
        "label_coverage": label_coverage,
        "hit_labeled_count": hit_labeled_count,
        "hit_bad_count": hit_bad_count,
        "hit_bad_rate": hit_bad_rate,
        "psi_vs_development": psi,
    }


def _monthly_rows(
    value: object,
    *,
    baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _sequence(value):
        raise CandidateStabilityError("monthly must be a non-empty array")
    if len(value) > CANDIDATE_STABILITY_MAX_MONTHS:
        raise CandidateStabilityError(
            "candidate stability exceeds the 240-month limit"
        )
    rows: list[dict[str, Any]] = []
    baseline_distribution = _distribution(baseline)
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise CandidateStabilityError(f"monthly[{index}] must be an object")
        _exact_fields(raw, _MONTHLY_FIELDS, f"monthly[{index}]")
        month = _month(raw["month"], f"monthly[{index}].month")
        metrics = _metric_row(
            {key: raw[key] for key in _METRIC_FIELDS},
            name=f"monthly[{index}]",
            baseline=False,
        )
        expected_psi = float(
            compute_psi(baseline_distribution, _distribution(metrics))
        )
        _same_number(
            metrics["psi_vs_development"],
            expected_psi,
            f"monthly[{index}].psi_vs_development",
        )
        rows.append({"month": month, **metrics})
    months = [row["month"] for row in rows]
    if months != sorted(months) or len(set(months)) != len(months):
        raise CandidateStabilityError(
            "monthly periods must be unique and sorted ascending"
        )
    return rows


def _require_monthly_rollup(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: Mapping[str, Any],
) -> None:
    fields = (
        "sample_count",
        "hit_count",
        "not_hit_count",
        "labeled_count",
        "hit_labeled_count",
        "hit_bad_count",
    )
    for field in fields:
        if sum(int(row[field]) for row in rows) != int(baseline[field]):
            raise CandidateStabilityError(
                f"monthly {field} does not roll to the development baseline"
            )


def _summary(
    value: object,
    *,
    baseline: Mapping[str, Any],
    monthly: Sequence[Mapping[str, Any]],
    red_flags: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateStabilityError("summary must be an object")
    _exact_fields(value, _SUMMARY_FIELDS, "summary")
    population_count = _positive_int(
        value["population_count"], "summary.population_count"
    )
    month_count = _positive_int(value["month_count"], "summary.month_count")
    max_psi = _non_negative_number(value["max_psi"], "summary.max_psi")
    max_psi_month = _month(value["max_psi_month"], "summary.max_psi_month")
    insufficient_month_count = _count(
        value["insufficient_month_count"],
        "summary.insufficient_month_count",
        month_count,
    )
    if population_count != baseline["sample_count"]:
        raise CandidateStabilityError(
            "summary.population_count does not match baseline"
        )
    if month_count != len(monthly):
        raise CandidateStabilityError("summary.month_count does not match monthly")
    expected_max = max(
        monthly,
        key=lambda row: (float(row["psi_vs_development"]), -int(row["month"])),
    )
    _same_number(max_psi, float(expected_max["psi_vs_development"]), "summary.max_psi")
    if max_psi_month != expected_max["month"]:
        raise CandidateStabilityError(
            "summary.max_psi_month does not match monthly"
        )
    if insufficient_month_count != len(red_flags):
        raise CandidateStabilityError(
            "summary.insufficient_month_count does not match red_flags"
        )
    return {
        "population_count": population_count,
        "month_count": month_count,
        "max_psi": max_psi,
        "max_psi_month": max_psi_month,
        "insufficient_month_count": insufficient_month_count,
    }


def _expected_red_flags(
    monthly: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "kind": "insufficient_month_rows",
            "month": row["month"],
            "observed_rows": int(row["sample_count"]),
            "minimum_rows": CANDIDATE_STABILITY_MIN_MONTH_ROWS,
        }
        for row in monthly
        if int(row["sample_count"]) < CANDIDATE_STABILITY_MIN_MONTH_ROWS
    ]


def _red_flags(value: object) -> list[dict[str, Any]]:
    if not _sequence(value, allow_empty=True):
        raise CandidateStabilityError("red_flags must be an array")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise CandidateStabilityError(f"red_flags[{index}] must be an object")
        _exact_fields(raw, _RED_FLAG_FIELDS, f"red_flags[{index}]")
        if raw["kind"] != "insufficient_month_rows":
            raise CandidateStabilityError(
                f"red_flags[{index}].kind is invalid"
            )
        result.append(
            {
                "kind": "insufficient_month_rows",
                "month": _month(raw["month"], f"red_flags[{index}].month"),
                "observed_rows": _count(
                    raw["observed_rows"],
                    f"red_flags[{index}].observed_rows",
                    CANDIDATE_STABILITY_MIN_MONTH_ROWS - 1,
                ),
                "minimum_rows": _exact_integer(
                    raw["minimum_rows"],
                    CANDIDATE_STABILITY_MIN_MONTH_ROWS,
                    f"red_flags[{index}].minimum_rows",
                ),
            }
        )
    return result


def _identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateStabilityError("identity must be an object")
    _exact_fields(value, _IDENTITY_FIELDS, "identity")
    return {
        "task_id": _text(value["task_id"], "identity.task_id"),
        "dataset_id": _text(value["dataset_id"], "identity.dataset_id"),
        "dataset_content_hash": _hash(
            value["dataset_content_hash"], "identity.dataset_content_hash"
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"], "identity.workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"], "identity.workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "identity.semantic_mapping_hash"
        ),
        "sample_context_hash": _hash(
            value["sample_context_hash"], "identity.sample_context_hash"
        ),
    }


def _source_ref(value: object, *, basis: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateStabilityError("source_ref must be an object")
    expected_kind = _BASIS_TO_SOURCE_KIND[basis]
    if value.get("source_kind") != expected_kind:
        raise CandidateStabilityError(
            "source_ref.source_kind does not match stability basis"
        )
    if expected_kind == "univariate_asset":
        _exact_fields(value, _ASSET_SOURCE_FIELDS, "source_ref")
        return {
            "source_kind": expected_kind,
            "artifact_id": _text(value["artifact_id"], "source_ref.artifact_id"),
            "artifact_content_hash": _hash(
                value["artifact_content_hash"],
                "source_ref.artifact_content_hash",
            ),
            "asset_id": _text(value["asset_id"], "source_ref.asset_id"),
            "asset_hash": _hash(value["asset_hash"], "source_ref.asset_hash"),
            "rule_id": _text(value["rule_id"], "source_ref.rule_id"),
        }
    _exact_fields(value, _POOL_SOURCE_FIELDS, "source_ref")
    return {
        "source_kind": expected_kind,
        "artifact_id": _text(value["artifact_id"], "source_ref.artifact_id"),
        "artifact_content_hash": _hash(
            value["artifact_content_hash"], "source_ref.artifact_content_hash"
        ),
        "pool_id": _text(value["pool_id"], "source_ref.pool_id"),
        "revision": _positive_int(value["revision"], "source_ref.revision"),
        "revision_id": _text(value["revision_id"], "source_ref.revision_id"),
        "snapshot_hash": _hash(
            value["snapshot_hash"], "source_ref.snapshot_hash"
        ),
        "entry_id": _text(value["entry_id"], "source_ref.entry_id"),
        "rule_id": _text(value["rule_id"], "source_ref.rule_id"),
    }


def _sample_design_ref(value: object) -> dict[str, str]:
    try:
        reference = StrategyRiskDevelopmentRef.from_value(value)
        if reference.partition not in {"development", "risk/development"}:
            raise StrategyError(
                "sample_design_ref.partition must be development or "
                "risk/development"
            )
        return reference.to_ref_dict()
    except StrategyError as exc:
        raise CandidateStabilityError(
            f"candidate stability sample_design_ref is invalid: {exc}"
        ) from exc


def _bindings(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateStabilityError("bindings must be an object")
    _exact_fields(value, _BINDING_FIELDS, "bindings")
    month_col = _text(value["month_col"], "bindings.month_col")
    target_col = _text(value["target_col"], "bindings.target_col")
    if month_col == target_col:
        raise CandidateStabilityError(
            "bindings month_col and target_col must be different"
        )
    return {
        "month_col": month_col,
        "target_col": target_col,
        "target_bad_value": _exact_integer(
            value["target_bad_value"], 1, "bindings.target_bad_value"
        ),
        "minimum_month_rows": _exact_integer(
            value["minimum_month_rows"],
            CANDIDATE_STABILITY_MIN_MONTH_ROWS,
            "bindings.minimum_month_rows",
        ),
    }


def _basis(value: object) -> str:
    normalized = _text(value, "basis")
    if normalized not in _BASIS_TO_SOURCE_KIND:
        raise CandidateStabilityError(
            "basis must be asset_rule_hit or "
            "pool_entry_incremental_first_match"
        )
    return normalized


def _distribution(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(row["not_hit_share"]), float(row["hit_share"])],
        dtype=float,
    )


def _month(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _MONTH_RE.fullmatch(normalized) is None:
        raise CandidateStabilityError(f"{name} must use YYYYMM")
    month_number = int(normalized[4:])
    if not 1 <= month_number <= 12:
        raise CandidateStabilityError(f"{name} must use a valid YYYYMM")
    return normalized


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized < 1:
        raise CandidateStabilityError(f"{name} must be a positive integer")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 0
        or int(value) > _MAX_SAFE_JSON_INTEGER
    ):
        raise CandidateStabilityError(
            f"{name} must be a non-negative JSON-safe integer"
        )
    return int(value)


def _count(value: object, name: str, maximum: int) -> int:
    normalized = _non_negative_int(value, name)
    if normalized > maximum:
        raise CandidateStabilityError(f"{name} exceeds its population")
    return normalized


def _exact_integer(value: object, expected: int, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized != expected:
        raise CandidateStabilityError(f"{name} must be {expected}")
    return normalized


def _rate(value: object, name: str) -> float:
    normalized = _number(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise CandidateStabilityError(f"{name} must be between 0 and 1")
    return normalized


def _optional_rate(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _rate(value, name)


def _non_negative_number(value: object, name: str) -> float:
    normalized = _number(value, name)
    if normalized < 0:
        raise CandidateStabilityError(f"{name} must be non-negative")
    return normalized


def _number(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise CandidateStabilityError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CandidateStabilityError(f"{name} must be a finite number")
    return normalized


def _same_number(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise CandidateStabilityError(f"{name} does not match derived metrics")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateStabilityError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise CandidateStabilityError(f"{name} must be a lowercase SHA-256")
    return normalized


def _sequence(value: object, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and (allow_empty or len(value) > 0)
    )


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields " + ", ".join(unexpected))
        raise CandidateStabilityError(f"{name} has " + "; ".join(details))


def _json_value(value: object, name: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        normalized = int(value)
        if abs(normalized) > _MAX_SAFE_JSON_INTEGER:
            raise CandidateStabilityError(
                f"{name} must contain JSON-safe integers"
            )
        return normalized
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise CandidateStabilityError(f"{name} must contain finite JSON")
        return normalized
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CandidateStabilityError(f"{name} keys must be strings")
        return {
            key: _json_value(child, f"{name}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [
            _json_value(child, f"{name}[{index}]")
            for index, child in enumerate(value)
        ]
    raise CandidateStabilityError(f"{name} must contain canonical JSON values")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _json_value(value, "candidate stability"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CandidateStabilityError(
            "candidate stability must contain finite canonical JSON"
        ) from exc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
