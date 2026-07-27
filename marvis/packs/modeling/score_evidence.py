"""Deterministic score-to-Strategy-evidence kernel.

This module is pure with respect to MARVIS runtime state.  It accepts an
independently authenticated StrategySampleDesign bundle, its decoded six masks,
the exact bound active frame, and one already-validated probability vector.
All bins, distributions, outcome metrics, and observation statuses are derived
by platform code; no caller-supplied metric value is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import hashlib
import hmac
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.feature.binning import assign_bins, equal_frequency_edges
from marvis.feature.metrics import compute_psi, feature_auc, feature_ks
from marvis.artifacts.model_score_vector import (
    MAX_MODEL_SCORE_VECTOR_ROWS,
    MODEL_SCORE_VECTOR_SCHEMA_VERSION,
    MODEL_SCORE_VECTOR_WRITER_VERSION,
    ModelScoreVector,
)
from marvis.packs.modeling.evidence import (
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
    MODEL_BINARY_REF_KIND,
    RAW_SCORE_PRODUCT,
    validate_modeling_training_evidence,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.model_evidence import (
    MAX_OBSERVATIONS_PER_EVIDENCE,
    StrategyModelEvidenceError,
    build_artifact_ref,
    build_evidence_source_ref,
    build_model_observation,
    build_score_bin,
    build_single_model_evidence,
    sample_partition_refs_from_strategy_sample_design_v2,
    validate_single_model_evidence,
)
from marvis.packs.strategy.sample_design_v2 import (
    validate_strategy_sample_design_v2_bundle,
)
from marvis.packs.strategy.sample_membership import MEMBERSHIP_MASK_ORDER


MODEL_SCORE_EVIDENCE_PRODUCER_VERSION = "marvis.modeling.score-evidence/1"
MODEL_SCORE_EVIDENCE_SCHEMA_VERSION = "modeling.model-score-evidence.v1"
MODEL_SCORE_EVIDENCE_ARTIFACT_KIND = "model_score_evidence_json"
MODEL_SCORE_VECTOR_ARTIFACT_KIND = "model_score_vector_parquet"
MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES = 10 * 1024 * 1024
MODEL_SCORE_INPUT_SPACE = "bound_active_training_dataset"
MODEL_SCORE_DIRECTION = "higher_is_riskier"
MAX_MODEL_SCORE_BINS = 10
MAX_GOVERNED_SCORE_MONTHS = 240

_MONTH_COMPACT_RE = re.compile(r"^([0-9]{4})(0[1-9]|1[0-2])$")
_MONTH_CANONICAL_RE = re.compile(r"^([0-9]{4})-(0[1-9]|1[0-2])$")
_DISTRIBUTION_METRICS = ("score_bin_count", "score_bin_share")
_DISCRIMINATION_METRICS = (
    ("auc", "ratio"),
    ("ks", "ratio"),
    ("lift_head_5", "multiple"),
    ("lift_tail_5", "multiple"),
    ("lift_head_10", "multiple"),
    ("lift_tail_10", "multiple"),
)
_CALIBRATION_METRICS = (
    "score_bin_bad_rate",
    "calibration_predicted_rate",
    "calibration_observed_rate",
    "calibration_gap",
    "calibration_abs_gap",
)
_MONTHLY_DISCRIMINATION_METRICS = (
    ("monthly_auc", "ratio"),
    ("monthly_ks", "ratio"),
    ("monthly_lift_head_10", "multiple"),
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "artifact_kind",
        "evidence_id",
        "task_id",
        "score_product",
        "scoring_contract",
        "training_evidence_ref",
        "model_ref",
        "sample_design_binding",
        "score_vector_ref",
        "score_vector_contract",
        "single_model_evidence",
        "resource_budgets",
        "governance",
        "content_hash",
    }
)
_SCORING_CONTRACT_FIELDS = frozenset(
    {
        "input_space",
        "load_calibration",
        "replay_preprocessing",
        "rows_scored_exactly_once",
        "row_ordinal",
        "score_direction",
    }
)
_ROW_ORDINAL_FIELDS = frozenset({"start", "stop", "step"})
_SCORE_VECTOR_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "writer_version",
        "format",
        "row_count",
        "row_ordinal",
        "score_dtype",
        "score_min",
        "score_max",
        "content_hash",
    }
)
_RESOURCE_BUDGET_FIELDS = frozenset(
    {
        "max_rows",
        "rows_scored",
        "max_score_bins",
        "score_bins_used",
        "max_months",
        "months_used",
        "max_observations",
        "observations_used",
    }
)
_GOVERNANCE_FIELDS = frozenset(
    {"not_compared", "not_selected", "not_adopted", "not_deployed"}
)
_TRAINING_REF_FIELDS = frozenset(
    {
        "sample_design_ref",
        "model_binary_artifact_id",
        "expected_model_binary_artifact_content_hash",
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "expected_experiment_id",
        "expected_model_artifact_id",
        "expected_evidence_id",
        "expected_evidence_content_hash",
    }
)
_SAMPLE_INPUT_REF_FIELDS = frozenset(
    {
        "membership_artifact_id",
        "expected_membership_artifact_content_hash",
        "bundle_artifact_id",
        "expected_bundle_artifact_content_hash",
        "expected_bundle_id",
        "expected_sample_design_id",
        "expected_sample_design_content_hash",
    }
)


class ModelScoreEvidenceError(ValueError):
    """Exact score evidence could not be derived from the governed inputs."""


def build_model_score_evidence_envelope(
    *,
    task_id: str,
    training_evidence_ref: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    sample_design_bundle: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
    score_vector: ModelScoreVector,
    single_model_evidence: Mapping[str, Any],
    producer_version: str = MODEL_SCORE_EVIDENCE_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build the canonical outer evidence that binds every exact artifact ref."""

    if not isinstance(score_vector, ModelScoreVector):
        raise ModelScoreEvidenceError("score_vector must be fully validated")
    normalized_single = validate_single_model_evidence(
        single_model_evidence,
        sample_design_bundle=sample_design_bundle,
    )
    periods = {
        item["period"]
        for item in normalized_single["observations"]
        if item["period"] is not None
    }
    body = {
        "schema_version": MODEL_SCORE_EVIDENCE_SCHEMA_VERSION,
        "producer_version": producer_version,
        "artifact_kind": MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        "task_id": task_id,
        "score_product": RAW_SCORE_PRODUCT,
        "scoring_contract": {
            "input_space": MODEL_SCORE_INPUT_SPACE,
            "load_calibration": False,
            "replay_preprocessing": False,
            "rows_scored_exactly_once": True,
            "row_ordinal": {
                "start": 0,
                "stop": score_vector.row_count,
                "step": 1,
            },
            "score_direction": MODEL_SCORE_DIRECTION,
        },
        "training_evidence_ref": training_evidence_ref,
        "model_ref": model_ref,
        "sample_design_binding": training_evidence["sample_design_binding"],
        "score_vector_ref": score_ref,
        "score_vector_contract": {
            "schema_version": MODEL_SCORE_VECTOR_SCHEMA_VERSION,
            "writer_version": MODEL_SCORE_VECTOR_WRITER_VERSION,
            "format": "parquet",
            "row_count": score_vector.row_count,
            "row_ordinal": {
                "start": 0,
                "stop": score_vector.row_count,
                "step": 1,
            },
            "score_dtype": "float64",
            "score_min": score_vector.score_min,
            "score_max": score_vector.score_max,
            "content_hash": score_vector.content_hash,
        },
        "single_model_evidence": normalized_single,
        "resource_budgets": {
            "max_rows": MAX_MODEL_SCORE_VECTOR_ROWS,
            "rows_scored": score_vector.row_count,
            "max_score_bins": MAX_MODEL_SCORE_BINS,
            "score_bins_used": len(normalized_single["score_bins"]),
            "max_months": MAX_GOVERNED_SCORE_MONTHS,
            "months_used": len(periods),
            "max_observations": MAX_OBSERVATIONS_PER_EVIDENCE,
            "observations_used": len(normalized_single["observations"]),
        },
        "governance": _governance(),
    }
    normalized_body = _normalize_envelope_body(
        body,
        sample_design_bundle=sample_design_bundle,
        training_evidence=training_evidence,
        expected_training_evidence_ref=training_evidence_ref,
        score_vector=score_vector,
    )
    digest = _sha256(_canonical_json(normalized_body).encode("utf-8"))
    evidence_id = f"model-score-evidence-{digest[:24]}"
    addressed = {**normalized_body, "evidence_id": evidence_id}
    result = {
        **addressed,
        "content_hash": _sha256(_canonical_json(addressed).encode("utf-8")),
    }
    return validate_model_score_evidence_envelope(
        result,
        sample_design_bundle=sample_design_bundle,
        training_evidence=training_evidence,
        expected_training_evidence_ref=training_evidence_ref,
        score_vector=score_vector,
    )


def validate_model_score_evidence_envelope(
    value: Mapping[str, Any],
    *,
    sample_design_bundle: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    expected_training_evidence_ref: Mapping[str, Any],
    score_vector: ModelScoreVector | None = None,
) -> dict[str, Any]:
    """Strictly validate the outer JSON against live training/sample/vector facts."""

    obj = _object(value, "model score evidence envelope")
    _exact_fields(obj, _ENVELOPE_FIELDS, "model score evidence envelope")
    supplied_id = _text(obj["evidence_id"], "evidence_id")
    supplied_hash = _hash(obj["content_hash"], "content_hash")
    body = {key: obj[key] for key in obj if key not in {"evidence_id", "content_hash"}}
    normalized_body = _normalize_envelope_body(
        body,
        sample_design_bundle=sample_design_bundle,
        training_evidence=training_evidence,
        expected_training_evidence_ref=expected_training_evidence_ref,
        score_vector=score_vector,
    )
    digest = _sha256(_canonical_json(normalized_body).encode("utf-8"))
    expected_id = f"model-score-evidence-{digest[:24]}"
    if supplied_id != expected_id:
        raise ModelScoreEvidenceError(
            "model score evidence evidence_id does not match content"
        )
    addressed = {**normalized_body, "evidence_id": supplied_id}
    expected_hash = _sha256(_canonical_json(addressed).encode("utf-8"))
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise ModelScoreEvidenceError(
            "model score evidence content_hash does not match content"
        )
    normalized = {**addressed, "content_hash": supplied_hash}
    if (
        len(_canonical_json(normalized).encode("utf-8"))
        > MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES
    ):
        raise ModelScoreEvidenceError("model score evidence exceeds JSON byte budget")
    return normalized


def canonical_model_score_evidence_json(
    value: Mapping[str, Any],
    *,
    sample_design_bundle: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    expected_training_evidence_ref: Mapping[str, Any],
    score_vector: ModelScoreVector | None = None,
) -> str:
    return _canonical_json(
        validate_model_score_evidence_envelope(
            value,
            sample_design_bundle=sample_design_bundle,
            training_evidence=training_evidence,
            expected_training_evidence_ref=expected_training_evidence_ref,
            score_vector=score_vector,
        )
    )


def model_score_evidence_from_json(
    raw: str | bytes | bytearray,
    *,
    sample_design_bundle: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    expected_training_evidence_ref: Mapping[str, Any],
    score_vector: ModelScoreVector | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, (str, bytes, bytearray)):
        raise ModelScoreEvidenceError("model score evidence JSON must be text or bytes")
    encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    if len(encoded) > MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES:
        raise ModelScoreEvidenceError("model score evidence exceeds JSON byte budget")
    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except ModelScoreEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelScoreEvidenceError(
            "model score evidence is not valid bounded JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ModelScoreEvidenceError(
            "model score evidence JSON must contain an object"
        )
    normalized = validate_model_score_evidence_envelope(
        value,
        sample_design_bundle=sample_design_bundle,
        training_evidence=training_evidence,
        expected_training_evidence_ref=expected_training_evidence_ref,
        score_vector=score_vector,
    )
    if encoded != _canonical_json(normalized).encode("utf-8"):
        raise ModelScoreEvidenceError(
            "model score evidence JSON must use canonical encoding"
        )
    return normalized


def build_single_model_score_evidence(
    *,
    sample_design_bundle: Mapping[str, Any],
    membership_masks: Mapping[str, object],
    frame: pd.DataFrame,
    scores: Sequence[float] | np.ndarray,
    training_evidence_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
    features: Sequence[str],
) -> dict[str, Any]:
    """Build a validated ``Strategy SingleModelEvidence`` from exact rows."""

    try:
        bundle = validate_strategy_sample_design_v2_bundle(sample_design_bundle)
        if not isinstance(frame, pd.DataFrame):
            raise ModelScoreEvidenceError(
                "bound active training dataset must be a DataFrame"
            )
        probability = _probabilities(scores, row_count=len(frame))
        masks = _membership_masks(
            membership_masks,
            row_count=len(frame),
            sample_design_bundle=bundle,
        )
        target_column, raw_good_value, raw_bad_value = _governed_target(bundle)
        if target_column not in frame.columns:
            raise ModelScoreEvidenceError(
                "governed target column is absent from bound active training dataset"
            )
        labels = _binary_labels(
            frame[target_column],
            field=target_column,
            raw_good_value=raw_good_value,
            raw_bad_value=raw_bad_value,
        )
        development_mask = masks["risk/development"]
        development_labels = labels[development_mask]
        development_labeled = development_labels[np.isfinite(development_labels)]
        if set(development_labeled.astype(int).tolist()) != {0, 1}:
            raise ModelScoreEvidenceError(
                "bound risk/development training dataset must contain both labeled classes"
            )
        month_field = bundle["sample_design"]["sample_semantics"]["field_bindings"][
            "month_field"
        ]
        if month_field is None:
            normalized_months: tuple[str, ...] | None = None
            distinct_months: tuple[str, ...] = ()
        else:
            if month_field not in frame.columns:
                raise ModelScoreEvidenceError(
                    "governed month field is absent from bound active training dataset"
                )
            normalized_months = normalize_governed_months(frame[month_field].tolist())
            distinct_months = tuple(sorted(set(normalized_months)))

        normalized_training_ref = _artifact_ref(
            training_evidence_ref,
            "training_evidence_ref",
        )
        normalized_model_ref = _artifact_ref(model_ref, "model_ref")
        normalized_score_ref = _artifact_ref(score_ref, "score_ref")
        normalized_features = _features(features)
        edges = _development_score_edges(probability[development_mask])
        bin_ids = tuple(f"score-bin-{index:02d}" for index in range(len(edges) - 1))
        training_source = _sample_source(
            bundle,
            population="risk",
            partition="development",
            artifact_ref=normalized_training_ref,
        )
        definition_source = _sample_source(
            bundle,
            population="risk",
            partition="development",
            artifact_ref=normalized_score_ref,
        )
        score_bins = [
            build_score_bin(
                sample_design_bundle=bundle,
                ordinal=index,
                bin_id=bin_ids[index],
                lower_bound=None if index == 0 else float(edges[index]),
                upper_bound=(
                    None if index == len(edges) - 2 else float(edges[index + 1])
                ),
                lower_inclusive=index != 0,
                upper_inclusive=False,
                definition_ref=definition_source,
                model_ref=normalized_model_ref,
                score_ref=normalized_score_ref,
            )
            for index in range(len(bin_ids))
        ]
        development_counts = _bin_counts(
            probability[development_mask],
            edges=edges,
        )
        observations: list[dict[str, Any]] = []
        for sample_ref in sample_partition_refs_from_strategy_sample_design_v2(bundle):
            population = sample_ref["population"]
            partition = sample_ref["partition"]
            sample_mask = masks[f"{population}/{partition}"]
            source = _sample_source(
                bundle,
                population=population,
                partition=partition,
                artifact_ref=normalized_score_ref,
            )
            observations.extend(
                _slice_observations(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    sample_mask=sample_mask,
                    scores=probability,
                    labels=labels,
                    edges=edges,
                    bin_ids=bin_ids,
                    development_counts=development_counts,
                    source_ref=source,
                    model_ref=normalized_model_ref,
                    score_ref=normalized_score_ref,
                )
            )
            if normalized_months is not None:
                observations.extend(
                    _monthly_observations(
                        bundle=bundle,
                        population=population,
                        partition=partition,
                        sample_mask=sample_mask,
                        months=np.asarray(normalized_months, dtype=object),
                        distinct_months=distinct_months,
                        scores=probability,
                        labels=labels,
                        edges=edges,
                        development_counts=development_counts,
                        source_ref=source,
                        model_ref=normalized_model_ref,
                        score_ref=normalized_score_ref,
                    )
                )
        evidence = build_single_model_evidence(
            sample_design_bundle=bundle,
            training_source_ref=training_source,
            model_ref=normalized_model_ref,
            score_ref=normalized_score_ref,
            features=normalized_features,
            score_bins=score_bins,
            observations=observations,
        )
        return validate_single_model_evidence(
            evidence,
            sample_design_bundle=bundle,
        )
    except ModelScoreEvidenceError:
        raise
    except (StrategyError, StrategyModelEvidenceError, TypeError, ValueError) as exc:
        raise ModelScoreEvidenceError(str(exc)) from exc


def normalize_governed_months(values: Sequence[object]) -> tuple[str, ...]:
    """Normalize governed month values to ``YYYY-MM`` under a 240-month budget."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ModelScoreEvidenceError("governed month values must be a sequence")
    result: list[str] = []
    distinct: set[str] = set()
    for index, value in enumerate(values):
        month = _month(value, index=index)
        result.append(month)
        distinct.add(month)
        if len(distinct) > MAX_GOVERNED_SCORE_MONTHS:
            raise ModelScoreEvidenceError(
                "governed score month budget exceeds 240 distinct months"
            )
    return tuple(result)


def _slice_observations(
    *,
    bundle: Mapping[str, Any],
    population: str,
    partition: str,
    sample_mask: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
    bin_ids: Sequence[str],
    development_counts: np.ndarray,
    source_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sample_scores = scores[sample_mask]
    sample_labels = labels[sample_mask]
    sample_count = int(sample_scores.size)
    if sample_count == 0:
        result = [
            _observation(
                bundle=bundle,
                population=population,
                partition=partition,
                metric_key=metric,
                status="unavailable",
                unit="count" if metric == "score_bin_count" else "ratio",
                source_ref=source_ref,
                model_ref=model_ref,
                score_ref=score_ref,
                bin_id=bin_id,
                reason="empty_sample_partition",
            )
            for bin_id in bin_ids
            for metric in _DISTRIBUTION_METRICS
        ]
        result.append(
            _observation(
                bundle=bundle,
                population=population,
                partition=partition,
                metric_key="score_psi",
                status="unavailable",
                unit="number",
                source_ref=source_ref,
                model_ref=model_ref,
                score_ref=score_ref,
                reason="empty_sample_partition",
            )
        )
        if population == "risk":
            result.extend(
                _unavailable_discrimination(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    source_ref=source_ref,
                    model_ref=model_ref,
                    score_ref=score_ref,
                    reason="empty_sample_partition",
                )
            )
            result.extend(
                _unavailable_calibration_bins(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    bin_ids=bin_ids,
                    source_ref=source_ref,
                    model_ref=model_ref,
                    score_ref=score_ref,
                    reason="empty_sample_partition",
                )
            )
        return result

    counts = _bin_counts(sample_scores, edges=edges)
    if int(counts.sum()) != sample_count:
        raise ModelScoreEvidenceError(
            f"{population}/{partition} score bins do not conserve rows"
        )
    shares = counts.astype(np.float64) / float(sample_count)
    if not math.isclose(float(shares.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ModelScoreEvidenceError(
            f"{population}/{partition} score-bin shares do not conserve population"
        )
    result: list[dict[str, Any]] = []
    for index, bin_id in enumerate(bin_ids):
        count = int(counts[index])
        result.append(
            _observation(
                bundle=bundle,
                population=population,
                partition=partition,
                metric_key="score_bin_count",
                status="present",
                value=count,
                numerator=count,
                denominator=sample_count,
                sample_count=sample_count,
                unit="count",
                source_ref=source_ref,
                model_ref=model_ref,
                score_ref=score_ref,
                bin_id=bin_id,
            )
        )
        result.append(
            _observation(
                bundle=bundle,
                population=population,
                partition=partition,
                metric_key="score_bin_share",
                status="present",
                value=float(shares[index]),
                numerator=count,
                denominator=sample_count,
                sample_count=sample_count,
                unit="ratio",
                source_ref=source_ref,
                model_ref=model_ref,
                score_ref=score_ref,
                bin_id=bin_id,
            )
        )
    psi = float(compute_psi(development_counts, counts))
    result.append(
        _observation(
            bundle=bundle,
            population=population,
            partition=partition,
            metric_key="score_psi",
            status="present",
            value=psi,
            sample_count=sample_count,
            unit="number",
            source_ref=source_ref,
            model_ref=model_ref,
            score_ref=score_ref,
        )
    )
    if population != "risk":
        return result

    labeled_mask = np.isfinite(sample_labels)
    labeled_scores = sample_scores[labeled_mask]
    labeled = sample_labels[labeled_mask].astype(np.int8)
    if labeled.size == 0:
        result.extend(
            _unavailable_discrimination(
                bundle=bundle,
                population=population,
                partition=partition,
                source_ref=source_ref,
                model_ref=model_ref,
                score_ref=score_ref,
                reason="no_labeled_rows",
            )
        )
    elif np.unique(labeled).size < 2:
        result.extend(
            _unavailable_discrimination(
                bundle=bundle,
                population=population,
                partition=partition,
                source_ref=source_ref,
                model_ref=model_ref,
                score_ref=score_ref,
                reason="requires_two_labeled_classes",
            )
        )
    else:
        metrics = _discrimination_values(labeled_scores, labeled)
        for metric_key, unit in _DISCRIMINATION_METRICS:
            result.append(
                _observation(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    metric_key=metric_key,
                    status="present",
                    value=metrics[metric_key],
                    sample_count=int(labeled.size),
                    unit=unit,
                    source_ref=source_ref,
                    model_ref=model_ref,
                    score_ref=score_ref,
                )
            )
    assigned = assign_bins(sample_scores, edges)
    for index, bin_id in enumerate(bin_ids):
        bin_labeled_mask = (assigned == index) & labeled_mask
        if not np.any(bin_labeled_mask):
            result.extend(
                _unavailable_calibration_bin(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    bin_id=bin_id,
                    source_ref=source_ref,
                    model_ref=model_ref,
                    score_ref=score_ref,
                    reason="bin_has_no_labeled_rows",
                )
            )
            continue
        bin_scores = sample_scores[bin_labeled_mask]
        bin_labels = sample_labels[bin_labeled_mask].astype(np.int8)
        labeled_count = int(bin_labels.size)
        bad_count = int(np.count_nonzero(bin_labels == 1))
        predicted = float(np.mean(bin_scores))
        observed = bad_count / labeled_count
        values = {
            "score_bin_bad_rate": (observed, bad_count, labeled_count),
            "calibration_predicted_rate": (predicted, None, None),
            "calibration_observed_rate": (observed, bad_count, labeled_count),
            "calibration_gap": (predicted - observed, None, None),
            "calibration_abs_gap": (abs(predicted - observed), None, None),
        }
        for metric_key in _CALIBRATION_METRICS:
            value, numerator, denominator = values[metric_key]
            result.append(
                _observation(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    metric_key=metric_key,
                    status="present",
                    value=float(value),
                    numerator=numerator,
                    denominator=denominator,
                    sample_count=labeled_count,
                    unit="ratio",
                    source_ref=source_ref,
                    model_ref=model_ref,
                    score_ref=score_ref,
                    bin_id=bin_id,
                )
            )
    return result


def _monthly_observations(
    *,
    bundle: Mapping[str, Any],
    population: str,
    partition: str,
    sample_mask: np.ndarray,
    months: np.ndarray,
    distinct_months: Sequence[str],
    scores: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
    development_counts: np.ndarray,
    source_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for period in distinct_months:
        month_mask = sample_mask & (months == period)
        month_scores = scores[month_mask]
        month_labels = labels[month_mask]
        month_count = int(month_scores.size)
        if month_count == 0:
            result.append(
                _observation(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    metric_key="monthly_psi",
                    status="unavailable",
                    unit="number",
                    source_ref=source_ref,
                    model_ref=model_ref,
                    score_ref=score_ref,
                    period=period,
                    reason="empty_month_sample",
                )
            )
        else:
            counts = _bin_counts(month_scores, edges=edges)
            if int(counts.sum()) != month_count:
                raise ModelScoreEvidenceError(
                    f"{population}/{partition}/{period} bins do not conserve rows"
                )
            result.append(
                _observation(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    metric_key="monthly_psi",
                    status="present",
                    value=float(compute_psi(development_counts, counts)),
                    sample_count=month_count,
                    unit="number",
                    source_ref=source_ref,
                    model_ref=model_ref,
                    score_ref=score_ref,
                    period=period,
                )
            )
        if population != "risk":
            continue
        labeled_mask = np.isfinite(month_labels)
        labeled_scores = month_scores[labeled_mask]
        labeled = month_labels[labeled_mask].astype(np.int8)
        if month_count == 0:
            reason = "empty_month_sample"
        elif labeled.size == 0:
            reason = "no_labeled_rows"
        elif np.unique(labeled).size < 2:
            reason = "requires_two_labeled_classes"
        else:
            reason = None
        if reason is not None:
            for metric_key, unit in _MONTHLY_DISCRIMINATION_METRICS:
                result.append(
                    _observation(
                        bundle=bundle,
                        population=population,
                        partition=partition,
                        metric_key=metric_key,
                        status="unavailable",
                        unit=unit,
                        source_ref=source_ref,
                        model_ref=model_ref,
                        score_ref=score_ref,
                        period=period,
                        reason=reason,
                    )
                )
            continue
        values = _discrimination_values(labeled_scores, labeled)
        monthly_values = {
            "monthly_auc": values["auc"],
            "monthly_ks": values["ks"],
            "monthly_lift_head_10": values["lift_head_10"],
        }
        for metric_key, unit in _MONTHLY_DISCRIMINATION_METRICS:
            result.append(
                _observation(
                    bundle=bundle,
                    population=population,
                    partition=partition,
                    metric_key=metric_key,
                    status="present",
                    value=monthly_values[metric_key],
                    sample_count=int(labeled.size),
                    unit=unit,
                    source_ref=source_ref,
                    model_ref=model_ref,
                    score_ref=score_ref,
                    period=period,
                )
            )
    return result


def _unavailable_discrimination(
    *,
    bundle: Mapping[str, Any],
    population: str,
    partition: str,
    source_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        _observation(
            bundle=bundle,
            population=population,
            partition=partition,
            metric_key=metric_key,
            status="unavailable",
            unit=unit,
            source_ref=source_ref,
            model_ref=model_ref,
            score_ref=score_ref,
            reason=reason,
        )
        for metric_key, unit in _DISCRIMINATION_METRICS
    ]


def _unavailable_calibration_bins(
    *,
    bundle: Mapping[str, Any],
    population: str,
    partition: str,
    bin_ids: Sequence[str],
    source_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        observation
        for bin_id in bin_ids
        for observation in _unavailable_calibration_bin(
            bundle=bundle,
            population=population,
            partition=partition,
            bin_id=bin_id,
            source_ref=source_ref,
            model_ref=model_ref,
            score_ref=score_ref,
            reason=reason,
        )
    ]


def _unavailable_calibration_bin(
    *,
    bundle: Mapping[str, Any],
    population: str,
    partition: str,
    bin_id: str,
    source_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        _observation(
            bundle=bundle,
            population=population,
            partition=partition,
            metric_key=metric_key,
            status="unavailable",
            unit="ratio",
            source_ref=source_ref,
            model_ref=model_ref,
            score_ref=score_ref,
            bin_id=bin_id,
            reason=reason,
        )
        for metric_key in _CALIBRATION_METRICS
    ]


def _observation(
    *,
    bundle: Mapping[str, Any],
    population: str,
    partition: str,
    metric_key: str,
    status: str,
    unit: str,
    source_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
    value: int | float | None = None,
    numerator: int | float | None = None,
    denominator: int | float | None = None,
    sample_count: int | None = None,
    bin_id: str | None = None,
    period: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return build_model_observation(
        sample_design_bundle=bundle,
        population=population,
        partition=partition,
        metric_key=metric_key,
        status=status,
        value=value,
        numerator=numerator,
        denominator=denominator,
        sample_count=sample_count,
        unit=unit,
        source_ref=source_ref,
        model_ref=model_ref,
        score_ref=score_ref,
        bin_id=bin_id,
        period=period,
        reason=reason,
    )


def _discrimination_values(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    lifts = _fixed_direction_lifts(scores, labels)
    return {
        "auc": float(feature_auc(scores, labels, direction_agnostic=False)),
        "ks": float(feature_ks(scores, labels)),
        **lifts,
    }


def _fixed_direction_lifts(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    probability = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int8)
    base_rate = float(np.mean(target == 1))
    if probability.size == 0 or base_rate <= 0.0 or base_rate >= 1.0:
        raise ModelScoreEvidenceError("lift requires both labeled classes")
    order = np.argsort(probability, kind="mergesort")
    result: dict[str, float] = {}
    for fraction in (0.05, 0.10):
        count = max(1, int(np.floor(fraction * probability.size)))
        suffix = int(round(fraction * 100))
        result[f"lift_head_{suffix}"] = float(
            np.mean(target[order[-count:]] == 1) / base_rate
        )
        result[f"lift_tail_{suffix}"] = float(
            np.mean(target[order[:count]] == 1) / base_rate
        )
    return result


def _development_score_edges(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        raise ModelScoreEvidenceError(
            "risk/development score reference population is empty"
        )
    edges = equal_frequency_edges(scores, MAX_MODEL_SCORE_BINS)
    if edges.ndim != 1 or not 2 <= edges.size <= MAX_MODEL_SCORE_BINS + 1:
        raise ModelScoreEvidenceError("score-bin edge budget is invalid")
    if np.any(np.diff(edges) <= 0):
        raise ModelScoreEvidenceError("score-bin edges must be strictly increasing")
    return np.asarray(edges, dtype=np.float64)


def _bin_counts(scores: np.ndarray, *, edges: np.ndarray) -> np.ndarray:
    assigned = assign_bins(scores, edges)
    if np.any(assigned < 0):
        raise ModelScoreEvidenceError("finite score was not assigned to a score bin")
    return np.bincount(assigned, minlength=len(edges) - 1).astype(np.int64)


def _membership_masks(
    value: Mapping[str, object],
    *,
    row_count: int,
    sample_design_bundle: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != set(MEMBERSHIP_MASK_ORDER):
        raise ModelScoreEvidenceError(
            "membership_masks must contain the exact six governed masks"
        )
    normalized: dict[str, np.ndarray] = {}
    expected_counts = {
        f"{item['population']}/{item['partition']}": int(item["row_count"])
        for item in sample_partition_refs_from_strategy_sample_design_v2(
            sample_design_bundle
        )
    }
    for name in MEMBERSHIP_MASK_ORDER:
        raw = np.asarray(value[name])
        if raw.ndim != 1 or raw.dtype.kind != "b" or len(raw) != row_count:
            raise ModelScoreEvidenceError(
                f"membership mask {name} does not match active dataset rows"
            )
        mask = np.ascontiguousarray(raw, dtype=np.bool_)
        if int(np.count_nonzero(mask)) != expected_counts[name]:
            raise ModelScoreEvidenceError(
                f"membership mask {name} count changed from SampleDesign"
            )
        normalized[name] = mask
    for population in ("approval", "risk"):
        combined = sum(
            (
                normalized[f"{population}/{partition}"].astype(np.int8)
                for partition in ("development", "validation", "oot")
            ),
            start=np.zeros(row_count, dtype=np.int8),
        )
        if np.any(combined > 1):
            raise ModelScoreEvidenceError(f"{population} partition masks overlap")
    return normalized


def _probabilities(
    value: Sequence[float] | np.ndarray,
    *,
    row_count: int,
) -> np.ndarray:
    if row_count <= 0:
        raise ModelScoreEvidenceError("bound active training dataset must not be empty")
    if row_count > MAX_MODEL_SCORE_VECTOR_ROWS:
        raise ModelScoreEvidenceError(
            "bound active training dataset exceeds score row budget"
        )
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ModelScoreEvidenceError("scores must be a one-dimensional numeric vector")
    scores = np.ascontiguousarray(raw, dtype=np.float64)
    if scores.size != row_count:
        raise ModelScoreEvidenceError(
            "score vector must contain every active dataset row exactly once"
        )
    if not np.all(np.isfinite(scores)):
        raise ModelScoreEvidenceError("scores must all be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ModelScoreEvidenceError("scores must all be in [0, 1]")
    return scores


def _binary_labels(
    series: pd.Series,
    *,
    field: str,
    raw_good_value: int,
    raw_bad_value: int,
) -> np.ndarray:
    result = np.full(len(series), np.nan, dtype=np.float64)
    for index, value in enumerate(series.tolist()):
        if _is_missing(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            raise ModelScoreEvidenceError(
                f"governed target {field} contains a boolean label"
            )
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if math.isfinite(numeric) and numeric == float(raw_good_value):
                result[index] = 0.0
                continue
            if math.isfinite(numeric) and numeric == float(raw_bad_value):
                result[index] = 1.0
                continue
        raise ModelScoreEvidenceError(
            f"governed target {field} contains a non-binary label"
        )
    return result


def _governed_target(bundle: Mapping[str, Any]) -> tuple[str, int, int]:
    target = bundle["sample_design"]["target_selector"]
    if (
        target["status"] != "resolved"
        or target["good_value"] == target["bad_value"]
        or {target["good_value"], target["bad_value"]} != {0, 1}
    ):
        raise ModelScoreEvidenceError(
            "score evidence requires governed complementary binary target semantics"
        )
    return (
        str(target["column"]),
        int(target["good_value"]),
        int(target["bad_value"]),
    )


def _sample_source(
    bundle: Mapping[str, Any],
    *,
    population: str,
    partition: str,
    artifact_ref: Mapping[str, str],
) -> dict[str, Any]:
    return build_evidence_source_ref(
        sample_design_bundle=bundle,
        population=population,
        partition=partition,
        kind=artifact_ref["kind"],
        ref_id=artifact_ref["ref_id"],
        content_hash=artifact_ref["content_hash"],
    )


def _artifact_ref(value: Mapping[str, Any], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ModelScoreEvidenceError(f"{name} must be an artifact reference")
    _exact_fields(
        value,
        frozenset({"kind", "ref_id", "content_hash"}),
        name,
    )
    try:
        return build_artifact_ref(
            kind=value["kind"],
            ref_id=value["ref_id"],
            content_hash=value["content_hash"],
        )
    except (KeyError, StrategyError, TypeError, ValueError) as exc:
        raise ModelScoreEvidenceError(f"{name} is invalid") from exc


def _normalize_envelope_body(
    value: Mapping[str, Any],
    *,
    sample_design_bundle: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    expected_training_evidence_ref: Mapping[str, Any],
    score_vector: ModelScoreVector | None,
) -> dict[str, Any]:
    obj = _object(value, "model score evidence body")
    expected = _ENVELOPE_FIELDS - {"evidence_id", "content_hash"}
    _exact_fields(obj, expected, "model score evidence body")
    if obj["schema_version"] != MODEL_SCORE_EVIDENCE_SCHEMA_VERSION:
        raise ModelScoreEvidenceError("model score evidence schema_version is invalid")
    if obj["artifact_kind"] != MODEL_SCORE_EVIDENCE_ARTIFACT_KIND:
        raise ModelScoreEvidenceError("model score evidence artifact_kind is invalid")
    bundle = validate_strategy_sample_design_v2_bundle(sample_design_bundle)
    frozen_training = validate_modeling_training_evidence(
        training_evidence,
        sample_design_bundle=bundle,
    )
    task_id = _text(obj["task_id"], "task_id")
    if task_id != frozen_training["task_id"]:
        raise ModelScoreEvidenceError(
            "model score evidence task does not match training evidence"
        )
    if obj["score_product"] != RAW_SCORE_PRODUCT:
        raise ModelScoreEvidenceError("model score evidence score_product is invalid")
    training_ref = _training_input_ref(obj["training_evidence_ref"])
    authenticated_training_ref = _training_input_ref(expected_training_evidence_ref)
    _require_training_ref_matches_evidence(
        training_ref,
        frozen_training,
        expected_reference=authenticated_training_ref,
    )
    model_ref = _artifact_ref(obj["model_ref"], "model_ref")
    expected_model = frozen_training["model_artifact"]["model_binary_ref"]
    if model_ref != {
        "kind": MODEL_BINARY_REF_KIND,
        "ref_id": expected_model["artifact_id"],
        "content_hash": expected_model["content_hash"],
    }:
        raise ModelScoreEvidenceError(
            "model score evidence model_ref does not match training evidence"
        )
    score_ref = _artifact_ref(obj["score_vector_ref"], "score_vector_ref")
    if score_ref["kind"] != MODEL_SCORE_VECTOR_ARTIFACT_KIND:
        raise ModelScoreEvidenceError(
            "score_vector_ref kind must be model_score_vector_parquet"
        )
    expected_sample_binding = _json_copy(frozen_training["sample_design_binding"])
    sample_binding = _json_object_copy(
        obj["sample_design_binding"],
        "sample_design_binding",
    )
    if sample_binding != expected_sample_binding:
        raise ModelScoreEvidenceError(
            "sample_design_binding does not match training evidence"
        )
    row_count = int(bundle["membership"]["row_count"])
    scoring_contract = _scoring_contract(
        obj["scoring_contract"],
        row_count=row_count,
    )
    vector_contract = _score_vector_contract(
        obj["score_vector_contract"],
        row_count=row_count,
        score_ref=score_ref,
        score_vector=score_vector,
    )
    single = validate_single_model_evidence(
        obj["single_model_evidence"],
        sample_design_bundle=bundle,
    )
    if single["model_ref"] != model_ref or single["score_ref"] != score_ref:
        raise ModelScoreEvidenceError(
            "embedded SingleModelEvidence refs do not match outer refs"
        )
    expected_training_source = _sample_source(
        bundle,
        population="risk",
        partition="development",
        artifact_ref={
            "kind": MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
            "ref_id": training_ref["evidence_artifact_id"],
            "content_hash": training_ref["expected_evidence_artifact_content_hash"],
        },
    )
    if single["training_source_ref"] != expected_training_source:
        raise ModelScoreEvidenceError(
            "embedded training source does not match authenticated training artifact"
        )
    resource_budgets = _resource_budgets(
        obj["resource_budgets"],
        row_count=row_count,
        evidence=single,
    )
    governance = _governance_value(obj["governance"])
    return {
        "schema_version": MODEL_SCORE_EVIDENCE_SCHEMA_VERSION,
        "producer_version": _text(obj["producer_version"], "producer_version"),
        "artifact_kind": MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        "task_id": task_id,
        "score_product": RAW_SCORE_PRODUCT,
        "scoring_contract": scoring_contract,
        "training_evidence_ref": training_ref,
        "model_ref": model_ref,
        "sample_design_binding": sample_binding,
        "score_vector_ref": score_ref,
        "score_vector_contract": vector_contract,
        "single_model_evidence": single,
        "resource_budgets": resource_budgets,
        "governance": governance,
    }


def _training_input_ref(value: object) -> dict[str, Any]:
    obj = _object(value, "training_evidence_ref")
    _exact_fields(obj, _TRAINING_REF_FIELDS, "training_evidence_ref")
    sample = _object(
        obj["sample_design_ref"], "training_evidence_ref.sample_design_ref"
    )
    _exact_fields(
        sample,
        _SAMPLE_INPUT_REF_FIELDS,
        "training_evidence_ref.sample_design_ref",
    )
    normalized_sample = {
        "membership_artifact_id": _hash(
            sample["membership_artifact_id"],
            "sample_design_ref.membership_artifact_id",
        ),
        "expected_membership_artifact_content_hash": _hash(
            sample["expected_membership_artifact_content_hash"],
            "sample_design_ref.expected_membership_artifact_content_hash",
        ),
        "bundle_artifact_id": _hash(
            sample["bundle_artifact_id"],
            "sample_design_ref.bundle_artifact_id",
        ),
        "expected_bundle_artifact_content_hash": _hash(
            sample["expected_bundle_artifact_content_hash"],
            "sample_design_ref.expected_bundle_artifact_content_hash",
        ),
        "expected_bundle_id": _text(
            sample["expected_bundle_id"],
            "sample_design_ref.expected_bundle_id",
        ),
        "expected_sample_design_id": _text(
            sample["expected_sample_design_id"],
            "sample_design_ref.expected_sample_design_id",
        ),
        "expected_sample_design_content_hash": _hash(
            sample["expected_sample_design_content_hash"],
            "sample_design_ref.expected_sample_design_content_hash",
        ),
    }
    result: dict[str, Any] = {"sample_design_ref": normalized_sample}
    for field in (
        "model_binary_artifact_id",
        "expected_model_binary_artifact_content_hash",
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "expected_evidence_content_hash",
    ):
        result[field] = _hash(obj[field], f"training_evidence_ref.{field}")
    for field in (
        "expected_experiment_id",
        "expected_model_artifact_id",
        "expected_evidence_id",
    ):
        result[field] = _text(obj[field], f"training_evidence_ref.{field}")
    return result


def _require_training_ref_matches_evidence(
    reference: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    expected_reference: Mapping[str, Any],
) -> None:
    binding = evidence["sample_design_binding"]
    pair = binding["artifact_pair"]
    sample = reference["sample_design_ref"]
    expected_sample = {
        "membership_artifact_id": pair["membership"]["artifact_id"],
        "expected_membership_artifact_content_hash": pair["membership"]["content_hash"],
        "bundle_artifact_id": pair["bundle"]["artifact_id"],
        "expected_bundle_artifact_content_hash": pair["bundle"]["content_hash"],
        "expected_bundle_id": binding["bundle_ref"]["bundle_id"],
        "expected_sample_design_id": binding["sample_design_ref"]["sample_design_id"],
        "expected_sample_design_content_hash": binding["sample_design_ref"][
            "content_hash"
        ],
    }
    model_binary = evidence["model_artifact"]["model_binary_ref"]
    scalar_expected = {
        "evidence_artifact_id": expected_reference["evidence_artifact_id"],
        "model_binary_artifact_id": model_binary["artifact_id"],
        "expected_model_binary_artifact_content_hash": model_binary["content_hash"],
        "expected_evidence_artifact_content_hash": _sha256(
            _canonical_json(evidence).encode("utf-8")
        ),
        "expected_experiment_id": evidence["experiment"]["experiment_id"],
        "expected_model_artifact_id": evidence["model_artifact"]["artifact_id"],
        "expected_evidence_id": evidence["evidence_id"],
        "expected_evidence_content_hash": evidence["content_hash"],
    }
    if sample != expected_sample or any(
        reference[field] != expected_value
        for field, expected_value in scalar_expected.items()
    ):
        raise ModelScoreEvidenceError(
            "training_evidence_ref does not match authenticated evidence"
        )
    if reference["evidence_artifact_id"] in {
        sample["membership_artifact_id"],
        sample["bundle_artifact_id"],
        reference["model_binary_artifact_id"],
    }:
        raise ModelScoreEvidenceError(
            "training evidence TaskArtifact refs must be distinct"
        )


def _scoring_contract(value: object, *, row_count: int) -> dict[str, Any]:
    obj = _object(value, "scoring_contract")
    _exact_fields(obj, _SCORING_CONTRACT_FIELDS, "scoring_contract")
    row_ordinal = _row_ordinal(obj["row_ordinal"], row_count=row_count)
    expected = {
        "input_space": MODEL_SCORE_INPUT_SPACE,
        "load_calibration": False,
        "replay_preprocessing": False,
        "rows_scored_exactly_once": True,
        "row_ordinal": row_ordinal,
        "score_direction": MODEL_SCORE_DIRECTION,
    }
    if obj != expected:
        raise ModelScoreEvidenceError(
            "scoring_contract must prove exact bound active training input space"
        )
    return expected


def _score_vector_contract(
    value: object,
    *,
    row_count: int,
    score_ref: Mapping[str, str],
    score_vector: ModelScoreVector | None,
) -> dict[str, Any]:
    obj = _object(value, "score_vector_contract")
    _exact_fields(obj, _SCORE_VECTOR_CONTRACT_FIELDS, "score_vector_contract")
    normalized = {
        "schema_version": _text(
            obj["schema_version"],
            "score_vector_contract.schema_version",
        ),
        "writer_version": _text(
            obj["writer_version"],
            "score_vector_contract.writer_version",
        ),
        "format": _text(obj["format"], "score_vector_contract.format"),
        "row_count": _non_negative_int(
            obj["row_count"],
            "score_vector_contract.row_count",
        ),
        "row_ordinal": _row_ordinal(
            obj["row_ordinal"],
            row_count=row_count,
        ),
        "score_dtype": _text(
            obj["score_dtype"],
            "score_vector_contract.score_dtype",
        ),
        "score_min": _probability(
            obj["score_min"],
            "score_vector_contract.score_min",
        ),
        "score_max": _probability(
            obj["score_max"],
            "score_vector_contract.score_max",
        ),
        "content_hash": _hash(
            obj["content_hash"],
            "score_vector_contract.content_hash",
        ),
    }
    if (
        normalized["schema_version"] != MODEL_SCORE_VECTOR_SCHEMA_VERSION
        or normalized["writer_version"] != MODEL_SCORE_VECTOR_WRITER_VERSION
        or normalized["format"] != "parquet"
        or normalized["row_count"] != row_count
        or normalized["score_dtype"] != "float64"
        or normalized["score_min"] > normalized["score_max"]
        or normalized["content_hash"] != score_ref["content_hash"]
    ):
        raise ModelScoreEvidenceError("score_vector_contract is inconsistent")
    if score_vector is not None and normalized != {
        "schema_version": MODEL_SCORE_VECTOR_SCHEMA_VERSION,
        "writer_version": MODEL_SCORE_VECTOR_WRITER_VERSION,
        "format": "parquet",
        "row_count": score_vector.row_count,
        "row_ordinal": {
            "start": 0,
            "stop": score_vector.row_count,
            "step": 1,
        },
        "score_dtype": "float64",
        "score_min": score_vector.score_min,
        "score_max": score_vector.score_max,
        "content_hash": score_vector.content_hash,
    }:
        raise ModelScoreEvidenceError(
            "score_vector_contract drifted from live Parquet values"
        )
    return normalized


def _resource_budgets(
    value: object,
    *,
    row_count: int,
    evidence: Mapping[str, Any],
) -> dict[str, int]:
    obj = _object(value, "resource_budgets")
    _exact_fields(obj, _RESOURCE_BUDGET_FIELDS, "resource_budgets")
    periods = {
        item["period"]
        for item in evidence["observations"]
        if item["period"] is not None
    }
    expected = {
        "max_rows": MAX_MODEL_SCORE_VECTOR_ROWS,
        "rows_scored": row_count,
        "max_score_bins": MAX_MODEL_SCORE_BINS,
        "score_bins_used": len(evidence["score_bins"]),
        "max_months": MAX_GOVERNED_SCORE_MONTHS,
        "months_used": len(periods),
        "max_observations": MAX_OBSERVATIONS_PER_EVIDENCE,
        "observations_used": len(evidence["observations"]),
    }
    normalized = {
        field: _non_negative_int(obj[field], f"resource_budgets.{field}")
        for field in sorted(_RESOURCE_BUDGET_FIELDS)
    }
    if normalized != expected:
        raise ModelScoreEvidenceError("resource_budgets drifted from evidence")
    return expected


def _governance() -> dict[str, bool]:
    return {
        "not_compared": True,
        "not_selected": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _governance_value(value: object) -> dict[str, bool]:
    obj = _object(value, "governance")
    _exact_fields(obj, _GOVERNANCE_FIELDS, "governance")
    if obj != _governance():
        raise ModelScoreEvidenceError(
            "model score evidence governance must remain false-state"
        )
    return _governance()


def _row_ordinal(value: object, *, row_count: int) -> dict[str, int]:
    obj = _object(value, "row_ordinal")
    _exact_fields(obj, _ROW_ORDINAL_FIELDS, "row_ordinal")
    normalized = {
        "start": _non_negative_int(obj["start"], "row_ordinal.start"),
        "stop": _non_negative_int(obj["stop"], "row_ordinal.stop"),
        "step": _non_negative_int(obj["step"], "row_ordinal.step"),
    }
    if normalized != {"start": 0, "stop": row_count, "step": 1}:
        raise ModelScoreEvidenceError(
            "row_ordinal must cover every active row exactly once"
        )
    return normalized


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ModelScoreEvidenceError(f"{name} must be an object with string keys")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ModelScoreEvidenceError(
            f"{name} fields are invalid ({'; '.join(details)})"
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ModelScoreEvidenceError(f"{name} must be canonical non-empty text")
    return value


def _hash(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ModelScoreEvidenceError(f"{name} must be lowercase SHA-256")
    return text


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelScoreEvidenceError(f"{name} must be a non-negative integer")
    return value


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ModelScoreEvidenceError(f"{name} must be a probability")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ModelScoreEvidenceError(f"{name} must be a finite probability")
    return number


def _json_object_copy(value: object, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    copied = _json_copy(obj)
    if not isinstance(copied, dict):
        raise ModelScoreEvidenceError(f"{name} must be an object")
    return copied


def _json_copy(value: object) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ModelScoreEvidenceError("value must be strict JSON") from exc


def _object_without_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelScoreEvidenceError(
                "model score evidence JSON contains duplicate keys"
            )
        result[key] = value
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _features(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ModelScoreEvidenceError("features must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip():
            raise ModelScoreEvidenceError("features must contain canonical text")
        result.append(item)
    if not result or len(result) != len(set(result)):
        raise ModelScoreEvidenceError("features must be non-empty and unique")
    return result


def _month(value: object, *, index: int) -> str:
    if _is_missing(value) or isinstance(value, (bool, np.bool_)):
        raise ModelScoreEvidenceError(
            f"governed month value at row {index} is missing or invalid"
        )
    if isinstance(value, (pd.Timestamp, datetime, date)):
        text = value.strftime("%Y-%m")
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif (
        isinstance(value, (float, np.floating))
        and math.isfinite(float(value))
        and float(value).is_integer()
    ):
        text = str(int(value))
    elif isinstance(value, str) and value == value.strip():
        text = value
    else:
        raise ModelScoreEvidenceError(f"governed month value at row {index} is invalid")
    compact = _MONTH_COMPACT_RE.fullmatch(text)
    canonical = _MONTH_CANONICAL_RE.fullmatch(text)
    if compact is not None:
        result = f"{compact.group(1)}-{compact.group(2)}"
    elif canonical is not None:
        result = text
    else:
        raise ModelScoreEvidenceError(f"governed month value at row {index} is invalid")
    year, month = result.split("-", maxsplit=1)
    try:
        date(int(year), int(month), 1)
    except ValueError as exc:
        raise ModelScoreEvidenceError(
            f"governed month value at row {index} is invalid"
        ) from exc
    return result


def _is_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


__all__ = [
    "MAX_GOVERNED_SCORE_MONTHS",
    "MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES",
    "MAX_MODEL_SCORE_BINS",
    "MODEL_SCORE_DIRECTION",
    "MODEL_SCORE_EVIDENCE_ARTIFACT_KIND",
    "MODEL_SCORE_EVIDENCE_PRODUCER_VERSION",
    "MODEL_SCORE_EVIDENCE_SCHEMA_VERSION",
    "MODEL_SCORE_INPUT_SPACE",
    "MODEL_SCORE_VECTOR_ARTIFACT_KIND",
    "ModelScoreEvidenceError",
    "build_model_score_evidence_envelope",
    "build_single_model_score_evidence",
    "canonical_model_score_evidence_json",
    "model_score_evidence_from_json",
    "normalize_governed_months",
    "validate_model_score_evidence_envelope",
]
