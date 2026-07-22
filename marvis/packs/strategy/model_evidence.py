"""Deterministic Strategy ModelEvidence V2 contracts.

The module records evidence produced by existing feature/modeling kernels.  It
never reads a dataset, trains a model, scores a row, or computes a metric.  All
numeric facts are observations bound to a validated StrategySampleDesign V2
bundle, an exact partition, and a content-addressed producer output.

This is a structural/offline contract, not an artifact-registry trust anchor.
Before persistence or governed consumption, the Tool layer must resolve every
source/model/score reference against the same task's allowlisted artifacts and
revalidate the outer artifact hash.  A self-consistent payload alone never
proves that an opaque reference exists or is task-owned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
from types import MappingProxyType
from typing import Any

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_v2 import (
    validate_strategy_sample_design_v2_bundle,
)


STRATEGY_MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION = (
    "strategy.model-evidence-bundle.v2"
)
STRATEGY_MODEL_OBSERVATION_SCHEMA_VERSION = "strategy.model-observation.v2"
STRATEGY_UNIVARIATE_EVIDENCE_SCHEMA_VERSION = "strategy.univariate-evidence.v2"
STRATEGY_SINGLE_MODEL_EVIDENCE_SCHEMA_VERSION = "strategy.single-model-evidence.v2"
STRATEGY_MODEL_COMPARISON_SCHEMA_VERSION = "strategy.model-comparison-evidence.v2"
STRATEGY_MODEL_COMPARISON_METRIC_SCHEMA_VERSION = (
    "strategy.model-comparison-metric.v2"
)
DEFAULT_PRODUCER_VERSION = "marvis.strategy.model-evidence/2"

POPULATIONS = frozenset({"approval", "risk"})
PARTITIONS = frozenset({"development", "validation", "oot"})
OBSERVATION_STATUSES = frozenset(
    {"present", "unavailable", "not_matured", "not_applicable"}
)
BIN_KINDS = frozenset({"interval", "category", "missing", "sentinel", "other"})
MISSING_TREATMENTS = frozenset({"separate_bin", "excluded", "as_regular"})
SENTINEL_TREATMENTS = frozenset(
    {"not_configured", "separate_bin", "treated_as_missing", "as_regular"}
)
SELECTION_STATUSES = frozenset({"selected", "no_selection"})

MAX_MODEL_EVIDENCE_JSON_BYTES = 8 * 1024 * 1024
MAX_MODEL_EVIDENCE_JSON_DEPTH = 32
MAX_MODEL_EVIDENCE_JSON_NODES = 300_000
MAX_UNIVARIATE_EVIDENCE = 500
MAX_MODEL_EVIDENCE = 100
MAX_COMPARISON_EVIDENCE = 100
MAX_BINS_PER_EVIDENCE = 1_000
MAX_OBSERVATIONS_PER_EVIDENCE = 50_000
MAX_FEATURES_PER_MODEL = 10_000

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MONTH_RE = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")
_ID_PATTERNS = {
    "bundle_id": re.compile(r"^strategy-model-evidence-bundle-[0-9a-f]{24}$"),
    "evidence_id": re.compile(
        r"^(strategy-univariate-evidence|strategy-model-evidence)-[0-9a-f]{24}$"
    ),
    "observation_id": re.compile(r"^strategy-model-observation-[0-9a-f]{24}$"),
    "comparison_id": re.compile(r"^strategy-model-comparison-[0-9a-f]{24}$"),
    "comparison_metric_id": re.compile(
        r"^strategy-model-comparison-metric-[0-9a-f]{24}$"
    ),
}

_ARTIFACT_REF_FIELDS = frozenset({"kind", "ref_id", "content_hash"})
_SAMPLE_DESIGN_REF_FIELDS = frozenset({"sample_design_id", "content_hash"})
_SAMPLE_DESIGN_BUNDLE_REF_FIELDS = frozenset({"bundle_id", "content_hash"})
_DATASET_REF_FIELDS = frozenset({"dataset_id", "content_hash", "role"})
_WORKSPACE_REF_FIELDS = frozenset(
    {"revision", "generation", "semantic_mapping_hash"}
)
_BASE_MEMBERSHIP_REF_FIELDS = frozenset({"membership_id", "content_hash"})
_PARTITION_MEMBERSHIP_REF_FIELDS = frozenset(
    {"membership_id", "membership_content_hash", "mask_name"}
)
_SAMPLE_REF_FIELDS = frozenset(
    {
        "sample_design_ref",
        "membership_ref",
        "dataset_ref",
        "workspace_ref",
        "population",
        "partition",
        "row_count",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "sample_design_bundle_ref",
        "sample_design_ref",
        "task_id",
        "dataset_ref",
        "workspace_ref",
        "membership_ref",
    }
)
_SOURCE_REF_FIELDS = frozenset(
    {"kind", "ref_id", "content_hash", *_SAMPLE_REF_FIELDS}
)
_OPERAND_FIELDS = ("value", "numerator", "denominator", "sample_count")
_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "observation_id",
        "observation_kind",
        "metric_key",
        "status",
        *_OPERAND_FIELDS,
        "unit",
        "sample_ref",
        "model_ref",
        "score_ref",
        "feature",
        "bin_id",
        "period",
        "source_ref",
        "reason",
        "content_hash",
    }
)
_UNIVARIATE_BIN_FIELDS = frozenset(
    {
        "ordinal",
        "bin_id",
        "kind",
        "lower_bound",
        "upper_bound",
        "lower_inclusive",
        "upper_inclusive",
        "categories_ref",
        "definition_ref",
    }
)
_SCORE_BIN_FIELDS = frozenset(
    {
        "ordinal",
        "bin_id",
        "lower_bound",
        "upper_bound",
        "lower_inclusive",
        "upper_inclusive",
        "definition_ref",
        "model_ref",
        "score_ref",
    }
)
_UNIVARIATE_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "sample_ref",
        "analysis_ref",
        "analysis_variant",
        "feature",
        "bins",
        "missing_treatment",
        "sentinel_treatment",
        "observations",
        "content_hash",
    }
)
_SINGLE_MODEL_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "training_sample_ref",
        "training_source_ref",
        "model_ref",
        "score_ref",
        "features",
        "score_bins",
        "observations",
        "content_hash",
    }
)
_MODEL_EVIDENCE_REF_FIELDS = frozenset({"evidence_id", "content_hash"})
_MODEL_VALUE_FIELDS = frozenset({"model_evidence_ref", "value"})
_COMPARISON_METRIC_FIELDS = frozenset(
    {
        "schema_version",
        "comparison_metric_id",
        "metric_key",
        "status",
        "unit",
        "evaluation_sample_ref",
        "period",
        "model_values",
        "delta",
        "source_ref",
        "reason",
        "content_hash",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "status",
        "selected_model_evidence_ref",
        "metric_key",
        "period",
        "direction",
        "reason",
    }
)
_COMPARISON_FIELDS = frozenset(
    {
        "schema_version",
        "comparison_id",
        "comparison_ref",
        "evaluation_sample_ref",
        "model_evidence_refs",
        "metrics",
        "selection",
        "content_hash",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "bundle_id",
        "sample_design_binding",
        "sample_refs",
        "univariate_evidence",
        "model_evidence",
        "comparison_evidence",
        "producer_version",
        "content_hash",
    }
)

_POPULATION_ORDER = {"approval": 0, "risk": 1}
_PARTITION_ORDER = {"development": 0, "validation": 1, "oot": 2}


class StrategyModelEvidenceError(StrategyError):
    """ModelEvidence input does not satisfy the exact V2 contract."""


@dataclass(frozen=True)
class _MetricRule:
    units: frozenset[str]
    minimum: float | None
    maximum: float | None
    integer: bool
    period: str
    bin_id: str
    populations: frozenset[str]
    partitions: frozenset[str]
    maturity_sensitive: bool
    requires_binary_classes: bool
    direction: str


def _rule(
    unit: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
    period: str = "forbidden",
    bin_id: str = "forbidden",
    populations: frozenset[str] = POPULATIONS,
    partitions: frozenset[str] = PARTITIONS,
    maturity_sensitive: bool = False,
    requires_binary_classes: bool = False,
    direction: str = "neutral",
) -> _MetricRule:
    return _MetricRule(
        units=frozenset({unit}),
        minimum=minimum,
        maximum=maximum,
        integer=integer,
        period=period,
        bin_id=bin_id,
        populations=populations,
        partitions=partitions,
        maturity_sensitive=maturity_sensitive,
        requires_binary_classes=requires_binary_classes,
        direction=direction,
    )


_RISK = frozenset({"risk"})
_OUTCOME = {
    "populations": _RISK,
    "maturity_sensitive": True,
}
_DISCRIMINATION = {
    **_OUTCOME,
    "requires_binary_classes": True,
}
_UNIVARIATE_RULES = {
    "bin_count": _rule("count", minimum=0, integer=True, bin_id="required"),
    "bin_share": _rule("ratio", minimum=0, maximum=1, bin_id="required"),
    "bin_good_count": _rule(
        "count", minimum=0, integer=True, bin_id="required", **_OUTCOME
    ),
    "bin_bad_count": _rule(
        "count", minimum=0, integer=True, bin_id="required", **_OUTCOME
    ),
    "bin_bad_rate": _rule(
        "ratio", minimum=0, maximum=1, bin_id="required", **_OUTCOME
    ),
    "bin_woe": _rule("number", bin_id="required", **_DISCRIMINATION),
    "bin_iv": _rule("number", minimum=0, bin_id="required", **_DISCRIMINATION),
    "iv": _rule("number", minimum=0, **_DISCRIMINATION),
    "ks": _rule("ratio", minimum=0, maximum=1, direction="higher_is_better", **_DISCRIMINATION),
    "auc": _rule("ratio", minimum=0, maximum=1, direction="higher_is_better", **_DISCRIMINATION),
    "lift": _rule(
        "multiple", minimum=0, bin_id="required", direction="higher_is_better", **_DISCRIMINATION
    ),
    "missing_rate": _rule("ratio", minimum=0, maximum=1),
    "sentinel_rate": _rule("ratio", minimum=0, maximum=1),
    "monthly_bad_rate": _rule(
        "ratio", minimum=0, maximum=1, period="required", **_OUTCOME
    ),
    "monthly_iv": _rule("number", minimum=0, period="required", **_DISCRIMINATION),
    "monthly_ks": _rule(
        "ratio", minimum=0, maximum=1, period="required", direction="higher_is_better", **_DISCRIMINATION
    ),
    "monthly_missing_rate": _rule(
        "ratio", minimum=0, maximum=1, period="required"
    ),
    "monthly_psi": _rule(
        "number", minimum=0, period="required", direction="lower_is_better"
    ),
}
_MODEL_RULES = {
    "auc": _rule("ratio", minimum=0, maximum=1, direction="higher_is_better", **_DISCRIMINATION),
    "ks": _rule("ratio", minimum=0, maximum=1, direction="higher_is_better", **_DISCRIMINATION),
    "lift_head_5": _rule("multiple", minimum=0, direction="higher_is_better", **_DISCRIMINATION),
    "lift_tail_5": _rule("multiple", minimum=0, direction="higher_is_better", **_DISCRIMINATION),
    "lift_head_10": _rule("multiple", minimum=0, direction="higher_is_better", **_DISCRIMINATION),
    "lift_tail_10": _rule("multiple", minimum=0, direction="higher_is_better", **_DISCRIMINATION),
    "calibration_predicted_rate": _rule(
        "ratio", minimum=0, maximum=1, bin_id="required", **_OUTCOME
    ),
    "calibration_observed_rate": _rule(
        "ratio", minimum=0, maximum=1, bin_id="required", **_OUTCOME
    ),
    "calibration_gap": _rule(
        "ratio", minimum=-1, maximum=1, bin_id="required", **_OUTCOME
    ),
    "calibration_abs_gap": _rule(
        "ratio",
        minimum=0,
        maximum=1,
        bin_id="required",
        direction="lower_is_better",
        **_OUTCOME,
    ),
    "score_bin_count": _rule("count", minimum=0, integer=True, bin_id="required"),
    "score_bin_share": _rule("ratio", minimum=0, maximum=1, bin_id="required"),
    "score_bin_bad_rate": _rule(
        "ratio", minimum=0, maximum=1, bin_id="required", **_OUTCOME
    ),
    "score_psi": _rule("number", minimum=0, direction="lower_is_better"),
    "monthly_auc": _rule(
        "ratio", minimum=0, maximum=1, period="required", direction="higher_is_better", **_DISCRIMINATION
    ),
    "monthly_ks": _rule(
        "ratio", minimum=0, maximum=1, period="required", direction="higher_is_better", **_DISCRIMINATION
    ),
    "monthly_lift_head_10": _rule(
        "multiple", minimum=0, period="required", direction="higher_is_better", **_DISCRIMINATION
    ),
    "monthly_psi": _rule(
        "number", minimum=0, period="required", direction="lower_is_better"
    ),
}
_COMPARISON_RULES = {
    key: _MODEL_RULES[key]
    for key in (
        "auc",
        "ks",
        "lift_head_5",
        "lift_head_10",
        "score_psi",
        "monthly_auc",
        "monthly_ks",
        "monthly_lift_head_10",
        "monthly_psi",
    )
}


def _rule_view(rule: _MetricRule) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "units": tuple(sorted(rule.units)),
            "minimum": rule.minimum,
            "maximum": rule.maximum,
            "integer": rule.integer,
            "period": rule.period,
            "bin_id": rule.bin_id,
            "populations": tuple(sorted(rule.populations)),
            "partitions": tuple(sorted(rule.partitions)),
            "maturity_sensitive": rule.maturity_sensitive,
            "requires_binary_classes": rule.requires_binary_classes,
            "direction": rule.direction,
        }
    )


METRIC_SCHEMA_TABLE = MappingProxyType(
    {
        "univariate": MappingProxyType(
            {key: _rule_view(value) for key, value in _UNIVARIATE_RULES.items()}
        ),
        "model": MappingProxyType(
            {key: _rule_view(value) for key, value in _MODEL_RULES.items()}
        ),
        "comparison": MappingProxyType(
            {key: _rule_view(value) for key, value in _COMPARISON_RULES.items()}
        ),
    }
)
UNIVARIATE_METRIC_KEYS = frozenset(_UNIVARIATE_RULES)
MODEL_METRIC_KEYS = frozenset(_MODEL_RULES)
COMPARISON_METRIC_KEYS = frozenset(_COMPARISON_RULES)


def build_artifact_ref(*, kind: str, ref_id: str, content_hash: str) -> dict[str, str]:
    """Build an opaque content-addressed artifact reference (never its facts)."""

    return _artifact_ref(
        {"kind": kind, "ref_id": ref_id, "content_hash": content_hash},
        "artifact_ref",
    )


def sample_partition_refs_from_strategy_sample_design_v2(
    sample_design_bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Derive the sole six legal ModelEvidence sample references."""

    return _sample_design_context(sample_design_bundle)["sample_refs"]


def build_evidence_source_ref(
    *,
    sample_design_bundle: Mapping[str, Any],
    population: str,
    partition: str,
    kind: str,
    ref_id: str,
    content_hash: str,
) -> dict[str, Any]:
    sample_ref = _sample_for(
        _sample_design_context(sample_design_bundle), population, partition
    )
    return {
        "kind": _text(kind, "source_ref.kind"),
        "ref_id": _text(ref_id, "source_ref.ref_id"),
        "content_hash": _hash(content_hash, "source_ref.content_hash"),
        **sample_ref,
    }


def build_univariate_observation(
    *,
    sample_design_bundle: Mapping[str, Any],
    population: str,
    partition: str,
    metric_key: str,
    status: str,
    value: int | float | None,
    numerator: int | float | None,
    denominator: int | float | None,
    sample_count: int | None,
    unit: str,
    source_ref: Mapping[str, Any],
    feature: str,
    bin_id: str | None = None,
    period: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    body = {
        "schema_version": STRATEGY_MODEL_OBSERVATION_SCHEMA_VERSION,
        "observation_kind": "univariate",
        "metric_key": metric_key,
        "status": status,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "sample_count": sample_count,
        "unit": unit,
        "sample_ref": _sample_for(context, population, partition),
        "model_ref": None,
        "score_ref": None,
        "feature": feature,
        "bin_id": bin_id,
        "period": period,
        "source_ref": source_ref,
        "reason": reason,
    }
    return _address(
        _normalize_observation_body(body, context),
        id_field="observation_id",
        prefix="strategy-model-observation-",
    )


def build_model_observation(
    *,
    sample_design_bundle: Mapping[str, Any],
    population: str,
    partition: str,
    metric_key: str,
    status: str,
    value: int | float | None,
    numerator: int | float | None,
    denominator: int | float | None,
    sample_count: int | None,
    unit: str,
    source_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
    bin_id: str | None = None,
    period: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    body = {
        "schema_version": STRATEGY_MODEL_OBSERVATION_SCHEMA_VERSION,
        "observation_kind": "model",
        "metric_key": metric_key,
        "status": status,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "sample_count": sample_count,
        "unit": unit,
        "sample_ref": _sample_for(context, population, partition),
        "model_ref": model_ref,
        "score_ref": score_ref,
        "feature": None,
        "bin_id": bin_id,
        "period": period,
        "source_ref": source_ref,
        "reason": reason,
    }
    return _address(
        _normalize_observation_body(body, context),
        id_field="observation_id",
        prefix="strategy-model-observation-",
    )


def build_univariate_bin_ref(
    *,
    sample_design_bundle: Mapping[str, Any],
    population: str,
    partition: str,
    ordinal: int,
    bin_id: str,
    kind: str,
    definition_ref: Mapping[str, Any],
    lower_bound: int | float | None = None,
    upper_bound: int | float | None = None,
    lower_inclusive: bool | None = None,
    upper_inclusive: bool | None = None,
    categories_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    sample = _sample_for(context, population, partition)
    return _univariate_bin(
        {
            "ordinal": ordinal,
            "bin_id": bin_id,
            "kind": kind,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "lower_inclusive": lower_inclusive,
            "upper_inclusive": upper_inclusive,
            "categories_ref": categories_ref,
            "definition_ref": definition_ref,
        },
        context,
        sample,
        "univariate bin",
    )


def build_score_bin(
    *,
    sample_design_bundle: Mapping[str, Any],
    ordinal: int,
    bin_id: str,
    lower_bound: int | float | None,
    upper_bound: int | float | None,
    lower_inclusive: bool,
    upper_inclusive: bool,
    definition_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    return _score_bin(
        {
            "ordinal": ordinal,
            "bin_id": bin_id,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "lower_inclusive": lower_inclusive,
            "upper_inclusive": upper_inclusive,
            "definition_ref": definition_ref,
            "model_ref": model_ref,
            "score_ref": score_ref,
        },
        context,
        "score bin",
    )


def build_univariate_evidence(
    *,
    sample_design_bundle: Mapping[str, Any],
    population: str,
    partition: str,
    analysis_ref: Mapping[str, Any],
    analysis_variant: str,
    feature: str,
    bins: Sequence[Mapping[str, Any]],
    missing_treatment: str,
    sentinel_treatment: str,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    body = {
        "schema_version": STRATEGY_UNIVARIATE_EVIDENCE_SCHEMA_VERSION,
        "sample_ref": _sample_for(context, population, partition),
        "analysis_ref": analysis_ref,
        "analysis_variant": analysis_variant,
        "feature": feature,
        "bins": bins,
        "missing_treatment": missing_treatment,
        "sentinel_treatment": sentinel_treatment,
        "observations": observations,
    }
    return _address(
        _normalize_univariate_body(body, context),
        id_field="evidence_id",
        prefix="strategy-univariate-evidence-",
    )


def build_single_model_evidence(
    *,
    sample_design_bundle: Mapping[str, Any],
    training_source_ref: Mapping[str, Any],
    model_ref: Mapping[str, Any],
    score_ref: Mapping[str, Any],
    features: Sequence[str],
    score_bins: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    body = {
        "schema_version": STRATEGY_SINGLE_MODEL_EVIDENCE_SCHEMA_VERSION,
        "training_sample_ref": _sample_for(context, "risk", "development"),
        "training_source_ref": training_source_ref,
        "model_ref": model_ref,
        "score_ref": score_ref,
        "features": features,
        "score_bins": score_bins,
        "observations": observations,
    }
    return _address(
        _normalize_single_model_body(body, context),
        id_field="evidence_id",
        prefix="strategy-model-evidence-",
    )


def build_model_evidence_ref(
    evidence: Mapping[str, Any], *, sample_design_bundle: Mapping[str, Any]
) -> dict[str, str]:
    normalized = validate_single_model_evidence(
        evidence, sample_design_bundle=sample_design_bundle
    )
    return {
        "evidence_id": normalized["evidence_id"],
        "content_hash": normalized["content_hash"],
    }


def build_model_comparison_metric(
    *,
    sample_design_bundle: Mapping[str, Any],
    population: str,
    partition: str,
    metric_key: str,
    status: str,
    unit: str,
    source_ref: Mapping[str, Any],
    model_values: Sequence[Mapping[str, Any]] | None,
    delta: int | float | None,
    period: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    body = {
        "schema_version": STRATEGY_MODEL_COMPARISON_METRIC_SCHEMA_VERSION,
        "metric_key": metric_key,
        "status": status,
        "unit": unit,
        "evaluation_sample_ref": _sample_for(context, population, partition),
        "period": period,
        "model_values": model_values,
        "delta": delta,
        "source_ref": source_ref,
        "reason": reason,
    }
    return _address(
        _normalize_comparison_metric_body(body, context),
        id_field="comparison_metric_id",
        prefix="strategy-model-comparison-metric-",
    )


def build_model_selection(
    *,
    status: str,
    selected_model_evidence_ref: Mapping[str, Any] | None = None,
    metric_key: str | None = None,
    period: str | None = None,
    direction: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return _selection(
        {
            "status": status,
            "selected_model_evidence_ref": selected_model_evidence_ref,
            "metric_key": metric_key,
            "period": period,
            "direction": direction,
            "reason": reason,
        }
    )


def build_model_comparison_evidence(
    *,
    sample_design_bundle: Mapping[str, Any],
    population: str,
    partition: str,
    comparison_ref: Mapping[str, Any],
    model_evidence_refs: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    body = {
        "schema_version": STRATEGY_MODEL_COMPARISON_SCHEMA_VERSION,
        "comparison_ref": comparison_ref,
        "evaluation_sample_ref": _sample_for(context, population, partition),
        "model_evidence_refs": model_evidence_refs,
        "metrics": metrics,
        "selection": selection,
    }
    return _address(
        _normalize_comparison_body(body, context),
        id_field="comparison_id",
        prefix="strategy-model-comparison-",
    )


def build_strategy_model_evidence_bundle(
    *,
    sample_design_bundle: Mapping[str, Any],
    univariate_evidence: Sequence[Mapping[str, Any]] = (),
    model_evidence: Sequence[Mapping[str, Any]] = (),
    comparison_evidence: Sequence[Mapping[str, Any]] = (),
    producer_version: str = DEFAULT_PRODUCER_VERSION,
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    body = {
        "schema_version": STRATEGY_MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "sample_design_binding": context["binding"],
        "sample_refs": context["sample_refs"],
        "univariate_evidence": univariate_evidence,
        "model_evidence": model_evidence,
        "comparison_evidence": comparison_evidence,
        "producer_version": producer_version,
    }
    result = _address(
        _normalize_bundle_body(body, context),
        id_field="bundle_id",
        prefix="strategy-model-evidence-bundle-",
    )
    return validate_strategy_model_evidence_bundle(
        result, sample_design_bundle=sample_design_bundle
    )


def validate_model_observation(
    payload: Mapping[str, Any], *, sample_design_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    return _normalize_observation(payload, context, "model observation")


def validate_univariate_evidence(
    payload: Mapping[str, Any], *, sample_design_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    return _normalize_univariate(payload, context, "univariate evidence")


def validate_single_model_evidence(
    payload: Mapping[str, Any], *, sample_design_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    return _normalize_single_model(payload, context, "single model evidence")


def validate_model_comparison_metric(
    payload: Mapping[str, Any], *, sample_design_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    return _normalize_comparison_metric(payload, context, "comparison metric")


def validate_model_comparison_evidence(
    payload: Mapping[str, Any], *, sample_design_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    return _normalize_comparison(payload, context, "model comparison evidence")


def validate_strategy_model_evidence_bundle(
    payload: Mapping[str, Any], *, sample_design_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    context = _sample_design_context(sample_design_bundle)
    obj = _object(payload, "model evidence bundle")
    _preflight_json_tree(obj, name="model evidence bundle")
    _exact_fields(obj, _BUNDLE_FIELDS, "model evidence bundle")
    normalized = _validate_addressed(
        obj,
        _normalize_bundle_body(
            {
                key: obj[key]
                for key in obj
                if key not in {"bundle_id", "content_hash"}
            },
            context,
        ),
        id_field="bundle_id",
        prefix="strategy-model-evidence-bundle-",
        name="model evidence bundle",
    )
    if len(_canonical_json(normalized).encode("utf-8")) > MAX_MODEL_EVIDENCE_JSON_BYTES:
        raise StrategyModelEvidenceError("model evidence bundle exceeds byte budget")
    return normalized


def canonical_strategy_model_evidence_bundle_json(
    payload: Mapping[str, Any], *, sample_design_bundle: Mapping[str, Any]
) -> str:
    return _canonical_json(
        validate_strategy_model_evidence_bundle(
            payload, sample_design_bundle=sample_design_bundle
        )
    )


def strategy_model_evidence_bundle_from_json(
    raw: str | bytes | bytearray, *, sample_design_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, (str, bytes, bytearray)):
        raise StrategyModelEvidenceError("model evidence JSON must be text or bytes")
    byte_length = len(raw if isinstance(raw, (bytes, bytearray)) else raw.encode("utf-8"))
    if byte_length > MAX_MODEL_EVIDENCE_JSON_BYTES:
        raise StrategyModelEvidenceError("model evidence JSON exceeds byte budget")
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except StrategyModelEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StrategyModelEvidenceError("model evidence is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise StrategyModelEvidenceError("model evidence JSON must contain an object")
    return validate_strategy_model_evidence_bundle(
        payload, sample_design_bundle=sample_design_bundle
    )


def _sample_design_context(bundle: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = validate_strategy_sample_design_v2_bundle(bundle)
    except StrategyError as exc:
        raise StrategyModelEvidenceError(
            "sample_design_bundle failed strict StrategySampleDesign V2 validation"
        ) from exc
    design = normalized["sample_design"]
    identity = _object(design.get("identity"), "sample design identity")
    _exact_fields(identity, frozenset({"task_id", "dataset_ref", "workspace_ref"}), "sample design identity")
    dataset_ref = _dataset_ref(identity["dataset_ref"])
    workspace_ref = _workspace_ref(identity["workspace_ref"])
    base_membership = _base_membership_ref(design["membership_ref"])
    header = normalized["membership"]
    if (
        header["membership_id"] != base_membership["membership_id"]
        or header["content_hash"] != base_membership["content_hash"]
        or header["dataset_ref"]["dataset_id"] != dataset_ref["dataset_id"]
        or header["dataset_ref"]["content_hash"] != dataset_ref["content_hash"]
    ):
        raise StrategyModelEvidenceError(
            "sample design identity, membership, and dataset bindings diverge"
        )
    sample_design_ref = {
        "sample_design_id": design["sample_design_id"],
        "content_hash": design["content_hash"],
    }
    binding = {
        "sample_design_bundle_ref": {
            "bundle_id": normalized["bundle_id"],
            "content_hash": normalized["content_hash"],
        },
        "sample_design_ref": sample_design_ref,
        "task_id": identity["task_id"],
        "dataset_ref": dataset_ref,
        "workspace_ref": workspace_ref,
        "membership_ref": base_membership,
    }
    samples = []
    for population in normalized["populations"]:
        for partition in population["partitions"]:
            samples.append(
                {
                    "sample_design_ref": sample_design_ref,
                    "membership_ref": partition["membership_ref"],
                    "dataset_ref": dataset_ref,
                    "workspace_ref": workspace_ref,
                    "population": population["role"],
                    "partition": partition["name"],
                    "row_count": partition["row_count"],
                }
            )
    samples.sort(key=_sample_sort_key)
    if len(samples) != 6:
        raise StrategyModelEvidenceError(
            "StrategySampleDesign V2 must expose exactly six partition refs"
        )
    return {
        "bundle": normalized,
        "binding": binding,
        "sample_refs": samples,
        "sample_statistics": _sample_statistics(normalized),
    }


def _sample_statistics(
    bundle: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    definition_keys = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    result: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for observation in bundle["metric_observations"]:
        partition = observation["partition"]
        if partition not in PARTITIONS:
            continue
        key = (observation["population"], partition)
        metric_key = definition_keys[
            observation["metric_definition_ref"]["metric_definition_id"]
        ]
        result.setdefault(key, {})[metric_key] = observation
    return result


def _sample_for(context: Mapping[str, Any], population: object, partition: object) -> dict[str, Any]:
    population_name = _enum(population, POPULATIONS, "population")
    partition_name = _enum(partition, PARTITIONS, "partition")
    for sample in context["sample_refs"]:
        if sample["population"] == population_name and sample["partition"] == partition_name:
            return _json_copy(sample)
    raise StrategyModelEvidenceError("sample partition is absent from SampleDesign V2")


def _normalize_observation(value: object, context: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _OBSERVATION_FIELDS, name)
    return _validate_addressed(
        obj,
        _normalize_observation_body(
            {key: obj[key] for key in obj if key not in {"observation_id", "content_hash"}},
            context,
        ),
        id_field="observation_id",
        prefix="strategy-model-observation-",
        name=name,
    )


def _normalize_observation_body(value: object, context: Mapping[str, Any]) -> dict[str, Any]:
    obj = _object(value, "observation body")
    expected = _OBSERVATION_FIELDS - {"observation_id", "content_hash"}
    _exact_fields(obj, expected, "observation body")
    if obj["schema_version"] != STRATEGY_MODEL_OBSERVATION_SCHEMA_VERSION:
        raise StrategyModelEvidenceError("observation schema_version is invalid")
    kind = _enum(obj["observation_kind"], frozenset({"univariate", "model"}), "observation.kind")
    rules = _UNIVARIATE_RULES if kind == "univariate" else _MODEL_RULES
    metric_key = _metric_key(obj["metric_key"], rules, f"{kind} metric_key")
    sample = _bound_sample_ref(obj["sample_ref"], context, "observation.sample_ref")
    source = _source_ref(obj["source_ref"], context, "observation.source_ref")
    _require_source_matches_sample(source, sample, "observation.source_ref")
    rule = rules[metric_key]
    status, operands, reason = _status_and_operands(obj, rule, sample, context)
    unit = _metric_unit(obj["unit"], rule, "observation.unit")
    if status == "present":
        _validate_metric_value(operands["value"], rule, "observation.value")
        _validate_observation_cardinality(
            operands,
            rule=rule,
            sample=sample,
        )
        _validate_ratio_operands(operands, unit, "observation")
    bin_id = _dimension_text(obj["bin_id"], rule.bin_id, "observation.bin_id")
    period = _period(obj["period"], rule.period, "observation.period")
    if kind == "univariate":
        model_ref = None
        score_ref = None
        if obj["model_ref"] is not None or obj["score_ref"] is not None:
            raise StrategyModelEvidenceError("univariate observation cannot bind model/score refs")
        feature = _text(obj["feature"], "observation.feature")
    else:
        model_ref = _artifact_ref(obj["model_ref"], "observation.model_ref")
        score_ref = _artifact_ref(obj["score_ref"], "observation.score_ref")
        if obj["feature"] is not None:
            raise StrategyModelEvidenceError("model observation feature must be null")
        feature = None
    return {
        "schema_version": STRATEGY_MODEL_OBSERVATION_SCHEMA_VERSION,
        "observation_kind": kind,
        "metric_key": metric_key,
        "status": status,
        **operands,
        "unit": unit,
        "sample_ref": sample,
        "model_ref": model_ref,
        "score_ref": score_ref,
        "feature": feature,
        "bin_id": bin_id,
        "period": period,
        "source_ref": source,
        "reason": reason,
    }


def _normalize_univariate(value: object, context: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _UNIVARIATE_FIELDS, name)
    return _validate_addressed(
        obj,
        _normalize_univariate_body(
            {key: obj[key] for key in obj if key not in {"evidence_id", "content_hash"}},
            context,
        ),
        id_field="evidence_id",
        prefix="strategy-univariate-evidence-",
        name=name,
    )


def _normalize_univariate_body(value: object, context: Mapping[str, Any]) -> dict[str, Any]:
    obj = _object(value, "univariate evidence body")
    expected = _UNIVARIATE_FIELDS - {"evidence_id", "content_hash"}
    _exact_fields(obj, expected, "univariate evidence body")
    if obj["schema_version"] != STRATEGY_UNIVARIATE_EVIDENCE_SCHEMA_VERSION:
        raise StrategyModelEvidenceError("univariate schema_version is invalid")
    sample = _bound_sample_ref(obj["sample_ref"], context, "univariate.sample_ref")
    analysis_ref = _source_ref(obj["analysis_ref"], context, "univariate.analysis_ref")
    _require_source_matches_sample(analysis_ref, sample, "univariate.analysis_ref")
    analysis_variant = _text(
        obj["analysis_variant"], "univariate.analysis_variant"
    )
    feature = _text(obj["feature"], "univariate.feature")
    bins_raw = _array(obj["bins"], "univariate.bins", required=True)
    if len(bins_raw) > MAX_BINS_PER_EVIDENCE:
        raise StrategyModelEvidenceError("univariate.bins exceeds item budget")
    bins = [_univariate_bin(item, context, sample, f"univariate.bins[{index}]") for index, item in enumerate(bins_raw)]
    _validate_ordinals(bins, "univariate.bins")
    observations = _observations(obj["observations"], context, "univariate.observations")
    bin_ids = {item["bin_id"] for item in bins}
    for observation in observations:
        if observation["observation_kind"] != "univariate":
            raise StrategyModelEvidenceError("univariate evidence contains a model observation")
        if observation["feature"] != feature or observation["sample_ref"] != sample:
            raise StrategyModelEvidenceError("univariate observation context does not match evidence")
        if observation["bin_id"] is not None and observation["bin_id"] not in bin_ids:
            raise StrategyModelEvidenceError("univariate observation bin_id is undeclared")
    return {
        "schema_version": STRATEGY_UNIVARIATE_EVIDENCE_SCHEMA_VERSION,
        "sample_ref": sample,
        "analysis_ref": analysis_ref,
        "analysis_variant": analysis_variant,
        "feature": feature,
        "bins": bins,
        "missing_treatment": _enum(obj["missing_treatment"], MISSING_TREATMENTS, "univariate.missing_treatment"),
        "sentinel_treatment": _enum(obj["sentinel_treatment"], SENTINEL_TREATMENTS, "univariate.sentinel_treatment"),
        "observations": observations,
    }


def _normalize_single_model(value: object, context: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _SINGLE_MODEL_FIELDS, name)
    return _validate_addressed(
        obj,
        _normalize_single_model_body(
            {key: obj[key] for key in obj if key not in {"evidence_id", "content_hash"}},
            context,
        ),
        id_field="evidence_id",
        prefix="strategy-model-evidence-",
        name=name,
    )


def _normalize_single_model_body(value: object, context: Mapping[str, Any]) -> dict[str, Any]:
    obj = _object(value, "single model evidence body")
    expected = _SINGLE_MODEL_FIELDS - {"evidence_id", "content_hash"}
    _exact_fields(obj, expected, "single model evidence body")
    if obj["schema_version"] != STRATEGY_SINGLE_MODEL_EVIDENCE_SCHEMA_VERSION:
        raise StrategyModelEvidenceError("single model schema_version is invalid")
    training_sample = _bound_sample_ref(obj["training_sample_ref"], context, "model.training_sample_ref")
    if (training_sample["population"], training_sample["partition"]) != ("risk", "development"):
        raise StrategyModelEvidenceError("classification model training must bind risk/development")
    if (
        context["bundle"]["populations"][1]["maturity_evidence"]["status"]
        != "confirmed_matured"
    ):
        raise StrategyModelEvidenceError(
            "classification model training requires confirmed_matured risk evidence"
        )
    _require_binary_class_support(
        context,
        training_sample,
        "classification model training",
    )
    training_source = _source_ref(obj["training_source_ref"], context, "model.training_source_ref")
    _require_source_matches_sample(training_source, training_sample, "model.training_source_ref")
    model_ref = _artifact_ref(obj["model_ref"], "model.model_ref")
    score_ref = _artifact_ref(obj["score_ref"], "model.score_ref")
    features = _text_array(obj["features"], "model.features", required=True)
    if len(features) > MAX_FEATURES_PER_MODEL:
        raise StrategyModelEvidenceError("model.features exceeds item budget")
    _reject_duplicates(features, "model.features")
    features.sort()
    score_bins_raw = _array(obj["score_bins"], "model.score_bins", required=True)
    if len(score_bins_raw) > MAX_BINS_PER_EVIDENCE:
        raise StrategyModelEvidenceError("model.score_bins exceeds item budget")
    score_bins = [_score_bin(item, context, f"model.score_bins[{index}]") for index, item in enumerate(score_bins_raw)]
    _validate_score_bins(score_bins, model_ref, score_ref)
    observations = _observations(obj["observations"], context, "model.observations")
    score_bin_ids = {item["bin_id"] for item in score_bins}
    for observation in observations:
        if observation["observation_kind"] != "model":
            raise StrategyModelEvidenceError("model evidence contains a univariate observation")
        if observation["model_ref"] != model_ref or observation["score_ref"] != score_ref:
            raise StrategyModelEvidenceError("model observation cannot be reused across model/score refs")
        if observation["bin_id"] is not None and observation["bin_id"] not in score_bin_ids:
            raise StrategyModelEvidenceError("model observation bin_id is undeclared")
    return {
        "schema_version": STRATEGY_SINGLE_MODEL_EVIDENCE_SCHEMA_VERSION,
        "training_sample_ref": training_sample,
        "training_source_ref": training_source,
        "model_ref": model_ref,
        "score_ref": score_ref,
        "features": features,
        "score_bins": score_bins,
        "observations": observations,
    }


def _normalize_comparison_metric(value: object, context: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _COMPARISON_METRIC_FIELDS, name)
    return _validate_addressed(
        obj,
        _normalize_comparison_metric_body(
            {key: obj[key] for key in obj if key not in {"comparison_metric_id", "content_hash"}},
            context,
        ),
        id_field="comparison_metric_id",
        prefix="strategy-model-comparison-metric-",
        name=name,
    )


def _normalize_comparison_metric_body(value: object, context: Mapping[str, Any]) -> dict[str, Any]:
    obj = _object(value, "comparison metric body")
    expected = _COMPARISON_METRIC_FIELDS - {"comparison_metric_id", "content_hash"}
    _exact_fields(obj, expected, "comparison metric body")
    if obj["schema_version"] != STRATEGY_MODEL_COMPARISON_METRIC_SCHEMA_VERSION:
        raise StrategyModelEvidenceError("comparison metric schema_version is invalid")
    metric_key = _metric_key(obj["metric_key"], _COMPARISON_RULES, "comparison.metric_key")
    rule = _COMPARISON_RULES[metric_key]
    sample = _bound_sample_ref(obj["evaluation_sample_ref"], context, "comparison metric sample_ref")
    source = _source_ref(obj["source_ref"], context, "comparison metric source_ref")
    _require_source_matches_sample(source, sample, "comparison metric source_ref")
    status = _enum(obj["status"], OBSERVATION_STATUSES, "comparison metric.status")
    _validate_metric_applicability(rule, sample, status, context)
    unit = _metric_unit(obj["unit"], rule, "comparison metric.unit")
    period = _period(obj["period"], rule.period, "comparison metric.period")
    if status == "present":
        values_raw = _array(obj["model_values"], "comparison metric.model_values", required=True)
        values = [_model_value(item, rule, f"comparison metric.model_values[{index}]") for index, item in enumerate(values_raw)]
        if len(values) < 2:
            raise StrategyModelEvidenceError("present comparison metric requires at least two model values")
        _reject_duplicates([item["model_evidence_ref"]["evidence_id"] for item in values], "comparison metric.model_values")
        values.sort(key=lambda item: item["model_evidence_ref"]["evidence_id"])
        delta = _finite_number(obj["delta"], "comparison metric.delta")
        if delta < 0 or not math.isclose(delta, max(item["value"] for item in values) - min(item["value"] for item in values), rel_tol=1e-12, abs_tol=1e-12):
            raise StrategyModelEvidenceError("comparison metric.delta must equal max minus min")
        if obj["reason"] is not None:
            raise StrategyModelEvidenceError("present comparison metric.reason must be null")
        reason = None
    else:
        if obj["model_values"] is not None or obj["delta"] is not None:
            raise StrategyModelEvidenceError("non-present comparison values and delta must be null")
        values = None
        delta = None
        reason = _text(obj["reason"], "comparison metric.reason")
    return {
        "schema_version": STRATEGY_MODEL_COMPARISON_METRIC_SCHEMA_VERSION,
        "metric_key": metric_key,
        "status": status,
        "unit": unit,
        "evaluation_sample_ref": sample,
        "period": period,
        "model_values": values,
        "delta": delta,
        "source_ref": source,
        "reason": reason,
    }


def _normalize_comparison(value: object, context: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _COMPARISON_FIELDS, name)
    return _validate_addressed(
        obj,
        _normalize_comparison_body(
            {key: obj[key] for key in obj if key not in {"comparison_id", "content_hash"}},
            context,
        ),
        id_field="comparison_id",
        prefix="strategy-model-comparison-",
        name=name,
    )


def _normalize_comparison_body(value: object, context: Mapping[str, Any]) -> dict[str, Any]:
    obj = _object(value, "comparison evidence body")
    expected = _COMPARISON_FIELDS - {"comparison_id", "content_hash"}
    _exact_fields(obj, expected, "comparison evidence body")
    if obj["schema_version"] != STRATEGY_MODEL_COMPARISON_SCHEMA_VERSION:
        raise StrategyModelEvidenceError("comparison schema_version is invalid")
    sample = _bound_sample_ref(obj["evaluation_sample_ref"], context, "comparison.evaluation_sample_ref")
    refs = [_model_evidence_ref(item, f"comparison.model_evidence_refs[{index}]") for index, item in enumerate(_array(obj["model_evidence_refs"], "comparison.model_evidence_refs", required=True))]
    if len(refs) < 2:
        raise StrategyModelEvidenceError("comparison requires at least two model evidence refs")
    _reject_duplicates([item["evidence_id"] for item in refs], "comparison.model_evidence_refs")
    refs.sort(key=lambda item: item["evidence_id"])
    metrics = [_normalize_comparison_metric(item, context, f"comparison.metrics[{index}]") for index, item in enumerate(_array(obj["metrics"], "comparison.metrics", required=True))]
    _reject_duplicates([(item["metric_key"], item["period"]) for item in metrics], "comparison.metrics")
    for metric in metrics:
        if metric["evaluation_sample_ref"] != sample:
            raise StrategyModelEvidenceError("comparison metric sample does not match comparison")
        if metric["status"] == "present" and { _canonical_json(item["model_evidence_ref"]) for item in metric["model_values"] } != {_canonical_json(item) for item in refs}:
            raise StrategyModelEvidenceError("comparison metric must cover every compared model exactly once")
    metrics.sort(key=lambda item: (item["metric_key"], item["period"] or ""))
    selection = _selection(obj["selection"])
    _validate_selection(selection, refs, metrics)
    return {
        "schema_version": STRATEGY_MODEL_COMPARISON_SCHEMA_VERSION,
        "comparison_ref": _artifact_ref(obj["comparison_ref"], "comparison.comparison_ref"),
        "evaluation_sample_ref": sample,
        "model_evidence_refs": refs,
        "metrics": metrics,
        "selection": selection,
    }


def _normalize_bundle_body(value: object, context: Mapping[str, Any]) -> dict[str, Any]:
    obj = _object(value, "model evidence bundle body")
    expected = _BUNDLE_FIELDS - {"bundle_id", "content_hash"}
    _exact_fields(obj, expected, "model evidence bundle body")
    if obj["schema_version"] != STRATEGY_MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise StrategyModelEvidenceError("bundle schema_version is invalid")
    binding = _binding(obj["sample_design_binding"])
    if binding != context["binding"]:
        raise StrategyModelEvidenceError("sample_design_binding does not match supplied V2 bundle")
    sample_refs = [_bound_sample_ref(item, context, f"sample_refs[{index}]") for index, item in enumerate(_array(obj["sample_refs"], "sample_refs", required=True))]
    if sample_refs != context["sample_refs"]:
        raise StrategyModelEvidenceError("six sample_refs must be derived from supplied V2 bundle")
    univariate_raw = _array(obj["univariate_evidence"], "univariate_evidence", required=False)
    models_raw = _array(obj["model_evidence"], "model_evidence", required=False)
    comparisons_raw = _array(obj["comparison_evidence"], "comparison_evidence", required=False)
    if len(univariate_raw) > MAX_UNIVARIATE_EVIDENCE or len(models_raw) > MAX_MODEL_EVIDENCE or len(comparisons_raw) > MAX_COMPARISON_EVIDENCE:
        raise StrategyModelEvidenceError("model evidence collection exceeds item budget")
    univariate = [_normalize_univariate(item, context, f"univariate_evidence[{index}]") for index, item in enumerate(univariate_raw)]
    models = [_normalize_single_model(item, context, f"model_evidence[{index}]") for index, item in enumerate(models_raw)]
    comparisons = [_normalize_comparison(item, context, f"comparison_evidence[{index}]") for index, item in enumerate(comparisons_raw)]
    _reject_duplicates([item["evidence_id"] for item in univariate], "univariate_evidence")
    _reject_duplicates([item["evidence_id"] for item in models], "model_evidence")
    _reject_duplicates([item["comparison_id"] for item in comparisons], "comparison_evidence")
    model_index = {item["evidence_id"]: item for item in models}
    for comparison in comparisons:
        for ref in comparison["model_evidence_refs"]:
            model = model_index.get(ref["evidence_id"])
            if model is None or model["content_hash"] != ref["content_hash"]:
                raise StrategyModelEvidenceError("comparison model_evidence_ref does not resolve in bundle")
        _reconcile_comparison_metrics(comparison, model_index)
    univariate.sort(
        key=lambda item: (
            *_sample_sort_key(item["sample_ref"]),
            item["feature"],
            item["analysis_variant"],
            item["evidence_id"],
        )
    )
    models.sort(key=lambda item: item["evidence_id"])
    comparisons.sort(key=lambda item: item["comparison_id"])
    return {
        "schema_version": STRATEGY_MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "sample_design_binding": binding,
        "sample_refs": sample_refs,
        "univariate_evidence": univariate,
        "model_evidence": models,
        "comparison_evidence": comparisons,
        "producer_version": _text(obj["producer_version"], "producer_version"),
    }


def _status_and_operands(
    obj: Mapping[str, Any],
    rule: _MetricRule,
    sample: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    status = _enum(obj["status"], OBSERVATION_STATUSES, "observation.status")
    _validate_metric_applicability(rule, sample, status, context)
    if status == "present":
        operands = {
            "value": _finite_number(obj["value"], "observation.value"),
            "numerator": _optional_finite_number(obj["numerator"], "observation.numerator"),
            "denominator": _optional_finite_number(obj["denominator"], "observation.denominator"),
            "sample_count": _optional_non_negative_int(obj["sample_count"], "observation.sample_count"),
        }
        if (operands["numerator"] is None) != (operands["denominator"] is None):
            raise StrategyModelEvidenceError("observation numerator and denominator must be paired")
        if operands["denominator"] is not None and operands["denominator"] <= 0:
            raise StrategyModelEvidenceError("observation denominator must be positive")
        if (
            operands["sample_count"] is not None
            and operands["sample_count"] > sample["row_count"]
        ):
            raise StrategyModelEvidenceError(
                "observation.sample_count exceeds bound sample partition row_count"
            )
        if obj["reason"] is not None:
            raise StrategyModelEvidenceError("present observation.reason must be null")
        reason = None
    else:
        if any(obj[field] is not None for field in _OPERAND_FIELDS):
            raise StrategyModelEvidenceError("non-present observation value and operands must be null")
        operands = {field: None for field in _OPERAND_FIELDS}
        reason = _text(obj["reason"], "non-present observation.reason")
    return status, operands, reason


def _validate_metric_applicability(
    rule: _MetricRule,
    sample: Mapping[str, Any],
    status: str,
    context: Mapping[str, Any],
) -> None:
    is_applicable = (
        sample["population"] in rule.populations
        and sample["partition"] in rule.partitions
    )
    if not is_applicable and status != "not_applicable":
        raise StrategyModelEvidenceError("metric is not applicable to this population/partition")
    if is_applicable and status == "not_applicable":
        raise StrategyModelEvidenceError(
            "applicable metric cannot use not_applicable; use unavailable"
        )
    if status == "not_matured" and (
        not rule.maturity_sensitive
        or sample["population"] != "risk"
        or context["bundle"]["populations"][1]["maturity_evidence"]["status"]
        != "not_matured"
    ):
        raise StrategyModelEvidenceError("not_matured is invalid for this metric/sample")
    maturity_status = context["bundle"]["populations"][1]["maturity_evidence"][
        "status"
    ]
    if (
        rule.maturity_sensitive
        and sample["population"] == "risk"
        and status == "present"
        and maturity_status != "confirmed_matured"
    ):
        raise StrategyModelEvidenceError(
            "present outcome metric requires confirmed_matured sample evidence"
        )
    if status == "present" and sample["row_count"] == 0:
        raise StrategyModelEvidenceError(
            "present metric requires a non-empty bound sample partition"
        )
    if status == "present" and rule.requires_binary_classes:
        _require_binary_class_support(context, sample, "discrimination metric")


def _require_binary_class_support(
    context: Mapping[str, Any],
    sample: Mapping[str, Any],
    name: str,
) -> None:
    statistics = context["sample_statistics"].get(
        (sample["population"], sample["partition"]), {}
    )
    labeled = statistics.get("labeled_count")
    bad = statistics.get("bad_count")
    if (
        labeled is None
        or bad is None
        or labeled["status"] != "present"
        or bad["status"] != "present"
    ):
        raise StrategyModelEvidenceError(
            f"{name} requires present labeled and bad counts"
        )
    labeled_count = int(labeled["value"])
    bad_count = int(bad["value"])
    if bad_count <= 0 or labeled_count - bad_count <= 0:
        raise StrategyModelEvidenceError(
            f"{name} requires both good and bad samples"
        )


def _validate_observation_cardinality(
    operands: Mapping[str, Any],
    *,
    rule: _MetricRule,
    sample: Mapping[str, Any],
) -> None:
    sample_count = operands["sample_count"]
    if sample_count is None or sample_count <= 0:
        raise StrategyModelEvidenceError(
            "present observation requires a positive sample_count"
        )
    if rule.requires_binary_classes and sample_count < 2:
        raise StrategyModelEvidenceError(
            "discrimination metric sample_count must include at least two rows"
        )
    if "count" in rule.units and operands["value"] > sample_count:
        raise StrategyModelEvidenceError(
            "count metric value exceeds observation.sample_count"
        )
    if "count" in rule.units and operands["value"] > sample["row_count"]:
        raise StrategyModelEvidenceError(
            "count metric value exceeds bound sample partition row_count"
        )


def _validate_metric_value(value: int | float, rule: _MetricRule, name: str) -> None:
    if rule.integer and (isinstance(value, bool) or not isinstance(value, int)):
        raise StrategyModelEvidenceError(f"{name} must be an integer")
    if rule.minimum is not None and value < rule.minimum:
        raise StrategyModelEvidenceError(f"{name} is below the metric minimum")
    if rule.maximum is not None and value > rule.maximum:
        raise StrategyModelEvidenceError(f"{name} exceeds the metric maximum")


def _validate_ratio_operands(operands: Mapping[str, Any], unit: str, name: str) -> None:
    numerator = operands["numerator"]
    denominator = operands["denominator"]
    if unit == "ratio" and numerator is not None and not math.isclose(
        float(operands["value"]), numerator / denominator, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise StrategyModelEvidenceError(f"{name} value is inconsistent with operands")


def _univariate_bin(value: object, context: Mapping[str, Any], sample: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _UNIVARIATE_BIN_FIELDS, name)
    kind = _enum(obj["kind"], BIN_KINDS, f"{name}.kind")
    definition = _source_ref(obj["definition_ref"], context, f"{name}.definition_ref")
    _require_source_matches_sample(definition, sample, f"{name}.definition_ref")
    lower = _optional_finite_number(obj["lower_bound"], f"{name}.lower_bound")
    upper = _optional_finite_number(obj["upper_bound"], f"{name}.upper_bound")
    categories_ref = None if obj["categories_ref"] is None else _artifact_ref(obj["categories_ref"], f"{name}.categories_ref")
    if kind == "interval":
        lower_inclusive = _boolean(obj["lower_inclusive"], f"{name}.lower_inclusive")
        upper_inclusive = _boolean(obj["upper_inclusive"], f"{name}.upper_inclusive")
        if lower is not None and upper is not None and lower >= upper:
            raise StrategyModelEvidenceError(f"{name} interval bounds must increase")
        if categories_ref is not None:
            raise StrategyModelEvidenceError(f"{name} interval cannot have categories_ref")
    elif kind == "category":
        if any(item is not None for item in (lower, upper, obj["lower_inclusive"], obj["upper_inclusive"])) or categories_ref is None:
            raise StrategyModelEvidenceError(f"{name} category requires only categories_ref")
        lower_inclusive = upper_inclusive = None
    else:
        if any(item is not None for item in (lower, upper, obj["lower_inclusive"], obj["upper_inclusive"], categories_ref)):
            raise StrategyModelEvidenceError(f"{name} special bin cannot carry bounds/categories")
        lower_inclusive = upper_inclusive = None
    return {
        "ordinal": _non_negative_int(obj["ordinal"], f"{name}.ordinal"),
        "bin_id": _text(obj["bin_id"], f"{name}.bin_id"),
        "kind": kind,
        "lower_bound": lower,
        "upper_bound": upper,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
        "categories_ref": categories_ref,
        "definition_ref": definition,
    }


def _score_bin(value: object, context: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _SCORE_BIN_FIELDS, name)
    source = _source_ref(obj["definition_ref"], context, f"{name}.definition_ref")
    training = _sample_for(context, "risk", "development")
    _require_source_matches_sample(source, training, f"{name}.definition_ref")
    return {
        "ordinal": _non_negative_int(obj["ordinal"], f"{name}.ordinal"),
        "bin_id": _text(obj["bin_id"], f"{name}.bin_id"),
        "lower_bound": _optional_finite_number(obj["lower_bound"], f"{name}.lower_bound"),
        "upper_bound": _optional_finite_number(obj["upper_bound"], f"{name}.upper_bound"),
        "lower_inclusive": _boolean(obj["lower_inclusive"], f"{name}.lower_inclusive"),
        "upper_inclusive": _boolean(obj["upper_inclusive"], f"{name}.upper_inclusive"),
        "definition_ref": source,
        "model_ref": _artifact_ref(obj["model_ref"], f"{name}.model_ref"),
        "score_ref": _artifact_ref(obj["score_ref"], f"{name}.score_ref"),
    }


def _validate_score_bins(bins: list[dict[str, Any]], model_ref: Mapping[str, Any], score_ref: Mapping[str, Any]) -> None:
    _validate_ordinals(bins, "model.score_bins")
    if bins[0]["lower_bound"] is not None or bins[-1]["upper_bound"] is not None:
        raise StrategyModelEvidenceError("score bins must cover unbounded tails")
    for index, item in enumerate(bins):
        if item["model_ref"] != model_ref or item["score_ref"] != score_ref:
            raise StrategyModelEvidenceError("score bin model/score refs do not match model evidence")
        if item["lower_bound"] is None and item["lower_inclusive"]:
            raise StrategyModelEvidenceError("unbounded score lower edge cannot be inclusive")
        if item["lower_bound"] is not None and not item["lower_inclusive"]:
            raise StrategyModelEvidenceError("score bins use canonical [lower, upper) bounds")
        if item["upper_inclusive"]:
            raise StrategyModelEvidenceError("score bins use canonical [lower, upper) bounds")
        if item["lower_bound"] is not None and item["upper_bound"] is not None and item["lower_bound"] >= item["upper_bound"]:
            raise StrategyModelEvidenceError("score bin bounds must increase")
        if index and bins[index - 1]["upper_bound"] != item["lower_bound"]:
            raise StrategyModelEvidenceError("score bins must be ordered and contiguous")


def _validate_ordinals(items: list[Mapping[str, Any]], name: str) -> None:
    if [item["ordinal"] for item in items] != list(range(len(items))):
        raise StrategyModelEvidenceError(f"{name} ordinals must be explicit, unique, and consecutive")
    _reject_duplicates([item["bin_id"] for item in items], name)


def _observations(value: object, context: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    raw = _array(value, name, required=False)
    if len(raw) > MAX_OBSERVATIONS_PER_EVIDENCE:
        raise StrategyModelEvidenceError(f"{name} exceeds item budget")
    result = [_normalize_observation(item, context, f"{name}[{index}]") for index, item in enumerate(raw)]
    _reject_duplicates(
        [
            (
                item["observation_kind"],
                item["metric_key"],
                item["sample_ref"]["population"],
                item["sample_ref"]["partition"],
                item["model_ref"],
                item["score_ref"],
                item["feature"],
                item["bin_id"],
                item["period"],
            )
            for item in result
        ],
        name,
    )
    result.sort(key=_observation_sort_key)
    return result


def _selection(value: object) -> dict[str, Any]:
    obj = _object(value, "selection")
    _exact_fields(obj, _SELECTION_FIELDS, "selection")
    status = _enum(obj["status"], SELECTION_STATUSES, "selection.status")
    if status == "selected":
        ref = _model_evidence_ref(obj["selected_model_evidence_ref"], "selection.selected_model_evidence_ref")
        metric_key = _metric_key(obj["metric_key"], _COMPARISON_RULES, "selection.metric_key")
        period = _period(
            obj["period"],
            _COMPARISON_RULES[metric_key].period,
            "selection.period",
        )
        direction = _enum(obj["direction"], frozenset({"higher_is_better", "lower_is_better"}), "selection.direction")
        if obj["reason"] is not None:
            raise StrategyModelEvidenceError("selected selection.reason must be null")
        reason = None
    else:
        if any(
            obj[field] is not None
            for field in (
                "selected_model_evidence_ref",
                "metric_key",
                "period",
                "direction",
            )
        ):
            raise StrategyModelEvidenceError("no_selection must not identify a model or metric")
        ref = metric_key = period = direction = None
        reason = _text(obj["reason"], "selection.reason")
    return {
        "status": status,
        "selected_model_evidence_ref": ref,
        "metric_key": metric_key,
        "period": period,
        "direction": direction,
        "reason": reason,
    }


def _reconcile_comparison_metrics(
    comparison: Mapping[str, Any],
    model_index: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require every compared value to resolve to one exact model observation."""

    for metric in comparison["metrics"]:
        if metric["status"] != "present":
            continue
        for model_value in metric["model_values"]:
            ref = model_value["model_evidence_ref"]
            model = model_index[ref["evidence_id"]]
            matches = [
                observation
                for observation in model["observations"]
                if observation["status"] == "present"
                and observation["metric_key"] == metric["metric_key"]
                and observation["sample_ref"] == metric["evaluation_sample_ref"]
                and observation["period"] == metric["period"]
                and observation["bin_id"] is None
            ]
            if len(matches) != 1:
                raise StrategyModelEvidenceError(
                    "comparison value must resolve to exactly one present model "
                    "observation"
                )
            observation = matches[0]
            if observation["unit"] != metric["unit"] or not math.isclose(
                float(model_value["value"]),
                float(observation["value"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise StrategyModelEvidenceError(
                    "comparison value does not match the bound model observation"
                )


def _validate_selection(selection: Mapping[str, Any], refs: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> None:
    if selection["status"] != "selected":
        return
    selected_ref = selection["selected_model_evidence_ref"]
    if selected_ref not in refs:
        raise StrategyModelEvidenceError("selected model is not in comparison refs")
    candidates = [
        item
        for item in metrics
        if item["metric_key"] == selection["metric_key"]
        and item["period"] == selection["period"]
        and item["status"] == "present"
    ]
    if len(candidates) != 1:
        raise StrategyModelEvidenceError("selection metric must resolve to one present comparison metric")
    metric = candidates[0]
    rule = _COMPARISON_RULES[metric["metric_key"]]
    if rule.direction != selection["direction"]:
        raise StrategyModelEvidenceError("selection direction contradicts metric schema")
    values = {item["model_evidence_ref"]["evidence_id"]: item["value"] for item in metric["model_values"]}
    selected_value = values[selected_ref["evidence_id"]]
    best = max(values.values()) if rule.direction == "higher_is_better" else min(values.values())
    if not math.isclose(selected_value, best, rel_tol=1e-12, abs_tol=1e-12):
        raise StrategyModelEvidenceError("selected model is not best under the typed comparison metric")


def _model_value(value: object, rule: _MetricRule, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _MODEL_VALUE_FIELDS, name)
    number = _finite_number(obj["value"], f"{name}.value")
    _validate_metric_value(number, rule, f"{name}.value")
    return {"model_evidence_ref": _model_evidence_ref(obj["model_evidence_ref"], f"{name}.model_evidence_ref"), "value": number}


def _binding(value: object) -> dict[str, Any]:
    obj = _object(value, "sample_design_binding")
    _exact_fields(obj, _BINDING_FIELDS, "sample_design_binding")
    return {
        "sample_design_bundle_ref": _sample_design_bundle_ref(obj["sample_design_bundle_ref"]),
        "sample_design_ref": _sample_design_ref(obj["sample_design_ref"]),
        "task_id": _text(obj["task_id"], "sample_design_binding.task_id"),
        "dataset_ref": _dataset_ref(obj["dataset_ref"]),
        "workspace_ref": _workspace_ref(obj["workspace_ref"]),
        "membership_ref": _base_membership_ref(obj["membership_ref"]),
    }


def _bound_sample_ref(value: object, context: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _SAMPLE_REF_FIELDS, name)
    normalized = {
        "sample_design_ref": _sample_design_ref(obj["sample_design_ref"]),
        "membership_ref": _partition_membership_ref(obj["membership_ref"]),
        "dataset_ref": _dataset_ref(obj["dataset_ref"]),
        "workspace_ref": _workspace_ref(obj["workspace_ref"]),
        "population": _enum(obj["population"], POPULATIONS, f"{name}.population"),
        "partition": _enum(obj["partition"], PARTITIONS, f"{name}.partition"),
        "row_count": _non_negative_int(obj["row_count"], f"{name}.row_count"),
    }
    if normalized != _sample_for(context, normalized["population"], normalized["partition"]):
        raise StrategyModelEvidenceError(f"{name} is not derived from supplied StrategySampleDesign V2")
    return normalized


def _source_ref(value: object, context: Mapping[str, Any], name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _SOURCE_REF_FIELDS, name)
    sample = _bound_sample_ref({key: obj[key] for key in _SAMPLE_REF_FIELDS}, context, name)
    return {
        "kind": _text(obj["kind"], f"{name}.kind"),
        "ref_id": _text(obj["ref_id"], f"{name}.ref_id"),
        "content_hash": _hash(obj["content_hash"], f"{name}.content_hash"),
        **sample,
    }


def _require_source_matches_sample(source: Mapping[str, Any], sample: Mapping[str, Any], name: str) -> None:
    if any(source[key] != sample[key] for key in _SAMPLE_REF_FIELDS):
        raise StrategyModelEvidenceError(f"{name} does not match exact sample binding; development evidence cannot masquerade as validation or OOT")


def _artifact_ref(value: object, name: str) -> dict[str, str]:
    obj = _object(value, name)
    _exact_fields(obj, _ARTIFACT_REF_FIELDS, name)
    return {"kind": _text(obj["kind"], f"{name}.kind"), "ref_id": _text(obj["ref_id"], f"{name}.ref_id"), "content_hash": _hash(obj["content_hash"], f"{name}.content_hash")}


def _sample_design_ref(value: object) -> dict[str, str]:
    obj = _object(value, "sample_design_ref")
    _exact_fields(obj, _SAMPLE_DESIGN_REF_FIELDS, "sample_design_ref")
    return {"sample_design_id": _text(obj["sample_design_id"], "sample_design_ref.sample_design_id"), "content_hash": _hash(obj["content_hash"], "sample_design_ref.content_hash")}


def _sample_design_bundle_ref(value: object) -> dict[str, str]:
    obj = _object(value, "sample_design_bundle_ref")
    _exact_fields(obj, _SAMPLE_DESIGN_BUNDLE_REF_FIELDS, "sample_design_bundle_ref")
    return {"bundle_id": _text(obj["bundle_id"], "sample_design_bundle_ref.bundle_id"), "content_hash": _hash(obj["content_hash"], "sample_design_bundle_ref.content_hash")}


def _dataset_ref(value: object) -> dict[str, str]:
    obj = _object(value, "dataset_ref")
    _exact_fields(obj, _DATASET_REF_FIELDS, "dataset_ref")
    role = _text(obj["role"], "dataset_ref.role")
    if role != "active":
        raise StrategyModelEvidenceError("dataset_ref.role must be active")
    return {"dataset_id": _text(obj["dataset_id"], "dataset_ref.dataset_id"), "content_hash": _hash(obj["content_hash"], "dataset_ref.content_hash"), "role": role}


def _workspace_ref(value: object) -> dict[str, Any]:
    obj = _object(value, "workspace_ref")
    _exact_fields(obj, _WORKSPACE_REF_FIELDS, "workspace_ref")
    return {"revision": _non_negative_int(obj["revision"], "workspace_ref.revision"), "generation": _non_negative_int(obj["generation"], "workspace_ref.generation"), "semantic_mapping_hash": _hash(obj["semantic_mapping_hash"], "workspace_ref.semantic_mapping_hash")}


def _base_membership_ref(value: object) -> dict[str, str]:
    obj = _object(value, "membership_ref")
    _exact_fields(obj, _BASE_MEMBERSHIP_REF_FIELDS, "membership_ref")
    return {"membership_id": _text(obj["membership_id"], "membership_ref.membership_id"), "content_hash": _hash(obj["content_hash"], "membership_ref.content_hash")}


def _partition_membership_ref(value: object) -> dict[str, str]:
    obj = _object(value, "partition membership_ref")
    _exact_fields(obj, _PARTITION_MEMBERSHIP_REF_FIELDS, "partition membership_ref")
    mask_name = _text(obj["mask_name"], "partition membership_ref.mask_name")
    if mask_name not in {f"{population}/{partition}" for population in POPULATIONS for partition in PARTITIONS}:
        raise StrategyModelEvidenceError("partition membership_ref.mask_name is invalid")
    return {"membership_id": _text(obj["membership_id"], "partition membership_ref.membership_id"), "membership_content_hash": _hash(obj["membership_content_hash"], "partition membership_ref.membership_content_hash"), "mask_name": mask_name}


def _model_evidence_ref(value: object, name: str) -> dict[str, str]:
    obj = _object(value, name)
    _exact_fields(obj, _MODEL_EVIDENCE_REF_FIELDS, name)
    return {"evidence_id": _text(obj["evidence_id"], f"{name}.evidence_id"), "content_hash": _hash(obj["content_hash"], f"{name}.content_hash")}


def _metric_key(value: object, rules: Mapping[str, _MetricRule], name: str) -> str:
    key = _text(value, name)
    if key not in rules:
        raise StrategyModelEvidenceError(f"unsupported {name}: {key}")
    return key


def _metric_unit(value: object, rule: _MetricRule, name: str) -> str:
    return _enum(value, rule.units, name)


def _dimension_text(value: object, mode: str, name: str) -> str | None:
    if mode == "required":
        return _text(value, name)
    if value is not None:
        raise StrategyModelEvidenceError(f"{name} is forbidden for this metric")
    return None


def _period(value: object, mode: str, name: str) -> str | None:
    result = _dimension_text(value, mode, name)
    if result is not None and not _MONTH_RE.fullmatch(result):
        raise StrategyModelEvidenceError(f"{name} must be canonical YYYY-MM")
    return result


def _address(body: Mapping[str, Any], *, id_field: str, prefix: str) -> dict[str, Any]:
    object_id = prefix + _sha256(_canonical_json(body))[:24]
    without_hash = {**body, id_field: object_id}
    return {**without_hash, "content_hash": _sha256(_canonical_json(without_hash))}


def _validate_addressed(original: Mapping[str, Any], body: Mapping[str, Any], *, id_field: str, prefix: str, name: str) -> dict[str, Any]:
    expected_id = prefix + _sha256(_canonical_json(body))[:24]
    supplied_id = _text(original[id_field], f"{name}.{id_field}")
    pattern = _ID_PATTERNS[id_field]
    if not pattern.fullmatch(supplied_id) or supplied_id != expected_id:
        raise StrategyModelEvidenceError(f"{name} {id_field} does not match canonical content")
    without_hash = {**body, id_field: supplied_id}
    supplied_hash = _hash(original["content_hash"], f"{name}.content_hash")
    if not hmac.compare_digest(supplied_hash, _sha256(_canonical_json(without_hash))):
        raise StrategyModelEvidenceError(f"{name} content_hash does not match content")
    return {**without_hash, "content_hash": supplied_hash}


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StrategyModelEvidenceError(f"{name} must be an object with string keys")
    return value


def _array(value: object, name: str, *, required: bool) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StrategyModelEvidenceError(f"{name} must be an array")
    result = list(value)
    if required and not result:
        raise StrategyModelEvidenceError(f"{name} must not be empty")
    return result


def _text_array(value: object, name: str, *, required: bool) -> list[str]:
    return [_text(item, f"{name}[{index}]") for index, item in enumerate(_array(value, name, required=required))]


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if unknown:
            parts.append("unknown: " + ", ".join(unknown))
        raise StrategyModelEvidenceError(f"{name} fields are invalid ({'; '.join(parts)})")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StrategyModelEvidenceError(f"{name} must be non-empty canonical text")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise StrategyModelEvidenceError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _enum(value: object, allowed: frozenset[str], name: str) -> str:
    if value not in allowed:
        raise StrategyModelEvidenceError(f"{name} must be one of {', '.join(sorted(allowed))}")
    assert isinstance(value, str)
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise StrategyModelEvidenceError(f"{name} must be boolean")
    return value


def _finite_number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StrategyModelEvidenceError(f"{name} must be a finite number")
    return value


def _optional_finite_number(value: object, name: str) -> int | float | None:
    return None if value is None else _finite_number(value, name)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyModelEvidenceError(f"{name} must be a non-negative integer")
    return value


def _optional_non_negative_int(value: object, name: str) -> int | None:
    return None if value is None else _non_negative_int(value, name)


def _reject_duplicates(values: Sequence[object], name: str) -> None:
    identities = [_canonical_json(value) for value in values]
    if len(set(identities)) != len(identities):
        raise StrategyModelEvidenceError(f"{name} contains duplicates")


def _sample_sort_key(value: Mapping[str, Any]) -> tuple[int, int]:
    return (_POPULATION_ORDER[value["population"]], _PARTITION_ORDER[value["partition"]])


def _observation_sort_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    sample = value["sample_ref"]
    return (_POPULATION_ORDER[sample["population"]], _PARTITION_ORDER[sample["partition"]], value["metric_key"], value["feature"] or "", value["bin_id"] or "", value["period"] or "")


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _preflight_json_tree(value: object, *, name: str) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    active: set[int] = set()
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_MODEL_EVIDENCE_JSON_NODES:
            raise StrategyModelEvidenceError(f"{name} exceeds node budget")
        if depth > MAX_MODEL_EVIDENCE_JSON_DEPTH:
            raise StrategyModelEvidenceError(f"{name} exceeds depth budget")
        if current is None or isinstance(current, (str, bool)):
            continue
        if isinstance(current, (int, float)):
            if isinstance(current, float) and not math.isfinite(current):
                raise StrategyModelEvidenceError(f"{name} contains non-finite number")
            continue
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in active:
                raise StrategyModelEvidenceError(f"{name} contains a cycle")
            if any(not isinstance(key, str) for key in current):
                raise StrategyModelEvidenceError(f"{name} keys must be strings")
            active.add(identity)
            stack.append((_Leave(identity), depth))
            stack.extend((child, depth + 1) for child in current.values())
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            identity = id(current)
            if identity in active:
                raise StrategyModelEvidenceError(f"{name} contains a cycle")
            active.add(identity)
            stack.append((_Leave(identity), depth))
            stack.extend((child, depth + 1) for child in current)
            continue
        if isinstance(current, _Leave):
            active.discard(current.identity)
            continue
        raise StrategyModelEvidenceError(f"{name} contains unsupported {type(current).__name__}")


@dataclass(frozen=True)
class _Leave:
    identity: int


def _canonical_json(value: object) -> str:
    _preflight_json_tree(value, name="canonical JSON")
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyModelEvidenceError("value is not finite canonical JSON") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyModelEvidenceError(f"model evidence JSON has duplicate key: {key}")
        result[key] = value
    return result


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "COMPARISON_METRIC_KEYS",
    "DEFAULT_PRODUCER_VERSION",
    "METRIC_SCHEMA_TABLE",
    "MODEL_METRIC_KEYS",
    "OBSERVATION_STATUSES",
    "PARTITIONS",
    "POPULATIONS",
    "STRATEGY_MODEL_COMPARISON_METRIC_SCHEMA_VERSION",
    "STRATEGY_MODEL_COMPARISON_SCHEMA_VERSION",
    "STRATEGY_MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "STRATEGY_MODEL_OBSERVATION_SCHEMA_VERSION",
    "STRATEGY_SINGLE_MODEL_EVIDENCE_SCHEMA_VERSION",
    "STRATEGY_UNIVARIATE_EVIDENCE_SCHEMA_VERSION",
    "UNIVARIATE_METRIC_KEYS",
    "StrategyModelEvidenceError",
    "build_artifact_ref",
    "build_evidence_source_ref",
    "build_model_comparison_evidence",
    "build_model_comparison_metric",
    "build_model_evidence_ref",
    "build_model_observation",
    "build_model_selection",
    "build_score_bin",
    "build_single_model_evidence",
    "build_strategy_model_evidence_bundle",
    "build_univariate_bin_ref",
    "build_univariate_evidence",
    "build_univariate_observation",
    "canonical_strategy_model_evidence_bundle_json",
    "sample_partition_refs_from_strategy_sample_design_v2",
    "strategy_model_evidence_bundle_from_json",
    "validate_model_comparison_evidence",
    "validate_model_comparison_metric",
    "validate_model_observation",
    "validate_single_model_evidence",
    "validate_strategy_model_evidence_bundle",
    "validate_univariate_evidence",
]
