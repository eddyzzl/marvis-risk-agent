"""Pure, content-addressed StrategySampleDesign V2 contracts.

V2 freezes two distinct populations over one exact analysis universe:
``approval`` and ``risk``.  Each population owns development, validation, and
OOT membership references.  This module accepts only an authenticated
membership header, already-resolved boolean masks, and already-computed typed
statistics.  It owns no database, DataFrame filtering, Tool, or Agent runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import hashlib
import hmac
import json
import math
import re
from typing import Any

import numpy as np

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef
from marvis.packs.strategy.sample_membership import (
    MEMBERSHIP_MASK_ORDER,
    SAMPLE_MEMBERSHIP_SCHEMA_VERSION,
    validate_sample_membership_header,
)


STRATEGY_SAMPLE_DESIGN_V2_BUNDLE_SCHEMA_VERSION = (
    "strategy.sample-design-bundle.v2"
)
STRATEGY_SAMPLE_DESIGN_V2_SCHEMA_VERSION = "strategy.sample-design.v2"
STRATEGY_SAMPLE_POPULATION_V2_SCHEMA_VERSION = "strategy.sample-population.v2"
STRATEGY_HISTORICAL_SCORE_V2_SCHEMA_VERSION = "strategy.historical-score.v2"
STRATEGY_SAMPLE_DIAGNOSTIC_V2_SCHEMA_VERSION = "strategy.sample-diagnostic.v2"
STRATEGY_SAMPLE_POLICY_V2_SCHEMA_VERSION = "strategy.sample-policy.v2"
STRATEGY_METRIC_DEFINITION_V2_SCHEMA_VERSION = "strategy.metric-definition.v2"
STRATEGY_METRIC_OBSERVATION_V2_SCHEMA_VERSION = "strategy.metric-observation.v2"
STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION = "marvis.strategy.sample-design/2"

SAMPLE_RELATIONSHIPS = frozenset(
    {"nested_same_cohort", "parallel_time_cohorts"}
)
POPULATION_ROLES = ("approval", "risk")
PARTITION_NAMES = ("development", "validation", "oot")
DIAGNOSTIC_STATUSES = frozenset(
    {"pass", "warn", "fail", "unavailable", "not_applicable"}
)
DIAGNOSTIC_CODES = (
    "target_selector",
    "entity_overlap",
    "temporal_oot",
    "risk_outside_approval",
    "maturity",
    "label_coverage",
    "historical_score_coverage",
    "group_coverage_gap",
    "sufficiency",
)
DIAGNOSTIC_CATEGORIES = frozenset(
    {"contract", "leakage", "maturity", "coverage", "bias", "sufficiency"}
)
HISTORICAL_SCORE_STATUSES = frozenset(
    {"available", "unavailable", "not_applicable"}
)
HISTORICAL_SCORE_DIRECTIONS = frozenset(
    {"higher_is_riskier", "lower_is_riskier"}
)
METRIC_OBSERVATION_V2_STATUSES = frozenset(
    {
        "present",
        "unavailable",
        "not_applicable",
        "not_matured",
        "insufficient_data",
    }
)
METRIC_OBSERVATION_V2_UNITS = frozenset(
    {"count", "ratio", "score", "days", "boolean"}
)

MAX_SAMPLE_DESIGN_V2_JSON_BYTES = 16 * 1024 * 1024
MAX_SAMPLE_DESIGN_V2_JSON_DEPTH = 32
MAX_SAMPLE_DESIGN_V2_JSON_NODES = 100_000

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_PATTERNS = {
    "bundle_id": re.compile(r"^strategy-sample-design-bundle-[0-9a-f]{24}$"),
    "sample_design_id": re.compile(r"^strategy-sample-design-[0-9a-f]{24}$"),
    "population_id": re.compile(r"^strategy-sample-population-[0-9a-f]{24}$"),
    "historical_score_id": re.compile(r"^strategy-historical-score-[0-9a-f]{24}$"),
    "policy_id": re.compile(r"^strategy-sample-policy-[0-9a-f]{24}$"),
    "metric_definition_id": re.compile(r"^metric-definition-[0-9a-f]{24}$"),
    "diagnostic_id": re.compile(r"^strategy-sample-diagnostic-[0-9a-f]{24}$"),
    "observation_id": re.compile(r"^metric-observation-[0-9a-f]{24}$"),
}
_ID_PREFIXES = {
    "bundle_id": "strategy-sample-design-bundle-",
    "sample_design_id": "strategy-sample-design-",
    "population_id": "strategy-sample-population-",
    "historical_score_id": "strategy-historical-score-",
    "policy_id": "strategy-sample-policy-",
    "metric_definition_id": "metric-definition-",
    "diagnostic_id": "strategy-sample-diagnostic-",
    "observation_id": "metric-observation-",
}

_SOURCE_REF_FIELDS = frozenset({"kind", "ref_id", "content_hash"})
_DATASET_REF_FIELDS = frozenset({"dataset_id", "content_hash"})
_EXACT_DATASET_REF_FIELDS = frozenset({"dataset_id", "content_hash", "role"})
_WORKSPACE_REF_FIELDS = frozenset(
    {"revision", "generation", "semantic_mapping_hash"}
)
_IDENTITY_FIELDS = frozenset({"task_id", "dataset_ref", "workspace_ref"})
_MEMBERSHIP_REF_FIELDS = frozenset({"membership_id", "content_hash"})
_PARTITION_MEMBERSHIP_REF_FIELDS = frozenset(
    {"membership_id", "membership_content_hash", "mask_name"}
)
_PARTITION_FIELDS = frozenset({"name", "membership_ref", "row_count"})
_MATURITY_FIELDS = frozenset(
    {
        "status",
        "performance_window_days",
        "cutoff_date",
        "eligible_count",
        "labeled_count",
        "source_refs",
        "reason",
    }
)
_POPULATION_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "population_id",
        "role",
        "membership_ref",
        "inclusion_predicate_ref",
        "exclusion_predicate_ref",
        "partitions",
        "total_count",
        "maturity_evidence",
        "source_refs",
        "content_hash",
    }
)
_TARGET_SELECTOR_FIELDS = frozenset(
    {
        "status",
        "column",
        "good_value",
        "bad_value",
        "drop_missing",
        "source_refs",
        "reason",
    }
)
_HISTORICAL_SCORE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "historical_score_id",
        "status",
        "column",
        "direction",
        "source_refs",
        "reason",
        "content_hash",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "policy_id",
        "minimum_partition_count",
        "minimum_bad_count",
        "minimum_label_coverage",
        "minimum_historical_score_coverage",
        "maximum_group_coverage_gap",
        "diagnostic_severities",
        "content_hash",
    }
)
_SEVERITY_FIELDS = frozenset(
    {
        "entity_overlap",
        "temporal_oot",
        "risk_outside_approval",
        "maturity",
        "label_coverage",
        "historical_score_coverage",
        "group_coverage_gap",
        "sufficiency",
    }
)
_ANALYSIS_UNIVERSE_FIELDS = frozenset(
    {"dataset_ref", "row_count", "row_ordinal"}
)
_ROW_ORDINAL_FIELDS = frozenset({"start", "stop", "step"})
_POPULATION_REF_FIELDS = frozenset({"role", "population_id", "content_hash"})
_HISTORICAL_SCORE_REF_FIELDS = frozenset(
    {"historical_score_id", "content_hash"}
)
_POLICY_REF_FIELDS = frozenset({"policy_id", "content_hash"})
_COMPATIBILITY_FIELDS = frozenset({"legacy_development_ref", "maps_to"})
_FIELD_BINDINGS_FIELDS = frozenset(
    {
        "entity_field",
        "time_field",
        "group_field",
        "month_field",
        "weight_field",
        "loan_amount_field",
        "overdue_amount_field",
    }
)
_PERFORMANCE_WINDOW_FIELDS = frozenset({"status", "days"})
_OBSERVATION_WINDOW_FIELDS = frozenset({"status", "start", "end"})
_SPLIT_DEFINITION_FIELDS = frozenset(
    {
        "status",
        "method",
        "column",
        "development_values",
        "validation_values",
        "oot_values",
        "source_refs",
    }
)
_SAMPLE_SEMANTICS_FIELDS = frozenset(
    {
        "field_bindings",
        "scope",
        "performance_window",
        "observation_window",
        "split_definition",
    }
)
_DESIGN_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "sample_design_id",
        "identity",
        "analysis_universe",
        "sample_semantics",
        "relationship",
        "target_selector",
        "membership_ref",
        "population_refs",
        "historical_score_ref",
        "policy_ref",
        "compatibility",
        "source_refs",
        "content_hash",
    }
)
_DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "diagnostic_id",
        "code",
        "category",
        "status",
        "message",
        "evidence",
        "policy_ref",
        "source_refs",
        "content_hash",
    }
)
_DIAGNOSTIC_EVIDENCE_FIELDS = frozenset(
    {"actual", "expected", "numerator", "denominator"}
)
_SAMPLE_DESIGN_REF_FIELDS = frozenset({"sample_design_id", "content_hash"})
_METRIC_DEFINITION_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "metric_definition_id",
        "metric_key",
        "display_name",
        "metric_family",
        "numerator_definition",
        "denominator_definition",
        "aggregation",
        "unit",
        "precision",
        "content_hash",
    }
)
_METRIC_DEFINITION_REF_FIELDS = frozenset(
    {"metric_definition_id", "content_hash"}
)
_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "observation_id",
        "sample_design_ref",
        "metric_definition_ref",
        "population",
        "partition",
        "status",
        "value",
        "numerator",
        "denominator",
        "sample_count",
        "unit",
        "source_refs",
        "content_hash",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "bundle_id",
        "sample_design",
        "populations",
        "membership",
        "historical_score",
        "policy",
        "diagnostics",
        "metric_definitions",
        "metric_observations",
        "content_hash",
    }
)

_DIAGNOSTIC_CATEGORY_BY_CODE = {
    "target_selector": "contract",
    "entity_overlap": "leakage",
    "temporal_oot": "leakage",
    "risk_outside_approval": "leakage",
    "maturity": "maturity",
    "label_coverage": "maturity",
    "historical_score_coverage": "coverage",
    "group_coverage_gap": "bias",
    "sufficiency": "sufficiency",
}
_DEFAULT_DIAGNOSTIC_SEVERITIES = {
    "entity_overlap": "fail",
    "temporal_oot": "fail",
    "risk_outside_approval": "fail",
    "maturity": "fail",
    "label_coverage": "fail",
    "historical_score_coverage": "warn",
    "group_coverage_gap": "warn",
    "sufficiency": "fail",
}

_METRIC_DEFINITION_SPECS = (
    {
        "metric_key": "population_count",
        "display_name": "样本件数",
        "metric_family": "volume",
        "numerator_definition": "Rows in the bound population and partition mask.",
        "denominator_definition": None,
        "aggregation": "count",
        "unit": "count",
        "precision": 0,
    },
    {
        "metric_key": "labeled_count",
        "display_name": "已标注件数",
        "metric_family": "coverage",
        "numerator_definition": "Rows with a resolved binary target label.",
        "denominator_definition": None,
        "aggregation": "count",
        "unit": "count",
        "precision": 0,
    },
    {
        "metric_key": "label_coverage",
        "display_name": "标签覆盖率",
        "metric_family": "coverage",
        "numerator_definition": "Rows with a resolved binary target label.",
        "denominator_definition": "Rows in the bound population and partition mask.",
        "aggregation": "ratio",
        "unit": "ratio",
        "precision": 6,
    },
    {
        "metric_key": "bad_count",
        "display_name": "坏样本件数",
        "metric_family": "risk",
        "numerator_definition": "Matured labeled rows equal to target bad_value.",
        "denominator_definition": None,
        "aggregation": "count",
        "unit": "count",
        "precision": 0,
    },
    {
        "metric_key": "bad_rate",
        "display_name": "坏样本率",
        "metric_family": "risk",
        "numerator_definition": "Matured labeled rows equal to target bad_value.",
        "denominator_definition": "Matured rows with a resolved binary target label.",
        "aggregation": "ratio",
        "unit": "ratio",
        "precision": 6,
    },
)


class StrategySampleDesignV2Error(StrategyError):
    """A value violates the exact StrategySampleDesign V2 contract."""


def build_target_selector_v2(
    *,
    status: str,
    column: str | None = None,
    good_value: str | bool | int | float | None = None,
    bad_value: str | bool | int | float | None = None,
    drop_missing: bool | None = None,
    source_refs: Sequence[Mapping[str, Any]] = (),
    reason: str | None = None,
) -> dict[str, Any]:
    """Build the explicit target-selector contract used by risk diagnostics."""

    return validate_target_selector_v2(
        {
            "status": status,
            "column": column,
            "good_value": good_value,
            "bad_value": bad_value,
            "drop_missing": drop_missing,
            "source_refs": list(source_refs),
            "reason": reason,
        }
    )


def validate_target_selector_v2(value: object) -> dict[str, Any]:
    obj = _object(value, "target_selector")
    _require_exact_fields(obj, _TARGET_SELECTOR_FIELDS, "target_selector")
    status = _enum(obj["status"], {"resolved", "unavailable"}, "target_selector.status")
    refs = _source_refs(obj["source_refs"], "target_selector.source_refs")
    if status == "resolved":
        column = _text(obj["column"], "target_selector.column")
        good_value = _binary_target_value(
            obj["good_value"], "target_selector.good_value"
        )
        bad_value = _binary_target_value(
            obj["bad_value"], "target_selector.bad_value"
        )
        if {good_value, bad_value} != {0, 1}:
            raise StrategySampleDesignV2Error(
                "target_selector good_value and bad_value must be complementary 0/1"
            )
        if not isinstance(obj["drop_missing"], bool):
            raise StrategySampleDesignV2Error(
                "target_selector.drop_missing must be a boolean"
            )
        if not refs:
            raise StrategySampleDesignV2Error(
                "resolved target_selector requires source_refs"
            )
        if obj["reason"] is not None:
            raise StrategySampleDesignV2Error(
                "resolved target_selector.reason must be null"
            )
        return {
            "status": status,
            "column": column,
            "good_value": good_value,
            "bad_value": bad_value,
            "drop_missing": obj["drop_missing"],
            "source_refs": refs,
            "reason": None,
        }
    if any(
        obj[field] is not None
        for field in ("column", "good_value", "bad_value", "drop_missing")
    ):
        raise StrategySampleDesignV2Error(
            "unavailable target_selector fields must be null"
        )
    return {
        "status": status,
        "column": None,
        "good_value": None,
        "bad_value": None,
        "drop_missing": None,
        "source_refs": refs,
        "reason": _text(obj["reason"], "target_selector.reason"),
    }


def build_sample_population_v2(
    *,
    role: str,
    membership_header: Mapping[str, Any],
    inclusion_predicate_ref: Mapping[str, Any] | None,
    exclusion_predicate_ref: Mapping[str, Any] | None,
    maturity_evidence: Mapping[str, Any] | None = None,
    source_refs: Sequence[Mapping[str, Any]] = (),
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build one population solely from an authenticated membership header."""

    header = _membership_header(membership_header)
    population_role = _enum(role, set(POPULATION_ROLES), "population.role")
    counts = header["counts"][population_role]
    membership_ref = _membership_ref_from_header(header)
    partitions = [
        {
            "name": name,
            "membership_ref": {
                "membership_id": header["membership_id"],
                "membership_content_hash": header["content_hash"],
                "mask_name": f"{population_role}/{name}",
            },
            "row_count": counts[name],
        }
        for name in PARTITION_NAMES
    ]
    if population_role == "approval" and maturity_evidence is None:
        maturity_evidence = {
            "status": "not_applicable",
            "performance_window_days": None,
            "cutoff_date": None,
            "eligible_count": None,
            "labeled_count": None,
            "source_refs": [],
            "reason": "Approval population does not require outcome maturity.",
        }
    normalized_maturity = _maturity_evidence(
        maturity_evidence,
        role=population_role,
        population_count=counts["total"],
    )
    body = {
        "schema_version": STRATEGY_SAMPLE_POPULATION_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(producer_version),
        "role": population_role,
        "membership_ref": membership_ref,
        "inclusion_predicate_ref": _predicate_ref(
            inclusion_predicate_ref,
            "sample population.inclusion_predicate_ref",
        ),
        "exclusion_predicate_ref": _predicate_ref(
            exclusion_predicate_ref,
            "sample population.exclusion_predicate_ref",
        ),
        "partitions": partitions,
        "total_count": counts["total"],
        "maturity_evidence": normalized_maturity,
        "source_refs": _source_refs(source_refs, "sample population.source_refs"),
    }
    return validate_sample_population_v2(
        _address_object(body, "population_id"),
        membership_header=header,
    )


def validate_sample_population_v2(
    value: object,
    *,
    membership_header: Mapping[str, Any],
) -> dict[str, Any]:
    obj = _object(value, "sample population")
    _require_exact_fields(obj, _POPULATION_FIELDS, "sample population")
    if obj["schema_version"] != STRATEGY_SAMPLE_POPULATION_V2_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error(
            "sample population schema_version is invalid"
        )
    producer = _producer_version(obj["producer_version"])
    role = _enum(obj["role"], set(POPULATION_ROLES), "sample population.role")
    header = _membership_header(membership_header)
    membership_ref = _membership_ref(obj["membership_ref"])
    if membership_ref != _membership_ref_from_header(header):
        raise StrategySampleDesignV2Error(
            "sample population membership_ref does not match membership header"
        )
    inclusion_ref = _predicate_ref(
        obj["inclusion_predicate_ref"],
        "sample population.inclusion_predicate_ref",
    )
    exclusion_ref = _predicate_ref(
        obj["exclusion_predicate_ref"],
        "sample population.exclusion_predicate_ref",
    )
    partitions = _population_partitions(
        obj["partitions"], role=role, membership_header=header
    )
    total_count = _non_negative_int(
        obj["total_count"], "sample population.total_count"
    )
    expected_total = header["counts"][role]["total"]
    if total_count != expected_total or total_count != sum(
        item["row_count"] for item in partitions
    ):
        raise StrategySampleDesignV2Error(
            "sample population counts do not conserve membership"
        )
    maturity = _maturity_evidence(
        obj["maturity_evidence"],
        role=role,
        population_count=total_count,
    )
    source_refs = _source_refs(obj["source_refs"], "sample population.source_refs")
    normalized_body = {
        "schema_version": STRATEGY_SAMPLE_POPULATION_V2_SCHEMA_VERSION,
        "producer_version": producer,
        "role": role,
        "membership_ref": membership_ref,
        "inclusion_predicate_ref": inclusion_ref,
        "exclusion_predicate_ref": exclusion_ref,
        "partitions": partitions,
        "total_count": total_count,
        "maturity_evidence": maturity,
        "source_refs": source_refs,
    }
    return _validate_addressed_object(obj, normalized_body, "population_id")


def build_historical_score_v2(
    *,
    status: str,
    column: str | None = None,
    direction: str | None = None,
    source_refs: Sequence[Mapping[str, Any]] = (),
    reason: str | None = None,
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> dict[str, Any]:
    body = {
        "schema_version": STRATEGY_HISTORICAL_SCORE_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(producer_version),
        "status": status,
        "column": column,
        "direction": direction,
        "source_refs": _source_refs(source_refs, "historical_score.source_refs"),
        "reason": reason,
    }
    return validate_historical_score_v2(
        _address_object(body, "historical_score_id")
    )


def validate_historical_score_v2(value: object) -> dict[str, Any]:
    obj = _object(value, "historical_score")
    _require_exact_fields(obj, _HISTORICAL_SCORE_FIELDS, "historical_score")
    if obj["schema_version"] != STRATEGY_HISTORICAL_SCORE_V2_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error("historical_score schema_version is invalid")
    producer = _producer_version(obj["producer_version"])
    status = _enum(
        obj["status"], HISTORICAL_SCORE_STATUSES, "historical_score.status"
    )
    source_refs = _source_refs(obj["source_refs"], "historical_score.source_refs")
    if status == "available":
        column = _text(obj["column"], "historical_score.column")
        direction = _enum(
            obj["direction"],
            HISTORICAL_SCORE_DIRECTIONS,
            "historical_score.direction",
        )
        if not source_refs:
            raise StrategySampleDesignV2Error(
                "available historical_score requires source_refs"
            )
        if obj["reason"] is not None:
            raise StrategySampleDesignV2Error(
                "available historical_score.reason must be null"
            )
        reason = None
    else:
        if obj["column"] is not None or obj["direction"] is not None:
            raise StrategySampleDesignV2Error(
                "non-available historical_score fields must be null"
            )
        column = None
        direction = None
        reason = _text(obj["reason"], "historical_score.reason")
    normalized_body = {
        "schema_version": STRATEGY_HISTORICAL_SCORE_V2_SCHEMA_VERSION,
        "producer_version": producer,
        "status": status,
        "column": column,
        "direction": direction,
        "source_refs": source_refs,
        "reason": reason,
    }
    return _validate_addressed_object(
        obj, normalized_body, "historical_score_id"
    )


def build_sample_design_policy_v2(
    *,
    minimum_partition_count: int,
    minimum_bad_count: int,
    minimum_label_coverage: float,
    minimum_historical_score_coverage: float,
    maximum_group_coverage_gap: float,
    diagnostic_severities: Mapping[str, str] | None = None,
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> dict[str, Any]:
    severity_input = (
        _DEFAULT_DIAGNOSTIC_SEVERITIES
        if diagnostic_severities is None
        else diagnostic_severities
    )
    severity_obj = _object(severity_input, "sample policy.diagnostic_severities")
    _require_exact_fields(
        severity_obj,
        _SEVERITY_FIELDS,
        "sample policy.diagnostic_severities",
    )
    severities = {
        key: _enum(
            severity_obj[key],
            {"warn", "fail"},
            f"sample policy.diagnostic_severities.{key}",
        )
        for key in sorted(_SEVERITY_FIELDS)
    }
    body = {
        "schema_version": STRATEGY_SAMPLE_POLICY_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(producer_version),
        "minimum_partition_count": _non_negative_int(
            minimum_partition_count, "sample policy.minimum_partition_count"
        ),
        "minimum_bad_count": _non_negative_int(
            minimum_bad_count, "sample policy.minimum_bad_count"
        ),
        "minimum_label_coverage": _ratio(
            minimum_label_coverage, "sample policy.minimum_label_coverage"
        ),
        "minimum_historical_score_coverage": _ratio(
            minimum_historical_score_coverage,
            "sample policy.minimum_historical_score_coverage",
        ),
        "maximum_group_coverage_gap": _ratio(
            maximum_group_coverage_gap,
            "sample policy.maximum_group_coverage_gap",
        ),
        "diagnostic_severities": severities,
    }
    return validate_sample_design_policy_v2(
        _address_object(body, "policy_id")
    )


def validate_sample_design_policy_v2(value: object) -> dict[str, Any]:
    obj = _object(value, "sample policy")
    _require_exact_fields(obj, _POLICY_FIELDS, "sample policy")
    if obj["schema_version"] != STRATEGY_SAMPLE_POLICY_V2_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error("sample policy schema_version is invalid")
    producer = _producer_version(obj["producer_version"])
    severities_obj = _object(
        obj["diagnostic_severities"], "sample policy.diagnostic_severities"
    )
    _require_exact_fields(
        severities_obj,
        _SEVERITY_FIELDS,
        "sample policy.diagnostic_severities",
    )
    severities = {
        key: _enum(
            severities_obj[key],
            {"warn", "fail"},
            f"sample policy.diagnostic_severities.{key}",
        )
        for key in sorted(_SEVERITY_FIELDS)
    }
    normalized_body = {
        "schema_version": STRATEGY_SAMPLE_POLICY_V2_SCHEMA_VERSION,
        "producer_version": producer,
        "minimum_partition_count": _non_negative_int(
            obj["minimum_partition_count"],
            "sample policy.minimum_partition_count",
        ),
        "minimum_bad_count": _non_negative_int(
            obj["minimum_bad_count"], "sample policy.minimum_bad_count"
        ),
        "minimum_label_coverage": _ratio(
            obj["minimum_label_coverage"],
            "sample policy.minimum_label_coverage",
        ),
        "minimum_historical_score_coverage": _ratio(
            obj["minimum_historical_score_coverage"],
            "sample policy.minimum_historical_score_coverage",
        ),
        "maximum_group_coverage_gap": _ratio(
            obj["maximum_group_coverage_gap"],
            "sample policy.maximum_group_coverage_gap",
        ),
        "diagnostic_severities": severities,
    }
    return _validate_addressed_object(obj, normalized_body, "policy_id")


def build_strategy_sample_design_v2(
    *,
    task_id: str,
    membership_header: Mapping[str, Any],
    workspace_revision: int,
    workspace_generation: int,
    semantic_mapping_hash: str,
    relationship: str,
    field_bindings: Mapping[str, Any],
    scope: str,
    performance_window: Mapping[str, Any],
    observation_window: Mapping[str, Any],
    split_definition: Mapping[str, Any],
    target_selector: Mapping[str, Any],
    approval_population: Mapping[str, Any],
    risk_population: Mapping[str, Any],
    historical_score: Mapping[str, Any],
    policy: Mapping[str, Any],
    legacy_development_ref: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]] = (),
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> dict[str, Any]:
    header = _membership_header(membership_header)
    task = _text(task_id, "sample design.task_id")
    if task != header["task_id"]:
        raise StrategySampleDesignV2Error(
            "sample design task_id does not match membership header"
        )
    producer = _producer_version(producer_version)
    approval = validate_sample_population_v2(
        approval_population, membership_header=header
    )
    risk = validate_sample_population_v2(risk_population, membership_header=header)
    if approval["role"] != "approval" or risk["role"] != "risk":
        raise StrategySampleDesignV2Error(
            "sample design requires approval and risk populations"
        )
    historical = validate_historical_score_v2(historical_score)
    normalized_policy = validate_sample_design_policy_v2(policy)
    _require_producer_versions(
        producer,
        (approval, risk, historical, normalized_policy),
        "sample design child",
    )
    normalized_relationship = _enum(
        relationship, SAMPLE_RELATIONSHIPS, "sample design.relationship"
    )
    _validate_relationship_counts(normalized_relationship, header)
    identity = _identity(
        {
            "task_id": task,
            "dataset_ref": {**header["dataset_ref"], "role": "active"},
            "workspace_ref": {
                "revision": workspace_revision,
                "generation": workspace_generation,
                "semantic_mapping_hash": semantic_mapping_hash,
            },
        },
        membership_header=header,
    )
    semantics = _sample_semantics(
        {
            "field_bindings": field_bindings,
            "scope": scope,
            "performance_window": performance_window,
            "observation_window": observation_window,
            "split_definition": split_definition,
        },
        risk_maturity=risk["maturity_evidence"],
    )
    body = {
        "schema_version": STRATEGY_SAMPLE_DESIGN_V2_SCHEMA_VERSION,
        "producer_version": producer,
        "identity": identity,
        "analysis_universe": {
            "dataset_ref": header["dataset_ref"],
            "row_count": header["row_count"],
            "row_ordinal": header["row_ordinal"],
        },
        "sample_semantics": semantics,
        "relationship": normalized_relationship,
        "target_selector": validate_target_selector_v2(target_selector),
        "membership_ref": _membership_ref_from_header(header),
        "population_refs": [
            _population_ref(approval),
            _population_ref(risk),
        ],
        "historical_score_ref": _historical_score_ref(historical),
        "policy_ref": _policy_ref(normalized_policy),
        "compatibility": {
            "legacy_development_ref": _legacy_development_ref(
                legacy_development_ref
            ),
            "maps_to": "risk/development",
        },
        "source_refs": _source_refs(source_refs, "sample design.source_refs"),
    }
    return validate_strategy_sample_design_v2(
        _address_object(body, "sample_design_id"),
        membership_header=header,
        populations=(approval, risk),
        historical_score=historical,
        policy=normalized_policy,
    )


def validate_strategy_sample_design_v2(
    value: object,
    *,
    membership_header: Mapping[str, Any],
    populations: Sequence[Mapping[str, Any]],
    historical_score: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    obj = _object(value, "sample design")
    _require_exact_fields(obj, _DESIGN_FIELDS, "sample design")
    if obj["schema_version"] != STRATEGY_SAMPLE_DESIGN_V2_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error("sample design schema_version is invalid")
    producer = _producer_version(obj["producer_version"])
    header = _membership_header(membership_header)
    normalized_populations = _two_populations(populations, header)
    historical = validate_historical_score_v2(historical_score)
    normalized_policy = validate_sample_design_policy_v2(policy)
    _require_producer_versions(
        producer,
        (*normalized_populations, historical, normalized_policy),
        "sample design child",
    )
    identity = _identity(obj["identity"], membership_header=header)
    analysis_universe = _analysis_universe(obj["analysis_universe"], header)
    semantics = _sample_semantics(
        obj["sample_semantics"],
        risk_maturity=normalized_populations[1]["maturity_evidence"],
    )
    membership_ref = _membership_ref(obj["membership_ref"])
    if membership_ref != _membership_ref_from_header(header):
        raise StrategySampleDesignV2Error(
            "sample design membership_ref does not match membership header"
        )
    population_refs = _population_refs(obj["population_refs"])
    expected_population_refs = [
        _population_ref(item) for item in normalized_populations
    ]
    if population_refs != expected_population_refs:
        raise StrategySampleDesignV2Error(
            "sample design population_refs do not match populations"
        )
    historical_ref = _historical_score_ref_from_value(
        obj["historical_score_ref"]
    )
    if historical_ref != _historical_score_ref(historical):
        raise StrategySampleDesignV2Error(
            "sample design historical_score_ref does not match historical_score"
        )
    policy_ref = _policy_ref_from_value(obj["policy_ref"])
    if policy_ref != _policy_ref(normalized_policy):
        raise StrategySampleDesignV2Error(
            "sample design policy_ref does not match policy"
        )
    compatibility = _compatibility(obj["compatibility"])
    relationship = _enum(
        obj["relationship"],
        SAMPLE_RELATIONSHIPS,
        "sample design.relationship",
    )
    _validate_relationship_counts(relationship, header)
    normalized_body = {
        "schema_version": STRATEGY_SAMPLE_DESIGN_V2_SCHEMA_VERSION,
        "producer_version": producer,
        "identity": identity,
        "analysis_universe": analysis_universe,
        "sample_semantics": semantics,
        "relationship": relationship,
        "target_selector": validate_target_selector_v2(obj["target_selector"]),
        "membership_ref": membership_ref,
        "population_refs": population_refs,
        "historical_score_ref": historical_ref,
        "policy_ref": policy_ref,
        "compatibility": compatibility,
        "source_refs": _source_refs(obj["source_refs"], "sample design.source_refs"),
    }
    return _validate_addressed_object(obj, normalized_body, "sample_design_id")


def build_sample_diagnostic_v2(
    *,
    code: str,
    status: str,
    message: str,
    evidence: Mapping[str, Any],
    policy_ref: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]] = (),
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> dict[str, Any]:
    diagnostic_code = _enum(code, set(DIAGNOSTIC_CODES), "diagnostic.code")
    body = {
        "schema_version": STRATEGY_SAMPLE_DIAGNOSTIC_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(producer_version),
        "code": diagnostic_code,
        "category": _DIAGNOSTIC_CATEGORY_BY_CODE[diagnostic_code],
        "status": _enum(status, DIAGNOSTIC_STATUSES, "sample diagnostic.status"),
        "message": _text(message, "sample diagnostic.message"),
        "evidence": _diagnostic_evidence(evidence),
        "policy_ref": _policy_ref_from_value(policy_ref),
        "source_refs": _source_refs(source_refs, "sample diagnostic.source_refs"),
    }
    return validate_sample_diagnostic_v2(
        _address_object(body, "diagnostic_id")
    )


def validate_sample_diagnostic_v2(value: object) -> dict[str, Any]:
    obj = _object(value, "sample diagnostic")
    _require_exact_fields(obj, _DIAGNOSTIC_FIELDS, "sample diagnostic")
    if obj["schema_version"] != STRATEGY_SAMPLE_DIAGNOSTIC_V2_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error("sample diagnostic schema_version is invalid")
    code = _enum(obj["code"], set(DIAGNOSTIC_CODES), "sample diagnostic.code")
    category = _enum(
        obj["category"], DIAGNOSTIC_CATEGORIES, "sample diagnostic.category"
    )
    if category != _DIAGNOSTIC_CATEGORY_BY_CODE[code]:
        raise StrategySampleDesignV2Error(
            "sample diagnostic category does not match code"
        )
    normalized_body = {
        "schema_version": STRATEGY_SAMPLE_DIAGNOSTIC_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(obj["producer_version"]),
        "code": code,
        "category": category,
        "status": _enum(
            obj["status"], DIAGNOSTIC_STATUSES, "sample diagnostic.status"
        ),
        "message": _text(obj["message"], "sample diagnostic.message"),
        "evidence": _diagnostic_evidence(obj["evidence"]),
        "policy_ref": _policy_ref_from_value(obj["policy_ref"]),
        "source_refs": _source_refs(
            obj["source_refs"], "sample diagnostic.source_refs"
        ),
    }
    return _validate_addressed_object(obj, normalized_body, "diagnostic_id")


def evaluate_sample_design_v2_diagnostics(
    *,
    sample_design: Mapping[str, Any],
    membership_header: Mapping[str, Any],
    membership_masks: Mapping[str, object],
    approval_population: Mapping[str, Any],
    risk_population: Mapping[str, Any],
    historical_score: Mapping[str, Any],
    policy: Mapping[str, Any],
    statistics: Mapping[str, Any],
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> list[dict[str, Any]]:
    """Evaluate deterministic diagnostics from resolved masks and typed stats."""

    header = _membership_header(membership_header)
    approval = validate_sample_population_v2(
        approval_population, membership_header=header
    )
    risk = validate_sample_population_v2(risk_population, membership_header=header)
    historical = validate_historical_score_v2(historical_score)
    normalized_policy = validate_sample_design_policy_v2(policy)
    design = validate_strategy_sample_design_v2(
        sample_design,
        membership_header=header,
        populations=(approval, risk),
        historical_score=historical,
        policy=normalized_policy,
    )
    producer = _producer_version(producer_version)
    _require_producer_versions(
        producer,
        (design, approval, risk, historical, normalized_policy),
        "diagnostic input",
    )
    masks = _resolved_membership_masks(membership_masks, header)
    stats = _diagnostic_statistics(statistics)
    _validate_diagnostic_statistics_conservation(
        stats,
        membership_header=header,
    )
    policy_ref = _policy_ref(normalized_policy)
    severities = normalized_policy["diagnostic_severities"]
    membership_source = {
        "kind": "sample_membership",
        "ref_id": header["membership_id"],
        "content_hash": header["content_hash"],
    }
    diagnostics: list[dict[str, Any]] = []

    target = design["target_selector"]
    target_status = "pass" if target["status"] == "resolved" else "unavailable"
    diagnostics.append(
        _diagnostic(
            code="target_selector",
            status=target_status,
            actual=target["column"],
            expected="resolved",
            source_refs=target["source_refs"],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )

    entity = stats["entity_overlap"]
    if entity["availability"] == "available":
        entity_status = (
            "pass"
            if entity["overlap_count"] == 0
            else severities["entity_overlap"]
        )
    else:
        entity_status = _availability_to_diagnostic_status(entity["availability"])
    diagnostics.append(
        _diagnostic(
            code="entity_overlap",
            status=entity_status,
            actual=entity["overlap_count"],
            expected=0,
            numerator=entity["overlap_count"],
            denominator=entity["compared_count"],
            source_refs=entity["source_refs"],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )

    temporal = stats["temporal_oot"]
    oot_count = (
        header["counts"]["approval"]["oot"]
        + header["counts"]["risk"]["oot"]
    )
    if oot_count == 0:
        temporal_status = "not_applicable"
    elif temporal["availability"] == "available":
        temporal_status = (
            "pass" if temporal["ordered"] else severities["temporal_oot"]
        )
    else:
        temporal_status = _availability_to_diagnostic_status(
            temporal["availability"]
        )
    diagnostics.append(
        _diagnostic(
            code="temporal_oot",
            status=temporal_status,
            actual=temporal["ordered"],
            expected=True,
            source_refs=temporal["source_refs"],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )

    risk_union = _population_union(masks, "risk")
    risk_outside_count = header["counts"]["relationship"][
        "risk_outside_approval"
    ]["total"]
    if design["relationship"] == "parallel_time_cohorts":
        risk_outside_status = "not_applicable"
    else:
        risk_outside_status = (
            "pass"
            if risk_outside_count == 0
            else severities["risk_outside_approval"]
        )
    diagnostics.append(
        _diagnostic(
            code="risk_outside_approval",
            status=risk_outside_status,
            actual=risk_outside_count,
            expected=0,
            numerator=risk_outside_count,
            denominator=int(np.count_nonzero(risk_union)),
            source_refs=[membership_source],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )

    maturity = risk["maturity_evidence"]
    if maturity["status"] == "confirmed_matured":
        maturity_status = (
            "pass"
            if maturity["eligible_count"] == risk["total_count"]
            else severities["maturity"]
        )
    elif maturity["status"] == "not_matured":
        maturity_status = "fail"
    else:
        maturity_status = "unavailable"
    diagnostics.append(
        _diagnostic(
            code="maturity",
            status=maturity_status,
            actual=maturity["eligible_count"],
            expected=risk["total_count"],
            source_refs=maturity["source_refs"],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )

    if maturity["status"] == "not_matured":
        label_status = "fail"
        label_ratio = (
            None
            if not maturity["eligible_count"]
            else maturity["labeled_count"] / maturity["eligible_count"]
        )
    elif maturity["status"] in {"unknown", "unavailable"}:
        label_status = "unavailable"
        label_ratio = None
    elif risk["total_count"] == 0:
        label_status = "not_applicable"
        label_ratio = None
    else:
        label_ratio = maturity["labeled_count"] / risk["total_count"]
        label_status = (
            "pass"
            if label_ratio >= normalized_policy["minimum_label_coverage"]
            else severities["label_coverage"]
        )
    diagnostics.append(
        _diagnostic(
            code="label_coverage",
            status=label_status,
            actual=label_ratio,
            expected=normalized_policy["minimum_label_coverage"],
            numerator=maturity["labeled_count"],
            denominator=(
                risk["total_count"]
                if maturity["status"] == "confirmed_matured"
                else maturity["eligible_count"]
            ),
            source_refs=maturity["source_refs"],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )

    score_coverage = stats["historical_score_coverage"]
    if historical["status"] == "not_applicable":
        score_status = "not_applicable"
        score_ratio = None
    elif historical["status"] == "unavailable":
        score_status = "unavailable"
        score_ratio = None
    elif score_coverage["availability"] != "available":
        score_status = _availability_to_diagnostic_status(
            score_coverage["availability"]
        )
        score_ratio = None
    elif score_coverage["eligible_count"] == 0:
        score_status = "not_applicable"
        score_ratio = None
    else:
        score_ratio = (
            score_coverage["covered_count"] / score_coverage["eligible_count"]
        )
        score_status = (
            "pass"
            if score_ratio
            >= normalized_policy["minimum_historical_score_coverage"]
            else severities["historical_score_coverage"]
        )
    diagnostics.append(
        _diagnostic(
            code="historical_score_coverage",
            status=score_status,
            actual=score_ratio,
            expected=normalized_policy["minimum_historical_score_coverage"],
            numerator=score_coverage["covered_count"],
            denominator=score_coverage["eligible_count"],
            source_refs=[*historical["source_refs"], *score_coverage["source_refs"]],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )

    group_gap = stats["group_coverage_gap"]
    if group_gap["availability"] == "available":
        group_status = (
            "pass"
            if group_gap["maximum_gap"]
            <= normalized_policy["maximum_group_coverage_gap"]
            else severities["group_coverage_gap"]
        )
    else:
        group_status = _availability_to_diagnostic_status(group_gap["availability"])
    diagnostics.append(
        _diagnostic(
            code="group_coverage_gap",
            status=group_status,
            actual=group_gap["maximum_gap"],
            expected=normalized_policy["maximum_group_coverage_gap"],
            numerator=group_gap["group_count"],
            source_refs=group_gap["source_refs"],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )

    sufficiency = stats["sufficiency"]
    development_count = header["counts"]["risk"]["development"]
    if sufficiency["availability"] == "available":
        bad_upper_bound = development_count
        if maturity["labeled_count"] is not None:
            bad_upper_bound = min(bad_upper_bound, maturity["labeled_count"])
        if sufficiency["bad_count"] > bad_upper_bound:
            raise StrategySampleDesignV2Error(
                "sufficiency bad_count exceeds matured/labeled development sample"
            )
        sufficient = (
            development_count >= normalized_policy["minimum_partition_count"]
            and sufficiency["bad_count"] >= normalized_policy["minimum_bad_count"]
        )
        sufficiency_status = (
            "pass" if sufficient else severities["sufficiency"]
        )
    else:
        sufficiency_status = _availability_to_diagnostic_status(
            sufficiency["availability"]
        )
    diagnostics.append(
        _diagnostic(
            code="sufficiency",
            status=sufficiency_status,
            actual=development_count,
            expected=normalized_policy["minimum_partition_count"],
            numerator=sufficiency["bad_count"],
            denominator=normalized_policy["minimum_bad_count"],
            source_refs=[membership_source, *sufficiency["source_refs"]],
            policy_ref=policy_ref,
            producer_version=producer,
        )
    )
    return diagnostics


def build_metric_definition_v2(
    metric_key: str,
    *,
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build one member of the fixed V2 sample metric vocabulary."""

    key = _text(metric_key, "metric_definition.metric_key")
    spec = next(
        (item for item in _METRIC_DEFINITION_SPECS if item["metric_key"] == key),
        None,
    )
    if spec is None:
        raise StrategySampleDesignV2Error(
            f"metric_definition.metric_key is unsupported: {key}"
        )
    body = {
        "schema_version": STRATEGY_METRIC_DEFINITION_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(producer_version),
        **spec,
    }
    return validate_metric_definition_v2(
        _address_object(body, "metric_definition_id")
    )


def build_metric_definitions_v2(
    *,
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> list[dict[str, Any]]:
    """Return the sole canonical fixed metric-definition set."""

    definitions = [
        build_metric_definition_v2(
            spec["metric_key"], producer_version=producer_version
        )
        for spec in _METRIC_DEFINITION_SPECS
    ]
    definitions.sort(key=lambda item: item["metric_key"])
    return definitions


def validate_metric_definition_v2(value: object) -> dict[str, Any]:
    obj = _object(value, "metric definition v2")
    _require_exact_fields(obj, _METRIC_DEFINITION_FIELDS, "metric definition v2")
    if obj["schema_version"] != STRATEGY_METRIC_DEFINITION_V2_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error(
            "metric definition v2 schema_version is invalid"
        )
    key = _text(obj["metric_key"], "metric definition v2.metric_key")
    spec = next(
        (item for item in _METRIC_DEFINITION_SPECS if item["metric_key"] == key),
        None,
    )
    if spec is None:
        raise StrategySampleDesignV2Error(
            f"metric definition v2 metric_key is unsupported: {key}"
        )
    body = {
        "schema_version": STRATEGY_METRIC_DEFINITION_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(obj["producer_version"]),
        **spec,
    }
    normalized = _validate_addressed_object(
        obj, body, "metric_definition_id"
    )
    if any(normalized[field] != spec[field] for field in spec):
        raise StrategySampleDesignV2Error(
            "metric definition v2 does not match fixed contract"
        )
    return normalized


def build_metric_observation_v2(
    *,
    sample_design_ref: Mapping[str, Any],
    metric_definition: Mapping[str, Any],
    population: str,
    partition: str,
    status: str,
    value: int | float | bool | None,
    numerator: int | float | None,
    denominator: int | float | None,
    sample_count: int,
    source_refs: Sequence[Mapping[str, Any]],
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> dict[str, Any]:
    definition = validate_metric_definition_v2(metric_definition)
    body = {
        "schema_version": STRATEGY_METRIC_OBSERVATION_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(producer_version),
        "sample_design_ref": _sample_design_ref(sample_design_ref),
        "metric_definition_ref": _metric_definition_ref(definition),
        "population": population,
        "partition": partition,
        "status": status,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "sample_count": sample_count,
        "unit": definition["unit"],
        "source_refs": _source_refs(source_refs, "metric observation v2.source_refs"),
    }
    return validate_metric_observation_v2(
        _address_object(body, "observation_id"),
        metric_definitions=(definition,),
    )


def validate_metric_observation_v2(
    value: object,
    *,
    metric_definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    obj = _object(value, "metric observation v2")
    _require_exact_fields(obj, _OBSERVATION_FIELDS, "metric observation v2")
    if obj["schema_version"] != STRATEGY_METRIC_OBSERVATION_V2_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error(
            "metric observation v2 schema_version is invalid"
        )
    population = _enum(
        obj["population"],
        set(POPULATION_ROLES),
        "metric observation v2.population",
    )
    partition = _enum(
        obj["partition"],
        {"overall", *PARTITION_NAMES},
        "metric observation v2.partition",
    )
    status = _enum(
        obj["status"],
        METRIC_OBSERVATION_V2_STATUSES,
        "metric observation v2.status",
    )
    definition_items = _array(metric_definitions, "metric_definitions")
    if not definition_items:
        raise StrategySampleDesignV2Error("metric_definitions must not be empty")
    definitions = [
        validate_metric_definition_v2(item) for item in definition_items
    ]
    definition_ids = [item["metric_definition_id"] for item in definitions]
    if len(definition_ids) != len(set(definition_ids)):
        raise StrategySampleDesignV2Error(
            "metric_definitions contain duplicate definitions"
        )
    definition_by_id = {
        item["metric_definition_id"]: item for item in definitions
    }
    definition_ref = _metric_definition_ref_from_value(
        obj["metric_definition_ref"]
    )
    definition = definition_by_id.get(definition_ref["metric_definition_id"])
    if definition is None or definition["content_hash"] != definition_ref["content_hash"]:
        raise StrategySampleDesignV2Error(
            "metric observation v2 definition ref is invalid"
        )
    unit = _enum(
        obj["unit"], METRIC_OBSERVATION_V2_UNITS, "metric observation v2.unit"
    )
    if unit != definition["unit"]:
        raise StrategySampleDesignV2Error(
            "metric observation v2 unit does not match definition"
        )
    refs = _source_refs(obj["source_refs"], "metric observation v2.source_refs")
    if not refs:
        raise StrategySampleDesignV2Error(
            "metric observation v2 requires source_refs"
        )
    if status == "present":
        observed_value = _metric_value(obj["value"], unit, "metric observation v2.value")
        numerator = _optional_number(
            obj["numerator"], "metric observation v2.numerator"
        )
        denominator = _optional_number(
            obj["denominator"], "metric observation v2.denominator"
        )
        if (numerator is None) != (denominator is None):
            raise StrategySampleDesignV2Error(
                "metric observation v2 numerator and denominator must be paired"
            )
        if denominator is not None and denominator < 0:
            raise StrategySampleDesignV2Error(
                "metric observation v2 denominator must be non-negative"
            )
        if unit == "ratio":
            if numerator is None or denominator is None:
                raise StrategySampleDesignV2Error(
                    "ratio metric observation requires numerator and denominator"
                )
            if denominator <= 0:
                raise StrategySampleDesignV2Error(
                    "ratio metric observation denominator must be positive"
                )
            if not math.isclose(
                float(observed_value),
                numerator / denominator,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise StrategySampleDesignV2Error(
                    "ratio metric observation value is inconsistent"
                )
        if not refs:
            raise StrategySampleDesignV2Error(
                "present metric observation v2 requires source_refs"
            )
    else:
        if any(obj[field] is not None for field in ("value", "numerator", "denominator")):
            raise StrategySampleDesignV2Error(
                "non-present metric observation v2 operands must be null"
            )
        observed_value = None
        numerator = None
        denominator = None
    normalized_body = {
        "schema_version": STRATEGY_METRIC_OBSERVATION_V2_SCHEMA_VERSION,
        "producer_version": _producer_version(obj["producer_version"]),
        "sample_design_ref": _sample_design_ref(obj["sample_design_ref"]),
        "metric_definition_ref": definition_ref,
        "population": population,
        "partition": partition,
        "status": status,
        "value": observed_value,
        "numerator": numerator,
        "denominator": denominator,
        "sample_count": _non_negative_int(
            obj["sample_count"], "metric observation v2.sample_count"
        ),
        "unit": unit,
        "source_refs": refs,
    }
    return _validate_addressed_object(obj, normalized_body, "observation_id")


def build_strategy_sample_design_v2_bundle(
    *,
    task_id: str,
    membership_header: Mapping[str, Any],
    membership_masks: Mapping[str, object],
    workspace_revision: int,
    workspace_generation: int,
    semantic_mapping_hash: str,
    relationship: str,
    field_bindings: Mapping[str, Any],
    scope: str,
    performance_window: Mapping[str, Any],
    observation_window: Mapping[str, Any],
    split_definition: Mapping[str, Any],
    target_selector: Mapping[str, Any],
    approval_population: Mapping[str, Any],
    risk_population: Mapping[str, Any],
    historical_score: Mapping[str, Any],
    policy: Mapping[str, Any],
    legacy_development_ref: Mapping[str, Any],
    diagnostic_statistics: Mapping[str, Any],
    metric_observations: Sequence[Mapping[str, Any]],
    source_refs: Sequence[Mapping[str, Any]] = (),
    producer_version: str = STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build an immutable V2 bundle without materializing or filtering data."""

    producer = _producer_version(producer_version)
    header = _membership_header(membership_header)
    approval = validate_sample_population_v2(
        approval_population, membership_header=header
    )
    risk = validate_sample_population_v2(risk_population, membership_header=header)
    historical = validate_historical_score_v2(historical_score)
    normalized_policy = validate_sample_design_policy_v2(policy)
    design = build_strategy_sample_design_v2(
        task_id=task_id,
        membership_header=header,
        workspace_revision=workspace_revision,
        workspace_generation=workspace_generation,
        semantic_mapping_hash=semantic_mapping_hash,
        relationship=relationship,
        field_bindings=field_bindings,
        scope=scope,
        performance_window=performance_window,
        observation_window=observation_window,
        split_definition=split_definition,
        target_selector=target_selector,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=normalized_policy,
        legacy_development_ref=legacy_development_ref,
        source_refs=source_refs,
        producer_version=producer,
    )
    diagnostics = evaluate_sample_design_v2_diagnostics(
        sample_design=design,
        membership_header=header,
        membership_masks=membership_masks,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=normalized_policy,
        statistics=diagnostic_statistics,
        producer_version=producer,
    )
    definitions = build_metric_definitions_v2(producer_version=producer)
    observations = _metric_observations(
        metric_observations,
        sample_design=design,
        populations=(approval, risk),
        membership_header=header,
        metric_definitions=definitions,
    )
    body = {
        "schema_version": STRATEGY_SAMPLE_DESIGN_V2_BUNDLE_SCHEMA_VERSION,
        "producer_version": producer,
        "sample_design": design,
        "populations": [approval, risk],
        "membership": header,
        "historical_score": historical,
        "policy": normalized_policy,
        "diagnostics": diagnostics,
        "metric_definitions": definitions,
        "metric_observations": observations,
    }
    return validate_strategy_sample_design_v2_bundle(
        _address_object(body, "bundle_id")
    )


def validate_strategy_sample_design_v2_bundle(value: object) -> dict[str, Any]:
    obj = _object(value, "sample-design v2 bundle")
    _preflight_json_tree(obj, "sample-design v2 bundle")
    _require_exact_fields(obj, _BUNDLE_FIELDS, "sample-design v2 bundle")
    if obj["schema_version"] != STRATEGY_SAMPLE_DESIGN_V2_BUNDLE_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error(
            "sample-design v2 bundle schema_version is invalid"
        )
    producer = _producer_version(obj["producer_version"])
    header = _membership_header(obj["membership"])
    populations = _two_populations(obj["populations"], header)
    historical = validate_historical_score_v2(obj["historical_score"])
    policy = validate_sample_design_policy_v2(obj["policy"])
    definitions = _metric_definitions(obj["metric_definitions"])
    design = validate_strategy_sample_design_v2(
        obj["sample_design"],
        membership_header=header,
        populations=populations,
        historical_score=historical,
        policy=policy,
    )
    diagnostics = _diagnostics(
        obj["diagnostics"], sample_design=design, policy=policy
    )
    observations = _metric_observations(
        obj["metric_observations"],
        sample_design=design,
        populations=populations,
        membership_header=header,
        metric_definitions=definitions,
    )
    _require_producer_versions(
        producer,
        (
            design,
            *populations,
            historical,
            policy,
            *diagnostics,
            *definitions,
            *observations,
        ),
        "sample-design v2 bundle child",
    )
    normalized_body = {
        "schema_version": STRATEGY_SAMPLE_DESIGN_V2_BUNDLE_SCHEMA_VERSION,
        "producer_version": producer,
        "sample_design": design,
        "populations": populations,
        "membership": header,
        "historical_score": historical,
        "policy": policy,
        "diagnostics": diagnostics,
        "metric_definitions": definitions,
        "metric_observations": observations,
    }
    normalized = _validate_addressed_object(obj, normalized_body, "bundle_id")
    _validate_observation_conservation(normalized)
    if len(_canonical_json(normalized).encode("utf-8")) > MAX_SAMPLE_DESIGN_V2_JSON_BYTES:
        raise StrategySampleDesignV2Error("sample-design v2 bundle exceeds byte budget")
    return normalized


def canonical_strategy_sample_design_v2_bundle_json(
    value: Mapping[str, Any],
) -> str:
    return _canonical_json(validate_strategy_sample_design_v2_bundle(value))


def strategy_sample_design_v2_bundle_from_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    if not isinstance(raw, (str, bytes, bytearray)):
        raise StrategySampleDesignV2Error(
            "sample-design v2 bundle JSON must be text or bytes"
        )
    size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
    if size > MAX_SAMPLE_DESIGN_V2_JSON_BYTES:
        raise StrategySampleDesignV2Error(
            "sample-design v2 bundle JSON exceeds byte budget"
        )
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except StrategySampleDesignV2Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise StrategySampleDesignV2Error(
            "sample-design v2 bundle is not valid bounded JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StrategySampleDesignV2Error(
            "sample-design v2 bundle JSON must contain an object"
        )
    return validate_strategy_sample_design_v2_bundle(payload)


def _diagnostic(
    *,
    code: str,
    status: str,
    actual: object,
    expected: object,
    policy_ref: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    producer_version: str,
    numerator: object = None,
    denominator: object = None,
) -> dict[str, Any]:
    message = f"{code}: {status}"
    return build_sample_diagnostic_v2(
        code=code,
        status=status,
        message=message,
        evidence={
            "actual": actual,
            "expected": expected,
            "numerator": numerator,
            "denominator": denominator,
        },
        policy_ref=policy_ref,
        source_refs=source_refs,
        producer_version=producer_version,
    )


def _diagnostic_statistics(value: object) -> dict[str, dict[str, Any]]:
    obj = _object(value, "diagnostic statistics")
    expected = frozenset(
        {
            "entity_overlap",
            "temporal_oot",
            "historical_score_coverage",
            "group_coverage_gap",
            "sufficiency",
        }
    )
    _require_exact_fields(obj, expected, "diagnostic statistics")
    return {
        "entity_overlap": _entity_overlap_stat(obj["entity_overlap"]),
        "temporal_oot": _temporal_oot_stat(obj["temporal_oot"]),
        "historical_score_coverage": _coverage_stat(
            obj["historical_score_coverage"]
        ),
        "group_coverage_gap": _group_gap_stat(obj["group_coverage_gap"]),
        "sufficiency": _sufficiency_stat(obj["sufficiency"]),
    }


def _validate_diagnostic_statistics_conservation(
    statistics: Mapping[str, Mapping[str, Any]],
    *,
    membership_header: Mapping[str, Any],
) -> None:
    universe = membership_header["row_count"]
    risk_total = membership_header["counts"]["risk"]["total"]
    entity = statistics["entity_overlap"]
    if (
        entity["availability"] == "available"
        and entity["compared_count"] > universe
    ):
        raise StrategySampleDesignV2Error(
            "entity_overlap compared_count exceeds analysis universe"
        )
    coverage = statistics["historical_score_coverage"]
    if (
        coverage["availability"] == "available"
        and coverage["eligible_count"] > risk_total
    ):
        raise StrategySampleDesignV2Error(
            "historical score eligible_count exceeds risk membership"
        )
    group = statistics["group_coverage_gap"]
    if group["availability"] == "available" and group["group_count"] > universe:
        raise StrategySampleDesignV2Error(
            "group coverage group_count exceeds analysis universe"
        )


def _entity_overlap_stat(value: object) -> dict[str, Any]:
    fields = frozenset(
        {"availability", "overlap_count", "compared_count", "source_refs"}
    )
    obj = _object(value, "entity_overlap statistic")
    _require_exact_fields(obj, fields, "entity_overlap statistic")
    availability = _availability(obj["availability"], "entity_overlap statistic")
    refs = _source_refs(obj["source_refs"], "entity_overlap statistic.source_refs")
    if availability == "available":
        overlap = _non_negative_int(
            obj["overlap_count"], "entity_overlap statistic.overlap_count"
        )
        compared = _non_negative_int(
            obj["compared_count"], "entity_overlap statistic.compared_count"
        )
        if overlap > compared:
            raise StrategySampleDesignV2Error(
                "entity_overlap statistic overlap_count exceeds compared_count"
            )
        _require_stat_sources(refs, "entity_overlap statistic")
    else:
        _require_null_stat_fields(obj, ("overlap_count", "compared_count"), "entity_overlap")
        overlap = None
        compared = None
    return {
        "availability": availability,
        "overlap_count": overlap,
        "compared_count": compared,
        "source_refs": refs,
    }


def _temporal_oot_stat(value: object) -> dict[str, Any]:
    fields = frozenset({"availability", "ordered", "source_refs"})
    obj = _object(value, "temporal_oot statistic")
    _require_exact_fields(obj, fields, "temporal_oot statistic")
    availability = _availability(obj["availability"], "temporal_oot statistic")
    refs = _source_refs(obj["source_refs"], "temporal_oot statistic.source_refs")
    if availability == "available":
        if not isinstance(obj["ordered"], bool):
            raise StrategySampleDesignV2Error(
                "temporal_oot statistic.ordered must be a boolean"
            )
        ordered = obj["ordered"]
        _require_stat_sources(refs, "temporal_oot statistic")
    else:
        _require_null_stat_fields(obj, ("ordered",), "temporal_oot")
        ordered = None
    return {"availability": availability, "ordered": ordered, "source_refs": refs}


def _coverage_stat(value: object) -> dict[str, Any]:
    fields = frozenset(
        {"availability", "covered_count", "eligible_count", "source_refs"}
    )
    obj = _object(value, "coverage statistic")
    _require_exact_fields(obj, fields, "coverage statistic")
    availability = _availability(obj["availability"], "coverage statistic")
    refs = _source_refs(obj["source_refs"], "coverage statistic.source_refs")
    if availability == "available":
        covered = _non_negative_int(
            obj["covered_count"], "coverage statistic.covered_count"
        )
        eligible = _non_negative_int(
            obj["eligible_count"], "coverage statistic.eligible_count"
        )
        if covered > eligible:
            raise StrategySampleDesignV2Error(
                "coverage statistic covered_count exceeds eligible_count"
            )
        _require_stat_sources(refs, "coverage statistic")
    else:
        _require_null_stat_fields(obj, ("covered_count", "eligible_count"), "coverage")
        covered = None
        eligible = None
    return {
        "availability": availability,
        "covered_count": covered,
        "eligible_count": eligible,
        "source_refs": refs,
    }


def _group_gap_stat(value: object) -> dict[str, Any]:
    fields = frozenset(
        {"availability", "maximum_gap", "group_count", "source_refs"}
    )
    obj = _object(value, "group gap statistic")
    _require_exact_fields(obj, fields, "group gap statistic")
    availability = _availability(obj["availability"], "group gap statistic")
    refs = _source_refs(obj["source_refs"], "group gap statistic.source_refs")
    if availability == "available":
        maximum_gap = _ratio(obj["maximum_gap"], "group gap statistic.maximum_gap")
        group_count = _non_negative_int(
            obj["group_count"], "group gap statistic.group_count"
        )
        _require_stat_sources(refs, "group gap statistic")
    else:
        _require_null_stat_fields(obj, ("maximum_gap", "group_count"), "group gap")
        maximum_gap = None
        group_count = None
    return {
        "availability": availability,
        "maximum_gap": maximum_gap,
        "group_count": group_count,
        "source_refs": refs,
    }


def _sufficiency_stat(value: object) -> dict[str, Any]:
    fields = frozenset({"availability", "bad_count", "source_refs"})
    obj = _object(value, "sufficiency statistic")
    _require_exact_fields(obj, fields, "sufficiency statistic")
    availability = _availability(obj["availability"], "sufficiency statistic")
    refs = _source_refs(obj["source_refs"], "sufficiency statistic.source_refs")
    if availability == "available":
        bad_count = _non_negative_int(
            obj["bad_count"], "sufficiency statistic.bad_count"
        )
        _require_stat_sources(refs, "sufficiency statistic")
    else:
        _require_null_stat_fields(obj, ("bad_count",), "sufficiency")
        bad_count = None
    return {
        "availability": availability,
        "bad_count": bad_count,
        "source_refs": refs,
    }


def _availability(value: object, name: str) -> str:
    return _enum(value, {"available", "unavailable", "not_applicable"}, f"{name}.availability")


def _availability_to_diagnostic_status(value: str) -> str:
    if value == "available":
        raise StrategySampleDesignV2Error(
            "available statistic requires an evaluated diagnostic status"
        )
    return value


def _require_stat_sources(refs: Sequence[Mapping[str, Any]], name: str) -> None:
    if not refs:
        raise StrategySampleDesignV2Error(f"available {name} requires source_refs")


def _require_null_stat_fields(
    value: Mapping[str, Any], fields: Sequence[str], name: str
) -> None:
    if any(value[field] is not None for field in fields):
        raise StrategySampleDesignV2Error(
            f"non-available {name} statistic values must be null"
        )


def _membership_header(value: object) -> dict[str, Any]:
    try:
        header = validate_sample_membership_header(_object(value, "membership header"))
    except StrategySampleDesignV2Error:
        raise
    except StrategyError as exc:
        raise StrategySampleDesignV2Error(str(exc)) from exc
    if header["schema_version"] != SAMPLE_MEMBERSHIP_SCHEMA_VERSION:
        raise StrategySampleDesignV2Error("membership header is not V2")
    return header


def _membership_ref_from_header(header: Mapping[str, Any]) -> dict[str, str]:
    return {
        "membership_id": header["membership_id"],
        "content_hash": header["content_hash"],
    }


def _membership_ref(value: object) -> dict[str, str]:
    obj = _object(value, "membership_ref")
    _require_exact_fields(obj, _MEMBERSHIP_REF_FIELDS, "membership_ref")
    return {
        "membership_id": _text(obj["membership_id"], "membership_ref.membership_id"),
        "content_hash": _hash(obj["content_hash"], "membership_ref.content_hash"),
    }


def _population_partitions(
    value: object, *, role: str, membership_header: Mapping[str, Any]
) -> list[dict[str, Any]]:
    items = _array(value, "sample population.partitions")
    if len(items) != len(PARTITION_NAMES):
        raise StrategySampleDesignV2Error(
            "sample population must have exactly three partitions"
        )
    normalized: list[dict[str, Any]] = []
    for expected_name, raw in zip(PARTITION_NAMES, items, strict=True):
        item = _object(raw, "sample population partition")
        _require_exact_fields(item, _PARTITION_FIELDS, "sample population partition")
        name = _enum(item["name"], set(PARTITION_NAMES), "sample population partition.name")
        if name != expected_name:
            raise StrategySampleDesignV2Error(
                "sample population partitions must use development/validation/oot order"
            )
        ref = _partition_membership_ref(item["membership_ref"])
        expected_ref = {
            "membership_id": membership_header["membership_id"],
            "membership_content_hash": membership_header["content_hash"],
            "mask_name": f"{role}/{name}",
        }
        if ref != expected_ref:
            raise StrategySampleDesignV2Error(
                "sample population partition membership_ref is invalid"
            )
        row_count = _non_negative_int(
            item["row_count"], "sample population partition.row_count"
        )
        if row_count != membership_header["counts"][role][name]:
            raise StrategySampleDesignV2Error(
                "sample population partition count does not match membership"
            )
        normalized.append(
            {"name": name, "membership_ref": ref, "row_count": row_count}
        )
    return normalized


def _partition_membership_ref(value: object) -> dict[str, str]:
    obj = _object(value, "partition membership_ref")
    _require_exact_fields(
        obj, _PARTITION_MEMBERSHIP_REF_FIELDS, "partition membership_ref"
    )
    mask_name = _enum(
        obj["mask_name"], set(MEMBERSHIP_MASK_ORDER), "partition membership_ref.mask_name"
    )
    return {
        "membership_id": _text(
            obj["membership_id"], "partition membership_ref.membership_id"
        ),
        "membership_content_hash": _hash(
            obj["membership_content_hash"],
            "partition membership_ref.membership_content_hash",
        ),
        "mask_name": mask_name,
    }


def _maturity_evidence(
    value: object, *, role: str, population_count: int
) -> dict[str, Any]:
    obj = _object(value, "maturity_evidence")
    _require_exact_fields(obj, _MATURITY_FIELDS, "maturity_evidence")
    status = _enum(
        obj["status"],
        {
            "confirmed_matured",
            "not_matured",
            "unknown",
            "unavailable",
            "not_applicable",
        },
        "maturity_evidence.status",
    )
    refs = _source_refs(obj["source_refs"], "maturity_evidence.source_refs")
    if role == "approval":
        if status != "not_applicable":
            raise StrategySampleDesignV2Error(
                "approval maturity_evidence must be not_applicable"
            )
    elif status == "not_applicable":
        raise StrategySampleDesignV2Error(
            "risk maturity_evidence cannot be not_applicable"
        )
    if status in {"confirmed_matured", "not_matured"}:
        days = _positive_int(
            obj["performance_window_days"],
            "maturity_evidence.performance_window_days",
        )
        cutoff = _iso_date(obj["cutoff_date"], "maturity_evidence.cutoff_date")
        eligible = _non_negative_int(
            obj["eligible_count"], "maturity_evidence.eligible_count"
        )
        labeled = _non_negative_int(
            obj["labeled_count"], "maturity_evidence.labeled_count"
        )
        if status == "confirmed_matured" and eligible != population_count:
            raise StrategySampleDesignV2Error(
                "confirmed-matured risk eligible_count must equal risk membership count"
            )
        if eligible > population_count:
            raise StrategySampleDesignV2Error(
                "maturity eligible_count exceeds risk membership count"
            )
        if labeled > eligible:
            raise StrategySampleDesignV2Error(
                "maturity labeled_count exceeds eligible_count"
            )
        if not refs:
            raise StrategySampleDesignV2Error(
                "evaluated maturity_evidence requires source_refs"
            )
        if status == "confirmed_matured" and obj["reason"] is not None:
            raise StrategySampleDesignV2Error(
                "confirmed_matured maturity_evidence.reason must be null"
            )
        reason = (
            None
            if status == "confirmed_matured"
            else _text(obj["reason"], "maturity_evidence.reason")
        )
    else:
        if any(
            obj[field] is not None
            for field in (
                "performance_window_days",
                "cutoff_date",
                "eligible_count",
                "labeled_count",
            )
        ):
            raise StrategySampleDesignV2Error(
                "non-available maturity_evidence values must be null"
            )
        days = None
        cutoff = None
        eligible = None
        labeled = None
        reason = _text(obj["reason"], "maturity_evidence.reason")
    return {
        "status": status,
        "performance_window_days": days,
        "cutoff_date": cutoff,
        "eligible_count": eligible,
        "labeled_count": labeled,
        "source_refs": refs,
        "reason": reason,
    }


def _predicate_ref(value: object, name: str) -> dict[str, str] | None:
    if value is None:
        return None
    ref = _source_ref(value, name)
    if ref["kind"] != "predicate_ast":
        raise StrategySampleDesignV2Error(f"{name}.kind must be predicate_ast")
    return ref


def _two_populations(
    value: object, membership_header: Mapping[str, Any]
) -> list[dict[str, Any]]:
    items = _array(value, "sample-design v2 populations")
    if len(items) != 2:
        raise StrategySampleDesignV2Error(
            "sample-design v2 requires exactly two populations"
        )
    normalized = [
        validate_sample_population_v2(item, membership_header=membership_header)
        for item in items
    ]
    if [item["role"] for item in normalized] != list(POPULATION_ROLES):
        raise StrategySampleDesignV2Error(
            "sample-design v2 populations must be approval then risk"
        )
    return normalized


def _analysis_universe(
    value: object, membership_header: Mapping[str, Any]
) -> dict[str, Any]:
    obj = _object(value, "analysis_universe")
    _require_exact_fields(obj, _ANALYSIS_UNIVERSE_FIELDS, "analysis_universe")
    dataset_ref = _dataset_ref(obj["dataset_ref"])
    row_count = _positive_int(obj["row_count"], "analysis_universe.row_count")
    ordinal_obj = _object(obj["row_ordinal"], "analysis_universe.row_ordinal")
    _require_exact_fields(ordinal_obj, _ROW_ORDINAL_FIELDS, "analysis_universe.row_ordinal")
    ordinal = {
        "start": _non_negative_int(ordinal_obj["start"], "analysis_universe.row_ordinal.start"),
        "stop": _positive_int(ordinal_obj["stop"], "analysis_universe.row_ordinal.stop"),
        "step": _positive_int(ordinal_obj["step"], "analysis_universe.row_ordinal.step"),
    }
    expected = {
        "dataset_ref": membership_header["dataset_ref"],
        "row_count": membership_header["row_count"],
        "row_ordinal": membership_header["row_ordinal"],
    }
    normalized = {
        "dataset_ref": dataset_ref,
        "row_count": row_count,
        "row_ordinal": ordinal,
    }
    if normalized != expected:
        raise StrategySampleDesignV2Error(
            "analysis_universe does not match membership header"
        )
    return normalized


def _identity(
    value: object, *, membership_header: Mapping[str, Any]
) -> dict[str, Any]:
    obj = _object(value, "sample design.identity")
    _require_exact_fields(obj, _IDENTITY_FIELDS, "sample design.identity")
    task_id = _text(obj["task_id"], "sample design.identity.task_id")
    if task_id != membership_header["task_id"]:
        raise StrategySampleDesignV2Error(
            "sample design task_id does not match membership header"
        )
    dataset_ref = _exact_dataset_ref(obj["dataset_ref"])
    expected_dataset_ref = {**membership_header["dataset_ref"], "role": "active"}
    if dataset_ref != expected_dataset_ref:
        raise StrategySampleDesignV2Error(
            "sample design dataset_ref does not match exact membership boundary"
        )
    return {
        "task_id": task_id,
        "dataset_ref": dataset_ref,
        "workspace_ref": _workspace_ref(obj["workspace_ref"]),
    }


def _exact_dataset_ref(value: object) -> dict[str, str]:
    obj = _object(value, "sample design.identity.dataset_ref")
    _require_exact_fields(
        obj, _EXACT_DATASET_REF_FIELDS, "sample design.identity.dataset_ref"
    )
    if obj["role"] != "active":
        raise StrategySampleDesignV2Error(
            "sample design identity dataset role must be active"
        )
    return {
        "dataset_id": _text(
            obj["dataset_id"], "sample design.identity.dataset_ref.dataset_id"
        ),
        "content_hash": _hash(
            obj["content_hash"],
            "sample design.identity.dataset_ref.content_hash",
        ),
        "role": "active",
    }


def _workspace_ref(value: object) -> dict[str, Any]:
    obj = _object(value, "sample design.identity.workspace_ref")
    _require_exact_fields(
        obj, _WORKSPACE_REF_FIELDS, "sample design.identity.workspace_ref"
    )
    return {
        "revision": _non_negative_int(
            obj["revision"], "sample design.identity.workspace_ref.revision"
        ),
        "generation": _non_negative_int(
            obj["generation"], "sample design.identity.workspace_ref.generation"
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"],
            "sample design.identity.workspace_ref.semantic_mapping_hash",
        ),
    }


def _sample_semantics(
    value: object, *, risk_maturity: Mapping[str, Any]
) -> dict[str, Any]:
    obj = _object(value, "sample design.sample_semantics")
    _require_exact_fields(
        obj, _SAMPLE_SEMANTICS_FIELDS, "sample design.sample_semantics"
    )
    fields = _field_bindings(obj["field_bindings"])
    performance = _performance_window(obj["performance_window"])
    observation = _observation_window(obj["observation_window"])
    split = _split_definition(obj["split_definition"])
    scope = _enum(
        obj["scope"],
        {"strategy_development", "exploration_only"},
        "sample design.sample_semantics.scope",
    )
    expected_scope = (
        "strategy_development"
        if performance["status"] == "provided"
        and observation["status"] == "provided"
        and risk_maturity["status"] == "confirmed_matured"
        else "exploration_only"
    )
    if scope != expected_scope:
        raise StrategySampleDesignV2Error(
            "sample design scope is inconsistent with windows and maturity"
        )
    return {
        "field_bindings": fields,
        "scope": scope,
        "performance_window": performance,
        "observation_window": observation,
        "split_definition": split,
    }


def _field_bindings(value: object) -> dict[str, str | None]:
    obj = _object(value, "sample design.field_bindings")
    _require_exact_fields(
        obj, _FIELD_BINDINGS_FIELDS, "sample design.field_bindings"
    )
    return {
        field: _optional_text(
            obj[field], f"sample design.field_bindings.{field}"
        )
        for field in sorted(_FIELD_BINDINGS_FIELDS)
    }


def _performance_window(value: object) -> dict[str, Any]:
    obj = _object(value, "sample design.performance_window")
    _require_exact_fields(
        obj, _PERFORMANCE_WINDOW_FIELDS, "sample design.performance_window"
    )
    status = _enum(
        obj["status"], {"provided", "unavailable"}, "performance_window.status"
    )
    if status == "provided":
        days = _positive_int(obj["days"], "performance_window.days")
    else:
        if obj["days"] is not None:
            raise StrategySampleDesignV2Error(
                "unavailable performance_window.days must be null"
            )
        days = None
    return {"status": status, "days": days}


def _observation_window(value: object) -> dict[str, Any]:
    obj = _object(value, "sample design.observation_window")
    _require_exact_fields(
        obj, _OBSERVATION_WINDOW_FIELDS, "sample design.observation_window"
    )
    status = _enum(
        obj["status"], {"provided", "unavailable"}, "observation_window.status"
    )
    if status == "provided":
        start = _iso_date(obj["start"], "observation_window.start")
        end = _iso_date(obj["end"], "observation_window.end")
        if start > end:
            raise StrategySampleDesignV2Error(
                "observation_window.start must not be after end"
            )
    else:
        if obj["start"] is not None or obj["end"] is not None:
            raise StrategySampleDesignV2Error(
                "unavailable observation_window bounds must be null"
            )
        start = None
        end = None
    return {"status": status, "start": start, "end": end}


def _split_definition(value: object) -> dict[str, Any]:
    obj = _object(value, "sample design.split_definition")
    _require_exact_fields(
        obj, _SPLIT_DEFINITION_FIELDS, "sample design.split_definition"
    )
    status = _enum(
        obj["status"], {"available", "unavailable"}, "split_definition.status"
    )
    refs = _source_refs(obj["source_refs"], "split_definition.source_refs")
    raw_values = {
        name: _bounded_scalar_array(
            obj[f"{name}_values"], f"split_definition.{name}_values"
        )
        for name in PARTITION_NAMES
    }
    if status == "unavailable":
        if (
            obj["method"] is not None
            or obj["column"] is not None
            or any(raw_values.values())
        ):
            raise StrategySampleDesignV2Error(
                "unavailable split_definition must not bind method, column, or values"
            )
        method = None
        column = None
    else:
        method = _enum(
            obj["method"],
            {"column_values", "time_ranges", "precomputed_masks"},
            "split_definition.method",
        )
        column = _optional_text(obj["column"], "split_definition.column")
        if method != "precomputed_masks" and column is None:
            raise StrategySampleDesignV2Error(
                "column/time split_definition requires a bound column"
            )
        if method != "precomputed_masks" and not raw_values["development"]:
            raise StrategySampleDesignV2Error(
                "column/time split_definition requires development values"
            )
        if not refs:
            raise StrategySampleDesignV2Error(
                "available split_definition requires source_refs"
            )
        identities: dict[str, str] = {}
        for partition, values in raw_values.items():
            for item in values:
                identity = _canonical_json(item)
                prior = identities.get(identity)
                if prior is not None:
                    raise StrategySampleDesignV2Error(
                        "split values overlap between "
                        f"{prior} and {partition}"
                    )
                identities[identity] = partition
    return {
        "status": status,
        "method": method,
        "column": column,
        **{f"{name}_values": raw_values[name] for name in PARTITION_NAMES},
        "source_refs": refs,
    }


def _bounded_scalar_array(value: object, name: str) -> list[str | bool | int | float]:
    items = _array(value, name)
    if len(items) > 100:
        raise StrategySampleDesignV2Error(f"{name} exceeds item budget")
    normalized = [_json_scalar(item, f"{name}[]") for item in items]
    if len({_canonical_json(item) for item in normalized}) != len(normalized):
        raise StrategySampleDesignV2Error(f"{name} contains duplicate values")
    return normalized


def _validate_relationship_counts(
    relationship: str, membership_header: Mapping[str, Any]
) -> None:
    if relationship != "nested_same_cohort":
        return
    outside = membership_header["counts"]["relationship"][
        "risk_outside_approval"
    ]
    violating = [name for name in PARTITION_NAMES if outside[name] != 0]
    if violating:
        raise StrategySampleDesignV2Error(
            "nested_same_cohort requires risk membership to be a subset of "
            "approval membership in every partition: " + ", ".join(violating)
        )


def _population_ref(population: Mapping[str, Any]) -> dict[str, str]:
    return {
        "role": population["role"],
        "population_id": population["population_id"],
        "content_hash": population["content_hash"],
    }


def _population_refs(value: object) -> list[dict[str, str]]:
    items = _array(value, "sample design.population_refs")
    if len(items) != 2:
        raise StrategySampleDesignV2Error(
            "sample design.population_refs must contain two refs"
        )
    refs: list[dict[str, str]] = []
    for expected_role, raw in zip(POPULATION_ROLES, items, strict=True):
        obj = _object(raw, "population_ref")
        _require_exact_fields(obj, _POPULATION_REF_FIELDS, "population_ref")
        role = _enum(obj["role"], set(POPULATION_ROLES), "population_ref.role")
        if role != expected_role:
            raise StrategySampleDesignV2Error(
                "population_refs must be approval then risk"
            )
        refs.append(
            {
                "role": role,
                "population_id": _text(obj["population_id"], "population_ref.population_id"),
                "content_hash": _hash(obj["content_hash"], "population_ref.content_hash"),
            }
        )
    return refs


def _historical_score_ref(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "historical_score_id": value["historical_score_id"],
        "content_hash": value["content_hash"],
    }


def _historical_score_ref_from_value(value: object) -> dict[str, str]:
    obj = _object(value, "historical_score_ref")
    _require_exact_fields(obj, _HISTORICAL_SCORE_REF_FIELDS, "historical_score_ref")
    return {
        "historical_score_id": _text(
            obj["historical_score_id"], "historical_score_ref.historical_score_id"
        ),
        "content_hash": _hash(
            obj["content_hash"], "historical_score_ref.content_hash"
        ),
    }


def _policy_ref(value: Mapping[str, Any]) -> dict[str, str]:
    return {"policy_id": value["policy_id"], "content_hash": value["content_hash"]}


def _policy_ref_from_value(value: object) -> dict[str, str]:
    obj = _object(value, "policy_ref")
    _require_exact_fields(obj, _POLICY_REF_FIELDS, "policy_ref")
    return {
        "policy_id": _text(obj["policy_id"], "policy_ref.policy_id"),
        "content_hash": _hash(obj["content_hash"], "policy_ref.content_hash"),
    }


def _compatibility(value: object) -> dict[str, Any]:
    obj = _object(value, "sample design.compatibility")
    _require_exact_fields(obj, _COMPATIBILITY_FIELDS, "sample design.compatibility")
    if obj["maps_to"] != "risk/development":
        raise StrategySampleDesignV2Error(
            "sample design compatibility must map to risk/development"
        )
    legacy_ref = _legacy_development_ref(obj["legacy_development_ref"])
    return {
        "legacy_development_ref": legacy_ref,
        "maps_to": "risk/development",
    }


def _legacy_development_ref(value: object) -> dict[str, str]:
    try:
        return StrategySampleDesignRef.from_value(value).to_ref_dict()
    except StrategyError as exc:
        raise StrategySampleDesignV2Error(str(exc)) from exc


def _diagnostic_evidence(value: object) -> dict[str, Any]:
    obj = _object(value, "sample diagnostic.evidence")
    _require_exact_fields(obj, _DIAGNOSTIC_EVIDENCE_FIELDS, "sample diagnostic.evidence")
    return {
        field: _nullable_json_scalar(obj[field], f"sample diagnostic.evidence.{field}")
        for field in ("actual", "expected", "numerator", "denominator")
    }


def _diagnostics(
    value: object,
    *,
    sample_design: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items = _array(value, "sample-design v2 diagnostics")
    normalized = [validate_sample_diagnostic_v2(item) for item in items]
    if [item["code"] for item in normalized] != list(DIAGNOSTIC_CODES):
        raise StrategySampleDesignV2Error(
            "sample-design v2 diagnostics must contain each code in canonical order"
        )
    expected_policy_ref = _policy_ref(policy)
    if any(item["policy_ref"] != expected_policy_ref for item in normalized):
        raise StrategySampleDesignV2Error(
            "sample-design v2 diagnostic policy_ref does not match policy"
        )
    if sample_design["policy_ref"] != expected_policy_ref:
        raise StrategySampleDesignV2Error(
            "sample design policy_ref does not match diagnostics"
        )
    return normalized


def _sample_design_ref(value: object) -> dict[str, str]:
    obj = _object(value, "sample_design_ref")
    _require_exact_fields(obj, _SAMPLE_DESIGN_REF_FIELDS, "sample_design_ref")
    return {
        "sample_design_id": _text(
            obj["sample_design_id"], "sample_design_ref.sample_design_id"
        ),
        "content_hash": _hash(obj["content_hash"], "sample_design_ref.content_hash"),
    }


def _metric_definition_ref(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "metric_definition_id": value["metric_definition_id"],
        "content_hash": value["content_hash"],
    }


def _metric_definition_ref_from_value(value: object) -> dict[str, str]:
    obj = _object(value, "metric_definition_ref")
    _require_exact_fields(
        obj, _METRIC_DEFINITION_REF_FIELDS, "metric_definition_ref"
    )
    return {
        "metric_definition_id": _text(
            obj["metric_definition_id"],
            "metric_definition_ref.metric_definition_id",
        ),
        "content_hash": _hash(
            obj["content_hash"], "metric_definition_ref.content_hash"
        ),
    }


def _metric_definitions(value: object) -> list[dict[str, Any]]:
    items = _array(value, "metric_definitions")
    if not items:
        raise StrategySampleDesignV2Error("metric_definitions must not be empty")
    normalized = [validate_metric_definition_v2(item) for item in items]
    normalized.sort(key=lambda item: item["metric_key"])
    expected = build_metric_definitions_v2(
        producer_version=normalized[0]["producer_version"]
    )
    if normalized != expected:
        raise StrategySampleDesignV2Error(
            "metric_definitions do not match the fixed V2 sample metric contract"
        )
    return normalized


def _metric_observations(
    value: object,
    *,
    sample_design: Mapping[str, Any],
    populations: Sequence[Mapping[str, Any]],
    membership_header: Mapping[str, Any],
    metric_definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    items = _array(value, "metric_observations")
    if not items:
        raise StrategySampleDesignV2Error("metric_observations must not be empty")
    normalized = [
        validate_metric_observation_v2(
            item, metric_definitions=metric_definitions
        )
        for item in items
    ]
    definition_by_id = {
        item["metric_definition_id"]: item for item in metric_definitions
    }
    expected_ref = {
        "sample_design_id": sample_design["sample_design_id"],
        "content_hash": sample_design["content_hash"],
    }
    if any(item["sample_design_ref"] != expected_ref for item in normalized):
        raise StrategySampleDesignV2Error(
            "metric observation sample_design_ref does not match sample design"
        )
    normalized.sort(
        key=lambda item: (
            item["population"],
            item["partition"],
            definition_by_id[
                item["metric_definition_ref"]["metric_definition_id"]
            ]["metric_key"],
            item["observation_id"],
        )
    )
    identities = [
        (
            item["population"],
            item["partition"],
            definition_by_id[
                item["metric_definition_ref"]["metric_definition_id"]
            ]["metric_key"],
        )
        for item in normalized
    ]
    if len(identities) != len(set(identities)):
        raise StrategySampleDesignV2Error(
            "metric observations contain duplicate metric slices"
        )
    expected_identities = {
        (role, partition, definition["metric_key"])
        for role in POPULATION_ROLES
        for partition in ("overall", *PARTITION_NAMES)
        for definition in metric_definitions
    }
    if set(identities) != expected_identities:
        raise StrategySampleDesignV2Error(
            "metric observations must cover every fixed metric for approval/risk "
            "overall and all three partitions"
        )
    population_by_role = {item["role"]: item for item in populations}
    required_sources = {
        (
            "dataset",
            membership_header["dataset_ref"]["dataset_id"],
            membership_header["dataset_ref"]["content_hash"],
        ),
        (
            "sample_membership",
            membership_header["membership_id"],
            membership_header["content_hash"],
        ),
        (
            "sample_design",
            sample_design["sample_design_id"],
            sample_design["content_hash"],
        ),
    }
    for item in normalized:
        if item["partition"] == "overall":
            expected_sample_count = population_by_role[item["population"]][
                "total_count"
            ]
        else:
            expected_sample_count = membership_header["counts"][item["population"]][
                item["partition"]
            ]
        if item["sample_count"] != expected_sample_count:
            raise StrategySampleDesignV2Error(
                "metric observation sample_count does not match membership"
            )
        actual_sources = {
            (ref["kind"], ref["ref_id"], ref["content_hash"])
            for ref in item["source_refs"]
        }
        if not required_sources.issubset(actual_sources):
            raise StrategySampleDesignV2Error(
                "metric observation source_refs do not bind dataset, membership, "
                "and sample design"
            )
    return normalized


def _validate_observation_conservation(bundle: Mapping[str, Any]) -> None:
    definition_keys = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for observation in bundle["metric_observations"]:
        key = (
            observation["population"],
            observation["partition"],
        )
        metric_key = definition_keys[
            observation["metric_definition_ref"]["metric_definition_id"]
        ]
        grouped.setdefault(key, {})[metric_key] = observation

    target_resolved = bundle["sample_design"]["target_selector"]["status"] == "resolved"
    maturity = next(
        item for item in bundle["populations"] if item["role"] == "risk"
    )["maturity_evidence"]
    for (role, partition), metrics in grouped.items():
        sample_count = metrics["population_count"]["sample_count"]
        if any(item["sample_count"] != sample_count for item in metrics.values()):
            raise StrategySampleDesignV2Error(
                "metric sample_count is inconsistent within a population slice"
            )
        _require_count_observation(
            metrics["population_count"],
            expected_value=sample_count,
            denominator=sample_count,
            name=f"{role}/{partition} population_count",
        )
        if not target_resolved:
            for metric_key in (
                "labeled_count",
                "label_coverage",
                "bad_count",
                "bad_rate",
            ):
                _require_observation_status(
                    metrics[metric_key],
                    "unavailable",
                    f"{role}/{partition} {metric_key}",
                )
            continue

        labeled = _require_bounded_count_observation(
            metrics["labeled_count"],
            upper_bound=sample_count,
            denominator=sample_count,
            name=f"{role}/{partition} labeled_count",
        )
        if sample_count == 0:
            _require_observation_status(
                metrics["label_coverage"],
                "insufficient_data",
                f"{role}/{partition} label_coverage",
            )
        else:
            _require_ratio_observation(
                metrics["label_coverage"],
                numerator=labeled,
                denominator=sample_count,
                name=f"{role}/{partition} label_coverage",
            )

        expected_bad_status = "present"
        if role == "risk":
            if maturity["status"] == "not_matured":
                expected_bad_status = "not_matured"
            elif maturity["status"] in {"unknown", "unavailable"}:
                expected_bad_status = "unavailable"
        if expected_bad_status != "present":
            _require_observation_status(
                metrics["bad_count"],
                expected_bad_status,
                f"{role}/{partition} bad_count",
            )
            _require_observation_status(
                metrics["bad_rate"],
                expected_bad_status,
                f"{role}/{partition} bad_rate",
            )
            continue
        bad = _require_bounded_count_observation(
            metrics["bad_count"],
            upper_bound=labeled,
            denominator=labeled,
            name=f"{role}/{partition} bad_count",
        )
        if labeled == 0:
            _require_observation_status(
                metrics["bad_rate"],
                "insufficient_data",
                f"{role}/{partition} bad_rate",
            )
        else:
            _require_ratio_observation(
                metrics["bad_rate"],
                numerator=bad,
                denominator=labeled,
                name=f"{role}/{partition} bad_rate",
            )

    for role in POPULATION_ROLES:
        overall = grouped[(role, "overall")]
        splits = [grouped[(role, partition)] for partition in PARTITION_NAMES]
        for metric_key in ("population_count", "labeled_count", "bad_count"):
            overall_metric = overall[metric_key]
            split_metrics = [item[metric_key] for item in splits]
            if overall_metric["status"] != "present":
                if any(
                    item["status"] != overall_metric["status"]
                    for item in split_metrics
                ):
                    raise StrategySampleDesignV2Error(
                        f"{role} split {metric_key} statuses are inconsistent"
                    )
                continue
            if any(item["status"] != "present" for item in split_metrics):
                raise StrategySampleDesignV2Error(
                    f"{role} split {metric_key} statuses are inconsistent"
                )
            if int(overall_metric["value"]) != sum(
                int(item["value"]) for item in split_metrics
            ):
                raise StrategySampleDesignV2Error(
                    f"{role} split {metric_key} does not conserve overall"
                )

    risk_overall = grouped[("risk", "overall")]
    if maturity["status"] in {"confirmed_matured", "not_matured"}:
        if risk_overall["labeled_count"]["value"] != maturity["labeled_count"]:
            raise StrategySampleDesignV2Error(
                "risk labeled_count does not match maturity evidence"
            )


def _require_observation_status(
    observation: Mapping[str, Any], expected: str, name: str
) -> None:
    if observation["status"] != expected or any(
        observation[field] is not None
        for field in ("value", "numerator", "denominator")
    ):
        raise StrategySampleDesignV2Error(
            f"{name} must have status {expected} with null operands"
        )


def _require_count_observation(
    observation: Mapping[str, Any],
    *,
    expected_value: int,
    denominator: int,
    name: str,
) -> None:
    if (
        observation["status"] != "present"
        or observation["value"] != expected_value
        or observation["numerator"] != expected_value
        or observation["denominator"] != denominator
    ):
        raise StrategySampleDesignV2Error(f"{name} is inconsistent")


def _require_bounded_count_observation(
    observation: Mapping[str, Any],
    *,
    upper_bound: int,
    denominator: int,
    name: str,
) -> int:
    value = observation["value"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrategySampleDesignV2Error(f"{name} must be an integer count")
    if value < 0 or value > upper_bound:
        raise StrategySampleDesignV2Error(f"{name} exceeds its bound denominator")
    _require_count_observation(
        observation,
        expected_value=value,
        denominator=denominator,
        name=name,
    )
    return value


def _require_ratio_observation(
    observation: Mapping[str, Any],
    *,
    numerator: int,
    denominator: int,
    name: str,
) -> None:
    if (
        observation["status"] != "present"
        or observation["numerator"] != numerator
        or observation["denominator"] != denominator
        or not math.isclose(
            float(observation["value"]),
            numerator / denominator,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise StrategySampleDesignV2Error(f"{name} is inconsistent")


def _resolved_membership_masks(
    value: object, membership_header: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    obj = _object(value, "resolved membership masks")
    _require_exact_fields(obj, frozenset(MEMBERSHIP_MASK_ORDER), "resolved membership masks")
    row_count = membership_header["row_count"]
    masks: dict[str, np.ndarray] = {}
    for name in MEMBERSHIP_MASK_ORDER:
        raw = obj[name]
        if not isinstance(raw, np.ndarray) or raw.ndim != 1 or raw.dtype.kind != "b":
            raise StrategySampleDesignV2Error(
                f"resolved membership mask {name} must be a boolean vector"
            )
        if len(raw) != row_count:
            raise StrategySampleDesignV2Error(
                f"resolved membership mask {name} has wrong row count"
            )
        masks[name] = np.ascontiguousarray(raw, dtype=np.bool_)
    for role in POPULATION_ROLES:
        role_masks = [masks[f"{role}/{name}"] for name in PARTITION_NAMES]
        if bool(
            np.any(
                (role_masks[0] & role_masks[1])
                | (role_masks[0] & role_masks[2])
                | (role_masks[1] & role_masks[2])
            )
        ):
            raise StrategySampleDesignV2Error(
                f"resolved {role} membership masks overlap"
            )
        counts = {
            name: int(np.count_nonzero(masks[f"{role}/{name}"]))
            for name in PARTITION_NAMES
        }
        if counts != {
            name: membership_header["counts"][role][name]
            for name in PARTITION_NAMES
        }:
            raise StrategySampleDesignV2Error(
                f"resolved {role} membership counts do not match header"
            )
    within = {
        partition: int(
            np.count_nonzero(
                masks[f"risk/{partition}"] & masks[f"approval/{partition}"]
            )
        )
        for partition in PARTITION_NAMES
    }
    outside = {
        partition: int(
            np.count_nonzero(
                masks[f"risk/{partition}"] & ~masks[f"approval/{partition}"]
            )
        )
        for partition in PARTITION_NAMES
    }
    expected_relationship = {
        "risk_within_approval": {**within, "total": sum(within.values())},
        "risk_outside_approval": {**outside, "total": sum(outside.values())},
    }
    if expected_relationship != membership_header["counts"]["relationship"]:
        raise StrategySampleDesignV2Error(
            "resolved relationship counts do not match membership header"
        )
    return masks


def _population_union(masks: Mapping[str, np.ndarray], role: str) -> np.ndarray:
    return np.logical_or.reduce(
        [masks[f"{role}/{partition}"] for partition in PARTITION_NAMES]
    )


def _dataset_ref(value: object) -> dict[str, str]:
    obj = _object(value, "dataset_ref")
    _require_exact_fields(obj, _DATASET_REF_FIELDS, "dataset_ref")
    return {
        "dataset_id": _text(obj["dataset_id"], "dataset_ref.dataset_id"),
        "content_hash": _hash(obj["content_hash"], "dataset_ref.content_hash"),
    }


def _source_ref(value: object, name: str) -> dict[str, str]:
    obj = _object(value, name)
    _require_exact_fields(obj, _SOURCE_REF_FIELDS, name)
    return {
        "kind": _text(obj["kind"], f"{name}.kind"),
        "ref_id": _text(obj["ref_id"], f"{name}.ref_id"),
        "content_hash": _hash(obj["content_hash"], f"{name}.content_hash"),
    }


def _source_refs(value: object, name: str) -> list[dict[str, str]]:
    items = _array(value, name)
    refs = [_source_ref(item, f"{name} item") for item in items]
    refs.sort(key=lambda item: (item["kind"], item["ref_id"], item["content_hash"]))
    identities = [(item["kind"], item["ref_id"]) for item in refs]
    if len(identities) != len(set(identities)):
        raise StrategySampleDesignV2Error(f"{name} contains duplicate refs")
    return refs


def _address_object(body: Mapping[str, Any], id_field: str) -> dict[str, Any]:
    object_id = _ID_PREFIXES[id_field] + _sha256(_canonical_json(body))[:24]
    without_hash = {**body, id_field: object_id}
    return {**without_hash, "content_hash": _sha256(_canonical_json(without_hash))}


def _validate_addressed_object(
    original: Mapping[str, Any],
    normalized_body: Mapping[str, Any],
    id_field: str,
) -> dict[str, Any]:
    object_id = original[id_field]
    if (
        not isinstance(object_id, str)
        or _ID_PATTERNS[id_field].fullmatch(object_id) is None
    ):
        raise StrategySampleDesignV2Error(f"{id_field} is invalid")
    expected_id = _ID_PREFIXES[id_field] + _sha256(_canonical_json(normalized_body))[:24]
    if not hmac.compare_digest(object_id, expected_id):
        raise StrategySampleDesignV2Error(f"{id_field} does not match content")
    content_hash = _hash(original["content_hash"], "content_hash")
    without_hash = {**normalized_body, id_field: object_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(content_hash, expected_hash):
        raise StrategySampleDesignV2Error("content_hash does not match content")
    return {**without_hash, "content_hash": content_hash}


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategySampleDesignV2Error(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StrategySampleDesignV2Error(f"{name} must be an array")
    return list(value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise StrategySampleDesignV2Error(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise StrategySampleDesignV2Error(
            f"{name} fields are invalid ({'; '.join(details)})"
        )


def _require_producer_versions(
    producer: str,
    values: Sequence[Mapping[str, Any]],
    name: str,
) -> None:
    if any(item["producer_version"] != producer for item in values):
        raise StrategySampleDesignV2Error(f"{name} producer_version is inconsistent")


def _preflight_json_tree(value: object, name: str) -> None:
    stack: list[tuple[object, int, bool]] = [(value, 1, False)]
    active: set[int] = set()
    nodes = 0
    while stack:
        current, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(current))
            continue
        nodes += 1
        if nodes > MAX_SAMPLE_DESIGN_V2_JSON_NODES:
            raise StrategySampleDesignV2Error(f"{name} exceeds JSON node budget")
        if depth > MAX_SAMPLE_DESIGN_V2_JSON_DEPTH:
            raise StrategySampleDesignV2Error(f"{name} exceeds JSON depth budget")
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in active:
                raise StrategySampleDesignV2Error(
                    f"{name} contains a cyclic container"
                )
            active.add(identity)
            stack.append((current, depth, True))
            stack.extend(
                (child, depth + 1, False) for child in current.values()
            )
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            identity = id(current)
            if identity in active:
                raise StrategySampleDesignV2Error(
                    f"{name} contains a cyclic container"
                )
            active.add(identity)
            stack.append((current, depth, True))
            stack.extend((child, depth + 1, False) for child in current)
        elif current is None or isinstance(current, (str, bool, int, float)):
            if isinstance(current, float) and not math.isfinite(current):
                raise StrategySampleDesignV2Error(
                    f"{name} contains a non-finite number"
                )
        else:
            raise StrategySampleDesignV2Error(
                f"{name} contains unsupported JSON value {type(current).__name__}"
            )


def _canonical_json(value: object) -> str:
    _preflight_json_tree(value, "canonical JSON")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise StrategySampleDesignV2Error("value is not canonical JSON") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategySampleDesignV2Error(
                f"sample-design v2 bundle JSON has duplicate key: {key}"
            )
        result[key] = value
    return result


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrategySampleDesignV2Error(f"{name} must be non-empty canonical text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _producer_version(value: object) -> str:
    return _text(value, "producer_version")


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategySampleDesignV2Error(
            f"{name} must be a lowercase SHA-256 hash"
        )
    return value


def _enum(value: object, allowed: set[str] | frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise StrategySampleDesignV2Error(
            f"{name} must be one of {', '.join(sorted(allowed))}"
        )
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategySampleDesignV2Error(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise StrategySampleDesignV2Error(f"{name} must be a positive integer")
    return result


def _ratio(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise StrategySampleDesignV2Error(f"{name} must be a finite ratio in [0, 1]")
    return float(value)


def _binary_target_value(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) not in {0.0, 1.0}
    ):
        raise StrategySampleDesignV2Error(
            f"{name} must be numeric 0 or 1 (boolean is not allowed)"
        )
    return int(value)


def _optional_number(value: object, name: str) -> int | float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise StrategySampleDesignV2Error(f"{name} must be a finite number or null")
    return value


def _metric_value(value: object, unit: str, name: str) -> int | float | bool:
    if unit == "boolean":
        if not isinstance(value, bool):
            raise StrategySampleDesignV2Error(f"{name} must be a boolean")
        return value
    number = _optional_number(value, name)
    if number is None:
        raise StrategySampleDesignV2Error(f"{name} must be present")
    if unit in {"count", "days"} and (
        isinstance(number, bool) or not isinstance(number, int) or number < 0
    ):
        raise StrategySampleDesignV2Error(
            f"{name} must be a non-negative integer for unit {unit}"
        )
    if unit == "ratio" and not 0 <= float(number) <= 1:
        raise StrategySampleDesignV2Error(f"{name} ratio must be in [0, 1]")
    return number


def _json_scalar(value: object, name: str) -> str | bool | int | float:
    if value is None or not isinstance(value, (str, bool, int, float)):
        raise StrategySampleDesignV2Error(f"{name} must be a non-null JSON scalar")
    if isinstance(value, str):
        return _text(value, name)
    if isinstance(value, float) and not math.isfinite(value):
        raise StrategySampleDesignV2Error(f"{name} must be finite")
    return value


def _nullable_json_scalar(
    value: object, name: str
) -> str | bool | int | float | None:
    if value is None:
        return None
    return _json_scalar(value, name)


def _iso_date(value: object, name: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise StrategySampleDesignV2Error(f"{name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise StrategySampleDesignV2Error(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise StrategySampleDesignV2Error(f"{name} must be a canonical ISO date")
    return value


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "DIAGNOSTIC_CATEGORIES",
    "DIAGNOSTIC_CODES",
    "DIAGNOSTIC_STATUSES",
    "HISTORICAL_SCORE_DIRECTIONS",
    "HISTORICAL_SCORE_STATUSES",
    "MAX_SAMPLE_DESIGN_V2_JSON_BYTES",
    "MAX_SAMPLE_DESIGN_V2_JSON_DEPTH",
    "MAX_SAMPLE_DESIGN_V2_JSON_NODES",
    "METRIC_OBSERVATION_V2_STATUSES",
    "METRIC_OBSERVATION_V2_UNITS",
    "PARTITION_NAMES",
    "POPULATION_ROLES",
    "SAMPLE_RELATIONSHIPS",
    "STRATEGY_HISTORICAL_SCORE_V2_SCHEMA_VERSION",
    "STRATEGY_METRIC_OBSERVATION_V2_SCHEMA_VERSION",
    "STRATEGY_SAMPLE_DESIGN_V2_BUNDLE_SCHEMA_VERSION",
    "STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION",
    "STRATEGY_SAMPLE_DESIGN_V2_SCHEMA_VERSION",
    "STRATEGY_SAMPLE_DIAGNOSTIC_V2_SCHEMA_VERSION",
    "STRATEGY_SAMPLE_POLICY_V2_SCHEMA_VERSION",
    "STRATEGY_SAMPLE_POPULATION_V2_SCHEMA_VERSION",
    "StrategySampleDesignV2Error",
    "build_historical_score_v2",
    "build_metric_observation_v2",
    "build_sample_design_policy_v2",
    "build_sample_diagnostic_v2",
    "build_sample_population_v2",
    "build_strategy_sample_design_v2",
    "build_strategy_sample_design_v2_bundle",
    "build_target_selector_v2",
    "canonical_strategy_sample_design_v2_bundle_json",
    "evaluate_sample_design_v2_diagnostics",
    "strategy_sample_design_v2_bundle_from_json",
    "validate_historical_score_v2",
    "validate_metric_observation_v2",
    "validate_sample_design_policy_v2",
    "validate_sample_diagnostic_v2",
    "validate_sample_population_v2",
    "validate_strategy_sample_design_v2",
    "validate_strategy_sample_design_v2_bundle",
    "validate_target_selector_v2",
]
