"""Pure, strict, immutable candidate assets for complete 2D cross matrices.

This kernel consumes one verified univariate CandidateEvidence parent and only
primary per-cell measurements. It derives every rule, metric, id, and hash. It
does not read frames, persist artifacts, choose actions, interact with a Pool,
or claim validation, selection, admission, adoption, or deployment.
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

from marvis.feature.iv import _smoothed_woe_iv
from marvis.packs.strategy.candidate_evidence import (
    CandidateEvidenceError,
    validate_candidate_evidence,
)
from marvis.packs.strategy.candidate_fragment import (
    CandidateFragmentError,
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.dsl import canonicalize_expression, semantic_expression_key
from marvis.packs.strategy.errors import StrategyError


CROSS_MATRIX_CANDIDATE_ASSET_SCHEMA_VERSION = "strategy.cross-matrix-candidate-asset.v1"
CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION = (
    "strategy.cross-matrix-candidate-asset.v2"
)
CROSS_MATRIX_MEASUREMENT_SCHEMA_VERSION = "strategy.cross-matrix-measurement.v1"
CROSS_MATRIX_CANDIDATE_ASSET_TYPE = "cross_matrix"
CROSS_MATRIX_CANDIDATE_ASSET_PRODUCER_VERSION = "strategy.cross-matrix-candidate-asset/1"
CROSS_MATRIX_CANDIDATE_ASSET_V2_PRODUCER_VERSION = (
    "strategy.cross-matrix-candidate-asset/2"
)
_UNIVARIATE_V1_SCHEMA_VERSION = "univariate-analysis-result.v1"
_UNIVARIATE_V2_SCHEMA_VERSION = "univariate-analysis-result.v2"
_UNIVARIATE_V1_PRODUCER_VERSION = "strategy.univariate-candidate/1"
_UNIVARIATE_V2_PRODUCER_VERSION = "strategy.univariate-candidate/2"
_MAX_AXIS_BINS = 20

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_LIFECYCLE = {
    "candidate_stage": "development",
    "observation_stage": "backtested",
    "validation_status": "unvalidated",
}
_TOP_FIELDS = frozenset(
    {
        "schema_version", "asset_type", "lifecycle", "parent", "sample_identity",
        "axes", "measurement", "budget", "matrix", "summary",
        "candidate_evidence", "producer_version", "asset_id", "asset_hash",
    }
)
_BODY_FIELDS = _TOP_FIELDS - {"asset_id", "asset_hash"}
_PARENT_V1_FIELDS = frozenset(
    {
        "candidate_id", "evidence_hash", "identity", "target_col", "row_count",
        "target_definition", "smoothing",
    }
)
_PARENT_V2_FIELDS = _PARENT_V1_FIELDS | frozenset({"analysis_schema_version"})
_PARENT_IDENTITY_FIELDS = frozenset(
    {
        "task_id", "dataset_id", "dataset_content_hash", "workspace_revision",
        "workspace_generation", "semantic_mapping_hash",
    }
)
_SAMPLE_FIELDS = _PARENT_IDENTITY_FIELDS | frozenset(
    {"sample_context_hash", "target_col", "row_count"}
)
_AXIS_SPEC_FIELDS = frozenset({"feature", "method"})
_AXIS_V1_FIELDS = frozenset(
    {"position", "feature", "method", "row_count", "bins", "axis_id", "axis_hash"}
)
_AXIS_V2_FIELDS = _AXIS_V1_FIELDS | frozenset(
    {"manual_breakpoints", "parent_evidence_hash"}
)
_BIN_FIELDS = frozenset(
    {
        "position", "source_index", "source_bin_id", "kind", "condition",
        "semantic_key", "count", "good", "bad", "amount_evidence", "bin_id", "bin_hash",
    }
)
_AMOUNT_FIELDS = frozenset({"loan_amount", "overdue_amount", "paired"})
_AMOUNT_OBS_FIELDS = frozenset({"status", "covered_count", "value"})
_PAIRED_OBS_FIELDS = frozenset(
    {"status", "covered_count", "loan_value", "overdue_value"}
)
_AMOUNT_EVIDENCE_FIELDS = frozenset(
    {"status", "covered_count", "value", "reason"}
)
_PAIRED_EVIDENCE_FIELDS = frozenset({"status", "covered_count", "value", "reason"})
_MEASUREMENT_BODY_FIELDS = frozenset(
    {"schema_version", "sample_context_hash", "population_count", "good", "bad", "cells"}
)
_MEASUREMENT_FIELDS = _MEASUREMENT_BODY_FIELDS | {"measurement_hash"}
_PRIMARY_CELL_FIELDS = frozenset(
    {"row_source_bin_id", "column_source_bin_id", "count", "good", "bad", "amounts"}
)
_BUDGET_FIELDS = frozenset({"unit", "limit", "required", "truncated"})
_MATRIX_FIELDS = frozenset(
    {"row_bin_count", "column_bin_count", "cell_count", "cells", "matrix_hash"}
)
_CELL_FIELDS = frozenset(
    {"row_bin_id", "column_bin_id", "cell_id", "cell_hash", "rule", "effect"}
)
_RULE_FIELDS = frozenset({"rule_id", "rule_hash", "condition", "semantic_key"})
_EFFECT_FIELDS = frozenset(
    {
        "effect_id", "effect_hash", "count", "good", "bad", "share", "bad_rate",
        "lift", "woe", "iv_contribution", "amount_metrics",
    }
)
_SUMMARY_FIELDS = frozenset(
    {"count", "good", "bad", "bad_rate", "total_iv", "amount_metrics", "summary_hash"}
)
_DERIVED_AMOUNT_FIELDS = frozenset(
    {"status", "covered_count", "coverage_rate", "value", "reason"}
)
_EVIDENCE_FIELDS = frozenset({"candidate_id", "evidence_hash"})


class CrossMatrixCandidateAssetError(StrategyError):
    """A cross-matrix input or immutable asset failed closed."""


def build_cross_matrix_candidate_asset(
    parent_evidence: Mapping[str, Any], *, row_axis: Mapping[str, Any],
    column_axis: Mapping[str, Any], sample_identity: Mapping[str, Any],
    measurement: Mapping[str, Any], budget: int,
    producer_version: str | None = None,
) -> dict[str, Any]:
    """Build a complete Cartesian asset from strict primary measurements.

    Axis specs have exact fields ``feature`` and ``method``. Measurement cells
    have exact fields ``row_source_bin_id``, ``column_source_bin_id``,
    ``count``, ``good``, ``bad``, and ``amounts``. Amount observations are
    status/coverage/raw-sum facts; all rates, WOE/IV, rules, ids, and hashes are
    derived here. Budget is measured in Cartesian cells and never truncates.
    """

    parent = _parent_from_evidence(parent_evidence)
    schema_version = (
        CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION
        if parent.get("analysis_schema_version") == _UNIVARIATE_V2_SCHEMA_VERSION
        else CROSS_MATRIX_CANDIDATE_ASSET_SCHEMA_VERSION
    )
    row = _axis_from_evidence(
        parent_evidence,
        row_axis,
        position="row",
        schema_version=schema_version,
    )
    column = _axis_from_evidence(
        parent_evidence,
        column_axis,
        position="column",
        schema_version=schema_version,
    )
    if row["feature"] == column["feature"]:
        raise CrossMatrixCandidateAssetError("cross-matrix axes must use different features")
    try:
        expected_sample_hash = sample_context_hash_from_candidate_evidence(
            parent_evidence
        )
    except CandidateFragmentError as exc:
        raise CrossMatrixCandidateAssetError(
            f"parent sample context is invalid: {exc}"
        ) from exc
    sample = _sample_identity(
        sample_identity, parent=parent, expected_sample_hash=expected_sample_hash
    )
    axes = [row, column]
    measured = _measurement(measurement, axes=axes, sample=sample, stored=False)
    budget_value = _budget(budget, required=len(row["bins"]) * len(column["bins"]))
    matrix = _derive_matrix(axes, measured, parent=parent, sample=sample)
    summary = _derive_summary(measured, matrix=matrix)
    core = {
        "schema_version": schema_version,
        "asset_type": CROSS_MATRIX_CANDIDATE_ASSET_TYPE,
        "lifecycle": dict(_LIFECYCLE),
        "parent": parent,
        "sample_identity": sample,
        "axes": axes,
        "measurement": measured,
        "budget": budget_value,
        "matrix": matrix,
        "summary": summary,
        "producer_version": _producer(
            (
                CROSS_MATRIX_CANDIDATE_ASSET_V2_PRODUCER_VERSION
                if schema_version == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION
                else CROSS_MATRIX_CANDIDATE_ASSET_PRODUCER_VERSION
            )
            if producer_version is None
            else producer_version,
            schema_version=schema_version,
        ),
    }
    evidence = _derive_candidate_evidence(core)
    body = {**core, "candidate_evidence": evidence}
    asset_id = _stable_id("candidate-asset", body)
    without_hash = {**body, "asset_id": asset_id}
    asset_hash = _sha256(_canonical_json(without_hash))
    return validate_cross_matrix_candidate_asset({**without_hash, "asset_hash": asset_hash})


def validate_cross_matrix_candidate_asset(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact fields and deterministically rebuild every derived value."""

    if not isinstance(payload, Mapping):
        raise CrossMatrixCandidateAssetError("cross-matrix candidate asset must be an object")
    _exact(payload, _TOP_FIELDS, "cross-matrix candidate asset")
    asset_id = _identifier(payload["asset_id"], "asset_id", prefix="candidate-asset")
    asset_hash = _hash(payload["asset_hash"], "asset_hash")
    body = _normalize_body({key: payload[key] for key in payload if key not in {"asset_id", "asset_hash"}})
    expected_id = _stable_id("candidate-asset", body)
    if not hmac.compare_digest(asset_id, expected_id):
        raise CrossMatrixCandidateAssetError("asset_id does not match canonical cross-matrix asset")
    without_hash = {**body, "asset_id": asset_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(asset_hash, expected_hash):
        raise CrossMatrixCandidateAssetError("asset_hash does not match canonical cross-matrix asset")
    return {**without_hash, "asset_hash": asset_hash}


def canonical_cross_matrix_candidate_asset_json(payload: Mapping[str, Any]) -> str:
    return _canonical_json(validate_cross_matrix_candidate_asset(payload))


def parse_cross_matrix_candidate_asset_json(raw: str | bytes | bytearray) -> dict[str, Any]:
    """Parse canonical JSON with duplicate-key rejection."""

    if not isinstance(raw, str | bytes | bytearray):
        raise CrossMatrixCandidateAssetError("cross-matrix candidate JSON must be text or bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_object_no_duplicates)
    except CrossMatrixCandidateAssetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CrossMatrixCandidateAssetError(f"cross-matrix candidate is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CrossMatrixCandidateAssetError("cross-matrix candidate JSON must contain an object")
    return validate_cross_matrix_candidate_asset(value)


def rebuild_cross_matrix_candidate_asset(
    payload: Mapping[str, Any], parent_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild an asset against an independently loaded exact parent evidence."""

    asset = validate_cross_matrix_candidate_asset(payload)
    parent = _parent_from_evidence(parent_evidence)
    if asset["parent"] != parent:
        raise CrossMatrixCandidateAssetError("asset does not bind the exact parent candidate evidence")
    raw_measurement = {
        key: asset["measurement"][key]
        for key in _MEASUREMENT_BODY_FIELDS
    }
    rebuilt = build_cross_matrix_candidate_asset(
        parent_evidence,
        row_axis={key: asset["axes"][0][key] for key in _AXIS_SPEC_FIELDS},
        column_axis={key: asset["axes"][1][key] for key in _AXIS_SPEC_FIELDS},
        sample_identity=asset["sample_identity"],
        measurement=raw_measurement,
        budget=asset["budget"]["limit"],
        producer_version=asset["producer_version"],
    )
    if rebuilt != asset:
        raise CrossMatrixCandidateAssetError("asset does not rebuild from exact parent and measurement")
    return asset


def _normalize_body(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact(value, _BODY_FIELDS, "cross-matrix candidate body")
    schema_version = value["schema_version"]
    if schema_version not in {
        CROSS_MATRIX_CANDIDATE_ASSET_SCHEMA_VERSION,
        CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION,
    }:
        raise CrossMatrixCandidateAssetError("invalid cross-matrix schema_version")
    if value["asset_type"] != CROSS_MATRIX_CANDIDATE_ASSET_TYPE:
        raise CrossMatrixCandidateAssetError("asset_type must be cross_matrix")
    lifecycle = _lifecycle(value["lifecycle"])
    parent = _parent_reference(value["parent"], schema_version=schema_version)
    sample = _sample_identity(value["sample_identity"], parent=parent)
    axes = _axes(value["axes"], parent=parent, schema_version=schema_version)
    if axes[0]["feature"] == axes[1]["feature"]:
        raise CrossMatrixCandidateAssetError("cross-matrix axes must use different features")
    measurement = _measurement(value["measurement"], axes=axes, sample=sample, stored=True)
    budget = _budget_object(value["budget"], required=len(axes[0]["bins"]) * len(axes[1]["bins"]))
    matrix = _normalize_matrix(value["matrix"], axes=axes, measurement=measurement, parent=parent, sample=sample)
    summary = _normalize_summary(value["summary"], measurement=measurement, matrix=matrix)
    producer = _producer(value["producer_version"], schema_version=schema_version)
    core = {
        "schema_version": schema_version,
        "asset_type": CROSS_MATRIX_CANDIDATE_ASSET_TYPE,
        "lifecycle": lifecycle,
        "parent": parent,
        "sample_identity": sample,
        "axes": axes,
        "measurement": measurement,
        "budget": budget,
        "matrix": matrix,
        "summary": summary,
        "producer_version": producer,
    }
    evidence = _candidate_evidence(value["candidate_evidence"], core=core)
    return {**core, "candidate_evidence": evidence}


def _parent_from_evidence(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("parent_evidence must be an object")
    try:
        evidence = validate_candidate_evidence(value)
    except CandidateEvidenceError as exc:
        raise CrossMatrixCandidateAssetError(f"parent CandidateEvidence is invalid: {exc}") from exc
    if evidence["candidate_type"] != "univariate":
        raise CrossMatrixCandidateAssetError("parent candidate_type must be univariate")
    analysis = evidence["analysis"]
    if not isinstance(analysis, Mapping) or analysis.get("schema_version") not in {
        _UNIVARIATE_V1_SCHEMA_VERSION,
        _UNIVARIATE_V2_SCHEMA_VERSION,
    }:
        raise CrossMatrixCandidateAssetError(
            "parent analysis must be a supported univariate analysis result"
        )
    analysis_schema_version = str(analysis["schema_version"])
    target = analysis.get("target_definition")
    if target != {"good": 0, "bad": 1}:
        raise CrossMatrixCandidateAssetError("parent target_definition must be exact binary good=0/bad=1")
    parameters = analysis.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CrossMatrixCandidateAssetError("parent analysis.parameters must be an object")
    _require_parent_univariate_version_contract(
        evidence,
        analysis=analysis,
        analysis_schema_version=analysis_schema_version,
    )
    smoothing = _positive_number(parameters.get("smoothing"), "parent smoothing")
    identity = _parent_identity(evidence["identity"])
    parent = {
        "candidate_id": _identifier(evidence["candidate_id"], "parent candidate_id", prefix="candidate"),
        "evidence_hash": _hash(evidence["evidence_hash"], "parent evidence_hash"),
        "identity": identity,
        "target_col": _text(analysis.get("target"), "parent analysis.target"),
        "row_count": _positive_int(analysis.get("row_count"), "parent analysis.row_count"),
        "target_definition": {"good": 0, "bad": 1},
        "smoothing": smoothing,
    }
    if analysis_schema_version == _UNIVARIATE_V2_SCHEMA_VERSION:
        parent["analysis_schema_version"] = analysis_schema_version
    return parent


def _require_parent_univariate_version_contract(
    evidence: Mapping[str, Any],
    *,
    analysis: Mapping[str, Any],
    analysis_schema_version: str,
) -> None:
    features = analysis.get("features")
    if not _sequence(features):
        raise CrossMatrixCandidateAssetError(
            "parent analysis.features must be a non-empty array"
        )
    manual_rows: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for feature_row in features:
        if not isinstance(feature_row, Mapping):
            raise CrossMatrixCandidateAssetError(
                "parent analysis.features must contain objects"
            )
        feature = _text(feature_row.get("feature"), "parent analysis feature")
        methods = feature_row.get("methods")
        if not _sequence(methods):
            raise CrossMatrixCandidateAssetError(
                f"parent analysis feature {feature!r} has no methods"
            )
        matches = [
            method
            for method in methods
            if isinstance(method, Mapping) and method.get("method") == "manual"
        ]
        if len(matches) > 1:
            raise CrossMatrixCandidateAssetError(
                f"parent analysis feature {feature!r} has duplicate manual methods"
            )
        if matches:
            manual_rows[feature] = (feature_row, matches[0])
    analysis_parameters = analysis["parameters"]
    generation_parameters = evidence["generation"].get("parameters")
    if not isinstance(generation_parameters, Mapping):
        raise CrossMatrixCandidateAssetError(
            "parent generation.parameters must be an object"
        )
    expected_producer_version = (
        _UNIVARIATE_V2_PRODUCER_VERSION
        if analysis_schema_version == _UNIVARIATE_V2_SCHEMA_VERSION
        else _UNIVARIATE_V1_PRODUCER_VERSION
    )
    if (
        generation_parameters.get("analysis_schema_version")
        != analysis_schema_version
        or evidence.get("producer_version") != expected_producer_version
    ):
        raise CrossMatrixCandidateAssetError(
            "parent analysis schema and producer versions do not match"
        )
    if analysis_schema_version == _UNIVARIATE_V1_SCHEMA_VERSION:
        if (
            manual_rows
            or "manual_breakpoints" in analysis_parameters
            or "manual_breakpoints" in generation_parameters
        ):
            raise CrossMatrixCandidateAssetError(
                "v1 parent analysis cannot carry manual binning evidence"
            )
        return

    analysis_mapping = analysis_parameters.get("manual_breakpoints")
    generation_mapping = generation_parameters.get("manual_breakpoints")
    if (
        not isinstance(analysis_mapping, Mapping)
        or not analysis_mapping
        or not isinstance(generation_mapping, Mapping)
        or not generation_mapping
    ):
        raise CrossMatrixCandidateAssetError(
            "v2 parent must freeze manual_breakpoints in analysis and generation"
        )
    if (
        set(analysis_mapping) != set(manual_rows)
        or set(generation_mapping) != set(manual_rows)
    ):
        raise CrossMatrixCandidateAssetError(
            "v2 parent manual_breakpoints must exactly match manual method features"
        )
    for feature, (feature_row, method_row) in manual_rows.items():
        breakpoints = _manual_axis_breakpoints(
            evidence,
            feature_row=feature_row,
            method_row=method_row,
            feature=feature,
        )
        _require_manual_bins_match_breakpoints(
            feature=feature,
            feature_row=feature_row,
            method_row=method_row,
            breakpoints=breakpoints,
        )


def _parent_reference(
    value: object,
    *,
    schema_version: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("parent must be an object")
    fields = (
        _PARENT_V2_FIELDS
        if schema_version == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION
        else _PARENT_V1_FIELDS
    )
    _exact(value, fields, "parent")
    if value["target_definition"] != {"good": 0, "bad": 1}:
        raise CrossMatrixCandidateAssetError("parent target_definition changed")
    result = {
        "candidate_id": _identifier(value["candidate_id"], "parent.candidate_id", prefix="candidate"),
        "evidence_hash": _hash(value["evidence_hash"], "parent.evidence_hash"),
        "identity": _parent_identity(value["identity"]),
        "target_col": _text(value["target_col"], "parent.target_col"),
        "row_count": _positive_int(value["row_count"], "parent.row_count"),
        "target_definition": {"good": 0, "bad": 1},
        "smoothing": _positive_number(value["smoothing"], "parent.smoothing"),
    }
    if schema_version == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION:
        if value["analysis_schema_version"] != _UNIVARIATE_V2_SCHEMA_VERSION:
            raise CrossMatrixCandidateAssetError(
                "v2 parent analysis_schema_version is invalid"
            )
        result["analysis_schema_version"] = _UNIVARIATE_V2_SCHEMA_VERSION
    return result


def _parent_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("parent identity must be an object")
    _exact(value, _PARENT_IDENTITY_FIELDS, "parent identity")
    return {
        "task_id": _text(value["task_id"], "identity.task_id"),
        "dataset_id": _text(value["dataset_id"], "identity.dataset_id"),
        "dataset_content_hash": _hash(value["dataset_content_hash"], "identity.dataset_content_hash"),
        "workspace_revision": _non_negative_int(value["workspace_revision"], "identity.workspace_revision"),
        "workspace_generation": _non_negative_int(value["workspace_generation"], "identity.workspace_generation"),
        "semantic_mapping_hash": _hash(value["semantic_mapping_hash"], "identity.semantic_mapping_hash"),
    }


def _sample_identity(
    value: object,
    *,
    parent: Mapping[str, Any],
    expected_sample_hash: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("sample_identity must be an object")
    _exact(value, _SAMPLE_FIELDS, "sample_identity")
    identity = _parent_identity({key: value[key] for key in _PARENT_IDENTITY_FIELDS})
    if identity != parent["identity"]:
        raise CrossMatrixCandidateAssetError("sample_identity must exactly match parent identity")
    result = {
        **identity,
        "sample_context_hash": _hash(value["sample_context_hash"], "sample_identity.sample_context_hash"),
        "target_col": _text(value["target_col"], "sample_identity.target_col"),
        "row_count": _positive_int(value["row_count"], "sample_identity.row_count"),
    }
    if result["target_col"] != parent["target_col"] or result["row_count"] != parent["row_count"]:
        raise CrossMatrixCandidateAssetError(
            "sample_identity target_col/row_count must exactly match parent analysis"
        )
    if expected_sample_hash is not None and not hmac.compare_digest(
        result["sample_context_hash"], expected_sample_hash
    ):
        raise CrossMatrixCandidateAssetError(
            "sample_context_hash does not match exact parent CandidateEvidence"
        )
    return result


def _axis_from_evidence(
    parent_evidence: Mapping[str, Any],
    spec: object,
    *,
    position: str,
    schema_version: str,
) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise CrossMatrixCandidateAssetError(f"{position}_axis must be an object")
    _exact(spec, _AXIS_SPEC_FIELDS, f"{position}_axis")
    feature = _text(spec["feature"], f"{position}_axis.feature")
    method = _text(spec["method"], f"{position}_axis.method")
    evidence = validate_candidate_evidence(parent_evidence)
    analysis = evidence["analysis"]
    features = analysis.get("features")
    if not _sequence(features):
        raise CrossMatrixCandidateAssetError("parent analysis.features must be a non-empty array")
    matched_features = [item for item in features if isinstance(item, Mapping) and item.get("feature") == feature]
    if len(matched_features) != 1:
        raise CrossMatrixCandidateAssetError(f"axis feature {feature!r} must match exactly once")
    methods = matched_features[0].get("methods")
    if not _sequence(methods):
        raise CrossMatrixCandidateAssetError(f"axis feature {feature!r} has no methods")
    matched_methods = [item for item in methods if isinstance(item, Mapping) and item.get("method") == method]
    if len(matched_methods) != 1 or matched_methods[0].get("status") != "available":
        raise CrossMatrixCandidateAssetError(f"axis method {feature}/{method} must be uniquely available")
    matched_method = matched_methods[0]
    raw_bins = matched_method.get("bins")
    if not _sequence(raw_bins):
        raise CrossMatrixCandidateAssetError(f"axis method {feature}/{method} must have bins")
    manual_breakpoints = None
    if method == "manual":
        manual_breakpoints = _manual_axis_breakpoints(
            evidence,
            feature_row=matched_features[0],
            method_row=matched_method,
            feature=feature,
        )
        _require_manual_bins_match_breakpoints(
            feature=feature,
            feature_row=matched_features[0],
            method_row=matched_method,
            breakpoints=manual_breakpoints,
        )
    elif "manual_breakpoints" in matched_method:
        raise CrossMatrixCandidateAssetError(
            f"non-manual axis method {feature}/{method} cannot carry manual_breakpoints"
        )
    bins = [_bin_from_parent(item, feature=feature, method=method, axis_position=position, position=index, parent_hash=evidence["evidence_hash"]) for index, item in enumerate(raw_bins)]
    source_indexes = [item["source_index"] for item in bins]
    if source_indexes != list(range(len(bins))):
        raise CrossMatrixCandidateAssetError(f"axis {position} source bin indexes must be unique and contiguous")
    if len({item["source_bin_id"] for item in bins}) != len(bins):
        raise CrossMatrixCandidateAssetError(f"axis {position} source bin ids must be unique")
    if len({item["semantic_key"] for item in bins}) != len(bins):
        raise CrossMatrixCandidateAssetError(f"axis {position} source bin conditions must be semantically unique")
    row_count = _positive_int(analysis.get("row_count"), "parent analysis.row_count")
    _conserve_bins(bins, row_count=row_count, name=f"axis {position}")
    return _derive_axis(
        position=position,
        feature=feature,
        method=method,
        row_count=row_count,
        bins=bins,
        schema_version=schema_version,
        parent_evidence_hash=evidence["evidence_hash"],
        manual_breakpoints=manual_breakpoints,
    )


def _manual_axis_breakpoints(
    evidence: Mapping[str, Any],
    *,
    feature_row: Mapping[str, Any],
    method_row: Mapping[str, Any],
    feature: str,
) -> list[float]:
    if feature_row.get("feature_type") != "numeric":
        raise CrossMatrixCandidateAssetError(
            f"manual axis feature {feature!r} must be numeric"
        )
    if "manual_breakpoints" not in method_row:
        raise CrossMatrixCandidateAssetError(
            f"manual axis {feature!r} must contain manual_breakpoints"
        )
    breakpoints = _strict_manual_breakpoints(
        method_row["manual_breakpoints"],
        name=f"manual axis {feature!r} manual_breakpoints",
    )
    analysis_parameters = evidence["analysis"].get("parameters")
    generation_parameters = evidence["generation"].get("parameters")
    for name, parameters in (
        ("analysis.parameters", analysis_parameters),
        ("generation.parameters", generation_parameters),
    ):
        if not isinstance(parameters, Mapping):
            raise CrossMatrixCandidateAssetError(f"{name} must be an object")
        mapping = parameters.get("manual_breakpoints")
        if not isinstance(mapping, Mapping) or feature not in mapping:
            raise CrossMatrixCandidateAssetError(
                f"{name}.manual_breakpoints must bind manual axis {feature!r}"
            )
        bound = _strict_manual_breakpoints(
            mapping[feature],
            name=f"{name}.manual_breakpoints[{feature!r}]",
        )
        if bound != breakpoints:
            raise CrossMatrixCandidateAssetError(
                f"{name}.manual_breakpoints changed from the manual method evidence"
            )
    return breakpoints


def _strict_manual_breakpoints(value: object, *, name: str) -> list[float]:
    if not _sequence(value):
        raise CrossMatrixCandidateAssetError(f"{name} must be a non-empty array")
    if len(value) + 1 > _MAX_AXIS_BINS:
        raise CrossMatrixCandidateAssetError(
            f"{name} exceeds the {_MAX_AXIS_BINS}-bin axis budget"
        )
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise CrossMatrixCandidateAssetError(
                f"{name} must contain finite numbers"
            )
        number = float(item)
        if not math.isfinite(number):
            raise CrossMatrixCandidateAssetError(
                f"{name} must contain finite numbers"
            )
        normalized.append(number)
    if any(
        left >= right
        for left, right in zip(normalized, normalized[1:], strict=False)
    ):
        raise CrossMatrixCandidateAssetError(
            f"{name} must be strictly increasing and unique"
        )
    if _canonical_json(value) != _canonical_json(normalized):
        raise CrossMatrixCandidateAssetError(
            f"{name} must already be canonical finite floats"
        )
    return normalized


def _require_manual_bins_match_breakpoints(
    *,
    feature: str,
    feature_row: Mapping[str, Any],
    method_row: Mapping[str, Any],
    breakpoints: Sequence[float],
) -> None:
    raw_bins = method_row.get("bins")
    if not _sequence(raw_bins):
        raise CrossMatrixCandidateAssetError(
            f"manual axis {feature!r} must contain bins"
        )
    regular = [
        row
        for row in raw_bins
        if isinstance(row, Mapping) and row.get("kind") == "numeric_interval"
    ]
    if len(regular) != len(breakpoints) + 1:
        raise CrossMatrixCandidateAssetError(
            f"manual axis {feature!r} manual_breakpoints do not match its bins"
        )
    raw_sentinels = feature_row.get("sentinel_values")
    if not isinstance(raw_sentinels, Sequence) or isinstance(
        raw_sentinels, str | bytes | bytearray
    ):
        raise CrossMatrixCandidateAssetError(
            f"manual axis {feature!r} sentinel_values must be an array"
        )
    sentinels: tuple[float, ...] = tuple(
        _manual_finite_number(
            item,
            name=f"manual axis {feature!r} sentinel_values",
        )
        for item in raw_sentinels
    )
    edges = [-math.inf, *breakpoints, math.inf]
    for index, row in enumerate(regular):
        expected = {
            "index": index,
            "id": f"regular:{index}",
            "kind": "numeric_interval",
            "condition": _manual_numeric_condition(
                feature,
                edges,
                index,
                sentinels,
            ),
            "lower": (
                None if not math.isfinite(edges[index]) else float(edges[index])
            ),
            "upper": (
                None
                if not math.isfinite(edges[index + 1])
                else float(edges[index + 1])
            ),
            "include_lower": True,
            "include_upper": False,
        }
        for field, expected_value in expected.items():
            if _canonical_json(row.get(field)) != _canonical_json(expected_value):
                raise CrossMatrixCandidateAssetError(
                    f"manual axis {feature!r} manual_breakpoints do not match "
                    f"bin {index} {field}"
                )


def _manual_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CrossMatrixCandidateAssetError(f"{name} must contain finite numbers")
    number = float(value)
    if not math.isfinite(number):
        raise CrossMatrixCandidateAssetError(f"{name} must contain finite numbers")
    return number


def _manual_numeric_condition(
    feature: str,
    edges: Sequence[float],
    index: int,
    sentinels: tuple[float, ...],
) -> dict[str, Any]:
    args: list[dict[str, Any]] = []
    lower = float(edges[index])
    upper = float(edges[index + 1])
    if math.isfinite(lower):
        args.append(_manual_compare(feature, ">=", lower))
    if math.isfinite(upper):
        args.append(_manual_compare(feature, "<", upper))
    if sentinels:
        args.append(
            {
                "op": "compare",
                "field": feature,
                "operator": "not_in",
                "value": list(sentinels),
                "missing": "no_match",
            }
        )
    if len(args) == 1:
        return args[0]
    if not args:
        return {"op": "is_not_null", "field": feature}
    return {"op": "and", "args": args}


def _manual_compare(feature: str, operator: str, value: float) -> dict[str, Any]:
    return {
        "op": "compare",
        "field": feature,
        "operator": operator,
        "value": value,
        "missing": "no_match",
    }


def _bin_from_parent(value: object, *, feature: str, method: str, axis_position: str, position: int, parent_hash: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("parent bin must be an object")
    source_index = _non_negative_int(value.get("index"), "parent bin.index")
    source_id = _text(value.get("id"), "parent bin.id")
    kind = _text(value.get("kind"), "parent bin.kind")
    condition = _condition(value.get("condition"), f"parent bin {source_id}.condition")
    if value.get("condition") != condition:
        raise CrossMatrixCandidateAssetError(
            f"parent bin {source_id} condition must already be canonical"
        )
    fields = _expression_fields(condition)
    if fields != {feature}:
        raise CrossMatrixCandidateAssetError(f"parent bin {source_id} condition must reference only {feature!r}")
    count = _non_negative_int(value.get("count"), "parent bin.count")
    good = _bounded_count(value.get("good"), "parent bin.good", count)
    bad = _bounded_count(value.get("bad"), "parent bin.bad", count)
    if good + bad != count:
        raise CrossMatrixCandidateAssetError("parent bin good + bad must equal count")
    amount = _amount_evidence(value.get("amount_metrics"), count=count)
    body = {
        "position": position, "source_index": source_index, "source_bin_id": source_id,
        "kind": kind, "condition": condition, "semantic_key": semantic_expression_key(condition),
        "count": count, "good": good, "bad": bad, "amount_evidence": amount,
    }
    identity = {"parent_evidence_hash": parent_hash, "axis_position": axis_position, "feature": feature, "method": method, **body}
    bin_id = _stable_id("cross-bin", identity)
    return {**body, "bin_id": bin_id, "bin_hash": _sha256(_canonical_json({**identity, "bin_id": bin_id}))}


def _derive_axis(
    *,
    position: str,
    feature: str,
    method: str,
    row_count: int,
    bins: list[dict[str, Any]],
    schema_version: str,
    parent_evidence_hash: str | None = None,
    manual_breakpoints: Sequence[float] | None = None,
) -> dict[str, Any]:
    body = {"position": position, "feature": feature, "method": method, "row_count": row_count, "bins": bins}
    if schema_version == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION:
        body["parent_evidence_hash"] = _hash(
            parent_evidence_hash,
            "axis parent_evidence_hash",
        )
        body["manual_breakpoints"] = (
            None
            if manual_breakpoints is None
            else list(manual_breakpoints)
        )
    axis_id = _stable_id("cross-axis", body)
    return {**body, "axis_id": axis_id, "axis_hash": _sha256(_canonical_json({**body, "axis_id": axis_id}))}


def _axes(
    value: object,
    *,
    parent: Mapping[str, Any],
    schema_version: str,
) -> list[dict[str, Any]]:
    if not _sequence(value) or len(value) != 2:
        raise CrossMatrixCandidateAssetError("axes must contain exactly row and column")
    axes = [
        _axis(
            item,
            expected_position=position,
            parent=parent,
            schema_version=schema_version,
        )
        for item, position in zip(value, ("row", "column"), strict=True)
    ]
    if any(axis["row_count"] != parent["row_count"] for axis in axes):
        raise CrossMatrixCandidateAssetError(
            "axis row_count must exactly match parent row_count"
        )
    return axes


def _axis(
    value: object,
    *,
    expected_position: str,
    parent: Mapping[str, Any],
    schema_version: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("axis must be an object")
    fields = (
        _AXIS_V2_FIELDS
        if schema_version == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION
        else _AXIS_V1_FIELDS
    )
    _exact(value, fields, "axis")
    if value["position"] != expected_position:
        raise CrossMatrixCandidateAssetError("axes must remain row then column")
    feature = _text(value["feature"], "axis.feature")
    method = _text(value["method"], "axis.method")
    row_count = _positive_int(value["row_count"], "axis.row_count")
    bins_raw = value["bins"]
    if not _sequence(bins_raw):
        raise CrossMatrixCandidateAssetError("axis.bins must be non-empty")
    bins = [_stored_bin(item, feature=feature, method=method, axis_position=expected_position, position=index, parent_hash=parent["evidence_hash"]) for index, item in enumerate(bins_raw)]
    if [item["source_index"] for item in bins] != list(range(len(bins))):
        raise CrossMatrixCandidateAssetError("axis source bin indexes must be contiguous")
    if len({item["source_bin_id"] for item in bins}) != len(bins):
        raise CrossMatrixCandidateAssetError("axis source bin ids must be unique")
    _conserve_bins(bins, row_count=row_count, name=f"axis {expected_position}")
    manual_breakpoints = None
    parent_evidence_hash = None
    if schema_version == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION:
        parent_evidence_hash = _hash(
            value["parent_evidence_hash"],
            "axis parent evidence hash",
        )
        if not hmac.compare_digest(
            parent_evidence_hash,
            parent["evidence_hash"],
        ):
            raise CrossMatrixCandidateAssetError(
                "axis parent evidence hash does not match parent"
            )
        if method == "manual":
            manual_breakpoints = _strict_manual_breakpoints(
                value["manual_breakpoints"],
                name="axis.manual_breakpoints",
            )
            _require_stored_manual_axis_matches_breakpoints(
                feature=feature,
                bins=bins,
                breakpoints=manual_breakpoints,
            )
        elif value["manual_breakpoints"] is not None:
            raise CrossMatrixCandidateAssetError(
                "non-manual v2 axis cannot carry manual_breakpoints"
            )
    expected = _derive_axis(
        position=expected_position,
        feature=feature,
        method=method,
        row_count=row_count,
        bins=bins,
        schema_version=schema_version,
        parent_evidence_hash=parent_evidence_hash,
        manual_breakpoints=manual_breakpoints,
    )
    _same_identity_hash(value, expected, id_field="axis_id", hash_field="axis_hash", prefix="cross-axis", name="axis")
    return expected


def _require_stored_manual_axis_matches_breakpoints(
    *,
    feature: str,
    bins: Sequence[Mapping[str, Any]],
    breakpoints: Sequence[float],
) -> None:
    regular = [row for row in bins if row["kind"] == "numeric_interval"]
    if len(regular) != len(breakpoints) + 1:
        raise CrossMatrixCandidateAssetError(
            "axis.manual_breakpoints do not match stored numeric bins"
        )
    sentinels = _manual_condition_sentinels(
        regular[0]["condition"],
        feature=feature,
    )
    edges = [-math.inf, *breakpoints, math.inf]
    for index, row in enumerate(regular):
        if (
            row["position"] != index
            or row["source_index"] != index
            or row["source_bin_id"] != f"regular:{index}"
        ):
            raise CrossMatrixCandidateAssetError(
                "axis.manual_breakpoints do not match stored bin order"
            )
        expected_condition = _manual_numeric_condition(
            feature,
            edges,
            index,
            sentinels,
        )
        if _canonical_json(row["condition"]) != _canonical_json(expected_condition):
            raise CrossMatrixCandidateAssetError(
                "axis.manual_breakpoints do not match stored bin conditions"
            )


def _manual_condition_sentinels(
    condition: Mapping[str, Any],
    *,
    feature: str,
) -> tuple[float, ...]:
    nodes = (
        condition.get("args")
        if condition.get("op") == "and"
        else [condition]
    )
    if not isinstance(nodes, Sequence) or isinstance(
        nodes,
        str | bytes | bytearray,
    ):
        raise CrossMatrixCandidateAssetError(
            "manual axis condition has an invalid conjunction"
        )
    sentinel_nodes = [
        node
        for node in nodes
        if isinstance(node, Mapping)
        and node.get("op") == "compare"
        and node.get("field") == feature
        and node.get("operator") == "not_in"
    ]
    if not sentinel_nodes:
        return ()
    if len(sentinel_nodes) != 1:
        raise CrossMatrixCandidateAssetError(
            "manual axis condition has duplicate sentinel exclusions"
        )
    values = sentinel_nodes[0].get("value")
    if not isinstance(values, Sequence) or isinstance(
        values,
        str | bytes | bytearray,
    ):
        raise CrossMatrixCandidateAssetError(
            "manual axis sentinel exclusion must be an array"
        )
    normalized = tuple(
        _manual_finite_number(value, name="manual axis sentinel exclusion")
        for value in values
    )
    if len(set(normalized)) != len(normalized):
        raise CrossMatrixCandidateAssetError(
            "manual axis sentinel exclusions must be unique"
        )
    return normalized


def _stored_bin(value: object, **context: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("axis bin must be an object")
    _exact(value, _BIN_FIELDS, "axis bin")
    source_index = _non_negative_int(value["source_index"], "axis bin.source_index")
    source_id = _text(value["source_bin_id"], "axis bin.source_bin_id")
    kind = _text(value["kind"], "axis bin.kind")
    condition = _condition(value["condition"], "axis bin.condition")
    if value["condition"] != condition:
        raise CrossMatrixCandidateAssetError(
            "axis bin condition must already be canonical"
        )
    if _expression_fields(condition) != {context["feature"]}:
        raise CrossMatrixCandidateAssetError("axis bin condition references another field")
    count = _non_negative_int(value["count"], "axis bin.count")
    good = _bounded_count(value["good"], "axis bin.good", count)
    bad = _bounded_count(value["bad"], "axis bin.bad", count)
    if good + bad != count:
        raise CrossMatrixCandidateAssetError("axis bin good + bad must equal count")
    body = {
        "position": context["position"], "source_index": source_index,
        "source_bin_id": source_id, "kind": kind, "condition": condition,
        "semantic_key": semantic_expression_key(condition), "count": count,
        "good": good, "bad": bad,
        "amount_evidence": _amount_evidence_stored(value["amount_evidence"]),
    }
    identity = {
        "parent_evidence_hash": context["parent_hash"],
        "axis_position": context["axis_position"], "feature": context["feature"],
        "method": context["method"], **body,
    }
    bin_id = _stable_id("cross-bin", identity)
    expected = {
        **body, "bin_id": bin_id,
        "bin_hash": _sha256(_canonical_json({**identity, "bin_id": bin_id})),
    }
    if value["position"] != context["position"] or value["semantic_key"] != expected["semantic_key"]:
        raise CrossMatrixCandidateAssetError("axis bin position or semantic key changed")
    _same_identity_hash(value, expected, id_field="bin_id", hash_field="bin_hash", prefix="cross-bin", name="axis bin")
    return expected


def _measurement(value: object, *, axes: list[dict[str, Any]], sample: Mapping[str, Any], stored: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("measurement must be an object")
    expected_fields = _MEASUREMENT_FIELDS if stored else _MEASUREMENT_BODY_FIELDS
    _exact(value, expected_fields, "measurement")
    if value["schema_version"] != CROSS_MATRIX_MEASUREMENT_SCHEMA_VERSION:
        raise CrossMatrixCandidateAssetError("measurement schema_version is invalid")
    sample_hash = _hash(value["sample_context_hash"], "measurement.sample_context_hash")
    if not hmac.compare_digest(sample_hash, sample["sample_context_hash"]):
        raise CrossMatrixCandidateAssetError("measurement sample_context_hash does not match sample_identity")
    population = _positive_int(value["population_count"], "measurement.population_count")
    if population != sample["row_count"]:
        raise CrossMatrixCandidateAssetError("measurement population_count must equal parent sample row_count")
    good = _bounded_count(value["good"], "measurement.good", population)
    bad = _bounded_count(value["bad"], "measurement.bad", population)
    if good + bad != population:
        raise CrossMatrixCandidateAssetError("measurement good + bad must equal population_count")
    cells_raw = value["cells"]
    if not isinstance(cells_raw, Sequence) or isinstance(cells_raw, str | bytes | bytearray):
        raise CrossMatrixCandidateAssetError("measurement.cells must be an array")
    cells = [_primary_cell(item) for item in cells_raw]
    by_pair = {(item["row_source_bin_id"], item["column_source_bin_id"]): item for item in cells}
    if len(by_pair) != len(cells):
        raise CrossMatrixCandidateAssetError("measurement cell pairs must be unique")
    expected_pairs = [(row["source_bin_id"], column["source_bin_id"]) for row in axes[0]["bins"] for column in axes[1]["bins"]]
    if set(by_pair) != set(expected_pairs):
        raise CrossMatrixCandidateAssetError("measurement must contain the complete Cartesian cell set")
    cells = [by_pair[pair] for pair in expected_pairs]
    if sum(item["count"] for item in cells) != population or sum(item["good"] for item in cells) != good or sum(item["bad"] for item in cells) != bad:
        raise CrossMatrixCandidateAssetError("measurement cell totals do not conserve population/good/bad")
    _verify_marginals(cells, axes=axes)
    body = {
        "schema_version": CROSS_MATRIX_MEASUREMENT_SCHEMA_VERSION,
        "sample_context_hash": sample_hash, "population_count": population,
        "good": good, "bad": bad, "cells": cells,
    }
    expected_hash = _sha256(_canonical_json(body))
    if stored and not hmac.compare_digest(_hash(value["measurement_hash"], "measurement.measurement_hash"), expected_hash):
        raise CrossMatrixCandidateAssetError("measurement_hash does not match primary measurement")
    return {**body, "measurement_hash": expected_hash}


def _primary_cell(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("measurement cell must be an object")
    _exact(value, _PRIMARY_CELL_FIELDS, "measurement cell")
    count = _non_negative_int(value["count"], "measurement cell.count")
    good = _bounded_count(value["good"], "measurement cell.good", count)
    bad = _bounded_count(value["bad"], "measurement cell.bad", count)
    if good + bad != count:
        raise CrossMatrixCandidateAssetError("measurement cell good + bad must equal count")
    return {
        "row_source_bin_id": _text(value["row_source_bin_id"], "measurement cell.row_source_bin_id"),
        "column_source_bin_id": _text(value["column_source_bin_id"], "measurement cell.column_source_bin_id"),
        "count": count, "good": good, "bad": bad,
        "amounts": _amounts(value["amounts"], count=count),
    }


def _amounts(value: object, *, count: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("measurement cell.amounts must be an object")
    _exact(value, _AMOUNT_FIELDS, "measurement cell.amounts")
    return {
        "loan_amount": _amount_observation(value["loan_amount"], count=count, name="loan_amount"),
        "overdue_amount": _amount_observation(value["overdue_amount"], count=count, name="overdue_amount"),
        "paired": _paired_observation(value["paired"], count=count),
    }


def _amount_observation(value: object, *, count: int, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError(f"amount {name} must be an object")
    _exact(value, _AMOUNT_OBS_FIELDS, f"amount {name}")
    status = value["status"]
    if status == "unavailable":
        if value["covered_count"] is not None or value["value"] is not None:
            raise CrossMatrixCandidateAssetError(f"unavailable {name} must have null coverage/value")
        return {"status": "unavailable", "covered_count": None, "value": None}
    if status != "available":
        raise CrossMatrixCandidateAssetError(f"amount {name} status must be available or unavailable")
    covered = _bounded_count(value["covered_count"], f"amount {name}.covered_count", count)
    amount = _non_negative_number(value["value"], f"amount {name}.value")
    if covered == 0 and amount != 0:
        raise CrossMatrixCandidateAssetError(
            f"amount {name}.value must be zero when covered_count is zero"
        )
    return {"status": "available", "covered_count": covered, "value": amount}


def _paired_observation(value: object, *, count: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("amount paired must be an object")
    _exact(value, _PAIRED_OBS_FIELDS, "amount paired")
    if value["status"] == "unavailable":
        if any(value[key] is not None for key in _PAIRED_OBS_FIELDS - {"status"}):
            raise CrossMatrixCandidateAssetError("unavailable paired amount must have null facts")
        return {"status": "unavailable", "covered_count": None, "loan_value": None, "overdue_value": None}
    if value["status"] != "available":
        raise CrossMatrixCandidateAssetError("paired status must be available or unavailable")
    covered = _bounded_count(value["covered_count"], "paired.covered_count", count)
    loan_value = _non_negative_number(value["loan_value"], "paired.loan_value")
    overdue_value = _non_negative_number(
        value["overdue_value"], "paired.overdue_value"
    )
    if covered == 0 and (loan_value != 0 or overdue_value != 0):
        raise CrossMatrixCandidateAssetError(
            "paired values must be zero when covered_count is zero"
        )
    return {
        "status": "available", "covered_count": covered,
        "loan_value": loan_value,
        "overdue_value": overdue_value,
    }


def _verify_marginals(cells: list[dict[str, Any]], *, axes: list[dict[str, Any]]) -> None:
    for axis_index, (axis, pair_key) in enumerate(zip(axes, ("row_source_bin_id", "column_source_bin_id"), strict=True)):
        for source_bin in axis["bins"]:
            selected = [cell for cell in cells if cell[pair_key] == source_bin["source_bin_id"]]
            for field in ("count", "good", "bad"):
                if sum(cell[field] for cell in selected) != source_bin[field]:
                    raise CrossMatrixCandidateAssetError(f"axis {axis_index} {source_bin['source_bin_id']} {field} marginal changed")
            _verify_amount_marginal(selected, source_bin["amount_evidence"])


def _verify_amount_marginal(cells: list[dict[str, Any]], evidence: Mapping[str, Any]) -> None:
    for dimension in ("loan_amount", "overdue_amount"):
        observations = [cell["amounts"][dimension] for cell in cells]
        expected = evidence[dimension]
        if expected["status"] == "unavailable" and expected["reason"].endswith(
            "_not_configured"
        ):
            if any(item["status"] != "unavailable" for item in observations):
                raise CrossMatrixCandidateAssetError(f"{dimension} availability does not match parent marginal")
        else:
            if any(item["status"] != "available" for item in observations):
                raise CrossMatrixCandidateAssetError(f"{dimension} must be available in every Cartesian cell")
            if sum(item["covered_count"] for item in observations) != expected["covered_count"] or not _close(sum(item["value"] for item in observations), expected["value"]):
                raise CrossMatrixCandidateAssetError(f"{dimension} primary facts do not reconcile with parent marginal")
    observations = [cell["amounts"]["paired"] for cell in cells]
    paired = evidence["paired"]
    if paired["status"] == "unavailable" and paired["reason"] == "amount_column_not_configured":
        if any(item["status"] != "unavailable" for item in observations):
            raise CrossMatrixCandidateAssetError("paired availability does not match parent marginal")
        return
    if any(item["status"] != "available" for item in observations):
        raise CrossMatrixCandidateAssetError("paired amounts must be available in every Cartesian cell")
    covered = sum(item["covered_count"] for item in observations)
    loan = sum(item["loan_value"] for item in observations)
    overdue = sum(item["overdue_value"] for item in observations)
    if paired["covered_count"] is not None and covered != paired["covered_count"]:
        raise CrossMatrixCandidateAssetError("paired covered_count does not reconcile with parent marginal")
    if paired["status"] == "available" and (loan <= 0 or not _close(overdue / loan, paired["value"])):
        raise CrossMatrixCandidateAssetError("paired rate does not reconcile with parent marginal")
    if paired["status"] == "not_applicable" and paired["reason"] == "zero_loan_amount" and not _close(loan, 0.0):
        raise CrossMatrixCandidateAssetError("zero-loan parent marginal must have zero paired loan value")
    if paired["reason"] in {"no_paired_amounts", "empty_bin"} and (
        covered != 0 or not _close(loan, 0.0) or not _close(overdue, 0.0)
    ):
        raise CrossMatrixCandidateAssetError(
            "no-paired/empty parent marginal must have zero paired primary facts"
        )


def _amount_evidence(value: object, *, count: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("parent bin amount_metrics must be an object")
    required = {"loan_amount", "overdue_amount", "overdue_rate"}
    if set(value) != required:
        raise CrossMatrixCandidateAssetError("parent bin amount_metrics has non-exact fields")
    result: dict[str, Any] = {}
    for dimension in ("loan_amount", "overdue_amount"):
        item = value[dimension]
        if not isinstance(item, Mapping):
            raise CrossMatrixCandidateAssetError(f"parent {dimension} must be an object")
        status = item.get("status")
        if status == "unavailable":
            reason = _text(item.get("reason"), f"parent {dimension}.reason")
            if reason == "no_covered_rows":
                result[dimension] = {
                    "status": "unavailable", "covered_count": 0,
                    "value": 0.0, "reason": reason,
                }
            elif reason == f"{dimension}_not_configured":
                result[dimension] = {
                    "status": "unavailable", "covered_count": None,
                    "value": None, "reason": reason,
                }
            else:
                raise CrossMatrixCandidateAssetError(
                    f"parent {dimension} unavailable reason is unsupported"
                )
        elif status == "available":
            result[dimension] = {
                "status": "available",
                "covered_count": _bounded_count(item.get("covered_count"), f"parent {dimension}.covered_count", count),
                "value": _non_negative_number(item.get("sum"), f"parent {dimension}.sum"),
                "reason": None,
            }
        else:
            raise CrossMatrixCandidateAssetError(f"parent {dimension} status is unsupported")
    rate = value["overdue_rate"]
    if not isinstance(rate, Mapping):
        raise CrossMatrixCandidateAssetError("parent overdue_rate must be an object")
    status = rate.get("status")
    if status == "available":
        result["paired"] = {
            "status": "available", "covered_count": _bounded_count(rate.get("paired_count"), "parent paired_count", count),
            "value": _non_negative_number(rate.get("value"), "parent overdue_rate.value"), "reason": None,
        }
    elif status == "unavailable":
        reason = _text(rate.get("reason"), "parent overdue_rate.reason")
        if reason == "amount_column_not_configured":
            result["paired"] = {
                "status": "unavailable", "covered_count": None,
                "value": None, "reason": reason,
            }
        elif reason == "no_paired_amounts":
            result["paired"] = {
                "status": "unavailable", "covered_count": 0,
                "value": 0.0, "reason": reason,
            }
        else:
            raise CrossMatrixCandidateAssetError(
                "parent overdue_rate unavailable reason is unsupported"
            )
    elif status == "not_applicable":
        reason = _text(rate.get("reason"), "parent overdue_rate.reason")
        if reason == "empty_bin":
            result["paired"] = {
                "status": "not_applicable", "covered_count": 0,
                "value": 0.0, "reason": reason,
            }
        elif reason == "zero_loan_amount":
            result["paired"] = {
                "status": "not_applicable", "covered_count": None,
                "value": None, "reason": reason,
            }
        else:
            raise CrossMatrixCandidateAssetError(
                "parent overdue_rate not_applicable reason is unsupported"
            )
    else:
        raise CrossMatrixCandidateAssetError("parent overdue_rate status is unsupported")
    return result


def _budget(limit: object, *, required: int) -> dict[str, Any]:
    value = _positive_int(limit, "budget")
    if value < required:
        raise CrossMatrixCandidateAssetError(f"cross-matrix budget {value} is insufficient for {required} complete cells")
    return {"unit": "matrix_cells", "limit": value, "required": required, "truncated": False}


def _budget_object(value: object, *, required: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("budget must be an object")
    _exact(value, _BUDGET_FIELDS, "budget")
    expected = _budget(value["limit"], required=required)
    if value != expected:
        raise CrossMatrixCandidateAssetError("budget must be exact, sufficient, and truncated=false")
    return expected


def _derive_matrix(axes: list[dict[str, Any]], measurement: Mapping[str, Any], *, parent: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    row_by_source = {item["source_bin_id"]: item for item in axes[0]["bins"]}
    column_by_source = {item["source_bin_id"]: item for item in axes[1]["bins"]}
    cells = []
    for primary in measurement["cells"]:
        row = row_by_source[primary["row_source_bin_id"]]
        column = column_by_source[primary["column_source_bin_id"]]
        cells.append(_derive_cell(primary, row=row, column=column, measurement=measurement, parent=parent, sample=sample))
    body = {"row_bin_count": len(axes[0]["bins"]), "column_bin_count": len(axes[1]["bins"]), "cell_count": len(cells), "cells": cells}
    return {**body, "matrix_hash": _sha256(_canonical_json(body))}


def _derive_cell(primary: Mapping[str, Any], *, row: Mapping[str, Any], column: Mapping[str, Any], measurement: Mapping[str, Any], parent: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    condition = _condition({"op": "and", "args": [row["condition"], column["condition"]]}, "cell rule condition")
    rule_body = {"condition": condition, "semantic_key": semantic_expression_key(condition)}
    rule_identity = {"row_bin_hash": row["bin_hash"], "column_bin_hash": column["bin_hash"], **rule_body}
    rule_id = _stable_id("candidate-rule", rule_identity)
    rule = {"rule_id": rule_id, **rule_body, "rule_hash": _sha256(_canonical_json({**rule_identity, "rule_id": rule_id}))}
    count, good, bad = primary["count"], primary["good"], primary["bad"]
    share = count / measurement["population_count"]
    bad_rate = None if count == 0 else bad / count
    overall_bad_rate = measurement["bad"] / measurement["population_count"]
    lift = None if bad_rate is None or overall_bad_rate == 0 else bad_rate / overall_bad_rate
    smoothing = parent["smoothing"]
    groups = len(measurement["cells"])
    woe, iv = _smoothed_woe_iv(
        bad,
        good,
        measurement["bad"],
        measurement["good"],
        groups,
        smoothing=float(smoothing),
    )
    effect_body = {
        "count": count, "good": good, "bad": bad, "share": share,
        "bad_rate": bad_rate, "lift": lift, "woe": woe, "iv_contribution": iv,
        "amount_metrics": _derived_amount_metrics(primary["amounts"], count=count),
    }
    effect_identity = {"rule_hash": rule["rule_hash"], "sample_context_hash": sample["sample_context_hash"], **effect_body}
    effect_id = _stable_id("candidate-effect", effect_identity)
    effect = {"effect_id": effect_id, **effect_body, "effect_hash": _sha256(_canonical_json({**effect_identity, "effect_id": effect_id}))}
    cell_identity = {"row_bin_id": row["bin_id"], "column_bin_id": column["bin_id"], "rule_hash": rule["rule_hash"], "effect_hash": effect["effect_hash"]}
    cell_id = _stable_id("cross-cell", cell_identity)
    return {"row_bin_id": row["bin_id"], "column_bin_id": column["bin_id"], "cell_id": cell_id, "cell_hash": _sha256(_canonical_json({**cell_identity, "cell_id": cell_id})), "rule": rule, "effect": effect}


def _derived_amount_metrics(amounts: Mapping[str, Any], *, count: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in ("loan_amount", "overdue_amount"):
        item = amounts[dimension]
        if item["status"] == "unavailable":
            result[dimension] = {"status": "unavailable", "covered_count": None, "coverage_rate": None, "value": None, "reason": "column_unavailable"}
        else:
            result[dimension] = {"status": "available", "covered_count": item["covered_count"], "coverage_rate": None if count == 0 else item["covered_count"] / count, "value": item["value"], "reason": None}
    paired = amounts["paired"]
    if paired["status"] == "unavailable":
        result["overdue_rate"] = {"status": "unavailable", "covered_count": None, "coverage_rate": None, "value": None, "reason": "columns_unavailable"}
    elif paired["covered_count"] == 0:
        reason = "no_observations" if count == 0 else "no_paired_observations"
        result["overdue_rate"] = {"status": "not_applicable", "covered_count": 0, "coverage_rate": None if count == 0 else 0.0, "value": None, "reason": reason}
    elif paired["loan_value"] == 0:
        result["overdue_rate"] = {"status": "not_applicable", "covered_count": paired["covered_count"], "coverage_rate": paired["covered_count"] / count, "value": None, "reason": "zero_loan_amount"}
    else:
        result["overdue_rate"] = {"status": "available", "covered_count": paired["covered_count"], "coverage_rate": paired["covered_count"] / count, "value": paired["overdue_value"] / paired["loan_value"], "reason": None}
    return result


def _normalize_matrix(value: object, *, axes: list[dict[str, Any]], measurement: Mapping[str, Any], parent: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("matrix must be an object")
    _exact(value, _MATRIX_FIELDS, "matrix")
    expected = _derive_matrix(axes, measurement, parent=parent, sample=sample)
    if value != expected:
        raise CrossMatrixCandidateAssetError("matrix cells, rules, effects, metrics, ids, or hashes are not deterministic")
    return expected


def _derive_summary(measurement: Mapping[str, Any], *, matrix: Mapping[str, Any]) -> dict[str, Any]:
    count = measurement["population_count"]
    raw_amounts = _aggregate_amounts(measurement["cells"])
    body = {
        "count": count, "good": measurement["good"], "bad": measurement["bad"],
        "bad_rate": measurement["bad"] / count,
        "total_iv": sum(cell["effect"]["iv_contribution"] for cell in matrix["cells"]),
        "amount_metrics": _derived_amount_metrics(raw_amounts, count=count),
    }
    return {**body, "summary_hash": _sha256(_canonical_json(body))}


def _aggregate_amounts(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in ("loan_amount", "overdue_amount"):
        items = [cell["amounts"][dimension] for cell in cells]
        result[dimension] = ({"status": "unavailable", "covered_count": None, "value": None} if any(item["status"] == "unavailable" for item in items) else {"status": "available", "covered_count": sum(item["covered_count"] for item in items), "value": sum(item["value"] for item in items)})
    items = [cell["amounts"]["paired"] for cell in cells]
    result["paired"] = ({"status": "unavailable", "covered_count": None, "loan_value": None, "overdue_value": None} if any(item["status"] == "unavailable" for item in items) else {"status": "available", "covered_count": sum(item["covered_count"] for item in items), "loan_value": sum(item["loan_value"] for item in items), "overdue_value": sum(item["overdue_value"] for item in items)})
    return result


def _normalize_summary(value: object, *, measurement: Mapping[str, Any], matrix: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("summary must be an object")
    _exact(value, _SUMMARY_FIELDS, "summary")
    expected = _derive_summary(measurement, matrix=matrix)
    if value != expected:
        raise CrossMatrixCandidateAssetError("summary must be deterministically derived")
    return expected


def _derive_candidate_evidence(core: Mapping[str, Any]) -> dict[str, Any]:
    evidence_schema_version = (
        "strategy.cross-matrix-candidate-evidence.v2"
        if core.get("schema_version")
        == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION
        else "strategy.cross-matrix-candidate-evidence.v1"
    )
    evidence_hash = _sha256(
        _canonical_json({"schema_version": evidence_schema_version, **core})
    )
    return {"candidate_id": "candidate-" + evidence_hash[:32], "evidence_hash": evidence_hash}


def _candidate_evidence(value: object, *, core: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("candidate_evidence must be an object")
    _exact(value, _EVIDENCE_FIELDS, "candidate_evidence")
    expected = _derive_candidate_evidence(core)
    if value != expected:
        raise CrossMatrixCandidateAssetError("candidate evidence does not authenticate canonical asset content")
    return expected


def _amount_evidence_stored(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError("amount_evidence must be an object")
    _exact(value, _AMOUNT_FIELDS, "amount_evidence")
    result = {}
    for dimension in ("loan_amount", "overdue_amount"):
        item = value[dimension]
        if not isinstance(item, Mapping):
            raise CrossMatrixCandidateAssetError("amount evidence observation must be an object")
        _exact(item, _AMOUNT_EVIDENCE_FIELDS, "amount evidence observation")
        status = item["status"]
        reason = _optional_text(item["reason"])
        if status == "available":
            normalized_covered = _non_negative_int(
                item["covered_count"], f"{dimension}.covered_count"
            )
            normalized_value = _non_negative_number(
                item["value"], f"{dimension}.value"
            )
            if normalized_covered == 0 and normalized_value != 0:
                raise CrossMatrixCandidateAssetError(
                    f"available {dimension} evidence with zero coverage must have zero value"
                )
            result[dimension] = {
                "status": status,
                "covered_count": normalized_covered,
                "value": normalized_value,
                "reason": reason,
            }
            if reason is not None:
                raise CrossMatrixCandidateAssetError(
                    f"available {dimension} evidence cannot have a reason"
                )
        elif status == "unavailable":
            normalized_reason = _text(reason, f"{dimension}.reason")
            if normalized_reason == f"{dimension}_not_configured":
                if item["covered_count"] is not None or item["value"] is not None:
                    raise CrossMatrixCandidateAssetError(
                        f"unconfigured {dimension} evidence must have null facts"
                    )
                covered_count: int | None = None
                amount_value: int | float | None = None
            elif normalized_reason == "no_covered_rows":
                if item["covered_count"] != 0 or item["value"] != 0:
                    raise CrossMatrixCandidateAssetError(
                        f"no-covered {dimension} evidence must have zero facts"
                    )
                covered_count = 0
                amount_value = 0.0
            else:
                raise CrossMatrixCandidateAssetError(
                    f"unavailable {dimension} evidence reason is unsupported"
                )
            result[dimension] = {
                "status": status,
                "covered_count": covered_count,
                "value": amount_value,
                "reason": normalized_reason,
            }
        else:
            raise CrossMatrixCandidateAssetError(
                f"{dimension} evidence status is unsupported"
            )
    paired = value["paired"]
    if not isinstance(paired, Mapping):
        raise CrossMatrixCandidateAssetError("paired evidence must be an object")
    _exact(paired, _PAIRED_EVIDENCE_FIELDS, "paired evidence")
    paired_status = paired["status"]
    paired_reason = _optional_text(paired["reason"])
    if paired_status == "available":
        covered = _positive_int(paired["covered_count"], "paired evidence.covered_count")
        paired_value = _non_negative_number(
            paired["value"], "paired evidence.value"
        )
        if paired_reason is not None:
            raise CrossMatrixCandidateAssetError(
                "available paired evidence cannot have a reason"
            )
    elif paired_status == "unavailable" and paired_reason == "amount_column_not_configured":
        if paired["covered_count"] is not None or paired["value"] is not None:
            raise CrossMatrixCandidateAssetError(
                "unconfigured paired evidence must have null facts"
            )
        covered, paired_value = None, None
    elif paired_status == "unavailable" and paired_reason == "no_paired_amounts":
        if paired["covered_count"] != 0 or paired["value"] != 0:
            raise CrossMatrixCandidateAssetError(
                "no-paired evidence must have zero facts"
            )
        covered, paired_value = 0, 0.0
    elif paired_status == "not_applicable" and paired_reason == "empty_bin":
        if paired["covered_count"] != 0 or paired["value"] != 0:
            raise CrossMatrixCandidateAssetError(
                "empty-bin paired evidence must have zero facts"
            )
        covered, paired_value = 0, 0.0
    elif paired_status == "not_applicable" and paired_reason == "zero_loan_amount":
        if paired["covered_count"] is not None or paired["value"] is not None:
            raise CrossMatrixCandidateAssetError(
                "zero-loan paired evidence must have null aggregate rate facts"
            )
        covered, paired_value = None, None
    else:
        raise CrossMatrixCandidateAssetError(
            "paired evidence status/reason combination is unreachable"
        )
    result["paired"] = {
        "status": paired_status,
        "covered_count": covered,
        "value": paired_value,
        "reason": paired_reason,
    }
    return result


def _conserve_bins(bins: Sequence[Mapping[str, Any]], *, row_count: int, name: str) -> None:
    if (
        sum(item["count"] for item in bins) != row_count
        or sum(item["good"] + item["bad"] for item in bins) != row_count
    ):
        raise CrossMatrixCandidateAssetError(
            f"{name} bins do not conserve row_count"
        )


def _condition(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCandidateAssetError(f"{name} must be an object")
    try:
        return canonicalize_expression(value)
    except StrategyError as exc:
        raise CrossMatrixCandidateAssetError(f"{name} is invalid: {exc}") from exc


def _expression_fields(expression: Mapping[str, Any]) -> set[str]:
    if "field" in expression:
        return {str(expression["field"])}
    if "args" in expression:
        return set().union(*(_expression_fields(item) for item in expression["args"]))
    if "arg" in expression:
        return _expression_fields(expression["arg"])
    return set()


def _lifecycle(value: object) -> dict[str, str]:
    if value != _LIFECYCLE:
        raise CrossMatrixCandidateAssetError("lifecycle must remain development/backtested/unvalidated")
    return dict(_LIFECYCLE)


def _producer(value: object, *, schema_version: str) -> str:
    producer = _text(value, "producer_version")
    expected = (
        CROSS_MATRIX_CANDIDATE_ASSET_V2_PRODUCER_VERSION
        if schema_version == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION
        else CROSS_MATRIX_CANDIDATE_ASSET_PRODUCER_VERSION
    )
    if producer != expected:
        raise CrossMatrixCandidateAssetError("producer_version is unsupported")
    return producer


def _same_identity_hash(actual: Mapping[str, Any], expected: Mapping[str, Any], *, id_field: str, hash_field: str, prefix: str, name: str) -> None:
    actual_id = _identifier(actual[id_field], f"{name}.{id_field}", prefix=prefix)
    actual_hash = _hash(actual[hash_field], f"{name}.{hash_field}")
    if not hmac.compare_digest(actual_id, expected[id_field]) or not hmac.compare_digest(actual_hash, expected[hash_field]):
        raise CrossMatrixCandidateAssetError(f"{name} id/hash changed")


def _exact(value: Mapping[str, Any], fields: frozenset[str] | set[str], name: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise CrossMatrixCandidateAssetError(f"{name} keys must be strings")
    missing, extra = sorted(set(fields) - set(value)), sorted(set(value) - set(fields))
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unsupported " + ", ".join(extra))
        raise CrossMatrixCandidateAssetError(f"{name} has non-exact fields: {'; '.join(detail)}")


def _identifier(value: object, name: str, *, prefix: str) -> str:
    text = _text(value, name)
    if re.fullmatch(rf"{re.escape(prefix)}-[0-9a-f]{{32}}", text) is None:
        raise CrossMatrixCandidateAssetError(f"{name} has invalid format")
    return text


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise CrossMatrixCandidateAssetError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CrossMatrixCandidateAssetError(f"{name} must be non-empty canonical text")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value, "reason")


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise CrossMatrixCandidateAssetError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(value: object, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise CrossMatrixCandidateAssetError(f"{name} must be positive")
    return result


def _bounded_count(value: object, name: str, upper: int) -> int:
    result = _non_negative_int(value, name)
    if result > upper:
        raise CrossMatrixCandidateAssetError(f"{name} exceeds count")
    return result


def _non_negative_number(value: object, name: str) -> int | float:
    result = _finite(value, name)
    if result < 0:
        raise CrossMatrixCandidateAssetError(f"{name} must be non-negative")
    return result


def _positive_number(value: object, name: str) -> int | float:
    result = _finite(value, name)
    if result <= 0:
        raise CrossMatrixCandidateAssetError(f"{name} must be positive")
    return result


def _finite(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CrossMatrixCandidateAssetError(f"{name} must be a finite number")
    number: int | float = int(value) if isinstance(value, Integral) else float(value)
    if not math.isfinite(float(number)):
        raise CrossMatrixCandidateAssetError(f"{name} must be a finite number")
    return number


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray) and bool(value)


def _close(left: int | float, right: int | float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CrossMatrixCandidateAssetError(f"value is not finite canonical JSON: {exc}") from exc


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CrossMatrixCandidateAssetError(f"cross-matrix candidate JSON has duplicate key: {key}")
        result[key] = value
    return result


__all__ = [
    "CROSS_MATRIX_CANDIDATE_ASSET_PRODUCER_VERSION",
    "CROSS_MATRIX_CANDIDATE_ASSET_SCHEMA_VERSION",
    "CROSS_MATRIX_CANDIDATE_ASSET_V2_PRODUCER_VERSION",
    "CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION",
    "CROSS_MATRIX_CANDIDATE_ASSET_TYPE",
    "CROSS_MATRIX_MEASUREMENT_SCHEMA_VERSION",
    "CrossMatrixCandidateAssetError",
    "build_cross_matrix_candidate_asset",
    "canonical_cross_matrix_candidate_asset_json",
    "parse_cross_matrix_candidate_asset_json",
    "rebuild_cross_matrix_candidate_asset",
    "validate_cross_matrix_candidate_asset",
]
