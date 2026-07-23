"""Pointer-only selections of one or more cells from a Cross Matrix asset.

The persisted selection is an immutable audit event.  It binds an exact live
Cross Matrix TaskArtifact, the matrix asset and candidate evidence identities,
and one explicit non-empty group of cell ids.  It deliberately does not copy
cell predicates, measured effects, actions, or lifecycle claims.

Executable and evidentiary facts are rebuilt only by replaying the pointer
against the independently verified full matrix asset.  A multi-cell group is a
row-major OR of the source cell rules.  Primary counts and amount observations
are aggregated before rates and the platform WOE/IV kernel are applied.  The
WOE/IV contribution is for the coalesced matrix partition: the selected cells
form one group and each unselected cell remains a separate group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
from typing import Any, TypedDict
import unicodedata

from marvis.feature.iv import _smoothed_woe_iv
from marvis.packs.strategy.candidate_fragment import (
    CandidateFragmentError,
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.cross_matrix_candidate import (
    CROSS_MATRIX_CANDIDATE_ASSET_SCHEMA_VERSION,
    CROSS_MATRIX_CANDIDATE_ASSET_PRODUCER_VERSION,
    CROSS_MATRIX_CANDIDATE_ASSET_TYPE,
    CROSS_MATRIX_CANDIDATE_ASSET_V2_PRODUCER_VERSION,
    CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION,
    CrossMatrixCandidateAssetError,
    canonical_cross_matrix_candidate_asset_json,
    validate_cross_matrix_candidate_asset,
)
from marvis.packs.strategy.dsl import canonicalize_expression, semantic_expression_key
from marvis.packs.strategy.errors import StrategyError


CROSS_MATRIX_CELL_SELECTION_SCHEMA_VERSION = "strategy.cross-matrix-cell-selection.v1"
CROSS_MATRIX_CELL_SELECTION_PRODUCER_VERSION = "strategy.cross-matrix-cell-selection/1"

CROSS_MATRIX_SOURCE_ARTIFACT_KIND = "strategy_cross_matrix_candidate_json"
CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION = (
    "strategy.cross-matrix-candidate-artifact.v1"
)
CROSS_MATRIX_SOURCE_ARTIFACT_V2_SCHEMA_VERSION = (
    "strategy.cross-matrix-candidate-artifact.v2"
)
CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL = "strategy.build_cross_matrix_candidate"

CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND = "strategy_cross_matrix_cell_selection_json"
CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION = (
    "strategy.cross-matrix-cell-selection-artifact.v1"
)
CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL = (
    "strategy.materialize_cross_matrix_cell_selection"
)
MAX_SELECTION_REASON_LENGTH = 500

# Short aliases keep Pool dispatch code readable.
CELL_SELECTION_ARTIFACT_KIND = CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND
CELL_SELECTION_ARTIFACT_SCHEMA_VERSION = (
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION
)
CELL_SELECTION_ORIGIN_TOOL = CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_CELL_ID_RE = re.compile(r"^cross-cell-[0-9a-f]{32}$")
_GROUP_ID_RE = re.compile(r"^cross-matrix-cell-group-[0-9a-f]{32}$")
_SELECTION_ID_RE = re.compile(r"^cross-matrix-cell-selection-[0-9a-f]{32}$")

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "source_artifact",
        "source_asset",
        "source_candidate",
        "group_id",
        "cell_ids",
        "selection_reason",
        "producer_version",
        "selection_id",
        "selection_hash",
    }
)
_BODY_FIELDS = _TOP_LEVEL_FIELDS - {"selection_id", "selection_hash"}
_SOURCE_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "task_id",
        "kind",
        "artifact_schema_version",
        "content_hash",
        "origin_tool",
        "path",
        "provenance_hash",
    }
)
_SOURCE_BINDING_FIELDS = _SOURCE_ARTIFACT_FIELDS | {
    "provenance",
    "canonical_bytes",
}
_SOURCE_ASSET_FIELDS = frozenset(
    {"schema_version", "asset_id", "asset_hash", "asset_type"}
)
_SOURCE_CANDIDATE_FIELDS = frozenset(
    {"candidate_id", "evidence_hash", "evidence_identity"}
)
_EVIDENCE_IDENTITY_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
    }
)
_SOURCE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "asset_schema_version",
        "asset_type",
        "asset_id",
        "asset_hash",
        "parent_candidate_id",
        "parent_evidence_hash",
        "candidate_id",
        "evidence_hash",
        "source_artifact_id",
        "source_artifact_content_hash",
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
        "target_col",
        "labeled_row_count",
        "row_axis",
        "column_axis",
        "cell_count",
        "candidate_stage",
        "observation_stage",
        "validation_status",
        "budget",
        "truncated",
    }
)
_AXIS_PROVENANCE_V1_FIELDS = frozenset({"feature", "method"})
_AXIS_PROVENANCE_V2_FIELDS = _AXIS_PROVENANCE_V1_FIELDS | frozenset(
    {"bin_count", "manual_breakpoints", "parent_evidence_hash"}
)
_SOURCE_VERSION_CONTRACTS = {
    CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION: (
        CROSS_MATRIX_CANDIDATE_ASSET_SCHEMA_VERSION,
        CROSS_MATRIX_CANDIDATE_ASSET_PRODUCER_VERSION,
    ),
    CROSS_MATRIX_SOURCE_ARTIFACT_V2_SCHEMA_VERSION: (
        CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION,
        CROSS_MATRIX_CANDIDATE_ASSET_V2_PRODUCER_VERSION,
    ),
}
_SOURCE_ASSET_TO_ARTIFACT_SCHEMA = {
    asset_schema: artifact_schema
    for artifact_schema, (asset_schema, _producer) in _SOURCE_VERSION_CONTRACTS.items()
}
_SELECTION_ARTIFACT_BINDING_FIELDS = frozenset(
    {
        "artifact_id",
        "task_id",
        "kind",
        "content_hash",
        "origin_tool",
        "artifact_schema_version",
        "producer_version",
        "selection_id",
        "selection_hash",
        "group_id",
        "source_artifact_id",
        "source_artifact_kind",
        "source_artifact_schema_version",
        "source_artifact_content_hash",
        "source_artifact_origin_tool",
        "source_artifact_path",
        "source_artifact_provenance_hash",
        "source_asset_schema_version",
        "source_asset_id",
        "source_asset_hash",
        "source_asset_type",
        "source_candidate_id",
        "source_evidence_hash",
        "source_evidence_identity",
        "cell_ids",
    }
)


class CrossMatrixCellSelectionError(StrategyError):
    """A Cross Matrix cell selection or replay failed closed."""


class IndependentlyVerifiedCrossMatrixArtifactBinding(TypedDict):
    """Caller-verified live source artifact facts required by the pure seam."""

    artifact_id: str
    task_id: str
    kind: str
    artifact_schema_version: str
    content_hash: str
    origin_tool: str
    path: str
    provenance_hash: str
    provenance: dict[str, Any]
    canonical_bytes: bytes


def build_cross_matrix_cell_selection(
    full_matrix_asset: Mapping[str, Any],
    *,
    source_artifact_binding: IndependentlyVerifiedCrossMatrixArtifactBinding,
    cell_ids: Sequence[str],
    selection_reason: str | None = None,
) -> dict[str, Any]:
    """Build one canonical pointer-only Cross Matrix cell-group selection."""

    asset = _full_matrix_asset(full_matrix_asset)
    binding = _verified_source_artifact_binding(source_artifact_binding)
    _require_source_artifact_binding_matches_asset(binding, asset=asset)
    facts = derive_cross_matrix_cell_group_facts(asset, cell_ids=cell_ids)
    reason = _canonicalize_selection_reason(selection_reason)
    body = _normalize_body(
        {
            "schema_version": CROSS_MATRIX_CELL_SELECTION_SCHEMA_VERSION,
            "source_artifact": _source_artifact_pointer(binding),
            "source_asset": _source_asset_reference_from_asset(asset),
            "source_candidate": _source_candidate_reference_from_asset(asset),
            "group_id": facts["group_id"],
            "cell_ids": facts["cell_ids"],
            "selection_reason": reason,
            "producer_version": CROSS_MATRIX_CELL_SELECTION_PRODUCER_VERSION,
        }
    )
    selection_id = _stable_id("cross-matrix-cell-selection", body)
    without_hash = {**body, "selection_id": selection_id}
    selection_hash = _sha256(_canonical_json(without_hash))
    return validate_cross_matrix_cell_selection(
        {**without_hash, "selection_hash": selection_hash}
    )


def validate_cross_matrix_cell_selection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact self-authenticating pointer-only selection."""

    if not isinstance(payload, Mapping):
        raise CrossMatrixCellSelectionError("cell selection must be an object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "cell selection")
    selection_id = _identifier(
        payload["selection_id"],
        "selection_id",
        pattern=_SELECTION_ID_RE,
    )
    selection_hash = _hash(payload["selection_hash"], "selection_hash")
    body = _normalize_body(
        {
            key: payload[key]
            for key in payload
            if key not in {"selection_id", "selection_hash"}
        }
    )
    expected_id = _stable_id("cross-matrix-cell-selection", body)
    if not hmac.compare_digest(selection_id, expected_id):
        raise CrossMatrixCellSelectionError(
            "selection_id does not match canonical cell selection"
        )
    without_hash = {**body, "selection_id": selection_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(selection_hash, expected_hash):
        raise CrossMatrixCellSelectionError(
            "selection_hash does not match canonical cell selection"
        )
    return {**without_hash, "selection_hash": selection_hash}


def canonical_cross_matrix_cell_selection_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole canonical JSON representation of a valid selection."""

    return _canonical_json(validate_cross_matrix_cell_selection(payload))


def cross_matrix_cell_selection_content_hash(
    payload: Mapping[str, Any],
) -> str:
    """Hash the canonical persisted bytes of one valid selection."""

    return _sha256(canonical_cross_matrix_cell_selection_json(payload))


def validate_cross_matrix_source_artifact_binding(
    full_matrix_asset: Mapping[str, Any],
    binding_payload: IndependentlyVerifiedCrossMatrixArtifactBinding,
) -> dict[str, Any]:
    """Strictly cross-check one caller-verified source snapshot and asset."""

    asset = _full_matrix_asset(full_matrix_asset)
    binding = _verified_source_artifact_binding(binding_payload)
    _require_source_artifact_binding_matches_asset(binding, asset=asset)
    return binding


def derive_cross_matrix_cell_group_facts(
    full_matrix_asset: Mapping[str, Any],
    *,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Derive reason-independent group rule, effect, and fragment facts.

    Input order is never authoritative.  Cell ids are duplicate-checked,
    then emitted in the source matrix's row-major order.  Primary observations
    come from ``measurement.cells``; metrics are never summed from stored cell
    effects.
    """

    asset = _full_matrix_asset(full_matrix_asset)
    normalized_ids = _requested_cell_ids(cell_ids)
    matrix_cells = asset["matrix"]["cells"]
    matrix_by_id = {
        cell["cell_id"]: (index, cell) for index, cell in enumerate(matrix_cells)
    }
    unknown = [cell_id for cell_id in normalized_ids if cell_id not in matrix_by_id]
    if unknown:
        raise CrossMatrixCellSelectionError(
            "cell_ids contain unknown Cross Matrix cells: " + ", ".join(unknown)
        )
    selected_set = set(normalized_ids)
    ordered_ids = [
        cell["cell_id"] for cell in matrix_cells if cell["cell_id"] in selected_set
    ]
    if len(ordered_ids) != len(normalized_ids):
        raise CrossMatrixCellSelectionError("cell_ids did not replay exactly once")

    selected_indexes = [matrix_by_id[cell_id][0] for cell_id in ordered_ids]
    selected_cells = [matrix_cells[index] for index in selected_indexes]
    primary_cells = [asset["measurement"]["cells"][index] for index in selected_indexes]
    count = sum(
        _non_negative_int(cell["count"], "cell count") for cell in primary_cells
    )
    good = sum(_non_negative_int(cell["good"], "cell good") for cell in primary_cells)
    bad = sum(_non_negative_int(cell["bad"], "cell bad") for cell in primary_cells)
    if good + bad != count:
        raise CrossMatrixCellSelectionError(
            "selected cell primary good/bad facts do not conserve count"
        )
    if count == 0:
        raise CrossMatrixCellSelectionError(
            "selected Cross Matrix cell group must have positive total count"
        )

    condition = _canonical_condition(
        {
            "op": "or",
            "args": [cell["rule"]["condition"] for cell in selected_cells],
        },
        "selected cell group condition",
    )
    source_asset = _source_asset_reference_from_asset(asset)
    source_candidate = _source_candidate_reference_from_asset(asset)
    group_identity = {
        "schema_version": "strategy.cross-matrix-cell-group.v1",
        "source_asset": source_asset,
        "source_candidate": source_candidate,
        "cell_ids": ordered_ids,
    }
    group_id = _stable_id("cross-matrix-cell-group", group_identity)

    rule_body = {
        "condition": condition,
        "semantic_key": semantic_expression_key(condition),
    }
    rule_identity = {
        "schema_version": "strategy.cross-matrix-cell-group-rule.v1",
        "group_id": group_id,
        "source_asset_hash": asset["asset_hash"],
        "source_evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "cell_ids": ordered_ids,
        **rule_body,
    }
    rule_id = _stable_id("candidate-rule", rule_identity)
    rule_without_hash = {"rule_id": rule_id, **rule_body}
    rule_hash = _sha256(_canonical_json({**rule_identity, "rule_id": rule_id}))
    rule = {**rule_without_hash, "rule_hash": rule_hash}

    population = _positive_int(
        asset["measurement"]["population_count"],
        "matrix population_count",
    )
    total_good = _non_negative_int(asset["measurement"]["good"], "matrix good")
    total_bad = _non_negative_int(asset["measurement"]["bad"], "matrix bad")
    if total_good + total_bad != population:
        raise CrossMatrixCellSelectionError(
            "matrix primary good/bad facts do not conserve population"
        )
    group_count = len(matrix_cells) - len(ordered_ids) + 1
    if group_count < 1:
        raise CrossMatrixCellSelectionError(
            "coalesced matrix partition must contain at least one group"
        )
    smoothing = _positive_number(asset["parent"]["smoothing"], "matrix smoothing")
    woe, iv_contribution = _smoothed_woe_iv(
        bad,
        good,
        total_bad,
        total_good,
        group_count,
        smoothing=float(smoothing),
    )
    bad_rate = bad / count
    overall_bad_rate = total_bad / population
    raw_amounts = _aggregate_amount_observations(primary_cells)
    effect_body = {
        "count": count,
        "good": good,
        "bad": bad,
        "share": count / population,
        "bad_rate": bad_rate,
        "lift": None if overall_bad_rate == 0 else bad_rate / overall_bad_rate,
        "woe": woe,
        "iv_contribution": iv_contribution,
        "woe_group_count": group_count,
        "amount_metrics": _derived_amount_metrics(raw_amounts, count=count),
    }
    effect_identity = {
        "schema_version": "strategy.cross-matrix-cell-group-effect.v1",
        "rule_hash": rule_hash,
        "sample_context_hash": asset["sample_identity"]["sample_context_hash"],
        **effect_body,
    }
    effect_id = _stable_id("candidate-effect", effect_identity)
    effect = {
        "effect_id": effect_id,
        **effect_body,
        "effect_hash": _sha256(
            _canonical_json({**effect_identity, "effect_id": effect_id})
        ),
    }
    return {
        "schema_version": "strategy.cross-matrix-cell-group-fragment.v1",
        "group_id": group_id,
        "cell_ids": ordered_ids,
        "fragment": {
            "fragment_id": group_id,
            "fragment_type": "cross_matrix_cell_group",
            "rule_id": rule_id,
            "condition": condition,
            "requirements": [],
            "effect_id": effect_id,
        },
        "rule": rule,
        "effect": effect,
    }


def cross_matrix_cell_selection_to_verified_candidate_fragment(
    selection_payload: Mapping[str, Any],
    full_matrix_asset: Mapping[str, Any],
    *,
    selection_artifact_binding: Mapping[str, Any],
    source_artifact_binding: IndependentlyVerifiedCrossMatrixArtifactBinding,
) -> dict[str, Any]:
    """Replay a verified selection and matrix into the generic Pool seam."""

    selection = validate_cross_matrix_cell_selection(selection_payload)
    asset = _full_matrix_asset(full_matrix_asset)
    selection_binding = _selection_artifact_binding(selection_artifact_binding)
    source_binding = _verified_source_artifact_binding(source_artifact_binding)
    _require_selection_binding_matches_payload(selection_binding, selection=selection)
    _require_source_binding_matches_selection_pointer(
        source_binding,
        selection=selection,
    )
    _require_source_artifact_binding_matches_asset(source_binding, asset=asset)
    _require_selection_replays_asset(selection, asset=asset)
    facts = derive_cross_matrix_cell_group_facts(asset, cell_ids=selection["cell_ids"])
    if facts["group_id"] != selection["group_id"]:
        raise CrossMatrixCellSelectionError(
            "group_id does not match replayed Cross Matrix cells"
        )
    fragment = facts["fragment"]
    evidence = asset["candidate_evidence"]
    sample = asset["sample_identity"]
    lifecycle = asset["lifecycle"]
    try:
        return build_verified_candidate_fragment(
            artifact={
                "artifact_id": selection_binding["artifact_id"],
                "artifact_kind": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
                "artifact_schema_version": (
                    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION
                ),
                "artifact_content_hash": selection_binding["content_hash"],
                "origin_tool": CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
            },
            asset={
                "schema_version": asset["schema_version"],
                "asset_id": asset["asset_id"],
                "asset_hash": asset["asset_hash"],
                "asset_type": asset["asset_type"],
            },
            fragment_id=fragment["fragment_id"],
            fragment_type=fragment["fragment_type"],
            rule_id=fragment["rule_id"],
            condition=fragment["condition"],
            requirements=fragment["requirements"],
            effect_id=fragment["effect_id"],
            evidence_id=evidence["candidate_id"],
            evidence_hash=evidence["evidence_hash"],
            evidence_identity={key: sample[key] for key in _EVIDENCE_IDENTITY_FIELDS},
            candidate_stage=lifecycle["candidate_stage"],
            observation_stage=lifecycle["observation_stage"],
            validation_status=lifecycle["validation_status"],
        )
    except CandidateFragmentError as exc:
        raise CrossMatrixCellSelectionError(
            "Cross Matrix cell group failed generic fragment projection"
        ) from exc


def _normalize_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(payload, _BODY_FIELDS, "cell selection body")
    if payload["schema_version"] != CROSS_MATRIX_CELL_SELECTION_SCHEMA_VERSION:
        raise CrossMatrixCellSelectionError(
            "schema_version must be " + CROSS_MATRIX_CELL_SELECTION_SCHEMA_VERSION
        )
    producer = _canonical_text(payload["producer_version"], "producer_version")
    if producer != CROSS_MATRIX_CELL_SELECTION_PRODUCER_VERSION:
        raise CrossMatrixCellSelectionError(
            "producer_version must be " + CROSS_MATRIX_CELL_SELECTION_PRODUCER_VERSION
        )
    cell_ids = _stored_cell_ids(payload["cell_ids"])
    group_id = _identifier(payload["group_id"], "group_id", pattern=_GROUP_ID_RE)
    source_artifact = _source_artifact(payload["source_artifact"])
    source_asset = _source_asset_reference(payload["source_asset"])
    _require_source_version_pair(
        source_artifact["artifact_schema_version"],
        source_asset["schema_version"],
        name="cell selection source",
    )
    return {
        "schema_version": CROSS_MATRIX_CELL_SELECTION_SCHEMA_VERSION,
        "source_artifact": source_artifact,
        "source_asset": source_asset,
        "source_candidate": _source_candidate_reference(payload["source_candidate"]),
        "group_id": group_id,
        "cell_ids": cell_ids,
        "selection_reason": _canonical_selection_reason(payload["selection_reason"]),
        "producer_version": producer,
    }


def _full_matrix_asset(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError("full Cross Matrix asset must be an object")
    try:
        asset = validate_cross_matrix_candidate_asset(value)
    except CrossMatrixCandidateAssetError as exc:
        raise CrossMatrixCellSelectionError(
            "full Cross Matrix asset failed strict validation"
        ) from exc
    if asset["schema_version"] not in _SOURCE_ASSET_TO_ARTIFACT_SCHEMA or (
        asset["asset_type"] != CROSS_MATRIX_CANDIDATE_ASSET_TYPE
    ):
        raise CrossMatrixCellSelectionError(
            "source asset must use the committed Cross Matrix contract"
        )
    return asset


def _source_artifact(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError("source_artifact must be an object")
    _exact_fields(value, _SOURCE_ARTIFACT_FIELDS, "source_artifact")
    kind = _canonical_text(value["kind"], "source_artifact.kind")
    schema = _canonical_text(
        value["artifact_schema_version"],
        "source_artifact.artifact_schema_version",
    )
    origin = _canonical_text(value["origin_tool"], "source_artifact.origin_tool")
    if kind != CROSS_MATRIX_SOURCE_ARTIFACT_KIND:
        raise CrossMatrixCellSelectionError(
            "source_artifact.kind must be " + CROSS_MATRIX_SOURCE_ARTIFACT_KIND
        )
    if schema not in _SOURCE_VERSION_CONTRACTS:
        raise CrossMatrixCellSelectionError(
            "source_artifact artifact schema is unsupported"
        )
    if origin != CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL:
        raise CrossMatrixCellSelectionError(
            "source_artifact.origin_tool must be "
            + CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL
        )
    return {
        "artifact_id": _canonical_text(
            value["artifact_id"], "source_artifact.artifact_id"
        ),
        "task_id": _canonical_text(value["task_id"], "source_artifact.task_id"),
        "kind": kind,
        "artifact_schema_version": schema,
        "content_hash": _hash(value["content_hash"], "source_artifact.content_hash"),
        "origin_tool": origin,
        "path": _canonical_text(value["path"], "source_artifact.path"),
        "provenance_hash": _hash(
            value["provenance_hash"], "source_artifact.provenance_hash"
        ),
    }


def _verified_source_artifact_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError("source_artifact_binding must be an object")
    _exact_fields(value, _SOURCE_BINDING_FIELDS, "source_artifact_binding")
    pointer = _source_artifact(
        {field: value[field] for field in _SOURCE_ARTIFACT_FIELDS}
    )
    provenance = _source_provenance(value["provenance"])
    expected_provenance_hash = _sha256(_canonical_json(provenance))
    if not hmac.compare_digest(pointer["provenance_hash"], expected_provenance_hash):
        raise CrossMatrixCellSelectionError(
            "source_artifact provenance_hash does not match provenance"
        )
    canonical_bytes = value["canonical_bytes"]
    if not isinstance(canonical_bytes, bytes):
        raise CrossMatrixCellSelectionError(
            "source_artifact_binding.canonical_bytes must be bytes"
        )
    return {**pointer, "provenance": provenance, "canonical_bytes": canonical_bytes}


def _source_artifact_pointer(binding: Mapping[str, Any]) -> dict[str, Any]:
    return _source_artifact(
        {field: binding[field] for field in _SOURCE_ARTIFACT_FIELDS}
    )


def _source_asset_reference(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError("source_asset must be an object")
    _exact_fields(value, _SOURCE_ASSET_FIELDS, "source_asset")
    schema = _canonical_text(value["schema_version"], "source_asset.schema_version")
    asset_type = _canonical_text(value["asset_type"], "source_asset.asset_type")
    if schema not in _SOURCE_ASSET_TO_ARTIFACT_SCHEMA:
        raise CrossMatrixCellSelectionError(
            "source_asset.schema_version is unsupported"
        )
    if asset_type != CROSS_MATRIX_CANDIDATE_ASSET_TYPE:
        raise CrossMatrixCellSelectionError(
            "source_asset.asset_type must be cross_matrix"
        )
    return {
        "schema_version": schema,
        "asset_id": _identifier(
            value["asset_id"], "source_asset.asset_id", pattern=_ASSET_ID_RE
        ),
        "asset_hash": _hash(value["asset_hash"], "source_asset.asset_hash"),
        "asset_type": asset_type,
    }


def _source_asset_reference_from_asset(asset: Mapping[str, Any]) -> dict[str, str]:
    return _source_asset_reference(
        {
            "schema_version": asset["schema_version"],
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "asset_type": asset["asset_type"],
        }
    )


def _source_candidate_reference(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError("source_candidate must be an object")
    _exact_fields(value, _SOURCE_CANDIDATE_FIELDS, "source_candidate")
    return {
        "candidate_id": _identifier(
            value["candidate_id"],
            "source_candidate.candidate_id",
            pattern=_CANDIDATE_ID_RE,
        ),
        "evidence_hash": _hash(
            value["evidence_hash"], "source_candidate.evidence_hash"
        ),
        "evidence_identity": _evidence_identity(value["evidence_identity"]),
    }


def _source_candidate_reference_from_asset(
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = asset["candidate_evidence"]
    sample = asset["sample_identity"]
    return _source_candidate_reference(
        {
            "candidate_id": evidence["candidate_id"],
            "evidence_hash": evidence["evidence_hash"],
            "evidence_identity": {
                key: sample[key] for key in _EVIDENCE_IDENTITY_FIELDS
            },
        }
    )


def _evidence_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError(
            "source_candidate.evidence_identity must be an object"
        )
    _exact_fields(
        value,
        _EVIDENCE_IDENTITY_FIELDS,
        "source_candidate.evidence_identity",
    )
    return {
        "dataset_id": _canonical_text(value["dataset_id"], "evidence dataset_id"),
        "dataset_content_hash": _hash(
            value["dataset_content_hash"], "evidence dataset_content_hash"
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"], "evidence workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"], "evidence workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "evidence semantic_mapping_hash"
        ),
        "sample_context_hash": _hash(
            value["sample_context_hash"], "evidence sample_context_hash"
        ),
    }


def _source_provenance(value: object) -> dict[str, Any]:
    provenance = _canonical_json_object(value, "source artifact provenance")
    _exact_fields(provenance, _SOURCE_PROVENANCE_FIELDS, "source artifact provenance")
    schema_version = _canonical_text(
        provenance["schema_version"],
        "source artifact provenance schema_version",
    )
    version_contract = _SOURCE_VERSION_CONTRACTS.get(schema_version)
    if version_contract is None:
        raise CrossMatrixCellSelectionError(
            "source artifact provenance schema_version is unsupported"
        )
    expected_asset_schema, expected_producer = version_contract
    fixed = {
        "producer_version": expected_producer,
        "asset_schema_version": expected_asset_schema,
        "asset_type": CROSS_MATRIX_CANDIDATE_ASSET_TYPE,
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
        "truncated": False,
    }
    for field, expected in fixed.items():
        actual = provenance[field]
        matches = (
            isinstance(actual, bool) and actual is expected
            if isinstance(expected, bool)
            else actual == expected
        )
        if not matches:
            raise CrossMatrixCellSelectionError(
                f"source artifact provenance {field} changed"
            )
    for field in (
        "producer_version",
        "source_artifact_id",
        "task_id",
        "dataset_id",
        "target_col",
    ):
        provenance[field] = _canonical_text(
            provenance[field], f"source artifact provenance {field}"
        )
    for field in (
        "asset_hash",
        "parent_evidence_hash",
        "evidence_hash",
        "source_artifact_content_hash",
        "dataset_content_hash",
        "registry_metadata_hash",
        "semantic_mapping_hash",
        "sample_context_hash",
    ):
        provenance[field] = _hash(
            provenance[field], f"source artifact provenance {field}"
        )
    provenance["asset_id"] = _identifier(
        provenance["asset_id"],
        "source artifact provenance asset_id",
        pattern=_ASSET_ID_RE,
    )
    for field in ("parent_candidate_id", "candidate_id"):
        provenance[field] = _identifier(
            provenance[field],
            f"source artifact provenance {field}",
            pattern=_CANDIDATE_ID_RE,
        )
    for field in (
        "workspace_revision",
        "workspace_generation",
        "labeled_row_count",
        "cell_count",
        "budget",
    ):
        provenance[field] = _non_negative_int(
            provenance[field], f"source artifact provenance {field}"
        )
    provenance["row_axis"] = _axis_provenance(
        provenance["row_axis"],
        "source artifact provenance row_axis",
        schema_version=schema_version,
    )
    provenance["column_axis"] = _axis_provenance(
        provenance["column_axis"],
        "source artifact provenance column_axis",
        schema_version=schema_version,
    )
    return provenance


def _axis_provenance(
    value: object,
    name: str,
    *,
    schema_version: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError(f"{name} must be an object")
    fields = (
        _AXIS_PROVENANCE_V2_FIELDS
        if schema_version == CROSS_MATRIX_SOURCE_ARTIFACT_V2_SCHEMA_VERSION
        else _AXIS_PROVENANCE_V1_FIELDS
    )
    _exact_fields(value, fields, name)
    normalized = {
        "feature": _canonical_text(value["feature"], f"{name}.feature"),
        "method": _canonical_text(value["method"], f"{name}.method"),
    }
    if schema_version == CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION:
        return normalized
    bin_count = _positive_int(value["bin_count"], f"{name}.bin_count")
    if bin_count > 400:
        raise CrossMatrixCellSelectionError(f"{name}.bin_count exceeds 400")
    parent_evidence_hash = _hash(
        value["parent_evidence_hash"],
        f"{name}.parent_evidence_hash",
    )
    manual_breakpoints = value["manual_breakpoints"]
    if normalized["method"] == "manual":
        manual_breakpoints = _manual_breakpoints(
            manual_breakpoints,
            name=f"{name}.manual_breakpoints",
        )
        if len(manual_breakpoints) + 1 > bin_count:
            raise CrossMatrixCellSelectionError(
                f"{name}.manual_breakpoints exceed total bin_count"
            )
    elif manual_breakpoints is not None:
        raise CrossMatrixCellSelectionError(
            f"{name}.manual_breakpoints must be null for non-manual axes"
        )
    return {
        **normalized,
        "bin_count": bin_count,
        "manual_breakpoints": manual_breakpoints,
        "parent_evidence_hash": parent_evidence_hash,
    }


def _manual_breakpoints(value: object, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        str | bytes | bytearray,
    ):
        raise CrossMatrixCellSelectionError(f"{name} must be a non-empty array")
    points = list(value)
    if not points or len(points) + 1 > 20:
        raise CrossMatrixCellSelectionError(
            f"{name} must define between 2 and 20 bins"
        )
    if any(type(item) is not float or not math.isfinite(item) for item in points):
        raise CrossMatrixCellSelectionError(
            f"{name} must contain canonical finite floats"
        )
    if any(left >= right for left, right in zip(points, points[1:])):
        raise CrossMatrixCellSelectionError(
            f"{name} must be strictly increasing and unique"
        )
    return points


def _require_source_version_pair(
    artifact_schema_version: str,
    asset_schema_version: str,
    *,
    name: str,
) -> None:
    contract = _SOURCE_VERSION_CONTRACTS.get(artifact_schema_version)
    if contract is None or contract[0] != asset_schema_version:
        raise CrossMatrixCellSelectionError(
            f"{name} artifact and asset schema versions do not match"
        )


def _axis_provenance_from_asset(
    axis: Mapping[str, Any],
    *,
    asset_schema_version: str,
) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "feature": axis["feature"],
        "method": axis["method"],
    }
    if asset_schema_version == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION:
        projected.update(
            {
                "bin_count": len(axis["bins"]),
                "manual_breakpoints": axis["manual_breakpoints"],
                "parent_evidence_hash": axis["parent_evidence_hash"],
            }
        )
    return projected


def _require_source_artifact_binding_matches_asset(
    binding: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
) -> None:
    expected_bytes = canonical_cross_matrix_candidate_asset_json(asset).encode("utf-8")
    if not hmac.compare_digest(binding["canonical_bytes"], expected_bytes):
        raise CrossMatrixCellSelectionError(
            "source artifact canonical bytes do not match Cross Matrix asset"
        )
    expected_content_hash = _sha256_bytes(expected_bytes)
    if not hmac.compare_digest(binding["content_hash"], expected_content_hash):
        raise CrossMatrixCellSelectionError(
            "source artifact content_hash does not match Cross Matrix asset"
        )
    sample = asset["sample_identity"]
    if binding["task_id"] != sample["task_id"]:
        raise CrossMatrixCellSelectionError(
            "source artifact task_id does not match Cross Matrix asset"
        )
    provenance = binding["provenance"]
    expected_artifact_schema = _SOURCE_ASSET_TO_ARTIFACT_SCHEMA.get(
        asset["schema_version"]
    )
    if (
        expected_artifact_schema is None
        or binding["artifact_schema_version"] != expected_artifact_schema
        or provenance["schema_version"] != expected_artifact_schema
    ):
        raise CrossMatrixCellSelectionError(
            "source artifact and asset schema versions do not match"
        )
    expected = {
        "schema_version": expected_artifact_schema,
        "producer_version": asset["producer_version"],
        "asset_schema_version": asset["schema_version"],
        "asset_type": asset["asset_type"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "parent_candidate_id": asset["parent"]["candidate_id"],
        "parent_evidence_hash": asset["parent"]["evidence_hash"],
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "task_id": sample["task_id"],
        "dataset_id": sample["dataset_id"],
        "dataset_content_hash": sample["dataset_content_hash"],
        "workspace_revision": sample["workspace_revision"],
        "workspace_generation": sample["workspace_generation"],
        "semantic_mapping_hash": sample["semantic_mapping_hash"],
        "sample_context_hash": sample["sample_context_hash"],
        "target_col": sample["target_col"],
        "labeled_row_count": sample["row_count"],
        "row_axis": _axis_provenance_from_asset(
            asset["axes"][0],
            asset_schema_version=asset["schema_version"],
        ),
        "column_axis": _axis_provenance_from_asset(
            asset["axes"][1],
            asset_schema_version=asset["schema_version"],
        ),
        "cell_count": asset["matrix"]["cell_count"],
        "candidate_stage": asset["lifecycle"]["candidate_stage"],
        "observation_stage": asset["lifecycle"]["observation_stage"],
        "validation_status": asset["lifecycle"]["validation_status"],
        "budget": asset["budget"]["limit"],
        "truncated": asset["budget"]["truncated"],
    }
    for field, expected_value in expected.items():
        actual = provenance[field]
        matches = (
            hmac.compare_digest(actual, expected_value)
            if field.endswith("hash")
            else actual == expected_value
        )
        if not matches:
            raise CrossMatrixCellSelectionError(
                f"source artifact provenance {field} does not match Cross Matrix asset"
            )


def _selection_artifact_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError(
            "selection_artifact_binding must be an object"
        )
    _exact_fields(
        value,
        _SELECTION_ARTIFACT_BINDING_FIELDS,
        "selection_artifact_binding",
    )
    normalized = {
        "artifact_id": _canonical_text(value["artifact_id"], "selection artifact_id"),
        "task_id": _canonical_text(value["task_id"], "selection task_id"),
        "kind": _canonical_text(value["kind"], "selection kind"),
        "content_hash": _hash(value["content_hash"], "selection content_hash"),
        "origin_tool": _canonical_text(value["origin_tool"], "selection origin_tool"),
        "artifact_schema_version": _canonical_text(
            value["artifact_schema_version"], "selection artifact_schema_version"
        ),
        "producer_version": _canonical_text(
            value["producer_version"], "selection producer_version"
        ),
        "selection_id": _identifier(
            value["selection_id"],
            "selection binding selection_id",
            pattern=_SELECTION_ID_RE,
        ),
        "selection_hash": _hash(
            value["selection_hash"], "selection binding selection_hash"
        ),
        "group_id": _identifier(
            value["group_id"], "selection binding group_id", pattern=_GROUP_ID_RE
        ),
        "source_artifact_id": _canonical_text(
            value["source_artifact_id"], "selection source_artifact_id"
        ),
        "source_artifact_kind": _canonical_text(
            value["source_artifact_kind"], "selection source_artifact_kind"
        ),
        "source_artifact_schema_version": _canonical_text(
            value["source_artifact_schema_version"],
            "selection source_artifact_schema_version",
        ),
        "source_artifact_content_hash": _hash(
            value["source_artifact_content_hash"],
            "selection source_artifact_content_hash",
        ),
        "source_artifact_origin_tool": _canonical_text(
            value["source_artifact_origin_tool"],
            "selection source_artifact_origin_tool",
        ),
        "source_artifact_path": _canonical_text(
            value["source_artifact_path"], "selection source_artifact_path"
        ),
        "source_artifact_provenance_hash": _hash(
            value["source_artifact_provenance_hash"],
            "selection source_artifact_provenance_hash",
        ),
        "source_asset_schema_version": _canonical_text(
            value["source_asset_schema_version"],
            "selection source_asset_schema_version",
        ),
        "source_asset_id": _identifier(
            value["source_asset_id"], "selection source_asset_id", pattern=_ASSET_ID_RE
        ),
        "source_asset_hash": _hash(
            value["source_asset_hash"], "selection source_asset_hash"
        ),
        "source_asset_type": _canonical_text(
            value["source_asset_type"], "selection source_asset_type"
        ),
        "source_candidate_id": _identifier(
            value["source_candidate_id"],
            "selection source_candidate_id",
            pattern=_CANDIDATE_ID_RE,
        ),
        "source_evidence_hash": _hash(
            value["source_evidence_hash"], "selection source_evidence_hash"
        ),
        "source_evidence_identity": _evidence_identity(
            value["source_evidence_identity"]
        ),
        "cell_ids": _stored_cell_ids(value["cell_ids"]),
    }
    fixed = {
        "kind": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        "origin_tool": CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
        "artifact_schema_version": (
            CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": CROSS_MATRIX_CELL_SELECTION_PRODUCER_VERSION,
        "source_artifact_kind": CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
        "source_artifact_origin_tool": CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
        "source_asset_type": CROSS_MATRIX_CANDIDATE_ASSET_TYPE,
    }
    for field, expected in fixed.items():
        if normalized[field] != expected:
            raise CrossMatrixCellSelectionError(f"selection {field} must be {expected}")
    _require_source_version_pair(
        normalized["source_artifact_schema_version"],
        normalized["source_asset_schema_version"],
        name="selection binding source",
    )
    return normalized


def _require_selection_binding_matches_payload(
    binding: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
) -> None:
    source_artifact = selection["source_artifact"]
    source_asset = selection["source_asset"]
    source_candidate = selection["source_candidate"]
    expected = {
        "task_id": source_artifact["task_id"],
        "content_hash": cross_matrix_cell_selection_content_hash(selection),
        "producer_version": selection["producer_version"],
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "group_id": selection["group_id"],
        "source_artifact_id": source_artifact["artifact_id"],
        "source_artifact_kind": source_artifact["kind"],
        "source_artifact_schema_version": source_artifact["artifact_schema_version"],
        "source_artifact_content_hash": source_artifact["content_hash"],
        "source_artifact_origin_tool": source_artifact["origin_tool"],
        "source_artifact_path": source_artifact["path"],
        "source_artifact_provenance_hash": source_artifact["provenance_hash"],
        "source_asset_schema_version": source_asset["schema_version"],
        "source_asset_id": source_asset["asset_id"],
        "source_asset_hash": source_asset["asset_hash"],
        "source_asset_type": source_asset["asset_type"],
        "source_candidate_id": source_candidate["candidate_id"],
        "source_evidence_hash": source_candidate["evidence_hash"],
        "source_evidence_identity": source_candidate["evidence_identity"],
        "cell_ids": selection["cell_ids"],
    }
    for field, expected_value in expected.items():
        actual = binding[field]
        matches = (
            hmac.compare_digest(actual, expected_value)
            if field.endswith("hash")
            else actual == expected_value
        )
        if not matches:
            raise CrossMatrixCellSelectionError(
                f"selection artifact binding {field} does not match payload"
            )


def _require_source_binding_matches_selection_pointer(
    binding: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
) -> None:
    actual = _source_artifact_pointer(binding)
    expected = selection["source_artifact"]
    for field in sorted(_SOURCE_ARTIFACT_FIELDS):
        actual_value = actual[field]
        expected_value = expected[field]
        matches = (
            hmac.compare_digest(actual_value, expected_value)
            if field.endswith("hash")
            else actual_value == expected_value
        )
        if not matches:
            raise CrossMatrixCellSelectionError(
                f"source artifact binding {field} does not match selection pointer"
            )


def _require_selection_replays_asset(
    selection: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
) -> None:
    expected_asset = _source_asset_reference_from_asset(asset)
    if selection["source_asset"] != expected_asset:
        raise CrossMatrixCellSelectionError(
            "source asset identity does not match cell selection"
        )
    expected_candidate = _source_candidate_reference_from_asset(asset)
    if selection["source_candidate"] != expected_candidate:
        raise CrossMatrixCellSelectionError(
            "source candidate evidence identity does not match cell selection"
        )
    facts = derive_cross_matrix_cell_group_facts(asset, cell_ids=selection["cell_ids"])
    if facts["cell_ids"] != selection["cell_ids"]:
        raise CrossMatrixCellSelectionError(
            "cell_ids are not in canonical source row-major order"
        )
    if facts["group_id"] != selection["group_id"]:
        raise CrossMatrixCellSelectionError(
            "group_id does not match selected source cells"
        )


def _requested_cell_ids(value: object) -> list[str]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise CrossMatrixCellSelectionError("cell_ids must be a non-empty array")
    if not value:
        raise CrossMatrixCellSelectionError("cell_ids must contain at least one cell")
    normalized = [
        _identifier(item, f"cell_ids[{index}]", pattern=_CELL_ID_RE)
        for index, item in enumerate(value)
    ]
    if len(set(normalized)) != len(normalized):
        raise CrossMatrixCellSelectionError("cell_ids must not contain duplicates")
    return normalized


def _stored_cell_ids(value: object) -> list[str]:
    return _requested_cell_ids(value)


def _aggregate_amount_observations(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in ("loan_amount", "overdue_amount"):
        items = [cell["amounts"][dimension] for cell in cells]
        if any(item["status"] == "unavailable" for item in items):
            result[dimension] = {
                "status": "unavailable",
                "covered_count": None,
                "value": None,
            }
        else:
            result[dimension] = {
                "status": "available",
                "covered_count": sum(item["covered_count"] for item in items),
                "value": sum(item["value"] for item in items),
            }
    paired_items = [cell["amounts"]["paired"] for cell in cells]
    if any(item["status"] == "unavailable" for item in paired_items):
        result["paired"] = {
            "status": "unavailable",
            "covered_count": None,
            "loan_value": None,
            "overdue_value": None,
        }
    else:
        result["paired"] = {
            "status": "available",
            "covered_count": sum(item["covered_count"] for item in paired_items),
            "loan_value": sum(item["loan_value"] for item in paired_items),
            "overdue_value": sum(item["overdue_value"] for item in paired_items),
        }
    return result


def _derived_amount_metrics(
    amounts: Mapping[str, Any],
    *,
    count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in ("loan_amount", "overdue_amount"):
        item = amounts[dimension]
        if item["status"] == "unavailable":
            result[dimension] = {
                "status": "unavailable",
                "covered_count": None,
                "coverage_rate": None,
                "value": None,
                "reason": "column_unavailable",
            }
        else:
            result[dimension] = {
                "status": "available",
                "covered_count": item["covered_count"],
                "coverage_rate": item["covered_count"] / count,
                "value": item["value"],
                "reason": None,
            }
    paired = amounts["paired"]
    if paired["status"] == "unavailable":
        result["overdue_rate"] = {
            "status": "unavailable",
            "covered_count": None,
            "coverage_rate": None,
            "value": None,
            "reason": "columns_unavailable",
        }
    elif paired["covered_count"] == 0:
        result["overdue_rate"] = {
            "status": "not_applicable",
            "covered_count": 0,
            "coverage_rate": 0.0,
            "value": None,
            "reason": "no_paired_observations",
        }
    elif paired["loan_value"] == 0:
        result["overdue_rate"] = {
            "status": "not_applicable",
            "covered_count": paired["covered_count"],
            "coverage_rate": paired["covered_count"] / count,
            "value": None,
            "reason": "zero_loan_amount",
        }
    else:
        result["overdue_rate"] = {
            "status": "available",
            "covered_count": paired["covered_count"],
            "coverage_rate": paired["covered_count"] / count,
            "value": paired["overdue_value"] / paired["loan_value"],
            "reason": None,
        }
    return result


def _canonical_condition(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossMatrixCellSelectionError(f"{name} must be an object")
    try:
        return canonicalize_expression(value)
    except StrategyError as exc:
        raise CrossMatrixCellSelectionError(f"{name} is invalid") from exc


def _canonicalize_selection_reason(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CrossMatrixCellSelectionError("selection_reason must be a string or null")
    normalized = " ".join(value.split())
    if not normalized:
        return None
    canonical = _canonical_text(
        unicodedata.normalize("NFC", normalized), "selection_reason"
    )
    if len(canonical) > MAX_SELECTION_REASON_LENGTH:
        raise CrossMatrixCellSelectionError(
            "selection_reason must be at most 500 characters"
        )
    return canonical


def _canonical_selection_reason(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CrossMatrixCellSelectionError("selection_reason must be a string or null")
    canonical = _canonicalize_selection_reason(value)
    if canonical != value:
        raise CrossMatrixCellSelectionError(
            "selection_reason must already be canonical text"
        )
    return canonical


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str] | set[str], name: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(str(field) for field in actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields " + ", ".join(unexpected))
        raise CrossMatrixCellSelectionError(f"{name} has " + "; ".join(details))


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CrossMatrixCellSelectionError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise CrossMatrixCellSelectionError(f"{name} must not contain NUL")
    canonical = unicodedata.normalize("NFC", value)
    if value != canonical or value != value.strip():
        raise CrossMatrixCellSelectionError(f"{name} must be canonical text")
    return value


def _identifier(value: object, name: str, *, pattern: re.Pattern[str]) -> str:
    normalized = _canonical_text(value, name)
    if pattern.fullmatch(normalized) is None:
        raise CrossMatrixCellSelectionError(f"{name} has an invalid format")
    return normalized


def _hash(value: object, name: str) -> str:
    normalized = _canonical_text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise CrossMatrixCellSelectionError(f"{name} must be a lowercase SHA-256")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise CrossMatrixCellSelectionError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized == 0:
        raise CrossMatrixCellSelectionError(f"{name} must be positive")
    return normalized


def _positive_number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CrossMatrixCellSelectionError(f"{name} must be a positive number")
    normalized: int | float = (
        int(value) if isinstance(value, Integral) else float(value)
    )
    if not math.isfinite(float(normalized)) or normalized <= 0:
        raise CrossMatrixCellSelectionError(f"{name} must be a positive number")
    return normalized


def _canonical_json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CrossMatrixCellSelectionError(f"{name} must be an object")
    try:
        normalized = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CrossMatrixCellSelectionError(f"{name} must contain finite JSON") from exc
    if not isinstance(normalized, dict):
        raise CrossMatrixCellSelectionError(f"{name} must be an object")
    return normalized


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CrossMatrixCellSelectionError(
            "cell selection must contain finite canonical JSON"
        ) from exc


__all__ = [
    "CELL_SELECTION_ARTIFACT_KIND",
    "CELL_SELECTION_ARTIFACT_SCHEMA_VERSION",
    "CELL_SELECTION_ORIGIN_TOOL",
    "CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND",
    "CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION",
    "CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL",
    "CROSS_MATRIX_CELL_SELECTION_PRODUCER_VERSION",
    "CROSS_MATRIX_CELL_SELECTION_SCHEMA_VERSION",
    "CROSS_MATRIX_SOURCE_ARTIFACT_KIND",
    "CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL",
    "CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION",
    "CROSS_MATRIX_SOURCE_ARTIFACT_V2_SCHEMA_VERSION",
    "CrossMatrixCellSelectionError",
    "IndependentlyVerifiedCrossMatrixArtifactBinding",
    "MAX_SELECTION_REASON_LENGTH",
    "build_cross_matrix_cell_selection",
    "canonical_cross_matrix_cell_selection_json",
    "cross_matrix_cell_selection_content_hash",
    "cross_matrix_cell_selection_to_verified_candidate_fragment",
    "derive_cross_matrix_cell_group_facts",
    "validate_cross_matrix_source_artifact_binding",
    "validate_cross_matrix_cell_selection",
]
