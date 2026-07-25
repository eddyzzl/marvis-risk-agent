"""Pure governed scorecard-band candidates and pointer-only cutoff selections.

The full band asset is deterministic evidence derived from one authenticated
raw bad-probability vector and its governed risk/development sample.  It does
not recommend, select, adopt, or deploy a cutoff.  A separate immutable
selection may point at one measured cutoff, and only replay of that pointer
against the independently authenticated full asset can produce the generic
candidate fragment accepted by Strategy Pool.

This module has no registry, database, or filesystem authority.  Artifact
bindings accepted by the selection/replay seams are caller-authenticated
snapshots; this module only freezes and cross-checks their canonical bytes.
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

import numpy as np

from marvis.feature.metrics import feature_auc, feature_ks
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.scoring import scorecard_points_from_raw_pd
from marvis.packs.strategy.candidate_fragment import (
    CandidateFragmentError,
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_requirement_resolver import (
    model_score_virtual_field,
)


SCORECARD_BAND_ASSET_SCHEMA_VERSION = "strategy.scorecard-band-asset.v1"
SCORECARD_BAND_ASSET_PRODUCER_VERSION = "strategy.scorecard-band-asset/1"
SCORECARD_BAND_ASSET_TYPE = "scorecard_band"
SCORECARD_BAND_ASSET_ARTIFACT_KIND = "strategy_scorecard_band_asset_json"
SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION = (
    "strategy.scorecard-band-asset-artifact.v1"
)
SCORECARD_BAND_ASSET_ORIGIN_TOOL = "strategy.build_scorecard_band_asset"

SCORECARD_CUTOFF_SELECTION_SCHEMA_VERSION = (
    "strategy.scorecard-cutoff-selection.v1"
)
SCORECARD_CUTOFF_SELECTION_PRODUCER_VERSION = (
    "strategy.scorecard-cutoff-selection/1"
)
SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND = (
    "strategy_scorecard_cutoff_selection_json"
)
SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION = (
    "strategy.scorecard-cutoff-selection-artifact.v1"
)
SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL = (
    "strategy.materialize_scorecard_cutoff_selection"
)

RAW_BAD_PROBABILITY_SCORE_PRODUCT = (
    "raw_native_uncalibrated_bad_probability"
)
MODEL_SCORE_DIRECTION = "higher_is_riskier"
SCORECARD_POINTS_DIRECTION = "higher_is_better"

MAX_SCORECARD_CANDIDATE_ROWS = 2_000_000
MAX_SCORECARD_BANDS = 20
MIN_SCORECARD_BANDS = 2
MAX_SCORECARD_TABLE_ROWS = 100_000
MAX_SCORECARD_CANDIDATE_JSON_BYTES = 16 * 1024 * 1024
MAX_SELECTION_REASON_LENGTH = 500

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^scorecard-band-asset-[0-9a-f]{32}$")
_CUTOFF_ID_RE = re.compile(r"^scorecard-cutoff-[0-9a-f]{32}$")
_SELECTION_ID_RE = re.compile(
    r"^scorecard-cutoff-selection-[0-9a-f]{32}$"
)

_ASSET_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "asset_type",
        "identity",
        "sample_design_ref",
        "source_refs",
        "score_contract",
        "score_vector",
        "bands",
        "performance",
        "cutoffs",
        "lifecycle",
        "governance",
        "resource_budget",
        "asset_id",
        "asset_hash",
    }
)
_ASSET_BODY_FIELDS = _ASSET_TOP_LEVEL_FIELDS - {"asset_id", "asset_hash"}
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
_SAMPLE_DESIGN_REF_FIELDS = frozenset(
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
_SOURCE_REFS_FIELDS = frozenset(
    {"training_evidence", "score_evidence", "score_vector"}
)
_EVIDENCE_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_content_hash",
        "evidence_id",
        "evidence_content_hash",
    }
)
_VECTOR_REF_FIELDS = frozenset({"artifact_id", "artifact_content_hash"})
_SCORE_CONTRACT_FIELDS = frozenset(
    {
        "score_product",
        "score_direction",
        "points_direction",
        "scale",
        "scorecard_table",
        "scorecard_table_hash",
    }
)
_SCALE_FIELDS = frozenset(
    {"base_score", "pdo", "base_odds", "factor", "offset"}
)
_BASE_SCORECARD_ROW_FIELDS = frozenset(
    {
        "feature",
        "bin_index",
        "bin_label",
        "lower",
        "upper",
        "count",
        "bad_count",
        "good_count",
        "bad_rate",
        "woe",
        "iv_contribution",
        "coefficient",
        "monotonic_direction",
        "points",
        "base_score",
        "pdo",
        "base_odds",
        "factor",
        "offset",
    }
)
_FEATURE_SCORECARD_ROW_FIELDS = _BASE_SCORECARD_ROW_FIELDS - _SCALE_FIELDS
_SCORE_VECTOR_FIELDS = frozenset(
    {
        "row_count",
        "development_count",
        "labeled_count",
        "bad_count",
        "raw_pd_content_hash",
    }
)
_INPUT_BIN_FIELDS = frozenset(
    {
        "ordinal",
        "bin_id",
        "lower_bound",
        "upper_bound",
        "lower_inclusive",
        "upper_inclusive",
    }
)
_BAND_FIELDS = _INPUT_BIN_FIELDS | {
    "count",
    "share",
    "labeled_count",
    "bad_count",
    "bad_rate",
    "average_pd",
}
_PERFORMANCE_FIELDS = frozenset({"auc", "ks"})
_CUTOFF_FIELDS = frozenset(
    {
        "ordinal",
        "cutoff_id",
        "execution_pd",
        "display_points",
        "lower_risk",
        "higher_risk",
        "mask_equivalence",
    }
)
_CUTOFF_SIDE_FIELDS = frozenset(
    {"count", "labeled_count", "bad_count", "bad_rate"}
)
_LIFECYCLE_FIELDS = frozenset(
    {"candidate_stage", "observation_stage", "validation_status"}
)
_GOVERNANCE_FIELDS = frozenset(
    {
        "best_cutoff_recommended",
        "selected",
        "adopted",
        "deployed",
    }
)
_RESOURCE_BUDGET_FIELDS = frozenset(
    {
        "max_rows",
        "rows_processed",
        "max_bands",
        "bands_used",
        "max_scorecard_table_rows",
        "scorecard_table_rows",
    }
)
_SELECTION_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "source_asset_ref",
        "cutoff_id",
        "selection_reason",
        "selection_id",
        "selection_hash",
    }
)
_SELECTION_BODY_FIELDS = _SELECTION_TOP_LEVEL_FIELDS - {
    "selection_id",
    "selection_hash",
}
_SOURCE_ASSET_POINTER_FIELDS = frozenset(
    {
        "artifact_id",
        "task_id",
        "kind",
        "artifact_schema_version",
        "artifact_content_hash",
        "origin_tool",
        "asset_schema_version",
        "asset_type",
        "asset_id",
        "asset_hash",
    }
)
_VERIFIED_ARTIFACT_BINDING_FIELDS = frozenset(
    {
        "artifact_id",
        "task_id",
        "kind",
        "artifact_schema_version",
        "content_hash",
        "origin_tool",
        "canonical_bytes",
    }
)


class ScorecardCandidateError(StrategyError):
    """A scorecard candidate or pointer replay failed closed."""


class IndependentlyVerifiedScorecardArtifactBinding(TypedDict):
    """Caller-authenticated canonical snapshot of one full-band artifact."""

    artifact_id: str
    task_id: str
    kind: str
    artifact_schema_version: str
    content_hash: str
    origin_tool: str
    canonical_bytes: bytes


class IndependentlyVerifiedScorecardSelectionArtifactBinding(TypedDict):
    """Caller-authenticated canonical snapshot of one selection artifact."""

    artifact_id: str
    task_id: str
    kind: str
    artifact_schema_version: str
    content_hash: str
    origin_tool: str
    canonical_bytes: bytes


def build_scorecard_band_asset(
    *,
    identity: Mapping[str, Any],
    sample_design_ref: Mapping[str, Any],
    training_evidence_ref: Mapping[str, Any],
    score_evidence_ref: Mapping[str, Any],
    score_vector_ref: Mapping[str, Any],
    score_product: str,
    score_direction: str,
    points_direction: str,
    scorecard_scale: Mapping[str, Any],
    scorecard_table: Sequence[Mapping[str, Any]],
    raw_pd: Sequence[float] | np.ndarray,
    risk_development_mask: Sequence[bool] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    score_bins: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build all measured score bands and every internal cutoff.

    Inputs are values from already-authenticated TrainingEvidence,
    ModelScoreEvidence, score-vector, and SampleDesign artifacts.  This pure
    seam still checks their exact shapes and binds their identities, but the
    caller remains responsible for proving registry liveness and provenance.
    """

    normalized_identity = _identity(identity)
    normalized_sample = _sample_design_ref(sample_design_ref)
    training_ref = _evidence_ref(
        training_evidence_ref, name="training_evidence_ref"
    )
    score_ref = _evidence_ref(score_evidence_ref, name="score_evidence_ref")
    vector_ref = _vector_ref(score_vector_ref)
    _directions(
        score_product=score_product,
        score_direction=score_direction,
        points_direction=points_direction,
    )
    scale = _scorecard_scale(scorecard_scale)
    table = _scorecard_table(scorecard_table, scale=scale)
    probabilities = _probability_vector(raw_pd)
    development = _boolean_vector(
        risk_development_mask,
        name="risk_development_mask",
        row_count=len(probabilities),
    )
    normalized_labels = _label_vector(labels, row_count=len(probabilities))
    bins = _input_bins(score_bins)
    development_count = int(np.count_nonzero(development))
    if development_count <= 0:
        raise ScorecardCandidateError(
            "risk/development sample must not be empty"
        )
    development_labels = normalized_labels[development]
    labeled_values = development_labels[np.isfinite(development_labels)]
    if set(labeled_values.astype(int).tolist()) != {0, 1}:
        raise ScorecardCandidateError(
            "risk/development labeled sample must contain both classes"
        )

    edges = np.asarray(
        [float(item["upper_bound"]) for item in bins[:-1]],
        dtype=np.float64,
    )
    assigned = np.searchsorted(edges, probabilities, side="right")
    measured_bands = [
        _measured_band(
            definition,
            ordinal=index,
            probabilities=probabilities,
            labels=normalized_labels,
            mask=np.logical_and(development, assigned == index),
            development_count=development_count,
        )
        for index, definition in enumerate(bins)
    ]
    if any(item["count"] <= 0 for item in measured_bands):
        raise ScorecardCandidateError(
            "caller-provided score bins must each contain development rows"
        )

    try:
        points = scorecard_points_from_raw_pd(
            probabilities,
            factor=scale["factor"],
            offset=scale["offset"],
        )
        boundary_points = scorecard_points_from_raw_pd(
            edges,
            factor=scale["factor"],
            offset=scale["offset"],
        )
    except ModelingError as exc:
        raise ScorecardCandidateError(str(exc)) from exc
    measured_cutoffs: list[dict[str, Any]] = []
    for ordinal, (edge, display_points) in enumerate(
        zip(edges, boundary_points, strict=True)
    ):
        higher_pd = probabilities >= edge
        higher_points = points <= display_points
        if not np.array_equal(
            np.logical_and(development, higher_pd),
            np.logical_and(development, higher_points),
        ):
            raise ScorecardCandidateError(
                "scorecard points and raw-PD cutoff masks are not equivalent"
            )
        cutoff_basis = {
            "score_vector_artifact_id": vector_ref["artifact_id"],
            "execution_pd": float(edge),
            "ordinal": ordinal,
        }
        measured_cutoffs.append(
            {
                "ordinal": ordinal,
                "cutoff_id": _stable_id("scorecard-cutoff", cutoff_basis),
                "execution_pd": float(edge),
                "display_points": _finite_number(
                    display_points, "cutoff display_points"
                ),
                "lower_risk": _cutoff_side(
                    development & ~higher_pd,
                    labels=normalized_labels,
                ),
                "higher_risk": _cutoff_side(
                    development & higher_pd,
                    labels=normalized_labels,
                ),
                "mask_equivalence": True,
            }
        )

    development_probability = probabilities[development]
    development_target = normalized_labels[development]
    body = {
        "schema_version": SCORECARD_BAND_ASSET_SCHEMA_VERSION,
        "producer_version": SCORECARD_BAND_ASSET_PRODUCER_VERSION,
        "asset_type": SCORECARD_BAND_ASSET_TYPE,
        "identity": normalized_identity,
        "sample_design_ref": normalized_sample,
        "source_refs": {
            "training_evidence": training_ref,
            "score_evidence": score_ref,
            "score_vector": vector_ref,
        },
        "score_contract": {
            "score_product": RAW_BAD_PROBABILITY_SCORE_PRODUCT,
            "score_direction": MODEL_SCORE_DIRECTION,
            "points_direction": SCORECARD_POINTS_DIRECTION,
            "scale": scale,
            "scorecard_table": table,
            "scorecard_table_hash": _sha256(_canonical_json(table)),
        },
        "score_vector": {
            "row_count": int(len(probabilities)),
            "development_count": development_count,
            "labeled_count": int(np.count_nonzero(np.isfinite(development_target))),
            "bad_count": int(np.nansum(development_target)),
            "raw_pd_content_hash": _float64_vector_hash(probabilities),
        },
        "bands": measured_bands,
        "performance": {
            "auc": float(feature_auc(development_probability, development_target)),
            "ks": float(feature_ks(development_probability, development_target)),
        },
        "cutoffs": measured_cutoffs,
        "lifecycle": _lifecycle(),
        "governance": _governance(),
        "resource_budget": {
            "max_rows": MAX_SCORECARD_CANDIDATE_ROWS,
            "rows_processed": int(len(probabilities)),
            "max_bands": MAX_SCORECARD_BANDS,
            "bands_used": len(measured_bands),
            "max_scorecard_table_rows": MAX_SCORECARD_TABLE_ROWS,
            "scorecard_table_rows": len(table),
        },
    }
    normalized_body = _normalize_asset_body(body)
    asset_id = _stable_id("scorecard-band-asset", normalized_body)
    without_hash = {**normalized_body, "asset_id": asset_id}
    result = {
        **without_hash,
        "asset_hash": _sha256(_canonical_json(without_hash)),
    }
    return validate_scorecard_band_asset(result)


def validate_scorecard_band_asset(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact fields, conservation, lifecycle, stable id, and hash."""

    if not isinstance(payload, Mapping):
        raise ScorecardCandidateError("scorecard band asset must be an object")
    _exact_fields(payload, _ASSET_TOP_LEVEL_FIELDS, "scorecard band asset")
    asset_id = _canonical_text(payload["asset_id"], "asset_id")
    if _ASSET_ID_RE.fullmatch(asset_id) is None:
        raise ScorecardCandidateError("asset_id has an invalid format")
    asset_hash = _hash(payload["asset_hash"], "asset_hash")
    body = _normalize_asset_body(
        {
            key: payload[key]
            for key in payload
            if key not in {"asset_id", "asset_hash"}
        }
    )
    expected_id = _stable_id("scorecard-band-asset", body)
    if not hmac.compare_digest(asset_id, expected_id):
        raise ScorecardCandidateError(
            "asset_id does not match canonical scorecard band asset"
        )
    without_hash = {**body, "asset_id": asset_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(asset_hash, expected_hash):
        raise ScorecardCandidateError(
            "asset_hash does not match canonical scorecard band asset"
        )
    normalized = {**without_hash, "asset_hash": asset_hash}
    if (
        len(_canonical_json(normalized).encode("utf-8"))
        > MAX_SCORECARD_CANDIDATE_JSON_BYTES
    ):
        raise ScorecardCandidateError(
            "scorecard band asset exceeds JSON byte budget"
        )
    return normalized


def canonical_scorecard_band_asset_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole canonical JSON encoding of a valid band asset."""

    return _canonical_json(validate_scorecard_band_asset(payload))


def scorecard_band_asset_content_hash(
    payload: Mapping[str, Any],
) -> str:
    """Hash canonical persisted bytes of one full band asset."""

    return _sha256(canonical_scorecard_band_asset_json(payload))


def build_scorecard_cutoff_selection(
    full_band_asset: Mapping[str, Any],
    *,
    source_artifact_binding: IndependentlyVerifiedScorecardArtifactBinding,
    cutoff_id: str,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    """Persist only an explicit pointer to one measured internal cutoff."""

    asset = validate_scorecard_band_asset(full_band_asset)
    binding = _verified_artifact_binding(
        source_artifact_binding,
        name="source_artifact_binding",
        expected_kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        expected_schema=SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
        expected_origin=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    )
    _require_asset_binding_matches_asset(binding, asset=asset)
    normalized_cutoff = _canonical_text(cutoff_id, "cutoff_id")
    if _CUTOFF_ID_RE.fullmatch(normalized_cutoff) is None:
        raise ScorecardCandidateError("cutoff_id has an invalid format")
    _cutoff_by_id(asset, normalized_cutoff)
    reason = _selection_reason(selection_reason)
    body = _normalize_selection_body(
        {
            "schema_version": SCORECARD_CUTOFF_SELECTION_SCHEMA_VERSION,
            "producer_version": SCORECARD_CUTOFF_SELECTION_PRODUCER_VERSION,
            "source_asset_ref": _source_asset_pointer(
                {
                    "artifact_id": binding["artifact_id"],
                    "task_id": binding["task_id"],
                    "kind": binding["kind"],
                    "artifact_schema_version": binding[
                        "artifact_schema_version"
                    ],
                    "artifact_content_hash": binding["content_hash"],
                    "origin_tool": binding["origin_tool"],
                    "asset_schema_version": asset["schema_version"],
                    "asset_type": asset["asset_type"],
                    "asset_id": asset["asset_id"],
                    "asset_hash": asset["asset_hash"],
                }
            ),
            "cutoff_id": normalized_cutoff,
            "selection_reason": reason,
        }
    )
    selection_id = _stable_id("scorecard-cutoff-selection", body)
    without_hash = {**body, "selection_id": selection_id}
    result = {
        **without_hash,
        "selection_hash": _sha256(_canonical_json(without_hash)),
    }
    return validate_scorecard_cutoff_selection(result)


def validate_scorecard_cutoff_selection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact pointer-only cutoff-selection audit event."""

    if not isinstance(payload, Mapping):
        raise ScorecardCandidateError("scorecard cutoff selection must be an object")
    _exact_fields(
        payload,
        _SELECTION_TOP_LEVEL_FIELDS,
        "scorecard cutoff selection",
    )
    selection_id = _canonical_text(payload["selection_id"], "selection_id")
    if _SELECTION_ID_RE.fullmatch(selection_id) is None:
        raise ScorecardCandidateError("selection_id has an invalid format")
    selection_hash = _hash(payload["selection_hash"], "selection_hash")
    body = _normalize_selection_body(
        {
            key: payload[key]
            for key in payload
            if key not in {"selection_id", "selection_hash"}
        }
    )
    expected_id = _stable_id("scorecard-cutoff-selection", body)
    if not hmac.compare_digest(selection_id, expected_id):
        raise ScorecardCandidateError(
            "selection_id does not match canonical cutoff selection"
        )
    without_hash = {**body, "selection_id": selection_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(selection_hash, expected_hash):
        raise ScorecardCandidateError(
            "selection_hash does not match canonical cutoff selection"
        )
    return {**without_hash, "selection_hash": selection_hash}


def canonical_scorecard_cutoff_selection_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole canonical JSON encoding of a valid selection."""

    return _canonical_json(validate_scorecard_cutoff_selection(payload))


def scorecard_cutoff_selection_content_hash(
    payload: Mapping[str, Any],
) -> str:
    """Hash canonical persisted bytes of one cutoff selection."""

    return _sha256(canonical_scorecard_cutoff_selection_json(payload))


def scorecard_cutoff_selection_to_verified_candidate_fragment(
    selection_payload: Mapping[str, Any],
    full_band_asset: Mapping[str, Any],
    *,
    selection_artifact_binding: (
        IndependentlyVerifiedScorecardSelectionArtifactBinding
    ),
    source_artifact_binding: IndependentlyVerifiedScorecardArtifactBinding,
) -> dict[str, Any]:
    """Purely replay one authenticated pointer against its authenticated asset."""

    selection = validate_scorecard_cutoff_selection(selection_payload)
    asset = validate_scorecard_band_asset(full_band_asset)
    selection_binding = _verified_artifact_binding(
        selection_artifact_binding,
        name="selection_artifact_binding",
        expected_kind=SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        expected_schema=SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
        expected_origin=SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    )
    source_binding = _verified_artifact_binding(
        source_artifact_binding,
        name="source_artifact_binding",
        expected_kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        expected_schema=SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
        expected_origin=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    )
    _require_selection_binding_matches_selection(
        selection_binding, selection=selection
    )
    _require_asset_binding_matches_asset(source_binding, asset=asset)
    _require_selection_source_matches_asset(
        selection,
        source_binding=source_binding,
        asset=asset,
    )
    if selection_binding["task_id"] != source_binding["task_id"]:
        raise ScorecardCandidateError(
            "selection and source asset must belong to the same task"
        )
    cutoff = _cutoff_by_id(asset, selection["cutoff_id"])
    score_evidence = asset["source_refs"]["score_evidence"]
    score_vector = asset["source_refs"]["score_vector"]
    try:
        virtual_field = model_score_virtual_field(score_vector["artifact_id"])
    except StrategyError as exc:
        raise ScorecardCandidateError(
            "score-vector virtual field could not be derived"
        ) from exc
    condition = {
        "op": "compare",
        "field": virtual_field,
        "operator": ">=",
        "value": cutoff["execution_pd"],
        "missing": "no_match",
    }
    requirement = {
        "type": "model_score_vector.v1",
        "virtual_field": virtual_field,
        "score_product": RAW_BAD_PROBABILITY_SCORE_PRODUCT,
        "score_evidence_artifact_id": score_evidence["artifact_id"],
        "score_evidence_artifact_content_hash": score_evidence[
            "artifact_content_hash"
        ],
        "score_vector_artifact_id": score_vector["artifact_id"],
        "score_vector_artifact_content_hash": score_vector[
            "artifact_content_hash"
        ],
    }
    rule_basis = {
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "cutoff_id": cutoff["cutoff_id"],
        "execution_pd": cutoff["execution_pd"],
    }
    identity = asset["identity"]
    lifecycle = asset["lifecycle"]
    try:
        return build_verified_candidate_fragment(
            artifact={
                "artifact_id": selection_binding["artifact_id"],
                "artifact_kind": selection_binding["kind"],
                "artifact_schema_version": selection_binding[
                    "artifact_schema_version"
                ],
                "artifact_content_hash": selection_binding["content_hash"],
                "origin_tool": selection_binding["origin_tool"],
            },
            asset={
                "schema_version": asset["schema_version"],
                "asset_id": asset["asset_id"],
                "asset_hash": asset["asset_hash"],
                "asset_type": asset["asset_type"],
            },
            fragment_type="strategy_rule",
            rule_id=_stable_id("scorecard-cutoff-rule", rule_basis),
            condition=condition,
            requirements=[requirement],
            effect_id=_stable_id("scorecard-cutoff-effect", rule_basis),
            evidence_id=asset["asset_id"],
            evidence_hash=asset["asset_hash"],
            evidence_identity={
                "dataset_id": identity["dataset_id"],
                "dataset_content_hash": identity["dataset_content_hash"],
                "workspace_revision": identity["workspace_revision"],
                "workspace_generation": identity["workspace_generation"],
                "semantic_mapping_hash": identity["semantic_mapping_hash"],
                "sample_context_hash": identity["sample_context_hash"],
            },
            candidate_stage=lifecycle["candidate_stage"],
            observation_stage=lifecycle["observation_stage"],
            validation_status=lifecycle["validation_status"],
        )
    except CandidateFragmentError as exc:
        raise ScorecardCandidateError(
            "scorecard cutoff failed generic fragment projection"
        ) from exc


def _normalize_selection_body(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_fields(
        payload, _SELECTION_BODY_FIELDS, "scorecard cutoff selection body"
    )
    if payload["schema_version"] != SCORECARD_CUTOFF_SELECTION_SCHEMA_VERSION:
        raise ScorecardCandidateError(
            "selection schema_version must be "
            + SCORECARD_CUTOFF_SELECTION_SCHEMA_VERSION
        )
    if (
        payload["producer_version"]
        != SCORECARD_CUTOFF_SELECTION_PRODUCER_VERSION
    ):
        raise ScorecardCandidateError(
            "selection producer_version must be "
            + SCORECARD_CUTOFF_SELECTION_PRODUCER_VERSION
        )
    cutoff_id = _canonical_text(payload["cutoff_id"], "cutoff_id")
    if _CUTOFF_ID_RE.fullmatch(cutoff_id) is None:
        raise ScorecardCandidateError("cutoff_id has an invalid format")
    return {
        "schema_version": SCORECARD_CUTOFF_SELECTION_SCHEMA_VERSION,
        "producer_version": SCORECARD_CUTOFF_SELECTION_PRODUCER_VERSION,
        "source_asset_ref": _source_asset_pointer(payload["source_asset_ref"]),
        "cutoff_id": cutoff_id,
        "selection_reason": _selection_reason(payload["selection_reason"]),
    }


def _source_asset_pointer(value: object) -> dict[str, Any]:
    obj = _object(value, "source_asset_ref")
    _exact_fields(obj, _SOURCE_ASSET_POINTER_FIELDS, "source_asset_ref")
    if obj["kind"] != SCORECARD_BAND_ASSET_ARTIFACT_KIND:
        raise ScorecardCandidateError("source_asset_ref.kind is invalid")
    if (
        obj["artifact_schema_version"]
        != SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION
    ):
        raise ScorecardCandidateError(
            "source_asset_ref.artifact_schema_version is invalid"
        )
    if obj["origin_tool"] != SCORECARD_BAND_ASSET_ORIGIN_TOOL:
        raise ScorecardCandidateError("source_asset_ref.origin_tool is invalid")
    if obj["asset_schema_version"] != SCORECARD_BAND_ASSET_SCHEMA_VERSION:
        raise ScorecardCandidateError(
            "source_asset_ref.asset_schema_version is invalid"
        )
    if obj["asset_type"] != SCORECARD_BAND_ASSET_TYPE:
        raise ScorecardCandidateError("source_asset_ref.asset_type is invalid")
    asset_id = _canonical_text(obj["asset_id"], "source_asset_ref.asset_id")
    if _ASSET_ID_RE.fullmatch(asset_id) is None:
        raise ScorecardCandidateError(
            "source_asset_ref.asset_id has an invalid format"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"], "source_asset_ref.artifact_id"
        ),
        "task_id": _canonical_text(obj["task_id"], "source_asset_ref.task_id"),
        "kind": SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        "artifact_schema_version": (
            SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION
        ),
        "artifact_content_hash": _hash(
            obj["artifact_content_hash"],
            "source_asset_ref.artifact_content_hash",
        ),
        "origin_tool": SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        "asset_schema_version": SCORECARD_BAND_ASSET_SCHEMA_VERSION,
        "asset_type": SCORECARD_BAND_ASSET_TYPE,
        "asset_id": asset_id,
        "asset_hash": _hash(obj["asset_hash"], "source_asset_ref.asset_hash"),
    }


def _selection_reason(value: object) -> str | None:
    if value is None:
        return None
    reason = _canonical_text(value, "selection_reason")
    if len(reason) > MAX_SELECTION_REASON_LENGTH:
        raise ScorecardCandidateError(
            "selection_reason exceeds character budget"
        )
    return reason


def _verified_artifact_binding(
    value: object,
    *,
    name: str,
    expected_kind: str,
    expected_schema: str,
    expected_origin: str,
) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _VERIFIED_ARTIFACT_BINDING_FIELDS, name)
    if obj["kind"] != expected_kind:
        raise ScorecardCandidateError(f"{name}.kind is invalid")
    if obj["artifact_schema_version"] != expected_schema:
        raise ScorecardCandidateError(
            f"{name}.artifact_schema_version is invalid"
        )
    if obj["origin_tool"] != expected_origin:
        raise ScorecardCandidateError(f"{name}.origin_tool is invalid")
    canonical_bytes = obj["canonical_bytes"]
    if not isinstance(canonical_bytes, bytes):
        raise ScorecardCandidateError(f"{name}.canonical_bytes must be bytes")
    return {
        "artifact_id": _hash(obj["artifact_id"], f"{name}.artifact_id"),
        "task_id": _canonical_text(obj["task_id"], f"{name}.task_id"),
        "kind": expected_kind,
        "artifact_schema_version": expected_schema,
        "content_hash": _hash(
            obj["content_hash"], f"{name}.content_hash"
        ),
        "origin_tool": expected_origin,
        "canonical_bytes": canonical_bytes,
    }


def _require_asset_binding_matches_asset(
    binding: Mapping[str, Any], *, asset: Mapping[str, Any]
) -> None:
    expected = canonical_scorecard_band_asset_json(asset).encode("utf-8")
    if binding["canonical_bytes"] != expected:
        raise ScorecardCandidateError(
            "source artifact canonical bytes do not match band asset"
        )
    observed = hashlib.sha256(binding["canonical_bytes"]).hexdigest()
    if not hmac.compare_digest(observed, binding["content_hash"]):
        raise ScorecardCandidateError(
            "source artifact content hash does not match canonical bytes"
        )
    if binding["task_id"] != asset["identity"]["task_id"]:
        raise ScorecardCandidateError(
            "source artifact task does not match band asset"
        )


def _require_selection_binding_matches_selection(
    binding: Mapping[str, Any], *, selection: Mapping[str, Any]
) -> None:
    expected = canonical_scorecard_cutoff_selection_json(selection).encode(
        "utf-8"
    )
    if binding["canonical_bytes"] != expected:
        raise ScorecardCandidateError(
            "selection artifact canonical bytes do not match selection"
        )
    observed = hashlib.sha256(binding["canonical_bytes"]).hexdigest()
    if not hmac.compare_digest(observed, binding["content_hash"]):
        raise ScorecardCandidateError(
            "selection artifact content hash does not match canonical bytes"
        )


def _require_selection_source_matches_asset(
    selection: Mapping[str, Any],
    *,
    source_binding: Mapping[str, Any],
    asset: Mapping[str, Any],
) -> None:
    expected = {
        "artifact_id": source_binding["artifact_id"],
        "task_id": source_binding["task_id"],
        "kind": source_binding["kind"],
        "artifact_schema_version": source_binding[
            "artifact_schema_version"
        ],
        "artifact_content_hash": source_binding["content_hash"],
        "origin_tool": source_binding["origin_tool"],
        "asset_schema_version": asset["schema_version"],
        "asset_type": asset["asset_type"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
    }
    if selection["source_asset_ref"] != expected:
        raise ScorecardCandidateError(
            "selection source pointer does not match authenticated band asset"
        )


def _cutoff_by_id(
    asset: Mapping[str, Any], cutoff_id: str
) -> dict[str, Any]:
    matches = [
        item for item in asset["cutoffs"] if item["cutoff_id"] == cutoff_id
    ]
    if len(matches) != 1:
        raise ScorecardCandidateError(
            "cutoff_id is not present exactly once in source asset"
        )
    return dict(matches[0])


def _normalize_asset_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(payload, _ASSET_BODY_FIELDS, "scorecard band asset body")
    if payload["schema_version"] != SCORECARD_BAND_ASSET_SCHEMA_VERSION:
        raise ScorecardCandidateError(
            "schema_version must be "
            + SCORECARD_BAND_ASSET_SCHEMA_VERSION
        )
    if payload["producer_version"] != SCORECARD_BAND_ASSET_PRODUCER_VERSION:
        raise ScorecardCandidateError(
            "producer_version must be "
            + SCORECARD_BAND_ASSET_PRODUCER_VERSION
        )
    if payload["asset_type"] != SCORECARD_BAND_ASSET_TYPE:
        raise ScorecardCandidateError(
            "asset_type must be " + SCORECARD_BAND_ASSET_TYPE
        )
    identity = _identity(payload["identity"])
    sample_design_ref = _sample_design_ref(payload["sample_design_ref"])
    source_refs = _source_refs(payload["source_refs"])
    score_contract = _score_contract(payload["score_contract"])
    score_vector = _score_vector_summary(payload["score_vector"])
    bands = _bands(payload["bands"], score_vector=score_vector)
    cutoffs = _cutoffs(
        payload["cutoffs"],
        bands=bands,
        score_vector=score_vector,
        scale=score_contract["scale"],
        vector_artifact_id=source_refs["score_vector"]["artifact_id"],
    )
    performance = _performance(payload["performance"])
    lifecycle = _lifecycle_value(payload["lifecycle"])
    governance = _governance_value(payload["governance"])
    resource_budget = _resource_budget(
        payload["resource_budget"],
        score_vector=score_vector,
        band_count=len(bands),
        scorecard_table_rows=len(score_contract["scorecard_table"]),
    )
    return {
        "schema_version": SCORECARD_BAND_ASSET_SCHEMA_VERSION,
        "producer_version": SCORECARD_BAND_ASSET_PRODUCER_VERSION,
        "asset_type": SCORECARD_BAND_ASSET_TYPE,
        "identity": identity,
        "sample_design_ref": sample_design_ref,
        "source_refs": source_refs,
        "score_contract": score_contract,
        "score_vector": score_vector,
        "bands": bands,
        "performance": performance,
        "cutoffs": cutoffs,
        "lifecycle": lifecycle,
        "governance": governance,
        "resource_budget": resource_budget,
    }


def _identity(value: object) -> dict[str, Any]:
    obj = _object(value, "identity")
    _exact_fields(obj, _IDENTITY_FIELDS, "identity")
    return {
        "task_id": _canonical_text(obj["task_id"], "identity.task_id"),
        "dataset_id": _canonical_text(
            obj["dataset_id"], "identity.dataset_id"
        ),
        "dataset_content_hash": _hash(
            obj["dataset_content_hash"], "identity.dataset_content_hash"
        ),
        "workspace_revision": _non_negative_int(
            obj["workspace_revision"], "identity.workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            obj["workspace_generation"], "identity.workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"], "identity.semantic_mapping_hash"
        ),
        "sample_context_hash": _hash(
            obj["sample_context_hash"], "identity.sample_context_hash"
        ),
    }


def _sample_design_ref(value: object) -> dict[str, str]:
    obj = _object(value, "sample_design_ref")
    _exact_fields(obj, _SAMPLE_DESIGN_REF_FIELDS, "sample_design_ref")
    result = {
        "membership_artifact_id": _hash(
            obj["membership_artifact_id"],
            "sample_design_ref.membership_artifact_id",
        ),
        "expected_membership_artifact_content_hash": _hash(
            obj["expected_membership_artifact_content_hash"],
            "sample_design_ref.expected_membership_artifact_content_hash",
        ),
        "bundle_artifact_id": _hash(
            obj["bundle_artifact_id"],
            "sample_design_ref.bundle_artifact_id",
        ),
        "expected_bundle_artifact_content_hash": _hash(
            obj["expected_bundle_artifact_content_hash"],
            "sample_design_ref.expected_bundle_artifact_content_hash",
        ),
        "expected_bundle_id": _canonical_text(
            obj["expected_bundle_id"],
            "sample_design_ref.expected_bundle_id",
        ),
        "expected_sample_design_id": _canonical_text(
            obj["expected_sample_design_id"],
            "sample_design_ref.expected_sample_design_id",
        ),
        "expected_sample_design_content_hash": _hash(
            obj["expected_sample_design_content_hash"],
            "sample_design_ref.expected_sample_design_content_hash",
        ),
    }
    return result


def _source_refs(value: object) -> dict[str, Any]:
    obj = _object(value, "source_refs")
    _exact_fields(obj, _SOURCE_REFS_FIELDS, "source_refs")
    return {
        "training_evidence": _evidence_ref(
            obj["training_evidence"],
            name="source_refs.training_evidence",
        ),
        "score_evidence": _evidence_ref(
            obj["score_evidence"],
            name="source_refs.score_evidence",
        ),
        "score_vector": _vector_ref(
            obj["score_vector"], name="source_refs.score_vector"
        ),
    }


def _evidence_ref(value: object, *, name: str) -> dict[str, str]:
    obj = _object(value, name)
    _exact_fields(obj, _EVIDENCE_REF_FIELDS, name)
    return {
        "artifact_id": _hash(obj["artifact_id"], f"{name}.artifact_id"),
        "artifact_content_hash": _hash(
            obj["artifact_content_hash"], f"{name}.artifact_content_hash"
        ),
        "evidence_id": _canonical_text(
            obj["evidence_id"], f"{name}.evidence_id"
        ),
        "evidence_content_hash": _hash(
            obj["evidence_content_hash"], f"{name}.evidence_content_hash"
        ),
    }


def _vector_ref(
    value: object, *, name: str = "score_vector_ref"
) -> dict[str, str]:
    obj = _object(value, name)
    _exact_fields(obj, _VECTOR_REF_FIELDS, name)
    return {
        "artifact_id": _hash(obj["artifact_id"], f"{name}.artifact_id"),
        "artifact_content_hash": _hash(
            obj["artifact_content_hash"], f"{name}.artifact_content_hash"
        ),
    }


def _directions(
    *,
    score_product: object,
    score_direction: object,
    points_direction: object,
) -> None:
    if score_product != RAW_BAD_PROBABILITY_SCORE_PRODUCT:
        raise ScorecardCandidateError(
            "score_product must be raw native uncalibrated bad probability"
        )
    if score_direction != MODEL_SCORE_DIRECTION:
        raise ScorecardCandidateError(
            "raw PD score_direction must be higher_is_riskier"
        )
    if points_direction != SCORECARD_POINTS_DIRECTION:
        raise ScorecardCandidateError(
            "scorecard points_direction must be higher_is_better"
        )


def _score_contract(value: object) -> dict[str, Any]:
    obj = _object(value, "score_contract")
    _exact_fields(obj, _SCORE_CONTRACT_FIELDS, "score_contract")
    _directions(
        score_product=obj["score_product"],
        score_direction=obj["score_direction"],
        points_direction=obj["points_direction"],
    )
    scale = _scorecard_scale(obj["scale"])
    table = _scorecard_table(obj["scorecard_table"], scale=scale)
    table_hash = _hash(
        obj["scorecard_table_hash"], "score_contract.scorecard_table_hash"
    )
    expected_hash = _sha256(_canonical_json(table))
    if not hmac.compare_digest(table_hash, expected_hash):
        raise ScorecardCandidateError(
            "scorecard_table_hash does not match canonical scorecard table"
        )
    return {
        "score_product": RAW_BAD_PROBABILITY_SCORE_PRODUCT,
        "score_direction": MODEL_SCORE_DIRECTION,
        "points_direction": SCORECARD_POINTS_DIRECTION,
        "scale": scale,
        "scorecard_table": table,
        "scorecard_table_hash": table_hash,
    }


def _scorecard_scale(value: object) -> dict[str, float | int]:
    obj = _object(value, "scorecard_scale")
    _exact_fields(obj, _SCALE_FIELDS, "scorecard_scale")
    base_score = _integer(obj["base_score"], "scorecard_scale.base_score")
    pdo = _positive_number(obj["pdo"], "scorecard_scale.pdo")
    base_odds = _positive_number(
        obj["base_odds"], "scorecard_scale.base_odds"
    )
    factor = _positive_number(obj["factor"], "scorecard_scale.factor")
    offset = _finite_number(obj["offset"], "scorecard_scale.offset")
    expected_factor = pdo / math.log(2.0)
    expected_offset = float(base_score) - factor * math.log(base_odds)
    if not math.isclose(factor, expected_factor, rel_tol=1e-12, abs_tol=1e-12):
        raise ScorecardCandidateError(
            "scorecard factor must equal pdo / log(2)"
        )
    if not math.isclose(offset, expected_offset, rel_tol=1e-12, abs_tol=1e-12):
        raise ScorecardCandidateError(
            "scorecard offset contradicts base score and base odds"
        )
    return {
        "base_score": base_score,
        "pdo": pdo,
        "base_odds": base_odds,
        "factor": factor,
        "offset": offset,
    }


def _scorecard_table(
    value: object, *, scale: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw = _sequence(value, "scorecard_table")
    if not raw or len(raw) > MAX_SCORECARD_TABLE_ROWS:
        raise ScorecardCandidateError(
            "scorecard_table row count is outside the budget"
        )
    result = [
        _scorecard_row(item, ordinal=index, scale=scale)
        for index, item in enumerate(raw)
    ]
    if result[0]["feature"] != "__base__":
        raise ScorecardCandidateError(
            "scorecard_table first row must be the unique base row"
        )
    if any(item["feature"] == "__base__" for item in result[1:]):
        raise ScorecardCandidateError(
            "scorecard_table must contain exactly one base row"
        )
    if len(result) == 1:
        raise ScorecardCandidateError(
            "scorecard_table must contain at least one feature-bin row"
        )
    identities = [
        (str(item["feature"]), int(item["bin_index"]))
        for item in result
    ]
    if len(set(identities)) != len(identities):
        raise ScorecardCandidateError(
            "scorecard_table contains duplicate feature/bin identity"
        )
    return result


def _scorecard_row(
    value: object,
    *,
    ordinal: int,
    scale: Mapping[str, Any],
) -> dict[str, Any]:
    obj = _object(value, f"scorecard_table[{ordinal}]")
    base = obj.get("feature") == "__base__"
    expected = (
        _BASE_SCORECARD_ROW_FIELDS if base else _FEATURE_SCORECARD_ROW_FIELDS
    )
    _exact_fields(obj, expected, f"scorecard_table[{ordinal}]")
    normalized = {
        key: _json_value(item, f"scorecard_table[{ordinal}].{key}")
        for key, item in obj.items()
    }
    if not isinstance(normalized["feature"], str) or not normalized[
        "feature"
    ]:
        raise ScorecardCandidateError(
            f"scorecard_table[{ordinal}].feature must be non-empty text"
        )
    if not isinstance(normalized["bin_index"], int) or isinstance(
        normalized["bin_index"], bool
    ):
        raise ScorecardCandidateError(
            f"scorecard_table[{ordinal}].bin_index must be an integer"
        )
    _canonical_text(
        normalized["bin_label"], f"scorecard_table[{ordinal}].bin_label"
    )
    _finite_number(
        normalized["points"], f"scorecard_table[{ordinal}].points"
    )
    if base:
        if normalized["bin_index"] != -999:
            raise ScorecardCandidateError(
                "scorecard base row bin_index must be -999"
            )
        for field, expected_value in scale.items():
            actual = normalized[field]
            if isinstance(expected_value, float):
                if not isinstance(actual, int | float) or isinstance(
                    actual, bool
                ) or float(actual) != float(expected_value):
                    raise ScorecardCandidateError(
                        f"scorecard base row {field} changed from scale"
                    )
            elif actual != expected_value:
                raise ScorecardCandidateError(
                    f"scorecard base row {field} changed from scale"
                )
        for field in (
            "lower",
            "upper",
            "count",
            "bad_count",
            "good_count",
            "bad_rate",
            "woe",
            "iv_contribution",
            "coefficient",
            "monotonic_direction",
        ):
            if normalized[field] is not None:
                raise ScorecardCandidateError(
                    f"scorecard base row {field} must be null"
                )
        return normalized
    if normalized["feature"] == "__base__":
        raise ScorecardCandidateError(
            "scorecard feature row cannot use the base feature"
        )
    for field in ("count", "bad_count", "good_count"):
        normalized[field] = _non_negative_int(
            normalized[field], f"scorecard_table[{ordinal}].{field}"
        )
    if (
        normalized["bad_count"] + normalized["good_count"]
        != normalized["count"]
    ):
        raise ScorecardCandidateError(
            f"scorecard_table[{ordinal}] count is not conserved"
        )
    expected_rate = (
        normalized["bad_count"] / normalized["count"]
        if normalized["count"]
        else 0.0
    )
    actual_rate = _ratio(
        normalized["bad_rate"], f"scorecard_table[{ordinal}].bad_rate"
    )
    if actual_rate != expected_rate:
        raise ScorecardCandidateError(
            f"scorecard_table[{ordinal}].bad_rate is inconsistent"
        )
    normalized["bad_rate"] = actual_rate
    for field in ("woe", "iv_contribution", "coefficient"):
        normalized[field] = _finite_number(
            normalized[field], f"scorecard_table[{ordinal}].{field}"
        )
    for field in ("lower", "upper"):
        if normalized[field] is not None:
            normalized[field] = _finite_number(
                normalized[field], f"scorecard_table[{ordinal}].{field}"
            )
    direction = normalized["monotonic_direction"]
    if direction not in {None, "increasing", "decreasing"}:
        raise ScorecardCandidateError(
            f"scorecard_table[{ordinal}].monotonic_direction is invalid"
        )
    return normalized


def _probability_vector(value: object) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ScorecardCandidateError(
            "raw_pd must be a one-dimensional numeric vector"
        ) from exc
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ScorecardCandidateError(
            "raw_pd must be a one-dimensional numeric vector"
        )
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.size <= 0 or result.size > MAX_SCORECARD_CANDIDATE_ROWS:
        raise ScorecardCandidateError("raw_pd row count is outside the budget")
    if not np.all(np.isfinite(result)) or np.any(
        (result < 0.0) | (result > 1.0)
    ):
        raise ScorecardCandidateError(
            "raw_pd must contain finite probabilities in [0, 1]"
        )
    return result


def _boolean_vector(
    value: object, *, name: str, row_count: int
) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ScorecardCandidateError(
            f"{name} must be a one-dimensional boolean vector"
        ) from exc
    if raw.ndim != 1 or raw.dtype.kind != "b" or len(raw) != row_count:
        raise ScorecardCandidateError(
            f"{name} must be a row-aligned boolean vector"
        )
    return np.ascontiguousarray(raw, dtype=np.bool_)


def _label_vector(value: object, *, row_count: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ScorecardCandidateError(
            "labels must be a one-dimensional 0/1/NaN numeric vector"
        ) from exc
    if (
        raw.ndim != 1
        or raw.dtype.kind not in "iuf"
        or len(raw) != row_count
    ):
        raise ScorecardCandidateError(
            "labels must be a row-aligned 0/1/NaN numeric vector"
        )
    result = np.ascontiguousarray(raw, dtype=np.float64)
    finite = result[np.isfinite(result)]
    if np.any((finite != 0.0) & (finite != 1.0)) or np.any(
        np.isinf(result)
    ):
        raise ScorecardCandidateError("labels may contain only 0, 1, or NaN")
    return result


def _input_bins(value: object) -> list[dict[str, Any]]:
    raw = _sequence(value, "score_bins")
    if not MIN_SCORECARD_BANDS <= len(raw) <= MAX_SCORECARD_BANDS:
        raise ScorecardCandidateError(
            "score_bins must contain between 2 and 20 bands"
        )
    result = [
        _input_bin(item, ordinal=index) for index, item in enumerate(raw)
    ]
    _require_contiguous_bins(result)
    return result


def _input_bin(value: object, *, ordinal: int) -> dict[str, Any]:
    obj = _object(value, f"score_bins[{ordinal}]")
    _exact_fields(obj, _INPUT_BIN_FIELDS, f"score_bins[{ordinal}]")
    stored_ordinal = _non_negative_int(
        obj["ordinal"], f"score_bins[{ordinal}].ordinal"
    )
    if stored_ordinal != ordinal:
        raise ScorecardCandidateError(
            "score_bins ordinals must be consecutive source ordinals"
        )
    return {
        "ordinal": stored_ordinal,
        "bin_id": _canonical_text(
            obj["bin_id"], f"score_bins[{ordinal}].bin_id"
        ),
        "lower_bound": _optional_finite_number(
            obj["lower_bound"], f"score_bins[{ordinal}].lower_bound"
        ),
        "upper_bound": _optional_finite_number(
            obj["upper_bound"], f"score_bins[{ordinal}].upper_bound"
        ),
        "lower_inclusive": _boolean(
            obj["lower_inclusive"],
            f"score_bins[{ordinal}].lower_inclusive",
        ),
        "upper_inclusive": _boolean(
            obj["upper_inclusive"],
            f"score_bins[{ordinal}].upper_inclusive",
        ),
    }


def _require_contiguous_bins(bins: Sequence[Mapping[str, Any]]) -> None:
    if bins[0]["lower_bound"] is not None:
        raise ScorecardCandidateError("score bins must cover the lower tail")
    if bins[-1]["upper_bound"] is not None:
        raise ScorecardCandidateError("score bins must cover the upper tail")
    seen_ids: set[str] = set()
    for index, item in enumerate(bins):
        if item["bin_id"] in seen_ids:
            raise ScorecardCandidateError("score bin ids must be unique")
        seen_ids.add(str(item["bin_id"]))
        lower = item["lower_bound"]
        upper = item["upper_bound"]
        if lower is None:
            if item["lower_inclusive"]:
                raise ScorecardCandidateError(
                    "unbounded lower score edge cannot be inclusive"
                )
        elif not item["lower_inclusive"]:
            raise ScorecardCandidateError(
                "score bins must use canonical [lower, upper) bounds"
            )
        if item["upper_inclusive"]:
            raise ScorecardCandidateError(
                "score bins must use canonical [lower, upper) bounds"
            )
        if lower is not None and upper is not None and lower >= upper:
            raise ScorecardCandidateError(
                "score bin bounds must be strictly increasing"
            )
        if index:
            previous = bins[index - 1]
            if previous["upper_bound"] != lower:
                raise ScorecardCandidateError(
                    "score bins must be strictly contiguous"
                )
        if upper is not None and not 0.0 < float(upper) < 1.0:
            raise ScorecardCandidateError(
                "raw-PD internal score-bin edges must be inside (0, 1)"
            )


def _measured_band(
    definition: Mapping[str, Any],
    *,
    ordinal: int,
    probabilities: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    development_count: int,
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    selected_labels = labels[mask]
    labeled = selected_labels[np.isfinite(selected_labels)]
    labeled_count = int(len(labeled))
    bad_count = int(np.sum(labeled))
    return {
        **definition,
        "ordinal": ordinal,
        "count": count,
        "share": float(count / development_count),
        "labeled_count": labeled_count,
        "bad_count": bad_count,
        "bad_rate": (
            None if labeled_count == 0 else float(bad_count / labeled_count)
        ),
        "average_pd": (
            None if count == 0 else float(np.mean(probabilities[mask]))
        ),
    }


def _cutoff_side(
    mask: np.ndarray, *, labels: np.ndarray
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    selected_labels = labels[mask]
    labeled = selected_labels[np.isfinite(selected_labels)]
    labeled_count = int(len(labeled))
    bad_count = int(np.sum(labeled))
    return {
        "count": count,
        "labeled_count": labeled_count,
        "bad_count": bad_count,
        "bad_rate": (
            None if labeled_count == 0 else float(bad_count / labeled_count)
        ),
    }


def _score_vector_summary(value: object) -> dict[str, Any]:
    obj = _object(value, "score_vector")
    _exact_fields(obj, _SCORE_VECTOR_FIELDS, "score_vector")
    result = {
        "row_count": _positive_int(
            obj["row_count"], "score_vector.row_count"
        ),
        "development_count": _positive_int(
            obj["development_count"], "score_vector.development_count"
        ),
        "labeled_count": _positive_int(
            obj["labeled_count"], "score_vector.labeled_count"
        ),
        "bad_count": _positive_int(
            obj["bad_count"], "score_vector.bad_count"
        ),
        "raw_pd_content_hash": _hash(
            obj["raw_pd_content_hash"],
            "score_vector.raw_pd_content_hash",
        ),
    }
    if result["row_count"] > MAX_SCORECARD_CANDIDATE_ROWS:
        raise ScorecardCandidateError("score_vector exceeds row budget")
    if not (
        result["bad_count"]
        <= result["labeled_count"]
        <= result["development_count"]
        <= result["row_count"]
    ):
        raise ScorecardCandidateError(
            "score_vector development label counts are not conserved"
        )
    if result["bad_count"] in {0, result["labeled_count"]}:
        raise ScorecardCandidateError(
            "score_vector development labels must contain both classes"
        )
    return result


def _bands(
    value: object, *, score_vector: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw = _sequence(value, "bands")
    if not MIN_SCORECARD_BANDS <= len(raw) <= MAX_SCORECARD_BANDS:
        raise ScorecardCandidateError(
            "bands must contain between 2 and 20 rows"
        )
    result = [
        _band(item, ordinal=index, development_count=score_vector["development_count"])
        for index, item in enumerate(raw)
    ]
    _require_contiguous_bins(result)
    if sum(item["count"] for item in result) != score_vector[
        "development_count"
    ]:
        raise ScorecardCandidateError("band counts do not conserve development")
    if sum(item["labeled_count"] for item in result) != score_vector[
        "labeled_count"
    ]:
        raise ScorecardCandidateError(
            "band labeled counts do not conserve development labels"
        )
    if sum(item["bad_count"] for item in result) != score_vector["bad_count"]:
        raise ScorecardCandidateError(
            "band bad counts do not conserve development bads"
        )
    if not math.isclose(
        sum(item["share"] for item in result),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ScorecardCandidateError("band shares must sum to one")
    return result


def _band(
    value: object, *, ordinal: int, development_count: int
) -> dict[str, Any]:
    obj = _object(value, f"bands[{ordinal}]")
    _exact_fields(obj, _BAND_FIELDS, f"bands[{ordinal}]")
    base = _input_bin(
        {key: obj[key] for key in _INPUT_BIN_FIELDS},
        ordinal=ordinal,
    )
    count = _positive_int(obj["count"], f"bands[{ordinal}].count")
    labeled_count = _non_negative_int(
        obj["labeled_count"], f"bands[{ordinal}].labeled_count"
    )
    bad_count = _non_negative_int(
        obj["bad_count"], f"bands[{ordinal}].bad_count"
    )
    if not bad_count <= labeled_count <= count:
        raise ScorecardCandidateError(
            f"bands[{ordinal}] label counts are not conserved"
        )
    share = _ratio(obj["share"], f"bands[{ordinal}].share")
    if share != count / development_count:
        raise ScorecardCandidateError(
            f"bands[{ordinal}].share is inconsistent"
        )
    bad_rate = _nullable_ratio(
        obj["bad_rate"], f"bands[{ordinal}].bad_rate"
    )
    expected_bad_rate = (
        None if labeled_count == 0 else bad_count / labeled_count
    )
    if bad_rate != expected_bad_rate:
        raise ScorecardCandidateError(
            f"bands[{ordinal}].bad_rate is inconsistent"
        )
    average_pd = _nullable_ratio(
        obj["average_pd"], f"bands[{ordinal}].average_pd"
    )
    if average_pd is None:
        raise ScorecardCandidateError(
            f"bands[{ordinal}].average_pd must be present"
        )
    lower = base["lower_bound"]
    upper = base["upper_bound"]
    if lower is not None and average_pd < lower:
        raise ScorecardCandidateError(
            f"bands[{ordinal}].average_pd is below its boundary"
        )
    if upper is not None and average_pd >= upper:
        raise ScorecardCandidateError(
            f"bands[{ordinal}].average_pd is above its boundary"
        )
    return {
        **base,
        "count": count,
        "share": share,
        "labeled_count": labeled_count,
        "bad_count": bad_count,
        "bad_rate": bad_rate,
        "average_pd": average_pd,
    }


def _performance(value: object) -> dict[str, float]:
    obj = _object(value, "performance")
    _exact_fields(obj, _PERFORMANCE_FIELDS, "performance")
    return {
        "auc": _ratio(obj["auc"], "performance.auc"),
        "ks": _ratio(obj["ks"], "performance.ks"),
    }


def _cutoffs(
    value: object,
    *,
    bands: Sequence[Mapping[str, Any]],
    score_vector: Mapping[str, Any],
    scale: Mapping[str, Any],
    vector_artifact_id: str,
) -> list[dict[str, Any]]:
    raw = _sequence(value, "cutoffs")
    if len(raw) != len(bands) - 1:
        raise ScorecardCandidateError(
            "cutoffs must contain every internal band boundary"
        )
    result = [
        _cutoff(
            item,
            ordinal=index,
            bands=bands,
            score_vector=score_vector,
            scale=scale,
            vector_artifact_id=vector_artifact_id,
        )
        for index, item in enumerate(raw)
    ]
    if len({item["cutoff_id"] for item in result}) != len(result):
        raise ScorecardCandidateError("cutoff ids must be unique")
    return result


def _cutoff(
    value: object,
    *,
    ordinal: int,
    bands: Sequence[Mapping[str, Any]],
    score_vector: Mapping[str, Any],
    scale: Mapping[str, Any],
    vector_artifact_id: str,
) -> dict[str, Any]:
    obj = _object(value, f"cutoffs[{ordinal}]")
    _exact_fields(obj, _CUTOFF_FIELDS, f"cutoffs[{ordinal}]")
    stored_ordinal = _non_negative_int(
        obj["ordinal"], f"cutoffs[{ordinal}].ordinal"
    )
    if stored_ordinal != ordinal:
        raise ScorecardCandidateError("cutoff ordinals must be consecutive")
    execution_pd = _ratio(
        obj["execution_pd"], f"cutoffs[{ordinal}].execution_pd"
    )
    expected_pd = bands[ordinal]["upper_bound"]
    if expected_pd is None or execution_pd != expected_pd:
        raise ScorecardCandidateError(
            f"cutoffs[{ordinal}] does not match its internal boundary"
        )
    expected_id = _stable_id(
        "scorecard-cutoff",
        {
            "score_vector_artifact_id": vector_artifact_id,
            "execution_pd": execution_pd,
            "ordinal": ordinal,
        },
    )
    cutoff_id = _canonical_text(
        obj["cutoff_id"], f"cutoffs[{ordinal}].cutoff_id"
    )
    if (
        _CUTOFF_ID_RE.fullmatch(cutoff_id) is None
        or not hmac.compare_digest(cutoff_id, expected_id)
    ):
        raise ScorecardCandidateError(
            f"cutoffs[{ordinal}].cutoff_id is inconsistent"
        )
    display_points = _finite_number(
        obj["display_points"], f"cutoffs[{ordinal}].display_points"
    )
    try:
        expected_points = float(
            scorecard_points_from_raw_pd(
                [execution_pd],
                factor=scale["factor"],
                offset=scale["offset"],
            )[0]
        )
    except ModelingError as exc:
        raise ScorecardCandidateError(str(exc)) from exc
    if display_points != expected_points:
        raise ScorecardCandidateError(
            f"cutoffs[{ordinal}].display_points changed from scale"
        )
    lower_expected = _aggregate_band_side(bands[: ordinal + 1])
    higher_expected = _aggregate_band_side(bands[ordinal + 1 :])
    lower = _cutoff_side_value(
        obj["lower_risk"], name=f"cutoffs[{ordinal}].lower_risk"
    )
    higher = _cutoff_side_value(
        obj["higher_risk"], name=f"cutoffs[{ordinal}].higher_risk"
    )
    if lower != lower_expected or higher != higher_expected:
        raise ScorecardCandidateError(
            f"cutoffs[{ordinal}] side metrics do not conserve bands"
        )
    if (
        lower["count"] + higher["count"]
        != score_vector["development_count"]
        or lower["labeled_count"] + higher["labeled_count"]
        != score_vector["labeled_count"]
        or lower["bad_count"] + higher["bad_count"]
        != score_vector["bad_count"]
    ):
        raise ScorecardCandidateError(
            f"cutoffs[{ordinal}] does not conserve development"
        )
    if obj["mask_equivalence"] is not True:
        raise ScorecardCandidateError(
            f"cutoffs[{ordinal}] must prove mask equivalence"
        )
    return {
        "ordinal": ordinal,
        "cutoff_id": cutoff_id,
        "execution_pd": execution_pd,
        "display_points": display_points,
        "lower_risk": lower,
        "higher_risk": higher,
        "mask_equivalence": True,
    }


def _aggregate_band_side(
    bands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = sum(int(item["count"]) for item in bands)
    labeled_count = sum(int(item["labeled_count"]) for item in bands)
    bad_count = sum(int(item["bad_count"]) for item in bands)
    return {
        "count": count,
        "labeled_count": labeled_count,
        "bad_count": bad_count,
        "bad_rate": (
            None if labeled_count == 0 else bad_count / labeled_count
        ),
    }


def _cutoff_side_value(value: object, *, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _CUTOFF_SIDE_FIELDS, name)
    count = _non_negative_int(obj["count"], f"{name}.count")
    labeled_count = _non_negative_int(
        obj["labeled_count"], f"{name}.labeled_count"
    )
    bad_count = _non_negative_int(obj["bad_count"], f"{name}.bad_count")
    if not bad_count <= labeled_count <= count:
        raise ScorecardCandidateError(f"{name} counts are not conserved")
    bad_rate = _nullable_ratio(obj["bad_rate"], f"{name}.bad_rate")
    expected = None if labeled_count == 0 else bad_count / labeled_count
    if bad_rate != expected:
        raise ScorecardCandidateError(f"{name}.bad_rate is inconsistent")
    return {
        "count": count,
        "labeled_count": labeled_count,
        "bad_count": bad_count,
        "bad_rate": bad_rate,
    }


def _lifecycle() -> dict[str, str]:
    return {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }


def _lifecycle_value(value: object) -> dict[str, str]:
    obj = _object(value, "lifecycle")
    _exact_fields(obj, _LIFECYCLE_FIELDS, "lifecycle")
    if obj != _lifecycle():
        raise ScorecardCandidateError(
            "scorecard candidate lifecycle must remain "
            "development/backtested/unvalidated"
        )
    return _lifecycle()


def _governance() -> dict[str, bool]:
    return {
        "best_cutoff_recommended": False,
        "selected": False,
        "adopted": False,
        "deployed": False,
    }


def _governance_value(value: object) -> dict[str, bool]:
    obj = _object(value, "governance")
    _exact_fields(obj, _GOVERNANCE_FIELDS, "governance")
    if obj != _governance():
        raise ScorecardCandidateError(
            "full scorecard band asset cannot recommend or select a cutoff"
        )
    return _governance()


def _resource_budget(
    value: object,
    *,
    score_vector: Mapping[str, Any],
    band_count: int,
    scorecard_table_rows: int,
) -> dict[str, int]:
    obj = _object(value, "resource_budget")
    _exact_fields(obj, _RESOURCE_BUDGET_FIELDS, "resource_budget")
    expected = {
        "max_rows": MAX_SCORECARD_CANDIDATE_ROWS,
        "rows_processed": score_vector["row_count"],
        "max_bands": MAX_SCORECARD_BANDS,
        "bands_used": band_count,
        "max_scorecard_table_rows": MAX_SCORECARD_TABLE_ROWS,
        "scorecard_table_rows": scorecard_table_rows,
    }
    normalized = {
        key: _non_negative_int(obj[key], f"resource_budget.{key}")
        for key in expected
    }
    if normalized != expected:
        raise ScorecardCandidateError(
            "resource_budget drifted from scorecard candidate"
        )
    return expected


def _float64_vector_hash(values: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ScorecardCandidateError(
            f"{name} must be an object with string keys"
        )
    return value


def _sequence(value: object, name: str) -> list[Any]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(
        value, Sequence
    ):
        raise ScorecardCandidateError(f"{name} must be an array")
    return list(value)


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ScorecardCandidateError(
            f"{name} fields are invalid ({'; '.join(details)})"
        )


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ScorecardCandidateError(f"{name} must be canonical text")
    normalized = unicodedata.normalize("NFC", value)
    if (
        not normalized
        or value != normalized
        or value != value.strip()
        or "\x00" in value
    ):
        raise ScorecardCandidateError(f"{name} must be canonical text")
    return value


def _hash(value: object, name: str) -> str:
    text = _canonical_text(value, name)
    if _HASH_RE.fullmatch(text) is None:
        raise ScorecardCandidateError(f"{name} must be a lowercase SHA-256")
    return text


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ScorecardCandidateError(f"{name} must be boolean")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ScorecardCandidateError(f"{name} must be an integer")
    return int(value)


def _non_negative_int(value: object, name: str) -> int:
    normalized = _integer(value, name)
    if normalized < 0:
        raise ScorecardCandidateError(f"{name} must be non-negative")
    return normalized


def _positive_int(value: object, name: str) -> int:
    normalized = _integer(value, name)
    if normalized <= 0:
        raise ScorecardCandidateError(f"{name} must be positive")
    return normalized


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ScorecardCandidateError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ScorecardCandidateError(f"{name} must be a finite number")
    return normalized


def _positive_number(value: object, name: str) -> float:
    normalized = _finite_number(value, name)
    if normalized <= 0.0:
        raise ScorecardCandidateError(f"{name} must be positive")
    return normalized


def _ratio(value: object, name: str) -> float:
    normalized = _finite_number(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise ScorecardCandidateError(f"{name} must be in [0, 1]")
    return normalized


def _nullable_ratio(value: object, name: str) -> float | None:
    return None if value is None else _ratio(value, name)


def _optional_finite_number(value: object, name: str) -> float | None:
    return None if value is None else _finite_number(value, name)


def _json_value(value: object, name: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _finite_number(value, name)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ScorecardCandidateError(f"{name} keys must be strings")
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
    raise ScorecardCandidateError(f"{name} must contain finite JSON")


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ScorecardCandidateError(
            "scorecard candidate must contain finite canonical JSON"
        ) from exc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "IndependentlyVerifiedScorecardArtifactBinding",
    "IndependentlyVerifiedScorecardSelectionArtifactBinding",
    "MAX_SCORECARD_BANDS",
    "MAX_SCORECARD_CANDIDATE_JSON_BYTES",
    "MAX_SCORECARD_CANDIDATE_ROWS",
    "MAX_SCORECARD_TABLE_ROWS",
    "MIN_SCORECARD_BANDS",
    "RAW_BAD_PROBABILITY_SCORE_PRODUCT",
    "SCORECARD_BAND_ASSET_ARTIFACT_KIND",
    "SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION",
    "SCORECARD_BAND_ASSET_ORIGIN_TOOL",
    "SCORECARD_BAND_ASSET_PRODUCER_VERSION",
    "SCORECARD_BAND_ASSET_SCHEMA_VERSION",
    "SCORECARD_BAND_ASSET_TYPE",
    "SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND",
    "SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION",
    "SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL",
    "SCORECARD_CUTOFF_SELECTION_PRODUCER_VERSION",
    "SCORECARD_CUTOFF_SELECTION_SCHEMA_VERSION",
    "SCORECARD_POINTS_DIRECTION",
    "ScorecardCandidateError",
    "build_scorecard_band_asset",
    "build_scorecard_cutoff_selection",
    "canonical_scorecard_band_asset_json",
    "canonical_scorecard_cutoff_selection_json",
    "scorecard_band_asset_content_hash",
    "scorecard_cutoff_selection_content_hash",
    "scorecard_cutoff_selection_to_verified_candidate_fragment",
    "validate_scorecard_band_asset",
    "validate_scorecard_cutoff_selection",
]
