"""Governed persistence Tools for scorecard bands and cutoff selections.

The build Tool consumes only authenticated TrainingEvidence, raw model-score
evidence, and the inseparable StrategySampleDesign V2 artifact pair.  It
publishes a complete immutable band asset without selecting a cutoff.  The
selection Tool is deliberately separate and persists only a pointer to one
cutoff in that full asset.  Neither Tool mutates Strategy Pool.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.db import ModelingRepository
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence import RAW_SCORE_PRODUCT
from marvis.packs.modeling.score_evidence_tools import (
    ModelScoreEvidenceArtifactBinding,
    load_historical_model_score_evidence_artifacts,
    load_model_score_evidence_artifacts,
    require_historical_model_score_evidence_artifact_binding_on_connection,
    require_model_score_evidence_artifact_binding_on_connection,
)
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_v2_tools import (
    load_any_strategy_sample_design_v2_artifacts,
    load_historical_any_strategy_sample_design_v2_artifacts,
    require_any_strategy_sample_design_v2_artifact_binding_on_connection,
    require_historical_any_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.packs.strategy.scorecard_candidate import (
    MAX_SCORECARD_BANDS,
    MAX_SCORECARD_CANDIDATE_JSON_BYTES,
    MIN_SCORECARD_BANDS,
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    build_scorecard_band_asset,
    build_scorecard_cutoff_selection,
    canonical_scorecard_band_asset_json,
    canonical_scorecard_cutoff_selection_json,
    validate_scorecard_band_asset,
    validate_scorecard_cutoff_selection,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


BUILD_SCORECARD_BAND_ASSET_TOOL_SCHEMA_VERSION = (
    "strategy.build-scorecard-band-asset-tool.v1"
)
MATERIALIZE_SCORECARD_CUTOFF_SELECTION_TOOL_SCHEMA_VERSION = (
    "strategy.materialize-scorecard-cutoff-selection-tool.v1"
)

_BUILD_REQUIRED_INPUT_FIELDS = frozenset(
    {"score_evidence_ref", "sample_design_ref"}
)
_BUILD_OPTIONAL_INPUT_FIELDS = frozenset(
    {"banding", "raw_pd_band_edges"}
)
_SCORE_EVIDENCE_REF_FIELDS = frozenset(
    {
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "score_vector_artifact_id",
        "expected_score_vector_artifact_content_hash",
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
_SELECTION_REQUIRED_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_source_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "cutoff_id",
    }
)
_SELECTION_OPTIONAL_INPUT_FIELDS = frozenset({"reason"})
_TASK_ARTIFACT_RECORD_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "kind",
        "path",
        "content_hash",
        "origin_tool",
        "provenance",
        "created_at",
    }
)
_BAND_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "asset_schema_version",
        "producer_version",
        "task_id",
        "asset_type",
        "asset_id",
        "asset_hash",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
        "sample_design_ref",
        "training_evidence_ref",
        "score_evidence_ref",
        "score_vector_ref",
        "score_product",
        "scorecard_table_hash",
        "raw_pd_internal_edges",
        "band_count",
        "cutoff_count",
    }
)
_SELECTION_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "selection_schema_version",
        "producer_version",
        "task_id",
        "selection_id",
        "selection_hash",
        "cutoff_id",
        "selection_reason",
        "source_artifact_id",
        "source_artifact_content_hash",
        "source_asset_id",
        "source_asset_hash",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_OUTPUT_DIRECTORY = "strategy_scorecard_candidates"
_MAX_ARTIFACT_BYTES = MAX_SCORECARD_CANDIDATE_JSON_BYTES
_BOUNDARY_ERRORS = (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class ScorecardBandAssetArtifactBinding:
    """Authenticated canonical full-band artifact."""

    task_id: str
    artifact_id: str
    path: Path
    content_hash: str
    provenance: dict[str, Any]
    canonical_bytes: bytes
    asset: dict[str, Any]
    score_evidence: ModelScoreEvidenceArtifactBinding
    sample_design: Any

    def to_domain_binding(self) -> dict[str, Any]:
        """Return the exact pure-domain artifact binding."""

        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "kind": SCORECARD_BAND_ASSET_ARTIFACT_KIND,
            "artifact_schema_version": (
                SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION
            ),
            "content_hash": self.content_hash,
            "origin_tool": SCORECARD_BAND_ASSET_ORIGIN_TOOL,
            "canonical_bytes": self.canonical_bytes,
        }


@dataclass(frozen=True)
class ScorecardCutoffSelectionArtifactBinding:
    """Authenticated canonical pointer-only cutoff selection."""

    task_id: str
    artifact_id: str
    path: Path
    content_hash: str
    provenance: dict[str, Any]
    canonical_bytes: bytes
    selection: dict[str, Any]
    source_asset_binding: ScorecardBandAssetArtifactBinding

    def to_domain_binding(self) -> dict[str, Any]:
        """Return the exact pure-domain selection binding."""

        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "kind": SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
            "artifact_schema_version": (
                SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION
            ),
            "content_hash": self.content_hash,
            "origin_tool": SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
            "canonical_bytes": self.canonical_bytes,
        }


def run_build_scorecard_band_asset(inputs: object, ctx, runtime) -> dict[str, Any]:
    """Build and atomically publish a complete scorecard-band asset."""

    request = _validate_build_inputs(inputs)
    task_id = _canonical_text(ctx.task_id, "task_id")
    try:
        score = load_model_score_evidence_artifacts(
            _modeling_runtime(runtime),
            task_id=task_id,
            **request["score_evidence_ref"],
        )
        sample = load_any_strategy_sample_design_v2_artifacts(
            runtime,
            task_id=task_id,
            **request["sample_design_ref"],
        )
        _require_same_sample_binding(score, sample=sample)
        metadata = _require_scorecard_contract(score)
        frame, labels, target_col = _read_governed_labels(
            runtime,
            sample=sample,
        )
        if len(frame) != score.vector.row_count:
            raise StrategyError(
                "scorecard score vector row count changed from governed dataset"
            )
        mask = np.asarray(
            sample.membership["masks"]["risk/development"],
            dtype=np.bool_,
        )
        if mask.ndim != 1 or len(mask) != len(frame):
            raise StrategyError(
                "scorecard risk/development membership shape changed"
            )
        sample_ref = request["sample_design_ref"]
        identity = _scorecard_identity(
            task_id=task_id,
            sample=sample,
            sample_ref=sample_ref,
            target_col=target_col,
            labels=labels,
            risk_development_mask=mask,
        )
        score_bins, banding = _resolve_score_bins(
            request["banding"],
            raw_pd=score.vector.scores,
            risk_development_mask=mask,
        )
        asset = build_scorecard_band_asset(
            identity=identity,
            sample_design_ref=sample_ref,
            training_evidence_ref=_training_evidence_ref(score),
            score_evidence_ref=_score_evidence_ref(score),
            score_vector_ref=_score_vector_ref(score),
            score_product=score.envelope["score_product"],
            score_direction=score.envelope["scoring_contract"][
                "score_direction"
            ],
            points_direction=metadata["points_direction"],
            scorecard_scale=_scorecard_scale(metadata),
            scorecard_table=metadata["scorecard_table"],
            raw_pd=score.vector.scores,
            risk_development_mask=mask,
            labels=labels,
            score_bins=score_bins,
        )
        canonical = canonical_scorecard_band_asset_json(asset).encode("utf-8")
        binding = _persist_band_asset(
            runtime,
            task_id=task_id,
            score=score,
            sample=sample,
            asset=asset,
            canonical=canonical,
        )
        return _band_tool_output(binding, banding=banding)
    except StrategyError:
        raise
    except ModelingError as exc:
        raise StrategyError(str(exc)) from exc
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def run_materialize_scorecard_cutoff_selection(
    inputs: object,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Persist an explicit pointer to one measured cutoff; never mutate Pool."""

    request = _validate_selection_inputs(inputs)
    task_id = _canonical_text(ctx.task_id, "task_id")
    try:
        source = load_scorecard_band_asset_artifact(
            runtime,
            task_id=task_id,
            artifact_id=request["source_artifact_id"],
            expected_artifact_content_hash=request[
                "expected_source_artifact_content_hash"
            ],
            expected_asset_id=request["expected_asset_id"],
            expected_asset_hash=request["expected_asset_hash"],
        )
        selection = build_scorecard_cutoff_selection(
            source.asset,
            source_artifact_binding=source.to_domain_binding(),
            cutoff_id=request["cutoff_id"],
            selection_reason=request["reason"],
        )
        canonical = canonical_scorecard_cutoff_selection_json(selection).encode(
            "utf-8"
        )
        binding = _persist_cutoff_selection(
            runtime,
            task_id=task_id,
            source=source,
            selection=selection,
            canonical=canonical,
        )
        return _selection_tool_output(binding)
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def load_scorecard_band_asset_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> ScorecardBandAssetArtifactBinding:
    """Strictly reload one canonical full-band artifact and registry binding."""

    return _load_scorecard_band_asset_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_artifact_content_hash=expected_artifact_content_hash,
        expected_asset_id=expected_asset_id,
        expected_asset_hash=expected_asset_hash,
        require_current_sources=True,
    )


def load_historical_scorecard_band_asset_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> ScorecardBandAssetArtifactBinding:
    """Reload an immutable band asset without requiring source sample head."""

    return _load_scorecard_band_asset_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_artifact_content_hash=expected_artifact_content_hash,
        expected_asset_id=expected_asset_id,
        expected_asset_hash=expected_asset_hash,
        require_current_sources=False,
    )


def _load_scorecard_band_asset_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    require_current_sources: bool,
) -> ScorecardBandAssetArtifactBinding:
    normalized_task = _safe_component(task_id, "task_id")
    normalized_artifact_id = _hash(artifact_id, "artifact_id")
    normalized_content_hash = _hash(
        expected_artifact_content_hash,
        "expected_artifact_content_hash",
    )
    normalized_asset_id = _safe_component(
        expected_asset_id,
        "expected_asset_id",
    )
    normalized_asset_hash = _hash(expected_asset_hash, "expected_asset_hash")
    expected_path = _artifact_path(
        runtime.settings.tasks_dir,
        task_id=normalized_task,
        document_id=normalized_asset_id,
    )
    record = runtime.task_artifacts.get_for_task(
        normalized_task,
        normalized_artifact_id,
    )
    if record is None:
        raise StrategyError("scorecard band artifact is not registered")
    _require_record(
        record,
        task_id=normalized_task,
        artifact_id=normalized_artifact_id,
        kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        origin_tool=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        path=expected_path,
        content_hash=normalized_content_hash,
    )
    raw = _read_exact_artifact(
        expected_path,
        root=Path(runtime.settings.tasks_dir).absolute(),
        expected_hash=normalized_content_hash,
    )
    asset = _parse_band_asset(raw)
    canonical = canonical_scorecard_band_asset_json(asset).encode("utf-8")
    if raw != canonical:
        raise StrategyError("scorecard band artifact is not canonical JSON")
    if (
        asset["asset_id"] != normalized_asset_id
        or not hmac.compare_digest(
            asset["asset_hash"],
            normalized_asset_hash,
        )
    ):
        raise StrategyError("scorecard band asset identity changed")
    provenance = _strict_provenance(record.get("provenance"))
    expected_provenance = _band_provenance(asset)
    if provenance != expected_provenance:
        raise StrategyError("scorecard band artifact provenance changed")
    score_evidence, sample_design = _load_and_rebuild_band_sources(
        runtime,
        task_id=normalized_task,
        asset=asset,
        require_current_sources=require_current_sources,
    )
    return ScorecardBandAssetArtifactBinding(
        task_id=normalized_task,
        artifact_id=normalized_artifact_id,
        path=expected_path,
        content_hash=normalized_content_hash,
        provenance=provenance,
        canonical_bytes=canonical,
        asset=asset,
        score_evidence=score_evidence,
        sample_design=sample_design,
    )


def load_scorecard_cutoff_selection_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_selection_id: str,
    expected_selection_hash: str,
) -> ScorecardCutoffSelectionArtifactBinding:
    """Strictly reload one canonical pointer-only cutoff selection."""

    return _load_scorecard_cutoff_selection_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_artifact_content_hash=expected_artifact_content_hash,
        expected_selection_id=expected_selection_id,
        expected_selection_hash=expected_selection_hash,
        require_current_sources=True,
    )


def load_historical_scorecard_cutoff_selection_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_selection_id: str,
    expected_selection_hash: str,
) -> ScorecardCutoffSelectionArtifactBinding:
    """Reload an immutable cutoff without requiring source sample head."""

    return _load_scorecard_cutoff_selection_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_artifact_content_hash=expected_artifact_content_hash,
        expected_selection_id=expected_selection_id,
        expected_selection_hash=expected_selection_hash,
        require_current_sources=False,
    )


def _load_scorecard_cutoff_selection_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_selection_id: str,
    expected_selection_hash: str,
    require_current_sources: bool,
) -> ScorecardCutoffSelectionArtifactBinding:
    normalized_task = _safe_component(task_id, "task_id")
    normalized_artifact_id = _hash(artifact_id, "artifact_id")
    normalized_content_hash = _hash(
        expected_artifact_content_hash,
        "expected_artifact_content_hash",
    )
    normalized_selection_id = _safe_component(
        expected_selection_id,
        "expected_selection_id",
    )
    normalized_selection_hash = _hash(
        expected_selection_hash,
        "expected_selection_hash",
    )
    expected_path = _artifact_path(
        runtime.settings.tasks_dir,
        task_id=normalized_task,
        document_id=normalized_selection_id,
    )
    record = runtime.task_artifacts.get_for_task(
        normalized_task,
        normalized_artifact_id,
    )
    if record is None:
        raise StrategyError("scorecard cutoff selection is not registered")
    _require_record(
        record,
        task_id=normalized_task,
        artifact_id=normalized_artifact_id,
        kind=SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        origin_tool=SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        path=expected_path,
        content_hash=normalized_content_hash,
    )
    raw = _read_exact_artifact(
        expected_path,
        root=Path(runtime.settings.tasks_dir).absolute(),
        expected_hash=normalized_content_hash,
    )
    selection = _parse_cutoff_selection(raw)
    canonical = canonical_scorecard_cutoff_selection_json(selection).encode(
        "utf-8"
    )
    if raw != canonical:
        raise StrategyError(
            "scorecard cutoff selection is not canonical JSON"
        )
    if (
        selection["selection_id"] != normalized_selection_id
        or not hmac.compare_digest(
            selection["selection_hash"],
            normalized_selection_hash,
        )
    ):
        raise StrategyError("scorecard cutoff selection identity changed")
    provenance = _strict_provenance(record.get("provenance"))
    expected_provenance = _selection_provenance(selection)
    if provenance != expected_provenance:
        raise StrategyError("scorecard cutoff selection provenance changed")
    source_ref = selection["source_asset_ref"]
    source_loader = (
        load_scorecard_band_asset_artifact
        if require_current_sources
        else load_historical_scorecard_band_asset_artifact
    )
    source_asset = source_loader(
        runtime,
        task_id=normalized_task,
        artifact_id=source_ref["artifact_id"],
        expected_artifact_content_hash=source_ref[
            "artifact_content_hash"
        ],
        expected_asset_id=source_ref["asset_id"],
        expected_asset_hash=source_ref["asset_hash"],
    )
    replayed = build_scorecard_cutoff_selection(
        source_asset.asset,
        source_artifact_binding=source_asset.to_domain_binding(),
        cutoff_id=selection["cutoff_id"],
        selection_reason=selection["selection_reason"],
    )
    if replayed != selection:
        raise StrategyError(
            "scorecard cutoff selection no longer replays against source asset"
        )
    return ScorecardCutoffSelectionArtifactBinding(
        task_id=normalized_task,
        artifact_id=normalized_artifact_id,
        path=expected_path,
        content_hash=normalized_content_hash,
        provenance=provenance,
        canonical_bytes=canonical,
        selection=selection,
        source_asset_binding=source_asset,
    )


def require_scorecard_band_asset_artifact_binding_on_connection(
    conn,
    binding: ScorecardBandAssetArtifactBinding,
) -> None:
    """CAS one loaded full-band artifact while a writer owns a transaction."""

    _require_scorecard_band_asset_artifact_binding_on_connection(
        conn,
        binding,
        require_current_sources=True,
    )


def require_historical_scorecard_band_asset_artifact_binding_on_connection(
    conn,
    binding: ScorecardBandAssetArtifactBinding,
) -> None:
    """CAS an immutable band asset without requiring source sample head."""

    _require_scorecard_band_asset_artifact_binding_on_connection(
        conn,
        binding,
        require_current_sources=False,
    )


def _require_scorecard_band_asset_artifact_binding_on_connection(
    conn,
    binding: ScorecardBandAssetArtifactBinding,
    *,
    require_current_sources: bool,
) -> None:
    if not isinstance(binding, ScorecardBandAssetArtifactBinding):
        raise StrategyError("scorecard band artifact binding is invalid")
    _require_binding_on_connection(
        conn,
        binding=binding,
        kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        origin_tool=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        expected_provenance=_band_provenance(binding.asset),
        parser=_parse_band_asset,
        canonicalizer=canonical_scorecard_band_asset_json,
    )
    try:
        if require_current_sources:
            require_model_score_evidence_artifact_binding_on_connection(
                conn,
                binding.score_evidence,
            )
        else:
            require_historical_model_score_evidence_artifact_binding_on_connection(
                conn,
                binding.score_evidence,
            )
    except ModelingError as exc:
        raise StrategyError(str(exc)) from exc
    if require_current_sources:
        require_any_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            binding.sample_design,
        )
    else:
        require_historical_any_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            binding.sample_design,
        )
    _require_same_sample_binding(
        binding.score_evidence,
        sample=binding.sample_design,
    )


def require_scorecard_cutoff_selection_artifact_binding_on_connection(
    conn,
    binding: ScorecardCutoffSelectionArtifactBinding,
) -> None:
    """CAS one loaded cutoff selection while a writer owns a transaction."""

    _require_scorecard_cutoff_selection_artifact_binding_on_connection(
        conn,
        binding,
        require_current_sources=True,
    )


def require_historical_scorecard_cutoff_selection_artifact_binding_on_connection(
    conn,
    binding: ScorecardCutoffSelectionArtifactBinding,
) -> None:
    """CAS an immutable cutoff without requiring source sample head."""

    _require_scorecard_cutoff_selection_artifact_binding_on_connection(
        conn,
        binding,
        require_current_sources=False,
    )


def _require_scorecard_cutoff_selection_artifact_binding_on_connection(
    conn,
    binding: ScorecardCutoffSelectionArtifactBinding,
    *,
    require_current_sources: bool,
) -> None:
    if not isinstance(binding, ScorecardCutoffSelectionArtifactBinding):
        raise StrategyError(
            "scorecard cutoff selection artifact binding is invalid"
        )
    _require_binding_on_connection(
        conn,
        binding=binding,
        kind=SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        origin_tool=SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        expected_provenance=_selection_provenance(binding.selection),
        parser=_parse_cutoff_selection,
        canonicalizer=canonical_scorecard_cutoff_selection_json,
    )
    if require_current_sources:
        require_scorecard_band_asset_artifact_binding_on_connection(
            conn,
            binding.source_asset_binding,
        )
    else:
        require_historical_scorecard_band_asset_artifact_binding_on_connection(
            conn,
            binding.source_asset_binding,
        )


def _validate_build_inputs(value: object) -> dict[str, Any]:
    obj = _object(value, "build_scorecard_band_asset inputs")
    fields = set(obj)
    missing = sorted(_BUILD_REQUIRED_INPUT_FIELDS - fields)
    unsupported = sorted(
        fields - _BUILD_REQUIRED_INPUT_FIELDS - _BUILD_OPTIONAL_INPUT_FIELDS
    )
    if missing or unsupported:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unsupported:
            details.append("unsupported: " + ", ".join(unsupported))
        raise StrategyError(
            "build_scorecard_band_asset input fields are invalid ("
            + "; ".join(details)
            + ")"
        )
    if "banding" in obj and "raw_pd_band_edges" in obj:
        raise StrategyError(
            "banding and raw_pd_band_edges are mutually exclusive"
        )
    score_ref = _object(obj["score_evidence_ref"], "score_evidence_ref")
    _exact_fields(
        score_ref,
        _SCORE_EVIDENCE_REF_FIELDS,
        "score_evidence_ref",
    )
    sample_ref = _object(obj["sample_design_ref"], "sample_design_ref")
    _exact_fields(
        sample_ref,
        _SAMPLE_DESIGN_REF_FIELDS,
        "sample_design_ref",
    )
    return {
        "score_evidence_ref": {
            field: _hash(score_ref[field], f"score_evidence_ref.{field}")
            for field in _SCORE_EVIDENCE_REF_FIELDS
        },
        "sample_design_ref": {
            "membership_artifact_id": _hash(
                sample_ref["membership_artifact_id"],
                "sample_design_ref.membership_artifact_id",
            ),
            "expected_membership_artifact_content_hash": _hash(
                sample_ref["expected_membership_artifact_content_hash"],
                "sample_design_ref.expected_membership_artifact_content_hash",
            ),
            "bundle_artifact_id": _hash(
                sample_ref["bundle_artifact_id"],
                "sample_design_ref.bundle_artifact_id",
            ),
            "expected_bundle_artifact_content_hash": _hash(
                sample_ref["expected_bundle_artifact_content_hash"],
                "sample_design_ref.expected_bundle_artifact_content_hash",
            ),
            "expected_bundle_id": _canonical_text(
                sample_ref["expected_bundle_id"],
                "sample_design_ref.expected_bundle_id",
            ),
            "expected_sample_design_id": _canonical_text(
                sample_ref["expected_sample_design_id"],
                "sample_design_ref.expected_sample_design_id",
            ),
            "expected_sample_design_content_hash": _hash(
                sample_ref["expected_sample_design_content_hash"],
                "sample_design_ref.expected_sample_design_content_hash",
            ),
        },
        "banding": _banding_request(
            obj.get("banding"),
            raw_pd_band_edges=obj.get("raw_pd_band_edges"),
        ),
    }


def _validate_selection_inputs(value: object) -> dict[str, Any]:
    obj = _object(value, "materialize_scorecard_cutoff_selection inputs")
    fields = set(obj)
    missing = sorted(_SELECTION_REQUIRED_INPUT_FIELDS - fields)
    unsupported = sorted(
        fields
        - _SELECTION_REQUIRED_INPUT_FIELDS
        - _SELECTION_OPTIONAL_INPUT_FIELDS
    )
    if missing or unsupported:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unsupported:
            details.append("unsupported: " + ", ".join(unsupported))
        raise StrategyError(
            "materialize_scorecard_cutoff_selection input fields are invalid ("
            + "; ".join(details)
            + ")"
        )
    reason = obj.get("reason")
    if reason is not None:
        reason = _canonical_text(reason, "reason")
    return {
        "source_artifact_id": _hash(
            obj["source_artifact_id"],
            "source_artifact_id",
        ),
        "expected_source_artifact_content_hash": _hash(
            obj["expected_source_artifact_content_hash"],
            "expected_source_artifact_content_hash",
        ),
        "expected_asset_id": _safe_component(
            obj["expected_asset_id"],
            "expected_asset_id",
        ),
        "expected_asset_hash": _hash(
            obj["expected_asset_hash"],
            "expected_asset_hash",
        ),
        "cutoff_id": _canonical_text(obj["cutoff_id"], "cutoff_id"),
        "reason": reason,
    }


def _require_same_sample_binding(
    score: ModelScoreEvidenceArtifactBinding,
    *,
    sample: Any,
) -> None:
    training_sample = score.training.sample
    comparisons = {
        "task_id": (training_sample.task_id, sample.task_id),
        "membership_artifact_id": (
            training_sample.membership_artifact_id,
            sample.membership_artifact_id,
        ),
        "membership_artifact_content_hash": (
            training_sample.membership_artifact_content_hash,
            sample.membership_artifact_content_hash,
        ),
        "bundle_artifact_id": (
            training_sample.bundle_artifact_id,
            sample.bundle_artifact_id,
        ),
        "bundle_artifact_content_hash": (
            training_sample.bundle_artifact_content_hash,
            sample.bundle_artifact_content_hash,
        ),
        "sample_design_id": (
            training_sample.bundle["sample_design"]["sample_design_id"],
            sample.bundle["sample_design"]["sample_design_id"],
        ),
        "sample_design_content_hash": (
            training_sample.bundle["sample_design"]["content_hash"],
            sample.bundle["sample_design"]["content_hash"],
        ),
    }
    changed = [
        field
        for field, (left, right) in comparisons.items()
        if left != right
    ]
    if changed:
        raise StrategyError(
            "score evidence and sample-design V2 binding differ: "
            + ", ".join(changed)
        )


def _load_and_rebuild_band_sources(
    runtime,
    *,
    task_id: str,
    asset: Mapping[str, Any],
    require_current_sources: bool,
) -> tuple[
    ModelScoreEvidenceArtifactBinding,
    Any,
]:
    refs = asset["source_refs"]
    score_ref = refs["score_evidence"]
    vector_ref = refs["score_vector"]
    try:
        score_loader = (
            load_model_score_evidence_artifacts
            if require_current_sources
            else load_historical_model_score_evidence_artifacts
        )
        score = score_loader(
            _modeling_runtime(runtime),
            task_id=task_id,
            evidence_artifact_id=score_ref["artifact_id"],
            expected_evidence_artifact_content_hash=score_ref[
                "artifact_content_hash"
            ],
            score_vector_artifact_id=vector_ref["artifact_id"],
            expected_score_vector_artifact_content_hash=vector_ref[
                "artifact_content_hash"
            ],
        )
    except ModelingError as exc:
        raise StrategyError(str(exc)) from exc
    sample_loader = (
        load_any_strategy_sample_design_v2_artifacts
        if require_current_sources
        else load_historical_any_strategy_sample_design_v2_artifacts
    )
    sample = sample_loader(
        runtime,
        task_id=task_id,
        **asset["sample_design_ref"],
    )
    _require_same_sample_binding(score, sample=sample)
    metadata = _require_scorecard_contract(score)
    _frame, labels, target_col = _read_governed_labels(
        runtime,
        sample=sample,
    )
    mask = np.asarray(
        sample.membership["masks"]["risk/development"],
        dtype=np.bool_,
    )
    identity = _scorecard_identity(
        task_id=task_id,
        sample=sample,
        sample_ref=asset["sample_design_ref"],
        target_col=target_col,
        labels=labels,
        risk_development_mask=mask,
    )
    if identity != asset["identity"]:
        raise StrategyError(
            "scorecard band asset sample identity no longer matches sources"
        )
    score_bins = [
        {
            field: band[field]
            for field in (
                "ordinal",
                "bin_id",
                "lower_bound",
                "upper_bound",
                "lower_inclusive",
                "upper_inclusive",
            )
        }
        for band in asset["bands"]
    ]
    rebuilt = build_scorecard_band_asset(
        identity=identity,
        sample_design_ref=asset["sample_design_ref"],
        training_evidence_ref=_training_evidence_ref(score),
        score_evidence_ref=_score_evidence_ref(score),
        score_vector_ref=_score_vector_ref(score),
        score_product=score.envelope["score_product"],
        score_direction=score.envelope["scoring_contract"][
            "score_direction"
        ],
        points_direction=metadata["points_direction"],
        scorecard_scale=_scorecard_scale(metadata),
        scorecard_table=metadata["scorecard_table"],
        raw_pd=score.vector.scores,
        risk_development_mask=mask,
        labels=labels,
        score_bins=score_bins,
    )
    if rebuilt != asset:
        raise StrategyError(
            "scorecard band asset no longer rebuilds from authenticated sources"
        )
    return score, sample


def _require_scorecard_contract(
    score: ModelScoreEvidenceArtifactBinding,
) -> dict[str, Any]:
    training = score.training
    if (
        training.experiment.recipe_id != "scorecard"
        or training.model_artifact.algorithm != "scorecard"
        or training.evidence["experiment"]["recipe_id"] != "scorecard"
        or training.evidence["model_artifact"]["algorithm"] != "scorecard"
    ):
        raise StrategyError(
            "scorecard band asset requires scorecard TrainingEvidence"
        )
    metadata = training.evidence["model_artifact"]["scoring_metadata"]
    if (
        score.envelope.get("score_product") != RAW_SCORE_PRODUCT
        or metadata.get("score_product") != RAW_SCORE_PRODUCT
        or score.envelope["scoring_contract"].get("score_direction")
        != "higher_is_riskier"
        or metadata.get("score_direction") != "higher_is_riskier"
        or metadata.get("points_direction") != "higher_is_better"
        or metadata.get("calibration_status") != "not_applied"
    ):
        raise StrategyError(
            "scorecard evidence must be raw uncalibrated bad probability "
            "with higher-is-better points"
        )
    if not isinstance(metadata.get("scorecard_table"), list) or not metadata[
        "scorecard_table"
    ]:
        raise StrategyError("scorecard TrainingEvidence lacks a scorecard table")
    return metadata


def _read_governed_labels(
    runtime,
    *,
    sample: Any,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    design = sample.bundle["sample_design"]
    target = design["target_selector"]
    if target.get("status") != "resolved":
        raise StrategyError("scorecard sample target selector is unresolved")
    target_col = _canonical_text(target.get("column"), "target column")
    frame = _read_authenticated_dataset_snapshot(
        sample.source_binding.dataset_path,
        root=runtime.settings.datasets_dir,
        expected_content_hash=sample.source_binding.dataset_content_hash,
        columns=[target_col],
    )
    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("scorecard governed dataset did not return a frame")
    if len(frame) != sample.source_binding.row_count:
        raise StrategyError("scorecard governed dataset row count changed")
    raw = frame[target_col]
    missing = raw.isna().to_numpy(dtype=np.bool_)
    booleans = raw.map(
        lambda item: isinstance(item, (bool, np.bool_))
    ).to_numpy(dtype=np.bool_)
    numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
    good_value = float(target["good_value"])
    bad_value = float(target["bad_value"])
    invalid = (
        (~missing & booleans)
        | (~missing & ~np.isfinite(numeric))
        | (
            ~missing
            & (numeric != good_value)
            & (numeric != bad_value)
        )
    )
    if np.any(invalid):
        raise StrategyError(
            "scorecard governed target contains out-of-contract values"
        )
    if np.any(missing) and target["drop_missing"] is not True:
        raise StrategyError(
            "scorecard governed target contains missing labels without "
            "drop_missing authorization"
        )
    labels = np.full(len(frame), np.nan, dtype=np.float64)
    labels[~missing & (numeric == good_value)] = 0.0
    labels[~missing & (numeric == bad_value)] = 1.0
    labels.setflags(write=False)
    return frame, labels, target_col


def _read_authenticated_dataset_snapshot(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
    columns: list[str],
) -> pd.DataFrame:
    """Read labels only from one retained, hash-authenticated private copy."""

    _require_dataset_path(path, root=root)
    source_fd = -1
    snapshot = None
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise StrategyError(
                "scorecard governed dataset must be a regular file"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_fd = os.open(path, flags)
        opened = os.fstat(source_fd)
        after_open = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(after_open.st_mode)
            or _dataset_file_identity(before)
            != _dataset_file_identity(opened)
            or _dataset_file_identity(opened)
            != _dataset_file_identity(after_open)
            or _dataset_stable_file_stat(before)
            != _dataset_stable_file_stat(opened)
            or _dataset_stable_file_stat(opened)
            != _dataset_stable_file_stat(after_open)
        ):
            raise StrategyError(
                "scorecard governed dataset changed while opening"
            )

        snapshot = tempfile.TemporaryFile(mode="w+b", dir=root)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            snapshot.write(chunk)
        snapshot.flush()
        if (
            _dataset_stable_file_stat(os.fstat(source_fd))
            != _dataset_stable_file_stat(opened)
            or copied != int(opened.st_size)
            or not hmac.compare_digest(
                digest.hexdigest(),
                expected_content_hash,
            )
        ):
            raise StrategyError(
                "scorecard governed dataset bytes changed before replay"
            )

        snapshot_stat = os.fstat(snapshot.fileno())
        if int(snapshot_stat.st_size) != copied:
            raise StrategyError(
                "scorecard governed dataset private snapshot is incomplete"
            )
        snapshot.seek(0)
        frame = pd.read_parquet(snapshot, columns=columns)
        current = os.lstat(path)
        if (
            _dataset_stable_file_stat(os.fstat(snapshot.fileno()))
            != _dataset_stable_file_stat(snapshot_stat)
            or _dataset_stable_file_stat(os.fstat(source_fd))
            != _dataset_stable_file_stat(opened)
            or stat.S_ISLNK(current.st_mode)
            or _dataset_stable_file_stat(current)
            != _dataset_stable_file_stat(opened)
        ):
            raise StrategyError(
                "scorecard governed dataset changed during replay"
            )
        return frame
    except StrategyError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StrategyError(
            "scorecard governed dataset could not be read"
        ) from exc
    finally:
        if snapshot is not None:
            snapshot.close()
        if source_fd >= 0:
            os.close(source_fd)


def _require_dataset_path(path: Path, *, root: Path) -> None:
    absolute = path.absolute()
    declared_root = root.absolute()
    try:
        absolute.relative_to(declared_root)
    except ValueError as exc:
        raise StrategyError(
            "scorecard governed dataset escaped dataset storage"
        ) from exc
    current = absolute
    while True:
        if current.is_symlink():
            raise StrategyError(
                "scorecard governed dataset must not use symlinks"
            )
        if current == declared_root:
            break
        if current == current.parent:
            raise StrategyError(
                "scorecard governed dataset escaped dataset storage"
            )
        current = current.parent


def _dataset_file_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
    )


def _dataset_stable_file_stat(
    value: os.stat_result,
) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _scorecard_identity(
    *,
    task_id: str,
    sample: Any,
    sample_ref: Mapping[str, Any],
    target_col: str,
    labels: np.ndarray,
    risk_development_mask: np.ndarray,
) -> dict[str, Any]:
    source = sample.source_binding
    development_labels = labels[risk_development_mask]
    nan_labels_dropped = int(
        np.count_nonzero(~np.isfinite(development_labels))
    )
    target = sample.bundle["sample_design"]["target_selector"]
    context = {
        "schema_version": "strategy.scorecard-sample-context.v1",
        "identity": {
            "task_id": task_id,
            "dataset_id": source.dataset_id,
            "dataset_content_hash": source.dataset_content_hash,
            "workspace_revision": source.workspace_revision,
            "workspace_generation": source.workspace_generation,
            "semantic_mapping_hash": source.semantic_mapping_hash,
        },
        "population": {
            "role": "risk",
            "partition": "development",
            "row_count": int(np.count_nonzero(risk_development_mask)),
            "labeled_row_count": int(
                np.count_nonzero(np.isfinite(development_labels))
            ),
        },
        "target": {
            "column": target_col,
            "good_value": target["good_value"],
            "bad_value": target["bad_value"],
            "drop_missing": target["drop_missing"],
            "nan_labels_dropped": nan_labels_dropped,
        },
        "sample_design_ref": dict(sample_ref),
    }
    return {
        "task_id": task_id,
        "dataset_id": source.dataset_id,
        "dataset_content_hash": source.dataset_content_hash,
        "workspace_revision": source.workspace_revision,
        "workspace_generation": source.workspace_generation,
        "semantic_mapping_hash": source.semantic_mapping_hash,
        "sample_context_hash": _sha256(
            _canonical_json(context).encode("utf-8")
        ),
    }


def _training_evidence_ref(
    score: ModelScoreEvidenceArtifactBinding,
) -> dict[str, str]:
    evidence = score.training.evidence
    return {
        "artifact_id": str(score.training.evidence_record["id"]),
        "artifact_content_hash": str(
            score.training.evidence_record["content_hash"]
        ),
        "evidence_id": str(evidence["evidence_id"]),
        "evidence_content_hash": str(evidence["content_hash"]),
    }


def _score_evidence_ref(
    score: ModelScoreEvidenceArtifactBinding,
) -> dict[str, str]:
    return {
        "artifact_id": str(score.evidence_record["id"]),
        "artifact_content_hash": str(score.evidence_record["content_hash"]),
        "evidence_id": str(score.envelope["evidence_id"]),
        "evidence_content_hash": str(score.envelope["content_hash"]),
    }


def _score_vector_ref(
    score: ModelScoreEvidenceArtifactBinding,
) -> dict[str, str]:
    return {
        "artifact_id": str(score.vector_record["id"]),
        "artifact_content_hash": str(score.vector_record["content_hash"]),
    }


def _scorecard_scale(metadata: Mapping[str, Any]) -> dict[str, Any]:
    params = metadata.get("params")
    if not isinstance(params, Mapping):
        raise StrategyError("scorecard scoring params are unavailable")
    required = ("base_score", "pdo", "base_odds", "factor", "offset")
    missing = [field for field in required if field not in params]
    if missing:
        raise StrategyError(
            "scorecard scale is incomplete: " + ", ".join(missing)
        )
    return {field: params[field] for field in required}


def _banding_request(
    value: object,
    *,
    raw_pd_band_edges: object,
) -> dict[str, Any]:
    if raw_pd_band_edges is not None:
        return {
            "method": "manual_raw_pd",
            "edges": _raw_pd_band_edges(raw_pd_band_edges),
        }
    if value is None:
        return {"method": "equal_frequency", "bin_count": 10}
    obj = _object(value, "banding")
    _exact_fields(
        obj,
        frozenset({"method", "bin_count"}),
        "banding",
    )
    if obj["method"] != "equal_frequency":
        raise StrategyError(
            "banding.method must be equal_frequency"
        )
    bin_count = obj["bin_count"]
    if (
        isinstance(bin_count, bool)
        or not isinstance(bin_count, int)
        or not MIN_SCORECARD_BANDS <= bin_count <= MAX_SCORECARD_BANDS
    ):
        raise StrategyError(
            "banding.bin_count must be an integer between 2 and 20"
        )
    return {"method": "equal_frequency", "bin_count": bin_count}


def _resolve_score_bins(
    request: Mapping[str, Any],
    *,
    raw_pd: Sequence[float] | np.ndarray,
    risk_development_mask: Sequence[bool] | np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request["method"] == "manual_raw_pd":
        edges = list(request["edges"])
        bins = _score_bins(edges)
        return bins, {
            "method": "manual_raw_pd",
            "requested_bin_count": len(edges) - 1,
            "effective_bin_count": len(bins),
            "internal_edges": edges[1:-1],
        }
    probabilities = np.asarray(raw_pd, dtype=np.float64)
    development = np.asarray(risk_development_mask, dtype=np.bool_)
    if (
        probabilities.ndim != 1
        or development.ndim != 1
        or len(probabilities) != len(development)
    ):
        raise StrategyError(
            "equal-frequency banding requires aligned score and membership vectors"
        )
    selected = probabilities[development]
    if (
        selected.size < MIN_SCORECARD_BANDS
        or np.any(~np.isfinite(selected))
        or np.any((selected < 0.0) | (selected > 1.0))
    ):
        raise StrategyError(
            "equal-frequency banding requires finite risk/development raw PD"
        )
    requested = int(request["bin_count"])
    ordered = np.sort(selected, kind="stable")
    internal_candidates: list[float] = []
    for cut in range(1, requested):
        index = int(math.ceil(cut * len(ordered) / requested))
        if index <= 0 or index >= len(ordered):
            continue
        lower = float(ordered[index - 1])
        upper = float(ordered[index])
        if lower >= upper:
            continue
        # Midpoints between adjacent, distinct order statistics preserve the
        # requested equal-frequency rank cut while guaranteeing non-empty
        # canonical [lower, upper) bands even for repeated model scores.
        boundary = lower + (upper - lower) / 2.0
        if 0.0 < boundary < 1.0:
            internal_candidates.append(boundary)
    internal = sorted(set(internal_candidates))
    edges = [0.0, *internal, 1.0]
    if len(edges) - 1 < MIN_SCORECARD_BANDS:
        raise StrategyError(
            "equal-frequency score banding produced fewer than 2 non-empty "
            "bands after de-duplicating raw-PD quantiles"
        )
    bins = _score_bins(edges)
    assigned = np.searchsorted(
        np.asarray(internal, dtype=np.float64),
        selected,
        side="right",
    )
    if any(
        int(np.count_nonzero(assigned == ordinal)) == 0
        for ordinal in range(len(bins))
    ):
        raise StrategyError(
            "equal-frequency score banding produced an empty development band"
        )
    return bins, {
        "method": "equal_frequency",
        "requested_bin_count": requested,
        "effective_bin_count": len(bins),
        "internal_edges": internal,
    }


def _raw_pd_band_edges(value: object) -> list[float]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(
        value,
        Sequence,
    ):
        raise StrategyError("raw_pd_band_edges must be a list")
    edges = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise StrategyError(
                f"raw_pd_band_edges[{index}] must be a finite number"
            )
        normalized = float(item)
        if not math.isfinite(normalized):
            raise StrategyError(
                f"raw_pd_band_edges[{index}] must be a finite number"
            )
        edges.append(normalized)
    band_count = len(edges) - 1
    if not MIN_SCORECARD_BANDS <= band_count <= MAX_SCORECARD_BANDS:
        raise StrategyError(
            "raw_pd_band_edges must define between 2 and 20 bands"
        )
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise StrategyError(
            "raw_pd_band_edges must start at 0 and end at 1"
        )
    if any(
        left >= right
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ):
        raise StrategyError(
            "raw_pd_band_edges must be strictly increasing"
        )
    return edges


def _score_bins(edges: Sequence[float]) -> list[dict[str, Any]]:
    band_count = len(edges) - 1
    return [
        {
            "ordinal": index,
            "bin_id": f"score-band-{index:02d}",
            "lower_bound": None if index == 0 else float(edges[index]),
            "upper_bound": (
                None
                if index == band_count - 1
                else float(edges[index + 1])
            ),
            "lower_inclusive": index != 0,
            "upper_inclusive": False,
        }
        for index in range(band_count)
    ]


def _persist_band_asset(
    runtime,
    *,
    task_id: str,
    score: ModelScoreEvidenceArtifactBinding,
    sample: Any,
    asset: Mapping[str, Any],
    canonical: bytes,
) -> ScorecardBandAssetArtifactBinding:
    normalized = validate_scorecard_band_asset(asset)
    final_path = _artifact_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        document_id=normalized["asset_id"],
    )
    provenance = _band_provenance(normalized)

    def require_sources(conn) -> None:
        try:
            require_model_score_evidence_artifact_binding_on_connection(
                conn,
                score,
            )
        except ModelingError as exc:
            raise StrategyError(str(exc)) from exc
        require_any_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            sample,
        )
        _require_same_sample_binding(score, sample=sample)

    record = _persist_document(
        runtime,
        task_id=task_id,
        kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        origin_tool=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        final_path=final_path,
        canonical=canonical,
        provenance=provenance,
        require_sources=require_sources,
    )
    return ScorecardBandAssetArtifactBinding(
        task_id=task_id,
        artifact_id=str(record["id"]),
        path=final_path,
        content_hash=str(record["content_hash"]),
        provenance=provenance,
        canonical_bytes=canonical,
        asset=normalized,
        score_evidence=score,
        sample_design=sample,
    )


def _persist_cutoff_selection(
    runtime,
    *,
    task_id: str,
    source: ScorecardBandAssetArtifactBinding,
    selection: Mapping[str, Any],
    canonical: bytes,
) -> ScorecardCutoffSelectionArtifactBinding:
    normalized = validate_scorecard_cutoff_selection(selection)
    final_path = _artifact_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        document_id=normalized["selection_id"],
    )
    provenance = _selection_provenance(normalized)

    def require_sources(conn) -> None:
        require_scorecard_band_asset_artifact_binding_on_connection(
            conn,
            source,
        )

    record = _persist_document(
        runtime,
        task_id=task_id,
        kind=SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        origin_tool=SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        final_path=final_path,
        canonical=canonical,
        provenance=provenance,
        require_sources=require_sources,
    )
    return ScorecardCutoffSelectionArtifactBinding(
        task_id=task_id,
        artifact_id=str(record["id"]),
        path=final_path,
        content_hash=str(record["content_hash"]),
        provenance=provenance,
        canonical_bytes=canonical,
        selection=normalized,
        source_asset_binding=source,
    )


def _persist_document(
    runtime,
    *,
    task_id: str,
    kind: str,
    origin_tool: str,
    final_path: Path,
    canonical: bytes,
    provenance: Mapping[str, Any],
    require_sources,
) -> dict[str, Any]:
    if len(canonical) > _MAX_ARTIFACT_BYTES:
        raise StrategyError("scorecard candidate artifact exceeds byte budget")
    content_hash = _sha256(canonical)
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    if final_path.parent != out_dir:
        raise StrategyError("scorecard candidate output path drifted")
    uow = ArtifactUnitOfWork()
    try:
        staged = uow.stage_file(out_dir, final_path.name)
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError(
            "scorecard candidate artifact could not be staged"
        ) from exc
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                require_sources(conn)
                _require_existing_consistent(
                    conn,
                    task_id=task_id,
                    kind=kind,
                    origin_tool=origin_tool,
                    final_path=final_path,
                    content_hash=content_hash,
                    provenance=provenance,
                    canonical=canonical,
                )
                uow.promote_all()
                _read_exact_artifact(
                    final_path,
                    root=Path(runtime.settings.tasks_dir).absolute(),
                    expected_hash=content_hash,
                    expected_bytes=canonical,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=kind,
                    path=str(final_path),
                    content_hash=content_hash,
                    origin_tool=origin_tool,
                    provenance=provenance,
                )
                _require_record(
                    record,
                    task_id=task_id,
                    artifact_id=str(record["id"]),
                    kind=kind,
                    origin_tool=origin_tool,
                    path=final_path,
                    content_hash=content_hash,
                    expected_provenance=provenance,
                )
                require_sources(conn)
                conn.commit()
                db_committed = True
            except Exception:
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
        uow.commit()
    except Exception:
        if not db_committed and not rollback_attempted_under_lock:
            uow.rollback()
        raise
    return dict(record)


def _require_existing_consistent(
    conn,
    *,
    task_id: str,
    kind: str,
    origin_tool: str,
    final_path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
    canonical: bytes,
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, kind, str(final_path)),
    ).fetchone()
    if row is None:
        if final_path.exists() or final_path.is_symlink():
            _read_exact_artifact(
                final_path,
                root=final_path.parents[2],
                expected_hash=content_hash,
                expected_bytes=canonical,
            )
        return
    record = _record_from_sql_row(row)
    _require_record(
        record,
        task_id=task_id,
        artifact_id=str(record["id"]),
        kind=kind,
        origin_tool=origin_tool,
        path=final_path,
        content_hash=content_hash,
        expected_provenance=provenance,
    )
    _read_exact_artifact(
        final_path,
        root=final_path.parents[2],
        expected_hash=content_hash,
        expected_bytes=canonical,
    )


def _require_binding_on_connection(
    conn,
    *,
    binding,
    kind: str,
    origin_tool: str,
    expected_provenance: Mapping[str, Any],
    parser,
    canonicalizer,
) -> None:
    if not getattr(conn, "in_transaction", False):
        raise StrategyError(
            "scorecard artifact revalidation requires an active transaction"
        )
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (binding.task_id, kind, str(binding.path)),
    ).fetchone()
    if row is None:
        raise StrategyError(
            "scorecard source artifact disappeared before write"
        )
    record = _record_from_sql_row(row)
    _require_record(
        record,
        task_id=binding.task_id,
        artifact_id=binding.artifact_id,
        kind=kind,
        origin_tool=origin_tool,
        path=binding.path,
        content_hash=binding.content_hash,
        expected_provenance=expected_provenance,
    )
    raw = _read_exact_artifact(
        binding.path,
        root=binding.path.parents[2],
        expected_hash=binding.content_hash,
        expected_bytes=binding.canonical_bytes,
    )
    parsed = parser(raw)
    if canonicalizer(parsed).encode("utf-8") != binding.canonical_bytes:
        raise StrategyError(
            "scorecard source artifact canonical content changed before write"
        )


def _band_provenance(asset: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_scorecard_band_asset(asset)
    identity = normalized["identity"]
    refs = normalized["source_refs"]
    provenance = {
        "schema_version": SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
        "asset_schema_version": normalized["schema_version"],
        "producer_version": normalized["producer_version"],
        "task_id": identity["task_id"],
        "asset_type": normalized["asset_type"],
        "asset_id": normalized["asset_id"],
        "asset_hash": normalized["asset_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "sample_context_hash": identity["sample_context_hash"],
        "sample_design_ref": normalized["sample_design_ref"],
        "training_evidence_ref": refs["training_evidence"],
        "score_evidence_ref": refs["score_evidence"],
        "score_vector_ref": refs["score_vector"],
        "score_product": normalized["score_contract"]["score_product"],
        "scorecard_table_hash": normalized["score_contract"][
            "scorecard_table_hash"
        ],
        "raw_pd_internal_edges": [
            band["upper_bound"]
            for band in normalized["bands"][:-1]
        ],
        "band_count": len(normalized["bands"]),
        "cutoff_count": len(normalized["cutoffs"]),
    }
    if set(provenance) != _BAND_PROVENANCE_FIELDS:
        raise StrategyError(
            "scorecard band artifact provenance fields are invalid"
        )
    return provenance


def _selection_provenance(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_scorecard_cutoff_selection(selection)
    source = normalized["source_asset_ref"]
    provenance = {
        "schema_version": (
            SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "selection_schema_version": normalized["schema_version"],
        "producer_version": normalized["producer_version"],
        "task_id": source["task_id"],
        "selection_id": normalized["selection_id"],
        "selection_hash": normalized["selection_hash"],
        "cutoff_id": normalized["cutoff_id"],
        "selection_reason": normalized["selection_reason"],
        "source_artifact_id": source["artifact_id"],
        "source_artifact_content_hash": source["artifact_content_hash"],
        "source_asset_id": source["asset_id"],
        "source_asset_hash": source["asset_hash"],
    }
    if set(provenance) != _SELECTION_PROVENANCE_FIELDS:
        raise StrategyError(
            "scorecard cutoff selection provenance fields are invalid"
        )
    return provenance


def _band_tool_output(
    binding: ScorecardBandAssetArtifactBinding,
    *,
    banding: Mapping[str, Any],
) -> dict[str, Any]:
    asset = binding.asset
    vector = asset["score_vector"]
    return {
        "schema_version": BUILD_SCORECARD_BAND_ASSET_TOOL_SCHEMA_VERSION,
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "dataset_id": asset["identity"]["dataset_id"],
        "sample_design_ref": asset["sample_design_ref"],
        "score_evidence_ref": asset["source_refs"]["score_evidence"],
        "score_vector_ref": asset["source_refs"]["score_vector"],
        "population_count": vector["row_count"],
        "development_count": vector["development_count"],
        "labeled_count": vector["labeled_count"],
        "bad_count": vector["bad_count"],
        "banding": dict(banding),
        "band_count": len(asset["bands"]),
        "cutoff_count": len(asset["cutoffs"]),
        "performance": asset["performance"],
        "scorecard_band_asset": asset,
        "artifacts": [_artifact_output(binding)],
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _selection_tool_output(
    binding: ScorecardCutoffSelectionArtifactBinding,
) -> dict[str, Any]:
    selection = binding.selection
    source = selection["source_asset_ref"]
    return {
        "schema_version": (
            MATERIALIZE_SCORECARD_CUTOFF_SELECTION_TOOL_SCHEMA_VERSION
        ),
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "source_asset_id": source["asset_id"],
        "source_asset_hash": source["asset_hash"],
        "cutoff_id": selection["cutoff_id"],
        "selection_reason": selection["selection_reason"],
        "selection": selection,
        "artifacts": [_artifact_output(binding)],
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _artifact_output(binding) -> dict[str, Any]:
    return {
        "artifact_id": binding.artifact_id,
        "kind": (
            SCORECARD_BAND_ASSET_ARTIFACT_KIND
            if isinstance(binding, ScorecardBandAssetArtifactBinding)
            else SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND
        ),
        "format": "json",
        "filename": binding.path.name,
        "content_hash": binding.content_hash,
        "download_url": (
            f"/api/tasks/{quote(binding.task_id, safe='')}"
            f"/task-artifacts/{quote(binding.artifact_id, safe='')}/download"
            f"?expected_content_hash={binding.content_hash}"
        ),
    }


def _modeling_runtime(runtime):
    """Add modeling repositories to a pack runtime without mutating its owner."""

    if hasattr(runtime, "experiments") and hasattr(runtime, "modeling_repo"):
        return runtime
    proxy = SimpleNamespace(**vars(runtime))
    proxy.experiments = ExperimentStore(runtime.settings.db_path)
    proxy.modeling_repo = ModelingRepository(runtime.settings.db_path)
    return proxy


def _artifact_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    document_id: str,
) -> Path:
    task = _safe_component(task_id, "task_id")
    document = _safe_component(document_id, "document_id")
    return (
        Path(tasks_dir).absolute()
        / task
        / _OUTPUT_DIRECTORY
        / f"{document}.json"
    )


def _prepare_output_directory(
    tasks_dir: Path | str,
    *,
    task_id: str,
) -> Path:
    task = _safe_component(task_id, "task_id")
    root = Path(tasks_dir).absolute()
    output = root / task / _OUTPUT_DIRECTORY
    try:
        if root.is_symlink():
            raise StrategyError(
                "task artifact root must not be a symlink"
            )
        root.mkdir(parents=True, exist_ok=True)
        root_resolved = root.resolve(strict=True)
        task_dir = root / task
        if task_dir.is_symlink():
            raise StrategyError(
                "task artifact directory must not be a symlink"
            )
        task_dir.mkdir(exist_ok=True)
        if task_dir.resolve(strict=True).parent != root_resolved:
            raise StrategyError(
                "scorecard task directory escaped task storage"
            )
        if output.is_symlink():
            raise StrategyError(
                "scorecard output directory must not be a symlink"
            )
        output.mkdir(exist_ok=True)
        if output.resolve(strict=True).parent != task_dir.resolve(strict=True):
            raise StrategyError(
                "scorecard output directory escaped task storage"
            )
    except OSError as exc:
        raise StrategyError(
            "scorecard output directory is unavailable"
        ) from exc
    return output


def _read_exact_artifact(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
    expected_bytes: bytes | None = None,
) -> bytes:
    _require_regular_path(path, root=root)
    before = path.lstat()
    if before.st_size > _MAX_ARTIFACT_BYTES:
        raise StrategyError("scorecard artifact exceeds byte budget")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StrategyError("scorecard artifact could not be read") from exc
    _require_regular_path(path, root=root)
    after = path.lstat()
    if _stat_identity(before) != _stat_identity(after):
        raise StrategyError("scorecard artifact changed while read")
    if not hmac.compare_digest(_sha256(raw), expected_hash):
        raise StrategyError("scorecard artifact content hash changed")
    if expected_bytes is not None and raw != expected_bytes:
        raise StrategyError("scorecard artifact canonical bytes changed")
    return raw


def _require_regular_path(path: Path, *, root: Path) -> None:
    absolute = path.absolute()
    declared_root = root.absolute()
    try:
        absolute.relative_to(declared_root)
    except ValueError as exc:
        raise StrategyError(
            "scorecard artifact escaped task storage"
        ) from exc
    current = absolute
    while True:
        if current.is_symlink():
            raise StrategyError("scorecard artifact must not use symlinks")
        if current == declared_root:
            break
        if current == current.parent:
            raise StrategyError(
                "scorecard artifact escaped task storage"
            )
        current = current.parent
    try:
        mode = absolute.stat().st_mode
    except OSError as exc:
        raise StrategyError("scorecard artifact is unavailable") from exc
    if not stat.S_ISREG(mode):
        raise StrategyError("scorecard artifact must be a regular file")


def _parse_band_asset(raw: bytes) -> dict[str, Any]:
    value = _parse_json_object(raw, "scorecard band artifact")
    try:
        return validate_scorecard_band_asset(value)
    except StrategyError:
        raise
    except (TypeError, ValueError) as exc:
        raise StrategyError("scorecard band artifact is invalid") from exc


def _parse_cutoff_selection(raw: bytes) -> dict[str, Any]:
    value = _parse_json_object(raw, "scorecard cutoff selection")
    try:
        return validate_scorecard_cutoff_selection(value)
    except StrategyError:
        raise
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "scorecard cutoff selection is invalid"
        ) from exc


def _parse_json_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StrategyError(f"{name} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise StrategyError(f"{name} must contain an object")
    return value


def _require_record(
    record: Mapping[str, Any],
    *,
    task_id: str,
    artifact_id: str,
    kind: str,
    origin_tool: str,
    path: Path,
    content_hash: str,
    expected_provenance: Mapping[str, Any] | None = None,
) -> None:
    if set(record) != _TASK_ARTIFACT_RECORD_FIELDS:
        raise StrategyError(
            "scorecard artifact registry fields changed"
        )
    expected = {
        "id": artifact_id,
        "task_id": task_id,
        "kind": kind,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": origin_tool,
    }
    changed = [
        field
        for field, expected_value in expected.items()
        if record.get(field) != expected_value
    ]
    if changed:
        raise StrategyError(
            "scorecard artifact registry binding changed: "
            + ", ".join(changed)
        )
    provenance = _strict_provenance(record.get("provenance"))
    if expected_provenance is not None and provenance != dict(
        expected_provenance
    ):
        raise StrategyError("scorecard artifact registry provenance changed")


def _record_from_sql_row(row) -> dict[str, Any]:
    provenance_json = row["provenance_json"]
    if not isinstance(provenance_json, str):
        raise StrategyError(
            "scorecard artifact provenance_json is invalid"
        )
    try:
        provenance = json.loads(
            provenance_json,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise StrategyError(
            "scorecard artifact provenance_json is invalid"
        ) from exc
    if (
        not isinstance(provenance, dict)
        or _canonical_json(provenance) != provenance_json
    ):
        raise StrategyError(
            "scorecard artifact provenance_json is not canonical"
        )
    return {
        "id": str(row["id"]),
        "task_id": str(row["task_id"]),
        "kind": str(row["kind"]),
        "path": str(row["path"]),
        "content_hash": str(row["content_hash"]),
        "origin_tool": str(row["origin_tool"]),
        "provenance": provenance,
        "created_at": str(row["created_at"]),
    }


def _strict_provenance(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(
            "scorecard artifact provenance must be an object"
        )
    try:
        encoded = _canonical_json(value)
        normalized = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(
            "scorecard artifact provenance is invalid"
        ) from exc
    if not isinstance(normalized, dict):
        raise StrategyError(
            "scorecard artifact provenance must be an object"
        )
    return normalized


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{name} keys must be strings")
    return dict(value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    unsupported = sorted(set(value) - expected)
    if missing or unsupported:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unsupported:
            details.append("unsupported: " + ", ".join(unsupported))
        raise StrategyError(
            f"{name} fields are invalid ({'; '.join(details)})"
        )


def _safe_component(value: object, name: str) -> str:
    text = _canonical_text(value, name)
    if (
        _SAFE_COMPONENT_RE.fullmatch(text) is None
        or Path(text).name != text
        or text in {".", ".."}
    ):
        raise StrategyError(f"{name} is unsafe for artifact storage")
    return text


def _canonical_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrategyError(f"{name} must be non-empty canonical text")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


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
        raise StrategyError(
            "scorecard artifact data must be finite JSON"
        ) from exc


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _stat_identity(value) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "BUILD_SCORECARD_BAND_ASSET_TOOL_SCHEMA_VERSION",
    "MATERIALIZE_SCORECARD_CUTOFF_SELECTION_TOOL_SCHEMA_VERSION",
    "SCORECARD_BAND_ASSET_ARTIFACT_KIND",
    "SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND",
    "ScorecardBandAssetArtifactBinding",
    "ScorecardCutoffSelectionArtifactBinding",
    "load_historical_scorecard_band_asset_artifact",
    "load_historical_scorecard_cutoff_selection_artifact",
    "load_scorecard_band_asset_artifact",
    "load_scorecard_cutoff_selection_artifact",
    "require_historical_scorecard_band_asset_artifact_binding_on_connection",
    "require_historical_scorecard_cutoff_selection_artifact_binding_on_connection",
    "require_scorecard_band_asset_artifact_binding_on_connection",
    "require_scorecard_cutoff_selection_artifact_binding_on_connection",
    "run_build_scorecard_band_asset",
    "run_materialize_scorecard_cutoff_selection",
]
