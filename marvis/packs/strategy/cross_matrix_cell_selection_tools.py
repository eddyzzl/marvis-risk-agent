"""Governed persistence and strict reload for Cross Matrix cell selections.

The full Cross Matrix remains authoritative for predicates, measurements, and
lifecycle.  This boundary verifies one exact task-owned matrix artifact and its
bound dataset, persists only the pointer-only cell-group audit event, and
rechecks source and dataset drift under the SQLite writer lock.  Public strict
loaders are shared with Strategy Pool so stateful admission can replay the same
lineage while holding its own writer lock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
import stat
from typing import Any
import unicodedata
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import DatasetContentDriftError
from marvis.files import sha256_file
from marvis.packs.strategy.cross_matrix_candidate import (
    CrossMatrixCandidateAssetError,
    canonical_cross_matrix_candidate_asset_json,
    parse_cross_matrix_candidate_asset_json,
)
from marvis.packs.strategy.cross_matrix_cell_selection import (
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION,
    CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
    CROSS_MATRIX_CELL_SELECTION_PRODUCER_VERSION,
    CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
    CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
    CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION,
    MAX_SELECTION_REASON_LENGTH,
    CrossMatrixCellSelectionError,
    IndependentlyVerifiedCrossMatrixArtifactBinding,
    build_cross_matrix_cell_selection,
    canonical_cross_matrix_cell_selection_json,
    derive_cross_matrix_cell_group_facts,
    validate_cross_matrix_cell_selection,
    validate_cross_matrix_source_artifact_binding,
)
from marvis.packs.strategy.errors import StrategyError


TOOL_SCHEMA_VERSION = "strategy.materialize-cross-matrix-cell-selection-tool.v1"

SOURCE_PROVENANCE_FIELDS = frozenset(
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
SELECTION_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
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

_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
        "cell_ids",
        "selection_reason",
    }
)
_REQUIRED_INPUT_FIELDS = _INPUT_FIELDS - {"selection_reason"}
_TASK_ARTIFACT_ROW_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "kind",
        "path",
        "content_hash",
        "origin_tool",
        "provenance_json",
        "created_at",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_SELECTION_ID_RE = re.compile(r"^cross-matrix-cell-selection-[0-9a-f]{32}$")


@dataclass(frozen=True)
class VerifiedCrossMatrixSource:
    """One fully verified live Cross Matrix row and canonical asset bytes."""

    artifact_id: str
    task_id: str
    kind: str
    path: Path
    content_hash: str
    origin_tool: str
    provenance: dict[str, Any]
    canonical_bytes: bytes
    asset: dict[str, Any]

    def builder_binding(self) -> IndependentlyVerifiedCrossMatrixArtifactBinding:
        """Project the exact source snapshot required by the pure builder."""

        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "artifact_schema_version": self.provenance["schema_version"],
            "content_hash": self.content_hash,
            "origin_tool": self.origin_tool,
            "path": str(self.path),
            "provenance_hash": _sha256_bytes(
                _canonical_json(self.provenance).encode("utf-8")
            ),
            "provenance": self.provenance,
            "canonical_bytes": self.canonical_bytes,
        }

    def source_binding(self) -> IndependentlyVerifiedCrossMatrixArtifactBinding:
        """Alias used by callers that describe the matrix as a source."""

        return self.builder_binding()


@dataclass(frozen=True)
class VerifiedCrossMatrixCellSelection:
    """One fully verified live pointer-only selection row and canonical bytes."""

    artifact_id: str
    task_id: str
    kind: str
    path: Path
    content_hash: str
    origin_tool: str
    provenance: dict[str, Any]
    canonical_bytes: bytes
    selection: dict[str, Any]

    def replay_binding(self) -> dict[str, Any]:
        """Project exact registry facts required by the pure replay adapter."""

        provenance = self.provenance
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "content_hash": self.content_hash,
            "origin_tool": self.origin_tool,
            "artifact_schema_version": provenance["schema_version"],
            "producer_version": provenance["producer_version"],
            "selection_id": provenance["selection_id"],
            "selection_hash": provenance["selection_hash"],
            "group_id": provenance["group_id"],
            "source_artifact_id": provenance["source_artifact_id"],
            "source_artifact_kind": provenance["source_artifact_kind"],
            "source_artifact_schema_version": provenance[
                "source_artifact_schema_version"
            ],
            "source_artifact_content_hash": provenance["source_artifact_content_hash"],
            "source_artifact_origin_tool": provenance["source_artifact_origin_tool"],
            "source_artifact_path": provenance["source_artifact_path"],
            "source_artifact_provenance_hash": provenance[
                "source_artifact_provenance_hash"
            ],
            "source_asset_schema_version": provenance["source_asset_schema_version"],
            "source_asset_id": provenance["source_asset_id"],
            "source_asset_hash": provenance["source_asset_hash"],
            "source_asset_type": provenance["source_asset_type"],
            "source_candidate_id": provenance["source_candidate_id"],
            "source_evidence_hash": provenance["source_evidence_hash"],
            "source_evidence_identity": provenance["source_evidence_identity"],
            "cell_ids": provenance["cell_ids"],
        }


@dataclass(frozen=True)
class _VerifiedDatasetBinding:
    dataset_id: str
    task_id: str
    source_path: str
    path: Path
    content_hash: str
    registry_metadata_hash: str
    columns: tuple[str, ...]
    row_count: int


def run_materialize_cross_matrix_cell_selection(
    inputs: object,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Persist one explicit pointer to a verified non-empty cell group."""

    request = _validate_inputs(inputs)
    task_id = _required_text(ctx.task_id, "task_id")
    source = load_verified_cross_matrix_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=request["source_artifact_id"],
        expected_content_hash=request["expected_artifact_content_hash"],
        expected_asset_id=request["expected_asset_id"],
        expected_asset_hash=request["expected_asset_hash"],
        expected_candidate_id=request["expected_candidate_id"],
        expected_evidence_hash=request["expected_evidence_hash"],
    )
    dataset = _load_verified_dataset_binding(runtime, source=source)
    selection = _build_selection(
        source,
        cell_ids=request["cell_ids"],
        selection_reason=request.get("selection_reason"),
    )
    facts = derive_cross_matrix_cell_group_facts(
        source.asset,
        cell_ids=selection["cell_ids"],
    )
    canonical_content = canonical_cross_matrix_cell_selection_json(selection).encode(
        "utf-8"
    )
    content_hash = _sha256_bytes(canonical_content)
    provenance = cross_matrix_cell_selection_provenance(selection)

    # Detect ordinary read-time drift before staging and repeat under lock.
    reloaded_source = load_verified_cross_matrix_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=request["source_artifact_id"],
        expected_content_hash=request["expected_artifact_content_hash"],
        expected_asset_id=request["expected_asset_id"],
        expected_asset_hash=request["expected_asset_hash"],
        expected_candidate_id=request["expected_candidate_id"],
        expected_evidence_hash=request["expected_evidence_hash"],
    )
    if reloaded_source != source:
        raise StrategyError("Cross Matrix source binding changed before persistence")
    _require_dataset_unchanged(runtime, dataset)
    artifact = _persist_selection(
        runtime,
        task_id=task_id,
        request=request,
        source=source,
        dataset=dataset,
        selection=selection,
        canonical_content=canonical_content,
        content_hash=content_hash,
        provenance=provenance,
    )
    fragment = facts["fragment"]
    lifecycle = source.asset["lifecycle"]
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "selection_reason": selection["selection_reason"],
        "group_id": selection["group_id"],
        "cell_ids": selection["cell_ids"],
        "source_asset_id": source.asset["asset_id"],
        "source_asset_hash": source.asset["asset_hash"],
        "source_candidate_id": source.asset["candidate_evidence"]["candidate_id"],
        "source_evidence_hash": source.asset["candidate_evidence"]["evidence_hash"],
        "fragment_id": fragment["fragment_id"],
        "fragment_type": fragment["fragment_type"],
        "rule_id": fragment["rule_id"],
        "effect_id": fragment["effect_id"],
        "candidate_stage": lifecycle["candidate_stage"],
        "observation_stage": lifecycle["observation_stage"],
        "validation_status": lifecycle["validation_status"],
        "artifacts": [artifact],
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def canonical_cross_matrix_source_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    asset_id: str,
    content_hash: str,
) -> Path:
    """Return the sole CROSS-1 source path without resolving symlinks."""

    normalized_task = _safe_component(task_id, "task_id")
    normalized_asset = _required_asset_id(asset_id, "asset_id")
    normalized_hash = _required_sha256(content_hash, "content_hash")
    return (
        Path(tasks_dir).absolute()
        / normalized_task
        / "strategy_cross_matrix_candidates"
        / f"{normalized_asset}_{normalized_hash[:12]}.json"
    )


def canonical_cross_matrix_cell_selection_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    selection_id: str,
) -> Path:
    """Return the sole pointer artifact path without resolving symlinks."""

    normalized_task = _safe_component(task_id, "task_id")
    normalized_selection = _required_text(selection_id, "selection_id")
    if _SELECTION_ID_RE.fullmatch(normalized_selection) is None:
        raise StrategyError("selection_id has an invalid format")
    return (
        Path(tasks_dir).absolute()
        / normalized_task
        / "strategy_cross_matrix_cell_selections"
        / f"{normalized_selection}.json"
    )


def verify_cross_matrix_source_provenance(
    provenance_payload: Mapping[str, Any],
    *,
    source_binding: Mapping[str, Any],
    asset_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify exact CROSS-1 provenance and return a detached mapping."""

    provenance = _canonical_json_object(provenance_payload, "source provenance")
    _require_exact_fields(provenance, SOURCE_PROVENANCE_FIELDS, "source provenance")
    candidate_binding = {
        **source_binding,
        "provenance_hash": _sha256_bytes(_canonical_json(provenance).encode("utf-8")),
        "provenance": provenance,
    }
    try:
        validate_cross_matrix_source_artifact_binding(
            asset_payload,
            candidate_binding,  # type: ignore[arg-type]
        )
    except CrossMatrixCellSelectionError as exc:
        raise StrategyError(
            "Cross Matrix source provenance does not match canonical asset"
        ) from exc
    return provenance


def cross_matrix_cell_selection_provenance(
    selection_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive exact pointer-only TaskArtifact provenance."""

    try:
        selection = validate_cross_matrix_cell_selection(selection_payload)
    except CrossMatrixCellSelectionError as exc:
        raise StrategyError(
            f"Cross Matrix cell selection failed validation: {exc}"
        ) from exc
    source_artifact = selection["source_artifact"]
    source_asset = selection["source_asset"]
    source_candidate = selection["source_candidate"]
    provenance = {
        "schema_version": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION,
        "producer_version": selection["producer_version"],
        "task_id": source_artifact["task_id"],
        "kind": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        "format": "json",
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
    if set(provenance) != SELECTION_PROVENANCE_FIELDS:
        raise StrategyError("Cross Matrix selection provenance fields drifted")
    return provenance


def verify_cross_matrix_cell_selection_provenance(
    provenance_payload: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and detach exact selection provenance for Pool reuse."""

    actual = _canonical_json_object(provenance_payload, "selection provenance")
    _require_exact_fields(
        actual,
        SELECTION_PROVENANCE_FIELDS,
        "selection provenance",
    )
    expected = cross_matrix_cell_selection_provenance(selection_payload)
    if not hmac.compare_digest(_canonical_json(actual), _canonical_json(expected)):
        raise StrategyError(
            "Cross Matrix selection provenance does not match selection"
        )
    return actual


def load_verified_cross_matrix_source_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    expected_candidate_id: str,
    expected_evidence_hash: str,
) -> VerifiedCrossMatrixSource:
    """Load and fully verify one current-task Cross Matrix artifact."""

    with runtime.task_artifacts.transaction() as conn:
        return load_verified_cross_matrix_source_artifact_on_connection(
            conn,
            tasks_dir=runtime.settings.tasks_dir,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
            expected_candidate_id=expected_candidate_id,
            expected_evidence_hash=expected_evidence_hash,
        )


def load_verified_cross_matrix_source_artifact_on_connection(
    conn,
    *,
    tasks_dir: Path | str,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    expected_candidate_id: str,
    expected_evidence_hash: str,
) -> VerifiedCrossMatrixSource:
    """Connection-scoped source verifier used under writer locks and by Pool."""

    normalized_task = _required_text(task_id, "task_id")
    normalized_artifact = _required_text(artifact_id, "source_artifact_id")
    normalized_content_hash = _required_sha256(
        expected_content_hash,
        "expected_artifact_content_hash",
    )
    normalized_asset_id = _required_asset_id(
        expected_asset_id,
        "expected_asset_id",
    )
    normalized_asset_hash = _required_sha256(
        expected_asset_hash,
        "expected_asset_hash",
    )
    normalized_candidate_id = _required_candidate_id(
        expected_candidate_id,
        "expected_candidate_id",
    )
    normalized_evidence_hash = _required_sha256(
        expected_evidence_hash,
        "expected_evidence_hash",
    )
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (normalized_task, normalized_artifact),
    ).fetchone()
    if row is None:
        raise StrategyError(
            f"Cross Matrix source artifact not found: {normalized_artifact}"
        )
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _require_exact_fields(record, _TASK_ARTIFACT_ROW_FIELDS, "source artifact row")
    fixed = {
        "id": normalized_artifact,
        "task_id": normalized_task,
        "kind": CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
        "origin_tool": CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
    }
    for field, expected in fixed.items():
        if _required_text(record[field], f"source artifact {field}") != expected:
            raise StrategyError(f"Cross Matrix source artifact {field} changed")
    registered_hash = _required_sha256(
        record["content_hash"],
        "source artifact content_hash",
    )
    if not hmac.compare_digest(registered_hash, normalized_content_hash):
        raise StrategyError("Cross Matrix source artifact content hash changed")
    path = Path(_required_text(record["path"], "source artifact path"))
    expected_path = canonical_cross_matrix_source_path(
        tasks_dir,
        task_id=normalized_task,
        asset_id=normalized_asset_id,
        content_hash=registered_hash,
    )
    if not path.is_absolute() or path != expected_path:
        raise StrategyError("Cross Matrix source artifact path is not canonical")
    canonical_bytes = _read_stable_regular_file(
        path,
        root=Path(tasks_dir).absolute(),
        label="Cross Matrix source artifact",
    )
    if not hmac.compare_digest(_sha256_bytes(canonical_bytes), registered_hash):
        raise StrategyError("Cross Matrix source artifact content hash drifted")
    try:
        asset = parse_cross_matrix_candidate_asset_json(canonical_bytes)
    except (CrossMatrixCandidateAssetError, TypeError, ValueError) as exc:
        raise StrategyError(
            "Cross Matrix source artifact failed strict validation"
        ) from exc
    canonical = canonical_cross_matrix_candidate_asset_json(asset).encode("utf-8")
    if not hmac.compare_digest(canonical_bytes, canonical):
        raise StrategyError("Cross Matrix source artifact is not canonical JSON")
    if asset["sample_identity"]["task_id"] != normalized_task:
        raise StrategyError("Cross Matrix source asset belongs to another task")
    comparisons = {
        "asset_id": (asset["asset_id"], normalized_asset_id),
        "asset_hash": (asset["asset_hash"], normalized_asset_hash),
        "candidate_id": (
            asset["candidate_evidence"]["candidate_id"],
            normalized_candidate_id,
        ),
        "evidence_hash": (
            asset["candidate_evidence"]["evidence_hash"],
            normalized_evidence_hash,
        ),
    }
    for field, (actual, expected) in comparisons.items():
        matches = (
            hmac.compare_digest(actual, expected)
            if field.endswith("hash")
            else actual == expected
        )
        if not matches:
            raise StrategyError(f"Cross Matrix source {field} changed")
    provenance_json = record["provenance_json"]
    if not isinstance(provenance_json, str):
        raise StrategyError("Cross Matrix source provenance_json is invalid")
    unverified_provenance = _strict_json_object_from_text(
        provenance_json,
        "Cross Matrix source provenance_json",
    )
    binding_without_provenance = {
        "artifact_id": normalized_artifact,
        "task_id": normalized_task,
        "kind": CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
        "artifact_schema_version": CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION,
        "content_hash": registered_hash,
        "origin_tool": CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
        "path": str(path),
        "canonical_bytes": canonical_bytes,
    }
    provenance = verify_cross_matrix_source_provenance(
        unverified_provenance,
        source_binding=binding_without_provenance,
        asset_payload=asset,
    )
    return VerifiedCrossMatrixSource(
        artifact_id=normalized_artifact,
        task_id=normalized_task,
        kind=CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
        path=path,
        content_hash=registered_hash,
        origin_tool=CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
        provenance=provenance,
        canonical_bytes=canonical_bytes,
        asset=asset,
    )


def load_verified_cross_matrix_cell_selection_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> VerifiedCrossMatrixCellSelection:
    """Load and fully verify one current-task persisted cell selection."""

    with runtime.task_artifacts.transaction() as conn:
        return load_verified_cross_matrix_cell_selection_artifact_on_connection(
            conn,
            tasks_dir=runtime.settings.tasks_dir,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
        )


def load_verified_cross_matrix_cell_selection_artifact_on_connection(
    conn,
    *,
    tasks_dir: Path | str,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> VerifiedCrossMatrixCellSelection:
    """Connection-scoped strict verifier for a selection TaskArtifact."""

    normalized_task = _required_text(task_id, "task_id")
    normalized_artifact = _required_text(artifact_id, "selection_artifact_id")
    normalized_content_hash = _required_sha256(
        expected_content_hash,
        "expected_artifact_content_hash",
    )
    normalized_asset_id = _required_asset_id(
        expected_asset_id,
        "expected_asset_id",
    )
    normalized_asset_hash = _required_sha256(
        expected_asset_hash,
        "expected_asset_hash",
    )
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (normalized_task, normalized_artifact),
    ).fetchone()
    if row is None:
        raise StrategyError(
            f"Cross Matrix cell selection artifact not found: {normalized_artifact}"
        )
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _require_exact_fields(record, _TASK_ARTIFACT_ROW_FIELDS, "selection artifact row")
    fixed = {
        "id": normalized_artifact,
        "task_id": normalized_task,
        "kind": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        "origin_tool": CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
    }
    for field, expected in fixed.items():
        if _required_text(record[field], f"selection artifact {field}") != expected:
            raise StrategyError(f"Cross Matrix selection artifact {field} changed")
    registered_hash = _required_sha256(
        record["content_hash"],
        "selection artifact content_hash",
    )
    if not hmac.compare_digest(registered_hash, normalized_content_hash):
        raise StrategyError("Cross Matrix selection artifact content hash changed")
    provenance_json = record["provenance_json"]
    if not isinstance(provenance_json, str):
        raise StrategyError("Cross Matrix selection provenance_json is invalid")
    unverified_provenance = _strict_json_object_from_text(
        provenance_json,
        "Cross Matrix selection provenance_json",
    )
    _require_exact_fields(
        unverified_provenance,
        SELECTION_PROVENANCE_FIELDS,
        "selection provenance",
    )
    provenance_fixed = {
        "schema_version": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION,
        "producer_version": CROSS_MATRIX_CELL_SELECTION_PRODUCER_VERSION,
        "task_id": normalized_task,
        "kind": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        "format": "json",
        "source_asset_id": normalized_asset_id,
        "source_asset_hash": normalized_asset_hash,
    }
    for field, expected in provenance_fixed.items():
        actual = unverified_provenance[field]
        matches = (
            hmac.compare_digest(str(actual), expected)
            if field.endswith("hash")
            else actual == expected
        )
        if not matches:
            raise StrategyError(f"Cross Matrix selection provenance {field} changed")
    selection_id = _required_text(
        unverified_provenance["selection_id"],
        "selection provenance selection_id",
    )
    path = Path(_required_text(record["path"], "selection artifact path"))
    expected_path = canonical_cross_matrix_cell_selection_path(
        tasks_dir,
        task_id=normalized_task,
        selection_id=selection_id,
    )
    if not path.is_absolute() or path != expected_path:
        raise StrategyError("Cross Matrix selection artifact path is not canonical")
    canonical_bytes = _read_stable_regular_file(
        path,
        root=Path(tasks_dir).absolute(),
        label="Cross Matrix selection artifact",
    )
    if not hmac.compare_digest(_sha256_bytes(canonical_bytes), registered_hash):
        raise StrategyError("Cross Matrix selection artifact content hash drifted")
    selection = _strict_cell_selection_from_bytes(canonical_bytes)
    canonical = canonical_cross_matrix_cell_selection_json(selection).encode("utf-8")
    if not hmac.compare_digest(canonical_bytes, canonical):
        raise StrategyError("Cross Matrix selection artifact is not canonical JSON")
    provenance = verify_cross_matrix_cell_selection_provenance(
        unverified_provenance,
        selection,
    )
    if selection["source_asset"][
        "asset_id"
    ] != normalized_asset_id or not hmac.compare_digest(
        selection["source_asset"]["asset_hash"],
        normalized_asset_hash,
    ):
        raise StrategyError("Cross Matrix selection asset binding changed")
    return VerifiedCrossMatrixCellSelection(
        artifact_id=normalized_artifact,
        task_id=normalized_task,
        kind=CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        path=path,
        content_hash=registered_hash,
        origin_tool=CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
        provenance=provenance,
        canonical_bytes=canonical_bytes,
        selection=selection,
    )


def _build_selection(
    source: VerifiedCrossMatrixSource,
    *,
    cell_ids: Sequence[str],
    selection_reason: str | None,
) -> dict[str, Any]:
    try:
        return build_cross_matrix_cell_selection(
            source.asset,
            source_artifact_binding=source.builder_binding(),
            cell_ids=cell_ids,
            selection_reason=selection_reason,
        )
    except CrossMatrixCellSelectionError as exc:
        raise StrategyError(
            f"Cross Matrix cell selection failed validation: {exc}"
        ) from exc


def _load_verified_dataset_binding(
    runtime,
    *,
    source: VerifiedCrossMatrixSource,
) -> _VerifiedDatasetBinding:
    sample = source.asset["sample_identity"]
    dataset_id = sample["dataset_id"]
    try:
        dataset = runtime.registry.get(dataset_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategyError(
            f"Cross Matrix source dataset not found: {dataset_id}"
        ) from exc
    if str(dataset.task_id) != source.task_id:
        raise StrategyError("Cross Matrix source dataset belongs to another task")
    content_hash = str(dataset.content_hash or "")
    if not _matches_sha256(content_hash, sample["dataset_content_hash"]):
        raise StrategyError("Cross Matrix source dataset content hash changed")
    try:
        path = Path(runtime.registry.resolve_verified_path(dataset_id))
    except (
        DatasetContentDriftError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError(
            "Cross Matrix source dataset failed hash verification"
        ) from exc
    _require_file_content_hash(
        path,
        content_hash,
        "Cross Matrix source dataset content hash drifted",
    )
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=source.task_id,
            dataset_id=dataset_id,
            expected_content_hash=content_hash,
        )
    if not hmac.compare_digest(
        metadata_hash,
        source.provenance["registry_metadata_hash"],
    ):
        raise StrategyError("Cross Matrix source dataset registry metadata changed")
    return _VerifiedDatasetBinding(
        dataset_id=dataset_id,
        task_id=source.task_id,
        source_path=str(dataset.source_path),
        path=path,
        content_hash=content_hash,
        registry_metadata_hash=metadata_hash,
        columns=tuple(str(profile.name) for profile in dataset.columns),
        row_count=int(dataset.row_count),
    )


def _persist_selection(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    source: VerifiedCrossMatrixSource,
    dataset: _VerifiedDatasetBinding,
    selection: Mapping[str, Any],
    canonical_content: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    selection_id = _required_text(selection["selection_id"], "selection_id")
    out_dir = _prepare_selection_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = canonical_cross_matrix_cell_selection_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        selection_id=selection_id,
    )
    if final_path.parent != out_dir:
        raise StrategyError("Cross Matrix selection output path drifted")
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    staged.path.write_bytes(canonical_content)
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                locked_source = (
                    load_verified_cross_matrix_source_artifact_on_connection(
                        conn,
                        tasks_dir=runtime.settings.tasks_dir,
                        task_id=task_id,
                        artifact_id=request["source_artifact_id"],
                        expected_content_hash=request["expected_artifact_content_hash"],
                        expected_asset_id=request["expected_asset_id"],
                        expected_asset_hash=request["expected_asset_hash"],
                        expected_candidate_id=request["expected_candidate_id"],
                        expected_evidence_hash=request["expected_evidence_hash"],
                    )
                )
                _require_dataset_on_connection(conn, dataset)
                _require_file_content_hash(
                    dataset.path,
                    dataset.content_hash,
                    "Cross Matrix source dataset content hash drifted",
                )
                locked_selection = _build_selection(
                    locked_source,
                    cell_ids=request["cell_ids"],
                    selection_reason=request.get("selection_reason"),
                )
                locked_content = canonical_cross_matrix_cell_selection_json(
                    locked_selection
                ).encode("utf-8")
                locked_provenance = cross_matrix_cell_selection_provenance(
                    locked_selection
                )
                if locked_source != source:
                    raise StrategyError(
                        "Cross Matrix source binding changed before registration"
                    )
                if locked_selection != selection or not hmac.compare_digest(
                    locked_content,
                    canonical_content,
                ):
                    raise StrategyError(
                        "Cross Matrix cell selection changed before registration"
                    )
                if not hmac.compare_digest(
                    _canonical_json(locked_provenance),
                    _canonical_json(provenance),
                ):
                    raise StrategyError(
                        "Cross Matrix selection provenance changed before registration"
                    )
                _prepare_selection_directory(
                    runtime.settings.tasks_dir,
                    task_id=task_id,
                )
                _require_existing_selection_consistent(
                    conn,
                    task_id=task_id,
                    final_path=final_path,
                    canonical_content=canonical_content,
                    content_hash=content_hash,
                    provenance=provenance,
                )
                uow.promote_all()
                _verify_selection_file(
                    final_path,
                    root=Path(runtime.settings.tasks_dir).absolute(),
                    expected_content=canonical_content,
                    expected_content_hash=content_hash,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=content_hash,
                    origin_tool=CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
                    provenance=provenance,
                )
                _require_registered_selection_record(
                    record,
                    task_id=task_id,
                    final_path=final_path,
                    content_hash=content_hash,
                    provenance=provenance,
                )
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
    return {
        "artifact_id": str(record["id"]),
        "kind": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        "format": "json",
        "filename": final_path.name,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
        ),
    }


def _require_existing_selection_consistent(
    conn,
    *,
    task_id: str,
    final_path: Path,
    canonical_content: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (
            task_id,
            CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
            str(final_path),
        ),
    ).fetchone()
    if row is None:
        if final_path.exists() or final_path.is_symlink():
            raise StrategyError(
                "Cross Matrix selection path exists without a registry row"
            )
        return
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _require_registered_selection_record(
        record,
        task_id=task_id,
        final_path=final_path,
        content_hash=content_hash,
        provenance=provenance,
        raw_provenance=True,
    )
    _verify_selection_file(
        final_path,
        root=final_path.parents[2],
        expected_content=canonical_content,
        expected_content_hash=content_hash,
    )


def _require_registered_selection_record(
    record: Mapping[str, Any],
    *,
    task_id: str,
    final_path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
    raw_provenance: bool = False,
) -> None:
    expected = {
        "task_id": task_id,
        "kind": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        "path": str(final_path),
        "content_hash": content_hash,
        "origin_tool": CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
    }
    for field, expected_value in expected.items():
        actual = record.get(field)
        matches = (
            hmac.compare_digest(str(actual), expected_value)
            if field == "content_hash"
            else actual == expected_value
        )
        if not matches:
            raise StrategyError(f"Cross Matrix selection registry {field} changed")
    if raw_provenance:
        provenance_json = record.get("provenance_json")
        if not isinstance(provenance_json, str):
            raise StrategyError("Cross Matrix selection provenance_json is invalid")
        actual_provenance = _strict_json_object_from_text(
            provenance_json,
            "Cross Matrix selection provenance_json",
        )
    else:
        actual_provenance = _canonical_json_object(
            record.get("provenance"),
            "Cross Matrix selection registry provenance",
        )
    if not hmac.compare_digest(
        _canonical_json(actual_provenance),
        _canonical_json(provenance),
    ):
        raise StrategyError("Cross Matrix selection registry provenance changed")


def _verify_selection_file(
    path: Path,
    *,
    root: Path,
    expected_content: bytes,
    expected_content_hash: str,
) -> None:
    persisted = _read_stable_regular_file(
        path,
        root=root,
        label="Cross Matrix selection artifact",
    )
    if not hmac.compare_digest(persisted, expected_content):
        raise StrategyError("Cross Matrix selection artifact bytes changed")
    if not hmac.compare_digest(_sha256_bytes(persisted), expected_content_hash):
        raise StrategyError("Cross Matrix selection artifact hash changed")
    parsed = _strict_cell_selection_from_bytes(persisted)
    canonical = canonical_cross_matrix_cell_selection_json(parsed).encode("utf-8")
    if not hmac.compare_digest(persisted, canonical):
        raise StrategyError("Cross Matrix selection artifact is not canonical JSON")


def _prepare_selection_directory(
    tasks_dir: Path | str,
    *,
    task_id: str,
) -> Path:
    normalized_task = _safe_component(task_id, "task_id")
    root = Path(tasks_dir).absolute()
    try:
        if root.is_symlink():
            raise StrategyError("task artifact root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve(strict=True)
        task_dir = root / normalized_task
        if task_dir.is_symlink():
            raise StrategyError("task artifact directory must not be a symlink")
        task_dir.mkdir(exist_ok=True)
        if task_dir.resolve(strict=True).parent != resolved_root:
            raise StrategyError("task artifact directory escaped task storage")
        out_dir = task_dir / "strategy_cross_matrix_cell_selections"
        if out_dir.is_symlink():
            raise StrategyError(
                "Cross Matrix selection directory must not be a symlink"
            )
        out_dir.mkdir(exist_ok=True)
        if out_dir.resolve(strict=True).parent != task_dir.resolve(strict=True):
            raise StrategyError("Cross Matrix selection directory escaped task storage")
    except OSError as exc:
        raise StrategyError("Cross Matrix selection directory is unavailable") from exc
    return out_dir


def _require_dataset_unchanged(
    runtime,
    dataset: _VerifiedDatasetBinding,
) -> None:
    try:
        live = runtime.registry.get(dataset.dataset_id)
        live_path = Path(runtime.registry.resolve_verified_path(dataset.dataset_id))
    except (
        DatasetContentDriftError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError("Cross Matrix source dataset binding changed") from exc
    live_columns = tuple(str(profile.name) for profile in live.columns)
    _require_file_content_hash(
        live_path,
        dataset.content_hash,
        "Cross Matrix source dataset binding changed",
    )
    if (
        str(live.task_id) != dataset.task_id
        or str(live.content_hash or "") != dataset.content_hash
        or int(live.row_count) != dataset.row_count
        or live_columns != dataset.columns
        or live_path != dataset.path
        or str(live.source_path) != dataset.source_path
    ):
        raise StrategyError("Cross Matrix source dataset binding changed")
    with runtime.task_artifacts.transaction() as conn:
        _require_dataset_on_connection(conn, dataset)


def _require_dataset_on_connection(
    conn,
    dataset: _VerifiedDatasetBinding,
) -> None:
    metadata_hash = _registry_metadata_hash_on_connection(
        conn,
        task_id=dataset.task_id,
        dataset_id=dataset.dataset_id,
        expected_content_hash=dataset.content_hash,
    )
    if not hmac.compare_digest(metadata_hash, dataset.registry_metadata_hash):
        raise StrategyError("Cross Matrix source dataset registry metadata changed")
    row = conn.execute(
        "SELECT source_path FROM datasets WHERE task_id = ? AND id = ?",
        (dataset.task_id, dataset.dataset_id),
    ).fetchone()
    if row is None or str(row["source_path"]) != dataset.source_path:
        raise StrategyError("Cross Matrix source dataset registry path changed")


def _registry_metadata_hash_on_connection(
    conn,
    *,
    task_id: str,
    dataset_id: str,
    expected_content_hash: str,
) -> str:
    row = conn.execute(
        """
        SELECT task_id, role, row_count, columns_json, has_target, target_col,
               content_hash
          FROM datasets
         WHERE id = ?
        """,
        (dataset_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError(f"Cross Matrix source dataset not found: {dataset_id}")
    if not _matches_sha256(row["content_hash"], expected_content_hash):
        raise StrategyError("Cross Matrix source dataset registered hash changed")
    columns_json = row["columns_json"]
    if not isinstance(columns_json, str):
        raise StrategyError("Cross Matrix source dataset schema is invalid")
    try:
        json.loads(columns_json)
    except json.JSONDecodeError as exc:
        raise StrategyError("Cross Matrix source dataset schema is invalid") from exc
    payload = {
        "role": str(row["role"]),
        "row_count": int(row["row_count"]),
        "columns_json": columns_json,
        "has_target": int(row["has_target"]),
        "target_col": row["target_col"],
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _read_stable_regular_file(path: Path, *, root: Path, label: str) -> bytes:
    _require_regular_path(path, root=root, label=label)
    before = path.lstat()
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise StrategyError(f"{label} could not be read") from exc
    _require_regular_path(path, root=root, label=label)
    after = path.lstat()
    if _stat_identity(before) != _stat_identity(after):
        raise StrategyError(f"{label} changed while read")
    return content


def _require_regular_path(path: Path, *, root: Path, label: str) -> None:
    if not path.is_absolute():
        raise StrategyError(f"{label} path must be absolute")
    declared_root = root.absolute()
    try:
        relative = path.relative_to(declared_root)
    except ValueError as exc:
        raise StrategyError(f"{label} path escapes task storage") from exc
    current = declared_root
    chain = [current]
    for part in relative.parts:
        current = current / part
        chain.append(current)
    try:
        for ancestor in chain[:-1]:
            metadata = ancestor.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise StrategyError(f"{label} path has a symlink ancestor")
            if not stat.S_ISDIR(metadata.st_mode):
                raise StrategyError(f"{label} path ancestor is not a directory")
        leaf_metadata = chain[-1].lstat()
        if stat.S_ISLNK(leaf_metadata.st_mode):
            raise StrategyError(f"{label} path must not be a symlink")
        if not stat.S_ISREG(leaf_metadata.st_mode):
            raise StrategyError(f"{label} path is not a regular file")
        resolved_root = declared_root.resolve(strict=True)
        path.resolve(strict=True).relative_to(resolved_root)
    except StrategyError:
        raise
    except FileNotFoundError as exc:
        raise StrategyError(f"{label} path is not a regular file") from exc
    except OSError as exc:
        raise StrategyError(f"{label} path is unavailable") from exc
    except ValueError as exc:
        raise StrategyError(f"{label} path escapes task storage") from exc


def _strict_cell_selection_from_bytes(value: bytes) -> dict[str, Any]:
    parsed = _strict_json_object_from_bytes(value, "Cross Matrix cell selection")
    try:
        return validate_cross_matrix_cell_selection(parsed)
    except (CrossMatrixCellSelectionError, TypeError, ValueError) as exc:
        raise StrategyError(
            "Cross Matrix cell selection failed strict validation"
        ) from exc


def _strict_json_object_from_bytes(value: bytes, name: str) -> dict[str, Any]:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StrategyError(f"{name} must be strict UTF-8 JSON") from exc
    return _strict_json_object_from_text(text, name)


def _strict_json_object_from_text(value: str, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise StrategyError(f"{name} contains a duplicate JSON key: {key}")
            result[key] = child
        return result

    def reject_constant(constant: str):
        raise StrategyError(f"{name} contains non-finite JSON: {constant}")

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except StrategyError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise StrategyError(f"{name} must be a JSON object")
    return parsed


def _validate_inputs(inputs: object) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise StrategyError(
            "materialize_cross_matrix_cell_selection inputs must be an object"
        )
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyError(
            "materialize_cross_matrix_cell_selection input keys must be strings"
        )
    actual = set(inputs)
    missing = sorted(_REQUIRED_INPUT_FIELDS - actual)
    unexpected = sorted(actual - _INPUT_FIELDS)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(
            "invalid materialize_cross_matrix_cell_selection inputs ("
            + "; ".join(details)
            + ")"
        )
    raw_cell_ids = inputs["cell_ids"]
    if (
        isinstance(raw_cell_ids, str | bytes | bytearray)
        or not isinstance(raw_cell_ids, Sequence)
        or not raw_cell_ids
    ):
        raise StrategyError("cell_ids must be a non-empty array")
    cell_ids = [
        _required_text(item, f"cell_ids[{index}]")
        for index, item in enumerate(raw_cell_ids)
    ]
    reason = inputs.get("selection_reason")
    if reason is not None:
        if not isinstance(reason, str):
            raise StrategyError("selection_reason must be a string or null")
        if len(reason) > MAX_SELECTION_REASON_LENGTH:
            raise StrategyError("selection_reason must be at most 500 characters")
    return {
        "source_artifact_id": _required_text(
            inputs["source_artifact_id"],
            "source_artifact_id",
        ),
        "expected_artifact_content_hash": _required_sha256(
            inputs["expected_artifact_content_hash"],
            "expected_artifact_content_hash",
        ),
        "expected_asset_id": _required_asset_id(
            inputs["expected_asset_id"],
            "expected_asset_id",
        ),
        "expected_asset_hash": _required_sha256(
            inputs["expected_asset_hash"],
            "expected_asset_hash",
        ),
        "expected_candidate_id": _required_candidate_id(
            inputs["expected_candidate_id"],
            "expected_candidate_id",
        ),
        "expected_evidence_hash": _required_sha256(
            inputs["expected_evidence_hash"],
            "expected_evidence_hash",
        ),
        "cell_ids": cell_ids,
        **({"selection_reason": reason} if "selection_reason" in inputs else {}),
    }


def _safe_component(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if Path(normalized).name != normalized or normalized in {".", ".."}:
        raise StrategyError(f"{name} is unsafe for artifact storage")
    return normalized


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StrategyError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise StrategyError(f"{name} must not contain NUL")
    normalized = unicodedata.normalize("NFC", value)
    if value != normalized or value != value.strip():
        raise StrategyError(f"{name} must be canonical text")
    return value


def _required_sha256(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256")
    return normalized


def _required_asset_id(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if _ASSET_ID_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} has an invalid format")
    return normalized


def _required_candidate_id(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if _CANDIDATE_ID_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} has an invalid format")
    return normalized


def _canonical_json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{name} must be a JSON object")
    try:
        normalized = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} must contain finite JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be a JSON object")
    return normalized


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(str(field) for field in actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported " + ", ".join(unexpected))
        raise StrategyError(f"{name} has " + "; ".join(details))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _matches_sha256(value: object, expected: str) -> bool:
    return (
        isinstance(value, str)
        and _HASH_RE.fullmatch(value) is not None
        and hmac.compare_digest(value, expected)
    )


def _require_file_content_hash(path: Path, expected: str, message: str) -> None:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise StrategyError(message) from exc
    if not hmac.compare_digest(actual, expected):
        raise StrategyError(message)


def _stat_identity(value) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


__all__ = [
    "MAX_SELECTION_REASON_LENGTH",
    "SELECTION_PROVENANCE_FIELDS",
    "SOURCE_PROVENANCE_FIELDS",
    "TOOL_SCHEMA_VERSION",
    "VerifiedCrossMatrixCellSelection",
    "VerifiedCrossMatrixSource",
    "canonical_cross_matrix_cell_selection_path",
    "canonical_cross_matrix_source_path",
    "cross_matrix_cell_selection_provenance",
    "load_verified_cross_matrix_cell_selection_artifact",
    "load_verified_cross_matrix_cell_selection_artifact_on_connection",
    "load_verified_cross_matrix_source_artifact",
    "load_verified_cross_matrix_source_artifact_on_connection",
    "run_materialize_cross_matrix_cell_selection",
    "verify_cross_matrix_cell_selection_provenance",
    "verify_cross_matrix_source_provenance",
]
