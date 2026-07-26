"""Governed Tool boundary for Strategy Pool validation/OOT replay evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.files import sha256_file
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
    StrategyPoolDevelopmentExecutionBinding,
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
    require_strategy_candidate_pool_artifact_binding_on_connection,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    ResolvedPoolRequirements,
    hydrate_requirement_fields,
    normalize_pool_requirements,
    pool_requirement_bindings_provenance,
    project_pool_entry_requirements,
    require_resolved_pool_requirements_on_connection,
    resolve_pool_requirements,
    validate_pool_requirement_bindings_provenance,
)
from marvis.packs.strategy.pool_validation import (
    STRATEGY_POOL_VALIDATION_PRODUCER_VERSION,
    build_strategy_pool_validation_evidence,
    canonical_strategy_pool_validation_json,
    validate_strategy_pool_validation_evidence,
)
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
    load_any_strategy_sample_design_v2_artifacts,
    require_strategy_sample_design_v2_artifact_binding_on_connection,
    resolve_strategy_sample_design_v2_source_mode,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
    stable_task_artifact_id,
)


POOL_VALIDATION_TOOL_SCHEMA_VERSION = (
    "strategy.measure-pool-validation-tool.v1"
)
POOL_VALIDATION_ARTIFACT_KIND = "strategy_pool_validation_json"
POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION = (
    "strategy.pool-validation-artifact.v1"
)
POOL_VALIDATION_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION = (
    "strategy.pool-validation-artifact.v2"
)
POOL_VALIDATION_ORIGIN_TOOL = "strategy.measure_strategy_pool_validation"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_POOL_ID_RE = re.compile(r"^strategy-pool-[0-9a-f]{32}$")
_POOL_REVISION_ID_RE = re.compile(
    r"^strategy-pool-revision-[0-9a-f]{32}$"
)
_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "pool_ref",
        "sample_design_ref",
        "partition",
        "population",
        "comparison_mode",
    }
)
_POOL_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_pool_id",
        "expected_revision",
        "expected_revision_id",
        "expected_snapshot_hash",
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
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "content_hash",
        "pool_id",
        "pool_revision",
        "pool_snapshot_hash",
        "partition",
        "population",
        "comparison_mode",
        "lifecycle_stage",
        "validation_status",
        "population_count",
        "labeled_count",
        "unlabeled_count",
        "evidence",
        "warnings",
        "artifact",
        "not_mutated_pool",
        "not_created_strategy",
        "not_adopted",
        "not_promoted",
        "not_deployed",
    }
)
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "format",
        "filename",
        "content_hash",
        "download_url",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "evidence_id",
        "evidence_content_hash",
        "pool_ref",
        "sample_design_ref",
        "dataset_binding",
        "target_binding",
        "field_bindings",
        "partition",
        "population",
        "comparison_mode",
        "lifecycle_stage",
        "validation_status",
    }
)
_PHYSICAL_FIELD_BINDING_FIELDS = frozenset(
    {"month_col", "loan_amount_col", "overdue_amount_col"}
)
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
_POOL_PROVENANCE_REF_FIELDS = _POOL_REF_FIELDS | {
    "pool_id",
    "revision_id",
}
_DATASET_BINDING_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "dataset_source_path",
        "dataset_registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    }
)
_TARGET_BINDING_FIELDS = frozenset(
    {"column", "good_value", "bad_value", "missing_policy"}
)
_POOL_VALIDATION_REF_FIELDS = frozenset(
    {
        "partition",
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_evidence_id",
        "expected_evidence_content_hash",
    }
)
_POOL_VALIDATION_EVIDENCE_ID_RE = re.compile(
    r"^strategy-pool-validation-[0-9a-f]{24}$"
)
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_PROVENANCE_BYTES = 1024 * 1024
_MAX_REGISTRY_PATH_BYTES = 16 * 1024
_MAX_DISCOVERABLE_ARTIFACTS = 64
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class StrategyPoolValidationArtifactBinding:
    """Authenticated independent Pool replay evidence for report writers."""

    task_id: str
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    artifact_provenance: dict[str, Any]
    artifact_provenance_json: str
    evidence: dict[str, Any]
    tasks_root: Path
    db_path: Path


def authenticate_strategy_pool_validation_artifact_record(
    *,
    task_id: str,
    record: Mapping[str, Any],
    evidence: object,
    tasks_root: Path | str,
) -> dict[str, Any]:
    """Authenticate one trusted registry record without current-Pool state.

    This boundary proves only that aggregate evidence is the canonical
    registered output of this Tool.  It deliberately does not decide whether
    the evidence is compatible with the task's current Candidate Pool.
    """

    normalized_task = _text(task_id, "task_id")
    if (
        not isinstance(record, Mapping)
        or set(record) != _TASK_ARTIFACT_RECORD_FIELDS
    ):
        raise StrategyError(
            "Strategy Pool validation artifact record fields are invalid"
        )
    validated_evidence = validate_strategy_pool_validation_evidence(evidence)
    if (
        validated_evidence != evidence
        or validated_evidence["identity"]["task_id"] != normalized_task
    ):
        raise StrategyError(
            "Strategy Pool validation artifact evidence identity changed"
        )
    root = Path(tasks_root).absolute()
    artifact_path_text = _text(
        record["path"],
        "Pool validation artifact record.path",
    )
    if len(artifact_path_text.encode("utf-8")) > _MAX_REGISTRY_PATH_BYTES:
        raise StrategyError(
            "Strategy Pool validation artifact registry path exceeds byte budget"
        )
    artifact_path = Path(artifact_path_text)
    expected_path = (
        root
        / normalized_task
        / "strategy_pool_validations"
        / f"{validated_evidence['evidence_id']}.json"
    )
    if artifact_path != expected_path:
        raise StrategyError(
            "Strategy Pool validation artifact record path is not canonical"
        )
    kind = _text(record["kind"], "Pool validation artifact record.kind")
    origin = _text(
        record["origin_tool"],
        "Pool validation artifact record.origin_tool",
    )
    if (
        record["task_id"] != normalized_task
        or kind != POOL_VALIDATION_ARTIFACT_KIND
        or origin != POOL_VALIDATION_ORIGIN_TOOL
    ):
        raise StrategyError(
            "Strategy Pool validation artifact record identity changed"
        )
    artifact_id = _hash(
        record["id"],
        "Pool validation artifact record.id",
    )
    if artifact_id != stable_task_artifact_id(
        task_id=normalized_task,
        kind=kind,
        path=str(artifact_path),
    ):
        raise StrategyError(
            "Strategy Pool validation artifact record stable id changed"
        )
    artifact_hash = _hash(
        record["content_hash"],
        "Pool validation artifact record.content_hash",
    )
    canonical = canonical_strategy_pool_validation_json(
        validated_evidence
    ).encode("utf-8")
    if not hmac.compare_digest(
        hashlib.sha256(canonical).hexdigest(),
        artifact_hash,
    ):
        raise StrategyError(
            "Strategy Pool validation artifact record content hash changed"
        )
    provenance = _validate_provenance(record["provenance"])
    if provenance != record["provenance"]:
        raise StrategyError(
            "Strategy Pool validation artifact record provenance changed"
        )
    _require_provenance_matches_evidence(
        provenance,
        validated_evidence,
    )
    _text(
        record["created_at"],
        "Pool validation artifact record.created_at",
    )
    raw = _read_regular_nofollow(
        artifact_path,
        root=root,
        expected_content_hash=artifact_hash,
    )
    if raw != canonical:
        raise StrategyError(
            "Strategy Pool validation artifact record canonical bytes changed"
        )
    return validated_evidence


def load_latest_strategy_pool_validation_artifacts(
    runtime,
    *,
    task_id: str,
    candidate_pool: StrategyCandidatePoolArtifactBinding,
    sample_design: StrategySampleDesignV2ArtifactBinding,
) -> tuple[StrategyPoolValidationArtifactBinding, ...]:
    """Select at most one latest compatible validation/OOT artifact.

    Compatibility is platform-owned: callers provide the already authenticated
    current Pool and exact SampleDesign V2 bindings, never artifact references.
    Once a compatible newest row is selected, any registry, provenance, path,
    byte, or embedded-evidence drift fails closed instead of falling back to an
    older result.
    """

    normalized_task = _text(task_id, "task_id")
    if (
        not isinstance(candidate_pool, StrategyCandidatePoolArtifactBinding)
        or not isinstance(
            sample_design,
            StrategySampleDesignV2ArtifactBinding,
        )
        or candidate_pool.task_id != normalized_task
        or sample_design.task_id != normalized_task
    ):
        raise StrategyError(
            "Pool validation discovery requires same-task authenticated "
            "Pool and SampleDesign V2 bindings"
        )
    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    db_path = Path(runtime.settings.db_path).absolute()
    selected: dict[str, StrategyPoolValidationArtifactBinding] = {}
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_strategy_candidate_pool_artifact_binding_on_connection(
                conn,
                candidate_pool,
            )
            require_strategy_sample_design_v2_artifact_binding_on_connection(
                conn,
                sample_design,
            )
            count = conn.execute(
                """
                SELECT COUNT(*) AS total
                  FROM task_artifacts
                 WHERE task_id = ? AND kind = ? AND origin_tool = ?
                """,
                (
                    normalized_task,
                    POOL_VALIDATION_ARTIFACT_KIND,
                    POOL_VALIDATION_ORIGIN_TOOL,
                ),
            ).fetchone()
            if (
                count is None
                or int(count["total"]) > _MAX_DISCOVERABLE_ARTIFACTS
            ):
                raise StrategyError(
                    "Strategy Pool validation artifact history exceeds "
                    "discovery budget"
                )
            rows = conn.execute(
                """
                SELECT CASE
                           WHEN length(CAST(id AS BLOB)) = 64 THEN id
                           ELSE NULL
                       END AS id,
                       length(CAST(path AS BLOB)) AS path_bytes,
                       length(CAST(content_hash AS BLOB)) AS hash_bytes,
                       length(CAST(provenance_json AS BLOB))
                           AS provenance_bytes
                  FROM task_artifacts
                 WHERE task_id = ? AND kind = ? AND origin_tool = ?
                 ORDER BY created_at DESC, id DESC
                """,
                (
                    normalized_task,
                    POOL_VALIDATION_ARTIFACT_KIND,
                    POOL_VALIDATION_ORIGIN_TOOL,
                ),
            ).fetchall()
            for row in rows:
                if len(selected) == 2:
                    break
                if row["id"] is None:
                    raise StrategyError(
                        "Strategy Pool validation artifact registry id changed"
                    )
                _require_bounded_registry_metadata(
                    path_bytes=row["path_bytes"],
                    hash_bytes=row["hash_bytes"],
                )
                provenance_json = _bounded_provenance_json_on_connection(
                    conn,
                    task_id=normalized_task,
                    artifact_id=str(row["id"]),
                    byte_length=row["provenance_bytes"],
                )
                provenance = _provenance_from_json(
                    provenance_json
                )
                partition = provenance["partition"]
                if partition in selected or partition not in {
                    "validation",
                    "oot",
                }:
                    continue
                if not _provenance_matches_current_sources(
                    provenance,
                    pool=candidate_pool,
                    sample=sample_design,
                ):
                    continue
                _require_provenance_matches_pool_requirements(
                    provenance,
                    pool=candidate_pool,
                )
                artifact_row = _bounded_artifact_row_on_connection(
                    conn,
                    task_id=normalized_task,
                    artifact_id=row["id"],
                    path_bytes=row["path_bytes"],
                    hash_bytes=row["hash_bytes"],
                )
                binding = _pool_validation_binding_from_row(
                    artifact_row,
                    provenance=provenance,
                    provenance_json=provenance_json,
                    tasks_root=tasks_root,
                    db_path=db_path,
                )
                _require_pool_validation_compatibility(
                    binding.evidence,
                    pool=candidate_pool,
                    sample=sample_design,
                )
                require_strategy_pool_validation_artifact_binding_on_connection(
                    conn,
                    binding,
                )
                selected[partition] = binding
            require_strategy_candidate_pool_artifact_binding_on_connection(
                conn,
                candidate_pool,
            )
            require_strategy_sample_design_v2_artifact_binding_on_connection(
                conn,
                sample_design,
            )
            conn.commit()
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc
    return tuple(
        selected[partition]
        for partition in ("validation", "oot")
        if partition in selected
    )


def select_latest_strategy_pool_validation_refs(
    runtime,
    *,
    task_id: str,
    candidate_pool: StrategyCandidatePoolArtifactBinding,
    sample_design: StrategySampleDesignV2ArtifactBinding,
) -> tuple[dict[str, Any], ...]:
    """Select immutable platform-owned refs for a report plan/turn."""

    return tuple(
        _pool_validation_artifact_ref(binding)
        for binding in load_latest_strategy_pool_validation_artifacts(
            runtime,
            task_id=task_id,
            candidate_pool=candidate_pool,
            sample_design=sample_design,
        )
    )


def load_strategy_pool_validation_artifacts(
    runtime,
    *,
    task_id: str,
    refs: object,
    candidate_pool: StrategyCandidatePoolArtifactBinding,
    sample_design: StrategySampleDesignV2ArtifactBinding,
) -> tuple[StrategyPoolValidationArtifactBinding, ...]:
    """Load only exact Pool validation refs frozen by the platform plan.

    An empty ref list intentionally remains empty even if newer compatible
    evidence appears later.  This preserves exact retry and prevents execution
    from silently rebinding a plan to a different report source.
    """

    normalized_task = _text(task_id, "task_id")
    normalized_refs = validate_strategy_pool_validation_artifact_refs(refs)
    if (
        not isinstance(candidate_pool, StrategyCandidatePoolArtifactBinding)
        or not isinstance(
            sample_design,
            StrategySampleDesignV2ArtifactBinding,
        )
        or candidate_pool.task_id != normalized_task
        or sample_design.task_id != normalized_task
    ):
        raise StrategyError(
            "Pool validation loading requires same-task authenticated "
            "Pool and SampleDesign V2 bindings"
        )
    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    db_path = Path(runtime.settings.db_path).absolute()
    loaded: list[StrategyPoolValidationArtifactBinding] = []
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_strategy_candidate_pool_artifact_binding_on_connection(
                conn,
                candidate_pool,
            )
            require_strategy_sample_design_v2_artifact_binding_on_connection(
                conn,
                sample_design,
            )
            for ref in normalized_refs:
                row = conn.execute(
                    """
                    SELECT CASE
                               WHEN length(CAST(id AS BLOB)) = 64 THEN id
                               ELSE NULL
                           END AS id,
                           kind = ? AS kind_matches,
                           origin_tool = ? AS origin_matches,
                           length(CAST(path AS BLOB)) AS path_bytes,
                           length(CAST(content_hash AS BLOB)) AS hash_bytes,
                           length(CAST(provenance_json AS BLOB))
                               AS provenance_bytes
                      FROM task_artifacts
                     WHERE task_id = ? AND id = ?
                    """,
                    (
                        POOL_VALIDATION_ARTIFACT_KIND,
                        POOL_VALIDATION_ORIGIN_TOOL,
                        normalized_task,
                        ref["artifact_id"],
                    ),
                ).fetchone()
                if row is None:
                    raise StrategyError(
                        "Strategy Pool validation referenced artifact "
                        "was not found"
                    )
                if (
                    row["id"] is None
                    or row["kind_matches"] != 1
                    or row["origin_matches"] != 1
                ):
                    raise StrategyError(
                        "Strategy Pool validation referenced artifact "
                        "registry binding changed"
                    )
                _require_bounded_registry_metadata(
                    path_bytes=row["path_bytes"],
                    hash_bytes=row["hash_bytes"],
                )
                provenance_json = _bounded_provenance_json_on_connection(
                    conn,
                    task_id=normalized_task,
                    artifact_id=str(row["id"]),
                    byte_length=row["provenance_bytes"],
                )
                provenance = _provenance_from_json(provenance_json)
                _require_provenance_matches_pool_requirements(
                    provenance,
                    pool=candidate_pool,
                )
                artifact_row = _bounded_artifact_row_on_connection(
                    conn,
                    task_id=normalized_task,
                    artifact_id=row["id"],
                    path_bytes=row["path_bytes"],
                    hash_bytes=row["hash_bytes"],
                )
                binding = _pool_validation_binding_from_row(
                    artifact_row,
                    provenance=provenance,
                    provenance_json=provenance_json,
                    tasks_root=tasks_root,
                    db_path=db_path,
                )
                _require_pool_validation_compatibility(
                    binding.evidence,
                    pool=candidate_pool,
                    sample=sample_design,
                )
                expected = _pool_validation_artifact_ref(binding)
                if expected != ref:
                    raise StrategyError(
                        "Strategy Pool validation referenced artifact "
                        "binding changed"
                    )
                require_strategy_pool_validation_artifact_binding_on_connection(
                    conn,
                    binding,
                )
                loaded.append(binding)
            require_strategy_candidate_pool_artifact_binding_on_connection(
                conn,
                candidate_pool,
            )
            require_strategy_sample_design_v2_artifact_binding_on_connection(
                conn,
                sample_design,
            )
            conn.commit()
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc
    return tuple(loaded)


def validate_strategy_pool_validation_artifact_refs(
    value: object,
) -> tuple[dict[str, Any], ...]:
    """Validate and canonicalize zero to two platform-owned report refs."""

    if (
        isinstance(value, str | bytes | bytearray)
        or not isinstance(value, list | tuple)
        or len(value) > 2
    ):
        raise StrategyError(
            "pool_validation_refs must contain at most validation and OOT"
        )
    by_partition: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        obj = _json_object(raw, f"pool_validation_refs[{index}]")
        _exact_fields(
            obj,
            _POOL_VALIDATION_REF_FIELDS,
            f"pool_validation_refs[{index}]",
        )
        partition = _text(
            obj["partition"],
            f"pool_validation_refs[{index}].partition",
        )
        if partition not in {"validation", "oot"}:
            raise StrategyError(
                f"pool_validation_refs[{index}].partition is invalid"
            )
        if partition in by_partition:
            raise StrategyError(
                f"pool_validation_refs duplicates {partition}"
            )
        evidence_id = _text(
            obj["expected_evidence_id"],
            f"pool_validation_refs[{index}].expected_evidence_id",
        )
        if _POOL_VALIDATION_EVIDENCE_ID_RE.fullmatch(evidence_id) is None:
            raise StrategyError(
                "pool_validation_refs expected_evidence_id is not canonical"
            )
        by_partition[partition] = {
            "partition": partition,
            "artifact_id": _hash(
                obj["artifact_id"],
                f"pool_validation_refs[{index}].artifact_id",
            ),
            "expected_artifact_content_hash": _hash(
                obj["expected_artifact_content_hash"],
                "pool_validation_refs"
                f"[{index}].expected_artifact_content_hash",
            ),
            "expected_evidence_id": evidence_id,
            "expected_evidence_content_hash": _hash(
                obj["expected_evidence_content_hash"],
                "pool_validation_refs"
                f"[{index}].expected_evidence_content_hash",
            ),
        }
    return tuple(
        by_partition[partition]
        for partition in ("validation", "oot")
        if partition in by_partition
    )


def require_strategy_pool_validation_artifact_binding_on_connection(
    conn,
    binding: StrategyPoolValidationArtifactBinding,
) -> None:
    """Re-authenticate one Pool validation artifact under a writer lock."""

    if not isinstance(binding, StrategyPoolValidationArtifactBinding):
        raise StrategyError("Strategy Pool validation artifact binding is invalid")
    if not conn.in_transaction:
        raise StrategyError(
            "Strategy Pool validation binding requires a caller-owned transaction"
        )
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != binding.db_path
    ):
        raise StrategyError("Strategy Pool validation binding database changed")
    task_id = _text(binding.task_id, "binding.task_id")
    artifact_id = _hash(binding.artifact_id, "binding.artifact_id")
    artifact_hash = _hash(
        binding.artifact_content_hash,
        "binding.artifact_content_hash",
    )
    evidence = validate_strategy_pool_validation_evidence(binding.evidence)
    if evidence != binding.evidence or evidence["identity"]["task_id"] != task_id:
        raise StrategyError("Strategy Pool validation binding payload changed")
    canonical = canonical_strategy_pool_validation_json(evidence).encode("utf-8")
    if not hmac.compare_digest(
        hashlib.sha256(canonical).hexdigest(),
        artifact_hash,
    ):
        raise StrategyError(
            "Strategy Pool validation binding artifact hash changed"
        )
    expected_path = (
        binding.tasks_root
        / task_id
        / "strategy_pool_validations"
        / f"{evidence['evidence_id']}.json"
    )
    if (
        binding.tasks_root != binding.tasks_root.absolute()
        or binding.artifact_path != expected_path
    ):
        raise StrategyError(
            "Strategy Pool validation artifact governed path changed"
        )
    provenance = _validate_provenance(binding.artifact_provenance)
    _require_provenance_matches_evidence(provenance, evidence)
    provenance_json = _canonical_json(provenance)
    if (
        provenance != binding.artifact_provenance
        or provenance_json != binding.artifact_provenance_json
    ):
        raise StrategyError(
            "Strategy Pool validation binding provenance changed"
        )
    metadata = conn.execute(
        """
        SELECT CASE
                   WHEN length(CAST(id AS BLOB)) = 64 THEN id
                   ELSE NULL
               END AS id,
               kind = ? AS kind_matches,
               origin_tool = ? AS origin_matches,
               length(CAST(path AS BLOB)) AS path_bytes,
               length(CAST(content_hash AS BLOB)) AS hash_bytes,
               length(CAST(provenance_json AS BLOB)) AS provenance_bytes
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (
            POOL_VALIDATION_ARTIFACT_KIND,
            POOL_VALIDATION_ORIGIN_TOOL,
            task_id,
            artifact_id,
        ),
    ).fetchone()
    if metadata is None:
        raise StrategyError(
            "Strategy Pool validation artifact registry row disappeared"
        )
    if (
        metadata["id"] is None
        or metadata["kind_matches"] != 1
        or metadata["origin_matches"] != 1
    ):
        raise StrategyError(
            "Strategy Pool validation artifact registry binding changed"
        )
    _require_bounded_registry_metadata(
        path_bytes=metadata["path_bytes"],
        hash_bytes=metadata["hash_bytes"],
    )
    persisted_provenance_json = _bounded_provenance_json_on_connection(
        conn,
        task_id=task_id,
        artifact_id=artifact_id,
        byte_length=metadata["provenance_bytes"],
    )
    row = _bounded_artifact_row_on_connection(
        conn,
        task_id=task_id,
        artifact_id=artifact_id,
        path_bytes=metadata["path_bytes"],
        hash_bytes=metadata["hash_bytes"],
    )
    expected_row = {
        "id": artifact_id,
        "task_id": task_id,
        "kind": POOL_VALIDATION_ARTIFACT_KIND,
        "path": str(binding.artifact_path),
        "content_hash": artifact_hash,
        "origin_tool": POOL_VALIDATION_ORIGIN_TOOL,
    }
    if any(str(row[field]) != value for field, value in expected_row.items()):
        raise StrategyError(
            "Strategy Pool validation artifact registry binding changed"
        )
    if persisted_provenance_json != provenance_json:
        raise StrategyError(
            "Strategy Pool validation artifact registry provenance changed"
        )
    raw = _read_regular_nofollow(
        binding.artifact_path,
        root=binding.tasks_root,
        expected_content_hash=artifact_hash,
    )
    if raw != canonical:
        raise StrategyError(
            "Strategy Pool validation artifact canonical bytes changed"
        )


def run_measure_strategy_pool_validation(inputs, ctx, runtime) -> dict[str, Any]:
    """Replay one exact current Pool on V2 risk validation/OOT membership."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        pool = _load_pool_binding(
            runtime,
            task_id=task_id,
            request=request,
        )
        sample = _load_sample_design_binding(
            runtime,
            task_id=task_id,
            request=request,
        )
        resolve_strategy_sample_design_v2_source_mode(
            sample.bundle["sample_design"],
            capability="legacy_development",
            consumer="strategy_pool_validation",
        )
        development = bind_strategy_pool_development_execution(runtime, pool)
        resolved_requirements = resolve_pool_requirements(
            runtime,
            task_id=task_id,
            compiled_design={
                "requirements": list(
                    project_pool_entry_requirements(pool.pool["entries"])
                )
            },
            sample_design=sample,
        )
        semantics = _require_independent_sample_contract(
            pool=pool,
            development=development,
            sample=sample,
            partition=request["partition"],
        )
        _require_bindings_under_lock(
            runtime,
            pool=pool,
            sample=sample,
            resolved_requirements=resolved_requirements,
        )
        selected = _read_selected_partition(
            runtime,
            pool=pool,
            sample=sample,
            partition=request["partition"],
            target_col=semantics["target_col"],
            month_col=semantics["month_col"],
            loan_amount_col=semantics["loan_amount_col"],
            overdue_amount_col=semantics["overdue_amount_col"],
            resolved_requirements=resolved_requirements,
        )
        evidence = build_strategy_pool_validation_evidence(
            pool=pool.pool,
            frame=selected,
            pool_artifact_ref={
                "artifact_id": pool.artifact_id,
                "artifact_content_hash": pool.artifact_content_hash,
            },
            sample_design_v2_ref=_sample_design_evidence_ref(
                sample,
                partition=request["partition"],
            ),
            dataset_binding=_dataset_evidence_binding(sample),
            legacy_development_ref=semantics["legacy_development_ref"],
            partition=request["partition"],
            population="risk",
            comparison_mode="absolute",
            target_col=semantics["target_col"],
            target_bad_value=semantics["target_bad_value"],
            month_col=semantics["month_col"],
            loan_amount_col=semantics["loan_amount_col"],
            overdue_amount_col=semantics["overdue_amount_col"],
            development_rows_excluded=True,
        )
        if (
            sha256_file(sample.source_binding.dataset_path)
            != sample.source_binding.dataset_content_hash
        ):
            raise StrategyError(
                "Strategy Pool validation dataset changed during computation"
            )
        return _persist_evidence(
            runtime,
            task_id=task_id,
            request=request,
            pool=pool,
            sample=sample,
            resolved_requirements=resolved_requirements,
            evidence=evidence,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_measure_strategy_pool_validation_tool_output(
    value: object,
    *,
    expected_task_id: str,
    expected_artifact_id: str,
) -> dict[str, Any]:
    """Reconstruct scalars and authenticate the task-artifact binding."""

    obj = _json_object(value, "measure_strategy_pool_validation output")
    trusted_task_id = _text(expected_task_id, "expected_task_id")
    trusted_artifact_id = _hash(
        expected_artifact_id,
        "expected_artifact_id",
    )
    _exact_fields(
        obj,
        _OUTPUT_FIELDS,
        "measure_strategy_pool_validation output",
    )
    evidence = validate_strategy_pool_validation_evidence(obj["evidence"])
    identity = evidence["identity"]
    if identity["task_id"] != trusted_task_id:
        raise StrategyError(
            "measure_strategy_pool_validation output task_id drifted"
        )
    population = evidence["population_metrics"]
    expected = {
        "schema_version": POOL_VALIDATION_TOOL_SCHEMA_VERSION,
        "evidence_id": evidence["evidence_id"],
        "content_hash": evidence["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "partition": evidence["partition"],
        "population": evidence["population"],
        "comparison_mode": evidence["comparison_mode"],
        "lifecycle_stage": evidence["lifecycle"]["stage"],
        "validation_status": evidence["lifecycle"]["validation_status"],
        "population_count": population["population_count"],
        "labeled_count": population["labelled_count"],
        "unlabeled_count": population["unlabelled_count"],
    }
    for field, expected_value in expected.items():
        if obj[field] != expected_value:
            raise StrategyError(
                f"measure_strategy_pool_validation output {field} drifted"
            )
    warnings = [
        str(flag["message"])
        for flag in evidence["red_flags"]
        if flag.get("level") in {"amber", "red"}
    ]
    if obj["warnings"] != warnings:
        raise StrategyError(
            "measure_strategy_pool_validation output warnings drifted"
        )
    for field in (
        "not_mutated_pool",
        "not_created_strategy",
        "not_adopted",
        "not_promoted",
        "not_deployed",
    ):
        if obj[field] is not True:
            raise StrategyError(
                f"measure_strategy_pool_validation output {field} must be true"
            )

    artifact = _json_object(
        obj["artifact"],
        "measure_strategy_pool_validation artifact",
    )
    _exact_fields(
        artifact,
        _OUTPUT_ARTIFACT_FIELDS,
        "measure_strategy_pool_validation artifact",
    )
    artifact_id = _hash(artifact["artifact_id"], "artifact_id")
    if artifact_id != trusted_artifact_id:
        raise StrategyError(
            "measure_strategy_pool_validation artifact_id drifted"
        )
    canonical = canonical_strategy_pool_validation_json(evidence).encode(
        "utf-8"
    )
    expected_artifact = {
        "artifact_id": artifact_id,
        "kind": POOL_VALIDATION_ARTIFACT_KIND,
        "format": "json",
        "filename": f"{evidence['evidence_id']}.json",
        "content_hash": hashlib.sha256(canonical).hexdigest(),
        "download_url": (
            f"/api/tasks/{quote(identity['task_id'], safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        ),
    }
    if artifact != expected_artifact:
        raise StrategyError(
            "measure_strategy_pool_validation artifact binding drifted"
        )
    obj["evidence"] = evidence
    return obj


def _pool_validation_binding_from_row(
    row,
    *,
    provenance: Mapping[str, Any],
    provenance_json: str,
    tasks_root: Path,
    db_path: Path,
) -> StrategyPoolValidationArtifactBinding:
    artifact_id = _hash(row["id"], "Pool validation artifact_id")
    task_id = _text(row["task_id"], "Pool validation task_id")
    artifact_hash = _hash(
        row["content_hash"],
        "Pool validation artifact content_hash",
    )
    if (
        str(row["kind"]) != POOL_VALIDATION_ARTIFACT_KIND
        or str(row["origin_tool"]) != POOL_VALIDATION_ORIGIN_TOOL
    ):
        raise StrategyError(
            "Strategy Pool validation artifact registry binding changed"
        )
    artifact_path = Path(str(row["path"]))
    raw = _read_regular_nofollow(
        artifact_path,
        root=tasks_root,
        expected_content_hash=artifact_hash,
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise StrategyError(
            "Strategy Pool validation artifact JSON is invalid"
        ) from exc
    evidence = validate_strategy_pool_validation_evidence(payload)
    canonical = canonical_strategy_pool_validation_json(evidence).encode("utf-8")
    if raw != canonical:
        raise StrategyError(
            "Strategy Pool validation artifact bytes are not canonical"
        )
    expected_path = (
        tasks_root
        / task_id
        / "strategy_pool_validations"
        / f"{evidence['evidence_id']}.json"
    )
    if artifact_path != expected_path:
        raise StrategyError(
            "Strategy Pool validation artifact path is not canonical"
        )
    normalized_provenance = _validate_provenance(provenance)
    _require_provenance_matches_evidence(
        normalized_provenance,
        evidence,
    )
    canonical_provenance_json = _canonical_json(normalized_provenance)
    if provenance_json != canonical_provenance_json:
        raise StrategyError(
            "Strategy Pool validation artifact provenance is not canonical"
        )
    return StrategyPoolValidationArtifactBinding(
        task_id=task_id,
        artifact_id=artifact_id,
        artifact_path=artifact_path,
        artifact_content_hash=artifact_hash,
        artifact_provenance=normalized_provenance,
        artifact_provenance_json=canonical_provenance_json,
        evidence=evidence,
        tasks_root=tasks_root,
        db_path=db_path,
    )


def _pool_validation_artifact_ref(
    binding: StrategyPoolValidationArtifactBinding,
) -> dict[str, Any]:
    evidence = binding.evidence
    return {
        "partition": evidence["partition"],
        "artifact_id": binding.artifact_id,
        "expected_artifact_content_hash": binding.artifact_content_hash,
        "expected_evidence_id": evidence["evidence_id"],
        "expected_evidence_content_hash": evidence["content_hash"],
    }


def _provenance_from_json(raw: str) -> dict[str, Any]:
    if (
        not isinstance(raw, str)
        or len(raw.encode("utf-8")) > _MAX_PROVENANCE_BYTES
    ):
        raise StrategyError(
            "Strategy Pool validation artifact provenance exceeds byte budget"
        )
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise StrategyError(
            "Strategy Pool validation artifact provenance is invalid"
        ) from exc
    provenance = _validate_provenance(payload)
    if raw != _canonical_json(provenance):
        raise StrategyError(
            "Strategy Pool validation artifact provenance is not canonical"
        )
    return provenance


def _require_provenance_byte_budget(value: Mapping[str, Any]) -> None:
    if len(_canonical_json(value).encode("utf-8")) > _MAX_PROVENANCE_BYTES:
        raise StrategyError(
            "Strategy Pool validation artifact provenance exceeds byte budget"
        )


def _require_bounded_registry_metadata(
    *,
    path_bytes: object,
    hash_bytes: object,
) -> None:
    if (
        isinstance(path_bytes, bool)
        or not isinstance(path_bytes, int)
        or path_bytes < 1
        or path_bytes > _MAX_REGISTRY_PATH_BYTES
    ):
        raise StrategyError(
            "Strategy Pool validation artifact registry path exceeds byte budget"
        )
    if (
        isinstance(hash_bytes, bool)
        or not isinstance(hash_bytes, int)
        or hash_bytes != 64
    ):
        raise StrategyError(
            "Strategy Pool validation artifact registry content hash changed"
        )


def _bounded_artifact_row_on_connection(
    conn,
    *,
    task_id: str,
    artifact_id: str,
    path_bytes: object,
    hash_bytes: object,
):
    _require_bounded_registry_metadata(
        path_bytes=path_bytes,
        hash_bytes=hash_bytes,
    )
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
           AND kind = ? AND origin_tool = ?
           AND length(CAST(id AS BLOB)) = 64
           AND length(CAST(path AS BLOB)) = ?
           AND length(CAST(content_hash AS BLOB)) = 64
        """,
        (
            task_id,
            artifact_id,
            POOL_VALIDATION_ARTIFACT_KIND,
            POOL_VALIDATION_ORIGIN_TOOL,
            path_bytes,
        ),
    ).fetchone()
    if row is None:
        raise StrategyError(
            "Strategy Pool validation artifact registry binding changed"
        )
    return row


def _bounded_provenance_json_on_connection(
    conn,
    *,
    task_id: str,
    artifact_id: str,
    byte_length: object,
) -> str:
    if (
        isinstance(byte_length, bool)
        or not isinstance(byte_length, int)
        or byte_length < 0
        or byte_length > _MAX_PROVENANCE_BYTES
    ):
        raise StrategyError(
            "Strategy Pool validation artifact provenance exceeds byte budget"
        )
    row = conn.execute(
        """
        SELECT provenance_json
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
           AND length(CAST(provenance_json AS BLOB)) = ?
        """,
        (task_id, artifact_id, byte_length),
    ).fetchone()
    if row is None or not isinstance(row["provenance_json"], str):
        raise StrategyError(
            "Strategy Pool validation artifact provenance changed"
        )
    raw = row["provenance_json"]
    if len(raw.encode("utf-8")) != byte_length:
        raise StrategyError(
            "Strategy Pool validation artifact provenance changed"
        )
    return raw


def _provenance_matches_current_sources(
    provenance: Mapping[str, Any],
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
) -> bool:
    return (
        provenance["pool_ref"] == _current_pool_provenance_ref(pool)
        and provenance["sample_design_ref"]
        == _current_sample_design_provenance_ref(sample)
        and provenance["dataset_binding"] == _dataset_evidence_binding(sample)
    )


def _require_provenance_matches_pool_requirements(
    provenance: Mapping[str, Any],
    *,
    pool: StrategyCandidatePoolArtifactBinding,
) -> None:
    compiled_requirements = list(
        normalize_pool_requirements(
            pool.compiled_design["requirements"]
        )
    )
    fields = provenance["field_bindings"]
    if compiled_requirements:
        if (
            provenance["schema_version"]
            != POOL_VALIDATION_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION
            or "requirements" not in fields
        ):
            raise StrategyError(
                "Strategy Pool validation requirements differ from the "
                "current Candidate Pool"
            )
        requirement_bindings = validate_pool_requirement_bindings_provenance(
            fields["requirements"]
        )
        if requirement_bindings["requirements"] != compiled_requirements:
            raise StrategyError(
                "Strategy Pool validation requirements differ from the "
                "current Candidate Pool"
            )
        return
    if (
        provenance["schema_version"] != POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION
        or "requirements" in fields
    ):
        raise StrategyError(
            "Strategy Pool validation requirements differ from the "
            "current Candidate Pool"
        )


def _current_pool_provenance_ref(
    pool: StrategyCandidatePoolArtifactBinding,
) -> dict[str, Any]:
    snapshot = pool.pool
    return {
        "artifact_id": pool.artifact_id,
        "expected_artifact_content_hash": pool.artifact_content_hash,
        "expected_pool_id": snapshot["pool_id"],
        "expected_revision": snapshot["revision"],
        "expected_revision_id": snapshot["revision_id"],
        "expected_snapshot_hash": snapshot["snapshot_hash"],
        "pool_id": snapshot["pool_id"],
        "revision_id": snapshot["revision_id"],
    }


def _current_sample_design_provenance_ref(
    sample: StrategySampleDesignV2ArtifactBinding,
) -> dict[str, Any]:
    design = sample.bundle["sample_design"]
    return {
        "membership_artifact_id": sample.membership_artifact_id,
        "expected_membership_artifact_content_hash": (
            sample.membership_artifact_content_hash
        ),
        "bundle_artifact_id": sample.bundle_artifact_id,
        "expected_bundle_artifact_content_hash": (
            sample.bundle_artifact_content_hash
        ),
        "expected_bundle_id": sample.bundle["bundle_id"],
        "expected_sample_design_id": design["sample_design_id"],
        "expected_sample_design_content_hash": design["content_hash"],
    }


def _require_pool_validation_compatibility(
    evidence: Mapping[str, Any],
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
) -> None:
    identity = evidence["identity"]
    compiled = pool.compiled_design
    expected_identity = {
        "pool_id": pool.pool["pool_id"],
        "task_id": pool.task_id,
        "strategy_type": pool.strategy_type,
        "revision": pool.pool["revision"],
        "revision_id": pool.pool["revision_id"],
        "snapshot_hash": pool.pool["snapshot_hash"],
        "design_hash": compiled["design_hash"],
        "strategy_spec_hash": strategy_spec_hash(compiled["strategy_spec"]),
    }
    sources = evidence["source_bindings"]
    partition = evidence["partition"]
    design = sample.bundle["sample_design"]
    fields = design["sample_semantics"]["field_bindings"]
    expected_target = {
        "column": design["target_selector"]["column"],
        "good_value": design["target_selector"]["good_value"],
        "bad_value": design["target_selector"]["bad_value"],
        "missing_policy": "retain_population_exclude_risk_denominator",
    }
    expected_fields = {
        "month_col": fields["month_field"],
        "loan_amount_col": fields["loan_amount_field"],
        "overdue_amount_col": fields["overdue_amount_field"],
    }
    if (
        identity != expected_identity
        or sources["pool_artifact"]
        != {
            "artifact_id": pool.artifact_id,
            "artifact_content_hash": pool.artifact_content_hash,
        }
        or sources["sample_design_v2"]
        != _sample_design_evidence_ref(sample, partition=partition)
        or sources["dataset"] != _dataset_evidence_binding(sample)
        or sources["target"] != expected_target
        or sources["fields"] != expected_fields
        or sources["development_lineage"]["legacy_development_ref"]
        != design["compatibility"]["legacy_development_ref"]
    ):
        raise StrategyError(
            "Strategy Pool validation evidence is not compatible with the "
            "current Pool, exact SampleDesign V2, and dataset"
        )


def _require_provenance_matches_evidence(
    provenance: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    identity = evidence["identity"]
    sources = evidence["source_bindings"]
    pool_ref = provenance["pool_ref"]
    sample_ref = provenance["sample_design_ref"]
    sample_source = sources["sample_design_v2"]
    expected_pool = {
        "artifact_id": sources["pool_artifact"]["artifact_id"],
        "expected_artifact_content_hash": sources["pool_artifact"][
            "artifact_content_hash"
        ],
        "expected_pool_id": identity["pool_id"],
        "expected_revision": identity["revision"],
        "expected_revision_id": identity["revision_id"],
        "expected_snapshot_hash": identity["snapshot_hash"],
        "pool_id": identity["pool_id"],
        "revision_id": identity["revision_id"],
    }
    expected_sample = {
        "membership_artifact_id": sample_source["membership_artifact_id"],
        "expected_membership_artifact_content_hash": sample_source[
            "membership_artifact_content_hash"
        ],
        "bundle_artifact_id": sample_source["bundle_artifact_id"],
        "expected_bundle_artifact_content_hash": sample_source[
            "bundle_artifact_content_hash"
        ],
        "expected_bundle_id": sample_source["bundle_id"],
        "expected_sample_design_id": sample_source["sample_design_id"],
        "expected_sample_design_content_hash": sample_source[
            "sample_design_content_hash"
        ],
    }
    physical_fields = {
        key: provenance["field_bindings"][key]
        for key in _PHYSICAL_FIELD_BINDING_FIELDS
    }
    if (
        provenance["task_id"] != identity["task_id"]
        or provenance["evidence_id"] != evidence["evidence_id"]
        or provenance["evidence_content_hash"] != evidence["content_hash"]
        or pool_ref != expected_pool
        or sample_ref != expected_sample
        or provenance["dataset_binding"] != sources["dataset"]
        or provenance["target_binding"] != sources["target"]
        or physical_fields != sources["fields"]
        or provenance["partition"] != evidence["partition"]
        or provenance["lifecycle_stage"] != evidence["lifecycle"]["stage"]
        or provenance["validation_status"]
        != evidence["lifecycle"]["validation_status"]
    ):
        raise StrategyError(
            "Strategy Pool validation artifact provenance does not match "
            "embedded evidence"
        )


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _json_object(value, "measure_strategy_pool_validation inputs")
    _exact_fields(
        obj,
        _INPUT_FIELDS,
        "measure_strategy_pool_validation inputs",
    )
    strategy_type = _text(obj["strategy_type"], "strategy_type")
    if strategy_type not in {"approval", "reject"}:
        raise StrategyError(
            "Strategy Pool validation supports approval/reject only"
        )
    partition = _text(obj["partition"], "partition")
    if partition not in {"validation", "oot"}:
        raise StrategyError("partition must be validation or oot")
    if obj["population"] != "risk":
        raise StrategyError("population must be risk")
    if obj["comparison_mode"] != "absolute":
        raise StrategyError("comparison_mode must be absolute")
    return {
        "strategy_type": strategy_type,
        "pool_ref": _validate_pool_ref(obj["pool_ref"]),
        "sample_design_ref": _validate_sample_design_ref(
            obj["sample_design_ref"]
        ),
        "partition": partition,
        "population": "risk",
        "comparison_mode": "absolute",
    }


def _validate_pool_ref(value: object) -> dict[str, Any]:
    obj = _json_object(value, "pool_ref")
    _exact_fields(obj, _POOL_REF_FIELDS, "pool_ref")
    pool_id = _text(obj["expected_pool_id"], "pool_ref.expected_pool_id")
    if _POOL_ID_RE.fullmatch(pool_id) is None:
        raise StrategyError("pool_ref.expected_pool_id is invalid")
    revision_id = _text(
        obj["expected_revision_id"],
        "pool_ref.expected_revision_id",
    )
    if _POOL_REVISION_ID_RE.fullmatch(revision_id) is None:
        raise StrategyError("pool_ref.expected_revision_id is invalid")
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "pool_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "pool_ref.expected_artifact_content_hash",
        ),
        "expected_pool_id": pool_id,
        "expected_revision": _positive_int(
            obj["expected_revision"],
            "pool_ref.expected_revision",
        ),
        "expected_revision_id": revision_id,
        "expected_snapshot_hash": _hash(
            obj["expected_snapshot_hash"],
            "pool_ref.expected_snapshot_hash",
        ),
    }


def _validate_sample_design_ref(value: object) -> dict[str, str]:
    obj = _json_object(value, "sample_design_ref")
    _exact_fields(
        obj,
        _SAMPLE_DESIGN_REF_FIELDS,
        "sample_design_ref",
    )
    result: dict[str, str] = {}
    for field in (
        "membership_artifact_id",
        "expected_membership_artifact_content_hash",
        "bundle_artifact_id",
        "expected_bundle_artifact_content_hash",
        "expected_sample_design_content_hash",
    ):
        result[field] = _hash(obj[field], f"sample_design_ref.{field}")
    for field in ("expected_bundle_id", "expected_sample_design_id"):
        result[field] = _text(obj[field], f"sample_design_ref.{field}")
    return result


def _load_pool_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> StrategyCandidatePoolArtifactBinding:
    ref = request["pool_ref"]
    binding = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        strategy_type=request["strategy_type"],
        expected_pool_revision=ref["expected_revision"],
        expected_pool_snapshot_hash=ref["expected_snapshot_hash"],
        expected_artifact_id=ref["artifact_id"],
        expected_artifact_content_hash=ref[
            "expected_artifact_content_hash"
        ],
    )
    if (
        binding.pool["pool_id"] != ref["expected_pool_id"]
        or binding.pool["revision_id"] != ref["expected_revision_id"]
    ):
        raise StrategyError("current Strategy Pool identity changed")
    if not binding.pool["entries"]:
        raise StrategyError("cannot validate an empty Strategy Pool")
    return binding


def _load_sample_design_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> StrategySampleDesignV2ArtifactBinding:
    ref = request["sample_design_ref"]
    return load_any_strategy_sample_design_v2_artifacts(
        runtime,
        task_id=task_id,
        membership_artifact_id=ref["membership_artifact_id"],
        expected_membership_artifact_content_hash=ref[
            "expected_membership_artifact_content_hash"
        ],
        bundle_artifact_id=ref["bundle_artifact_id"],
        expected_bundle_artifact_content_hash=ref[
            "expected_bundle_artifact_content_hash"
        ],
        expected_bundle_id=ref["expected_bundle_id"],
        expected_sample_design_id=ref["expected_sample_design_id"],
        expected_sample_design_content_hash=ref[
            "expected_sample_design_content_hash"
        ],
    )


def _require_independent_sample_contract(
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    development: StrategyPoolDevelopmentExecutionBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    partition: str,
) -> dict[str, Any]:
    design = sample.bundle["sample_design"]
    target = design["target_selector"]
    if target["status"] != "resolved":
        raise StrategyError(
            "Strategy Pool validation requires a resolved V2 target selector"
        )
    if design["sample_semantics"]["scope"] != "strategy_development":
        raise StrategyError(
            "Strategy Pool validation requires governed strategy scope"
        )
    risk = next(
        item
        for item in sample.bundle["populations"]
        if item["role"] == "risk"
    )
    if risk["maturity_evidence"]["status"] != "confirmed_matured":
        raise StrategyError(
            "Strategy Pool validation requires confirmed_matured risk outcomes"
        )
    partition_count = sample.membership["header"]["counts"]["risk"][
        partition
    ]
    if partition_count == 0:
        raise StrategyError(f"Strategy Pool {partition} partition is empty")

    legacy_ref = StrategySampleDesignRef.from_value(
        design["compatibility"]["legacy_development_ref"]
    )
    if legacy_ref != sample.source_binding.legacy.reference:
        raise StrategyError(
            "StrategySampleDesign V2 legacy development mapping changed"
        )
    _require_pool_development_contract(
        pool=pool,
        development=development,
        sample=sample,
        legacy_ref=legacy_ref,
        target_col=target["column"],
    )
    if (
        sample.source_binding.legacy.target_col != target["column"]
        or sample.source_binding.legacy.target_bad_value
        != target["bad_value"]
    ):
        raise StrategyError(
            "StrategySampleDesign V2 target polarity changed from legacy lineage"
        )
    fields = design["sample_semantics"]["field_bindings"]
    return {
        "legacy_development_ref": legacy_ref.to_ref_dict(),
        "target_col": target["column"],
        "target_bad_value": target["bad_value"],
        "month_col": fields["month_field"],
        "loan_amount_col": fields["loan_amount_field"],
        "overdue_amount_col": fields["overdue_amount_field"],
    }


def _require_pool_development_contract(
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    development: StrategyPoolDevelopmentExecutionBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    legacy_ref: StrategySampleDesignRef,
    target_col: str,
) -> None:
    if (
        development.pool is not pool
        or development.task_id != pool.task_id
        or development.sample_design.reference != legacy_ref
        or development.target_col != target_col
    ):
        raise StrategyError(
            "Strategy Pool development binding does not match "
            "StrategySampleDesign V2"
        )
    dataset = development.dataset
    source = sample.source_binding
    if (
        dataset.task_id != source.task_id
        or dataset.dataset_id != source.dataset_id
        or dataset.source_path != source.dataset_source_path
        or dataset.path != source.dataset_path
        or not hmac.compare_digest(
            dataset.content_hash,
            source.dataset_content_hash,
        )
        or dataset.columns != source.columns
        or dataset.row_count != source.row_count
    ):
        raise StrategyError(
            "Strategy Pool development dataset does not match "
            "StrategySampleDesign V2"
        )
    bound_v2 = development.sample_design_v2
    if bound_v2 is not None and _sample_design_v2_identity(
        bound_v2
    ) != _sample_design_v2_identity(sample):
        raise StrategyError(
            "Strategy Pool development SampleDesign V2 binding changed"
        )


def _sample_design_v2_identity(
    binding: StrategySampleDesignV2ArtifactBinding,
) -> tuple[str, ...]:
    return (
        binding.task_id,
        binding.membership_artifact_id,
        binding.membership_artifact_content_hash,
        binding.bundle_artifact_id,
        binding.bundle_artifact_content_hash,
        str(binding.bundle["bundle_id"]),
        str(binding.bundle["sample_design"]["sample_design_id"]),
        str(binding.bundle["sample_design"]["content_hash"]),
    )


def _require_bindings_under_lock(
    runtime,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    resolved_requirements: ResolvedPoolRequirements,
) -> None:
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_candidate_pool_artifact_binding_on_connection(
            conn,
            pool,
        )
        require_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            sample,
        )
        require_resolved_pool_requirements_on_connection(
            conn,
            resolved_requirements,
        )
        conn.commit()


def _require_bindings_on_connection(
    conn,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    resolved_requirements: ResolvedPoolRequirements,
) -> None:
    require_strategy_candidate_pool_artifact_binding_on_connection(conn, pool)
    require_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        sample,
    )
    require_resolved_pool_requirements_on_connection(
        conn,
        resolved_requirements,
    )


def _read_selected_partition(
    runtime,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    partition: str,
    target_col: str,
    month_col: str | None,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
    resolved_requirements: ResolvedPoolRequirements,
) -> pd.DataFrame:
    path = sample.source_binding.dataset_path
    _require_dataset_path(
        path,
        root=Path(runtime.settings.datasets_dir).absolute(),
    )
    fields = _expression_fields(pool.compiled_design["strategy_spec"])
    fields.add(target_col)
    fields.update(
        value
        for value in (month_col, loan_amount_col, overdue_amount_col)
        if value is not None
    )
    virtual_fields = set(resolved_requirements.virtual_fields)
    physical_fields = fields - virtual_fields
    unknown = sorted(physical_fields - set(sample.source_binding.columns))
    if unknown:
        raise StrategyError(
            "Strategy Pool rules reference missing V2 dataset columns: "
            + ", ".join(unknown)
        )
    frame = _read_authenticated_parquet_snapshot(
        path,
        root=Path(runtime.settings.datasets_dir).absolute(),
        expected_content_hash=sample.source_binding.dataset_content_hash,
        columns=sorted(physical_fields),
    )
    if not isinstance(frame, pd.DataFrame) or len(frame) != (
        sample.source_binding.row_count
    ):
        raise StrategyError(
            "Strategy Pool validation analysis universe row count changed"
        )
    # Score vectors use raw row ordinals. A persisted Parquet index is not a
    # business column and must not alter the authenticated row universe.
    frame = frame.reset_index(drop=True)
    frame = hydrate_requirement_fields(
        frame,
        resolved=resolved_requirements,
    )
    mask = np.asarray(
        sample.membership["masks"][f"risk/{partition}"],
        dtype=np.bool_,
    )
    development = np.asarray(
        sample.membership["masks"]["risk/development"],
        dtype=np.bool_,
    )
    if len(mask) != len(frame) or len(development) != len(frame):
        raise StrategyError(
            "StrategySampleDesign V2 membership row order changed"
        )
    if bool(np.any(mask & development)):
        raise StrategyError(
            "Strategy Pool validation membership overlaps development rows"
        )
    expected_count = sample.membership["header"]["counts"]["risk"][
        partition
    ]
    if int(np.count_nonzero(mask)) != expected_count:
        raise StrategyError(
            "Strategy Pool validation membership count changed"
        )
    if expected_count == 0:
        raise StrategyError(f"Strategy Pool {partition} partition is empty")
    selected = frame.loc[
        pd.Series(mask, index=frame.index, dtype=bool)
    ].reset_index(drop=True)
    if len(selected) != expected_count:
        raise StrategyError(
            "Strategy Pool validation selected partition count changed"
        )
    return selected


def _read_authenticated_parquet_snapshot(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
    columns: list[str],
) -> pd.DataFrame:
    """Read only bytes copied from one authenticated, retained source fd."""

    _require_dataset_path(path, root=root)
    source_fd = -1
    snapshot = None
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise StrategyError(
                "Strategy Pool validation dataset must be a regular file"
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
            or _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(after_open)
            or _stable_file_stat(before) != _stable_file_stat(opened)
            or _stable_file_stat(opened) != _stable_file_stat(after_open)
        ):
            raise StrategyError(
                "Strategy Pool validation dataset changed while opening"
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
        source_after_copy = os.fstat(source_fd)
        if (
            _stable_file_stat(source_after_copy)
            != _stable_file_stat(opened)
            or copied != int(opened.st_size)
            or not hmac.compare_digest(
                digest.hexdigest(),
                expected_content_hash,
            )
        ):
            raise StrategyError(
                "Strategy Pool validation dataset bytes changed before replay"
            )

        snapshot_stat = os.fstat(snapshot.fileno())
        if int(snapshot_stat.st_size) != copied:
            raise StrategyError(
                "Strategy Pool validation private snapshot is incomplete"
            )
        snapshot.seek(0)
        frame = pd.read_parquet(snapshot, columns=columns)
        snapshot_after_read = os.fstat(snapshot.fileno())
        current = os.lstat(path)
        if (
            _stable_file_stat(snapshot_after_read)
            != _stable_file_stat(snapshot_stat)
            or _stable_file_stat(os.fstat(source_fd))
            != _stable_file_stat(opened)
            or stat.S_ISLNK(current.st_mode)
            or _stable_file_stat(current) != _stable_file_stat(opened)
        ):
            raise StrategyError(
                "Strategy Pool validation dataset changed during replay"
            )
        return frame
    except StrategyError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StrategyError(
            "Strategy Pool validation dataset could not be read"
        ) from exc
    finally:
        if snapshot is not None:
            snapshot.close()
        if source_fd >= 0:
            os.close(source_fd)


def _file_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
    )


def _stable_file_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _expression_fields(value: object) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, Mapping):
        field = value.get("field")
        if isinstance(field, str):
            fields.add(field)
        for item in value.values():
            fields.update(_expression_fields(item))
    elif isinstance(value, list | tuple):
        for item in value:
            fields.update(_expression_fields(item))
    return fields


def _sample_design_evidence_ref(
    sample: StrategySampleDesignV2ArtifactBinding,
    *,
    partition: str,
) -> dict[str, Any]:
    header = sample.membership["header"]
    bundle = sample.bundle
    design = bundle["sample_design"]
    return {
        "membership_artifact_id": sample.membership_artifact_id,
        "membership_artifact_content_hash": (
            sample.membership_artifact_content_hash
        ),
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "bundle_artifact_id": sample.bundle_artifact_id,
        "bundle_artifact_content_hash": (
            sample.bundle_artifact_content_hash
        ),
        "bundle_id": bundle["bundle_id"],
        "bundle_content_hash": bundle["content_hash"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "partition_key": f"risk/{partition}",
        "partition_count": header["counts"]["risk"][partition],
        "analysis_universe_row_count": header["row_count"],
    }


def _dataset_evidence_binding(
    sample: StrategySampleDesignV2ArtifactBinding,
) -> dict[str, Any]:
    source = sample.source_binding
    return {
        "task_id": source.task_id,
        "dataset_id": source.dataset_id,
        "dataset_content_hash": source.dataset_content_hash,
        "dataset_source_path": source.dataset_source_path,
        "dataset_registry_metadata_hash": (
            source.dataset_registry_metadata_hash
        ),
        "workspace_revision": source.workspace_revision,
        "workspace_generation": source.workspace_generation,
        "semantic_mapping_hash": source.semantic_mapping_hash,
    }


def _persist_evidence(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    resolved_requirements: ResolvedPoolRequirements,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = canonical_strategy_pool_validation_json(evidence).encode(
        "utf-8"
    )
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    sources = evidence["source_bindings"]
    provenance = {
        "schema_version": (
            POOL_VALIDATION_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION
            if resolved_requirements.requirements
            else POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": STRATEGY_POOL_VALIDATION_PRODUCER_VERSION,
        "task_id": task_id,
        "evidence_id": evidence["evidence_id"],
        "evidence_content_hash": evidence["content_hash"],
        "pool_ref": {
            **dict(request["pool_ref"]),
            "pool_id": evidence["identity"]["pool_id"],
            "revision_id": evidence["identity"]["revision_id"],
        },
        "sample_design_ref": dict(request["sample_design_ref"]),
        "dataset_binding": dict(sources["dataset"]),
        "target_binding": dict(sources["target"]),
        "field_bindings": _provenance_field_bindings(
            sources["fields"],
            resolved_requirements=resolved_requirements,
        ),
        "partition": evidence["partition"],
        "population": "risk",
        "comparison_mode": "absolute",
        "lifecycle_stage": evidence["lifecycle"]["stage"],
        "validation_status": "independent_evidence",
    }
    _validate_provenance(provenance)
    _require_provenance_byte_budget(provenance)
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = out_dir / f"{evidence['evidence_id']}.json"
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError(
            "Strategy Pool validation artifact could not be staged"
        ) from exc

    db_committed = False
    rollback_under_lock = False
    reused = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
                    resolved_requirements=resolved_requirements,
                )
                metadata = _select_artifact_row(
                    conn,
                    task_id=task_id,
                    path=final_path,
                )
                if metadata is not None:
                    if (
                        metadata["id"] is None
                        or metadata["origin_matches"] != 1
                    ):
                        raise StrategyError(
                            "existing Strategy Pool validation artifact "
                            "registry row changed"
                        )
                    _require_bounded_registry_metadata(
                        path_bytes=metadata["path_bytes"],
                        hash_bytes=metadata["hash_bytes"],
                    )
                    persisted_provenance_json = (
                        _bounded_provenance_json_on_connection(
                            conn,
                            task_id=task_id,
                            artifact_id=metadata["id"],
                            byte_length=metadata["provenance_bytes"],
                        )
                    )
                    row = _bounded_artifact_row_on_connection(
                        conn,
                        task_id=task_id,
                        artifact_id=metadata["id"],
                        path_bytes=metadata["path_bytes"],
                        hash_bytes=metadata["hash_bytes"],
                    )
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        path=final_path,
                        canonical=canonical,
                        content_hash=artifact_hash,
                        provenance=provenance,
                        persisted_provenance_json=(
                            persisted_provenance_json
                        ),
                    )
                    uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        _require_exact_file(
                            final_path,
                            root=Path(
                                runtime.settings.tasks_dir
                            ).absolute(),
                            canonical=canonical,
                            content_hash=artifact_hash,
                        )
                        uow.rollback()
                        reused = True
                    else:
                        uow.promote_all()
                        _require_exact_file(
                            final_path,
                            root=Path(
                                runtime.settings.tasks_dir
                            ).absolute(),
                            canonical=canonical,
                            content_hash=artifact_hash,
                        )
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
                    resolved_requirements=resolved_requirements,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=POOL_VALIDATION_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=POOL_VALIDATION_ORIGIN_TOOL,
                    provenance=provenance,
                )
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
                    resolved_requirements=resolved_requirements,
                )
                conn.commit()
                db_committed = True
            except Exception:
                rollback_under_lock = True
                uow.rollback()
                raise
        if not reused:
            uow.commit()
    except Exception:
        if not db_committed and not rollback_under_lock:
            uow.rollback()
        raise
    return validate_measure_strategy_pool_validation_tool_output(
        _tool_output(
            evidence=evidence,
            record=record,
            task_id=task_id,
        ),
        expected_task_id=task_id,
        expected_artifact_id=record["id"],
    )


def _tool_output(
    *,
    evidence: Mapping[str, Any],
    record: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    identity = evidence["identity"]
    population = evidence["population_metrics"]
    artifact_id = str(record["id"])
    return {
        "schema_version": POOL_VALIDATION_TOOL_SCHEMA_VERSION,
        "evidence_id": evidence["evidence_id"],
        "content_hash": evidence["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "partition": evidence["partition"],
        "population": "risk",
        "comparison_mode": "absolute",
        "lifecycle_stage": evidence["lifecycle"]["stage"],
        "validation_status": "independent_evidence",
        "population_count": population["population_count"],
        "labeled_count": population["labelled_count"],
        "unlabeled_count": population["unlabelled_count"],
        "evidence": dict(evidence),
        "warnings": [
            str(flag["message"])
            for flag in evidence["red_flags"]
            if flag.get("level") in {"amber", "red"}
        ],
        "artifact": {
            "artifact_id": artifact_id,
            "kind": POOL_VALIDATION_ARTIFACT_KIND,
            "format": "json",
            "filename": Path(str(record["path"])).name,
            "content_hash": str(record["content_hash"]),
            "download_url": (
                f"/api/tasks/{quote(task_id, safe='')}"
                f"/task-artifacts/{quote(artifact_id, safe='')}/download"
            ),
        },
        "not_mutated_pool": True,
        "not_created_strategy": True,
        "not_adopted": True,
        "not_promoted": True,
        "not_deployed": True,
    }


def _provenance_field_bindings(
    physical: Mapping[str, Any],
    *,
    resolved_requirements: ResolvedPoolRequirements,
) -> dict[str, Any]:
    result = dict(physical)
    if not resolved_requirements.requirements:
        return result
    result["requirements"] = pool_requirement_bindings_provenance(
        resolved_requirements
    )
    return result


def _validate_provenance(value: object) -> dict[str, Any]:
    obj = _json_object(value, "Strategy Pool validation provenance")
    _exact_fields(
        obj,
        _PROVENANCE_FIELDS,
        "Strategy Pool validation provenance",
    )
    schema_version = obj["schema_version"]
    if schema_version not in {
        POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION,
        POOL_VALIDATION_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION,
    }:
        raise StrategyError(
            "Strategy Pool validation provenance schema_version is invalid"
        )
    if obj["producer_version"] != STRATEGY_POOL_VALIDATION_PRODUCER_VERSION:
        raise StrategyError(
            "Strategy Pool validation provenance producer_version is invalid"
        )
    for field in ("task_id", "evidence_id", "partition"):
        _text(obj[field], f"validation provenance.{field}")
    if obj["partition"] not in {"validation", "oot"}:
        raise StrategyError(
            "Strategy Pool validation provenance partition is invalid"
        )
    _hash(
        obj["evidence_content_hash"],
        "validation provenance.evidence_content_hash",
    )
    if obj["population"] != "risk" or obj["comparison_mode"] != "absolute":
        raise StrategyError(
            "Strategy Pool validation provenance controls changed"
        )
    if (
        obj["lifecycle_stage"] != obj["partition"]
        or obj["validation_status"] != "independent_evidence"
    ):
        raise StrategyError(
            "Strategy Pool validation provenance lifecycle changed"
        )
    pool_ref = _json_object(
        obj["pool_ref"],
        "Strategy Pool validation provenance.pool_ref",
    )
    _exact_fields(
        pool_ref,
        _POOL_PROVENANCE_REF_FIELDS,
        "Strategy Pool validation provenance.pool_ref",
    )
    for field in (
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_snapshot_hash",
    ):
        _hash(pool_ref[field], f"validation provenance.pool_ref.{field}")
    for field in ("expected_pool_id", "pool_id"):
        pool_id = _text(
            pool_ref[field],
            f"validation provenance.pool_ref.{field}",
        )
        if _POOL_ID_RE.fullmatch(pool_id) is None:
            raise StrategyError(
                f"validation provenance.pool_ref.{field} is invalid"
            )
    for field in ("expected_revision_id", "revision_id"):
        revision_id = _text(
            pool_ref[field],
            f"validation provenance.pool_ref.{field}",
        )
        if _POOL_REVISION_ID_RE.fullmatch(revision_id) is None:
            raise StrategyError(
                f"validation provenance.pool_ref.{field} is invalid"
            )
    _positive_int(
        pool_ref["expected_revision"],
        "validation provenance.pool_ref.expected_revision",
    )
    _validate_sample_design_ref(obj["sample_design_ref"])
    dataset = _json_object(
        obj["dataset_binding"],
        "Strategy Pool validation provenance.dataset_binding",
    )
    _exact_fields(
        dataset,
        _DATASET_BINDING_FIELDS,
        "Strategy Pool validation provenance.dataset_binding",
    )
    for field in (
        "dataset_content_hash",
        "dataset_registry_metadata_hash",
        "semantic_mapping_hash",
    ):
        _hash(dataset[field], f"validation provenance.dataset_binding.{field}")
    for field in ("task_id", "dataset_id", "dataset_source_path"):
        _text(dataset[field], f"validation provenance.dataset_binding.{field}")
    for field in ("workspace_revision", "workspace_generation"):
        value = dataset[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StrategyError(
                f"validation provenance.dataset_binding.{field} "
                "must be a non-negative integer"
            )
    target = _json_object(
        obj["target_binding"],
        "Strategy Pool validation provenance.target_binding",
    )
    _exact_fields(
        target,
        _TARGET_BINDING_FIELDS,
        "Strategy Pool validation provenance.target_binding",
    )
    _text(target["column"], "validation provenance.target_binding.column")
    if (
        {target["good_value"], target["bad_value"]} != {0, 1}
        or target["missing_policy"]
        != "retain_population_exclude_risk_denominator"
    ):
        raise StrategyError(
            "Strategy Pool validation provenance target binding changed"
        )
    fields = _json_object(
        obj["field_bindings"],
        "Strategy Pool validation provenance.field_bindings",
    )
    if (
        schema_version == POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION
        and set(fields) == _PHYSICAL_FIELD_BINDING_FIELDS
    ):
        pass
    elif (
        schema_version
        == POOL_VALIDATION_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION
        and set(fields)
        == _PHYSICAL_FIELD_BINDING_FIELDS | {"requirements"}
    ):
        validate_pool_requirement_bindings_provenance(fields["requirements"])
    else:
        raise StrategyError(
            "Strategy Pool validation provenance field bindings changed"
        )
    return obj


def _prepare_output_directory(
    tasks_dir: Path | str,
    *,
    task_id: str,
) -> Path:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise StrategyError("task_id cannot escape task storage")
    root = Path(tasks_dir).absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StrategyError(
            "task artifact root must be a regular directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / task_id
    if task_dir.exists() and (
        task_dir.is_symlink() or not task_dir.is_dir()
    ):
        raise StrategyError(
            "task artifact directory must be a regular directory"
        )
    task_dir.mkdir(exist_ok=True)
    if (
        task_dir.is_symlink()
        or task_dir.resolve(strict=True).parent
        != root.resolve(strict=True)
    ):
        raise StrategyError(
            "Strategy Pool validation task directory escaped storage"
        )
    out_dir = task_dir / "strategy_pool_validations"
    if out_dir.exists() and (
        out_dir.is_symlink() or not out_dir.is_dir()
    ):
        raise StrategyError(
            "Strategy Pool validation output path must be a regular directory"
        )
    out_dir.mkdir(exist_ok=True)
    if (
        out_dir.is_symlink()
        or out_dir.resolve(strict=True).parent
        != task_dir.resolve(strict=True)
    ):
        raise StrategyError(
            "Strategy Pool validation output directory escaped storage"
        )
    return out_dir


def _select_artifact_row(conn, *, task_id: str, path: Path):
    return conn.execute(
        """
        SELECT CASE
                   WHEN length(CAST(id AS BLOB)) = 64 THEN id
                   ELSE NULL
               END AS id,
               origin_tool = ? AS origin_matches,
               length(CAST(path AS BLOB)) AS path_bytes,
               length(CAST(content_hash AS BLOB)) AS hash_bytes,
               length(CAST(provenance_json AS BLOB)) AS provenance_bytes
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (
            POOL_VALIDATION_ORIGIN_TOOL,
            task_id,
            POOL_VALIDATION_ARTIFACT_KIND,
            str(path),
        ),
    ).fetchone()


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
    persisted_provenance_json: str,
) -> None:
    expected = {
        "task_id": task_id,
        "kind": POOL_VALIDATION_ARTIFACT_KIND,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": POOL_VALIDATION_ORIGIN_TOOL,
    }
    if any(str(row[field]) != value for field, value in expected.items()):
        raise StrategyError(
            "existing Strategy Pool validation artifact registry row changed"
        )
    provenance_json = json.dumps(
        dict(provenance),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if persisted_provenance_json != provenance_json:
        raise StrategyError(
            "existing Strategy Pool validation provenance changed"
        )
    _require_exact_file(
        path,
        root=path.parents[2],
        canonical=canonical,
        content_hash=content_hash,
    )


def _require_exact_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
) -> None:
    raw = _read_regular_nofollow(
        path,
        root=root,
        expected_content_hash=content_hash,
    )
    if raw != canonical:
        raise StrategyError(
            "Strategy Pool validation artifact bytes changed"
        )


def _read_regular_nofollow(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise StrategyError(
            "Strategy Pool validation artifact must be a regular file"
        )
    current = path.parent
    resolved_root = root.absolute()
    while current != resolved_root:
        if current.is_symlink():
            raise StrategyError(
                "Strategy Pool validation artifact path traverses a symlink"
            )
        if current == current.parent:
            break
        current = current.parent
    try:
        path.resolve(strict=True).relative_to(
            resolved_root.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "Strategy Pool validation artifact escaped task storage"
        ) from exc

    descriptor = -1
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    total = 0
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StrategyError(
                "Strategy Pool validation artifact must be a regular file"
            )
        if before.st_size < 0 or before.st_size > _MAX_ARTIFACT_BYTES:
            raise StrategyError(
                "Strategy Pool validation artifact exceeds byte budget"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_ARTIFACT_BYTES:
                raise StrategyError(
                    "Strategy Pool validation artifact exceeds byte budget"
                )
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise StrategyError(
                "Strategy Pool validation artifact changed while read"
            )
        live = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(live.st_mode)
            or (live.st_dev, live.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise StrategyError(
                "Strategy Pool validation artifact path changed while read"
            )
    except OSError as exc:
        raise StrategyError(
            "Strategy Pool validation artifact is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or not hmac.compare_digest(
            digest.hexdigest(),
            expected_content_hash,
        )
    ):
        raise StrategyError(
            "Strategy Pool validation artifact bytes changed"
        )
    return raw


def _require_dataset_path(path: Path, *, root: Path) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise StrategyError(
            "Strategy Pool validation dataset must be a regular file"
        )
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "Strategy Pool validation dataset escaped dataset storage"
        ) from exc


def _json_object(value: object, name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError(f"{name} must be canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    if normalized != value:
        raise StrategyError(f"{name} contains non-canonical JSON values")
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise StrategyError(f"JSON contains non-finite constant: {value}")


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        unexpected = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        details: list[str] = []
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        raise StrategyError(f"{name} has invalid fields ({'; '.join(details)})")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StrategyError(f"{name} must be a positive integer")
    return value


__all__ = [
    "POOL_VALIDATION_ARTIFACT_KIND",
    "POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION",
    "POOL_VALIDATION_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION",
    "POOL_VALIDATION_ORIGIN_TOOL",
    "POOL_VALIDATION_TOOL_SCHEMA_VERSION",
    "StrategyPoolValidationArtifactBinding",
    "authenticate_strategy_pool_validation_artifact_record",
    "load_latest_strategy_pool_validation_artifacts",
    "load_strategy_pool_validation_artifacts",
    "require_strategy_pool_validation_artifact_binding_on_connection",
    "run_measure_strategy_pool_validation",
    "select_latest_strategy_pool_validation_refs",
    "validate_strategy_pool_validation_artifact_refs",
    "validate_measure_strategy_pool_validation_tool_output",
]
