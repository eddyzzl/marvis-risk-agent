"""Governed Tool boundary for cross-partition Strategy Pool stability.

The Tool consumes one exact, authenticated ImpactCube reference.  It never
discovers a latest artifact, accepts raw cube JSON, or rereads source data.
The derived aggregate-only stability document, its task-artifact row, and one
measurement audit are published as one fail-closed unit of work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube_binding import (
    StrategyImpactCubeArtifactBinding,
    load_strategy_impact_cube_artifact,
    require_strategy_impact_cube_artifact_binding_on_connection,
)
from marvis.packs.strategy.pool_stability import (
    MAX_POOL_STABILITY_JSON_BYTES,
    POOL_STABILITY_PRODUCER_VERSION,
    build_strategy_pool_stability,
    canonical_strategy_pool_stability_json,
    validate_strategy_pool_stability,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
    stable_task_artifact_id,
)


POOL_STABILITY_TOOL_SCHEMA_VERSION = (
    "strategy.measure-pool-stability-tool.v1"
)
POOL_STABILITY_ARTIFACT_KIND = "strategy_pool_stability_json"
POOL_STABILITY_ARTIFACT_SCHEMA_VERSION = (
    "strategy.pool-stability-task-artifact.v1"
)
POOL_STABILITY_ORIGIN_TOOL = "strategy.measure_strategy_pool_stability"
POOL_STABILITY_PRODUCER_RUN_SCHEMA_VERSION = (
    "strategy.pool-stability-producer-run.v1"
)
POOL_STABILITY_MEASUREMENT_AUDIT_KIND = (
    "strategy.pool_stability.measure"
)

_INPUT_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_cube_id",
        "expected_cube_content_hash",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "stability_id",
        "content_hash",
        "pool_id",
        "pool_revision",
        "pool_snapshot_hash",
        "strategy_type",
        "baseline_partition",
        "comparison_partitions",
        "max_psi",
        "stability",
        "warnings",
        "artifact",
        "producer_run_ref",
        "read_only",
        "effect_validation",
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
_PRODUCER_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "task_id",
        "input_hash",
        "request",
        "tool_ref",
        "stability_ref",
        "artifact_ref",
        "content_hash",
    }
)
_PRODUCER_TOOL_REF_FIELDS = frozenset(
    {
        "plugin",
        "tool",
        "origin_tool",
        "tool_schema_version",
        "producer_version",
    }
)
_PRODUCER_STABILITY_REF_FIELDS = frozenset(
    {"stability_id", "content_hash"}
)
_PRODUCER_ARTIFACT_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "filename",
        "content_hash",
        "origin_tool",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "stability_id",
        "stability_content_hash",
        "impact_cube_ref",
        "pool_identity",
        "sample_design_v2",
        "dataset_binding",
        "baseline_partition",
        "comparison_partitions",
        "lifecycle",
        "producer_run",
    }
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
_TASK_ARTIFACT_ROW_FIELDS = (
    "id",
    "task_id",
    "kind",
    "path",
    "content_hash",
    "origin_tool",
    "provenance_json",
    "created_at",
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_STABILITY_ID_RE = re.compile(
    r"^strategy-pool-stability-[0-9a-f]{24}$"
)
_RUN_ID_RE = re.compile(
    r"^strategy-pool-stability-run-[0-9a-f]{24}$"
)
_BOUNDARY_ERRORS = (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class StrategyPoolStabilityArtifactBinding:
    """Authenticated Pool-stability evidence and its exact ImpactCube source."""

    task_id: str
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    artifact_provenance: dict[str, Any]
    artifact_provenance_json: str
    stability: dict[str, Any]
    impact_cube: StrategyImpactCubeArtifactBinding
    tasks_root: Path
    db_path: Path

    @property
    def path(self) -> Path:
        return self.artifact_path

    @property
    def provenance(self) -> dict[str, Any]:
        return self.artifact_provenance

    @property
    def provenance_json(self) -> str:
        return self.artifact_provenance_json


def run_measure_strategy_pool_stability(
    inputs,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Derive and atomically publish Pool stability from one exact ImpactCube."""

    try:
        request = _validate_inputs(inputs)
        task_id = _safe_id(ctx.task_id, "task_id")
        source = load_strategy_impact_cube_artifact(
            runtime,
            task_id=task_id,
            **request,
        )
        stability = build_strategy_pool_stability(
            impact_cube=source.cube,
            impact_cube_ref=request,
        )
        return _persist_stability(
            runtime,
            task_id=task_id,
            request=request,
            source=source,
            stability=stability,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_measure_strategy_pool_stability_tool_output(
    value: object,
    *,
    trusted_task_id: str,
    trusted_artifact_id: str,
    trusted_artifact_content_hash: str,
    trusted_producer_run_id: str,
    trusted_producer_run_content_hash: str,
) -> dict[str, Any]:
    """Strictly validate the deterministic terminal Tool envelope."""

    obj = _json_object(value, "Pool stability Tool output")
    _exact_fields(obj, _OUTPUT_FIELDS, "Pool stability Tool output")
    if obj["schema_version"] != POOL_STABILITY_TOOL_SCHEMA_VERSION:
        raise StrategyError("Pool stability Tool schema_version changed")
    stability = validate_strategy_pool_stability(obj["stability"])
    identity = stability["identity"]
    expected_scalars = {
        "stability_id": stability["stability_id"],
        "content_hash": stability["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "strategy_type": identity["strategy_type"],
        "baseline_partition": stability["baseline_partition"],
        "comparison_partitions": stability["comparison_partitions"],
        "max_psi": _max_psi(stability),
        "warnings": _warnings(stability),
        "read_only": True,
        "effect_validation": False,
        "not_mutated_pool": True,
        "not_created_strategy": True,
        "not_adopted": True,
        "not_promoted": True,
        "not_deployed": True,
    }
    for field, expected in expected_scalars.items():
        if obj[field] != expected:
            raise StrategyError(
                f"Pool stability Tool output {field} changed"
            )
    artifact = _json_object(
        obj["artifact"],
        "Pool stability Tool output artifact",
    )
    _exact_fields(
        artifact,
        _OUTPUT_ARTIFACT_FIELDS,
        "Pool stability Tool output artifact",
    )
    canonical = canonical_strategy_pool_stability_json(stability).encode(
        "utf-8"
    )
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    expected_filename = f"{stability['stability_id']}.json"
    if (
        artifact["kind"] != POOL_STABILITY_ARTIFACT_KIND
        or artifact["format"] != "json"
        or artifact["filename"] != expected_filename
        or artifact["content_hash"] != artifact_hash
        or not isinstance(artifact["download_url"], str)
        or not artifact["download_url"]
    ):
        raise StrategyError(
            "Pool stability Tool output artifact binding changed"
        )
    artifact_id = _hash(
        artifact["artifact_id"],
        "Pool stability Tool output artifact_id",
    )
    expected_download_url = (
        f"/api/tasks/{quote(identity['task_id'], safe='')}"
        f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        f"?expected_content_hash={artifact_hash}"
    )
    if artifact["download_url"] != expected_download_url:
        raise StrategyError(
            "Pool stability Tool output download_url changed"
        )
    producer_ref = _json_object(
        obj["producer_run_ref"],
        "Pool stability Tool output producer_run_ref",
    )
    _exact_fields(
        producer_ref,
        frozenset({"kind", "ref_id", "content_hash"}),
        "Pool stability Tool output producer_run_ref",
    )
    producer_run_id = _text(
        producer_ref["ref_id"],
        "Pool stability Tool output producer_run_ref.ref_id",
    )
    if (
        producer_ref["kind"] != "tool_run"
        or _RUN_ID_RE.fullmatch(producer_run_id) is None
    ):
        raise StrategyError(
            "Pool stability Tool producer_run_ref changed"
        )
    producer_run_hash = _hash(
        producer_ref["content_hash"],
        "Pool stability Tool producer_run_ref.content_hash",
    )
    task_id = _safe_id(trusted_task_id, "trusted_task_id")
    if (
        identity["task_id"] != task_id
        or artifact_id != _hash(
            trusted_artifact_id,
            "trusted_artifact_id",
        )
        or artifact_hash
        != _hash(
            trusted_artifact_content_hash,
            "trusted_artifact_content_hash",
        )
        or producer_run_id
        != _text(
            trusted_producer_run_id,
            "trusted_producer_run_id",
        )
        or producer_run_hash
        != _hash(
            trusted_producer_run_content_hash,
            "trusted_producer_run_content_hash",
        )
    ):
        raise StrategyError(
            "Pool stability trusted Tool output binding changed"
        )
    return {
        **expected_scalars,
        "schema_version": POOL_STABILITY_TOOL_SCHEMA_VERSION,
        "stability": stability,
        "artifact": artifact,
        "producer_run_ref": producer_ref,
    }


def authenticate_strategy_pool_stability_artifact_record(
    *,
    task_id: str,
    record: Mapping[str, Any],
    stability: Mapping[str, Any],
    tasks_root: str | Path,
) -> dict[str, Any]:
    """Authenticate trusted registry context for deterministic rendering."""

    normalized_task = _safe_id(task_id, "task_id")
    normalized = validate_strategy_pool_stability(stability)
    root = Path(tasks_root)
    if not root.is_absolute():
        raise StrategyError("Pool stability trusted task root changed")
    if normalized["identity"]["task_id"] != normalized_task:
        raise StrategyError("Pool stability trusted task binding changed")
    expected_path = _expected_artifact_path(
        root,
        task_id=normalized_task,
        stability_id=normalized["stability_id"],
    )
    canonical = canonical_strategy_pool_stability_json(normalized).encode(
        "utf-8"
    )
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    artifact_id = stable_task_artifact_id(
        task_id=normalized_task,
        kind=POOL_STABILITY_ARTIFACT_KIND,
        path=str(expected_path),
    )
    if (
        not isinstance(record, Mapping)
        or set(record) != _TASK_ARTIFACT_RECORD_FIELDS
        or record["id"] != artifact_id
        or record["task_id"] != normalized_task
        or record["kind"] != POOL_STABILITY_ARTIFACT_KIND
        or record["path"] != str(expected_path)
        or record["content_hash"] != artifact_hash
        or record["origin_tool"] != POOL_STABILITY_ORIGIN_TOOL
        or not isinstance(record["created_at"], str)
        or not record["created_at"]
    ):
        raise StrategyError(
            "Pool stability trusted registry record changed"
        )
    provenance = _validate_provenance(
        record["provenance"],
        task_id=normalized_task,
        stability=normalized,
        artifact_id=artifact_id,
        artifact_filename=expected_path.name,
        artifact_content_hash=artifact_hash,
    )
    return {
        "stability": normalized,
        "artifact_id": artifact_id,
        "artifact_content_hash": artifact_hash,
        "producer_run": provenance["producer_run"],
    }


def load_strategy_pool_stability_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_stability_id: str,
    expected_stability_content_hash: str,
) -> StrategyPoolStabilityArtifactBinding:
    """Load one exact task-owned Pool-stability artifact for a downstream Tool."""

    try:
        normalized_task = _safe_id(task_id, "task_id")
        normalized_artifact_id = _hash(artifact_id, "artifact_id")
        artifact_hash = _hash(
            expected_artifact_content_hash,
            "expected_artifact_content_hash",
        )
        stability_id = _stability_id(
            expected_stability_id,
            "expected_stability_id",
        )
        stability_hash = _hash(
            expected_stability_content_hash,
            "expected_stability_content_hash",
        )
        tasks_root = Path(runtime.settings.tasks_dir).absolute()
        db_path = Path(runtime.settings.db_path).absolute()
        expected_path = _expected_artifact_path(
            tasks_root,
            task_id=normalized_task,
            stability_id=stability_id,
        )
        record = runtime.task_artifacts.get_for_task(
            normalized_task,
            normalized_artifact_id,
        )
        if (
            not isinstance(record, Mapping)
            or set(record) != _TASK_ARTIFACT_RECORD_FIELDS
            or record["id"] != normalized_artifact_id
            or record["task_id"] != normalized_task
            or record["kind"] != POOL_STABILITY_ARTIFACT_KIND
            or record["origin_tool"] != POOL_STABILITY_ORIGIN_TOOL
            or not isinstance(record["path"], str)
            or record["path"] != str(expected_path)
            or not isinstance(record["provenance"], Mapping)
            or not isinstance(record["created_at"], str)
            or not record["created_at"]
            or not hmac.compare_digest(
                str(record["content_hash"]),
                artifact_hash,
            )
            or normalized_artifact_id
            != stable_task_artifact_id(
                task_id=normalized_task,
                kind=POOL_STABILITY_ARTIFACT_KIND,
                path=str(expected_path),
            )
        ):
            raise StrategyError(
                "Pool stability artifact registry binding changed"
            )
        raw = _read_regular_nofollow(
            expected_path,
            root=tasks_root,
            expected_content_hash=artifact_hash,
        )
        stability = _stability_from_bytes(raw)
        canonical = canonical_strategy_pool_stability_json(stability).encode(
            "utf-8"
        )
        if raw != canonical:
            raise StrategyError(
                "Pool stability artifact bytes are not canonical"
            )
        if (
            stability["stability_id"] != stability_id
            or not hmac.compare_digest(
                stability["content_hash"],
                stability_hash,
            )
            or not hmac.compare_digest(
                hashlib.sha256(canonical).hexdigest(),
                artifact_hash,
            )
            or stability["identity"]["task_id"] != normalized_task
        ):
            raise StrategyError(
                "Pool stability artifact embedded identity changed"
            )
        source_ref = _validate_inputs(
            stability["source_bindings"]["impact_cube"]
        )
        source = load_strategy_impact_cube_artifact(
            runtime,
            task_id=normalized_task,
            **source_ref,
        )
        rebound = validate_strategy_pool_stability(
            stability,
            impact_cube=source.cube,
            impact_cube_ref=source_ref,
        )
        if rebound != stability:
            raise StrategyError(
                "Pool stability artifact source binding changed"
            )
        provenance = _validate_provenance(
            record["provenance"],
            task_id=normalized_task,
            stability=stability,
            artifact_id=normalized_artifact_id,
            artifact_filename=expected_path.name,
            artifact_content_hash=artifact_hash,
        )
        binding = StrategyPoolStabilityArtifactBinding(
            task_id=normalized_task,
            artifact_id=normalized_artifact_id,
            artifact_path=expected_path,
            artifact_content_hash=artifact_hash,
            artifact_provenance=provenance,
            artifact_provenance_json=_canonical_json(provenance),
            stability=stability,
            impact_cube=source,
            tasks_root=tasks_root,
            db_path=db_path,
        )
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_strategy_pool_stability_artifact_binding_on_connection(
                conn,
                binding,
            )
            conn.commit()
        return binding
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def _require_impact_cube_source_binding_on_connection(
    conn,
    binding: StrategyImpactCubeArtifactBinding,
) -> None:
    """Authenticate the exact source through the shared main-only boundary."""

    require_strategy_impact_cube_artifact_binding_on_connection(
        conn,
        binding,
    )


def require_strategy_pool_stability_artifact_binding_on_connection(
    conn,
    binding: StrategyPoolStabilityArtifactBinding,
) -> None:
    """Re-authenticate stability, source, registry, bytes and audit under lock."""

    if not isinstance(binding, StrategyPoolStabilityArtifactBinding):
        raise StrategyError("Pool stability artifact binding is invalid")
    _require_binding_connection(conn, db_path=binding.db_path)
    task_id = _safe_id(binding.task_id, "binding.task_id")
    artifact_id = _hash(binding.artifact_id, "binding.artifact_id")
    artifact_hash = _hash(
        binding.artifact_content_hash,
        "binding.artifact_content_hash",
    )
    stability = validate_strategy_pool_stability(binding.stability)
    source_ref = _validate_inputs(
        stability["source_bindings"]["impact_cube"]
    )
    _require_impact_cube_source_binding_on_connection(
        conn,
        binding.impact_cube,
    )
    if (
        binding.impact_cube.task_id != task_id
        or binding.impact_cube.artifact_id != source_ref["artifact_id"]
        or binding.impact_cube.artifact_content_hash
        != source_ref["expected_artifact_content_hash"]
        or binding.impact_cube.cube["cube_id"]
        != source_ref["expected_cube_id"]
        or binding.impact_cube.cube["content_hash"]
        != source_ref["expected_cube_content_hash"]
    ):
        raise StrategyError(
            "Pool stability embedded ImpactCube binding changed"
        )
    stability = validate_strategy_pool_stability(
        stability,
        impact_cube=binding.impact_cube.cube,
        impact_cube_ref=source_ref,
    )
    if stability != binding.stability:
        raise StrategyError("Pool stability binding payload changed")
    expected_path = _expected_artifact_path(
        binding.tasks_root,
        task_id=task_id,
        stability_id=stability["stability_id"],
    )
    if (
        binding.tasks_root != binding.tasks_root.absolute()
        or binding.artifact_path != expected_path
        or artifact_id
        != stable_task_artifact_id(
            task_id=task_id,
            kind=POOL_STABILITY_ARTIFACT_KIND,
            path=str(expected_path),
        )
    ):
        raise StrategyError(
            "Pool stability artifact governed path changed"
        )
    canonical = canonical_strategy_pool_stability_json(stability).encode(
        "utf-8"
    )
    if not hmac.compare_digest(
        hashlib.sha256(canonical).hexdigest(),
        artifact_hash,
    ):
        raise StrategyError("Pool stability binding artifact hash changed")
    provenance = _validate_provenance(
        binding.artifact_provenance,
        task_id=task_id,
        stability=stability,
        artifact_id=artifact_id,
        artifact_filename=expected_path.name,
        artifact_content_hash=artifact_hash,
    )
    if _canonical_json(provenance) != binding.artifact_provenance_json:
        raise StrategyError("Pool stability provenance binding changed")
    _require_artifact_row_on_connection(
        conn,
        task_id=task_id,
        artifact_id=artifact_id,
        path=expected_path,
        content_hash=artifact_hash,
        provenance_json=binding.artifact_provenance_json,
    )
    raw = _read_regular_nofollow(
        expected_path,
        root=binding.tasks_root,
        expected_content_hash=artifact_hash,
    )
    if raw != canonical:
        raise StrategyError("Pool stability artifact bytes changed")
    require_pool_stability_measurement_audit_on_connection(
        conn,
        provenance["producer_run"],
    )


def require_pool_stability_measurement_audit_on_connection(
    conn,
    producer_run: Mapping[str, Any],
) -> None:
    """Require the unique succeeded measurement audit for one producer run."""

    if not conn.in_transaction:
        raise StrategyError(
            "Pool stability measurement audit requires a transaction"
        )
    run = _validate_producer_run(producer_run)
    rows = conn.execute(
        """
        SELECT actor, inputs_hash, outcome, detail_json
          FROM main.audit
         WHERE kind = ? AND target_ref = ?
         ORDER BY at, id
        """,
        (POOL_STABILITY_MEASUREMENT_AUDIT_KIND, run["run_id"]),
    ).fetchall()
    if len(rows) != 1:
        state = "missing" if not rows else "duplicated"
        raise StrategyError(
            f"Pool stability measurement audit is {state}"
        )
    row = rows[0]
    try:
        detail = json.loads(
            str(row["detail_json"]),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StrategyError(
            "Pool stability measurement audit detail is invalid"
        ) from exc
    if (
        str(row["actor"]) != "system"
        or str(row["inputs_hash"]) != run["input_hash"]
        or str(row["outcome"]) != "succeeded"
        or detail != {"producer_run": run}
    ):
        raise StrategyError(
            "Pool stability measurement audit binding changed"
        )


def _persist_stability(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    source: StrategyImpactCubeArtifactBinding,
    stability: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_strategy_pool_stability(
        stability,
        impact_cube=source.cube,
        impact_cube_ref=request,
    )
    canonical = canonical_strategy_pool_stability_json(normalized).encode(
        "utf-8"
    )
    if len(canonical) > MAX_POOL_STABILITY_JSON_BYTES:
        raise StrategyError("Pool stability artifact exceeds byte budget")
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = out_dir / f"{normalized['stability_id']}.json"
    artifact_id = stable_task_artifact_id(
        task_id=task_id,
        kind=POOL_STABILITY_ARTIFACT_KIND,
        path=str(final_path),
    )
    producer_run = _build_producer_run(
        task_id=task_id,
        request=request,
        stability=normalized,
        artifact_id=artifact_id,
        artifact_filename=final_path.name,
        artifact_content_hash=artifact_hash,
    )
    provenance = _artifact_provenance(
        task_id=task_id,
        stability=normalized,
        producer_run=producer_run,
    )
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError(
            "Pool stability artifact could not be staged"
        ) from exc

    db_committed = False
    rollback_under_lock = False
    reused = False
    retained_fd = -1
    retained_identity: tuple[int, ...] | None = None
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_impact_cube_source_binding_on_connection(
                    conn,
                    source,
                )
                row = _select_artifact_row(
                    conn,
                    task_id=task_id,
                    path=final_path,
                )
                artifact_row_existed = row is not None
                if row is not None:
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        path=final_path,
                        canonical=canonical,
                        content_hash=artifact_hash,
                        provenance=provenance,
                        root=Path(runtime.settings.tasks_dir).absolute(),
                    )
                    uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "Pool stability path exists without a registry row"
                        )
                    uow.promote_all()
                    _require_exact_file(
                        final_path,
                        root=Path(runtime.settings.tasks_dir).absolute(),
                        canonical=canonical,
                        content_hash=artifact_hash,
                    )
                _require_impact_cube_source_binding_on_connection(
                    conn,
                    source,
                )
                retained_fd, retained_identity = _open_retained_exact_file(
                    final_path,
                    root=Path(runtime.settings.tasks_dir).absolute(),
                    canonical=canonical,
                    content_hash=artifact_hash,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=POOL_STABILITY_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=POOL_STABILITY_ORIGIN_TOOL,
                    provenance=provenance,
                )
                if str(record["id"]) != artifact_id:
                    raise StrategyError(
                        "Pool stability artifact stable identity changed"
                    )
                _write_or_require_measurement_audit(
                    conn,
                    runtime=runtime,
                    producer_run=producer_run,
                    artifact_row_existed=artifact_row_existed,
                )
                _require_impact_cube_source_binding_on_connection(
                    conn,
                    source,
                )
                _require_artifact_row_on_connection(
                    conn,
                    task_id=task_id,
                    artifact_id=artifact_id,
                    path=final_path,
                    content_hash=artifact_hash,
                    provenance_json=_canonical_json(provenance),
                )
                require_pool_stability_measurement_audit_on_connection(
                    conn,
                    producer_run,
                )
                _require_retained_exact_file(
                    retained_fd,
                    retained_identity=retained_identity,
                    path=final_path,
                    canonical=canonical,
                    content_hash=artifact_hash,
                )
                published_binding = StrategyPoolStabilityArtifactBinding(
                    task_id=task_id,
                    artifact_id=artifact_id,
                    artifact_path=final_path,
                    artifact_content_hash=artifact_hash,
                    artifact_provenance=provenance,
                    artifact_provenance_json=_canonical_json(provenance),
                    stability=normalized,
                    impact_cube=source,
                    tasks_root=Path(
                        runtime.settings.tasks_dir
                    ).absolute(),
                    db_path=Path(runtime.settings.db_path).absolute(),
                )
                conn.commit()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    require_strategy_pool_stability_artifact_binding_on_connection(
                        conn,
                        published_binding,
                    )
                    _require_retained_exact_file(
                        retained_fd,
                        retained_identity=retained_identity,
                        path=final_path,
                        canonical=canonical,
                        content_hash=artifact_hash,
                    )
                    conn.commit()
                except Exception as exc:
                    if conn.in_transaction:
                        conn.rollback()
                    if not artifact_row_existed:
                        _compensate_new_stability_publication(
                            conn,
                            task_id=task_id,
                            artifact_id=artifact_id,
                            artifact_path=final_path,
                            producer_run=producer_run,
                            db_path=Path(
                                runtime.settings.db_path
                            ).absolute(),
                        )
                    raise StrategyError(
                        "Pool stability post-commit authentication failed"
                    ) from exc
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
    finally:
        if retained_fd >= 0:
            os.close(retained_fd)
    return validate_measure_strategy_pool_stability_tool_output(
        _tool_output(
            stability=normalized,
            task_id=task_id,
            record=record,
            producer_run=producer_run,
        ),
        trusted_task_id=task_id,
        trusted_artifact_id=str(record["id"]),
        trusted_artifact_content_hash=str(record["content_hash"]),
        trusted_producer_run_id=producer_run["run_id"],
        trusted_producer_run_content_hash=producer_run["content_hash"],
    )


def _compensate_new_stability_publication(
    conn,
    *,
    task_id: str,
    artifact_id: str,
    artifact_path: Path,
    producer_run: Mapping[str, Any],
    db_path: Path,
) -> None:
    """Remove only this attempt's newly committed registry and audit rows."""

    run = _validate_producer_run(producer_run)
    conn.execute("BEGIN IMMEDIATE")
    _require_binding_connection(conn, db_path=db_path)
    conn.execute(
        """
        DELETE FROM main.audit
         WHERE kind = ? AND target_ref = ?
        """,
        (POOL_STABILITY_MEASUREMENT_AUDIT_KIND, run["run_id"]),
    )
    conn.execute(
        """
        DELETE FROM main.task_artifacts
         WHERE id = ? AND task_id = ? AND kind = ? AND path = ?
        """,
        (
            artifact_id,
            task_id,
            POOL_STABILITY_ARTIFACT_KIND,
            str(artifact_path),
        ),
    )
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        """
        SELECT 1
          FROM main.task_artifacts
         WHERE id = ?
        """,
        (artifact_id,),
    ).fetchone()
    audits = conn.execute(
        """
        SELECT 1
          FROM main.audit
         WHERE kind = ? AND target_ref = ?
        """,
        (POOL_STABILITY_MEASUREMENT_AUDIT_KIND, run["run_id"]),
    ).fetchall()
    if row is not None or audits:
        raise StrategyError(
            "Pool stability post-commit compensation failed"
        )
    conn.commit()


def _validate_inputs(value: object) -> dict[str, str]:
    obj = _json_object(value, "measure_strategy_pool_stability inputs")
    _exact_fields(
        obj,
        _INPUT_FIELDS,
        "measure_strategy_pool_stability inputs",
    )
    return {
        field: _hash(obj[field], field)
        for field in (
            "artifact_id",
            "expected_artifact_content_hash",
            "expected_cube_content_hash",
        )
    } | {
        "expected_cube_id": _cube_id(
            obj["expected_cube_id"],
            "expected_cube_id",
        )
    }


def _artifact_provenance(
    *,
    task_id: str,
    stability: Mapping[str, Any],
    producer_run: Mapping[str, Any],
) -> dict[str, Any]:
    source = stability["source_bindings"]
    provenance = {
        "schema_version": POOL_STABILITY_ARTIFACT_SCHEMA_VERSION,
        "producer_version": POOL_STABILITY_PRODUCER_VERSION,
        "task_id": task_id,
        "stability_id": stability["stability_id"],
        "stability_content_hash": stability["content_hash"],
        "impact_cube_ref": dict(source["impact_cube"]),
        "pool_identity": dict(stability["identity"]),
        "sample_design_v2": dict(source["sample_design_v2"]),
        "dataset_binding": dict(source["dataset"]),
        "baseline_partition": stability["baseline_partition"],
        "comparison_partitions": list(
            stability["comparison_partitions"]
        ),
        "lifecycle": dict(stability["lifecycle"]),
        "producer_run": dict(producer_run),
    }
    return _validate_provenance(
        provenance,
        task_id=task_id,
        stability=stability,
        artifact_id=producer_run["artifact_ref"]["artifact_id"],
        artifact_filename=producer_run["artifact_ref"]["filename"],
        artifact_content_hash=producer_run["artifact_ref"]["content_hash"],
    )


def _validate_provenance(
    value: object,
    *,
    task_id: str,
    stability: Mapping[str, Any],
    artifact_id: str,
    artifact_filename: str,
    artifact_content_hash: str,
) -> dict[str, Any]:
    obj = _json_object(value, "Pool stability artifact provenance")
    _exact_fields(
        obj,
        _PROVENANCE_FIELDS,
        "Pool stability artifact provenance",
    )
    source = stability["source_bindings"]
    expected = {
        "schema_version": POOL_STABILITY_ARTIFACT_SCHEMA_VERSION,
        "producer_version": POOL_STABILITY_PRODUCER_VERSION,
        "task_id": task_id,
        "stability_id": stability["stability_id"],
        "stability_content_hash": stability["content_hash"],
        "impact_cube_ref": source["impact_cube"],
        "pool_identity": stability["identity"],
        "sample_design_v2": source["sample_design_v2"],
        "dataset_binding": source["dataset"],
        "baseline_partition": stability["baseline_partition"],
        "comparison_partitions": stability["comparison_partitions"],
        "lifecycle": stability["lifecycle"],
    }
    for field, expected_value in expected.items():
        if obj[field] != expected_value:
            raise StrategyError(
                f"Pool stability artifact provenance {field} changed"
            )
    run = _validate_producer_run(
        obj["producer_run"],
        expected_task_id=task_id,
        expected_request=source["impact_cube"],
        expected_stability_id=stability["stability_id"],
        expected_stability_content_hash=stability["content_hash"],
        expected_artifact_id=artifact_id,
        expected_artifact_filename=artifact_filename,
        expected_artifact_content_hash=artifact_content_hash,
    )
    return {**expected, "producer_run": run}


def _build_producer_run(
    *,
    task_id: str,
    request: Mapping[str, Any],
    stability: Mapping[str, Any],
    artifact_id: str,
    artifact_filename: str,
    artifact_content_hash: str,
) -> dict[str, Any]:
    body = {
        "schema_version": POOL_STABILITY_PRODUCER_RUN_SCHEMA_VERSION,
        "task_id": task_id,
        "input_hash": hashlib.sha256(
            _canonical_json(
                {
                    "task_id": task_id,
                    "request": request,
                    "producer": _producer_tool_ref(),
                }
            ).encode("utf-8")
        ).hexdigest(),
        "request": dict(request),
        "tool_ref": _producer_tool_ref(),
        "stability_ref": {
            "stability_id": stability["stability_id"],
            "content_hash": stability["content_hash"],
        },
        "artifact_ref": {
            "artifact_id": artifact_id,
            "kind": POOL_STABILITY_ARTIFACT_KIND,
            "filename": artifact_filename,
            "content_hash": artifact_content_hash,
            "origin_tool": POOL_STABILITY_ORIGIN_TOOL,
        },
    }
    run_id = (
        "strategy-pool-stability-run-"
        + hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:24]
    )
    without_hash = {**body, "run_id": run_id}
    run = {
        **without_hash,
        "content_hash": hashlib.sha256(
            _canonical_json(without_hash).encode("utf-8")
        ).hexdigest(),
    }
    return _validate_producer_run(run)


def _producer_tool_ref() -> dict[str, str]:
    return {
        "plugin": "strategy",
        "tool": "measure_strategy_pool_stability",
        "origin_tool": POOL_STABILITY_ORIGIN_TOOL,
        "tool_schema_version": POOL_STABILITY_TOOL_SCHEMA_VERSION,
        "producer_version": POOL_STABILITY_PRODUCER_VERSION,
    }


def _validate_producer_run(
    value: object,
    *,
    expected_task_id: str | None = None,
    expected_request: Mapping[str, Any] | None = None,
    expected_stability_id: str | None = None,
    expected_stability_content_hash: str | None = None,
    expected_artifact_id: str | None = None,
    expected_artifact_filename: str | None = None,
    expected_artifact_content_hash: str | None = None,
) -> dict[str, Any]:
    obj = _json_object(value, "Pool stability producer_run")
    _exact_fields(
        obj,
        _PRODUCER_RUN_FIELDS,
        "Pool stability producer_run",
    )
    if obj["schema_version"] != POOL_STABILITY_PRODUCER_RUN_SCHEMA_VERSION:
        raise StrategyError("Pool stability producer_run schema changed")
    task_id = _safe_id(obj["task_id"], "producer_run.task_id")
    request = _validate_inputs(obj["request"])
    tool_ref = _json_object(
        obj["tool_ref"],
        "Pool stability producer_run.tool_ref",
    )
    _exact_fields(
        tool_ref,
        _PRODUCER_TOOL_REF_FIELDS,
        "Pool stability producer_run.tool_ref",
    )
    if tool_ref != _producer_tool_ref():
        raise StrategyError("Pool stability producer_run tool_ref changed")
    input_hash = _hash(
        obj["input_hash"],
        "producer_run.input_hash",
    )
    if input_hash != hashlib.sha256(
        _canonical_json(
            {
                "task_id": task_id,
                "request": request,
                "producer": tool_ref,
            }
        ).encode("utf-8")
    ).hexdigest():
        raise StrategyError("Pool stability producer_run input_hash changed")
    stability_ref = _json_object(
        obj["stability_ref"],
        "Pool stability producer_run.stability_ref",
    )
    _exact_fields(
        stability_ref,
        _PRODUCER_STABILITY_REF_FIELDS,
        "Pool stability producer_run.stability_ref",
    )
    stability_ref = {
        "stability_id": _stability_id(
            stability_ref["stability_id"],
            "producer_run.stability_ref.stability_id",
        ),
        "content_hash": _hash(
            stability_ref["content_hash"],
            "producer_run.stability_ref.content_hash",
        ),
    }
    artifact_ref = _json_object(
        obj["artifact_ref"],
        "Pool stability producer_run.artifact_ref",
    )
    _exact_fields(
        artifact_ref,
        _PRODUCER_ARTIFACT_REF_FIELDS,
        "Pool stability producer_run.artifact_ref",
    )
    artifact_ref = {
        "artifact_id": _hash(
            artifact_ref["artifact_id"],
            "producer_run.artifact_ref.artifact_id",
        ),
        "kind": artifact_ref["kind"],
        "filename": _text(
            artifact_ref["filename"],
            "producer_run.artifact_ref.filename",
        ),
        "content_hash": _hash(
            artifact_ref["content_hash"],
            "producer_run.artifact_ref.content_hash",
        ),
        "origin_tool": artifact_ref["origin_tool"],
    }
    if (
        artifact_ref["kind"] != POOL_STABILITY_ARTIFACT_KIND
        or artifact_ref["origin_tool"] != POOL_STABILITY_ORIGIN_TOOL
        or artifact_ref["filename"]
        != f"{stability_ref['stability_id']}.json"
    ):
        raise StrategyError(
            "Pool stability producer_run artifact_ref changed"
        )
    normalized_body = {
        "schema_version": POOL_STABILITY_PRODUCER_RUN_SCHEMA_VERSION,
        "task_id": task_id,
        "input_hash": input_hash,
        "request": request,
        "tool_ref": tool_ref,
        "stability_ref": stability_ref,
        "artifact_ref": artifact_ref,
    }
    run_id = _text(obj["run_id"], "producer_run.run_id")
    expected_run_id = (
        "strategy-pool-stability-run-"
        + hashlib.sha256(
            _canonical_json(normalized_body).encode("utf-8")
        ).hexdigest()[:24]
    )
    if (
        _RUN_ID_RE.fullmatch(run_id) is None
        or not hmac.compare_digest(run_id, expected_run_id)
    ):
        raise StrategyError("Pool stability producer_run run_id changed")
    without_hash = {**normalized_body, "run_id": run_id}
    content_hash = _hash(
        obj["content_hash"],
        "producer_run.content_hash",
    )
    if not hmac.compare_digest(
        content_hash,
        hashlib.sha256(
            _canonical_json(without_hash).encode("utf-8")
        ).hexdigest(),
    ):
        raise StrategyError(
            "Pool stability producer_run content_hash changed"
        )
    expected_values = (
        (expected_task_id, task_id, "task_id"),
        (expected_request, request, "request"),
        (
            expected_stability_id,
            stability_ref["stability_id"],
            "stability_id",
        ),
        (
            expected_stability_content_hash,
            stability_ref["content_hash"],
            "stability_content_hash",
        ),
        (
            expected_artifact_id,
            artifact_ref["artifact_id"],
            "artifact_id",
        ),
        (
            expected_artifact_filename,
            artifact_ref["filename"],
            "artifact_filename",
        ),
        (
            expected_artifact_content_hash,
            artifact_ref["content_hash"],
            "artifact_content_hash",
        ),
    )
    for expected, actual, name in expected_values:
        if expected is not None and expected != actual:
            raise StrategyError(
                f"Pool stability producer_run {name} binding changed"
            )
    return {**without_hash, "content_hash": content_hash}


def _write_or_require_measurement_audit(
    conn,
    *,
    runtime,
    producer_run: Mapping[str, Any],
    artifact_row_existed: bool,
) -> None:
    run = _validate_producer_run(producer_run)
    rows = conn.execute(
        "SELECT 1 FROM main.audit WHERE kind = ? AND target_ref = ?",
        (POOL_STABILITY_MEASUREMENT_AUDIT_KIND, run["run_id"]),
    ).fetchall()
    if artifact_row_existed:
        require_pool_stability_measurement_audit_on_connection(conn, run)
        return
    if rows:
        raise StrategyError(
            "Pool stability measurement audit exists without its artifact"
        )
    runtime.repo.write_audit_on_connection(
        conn,
        kind=POOL_STABILITY_MEASUREMENT_AUDIT_KIND,
        target_ref=run["run_id"],
        inputs_hash=run["input_hash"],
        outcome="succeeded",
        detail={"producer_run": run},
    )
    require_pool_stability_measurement_audit_on_connection(conn, run)


def _tool_output(
    *,
    stability: Mapping[str, Any],
    task_id: str,
    record: Mapping[str, Any],
    producer_run: Mapping[str, Any],
) -> dict[str, Any]:
    identity = stability["identity"]
    artifact_id = str(record["id"])
    return {
        "schema_version": POOL_STABILITY_TOOL_SCHEMA_VERSION,
        "stability_id": stability["stability_id"],
        "content_hash": stability["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "strategy_type": identity["strategy_type"],
        "baseline_partition": stability["baseline_partition"],
        "comparison_partitions": list(
            stability["comparison_partitions"]
        ),
        "max_psi": _max_psi(stability),
        "stability": dict(stability),
        "warnings": _warnings(stability),
        "artifact": {
            "artifact_id": artifact_id,
            "kind": POOL_STABILITY_ARTIFACT_KIND,
            "format": "json",
            "filename": f"{stability['stability_id']}.json",
            "content_hash": str(record["content_hash"]),
            "download_url": (
                f"/api/tasks/{quote(task_id, safe='')}"
                f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                f"?expected_content_hash="
                f"{quote(str(record['content_hash']), safe='')}"
            ),
        },
        "producer_run_ref": {
            "kind": "tool_run",
            "ref_id": producer_run["run_id"],
            "content_hash": producer_run["content_hash"],
        },
        "read_only": True,
        "effect_validation": False,
        "not_mutated_pool": True,
        "not_created_strategy": True,
        "not_adopted": True,
        "not_promoted": True,
        "not_deployed": True,
    }


def _max_psi(stability: Mapping[str, Any]) -> float:
    values = [
        float(distribution["psi"])
        for population in stability["populations"]
        for comparison in population["comparisons"]
        for distribution in comparison["distributions"]
    ]
    return max(values)


def _warnings(stability: Mapping[str, Any]) -> list[str]:
    return [
        (
            f"{population['population_role']}/"
            f"{comparison['partition']}/{distribution['basis']} "
            f"PSI={float(distribution['psi']):.6g} "
            f"severity={distribution['severity']}"
        )
        for population in stability["populations"]
        for comparison in population["comparisons"]
        for distribution in comparison["distributions"]
        if distribution["severity"] != "stable"
    ]


def _prepare_output_directory(
    tasks_dir: Path | str,
    *,
    task_id: str,
) -> Path:
    task_id = _safe_id(task_id, "task_id")
    root = Path(tasks_dir).absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StrategyError(
            "Pool stability task root must be a regular directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / task_id
    if task_dir.exists() and (
        task_dir.is_symlink() or not task_dir.is_dir()
    ):
        raise StrategyError(
            "Pool stability task directory must be a regular directory"
        )
    task_dir.mkdir(exist_ok=True)
    if (
        task_dir.is_symlink()
        or task_dir.resolve(strict=True).parent
        != root.resolve(strict=True)
    ):
        raise StrategyError("Pool stability task directory escaped storage")
    out_dir = task_dir / "strategy_pool_stabilities"
    if out_dir.exists() and (
        out_dir.is_symlink() or not out_dir.is_dir()
    ):
        raise StrategyError(
            "Pool stability output path must be a regular directory"
        )
    out_dir.mkdir(exist_ok=True)
    if (
        out_dir.is_symlink()
        or out_dir.resolve(strict=True).parent
        != task_dir.resolve(strict=True)
    ):
        raise StrategyError(
            "Pool stability output directory escaped storage"
        )
    return out_dir


def _expected_artifact_path(
    tasks_root: Path,
    *,
    task_id: str,
    stability_id: str,
) -> Path:
    if not tasks_root.is_absolute():
        raise StrategyError("Pool stability task root must be absolute")
    return (
        tasks_root
        / _safe_id(task_id, "task_id")
        / "strategy_pool_stabilities"
        / f"{_stability_id(stability_id, 'stability_id')}.json"
    )


def _select_artifact_row(conn, *, task_id: str, path: Path):
    return conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM main.task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, POOL_STABILITY_ARTIFACT_KIND, str(path)),
    ).fetchone()


def _require_artifact_row_on_connection(
    conn,
    *,
    task_id: str,
    artifact_id: str,
    path: Path,
    content_hash: str,
    provenance_json: str,
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json
          FROM main.task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (task_id, artifact_id),
    ).fetchone()
    if (
        row is None
        or str(row["id"]) != artifact_id
        or str(row["task_id"]) != task_id
        or str(row["kind"]) != POOL_STABILITY_ARTIFACT_KIND
        or str(row["path"]) != str(path)
        or not hmac.compare_digest(
            str(row["content_hash"]),
            content_hash,
        )
        or str(row["origin_tool"]) != POOL_STABILITY_ORIGIN_TOOL
        or str(row["provenance_json"]) != provenance_json
        or artifact_id
        != stable_task_artifact_id(
            task_id=task_id,
            kind=POOL_STABILITY_ARTIFACT_KIND,
            path=str(path),
        )
    ):
        raise StrategyError(
            "Pool stability artifact registry binding changed"
        )


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
    root: Path,
) -> None:
    expected = {
        "task_id": task_id,
        "kind": POOL_STABILITY_ARTIFACT_KIND,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": POOL_STABILITY_ORIGIN_TOOL,
        "provenance_json": _canonical_json(provenance),
    }
    if any(str(row[field]) != value for field, value in expected.items()):
        raise StrategyError(
            "existing Pool stability artifact registry row changed"
        )
    expected_id = stable_task_artifact_id(
        task_id=task_id,
        kind=POOL_STABILITY_ARTIFACT_KIND,
        path=str(path),
    )
    if str(row["id"]) != expected_id:
        raise StrategyError(
            "existing Pool stability artifact stable id changed"
        )
    _require_exact_file(
        path,
        root=root,
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
        raise StrategyError("Pool stability artifact bytes changed")


def _open_retained_exact_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
) -> tuple[int, tuple[int, ...]]:
    _require_exact_file(
        path,
        root=root,
        canonical=canonical,
        content_hash=content_hash,
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        live = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(live.st_mode)
            or _file_identity(opened) != _file_identity(live)
        ):
            raise StrategyError(
                "Pool stability artifact changed before registration"
            )
        identity = _stable_file_stat(opened)
        _require_retained_exact_file(
            descriptor,
            retained_identity=identity,
            path=path,
            canonical=canonical,
            content_hash=content_hash,
        )
        return descriptor, identity
    except OSError as exc:
        raise StrategyError(
            "Pool stability artifact is unavailable before registration"
        ) from exc
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _require_retained_exact_file(
    descriptor: int,
    *,
    retained_identity: tuple[int, ...] | None,
    path: Path,
    canonical: bytes,
    content_hash: str,
) -> None:
    if descriptor < 0 or retained_identity is None:
        raise StrategyError(
            "Pool stability artifact verification fd is missing"
        )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_POOL_STABILITY_JSON_BYTES:
                raise StrategyError(
                    "Pool stability artifact exceeds byte budget"
                )
            digest.update(chunk)
            chunks.append(chunk)
        opened = os.fstat(descriptor)
        live = os.lstat(path)
    except OSError as exc:
        raise StrategyError(
            "Pool stability artifact changed during registration"
        ) from exc
    if (
        _stable_file_stat(opened) != retained_identity
        or not stat.S_ISREG(live.st_mode)
        or stat.S_ISLNK(live.st_mode)
        or _file_identity(opened) != _file_identity(live)
        or b"".join(chunks) != canonical
        or not hmac.compare_digest(digest.hexdigest(), content_hash)
    ):
        raise StrategyError(
            "Pool stability artifact changed during registration"
        )


def _read_regular_nofollow(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> bytes:
    _require_storage_path(path, root=root)
    descriptor = -1
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        live = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(live.st_mode)
            or _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(live)
            or _stable_file_stat(before) != _stable_file_stat(opened)
            or _stable_file_stat(opened) != _stable_file_stat(live)
        ):
            raise StrategyError(
                "Pool stability artifact changed while opening"
            )
        if (
            int(opened.st_size) < 0
            or int(opened.st_size) > MAX_POOL_STABILITY_JSON_BYTES
        ):
            raise StrategyError(
                "Pool stability artifact exceeds byte budget"
            )
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_POOL_STABILITY_JSON_BYTES:
                raise StrategyError(
                    "Pool stability artifact exceeds byte budget"
                )
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        final = os.lstat(path)
        if (
            total != int(opened.st_size)
            or _stable_file_stat(opened) != _stable_file_stat(after)
            or _stable_file_stat(after) != _stable_file_stat(final)
        ):
            raise StrategyError(
                "Pool stability artifact changed while reading"
            )
    except StrategyError:
        raise
    except OSError as exc:
        raise StrategyError(
            "Pool stability artifact could not be read"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    if not hmac.compare_digest(
        digest.hexdigest(),
        expected_content_hash,
    ):
        raise StrategyError(
            "Pool stability artifact content hash drifted"
        )
    return raw


def _require_storage_path(path: Path, *, root: Path) -> None:
    if (
        not root.is_absolute()
        or not path.is_absolute()
        or path != Path(os.path.abspath(path))
    ):
        raise StrategyError(
            "Pool stability artifact path is not canonical"
        )
    try:
        path.relative_to(root)
        root_stat = os.lstat(root)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or stat.S_ISLNK(root_stat.st_mode)
        ):
            raise StrategyError(
                "Pool stability task root must be a regular directory"
            )
        current = path.parent
        while current != root:
            current_stat = os.lstat(current)
            if (
                not stat.S_ISDIR(current_stat.st_mode)
                or stat.S_ISLNK(current_stat.st_mode)
            ):
                raise StrategyError(
                    "Pool stability path traverses a symlink"
                )
            if current == current.parent:
                raise StrategyError(
                    "Pool stability artifact escaped task storage"
                )
            current = current.parent
        final = os.lstat(path)
    except StrategyError:
        raise
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "Pool stability artifact is unavailable"
        ) from exc
    if not stat.S_ISREG(final.st_mode) or stat.S_ISLNK(final.st_mode):
        raise StrategyError(
            "Pool stability artifact must be a regular file"
        )


def _stability_from_bytes(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        return validate_strategy_pool_stability(value)
    except StrategyError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise StrategyError(
            "Pool stability artifact JSON is invalid"
        ) from exc


def _require_binding_connection(conn, *, db_path: Path) -> None:
    if not conn.in_transaction or not db_path.is_absolute():
        raise StrategyError(
            "Pool stability binding requires its governed transaction"
        )
    try:
        database = next(
            (
                row
                for row in conn.execute("PRAGMA database_list").fetchall()
                if str(row["name"]) == "main"
            ),
            None,
        )
    except (KeyError, TypeError, sqlite3.DatabaseError) as exc:
        raise StrategyError(
            "Pool stability binding database is unavailable"
        ) from exc
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != db_path
    ):
        raise StrategyError("Pool stability binding database changed")


def _file_identity(value) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _stable_file_stat(value) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyError(
                f"Pool stability JSON has duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise StrategyError(
        f"Pool stability JSON has non-finite value: {value}"
    )


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{name} keys must be strings")
    try:
        canonical = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized = json.loads(canonical)
    except (
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        json.JSONDecodeError,
    ) as exc:
        raise StrategyError(f"{name} must contain finite JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    return normalized


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise StrategyError("Pool stability JSON is invalid") from exc


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported fields " + ", ".join(extra))
        raise StrategyError(f"{name} has " + "; ".join(details))


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
    ):
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _safe_id(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _SAFE_ID_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} is invalid")
    return normalized


def _hash(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256")
    return normalized


def _cube_id(value: object, name: str) -> str:
    normalized = _text(value, name)
    if re.fullmatch(
        r"strategy-impact-cube-[0-9a-f]{24}",
        normalized,
    ) is None:
        raise StrategyError(f"{name} is invalid")
    return normalized


def _stability_id(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _STABILITY_ID_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} is invalid")
    return normalized


__all__ = [
    "POOL_STABILITY_ARTIFACT_KIND",
    "POOL_STABILITY_ARTIFACT_SCHEMA_VERSION",
    "POOL_STABILITY_MEASUREMENT_AUDIT_KIND",
    "POOL_STABILITY_ORIGIN_TOOL",
    "POOL_STABILITY_PRODUCER_RUN_SCHEMA_VERSION",
    "POOL_STABILITY_TOOL_SCHEMA_VERSION",
    "StrategyPoolStabilityArtifactBinding",
    "authenticate_strategy_pool_stability_artifact_record",
    "load_strategy_pool_stability_artifact",
    "require_pool_stability_measurement_audit_on_connection",
    "require_strategy_pool_stability_artifact_binding_on_connection",
    "run_measure_strategy_pool_stability",
    "validate_measure_strategy_pool_stability_tool_output",
]
