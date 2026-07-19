"""Immutable, task-owned Strategy Candidate Pool revision persistence.

The repository owns only persistence invariants.  Candidate semantics and the
add/remove/reorder commands live in the Strategy Pool core; this boundary
accepts their complete canonical snapshot, verifies its stable identity, and
advances one mutable head with optimistic compare-and-swap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from marvis.db_schema import connect
from marvis.domain import STRATEGY_TYPES
from marvis.repositories.audit import _write_audit_row


POOL_SCHEMA_VERSION = "strategy.candidate-pool.v1"
POOL_HEAD_SCHEMA_VERSION = "strategy.candidate-pool-head.v1"
POOL_ARTIFACT_KIND = "strategy_candidate_pool_json"
SOURCE_ARTIFACT_KIND = "strategy_candidate_asset_json"
ABSENT_POOL_REVISION = 0
ABSENT_POOL_SNAPSHOT_HASH = hashlib.sha256(
    b"strategy.candidate-pool.absent.v1"
).hexdigest()

_POOL_ID_RE = re.compile(r"^strategy-pool-[0-9a-f]{32}$")
_REVISION_ID_RE = re.compile(r"^strategy-pool-revision-[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "pool_id",
        "task_id",
        "strategy_type",
        "revision",
        "revision_id",
        "parent_revision_id",
        "operation",
        "default_action",
        "entries",
        "status",
        "validation_status",
    }
)
_SNAPSHOT_FIELDS = _SNAPSHOT_BODY_FIELDS | {"snapshot_hash"}
_OPERATION_FIELDS = frozenset({"kind", "operation_hash", "reason"})
_ENTRY_FIELDS = frozenset(
    {
        "entry_id",
        "rule_id",
        "position",
        "source",
        "execution",
        "action",
        "enabled",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "content_hash",
        "asset_id",
        "asset_hash",
        "candidate_kind",
        "fragment_id",
        "effect_id",
        "effect_stage",
        "validation_status",
        "parent_candidate_id",
        "parent_evidence_hash",
        "evidence_identity",
    }
)
_EVIDENCE_IDENTITY_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    }
)
_EXECUTION_FIELDS = frozenset({"condition", "requirements"})


class StrategyCandidatePoolConflictError(RuntimeError):
    """The caller's expected pool head is stale or collided."""


class StrategyCandidatePoolDataError(ValueError):
    """A supplied or persisted pool value violates the canonical contract."""


class StrategyCandidatePoolNotFoundError(KeyError):
    """The owning task or a referenced artifact does not exist."""


def strategy_pool_id(task_id: str, strategy_type: str) -> str:
    """Return the stable identity of one task/type pool."""

    task = _required_text(task_id, field="task_id")
    kind = _strategy_type(strategy_type)
    return _stable_id(
        "strategy-pool",
        {"task_id": task, "strategy_type": kind},
    )


def strategy_pool_operation_hash(
    *,
    pool_id: str,
    parent_revision_id: str | None,
    kind: str,
    reason: str | None,
    default_action: object,
    entries: Sequence[object],
    status: str,
    validation_status: str,
) -> str:
    """Hash one requested transition without its circular revision fields."""

    normalized_pool_id = _pool_id(pool_id)
    parent = _optional_revision_id(parent_revision_id, field="parent_revision_id")
    operation_kind = _required_text(kind, field="operation.kind")
    operation_reason = _optional_text(reason, field="operation.reason")
    normalized_status = _required_text(status, field="status")
    if normalized_status != "draft":
        raise StrategyCandidatePoolDataError("status must remain draft")
    payload = {
        "schema_version": "strategy.candidate-pool-operation.v1",
        "pool_id": normalized_pool_id,
        "parent_revision_id": parent,
        "kind": operation_kind,
        "reason": operation_reason,
        "default_action": _json_object_or_none(default_action, field="default_action"),
        "entries": _json_array(entries, field="entries"),
        "status": normalized_status,
        "validation_status": _required_text(
            validation_status, field="validation_status"
        ),
    }
    return _digest(_canonical_json(payload))


def strategy_pool_revision_id(
    pool_id: str,
    parent_revision_id: str | None,
    operation_hash: str,
) -> str:
    """Return the stable id for a parent-bound pool operation."""

    return _stable_id(
        "strategy-pool-revision",
        {
            "pool_id": _pool_id(pool_id),
            "parent_revision_id": _optional_revision_id(
                parent_revision_id, field="parent_revision_id"
            ),
            "operation_hash": _sha256(operation_hash, field="operation_hash"),
        },
    )


def canonical_strategy_pool_snapshot_json(snapshot: Mapping[str, Any]) -> str:
    """Return the sole canonical JSON serialization for a verified snapshot."""

    return _canonical_json(_normalize_snapshot(snapshot))


def strategy_pool_snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    """Hash the canonical snapshot body, excluding its self-authenticating hash.

    The helper accepts either a complete snapshot or a body that does not yet
    contain ``snapshot_hash`` so callers can calculate and then attach it.
    """

    if not isinstance(snapshot, Mapping):
        raise StrategyCandidatePoolDataError("pool snapshot must be an object")
    if set(snapshot) == _SNAPSHOT_FIELDS:
        body = {key: snapshot[key] for key in _SNAPSHOT_BODY_FIELDS}
    elif set(snapshot) == _SNAPSHOT_BODY_FIELDS:
        body = dict(snapshot)
    else:
        _exact_fields(snapshot, _SNAPSHOT_FIELDS, field="pool snapshot")
        raise AssertionError("unreachable")
    return _digest(_canonical_json(_normalize_snapshot_body(body)))


def strategy_pool_artifact_content_hash(snapshot: Mapping[str, Any]) -> str:
    """Hash the complete canonical JSON bytes registered as the pool artifact."""

    return _digest(canonical_strategy_pool_snapshot_json(snapshot))


class StrategyCandidatePoolRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def transaction(self):
        """Return a configured connection for a caller-owned unit of work."""

        return connect(self.db_path)

    def get_current(self, task_id: str, strategy_type: str) -> dict[str, Any] | None:
        task = _required_text(task_id, field="task_id")
        kind = _strategy_type(strategy_type)
        with connect(self.db_path) as conn:
            head = _select_head(conn, task_id=task, strategy_type=kind)
            if head is None:
                return None
            _validate_head_row(head, task_id=task, strategy_type=kind)
            if int(head["current_revision"]) == ABSENT_POOL_REVISION:
                return None
            row = conn.execute(
                """
                SELECT * FROM strategy_candidate_pool_revisions
                 WHERE id = ? AND pool_id = ?
                """,
                (str(head["current_revision_id"]), str(head["id"])),
            ).fetchone()
            if row is None:
                raise StrategyCandidatePoolDataError(
                    "pool head references a missing revision"
                )
            snapshot = _snapshot_from_row(conn, row)
            if snapshot["revision"] != int(
                head["current_revision"]
            ) or not hmac.compare_digest(
                strategy_pool_snapshot_hash(snapshot),
                str(head["current_snapshot_hash"]),
            ):
                raise StrategyCandidatePoolDataError(
                    "pool head does not match its current revision"
                )
            return snapshot

    def get_revision(
        self,
        task_id: str,
        strategy_type: str,
        revision: int,
    ) -> dict[str, Any] | None:
        task = _required_text(task_id, field="task_id")
        kind = _strategy_type(strategy_type)
        number = _positive_int(revision, field="revision")
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT revision.*
                  FROM strategy_candidate_pool_revisions AS revision
                  JOIN strategy_candidate_pools AS pool
                    ON pool.id = revision.pool_id
                 WHERE pool.task_id = ?
                   AND pool.strategy_type = ?
                   AND revision.revision = ?
                """,
                (task, kind, number),
            ).fetchone()
            return None if row is None else _snapshot_from_row(conn, row)

    def get_revision_by_id(
        self,
        task_id: str,
        strategy_type: str,
        revision_id: str,
    ) -> dict[str, Any] | None:
        task = _required_text(task_id, field="task_id")
        kind = _strategy_type(strategy_type)
        identity = _revision_id(revision_id, field="revision_id")
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM strategy_candidate_pool_revisions
                 WHERE id = ? AND task_id = ? AND strategy_type = ?
                """,
                (identity, task, kind),
            ).fetchone()
            return None if row is None else _snapshot_from_row(conn, row)

    def apply_snapshot(
        self,
        *,
        snapshot: Mapping[str, Any],
        expected_revision: int,
        expected_snapshot_hash: str,
        artifact_id: str,
        artifact_content_hash: str,
        audit: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Own a writer transaction around :meth:`apply_snapshot_on_connection`."""

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.apply_snapshot_on_connection(
                conn,
                snapshot=snapshot,
                expected_revision=expected_revision,
                expected_snapshot_hash=expected_snapshot_hash,
                artifact_id=artifact_id,
                artifact_content_hash=artifact_content_hash,
                audit=audit,
            )

    def apply_snapshot_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        snapshot: Mapping[str, Any],
        expected_revision: int,
        expected_snapshot_hash: str,
        artifact_id: str,
        artifact_content_hash: str,
        audit: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append one immutable revision and CAS-advance its task/type head.

        An exact retry is a no-op only while the exact resulting revision is
        still the head.  Once a later revision advances the pool, the old
        request is stale even if its immutable row remains available.
        """

        normalized = _normalize_snapshot(snapshot)
        expected = _non_negative_int(expected_revision, field="expected_revision")
        expected_hash = _sha256(expected_snapshot_hash, field="expected_snapshot_hash")
        artifact_identity = _required_text(artifact_id, field="artifact_id")
        artifact_hash = _sha256(artifact_content_hash, field="artifact_content_hash")
        snapshot_hash = strategy_pool_snapshot_hash(normalized)
        expected_artifact_hash = strategy_pool_artifact_content_hash(normalized)
        if not hmac.compare_digest(expected_artifact_hash, artifact_hash):
            raise StrategyCandidatePoolDataError(
                "pool artifact content hash must match canonical snapshot JSON"
            )
        audit_payload = _normalize_audit(audit, snapshot=normalized)

        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        task = conn.execute(
            "SELECT id, task_type FROM tasks WHERE id = ?",
            (normalized["task_id"],),
        ).fetchone()
        if task is None:
            raise StrategyCandidatePoolNotFoundError(
                f"task not found: {normalized['task_id']}"
            )
        if str(task["task_type"]) != "strategy":
            raise StrategyCandidatePoolDataError(
                "strategy candidate pools require a strategy task"
            )

        head = _select_head(
            conn,
            task_id=normalized["task_id"],
            strategy_type=normalized["strategy_type"],
        )
        timestamp = _now()
        if head is None:
            if expected != ABSENT_POOL_REVISION or not hmac.compare_digest(
                expected_hash, ABSENT_POOL_SNAPSHOT_HASH
            ):
                raise StrategyCandidatePoolConflictError(
                    "initial pool write requires revision 0 and absent snapshot hash"
                )
            conn.execute(
                """
                INSERT INTO strategy_candidate_pools(
                    id, schema_version, task_id, strategy_type,
                    current_revision, current_revision_id,
                    current_snapshot_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, NULL, ?, ?, ?)
                """,
                (
                    normalized["pool_id"],
                    POOL_HEAD_SCHEMA_VERSION,
                    normalized["task_id"],
                    normalized["strategy_type"],
                    ABSENT_POOL_SNAPSHOT_HASH,
                    timestamp,
                    timestamp,
                ),
            )
            current_revision = ABSENT_POOL_REVISION
            current_revision_id = None
            current_snapshot_hash = ABSENT_POOL_SNAPSHOT_HASH
        else:
            _validate_head_row(
                head,
                task_id=normalized["task_id"],
                strategy_type=normalized["strategy_type"],
            )
            if str(head["id"]) != normalized["pool_id"]:
                raise StrategyCandidatePoolDataError(
                    "pool_id does not match persisted task/type identity"
                )
            current_revision = int(head["current_revision"])
            current_revision_id = head["current_revision_id"]
            current_snapshot_hash = str(head["current_snapshot_hash"])

        existing = conn.execute(
            "SELECT * FROM strategy_candidate_pool_revisions WHERE id = ?",
            (normalized["revision_id"],),
        ).fetchone()
        if existing is not None:
            persisted = _snapshot_from_row(conn, existing)
            exact = (
                persisted == normalized
                and str(existing["artifact_id"]) == artifact_identity
                and hmac.compare_digest(
                    str(existing["artifact_content_hash"]), artifact_hash
                )
            )
            if not exact:
                raise StrategyCandidatePoolDataError(
                    "stable pool revision identity collided with different evidence"
                )
            if expected != normalized["revision"] - 1 or not hmac.compare_digest(
                str(existing["parent_snapshot_hash"]), expected_hash
            ):
                raise StrategyCandidatePoolConflictError(
                    "exact pool retry must use its original parent revision and hash"
                )
            if (
                current_revision_id == normalized["revision_id"]
                and current_revision == normalized["revision"]
                and hmac.compare_digest(current_snapshot_hash, snapshot_hash)
            ):
                return _apply_result(
                    normalized,
                    snapshot_hash=snapshot_hash,
                    artifact_id=artifact_identity,
                    artifact_content_hash=artifact_hash,
                    created=False,
                    replayed=True,
                )
            raise StrategyCandidatePoolConflictError(
                "exact pool operation is no longer the current head"
            )

        operation_collision = conn.execute(
            """
            SELECT id FROM strategy_candidate_pool_revisions
             WHERE pool_id = ?
               AND COALESCE(parent_revision_id, '') = COALESCE(?, '')
               AND operation_hash = ?
            """,
            (
                normalized["pool_id"],
                normalized["parent_revision_id"],
                normalized["operation"]["operation_hash"],
            ),
        ).fetchone()
        if operation_collision is not None:
            raise StrategyCandidatePoolDataError(
                "parent and operation hash map to a different revision id"
            )

        if current_revision != expected or not hmac.compare_digest(
            current_snapshot_hash, expected_hash
        ):
            raise StrategyCandidatePoolConflictError(
                "stale strategy candidate pool head: "
                f"expected revision {expected}, found {current_revision}"
            )
        if normalized["revision"] != expected + 1:
            raise StrategyCandidatePoolDataError(
                "new pool revision must equal expected_revision + 1"
            )
        if normalized["parent_revision_id"] != current_revision_id:
            raise StrategyCandidatePoolDataError(
                "parent_revision_id must identify the current pool head"
            )

        _require_artifact(
            conn,
            task_id=normalized["task_id"],
            artifact_id=artifact_identity,
            kind=POOL_ARTIFACT_KIND,
            content_hash=artifact_hash,
            source=None,
        )
        for entry in normalized["entries"]:
            _require_artifact(
                conn,
                task_id=normalized["task_id"],
                artifact_id=entry["source"]["artifact_id"],
                kind=entry["source"]["kind"],
                content_hash=entry["source"]["content_hash"],
                source=entry["source"],
            )

        operation = normalized["operation"]
        conn.execute(
            """
            INSERT INTO strategy_candidate_pool_revisions(
                id, schema_version, pool_id, task_id, strategy_type, revision,
                parent_revision_id, parent_snapshot_hash, operation_kind,
                operation_hash, operation_reason, default_action_json, status,
                validation_status, snapshot_json, snapshot_hash, artifact_id,
                artifact_content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized["revision_id"],
                POOL_SCHEMA_VERSION,
                normalized["pool_id"],
                normalized["task_id"],
                normalized["strategy_type"],
                normalized["revision"],
                normalized["parent_revision_id"],
                expected_hash,
                operation["kind"],
                operation["operation_hash"],
                operation["reason"],
                _canonical_json(normalized["default_action"]),
                normalized["status"],
                normalized["validation_status"],
                canonical_strategy_pool_snapshot_json(normalized),
                snapshot_hash,
                artifact_identity,
                artifact_hash,
                timestamp,
            ),
        )
        for entry in normalized["entries"]:
            _insert_item(conn, normalized, entry)

        cursor = conn.execute(
            """
            UPDATE strategy_candidate_pools
               SET current_revision = ?, current_revision_id = ?,
                   current_snapshot_hash = ?, updated_at = ?
             WHERE id = ? AND current_revision = ?
               AND current_snapshot_hash = ?
               AND current_revision_id IS ?
            """,
            (
                normalized["revision"],
                normalized["revision_id"],
                snapshot_hash,
                timestamp,
                normalized["pool_id"],
                expected,
                expected_hash,
                current_revision_id,
            ),
        )
        if cursor.rowcount != 1:
            raise StrategyCandidatePoolConflictError(
                "strategy candidate pool head changed while saving"
            )
        _write_audit_row(conn, **audit_payload)
        return _apply_result(
            normalized,
            snapshot_hash=snapshot_hash,
            artifact_id=artifact_identity,
            artifact_content_hash=artifact_hash,
            created=True,
            replayed=False,
        )


def _normalize_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyCandidatePoolDataError("pool snapshot must be an object")
    _exact_fields(value, _SNAPSHOT_FIELDS, field="pool snapshot")
    supplied_hash = _sha256(value["snapshot_hash"], field="snapshot_hash")
    body = _normalize_snapshot_body({key: value[key] for key in _SNAPSHOT_BODY_FIELDS})
    expected_hash = _digest(_canonical_json(body))
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise StrategyCandidatePoolDataError(
            "snapshot_hash does not match canonical pool snapshot body"
        )
    return {**body, "snapshot_hash": supplied_hash}


def _normalize_snapshot_body(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(value, _SNAPSHOT_BODY_FIELDS, field="pool snapshot body")
    if value["schema_version"] != POOL_SCHEMA_VERSION:
        raise StrategyCandidatePoolDataError(
            f"schema_version must be {POOL_SCHEMA_VERSION}"
        )
    task_id = _required_text(value["task_id"], field="task_id")
    strategy_type = _strategy_type(value["strategy_type"])
    pool_id = _pool_id(value["pool_id"])
    expected_pool_id = strategy_pool_id(task_id, strategy_type)
    if pool_id != expected_pool_id:
        raise StrategyCandidatePoolDataError(
            "pool_id does not match task_id and strategy_type"
        )
    revision = _positive_int(value["revision"], field="revision")
    parent_revision_id = _optional_revision_id(
        value["parent_revision_id"], field="parent_revision_id"
    )
    if (revision == 1) != (parent_revision_id is None):
        raise StrategyCandidatePoolDataError(
            "only revision 1 may omit parent_revision_id"
        )
    operation_raw = value["operation"]
    if not isinstance(operation_raw, Mapping):
        raise StrategyCandidatePoolDataError("operation must be an object")
    _exact_fields(operation_raw, _OPERATION_FIELDS, field="operation")
    operation = {
        "kind": _required_text(operation_raw["kind"], field="operation.kind"),
        "operation_hash": _sha256(
            operation_raw["operation_hash"], field="operation.operation_hash"
        ),
        "reason": _optional_text(operation_raw["reason"], field="operation.reason"),
    }
    entries_raw = value["entries"]
    if not _is_sequence(entries_raw):
        raise StrategyCandidatePoolDataError("entries must be an array")
    entries = [
        _normalize_entry(entry, expected_position=index)
        for index, entry in enumerate(entries_raw)
    ]
    _require_unique_entries(entries)
    _require_consistent_evidence_identity(entries)
    default_action = _json_object_or_none(
        value["default_action"], field="default_action"
    )
    status = _required_text(value["status"], field="status")
    if status != "draft":
        raise StrategyCandidatePoolDataError("status must remain draft")
    validation_status = _required_text(
        value["validation_status"], field="validation_status"
    )
    expected_operation_hash = strategy_pool_operation_hash(
        pool_id=pool_id,
        parent_revision_id=parent_revision_id,
        kind=operation["kind"],
        reason=operation["reason"],
        default_action=default_action,
        entries=entries,
        status=status,
        validation_status=validation_status,
    )
    if not hmac.compare_digest(operation["operation_hash"], expected_operation_hash):
        raise StrategyCandidatePoolDataError(
            "operation_hash does not match canonical pool operation"
        )
    revision_id = _revision_id(value["revision_id"], field="revision_id")
    expected_revision_id = strategy_pool_revision_id(
        pool_id,
        parent_revision_id,
        operation["operation_hash"],
    )
    if revision_id != expected_revision_id:
        raise StrategyCandidatePoolDataError(
            "revision_id does not match pool lineage and operation hash"
        )
    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "pool_id": pool_id,
        "task_id": task_id,
        "strategy_type": strategy_type,
        "revision": revision,
        "revision_id": revision_id,
        "parent_revision_id": parent_revision_id,
        "operation": operation,
        "default_action": default_action,
        "entries": entries,
        "status": status,
        "validation_status": validation_status,
    }


def _normalize_entry(value: object, *, expected_position: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyCandidatePoolDataError("pool entry must be an object")
    _exact_fields(value, _ENTRY_FIELDS, field="pool entry")
    position = _non_negative_int(value["position"], field="entry.position")
    if position != expected_position:
        raise StrategyCandidatePoolDataError(
            "entry positions must be contiguous from zero"
        )
    source_raw = value["source"]
    if not isinstance(source_raw, Mapping):
        raise StrategyCandidatePoolDataError("entry.source must be an object")
    _exact_fields(source_raw, _SOURCE_FIELDS, field="entry.source")
    source_kind = _required_text(source_raw["kind"], field="source.kind")
    if source_kind != SOURCE_ARTIFACT_KIND:
        raise StrategyCandidatePoolDataError(
            f"source.kind must be {SOURCE_ARTIFACT_KIND}"
        )
    identity_raw = source_raw["evidence_identity"]
    if not isinstance(identity_raw, Mapping):
        raise StrategyCandidatePoolDataError(
            "source.evidence_identity must be an object"
        )
    _exact_fields(
        identity_raw,
        _EVIDENCE_IDENTITY_FIELDS,
        field="source.evidence_identity",
    )
    evidence_identity = {
        "dataset_id": _required_text(
            identity_raw["dataset_id"], field="evidence_identity.dataset_id"
        ),
        "dataset_content_hash": _sha256(
            identity_raw["dataset_content_hash"],
            field="evidence_identity.dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            identity_raw["workspace_revision"],
            field="evidence_identity.workspace_revision",
        ),
        "workspace_generation": _non_negative_int(
            identity_raw["workspace_generation"],
            field="evidence_identity.workspace_generation",
        ),
        "semantic_mapping_hash": _sha256(
            identity_raw["semantic_mapping_hash"],
            field="evidence_identity.semantic_mapping_hash",
        ),
    }
    source = {
        "artifact_id": _required_text(
            source_raw["artifact_id"], field="source.artifact_id"
        ),
        "kind": source_kind,
        "content_hash": _sha256(
            source_raw["content_hash"], field="source.content_hash"
        ),
        "asset_id": _required_text(source_raw["asset_id"], field="source.asset_id"),
        "asset_hash": _sha256(source_raw["asset_hash"], field="source.asset_hash"),
        "candidate_kind": _required_text(
            source_raw["candidate_kind"], field="source.candidate_kind"
        ),
        "fragment_id": _required_text(
            source_raw["fragment_id"], field="source.fragment_id"
        ),
        "effect_id": _required_text(source_raw["effect_id"], field="source.effect_id"),
        "effect_stage": _required_text(
            source_raw["effect_stage"], field="source.effect_stage"
        ),
        "validation_status": _required_text(
            source_raw["validation_status"], field="source.validation_status"
        ),
        "parent_candidate_id": _required_text(
            source_raw["parent_candidate_id"], field="source.parent_candidate_id"
        ),
        "parent_evidence_hash": _sha256(
            source_raw["parent_evidence_hash"],
            field="source.parent_evidence_hash",
        ),
        "evidence_identity": evidence_identity,
    }
    execution_raw = value["execution"]
    if not isinstance(execution_raw, Mapping):
        raise StrategyCandidatePoolDataError("entry.execution must be an object")
    _exact_fields(execution_raw, _EXECUTION_FIELDS, field="entry.execution")
    condition = _json_object(execution_raw["condition"], field="execution.condition")
    requirements = _json_array(
        execution_raw["requirements"], field="execution.requirements"
    )
    enabled = value["enabled"]
    if not isinstance(enabled, bool):
        raise StrategyCandidatePoolDataError("entry.enabled must be a boolean")
    return {
        "entry_id": _required_text(value["entry_id"], field="entry.entry_id"),
        "rule_id": _required_text(value["rule_id"], field="entry.rule_id"),
        "position": position,
        "source": source,
        "execution": {"condition": condition, "requirements": requirements},
        "action": _json_object_or_none(value["action"], field="entry.action"),
        "enabled": enabled,
    }


def _require_unique_entries(entries: Sequence[Mapping[str, Any]]) -> None:
    for label, values in (
        ("entry_id", [entry["entry_id"] for entry in entries]),
        ("rule_id", [entry["rule_id"] for entry in entries]),
        (
            "asset fragment",
            [
                (entry["source"]["asset_id"], entry["source"]["fragment_id"])
                for entry in entries
            ],
        ),
    ):
        if len(set(values)) != len(values):
            raise StrategyCandidatePoolDataError(
                f"pool entries contain duplicate {label}"
            )


def _require_consistent_evidence_identity(
    entries: Sequence[Mapping[str, Any]],
) -> None:
    if not entries:
        return
    expected = _canonical_json(entries[0]["source"]["evidence_identity"])
    if any(
        _canonical_json(entry["source"]["evidence_identity"]) != expected
        for entry in entries[1:]
    ):
        raise StrategyCandidatePoolDataError(
            "pool entries must share one evidence identity"
        )


def _insert_item(
    conn: sqlite3.Connection,
    snapshot: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    source = entry["source"]
    execution = entry["execution"]
    conn.execute(
        """
        INSERT INTO strategy_candidate_pool_items(
            revision_id, pool_id, task_id, position, entry_id, rule_id,
            source_artifact_id, source_kind, source_content_hash, asset_id,
            asset_hash, candidate_kind, fragment_id, effect_id, effect_stage,
            source_validation_status, parent_candidate_id,
            parent_evidence_hash, dataset_id, dataset_content_hash,
            workspace_revision, workspace_generation, semantic_mapping_hash,
            condition_json, requirements_json, action_json, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot["revision_id"],
            snapshot["pool_id"],
            snapshot["task_id"],
            entry["position"],
            entry["entry_id"],
            entry["rule_id"],
            source["artifact_id"],
            source["kind"],
            source["content_hash"],
            source["asset_id"],
            source["asset_hash"],
            source["candidate_kind"],
            source["fragment_id"],
            source["effect_id"],
            source["effect_stage"],
            source["validation_status"],
            source["parent_candidate_id"],
            source["parent_evidence_hash"],
            source["evidence_identity"]["dataset_id"],
            source["evidence_identity"]["dataset_content_hash"],
            source["evidence_identity"]["workspace_revision"],
            source["evidence_identity"]["workspace_generation"],
            source["evidence_identity"]["semantic_mapping_hash"],
            _canonical_json(execution["condition"]),
            _canonical_json(execution["requirements"]),
            _canonical_json(entry["action"]),
            int(entry["enabled"]),
        ),
    )


def _snapshot_from_row(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    try:
        raw = json.loads(str(row["snapshot_json"]), object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyCandidatePoolDataError(
            "persisted pool snapshot_json is invalid"
        ) from exc
    snapshot = _normalize_snapshot(raw)
    canonical = canonical_strategy_pool_snapshot_json(snapshot)
    if canonical != str(row["snapshot_json"]):
        raise StrategyCandidatePoolDataError(
            "persisted pool snapshot_json is not canonical"
        )
    snapshot_hash = strategy_pool_snapshot_hash(snapshot)
    comparisons = {
        "id": snapshot["revision_id"],
        "schema_version": snapshot["schema_version"],
        "pool_id": snapshot["pool_id"],
        "task_id": snapshot["task_id"],
        "strategy_type": snapshot["strategy_type"],
        "revision": snapshot["revision"],
        "parent_revision_id": snapshot["parent_revision_id"],
        "operation_kind": snapshot["operation"]["kind"],
        "operation_hash": snapshot["operation"]["operation_hash"],
        "operation_reason": snapshot["operation"]["reason"],
        "status": snapshot["status"],
        "validation_status": snapshot["validation_status"],
        "snapshot_hash": snapshot_hash,
    }
    for column, expected in comparisons.items():
        actual = row[column]
        if actual != expected:
            raise StrategyCandidatePoolDataError(
                f"persisted pool revision {column} does not match snapshot"
            )
    if str(row["default_action_json"]) != _canonical_json(snapshot["default_action"]):
        raise StrategyCandidatePoolDataError(
            "persisted default_action projection does not match snapshot"
        )
    _require_artifact(
        conn,
        task_id=snapshot["task_id"],
        artifact_id=str(row["artifact_id"]),
        kind=POOL_ARTIFACT_KIND,
        content_hash=str(row["artifact_content_hash"]),
        source=None,
    )
    if not hmac.compare_digest(
        str(row["artifact_content_hash"]),
        strategy_pool_artifact_content_hash(snapshot),
    ):
        raise StrategyCandidatePoolDataError(
            "persisted pool artifact hash does not match snapshot"
        )
    items = conn.execute(
        """
        SELECT * FROM strategy_candidate_pool_items
         WHERE revision_id = ? ORDER BY position
        """,
        (snapshot["revision_id"],),
    ).fetchall()
    projected = [_entry_from_item_row(item) for item in items]
    if projected != snapshot["entries"]:
        raise StrategyCandidatePoolDataError(
            "persisted pool item projection does not match snapshot"
        )
    return snapshot


def _entry_from_item_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        condition = json.loads(
            str(row["condition_json"]), object_pairs_hook=_unique_object
        )
        requirements = json.loads(
            str(row["requirements_json"]), object_pairs_hook=_unique_object
        )
        action = json.loads(str(row["action_json"]), object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyCandidatePoolDataError(
            "persisted pool item JSON is invalid"
        ) from exc
    return {
        "entry_id": str(row["entry_id"]),
        "rule_id": str(row["rule_id"]),
        "position": int(row["position"]),
        "source": {
            "artifact_id": str(row["source_artifact_id"]),
            "kind": str(row["source_kind"]),
            "content_hash": str(row["source_content_hash"]),
            "asset_id": str(row["asset_id"]),
            "asset_hash": str(row["asset_hash"]),
            "candidate_kind": str(row["candidate_kind"]),
            "fragment_id": str(row["fragment_id"]),
            "effect_id": str(row["effect_id"]),
            "effect_stage": str(row["effect_stage"]),
            "validation_status": str(row["source_validation_status"]),
            "parent_candidate_id": str(row["parent_candidate_id"]),
            "parent_evidence_hash": str(row["parent_evidence_hash"]),
            "evidence_identity": {
                "dataset_id": str(row["dataset_id"]),
                "dataset_content_hash": str(row["dataset_content_hash"]),
                "workspace_revision": int(row["workspace_revision"]),
                "workspace_generation": int(row["workspace_generation"]),
                "semantic_mapping_hash": str(row["semantic_mapping_hash"]),
            },
        },
        "execution": {"condition": condition, "requirements": requirements},
        "action": action,
        "enabled": bool(row["enabled"]),
    }


def _require_artifact(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    artifact_id: str,
    kind: str,
    content_hash: str,
    source: Mapping[str, Any] | None,
) -> None:
    row = conn.execute(
        "SELECT * FROM task_artifacts WHERE id = ? AND task_id = ?",
        (artifact_id, task_id),
    ).fetchone()
    if row is None:
        raise StrategyCandidatePoolNotFoundError(
            f"artifact not found for task: {artifact_id}"
        )
    if str(row["kind"]) != kind or not hmac.compare_digest(
        str(row["content_hash"]), content_hash
    ):
        raise StrategyCandidatePoolDataError(
            f"artifact binding does not match pool evidence: {artifact_id}"
        )
    if source is None:
        return
    try:
        provenance = json.loads(
            str(row["provenance_json"]), object_pairs_hook=_unique_object
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyCandidatePoolDataError(
            "candidate asset provenance is invalid"
        ) from exc
    if not isinstance(provenance, dict):
        raise StrategyCandidatePoolDataError(
            "candidate asset provenance must be an object"
        )
    bindings = {
        "asset_id": source["asset_id"],
        "asset_hash": source["asset_hash"],
        "candidate_id": source["parent_candidate_id"],
        "evidence_hash": source["parent_evidence_hash"],
    }
    for key, expected in bindings.items():
        if provenance.get(key) != expected:
            raise StrategyCandidatePoolDataError(
                f"candidate asset provenance {key} does not match pool source"
            )
    identity = source["evidence_identity"]
    direct_identity = {
        key: provenance.get(key)
        for key in _EVIDENCE_IDENTITY_FIELDS
        if key in provenance
    }
    if set(direct_identity) == _EVIDENCE_IDENTITY_FIELDS:
        identity_provenance = direct_identity
    else:
        parent_artifact_id = provenance.get("source_artifact_id")
        if not isinstance(parent_artifact_id, str) or not parent_artifact_id:
            raise StrategyCandidatePoolDataError(
                "candidate asset provenance lacks evidence identity lineage"
            )
        parent = conn.execute(
            "SELECT * FROM task_artifacts WHERE id = ? AND task_id = ?",
            (parent_artifact_id, task_id),
        ).fetchone()
        if parent is None:
            raise StrategyCandidatePoolNotFoundError(
                f"parent candidate artifact not found for task: {parent_artifact_id}"
            )
        try:
            parent_provenance = json.loads(
                str(parent["provenance_json"]), object_pairs_hook=_unique_object
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StrategyCandidatePoolDataError(
                "parent candidate artifact provenance is invalid"
            ) from exc
        if not isinstance(parent_provenance, dict):
            raise StrategyCandidatePoolDataError(
                "parent candidate artifact provenance must be an object"
            )
        identity_provenance = {
            key: parent_provenance.get(key) for key in _EVIDENCE_IDENTITY_FIELDS
        }
    for key, expected in identity.items():
        if identity_provenance.get(key) != expected:
            raise StrategyCandidatePoolDataError(
                f"candidate asset evidence identity {key} does not match pool source"
            )


def _select_head(
    conn: sqlite3.Connection, *, task_id: str, strategy_type: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM strategy_candidate_pools
         WHERE task_id = ? AND strategy_type = ?
        """,
        (task_id, strategy_type),
    ).fetchone()


def _validate_head_row(row: sqlite3.Row, *, task_id: str, strategy_type: str) -> None:
    if (
        str(row["schema_version"]) != POOL_HEAD_SCHEMA_VERSION
        or str(row["task_id"]) != task_id
        or str(row["strategy_type"]) != strategy_type
        or str(row["id"]) != strategy_pool_id(task_id, strategy_type)
    ):
        raise StrategyCandidatePoolDataError("persisted pool head identity is invalid")
    revision = int(row["current_revision"])
    snapshot_hash = _sha256(
        row["current_snapshot_hash"], field="persisted current_snapshot_hash"
    )
    if revision == ABSENT_POOL_REVISION:
        if row["current_revision_id"] is not None or not hmac.compare_digest(
            snapshot_hash, ABSENT_POOL_SNAPSHOT_HASH
        ):
            raise StrategyCandidatePoolDataError(
                "persisted absent pool head is invalid"
            )
    elif revision < 1 or row["current_revision_id"] is None:
        raise StrategyCandidatePoolDataError("persisted pool head is invalid")


def _normalize_audit(
    audit: Mapping[str, Any], *, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise StrategyCandidatePoolDataError("audit must be an object")
    unexpected = set(audit) - {
        "kind",
        "target_ref",
        "actor",
        "inputs_hash",
        "outcome",
        "detail",
    }
    if unexpected:
        raise StrategyCandidatePoolDataError(
            "audit contains unsupported fields: " + ", ".join(sorted(unexpected))
        )
    kind = _required_text(audit.get("kind"), field="audit.kind")
    target = _required_text(audit.get("target_ref"), field="audit.target_ref")
    if target != snapshot["revision_id"]:
        raise StrategyCandidatePoolDataError(
            "audit.target_ref must identify the new pool revision"
        )
    detail = audit.get("detail", {})
    if not isinstance(detail, Mapping):
        raise StrategyCandidatePoolDataError("audit.detail must be an object")
    normalized_detail = _json_object(detail, field="audit.detail")
    normalized_detail.update(
        {
            "task_id": snapshot["task_id"],
            "pool_id": snapshot["pool_id"],
            "strategy_type": snapshot["strategy_type"],
            "revision": snapshot["revision"],
            "revision_id": snapshot["revision_id"],
            "parent_revision_id": snapshot["parent_revision_id"],
            "operation_kind": snapshot["operation"]["kind"],
            "operation_hash": snapshot["operation"]["operation_hash"],
        }
    )
    return {
        "kind": kind,
        "target_ref": target,
        "actor": _required_text(audit.get("actor", "system"), field="audit.actor"),
        "inputs_hash": (
            None
            if audit.get("inputs_hash") is None
            else _sha256(audit["inputs_hash"], field="audit.inputs_hash")
        ),
        "outcome": (
            None
            if audit.get("outcome") is None
            else _required_text(audit["outcome"], field="audit.outcome")
        ),
        "detail": normalized_detail,
    }


def _apply_result(
    snapshot: Mapping[str, Any],
    *,
    snapshot_hash: str,
    artifact_id: str,
    artifact_content_hash: str,
    created: bool,
    replayed: bool,
) -> dict[str, Any]:
    return {
        "snapshot": json.loads(canonical_strategy_pool_snapshot_json(snapshot)),
        "snapshot_hash": snapshot_hash,
        "artifact_id": artifact_id,
        "artifact_content_hash": artifact_content_hash,
        "created": created,
        "replayed": replayed,
    }


def _strategy_type(value: object) -> str:
    normalized = _required_text(value, field="strategy_type")
    if normalized not in STRATEGY_TYPES:
        raise StrategyCandidatePoolDataError("unsupported strategy_type: " + normalized)
    return normalized


def _pool_id(value: object) -> str:
    normalized = _required_text(value, field="pool_id")
    if _POOL_ID_RE.fullmatch(normalized) is None:
        raise StrategyCandidatePoolDataError("pool_id has an invalid format")
    return normalized


def _revision_id(value: object, *, field: str) -> str:
    normalized = _required_text(value, field=field)
    if _REVISION_ID_RE.fullmatch(normalized) is None:
        raise StrategyCandidatePoolDataError(f"{field} has an invalid format")
    return normalized


def _optional_revision_id(value: object, *, field: str) -> str | None:
    return None if value is None else _revision_id(value, field=field)


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyCandidatePoolDataError(f"{field} must be non-empty text")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    return None if value is None else _required_text(value, field=field)


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyCandidatePoolDataError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    normalized = _non_negative_int(value, field=field)
    if normalized < 1:
        raise StrategyCandidatePoolDataError(f"{field} must be >= 1")
    return normalized


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrategyCandidatePoolDataError(
            f"{field} must be a lowercase SHA-256 hash"
        )
    return value


def _json_object(value: object, *, field: str) -> dict[str, Any]:
    normalized = _json_value(value, field=field)
    if not isinstance(normalized, dict):
        raise StrategyCandidatePoolDataError(f"{field} must be a JSON object")
    return normalized


def _json_object_or_none(value: object, *, field: str) -> dict[str, Any] | None:
    return None if value is None else _json_object(value, field=field)


def _json_array(value: object, *, field: str) -> list[Any]:
    if not _is_sequence(value):
        raise StrategyCandidatePoolDataError(f"{field} must be a JSON array")
    normalized = _json_value(list(value), field=field)
    assert isinstance(normalized, list)
    return normalized


def _json_value(value: object, *, field: str) -> Any:
    try:
        return json.loads(_canonical_json(value), object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyCandidatePoolDataError(f"{field} must be finite JSON") from exc


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, field: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise StrategyCandidatePoolDataError(f"{field} keys must be strings")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown: " + ", ".join(sorted(unknown)))
        raise StrategyCandidatePoolDataError(
            f"{field} fields are invalid ({'; '.join(details)})"
        )


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _stable_id(prefix: str, payload: object) -> str:
    return f"{prefix}-{_digest(_canonical_json(payload))[:32]}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyCandidatePoolDataError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "ABSENT_POOL_REVISION",
    "ABSENT_POOL_SNAPSHOT_HASH",
    "POOL_ARTIFACT_KIND",
    "POOL_HEAD_SCHEMA_VERSION",
    "POOL_SCHEMA_VERSION",
    "SOURCE_ARTIFACT_KIND",
    "StrategyCandidatePoolConflictError",
    "StrategyCandidatePoolDataError",
    "StrategyCandidatePoolNotFoundError",
    "StrategyCandidatePoolRepository",
    "canonical_strategy_pool_snapshot_json",
    "strategy_pool_id",
    "strategy_pool_artifact_content_hash",
    "strategy_pool_operation_hash",
    "strategy_pool_revision_id",
    "strategy_pool_snapshot_hash",
]
