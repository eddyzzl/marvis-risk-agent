"""Immutable task-scoped artifact registry.

Workflow-specific tables may still own their domain records, but downloadable
files are registered here with a content hash and canonical provenance.  The
``(task_id, kind, path)`` tuple is the logical identity: an exact replay is
idempotent, while any content/provenance drift is rejected instead of silently
rewriting evidence.
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


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_IDENTITY_NAMESPACE = "marvis.task_artifact.v1"


class TaskArtifactConflictError(RuntimeError):
    """A logical artifact identity already exists with different evidence."""


class TaskArtifactDataError(ValueError):
    """Supplied or persisted artifact data violates the registry contract."""


class TaskArtifactNotFoundError(KeyError):
    """The owning task does not exist."""


class TaskArtifactRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def transaction(self):
        """Return a configured connection for a caller-owned unit of work."""

        return connect(self.db_path)

    def register(
        self,
        *,
        task_id: str,
        kind: str,
        path: str,
        content_hash: str,
        origin_tool: str,
        provenance: Mapping[str, Any],
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Atomically register one artifact or return its exact prior record."""

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.register_on_connection(
                conn,
                task_id=task_id,
                kind=kind,
                path=path,
                content_hash=content_hash,
                origin_tool=origin_tool,
                provenance=provenance,
                created_at=created_at,
            )

    def register_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        task_id: str,
        kind: str,
        path: str,
        content_hash: str,
        origin_tool: str,
        provenance: Mapping[str, Any],
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Register within the caller's transaction.

        If no transaction is active, an immediate transaction is opened.  It
        is deliberately not committed here so the artifact row can share the
        caller's domain/audit unit of work.
        """

        normalized_task_id = _required_text(task_id, field="task_id")
        normalized_kind = _required_text(kind, field="kind")
        normalized_path = _required_text(path, field="path")
        normalized_hash = _content_hash(content_hash)
        normalized_origin = _required_text(origin_tool, field="origin_tool")
        normalized_provenance, provenance_json = _canonical_provenance(provenance)
        timestamp = (
            _required_text(created_at, field="created_at")
            if created_at is not None
            else _now()
        )
        artifact_id = stable_task_artifact_id(
            task_id=normalized_task_id,
            kind=normalized_kind,
            path=normalized_path,
        )

        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        task = conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (normalized_task_id,)
        ).fetchone()
        if task is None:
            raise TaskArtifactNotFoundError(f"task not found: {normalized_task_id}")

        existing = _select_identity_row(
            conn,
            task_id=normalized_task_id,
            kind=normalized_kind,
            path=normalized_path,
        )
        if existing is not None:
            return _resolve_replay(
                existing,
                content_hash=normalized_hash,
                origin_tool=normalized_origin,
                provenance_json=provenance_json,
            )

        collision = conn.execute(
            "SELECT * FROM task_artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        if collision is not None:
            raise TaskArtifactConflictError(
                "stable task artifact id collided with another logical identity"
            )

        try:
            conn.execute(
                """
                INSERT INTO task_artifacts(
                    id, task_id, kind, path, content_hash, origin_tool,
                    provenance_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    normalized_task_id,
                    normalized_kind,
                    normalized_path,
                    normalized_hash,
                    normalized_origin,
                    provenance_json,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            # A caller-owned deferred transaction can race another writer.
            # Resolve a same-identity insert as an ordinary replay; preserve a
            # typed conflict for any other constraint/collision.
            existing = _select_identity_row(
                conn,
                task_id=normalized_task_id,
                kind=normalized_kind,
                path=normalized_path,
            )
            if existing is None:
                raise TaskArtifactConflictError(
                    "could not register task artifact"
                ) from exc
            return _resolve_replay(
                existing,
                content_hash=normalized_hash,
                origin_tool=normalized_origin,
                provenance_json=provenance_json,
            )

        return {
            "id": artifact_id,
            "task_id": normalized_task_id,
            "kind": normalized_kind,
            "path": normalized_path,
            "content_hash": normalized_hash,
            "origin_tool": normalized_origin,
            "provenance": normalized_provenance,
            "created_at": timestamp,
        }

    def list_for_task(self, task_id: str) -> list[dict[str, Any]]:
        normalized_task_id = _required_text(task_id, field="task_id")
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM task_artifacts
                 WHERE task_id = ?
                 ORDER BY created_at, id
                """,
                (normalized_task_id,),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def list_recent_for_task_kind_with_count(
        self,
        task_id: str,
        kind: str,
        *,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return a newest-first bounded window plus the exact kind count."""

        normalized_task_id = _required_text(task_id, field="task_id")
        normalized_kind = _required_text(kind, field="kind")
        bounded_limit = max(1, int(limit))
        with connect(self.db_path) as conn:
            # The count and bounded window form one selection decision.  An
            # explicit read transaction is required because sqlite3's
            # DEFERRED isolation does not begin a snapshot for bare SELECTs.
            conn.execute("BEGIN")
            count_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                  FROM task_artifacts
                 WHERE task_id = ? AND kind = ?
                """,
                (normalized_task_id, normalized_kind),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                  FROM task_artifacts
                 WHERE task_id = ? AND kind = ?
                 ORDER BY created_at DESC, id DESC
                 LIMIT ?
                """,
                (normalized_task_id, normalized_kind, bounded_limit),
            ).fetchall()
        return [_record_from_row(row) for row in rows], int(count_row["total"])

    def get_for_task_kind_path(
        self,
        task_id: str,
        kind: str,
        path: str,
    ) -> dict[str, Any] | None:
        """Resolve one exact task/kind/path artifact identity."""

        normalized_task_id = _required_text(task_id, field="task_id")
        normalized_kind = _required_text(kind, field="kind")
        normalized_path = _required_text(path, field="path")
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                  FROM task_artifacts
                 WHERE task_id = ? AND kind = ? AND path = ?
                """,
                (normalized_task_id, normalized_kind, normalized_path),
            ).fetchone()
        return None if row is None else _record_from_row(row)

    def find_for_task_kind_origin_by_provenance_candidate_id(
        self,
        task_id: str,
        kind: str,
        origin_tool: str,
        candidate_id: str,
    ) -> list[dict[str, Any]]:
        """Return at most two exact JSON candidate bindings.

        Two rows are sufficient for callers to distinguish missing, unique,
        and ambiguous lineage without scanning or decoding artifact history.
        """

        normalized_task_id = _required_text(task_id, field="task_id")
        normalized_kind = _required_text(kind, field="kind")
        normalized_origin = _required_text(origin_tool, field="origin_tool")
        normalized_candidate_id = _required_text(
            candidate_id,
            field="candidate_id",
        )
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                  FROM task_artifacts
                 WHERE task_id = ?
                   AND kind = ?
                   AND origin_tool = ?
                   AND json_valid(provenance_json)
                   AND json_type(provenance_json, '$.candidate_id') = 'text'
                   AND json_extract(provenance_json, '$.candidate_id') = ?
                 ORDER BY created_at DESC, id DESC
                 LIMIT 2
                """,
                (
                    normalized_task_id,
                    normalized_kind,
                    normalized_origin,
                    normalized_candidate_id,
                ),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def get_for_task(
        self, task_id: str, artifact_id: str
    ) -> dict[str, Any] | None:
        normalized_task_id = _required_text(task_id, field="task_id")
        normalized_artifact_id = _required_text(artifact_id, field="artifact_id")
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM task_artifacts WHERE task_id = ? AND id = ?",
                (normalized_task_id, normalized_artifact_id),
            ).fetchone()
        return None if row is None else _record_from_row(row)

    def get_many_for_task(
        self,
        task_id: str,
        artifact_ids: Sequence[str],
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Resolve a bounded task-owned artifact set in one read snapshot."""

        normalized_task_id = _required_text(task_id, field="task_id")
        bounded_limit = max(1, min(int(limit), 500))
        normalized_ids = tuple(
            dict.fromkeys(
                _required_text(artifact_id, field="artifact_id")
                for artifact_id in artifact_ids
            )
        )
        if not normalized_ids:
            return []
        selected_ids = normalized_ids[:bounded_limit]
        placeholders = ",".join("?" for _item in selected_ids)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                  FROM task_artifacts
                 WHERE task_id = ?
                   AND id IN ({placeholders})
                 ORDER BY created_at DESC, id DESC
                """,
                (normalized_task_id, *selected_ids),
            ).fetchall()
        return [_record_from_row(row) for row in rows]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskArtifactDataError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if "\x00" in normalized:
        raise TaskArtifactDataError(f"{field} must not contain NUL bytes")
    return normalized


def _content_hash(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TaskArtifactDataError("content_hash must be a 64-character SHA-256 hex")
    return value.lower()


def _canonical_provenance(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise TaskArtifactDataError("provenance must be a JSON object")
    try:
        payload = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise TaskArtifactDataError("provenance must be a JSON object") from exc
    if not isinstance(normalized, dict):
        raise TaskArtifactDataError("provenance must be a JSON object")
    return normalized, payload


def stable_task_artifact_id(*, task_id: str, kind: str, path: str) -> str:
    """Return the canonical registry id for one logical artifact identity."""

    normalized_task_id = _required_text(task_id, field="task_id")
    normalized_kind = _required_text(kind, field="kind")
    normalized_path = _required_text(path, field="path")
    identity_json = json.dumps(
        [normalized_task_id, normalized_kind, normalized_path],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        f"{_IDENTITY_NAMESPACE}:{identity_json}".encode("utf-8")
    ).hexdigest()


def _stable_artifact_id(*, task_id: str, kind: str, path: str) -> str:
    """Backward-compatible internal alias for existing artifact consumers."""

    return stable_task_artifact_id(task_id=task_id, kind=kind, path=path)


def _select_identity_row(
    conn: sqlite3.Connection, *, task_id: str, kind: str, path: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, kind, path),
    ).fetchone()


def _resolve_replay(
    row: sqlite3.Row,
    *,
    content_hash: str,
    origin_tool: str,
    provenance_json: str,
) -> dict[str, Any]:
    drifted: list[str] = []
    if not hmac.compare_digest(str(row["content_hash"]), content_hash):
        drifted.append("content_hash")
    if str(row["origin_tool"]) != origin_tool:
        drifted.append("origin_tool")
    if str(row["provenance_json"]) != provenance_json:
        drifted.append("provenance")
    if drifted:
        raise TaskArtifactConflictError(
            "task artifact identity already exists with drift in "
            + ", ".join(drifted)
        )
    return _record_from_row(row)


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        provenance = json.loads(str(row["provenance_json"]))
    except (TypeError, ValueError) as exc:
        raise TaskArtifactDataError("persisted provenance_json is invalid") from exc
    if not isinstance(provenance, dict):
        raise TaskArtifactDataError("persisted provenance_json must be an object")
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
