"""Immutable, task-owned StrategyProjectContext revision persistence.

The contract module owns context semantics and deterministic identities.  This
repository owns the durable append-only chain, an optimistic triple-CAS head,
and revalidation of canonical persisted bytes at every read boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hmac
from pathlib import Path
import re
import sqlite3
from typing import Any

from marvis.db_schema import connect
from marvis.packs.strategy.project_context import (
    canonical_strategy_project_context_revision_json,
    strategy_project_context_revision_from_json,
    strategy_project_context_state_hash,
    validate_strategy_project_context_revision,
)


PROJECT_CONTEXT_HEAD_SCHEMA_VERSION = "strategy.project-context-head.v1"
ABSENT_PROJECT_CONTEXT_REVISION = 0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StrategyProjectContextConflictError(RuntimeError):
    """The caller's expected context head is stale or collided."""


class StrategyProjectContextDataError(ValueError):
    """A supplied or persisted project context violates its contract."""


class StrategyProjectContextNotFoundError(KeyError):
    """The owning task does not exist."""


class StrategyProjectContextRepository:
    """Persist one governed StrategyProjectContext revision chain per task."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def transaction(self):
        """Return a configured connection for a caller-owned unit of work."""

        return connect(self.db_path)

    def get_current(self, task_id: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            return self.get_current_on_connection(conn, task_id)

    @staticmethod
    def get_current_on_connection(
        conn: sqlite3.Connection,
        task_id: str,
    ) -> dict[str, Any] | None:
        task = _required_text(task_id, field="task_id")
        head = _select_head(conn, task_id=task)
        if head is None:
            return None
        _validate_head_row(head, task_id=task)
        if int(head["current_revision"]) == ABSENT_PROJECT_CONTEXT_REVISION:
            return None
        row = conn.execute(
            """
            SELECT * FROM strategy_project_context_revisions
             WHERE revision_id = ? AND task_id = ?
            """,
            (str(head["current_revision_id"]), task),
        ).fetchone()
        if row is None:
            raise StrategyProjectContextDataError(
                "project context head references a missing revision"
            )
        revision = _revision_from_row(row)
        if (
            revision["revision"] != int(head["current_revision"])
            or revision["revision_id"] != str(head["current_revision_id"])
            or not hmac.compare_digest(
                revision["state_hash"], str(head["current_state_hash"])
            )
        ):
            raise StrategyProjectContextDataError(
                "project context head does not match its current revision"
            )
        return revision

    def get_revision(
        self,
        task_id: str,
        revision: int,
    ) -> dict[str, Any] | None:
        task = _required_text(task_id, field="task_id")
        number = _positive_int(revision, field="revision")
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM strategy_project_context_revisions
                 WHERE task_id = ? AND revision = ?
                """,
                (task, number),
            ).fetchone()
            return None if row is None else _revision_from_row(row)

    def get_revision_by_id(
        self,
        task_id: str,
        revision_id: str,
    ) -> dict[str, Any] | None:
        task = _required_text(task_id, field="task_id")
        identity = _required_text(revision_id, field="revision_id")
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM strategy_project_context_revisions
                 WHERE task_id = ? AND revision_id = ?
                """,
                (task, identity),
            ).fetchone()
            return None if row is None else _revision_from_row(row)

    def refresh(
        self,
        *,
        revision: Mapping[str, Any],
        expected_revision: int,
        expected_revision_id: str | None,
        expected_state_hash: str | None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Append a changed revision or return the unchanged current revision.

        An exact retry succeeds only while its immutable revision remains the
        current head.  All other writes must match the complete current
        ``(revision, revision_id, state_hash)`` triple.
        """

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.refresh_on_connection(
                conn,
                revision=revision,
                expected_revision=expected_revision,
                expected_revision_id=expected_revision_id,
                expected_state_hash=expected_state_hash,
                created_at=created_at,
            )

    @staticmethod
    def refresh_on_connection(
        conn: sqlite3.Connection,
        *,
        revision: Mapping[str, Any],
        expected_revision: int,
        expected_revision_id: str | None,
        expected_state_hash: str | None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Connection-scoped form of :meth:`refresh` for atomic tool effects."""

        normalized = _validate_revision(revision)
        expected, expected_id, expected_hash = _expected_head(
            expected_revision,
            expected_revision_id,
            expected_state_hash,
        )
        timestamp = _optional_timestamp(created_at)

        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        task = conn.execute(
            "SELECT id, task_type FROM tasks WHERE id = ?",
            (normalized["task_id"],),
        ).fetchone()
        if task is None:
            raise StrategyProjectContextNotFoundError(
                f"task not found: {normalized['task_id']}"
            )
        if str(task["task_type"]) != "strategy":
            raise StrategyProjectContextDataError(
                "StrategyProjectContext requires a strategy task"
            )

        head = _select_head(conn, task_id=normalized["task_id"])
        if head is None:
            if expected != 0 or expected_id is not None or expected_hash is not None:
                raise StrategyProjectContextConflictError(
                    "initial project context write requires the absent head triple"
                )
            conn.execute(
                """
                INSERT INTO strategy_project_context_heads(
                    task_id, schema_version, current_revision,
                    current_revision_id, current_state_hash, created_at, updated_at
                ) VALUES (?, ?, 0, NULL, NULL, ?, ?)
                """,
                (
                    normalized["task_id"],
                    PROJECT_CONTEXT_HEAD_SCHEMA_VERSION,
                    timestamp,
                    timestamp,
                ),
            )
            current_revision = 0
            current_revision_id = None
            current_state_hash = None
        else:
            _validate_head_row(head, task_id=normalized["task_id"])
            current_revision = int(head["current_revision"])
            current_revision_id = _optional_row_text(head["current_revision_id"])
            current_state_hash = _optional_row_text(head["current_state_hash"])

        existing = conn.execute(
            """
            SELECT * FROM strategy_project_context_revisions
             WHERE revision_id = ?
            """,
            (normalized["revision_id"],),
        ).fetchone()
        if existing is not None:
            persisted = _revision_from_row(existing)
            if persisted != normalized:
                raise StrategyProjectContextDataError(
                    "stable project context revision identity collided"
                )
            if (
                expected != normalized["revision"] - 1
                or expected_id != normalized["parent_revision_id"]
                or expected_hash != normalized["parent_state_hash"]
            ):
                raise StrategyProjectContextConflictError(
                    "exact project context retry must use its original parent head"
                )
            if (
                current_revision == normalized["revision"]
                and current_revision_id == normalized["revision_id"]
                and current_state_hash == normalized["state_hash"]
            ):
                return persisted
            raise StrategyProjectContextConflictError(
                "exact project context operation is no longer the current head"
            )

        if (
            current_revision != expected
            or current_revision_id != expected_id
            or current_state_hash != expected_hash
        ):
            raise StrategyProjectContextConflictError(
                "stale project context head: expected "
                f"({expected}, {expected_id!r}, {expected_hash!r}), found "
                f"({current_revision}, {current_revision_id!r}, "
                f"{current_state_hash!r})"
            )
        if normalized["revision"] != expected + 1:
            raise StrategyProjectContextDataError(
                "new project context revision must equal expected_revision + 1"
            )
        if (
            normalized["parent_revision_id"] != expected_id
            or normalized["parent_state_hash"] != expected_hash
        ):
            raise StrategyProjectContextDataError(
                "project context parent must match the expected head triple"
            )

        if current_revision > 0 and hmac.compare_digest(
            normalized["state_hash"], current_state_hash or ""
        ):
            current_row = conn.execute(
                """
                SELECT * FROM strategy_project_context_revisions
                 WHERE revision_id = ? AND task_id = ?
                """,
                (current_revision_id, normalized["task_id"]),
            ).fetchone()
            if current_row is None:
                raise StrategyProjectContextDataError(
                    "project context head references a missing revision"
                )
            return _revision_from_row(current_row)

        operation_collision = conn.execute(
            """
            SELECT revision_id FROM strategy_project_context_revisions
             WHERE task_id = ?
               AND COALESCE(parent_revision_id, '') = COALESCE(?, '')
               AND operation_hash = ?
            """,
            (
                normalized["task_id"],
                normalized["parent_revision_id"],
                normalized["operation_hash"],
            ),
        ).fetchone()
        if operation_collision is not None:
            raise StrategyProjectContextDataError(
                "parent and operation hash map to a different revision id"
            )

        conn.execute(
            """
            INSERT INTO strategy_project_context_revisions(
                revision_id, schema_version, producer_version, task_id,
                revision, parent_revision_id, parent_state_hash,
                operation_kind, operation_hash, revision_json, state_hash,
                content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized["revision_id"],
                normalized["schema_version"],
                normalized["producer_version"],
                normalized["task_id"],
                normalized["revision"],
                normalized["parent_revision_id"],
                normalized["parent_state_hash"],
                normalized["operation_kind"],
                normalized["operation_hash"],
                canonical_strategy_project_context_revision_json(normalized),
                normalized["state_hash"],
                normalized["content_hash"],
                timestamp,
            ),
        )
        cursor = conn.execute(
            """
            UPDATE strategy_project_context_heads
               SET current_revision = ?, current_revision_id = ?,
                   current_state_hash = ?, updated_at = ?
             WHERE task_id = ?
               AND current_revision = ?
               AND current_revision_id IS ?
               AND current_state_hash IS ?
            """,
            (
                normalized["revision"],
                normalized["revision_id"],
                normalized["state_hash"],
                timestamp,
                normalized["task_id"],
                expected,
                expected_id,
                expected_hash,
            ),
        )
        if cursor.rowcount != 1:
            raise StrategyProjectContextConflictError(
                "project context head changed while saving"
            )
        return normalized


def _revision_from_row(row: sqlite3.Row) -> dict[str, Any]:
    raw = row["revision_json"]
    if not isinstance(raw, str):
        raise StrategyProjectContextDataError(
            "persisted project context revision_json is invalid"
        )
    try:
        revision = strategy_project_context_revision_from_json(raw)
        canonical = canonical_strategy_project_context_revision_json(revision)
        expected_state_hash = strategy_project_context_state_hash(revision["state"])
    except (TypeError, ValueError) as exc:
        raise StrategyProjectContextDataError(
            "persisted project context revision_json is invalid"
        ) from exc
    if canonical != raw:
        raise StrategyProjectContextDataError(
            "persisted project context revision_json is not canonical"
        )
    comparisons = {
        "revision_id": revision["revision_id"],
        "schema_version": revision["schema_version"],
        "producer_version": revision["producer_version"],
        "task_id": revision["task_id"],
        "revision": revision["revision"],
        "parent_revision_id": revision["parent_revision_id"],
        "parent_state_hash": revision["parent_state_hash"],
        "operation_kind": revision["operation_kind"],
        "operation_hash": revision["operation_hash"],
        "state_hash": revision["state_hash"],
        "content_hash": revision["content_hash"],
    }
    for field, expected in comparisons.items():
        actual = row[field]
        if isinstance(expected, int):
            matches = isinstance(actual, int) and actual == expected
        elif expected is None:
            matches = actual is None
        else:
            matches = isinstance(actual, str) and actual == expected
        if not matches:
            raise StrategyProjectContextDataError(
                f"persisted project context {field} does not match revision_json"
            )
    if not hmac.compare_digest(expected_state_hash, revision["state_hash"]):
        raise StrategyProjectContextDataError(
            "persisted project context state_hash does not match state"
        )
    return revision


def _validate_revision(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyProjectContextDataError("revision must be an object")
    try:
        normalized = validate_strategy_project_context_revision(value)
    except (TypeError, ValueError) as exc:
        raise StrategyProjectContextDataError(
            f"invalid StrategyProjectContext revision: {exc}"
        ) from exc
    if not isinstance(normalized, dict):
        normalized = dict(normalized)
    return normalized


def _expected_head(
    revision: object,
    revision_id: object,
    state_hash: object,
) -> tuple[int, str | None, str | None]:
    number = _non_negative_int(revision, field="expected_revision")
    if number == 0:
        if revision_id is not None or state_hash is not None:
            raise StrategyProjectContextDataError(
                "absent expected head requires null revision_id and state_hash"
            )
        return number, None, None
    return (
        number,
        _required_text(revision_id, field="expected_revision_id"),
        _sha256(state_hash, field="expected_state_hash"),
    )


def _select_head(conn: sqlite3.Connection, *, task_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM strategy_project_context_heads WHERE task_id = ?",
        (task_id,),
    ).fetchone()


def _validate_head_row(row: sqlite3.Row, *, task_id: str) -> None:
    if str(row["task_id"]) != task_id:
        raise StrategyProjectContextDataError("project context head task mismatch")
    if str(row["schema_version"]) != PROJECT_CONTEXT_HEAD_SCHEMA_VERSION:
        raise StrategyProjectContextDataError(
            "project context head schema_version is unsupported"
        )
    revision = row["current_revision"]
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise StrategyProjectContextDataError(
            "project context head revision is invalid"
        )
    revision_id = _optional_row_text(row["current_revision_id"])
    state_hash = _optional_row_text(row["current_state_hash"])
    if revision == 0:
        if revision_id is not None or state_hash is not None:
            raise StrategyProjectContextDataError(
                "absent project context head triple is incomplete"
            )
    elif revision_id is None or state_hash is None or not _SHA256_RE.fullmatch(
        state_hash
    ):
        raise StrategyProjectContextDataError(
            "current project context head triple is incomplete"
        )
    _required_text(row["created_at"], field="head.created_at")
    _required_text(row["updated_at"], field="head.updated_at")


def _non_negative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StrategyProjectContextDataError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    number = _non_negative_int(value, field=field)
    if number == 0:
        raise StrategyProjectContextDataError(f"{field} must be positive")
    return number


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyProjectContextDataError(f"{field} must be non-empty text")
    return value


def _sha256(value: object, *, field: str) -> str:
    normalized = _required_text(value, field=field)
    if _SHA256_RE.fullmatch(normalized) is None:
        raise StrategyProjectContextDataError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return normalized


def _optional_row_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise StrategyProjectContextDataError("persisted head text is invalid")
    return value


def _optional_timestamp(value: object) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    return _required_text(value, field="created_at")
