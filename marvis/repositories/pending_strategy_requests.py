"""One-shot persistence for validated natural-language strategy requests.

The conversation stores only an opaque request id and an integrity hash.  The
validated Strategy DSL and its bound dataset identity remain in this task-scoped
table until the request is consumed, cancelled, or invalidated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal
import uuid

from marvis.db_schema import connect
from marvis.repositories.audit import _write_audit_row


PendingStrategyRequestStatus = Literal[
    "pending", "consumed", "cancelled", "invalidated"
]

_PENDING = "pending"
_TERMINAL_STATUSES = frozenset({"consumed", "cancelled", "invalidated"})
_PAYLOAD_SCHEMA_VERSION = "marvis.pending_strategy_request.v1"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PendingStrategyRequestRecord:
    id: str
    nonce: str
    task_id: str
    validated_draft: dict[str, Any]
    dataset_identity: dict[str, Any] | None
    target_col: str | None
    payload_sha256: str
    status: PendingStrategyRequestStatus
    created_at: str
    updated_at: str

    def to_metadata_reference(self) -> dict[str, str]:
        """Return the only fields that should be copied into message metadata."""

        return {
            "request_id": self.id,
            "payload_sha256": self.payload_sha256,
        }


class PendingStrategyRequestNotFoundError(KeyError):
    """The request does not exist within the caller's task boundary."""


class PendingStrategyRequestConflictError(RuntimeError):
    """The one-shot request is no longer pending or its reference is stale."""


class PendingStrategyRequestDataError(ValueError):
    """Stored or supplied request data violates the persistence contract."""


class PendingStrategyRequestRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def create(
        self,
        *,
        task_id: str,
        validated_draft: dict[str, Any],
        dataset_identity: dict[str, Any] | None,
        target_col: str | None,
    ) -> PendingStrategyRequestRecord:
        """Persist one validated draft and return its opaque metadata reference."""

        normalized_task_id = _required_text(task_id, field="task_id")
        normalized_draft = _json_object(validated_draft, field="validated_draft")
        if not normalized_draft:
            raise PendingStrategyRequestDataError("validated_draft must not be empty")
        normalized_identity = (
            None
            if dataset_identity is None
            else _json_object(dataset_identity, field="dataset_identity")
        )
        normalized_target = _optional_text(target_col, field="target_col")
        request_id = uuid.uuid4().hex
        nonce = uuid.uuid4().hex
        created_at = _now()
        payload_sha256 = _request_payload_sha256(
            request_id=request_id,
            nonce=nonce,
            task_id=normalized_task_id,
            validated_draft=normalized_draft,
            dataset_identity=normalized_identity,
            target_col=normalized_target,
        )
        record = PendingStrategyRequestRecord(
            id=request_id,
            nonce=nonce,
            task_id=normalized_task_id,
            validated_draft=normalized_draft,
            dataset_identity=normalized_identity,
            target_col=normalized_target,
            payload_sha256=payload_sha256,
            status=_PENDING,
            created_at=created_at,
            updated_at=created_at,
        )
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            task = conn.execute(
                "SELECT 1 FROM tasks WHERE id = ?", (normalized_task_id,)
            ).fetchone()
            if task is None:
                raise PendingStrategyRequestNotFoundError(
                    f"task not found: {normalized_task_id}"
                )
            conn.execute(
                """
                INSERT INTO pending_strategy_requests(
                    id, nonce, task_id, validated_draft_json,
                    dataset_identity_json, target_col, payload_sha256,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    record.id,
                    record.nonce,
                    record.task_id,
                    _canonical_json(record.validated_draft),
                    _canonical_json(record.dataset_identity),
                    record.target_col,
                    record.payload_sha256,
                    record.created_at,
                    record.updated_at,
                ),
            )
            _write_audit_row(
                conn,
                kind="strategy.request.create",
                target_ref=record.id,
                inputs_hash=record.payload_sha256,
                outcome="succeeded",
                detail={"task_id": record.task_id, "status": record.status},
            )
        return record

    def get(
        self,
        task_id: str,
        request_id: str,
    ) -> PendingStrategyRequestRecord | None:
        """Load only when both request id and owning task match."""

        with connect(self.db_path) as conn:
            row = _select_request_row(conn, task_id=task_id, request_id=request_id)
        return None if row is None else _record_from_row(row)

    def consume(
        self,
        *,
        task_id: str,
        request_id: str,
        expected_payload_sha256: str,
    ) -> PendingStrategyRequestRecord:
        """Atomically claim a pending request exactly once."""

        return self._transition(
            task_id=task_id,
            request_id=request_id,
            expected_payload_sha256=expected_payload_sha256,
            target_status="consumed",
        )

    def cancel(
        self,
        *,
        task_id: str,
        request_id: str,
        expected_payload_sha256: str,
    ) -> PendingStrategyRequestRecord:
        return self._transition(
            task_id=task_id,
            request_id=request_id,
            expected_payload_sha256=expected_payload_sha256,
            target_status="cancelled",
        )

    def invalidate(
        self,
        *,
        task_id: str,
        request_id: str,
        expected_payload_sha256: str,
    ) -> PendingStrategyRequestRecord:
        return self._transition(
            task_id=task_id,
            request_id=request_id,
            expected_payload_sha256=expected_payload_sha256,
            target_status="invalidated",
        )

    def release_after_failed_start(
        self,
        *,
        task_id: str,
        request_id: str,
        expected_payload_sha256: str,
        existing_plan_ids: set[str] | frozenset[str],
    ) -> PendingStrategyRequestRecord:
        """Release a consumed claim only when plan creation left no trace.

        Legacy confirmations still claim before starting so concurrent callers
        cannot create duplicate plans. If start fails before persisting a plan,
        this guarded transition makes the same opaque request retryable. A new
        plan of any status proves ownership transferred to the plan runtime and
        permanently blocks release.
        """

        normalized_task_id = _required_text(task_id, field="task_id")
        normalized_request_id = _required_text(request_id, field="request_id")
        expected_hash = _sha256_text(
            expected_payload_sha256, field="expected_payload_sha256"
        )
        if not isinstance(existing_plan_ids, (set, frozenset)):
            raise PendingStrategyRequestDataError(
                "existing_plan_ids must be a set or frozenset"
            )
        normalized_existing_plan_ids = {
            _required_text(plan_id, field="existing_plan_ids item")
            for plan_id in existing_plan_ids
        }

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = _select_request_row(
                conn,
                task_id=normalized_task_id,
                request_id=normalized_request_id,
            )
            if row is None:
                raise PendingStrategyRequestNotFoundError(normalized_request_id)
            current = _record_from_row(row)
            if not hmac.compare_digest(current.payload_sha256, expected_hash):
                raise PendingStrategyRequestConflictError(
                    "pending strategy request payload hash does not match"
                )
            if current.status != "consumed":
                raise PendingStrategyRequestConflictError(
                    f"pending strategy request is {current.status}, not consumed"
                )

            plan_rows = conn.execute(
                "SELECT id FROM plans WHERE task_id = ?",
                (normalized_task_id,),
            ).fetchall()
            new_plan_ids = {
                str(plan_row["id"])
                for plan_row in plan_rows
                if str(plan_row["id"]) not in normalized_existing_plan_ids
            }
            if new_plan_ids:
                raise PendingStrategyRequestConflictError(
                    "strategy request already created a plan; claim cannot be released"
                )

            updated_at = _now()
            cursor = conn.execute(
                """
                UPDATE pending_strategy_requests
                   SET status = 'pending', updated_at = ?
                 WHERE id = ? AND task_id = ? AND status = 'consumed'
                   AND payload_sha256 = ?
                """,
                (
                    updated_at,
                    current.id,
                    current.task_id,
                    current.payload_sha256,
                ),
            )
            if cursor.rowcount != 1:
                raise PendingStrategyRequestConflictError(
                    "pending strategy request changed during release"
                )
            _write_audit_row(
                conn,
                kind="strategy.request.release",
                target_ref=current.id,
                inputs_hash=current.payload_sha256,
                outcome="succeeded",
                detail={
                    "task_id": current.task_id,
                    "from": current.status,
                    "to": "pending",
                    "reason": "plan_start_failed_before_persistence",
                },
            )
        return replace(current, status="pending", updated_at=updated_at)

    def _transition(
        self,
        *,
        task_id: str,
        request_id: str,
        expected_payload_sha256: str,
        target_status: PendingStrategyRequestStatus,
    ) -> PendingStrategyRequestRecord:
        if target_status not in _TERMINAL_STATUSES:
            raise ValueError(f"unsupported terminal status: {target_status}")
        normalized_task_id = _required_text(task_id, field="task_id")
        normalized_request_id = _required_text(request_id, field="request_id")
        expected_hash = _sha256_text(
            expected_payload_sha256, field="expected_payload_sha256"
        )

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = _select_request_row(
                conn,
                task_id=normalized_task_id,
                request_id=normalized_request_id,
            )
            if row is None:
                raise PendingStrategyRequestNotFoundError(normalized_request_id)
            current = _record_from_row(row)
            if not hmac.compare_digest(current.payload_sha256, expected_hash):
                raise PendingStrategyRequestConflictError(
                    "pending strategy request payload hash does not match"
                )
            if current.status != _PENDING:
                raise PendingStrategyRequestConflictError(
                    f"pending strategy request already {current.status}"
                )

            updated_at = _now()
            cursor = conn.execute(
                """
                UPDATE pending_strategy_requests
                   SET status = ?, updated_at = ?
                 WHERE id = ? AND task_id = ? AND status = 'pending'
                   AND payload_sha256 = ?
                """,
                (
                    target_status,
                    updated_at,
                    current.id,
                    current.task_id,
                    current.payload_sha256,
                ),
            )
            if cursor.rowcount != 1:
                raise PendingStrategyRequestConflictError(
                    "pending strategy request changed during transition"
                )
            _write_audit_row(
                conn,
                kind=f"strategy.request.{_audit_verb(target_status)}",
                target_ref=current.id,
                inputs_hash=current.payload_sha256,
                outcome="succeeded",
                detail={
                    "task_id": current.task_id,
                    "from": current.status,
                    "to": target_status,
                },
            )
        return replace(current, status=target_status, updated_at=updated_at)


def _select_request_row(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    request_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT id, nonce, task_id, validated_draft_json,
               dataset_identity_json, target_col, payload_sha256,
               status, created_at, updated_at
          FROM pending_strategy_requests
         WHERE id = ? AND task_id = ?
        """,
        (request_id, task_id),
    ).fetchone()


def _record_from_row(row: sqlite3.Row) -> PendingStrategyRequestRecord:
    validated_draft = _load_json_object(
        row["validated_draft_json"], field="validated_draft_json"
    )
    dataset_identity = _load_optional_json_object(
        row["dataset_identity_json"], field="dataset_identity_json"
    )
    status = str(row["status"])
    if status not in {_PENDING, *_TERMINAL_STATUSES}:
        raise PendingStrategyRequestDataError(
            f"unsupported pending strategy request status: {status}"
        )
    record = PendingStrategyRequestRecord(
        id=str(row["id"]),
        nonce=str(row["nonce"]),
        task_id=str(row["task_id"]),
        validated_draft=validated_draft,
        dataset_identity=dataset_identity,
        target_col=(None if row["target_col"] is None else str(row["target_col"])),
        payload_sha256=_sha256_text(row["payload_sha256"], field="payload_sha256"),
        status=status,  # type: ignore[arg-type]
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
    actual_hash = _request_payload_sha256(
        request_id=record.id,
        nonce=record.nonce,
        task_id=record.task_id,
        validated_draft=record.validated_draft,
        dataset_identity=record.dataset_identity,
        target_col=record.target_col,
    )
    if not hmac.compare_digest(actual_hash, record.payload_sha256):
        raise PendingStrategyRequestDataError(
            "stored pending strategy request failed payload integrity validation"
        )
    return record


def _request_payload_sha256(
    *,
    request_id: str,
    nonce: str,
    task_id: str,
    validated_draft: dict[str, Any],
    dataset_identity: dict[str, Any] | None,
    target_col: str | None,
) -> str:
    payload = {
        "schema_version": _PAYLOAD_SCHEMA_VERSION,
        "request_id": request_id,
        "nonce": nonce,
        "task_id": task_id,
        "validated_draft": validated_draft,
        "dataset_identity": dataset_identity,
        "target_col": target_col,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _json_object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PendingStrategyRequestDataError(f"{field} must be a JSON object")
    try:
        parsed = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PendingStrategyRequestDataError(f"{field} must be valid JSON") from exc
    if not isinstance(parsed, dict):  # defensive: root shape cannot change on round-trip
        raise PendingStrategyRequestDataError(f"{field} must be a JSON object")
    return parsed


def _load_json_object(value: object, *, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PendingStrategyRequestDataError(f"{field} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise PendingStrategyRequestDataError(f"{field} must contain a JSON object")
    return parsed


def _load_optional_json_object(value: object, *, field: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PendingStrategyRequestDataError(f"{field} is not valid JSON") from exc
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise PendingStrategyRequestDataError(
            f"{field} must contain a JSON object or null"
        )
    return parsed


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PendingStrategyRequestDataError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256_text(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PendingStrategyRequestDataError(
            f"{field} must be a lowercase SHA256 hex digest"
        )
    return text


def _audit_verb(status: PendingStrategyRequestStatus) -> str:
    return {
        "consumed": "consume",
        "cancelled": "cancel",
        "invalidated": "invalidate",
    }[status]
