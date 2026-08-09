from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from collections.abc import Callable, Sequence
from pathlib import Path
import sqlite3
import uuid

from marvis.db_schema import connect
from marvis.data.validation_batch_upload_gc import (
    consume_validation_batch_uploads_on_connection,
)
from marvis.domain import (
    TASK_TYPE_VALIDATION,
    TASK_TYPE_VALIDATION_BATCH,
    TaskCreate,
)
from marvis.repositories.tasks import TaskRepository
from marvis.repositories.audit import _write_audit_row
from marvis.state_machine import ConflictError


@dataclass(frozen=True)
class ValidationBatchRecord:
    parent_task_id: str
    status: str
    item_count: int
    summary_path: str
    error_message: str
    started_at: str | None
    finished_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ValidationBatchItemRecord:
    id: str
    parent_task_id: str
    ordinal: int
    child_task_id: str
    model_name: str
    model_version: str
    status: str
    stage: str
    oot_ks: float | None
    oot_psi: float | None
    pmml_status: str
    stress_risk: str
    report_complete: bool
    outcome: str
    error_code: str
    error_message: str
    started_at: str | None
    finished_at: str | None
    created_at: str
    updated_at: str


class ValidationBatchStateConflict(RuntimeError):
    def __init__(self, current_status: str):
        super().__init__(f"validation batch is not startable: {current_status}")
        self.current_status = current_status


class ValidationBatchRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.task_repo = TaskRepository(self.db_path)

    def create_batch(
        self,
        parent_payload: TaskCreate,
        item_payloads: list[TaskCreate],
        *,
        parent_source_relative_path: str | None = None,
        claimed_upload_tokens: Sequence[str] = (),
        claimed_upload_owner: str | None = None,
    ) -> ValidationBatchRecord:
        if parent_payload.task_type != TASK_TYPE_VALIDATION_BATCH:
            raise ValueError("parent task must use validation_batch task type")
        if not 1 <= len(item_payloads) <= 10:
            raise ValueError("validation batch requires 1 to 10 child tasks")
        if any(item.task_type != TASK_TYPE_VALIDATION for item in item_payloads):
            raise ValueError("validation batch children must use validation task type")

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = self.task_repo.create_task_on_connection(conn, parent_payload)
            if parent_source_relative_path is not None:
                self.task_repo.record_task_filesystem_provision_on_connection(
                    conn,
                    parent,
                    target_type="validation_batch_source_dir",
                    relative_path=parent_source_relative_path,
                )
            now = parent.created_at
            conn.execute(
                """
                INSERT INTO validation_batches (
                    parent_task_id, status, item_count, summary_path,
                    error_message, created_at, updated_at
                ) VALUES (?, 'created', ?, '', '', ?, ?)
                """,
                (parent.id, len(item_payloads), now, now),
            )
            for ordinal, item_payload in enumerate(item_payloads, start=1):
                child = self.task_repo.create_task_on_connection(conn, item_payload)
                conn.execute(
                    """
                    INSERT INTO validation_batch_items (
                        id, parent_task_id, ordinal, child_task_id,
                        model_name, model_version, status, stage,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 'queued', ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        parent.id,
                        ordinal,
                        child.id,
                        child.model_name,
                        child.model_version,
                        now,
                        now,
                    ),
                )
            consume_validation_batch_uploads_on_connection(
                conn,
                claimed_upload_tokens,
                claim_owner=claimed_upload_owner,
            )
            batch = _get_batch_on_connection(conn, parent.id)
        return batch

    def purge_preview(self, parent_task_id: str) -> dict:
        """Return one consistent purge preview for the entire batch family."""

        with connect(self.db_path) as conn:
            family = _batch_family_on_connection(conn, parent_task_id)
            active_jobs = _active_batch_jobs_on_connection(conn, family)
            summaries = {
                task_id: self.task_repo.purge_preview_on_connection(conn, task_id)
                for task_id in family
            }
        return _batch_purge_result(
            parent_task_id,
            family,
            summaries,
            active_jobs=active_jobs,
        )

    def purge_batch(
        self,
        parent_task_id: str,
        *,
        actor: str = "system",
        validate_dataset_source_path: Callable[[str], None] | None = None,
        task_fs_targets_by_task: dict[str, Sequence[object]] | None = None,
    ) -> dict:
        """Atomically purge children, parent, ledgers, and cleanup intent."""

        targets_by_task = task_fs_targets_by_task or {}
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            family = _batch_family_on_connection(conn, parent_task_id)
            active_jobs = _active_batch_jobs_on_connection(conn, family)
            if active_jobs:
                joined = ", ".join(
                    f"{job['task_id']}:{job['kind']}" for job in active_jobs
                )
                raise ConflictError(
                    f"validation batch has active jobs: {joined}"
                )
            batch_row = conn.execute(
                """
                SELECT status, item_count, created_at
                  FROM validation_batches
                 WHERE parent_task_id = ?
                """,
                (parent_task_id,),
            ).fetchone()
            summaries: dict[str, dict] = {}
            # Membership rows cascade when a child task is deleted.  Capture the
            # family first, then purge children before the parent so no task can
            # survive as an orphan if the transaction commits.
            for task_id in family[1:]:
                summaries[task_id] = self.task_repo.purge_task_on_connection(
                    conn,
                    task_id,
                    actor=actor,
                    validate_dataset_source_path=validate_dataset_source_path,
                    task_fs_targets=targets_by_task.get(task_id, ()),
                )
            summaries[parent_task_id] = self.task_repo.purge_task_on_connection(
                conn,
                parent_task_id,
                actor=actor,
                validate_dataset_source_path=validate_dataset_source_path,
                task_fs_targets=targets_by_task.get(parent_task_id, ()),
            )
            result = _batch_purge_result(
                parent_task_id,
                family,
                summaries,
                active_jobs=[],
            )
            _write_audit_row(
                conn,
                kind="validation_batch.delete",
                target_ref=parent_task_id,
                actor=actor,
                outcome=(
                    "cleanup_pending"
                    if result["task_fs_gc_candidates"]
                    or result["dataset_source_gc_candidates"]
                    else "db_purged"
                ),
                detail={
                    "batch_status": str(batch_row["status"]),
                    "item_count": int(batch_row["item_count"]),
                    "batch_created_at": str(batch_row["created_at"]),
                    "family_task_ids": family,
                    "purge_summary": result["purge_summary"],
                },
            )
        return result

    def get_batch(self, parent_task_id: str) -> ValidationBatchRecord:
        with connect(self.db_path) as conn:
            batch = _get_batch_on_connection(conn, parent_task_id)
        return batch

    def list_items(self, parent_task_id: str) -> list[ValidationBatchItemRecord]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM validation_batch_items
                 WHERE parent_task_id = ?
                 ORDER BY ordinal ASC
                """,
                (parent_task_id,),
            ).fetchall()
        return [_item_from_row(row) for row in rows]

    def get_item(self, item_id: str) -> ValidationBatchItemRecord:
        with connect(self.db_path) as conn:
            item = _get_item_on_connection(conn, item_id)
        return item

    def claim_start(
        self,
        parent_task_id: str,
        *,
        allowed_statuses: tuple[str, ...] = (
            "created",
            "awaiting_confirmation",
            "partial_failure",
            "failed",
        ),
    ) -> tuple[ValidationBatchRecord, str]:
        now = _now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM validation_batches WHERE parent_task_id = ?",
                (parent_task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Validation batch not found: {parent_task_id}")
            previous_status = str(row["status"])
            if previous_status not in allowed_statuses:
                raise ValidationBatchStateConflict(previous_status)
            conn.execute(
                """
                UPDATE validation_batches
                   SET status = 'running',
                       summary_path = '',
                       error_message = '',
                       started_at = COALESCE(started_at, ?),
                       finished_at = NULL,
                       updated_at = ?
                 WHERE parent_task_id = ?
                """,
                (now, now, parent_task_id),
            )
            batch = _get_batch_on_connection(conn, parent_task_id)
        return batch, previous_status

    def update_batch(
        self,
        parent_task_id: str,
        *,
        status: str,
        summary_path: str | None = None,
        error_message: str | None = None,
        started: bool = False,
        finished: bool = False,
        reopened: bool = False,
    ) -> ValidationBatchRecord:
        now = _now()
        assignments = ["status = ?", "updated_at = ?"]
        values: list[object] = [status, now]
        if summary_path is not None:
            assignments.append("summary_path = ?")
            values.append(summary_path)
        if error_message is not None:
            assignments.append("error_message = ?")
            values.append(error_message)
        if started:
            assignments.append("started_at = COALESCE(started_at, ?)")
            values.append(now)
        if finished:
            assignments.append("finished_at = ?")
            values.append(now)
        elif reopened:
            assignments.append("finished_at = NULL")
        values.append(parent_task_id)
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE validation_batches SET {', '.join(assignments)} "
                "WHERE parent_task_id = ?",  # noqa: S608 - assignments use fixed literals.
                values,
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Validation batch not found: {parent_task_id}")
            batch = _get_batch_on_connection(conn, parent_task_id)
        return batch

    def restore_start_claim(
        self,
        record: ValidationBatchRecord,
    ) -> ValidationBatchRecord:
        """Restore the exact pre-claim snapshot after parent start fails."""

        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE validation_batches
                   SET status = ?,
                       summary_path = ?,
                       error_message = ?,
                       started_at = ?,
                       finished_at = ?,
                       updated_at = ?
                 WHERE parent_task_id = ?
                   AND status = 'running'
                """,
                (
                    record.status,
                    record.summary_path,
                    record.error_message,
                    record.started_at,
                    record.finished_at,
                    record.updated_at,
                    record.parent_task_id,
                ),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT status FROM validation_batches WHERE parent_task_id = ?",
                    (record.parent_task_id,),
                ).fetchone()
                current_status = "missing" if row is None else str(row["status"])
                raise ValidationBatchStateConflict(current_status)
            restored = _get_batch_on_connection(conn, record.parent_task_id)
        return restored

    def update_item(
        self,
        item_id: str,
        *,
        status: str,
        stage: str,
        oot_ks: float | None = None,
        oot_psi: float | None = None,
        pmml_status: str | None = None,
        stress_risk: str | None = None,
        report_complete: bool | None = None,
        outcome: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        started: bool = False,
        finished: bool = False,
        reopened: bool = False,
        clear_results: bool = False,
    ) -> ValidationBatchItemRecord:
        now = _now()
        assignments = ["status = ?", "stage = ?", "updated_at = ?"]
        values: list[object] = [status, stage, now]
        if clear_results:
            assignments.extend([
                "oot_ks = NULL",
                "oot_psi = NULL",
                "pmml_status = ''",
                "stress_risk = ''",
                "report_complete = 0",
                "outcome = ''",
                "error_code = ''",
                "error_message = ''",
            ])
        optional = {
            "oot_ks": oot_ks,
            "oot_psi": oot_psi,
            "pmml_status": pmml_status,
            "stress_risk": stress_risk,
            "report_complete": (
                None if report_complete is None else int(report_complete)
            ),
            "outcome": outcome,
            "error_code": error_code,
            "error_message": error_message,
        }
        for column, value in optional.items():
            if value is None:
                continue
            assignments.append(f"{column} = ?")
            values.append(value)
        if started:
            assignments.append("started_at = COALESCE(started_at, ?)")
            values.append(now)
        if finished:
            assignments.append("finished_at = ?")
            values.append(now)
        elif reopened:
            assignments.append("finished_at = NULL")
        values.append(item_id)
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE validation_batch_items SET {', '.join(assignments)} "
                "WHERE id = ?",  # noqa: S608 - assignments use fixed literals.
                values,
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Validation batch item not found: {item_id}")
            item = _get_item_on_connection(conn, item_id)
        return item


def _get_batch_on_connection(
    conn: sqlite3.Connection,
    parent_task_id: str,
) -> ValidationBatchRecord:
    row = conn.execute(
        "SELECT * FROM validation_batches WHERE parent_task_id = ?",
        (parent_task_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Validation batch not found: {parent_task_id}")
    return _batch_from_row(row)


def _get_item_on_connection(
    conn: sqlite3.Connection,
    item_id: str,
) -> ValidationBatchItemRecord:
    row = conn.execute(
        "SELECT * FROM validation_batch_items WHERE id = ?",
        (item_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Validation batch item not found: {item_id}")
    return _item_from_row(row)


def _batch_from_row(row: sqlite3.Row) -> ValidationBatchRecord:
    return ValidationBatchRecord(
        parent_task_id=str(row["parent_task_id"]),
        status=str(row["status"]),
        item_count=int(row["item_count"]),
        summary_path=str(row["summary_path"] or ""),
        error_message=str(row["error_message"] or ""),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _item_from_row(row: sqlite3.Row) -> ValidationBatchItemRecord:
    return ValidationBatchItemRecord(
        id=str(row["id"]),
        parent_task_id=str(row["parent_task_id"]),
        ordinal=int(row["ordinal"]),
        child_task_id=str(row["child_task_id"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"] or ""),
        status=str(row["status"]),
        stage=str(row["stage"]),
        oot_ks=None if row["oot_ks"] is None else float(row["oot_ks"]),
        oot_psi=None if row["oot_psi"] is None else float(row["oot_psi"]),
        pmml_status=str(row["pmml_status"] or ""),
        stress_risk=str(row["stress_risk"] or ""),
        report_complete=bool(row["report_complete"]),
        outcome=str(row["outcome"] or ""),
        error_code=str(row["error_code"] or ""),
        error_message=str(row["error_message"] or ""),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _batch_family_on_connection(
    conn: sqlite3.Connection,
    parent_task_id: str,
) -> list[str]:
    parent = conn.execute(
        """
        SELECT parent_task_id FROM validation_batches
         WHERE parent_task_id = ?
        """,
        (parent_task_id,),
    ).fetchone()
    if parent is None:
        raise KeyError(f"Validation batch not found: {parent_task_id}")
    children = conn.execute(
        """
        SELECT child_task_id FROM validation_batch_items
         WHERE parent_task_id = ?
         ORDER BY ordinal ASC
        """,
        (parent_task_id,),
    ).fetchall()
    return [parent_task_id, *(str(row["child_task_id"]) for row in children)]


def _active_batch_jobs_on_connection(
    conn: sqlite3.Connection,
    family: list[str],
) -> list[dict[str, str]]:
    placeholders = ",".join("?" for _ in family)
    rows = conn.execute(
        f"""
        SELECT id, task_id, kind, status FROM jobs
         WHERE task_id IN ({placeholders})
           AND status IN ('queued', 'running')
         ORDER BY created_at ASC, id ASC
        """,  # noqa: S608 - placeholders are generated from a bounded task list.
        tuple(family),
    ).fetchall()
    return [
        {
            "id": str(row["id"]),
            "task_id": str(row["task_id"]),
            "kind": str(row["kind"]),
            "status": str(row["status"]),
        }
        for row in rows
    ]


def _batch_purge_result(
    parent_task_id: str,
    family: list[str],
    summaries: dict[str, dict],
    *,
    active_jobs: list[dict[str, str]],
) -> dict:
    aggregate: dict[str, int] = {}
    dataset_source_paths: set[str] = set()
    dataset_source_gc_candidates = 0
    task_fs_gc_candidates = 0
    for summary in summaries.values():
        for key, value in summary.items():
            if key == "dataset_source_paths":
                dataset_source_paths.update(str(path) for path in value)
            elif key == "dataset_source_gc_candidates":
                dataset_source_gc_candidates += int(value)
            elif key == "task_fs_gc_candidates":
                task_fs_gc_candidates += int(value)
            elif isinstance(value, int):
                aggregate[key] = aggregate.get(key, 0) + value
    return {
        "parent_task_id": parent_task_id,
        "family_task_ids": list(family),
        "child_task_ids": list(family[1:]),
        "task_count": len(family),
        "active_jobs": active_jobs,
        "active_job_count": len(active_jobs),
        "purge_summary": aggregate,
        "dataset_source_paths": sorted(dataset_source_paths),
        "dataset_source_gc_candidates": dataset_source_gc_candidates,
        "task_fs_gc_candidates": task_fs_gc_candidates,
        "task_summaries": summaries,
    }


__all__ = [
    "ValidationBatchItemRecord",
    "ValidationBatchRecord",
    "ValidationBatchRepository",
    "ValidationBatchStateConflict",
]
