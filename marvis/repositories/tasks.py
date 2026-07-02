import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from marvis.db_schema import connect
from marvis.domain import (
    TASK_TYPE_VALIDATION,
    VALID_TASK_TYPES,
    TaskCreate,
    TaskRecord,
    TaskStatus,
)
from marvis.model_algorithms import normalize_algorithm
from marvis.report_texts import COMPUTED_REPORT_TEXT_KEYS
from marvis.repositories.audit import _count_audit_rows, _list_audit_rows, _write_audit_row
from marvis.repositories.modeling import _set_experiment_status_row
from marvis.state_machine import (
    ConflictError,
    IllegalTransition,
    assert_transition,
)

AGENT_REPORT_CONCLUSION_KEYS = frozenset({
    "TEXT:pressure_test_summary",
    "TEXT:pressure_impact_recommendation",
    "TEXT:final_validation_conclusion",
})


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TaskRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def transaction(self):
        return connect(self.db_path)

    def create_task(self, payload: TaskCreate) -> TaskRecord:
        record = _task_record_from_create(payload)
        with connect(self.db_path) as conn:
            _insert_task_record_row(conn, record, report_values=payload.report_values)
        return record

    def create_task_with_audit(self, payload: TaskCreate, *, audit_factory) -> TaskRecord:
        with connect(self.db_path) as conn:
            return self.create_task_with_audit_on_connection(
                conn,
                payload,
                audit_factory=audit_factory,
            )

    def create_task_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        payload: TaskCreate,
        *,
        audit_factory,
    ) -> TaskRecord:
        record = _task_record_from_create(payload)
        audit = audit_factory(record)
        _insert_task_record_row(conn, record, report_values=payload.report_values)
        _write_audit_row(conn, **audit)
        return record

    def create_validation_handoff_with_audit(
        self,
        payload: TaskCreate,
        *,
        experiment_id: str,
        experiment_status: str,
        audit_factory,
    ) -> TaskRecord:
        with connect(self.db_path) as conn:
            return self.create_validation_handoff_with_audit_on_connection(
                conn,
                payload,
                experiment_id=experiment_id,
                experiment_status=experiment_status,
                audit_factory=audit_factory,
            )

    def create_validation_handoff_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        payload: TaskCreate,
        *,
        experiment_id: str,
        experiment_status: str,
        audit_factory,
    ) -> TaskRecord:
        record = _task_record_from_create(payload)
        audit = audit_factory(record)
        _insert_task_record_row(conn, record, report_values=payload.report_values)
        _set_experiment_status_row(conn, experiment_id, experiment_status)
        _write_audit_row(conn, **audit)
        return record

    def get_task(self, task_id: str) -> TaskRecord:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        return _row_to_task(row)

    def update_algorithm(self, task_id: str, algorithm: str) -> TaskRecord:
        normalized = normalize_algorithm(algorithm, allow_empty=True)
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE tasks
                   SET algorithm = ?, updated_at = ?
                 WHERE id = ?
                """,
                (normalized, _now(), task_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Task not found: {task_id}")
        return self.get_task(task_id)

    def delete_task(self, task_id: str) -> None:
        with connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            if cursor.rowcount == 0:
                raise KeyError(f"Task not found: {task_id}")

    def purge_preview(self, task_id: str) -> dict:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            summary = _task_purge_summary(conn, task_id)
        summary.pop("_dataset_source_paths", None)
        return summary

    def purge_task(self, task_id: str, *, actor: str = "system") -> dict:
        with connect(self.db_path) as conn:
            return self.purge_task_on_connection(conn, task_id, actor=actor)

    def purge_task_on_connection(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        *,
        actor: str = "system",
    ) -> dict:
        row = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        summary = _task_purge_summary(conn, task_id)
        source_paths = summary.pop("_dataset_source_paths")
        # datasets/joins/plans/experiments/strategies/sub_agents have no ON DELETE
        # CASCADE from tasks (see marvis/db_schema.py); their own children
        # (model_artifacts, backtests, plan_steps/outputs/runs) do cascade once the
        # parent row is removed. jobs/agent_messages already cascade from tasks.
        conn.execute("DELETE FROM datasets WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM joins WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM plans WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM experiments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM strategies WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM sub_agents WHERE parent_task_id = ?", (task_id,))
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        _write_audit_row(
            conn,
            kind="task.delete",
            target_ref=task_id,
            actor=actor,
            outcome="succeeded",
            detail={"purge_summary": summary},
        )
        summary["dataset_source_paths"] = source_paths
        return summary

    def list_tasks(self, *, limit: int | None = None, offset: int = 0) -> list[TaskRecord]:
        bounded_limit = None if limit is None else max(1, int(limit))
        bounded_offset = max(0, int(offset))
        with connect(self.db_path) as conn:
            if bounded_limit is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM tasks
                     ORDER BY created_at DESC, id DESC
                     LIMIT ? OFFSET ?
                    """,
                    (bounded_limit, bounded_offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC, id DESC"
                ).fetchall()
        return [_row_to_task(row) for row in rows]

    def list_audit(
        self,
        *,
        kind: str | None = None,
        kind_prefix: str | None = None,
        target_ref: str | None = None,
        target_ref_prefix: str | None = None,
        task_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict]:
        return _list_audit_rows(
            self.db_path,
            kind=kind,
            kind_prefix=kind_prefix,
            target_ref=target_ref,
            target_ref_prefix=target_ref_prefix,
            task_id=task_id,
            after=after,
            before=before,
            limit=limit,
            offset=offset,
        )

    def count_audit(
        self,
        *,
        kind: str | None = None,
        kind_prefix: str | None = None,
        target_ref: str | None = None,
        target_ref_prefix: str | None = None,
        task_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> int:
        return _count_audit_rows(
            self.db_path,
            kind=kind,
            kind_prefix=kind_prefix,
            target_ref=target_ref,
            target_ref_prefix=target_ref_prefix,
            task_id=task_id,
            after=after,
            before=before,
        )

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        message: str,
        *,
        expected: TaskStatus | set[TaskStatus] | None = None,
        reason_code: str = "",
    ) -> None:
        with connect(self.db_path) as conn:
            self.update_status_on_connection(
                conn,
                task_id,
                status,
                message,
                expected=expected,
                reason_code=reason_code,
                begin_immediate=True,
            )

    def update_status_on_connection(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        status: TaskStatus,
        message: str,
        *,
        expected: TaskStatus | set[TaskStatus] | None = None,
        reason_code: str = "",
        begin_immediate: bool = False,
    ) -> None:
        expected_set = _expected_status_set(expected)
        if begin_immediate:
            conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if current_row is None:
            raise KeyError(f"Task not found: {task_id}")
        current = TaskStatus(current_row["status"])
        if current not in expected_set:
            raise IllegalTransition(current, status)
        assert_transition(current, status)
        placeholders = ",".join(["?"] * len(expected_set))
        cursor = conn.execute(
            f"""
            UPDATE tasks
               SET status = ?,
                   status_message = ?,
                   status_reason_code = ?,
                   updated_at = ?
             WHERE id = ?
               AND status IN ({placeholders})
            """,
            (
                status.value,
                message,
                reason_code,
                _now(),
                task_id,
                *(allowed.value for allowed in expected_set),
            ),
        )
        if cursor.rowcount == 0:
            latest = conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if latest is None:
                raise KeyError(f"Task not found: {task_id}")
            raise IllegalTransition(TaskStatus(latest["status"]), status)

    def update_status_message(
        self,
        task_id: str,
        message: str,
        *,
        reason_code: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            if reason_code is None:
                cursor = conn.execute(
                    """
                    UPDATE tasks
                       SET status_message = ?,
                           updated_at = ?
                     WHERE id = ?
                    """,
                    (message, _now(), task_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE tasks
                       SET status_message = ?,
                           status_reason_code = ?,
                           updated_at = ?
                     WHERE id = ?
                    """,
                    (message, reason_code, _now(), task_id),
                )
            if cursor.rowcount == 0:
                raise KeyError(f"Task not found: {task_id}")

    def reset_status_for_agent_rerun(
        self,
        task_id: str,
        status: TaskStatus,
        message: str,
        *,
        clear_agent_report_conclusions: bool = False,
    ) -> None:
        """Controlled status rewind for explicit Agent rerun requests.

        Normal status transitions are forward-only. Agent reruns intentionally
        rewind the visible workflow while keeping chat history intact.
        """
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT report_values_json
                  FROM tasks
                 WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            values = _load_json_dict(row["report_values_json"])
            revision_increment = 0
            if clear_agent_report_conclusions:
                for key in AGENT_REPORT_CONCLUSION_KEYS:
                    values[key] = ""
                revision_increment = 1
            conn.execute(
                """
                UPDATE tasks
                   SET status = ?,
                       status_message = ?,
                       status_reason_code = '',
                       report_values_json = ?,
                       report_values_revision = report_values_revision + ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    status.value,
                    message,
                    _dump_json_dict(values),
                    revision_increment,
                    _now(),
                    task_id,
                ),
            )

    def start_job(self, task_id: str, kind: str) -> str:
        job_id = uuid.uuid4().hex
        now = _now()
        try:
            with connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO jobs(id, task_id, kind, status, created_at)
                    VALUES (?, ?, ?, 'queued', ?)
                    """,
                    (job_id, task_id, kind, now),
                )
        except sqlite3.IntegrityError as exc:
            # A UNIQUE violation on idx_jobs_active_task means another job for
            # this task was queued/running at INSERT time -- that is the race
            # this constraint exists to catch, so surface it as a 409 directly.
            # Do NOT re-query task_has_active_job to "confirm": if the competing
            # job finished in the interim the re-check returns False and the raw
            # IntegrityError escapes as an unhandled 500 (TST-9b TOCTOU). The
            # constraint already decided there was contention at this instant;
            # {202, 409} are the only outcomes.
            raise ConflictError(
                f"task {task_id} already has an active job"
            ) from exc
        return job_id

    def mark_job_running(self, job_id: str) -> None:
        now = _now()
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                   SET status = 'running',
                       started_at = COALESCE(started_at, ?),
                       heartbeat_at = ?
                 WHERE id = ?
                """,
                (now, now, job_id),
            )

    def touch_job_heartbeat(self, job_id: str) -> bool:
        """Bump ``heartbeat_at`` for a still-running job (REL-5). Long job
        executors (notebook/metrics/join/plan/driver) call this periodically
        from a background thread so the watchdog can tell "still working" from
        "process died mid-job". Only updates rows still queued/running so a
        heartbeat racing a concurrent finish_job() can't resurrect a terminal
        job; returns whether the row was actually touched."""
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                   SET heartbeat_at = ?
                 WHERE id = ?
                   AND status IN ('queued', 'running')
                """,
                (_now(), job_id),
            )
        return cursor.rowcount > 0

    def count_heartbeat_stale_running_jobs(self, *, older_than_seconds: int) -> int:
        """Read-only count of RUNNING jobs whose heartbeat is already stale
        (would be released by the next watchdog sweep). Exposed on /api/health
        so "stuck" is at least observable (REL-5) without waiting for the
        watchdog to actually fire."""
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        ).isoformat()
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS stale_count
                  FROM jobs
                 WHERE status = 'running'
                   AND COALESCE(heartbeat_at, started_at, created_at) <= ?
                """,
                (cutoff,),
            ).fetchone()
        return int(row["stale_count"]) if row is not None else 0

    def fail_heartbeat_lost_jobs(self, *, older_than_seconds: int) -> list[dict]:
        """Watchdog sweep (REL-5): fail every RUNNING job whose heartbeat
        (falling back to started_at/created_at for jobs predating this column,
        or one that never ticked) is older than the threshold, releasing
        idx_jobs_active_task so the task isn't wedged behind a 409 forever
        because a background thread hung. Select-then-update inside one
        connection/transaction so a job that finishes normally in the same
        instant it goes stale can't be double-failed; each release is audited.
        Returns the released job rows (id/task_id/kind)."""
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        ).isoformat()
        now = _now()
        released: list[dict] = []
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, kind
                  FROM jobs
                 WHERE status = 'running'
                   AND COALESCE(heartbeat_at, started_at, created_at) <= ?
                """,
                (cutoff,),
            ).fetchall()
            for row in rows:
                job_id = str(row["id"])
                cursor = conn.execute(
                    """
                    UPDATE jobs
                       SET status = 'failed',
                           error_name = 'HeartbeatLost',
                           error_value = ?,
                           finished_at = ?
                     WHERE id = ?
                       AND status = 'running'
                    """,
                    (
                        f"job heartbeat exceeded {older_than_seconds}s without an update",
                        now,
                        job_id,
                    ),
                )
                if cursor.rowcount == 0:
                    continue
                released.append(
                    {"id": job_id, "task_id": str(row["task_id"]), "kind": str(row["kind"])}
                )
                _write_audit_row(
                    conn,
                    kind="job.heartbeat_lost",
                    target_ref=job_id,
                    outcome="failed",
                    detail={
                        "task_id": str(row["task_id"]),
                        "job_kind": str(row["kind"]),
                        "stale_after_seconds": older_than_seconds,
                    },
                )
        return released

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        error_name: str | None = None,
        error_value: str | None = None,
        traceback: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                   SET status = ?,
                       error_name = ?,
                       error_value = ?,
                       traceback = ?,
                       finished_at = ?
                 WHERE id = ?
                   AND status IN ('queued', 'running')
                """,
                (status, error_name, error_value, traceback, _now(), job_id),
            )

    def task_has_active_job(self, task_id: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT 1
                  FROM jobs
                 WHERE task_id = ?
                   AND status IN ('queued', 'running')
                 LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        return row is not None

    def get_job(self, job_id: str) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, kind, status, progress_message,
                       error_name, error_value, created_at, started_at,
                       finished_at, log_path, heartbeat_at
                  FROM jobs
                 WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def get_active_job_kind(self, task_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT kind
                  FROM jobs
                 WHERE task_id = ?
                   AND status IN ('queued', 'running')
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        return None if row is None else str(row["kind"])

    def get_active_job_kinds_for_tasks(self, task_ids: list[str]) -> dict[str, str]:
        """Batched form of get_active_job_kind for the polling task-list endpoint
        (PERF-6): one connection + one query for N tasks instead of N connections,
        via a window function that keeps only the most-recent active job per task.
        Semantics match get_active_job_kind exactly (same status filter, same
        "most recent by created_at" tie-break); an empty task_ids list short-circuits
        without opening a connection."""
        if not task_ids:
            return {}
        placeholders = ",".join("?" for _ in task_ids)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT task_id, kind
                  FROM (
                        SELECT task_id, kind,
                               ROW_NUMBER() OVER (
                                   PARTITION BY task_id
                                   ORDER BY created_at DESC
                               ) AS rn
                          FROM jobs
                         WHERE task_id IN ({placeholders})
                           AND status IN ('queued', 'running')
                       )
                 WHERE rn = 1
                """,
                tuple(task_ids),
            ).fetchall()
        return {str(row["task_id"]): str(row["kind"]) for row in rows}

    def get_latest_failed_job_kind(self, task_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT kind
                  FROM jobs
                 WHERE task_id = ?
                   AND status = 'failed'
                 ORDER BY COALESCE(finished_at, started_at, created_at) DESC,
                          created_at DESC,
                          id DESC
                 LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        return None if row is None else str(row["kind"])

    def get_latest_job(self, task_id: str, *, kind: str | None = None) -> dict | None:
        params: list[str] = [task_id]
        kind_clause = ""
        if kind:
            kind_clause = " AND kind = ?"
            params.append(kind)
        with connect(self.db_path) as conn:
            row = conn.execute(
                f"""
                SELECT id, task_id, kind, status, progress_message,
                       error_name, error_value, created_at, started_at,
                       finished_at, log_path
                  FROM jobs
                 WHERE task_id = ?
                       {kind_clause}
                 ORDER BY COALESCE(finished_at, started_at, created_at) DESC,
                          created_at DESC,
                          id DESC
                 LIMIT 1
                """,
                tuple(params),
            ).fetchone()
        return None if row is None else dict(row)

    def get_report_values(self, task_id: str) -> tuple[dict[str, str], int]:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT report_values_json, report_values_revision
                  FROM tasks
                 WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        return _load_json_dict(row["report_values_json"]), int(
            row["report_values_revision"]
        )

    def update_report_values(
        self,
        task_id: str,
        values: dict[str, str],
        expected_revision: int,
    ) -> int:
        _validate_report_values(values)
        _reject_computed_report_values(values)
        return self._merge_report_values(task_id, values, expected_revision)

    def update_report_values_with_audit(
        self,
        task_id: str,
        values: dict[str, str],
        expected_revision: int,
        *,
        audit: dict,
    ) -> int:
        _validate_report_values(values)
        _reject_computed_report_values(values)
        return self._merge_report_values(
            task_id,
            values,
            expected_revision,
            audit=audit,
        )

    def update_agent_report_conclusions(
        self,
        task_id: str,
        values: dict[str, str],
        expected_revision: int,
    ) -> int:
        _validate_report_values(values)
        invalid_keys = sorted(set(values) - AGENT_REPORT_CONCLUSION_KEYS)
        if invalid_keys:
            raise ValueError(
                "agent confirmation can only update agent conclusion keys: "
                + ", ".join(invalid_keys)
            )
        return self._merge_report_values(task_id, values, expected_revision)

    def update_agent_report_conclusions_with_audit(
        self,
        task_id: str,
        values: dict[str, str],
        expected_revision: int,
        *,
        audit: dict,
    ) -> int:
        _validate_report_values(values)
        invalid_keys = sorted(set(values) - AGENT_REPORT_CONCLUSION_KEYS)
        if invalid_keys:
            raise ValueError(
                "agent confirmation can only update agent conclusion keys: "
                + ", ".join(invalid_keys)
            )
        return self._merge_report_values(
            task_id,
            values,
            expected_revision,
            audit=audit,
        )

    def add_agent_message(
        self,
        task_id: str,
        *,
        role: str,
        stage: str,
        content: str,
        metadata: dict | None = None,
    ) -> dict:
        message_id = uuid.uuid4().hex
        now = _now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        with connect(self.db_path) as conn:
            task_row = conn.execute(
                "SELECT 1 FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if task_row is None:
                raise KeyError(f"Task not found: {task_id}")
            conn.execute(
                """
                INSERT INTO agent_messages
                (id, task_id, role, stage, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, task_id, role, stage, content, now, metadata_json),
            )
        return {
            "id": message_id,
            "task_id": task_id,
            "role": role,
            "stage": stage,
            "content": content,
            "created_at": now,
            "metadata": metadata or {},
        }

    def list_agent_messages(
        self,
        task_id: str,
        *,
        after_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        bounded_limit = None if limit is None else max(1, int(limit))
        with connect(self.db_path) as conn:
            task_row = conn.execute(
                "SELECT 1 FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if task_row is None:
                raise KeyError(f"Task not found: {task_id}")
            after_row = None
            if after_id:
                after_row = conn.execute(
                    """
                    SELECT created_at, id
                      FROM agent_messages
                     WHERE task_id = ?
                       AND id = ?
                    """,
                    (task_id, after_id),
                ).fetchone()
            if after_id and after_row is not None:
                limit_clause = " LIMIT ?" if bounded_limit is not None else ""
                params: tuple = (
                    task_id,
                    after_row["created_at"],
                    after_row["created_at"],
                    after_row["id"],
                )
                if bounded_limit is not None:
                    params = (*params, bounded_limit)
                rows = conn.execute(
                    f"""
                    SELECT id, task_id, role, stage, content, created_at, metadata_json
                      FROM agent_messages
                     WHERE task_id = ?
                       AND (created_at > ? OR (created_at = ? AND id > ?))
                     ORDER BY created_at ASC, id ASC
                     {limit_clause}
                    """,
                    params,
                ).fetchall()
                return [_row_to_agent_message(row) for row in rows]
            limit_clause = " LIMIT ?" if bounded_limit is not None else ""
            params = (task_id,)
            if bounded_limit is not None:
                params = (*params, bounded_limit)
            rows = conn.execute(
                f"""
                SELECT id, task_id, role, stage, content, created_at, metadata_json
                  FROM agent_messages
                 WHERE task_id = ?
                 ORDER BY created_at ASC, id ASC
                 {limit_clause}
                """,
                params,
            ).fetchall()
        return [_row_to_agent_message(row) for row in rows]

    def has_agent_message(self, task_id: str, message_id: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT 1
                  FROM agent_messages
                 WHERE task_id = ?
                   AND id = ?
                """,
                (task_id, message_id),
            ).fetchone()
        return row is not None

    def get_agent_message(self, task_id: str, message_id: str) -> dict:
        with connect(self.db_path) as conn:
            task_row = conn.execute(
                "SELECT 1 FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if task_row is None:
                raise KeyError(f"Task not found: {task_id}")
            row = conn.execute(
                """
                SELECT id, task_id, role, stage, content, created_at, metadata_json
                  FROM agent_messages
                 WHERE task_id = ?
                   AND id = ?
                """,
                (task_id, message_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Agent message not found: {message_id}")
        return _row_to_agent_message(row)

    def update_agent_message(
        self,
        message_id: str,
        *,
        content: str,
        metadata: dict | None = None,
    ) -> dict:
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE agent_messages
                   SET content = ?,
                       metadata_json = ?
                 WHERE id = ?
                """,
                (content, metadata_json, message_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Agent message not found: {message_id}")
            row = conn.execute(
                """
                SELECT id, task_id, role, stage, content, created_at, metadata_json
                  FROM agent_messages
                 WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        return _row_to_agent_message(row)

    def _merge_report_values(
        self,
        task_id: str,
        values: dict[str, str],
        expected_revision: int,
        *,
        audit: dict | None = None,
    ) -> int:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT report_values_json, report_values_revision
                  FROM tasks
                 WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            current_revision = int(row["report_values_revision"])
            if current_revision != expected_revision:
                raise ConflictError(
                    f"stale report values revision: expected={expected_revision}, "
                    f"server={current_revision}"
                )
            merged = _load_json_dict(row["report_values_json"])
            merged.update(values)
            new_revision = current_revision + 1
            cursor = conn.execute(
                """
                UPDATE tasks
                   SET report_values_json = ?,
                       report_values_revision = ?,
                       updated_at = ?
                 WHERE id = ?
                   AND report_values_revision = ?
                """,
                (
                    _dump_json_dict(merged),
                    new_revision,
                    _now(),
                    task_id,
                    current_revision,
                ),
            )
            if cursor.rowcount == 0:
                raise ConflictError("stale report values revision")
            if audit is not None:
                _write_audit_row(conn, **audit)
        return new_revision


def _row_to_agent_message(row: sqlite3.Row) -> dict:
    metadata = _load_json_object(row["metadata_json"])
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "role": row["role"],
        "stage": row["stage"],
        "content": row["content"],
        "created_at": row["created_at"],
        "metadata": metadata,
    }


def _task_record_from_create(payload: TaskCreate) -> TaskRecord:
    now = _now()
    return TaskRecord(
        id=uuid.uuid4().hex,
        task_type=_normalize_task_type(payload.task_type),
        model_name=payload.model_name,
        model_version=payload.model_version,
        validator=payload.validator,
        source_dir=payload.source_dir,
        algorithm=_normalize_algorithm(payload.algorithm),
        run_mode=_normalize_run_mode(payload.run_mode),
        target_col=payload.target_col,
        score_col=payload.score_col,
        split_col=payload.split_col,
        time_col=payload.time_col,
        feature_columns=list(payload.feature_columns),
        target_type=payload.target_type,
        recipes=list(payload.recipes),
        sample_weight_col=payload.sample_weight_col,
        oot_ks_min=payload.oot_ks_min,
        metrics=list(payload.metrics),
        capability_tier=payload.capability_tier,
        notebook_path=payload.notebook_path,
        sample_path=payload.sample_path,
        pmml_path=payload.pmml_path,
        dictionary_path=payload.dictionary_path,
        report_values_revision=0,
        status=TaskStatus.CREATED,
        status_message="created",
        status_reason_code="",
        created_at=now,
        updated_at=now,
    )


def _insert_task_record_row(
    conn: sqlite3.Connection,
    record: TaskRecord,
    *,
    report_values: dict[str, str] | None,
) -> None:
    conn.execute(
        """
        INSERT INTO tasks
        (
            id, task_type, model_name, model_version, validator, source_dir,
            algorithm, run_mode, target_col, score_col, split_col,
            time_col, feature_columns_json, target_type, recipes_json, sample_weight_col, oot_ks_min, metrics_json, capability_tier, notebook_path, sample_path,
            pmml_path, dictionary_path, report_values_json,
            report_values_revision, status, status_message,
            status_reason_code, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.id,
            record.task_type,
            record.model_name,
            record.model_version,
            record.validator,
            record.source_dir,
            record.algorithm,
            record.run_mode,
            record.target_col,
            record.score_col,
            record.split_col,
            record.time_col,
            _dump_json_list(record.feature_columns),
            record.target_type,
            _dump_json_list(record.recipes),
            record.sample_weight_col,
            record.oot_ks_min,
            _dump_json_list(record.metrics),
            record.capability_tier,
            record.notebook_path,
            record.sample_path,
            record.pmml_path,
            record.dictionary_path,
            _dump_json_dict(report_values or {}),
            record.report_values_revision,
            record.status.value,
            record.status_message,
            record.status_reason_code,
            record.created_at,
            record.updated_at,
        ),
    )


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        id=row["id"],
        task_type=row["task_type"] or TASK_TYPE_VALIDATION,
        model_name=row["model_name"],
        model_version=row["model_version"],
        validator=row["validator"],
        source_dir=row["source_dir"],
        algorithm=row["algorithm"],
        run_mode=row["run_mode"],
        target_col=row["target_col"],
        score_col=row["score_col"],
        split_col=row["split_col"],
        time_col=row["time_col"],
        feature_columns=_load_json_list(row["feature_columns_json"]),
        target_type=(row["target_type"] if "target_type" in row.keys() else "") or "",
        recipes=_load_json_list(row["recipes_json"]),
        sample_weight_col=(row["sample_weight_col"] if "sample_weight_col" in row.keys() else "") or "",
        oot_ks_min=(row["oot_ks_min"] if "oot_ks_min" in row.keys() else None),
        metrics=_load_json_list(row["metrics_json"]),
        capability_tier=(row["capability_tier"] if "capability_tier" in row.keys() else "") or "",
        notebook_path=row["notebook_path"],
        sample_path=row["sample_path"],
        pmml_path=row["pmml_path"],
        dictionary_path=row["dictionary_path"],
        report_values_revision=int(row["report_values_revision"]),
        status=TaskStatus(row["status"]),
        status_message=row["status_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        status_reason_code=row["status_reason_code"],
    )


def _task_purge_summary(conn: sqlite3.Connection, task_id: str) -> dict:
    dataset_rows = conn.execute(
        "SELECT id, source_path FROM datasets WHERE task_id = ?",
        (task_id,),
    ).fetchall()
    dataset_ids = [str(row["id"]) for row in dataset_rows]
    source_paths = [str(row["source_path"]) for row in dataset_rows]
    # A dataset's underlying parquet file may be shared with other tasks (GAP-7
    # content-fingerprint reuse writes multiple dataset rows pointing at the same
    # source_path). Only source paths with zero remaining references after this
    # task's dataset rows are removed are safe for the caller to delete from disk.
    shared_paths: set[str] = set()
    for source_path in set(source_paths):
        (other_count,) = conn.execute(
            "SELECT COUNT(*) FROM datasets WHERE source_path = ? AND task_id != ?",
            (source_path, task_id),
        ).fetchone()
        if other_count:
            shared_paths.add(source_path)
    removable_paths = [path for path in source_paths if path not in shared_paths]

    def _count(table: str, column: str = "task_id") -> int:
        (value,) = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} = ?",
            (task_id,),
        ).fetchone()
        return int(value)

    plan_ids = [
        str(row["id"])
        for row in conn.execute(
            "SELECT id FROM plans WHERE task_id = ?", (task_id,)
        ).fetchall()
    ]
    experiment_ids = [
        str(row["id"])
        for row in conn.execute(
            "SELECT id FROM experiments WHERE task_id = ?", (task_id,)
        ).fetchall()
    ]
    strategy_ids = [
        str(row["id"])
        for row in conn.execute(
            "SELECT id FROM strategies WHERE task_id = ?", (task_id,)
        ).fetchall()
    ]
    model_artifact_count = 0
    if experiment_ids:
        placeholders = ",".join("?" for _ in experiment_ids)
        (model_artifact_count,) = conn.execute(
            f"SELECT COUNT(*) FROM model_artifacts WHERE experiment_id IN ({placeholders})",
            tuple(experiment_ids),
        ).fetchone()
    backtest_count = 0
    if strategy_ids:
        placeholders = ",".join("?" for _ in strategy_ids)
        (backtest_count,) = conn.execute(
            f"SELECT COUNT(*) FROM backtests WHERE strategy_id IN ({placeholders})",
            tuple(strategy_ids),
        ).fetchone()
    plan_step_count = 0
    if plan_ids:
        placeholders = ",".join("?" for _ in plan_ids)
        (plan_step_count,) = conn.execute(
            f"SELECT COUNT(*) FROM plan_steps WHERE plan_id IN ({placeholders})",
            tuple(plan_ids),
        ).fetchone()

    return {
        "datasets": len(dataset_ids),
        "datasets_shared_with_other_tasks": len(shared_paths),
        "joins": _count("joins"),
        "plans": len(plan_ids),
        "plan_steps": int(plan_step_count),
        "experiments": len(experiment_ids),
        "model_artifacts": int(model_artifact_count),
        "strategies": len(strategy_ids),
        "backtests": int(backtest_count),
        "sub_agents": _count("sub_agents", "parent_task_id"),
        "_dataset_source_paths": removable_paths,
    }


def _normalize_run_mode(value: str | None) -> str:
    return "agent" if value == "agent" else "manual"


def _normalize_task_type(value: str | None) -> str:
    # Whitelist known task types; unrecognized or empty values fall back to the
    # default rather than letting arbitrary client-supplied strings persist.
    if value in (None, ""):
        return TASK_TYPE_VALIDATION
    text = str(value)
    return text if text in VALID_TASK_TYPES else TASK_TYPE_VALIDATION


def _normalize_algorithm(value: str | None) -> str:
    return normalize_algorithm(value, allow_empty=True)


def _dump_json_list(values: list[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _dump_json_dict(values: dict[str, str]) -> str:
    _validate_report_values(values)
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _load_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    value = json.loads(raw)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError("feature_columns_json must be a JSON array of strings")
    return value


def _load_json_dict(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("report_values_json must be a JSON object")
    _validate_report_values(value)
    return value


def _load_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        return {}
    return value


def _validate_report_values(values: dict[str, str]) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key.startswith("TEXT:"):
            raise ValueError("report value keys must start with TEXT:")
        if not isinstance(value, str):
            raise ValueError("report values must be strings")


def _reject_computed_report_values(values: dict[str, str]) -> None:
    computed_keys = sorted(set(values) & COMPUTED_REPORT_TEXT_KEYS)
    if computed_keys:
        raise ValueError(
            "platform-computed report values cannot be updated: "
            + ", ".join(computed_keys)
        )


def _expected_status_set(
    expected: TaskStatus | set[TaskStatus] | None,
) -> set[TaskStatus]:
    if expected is None:
        return set(TaskStatus)
    if isinstance(expected, TaskStatus):
        return {expected}
    return set(expected)
