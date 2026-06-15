import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from marvis.domain import (
    TASK_TYPE_VALIDATION,
    VALID_TASK_TYPES,
    TaskCreate,
    TaskRecord,
    TaskStatus,
)
from marvis.model_algorithms import normalize_algorithm
from marvis.report_texts import COMPUTED_REPORT_TEXT_KEYS
from marvis.state_machine import (
    ConflictError,
    IllegalTransition,
    assert_transition,
)

logger = logging.getLogger(__name__)

AGENT_REPORT_CONCLUSION_KEYS = frozenset({
    "TEXT:pressure_test_summary",
    "TEXT:pressure_impact_recommendation",
    "TEXT:final_validation_conclusion",
})
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MIGRATION_TABLES = frozenset({"tasks"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL DEFAULT 'validation',
                    model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                validator TEXT NOT NULL,
                source_dir TEXT NOT NULL,
                algorithm TEXT NOT NULL DEFAULT 'lgb',
                run_mode TEXT NOT NULL DEFAULT 'manual',
                target_col TEXT NOT NULL DEFAULT 'y',
                score_col TEXT NOT NULL DEFAULT 'pred',
                split_col TEXT NOT NULL DEFAULT 'split',
                time_col TEXT NOT NULL DEFAULT 'apply_month',
                status TEXT NOT NULL,
                status_message TEXT NOT NULL,
                status_reason_code TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_column(
            conn,
            table="tasks",
            column="task_type",
            definition="TEXT NOT NULL DEFAULT 'validation'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="algorithm",
            definition="TEXT NOT NULL DEFAULT 'lgb'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="run_mode",
            definition="TEXT NOT NULL DEFAULT 'manual'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="target_col",
            definition="TEXT NOT NULL DEFAULT 'y'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="score_col",
            definition="TEXT NOT NULL DEFAULT 'pred'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="split_col",
            definition="TEXT NOT NULL DEFAULT 'split'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="time_col",
            definition="TEXT NOT NULL DEFAULT 'apply_month'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="feature_columns_json",
            definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="notebook_path",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="sample_path",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="pmml_path",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="dictionary_path",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="report_values_json",
            definition="TEXT NOT NULL DEFAULT '{}'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="report_values_revision",
            definition="INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="status_reason_code",
            definition="TEXT NOT NULL DEFAULT ''",
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_message TEXT NOT NULL DEFAULT '',
                error_name TEXT,
                error_value TEXT,
                traceback TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                log_path TEXT,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_task ON jobs(task_id, kind, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)"
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_active_task
                ON jobs(task_id)
             WHERE status IN ('queued', 'running')
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                role TEXT NOT NULL,
                stage TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_messages_task
                ON agent_messages(task_id, created_at, id)
            """
        )
        from marvis.agent_memory.store import ensure_agent_memory_schema

        ensure_agent_memory_schema(conn)


class TaskRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_task(self, payload: TaskCreate) -> TaskRecord:
        task_id = uuid.uuid4().hex
        now = _now()
        record = TaskRecord(
            id=task_id,
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
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO tasks
                (
                    id, task_type, model_name, model_version, validator, source_dir,
                    algorithm, run_mode, target_col, score_col, split_col,
                    time_col, feature_columns_json, notebook_path, sample_path,
                    pmml_path, dictionary_path, report_values_json,
                    report_values_revision, status, status_message,
                    status_reason_code, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.notebook_path,
                    record.sample_path,
                    record.pmml_path,
                    record.dictionary_path,
                    _dump_json_dict(payload.report_values),
                    record.report_values_revision,
                    record.status.value,
                    record.status_message,
                    record.status_reason_code,
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def get_task(self, task_id: str) -> TaskRecord:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        return _row_to_task(row)

    def update_algorithm(self, task_id: str, algorithm: str) -> TaskRecord:
        normalized = normalize_algorithm(algorithm)
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

    def list_tasks(self) -> list[TaskRecord]:
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
        return [_row_to_task(row) for row in rows]

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        message: str,
        *,
        expected: TaskStatus | set[TaskStatus] | None = None,
        reason_code: str = "",
    ) -> None:
        expected_set = _expected_status_set(expected)
        with connect(self.db_path) as conn:
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
            if self.task_has_active_job(task_id):
                raise ConflictError(
                    f"task {task_id} already has an active job"
                ) from exc
            raise
        return job_id

    def mark_job_running(self, job_id: str) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                   SET status = 'running',
                       started_at = COALESCE(started_at, ?)
                 WHERE id = ?
                """,
                (_now(), job_id),
            )

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

    def list_agent_messages(self, task_id: str) -> list[dict]:
        with connect(self.db_path) as conn:
            task_row = conn.execute(
                "SELECT 1 FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if task_row is None:
                raise KeyError(f"Task not found: {task_id}")
            rows = conn.execute(
                """
                SELECT id, task_id, role, stage, content, created_at, metadata_json
                  FROM agent_messages
                 WHERE task_id = ?
                 ORDER BY created_at ASC, id ASC
                """,
                (task_id,),
            ).fetchall()
        return [_row_to_agent_message(row) for row in rows]

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


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    table_sql = _migration_table_identifier(table)
    column_sql = _sql_identifier(column)
    existing_columns = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_sql})").fetchall()
    }
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table_sql} ADD COLUMN {column_sql} {definition}")


def _migration_table_identifier(table: str) -> str:
    if table not in _MIGRATION_TABLES:
        raise ValueError(f"unsupported migration table: {table}")
    return _sql_identifier(table)


def _sql_identifier(identifier: str) -> str:
    if not _SQL_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def _expected_status_set(
    expected: TaskStatus | set[TaskStatus] | None,
) -> set[TaskStatus]:
    if expected is None:
        return set(TaskStatus)
    if isinstance(expected, TaskStatus):
        return {expected}
    return set(expected)


def _configure_connection(conn: sqlite3.Connection) -> None:
    mode_row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    # WAL is requested for concurrent readers/writers. It silently degrades on
    # read-only or networked filesystems; surface that instead of assuming the
    # concurrency guarantees hold. In-memory databases legitimately report
    # "memory" and are exempt.
    if mode_row is not None and str(mode_row[0]).lower() not in ("wal", "memory"):
        logger.warning("Failed to enable WAL journal mode; got %r", mode_row[0])
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")


@contextmanager
def connect(db_path: Path):
    conn = sqlite3.connect(db_path, timeout=5.0, isolation_level="DEFERRED")
    conn.row_factory = sqlite3.Row
    try:
        _configure_connection(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
