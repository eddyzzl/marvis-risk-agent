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
from marvis.plugins.errors import PluginNotFoundError
from marvis.plugins.manifest import PluginManifest, ToolRef, manifest_to_dict
from marvis.orchestrator.contracts import (
    AgentStatus,
    Plan,
    PlanStatus,
    SubAgent,
    plan_from_dict,
    plan_to_dict,
)
from marvis.orchestrator.errors import PlanNotFoundError
from marvis.orchestrator.harness_state import assert_plan_transition
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plugins (
                name TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                display_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                module TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                checksum TEXT NOT NULL DEFAULT '',
                builtin INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                installed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                plugin TEXT NOT NULL,
                name TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                input_schema_json TEXT NOT NULL,
                output_schema_json TEXT NOT NULL,
                determinism TEXT NOT NULL,
                timeout_seconds INTEGER NOT NULL,
                failure_policy TEXT NOT NULL,
                side_effects_json TEXT NOT NULL DEFAULT '[]',
                entrypoint TEXT NOT NULL DEFAULT '',
                memory_limit_mb INTEGER NOT NULL DEFAULT 2048,
                PRIMARY KEY (plugin, name),
                FOREIGN KEY(plugin) REFERENCES plugins(name) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                actor TEXT,
                target_ref TEXT,
                inputs_hash TEXT,
                outcome TEXT,
                detail_json TEXT NOT NULL DEFAULT '{}',
                at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                source TEXT NOT NULL,
                template_id TEXT,
                autonomy_level INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_steps (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                title TEXT NOT NULL,
                tool_plugin TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_version TEXT,
                inputs_json TEXT NOT NULL,
                depends_on_json TEXT NOT NULL,
                post_checks_json TEXT NOT NULL,
                needs_confirmation INTEGER NOT NULL,
                sub_agent_scope TEXT,
                granted_tools_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                sub_agent_id TEXT,
                output_ref TEXT,
                review_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                confirmed INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_step_outputs (
                step_id TEXT PRIMARY KEY,
                output_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(step_id) REFERENCES plan_steps(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sub_agents (
                id TEXT PRIMARY KEY,
                parent_task_id TEXT NOT NULL,
                parent_step_id TEXT,
                scope TEXT NOT NULL,
                granted_tools_json TEXT NOT NULL DEFAULT '[]',
                context_budget INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_ref TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tools_plugin ON tools(plugin)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_kind_at ON audit(kind, at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps(plan_id)")
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


class PlanRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_plan(self, plan: Plan) -> None:
        payload = plan_to_dict(plan)
        now = _now()
        created_at = payload.get("created_at") or now
        updated_at = payload.get("updated_at") or now
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO plans(
                    id, task_id, goal, source, template_id, autonomy_level,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["task_id"],
                    payload["goal"],
                    payload["source"],
                    payload["template_id"],
                    payload["autonomy_level"],
                    payload["status"],
                    created_at,
                    updated_at,
                ),
            )
            for step in payload["steps"]:
                self._insert_step(conn, step)
            _write_audit_row(
                conn,
                kind="plan.create",
                target_ref=plan.id,
                outcome="succeeded",
                detail={"task_id": plan.task_id, "step_count": len(plan.steps)},
            )

    def load_plan(self, plan_id: str) -> Plan:
        with connect(self.db_path) as conn:
            plan_row = conn.execute(
                """
                SELECT id, task_id, goal, source, template_id, autonomy_level,
                       status, created_at, updated_at
                  FROM plans
                 WHERE id = ?
                """,
                (plan_id,),
            ).fetchone()
            if plan_row is None:
                raise PlanNotFoundError(plan_id)
            step_rows = conn.execute(
                """
                SELECT id, plan_id, idx, title, tool_plugin, tool_name, tool_version,
                       inputs_json, depends_on_json, post_checks_json,
                       needs_confirmation, sub_agent_scope, granted_tools_json,
                       status, sub_agent_id, output_ref, review_json, error
                  FROM plan_steps
                 WHERE plan_id = ?
                 ORDER BY idx, id
                """,
                (plan_id,),
            ).fetchall()
        return plan_from_dict(_plan_payload_from_rows(plan_row, step_rows))

    def update_step(self, step) -> None:
        payload = plan_to_dict(
            Plan(
                id=step.plan_id,
                task_id="",
                goal="",
                source="template",
                template_id=None,
                steps=[step],
                autonomy_level=0,
            )
        )["steps"][0]
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE plan_steps
                   SET idx = ?,
                       title = ?,
                       tool_plugin = ?,
                       tool_name = ?,
                       tool_version = ?,
                       inputs_json = ?,
                       depends_on_json = ?,
                       post_checks_json = ?,
                       needs_confirmation = ?,
                       sub_agent_scope = ?,
                       granted_tools_json = ?,
                       status = ?,
                       sub_agent_id = ?,
                       output_ref = ?,
                       review_json = ?,
                       error = ?
                 WHERE id = ?
                """,
                _step_update_values(payload),
            )
            if cursor.rowcount == 0:
                raise KeyError(step.id)

    def set_plan_status(self, plan_id: str, status: PlanStatus) -> None:
        current = self.load_plan(plan_id).status
        assert_plan_transition(current, status)
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE plans SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, _now(), plan_id),
            )
            _write_audit_row(
                conn,
                kind="plan.status",
                target_ref=plan_id,
                outcome="succeeded",
                detail={"from": current.value, "to": status.value},
            )

    def confirm_plan(self, plan_id: str) -> None:
        self.set_plan_status(plan_id, PlanStatus.CONFIRMED)

    def confirm_step(self, step_id: str) -> None:
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE plan_steps SET confirmed = 1 WHERE id = ?",
                (step_id,),
            )
            if cursor.rowcount == 0:
                raise KeyError(step_id)
            _write_audit_row(
                conn,
                kind="plan.step.confirm",
                target_ref=step_id,
                outcome="succeeded",
            )

    def is_step_confirmed(self, step_id: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT confirmed FROM plan_steps WHERE id = ?",
                (step_id,),
            ).fetchone()
        if row is None:
            raise KeyError(step_id)
        return bool(row["confirmed"])

    def store_step_output(self, step_id: str, output: dict) -> str:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO plan_step_outputs(step_id, output_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(step_id) DO UPDATE SET
                    output_json = excluded.output_json,
                    created_at = excluded.created_at
                """,
                (step_id, _dump_json_any(output), _now()),
            )
        return f"metrics:{step_id}"

    def load_step_output(self, step_id: str) -> dict:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT output_json FROM plan_step_outputs WHERE step_id = ?",
                (step_id,),
            ).fetchone()
        if row is None:
            raise KeyError(step_id)
        value = json.loads(row["output_json"])
        return value if isinstance(value, dict) else {}

    def upsert_sub_agent(self, sub: SubAgent) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sub_agents(
                    id, parent_task_id, parent_step_id, scope, granted_tools_json,
                    context_budget, status, result_ref, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    parent_task_id = excluded.parent_task_id,
                    parent_step_id = excluded.parent_step_id,
                    scope = excluded.scope,
                    granted_tools_json = excluded.granted_tools_json,
                    context_budget = excluded.context_budget,
                    status = excluded.status,
                    result_ref = excluded.result_ref
                """,
                (
                    sub.id,
                    sub.parent_task_id,
                    sub.parent_step_id,
                    sub.scope,
                    _dump_json_any([_tool_ref_to_dict(ref) for ref in sub.granted_tools]),
                    sub.context_budget,
                    sub.status.value,
                    sub.result_ref,
                    _now(),
                ),
            )

    def set_sub_agent_status(
        self,
        sub_id: str,
        status: AgentStatus,
        *,
        result_ref: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE sub_agents
                   SET status = ?,
                       result_ref = COALESCE(?, result_ref)
                 WHERE id = ?
                """,
                (status.value, result_ref, sub_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(sub_id)

    def get_sub_agent(self, sub_id: str) -> SubAgent:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, parent_task_id, parent_step_id, scope, granted_tools_json,
                       context_budget, status, result_ref
                  FROM sub_agents
                 WHERE id = ?
                """,
                (sub_id,),
            ).fetchone()
        if row is None:
            raise KeyError(sub_id)
        return SubAgent(
            id=row["id"],
            parent_task_id=row["parent_task_id"],
            parent_step_id=row["parent_step_id"],
            scope=row["scope"],
            granted_tools=[
                _tool_ref_from_dict(item)
                for item in _load_json_array(row["granted_tools_json"])
            ],
            context_budget=int(row["context_budget"]),
            status=AgentStatus(row["status"]),
            result_ref=row["result_ref"],
        )

    def write_audit(self, **kwargs) -> None:
        with connect(self.db_path) as conn:
            _write_audit_row(conn, **kwargs)

    def list_audit(self, *, kind: str | None = None) -> list[dict]:
        return _list_audit_rows(self.db_path, kind=kind)

    def _insert_step(self, conn: sqlite3.Connection, step: dict) -> None:
        conn.execute(
            """
            INSERT INTO plan_steps(
                id, plan_id, idx, title, tool_plugin, tool_name, tool_version,
                inputs_json, depends_on_json, post_checks_json,
                needs_confirmation, sub_agent_scope, granted_tools_json,
                status, sub_agent_id, output_ref, review_json, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _step_insert_values(step),
        )


class PluginRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def upsert_plugin(self, manifest: PluginManifest, *, enabled: bool) -> None:
        manifest_json = json.dumps(
            manifest_to_dict(manifest),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        now = _now()
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO plugins(
                    name, version, display_name, description, module,
                    manifest_json, checksum, builtin, enabled, installed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    version = excluded.version,
                    display_name = excluded.display_name,
                    description = excluded.description,
                    module = excluded.module,
                    manifest_json = excluded.manifest_json,
                    checksum = excluded.checksum,
                    builtin = excluded.builtin,
                    enabled = excluded.enabled,
                    installed_at = excluded.installed_at
                """,
                (
                    manifest.name,
                    manifest.version,
                    manifest.display_name,
                    manifest.description,
                    manifest.module,
                    manifest_json,
                    manifest.checksum,
                    int(manifest.builtin),
                    int(enabled),
                    now,
                ),
            )
            conn.execute("DELETE FROM tools WHERE plugin = ?", (manifest.name,))
            for tool in manifest.tools:
                conn.execute(
                    """
                    INSERT INTO tools(
                        plugin, name, summary, input_schema_json,
                        output_schema_json, determinism, timeout_seconds,
                        failure_policy, side_effects_json, entrypoint,
                        memory_limit_mb
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        manifest.name,
                        tool.name,
                        tool.summary,
                        json.dumps(tool.input_schema, ensure_ascii=False, separators=(",", ":")),
                        json.dumps(tool.output_schema, ensure_ascii=False, separators=(",", ":")),
                        tool.determinism,
                        tool.timeout_seconds,
                        tool.failure_policy,
                        json.dumps(list(tool.side_effects), ensure_ascii=False, separators=(",", ":")),
                        tool.entrypoint,
                        tool.memory_limit_mb,
                    ),
                )

    def set_enabled(self, name: str, enabled: bool) -> None:
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE plugins SET enabled = ? WHERE name = ?",
                (int(enabled), name),
            )
            if cursor.rowcount == 0:
                raise PluginNotFoundError(name)

    def delete_plugin(self, name: str) -> None:
        with connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM plugins WHERE name = ?", (name,))
            if cursor.rowcount == 0:
                raise PluginNotFoundError(name)

    def get_plugin(self, name: str) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT p.*, COUNT(t.name) AS tool_count
                  FROM plugins p
                  LEFT JOIN tools t ON t.plugin = p.name
                 WHERE p.name = ?
                 GROUP BY p.name
                """,
                (name,),
            ).fetchone()
        return _plugin_row_to_dict(row) if row is not None else None

    def list_plugins(self, *, include_disabled: bool = False) -> list[dict]:
        where = "" if include_disabled else "WHERE p.enabled = 1"
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT p.*, COUNT(t.name) AS tool_count
                  FROM plugins p
                  LEFT JOIN tools t ON t.plugin = p.name
                  {where}
                 GROUP BY p.name
                 ORDER BY p.name
                """
            ).fetchall()
        return [_plugin_row_to_dict(row) for row in rows]

    def list_tools(self) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT plugin, name, summary, input_schema_json,
                       output_schema_json, determinism, timeout_seconds,
                       failure_policy, side_effects_json, entrypoint,
                       memory_limit_mb
                  FROM tools
                 ORDER BY plugin, name
                """
            ).fetchall()
        return [_tool_row_to_dict(row) for row in rows]

    def write_audit(
        self,
        *,
        kind: str,
        target_ref: str,
        actor: str = "system",
        inputs_hash: str | None = None,
        outcome: str | None = None,
        detail: dict | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit(
                    id, kind, actor, target_ref, inputs_hash, outcome,
                    detail_json, at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    kind,
                    actor,
                    target_ref,
                    inputs_hash,
                    outcome,
                    json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
                    _now(),
                ),
            )

    def list_audit(self, *, kind: str | None = None) -> list[dict]:
        query = (
            "SELECT id, kind, actor, target_ref, inputs_hash, outcome, detail_json, at "
            "FROM audit"
        )
        params: tuple[str, ...] = ()
        if kind is not None:
            query += " WHERE kind = ?"
            params = (kind,)
        query += " ORDER BY at, id"
        with connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [_audit_row_to_dict(row) for row in rows]


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


def _plugin_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "name": row["name"],
        "version": row["version"],
        "display_name": row["display_name"],
        "description": row["description"],
        "module": row["module"],
        "manifest_json": row["manifest_json"],
        "checksum": row["checksum"],
        "builtin": bool(row["builtin"]),
        "enabled": bool(row["enabled"]),
        "installed_at": row["installed_at"],
        "tool_count": int(row["tool_count"]),
    }


def _tool_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "plugin": row["plugin"],
        "name": row["name"],
        "summary": row["summary"],
        "input_schema_json": row["input_schema_json"],
        "output_schema_json": row["output_schema_json"],
        "determinism": row["determinism"],
        "timeout_seconds": int(row["timeout_seconds"]),
        "failure_policy": row["failure_policy"],
        "side_effects_json": row["side_effects_json"],
        "entrypoint": row["entrypoint"],
        "memory_limit_mb": int(row["memory_limit_mb"]),
    }


def _audit_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "actor": row["actor"],
        "target_ref": row["target_ref"],
        "inputs_hash": row["inputs_hash"],
        "outcome": row["outcome"],
        "detail": _load_json_object(row["detail_json"]),
        "at": row["at"],
    }


def _plan_payload_from_rows(plan_row: sqlite3.Row, step_rows: list[sqlite3.Row]) -> dict:
    return {
        "id": plan_row["id"],
        "task_id": plan_row["task_id"],
        "goal": plan_row["goal"],
        "source": plan_row["source"],
        "template_id": plan_row["template_id"],
        "steps": [_step_payload_from_row(row) for row in step_rows],
        "autonomy_level": int(plan_row["autonomy_level"]),
        "status": plan_row["status"],
        "created_at": plan_row["created_at"],
        "updated_at": plan_row["updated_at"],
    }


def _step_payload_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "plan_id": row["plan_id"],
        "index": int(row["idx"]),
        "title": row["title"],
        "tool_ref": {
            "plugin": row["tool_plugin"],
            "tool": row["tool_name"],
            "version": row["tool_version"] or "",
        },
        "inputs": _load_json_object_unchecked(row["inputs_json"]),
        "depends_on": _load_json_array(row["depends_on_json"]),
        "post_checks": _load_json_array(row["post_checks_json"]),
        "needs_confirmation": bool(row["needs_confirmation"]),
        "sub_agent_scope": row["sub_agent_scope"],
        "granted_tools": _load_json_array(row["granted_tools_json"]),
        "status": row["status"],
        "sub_agent_id": row["sub_agent_id"],
        "output_ref": row["output_ref"],
        "review_verdicts": _load_json_array(row["review_json"]),
        "error": row["error"],
    }


def _step_insert_values(step: dict) -> tuple:
    tool_ref = step["tool_ref"]
    return (
        step["id"],
        step["plan_id"],
        step["index"],
        step["title"],
        tool_ref["plugin"],
        tool_ref["tool"],
        tool_ref.get("version") or "",
        _dump_json_any(step.get("inputs") or {}),
        _dump_json_any(step.get("depends_on") or []),
        _dump_json_any(step.get("post_checks") or []),
        int(bool(step.get("needs_confirmation"))),
        step.get("sub_agent_scope"),
        _dump_json_any(step.get("granted_tools") or []),
        step["status"],
        step.get("sub_agent_id"),
        step.get("output_ref"),
        _dump_json_any(step.get("review_verdicts") or []),
        step.get("error"),
    )


def _step_update_values(step: dict) -> tuple:
    return (*_step_insert_values(step)[2:], step["id"])


def _tool_ref_to_dict(ref: ToolRef) -> dict[str, str]:
    return {"plugin": ref.plugin, "tool": ref.tool, "version": ref.version}


def _tool_ref_from_dict(payload: dict) -> ToolRef:
    return ToolRef(
        plugin=str(payload["plugin"]),
        tool=str(payload["tool"]),
        version=str(payload.get("version") or ""),
    )


def _dump_json_any(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_object_unchecked(raw: str | None) -> dict:
    if not raw:
        return {}
    value = json.loads(raw)
    return value if isinstance(value, dict) else {}


def _load_json_array(raw: str | None) -> list:
    if not raw:
        return []
    value = json.loads(raw)
    return value if isinstance(value, list) else []


def _write_audit_row(
    conn: sqlite3.Connection,
    *,
    kind: str,
    target_ref: str,
    actor: str = "system",
    inputs_hash: str | None = None,
    outcome: str | None = None,
    detail: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audit(
            id, kind, actor, target_ref, inputs_hash, outcome,
            detail_json, at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            kind,
            actor,
            target_ref,
            inputs_hash,
            outcome,
            json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
            _now(),
        ),
    )


def _list_audit_rows(db_path: Path, *, kind: str | None = None) -> list[dict]:
    query = (
        "SELECT id, kind, actor, target_ref, inputs_hash, outcome, detail_json, at "
        "FROM audit"
    )
    params: tuple[str, ...] = ()
    if kind is not None:
        query += " WHERE kind = ?"
        params = (kind,)
    query += " ORDER BY at, id"
    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [_audit_row_to_dict(row) for row in rows]


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
