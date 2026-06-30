import json
import sqlite3
import uuid
from dataclasses import asdict, is_dataclass, replace as dataclass_replace
from datetime import UTC, datetime
from pathlib import Path

from marvis.data.contracts import (
    ColumnFingerprint,
    ColumnProfile,
    Dataset,
    JoinDiagnostics,
    JoinPlan,
    JoinSpec,
    KeyPair,
)
from marvis.domain import (
    TASK_TYPE_VALIDATION,
    VALID_TASK_TYPES,
    TaskCreate,
    TaskRecord,
    TaskStatus,
)
from marvis.db_schema import (
    _ensure_column as _ensure_column,
    connect as connect,
    init_db as init_db,
    sqlite_health as sqlite_health,
)
from marvis.agent.gates.contracts import EvidenceEnvelope
from marvis.model_algorithms import normalize_algorithm
from marvis.packs.modeling.contracts import (
    Experiment,
    ModelArtifact,
    ModelMetrics,
    TrainConfig,
)
from marvis.plugins.manifest import ToolRef
from marvis.redaction import redact_value
from marvis.repositories.drafts import DraftRepository as DraftRepository  # noqa: F401
from marvis.repositories.plugins import PluginRepository as PluginRepository  # noqa: F401
from marvis.repositories.strategy import StrategyRepository as StrategyRepository  # noqa: F401
from marvis.orchestrator.contracts import (
    AgentStatus,
    Plan,
    PlanStatus,
    StepStatus,
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

    def create_task(self, payload: TaskCreate) -> TaskRecord:
        record = _task_record_from_create(payload)
        with connect(self.db_path) as conn:
            _insert_task_record_row(conn, record, report_values=payload.report_values)
        return record

    def create_task_with_audit(self, payload: TaskCreate, *, audit_factory) -> TaskRecord:
        record = _task_record_from_create(payload)
        audit = audit_factory(record)
        with connect(self.db_path) as conn:
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
        record = _task_record_from_create(payload)
        audit = audit_factory(record)
        with connect(self.db_path) as conn:
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
                    status, novel_mode, tier, replan_count, loop_events_json,
                    success_criteria_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["task_id"],
                    payload["goal"],
                    payload["source"],
                    payload["template_id"],
                    payload["autonomy_level"],
                    payload["status"],
                    payload["novel_mode"],
                    payload["tier"],
                    payload["replan_count"],
                    _dump_json_any(payload["loop_events"]),
                    _dump_json_any(payload["success_criteria"]),
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
                       status, novel_mode, tier, replan_count, loop_events_json,
                       success_criteria_json, created_at, updated_at
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
                       needs_confirmation, decision_point, sub_agent_scope, granted_tools_json,
                       status, sub_agent_id, output_ref, review_json, error, phase
                  FROM plan_steps
                 WHERE plan_id = ?
                 ORDER BY idx, id
                """,
                (plan_id,),
            ).fetchall()
        return plan_from_dict(_plan_payload_from_rows(plan_row, step_rows))

    def list_plans_for_task(self, task_id: str) -> list[Plan]:
        """All plans for a task, oldest first. Used to resume/reload a task's
        plan in the right rail (create returns the plan_id for first build)."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id FROM plans WHERE task_id = ? ORDER BY created_at, id",
                (task_id,),
            ).fetchall()
        return [self.load_plan(row["id"]) for row in rows]

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
                       decision_point = ?,
                       sub_agent_scope = ?,
                       granted_tools_json = ?,
                       status = ?,
                       sub_agent_id = ?,
                       output_ref = ?,
                       review_json = ?,
                       error = ?,
                       phase = ?
                 WHERE id = ?
                """,
                _step_update_values(payload),
            )
            if cursor.rowcount == 0:
                raise KeyError(step.id)

    def set_plan_status(self, plan_id: str, status: PlanStatus) -> None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if row is None:
                raise PlanNotFoundError(plan_id)
            current = PlanStatus(str(row["status"]))
            assert_plan_transition(current, status)
            cursor = conn.execute(
                "UPDATE plans SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (status.value, _now(), plan_id, current.value),
            )
            if cursor.rowcount == 0:
                raise ConflictError(f"plan {plan_id} changed while updating status")
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
                """
                UPDATE plan_steps
                   SET confirmed = 1
                 WHERE id = ?
                   AND status = ?
                """,
                (step_id, StepStatus.AWAITING_CONFIRM.value),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT status FROM plan_steps WHERE id = ?",
                    (step_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(step_id)
                raise ConflictError(
                    f"step is not awaiting confirmation: {row['status']}"
                )
            _write_audit_row(
                conn,
                kind="plan.step.confirm",
                target_ref=step_id,
                outcome="succeeded",
            )

    def reset_step(self, step_id: str, *, inputs: dict | None = None) -> None:
        """Reset a step to pending and clear its output / error / confirmation so it
        can run again (the gate-adjust path). Optionally replace its inputs with a
        parameter override. Used to re-run an analysis step with new parameters when
        the user asks for an adjustment at a gate."""
        with connect(self.db_path) as conn:
            if inputs is not None:
                conn.execute(
                    "UPDATE plan_steps SET inputs_json = ? WHERE id = ?",
                    (json.dumps(inputs, ensure_ascii=False), step_id),
                )
            cursor = conn.execute(
                "UPDATE plan_steps SET status = 'pending', confirmed = 0, "
                "output_ref = NULL, review_json = '[]', error = NULL WHERE id = ?",
                (step_id,),
            )
            if cursor.rowcount == 0:
                raise KeyError(step_id)
            _write_audit_row(
                conn,
                kind="plan.step.reset",
                target_ref=step_id,
                outcome="succeeded",
            )

    def retry_failed_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        inputs: dict | None = None,
    ) -> list[str]:
        """Explicitly retry a failed step and every downstream dependent step."""
        with connect(self.db_path) as conn:
            plan_row = conn.execute(
                "SELECT status FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if plan_row is None:
                raise PlanNotFoundError(plan_id)
            if str(plan_row["status"]) != PlanStatus.FAILED.value:
                raise ConflictError(f"plan is not failed: {plan_row['status']}")

            step_rows = conn.execute(
                """
                SELECT id, idx, depends_on_json, status
                  FROM plan_steps
                 WHERE plan_id = ?
                 ORDER BY idx, id
                """,
                (plan_id,),
            ).fetchall()
            rows_by_id = {str(row["id"]): row for row in step_rows}
            target = rows_by_id.get(step_id)
            if target is None:
                raise KeyError(step_id)
            if str(target["status"]) != StepStatus.FAILED.value:
                raise ConflictError(f"step is not failed: {target['status']}")

            reset_ids = {step_id}
            changed = True
            while changed:
                changed = False
                for row in step_rows:
                    row_id = str(row["id"])
                    if row_id in reset_ids:
                        continue
                    depends_on = {str(item) for item in _load_json_array(row["depends_on_json"])}
                    if depends_on.intersection(reset_ids):
                        reset_ids.add(row_id)
                        changed = True

            ordered_reset_ids = [
                str(row["id"]) for row in step_rows if str(row["id"]) in reset_ids
            ]
            for reset_id in ordered_reset_ids:
                if reset_id == step_id and inputs is not None:
                    conn.execute(
                        """
                        UPDATE plan_steps
                           SET inputs_json = ?,
                               status = 'pending',
                               confirmed = 0,
                               sub_agent_id = NULL,
                               output_ref = NULL,
                               review_json = '[]',
                               error = NULL
                         WHERE id = ?
                        """,
                        (_dump_json_any(inputs), reset_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE plan_steps
                           SET status = 'pending',
                               confirmed = 0,
                               sub_agent_id = NULL,
                               output_ref = NULL,
                               review_json = '[]',
                               error = NULL
                         WHERE id = ?
                        """,
                        (reset_id,),
                    )

            cursor = conn.execute(
                "UPDATE plans SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (PlanStatus.RUNNING.value, _now(), plan_id, PlanStatus.FAILED.value),
            )
            if cursor.rowcount == 0:
                raise ConflictError(f"plan {plan_id} changed while retrying step")
            _write_audit_row(
                conn,
                kind="plan.step.retry",
                target_ref=step_id,
                outcome="succeeded",
                detail={
                    "plan_id": plan_id,
                    "reset_step_ids": ordered_reset_ids,
                    "inputs_replaced": inputs is not None,
                },
            )
            return ordered_reset_ids

    def is_step_confirmed(self, step_id: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT confirmed FROM plan_steps WHERE id = ?",
                (step_id,),
            ).fetchone()
        if row is None:
            raise KeyError(step_id)
        return bool(row["confirmed"])

    def start_step_run(self, *, plan_id: str, step_id: str, tool_ref: str, inputs: dict) -> str:
        run_id = uuid.uuid4().hex
        now = _now()
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(attempt), 0) + 1 AS next_attempt FROM plan_step_runs WHERE step_id = ?",
                (step_id,),
            ).fetchone()
            attempt = int(row["next_attempt"] if row is not None else 1)
            conn.execute(
                """
                INSERT INTO plan_step_runs(
                    id, plan_id, step_id, attempt, tool_ref, status, input_json, started_at
                )
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (run_id, plan_id, step_id, attempt, tool_ref, _dump_json_any(inputs), now),
            )
        return run_id

    def finish_step_run(
        self,
        run_id: str,
        *,
        status: str,
        output_ref: str | None = None,
        error: str | None = None,
        error_kind: str | None = None,
        duration_ms: int | None = None,
        side_effects: list | None = None,
    ) -> None:
        if status not in {"succeeded", "failed", "interrupted"}:
            raise ValueError(f"unsupported step run status: {status}")
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE plan_step_runs
                   SET status = ?,
                       output_ref = ?,
                       error = ?,
                       error_kind = ?,
                       duration_ms = ?,
                       side_effects_json = ?,
                       finished_at = ?
                 WHERE id = ?
                   AND status = 'running'
                """,
                (
                    status,
                    output_ref,
                    error,
                    error_kind,
                    duration_ms,
                    _dump_json_any(side_effects or []),
                    _now(),
                    run_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(run_id)

    def list_step_runs(self, step_id: str) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                  FROM plan_step_runs
                 WHERE step_id = ?
                 ORDER BY attempt ASC
                """,
                (step_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["input"] = _load_json_object(item.pop("input_json", "{}"))
            item["side_effects"] = _load_json_array(item.pop("side_effects_json", "[]"))
            result.append(item)
        return result

    def latest_failed_step_run_error_kind(self, step_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT error_kind
                  FROM plan_step_runs
                 WHERE step_id = ?
                   AND status IN ('failed', 'interrupted')
                 ORDER BY attempt DESC, COALESCE(finished_at, started_at) DESC
                 LIMIT 1
                """,
                (step_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["error_kind"] or "").strip() or None

    def list_running_step_runs(self, plan_id: str) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                  FROM plan_step_runs
                 WHERE plan_id = ?
                   AND status = 'running'
                 ORDER BY started_at ASC, attempt ASC
                """,
                (plan_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["input"] = _load_json_object(item.pop("input_json", "{}"))
            item["side_effects"] = _load_json_array(item.pop("side_effects_json", "[]"))
            result.append(item)
        return result

    def latest_succeeded_step_run_output_ref(self, step_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT output_ref
                  FROM plan_step_runs
                 WHERE step_id = ?
                   AND status = 'succeeded'
                   AND output_ref IS NOT NULL
                 ORDER BY attempt DESC, finished_at DESC
                 LIMIT 1
                """,
                (step_id,),
            ).fetchone()
        return None if row is None else str(row["output_ref"] or "") or None

    def store_step_output(self, step_id: str, output: dict, *, evidence: dict | EvidenceEnvelope | None = None) -> str:
        now = _now()
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS next_version "
                "FROM plan_step_output_versions WHERE step_id = ?",
                (step_id,),
            ).fetchone()
            version = int(row["next_version"] if row is not None else 1)
            output_ref = f"metrics:{step_id}:v{version}"
            safe_output = redact_value(output)
            evidence_payload = _step_evidence_payload(output_ref, evidence)
            safe_evidence = redact_value(evidence_payload)
            total_redacted = int(safe_output.redacted_count) + int(safe_evidence.redacted_count)
            evidence_value = safe_evidence.value
            if total_redacted and isinstance(evidence_value, dict):
                evidence_value["persistence_redacted_count"] = int(
                    evidence_value.get("persistence_redacted_count") or 0
                ) + total_redacted
            output_json = _dump_json_any(safe_output.value)
            evidence_json = _dump_json_any(evidence_value)
            conn.execute(
                """
                INSERT INTO plan_step_output_versions(step_id, version, output_json, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (step_id, version, output_json, evidence_json, now),
            )
            conn.execute(
                """
                INSERT INTO plan_step_outputs(step_id, output_json, evidence_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(step_id) DO UPDATE SET
                    output_json = excluded.output_json,
                    evidence_json = excluded.evidence_json,
                    created_at = excluded.created_at
                """,
                (step_id, output_json, evidence_json, now),
            )
        return output_ref

    def load_step_output(self, step_id: str, *, version: int | None = None) -> dict:
        with connect(self.db_path) as conn:
            if version is None:
                row = conn.execute(
                    "SELECT output_json FROM plan_step_outputs WHERE step_id = ?",
                    (step_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT output_json FROM plan_step_output_versions WHERE step_id = ? AND version = ?",
                    (step_id, int(version)),
                ).fetchone()
        if row is None:
            raise KeyError(step_id)
        value = json.loads(row["output_json"])
        return value if isinstance(value, dict) else {}

    def load_step_evidence(self, step_id: str, *, version: int | None = None) -> dict:
        with connect(self.db_path) as conn:
            if version is None:
                row = conn.execute(
                    "SELECT evidence_json FROM plan_step_outputs WHERE step_id = ?",
                    (step_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT evidence_json FROM plan_step_output_versions WHERE step_id = ? AND version = ?",
                    (step_id, int(version)),
                ).fetchone()
        if row is None:
            raise KeyError(step_id)
        value = json.loads(row["evidence_json"] or "{}")
        evidence = value if isinstance(value, dict) else {}
        if not evidence.get("output_ref"):
            if version is None:
                output_ref = self.latest_step_output_ref(step_id) or f"metrics:{step_id}"
            else:
                output_ref = f"metrics:{step_id}:v{int(version)}"
            evidence = EvidenceEnvelope(output_ref=output_ref).to_dict()
        return evidence

    def latest_step_output_ref(self, step_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(version) AS version FROM plan_step_output_versions WHERE step_id = ?",
                (step_id,),
            ).fetchone()
        if row is None or row["version"] is None:
            return None
        return f"metrics:{step_id}:v{int(row['version'])}"

    def replace_remaining_steps(
        self,
        plan_id: str,
        new_plan: Plan,
        *,
        loop_event: dict | None = None,
    ) -> None:
        payload = plan_to_dict(new_plan)
        with connect(self.db_path) as conn:
            loop_events = _load_plan_loop_events(conn, plan_id)
            _append_normalized_loop_event(loop_events, loop_event)
            completed_rows = conn.execute(
                """
                SELECT id
                  FROM plan_steps
                 WHERE plan_id = ?
                   AND status IN ('done', 'skipped')
                """,
                (plan_id,),
            ).fetchall()
            completed_ids = {row["id"] for row in completed_rows}
            conn.execute(
                """
                DELETE FROM plan_steps
                 WHERE plan_id = ?
                   AND status NOT IN ('done', 'skipped')
                """,
                (plan_id,),
            )
            for step in payload["steps"]:
                if step["id"] in completed_ids or step["status"] in {"done", "skipped"}:
                    continue
                step["plan_id"] = plan_id
                self._insert_step(conn, step)
            conn.execute(
                """
                UPDATE plans
                   SET replan_count = replan_count + 1,
                       tier = ?,
                       novel_mode = ?,
                       loop_events_json = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    payload["tier"],
                    payload["novel_mode"],
                    _dump_json_any(loop_events),
                    _now(),
                    plan_id,
                ),
            )
            _write_audit_row(
                conn,
                kind="plan.replan",
                target_ref=plan_id,
                outcome="succeeded",
                detail={"step_count": len(payload["steps"])},
            )

    def append_steps(
        self,
        plan_id: str,
        steps: list,
        *,
        loop_event: dict | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            loop_events = _load_plan_loop_events(conn, plan_id)
            _append_normalized_loop_event(loop_events, loop_event)
            row = conn.execute(
                "SELECT COALESCE(MAX(idx), -1) AS max_idx FROM plan_steps WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
            next_index = int(row["max_idx"]) + 1
            for offset, step in enumerate(steps):
                normalized = dataclass_replace(
                    step,
                    plan_id=plan_id,
                    index=next_index + offset,
                )
                payload = plan_to_dict(
                    Plan(
                        id=plan_id,
                        task_id="",
                        goal="",
                        source="generated",
                        template_id=None,
                        steps=[normalized],
                        autonomy_level=0,
                    )
                )["steps"][0]
                self._insert_step(conn, payload)
            conn.execute(
                """
                UPDATE plans
                   SET replan_count = replan_count + 1,
                       novel_mode = 'explore',
                       loop_events_json = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (_dump_json_any(loop_events), _now(), plan_id),
            )

    def append_loop_event(self, plan_id: str, loop_event: dict) -> None:
        with connect(self.db_path) as conn:
            loop_events = _load_plan_loop_events(conn, plan_id)
            normalized_event = _normalize_loop_event(loop_event)
            if normalized_event is None:
                return
            loop_events.append(normalized_event)
            conn.execute(
                """
                UPDATE plans
                   SET loop_events_json = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (_dump_json_any(loop_events), _now(), plan_id),
            )
            _write_audit_row(
                conn,
                kind="plan.loop_event",
                target_ref=plan_id,
                outcome="succeeded",
                detail={"type": normalized_event["type"], "reason": normalized_event["reason"]},
            )

    def recent_failed_tool_refs(self, plan_id: str, *, limit: int) -> list[str]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT tool_plugin, tool_name
                  FROM plan_steps
                 WHERE plan_id = ?
                   AND status = 'failed'
                 ORDER BY idx DESC, id DESC
                 LIMIT ?
                """,
                (plan_id, int(limit)),
            ).fetchall()
            loop_events = _load_plan_loop_events(conn, plan_id)
        refs = [f"{row['tool_plugin']}.{row['tool_name']}" for row in rows]
        for event in reversed(loop_events):
            if not isinstance(event, dict):
                continue
            if event.get("reason") != "failure":
                continue
            tool_ref = _optional_str(event.get("tool_ref"))
            if tool_ref:
                refs.append(tool_ref)
            if len(refs) >= int(limit):
                break
        return refs[: int(limit)]

    def store_plan_summary(self, plan_id: str, summary) -> str:
        summary_id = uuid.uuid4().hex
        payload = asdict(summary) if is_dataclass(summary) else dict(summary)
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO plan_summaries(id, plan_id, summary_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (summary_id, plan_id, _dump_json_any(payload), _now()),
            )
        return f"artifact:{summary_id}"

    def load_plan_summary(self, summary_ref: str) -> dict:
        summary_id = summary_ref.split(":", 1)[1] if summary_ref.startswith("artifact:") else summary_ref
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT summary_json FROM plan_summaries WHERE id = ?",
                (summary_id,),
            ).fetchone()
        if row is None:
            raise KeyError(summary_ref)
        value = json.loads(row["summary_json"])
        return value if isinstance(value, dict) else {}

    def latest_plan_summary_ref(self, plan_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id
                  FROM plan_summaries
                 WHERE plan_id = ?
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1
                """,
                (plan_id,),
            ).fetchone()
        return None if row is None else f"artifact:{row['id']}"

    def upsert_sub_agent(self, sub: SubAgent) -> None:
        with connect(self.db_path) as conn:
            _upsert_sub_agent_row(conn, sub)

    def upsert_sub_agent_with_audit(self, sub: SubAgent, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _upsert_sub_agent_row(conn, sub)
            _write_audit_row(conn, **audit)

    def set_sub_agent_status(
        self,
        sub_id: str,
        status: AgentStatus,
        *,
        result_ref: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_sub_agent_status_row(conn, sub_id, status, result_ref=result_ref)

    def set_sub_agent_status_with_audit(
        self,
        sub_id: str,
        status: AgentStatus,
        *,
        audit: dict,
        result_ref: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_sub_agent_status_row(conn, sub_id, status, result_ref=result_ref)
            _write_audit_row(conn, **audit)

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
        return _sub_agent_from_row(row)

    def list_sub_agents_for_plan(self, plan_id: str) -> list[SubAgent]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT sub_agents.id, sub_agents.parent_task_id, sub_agents.parent_step_id,
                       sub_agents.scope, sub_agents.granted_tools_json,
                       sub_agents.context_budget, sub_agents.status, sub_agents.result_ref
                  FROM sub_agents
                  JOIN plan_steps ON plan_steps.id = sub_agents.parent_step_id
                 WHERE plan_steps.plan_id = ?
                 ORDER BY sub_agents.created_at, sub_agents.id
                """,
                (plan_id,),
            ).fetchall()
        return [_sub_agent_from_row(row) for row in rows]

    def write_audit(self, **kwargs) -> None:
        with connect(self.db_path) as conn:
            _write_audit_row(conn, **kwargs)

    def list_audit(
        self,
        *,
        kind: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict]:
        return _list_audit_rows(self.db_path, kind=kind, limit=limit, offset=offset)

    def _insert_step(self, conn: sqlite3.Connection, step: dict) -> None:
        conn.execute(
            """
            INSERT INTO plan_steps(
                id, plan_id, idx, title, tool_plugin, tool_name, tool_version,
                inputs_json, depends_on_json, post_checks_json,
                needs_confirmation, decision_point, sub_agent_scope, granted_tools_json,
                status, sub_agent_id, output_ref, review_json, error, phase
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _step_insert_values(step),
        )


class DatasetRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def transaction(self):
        return connect(self.db_path)

    def create_dataset(self, dataset: Dataset) -> None:
        with connect(self.db_path) as conn:
            _insert_dataset_row(conn, dataset)

    def create_dataset_on_connection(self, conn: sqlite3.Connection, dataset: Dataset) -> None:
        _insert_dataset_row(conn, dataset)

    def create_dataset_with_audit(self, dataset: Dataset, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _insert_dataset_row(conn, dataset)
            _write_audit_row(conn, **audit)

    def create_dataset_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        dataset: Dataset,
        *,
        audit: dict,
    ) -> None:
        _insert_dataset_row(conn, dataset)
        _write_audit_row(conn, **audit)

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, role, source_path, format, sheet, row_count,
                       columns_json, has_target, target_col, created_at
                  FROM datasets
                 WHERE id = ?
                """,
                (dataset_id,),
            ).fetchone()
        return None if row is None else _dataset_from_row(row)

    def list_datasets(self, task_id: str) -> list[Dataset]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, role, source_path, format, sheet, row_count,
                       columns_json, has_target, target_col, created_at
                  FROM datasets
                 WHERE task_id = ?
                 ORDER BY created_at, id
                """,
                (task_id,),
            ).fetchall()
        return [_dataset_from_row(row) for row in rows]

    def set_dataset_role(self, dataset_id: str, role: str) -> None:
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE datasets SET role = ? WHERE id = ?",
                (role, dataset_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(dataset_id)

    def create_join_plan(self, plan: JoinPlan) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO joins(
                    id, task_id, anchor_dataset_id, joins_json, status,
                    result_dataset_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.id,
                    plan.task_id,
                    plan.anchor_dataset_id,
                    _dump_json_any([_join_spec_to_dict(spec) for spec in plan.joins]),
                    plan.status,
                    plan.result_dataset_id,
                    _now(),
                ),
            )

    def load_join_plan(self, plan_id: str) -> JoinPlan:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, anchor_dataset_id, joins_json, status,
                       result_dataset_id
                  FROM joins
                 WHERE id = ?
                """,
                (plan_id,),
            ).fetchone()
        if row is None:
            raise KeyError(plan_id)
        return _join_plan_from_row(row)

    def update_join_spec(self, plan_id: str, spec: JoinSpec) -> None:
        with connect(self.db_path) as conn:
            _update_join_spec_row(conn, plan_id, spec)

    def update_join_spec_with_audit(
        self,
        plan_id: str,
        spec: JoinSpec,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _update_join_spec_row(conn, plan_id, spec)
            _write_audit_row(conn, **audit)

    def set_join_plan_executed(self, plan_id: str, result_dataset_id: str) -> None:
        with connect(self.db_path) as conn:
            _set_join_plan_executed_row(conn, plan_id, result_dataset_id)

    def set_join_plan_executed_with_audit(
        self,
        plan_id: str,
        result_dataset_id: str,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_join_plan_executed_row(conn, plan_id, result_dataset_id)
            _write_audit_row(conn, **audit)

    def record_join_result_with_audit(
        self,
        plan_id: str,
        dataset: Dataset,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            self.record_join_result_with_audit_on_connection(
                conn,
                plan_id,
                dataset,
                audit=audit,
            )

    def record_join_result_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        plan_id: str,
        dataset: Dataset,
        *,
        audit: dict,
    ) -> None:
        _insert_dataset_row(conn, dataset)
        _set_join_plan_executed_row(conn, plan_id, dataset.id)
        _write_audit_row(conn, **audit)

    def write_audit(self, **kwargs) -> None:
        with connect(self.db_path) as conn:
            _write_audit_row(conn, **kwargs)


class ModelingRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_experiment(self, experiment: Experiment) -> None:
        with connect(self.db_path) as conn:
            _insert_experiment_row(conn, experiment)

    def create_experiment_with_audit(self, experiment: Experiment, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _insert_experiment_row(conn, experiment)
            _write_audit_row(conn, **audit)

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, recipe_id, config_json, metrics_json,
                       artifact_id, status, created_at
                  FROM experiments
                 WHERE id = ?
                """,
                (experiment_id,),
            ).fetchone()
        return None if row is None else _experiment_from_row(row)

    def list_experiments(self, task_id: str) -> list[Experiment]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, recipe_id, config_json, metrics_json,
                       artifact_id, status, created_at
                  FROM experiments
                 WHERE task_id = ?
                 ORDER BY created_at, id
                """,
                (task_id,),
            ).fetchall()
        return [_experiment_from_row(row) for row in rows]

    def attach_experiment_result(
        self,
        experiment_id: str,
        *,
        metrics: ModelMetrics,
        artifact_id: str,
        status: str = "trained",
    ) -> None:
        with connect(self.db_path) as conn:
            _attach_experiment_result_row(
                conn,
                experiment_id,
                metrics=metrics,
                artifact_id=artifact_id,
                status=status,
            )

    def attach_experiment_result_with_artifact_and_audit(
        self,
        experiment_id: str,
        *,
        artifact: ModelArtifact,
        metrics: ModelMetrics,
        status: str = "trained",
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _insert_model_artifact_row(conn, artifact)
            _attach_experiment_result_row(
                conn,
                experiment_id,
                metrics=metrics,
                artifact_id=artifact.id,
                status=status,
            )
            _write_audit_row(conn, **audit)

    def set_experiment_status(self, experiment_id: str, status: str) -> None:
        with connect(self.db_path) as conn:
            _set_experiment_status_row(conn, experiment_id, status)

    def set_experiment_status_with_audit(
        self,
        experiment_id: str,
        status: str,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_experiment_status_row(conn, experiment_id, status)
            _write_audit_row(conn, **audit)

    def create_model_artifact(self, artifact: ModelArtifact) -> None:
        with connect(self.db_path) as conn:
            _insert_model_artifact_row(conn, artifact)

    def set_model_artifact_pmml_path(self, artifact_id: str, pmml_path: str) -> None:
        with connect(self.db_path) as conn:
            _set_model_artifact_pmml_path_row(conn, artifact_id, pmml_path)

    def set_model_artifact_pmml_path_with_audit(
        self,
        artifact_id: str,
        pmml_path: str,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_model_artifact_pmml_path_row(conn, artifact_id, pmml_path)
            _write_audit_row(conn, **audit)

    def set_model_artifact_params(self, artifact_id: str, params: dict) -> None:
        with connect(self.db_path) as conn:
            _set_model_artifact_params_row(conn, artifact_id, params)

    def set_model_artifact_params_with_audit(
        self,
        artifact_id: str,
        params: dict,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_model_artifact_params_row(conn, artifact_id, params)
            _write_audit_row(conn, **audit)

    def get_model_artifact(self, artifact_id: str) -> ModelArtifact | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, experiment_id, algorithm, model_path, pmml_path,
                       feature_list_json, feature_importance_json, params_json, woe_maps_json,
                       scorecard_table_json, created_at
                  FROM model_artifacts
                 WHERE id = ?
                """,
                (artifact_id,),
            ).fetchone()
        return None if row is None else _model_artifact_from_row(row)

    def list_model_artifacts(
        self,
        *,
        experiment_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ModelArtifact]:
        bounded_limit = None if limit is None else max(1, int(limit))
        bounded_offset = max(0, int(offset))
        params: list[object] = []
        query = """
            SELECT id, experiment_id, algorithm, model_path, pmml_path,
                   feature_list_json, feature_importance_json, params_json, woe_maps_json,
                   scorecard_table_json, created_at
              FROM model_artifacts
        """
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            params.append(experiment_id)
        query += " ORDER BY created_at, id"
        if bounded_limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([bounded_limit, bounded_offset])
        with connect(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [_model_artifact_from_row(row) for row in rows]


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
            time_col, feature_columns_json, target_type, recipes_json, sample_weight_col, metrics_json, capability_tier, notebook_path, sample_path,
            pmml_path, dictionary_path, report_values_json,
            report_values_revision, status, status_message,
            status_reason_code, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def _upsert_sub_agent_row(conn: sqlite3.Connection, sub: SubAgent) -> None:
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


def _set_sub_agent_status_row(
    conn: sqlite3.Connection,
    sub_id: str,
    status: AgentStatus,
    *,
    result_ref: str | None = None,
) -> None:
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
        "novel_mode": plan_row["novel_mode"],
        "tier": plan_row["tier"],
        "replan_count": int(plan_row["replan_count"]),
        "loop_events": _load_json_array(plan_row["loop_events_json"]),
        "success_criteria": _load_json_array(plan_row["success_criteria_json"]),
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
        "decision_point": bool(row["decision_point"]),
        "sub_agent_scope": row["sub_agent_scope"],
        "granted_tools": _load_json_array(row["granted_tools_json"]),
        "status": row["status"],
        "sub_agent_id": row["sub_agent_id"],
        "output_ref": row["output_ref"],
        "review_verdicts": _load_json_array(row["review_json"]),
        "error": row["error"],
        "phase": row["phase"],
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
        int(bool(step.get("decision_point"))),
        step.get("sub_agent_scope"),
        _dump_json_any(step.get("granted_tools") or []),
        step["status"],
        step.get("sub_agent_id"),
        step.get("output_ref"),
        _dump_json_any(step.get("review_verdicts") or []),
        step.get("error"),
        step.get("phase"),
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


def _sub_agent_from_row(row: sqlite3.Row) -> SubAgent:
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


def _dataset_insert_values(dataset: Dataset) -> tuple:
    return (
        dataset.id,
        dataset.task_id,
        dataset.role,
        dataset.source_path,
        dataset.format,
        dataset.sheet,
        dataset.row_count,
        _dump_json_any([_column_profile_to_dict(column) for column in dataset.columns]),
        1 if dataset.has_target else 0,
        dataset.target_col,
        dataset.created_at,
    )


def _insert_dataset_row(conn: sqlite3.Connection, dataset: Dataset) -> None:
    conn.execute(
        """
        INSERT INTO datasets(
            id, task_id, role, source_path, format, sheet, row_count,
            columns_json, has_target, target_col, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _dataset_insert_values(dataset),
    )


def _dataset_from_row(row: sqlite3.Row) -> Dataset:
    return Dataset(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        role=str(row["role"]),
        source_path=str(row["source_path"]),
        format=str(row["format"]),
        sheet=row["sheet"],
        row_count=int(row["row_count"]),
        columns=tuple(
            _column_profile_from_dict(item)
            for item in _load_json_array(row["columns_json"])
        ),
        has_target=bool(row["has_target"]),
        target_col=row["target_col"],
        created_at=str(row["created_at"]),
    )


def _update_join_spec_row(conn: sqlite3.Connection, plan_id: str, spec: JoinSpec) -> None:
    row = conn.execute(
        "SELECT joins_json FROM joins WHERE id = ?",
        (plan_id,),
    ).fetchone()
    if row is None:
        raise KeyError(plan_id)
    original_json = str(row["joins_json"])
    replaced = False
    joins = []
    for item in (
        _join_spec_from_dict(payload) for payload in _load_json_array(original_json)
    ):
        if item.feature_dataset_id == spec.feature_dataset_id:
            joins.append(spec)
            replaced = True
        else:
            joins.append(item)
    if not replaced:
        raise KeyError(spec.feature_dataset_id)
    cursor = conn.execute(
        """
        UPDATE joins
           SET joins_json = ?
         WHERE id = ?
           AND joins_json = ?
        """,
        (
            _dump_json_any([_join_spec_to_dict(item) for item in joins]),
            plan_id,
            original_json,
        ),
    )
    if cursor.rowcount == 0:
        raise ConflictError(f"join plan {plan_id} changed while updating spec")


def _set_join_plan_executed_row(
    conn: sqlite3.Connection,
    plan_id: str,
    result_dataset_id: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE joins
           SET status = 'executed',
               result_dataset_id = ?
         WHERE id = ?
           AND status = 'draft'
        """,
        (result_dataset_id, plan_id),
    )
    if cursor.rowcount != 0:
        return
    row = conn.execute("SELECT status FROM joins WHERE id = ?", (plan_id,)).fetchone()
    if row is None:
        raise KeyError(plan_id)
    raise ConflictError(f"join plan {plan_id} is already {row['status']}; cannot execute again")


def _column_profile_to_dict(profile: ColumnProfile) -> dict:
    return asdict(profile)


def _column_profile_from_dict(payload: dict) -> ColumnProfile:
    fingerprint_payload = dict(payload["fingerprint"])
    return ColumnProfile(
        name=str(payload["name"]),
        dtype=str(payload["dtype"]),
        semantic_role=str(payload["semantic_role"]),
        fingerprint=ColumnFingerprint(
            value_kind=str(fingerprint_payload["value_kind"]),
            length_mode=_optional_int(fingerprint_payload.get("length_mode")),
            regex_pattern=_optional_str(fingerprint_payload.get("regex_pattern")),
            is_hashed=bool(fingerprint_payload["is_hashed"]),
            hash_type=_optional_str(fingerprint_payload.get("hash_type")),
            hex_case=_optional_str(fingerprint_payload.get("hex_case")),
            date_format=_optional_str(fingerprint_payload.get("date_format")),
        ),
        null_rate=float(payload["null_rate"]),
        cardinality=int(payload["cardinality"]),
        sample_values=tuple(payload.get("sample_values") or ()),
    )


def _join_spec_to_dict(spec: JoinSpec) -> dict:
    return {
        "feature_dataset_id": spec.feature_dataset_id,
        "key_pairs": [asdict(pair) for pair in spec.key_pairs],
        "diagnostics": asdict(spec.diagnostics),
        "dedup_strategy": spec.dedup_strategy,
        "confirmed": spec.confirmed,
    }


def _join_plan_from_row(row: sqlite3.Row) -> JoinPlan:
    return JoinPlan(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        anchor_dataset_id=str(row["anchor_dataset_id"]),
        joins=[
            _join_spec_from_dict(item)
            for item in _load_json_array(row["joins_json"])
        ],
        status=str(row["status"]),
        result_dataset_id=_optional_str(row["result_dataset_id"]),
    )


def _join_spec_from_dict(payload: dict) -> JoinSpec:
    return JoinSpec(
        feature_dataset_id=str(payload["feature_dataset_id"]),
        key_pairs=[
            KeyPair(
                anchor_col=str(item["anchor_col"]),
                feature_col=str(item["feature_col"]),
                match_method=str(item["match_method"]),
                transform_side=str(item["transform_side"]),
                match_rate=float(item["match_rate"]),
                resolved_by=str(item["resolved_by"]),
            )
            for item in payload.get("key_pairs") or []
        ],
        diagnostics=JoinDiagnostics(**dict(payload["diagnostics"])),
        dedup_strategy=_optional_str(payload.get("dedup_strategy")),
        confirmed=bool(payload.get("confirmed", False)),
    )


def _experiment_insert_values(experiment: Experiment) -> tuple:
    return (
        experiment.id,
        experiment.task_id,
        experiment.recipe_id,
        _dump_json_any(_train_config_to_dict(experiment.config)),
        None if experiment.metrics is None else _dump_json_any(_model_metrics_to_dict(experiment.metrics)),
        experiment.artifact_id,
        experiment.status,
        experiment.created_at,
    )


def _insert_experiment_row(conn: sqlite3.Connection, experiment: Experiment) -> None:
    conn.execute(
        """
        INSERT INTO experiments(
            id, task_id, recipe_id, config_json, metrics_json,
            artifact_id, status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _experiment_insert_values(experiment),
    )


def _attach_experiment_result_row(
    conn: sqlite3.Connection,
    experiment_id: str,
    *,
    metrics: ModelMetrics,
    artifact_id: str,
    status: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE experiments
           SET metrics_json = ?,
               artifact_id = ?,
               status = ?
         WHERE id = ?
        """,
        (
            _dump_json_any(_model_metrics_to_dict(metrics)),
            artifact_id,
            status,
            experiment_id,
        ),
    )
    if cursor.rowcount == 0:
        raise KeyError(experiment_id)


def _set_experiment_status_row(
    conn: sqlite3.Connection,
    experiment_id: str,
    status: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE experiments
           SET status = ?
         WHERE id = ?
        """,
        (status, experiment_id),
    )
    if cursor.rowcount == 0:
        raise KeyError(experiment_id)


def _experiment_from_row(row: sqlite3.Row) -> Experiment:
    metrics_json = row["metrics_json"]
    return Experiment(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        recipe_id=str(row["recipe_id"]),
        config=_train_config_from_dict(_load_json_object(row["config_json"])),
        metrics=None if metrics_json is None else _model_metrics_from_dict(_load_json_object(metrics_json)),
        artifact_id=_optional_str(row["artifact_id"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
    )


def _model_artifact_insert_values(artifact: ModelArtifact) -> tuple:
    return (
        artifact.id,
        artifact.experiment_id,
        artifact.algorithm,
        artifact.model_path,
        artifact.pmml_path,
        _dump_json_any(list(artifact.feature_list)),
        _dump_json_any([list(item) for item in artifact.feature_importance]),
        _dump_json_any(artifact.params),
        None if artifact.woe_maps is None else _dump_json_any(artifact.woe_maps),
        _dump_json_any(list(artifact.scorecard_table)),
        artifact.created_at,
    )


def _insert_model_artifact_row(conn: sqlite3.Connection, artifact: ModelArtifact) -> None:
    conn.execute(
        """
        INSERT INTO model_artifacts(
            id, experiment_id, algorithm, model_path, pmml_path,
            feature_list_json, feature_importance_json, params_json, woe_maps_json,
            scorecard_table_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _model_artifact_insert_values(artifact),
    )


def _set_model_artifact_pmml_path_row(
    conn: sqlite3.Connection,
    artifact_id: str,
    pmml_path: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE model_artifacts
           SET pmml_path = ?
         WHERE id = ?
        """,
        (pmml_path, artifact_id),
    )
    if cursor.rowcount == 0:
        raise KeyError(artifact_id)


def _set_model_artifact_params_row(
    conn: sqlite3.Connection,
    artifact_id: str,
    params: dict,
) -> None:
    cursor = conn.execute(
        """
        UPDATE model_artifacts
           SET params_json = ?
         WHERE id = ?
        """,
        (_dump_json_any(params), artifact_id),
    )
    if cursor.rowcount == 0:
        raise KeyError(artifact_id)


def _model_artifact_from_row(row: sqlite3.Row) -> ModelArtifact:
    woe_maps_json = row["woe_maps_json"]
    return ModelArtifact(
        id=str(row["id"]),
        experiment_id=str(row["experiment_id"]),
        algorithm=str(row["algorithm"]),
        model_path=str(row["model_path"]),
        pmml_path=_optional_str(row["pmml_path"]),
        feature_list=tuple(str(item) for item in _load_json_array(row["feature_list_json"])),
        feature_importance=_feature_importance_from_json(row["feature_importance_json"]),
        params=_load_json_object(row["params_json"]),
        woe_maps=None if woe_maps_json is None else _load_json_object(woe_maps_json),
        created_at=str(row["created_at"]),
        scorecard_table=tuple(
            dict(item)
            for item in _load_json_array(row["scorecard_table_json"])
            if isinstance(item, dict)
        ),
    )


def _feature_importance_from_json(raw: str | None) -> tuple[tuple[str, float], ...]:
    pairs = []
    for item in _load_json_array(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        feature, importance = item
        try:
            pairs.append((str(feature), float(importance)))
        except (TypeError, ValueError):
            continue
    return tuple(pairs)


def _train_config_to_dict(config: TrainConfig) -> dict:
    payload = asdict(config)
    payload["features"] = list(config.features)
    return payload


def _train_config_from_dict(payload: dict) -> TrainConfig:
    return TrainConfig(
        dataset_id=str(payload["dataset_id"]),
        features=tuple(str(item) for item in payload.get("features") or ()),
        target_col=str(payload["target_col"]),
        split_col=str(payload["split_col"]),
        split_values=dict(payload.get("split_values") or {}),
        params=dict(payload.get("params") or {}),
        seed=int(payload["seed"]),
        early_stopping_rounds=_optional_int(payload.get("early_stopping_rounds")),
        recipe_id=_optional_str(payload.get("recipe_id")),
        scenario_id=_optional_str(payload.get("scenario_id")),
        target_type=str(payload.get("target_type") or "binary"),
        eval_metric=str(payload.get("eval_metric") or "ks_auc"),
    )


def _model_metrics_to_dict(metrics: ModelMetrics) -> dict:
    return asdict(metrics)


def _model_metrics_from_dict(payload: dict) -> ModelMetrics:
    return ModelMetrics(
        train_ks=_optional_float(payload.get("train_ks")),
        test_ks=_optional_float(payload.get("test_ks")),
        oot_ks=_optional_float(payload.get("oot_ks")),
        train_auc=_optional_float(payload.get("train_auc")),
        test_auc=_optional_float(payload.get("test_auc")),
        oot_auc=_optional_float(payload.get("oot_auc")),
        psi_test_vs_train=_optional_float(payload.get("psi_test_vs_train")),
        psi_oot_vs_train=_optional_float(payload.get("psi_oot_vs_train")),
        overfit_train_test_gap=float(payload.get("overfit_train_test_gap") or 0.0),
        overfit_train_oot_gap=_optional_float(payload.get("overfit_train_oot_gap")),
        overfit_flag=bool(payload.get("overfit_flag")),
        weighted_train_ks=_optional_float(payload.get("weighted_train_ks")),
        weighted_test_ks=_optional_float(payload.get("weighted_test_ks")),
        weighted_oot_ks=_optional_float(payload.get("weighted_oot_ks")),
        weighted_train_auc=_optional_float(payload.get("weighted_train_auc")),
        weighted_test_auc=_optional_float(payload.get("weighted_test_auc")),
        weighted_oot_auc=_optional_float(payload.get("weighted_oot_auc")),
        weighted_psi_test_vs_train=_optional_float(payload.get("weighted_psi_test_vs_train")),
        weighted_psi_oot_vs_train=_optional_float(payload.get("weighted_psi_oot_vs_train")),
        train_rmse=_optional_float(payload.get("train_rmse")),
        test_rmse=_optional_float(payload.get("test_rmse")),
        oot_rmse=_optional_float(payload.get("oot_rmse")),
        train_mae=_optional_float(payload.get("train_mae")),
        test_mae=_optional_float(payload.get("test_mae")),
        oot_mae=_optional_float(payload.get("oot_mae")),
        train_r2=_optional_float(payload.get("train_r2")),
        test_r2=_optional_float(payload.get("test_r2")),
        oot_r2=_optional_float(payload.get("oot_r2")),
        train_macro_auc=_optional_float(payload.get("train_macro_auc")),
        test_macro_auc=_optional_float(payload.get("test_macro_auc")),
        oot_macro_auc=_optional_float(payload.get("oot_macro_auc")),
        train_logloss=_optional_float(payload.get("train_logloss")),
        test_logloss=_optional_float(payload.get("test_logloss")),
        oot_logloss=_optional_float(payload.get("oot_logloss")),
        train_accuracy=_optional_float(payload.get("train_accuracy")),
        test_accuracy=_optional_float(payload.get("test_accuracy")),
        oot_accuracy=_optional_float(payload.get("oot_accuracy")),
    )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _dump_json_any(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _step_evidence_payload(output_ref: str, evidence: dict | EvidenceEnvelope | None) -> dict:
    if isinstance(evidence, EvidenceEnvelope):
        payload = evidence.to_dict()
    elif isinstance(evidence, dict):
        payload = dict(evidence)
    else:
        payload = EvidenceEnvelope(output_ref=output_ref).to_dict()
    payload.setdefault("schema_version", "evidence.v1")
    payload["output_ref"] = output_ref
    return payload


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


def _normalize_loop_event(event: dict | None) -> dict | None:
    if event is None:
        return None
    payload = asdict(event) if is_dataclass(event) else dict(event)
    normalized = {
        "type": str(payload.get("type") or "unknown"),
        "reason": str(payload.get("reason") or ""),
        "at": str(payload.get("at") or _now()),
    }
    trigger_step_id = _optional_str(payload.get("trigger_step_id"))
    if trigger_step_id is not None:
        normalized["trigger_step_id"] = trigger_step_id
    instruction = _optional_str(payload.get("instruction"))
    if instruction is not None:
        normalized["instruction"] = instruction[:500]  # keep the replan rationale, bounded
    tool_ref = _optional_str(payload.get("tool_ref"))
    if tool_ref is not None:
        normalized["tool_ref"] = tool_ref[:200]
    return normalized


def _load_plan_loop_events(conn: sqlite3.Connection, plan_id: str) -> list:
    plan_row = conn.execute(
        "SELECT loop_events_json FROM plans WHERE id = ?",
        (plan_id,),
    ).fetchone()
    if plan_row is None:
        raise PlanNotFoundError(plan_id)
    return _load_json_array(plan_row["loop_events_json"])


def _append_normalized_loop_event(loop_events: list, event: dict | None) -> None:
    normalized_event = _normalize_loop_event(event)
    if normalized_event is not None:
        loop_events.append(normalized_event)


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


def _list_audit_rows(
    db_path: Path,
    *,
    kind: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    bounded_limit = None if limit is None else max(1, int(limit))
    bounded_offset = max(0, int(offset))
    query = (
        "SELECT id, kind, actor, target_ref, inputs_hash, outcome, detail_json, at "
        "FROM audit"
    )
    params: list[object] = []
    if kind is not None:
        query += " WHERE kind = ?"
        params.append(kind)
    query += " ORDER BY at, id"
    if bounded_limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([bounded_limit, bounded_offset])
    with connect(db_path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
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
        target_type=(row["target_type"] if "target_type" in row.keys() else "") or "",
        recipes=_load_json_list(row["recipes_json"]),
        sample_weight_col=(row["sample_weight_col"] if "sample_weight_col" in row.keys() else "") or "",
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
