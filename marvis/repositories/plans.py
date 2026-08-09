import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass, replace as dataclass_replace
from datetime import UTC, datetime
from pathlib import Path

from marvis.agent.gates.contracts import EvidenceEnvelope
from marvis.canonical_results import CANONICAL_RESULT_TOOLS
from marvis.db_schema import connect
from marvis.orchestrator.contracts import (
    AgentStatus,
    Plan,
    PlanStatus,
    StepStatus,
    SubAgent,
    plan_from_dict,
    plan_payload_fingerprint,
    plan_step_payload_confirmation_fingerprint,
    plan_to_dict,
)
from marvis.orchestrator.errors import PlanNotFoundError
from marvis.orchestrator.evidence import (
    artifact_bindings,
    artifact_refs,
    dataset_refs,
    payload_hash,
    result_dataset_ids,
    step_output_references,
)
from marvis.orchestrator.harness_state import assert_plan_transition
from marvis.plugins.errors import ManifestError
from marvis.plugins.manifest import GovernancePolicy, ToolRef
from marvis.redaction import redact_value
from marvis.repositories.audit import _list_audit_rows, _write_audit_row
from marvis.repositories.datasets import DatasetRepository
from marvis.state_machine import ConflictError


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _normalized_workflow_status(
    plan_status: str,
    *,
    has_failed_step: bool,
) -> str:
    if has_failed_step:
        return PlanStatus.FAILED.value
    normalized = str(plan_status or "").strip().lower()
    if normalized == PlanStatus.CONFIRMED.value:
        return PlanStatus.RUNNING.value
    return normalized


class PlanRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_plan(
        self,
        plan: Plan,
        *,
        on_connection: Callable[[sqlite3.Connection], None] | None = None,
    ) -> None:
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
            if on_connection is not None:
                on_connection(conn)

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
                       needs_confirmation, policy_json, decision_point, sub_agent_scope,
                       granted_tools_json,
                       status, sub_agent_id, output_ref, review_json, error, phase
                  FROM plan_steps
                 WHERE plan_id = ?
                 ORDER BY idx, id
                """,
                (plan_id,),
            ).fetchall()
        return plan_from_dict(_plan_payload_from_rows(plan_row, step_rows))

    def list_plans_for_task(
        self,
        task_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Plan]:
        """Plans for a task, oldest first. Used to resume/reload a task's
        plan in the right rail (create returns the plan_id for first build).
        ``limit``/``offset`` are optional (LT-13): omitting them returns the
        full task history, matching prior behavior."""
        bounded_limit = None if limit is None else max(1, int(limit))
        bounded_offset = max(0, int(offset))
        query = "SELECT id FROM plans WHERE task_id = ? ORDER BY created_at, id"
        params: list[object] = [task_id]
        if bounded_limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([bounded_limit, bounded_offset])
        with connect(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self.load_plan(row["id"]) for row in rows]

    def count_plans_for_task(self, task_id: str) -> int:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM plans WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return int(row["total"])

    def latest_nonterminal_summary_for_task(
        self,
        task_id: str,
    ) -> dict[str, str] | None:
        """Return only the newest active plan identity needed by UI guards."""

        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, status
                  FROM plans
                 WHERE task_id = ?
                   AND status NOT IN ('done', 'failed', 'cancelled')
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return {"plan_id": str(row["id"]), "status": str(row["status"])}

    def latest_workflow_statuses_for_tasks(
        self,
        task_ids: list[str],
    ) -> dict[str, str]:
        """Return each task's normalized latest-plan execution state in one query.

        Task rows intentionally keep their own lifecycle status (driver tasks often
        remain ``created``), so list/detail API callers need a separate workflow
        projection. A failed step wins over the plan row status because recovery
        can park a failed plan at ``awaiting_confirm`` while asking the user what to
        do next; presenting that as a healthy confirmation gate is stale state.
        """

        normalized_task_ids = list(dict.fromkeys(
            str(task_id).strip()
            for task_id in task_ids
            if str(task_id).strip()
        ))
        if not normalized_task_ids:
            return {}
        placeholders = ", ".join("?" for _ in normalized_task_ids)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                WITH ranked_plans AS (
                    SELECT id,
                           task_id,
                           status,
                           ROW_NUMBER() OVER (
                               PARTITION BY task_id
                               ORDER BY created_at DESC, id DESC
                           ) AS plan_rank
                      FROM plans
                     WHERE task_id IN ({placeholders})
                )
                SELECT ranked.task_id,
                       ranked.status,
                       EXISTS (
                           SELECT 1
                             FROM plan_steps
                            WHERE plan_steps.plan_id = ranked.id
                              AND plan_steps.status = ?
                       ) AS has_failed_step
                  FROM ranked_plans AS ranked
                 WHERE ranked.plan_rank = 1
                """,
                (*normalized_task_ids, StepStatus.FAILED.value),
            ).fetchall()
        return {
            str(row["task_id"]): _normalized_workflow_status(
                str(row["status"]),
                has_failed_step=bool(row["has_failed_step"]),
            )
            for row in rows
        }

    def list_plans_by_status(self, status: PlanStatus) -> list[Plan]:
        """Plans currently in ``status`` across every task, oldest first. Used
        by the startup reclaim pass (REL-4) to find RUNNING V2 plans left
        behind by a crash/restart — the plan layer has no per-task index, so
        this scans by status directly."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id FROM plans WHERE status = ? ORDER BY created_at, id",
                (status.value,),
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
                       policy_json = ?,
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

    def confirm_plan(
        self,
        plan_id: str,
        *,
        expected_plan_fingerprint: str | None = None,
        expected_plan_revision: int | None = None,
        expected_plan_status: PlanStatus | str | None = None,
    ) -> None:
        """Confirm the exact reviewed plan snapshot in one write transaction.

        The optional expectations are a compare-and-swap boundary for semantic
        authorization.  Historical callers can omit them and retain the original
        state-machine behavior.
        """

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            plan_row, step_rows = _load_plan_snapshot_rows(conn, plan_id)
            _assert_expected_plan_snapshot(
                plan_row,
                step_rows,
                plan_id=plan_id,
                expected_fingerprint=expected_plan_fingerprint,
                expected_revision=expected_plan_revision,
                expected_status=expected_plan_status,
            )
            current = PlanStatus(str(plan_row["status"]))
            assert_plan_transition(current, PlanStatus.CONFIRMED)
            cursor = conn.execute(
                """
                UPDATE plans
                   SET status = ?, updated_at = ?
                 WHERE id = ? AND status = ? AND replan_count = ?
                """,
                (
                    PlanStatus.CONFIRMED.value,
                    _now(),
                    plan_id,
                    current.value,
                    int(plan_row["replan_count"]),
                ),
            )
            if cursor.rowcount == 0:
                raise ConflictError(f"plan {plan_id} changed while confirming")
            _write_audit_row(
                conn,
                kind="plan.status",
                target_ref=plan_id,
                outcome="succeeded",
                detail={"from": current.value, "to": PlanStatus.CONFIRMED.value},
            )

    def confirm_step(
        self,
        step_id: str,
        *,
        expected_step_fingerprint: str | None = None,
        expected_plan_fingerprint: str | None = None,
        expected_plan_revision: int | None = None,
        expected_plan_status: PlanStatus | str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            link = conn.execute(
                "SELECT plan_id FROM plan_steps WHERE id = ?",
                (step_id,),
            ).fetchone()
            if link is None:
                raise KeyError(step_id)
            plan_id = str(link["plan_id"])
            plan_row, step_rows = _load_plan_snapshot_rows(conn, plan_id)
            row = next(
                (candidate for candidate in step_rows if candidate["id"] == step_id),
                None,
            )
            if row is None:
                raise KeyError(step_id)
            _assert_expected_plan_snapshot(
                plan_row,
                step_rows,
                plan_id=plan_id,
                expected_fingerprint=expected_plan_fingerprint,
                expected_revision=expected_plan_revision,
                expected_status=expected_plan_status,
            )
            _assert_expected_step_snapshot(
                row,
                step_id=step_id,
                expected_fingerprint=expected_step_fingerprint,
            )
            _assert_raw_confirmation_allowed(row, step_id=step_id)
            # Atomic one-shot transition: only an AWAITING_CONFIRM step that has
            # NOT already been confirmed flips confirmed 0 -> 1. confirm_step
            # never changes status (the executor advances the step later), so
            # guarding on status alone let every repeat call in the
            # AWAITING_CONFIRM window "succeed" -- a no-op double-confirm guard
            # (TST-9b). Adding ``AND confirmed = 0`` makes this a real
            # compare-and-swap: the second confirm matches zero rows and raises.
            cursor = conn.execute(
                """
                UPDATE plan_steps
                   SET confirmed = 1
                 WHERE id = ?
                   AND status = ?
                   AND confirmed = 0
                """,
                (step_id, StepStatus.AWAITING_CONFIRM.value),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT status, confirmed FROM plan_steps WHERE id = ?",
                    (step_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(step_id)
                if int(row["confirmed"] or 0):
                    raise ConflictError("step is already confirmed")
                raise ConflictError(
                    f"step is not awaiting confirmation: {row['status']}"
                )
            _write_audit_row(
                conn,
                kind="plan.step.confirm",
                target_ref=step_id,
                outcome="succeeded",
            )

    def confirm_step_with_inputs(
        self,
        step_id: str,
        *,
        input_updates: dict,
        expected_step_fingerprint: str | None = None,
        expected_plan_fingerprint: str | None = None,
        expected_plan_revision: int | None = None,
        expected_plan_status: PlanStatus | str | None = None,
    ) -> None:
        """Atomically merge reviewed gate inputs and confirm the current step.

        This is the structured-control counterpart to ``confirm_step``.  The
        input mutation and one-shot confirmation share the same transaction and
        compare-and-swap guard, so an adoption reason cannot be persisted without
        the confirmation it belongs to (or vice versa).
        """

        if not isinstance(input_updates, dict) or not input_updates:
            raise ValueError("input_updates must be a non-empty object")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            link = conn.execute(
                "SELECT plan_id FROM plan_steps WHERE id = ?",
                (step_id,),
            ).fetchone()
            if link is None:
                raise KeyError(step_id)
            plan_id = str(link["plan_id"])
            plan_row, step_rows = _load_plan_snapshot_rows(conn, plan_id)
            row = next(
                (candidate for candidate in step_rows if candidate["id"] == step_id),
                None,
            )
            if row is None:
                raise KeyError(step_id)
            _assert_expected_plan_snapshot(
                plan_row,
                step_rows,
                plan_id=plan_id,
                expected_fingerprint=expected_plan_fingerprint,
                expected_revision=expected_plan_revision,
                expected_status=expected_plan_status,
            )
            _assert_expected_step_snapshot(
                row,
                step_id=step_id,
                expected_fingerprint=expected_step_fingerprint,
            )
            _assert_raw_confirmation_allowed(row, step_id=step_id)
            current_inputs = json.loads(str(row["inputs_json"] or "{}"))
            if not isinstance(current_inputs, dict):
                raise ValueError(f"step {step_id} inputs are not an object")
            merged_inputs = {**current_inputs, **input_updates}
            cursor = conn.execute(
                """
                UPDATE plan_steps
                   SET inputs_json = ?, confirmed = 1
                 WHERE id = ?
                   AND status = ?
                   AND confirmed = 0
                """,
                (
                    json.dumps(merged_inputs, ensure_ascii=False),
                    step_id,
                    StepStatus.AWAITING_CONFIRM.value,
                ),
            )
            if cursor.rowcount == 0:
                latest = conn.execute(
                    "SELECT status, confirmed FROM plan_steps WHERE id = ?",
                    (step_id,),
                ).fetchone()
                if latest is None:
                    raise KeyError(step_id)
                if int(latest["confirmed"] or 0):
                    raise ConflictError("step is already confirmed")
                raise ConflictError(
                    f"step is not awaiting confirmation: {latest['status']}"
                )
            _write_audit_row(
                conn,
                kind="plan.step.confirm",
                target_ref=step_id,
                outcome="succeeded",
                detail={"updated_input_keys": sorted(str(key) for key in input_updates)},
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

    def apply_gate_adjustment(
        self,
        plan_id: str,
        *,
        target_step_id: str,
        reset_step_ids: list[str],
        replacement_inputs_by_step: dict[str, dict] | None,
        expected_plan_status: PlanStatus | str,
        expected_plan_revision: int,
        expected_plan_fingerprint: str,
        expected_target_step_fingerprint: str,
    ) -> None:
        """Atomically revise gate inputs and invalidate every affected step.

        A typed adjustment is authorized against one rendered plan/gate snapshot.
        Acquiring the write lock before reloading that snapshot closes the gap
        between a driver's initial read and the first reset.  Every input replacement,
        output invalidation and audit row is committed together; a stale/cancelled
        plan or any later write failure leaves the complete reviewed state intact.
        """

        normalized_reset_ids = list(
            dict.fromkeys(
                str(step_id).strip()
                for step_id in reset_step_ids
                if str(step_id).strip()
            )
        )
        normalized_target_id = str(target_step_id).strip()
        if not normalized_reset_ids:
            raise ValueError("reset_step_ids must be non-empty")
        if not normalized_target_id or normalized_target_id not in normalized_reset_ids:
            raise ValueError("target_step_id must be included in reset_step_ids")

        serialized_inputs: dict[str, str] = {}
        for raw_step_id, inputs in (replacement_inputs_by_step or {}).items():
            step_id = str(raw_step_id).strip()
            if step_id not in normalized_reset_ids:
                raise ValueError(
                    "replacement input step must be included in reset_step_ids"
                )
            if not isinstance(inputs, dict):
                raise ValueError("replacement step inputs must be objects")
            serialized_inputs[step_id] = _dump_json_any(inputs)

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            plan_row, step_rows = _load_plan_snapshot_rows(conn, plan_id)
            _assert_expected_plan_snapshot(
                plan_row,
                step_rows,
                plan_id=plan_id,
                expected_fingerprint=expected_plan_fingerprint,
                expected_revision=expected_plan_revision,
                expected_status=expected_plan_status,
            )
            rows_by_id = {str(row["id"]): row for row in step_rows}
            target = rows_by_id.get(normalized_target_id)
            if target is None:
                raise KeyError(normalized_target_id)
            _assert_expected_step_snapshot(
                target,
                step_id=normalized_target_id,
                expected_fingerprint=expected_target_step_fingerprint,
            )
            missing = [
                step_id
                for step_id in normalized_reset_ids
                if step_id not in rows_by_id
            ]
            if missing:
                raise KeyError(missing[0])

            for step_id in normalized_reset_ids:
                replacement = serialized_inputs.get(step_id)
                if replacement is None:
                    cursor = conn.execute(
                        """
                        UPDATE plan_steps
                           SET status = 'pending',
                               confirmed = 0,
                               output_ref = NULL,
                               review_json = '[]',
                               error = NULL
                         WHERE id = ? AND plan_id = ?
                        """,
                        (step_id, plan_id),
                    )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE plan_steps
                           SET inputs_json = ?,
                               status = 'pending',
                               confirmed = 0,
                               output_ref = NULL,
                               review_json = '[]',
                               error = NULL
                         WHERE id = ? AND plan_id = ?
                        """,
                        (replacement, step_id, plan_id),
                    )
                if cursor.rowcount != 1:
                    raise ConflictError(
                        f"step {step_id} changed while applying gate adjustment"
                    )
                _write_audit_row(
                    conn,
                    kind="plan.step.reset",
                    target_ref=step_id,
                    outcome="succeeded",
                    detail={
                        "plan_id": plan_id,
                        "inputs_replaced": replacement is not None,
                        "adjustment_target_step_id": normalized_target_id,
                    },
                )

    def retry_failed_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        inputs: dict | None = None,
        preserve_target_confirmation: bool = False,
    ) -> list[str]:
        """Explicitly retry an interrupted step and every downstream dependent step.

        A caller may preserve the target's prior confirmation only when the
        same step was already confirmed before it failed and the replacement
        inputs are structurally identical. This supports an explicit human
        retry of the same governed action without carrying authorization over
        to changed parameters. A user-cancelled plan is also recoverable here:
        its interrupted step is persisted as ``failed`` while prior successful
        outputs and step-run evidence remain untouched. Downstream
        confirmations are always cleared.
        """
        with connect(self.db_path) as conn:
            plan_row = conn.execute(
                "SELECT status FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if plan_row is None:
                raise PlanNotFoundError(plan_id)
            source_plan_status = str(plan_row["status"])
            if source_plan_status not in {
                PlanStatus.FAILED.value,
                PlanStatus.CANCELLED.value,
            }:
                raise ConflictError(f"plan is not failed: {plan_row['status']}")

            step_rows = conn.execute(
                """
                SELECT id, idx, depends_on_json, status, confirmed, inputs_json
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
            previous_inputs = _load_json_object_unchecked(target["inputs_json"])
            inputs_unchanged = inputs is None or inputs == previous_inputs
            confirmation_preserved = bool(
                preserve_target_confirmation
                and int(target["confirmed"] or 0)
                and inputs_unchanged
            )

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
                reset_confirmation = int(
                    reset_id == step_id and confirmation_preserved
                )
                if reset_id == step_id and inputs is not None:
                    conn.execute(
                        """
                        UPDATE plan_steps
                           SET inputs_json = ?,
                               status = 'pending',
                               confirmed = ?,
                               sub_agent_id = NULL,
                               output_ref = NULL,
                               review_json = '[]',
                               error = NULL
                         WHERE id = ?
                        """,
                        (_dump_json_any(inputs), reset_confirmation, reset_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE plan_steps
                           SET status = 'pending',
                               confirmed = ?,
                               sub_agent_id = NULL,
                               output_ref = NULL,
                               review_json = '[]',
                               error = NULL
                         WHERE id = ?
                        """,
                        (reset_confirmation, reset_id),
                    )

            cursor = conn.execute(
                "UPDATE plans SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (PlanStatus.RUNNING.value, _now(), plan_id, source_plan_status),
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
                    "inputs_unchanged": inputs_unchanged,
                    "confirmation_preserved": confirmation_preserved,
                    "from_plan_status": source_plan_status,
                },
            )
            return ordered_reset_ids

    def rollback_failed_plan_from_step(
        self,
        plan_id: str,
        root_step_id: str,
        failed_step_id: str,
        *,
        root_inputs: dict,
        excluded_features: list[str] | None = None,
        tuning_budgets: dict[str, int] | None = None,
        expected_plan_revision: int,
        expected_root_output_ref: str,
    ) -> list[str]:
        """Atomically revise a completed ancestor and invalidate its descendants.

        Unlike ``retry_failed_step``, this is an upstream revision: the root's
        inputs changed, so its old output, confirmation, reviews, sub-agent and
        every transitive downstream result/decision must be invalidated.  The
        completed prefix remains untouched and the plan revision increments so
        prior governance decisions cannot authorize the revised execution.
        """

        normalized_exclusions = [
            str(item).strip() for item in (excluded_features or []) if str(item).strip()
        ]
        normalized_budgets: dict[str, int] = {}
        for raw_recipe, raw_count in (tuning_budgets or {}).items():
            recipe = str(raw_recipe).strip()
            if (
                not recipe
                or isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 1
                or raw_count > 200
            ):
                raise ValueError("tuning_budgets must contain 1..200 integer trials")
            normalized_budgets[recipe] = raw_count
        if bool(normalized_exclusions) == bool(normalized_budgets):
            raise ValueError("exactly one upstream revision kind is required")

        if normalized_exclusions:
            revised_features = (
                root_inputs.get("features") if isinstance(root_inputs, dict) else None
            )
            if (
                not isinstance(revised_features, list)
                or not revised_features
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in revised_features
                )
            ):
                raise ValueError("revised root inputs must contain a non-empty feature list")
            expected_tool = "screen_features"
            revision_kind = "feature_exclusion"
            revision_reason = "exclude_features_and_rerun"
            revision_instruction = "exclude: " + ", ".join(normalized_exclusions)
            revision_detail = {
                "excluded_features": normalized_exclusions,
                "remaining_feature_count": len(revised_features),
            }
        else:
            revised_budgets = (
                root_inputs.get("n_trials_by_recipe")
                if isinstance(root_inputs, dict)
                else None
            )
            if revised_budgets != normalized_budgets:
                raise ValueError(
                    "revised root inputs must contain the concrete tuning budget"
                )
            expected_tool = "configure_tuning"
            revision_kind = "tuning_budget"
            revision_reason = "revise_tuning_budget"
            revision_instruction = "n_trials_by_recipe: " + ", ".join(
                f"{recipe}={count}"
                for recipe, count in sorted(normalized_budgets.items())
            )
            revision_detail = {"n_trials_by_recipe": normalized_budgets}

        with connect(self.db_path) as conn:
            # Serialize the validation and mutation. A concurrent retry/replan
            # must win or lose as a whole; it may never leave a half-reset DAG.
            conn.execute("BEGIN IMMEDIATE")
            plan_row = conn.execute(
                "SELECT status, replan_count, loop_events_json FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if plan_row is None:
                raise PlanNotFoundError(plan_id)
            if str(plan_row["status"]) != PlanStatus.FAILED.value:
                raise ConflictError(f"plan is not failed: {plan_row['status']}")
            current_revision = int(plan_row["replan_count"] or 0)
            if current_revision != int(expected_plan_revision):
                raise ConflictError(
                    f"plan revision changed: expected {expected_plan_revision}, "
                    f"found {current_revision}"
                )

            step_rows = conn.execute(
                """
                SELECT id, idx, tool_plugin, tool_name, depends_on_json, status,
                       output_ref
                  FROM plan_steps
                 WHERE plan_id = ?
                 ORDER BY idx, id
                """,
                (plan_id,),
            ).fetchall()
            rows_by_id = {str(row["id"]): row for row in step_rows}
            root = rows_by_id.get(root_step_id)
            failed = rows_by_id.get(failed_step_id)
            if root is None:
                raise KeyError(root_step_id)
            if failed is None:
                raise KeyError(failed_step_id)
            if (
                str(root["tool_plugin"]) != "modeling"
                or str(root["tool_name"]) != expected_tool
            ):
                raise ConflictError(
                    f"rollback root is not the modeling {expected_tool} step"
                )
            if str(root["status"]) != StepStatus.DONE.value:
                raise ConflictError(f"rollback root is not completed: {root['status']}")
            if str(root["output_ref"] or "") != str(expected_root_output_ref or ""):
                raise ConflictError("rollback root output changed before revision")
            if str(failed["status"]) != StepStatus.FAILED.value:
                raise ConflictError(f"target step is not failed: {failed['status']}")

            reset_ids = {root_step_id}
            changed = True
            while changed:
                changed = False
                for row in step_rows:
                    row_id = str(row["id"])
                    if row_id in reset_ids:
                        continue
                    depends_on = {
                        str(item) for item in _load_json_array(row["depends_on_json"])
                    }
                    if depends_on.intersection(reset_ids):
                        reset_ids.add(row_id)
                        changed = True
            if failed_step_id not in reset_ids or failed_step_id == root_step_id:
                raise ConflictError("rollback root is not an ancestor of the failed step")

            ordered_reset_ids = [
                str(row["id"]) for row in step_rows if str(row["id"]) in reset_ids
            ]
            for reset_id in ordered_reset_ids:
                if reset_id == root_step_id:
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
                        (_dump_json_any(root_inputs), reset_id),
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

            next_revision = current_revision + 1
            loop_events = _load_json_array(plan_row["loop_events_json"])
            _append_normalized_loop_event(
                loop_events,
                {
                    "type": "upstream_revision",
                    "reason": revision_reason,
                    "trigger_step_id": root_step_id,
                    "instruction": revision_instruction,
                },
            )
            cursor = conn.execute(
                """
                UPDATE plans
                   SET status = ?,
                       replan_count = ?,
                       loop_events_json = ?,
                       updated_at = ?
                 WHERE id = ?
                   AND status = ?
                   AND replan_count = ?
                """,
                (
                    PlanStatus.RUNNING.value,
                    next_revision,
                    _dump_json_any(loop_events),
                    _now(),
                    plan_id,
                    PlanStatus.FAILED.value,
                    current_revision,
                ),
            )
            if cursor.rowcount == 0:
                raise ConflictError(f"plan {plan_id} changed while rolling back")
            _write_audit_row(
                conn,
                kind="plan.step.rollback",
                target_ref=root_step_id,
                outcome="succeeded",
                detail={
                    "plan_id": plan_id,
                    "failed_step_id": failed_step_id,
                    "reset_step_ids": ordered_reset_ids,
                    "revision_kind": revision_kind,
                    **revision_detail,
                    "plan_revision_before": current_revision,
                    "plan_revision_after": next_revision,
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
            # Guard: a run may only be opened for a step the executor has already
            # moved into RUNNING (executor._execute_step sets RUNNING immediately
            # before calling start_step_run; the retry path resets a failed step
            # to pending and the executor loop re-runs it, passing through RUNNING
            # again). Opening a run against a DONE/CHECKING/FAILED/pending step is
            # a lifecycle violation -- without this guard a stale or concurrent
            # caller could attach a spurious "running" run row to an already
            # finished step. Read the status under BEGIN IMMEDIATE so the check
            # and the INSERT are atomic against a concurrent status change.
            conn.execute("BEGIN IMMEDIATE")
            status_row = conn.execute(
                """
                SELECT status, plan_id, tool_plugin, tool_name
                  FROM plan_steps
                 WHERE id = ?
                """,
                (step_id,),
            ).fetchone()
            if status_row is None:
                raise KeyError(step_id)
            if str(status_row["status"]) != StepStatus.RUNNING.value:
                raise ConflictError(
                    f"cannot start run for step {step_id}: status is "
                    f"{status_row['status']}, expected running"
                )
            expected_tool_ref = (
                f"{status_row['tool_plugin']}.{status_row['tool_name']}"
            )
            if (
                str(status_row["plan_id"] or "") != str(plan_id)
                or str(tool_ref) != expected_tool_ref
            ):
                raise ConflictError(
                    f"cannot start run for step {step_id}: plan/tool binding changed"
                )
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
            if status == "succeeded":
                conn.execute("BEGIN IMMEDIATE")
                binding = conn.execute(
                    """
                    SELECT tool_ref, output_ref, output_hash, invocation_id,
                           raw_output_hash, canonical_binding_verified,
                           tool_version, manifest_hash
                      FROM plan_step_runs
                     WHERE id = ?
                    """,
                    (run_id,),
                ).fetchone()
                canonical_tool = (
                    binding is not None
                    and str(binding["tool_ref"] or "").rsplit(".", 1)[-1]
                    in CANONICAL_RESULT_TOOLS
                )
                if (
                    binding is None
                    or str(binding["output_ref"] or "") != str(output_ref or "")
                    or not str(binding["output_hash"] or "").startswith("sha256:")
                    or (
                        canonical_tool
                        and (
                            str(binding["invocation_id"] or "") != run_id
                            or not str(binding["raw_output_hash"] or "").startswith(
                                "sha256:"
                            )
                            or int(binding["canonical_binding_verified"] or 0) != 1
                            or not str(binding["tool_version"] or "").strip()
                            or not _is_sha256_ref(binding["manifest_hash"])
                        )
                    )
                ):
                    raise ConflictError(
                        "succeeded step run has no matching immutable output binding"
                    )
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

    def update_step_run_progress(self, run_id: str, progress: dict) -> bool:
        """Store the latest progress snapshot while a run is still active.

        Late watcher events are expected around process completion, so a
        finished/missing run returns ``False`` instead of raising or reopening
        its lifecycle.
        """

        if not isinstance(progress, dict):
            return False
        safe_progress = redact_value(progress).value
        if not isinstance(safe_progress, dict):
            return False
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE plan_step_runs
                   SET progress_json = ?,
                       progress_updated_at = ?
                 WHERE id = ?
                   AND status = 'running'
                """,
                (_dump_json_any(safe_progress), _now(), run_id),
            )
        return cursor.rowcount == 1

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
            item["input"] = _load_json_object_unchecked(item.pop("input_json", "{}"))
            item["side_effects"] = _load_json_array(item.pop("side_effects_json", "[]"))
            item["progress"] = _load_json_object_unchecked(item.pop("progress_json", "{}"))
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
            item["input"] = _load_json_object_unchecked(item.pop("input_json", "{}"))
            item["side_effects"] = _load_json_array(item.pop("side_effects_json", "[]"))
            item["progress"] = _load_json_object_unchecked(item.pop("progress_json", "{}"))
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
            output_hash = payload_hash(safe_output.value)
            evidence_payload = _step_evidence_payload(output_ref, evidence)
            step_run_id = str(evidence_payload.get("step_run_id") or "")
            producer_invocation_id = str(
                evidence_payload.get("producer_invocation_id") or ""
            )
            raw_output_hash = str(evidence_payload.get("raw_output_hash") or "")
            canonical_binding_verified = (
                evidence_payload.get("canonical_binding_verified") is True
            )
            receipt_tool_version = str(
                evidence_payload.get("tool_version") or ""
            ).strip()
            receipt_manifest_hash = str(
                evidence_payload.get("manifest_hash") or ""
            ).strip()
            run_binding = None
            if step_run_id:
                run_binding = conn.execute(
                    """
                    SELECT r.plan_id, r.step_id, r.tool_ref, r.input_json,
                           s.tool_name, s.tool_version AS planned_tool_version,
                           p.task_id
                      FROM plan_step_runs AS r
                      JOIN plan_steps AS s ON s.id = r.step_id
                      JOIN plans AS p ON p.id = r.plan_id
                     WHERE r.id = ?
                       AND r.step_id = ?
                       AND s.plan_id = r.plan_id
                       AND r.status = 'running'
                       AND r.output_ref IS NULL
                       AND r.output_hash IS NULL
                    """,
                    (step_run_id, step_id),
                ).fetchone()
                if run_binding is None:
                    raise ConflictError(
                        "step output could not bind to its active execution run"
                    )
                run_input = _load_json_object_unchecked(run_binding["input_json"])
                canonical_tool = (
                    str(run_binding["tool_ref"] or "").rsplit(".", 1)[-1]
                    in CANONICAL_RESULT_TOOLS
                )
                receipt_supplied = bool(
                    producer_invocation_id or raw_output_hash
                )
                if receipt_supplied and (
                    producer_invocation_id != step_run_id
                    or raw_output_hash != payload_hash(output)
                ):
                    raise ConflictError(
                        "step output receipt does not match the active invocation"
                    )
                if canonical_tool and (
                    producer_invocation_id != step_run_id
                    or raw_output_hash != payload_hash(output)
                    or canonical_binding_verified is not True
                    or not receipt_tool_version
                    or not _is_sha256_ref(receipt_manifest_hash)
                    or (
                        bool(str(run_binding["planned_tool_version"] or ""))
                        and receipt_tool_version
                        != str(run_binding["planned_tool_version"])
                    )
                ):
                    raise ConflictError(
                        "canonical step output has no verified producer receipt"
                    )
                parent_output_bindings, resolved_parent_refs = (
                    _exact_parent_result_bindings(conn, step_id)
                )
                result_dataset_bindings = _registered_result_dataset_bindings(
                    self.db_path,
                    task_id=str(run_binding["task_id"]),
                    output=output,
                )
                evidence_payload.update(
                    {
                        "plan_id": str(run_binding["plan_id"]),
                        "task_id": str(run_binding["task_id"]),
                        "step_id": step_id,
                        "tool_name": str(run_binding["tool_ref"]),
                        "renderer_hint": str(run_binding["tool_name"]),
                        "input_hash": payload_hash(run_input),
                        "source_dataset_refs": dataset_refs(run_input),
                        "artifact_refs": artifact_refs(safe_output.value),
                        "artifact_bindings": artifact_bindings(safe_output.value),
                        "producer_invocation_id": (
                            producer_invocation_id or None
                        ),
                        "raw_output_hash": raw_output_hash or None,
                        "canonical_binding_verified": (
                            canonical_binding_verified
                        ),
                        "tool_version": receipt_tool_version or None,
                        "manifest_hash": receipt_manifest_hash or None,
                        "parent_output_refs": [
                            item["output_ref"]
                            for item in parent_output_bindings
                        ],
                        "parent_output_bindings": parent_output_bindings,
                        "resolved_parent_refs": resolved_parent_refs,
                        "result_dataset_bindings": result_dataset_bindings,
                    }
                )
            safe_evidence = redact_value(evidence_payload)
            total_redacted = int(safe_output.redacted_count) + int(safe_evidence.redacted_count)
            evidence_value = safe_evidence.value
            if isinstance(evidence_value, dict):
                evidence_value["output_hash"] = output_hash
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
            if step_run_id:
                cursor = conn.execute(
                    """
                    UPDATE plan_step_runs
                       SET output_ref = ?, output_hash = ?,
                           invocation_id = ?, raw_output_hash = ?,
                           canonical_binding_verified = ?,
                           tool_version = ?, manifest_hash = ?
                     WHERE id = ?
                       AND step_id = ?
                       AND status = 'running'
                       AND output_ref IS NULL
                       AND output_hash IS NULL
                    """,
                    (
                        output_ref,
                        output_hash,
                        producer_invocation_id or None,
                        raw_output_hash or None,
                        int(canonical_binding_verified),
                        receipt_tool_version or None,
                        receipt_manifest_hash or None,
                        step_run_id,
                        step_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ConflictError(
                        "step output could not bind to its active execution run"
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

    def load_step_presentation_binding(
        self,
        step_id: str,
        output_ref: str,
    ) -> dict:
        """Load one exact execution result and verify its durable run binding."""

        return self._load_step_result_binding(
            step_id,
            output_ref,
            allowed_step_statuses=frozenset({StepStatus.DONE.value}),
        )

    def load_step_recovery_binding(
        self,
        step_id: str,
        output_ref: str,
    ) -> dict:
        """Authenticate a crash-persisted result before completing recovery."""

        return self._load_step_result_binding(
            step_id,
            output_ref,
            allowed_step_statuses=frozenset(
                {StepStatus.RUNNING.value, StepStatus.CHECKING.value}
            ),
        )

    def load_bound_step_output(self, step_id: str) -> dict:
        """Load only the exact immutable output attached to a completed step."""

        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT output_ref FROM plan_steps WHERE id = ?",
                (step_id,),
            ).fetchone()
        if row is None:
            raise KeyError(step_id)
        output_ref = str(row["output_ref"] or "")
        if not output_ref:
            raise ValueError("completed step has no immutable output reference")
        return dict(
            self.load_step_presentation_binding(step_id, output_ref)["output"]
        )

    def _load_step_result_binding(
        self,
        step_id: str,
        output_ref: str,
        *,
        allowed_step_statuses: frozenset[str],
    ) -> dict:

        prefix = f"metrics:{step_id}:v"
        if not isinstance(output_ref, str) or not output_ref.startswith(prefix):
            raise ValueError("step output_ref is not versioned for this step")
        raw_version = output_ref.removeprefix(prefix)
        if not raw_version.isdigit() or int(raw_version) < 1:
            raise ValueError("step output_ref version is invalid")
        version = int(raw_version)
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT v.output_json, v.evidence_json,
                       s.plan_id, s.tool_name, s.status AS step_status,
                       s.output_ref AS current_output_ref,
                       p.task_id
                  FROM plan_step_output_versions AS v
                  JOIN plan_steps AS s ON s.id = v.step_id
                  JOIN plans AS p ON p.id = s.plan_id
                 WHERE v.step_id = ? AND v.version = ?
                """,
                (step_id, version),
            ).fetchone()
            if row is None:
                raise KeyError(step_id)
            output = _load_json_object_unchecked(row["output_json"])
            evidence = _load_json_object_unchecked(row["evidence_json"])
            step_run_id = str(evidence.get("step_run_id") or "")
            run = (
                None
                if not step_run_id
                else conn.execute(
                    """
                    SELECT id, plan_id, step_id, tool_ref, status, input_json,
                           output_ref, output_hash, invocation_id,
                           raw_output_hash, canonical_binding_verified,
                           tool_version, manifest_hash
                      FROM plan_step_runs
                     WHERE id = ?
                    """,
                    (step_run_id,),
                ).fetchone()
            )
            parent_output_bindings, resolved_parent_refs = (
                _exact_parent_result_bindings(conn, step_id)
            )
        computed_output_hash = payload_hash(output)
        run_input = (
            {}
            if run is None
            else _load_json_object_unchecked(run["input_json"])
        )
        canonical_tool = (
            run is not None
            and str(run["tool_ref"] or "").rsplit(".", 1)[-1]
            in CANONICAL_RESULT_TOOLS
        )
        try:
            result_dataset_bindings = _registered_result_dataset_bindings(
                self.db_path,
                task_id=str(row["task_id"] or ""),
                output=output,
            )
        except ConflictError as exc:
            raise ValueError("step result dataset binding is no longer valid") from exc
        if (
            str(row["step_status"] or "") not in allowed_step_statuses
            or str(row["current_output_ref"] or "") != output_ref
            or evidence.get("output_ref") != output_ref
            or evidence.get("plan_id") != str(row["plan_id"] or "")
            or evidence.get("task_id") != str(row["task_id"] or "")
            or evidence.get("step_id") != step_id
            or evidence.get("renderer_hint") != str(row["tool_name"] or "")
            or run is None
            or str(run["plan_id"] or "") != str(row["plan_id"] or "")
            or str(run["step_id"] or "") != step_id
            or str(run["tool_ref"] or "") != str(evidence.get("tool_name") or "")
            or str(run["status"] or "") != "succeeded"
            or str(run["output_ref"] or "") != output_ref
            or str(run["output_hash"] or "") != computed_output_hash
            or str(evidence.get("output_hash") or "") != computed_output_hash
            or str(evidence.get("input_hash") or "") != payload_hash(run_input)
            or list(evidence.get("source_dataset_refs") or [])
            != dataset_refs(run_input)
            or list(evidence.get("artifact_refs") or []) != artifact_refs(output)
            or list(evidence.get("artifact_bindings") or [])
            != artifact_bindings(output)
            or list(evidence.get("parent_output_refs") or [])
            != [item["output_ref"] for item in parent_output_bindings]
            or list(evidence.get("parent_output_bindings") or [])
            != parent_output_bindings
            or list(evidence.get("resolved_parent_refs") or [])
            != resolved_parent_refs
            or list(evidence.get("result_dataset_bindings") or [])
            != result_dataset_bindings
            or (
                run is not None
                and str(run["invocation_id"] or "")
                != str(evidence.get("producer_invocation_id") or "")
            )
            or (
                run is not None
                and str(run["raw_output_hash"] or "")
                != str(evidence.get("raw_output_hash") or "")
            )
            or (
                run is not None
                and bool(run["canonical_binding_verified"])
                != bool(evidence.get("canonical_binding_verified"))
            )
            or (
                run is not None
                and str(run["tool_version"] or "")
                != str(evidence.get("tool_version") or "")
            )
            or (
                run is not None
                and str(run["manifest_hash"] or "")
                != str(evidence.get("manifest_hash") or "")
            )
            or (
                canonical_tool
                and (
                    str(run["invocation_id"] or "") != step_run_id
                    or not str(run["raw_output_hash"] or "").startswith(
                        "sha256:"
                    )
                    or bool(run["canonical_binding_verified"]) is not True
                    or not str(run["tool_version"] or "").strip()
                    or not _is_sha256_ref(run["manifest_hash"])
                )
            )
        ):
            raise ValueError("step presentation binding failed integrity checks")
        return {
            "plan_id": str(row["plan_id"]),
            "task_id": str(row["task_id"]),
            "step_id": step_id,
            "output_ref": output_ref,
            "output": output,
            "evidence": evidence,
            "inputs": run_input,
        }

    def latest_step_output_ref(self, step_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(version) AS version FROM plan_step_output_versions WHERE step_id = ?",
                (step_id,),
            ).fetchone()
        if row is None or row["version"] is None:
            return None
        return f"metrics:{step_id}:v{int(row['version'])}"

    def latest_step_output_ref_for_runs(
        self,
        step_id: str,
        *,
        run_ids: list[str],
    ) -> str | None:
        """Return the newest output explicitly bound to one of ``run_ids``."""
        normalized_run_ids = {str(run_id) for run_id in run_ids if str(run_id)}
        if not normalized_run_ids:
            return None
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT version, evidence_json
                  FROM plan_step_output_versions
                 WHERE step_id = ?
                 ORDER BY version DESC
                """,
                (step_id,),
            ).fetchall()
        for row in rows:
            try:
                evidence = json.loads(row["evidence_json"] or "{}")
            except (TypeError, ValueError):
                continue
            if not isinstance(evidence, dict):
                continue
            if str(evidence.get("step_run_id") or "") in normalized_run_ids:
                return f"metrics:{step_id}:v{int(row['version'])}"
        return None

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
            normalized_loop_event = _normalize_loop_event(loop_event)
            if normalized_loop_event is not None:
                loop_events.append(normalized_loop_event)
            _assert_mandatory_policies_preserved(
                conn,
                plan_id,
                payload["steps"],
                loop_event=normalized_loop_event,
            )
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
                needs_confirmation, policy_json, decision_point, sub_agent_scope,
                granted_tools_json, status, sub_agent_id, output_ref, review_json, error, phase
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _step_insert_values(step),
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


def _load_plan_snapshot_rows(
    conn: sqlite3.Connection,
    plan_id: str,
) -> tuple[sqlite3.Row, list[sqlite3.Row]]:
    """Load one complete plan snapshot from the caller's transaction."""

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
               needs_confirmation, policy_json, decision_point, sub_agent_scope,
               granted_tools_json, status, sub_agent_id, output_ref, review_json,
               error, phase, confirmed
          FROM plan_steps
         WHERE plan_id = ?
         ORDER BY idx, id
        """,
        (plan_id,),
    ).fetchall()
    return plan_row, list(step_rows)


def _assert_expected_plan_snapshot(
    plan_row: sqlite3.Row,
    step_rows: list[sqlite3.Row],
    *,
    plan_id: str,
    expected_fingerprint: str | None,
    expected_revision: int | None,
    expected_status: PlanStatus | str | None,
) -> None:
    if (
        expected_revision is not None
        and int(plan_row["replan_count"]) != int(expected_revision)
    ):
        raise ConflictError(f"plan {plan_id} revision changed while confirming")
    if expected_status is not None:
        expected_status_value = (
            expected_status.value
            if isinstance(expected_status, PlanStatus)
            else str(expected_status)
        )
        if str(plan_row["status"]) != expected_status_value:
            raise ConflictError(f"plan {plan_id} status changed while confirming")
    if expected_fingerprint is not None:
        actual_fingerprint = plan_payload_fingerprint(
            _plan_payload_from_rows(plan_row, step_rows)
        )
        if actual_fingerprint != str(expected_fingerprint):
            raise ConflictError(f"plan {plan_id} fingerprint changed while confirming")


def _assert_expected_step_snapshot(
    row: sqlite3.Row,
    *,
    step_id: str,
    expected_fingerprint: str | None,
) -> None:
    if expected_fingerprint is None:
        return
    actual_fingerprint = plan_step_payload_confirmation_fingerprint(
        _step_payload_from_row(row),
        confirmed=bool(row["confirmed"]),
    )
    if actual_fingerprint != str(expected_fingerprint):
        raise ConflictError(f"step {step_id} fingerprint changed while confirming")


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
        "policy": _load_json_object_unchecked(row["policy_json"]),
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
        _dump_json_any(step.get("policy") or {}),
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


def _assert_raw_confirmation_allowed(
    row: sqlite3.Row | None,
    *,
    step_id: str,
) -> None:
    """Fail closed before a raw repository confirmation can bypass governance.

    This check intentionally uses the policy snapshot persisted on the step,
    not the legacy ``needs_confirmation`` bit.  A policy with no human gate
    keeps historical callers working; a required or unreadable policy must go
    through ``GovernanceRepository.authorize_step`` so the confirmation and
    immutable DecisionRecord share one transaction.
    """

    if row is None:
        raise KeyError(step_id)
    if int(row["confirmed"] or 0):
        raise ConflictError("step is already confirmed")
    if str(row["status"]) != StepStatus.AWAITING_CONFIRM.value:
        raise ConflictError(f"step is not awaiting confirmation: {row['status']}")

    raw_policy = row["policy_json"]
    try:
        policy_payload = {} if raw_policy in (None, "") else json.loads(str(raw_policy))
        if not isinstance(policy_payload, dict):
            raise ManifestError("persisted tool policy must be an object")
        policy = GovernancePolicy.from_dict(policy_payload)
    except (json.JSONDecodeError, ManifestError) as exc:
        raise ConflictError(
            f"step {step_id} has an invalid persisted governance policy"
        ) from exc
    if policy.human_decision_gate == "required":
        raise ConflictError(
            f"step {step_id} requires governed human-decision authorization"
        )


def _assert_mandatory_policies_preserved(
    conn: sqlite3.Connection,
    plan_id: str,
    replacement_steps: list[dict],
    *,
    loop_event: dict | None = None,
) -> None:
    """Prevent adaptive replanning from deleting or weakening a human gate.

    The repository is the final write boundary for every replan path.  Checking
    here protects decision-point, failure, final-review, user-instruction, and
    future callers even if a planner/validator integration is accidentally
    skipped.
    """

    rows = conn.execute(
        """
        SELECT plan_steps.id, plan_steps.tool_plugin, plan_steps.tool_name,
               plan_steps.tool_version, plan_steps.needs_confirmation,
               plan_steps.policy_json, plan_steps.status,
               EXISTS (
                   SELECT 1
                     FROM plan_step_runs
                    WHERE plan_step_runs.step_id = plan_steps.id
               ) AS has_run
          FROM plan_steps
         WHERE plan_id = ?
           AND status NOT IN ('done', 'skipped')
        """,
        (plan_id,),
    ).fetchall()
    replacements = {
        str(step["id"]): step
        for step in replacement_steps
        if step.get("status") not in {"done", "skipped"}
    }
    for row in rows:
        required = GovernancePolicy.from_dict(
            _load_json_object_unchecked(row["policy_json"])
        )
        # Phase 0B's immutable source is the policy snapshot.  Migration 5
        # backfills historical ``needs_confirmation`` rows into that snapshot;
        # do not keep treating the legacy display/execution bit as a second
        # policy source after migration.
        required_human = required.human_decision_gate == "required"
        required_effect = required.effect_authorization == "required"
        if not required_human and not required_effect:
            continue
        replacement = replacements.get(str(row["id"]))
        if replacement is None:
            # Free-form instructions and planner output are not authorization.
            # A mandatory human/effect gate must remain present until a future
            # typed, target-bound governance action explicitly supports removal.
            raise ConflictError(
                f"mandatory governance policy step {row['id']} cannot be deleted by replan"
            )
        tool_ref = replacement["tool_ref"]
        if (
            str(tool_ref.get("plugin") or "") != str(row["tool_plugin"])
            or str(tool_ref.get("tool") or "") != str(row["tool_name"])
            or str(tool_ref.get("version") or "") != str(row["tool_version"] or "")
        ):
            raise ConflictError(
                f"mandatory governance policy step {row['id']} cannot change Tool"
            )
        actual = GovernancePolicy.from_dict(replacement.get("policy"))
        if required_human and (
            actual.human_decision_gate != "required"
            or not bool(replacement.get("needs_confirmation"))
        ):
            raise ConflictError(
                f"mandatory governance policy step {row['id']} cannot lower human gate"
            )
        if required_effect and (
            actual.effect_authorization != "required"
            or actual.effect_target != required.effect_target
        ):
            raise ConflictError(
                f"mandatory governance policy step {row['id']} cannot lower effect authorization"
            )


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


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _dump_json_any(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _exact_parent_result_bindings(
    conn: sqlite3.Connection,
    step_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Resolve dependency lineage from persisted plan state, never ``latest``."""

    step = conn.execute(
        "SELECT inputs_json, depends_on_json FROM plan_steps WHERE id = ?",
        (step_id,),
    ).fetchone()
    if step is None:
        raise KeyError(step_id)
    dependency_ids = [str(value) for value in _load_json_array(step["depends_on_json"])]
    bindings: list[dict[str, str]] = []
    by_step_id: dict[str, dict[str, str]] = {}
    for dependency_id in dependency_ids:
        dependency = conn.execute(
            "SELECT status, output_ref FROM plan_steps WHERE id = ?",
            (dependency_id,),
        ).fetchone()
        if dependency is None:
            raise ConflictError(
                f"step {step_id} depends on missing step {dependency_id}"
            )
        output_ref = str(dependency["output_ref"] or "")
        if not output_ref:
            if str(dependency["status"] or "") == StepStatus.SKIPPED.value:
                continue
            raise ConflictError(
                f"step {step_id} dependency {dependency_id} has no bound output"
            )
        prefix = f"metrics:{dependency_id}:v"
        raw_version = output_ref.removeprefix(prefix)
        if not output_ref.startswith(prefix) or not raw_version.isdigit():
            raise ConflictError(
                f"step {step_id} dependency {dependency_id} has an invalid output ref"
            )
        version = conn.execute(
            """
            SELECT output_json, evidence_json
              FROM plan_step_output_versions
             WHERE step_id = ? AND version = ?
            """,
            (dependency_id, int(raw_version)),
        ).fetchone()
        if version is None:
            raise ConflictError(
                f"step {step_id} dependency {dependency_id} output is missing"
            )
        output = _load_json_object_unchecked(version["output_json"])
        evidence = _load_json_object_unchecked(version["evidence_json"])
        output_hash = payload_hash(output)
        run_id = str(evidence.get("step_run_id") or "")
        run = conn.execute(
            """
            SELECT status, output_ref, output_hash
              FROM plan_step_runs
             WHERE id = ? AND step_id = ?
            """,
            (run_id, dependency_id),
        ).fetchone()
        if (
            str(dependency["status"] or "") != StepStatus.DONE.value
            or evidence.get("output_ref") != output_ref
            or evidence.get("output_hash") != output_hash
            or run is None
            or str(run["status"] or "") != "succeeded"
            or str(run["output_ref"] or "") != output_ref
            or str(run["output_hash"] or "") != output_hash
        ):
            raise ConflictError(
                f"step {step_id} dependency {dependency_id} result binding is invalid"
            )
        binding = {
            "step_id": dependency_id,
            "output_ref": output_ref,
            "output_hash": output_hash,
        }
        bindings.append(binding)
        by_step_id[dependency_id] = binding

    original_inputs = _load_json_object_unchecked(step["inputs_json"])
    resolved_refs: list[dict[str, str]] = []
    try:
        reference_specs = step_output_references(original_inputs)
    except ValueError as exc:
        raise ConflictError(f"step {step_id} contains an invalid output reference") from exc
    for reference in reference_specs:
        binding = by_step_id.get(reference["step_id"])
        if binding is None:
            raise ConflictError(
                f"step {step_id} references an unbound non-dependency output"
            )
        resolved_refs.append(
            {
                **reference,
                "output_ref": binding["output_ref"],
                "output_hash": binding["output_hash"],
            }
        )
    return bindings, resolved_refs


def _registered_result_dataset_bindings(
    db_path: Path,
    *,
    task_id: str,
    output: dict,
) -> list[dict[str, str]]:
    """Bind declared result datasets to their live task/content identity."""

    repository = DatasetRepository(db_path)
    bindings: list[dict[str, str]] = []
    for dataset_id in result_dataset_ids(output):
        try:
            dataset = repository.get_dataset(dataset_id)
        except KeyError as exc:
            raise ConflictError(
                f"result dataset is not registered: {dataset_id}"
            ) from exc
        if dataset is None:
            raise ConflictError(
                f"result dataset is not registered: {dataset_id}"
            )
        content_hash = str(getattr(dataset, "content_hash", "") or "")
        if (
            str(getattr(dataset, "id", "") or "") != dataset_id
            or str(getattr(dataset, "task_id", "") or "") != task_id
            or len(content_hash) != 64
            or any(character not in "0123456789abcdef" for character in content_hash)
        ):
            raise ConflictError(
                f"result dataset has no exact task/content binding: {dataset_id}"
            )
        bindings.append(
            {
                "dataset_id": dataset_id,
                "content_hash": content_hash,
            }
        )
    return bindings


def _is_sha256_ref(value: object) -> bool:
    text = str(value or "")
    digest = text.removeprefix("sha256:")
    return (
        text.startswith("sha256:")
        and len(digest) == 64
        and all(character in "0123456789abcdefABCDEF" for character in digest)
    )


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
