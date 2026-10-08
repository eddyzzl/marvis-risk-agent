"""Observe confirmable native state only after execution has relinquished a task.

These intervals describe pending external confirmation, not human thinking or
identity. A script can submit the same authorized API action as an operator.
"""
from marvis.runtime_observations import (
    observing_runtime, confirmation_wait, end_confirmation_wait, invalidate_runtime_observation,
)


def refresh_confirmation_wait(repo, task_id):
    if not observing_runtime():
        return
    try:
        _refresh(repo, task_id)
    except Exception:
        # Observation must never replace an application result or turn a failed
        # state read into a fabricated zero wait. No exception text is recorded.
        invalidate_runtime_observation()


def job_finished(repo, job_id):
    if not observing_runtime():
        return
    try:
        job = repo.get_job(job_id)
        if job is None:
            invalidate_runtime_observation()
        elif job["status"] in {"cancelled", "interrupted"}:
            end_confirmation_wait(job["task_id"], "withdrawn")
        else:
            _refresh(repo, job["task_id"])
    except Exception:
        invalidate_runtime_observation()


def _refresh(repo, task_id):
    from marvis.db_schema import connect

    # Evaluation-only: keep readiness reads and the observation ordered before
    # a concurrent job/draft mutation. Otherwise a job could commit between the
    # idle check and the wait event, leaving a false wait during active work.
    with connect(repo.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _refresh_locked(repo, task_id)


def _refresh_locked(repo, task_id):
    from marvis.agent.service import latest_report_draft_context, agent_conclusions_confirmed
    from marvis.domain import TASK_TYPE_VALIDATION, TaskStatus
    from marvis.orchestrator.contracts import PlanStatus, StepStatus, plan_to_dict, plan_payload_fingerprint
    from marvis.repositories.plans import PlanRepository

    task = repo.get_task(task_id)
    if repo.task_has_active_job(task_id):
        end_confirmation_wait(task_id)
        return
    if task.status == TaskStatus.FAILED:
        end_confirmation_wait(task_id, "withdrawn")
        return
    if task.task_type == TASK_TYPE_VALIDATION:
        messages = repo.list_agent_messages(task_id)
        draft = latest_report_draft_context(messages)
        message = next((item for item in messages if draft and item["id"] == draft["message_id"]), {})
        metadata = message.get("metadata") or {}
        if (draft and draft["report_revision"] == task.report_values_revision
                and isinstance(draft["message_id"], str) and bool(draft["message_id"])
                and type(draft["draft_edit_revision"]) is int and draft["draft_edit_revision"] >= 0
                and metadata.get("confirmable") is not False and metadata.get("streaming") is not True
                and task.status in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED, TaskStatus.SUCCEEDED}
                and agent_conclusions_confirmed(draft["text_values"])):
            confirmation_wait(task_id, "report_confirmation_wait", {
                "draft_id": draft["message_id"], "edit_revision": draft["draft_edit_revision"],
                "report_revision": draft["report_revision"],
            })
            return
    plans = PlanRepository(repo.db_path).list_plans_for_task(task_id)
    if plans:
        plan = plans[-1]
        if plan.status == PlanStatus.VALIDATED:
            confirmation_wait(task_id, "plan_confirmation_wait", {
                "plan_id": plan.id, "revision": plan.replan_count,
                "snapshot": plan_payload_fingerprint(plan_to_dict(plan)),
            })
            return
        if plan.status == PlanStatus.AWAITING_CONFIRM:
            gates = [step.id for step in plan.steps if step.status == StepStatus.AWAITING_CONFIRM
                     and not PlanRepository(repo.db_path).is_step_confirmed(step.id)]
            if gates:
                confirmation_wait(task_id, "workflow_confirmation_wait", {
                    "plan_id": plan.id, "revision": plan.replan_count, "gate_ids": sorted(gates),
                    "snapshot": plan_payload_fingerprint(plan_to_dict(plan)),
                })
                return
            if any(step.status == StepStatus.AWAITING_CONFIRM for step in plan.steps):
                end_confirmation_wait(task_id)
                return
        if plan.status in {PlanStatus.CONFIRMED, PlanStatus.RUNNING}:
            end_confirmation_wait(task_id)
            return
    end_confirmation_wait(task_id, "withdrawn")
