from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marvis.orchestrator.contracts import (
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from marvis.orchestrator.reviewer import FinalReview
from marvis.plugins.runner import ToolResult


MAX_STEP_RETRIES = 1


@dataclass
class ExecutionResult:
    plan_id: str
    status: PlanStatus
    summary_ref: str | None
    final_review: FinalReview | None


class PlanExecutor:
    def __init__(
        self,
        plan_repo,
        tool_runner,
        reviewer,
        subagent_dispatcher,
        hook_dispatcher,
        harness_state,
    ):
        self._repo = plan_repo
        self._runner = tool_runner
        self._reviewer = reviewer
        self._subagents = subagent_dispatcher
        self._hooks = hook_dispatcher
        self._state = harness_state

    def run(self, plan_id: str) -> ExecutionResult:
        plan = self._repo.load_plan(plan_id)
        if plan.status in {PlanStatus.DONE, PlanStatus.FAILED, PlanStatus.CANCELLED}:
            return ExecutionResult(plan.id, plan.status, None, None)
        if plan.status not in {
            PlanStatus.CONFIRMED,
            PlanStatus.AWAITING_CONFIRM,
            PlanStatus.RUNNING,
        }:
            return ExecutionResult(plan.id, plan.status, None, None)
        if plan.status in {PlanStatus.CONFIRMED, PlanStatus.AWAITING_CONFIRM}:
            self._set_plan_status(plan, PlanStatus.RUNNING)
        self._recover_inflight_steps(plan)

        while True:
            plan = self._repo.load_plan(plan_id)
            failed = [step for step in plan.steps if step.status == StepStatus.FAILED]
            if failed:
                self._set_plan_status(plan, PlanStatus.FAILED)
                return ExecutionResult(plan.id, PlanStatus.FAILED, None, None)

            step = self._next_ready_step(plan)
            if step is None:
                break
            if step.needs_confirmation and not self._repo.is_step_confirmed(step.id):
                self._set_step_status(step, StepStatus.AWAITING_CONFIRM)
                self._set_plan_status(plan, PlanStatus.AWAITING_CONFIRM)
                return ExecutionResult(plan.id, PlanStatus.AWAITING_CONFIRM, None, None)
            self._execute_step(plan, step)

        plan = self._repo.load_plan(plan_id)
        return self._finalize(plan)

    def _next_ready_step(self, plan: Plan) -> PlanStep | None:
        complete = {
            step.id
            for step in plan.steps
            if step.status in {StepStatus.DONE, StepStatus.SKIPPED}
        }
        for step in sorted(plan.steps, key=lambda item: (item.index, item.id)):
            if step.status not in {
                StepStatus.PENDING,
                StepStatus.BLOCKED,
                StepStatus.AWAITING_CONFIRM,
            }:
                continue
            if all(dependency in complete for dependency in step.depends_on):
                return step
        return None

    def _execute_step(self, plan: Plan, step: PlanStep) -> None:
        try:
            self._set_step_status(step, StepStatus.RUNNING)
            resolved_inputs = self._resolve_refs(step.inputs)
            result = self._invoke_step(plan, step, resolved_inputs)
            if not result.ok:
                self._handle_step_failure(step, result)
                return

            output = result.output or {}
            self._set_step_status(step, StepStatus.CHECKING)
            deterministic = self._reviewer.deterministic_check(step, output)
            step.review_verdicts.append(deterministic)
            if not deterministic.passed:
                failed = ToolResult(
                    ok=False,
                    output=None,
                    error="; ".join(deterministic.reasons),
                    error_kind="postcheck",
                    duration_ms=result.duration_ms,
                )
                self._handle_step_failure(step, failed, apply_policy=False)
                return

            critique = self._reviewer.llm_critique(step, output, plan.goal)
            step.review_verdicts.append(critique)
            step.output_ref = self._repo.store_step_output(step.id, output)
            step.status = StepStatus.DONE
            self._repo.update_step(step)
            self._dispatch(
                "step.completed",
                {"plan_id": plan.id, "step_id": step.id},
                task_id=plan.task_id,
            )
        except Exception as exc:
            self._handle_step_exception(step, exc)

    def _invoke_step(self, plan: Plan, step: PlanStep, resolved_inputs: dict) -> ToolResult:
        policy = self._failure_policy(step)
        attempts = MAX_STEP_RETRIES + 1 if policy == "retry" else 1
        last_result = None
        for _attempt in range(attempts):
            if step.sub_agent_scope:
                sub = self._subagents.spawn(step, parent_task_id=plan.task_id)
                step.sub_agent_id = sub.id
                result = self._subagents.run(sub, goal_inputs=resolved_inputs)
            else:
                result = self._runner.invoke(
                    step.tool_ref,
                    resolved_inputs,
                    task_id=plan.task_id,
                )
            if result.ok:
                return result
            last_result = result
        return last_result or ToolResult(
            ok=False,
            output=None,
            error="step execution failed",
            error_kind="execution",
            duration_ms=0,
        )

    def _handle_step_failure(
        self,
        step: PlanStep,
        result: ToolResult,
        *,
        apply_policy: bool = True,
    ) -> None:
        policy = self._failure_policy(step) if apply_policy else "fail"
        step.error = result.error or "step failed"
        if policy == "skip":
            self._set_step_status(step, StepStatus.SKIPPED)
        else:
            self._set_step_status(step, StepStatus.FAILED)

    def _handle_step_exception(self, step: PlanStep, exc: Exception) -> None:
        step.error = str(exc)
        if step.status in {StepStatus.PENDING, StepStatus.BLOCKED, StepStatus.AWAITING_CONFIRM}:
            self._set_step_status(step, StepStatus.RUNNING)
        if step.status == StepStatus.CHECKING:
            self._set_step_status(step, StepStatus.FAILED)
        elif step.status == StepStatus.RUNNING:
            self._set_step_status(step, StepStatus.FAILED)
        else:
            self._repo.update_step(step)

    def _resolve_refs(self, inputs: dict) -> dict:
        return {key: self._resolve_value(value) for key, value in inputs.items()}

    def _resolve_value(self, value):
        if isinstance(value, str) and value.startswith("$ref:"):
            step_id, field = _parse_ref(value)
            output = self._repo.load_step_output(step_id)
            return _dig(output, field) if field else output
        if isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve_value(item) for key, item in value.items()}
        return value

    def _finalize(self, plan: Plan) -> ExecutionResult:
        incomplete = [
            step
            for step in plan.steps
            if step.status not in {StepStatus.DONE, StepStatus.SKIPPED}
        ]
        if incomplete:
            self._set_plan_status(plan, PlanStatus.FAILED)
            return ExecutionResult(plan.id, PlanStatus.FAILED, None, None)

        self._set_plan_status(plan, PlanStatus.REVIEW)
        outputs = {
            step.id: self._repo.load_step_output(step.id)
            for step in plan.steps
            if step.output_ref
        }
        review = self._reviewer.final_review(plan, outputs, plan.goal)
        summary_ref = self._repo.store_plan_summary(plan.id, review)
        if review.goal_doubt:
            return ExecutionResult(plan.id, PlanStatus.REVIEW, summary_ref, review)
        final_status = PlanStatus.DONE if review.goal_met else PlanStatus.FAILED
        self._set_plan_status(plan, final_status)
        self._dispatch(
            "workflow.completed",
            {"plan_id": plan.id, "summary_ref": summary_ref},
            task_id=plan.task_id,
        )
        return ExecutionResult(plan.id, final_status, summary_ref, review)

    def _recover_inflight_steps(self, plan: Plan) -> None:
        for step in plan.steps:
            if step.status in {StepStatus.RUNNING, StepStatus.CHECKING}:
                step.status = StepStatus.PENDING
                step.error = None
                self._repo.update_step(step)

    def _failure_policy(self, step: PlanStep) -> str:
        tools = getattr(self._runner, "_tools", None)
        if tools is None:
            return "fail"
        try:
            return str(tools.resolve(step.tool_ref).failure_policy)
        except Exception:
            return "fail"

    def _set_plan_status(self, plan: Plan, status: PlanStatus) -> None:
        if plan.status == status:
            return
        self._repo.set_plan_status(plan.id, status)
        plan.status = status

    def _set_step_status(self, step: PlanStep, status: StepStatus) -> None:
        if step.status != status:
            self._state.assert_step_transition(step.status, status)
            step.status = status
        self._repo.update_step(step)

    def _dispatch(self, event: str, payload: dict, *, task_id: str) -> None:
        if self._hooks is None:
            return
        try:
            self._hooks.dispatch(event, payload, task_id=task_id)
        except Exception:
            return


def _parse_ref(value: str) -> tuple[str, str]:
    raw = value[len("$ref:"):]
    marker = ".output"
    if marker not in raw:
        raise ValueError(f"invalid ref {value}")
    step_id, tail = raw.split(marker, 1)
    if not step_id:
        raise ValueError(f"invalid ref {value}")
    if not tail:
        return step_id, ""
    if not tail.startswith(".") or tail == ".":
        raise ValueError(f"invalid ref {value}")
    return step_id, tail[1:]


def _dig(value: dict, path: str):
    current: Any = value
    for part in path.split("."):
        if not part:
            return None
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
