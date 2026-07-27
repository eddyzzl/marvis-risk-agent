from __future__ import annotations

import hashlib
import inspect
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from marvis.governance.errors import AuthorizationError
from marvis.job_cancellation import JobCancelled
from marvis.orchestrator.capability import CapabilityTier, resolve_tier
from marvis.orchestrator.context.observation import summarize_failure, summarize_output
from marvis.orchestrator.contracts import (
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from marvis.orchestrator.errors import RefResolutionError
from marvis.llm_client import LLMClientError
from marvis.llm_settings import LLMSettingsError
from marvis.orchestrator.planner import PlanningError, ReplanError
from marvis.orchestrator.plan_recovery import PlanStepRecovery
from marvis.orchestrator.reviewer import FinalReview, ReviewVerdict
from marvis.orchestrator.safety import is_safety_step
from marvis.orchestrator.validator import METRIC_FIELDS
from marvis.plugins.manifest import manifest_to_dict
from marvis.plugins.runner import ToolResult
from marvis.repositories.tasks import TaskRepository


MAX_STEP_RETRIES = 1
NO_PROGRESS_WINDOW = 4
NO_PROGRESS_THRESHOLD = 2


def _accepts_progress_callback(invoke) -> bool:
    """Keep lightweight test/custom runners source-compatible."""

    try:
        parameters = inspect.signature(invoke).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "progress_callback"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _accepts_cancellation_check(invoke) -> bool:
    """Keep lightweight test/custom runners source-compatible."""

    try:
        parameters = inspect.signature(invoke).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "cancellation_check"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _tool_progress_content(progress: dict, *, status: str) -> str:
    labels = {
        "running": "模型调参正在执行",
        "succeeded": "模型调参已完成",
        "failed": "模型调参执行失败，已保留最后进度",
        "cancelled": "模型调参已取消，已保留最后进度",
        "interrupted": "模型调参已中断，已保留最后进度",
    }
    details = []
    algorithm = str(progress.get("algorithm") or "").strip()
    if algorithm:
        details.append(algorithm)
    trial = progress.get("trial")
    trial_total = progress.get("trial_total")
    if trial is not None or trial_total is not None:
        details.append(f"当前轮次 {trial or 0}/{trial_total or '?'}")
    completed = progress.get("completed_trials")
    total = progress.get("total_trials")
    if completed is not None or total is not None:
        details.append(f"总进度 {completed or 0}/{total or '?'}")
    prefix = labels.get(status, labels["running"])
    return f"{prefix}：{'，'.join(details)}。" if details else f"{prefix}。"


class _ToolProgressPublisher:
    """Mirror one tool run into the run ledger and one stable timeline message."""

    def __init__(
        self,
        *,
        plan_repo,
        task_repo,
        fallback_task_repo,
        task_id: str,
        plan_id: str,
        step_id: str,
        run_id: str,
    ):
        self._plan_repo = plan_repo
        self._task_repo = task_repo
        self._fallback_task_repo = fallback_task_repo
        self._task_id = task_id
        self._plan_id = plan_id
        self._step_id = step_id
        self._run_id = run_id
        self._message_id: str | None = None
        self._last_progress: dict | None = None
        self._closed = False
        self._lock = threading.Lock()

    def publish(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        snapshot = dict(payload)
        with self._lock:
            if self._closed:
                return
            self._last_progress = snapshot
            try:
                self._plan_repo.update_step_run_progress(self._run_id, snapshot)
            except Exception:
                # Progress is observability, never a reason to fail the tool.
                pass
            self._write_message(snapshot, status="running", streaming=True)

    def flush(self) -> None:
        """Wait for an in-flight watcher callback before the run is finalized."""

        with self._lock:
            return

    def finish(self, status: str) -> None:
        with self._lock:
            if self._closed:
                return
            if self._last_progress is None:
                self._closed = True
                return
            status = self._terminal_status(status)
            # A transient SQLite lock must not leave a completed run presented
            # as permanently streaming. Both attempts remain best-effort.
            for _attempt in range(3):
                if self._write_message(
                    self._last_progress,
                    status=status,
                    streaming=False,
                ):
                    break
            self._closed = True

    def _terminal_status(self, fallback: str) -> str:
        if fallback == "cancelled":
            return fallback
        try:
            plan = self._plan_repo.load_plan(self._plan_id)
        except Exception:
            return fallback
        return "cancelled" if plan.status == PlanStatus.CANCELLED else fallback

    def _write_message(
        self,
        progress: dict,
        *,
        status: str,
        streaming: bool,
    ) -> bool:
        repositories = [
            repository
            for repository in (self._task_repo, self._fallback_task_repo)
            if repository is not None
        ]
        if not repositories:
            return False
        metadata = {
            "kind": "tool_progress",
            "plan_id": self._plan_id,
            "step_id": self._step_id,
            "run_id": self._run_id,
            "status": status,
            "streaming": streaming,
            "progress": progress,
            "progress_updated_at": datetime.now(UTC).isoformat(),
        }
        content = _tool_progress_content(progress, status=status)
        for repository in repositories:
            try:
                if self._message_id is None:
                    message = repository.add_agent_message(
                        self._task_id,
                        role="assistant",
                        stage="chat",
                        content=content,
                        metadata=metadata,
                    )
                    message_id = (
                        message.get("id") if isinstance(message, dict) else None
                    )
                    if message_id:
                        self._message_id = str(message_id)
                        return True
                    continue
                repository.update_agent_message(
                    self._message_id,
                    content=content,
                    metadata=metadata,
                )
                return True
            except Exception:
                # Conversation persistence is best-effort and must not change
                # the deterministic result. Try the canonical repository next.
                continue
        return False


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
        planner=None,
        authorizer=None,
        task_repo=None,
    ):
        self._repo = plan_repo
        self._runner = tool_runner
        self._reviewer = reviewer
        self._subagents = subagent_dispatcher
        self._hooks = hook_dispatcher
        self._state = harness_state
        self._planner = planner
        self._authorizer = authorizer
        self._task_repo = task_repo
        self._progress_fallback_task_repo = None
        db_path = getattr(plan_repo, "db_path", None)
        if db_path is not None:
            canonical_task_repo = TaskRepository(db_path)
            if self._task_repo is None:
                self._task_repo = canonical_task_repo
            else:
                self._progress_fallback_task_repo = canonical_task_repo
        self._step_recovery = PlanStepRecovery(plan_repo, reviewer, hook_dispatcher, harness_state)

    def run(self, plan_id: str, *, cancellation_check=None) -> ExecutionResult:
        plan = self._repo.load_plan(plan_id)
        tier = resolve_tier(plan.tier)
        if plan.status in {PlanStatus.DONE, PlanStatus.FAILED, PlanStatus.CANCELLED}:
            return ExecutionResult(plan.id, plan.status, None, None)
        if plan.status == PlanStatus.REVIEW:
            return ExecutionResult(
                plan.id,
                PlanStatus.REVIEW,
                self._repo.latest_plan_summary_ref(plan.id),
                None,
            )
        if plan.status not in {
            PlanStatus.CONFIRMED,
            PlanStatus.AWAITING_CONFIRM,
            PlanStatus.RUNNING,
        }:
            return ExecutionResult(plan.id, plan.status, None, None)
        try:
            self._raise_if_cancelled(cancellation_check)
        except JobCancelled:
            self._cancel_plan(plan)
            return ExecutionResult(plan.id, PlanStatus.CANCELLED, None, None)
        if plan.status in {PlanStatus.CONFIRMED, PlanStatus.AWAITING_CONFIRM}:
            self._set_plan_status(plan, PlanStatus.RUNNING)
        self._step_recovery.recover_inflight_steps(plan)

        while True:
            plan = self._repo.load_plan(plan_id)
            if plan.status == PlanStatus.CANCELLED:
                # Cooperative cancel checkpoint (REL-5): a cancel request can
                # land between two step executions (each _execute_step call is
                # itself uninterruptible mid-tool-invocation). Recognize the
                # externally-applied CANCELLED status here instead of trying
                # another _set_plan_status transition, which would raise
                # IllegalPlanTransition since CANCELLED has no further moves.
                return ExecutionResult(plan.id, PlanStatus.CANCELLED, None, None)
            try:
                self._raise_if_cancelled(cancellation_check)
            except JobCancelled:
                self._cancel_plan(plan)
                return ExecutionResult(plan.id, PlanStatus.CANCELLED, None, None)
            failed = [step for step in plan.steps if step.status == StepStatus.FAILED]
            if failed:
                no_progress_step = None
                replanned = False
                for step in failed:
                    if not self._should_failure_replan(tier, plan, step):
                        continue
                    if self._no_progress(plan, step):
                        no_progress_step = no_progress_step or step
                        continue
                    if self._try_replan(plan, step, reason="failure", tier=tier):
                        replanned = True
                        break
                if replanned:
                    continue
                if no_progress_step is not None:
                    self._repo.append_loop_event(
                        plan.id,
                        {
                            "type": "no_progress",
                            "reason": "failure",
                            "trigger_step_id": no_progress_step.id,
                            "tool_ref": no_progress_step.tool_ref.label(),
                        },
                    )
                self._set_plan_status(plan, PlanStatus.FAILED)
                return ExecutionResult(plan.id, PlanStatus.FAILED, None, None)

            step = self._next_ready_step(plan)
            if step is None:
                if plan.novel_mode == "explore" and self._try_append_explore_segment(plan, tier):
                    continue
                result = self._finalize(plan, tier)
                if result.status == PlanStatus.RUNNING:
                    continue
                return result
            if (
                self._requires_human_decision(step)
                and self._special_value_requires_decision(plan, step)
                and not self._repo.is_step_confirmed(step.id)
            ):
                self._set_step_status(step, StepStatus.AWAITING_CONFIRM)
                self._set_plan_status(plan, PlanStatus.AWAITING_CONFIRM)
                return ExecutionResult(plan.id, PlanStatus.AWAITING_CONFIRM, None, None)
            self._execute_step(plan, step, cancellation_check=cancellation_check)
            plan = self._repo.load_plan(plan_id)
            last = _find_step(plan, step.id)
            if (
                last is not None
                and last.status == StepStatus.DONE
                and last.decision_point
                and tier.decision_point_replan
                and not is_safety_step(last)
                and not self._requires_effect_authorization(last)
            ):
                self._try_replan(plan, last, reason="decision_point", tier=tier)

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

    def _execute_step(self, plan: Plan, step: PlanStep, *, cancellation_check=None) -> None:
        run_id = None
        progress_publisher = None
        try:
            self._set_step_status(step, StepStatus.RUNNING)
            resolved_inputs = self._resolve_refs(step.inputs)
            run_id = self._repo.start_step_run(
                plan_id=plan.id,
                step_id=step.id,
                tool_ref=step.tool_ref.label(),
                inputs=resolved_inputs,
            )
            progress_publisher = self._step_progress_publisher(plan, step, run_id)
            try:
                result = self._invoke_step(
                    plan,
                    step,
                    resolved_inputs,
                    progress_callback=(
                        progress_publisher.publish if progress_publisher else None
                    ),
                    cancellation_check=cancellation_check,
                )
            except BaseException as exc:
                if progress_publisher is not None:
                    progress_publisher.finish(
                        "cancelled"
                        if isinstance(exc, JobCancelled) or not isinstance(exc, Exception)
                        else "failed"
                    )
                raise
            if progress_publisher is not None:
                progress_publisher.flush()
            if not result.ok:
                if result.error_kind == "cancelled":
                    self._finish_cancelled_step(
                        plan,
                        step,
                        run_id=run_id,
                        progress_publisher=progress_publisher,
                        duration_ms=result.duration_ms,
                        error=result.error,
                    )
                    return
                self._finish_step_run(
                    run_id,
                    status="failed",
                    error=result.error or "step failed",
                    error_kind=result.error_kind,
                    duration_ms=result.duration_ms,
                )
                if progress_publisher is not None:
                    progress_publisher.finish("failed")
                self._handle_step_failure(step, result)
                return

            output = result.output or {}
            self._set_step_status(step, StepStatus.CHECKING)
            step.output_ref = self._repo.store_step_output(
                step.id,
                output,
                evidence=self._step_evidence(
                    step,
                    resolved_inputs,
                    output,
                    run_id=run_id,
                ),
            )
            self._finish_step_run(
                run_id,
                status="succeeded",
                output_ref=step.output_ref,
                duration_ms=result.duration_ms,
            )
            self._repo.update_step(step)
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
                if progress_publisher is not None:
                    progress_publisher.finish("failed")
                self._handle_step_failure(step, failed, apply_policy=False)
                return

            critique = self._critique_step(step, output, plan.goal)
            if critique is not None:
                step.review_verdicts.append(critique)
            step.status = StepStatus.DONE
            self._repo.update_step(step)
            self._dispatch_step_completed(plan, step, output)
            if progress_publisher is not None:
                progress_publisher.finish("succeeded")
        except JobCancelled as exc:
            self._finish_cancelled_step(
                plan,
                step,
                run_id=run_id,
                progress_publisher=progress_publisher,
                error=str(exc),
            )
        except Exception as exc:
            if progress_publisher is not None:
                progress_publisher.finish("failed")
            if run_id is not None:
                try:
                    self._finish_step_run(
                        run_id,
                        status="failed",
                        error=str(exc),
                        error_kind=exc.__class__.__name__,
                    )
                except Exception as finish_exc:
                    step.error = f"{exc}; step run finalization failed: {finish_exc}"
                    self._set_step_status(step, StepStatus.FAILED)
                    return
            self._handle_step_exception(step, exc)

    def _critique_step(self, step: PlanStep, output: dict, goal: str) -> ReviewVerdict | None:
        # AGT-6: llm_critique used to run unconditionally for every step, adding a
        # synchronous 5-20s LLM round-trip per step and rendering "skipped: no LLM
        # configured" noise even where the deterministic checks are all that
        # matters. Narrow the trigger surface to steps where a soft second opinion
        # is actually useful: decision points, confirmation gates, and any step
        # whose output carries metric values worth a sanity check. Plain read/
        # profile-type steps skip outright — no call, no verdict, no noise.
        if not (step.decision_point or step.needs_confirmation or _output_has_metrics(output)):
            return None
        return self._reviewer.llm_critique(step, output, goal)

    def _finish_step_run(self, run_id: str, **kwargs) -> None:
        self._repo.finish_step_run(run_id, **kwargs)

    def _step_evidence(
        self,
        step: PlanStep,
        resolved_inputs: dict,
        output: dict,
        *,
        run_id: str,
    ) -> dict:
        seed = resolved_inputs.get("seed")
        tool_version, manifest_hash = _tool_manifest_details(getattr(self._runner, "_tools", None), step.tool_ref)
        return {
            "step_run_id": run_id,
            "tool_name": step.tool_ref.label(),
            "tool_version": tool_version,
            "manifest_hash": manifest_hash,
            "input_hash": _payload_hash(resolved_inputs),
            "input_summary": _bounded_input_summary(resolved_inputs),
            "source_dataset_refs": _dataset_refs(resolved_inputs),
            "artifact_refs": _artifact_refs(output),
            "parent_output_refs": _parent_output_refs(self._repo, step),
            "random_seed": seed if isinstance(seed, int) else None,
            "renderer_hint": step.tool_ref.tool,
        }

    def _invoke_step(
        self,
        plan: Plan,
        step: PlanStep,
        resolved_inputs: dict,
        *,
        progress_callback=None,
        cancellation_check=None,
    ) -> ToolResult:
        policy = self._failure_policy(step)
        protected_execution = (
            self._is_governed_step(step)
            and self._special_value_requires_decision(plan, step)
        )
        attempts = (
            MAX_STEP_RETRIES + 1
            if policy == "retry" and not protected_execution
            else 1
        )
        execution_context = None
        if protected_execution:
            if step.sub_agent_scope:
                return ToolResult(
                    ok=False,
                    output=None,
                    error="governed steps cannot run in a sub-agent",
                    error_kind="authorization",
                    duration_ms=0,
                )
            try:
                execution_context = self._resolve_execution_context(
                    plan,
                    step,
                    resolved_inputs,
                )
            except AuthorizationError as exc:
                return ToolResult(
                    ok=False,
                    output=None,
                    error=f"governed execution context unavailable: {exc}",
                    error_kind="authorization",
                    duration_ms=0,
                )
            if execution_context is None:
                return ToolResult(
                    ok=False,
                    output=None,
                    error="governed execution context unavailable",
                    error_kind="authorization",
                    duration_ms=0,
                )
        last_result = None
        for _attempt in range(attempts):
            self._raise_if_cancelled(cancellation_check)
            if step.sub_agent_scope:
                sub = self._subagents.spawn(step, parent_task_id=plan.task_id)
                step.sub_agent_id = sub.id
                result = self._subagents.run(sub, goal_inputs=resolved_inputs)
            else:
                invoke_kwargs = {"task_id": plan.task_id}
                if protected_execution:
                    invoke_kwargs["execution_context"] = execution_context
                if progress_callback is not None and _accepts_progress_callback(
                    self._runner.invoke
                ):
                    invoke_kwargs["progress_callback"] = progress_callback
                if cancellation_check is not None and _accepts_cancellation_check(
                    self._runner.invoke
                ):
                    invoke_kwargs["cancellation_check"] = cancellation_check
                if protected_execution:
                    result = self._runner.invoke(
                        step.tool_ref,
                        resolved_inputs,
                        **invoke_kwargs,
                    )
                else:
                    result = self._runner.invoke(
                        step.tool_ref,
                        resolved_inputs,
                        **invoke_kwargs,
                    )
            if result.ok:
                return result
            if result.error_kind == "cancelled":
                return result
            last_result = result
        return last_result or ToolResult(
            ok=False,
            output=None,
            error="step execution failed",
            error_kind="execution",
            duration_ms=0,
        )

    def _finish_cancelled_step(
        self,
        plan: Plan,
        step: PlanStep,
        *,
        run_id: str | None,
        progress_publisher,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        message = (
            "用户已停止当前动作；已完成步骤、调参检查点和最后进度均已保留，"
            "可重新执行当前步骤。"
        )
        if run_id is not None:
            try:
                self._finish_step_run(
                    run_id,
                    status="interrupted",
                    error=error or message,
                    error_kind="user_cancelled",
                    duration_ms=duration_ms,
                )
            except KeyError:
                # A concurrent recovery pass may already have closed the run.
                pass
        if progress_publisher is not None:
            progress_publisher.finish("cancelled")
        step.error = message
        if step.status in {StepStatus.RUNNING, StepStatus.CHECKING}:
            self._set_step_status(step, StepStatus.FAILED)
        else:
            self._repo.update_step(step)
        self._cancel_plan(plan, trigger_step_id=step.id)

    def _cancel_plan(self, plan: Plan, *, trigger_step_id: str | None = None) -> None:
        latest = self._repo.load_plan(plan.id)
        if latest.status != PlanStatus.CANCELLED:
            self._set_plan_status(latest, PlanStatus.CANCELLED)
        plan.status = PlanStatus.CANCELLED
        try:
            self._repo.append_loop_event(
                plan.id,
                {
                    "type": "cancelled",
                    "reason": "user_cancelled",
                    "trigger_step_id": trigger_step_id,
                },
            )
        except Exception:
            pass
        self._dispatch(
            "workflow.cancelled",
            {"plan_id": plan.id, "step_id": trigger_step_id},
            task_id=plan.task_id,
        )

    @staticmethod
    def _raise_if_cancelled(cancellation_check) -> None:
        if cancellation_check is not None:
            cancellation_check()

    def _step_progress_publisher(
        self,
        plan: Plan,
        step: PlanStep,
        run_id: str | None,
    ) -> _ToolProgressPublisher | None:
        if run_id is None or step.tool_ref.tool != "tune_hyperparameters":
            return None
        return _ToolProgressPublisher(
            plan_repo=self._repo,
            task_repo=self._task_repo,
            fallback_task_repo=self._progress_fallback_task_repo,
            task_id=plan.task_id,
            plan_id=plan.id,
            step_id=step.id,
            run_id=run_id,
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
            try:
                step_id, field = _parse_ref(value)
            except ValueError as exc:
                raise RefResolutionError(value, str(exc)) from exc
            try:
                output = self._repo.load_step_output(step_id)
            except KeyError as exc:
                raise RefResolutionError(value, f"upstream output {step_id} is missing") from exc
            return _dig(output, field, ref=value) if field else output
        if isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve_value(item) for key, item in value.items()}
        return value

    def _finalize(self, plan: Plan, tier: CapabilityTier) -> ExecutionResult:
        incomplete = [
            step
            for step in plan.steps
            if step.status not in {StepStatus.DONE, StepStatus.SKIPPED}
        ]
        if incomplete:
            self._set_plan_status(plan, PlanStatus.FAILED)
            return ExecutionResult(plan.id, PlanStatus.FAILED, None, None)

        outputs = {
            step.id: self._repo.load_step_output(step.id)
            for step in plan.steps
            if step.output_ref
        }
        review = self._reviewer.final_review(plan, outputs, plan.goal)
        summary_ref = self._repo.store_plan_summary(plan.id, review)
        if review.goal_doubt:
            self._set_plan_status(plan, PlanStatus.REVIEW)
            return ExecutionResult(plan.id, PlanStatus.REVIEW, summary_ref, review)
        if (
            not review.goal_met
            and _final_review_failure_replannable(review)
            and self._try_final_review_replan(plan, review, tier)
        ):
            return ExecutionResult(plan.id, PlanStatus.RUNNING, summary_ref, review)
        self._set_plan_status(plan, PlanStatus.REVIEW)
        final_status = PlanStatus.DONE if review.goal_met else PlanStatus.FAILED
        self._set_plan_status(plan, final_status)
        self._dispatch(
            "workflow.completed",
            {"plan_id": plan.id, "summary_ref": summary_ref},
            task_id=plan.task_id,
        )
        return ExecutionResult(plan.id, final_status, summary_ref, review)

    def _failure_policy(self, step: PlanStep) -> str:
        if self._is_governed_step(step):
            return "fail"
        tools = getattr(self._runner, "_tools", None)
        if tools is None:
            return "fail"
        try:
            return str(tools.resolve(step.tool_ref).failure_policy)
        except Exception:
            return "fail"

    def _dispatch_feature_computed(self, plan: Plan, step: PlanStep, output: dict) -> None:
        if step.tool_ref.plugin != "feature":
            return
        payload = {
            "plan_id": plan.id,
            "step_id": step.id,
            "tool": step.tool_ref.tool,
            "output_ref": step.output_ref,
        }
        for field in (
            "dataset_id",
            "derived_dataset_id",
            "features",
            "new_columns",
            "feature",
            "target_col",
        ):
            if field in output:
                payload[field] = output[field]
        self._dispatch("feature.computed", payload, task_id=plan.task_id)

    def _dispatch_step_completed(self, plan: Plan, step: PlanStep, output: dict) -> None:
        self._dispatch_feature_computed(plan, step, output)
        self._dispatch(
            "step.completed",
            {
                "plan_id": plan.id,
                "step_id": step.id,
                **_review_warning_payload(step),
            },
            task_id=plan.task_id,
        )

    def _should_failure_replan(
        self,
        tier: CapabilityTier,
        plan: Plan,
        step: PlanStep,
    ) -> bool:
        if self._planner is None:
            return False
        if self._is_governed_step(step):
            return False
        if not tier.failure_driven_replan:
            return False
        if plan.replan_count >= tier.max_replan_iterations:
            return False
        if _has_deterministic_failure(step):
            return False
        return not _is_fatal_error(step.error)

    def _try_replan(
        self,
        plan: Plan,
        trigger_step: PlanStep,
        *,
        reason: str,
        tier: CapabilityTier,
    ) -> bool:
        if self._planner is None:
            return False
        try:
            new_plan = self._planner.replan(
                plan,
                completed_summaries=self._summaries(plan),
                observation=self._observation(trigger_step, reason),
                reason=reason,
                tier=tier,
            )
            self._repo.replace_remaining_steps(
                plan.id,
                new_plan,
                loop_event={
                    "type": "replan",
                    "reason": reason,
                    "trigger_step_id": trigger_step.id,
                    "tool_ref": trigger_step.tool_ref.label(),
                },
            )
            self._dispatch(
                "plan.replanned",
                {"plan_id": plan.id, "reason": reason, "trigger_step_id": trigger_step.id},
                task_id=plan.task_id,
            )
            return True
        except (KeyError, PlanningError, LLMClientError, LLMSettingsError):
            # Replan is a best-effort enhancement. In manual mode (no LLM configured) the
            # planner cannot replan — that is NOT a flow error; swallow it and let the plan
            # continue to its confirmation gate. PlanningError covers ReplanError + invalid
            # replans; LLMClientError/LLMSettingsError cover unavailable models.
            return False

    def _try_final_review_replan(
        self,
        plan: Plan,
        review: FinalReview,
        tier: CapabilityTier,
    ) -> bool:
        if self._planner is None or not tier.decision_point_replan:
            return False
        if plan.replan_count >= tier.max_replan_iterations:
            return False
        trigger = _last_executed_step(plan)
        if trigger is None:
            return False
        if self._requires_effect_authorization(trigger):
            return False
        try:
            new_plan = self._planner.replan(
                plan,
                completed_summaries=self._summaries(plan),
                observation={
                    "reason": "final_review",
                    "summary": review.summary,
                    "open_items": review.open_items,
                    "goal_met": review.goal_met,
                },
                reason="final_review",
                tier=tier,
            )
            self._repo.replace_remaining_steps(
                plan.id,
                new_plan,
                loop_event={
                    "type": "replan",
                    "reason": "final_review",
                    "trigger_step_id": trigger.id,
                },
            )
            self._dispatch(
                "plan.replanned",
                {"plan_id": plan.id, "reason": "final_review", "trigger_step_id": trigger.id},
                task_id=plan.task_id,
            )
            return True
        except (KeyError, PlanningError, LLMClientError, LLMSettingsError):
            return False

    def replan_from_instruction(self, plan_id: str, instruction: str) -> bool:
        """User-driven structural replan (driver §3 提指令→重规划): regenerate the
        remaining steps to satisfy a free-text instruction, then persist. Returns True on
        success, False when no planner, the replan budget is exhausted, or the LLM cannot
        produce a valid revised plan (the caller then keeps the current plan)."""
        if self._planner is None:
            return False
        plan = self._repo.load_plan(plan_id)
        tier = resolve_tier(plan.tier)
        if plan.replan_count >= tier.max_replan_iterations:
            return False
        pending = [s for s in plan.steps if s.status not in {StepStatus.DONE, StepStatus.SKIPPED}]
        trigger = pending[0] if pending else (plan.steps[-1] if plan.steps else None)
        if trigger is None:
            return False
        try:
            new_plan = self._planner.replan(
                plan,
                completed_summaries=self._summaries(plan),
                observation={"reason": "user_instruction", "instruction": instruction},
                reason="user_instruction",
                tier=tier,
                instruction=instruction,
            )
            self._repo.replace_remaining_steps(
                plan.id,
                new_plan,
                loop_event={"type": "replan", "reason": "user_instruction", "instruction": instruction},
            )
            self._dispatch(
                "plan.replanned",
                {"plan_id": plan.id, "reason": "user_instruction", "trigger_step_id": trigger.id},
                task_id=plan.task_id,
            )
            return True
        except (KeyError, PlanningError, LLMClientError, LLMSettingsError):
            # Replan is a best-effort enhancement. In manual mode (no LLM configured) the
            # planner cannot replan — that is NOT a flow error; swallow it and let the plan
            # continue to its confirmation gate. PlanningError covers ReplanError + invalid
            # replans; LLMClientError/LLMSettingsError cover unavailable models.
            return False

    def _try_append_explore_segment(self, plan: Plan, tier: CapabilityTier) -> bool:
        if self._planner is None:
            return False
        try:
            segment, done = self._planner.next_explore_segment(
                plan,
                completed_summaries=self._summaries(plan),
                tier=tier,
            )
        except (ReplanError, LLMClientError, LLMSettingsError):
            return False
        if done or not segment:
            return False
        self._repo.append_steps(
            plan.id,
            segment,
            loop_event={
                "type": "explore_segment",
                "reason": "explore_segment",
            },
        )
        self._dispatch(
            "plan.replanned",
            {"plan_id": plan.id, "reason": "explore_segment"},
            task_id=plan.task_id,
        )
        return True

    def _summaries(self, plan: Plan) -> dict[str, dict]:
        summaries = {}
        for step in plan.steps:
            if step.status not in {StepStatus.DONE, StepStatus.SKIPPED} or not step.output_ref:
                continue
            try:
                output = self._repo.load_step_output(step.id)
            except KeyError:
                continue
            summaries[step.id] = summarize_output(output, self._tool_spec(step))
        return summaries

    def _observation(self, step: PlanStep, reason: str) -> dict:
        if reason == "failure":
            return summarize_failure(step.error or "", "execution")
        try:
            return summarize_output(self._repo.load_step_output(step.id), self._tool_spec(step))
        except KeyError:
            return {}

    def _tool_spec(self, step: PlanStep):
        tools = getattr(self._runner, "_tools", None)
        if tools is None:
            return None
        try:
            return tools.resolve(step.tool_ref)
        except Exception:
            return None

    def _requires_effect_authorization(self, step: PlanStep) -> bool:
        step_policy = getattr(step, "policy", None)
        if getattr(step_policy, "effect_authorization", "none") == "required":
            return True
        tool = self._tool_spec(step)
        tool_policy = getattr(tool, "policy", None)
        return getattr(tool_policy, "effect_authorization", "none") == "required"

    def _requires_human_decision(self, step: PlanStep) -> bool:
        return self._requires_governed_human_decision(step) or bool(
            step.needs_confirmation
        )

    def _special_value_requires_decision(
        self,
        plan: Plan,
        step: PlanStep,
    ) -> bool:
        """Return whether ``resolve_special_values`` has a real HITL decision.

        The template marks the step as a canonical human-decision gate so AUTO
        can never silently choose mask/retain/drop.  That policy is conditional
        on actual evidence, though: when the completed screen selected no
        sentinel-bearing columns, the tool is a deterministic no-op and should
        run without showing an empty confirmation or requiring an authorization
        binding. Missing/malformed screen evidence fails closed.
        """

        if step.tool_ref.tool != "resolve_special_values":
            return True
        for dependency_id in step.depends_on or []:
            dependency = _find_step(plan, dependency_id)
            if (
                dependency is None
                or dependency.tool_ref.tool != "screen_features"
            ):
                continue
            try:
                output = self._repo.load_step_output(dependency.id)
            except KeyError:
                return True
            if not isinstance(output, dict):
                return True
            raw_selected = output.get("selected")
            if not isinstance(raw_selected, list):
                return True
            selected = {
                str(item).strip()
                for item in raw_selected
                if str(item).strip()
            }
            sentinel_columns = output.get("sentinel_columns")
            if not isinstance(sentinel_columns, dict):
                return True
            return any(
                str(column) in selected and bool(rows)
                for column, rows in sentinel_columns.items()
            )
        return True

    def _requires_governed_human_decision(self, step: PlanStep) -> bool:
        step_policy = getattr(step, "policy", None)
        if (
            getattr(step_policy, "human_decision_gate", "none") == "required"
            or getattr(step_policy, "effect_authorization", "none") == "required"
        ):
            return True
        tool = self._tool_spec(step)
        tool_policy = getattr(tool, "policy", None)
        if (
            getattr(tool_policy, "human_decision_gate", "none") == "required"
            or getattr(tool_policy, "effect_authorization", "none") == "required"
        ):
            return True
        return False

    def _is_governed_step(self, step: PlanStep) -> bool:
        return self._requires_governed_human_decision(
            step
        ) or self._requires_effect_authorization(step)

    def _resolve_execution_context(
        self,
        plan: Plan,
        step: PlanStep,
        resolved_inputs: dict,
    ):
        authorizer = self._authorizer
        if authorizer is None:
            return None
        resolve = getattr(authorizer, "execution_context_for", None)
        if callable(resolve):
            return resolve(plan=plan, step=step, inputs=resolved_inputs)
        if not callable(authorizer):
            raise TypeError("authorizer must be callable or expose execution_context_for")
        return authorizer(plan=plan, step=step, inputs=resolved_inputs)

    def _no_progress(self, plan: Plan, failed_step: PlanStep) -> bool:
        try:
            recent = self._repo.recent_failed_tool_refs(plan.id, limit=NO_PROGRESS_WINDOW)
        except Exception:
            return False
        return recent.count(failed_step.tool_ref.label()) >= NO_PROGRESS_THRESHOLD

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


def _dig(value: Any, path: str, *, ref: str):
    current: Any = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index < len(current):
                current = current[index]
                continue
        raise RefResolutionError(ref, f"path segment {part!r} is missing")
    return current


def _payload_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bounded_input_summary(payload: dict) -> dict:
    summary = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, list):
            summary[key] = {"type": "list", "len": len(value), "sample": value[:5]}
        elif isinstance(value, dict):
            summary[key] = {"type": "dict", "keys": sorted(str(item) for item in value.keys())[:20]}
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def _tool_manifest_details(registry, ref) -> tuple[str | None, str | None]:
    if registry is None or not hasattr(registry, "resolve_with_manifest"):
        return ref.version or None, None
    try:
        manifest, _tool = registry.resolve_with_manifest(ref)
    except Exception:
        return ref.version or None, None
    version = str(getattr(manifest, "version", "") or ref.version or "").strip() or None
    checksum = str(getattr(manifest, "checksum", "") or "").strip()
    if checksum:
        manifest_hash = checksum if checksum.startswith("sha256:") else f"sha256:{checksum}"
    else:
        manifest_hash = _payload_hash(manifest_to_dict(manifest))
    return version, manifest_hash


def _dataset_refs(payload: Any) -> list[str]:
    refs: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        normalized = key.lower()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key)
            return
        if not isinstance(value, str) or not value.strip():
            return
        text = value.strip()
        if text.startswith("dataset:"):
            _append_unique(refs, text)
        elif normalized.endswith("dataset_id") or normalized.endswith("dataset_ids"):
            _append_unique(refs, f"dataset:{text}")

    visit(payload)
    return refs


def _artifact_refs(payload: Any) -> list[str]:
    refs: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        normalized = key.lower()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key)
            return
        if not isinstance(value, str) or not value.strip():
            return
        text = value.strip()
        if text.startswith("artifact:"):
            _append_unique(refs, text)
        elif normalized == "path" or normalized.endswith("_path"):
            _append_unique(refs, f"artifact:{text}")
        elif normalized.endswith("artifact_id") or normalized.endswith("artifact_ref"):
            _append_unique(refs, f"artifact:{text}")

    visit(payload)
    return refs


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _parent_output_refs(repo, step: PlanStep) -> list[str]:
    refs = []
    for dep_id in step.depends_on:
        try:
            ref = repo.latest_step_output_ref(dep_id)
        except Exception:
            ref = None
        if ref:
            refs.append(ref)
    return refs


_METRIC_SCAN_MAX_DEPTH = 3


def _output_has_metrics(output: Any, *, _depth: int = 0) -> bool:
    """True when a tool output carries at least one metric-shaped value (a known
    METRIC_FIELDS key, a numeric leaf whose name carries a metric token like
    "oot_ks"/"test_auc", or a plain numeric value nested in the output) — the
    AGT-6 trigger surface for llm_critique on steps that aren't already a
    decision_point/needs_confirmation gate. Bounded depth keeps this a cheap
    pre-check, not a full tree walk."""
    if _depth > _METRIC_SCAN_MAX_DEPTH:
        return False
    if isinstance(output, dict):
        for key, value in output.items():
            name = str(key)
            if name in METRIC_FIELDS or any(part in METRIC_FIELDS for part in name.split("_")):
                return True
            if _output_has_metrics(value, _depth=_depth + 1):
                return True
        return False
    if isinstance(output, (list, tuple)):
        return any(_output_has_metrics(item, _depth=_depth + 1) for item in output)
    return False


def _find_step(plan: Plan, step_id: str) -> PlanStep | None:
    for step in plan.steps:
        if step.id == step_id:
            return step
    return None


def _last_executed_step(plan: Plan) -> PlanStep | None:
    executed = [
        step
        for step in plan.steps
        if step.status in {StepStatus.DONE, StepStatus.SKIPPED}
    ]
    return max(executed, key=lambda step: (step.index, step.id), default=None)


def _review_warning_payload(step: PlanStep) -> dict[str, Any]:
    warnings = [
        {
            "reviewer": verdict.reviewer,
            "reasons": list(verdict.reasons),
        }
        for verdict in step.review_verdicts
        if not verdict.passed
    ]
    return {
        "review_warning_count": len(warnings),
        "review_warnings": warnings,
    }


def _final_review_failure_replannable(review: FinalReview) -> bool:
    return not any(
        "invalid " in item and " threshold" in item
        for item in review.open_items
    )


def _has_deterministic_failure(step: PlanStep) -> bool:
    return any(
        verdict.reviewer == "deterministic" and not verdict.passed
        for verdict in step.review_verdicts
    )


def _is_fatal_error(error: str | None) -> bool:
    lowered = str(error or "").lower()
    return any(
        marker in lowered
        for marker in (
            "schema",
            "contract",
            "explicit retry required",
            "interrupted during running",
        )
    )
