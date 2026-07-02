from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from marvis.orchestrator.capability import CapabilityTier, resolve_tier
from marvis.orchestrator.context.observation import summarize_failure, summarize_output
from marvis.orchestrator.contracts import (
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from marvis.llm_settings import LLMSettingsError
from marvis.orchestrator.planner import PlanningError, ReplanError
from marvis.orchestrator.plan_recovery import PlanStepRecovery
from marvis.orchestrator.reviewer import FinalReview
from marvis.orchestrator.safety import is_safety_step
from marvis.plugins.manifest import manifest_to_dict
from marvis.plugins.runner import ToolResult


MAX_STEP_RETRIES = 1
NO_PROGRESS_WINDOW = 4
NO_PROGRESS_THRESHOLD = 2


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
    ):
        self._repo = plan_repo
        self._runner = tool_runner
        self._reviewer = reviewer
        self._subagents = subagent_dispatcher
        self._hooks = hook_dispatcher
        self._state = harness_state
        self._planner = planner
        self._step_recovery = PlanStepRecovery(plan_repo, reviewer, hook_dispatcher, harness_state)

    def run(self, plan_id: str) -> ExecutionResult:
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
        if plan.status in {PlanStatus.CONFIRMED, PlanStatus.AWAITING_CONFIRM}:
            self._set_plan_status(plan, PlanStatus.RUNNING)
        self._step_recovery.recover_inflight_steps(plan)

        while True:
            plan = self._repo.load_plan(plan_id)
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
            if step.needs_confirmation and not self._repo.is_step_confirmed(step.id):
                self._set_step_status(step, StepStatus.AWAITING_CONFIRM)
                self._set_plan_status(plan, PlanStatus.AWAITING_CONFIRM)
                return ExecutionResult(plan.id, PlanStatus.AWAITING_CONFIRM, None, None)
            self._execute_step(plan, step)
            plan = self._repo.load_plan(plan_id)
            last = _find_step(plan, step.id)
            if (
                last is not None
                and last.status == StepStatus.DONE
                and last.decision_point
                and tier.decision_point_replan
                and not is_safety_step(last)
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

    def _execute_step(self, plan: Plan, step: PlanStep) -> None:
        run_id = None
        try:
            self._set_step_status(step, StepStatus.RUNNING)
            resolved_inputs = self._resolve_refs(step.inputs)
            run_id = self._repo.start_step_run(
                plan_id=plan.id,
                step_id=step.id,
                tool_ref=step.tool_ref.label(),
                inputs=resolved_inputs,
            )
            result = self._invoke_step(plan, step, resolved_inputs)
            if not result.ok:
                self._finish_step_run(
                    run_id,
                    status="failed",
                    error=result.error or "step failed",
                    error_kind=result.error_kind,
                    duration_ms=result.duration_ms,
                )
                self._handle_step_failure(step, result)
                return

            output = result.output or {}
            self._set_step_status(step, StepStatus.CHECKING)
            step.output_ref = self._repo.store_step_output(
                step.id,
                output,
                evidence=self._step_evidence(step, resolved_inputs, output),
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
                self._handle_step_failure(step, failed, apply_policy=False)
                return

            critique = self._reviewer.llm_critique(step, output, plan.goal)
            step.review_verdicts.append(critique)
            step.status = StepStatus.DONE
            self._repo.update_step(step)
            self._dispatch_step_completed(plan, step, output)
        except Exception as exc:
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

    def _finish_step_run(self, run_id: str, **kwargs) -> None:
        self._repo.finish_step_run(run_id, **kwargs)

    def _step_evidence(self, step: PlanStep, resolved_inputs: dict, output: dict) -> dict:
        seed = resolved_inputs.get("seed")
        tool_version, manifest_hash = _tool_manifest_details(getattr(self._runner, "_tools", None), step.tool_ref)
        return {
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
        except (KeyError, PlanningError, LLMSettingsError):
            # Replan is a best-effort enhancement. In manual mode (no LLM configured) the
            # planner cannot replan — that is NOT a flow error; swallow it and let the plan
            # continue to its confirmation gate. PlanningError covers ReplanError + invalid
            # replans; LLMSettingsError covers "no enabled model".
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
        except (KeyError, PlanningError, LLMSettingsError):
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
        except (KeyError, PlanningError, LLMSettingsError):
            # Replan is a best-effort enhancement. In manual mode (no LLM configured) the
            # planner cannot replan — that is NOT a flow error; swallow it and let the plan
            # continue to its confirmation gate. PlanningError covers ReplanError + invalid
            # replans; LLMSettingsError covers "no enabled model".
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
        except ReplanError:
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


def _dig(value: dict, path: str):
    current: Any = value
    for part in path.split("."):
        if not part:
            return None
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
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

