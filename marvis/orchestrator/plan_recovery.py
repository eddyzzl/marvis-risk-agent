from __future__ import annotations

from marvis.orchestrator.contracts import Plan, PlanStep, StepStatus
from marvis.plugins.runner import ToolResult


class PlanStepRecovery:
    """Reclaims RUNNING/CHECKING plan steps against the step-run ledger.

    Extracted from ``PlanExecutor`` (REL-1/REL-4) so the exact same recovery
    semantics can run from two call sites: lazily, the first time ``run()``
    resumes a plan after a crash (``PlanExecutor._recover_inflight_steps``),
    and eagerly, from the app-startup reclaim pass (``marvis.recovery``) so a
    RUNNING plan doesn't spin forever waiting for the next user message.
    """

    def __init__(self, plan_repo, reviewer, hook_dispatcher, harness_state):
        self._repo = plan_repo
        self._reviewer = reviewer
        self._hooks = hook_dispatcher
        self._state = harness_state

    def recover_inflight_steps(self, plan: Plan) -> None:
        running_runs: dict[str, list[dict]] = {}
        for run in self._repo.list_running_step_runs(plan.id):
            running_runs.setdefault(str(run["step_id"]), []).append(run)
        for step in plan.steps:
            step_runs = running_runs.get(step.id, [])
            if step.status == StepStatus.RUNNING:
                latest_output_ref = step.output_ref or (
                    self._repo.latest_step_output_ref(step.id) if step_runs else None
                )
                if latest_output_ref:
                    step.output_ref = latest_output_ref
                    self._recover_step_runs(
                        step_runs,
                        status="succeeded",
                        output_ref=latest_output_ref,
                    )
                    self._recover_checking_step(plan, step)
                    continue
                step.error = (
                    "interrupted during running before output was persisted; "
                    "explicit retry required"
                )
                self._recover_step_runs(
                    step_runs,
                    status="interrupted",
                    error=step.error,
                    error_kind="ServerRestart",
                )
                self._set_step_status(step, StepStatus.FAILED)
            elif step.status == StepStatus.CHECKING:
                latest_output_ref = step.output_ref or (
                    self._repo.latest_step_output_ref(step.id) if step_runs else None
                ) or (
                    self._repo.latest_succeeded_step_run_output_ref(step.id)
                )
                if latest_output_ref:
                    step.output_ref = latest_output_ref
                    self._recover_step_runs(
                        step_runs,
                        status="succeeded",
                        output_ref=latest_output_ref,
                    )
                else:
                    self._recover_step_runs(
                        step_runs,
                        status="interrupted",
                        error="interrupted during checking before output was persisted",
                        error_kind="ServerRestart",
                    )
                self._recover_checking_step(plan, step)

    def _recover_step_runs(self, runs: list[dict], **kwargs) -> None:
        for run in runs:
            run_id = str(run.get("id") or "")
            if run_id:
                try:
                    self._repo.finish_step_run(run_id, **kwargs)
                except Exception:
                    continue

    def _recover_checking_step(self, plan: Plan, step: PlanStep) -> None:
        version = _step_output_version(step)
        if version is None:
            step.error = "interrupted during checking before output was persisted"
            self._set_step_status(step, StepStatus.FAILED)
            return
        try:
            output = self._repo.load_step_output(step.id, version=version)
        except KeyError:
            step.error = "interrupted during checking before output was persisted"
            self._set_step_status(step, StepStatus.FAILED)
            return
        deterministic = self._reviewer.deterministic_check(step, output)
        step.review_verdicts.append(deterministic)
        if not deterministic.passed:
            failed = ToolResult(
                ok=False,
                output=None,
                error="; ".join(deterministic.reasons),
                error_kind="postcheck",
                duration_ms=0,
            )
            self._handle_step_failure(step, failed)
            return
        critique = self._reviewer.llm_critique(step, output, plan.goal)
        step.review_verdicts.append(critique)
        step.status = StepStatus.DONE
        self._repo.update_step(step)
        self._dispatch_step_completed(plan, step, output)

    def _handle_step_failure(self, step: PlanStep, result: ToolResult) -> None:
        step.error = result.error or "step failed"
        self._set_step_status(step, StepStatus.FAILED)

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
            },
            task_id=plan.task_id,
        )

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


def _step_output_version(step: PlanStep) -> int | None:
    ref = str(step.output_ref or "")
    prefix = f"metrics:{step.id}:v"
    if not ref.startswith(prefix):
        return None
    version_text = ref[len(prefix):]
    if not version_text.isdigit():
        return None
    return int(version_text)
