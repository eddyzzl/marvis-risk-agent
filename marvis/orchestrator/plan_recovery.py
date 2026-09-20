from __future__ import annotations

from marvis.orchestrator.contracts import Plan, PlanStep, StepStatus
from marvis.orchestrator.completion import complete_step
from marvis.plugins.invocation import retry_safe_run
from marvis.state_machine import ConflictError


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
                latest_output_ref = step.output_ref or self._current_run_output_ref(
                    step.id,
                    step_runs,
                )
                if latest_output_ref:
                    step.output_ref = latest_output_ref
                    self._repo.update_step(step)
                    self._recover_step_runs_for_output(
                        step_runs,
                        output_ref=latest_output_ref,
                    )
                    self._recover_checking_step(plan, step)
                    continue
                safe = bool(step_runs) and all(retry_safe_run(run) for run in step_runs)
                step.error = "interrupted during running before output was persisted; " + (
                    "explicit retry required" if safe else "explicit reconciliation required"
                )
                self._recover_step_runs(
                    step_runs,
                    status="interrupted",
                    error=step.error,
                    error_kind="ServerRestart" if safe else "unknown_effect",
                )
                self._set_step_status(step, StepStatus.FAILED)
            elif step.status == StepStatus.CHECKING:
                latest_output_ref = step.output_ref or self._current_run_output_ref(
                    step.id,
                    step_runs,
                )
                if latest_output_ref is None and not step_runs:
                    latest_output_ref = self._repo.latest_succeeded_step_run_output_ref(
                        step.id
                    )
                if latest_output_ref:
                    step.output_ref = latest_output_ref
                    self._repo.update_step(step)
                    self._recover_step_runs_for_output(
                        step_runs,
                        output_ref=latest_output_ref,
                    )
                else:
                    self._recover_step_runs(
                        step_runs,
                        status="interrupted",
                        error="interrupted during checking before output was persisted; explicit reconciliation required",
                        error_kind="unknown_effect",
                    )
                self._recover_checking_step(plan, step)

    def _current_run_output_ref(self, step_id: str, runs: list[dict]) -> str | None:
        return self._repo.latest_step_output_ref_for_runs(
            step_id,
            run_ids=[str(run.get("id") or "") for run in runs],
        )

    def _recover_step_runs(self, runs: list[dict], **kwargs) -> None:
        for run in runs:
            run_id = str(run.get("id") or "")
            if run_id:
                try:
                    self._repo.finish_step_run(run_id, **kwargs)
                except Exception:
                    continue

    def _recover_step_runs_for_output(
        self,
        runs: list[dict],
        *,
        output_ref: str,
    ) -> None:
        """Close only the run durably bound to ``output_ref`` as successful.

        Multiple attempts can be left RUNNING by a process crash.  An output is
        immutable evidence for exactly one attempt; older/unbound attempts must
        be interrupted instead of inheriting another attempt's result.
        """

        for run in runs:
            run_id = str(run.get("id") or "")
            if not run_id:
                continue
            bound_ref = str(run.get("output_ref") or "")
            try:
                if bound_ref == output_ref:
                    self._repo.finish_step_run(
                        run_id,
                        status="succeeded",
                        output_ref=output_ref,
                    )
                else:
                    self._repo.finish_step_run(
                        run_id,
                        status="interrupted",
                        error=(
                            "superseded by the execution attempt bound to the "
                            "persisted output"
                        ),
                        error_kind="ServerRestart",
                    )
            except Exception:
                continue

    def _recover_checking_step(self, plan: Plan, step: PlanStep) -> None:
        version = _step_output_version(step)
        if version is None:
            step.error = "interrupted during checking before output was persisted; explicit reconciliation required"
            self._set_step_status(step, StepStatus.FAILED)
            return
        try:
            load_recovery_binding = getattr(
                self._repo,
                "load_step_recovery_binding",
                None,
            )
            if callable(load_recovery_binding):
                binding = load_recovery_binding(step.id, str(step.output_ref))
                output = binding["output"]
            else:
                output = self._repo.load_step_output(step.id, version=version)
        except ConflictError as exc:
            step.error = (
                "persisted step output failed integrity checks during recovery: "
                f"{exc}; explicit reconciliation required"
            )
            self._set_step_status(step, StepStatus.FAILED)
            return
        except (KeyError, TypeError, ValueError):
            step.error = "interrupted during checking before output was persisted; explicit reconciliation required"
            self._set_step_status(step, StepStatus.FAILED)
            return
        if step.status == StepStatus.RUNNING:
            self._set_step_status(step, StepStatus.CHECKING)
        try:
            complete_step(self._repo, self._reviewer, self._hooks, plan, step, output)
        except Exception as exc:
            step.error = f"step completion interrupted: {exc}; explicit reconciliation required"
            self._set_step_status(step, StepStatus.FAILED)

    def _set_step_status(self, step: PlanStep, status: StepStatus) -> None:
        if step.status != status:
            self._state.assert_step_transition(step.status, status)
            step.status = status
        self._repo.update_step(step)


def _step_output_version(step: PlanStep) -> int | None:
    ref = str(step.output_ref or "")
    prefix = f"metrics:{step.id}:v"
    if not ref.startswith(prefix):
        return None
    version_text = ref[len(prefix):]
    if not version_text.isdigit():
        return None
    return int(version_text)
