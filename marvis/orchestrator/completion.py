"""One completion protocol for execution and crash recovery.

Tool execution has already finished here. Durable output identity, review
snapshots and Hook receipts decide completion; this module never invokes a Tool
again to repair an uncertain side effect.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from marvis.orchestrator.contracts import PlanStatus, ReviewVerdict, StepStatus
from marvis.orchestrator.evidence import payload_hash
from marvis.orchestrator.validator import METRIC_FIELDS
from marvis.repositories.hook_deliveries import HookDeliveryRepository


class CompletionError(RuntimeError):
    """Completion requires explicit reconciliation; never replan/reexecute."""


def output_has_metrics(output: Any, *, _depth: int = 0) -> bool:
    if _depth > 3:
        return False
    if isinstance(output, dict):
        return any(
            str(key) in METRIC_FIELDS
            or any(part in METRIC_FIELDS for part in str(key).split("_"))
            or output_has_metrics(value, _depth=_depth + 1)
            for key, value in output.items()
        )
    if isinstance(output, (list, tuple)):
        return any(output_has_metrics(item, _depth=_depth + 1) for item in output)
    return False


def review_warning_payload(step) -> dict:
    warnings = [
        {"reviewer": verdict.reviewer, "reasons": list(verdict.reasons)}
        for verdict in step.review_verdicts
        if not verdict.passed
    ]
    return {"review_warning_count": len(warnings), "review_warnings": warnings}


def _ledger(repo) -> HookDeliveryRepository | None:
    path = getattr(repo, "db_path", None)
    return HookDeliveryRepository(path) if path is not None else None


def _dispatch(repo, hooks, event: str, payload: dict, task_id: str) -> list[dict]:
    if hooks is None:
        ledger = _ledger(repo)
        if ledger and any(
            target["required"] for target in ledger.event_targets(payload["event_id"])
        ):
            raise CompletionError(
                "required hook dispatcher unavailable; explicit reconciliation required"
            )
        return []
    try:
        results = hooks.dispatch(event, payload, task_id=task_id)
    except Exception as exc:
        raise CompletionError(
            f"hook completion outcome unknown ({type(exc).__name__}); explicit reconciliation required"
        ) from exc
    failures = getattr(results, "required_failures", [])
    if failures:
        labels = "; ".join(
            f"{item['target_ref']}: {item['error']}" for item in failures
        )
        raise CompletionError(
            f"required hook completion failed: {labels}; explicit reconciliation required"
        )
    return list(getattr(results, "warnings", []))


def _prepare_events(repo, hooks, events: list[tuple[str, str]], task_id: str) -> None:
    prepare = getattr(hooks, "prepare_many", None)
    if callable(prepare):
        prepare(events, task_id=task_id)
    else:
        ledger = _ledger(repo)
        if ledger:
            ledger.prepare_events(
                [
                    {
                        "event_id": identity,
                        "event": event,
                        "task_id": task_id,
                        "targets": [],
                    }
                    for event, identity in events
                ]
            )


def step_completion_identity(repo, plan, step, output=None) -> tuple[dict, dict, dict]:
    """Authenticate the original result for both completion and reconciliation."""
    if output is None:
        output = repo.load_step_output(step.id)
    binding_loader = getattr(repo, "load_step_recovery_binding", None)
    if callable(binding_loader):
        binding = binding_loader(step.id, str(step.output_ref))
        output, evidence = binding["output"], binding["evidence"]
    else:
        evidence_loader = getattr(repo, "load_step_evidence", None)
        version = int(str(step.output_ref).rsplit(":v", 1)[1])
        evidence = (
            evidence_loader(step.id, version=version)
            if callable(evidence_loader)
            else {}
        )
    if evidence.get("tool_name") and evidence["tool_name"] != step.tool_ref.label():
        raise CompletionError(
            "step tool binding changed; explicit reconciliation required"
        )
    if step.tool_ref.version and evidence.get("tool_version") != step.tool_ref.version:
        raise CompletionError(
            "step tool version changed; explicit reconciliation required"
        )
    identity = {
        "plan_id": plan.id,
        "step_id": step.id,
        "output_ref": step.output_ref,
        "execution_id": evidence.get("step_run_id"),
        "output_hash": evidence.get("output_hash") or payload_hash(output),
    }
    return identity, output, evidence


def complete_step(repo, reviewer, hooks, plan, step, output: dict) -> None:
    ledger = _ledger(repo)
    identity, output, evidence = step_completion_identity(repo, plan, step, output)
    event_names = (
        ["feature.computed"] if step.tool_ref.plugin == "feature" else []
    ) + ["step.completed"]
    _prepare_events(
        repo,
        hooks,
        [(event, payload_hash({"event": event, **identity})) for event in event_names],
        plan.task_id,
    )
    checkpoint_id = f"step:{step.id}:{step.output_ref}"
    binding_hash = payload_hash(
        {
            **identity,
            "goal": plan.goal,
            "post_checks": [asdict(check) for check in step.post_checks],
            "decision_point": step.decision_point,
            "needs_confirmation": step.needs_confirmation,
            "tool_ref": asdict(step.tool_ref),
            "inputs": step.inputs,
            "policy": step.policy.to_dict(),
        }
    )
    snapshot = ledger.load_checkpoint(checkpoint_id, binding_hash) if ledger else None
    if snapshot is None:
        verdicts = [reviewer.deterministic_check(step, output)]
        if verdicts[0].passed and (
            step.decision_point or step.needs_confirmation or output_has_metrics(output)
        ):
            verdicts.append(reviewer.llm_critique(step, output, plan.goal))
        snapshot = {"verdicts": [asdict(verdict) for verdict in verdicts]}
        if ledger:
            snapshot = ledger.store_checkpoint(checkpoint_id, binding_hash, snapshot)
    step.review_verdicts = [ReviewVerdict(**item) for item in snapshot["verdicts"]]
    # Persist the review before publishing any completion event. A crash after
    # this point reuses the frozen critique instead of asking a model again.
    repo.update_step(step)
    deterministic = step.review_verdicts[0]
    if not deterministic.passed:
        step.error = "; ".join(deterministic.reasons)
        step.status = StepStatus.FAILED
        repo.update_step(step)
        return
    events = []
    if step.tool_ref.plugin == "feature":
        payload = {**identity, "tool": step.tool_ref.tool}
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
        events.append(("feature.computed", payload))
    events.append(("step.completed", {**identity, **review_warning_payload(step)}))
    warnings = []
    for event, payload in events:
        payload["event_id"] = payload_hash({"event": event, **identity})
        warnings.extend(_dispatch(repo, hooks, event, payload, plan.task_id))
    if warnings:
        step.review_verdicts.append(
            ReviewVerdict(
                reviewer="optional_hooks",
                passed=False,
                reasons=[f"{item['target_ref']}: {item['error']}" for item in warnings],
                at=datetime.now(UTC).isoformat(),
            )
        )
    previous_status, previous_error = step.status, step.error
    step.status, step.error = StepStatus.DONE, None
    try:
        repo.update_step(step)
    except BaseException:
        step.status, step.error = previous_status, previous_error
        raise


def prepare_workflow_completion(
    repo, plan, summary_ref: str, review, *, hooks=None
) -> dict:
    snapshot = {
        "summary_ref": summary_ref,
        "final_status": (
            PlanStatus.DONE if (
                review.goal_met if review.execution_completed is None
                else review.execution_completed
            ) else PlanStatus.FAILED
        ).value,
        "review": asdict(review),
        "outputs": [
            {"step_id": step.id, "output_ref": step.output_ref}
            for step in plan.steps
            if step.status == StepStatus.DONE
        ],
    }
    _prepare_events(
        repo, hooks, [("workflow.completed", _workflow_event_id(plan))], plan.task_id
    )
    ledger = _ledger(repo)
    if ledger:
        snapshot = ledger.store_checkpoint(
            f"workflow:{plan.id}", _workflow_binding(plan), snapshot
        )
    return snapshot


def pending_workflow_completion(repo, plan) -> dict | None:
    ledger = _ledger(repo)
    return (
        ledger.load_checkpoint(
            f"workflow:{plan.id}",
            _workflow_binding(plan),
        )
        if ledger
        else None
    )


def _workflow_event_id(plan) -> str:
    # A regenerated review/summary must not create a new obligation set.
    return payload_hash(
        {
            "event": "workflow.completed",
            "plan_id": plan.id,
            "binding": _workflow_binding(plan),
        }
    )


def _workflow_binding(plan) -> str:
    return payload_hash(
        {
            "task_id": plan.task_id,
            "plan_id": plan.id,
            "goal": plan.goal,
            "success_criteria": plan.success_criteria,
            "steps": [
                {
                    "id": step.id,
                    "tool_ref": asdict(step.tool_ref),
                    "inputs": step.inputs,
                    "policy": step.policy.to_dict(),
                    "output_ref": step.output_ref,
                    "status": step.status.value,
                    "post_checks": [asdict(check) for check in step.post_checks],
                }
                for step in plan.steps
            ],
        }
    )


def complete_workflow(repo, hooks, state, plan, snapshot: dict) -> list[dict]:
    # Revalidate bound outputs on restart; a completion snapshot is not an
    # alternative to the existing immutable evidence checks.
    for item in snapshot["outputs"]:
        step = next((step for step in plan.steps if step.id == item["step_id"]), None)
        if (
            step is None
            or step.status != StepStatus.DONE
            or step.output_ref != item["output_ref"]
        ):
            raise CompletionError(
                "workflow output binding changed; explicit reconciliation required"
            )
        loader = getattr(repo, "load_bound_step_output", None)
        if callable(loader):
            loader(step.id)
    if plan.status == PlanStatus.RUNNING:
        state.assert_plan_transition(plan.status, PlanStatus.REVIEW)
        repo.set_plan_status(plan.id, PlanStatus.REVIEW)
        plan.status = PlanStatus.REVIEW
    payload = {
        "plan_id": plan.id,
        "summary_ref": snapshot["summary_ref"],
        "final_status": snapshot["final_status"],
    }
    payload["event_id"] = _workflow_event_id(plan)
    warnings = _dispatch(repo, hooks, "workflow.completed", payload, plan.task_id)
    if warnings:
        repo.append_loop_event(
            plan.id,
            {
                "type": "hook_warning",
                "reason": "optional workflow completion hook failed; see hook delivery ledger",
            },
        )
    final_status = PlanStatus(snapshot["final_status"])
    state.assert_plan_transition(plan.status, final_status)
    repo.set_plan_status(plan.id, final_status)
    plan.status = final_status
    return warnings
