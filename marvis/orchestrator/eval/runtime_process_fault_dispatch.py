"""Declared fault dispatch on an actual API-owned worker and explicit recovery."""
import json
from pathlib import Path
import time

from .runtime_contracts import digest
from .runtime_fault_launcher import arm_process_fault, attached_launch, downstream_commit_target
from .runtime_process_faults import KINDS, verify_worker_crash_originals, write_original


def _terminal(fault, deadline):
    path = fault.directory / "terminal.json"
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(0.01)
    raise TimeoutError("process_fault_terminal_not_observed")


def dispatch_process_fault(journey, action, workspace, plan, step, body, *, actor=None):
    from .runtime_fault_capture import capture_target_original

    workspace = Path(workspace).resolve(strict=True)
    if action.fault.kind not in KINDS:
        return False
    if getattr(journey, "runtime_faults", ()):
        raise ValueError("process_fault_requires_one_exact_fault_per_journey")
    target_id, target_tool = step["id"], action.tool
    downstream = action.fault.downstream_tool
    if downstream is not None:
        if (action.fault.kind != "worker_crash_after_commit_once" or (action.tool, downstream) not in {("analysis.portfolio_gate_summary", "analysis.portfolio_report"),
                    ("modeling.configure_tuning", "modeling.train_models")}):
            raise ValueError("process_fault_downstream_action_invalid")
        report, _ = downstream_commit_target(workspace, journey.task_id, plan["id"], step["id"], downstream)
        target_id, target_tool = report["id"], downstream
    row = {"schema": "marvis.runtime-process-fault.v1", "action_index": journey.current_action_index,
           "action_sha256": digest(action.model_dump()), "kind": action.fault.kind,
           "task_id": journey.task_id, "plan_id": plan["id"], "step_id": target_id,
           "tool": target_tool, "actor": actor, "original_request": body,
           "before": capture_target_original(workspace, journey.task_id, plan["id"], target_id,
                                             origin_ns=journey.origin_ns),
           "fault_executed": False, "recovery_outcome": "not_evaluated"}
    if downstream is not None:
        row.update(approval_step_id=step["id"], approval_tool=action.tool)
    journey.runtime_faults = [row]  # preserve partial failure, never manufacture a receipt
    directory = arm_process_fault(workspace, kind=action.fault.kind, task_id=journey.task_id,
                                  plan_id=plan["id"], step_id=target_id, tool=target_tool,
                                  approval_step_id=step["id"] if downstream is not None else None)
    response = journey.request("POST", f"/api/plans/{plan['id']}/steps/{step['id']}/decisions",
                               label="process_fault_native_approval", json=body)
    row.update(http_ordinal=journey.count, http_status=response.status_code)
    if response.status_code != 202:
        raise ValueError("process_fault_approval_not_accepted")
    row["received_response"] = response.json()
    decision = {"tool": action.tool, "actor": actor, "request": body, "response": response.json(),
                "http_ordinal": journey.count, "case_action_index": journey.current_action_index,
                "action_sha256": digest(action.model_dump()), "fault_kind": action.fault.kind, "plan_before": plan}
    journey.human_decisions = [*getattr(journey, "human_decisions", []), decision]
    fault = attached_launch(directory, deadline=min(journey.deadline, time.monotonic() + 15))
    ready = fault.ready(timeout=max(0.01, min(15, journey.deadline - time.monotonic())))
    row.update(ready=ready, originals_relative_path=str(directory.relative_to(workspace)))
    if action.fault.kind == "worker_crash_after_commit_once":
        fault.kill_worker(ready)
        _terminal(fault, journey.deadline)
        row["fault_verification"] = verify_worker_crash_originals(fault.directory, fault.job)
    elif action.fault.kind == "tool_timeout_once":
        terminal = _terminal(fault, journey.deadline)
        if (terminal["terminal"] != "timeout" or terminal["observed_timeout_seconds"] != 2
                or terminal["finished_elapsed_ns"] - terminal["started_elapsed_ns"] < 2_000_000_000):
            raise ValueError("process_fault_native_timeout_not_observed")
        row["fault_verification"] = {"fault_executed": True, "terminal": terminal}
    elif action.fault.kind == "cancel_then_resume_once":
        cancelled = journey.request("POST", f"/api/plans/{plan['id']}/cancel", label="process_fault_plan_cancel")
        if cancelled.status_code != 200:
            raise ValueError("process_fault_native_cancel_not_accepted")
        row["cancellation"] = {"http_ordinal": journey.count, "status": 200, "body": cancelled.json()}
        # This API cancels at a plan boundary. Releasing the real worker permits
        # the existing checkpoint to settle; do not pretend the API killed it.
        write_original(fault.directory / "release", {"reason": "native_plan_cancel_accepted"})
        row["fault_verification"] = {"fault_executed": True, "cancellation_boundary": "native_plan_checkpoint",
                                     "terminal": _terminal(fault, journey.deadline)}
    else:
        control = getattr(journey, "owned_process_control", None)
        if control is None:
            raise ValueError("process_fault_service_control_unavailable")
        row["restart"] = control.restart_same_workspace()
        write_original(directory / "restart.json", row["restart"])
        if row["restart"]["before"]["pid"] != ready["worker"]["ppid"]:
            raise ValueError("process_fault_restarted_wrong_service")
        row["fault_verification"] = {"fault_executed": True, "restart": row["restart"]}
    row["fault_executed"] = True
    settled = journey.wait_idle(changed_step=target_id)
    decision["plan_after"] = next(item for item in settled if item["id"] == plan["id"])
    row["after_fault"] = capture_target_original(workspace, journey.task_id, plan["id"], target_id,
                                                origin_ns=journey.origin_ns)
    return True


def recover_process_fault(journey, action, workspace, row):
    from .runtime_fault_capture import capture_target_original

    if row.get("recovery") or not row.get("fault_executed") or row.get("approval_tool", row["tool"]) != action.tool:
        raise ValueError("process_recovery_requires_one_executed_matching_fault")
    journey.record_human_action(action)
    ordinal = journey.count
    plans = journey.plans()
    matches = [plan for plan in plans if plan["id"] == row["plan_id"] and plan["task_id"] == row["task_id"]]
    if len(matches) != 1:
        raise ValueError("process_recovery_plan_binding_missing")
    plan = matches[0]
    targets = [target for target in plan.get("reconciliation", {}).get("targets", [])
               if target.get("kind") == "tool" and target.get("supported") is True
               and target.get("binding", {}).get("step_id") == row["step_id"]]
    target_steps = [step for step in plan["steps"] if step["id"] == row["step_id"]]
    if len(target_steps) != 1:
        raise ValueError("process_recovery_step_binding_missing")
    response_status = None
    if target_steps[0]["status"] != "done" or plan["status"] == "cancelled":
        if len(targets) == 1:
            response = journey.request("POST", f"/api/plans/{row['plan_id']}/reconcile",
                label="process_fault_native_reconcile", json={"target_id": targets[0]["id"]})
        else:
            # Retry safety remains entirely owned by the production API. A 409
            # / unknown_effect is a recorded recovery failure, never a bypass.
            response = journey.request("POST", f"/api/plans/{row['plan_id']}/steps/{row['step_id']}/retry",
                label="process_fault_native_retry", json={})
        response_status = response.status_code
    journey.wait_idle()
    final = capture_target_original(workspace, journey.task_id, row["plan_id"], row["step_id"],
                                    origin_ns=journey.origin_ns)
    recovered = (final["target"]["plan_status"] == "done" and final["target"]["status"] == "done"
                 and bool(final["target"]["output_ref"]))
    row["recovery_outcome"] = "done" if recovered else "not_recovered"
    row["recovery"] = {"action_index": journey.current_action_index,
        "action_sha256": digest(action.model_dump()), "http_ordinals": list(range(ordinal + 1, journey.count + 1)),
        "response_status": response_status, "after_readback": final, "outcome": row["recovery_outcome"]}
