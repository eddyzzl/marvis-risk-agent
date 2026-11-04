"""Native monitoring plan-confirmation CAS and actual scoring-worker timeout."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import time

from .runtime_contracts import digest
from .runtime_fault_capture import capture_target_original
from .runtime_fault_launcher import arm_process_fault, attached_launch
from .runtime_process_fault_dispatch import _terminal
from .runtime_process_faults import write_original


def _state(journey, plan_id, step_id):
    workspace = Path(journey.workspace).resolve()
    target = capture_target_original(workspace, journey.task_id, plan_id, step_id, origin_ns=journey.origin_ns)
    with closing(sqlite3.connect((workspace / "marvis.sqlite").as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        runs = [dict(r) for r in conn.execute("SELECT id,step_id,status,tool_ref FROM plan_step_runs WHERE plan_id=? ORDER BY id", (plan_id,))]
        datasets = [r[0] for r in conn.execute("SELECT id FROM datasets WHERE task_id=? ORDER BY id", (journey.task_id,))]
        jobs = [dict(r) for r in conn.execute("SELECT id,kind,status,started_at,finished_at,error_name FROM jobs WHERE task_id=? ORDER BY created_at,id", (journey.task_id,))]
    return {"target": target, "runs": runs, "dataset_ids": datasets, "jobs": jobs}


def start_body(action, plan):
    return {"content": action.content, "ui_action": "start_plan", "expected_plan_id": plan["id"],
            **plan["confirmation_snapshot"], "acceptance_mode": "manual_review"}


def dispatch_monitoring_fault(journey, action, plan):
    if (action.fault.kind not in {"stale_monitoring_revision_once", "tool_timeout_once"}
            or plan["template_id"] != "monitoring_run" or plan["task_id"] != journey.task_id
            or plan["status"] != "validated" or getattr(journey, "runtime_faults", [])):
        raise ValueError("monitoring_fault_requires_original_monitoring_proposal")
    steps = [s for s in plan["steps"] if (s["tool_ref"].get("plugin"), s["tool_ref"].get("tool")) == ("modeling", "score_dataset")]
    if len(steps) != 1 or steps[0]["status"] != "pending" or steps[0].get("output_ref"):
        raise ValueError("monitoring_fault_requires_fresh_scoring_step")
    step = steps[0]
    root = Path(journey.workspace).resolve()
    original = start_body(action, plan)
    fault = {"schema": "marvis.native-monitoring-fault.v1", "action_index": journey.current_action_index,
        "action_sha256": digest(action.model_dump()), "kind": action.fault.kind,
        "task_id": journey.task_id, "plan_id": plan["id"], "step_id": step["id"],
        "tool": "modeling.score_dataset", "plan_before": plan, "original_request": original,
        "before": _state(journey, plan["id"], step["id"]), "submission": journey.monitoring_submission,
        "fault_executed": False}
    journey.runtime_faults = [fault]
    journey.record_human_action(action)
    route = f"/api/tasks/{journey.task_id}/agent/messages"
    if action.fault.kind == "stale_monitoring_revision_once":
        request = original | {"expected_plan_fingerprint": "0" * 64}
        fault["dispatched_request"] = request
        response = journey.request("POST", route, label="monitoring_stale_plan_confirmation", json=request)
        fault.update(http_ordinal=journey.count, http_status=response.status_code, response=response.json())
        if response.status_code != 409:
            raise ValueError("monitoring_stale_confirmation_did_not_reject")
    else:
        directory = arm_process_fault(root, kind="tool_timeout_once", task_id=journey.task_id,
            plan_id=plan["id"], step_id=step["id"], tool="modeling.score_dataset")
        fault["dispatched_request"] = original
        # This Agent endpoint is synchronous. Observe its actual worker while
        # that one request executes; no second API request races the first.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(journey.request, "POST", route, label="human_monitoring_plan_start", json=original)
            launch = attached_launch(directory, deadline=min(journey.deadline, time.monotonic() + 20))
            fault.update(ready=launch.ready(timeout=10), originals_relative_path=".runtime-eval-fault")
            terminal = _terminal(launch, journey.deadline)
            response = future.result(timeout=max(1, journey.deadline - time.monotonic()))
        fault.update(http_ordinal=journey.count, http_status=response.status_code, response=response.json())
        if terminal["terminal"] != "timeout" or terminal["observed_timeout_seconds"] != 2:
            raise ValueError("monitoring_native_worker_timeout_missing")
    journey.wait_idle()
    fault.update(after_fault=_state(journey, plan["id"], step["id"]), fault_executed=True)
    directory = root / ".runtime-eval-fault/monitoring"
    directory.mkdir(mode=0o700, parents=True)
    write_original(directory / "fault.json", fault)


def recover_monitoring_execution(journey, action):
    faults = getattr(journey, "runtime_faults", [])
    if (len(faults) != 1 or faults[0].get("schema") != "marvis.native-monitoring-fault.v1"
            or faults[0]["action_index"] != action.fault_source_action or not faults[0].get("fault_executed")
            or "recovery" in faults[0]):
        raise ValueError("monitoring_recovery_requires_exact_original_fault")
    fault = faults[0]
    journey.record_human_action(action)
    recovery = {"action_index": journey.current_action_index, "action_sha256": digest(action.model_dump()),
                "before": _state(journey, fault["plan_id"], fault["step_id"])}
    fault["recovery"] = recovery
    plan = journey.json_request("GET", f"/api/plans/{fault['plan_id']}", label="read_monitoring_recovery_plan")["plan"]
    if plan["id"] != fault["plan_id"] or plan["task_id"] != journey.task_id or plan["template_id"] != "monitoring_run":
        raise ValueError("monitoring_recovery_plan_changed")
    recovery["plan_before"] = plan
    if fault["kind"] == "stale_monitoring_revision_once":
        body = start_body(action, plan)
        recovery["request"] = body
        response = journey.request("POST", f"/api/tasks/{journey.task_id}/agent/messages",
            label="human_monitoring_plan_start", json=body)
    else:
        recovery["request"] = {}
        response = journey.request("POST", f"/api/plans/{fault['plan_id']}/steps/{fault['step_id']}/retry",
            label="monitoring_native_timeout_retry", json={})
    recovery.update(http_ordinal=journey.count, http_status=response.status_code, response=response.json())
    journey.wait_idle()
    recovery["after"] = _state(journey, fault["plan_id"], fault["step_id"])
    target = recovery["after"]["target"]["target"]
    recovery["completed"] = target["plan_status"] == "done" and target["status"] == "done" and bool(target["output_ref"])
    write_original(Path(journey.workspace).resolve() / ".runtime-eval-fault/monitoring/recovery.json", recovery)
