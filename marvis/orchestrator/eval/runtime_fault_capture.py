"""Fixed actual native target originals around future reference faults.

The caller supplies an already-read plan/step, never SQL, paths or callbacks.
Every snapshot is a committed read transaction. Live capability material is
excluded; native opaque IDs and immutable content commitments are retained.
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .runtime_contracts import digest

_TABLES = {
    "decision_records": ("id", "plan_id", "step_id", "principal_id", "decision", "manifest_hash", "policy_hash", "input_hash", "evidence_hash", "created_at"),
    "approval_records": ("id", "decision_id", "plan_id", "step_id", "principal_id", "status", "reservation_id", "issued_at", "expires_at", "consumed_at", "revoked_at"),
    "plan_step_runs": ("id", "plan_id", "step_id", "attempt", "tool_ref", "status", "invocation_id", "output_ref", "output_hash", "raw_output_hash", "manifest_hash", "tool_version"),
    "plan_step_output_versions": ("step_id", "version", "created_at"),
    "effect_executions": ("id", "approval_id", "invocation_id", "reservation_id", "status", "released_at"),
}


def _native_json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def capture_target_original(workspace: Path, task_id, plan_id, step_id, *, origin_ns):
    database = (Path(workspace) / "marvis.sqlite").resolve(strict=True)
    started = time.monotonic_ns() - origin_ns
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        step = conn.execute("SELECT s.*,p.task_id,p.status AS plan_status,p.replan_count FROM plan_steps s JOIN plans p ON p.id=s.plan_id WHERE s.id=? AND p.id=? AND p.task_id=?", (step_id, plan_id, task_id)).fetchone()
        if step is None:
            raise ValueError("fault_original_target_missing")
        target = {key: step[key] for key in ("id", "plan_id", "task_id", "plan_status", "replan_count", "tool_plugin", "tool_name", "status", "confirmed", "needs_confirmation", "output_ref")}
        target.update(input_sha256=_native_json_hash(json.loads(step["inputs_json"])), policy_sha256=_native_json_hash(json.loads(step["policy_json"])))
        tables = {}
        for table, fields in _TABLES.items():
            if table == "effect_executions":
                rows = conn.execute("SELECT e.* FROM effect_executions e JOIN approval_records a ON a.id=e.approval_id WHERE a.step_id=?", (step_id,)).fetchmany(101)
            else:
                rows = conn.execute(f"SELECT * FROM {table} WHERE step_id=?", (step_id,)).fetchmany(101)  # nosec B608 -- table iterates only source-owned _TABLES literal keys; step id is bound
            if len(rows) > 100:
                raise ValueError("fault_original_row_budget")
            values = []
            for row in rows:
                value = {key: row[key] for key in fields}
                if table == "plan_step_output_versions":
                    value.update(output_sha256=_native_json_hash(json.loads(row["output_json"])), evidence_sha256=_native_json_hash(json.loads(row["evidence_json"])))
                values.append(value)
            tables[table] = sorted(values, key=digest)
        conn.rollback()
    return {"schema": "marvis.runtime-target-originals.v1", "target": target, "tables": tables,
            "started_elapsed_ns": started, "finished_elapsed_ns": time.monotonic_ns() - origin_ns,
            "capture": "fixed_native_fields_from_one_committed_read_transaction", "authority": "not_established"}


def dispatch_fault(journey, action, workspace, plan, step, body, *, actor=None):
    """Return true iff one declared fault was actually dispatched and retained."""
    if action.fault is None:
        return False
    index = journey.current_action_index
    rows = getattr(journey, "runtime_faults", [])
    if any(row["action_index"] == index for row in rows):
        raise ValueError("fault_action_already_dispatched")
    before = capture_target_original(workspace, journey.task_id, plan["id"], step["id"], origin_ns=journey.origin_ns)
    fault = {"schema": "marvis.runtime-reference-fault.v1", "action_index": index, "action_sha256": digest(action.model_dump()),
             "kind": action.fault.kind, "task_id": journey.task_id, "plan_id": plan["id"], "step_id": step["id"], "tool": action.tool,
             "actor": actor, "original_request": body, "plan_before": plan, "before": before,
             "scope": "local_client_reference_fault; not_institution_network_fault_proof"}
    dispatch = dict(body)
    if action.fault.kind == "stale_step_cas_once":
        dispatch["expected_step_fingerprint"] = "0" * 64
    fault["dispatched_request"] = dispatch
    route = f"/api/plans/{plan['id']}/steps/{step['id']}/decisions"
    label = "reference_stale_cas_fault" if action.fault.kind == "stale_step_cas_once" else "reference_discard_response_fault"
    reply = journey.request("POST", route, label=label, json=dispatch)
    fault["http_ordinal"], fault["http_status"] = journey.count, reply.status_code
    expected = 409 if action.fault.kind == "stale_step_cas_once" else 202
    if reply.status_code != expected:
        raise ValueError("declared_fault_did_not_reach_expected_native_boundary")
    fault["after_response"] = capture_target_original(workspace, journey.task_id, plan["id"], step["id"], origin_ns=journey.origin_ns)
    if action.fault.kind == "discard_success_response_once":
        # This is the actual response retained privately as an original, while
        # the journey deliberately cannot use it as its received result.
        response = reply.json()
        fault["received_response"] = response
        fault["visible_response"] = "discarded_by_declared_client_reference_fault"
        decisions = getattr(journey, "human_decisions", [])
        decisions.append({"tool": action.tool, "actor": actor, "request": body, "response": response,
                          "http_ordinal": journey.count, "case_action_index": index, "plan_before": plan,
                          "action_sha256": digest(action.model_dump()), "fault_kind": action.fault.kind})
        journey.human_decisions = decisions
    rows.append(fault)
    journey.runtime_faults = rows
    return True


def recover_fault(journey, action, workspace):
    sources = [row for row in getattr(journey, "runtime_faults", ()) if row["action_index"] == action.fault_source_action]
    if len(sources) != 1 or sources[0]["kind"] != "discard_success_response_once" or sources[0]["tool"] != action.tool or action.fault_source_action >= journey.current_action_index:
        raise ValueError("recovery_requires_exact_prior_discarded_native_action")
    fault = sources[0]
    if "recovery" in fault:
        raise ValueError("fault_already_recovered")
    journey.record_human_action(action)
    before = capture_target_original(workspace, journey.task_id, fault["plan_id"], fault["step_id"], origin_ns=journey.origin_ns)
    ordinal_before = journey.count
    if fault["actor"] is not None:
        from .runtime_extensions import actor_session
        with actor_session(journey, "maker"):
            after = journey.wait_idle(changed_step=fault["step_id"])
    else:
        after = journey.wait_idle(changed_step=fault["step_id"])
    new_events = journey.events[ordinal_before:]
    if not new_events or any(row.get("method") != "GET" or row.get("stage") not in {"read_plans", "read_job"} for row in new_events):
        raise ValueError("recovery_must_use_fixed_readback_only")
    matches = [plan for plan in after if plan["id"] == fault["plan_id"] and plan["task_id"] == journey.task_id]
    if len(matches) != 1:
        raise ValueError("recovery_original_plan_not_read_back")
    steps = [step for step in matches[0]["steps"] if step["id"] == fault["step_id"]]
    if len(steps) != 1 or steps[0]["status"] != "done" or not steps[0]["output_ref"]:
        raise ValueError("recovery_requires_actual_native_completed_output")
    native_after = capture_target_original(workspace, journey.task_id, fault["plan_id"], fault["step_id"], origin_ns=journey.origin_ns)
    fault["recovery"] = {"action_index": journey.current_action_index, "action_sha256": digest(action.model_dump()),
                         "before_readback": before, "after_readback": native_after, "plan_after": matches[0],
                         "http_ordinals": list(range(ordinal_before + 1, journey.count + 1)), "transport": "existing_fixed_get_plan_readback_only"}
    for decision in journey.human_decisions:
        if decision["case_action_index"] == fault["action_index"]:
            decision["plan_after"] = matches[0]
