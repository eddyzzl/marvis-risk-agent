"""Explicit, case-declared fresh approval after a real rejected stale CAS."""
from .runtime_contracts import digest
from .runtime_fault_capture import capture_target_original


def recover_stale(journey, action, workspace, fault):
    if (fault["kind"] != "stale_step_cas_once" or fault["actor"] is not None
            or fault["tool"] != action.tool or action.fault_source_action >= journey.current_action_index
            or "recovery" in fault):
        raise ValueError("stale_recovery_requires_exact_original_local_gate")
    journey.record_human_action(action)
    begin = journey.count
    plans = journey.plans()
    matches = [(p, s) for p in plans for s in p["steps"]
               if p["id"] == fault["plan_id"] and p["task_id"] == journey.task_id
               and s["id"] == fault["step_id"] and s["status"] == "awaiting_confirm"]
    if len(matches) != 1:
        raise ValueError("stale_recovery_current_gate_unavailable")
    plan, step = matches[0]
    body = {"decision": "approve", "reason": action.content, **step["confirmation_snapshot"]}
    before = capture_target_original(workspace, journey.task_id, plan["id"], step["id"], origin_ns=journey.origin_ns)
    response = journey.json_request("POST", f"/api/plans/{plan['id']}/steps/{step['id']}/decisions",
                                    label="human_approval", json=body)
    capture = {"tool": action.tool, "actor": None, "request": body, "response": response,
               "http_ordinal": journey.count, "case_action_index": journey.current_action_index,
               "action_sha256": digest(action.model_dump()), "plan_before": plan}
    decisions = getattr(journey, "human_decisions", [])
    decisions.append(capture)
    journey.human_decisions = decisions
    after = journey.wait_idle(changed_step=step["id"])
    capture["plan_after"] = next(p for p in after if p["id"] == plan["id"])
    post = capture_target_original(workspace, journey.task_id, plan["id"], step["id"], origin_ns=journey.origin_ns)
    fault["recovery"] = {"action_index": journey.current_action_index, "action_sha256": digest(action.model_dump()),
        "before_approval": before, "after_readback": post, "approval_http_ordinal": capture["http_ordinal"],
        "http_ordinals": list(range(begin + 1, journey.count + 1)),
        "transport": "explicit_current_CAS_approval_then_fixed_GET_readback"}
