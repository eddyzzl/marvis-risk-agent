"""Independent stale refusal, fresh explicit authorization and native output."""
from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _json
from .runtime_contracts import digest
from .runtime_fault_reference import archived_target, native_completed_output, original_target


def revalidate_stale_recovery_archive(archive, *, expected_manifest_sha256, frozen_binding):
    from .runtime_archive_safety import _before, _fault_original, _governed, _private, _revalidate

    def work(ctx):
        _before(ctx)
        if len(ctx.binding.recovery_action_indices) != 1:
            raise _Unsupported("one exact stale recovery is required")
        index = ctx.binding.recovery_action_indices[0]
        action = ctx.binding.case.actions[index - 1]
        source = action.fault_source_action
        fault, before, after, event = _fault_original(ctx, source, "stale_step_cas_once")
        equal(fault["dispatched_request"], {**fault["original_request"], "expected_step_fingerprint": "0" * 64})
        equal({k: before[k] for k in ("target", "tables")}, {k: after[k] for k in ("target", "tables")})
        if any(before["tables"].values()):
            raise ValueError("stale recovery requires an unexecuted original target")
        recovery = fault.get("recovery")
        if not isinstance(recovery, dict):
            raise _Unsupported("explicit current approval original unavailable")
        equal((recovery["action_index"], recovery["action_sha256"], recovery["transport"]),
              (index, digest(action.model_dump()), "explicit_current_CAS_approval_then_fixed_GET_readback"))
        pre, post = [original_target(recovery.get(key), task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])
                     for key in ("before_approval", "after_readback")]
        equal({k: pre[k] for k in ("target", "tables")}, {k: after[k] for k in ("target", "tables")})
        events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
        ordinals = recovery["http_ordinals"]
        if not ordinals or ordinals != list(range(fault["http_ordinal"] + 1, ordinals[-1] + 1)):
            raise ValueError("stale recovery original request sequence differs")
        requests = [e for e in events if e.get("ordinal") in ordinals]
        approval = [e for e in requests if e.get("ordinal") == recovery["approval_http_ordinal"]]
        if len(requests) != len(ordinals) or len(approval) != 1:
            raise ValueError("stale recovery approval HTTP original absent")
        approved = approval[0]
        equal((approved["method"], approved["stage"], approved["status_code"]), ("POST", "human_approval", 202))
        if any(e["method"] != "GET" or e["stage"] not in {"read_plans", "read_job"} or e["status_code"] != 200
               for e in requests if e is not approved):
            raise ValueError("stale recovery contains another mutation")
        if any(e["method"] != "GET" and e["ordinal"] != approved["ordinal"] for e in events if e["ordinal"] > event["ordinal"]):
            raise ValueError("stale recovery repeated a mutation")
        if not (after["finished_elapsed_ns"] <= requests[0]["started_elapsed_ns"]
                <= requests[0]["finished_elapsed_ns"] <= pre["started_elapsed_ns"]
                <= pre["finished_elapsed_ns"] <= approved["started_elapsed_ns"]
                <= approved["finished_elapsed_ns"] <= post["started_elapsed_ns"]):
            raise ValueError("stale recovery approval timeline differs")
        previous = after["finished_elapsed_ns"]
        for request in requests:
            if not previous <= request["started_elapsed_ns"] <= request["finished_elapsed_ns"] <= post["started_elapsed_ns"]:
                raise ValueError("stale recovery request timeline differs")
            previous = request["finished_elapsed_ns"]
        equal(archived_target(ctx, fault["plan_id"], fault["step_id"]), {k: post[k] for k in ("target", "tables")})
        for key in ("id", "plan_id", "task_id", "tool_plugin", "tool_name", "input_sha256", "policy_sha256"):
            equal(before["target"][key], post["target"][key])
        if any(len(post["tables"][table]) != 1 for table in ("decision_records", "plan_step_runs", "plan_step_output_versions")):
            raise ValueError("stale recovery must perform exactly one native operation")
        declared = [item for item in ctx.binding.governed_actions if item.action_index == index]
        if len(declared) != 1 or (declared[0].plan_id, declared[0].step_id) != (fault["plan_id"], fault["step_id"]):
            raise _Unsupported("fresh exact decision must be independently frozen")
        captures = [c for c in _private(ctx).get("human_decisions", []) if c.get("case_action_index") == index]
        if len(captures) != 1 or captures[0]["http_ordinal"] != approved["ordinal"]:
            raise ValueError("stale recovery approval capture differs")
        governed = _governed(ctx, declared[0])
        output = native_completed_output(ctx, fault["plan_id"], fault["step_id"], _private(ctx))
        return {"recovery_completed": True, "recoveries": [{"fault_action_index": source,
            "action_index": index, "governed_action_index": index, "governed_action": governed, **output}],
            "stale_boundary_effects": 0, "fresh_approvals": 1, "native_successful_invocations": 1}
    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "stale-recovery", "original_stale_recovery", work)
