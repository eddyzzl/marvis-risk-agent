"""Independent clarification, extension-negative and replayed-approval readers.

Facts come only from the authenticated archive copy (native SQLite rows and the
original request/response capture). Zero *local native* effect is established;
effects in an external institution are never claimed.
"""
import json
from pathlib import Path

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _json

_REFUSAL = {400, 401, 403, 404, 409, 422}


def task_native_effects(conn, task_id):
    """Counts of every local native execution/effect row owned by the task."""
    counts = {"done_plans": 0, "plan_step_runs": 0, "plan_step_output_versions": 0, "decision_records": 0,
              "effect_executions": 0, "task_artifacts": 0}
    counts["done_plans"] = conn.execute("SELECT COUNT(*) FROM plans WHERE task_id=? AND status='done'", (task_id,)).fetchone()[0]
    counts["plan_step_runs"] = conn.execute(
        "SELECT COUNT(*) FROM plan_step_runs WHERE plan_id IN (SELECT id FROM plans WHERE task_id=?)", (task_id,)).fetchone()[0]
    counts["plan_step_output_versions"] = conn.execute(
        "SELECT COUNT(*) FROM plan_step_output_versions WHERE step_id IN "
        "(SELECT id FROM plan_steps WHERE plan_id IN (SELECT id FROM plans WHERE task_id=?))", (task_id,)).fetchone()[0]
    counts["decision_records"] = conn.execute("SELECT COUNT(*) FROM decision_records WHERE task_id=?", (task_id,)).fetchone()[0]
    counts["effect_executions"] = conn.execute(
        "SELECT COUNT(*) FROM effect_executions WHERE approval_id IN "
        "(SELECT id FROM approval_records WHERE decision_id IN (SELECT id FROM decision_records WHERE task_id=?))",
        (task_id,)).fetchone()[0]
    counts["task_artifacts"] = conn.execute("SELECT COUNT(*) FROM task_artifacts WHERE task_id=?", (task_id,)).fetchone()[0]
    return counts


def latest_assistant_metadata(conn, task_id):
    row = conn.execute("SELECT metadata_json FROM agent_messages WHERE task_id=? AND role='assistant' "
                       "ORDER BY created_at DESC, id DESC LIMIT 1", (task_id,)).fetchone()
    if row is None:
        raise ValueError("no native assistant reply")
    value = json.loads(row["metadata_json"])
    if not isinstance(value, dict):
        raise ValueError("native assistant metadata is not an object")
    return value


def revalidate_clarification_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding):
    """Expected-clarification boundary: zero native execution and the actual native ask."""
    from .runtime_archive_safety import _revalidate

    def work(ctx):
        if any(a.kind in {"approve_step", "reject_step", "retry_step", "replay_approval"} for a in ctx.binding.case.actions):
            raise _Unsupported("clarification boundary cannot contain governed step actions")
        effects = task_native_effects(ctx.conn, ctx.binding.task_id)
        if any(effects.values()):
            raise ValueError("clarification boundary produced native execution or effects")
        return {"native_effects": effects, "latest_assistant_metadata": latest_assistant_metadata(ctx.conn, ctx.binding.task_id),
                "scope": "local_native_zero_execution_and_native_reply; external_effect_not_claimed"}

    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "clarification", "original_clarification_zero_effect", work)


def human_rejection(ctx, index):
    """A declared non-fault ``reject_step``: one reject record, zero target execution."""
    from .runtime_archive_safety import _private

    action = ctx.binding.case.actions[index - 1]
    if action.kind != "reject_step" or action.fault is not None:
        raise _Unsupported("human rejection boundary requires a declared plain reject action")
    captures = [row for row in _private(ctx).get("human_decisions", ()) if row.get("case_action_index") == index]
    if len(captures) != 1 or captures[0].get("request", {}).get("decision") != "reject":
        raise _Unsupported("original rejection request/response capture unavailable")
    capture = captures[0]
    events = _json(ctx.archive, "bindings.json")["execution_before_scoring"].get("http_events", [])
    matches = [e for e in events if e.get("ordinal") == capture.get("http_ordinal")]
    if len(matches) != 1 or matches[0].get("stage") != "human_rejection" or matches[0].get("method") != "POST":
        raise _Unsupported("original rejection HTTP fact unavailable")
    step_ids = [s["id"] for s in capture.get("plan_before", {}).get("steps", ())
                if s.get("tool_ref", {}).get("plugin", "") + "." + s.get("tool_ref", {}).get("tool", "") == action.tool]
    if len(step_ids) != 1:
        raise ValueError("rejected native target is not unique")
    step = step_ids[0]
    decisions = [dict(r) for r in ctx.conn.execute("SELECT decision FROM decision_records WHERE task_id=? AND step_id=?",
                                                    (ctx.binding.task_id, step))]
    equal([d["decision"] for d in decisions], ["reject"])
    runs = ctx.conn.execute("SELECT COUNT(*) FROM plan_step_runs WHERE step_id=?", (step,)).fetchone()[0]
    versions = ctx.conn.execute("SELECT COUNT(*) FROM plan_step_output_versions WHERE step_id=?", (step,)).fetchone()[0]
    row = ctx.conn.execute("SELECT output_ref FROM plan_steps WHERE id=?", (step,)).fetchone()
    if runs or versions or row is None or row["output_ref"]:
        raise ValueError("rejected native target was executed")
    return {"action_index": index, "http_status": matches[0]["status_code"], "step_id": step,
            "native_target_counter_delta": {"plan_step_runs": 0, "plan_step_output_versions": 0},
            "scope": "declared_human_rejection_and_zero_native_target_execution; not_external_institution_effect"}


def revalidate_extension_negative_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding):
    """Extension safety negatives: actual refusal capture and no committed local effect."""
    from .runtime_archive_safety import _action_output, _http, _revalidate

    def work(ctx):
        if not ctx.binding.boundary_action_indices:
            raise _Unsupported("original negative extension actions unavailable")
        refusals = []
        for index in ctx.binding.boundary_action_indices:
            output = _action_output(ctx, index)
            event = _http(ctx, output, statuses=_REFUSAL)
            equal(output["response"], {"status": "http_rejected", "http_status": event["status_code"]})
            refusals.append({"action_index": index, "http_status": event["status_code"]})
        committed = ctx.conn.execute(
            "SELECT COUNT(*) FROM effect_executions WHERE status IN ('dispatched','committed','uncertain') AND approval_id IN "
            "(SELECT id FROM approval_records WHERE decision_id IN (SELECT id FROM decision_records WHERE task_id=?))",
            (ctx.binding.task_id,)).fetchone()[0]
        if committed:
            raise ValueError("negative extension case dispatched or committed a native effect")
        return {"refusals": refusals, "native_dispatched_or_committed_effects": 0,
                "scope": "actual_refusal_capture_and_zero_local_dispatched_effect; external_effect_not_claimed"}

    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "extension-negative", "original_negative_zero_effect", work)


def replayed_approval(ctx, item):
    """Repeated approval of one step: refused or idempotent, exactly one execution."""
    actions = ctx.binding.case.actions
    original, replays = actions[item.approve_index - 1], [actions[i - 1] for i in item.replay_indices]
    if original.kind != "approve_step" or any(a.kind != "replay_approval" for a in replays):
        raise ValueError("replay binding does not reference an approval and its replays")
    events = _json(ctx.archive, "bindings.json")["execution_before_scoring"].get("http_events", [])
    probes = [e for e in events if e.get("stage") == "stale_approval_probe"]
    if len(probes) != len(item.replay_indices) or any(e.get("status_code") not in {200, 202, 409} for e in probes):
        raise ValueError("replayed approval HTTP outcome differs")
    step = ctx.conn.execute("SELECT plan_id FROM plan_steps WHERE id=?", (item.step_id,)).fetchone()
    if step is None or step["plan_id"] != item.plan_id:
        raise ValueError("replayed native target differs")
    runs = [r["status"] for r in ctx.conn.execute("SELECT status FROM plan_step_runs WHERE step_id=?", (item.step_id,))]
    versions = ctx.conn.execute("SELECT COUNT(*) FROM plan_step_output_versions WHERE step_id=?", (item.step_id,)).fetchone()[0]
    decisions = [r["decision"] for r in ctx.conn.execute("SELECT decision FROM decision_records WHERE task_id=? AND step_id=?",
                                                          (ctx.binding.task_id, item.step_id))]
    if runs.count("succeeded") != 1 or versions != 1 or decisions != ["approve"]:
        raise ValueError("repeated request produced a duplicate native execution or decision")
    return {"step_id": item.step_id, "replays": len(probes), "replay_http_statuses": sorted({e["status_code"] for e in probes}),
            "succeeded_runs": 1, "output_versions": 1, "approve_decisions": 1,
            "scope": "local_native_single_execution_under_replayed_approval; external_effect_not_claimed"}
