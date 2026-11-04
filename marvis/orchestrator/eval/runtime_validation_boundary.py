"""Fixed stale-report CAS probe and independent original rejection checks.

Only the normal report-confirmation URL is used. No caller supplies a route,
callback, result, credential or replacement business value.
"""
import json

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _file, _json
from .runtime_contracts import digest

STAGE = "validation_stale_report_revision"
KIND = "stale_report_revision_once"


def reject_stale_report_confirmation(journey, action, body):
    from .runtime_runner import RuntimeJourneyError
    from .runtime_validation_fault import _capture

    if action.kind != "confirm_current_validation_report" or getattr(action.fault, "kind", None) != KIND:
        raise RuntimeJourneyError("validation_boundary_not_declared")
    if getattr(journey, "validation_faults", None):
        raise RuntimeJourneyError("validation_fault_already_dispatched")
    changed = {**body, "revision": body["revision"] - 1}
    fault = {"schema": "marvis.validation-report-boundary.v1", "kind": KIND,
             "action_index": journey.current_action_index, "action_sha256": digest(action.model_dump()),
             "task_id": journey.task_id, "original_request": body, "dispatched_request": changed,
             "before": _capture(journey), "scope": "local_fixed_stale_revision; not_external_actor_authority"}
    response = journey.request("POST", f"/api/tasks/{journey.task_id}/agent/report-draft/confirm",
                               label=STAGE, json=changed)
    fault.update(http_ordinal=journey.count, http_status=response.status_code, after_response=_capture(journey))
    journey.validation_faults = [fault]
    if response.status_code != 409:
        raise RuntimeJourneyError("validation_stale_report_was_not_rejected")


def _boundary_values(fault, events):
    if not isinstance(fault.get("before"), dict) or not isinstance(fault.get("after_response"), dict):
        raise _Unsupported("original report boundary snapshots unavailable")
    original, request = fault["original_request"], fault["dispatched_request"]
    if (set(original) != {"draft_message_id", "draft_edit_revision", "revision", "text_values"}
            or type(original["revision"]) is not int or original["revision"] < 0):
        raise ValueError("original report confirmation request differs")
    equal(request, {**original, "revision": original["revision"] - 1})
    matches = [e for e in events if e.get("ordinal") == fault["http_ordinal"]]
    if len(matches) != 1:
        raise ValueError("original boundary HTTP identity missing")
    event = matches[0]
    equal((event.get("method"), event.get("stage"), event.get("status_code"), fault["http_status"]),
          ("POST", STAGE, 409, 409))
    before, after = fault["before"], fault["after_response"]
    if not (0 <= before["started_elapsed_ns"] <= before["finished_elapsed_ns"] <= event["started_elapsed_ns"]
            <= event["finished_elapsed_ns"] <= after["started_elapsed_ns"] <= after["finished_elapsed_ns"]):
        raise ValueError("boundary original timeline differs")
    if any(e.get("method") != "GET" for e in events if e.get("ordinal", 0) > fault["http_ordinal"]):
        raise ValueError("boundary followed by another mutating request")
    equal(before["state"], after["state"])
    if before["state"]["confirmation_audits"]:
        raise ValueError("boundary requires a fresh unconfirmed original draft")
    equal(before["state"]["task"]["report_values_revision"], original["revision"])
    return {"http_status": 409, "confirmation_count": 0, "new_jobs": 0,
            "metrics_unchanged": True, "revision_unchanged": True}


def boundary_observation(record, private):
    """Runtime regression observation; formal acceptance additionally reads DB originals."""
    if record.get("runtime_entry") != "standard_validation_agent_v2":
        return None
    rows = private.get("validation_faults", [])
    if len(rows) != 1 or rows[0].get("kind") != KIND:
        return None
    try:
        return _boundary_values(rows[0], record.get("http_events", []))
    except (KeyError, TypeError, ValueError, _Unsupported):
        return None


def revalidate_validation_boundary_archive(archive, *, expected_manifest_sha256, frozen_binding):
    from .runtime_archive_safety import _before, _private, _revalidate

    def work(ctx):
        _before(ctx)
        indices = ctx.binding.boundary_action_indices
        if len(indices) != 1 or ctx.binding.case.task.task_type != "validation":
            raise _Unsupported("one original validation report boundary is required")
        index = indices[0]
        action = ctx.binding.case.actions[index - 1]
        if action.kind != "confirm_current_validation_report" or getattr(action.fault, "kind", None) != KIND:
            raise _Unsupported("fixed stale validation report confirmation unavailable")
        rows = _private(ctx).get("validation_faults", [])
        if len(rows) != 1 or rows[0].get("action_index") != index or rows[0].get("kind") != KIND:
            raise _Unsupported("original stale validation report fault unavailable")
        fault = rows[0]
        equal(fault["schema"], "marvis.validation-report-boundary.v1")
        equal(fault["task_id"], ctx.binding.task_id)
        equal(fault["action_sha256"], digest(action.model_dump()))
        events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
        values = _boundary_values(fault, events)
        task = ctx.conn.execute("SELECT * FROM tasks WHERE id=?", (ctx.binding.task_id,)).fetchone()
        if (task is None or task["task_type"] != "validation" or task["run_mode"] != "agent"
                or task["validation_workflow_version"] != 2
                or ctx.conn.execute("SELECT COUNT(*) FROM plans WHERE task_id=?", (ctx.binding.task_id,)).fetchone()[0]):
            raise ValueError("standard validation task identity differs")
        state = fault["after_response"]["state"]
        equal(state["task"], {key: task[key] for key in ("id", "task_type", "run_mode", "report_values_revision")})
        audits = [dict(r) for r in ctx.conn.execute(
            "SELECT id,kind FROM audit WHERE target_ref=? AND kind='report.agent_conclusions.confirm' ORDER BY id",
            (ctx.binding.task_id,))]
        jobs = [dict(r) for r in ctx.conn.execute("SELECT id,kind FROM jobs WHERE task_id=? ORDER BY id", (ctx.binding.task_id,))]
        equal(state["confirmation_audits"], audits)
        equal(state["jobs"], jobs)
        if any(row["kind"] == "report" for row in jobs):
            raise ValueError("rejected confirmation created a report job")
        hashes = {name: digest(_file(ctx.archive, f"workspace/tasks/{ctx.binding.task_id}/outputs/{name}").read_bytes())
                  for name in ("pmml_scoring_result.json", "pmml_scores.parquet", "validation_results.json")}
        equal(state["metric_hashes"], hashes)
        draft = ctx.conn.execute("SELECT * FROM agent_messages WHERE task_id=? AND role='assistant' "
            "AND stage IN ('word_conclusion_draft','word_conclusion_confirmed') ORDER BY created_at DESC,id DESC LIMIT 1",
            (ctx.binding.task_id,)).fetchone()
        original = fault["original_request"]
        if draft is None or draft["id"] != original["draft_message_id"] or draft["stage"] != "word_conclusion_draft":
            raise ValueError("original pending report draft differs")
        metadata = json.loads(draft["metadata_json"])
        if metadata.get("fallback") is not False or metadata.get("streaming") is True or metadata.get("confirmable") is False:
            raise ValueError("original draft was not confirmable")
        equal(metadata.get("report_revision"), original["revision"])
        equal(metadata.get("draft_edit_revision", 0), original["draft_edit_revision"])
        equal(metadata.get("draft_values"), original["text_values"])
        return {**values, "fault_action_index": index, "draft_message_id": draft["id"]}

    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "validation-boundary",
                       "original_validation_report_boundary", work)
