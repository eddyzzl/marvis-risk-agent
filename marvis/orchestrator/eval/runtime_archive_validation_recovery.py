"""Read-only validation recovery adjudication from actual native originals."""
import json

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _file, _json
from .runtime_contracts import digest


def revalidate_validation_recovery_archive(archive, *, expected_manifest_sha256, frozen_binding):
    from .runtime_archive_safety import _private, _revalidate

    def work(ctx):
        indices = ctx.binding.recovery_action_indices
        if len(indices) != 1:
            raise _Unsupported("one original validation report recovery is required")
        index = indices[0]
        action = ctx.binding.case.actions[index - 1]
        source = action.fault_source_action
        if action.kind != "recover_validation_report" or source is None:
            raise _Unsupported("native validation recovery action unavailable")
        rows = _private(ctx).get("validation_faults", [])
        if len(rows) != 1 or rows[0].get("action_index") != source:
            raise _Unsupported("original validation report fault unavailable")
        fault = rows[0]
        original = ctx.binding.case.actions[source - 1]
        stale = original.fault.kind == "stale_report_revision_once"
        if stale:
            from .runtime_validation_stale_reference import confirmation_original
            fault = confirmation_original(ctx, fault, original, action, index)
        else:
            equal(fault["schema"], "marvis.validation-report-fault.v1")
            equal(fault["task_id"], ctx.binding.task_id)
            equal(fault["kind"], "discard_success_response_once")
            equal(fault["action_sha256"], digest(original.model_dump()))
            equal(fault["visible_response"], "discarded_by_declared_client_reference_fault")
        recovery = fault.get("recovery")
        if not isinstance(recovery, dict):
            raise _Unsupported("original validation readback unavailable")
        equal((recovery["action_index"], recovery["action_sha256"]), (index, digest(action.model_dump())))
        events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
        posted = [e for e in events if e.get("ordinal") == fault["http_ordinal"]]
        if len(posted) != 1:
            raise ValueError("validation original confirmation HTTP missing")
        event = posted[0]
        equal((event["method"], event["stage"], event["status_code"], fault["http_status"]),
              ("POST", "human_validation_report_confirmation" if stale else "validation_discard_report_response", 202, 202))
        before, after, final = (fault["before"], fault["after_response"], recovery["after_readback"])
        if not (0 <= before["started_elapsed_ns"] <= before["finished_elapsed_ns"] <= event["started_elapsed_ns"]
                <= event["finished_elapsed_ns"] <= after["started_elapsed_ns"] <= after["finished_elapsed_ns"]
                <= final["started_elapsed_ns"] <= final["finished_elapsed_ns"]):
            raise ValueError("validation fault originals are out of order")
        ordinals = recovery["http_ordinals"]
        if not ordinals or ordinals != sorted(set(ordinals)) or ordinals[0] <= fault["http_ordinal"]:
            raise ValueError("validation recovery HTTP identity missing")
        readback = [e for e in events if e.get("ordinal") in ordinals]
        if len(readback) != len(ordinals) or any(e["method"] != "GET" or e["status_code"] != 200 for e in readback):
            raise ValueError("validation recovery must only read the existing execution")
        previous = after["finished_elapsed_ns"]
        for read in readback:
            if not (type(read["started_elapsed_ns"]) is int and type(read["finished_elapsed_ns"]) is int
                    and previous <= read["started_elapsed_ns"] <= read["finished_elapsed_ns"] <= final["started_elapsed_ns"]):
                raise ValueError("validation readback is outside the original recovery interval")
            previous = read["finished_elapsed_ns"]
        if any(e["method"] != "GET" for e in events if e.get("ordinal", 0) > fault["http_ordinal"]):
            raise ValueError("validation recovery repeated a mutating request")
        states = [v["state"] for v in (before, after, final)]
        task = dict(ctx.conn.execute("SELECT id,task_type,run_mode,report_values_revision FROM tasks WHERE id=?",
                                     (ctx.binding.task_id,)).fetchone())
        audits = [dict(r) for r in ctx.conn.execute(
            "SELECT id,kind FROM audit WHERE target_ref=? AND kind='report.agent_conclusions.confirm' ORDER BY id",
            (ctx.binding.task_id,))]
        jobs = [dict(r) for r in ctx.conn.execute("SELECT id,kind FROM jobs WHERE task_id=? ORDER BY id", (ctx.binding.task_id,))]
        equal(states[2]["task"], task)
        request = fault["request"]
        if (set(request) != {"draft_message_id", "draft_edit_revision", "revision", "text_values"}
                or type(request["revision"]) is not int or request["revision"] < 0
                or type(request["draft_edit_revision"]) is not int or request["draft_edit_revision"] < 0):
            raise ValueError("validation original confirmation request differs")
        expected_task = {"id": ctx.binding.task_id, "task_type": "validation", "run_mode": "agent",
                         "report_values_revision": request["revision"]}
        equal(states[0]["task"], expected_task)
        equal(states[1]["task"], {**expected_task, "report_values_revision": request["revision"] + 1})
        equal(states[1]["task"], task)
        equal(states[2]["confirmation_audits"], audits)
        equal(states[2]["jobs"], jobs)
        if states[0]["confirmation_audits"] or len(audits) != 1:
            raise ValueError("validation report confirmation is not unique")
        equal(states[1]["confirmation_audits"], audits)
        equal(states[1]["jobs"], jobs)
        old_jobs = {r["id"]: r for r in states[0]["jobs"]}
        new_jobs = {r["id"]: r for r in jobs}
        if len(new_jobs) != len(old_jobs) + 1 or any(new_jobs.get(k) != v for k, v in old_jobs.items()):
            raise ValueError("validation recovery duplicated or replaced work")
        added = next(row for identity, row in new_jobs.items() if identity not in old_jobs)
        equal(added["kind"], "report")
        completed = ctx.conn.execute("SELECT status FROM jobs WHERE id=? AND task_id=?",
                                     (added["id"], ctx.binding.task_id)).fetchone()
        equal(completed["status"], "succeeded")
        _confirmation_originals(ctx, request, fault["received_response"], audits[0]["id"])
        hashes = {}
        for name in ("pmml_scoring_result.json", "pmml_scores.parquet", "validation_results.json"):
            hashes[name] = digest(_file(ctx.archive, f"workspace/tasks/{ctx.binding.task_id}/outputs/{name}").read_bytes())
        for state in states:
            equal(state["metric_hashes"], hashes)
        downloads = recovery["downloads"]
        if len(downloads) != 2 or {r["kind"] for r in downloads} != {"word", "excel"}:
            raise ValueError("validation recovery report downloads missing")
        for download in downloads:
            matches = [e for e in readback if e["stage"] == "download_validation_" + download["kind"]]
            if len(matches) != 1 or matches[0].get("sha256") != download["sha256"]:
                raise ValueError("validation recovery downloaded bytes differ")
            filename = "validation_report.docx" if download["kind"] == "word" else "validation.xlsx"
            equal(download["sha256"], digest(_file(ctx.archive, f"workspace/tasks/{ctx.binding.task_id}/outputs/{filename}").read_bytes()))
        return {"recoveries": [{"fault_action_index": source, "recovery_action_index": index}],
                "confirmation_count": len(audits), "new_jobs": 1, "metrics_unchanged": True,
                "confirmation_request": fault["request"], "downloads": downloads}

    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "validation-recovery",
                       "original_validation_report_recovery", work)


def _confirmation_originals(ctx, request, response, audit_id):
    """Bind private transport captures to native draft, audit and report rows."""
    task_id = ctx.binding.task_id
    draft = ctx.conn.execute(
        "SELECT metadata_json FROM agent_messages WHERE id=? AND task_id=? AND role='assistant' AND stage='word_conclusion_draft'",
        (request["draft_message_id"], task_id),
    ).fetchone()
    if draft is None:
        raise ValueError("validation original confirmed draft missing")
    metadata = json.loads(draft["metadata_json"])
    if metadata.get("fallback") is not False or metadata.get("confirmable") is False or metadata.get("streaming") is True:
        raise ValueError("validation original draft was not confirmable")
    equal(metadata["report_revision"], request["revision"])
    equal(metadata.get("draft_edit_revision", 0), request["draft_edit_revision"])
    equal(metadata["draft_values"], request["text_values"])
    task = ctx.conn.execute("SELECT report_values_json FROM tasks WHERE id=?", (task_id,)).fetchone()
    values = json.loads(task["report_values_json"])
    for key, value in request["text_values"].items():
        equal(values[key], value)
    audit = ctx.conn.execute("SELECT outcome,detail_json FROM audit WHERE id=?", (audit_id,)).fetchone()
    equal(audit["outcome"], "succeeded")
    detail = json.loads(audit["detail_json"])
    equal(detail["expected_revision"], request["revision"])
    equal(detail["keys"], sorted(request["text_values"]))
    equal((response["task_id"], response["status"], response["revision"]),
          (task_id, "accepted", request["revision"] + 1))
    confirmations = [message for message in response["messages"] if message.get("stage") == "word_conclusion_confirmed"]
    if len(confirmations) != 1:
        raise ValueError("validation original response confirmation missing")
    confirmation = confirmations[0]
    equal(confirmation["task_id"], task_id)
    native = ctx.conn.execute(
        "SELECT metadata_json FROM agent_messages WHERE id=? AND task_id=? AND role='assistant' AND stage='word_conclusion_confirmed'",
        (confirmation["id"], task_id),
    ).fetchone()
    if native is None:
        raise ValueError("validation original response is not bound to its native confirmation")
    native_metadata = json.loads(native["metadata_json"])
    equal(confirmation["metadata"], native_metadata)
    equal(native_metadata["revision"], request["revision"] + 1)
    equal(native_metadata["confirmed_keys"], sorted(request["text_values"]))
