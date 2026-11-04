"""Independent raw-reader closure of the fixed native PMML timeout and retry."""
from datetime import datetime
import json

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _file, _json
from .runtime_contracts import digest

_ROOT = "workspace/.runtime-eval-fault/validation-timeout/"


def _read(ctx, name):
    return _json(ctx.archive, _ROOT + name + ".json")


def _http(events, ordinal, stage):
    found = [e for e in events if e["ordinal"] == ordinal]
    if len(found) != 1:
        raise ValueError("validation timeout HTTP original missing")
    event = found[0]
    equal((event["method"], event["stage"]), ("POST", stage))
    if not 0 <= event["started_elapsed_ns"] <= event["finished_elapsed_ns"]:
        raise ValueError("validation timeout HTTP interval differs")
    return event


def _jobs(ctx, snapshot):
    rows = [dict(r) for r in ctx.conn.execute(
        "SELECT id,task_id,kind,status,started_at,finished_at,error_name "
        "FROM jobs WHERE task_id=? ORDER BY created_at,id", (ctx.binding.task_id,))]
    for row in rows:
        if row["status"] in {"succeeded", "failed", "cancelled"}:
            if not row["started_at"] or not row["finished_at"]:
                raise ValueError("validation timeout completed job timing missing")
            start, end = (datetime.fromisoformat(row[k]) for k in ("started_at", "finished_at"))
            if start.tzinfo is None or end.tzinfo is None or start > end:
                raise ValueError("validation timeout completed job time is reversed")
    identities = {r["id"] for r in snapshot}
    if len(identities) != len(snapshot):
        raise ValueError("validation timeout duplicated job identity")
    equal([r for r in rows if r["id"] in identities], snapshot)
    return rows


def revalidate_validation_timeout_archive(archive, *, expected_manifest_sha256, frozen_binding):
    from .runtime_archive_safety import _private, _revalidate
    from .runtime_archive_validation_recovery import _confirmation_originals

    def work(ctx):
        equal(ctx.binding.recovery_action_indices, [2])
        actions = ctx.binding.case.actions
        equal([a.kind for a in actions], ["start_validation_agent", "recover_validation_execution", "confirm_current_validation_report"])
        equal((actions[0].fault.kind, actions[1].fault_source_action), ("tool_timeout_once", 1))
        arm = _read(ctx, "armed")
        expected_arm = {"schema": "marvis.validation-timeout-arm.v1", "kind": "tool_timeout_once",
            "task_id": ctx.binding.task_id, "action_index": 1, "action_sha256": digest(actions[0].model_dump()),
            "pmml_sha256": next(m.sha256 for m in ctx.binding.case.materials if m.role == "pmml")}
        equal(arm, expected_arm)
        faults = _private(ctx).get("validation_faults", [])
        if len(faults) != 1:
            raise _Unsupported("one native validation timeout original required")
        fault = faults[0]
        original = _read(ctx, "after-fault")
        equal({k: v for k, v in fault.items() if k not in {"recovery", "confirmation"}}, original)
        equal({k: fault[k] for k in arm}, arm)
        terminal, claimed = _read(ctx, "terminal"), _read(ctx, "claimed")
        equal(claimed["socket_timeouts"], [])
        equal({k: terminal[k] for k in claimed if k != "socket_timeouts"},
              {k: v for k, v in claimed.items() if k != "socket_timeouts"})
        equal(terminal["arm_sha256"], digest(arm))
        equal(fault["terminal_sha256"], digest(terminal))
        equal((terminal["schema"], terminal["task_id"], terminal["pmml_sha256"]),
              ("marvis.validation-pmml-timeout.v1", ctx.binding.task_id, arm["pmml_sha256"]))
        equal((terminal["outcome"], terminal["signal"], terminal["stopped_status"], terminal["resume_signal"]),
              ("native_exception", "SIGSTOP", "stopped", "SIGCONT"))
        equal(terminal["timeout_seconds"], 2)
        equal(terminal["jvm"], terminal["resumed_identity"])
        host, jvm = terminal["host"], terminal["jvm"]
        if (type(host["pid"]) is not int or type(jvm["pid"]) is not int or host["pid"] <= 0
                or jvm["pid"] <= 0 or host["pid"] == jvm["pid"] or jvm["ppid"] != host["pid"]
                or not 0 < host["create_time"] <= jvm["create_time"]):
            raise ValueError("validation timeout JVM ownership differs")
        start, end, resumed = (terminal[k] for k in ("started_elapsed_ns", "finished_elapsed_ns", "resumed_elapsed_ns"))
        if not (all(type(v) is int for v in (start, end, resumed)) and 0 <= start < end <= resumed and end - start >= 2_000_000_000):
            raise ValueError("validation timeout real deadline differs")
        sockets = terminal["socket_timeouts"]
        if not sockets:
            raise ValueError("validation timeout transport exception missing")
        for entry in sockets:
            if (entry["error_type"] not in {"TimeoutError", "timeout"} or entry["peer_host"] not in {"127.0.0.1", "::1"}
                    or entry["peer_port"] != terminal["gateway_port"] or entry["socket_timeout_seconds"] != 2
                    or not start + 2_000_000_000 <= entry["elapsed_ns"] <= end):
                raise ValueError("validation timeout socket original differs")
        equal(terminal["before"]["active_jobs"], [{"id": terminal["job_id"], "task_id": ctx.binding.task_id, "kind": "agent"}])
        running = [j for j in terminal["before"]["jobs"] if j["id"] == terminal["job_id"]]
        if len(running) != 1 or running[0]["status"] != "running":
            raise ValueError("validation timeout original active job missing")
        before, after = fault["before"], fault["after_fault"]
        _jobs(ctx, before["jobs"])
        final_jobs = _jobs(ctx, after["jobs"])
        equal(after["active_jobs"], [])
        added = [j for j in after["jobs"] if j["id"] not in {b["id"] for b in before["jobs"]}]
        if len(added) != 1 or added[0]["id"] != terminal["job_id"] or added[0]["kind"] != "agent" or added[0]["status"] != "failed":
            raise ValueError("validation timeout native failure missing or duplicated")
        failed = added[0]
        equal({k: v for k, v in running[0].items() if k not in {"status", "finished_at", "error_name"}},
              {k: v for k, v in failed.items() if k not in {"status", "finished_at", "error_name"}})
        if not failed["error_name"] or not failed["started_at"] or not failed["finished_at"]:
            raise ValueError("validation timeout native failure fields missing")
        task = ctx.conn.execute("SELECT * FROM tasks WHERE id=?", (ctx.binding.task_id,)).fetchone()
        equal((task["task_type"], task["run_mode"], task["validation_workflow_version"]), ("validation", "agent", 2))
        contract = ctx.conn.execute("SELECT status,material_hashes_json,confirmed_json FROM validation_input_contracts WHERE task_id=?", (ctx.binding.task_id,)).fetchone()
        equal(contract["status"], "ready")
        equal(json.loads(contract["material_hashes_json"])["pmml"], arm["pmml_sha256"])
        equal(json.loads(contract["confirmed_json"])["pmml_output_field"], terminal["output_field"])
        events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
        posted = _http(events, fault["http_ordinal"], "agent_initial_turn")
        equal(posted["status_code"], 202)
        equal((fault["response"]["task_id"], fault["response"]["status"]), (ctx.binding.task_id, "accepted"))
        if not posted["started_elapsed_ns"] <= start:
            raise ValueError("validation timeout preceded its native start")
        recovery = fault.get("recovery")
        if not recovery:
            raise _Unsupported("explicit native validation retry unavailable")
        equal(recovery, _read(ctx, "after-retry"))
        equal((recovery["action_index"], recovery["action_sha256"]), (2, digest(actions[1].model_dump())))
        equal(recovery["before"], after)
        retried = _http(events, recovery["http_ordinal"], "validation_timeout_retry")
        equal(retried["status_code"], recovery["http_status"])
        if not resumed <= retried["started_elapsed_ns"] or recovery["http_ordinal"] <= fault["http_ordinal"]:
            raise ValueError("validation retry preceded failure or resume")
        equal(recovery["after"]["active_jobs"], [])
        _jobs(ctx, recovery["after"]["jobs"])
        retry_jobs = [j for j in recovery["after"]["jobs"] if j["id"] not in {b["id"] for b in after["jobs"]}]
        if retried["status_code"] != 202:
            equal(retry_jobs, [])
            equal(final_jobs, after["jobs"])
            return {"fault_executed": True, "recovery_completed": False, "recovery_outcome": "native_retry_rejected"}
        equal((recovery["response"]["task_id"], recovery["response"]["status"]), (ctx.binding.task_id, "accepted"))
        if len(retry_jobs) != 1 or retry_jobs[0]["kind"] != "agent":
            raise ValueError("validation retry must create exactly one new native job")
        retry = retry_jobs[0]
        if datetime.fromisoformat(failed["finished_at"]) > datetime.fromisoformat(retry["started_at"]):
            raise ValueError("validation jobs overlap before retry")
        starts = [e for e in events if e["stage"] in {"validation_timeout_start", "validation_timeout_retry", "agent_initial_turn"}]
        equal([e["ordinal"] for e in starts], [posted["ordinal"], retried["ordinal"]])
        confirmation = fault.get("confirmation")
        if retry["status"] != "succeeded" or confirmation is None:
            equal(final_jobs, recovery["after"]["jobs"])
            return {"fault_executed": True, "recovery_completed": False, "recovery_outcome": "native_retry_incomplete"}
        equal(confirmation, _read(ctx, "confirmation"))
        equal((confirmation["action_index"], confirmation["action_sha256"]), (3, digest(actions[2].model_dump())))
        confirmed = _http(events, confirmation["http_ordinal"], "human_validation_report_confirmation")
        equal(confirmed["status_code"], 202)
        if confirmed["ordinal"] <= retried["ordinal"]:
            raise ValueError("validation confirmation preceded retry")
        audits = list(ctx.conn.execute("SELECT id FROM audit WHERE target_ref=? AND kind='report.agent_conclusions.confirm'", (ctx.binding.task_id,)))
        if len(audits) != 1:
            raise ValueError("validation retry confirmation duplicated")
        _confirmation_originals(ctx, confirmation["request"], confirmation["response"], audits[0]["id"])
        request = confirmation["request"]
        draft = ctx.conn.execute("SELECT created_at FROM agent_messages WHERE id=?", (request["draft_message_id"],)).fetchone()
        if not datetime.fromisoformat(retry["started_at"]) <= datetime.fromisoformat(draft["created_at"]) <= datetime.fromisoformat(retry["finished_at"]):
            raise ValueError("validation confirmed draft does not belong to retry")
        added_reports = [j for j in final_jobs if j["id"] not in {b["id"] for b in recovery["after"]["jobs"]}]
        if len(added_reports) != 1 or added_reports[0]["kind"] != "report" or added_reports[0]["status"] != "succeeded":
            raise ValueError("validation retry report job missing or duplicated")
        if datetime.fromisoformat(retry["finished_at"]) > datetime.fromisoformat(added_reports[0]["started_at"]):
            raise ValueError("validation report started before retry finished")
        equal(confirmation["final"]["jobs"], final_jobs)
        downloads = confirmation["downloads"]
        if len(downloads) != 2 or {d["kind"] for d in downloads} != {"word", "excel"}:
            raise ValueError("validation retry downloads unavailable")
        for download in downloads:
            filename = "validation_report.docx" if download["kind"] == "word" else "validation.xlsx"
            equal(download["sha256"], digest(_file(ctx.archive, f"workspace/tasks/{ctx.binding.task_id}/outputs/{filename}").read_bytes()))
            matches = [e for e in events if e["stage"] == "download_validation_" + download["kind"]]
            if len(matches) != 1 or matches[0]["method"] != "GET" or matches[0]["status_code"] != 200 or matches[0].get("sha256") != download["sha256"] or matches[0]["ordinal"] <= confirmed["ordinal"]:
                raise ValueError("validation retry report download differs")
        return {"fault_executed": True, "recovery_completed": True,
                "recovery_outcome": "native_retry_and_report_completed", "failed_job_id": failed["id"],
                "retry_job_id": retry["id"], "confirmation_request": request, "downloads": downloads,
                "business_math": "requires_separate_independent_validation_reader"}

    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "validation-timeout",
                       "original_validation_timeout_recovery", work)
