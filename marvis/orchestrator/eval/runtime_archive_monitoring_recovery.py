"""Original monitoring-plan CAS/scoring timeout and native recovery closure."""
from datetime import datetime
import json

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _json
from .runtime_contracts import digest
from .runtime_fault_reference import archived_target, original_target, native_completed_output, native_json_sha


def _event(events, ordinal, stage, statuses):
    found = [e for e in events if e["ordinal"] == ordinal]
    if len(found) != 1:
        raise ValueError("monitoring original HTTP missing")
    e = found[0]
    if e["method"] != "POST" or e["stage"] != stage or e["status_code"] not in statuses:
        raise ValueError("monitoring original HTTP method/stage/status differs")
    if not 0 <= e["started_elapsed_ns"] <= e["finished_elapsed_ns"]:
        raise ValueError("monitoring original HTTP timing differs")
    return e


def _plan_snapshot(plan, task_id):
    from .runtime_archive_safety import _PLAN_FIELDS, _STEP_FIELDS

    if plan["task_id"] != task_id or plan["template_id"] != "monitoring_run":
        raise ValueError("monitoring recovery borrowed a prerequisite plan")
    canonical = {key: plan[key] for key in _PLAN_FIELDS}
    canonical["steps"] = [{key: s[key] for key in _STEP_FIELDS} for s in plan["steps"]]
    expected = {"expected_plan_status": plan["status"], "expected_plan_revision": plan["replan_count"],
                "expected_plan_fingerprint": digest(canonical)}
    equal(plan["confirmation_snapshot"], expected)
    return expected


def _request(body, action, plan, task_id):
    expected = {"content": action.content, "ui_action": "start_plan", "expected_plan_id": plan["id"],
                **_plan_snapshot(plan, task_id), "acceptance_mode": "manual_review"}
    equal(body, expected)


def _native_plan_preimage(ctx, plan):
    """Compare static plan fields with native rows, beyond private self-hashes."""
    row = ctx.conn.execute("SELECT * FROM plans WHERE id=? AND task_id=?", (plan["id"], ctx.binding.task_id)).fetchone()
    if row is None:
        raise ValueError("monitoring plan preimage has no native owner")
    for key in ("id", "task_id", "goal", "source", "template_id", "autonomy_level", "novel_mode", "tier", "replan_count", "created_at"):
        equal(plan[key], row[key])
    equal(plan["success_criteria"], json.loads(row["success_criteria_json"]))
    steps = list(ctx.conn.execute("SELECT * FROM plan_steps WHERE plan_id=? ORDER BY idx", (plan["id"],)))
    equal([s["id"] for s in plan["steps"]], [s["id"] for s in steps])
    for snapshot, native in zip(plan["steps"], steps, strict=True):
        for key in ("id", "plan_id", "title", "sub_agent_scope", "phase"):
            equal(snapshot[key], native[key])
        equal(snapshot["index"], native["idx"])
        equal(snapshot["tool_ref"], {"plugin": native["tool_plugin"], "tool": native["tool_name"], "version": native["tool_version"] or ""})
        for key in ("needs_confirmation", "decision_point"):
            equal(snapshot[key], bool(native[key]))
        for key in ("inputs", "policy", "depends_on", "post_checks", "granted_tools"):
            equal(native_json_sha(snapshot[key]), native_json_sha(json.loads(native[key + "_json"])))


def _native_state(ctx, plan_id, step_id):
    target = archived_target(ctx, plan_id, step_id)
    runs = [dict(r) for r in ctx.conn.execute("SELECT id,step_id,status,tool_ref FROM plan_step_runs WHERE plan_id=? ORDER BY id", (plan_id,))]
    datasets = [r[0] for r in ctx.conn.execute("SELECT id FROM datasets WHERE task_id=? ORDER BY id", (ctx.binding.task_id,))]
    jobs = [dict(r) for r in ctx.conn.execute("SELECT id,kind,status,started_at,finished_at,error_name FROM jobs WHERE task_id=? ORDER BY created_at,id", (ctx.binding.task_id,))]
    for job in jobs:
        if job["status"] in {"failed", "succeeded", "cancelled"}:
            if job["started_at"] is None:
                # Native /retry records a rejected plan lease before the
                # conflict check; it has no execution start or tool effect.
                if (job["kind"], job["status"], job["error_name"]) != ("plan", "failed", "ConflictError") or not job["finished_at"]:
                    raise ValueError("monitoring native unstarted job is not a rejected retry")
                created = ctx.conn.execute("SELECT created_at FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
                start, end = datetime.fromisoformat(created), datetime.fromisoformat(job["finished_at"])
                if start.tzinfo is None or end.tzinfo is None or start > end:
                    raise ValueError("monitoring rejected retry job interval reversed")
                continue
            start, end = (datetime.fromisoformat(job[k]) for k in ("started_at", "finished_at"))
            if start.tzinfo is None or end.tzinfo is None or start > end:
                raise ValueError("monitoring native job interval reversed")
    return {"target": target, "runs": runs, "dataset_ids": datasets, "jobs": jobs}


def _snapshot_target(ctx, snapshot, fault):
    return original_target(snapshot["target"], task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])


def revalidate_monitoring_recovery_archive(archive, *, expected_manifest_sha256, frozen_binding):
    from .runtime_archive_safety import _before, _private, _revalidate
    from .runtime_archive_process_faults import _process_original

    def work(ctx):
        _before(ctx)
        private = _private(ctx)
        faults = private.get("runtime_faults", [])
        if len(faults) != 1 or len(ctx.binding.recovery_action_indices) != 1:
            raise _Unsupported("one original monitoring fault/recovery required")
        fault = faults[0]
        index = ctx.binding.recovery_action_indices[0]
        action = ctx.binding.case.actions[index - 1]
        source = action.fault_source_action
        original = ctx.binding.case.actions[source - 1]
        equal((original.kind, action.kind), ("confirm_model_monitoring_plan", "recover_monitoring_execution"))
        equal((fault["schema"], fault["kind"], fault["task_id"], fault["tool"]),
              ("marvis.native-monitoring-fault.v1", original.fault.kind, ctx.binding.task_id, "modeling.score_dataset"))
        equal((fault["action_index"], fault["action_sha256"]), (source, digest(original.model_dump())))
        prefix = "workspace/.runtime-eval-fault/monitoring/"
        equal({k: v for k, v in fault.items() if k != "recovery"}, _json(ctx.archive, prefix + "fault.json"))
        plan = fault["plan_before"]
        equal((plan["id"], plan["status"]), (fault["plan_id"], "validated"))
        _request(fault["original_request"], original, plan, ctx.binding.task_id)
        _native_plan_preimage(ctx, plan)
        native_plan = ctx.conn.execute("SELECT task_id,template_id,replan_count FROM plans WHERE id=?", (fault["plan_id"],)).fetchone()
        equal(tuple(native_plan), (ctx.binding.task_id, "monitoring_run", plan["replan_count"]))
        targets = [s for s in plan["steps"] if s["id"] == fault["step_id"]]
        if len(targets) != 1:
            raise ValueError("monitoring original score step missing")
        score = targets[0]
        equal((score["tool_ref"]["plugin"], score["tool_ref"]["tool"], score["status"]), ("modeling", "score_dataset", "pending"))
        submission = fault["submission"]
        equal((submission["plan_id"], submission["runtime_entry"]), (fault["plan_id"], "standard_model_monitoring_agent"))
        proposal = ctx.binding.case.actions[source - 2]
        equal(proposal.kind, "submit_model_monitoring_request")
        request = submission["request"]
        equal(request["target_col"], proposal.monitoring_request.target_col)
        material = next(m for m in ctx.binding.case.materials if m.path == proposal.monitoring_request.material_path)
        dataset = ctx.conn.execute("SELECT task_id,content_hash FROM datasets WHERE id=?", (request["dataset_id"],)).fetchone()
        if dataset is None or dataset["task_id"] != ctx.binding.task_id:
            raise ValueError("monitoring proposal dataset owner differs")
        equal(dataset["content_hash"], request["expected_content_hash"])
        # Uploaded-file commitment is pinned separately; native parquet content
        # may differ from CSV/XLSX bytes, so do not equate the two hash domains.
        equal(score["inputs"]["dataset_id"], request["dataset_id"])
        frozen = score["inputs"]["monitoring_binding"]
        equal(frozen["experiment_id"], request["experiment_id"])
        if material.role != "unknown":
            raise ValueError("monitoring proposal is not its separate declared input")
        events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
        before, after = fault["before"], fault["after_fault"]
        first = _snapshot_target(ctx, before, fault)
        failed = _snapshot_target(ctx, after, fault)
        if before["runs"] or any(first["tables"].values()):
            raise ValueError("monitoring fault was not before native scoring")
        if fault["kind"] == "stale_monitoring_revision_once":
            event = _event(events, fault["http_ordinal"], "monitoring_stale_plan_confirmation", {409})
            equal(fault["http_status"], 409)
            equal(fault["dispatched_request"], fault["original_request"] | {"expected_plan_fingerprint": "0" * 64})
            equal(first["target"], failed["target"])
            equal(first["tables"], failed["tables"])
            equal((before["runs"], before["dataset_ids"]), (after["runs"], after["dataset_ids"]))
            if not first["finished_elapsed_ns"] <= event["started_elapsed_ns"] <= event["finished_elapsed_ns"] <= failed["started_elapsed_ns"]:
                raise ValueError("monitoring original stale rejection interval differs")
        elif fault["kind"] == "tool_timeout_once":
            event = _event(events, fault["http_ordinal"], "human_monitoring_plan_start", {200, 202})
            equal(fault["http_status"], event["status_code"])
            equal(fault["dispatched_request"], fault["original_request"])
            actual = _process_original(ctx, fault, events)
            terminal = actual["terminal"]
            if not event["started_elapsed_ns"] <= terminal["started_elapsed_ns"] <= terminal["finished_elapsed_ns"] <= event["finished_elapsed_ns"]:
                raise ValueError("monitoring worker timeout is outside native confirmation request")
        else:
            raise _Unsupported("monitoring fault executor unavailable")
        recovery = fault.get("recovery")
        if not recovery:
            raise _Unsupported("explicit native monitoring recovery missing")
        equal(recovery, _json(ctx.archive, prefix + "recovery.json"))
        equal((recovery["action_index"], recovery["action_sha256"]), (index, digest(action.model_dump())))
        intermediate = _snapshot_target(ctx, recovery["before"], fault)
        equal({k: failed[k] for k in ("target", "tables")}, {k: intermediate[k] for k in ("target", "tables")})
        if failed["finished_elapsed_ns"] > intermediate["started_elapsed_ns"]:
            raise ValueError("monitoring retry observation preceded its fault")
        final = _snapshot_target(ctx, recovery["after"], fault)
        native = _native_state(ctx, fault["plan_id"], fault["step_id"])
        equal(native, recovery["after"] | {"target": {k: final[k] for k in ("target", "tables")}})
        for key in ("id", "plan_id", "task_id", "tool_plugin", "tool_name", "input_sha256", "policy_sha256"):
            equal(first["target"][key], final["target"][key])
        original_jobs = {j["id"]: j for j in before["jobs"]}
        old_jobs = {j["id"]: j for j in after["jobs"]}
        final_jobs = {j["id"]: j for j in native["jobs"]}
        if any(old_jobs.get(k) != v for k, v in original_jobs.items()) or any(final_jobs.get(k) != v for k, v in old_jobs.items()):
            raise ValueError("monitoring recovery rewrote prior job")
        added = [j for k, j in old_jobs.items() if k not in original_jobs]
        retries = [j for k, j in final_jobs.items() if k not in old_jobs]
        if len(added) != 1 or added[0]["kind"] != "driver" or len(retries) != 1:
            raise ValueError("monitoring confirmation/retry job count differs")
        if fault["kind"] == "stale_monitoring_revision_once":
            equal((added[0]["status"], added[0]["error_name"]), ("failed", "DriverError"))
            retried = _event(events, recovery["http_ordinal"], "human_monitoring_plan_start", {200, 202, 409})
            equal(recovery["plan_before"]["id"], fault["plan_id"])
            _request(recovery["request"], action, recovery["plan_before"], ctx.binding.task_id)
            _native_plan_preimage(ctx, recovery["plan_before"])
            equal(recovery["plan_before"]["confirmation_snapshot"], plan["confirmation_snapshot"])
        else:
            retried = _event(events, recovery["http_ordinal"], "monitoring_native_timeout_retry", {200, 202, 409})
            equal(recovery["request"], {})
        equal(recovery["http_status"], retried["status_code"])
        stages = {"monitoring_stale_plan_confirmation", "human_monitoring_plan_start", "monitoring_native_timeout_retry"}
        equal([e["ordinal"] for e in events if e["stage"] in stages], [event["ordinal"], retried["ordinal"]])
        if not event["ordinal"] < retried["ordinal"] or not intermediate["finished_elapsed_ns"] <= retried["started_elapsed_ns"] <= retried["finished_elapsed_ns"] <= final["started_elapsed_ns"]:
            raise ValueError("monitoring recovery request/time order differs")
        completed = final["target"]["plan_status"] == "done" and final["target"]["status"] == "done" and bool(final["target"]["output_ref"])
        equal(recovery["completed"], completed)
        if completed:
            equal((retries[0]["kind"], retries[0]["status"]), ("driver", "succeeded"))
            if datetime.fromisoformat(added[0]["finished_at"]) > datetime.fromisoformat(retries[0]["started_at"]):
                raise ValueError("monitoring retry job preceded its failure")
            native_completed_output(ctx, fault["plan_id"], fault["step_id"], private)
            other = ctx.conn.execute("SELECT id,status FROM plan_steps WHERE plan_id=? AND tool_plugin='modeling' AND tool_name='monitor_run'", (fault["plan_id"],)).fetchall()
            if len(other) != 1 or other[0]["status"] != "done":
                raise ValueError("monitoring final metrics step incomplete")
            native_completed_output(ctx, fault["plan_id"], other[0]["id"], private)
            if len(native["runs"]) != 2 or any(r["status"] != "succeeded" for r in native["runs"]):
                raise ValueError("monitoring recovery duplicated native score or metrics")
            if len(set(native["dataset_ids"]) - set(after["dataset_ids"])) != 1 or not set(after["dataset_ids"]) <= set(native["dataset_ids"]):
                raise ValueError("monitoring recovery created unexpected datasets")
        elif fault["kind"] == "tool_timeout_once":
            equal(recovery["http_status"], 409)
            equal((retries[0]["kind"], retries[0]["status"], retries[0]["error_name"], retries[0]["started_at"]),
                  ("plan", "failed", "ConflictError", None))
            retry_created = ctx.conn.execute("SELECT created_at FROM jobs WHERE id=?", (retries[0]["id"],)).fetchone()[0]
            if datetime.fromisoformat(added[0]["finished_at"]) > datetime.fromisoformat(retry_created):
                raise ValueError("monitoring rejected retry preceded original failure")
            equal(native["runs"], after["runs"])
            equal(native["dataset_ids"], after["dataset_ids"])
        return {"fault_executed": True, "recovery_completed": completed,
                "monitoring_plan_id": fault["plan_id"], "score_step_id": fault["step_id"],
                "business_math": "requires_separate_independent_monitoring_reader"}

    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "monitoring-recovery",
                       "original_monitoring_fault_and_recovery", work)
