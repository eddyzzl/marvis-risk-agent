"""Read only original process/HTTP/native records; fault execution is not recovery."""
import hashlib
import json
from pathlib import PurePath

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _file, _json
from .runtime_contracts import digest
from .runtime_fault_reference import archived_target, original_target, native_completed_output


def _sha(value):
    raw = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _batch_commit(ctx, job, launch, output, commit):
    equal((job["module"], job["entrypoint"]), ("marvis.packs.modeling.tools", "tool_train_models"))
    identities, summaries, recipes = output.get("experiment_ids"), output.get("experiments"), job["inputs"].get("recipes")
    if (not isinstance(identities, list) or not 1 <= len(identities) <= 20
            or any(not isinstance(value, str) or not value for value in identities)
            or len(set(identities)) != len(identities) or set(identities).intersection(launch["before_ids"])
            or not isinstance(summaries, list) or len(summaries) != len(identities)
            or [item.get("experiment_id") for item in summaries] != identities
            or not isinstance(recipes, list) or len(set(recipes)) != len(recipes)
            or sorted(item.get("recipe") for item in summaries) != sorted(recipes)
            or output.get("failed") != [] or output.get("best_experiment_id") not in identities):
        raise ValueError("process_fault_model_batch_output_incomplete")
    fresh = {row["id"] for row in ctx.conn.execute("SELECT id FROM experiments WHERE task_id=?", (ctx.binding.task_id,))} - set(launch["before_ids"])
    equal(sorted(fresh), sorted(identities))
    rows, artifacts, expected_files, audits = [], [], [], []
    base = PurePath("tasks") / ctx.binding.task_id / "modeling_artifacts"
    for summary in summaries:
        row = ctx.conn.execute("SELECT * FROM experiments WHERE id=? AND task_id=?",
                               (summary["experiment_id"], ctx.binding.task_id)).fetchone()
        if row is None or row["status"] != "trained" or row["recipe_id"] != summary["recipe"]:
            raise ValueError("process_fault_model_batch_experiment_incomplete")
        artifact = ctx.conn.execute("SELECT * FROM model_artifacts WHERE id=? AND experiment_id=?",
                                    (row["artifact_id"], row["id"])).fetchone()
        if artifact is None or not artifact["model_path"]:
            raise ValueError("process_fault_model_batch_artifact_missing")
        paths = [str(PurePath(artifact[key]).relative_to(job["workspace"])
                     if PurePath(artifact[key]).is_absolute() else base / artifact[key])
                 for key in ("model_path", "pmml_path") if artifact[key]]
        for name in paths:
            if ".." in PurePath(name).parts:
                raise ValueError("process_fault_model_batch_path_escape")
            PurePath(name).relative_to(base)
        expected_detail = {"artifact_id": artifact["id"], "algorithm": artifact["algorithm"],
                           "feature_count": len(json.loads(artifact["feature_list_json"]))}
        attached = [dict(item) for item in ctx.conn.execute(
            "SELECT * FROM audit WHERE kind='modeling.experiment.trained' AND target_ref=? AND outcome='succeeded'",
            (row["id"],)) if json.loads(item["detail_json"]) == expected_detail]
        if len(attached) != 1:
            raise ValueError("process_fault_model_batch_training_audit_missing")
        rows.append(dict(row))
        artifacts.append(dict(artifact))
        expected_files.extend(paths)
        audits.extend(attached)
    if len(set(expected_files)) != len(expected_files):
        raise ValueError("process_fault_model_batch_artifact_alias")
    equal(commit["rows"], rows)
    equal(commit["artifacts"], artifacts)
    equal(commit["audits"], audits)
    return expected_files


def _commit(ctx, job, launch, ready, signal, read):
    output, commit = read("worker-output.json"), read("commit.json")
    equal(_sha(output), ready["output_sha256"])
    equal(_sha(commit), ready["commit_sha256"])
    equal(ready["commit_sha256"], signal["commit_sha256"])
    if commit["table"] == "datasets":
        field = "result_dataset_id" if job["entrypoint"] == "tool_execute_join" else "dataset_id"
        row = ctx.conn.execute("SELECT * FROM datasets WHERE id=? AND task_id=?",
                               (output.get(field), ctx.binding.task_id)).fetchone()
        if row is None or row["id"] in launch["before_ids"]:
            raise ValueError("process_fault_fresh_dataset_missing")
        equal(dict(row), commit["row"])
        path = PurePath(row["source_path"])
        if path.is_absolute():
            path = path.relative_to(job["datasets_root"])
        expected_files = ["datasets/" + str(path)]
        equal(commit["files"][0]["sha256"], row["content_hash"])
    elif commit["table"] == "experiments":
        row = ctx.conn.execute("SELECT * FROM experiments WHERE id=? AND task_id=?",
                               (output.get("experiment_id"), ctx.binding.task_id)).fetchone()
        if row is None or row["id"] in launch["before_ids"] or row["status"] != "trained":
            raise ValueError("process_fault_fresh_model_missing")
        equal(dict(row), commit["row"])
        equal(output["artifact_id"], row["artifact_id"])
        artifact = ctx.conn.execute("SELECT * FROM model_artifacts WHERE id=? AND experiment_id=?",
                                    (row["artifact_id"], row["id"])).fetchone()
        if artifact is None:
            raise ValueError("process_fault_model_artifact_missing")
        equal(dict(artifact), commit["artifact"])
        base = PurePath("tasks") / job["task_id"] / "modeling_artifacts"
        expected_files = [str(PurePath(artifact[key]).relative_to(job["workspace"])
                              if PurePath(artifact[key]).is_absolute() else base / artifact[key])
                          for key in ("model_path", "pmml_path") if artifact[key]]
        for name in expected_files:
            if ".." in PurePath(name).parts:
                raise ValueError("process_fault_model_path_escape")
            PurePath(name).relative_to(base)
    elif commit["table"] == "experiments_batch":
        expected_files = _batch_commit(ctx, job, launch, output, commit)
    elif commit["table"] == "task_artifacts":
        equal((job["module"], job["entrypoint"]),
              ("marvis.packs.analysis.tools", "tool_portfolio_report"))
        row = ctx.conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?",
                               (output.get("artifact_id"), ctx.binding.task_id)).fetchone()
        if row is None or row["id"] in launch["before_ids"]:
            raise ValueError("process_fault_fresh_portfolio_missing")
        equal(dict(row), commit["row"])
        relative = str(PurePath("tasks") / ctx.binding.task_id / "portfolio" /
                       f"portfolio_report_{row['content_hash'][:16]}.xlsx")
        original_path = str(PurePath(job["workspace"]) / relative)
        equal((row["kind"], row["origin_tool"], row["path"], output.get("report_path"),
               output.get("artifact_content_hash")),
              ("portfolio_report_xlsx", "analysis.portfolio_report", original_path,
               original_path, row["content_hash"]))
        equal(json.loads(row["provenance_json"]), {
            "schema_version": "portfolio-report-artifact.v1", "producer_version": "analysis.portfolio_report.v1",
            "task_id": ctx.binding.task_id, "input_hash": _sha(job["inputs"]), "sheets": output.get("sheets")})
        expected_detail = {"report_path": original_path, "sheets": output["sheets"],
                           "artifact_id": row["id"], "artifact_content_hash": row["content_hash"]}
        audits = [dict(item) for item in ctx.conn.execute(
            "SELECT * FROM audit WHERE kind='analysis.portfolio.report' AND target_ref=? AND outcome='succeeded'",
            (ctx.binding.task_id,)) if json.loads(item["detail_json"]) == expected_detail]
        if len(audits) != 1:
            raise ValueError("process_fault_portfolio_atomic_audit_missing")
        equal(commit["audit"], audits[0])
        expected_files = [relative]
        equal(commit["files"][0]["sha256"], row["content_hash"])
    else:
        raise _Unsupported("process_fault_commit_producer_unavailable")
    equal([item["relative_path"] for item in commit["files"]], expected_files)
    for item in commit["files"]:
        equal(_sha(_file(ctx.archive, "workspace/" + item["relative_path"]).read_bytes()), item["sha256"])
    return _sha(commit)


def _process_original(ctx, fault, events):
    prefix = "workspace/.runtime-eval-fault/"
    equal(fault["originals_relative_path"], ".runtime-eval-fault")
    def read(name):
        return _json(ctx.archive, prefix + "launch/" + name)
    arm, claimed, bound = (_json(ctx.archive, prefix + name) for name in
                           ("armed.json", "claimed.json", "job-binding.json"))
    launch, ready = read("launch.json"), read("ready.json")
    job = bound["job"]
    kind = fault["kind"]
    equal((arm["task_id"], arm["plan_id"], arm["step_id"], arm["tool"], arm["kind"]),
          (ctx.binding.task_id, fault["plan_id"], fault["step_id"], fault["tool"], kind))
    equal(arm.get("approval_step_id"), fault.get("approval_step_id"))
    equal(arm.get("approval_tool"), fault.get("approval_tool"))
    equal(bound["arm_sha256"], _sha(arm))
    equal(claimed["arm_sha256"], _sha(arm))
    equal(claimed["invocation_id"], job["invocation_id"])
    equal(launch["kind"], kind)
    equal(ready["kind"], kind)
    equal(ready, fault["ready"])
    equal(ready["launch_sha256"], _sha(launch))
    expected_binding = {key: job.get(key) for key in ("task_id", "module", "entrypoint", "invocation_id")}
    expected_binding.update(inputs_sha256=_sha(job["inputs"]), workspace=job["workspace"])
    equal(launch["binding"], expected_binding)
    equal(ready["binding"], expected_binding)
    run = ctx.conn.execute("SELECT * FROM plan_step_runs WHERE id=? AND plan_id=? AND step_id=?",
                           (job["invocation_id"], fault["plan_id"], fault["step_id"])).fetchone()
    if run is None or not run["dispatch_started_at"]:
        raise ValueError("process_fault_original_native_dispatch_missing")
    equal(_sha(json.loads(run["input_json"])), _sha(job["inputs"]))
    equal(run["tool_ref"].split("@", 1)[0], fault["tool"])
    equal(job["task_id"], ctx.binding.task_id)
    worker, host = ready["worker"], launch["host"]
    if (worker["ppid"] != host["pid"] or worker["pid"] == host["pid"]
            or worker["create_time"] < host["create_time"] or ready["elapsed_ns"] < claimed["elapsed_ns"]):
        raise ValueError("process_fault_worker_identity_order_invalid")
    if kind == "service_restart_mid_step_once":
        restart = _json(ctx.archive, prefix + "restart.json")
        equal(restart, fault["restart"])
        before, after = restart["before"], restart["after"]
        if (ready["phase"] != "before_tool" or before["pid"] != host["pid"]
                or before["create_time"] != host["create_time"] or before["pid"] == after["pid"]
                or before["create_time"] == after["create_time"]
                or before["workspace_sha256"] != after["workspace_sha256"]
                or restart["signal"] != "SIGKILL" or restart["old_returncode"] != -9
                or ready["elapsed_ns"] > restart["started_elapsed_ns"]
                or restart["started_elapsed_ns"] > restart["finished_elapsed_ns"]
                or [epoch["ordinal"] for epoch in restart["epochs"]] != [0, 1]):
            raise ValueError("process_fault_real_service_restart_missing")
        equal(restart["epochs"][0]["identity"], before)
        equal(restart["epochs"][1]["identity"], after)
        return {"native_run_id": run["id"], "worker": worker, "service_epochs": restart["epochs"]}
    terminal = read("terminal.json")
    equal(terminal["kind"], kind)
    equal(terminal["launch_sha256"], _sha(launch))
    if not 0 <= terminal["started_elapsed_ns"] <= ready["elapsed_ns"] <= terminal["finished_elapsed_ns"]:
        raise ValueError("process_fault_terminal_order_invalid")
    if kind == "worker_crash_after_commit_once":
        signal = read("signal.json")
        equal(signal["worker"], worker)
        equal(signal["host"], host)
        equal(signal["launch_sha256"], _sha(launch))
        if (ready["phase"] != "after_business_commit_before_protocol" or signal["signal"] != 9
                or terminal["terminal"] != "returned" or terminal["returncode"] != -9
                or not ready["elapsed_ns"] <= signal["started_elapsed_ns"] <= terminal["finished_elapsed_ns"]
                or signal["finished_elapsed_ns"] < signal["started_elapsed_ns"]):
            raise ValueError("process_fault_actual_worker_crash_missing")
        _commit(ctx, job, launch, ready, signal, read)
    elif kind == "tool_timeout_once":
        if (ready["phase"] != "before_tool" or terminal["terminal"] != "timeout"
                or terminal["supervisor_timeout_seconds"] != 2 or terminal["observed_timeout_seconds"] != 2
                or terminal["finished_elapsed_ns"] - terminal["started_elapsed_ns"] < 2_000_000_000):
            raise ValueError("process_fault_actual_timeout_missing")
    elif kind == "cancel_then_resume_once":
        cancelled = fault["cancellation"]
        actual = [event for event in events if event["ordinal"] == cancelled["http_ordinal"]]
        if (len(actual) != 1 or actual[0]["method"] != "POST"
                or actual[0]["stage"] != "process_fault_plan_cancel" or actual[0]["status_code"] != 200
                or actual[0]["started_elapsed_ns"] < ready["elapsed_ns"]
                or cancelled["body"]["plan"]["id"] != fault["plan_id"]
                or cancelled["body"]["plan"]["status"] != "cancelled"):
            raise ValueError("process_fault_actual_plan_cancel_missing")
        equal(read("release"), {"reason": "native_plan_cancel_accepted"})
    else:
        raise _Unsupported("process_fault_kind_unavailable")
    return {"native_run_id": run["id"], "worker": worker, "terminal": terminal}


def revalidate_process_recovery_archive(archive, *, expected_manifest_sha256, frozen_binding):
    from .runtime_archive_safety import _before, _private, _revalidate

    def work(ctx):
        _before(ctx)
        private = _private(ctx)
        events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
        rows = private.get("runtime_faults", [])
        if len(rows) != 1 or len(ctx.binding.recovery_action_indices) != 1:
            raise _Unsupported("one original process fault and recovery action required")
        fault = rows[0]
        recovery_index = ctx.binding.recovery_action_indices[0]
        action = ctx.binding.case.actions[recovery_index - 1]
        source = action.fault_source_action
        original = ctx.binding.case.actions[source - 1]
        downstream = original.fault.downstream_tool
        approval_step = fault.get("approval_step_id", fault["step_id"])
        equal(fault["schema"], "marvis.runtime-process-fault.v1")
        equal((fault["action_index"], fault["action_sha256"], fault["kind"], fault["tool"]),
              (source, digest(original.model_dump()), original.fault.kind, downstream or action.tool))
        equal(action.tool, original.tool)
        if downstream is not None:
            gates = {"analysis.portfolio_report": "analysis.portfolio_gate_summary",
                     "modeling.train_models": "modeling.configure_tuning"}
            if downstream not in gates:
                raise ValueError("process_fault_downstream_tool_unsupported")
            gate_tool = gates[downstream]
            equal((original.kind, original.tool, original.fault.kind, fault.get("approval_tool")),
                  ("approve_step", gate_tool, "worker_crash_after_commit_once", gate_tool))
            gate = ctx.conn.execute("SELECT s.*,p.task_id FROM plan_steps s JOIN plans p ON p.id=s.plan_id "
                "WHERE s.id=? AND s.plan_id=? AND p.task_id=?",
                (approval_step, fault["plan_id"], ctx.binding.task_id)).fetchone()
            target_plugin, target_name = downstream.split(".")
            reports = ctx.conn.execute("SELECT * FROM plan_steps WHERE plan_id=? AND tool_plugin=? AND tool_name=?",
                                       (fault["plan_id"], target_plugin, target_name)).fetchall()
            if (gate is None or (gate["tool_plugin"], gate["tool_name"]) != tuple(gate_tool.split("."))
                    or not gate["needs_confirmation"] or not gate["confirmed"] or gate["status"] != "done"
                    or len(reports) != 1 or reports[0]["id"] != fault["step_id"]
                    or approval_step not in json.loads(reports[0]["depends_on_json"])
                    or reports[0]["needs_confirmation"]):
                raise ValueError("process_fault_gate_downstream_binding_missing")
        elif "approval_step_id" in fault or "approval_tool" in fault:
            raise ValueError("process_fault_undeclared_downstream_target")
        equal(fault["task_id"], ctx.binding.task_id)
        before = original_target(fault["before"], task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])
        if any(before["tables"].values()):
            raise _Unsupported("process fault requires an original fresh native target")
        posted = [event for event in events if event["ordinal"] == fault["http_ordinal"]]
        if len(posted) != 1:
            raise ValueError("process_fault_original_approval_missing")
        equal((posted[0]["method"], posted[0]["stage"], posted[0]["status_code"]),
              ("POST", "process_fault_native_approval", 202))
        decision = ctx.conn.execute("SELECT * FROM decision_records WHERE id=? AND task_id=? AND plan_id=? AND step_id=?",
            (fault["received_response"]["decision_id"], ctx.binding.task_id, fault["plan_id"], approval_step)).fetchone()
        declarations = [item for item in ctx.binding.governed_actions if item.action_index == source]
        if decision is None or len(declarations) != 1:
            raise _Unsupported("frozen original native decision binding required")
        equal(digest(dict(decision)), declarations[0].decision_record_sha256)
        equal((declarations[0].plan_id, declarations[0].step_id, declarations[0].decision_id),
              (fault["plan_id"], approval_step, decision["id"]))
        equal((decision["decision"], decision["reason"]), ("approve", original.content))
        evidence = _process_original(ctx, fault, events)
        recovery = fault.get("recovery")
        if not isinstance(recovery, dict):
            raise _Unsupported("explicit recovery request and actual readback required")
        equal((recovery["action_index"], recovery["action_sha256"]), (recovery_index, digest(action.model_dump())))
        final = original_target(recovery["after_readback"], task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])
        equal(archived_target(ctx, fault["plan_id"], fault["step_id"]), {"target": final["target"], "tables": final["tables"]})
        ordinals = recovery["http_ordinals"]
        if not ordinals or ordinals != list(range(ordinals[0], ordinals[-1] + 1)) or ordinals[0] <= fault["http_ordinal"]:
            raise ValueError("process_fault_recovery_http_sequence_invalid")
        reads = [event for event in events if event["ordinal"] in ordinals]
        allowed = {("GET", "read_plans"), ("GET", "read_job"),
                   ("POST", "process_fault_native_reconcile"), ("POST", "process_fault_native_retry")}
        if len(reads) != len(ordinals) or any((event["method"], event["stage"]) not in allowed for event in reads):
            raise ValueError("process_fault_recovery_http_boundary_invalid")
        writes = [event for event in reads if event["method"] == "POST"]
        if len(writes) > 1 or (writes and writes[0]["status_code"] != recovery["response_status"]):
            raise ValueError("process_fault_recovery_repeated_write")
        completed = final["target"]["plan_status"] == "done" and final["target"]["status"] == "done" and bool(final["target"]["output_ref"])
        equal(recovery["outcome"], "done" if completed else "not_recovered")
        if completed:
            native_completed_output(ctx, fault["plan_id"], fault["step_id"], private)
        return {"recoveries": [{"action_index": recovery_index, "fault_action_index": source,
                                 "fault_executed": True, "recovery_completed": completed, **evidence}],
                "recovery_completed": completed, "fault_executed": True}

    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "process-recovery",
                       "original_process_fault_and_native_recovery", work)
