"""Declared report-response loss with read-only recovery through the real API."""
from contextlib import closing
from pathlib import Path
import sqlite3
import time

from .runtime_contracts import digest

_OUTPUTS = ("pmml_scoring_result.json", "pmml_scores.parquet", "validation_results.json")


def validation_state(workspace, task_id):
    workspace = Path(workspace).resolve(strict=True)
    database = workspace / "marvis.sqlite"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        task = conn.execute("SELECT id,task_type,run_mode,report_values_revision FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None:
            raise ValueError("validation_fault_task_missing")
        audits = [dict(row) for row in conn.execute(
            "SELECT id,kind FROM audit WHERE target_ref=? AND kind='report.agent_conclusions.confirm' ORDER BY id LIMIT 101", (task_id,))]
        jobs = [dict(row) for row in conn.execute("SELECT id,kind FROM jobs WHERE task_id=? ORDER BY id LIMIT 101", (task_id,))]
        if len(audits) > 100 or len(jobs) > 100:
            raise ValueError("validation_fault_snapshot_limit")
        conn.rollback()
    hashes = {}
    for name in _OUTPUTS:
        path = workspace / "tasks" / task_id / "outputs" / name
        if path.resolve() != path or not path.is_file() or path.stat().st_size > 64_000_000:
            raise ValueError("validation_fault_original_metrics_unavailable")
        hashes[name] = digest(path.read_bytes())
    return {"task": dict(task), "confirmation_audits": audits, "jobs": jobs, "metric_hashes": hashes}


def _capture(journey):
    start = time.monotonic_ns() - journey.origin_ns
    state = validation_state(journey.workspace, journey.task_id)
    return {"started_elapsed_ns": start, "finished_elapsed_ns": time.monotonic_ns() - journey.origin_ns,
            "state": state}


def discard_report_confirmation(journey, action, body):
    from .runtime_runner import RuntimeJourneyError

    if action.fault is None or action.fault.kind != "discard_success_response_once":
        raise RuntimeJourneyError("validation_fault_not_declared")
    if getattr(journey, "validation_faults", None):
        raise RuntimeJourneyError("validation_fault_already_dispatched")
    fault = {"schema": "marvis.validation-report-fault.v1", "action_index": journey.current_action_index,
             "action_sha256": digest(action.model_dump()), "task_id": journey.task_id,
             "kind": action.fault.kind, "request": body, "before": _capture(journey),
             "scope": "local_client_response_loss; not_remote_network_failure"}
    response = journey.request("POST", f"/api/tasks/{journey.task_id}/agent/report-draft/confirm",
                               label="validation_discard_report_response", json=body)
    fault.update(http_ordinal=journey.count, http_status=response.status_code, after_response=_capture(journey))
    journey.validation_faults = [fault]
    if response.status_code != 202:
        raise RuntimeJourneyError("validation_fault_confirmation_not_accepted")
    # Keep the actual reply only as an original. Recovery cannot use its body
    # or repeat the POST; it must query the native task and download its result.
    fault["received_response"] = response.json()
    fault["visible_response"] = "discarded_by_declared_client_reference_fault"


def recover_report(journey, action):
    from .runtime_runner import RuntimeJourneyError
    from .runtime_validation_adapter import download_reports

    faults = getattr(journey, "validation_faults", [])
    matches = [row for row in faults if row["action_index"] == action.fault_source_action]
    if len(matches) != 1 or "recovery" in matches[0]:
        raise RuntimeJourneyError("validation_recovery_original_unavailable")
    fault = matches[0]
    if fault["kind"] == "stale_report_revision_once":
        from .runtime_validation_stale_recovery import recover_stale_report
        return recover_stale_report(journey, action, fault)
    journey.record_human_action(action)
    begin = journey.count
    journey.wait_idle()
    download_reports(journey)
    fault["recovery"] = {"action_index": journey.current_action_index,
        "action_sha256": digest(action.model_dump()), "http_ordinals": list(range(begin + 1, journey.count + 1)),
        "after_readback": _capture(journey), "downloads": list(journey.validation_downloads)}
