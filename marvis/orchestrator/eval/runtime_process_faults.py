"""Source-owned process fault controls; ordinary application launch never imports these.

An executed fault is not evidence of recovery. Callers retain the native failed /
unknown-effect result and must evaluate recovery through the ordinary API reader.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

import psutil


DATASET_PRODUCERS = frozenset({
    ("marvis.packs.data_ops.tools", "tool_clean_format"),
    ("marvis.packs.data_ops.tools", "tool_dedup_rows"),
    ("marvis.packs.data_ops.tools", "tool_execute_join"),
})
MODEL_PRODUCERS = frozenset({("marvis.packs.modeling.tools", "tool_train_model")})
MODEL_BATCH_PRODUCERS = frozenset({("marvis.packs.modeling.tools", "tool_train_models")})
PORTFOLIO_PRODUCERS = frozenset({("marvis.packs.analysis.tools", "tool_portfolio_report")})
KINDS = frozenset({"worker_crash_after_commit_once", "tool_timeout_once",
                   "service_restart_mid_step_once", "cancel_then_resume_once"})


def sha(value):
    raw = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_original(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def process_identity(pid):
    process = psutil.Process(pid)
    return {"pid": process.pid, "create_time": process.create_time(), "ppid": process.ppid()}


def exact_process(identity):
    process = psutil.Process(identity["pid"])
    if process.create_time() != identity["create_time"]:
        raise ValueError("fault_process_identity_changed")
    return process


def binding(job):
    return {key: job.get(key) for key in ("task_id", "module", "entrypoint", "invocation_id")} | {
        "inputs_sha256": sha(job["inputs"]), "workspace": str(Path(job["workspace"]).resolve(strict=True))}


def _database(workspace):
    path = Path(workspace) / "marvis.sqlite"
    if path.is_symlink() or not path.is_file():
        raise ValueError("fault_database_unavailable")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")
    return connection


def _file(workspace, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(workspace) / path
    resolved = path.resolve(strict=True)
    resolved.relative_to(Path(workspace).resolve(strict=True))
    if path.absolute() != resolved or not resolved.is_file():
        raise ValueError("fault_commit_file_unsafe")
    return {"relative_path": str(resolved.relative_to(workspace)), "sha256": sha(resolved.read_bytes())}


def committed_ids(job):
    """Snapshot before dispatch; after-commit evidence must name a new row."""
    producer = (job["module"], job["entrypoint"])
    if producer not in DATASET_PRODUCERS | MODEL_PRODUCERS | MODEL_BATCH_PRODUCERS | PORTFOLIO_PRODUCERS:
        raise ValueError("fault_after_commit_producer_unsupported")
    query = ("SELECT id FROM datasets WHERE task_id=?" if producer in DATASET_PRODUCERS else
             "SELECT id FROM task_artifacts WHERE task_id=?" if producer in PORTFOLIO_PRODUCERS else
             "SELECT id FROM experiments WHERE task_id=?")
    with closing(_database(job["workspace"])) as conn:
        return [row["id"] for row in conn.execute(query, (job["task_id"],))]


def verify_business_commit(job, output, before_ids):
    """Read actual committed rows/files; a returned dict alone never proves commit."""
    producer = (job["module"], job["entrypoint"])
    if producer not in DATASET_PRODUCERS | MODEL_PRODUCERS | MODEL_BATCH_PRODUCERS | PORTFOLIO_PRODUCERS:
        raise ValueError("fault_after_commit_producer_unsupported")
    with closing(_database(job["workspace"])) as conn:
        if producer in MODEL_BATCH_PRODUCERS:
            return _model_batch_commit(conn, job, output, before_ids)
        if producer in PORTFOLIO_PRODUCERS:
            identity = output.get("artifact_id")
            row = conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?",
                               (identity, job["task_id"])).fetchone()
            if row is None or identity in before_ids:
                raise ValueError("fault_fresh_portfolio_commit_missing")
            content_hash = row["content_hash"]
            expected_path = (Path(job["workspace"]) / "tasks" / job["task_id"] / "portfolio" /
                             f"portfolio_report_{content_hash[:16]}.xlsx")
            provenance = json.loads(row["provenance_json"])
            expected_provenance = {"schema_version": "portfolio-report-artifact.v1",
                "producer_version": "analysis.portfolio_report.v1", "task_id": job["task_id"],
                "input_hash": sha(job["inputs"]), "sheets": output.get("sheets")}
            if (row["kind"] != "portfolio_report_xlsx" or row["origin_tool"] != "analysis.portfolio_report"
                    or row["path"] != str(expected_path) or output.get("report_path") != str(expected_path)
                    or output.get("artifact_content_hash") != content_hash or provenance != expected_provenance):
                raise ValueError("fault_portfolio_commit_binding_invalid")
            artifact = _file(job["workspace"], expected_path)
            if artifact["sha256"] != content_hash:
                raise ValueError("fault_portfolio_commit_content_changed")
            expected_detail = {"report_path": str(expected_path), "sheets": output["sheets"],
                               "artifact_id": identity, "artifact_content_hash": content_hash}
            audits = [dict(item) for item in conn.execute(
                "SELECT * FROM audit WHERE kind='analysis.portfolio.report' AND target_ref=? AND outcome='succeeded'",
                (job["task_id"],)) if json.loads(item["detail_json"]) == expected_detail]
            if len(audits) != 1:
                raise ValueError("fault_portfolio_atomic_audit_missing")
            return {"table": "task_artifacts", "row": dict(row), "audit": audits[0], "files": [artifact]}
        if producer in DATASET_PRODUCERS:
            field = "result_dataset_id" if job["entrypoint"] == "tool_execute_join" else "dataset_id"
            identity = output.get(field)
            row = conn.execute("SELECT * FROM datasets WHERE id=? AND task_id=?",
                               (identity, job["task_id"])).fetchone()
            if row is None or identity in before_ids:
                raise ValueError("fault_fresh_dataset_commit_missing")
            dataset_path = Path(row["source_path"])
            if not dataset_path.is_absolute():
                dataset_path = Path(job["datasets_root"]) / dataset_path
            artifact = _file(job["workspace"], dataset_path)
            if row["content_hash"] != artifact["sha256"]:
                raise ValueError("fault_dataset_commit_content_changed")
            return {"table": "datasets", "row": dict(row), "files": [artifact]}
        identity = output.get("experiment_id")
        row = conn.execute("SELECT * FROM experiments WHERE id=? AND task_id=?",
                           (identity, job["task_id"])).fetchone()
        if row is None or identity in before_ids or row["artifact_id"] != output.get("artifact_id"):
            raise ValueError("fault_fresh_model_commit_missing")
        artifact = conn.execute("SELECT * FROM model_artifacts WHERE id=? AND experiment_id=?",
                                (row["artifact_id"], identity)).fetchone()
        if artifact is None or row["status"] != "trained":
            raise ValueError("fault_model_commit_incomplete")
        base = Path(job["workspace"]) / "tasks" / job["task_id"] / "modeling_artifacts"
        paths = [Path(artifact[key]) if Path(artifact[key]).is_absolute() else base / artifact[key]
                 for key in ("model_path", "pmml_path") if artifact[key]]
        for path in paths:
            path.resolve(strict=True).relative_to(base.resolve(strict=True))
        return {"table": "experiments", "row": dict(row), "artifact": dict(artifact),
                "files": [_file(job["workspace"], path) for path in paths]}


def _model_batch_commit(conn, job, output, before_ids):
    identities = output.get("experiment_ids")
    recipes = job["inputs"].get("recipes")
    summaries = output.get("experiments")
    if (not isinstance(identities, list) or not 1 <= len(identities) <= 20
            or any(not isinstance(value, str) or not value for value in identities)
            or len(set(identities)) != len(identities) or set(identities).intersection(before_ids)
            or not isinstance(summaries, list) or len(summaries) != len(identities)
            or [item.get("experiment_id") for item in summaries] != identities
            or not isinstance(recipes, list) or len(set(recipes)) != len(recipes)
            or sorted(item.get("recipe") for item in summaries) != sorted(recipes)
            or output.get("failed") != [] or output.get("best_experiment_id") not in identities):
        raise ValueError("fault_model_batch_output_incomplete")
    fresh = {row["id"] for row in conn.execute("SELECT id FROM experiments WHERE task_id=?", (job["task_id"],))} - set(before_ids)
    if fresh != set(identities):
        raise ValueError("fault_model_batch_fresh_inventory_differs")
    rows, artifacts, files, audits = [], [], [], []
    base = Path(job["workspace"]) / "tasks" / job["task_id"] / "modeling_artifacts"
    for summary in summaries:
        row = conn.execute("SELECT * FROM experiments WHERE id=? AND task_id=?",
                           (summary["experiment_id"], job["task_id"])).fetchone()
        if row is None or row["status"] != "trained" or row["recipe_id"] != summary["recipe"]:
            raise ValueError("fault_model_batch_experiment_incomplete")
        artifact = conn.execute("SELECT * FROM model_artifacts WHERE id=? AND experiment_id=?",
                                (row["artifact_id"], row["id"])).fetchone()
        if artifact is None or not artifact["model_path"]:
            raise ValueError("fault_model_batch_artifact_missing")
        paths = [Path(artifact[key]) if Path(artifact[key]).is_absolute() else base / artifact[key]
                 for key in ("model_path", "pmml_path") if artifact[key]]
        for path in paths:
            path.resolve(strict=True).relative_to(base.resolve(strict=True))
        expected = {"artifact_id": artifact["id"], "algorithm": artifact["algorithm"],
                    "feature_count": len(json.loads(artifact["feature_list_json"]))}
        attached = [dict(item) for item in conn.execute(
            "SELECT * FROM audit WHERE kind='modeling.experiment.trained' AND target_ref=? AND outcome='succeeded'",
            (row["id"],)) if json.loads(item["detail_json"]) == expected]
        if len(attached) != 1:
            raise ValueError("fault_model_batch_training_audit_missing")
        rows.append(dict(row))
        artifacts.append(dict(artifact))
        files.extend(_file(job["workspace"], path) for path in paths)
        audits.extend(attached)
    if len({file["relative_path"] for file in files}) != len(files):
        raise ValueError("fault_model_batch_artifact_alias")
    return {"table": "experiments_batch", "rows": rows, "artifacts": artifacts, "files": files, "audits": audits}


class FaultLaunch:
    """One fixed launcher boundary, never constructed from public API input."""

    def __init__(self, directory, job, kind, *, origin_ns=None):
        if kind not in KINDS or not job.get("builtin"):
            raise ValueError("fault_requires_declared_builtin_worker")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(mode=0o700)
        self.job = dict(job)
        self.kind = kind
        self.origin_ns = time.monotonic_ns() if origin_ns is None else origin_ns
        self.host = process_identity(os.getpid())
        self.before_ids = committed_ids(job) if kind == "worker_crash_after_commit_once" else []
        self.contract = {"schema": "marvis.process-fault-launch.v1", "kind": kind,
                         "binding": binding(job), "host": self.host, "origin_ns": self.origin_ns,
                         "before_ids": self.before_ids}
        write_original(self.directory / "launch.json", self.contract)

    def launch_args(self, python):
        return [python, "-m", "marvis.orchestrator.eval.runtime_fault_worker",
                "--control-directory", str(self.directory)]

    def ready(self, timeout=10):
        until = time.monotonic() + timeout
        path = self.directory / "ready.json"
        while time.monotonic() < until:
            if path.is_file():
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    time.sleep(0.01)
                    continue
                if row["launch_sha256"] != sha(self.contract) or row["binding"] != binding(self.job):
                    raise ValueError("fault_worker_marker_wrong_binding")
                worker = exact_process(row["worker"])
                if worker.ppid() != self.host["pid"] or worker.pid == self.host["pid"]:
                    raise ValueError("fault_worker_not_owned_child")
                if worker.cmdline()[1:] != self.launch_args("")[1:]:
                    raise ValueError("fault_worker_launch_command_changed")
                exact_process(self.host)
                if self.kind == "worker_crash_after_commit_once":
                    output = json.loads((self.directory / "worker-output.json").read_text())
                    proof = verify_business_commit(self.job, output, self.before_ids)
                    if sha(proof) != row["commit_sha256"] or sha(output) != row["output_sha256"]:
                        raise ValueError("fault_worker_commit_changed")
                return row
            time.sleep(0.01)
        raise TimeoutError("fault_worker_rendezvous_timeout")

    def run(self, native_run, python, **kwargs):
        """Invoke the unchanged source-owned supervisor and retain its real exit."""
        from marvis.job_cancellation import JobCancelled

        started = time.monotonic_ns() - self.origin_ns
        terminal = {"schema": "marvis.process-fault-terminal.v1", "kind": self.kind,
                    "launch_sha256": sha(self.contract), "started_elapsed_ns": started,
                    "supervisor_timeout_seconds": kwargs["timeout"]}
        try:
            completed = native_run(python, self.job, **(kwargs | {"launch_args": self.launch_args(python)}))
            terminal.update(terminal="returned", returncode=completed.returncode,
                            stdout_sha256=sha(completed.stdout.encode()), stderr_sha256=sha(completed.stderr.encode()))
            return completed
        except subprocess.TimeoutExpired as exc:
            terminal.update(terminal="timeout", observed_timeout_seconds=exc.timeout)
            raise
        except JobCancelled:
            terminal.update(terminal="cancelled")
            raise
        except BaseException as exc:
            terminal.update(terminal="unexpected_exception", exception_type=type(exc).__name__)
            raise
        finally:
            terminal["finished_elapsed_ns"] = time.monotonic_ns() - self.origin_ns
            write_original(self.directory / "terminal.json", terminal)

    def kill_worker(self, ready):
        if self.kind != "worker_crash_after_commit_once":
            raise ValueError("fault_kill_requires_after_commit_kind")
        # Recheck both the live parent chain and immutable commit immediately
        # before signaling. Do not accept a PID supplied by a case or marker alone.
        observed = self.ready()
        if ready != observed:
            raise ValueError("fault_worker_marker_changed")
        worker = exact_process(ready["worker"])
        started = time.monotonic_ns() - self.origin_ns
        worker.send_signal(signal.SIGKILL)
        row = {"schema": "marvis.process-fault-signal.v1", "kind": self.kind,
               "launch_sha256": sha(self.contract), "worker": ready["worker"], "host": self.host,
               "signal": int(signal.SIGKILL), "started_elapsed_ns": started,
               "finished_elapsed_ns": time.monotonic_ns() - self.origin_ns,
               "commit_sha256": ready["commit_sha256"], "recovery_outcome": "not_evaluated"}
        write_original(self.directory / "signal.json", row)
        return row


def verify_worker_crash_originals(directory, job):
    """Re-read fixed originals and native commit; never infer recovered from kill."""
    directory = Path(directory)
    launch, ready, signal_row, terminal, output, commit = (
        json.loads((directory / name).read_text()) for name in
        ("launch.json", "ready.json", "signal.json", "terminal.json", "worker-output.json", "commit.json"))
    kind = "worker_crash_after_commit_once"
    if (launch["kind"] != kind or launch["binding"] != binding(job)
            or any(row["kind"] != kind or row["launch_sha256"] != sha(launch)
                   for row in (ready, signal_row, terminal))
            or ready["binding"] != launch["binding"]
            or ready["worker"] != signal_row["worker"]
            or signal_row["host"] != launch["host"]
            or ready["worker"]["ppid"] != launch["host"]["pid"]
            or ready["worker"]["pid"] == launch["host"]["pid"]
            or ready["worker"]["create_time"] < launch["host"]["create_time"]
            or ready["phase"] != "after_business_commit_before_protocol"
            or signal_row["signal"] != signal.SIGKILL
            or terminal["terminal"] != "returned" or terminal["returncode"] != -signal.SIGKILL
            or not 0 <= terminal["started_elapsed_ns"] <= ready["elapsed_ns"]
                <= signal_row["started_elapsed_ns"] <= signal_row["finished_elapsed_ns"]
                <= terminal["finished_elapsed_ns"]):
        raise ValueError("fault_crash_original_binding_invalid")
    actual = verify_business_commit(job, output, launch["before_ids"])
    if (commit != actual or sha(commit) != ready["commit_sha256"]
            or ready["commit_sha256"] != signal_row["commit_sha256"]
            or sha(output) != ready["output_sha256"]):
        raise ValueError("fault_crash_original_commit_invalid")
    return {"fault_executed": True, "fault_kind": kind, "recovery_outcome": "not_evaluated",
            "commit_sha256": sha(commit), "worker": ready["worker"],
            "elapsed_ns": terminal["finished_elapsed_ns"] - terminal["started_elapsed_ns"]}
