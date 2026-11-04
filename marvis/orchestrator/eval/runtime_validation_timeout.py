"""Evaluator-only real PMML socket timeout; ordinary application paths are untouched.

The fixed launcher binds an exact task, loaded PMML and active native job. It
suspends only the owned Py4J JVM while the real scorer reads its socket with a
two-second deadline. No exception, score, metric or product recovery is invented.
"""
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import threading
import time

from .runtime_contracts import digest
from .runtime_process_faults import write_original

TIMEOUT_SECONDS = 2


def _state(workspace, task_id):
    db = Path(workspace).resolve(strict=True) / "marvis.sqlite"
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        task = conn.execute("SELECT id,task_type,run_mode,validation_workflow_version,status,report_values_revision FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None:
            raise ValueError("validation_timeout_task_missing")
        jobs = [dict(r) for r in conn.execute("SELECT id,task_id,kind,status,started_at,finished_at,error_name FROM jobs WHERE task_id=? ORDER BY created_at,id", (task_id,))]
        active = [dict(r) for r in conn.execute("SELECT id,task_id,kind FROM jobs WHERE status='running'")]
        return {"task": dict(task), "jobs": jobs, "active_jobs": active}


def _identity(process):
    return {"pid": process.pid, "create_time": process.create_time(), "ppid": process.ppid()}


def _owned_gateway(gateway):
    import psutil

    port = gateway.gateway_parameters.port
    host = psutil.Process(os.getpid())
    owned = []
    for child in host.children():
        if child.name() not in {"java", "java.exe"}:
            continue
        listeners = [c for c in child.net_connections(kind="tcp") if c.status == "LISTEN" and c.laddr.port == port]
        if listeners:
            owned.append(child)
    if len(owned) != 1:
        raise ValueError("validation_timeout_owned_gateway_ambiguous")
    return host, owned[0], port


def _same_process(process, identity):
    if _identity(process) != identity or not process.is_running():
        raise ValueError("validation_timeout_process_identity_changed")


@contextmanager
def evaluation_validation_timeout(workspace, *, origin_ns):
    """Installed solely by the fixed evaluation --serve-config launcher."""
    from pypmml.base import PMMLContext
    from py4j.java_gateway import GatewayConnection
    from py4j.protocol import Py4JNetworkError
    from marvis.validation.pmml_scoring import PmmlScorer, TASK_PMML_SCORERS

    directory = Path(workspace).resolve() / ".runtime-eval-fault/validation-timeout"
    score_native, send_native = PmmlScorer.score_chunk, GatewayConnection.send_command
    local = threading.local()

    def observe_send(connection, command):
        try:
            return send_native(connection, command)
        except Py4JNetworkError as exc:
            record = getattr(local, "record", None)
            if record is not None and isinstance(exc.cause, socket.timeout):
                peer = connection.socket.getpeername()
                if peer[1] != record["gateway_port"]:
                    raise ValueError("validation_timeout_socket_owner_changed") from exc
                record["socket_timeouts"].append({"error_type": type(exc.cause).__name__,
                    "peer_host": peer[0], "peer_port": peer[1], "socket_timeout_seconds": connection.socket.gettimeout(),
                    "elapsed_ns": time.monotonic_ns() - origin_ns})
            raise

    def score(scorer, frame):
        arm_path = directory / "armed.json"
        if not arm_path.is_file() or (directory / "claimed.json").exists() or not len(frame):
            return score_native(scorer, frame)
        arm = json.loads(arm_path.read_bytes())
        with TASK_PMML_SCORERS.lock:
            matches = [(task_id, identity) for task_id, (identity, cached) in TASK_PMML_SCORERS.entries.items() if cached is scorer]
        if len(matches) != 1 or matches[0][0] != arm["task_id"]:
            return score_native(scorer, frame)
        state = _state(workspace, arm["task_id"])
        if (matches[0][1][0] != arm["pmml_sha256"] or state["task"]["task_type"] != "validation"
                or state["task"]["run_mode"] != "agent" or state["task"]["validation_workflow_version"] != 2
                or len(state["active_jobs"]) != 1 or state["active_jobs"][0]["task_id"] != arm["task_id"]
                or state["active_jobs"][0]["kind"] != "agent"):
            raise ValueError("validation_timeout_native_binding_changed")
        gateway = PMMLContext.getOrCreate()._gateway._gateway
        host, jvm, port = _owned_gateway(gateway)
        identity = _identity(jvm)
        record = {"schema": "marvis.validation-pmml-timeout.v1", "arm_sha256": digest(arm),
            "task_id": arm["task_id"], "job_id": state["active_jobs"][0]["id"],
            "pmml_sha256": matches[0][1][0], "output_field": matches[0][1][1],
            "host": _identity(host), "jvm": identity, "gateway_port": port,
            "timeout_seconds": TIMEOUT_SECONDS, "socket_timeouts": [], "before": state}
        write_original(directory / "claimed.json", record)
        previous = gateway.gateway_parameters.read_timeout
        sockets = [(c.socket, c.socket.gettimeout()) for c in gateway._gateway_client.deque if c.socket]
        gateway.gateway_parameters.read_timeout = TIMEOUT_SECONDS
        for connection, _ in sockets:
            connection.settimeout(TIMEOUT_SECONDS)
        stopped = False
        local.record = record
        try:
            _same_process(jvm, identity)
            jvm.send_signal(signal.SIGSTOP)
            stopped = True
            deadline = time.monotonic() + 2
            while jvm.status() != "stopped" and time.monotonic() < deadline:
                time.sleep(0.005)
            if jvm.status() != "stopped":
                raise ValueError("validation_timeout_jvm_did_not_stop")
            record.update(signal="SIGSTOP", stopped_status=jvm.status(), started_elapsed_ns=time.monotonic_ns() - origin_ns)
            try:
                value = score_native(scorer, frame)
            except Exception as exc:
                record.update(outcome="native_exception", error_type=type(exc).__name__)
                raise
            else:
                record.update(outcome="native_return")
                return value
            finally:
                record["finished_elapsed_ns"] = time.monotonic_ns() - origin_ns
        finally:
            local.record = None
            gateway.gateway_parameters.read_timeout = previous
            for connection, old_timeout in sockets:
                try:
                    connection.settimeout(old_timeout)
                except OSError:
                    pass  # Native Py4J closes a timed-out connection.
            if stopped:
                _same_process(jvm, identity)
                jvm.send_signal(signal.SIGCONT)
                record.update(resume_signal="SIGCONT", resumed_elapsed_ns=time.monotonic_ns() - origin_ns,
                              resumed_identity=_identity(jvm))
            write_original(directory / "terminal.json", record)

    PmmlScorer.score_chunk, GatewayConnection.send_command = score, observe_send
    try:
        yield
    finally:
        PmmlScorer.score_chunk, GatewayConnection.send_command = score_native, send_native


def dispatch_validation_timeout(journey, action):
    from .runtime_runner import RuntimeJourneyError

    if action.kind != "start_validation_agent" or action.fault.kind != "tool_timeout_once":
        raise RuntimeJourneyError("validation_timeout_action_unsupported")
    if getattr(journey, "validation_faults", []):
        raise RuntimeJourneyError("validation_timeout_already_dispatched")
    directory = Path(journey.workspace).resolve() / ".runtime-eval-fault/validation-timeout"
    directory.mkdir(mode=0o700, parents=True)
    arm = {"schema": "marvis.validation-timeout-arm.v1", "kind": "tool_timeout_once", "task_id": journey.task_id,
           "action_index": journey.current_action_index, "action_sha256": digest(action.model_dump()),
           "pmml_sha256": next(m.sha256 for m in journey.case.materials if m.role == "pmml")}
    write_original(directory / "armed.json", arm)
    fault = {**arm, "before": _state(journey.workspace, journey.task_id)}
    journey.validation_faults = [fault]
    journey.record_human_action(action)
    response = journey.json_request("POST", f"/api/tasks/{journey.task_id}/agent/start",
        label="agent_initial_turn", json={"acceptance_mode": journey.case.acceptance_mode})
    fault.update(http_ordinal=journey.count, response=response)
    journey.wait_idle()
    fault["after_fault"] = _state(journey.workspace, journey.task_id)
    terminal = json.loads((directory / "terminal.json").read_bytes())
    fault["terminal_sha256"] = digest(terminal)
    fault["fault_executed"] = bool(terminal.get("socket_timeouts") and terminal.get("outcome") == "native_exception")
    write_original(directory / "after-fault.json", fault)
    if not fault["fault_executed"]:
        raise RuntimeJourneyError("validation_timeout_not_observed")


def recover_validation_execution(journey, action):
    from .runtime_runner import RuntimeJourneyError

    faults = getattr(journey, "validation_faults", [])
    if (len(faults) != 1 or faults[0]["action_index"] != action.fault_source_action
            or not faults[0].get("fault_executed") or "recovery" in faults[0]):
        raise RuntimeJourneyError("validation_timeout_original_missing")
    fault = faults[0]
    recovery = {"action_index": journey.current_action_index, "action_sha256": digest(action.model_dump()),
                "before": _state(journey.workspace, journey.task_id)}
    fault["recovery"] = recovery
    journey.record_human_action(action)
    response = journey.request("POST", f"/api/tasks/{journey.task_id}/agent/start",
        label="validation_timeout_retry", json={"acceptance_mode": journey.case.acceptance_mode})
    recovery.update(http_ordinal=journey.count, http_status=response.status_code, response=response.json())
    if response.status_code == 202:
        journey.wait_idle()
    recovery["after"] = _state(journey.workspace, journey.task_id)
    directory = Path(journey.workspace).resolve() / ".runtime-eval-fault/validation-timeout"
    write_original(directory / "after-retry.json", recovery)


def capture_validation_timeout_final(journey, fault):
    fault["confirmation"].update(downloads=list(journey.validation_downloads),
                                  final=_state(journey.workspace, journey.task_id))
    directory = Path(journey.workspace).resolve() / ".runtime-eval-fault/validation-timeout"
    write_original(directory / "confirmation.json", fault["confirmation"])
