"""Explicit evaluator-only launch scope and one exact native invocation binding."""
from contextlib import closing, contextmanager
import json
from pathlib import Path
import time

from .runtime_process_faults import FaultLaunch, KINDS, _database, sha, write_original


_DOWNSTREAM_GATES = {"analysis.portfolio_report": "analysis.portfolio_gate_summary",
                     "modeling.train_models": "modeling.configure_tuning"}


def downstream_commit_target(workspace, task_id, plan_id, approval_step_id, target_tool):
    """One exact native commit step directly dependent on its explicit gate."""
    if target_tool not in _DOWNSTREAM_GATES:
        raise ValueError("fault_downstream_tool_unsupported")
    gate_tool = tuple(_DOWNSTREAM_GATES[target_tool].split("."))
    target_plugin, target_name = target_tool.split(".")
    with closing(_database(workspace)) as conn:
        gate = conn.execute("SELECT s.*,p.task_id FROM plan_steps s JOIN plans p ON p.id=s.plan_id "
            "WHERE s.id=? AND s.plan_id=? AND p.task_id=?", (approval_step_id, plan_id, task_id)).fetchone()
        targets = conn.execute("SELECT * FROM plan_steps WHERE plan_id=? AND tool_plugin=? AND tool_name=?",
                               (plan_id, target_plugin, target_name)).fetchall()
        if (gate is None or (gate["tool_plugin"], gate["tool_name"]) != gate_tool
                or not gate["needs_confirmation"] or len(targets) != 1
                or approval_step_id not in json.loads(targets[0]["depends_on_json"])
                or targets[0]["needs_confirmation"]):
            raise ValueError("fault_exact_gate_downstream_binding_missing")
        return dict(targets[0]), dict(gate)


def arm_process_fault(workspace, *, kind, task_id, plan_id, step_id, tool, approval_step_id=None):
    if kind not in KINDS:
        raise ValueError("process_fault_kind_unsupported")
    directory = Path(workspace).resolve(strict=True) / ".runtime-eval-fault"
    directory.mkdir(mode=0o700)
    arm = {"schema": "marvis.process-fault-arm.v1", "kind": kind, "task_id": task_id,
           "plan_id": plan_id, "step_id": step_id, "tool": tool,
           "timeout_seconds": 2 if kind == "tool_timeout_once" else None}
    if approval_step_id is not None:
        report, _ = downstream_commit_target(workspace, task_id, plan_id, approval_step_id, tool)
        if (kind != "worker_crash_after_commit_once" or tool not in _DOWNSTREAM_GATES
                or step_id != report["id"] or report["status"] != "pending"):
            raise ValueError("portfolio_fault_target_not_fresh")
        with closing(_database(workspace)) as conn:
            if conn.execute("SELECT 1 FROM plan_step_runs WHERE step_id=?", (step_id,)).fetchone():
                raise ValueError("portfolio_fault_target_already_dispatched")
        arm.update(approval_step_id=approval_step_id, approval_tool=_DOWNSTREAM_GATES[tool])
    write_original(directory / "armed.json", arm)
    return directory


def _matches(job, arm):
    if job.get("task_id") != arm["task_id"] or not job.get("invocation_id"):
        return False
    if arm.get("approval_step_id"):
        report, gate = downstream_commit_target(job["workspace"], arm["task_id"], arm["plan_id"], arm["approval_step_id"], arm["tool"])
        if (report["id"] != arm["step_id"] or arm["tool"] not in _DOWNSTREAM_GATES
                or arm["approval_tool"] != _DOWNSTREAM_GATES[arm["tool"]]
                or arm["kind"] != "worker_crash_after_commit_once"):
            raise ValueError("portfolio_fault_downstream_identity_changed")
    with closing(_database(job["workspace"])) as conn:
        row = conn.execute(
            "SELECT r.*,p.task_id FROM plan_step_runs r JOIN plans p ON p.id=r.plan_id WHERE r.id=?",
            (job["invocation_id"],)).fetchone()
        if row is None or (row["task_id"], row["plan_id"], row["step_id"]) != (
                arm["task_id"], arm["plan_id"], arm["step_id"]):
            return False
        if arm.get("approval_step_id") and (gate["status"] != "done" or not gate["confirmed"]):
            raise ValueError("portfolio_fault_gate_not_completed")
        if (str(row["tool_ref"]).split("@", 1)[0] != arm["tool"]
                or sha(json.loads(row["input_json"])) != sha(job["inputs"])
                or not row["dispatch_started_at"]):
            raise ValueError("process_fault_native_invocation_binding_changed")
        return True


@contextmanager
def evaluation_fault_launcher(workspace, *, origin_ns):
    """Used only by the fixed evaluation --serve-config entrypoint.

    No environment flag or application API installs this scope. Its armed
    command never accepts executable code, paths, process IDs or callbacks.
    """
    from marvis.plugins import runner

    directory = Path(workspace).resolve() / ".runtime-eval-fault"
    original = runner._run_worker_process

    def launch(python, job, **kwargs):
        arm_path = directory / "armed.json"
        if not arm_path.is_file() or (directory / "claimed.json").exists():
            return original(python, job, **kwargs)
        arm = json.loads(arm_path.read_text())
        if not _matches(job, arm):
            return original(python, job, **kwargs)
        # O_EXCL claims exactly once, including across a real server restart.
        write_original(directory / "claimed.json", {"arm_sha256": sha(arm),
                       "invocation_id": job["invocation_id"], "elapsed_ns": time.monotonic_ns() - origin_ns})
        fault = FaultLaunch(directory / "launch", job, arm["kind"], origin_ns=origin_ns)
        public_binding = {key: job[key] for key in (
            "builtin", "task_id", "module", "entrypoint", "invocation_id", "workspace", "datasets_root", "inputs")}
        write_original(directory / "job-binding.json", {"job": public_binding, "arm_sha256": sha(arm)})
        if arm["kind"] == "tool_timeout_once":
            kwargs = kwargs | {"timeout": arm["timeout_seconds"]}
        return fault.run(original, python, **kwargs)

    runner._run_worker_process = launch
    try:
        yield
    finally:
        runner._run_worker_process = original


def attached_launch(directory, *, deadline):
    """Attach only to the launcher-written binding, then revalidate the real PID."""
    directory = Path(directory).resolve(strict=True)
    while time.monotonic() < deadline:
        path = directory / "job-binding.json"
        if path.is_file():
            try:
                bound = json.loads(path.read_text())
                arm = json.loads((directory / "armed.json").read_text())
                contract = json.loads((directory / "launch/launch.json").read_text())
            except json.JSONDecodeError:
                time.sleep(0.01)
                continue
            if bound["arm_sha256"] != sha(arm) or not _matches(bound["job"], arm):
                raise ValueError("process_fault_launcher_binding_changed")
            fault = object.__new__(FaultLaunch)
            fault.directory, fault.job, fault.kind = directory / "launch", bound["job"], arm["kind"]
            fault.contract, fault.host = contract, contract["host"]
            fault.origin_ns, fault.before_ids = contract["origin_ns"], contract["before_ids"]
            return fault
        time.sleep(0.01)
    raise TimeoutError("process_fault_launcher_not_reached")
