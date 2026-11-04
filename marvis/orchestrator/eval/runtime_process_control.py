"""Source-owned application lifecycle for a fixed recovery experiment.

No PID, command, environment or workspace can be supplied by a case. Every
restart retains a separate process epoch and uses the original launch closure.
"""
from __future__ import annotations

import time

import httpx
import psutil

from .runtime_contracts import digest


class OwnedProcessControl:
    def __init__(self, process, *, workspace, root, origin_ns, deadline, launch, client):
        self.process = process
        self.workspace = workspace.resolve()
        self.root = root
        self.origin_ns = origin_ns
        self.deadline = deadline
        self.launch = launch
        self.client = client
        self.epoch = 0
        self._identity = self._snapshot()
        self.epochs = [self._epoch_record()]
        self.restarts = []

    def _snapshot(self):
        return {"epoch": self.epoch, "pid": self.process.pid,
                "create_time": psutil.Process(self.process.pid).create_time(),
                "workspace_sha256": digest(str(self.workspace))}

    def snapshot(self):
        observed = self._snapshot()
        if observed != self._identity:
            raise ValueError("owned_process_identity_changed")
        return dict(observed)

    def _epoch_record(self):
        suffix = "" if self.epoch == 0 else f"-epoch-{self.epoch}"
        return {"ordinal": self.epoch, "identity": dict(self._identity),
                "process_observations_path": str(self.root / f"process-observations{suffix}.jsonl"),
                "process_timing_path": str(self.root / f"process-timing{suffix}.jsonl")}

    def restart_same_workspace(self):
        before = self.snapshot()
        started = time.monotonic_ns() - self.origin_ns
        parent = psutil.Process(before["pid"])
        children = parent.children(recursive=True)
        # Deliberately crash the owned service; worker processes are separately
        # retired, including their own groups. PID reuse is guarded by psutil.
        parent.kill()
        self.process.wait(timeout=5)
        old_returncode = self.process.returncode
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=5)
        if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
            raise ValueError("owned_worker_retirement_failed")
        self.epoch += 1
        self.process = self.launch(self.epoch)
        self._identity = self._snapshot()
        self.epochs.append(self._epoch_record())
        after = self.snapshot()
        if (after["pid"] == before["pid"] or after["create_time"] == before["create_time"]
                or after["workspace_sha256"] != before["workspace_sha256"]):
            raise ValueError("restart_identity_not_distinct")
        while time.monotonic() < self.deadline:
            if self.process.poll() is not None:
                raise ValueError("restart_application_exited")
            try:
                if self.client.get("/api/branding", timeout=.5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.05)
        else:
            raise ValueError("restart_application_ready_timeout")
        result = {"before": before, "after": after, "signal": "SIGKILL", "old_returncode": old_returncode,
                  "started_elapsed_ns": started, "finished_elapsed_ns": time.monotonic_ns() - self.origin_ns,
                  "epochs": [dict(item) for item in self.epochs]}
        self.restarts.append(result)
        return result
