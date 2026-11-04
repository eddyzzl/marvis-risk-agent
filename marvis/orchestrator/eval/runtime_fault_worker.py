"""Evaluation-only worker entrypoint, selected explicitly by FaultLaunch argv."""
import argparse
import json
import os
from pathlib import Path
import time

from .runtime_process_faults import binding, process_identity, sha, verify_business_commit, write_original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.control_directory.resolve(strict=True)
    contract = json.loads((directory / "launch.json").read_text())
    if contract["host"] != process_identity(os.getppid()):
        raise ValueError("fault_launcher_parent_changed")
    from marvis.plugins import subprocess_worker as native

    original = native._run_tool

    def execute(job):
        if not job.get("builtin") or binding(job) != contract["binding"]:
            raise ValueError("fault_launch_job_binding_changed")
        marker = {"schema": "marvis.process-fault-ready.v1", "kind": contract["kind"],
                  "binding": binding(job), "launch_sha256": sha(contract),
                  "worker": process_identity(os.getpid()),
                  "phase": "before_tool", "elapsed_ns": time.monotonic_ns() - contract["origin_ns"]}
        if contract["kind"] == "worker_crash_after_commit_once":
            output = original(job)
            proof = verify_business_commit(job, output, contract["before_ids"])
            write_original(directory / "worker-output.json", output)
            write_original(directory / "commit.json", proof)
            marker.update(phase="after_business_commit_before_protocol", commit_sha256=sha(proof),
                          output_sha256=sha(output), elapsed_ns=time.monotonic_ns() - contract["origin_ns"])
        write_original(directory / "ready.json", marker)
        # Real wall time is spent in this exact worker. Its unchanged supervisor
        # performs timeout/cancellation. A source-owned controller may instead
        # kill this worker or restart the actual service process. No fake return.
        while not (directory / "release").exists():
            time.sleep(0.01)
        if marker["phase"] == "before_tool":
            return original(job)
        return output

    native._run_tool = execute
    native.worker_main()


if __name__ == "__main__":
    main()
