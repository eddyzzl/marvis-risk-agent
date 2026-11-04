"""Actual native data worker interruption, with committed data and negative bindings."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from queue import Queue
import signal
import sys
import time

import pandas as pd
import psutil
import pytest

from marvis.plugins import runner as runner_module
from marvis.plugins.manifest import ToolRef
from marvis.plugins.contracts import PROTOCOL_VERSION
from marvis.job_cancellation import JobCancellationToken, JobCancelled
from marvis.orchestrator.eval.runtime_process_faults import (
    FaultLaunch, exact_process, verify_business_commit, verify_worker_crash_originals,
)
from test_data_ops_pack import _runtime, _register_csv


def _job(tmp_path):
    runner, registry, repository = _runtime(tmp_path)
    source = _register_csv(registry, tmp_path, "raw", pd.DataFrame({"name": [" A ", " B "]}), role="sample")
    job = {"protocol_version": PROTOCOL_VERSION, "builtin": True,
           "module": "marvis.packs.data_ops.tools", "entrypoint": "tool_clean_format",
           "task_id": "task-1", "workspace": str(tmp_path / "workspace"),
           "datasets_root": str(tmp_path / "workspace/datasets"),
           "inputs": {"dataset_id": source.id, "ops": [{"col": "name", "op": "strip"}]},
           "side_effects": ["workspace:write"], "seed": 1}
    return job, runner, registry, repository


@pytest.mark.e2e
def test_actual_worker_is_killed_after_native_dataset_commit_without_claiming_recovery(tmp_path, monkeypatch):
    job, runner, registry, _ = _job(tmp_path)
    launches = Queue()
    original = runner_module._run_worker_process

    def injected(python, prepared_job, **kwargs):
        launch = FaultLaunch(tmp_path / "fault", prepared_job, "worker_crash_after_commit_once")
        launches.put(launch)
        return launch.run(original, python, **kwargs)

    monkeypatch.setattr(runner_module, "_run_worker_process", injected)
    with ThreadPoolExecutor() as pool:
        result = pool.submit(runner.invoke, ToolRef("data_ops", "clean_format"), job["inputs"], task_id="task-1")
        launch = launches.get(timeout=5)
        ready = launch.ready()
        signal_record = launch.kill_worker(ready)
        failed = result.result(timeout=10)
    assert not failed.ok
    assert signal_record["signal"] == signal.SIGKILL
    assert signal_record["recovery_outcome"] == "not_evaluated"
    assert psutil.Process(os.getpid()).create_time() == launch.host["create_time"]
    assert not psutil.pid_exists(ready["worker"]["pid"])
    output = json.loads((launch.directory / "worker-output.json").read_text())
    committed = verify_business_commit(launch.job, output, launch.before_ids)
    actual = pd.read_parquet(registry.resolve_path(output["dataset_id"]))
    assert actual.to_dict("list") == {"name": ["A", "B"]}
    assert committed["row"]["id"] == output["dataset_id"]
    assert ready["phase"] == "after_business_commit_before_protocol"
    assert ready["elapsed_ns"] <= signal_record["started_elapsed_ns"]
    archived = verify_worker_crash_originals(launch.directory, launch.job)
    assert archived["fault_executed"] is True
    assert archived["recovery_outcome"] == "not_evaluated"
    terminal_path = launch.directory / "terminal.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["returncode"] = 0
    terminal_path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="binding_invalid"):
        verify_worker_crash_originals(launch.directory, launch.job)


@pytest.mark.e2e
@pytest.mark.parametrize("kind", ["tool_timeout_once", "cancel_then_resume_once"])
def test_real_supervisor_timeout_or_cancellation_reaps_paused_worker_before_native_write(tmp_path, kind):
    job, _, _, _ = _job(tmp_path)
    launch = FaultLaunch(tmp_path / "fault", job, kind)
    token = JobCancellationToken("actual-local-test-job")
    started = time.monotonic()
    with ThreadPoolExecutor() as pool:
        future = pool.submit(launch.run, runner_module._run_worker_process, sys.executable,
                             timeout=2 if kind == "tool_timeout_once" else 20,
                             cancellation_check=token.raise_if_cancelled)
        ready = launch.ready()
        if kind == "cancel_then_resume_once":
            token.cancel()
            with pytest.raises(JobCancelled):
                future.result(timeout=10)
        else:
            with pytest.raises(runner_module.WorkerTimeoutExpired) as timed_out:
                future.result(timeout=10)
            assert timed_out.value.timeout == 2
            assert time.monotonic() - started >= 2
    assert ready["phase"] == "before_tool"
    assert not psutil.pid_exists(ready["worker"]["pid"])
    assert not (launch.directory / "worker-output.json").exists()
    # A fresh normal worker actually executes the operation after the cancelled
    # carrier has exited. API plan resumption is a separate integration boundary.
    resumed = runner_module._run_worker(sys.executable, job, timeout=20)
    protocol = runner_module._parse_worker_result(resumed.stdout)
    assert protocol["ok"] is True
    assert protocol["output"]["dataset_id"] != job["inputs"]["dataset_id"]


def test_after_commit_requires_supported_producer_and_fresh_same_task_committed_content(tmp_path):
    job, _, _, _ = _job(tmp_path)
    with pytest.raises(ValueError, match="producer_unsupported"):
        FaultLaunch(tmp_path / "unsupported", job | {"entrypoint": "tool_profile_dataset"},
                    "worker_crash_after_commit_once")
    with pytest.raises(ValueError, match="fresh_dataset_commit_missing"):
        verify_business_commit(job, {"dataset_id": job["inputs"]["dataset_id"]}, [job["inputs"]["dataset_id"]])
    with pytest.raises(ValueError, match="fresh_dataset_commit_missing"):
        verify_business_commit(job | {"task_id": "other"}, {"dataset_id": job["inputs"]["dataset_id"]}, [])
    with pytest.raises(ValueError, match="identity_changed"):
        exact_process({"pid": os.getpid(), "create_time": 0})


@pytest.mark.e2e
@pytest.mark.parametrize("tamper", ["binding", "commit", "output", "pid"])
def test_tampered_worker_originals_never_authorize_a_signal(tmp_path, tamper):
    job, _, _, _ = _job(tmp_path)
    launch = FaultLaunch(tmp_path / "fault", job, "worker_crash_after_commit_once")
    with ThreadPoolExecutor() as pool:
        future = pool.submit(launch.run, runner_module._run_worker_process, sys.executable, timeout=20)
        ready = launch.ready()
        try:
            if tamper == "output":
                path = launch.directory / "worker-output.json"
                payload = json.loads(path.read_text())
                payload["dataset_id"] = job["inputs"]["dataset_id"]
            else:
                path = launch.directory / "ready.json"
                payload = dict(ready)
                if tamper == "binding":
                    payload["binding"] = payload["binding"] | {"task_id": "other"}
                elif tamper == "commit":
                    payload["commit_sha256"] = "0" * 64
                else:
                    payload["worker"] = launch.host
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError):
                launch.kill_worker(ready)
            assert not (launch.directory / "signal.json").exists()
            assert exact_process(ready["worker"]).is_running()
        finally:
            (launch.directory / "release").touch()
        assert future.result(timeout=10).returncode == 0
