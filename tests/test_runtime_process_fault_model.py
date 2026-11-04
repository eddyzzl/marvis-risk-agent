"""Actual local LR commit then owned worker SIGKILL; no recovery claim."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from queue import Queue
import signal
import shutil
import sqlite3
import time

import psutil
import pytest

from marvis.db import ModelingRepository
from marvis.plugins import runner as runner_module
from marvis.plugins.manifest import ToolRef
from marvis.orchestrator.eval.runtime_process_faults import (
    FaultLaunch, verify_business_commit, verify_worker_crash_originals,
)
from test_modeling_pack import _runtime, _register_modeling_sample


@pytest.mark.e2e
@pytest.mark.parametrize("tamper", [None, "other_task", "parent_alias"])
def test_actual_lr_worker_crash_after_model_and_experiment_commit(tmp_path, monkeypatch, tamper):
    runner, _, registry, _, settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)
    inputs = {"dataset_id": dataset.id, "recipe": "lr", "features": ["x1", "x2"],
              "target_col": "y", "split_col": "split",
              "split_values": {"train": "train", "test": "test", "oot": "oot"}, "seed": 23}
    launches = Queue()
    native = runner_module._run_worker_process

    def injected(python, prepared_job, **kwargs):
        launch = FaultLaunch(tmp_path / "model-fault", prepared_job, "worker_crash_after_commit_once")
        launches.put(launch)
        return launch.run(native, python, **(kwargs | {"timeout": min(kwargs["timeout"], 40)}))

    monkeypatch.setattr(runner_module, "_run_worker_process", injected)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(runner.invoke, ToolRef("modeling", "train_model"), inputs, task_id=task.id)
        launch = launches.get(timeout=5)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if result.done():
                    early = result.result()
                    raise AssertionError(f"native worker exited before after-commit marker: {early.error}")
                try:
                    ready = launch.ready(timeout=.2)
                    break
                except TimeoutError:
                    continue
            else:
                raise TimeoutError("model_after_commit_marker_deadline")
            signal_record = launch.kill_worker(ready)
            failed = result.result(timeout=8)
        finally:
            if not result.done():
                (launch.directory / "release").touch()
    assert not failed.ok
    assert signal_record["signal"] == signal.SIGKILL
    assert signal_record["recovery_outcome"] == "not_evaluated"
    assert psutil.Process(os.getpid()).create_time() == launch.host["create_time"]
    assert not psutil.pid_exists(ready["worker"]["pid"])
    output = json.loads((launch.directory / "worker-output.json").read_bytes())
    committed = verify_business_commit(launch.job, output, launch.before_ids)
    assert committed["table"] == "experiments" and committed["row"]["status"] == "trained"
    assert committed["row"]["id"] == output["experiment_id"] not in launch.before_ids
    assert committed["artifact"]["id"] == output["artifact_id"]
    artifact = ModelingRepository(settings.db_path).get_model_artifact(output["artifact_id"])
    assert artifact is not None
    assert committed["files"]
    for file in committed["files"]:
        actual = settings.workspace / file["relative_path"]
        assert actual.is_file() and actual.stat().st_size > 0
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == file["sha256"]
    assert ready["phase"] == "after_business_commit_before_protocol"
    assert ready["elapsed_ns"] <= signal_record["started_elapsed_ns"]
    proof = verify_worker_crash_originals(launch.directory, launch.job)
    assert proof["fault_executed"] is True
    assert proof["recovery_outcome"] == "not_evaluated"
    if tamper is not None:
        source = settings.workspace / committed["files"][0]["relative_path"]
        if tamper == "other_task":
            foreign = settings.workspace / "tasks/another-task/modeling_artifacts" / source.name
            replacement = str(foreign)
        else:
            foreign = source.parent.parent / source.name
            replacement = "../" + source.name
        foreign.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, foreign)
        connection = sqlite3.connect(settings.db_path)
        try:
            connection.execute("UPDATE model_artifacts SET model_path=? WHERE id=?", (replacement, output["artifact_id"]))
            connection.commit()
            with pytest.raises(ValueError):
                verify_worker_crash_originals(launch.directory, launch.job)
        finally:
            connection.execute("UPDATE model_artifacts SET model_path=? WHERE id=?", (committed["artifact"]["model_path"], output["artifact_id"]))
            connection.commit()
            connection.close()
        assert verify_worker_crash_originals(launch.directory, launch.job)["fault_executed"] is True
