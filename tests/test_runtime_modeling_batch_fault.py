"""Actual standard modeling HTTP train_models commit, interruption and originals."""
from contextlib import closing
import json
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_safety import revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_contracts import digest
from tests.test_runtime_modeling_portfolio_recovery import _run_case
from tests.test_runtime_process_fault_portfolio import _repin


@pytest.fixture(scope="module")
def batch_originals(tmp_path_factory):
    return _run_case(tmp_path_factory.mktemp("modeling-batch-after-commit"), "modeling",
                     "worker_crash_after_commit_once", "modeling.train_models")


def test_standard_modeling_gate_commits_all_listed_models_before_actual_worker_kill(batch_originals):
    archive, sha, binding = batch_originals
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result
    assert result["values"]["fault_executed"] is True
    assert result["values"]["recovery_completed"] is False
    directory = archive / "workspace/.runtime-eval-fault/launch"
    output = json.loads((directory / "worker-output.json").read_bytes())
    commit = json.loads((directory / "commit.json").read_bytes())
    assert commit["table"] == "experiments_batch"
    assert [row["id"] for row in commit["rows"]] == output["experiment_ids"]
    assert len(commit["artifacts"]) == len(commit["audits"]) == len(output["experiment_ids"])
    for item in commit["files"]:
        assert digest((archive / "workspace" / item["relative_path"]).read_bytes()) == item["sha256"]
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    assert fault["approval_tool"] == "modeling.configure_tuning"
    assert fault["tool"] == "modeling.train_models"
    assert fault["approval_step_id"] != fault["step_id"]
    assert json.loads((directory / "terminal.json").read_bytes())["returncode"] == -9


@pytest.mark.parametrize("damage", ["missing_artifact", "foreign_path", "model_bytes", "omitted_experiment", "missing_audit", "dependency"])
def test_batch_commit_repin_cannot_bless_missing_or_cross_task_models(batch_originals, tmp_path, damage):
    original, _, binding = batch_originals
    archive = tmp_path / "archive"
    shutil.copytree(original, archive)
    directory = archive / "workspace/.runtime-eval-fault/launch"
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    commit = json.loads((directory / "commit.json").read_bytes())
    output = json.loads((directory / "worker-output.json").read_bytes())
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        if damage == "missing_artifact":
            conn.execute("DELETE FROM model_artifacts WHERE id=?", (commit["artifacts"][-1]["id"],))
        elif damage == "foreign_path":
            artifact = commit["artifacts"][-1]
            artifact["model_path"] = str(archive / "workspace/tasks/another-task/modeling_artifacts/model.joblib")
            conn.execute("UPDATE model_artifacts SET model_path=? WHERE id=?", (artifact["model_path"], artifact["id"]))
        elif damage == "model_bytes":
            (archive / "workspace" / commit["files"][-1]["relative_path"]).write_bytes(b"changed model file")
        elif damage == "omitted_experiment":
            output["experiment_ids"] = output["experiment_ids"][:-1]
            output["experiments"] = output["experiments"][:-1]
            commit["rows"] = commit["rows"][:-1]
            commit["artifacts"] = commit["artifacts"][:-1]
            commit["audits"] = commit["audits"][:-1]
        elif damage == "missing_audit":
            conn.execute("DELETE FROM audit WHERE id=?", (commit["audits"][-1]["id"],))
        else:
            conn.execute("UPDATE plan_steps SET depends_on_json='[]' WHERE id=?", (fault["step_id"],))
        conn.commit()
    (directory / "worker-output.json").write_text(json.dumps(output))
    (directory / "commit.json").write_text(json.dumps(commit))
    fault["ready"].update(commit_sha256=digest(commit), output_sha256=digest(output))
    (directory / "ready.json").write_text(json.dumps(fault["ready"]))
    signal = json.loads((directory / "signal.json").read_bytes())
    signal["commit_sha256"] = digest(commit)
    (directory / "signal.json").write_text(json.dumps(signal))
    (archive / "private.json").write_text(json.dumps(private))
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=_repin(archive), frozen_binding=binding)
    assert result["status"] == "mismatch", result


def test_actual_two_model_batch_requires_second_experiment_and_its_file(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    from types import SimpleNamespace
    from marvis.plugins import runner as runner_module
    from marvis.plugins.manifest import ToolRef
    from marvis.orchestrator.eval.runtime_archive_process_faults import _commit
    from marvis.orchestrator.eval.runtime_process_faults import FaultLaunch, verify_business_commit
    from test_modeling_pack import _runtime, _register_modeling_sample

    runner, _, registry, _, settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)
    inputs = {"dataset_id": dataset.id, "recipes": ["lr", "lgb"], "features": ["x1", "x2"],
              "target_col": "y", "split_col": "split", "split_values": {"train": "train", "test": "test", "oot": "oot"},
              "seed": 23, "params": {"lr": {}, "lgb": {"n_estimators": 8, "num_leaves": 4}}}
    launches = Queue()
    original = runner_module._run_worker_process

    def injected(python, job, **kwargs):
        launch = FaultLaunch(tmp_path / "batch-fault", job, "worker_crash_after_commit_once")
        launches.put(launch)
        return launch.run(original, python, **(kwargs | {"timeout": 60}))

    monkeypatch.setattr(runner_module, "_run_worker_process", injected)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(runner.invoke, ToolRef("modeling", "train_models"), inputs, task_id=task.id)
        launch = launches.get(timeout=5)
        try:
            ready = launch.ready(timeout=40)
            signal = launch.kill_worker(ready)
            assert not result.result(timeout=10).ok
        finally:
            if not result.done():
                (launch.directory / "release").touch()
    def read(name):
        return json.loads((launch.directory / name).read_bytes())
    output, commit = read("worker-output.json"), read("commit.json")
    assert len(output["experiment_ids"]) == len(commit["rows"]) == 2
    assert len(commit["artifacts"]) == len(commit["audits"]) == 2
    assert verify_business_commit(launch.job, output, launch.before_ids) == commit
    with closing(sqlite3.connect(settings.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        ctx = SimpleNamespace(archive=tmp_path, conn=conn, binding=SimpleNamespace(task_id=task.id))
        assert _commit(ctx, launch.job, launch.contract, ready, signal, read) == digest(commit)
        omitted = output | {"experiment_ids": output["experiment_ids"][:1], "experiments": output["experiments"][:1]}
        with pytest.raises(ValueError):
            verify_business_commit(launch.job, omitted, launch.before_ids)
        claimed = commit | {"rows": commit["rows"][:1], "artifacts": commit["artifacts"][:1], "audits": commit["audits"][:1]}
        fake_ready = ready | {"output_sha256": digest(omitted), "commit_sha256": digest(claimed)}
        fake_signal = signal | {"commit_sha256": digest(claimed)}
        with pytest.raises(ValueError):
            _commit(ctx, launch.job, launch.contract, fake_ready, fake_signal,
                    lambda name: omitted if name == "worker-output.json" else claimed)
        second = next(item for item in commit["files"] if commit["artifacts"][1]["model_path"] in item["relative_path"])
        path = settings.workspace / second["relative_path"]
        original_bytes = path.read_bytes()
        try:
            path.write_bytes(b"changed second model")
            with pytest.raises(ValueError):
                _commit(ctx, launch.job, launch.contract, ready, signal, read)
        finally:
            path.write_bytes(original_bytes)
