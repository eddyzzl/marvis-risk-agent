"""Real worker effects must remain blocked when their outcome is uncertain."""

from dataclasses import replace
import json
import sqlite3
import sys

import pytest

import marvis.db_schema as db_schema
from marvis.agent.plan_driver import PlanDriver
from marvis.db import PluginRepository, connect
from marvis.job_cancellation import JobCancelled
from marvis.orchestrator.contracts import PlanStatus, StepStatus
from marvis.plugins.manifest import ToolRef, parse_manifest
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.plans import PlanRepository
from marvis.state_machine import ConflictError
from tests.test_orch_api import _client
from tests.test_orch_completion import CountingReviewer
from tests.test_orch_executor import _executor, _plan, _step


class HostCrash(BaseException):
    pass


_PROBE_SOURCE = '''\
import os
import time

def probe(inputs, ctx):
    mode = inputs["mode"]
    if mode.startswith("write_"):
        with (ctx.workspace / "effect.txt").open("a", encoding="utf-8") as handle:
            handle.write("effect\\n")
            handle.flush()
            os.fsync(handle.fileno())
    if mode.endswith("fail"):
        raise RuntimeError("probe failed after its action")
    if mode.endswith("exit"):
        os._exit(17)
    if mode.endswith("wait"):
        time.sleep(20)
    return {"ok": True}
'''


def _runtime(tmp_path, *, effects=("write:artifact",), repo=None):
    client = _client(tmp_path) if repo is None else None
    repo = client.app.state.plan_repo if repo is None else repo
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin_root = tmp_path / "probe_plugin"
    plugin_root.mkdir()
    (plugin_root / "invocation_probe.py").write_text(_PROBE_SOURCE, encoding="utf-8")
    manifest = parse_manifest({
        "name": "invocation_probe",
        "version": "1.0.0",
        "module": "invocation_probe",
        "permissions": list(effects),
        "tools": [{
            "name": "probe",
            "summary": "Test the persisted invocation boundary",
            "entrypoint": "probe",
            "input_schema": {
                "type": "object",
                "properties": {"mode": {"type": "string"}},
                "required": ["mode"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            "determinism": "deterministic",
            "timeout_seconds": 8,
            "failure_policy": "retry",
            "side_effects": list(effects),
        }],
    })
    plugins = PluginRepository(repo.db_path)
    registry = PluginRegistry(plugins)
    registry.register(manifest)
    runner = ToolRunner(
        ToolRegistry(registry), plugins, python_executable=sys.executable,
        datasets_root=tmp_path / "datasets", workspace=workspace,
        plugin_paths=[plugin_root],
    )
    return client, repo, runner, registry, workspace / "effect.txt"


def _create_probe_plan(repo, mode, *, status=StepStatus.PENDING):
    step = _step(
        "step-1", plugin="invocation_probe", tool="probe",
        inputs={"mode": mode}, status=status,
    )
    plan = _plan(step, status=PlanStatus.RUNNING if status == StepStatus.RUNNING else PlanStatus.CONFIRMED)
    repo.create_plan(plan)
    return plan, step


def _interrupt_after_real_worker(monkeypatch, runner):
    original = runner._invoke_without_receipt

    def crash_after_worker(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.ok, result.error
        raise HostCrash()

    monkeypatch.setattr(runner, "_invoke_without_receipt", crash_after_worker)


@pytest.mark.parametrize("change", ["upgrade", "same_checksum_declaration", "disable"])
def test_real_registry_change_after_prepare_never_dispatches_worker(tmp_path, monkeypatch, change):
    _client_, _repo, runner, registry, effect = _runtime(tmp_path)
    ref = ToolRef("invocation_probe", "probe")
    manifest = registry.get(ref.plugin)
    # A checksum alone must not hide a changed permission declaration.
    manifest = replace(manifest, checksum="a" * 64)
    registry._plugins[manifest.name] = (manifest, True)
    contract = runner.prepare_invocation(ref)
    dispatched = []
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *_a, **_kw: pytest.fail("changed contract reached the worker"),
    )
    if change == "upgrade":
        registry.register(replace(manifest, version="1.1.0"))
    elif change == "disable":
        registry.set_enabled(manifest.name, False)
    else:
        # Rehydrate a changed stored declaration under the same package identity.
        changed_tool = replace(manifest.tools[0], side_effects=("read:input",))
        changed = replace(manifest, tools=(changed_tool,), permissions=("read:input",))
        registry._repo.upsert_plugin(changed, enabled=True)
        registry.load_from_db()
        assert runner.prepare_invocation(ref)["manifest_hash"] == contract["manifest_hash"]
        assert runner.prepare_invocation(ref)["manifest_declaration_hash"] != contract["manifest_declaration_hash"]
    if change == "disable":
        from marvis.plugins.errors import PluginNotFoundError

        with pytest.raises(PluginNotFoundError):
            runner.invoke(ref, {"mode": "write_return"}, task_id="task-1", expected_invocation=contract, on_dispatch=lambda: dispatched.append(True))
    else:
        result = runner.invoke(ref, {"mode": "write_return"}, task_id="task-1", expected_invocation=contract, on_dispatch=lambda: dispatched.append(True))
        assert not result.ok and result.error_kind == "invocation_changed"
    assert dispatched == []
    assert not effect.exists()


@pytest.mark.parametrize("failure", ["worker_failure", "worker_exit", "cancel", "host_crash"])
def test_real_write_without_output_cannot_be_retried_after_restart(tmp_path, monkeypatch, failure):
    client, repo, runner, _registry, effect = _runtime(tmp_path)
    mode = {
        "worker_failure": "write_fail", "worker_exit": "write_exit",
        "cancel": "write_wait", "host_crash": "write_return",
    }[failure]
    _create_probe_plan(repo, mode)
    executor = _executor(repo, runner, reviewer=CountingReviewer())

    def cancel_after_effect():
        if effect.exists():
            raise JobCancelled("cancel after the worker persisted its effect")

    if failure == "host_crash":
        with monkeypatch.context() as patch:
            _interrupt_after_real_worker(patch, runner)
            with pytest.raises(HostCrash):
                executor.run("plan-1")
        assert repo.load_plan("plan-1").steps[0].status == StepStatus.RUNNING
    else:
        executor.run("plan-1", cancellation_check=cancel_after_effect if failure == "cancel" else None)
    assert effect.read_text(encoding="utf-8") == "effect\n"
    run = repo.list_step_runs("step-1")[0]
    assert run["dispatch_started_at"]
    assert run["invocation_contract"]["side_effects"] == ["write:artifact"]
    assert run["output_ref"] is None

    restored = PlanRepository(repo.db_path)
    _executor(restored, runner, reviewer=CountingReviewer()).run("plan-1")
    assert restored.list_step_runs("step-1")[0]["error_kind"] == "unknown_effect"
    assert "explicit reconciliation required" in restored.load_plan("plan-1").steps[0].error

    class NeverRun:
        def run(self, _plan_id):
            pytest.fail("uncertain write reached a second execution")

    driver = PlanDriver(restored, NeverRun())
    with pytest.raises(ConflictError, match="尚未核对"):
        driver.retry_failed_step("plan-1", "step-1", inputs={"mode": "write_return"})
    client.app.state.plan_repo = restored
    response = client.post("/api/plans/plan-1/steps/step-1/retry", json={"inputs": {"mode": "write_return"}})
    assert response.status_code == 409, response.text
    assert client.app.state.plan_executor.calls == []
    assert len(restored.list_step_runs("step-1")) == 1
    assert effect.read_text(encoding="utf-8") == "effect\n"


@pytest.mark.parametrize("contract_kind", ["legacy_null", "malformed", "prepared_write", "dispatched_read"])
def test_recovery_uses_frozen_dispatch_evidence_not_current_registry(tmp_path, monkeypatch, contract_kind):
    effects = ("read:input",) if contract_kind == "dispatched_read" else ("write:artifact",)
    _client_, repo, runner, registry, effect = _runtime(tmp_path, effects=effects)
    if contract_kind == "dispatched_read":
        _create_probe_plan(repo, "read_return")
        with monkeypatch.context() as patch:
            _interrupt_after_real_worker(patch, runner)
            with pytest.raises(HostCrash):
                _executor(repo, runner, reviewer=CountingReviewer()).run("plan-1")
    else:
        plan, step = _create_probe_plan(repo, "write_return", status=StepStatus.RUNNING)
        contract = runner.prepare_invocation(step.tool_ref)
        if contract_kind == "legacy_null":
            contract = None
        elif contract_kind == "malformed":
            contract = {"schema_version": "unknown"}
        repo.start_step_run(plan_id=plan.id, step_id=step.id, tool_ref=step.tool_ref.label(), inputs=step.inputs, invocation_contract=contract)
    # Upgrade now; recovery must never infer the old invocation from this read-only declaration.
    manifest = registry.get("invocation_probe")
    registry.register(replace(manifest, version="2.0.0", tools=(replace(manifest.tools[0], side_effects=("read:input",)),), permissions=("read:input",)))
    restored = PlanRepository(repo.db_path)
    result = _executor(restored, runner, reviewer=CountingReviewer()).run("plan-1")
    assert result.status == PlanStatus.FAILED
    safe = contract_kind in {"prepared_write", "dispatched_read"}
    run = restored.list_step_runs("step-1")[0]
    assert run["error_kind"] == ("ServerRestart" if safe else "unknown_effect")
    if safe:
        assert restored.retry_failed_step("plan-1", "step-1") == ["step-1"]
    else:
        with pytest.raises(ConflictError, match="尚未核对"):
            restored.retry_failed_step("plan-1", "step-1")
    assert not effect.exists()


def test_invocation_contract_and_dispatch_marker_are_immutable_in_database(tmp_path):
    _client_, repo, runner, _registry, _effect = _runtime(tmp_path)
    plan, step = _create_probe_plan(repo, "write_return", status=StepStatus.RUNNING)
    contract = runner.prepare_invocation(step.tool_ref)
    run_id = repo.start_step_run(plan_id=plan.id, step_id=step.id, tool_ref=step.tool_ref.label(), inputs=step.inputs, invocation_contract=contract)
    with pytest.raises(sqlite3.IntegrityError, match="invocation contract is immutable"):
        with connect(repo.db_path) as conn:
            conn.execute("UPDATE plan_step_runs SET invocation_contract_json = ? WHERE id = ?", (json.dumps({**contract, "side_effects": []}), run_id))
    repo.mark_step_run_dispatched(run_id)
    dispatch_time = repo.list_step_runs(step.id)[0]["dispatch_started_at"]
    repo.mark_step_run_dispatched(run_id)
    with pytest.raises(sqlite3.IntegrityError, match="dispatch marker is immutable"):
        with connect(repo.db_path) as conn:
            conn.execute("UPDATE plan_step_runs SET dispatch_started_at = NULL WHERE id = ?", (run_id,))
    repo.finish_step_run(run_id, status="interrupted", error_kind="unknown_effect")
    after = repo.list_step_runs(step.id)[0]
    assert after["invocation_contract"] == contract
    assert after["dispatch_started_at"] == dispatch_time


def test_legacy_null_contract_cannot_be_backfilled_with_current_manifest(tmp_path):
    _client_, repo, runner, _registry, _effect = _runtime(tmp_path)
    plan, step = _create_probe_plan(repo, "write_return", status=StepStatus.RUNNING)
    run_id = repo.start_step_run(plan_id=plan.id, step_id=step.id, tool_ref=step.tool_ref.label(), inputs=step.inputs)
    with pytest.raises(sqlite3.IntegrityError, match="invocation contract is immutable"):
        with connect(repo.db_path) as conn:
            conn.execute("UPDATE plan_step_runs SET invocation_contract_json = ? WHERE id = ?", (json.dumps(runner.prepare_invocation(step.tool_ref)), run_id))
    assert not repo.step_run_retry_safe(run_id)


def test_dispatch_checkpoint_failure_stops_before_real_worker(tmp_path, monkeypatch):
    _client_, repo, runner, _registry, effect = _runtime(tmp_path)
    _create_probe_plan(repo, "write_return")

    def unavailable_dispatch_checkpoint(_run_id):
        raise OSError("cannot persist dispatch checkpoint")

    monkeypatch.setattr(repo, "mark_step_run_dispatched", unavailable_dispatch_checkpoint)
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *_a, **_kw: pytest.fail("worker ran without a durable dispatch checkpoint"),
    )
    result = _executor(repo, runner, reviewer=CountingReviewer()).run("plan-1")
    assert result.status == PlanStatus.FAILED
    run = repo.list_step_runs("step-1")[0]
    assert run["dispatch_started_at"] is None
    assert repo.step_run_retry_safe(run["id"])
    assert not effect.exists()


def test_registry_change_after_real_write_cannot_reclassify_failure_as_safe(tmp_path, monkeypatch):
    _client_, repo, runner, registry, effect = _runtime(tmp_path)
    _create_probe_plan(repo, "write_fail")
    original = runner._invoke_without_receipt

    def change_declaration_after_worker(*args, **kwargs):
        result = original(*args, **kwargs)
        assert not result.ok
        assert effect.read_text(encoding="utf-8") == "effect\n"
        manifest = registry.get("invocation_probe")
        registry.register(replace(
            manifest, version="2.0.0", permissions=("read:input",),
            tools=(replace(manifest.tools[0], side_effects=("read:input",)),),
        ))
        return result

    monkeypatch.setattr(runner, "_invoke_without_receipt", change_declaration_after_worker)
    result = _executor(repo, runner, reviewer=CountingReviewer()).run("plan-1")
    assert result.status == PlanStatus.FAILED
    run = repo.list_step_runs("step-1")[0]
    assert run["invocation_contract"]["side_effects"] == ["write:artifact"]
    assert run["invocation_contract"]["tool_version"] == "1.0.0"
    assert run["error_kind"] == "unknown_effect"
    assert not repo.step_run_retry_safe(run["id"])
    with pytest.raises(ConflictError, match="尚未核对"):
        repo.retry_failed_step("plan-1", "step-1")
    assert effect.read_text(encoding="utf-8") == "effect\n"


@pytest.mark.parametrize("old_status", ["running", "failed", "interrupted"])
def test_migration_36_to_37_does_not_invent_historical_invocation_contract(tmp_path, monkeypatch, old_status):
    with monkeypatch.context() as patch:
        patch.setattr(db_schema, "_MIGRATIONS", [item for item in db_schema._MIGRATIONS if item[0] <= 36])
        # Install the real old schema. Modern TaskRepository writes later
        # columns and cannot be used to manufacture a schema-36 task.
        db_path = tmp_path / "app.sqlite"
        db_schema.init_db(db_path)
        with connect(db_path) as conn:
            conn.execute(
                """INSERT INTO tasks
                   (id,model_name,model_version,validator,source_dir,status,status_message,created_at,updated_at)
                   VALUES ('task-1','test','v1','qa',?,'draft','','2026-09-20','2026-09-20')""",
                (str(tmp_path),),
            )
        _client_, repo, runner, _registry, effect = _runtime(
            tmp_path, effects=("read:input",), repo=PlanRepository(db_path)
        )
        _plan_, step = _create_probe_plan(repo, "read_return", status=StepStatus.RUNNING)
        with connect(repo.db_path) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 36
            assert "invocation_contract_json" not in {
                row["name"] for row in conn.execute("PRAGMA table_info(plan_step_runs)")
            }
            conn.execute(
                """INSERT INTO plan_step_runs
                   (id, plan_id, step_id, attempt, tool_ref, status, input_json,
                    error, error_kind, started_at)
                   VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
                ("historical-run", "plan-1", step.id, step.tool_ref.label(), old_status,
                 json.dumps(step.inputs), "legacy display error", "ServerRestart",
                 "2026-09-20T00:00:00+00:00"),
            )

    db_schema.init_db(repo.db_path)
    # The installed declaration is read-only, but says nothing about this old run.
    assert runner.prepare_invocation(step.tool_ref)["side_effects"] == ["read:input"]
    with connect(repo.db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db_schema.SCHEMA_VERSION
        row = conn.execute("SELECT * FROM plan_step_runs WHERE id = 'historical-run'").fetchone()
        assert row["invocation_contract_json"] is None
        assert row["dispatch_started_at"] is None
        assert row["status"] == old_status
        assert row["error"] == "legacy display error"
    assert not repo.step_run_retry_safe("historical-run")
    assert repo.unreconciled_step_ids([step.id]) == [step.id]
    with pytest.raises(ConflictError, match="尚未核对"):
        repo.reset_step(step.id)
    assert not effect.exists()
