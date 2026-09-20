from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest

from marvis.db import PluginRepository, init_db
from marvis.plugins.hooks import HookDispatcher
from marvis.plugins.manifest import HookSpec, ToolRef, parse_manifest
from marvis.plugins.registry import PluginRegistry
from marvis.plugins.runner import ToolResult
from marvis.repositories.hook_deliveries import HookDeliveryRepository
from marvis.state_machine import ConflictError


class FakeRunner:
    def __init__(self):
        self.calls = []

    def invoke(self, ref, inputs, *, task_id, seed=None):
        self.calls.append((ref, inputs, task_id, seed))
        if inputs.get("fail"):
            return ToolResult(
                ok=False,
                output=None,
                error="failed",
                error_kind="execution",
                duration_ms=1,
            )
        return ToolResult(
            ok=True,
            output={"ok": True},
            error=None,
            error_kind=None,
            duration_ms=1,
        )


def _manifest(name: str = "hook_pack"):
    return parse_manifest(
        {
            "name": name,
            "version": "0.1.0",
            "display_name": "Hook Pack",
            "description": "Hook test pack",
            "module": f"{name}.tools",
            "tools": [
                {
                    "name": "on_task_created",
                    "summary": "Handle task creation",
                    "input_schema": {"type": "object", "properties": {}, "required": []},
                    "output_schema": {"type": "object", "properties": {}, "required": []},
                    "determinism": "deterministic",
                    "timeout_seconds": 10,
                    "failure_policy": "fail",
                    "entrypoint": "tool_on_task_created",
                }
            ],
            "hooks": [{"event": "task.created", "tool": "on_task_created"}],
            "permissions": [],
        },
        builtin=True,
    )


def test_hook_dispatcher_invokes_tools_registered_for_event(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    registry.register(_manifest(), enabled=True)
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner, repo)
    dispatcher.rebuild_index()

    results = dispatcher.dispatch("task.created", {"task_id": "t1"}, task_id="t1")

    assert len(results) == 1
    assert results[0].ok is True
    assert runner.calls == [
        (
            ToolRef("hook_pack", "on_task_created", "0.1.0"),
            {"task_id": "t1"},
            "t1",
            None,
        )
    ]
    audits = repo.list_audit(kind="hook.dispatch")
    started = repo.list_audit(kind="hook.dispatch.started")
    assert len(started) == 1
    assert started[0]["target_ref"] == "hook_pack.on_task_created@0.1.0"
    assert len(audits) == 1
    assert audits[0]["target_ref"] == "hook_pack.on_task_created@0.1.0"
    assert audits[0]["outcome"] == "succeeded"
    assert audits[0]["detail"]["event"] == "task.created"


def test_hook_dispatcher_invokes_builtin_listeners_without_plugin_results(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner)
    calls = []
    dispatcher.register_listener(
        "validation.completed",
        lambda event, payload: calls.append((event, payload)),
    )

    results = dispatcher.dispatch(
        "validation.completed",
        {"task_id": "t1", "status": "succeeded"},
        task_id="t1",
    )

    assert results == []
    assert runner.calls == []
    assert calls == [
        ("validation.completed", {"task_id": "t1", "status": "succeeded"})
    ]
    assert dispatcher.listener_count("validation.completed") == 1


def test_hook_dispatcher_isolates_failed_builtin_listener(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    registry.register(_manifest(), enabled=True)
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner, repo)
    dispatcher.rebuild_index()

    def broken_listener(_event, _payload):
        raise RuntimeError("boom")

    dispatcher.register_listener("task.created", broken_listener)

    results = dispatcher.dispatch("task.created", {"task_id": "t1"}, task_id="t1")

    assert len(results) == 1
    assert results[0].ok is True
    assert runner.calls == [
        (ToolRef("hook_pack", "on_task_created", "0.1.0"), {"task_id": "t1"}, "t1", None)
    ]


def test_hook_dispatcher_skips_disabled_plugins_after_rebuild(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(), enabled=True)
    registry.set_enabled("hook_pack", False)
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner)
    dispatcher.rebuild_index()

    assert dispatcher.dispatch("task.created", {"task_id": "t1"}, task_id="t1") == []
    assert runner.calls == []


def test_hook_dispatcher_isolates_failed_hook_results(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    registry.register(_manifest(), enabled=True)
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner, repo)
    dispatcher.rebuild_index()

    results = dispatcher.dispatch("task.created", {"fail": True}, task_id="t1")

    assert len(results) == 1
    assert results[0].ok is False
    assert runner.calls[0][0] == ToolRef("hook_pack", "on_task_created", "0.1.0")
    audits = repo.list_audit(kind="hook.dispatch")
    assert audits[0]["outcome"] == "failed"
    assert audits[0]["detail"]["error_kind"] == "execution"


def test_hook_dispatcher_does_not_invoke_plugin_when_started_audit_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    registry.register(_manifest(), enabled=True)
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner, repo)
    dispatcher.rebuild_index()
    original_write_audit = repo.write_audit

    def fail_started_audit(**kwargs):
        if kwargs.get("kind") == "hook.dispatch.started":
            raise RuntimeError("audit down")
        return original_write_audit(**kwargs)

    monkeypatch.setattr(repo, "write_audit", fail_started_audit)

    results = dispatcher.dispatch("task.created", {"task_id": "t1"}, task_id="t1")

    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error_kind == "audit"
    assert results[0].error_detail["audit_phase"] == "start"
    assert runner.calls == []
    assert repo.list_audit(kind="hook.dispatch") == []


def test_hook_dispatcher_returns_audit_failure_when_finish_audit_fails(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    registry.register(_manifest(), enabled=True)
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner, repo)
    dispatcher.rebuild_index()
    original_write_audit = repo.write_audit

    def fail_finish_audit(**kwargs):
        if kwargs.get("kind") == "hook.dispatch":
            raise RuntimeError("audit down")
        return original_write_audit(**kwargs)

    monkeypatch.setattr(repo, "write_audit", fail_finish_audit)

    results = dispatcher.dispatch("task.created", {"task_id": "t1"}, task_id="t1")

    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error_kind == "audit"
    assert results[0].error_detail["audit_phase"] == "finish"
    assert results[0].error_detail["result_ok"] is True
    assert len(repo.list_audit(kind="hook.dispatch.started")) == 1
    assert repo.list_audit(kind="hook.dispatch") == []


def test_hook_dispatcher_audits_builtin_listener_when_repo_is_available(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner, repo)
    calls = []
    dispatcher.register_listener("validation.completed", lambda event, payload: calls.append((event, payload)))

    results = dispatcher.dispatch("validation.completed", {"task_id": "t1"}, task_id="t1")

    assert results == []
    assert calls == [("validation.completed", {"task_id": "t1"})]
    assert len(repo.list_audit(kind="hook.listener.started")) == 1
    listener_audit = repo.list_audit(kind="hook.listener")[0]
    assert listener_audit["outcome"] == "succeeded"
    assert listener_audit["detail"]["event"] == "validation.completed"


def test_hook_dispatcher_unknown_event_is_noop(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(), enabled=True)
    runner = FakeRunner()
    dispatcher = HookDispatcher(registry, runner)
    dispatcher.rebuild_index()

    assert dispatcher.dispatch("validation.completed", {}, task_id="t1") == []
    assert runner.calls == []
def _durable_dispatcher(tmp_path, *, required=True, runner=None):
    db_path = tmp_path / 'hooks.sqlite'
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    registry.register(replace(_manifest(), hooks=(HookSpec('step.completed', 'on_task_created', required),)), enabled=True)
    runner = runner or FakeRunner()
    dispatcher = HookDispatcher(registry, runner, repo)
    dispatcher.rebuild_index()
    return dispatcher, runner, repo, registry


def test_durable_hook_receipt_deduplicates_after_reconstructing_dispatcher(tmp_path):
    dispatcher, runner, repo, registry = _durable_dispatcher(tmp_path)
    payload = {'event_id': 'event-1', 'output_ref': 'metrics:step:v1'}
    first = dispatcher.dispatch('step.completed', payload, task_id='task')
    restored = HookDispatcher(registry, runner, repo)
    restored.rebuild_index()
    second = restored.dispatch('step.completed', payload, task_id='task')
    assert not first.required_failures and not second.required_failures
    assert len(runner.calls) == 1
    assert HookDeliveryRepository(repo.db_path).list_deliveries('event-1')[0]['status'] == 'succeeded'
    with pytest.raises(ConflictError, match='different binding'):
        restored.dispatch('step.completed', {**payload, 'output_ref': 'metrics:step:v2'}, task_id='task')


def test_durable_hook_concurrent_dispatch_executes_one_attempt(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class BlockingRunner(FakeRunner):
        def invoke(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().invoke(*args, **kwargs)

    dispatcher, runner, repo, registry = _durable_dispatcher(tmp_path, runner=BlockingRunner())
    other = HookDispatcher(registry, runner, repo)
    other.rebuild_index()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(dispatcher.dispatch, 'step.completed', {'event_id': 'shared'}, task_id='task')
        assert entered.wait(5)
        second = pool.submit(other.dispatch, 'step.completed', {'event_id': 'shared'}, task_id='task').result(timeout=5)
        assert second.required_failures[0]['error_kind'] == 'unknown'
        release.set()
        assert not first.result(timeout=5).required_failures
    assert len(runner.calls) == 1
    assert not other.dispatch('step.completed', {'event_id': 'shared'}, task_id='task').required_failures


@pytest.mark.parametrize('crash_after_receipt', [False, True])
def test_hook_crash_window_never_replays_uncertain_effect(tmp_path, monkeypatch, crash_after_receipt):
    dispatcher, runner, repo, registry = _durable_dispatcher(tmp_path)
    original = dispatcher._deliveries.finish

    class Crash(BaseException):
        pass

    def crash(*args, **kwargs):
        if crash_after_receipt:
            original(*args, **kwargs)
        raise Crash()

    monkeypatch.setattr(dispatcher._deliveries, 'finish', crash)
    with pytest.raises(Crash):
        dispatcher.dispatch('step.completed', {'event_id': 'crashed'}, task_id='task')
    recovered = HookDispatcher(registry, runner, repo)
    recovered.rebuild_index()
    result = recovered.dispatch('step.completed', {'event_id': 'crashed'}, task_id='task')
    assert len(runner.calls) == 1
    assert bool(result.required_failures) is (not crash_after_receipt)
    if not crash_after_receipt:
        assert result.required_failures[0]['error_kind'] == 'unknown'


def test_required_recipients_cannot_disappear_on_recovery(tmp_path, monkeypatch):
    dispatcher, runner, repo, registry = _durable_dispatcher(tmp_path)
    original = dispatcher._deliveries.claim

    class Crash(BaseException):
        pass

    monkeypatch.setattr(dispatcher._deliveries, 'claim', lambda *_a, **_k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        dispatcher.dispatch('step.completed', {'event_id': 'frozen'}, task_id='task')
    monkeypatch.setattr(dispatcher._deliveries, 'claim', original)
    registry.set_enabled('hook_pack', False)
    dispatcher.rebuild_index()
    result = dispatcher.dispatch('step.completed', {'event_id': 'frozen'}, task_id='task')
    assert result.required_failures[0]['error_kind'] == 'binding'
    assert runner.calls == []


def test_required_hook_failure_and_optional_warning_are_distinct(tmp_path):
    dispatcher, runner, repo, registry = _durable_dispatcher(tmp_path, required=False)
    result = dispatcher.dispatch('step.completed', {'event_id': 'optional', 'fail': True}, task_id='task')
    assert not result.required_failures
    assert len(result.warnings) == 1
    assert dispatcher.dispatch('step.completed', {'event_id': 'optional', 'fail': True}, task_id='task').warnings
    assert len(runner.calls) == 1
    registry.register(replace(_manifest(), version='0.2.0', hooks=(HookSpec('step.completed', 'on_task_created', True),)), enabled=True)
    dispatcher.rebuild_index()
    result = dispatcher.dispatch('step.completed', {'fail': True}, task_id='task')
    assert result.required_failures[0]['error_kind'] == 'identity'
    assert len(runner.calls) == 1


def test_required_listener_unknown_effect_is_not_retried(tmp_path):
    dispatcher, runner, repo, registry = _durable_dispatcher(tmp_path)
    calls = []

    def fail_after_effect(_event, payload):
        calls.append(payload['event_id'])
        raise RuntimeError('after external effect')

    dispatcher.register_listener('step.completed', fail_after_effect, required=True)
    for _ in range(2):
        result = dispatcher.dispatch('step.completed', {'event_id': 'listener'}, task_id='task')
        assert result.required_failures[0]['error_kind'] == 'unknown'
    assert calls == ['listener']
