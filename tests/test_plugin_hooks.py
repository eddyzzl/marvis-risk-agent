from marvis.db import PluginRepository, init_db
from marvis.plugins.hooks import HookDispatcher
from marvis.plugins.manifest import ToolRef, parse_manifest
from marvis.plugins.registry import PluginRegistry
from marvis.plugins.runner import ToolResult


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
