from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.db import PlanRepository, PluginRepository, connect, init_db
import marvis.db as db_module
from marvis.orchestrator.contracts import AgentStatus, Plan, PlanStatus, PlanStep
from marvis.orchestrator.subagent import SubAgentDispatcher
from marvis.orchestrator.templates import load_builtin_templates
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


class FakeIntentRouter:
    def __init__(self, kind="template"):
        self.kind = kind
        self.calls = []

    def route(self, scope, goal_inputs):
        self.calls.append((scope, goal_inputs))
        return SimpleNamespace(
            kind=self.kind,
            template_id="sample_echo" if self.kind == "template" else None,
            slots={"message": "hi"},
        )


class FakePlanner:
    def __init__(self):
        self.from_template_calls = []
        self.generate_calls = []

    def from_template(self, template, slots, task_id):
        self.from_template_calls.append((template.id, slots, task_id))
        return _mini_plan(task_id)

    def generate(self, scope, task_id, *, memory_context, task_context):
        self.generate_calls.append((scope, task_id, memory_context, task_context))
        return _mini_plan(task_id)


def _tool_registry(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    plugin_repo = PluginRepository(db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(plugin_registry, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugin_registry)


def _step(*, grants=None, scope="summarize") -> PlanStep:
    return PlanStep(
        id="step-1",
        plan_id="plan-1",
        index=0,
        title="Sub",
        tool_ref=ToolRef("_sample", "echo"),
        inputs={},
        depends_on=[],
        post_checks=[],
        sub_agent_scope=scope,
        granted_tools=grants if grants is not None else [ToolRef("_sample", "echo")],
    )


def _mini_plan(task_id: str) -> Plan:
    return Plan(
        id="mini-plan",
        task_id=task_id,
        goal="mini",
        source="template",
        template_id="sample_echo",
        steps=[
            PlanStep(
                id="mini-step",
                plan_id="mini-plan",
                index=0,
                title="Echo",
                tool_ref=ToolRef("_sample", "echo"),
                inputs={"message": "hi"},
                depends_on=[],
                post_checks=[],
            )
        ],
        autonomy_level=1,
    )


def test_subagent_spawn_persists_minimal_grants_and_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    dispatcher = SubAgentDispatcher(repo, FakePlanner(), lambda _registry: None, None, FakeIntentRouter())

    sub = dispatcher.spawn(_step(), parent_task_id="task-1")

    loaded = repo.get_sub_agent(sub.id)
    assert loaded.status == AgentStatus.SPAWNED
    assert loaded.granted_tools == [ToolRef("_sample", "echo")]
    assert repo.list_audit(kind="subagent.spawn")[0]["target_ref"] == sub.id


def test_subagent_spawn_rolls_back_when_audit_write_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    dispatcher = SubAgentDispatcher(repo, FakePlanner(), lambda _registry: None, None, FakeIntentRouter())

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        dispatcher.spawn(_step(), parent_task_id="task-1")

    with connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM sub_agents").fetchone()[0]
    assert count == 0


def test_subagent_spawn_rejects_empty_grants(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    dispatcher = SubAgentDispatcher(repo, FakePlanner(), lambda _registry: None, None, FakeIntentRouter())

    with pytest.raises(ValueError, match="granted_tools"):
        dispatcher.spawn(_step(grants=[]), parent_task_id="task-1")


def test_subagent_run_uses_template_path_and_restricted_registry(tmp_path):
    load_builtin_templates()
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    planner = FakePlanner()
    intent = FakeIntentRouter(kind="template")
    seen_catalogs = []

    def executor_factory(restricted_registry):
        seen_catalogs.append(restricted_registry.catalog_for_planner())
        return SimpleNamespace(run=lambda plan_id: SimpleNamespace(summary_ref="artifact:summary"))

    dispatcher = SubAgentDispatcher(
        repo,
        planner,
        executor_factory,
        _tool_registry(tmp_path),
        intent,
    )
    sub = dispatcher.spawn(_step(), parent_task_id="task-1")

    result = dispatcher.run(sub, goal_inputs={"message": "hi"})

    assert result.ok is True
    assert result.output == {"result_ref": "artifact:summary"}
    assert planner.from_template_calls
    assert planner.generate_calls == []
    assert {tool["tool"] for tool in seen_catalogs[0]} == {"echo"}
    assert repo.get_sub_agent(sub.id).status == AgentStatus.RETURNED


def test_subagent_run_does_not_return_when_final_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    load_builtin_templates()
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)

    def executor_factory(_restricted_registry):
        return SimpleNamespace(run=lambda _plan_id: SimpleNamespace(summary_ref="artifact:summary"))

    dispatcher = SubAgentDispatcher(
        repo,
        FakePlanner(),
        executor_factory,
        _tool_registry(tmp_path),
        FakeIntentRouter(kind="template"),
    )
    sub = dispatcher.spawn(_step(), parent_task_id="task-1")

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        dispatcher.run(sub, goal_inputs={"message": "hi"})

    loaded = repo.get_sub_agent(sub.id)
    assert loaded.status == AgentStatus.RUNNING
    assert loaded.result_ref is None


def test_subagent_run_confirms_mini_plan_before_executor_run(tmp_path):
    load_builtin_templates()
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    seen_statuses = []

    def executor_factory(_restricted_registry):
        def run(plan_id):
            seen_statuses.append(repo.load_plan(plan_id).status)
            return SimpleNamespace(summary_ref="artifact:summary")

        return SimpleNamespace(run=run)

    dispatcher = SubAgentDispatcher(
        repo,
        FakePlanner(),
        executor_factory,
        _tool_registry(tmp_path),
        FakeIntentRouter(kind="template"),
    )
    sub = dispatcher.spawn(_step(), parent_task_id="task-1")

    result = dispatcher.run(sub, goal_inputs={"message": "hi"})

    assert result.ok is True
    assert seen_statuses == [PlanStatus.CONFIRMED]


def test_subagent_run_does_not_return_success_for_paused_inner_plan(tmp_path):
    load_builtin_templates()
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)

    def executor_factory(_restricted_registry):
        return SimpleNamespace(
            run=lambda _plan_id: SimpleNamespace(status=PlanStatus.AWAITING_CONFIRM, summary_ref=None),
        )

    dispatcher = SubAgentDispatcher(
        repo,
        FakePlanner(),
        executor_factory,
        _tool_registry(tmp_path),
        FakeIntentRouter(kind="template"),
    )
    sub = dispatcher.spawn(_step(), parent_task_id="task-1")

    result = dispatcher.run(sub, goal_inputs={"message": "hi"})

    assert result.ok is False
    assert result.error_kind == "paused"
    assert "awaiting_confirm" in result.error
    assert repo.get_sub_agent(sub.id).status == AgentStatus.FAILED
    audit = repo.list_audit(kind="subagent.run")[0]
    assert audit["outcome"] == "failed"
    assert audit["detail"]["status"] == PlanStatus.AWAITING_CONFIRM.value


def test_subagent_run_uses_restricted_planner_factory_for_novel_plans(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    restricted_catalogs = []
    planner = FakePlanner()

    def planner_factory(restricted_registry):
        restricted_catalogs.append(restricted_registry.catalog_for_planner())
        return planner

    def executor_factory(_restricted_registry):
        return SimpleNamespace(run=lambda _plan_id: SimpleNamespace(summary_ref="artifact:summary"))

    dispatcher = SubAgentDispatcher(
        repo,
        FakePlanner(),
        executor_factory,
        _tool_registry(tmp_path),
        FakeIntentRouter(kind="novel"),
        planner_factory=planner_factory,
    )
    sub = dispatcher.spawn(_step(), parent_task_id="task-1")

    result = dispatcher.run(sub, goal_inputs={})

    assert result.ok is True
    assert planner.generate_calls
    assert {tool["tool"] for tool in restricted_catalogs[0]} == {"echo"}


def test_subagent_run_isolates_executor_failures(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)

    def executor_factory(_restricted_registry):
        def fail(_plan_id):
            raise RuntimeError("boom")

        return SimpleNamespace(run=fail)

    dispatcher = SubAgentDispatcher(
        repo,
        FakePlanner(),
        executor_factory,
        _tool_registry(tmp_path),
        FakeIntentRouter(kind="novel"),
    )
    sub = dispatcher.spawn(_step(), parent_task_id="task-1")

    result = dispatcher.run(sub, goal_inputs={})

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "boom" in result.error
    assert repo.get_sub_agent(sub.id).status == AgentStatus.FAILED
