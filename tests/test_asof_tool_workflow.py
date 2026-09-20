"""Real registered Plugin -> governed PlanDriver/Executor -> subprocess ToolRunner."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from marvis.agent.plan_driver import PlanDriver
from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PlanRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.governance.repository import GovernanceRepository
from marvis.governance.service import GovernanceService
from marvis.llm_settings import LLMSettingsError
from marvis.orchestrator.contracts import PlanStatus
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.reviewer import Reviewer
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.hooks import HookDispatcher
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import GovernancePolicy, ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
from tests.test_data_time_contracts import contracts, frames, spec

REF = ToolRef("data_ops", "asof_join")


def no_llm():
    raise LLMSettingsError("No model in deterministic integration test")


@pytest.fixture
def env(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugins = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugins)
    load_builtin_packs(plugin_registry, Path(__file__).parents[1] / "marvis" / "packs")
    tools = ToolRegistry(plugin_registry)
    plans = PlanRepository(settings.db_path)
    governance = GovernanceRepository(settings.db_path)
    principal = governance.create_local_principal(display_name="Synthetic PIT reviewer")
    service = GovernanceService(plan_repo=plans, tool_registry=tools,
                                strategy_repo=StrategyRepository(settings.db_path), governance_repo=governance)
    runner = ToolRunner(tools, plugins, python_executable=sys.executable,
                        datasets_root=settings.datasets_dir, workspace=settings.workspace,
                        governance=governance, binding_resolver=service)
    hooks = HookDispatcher(plugin_registry, runner, plugins)
    hooks.rebuild_index()
    validator = PlanValidator(tools)
    planner = Planner(tools, no_llm, validator)
    executor = PlanExecutor(plans, runner, Reviewer(no_llm), None, hooks, HarnessState(plans), authorizer=service)
    driver = PlanDriver(plans, executor, planner=planner, validator=validator,
                        governance_service=service, local_principal=principal)
    load_builtin_templates()
    task = TaskRepository(settings.db_path).create_task(TaskCreate(
        model_name="synthetic asof", model_version="v1", validator="test",
        source_dir=str(tmp_path), task_type="data_join",
    ))
    repo = DatasetRepository(settings.db_path)
    registry = DatasetRegistry(repo, DataBackend(settings.datasets_dir), settings.datasets_dir)
    inputs = {}
    for label, frame, contract in zip(("decision_contract", "feature_contract"), frames(), contracts(), strict=True):
        path = settings.datasets_dir / task.id / f"{label}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        with repo.transaction() as conn:
            dataset = registry.register_existing_on_connection(conn, path, task_id=task.id, role=contract.role, target_col_override=None)
        inputs[label] = {**contract.model_dump(mode="json"), "dataset_id": dataset.id, "content_hash": dataset.content_hash}
    inputs["spec"] = spec().model_dump(mode="json")
    return SimpleNamespace(settings=settings, driver=driver, plans=plans, task=task, runner=runner,
                           registry=registry, repo=repo, inputs=inputs, tools=tools, validator=validator,
                           planner=planner, executor=executor, service=service, principal=principal,
                           artifacts=TaskArtifactRepository(settings.db_path), plugins=plugins)


def pause(env, inputs=None):
    turn = env.driver.start(task_id=env.task.id, template_id="dataset_asof_join", slots=inputs or env.inputs)
    assert turn.status == PlanStatus.VALIDATED.value
    turn = env.driver.resume(plan_id=turn.plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    return turn.plan_id


def finish(env, plan_id):
    return env.driver.resume(plan_id=plan_id, user_text="确认", run_seq=2)


def test_real_workflow_gates_then_materializes_authenticated_asof_dataset(env):
    plan_id = pause(env)
    assert env.artifacts.list_for_task(env.task.id) == []
    plan = env.plans.load_plan(plan_id)
    assert env.plans.list_step_runs(plan.steps[0].id) == []
    turn = finish(env, plan_id)
    assert turn.status == PlanStatus.DONE.value
    step = env.plans.load_plan(plan_id).steps[0]
    output = env.plans.load_step_output(step.id)
    assert output["schema_version"] == "dataset-asof-tool-result.v1"
    assert output["assurance"] == "verified"
    assert output["mode"] == "verified"
    assert output["row_count"] == 2
    assert [p["content_hash"] for p in output["parents"]] == [env.inputs[c]["content_hash"] for c in ("decision_contract", "feature_contract")]
    dataset = env.registry.get(output["result_dataset_id"])
    assert dataset.content_hash == output["result_content_hash"]
    assert env.registry.read_authenticated_parquet_snapshot(dataset.id).asof__amount.tolist() == [10, 20]
    record = env.artifacts.get_for_task(env.task.id, output["evidence"]["artifact_id"])
    assert record["content_hash"] == output["evidence"]["content_hash"]
    assert AsOfJoinEngine(env.registry, env.artifacts, workspace_root=env.settings.workspace).dataset_time_status(dataset.id).assurance == "verified"
    runs = env.plans.list_step_runs(step.id)
    assert len(runs) == 1
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["invocation_contract"]["tool_ref"] == "data_ops.asof_join"
    evidence = env.plans.load_step_evidence(step.id)
    assert evidence["tool_name"] == "data_ops.asof_join"
    assert evidence["raw_output_hash"]
    assert evidence["result_dataset_bindings"] == [{"dataset_id": dataset.id, "content_hash": dataset.content_hash}]
    assert evidence["artifact_bindings"] == [output["evidence"]]
    assert output["verification_scope"] == "recorded_temporal_constraints"
    assert output["external_source_attestation"] == "not_independently_verified"


def test_direct_runner_cannot_bypass_live_governance_decision(env):
    result = env.runner.invoke(REF, env.inputs, task_id=env.task.id)
    assert not result.ok
    assert result.error_kind == "authorization"
    assert len(env.registry.list_for_task(env.task.id)) == 2
    assert env.artifacts.list_for_task(env.task.id) == []


@pytest.mark.parametrize("path", [("spec", "mode"), ("feature_contract", "available_at"),
                                 ("decision_contract", "decision_at", "timezone"),
                                 ("feature_contract", "event_at", "evidence")])
def test_registered_tool_schema_requires_explicit_time_facts_before_dispatch(env, path):
    inputs = deepcopy(env.inputs)
    target = inputs
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    result = env.runner.invoke(REF, inputs, task_id=env.task.id)
    assert not result.ok
    assert result.error_kind == "schema"
    assert env.artifacts.list_for_task(env.task.id) == []


@pytest.mark.parametrize("mode,expected", [("verified", PlanStatus.FAILED), ("exploration", PlanStatus.DONE)])
def test_missing_historical_availability_never_becomes_verified_through_plan(env, mode, expected):
    inputs = deepcopy(env.inputs)
    inputs["feature_contract"]["available_at"] = None
    inputs["spec"]["mode"] = mode
    plan_id = pause(env, inputs)
    assert finish(env, plan_id).status == expected.value
    if mode == "verified":
        assert env.artifacts.list_for_task(env.task.id) == []
    else:
        step = env.plans.load_plan(plan_id).steps[0]
        output = env.plans.load_step_output(step.id)
        assert output["assurance"] == "unknown"
        assert "historical_available_at_missing" in output["assurance_reasons"]


def test_tool_checks_both_parents_belong_to_active_task(env, caplog):
    plan_id = pause(env)
    with env.repo.transaction() as conn:
        conn.execute("UPDATE datasets SET task_id='another-task' WHERE id=?", (env.inputs["feature_contract"]["dataset_id"],))
    assert finish(env, plan_id).status == PlanStatus.FAILED.value
    assert env.artifacts.list_for_task(env.task.id) == []
    step = env.plans.load_plan(plan_id).steps[0]
    assert "explicit reconciliation required" in step.error
    assert env.plans.list_step_runs(step.id)[0]["status"] == "failed"
    assert "expected task" in caplog.text


def test_stale_confirmation_cannot_authorize_changed_selection_inputs(env):
    plan_id = pause(env)
    plan = env.plans.load_plan(plan_id)
    step = plan.steps[0]
    env.service.authorize_step(plan_id=plan_id, step_id=step.id, principal=env.principal,
                               reason="Reviewed the temporal source declarations", expected_plan_revision=0)
    context = env.service.execution_context_for(plan=plan, step=step, inputs=step.inputs)
    changed = deepcopy(step.inputs)
    changed["spec"]["mode"] = "exploration"
    env.plans.update_step(replace(step, inputs=changed))
    result = env.runner.invoke(REF, changed, task_id=env.task.id, execution_context=context)
    assert not result.ok
    assert result.error_kind == "authorization"
    assert env.artifacts.list_for_task(env.task.id) == []


def test_plan_validator_rejects_lowering_manifest_confirmation_policy(env):
    plan = env.planner.from_template(get_template("dataset_asof_join"), env.inputs, env.task.id)
    assert env.validator.validate(plan) == []
    plan.steps[0] = replace(plan.steps[0], policy=GovernancePolicy(), needs_confirmation=False)
    assert any("cannot be lower" in error for error in env.validator.validate(plan))


def test_future_decision_is_rejected_inside_real_registered_tool(env, caplog):
    inputs = deepcopy(env.inputs)
    inputs["spec"]["as_of"] = "2026-01-01T00:00:00Z"
    plan_id = pause(env, inputs)
    assert finish(env, plan_id).status == PlanStatus.FAILED.value
    assert env.artifacts.list_for_task(env.task.id) == []
    assert "explicit reconciliation required" in env.plans.load_plan(plan_id).steps[0].error
    assert "decision_at is after replay as_of" in caplog.text



def test_tool_cannot_accept_an_input_override_of_trusted_task_or_output_path(env):
    for extra in ({"task_id": "foreign-task"}, {"output_path": "/tmp/unguarded.parquet"}):
        result = env.runner.invoke(REF, {**env.inputs, **extra}, task_id=env.task.id)
        assert not result.ok
        assert result.error_kind == "schema"
    assert env.artifacts.list_for_task(env.task.id) == []
