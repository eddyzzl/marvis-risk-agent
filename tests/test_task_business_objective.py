from dataclasses import replace
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from marvis.api_schemas import AgentMessageRequest, CreateTaskRequest
from marvis.business_acceptance import BusinessObjective
from marvis.db_schema import connect, init_db, _MIGRATIONS
from marvis.domain import StrategyTaskInput
from marvis.repositories.tasks import TaskRepository
from marvis.repositories.plans import PlanRepository
from marvis.routers.tasks import router
from marvis.state_machine import ConflictError
from tests.test_business_acceptance import objective
from tests.test_db import _task_create
from tests.test_orch_api import _plan


def setup(tmp_path):
    db = tmp_path / "app.sqlite"
    init_db(db)
    repo = TaskRepository(db)
    task = repo.create_task(_task_create(task_type="modeling"))
    app = FastAPI()
    app.state.settings = SimpleNamespace(db_path=db, tasks_dir=tmp_path / "tasks")
    app.include_router(router)
    return repo, task, TestClient(app)


@pytest.mark.parametrize(
    "kind",
    [
        "modeling",
        "strategy",
        "portfolio",
        "feature_analysis",
        "data_join",
        "vintage",
        "validation",
    ],
)
def test_task_objective_round_trip_all_workflows(tmp_path, kind):
    repo, _, _ = setup(tmp_path)
    task = repo.create_task(
        _task_create(task_type=kind, business_objective=objective())
    )
    assert (
        task.business_objective
        == repo.get_task(task.id).business_objective
        == objective()
    )


def test_put_contract_owns_lease_then_locks_after_plan_even_terminal(tmp_path):
    repo, task, client = setup(tmp_path)
    endpoint = f"/api/tasks/{task.id}/business-objective"
    assert (
        client.put(
            endpoint, json={"business_objective": objective().to_dict()}
        ).status_code
        == 200
    )
    detail = client.get(f"/api/tasks/{task.id}").json()
    assert detail["business_objective"]["business_line"] == objective().business_line
    assert detail["business_objective_locked"] is False
    job = repo.start_job(task.id, "other")
    assert client.put(endpoint, json={"business_objective": None}).status_code == 409
    repo.finish_job(job, status="succeeded")
    plans = PlanRepository(repo.db_path)
    plan = _plan(task_id=task.id)
    from marvis.orchestrator.contracts import PlanStatus

    plan.status = PlanStatus.DONE
    plans.create_plan(plan)
    assert client.put(endpoint, json={"business_objective": None}).status_code == 409
    assert (
        client.get(f"/api/tasks/{task.id}").json()["business_objective_locked"] is True
    )
    assert repo.get_task(task.id).business_objective == objective()
    assert not repo.task_has_active_job(task.id)


def test_repository_rejects_missing_wrong_and_finished_lease(tmp_path):
    repo, task, _ = setup(tmp_path)
    for job in (None, "missing"):
        with pytest.raises(ConflictError, match="lease"):
            repo.update_business_objective(task.id, objective(), job_id=job)
    other = repo.create_task(_task_create())
    job = repo.start_job(other.id, "business_objective")
    repo.mark_job_running(job)
    with pytest.raises(ConflictError, match="lease"):
        repo.update_business_objective(task.id, objective(), job_id=job)


def test_strategy_legacy_compatibility_and_conflicting_dual_write(tmp_path):
    repo, _, _ = setup(tmp_path)
    nested = StrategyTaskInput(business_objective=objective())
    task = repo.create_task(_task_create(task_type="strategy", strategy_input=nested))
    assert repo.get_task(task.id).business_objective == objective()
    with pytest.raises(ValueError, match="conflicts"):
        repo.create_task(
            _task_create(
                task_type="strategy",
                strategy_input=nested,
                business_objective=replace(objective(), business_line="different"),
            )
        )
    for cls, fields in (
        (CreateTaskRequest, {"model_name": "x", "validator": "x"}),
        (AgentMessageRequest, {"content": "继续"}),
    ):
        with pytest.raises(ValueError, match="conflicts"):
            cls(
                **fields,
                strategy_input={"business_objective": objective().to_dict()},
                business_objective=replace(
                    objective(), business_line="different"
                ).to_dict(),
            )
    job = repo.start_job(task.id, "business_objective")
    repo.mark_job_running(job)
    new = replace(objective(), population="another")
    repo.update_business_objective(task.id, new, job_id=job)
    repo.finish_job(job, status="succeeded")
    assert repo.get_task(task.id).strategy_input.business_objective == new
    # Legacy strategy continuation does not silently drop the general contract.
    repo.update_strategy_input(task.id, StrategyTaskInput())
    assert repo.get_task(task.id).business_objective == new
    assert repo.get_task(task.id).strategy_input.business_objective == new


def test_migration_39_backfills_exact_declared_contract(tmp_path):
    db = tmp_path / "old.sqlite"
    with connect(db) as conn:
        for version, migration in _MIGRATIONS:
            if version == 39:
                break
            migration(conn)
        conn.execute("PRAGMA user_version = 38")
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO tasks(id, model_name, model_version, validator, source_dir, algorithm, run_mode, target_col, score_col, split_col, time_col, status, status_message, created_at, updated_at, strategy_input_json, task_type) VALUES ('old','x','v1','x','/tmp','lr','manual','y','p','s','t','created','created','now','now',?,'strategy')",
            (json.dumps({"business_objective": objective().to_dict()}),),
        )
    init_db(db)
    task = TaskRepository(db).get_task("old")
    assert isinstance(task.business_objective, BusinessObjective)
    assert task.business_objective == objective()


def test_agent_top_level_declaration_is_saved_under_driver_lease_and_clarifies(
    tmp_path,
):
    from tests.test_strategy_clarification_resume import _client, _create_strategy_task

    client = _client(tmp_path)
    task = _create_strategy_task(client, tmp_path)
    response = client.post(
        f"/api/tasks/{task['id']}/agent/messages",
        json={
            "content": "补充业务验收合同",
            "business_objective": objective(target_kind="strategy").to_dict(),
        },
    )
    assert response.status_code == 202, response.text
    stored = client.get(f"/api/tasks/{task['id']}").json()
    assert stored["business_objective"]["target_kind"] == "strategy"
    assert (
        response.json()["clarification"]["current_input"]["business_objective"]
        == stored["business_objective"]
    )
    assert not stored["business_objective_locked"]


def test_driver_binds_task_objective_for_every_template(tmp_path):
    from marvis.agent.plan_driver import PlanDriver
    from marvis.orchestrator.templates import load_builtin_templates
    from tests.test_orch_api import FakePlanner

    repo, task, _ = setup(tmp_path)
    job = repo.start_job(task.id, "driver")
    repo.mark_job_running(job)
    repo.update_business_objective(task.id, objective(), job_id=job)
    plans = PlanRepository(repo.db_path)
    load_builtin_templates()
    driver = PlanDriver(plan_repo=plans, executor=None, planner=FakePlanner())
    plan = driver._prepare_plan(
        task_id=task.id,
        template_id="sample_echo",
        slots={},
        success_criteria=[{"metric": "oot_ks", "min": 0.1, "aggregate": "max"}],
    )
    assert plan.success_criteria[0] == objective().to_dict()
    assert plan.success_criteria[1]["min"] == 0.1


@pytest.mark.parametrize("minimum", [0.4, 1.0])
def test_driver_cannot_drop_stricter_persisted_oot_threshold(tmp_path, minimum):
    from marvis.agent.plan_driver import PlanDriver, DriverError
    from marvis.orchestrator.templates import load_builtin_templates
    from tests.test_orch_api import FakePlanner

    repo, _, _ = setup(tmp_path)
    task = repo.create_task(
        _task_create(
            task_type="modeling", oot_ks_min=minimum, business_objective=objective()
        )
    )
    load_builtin_templates()
    driver = PlanDriver(PlanRepository(repo.db_path), None, planner=FakePlanner())
    with pytest.raises(DriverError, match="旧阈值"):
        driver._prepare_plan(task_id=task.id, template_id="sample_echo", slots={})


def test_planner_app_loads_general_objective_and_checks_old_threshold(tmp_path):
    from marvis.app import create_app
    from marvis.orchestrator.planner import PlanningError

    app = create_app(tmp_path)
    repo = TaskRepository(app.state.settings.db_path)
    task = repo.create_task(
        _task_create(
            task_type="modeling", oot_ks_min=0.5, business_objective=objective()
        )
    )
    assert app.state.planner._business_objective_loader(task.id) == objective()
    with pytest.raises(PlanningError, match="旧阈值"):
        app.state.planner._business_criteria(task.id, [])
