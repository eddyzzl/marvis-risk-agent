import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.api_task_payloads import task_payload
from marvis.db import PlanRepository, TaskRepository, init_db
from marvis.domain import TASK_TYPE_MODELING, TaskCreate, TaskStatus
from marvis.orchestrator.contracts import (
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from marvis.plugins.manifest import ToolRef
from marvis.routers.tasks import router as tasks_router
from marvis.settings import build_settings


def _client(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    app = FastAPI()
    app.state.settings = settings
    app.include_router(tasks_router)
    return TestClient(app), settings


def _task(repo: TaskRepository, name: str):
    return repo.create_task(
        TaskCreate(
            model_name=name,
            model_version="",
            validator="pytest",
            source_dir="/tmp",
            task_type=TASK_TYPE_MODELING,
        )
    )


def _plan(
    *,
    plan_id: str,
    task_id: str,
    created_at: str,
    plan_status: PlanStatus,
    step_status: StepStatus,
) -> Plan:
    return Plan(
        id=plan_id,
        task_id=task_id,
        goal="model",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=plan_status,
        created_at=created_at,
        updated_at=created_at,
        steps=[
            PlanStep(
                id=f"{plan_id}-step",
                plan_id=plan_id,
                index=0,
                title="训练模型",
                tool_ref=ToolRef("modeling", "train_model"),
                inputs={},
                depends_on=[],
                post_checks=[],
                status=step_status,
            )
        ],
    )


def test_task_apis_expose_latest_normalized_plan_status_without_mutating_task(
    tmp_path,
    monkeypatch,
):
    client, settings = _client(tmp_path)
    task_repo = TaskRepository(settings.db_path)
    plan_repo = PlanRepository(settings.db_path)
    failed_task = _task(task_repo, "failed workflow")
    done_task = _task(task_repo, "done workflow")
    confirmed_task = _task(task_repo, "confirmed workflow")
    no_plan_task = _task(task_repo, "no plan")

    # An older successful plan must not mask the latest attempt.
    plan_repo.create_plan(
        _plan(
            plan_id="failed-old",
            task_id=failed_task.id,
            created_at="2026-07-20T00:00:00+00:00",
            plan_status=PlanStatus.DONE,
            step_status=StepStatus.DONE,
        )
    )
    # Recovery may park the plan at awaiting_confirm, but the unresolved failed
    # step remains the authoritative workflow state.
    plan_repo.create_plan(
        _plan(
            plan_id="failed-latest",
            task_id=failed_task.id,
            created_at="2026-07-21T00:00:00+00:00",
            plan_status=PlanStatus.AWAITING_CONFIRM,
            step_status=StepStatus.FAILED,
        )
    )
    plan_repo.create_plan(
        _plan(
            plan_id="done-latest",
            task_id=done_task.id,
            created_at="2026-07-22T00:00:00+00:00",
            plan_status=PlanStatus.DONE,
            step_status=StepStatus.DONE,
        )
    )
    plan_repo.create_plan(
        _plan(
            plan_id="confirmed-latest",
            task_id=confirmed_task.id,
            created_at="2026-07-22T01:00:00+00:00",
            plan_status=PlanStatus.CONFIRMED,
            step_status=StepStatus.PENDING,
        )
    )

    calls: list[list[str]] = []
    original = PlanRepository.latest_workflow_statuses_for_tasks

    def spy(self, task_ids):
        calls.append(list(task_ids))
        return original(self, task_ids)

    monkeypatch.setattr(
        PlanRepository,
        "latest_workflow_statuses_for_tasks",
        spy,
    )

    listed_response = client.get("/api/tasks")

    assert listed_response.status_code == 200, listed_response.text
    listed = {row["id"]: row for row in listed_response.json()}
    assert listed[failed_task.id]["status"] == TaskStatus.CREATED.value
    assert listed[failed_task.id]["workflow_status"] == "failed"
    assert listed[done_task.id]["status"] == TaskStatus.CREATED.value
    assert listed[done_task.id]["workflow_status"] == "done"
    assert listed[confirmed_task.id]["workflow_status"] == "running"
    assert listed[no_plan_task.id]["workflow_status"] is None
    assert len(calls) == 1
    assert set(calls[0]) == {
        failed_task.id,
        done_task.id,
        confirmed_task.id,
        no_plan_task.id,
    }

    detail_response = client.get(f"/api/tasks/{failed_task.id}")

    assert detail_response.status_code == 200, detail_response.text
    assert detail_response.json()["status"] == TaskStatus.CREATED.value
    assert detail_response.json()["workflow_status"] == "failed"
    assert len(calls) == 2
    assert calls[1] == [failed_task.id]
    assert task_repo.get_task(failed_task.id).status == TaskStatus.CREATED


def test_task_payload_treats_missing_optional_plan_schema_as_no_workflow(
    tmp_path,
    monkeypatch,
):
    _client_instance, settings = _client(tmp_path)
    task_repo = TaskRepository(settings.db_path)
    task = _task(task_repo, "legacy validation")

    def missing_plan_schema(_self, _task_ids):
        raise sqlite3.OperationalError("no such table: plans")

    monkeypatch.setattr(
        PlanRepository,
        "latest_workflow_statuses_for_tasks",
        missing_plan_schema,
    )

    payload = task_payload(task_repo, task, settings.tasks_dir)

    assert payload["workflow_status"] is None
