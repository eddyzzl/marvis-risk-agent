from __future__ import annotations

import sqlite3
import traceback
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from marvis.db import TaskRepository
from marvis.orchestrator.contracts import PlanStatus, plan_to_dict
from marvis.orchestrator.errors import IllegalPlanTransition, PlanNotFoundError
from marvis.orchestrator.planner import PlanningError
from marvis.orchestrator.templates import get_template
from marvis.state_machine import ConflictError


router = APIRouter(prefix="/api", tags=["plans"])
PLAN_JOB_KIND = "plan"
ACTIVE_JOB_DETAIL = "task already has an active job"


class CreatePlanRequest(BaseModel):
    goal: str
    autonomy_level: int | None = None
    slots: dict = Field(default_factory=dict)
    task_context: dict = Field(default_factory=dict)
    memory_context: dict = Field(default_factory=dict)


@router.post("/tasks/{task_id}/plans", status_code=201)
def create_plan(request: Request, task_id: str, body: CreatePlanRequest) -> dict:
    intent_router = request.app.state.intent_router
    planner = request.app.state.planner
    validator = request.app.state.plan_validator
    repo = request.app.state.plan_repo
    task_context = _task_context(task_id, body)

    try:
        intent = intent_router.route(body.goal, task_context)
        if intent.kind == "template":
            plan = planner.from_template(
                get_template(intent.template_id),
                intent.slots,
                task_id,
                autonomy=body.autonomy_level,
            )
        else:
            plan = planner.generate(
                body.goal,
                task_id,
                memory_context=dict(body.memory_context),
                task_context=task_context,
            )
    except (KeyError, PlanningError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    problems = validator.validate(plan)
    if problems:
        raise HTTPException(status_code=422, detail={"problems": problems})

    plan.status = PlanStatus.VALIDATED
    try:
        repo.create_plan(plan)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="plan already exists") from exc
    return _plan_payload(plan)


@router.get("/plans/{plan_id}")
def get_plan(request: Request, plan_id: str) -> dict:
    return _load_plan_payload(request, plan_id)


@router.post("/plans/{plan_id}/confirm")
def confirm_plan(request: Request, plan_id: str) -> dict:
    try:
        request.app.state.plan_repo.confirm_plan(plan_id)
    except PlanNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IllegalPlanTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _load_plan_payload(request, plan_id)


@router.post("/plans/{plan_id}/run", status_code=202)
def run_plan(request: Request, plan_id: str, background_tasks: BackgroundTasks) -> dict:
    plan = _load_plan(request, plan_id)
    if plan.status not in {PlanStatus.CONFIRMED, PlanStatus.AWAITING_CONFIRM, PlanStatus.RUNNING}:
        raise HTTPException(status_code=409, detail=f"plan is not runnable: {plan.status.value}")
    job_id = _start_plan_job(request, plan.task_id)
    background_tasks.add_task(
        _run_plan_job,
        job_id,
        _db_path(request),
        request.app.state.plan_executor,
        plan_id,
    )
    return {"ok": True, "plan_id": plan_id, "job_id": job_id, "status": plan.status.value}


@router.post("/plans/{plan_id}/steps/{step_id}/confirm", status_code=202)
def confirm_step(
    request: Request,
    plan_id: str,
    step_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    plan = _load_plan(request, plan_id)
    if step_id not in {step.id for step in plan.steps}:
        raise HTTPException(status_code=404, detail="step not found")
    job_id = _start_plan_job(request, plan.task_id)
    try:
        request.app.state.plan_repo.confirm_step(step_id)
    except KeyError as exc:
        _fail_plan_job(_db_path(request), job_id, exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    background_tasks.add_task(
        _run_plan_job,
        job_id,
        _db_path(request),
        request.app.state.plan_executor,
        plan_id,
    )
    return {"ok": True, "plan_id": plan_id, "step_id": step_id, "job_id": job_id}


@router.post("/plans/{plan_id}/cancel")
def cancel_plan(request: Request, plan_id: str) -> dict:
    repo = request.app.state.plan_repo
    try:
        repo.set_plan_status(plan_id, PlanStatus.CANCELLED)
    except PlanNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IllegalPlanTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _load_plan_payload(request, plan_id)


def _load_plan_payload(request: Request, plan_id: str) -> dict:
    return _plan_payload(_load_plan(request, plan_id))


def _load_plan(request: Request, plan_id: str):
    try:
        return request.app.state.plan_repo.load_plan(plan_id)
    except PlanNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _plan_payload(plan) -> dict:
    return {"plan": plan_to_dict(plan)}


def _task_context(task_id: str, body: CreatePlanRequest) -> dict:
    context = {"task_id": task_id}
    context.update(dict(body.task_context))
    context.update(dict(body.slots))
    return context


def _start_plan_job(request: Request, task_id: str) -> str:
    try:
        return TaskRepository(_db_path(request)).start_job(task_id, PLAN_JOB_KIND)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=ACTIVE_JOB_DETAIL) from exc


def _run_plan_job(job_id: str, db_path: Path, executor, plan_id: str) -> None:
    repo = TaskRepository(db_path)
    repo.mark_job_running(job_id)
    try:
        result = executor.run(plan_id)
    except Exception as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    status = "failed" if result.status == PlanStatus.FAILED else "succeeded"
    repo.finish_job(job_id, status=status)


def _fail_plan_job(db_path: Path, job_id: str, exc: Exception) -> None:
    TaskRepository(db_path).finish_job(
        job_id,
        status="failed",
        error_name=exc.__class__.__name__,
        error_value=str(exc),
        traceback="",
    )


def _db_path(request: Request) -> Path:
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        return settings.db_path
    return request.app.state.plan_repo.db_path
