from __future__ import annotations

from dataclasses import asdict
import logging
import sqlite3
import traceback
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from marvis.errors import conflict, not_found, unprocessable
from pydantic import BaseModel, Field

from marvis.db import PlanRepository, TaskRepository
from marvis.orchestrator.capability import TIERS, resolve_tier, tier_from_settings
from marvis.agent.gates import build_failure_envelope
from marvis.orchestrator.contracts import PlanStatus, StepStatus, plan_to_dict
from marvis.orchestrator.errors import IllegalPlanTransition, PlanNotFoundError
from marvis.orchestrator.planner import PlanningError
from marvis.orchestrator.templates import get_template
from marvis.job_heartbeat import heartbeat_job
from marvis.state_machine import ConflictError


router = APIRouter(prefix="/api", tags=["plans"])
logger = logging.getLogger(__name__)
PLAN_JOB_KIND = "plan"
ACTIVE_JOB_DETAIL = "task already has an active job"


class CreatePlanRequest(BaseModel):
    goal: str
    autonomy_level: int | None = None
    novel_mode: Literal["plan_ahead", "explore"] = "plan_ahead"
    tier: str | None = None
    slots: dict = Field(default_factory=dict)
    task_context: dict = Field(default_factory=dict)
    memory_context: dict = Field(default_factory=dict)


class RetryStepRequest(BaseModel):
    inputs: dict | None = None


@router.post("/tasks/{task_id}/plans", status_code=201)
def create_plan(request: Request, task_id: str, body: CreatePlanRequest) -> dict:
    intent_router = request.app.state.intent_router
    planner = request.app.state.planner
    validator = request.app.state.plan_validator
    repo = request.app.state.plan_repo
    task_context = _task_context(task_id, body)
    tier = _requested_tier(request, body.tier)

    try:
        intent = intent_router.route(body.goal, task_context)
        if intent.kind == "template":
            plan = planner.from_template(
                get_template(intent.template_id),
                intent.slots,
                task_id,
                autonomy=body.autonomy_level,
            )
            plan.tier = tier.name
        else:
            plan = planner.generate(
                body.goal,
                task_id,
                memory_context=dict(body.memory_context),
                task_context=task_context,
                tier=tier,
                novel_mode=body.novel_mode,
            )
    except (KeyError, PlanningError, ValueError) as exc:
        raise unprocessable(str(exc)) from exc

    problems = validator.validate(plan)
    if problems:
        raise HTTPException(status_code=422, detail={"problems": problems})

    plan.status = PlanStatus.VALIDATED
    try:
        repo.create_plan(plan)
    except sqlite3.IntegrityError as exc:
        raise conflict("plan already exists") from exc
    return _plan_payload(request, plan)


@router.get("/capability-tiers")
def list_capability_tiers() -> dict:
    return {
        "tiers": [asdict(tier) for tier in TIERS.values()],
        "default": "balanced",
    }


@router.get("/step-outputs/{step_id}")
def get_step_output(request: Request, step_id: str) -> dict:
    resolved_step_id, version = _parse_step_output_id(step_id)
    try:
        return request.app.state.plan_repo.load_step_output(resolved_step_id, version=version)
    except KeyError as exc:
        raise not_found("step output not found") from exc


def _parse_step_output_id(raw: str) -> tuple[str, int | None]:
    text = str(raw or "")
    step_id, sep, version_text = text.rpartition(":v")
    if not sep:
        return text, None
    if not step_id or not version_text.isdigit():
        return text, None
    return step_id, int(version_text)


_PLAN_LIST_MAX_LIMIT = 500


@router.get("/tasks/{task_id}/plans")
def list_task_plans(
    request: Request,
    task_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    """Plans belonging to a task (oldest first). Lets the right rail resume an
    existing task's plan; empty list (not 404) when the task has no plans yet.
    LT-13: limit/offset are optional -- omitting them preserves the prior
    full-history behavior for existing callers."""
    repo = request.app.state.plan_repo
    bounded_limit = None if limit is None else max(1, min(int(limit), _PLAN_LIST_MAX_LIMIT))
    bounded_offset = max(0, int(offset))
    plans = repo.list_plans_for_task(task_id, limit=bounded_limit, offset=bounded_offset)
    payload = {"plans": [_plan_payload(request, plan)["plan"] for plan in plans]}
    if bounded_limit is not None or bounded_offset:
        total = repo.count_plans_for_task(task_id)
        payload["total"] = total
        payload["limit"] = bounded_limit
        payload["offset"] = bounded_offset
        payload["has_more"] = bounded_offset + len(plans) < total
    return payload


@router.get("/plans/{plan_id}")
def get_plan(request: Request, plan_id: str) -> dict:
    return _load_plan_payload(request, plan_id)


@router.post("/plans/{plan_id}/confirm")
def confirm_plan(request: Request, plan_id: str) -> dict:
    try:
        request.app.state.plan_repo.confirm_plan(plan_id)
    except PlanNotFoundError as exc:
        raise not_found(str(exc)) from exc
    except (IllegalPlanTransition, ConflictError) as exc:
        raise conflict(str(exc)) from exc
    plan = _load_plan(request, plan_id)
    _dispatch_platform_hook(
        getattr(request.app.state, "hook_dispatcher", None),
        "plan.confirmed",
        {"task_id": plan.task_id, "plan_id": plan.id},
        task_id=plan.task_id,
    )
    return _plan_payload(request, plan)


@router.post("/plans/{plan_id}/run", status_code=202)
def run_plan(request: Request, plan_id: str, background_tasks: BackgroundTasks) -> dict:
    plan = _load_plan(request, plan_id)
    if plan.status not in {PlanStatus.CONFIRMED, PlanStatus.AWAITING_CONFIRM, PlanStatus.RUNNING}:
        raise conflict(f"plan is not runnable: {plan.status.value}")
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
        raise not_found("step not found")
    job_id = _start_plan_job(request, plan.task_id)
    try:
        request.app.state.plan_repo.confirm_step(step_id)
    except KeyError as exc:
        _fail_plan_job(_db_path(request), job_id, exc)
        raise not_found(str(exc)) from exc
    except ConflictError as exc:
        _fail_plan_job(_db_path(request), job_id, exc)
        raise conflict(str(exc)) from exc
    background_tasks.add_task(
        _run_plan_job,
        job_id,
        _db_path(request),
        request.app.state.plan_executor,
        plan_id,
    )
    return {"ok": True, "plan_id": plan_id, "step_id": step_id, "job_id": job_id}


@router.post("/plans/{plan_id}/steps/{step_id}/retry", status_code=202)
def retry_step(
    request: Request,
    plan_id: str,
    step_id: str,
    background_tasks: BackgroundTasks,
    body: RetryStepRequest | None = None,
) -> dict:
    plan = _load_plan(request, plan_id)
    job_id = _start_plan_job(request, plan.task_id)
    try:
        reset_step_ids = request.app.state.plan_repo.retry_failed_step(
            plan_id,
            step_id,
            inputs=None if body is None else body.inputs,
        )
    except KeyError as exc:
        _fail_plan_job(_db_path(request), job_id, exc)
        raise not_found("step not found") from exc
    except ConflictError as exc:
        _fail_plan_job(_db_path(request), job_id, exc)
        raise conflict(str(exc)) from exc
    background_tasks.add_task(
        _run_plan_job,
        job_id,
        _db_path(request),
        request.app.state.plan_executor,
        plan_id,
    )
    return {
        "ok": True,
        "plan_id": plan_id,
        "step_id": step_id,
        "reset_step_ids": reset_step_ids,
        "job_id": job_id,
    }


@router.post("/plans/{plan_id}/cancel")
def cancel_plan(request: Request, plan_id: str) -> dict:
    repo = request.app.state.plan_repo
    _load_plan(request, plan_id)
    try:
        repo.set_plan_status(plan_id, PlanStatus.CANCELLED)
    except PlanNotFoundError as exc:
        raise not_found(str(exc)) from exc
    except (IllegalPlanTransition, ConflictError) as exc:
        raise conflict(str(exc)) from exc
    # Cancellation is cooperative: the executor checkpoints the plan status
    # and finishes the exact job row that owns its execution lease. Jobs do not
    # currently carry a plan id, so ending the latest task-level plan job here
    # could cancel another plan and would release the task lock while the old
    # callback was still running.
    return _load_plan_payload(request, plan_id)


def _load_plan_payload(request: Request, plan_id: str) -> dict:
    return _plan_payload(request, _load_plan(request, plan_id))


def _load_plan(request: Request, plan_id: str):
    try:
        return request.app.state.plan_repo.load_plan(plan_id)
    except PlanNotFoundError as exc:
        raise not_found(str(exc)) from exc


def _plan_payload(request: Request, plan) -> dict:
    payload = plan_to_dict(plan)
    _attach_failure_envelopes(payload)
    _attach_running_step_started_at(request, payload, plan.id)
    payload["sub_agents"] = [
        _sub_agent_payload(sub)
        for sub in request.app.state.plan_repo.list_sub_agents_for_plan(plan.id)
    ]
    return {"plan": payload}


def _attach_running_step_started_at(request: Request, payload: dict, plan_id: str) -> None:
    """UX-1/REL-6: give the plan rail a ``started_at`` for the step currently
    RUNNING, sourced from plan_step_runs (already recorded per attempt), so the
    rail can reuse the validation stepper's formatStepElapsed() to show elapsed
    time instead of a plain spinner during a long driver-turn step."""
    steps = payload.get("steps") or []
    running_step_ids = {
        str(step.get("id") or "") for step in steps if step.get("status") == StepStatus.RUNNING.value
    }
    if not running_step_ids:
        return
    running_runs = request.app.state.plan_repo.list_running_step_runs(plan_id)
    started_at_by_step: dict[str, str] = {}
    for run in running_runs:
        step_id = str(run.get("step_id") or "")
        if step_id not in running_step_ids:
            continue
        started_at = str(run.get("started_at") or "")
        if not started_at:
            continue
        # ORDER BY started_at ASC in list_running_step_runs; keep the earliest
        # attempt's start time per step.
        started_at_by_step.setdefault(step_id, started_at)
    for step in steps:
        step_id = str(step.get("id") or "")
        if step_id in started_at_by_step:
            step["started_at"] = started_at_by_step[step_id]


def _attach_failure_envelopes(payload: dict) -> None:
    steps = payload.get("steps") or []
    for step in steps:
        if step.get("status") != StepStatus.FAILED.value:
            continue
        reset_steps = _downstream_step_ids(steps, str(step.get("id") or ""))
        detail = f"「{step.get('title') or step.get('id') or '步骤'}」失败:{step.get('error') or '执行中断。'}"
        step["failure_envelope"] = build_failure_envelope(
            plan_id=str(payload.get("id") or ""),
            step_id=str(step.get("id") or "") or None,
            run_seq=0,
            message=detail,
            step_inputs=step.get("inputs") if isinstance(step.get("inputs"), dict) else None,
            downstream_reset_steps=tuple(reset_steps),
            retryable=True,
        ).to_dict()


def _downstream_step_ids(steps: list[dict], root_id: str) -> list[str]:
    if not root_id:
        return []
    reset_ids = {root_id}
    changed = True
    while changed:
        changed = False
        for step in steps:
            step_id = str(step.get("id") or "")
            if not step_id or step_id in reset_ids:
                continue
            depends_on = {str(item) for item in step.get("depends_on") or []}
            if depends_on.intersection(reset_ids):
                reset_ids.add(step_id)
                changed = True
    return [
        str(step.get("id"))
        for step in sorted(steps, key=lambda item: (int(item.get("index") or 0), str(item.get("id") or "")))
        if str(step.get("id") or "") in reset_ids
    ]


def _sub_agent_payload(sub) -> dict:
    return {
        "id": sub.id,
        "parent_task_id": sub.parent_task_id,
        "parent_step_id": sub.parent_step_id,
        "scope": sub.scope,
        "granted_tools": [
            {"plugin": ref.plugin, "tool": ref.tool, "version": ref.version}
            for ref in sub.granted_tools
        ],
        "context_budget": sub.context_budget,
        "status": sub.status.value,
        "result_ref": sub.result_ref,
    }


def _task_context(task_id: str, body: CreatePlanRequest) -> dict:
    context = {"task_id": task_id}
    context.update(dict(body.task_context))
    context.update(dict(body.slots))
    return context


def _requested_tier(request: Request, tier_name: str | None):
    if tier_name:
        return resolve_tier(tier_name)
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return resolve_tier(None)
    return tier_from_settings(settings)


def _start_plan_job(request: Request, task_id: str) -> str:
    try:
        return TaskRepository(_db_path(request)).start_job(task_id, PLAN_JOB_KIND)
    except ConflictError as exc:
        raise conflict(ACTIVE_JOB_DETAIL) from exc


def _run_plan_job(job_id: str, db_path: Path, executor, plan_id: str) -> None:
    repo = TaskRepository(db_path)
    if not repo.mark_job_running(job_id):
        return
    try:
        with heartbeat_job(repo, job_id):
            result = executor.run(plan_id)
    except (ConflictError, IllegalPlanTransition) as exc:
        try:
            cancelled = (
                PlanRepository(db_path).load_plan(plan_id).status
                == PlanStatus.CANCELLED
            )
        except Exception:
            cancelled = False
        if cancelled:
            repo.finish_job(job_id, status="cancelled")
            return
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    except Exception as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    if result.status == PlanStatus.FAILED:
        status = "failed"
    elif result.status == PlanStatus.CANCELLED:
        status = "cancelled"
    else:
        status = "succeeded"
    repo.finish_job(job_id, status=status)


def _dispatch_platform_hook(
    hook_dispatcher,
    event: str,
    payload: dict,
    *,
    task_id: str,
) -> None:
    if hook_dispatcher is None:
        return
    try:
        hook_dispatcher.dispatch(event, payload, task_id=task_id)
    except Exception as exc:
        logger.warning("platform hook dispatch failed for %s/%s: %s", event, task_id, exc)


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
