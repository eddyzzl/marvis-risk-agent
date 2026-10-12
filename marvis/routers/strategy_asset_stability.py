"""Task-bound manual API for exact frozen asset stability evidence."""

from types import SimpleNamespace

from fastapi import APIRouter, Request

from marvis.api_task_helpers import get_task_or_404
from marvis.domain import TASK_TYPE_STRATEGY
from marvis.errors import conflict, unprocessable
from marvis.packs.strategy import tools
from marvis.packs.strategy.asset_stability_tools import (
    candidate_asset_stability_catalog,
    run_freeze_candidate_asset_stability,
    run_measure_candidate_asset_stability,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.repositories.tasks import TaskRepository
from marvis.packs.strategy.asset_stability_report_service import (
    prepare_asset_stability_report,
    publish_asset_stability_report,
)

router = APIRouter(
    prefix="/api/tasks/{task_id}/strategy-asset-stability",
    tags=["strategy-asset-stability"],
)


def _context(task_id, request):
    settings = request.app.state.settings
    task = get_task_or_404(TaskRepository(settings.db_path), task_id)
    if task.task_type != TASK_TYPE_STRATEGY:
        raise unprocessable("independent candidate stability requires a strategy task")
    ctx = SimpleNamespace(
        task_id=task_id,
        workspace=settings.workspace,
        datasets_root=settings.datasets_dir,
    )
    return ctx, tools._runtime(ctx)


@router.get("")
def catalogue(task_id: str, request: Request):
    ctx, runtime = _context(task_id, request)
    try:
        return candidate_asset_stability_catalog(runtime, task_id=ctx.task_id)
    except StrategyError as exc:
        raise conflict(str(exc)) from exc


@router.post("/freeze")
def freeze(task_id: str, payload: dict, request: Request):
    ctx, runtime = _context(task_id, request)
    try:
        return run_freeze_candidate_asset_stability(payload, ctx, runtime)
    except (StrategyError, KeyError, ValueError, TypeError) as exc:
        raise unprocessable(str(exc)) from exc


@router.post("/measure")
def measure(task_id: str, payload: dict, request: Request):
    ctx, runtime = _context(task_id, request)
    try:
        return run_measure_candidate_asset_stability(payload, ctx, runtime)
    except (StrategyError, KeyError, ValueError, TypeError) as exc:
        raise conflict(str(exc)) from exc


@router.post("/report-proposal")
def report_proposal(task_id: str, payload: dict, request: Request):
    ctx, runtime = _context(task_id, request)
    task = get_task_or_404(TaskRepository(runtime.settings.db_path), task_id)
    try:
        return prepare_asset_stability_report(payload, task=task, runtime=runtime)
    except (StrategyError, KeyError, ValueError, TypeError) as exc:
        raise conflict(str(exc)) from exc


@router.post("/report")
def report(task_id: str, payload: dict, request: Request):
    ctx, runtime = _context(task_id, request)
    try:
        return publish_asset_stability_report(payload, ctx, runtime)
    except (StrategyError, KeyError, ValueError, TypeError) as exc:
        raise conflict(str(exc)) from exc
