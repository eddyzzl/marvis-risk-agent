from __future__ import annotations

from fastapi import APIRouter, Request

from marvis.api_task_helpers import get_task_or_404
from marvis.db import TaskRepository
from marvis.domain import TASK_TYPE_STRATEGY
from marvis.errors import conflict, unprocessable
from marvis.packs.strategy.candidate_lab_projection import (
    CandidateLabProjectionError,
    build_strategy_candidate_lab_projection,
)


router = APIRouter(prefix="/api", tags=["strategy-candidate-lab"])


@router.get("/tasks/{task_id}/strategy-candidate-lab")
def get_strategy_candidate_lab(task_id: str, request: Request) -> dict:
    settings = request.app.state.settings
    task = get_task_or_404(TaskRepository(settings.db_path), task_id)
    if task.task_type != TASK_TYPE_STRATEGY:
        raise unprocessable("strategy candidate lab requires a strategy task")
    try:
        return build_strategy_candidate_lab_projection(settings, task_id)
    except CandidateLabProjectionError as exc:
        raise conflict(
            "strategy candidate lab evidence verification failed"
        ) from exc


__all__ = ["router"]
