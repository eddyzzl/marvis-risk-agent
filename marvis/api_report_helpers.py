from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from marvis.agent.service import agent_conclusions_confirmed
from marvis.db import TaskRepository
from marvis.domain import TaskRecord


def require_confirmed_agent_conclusions(repo: TaskRepository, task: TaskRecord) -> None:
    if task.run_mode != "agent":
        return
    values, _ = repo.get_report_values(task.id)
    if agent_conclusions_confirmed(values):
        return
    raise HTTPException(
        status_code=409,
        detail="请先确认三段报告结论，确认后将生成 Word 报告",
    )


def latest_driver_report_path(state, task_id: str):
    """Return the latest plan-produced report_path inside task outputs, if any."""
    plan_repo = state.plan_repo
    outputs_dir = (Path(state.settings.tasks_dir) / task_id / "outputs").resolve()
    for plan in reversed(plan_repo.list_plans_for_task(task_id)):
        for step in sorted(plan.steps, key=lambda step: -(int(getattr(step, "index", 0) or 0))):
            try:
                output = plan_repo.load_step_output(step.id)
            except KeyError:
                continue
            raw = (output or {}).get("report_path")
            if not raw:
                continue
            path = Path(str(raw)).resolve()
            try:
                path.relative_to(outputs_dir)
            except ValueError:
                continue
            return path
    return None
