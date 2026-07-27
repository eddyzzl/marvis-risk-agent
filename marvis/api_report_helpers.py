from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from marvis.errors import conflict

from marvis.agent.service import agent_conclusions_confirmed
from marvis.db import TaskRepository
from marvis.domain import TaskRecord


@dataclass(frozen=True)
class DriverReportArtifact:
    report_id: str
    path: Path
    experiment_id: str = ""
    recipe: str = ""


def require_confirmed_agent_conclusions(repo: TaskRepository, task: TaskRecord) -> None:
    if task.run_mode != "agent":
        return
    values, _ = repo.get_report_values(task.id)
    if agent_conclusions_confirmed(values):
        return
    raise conflict("请先确认三段报告结论，确认后将生成 Word 报告")


def driver_report_id(plan_id: str, step_id: str, index: int) -> str:
    """Return an opaque, task-scoped report handle.

    The handle contains no filesystem data.  The download route resolves it
    against persisted step output and re-applies the task-output containment
    check before serving a file.
    """
    raw = f"{plan_id}\0{step_id}\0{int(index)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def driver_report_download_metadata(
    *,
    plan_id: str,
    task_id: str,
    step_id: str,
    output: dict[str, Any],
    default_label: str,
) -> list[dict[str, str]]:
    candidates = _driver_report_candidates(output)
    multiple = len(candidates) > 1
    items: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        report_id = driver_report_id(plan_id, step_id, index)
        recipe = str(candidate.get("recipe") or "").strip()
        experiment_id = str(candidate.get("experiment_id") or "").strip()
        label_detail = recipe or experiment_id
        label = (
            f"下载 {label_detail} 模型报告"
            if multiple and label_detail
            else default_label
        )
        items.append({
            "report_id": report_id,
            "label": label,
            "download_url": (
                f"/api/tasks/{task_id}/driver-reports/{report_id}/download"
            ),
            "experiment_id": experiment_id,
            "recipe": recipe,
        })
    return items


def latest_driver_report_artifacts(state, task_id: str) -> list[DriverReportArtifact]:
    """Resolve the newest report-producing step to task-contained artifacts."""
    plan_repo = state.plan_repo
    outputs_dir = (Path(state.settings.tasks_dir) / task_id / "outputs").resolve()
    for plan in reversed(plan_repo.list_plans_for_task(task_id)):
        for step in sorted(
            plan.steps,
            key=lambda item: -(int(getattr(item, "index", 0) or 0)),
        ):
            try:
                output = plan_repo.load_step_output(step.id)
            except KeyError:
                continue
            candidates = _driver_report_candidates(output)
            if not candidates:
                continue
            artifacts: list[DriverReportArtifact] = []
            for index, candidate in enumerate(candidates):
                raw_path = str(candidate.get("report_path") or "").strip()
                if not raw_path:
                    continue
                path = Path(raw_path).resolve()
                try:
                    path.relative_to(outputs_dir)
                except ValueError:
                    continue
                artifacts.append(DriverReportArtifact(
                    report_id=driver_report_id(plan.id, step.id, index),
                    path=path,
                    experiment_id=str(candidate.get("experiment_id") or ""),
                    recipe=str(candidate.get("recipe") or ""),
                ))
            if artifacts:
                return artifacts
    return []


def driver_report_artifact(state, task_id: str, report_id: str) -> DriverReportArtifact | None:
    expected = str(report_id or "").strip()
    if not expected:
        return None
    return next(
        (
            artifact
            for artifact in latest_driver_report_artifacts(state, task_id)
            if artifact.report_id == expected
        ),
        None,
    )


def latest_driver_report_path(state, task_id: str):
    """Return the latest plan-produced report_path inside task outputs, if any."""
    artifacts = latest_driver_report_artifacts(state, task_id)
    return artifacts[0].path if artifacts else None


def _driver_report_candidates(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, dict):
        return []
    reports = [
        dict(item)
        for item in (output.get("reports") or [])
        if isinstance(item, dict) and str(item.get("report_path") or "").strip()
    ]
    if reports:
        return reports
    report_path = str(output.get("report_path") or "").strip()
    return [{"report_path": report_path}] if report_path else []
