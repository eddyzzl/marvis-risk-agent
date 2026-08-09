from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
from typing import Any

from marvis.errors import conflict

from marvis.agent.service import agent_conclusions_confirmed
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskRecord
from marvis.files import sha256_file
from marvis.repositories.task_artifacts import TaskArtifactRepository


_REPORT_PRODUCING_TOOLS = frozenset({
    ("analysis", "portfolio_report"),
    ("feature", "generate_feature_report"),
    ("modeling", "generate_model_report"),
    ("modeling", "generate_model_reports"),
    ("risk_analysis", "generate_risk_analysis_report"),
    ("strategy", "build_report_bundle_v2"),
    ("strategy", "render_challenger_report"),
    ("strategy", "render_monitoring_report"),
    ("v1_compat", "render_reports"),
})


@dataclass(frozen=True)
class DriverReportArtifact:
    report_id: str
    path: Path
    experiment_id: str = ""
    recipe: str = ""
    artifact_id: str = ""
    content_hash: str = ""


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
    task_root = (Path(state.settings.tasks_dir) / task_id).resolve()
    outputs_dir = (task_root / "outputs").resolve()
    for plan in reversed(plan_repo.list_plans_for_task(task_id)):
        for step in sorted(
            plan.steps,
            key=lambda item: -(int(getattr(item, "index", 0) or 0)),
        ):
            if not _is_report_step(step):
                continue
            # Any newer report-producing attempt is a freshness barrier.  A
            # failed, running, cancelled-at-plan-level, or otherwise incomplete
            # attempt must never be hidden by serving an older successful file
            # as though it were the latest evidence.
            if _step_status(step) != "done":
                return []
            try:
                output = plan_repo.load_step_output(step.id)
            except KeyError:
                # A completed, explicit report step without its output is
                # corrupt evidence. Never hide it by serving an older plan.
                return []
            if not isinstance(output, dict):
                return []
            candidates = _driver_report_candidates(output)
            if not candidates:
                return []
            artifacts: list[DriverReportArtifact] = []
            for index, candidate in enumerate(candidates):
                raw_path = str(candidate.get("report_path") or "").strip()
                if not raw_path:
                    continue
                declared_path = Path(raw_path)
                try:
                    path = declared_path.resolve()
                except (OSError, RuntimeError):
                    continue
                try:
                    path.relative_to(outputs_dir)
                except ValueError:
                    registered = _registered_portfolio_report(
                        state,
                        task_id=task_id,
                        task_root=task_root,
                        step=step,
                        candidate=candidate,
                        declared_path=declared_path,
                        resolved_path=path,
                    )
                    if registered is None:
                        continue
                else:
                    registered = None
                artifacts.append(DriverReportArtifact(
                    report_id=driver_report_id(plan.id, step.id, index),
                    path=path,
                    experiment_id=str(candidate.get("experiment_id") or ""),
                    recipe=str(candidate.get("recipe") or ""),
                    artifact_id=(
                        str(registered.get("id") or "")
                        if registered is not None
                        else ""
                    ),
                    content_hash=(
                        str(registered.get("content_hash") or "")
                        if registered is not None
                        else ""
                    ),
                ))
            if artifacts:
                return artifacts
            # A newer step explicitly declared report candidates, but none of
            # them survived containment / artifact / content authentication.
            # Falling through to an older plan would serve stale evidence while
            # hiding corruption in the latest result, so fail closed here.
            return []
    return []


def _is_report_step(step: Any) -> bool:
    tool_ref = getattr(step, "tool_ref", None)
    key = (
        str(getattr(tool_ref, "plugin", "") or ""),
        str(getattr(tool_ref, "tool", "") or ""),
    )
    return key in _REPORT_PRODUCING_TOOLS


def _step_status(step: Any) -> str:
    raw_status = getattr(step, "status", None)
    status = getattr(raw_status, "value", raw_status)
    return str(status or "").strip().lower()


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
    return [dict(output)] if report_path else []


def _registered_portfolio_report(
    state,
    *,
    task_id: str,
    task_root: Path,
    step,
    candidate: dict[str, Any],
    declared_path: Path,
    resolved_path: Path,
) -> dict[str, Any] | None:
    """Resolve the one governed report kind allowed outside ``outputs``.

    Portfolio reports predate the shared ``outputs`` convention and live under
    ``tasks/<task>/portfolio``.  They are admitted only through their immutable
    task-artifact identity; arbitrary task-owned files remain unavailable.
    """

    if step.tool_ref.plugin != "analysis" or step.tool_ref.tool != "portfolio_report":
        return None
    if declared_path.is_symlink() or resolved_path.suffix.lower() != ".xlsx":
        return None
    try:
        resolved_path.relative_to(task_root)
    except ValueError:
        return None
    if not resolved_path.is_file():
        return None

    artifact_id = str(candidate.get("artifact_id") or "").strip()
    expected_hash = str(candidate.get("artifact_content_hash") or "").strip().lower()
    if not artifact_id or len(expected_hash) != 64:
        return None
    if any(character not in "0123456789abcdef" for character in expected_hash):
        return None

    record = TaskArtifactRepository(state.settings.db_path).get_for_task(
        task_id,
        artifact_id,
    )
    if record is None:
        return None
    if (
        record.get("kind") != "portfolio_report_xlsx"
        or record.get("origin_tool") != "analysis.portfolio_report"
        or not hmac.compare_digest(str(record.get("content_hash") or ""), expected_hash)
    ):
        return None
    try:
        registered_path = Path(str(record.get("path") or "")).resolve()
    except (OSError, RuntimeError):
        return None
    if registered_path != resolved_path:
        return None
    try:
        actual_hash = sha256_file(resolved_path)
    except OSError:
        return None
    if not hmac.compare_digest(actual_hash, expected_hash):
        return None
    return record
