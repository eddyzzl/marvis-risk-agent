from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from marvis.db import TaskRepository
from marvis.domain import FileArtifact, FileRole, TaskStatus
from marvis.execution_environment import load_execution_environment
from marvis.files import scan_source_dir
from marvis.pipeline import PipelineSettings
from marvis.packs.v1_compat.contracts import V1TaskContext
from marvis.settings import build_settings


REQUIRED_ROLES = (
    FileRole.NOTEBOOK,
    FileRole.SAMPLE,
    FileRole.MODEL_PMML,
    FileRole.DATA_DICTIONARY,
)


def load_v1_task_context(ctx, task_id: str) -> V1TaskContext:
    settings = build_settings(ctx.workspace)
    repo = TaskRepository(settings.db_path)
    try:
        task = repo.get_task(task_id)
    except KeyError as exc:
        raise KeyError(f"Task not found: {task_id}") from exc
    task_dir = settings.tasks_dir / task_id
    pipeline_settings = PipelineSettings(
        workspace=settings.workspace,
        db_path=settings.db_path,
        report_template_path=settings.report_template_path,
        feature_columns=list(task.feature_columns),
        notebook_kernel_name=load_execution_environment(settings.workspace).kernel_name,
    )
    return V1TaskContext(
        task_id=task_id,
        workspace=settings.workspace,
        settings=settings,
        pipeline_settings=pipeline_settings,
        repo=repo,
        task=task,
        task_dir=task_dir,
        execution_dir=task_dir / "execution",
        outputs_dir=task_dir / "outputs",
        images_dir=task_dir / "images",
    )


def scan_materials(context: V1TaskContext) -> list[FileArtifact]:
    return scan_source_dir(Path(context.task.source_dir))


def material_payloads(context: V1TaskContext, artifacts: list[FileArtifact]) -> list[dict]:
    source_dir = Path(context.task.source_dir).resolve()
    return [
        {
            "role": artifact.role.value,
            "path": safe_relative_path(artifact.path, source_dir),
            "name": artifact.path.name,
            "size_bytes": int(artifact.size_bytes),
            "sha256": artifact.sha256,
        }
        for artifact in artifacts
    ]


def material_checks(artifacts: list[FileArtifact]) -> list[dict]:
    by_role = {role: [artifact for artifact in artifacts if artifact.role == role] for role in REQUIRED_ROLES}
    checks = []
    for role in REQUIRED_ROLES:
        matches = by_role[role]
        if not matches:
            checks.append({
                "name": role.value,
                "status": "missing",
                "detail": f"{role.value} material is missing",
            })
        elif len(matches) > 1:
            checks.append({
                "name": role.value,
                "status": "ambiguous",
                "detail": f"{len(matches)} {role.value} materials found",
            })
        else:
            checks.append({
                "name": role.value,
                "status": "ok",
                "detail": matches[0].path.name,
            })
    return checks


def update_scan_status(context: V1TaskContext) -> None:
    if context.task.status in {TaskStatus.CREATED, TaskStatus.SCANNED, TaskStatus.FAILED}:
        context.repo.update_status(
            context.task_id,
            TaskStatus.SCANNED,
            message="source scanned",
            expected={TaskStatus.CREATED, TaskStatus.SCANNED, TaskStatus.FAILED},
        )


def notebook_cell_count(context: V1TaskContext) -> int:
    payload = read_json(context.execution_dir / "notebook_steps.json")
    cells = payload.get("cells") if isinstance(payload, dict) else None
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if isinstance(cells, list):
        return len(cells)
    if isinstance(steps, list):
        return len(steps)
    return 0


def validation_metric_summary(context: V1TaskContext) -> dict:
    results_path = context.outputs_dir / "validation_results.json"
    payload = read_json(results_path)
    if not payload:
        raise RuntimeError("metrics output missing: validation_results.json")
    overall = ((payload.get("effectiveness") or {}).get("overall") or [])
    selected = _preferred_overall_row(overall)
    if not selected:
        raise RuntimeError("metrics output missing effectiveness.overall")
    reproducibility = (payload.get("reproducibility") or {}).get("summary") or {}
    status = context.repo.get_task(context.task_id).status
    return {
        "task_id": context.task_id,
        "status": _metrics_status(status),
        "ks": _float_or_zero(selected.get("ks")),
        "auc": _float_or_zero(selected.get("auc")),
        "psi": _optional_float(selected.get("psi_vs_train")),
        "score_consistency_passed": reproducibility.get("status") == "pass",
        "validation_results_ref": artifact_ref(context, context.outputs_dir / "validation_results.json"),
    }


def report_artifacts(context: V1TaskContext) -> list[dict]:
    artifacts = []
    excel_path = context.outputs_dir / "validation.xlsx"
    word_path = context.outputs_dir / "validation_report.docx"
    if excel_path.exists():
        artifacts.append(_artifact_payload("excel", context, excel_path))
    if word_path.exists():
        artifacts.append(_artifact_payload("word", context, word_path))
    return artifacts


def artifact_ref(context: V1TaskContext, path: Path) -> str:
    return f"artifact:{safe_relative_path(path, context.workspace)}"


def safe_relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise PermissionError(f"path escapes allowed root: {path}") from exc


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _preferred_overall_row(rows: list) -> dict:
    if not isinstance(rows, list):
        return {}
    for preferred in ("oot", "test", "validation", "train"):
        for row in rows:
            if isinstance(row, dict) and str(row.get("split") or "").lower() == preferred:
                return row
    for row in rows:
        if isinstance(row, dict):
            return row
    return {}


def _metrics_status(status: TaskStatus) -> str:
    if status is TaskStatus.FAILED:
        return "failed"
    if status is TaskStatus.REVIEW_REQUIRED:
        return "review_required"
    return "writing_artifacts"


def _float_or_zero(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return _float_or_zero(value)


def _artifact_payload(kind: str, context: V1TaskContext, path: Path) -> dict:
    endpoint = "analysis" if kind == "excel" else "report"
    return {
        "kind": kind,
        "path": safe_relative_path(path, context.workspace),
        "download_url": f"/api/tasks/{context.task_id}/{endpoint}/download",
    }
