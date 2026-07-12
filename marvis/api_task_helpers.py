from __future__ import annotations

import logging
import os
from pathlib import Path
import re

from marvis.errors import conflict, not_found, unprocessable

from marvis.db import TaskRepository
from marvis.domain import TaskRecord
from marvis.safe_paths import assert_within


logger = logging.getLogger(__name__)
MODEL_ID_RE = re.compile(r"^[\w一-鿿\- ]{1,64}$", re.UNICODE)
ACTIVE_JOB_DETAIL = "task already has an active stage"


def get_task_or_404(repo: TaskRepository, task_id: str) -> TaskRecord:
    try:
        return repo.get_task(task_id)
    except KeyError as exc:
        raise not_found(f"Task not found: {task_id}") from exc


def reject_if_task_has_active_job(repo: TaskRepository, task_id: str) -> None:
    if repo.task_has_active_job(task_id):
        raise conflict(ACTIVE_JOB_DETAIL)


def dispatch_platform_hook(
    hook_dispatcher,
    event: str | None,
    payload: dict,
    *,
    task_id: str | None,
) -> None:
    if hook_dispatcher is None or not event or not task_id:
        return
    try:
        hook_dispatcher.dispatch(event, payload, task_id=task_id)
    except Exception as exc:
        logger.warning("platform hook dispatch failed for %s/%s: %s", event, task_id, exc)


def task_hook_payload(task: TaskRecord) -> dict:
    payload = {
        "task_id": task.id,
        "task_type": task.task_type,
        "validation_workflow_version": task.validation_workflow_version,
        "status": task.status.value,
        "run_mode": task.run_mode,
        "algorithm": task.algorithm,
    }
    if getattr(task, "target_type", ""):
        payload["target_type"] = task.target_type
    if getattr(task, "sample_weight_col", ""):
        payload["sample_weight_col"] = task.sample_weight_col
    return payload


def validate_model_identifier(field_name: str, value: str) -> None:
    if not MODEL_ID_RE.match(value):
        raise unprocessable(f"{field_name} contains illegal characters")


def normalize_source_dir(source_dir: str, settings) -> Path:
    resolved = Path(source_dir).expanduser().resolve()
    allowed_roots = allowed_material_roots(settings)
    if not any(path_is_within(root, resolved) for root in allowed_roots):
        allowed = "、".join(str(root) for root in allowed_roots)
        raise unprocessable(
            f"source_dir must be under an allowed material root: {allowed}. "
            "Set RMC_MATERIAL_ROOTS to allow another local material directory."
        )
    return resolved


def allowed_material_roots(settings) -> tuple[Path, ...]:
    roots = [settings.workspace, Path.home()]
    extra_roots = os.environ.get("RMC_MATERIAL_ROOTS", "")
    roots.extend(Path(raw).expanduser() for raw in extra_roots.split(os.pathsep) if raw)
    resolved: list[Path] = []
    for root in roots:
        candidate = root.resolve()
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def path_is_within(root: Path, candidate: Path) -> bool:
    try:
        assert_within(root, candidate)
    except PermissionError:
        return False
    return True


def normalized_capability_tier(value: str | None) -> str:
    """Return a known capability tier, or empty so settings provide the fallback."""
    from marvis.orchestrator.capability import TIERS

    name = str(value or "").strip().lower()
    return name if name in TIERS else ""


def normalized_target_type(value: str | None) -> str:
    name = str(value or "").strip().lower()
    return name if name in {"binary", "continuous", "multiclass"} else ""
