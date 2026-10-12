"""Authenticated, task-owned confirmation of explicit field risk semantics."""
from fastapi import APIRouter, Request
from marvis.errors import conflict, forbidden, precondition_failed
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from marvis.api_schemas import StrictCanonicalNonEmptyStr, StrictSha256
from marvis.api_task_helpers import get_task_or_404
from marvis.data.backend import DataBackend
from marvis.data.field_directions import confirm_field_directions, confirmed_field_directions
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.repositories.data_workspace import DataWorkspaceRepository, DataWorkspaceRevisionConflict
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.tasks import TaskRepository

router = APIRouter(prefix="/api/tasks/{task_id}/data-workspace/field-directions", tags=["data-field-directions"])


class Confirm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    expected_revision: StrictInt = Field(ge=0)
    expected_semantic_mapping_hash: StrictSha256
    dataset_id: StrictCanonicalNonEmptyStr
    dataset_content_hash: StrictSha256
    directions: dict[StrictCanonicalNonEmptyStr, StrictCanonicalNonEmptyStr] = Field(min_length=1, max_length=128)


def _runtime(request, task_id):
    settings = request.app.state.settings
    get_task_or_404(TaskRepository(settings.db_path), task_id)
    actor = getattr(getattr(request.state, "local_principal", None), "id", None)
    if not getattr(request.state, "private_client_authenticated", False) or not actor:
        raise forbidden({"code": "field_direction_principal_required",
            "message": "请在已认证的本地会话中确认字段风险方向。"})
    return settings, actor


def _projection(settings, task_id):
    snapshot = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
    rows = []
    if snapshot.active_dataset_id:
        registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
        dataset = registry.get(snapshot.active_dataset_id)
        if dataset.task_id != task_id or dataset.content_hash != snapshot.active_dataset_content_hash:
            raise ValueError("活动数据来源已变化。")
        registry.resolve_verified_path(dataset.id)
        rows = confirmed_field_directions(snapshot, dataset)
    return {"task_id": task_id, "revision": snapshot.revision,
        "semantic_mapping_hash": data_semantic_mapping_hash(snapshot.semantic_mapping),
        "dataset_id": snapshot.active_dataset_id, "dataset_content_hash": snapshot.active_dataset_content_hash,
        "fields": rows, "external_requests": 0}


def _failure(exc):
    if isinstance(exc, DataWorkspaceRevisionConflict):
        return precondition_failed({"code": "field_direction_stale_context", "message": str(exc)})
    return conflict({"code": "field_direction_source_unavailable",
        "message": "字段或来源无法验证，未保存更改。请刷新数据语义后明确选择物理字段和方向。"})


@router.get("")
def current(task_id: str, request: Request):
    settings, _ = _runtime(request, task_id)
    try:
        return _projection(settings, task_id)
    except Exception as exc:
        raise _failure(exc) from None


@router.post("/confirm")
def confirm(task_id: str, payload: Confirm, request: Request):
    settings, actor = _runtime(request, task_id)
    try:
        confirm_field_directions(settings, task_id=task_id, actor_id=actor, **payload.model_dump())
        return {"status": "confirmed", **_projection(settings, task_id)}
    except Exception as exc:
        raise _failure(exc) from None
