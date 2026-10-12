"""Local, authenticated history mapping preview and explicit confirmation."""
from fastapi import APIRouter, Query, Request
from marvis.errors import conflict, domain_error, forbidden, unprocessable
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from marvis.api_task_helpers import get_task_or_404
from marvis.packs.strategy.history_mapping import HistoryMappingError
from marvis.repositories.strategy_history_mapping import StrategyHistoryMappingRepository
from marvis.repositories.tasks import TaskRepository


router = APIRouter(prefix="/api/tasks/{task_id}/strategy/history-mapping", tags=["strategy-history-mapping"])


class Head(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: StrictInt = Field(ge=0)
    artifact_id: StrictStr | None
    content_hash: StrictStr | None


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_head: Head
    bindings: list[dict] | None = Field(default=None, max_length=128)
    instruction: StrictStr | None = Field(default=None, max_length=12000)
    source_id: StrictStr | None = Field(default=None, max_length=128)


class Confirm(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    proposal_token: StrictStr = Field(min_length=1, max_length=65536)


def _runtime(request, task_id):
    settings = request.app.state.settings
    task = get_task_or_404(TaskRepository(settings.db_path), task_id)
    if task.task_type != "strategy":
        raise unprocessable({"code": "history_strategy_task_required", "message": "请在策略任务中确认历史材料映射。"})
    actor = getattr(getattr(request.state, "local_principal", None), "id", None)
    if not getattr(request.state, "private_client_authenticated", False) or not actor:
        raise forbidden({"code": "history_private_principal_required", "message": "请在已认证的本地会话中读取和确认历史材料。"})
    return StrategyHistoryMappingRepository(settings), actor


def _failure(exc):
    if isinstance(exc, HistoryMappingError):
        return domain_error(exc.status, {"code": exc.code, "message": str(exc)})
    return conflict({"code": "history_evidence_verification_failed", "message": "历史材料或映射无法验证，未保存更改；请恢复原登记材料，或重新登记并确认。"})


@router.get("")
def current(task_id: str, request: Request):
    repository, _ = _runtime(request, task_id)
    try:
        mapping = repository.current(task_id)
        return {"head": repository.head(mapping), "mapping": mapping}
    except Exception as exc:
        raise _failure(exc) from None


@router.get("/sources")
def sources(task_id: str, request: Request):
    repository, _ = _runtime(request, task_id)
    try:
        return {"sources": repository.sources(task_id), "external_requests": 0}
    except Exception as exc:
        raise _failure(exc) from None


@router.get("/sources/{source_id}/cells")
def cells(task_id: str, source_id: str, request: Request, offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=200)):
    repository, _ = _runtime(request, task_id)
    try:
        return repository.cells(task_id, source_id, offset=offset, limit=limit)
    except Exception as exc:
        raise _failure(exc) from None


@router.post("/proposals")
def propose(task_id: str, payload: Proposal, request: Request):
    repository, actor = _runtime(request, task_id)
    try:
        return repository.propose(task_id=task_id, actor_id=actor, expected_head=payload.expected_head.model_dump(),
            bindings=payload.bindings, instruction=payload.instruction, source_id=payload.source_id)
    except Exception as exc:
        raise _failure(exc) from None


@router.post("/confirm")
def confirm(task_id: str, payload: Confirm, request: Request):
    repository, actor = _runtime(request, task_id)
    try:
        mapping = repository.confirm(task_id=task_id, actor_id=actor, token=payload.proposal_token)
        return {"status": "confirmed", "head": repository.head(mapping), "mapping": mapping,
                "next_action": "已保存来源映射。请刷新项目现状与历史材料步骤，再生成使用该版本的报告。"}
    except Exception as exc:
        raise _failure(exc) from None
