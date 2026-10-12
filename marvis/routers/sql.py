"""Private task-bound SQL source discovery, proposal and confirmation API."""
from fastapi import APIRouter, Request
from marvis.errors import bad_gateway, conflict, domain_error, forbidden, unprocessable
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from marvis.agent.sql_import import SqlProposalError, _secret, confirm_sql_import, propose_sql_import
from marvis.api_data_payloads import dataset_payload
from marvis.data.errors import DatasetTooLargeError
from marvis.data.sql_query import SqlQueryError
from marvis.data.sql_ingest import SqlQueryTimeoutError
from marvis.data.sql_sources import SqlSourceError, list_sql_sources
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.llm_outbound import OutboundContextError
from marvis.llm_settings import LLMSettingsError, resolve_llm_model
from marvis.repositories.data_workspace import DataWorkspaceRevisionConflict
from marvis.routers.data import _data_runtime, _data_workspace_payload, _require_task


router = APIRouter(prefix="/api/tasks/{task_id}/sql", tags=["sql"])


class SqlProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: StrictStr = Field(min_length=1, max_length=64)
    expected_revision: StrictInt = Field(ge=0)
    query: StrictStr | None = Field(default=None, max_length=16384)
    question: StrictStr | None = Field(default=None, max_length=16384)
    model_id: StrictStr | None = Field(default=None, max_length=128)


class SqlConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    proposal_token: StrictStr = Field(min_length=1, max_length=65536)


def _actor(request, task_id):
    _require_task(request, task_id)
    actor = getattr(getattr(request.state, "local_principal", None), "id", None)
    if not getattr(request.state, "private_client_authenticated", False) or not actor:
        raise forbidden({"code": "sql_private_principal_required", "message": "请在已认证的本地会话中使用 SQL 数据源。"})
    return actor


def _llm(settings, model_id):
    return OpenAICompatibleLLMClient(resolve_llm_model(settings.workspace, model_id))


def _failure(exc):
    if isinstance(exc, DataWorkspaceRevisionConflict):
        return conflict({"code": "sql_workspace_revision_conflict", "message": str(exc)})
    if isinstance(exc, (SqlSourceError, SqlQueryError, SqlProposalError)):
        status = 409 if getattr(exc, "code", "") in {"sql_source_changed", "sql_proposal_expired", "sql_import_in_progress"} else getattr(exc, "status", 422)
        detail = {"code": exc.code, "message": str(exc)}
        if getattr(exc, "execution_consumed", False):
            detail.update({"execution_consumed": True, "next_action": exc.next_action})
        return domain_error(status, detail)
    if isinstance(exc, OutboundContextError):
        return unprocessable({"code": "sql_instruction_not_admitted", "message": str(exc)})
    if isinstance(exc, DatasetTooLargeError):
        return unprocessable({"code": "sql_result_row_limit", "message": "SQL 结果超过当前行数上限，未截断或导入；请增加筛选条件。"})
    if isinstance(exc, SqlQueryTimeoutError):
        return domain_error(408, {"code": "sql_query_timeout", "message": str(exc)})
    if isinstance(exc, LLMClientError):
        return bad_gateway({"code": "sql_model_unavailable", "message": "查询编译尚未完成，请重试或使用手动 SELECT；未执行查询。"})
    if isinstance(exc, LLMSettingsError):
        return unprocessable({"code": "sql_model_configuration_required", "message": "请先在设置中配置可用的大模型，或使用手动 SELECT；未执行查询。"})
    return unprocessable({"code": "sql_request_failed", "message": "SQL 请求未完成，未更新活动数据；请在本地核对来源权限与查询条件。"})


@router.get("/sources")
def sources(task_id: str, request: Request):
    actor = _actor(request, task_id)
    settings = request.app.state.settings
    try:
        return {"sources": list_sql_sources(settings, actor, secret=_secret(settings)), "credentials_local_only": True}
    except Exception as exc:
        raise _failure(exc) from None


@router.post("/proposals")
def propose(task_id: str, payload: SqlProposalRequest, request: Request):
    actor = _actor(request, task_id)
    settings = request.app.state.settings
    try:
        client = _llm(settings, payload.model_id) if payload.question is not None else None
        return propose_sql_import(settings, task_id=task_id, actor_id=actor, source_id=payload.source_id,
            expected_revision=payload.expected_revision, query=payload.query, question=payload.question, client=client)
    except Exception as exc:
        raise _failure(exc) from None


@router.post("/confirm")
def confirm(task_id: str, payload: SqlConfirmRequest, request: Request):
    actor = _actor(request, task_id)
    settings = request.app.state.settings
    try:
        _, _, registry, _ = _data_runtime(request)
        dataset, snapshot, replayed = confirm_sql_import(settings, task_id=task_id, actor_id=actor, token=payload.proposal_token, registry=registry)
        return {"status": "completed", "dataset": dataset_payload(dataset), "workspace": _data_workspace_payload(snapshot),
            "replayed": replayed, "evidence": {"content_hash": dataset.content_hash, "row_count": dataset.row_count}}
    except Exception as exc:
        raise _failure(exc) from None
