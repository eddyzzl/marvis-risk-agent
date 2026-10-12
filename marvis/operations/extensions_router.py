"""Authenticated product entry points for typed periodic operations."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from marvis.db_schema import connect
from marvis.institution.policy import require_human_principal, authorize
from marvis.operations.extension_bindings import (
    FeedbackImport,
    PortfolioBinding,
    DecisionFeedbackBinding,
)
from marvis.operations.dispositions import (
    DispositionRequest,
    DispositionConsumed,
    apply_disposition,
    reconcile_disposition,
)
from marvis.operations.schedule_proposals import propose_schedule, confirm_schedule
from marvis.operations.repository import ScheduleRevisionConflict
from marvis.operations.http_schemas import SchedulePayload
from marvis.data.errors import DataLayerError
from marvis.operations.decision_feedback import import_feedback


def _require_role(*args):
    from marvis.operations.router import _require_role as require

    return require(*args)


def _runtime(request):
    from marvis.operations.router import _runtime as runtime

    return runtime(request)


def _schedule_contract(payload):
    from marvis.operations.router import _schedule_contract as contract

    return contract(payload)


router = APIRouter()


class ProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: StrictInt = Field(ge=0)
    instruction: str = Field(default="", max_length=16_000)
    schedule: SchedulePayload


class ProposalConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def _human(request):
    require_human_principal(request)
    return _require_role(request, "maker", "admin")


def _reject(exc):
    if isinstance(exc, KeyError):
        raise HTTPException(
            404, detail={"code": "operations_evidence_not_found"}
        ) from exc
    if isinstance(exc, (ScheduleRevisionConflict, DispositionConsumed)):
        raise HTTPException(
            409,
            detail={
                "code": "operations_reviewed_evidence_changed_or_consumed",
                "next_action": "刷新已确认合同或核对原生执行证据后再作决定",
            },
        ) from exc
    raise HTTPException(
        422,
        detail={
            "code": "operations_contract_or_source_invalid",
            "next_action": "补齐明确的来源、时间、统计口径和授权；本请求未被接受",
        },
    ) from exc


@router.get("/extensions/contracts")
def contracts(request: Request):
    _require_role(request, "maker", "checker", "admin")
    return {
        "proposal_schema": ProposalRequest.model_json_schema(),
        "feedback_import_schema": FeedbackImport.model_json_schema(),
        "disposition_schema": DispositionRequest.model_json_schema(),
        "portfolio_binding_schema": PortfolioBinding.model_json_schema(),
        "feedback_binding_schema": DecisionFeedbackBinding.model_json_schema(),
        "language_scope": "显式 every/每 N 分钟、小时、天、月与一个含时区 ISO 起点；其他表述需要澄清",
        "automatic_action_permitted": False,
    }


@router.get("/extensions/sources")
def sources(request: Request, task_id: str):
    _require_role(request, "maker", "checker", "admin")
    authorize(request.app.state.settings.db_path, "task", task_id, "read")
    from marvis.repositories.datasets import DatasetRepository

    rows = DatasetRepository(request.app.state.settings.db_path).list_datasets(task_id)
    return {
        "task_id": task_id,
        "datasets": [
            {
                "id": row.id,
                "content_hash": row.content_hash,
                "row_count": row.row_count,
                "columns": list(row.columns),
            }
            for row in rows
        ],
        "automatic_action_permitted": False,
    }


@router.post("/proposals", status_code=201)
def proposal(payload: ProposalRequest, request: Request):
    principal = _human(request)
    try:
        return propose_schedule(
            _runtime(request),
            _schedule_contract(payload.schedule),
            expected_revision=payload.expected_revision,
            instruction=payload.instruction,
            actor_id=str(principal["id"]),
        )
    except (ValueError, KeyError, DataLayerError, ScheduleRevisionConflict) as exc:
        _reject(exc)


@router.get("/proposals/{proposal_id}")
def read_proposal(proposal_id: str, request: Request):
    principal = _require_role(request, "maker", "checker", "admin")
    import json

    with connect(request.app.state.settings.db_path) as conn:
        row = conn.execute(
            "SELECT * FROM operations_schedule_proposals WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()
    if row is None or row["actor_id"] != principal["id"]:
        raise HTTPException(404, detail={"code": "operations_proposal_not_found"})
    authorize(request.app.state.settings.db_path, "task", row["task_id"], "read")
    return {
        "proposal_id": row["proposal_id"],
        "contract": json.loads(row["contract_json"]),
        "contract_hash": row["contract_hash"],
        "binding_hash": row["binding_hash"],
        "expected_revision": row["expected_revision"],
        "expires_at": row["expires_at"],
        "automatic_action_permitted": False,
    }


@router.post("/proposals/{proposal_id}/confirm")
def confirm(proposal_id: str, payload: ProposalConfirmation, request: Request):
    principal = _human(request)
    try:
        return confirm_schedule(
            _runtime(request),
            proposal_id,
            expected_contract_hash=payload.expected_contract_hash,
            actor_id=principal["id"],
        )
    except (ValueError, KeyError, DataLayerError, ScheduleRevisionConflict) as exc:
        _reject(exc)


@router.post("/schedules/{schedule_id}/dispositions")
def disposition(schedule_id: str, payload: DispositionRequest, request: Request):
    _human(request)
    principal = request.state.local_principal
    from marvis.operations.router import _period_access

    _period_access(request, schedule_id, payload.period_key, "approve")
    try:
        return apply_disposition(
            _runtime(request),
            schedule_id,
            payload,
            principal=principal,
            governance_service=request.app.state.governance_service,
            plan_repo=request.app.state.plan_repo,
            plan_executor=request.app.state.plan_executor,
        )
    except (ValueError, KeyError, DispositionConsumed) as exc:
        _reject(exc)
    except Exception as exc:
        # A durable consumed claim is retained even after a native process or
        # post-commit response failure. The next request is read-side recovery.
        raise HTTPException(
            409,
            detail={
                "code": "operations_disposition_consumed",
                "next_action": "核对原生处置证据；此请求不会自动重执行",
            },
        ) from exc


@router.post("/dispositions/{key}/reconcile")
def reconcile(key: str, request: Request):
    principal = _human(request)
    try:
        return reconcile_disposition(_runtime(request), key, actor_id=principal["id"])
    except (ValueError, KeyError, DispositionConsumed) as exc:
        _reject(exc)


@router.post("/decision-feedback", status_code=201)
def feedback(payload: FeedbackImport, request: Request):
    principal = _human(request)
    try:
        return import_feedback(
            request.app.state.settings, payload, actor_id=principal["id"]
        )
    except (ValueError, KeyError, DataLayerError) as exc:
        _reject(exc)


@router.get("/decision-feedback")
def feedback_summary(task_id: str, request: Request):
    _require_role(request, "maker", "checker", "admin")
    authorize(request.app.state.settings.db_path, "task", task_id, "read")
    with connect(request.app.state.settings.db_path) as conn:
        rows = conn.execute(
            "SELECT package_hash,count(*) AS recorded_count,max(available_at) AS last_declared_available_at FROM operations_decision_feedback WHERE task_id=? GROUP BY package_hash",
            (task_id,),
        ).fetchall()
    return {
        "task_id": task_id,
        "packages": [dict(row) for row in rows],
        "outcome_assurance": "publisher_declared",
        "automatic_action_permitted": False,
    }
