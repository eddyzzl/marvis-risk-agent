"""Event-specific authenticated APIs; execution is through the governed Workflow."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from marvis.db_schema import connect
from marvis.production_governance.router import _local_principal_id
from marvis.risk_context.event_contracts import (
    EventError,
    EventSourceContract,
    EventSourceGrant,
    canonical_json,
)
from marvis.risk_context.event_repository import _decode
from marvis.risk_context.event_service import EventDatasetImport, EventRequest
from marvis.risk_context.source_contracts import SourceError
from marvis.risk_context.source_repository import require_actor

router = APIRouter(prefix="/api", tags=["risk-events"])


def service(request):
    return request.app.state.risk_events


def actor(request, roles):
    principal = _local_principal_id(request)
    try:
        with connect(service(request).repo.db_path) as conn:
            require_actor(conn, principal, roles)
    except SourceError as exc:
        error(exc)
    return principal


def error(exc):
    known = isinstance(exc, (EventError, SourceError))
    raise HTTPException(
        exc.status if known else 409,
        {
            "code": exc.code if known else "event_material_invalid",
            "next_action": "required_review",
        },
    ) from exc


async def body(request, model):
    raw = await request.body()
    if len(raw) > 64_000:
        raise HTTPException(413, {"code": "event_request_too_large"})
    try:
        return model.model_validate(_decode(raw))
    except ValueError as exc:
        raise HTTPException(422, {"code": "event_contract_invalid"}) from exc


@router.get("/risk-events/capabilities")
def capabilities(request: Request):
    actor(request, {"maker", "checker", "admin"})
    return {
        "source_schema": EventSourceContract.model_json_schema(),
        "grant_schema": EventSourceGrant.model_json_schema(),
        "import_schema": EventDatasetImport.model_json_schema(),
        "request_schema": EventRequest.model_json_schema(),
        "workflow_id": "event_feature_replay",
        "source_assurance": "publisher_declared",
        "automated_clearance": False,
        "grant_basis": "existing_same_task_registered_artifact",
    }


@router.post("/risk-events/sources", status_code=201)
async def register_source(request: Request):
    principal = actor(request, {"admin"})
    payload = await body(request, EventSourceContract)
    try:
        return service(request).repo.register_source(payload, principal)
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/risk-events/grants", status_code=201)
async def grant(request: Request):
    principal = actor(request, {"admin"})
    payload = await body(request, EventSourceGrant)
    try:
        return service(request).repo.create_grant(payload, principal)
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/tasks/{task_id}/risk-events/grants/{grant_id}/revoke")
def revoke(task_id: str, grant_id: str, request: Request):
    try:
        service(request).repo.revoke_grant(task_id, grant_id, actor(request, {"admin"}))
        return {"grant_id": grant_id, "status": "revoked"}
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/tasks/{task_id}/risk-events/imports", status_code=201)
async def ingest(task_id: str, request: Request):
    principal = actor(request, {"maker", "admin"})
    payload = await body(request, EventDatasetImport)
    try:
        return service(request).repo.ingest_dataset(
            task_id,
            payload.dataset,
            grant_id=payload.grant_id,
            actor_id=principal,
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/tasks/{task_id}/risk-events/requests", status_code=201)
async def prepare(task_id: str, request: Request):
    principal = actor(request, {"maker", "admin"})
    payload = await body(request, EventRequest)
    try:
        return service(request).prepare(task_id, payload, principal)
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/tasks/{task_id}/risk-events/requests/{request_id}/proposal")
def proposal(task_id: str, request_id: str, grant_id: str, request: Request):
    try:
        return service(request).proposal(
            task_id, request_id, actor(request, {"maker", "checker", "admin"}), grant_id
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/tasks/{task_id}/risk-events/requests/{request_id}")
def read(task_id: str, request_id: str, grant_id: str, request: Request):
    try:
        return service(request).read(
            task_id, request_id, actor(request, {"maker", "checker", "admin"}), grant_id
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/tasks/{task_id}/risk-events/requests/{request_id}/evidence")
def evidence(task_id: str, request_id: str, grant_id: str, request: Request):
    try:
        return service(request).evidence(
            task_id, request_id, actor(request, {"maker", "checker", "admin"}), grant_id
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/tasks/{task_id}/risk-events/requests/{request_id}/export/json")
def export(task_id: str, request_id: str, grant_id: str, request: Request):
    result = evidence(task_id, request_id, grant_id, request)
    return Response(
        canonical_json(result),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="event-evidence.json"'},
    )
