from fastapi import APIRouter, HTTPException, Request

from marvis.db_schema import connect
from marvis.plugins.manifest import ToolRef
from marvis.production_governance.router import _local_principal_id
from marvis.risk_context.source_contracts import (
    AuthorizationBasis,
    RegisterProfile,
    SourceError,
    SourceGrant,
    SourceQuery,
)
from marvis.risk_context.source_repository import require_actor

router = APIRouter(prefix="/api", tags=["risk-sources"])


def service(request):
    return request.app.state.risk_sources


def actor(request, roles=None):
    principal_id = _local_principal_id(request)
    if roles:
        with connect(service(request).repo.db_path) as conn:
            try:
                require_actor(conn, principal_id, roles)
            except SourceError as exc:
                error(exc)
    return principal_id


def error(exc):
    raise HTTPException(
        exc.status if isinstance(exc, SourceError) else 409,
        {
            "code": exc.code
            if isinstance(exc, SourceError)
            else "source_material_invalid"
        },
    ) from exc


async def body(request, contract):
    data = await request.body()
    if len(data) > 32_000:
        raise HTTPException(413, {"code": "source_request_too_large"})
    try:
        return contract.model_validate_json(data)
    except ValueError as exc:
        # Pydantic's ordinary HTTP diagnostics echo input values, including secrets.
        raise HTTPException(422, {"code": "invalid_source_contract"}) from exc


@router.get("/risk-sources/capabilities")
def capabilities(request: Request):
    actor(request, {"maker", "admin", "checker"})
    return {
        "scope": "local_reference_and_deidentified_history",
        "external_authenticity": "not_independently_verified",
        "profile_schema": RegisterProfile.model_json_schema(),
        "grant_schema": SourceGrant.model_json_schema(),
        "query_schema": SourceQuery.model_json_schema(),
        "workflow_id": "risk_source_query",
        "rate_limit_scope": "all_http_attempts_including_readback_per_profile",
    }


@router.post("/risk-sources/profiles", status_code=201)
async def profiles_create(request: Request):
    principal = actor(request, {"admin"})
    payload = await body(request, RegisterProfile)
    try:
        return service(request).repo.register_profile(payload, principal)
    except (SourceError, ValueError, OSError) as exc:
        error(exc)


@router.get("/risk-sources/profiles")
def profiles_list(request: Request):
    try:
        return {"profiles": service(request).repo.list_profiles(actor(request))}
    except SourceError as exc:
        error(exc)


@router.post("/tasks/{task_id}/risk-sources/authorization-bases", status_code=201)
async def basis_create(task_id: str, request: Request):
    principal = actor(request, {"admin"})
    payload = await body(request, AuthorizationBasis)
    try:
        return service(request).authorization_basis(task_id, payload, principal)
    except (SourceError, ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/risk-sources/grants", status_code=201)
async def grant_create(request: Request):
    principal = actor(request, {"admin"})
    payload = await body(request, SourceGrant)
    try:
        return service(request).repo.create_grant(payload, principal)
    except (SourceError, ValueError, OSError) as exc:
        error(exc)


@router.post("/risk-sources/grants/{grant_id}/revoke")
def grant_revoke(grant_id: str, request: Request):
    try:
        return service(request).repo.revoke_grant(grant_id, actor(request))
    except SourceError as exc:
        error(exc)


@router.get("/tasks/{task_id}/risk-sources/grants")
def grants_list(task_id: str, request: Request):
    try:
        return {"grants": service(request).repo.list_grants(task_id, actor(request))}
    except SourceError as exc:
        error(exc)


@router.post("/tasks/{task_id}/risk-sources/requests", status_code=201)
async def prepare(task_id: str, request: Request):
    principal = actor(request, {"maker"})
    payload = await body(request, SourceQuery)
    try:
        return service(request).prepare(task_id, payload, principal)
    except (SourceError, ValueError, OSError) as exc:
        error(exc)


@router.post("/tasks/{task_id}/risk-sources/requests/{request_id}/execute")
def execute(task_id: str, request_id: str, request: Request):
    try:
        principal = actor(request, {"maker"})
        summary = service(request).read(task_id, request_id, principal)
        with connect(service(request).repo.db_path) as conn:
            service(request).repo.grant_tx(conn, task_id, summary["grant_id"], principal)
        result = request.app.state.tool_runner.invoke(
            ToolRef("risk_context", "query_source"),
            {"request_id": request_id},
            task_id=task_id,
        )
        if not result.ok:
            raise SourceError("source_tool_execution_failed")
        return result.output
    except (SourceError, ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/tasks/{task_id}/risk-sources/requests")
def requests_list(task_id: str, request: Request):
    try:
        return {"requests": service(request).list_requests(task_id, actor(request))}
    except (SourceError, ValueError, KeyError, OSError) as exc:
        error(exc)


@router.get("/tasks/{task_id}/risk-sources/requests/{request_id}")
def read(task_id: str, request_id: str, request: Request):
    try:
        return service(request).read(task_id, request_id, actor(request))
    except (SourceError, ValueError, KeyError, OSError) as exc:
        error(exc)


@router.get("/tasks/{task_id}/risk-sources/requests/{request_id}/evidence")
def evidence(task_id: str, request_id: str, request: Request):
    try:
        return service(request).evidence(task_id, request_id, actor(request))
    except (SourceError, ValueError, KeyError, OSError) as exc:
        error(exc)
