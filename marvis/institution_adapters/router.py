"""Task-bound fixed protocol actions; endpoints and credentials never enter HTTP DTOs."""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import TypeAdapter

from marvis.routers.data import _require_task
from marvis.production_governance.router import _current_principal
from marvis.risk_context.source_contracts import SourceGrant, SourceError
from marvis.production_governance.errors import GovernanceConflict, GovernanceForbidden, GovernanceNotFound

from .contracts import AdapterError, DeploymentActivate, DeploymentInstall, DeploymentRollback, EndpointConfig, RegistryTransition, RegistryUpload, SourceGrantRevocation, SourceRequest
from .service import InstitutionAdapters

router = APIRouter(prefix="/api/tasks/{task_id}/institution-adapters", tags=["institution-adapters"])


def configure_institution_adapters(app, endpoints=None):
    if endpoints is None:
        path = os.environ.get("MARVIS_INSTITUTION_ADAPTERS_CONFIG")
        try:
            endpoints = TypeAdapter(list[EndpointConfig]).validate_json(Path(path).read_bytes()) if path else []
        except Exception as exc:
            raise AdapterError("institution_local_configuration_invalid", 503) from exc
    app.state.institution_adapters = InstitutionAdapters(app.state.settings, endpoints)
    return app.state.institution_adapters


def runtime(request, task_id):
    _require_task(request, task_id)
    actor = _current_principal(request)
    value = getattr(request.app.state, "institution_adapters", None)
    if value is None:
        raise HTTPException(503, detail={"code": "institution_adapter_unconfigured", "next_action": "请在本地配置受信机构协议"})
    return value, actor["id"]


def call(request, task_id, method, *args):
    try:
        value, actor = runtime(request, task_id)
        return getattr(value, method)(args[0], task_id, actor, *args[1:])
    except AdapterError as exc:
        raise HTTPException(exc.status, detail={"code": exc.code, "next_action": exc.next_action}) from None
    except (SourceError, GovernanceForbidden) as exc:
        raise HTTPException(getattr(exc, "status", 403), detail={"code": "institution_role_or_source_scope_forbidden"}) from None
    except GovernanceNotFound:
        raise HTTPException(404, detail={"code": "institution_native_binding_not_found"}) from None
    except GovernanceConflict:
        raise HTTPException(409, detail={"code": "institution_native_binding_conflict"}) from None
    except (ValueError, KeyError, OSError):
        raise HTTPException(422, detail={"code": "institution_protocol_evidence_unavailable", "next_action": "检查原登记证据、字段合同和远端状态"}) from None


@router.get("/capabilities")
def capabilities(task_id: str, request: Request):
    value, _ = runtime(request, task_id)
    return {**value.capabilities(), "schemas": {
        "registry_upload": RegistryUpload.model_json_schema(),
        "registry_transition": RegistryTransition.model_json_schema(),
        "deployment_install": DeploymentInstall.model_json_schema(),
        "deployment_activate": DeploymentActivate.model_json_schema(),
        "deployment_rollback": DeploymentRollback.model_json_schema(),
        "source_grant": SourceGrant.model_json_schema(),
        "source_query": SourceRequest.model_json_schema(),
        "source_revoke": SourceGrantRevocation.model_json_schema(),
    }}


@router.post("/{alias}/registry/uploads", status_code=201)
def upload(task_id: str, alias: str, payload: RegistryUpload, request: Request):
    return call(request, task_id, "upload", alias, payload)


@router.post("/{alias}/registry/promotions")
def promote(task_id: str, alias: str, payload: RegistryTransition, request: Request):
    return call(request, task_id, "registry_transition", alias, payload, "registry_promote")


@router.post("/{alias}/registry/revocations")
def revoke(task_id: str, alias: str, payload: RegistryTransition, request: Request):
    return call(request, task_id, "registry_transition", alias, payload, "registry_revoke")


@router.get("/{alias}/registry/uploads/{operation_id}/readback")
def registry_readback(task_id: str, alias: str, operation_id: str, request: Request):
    return call(request, task_id, "registry_readback", alias, operation_id)


@router.post("/{alias}/deployments/install", status_code=201)
def install(task_id: str, alias: str, payload: DeploymentInstall, request: Request):
    return call(request, task_id, "install", alias, payload)


@router.get("/{alias}/deployments/installations/{operation_id}/readback")
def deployment_readback(task_id: str, alias: str, operation_id: str, request: Request):
    return call(request, task_id, "deployment_readback", alias, operation_id)


@router.post("/{alias}/deployments/activate")
def activate(task_id: str, alias: str, payload: DeploymentActivate, request: Request):
    return call(request, task_id, "activate", alias, payload)


@router.post("/{alias}/deployments/rollback")
def rollback(task_id: str, alias: str, payload: DeploymentRollback, request: Request):
    return call(request, task_id, "rollback", alias, payload)


@router.get("/{alias}/operations/{operation_id}")
def operation(task_id: str, alias: str, operation_id: str, request: Request):
    return call(request, task_id, "read", alias, operation_id)


@router.post("/{alias}/operations/{operation_id}/recover")
def recover(task_id: str, alias: str, operation_id: str, request: Request):
    return call(request, task_id, "recover", alias, operation_id)


@router.post("/{alias}/sources/grants", status_code=201)
def grant(task_id: str, alias: str, payload: SourceGrant, request: Request):
    return call(request, task_id, "grant_source", alias, payload)


@router.post("/{alias}/sources/grants/{grant_id}/revoke")
def revoke_grant(task_id: str, alias: str, grant_id: str, request: Request):
    value, actor = runtime(request, task_id)
    try:
        value.role(task_id, actor, {"admin"})
        body = value.journal.read_grant(alias, grant_id, allow_revoked=True)
        if body["grant"]["task_id"] != task_id:
            raise AdapterError("source_grant_task_mismatch", 403)
        value.journal.revoke_grant(alias, grant_id)
        return {"grant_id": grant_id, "status": "revoked"}
    except AdapterError as exc:
        raise HTTPException(exc.status, detail={"code": exc.code}) from None
    except SourceError:
        raise HTTPException(403, detail={"code": "institution_role_or_source_scope_forbidden"}) from None


@router.post("/{alias}/sources/requests", status_code=201)
def query(task_id: str, alias: str, payload: SourceRequest, request: Request):
    return call(request, task_id, "query_source", alias, payload)
