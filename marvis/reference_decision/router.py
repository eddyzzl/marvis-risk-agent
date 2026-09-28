from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from marvis.production_governance.errors import GovernanceConflict
from marvis.production_governance.router import _current_principal
from marvis.reference_decision.contracts import (
    DecisionError,
    DecisionRequest,
    PackageBuildRequest,
)
from marvis.reference_decision.readiness import package_readiness, rule_package_readiness
from marvis.reference_decision.event_contracts import EventEvidenceReference


router = APIRouter(prefix="/api/reference-decision", tags=["reference-decision"])


class InstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    promotion_id: str = Field(min_length=1, max_length=160)
    probe_features: dict = Field(max_length=500)
    event_evidence: EventEvidenceReference | None = None


@router.post("/installations", status_code=201)
def install(payload: InstallRequest, request: Request):
    actor = _current_principal(request)
    if actor["role"] != "admin":
        raise HTTPException(403, "only an admin can install an approved package")
    try:
        return request.app.state.reference_deployment_adapter.install(payload.promotion_id, payload.probe_features,
            event_evidence=payload.event_evidence, actor_id=actor["id"])
    except (DecisionError, GovernanceConflict) as exc:
        public_error(exc)


@router.get("/installations")
def list_installations(request: Request, limit: int = Query(default=100, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    _current_principal(request)
    return request.app.state.reference_deployment_adapter.list(limit=limit, offset=offset)


@router.get("/installations/{promotion_id}")
def read_installation(promotion_id: str, request: Request):
    _current_principal(request)
    try:
        return request.app.state.reference_deployment_adapter.readback(promotion_id)
    except (DecisionError, GovernanceConflict) as exc:
        public_error(exc)


def service(request):
    return request.app.state.reference_decision


def public_error(exc):
    if isinstance(exc, DecisionError):
        raise HTTPException(
            exc.status,
            detail={
                "code": exc.code,
                "next_action": "检查冻结包、字段合同或部署状态后重试",
            },
        ) from exc
    raise HTTPException(409, detail={"code": "governance_binding_invalid"}) from exc


@router.get("/capabilities")
def capabilities(request: Request):
    _current_principal(request)
    return {
        "environment": "local-reference",
        "deployment_scope": "local_reference_only",
        "request_schema": DecisionRequest.model_json_schema(),
        "package_schema": TypeAdapter(PackageBuildRequest).json_schema(),
        "admission": {
            "concurrent_per_slot": 4,
            "new_requests_per_minute_per_slot": 120,
        },
        "business_slo": "not_declared",
        "institution_adapter_ids": [],
    }


@router.post("/packages", status_code=201)
def build_package(payload: PackageBuildRequest, request: Request):
    actor = _current_principal(request)
    if actor["role"] != "maker":
        raise HTTPException(403, "only a maker can build a package")
    try:
        package_hash, manifest = service(request).packages.build(
            payload, actor_id=actor["id"]
        )
        return {
            "package_hash": package_hash,
            "manifest": manifest,
            "state": "built_not_installed",
        }
    except (DecisionError, GovernanceConflict) as exc:
        public_error(exc)


@router.get("/readiness")
def readiness(request: Request, model_artifact_id: str | None = Query(default=None, min_length=1, max_length=160),
              package_kind: Literal["model", "rule_only"] = "model",
              strategy_id: str | None = Query(default=None, min_length=1, max_length=160),
              strategy_version: int | None = Query(default=None, ge=1)):
    _current_principal(request)
    if package_kind == "rule_only":
        if model_artifact_id is not None or strategy_id is None or strategy_version is None:
            raise HTTPException(422, detail={"code": "rule_readiness_requires_strategy_only"})
        return rule_package_readiness(service(request).packages, strategy_id=strategy_id,
                                      strategy_version=strategy_version)
    if model_artifact_id is None:
        raise HTTPException(422, detail={"code": "model_artifact_id_required"})
    return package_readiness(service(request).packages, model_artifact_id,
                             strategy_id=strategy_id, strategy_version=strategy_version)


@router.get("/packages")
def list_packages(request: Request, task_id: str | None = None,
                  limit: int = Query(default=100, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    _current_principal(request)
    try:
        return service(request).packages.list(task_id=task_id, limit=limit, offset=offset)
    except DecisionError as exc:
        public_error(exc)


@router.get("/packages/{package_hash}")
def get_package(package_hash: str, request: Request):
    _current_principal(request)
    try:
        return {
            "package_hash": package_hash,
            "manifest": service(request).packages.get(package_hash),
        }
    except DecisionError as exc:
        public_error(exc)


@router.get("/status")
def status(request: Request, slot: Literal["production", "shadow"] = "production"):
    _current_principal(request)
    try:
        return service(request).head(slot)
    except DecisionError as exc:
        return {
            "state": "unavailable",
            "error_code": exc.code,
            "next_action": "构建发布包并完成独立审批、安装和激活",
        }


@router.post("/decisions")
def decide(
    payload: DecisionRequest,
    request: Request,
    slot: Literal["production", "shadow"] = "production",
):
    actor = _current_principal(request)
    try:
        return service(request).decide(payload, slot=slot, actor_id=actor["id"])
    except DecisionError as exc:
        public_error(exc)
