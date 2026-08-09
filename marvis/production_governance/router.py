from __future__ import annotations

import hmac
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from marvis.production_governance.errors import (
    GovernanceConflict,
    GovernanceForbidden,
    GovernanceNotFound,
)
from marvis.production_governance.evidence import (
    ActivationVerificationContext,
    resolve_activation_evidence,
)
from marvis.production_governance.repository import ProductionGovernanceRepository


router = APIRouter(
    prefix="/api/production-governance",
    tags=["production-governance"],
)
GOVERNANCE_ADMIN_HEADER = "x-marvis-governance-admin"


class ClaimPrincipalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    display_name: str = Field(min_length=1, max_length=120)
    role: Literal["maker", "checker", "admin"]


class CreatePromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    environment: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    deployment_slot: Literal["shadow", "production"]
    strategy_id: str = Field(min_length=1, max_length=160)
    strategy_version: StrictInt = Field(ge=1)
    reason: str = Field(min_length=1, max_length=4000)
    expires_in_seconds: StrictInt = Field(default=900, ge=1, le=86_400)


class PromotionApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision: Literal["approve"]
    reason: str = Field(min_length=1, max_length=4000)


class ActivatePromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    verifier_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$",
    )
    receipt_id: str = Field(min_length=1, max_length=240)
    reason: str = Field(min_length=1, max_length=4000)


class RollbackDeploymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reason: str = Field(min_length=1, max_length=4000)
    expected_active_deployment_id: str = Field(min_length=1, max_length=160)


def _repo(request: Request) -> ProductionGovernanceRepository:
    return ProductionGovernanceRepository(request.app.state.settings.db_path)


def _local_principal_id(request: Request) -> str:
    principal = getattr(request.state, "local_principal", None)
    principal_id = str(getattr(principal, "id", "")).strip()
    if not principal_id:
        raise HTTPException(status_code=401, detail="server session principal required")
    return principal_id


def _require_workspace_admin(request: Request) -> None:
    expected = str(getattr(request.app.state, "plugin_admin_token", "") or "")
    presented = request.headers.get(GOVERNANCE_ADMIN_HEADER, "")
    if not expected or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=403, detail="workspace governance admin required")


@router.post("/principals/claim", status_code=201)
def claim_principal(payload: ClaimPrincipalRequest, request: Request) -> dict:
    _require_workspace_admin(request)
    try:
        return _repo(request).claim_principal(
            local_principal_id=_local_principal_id(request),
            display_name=payload.display_name,
            role=payload.role,
        )
    except GovernanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GovernanceNotFound as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/me")
def get_current_principal(request: Request) -> dict:
    try:
        return _repo(request).get_principal(_local_principal_id(request))
    except GovernanceNotFound as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _current_principal(request: Request) -> dict:
    try:
        return _repo(request).get_principal(_local_principal_id(request))
    except GovernanceNotFound as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _raise_public_error(exc: Exception) -> None:
    if isinstance(exc, GovernanceForbidden):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, GovernanceNotFound):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, GovernanceConflict):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise exc


@router.post("/promotion-requests", status_code=201)
def create_promotion_request(payload: CreatePromotionRequest, request: Request) -> dict:
    actor = _current_principal(request)
    try:
        return _repo(request).create_promotion_request(
            actor_principal_id=actor["id"],
            environment=payload.environment,
            deployment_slot=payload.deployment_slot,
            strategy_id=payload.strategy_id,
            strategy_version=payload.strategy_version,
            reason=payload.reason,
            expires_in_seconds=payload.expires_in_seconds,
        )
    except (GovernanceForbidden, GovernanceNotFound, GovernanceConflict) as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable")


@router.post("/promotion-requests/{request_id}/approvals")
def approve_promotion_request(
    request_id: str,
    payload: PromotionApprovalRequest,
    request: Request,
) -> dict:
    actor = _current_principal(request)
    try:
        return _repo(request).approve_promotion_request(
            request_id=request_id,
            actor_principal_id=actor["id"],
            reason=payload.reason,
        )
    except (GovernanceForbidden, GovernanceNotFound, GovernanceConflict) as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable")


@router.get("/promotion-requests/{request_id}")
def get_promotion_request(request_id: str, request: Request) -> dict:
    _current_principal(request)
    try:
        return _repo(request).get_promotion_request(request_id)
    except GovernanceNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/environments/{environment}/promotion-requests/{request_id}/activate",
    status_code=201,
)
def activate_promotion_request(
    environment: str,
    request_id: str,
    payload: ActivatePromotionRequest,
    request: Request,
) -> dict:
    actor = _current_principal(request)
    if actor["role"] != "admin":
        raise HTTPException(status_code=403, detail="only an admin can activate a deployment")
    try:
        promotion = _repo(request).get_promotion_request(request_id)
        context = ActivationVerificationContext(
            promotion_request_id=request_id,
            environment=environment,
            deployment_slot=promotion["deployment_slot"],
            strategy_id=promotion["strategy_id"],
            strategy_version=promotion["strategy_version"],
            strategy_content_hash=promotion["strategy_content_hash"],
            manifest_hash=promotion["manifest_hash"],
        )
        evidence = resolve_activation_evidence(
            verifiers=getattr(
                request.app.state,
                "production_activation_verifiers",
                {},
            ),
            verifier_id=payload.verifier_id,
            receipt_id=payload.receipt_id,
            context=context,
        )
        return _repo(request).activate_promotion_request(
            request_id=request_id,
            environment=environment,
            actor_principal_id=actor["id"],
            evidence=evidence,
            reason=payload.reason,
        )
    except (GovernanceForbidden, GovernanceNotFound, GovernanceConflict) as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable")


@router.get("/environments/{environment}")
def get_environment(environment: str, request: Request) -> dict:
    _current_principal(request)
    try:
        return _repo(request).get_environment(environment)
    except GovernanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/strategies/{strategy_id}/deployments")
def get_strategy_deployments(strategy_id: str, request: Request) -> dict:
    _current_principal(request)
    try:
        return _repo(request).get_strategy_deployments(strategy_id)
    except GovernanceNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/environments/{environment}/deployments/{deployment_id}/rollback")
def rollback_deployment(
    environment: str,
    deployment_id: str,
    payload: RollbackDeploymentRequest,
    request: Request,
) -> dict:
    actor = _current_principal(request)
    try:
        return _repo(request).rollback_deployment(
            environment=environment,
            deployment_id=deployment_id,
            expected_active_deployment_id=payload.expected_active_deployment_id,
            actor_principal_id=actor["id"],
            reason=payload.reason,
        )
    except (GovernanceForbidden, GovernanceNotFound, GovernanceConflict) as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable")


@router.get("/audit/export")
def export_audit(request: Request) -> JSONResponse:
    actor = _current_principal(request)
    if actor["role"] != "admin":
        raise HTTPException(status_code=403, detail="only an admin can export audit")
    try:
        payload = _repo(request).export_audit()
    except GovernanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": (
                'attachment; filename="production_governance_audit.json"'
            )
        },
    )


__all__ = ["GOVERNANCE_ADMIN_HEADER", "router"]
