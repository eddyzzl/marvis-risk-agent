"""Typed local collection APIs; all effects execute through existing Plans."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from marvis.collection.batches import CollectionBatchRequest
from marvis.collection.contracts import (
    CollectionCase,
    InstallmentSchedule,
    ReconciliationRequest,
)
from marvis.collection.ledger import CollectionEvidenceError
from marvis.collection.service import CashflowImport, CollectionMaterial
from marvis.decision_twin._canonical import canonical_json
from marvis.production_governance.router import _local_principal_id
from marvis.risk_context.event_repository import _decode
from marvis.risk_context.source_contracts import SourceError

router = APIRouter(prefix="/api/tasks/{task_id}/collection", tags=["collection"])


def service(request):
    return request.app.state.collection


def error(exc):
    if isinstance(exc, SourceError):
        raise HTTPException(
            exc.status, {"code": exc.code, "next_action": "required_review"}
        ) from exc
    code = (
        str(exc)
        if isinstance(exc, CollectionEvidenceError)
        else "collection_material_invalid"
    )
    raise HTTPException(409, {"code": code, "next_action": "required_review"}) from exc


async def body(request, model):
    raw = await request.body()
    if len(raw) > 16_000_000:
        raise HTTPException(413, {"code": "collection_request_too_large"})
    try:
        return model.model_validate(_decode(raw))
    except ValueError:
        raise HTTPException(422, {"code": "collection_contract_invalid"}) from None


@router.get("/capabilities")
def capabilities(task_id: str, request: Request):
    _local_principal_id(request)
    return {
        "material_schema": CollectionMaterial.model_json_schema(),
        "case_schema": CollectionCase.model_json_schema(),
        "schedule_schema": InstallmentSchedule.model_json_schema(),
        "cashflow_schema": CashflowImport.model_json_schema(),
        "reconciliation_schema": ReconciliationRequest.model_json_schema(),
        "batch_schema": CollectionBatchRequest.model_json_schema(),
        "workflows": [
            "collection_queue_batch",
            "collection_execute_reference",
            "collection_cancel_batch",
        ],
        "execution_mode": "local_reference",
        "customer_contacted": False,
        "policy_assurance": "business_declared_not_legal_or_contact_authority",
    }


@router.post("/materials", status_code=201)
async def material(task_id: str, request: Request):
    payload = await body(request, CollectionMaterial)
    try:
        return service(request).material(task_id, payload, _local_principal_id(request))
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/cases", status_code=201)
async def create_case(task_id: str, request: Request):
    payload = await body(request, CollectionCase)
    try:
        return service(request).create_case(
            task_id, payload, _local_principal_id(request)
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/cases/{case_id}")
def case(task_id: str, case_id: str, request: Request):
    try:
        return service(request).case(task_id, case_id, _local_principal_id(request))
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/schedules", status_code=201)
async def schedule(task_id: str, request: Request):
    payload = await body(request, InstallmentSchedule)
    try:
        return service(request).schedule(task_id, payload, _local_principal_id(request))
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/cashflows", status_code=201)
async def cashflows(task_id: str, request: Request):
    payload = await body(request, CashflowImport)
    try:
        return service(request).cashflows(
            task_id, payload, _local_principal_id(request)
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/reconciliations", status_code=201)
async def reconcile(task_id: str, request: Request):
    payload = await body(request, ReconciliationRequest)
    try:
        return service(request).reconcile(
            task_id, payload, _local_principal_id(request)
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/cases/{case_id}/reconciliations/{digest}")
def report(task_id: str, case_id: str, digest: str, request: Request):
    try:
        return service(request).report(
            task_id, case_id, digest, _local_principal_id(request)
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.post("/batches", status_code=201)
async def prepare(task_id: str, request: Request):
    payload = await body(request, CollectionBatchRequest)
    try:
        return service(request).prepare(task_id, payload, _local_principal_id(request))
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/batches/{batch_id}")
def read(task_id: str, batch_id: str, request: Request):
    try:
        return service(request).read(task_id, batch_id, _local_principal_id(request))
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/batches/{batch_id}/effects/{effect_id}")
def evidence(task_id: str, batch_id: str, effect_id: str, request: Request):
    try:
        return service(request).effect(
            task_id, batch_id, effect_id, _local_principal_id(request)
        )
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/batches/{batch_id}/effects/{effect_id}/export")
def export(task_id: str, batch_id: str, effect_id: str, request: Request):
    value = evidence(task_id, batch_id, effect_id, request)
    return Response(
        canonical_json(value["facts"]),
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="collection-reference-receipt.json"'
        },
    )


@router.get("/cases")
def list_cases(task_id: str, request: Request):
    try:
        return {
            "cases": service(request).list_cases(task_id, _local_principal_id(request))
        }
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)


@router.get("/batches")
def list_batches(task_id: str, request: Request):
    try:
        return {
            "batches": service(request).list_batches(
                task_id, _local_principal_id(request)
            )
        }
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        error(exc)
