from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from marvis.decision_twin.batch_contracts import (
    HistoricalReconciliationRequest,
    HistoricalReplayRequest,
)
from marvis.decision_twin.batch_material import (
    BatchMaterial,
    MAX_BATCH_ROWS,
    OUTCOME_KIND,
    REPLAY_KIND,
)
from marvis.output.decision_twin import render_historical_replay
from marvis.reference_decision.contracts import DecisionError


router = APIRouter(prefix="/api", tags=["decision-twin"])


def _material(request, task_id):
    try:
        return BatchMaterial(request.app.state.settings, task_id)
    except KeyError as exc:
        raise HTTPException(404, "task not found") from exc


def _load(material, artifact_id):
    for kind in (REPLAY_KIND, OUTCOME_KIND):
        try:
            return material.load(artifact_id, kind=kind)
        except DecisionError as exc:
            if exc.status != 404:
                raise
    raise HTTPException(404, "historical artifact not found")


@router.get("/decision-twin/capabilities")
def capabilities():
    return {
        "schema_version": "decision_twin.capabilities.v2",
        "replay_schema": HistoricalReplayRequest.model_json_schema(),
        "reconciliation_schema": HistoricalReconciliationRequest.model_json_schema(),
        "workflow_ids": [
            "historical_decision_replay",
            "historical_outcome_reconciliation",
        ],
        "maximum_batch_rows": MAX_BATCH_ROWS,
        "authority": "proposal_only",
        "external_execution_authentication": "not_available",
        "causal_identifiability": "unidentified",
    }


@router.post("/tasks/{task_id}/decision-twin/proposal")
def proposal(task_id: str, body: HistoricalReplayRequest, request: Request):
    try:
        return _material(request, task_id).prepare(body)[2]
    except (ValueError, RuntimeError, KeyError) as exc:
        _error(exc)


@router.post("/tasks/{task_id}/decision-twin/reconciliation-proposal")
def reconciliation_proposal(
    task_id: str, body: HistoricalReconciliationRequest, request: Request
):
    try:
        material = _material(request, task_id)
        material.load(body.replay_artifact_id)
        material.dataset(body.dataset_id, body.expected_content_hash)
        return {
            "contract": body.model_dump(),
            "proposal_hash": body.contract_hash,
            "origin": "external_history_import",
            "authority": "proposal_only",
        }
    except (ValueError, RuntimeError, KeyError) as exc:
        _error(exc)


@router.get("/tasks/{task_id}/decision-twin")
def list_receipts(task_id: str, request: Request):
    material = _material(request, task_id)
    receipts = []
    for record in material.artifacts.list_for_task(task_id):
        if record["kind"] in (REPLAY_KIND, OUTCOME_KIND):
            # Listing verifies the same signatures as detail/export; altered receipts
            # remain visible as failures instead of silently disappearing.
            try:
                loaded = material.load(record["content_hash"], kind=record["kind"])
                receipts.append(
                    {
                        "artifact_id": loaded["artifact_id"],
                        "kind": loaded["kind"],
                        "status": "available",
                        "contract_hash": loaded["payload"]["contract_hash"],
                        "created_at": record["created_at"],
                    }
                )
            except (ValueError, RuntimeError, KeyError):
                receipts.append(
                    {
                        "artifact_id": record["content_hash"],
                        "kind": record["kind"],
                        "status": "integrity_failed",
                    }
                )
    return {"task_id": task_id, "artifacts": receipts}


@router.get("/tasks/{task_id}/decision-twin/{artifact_id}")
def get_receipt(task_id: str, artifact_id: str, request: Request):
    try:
        return _load(_material(request, task_id), artifact_id)
    except (ValueError, RuntimeError, KeyError) as exc:
        _error(exc)


@router.get("/tasks/{task_id}/decision-twin/{artifact_id}/export/{format}")
def export_receipt(
    task_id: str,
    artifact_id: str,
    format: Literal["json", "xlsx", "docx"],
    request: Request,
):
    from marvis.decision_twin._canonical import canonical_json

    try:
        receipt = _load(_material(request, task_id), artifact_id)
        data = (
            canonical_json(receipt).encode()
            if format == "json"
            else render_historical_replay(receipt, format)
        )
        media = {
            "json": "application/json",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }[format]
        return Response(
            data,
            media_type=media,
            headers={
                "Content-Disposition": f'attachment; filename="historical-replay-{artifact_id[:12]}.{format}"'
            },
        )
    except (ValueError, RuntimeError, KeyError) as exc:
        _error(exc)


def _error(exc):
    code = exc.code if isinstance(exc, DecisionError) else "historical_material_invalid"
    raise HTTPException(
        exc.status if isinstance(exc, DecisionError) else 409,
        detail={
            "code": code,
            "next_action": "核对同任务数据版本、时点口径及冻结包后重新生成提案",
        },
    ) from exc
