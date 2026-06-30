from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request

from marvis.api_report_field_helpers import build_report_field_payload
from marvis.api_schemas import ReportFieldsUpdateRequest
from marvis.api_task_helpers import get_task_or_404
from marvis.db import TaskRepository
from marvis.state_machine import ConflictError


router = APIRouter(prefix="/api", tags=["report-fields"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.get("/tasks/{task_id}/report-fields")
def get_report_fields(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    values, revision = repo.get_report_values(task_id)
    return build_report_field_payload(
        request,
        task,
        values,
        revision,
        include_metric_table_sections=True,
    )


@router.put("/tasks/{task_id}/report-fields")
def update_report_fields(
    task_id: str,
    payload: ReportFieldsUpdateRequest,
    request: Request,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    if if_match is None:
        raise HTTPException(status_code=428, detail="If-Match header is required")
    try:
        expected_revision = int(if_match)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="If-Match must be an integer",
        ) from exc
    try:
        update_values = getattr(repo, "update_report_values_with_audit", None)
        if callable(update_values):
            revision = update_values(
                task_id,
                payload.text_values,
                expected_revision=expected_revision,
                audit={
                    "kind": "report.values.update",
                    "target_ref": task_id,
                    "outcome": "succeeded",
                    "detail": {
                        "keys": sorted(payload.text_values),
                        "expected_revision": expected_revision,
                    },
                },
            )
        else:
            revision = repo.update_report_values(
                task_id,
                payload.text_values,
                expected_revision=expected_revision,
            )
        values, _ = repo.get_report_values(task_id)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return build_report_field_payload(request, task, values, revision)
