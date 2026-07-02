from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from marvis.api_task_helpers import get_task_or_404
from marvis.db import TaskRepository


router = APIRouter(prefix="/api", tags=["audit"])

_EXPORT_COLUMNS = (
    "id",
    "kind",
    "actor",
    "target_ref",
    "inputs_hash",
    "outcome",
    "detail",
    "at",
)


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def _bounded_limit(limit: int | None, *, default: int, maximum: int) -> int:
    if limit is None:
        return default
    return max(1, min(int(limit), maximum))


def _bounded_offset(offset: int) -> int:
    return max(0, int(offset))


@router.get("/audit")
def list_audit(
    request: Request,
    kind: str | None = None,
    kind_prefix: str | None = None,
    target_ref: str | None = None,
    target_ref_prefix: str | None = None,
    task_id: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    repo = _repo(request)
    bounded_limit = _bounded_limit(limit, default=100, maximum=500)
    bounded_offset = _bounded_offset(offset)
    filters = dict(
        kind=kind,
        kind_prefix=kind_prefix,
        target_ref=target_ref,
        target_ref_prefix=target_ref_prefix,
        task_id=task_id,
        after=after,
        before=before,
    )
    rows = repo.list_audit(limit=bounded_limit, offset=bounded_offset, **filters)
    total = repo.count_audit(**filters)
    return {
        "items": rows,
        "total": total,
        "limit": bounded_limit,
        "offset": bounded_offset,
        "has_more": bounded_offset + len(rows) < total,
    }


@router.get("/audit/export")
def export_audit(
    request: Request,
    kind: str | None = None,
    kind_prefix: str | None = None,
    target_ref: str | None = None,
    target_ref_prefix: str | None = None,
    task_id: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> StreamingResponse:
    repo = _repo(request)
    filters = dict(
        kind=kind,
        kind_prefix=kind_prefix,
        target_ref=target_ref,
        target_ref_prefix=target_ref_prefix,
        task_id=task_id,
        after=after,
        before=before,
    )
    rows = repo.list_audit(limit=None, offset=0, **filters)
    filename = "audit_export.csv" if not task_id else f"audit_export_{task_id}.csv"
    return StreamingResponse(
        _csv_rows(rows),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/tasks/{task_id}/audit")
def get_task_audit(
    task_id: str,
    request: Request,
    kind: str | None = None,
    kind_prefix: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    repo = _repo(request)
    get_task_or_404(repo, task_id)
    bounded_limit = _bounded_limit(limit, default=200, maximum=1000)
    bounded_offset = _bounded_offset(offset)
    filters = dict(
        kind=kind,
        kind_prefix=kind_prefix,
        task_id=task_id,
        after=after,
        before=before,
    )
    rows = repo.list_audit(limit=bounded_limit, offset=bounded_offset, **filters)
    total = repo.count_audit(**filters)
    return {
        "task_id": task_id,
        "items": rows,
        "total": total,
        "limit": bounded_limit,
        "offset": bounded_offset,
        "has_more": bounded_offset + len(rows) < total,
    }


def _csv_rows(rows: list[dict]) -> Iterator[str]:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_EXPORT_COLUMNS)
    yield buffer.getvalue()
    for row in rows:
        buffer.seek(0)
        buffer.truncate(0)
        writer.writerow(_export_row_values(row))
        yield buffer.getvalue()


def _export_row_values(row: dict) -> list[str]:
    values = []
    for column in _EXPORT_COLUMNS:
        value = row.get(column)
        if column == "detail":
            values.append(json.dumps(value or {}, ensure_ascii=False, separators=(",", ":")))
        else:
            values.append("" if value is None else str(value))
    return values
