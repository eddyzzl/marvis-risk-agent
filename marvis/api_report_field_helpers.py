from __future__ import annotations

import json

from fastapi import Request

from marvis.domain import TaskRecord, TaskStatus
from marvis.metric_tables import (
    metric_table_sections_from_payload as _metric_table_sections_from_payload,
)
from marvis.pipeline import REPORT_STAGE_FAILURE_PREFIX
from marvis.report_fields import report_field_payload
from marvis.report_texts import computed_report_text_values_from_payload


def build_report_field_payload(
    request: Request,
    task: TaskRecord,
    values: dict[str, str],
    revision: int,
    *,
    include_metric_table_sections: bool = False,
) -> dict:
    payload = validation_results_payload_for_task(request, task)
    return report_field_payload(
        task,
        values,
        revision,
        metric_values=metric_values_from_payload(payload),
        metric_table_sections=metric_table_sections_from_payload(payload)
        if include_metric_table_sections
        else None,
    )


def metric_values_from_payload(payload: dict | None) -> dict[str, str]:
    if payload is None:
        return {}
    return computed_report_text_values_from_payload(payload)


def metric_table_sections_from_payload(payload: dict | None) -> list[dict]:
    if payload is None:
        return []
    return _metric_table_sections_from_payload(payload)


def validation_results_payload_for_task(request: Request, task: TaskRecord) -> dict | None:
    if task.status not in {
        TaskStatus.WRITING_ARTIFACTS,
        TaskStatus.SUCCEEDED,
        TaskStatus.REVIEW_REQUIRED,
    } and not (
        task.status == TaskStatus.FAILED
        and task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
    ):
        return None
    result_path = (
        request.app.state.settings.tasks_dir
        / task.id
        / "outputs"
        / "validation_results.json"
    )
    if not result_path.exists():
        return None
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
