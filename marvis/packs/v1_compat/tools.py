from __future__ import annotations

from marvis.domain import TaskStatus
from marvis.packs.v1_compat.adapters import (
    artifact_ref,
    load_v1_task_context,
    material_checks,
    material_payloads,
    notebook_cell_count,
    report_artifacts,
    scan_materials,
    update_scan_status,
    validation_metric_summary,
)
from marvis.pipeline import run_metrics_stage, run_notebook_stage, run_report_stage


def tool_scan_materials(inputs: dict, ctx) -> dict:
    task_id = str(inputs["task_id"])
    context = load_v1_task_context(ctx, task_id)
    artifacts = scan_materials(context)
    update_scan_status(context)
    checks = material_checks(artifacts)
    return {
        "task_id": task_id,
        "status": "failed" if any(check["status"] != "ok" for check in checks) else "scanned",
        "materials": material_payloads(context, artifacts),
        "checks": checks,
    }


def tool_run_notebook(inputs: dict, ctx) -> dict:
    task_id = str(inputs["task_id"])
    context = load_v1_task_context(ctx, task_id)
    run_notebook_stage(task_id=task_id, settings=context.pipeline_settings)
    task = context.repo.get_task(task_id)
    return {
        "task_id": task_id,
        "status": "executed" if task.status is TaskStatus.EXECUTED else "failed",
        "notebook_cells": notebook_cell_count(context),
        "sample_ref": artifact_ref(context.execution_dir / "code_model_scores.csv"),
        "runtime_model_ref": artifact_ref(context.execution_dir / "runtime_contract.json"),
        "evidence_ref": artifact_ref(context.outputs_dir / "reproducibility_result.json"),
    }


def tool_compute_validation_metrics(inputs: dict, ctx) -> dict:
    task_id = str(inputs["task_id"])
    context = load_v1_task_context(ctx, task_id)
    run_metrics_stage(task_id=task_id, settings=context.pipeline_settings)
    return validation_metric_summary(context)


def tool_render_reports(inputs: dict, ctx) -> dict:
    task_id = str(inputs["task_id"])
    context = load_v1_task_context(ctx, task_id)
    run_report_stage(task_id=task_id, settings=context.pipeline_settings)
    status = context.repo.get_task(task_id).status
    return {
        "task_id": task_id,
        "status": _report_status(status),
        "artifacts": report_artifacts(context),
    }


def _report_status(status: TaskStatus) -> str:
    if status is TaskStatus.SUCCEEDED:
        return "succeeded"
    if status is TaskStatus.REVIEW_REQUIRED:
        return "review_required"
    return "failed"
