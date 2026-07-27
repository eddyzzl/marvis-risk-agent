"""Builtin tool entrypoint for deterministic risk/profitability Excel reports."""

from __future__ import annotations

from pathlib import Path

from marvis.artifacts import ArtifactUnitOfWork
from marvis.output.risk_analysis_report import (
    RiskAnalysisReportPayload,
    render_risk_analysis_report,
)
from marvis.packs.risk_analysis.calculations import (
    RiskAnalysisError,
    calculate_risk_analysis,
)
from marvis.plugins.sdk import PackRuntime


class _Runtime(PackRuntime):
    pass


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def tool_generate_risk_analysis_report(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    analysis_kind = str(inputs["analysis_kind"])
    dataset_id = str(inputs["dataset_id"])
    dataset = _owned_dataset(runtime, dataset_id, task_id=str(ctx.task_id))
    raw_column_map = inputs.get("column_map")
    if not isinstance(raw_column_map, dict):
        raise RiskAnalysisError(
            "column_map 必须是 canonical->source 的对象。",
            analysis_kind=analysis_kind,
        )
    requested_columns = list(
        dict.fromkeys(
            str(value) for value in raw_column_map.values() if str(value).strip()
        )
    )
    frame = runtime.backend.read_frame(
        runtime.registry.resolve_path(dataset.id),
        columns=requested_columns or None,
    )
    calculation = calculate_risk_analysis(
        frame,
        analysis_kind=analysis_kind,
        column_map={str(key): str(value) for key, value in raw_column_map.items()},
    )
    payload = RiskAnalysisReportPayload(
        analysis_kind=calculation.analysis_kind,
        column_map=calculation.column_map,
        product_scope=calculation.product_scope,
        as_of_period=calculation.as_of_period,
        headline_metrics=calculation.headline_metrics,
        key_points=calculation.key_points,
        red_flags=calculation.red_flags,
        assumptions=calculation.assumptions,
        source_row_count=calculation.source_row_count,
        row_count=calculation.row_count,
        detail_rows=calculation.detail_rows,
        summary_rows=calculation.summary_rows,
        formula_definitions=calculation.formula_definitions,
        data_quality=calculation.data_quality,
    )

    outputs_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "outputs"
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(outputs_dir, "risk_analysis_report.xlsx")
    try:
        render_risk_analysis_report(payload, artifact.path)
        audit = {
            "kind": "risk_analysis.report.generated",
            "target_ref": str(ctx.task_id),
            "outcome": "succeeded",
            "detail": {
                "analysis_kind": calculation.analysis_kind,
                "dataset_id": dataset.id,
                "report_path": str(artifact.final_path),
                "source_row_count": calculation.source_row_count,
                "row_count": calculation.row_count,
                "product_scope": calculation.product_scope,
                "as_of_period": calculation.as_of_period,
                "column_map": calculation.column_map,
            },
        }
        uow.finalize_with_connection(
            runtime.repo.transaction,
            lambda conn: runtime.repo.write_audit_on_connection(conn, **audit),
        )
    except Exception:
        uow.rollback()
        raise

    return {
        "analysis_kind": calculation.analysis_kind,
        "report_path": str(artifact.final_path),
        "headline_metrics": calculation.headline_metrics,
        "key_points": calculation.key_points,
        "red_flags": calculation.red_flags,
        "assumptions": calculation.assumptions,
        "source_row_count": calculation.source_row_count,
        "row_count": calculation.row_count,
        "column_map": calculation.column_map,
        "product_scope": calculation.product_scope,
        "as_of_period": calculation.as_of_period,
    }


def _owned_dataset(runtime: _Runtime, dataset_id: str, *, task_id: str):
    try:
        dataset = runtime.registry.get(dataset_id)
    except KeyError:
        raise RiskAnalysisError(
            f"dataset not found: {dataset_id}",
            field="dataset_id",
        ) from None
    if str(dataset.task_id) != str(task_id):
        raise RiskAnalysisError(
            f"dataset not found: {dataset_id}",
            field="dataset_id",
        )
    return dataset


__all__ = ["tool_generate_risk_analysis_report"]
