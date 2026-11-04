"""Strategy Tools for evidence-only leakage, selection-bias and segment diagnostics.

These wire :mod:`marvis.packs.strategy.leakage_diagnostics` and
:mod:`marvis.packs.strategy.fairness_evidence` into the strategy pack. Each tool
reads task-owned datasets, returns aggregate evidence only (counts, rates,
column summaries, red-flag markers) and never blocks a workflow, adopts a
strategy or renders a "leaked / biased / unfair" verdict.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.fairness_evidence import (
    NEUTRAL_PHRASING_DRAFT_NOTICE,
    NEUTRAL_PHRASING_TEMPLATES,
    segment_disparity_evidence,
)
from marvis.packs.strategy.leakage_diagnostics import (
    selection_bias_diagnostics,
    time_leakage_diagnostics,
)


def _owned_frame(runtime, dataset_id: str, *, task_id: str, columns: list[str]) -> pd.DataFrame:
    dataset = runtime.registry.get(dataset_id)
    if str(dataset.task_id) != str(task_id):
        raise StrategyError(f"dataset not found: {dataset_id}")
    return runtime.backend.read_frame(
        runtime.registry.resolve_path(dataset.id), columns=columns
    )


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def run_diagnose_time_leakage(inputs: dict, ctx, runtime) -> dict[str, Any]:
    feature_cols = [str(column) for column in inputs["feature_cols"]]
    feature_time_cols = inputs.get("feature_time_cols")
    columns = _unique(
        [
            *feature_cols,
            *(dict(feature_time_cols or {}).values()),
            inputs.get("observation_time_col"),
            inputs.get("label_window_start_col"),
            inputs.get("label_window_end_col"),
        ]
    )
    frame = _owned_frame(
        runtime, str(inputs["dataset_id"]), task_id=str(ctx.task_id), columns=columns
    )
    result = time_leakage_diagnostics(
        frame,
        feature_cols=feature_cols,
        feature_time_cols=feature_time_cols,
        observation_time_col=inputs.get("observation_time_col"),
        label_window_start_col=inputs.get("label_window_start_col"),
        label_window_end_col=inputs.get("label_window_end_col"),
        label_window_start=inputs.get("label_window_start"),
        label_window_end=inputs.get("label_window_end"),
    )
    return {"evidence_only": True, **result.to_dict()}


def run_diagnose_selection_bias(inputs: dict, ctx, runtime) -> dict[str, Any]:
    columns = [str(column) for column in inputs["columns"]]
    target = str(inputs["target"])
    read_columns = _unique([*columns, target])
    task_id = str(ctx.task_id)
    approval = _owned_frame(
        runtime, str(inputs["approval_dataset_id"]), task_id=task_id, columns=read_columns
    )
    risk = _owned_frame(
        runtime, str(inputs["risk_dataset_id"]), task_id=task_id, columns=read_columns
    )
    result = selection_bias_diagnostics(
        approval,
        risk,
        columns,
        target,
        target_bad_value=inputs.get("target_bad_value", 1),
    )
    return {"evidence_only": True, **result.to_dict()}


def _segment_payload(segment) -> dict[str, Any]:
    # SegmentEvidence holds a read-only mapping that neither asdict nor the
    # worker's pickle can copy; emit a plain dict.
    payload = {name: getattr(segment, name) for name in segment.__dataclass_fields__}
    distribution = payload["reason_code_distribution"]
    payload["reason_code_distribution"] = None if distribution is None else dict(distribution)
    return payload


def run_segment_disparity_evidence(inputs: dict, ctx, runtime) -> dict[str, Any]:
    segment_col = str(inputs["segment_col"])
    target_col = str(inputs["target_col"])
    outcome_columns = {"action": str(inputs["action_col"])}
    if inputs.get("reason_code_col"):
        outcome_columns["reason_code"] = str(inputs["reason_code_col"])
    frame = _owned_frame(
        runtime,
        str(inputs["dataset_id"]),
        task_id=str(ctx.task_id),
        columns=_unique([segment_col, target_col, *outcome_columns.values()]),
    )
    result = segment_disparity_evidence(frame, segment_col, outcome_columns, target_col)
    return {
        "evidence_only": True,
        "segment_column": result.segment_column,
        "action_column": result.action_column,
        "reason_code_column": result.reason_code_column,
        "target_column": result.target_column,
        "overall": _segment_payload(result.overall),
        "segments": [_segment_payload(segment) for segment in result.segments],
        "phrasing_draft_notice": NEUTRAL_PHRASING_DRAFT_NOTICE,
        "phrasing_templates": list(NEUTRAL_PHRASING_TEMPLATES),
    }
