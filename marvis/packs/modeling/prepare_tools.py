from __future__ import annotations

import pandas as pd
import uuid
from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.direction import normalize_score_direction
from marvis.packs.modeling.prepare import SPLIT_COLUMN, prepare_modeling_frame
from marvis.packs.modeling.readiness import check_data_quality, modeling_readiness
from marvis.packs.modeling.reject_inference import reject_inference

from marvis.packs.modeling._common import _effective_seed, _json_safe, _jsonable, _optional_float, _optional_str
from marvis.packs.modeling._runtime import _resolve_feature_cols, _runtime


def tool_check_data_quality(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    issues = check_data_quality(
        runtime.backend,
        dataset,
        runtime.registry.resolve_path(dataset.id),
        target_col=_optional_str(inputs.get("target_col")),
    )
    return {"issues": [_jsonable(issue) for issue in issues]}


def tool_modeling_readiness(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    return _jsonable(
        modeling_readiness(
            runtime.backend,
            dataset,
            runtime.registry.resolve_path(dataset.id),
            target_col=str(inputs["target_col"]),
            split_col=_optional_str(inputs.get("split_col")),
        )
    )


def tool_reject_inference(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id))
    result = reject_inference(
        frame,
        target_col=str(inputs["target_col"]),
        decision_col=str(inputs["decision_col"]),
        method=str(inputs.get("method") or "parceling"),
        score_col=_optional_str(inputs.get("score_col")),
        reject_bad_rate=_optional_float(inputs.get("reject_bad_rate")),
        reject_weight=float(inputs.get("reject_weight") or 1.0),
        score_direction=normalize_score_direction(_optional_str(inputs.get("score_direction"))),
        confirm_direction_conflict=bool(inputs.get("confirm_direction_conflict")),
    )
    out_dir = runtime.datasets_root / str(ctx.task_id) / "modeling"
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(out_dir, f"reject_inference_{uuid.uuid4().hex}.parquet")
    try:
        result.frame.to_parquet(artifact.path, index=False)
        def audit_factory(registered_dataset):
            return {
                "kind": "modeling.reject_inference.created",
                "target_ref": registered_dataset.id,
                "outcome": "succeeded",
                "detail": {
                    "source_dataset_id": dataset.id,
                    "method": str(inputs.get("method") or "parceling"),
                    "target_col": str(inputs["target_col"]),
                    "decision_col": str(inputs["decision_col"]),
                    "sample_weight_col": result.sample_weight_col,
                },
            }

        registered = uow.finalize_with_connection(
            runtime.repo.transaction,
            lambda conn: runtime.registry.register_existing_with_audit_on_connection(
                conn,
                artifact.final_path,
                audit_factory=audit_factory,
                task_id=str(ctx.task_id),
                role="reject_inference",
                anchor_target=dataset.id,
                seed=_effective_seed(inputs, ctx),
            ),
        )
    except Exception:
        uow.rollback()
        raise
    return {
        "result_dataset_id": registered.id,
        "target_col": result.target_col,
        "sample_weight_col": result.sample_weight_col,
        "diagnostics": _jsonable(result.diagnostics),
    }


def tool_prepare_modeling_frame(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    split_col = _optional_str(inputs.get("split_col"))
    feature_cols = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("feature_cols") or [],
        target_col=str(inputs["target_col"]),
        split_col=split_col,
    )
    result = prepare_modeling_frame(
        runtime.registry,
        runtime.backend,
        dataset.id,
        target_col=str(inputs["target_col"]),
        feature_cols=feature_cols,
        split_col=split_col,
        split_config=inputs.get("split_config") or {},
        passthrough_cols=[str(item) for item in inputs.get("passthrough_cols") or [] if str(item).strip()],
        seed=_effective_seed(inputs, ctx),
        audit_kind="modeling.dataset.derived",
        audit_detail={"tool": "prepare_modeling_frame"},
    )
    split_col = split_col or "split"
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(result.id), columns=[split_col])
    counts = {
        str(key): int(value)
        for key, value in frame[split_col].value_counts().sort_index().items()
    }
    return {
        "result_dataset_id": result.id,
        "split_counts": counts,
        "split_col": split_col,
        "split_values": {key: key for key in counts},
        "holdout_values": ["oot"] if "oot" in counts else [],
        "feature_cols": feature_cols,
    }


def tool_make_split(inputs: dict, ctx) -> dict:
    """MODELING G1 split gate: build a derived modeling frame from an arbitrary
    rule set (e.g. channel A → train, channel B before a cutoff → test) plus the
    existing random/time fallback, then return per-split counts and, when month or
    channel columns are present, a per-split × per-group distribution table for the
    confirmation gate UI."""
    runtime = _runtime(ctx)
    # split_col present → pass the EXISTING split through unchanged (the gate just surfaces
    # it for review); absent → generate from split_config (rules / time-OOT / grouped-random
    # fallback). prepare_modeling_frame keeps the passed-through column's name, and names a
    # generated column SPLIT_COLUMN, so the effective name is one or the other.
    split_col = str(inputs["split_col"]) if inputs.get("split_col") else None
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    feature_cols = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("feature_cols") or [],
        target_col=str(inputs["target_col"]),
        split_col=split_col,
    )
    result = prepare_modeling_frame(
        runtime.registry,
        runtime.backend,
        dataset.id,
        target_col=str(inputs["target_col"]),
        feature_cols=feature_cols,
        split_col=split_col,
        split_config=inputs.get("split_config") or {},
        passthrough_cols=[str(item) for item in inputs.get("passthrough_cols") or [] if str(item).strip()],
        seed=_effective_seed(inputs, ctx),
        audit_kind="modeling.dataset.derived",
        audit_detail={"tool": "make_split"},
    )
    effective_split_col = split_col or SPLIT_COLUMN
    split_frame = runtime.backend.read_frame(
        runtime.registry.resolve_path(result.id), columns=[effective_split_col]
    )
    dataset_path = runtime.registry.resolve_path(dataset.id)
    source_columns = [profile.name for profile in dataset.columns] or runtime.backend.column_names(dataset_path)
    group_columns = _detect_group_columns(source_columns)
    source_frame = (
        runtime.backend.read_frame(dataset_path, columns=group_columns)
        if group_columns
        else pd.DataFrame(index=split_frame.index)
    )
    sample_analysis = _split_sample_analysis(split_frame[effective_split_col], source_frame)
    split_counts = sample_analysis["split_counts"]
    return {
        "result_dataset_id": result.id,
        "split_col": effective_split_col,
        "split_values": {key: key for key in split_counts},
        "holdout_values": ["oot"] if "oot" in split_counts else [],
        "feature_cols": feature_cols,
        "sample_analysis": _json_safe(sample_analysis),
    }


_GROUP_COLUMN_HINTS = ("month", "channel", "渠道", "月", "split_month")


def _split_sample_analysis(split_series: pd.Series, source_frame: pd.DataFrame) -> dict:
    """Row counts per split plus, for each detected month/channel-like column, a
    per-split × per-group count table. The split frame and the source frame share row
    order (prepare_modeling_frame preserves it), so we align by position."""
    splits = [str(value) for value in split_series.tolist()]
    counts: dict[str, int] = {}
    for split in splits:
        counts[split] = counts.get(split, 0) + 1
    group_tables: dict[str, dict] = {}
    for column in _detect_group_columns(source_frame.columns):
        values = source_frame[column].astype("object").where(source_frame[column].notna(), None)
        table: dict[str, dict[str, int]] = {}
        for split, value in zip(splits, values.tolist()):
            key = "(missing)" if value is None else str(value)
            row = table.setdefault(split, {})
            row[key] = row.get(key, 0) + 1
        group_tables[str(column)] = {split: dict(sorted(row.items())) for split, row in table.items()}
    return {
        "split_counts": dict(sorted(counts.items())),
        "total_rows": len(splits),
        "group_distributions": group_tables,
    }


def _detect_group_columns(columns) -> list[str]:
    return [str(column) for column in columns if any(hint in str(column) for hint in _GROUP_COLUMN_HINTS)]
