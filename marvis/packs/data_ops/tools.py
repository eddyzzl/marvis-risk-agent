from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.dedup import two_level_dedup
from marvis.data.errors import DedupRequiredError
from marvis.data.excel_ingest import ingest_sheet, list_sheets
from marvis.data.join_engine import JoinEngine
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository
from marvis.settings import build_settings


def tool_ingest_excel(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    path = Path(str(inputs["path"]))
    requested_sheets = inputs.get("sheets") or list_sheets(path)
    role = str(inputs.get("role") or "feature")
    out_dir = runtime.datasets_root / ctx.task_id / "excel"
    datasets = []
    reports = []
    for sheet in requested_sheets:
        parquet_path, report = ingest_sheet(path, str(sheet), out_dir)
        dataset = runtime.registry.register_existing(
            parquet_path,
            task_id=ctx.task_id,
            role=role,
            seed=_seed(ctx),
        )
        datasets.append(_dataset_payload(dataset))
        reports.append({
            "sheet": report.sheet,
            "header_rows": report.header_rows,
            "data_start_row": report.data_start_row,
            "flattened_columns": report.flattened_columns,
            "original_shape": list(report.original_shape),
            "warnings": [],
        })
    return {"datasets": datasets, "reports": reports}


def tool_infer_schema(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    return {
        "dataset_id": dataset.id,
        "columns": [_column_payload(column) for column in dataset.columns],
        "has_target": dataset.has_target,
        "target_col": dataset.target_col,
    }


def tool_align_columns(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    anchor = runtime.registry.get(str(inputs["anchor_id"]))
    anchor_path = runtime.registry.resolve_path(anchor.id)
    alignments = []
    for feature_id in inputs.get("feature_ids") or []:
        feature = runtime.registry.get(str(feature_id))
        key_pairs = runtime.aligner.align(
            anchor,
            anchor_path,
            feature,
            runtime.registry.resolve_path(feature.id),
            seed=_seed(ctx),
        )
        alignments.append({
            "feature_id": feature.id,
            "key_pairs": [_key_pair_payload(pair) for pair in key_pairs],
        })
    return {"alignments": alignments}


def tool_propose_join(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    plan = runtime.join_engine.propose_join_plan(
        str(inputs["anchor_id"]),
        [str(item) for item in inputs.get("feature_ids") or []],
        ctx.task_id,
        seed=_seed(ctx),
    )
    payload = _join_plan_payload(plan)
    for join in payload.get("joins", []):
        join["feature_name"] = _friendly_name(runtime.registry, join.get("feature_id"))
    return payload


def tool_confirm_join(inputs: dict, ctx) -> dict:
    """Confirm a proposed join plan's feature specs so execute_join is allowed.

    Confirmation is per-feature (the engine's forced-confirmation invariant): a
    feature whose join key is not unique requires a dedup strategy ("first"/"last")
    or the engine refuses. ``dedup_strategies`` maps feature_dataset_id -> strategy.

    A feature needing a strategy that wasn't supplied is reported in ``needs_dedup``
    (status="needs_dedup") rather than HARD-FAILING the plan: the conversational flow
    then reaches the C2 gate (which surfaces the conflicts), where the user supplies the
    strategy and re-confirms. The mutating execute_join still refuses to run until every
    spec is confirmed, so nothing is silently joined.
    """
    runtime = _runtime(ctx)
    join_plan_id = str(inputs["join_plan_id"])
    strategies = inputs.get("dedup_strategies") or {}
    plan = runtime.repo.load_join_plan(join_plan_id)
    confirmed: list[str] = []
    needs_dedup: list[str] = []
    for spec in plan.joins:
        feature_id = spec.feature_dataset_id
        strategy = strategies.get(feature_id)
        try:
            runtime.join_engine.confirm_join_spec(
                join_plan_id, feature_id, dedup_strategy=strategy
            )
            confirmed.append(feature_id)
        except DedupRequiredError:
            needs_dedup.append(feature_id)
    return {
        "join_plan_id": join_plan_id,
        "confirmed": confirmed,
        "needs_dedup": needs_dedup,
        # friendly file names for the gate message (raw ids stay in needs_dedup for the picker)
        "needs_dedup_labels": {fid: _friendly_name(runtime.registry, fid) for fid in needs_dedup},
        "status": "needs_dedup" if needs_dedup else "confirmed",
    }


def tool_execute_join(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    plan = runtime.repo.load_join_plan(str(inputs["join_plan_id"]))
    anchor = runtime.registry.get(plan.anchor_dataset_id)
    result = runtime.join_engine.execute_join_plan(
        plan.id,
        out_dir=runtime.datasets_root / ctx.task_id / "joins",
    )
    # §8 stage-completion summary from real per-table diagnostics (no longer hard-coded).
    per_table = []
    warnings = []
    for spec in plan.joins:
        diag = spec.diagnostics
        per_table.append({
            "feature_id": spec.feature_dataset_id,
            "match_rate": round(float(diag.match_rate), 4),
            "new_columns": int(diag.new_columns),
            "new_columns_null_rate": round(float(diag.new_columns_null_rate), 4),
            "dedup_strategy": spec.dedup_strategy or "无",
        })
        if diag.shrink_detected:
            warnings.append(
                f"{spec.feature_dataset_id}:命中率偏低({diag.match_rate:.2f}),新列缺失较多"
            )
        # conflict_report is a ConflictReport in-memory but an asdict-flattened dict after a
        # DB round-trip (load_join_plan) — handle both so the warning never crashes here.
        report = getattr(diag, "conflict_report", None)
        if isinstance(report, dict):
            conflict_keys = int(report.get("n_conflict_keys") or 0)
        elif report is not None:
            conflict_keys = int(getattr(report, "n_conflict_keys", 0) or 0)
        else:
            conflict_keys = 0
        if conflict_keys and spec.dedup_strategy:
            warnings.append(
                f"{spec.feature_dataset_id}:{conflict_keys} 个同键冲突已按 "
                f"'{spec.dedup_strategy}' 解决"
            )
    return {
        "result_dataset_id": result.id,
        "anchor_rows": anchor.row_count,
        "joined_rows": result.row_count,
        "fan_out": result.row_count > anchor.row_count,
        "warnings": warnings,
        "per_table": per_table,
    }


def tool_clean_format(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    path = runtime.registry.resolve_path(dataset.id)
    frame = runtime.backend.read_frame(path)
    changed_columns = []
    for operation in inputs.get("ops") or []:
        column = str(operation["col"])
        op = str(operation["op"])
        if column not in frame.columns:
            raise KeyError(f"unknown column: {column}")
        frame[column] = _apply_clean_op(frame[column], op)
        changed_columns.append(column)
    out_path = runtime.datasets_root / ctx.task_id / "clean" / f"{dataset.id}_clean.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    result = runtime.registry.register_existing(
        out_path,
        task_id=ctx.task_id,
        role=dataset.role,
        anchor_target=dataset.id,
        seed=_seed(ctx),
    )
    return {"dataset_id": result.id, "changed_columns": changed_columns}


def tool_dedup_rows(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    keys = [str(item) for item in inputs.get("keys") or []]
    strategy = inputs.get("strategy")
    path = runtime.registry.resolve_path(dataset.id)
    frame = runtime.backend.read_frame(path)
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise KeyError(f"unknown keys: {', '.join(missing)}")
    before = len(frame)
    # Level-1 safe dedup (always) + level-2 conflict detection (never auto-dropped).
    deduped, report = two_level_dedup(frame, keys)
    # A same-key value-conflict is only resolved on an EXPLICIT, deterministic strategy
    # (spec §6: 告警不静默删). With no strategy, conflicts are surfaced for review.
    needs_conflict_review = report.has_conflicts and not strategy
    if strategy and report.has_conflicts and keys:
        keep = "first" if str(strategy) == "first" else "last"
        deduped = deduped.drop_duplicates(subset=keys, keep=keep, ignore_index=True)
    out_path = runtime.datasets_root / ctx.task_id / "dedup" / f"{dataset.id}_dedup.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_parquet(out_path, index=False)
    result = runtime.registry.register_existing(
        out_path,
        task_id=ctx.task_id,
        role=dataset.role,
        anchor_target=dataset.id,
        seed=_seed(ctx),
    )
    return {
        "dataset_id": result.id,
        "removed_rows": before - len(deduped),
        "safe_dropped": report.safe_dropped,
        "needs_conflict_review": needs_conflict_review,
        "conflict_report": _conflict_report_json(report),
    }


def _conflict_report_json(report) -> dict:
    return {
        "key_columns": list(report.key_columns),
        "conflict_columns": list(report.conflict_columns),
        "n_conflict_keys": report.n_conflict_keys,
        "n_conflict_rows": report.n_conflict_rows,
        "safe_dropped": report.safe_dropped,
        "sample_keys": [list(key) for key in report.sample_keys],
    }


class _Runtime:
    def __init__(self, ctx):
        settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self.aligner = ColumnAligner(self.backend)
        self.join_engine = JoinEngine(self.backend, self.aligner, self.registry, self.repo)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _seed(ctx) -> int:
    return int(ctx.seed or 0)


def _apply_clean_op(series: pd.Series, op: str) -> pd.Series:
    if op == "strip":
        return series.map(lambda value: value.strip() if isinstance(value, str) else value)
    if op == "lower":
        return series.map(lambda value: value.lower() if isinstance(value, str) else value)
    if op == "upper":
        return series.map(lambda value: value.upper() if isinstance(value, str) else value)
    if op == "to_numeric":
        return pd.to_numeric(series, errors="coerce")
    if op == "to_datetime":
        return pd.to_datetime(series, errors="coerce")
    raise ValueError(f"unsupported clean op: {op}")


def _dataset_payload(dataset) -> dict:
    return {
        "id": dataset.id,
        "task_id": dataset.task_id,
        "role": dataset.role,
        "source_path": dataset.source_path,
        "format": dataset.format,
        "sheet": dataset.sheet,
        "row_count": dataset.row_count,
        "columns": [_column_payload(column) for column in dataset.columns],
        "has_target": dataset.has_target,
        "target_col": dataset.target_col,
    }


def _column_payload(column) -> dict:
    return {
        "name": column.name,
        "dtype": column.dtype,
        "semantic_role": column.semantic_role,
        "null_rate": column.null_rate,
        "cardinality": column.cardinality,
        "sample_values": list(column.sample_values),
        "fingerprint": asdict(column.fingerprint),
    }


def _key_pair_payload(pair) -> dict:
    return {
        "anchor_col": pair.anchor_col,
        "feature_col": pair.feature_col,
        "match_method": pair.match_method,
        "transform_side": pair.transform_side,
        "match_rate": pair.match_rate,
        "resolved_by": pair.resolved_by,
    }


def _diagnostics_payload(diagnostics) -> dict:
    return asdict(diagnostics)


def _friendly_name(registry, dataset_id) -> str:
    """A human-readable file name for a dataset id (e.g. ``features.parquet``) so the
    diagnostics / dedup gate show the source file rather than a raw ``ds_<hash>``."""
    try:
        dataset = registry.get(str(dataset_id))
        source = getattr(dataset, "source_path", None)
        return Path(source).name if source else str(dataset_id)
    except Exception:
        return str(dataset_id)


def _join_plan_payload(plan) -> dict:
    return {
        "join_plan_id": plan.id,
        "anchor_dataset_id": plan.anchor_dataset_id,
        "status": plan.status,
        "joins": [
            {
                "feature_id": spec.feature_dataset_id,
                "key_pairs": [_key_pair_payload(pair) for pair in spec.key_pairs],
                "diagnostics": _diagnostics_payload(spec.diagnostics),
                "dedup_strategy": spec.dedup_strategy,
                "confirmed": spec.confirmed,
            }
            for spec in plan.joins
        ],
    }
