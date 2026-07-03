from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path

import pandas as pd

from marvis.agent.data_dictionary import first_data_dictionary_id, load_business_names
from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.align import ColumnAligner
from marvis.data.backend import (
    connect_duckdb,
    sql_identifier,
)
from marvis.data.dedup import two_level_dedup
from marvis.data.errors import DedupRequiredError
from marvis.data.excel_ingest import ingest_sheet, list_sheets
from marvis.data.join_engine import JoinEngine
from marvis.db_schema import connect
from marvis.plugins.sdk import PackRuntime
from marvis.repositories.strategy import _write_audit_row
from marvis.safe_paths import assert_within


def tool_ingest_excel(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    path = _resolve_material_path(str(inputs["path"]), ctx)
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


def _resolve_material_path(raw_path: str, ctx) -> Path:
    path = Path(raw_path).expanduser()
    resolved = path.resolve(strict=True)
    roots = _allowed_material_roots(ctx)
    if any(_path_is_within(root, resolved) for root in roots):
        return resolved
    allowed = ", ".join(str(root) for root in roots)
    raise PermissionError(
        f"Excel path must be under an allowed material root: {allowed}. "
        "Set RMC_MATERIAL_ROOTS to allow another local material directory."
    )


def _allowed_material_roots(ctx) -> tuple[Path, ...]:
    roots = [Path(ctx.workspace), Path.home()]
    extra_roots = os.environ.get("RMC_MATERIAL_ROOTS", "")
    roots.extend(Path(raw).expanduser() for raw in extra_roots.split(os.pathsep) if raw)
    resolved: list[Path] = []
    for root in roots:
        candidate = root.resolve()
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def _path_is_within(root: Path, candidate: Path) -> bool:
    try:
        assert_within(root, candidate)
    except PermissionError:
        return False
    return True


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
    # GAP-4: {column: business_name} map for every key column in this proposal, so
    # the C1/dedup gate can show a meaning tooltip next to raw column-name codes.
    # Best-effort — {} when the task has no registered data dictionary.
    dictionary = _join_dictionary(runtime, ctx, payload)
    if dictionary:
        payload["dictionary"] = dictionary
    return payload


def _join_dictionary(runtime: "_Runtime", ctx, payload: dict) -> dict:
    dictionary_id = first_data_dictionary_id(runtime.registry.list_for_task(ctx.task_id))
    if not dictionary_id:
        return {}
    names = load_business_names(runtime.backend, runtime.registry, dictionary_id)
    if not names:
        return {}
    columns: set[str] = set()
    for join in payload.get("joins", []):
        for pair in join.get("key_pairs") or []:
            if pair.get("anchor_col"):
                columns.add(str(pair["anchor_col"]))
            if pair.get("feature_col"):
                columns.add(str(pair["feature_col"]))
    return {column: names[column] for column in columns if column in names}


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
    result = _register_derived_frame(
        runtime,
        ctx,
        frame,
        subdir="clean",
        filename=f"{dataset.id}_clean.parquet",
        role=dataset.role,
        anchor_target=dataset.id,
    )
    return {"dataset_id": result.id, "changed_columns": changed_columns}


def _register_derived_frame(
    runtime,
    ctx,
    frame: pd.DataFrame,
    *,
    subdir: str,
    filename: str,
    role: str,
    anchor_target: str,
):
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(runtime.datasets_root / ctx.task_id / subdir, filename)
    try:
        frame.to_parquet(artifact.path, index=False)
        register_on_connection = getattr(runtime.registry, "register_existing_on_connection", None)
        if callable(register_on_connection):
            return uow.finalize_with_connection(
                runtime.repo.transaction,
                lambda conn: register_on_connection(
                    conn,
                    artifact.final_path,
                    task_id=ctx.task_id,
                    role=role,
                    anchor_target=anchor_target,
                    seed=_seed(ctx),
                ),
            )
        return uow.finalize(
            lambda: runtime.registry.register_existing(
                artifact.final_path,
                task_id=ctx.task_id,
                role=role,
                anchor_target=anchor_target,
                seed=_seed(ctx),
            )
        )
    except Exception:
        uow.rollback()
        raise


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
    result = _register_derived_frame(
        runtime,
        ctx,
        deduped,
        subdir="dedup",
        filename=f"{dataset.id}_dedup.parquet",
        role=dataset.role,
        anchor_target=dataset.id,
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


# ---------------------------------------------------------------------------
# S6 ad-hoc slice/aggregate: a deterministic, whitelisted group-by aggregate over
# a registered dataset. Every group_by/metric/filter column is validated against
# the dataset's column profile (``sql_identifier`` raises on any unknown name), the
# op->SQL mapping is a fixed dictionary, and a single parameterized DuckDB SQL is
# compiled with an explicit ORDER BY -- so the LLM only ever produces a structured
# spec (it never computes a number), and a `; DROP` style injected column name is
# rejected as a typed error before any SQL runs (INV-1).
# ---------------------------------------------------------------------------

# Whitelisted aggregate operators. Each maps to a DuckDB SQL template that takes a
# single already-quoted column identifier. bad_rate/approval_rate encode the fixed
# credit-risk conventions (mean of a 0/1 target, share of an approve decision).
_AGG_COMPARATORS = {"==": "=", "!=": "<>", ">": ">", ">=": ">=", "<": "<", "<=": "<="}
_MAX_GROUP_BY = 3
_MAX_FILTERS = 8
_DEFAULT_TOP_K = 50


def tool_slice_aggregate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    dataset = runtime.registry.get(dataset_id)
    path = runtime.registry.resolve_path(dataset.id)
    # The column whitelist IS the dataset profile: only names the backend can see in
    # the physical file are legal anywhere in the spec (group_by/metrics/filters/
    # month_col/sort_by). Anything else -> DataSecurityError from sql_identifier.
    allowed_columns = set(runtime.backend.column_names(path))

    group_by = [str(col) for col in (inputs.get("group_by") or [])]
    if len(group_by) > _MAX_GROUP_BY:
        raise ValueError(f"group_by supports at most {_MAX_GROUP_BY} columns")
    metrics = [dict(metric) for metric in (inputs.get("metrics") or []) if isinstance(metric, dict)]
    if not metrics:
        raise ValueError("slice_aggregate requires at least one metric")
    filters = [dict(f) for f in (inputs.get("filters") or []) if isinstance(f, dict)]
    if len(filters) > _MAX_FILTERS:
        raise ValueError(f"filters supports at most {_MAX_FILTERS} conditions")
    top_k = int(inputs.get("top_k") or _DEFAULT_TOP_K)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    group_sql = [sql_identifier(col, allowed_columns) for col in group_by]
    metric_selects, metric_labels = _metric_selects(metrics, allowed_columns)
    where_sql, where_params = _filter_clause(filters, allowed_columns)
    month_where_sql, month_params = _month_clause(
        _optional_str(inputs.get("month_col")),
        inputs.get("months"),
        allowed_columns,
    )
    all_where = [clause for clause in (where_sql, month_where_sql) if clause]
    where_clause = f" WHERE {' AND '.join(all_where)}" if all_where else ""

    order_sql = _order_clause(
        _optional_str(inputs.get("sort_by")), group_by, metric_labels, allowed_columns
    )
    rel = runtime.backend._duckdb_rel(path)  # parquet_rel/csv_rel -- read-only scan
    select_parts = [*group_sql, *metric_selects]
    query = (
        f"SELECT {', '.join(select_parts)} FROM {rel}{where_clause}"
        + (f" GROUP BY {', '.join(group_sql)}" if group_sql else "")
        + f" ORDER BY {order_sql}"
        + f" LIMIT {int(top_k) + 1}"  # fetch one extra row to detect truncation
    )
    params = [*where_params, *month_params]
    with connect_duckdb(runtime.backend._temp_directory) as conn:
        scanned_row = conn.execute(f"SELECT count(*) FROM {rel}{where_clause}", params).fetchone()
        frame = conn.execute(query, params).df()

    n_rows_scanned = int(scanned_row[0] or 0)
    truncated = len(frame) > top_k
    if truncated:
        frame = frame.head(top_k)
    columns = [*group_by, *metric_labels]
    rows = [
        {column: _jsonable_cell(value) for column, value in zip(columns, record, strict=True)}
        for record in frame.itertuples(index=False, name=None)
    ]

    red_flags: list[dict] = []
    if not rows:
        red_flags.append({
            "code": "empty_result",
            "level": "amber",
            "message": "当前口径下无匹配样本，请检查筛选条件或时间范围。",
        })
    if truncated:
        red_flags.append({
            "code": "truncated",
            "level": "amber",
            "message": f"结果超过 top_k={top_k} 行已截断，请收窄分组或加筛选。",
        })

    spec_echo = {
        "dataset_id": dataset_id,
        "group_by": group_by,
        "metrics": [{"op": str(m.get("op")), "col": _optional_str(m.get("col"))} for m in metrics],
        "filters": [
            {"col": str(f.get("col")), "op": str(f.get("op")), "value": _jsonable_cell(f.get("value"))}
            for f in filters
        ],
        "month_col": _optional_str(inputs.get("month_col")),
        "months": [str(month) for month in (inputs.get("months") or [])],
        "top_k": top_k,
        "sort_by": _optional_str(inputs.get("sort_by")),
    }

    with connect(runtime.settings.db_path) as conn:
        _write_audit_row(
            conn,
            kind="data.slice_aggregate",
            target_ref=dataset_id,
            outcome="succeeded",
            detail={
                "task_id": str(ctx.task_id),
                "group_by": group_by,
                "metrics": spec_echo["metrics"],
                "n_rows_scanned": n_rows_scanned,
                "n_rows_returned": len(rows),
                "truncated": truncated,
            },
        )

    return {
        "columns": columns,
        "rows": rows,
        "spec_echo": spec_echo,
        "n_rows_scanned": n_rows_scanned,
        "red_flags": red_flags,
    }


def _metric_selects(metrics: list[dict], allowed_columns: set[str]) -> tuple[list[str], list[str]]:
    """(select_expr, output_label) per metric. The op->SQL mapping is a fixed dict so
    an LLM can only pick an operator name, never inject an expression; the target
    column (when the op needs one) is validated against the profile whitelist."""
    selects: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    for metric in metrics:
        op = str(metric.get("op") or "")
        col = _optional_str(metric.get("col"))
        label = _metric_label(op, col)
        if label in seen:
            raise ValueError(f"duplicate metric label: {label}")
        seen.add(label)
        selects.append(f"{_metric_expr(op, col, allowed_columns)} AS {_quote(label)}")
        labels.append(label)
    return selects, labels


def _metric_expr(op: str, col: str | None, allowed_columns: set[str]) -> str:
    if op == "count":
        return "count(*)"
    if op in {"sum", "mean", "min", "max", "distinct"}:
        if not col:
            raise ValueError(f"metric op {op!r} requires a column")
        ident = sql_identifier(col, allowed_columns)
        numeric = f"try_cast({ident} AS DOUBLE)"
        return {
            "sum": f"coalesce(sum({numeric}), 0)",
            "mean": f"avg({numeric})",
            "min": f"min({numeric})",
            "max": f"max({numeric})",
            "distinct": f"count(DISTINCT {ident})",
        }[op]
    if op == "bad_rate":
        if not col:
            raise ValueError("metric op 'bad_rate' requires the target column")
        ident = sql_identifier(col, allowed_columns)
        return f"avg(CASE WHEN try_cast({ident} AS DOUBLE) = 1 THEN 1.0 ELSE 0.0 END)"
    if op == "approval_rate":
        if not col:
            raise ValueError("metric op 'approval_rate' requires the decision column")
        ident = sql_identifier(col, allowed_columns)
        return f"avg(CASE WHEN lower(trim(CAST({ident} AS VARCHAR))) = 'approve' THEN 1.0 ELSE 0.0 END)"
    raise ValueError(f"unsupported metric op: {op}")


def _metric_label(op: str, col: str | None) -> str:
    return op if op == "count" or not col else f"{op}_{col}"


def _filter_clause(filters: list[dict], allowed_columns: set[str]) -> tuple[str, list]:
    """Compile filters into a parameterized WHERE (values bound, never interpolated)."""
    clauses: list[str] = []
    params: list = []
    for f in filters:
        col = str(f.get("col") or "")
        op = str(f.get("op") or "")
        value = f.get("value")
        ident = sql_identifier(col, allowed_columns)
        if op in _AGG_COMPARATORS:
            clauses.append(f"{ident} {_AGG_COMPARATORS[op]} ?")
            params.append(value)
        elif op == "in":
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            if not values:
                raise ValueError("filter op 'in' requires a non-empty value list")
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"{ident} IN ({placeholders})")
            params.extend(values)
        elif op == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("filter op 'between' requires a [low, high] value pair")
            clauses.append(f"{ident} BETWEEN ? AND ?")
            params.extend([value[0], value[1]])
        else:
            raise ValueError(f"unsupported filter op: {op}")
    return " AND ".join(clauses), params


def _month_clause(month_col: str | None, months, allowed_columns: set[str]) -> tuple[str, list]:
    if not month_col:
        return "", []
    month_values = [str(month) for month in (months or [])]
    if not month_values:
        return "", []
    ident = sql_identifier(month_col, allowed_columns)
    placeholders = ", ".join("?" for _ in month_values)
    return f"CAST({ident} AS VARCHAR) IN ({placeholders})", month_values


def _order_clause(
    sort_by: str | None,
    group_by: list[str],
    metric_labels: list[str],
    allowed_columns: set[str],
) -> str:
    """Explicit deterministic ordering. sort_by may name a group column or a metric
    output label; default is group_by lexicographic (or the first metric when there
    is no group_by), so identical inputs always yield identical row order (INV-1)."""
    if sort_by:
        if sort_by in metric_labels:
            return f"{_quote(sort_by)} DESC"
        # A group column must be whitelisted; sort ascending for stable ordering.
        return f"{sql_identifier(sort_by, allowed_columns)} ASC"
    if group_by:
        return ", ".join(f"{sql_identifier(col, allowed_columns)} ASC" for col in group_by)
    return f"{_quote(metric_labels[0])} DESC"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _jsonable_cell(value):
    if value is None:
        return None
    try:
        import math

        if isinstance(value, float):
            return value if math.isfinite(value) else None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _Runtime(PackRuntime):
    def _extend(self, ctx) -> None:
        self.aligner = ColumnAligner(self.backend)
        self.join_engine = JoinEngine(self.backend, self.aligner, self.registry, self.repo)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _seed(ctx) -> int:
    return int(ctx.seed or 0)


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


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
