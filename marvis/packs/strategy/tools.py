from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.direction import check_score_direction, normalize_score_direction
from marvis.data.errors import LabelSemanticsNotDeclaredError
from marvis.data.labels import require_labels_confirmed, resolve_labeled_frame
from marvis.db import StrategyRepository
from marvis.files import sha256_file
from marvis.packs.strategy.backtest_compat import (
    BacktestRecord,
    approval_backtest_projection,
    backtest_record_payload,
)
from marvis.packs.strategy.bands import design_cutoff_bands
from marvis.packs.strategy.compare import compare_strategies
from marvis.packs.strategy.contracts import Strategy
from marvis.packs.strategy.deliverables import decision_table_csv
from marvis.packs.strategy.dsl import (
    canonical_strategy_json,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.doc import render_strategy_doc_markdown
from marvis.packs.strategy.monitor_tools import (  # noqa: F401
    tool_render_monitoring_report,
    tool_run_strategy_monitoring,
)
from marvis.packs.strategy.monitoring_plan import (
    DEFAULT_CADENCE_DAYS,
    PLAN_VERSION,
    build_monitoring_plan,
    save_monitoring_plan,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pricing import (
    LimitPricingResult,
    PricingParams,
    limit_pricing_matrix,
)
from marvis.packs.strategy.profit import ProfitParams, profit_calc
from marvis.packs.strategy.roll_rate import roll_rate_matrix
from marvis.packs.strategy.rules import (
    DEFAULT_MINE_SEED,
    evaluate_rule_set,
    mine_rules,
)
from marvis.packs.strategy.legacy_adapter import legacy_strategy_to_spec
from marvis.packs.strategy.strategy import (
    build_strategy,
    build_strategy_from_spec,
    infer_strategy_rule_direction,
)
from marvis.packs.strategy.tradeoff import (
    recommend_operating_point,
    tradeoff_feasible_flags,
    tradeoff_view,
)
from marvis.packs.strategy.typed_backtest import (
    ApprovalProfitInputs,
    StrategyBacktestResult,
    run_typed_backtest,
)
from marvis.packs.strategy.vintage import vintage_curve, vintage_summary
from marvis.plugins.sdk import PackRuntime
from marvis.strategy_adoption import AdoptionReasonError, normalize_adoption_reason
from marvis.validation.vintage import compute_vintage_curve


def tool_vintage_curve(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    cohort_col = str(inputs["cohort_col"])
    mob_col = str(inputs["mob_col"])
    bad_col = str(inputs["bad_col"])
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        task_id=str(ctx.task_id),
        columns=[cohort_col, mob_col, bad_col],
    )
    # NaN-label gate runs FIRST (an unusable label is a harder problem than an
    # undeclared cumulation basis); label_semantics is checked on the resolved frame.
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        bad_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    # A1: the strategy path must NOT guess the cumulation basis. The kernel always
    # accumulates the target across MOBs; on a snapshot/ever-bad flag that double-counts
    # silently. When the caller has not declared label_semantics, stop at a gate and hand
    # the two concrete semantics to the user (mirrors the NaN-label gate).
    label_semantics = _optional_str(inputs.get("label_semantics"))
    if label_semantics is None:
        raise LabelSemanticsNotDeclaredError(
            target_col=bad_col,
            n_cohorts=_vintage_cohort_count(frame, cohort_col),
            monotone_heuristic=_vintage_looks_like_snapshot(frame, cohort_col, mob_col, bad_col),
        )
    curve = vintage_curve(
        frame,
        cohort_col=cohort_col,
        mob_col=mob_col,
        bad_col=bad_col,
        mob_max=int(inputs.get("mob_max", 12)),
        label_semantics=label_semantics,
    )
    return {
        "cohorts": list(curve.cohorts),
        "mob_axis": list(curve.mob_axis),
        "curves": _jsonable(curve.curves),
        "counts": _jsonable(curve.counts),
        "summary": vintage_summary(curve, ref_mob=int(inputs.get("ref_mob", 6))),
        "nan_labels_dropped": nan_labels_dropped,
        "warnings": list(curve.warnings),
    }


def _vintage_cohort_count(frame, cohort_col: str) -> int:
    try:
        return int(frame[cohort_col].nunique(dropna=True))
    except Exception:
        return 0


def _vintage_looks_like_snapshot(frame, cohort_col: str, mob_col: str, bad_col: str) -> bool:
    """Reuse the kernel's own conservative snapshot heuristic (single source of truth):
    the incremental path attaches a snapshot red flag exactly when the data looks
    cumulative. If any point carries it, the data looks snapshot-shaped."""
    try:
        points = compute_vintage_curve(
            frame,
            cohort_col=cohort_col,
            mob_col=mob_col,
            target_col=bad_col,
            label_semantics="incremental",
        )
    except Exception:
        return False
    return any(
        "snapshot" in warning.lower() or "快照" in warning
        for point in points
        for warning in point.data_quality_warnings
    )


def tool_roll_rate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    balance_col = _optional_str(inputs.get("balance_col"))
    columns = _unique([str(inputs["id_col"]), str(inputs["time_col"]), str(inputs["status_col"]), balance_col])
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        task_id=str(ctx.task_id),
        columns=columns,
    )
    matrix = roll_rate_matrix(
        frame,
        id_col=str(inputs["id_col"]),
        time_col=str(inputs["time_col"]),
        status_col=str(inputs["status_col"]),
        states=[str(item) for item in inputs["states"]],
        balance_col=balance_col,
    )
    return {
        "states": list(matrix.states),
        "matrix": [list(row) for row in matrix.matrix],
        "base_counts": dict(matrix.base_counts),
        "data_quality_warnings": [dict(warning) for warning in matrix.data_quality_warnings],
    }


def tool_profit_calc(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    segment_col = _optional_str(inputs.get("segment_col"))
    columns = _unique([segment_col, str(inputs["ead_col"]), str(inputs["pd_col"])])
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        task_id=str(ctx.task_id),
        columns=columns,
    )
    results = profit_calc(
        frame,
        segment_col=segment_col,
        ead_col=str(inputs["ead_col"]),
        pd_col=str(inputs["pd_col"]),
        params=_profit_params(inputs["params"]),
    )
    return {"results": [_jsonable(result) for result in results]}


def tool_build_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    has_spec = inputs.get("strategy_spec") is not None
    has_rules = inputs.get("rules") is not None
    if has_spec == has_rules:
        raise StrategyError(
            "build_strategy requires exactly one of rules or strategy_spec"
        )
    if has_spec:
        strategy = build_strategy_from_spec(
            dict(inputs["strategy_spec"]),
            score_col=_optional_str(inputs.get("score_col")),
            description=str(inputs.get("description") or ""),
        )
    else:
        strategy = build_strategy(
            str(inputs["strategy_type"]),
            list(inputs["rules"]),
            score_col=_optional_str(inputs.get("score_col")),
            default_decision=inputs.get("default_decision"),
            description=str(inputs.get("description") or ""),
        )
    strategy = replace(
        strategy,
        id=_strategy_instance_id(str(ctx.task_id), strategy),
    )
    persisted = runtime.strategies.get_strategy(strategy.id)
    if persisted is None:
        runtime.strategies.create_strategy_with_audit(
            ctx.task_id,
            strategy,
            audit={
                "kind": "strategy.create",
                "target_ref": strategy.id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": str(ctx.task_id),
                    "strategy_type": strategy.strategy_type,
                    "rule_count": len(strategy.rules),
                },
            },
        )
    else:
        metadata = runtime.strategies.get_strategy_meta(strategy.id)
        if (
            metadata is None
            or str(metadata["task_id"]) != str(ctx.task_id)
            or not _same_strategy_payload(persisted, strategy)
        ):
            raise StrategyError(
                "strategy instance identity collision; refusing to reuse another "
                "task or a different persisted payload"
            )
        # Idempotent retries must report the row that actually exists, not a
        # transient object with display metadata the database never stored.
        strategy = persisted
    return {
        "strategy_id": strategy.id,
        "strategy_type": strategy.strategy_type,
        "score_col": strategy.score_col,
        "default_decision": strategy.default_decision,
        "description": strategy.description,
        "rules": [_jsonable(rule) for rule in strategy.rules],
        "dsl_schema_version": strategy.spec.schema_version,
        "strategy_spec": strategy.spec.to_dict(),
        "inferred_score_direction": (
            infer_strategy_rule_direction(list(strategy.rules), strategy.score_col)
            if not has_spec
            else None
        ),
    }


def _strategy_instance_id(task_id: str, strategy: Strategy) -> str:
    if strategy.spec is None:
        raise StrategyError("canonical strategy spec is required for persistence")
    semantic_digest = strategy_spec_hash(strategy.spec)
    payload = {
        "task_id": str(task_id),
        "dsl": json.loads(
            canonical_strategy_json(strategy.spec, include_display_metadata=True)
        ),
        "score_col": strategy.score_col,
        "description": strategy.description,
    }
    instance_digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"strategy-{semantic_digest[:12]}-{instance_digest[:12]}"


def _same_strategy_payload(left: Strategy, right: Strategy) -> bool:
    if left.spec is None or right.spec is None:
        return False
    return (
        left.strategy_type == right.strategy_type
        and left.score_col == right.score_col
        and left.default_decision == right.default_decision
        and left.description == right.description
        and canonical_strategy_json(left.spec, include_display_metadata=True)
        == canonical_strategy_json(right.spec, include_display_metadata=True)
    )


_APPLY_SCHEMA_VERSION = "strategy.apply.v1"
_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_APPLY_OUTPUT_SUFFIXES = {
    "action": "action",
    "value": "value",
    "value_type": "value_type",
    "rule_id": "rule_id",
    "reason_code": "reason_code",
}


def tool_apply_strategy(inputs: dict, ctx) -> dict:
    """Apply one persisted canonical Strategy DSL to a task-owned dataset.

    Execution delegates all condition and first-match semantics to the canonical
    vectorized evaluator.  This layer only projects its typed actions into new
    columns, atomically registers the derived parquet, and records evidence.
    """

    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    strategy = _strategy(runtime, str(inputs["strategy_id"]), task_id=task_id)
    spec = parse_strategy_spec(
        strategy.spec or legacy_strategy_to_spec(strategy)
    )
    dataset = _owned_dataset(runtime, dataset_id, task_id=task_id)
    source_path = runtime.registry.resolve_path(dataset.id)
    source_hash = sha256_file(source_path)
    frame = runtime.backend.read_frame(source_path)
    if sha256_file(source_path) != source_hash:
        raise StrategyError(
            "source dataset changed while the strategy was being applied"
        )
    output_columns = _strategy_apply_output_columns(inputs, frame)

    evaluation = evaluate_strategy_frame(frame, spec)
    action_values, action_value_types = _strategy_apply_values(
        evaluation.decisions,
        strategy_type=spec.strategy_type,
    )
    derived = frame.copy()
    derived[output_columns["action"]] = evaluation.action_type
    derived[output_columns["value"]] = action_values
    derived[output_columns["value_type"]] = action_value_types
    derived[output_columns["rule_id"]] = evaluation.matched_rule_id
    derived[output_columns["reason_code"]] = evaluation.reason_code

    action_counts = _string_counts(evaluation.action_type)
    rule_counts = _rule_counts(evaluation.matched_rule_id)
    default_count = int(evaluation.matched_rule_id.isna().sum())
    strategy_hash = strategy_spec_hash(spec)
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(
        runtime.datasets_root / task_id / "strategy_apply",
        f"applied_{strategy_hash[:12]}_{uuid.uuid4().hex}.parquet",
    )
    try:
        derived.to_parquet(staged.path, index=False)
        result_hash = sha256_file(staged.path)
        evidence = {
            "source_dataset_content_hash": source_hash,
            "strategy_effect_hash": strategy_hash,
            "result_dataset_content_hash": result_hash,
        }

        def audit_factory(registered_dataset):
            return {
                "kind": "strategy.apply",
                "target_ref": registered_dataset.id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": task_id,
                    "source_dataset_id": dataset.id,
                    "strategy_id": strategy.id,
                    "strategy_type": spec.strategy_type,
                    "population_count": int(len(frame)),
                    "action_counts": action_counts,
                    "rule_counts": rule_counts,
                    "default_count": default_count,
                    "output_columns": output_columns,
                    "evidence": evidence,
                },
            }

        registered = uow.finalize_with_connection(
            runtime.repo.transaction,
            lambda conn: runtime.registry.register_existing_with_audit_on_connection(
                conn,
                staged.final_path,
                audit_factory=audit_factory,
                task_id=task_id,
                role="strategy.applied",
                anchor_target=dataset.id,
                seed=int(ctx.seed or 0),
            ),
        )
    except Exception:
        uow.rollback()
        raise

    return {
        "schema_version": _APPLY_SCHEMA_VERSION,
        "strategy_id": strategy.id,
        "strategy_type": spec.strategy_type,
        "source_dataset_id": dataset.id,
        "result_dataset_id": registered.id,
        "population_count": int(len(frame)),
        "action_counts": action_counts,
        "rule_counts": rule_counts,
        "default_count": default_count,
        "output_columns": output_columns,
        "evidence": evidence,
    }


def _strategy_apply_output_columns(inputs: dict, frame: pd.DataFrame) -> dict[str, str]:
    raw_prefix = inputs.get("output_prefix")
    raw_columns = inputs.get("output_columns")
    if raw_prefix is not None and raw_columns is not None:
        raise StrategyError(
            "apply_strategy accepts output_prefix or output_columns, not both"
        )
    if raw_columns is not None and not isinstance(raw_columns, dict):
        raise StrategyError("output_columns must be an object")

    if raw_columns is None:
        if raw_prefix is not None and not isinstance(raw_prefix, str):
            raise StrategyError("output_prefix must be a string")
        prefix = "strategy_" if raw_prefix is None else raw_prefix
        _require_safe_output_name(prefix, name="output_prefix", is_prefix=True)
        columns = {
            key: f"{prefix}{suffix}"
            for key, suffix in _APPLY_OUTPUT_SUFFIXES.items()
        }
    else:
        unsupported = sorted(set(raw_columns) - set(_APPLY_OUTPUT_SUFFIXES))
        if unsupported:
            raise StrategyError(
                "output_columns has unsupported fields: " + ", ".join(unsupported)
            )
        columns = {}
        for key, suffix in _APPLY_OUTPUT_SUFFIXES.items():
            value = raw_columns.get(key)
            if value is None:
                columns[key] = f"strategy_{suffix}"
            elif not isinstance(value, str):
                raise StrategyError(f"output_columns.{key} must be a string")
            else:
                columns[key] = value

    for key, column in columns.items():
        _require_safe_output_name(column, name=f"output_columns.{key}")
    normalized_outputs = [column.casefold() for column in columns.values()]
    if len(set(normalized_outputs)) != len(normalized_outputs):
        raise StrategyError(
            "strategy output column names must be case-insensitively unique"
        )
    source_columns = {
        str(column).casefold()
        for column in frame.columns
    }
    collisions = sorted(
        column
        for column in columns.values()
        if column.casefold() in source_columns
    )
    if collisions:
        raise StrategyError(
            "strategy output columns already exist (case-insensitive): "
            + ", ".join(collisions)
        )
    return columns


def _require_safe_output_name(
    value: str,
    *,
    name: str,
    is_prefix: bool = False,
) -> None:
    limit = 48 if is_prefix else 64
    if not isinstance(value, str) or not value or len(value) > limit:
        raise StrategyError(f"{name} must be a non-empty safe identifier")
    if _SAFE_OUTPUT_NAME.fullmatch(value) is None:
        raise StrategyError(
            f"{name} must contain only ASCII letters, digits, and underscores "
            "and cannot start with a digit"
        )


def _strategy_apply_values(
    decisions: pd.Series,
    *,
    strategy_type: str,
) -> tuple[pd.Series, pd.Series]:
    decision_values = decisions.tolist()
    value_types = [_strategy_value_type(value) for value in decision_values]
    numeric_storage = strategy_type in {"limit", "pricing"} and all(
        value_type in {"integer", "number"} for value_type in value_types
    )
    values: list[object] = []
    for value, value_type in zip(decision_values, value_types, strict=True):
        values.append(
            _strategy_storage_value(
                value,
                value_type=value_type,
                numeric_storage=numeric_storage,
            )
        )
    return (
        pd.Series(values, index=decisions.index, dtype="object"),
        pd.Series(value_types, index=decisions.index, dtype="object"),
    )


def _strategy_storage_value(value, *, value_type: str, numeric_storage: bool):
    # Segment ids and legacy approval/reject output aliases may legally mix JSON
    # scalar types; parquet has no union column. Preserve their exact type in the
    # adjacent value_type column and use deterministic text storage. Canonical
    # limit/pricing decision values remain numeric for immediate downstream use.
    if numeric_storage:
        return value
    if value_type == "string":
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strategy_value_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    raise StrategyError("strategy decision value must be JSON serializable")


def _string_counts(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts(dropna=False).to_dict()
    return {
        str(key): int(counts[key])
        for key in sorted(counts, key=lambda item: str(item))
    }


def _rule_counts(values: pd.Series) -> dict[str, int]:
    return _string_counts(values.loc[values.notna()].map(str))


def tool_backtest_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy = _strategy(runtime, str(inputs["strategy_id"]), task_id=str(ctx.task_id))
    baseline_id = _optional_str(inputs.get("baseline_strategy_id"))
    baseline = (
        _strategy(runtime, baseline_id, task_id=str(ctx.task_id))
        if baseline_id
        else None
    )
    dataset_id = str(inputs["dataset_id"])
    dataset = _owned_dataset(runtime, dataset_id, task_id=str(ctx.task_id))
    source_path = runtime.registry.resolve_path(dataset.id)
    source_dataset_content_hash = sha256_file(source_path)
    frame = runtime.backend.read_frame(source_path)
    if sha256_file(source_path) != source_dataset_content_hash:
        raise StrategyError(
            "source dataset changed while the strategy backtest was running"
        )
    target_col = str(inputs["target_col"])
    # Keep the full population in the typed envelope while still requiring an
    # explicit confirmation before label metrics exclude missing supervision.
    nan_labels_dropped = require_labels_confirmed(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    economics_inputs = _typed_economics_inputs(
        frame,
        strategy_type=strategy.strategy_type,
        payload=inputs.get("economics_inputs"),
    )
    precomputed_economics, approval_profit_inputs = _approval_profit_inputs(
        strategy_type=strategy.strategy_type,
        profit_params=inputs.get("profit_params"),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    result = run_typed_backtest(
        frame,
        strategy.spec or legacy_strategy_to_spec(strategy),
        target_col=target_col,
        strategy_id=strategy.id,
        baseline=(
            None
            if baseline is None
            else baseline.spec or legacy_strategy_to_spec(baseline)
        ),
        economics=precomputed_economics,
        economics_inputs=economics_inputs,
        approval_profit_inputs=approval_profit_inputs,
    )
    backtest_id = _backtest_id(
        dataset_id,
        result,
        source_dataset_content_hash=source_dataset_content_hash,
    )
    existing = runtime.strategies.get_backtest(backtest_id)
    if existing is None:
        runtime.strategies.save_backtest_with_audit(
            backtest_id,
            strategy.id,
            dataset_id,
            result,
            audit={
                "kind": "strategy.backtest",
                "target_ref": backtest_id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": str(ctx.task_id),
                    "strategy_id": strategy.id,
                    "dataset_id": dataset_id,
                    "source_dataset_content_hash": source_dataset_content_hash,
                    "schema_version": result.schema_version,
                    "strategy_type": result.strategy_type,
                    "population_count": result.population_count,
                    "labeled_count": result.labeled_count,
                    **_backtest_audit_summary(result),
                },
            },
        )
    elif backtest_record_payload(existing) != result.to_dict():
        raise StrategyError(
            "backtest identity collision; refusing to reuse different evidence"
        )
    payload = result.to_dict()
    payload["backtest_id"] = backtest_id
    payload["source_dataset_content_hash"] = source_dataset_content_hash
    payload["nan_labels_dropped"] = nan_labels_dropped
    if result.strategy_type in {"approval", "reject"}:
        payload.update(approval_backtest_projection(result))
    profit_note = result.economics.get("profit_note")
    if profit_note:
        # FIN-3 #4: a profit backtest was requested but the EL chain inputs
        # (pd_col/ead_col) were missing, so expected_profit is None rather than a
        # fabricated 0.0. Surface the reason as a red flag instead of failing silently.
        payload["red_flags"] = [
            *payload.get("red_flags", []),
            {
                "code": "expected_profit_unavailable",
                "level": "amber",
                "message": profit_note,
            },
        ]
    return payload


def tool_tradeoff_view(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        task_id=str(ctx.task_id),
    )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, str(inputs["target_col"]), drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    score_col = str(inputs["score_col"])
    target_col = str(inputs["target_col"])
    score_direction = normalize_score_direction(_optional_str(inputs.get("score_direction")))
    effective_direction = score_direction or "higher_is_better"
    points = tradeoff_view(
        frame,
        score_col=score_col,
        target_col=target_col,
        cutoffs=[float(item) for item in inputs["cutoffs"]] if inputs.get("cutoffs") is not None else None,
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
        score_direction=score_direction,
        confirm_direction_conflict=bool(inputs.get("confirm_direction_conflict")),
    )
    max_bad_rate = _optional_float(inputs.get("max_bad_rate"))
    min_approval_rate = _optional_float(inputs.get("min_approval_rate"))
    feasible_flags = tradeoff_feasible_flags(
        points, max_bad_rate=max_bad_rate, min_approval_rate=min_approval_rate
    )
    red_flags: list[dict] = []
    recommended = None
    if points and any(feasible_flags):
        recommended = recommend_operating_point(
            [point for point, ok in zip(points, feasible_flags, strict=True) if ok],
            objective=str(inputs.get("objective") or "max_profit"),
            max_bad_rate=max_bad_rate,
        )
    elif points and (max_bad_rate is not None or min_approval_rate is not None):
        red_flags.append(
            {
                "code": "infeasible_constraints",
                "level": "red",
                "message": "在给定 max_bad_rate/min_approval_rate 约束下没有可行 cutoff。",
            }
        )
    direction_check = check_score_direction(
        pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float),
        declared_direction=effective_direction,
    )
    point_rows = []
    for point, feasible in zip(points, feasible_flags, strict=True):
        row = _jsonable(point)
        row["feasible"] = bool(feasible)
        point_rows.append(row)
    result = {
        "points": point_rows,
        "recommended": _jsonable(recommended),
        "nan_labels_dropped": nan_labels_dropped,
        "score_direction": effective_direction,
        "red_flags": red_flags,
    }
    if direction_check.status != "skipped":
        result["direction_diagnostics"] = _jsonable(direction_check)
    return result


def tool_design_cutoff_bands(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        task_id=str(ctx.task_id),
    )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, str(inputs["target_col"]), drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    score_col = str(inputs["score_col"])
    target_col = str(inputs["target_col"])
    score_direction = normalize_score_direction(_optional_str(inputs.get("score_direction")))
    effective_direction = score_direction or "higher_is_better"
    red_flags: list[dict] = []
    # Direction self-check (S1a): a conflict is a red flag and blocks unless the
    # caller confirms, mirroring tradeoff_view's confirm_direction_conflict gate.
    direction_check = check_score_direction(
        pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float),
        declared_direction=effective_direction,
    )
    if direction_check.status == "conflict" and not bool(inputs.get("confirm_direction_conflict")):
        from marvis.data.errors import ScoreDirectionConflictError

        raise ScoreDirectionConflictError(
            tool="design_cutoff_bands",
            score_col=score_col,
            target_col=target_col,
            declared_direction=effective_direction,
            implied_direction=direction_check.implied_direction,
            corr=direction_check.corr,
            n_labeled=direction_check.n,
        )
    if direction_check.status == "conflict":
        red_flags.append(
            {
                "code": "direction_conflict",
                "level": "red",
                "message": (
                    f"分数方向自检冲突：声明 {effective_direction}，数据隐含 "
                    f"{direction_check.implied_direction}（corr={direction_check.corr:.3f}）。"
                ),
            }
        )
    result = design_cutoff_bands(
        frame,
        score_col=score_col,
        target_col=target_col,
        score_direction=effective_direction,
        n_bands=int(inputs.get("n_bands", 5)),
        band_edges=[float(edge) for edge in inputs["band_edges"]]
        if inputs.get("band_edges") is not None
        else None,
        objective=str(inputs.get("objective") or "max_profit"),
        max_bad_rate=_optional_float(inputs.get("max_bad_rate")),
        min_approval_rate=_optional_float(inputs.get("min_approval_rate")),
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    red_flags.extend(_jsonable(flag) for flag in result.red_flags)
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    return {
        "bands": [_jsonable(band) for band in result.bands],
        "band_edges": [float(edge) for edge in result.band_edges],
        "recommended_rules": [dict(rule) for rule in result.recommended_rules],
        "red_flags": red_flags,
        "score_direction": effective_direction,
        "nan_labels_dropped": nan_labels_dropped,
    }


def tool_compare_strategies(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    baseline_id = _optional_str(inputs.get("baseline_strategy_id"))
    if baseline_id is None:
        # No baseline supplied (e.g. the template's optional compare step ran
        # without a baseline_strategy_id slot): degrade to a no-op result
        # instead of failing the plan -- the step is informational, not gating.
        return {
            "matrix_2x2": {
                cell: {"count": 0, "bad_rate": None}
                for cell in ("both_approve", "only_new", "only_baseline", "both_decline")
            },
            "deltas": {"approval_rate": 0.0, "approved_bad_rate": 0.0, "expected_profit": 0.0},
            "summary_text": "未提供基线策略，跳过对比。",
            "red_flags": [],
            "nan_labels_dropped": 0,
            "label_coverage": 1.0,
        }
    strategy = _strategy(
        runtime, str(inputs["strategy_id"]), task_id=str(ctx.task_id)
    )
    baseline = _strategy(runtime, baseline_id, task_id=str(ctx.task_id))
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        task_id=str(ctx.task_id),
    )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, str(inputs["target_col"]), drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    result = compare_strategies(
        frame,
        strategy,
        baseline,
        target_col=str(inputs["target_col"]),
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    payload = _jsonable(result)
    payload["nan_labels_dropped"] = nan_labels_dropped
    payload["label_coverage"] = _label_coverage(len(frame) + nan_labels_dropped, nan_labels_dropped)
    return payload


def tool_limit_pricing_matrix(inputs: dict, ctx) -> dict:
    """S6 (A3): a band x limit x rate expected-profit grid with an EL simulation.

    Always computes and returns the full matrix + per-band recommended feasible cell.
    The strategy_artifacts(kind='limit_pricing_csv') deliverable is written ONLY when
    ``confirm`` is true -- the driver flips it after the matrix confirmation gate
    (矩阵确认门后才落 artifact), mirroring adopt_strategy's forced-gate precedent. The
    CSV is attached to ``strategy_id`` so it rides the same per-strategy artifact list.
    """
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    score_col = str(inputs["score_col"])
    target_col = _optional_str(inputs.get("target_col"))
    pd_col = _optional_str(inputs.get("pd_col"))
    columns = _unique([score_col, target_col, pd_col])
    frame = _dataset_frame(
        runtime,
        dataset_id,
        task_id=str(ctx.task_id),
        columns=columns,
    )
    if target_col:
        frame, nan_labels_dropped = resolve_labeled_frame(
            frame, target_col, drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        )
    else:
        nan_labels_dropped = 0

    params = PricingParams(
        lgd=float(inputs.get("lgd", 0.6)),
        funding_rate=float(inputs["funding_rate"]),
        term_months=int(inputs["term_months"]),
        cost_per_loan=float(inputs["cost_per_loan"]),
        el_ead_max=float(inputs.get("el_ead_max", 0.20)),
    )
    result = limit_pricing_matrix(
        frame,
        score_col=score_col,
        limit_grid=[float(item) for item in inputs["limit_grid"]],
        rate_grid=[float(item) for item in inputs["rate_grid"]],
        params=params,
        target_col=target_col,
        pd_col=pd_col,
        band_edges=[float(edge) for edge in inputs["band_edges"]]
        if inputs.get("band_edges") is not None
        else None,
        n_bands=int(inputs.get("n_bands", 5)),
    )
    red_flags = [dict(flag) for flag in result.red_flags]
    if nan_labels_dropped:
        red_flags.append({
            "code": "nan_labels_dropped",
            "level": "amber",
            "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
        })

    assumptions = {
        "dataset_id": dataset_id,
        "score_col": score_col,
        "target_col": target_col,
        "pd_col": pd_col,
        "lgd": params.lgd,
        "funding_rate": params.funding_rate,
        "term_months": params.term_months,
        "cost_per_loan": params.cost_per_loan,
        "el_ead_max": params.el_ead_max,
        "limit_grid": [float(item) for item in inputs["limit_grid"]],
        "rate_grid": [float(item) for item in inputs["rate_grid"]],
        "band_edges": [float(edge) for edge in result.band_edges],
        "n_bands": int(inputs.get("n_bands", 5)),
    }

    payload = {
        "matrix": [_jsonable(cell) for cell in result.matrix],
        "recommended": [dict(item) for item in result.recommended],
        "band_edges": [float(edge) for edge in result.band_edges],
        "assumptions": assumptions,
        "red_flags": red_flags,
        "nan_labels_dropped": nan_labels_dropped,
    }

    strategy_id = _optional_str(inputs.get("strategy_id"))
    artifacts: list[dict] = []
    # 矩阵确认门后才落 artifact: only after the user confirms the matrix does the CSV
    # deliverable get written and registered (adopt_strategy forced-gate precedent).
    if bool(inputs.get("confirm")) and strategy_id:
        strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        csv_path = strategy_dir / f"limit_pricing_{strategy_id}.csv"
        csv_path.write_text(_limit_pricing_csv(result), encoding="utf-8")
        runtime.strategies.save_strategy_artifact(
            strategy_id, kind="limit_pricing_csv", path=str(csv_path)
        )
        _write_strategy_artifact_audit(runtime, ctx, strategy_id, "limit_pricing_csv", csv_path)
        artifacts.append({"kind": "limit_pricing_csv", "path": str(csv_path)})
    payload["artifacts"] = artifacts
    return payload


def _limit_pricing_csv(result: LimitPricingResult) -> str:
    header = "band,limit,rate,count,pd,el,ead,expected_profit,roa,feasible,recommended"
    recommended = {
        (item["band"], float(item["limit"]), float(item["rate"])) for item in result.recommended
    }
    lines = [header]
    for cell in result.matrix:
        is_reco = (cell.band, float(cell.limit), float(cell.rate)) in recommended
        lines.append(
            ",".join([
                cell.band,
                _csv_num(cell.limit),
                _csv_num(cell.rate),
                str(cell.count),
                _csv_num(cell.pd),
                _csv_num(cell.el),
                _csv_num(cell.ead),
                _csv_num(cell.expected_profit),
                _csv_num(cell.roa),
                "1" if cell.feasible else "0",
                "1" if is_reco else "0",
            ])
        )
    return "\n".join(lines) + "\n"


def _csv_num(value) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}"


_ADOPTION_EVIDENCE_SCHEMA_VERSION = "strategy.adoption-evidence.v1"
_LEGACY_BACKTEST_SCHEMA_VERSION = "strategy.backtest.v1"


def tool_adopt_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id, task_id=task_id)
    backtest_id = str(inputs["backtest_id"])
    backtest = runtime.strategies.get_backtest(backtest_id)
    if backtest is None or backtest.strategy_id != strategy_id:
        raise StrategyError(
            f"backtest {backtest_id} does not belong to strategy {strategy_id}"
        )
    if isinstance(backtest, StrategyBacktestResult):
        if backtest.strategy_type != strategy.strategy_type:
            raise StrategyError(
                "backtest strategy_type does not match the persisted strategy"
            )
    elif strategy.strategy_type not in {"approval", "reject"}:
        raise StrategyError(
            f"{strategy.strategy_type} adoption requires a typed StrategyBacktestResult"
        )

    adoption_evidence, approval_metrics = _strategy_adoption_evidence(
        runtime,
        strategy=strategy,
        backtest=backtest,
        backtest_id=backtest_id,
        task_id=task_id,
    )
    experiment_id = _strategy_monitoring_experiment_id(
        runtime,
        inputs.get("experiment_id"),
        task_id=task_id,
    )
    try:
        adoption_reason = normalize_adoption_reason(inputs.get("adoption_reason"))
    except AdoptionReasonError as exc:
        raise StrategyError(str(exc)) from exc
    effect_execution_id = _optional_str(getattr(ctx, "effect_execution_id", None))
    runtime_generation = _optional_str(getattr(ctx, "runtime_generation", None))
    if (effect_execution_id is None) != (runtime_generation is None):
        raise StrategyError("治理执行元数据不完整，拒绝采纳策略")
    strategy_meta = runtime.strategies.get_strategy_meta(strategy_id)
    if strategy_meta is None:
        raise StrategyError(f"strategy not found: {strategy_id}")
    version = int(strategy_meta["version"])
    strategy_dir = Path(runtime.settings.tasks_dir) / task_id / "strategy"
    stem = f"{strategy_id}_v{version}"

    # band_stats is retained in the manifest only for stored-plan compatibility.
    # It is caller-supplied and not bound to this backtest, so adoption artifacts
    # must not present it as verified evidence. The decision table comes only
    # from the persisted canonical strategy and includes unmatched/default flow.
    rules = _adoption_decision_table_rules(strategy)
    csv_text = decision_table_csv(rules, [])
    monitoring_plan = _build_adoption_monitoring_plan(
        strategy_id=strategy_id,
        strategy_type=strategy.strategy_type,
        version=version,
        evidence=adoption_evidence,
        approval_metrics=approval_metrics,
        experiment_id=experiment_id,
    )

    # Adoption is one multi-resource commit: the two required deliverables are
    # staged first, then promoted and recorded together with lifecycle/effect
    # state on one caller-owned SQLite transaction. Any filesystem, artifact,
    # audit, or commit failure restores both files and every database mutation.
    uow = ArtifactUnitOfWork()
    staged_csv = uow.stage_file(strategy_dir, f"decision_table_{stem}.csv")
    staged_json = uow.stage_file(strategy_dir, f"monitoring_plan_{stem}.json")
    artifact_specs = (
        ("decision_table_csv", staged_csv),
        ("monitoring_plan_json", staged_json),
    )
    try:
        staged_csv.path.write_text(csv_text, encoding="utf-8")
        save_monitoring_plan(staged_json.path, monitoring_plan)

        def finalize_adoption(conn):
            adopt_result = runtime.strategies.adopt_strategy_with_audit_on_connection(
                conn,
                strategy_id,
                reason=adoption_reason,
                audit={
                    "kind": "strategy.adopt",
                    "target_ref": strategy_id,
                    "outcome": "succeeded",
                    "detail": {
                        "task_id": task_id,
                        "backtest_id": backtest_id,
                        "strategy_type": strategy.strategy_type,
                        "experiment_id": experiment_id,
                        "adoption_reason": adoption_reason,
                        "adoption_evidence": adoption_evidence,
                        **_approval_adoption_audit_summary(approval_metrics),
                    },
                },
                effect_execution_id=effect_execution_id,
                runtime_generation=runtime_generation,
            )
            if int(adopt_result["version"]) != version:
                raise StrategyError("strategy version changed during adoption")
            for kind, staged in artifact_specs:
                final_path = str(staged.final_path)
                runtime.strategies.save_strategy_artifact_with_audit_on_connection(
                    conn,
                    strategy_id,
                    kind=kind,
                    path=final_path,
                    audit={
                        "kind": "strategy.artifact",
                        "target_ref": strategy_id,
                        "outcome": "succeeded",
                        "detail": {
                            "task_id": task_id,
                            "kind": kind,
                            "path": final_path,
                        },
                    },
                )
            return adopt_result

        adopt_result = uow.finalize_with_connection(
            runtime.strategies.transaction,
            finalize_adoption,
        )
    except Exception:
        uow.rollback()
        raise

    artifacts = [
        {"kind": kind, "path": str(staged.final_path)}
        for kind, staged in artifact_specs
    ]

    return {
        "strategy_id": strategy_id,
        "strategy_type": strategy.strategy_type,
        "backtest_id": backtest_id,
        "version": version,
        "status": "adopted",
        "retired_strategy_ids": list(adopt_result["retired_strategy_ids"]),
        "adoption_evidence": adoption_evidence,
        "artifacts": artifacts,
    }


def _strategy_adoption_evidence(
    runtime,
    *,
    strategy: Strategy,
    backtest: BacktestRecord,
    backtest_id: str,
    task_id: str,
) -> tuple[dict, dict | None]:
    if isinstance(backtest, StrategyBacktestResult):
        _require_typed_adoption_quality(backtest)
        expected_effect_hash = strategy_spec_hash(
            strategy.spec or legacy_strategy_to_spec(strategy)
        )
        actual_effect_hash = str(
            backtest.normalized_input.get("strategy_effect_hash") or ""
        )
        if actual_effect_hash != expected_effect_hash:
            raise StrategyError(
                "backtest strategy effect hash does not match the persisted strategy"
            )
        binding = _backtest_binding(runtime, backtest_id)
        if binding["strategy_id"] != strategy.id:
            raise StrategyError(
                f"backtest {backtest_id} does not belong to strategy {strategy.id}"
            )
        if binding["dataset_task_id"] is None:
            raise StrategyError(
                "typed backtest source dataset is not registered; rerun the backtest"
            )
        if binding["dataset_task_id"] != task_id:
            raise StrategyError(
                "typed backtest source dataset must belong to the same task as the strategy"
            )
        if binding["dataset_content_hash"] is None:
            raise StrategyError(
                "typed backtest source dataset file is unavailable; rerun the backtest"
            )
        if binding["backtest_dataset_content_hash"] is None:
            raise StrategyError(
                "typed backtest is missing backtest-time source dataset hash evidence; "
                "rerun the backtest"
            )
        if (
            binding["dataset_content_hash"]
            != binding["backtest_dataset_content_hash"]
        ):
            raise StrategyError(
                "source dataset content hash no longer matches the backtest evidence"
            )
        evidence = {
            "schema_version": _ADOPTION_EVIDENCE_SCHEMA_VERSION,
            "backtest_schema_version": backtest.schema_version,
            "backtest_id": backtest_id,
            "strategy_id": strategy.id,
            "strategy_type": strategy.strategy_type,
            "source_dataset_id": binding["dataset_id"],
            "source_dataset_content_hash": binding[
                "backtest_dataset_content_hash"
            ],
            "strategy_effect_hash": expected_effect_hash,
            "baseline_effect_hash": backtest.normalized_input[
                "baseline_effect_hash"
            ],
            "target_col": str(backtest.normalized_input["target_col"]),
            "population_count": int(backtest.population_count),
            "labeled_count": int(backtest.labeled_count),
            "label_coverage": float(backtest.label_coverage),
            "metrics": _jsonable(dict(backtest.metrics)),
            "breakdown": [_jsonable(dict(row)) for row in backtest.breakdown],
            "transitions": [
                _jsonable(dict(row)) for row in backtest.transitions
            ],
            # Per-row pricing economics is deliberately excluded from adoption
            # evidence and audit. The typed envelope has already reconciled it to
            # these aggregates, which are sufficient for governance decisions.
            "economics": _aggregate_economics(backtest.economics),
            "economics_input_evidence": _jsonable(
                dict(backtest.normalized_input["economics_input_evidence"])
            ),
            "warnings": list(backtest.warnings),
        }
        approval_metrics = (
            approval_backtest_projection(
                backtest,
                preserve_undefined_rates=True,
            )
            if strategy.strategy_type in {"approval", "reject"}
            else None
        )
        return evidence, approval_metrics

    approval_metrics = approval_backtest_projection(
        backtest,
        preserve_undefined_rates=True,
    )
    if approval_metrics["approved_bad_rate"] is None:
        raise StrategyError(
            "cannot adopt strategy because approved bad rate is undefined; "
            "provide labeled approved observations and rerun the backtest"
        )
    binding = _backtest_binding(runtime, backtest_id)
    if (
        binding["dataset_task_id"] is not None
        and binding["dataset_task_id"] != task_id
    ):
        raise StrategyError(
            "legacy backtest source dataset must belong to the same task as the strategy"
        )
    effect_hash = strategy_spec_hash(
        strategy.spec or legacy_strategy_to_spec(strategy)
    )
    metrics = {
        key: _jsonable(value)
        for key, value in approval_metrics.items()
        if key
        not in {
            "strategy_id",
            "by_segment",
            "expected_profit",
            "profit_note",
        }
    }
    return {
        "schema_version": _ADOPTION_EVIDENCE_SCHEMA_VERSION,
        "backtest_schema_version": _LEGACY_BACKTEST_SCHEMA_VERSION,
        "backtest_id": backtest_id,
        "strategy_id": strategy.id,
        "strategy_type": strategy.strategy_type,
        "source_dataset_id": binding["dataset_id"],
        "source_dataset_content_hash": binding["dataset_content_hash"],
        "strategy_effect_hash": effect_hash,
        "baseline_effect_hash": None,
        "target_col": None,
        # Legacy rows never carried population/label provenance. Keep that gap
        # explicit rather than reconstructing counts from rounded rates.
        "population_count": None,
        "labeled_count": None,
        "label_coverage": None,
        "metrics": metrics,
        "breakdown": [
            _jsonable(dict(row)) for row in approval_metrics.get("by_segment", [])
        ],
        "transitions": [],
        "economics": {
            "expected_profit": _jsonable(approval_metrics.get("expected_profit")),
            "profit_note": _jsonable(approval_metrics.get("profit_note")),
        },
        "economics_input_evidence": {},
        "warnings": [
            "legacy backtest has no task-bound dataset or label-provenance contract"
        ],
    }, approval_metrics


def _adoption_decision_table_rules(strategy: Strategy) -> list[dict]:
    spec = parse_strategy_spec(
        strategy.spec or legacy_strategy_to_spec(strategy)
    )
    rows = [_jsonable(rule) for rule in strategy.rules]
    default_action = spec.default_action
    default_decision = {
        "approval": "approve",
        "reject": "reject",
        "review": "review",
        "limit": "limit",
        "pricing": "price",
        "segment": "segment",
    }[default_action.type]
    default_value = (
        default_action.value
        if default_action.type in {"limit", "pricing", "segment"}
        else default_action.output_value
    )
    rows.append(
        {
            "condition": "未命中任何规则（默认动作）",
            "decision": default_decision,
            "value": _jsonable(default_value),
            "rule_id": "__default__",
            "priority": None,
            "reason_code": default_action.reason_code,
        }
    )
    return rows


def _require_typed_adoption_quality(result: StrategyBacktestResult) -> None:
    if result.population_count <= 0:
        raise StrategyError("cannot adopt strategy from an empty backtest population")
    if result.labeled_count <= 0:
        raise StrategyError(
            "cannot adopt strategy without labeled backtest observations"
        )

    metrics = result.metrics
    if result.strategy_type == "approval":
        if metrics.get("approve_bad_rate") is None:
            raise StrategyError(
                "cannot adopt strategy because approved bad rate is undefined; "
                "provide labeled approved observations and rerun the backtest"
            )
        return
    if result.strategy_type == "reject":
        if metrics.get("bad_capture_rate") is None:
            raise StrategyError(
                "cannot adopt reject strategy because bad capture rate is undefined"
            )
        if metrics.get("good_reject_rate") is None:
            raise StrategyError(
                "cannot adopt reject strategy because good reject rate is undefined"
            )
        return
    if result.strategy_type == "limit":
        if metrics.get("mean_limit") is None or set(result.economics) != {
            "expected_ead",
            "expected_loss",
        }:
            raise StrategyError(
                "limit adoption requires complete limit economics evidence"
            )
        return
    if result.strategy_type == "pricing":
        required = {
            "total_ead",
            "ead_weighted_rate",
            "revenue",
            "expected_loss",
            "funding_cost",
            "operating_cost",
            "profit",
            "roa",
            "baseline_profit",
            "profit_delta_vs_baseline",
            "by_row",
        }
        if (
            metrics.get("mean_rate") is None
            or set(result.economics) != required
            or result.economics.get("total_ead") in {None, 0.0}
            or result.economics.get("profit") is None
            or result.economics.get("roa") is None
        ):
            raise StrategyError(
                "pricing adoption requires complete pricing economics evidence"
            )
        return
    if (
        metrics.get("segment_count", 0) <= 0
        or metrics.get("overall_bad_rate") is None
    ):
        raise StrategyError(
            "segmentation adoption requires non-empty labeled segment evidence"
        )


def _backtest_binding(runtime, backtest_id: str) -> dict[str, str | None]:
    from marvis.db_schema import connect

    with connect(runtime.settings.db_path) as conn:
        row = conn.execute(
            """
            SELECT b.strategy_id, b.dataset_id, d.task_id AS dataset_task_id
              FROM backtests b
              LEFT JOIN datasets d ON d.id = b.dataset_id
             WHERE b.id = ?
            """,
            (backtest_id,),
        ).fetchone()
        audit_row = conn.execute(
            """
            SELECT detail_json
              FROM audit
             WHERE kind = 'strategy.backtest'
               AND target_ref = ?
               AND outcome = 'succeeded'
             ORDER BY at DESC, id DESC
             LIMIT 1
            """,
            (backtest_id,),
        ).fetchone()
    if row is None:
        raise StrategyError(f"backtest not found: {backtest_id}")
    dataset_id = str(row["dataset_id"])
    dataset_content_hash = None
    if row["dataset_task_id"] is not None:
        try:
            dataset_content_hash = sha256_file(
                runtime.registry.resolve_path(dataset_id)
            )
        except (KeyError, OSError):
            # Typed adoption turns this into a hard failure. Legacy rows keep a
            # nullable provenance field because historical fixtures and migrated
            # databases may no longer have the original registered file.
            dataset_content_hash = None
    backtest_dataset_content_hash = None
    if audit_row is not None:
        try:
            audit_detail = json.loads(str(audit_row["detail_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            audit_detail = None
        if isinstance(audit_detail, dict):
            candidate = audit_detail.get("source_dataset_content_hash")
            if isinstance(candidate, str) and re.fullmatch(
                r"[0-9a-f]{64}", candidate
            ):
                backtest_dataset_content_hash = candidate
    return {
        "strategy_id": str(row["strategy_id"]),
        "dataset_id": dataset_id,
        "dataset_task_id": (
            None if row["dataset_task_id"] is None else str(row["dataset_task_id"])
        ),
        "dataset_content_hash": dataset_content_hash,
        "backtest_dataset_content_hash": backtest_dataset_content_hash,
    }


def _strategy_monitoring_experiment_id(
    runtime,
    value,
    *,
    task_id: str,
) -> str | None:
    experiment_id = _optional_str(value)
    if experiment_id is None:
        return None
    from marvis.db_schema import connect

    with connect(runtime.settings.db_path) as conn:
        row = conn.execute(
            "SELECT task_id FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError(
            "monitoring experiment must exist and belong to the same task as the strategy"
        )
    return experiment_id


def _aggregate_economics(economics) -> dict:
    return {
        str(key): _jsonable(value)
        for key, value in economics.items()
        if key != "by_row"
    }


def _approval_adoption_audit_summary(approval_metrics: dict | None) -> dict:
    if approval_metrics is None:
        return {}
    return {
        "approval_rate": approval_metrics["approval_rate"],
        "approved_bad_rate": approval_metrics["approved_bad_rate"],
        "expected_profit": approval_metrics["expected_profit"],
    }


def _build_adoption_monitoring_plan(
    *,
    strategy_id: str,
    strategy_type: str,
    version: int,
    evidence: dict,
    approval_metrics: dict | None,
    experiment_id: str | None,
) -> dict:
    baseline = {
        "strategy_type": strategy_type,
        "backtest_schema_version": evidence["backtest_schema_version"],
        "strategy_effect_hash": evidence["strategy_effect_hash"],
        "baseline_effect_hash": evidence["baseline_effect_hash"],
        "source_dataset_id": evidence["source_dataset_id"],
        "source_dataset_content_hash": evidence[
            "source_dataset_content_hash"
        ],
        "source_backtest_id": evidence["backtest_id"],
        "population_count": evidence["population_count"],
        "labeled_count": evidence["labeled_count"],
        "label_coverage": evidence["label_coverage"],
        "metrics": dict(evidence["metrics"]),
        "economics": dict(evidence["economics"]),
        "breakdown": [dict(row) for row in evidence["breakdown"]],
        "transitions": [dict(row) for row in evidence["transitions"]],
    }
    if (
        strategy_type == "approval"
        or evidence["backtest_schema_version"] == _LEGACY_BACKTEST_SCHEMA_VERSION
    ):
        assert approval_metrics is not None
        plan = build_monitoring_plan(
            strategy_id=strategy_id,
            version=version,
            approved_bad_rate=float(approval_metrics["approved_bad_rate"]),
            approval_rate=float(approval_metrics["approval_rate"]),
            experiment_id=experiment_id,
            source_backtest_id=evidence["backtest_id"],
        )
        plan["expectation_baseline"].update(baseline)
        return plan

    thresholds = _typed_monitoring_thresholds(
        strategy_type,
        evidence=evidence,
        approval_metrics=approval_metrics,
    )
    if approval_metrics is not None:
        baseline.update(
            {
                "approval_rate": approval_metrics["approval_rate"],
                "approved_bad_rate": approval_metrics["approved_bad_rate"],
            }
        )
    return {
        "plan_version": PLAN_VERSION,
        "strategy_id": strategy_id,
        "version": int(version),
        "cadence_days": DEFAULT_CADENCE_DAYS,
        "experiment_id": experiment_id,
        "last_run_at": None,
        "thresholds": thresholds,
        "expectation_baseline": baseline,
    }


def _typed_monitoring_thresholds(
    strategy_type: str,
    *,
    evidence: dict,
    approval_metrics: dict | None,
) -> dict:
    metrics = evidence["metrics"]
    economics = evidence["economics"]
    if strategy_type == "reject":
        assert approval_metrics is not None
        approval_rate = float(approval_metrics["approval_rate"])
        bad_capture_rate = float(metrics["bad_capture_rate"])
        good_reject_rate = float(metrics["good_reject_rate"])
        thresholds = {
            "approval_rate": _monitor_threshold(
                "审批率下滑",
                "approval_rate",
                "min",
                max(0.0, approval_rate - 0.05),
                max(0.0, approval_rate - 0.10),
            ),
            "bad_capture_rate": _monitor_threshold(
                "坏客户捕获率下滑",
                "bad_capture_rate",
                "min",
                max(0.0, bad_capture_rate - 0.05),
                max(0.0, bad_capture_rate - 0.10),
            ),
            "good_reject_rate": _monitor_threshold(
                "好客户误拒率上升",
                "good_reject_rate",
                "max",
                min(1.0, good_reject_rate + 0.02),
                min(1.0, good_reject_rate + 0.05),
            ),
        }
        approved_bad_rate = approval_metrics["approved_bad_rate"]
        if approved_bad_rate is not None:
            value = float(approved_bad_rate)
            thresholds["approved_bad_rate"] = _monitor_threshold(
                "通过客群坏率漂移",
                "approved_bad_rate",
                "max",
                min(1.0, value + 0.02),
                min(1.0, value + 0.05),
            )
        return thresholds

    if strategy_type == "limit":
        mean_limit = float(metrics["mean_limit"])
        expected_loss = float(economics["expected_loss"])
        return {
            "mean_limit": _monitor_threshold(
                "户均额度上升",
                "mean_limit",
                "max",
                mean_limit * 1.10,
                mean_limit * 1.20,
            ),
            "expected_loss": _monitor_threshold(
                "额度策略预期损失上升",
                "expected_loss",
                "max",
                expected_loss * 1.10,
                expected_loss * 1.20,
            ),
        }

    if strategy_type == "pricing":
        mean_rate = float(metrics["mean_rate"])
        expected_loss = float(economics["expected_loss"])
        profit = float(economics["profit"])
        roa = float(economics["roa"])
        return {
            "mean_rate": _monitor_threshold(
                "平均利率上升",
                "mean_rate",
                "max",
                min(1.0, mean_rate + 0.02),
                min(1.0, mean_rate + 0.05),
            ),
            "expected_loss": _monitor_threshold(
                "定价策略预期损失上升",
                "expected_loss",
                "max",
                expected_loss * 1.10,
                expected_loss * 1.20,
            ),
            "profit": _monitor_threshold(
                "利润下滑",
                "profit",
                "min",
                profit - abs(profit) * 0.10,
                profit - abs(profit) * 0.20,
            ),
            "roa": _monitor_threshold(
                "ROA 下滑",
                "roa",
                "min",
                roa - 0.01,
                roa - 0.02,
            ),
        }

    overall_bad_rate = float(metrics["overall_bad_rate"])
    return {
        "overall_bad_rate": _monitor_threshold(
            "分群总体坏率上升",
            "overall_bad_rate",
            "max",
            min(1.0, overall_bad_rate + 0.02),
            min(1.0, overall_bad_rate + 0.05),
        ),
        "segment_share_psi": _monitor_threshold(
            "分群占比漂移",
            "segment_share_psi",
            "max",
            0.10,
            0.25,
        ),
    }


def _monitor_threshold(
    label: str,
    metric: str,
    direction: str,
    warn: float,
    fail: float,
) -> dict:
    return {
        "label": label,
        "metric": metric,
        "direction": direction,
        "warn": float(warn),
        "fail": float(fail),
    }


def tool_render_challenger_report(inputs: dict, ctx) -> dict:
    """S6 Commit 3: assemble a challenger-vs-champion Markdown report from the compare
    output + both backtests + the adoption status, register it as
    strategy_artifacts(kind='challenger_report_md'), and audit it.

    Graceful degradation (compare_strategies precedent): with no champion/baseline the
    report is a no-op — it returns status='no_baseline' + a 「未提供基线」 markdown and
    writes NO artifact, so an optional template step that ran without a champion slot
    does not fail the plan. Every number in the report comes straight from the passed-in
    compare/backtest tool outputs (INV-1: presentation only, report follows tool output).
    """
    runtime = _runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    champion_id = _optional_str(inputs.get("champion_strategy_id"))
    compare = _as_dict(inputs.get("compare"))
    # A compare that itself degraded to the no-baseline no-op carries this text; treat it
    # as "no champion" too so the report degrades in lockstep with compare_strategies.
    compare_degraded = str(compare.get("summary_text") or "").startswith("未提供基线")

    if not champion_id or compare_degraded:
        markdown = "# 挑战者对比报告\n\n未提供基线（champion）策略，跳过对比报告。\n"
        return {
            "status": "no_baseline",
            "report_md": markdown,
            "artifacts": [],
        }

    challenger_backtest = _as_dict(inputs.get("challenger_backtest"))
    champion_backtest = _as_dict(inputs.get("champion_backtest"))
    adopted = bool(inputs.get("adopted"))
    markdown = _challenger_report_markdown(
        strategy_id=strategy_id,
        champion_id=champion_id,
        compare=compare,
        challenger_backtest=challenger_backtest,
        champion_backtest=champion_backtest,
        adopted=adopted,
    )

    strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    report_path = strategy_dir / f"challenger_report_{strategy_id}.md"
    report_path.write_text(markdown, encoding="utf-8")
    runtime.strategies.save_strategy_artifact(
        strategy_id, kind="challenger_report_md", path=str(report_path)
    )
    _write_strategy_artifact_audit(runtime, ctx, strategy_id, "challenger_report_md", report_path)
    return {
        "status": "rendered",
        "report_md": markdown,
        "report_path": str(report_path),
        "artifacts": [{"kind": "challenger_report_md", "path": str(report_path)}],
    }


def _challenger_report_markdown(
    *,
    strategy_id: str,
    champion_id: str,
    compare: dict,
    challenger_backtest: dict,
    champion_backtest: dict,
    adopted: bool,
) -> str:
    deltas = _as_dict(compare.get("deltas"))
    lines = [
        "# 挑战者对比报告",
        "",
        f"- 挑战者策略：`{strategy_id}`",
        f"- 基线（champion）策略：`{champion_id}`",
        f"- 采纳状态：{'已采纳挑战者' if adopted else '未采纳（仍以基线为准）'}",
        "",
        "## 关键指标并排",
        "",
        "| 指标 | 挑战者 | 基线 | 挑战者−基线 |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in (
        ("审批率", "approval_rate"),
        ("通过客群坏率", "approved_bad_rate"),
        ("预期利润", "expected_profit"),
    ):
        lines.append(
            f"| {label} | {_report_num(challenger_backtest.get(key))} | "
            f"{_report_num(champion_backtest.get(key))} | {_report_num(deltas.get(key))} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        str(compare.get("summary_text") or ""),
        "",
    ])
    red_flags = [flag for flag in (compare.get("red_flags") or []) if isinstance(flag, dict)]
    if red_flags:
        lines.append("## 红旗")
        lines.append("")
        for flag in red_flags:
            lines.append(f"- [{flag.get('level', '')}] {flag.get('code', '')}: {flag.get('message', '')}")
        lines.append("")
    return "\n".join(lines)


def _report_num(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _as_dict(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def tool_render_strategy_doc(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id, task_id=str(ctx.task_id))
    meta = runtime.strategies.get_strategy_meta(strategy_id)
    backtests = [_jsonable(result) for result in runtime.strategies.list_backtests(strategy_id)]
    artifacts = runtime.strategies.list_strategy_artifacts(strategy_id)
    band_stats = _band_stats_from_inputs(inputs.get("band_stats"))
    markdown, sections = render_strategy_doc_markdown(
        strategy=_jsonable(strategy),
        meta=meta or {},
        backtests=backtests,
        artifacts=artifacts,
        band_stats=band_stats,
    )
    strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    version = int((meta or {}).get("version", 1))
    doc_path = strategy_dir / f"strategy_doc_{strategy_id}_v{version}.md"
    doc_path.write_text(markdown, encoding="utf-8")
    runtime.strategies.save_strategy_artifact(
        strategy_id, kind="strategy_doc_md", path=str(doc_path)
    )
    _write_strategy_artifact_audit(runtime, ctx, strategy_id, "strategy_doc_md", doc_path)
    return {"doc_path": str(doc_path), "sections": list(sections)}


# ---------------------------------------------------------------------------
# S4 rule strategy: mining, evaluation, and the rule-set selection gate helper.
# ---------------------------------------------------------------------------
# A single-rule lift this high (or a hit bad rate this high) usually means a
# leakage/near-target feature slipped into the candidate set, not a genuine
# reject rule -- surfaced so a reviewer can drop it before adoption.
_SUSPECT_LEAKAGE_LIFT = 10.0
_SUSPECT_LEAKAGE_BAD_RATE = 0.9
# Two rules co-hitting more than this share (Jaccard) are largely redundant.
_HIGH_OVERLAP_THRESHOLD = 0.8
# An included rule whose population share is below this fixed floor is flagged
# low_support (mirrors bands.py's _SPARSE_BAND_THRESHOLD). Distinct from the
# caller's min_support MINING filter: a caller may mine at a looser min_support
# (e.g. 0.01) yet still want a warning on any sub-2% rule before adoption.
_LOW_SUPPORT_FLOOR = 0.02


def tool_mine_rules(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    feature_cols = _optional_str_list(inputs.get("feature_cols"))
    columns = _unique([*(feature_cols or []), target_col]) if feature_cols else None
    frame = _dataset_frame(
        runtime,
        dataset_id,
        task_id=str(ctx.task_id),
        columns=columns,
    )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, target_col, drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    resolved_features = feature_cols or _default_feature_cols(frame, target_col)
    min_support = _float_or(inputs.get("min_support"), 0.02)
    min_lift = _float_or(inputs.get("min_lift"), 1.5)
    candidates = mine_rules(
        frame,
        feature_cols=resolved_features,
        target_col=target_col,
        max_depth=int(inputs.get("max_depth", 3)),
        min_support=min_support,
        min_lift=min_lift,
        top_k=int(inputs.get("top_k", 20)),
        seed=int(inputs.get("seed", DEFAULT_MINE_SEED)),
    )
    candidate_rules = [rule.as_dict() for rule in candidates]
    red_flags = _mine_red_flags(candidate_rules, nan_labels_dropped)
    return {
        "candidate_rules": candidate_rules,
        "n_rows": int(len(frame)),
        "feature_cols": list(resolved_features),
        "red_flags": red_flags,
        "nan_labels_dropped": nan_labels_dropped,
    }


def tool_evaluate_rule_set(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    rules_ordered = [dict(rule) for rule in (inputs.get("rules") or []) if isinstance(rule, dict)]
    frame = _dataset_frame(runtime, dataset_id, task_id=str(ctx.task_id))
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, target_col, drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    result = evaluate_rule_set(
        frame,
        rules_ordered,
        target_col=target_col,
        decision=str(inputs.get("decision") or "reject"),
    )
    red_flags = _evaluate_red_flags(result, rules_ordered, nan_labels_dropped)
    result["red_flags"] = red_flags
    result["nan_labels_dropped"] = nan_labels_dropped
    return result


def tool_select_rule_set(inputs: dict, ctx) -> dict:
    """Lightweight rule-set selection gate helper (S4).

    Assembles the user-selected ordered subset of the mined candidate rules into
    a gate payload and passes it through unchanged. ``selection`` is a literal
    ``None`` default in the template step's inputs so the generic apply_adjust
    gate-override channel (agent/gate_execution_adapter.py) can overwrite it with
    the parsed 「选 1,3,5」/「全选」/「去掉 2」 instruction -- exactly the band_edges
    precedent. A ``None`` selection means "keep all candidates" (no filter yet).
    """
    candidate_rules = [dict(rule) for rule in (inputs.get("candidate_rules") or []) if isinstance(rule, dict)]
    selection = inputs.get("selection")
    decision = str(inputs.get("decision") or "reject")
    selected = [_build_ready_rule(rule, decision) for rule in _apply_rule_selection(candidate_rules, selection)]
    return {
        "selected_rules": selected,
        "selected_count": len(selected),
        "candidate_count": len(candidate_rules),
    }


def _build_ready_rule(rule: dict, decision: str) -> dict:
    """Shape a mined candidate into a build_strategy-ready rule dict.

    build_strategy needs {condition, decision(, value)}; a mined CandidateRule
    carries only condition + display stats (lift/support/source/hit_bad_rate).
    Attach the reject decision and keep the display fields (build_strategy reads
    only condition/decision/value and ignores the rest, so they ride along for
    the renderer/waterfall without affecting the strategy)."""
    ready = dict(rule)
    ready["condition"] = str(rule.get("condition", ""))
    ready["decision"] = decision
    return ready


def _apply_rule_selection(candidate_rules: list[dict], selection) -> list[dict]:
    """Resolve a parsed selection into an ordered subset of candidate_rules.

    ``selection`` is None (keep all) or a list of 1-based indices in the display
    order the user chose (e.g. [1, 3, 5]); the returned order follows the
    selection order, not the candidate order, so the user can also reorder.
    Out-of-range/duplicate indices are dropped defensively -- the gate reply
    parser already validated them, this is belt-and-braces."""
    if selection is None:
        return [dict(rule) for rule in candidate_rules]
    ordered: list[dict] = []
    seen: set[int] = set()
    for raw in selection:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if index < 1 or index > len(candidate_rules) or index in seen:
            continue
        seen.add(index)
        ordered.append(dict(candidate_rules[index - 1]))
    return ordered


def _mine_red_flags(candidate_rules: list[dict], nan_labels_dropped: int) -> list[dict]:
    red_flags: list[dict] = []
    for rule in candidate_rules:
        lift = _finite(rule.get("lift"))
        hit_bad_rate = _finite(rule.get("hit_bad_rate"))
        if (lift is not None and lift > _SUSPECT_LEAKAGE_LIFT) or (
            hit_bad_rate is not None and hit_bad_rate > _SUSPECT_LEAKAGE_BAD_RATE
        ):
            red_flags.append(
                {
                    "code": "suspect_leakage",
                    "level": "red",
                    "message": (
                        f"规则 {rule.get('rule_id')}（{rule.get('condition')}）lift="
                        f"{_fmt_num(lift)}、命中坏率={_fmt_pct(hit_bad_rate)}，疑似泄漏/近目标特征入选，请核查。"
                    ),
                }
            )
        support = _finite(rule.get("support"))
        if support is not None and support < _LOW_SUPPORT_FLOOR:
            red_flags.append(
                {
                    "code": "low_support",
                    "level": "amber",
                    "message": (
                        f"规则 {rule.get('rule_id')}（{rule.get('condition')}）支持度 "
                        f"{_fmt_pct(support)} 低于 {_fmt_pct(_LOW_SUPPORT_FLOOR)} 底线，样本量偏小。"
                    ),
                }
            )
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    return red_flags


def _evaluate_red_flags(result: dict, rules_ordered: list[dict], nan_labels_dropped: int) -> list[dict]:
    red_flags: list[dict] = []
    waterfall = result.get("waterfall") or []
    for row in waterfall:
        if int(row.get("incremental_hits") or 0) == 0:
            red_flags.append(
                {
                    "code": "rule_shadowed",
                    "level": "amber",
                    "message": (
                        f"规则 {row.get('rule_id')} 在瀑布中零增量命中（被前序规则完全覆盖），可考虑移除。"
                    ),
                }
            )
    overlap = result.get("overlap_matrix") or []
    for i in range(len(overlap)):
        for j in range(i + 1, len(overlap)):
            share = _finite(overlap[i][j])
            if share is not None and share > _HIGH_OVERLAP_THRESHOLD:
                red_flags.append(
                    {
                        "code": "high_overlap",
                        "level": "amber",
                        "message": (
                            f"规则 {waterfall[i].get('rule_id')} 与 "
                            f"{waterfall[j].get('rule_id')} 重叠 {_fmt_pct(share)} "
                            f"(>{_fmt_pct(_HIGH_OVERLAP_THRESHOLD)})，高度冗余。"
                        ),
                    }
                )
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    return red_flags
def _default_feature_cols(frame: pd.DataFrame, target_col: str) -> list[str]:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    return [column for column in numeric if column != target_col]


def _optional_str_list(value) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        cleaned = [str(item) for item in value if str(item).strip()]
        return cleaned or None
    return None


def _float_or(value, default: float) -> float:
    number = _optional_float(value)
    return default if number is None else number


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt_num(value) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.2f}"


def _fmt_pct(value) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:.1f}%"


def _write_strategy_artifact_audit(runtime, ctx, strategy_id: str, kind: str, path) -> None:
    from marvis.repositories.strategy import _write_audit_row

    from marvis.db_schema import connect

    with connect(runtime.settings.db_path) as conn:
        _write_audit_row(
            conn,
            kind="strategy.artifact",
            target_ref=strategy_id,
            outcome="succeeded",
            detail={"task_id": str(ctx.task_id), "kind": kind, "path": str(path)},
        )


def _band_stats_from_inputs(value) -> list[dict]:
    if value in (None, ""):
        return []
    if isinstance(value, dict):
        bands = value.get("bands")
        if isinstance(bands, list):
            return [dict(band) for band in bands if isinstance(band, dict)]
        return []
    if isinstance(value, list):
        return [dict(band) for band in value if isinstance(band, dict)]
    return []


class _Runtime(PackRuntime):
    def _extend(self, ctx) -> None:
        self.strategies = StrategyRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _dataset_frame(
    runtime: _Runtime,
    dataset_id: str,
    *,
    task_id: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    dataset = runtime.registry.get(dataset_id)
    if str(dataset.task_id) != str(task_id):
        raise StrategyError(f"dataset not found: {dataset_id}")
    return runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=columns)


def _owned_dataset(runtime: _Runtime, dataset_id: str, *, task_id: str):
    try:
        dataset = runtime.registry.get(dataset_id)
    except KeyError:
        raise StrategyError(f"dataset not found: {dataset_id}") from None
    if str(dataset.task_id) != str(task_id):
        raise StrategyError(f"dataset not found: {dataset_id}")
    return dataset


def _strategy(runtime: _Runtime, strategy_id: str, *, task_id: str) -> Strategy:
    strategy = runtime.strategies.get_strategy(strategy_id)
    metadata = runtime.strategies.get_strategy_meta(strategy_id)
    if (
        strategy is None
        or metadata is None
        or str(metadata["task_id"]) != str(task_id)
    ):
        raise StrategyError(f"strategy not found: {strategy_id}")
    return strategy


def _profit_params(payload: dict) -> ProfitParams:
    return ProfitParams(
        annual_rate=float(payload["annual_rate"]),
        funding_rate=float(payload["funding_rate"]),
        lgd=float(payload["lgd"]),
        operating_cost_per_loan=float(payload["operating_cost_per_loan"]),
        term_months=int(payload["term_months"]),
    )


def _optional_profit_params(payload) -> ProfitParams | None:
    return None if payload in (None, "") else _profit_params(dict(payload))


def _approval_profit_inputs(
    *,
    strategy_type: str,
    profit_params,
    ead_col: str | None,
    pd_col: str | None,
) -> tuple[dict | None, ApprovalProfitInputs | None]:
    if strategy_type not in {"approval", "reject"}:
        if profit_params not in (None, "") or ead_col is not None or pd_col is not None:
            raise StrategyError(
                "profit_params/ead_col/pd_col are only valid for approval/reject; "
                "use economics_inputs for limit or pricing"
            )
        return None, None
    if profit_params in (None, ""):
        # No economics was requested: the canonical envelope must stay empty.
        # The historical flat Tool projection alone retains expected_profit=0.0.
        return None, None
    if not ead_col or not pd_col:
        return {
            "expected_profit": None,
            "profit_note": (
                "已请求利润回测，但缺少 pd_col/ead_col，无法计算预期损失链，"
                "expected_profit 记为不可用（未用 0 冒充）。"
            ),
        }, None
    return None, ApprovalProfitInputs(
        params=_profit_params(dict(profit_params)),
        ead_col=ead_col,
        pd_col=pd_col,
    )


def _typed_economics_inputs(
    frame: pd.DataFrame,
    *,
    strategy_type: str,
    payload,
) -> dict | None:
    if payload in (None, ""):
        return None
    if strategy_type not in {"limit", "pricing"}:
        raise StrategyError(
            "economics_inputs are only valid for limit or pricing strategies"
        )
    values = dict(payload)
    required = (
        ("pd", "lgd", "utilization")
        if strategy_type == "limit"
        else (
            "ead",
            "pd",
            "lgd",
            "funding_rate",
            "term_months",
            "operating_cost_per_loan",
        )
    )
    allowed = {
        key
        for name in required
        for key in (f"{name}_col", f"{name}_value")
    }
    unsupported = sorted(set(values) - allowed)
    if unsupported:
        raise StrategyError(
            f"unsupported {strategy_type} economics_inputs: "
            + ", ".join(unsupported)
        )
    normalized: dict = {}
    missing: list[str] = []
    for name in required:
        column_key = f"{name}_col"
        value_key = f"{name}_value"
        has_column = values.get(column_key) not in (None, "")
        has_value = values.get(value_key) not in (None, "")
        if has_column == has_value:
            if has_column:
                raise StrategyError(
                    f"economics_inputs requires exactly one of {column_key} or "
                    f"{value_key}"
                )
            missing.append(f"{column_key}/{value_key}")
            continue
        if has_column:
            column = str(values[column_key])
            if column not in frame.columns:
                raise StrategyError(f"missing columns: {column}")
            normalized[name] = frame[column]
        else:
            if isinstance(values[value_key], bool):
                raise StrategyError(f"{value_key} must be numeric, not boolean")
            normalized[name] = float(values[value_key])
    if missing:
        raise StrategyError(
            f"{strategy_type} economics_inputs is incomplete; missing "
            + ", ".join(missing)
        )
    return normalized


def _label_coverage(total_rows: int, n_dropped: int) -> float:
    # drop_nan_labels semantics: coverage = labeled rows / total rows (DOM-11), so
    # callers see how much of the sample actually carried supervision signal.
    if total_rows <= 0:
        return 0.0
    return float((total_rows - n_dropped) / total_rows)


def _backtest_audit_summary(result: StrategyBacktestResult) -> dict[str, object]:
    """Keep audit rows useful without flattening typed result semantics."""

    if result.strategy_type in {"approval", "reject"}:
        return {
            "approve_rate": result.metrics.get("approve_rate"),
            "approve_bad_rate": result.metrics.get("approve_bad_rate"),
            "expected_profit": result.economics.get("expected_profit"),
        }
    if result.strategy_type == "limit":
        return {
            "total_limit": result.metrics.get("total_limit"),
            "mean_limit": result.metrics.get("mean_limit"),
            "expected_loss": result.economics.get("expected_loss"),
        }
    if result.strategy_type == "pricing":
        return {
            "mean_rate": result.metrics.get("mean_rate"),
            "profit": result.economics.get("profit"),
        }
    return {"segment_count": result.metrics.get("segment_count")}


def _backtest_id(
    dataset_id: str,
    result: BacktestRecord,
    *,
    source_dataset_content_hash: str | None = None,
) -> str:
    payload = {"dataset_id": dataset_id, "result": backtest_record_payload(result)}
    if not isinstance(result, StrategyBacktestResult):
        # Preserve historical legacy IDs byte-for-byte.  Typed envelopes use the
        # stricter canonical JSON path below; old rows and external references do
        # not change merely because V2 introduced a versioned result contract.
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"backtest-{digest[:12]}"
    if source_dataset_content_hash is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", source_dataset_content_hash):
            raise StrategyError(
                "source_dataset_content_hash must be a lowercase SHA256 digest"
            )
        payload["source_dataset_content_hash"] = source_dataset_content_hash
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"backtest-{digest[:12]}"


def _jsonable(value):
    if value is None:
        return None
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _unique(values: list[str | None]) -> list[str]:
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
