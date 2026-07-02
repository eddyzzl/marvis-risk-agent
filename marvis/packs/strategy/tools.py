from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.direction import check_score_direction, normalize_score_direction
from marvis.data.labels import resolve_labeled_frame
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, StrategyRepository
from marvis.packs.strategy.backtest import backtest_strategy
from marvis.packs.strategy.bands import design_cutoff_bands
from marvis.packs.strategy.compare import compare_strategies
from marvis.packs.strategy.contracts import BacktestResult, Strategy
from marvis.packs.strategy.deliverables import (
    build_monitoring_plan,
    decision_table_csv,
)
from marvis.packs.strategy.doc import render_strategy_doc_markdown
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.profit import ProfitParams, profit_calc
from marvis.packs.strategy.roll_rate import roll_rate_matrix
from marvis.packs.strategy.strategy import build_strategy, infer_strategy_rule_direction
from marvis.packs.strategy.tradeoff import (
    recommend_operating_point,
    tradeoff_feasible_flags,
    tradeoff_view,
)
from marvis.packs.strategy.vintage import vintage_curve, vintage_summary
from marvis.settings import build_settings


def tool_vintage_curve(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        columns=[str(inputs["cohort_col"]), str(inputs["mob_col"]), str(inputs["bad_col"])],
    )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        str(inputs["bad_col"]),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    curve = vintage_curve(
        frame,
        cohort_col=str(inputs["cohort_col"]),
        mob_col=str(inputs["mob_col"]),
        bad_col=str(inputs["bad_col"]),
        mob_max=int(inputs.get("mob_max", 12)),
    )
    return {
        "cohorts": list(curve.cohorts),
        "mob_axis": list(curve.mob_axis),
        "curves": _jsonable(curve.curves),
        "counts": _jsonable(curve.counts),
        "summary": vintage_summary(curve, ref_mob=int(inputs.get("ref_mob", 6))),
        "nan_labels_dropped": nan_labels_dropped,
    }


def tool_roll_rate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        columns=[str(inputs["id_col"]), str(inputs["time_col"]), str(inputs["status_col"])],
    )
    matrix = roll_rate_matrix(
        frame,
        id_col=str(inputs["id_col"]),
        time_col=str(inputs["time_col"]),
        status_col=str(inputs["status_col"]),
        states=[str(item) for item in inputs["states"]],
    )
    return {
        "states": list(matrix.states),
        "matrix": [list(row) for row in matrix.matrix],
        "base_counts": dict(matrix.base_counts),
    }


def tool_profit_calc(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    segment_col = _optional_str(inputs.get("segment_col"))
    columns = _unique([segment_col, str(inputs["ead_col"]), str(inputs["pd_col"])])
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]), columns=columns)
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
    strategy = build_strategy(
        str(inputs["strategy_type"]),
        list(inputs["rules"]),
        score_col=_optional_str(inputs.get("score_col")),
        default_decision=inputs.get("default_decision"),
        description=str(inputs.get("description") or ""),
    )
    if runtime.strategies.get_strategy(strategy.id) is None:
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
    return {
        "strategy_id": strategy.id,
        "strategy_type": strategy.strategy_type,
        "score_col": strategy.score_col,
        "default_decision": strategy.default_decision,
        "description": strategy.description,
        "rules": [_jsonable(rule) for rule in strategy.rules],
        "inferred_score_direction": infer_strategy_rule_direction(list(strategy.rules), strategy.score_col),
    }


def tool_backtest_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy = _strategy(runtime, str(inputs["strategy_id"]))
    baseline_id = _optional_str(inputs.get("baseline_strategy_id"))
    baseline = _strategy(runtime, baseline_id) if baseline_id else None
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, str(inputs["target_col"]), drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    result = backtest_strategy(
        frame,
        strategy,
        target_col=str(inputs["target_col"]),
        baseline=baseline,
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    backtest_id = _backtest_id(str(inputs["dataset_id"]), result)
    if runtime.strategies.get_backtest(backtest_id) is None:
        runtime.strategies.save_backtest_with_audit(
            backtest_id,
            strategy.id,
            str(inputs["dataset_id"]),
            result,
            audit={
                "kind": "strategy.backtest",
                "target_ref": backtest_id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": str(ctx.task_id),
                    "strategy_id": strategy.id,
                    "dataset_id": str(inputs["dataset_id"]),
                    "approval_rate": result.approval_rate,
                    "expected_profit": result.expected_profit,
                },
            },
        )
    payload = _jsonable(result)
    payload["backtest_id"] = backtest_id
    payload["nan_labels_dropped"] = nan_labels_dropped
    return payload


def tool_tradeoff_view(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
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
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
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
    strategy = _strategy(runtime, str(inputs["strategy_id"]))
    baseline = _strategy(runtime, str(inputs["baseline_strategy_id"]))
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
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
    return payload


def tool_adopt_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id)
    backtest_id = str(inputs["backtest_id"])
    backtest = runtime.strategies.get_backtest(backtest_id)
    if backtest is None or backtest.strategy_id != strategy_id:
        raise StrategyError(
            f"backtest {backtest_id} does not belong to strategy {strategy_id}"
        )
    adoption_reason = str(inputs["adoption_reason"])
    adopt_result = runtime.strategies.adopt_strategy_with_audit(
        strategy_id,
        reason=adoption_reason,
        audit={
            "kind": "strategy.adopt",
            "target_ref": strategy_id,
            "outcome": "succeeded",
            "detail": {
                "task_id": str(ctx.task_id),
                "backtest_id": backtest_id,
                "approval_rate": backtest.approval_rate,
                "approved_bad_rate": backtest.approved_bad_rate,
                "expected_profit": backtest.expected_profit,
            },
        },
    )
    version = int(adopt_result["version"])
    strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{strategy_id}_v{version}"

    band_stats = _band_stats_from_inputs(inputs.get("band_stats"))
    rules = [_jsonable(rule) for rule in strategy.rules]
    csv_text = decision_table_csv(rules, band_stats)
    csv_path = strategy_dir / f"decision_table_{stem}.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    monitoring_plan = build_monitoring_plan(
        strategy_id=strategy_id,
        version=version,
        approved_bad_rate=backtest.approved_bad_rate,
        approval_rate=backtest.approval_rate,
    )
    json_path = strategy_dir / f"monitoring_plan_{stem}.json"
    json_path.write_text(
        json.dumps(monitoring_plan, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    artifacts = []
    for kind, path in (
        ("decision_table_csv", csv_path),
        ("monitoring_plan_json", json_path),
    ):
        runtime.strategies.save_strategy_artifact(strategy_id, kind=kind, path=str(path))
        _write_strategy_artifact_audit(runtime, ctx, strategy_id, kind, path)
        artifacts.append({"kind": kind, "path": str(path)})

    return {
        "strategy_id": strategy_id,
        "version": version,
        "status": "adopted",
        "retired_strategy_ids": list(adopt_result["retired_strategy_ids"]),
        "artifacts": artifacts,
    }


def tool_render_strategy_doc(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id)
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


class _Runtime:
    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self.strategies = StrategyRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _dataset_frame(runtime: _Runtime, dataset_id: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    dataset = runtime.registry.get(dataset_id)
    return runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=columns)


def _strategy(runtime: _Runtime, strategy_id: str) -> Strategy:
    strategy = runtime.strategies.get_strategy(strategy_id)
    if strategy is None:
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


def _backtest_id(dataset_id: str, result: BacktestResult) -> str:
    payload = {"dataset_id": dataset_id, "result": _jsonable(result)}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
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
