from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.labels import resolve_labeled_frame
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, StrategyRepository
from marvis.packs.strategy.backtest import backtest_strategy
from marvis.packs.strategy.contracts import BacktestResult, Strategy
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.profit import ProfitParams, profit_calc
from marvis.packs.strategy.roll_rate import roll_rate_matrix
from marvis.packs.strategy.strategy import build_strategy
from marvis.packs.strategy.tradeoff import recommend_operating_point, tradeoff_view
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
    points = tradeoff_view(
        frame,
        score_col=str(inputs["score_col"]),
        target_col=str(inputs["target_col"]),
        cutoffs=[float(item) for item in inputs["cutoffs"]] if inputs.get("cutoffs") is not None else None,
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    recommended = None
    if points:
        recommended = recommend_operating_point(
            points,
            objective=str(inputs.get("objective") or "max_profit"),
            max_bad_rate=_optional_float(inputs.get("max_bad_rate")),
        )
    return {
        "points": [_jsonable(point) for point in points],
        "recommended": _jsonable(recommended),
        "nan_labels_dropped": nan_labels_dropped,
    }


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
