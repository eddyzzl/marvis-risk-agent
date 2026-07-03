"""S3 组合分析套件 pack 入口 (manifest module = marvis.packs.analysis.tools).

每个 tool_* 走既有 subprocess runner：签名 (inputs: dict, ctx) -> dict，
ctx 暴露 workspace/datasets_root/task_id/seed。工具从注册表读表现期数据集为
pandas DataFrame，调用纯内核，把 dataclass 结果 _jsonable 后返回。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository
from marvis.packs.analysis.flow import bucket_migration, flow_rate
from marvis.packs.analysis.loss import expected_loss_estimate
from marvis.packs.analysis.segment import ProfitParams, segment_profile
from marvis.settings import build_settings


class _Runtime:
    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def tool_flow_rate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    result = flow_rate(
        frame,
        id_col=str(inputs["id_col"]),
        snapshot_col=str(inputs["snapshot_col"]),
        bucket_col=str(inputs["bucket_col"]),
        states=[str(state) for state in inputs["states"]],
        balance_col=_optional_str(inputs.get("balance_col")),
        bad_states=_optional_str_list(inputs.get("bad_states")),
    )
    base_kind = "balance" if inputs.get("balance_col") else "count"
    return {
        "states": list(result.states),
        "to_states": list(result.to_states),
        "months": list(result.months),
        "matrix_by_month": [
            {
                "month": transition.month,
                "to_month": transition.to_month,
                "from_to_matrix": [[float(cell) for cell in row] for row in transition.from_to_matrix],
                "base": {state: float(value) for state, value in transition.base.items()},
                "base_kind": base_kind,
                "pair_count": transition.pair_count,
            }
            for transition in result.transitions
        ],
        "net_flows": [
            {
                "month": transition.month,
                "into_bad": float(transition.into_bad),
                "out_of_bad": float(transition.out_of_bad),
            }
            for transition in result.transitions
        ],
        "red_flags": _jsonable(result.red_flags),
    }


def tool_bucket_migration(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    result = bucket_migration(
        frame,
        id_col=str(inputs["id_col"]),
        snapshot_col=str(inputs["snapshot_col"]),
        bucket_col=str(inputs["bucket_col"]),
        states=[str(state) for state in inputs["states"]],
        balance_col=_optional_str(inputs.get("balance_col")),
        window=_optional_str_list(inputs.get("window")),
        bad_states=_optional_str_list(inputs.get("bad_states")),
    )
    return {
        "states": list(result.states),
        "to_states": list(result.to_states),
        "window_months": list(result.window_months),
        "avg_matrix": [[float(cell) for cell in row] for row in result.avg_matrix],
        "worst_matrix": [[float(cell) for cell in row] for row in result.worst_matrix],
        "heat_table": _jsonable(result.heat_table),
        "red_flags": _jsonable(result.red_flags),
    }


def tool_segment_profile(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    result = segment_profile(
        frame,
        segment_col=str(inputs["segment_col"]),
        target_col=_optional_str(inputs.get("target_col")),
        score_col=_optional_str(inputs.get("score_col")),
        approved_col=_optional_str(inputs.get("approved_col")),
        profit_params=_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
        top_k=int(inputs.get("top_k", 20)),
    )
    return {
        "segments": [_jsonable(asdict(row)) for row in result.segments],
        "concentration": _jsonable(asdict(result.concentration)),
        "red_flags": _jsonable(result.red_flags),
    }


def tool_expected_loss_estimate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    result = expected_loss_estimate(
        frame,
        id_col=str(inputs["id_col"]),
        snapshot_col=str(inputs["snapshot_col"]),
        bucket_col=str(inputs["bucket_col"]),
        states=[str(state) for state in inputs["states"]],
        balance_col=str(inputs["balance_col"]),
        loss_state=_optional_str(inputs.get("loss_state")),
        lgd=float(inputs.get("lgd", 0.6)),
        horizon_months=int(inputs.get("horizon_months", 12)),
        window=_optional_str_list(inputs.get("window")),
    )
    return {
        "loss_state": result.loss_state,
        "chain": [_jsonable(asdict(row)) for row in result.chain],
        "el_by_month": [_jsonable(asdict(row)) for row in result.el_by_month],
        "total_el": float(result.total_el),
        "assumptions": _jsonable(result.assumptions),
        "red_flags": _jsonable(result.red_flags),
    }


# ---- helpers -----------------------------------------------------------------


def _dataset_frame(runtime: _Runtime, dataset_id: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    dataset = runtime.registry.get(dataset_id)
    return runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=columns)


def _profit_params(payload) -> ProfitParams | None:
    if not payload:
        return None
    data = dict(payload)
    return ProfitParams(
        annual_rate=float(data["annual_rate"]),
        funding_rate=float(data["funding_rate"]),
        lgd=float(data["lgd"]),
        operating_cost_per_loan=float(data["operating_cost_per_loan"]),
        term_months=int(data["term_months"]),
    )


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_str_list(value) -> list[str] | None:
    if not value:
        return None
    return [str(item) for item in value]


def _jsonable(value):
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


__all__ = [
    "tool_bucket_migration",
    "tool_expected_loss_estimate",
    "tool_flow_rate",
    "tool_segment_profile",
]
