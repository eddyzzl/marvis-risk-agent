"""Labeling pack tool entrypoints (C1 标签构造与成熟度工具).

三个工具：

- ``define_label``：从 DPD 长表构造 0/1 坏标签，落一个带 target 的衍生数据集 +
  定坏口径元数据。成熟度确认门内置——表现期未闭合的 cohort 默认阻断，须显式确认。
- ``check_cohort_maturity``：只读，按 vintage cohort 报告表现期是否闭合。
- ``suggest_bad_definition``：纯桥接，从既有 roll_rate_matrix 输出推定坏口径建议。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Any

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import CohortMaturityNotConfirmedError
from marvis.data.label_construction import (
    check_cohort_maturity,
    construct_label,
    suggest_bad_definition,
)
from marvis.plugins.sdk import PackRuntime


def tool_define_label(inputs: dict, ctx) -> dict:
    """从 DPD 长表构造 0/1 坏标签，落衍生数据集 + 定坏口径元数据。

    成熟度确认门：给了 ``cohort_col`` 时先按 vintage 判定表现期是否闭合到定坏 MOB；
    有未成熟 cohort 且未 ``confirm_immature_cohorts`` -> 抛 CohortMaturityNotConfirmedError。
    """
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    id_col = str(inputs["id_col"])
    mob_col = str(inputs["mob_col"])
    cohort_col = _optional_str(inputs.get("cohort_col"))
    dpd_col = _optional_str(inputs.get("dpd_col"))
    status_col = _optional_str(inputs.get("status_col"))
    target_col = str(inputs.get("target_col") or "target")

    columns = _unique([id_col, mob_col, cohort_col, dpd_col, status_col])
    dataset = runtime.registry.get(dataset_id)
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=columns)

    observation_window = int(inputs["observation_window"])
    performance_window = int(inputs["performance_window"])
    at_mob = int(inputs["at_mob"]) if inputs.get("at_mob") is not None else None
    resolved_at_mob = observation_window + performance_window if at_mob is None else at_mob

    # 成熟度确认门（cohort_col 给定时）：表现期未闭合的 cohort 不静默纳入。
    maturity_payload: dict | None = None
    if cohort_col:
        report = check_cohort_maturity(
            frame,
            id_col=id_col,
            mob_col=mob_col,
            cohort_col=cohort_col,
            required_mob=resolved_at_mob,
        )
        maturity_payload = _maturity_to_payload(report)
        if report.immature_cohorts and not bool(inputs.get("confirm_immature_cohorts")):
            raise CohortMaturityNotConfirmedError(
                required_mob=report.required_mob,
                immature_cohorts=list(report.immature_cohorts),
                cohort_diagnostics=[_jsonable(cohort) for cohort in report.cohorts],
            )

    result = construct_label(
        frame,
        id_col=id_col,
        mob_col=mob_col,
        observation_window=observation_window,
        performance_window=performance_window,
        dpd_col=dpd_col,
        threshold_dpd=_optional_float(inputs.get("threshold_dpd")),
        status_col=status_col,
        threshold_status=_optional_str(inputs.get("threshold_status")),
        states=[str(state) for state in inputs["states"]] if inputs.get("states") else None,
        at_mob=at_mob,
        cohort_col=cohort_col,
        target_col=target_col,
    )

    registered = _register_label_frame(runtime, result.frame, dataset, ctx)
    red_flags: list[dict] = []
    if result.n_unmatured:
        red_flags.append({
            "code": "unmatured_loans",
            "level": "amber",
            "message": (
                f"{result.n_unmatured}/{result.n_loans} 笔贷款表现期未闭合到 mob"
                f"{result.definition.at_mob}，标签为 NaN（下游 NaN 标签门决定丢弃或补数据）。"
            ),
        })
    if maturity_payload and maturity_payload["immature_cohorts"]:
        red_flags.append({
            "code": "immature_cohorts_included",
            "level": "amber",
            "message": (
                f"已按确认纳入 {len(maturity_payload['immature_cohorts'])} 个未成熟 cohort，"
                f"坏率可能被低估。"
            ),
        })

    return {
        "result_dataset_id": registered.id,
        "target_col": result.target_col,
        "bad_definition": result.definition.to_dict(),
        "n_loans": result.n_loans,
        "n_bad": result.n_bad,
        "n_good": result.n_good,
        "n_unmatured": result.n_unmatured,
        "bad_rate": _safe_ratio(result.n_bad, result.n_bad + result.n_good),
        "maturity": maturity_payload,
        "red_flags": red_flags,
    }


def tool_check_cohort_maturity(inputs: dict, ctx) -> dict:
    """只读成熟度检查：按 vintage cohort 报告表现期是否闭合到定坏 MOB。"""
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    id_col = str(inputs["id_col"])
    mob_col = str(inputs["mob_col"])
    cohort_col = str(inputs["cohort_col"])
    columns = _unique([id_col, mob_col, cohort_col])
    dataset = runtime.registry.get(dataset_id)
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=columns)

    # required_mob 优先取显式值；否则由 obs+perf 推出（与 define_label 口径一致）。
    if inputs.get("required_mob") is not None:
        required_mob = int(inputs["required_mob"])
    else:
        required_mob = int(inputs["observation_window"]) + int(inputs["performance_window"])

    report = check_cohort_maturity(
        frame,
        id_col=id_col,
        mob_col=mob_col,
        cohort_col=cohort_col,
        required_mob=required_mob,
    )
    payload = _maturity_to_payload(report)
    if report.immature_cohorts:
        payload["red_flags"] = [{
            "code": "immature_cohorts",
            "level": "amber",
            "message": (
                f"{len(report.immature_cohorts)} 个 cohort 表现期未闭合到 mob{required_mob}："
                f"{', '.join(report.immature_cohorts[:5])}；纳入建模将低估坏率。"
            ),
        }]
    else:
        payload["red_flags"] = []
    return payload


def tool_suggest_bad_definition(inputs: dict, ctx) -> dict:
    """从既有 roll_rate_matrix 输出推定坏口径建议（纯桥接，不读数据集）。"""
    states = [str(state) for state in inputs["states"]]
    matrix = [[float(value) for value in row] for row in inputs["matrix"]]
    at_mob = int(inputs["at_mob"])
    threshold = inputs.get("roll_back_threshold")
    suggestion = suggest_bad_definition(
        states=states,
        matrix=matrix,
        at_mob=at_mob,
        roll_back_threshold=float(threshold) if threshold is not None else 0.10,
    )
    if suggestion is None:
        return {
            "suggestion": None,
            "message": (
                "在给定 roll_rate 矩阵与回滚率阈值下，没有回滚率足够低的逾期桶可作稳定定坏点；"
                "请手动指定定坏口径或放宽阈值。"
            ),
        }
    return {"suggestion": suggestion.to_dict(), "message": suggestion.rationale}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _maturity_to_payload(report) -> dict:
    return {
        "required_mob": report.required_mob,
        "cohorts": [_jsonable(cohort) for cohort in report.cohorts],
        "immature_cohorts": list(report.immature_cohorts),
        "all_matured": report.all_matured,
    }


def _register_label_frame(runtime, frame: pd.DataFrame, source_dataset, ctx):
    """把带 target 的标签数据集落盘并注册为衍生数据集（feature 包 _register_frame 范式）。"""
    out_path = (
        runtime.datasets_root / ctx.task_id / "labeling" / f"{source_dataset.id}_labeled.parquet"
    )
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(out_path.parent, out_path.name)
    try:
        frame.to_parquet(artifact.path, index=False)
        register_kwargs = {
            "task_id": ctx.task_id,
            "role": "derived",
            "anchor_target": source_dataset.id,
            "seed": int(ctx.seed or 0),
        }
        register_on_connection = getattr(runtime.registry, "register_existing_on_connection", None)
        transaction = getattr(runtime.registry, "transaction", None)
        if callable(register_on_connection) and callable(transaction):
            return uow.finalize_with_connection(
                transaction,
                lambda conn: register_on_connection(conn, artifact.final_path, **register_kwargs),
            )
        return uow.finalize(
            lambda: runtime.registry.register_existing(artifact.final_path, **register_kwargs)
        )
    except Exception:
        uow.rollback()
        raise


class _Runtime(PackRuntime):
    """Labeling pack needs only the base five objects (dataset read/register)."""


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _jsonable(value: Any):
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


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _unique(values: list[str | None]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
