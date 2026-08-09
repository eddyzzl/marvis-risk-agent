"""S3 组合分析套件 pack 入口 (manifest module = marvis.packs.analysis.tools).

每个 tool_* 走既有 subprocess runner：签名 (inputs: dict, ctx) -> dict，
ctx 暴露 workspace/datasets_root/task_id/seed。工具从注册表读表现期数据集为
pandas DataFrame，调用纯内核，把 dataclass 结果 _jsonable 后返回。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.authenticated_snapshot import (
    AuthenticatedSnapshotError,
    read_authenticated_parquet_snapshot,
)
from marvis.repositories.modeling import ModelingRepository
from marvis.files import sha256_file
from marvis.packs.analysis.errors import AnalysisError, MissingBaselineError
from marvis.packs.analysis.flow import bucket_migration, flow_rate
from marvis.packs.analysis.loss import expected_loss_estimate
from marvis.packs.analysis.segment import ProfitParams, segment_profile
from marvis.packs.analysis.report import build_report, gate_summary_payload
from marvis.packs.analysis.trend import feature_csi_trend, score_stability_trend
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.plugins.sdk import PackRuntime
from marvis.repositories.task_artifacts import TaskArtifactRepository


_PORTFOLIO_REPORT_ARTIFACT_KIND = "portfolio_report_xlsx"
_PORTFOLIO_REPORT_ARTIFACT_SCHEMA_VERSION = "portfolio-report-artifact.v1"
_PORTFOLIO_REPORT_ORIGIN_TOOL = "analysis.portfolio_report"
_PORTFOLIO_REPORT_PRODUCER_VERSION = "analysis.portfolio_report.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _Runtime(PackRuntime):
    def _extend(self, ctx) -> None:
        self.experiments = ExperimentStore(self.settings.db_path)
        self.modeling_repo = ModelingRepository(self.settings.db_path)
        self.task_artifacts = TaskArtifactRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def tool_flow_rate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        expected_content_hash=str(inputs["expected_content_hash"]),
        task_id=str(ctx.task_id),
    )
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
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        expected_content_hash=str(inputs["expected_content_hash"]),
        task_id=str(ctx.task_id),
    )
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
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        expected_content_hash=str(inputs["expected_content_hash"]),
        task_id=str(ctx.task_id),
    )
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
        "concentration_basis": result.concentration_basis,
        "ead_concentration": (
            _jsonable(asdict(result.ead_concentration))
            if result.ead_concentration is not None
            else None
        ),
        "ead_concentration_basis": result.ead_concentration_basis,
        "red_flags": _jsonable(result.red_flags),
    }


def tool_expected_loss_estimate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        expected_content_hash=str(inputs["expected_content_hash"]),
        task_id=str(ctx.task_id),
    )
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


def tool_score_stability_trend(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    baseline = _baseline(runtime, str(inputs["experiment_id"]))
    score_col = str(inputs.get("score_col") or "model_score")
    month_frames = _month_frames(
        runtime,
        inputs,
        score_col=score_col,
        task_id=str(ctx.task_id),
    )
    thresholds = _trend_thresholds(inputs.get("thresholds"))
    result = score_stability_trend(
        baseline,
        month_frames,
        score_col=score_col,
        warn=thresholds[0],
        fail=thresholds[1],
    )
    return _trend_output(result)


def tool_feature_csi_trend(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    baseline = _baseline(runtime, str(inputs["experiment_id"]))
    score_col = str(inputs.get("score_col") or "model_score")
    month_frames = _month_frames(
        runtime,
        inputs,
        score_col=score_col,
        task_id=str(ctx.task_id),
    )
    thresholds = _trend_thresholds(inputs.get("thresholds"))
    result = feature_csi_trend(
        baseline,
        month_frames,
        feature_cols=_optional_str_list(inputs.get("feature_cols")),
        warn=thresholds[0],
        fail=thresholds[1],
    )
    return _trend_output(result)


def tool_portfolio_gate_summary(inputs: dict, ctx) -> dict:
    """Pure assembler: aggregate each injected step output's red_flags + key
    numbers into a gate payload (the red-flag list is the gate checklist)."""
    return gate_summary_payload(
        flow=_as_dict(inputs.get("flow")),
        migration=_as_dict(inputs.get("migration")),
        segment=_as_dict(inputs.get("segment")),
        trend=_as_dict(inputs.get("trend")),
        expected_loss=_as_dict(inputs.get("expected_loss")),
    )


def tool_portfolio_report(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    out_dir = Path(runtime.settings.tasks_dir) / task_id / "portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_hash = _canonical_payload_hash(inputs)
    with tempfile.TemporaryDirectory(prefix=".portfolio-report-", dir=out_dir) as temp_dir:
        rendered_path, sheets = build_report(
            project_meta=_as_dict(inputs.get("project_meta")) or {},
            flow=_as_dict(inputs.get("flow")),
            migration=_as_dict(inputs.get("migration")),
            segment=_as_dict(inputs.get("segment")),
            trend=_as_dict(inputs.get("trend")),
            expected_loss=_as_dict(inputs.get("expected_loss")),
            out_path=Path(temp_dir) / "portfolio_report.xlsx",
        )
        content_hash = sha256_file(rendered_path)
        uow = ArtifactUnitOfWork()
        artifact = uow.stage_file(
            out_dir,
            f"portfolio_report_{content_hash[:16]}.xlsx",
        )
        try:
            shutil.copyfile(rendered_path, artifact.path)
            if sha256_file(artifact.path) != content_hash:
                raise AnalysisError("组合报告暂存内容哈希校验失败。")
            provenance = {
                "schema_version": _PORTFOLIO_REPORT_ARTIFACT_SCHEMA_VERSION,
                "producer_version": _PORTFOLIO_REPORT_PRODUCER_VERSION,
                "task_id": task_id,
                "input_hash": input_hash,
                "sheets": list(sheets),
            }

            def _register_and_audit(conn):
                if sha256_file(artifact.final_path) != content_hash:
                    raise AnalysisError("组合报告发布前内容哈希校验失败。")
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=_PORTFOLIO_REPORT_ARTIFACT_KIND,
                    path=str(artifact.final_path),
                    content_hash=content_hash,
                    origin_tool=_PORTFOLIO_REPORT_ORIGIN_TOOL,
                    provenance=provenance,
                )
                runtime.repo.write_audit_on_connection(
                    conn,
                    kind="analysis.portfolio.report",
                    target_ref=task_id,
                    outcome="succeeded",
                    detail={
                        "report_path": str(artifact.final_path),
                        "sheets": sheets,
                        "artifact_id": str(record["id"]),
                        "artifact_content_hash": content_hash,
                    },
                )
                return record

            record = uow.finalize_with_connection(
                runtime.task_artifacts.transaction,
                _register_and_audit,
            )
        except Exception:
            uow.rollback()
            raise
    return {
        "report_path": str(artifact.final_path),
        "sheets": sheets,
        "artifact_id": str(record["id"]),
        "artifact_content_hash": content_hash,
    }


# ---- helpers -----------------------------------------------------------------


def _dataset_frame(
    runtime: _Runtime,
    dataset_id: str,
    *,
    expected_content_hash: str | None,
    task_id: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    try:
        dataset = runtime.registry.get(dataset_id)
    except KeyError as exc:
        raise AnalysisError(f"组合分析数据集不存在：{dataset_id}。") from exc
    if str(dataset.task_id) != task_id:
        raise AnalysisError(f"组合分析数据集不属于当前任务：{dataset_id}。")

    registered_hash = str(dataset.content_hash or "")
    expected_hash = (
        registered_hash
        if expected_content_hash is None
        else str(expected_content_hash)
    )
    if (
        _SHA256_RE.fullmatch(expected_hash) is None
        or _SHA256_RE.fullmatch(registered_hash) is None
        or not hmac.compare_digest(registered_hash, expected_hash)
    ):
        raise AnalysisError("组合分析数据绑定在人工确认后发生变化，已拒绝执行。")

    source_path = str(dataset.source_path)
    try:
        frame = read_authenticated_parquet_snapshot(
            runtime.datasets_root / source_path,
            root=runtime.datasets_root,
            expected_sha256=expected_hash,
            columns=columns,
        )
    except AuthenticatedSnapshotError as exc:
        raise AnalysisError(
            "组合分析确认的数据快照未通过不可变内容校验，已拒绝执行。"
        ) from exc
    try:
        current = runtime.registry.get(dataset_id)
    except KeyError as exc:
        raise AnalysisError(
            "组合分析数据绑定在执行期间消失，已拒绝使用分析结果。"
        ) from exc
    if (
        str(current.task_id) != task_id
        or str(current.source_path) != source_path
        or not isinstance(current.content_hash, str)
        or not hmac.compare_digest(current.content_hash, expected_hash)
    ):
        raise AnalysisError(
            "组合分析数据绑定在执行期间发生变化，已拒绝使用分析结果。"
        )
    return frame


def _as_dict(value) -> dict | None:
    return dict(value) if isinstance(value, dict) else None


def _canonical_payload_hash(payload: dict) -> str:
    canonical = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

def _baseline(runtime: _Runtime, experiment_id: str) -> dict:
    experiment = runtime.experiments.get(experiment_id)
    if experiment.artifact_id is None:
        raise MissingBaselineError(
            experiment_id=experiment_id,
            reason=f"实验 {experiment_id} 尚无训练产物，无法读取基准分布快照。",
        )
    artifact = runtime.modeling_repo.get_model_artifact(experiment.artifact_id)
    if artifact is None or not artifact.baseline_distributions:
        raise MissingBaselineError(
            experiment_id=experiment_id,
            reason=(
                f"实验 {experiment_id} 的产物无训练期基准分布快照（S1b 之前训练或非二分类目标）；"
                "稳定性趋势没有可对比的基准，请重训该实验以捕获基准，或改用带基准的实验。"
            ),
        )
    return dict(artifact.baseline_distributions)


def _month_frames(
    runtime: _Runtime,
    inputs: dict,
    *,
    score_col: str,
    task_id: str,
) -> list[tuple[str, pd.DataFrame]]:
    """把趋势输入解析成按月排好的 (month, frame) 列表。

    支持两种模式：
      - ``dataset_ids``：每个 id 是一张打分衍生集，用其 ``month`` 输入或数据里的
        ``month_col`` 推断月份；这里要求每张表要么整表同月（取 month_col 众数/首值），
        要么直接由并列的 ``months`` 列表逐一对应。
      - ``dataset_id`` + ``month_col``：单张打分表按 month_col 切分成逐月子表。
    """
    dataset_ids = inputs.get("dataset_ids")
    dataset_id = _optional_str(inputs.get("dataset_id"))
    month_col = _optional_str(inputs.get("month_col"))
    if dataset_ids:
        months = inputs.get("months")
        frames: list[tuple[str, pd.DataFrame]] = []
        for index, raw_id in enumerate(dataset_ids):
            frame = _dataset_frame(
                runtime,
                str(raw_id),
                expected_content_hash=None,
                task_id=task_id,
            )
            if months and index < len(months):
                month = str(months[index])
            elif month_col and month_col in frame.columns:
                month = str(frame[month_col].iloc[0]) if len(frame) else str(index)
            else:
                month = str(index)
            frames.append((month, frame))
        return sorted(frames, key=lambda item: item[0])
    if dataset_id and month_col:
        frame = _dataset_frame(
            runtime,
            dataset_id,
            expected_content_hash=_optional_str(
                inputs.get("expected_content_hash")
            ),
            task_id=task_id,
        )
        if month_col not in frame.columns:
            raise AnalysisError(f"单表趋势要求月份列 `{month_col}` 存在。")
        out: list[tuple[str, pd.DataFrame]] = []
        for month, group in frame.groupby(frame[month_col].astype(str), sort=True):
            out.append((str(month), group))
        return out
    raise AnalysisError("趋势工具需要 dataset_ids 或 (dataset_id + month_col)。")


def _trend_thresholds(payload) -> tuple[float, float]:
    if isinstance(payload, dict):
        warn = payload.get("warn")
        fail = payload.get("fail")
        if warn is not None and fail is not None:
            return float(warn), float(fail)
    from marvis.packs.analysis.trend import _PSI_FAIL, _PSI_WARN

    return _PSI_WARN, _PSI_FAIL


def _trend_output(result) -> dict:
    return {
        "metric_name": result.metric_name,
        "trend": [_jsonable(asdict(point)) for point in result.trend],
        "per_feature_trend": [_jsonable(asdict(point)) for point in result.per_feature_trend],
        "red_flags": _jsonable(result.red_flags),
    }

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
    "tool_feature_csi_trend",
    "tool_flow_rate",
    "tool_portfolio_gate_summary",
    "tool_portfolio_report",
    "tool_score_stability_trend",
    "tool_segment_profile",
]
