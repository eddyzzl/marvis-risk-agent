"""表现期快照表契约 (S3 组合分析套件, Commit 1).

组合分析套件消费的"表现期快照"是一张长表：每笔贷款在每个观察快照月上的
逾期桶状态（可选余额）。本模块定义该表的最小列契约与确定性校验函数
``validate_performance_frame``，校验失败抛结构化 typed error
(:class:`marvis.data.errors.PerformanceFrameError`)，跨子进程边界带 ``to_detail()``
诊断（与 NanLabelNotConfirmedError 同款模式），中文说明缺哪列/哪行不可解析。

桶状态的语义顺序（由好到坏 / 恶化度）机器不可猜，永远由调用方通过 ``states``
显式给定；本模块只校验"出现的桶都在 states 内"，不推断顺序。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from marvis.data.errors import PerformanceFrameError


#: 展示不可解析样例值时截断的最大字符数（错误文案里嵌具体值，避免过长）。
_SAMPLE_VALUE_MAX_LEN = 40
#: 每类问题最多展示的样例值个数。
_MAX_SAMPLES = 5


@dataclass(frozen=True)
class PerformanceFrameContract:
    """表现期快照表的已解析列契约（校验通过后返回，供下游工具直接使用）。"""

    id_col: str
    snapshot_col: str
    bucket_col: str
    balance_col: str | None
    row_count: int
    #: 数据中实际出现的桶取值（已按 states 顺序排列，仅含出现过的）。
    observed_states: tuple[str, ...]


def validate_performance_frame(
    df: pd.DataFrame,
    *,
    id_col: str,
    snapshot_col: str,
    bucket_col: str,
    states: list[str] | tuple[str, ...],
    balance_col: str | None = None,
) -> PerformanceFrameContract:
    """校验一张表现期快照表满足最小列契约，返回已解析的 :class:`PerformanceFrameContract`。

    契约要求：

    - ``id_col`` / ``snapshot_col`` / ``bucket_col`` 三列必须存在（``balance_col`` 若给定也必须存在）；
    - ``snapshot_col`` 每行都能解析成 ``YYYY-MM`` 快照月；
    - ``bucket_col`` 每个非空取值都落在 ``states`` 枚举内；
    - ``balance_col`` 若给定，每行都能解析成数值。

    任一条不满足抛 :class:`PerformanceFrameError`，``to_detail()`` 携带
    ``reason`` / ``missing_columns`` / 样例值等结构化诊断（中文文案 + 截断样例）。
    """
    state_order = tuple(str(state) for state in states)
    if not state_order:
        raise PerformanceFrameError(
            reason="states 不能为空：桶状态语义顺序必须由调用方显式给定。",
            problem="empty_states",
        )
    if len(set(state_order)) != len(state_order):
        raise PerformanceFrameError(
            reason="states 含重复桶取值；每个桶状态只能出现一次。",
            problem="duplicate_states",
            samples=[state for state in state_order if state_order.count(state) > 1][:_MAX_SAMPLES],
        )

    required = [id_col, snapshot_col, bucket_col]
    if balance_col:
        required.append(balance_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise PerformanceFrameError(
            reason=f"表现期快照表缺少必需列：{', '.join(missing)}。",
            problem="missing_columns",
            missing_columns=missing,
        )

    row_count = int(len(df))

    # snapshot 列逐行必须可解析成 YYYY-MM。
    bad_snapshots = _unparseable_snapshots(df[snapshot_col])
    if bad_snapshots:
        raise PerformanceFrameError(
            reason=f"快照月列 `{snapshot_col}` 有 {len(bad_snapshots)} 行不可解析为 YYYY-MM 快照月。",
            problem="bad_snapshot",
            column=snapshot_col,
            samples=[_truncate(value) for value in bad_snapshots[:_MAX_SAMPLES]],
        )

    # bucket 列每个非空取值必须在 states 内。
    unknown = _unknown_buckets(df[bucket_col], state_order)
    if unknown:
        raise PerformanceFrameError(
            reason=(
                f"逾期桶列 `{bucket_col}` 出现 states 之外的取值：{', '.join(_truncate(v) for v in unknown[:_MAX_SAMPLES])}；"
                f"已声明桶：{', '.join(state_order)}。"
            ),
            problem="unknown_bucket",
            column=bucket_col,
            samples=[_truncate(value) for value in unknown[:_MAX_SAMPLES]],
        )

    if balance_col:
        bad_balances = _unparseable_balances(df[balance_col])
        if bad_balances:
            raise PerformanceFrameError(
                reason=f"余额列 `{balance_col}` 有 {len(bad_balances)} 行不可解析为数值。",
                problem="bad_balance",
                column=balance_col,
                samples=[_truncate(value) for value in bad_balances[:_MAX_SAMPLES]],
            )

    observed = _observed_states(df[bucket_col], state_order)
    return PerformanceFrameContract(
        id_col=str(id_col),
        snapshot_col=str(snapshot_col),
        bucket_col=str(bucket_col),
        balance_col=str(balance_col) if balance_col else None,
        row_count=row_count,
        observed_states=observed,
    )


def parse_snapshot_month(value) -> str | None:
    """把一个快照值规范化成 ``YYYY-MM``；不可解析返回 ``None``（不抛异常）。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m")
    if hasattr(value, "strftime") and not isinstance(value, str):
        try:
            return pd.Timestamp(value).strftime("%Y-%m")
        except (ValueError, TypeError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 6 and text.isdigit():
        year, month = int(text[:4]), int(text[4:])
        return f"{year:04d}-{month:02d}" if _valid_month(year, month) else None
    if len(text) >= 7 and text[4] == "-" and text[:4].isdigit() and text[5:7].isdigit():
        year, month = int(text[:4]), int(text[5:7])
        return f"{year:04d}-{month:02d}" if _valid_month(year, month) else None
    try:
        parsed = pd.to_datetime(text, errors="raise")
    except (ValueError, TypeError):
        return None
    return pd.Timestamp(parsed).strftime("%Y-%m")


def _valid_month(year: int, month: int) -> bool:
    return year >= 1 and 1 <= month <= 12


def _unparseable_snapshots(series: pd.Series) -> list:
    bad: list = []
    for value in series.tolist():
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            bad.append(value)
            continue
        if parse_snapshot_month(value) is None:
            bad.append(value)
    return bad


def _unknown_buckets(series: pd.Series, states: tuple[str, ...]) -> list[str]:
    valid = set(states)
    seen: list[str] = []
    seen_set: set[str] = set()
    for value in series.tolist():
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        text = str(value)
        if text not in valid and text not in seen_set:
            seen_set.add(text)
            seen.append(text)
    return seen


def _observed_states(series: pd.Series, states: tuple[str, ...]) -> tuple[str, ...]:
    present: set[str] = set()
    for value in series.tolist():
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        present.add(str(value))
    return tuple(state for state in states if state in present)


def _unparseable_balances(series: pd.Series) -> list:
    numeric = pd.to_numeric(series, errors="coerce")
    original_na = series.isna().to_numpy()
    coerced_na = numeric.isna().to_numpy()
    bad: list = []
    values = series.tolist()
    for index in range(len(values)):
        if coerced_na[index] and not original_na[index]:
            bad.append(values[index])
    return bad


def _truncate(value) -> str:
    text = str(value)
    if len(text) > _SAMPLE_VALUE_MAX_LEN:
        return text[:_SAMPLE_VALUE_MAX_LEN] + "…"
    return text


__all__ = [
    "PerformanceFrameContract",
    "parse_snapshot_month",
    "validate_performance_frame",
]
