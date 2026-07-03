"""桶流量 / 桶迁徙内核 (flow_rate + bucket_migration).

两个工具共用同一个"相邻快照对齐"内核 ``_aligned_transitions``：对每笔贷款按
快照月排序，取相邻月对 (from_month -> to_month)，统计 from 桶 -> to 桶 的转移。
缺失下月快照的贷款计入显式的 ``exited`` 伪状态（显式列出，不静默丢弃）。

- ``flow_rate``：逐相邻月对给出 NxN 转移占比矩阵 + into_bad/out_of_bad 净流量。
- ``bucket_migration``：把窗口内各月对聚合成平均迁徙率矩阵 + 逐单元格最差月矩阵。

roll_rate_matrix (strategy 包) 保留不动——那是"状态×时间长表"口径，本内核是
"相邻快照对齐"口径，两者语义不同。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from marvis.data.performance import parse_snapshot_month, validate_performance_frame
from marvis.packs.analysis.errors import AnalysisError

#: 显式伪状态：贷款在某月有快照，但下月没有对齐快照（退出观察窗）。
EXITED_STATE = "exited"
#: 某个相邻月对的对齐对数低于该阈值 -> sparse_month 红旗。
SPARSE_MONTH_MIN_PAIRS = 100


@dataclass(frozen=True)
class MonthTransition:
    """一个相邻月对 (from_month -> to_month) 的转移统计。"""

    month: str  # from_month (转移发生的起始月)
    to_month: str
    #: from_to_matrix[i][j] = 从 states[i] 转移到 (states + [exited])[j] 的占比。
    from_to_matrix: list[list[float]]
    #: 每个 from 桶的基数（count 或 balance 口径）。
    base: dict[str, float]
    into_bad: float
    out_of_bad: float
    pair_count: int


@dataclass(frozen=True)
class FlowRateResult:
    states: tuple[str, ...]
    #: 输出列顺序 = states + [exited]。
    to_states: tuple[str, ...]
    months: tuple[str, ...]
    transitions: list[MonthTransition]
    red_flags: list[dict] = field(default_factory=list)


def flow_rate(
    df: pd.DataFrame,
    *,
    id_col: str,
    snapshot_col: str,
    bucket_col: str,
    states: list[str] | tuple[str, ...],
    balance_col: str | None = None,
    bad_states: list[str] | tuple[str, ...] | None = None,
) -> FlowRateResult:
    """逐相邻月对统计桶间转移（纯函数）。

    ``bad_states`` 缺省为 states 后半段（用于 into_bad/out_of_bad 净流量口径）；
    这里默认取除首个状态外的所有更坏状态里最坏的那个作为"坏"边界不够通用，
    因此明确要求：缺省时把 states 最后一个作为唯一坏态，调用方可显式覆盖。
    """
    contract = validate_performance_frame(
        df,
        id_col=id_col,
        snapshot_col=snapshot_col,
        bucket_col=bucket_col,
        states=states,
        balance_col=balance_col,
    )
    state_order = tuple(str(state) for state in states)
    bad_set = _resolve_bad_states(state_order, bad_states)
    prepared = _prepare_frame(df, contract)

    pairs = _adjacent_pairs(prepared, state_order)
    months = tuple(sorted({pair["from_month"] for pair in pairs}))
    transitions: list[MonthTransition] = []
    red_flags: list[dict] = []
    to_states = (*state_order, EXITED_STATE)

    for from_month in months:
        month_pairs = [pair for pair in pairs if pair["from_month"] == from_month]
        to_month = month_pairs[0]["to_month"] if month_pairs else ""
        matrix, base, into_bad, out_of_bad = _month_matrix(
            month_pairs, state_order, to_states, bad_set, use_balance=bool(balance_col)
        )
        pair_count = len(month_pairs)
        transitions.append(
            MonthTransition(
                month=from_month,
                to_month=to_month,
                from_to_matrix=matrix,
                base=base,
                into_bad=into_bad,
                out_of_bad=out_of_bad,
                pair_count=pair_count,
            )
        )
        if pair_count < SPARSE_MONTH_MIN_PAIRS:
            red_flags.append(
                {
                    "kind": "sparse_month",
                    "month": from_month,
                    "pair_count": pair_count,
                    "message": f"月 {from_month} 对齐月对仅 {pair_count} 对（<{SPARSE_MONTH_MIN_PAIRS}），转移率不稳健。",
                }
            )

    return FlowRateResult(
        states=state_order,
        to_states=to_states,
        months=months,
        transitions=transitions,
        red_flags=red_flags,
    )


@dataclass(frozen=True)
class BucketMigrationResult:
    states: tuple[str, ...]
    to_states: tuple[str, ...]
    window_months: tuple[str, ...]
    avg_matrix: list[list[float]]
    worst_matrix: list[list[float]]
    #: 渲染用行列表：每行 {"from": state, "<to_state>": rate, ...}。
    heat_table: list[dict]
    red_flags: list[dict] = field(default_factory=list)


def bucket_migration(
    df: pd.DataFrame,
    *,
    id_col: str,
    snapshot_col: str,
    bucket_col: str,
    states: list[str] | tuple[str, ...],
    balance_col: str | None = None,
    window: list[str] | tuple[str, ...] | None = None,
    bad_states: list[str] | tuple[str, ...] | None = None,
) -> BucketMigrationResult:
    """把窗口内相邻月对聚合成平均迁徙率矩阵 + 逐单元格最差月矩阵（与 flow_rate 共用对齐内核）。"""
    result = flow_rate(
        df,
        id_col=id_col,
        snapshot_col=snapshot_col,
        bucket_col=bucket_col,
        states=states,
        balance_col=balance_col,
        bad_states=bad_states,
    )
    window_months = _resolve_window(result.months, window)
    selected = [t for t in result.transitions if t.month in window_months]
    state_order = result.states
    to_states = result.to_states

    n_from = len(state_order)
    n_to = len(to_states)
    if not selected:
        empty = [[0.0] * n_to for _ in range(n_from)]
        return BucketMigrationResult(
            states=state_order,
            to_states=to_states,
            window_months=tuple(window_months),
            avg_matrix=empty,
            worst_matrix=[[0.0] * n_to for _ in range(n_from)],
            heat_table=_heat_table(empty, state_order, to_states),
            red_flags=list(result.red_flags),
        )

    avg = [[0.0] * n_to for _ in range(n_from)]
    worst = [[0.0] * n_to for _ in range(n_from)]
    for i in range(n_from):
        for j in range(n_to):
            cell_values = [t.from_to_matrix[i][j] for t in selected]
            avg[i][j] = sum(cell_values) / len(cell_values)
            # worst = 最差月：对角(留存/改善)取最小，非对角向坏取最大。这里统一
            # 用"最大迁出率"语义 -- worst 关注最坏迁移强度，取每个单元格跨月最大值。
            worst[i][j] = max(cell_values)

    return BucketMigrationResult(
        states=state_order,
        to_states=to_states,
        window_months=tuple(window_months),
        avg_matrix=avg,
        worst_matrix=worst,
        heat_table=_heat_table(avg, state_order, to_states),
        red_flags=list(result.red_flags),
    )


# ---- shared alignment kernel -------------------------------------------------


def _prepare_frame(df: pd.DataFrame, contract) -> pd.DataFrame:
    frame = df[[contract.id_col, contract.snapshot_col, contract.bucket_col]].copy()
    if contract.balance_col:
        frame[contract.balance_col] = pd.to_numeric(df[contract.balance_col], errors="coerce").fillna(0.0)
    frame["_id"] = frame[contract.id_col].astype(str)
    frame["_month"] = frame[contract.snapshot_col].map(parse_snapshot_month)
    frame["_bucket"] = frame[contract.bucket_col].astype(str)
    frame = frame[frame["_month"].notna()].copy()
    if contract.balance_col:
        frame["_balance"] = frame[contract.balance_col].astype(float)
    else:
        frame["_balance"] = 1.0
    return frame


def _adjacent_pairs(frame: pd.DataFrame, states: tuple[str, ...]) -> list[dict]:
    """对每笔贷款按月排序，产出相邻月对转移；缺下月快照 -> exited。"""
    valid_states = set(states)
    all_months = sorted({str(month) for month in frame["_month"].tolist()})
    month_pos = {month: index for index, month in enumerate(all_months)}
    pairs: list[dict] = []
    for _loan_id, group in frame.sort_values(["_id", "_month"], kind="mergesort").groupby("_id", sort=False):
        rows = group[["_month", "_bucket", "_balance"]].to_dict("records")
        by_month = {str(row["_month"]): row for row in rows}
        months_present = sorted(by_month.keys())
        for month in months_present:
            row = by_month[month]
            from_bucket = str(row["_bucket"])
            if from_bucket not in valid_states:
                # already validated, but guard defensively
                raise AnalysisError(f"unknown bucket in flow alignment: {from_bucket!r}")
            next_index = month_pos[month] + 1
            if next_index >= len(all_months):
                continue  # no defined next month in the panel; not an exit, just window edge
            next_month = all_months[next_index]
            if next_month in by_month:
                to_bucket = str(by_month[next_month]["_bucket"])
            else:
                to_bucket = EXITED_STATE
            pairs.append(
                {
                    "from_month": month,
                    "to_month": next_month,
                    "from_bucket": from_bucket,
                    "to_bucket": to_bucket,
                    "balance": float(row["_balance"]),
                }
            )
    return pairs


def _month_matrix(
    month_pairs: list[dict],
    states: tuple[str, ...],
    to_states: tuple[str, ...],
    bad_set: set[str],
    *,
    use_balance: bool,
) -> tuple[list[list[float]], dict[str, float], float, float]:
    from_index = {state: i for i, state in enumerate(states)}
    to_index = {state: j for j, state in enumerate(to_states)}
    n_from = len(states)
    n_to = len(to_states)
    counts = [[0.0] * n_to for _ in range(n_from)]
    base = {state: 0.0 for state in states}
    into_bad = 0.0
    out_of_bad = 0.0
    for pair in month_pairs:
        weight = pair["balance"] if use_balance else 1.0
        i = from_index[pair["from_bucket"]]
        j = to_index[pair["to_bucket"]]
        counts[i][j] += weight
        base[pair["from_bucket"]] += weight
        from_bad = pair["from_bucket"] in bad_set
        to_bad = pair["to_bucket"] in bad_set
        if not from_bad and to_bad:
            into_bad += weight
        elif from_bad and not to_bad and pair["to_bucket"] != EXITED_STATE:
            out_of_bad += weight
    matrix = [
        [(counts[i][j] / base[states[i]] if base[states[i]] else 0.0) for j in range(n_to)]
        for i in range(n_from)
    ]
    return matrix, base, into_bad, out_of_bad


def _heat_table(matrix: list[list[float]], states: tuple[str, ...], to_states: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for i, from_state in enumerate(states):
        row = {"from": from_state}
        for j, to_state in enumerate(to_states):
            row[to_state] = matrix[i][j]
        rows.append(row)
    return rows


def _resolve_bad_states(
    states: tuple[str, ...], bad_states: list[str] | tuple[str, ...] | None
) -> set[str]:
    if bad_states:
        unknown = [state for state in bad_states if str(state) not in set(states)]
        if unknown:
            raise AnalysisError(f"bad_states not in states: {', '.join(str(s) for s in unknown)}")
        return {str(state) for state in bad_states}
    # default: the single worst (last) state
    return {states[-1]} if states else set()


def _resolve_window(
    months: tuple[str, ...], window: list[str] | tuple[str, ...] | None
) -> list[str]:
    if not window:
        return list(months)
    requested = [str(month) for month in window]
    return [month for month in months if month in requested]


__all__ = [
    "EXITED_STATE",
    "SPARSE_MONTH_MIN_PAIRS",
    "BucketMigrationResult",
    "FlowRateResult",
    "MonthTransition",
    "bucket_migration",
    "flow_rate",
]
