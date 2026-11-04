"""预期损失估计内核 (expected_loss_estimate).

用 bucket_migration 的 avg_matrix 作为月度桶间转移矩阵，把 loss_state 当作吸收态，
求各状态在 horizon 步内被吸收到 loss_state 的概率（马尔可夫吸收链，确定性线性代数，
无迭代随机）。再按各状态当前余额分布估计逐月/总预期损失 EL = balance * P(loss) * lgd。
选定迁移窗口中的每个快照对必须相隔一个日历月；存在整月缺口时拒绝月度推算。

链式近似：P_h = (T^h)[:, loss]，其中 T 为强制 loss 行吸收后的方阵（去掉 exited 列，
把概率质量重新归一到 states 内），T^h 通过矩阵幂逐步相乘得到（h 次矩阵乘法）。

迁徙窗口内没有桶内迁出观测的状态（bucket_migration 的未定义行，或全部迁出都是
exited）不能估计转移行：
- 若它是声明的 loss_state，按声明作吸收态（与有观测时的强制吸收一致）；
- 否则该状态的 P(损失) 为 None（未定义），链式计算中暂按"停留原状态"占位，
  能迁入它的状态的 P(损失) 因此是偏低下限；该状态的余额不计入 EL。两者都由
  ``unestimated_state`` 红旗显式说明，不以 0 冒充估计值。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from marvis.data.performance import parse_snapshot_month, validate_performance_frame
from marvis.data.errors import PerformanceFrameError
from marvis.packs.analysis.errors import AnalysisError
from marvis.packs.analysis.flow import bucket_migration

#: 可用月份 < 该值 -> short_history 红旗。
SHORT_HISTORY_MIN_MONTHS = 3
#: (T^h)[i][u] 超过该值视为状态 i 能在 horizon 内迁入未估计状态 u。
_REACH_EPSILON = 1e-12


@dataclass(frozen=True)
class ChainRow:
    from_state: str
    #: None = 该状态在迁徙窗口内无桶内迁出观测，P(损失) 无法估计。
    p_to_loss: float | None


@dataclass(frozen=True)
class MonthEL:
    month: str
    balance: float
    expected_loss: float
    is_reference: bool = False


@dataclass(frozen=True)
class ExpectedLossResult:
    loss_state: str
    lgd: float
    horizon_months: int
    chain: list[ChainRow]
    el_by_month: list[MonthEL]
    total_el: float
    assumptions: dict
    red_flags: list[dict] = field(default_factory=list)


def expected_loss_estimate(
    df: pd.DataFrame,
    *,
    id_col: str,
    snapshot_col: str,
    bucket_col: str,
    states: list[str] | tuple[str, ...],
    balance_col: str,
    loss_state: str | None = None,
    lgd: float = 0.6,
    horizon_months: int = 12,
    window: list[str] | tuple[str, ...] | None = None,
) -> ExpectedLossResult:
    if horizon_months < 1:
        raise AnalysisError("horizon_months 必须为正整数")
    contract = validate_performance_frame(
        df,
        id_col=id_col,
        snapshot_col=snapshot_col,
        bucket_col=bucket_col,
        states=states,
        balance_col=balance_col,
    )
    state_order = tuple(str(state) for state in states)
    resolved_loss = str(loss_state) if loss_state else state_order[-1]
    if resolved_loss not in state_order:
        raise AnalysisError(f"loss_state {resolved_loss!r} 不在 states 内")

    migration = bucket_migration(
        df,
        id_col=id_col,
        snapshot_col=snapshot_col,
        bucket_col=bucket_col,
        states=state_order,
        balance_col=None,  # transition probabilities are count-based, not balance-weighted
        window=window,
    )

    selected_gaps = [flag for flag in migration.red_flags
                     if flag.get("kind") == "month_gap" and flag.get("month") in migration.window_months]
    if selected_gaps:
        raise PerformanceFrameError(
            reason="所选迁移窗口包含非连续月快照，不能把跨期转移当作月度概率；请补齐月份或选择实际连续月份的迁移窗口。",
            problem="non_monthly_migration", column=snapshot_col,
            samples=[f"{flag['month']} -> {flag['to_month']}" for flag in selected_gaps[:5]],
        )

    transition, was_absorbing, unestimated = _absorbing_transition(
        migration.avg_matrix, state_order, resolved_loss
    )
    powered = _matrix_power(transition, horizon_months)
    loss_index = state_order.index(resolved_loss)
    chain = [
        ChainRow(
            from_state=state,
            p_to_loss=None if state in unestimated else float(powered[i][loss_index]),
        )
        for i, state in enumerate(state_order)
    ]
    p_to_loss = {row.from_state: row.p_to_loss for row in chain}

    el_by_month, total_el, months_available, reference_month, reference_unestimated = _el_by_month(
        df, contract, state_order, p_to_loss, lgd=lgd
    )

    red_flags: list[dict] = []
    if unestimated:
        unestimated_index = [state_order.index(state) for state in unestimated]
        affected = [
            state
            for i, state in enumerate(state_order)
            if state not in unestimated
            and state != resolved_loss
            and float(sum(powered[i][u] for u in unestimated_index)) > _REACH_EPSILON
        ]
        red_flags.append(
            {
                "kind": "unestimated_state",
                "states": list(unestimated),
                "affected_states": affected,
                "reference_snapshot": reference_month,
                "unestimated_balance": float(reference_unestimated),
                "message": _unestimated_message(
                    unestimated, affected, reference_month, reference_unestimated
                ),
            }
        )
    if not was_absorbing:
        red_flags.append(
            {
                "kind": "matrix_not_absorbing",
                "loss_state": resolved_loss,
                "message": (
                    f"损失态 `{resolved_loss}` 在观测迁徙矩阵中并非吸收态（有迁出概率）；"
                    "已按吸收态强制处理（该行自环=1）后估计，结果偏保守下限。"
                ),
            }
        )
    if months_available < SHORT_HISTORY_MIN_MONTHS:
        red_flags.append(
            {
                "kind": "short_history",
                "months_available": months_available,
                "message": f"可用快照月仅 {months_available} 个（<{SHORT_HISTORY_MIN_MONTHS}），迁徙矩阵估计不稳健。",
            }
        )

    assumptions = {
        "lgd": float(lgd),
        "horizon_months": int(horizon_months),
        "matrix_window": list(migration.window_months),
        "loss_state": resolved_loss,
        # total_el is a point-in-time EL of the reference snapshot (latest month),
        # NOT a cross-month sum; these keys document that口径 to gate/xlsx/renderer.
        "total_el_basis": "reference_snapshot",
        "reference_snapshot": reference_month,
    }
    return ExpectedLossResult(
        loss_state=resolved_loss,
        lgd=float(lgd),
        horizon_months=int(horizon_months),
        chain=chain,
        el_by_month=el_by_month,
        total_el=float(total_el),
        assumptions=assumptions,
        red_flags=red_flags,
    )


def _absorbing_transition(
    avg_matrix: list[list[float | None]], states: tuple[str, ...], loss_state: str
) -> tuple[np.ndarray, bool, list[str]]:
    """把 avg_matrix (NxM, M=N+1 含 exited) 收缩成 NxN 方阵并强制 loss 行吸收。

    - 丢掉 exited 列，把每行概率质量重新归一到 states 内（行和=1）；
    - loss 行强制为 one-hot 自环（吸收态）；
    - 无桶内迁出观测的行（None 行，或只迁往 exited）：loss 态按声明吸收；其它
      状态放自环占位并记入 unestimated（其 P(损失) 由调用方置为 None 并示警）。
    返回 (方阵, loss 态原本是否已近似吸收, 无法估计的非 loss 状态)。
    """
    n = len(states)
    loss_index = states.index(loss_state)
    square = np.zeros((n, n), dtype=float)
    unestimated: list[str] = []
    for i in range(n):
        cells = avg_matrix[i][:n]  # drop exited (last) column
        total = 0.0 if any(cell is None for cell in cells) else float(sum(cells))
        if total > 0:
            square[i] = np.asarray(cells, dtype=float) / total
        else:
            square[i, i] = 1.0  # placeholder; never presented as an estimate
            if i != loss_index:
                unestimated.append(states[i])
    # detect whether loss row was already (near-)absorbing before we force it;
    # an unobserved loss row carries no evidence of out-flow, so it counts as absorbing.
    was_absorbing = bool(square[loss_index, loss_index] >= 1.0 - 1e-9)
    square[loss_index] = 0.0
    square[loss_index, loss_index] = 1.0
    return square, was_absorbing, unestimated


def _unestimated_message(
    states: list[str], affected: list[str], reference_month: str | None, balance: float
) -> str:
    text = (
        f"状态 {'、'.join(states)} 在迁徙窗口内没有桶内迁出观测，到损失态的概率无法估计（显示为空）"
    )
    if affected:
        text += (
            f"；链式计算暂按其停留原状态处理，{'、'.join(affected)} 的 P(损失) 是偏低下限"
        )
    if reference_month is not None:
        text += f"；参考快照 {reference_month} 中这些状态的余额 {balance:,.2f} 未计入 EL"
    return text + "。"


def _matrix_power(matrix: np.ndarray, power: int) -> np.ndarray:
    result = np.eye(matrix.shape[0], dtype=float)
    for _ in range(power):
        result = result @ matrix
    return result


def _el_by_month(
    df: pd.DataFrame,
    contract,
    states: tuple[str, ...],
    p_to_loss: dict[str, float | None],
    *,
    lgd: float,
) -> tuple[list[MonthEL], float, int, str | None, float]:
    """Per-month point-in-time EL rows + a reference-snapshot headline total_el.

    Each MonthEL row is a self-contained point-in-time EL for that snapshot month.
    `total_el` is NOT the cross-month sum (that double-counts the same loans once
    per snapshot they appear in, inflating ~N× on an N-month panel); it is the EL
    of the reference snapshot (default = latest month present in the frame). The
    reference row carries is_reference=True. Balance in a state whose P(loss) is
    undefined (None) is excluded from EL, and the reference snapshot's excluded
    balance is returned for the caller's red flag. Returns
    (rows, total_el, months_available, reference_month, reference_unestimated_balance).
    """
    frame = df[[contract.snapshot_col, contract.bucket_col, contract.balance_col]].copy()
    frame["_month"] = frame[contract.snapshot_col].map(parse_snapshot_month)
    frame["_bucket"] = frame[contract.bucket_col].astype(str)
    frame["_balance"] = pd.to_numeric(frame[contract.balance_col], errors="coerce").fillna(0.0).astype(float)
    frame = frame[frame["_month"].notna()]
    months = sorted({str(month) for month in frame["_month"].tolist()})
    reference_month = months[-1] if months else None
    rows: list[MonthEL] = []
    reference_unestimated = 0.0
    for month in months:
        month_frame = frame[frame["_month"] == month]
        balance = float(month_frame["_balance"].sum())
        el = 0.0
        for bucket, group in month_frame.groupby("_bucket", sort=False):
            probability = p_to_loss.get(str(bucket), 0.0)
            if probability is None:
                if month == reference_month:
                    reference_unestimated += float(group["_balance"].sum())
                continue
            el += float(group["_balance"].sum()) * probability * float(lgd)
        rows.append(
            MonthEL(
                month=month,
                balance=balance,
                expected_loss=el,
                is_reference=(month == reference_month),
            )
        )
    total_el = next(
        (row.expected_loss for row in rows if row.month == reference_month), 0.0
    )
    return rows, total_el, len(months), reference_month, reference_unestimated


__all__ = [
    "SHORT_HISTORY_MIN_MONTHS",
    "ChainRow",
    "ExpectedLossResult",
    "MonthEL",
    "expected_loss_estimate",
]
