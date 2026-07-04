"""预期损失估计内核 (expected_loss_estimate).

用 bucket_migration 的 avg_matrix 作为月度桶间转移矩阵，把 loss_state 当作吸收态，
求各状态在 horizon 步内被吸收到 loss_state 的概率（马尔可夫吸收链，确定性线性代数，
无迭代随机）。再按各状态当前余额分布估计逐月/总预期损失 EL = balance * P(loss) * lgd。

链式近似：P_h = (T^h)[:, loss]，其中 T 为强制 loss 行吸收后的方阵（去掉 exited 列，
把概率质量重新归一到 states 内），T^h 通过矩阵幂逐步相乘得到（h 次矩阵乘法）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from marvis.data.performance import parse_snapshot_month, validate_performance_frame
from marvis.packs.analysis.errors import AnalysisError
from marvis.packs.analysis.flow import bucket_migration

#: 可用月份 < 该值 -> short_history 红旗。
SHORT_HISTORY_MIN_MONTHS = 3


@dataclass(frozen=True)
class ChainRow:
    from_state: str
    p_to_loss: float


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

    transition, was_absorbing = _absorbing_transition(migration.avg_matrix, state_order, resolved_loss)
    powered = _matrix_power(transition, horizon_months)
    loss_index = state_order.index(resolved_loss)
    chain = [
        ChainRow(from_state=state, p_to_loss=float(powered[i][loss_index]))
        for i, state in enumerate(state_order)
    ]
    p_to_loss = {row.from_state: row.p_to_loss for row in chain}

    el_by_month, total_el, months_available, reference_month = _el_by_month(
        df, contract, state_order, p_to_loss, lgd=lgd
    )

    red_flags: list[dict] = []
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
    avg_matrix: list[list[float]], states: tuple[str, ...], loss_state: str
) -> tuple[np.ndarray, bool]:
    """把 avg_matrix (NxM, M=N+1 含 exited) 收缩成 NxN 方阵并强制 loss 行吸收。

    - 丢掉 exited 列，把每行概率质量重新归一到 states 内（行和=1）；
    - loss 行强制为 one-hot 自环（吸收态）。
    返回 (方阵, loss 态原本是否已近似吸收)。
    """
    n = len(states)
    square = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = np.asarray(avg_matrix[i][:n], dtype=float)  # drop exited (last) column
        total = float(row.sum())
        if total > 0:
            square[i] = row / total
        else:
            square[i, i] = 1.0  # no observed transitions -> treat as self-absorbing
    loss_index = states.index(loss_state)
    # detect whether loss row was already (near-)absorbing before we force it
    was_absorbing = bool(square[loss_index, loss_index] >= 1.0 - 1e-9)
    square[loss_index] = 0.0
    square[loss_index, loss_index] = 1.0
    return square, was_absorbing


def _matrix_power(matrix: np.ndarray, power: int) -> np.ndarray:
    result = np.eye(matrix.shape[0], dtype=float)
    for _ in range(power):
        result = result @ matrix
    return result


def _el_by_month(
    df: pd.DataFrame,
    contract,
    states: tuple[str, ...],
    p_to_loss: dict[str, float],
    *,
    lgd: float,
) -> tuple[list[MonthEL], float, int, str | None]:
    """Per-month point-in-time EL rows + a reference-snapshot headline total_el.

    Each MonthEL row is a self-contained point-in-time EL for that snapshot month.
    `total_el` is NOT the cross-month sum (that double-counts the same loans once
    per snapshot they appear in, inflating ~N× on an N-month panel); it is the EL
    of the reference snapshot (default = latest month present in the frame). The
    reference row carries is_reference=True. Returns
    (rows, total_el, months_available, reference_month).
    """
    frame = df[[contract.snapshot_col, contract.bucket_col, contract.balance_col]].copy()
    frame["_month"] = frame[contract.snapshot_col].map(parse_snapshot_month)
    frame["_bucket"] = frame[contract.bucket_col].astype(str)
    frame["_balance"] = pd.to_numeric(frame[contract.balance_col], errors="coerce").fillna(0.0).astype(float)
    frame = frame[frame["_month"].notna()]
    months = sorted({str(month) for month in frame["_month"].tolist()})
    reference_month = months[-1] if months else None
    rows: list[MonthEL] = []
    for month in months:
        month_frame = frame[frame["_month"] == month]
        balance = float(month_frame["_balance"].sum())
        el = 0.0
        for bucket, group in month_frame.groupby("_bucket", sort=False):
            probability = p_to_loss.get(str(bucket), 0.0)
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
    return rows, total_el, len(months), reference_month


__all__ = [
    "SHORT_HISTORY_MIN_MONTHS",
    "ChainRow",
    "ExpectedLossResult",
    "MonthEL",
    "expected_loss_estimate",
]
