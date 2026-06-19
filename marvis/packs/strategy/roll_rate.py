from __future__ import annotations

import pandas as pd

from marvis.packs.strategy.contracts import RollRateMatrix
from marvis.validation.vintage import RollRatePoint, compute_roll_rate


def roll_rate_matrix(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    status_col: str,
    states: list[str],
) -> RollRateMatrix:
    state_order = tuple(str(state) for state in states)
    if not state_order:
        raise ValueError("states must not be empty")
    _assert_columns(df, [id_col, time_col, status_col])
    _assert_known_statuses(df, status_col=status_col, states=state_order)

    pairs = _adjacent_pairs(df, id_col=id_col, time_col=time_col, status_col=status_col)
    points = compute_roll_rate(
        pairs,
        from_bucket_col="from",
        to_bucket_col="to",
        buckets=state_order,
    )
    matrix, base_counts = _points_to_matrix(points, state_order)
    return RollRateMatrix(
        states=state_order,
        matrix=matrix,
        period="month",
        base_counts=base_counts,
    )


def _adjacent_pairs(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    status_col: str,
) -> pd.DataFrame:
    sorted_frame = df[[id_col, time_col, status_col]].dropna(subset=[id_col, time_col, status_col]).sort_values(
        [id_col, time_col]
    )
    rows = []
    for _, group in sorted_frame.groupby(id_col, sort=False):
        statuses = group[status_col].map(str).tolist()
        rows.extend({"from": current, "to": next_status} for current, next_status in zip(statuses[:-1], statuses[1:]))
    return pd.DataFrame(rows, columns=["from", "to"])


def _points_to_matrix(
    points: list[RollRatePoint],
    states: tuple[str, ...],
) -> tuple[tuple[tuple[float, ...], ...], dict[str, int]]:
    by_pair = {(point.from_bucket, point.to_bucket): point for point in points}
    matrix = tuple(
        tuple(float(by_pair[(from_state, to_state)].rate) for to_state in states)
        for from_state in states
    )
    base_counts = {
        from_state: sum(int(by_pair[(from_state, to_state)].count) for to_state in states)
        for from_state in states
    }
    return matrix, base_counts


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")


def _assert_known_statuses(
    df: pd.DataFrame,
    *,
    status_col: str,
    states: tuple[str, ...],
) -> None:
    valid = set(states)
    observed = df[status_col].dropna().map(str)
    unknown = sorted(set(observed) - valid)
    if unknown:
        raise ValueError(f"unknown status: {', '.join(unknown)}")


__all__ = ["roll_rate_matrix"]
