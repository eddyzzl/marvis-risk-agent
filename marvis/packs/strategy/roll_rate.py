"""Roll-rate transition matrix (状态x时间长表口径).

roll_rate_matrix consumes a long "id x time x status" table and pairs each id's
adjacent observations in time order (see ``_adjacent_pairs``). This is a different
input shape from ``marvis.packs.analysis.flow`` (bucket_migration / flow_rate),
which aligns snapshot-month-indexed rows and explicitly tracks an "exited"
pseudo-state for missing next-month snapshots. The two are NOT interchangeable:

- roll_rate_matrix: any two chronologically adjacent observations for an id become
  a transition, even if months are skipped in between (see the missing-month
  warning below) -- there's no fixed snapshot cadence assumption.
- bucket_migration: assumes a monthly snapshot cadence and an explicit exited
  state; prefer it when your data is a true monthly panel and you need
  into_bad/out_of_bad net-flow or a worst-month-per-cell view.

Use roll_rate_matrix for irregular/ad-hoc status histories; use bucket_migration
for regular monthly snapshot panels. Both are deterministic and count-based by
default; both accept an optional balance-weighted transition ratio.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from marvis.packs.strategy.contracts import RollRateMatrix
from marvis.validation.vintage import RollRatePoint, compute_roll_rate

#: DOM-8: an id's two chronologically adjacent observations spanning more than this
#: many calendar months apart is flagged as a missing-month gap (informational only).
_MAX_ADJACENT_MONTH_GAP = 1


def roll_rate_matrix(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    status_col: str,
    states: list[str],
    balance_col: str | None = None,
) -> RollRateMatrix:
    """Compute the roll-rate transition matrix (see module docstring for the
    roll_rate_matrix vs. bucket_migration distinction).

    ``balance_col`` is optional: when supplied, each id's balance at the "from"
    observation weights its transition (deterministic sum, matching
    ``bucket_migration``'s balance-weighting convention); when omitted (default),
    every transition counts as 1 -- identical to the pre-existing behavior.
    """
    state_order = tuple(str(state) for state in states)
    if not state_order:
        raise ValueError("states must not be empty")
    required = [id_col, time_col, status_col]
    if balance_col:
        required.append(balance_col)
    _assert_columns(df, required)
    _assert_known_statuses(df, status_col=status_col, states=state_order)

    pairs, warnings = _adjacent_pairs(
        df, id_col=id_col, time_col=time_col, status_col=status_col, balance_col=balance_col,
    )
    if balance_col:
        matrix, base_counts = _balance_weighted_matrix(pairs, state_order)
    else:
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
        data_quality_warnings=warnings,
    )


def _adjacent_pairs(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    status_col: str,
    balance_col: str | None = None,
) -> tuple[pd.DataFrame, tuple[dict, ...]]:
    columns = [id_col, time_col, status_col] + ([balance_col] if balance_col else [])
    sorted_frame = df[columns].dropna(subset=[id_col, time_col, status_col]).copy()
    sorted_frame["_marvis_time_order"] = _parse_time_order(sorted_frame[time_col])
    sorted_frame = sorted_frame.sort_values([id_col, "_marvis_time_order", time_col], kind="mergesort")
    rows = []
    warnings: list[dict] = []
    for id_value, group in sorted_frame.groupby(id_col, sort=False):
        statuses = group[status_col].map(str).tolist()
        orders = group["_marvis_time_order"].tolist()
        balances = (
            pd.to_numeric(group[balance_col], errors="coerce").tolist() if balance_col else None
        )
        gap_months = 0
        for i in range(len(statuses) - 1):
            row = {"from": statuses[i], "to": statuses[i + 1]}
            if balance_col:
                row["balance"] = balances[i] if balances[i] == balances[i] else 0.0  # NaN-safe
            rows.append(row)
            months_apart = _month_delta(orders[i], orders[i + 1])
            if months_apart > _MAX_ADJACENT_MONTH_GAP:
                gap_months += months_apart - 1
        if gap_months:
            warnings.append({
                "code": "missing_month",
                "id": str(id_value),
                "gap_months": int(gap_months),
                "message": (
                    f"id={id_value} 的相邻观测跳过 {gap_months} 个月，roll-rate 转移可能"
                    "跨越缺失月份（不改变矩阵计算，仅提示口径风险）。"
                ),
            })
    columns_out = ["from", "to", "balance"] if balance_col else ["from", "to"]
    return pd.DataFrame(rows, columns=columns_out), tuple(warnings)


def _month_delta(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _parse_time_order(values: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="raise")
    try:
        return pd.Series([_parse_time_value(value) for value in values], index=values.index)
    except Exception as exc:
        raise ValueError("time_col must contain parseable dates or month labels") from exc


def _parse_time_value(value) -> pd.Timestamp:
    text = str(value).strip()
    if not text:
        raise ValueError("empty time value")
    if text.isdigit() and len(text) == 6:
        return pd.to_datetime(text, format="%Y%m", errors="raise")
    if text.isdigit() and len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d", errors="raise")
    return pd.to_datetime(text, errors="raise")


def _points_to_matrix(
    points: list[RollRatePoint],
    states: tuple[str, ...],
) -> tuple[tuple[tuple[float, ...], ...], dict[str, float]]:
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


def _balance_weighted_matrix(
    pairs: pd.DataFrame,
    states: tuple[str, ...],
) -> tuple[tuple[tuple[float, ...], ...], dict[str, float]]:
    # DOM-8: each transition weighted by the id's balance at the "from" observation,
    # mirroring bucket_migration's weight-sum-then-ratio convention (deterministic).
    weighted = {(from_state, to_state): 0.0 for from_state in states for to_state in states}
    base = dict.fromkeys(states, 0.0)
    for record in pairs.to_dict("records"):
        weight = float(record["balance"])
        weighted[(record["from"], record["to"])] += weight
        base[record["from"]] += weight
    matrix = tuple(
        tuple(
            (weighted[(from_state, to_state)] / base[from_state]) if base[from_state] else 0.0
            for to_state in states
        )
        for from_state in states
    )
    return matrix, base


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
