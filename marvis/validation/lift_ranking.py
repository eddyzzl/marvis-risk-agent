"""Deterministic lift / bin-ranking assessment for validation narratives.

Head is low-risk (good) and tail is high-risk (bad). Crossing 1.0 is not
enough: a head lift of 0.99 is still a weak discriminator.
"""

from __future__ import annotations

import math
from typing import Any

HEAD_LIFT_STRONG = 0.50
HEAD_LIFT_OK = 0.80
TAIL_LIFT_STRONG = 2.0
TAIL_LIFT_OK = 1.5
NEAR_ONE_LOWER = 0.85
NEAR_ONE_UPPER = 1.15
GUIDANCE = (
    "头部好、尾部坏。头部 lift 应明显低于 1，尾部应明显高于 1；"
    "0.99 虽小于 1 仍表示头部几乎没有风险压降。"
    "须看分箱逾期率/单组 lift 是否大致单调，以及头尾区分是否拉开。"
    "累计 lift 末行等于 1 是覆盖全部样本的数学结果，不得当作尾部失效或排序性好的证据。"
)


def assess_lift_ranking(effectiveness: dict[str, Any] | None) -> dict[str, Any]:
    payload = effectiveness if isinstance(effectiveness, dict) else {}
    overall_by_split = _rows_by_split(payload.get("overall"))
    aligned_tables = payload.get("bin_tables")
    independent_tables = payload.get("independent_quantile_bin_tables")
    splits: list[dict[str, Any]] = []
    for split in ("train", "test", "oot"):
        overall = overall_by_split.get(split) or {}
        head_5pct = _optional_float(overall.get("head_lift_5pct"))
        tail_5pct = _optional_float(overall.get("tail_lift_5pct"))
        splits.append(
            {
                "split": split,
                "head_lift_5pct": head_5pct,
                "tail_lift_5pct": tail_5pct,
                "head_5pct_strength": head_lift_strength(head_5pct),
                "tail_5pct_strength": tail_lift_strength(tail_5pct),
                "train_aligned": assess_bin_rows(
                    aligned_tables.get(split) if isinstance(aligned_tables, dict) else None,
                    source="train_aligned",
                ),
                "independent_quantile": assess_bin_rows(
                    independent_tables.get(split)
                    if isinstance(independent_tables, dict)
                    else None,
                    source="independent_quantile",
                ),
            }
        )
    return {
        "order": "good_to_bad",
        "guidance": GUIDANCE,
        "splits": splits,
    }


def assess_bin_rows(rows: object, *, source: str) -> dict[str, Any] | None:
    parsed = [_bin_metrics(row) for row in rows] if isinstance(rows, list) else []
    parsed = [row for row in parsed if row is not None]
    if not parsed:
        return None
    head = parsed[0]
    tail = parsed[-1]
    bad_rates = [row["bad_rate"] for row in parsed if row["bad_rate"] is not None]
    lifts = [row["lift"] for row in parsed if row["lift"] is not None]
    inversions = _increasing_inversions(bad_rates)
    return {
        "source": source,
        "bin_count": len(parsed),
        "head_group_lift": head["lift"],
        "tail_group_lift": tail["lift"],
        "head_group_bad_rate": head["bad_rate"],
        "tail_group_bad_rate": tail["bad_rate"],
        "head_group_strength": head_lift_strength(head["lift"]),
        "tail_group_strength": tail_lift_strength(tail["lift"]),
        "bad_rate_monotonicity": _monotonicity_label(inversions, len(bad_rates)),
        "bad_rate_inversions": inversions,
        "group_lift_spread": _ratio(tail["lift"], head["lift"]),
        "notes": _bin_notes(head["lift"], tail["lift"], lifts),
    }


def head_lift_strength(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 1:
        return "fail"
    if value > HEAD_LIFT_OK:
        return "weak"
    if value > HEAD_LIFT_STRONG:
        return "ok"
    return "strong"


def tail_lift_strength(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 1:
        return "fail"
    if value < TAIL_LIFT_OK:
        return "weak"
    if value < TAIL_LIFT_STRONG:
        return "ok"
    return "strong"


def _rows_by_split(rows: object) -> dict[str, dict[str, Any]]:
    by_split: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return by_split
    for row in rows:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split") or "").strip().lower()
        if split:
            by_split[split] = row
    return by_split


def _bin_metrics(row: object) -> dict[str, float | None] | None:
    if not isinstance(row, dict):
        return None
    return {
        "lift": _optional_float(row.get("lift")),
        "bad_rate": _optional_float(row.get("bad_rate")),
    }


def _increasing_inversions(values: list[float]) -> int:
    inversions = 0
    for left, right in zip(values, values[1:]):
        if right + 1e-12 < left:
            inversions += 1
    return inversions


def _monotonicity_label(inversions: int, count: int) -> str:
    if count < 2:
        return "unknown"
    if inversions == 0:
        return "monotonic"
    if inversions == 1:
        return "mostly_monotonic"
    return "broken"


def _bin_notes(
    head_lift: float | None,
    tail_lift: float | None,
    lifts: list[float],
) -> list[str]:
    notes: list[str] = []
    if head_lift is not None and NEAR_ONE_LOWER < head_lift < 1:
        notes.append("head_lift_near_one")
    if tail_lift is not None and 1 < tail_lift < NEAR_ONE_UPPER:
        notes.append("tail_lift_near_one")
    if head_lift is not None and head_lift >= 1:
        notes.append("head_not_better_than_average")
    if tail_lift is not None and tail_lift <= 1:
        notes.append("tail_not_worse_than_average")
    if len(lifts) >= 2 and max(lifts) - min(lifts) < 0.15:
        notes.append("flat_discrimination")
    return notes


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
