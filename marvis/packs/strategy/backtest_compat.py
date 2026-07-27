from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypeAlias

from marvis.packs.strategy.contracts import BacktestResult
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.typed_backtest import StrategyBacktestResult


BacktestRecord: TypeAlias = BacktestResult | StrategyBacktestResult


def backtest_record_payload(result: BacktestRecord) -> dict[str, Any]:
    """Return the canonical persisted payload for either backtest generation."""

    if isinstance(result, StrategyBacktestResult):
        return result.to_dict()
    payload = asdict(result)
    payload["by_segment"] = [dict(row) for row in result.by_segment]
    return payload


def approval_backtest_projection(
    result: BacktestRecord,
    *,
    preserve_undefined_rates: bool = False,
) -> dict[str, Any]:
    """Project typed approval evidence onto the historical flat Tool contract.

    The typed envelope remains authoritative.  This adapter exists only so stored
    plans, adoption, monitoring and external callers can migrate without treating
    limit/pricing/segmentation results as approval results.
    """

    if isinstance(result, BacktestResult):
        return backtest_record_payload(result)
    if result.strategy_type not in {"approval", "reject"}:
        raise StrategyError(
            "approval compatibility fields are not defined for "
            f"strategy type {result.strategy_type}"
        )

    metrics = result.metrics
    economics = result.economics
    swap_in = _transition_group(
        result.transitions,
        lambda row: row.get("from_action") != "approve"
        and row.get("to_action") == "approve",
    )
    swap_out = _transition_group(
        result.transitions,
        lambda row: row.get("from_action") == "approve"
        and row.get("to_action") != "approve",
    )
    return {
        "strategy_id": result.strategy_id,
        "approval_rate": metrics.get("approve_rate"),
        "approved_count": metrics.get("approve_count"),
        # Legacy callers historically represented an empty group as 0.0.  The
        # canonical metrics retain None, so this compatibility-only zero cannot
        # contaminate deterministic V2 evidence.
        "approved_bad_rate": _project_rate(
            metrics.get("approve_bad_rate"),
            preserve_undefined=preserve_undefined_rates,
        ),
        "rejected_bad_rate": _project_rate(
            metrics.get("reject_bad_rate"),
            preserve_undefined=preserve_undefined_rates,
        ),
        "expected_profit": (
            economics.get("expected_profit")
            if preserve_undefined_rates or "expected_profit" in economics
            else 0.0
        ),
        "swap_in_count": swap_in["count"],
        "swap_out_count": swap_out["count"],
        "swap_in_bad_rate": swap_in["bad_rate"],
        "swap_out_bad_rate": swap_out["bad_rate"],
        "by_segment": [
            {
                "decision": row.get("action"),
                "count": row.get("count"),
                "bad_count": row.get("bad_count"),
                "bad_rate": _project_rate(
                    row.get("bad_rate"),
                    preserve_undefined=preserve_undefined_rates,
                ),
            }
            for row in result.breakdown
        ],
        "profit_note": economics.get("profit_note"),
        "rejected_count": metrics.get("reject_count"),
        "review_count": metrics.get("review_count"),
        "review_rate": metrics.get("review_rate"),
        "review_bad_rate": metrics.get("review_bad_rate"),
    }


def _transition_group(transitions, predicate) -> dict[str, Any]:
    selected = [row for row in transitions if predicate(row)]
    count = sum(int(row.get("count") or 0) for row in selected)
    labeled_count = sum(int(row.get("labeled_count") or 0) for row in selected)
    bad_count = sum(int(row.get("bad_count") or 0) for row in selected)
    return {
        "count": count,
        "bad_rate": None if labeled_count == 0 else float(bad_count / labeled_count),
    }


def _project_rate(value: object, *, preserve_undefined: bool) -> object:
    return value if preserve_undefined or value is not None else 0.0


__all__ = [
    "BacktestRecord",
    "approval_backtest_projection",
    "backtest_record_payload",
]
