"""Paired amount coverage and rates shared by analysis and candidate replay.

Callers own input validation; missing values stay unavailable rather than zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def amount_metrics(
    mask: np.ndarray,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
) -> dict[str, Any]:
    loan = amount_measure(mask, loan_values, "loan_amount")
    overdue = amount_measure(mask, overdue_values, "overdue_amount")
    if loan_values is None or overdue_values is None:
        rate = {"status": "unavailable", "reason": "amount_column_not_configured"}
    else:
        paired = mask & np.isfinite(loan_values) & np.isfinite(overdue_values)
        if not np.any(mask):
            rate = {"status": "not_applicable", "reason": "empty_bin"}
        elif not np.any(paired):
            rate = {"status": "unavailable", "reason": "no_paired_amounts"}
        else:
            denominator = float(np.sum(loan_values[paired]))
            if denominator == 0:
                rate = {"status": "not_applicable", "reason": "zero_loan_amount"}
            else:
                rate = {
                    "status": "available",
                    "value": float(np.sum(overdue_values[paired]) / denominator),
                    "paired_count": int(np.sum(paired)),
                }
    return {"loan_amount": loan, "overdue_amount": overdue, "overdue_rate": rate}


def amount_measure(
    mask: np.ndarray, values: np.ndarray | None, name: str
) -> dict[str, Any]:
    if values is None:
        return {"status": "unavailable", "reason": f"{name}_not_configured"}
    covered = mask & np.isfinite(values)
    selected_count = int(np.sum(mask))
    if selected_count == 0:
        return {
            "status": "available",
            "sum": 0.0,
            "covered_count": 0,
            "coverage_rate": 1.0,
        }
    if not np.any(covered):
        return {
            "status": "unavailable",
            "reason": "no_covered_rows",
            "coverage_rate": 0.0,
        }
    return {
        "status": "available",
        "sum": float(np.sum(values[covered])),
        "covered_count": int(np.sum(covered)),
        "coverage_rate": float(np.sum(covered) / selected_count),
    }
