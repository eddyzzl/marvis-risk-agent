"""Shared report/metric-table formatting helpers (T2-5).

Single home for the score-interval / compact-number / period / ratio / PSI-reference
formatters that were previously copied byte-for-byte across
``marvis.metric_tables``, ``marvis.output.excel`` and
``marvis.validation.effectiveness``. Every function is behaviour-preserving vs. the
previous copies for the inputs those call sites actually pass (real numbers / month
strings / None); the ``Any``-typed, None-tolerant shape here is the superset of the
three, so importing it changes no rendered number or string.
"""

from __future__ import annotations

from typing import Any


def compact_number(value: Any) -> str:
    """Compact numeric text: integers as ``"3"``, floats trimmed to <=3 decimals with
    trailing zeros/dot stripped, ``inf``/``-inf`` as ``"inf"``/``"-inf"``, and a ``"-"``
    fallback for values that are ``None`` or not float-parseable."""
    numeric = _to_float(value)
    if numeric is None:
        return "-"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def score_interval(lower: Any, upper: Any) -> str:
    return f"[{compact_number(lower)},{compact_number(upper)}]"


def period_text(start: Any, end: Any, *, default: str) -> str:
    start_text = str(start or "")
    end_text = str(end or "")
    if not start_text and not end_text:
        return default
    if not start_text:
        return end_text
    if not end_text:
        return start_text
    return start_text if start_text == end_text else f"{start_text}-{end_text}"


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def psi_reference_month_text(month: str, *, has_calendar_gap: bool) -> str:
    if not month or month == "-":
        return ""
    return f"{month}(跨月)" if has_calendar_gap else str(month)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "compact_number",
    "period_text",
    "psi_reference_month_text",
    "ratio",
    "score_interval",
]
