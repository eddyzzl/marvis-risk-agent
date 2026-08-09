"""Small presentation primitives shared across domain registries."""

from __future__ import annotations

from typing import Any


MONITOR_LEVEL_LABEL = {"green": "绿", "amber": "黄", "red": "红"}


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_percent(value: object) -> str:
    """Format a decimal ratio as a percentage; missing/invalid values are n/a."""

    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def format_number(value: object) -> str:
    return "n/a" if value is None else format_value(value)


def red_flag_table(red_flags: list[object]) -> dict[str, object]:
    return {
        "title": "红旗清单",
        "columns": ["等级", "code", "说明"],
        "rows": [
            [
                str(flag.get("level", "")),
                str(flag.get("code", "")),
                str(flag.get("message", "")),
            ]
            for flag in red_flags
            if isinstance(flag, dict)
        ],
    }


__all__ = [
    "MONITOR_LEVEL_LABEL",
    "format_number",
    "format_percent",
    "format_value",
    "red_flag_table",
]
