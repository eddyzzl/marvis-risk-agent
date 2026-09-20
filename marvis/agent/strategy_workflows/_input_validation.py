"""Pure input checks shared by migrated strategy workflows.

Field-error metadata is a caller contract: use the explicit metadata variant
for tree and pool workflows and the text-only variant for older workflows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .contracts import StrategyWorkflowValidationError


def reject_fields(
    inputs: Mapping[str, Any], allowed: set[str], *, workflow: str
) -> None:
    """Preserve the text-only error contract of cross and scorecard workflows."""
    _reject_fields(inputs, allowed, workflow=workflow, include_fields=False)


def reject_fields_with_metadata(
    inputs: Mapping[str, Any], allowed: set[str], *, workflow: str
) -> None:
    """Preserve field-addressable errors required by tree and pool workflows."""
    _reject_fields(inputs, allowed, workflow=workflow, include_fields=True)


def _reject_fields(
    inputs: Mapping[str, Any],
    allowed: set[str],
    *,
    workflow: str,
    include_fields: bool,
) -> None:
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyWorkflowValidationError(
            f"{workflow} workflow_inputs 字段名必须是文本。"
        )
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{workflow} workflow_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。",
            fields=unexpected if include_fields else (),
        )


def required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyWorkflowValidationError(f"{name} 必须是非空文本。")
    return value.strip()


def column(
    value: object,
    *,
    name: str,
    whitelist: Sequence[str],
) -> str:
    column = required_text(value, name=name)
    if column not in whitelist:
        raise StrategyWorkflowValidationError(
            f"{name} 使用了数据集中不存在的列「{column}」。"
        )
    return column


def bounded_number(
    value: object,
    *,
    name: str,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StrategyWorkflowValidationError(f"{name} 必须是有限数字。")
    number = float(value)
    if (
        not math.isfinite(number)
        or number < 0
        or (maximum is not None and number > maximum)
    ):
        if maximum is None:
            raise StrategyWorkflowValidationError(
                f"{name} 必须是大于等于 0 的有限数字。"
            )
        raise StrategyWorkflowValidationError(
            f"{name} 必须是 0 到 {maximum:g} 之间的有限数字。"
        )
    return number
