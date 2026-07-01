"""Gate response validation for PlanDriver.

This module owns the gate-control contract: structured UI/AUTO controls must be
bound to the current awaiting gate, and each control family may only operate on
gates whose dependency outputs can actually be recomputed.
"""

from __future__ import annotations

from marvis.agent.adjust_specs import (
    has_modeling_setup_adjust,
    has_screen_adjust,
    has_tuning_adjust,
)
from marvis.agent.plan_utils import gate_depends_on_tool
from marvis.orchestrator.contracts import Plan, PlanStep


class GateControlValidationError(Exception):
    pass


_STRUCTURED_DEDUP_STRATEGIES = frozenset({"first", "last"})


def validate_gate_control(
    plan: Plan,
    gate: PlanStep | None,
    *,
    expected_step_id: str | None,
    selection,
    dedup_strategies,
    adjust_params,
) -> None:
    if expected_step_id:
        if gate is None or gate.id != str(expected_step_id):
            raise GateControlValidationError("当前待确认步骤已变化,请刷新后使用最新步骤的控件。")
    screen_adjust = has_screen_adjust(adjust_params)
    modeling_setup_adjust = has_modeling_setup_adjust(adjust_params)
    tuning_adjust = has_tuning_adjust(adjust_params)
    dedup_adjust = bool(dedup_strategies)
    if selection is None and not dedup_adjust and not screen_adjust and not modeling_setup_adjust and not tuning_adjust:
        return
    if gate is None:
        raise GateControlValidationError("当前没有待确认步骤,无法应用该控件。")
    if not expected_step_id:
        raise GateControlValidationError("该控件缺少待确认步骤校验信息,请刷新后重试。")
    if (selection is not None or screen_adjust) and not gate_depends_on_tool(plan, gate, "screen_features"):
        raise GateControlValidationError("该控件只适用于特征筛选确认步骤。")
    if dedup_adjust and not gate_depends_on_tool(plan, gate, "confirm_join"):
        raise GateControlValidationError("该控件只适用于拼接去重确认步骤。")
    if dedup_adjust:
        invalid = sorted({
            str(value)
            for value in (dedup_strategies or {}).values()
            if str(value).strip() not in _STRUCTURED_DEDUP_STRATEGIES
        })
        if invalid:
            raise GateControlValidationError(
                f"不支持的去重策略: {', '.join(invalid)};请使用 first 或 last。"
            )
    if modeling_setup_adjust and not gate_depends_on_tool(plan, gate, "choose_modeling_spec"):
        raise GateControlValidationError("该控件只适用于建模规格确认步骤。")
    if tuning_adjust and not (
        gate_depends_on_tool(plan, gate, "choose_modeling_spec")
        or gate_depends_on_tool(plan, gate, "tune_hyperparameters")
    ):
        raise GateControlValidationError("该控件只适用于建模规格或调参确认步骤。")


__all__ = ["GateControlValidationError", "validate_gate_control"]
