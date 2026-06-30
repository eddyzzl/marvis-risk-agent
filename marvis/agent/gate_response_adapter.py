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
from marvis.orchestrator.contracts import Plan, PlanStep


class GateControlValidationError(Exception):
    pass


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
    if (selection is not None or screen_adjust) and not _gate_depends_on_tool(plan, gate, "screen_features"):
        raise GateControlValidationError("该控件只适用于特征筛选确认步骤。")
    if dedup_adjust and not _gate_depends_on_tool(plan, gate, "confirm_join"):
        raise GateControlValidationError("该控件只适用于拼接去重确认步骤。")
    if modeling_setup_adjust and not _gate_depends_on_tool(plan, gate, "choose_modeling_spec"):
        raise GateControlValidationError("该控件只适用于建模规格确认步骤。")
    if tuning_adjust and not (
        _gate_depends_on_tool(plan, gate, "choose_modeling_spec")
        or _gate_depends_on_tool(plan, gate, "tune_hyperparameters")
    ):
        raise GateControlValidationError("该控件只适用于建模规格或调参确认步骤。")


def _gate_depends_on_tool(plan: Plan, gate: PlanStep, tool: str) -> bool:
    for dep_id in gate.depends_on or []:
        dep = _find_step(plan, dep_id)
        if dep is not None and dep.tool_ref.tool == tool:
            return True
    return False


def _find_step(plan: Plan, step_id: str) -> PlanStep | None:
    for step in plan.steps:
        if step.id == step_id:
            return step
    return None


__all__ = ["GateControlValidationError", "validate_gate_control"]
