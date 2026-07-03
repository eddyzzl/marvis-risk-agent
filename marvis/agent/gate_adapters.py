"""Gate rendering adapters for PlanDriver.

This module bridges completed dependency outputs to the metadata/content needed
by an interactive gate. It deliberately has no repository or executor
dependency: callers provide a ``load_output(step_id)`` callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from marvis.agent.gate_payloads import (
    build_dedup_payload,
    build_model_delivery_payload,
    build_modeling_setup_payload,
    build_screen_payload,
)
from marvis.agent.modeling_red_flags import select_experiment_red_flags, tuning_setup_red_flags
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStep


@dataclass
class GateRenderResult:
    parts: list[str] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    output_refs: dict[str, str] = field(default_factory=dict)
    screen: dict | None = None
    dedup: dict | None = None
    modeling_setup: dict | None = None
    model_delivery: dict | None = None
    # AGT-9: deterministic red flags for the tuning-config / select-experiment
    # gates, computed straight from those gates' dependency outputs (never from
    # the rendered table strings). Empty list when nothing tripped, or when
    # neither red-flag family's inputs are present at this gate.
    red_flags: list[str] = field(default_factory=list)


def render_gate_dependencies(
    plan: Plan,
    gate: PlanStep | None,
    load_output: Callable[[str], Any],
) -> GateRenderResult:
    result = GateRenderResult()
    confirm_join_o: dict | None = None
    propose_join_o: dict | None = None
    split_o: dict | None = None
    modeling_spec_o: dict | None = None
    modeling_spec_step: PlanStep | None = None
    model_delivery_o: dict | None = None
    model_delivery_step: PlanStep | None = None
    report_o: dict | None = None
    report_step: PlanStep | None = None
    tune_o: dict | None = None
    train_models_o: dict | None = None
    for dep_id in gate.depends_on if gate else []:
        dep = _find_step(plan, dep_id)
        if dep is None:
            continue
        if dep.output_ref:
            result.output_refs[dep.id] = dep.output_ref
        output = load_output(dep_id)
        if output is None:
            continue
        text, dep_tables = render_tool_output(dep.tool_ref.tool, output)
        if text:
            result.parts.append(text)
        result.tables.extend(dep_tables)
        if dep.tool_ref.tool == "screen_features":
            result.screen = build_screen_payload(output, dep)
        elif dep.tool_ref.tool == "confirm_join":
            confirm_join_o = output
        elif dep.tool_ref.tool == "propose_join":
            propose_join_o = output
        elif dep.tool_ref.tool == "make_split" and isinstance(output, dict):
            split_o = output
        elif dep.tool_ref.tool == "choose_modeling_spec":
            modeling_spec_o = output if isinstance(output, dict) else None
            modeling_spec_step = dep
        elif dep.tool_ref.tool in {"compare_experiments", "select_experiment", "post_training_action"}:
            model_delivery_o = output if isinstance(output, dict) else None
            model_delivery_step = dep
        elif dep.tool_ref.tool == "generate_model_report":
            report_o = output if isinstance(output, dict) else None
            report_step = dep
        elif dep.tool_ref.tool == "tune_hyperparameters" and isinstance(output, dict):
            tune_o = output
        elif dep.tool_ref.tool == "train_models" and isinstance(output, dict):
            train_models_o = output
    if modeling_spec_o is not None and modeling_spec_step is not None:
        result.modeling_setup = build_modeling_setup_payload(
            modeling_spec_o,
            modeling_spec_step,
            split_output=split_o,
        )
    if model_delivery_o is not None and model_delivery_step is not None:
        result.model_delivery = build_model_delivery_payload(
            model_delivery_o,
            model_delivery_step,
            report_output=report_o,
            report_step=report_step,
        )
    result.dedup = build_dedup_payload(confirm_join_o, propose_join_o)
    result.red_flags = [
        *tuning_setup_red_flags(split_output=split_o, modeling_spec_output=modeling_spec_o),
        *select_experiment_red_flags(tune_output=tune_o, train_models_output=train_models_o),
    ]
    return result


def _find_step(plan: Plan, step_id: str) -> PlanStep | None:
    for step in plan.steps:
        if step.id == step_id:
            return step
    return None


__all__ = ["GateRenderResult", "render_gate_dependencies"]
