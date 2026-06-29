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
    build_modeling_setup_payload,
    build_screen_payload,
)
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


def render_gate_dependencies(
    plan: Plan,
    gate: PlanStep | None,
    load_output: Callable[[str], Any],
) -> GateRenderResult:
    result = GateRenderResult()
    confirm_join_o: dict | None = None
    propose_join_o: dict | None = None
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
        elif dep.tool_ref.tool == "choose_modeling_spec":
            result.modeling_setup = build_modeling_setup_payload(output, dep)
    result.dedup = build_dedup_payload(confirm_join_o, propose_join_o)
    return result


def _find_step(plan: Plan, step_id: str) -> PlanStep | None:
    for step in plan.steps:
        if step.id == step_id:
            return step
    return None


__all__ = ["GateRenderResult", "render_gate_dependencies"]
