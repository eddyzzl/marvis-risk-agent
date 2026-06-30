"""Small shared helpers for inspecting orchestrator plans."""

from __future__ import annotations

from marvis.orchestrator.contracts import Plan, PlanStep


def find_step(plan: Plan, step_id: str) -> PlanStep | None:
    for step in plan.steps:
        if step.id == step_id:
            return step
    return None


def gate_depends_on_tool(plan: Plan, gate: PlanStep, tool: str) -> bool:
    for dep_id in gate.depends_on or []:
        dep = find_step(plan, dep_id)
        if dep is not None and dep.tool_ref.tool == tool:
            return True
    return False


def downstream_step_ids(plan: Plan, root_ids: list[str]) -> set[str]:
    roots = set(root_ids or [])
    downstream: set[str] = set()
    changed = True
    while changed:
        changed = False
        known_upstream = roots | downstream
        for step in plan.steps:
            if step.id in known_upstream:
                continue
            if any(dep_id in known_upstream for dep_id in step.depends_on):
                downstream.add(step.id)
                changed = True
    return downstream


__all__ = ["downstream_step_ids", "find_step", "gate_depends_on_tool"]
