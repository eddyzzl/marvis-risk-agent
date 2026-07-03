from __future__ import annotations

from marvis.orchestrator.contracts import PlanStep


def is_safety_step(step: PlanStep) -> bool:
    if step.tool_ref.tool == "execute_join" or is_draft_run_step(step):
        return True
    if step.tool_ref.plugin == "strategy" and step.tool_ref.tool == "backtest_strategy":
        return False
    return any(check.kind == "range" for check in step.post_checks)


def is_draft_run_step(step: PlanStep) -> bool:
    return step.tool_ref.plugin == "drafts" and step.tool_ref.tool == "run_draft"
