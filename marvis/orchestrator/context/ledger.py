from __future__ import annotations

import json

from marvis.orchestrator.contracts import Plan, StepStatus


def build_progress_ledger(
    plan: Plan,
    step_summaries: dict[str, dict],
    *,
    max_chars: int = 2000,
) -> str:
    lines = [f"目标: {plan.goal}"]
    for step in sorted(plan.steps, key=lambda item: (item.index, item.id)):
        if step.status == StepStatus.DONE:
            summary = json.dumps(
                step_summaries.get(step.id, {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            lines.append(f"[done] {step.title} -> {summary}")
        elif step.status == StepStatus.FAILED:
            lines.append(f"[failed] {step.title}: {step.error or ''}")
        elif step.status == StepStatus.SKIPPED:
            lines.append(f"[skipped] {step.title}")
    return _truncate("\n".join(lines), max_chars)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 20:
        return value[:max_chars]
    return value[: max_chars - 15] + "\n...[truncated]"
