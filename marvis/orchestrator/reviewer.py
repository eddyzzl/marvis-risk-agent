from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import operator
import re
from typing import Any

from marvis.orchestrator.contracts import (
    Plan,
    PlanStep,
    PostCheck,
    ReviewVerdict,
    StepStatus,
)
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.schema_validation import validate_against_schema


CRITIC_SYS = (
    "You are MARVIS plan reviewer. Return JSON with passed and reasons. "
    "Do not change deterministic metrics."
)


@dataclass
class FinalReview:
    goal_met: bool
    summary: str
    open_items: list[str]
    goal_doubt: bool = False


class Reviewer:
    def __init__(self, llm_factory):
        self._llm_factory = llm_factory

    def deterministic_check(self, step: PlanStep, output: dict) -> ReviewVerdict:
        reasons = []
        for post_check in step.post_checks:
            ok, reason = _run_post_check(post_check, output, step)
            if not ok:
                reasons.append(reason)
        return ReviewVerdict(
            reviewer="deterministic",
            passed=not reasons,
            reasons=reasons,
            at=_now_iso(),
        )

    def llm_critique(self, step: PlanStep, output: dict, goal: str) -> ReviewVerdict:
        try:
            raw = self._llm_factory().complete(
                system_prompt=CRITIC_SYS,
                user_prompt=json.dumps(
                    {
                        "goal": goal,
                        "step": step.title,
                        "output_summary": _summarize_output(output),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                stream=False,
            )
            passed, reasons = _parse_soft_verdict(str(raw))
        except Exception as exc:
            passed, reasons = True, [f"llm critique unavailable: {exc}"]
        return ReviewVerdict(
            reviewer="llm_critic",
            passed=passed,
            reasons=reasons,
            at=_now_iso(),
        )

    def final_review(self, plan: Plan, outputs: dict[str, dict], goal: str) -> FinalReview:
        incomplete = [
            step.title
            for step in plan.steps
            if step.status not in {StepStatus.DONE, StepStatus.SKIPPED}
        ]
        summary, llm_items, goal_doubt = self._llm_summarize(goal, plan, outputs)
        return FinalReview(
            goal_met=not incomplete,
            summary=summary,
            open_items=incomplete + llm_items,
            goal_doubt=goal_doubt,
        )

    def _llm_summarize(
        self,
        goal: str,
        plan: Plan,
        outputs: dict[str, dict],
    ) -> tuple[str, list[str], bool]:
        try:
            raw = self._llm_factory().complete(
                system_prompt=CRITIC_SYS,
                user_prompt=json.dumps(
                    {
                        "goal": goal,
                        "plan_id": plan.id,
                        "step_count": len(plan.steps),
                        "outputs": _summarize_output(outputs),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                stream=False,
            )
            data = json.loads(str(raw))
        except Exception:
            return "Plan execution reviewed.", [], False
        if not isinstance(data, dict):
            return "Plan execution reviewed.", [], False
        summary = str(data.get("summary") or "Plan execution reviewed.")
        open_items = [
            str(item)
            for item in data.get("open_items") or []
            if isinstance(item, str)
        ]
        return summary, open_items, bool(data.get("goal_doubt", False))


def _run_post_check(pc: PostCheck, output: dict, step: PlanStep) -> tuple[bool, str]:
    if pc.kind == "schema":
        schema = pc.spec.get("schema", pc.spec)
        try:
            validate_against_schema(output, schema, label=step.title)
        except SchemaValidationError as exc:
            return False, str(exc)
        return True, ""
    if pc.kind == "range":
        field = str(pc.spec.get("field") or "")
        value = _dig(output, field)
        if value is None:
            return False, f"{field} missing"
        minimum = pc.spec.get("min")
        maximum = pc.spec.get("max")
        if minimum is not None and value < minimum:
            return False, f"{field}={value} < {minimum}"
        if maximum is not None and value > maximum:
            return False, f"{field}={value} > {maximum}"
        return True, ""
    if pc.kind == "rowcount":
        return _run_numeric_threshold(pc, output)
    if pc.kind == "invariant":
        return _run_invariant(str(pc.spec.get("rule") or ""), output)
    if pc.kind == "nonempty":
        field = str(pc.spec.get("field") or "")
        value = _dig(output, field)
        return bool(value), f"{field} empty" if not value else ""
    if pc.kind == "match_rate":
        field = str(pc.spec.get("field") or "match_rate")
        value = _dig(output, field)
        minimum = pc.spec.get("min")
        if value is None:
            return False, f"{field} missing"
        if minimum is not None and value < minimum:
            return False, f"{field} {value} < {minimum}"
        return True, ""
    return False, f"unknown post_check kind {pc.kind}"


def _run_numeric_threshold(pc: PostCheck, output: dict) -> tuple[bool, str]:
    field = str(pc.spec.get("field") or "rows")
    value = _dig(output, field)
    if value is None:
        return False, f"{field} missing"
    if "equals" in pc.spec and value != pc.spec["equals"]:
        return False, f"{field}={value} != {pc.spec['equals']}"
    if "min" in pc.spec and value < pc.spec["min"]:
        return False, f"{field}={value} < {pc.spec['min']}"
    if "max" in pc.spec and value > pc.spec["max"]:
        return False, f"{field}={value} > {pc.spec['max']}"
    return True, ""


def _run_invariant(rule: str, output: dict) -> tuple[bool, str]:
    match = re.fullmatch(r"\s*([\w.]+|-?\d+(?:\.\d+)?)\s*(<=|>=|<|>|==)\s*([\w.]+|-?\d+(?:\.\d+)?)\s*", rule)
    if match is None:
        return False, f"invalid invariant {rule}"
    left_raw, op_raw, right_raw = match.groups()
    left = _operand(left_raw, output)
    right = _operand(right_raw, output)
    if left is None or right is None:
        return False, f"invariant {rule} has missing operand"
    ok = _OPERATORS[op_raw](left, right)
    return ok, "" if ok else f"invariant failed: {rule}"


_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}


def _operand(raw: str, output: dict):
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return _dig(output, raw)


def _dig(value: dict, path: str):
    current: Any = value
    for part in path.split("."):
        if not part:
            return None
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _parse_soft_verdict(raw: str) -> tuple[bool, list[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return True, ["llm critique returned non-json"]
    if not isinstance(data, dict):
        return True, ["llm critique returned non-object"]
    reasons = [
        str(item)
        for item in data.get("reasons") or []
        if isinstance(item, str)
    ]
    return bool(data.get("passed", True)), reasons


def _summarize_output(output) -> dict:
    if not isinstance(output, dict):
        return {"type": type(output).__name__}
    summary = {}
    for key, value in output.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, list):
            summary[key] = {"type": "list", "count": len(value)}
        elif isinstance(value, dict):
            summary[key] = {"type": "object", "keys": sorted(value)[:10]}
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
