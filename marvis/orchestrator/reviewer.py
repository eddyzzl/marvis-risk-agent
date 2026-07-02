from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import operator
import re
from typing import Any

from marvis.agent.json_reply import load_json_object
from marvis.orchestrator.contracts import (
    Plan,
    PlanStep,
    PostCheck,
    ReviewVerdict,
    StepStatus,
)
from marvis.orchestrator.validator import METRIC_FIELDS
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
    llm_goal_met: bool | None = None


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
            prompt = json.dumps(
                {
                    "goal": goal,
                    "step": step.title,
                    "output_summary": _summarize_output(output),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            raw = self._llm_factory().complete(
                system_prompt=CRITIC_SYS,
                user_prompt=prompt,
                stream=False,
            )
            passed, reasons, ok = _parse_soft_verdict(raw)
            if not ok:
                raw = self._llm_factory().complete(
                    system_prompt=CRITIC_SYS,
                    user_prompt=_retry_json_prompt(
                        prompt,
                        raw,
                        '{"passed": true|false, "reasons": ["..."]}',
                    ),
                    stream=False,
                )
                passed, reasons, _ok = _parse_soft_verdict(raw)
        except Exception as exc:
            passed, reasons = False, [f"llm critique unavailable: {exc}"]
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
        criteria_failures = _evaluate_success_criteria(plan.success_criteria, outputs)
        summary, llm_items, goal_doubt, llm_goal_met = self._llm_summarize(goal, plan, outputs)
        if criteria_failures:
            summary = f"{summary} 成功标准未达成: {'; '.join(criteria_failures)}"
        if llm_goal_met is False and not llm_items:
            llm_items = ["LLM final review marked goal_met=false"]
        # AGT-3: narrow the LLM's authority. A weak model saying goal_met=false on a
        # plan where every step passed its deterministic post_checks (no incomplete
        # steps) and no configured success_criteria failed is treated as *doubt*, not
        # a veto — it routes to REVIEW (human re-check, executor.py already has this
        # channel) instead of FAILED or an automatic "fill in remaining steps" replan.
        # Only deterministic success_criteria failures — or genuinely incomplete
        # steps — may still trigger FAILED / replan; the LLM's opinion can pause a
        # plan but never fail one outright (INV-1: the platform, not the LLM,
        # computes truth).
        if llm_goal_met is False and not incomplete and not criteria_failures:
            goal_doubt = True
        return FinalReview(
            goal_met=not incomplete and not criteria_failures and not goal_doubt and llm_goal_met is not False,
            summary=summary,
            open_items=incomplete + criteria_failures + llm_items,
            goal_doubt=goal_doubt,
            llm_goal_met=llm_goal_met,
        )

    def _llm_summarize(
        self,
        goal: str,
        plan: Plan,
        outputs: dict[str, dict],
    ) -> tuple[str, list[str], bool, bool | None]:
        try:
            prompt = json.dumps(
                {
                    "goal": goal,
                    "plan_id": plan.id,
                    "step_count": len(plan.steps),
                    "outputs": _summarize_output(outputs),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            raw = self._llm_factory().complete(
                system_prompt=CRITIC_SYS,
                user_prompt=prompt,
                stream=False,
            )
            data, error = load_json_object(raw)
            if data is None:
                raw = self._llm_factory().complete(
                    system_prompt=CRITIC_SYS,
                    user_prompt=_retry_json_prompt(
                        prompt,
                        raw,
                        '{"summary": "...", "open_items": [], "goal_doubt": false, "goal_met": true|false}',
                    ),
                    stream=False,
                )
                data, error = load_json_object(raw)
        except Exception:
            return "Plan execution reviewed.", [], False, None
        if not isinstance(data, dict) or error is not None:
            return "Plan execution reviewed.", [], False, None
        summary = str(data.get("summary") or "Plan execution reviewed.")
        open_items = [
            str(item)
            for item in data.get("open_items") or []
            if isinstance(item, str)
        ]
        raw_goal_met = data.get("goal_met")
        llm_goal_met = raw_goal_met if isinstance(raw_goal_met, bool) else None
        return summary, open_items, bool(data.get("goal_doubt", False)), llm_goal_met


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
            if pc.spec.get("allow_null") is True:
                return True, ""
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
    if pc.kind == "one_of":
        field = str(pc.spec.get("field") or "")
        value = _dig(output, field)
        allowed = pc.spec.get("values") or []
        return value in allowed, f"{field}={value} not in {allowed}" if value not in allowed else ""
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
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list | tuple) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def _retry_json_prompt(original_prompt: str, raw_reply, expected_shape: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Previous reply was not parseable JSON:\n{raw_reply}\n\n"
        f"Return only a JSON object matching this shape: {expected_shape}"
    )


def _parse_soft_verdict(raw) -> tuple[bool, list[str], bool]:
    data, error = load_json_object(raw)
    if data is None:
        return False, ["llm critique returned non-json"], False
    if not isinstance(data, dict):
        return False, ["llm critique returned non-object"], False
    reasons = [
        str(item)
        for item in data.get("reasons") or []
        if isinstance(item, str)
    ]
    return bool(data.get("passed", True)), reasons, error is None


# AGT-3: final_review/llm_critique previously saw only key names (dict -> {"type":
# "object", "keys": [...10 names...]}), so a step's train/test/oot KS/AUC never
# reached the LLM at all — it could only guess from field names. Keep real numeric
# values one level deeper (depth 2) for entries that are themselves metrics (key in
# METRIC_FIELDS) or plain numbers, so the critic/final-review prompts actually see
# the platform-computed metrics instead of just their names. Bounded to 20 keys /
# 600 characters per nested object so a wide experiments table can't blow the
# prompt budget.
_SUMMARY_MAX_KEYS = 20
_SUMMARY_MAX_CHARS = 600


def _summarize_output(output, _depth: int = 0) -> dict:
    if not isinstance(output, dict):
        return {"type": type(output).__name__}
    summary = {}
    for key, value in output.items():
        if isinstance(value, bool) or value is None:
            summary[key] = value
        elif isinstance(value, (int, float)):
            summary[key] = value
        elif isinstance(value, str):
            summary[key] = value
        elif isinstance(value, list):
            summary[key] = _summarize_list(value, _depth)
        elif isinstance(value, dict):
            summary[key] = _summarize_nested_dict(value, _depth)
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def _summarize_nested_dict(value: dict, depth: int) -> dict:
    if depth >= 2:
        # Depth 2 is as deep as we recurse with real values (outputs -> step ->
        # metrics is exactly 2 dict layers); beyond that, fall back to the
        # original key-name-only shape to keep the summary bounded.
        return _bounded_keys_summary(value)
    if _has_metric_values(value):
        return _bounded_metric_summary(value, depth)
    return _summarize_output(value, depth + 1)


def _summarize_list(value: list, depth: int) -> dict:
    if not value:
        return {"type": "list", "count": len(value)}
    # A list of metric dicts (e.g. experiments: [{"metrics": {...}}, ...]) is a
    # common modeling-step shape; summarizing each element preserves the numbers
    # instead of collapsing the whole list to a bare count. Depth is not
    # advanced here — the dicts inside the list are the object of interest, not
    # an extra nesting layer to budget against.
    if all(isinstance(item, dict) for item in value):
        return {
            "type": "list",
            "count": len(value),
            "items": [_summarize_output(item, depth) for item in value[:5]],
        }
    return {"type": "list", "count": len(value)}


def _has_metric_values(value: dict) -> bool:
    """True when this dict is worth keeping real numbers for: any key is a known
    metric name (METRIC_FIELDS, e.g. "ks") or a numeric leaf whose name carries a
    metric field as a token (e.g. "oot_ks", "test_auc" — the platform's actual
    train/test/oot-prefixed naming convention), or any value is simply numeric."""
    for key, item in value.items():
        name = str(key)
        if name in METRIC_FIELDS or any(part in METRIC_FIELDS for part in name.split("_")):
            return True
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return True
    return False


def _bounded_metric_summary(value: dict, depth: int) -> dict:
    bounded = {}
    used_chars = 0
    for key in sorted(value)[:_SUMMARY_MAX_KEYS]:
        item = value[key]
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            bounded[key] = item
            used_chars += len(f"{key}={item}")
        elif isinstance(item, bool) or item is None:
            bounded[key] = item
        elif isinstance(item, str):
            bounded[key] = item
            used_chars += len(f"{key}={item}")
        elif isinstance(item, dict) and depth + 1 < 2:
            bounded[key] = _summarize_nested_dict(item, depth + 1)
        else:
            bounded[key] = {"type": type(item).__name__}
        if used_chars > _SUMMARY_MAX_CHARS:
            break
    return {"type": "object", "metrics": bounded}


def _bounded_keys_summary(value: dict) -> dict:
    return {"type": "object", "keys": sorted(value)[:10]}


def _evaluate_success_criteria(
    criteria: list[dict[str, Any]],
    outputs: dict[str, dict],
) -> list[str]:
    failures: list[str] = []
    observed_target_type = _first_metric_value(outputs, "target_type")
    for criterion in criteria or []:
        if not isinstance(criterion, dict):
            continue
        target_type = str(criterion.get("target_type") or "").strip()
        if (
            target_type
            and observed_target_type is not None
            and str(observed_target_type) != target_type
        ):
            continue
        metric = str(criterion.get("metric") or criterion.get("field") or "").strip()
        if not metric:
            continue
        values = _numeric_metric_values(outputs, metric)
        label = str(criterion.get("label") or metric)
        if not values:
            failures.append(f"{label} missing for success criterion")
            continue
        aggregate = str(criterion.get("aggregate") or "max").lower()
        value = min(values) if aggregate == "min" else max(values)
        minimum, min_error = _coerce_threshold(criterion.get("min"), label, "min")
        maximum, max_error = _coerce_threshold(criterion.get("max"), label, "max")
        if min_error or max_error:
            failures.extend(item for item in (min_error, max_error) if item)
            continue
        if minimum is not None and value < minimum:
            failures.append(f"{label}={value:.4g} < {minimum:.4g}")
        if maximum is not None and value > maximum:
            failures.append(f"{label}={value:.4g} > {maximum:.4g}")
    return failures


def _coerce_threshold(
    raw_value: Any,
    label: str,
    threshold_name: str,
) -> tuple[float | None, str | None]:
    if raw_value is None:
        return None, None
    try:
        return float(raw_value), None
    except (TypeError, ValueError):
        return None, f"{label} invalid {threshold_name} threshold: {raw_value!r}"


def _numeric_metric_values(value, metric: str) -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == metric and isinstance(item, (int, float)) and not isinstance(item, bool):
                values.append(float(item))
            else:
                values.extend(_numeric_metric_values(item, metric))
    elif isinstance(value, list | tuple):
        for item in value:
            values.extend(_numeric_metric_values(item, metric))
    return values


def _first_metric_value(value, metric: str):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == metric:
                return item
            found = _first_metric_value(item, metric)
            if found is not None:
                return found
    elif isinstance(value, list | tuple):
        for item in value:
            found = _first_metric_value(item, metric)
            if found is not None:
                return found
    return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
