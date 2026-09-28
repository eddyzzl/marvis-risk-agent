from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import operator
import re
from typing import Any

from marvis.agent.json_reply import load_json_object, rejects_positive_decision
from marvis.llm_prompts import CRITIC_SYS as _CRITIC_SYS_SPEC
from marvis.llm_settings import LLMSettingsError
from marvis.orchestrator.contracts import (
    Plan,
    PlanStep,
    PostCheck,
    ReviewVerdict,
    StepStatus,
)
from marvis.orchestrator.validator import METRIC_FIELDS
from marvis.orchestrator.business_acceptance import business_review_binding, review_business_acceptance
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.manifest import ToolRef
from marvis.plugins.schema_validation import validate_against_schema


# LLM-10: text/version now live in marvis.llm_prompts; kept as a module-level
# constant so existing imports of CRITIC_SYS from here keep working unchanged.
CRITIC_SYS = _CRITIC_SYS_SPEC.text

# Explicit contracts are sent on the first request, including JSON-object-only
# providers. The response parser still validates types locally.
_CRITIQUE_SCHEMA = {
    "name": "step_critique",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["passed", "reasons"],
        "additionalProperties": False,
    },
}
_NARRATIVE_SCHEMA = {
    "name": "plan_review_summary",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 1},
            "open_items": {"type": "array", "items": {"type": "string"}},
            "goal_doubt": {"type": "boolean"},
            "goal_met": {"type": ["boolean", "null"]},
        },
        # Keep legacy minimal summaries valid; omitted opinion is unknown.
        "required": ["summary"],
        "additionalProperties": False,
    },
}


@dataclass
class FinalReview:
    goal_met: bool
    summary: str
    open_items: list[str]
    goal_doubt: bool = False
    llm_goal_met: bool | None = None
    # None identifies pre-contract summaries and keeps their recovery semantics.
    execution_completed: bool | None = None
    business_acceptance: dict | None = None
    explanation_status: str = "unavailable"
    explanation_items: list[str] | None = None


class Reviewer:
    def __init__(self, llm_factory, *, plan_repository=None):
        self._llm_factory = llm_factory
        self._plan_repository = plan_repository

    def deterministic_check(self, step: PlanStep, output: dict) -> ReviewVerdict:
        reasons = []
        for post_check in step.post_checks:
            # Compatibility for plans persisted before screen_features stopped
            # treating an empty recommendation set as an execution error. The
            # gate must render the metrics/reasons so the user can review or
            # repair the data instead of being sent to an internal $ref editor.
            if _is_reviewable_empty_screen_check(step, post_check):
                continue
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
        # AGT-6: no LLM configured is the common manual-mode case, not a failure —
        # every step used to deterministically render a "failed" llm_critic verdict
        # (a full English exception message) purely because settings had no model
        # enabled. Skip cleanly instead: passed=True, status="skipped", so manual
        # mode doesn't drown real deterministic failures in critique noise.
        try:
            llm = self._llm_factory()
        except LLMSettingsError:
            return ReviewVerdict(
                reviewer="llm_critic",
                passed=True,
                reasons=["skipped: no LLM configured"],
                at=_now_iso(),
                status="skipped",
            )
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
            raw = llm.complete(
                system_prompt=CRITIC_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                json_schema=_CRITIQUE_SCHEMA,
                caller="critic",
                prompt_name=_CRITIC_SYS_SPEC.name,
                prompt_version=_CRITIC_SYS_SPEC.version,
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
                    response_format={"type": "json_object"},
                    json_schema=_CRITIQUE_SCHEMA,
                    caller="critic",
                    prompt_name=_CRITIC_SYS_SPEC.name,
                    prompt_version=_CRITIC_SYS_SPEC.version,
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
        business = review_business_acceptance(plan, self._plan_repository)
        business["execution_binding"] = business_review_binding(plan)
        summary, llm_items, _goal_doubt, llm_goal_met = self._llm_summarize(goal, plan, outputs, business)
        return FinalReview(
            goal_met=not incomplete and business["status"] == "passed",
            summary=summary,
            open_items=incomplete + business["reasons"],
            goal_doubt=False,
            llm_goal_met=llm_goal_met,
            execution_completed=not incomplete,
            business_acceptance=business,
            explanation_status="available" if summary != "Plan execution reviewed." else "unavailable",
            explanation_items=llm_items,
        )

    def _llm_summarize(
        self,
        goal: str,
        plan: Plan,
        outputs: dict[str, dict],
        business_acceptance: dict,
    ) -> tuple[str, list[str], bool, bool | None]:
        try:
            prompt = json.dumps(
                {
                    "goal": goal,
                    "plan_id": plan.id,
                    "step_count": len(plan.steps),
                    "outputs": _summarize_output(outputs),
                    "business_acceptance": _business_acceptance_prompt(business_acceptance),
                    "acceptance_authority": (
                        "业务验收由平台确定性判定。解释必须遵守此status、objective_hash和采用对象，"
                        "不得用候选指标改写结论；流程完成不代表业务达标。"
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            raw = self._llm_factory().complete(
                system_prompt=CRITIC_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                json_schema=_NARRATIVE_SCHEMA,
                caller="reviewer_summary",
                prompt_name=_CRITIC_SYS_SPEC.name,
                prompt_version=_CRITIC_SYS_SPEC.version,
                stream=False,
            )
            data = _parse_narrative(raw)
            if data is None:
                raw = self._llm_factory().complete(
                    system_prompt=CRITIC_SYS,
                    user_prompt=_retry_json_prompt(
                        prompt,
                        raw,
                        '{"summary": "...", "open_items": [], "goal_doubt": false, "goal_met": true|false}',
                    ),
                    response_format={"type": "json_object"},
                    json_schema=_NARRATIVE_SCHEMA,
                    caller="reviewer_summary",
                    prompt_name=_CRITIC_SYS_SPEC.name,
                    prompt_version=_CRITIC_SYS_SPEC.version,
                    stream=False,
                )
                data = _parse_narrative(raw)
        except Exception:
            return "Plan execution reviewed.", [], False, None
        if data is None:
            return "Plan execution reviewed.", [], False, None
        summary = data["summary"]
        open_items = data.get("open_items", [])
        raw_goal_met = data.get("goal_met")
        llm_goal_met = raw_goal_met if isinstance(raw_goal_met, bool) else None
        return summary, open_items, data.get("goal_doubt", False), llm_goal_met


def _is_reviewable_empty_screen_check(step: PlanStep, post_check: PostCheck) -> bool:
    return (
        step.tool_ref == ToolRef("modeling", "screen_features")
        and post_check.kind == "nonempty"
        and str(post_check.spec.get("field") or "") == "selected"
    )


def _business_acceptance_prompt(result: dict) -> dict:
    """Bound the explanatory context; never send the full business contract."""
    target = result.get("target") or {}
    evidence = result.get("evidence") or {}
    return {
        "status": result["status"], "objective_hash": result.get("objective_hash"),
        "target": {key: str(value)[:200] for key, value in target.items()},
        "evidence": {key: str(evidence[key])[:200] for key in ("source_ref", "source_hash", "effect_stage", "period_start", "period_end", "labels_mature") if key in evidence},
        "criteria_count": len(result["criteria"]),
        "criteria": [{"metric": str(item["metric"])[:100], "value": item.get("value"), "status": item["status"]} for item in result["criteria"][:20]],
        "reasons": [str(reason)[:300] for reason in result["reasons"][:6]],
    }


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
        f"Previous reply was not parseable JSON or did not match the output contract:\n{raw_reply}\n\n"
        f"Return only a JSON object matching this shape: {expected_shape}"
    )


def _parse_soft_verdict(raw) -> tuple[bool, list[str], bool]:
    data, error = load_json_object(raw)
    if data is None:
        return False, ["llm critique returned non-json"], False
    try:
        validate_against_schema(data, _CRITIQUE_SCHEMA["schema"], label="step critique")
    except SchemaValidationError:
        return False, ["llm critique returned invalid schema"], False
    reasons = data["reasons"]
    passed = data["passed"]
    if passed and any(rejects_positive_decision(reason) for reason in reasons):
        passed = False
    return passed, reasons, error is None


def _parse_narrative(raw) -> dict | None:
    data, error = load_json_object(raw)
    if data is None or error is not None:
        return None
    try:
        validate_against_schema(data, _NARRATIVE_SCHEMA["schema"], label="review summary")
    except SchemaValidationError:
        return None
    return data if data["summary"].strip() else None


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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
