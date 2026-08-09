from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
import math
import re
from typing import Any
import uuid

from marvis.agent.json_reply import load_json_object
from marvis.llm_client import DEFAULT_CONTEXT_WINDOW, estimate_tokens
from marvis.llm_prompts import EXPLORE_SYS as _EXPLORE_SYS_SPEC
from marvis.llm_prompts import PLAN_SYS as _PLAN_SYS_SPEC
from marvis.llm_prompts import REPLAN_SYS as _REPLAN_SYS_SPEC
from marvis.orchestrator.capability import CapabilityTier, resolve_tier
from marvis.orchestrator.context.budget import fit_to_budget
from marvis.orchestrator.context.ledger import build_progress_ledger
from marvis.orchestrator.contracts import Plan, PlanStep, PostCheck, StepStatus
from marvis.orchestrator.templates import UNSET_SLOT_DEFAULT, WorkflowTemplate
from marvis.plugins.errors import ManifestError, PluginNotFoundError, ToolNotFoundError
from marvis.plugins.manifest import (
    GovernancePolicy,
    ToolRef,
    merge_governance_policies,
)

logger = logging.getLogger(__name__)


# LLM-10: prompt text/versions now live in marvis.llm_prompts; these module-level
# constants are kept so existing call sites and tests that import PLAN_SYS /
# REPLAN_SYS / EXPLORE_SYS from here keep working unchanged.
PLAN_SYS = _PLAN_SYS_SPEC.text
_OMIT = object()
REPLAN_SYS = _REPLAN_SYS_SPEC.text
EXPLORE_SYS = _EXPLORE_SYS_SPEC.text
MAX_REPLAN_PARSE_RETRY = 2
# Reasoning-capable OpenAI-compatible models count hidden reasoning inside the
# completion budget.  A 4096-token ceiling can therefore end at ``length``
# before a single byte of the required compact JSON is emitted.  Keep bounded
# headroom for reasoning plus the final plan; the client still enforces the
# configured context window before sending the request.
PLANNER_MAX_OUTPUT_TOKENS = 16384
PLANNER_MIN_OUTPUT_TOKENS = 4096
PLANNER_CONTEXT_SAFETY_MARGIN_TOKENS = 512
MAX_CATALOG_FIELDS = 12
MAX_REPLAN_LOG_MESSAGE_CHARS = 500

PLANNING_EXAMPLES = (
    {
        "purpose": "Call a first tool with literal inputs.",
        "step": {
            "id": "profile-step",
            "title": "读取数据概况",
            "tool": {"plugin": "data_ops", "tool": "profile_dataset"},
            "inputs": {
                "dataset_id": "dataset-1",
                "expected_content_hash": "a" * 64,
                "workspace_revision": 1,
                "analysis_generation": 1,
                "semantic_mapping_hash": "b" * 64,
            },
            "depends_on": [],
            "post_checks": [
                {"kind": "rowcount", "spec": {"field": "row_count", "min": 0}},
                {
                    "kind": "invariant",
                    "spec": {"rule": "dataset_content_hash==expected_content_hash"},
                },
            ],
        },
    },
    {
        "purpose": "Reference a previous tool output; depends_on must include that step id.",
        "step": {
            "id": "report-step",
            "title": "生成报告",
            "tool": {"plugin": "report", "tool": "generate"},
            "inputs": {"experiment_id": "$ref:train-step.output.experiment_id"},
            "depends_on": ["train-step"],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "report_path"}}],
        },
    },
)

PLAN_STEPS_SCHEMA = {
    "name": "plan_steps",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "done": {"type": "boolean"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "tool": {"type": "object"},
                        "inputs": {"type": "object"},
                        "depends_on": {"type": "array"},
                    },
                    "required": ["title", "tool"],
                    "additionalProperties": True,
                },
            },
        },
        "required": ["steps"],
        "additionalProperties": True,
    },
}


@dataclass(frozen=True)
class RequiredLiteralInput:
    """A literal subset that at least one invocation of a Tool must contain."""

    tool_ref: ToolRef
    inputs: Mapping[str, Any]


@dataclass(frozen=True)
class PlannerConstraints:
    """Trusted constraints supplied explicitly by a Planner caller.

    Constraints are never inferred from LLM-controlled goal or task context.
    A required literal input also makes its Tool required, preventing vacuous
    success when that Tool is absent from the plan.
    """

    required_tool_refs: tuple[ToolRef, ...] = ()
    forbidden_tool_refs: tuple[ToolRef, ...] = ()
    required_literal_inputs: tuple[RequiredLiteralInput, ...] = ()


class PlanningError(Exception):
    def __init__(
        self,
        message: str,
        *,
        validation_problems: tuple[Any, ...] = (),
    ) -> None:
        super().__init__(message)
        # Structured validator provenance is retained separately from the
        # human-readable retry text so callers never infer safety from LLM-
        # controlled titles, labels, or input strings embedded in that text.
        self.validation_problems = tuple(validation_problems)


class ReplanError(PlanningError):
    pass


class ContextBudgetExhaustedError(ReplanError):
    """The protected prompt cannot leave a safe JSON completion budget."""

    error_kind = "context_budget_exhausted"

    def __init__(
        self,
        *,
        context_window: int,
        prompt_tokens: int,
        available_tokens: int,
        required_tokens: int = PLANNER_MIN_OUTPUT_TOKENS,
    ) -> None:
        super().__init__(
            "context_budget_exhausted: planner prompt requires approximately "
            f"{prompt_tokens} tokens in a {context_window}-token context window; "
            f"only {max(available_tokens, 0)} completion tokens remain after the "
            f"safety margin, below the required JSON floor of {required_tokens}."
        )
        self.context_window = context_window
        self.prompt_tokens = prompt_tokens
        self.available_tokens = max(available_tokens, 0)
        self.required_tokens = required_tokens


class Planner:
    def __init__(self, tool_registry, llm_factory, validator):
        self._tools = tool_registry
        self._llm_factory = llm_factory
        self._validator = validator

    def from_template(
        self,
        template: WorkflowTemplate,
        slots: dict,
        task_id: str,
        *,
        autonomy: int | None = None,
    ) -> Plan:
        missing = [
            slot.name
            for slot in template.slots
            if slot.required and (slot.name not in slots or slots[slot.name] is None)
        ]
        if missing:
            raise PlanningError(f"missing required slots: {', '.join(missing)}")

        effective_slots = {
            slot.name: (
                slots[slot.name]
                if slot.name in slots and slots[slot.name] is not None
                else slot.default
                if slot.default is not UNSET_SLOT_DEFAULT
                else _OMIT
            )
            for slot in template.slots
        }
        plan_id = uuid.uuid4().hex
        title_to_id = _title_to_step_id(template, plan_id)
        steps = []
        for index, step_template in enumerate(template.steps):
            step_id = title_to_id[step_template.title]
            manifest_policy = self._tools.resolve(step_template.tool_ref).policy
            legacy_gate_policy = GovernancePolicy(
                human_decision_gate=(
                    "required" if step_template.needs_confirmation else "none"
                )
            )
            effective_policy = merge_governance_policies(
                manifest_policy,
                step_template.policy,
                legacy_gate_policy,
            )
            steps.append(
                PlanStep(
                    id=step_id,
                    plan_id=plan_id,
                    index=index,
                    title=step_template.title,
                    tool_ref=step_template.tool_ref,
                    inputs=_fill_inputs(step_template.inputs_template, effective_slots, title_to_id),
                    depends_on=[
                        _dependency_id(title, title_to_id)
                        for title in step_template.depends_on_titles
                    ],
                    post_checks=list(step_template.post_checks),
                    needs_confirmation=(
                        effective_policy.human_decision_gate == "required"
                    ),
                    policy=effective_policy,
                    decision_point=step_template.decision_point,
                    sub_agent_scope=step_template.sub_agent_scope,
                    granted_tools=list(step_template.granted_tools),
                    phase=step_template.phase,
                )
            )
        return Plan(
            id=plan_id,
            task_id=task_id,
            goal=_template_goal_with_slot_summary(template, slots),
            source="template",
            template_id=template.id,
            steps=steps,
            autonomy_level=autonomy if autonomy is not None else template.default_autonomy,
            success_criteria=[dict(item) for item in template.success_criteria],
        )

    def generate(
        self,
        goal: str,
        task_id: str,
        *,
        memory_context: dict,
        task_context: dict,
        tier: CapabilityTier | None = None,
        novel_mode: str = "plan_ahead",
        max_retries: int = 2,
        constraints: PlannerConstraints | None = None,
    ) -> Plan:
        tier = tier or resolve_tier(None)
        effective_mode = _effective_novel_mode(novel_mode, tier)
        max_steps = (
            tier.explore_segment_size
            if effective_mode == "explore"
            else tier.max_plan_depth
        )
        raw_catalog = self._tools.catalog_for_planner()
        resolved_constraints = _resolve_planner_constraints(
            constraints,
            raw_catalog,
        )
        client = self._llm_factory()
        catalog, catalog_truncated = _truncate_catalog(
            raw_catalog,
            client,
            goal=goal,
            task_context=task_context,
            constraints=resolved_constraints,
        )
        last_error = None
        last_validation_problems: tuple[Any, ...] = ()
        for _attempt in range(max_retries + 1):
            prompt = build_plan_prompt(
                goal,
                catalog,
                memory_context,
                task_context,
                last_error,
                novel_mode=effective_mode,
                max_steps=max_steps,
                constraints=resolved_constraints,
            )
            prompt, max_tokens, request_truncated = _fit_planner_request(
                client,
                system_prompt=PLAN_SYS,
                user_prompt=prompt,
            )
            raw = client.complete(
                system_prompt=PLAN_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                json_schema=PLAN_STEPS_SCHEMA,
                stream=False,
                temperature=0,
                max_tokens=max_tokens,
                caller="planner",
                prompt_name=_PLAN_SYS_SPEC.name,
                prompt_version=_PLAN_SYS_SPEC.version,
                truncated=catalog_truncated or request_truncated,
            )
            try:
                plan = self._parse_plan_json(
                    str(raw),
                    goal,
                    task_id,
                    tier=tier,
                    novel_mode=effective_mode,
                    max_steps=max_steps,
                )
            except PlanningError as exc:
                last_error = str(exc)
                last_validation_problems = ()
                continue
            validation_problems = self._validator.validate_problems(plan)
            problems = [problem.message for problem in validation_problems]
            problems.extend(
                _planner_constraint_violations(
                    plan,
                    resolved_constraints,
                    require_all=effective_mode != "explore",
                )
            )
            if not problems:
                return plan
            last_validation_problems = tuple(validation_problems)
            last_error = "; ".join(problems)
        raise PlanningError(
            f"could not generate valid plan after retries: {last_error}",
            validation_problems=last_validation_problems,
        )

    def replan(
        self,
        plan: Plan,
        *,
        completed_summaries: dict[str, dict],
        observation: dict,
        reason: str,
        tier: CapabilityTier,
        instruction: str | None = None,
        constraints: PlannerConstraints | None = None,
    ) -> Plan:
        if plan.replan_count >= tier.max_replan_iterations:
            raise ReplanError(f"replan budget exhausted ({tier.max_replan_iterations})")

        raw_catalog = self._tools.catalog_for_planner()
        resolved_constraints = _resolve_planner_constraints(
            constraints,
            raw_catalog,
            error_type=ReplanError,
        )
        client = self._llm_factory()
        replan_catalog = _catalog_for_replan(
            raw_catalog,
            plan,
            required_tool_refs=resolved_constraints.required_tool_refs,
            allow_global_fallback=not _has_planner_constraints(
                resolved_constraints
            ),
        )
        catalog, catalog_truncated = _truncate_catalog(
            replan_catalog,
            client,
            goal=plan.goal,
            task_context={
                "workflow_plugins": [step.tool_ref.plugin for step in plan.steps],
                "workflow_tools": [step.tool_ref.label() for step in plan.steps],
                "reason": reason,
                "instruction": instruction,
                "observation": observation,
            },
            constraints=resolved_constraints,
            error_type=ReplanError,
        )
        catalog_scope_refs = {_catalog_ref(item) for item in catalog}
        if _has_planner_constraints(resolved_constraints) and not catalog_scope_refs:
            raise ReplanError(
                "replan catalog scope is empty under planner constraints"
            )
        ledger = build_progress_ledger(plan, completed_summaries)
        context_items = fit_to_budget(
            [
                {"priority": 3, "type": "progress_ledger", "value": ledger},
                {"priority": 2, "type": "observation", "value": observation},
            ],
            max_chars=4000,
        )
        last_error = None
        for attempt in range(MAX_REPLAN_PARSE_RETRY + 1):
            prompt = build_replan_prompt(
                plan,
                catalog,
                context_items,
                observation=observation,
                reason=reason,
                last_error=last_error,
                instruction=instruction,
                constraints=resolved_constraints,
            )
            prompt, max_tokens, request_truncated = _fit_planner_request(
                client,
                system_prompt=REPLAN_SYS,
                user_prompt=prompt,
            )
            allowed_replan_refs = _available_tool_refs_from_prompt(prompt)
            raw = client.complete(
                system_prompt=REPLAN_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                json_schema=PLAN_STEPS_SCHEMA,
                stream=False,
                temperature=0,
                max_tokens=max_tokens,
                caller="planner",
                prompt_name=_REPLAN_SYS_SPEC.name,
                prompt_version=_REPLAN_SYS_SPEC.version,
                truncated=catalog_truncated or request_truncated,
            )
            try:
                revised_remaining = _parse_steps_json(
                    str(raw),
                    plan_id=plan.id,
                    start_index=_next_remaining_index(plan),
                    max_steps=tier.max_plan_depth,
                )
                self._apply_governance_policies(revised_remaining)
                _validate_replan_tool_scope(
                    revised_remaining,
                    allowed_refs=allowed_replan_refs,
                )
                new_plan = _splice_remaining(plan, revised_remaining, tier)
            except (PlanningError, ManifestError) as exc:
                last_error = str(exc)
                logger.warning(
                    "replan attempt rejected stage=parse attempt=%s "
                    "error_type=%s message=%s",
                    attempt + 1,
                    exc.__class__.__name__,
                    _safe_replan_log_message(last_error),
                )
                continue
            problems = self._validator.validate(new_plan)
            problems.extend(
                _planner_constraint_violations(new_plan, resolved_constraints)
            )
            if not problems:
                return new_plan
            last_error = "; ".join(problems)
            logger.warning(
                "replan attempt rejected stage=validator attempt=%s "
                "error_type=PlanValidationError message=%s",
                attempt + 1,
                _safe_replan_log_message(last_error),
            )
        raise ReplanError(f"replan could not produce valid plan: {last_error}")

    def next_explore_segment(
        self,
        plan: Plan,
        *,
        completed_summaries: dict[str, dict],
        tier: CapabilityTier,
        task_context: dict | None = None,
        constraints: PlannerConstraints | None = None,
    ) -> tuple[list[PlanStep], bool]:
        raw_catalog = self._tools.catalog_for_planner()
        resolved_constraints = _resolve_planner_constraints(
            constraints,
            raw_catalog,
            error_type=ReplanError,
        )
        if plan.replan_count >= tier.max_replan_iterations:
            violations = _planner_constraint_violations(
                plan,
                resolved_constraints,
            )
            if violations:
                raise ReplanError(
                    "explore budget exhausted before planner constraints were "
                    f"satisfied: {'; '.join(violations)}"
                )
            return [], True

        client = self._llm_factory()
        catalog_context = dict(task_context or {})
        catalog_context.update(
            {
                "workflow_plugins": [
                    step.tool_ref.plugin for step in plan.steps
                ],
                "workflow_tools": [
                    step.tool_ref.label() for step in plan.steps
                ],
            }
        )
        catalog, catalog_truncated = _truncate_catalog(
            raw_catalog,
            client,
            goal=plan.goal,
            task_context=catalog_context,
            constraints=resolved_constraints,
            error_type=ReplanError,
        )
        ledger = build_progress_ledger(plan, completed_summaries)
        unmet_required_before = _unmet_required_planner_constraints(
            plan,
            resolved_constraints,
        )
        last_error = None
        for _attempt in range(MAX_REPLAN_PARSE_RETRY + 1):
            prompt = build_explore_prompt(
                plan,
                catalog,
                ledger,
                max_steps=tier.explore_segment_size,
                last_error=last_error,
                task_context=task_context or {},
                constraints=resolved_constraints,
            )
            prompt, max_tokens, request_truncated = _fit_planner_request(
                client,
                system_prompt=EXPLORE_SYS,
                user_prompt=prompt,
            )
            raw = client.complete(
                system_prompt=EXPLORE_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                json_schema=PLAN_STEPS_SCHEMA,
                stream=False,
                temperature=0,
                max_tokens=max_tokens,
                caller="planner",
                prompt_name=_EXPLORE_SYS_SPEC.name,
                prompt_version=_EXPLORE_SYS_SPEC.version,
                truncated=catalog_truncated or request_truncated,
            )
            try:
                data = _parse_json_object(str(raw), label="explore JSON")
                if _explore_done_from_json(data):
                    violations = _planner_constraint_violations(
                        plan,
                        resolved_constraints,
                    )
                    if not violations:
                        return [], True
                    last_error = "; ".join(violations)
                    continue
                steps = _parse_steps_json(
                    str(raw),
                    plan_id=plan.id,
                    start_index=_next_append_index(plan),
                    max_steps=tier.explore_segment_size,
                )
                self._apply_governance_policies(steps)
                candidate = _append_segment_plan(plan, steps, tier)
            except (PlanningError, ManifestError) as exc:
                last_error = str(exc)
                continue
            problems = self._validator.validate(candidate)
            problems.extend(
                _planner_constraint_violations(
                    candidate,
                    resolved_constraints,
                    require_all=False,
                )
            )
            unmet_required_after = _unmet_required_planner_constraints(
                candidate,
                resolved_constraints,
            )
            if (
                plan.steps
                and unmet_required_after
                and len(unmet_required_after) >= len(unmet_required_before)
            ):
                problems.append(
                    "explore segment made no progress toward unmet planner "
                    f"constraints: {'; '.join(unmet_required_after)}"
                )
            if not problems:
                return steps, False
            last_error = "; ".join(problems)
        raise ReplanError(f"explore could not produce valid segment: {last_error}")

    def _parse_plan_json(
        self,
        raw: str,
        goal: str,
        task_id: str,
        *,
        tier: CapabilityTier,
        novel_mode: str,
        max_steps: int,
    ) -> Plan:
        # AGT-10: load_json_object tolerates ```json fences and leading/trailing
        # explanatory text that a bare json.loads would reject outright, matching
        # the fence-tolerant parsing already used by decide_gate/route_instruction.
        data, error = load_json_object(raw)
        if data is None:
            raise PlanningError(f"not json: {error}")
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanningError("plan JSON must include non-empty steps")

        try:
            plan_id = uuid.uuid4().hex
            steps = [
                _step_from_json(item, index=index, plan_id=plan_id)
                for index, item in enumerate(raw_steps[:max_steps])
            ]
            self._apply_governance_policies(steps)
            return Plan(
                id=plan_id,
                task_id=task_id,
                goal=goal,
                source="generated",
                template_id=None,
                steps=steps,
                autonomy_level=int(data.get("autonomy_level", tier.default_autonomy_level)),
                novel_mode=novel_mode,
                tier=tier.name,
                success_criteria=_optional_object_list_field(
                    data,
                    "success_criteria",
                    "plan JSON",
                ),
            )
        except (TypeError, ValueError, ManifestError) as exc:
            raise PlanningError(f"invalid plan fields: {exc}") from exc

    def _apply_governance_policies(self, steps: list[PlanStep]) -> None:
        """Resolve every dynamic step against the installed Tool policy.

        LLM/user-authored policy is only a possible raise.  The manifest remains
        the authoritative lower bound, so omitting a policy can never turn a
        protected Tool into an unguarded step.
        """

        for step in steps:
            try:
                manifest_policy = self._tools.resolve(step.tool_ref).policy
            except (PluginNotFoundError, ToolNotFoundError):
                # Preserve the normal validator/retry path for unknown tools.
                continue
            legacy_gate_policy = GovernancePolicy(
                human_decision_gate=(
                    "required" if step.needs_confirmation else "none"
                )
            )
            step.policy = merge_governance_policies(
                manifest_policy,
                step.policy,
                legacy_gate_policy,
            )
            step.needs_confirmation = (
                step.policy.human_decision_gate == "required"
            )


_STALE_CONTEXT_KEY_PREFIXES = (
    "archive_",
    "archived_",
    "deprecated_",
    "historical_",
    "history_",
    "obsolete_",
    "old_",
    "previous_",
    "prior_",
    "stale_",
)
_STALE_CONTEXT_KEYS = frozenset({
    "archive",
    "archived",
    "deprecated",
    "historical",
    "history",
    "obsolete",
    "old",
    "previous",
    "prior",
    "stale",
})
_SECONDARY_CONTEXT_KEYS = {
    "context_items": 2,
    "memory_context": 1,
    "progress_ledger": 2,
}
_PROTECTED_CONTEXT_KEYS = frozenset({
    "active_dataset_revision",
    "analysis_generation",
    "column_map",
    "constraints",
    "dataset_content_hash",
    "dataset_id",
    "dataset_revision",
    "expected_content_hash",
    "feature_columns",
    "feature_names",
    "features",
    "goal",
    "instruction",
    "last_error",
    "observation",
    "planner_constraints",
    "remaining_steps",
    "semantic_mapping_hash",
    "split",
    "split_col",
    "target",
    "target_col",
    "user_constraint",
    "user_constraints",
    "validation_errors",
    "validator_errors",
    "workspace_revision",
})
_PROTECTED_PACKET_LABEL_KEYS = frozenset({"category", "kind", "name", "type"})
_PROTECTED_PACKET_LABEL_TERMS = (
    "active",
    "constraint",
    "current",
    "error",
    "instruction",
    "observation",
    "validation",
    "validator",
)


def _fit_planner_request(
    client,
    *,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, int, bool]:
    """Fit one planner request without rewriting protected business context."""

    profile = getattr(client, "profile", None)
    if not isinstance(profile, dict):
        profile = {}
    context_window = _positive_profile_int(
        profile,
        "context_window",
        default=DEFAULT_CONTEXT_WINDOW,
    )
    profile_cap = _positive_profile_int(
        profile,
        "max_output_tokens",
        default=PLANNER_MAX_OUTPUT_TOKENS,
    )
    requested_output = min(PLANNER_MAX_OUTPUT_TOKENS, profile_cap)
    original_prompt_tokens = _planner_prompt_tokens(system_prompt, user_prompt)
    if requested_output < PLANNER_MIN_OUTPUT_TOKENS:
        raise ContextBudgetExhaustedError(
            context_window=context_window,
            prompt_tokens=original_prompt_tokens,
            available_tokens=requested_output,
        )

    # A small model window may never fit the preferred 16K completion. In that
    # case, trim only far enough to protect the 4K JSON floor; the exact dynamic
    # completion is recomputed from the final prompt below.
    trim_completion_target = (
        requested_output
        if context_window
        > requested_output + PLANNER_CONTEXT_SAFETY_MARGIN_TOKENS
        else PLANNER_MIN_OUTPUT_TOKENS
    )
    prompt_target = max(
        context_window
        - PLANNER_CONTEXT_SAFETY_MARGIN_TOKENS
        - trim_completion_target,
        0,
    )
    request_truncated = False
    try:
        payload = json.loads(user_prompt)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict) and original_prompt_tokens > prompt_target:
        payload, request_truncated = _trim_planner_payload(
            payload,
            system_prompt=system_prompt,
            prompt_target=prompt_target,
        )
        user_prompt = _dump_planner_prompt(payload)

    prompt_tokens = _planner_prompt_tokens(system_prompt, user_prompt)
    available_tokens = (
        context_window
        - prompt_tokens
        - PLANNER_CONTEXT_SAFETY_MARGIN_TOKENS
    )
    max_tokens = min(requested_output, available_tokens)
    if max_tokens < PLANNER_MIN_OUTPUT_TOKENS:
        raise ContextBudgetExhaustedError(
            context_window=context_window,
            prompt_tokens=prompt_tokens,
            available_tokens=available_tokens,
        )
    return user_prompt, max_tokens, request_truncated


def _positive_profile_int(profile: dict, key: str, *, default: int) -> int:
    try:
        value = int(profile.get(key) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _trim_planner_payload(
    payload: dict,
    *,
    system_prompt: str,
    prompt_target: int,
) -> tuple[dict, bool]:
    truncated = False

    def over_budget() -> bool:
        return (
            _planner_prompt_tokens(system_prompt, _dump_planner_prompt(payload))
            > prompt_target
        )

    for path in _context_trim_candidates(payload):
        if not over_budget():
            break
        current = _context_value_at_path(payload, path)
        _set_context_value_at_path(payload, path, _context_omission_marker(current))
        truncated = True

    tools = payload.get("available_tools")
    required_tool_refs = _planner_constraint_refs_from_prompt(payload)
    if isinstance(tools, list):
        # Preserve ranked Tool identities and governance metadata before dropping
        # any entry. Field names are enough to draft refs/inputs; detailed types
        # and descriptions are lower-priority catalog metadata.
        for item in reversed(tools):
            if not over_budget():
                break
            if not isinstance(item, dict):
                continue
            for field_name in ("output_fields", "input_fields"):
                fields = item.get(field_name)
                if not isinstance(fields, list):
                    continue
                compact_fields = [
                    {
                        key: field[key]
                        for key in ("name", "required")
                        if isinstance(field, dict) and key in field
                    }
                    for field in fields
                ]
                if compact_fields != fields:
                    item[field_name] = compact_fields
                    truncated = True
                if not over_budget():
                    break

    omitted_tool_refs: list[tuple[str, str]] = []
    while over_budget() and isinstance(tools, list):
        removable_index = next(
            (
                index
                for index in range(len(tools) - 1, -1, -1)
                if isinstance(tools[index], dict)
                and not _always_retain_catalog_item(tools[index])
                and _catalog_ref(tools[index]) not in required_tool_refs
            ),
            None,
        )
        if removable_index is None:
            break
        item = tools.pop(removable_index)
        omitted_tool_refs.append(_catalog_ref(item))
        payload["available_tools_omission"] = _context_omission_marker(
            omitted_tool_refs
        )
        truncated = True
    return payload, truncated


def _planner_constraint_refs_from_prompt(payload: dict) -> set[tuple[str, str]]:
    constraints = payload.get("planner_constraints")
    if not isinstance(constraints, dict):
        return set()
    raw_refs = constraints.get("required_tool_refs")
    if not isinstance(raw_refs, list):
        return set()
    return {
        _catalog_ref(item)
        for item in raw_refs
        if isinstance(item, dict) and all(_catalog_ref(item))
    }


def _available_tool_refs_from_prompt(user_prompt: str) -> set[tuple[str, str]]:
    try:
        payload = json.loads(user_prompt)
    except (TypeError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    tools = payload.get("available_tools")
    if not isinstance(tools, list):
        return set()
    return {
        _catalog_ref(item)
        for item in tools
        if isinstance(item, dict) and all(_catalog_ref(item))
    }


def _context_trim_candidates(payload: dict) -> list[tuple[str | int, ...]]:
    candidates: dict[tuple[str | int, ...], int] = {}

    def add(path: tuple[str | int, ...], priority: int) -> None:
        candidates[path] = min(priority, candidates.get(path, priority))

    def scan_stale(value: object, path: tuple[str | int, ...]) -> None:
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                child_path = (*path, key)
                if _is_protected_context_key(key):
                    continue
                if _is_stale_context_key(key):
                    add(child_path, 0)
                    continue
                scan_stale(value[key], child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                scan_stale(item, (*path, index))

    def scan_secondary(
        value: object,
        path: tuple[str | int, ...],
        priority: int,
    ) -> None:
        if _is_protected_context_packet(value):
            return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                if _is_protected_context_key(key):
                    continue
                child = value[key]
                child_path = (*path, key)
                if _contains_protected_context(child):
                    scan_secondary(child, child_path, priority)
                else:
                    add(child_path, priority)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                if not _is_protected_context_packet(item):
                    add((*path, index), priority)
            return
        add(path, priority)

    scan_stale(payload, ())
    for key, priority in _SECONDARY_CONTEXT_KEYS.items():
        if key in payload:
            scan_secondary(payload[key], (key,), priority)

    return sorted(
        candidates,
        key=lambda path: (
            candidates[path],
            -estimate_tokens(_canonical_context_value(_context_value_at_path(payload, path))),
            _context_path_label(path),
        ),
    )


def _is_stale_context_key(value: object) -> bool:
    normalized = str(value).strip().casefold()
    return normalized in _STALE_CONTEXT_KEYS or normalized.startswith(
        _STALE_CONTEXT_KEY_PREFIXES
    )


def _is_protected_context_key(value: object) -> bool:
    normalized = str(value).strip().casefold()
    return (
        normalized in _PROTECTED_CONTEXT_KEYS
        or normalized.startswith(("active_", "current_"))
        or any(
            term in normalized
            for term in ("constraint", "instruction", "validator", "validation_error")
        )
    )


def _is_protected_context_packet(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    for key in _PROTECTED_PACKET_LABEL_KEYS:
        label = str(value.get(key) or "").strip().casefold()
        if any(term in label for term in _PROTECTED_PACKET_LABEL_TERMS):
            return True
    return False


def _contains_protected_context(value: object) -> bool:
    if _is_protected_context_packet(value):
        return True
    if isinstance(value, dict):
        return any(
            _is_protected_context_key(key) or _contains_protected_context(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_protected_context(item) for item in value)
    return False


def _context_value_at_path(payload: object, path: tuple[str | int, ...]) -> object:
    value = payload
    for segment in path:
        value = value[segment]  # type: ignore[index]
    return value


def _set_context_value_at_path(
    payload: object,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    parent = payload
    for segment in path[:-1]:
        parent = parent[segment]  # type: ignore[index]
    parent[path[-1]] = value  # type: ignore[index]


def _context_path_label(path: tuple[str | int, ...]) -> str:
    return ".".join(str(segment) for segment in path)


def _context_omission_marker(value: object) -> dict[str, object]:
    return {
        "__omitted_context__": "context_budget",
        "original_type": type(value).__name__,
    }


def _canonical_context_value(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _dump_planner_prompt(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _planner_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    return estimate_tokens(system_prompt) + estimate_tokens(user_prompt)


def build_plan_prompt(
    goal: str,
    catalog: list[dict],
    memory_context: dict,
    task_context: dict,
    last_error: str | None,
    *,
    novel_mode: str = "plan_ahead",
    max_steps: int | None = None,
    constraints: PlannerConstraints | None = None,
) -> str:
    payload = {
        "goal": goal,
        "available_tools": compact_catalog_for_prompt(catalog),
        "planning_examples": PLANNING_EXAMPLES,
        "memory_context": memory_context,
        "task_context": task_context,
        "novel_mode": novel_mode,
        "max_steps": max_steps,
        "last_error": last_error,
        "instruction": (
            "Return a JSON object with steps. Give every step a unique `id`; `id` is "
            "the only step-id field, never `step_id`. Omit optional fields that have "
            "no value instead of null. Each step chooses a tool and inputs; use "
            "$ref:<step_id>.output.<field> for upstream outputs; do not compute "
            "metrics yourself. planner_constraints, when present, are hard "
            "server-authored constraints and must all be satisfied."
        ),
    }
    if _has_planner_constraints(constraints):
        payload["planner_constraints"] = _planner_constraints_payload(constraints)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_explore_prompt(
    plan: Plan,
    catalog: list[dict],
    ledger: str,
    *,
    max_steps: int,
    last_error: str | None,
    task_context: dict,
    constraints: PlannerConstraints | None = None,
) -> str:
    payload = {
        "goal": plan.goal,
        "task_context": task_context,
        "available_tools": compact_catalog_for_prompt(catalog),
        "planning_examples": PLANNING_EXAMPLES,
        "progress_ledger": ledger,
        "max_steps": max_steps,
        "last_error": last_error,
        "instruction": (
            "Return {done: true, steps: []} only if the goal and every required "
            "planner_constraint are complete. Otherwise return only the next segment "
            "steps, no more than max_steps. Give every step a unique `id`; `id` is "
            "the only step-id field, never `step_id`. Omit optional fields that have "
            "no value instead of null. Use $ref:<step_id>.output.<field> for upstream "
            "outputs. planner_constraints, when present, are hard server-authored "
            "constraints. After prior exploration steps exist, the next segment must "
            "advance at least one currently unmet required tool or literal-input "
            "constraint; do not repeat work that makes no constraint progress."
        ),
    }
    if _has_planner_constraints(constraints):
        payload["planner_constraints"] = _planner_constraints_payload(constraints)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_replan_prompt(
    plan: Plan,
    catalog: list[dict],
    context_items: list[dict],
    *,
    observation: dict,
    reason: str,
    last_error: str | None,
    instruction: str | None = None,
    constraints: PlannerConstraints | None = None,
) -> str:
    payload = {
        "goal": plan.goal,
        "reason": reason,
        "available_tools": compact_catalog_for_prompt(catalog),
        "planning_examples": PLANNING_EXAMPLES,
        "context_items": context_items,
        "observation": observation,
        "remaining_steps": [
            _source_step_contract(step)
            for step in plan.steps
            if step.status not in {StepStatus.DONE, StepStatus.SKIPPED}
        ],
        "last_error": last_error,
        "instruction": (
            "Treat remaining_steps as reusable source contracts: copy retained steps "
            "field-for-field unless the user asks to change them. When removing or "
            "replacing a step, remove or rewrite every depends_on edge and $ref that "
            "targets it. Return the full revised remaining plan, including every "
            "retained step; represent a deletion by omitting only the deleted step "
            "from that complete list. When work remains, steps must be non-empty. "
            "Preserve useful "
            "dependencies on completed step ids when needed; do not include completed "
            "steps. "
            "Give every returned step a unique `id`; `id` is the only step-id field, "
            "never `step_id`. Omit optional fields that have no value instead of null. "
            "Use $ref:<step_id>.output.<field> for upstream outputs. "
            "planner_constraints, when present, are hard server-authored constraints "
            "and apply to the complete revised plan."
        ),
    }
    if instruction:
        # The user's free-text replanning constraint (driver §3 提指令→重规划): the
        # revised steps MUST honour it.
        payload["user_constraint"] = instruction
    if _has_planner_constraints(constraints):
        payload["planner_constraints"] = _planner_constraints_payload(constraints)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _source_step_contract(step: PlanStep) -> dict[str, Any]:
    """Expose reusable planning structure without runtime outputs or reviews."""
    payload = {
        "id": step.id,
        "title": step.title,
        "tool": {
            "plugin": step.tool_ref.plugin,
            "tool": step.tool_ref.tool,
            "version": step.tool_ref.version,
        },
        "inputs": dict(step.inputs),
        "depends_on": list(step.depends_on),
        "post_checks": [
            {"kind": check.kind, "spec": dict(check.spec)}
            for check in step.post_checks
        ],
        "needs_confirmation": step.needs_confirmation,
        "policy": step.policy.to_dict(),
        "decision_point": step.decision_point,
        "granted_tools": [
            {
                "plugin": tool_ref.plugin,
                "tool": tool_ref.tool,
                "version": tool_ref.version,
            }
            for tool_ref in step.granted_tools
        ],
    }
    if step.sub_agent_scope is not None:
        payload["sub_agent_scope"] = step.sub_agent_scope
    return payload


def _safe_replan_log_message(value: object) -> str:
    """Keep planner diagnostics single-line and bounded without raw LLM payloads."""
    normalized = " ".join(str(value or "unknown").replace("\x00", "").split())
    return (normalized or "unknown")[:MAX_REPLAN_LOG_MESSAGE_CHARS]


def _resolve_planner_constraints(
    constraints: PlannerConstraints | None,
    catalog: list[dict],
    *,
    error_type: type[PlanningError] = PlanningError,
) -> PlannerConstraints:
    """Validate caller-owned constraints and bind refs to catalog versions."""

    if constraints is None:
        return PlannerConstraints()
    if not isinstance(constraints, PlannerConstraints):
        raise error_type("planner constraints must be a PlannerConstraints object")

    catalog_by_ref: dict[tuple[str, str], list[dict]] = {}
    for item in catalog:
        ref = _catalog_ref(item)
        if all(ref):
            catalog_by_ref.setdefault(ref, []).append(item)

    def resolve(ref: object, label: str) -> ToolRef:
        if not isinstance(ref, ToolRef):
            raise error_type(f"{label} must contain ToolRef values")
        key = _tool_ref_key(ref)
        matches = catalog_by_ref.get(key, [])
        if ref.version:
            matches = [
                item
                for item in matches
                if str(item.get("version") or "") == ref.version
            ]
        versions = {
            str(item.get("version") or "")
            for item in matches
        }
        if not matches:
            raise error_type(f"planner constraints contain unknown tool {ref.label()}")
        if len(versions) != 1:
            raise error_type(
                f"planner constraints contain ambiguous tool {ref.label()}"
            )
        return ToolRef(ref.plugin, ref.tool, versions.pop())

    required: list[ToolRef] = []
    forbidden: list[ToolRef] = []
    required_keys: set[tuple[str, str]] = set()
    forbidden_keys: set[tuple[str, str]] = set()
    for raw_ref in _planner_constraint_items(
        constraints.required_tool_refs,
        "required_tool_refs",
        error_type,
    ):
        ref = resolve(raw_ref, "required_tool_refs")
        key = _tool_ref_key(ref)
        if key not in required_keys:
            required.append(ref)
            required_keys.add(key)
    for raw_ref in _planner_constraint_items(
        constraints.forbidden_tool_refs,
        "forbidden_tool_refs",
        error_type,
    ):
        ref = resolve(raw_ref, "forbidden_tool_refs")
        key = _tool_ref_key(ref)
        if key not in forbidden_keys:
            forbidden.append(ref)
            forbidden_keys.add(key)

    literal_inputs: list[RequiredLiteralInput] = []
    for raw_requirement in _planner_constraint_items(
        constraints.required_literal_inputs,
        "required_literal_inputs",
        error_type,
    ):
        if not isinstance(raw_requirement, RequiredLiteralInput):
            raise error_type(
                "required_literal_inputs must contain RequiredLiteralInput values"
        )
        ref = resolve(raw_requirement.tool_ref, "required_literal_inputs.tool_ref")
        key = _tool_ref_key(ref)
        normalized_inputs = _normalize_literal_input_subset(
            raw_requirement.inputs,
            label=f"required_literal_inputs[{ref.label()}]",
            error_type=error_type,
        )
        literal_inputs.append(RequiredLiteralInput(ref, normalized_inputs))
        if key not in required_keys:
            required.append(ref)
            required_keys.add(key)

    conflicts = sorted(required_keys & forbidden_keys)
    if conflicts:
        labels = ", ".join(f"{plugin}.{tool}" for plugin, tool in conflicts)
        raise error_type(f"planner constraints conflict for tool {labels}")
    return PlannerConstraints(
        required_tool_refs=tuple(required),
        forbidden_tool_refs=tuple(forbidden),
        required_literal_inputs=tuple(literal_inputs),
    )


def _planner_constraint_items(
    value: object,
    label: str,
    error_type: type[PlanningError],
) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise error_type(f"planner constraints {label} must be a list or tuple")
    return tuple(value)


def _normalize_literal_input_subset(
    value: object,
    *,
    label: str,
    error_type: type[PlanningError],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise error_type(f"{label} must be a non-empty mapping")
    normalized = _normalize_literal_value(
        value,
        label=label,
        error_type=error_type,
    )
    assert isinstance(normalized, dict)
    return normalized


def _normalize_literal_value(
    value: object,
    *,
    label: str,
    error_type: type[PlanningError],
) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise error_type(f"{label} must contain finite JSON literals")
    if isinstance(value, str):
        if value.startswith("$ref:"):
            raise error_type(f"{label} must contain literals, not step references")
        return value
    if isinstance(value, list):
        return [
            _normalize_literal_value(
                item,
                label=f"{label}[]",
                error_type=error_type,
            )
            for item in value
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise error_type(f"{label} must use string object keys")
            normalized[key] = _normalize_literal_value(
                item,
                label=f"{label}.{key}",
                error_type=error_type,
            )
        return normalized
    raise error_type(f"{label} must contain only JSON literal values")


def _planner_constraint_violations(
    plan: Plan,
    constraints: PlannerConstraints,
    *,
    require_all: bool = True,
) -> list[str]:
    if not _has_planner_constraints(constraints):
        return []
    steps_by_ref: dict[tuple[str, str], list[PlanStep]] = {}
    granted_refs: set[tuple[str, str]] = set()
    for step in plan.steps:
        steps_by_ref.setdefault(_tool_ref_key(step.tool_ref), []).append(step)
        granted_refs.update(_tool_ref_key(ref) for ref in step.granted_tools)

    violations: list[str] = []
    for ref in constraints.forbidden_tool_refs:
        ref_key = _tool_ref_key(ref)
        if ref_key in steps_by_ref or ref_key in granted_refs:
            violations.append(
                "planner constraint violated: forbidden tool "
                f"{ref.label()} was selected or granted"
            )
    if require_all:
        for ref in constraints.required_tool_refs:
            if _tool_ref_key(ref) not in steps_by_ref:
                violations.append(
                    "planner constraint violated: required tool "
                    f"{ref.label()} is missing"
                )
        for requirement in constraints.required_literal_inputs:
            matching_steps = steps_by_ref.get(
                _tool_ref_key(requirement.tool_ref),
                [],
            )
            if matching_steps and not any(
                _literal_subset_matches(step.inputs, requirement.inputs)
                for step in matching_steps
            ):
                violations.append(
                    "planner constraint violated: required literal inputs mismatch "
                    f"for tool {requirement.tool_ref.label()}"
                )
    return violations


def _unmet_required_planner_constraints(
    plan: Plan,
    constraints: PlannerConstraints,
) -> list[str]:
    """Return each unmet positive completion condition for progress checks."""

    if not _has_planner_constraints(constraints):
        return []
    steps_by_ref: dict[tuple[str, str], list[PlanStep]] = {}
    for step in plan.steps:
        steps_by_ref.setdefault(_tool_ref_key(step.tool_ref), []).append(step)

    unmet: list[str] = []
    for ref in constraints.required_tool_refs:
        if _tool_ref_key(ref) not in steps_by_ref:
            unmet.append(
                "planner constraint violated: required tool "
                f"{ref.label()} is missing"
            )
    for requirement in constraints.required_literal_inputs:
        matching_steps = steps_by_ref.get(
            _tool_ref_key(requirement.tool_ref),
            [],
        )
        if not any(
            _literal_subset_matches(step.inputs, requirement.inputs)
            for step in matching_steps
        ):
            unmet.append(
                "planner constraint violated: required literal inputs mismatch "
                f"for tool {requirement.tool_ref.label()}"
            )
    return unmet


def _literal_subset_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _literal_subset_matches(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _literal_subset_matches(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _has_planner_constraints(constraints: PlannerConstraints | None) -> bool:
    return bool(
        constraints
        and (
            constraints.required_tool_refs
            or constraints.forbidden_tool_refs
            or constraints.required_literal_inputs
        )
    )


def _planner_constraints_payload(constraints: PlannerConstraints) -> dict[str, Any]:
    return {
        "required_tool_refs": [
            _tool_ref_payload(ref)
            for ref in constraints.required_tool_refs
        ],
        "forbidden_tool_refs": [
            _tool_ref_payload(ref)
            for ref in constraints.forbidden_tool_refs
        ],
        "required_literal_inputs": [
            {
                "tool_ref": _tool_ref_payload(requirement.tool_ref),
                "inputs": dict(requirement.inputs),
            }
            for requirement in constraints.required_literal_inputs
        ],
    }


def _tool_ref_payload(ref: ToolRef) -> dict[str, str]:
    return {
        "plugin": ref.plugin,
        "tool": ref.tool,
        "version": ref.version,
    }


def _tool_ref_key(ref: ToolRef) -> tuple[str, str]:
    return (ref.plugin, ref.tool)


def compact_catalog_for_prompt(catalog: list[dict]) -> list[dict]:
    """Keep full schemas out of the LLM prompt; validation still uses them later."""
    compact: list[dict] = []
    for item in catalog:
        prompt_item = {
            "plugin": item.get("plugin"),
            "tool": item.get("tool"),
            "version": item.get("version"),
            "summary": item.get("summary"),
            "determinism": item.get("determinism"),
            "required_inputs": _schema_required(item.get("input_schema")),
            "input_fields": _schema_field_summary(item.get("input_schema")),
            "output_fields": _schema_field_summary(item.get("output_schema")),
        }
        if _catalog_item_has_safety_metadata(item):
            for key in _CATALOG_SAFETY_METADATA_FIELDS:
                if key in item:
                    prompt_item[key] = item[key]
        compact.append(prompt_item)
    return compact


def _catalog_for_replan(
    catalog: list[dict],
    plan: Plan,
    *,
    required_tool_refs: tuple[ToolRef, ...] = (),
    allow_global_fallback: bool = True,
) -> list[dict]:
    """Keep sibling tools and explicitly granted cross-plugin alternatives."""
    source_plugins = {
        step.tool_ref.plugin
        for step in plan.steps
    }
    granted_refs = {
        (tool_ref.plugin, tool_ref.tool)
        for step in plan.steps
        for tool_ref in step.granted_tools
    }
    required_refs = {
        _tool_ref_key(tool_ref)
        for tool_ref in required_tool_refs
    }
    relevant = [
        item
        for item in catalog
        if str(item.get("plugin") or "") in source_plugins
        or _catalog_ref(item) in granted_refs
        or _catalog_ref(item) in required_refs
    ]
    if relevant or not allow_global_fallback:
        return relevant
    return catalog


def _validate_replan_tool_scope(
    steps: list[PlanStep],
    *,
    allowed_refs: set[tuple[str, str]],
) -> None:
    unauthorized = sorted({
        (ref.plugin, ref.tool)
        for step in steps
        for ref in (step.tool_ref, *step.granted_tools)
        if (ref.plugin, ref.tool) not in allowed_refs
    })
    if unauthorized:
        labels = ", ".join(f"{plugin}.{tool}" for plugin, tool in unauthorized)
        raise PlanningError(
            f"tools outside the allowed replan catalog: {labels}"
        )


# LLM-5: the planner's tool catalog is one of the three named highest-volume
# touch points — as the plugin ecosystem grows it balloons linearly with every
# installed pack. Reserve roughly half the model's context window for the
# catalog (the rest covers the system prompt, planning examples, memory/task
# context and completion budget). Catalog entries are ranked against the goal
# and structured task/workflow context before the budget is applied. This keeps
# late-registered packs discoverable without turning retrieval into a fixed
# workflow: the model still chooses and sequences tools, and PlanValidator
# remains the fail-closed authority over its output.
_CATALOG_PROMPT_BUDGET_FRACTION = 0.5
_CATALOG_SAFETY_BUDGET_FRACTION = 0.25
_CATALOG_MAX_SAFETY_ITEMS = 8
_CATALOG_MAX_TERM_DEPTH = 8
_CATALOG_MAX_TERM_NODES = 256
_CATALOG_MAX_TERM_CHARS = 16_384
_CATALOG_MAX_UNIQUE_TERMS = 512
_CATALOG_MAX_CONTAINER_ITEMS = 128

_UNIVERSAL_CORE_TOOL_REFS = frozenset({
    ("data_ops", "infer_schema"),
    ("data_ops", "profile_dataset"),
    ("data_ops", "propose_join"),
})
_CATALOG_SAFETY_METADATA_FIELDS = (
    "policy",
    "requires_confirmation",
    "needs_confirmation",
    "safety",
    "safety_critical",
)
_CATALOG_ROUTE_CONTEXT_KEYS = frozenset({
    "analysis_kind",
    "entrypoint",
    "intent",
    "plugin",
    "plugin_id",
    "plugin_name",
    "plugins",
    "risk_categories",
    "scenario",
    "task_type",
    "template",
    "template_candidates",
    "template_id",
    "tool",
    "tool_ref",
    "tools",
    "workflow",
    "workflow_family",
    "workflow_id",
    "workflow_name",
    "workflow_plugins",
    "workflow_tools",
})
_CATALOG_TERM_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)


def _truncate_catalog(
    catalog: list[dict],
    client,
    *,
    goal: str = "",
    task_context: dict | None = None,
    constraints: PlannerConstraints | None = None,
    error_type: type[PlanningError] = PlanningError,
) -> tuple[list[dict], bool]:
    profile = getattr(client, "profile", None) or {}
    context_window = int(profile.get("context_window") or DEFAULT_CONTEXT_WINDOW)
    budget = max(int(context_window * _CATALOG_PROMPT_BUDGET_FRACTION), 256)
    resolved_constraints = constraints or PlannerConstraints()
    forbidden_refs = {
        _tool_ref_key(ref)
        for ref in resolved_constraints.forbidden_tool_refs
    }
    required_refs = {
        _tool_ref_key(ref)
        for ref in resolved_constraints.required_tool_refs
    }
    allowed_catalog = [
        item
        for item in catalog
        if _catalog_ref(item) not in forbidden_refs
    ]
    ranked = _rank_catalog(
        allowed_catalog,
        goal=goal,
        task_context=task_context or {},
    )
    compact_by_ref = {
        _catalog_ref(item): compact
        for item, compact in zip(ranked, compact_catalog_for_prompt(ranked), strict=True)
    }

    def item_tokens(item: dict) -> int:
        compact = compact_by_ref[_catalog_ref(item)]
        return estimate_tokens(json.dumps(compact, ensure_ascii=False, sort_keys=True))

    ranked_by_ref = {_catalog_ref(item): item for item in ranked}
    missing_required = sorted(required_refs - ranked_by_ref.keys())
    if missing_required:
        labels = ", ".join(f"{plugin}.{tool}" for plugin, tool in missing_required)
        raise error_type(f"planner required tools are unavailable: {labels}")

    total_tokens = sum(item_tokens(item) for item in ranked)
    if total_tokens <= budget:
        return ranked, False

    # Universal introspection is a small fixed reservation. Safety-tagged tools
    # receive their own bounded relevance-ranked slice; an installed pack cannot
    # exhaust the prompt merely by marking hundreds of tools as gated.
    retained = [item for item in ranked if _catalog_ref(item) in required_refs]
    retained_refs = {_catalog_ref(item) for item in retained}
    used_tokens = sum(item_tokens(item) for item in retained)
    if used_tokens > budget:
        labels = ", ".join(
            f"{plugin}.{tool}"
            for plugin, tool in sorted(required_refs)
        )
        raise error_type(
            "planner required tools exceed catalog token budget: "
            f"{labels}"
        )
    for item in ranked:
        ref = _catalog_ref(item)
        if ref in retained_refs or not _always_retain_catalog_item(item):
            continue
        cost = item_tokens(item)
        if used_tokens + cost > budget and required_refs:
            raise error_type(
                "planner required tools and universal catalog tools exceed "
                "catalog token budget"
            )
        retained.append(item)
        retained_refs.add(ref)
        used_tokens += cost
    safety_budget = max(int(budget * _CATALOG_SAFETY_BUDGET_FRACTION), 256)
    safety_tokens = 0
    safety_count = 0
    for item in ranked:
        ref = _catalog_ref(item)
        if ref in retained_refs or not _catalog_item_has_safety_metadata(item):
            continue
        if safety_count >= _CATALOG_MAX_SAFETY_ITEMS:
            continue
        cost = item_tokens(item)
        if safety_tokens + cost > safety_budget or used_tokens + cost > budget:
            continue
        retained.append(item)
        retained_refs.add(ref)
        used_tokens += cost
        safety_tokens += cost
        safety_count += 1

    for item in ranked:
        ref = _catalog_ref(item)
        if ref in retained_refs or _catalog_item_has_safety_metadata(item):
            continue
        cost = item_tokens(item)
        if used_tokens + cost > budget:
            continue
        retained.append(item)
        retained_refs.add(ref)
        used_tokens += cost

    order = {_catalog_ref(item): index for index, item in enumerate(ranked)}
    retained.sort(key=lambda item: order[_catalog_ref(item)])
    if not required_refs.issubset(retained_refs):
        raise error_type("planner required tools were lost during catalog truncation")
    return retained, len(retained) < len(ranked)


def _rank_catalog(
    catalog: list[dict],
    *,
    goal: str,
    task_context: dict,
) -> list[dict]:
    query_weights = _catalog_query_weights(goal, task_context)
    documents = [_catalog_document_weights(item) for item in catalog]
    frequencies = Counter()
    for document in documents:
        frequencies.update(sorted(document))
    catalog_size = max(len(catalog), 1)

    def relevance(index: int) -> float:
        document = documents[index]
        matched_terms = sorted(query_weights.keys() & document.keys())
        lexical_score = math.fsum(
            query_weight
            * document[term]
            * (1.0 + math.log((catalog_size + 1) / (frequencies[term] + 1)))
            for term in matched_terms
            for query_weight in (query_weights[term],)
        )
        tool_terms = _catalog_terms(catalog[index].get("tool"))
        if not tool_terms:
            return lexical_score
        identifier_matches = sorted(tool_terms.intersection(query_weights))
        identifier_coverage = len(identifier_matches) / len(tool_terms)
        return lexical_score + (200.0 * identifier_coverage)

    return [
        item
        for index, item in sorted(
            enumerate(catalog),
            key=lambda pair: (-relevance(pair[0]), pair[0]),
        )
    ]


def _catalog_query_weights(goal: str, task_context: dict) -> dict[str, float]:
    weights: dict[str, float] = {}

    def add(value: object, weight: float) -> None:
        for term in sorted(_catalog_terms(value)):
            if term not in weights and len(weights) >= _CATALOG_MAX_UNIQUE_TERMS:
                continue
            weights[term] = max(weights.get(term, 0.0), weight)

    route_context = {
        key: task_context[key]
        for key in sorted(_CATALOG_ROUTE_CONTEXT_KEYS)
        if key in task_context
    }
    add(route_context, 8.0)
    add(goal, 4.0)
    add(task_context, 2.0)
    return dict(sorted(weights.items()))


def _catalog_document_weights(item: dict) -> dict[str, float]:
    weights: dict[str, float] = {}

    def add(value: object, weight: float) -> None:
        for term in sorted(_catalog_terms(value)):
            weights[term] = max(weights.get(term, 0.0), weight)

    add(item.get("plugin"), 8.0)
    add(item.get("plugin_name"), 6.0)
    add(item.get("plugin_description"), 4.0)
    add(item.get("tool"), 10.0)
    add(item.get("summary"), 4.0)
    add(_schema_search_metadata(item.get("input_schema")), 1.5)
    add(_schema_search_metadata(item.get("output_schema")), 1.0)
    return weights


def _catalog_terms(value: object) -> set[str]:
    terms: set[str] = set()
    stack: list[tuple[object, int]] = [(value, 0)]
    visited_nodes = 0
    visited_chars = 0
    while (
        stack
        and visited_nodes < _CATALOG_MAX_TERM_NODES
        and visited_chars < _CATALOG_MAX_TERM_CHARS
        and len(terms) < _CATALOG_MAX_UNIQUE_TERMS
    ):
        current, depth = stack.pop()
        visited_nodes += 1
        if current is None or depth > _CATALOG_MAX_TERM_DEPTH:
            continue
        children = _catalog_term_children(current)
        if children is not None:
            stack.extend((child, depth + 1) for child in reversed(children))
            continue
        remaining_chars = _CATALOG_MAX_TERM_CHARS - visited_chars
        text = str(current).casefold()[:remaining_chars]
        visited_chars += len(text)
        for token in _CATALOG_TERM_RE.findall(text):
            candidates = {token, *_catalog_term_variants(token)}
            if token and ord(token[0]) >= 0x3400 and len(token) > 2:
                candidates.update(
                    token[index:index + 2]
                    for index in range(len(token) - 1)
                )
            for candidate in sorted(candidates):
                if len(terms) >= _CATALOG_MAX_UNIQUE_TERMS:
                    break
                terms.add(candidate)
    return terms


def _catalog_term_children(value: object) -> list[object] | None:
    if isinstance(value, dict):
        if len(value) > _CATALOG_MAX_CONTAINER_ITEMS:
            return []
        items = sorted(value.items(), key=lambda item: _catalog_sort_key(item[0]))
        return [child for key, item in items for child in (key, item)]
    if isinstance(value, (set, frozenset)):
        if len(value) > _CATALOG_MAX_CONTAINER_ITEMS:
            return []
        return sorted(value, key=_catalog_sort_key)
    if isinstance(value, (list, tuple)):
        return list(value[:_CATALOG_MAX_CONTAINER_ITEMS])
    return None


def _catalog_sort_key(value: object) -> tuple[str, str]:
    return (type(value).__name__, str(value)[:128])


def _catalog_term_variants(term: str) -> set[str]:
    variants: set[str] = set()
    if len(term) > 6 and term.endswith("ation"):
        variants.add(term[:-5] + "ate")
    if len(term) > 5 and term.endswith("ies"):
        variants.add(term[:-3] + "y")
    if len(term) > 5 and term.endswith("ing"):
        variants.add(term[:-3])
    if len(term) > 4 and term.endswith("ed"):
        variants.add(term[:-2])
    if len(term) > 4 and term.endswith("s"):
        variants.add(term[:-1])
    return variants


def _schema_search_metadata(schema: object) -> list[object]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    metadata: list[object] = []
    for name, spec in sorted(properties.items())[:MAX_CATALOG_FIELDS * 2]:
        metadata.append(name)
        if isinstance(spec, dict) and isinstance(spec.get("description"), str):
            metadata.append(spec["description"][:160])
    return metadata


def _always_retain_catalog_item(item: dict) -> bool:
    return _catalog_ref(item) in _UNIVERSAL_CORE_TOOL_REFS


def _catalog_item_has_safety_metadata(item: dict) -> bool:
    policy = item.get("policy")
    if isinstance(policy, dict) and (
        str(policy.get("human_decision_gate") or "none").casefold() == "required"
        or str(policy.get("effect_authorization") or "none").casefold() == "required"
    ):
        return True
    for key in _CATALOG_SAFETY_METADATA_FIELDS[1:]:
        value = item.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().casefold() not in {"", "false", "none"}:
            return True
        if isinstance(value, dict) and value:
            return True
    return False


def _catalog_ref(item: dict) -> tuple[str, str]:
    return (
        str(item.get("plugin") or ""),
        str(item.get("tool") or ""),
    )


def _schema_required(schema) -> list[str]:
    if not isinstance(schema, dict):
        return []
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [str(item) for item in required if isinstance(item, str)]


def _schema_field_summary(schema) -> list[dict[str, Any]]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    fields: list[dict[str, Any]] = []
    required = set(_schema_required(schema))
    for name, spec in sorted(properties.items())[:MAX_CATALOG_FIELDS]:
        field = {"name": str(name), "type": _schema_type_label(spec)}
        if name in required:
            field["required"] = True
        description = spec.get("description") if isinstance(spec, dict) else None
        if isinstance(description, str) and description.strip():
            field["description"] = description.strip()[:120]
        fields.append(field)
    remaining = len(properties) - len(fields)
    if remaining > 0:
        fields.append({
            "name": "...",
            "type": "truncated",
            "description": f"{remaining} more fields omitted from prompt",
        })
    return fields


def _schema_type_label(schema) -> str:
    if not isinstance(schema, dict):
        return "unknown"
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return "|".join(str(item) for item in schema_type)
    if isinstance(schema_type, str):
        return schema_type
    if "anyOf" in schema:
        return "anyOf"
    if "oneOf" in schema:
        return "oneOf"
    if "allOf" in schema:
        return "allOf"
    return "object" if "properties" in schema else "unknown"


def _parse_steps_json(
    raw: str,
    *,
    plan_id: str,
    start_index: int,
    max_steps: int,
) -> list[PlanStep]:
    # AGT-10: the top-level payload here may legitimately be a bare JSON list
    # (not just {"steps": [...]}), which load_json_object never returns (it only
    # extracts objects) -- so try a strict parse first for that shape, and only
    # fall back to load_json_object's ```json fence/prefix-tolerant object
    # extraction when the strict parse fails (e.g. a fenced or prefixed reply).
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        data, error = load_json_object(raw)
        if data is None:
            raise PlanningError(f"not json: {error}") from None
    raw_steps = data.get("steps") if isinstance(data, dict) else data
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanningError("replan JSON must include non-empty steps")
    try:
        return [
            _step_from_json(item, index=start_index + offset, plan_id=plan_id)
            for offset, item in enumerate(raw_steps[:max_steps])
        ]
    except (TypeError, ValueError) as exc:
        raise PlanningError(f"invalid plan fields: {exc}") from exc


def _parse_json_object(raw: str, *, label: str) -> dict:
    # AGT-10: fence/prefix-tolerant like the other two parse sites.
    data, error = load_json_object(raw)
    if data is None:
        raise PlanningError(f"{label} not json: {error}")
    return data


def _next_remaining_index(plan: Plan) -> int:
    preserved = [
        step.index
        for step in plan.steps
        if step.status in {StepStatus.DONE, StepStatus.SKIPPED}
    ]
    return (max(preserved) + 1) if preserved else 0


def _next_append_index(plan: Plan) -> int:
    return max((step.index for step in plan.steps), default=-1) + 1


def _splice_remaining(plan: Plan, revised_remaining: list[PlanStep], tier: CapabilityTier) -> Plan:
    preserved = [
        step
        for step in plan.steps
        if step.status in {StepStatus.DONE, StepStatus.SKIPPED}
    ]
    return Plan(
        id=plan.id,
        task_id=plan.task_id,
        goal=plan.goal,
        source=plan.source,
        template_id=plan.template_id,
        steps=sorted(preserved, key=lambda item: (item.index, item.id)) + revised_remaining,
        autonomy_level=plan.autonomy_level,
        status=plan.status,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
        novel_mode=plan.novel_mode,
        tier=tier.name,
        replan_count=plan.replan_count + 1,
        success_criteria=[dict(item) for item in plan.success_criteria],
    )


def _append_segment_plan(plan: Plan, segment: list[PlanStep], tier: CapabilityTier) -> Plan:
    return Plan(
        id=plan.id,
        task_id=plan.task_id,
        goal=plan.goal,
        source=plan.source,
        template_id=plan.template_id,
        steps=list(plan.steps) + segment,
        autonomy_level=plan.autonomy_level,
        status=plan.status,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
        novel_mode="explore",
        tier=tier.name,
        replan_count=plan.replan_count + 1,
        success_criteria=[dict(item) for item in plan.success_criteria],
    )


def _effective_novel_mode(novel_mode: str, tier: CapabilityTier) -> str:
    if str(novel_mode or "").strip().lower() == "explore" and tier.allow_explore_mode:
        return "explore"
    return "plan_ahead"


def _step_from_json(item: Any, *, index: int, plan_id: str) -> PlanStep:
    if not isinstance(item, dict):
        raise PlanningError(f"steps[{index}] must be an object")
    label = f"steps[{index}]"
    inputs = _optional_object_field(item, "inputs", label)
    depends_on = _optional_list_field(item, "depends_on", label)
    post_checks = _optional_list_field(item, "post_checks", label)
    policy = _optional_object_field(item, "policy", label)
    granted_tools = _optional_list_field(item, "granted_tools", label)
    return PlanStep(
        # Namespace the fallback id by plan_id so LLM responses that omit step ids
        # do not collide across plans (plan_steps.id is a primary key).
        id=_step_id_from_json(item, index=index, plan_id=plan_id),
        plan_id=plan_id,
        index=index,
        title=_required_text(item, "title", label),
        tool_ref=_tool_ref_from_json(item.get("tool"), f"{label}.tool"),
        inputs=inputs,
        depends_on=[str(value) for value in depends_on],
        post_checks=[
            _post_check_from_json(check, check_index)
            for check_index, check in enumerate(post_checks)
        ],
        needs_confirmation=_optional_bool_field(
            item,
            "needs_confirmation",
            label,
        ),
        policy=GovernancePolicy.from_dict(policy),
        decision_point=_optional_bool_field(item, "decision_point", label),
        sub_agent_scope=_optional_text_field(item, "sub_agent_scope", label),
        granted_tools=[
            _tool_ref_from_json(ref, f"steps[{index}].granted_tools")
            for ref in granted_tools
        ],
        phase=_optional_text_field(item, "phase", label),
    )


def _step_id_from_json(item: dict, *, index: int, plan_id: str) -> str:
    label = f"steps[{index}]"
    if "id" in item:
        step_id = _required_text(item, "id", label)
        if "step_id" in item:
            alias = _required_text(item, "step_id", label)
            if alias != step_id:
                raise PlanningError(f"{label} has conflicting id and step_id")
        return step_id
    if "step_id" in item:
        return _required_text(item, "step_id", label)
    return f"{plan_id}-step-{index + 1}"


def _optional_object_field(data: dict, field_name: str, label: str) -> dict:
    if field_name not in data:
        return {}
    value = data[field_name]
    if value is None:
        raise TypeError(
            f"{label}.{field_name} must be omitted rather than null"
        )
    if not isinstance(value, dict):
        raise TypeError(f"{label}.{field_name} must be an object")
    return dict(value)


def _optional_list_field(data: dict, field_name: str, label: str) -> list:
    if field_name not in data:
        return []
    value = data[field_name]
    if value is None:
        raise TypeError(
            f"{label}.{field_name} must be omitted rather than null"
        )
    if not isinstance(value, list):
        raise TypeError(f"{label}.{field_name} must be an array")
    return list(value)


def _optional_object_list_field(
    data: dict,
    field_name: str,
    label: str,
) -> list[dict]:
    values = _optional_list_field(data, field_name, label)
    if any(not isinstance(value, dict) for value in values):
        raise TypeError(f"{label}.{field_name} must contain only objects")
    return [dict(value) for value in values]


def _optional_bool_field(data: dict, field_name: str, label: str) -> bool:
    if field_name not in data:
        return False
    value = data[field_name]
    if value is None:
        raise TypeError(
            f"{label}.{field_name} must be omitted rather than null"
        )
    if not isinstance(value, bool):
        raise TypeError(f"{label}.{field_name} must be a boolean")
    return value


def _explore_done_from_json(data: dict) -> bool:
    try:
        return _optional_bool_field(data, "done", "explore JSON")
    except TypeError as exc:
        raise PlanningError(f"invalid explore fields: {exc}") from exc


def _optional_text_field(
    data: dict,
    field_name: str,
    label: str,
) -> str | None:
    if field_name not in data:
        return None
    value = data[field_name]
    if value is None:
        raise TypeError(
            f"{label}.{field_name} must be omitted rather than null"
        )
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"{label}.{field_name} must be a non-empty string or omitted"
        )
    return value.strip()


def _tool_ref_from_json(value: Any, label: str) -> ToolRef:
    if not isinstance(value, dict):
        raise PlanningError(f"{label} must be an object")
    if "version" in value and value["version"] is None:
        raise TypeError(f"{label}.version must be omitted rather than null")
    return ToolRef(
        plugin=_required_text(value, "plugin", label),
        tool=_required_text(value, "tool", label),
        version=str(value.get("version") or ""),
    )


def _post_check_from_json(value: Any, index: int) -> PostCheck:
    if not isinstance(value, dict):
        raise PlanningError(f"post_checks[{index}] must be an object")
    spec = value.get("spec")
    if not isinstance(spec, dict):
        raise PlanningError(f"post_checks[{index}].spec must be an object")
    return PostCheck(kind=_required_text(value, "kind", f"post_checks[{index}]"), spec=spec)


# AGT-3 (final_review's LLM sees only a "key-name-level" summary of a plan whose
# from_template goal is otherwise just the four-character template title, e.g.
# "标准建模" — giving llm_critique/final_review essentially nothing to reason
# about). Splice a compact summary of the key identifying slots onto the goal so
# the LLM has at least "which dataset/target/recipe" context, without leaking the
# full slot payload (which can contain large lists) into every LLM prompt.
_GOAL_SUMMARY_SLOTS = ("dataset_id", "anchor_id", "target_col", "recipe")


def _template_goal_with_slot_summary(template: WorkflowTemplate, slots: dict) -> str:
    parts = []
    for name in _GOAL_SUMMARY_SLOTS:
        value = slots.get(name)
        if value is None or value == "":
            continue
        parts.append(f"{name}={value}")
    if not parts:
        return template.title
    return f"{template.title}: {' '.join(parts)}"


def _title_to_step_id(template: WorkflowTemplate, plan_id: str) -> dict[str, str]:
    # Step ids must be globally unique (plan_steps.id is a primary key), so namespace
    # them by the plan's unique id — otherwise instantiating the same template twice
    # collides on "step-1"/"step-2"/… across plans (UNIQUE constraint failure).
    title_to_id = {}
    for index, step in enumerate(template.steps):
        if step.title in title_to_id:
            raise PlanningError(f"duplicate step title: {step.title}")
        title_to_id[step.title] = f"{plan_id}-step-{index + 1}"
    return title_to_id


def _dependency_id(title: str, title_to_id: dict[str, str]) -> str:
    try:
        return title_to_id[title]
    except KeyError as exc:
        raise PlanningError(f"unknown dependency title: {title}") from exc


def _fill_inputs(value, slots: dict, title_to_id: dict[str, str]):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            filled = _fill_inputs(item, slots, title_to_id)
            if filled is not _OMIT:
                output[key] = filled
        return output
    if isinstance(value, list):
        return [
            filled
            for item in value
            if (filled := _fill_inputs(item, slots, title_to_id)) is not _OMIT
        ]
    if isinstance(value, str):
        if value.startswith("{slot:") and value.endswith("}"):
            slot_name = value[len("{slot:"):-1]
            if slot_name not in slots:
                raise PlanningError(f"unknown slot: {slot_name}")
            return slots[slot_name]
        if value.startswith("$ref:"):
            return _rewrite_ref(value, title_to_id)
    return value


def _rewrite_ref(value: str, title_to_id: dict[str, str]) -> str:
    raw = value[len("$ref:"):]
    marker = ".output"
    if marker not in raw:
        raise PlanningError(f"invalid ref: {value}")
    title, tail = raw.split(marker, 1)
    if title not in title_to_id:
        raise PlanningError(f"unknown ref title: {title}")
    return f"$ref:{title_to_id[title]}.output{tail}"


def _required_text(data: dict, field_name: str, label: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise PlanningError(f"{label}.{field_name} is required")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PlanningError("optional text fields must be strings")
    return value.strip() or None
