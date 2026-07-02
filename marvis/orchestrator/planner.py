from __future__ import annotations

import json
from typing import Any
import uuid

from marvis.orchestrator.capability import CapabilityTier, resolve_tier
from marvis.orchestrator.context.budget import fit_to_budget
from marvis.orchestrator.context.ledger import build_progress_ledger
from marvis.orchestrator.contracts import Plan, PlanStep, PostCheck, StepStatus
from marvis.orchestrator.templates import WorkflowTemplate
from marvis.plugins.manifest import ToolRef


PLAN_SYS = (
    "你是 MARVIS 的规划器。只能从给定工具目录选工具、把它们连成 DAG。"
    "铁律：你不计算任何指标；指标由工具产出。"
    "你只决定调用哪些工具、参数怎么接、依赖顺序。输出严格 JSON。"
)

_OMIT = object()
REPLAN_SYS = (
    "你在修订一个 MARVIS 执行计划的剩余步骤。已完成步骤和结果在进度里，"
    "不要重做。只能从工具目录选工具。不要计算任何指标。不要偏离原始目标。"
    "输出严格 JSON，格式为 {\"steps\": [...]}。"
)
EXPLORE_SYS = (
    "你在 MARVIS explore 模式下规划下一小段步骤。基于进度判断目标是否已完成。"
    "若已完成，输出 {\"done\": true, \"steps\": []}；否则只输出下一小段 steps。"
    "只能从工具目录选工具，不计算指标，输出严格 JSON。"
)
MAX_REPLAN_PARSE_RETRY = 1
MAX_CATALOG_FIELDS = 12

PLANNING_EXAMPLES = (
    {
        "purpose": "Call a first tool with literal inputs.",
        "step": {
            "title": "读取数据概况",
            "tool": {"plugin": "data_ops", "tool": "profile_dataset"},
            "inputs": {"dataset_id": "dataset-1"},
            "depends_on": [],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "row_count"}}],
        },
    },
    {
        "purpose": "Reference a previous tool output; depends_on must include that step id.",
        "step": {
            "title": "生成报告",
            "tool": {"plugin": "report", "tool": "generate"},
            "inputs": {"experiment_id": "$ref:train-step.output.experiment_id"},
            "depends_on": ["train-step"],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "report_path"}}],
        },
    },
)


class PlanningError(Exception):
    pass


class ReplanError(PlanningError):
    pass


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
            if slot.required and not slots.get(slot.name)
        ]
        if missing:
            raise PlanningError(f"missing required slots: {', '.join(missing)}")

        effective_slots = {
            slot.name: slots[slot.name] if slot.name in slots and slots[slot.name] is not None else _OMIT
            for slot in template.slots
        }
        plan_id = uuid.uuid4().hex
        title_to_id = _title_to_step_id(template, plan_id)
        steps = []
        for index, step_template in enumerate(template.steps):
            step_id = title_to_id[step_template.title]
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
                    needs_confirmation=step_template.needs_confirmation,
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
    ) -> Plan:
        tier = tier or resolve_tier(None)
        effective_mode = _effective_novel_mode(novel_mode, tier)
        max_steps = (
            tier.explore_segment_size
            if effective_mode == "explore"
            else tier.max_plan_depth
        )
        catalog = self._tools.catalog_for_planner()
        last_error = None
        for _attempt in range(max_retries + 1):
            prompt = build_plan_prompt(
                goal,
                catalog,
                memory_context,
                task_context,
                last_error,
                novel_mode=effective_mode,
                max_steps=max_steps,
            )
            raw = self._llm_factory().complete(
                system_prompt=PLAN_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                stream=False,
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
                continue
            problems = self._validator.validate(plan)
            if not problems:
                return plan
            last_error = "; ".join(problems)
        raise PlanningError(f"could not generate valid plan after retries: {last_error}")

    def replan(
        self,
        plan: Plan,
        *,
        completed_summaries: dict[str, dict],
        observation: dict,
        reason: str,
        tier: CapabilityTier,
        instruction: str | None = None,
    ) -> Plan:
        if plan.replan_count >= tier.max_replan_iterations:
            raise ReplanError(f"replan budget exhausted ({tier.max_replan_iterations})")

        catalog = self._tools.catalog_for_planner()
        ledger = build_progress_ledger(plan, completed_summaries)
        context_items = fit_to_budget(
            [
                {"priority": 3, "type": "progress_ledger", "value": ledger},
                {"priority": 2, "type": "observation", "value": observation},
            ],
            max_chars=4000,
        )
        last_error = None
        for _attempt in range(MAX_REPLAN_PARSE_RETRY + 1):
            prompt = build_replan_prompt(
                plan,
                catalog,
                context_items,
                observation=observation,
                reason=reason,
                last_error=last_error,
                instruction=instruction,
            )
            raw = self._llm_factory().complete(
                system_prompt=REPLAN_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                stream=False,
            )
            try:
                revised_remaining = _parse_steps_json(
                    str(raw),
                    plan_id=plan.id,
                    start_index=_next_remaining_index(plan),
                    max_steps=tier.max_plan_depth,
                )
                new_plan = _splice_remaining(plan, revised_remaining, tier)
            except PlanningError as exc:
                last_error = str(exc)
                continue
            problems = self._validator.validate(new_plan)
            if not problems:
                return new_plan
            last_error = "; ".join(problems)
        raise ReplanError(f"replan could not produce valid plan: {last_error}")

    def next_explore_segment(
        self,
        plan: Plan,
        *,
        completed_summaries: dict[str, dict],
        tier: CapabilityTier,
    ) -> tuple[list[PlanStep], bool]:
        if plan.replan_count >= tier.max_replan_iterations:
            return [], True

        catalog = self._tools.catalog_for_planner()
        ledger = build_progress_ledger(plan, completed_summaries)
        last_error = None
        for _attempt in range(MAX_REPLAN_PARSE_RETRY + 1):
            prompt = build_explore_prompt(
                plan,
                catalog,
                ledger,
                max_steps=tier.explore_segment_size,
                last_error=last_error,
            )
            raw = self._llm_factory().complete(
                system_prompt=EXPLORE_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                stream=False,
            )
            try:
                data = _parse_json_object(str(raw), label="explore JSON")
                if bool(data.get("done", False)):
                    return [], True
                steps = _parse_steps_json(
                    str(raw),
                    plan_id=plan.id,
                    start_index=_next_append_index(plan),
                    max_steps=tier.explore_segment_size,
                )
                candidate = _append_segment_plan(plan, steps, tier)
            except PlanningError as exc:
                last_error = str(exc)
                continue
            problems = self._validator.validate(candidate)
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
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"not json: {exc}") from exc
        if not isinstance(data, dict):
            raise PlanningError("plan JSON must be an object")
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanningError("plan JSON must include non-empty steps")

        plan_id = uuid.uuid4().hex
        steps = [
            _step_from_json(item, index=index, plan_id=plan_id)
            for index, item in enumerate(raw_steps[:max_steps])
        ]
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
            success_criteria=[
                dict(item)
                for item in data.get("success_criteria") or []
                if isinstance(item, dict)
            ],
        )


def build_plan_prompt(
    goal: str,
    catalog: list[dict],
    memory_context: dict,
    task_context: dict,
    last_error: str | None,
    *,
    novel_mode: str = "plan_ahead",
    max_steps: int | None = None,
) -> str:
    return json.dumps(
        {
            "goal": goal,
            "available_tools": compact_catalog_for_prompt(catalog),
            "planning_examples": PLANNING_EXAMPLES,
            "memory_context": memory_context,
            "task_context": task_context,
            "novel_mode": novel_mode,
            "max_steps": max_steps,
            "last_error": last_error,
            "instruction": (
                "Return a JSON object with steps. Each step chooses a tool and inputs; "
                "use $ref:<step_id>.output.<field> for upstream outputs; "
                "do not compute metrics yourself."
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_explore_prompt(
    plan: Plan,
    catalog: list[dict],
    ledger: str,
    *,
    max_steps: int,
    last_error: str | None,
) -> str:
    return json.dumps(
        {
            "goal": plan.goal,
            "available_tools": compact_catalog_for_prompt(catalog),
            "planning_examples": PLANNING_EXAMPLES,
            "progress_ledger": ledger,
            "max_steps": max_steps,
            "last_error": last_error,
            "instruction": (
                "Return {done: true, steps: []} if the goal is complete. Otherwise "
                "return only the next segment steps, no more than max_steps. "
                "Use $ref:<step_id>.output.<field> for upstream outputs."
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_replan_prompt(
    plan: Plan,
    catalog: list[dict],
    context_items: list[dict],
    *,
    observation: dict,
    reason: str,
    last_error: str | None,
    instruction: str | None = None,
) -> str:
    payload = {
        "goal": plan.goal,
        "reason": reason,
        "available_tools": compact_catalog_for_prompt(catalog),
        "planning_examples": PLANNING_EXAMPLES,
        "context_items": context_items,
        "observation": observation,
        "remaining_steps": [
            {"id": step.id, "title": step.title, "status": step.status.value}
            for step in plan.steps
            if step.status not in {StepStatus.DONE, StepStatus.SKIPPED}
        ],
        "last_error": last_error,
        "instruction": (
            "Return only revised remaining steps. Preserve useful dependencies on "
            "completed step ids when needed; do not include completed steps. "
            "Use $ref:<step_id>.output.<field> for upstream outputs."
        ),
    }
    if instruction:
        # The user's free-text replanning constraint (driver §3 提指令→重规划): the
        # revised steps MUST honour it.
        payload["user_constraint"] = instruction
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def compact_catalog_for_prompt(catalog: list[dict]) -> list[dict]:
    """Keep full schemas out of the LLM prompt; validation still uses them later."""
    return [
        {
            "plugin": item.get("plugin"),
            "tool": item.get("tool"),
            "version": item.get("version"),
            "summary": item.get("summary"),
            "determinism": item.get("determinism"),
            "required_inputs": _schema_required(item.get("input_schema")),
            "input_fields": _schema_field_summary(item.get("input_schema")),
            "output_fields": _schema_field_summary(item.get("output_schema")),
        }
        for item in catalog
    ]


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
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlanningError(f"not json: {exc}") from exc
    raw_steps = data.get("steps") if isinstance(data, dict) else data
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanningError("replan JSON must include non-empty steps")
    return [
        _step_from_json(item, index=start_index + offset, plan_id=plan_id)
        for offset, item in enumerate(raw_steps[:max_steps])
    ]


def _parse_json_object(raw: str, *, label: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlanningError(f"not json: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanningError(f"{label} must be an object")
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
    return PlanStep(
        # Namespace the fallback id by plan_id so LLM responses that omit step ids
        # do not collide across plans (plan_steps.id is a primary key).
        id=str(item.get("id") or f"{plan_id}-step-{index + 1}"),
        plan_id=plan_id,
        index=index,
        title=_required_text(item, "title", f"steps[{index}]"),
        tool_ref=_tool_ref_from_json(item.get("tool"), f"steps[{index}].tool"),
        inputs=dict(item.get("inputs") or {}),
        depends_on=[str(value) for value in item.get("depends_on") or []],
        post_checks=[
            _post_check_from_json(check, check_index)
            for check_index, check in enumerate(item.get("post_checks") or [])
        ],
        needs_confirmation=bool(item.get("needs_confirmation", False)),
        decision_point=bool(item.get("decision_point", False)),
        sub_agent_scope=_optional_text(item.get("sub_agent_scope")),
        granted_tools=[
            _tool_ref_from_json(ref, f"steps[{index}].granted_tools")
            for ref in item.get("granted_tools") or []
        ],
        phase=_optional_text(item.get("phase")),
    )


def _tool_ref_from_json(value: Any, label: str) -> ToolRef:
    if not isinstance(value, dict):
        raise PlanningError(f"{label} must be an object")
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
