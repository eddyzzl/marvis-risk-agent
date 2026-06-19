from __future__ import annotations

import json
from typing import Any
import uuid

from marvis.orchestrator.contracts import Plan, PlanStep, PostCheck
from marvis.orchestrator.templates import WorkflowTemplate
from marvis.plugins.manifest import ToolRef


PLAN_SYS = (
    "你是 MARVIS 的规划器。只能从给定工具目录选工具、把它们连成 DAG。"
    "铁律：你不计算任何指标；指标由工具产出。"
    "你只决定调用哪些工具、参数怎么接、依赖顺序。输出严格 JSON。"
)


class PlanningError(Exception):
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

        plan_id = uuid.uuid4().hex
        title_to_id = _title_to_step_id(template)
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
                    inputs=_fill_inputs(step_template.inputs_template, slots, title_to_id),
                    depends_on=[
                        _dependency_id(title, title_to_id)
                        for title in step_template.depends_on_titles
                    ],
                    post_checks=list(step_template.post_checks),
                    needs_confirmation=step_template.needs_confirmation,
                    sub_agent_scope=step_template.sub_agent_scope,
                    granted_tools=list(step_template.granted_tools),
                )
            )
        return Plan(
            id=plan_id,
            task_id=task_id,
            goal=template.title,
            source="template",
            template_id=template.id,
            steps=steps,
            autonomy_level=autonomy if autonomy is not None else template.default_autonomy,
        )

    def generate(
        self,
        goal: str,
        task_id: str,
        *,
        memory_context: dict,
        task_context: dict,
        max_retries: int = 2,
    ) -> Plan:
        catalog = self._tools.catalog_for_planner()
        last_error = None
        for _attempt in range(max_retries + 1):
            prompt = build_plan_prompt(
                goal,
                catalog,
                memory_context,
                task_context,
                last_error,
            )
            raw = self._llm_factory().complete(
                system_prompt=PLAN_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                stream=False,
            )
            try:
                plan = self._parse_plan_json(str(raw), goal, task_id)
            except PlanningError as exc:
                last_error = str(exc)
                continue
            problems = self._validator.validate(plan)
            if not problems:
                return plan
            last_error = "; ".join(problems)
        raise PlanningError(f"could not generate valid plan after retries: {last_error}")

    def _parse_plan_json(self, raw: str, goal: str, task_id: str) -> Plan:
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
            for index, item in enumerate(raw_steps)
        ]
        return Plan(
            id=plan_id,
            task_id=task_id,
            goal=goal,
            source="generated",
            template_id=None,
            steps=steps,
            autonomy_level=int(data.get("autonomy_level", 1)),
        )


def build_plan_prompt(
    goal: str,
    catalog: list[dict],
    memory_context: dict,
    task_context: dict,
    last_error: str | None,
) -> str:
    return json.dumps(
        {
            "goal": goal,
            "available_tools": catalog,
            "memory_context": memory_context,
            "task_context": task_context,
            "last_error": last_error,
            "instruction": (
                "Return a JSON object with steps. Each step chooses a tool and inputs; "
                "do not compute metrics yourself."
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _step_from_json(item: Any, *, index: int, plan_id: str) -> PlanStep:
    if not isinstance(item, dict):
        raise PlanningError(f"steps[{index}] must be an object")
    return PlanStep(
        id=str(item.get("id") or f"step-{index + 1}"),
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
        sub_agent_scope=_optional_text(item.get("sub_agent_scope")),
        granted_tools=[
            _tool_ref_from_json(ref, f"steps[{index}].granted_tools")
            for ref in item.get("granted_tools") or []
        ],
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


def _title_to_step_id(template: WorkflowTemplate) -> dict[str, str]:
    title_to_id = {}
    for index, step in enumerate(template.steps):
        if step.title in title_to_id:
            raise PlanningError(f"duplicate step title: {step.title}")
        title_to_id[step.title] = f"step-{index + 1}"
    return title_to_id


def _dependency_id(title: str, title_to_id: dict[str, str]) -> str:
    try:
        return title_to_id[title]
    except KeyError as exc:
        raise PlanningError(f"unknown dependency title: {title}") from exc


def _fill_inputs(value, slots: dict, title_to_id: dict[str, str]):
    if isinstance(value, dict):
        return {key: _fill_inputs(item, slots, title_to_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill_inputs(item, slots, title_to_id) for item in value]
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
