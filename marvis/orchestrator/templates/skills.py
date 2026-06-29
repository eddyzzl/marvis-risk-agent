from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from marvis.orchestrator.contracts import Plan, PlanStep, PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
    builtin_template_ids,
    register_user_template,
)
from marvis.plugins.manifest import ToolRef


class SkillTemplateError(Exception):
    pass


@dataclass
class SkillLoadReport:
    active: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    rejected: list[tuple[str, list[str]]] = field(default_factory=list)


def parse_skill_template(data: dict) -> WorkflowTemplate:
    if not isinstance(data, dict):
        raise SkillTemplateError("skill template must be an object")
    template_id = _required_text(data, "id")
    title = _required_text(data, "title")
    goal_patterns = tuple(_string_list(data.get("goal_patterns", []), "goal_patterns"))
    slots = tuple(_parse_slot(item, index) for index, item in enumerate(_required_list(data, "slots")))
    steps = tuple(_parse_step(item, index) for index, item in enumerate(_required_list(data, "steps")))
    if not steps:
        raise SkillTemplateError("steps must not be empty")
    default_autonomy = data.get("default_autonomy", 1)
    if not isinstance(default_autonomy, int) or isinstance(default_autonomy, bool):
        raise SkillTemplateError("default_autonomy must be an integer")
    return WorkflowTemplate(
        id=template_id,
        title=title,
        goal_patterns=goal_patterns,
        slots=slots,
        steps=steps,
        default_autonomy=default_autonomy,
        success_criteria=tuple(
            dict(item)
            for item in data.get("success_criteria") or []
            if isinstance(item, dict)
        ),
        source="user",
    )


def validate_skill_template(template, tool_registry, plan_validator) -> list[str]:
    problems: list[str] = []
    if template.id in builtin_template_ids():
        problems.append(f"skill id '{template.id}' shadows a builtin template")
    try:
        plan = _dry_instantiate(template)
    except SkillTemplateError as exc:
        problems.append(str(exc))
    else:
        problems.extend(plan_validator.validate(plan))
    return problems


def load_user_skill_templates(workspace, tool_registry, plan_validator) -> SkillLoadReport:
    report = SkillLoadReport()
    skills_dir = Path(workspace) / "skills"
    if not skills_dir.exists():
        return report
    for path in sorted(skills_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.rejected.append((path.stem, [f"unreadable: {exc}"]))
            continue
        skill_id = str(data.get("id") or path.stem) if isinstance(data, dict) else path.stem
        if isinstance(data, dict) and data.get("enabled", True) is False:
            report.disabled.append(skill_id)
            continue
        try:
            template = parse_skill_template(data)
        except SkillTemplateError as exc:
            report.rejected.append((skill_id, [str(exc)]))
            continue
        problems = validate_skill_template(template, tool_registry, plan_validator)
        if problems:
            report.rejected.append((template.id, problems))
            continue
        register_user_template(template)
        report.active.append(template.id)
    return report


def _dry_instantiate(template: WorkflowTemplate) -> Plan:
    slot_values = {slot.name: f"{{slot:{slot.name}}}" for slot in template.slots}
    title_to_id = _title_to_step_id(template)
    steps = []
    for index, step_template in enumerate(template.steps):
        step_id = title_to_id[step_template.title]
        steps.append(
            PlanStep(
                id=step_id,
                plan_id=f"dry-{template.id}",
                index=index,
                title=step_template.title,
                tool_ref=step_template.tool_ref,
                inputs=_fill_inputs(step_template.inputs_template, slot_values, title_to_id),
                depends_on=[
                    _dependency_id(title, title_to_id)
                    for title in step_template.depends_on_titles
                ],
                post_checks=list(step_template.post_checks),
                needs_confirmation=step_template.needs_confirmation,
                decision_point=step_template.decision_point,
                sub_agent_scope=step_template.sub_agent_scope,
                granted_tools=list(step_template.granted_tools),
            )
        )
    return Plan(
        id=f"dry-{template.id}",
        task_id="<dry>",
        goal=template.title,
        source="template",
        template_id=template.id,
        steps=steps,
        autonomy_level=template.default_autonomy,
        success_criteria=[dict(item) for item in template.success_criteria],
    )


def _title_to_step_id(template: WorkflowTemplate) -> dict[str, str]:
    title_to_id = {}
    for index, step in enumerate(template.steps):
        if step.title in title_to_id:
            raise SkillTemplateError(f"duplicate step title: {step.title}")
        title_to_id[step.title] = f"step-{index + 1}"
    return title_to_id


def _dependency_id(title: str, title_to_id: dict[str, str]) -> str:
    try:
        return title_to_id[title]
    except KeyError as exc:
        raise SkillTemplateError(f"unknown dependency title: {title}") from exc


def _fill_inputs(value, slots: dict[str, str], title_to_id: dict[str, str]):
    if isinstance(value, dict):
        return {
            key: _fill_inputs(item, slots, title_to_id)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_fill_inputs(item, slots, title_to_id) for item in value]
    if isinstance(value, str):
        if value.startswith("{slot:") and value.endswith("}"):
            slot_name = value[len("{slot:"):-1]
            if slot_name not in slots:
                raise SkillTemplateError(f"unknown slot: {slot_name}")
            return slots[slot_name]
        if value.startswith("$ref:"):
            return _rewrite_ref(value, title_to_id)
    return value


def _rewrite_ref(value: str, title_to_id: dict[str, str]) -> str:
    raw = value[len("$ref:"):]
    marker = ".output"
    if marker not in raw:
        raise SkillTemplateError(f"invalid ref: {value}")
    title, tail = raw.split(marker, 1)
    if title not in title_to_id:
        raise SkillTemplateError(f"unknown ref title: {title}")
    return f"$ref:{title_to_id[title]}.output{tail}"


def _parse_slot(item: Any, index: int) -> SlotSpec:
    if not isinstance(item, dict):
        raise SkillTemplateError(f"slots[{index}] must be an object")
    return SlotSpec(
        name=_required_text(item, "name", label=f"slots[{index}]"),
        required=bool(item.get("required", True)),
        source=_required_text(item, "source", label=f"slots[{index}]"),
        description=_required_text(item, "description", label=f"slots[{index}]"),
    )


def _parse_step(item: Any, index: int) -> StepTemplate:
    if not isinstance(item, dict):
        raise SkillTemplateError(f"steps[{index}] must be an object")
    return StepTemplate(
        title=_required_text(item, "title", label=f"steps[{index}]"),
        tool_ref=_parse_tool_ref(item.get("tool"), f"steps[{index}].tool"),
        inputs_template=dict(item.get("inputs") or {}),
        depends_on_titles=tuple(_string_list(item.get("depends_on", []), f"steps[{index}].depends_on")),
        post_checks=tuple(
            _parse_post_check(check, check_index)
            for check_index, check in enumerate(item.get("post_checks") or [])
        ),
        needs_confirmation=bool(item.get("needs_confirmation", False)),
        decision_point=bool(item.get("decision_point", False)),
        sub_agent_scope=_optional_text(item.get("sub_agent_scope")),
        granted_tools=tuple(
            _parse_tool_ref(ref, f"steps[{index}].granted_tools")
            for ref in item.get("granted_tools") or []
        ),
    )


def _parse_tool_ref(value: Any, label: str) -> ToolRef:
    if not isinstance(value, dict):
        raise SkillTemplateError(f"{label} must be an object")
    return ToolRef(
        plugin=_required_text(value, "plugin", label=label),
        tool=_required_text(value, "tool", label=label),
        version=str(value.get("version") or ""),
    )


def _parse_post_check(value: Any, index: int) -> PostCheck:
    if not isinstance(value, dict):
        raise SkillTemplateError(f"post_checks[{index}] must be an object")
    spec = value.get("spec")
    if not isinstance(spec, dict):
        raise SkillTemplateError(f"post_checks[{index}].spec must be an object")
    return PostCheck(kind=_required_text(value, "kind", label=f"post_checks[{index}]"), spec=spec)


def _required_list(data: dict, field_name: str) -> list:
    value = data.get(field_name)
    if not isinstance(value, list):
        raise SkillTemplateError(f"{field_name} must be a list")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SkillTemplateError(f"{label} must be a list of strings")
    return list(value)


def _required_text(data: dict, field_name: str, *, label: str = "skill") -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise SkillTemplateError(f"{label}.{field_name} is required")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SkillTemplateError("optional text fields must be strings")
    return value.strip() or None
