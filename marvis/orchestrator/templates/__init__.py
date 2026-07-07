from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from marvis.orchestrator.contracts import PostCheck
from marvis.plugins.manifest import ToolRef


@dataclass(frozen=True)
class SlotSpec:
    name: str
    required: bool
    source: str
    description: str


@dataclass(frozen=True)
class StepTemplate:
    title: str
    tool_ref: ToolRef
    inputs_template: dict
    depends_on_titles: tuple[str, ...]
    post_checks: tuple[PostCheck, ...]
    needs_confirmation: bool = False
    decision_point: bool = False
    sub_agent_scope: str | None = None
    granted_tools: tuple[ToolRef, ...] = ()
    # Display-only big-step grouping label, threaded onto each PlanStep.phase.
    phase: str | None = None


@dataclass(frozen=True)
class WorkflowTemplate:
    id: str
    title: str
    goal_patterns: tuple[str, ...]
    slots: tuple[SlotSpec, ...]
    steps: tuple[StepTemplate, ...]
    default_autonomy: int = 1
    source: str = "builtin"
    success_criteria: tuple[dict, ...] = ()


_TEMPLATES: dict[str, WorkflowTemplate] = {}


def register_template(template: WorkflowTemplate) -> None:
    if template.id in _TEMPLATES:
        raise ValueError(f"duplicate template id: {template.id}")
    _TEMPLATES[template.id] = template


def register_user_template(template: WorkflowTemplate) -> None:
    if template.id in builtin_template_ids():
        raise ValueError(f"user template cannot shadow builtin template: {template.id}")
    _TEMPLATES[template.id] = _with_source(template, "user")


def get_template(template_id: str) -> WorkflowTemplate:
    try:
        return _TEMPLATES[template_id]
    except KeyError as exc:
        raise KeyError(template_id) from exc


def list_templates() -> list[WorkflowTemplate]:
    return [
        template
        for _template_id, template in sorted(_TEMPLATES.items(), key=lambda item: item[0])
    ]


def builtin_template_ids() -> set[str]:
    return {
        template.id
        for template in _TEMPLATES.values()
        if template.source == "builtin"
    }


def clear_user_templates() -> None:
    for template_id in [
        template.id
        for template in _TEMPLATES.values()
        if template.source == "user"
    ]:
        del _TEMPLATES[template_id]


def load_builtin_templates() -> None:
    module = import_module("marvis.orchestrator.templates.sample")
    module.register_all_builtin_templates()


def _register_builtin_template(template: WorkflowTemplate) -> None:
    existing = _TEMPLATES.get(template.id)
    if existing is not None:
        if existing.source == "builtin":
            _TEMPLATES[template.id] = _with_source(template, "builtin")
            return
        raise ValueError(f"builtin template id conflicts with user template: {template.id}")
    _TEMPLATES[template.id] = _with_source(template, "builtin")


def _with_source(template: WorkflowTemplate, source: str) -> WorkflowTemplate:
    return WorkflowTemplate(
        id=template.id,
        title=template.title,
        goal_patterns=template.goal_patterns,
        slots=template.slots,
        steps=template.steps,
        default_autonomy=template.default_autonomy,
        source=source,
        success_criteria=template.success_criteria,
    )
