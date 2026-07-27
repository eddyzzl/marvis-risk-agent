from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_PROJECT_CONTEXT
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugins)


def _slots() -> dict[str, object]:
    return {
        "expected_revision": 0,
        "expected_revision_id": None,
        "expected_state_hash": None,
        "user_message_ref": {
            "message_id": "message-1",
            "content_hash": "a" * 64,
        },
        "as_of": "2026-06-30",
        "scope": None,
        "business_context": {"project.background": "test"},
        "explicit_unavailable": ["current.status_fields.economics"],
        "external_report_filenames": [],
    }


def test_project_context_is_one_nongated_builtin_tool_step() -> None:
    template = STRATEGY_PROJECT_CONTEXT

    assert template in BUILTIN_TEMPLATES
    assert template == get_template("strategy_project_context")
    assert len(template.steps) == 1
    assert {slot.name for slot in template.slots if slot.source == "task_context"} == {
        "expected_revision",
        "expected_revision_id",
        "expected_state_hash",
        "user_message_ref",
    }
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "materialize_project_context")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "revision"}),
        PostCheck("nonempty", {"field": "context_artifact"}),
    )


def test_project_context_template_maps_exact_cas_and_user_inputs(tmp_path: Path) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)
    plan = Planner(tools, lambda: None, validator).from_template(
        get_template("strategy_project_context"),
        _slots(),
        task_id="task-1",
    )

    assert validator.validate(plan) == []
    assert plan.steps[0].inputs == {
        key: value for key, value in _slots().items() if value is not None
    }


def test_project_context_manifest_closes_platform_owned_inputs(tmp_path: Path) -> None:
    tool = _tool_registry(tmp_path).resolve(
        ToolRef("strategy", "materialize_project_context")
    )

    assert tool.input_schema["additionalProperties"] is False
    assert set(tool.input_schema["required"]) == {
        key for key, value in _slots().items() if value is not None
    }
    assert "metrics" not in tool.input_schema["properties"]
    assert "dataset_id" not in tool.input_schema["properties"]
    assert tool.output_schema["additionalProperties"] is False
    assert tool.entrypoint == "tool_materialize_project_context"
    assert tool.policy.effect_authorization == "none"
