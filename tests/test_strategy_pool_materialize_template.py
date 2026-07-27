"""Builtin Workflow contract for governed Pool-to-draft materialization."""

from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_POOL_MATERIALIZE
from marvis.plugins.manifest import ToolRef


def test_pool_materialize_is_one_nongated_builtin_tool_step() -> None:
    template = STRATEGY_POOL_MATERIALIZE

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_pool_materialize"
    assert template.default_autonomy == 1
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "materialize_strategy_from_pool")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "schema_version"}),
        PostCheck("nonempty", {"field": "materialization_id"}),
        PostCheck("nonempty", {"field": "strategy_ref.strategy_id"}),
        PostCheck("nonempty", {"field": "strategy_ref.strategy_spec_hash"}),
        PostCheck("nonempty", {"field": "pool_ref.revision_id"}),
        PostCheck("nonempty", {"field": "lifecycle.current_status"}),
    )


def test_pool_materialize_template_passes_exactly_six_authenticated_inputs() -> None:
    template = STRATEGY_POOL_MATERIALIZE
    slot_sources = {slot.name: slot.source for slot in template.slots}

    assert slot_sources == {
        "strategy_type": "user",
        "expected_pool_revision": "task_context",
        "expected_pool_snapshot_hash": "task_context",
        "expected_pool_artifact_id": "task_context",
        "expected_pool_artifact_content_hash": "task_context",
        "expected_design_hash": "task_context",
    }
    assert template.steps[0].inputs_template == {
        name: f"{{slot:{name}}}" for name in slot_sources
    }
