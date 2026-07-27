"""Builtin Workflow contract for unified Strategy ImpactCube evidence."""

from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_IMPACT_CUBE
from marvis.plugins.manifest import ToolRef


def test_impact_cube_is_one_reversible_nongated_tool_step() -> None:
    template = STRATEGY_IMPACT_CUBE

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_impact_cube"
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef(
        "strategy",
        "measure_strategy_impact_cube",
    )
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "cube_id"}),
        PostCheck("nonempty", {"field": "content_hash"}),
        PostCheck("nonempty", {"field": "artifact"}),
    )


def test_impact_cube_template_passes_only_exact_governed_bindings() -> None:
    template = STRATEGY_IMPACT_CUBE
    slot_sources = {slot.name: slot.source for slot in template.slots}

    assert {
        name for name, source in slot_sources.items() if source == "user"
    } == {
        "strategy_type",
        "dimension_bindings",
        "current_strategy_ref",
        "economics_inputs",
    }
    assert {
        name for name, source in slot_sources.items() if source == "task_context"
    } == {
        "pool_ref",
        "sample_design_ref",
        "partitions",
        "population",
    }
    assert set(template.steps[0].inputs_template) == set(slot_sources)
    required = {slot.name for slot in template.slots if slot.required}
    assert "current_strategy_ref" not in required
    assert "economics_inputs" not in required
    assert not {
        "metrics",
        "strategy_spec",
        "create_strategy",
        "adopt",
        "promote",
        "deploy",
    } & set(template.steps[0].inputs_template)
