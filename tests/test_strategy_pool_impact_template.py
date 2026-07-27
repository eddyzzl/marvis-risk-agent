"""Builtin Workflow contract for Strategy Pool impact measurement."""

from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_POOL_IMPACT
from marvis.plugins.manifest import ToolRef


def test_pool_impact_is_one_reversible_nongated_tool_step() -> None:
    template = STRATEGY_POOL_IMPACT

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_pool_impact"
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "measure_pool_impact")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "assessment_id"}),
        PostCheck("nonempty", {"field": "content_hash"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_pool_impact_template_passes_only_governed_measurement_slots() -> None:
    template = STRATEGY_POOL_IMPACT
    slot_sources = {slot.name: slot.source for slot in template.slots}
    nan_slot = next(slot for slot in template.slots if slot.name == "drop_nan_labels")

    assert {
        name for name, source in slot_sources.items() if source == "task_context"
    } == {
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
    }
    assert set(template.steps[0].inputs_template) == set(slot_sources)
    assert "retaining sample rows" in nan_slot.description
    assert not {
        "strategy_spec",
        "condition",
        "metrics",
        "create_strategy",
        "adopt",
        "deploy",
    } & set(template.steps[0].inputs_template)
