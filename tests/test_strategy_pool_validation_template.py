"""Builtin Workflow contract for independent current-Pool replay evidence."""

from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_POOL_VALIDATION
from marvis.plugins.manifest import ToolRef


def test_pool_validation_is_one_read_only_nongated_step() -> None:
    template = STRATEGY_POOL_VALIDATION

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_pool_validation"
    assert template.default_autonomy == 1
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "measure_strategy_pool_validation")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "schema_version"}),
        PostCheck("nonempty", {"field": "evidence_id"}),
        PostCheck("nonempty", {"field": "artifact.artifact_id"}),
        PostCheck("range", {"field": "population_count", "min": 1}),
    )


def test_pool_validation_template_keeps_only_two_controls_user_owned() -> None:
    template = STRATEGY_POOL_VALIDATION
    slot_sources = {slot.name: slot.source for slot in template.slots}

    assert slot_sources == {
        "strategy_type": "user",
        "partition": "user",
        "pool_ref": "task_context",
        "sample_design_ref": "task_context",
        "population": "task_context",
        "comparison_mode": "task_context",
    }
    assert template.steps[0].inputs_template == {
        "strategy_type": "{slot:strategy_type}",
        "partition": "{slot:partition}",
        "pool_ref": "{slot:pool_ref}",
        "sample_design_ref": "{slot:sample_design_ref}",
        "population": "{slot:population}",
        "comparison_mode": "{slot:comparison_mode}",
    }
