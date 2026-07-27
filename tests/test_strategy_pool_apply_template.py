"""Builtin Workflow contract for governed Strategy Pool application."""

from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_POOL_APPLY
from marvis.plugins.manifest import ToolRef


def test_pool_apply_is_one_nongated_builtin_tool_step() -> None:
    template = STRATEGY_POOL_APPLY

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_pool_apply"
    assert template.default_autonomy == 1
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "apply_strategy_pool")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "schema_version"}),
        PostCheck("nonempty", {"field": "run_id"}),
        PostCheck("nonempty", {"field": "result.dataset_id"}),
        PostCheck("range", {"field": "result.row_count", "min": 0}),
        PostCheck("nonempty", {"field": "evidence.artifact_id"}),
    )


def test_pool_apply_template_passes_only_user_controls_and_pool_cas() -> None:
    template = STRATEGY_POOL_APPLY
    slot_sources = {slot.name: slot.source for slot in template.slots}

    assert slot_sources == {
        "strategy_type": "user",
        "output_prefix": "user",
        "expected_pool_revision": "task_context",
        "expected_pool_snapshot_hash": "task_context",
    }
    assert template.steps[0].inputs_template == {
        "strategy_type": "{slot:strategy_type}",
        "output_prefix": "{slot:output_prefix}",
        "expected_pool_revision": "{slot:expected_pool_revision}",
        "expected_pool_snapshot_hash": "{slot:expected_pool_snapshot_hash}",
    }
    assert not {
        "pool_id",
        "artifact_id",
        "dataset_id",
        "sample_design_ref",
        "requirements",
        "strategy_spec",
        "activated",
        "adopted",
        "deployed",
    } & set(template.steps[0].inputs_template)
