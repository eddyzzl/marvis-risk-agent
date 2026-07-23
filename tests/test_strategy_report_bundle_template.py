"""Builtin Workflow contract for StrategyReportBundle V2."""

from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_REPORT_BUNDLE_V2
from marvis.plugins.manifest import ToolRef


def test_report_bundle_is_one_reversible_nongated_tool_step() -> None:
    template = STRATEGY_REPORT_BUNDLE_V2

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_report_bundle_v2"
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "build_report_bundle_v2")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "report_id"}),
        PostCheck("nonempty", {"field": "report_revision"}),
        PostCheck("nonempty", {"field": "content_hash"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_report_bundle_exposes_only_title_and_status_as_user_slots() -> None:
    template = STRATEGY_REPORT_BUNDLE_V2
    slot_sources = {slot.name: slot.source for slot in template.slots}

    assert {
        name for name, source in slot_sources.items() if source == "user"
    } == {"title", "status"}
    assert {
        name for name, source in slot_sources.items() if source == "task_context"
    } == {
        "project_context_ref",
        "sample_design_ref",
        "candidate_pool_ref",
        "pool_impact_ref",
        "report_revision",
        "previous_report_id",
        "previous_report_content_hash",
        "generated_at",
        "strategy_identity",
        "model_evidence_ref",
        "training_evidence_ref",
        "score_evidence_ref",
    }
    assert set(template.steps[0].inputs_template) == set(slot_sources)
    assert not {
        "metrics",
        "strategy_spec",
        "create_strategy",
        "adopt",
        "deploy",
    } & set(template.steps[0].inputs_template)
