from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT,
    STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT_EXISTING,
)
from marvis.plugins.manifest import ToolRef


def test_candidate_refinement_is_one_governed_two_step_builtin_workflow():
    template = STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_univariate_candidate_refinement"
    assert {slot.name for slot in template.slots if slot.required} == {
        "dataset_id",
        "expected_content_hash",
        "semantic_mapping_hash",
        "target_col",
        "bin_count",
        "min_bin_pct",
        "feature",
        "method",
        "selection",
    }
    assert len(template.steps) == 2
    analyze, refine = template.steps
    assert analyze.tool_ref == ToolRef("strategy", "analyze_univariate_candidates")
    assert refine.tool_ref == ToolRef("strategy", "refine_univariate_candidate")
    assert refine.depends_on_titles == ("分析单变量候选",)
    assert refine.inputs_template["source_artifact_id"] == (
        "$ref:分析单变量候选.output.artifacts.0.artifact_id"
    )
    assert refine.inputs_template["expected_artifact_content_hash"] == (
        "$ref:分析单变量候选.output.artifacts.0.content_hash"
    )
    assert refine.inputs_template["expected_candidate_id"] == (
        "$ref:分析单变量候选.output.candidate_id"
    )
    assert refine.inputs_template["expected_evidence_hash"] == (
        "$ref:分析单变量候选.output.evidence_hash"
    )
    assert refine.needs_confirmation is False
    assert refine.decision_point is False
    assert refine.post_checks == (
        PostCheck("nonempty", {"field": "asset_id"}),
        PostCheck("nonempty", {"field": "asset_hash"}),
        PostCheck("nonempty", {"field": "effect_id"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_existing_candidate_refinement_never_regenerates_ordinal_source_bins():
    template = STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT_EXISTING

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_univariate_candidate_refinement_existing"
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "refine_univariate_candidate")
    assert step.depends_on_titles == ()
    assert step.inputs_template["source_artifact_id"] == "{slot:source_artifact_id}"
    assert step.inputs_template["expected_candidate_id"] == (
        "{slot:expected_candidate_id}"
    )
    assert "dataset_id" not in step.inputs_template
