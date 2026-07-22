from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS,
)
from marvis.plugins.manifest import ToolRef


def test_univariate_candidate_analysis_is_one_governed_builtin_step():
    template = STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_univariate_candidate_analysis"
    assert {slot.name for slot in template.slots if slot.required} == {
        "dataset_id",
        "expected_content_hash",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
        "bin_count",
        "min_bin_pct",
    }
    assert {slot.name for slot in template.slots if not slot.required} == {
        "workspace_revision",
        "analysis_generation",
        "drop_nan_labels",
        "features",
        "methods",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
    }
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "analyze_univariate_candidates")
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "candidate_id"}),
        PostCheck("nonempty", {"field": "evidence_hash"}),
        PostCheck("nonempty", {"field": "artifacts"}),
        PostCheck("range", {"field": "rankings.0.iv", "min": 0.0}),
        PostCheck(
            "range",
            {"field": "rankings.0.ks", "min": 0.0, "max": 1.0},
        ),
        PostCheck(
            "range",
            {"field": "rankings.0.auc", "min": 0.0, "max": 1.0},
        ),
    )
    assert step.inputs_template["expected_content_hash"] == (
        "{slot:expected_content_hash}"
    )
