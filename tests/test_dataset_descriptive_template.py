from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.data import DATASET_DESCRIPTIVE_ANALYSIS
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.plugins.manifest import ToolRef


def test_dataset_descriptive_analysis_is_a_single_read_only_builtin_step():
    template = DATASET_DESCRIPTIVE_ANALYSIS

    assert template in BUILTIN_TEMPLATES
    assert template.id == "dataset_descriptive_analysis"
    assert {
        "分析这份样本",
        "目标分布",
        "缺失分析",
        "相关性分析",
    } <= set(template.goal_patterns)
    assert {slot.name for slot in template.slots if slot.required} == {
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
    }
    assert {slot.name for slot in template.slots if not slot.required} == {
        "sections",
        "columns",
        "target_col",
        "frequency_top_k",
        "low_cardinality_threshold",
        "histogram_bins",
        "correlation_batch_size",
    }
    assert len(template.steps) == 1

    step = template.steps[0]
    assert step.tool_ref == ToolRef("data_ops", "profile_dataset")
    assert step.inputs_template == {
        "dataset_id": "{slot:dataset_id}",
        "expected_content_hash": "{slot:expected_content_hash}",
        "workspace_revision": "{slot:workspace_revision}",
        "analysis_generation": "{slot:analysis_generation}",
        "semantic_mapping_hash": "{slot:semantic_mapping_hash}",
        "sections": "{slot:sections}",
        "columns": "{slot:columns}",
        "target_col": "{slot:target_col}",
        "frequency_top_k": "{slot:frequency_top_k}",
        "low_cardinality_threshold": "{slot:low_cardinality_threshold}",
        "histogram_bins": "{slot:histogram_bins}",
        "correlation_batch_size": "{slot:correlation_batch_size}",
    }
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("rowcount", {"field": "row_count_scanned", "min": 0}),
        PostCheck(
            "invariant",
            {"rule": "row_count_scanned==row_count"},
        ),
        PostCheck(
            "invariant",
            {"rule": "dataset_content_hash==expected_content_hash"},
        ),
        PostCheck(
            "one_of",
            {"field": "scan_scope", "values": ["full_dataset"]},
        ),
    )
