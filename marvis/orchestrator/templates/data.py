from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import ToolRef


DATASET_DESCRIPTIVE_ANALYSIS = WorkflowTemplate(
    id="dataset_descriptive_analysis",
    title="样本描述分析",
    goal_patterns=(
        "样本描述分析",
        "分析这份样本",
        "数据概况",
        "目标分布",
        "缺失分析",
        "相关性分析",
        "dataset descriptive analysis",
        "dataset profile",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Active task-owned dataset id"),
        SlotSpec(
            "expected_content_hash",
            True,
            "task_context",
            "Registered SHA-256 bound to the active data workspace",
        ),
        SlotSpec(
            "workspace_revision",
            True,
            "task_context",
            "Exact data-workspace revision used for the analysis",
        ),
        SlotSpec(
            "analysis_generation",
            True,
            "task_context",
            "Exact active-dataset analysis generation",
        ),
        SlotSpec(
            "semantic_mapping_hash",
            True,
            "task_context",
            "SHA-256 of the confirmed semantic mapping",
        ),
        SlotSpec("sections", False, "user", "Requested descriptive-analysis sections"),
        SlotSpec("columns", False, "user", "Optional explicit analysis columns"),
        SlotSpec(
            "target_col",
            False,
            "task_context",
            "Optional target column; must match the workspace semantic mapping",
        ),
        SlotSpec(
            "frequency_top_k",
            False,
            "user",
            "Maximum frequency rows per low-cardinality field",
        ),
        SlotSpec(
            "low_cardinality_threshold",
            False,
            "user",
            "Distinct-count threshold for frequency output",
        ),
        SlotSpec(
            "histogram_bins",
            False,
            "user",
            "Histogram bin count for numeric distributions",
        ),
        SlotSpec(
            "correlation_batch_size",
            False,
            "user",
            "Numeric-column batch size for correlation computation",
        ),
    ),
    steps=(
        StepTemplate(
            title="计算样本描述分析",
            tool_ref=ToolRef("data_ops", "profile_dataset"),
            inputs_template={
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
            },
            depends_on_titles=(),
            post_checks=(
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
            ),
            needs_confirmation=False,
            decision_point=False,
            phase="数据分析",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


__all__ = ["DATASET_DESCRIPTIVE_ANALYSIS"]
