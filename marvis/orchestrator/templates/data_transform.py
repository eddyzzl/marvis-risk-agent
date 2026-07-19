from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import ToolRef


DATASET_TRANSFORM = WorkflowTemplate(
    id="dataset_transform",
    title="数据加工",
    goal_patterns=(
        "数据加工",
        "清洗数据",
        "填补缺失",
        "删除字段",
        "重命名字段",
        "转换字段类型",
        "筛选样本",
        "生成字段",
        "去重",
        "dataset transform",
        "clean dataset",
        "fill missing values",
        "rename columns",
        "filter rows",
        "deduplicate rows",
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
            "Exact source data-workspace revision",
        ),
        SlotSpec(
            "analysis_generation",
            True,
            "task_context",
            "Exact source active-dataset analysis generation",
        ),
        SlotSpec(
            "semantic_mapping_hash",
            True,
            "task_context",
            "SHA-256 of the confirmed source semantic mapping",
        ),
        SlotSpec(
            "operations",
            True,
            "user",
            "Ordered closed transform operations compiled from the user request",
        ),
        SlotSpec(
            "confirm_protected_drop",
            False,
            "user",
            "Explicit confirmation already obtained for dropping target or key fields",
        ),
    ),
    steps=(
        StepTemplate(
            title="执行数据加工并登记血缘",
            tool_ref=ToolRef("data_ops", "transform_dataset"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "expected_content_hash": "{slot:expected_content_hash}",
                "workspace_revision": "{slot:workspace_revision}",
                "analysis_generation": "{slot:analysis_generation}",
                "semantic_mapping_hash": "{slot:semantic_mapping_hash}",
                "operations": "{slot:operations}",
                "confirm_protected_drop": "{slot:confirm_protected_drop}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck(
                    "one_of",
                    {
                        "field": "schema_version",
                        "values": ["data-transform-tool-result.v1"],
                    },
                ),
                PostCheck(
                    "invariant",
                    {"rule": "source_dataset_id==lineage.parent_dataset_id"},
                ),
                PostCheck(
                    "invariant",
                    {"rule": "result_dataset_id==lineage.child_dataset_id"},
                ),
                PostCheck(
                    "schema",
                    {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "result_content_hash": {
                                    "type": "string",
                                    "pattern": "^[0-9a-f]{64}$",
                                }
                            },
                            "required": ["result_content_hash"],
                        }
                    },
                ),
                PostCheck("rowcount", {"field": "row_count_after", "min": 0}),
                PostCheck("nonempty", {"field": "evidence_artifact_id"}),
                PostCheck("nonempty", {"field": "evidence_download_url"}),
            ),
            # The operation creates an immutable child dataset and activates it;
            # normal transforms are reversible by selecting the parent again.  A
            # protected-field drop is confirmed before this slot is set, rather
            # than imposing a redundant gate on every transform.
            needs_confirmation=False,
            decision_point=False,
            phase="数据准备",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


__all__ = ["DATASET_TRANSFORM"]
