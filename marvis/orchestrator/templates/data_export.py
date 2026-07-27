from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import ToolRef


DATASET_EXPORT = WorkflowTemplate(
    id="dataset_export",
    title="数据集导出",
    goal_patterns=(
        "导出当前数据为 CSV",
        "导出当前数据为 Excel",
        "下载当前样本",
        "export current dataset as csv",
        "export current dataset as xlsx",
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
            "Exact data-workspace revision used for export",
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
        SlotSpec("format", True, "user", "Closed export format: csv or xlsx"),
        SlotSpec(
            "text_columns",
            False,
            "user",
            "Optional fields that must be preserved as text",
        ),
    ),
    steps=(
        StepTemplate(
            title="安全导出当前数据集",
            tool_ref=ToolRef("data_ops", "export_dataset"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "expected_content_hash": "{slot:expected_content_hash}",
                "workspace_revision": "{slot:workspace_revision}",
                "analysis_generation": "{slot:analysis_generation}",
                "semantic_mapping_hash": "{slot:semantic_mapping_hash}",
                "format": "{slot:format}",
                "text_columns": "{slot:text_columns}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck(
                    "one_of",
                    {
                        "field": "schema_version",
                        "values": ["dataset-export-tool-result.v1"],
                    },
                ),
                PostCheck(
                    "one_of", {"field": "format", "values": ["csv", "xlsx"]}
                ),
                PostCheck("rowcount", {"field": "row_count", "min": 0}),
                PostCheck(
                    "schema",
                    {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "input_hash": {
                                    "type": "string",
                                    "pattern": "^[0-9a-f]{64}$",
                                },
                                "dataset_content_hash": {
                                    "type": "string",
                                    "pattern": "^[0-9a-f]{64}$",
                                },
                                "semantic_mapping_hash": {
                                    "type": "string",
                                    "pattern": "^[0-9a-f]{64}$",
                                },
                                "content_hash": {
                                    "type": "string",
                                    "pattern": "^[0-9a-f]{64}$",
                                }
                            },
                            "required": [
                                "input_hash",
                                "dataset_content_hash",
                                "semantic_mapping_hash",
                                "content_hash",
                            ],
                        }
                    },
                ),
                PostCheck("nonempty", {"field": "dataset_id"}),
                PostCheck("nonempty", {"field": "artifact_id"}),
                PostCheck("nonempty", {"field": "download_url"}),
            ),
            # This produces a new local immutable artifact and does not mutate
            # the source dataset, so a confirmation gate adds no safety value.
            needs_confirmation=False,
            decision_point=False,
            phase="数据准备",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


__all__ = ["DATASET_EXPORT"]
