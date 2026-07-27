from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.data_export import DATASET_EXPORT
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.plugins.manifest import ToolRef


def test_dataset_export_is_a_single_safe_builtin_step_without_confirmation_gate():
    template = DATASET_EXPORT

    assert template in BUILTIN_TEMPLATES
    assert template.id == "dataset_export"
    assert {slot.name for slot in template.slots if slot.required} == {
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "format",
    }
    assert {slot.name for slot in template.slots if not slot.required} == {
        "text_columns",
    }
    assert len(template.steps) == 1

    step = template.steps[0]
    assert step.tool_ref == ToolRef("data_ops", "export_dataset")
    assert step.inputs_template == {
        "dataset_id": "{slot:dataset_id}",
        "expected_content_hash": "{slot:expected_content_hash}",
        "workspace_revision": "{slot:workspace_revision}",
        "analysis_generation": "{slot:analysis_generation}",
        "semantic_mapping_hash": "{slot:semantic_mapping_hash}",
        "format": "{slot:format}",
        "text_columns": "{slot:text_columns}",
    }
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck(
            "one_of",
            {
                "field": "schema_version",
                "values": ["dataset-export-tool-result.v1"],
            },
        ),
        PostCheck("one_of", {"field": "format", "values": ["csv", "xlsx"]}),
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
    )
