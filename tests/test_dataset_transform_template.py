from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates.data_transform import DATASET_TRANSFORM
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.plugins.manifest import ToolRef


def test_dataset_transform_is_a_single_bound_builtin_step_without_a_reversible_gate():
    template = DATASET_TRANSFORM

    assert template in BUILTIN_TEMPLATES
    assert template.id == "dataset_transform"
    assert {
        "数据加工",
        "清洗数据",
        "填补缺失",
        "重命名字段",
        "筛选样本",
        "去重",
    } <= set(template.goal_patterns)
    assert {slot.name for slot in template.slots if slot.required} == {
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "operations",
    }
    assert {slot.name for slot in template.slots if not slot.required} == {
        "confirm_protected_drop",
    }
    assert len(template.steps) == 1

    step = template.steps[0]
    assert step.tool_ref == ToolRef("data_ops", "transform_dataset")
    assert step.inputs_template == {
        "dataset_id": "{slot:dataset_id}",
        "expected_content_hash": "{slot:expected_content_hash}",
        "workspace_revision": "{slot:workspace_revision}",
        "analysis_generation": "{slot:analysis_generation}",
        "semantic_mapping_hash": "{slot:semantic_mapping_hash}",
        "operations": "{slot:operations}",
        "confirm_protected_drop": "{slot:confirm_protected_drop}",
    }
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
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
    )
