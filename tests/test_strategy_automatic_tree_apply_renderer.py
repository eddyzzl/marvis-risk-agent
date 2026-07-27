"""Dedicated rendering for governed automatic-tree full writeback."""

from __future__ import annotations

from copy import deepcopy

from marvis.agent.renderers import render_tool_output


def _output() -> dict:
    return {
        "schema_version": "strategy.apply-automatic-tree-tool.v1",
        "run_id": "atar_" + "1" * 32,
        "input_hash": "2" * 64,
        "cached": False,
        "activated": False,
        "source": {
            "tree_artifact_id": "artifact-tree",
            "tree_artifact_content_hash": "3" * 64,
            "asset_id": "candidate-asset-" + "4" * 32,
            "asset_hash": "5" * 64,
            "tree_result_hash": "6" * 64,
            "dataset_id": "dataset-source",
            "dataset_content_hash": "7" * 64,
            "row_count": 24,
        },
        "result": {
            "dataset_id": "dataset-derived",
            "dataset_content_hash": "8" * 64,
            "row_count": 24,
            "result_hash": "9" * 64,
        },
        "columns": {
            "leaf_id": "tree_leaf_bucket",
            "rule_id": "tree_rule_bucket",
        },
        "leaf_distribution": [
            {"leaf_id": "leaf-left", "rule_id": "rule-left", "row_count": 10},
            {
                "leaf_id": "leaf-right",
                "rule_id": "rule-right",
                "row_count": 14,
            },
        ],
        "workspace": {
            "source_revision": 3,
            "source_analysis_generation": 1,
            "source_semantic_mapping_hash": "a" * 64,
            "result_revision": None,
            "result_analysis_generation": None,
            "result_semantic_mapping_hash": "b" * 64,
            "active_dataset_id": "dataset-source",
        },
        "evidence": {
            "artifact_id": "artifact-apply-evidence",
            "content_hash": "c" * 64,
            "download_url": (
                "/api/tasks/task-1/task-artifacts/artifact-apply-evidence/download"
            ),
        },
    }


def test_automatic_tree_apply_renderer_surfaces_identity_and_reversible_boundary() -> (
    None
):
    output = _output()

    text, tables = render_tool_output("apply_automatic_tree", output)

    assert output["source"]["asset_id"] in text
    assert output["source"]["tree_artifact_id"] in text
    assert output["source"]["dataset_id"] in text
    assert output["result"]["dataset_id"] in text
    assert output["columns"]["leaf_id"] in text
    assert output["columns"]["rule_id"] in text
    assert output["evidence"]["artifact_id"] in text
    assert output["evidence"]["download_url"] in text
    assert "development / unvalidated" in text
    assert "当前 workspace 未切换" in text
    assert "未入池、未采纳、未部署" in text
    assert "最佳" not in text
    assert "推荐" not in text
    assert [table["title"] for table in tables] == [
        "自动树全量写回身份",
        "自动树叶节点写回分布",
    ]
    assert tables[0]["rows"] == [
        ["Source Tree Asset", output["source"]["asset_id"]],
        ["Source Tree Artifact", output["source"]["tree_artifact_id"]],
        ["Source Dataset", output["source"]["dataset_id"]],
        ["Result Dataset", output["result"]["dataset_id"]],
        ["Leaf ID Column", output["columns"]["leaf_id"]],
        ["Rule ID Column", output["columns"]["rule_id"]],
        ["Evidence Artifact", output["evidence"]["artifact_id"]],
    ]
    assert tables[1]["rows"] == [
        ["leaf-left", "rule-left", "10"],
        ["leaf-right", "rule-right", "14"],
    ]


def test_automatic_tree_apply_renderer_fails_closed_on_spoofed_counts() -> None:
    output = deepcopy(_output())
    output["result"]["dataset_id"] = "attacker-result"
    output["result"]["row_count"] = 25

    text, tables = render_tool_output("apply_automatic_tree", output)

    assert "自动树全量写回结果完整性校验失败" in text
    assert "attacker-result" not in text
    assert output["source"]["asset_id"] not in text
    assert tables == []
