from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _output(*, reason: str | None = "人工确认 leaf-node-7") -> dict:
    return {
        "selection_id": "automatic-tree-leaf-selection-" + "1" * 32,
        "selection_hash": "2" * 64,
        "selection_reason": reason,
        "tree_asset_id": "candidate-asset-" + "3" * 32,
        "tree_asset_hash": "4" * 64,
        "tree_result_hash": "5" * 64,
        "leaf_id": "leaf-node-7",
        "fragment_id": "candidate-fragment-" + "6" * 32,
        "fragment_hash": "7" * 64,
        "rule_id": "rule-node-7",
        "effect_id": "candidate-effect-" + "8" * 32,
        "artifacts": [
            {
                "artifact_id": "artifact-leaf-selection",
                "kind": "strategy_automatic_tree_leaf_fragment_json",
                "format": "json",
                "filename": "automatic-tree-leaf-selection.json",
                "content_hash": "9" * 64,
                "download_url": (
                    "/api/tasks/task-1/task-artifacts/artifact-leaf-selection/download"
                ),
            }
        ],
    }


def test_automatic_tree_leaf_renderer_surfaces_pointer_identity_and_one_download() -> (
    None
):
    output = _output()

    text, tables = render_tool_output(
        "materialize_automatic_tree_leaf_fragment",
        output,
    )

    assert "pointer-only" in text
    assert "没有复制规则、指标或动作" in text
    assert "未入池" in text
    assert "未配置动作" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert "automatic-tree-leaf-selection.json" in text
    assert text.count("automatic-tree-leaf-selection.json") == 1
    assert text.count("/api/tasks/task-1/task-artifacts/") == 1
    assert "最佳" not in text
    assert "最优" not in text
    assert "推荐" not in text
    assert "采用" not in text

    [identity] = tables
    assert identity["title"] == "自动树精确叶节点引用"
    assert identity["columns"] == ["字段", "值"]
    assert identity["rows"] == [
        ["Selection ID", output["selection_id"]],
        ["Selection Hash", output["selection_hash"]],
        ["Tree Asset ID", output["tree_asset_id"]],
        ["Tree Asset Hash", output["tree_asset_hash"]],
        ["Tree Result Hash", output["tree_result_hash"]],
        ["Leaf ID", output["leaf_id"]],
        ["Fragment ID", output["fragment_id"]],
        ["Fragment Hash", output["fragment_hash"]],
        ["Rule ID", output["rule_id"]],
        ["Effect ID", output["effect_id"]],
        ["Artifact ID", output["artifacts"][0]["artifact_id"]],
        ["Artifact Content Hash", output["artifacts"][0]["content_hash"]],
        ["Selection Reason", output["selection_reason"]],
    ]


def test_automatic_tree_leaf_renderer_marks_absent_optional_reason() -> None:
    output = _output(reason=None)

    text, tables = render_tool_output(
        "materialize_automatic_tree_leaf_fragment",
        output,
    )

    assert "pointer-only" in text
    assert tables[0]["rows"][-1] == ["Selection Reason", "未提供"]
