from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _output(*, reason: str | None = "人工确认用于入池前复核") -> dict:
    return {
        "schema_version": (
            "strategy.materialize-interactive-tree-frontier-selection-tool.v1"
        ),
        "selection_id": "interactive-tree-frontier-selection-" + "1" * 32,
        "selection_hash": "2" * 64,
        "selection_reason": reason,
        "revision_id": "interactive-tree-revision-" + "3" * 32,
        "semantic_tree_id": "interactive-tree-" + "4" * 32,
        "tree_hash": "5" * 64,
        "source_node_id": "node-" + "6" * 20,
        "leaf_id": "interactive-leaf-" + "7" * 32,
        "fragment_id": "candidate-fragment-" + "8" * 32,
        "fragment_hash": "9" * 64,
        "rule_id": "candidate-rule-" + "a" * 32,
        "effect_id": "candidate-effect-" + "b" * 32,
        "artifacts": [
            {
                "artifact_id": "artifact-interactive-tree-frontier-selection",
                "kind": "strategy_interactive_tree_frontier_selection_json",
                "format": "json",
                "filename": "interactive-tree-frontier-selection.json",
                "content_hash": "c" * 64,
                "download_url": (
                    "/api/tasks/task-1/task-artifacts/"
                    "artifact-interactive-tree-frontier-selection/download"
                ),
            }
        ],
    }


def test_interactive_tree_frontier_renderer_surfaces_pointer_identity_and_boundary() -> (
    None
):
    output = _output()

    text, tables = render_tool_output(
        "materialize_interactive_tree_frontier_selection",
        output,
    )

    assert "pointer-only" in text
    assert "精确修订版本" in text
    assert "没有复制条件、指标或动作" in text
    assert "未入池" in text
    assert "未配置动作" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert text.count("interactive-tree-frontier-selection.json") == 1
    assert text.count("/api/tasks/task-1/task-artifacts/") == 1
    assert "最佳" not in text
    assert "最优" not in text
    assert "推荐" not in text

    [identity] = tables
    assert identity == {
        "title": "交互式决策树前沿节点引用",
        "columns": ["字段", "值"],
        "rows": [
            ["Selection ID", output["selection_id"]],
            ["Selection Hash", output["selection_hash"]],
            ["Revision ID", output["revision_id"]],
            ["Semantic Tree ID", output["semantic_tree_id"]],
            ["Tree Hash", output["tree_hash"]],
            ["Source Node ID", output["source_node_id"]],
            ["Leaf ID", output["leaf_id"]],
            ["Fragment ID", output["fragment_id"]],
            ["Fragment Hash", output["fragment_hash"]],
            ["Rule ID", output["rule_id"]],
            ["Effect ID", output["effect_id"]],
            ["Artifact ID", output["artifacts"][0]["artifact_id"]],
            ["Artifact Content Hash", output["artifacts"][0]["content_hash"]],
            ["Selection Reason", output["selection_reason"]],
        ],
    }


def test_interactive_tree_frontier_renderer_marks_absent_optional_reason() -> None:
    text, tables = render_tool_output(
        "materialize_interactive_tree_frontier_selection",
        _output(reason=None),
    )

    assert "pointer-only" in text
    assert tables[0]["rows"][-1] == ["Selection Reason", "未提供"]
