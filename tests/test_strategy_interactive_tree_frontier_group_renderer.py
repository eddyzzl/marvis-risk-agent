"""Renderer contract for pointer-only interactive-tree frontier OR groups."""

from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _output(*, reason: str | None = "人工确认任一成员命中即进入候选规则") -> dict:
    return {
        "schema_version": (
            "strategy.materialize-interactive-tree-frontier-group-selection-tool.v1"
        ),
        "selection_id": (
            "interactive-tree-frontier-group-selection-" + "1" * 32
        ),
        "selection_hash": "2" * 64,
        "group_id": "interactive-tree-frontier-group-" + "3" * 32,
        "selection_reason": reason,
        "revision_id": "interactive-tree-revision-" + "4" * 32,
        "semantic_tree_id": "interactive-tree-" + "5" * 32,
        "tree_hash": "6" * 64,
        "source_node_ids": [
            "node-" + "7" * 20,
            "leaf-" + "8" * 20,
        ],
        "member_count": 2,
        "fragment_id": "candidate-fragment-" + "9" * 32,
        "rule_id": "candidate-rule-" + "a" * 32,
        "effect_id": "candidate-effect-" + "b" * 32,
        "artifacts": [
            {
                "artifact_id": (
                    "artifact-interactive-tree-frontier-group-selection"
                ),
                "kind": (
                    "strategy_interactive_tree_frontier_group_selection_json"
                ),
                "format": "json",
                "filename": "interactive-tree-frontier-group-selection.json",
                "content_hash": "c" * 64,
                "download_url": (
                    "/api/tasks/task-1/task-artifacts/"
                    "artifact-interactive-tree-frontier-group-selection/download"
                ),
            }
        ],
    }


def test_frontier_group_renderer_surfaces_or_members_and_safety_boundary() -> None:
    output = _output()

    text, tables = render_tool_output(
        "materialize_interactive_tree_frontier_group_selection",
        output,
    )

    assert "pointer-only" in text
    assert "OR" in text
    assert "2" in text
    assert "没有复制条件、指标或动作" in text
    assert "未入池" in text
    assert "未配置动作" in text
    assert "未应用" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert text.count("interactive-tree-frontier-group-selection.json") == 1
    assert text.count("/api/tasks/task-1/task-artifacts/") == 1
    assert "最佳" not in text
    assert "最优" not in text
    assert "推荐" not in text

    [identity] = tables
    assert identity == {
        "title": "交互式决策树前沿 OR 分组引用",
        "columns": ["字段", "值"],
        "rows": [
            ["Selection ID", output["selection_id"]],
            ["Selection Hash", output["selection_hash"]],
            ["Group ID", output["group_id"]],
            ["Revision ID", output["revision_id"]],
            ["Semantic Tree ID", output["semantic_tree_id"]],
            ["Tree Hash", output["tree_hash"]],
            ["Member Count", "2"],
            ["Source Node IDs", "、".join(output["source_node_ids"])],
            ["Fragment ID", output["fragment_id"]],
            ["Rule ID", output["rule_id"]],
            ["Effect ID", output["effect_id"]],
            ["Artifact ID", output["artifacts"][0]["artifact_id"]],
            ["Artifact Content Hash", output["artifacts"][0]["content_hash"]],
            ["Selection Reason", output["selection_reason"]],
        ],
    }


def test_frontier_group_renderer_marks_absent_optional_reason() -> None:
    text, tables = render_tool_output(
        "materialize_interactive_tree_frontier_group_selection",
        _output(reason=None),
    )

    assert "pointer-only" in text
    assert tables[0]["rows"][-1] == ["Selection Reason", "未提供"]
