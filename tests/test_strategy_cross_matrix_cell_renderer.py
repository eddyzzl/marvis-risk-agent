from __future__ import annotations

from marvis.agent.renderers import render_tool_output


CELL_A = "cross-cell-" + "1" * 32
CELL_B = "cross-cell-" + "2" * 32


def _output(*, reason: str | None = "人工确认用于风险复核") -> dict:
    return {
        "selection_id": "cross-matrix-cell-selection-" + "3" * 32,
        "selection_hash": "4" * 64,
        "selection_reason": reason,
        "group_id": "cross-cell-group-" + "5" * 32,
        "cell_ids": [CELL_B, CELL_A],
        "source_asset_id": "candidate-asset-" + "6" * 32,
        "source_asset_hash": "7" * 64,
        "source_candidate_id": "candidate-" + "8" * 32,
        "source_evidence_hash": "9" * 64,
        "fragment_id": "candidate-fragment-" + "a" * 32,
        "fragment_type": "cross_matrix_cell_group",
        "rule_id": "candidate-rule-" + "b" * 32,
        "effect_id": "candidate-effect-" + "c" * 32,
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
        "artifacts": [
            {
                "artifact_id": "artifact-cross-cell-selection",
                "kind": "strategy_cross_matrix_cell_selection_json",
                "format": "json",
                "filename": "cross-matrix-cell-selection.json",
                "content_hash": "d" * 64,
                "download_url": (
                    "/api/tasks/task-1/task-artifacts/"
                    "artifact-cross-cell-selection/download"
                ),
            }
        ],
    }


def test_cross_cell_renderer_surfaces_pointer_identity_and_source_order() -> None:
    output = _output()

    text, tables = render_tool_output(
        "materialize_cross_matrix_cell_selection",
        output,
    )

    assert "不可变引用" in text
    assert "确定性 OR" in text
    assert "不复制观测指标" in text
    assert "不执行排名或推荐" in text
    assert "不生成业务动作" in text
    assert "未入池" in text
    assert "未应用" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert text.count("cross-matrix-cell-selection.json") == 1
    assert text.count("/api/tasks/task-1/task-artifacts/") == 1

    identity, cells = tables
    assert identity["title"] == "Cross Matrix 精确单元格选择引用"
    assert ["Selection ID", output["selection_id"]] in identity["rows"]
    assert ["Source Asset ID", output["source_asset_id"]] in identity["rows"]
    assert ["Rule ID", output["rule_id"]] in identity["rows"]
    assert ["Selection Reason", output["selection_reason"]] in identity["rows"]
    assert cells == {
        "title": "已选择 Cell IDs（源矩阵顺序）",
        "columns": ["顺序", "Cell ID"],
        "rows": [["1", CELL_B], ["2", CELL_A]],
    }


def test_cross_cell_renderer_marks_absent_optional_reason() -> None:
    text, tables = render_tool_output(
        "materialize_cross_matrix_cell_selection",
        _output(reason=None),
    )

    assert "未入池" in text
    assert ["Selection Reason", "未提供"] in tables[0]["rows"]
