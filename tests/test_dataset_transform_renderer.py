from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _output() -> dict:
    return {
        "schema_version": "data-transform-tool-result.v1",
        "run_id": "transform-run-1",
        "source_dataset_id": "dataset-source",
        "result_dataset_id": "dataset-result",
        "result_content_hash": "c" * 64,
        "row_count_before": 137,
        "row_count_after": 101,
        "column_count_before": 8,
        "column_count_after": 9,
        "operations": [
            {"op": "rename_columns", "mapping": {"bad": "label"}},
            {"op": "fill_missing", "fills": [{"column": "amount"}]},
            {"op": "filter_rows", "predicate": {"op": "gte"}},
        ],
        "steps": [
            {
                "step": 1,
                "op": "rename_columns",
                "row_count_before": 137,
                "row_count_after": 137,
                "row_delta": 0,
                "columns_before": [
                    {"name": "bad", "duckdb_type": "BIGINT"},
                    {"name": "amount", "duckdb_type": "DOUBLE"},
                ],
                "columns_after": [
                    {"name": "label", "duckdb_type": "BIGINT"},
                    {"name": "amount", "duckdb_type": "DOUBLE"},
                ],
                "impact": {
                    "renamed_count": 1,
                    "mapping": {"bad": "label"},
                },
            },
            {
                "step": 2,
                "op": "fill_missing",
                "row_count_before": 137,
                "row_count_after": 137,
                "row_delta": 0,
                "columns_before": [],
                "columns_after": [],
                "impact": {
                    "columns": ["amount"],
                    "filled_count": 11,
                    "by_column": {
                        "amount": {
                            "missing_before": 11,
                            "missing_after": 0,
                            "filled_count": 11,
                        }
                    },
                },
            },
            {
                "step": 3,
                "op": "filter_rows",
                "row_count_before": 137,
                "row_count_after": 101,
                "row_delta": -36,
                "columns_before": [],
                "columns_after": [],
                "impact": {"kept_rows": 101, "removed_rows": 36},
            },
        ],
        "semantic_migration": {
            "before_hash": "a" * 64,
            "after_hash": "b" * 64,
            "renamed_fields": {"bad": "label"},
            "dropped_fields": ["temporary_flag"],
            "dropped_protected_fields": [],
        },
        "workspace": {
            "source_revision": 4,
            "result_revision": 5,
            "source_analysis_generation": 2,
            "result_analysis_generation": 3,
        },
        "lineage": {
            "parent_dataset_id": "dataset-source",
            "child_dataset_id": "dataset-result",
            "relation_kind": "transform",
            "edge_order": 0,
        },
        "evidence_artifact_id": "artifact-transform-1",
        "evidence_download_url": (
            "/api/tasks/task-1/task-artifacts/artifact-transform-1/download"
        ),
        "cached": False,
    }


def test_transform_renderer_surfaces_only_the_tool_evidence_contract():
    text, tables = render_tool_output("transform_dataset", _output())

    assert "dataset-source" in text
    assert "dataset-result" in text
    assert "137 行 / 8 列" in text
    assert "101 行 / 9 列" in text
    by_title = {table["title"]: table for table in tables}
    assert {
        "加工步骤影响",
        "字段语义迁移",
        "Workspace 版本迁移",
        "数据血缘",
        "证据与下载",
    } <= set(by_title)

    step_rows = by_title["加工步骤影响"]["rows"]
    assert step_rows[0] == ["1", "重命名字段", "137", "137", "0", "重命名 1 个：bad → label"]
    assert step_rows[1][-1] == "填补 11 个缺失值：amount"
    assert step_rows[2][-1] == "保留 101 行；移除 36 行"

    semantic_rows = by_title["字段语义迁移"]["rows"]
    assert ["重命名字段", "bad → label"] in semantic_rows
    assert ["删除字段", "temporary_flag"] in semantic_rows
    workspace_rows = by_title["Workspace 版本迁移"]["rows"]
    assert workspace_rows == [["Revision", "4", "5"], ["分析代次", "2", "3"]]
    assert by_title["数据血缘"]["rows"] == [
        ["dataset-source", "dataset-result", "transform", "0"]
    ]
    evidence_rows = by_title["证据与下载"]["rows"]
    assert ["结果 SHA-256", "c" * 64] in evidence_rows
    assert ["证据产物", "artifact-transform-1"] in evidence_rows
    assert [
        "下载地址",
        "/api/tasks/task-1/task-artifacts/artifact-transform-1/download",
    ] in evidence_rows


def test_transform_renderer_warns_when_a_protected_field_was_explicitly_dropped():
    output = _output()
    output["semantic_migration"]["dropped_protected_fields"] = ["label"]
    output["cached"] = True

    text, tables = render_tool_output("transform_dataset", output)

    assert "复用已验证证据" in text
    assert "受保护字段" in text
    semantics = next(table for table in tables if table["title"] == "字段语义迁移")
    assert ["已确认删除受保护字段", "label"] in semantics["rows"]
