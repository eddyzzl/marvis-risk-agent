from marvis.agent.renderers import render_tool_output


def _output(*, sections=None):
    return {
        "dataset_id": "dataset-1",
        "dataset_content_hash": "a" * 64,
        "expected_content_hash": "a" * 64,
        "workspace_revision": 2,
        "analysis_generation": 1,
        "semantic_mapping_hash": "b" * 64,
        "scan_scope": "full_dataset",
        "row_count": 4,
        "row_count_scanned": 4,
        "options_echo": {
            "sections": sections
            or ["overview", "target", "missing", "distribution", "correlation"],
            "columns": None,
            "target_col": "bad",
        },
        "semantics": {
            "target_col": "bad",
            "field_roles": {"bad": "target", "score": "score"},
            "business_names": {"bad": "风险标签", "score": "模型分"},
        },
        "result": {
            "schema_version": "data-analysis.v1",
            "config": {},
            "dataset": {
                "row_count": 4,
                "column_count": 3,
                "numeric_column_count": 2,
            },
            "fields": [
                {
                    "name": "score",
                    "duckdb_type": "DOUBLE",
                    "kind": "numeric",
                    "row_count": 4,
                    "null_count": 1,
                    "null_rate": 0.25,
                    "distinct_count": 3,
                    "numeric": {
                        "min": 500.0,
                        "max": 700.0,
                        "mean": 600.0,
                        "p25": 550.0,
                        "p50": 600.0,
                        "p75": 650.0,
                    },
                    "frequency": {"items": [], "other_count": 0},
                    "histogram": {"bins": []},
                },
                {
                    "name": "bad",
                    "duckdb_type": "BIGINT",
                    "kind": "numeric",
                    "row_count": 4,
                    "null_count": 0,
                    "null_rate": 0.0,
                    "distinct_count": 2,
                    "numeric": None,
                    "frequency": {
                        "items": [
                            {
                                "value": {"type": "int", "value": 0},
                                "count": 3,
                                "rate_all": 0.75,
                            },
                            {
                                "value": {"type": "int", "value": 1},
                                "count": 1,
                                "rate_all": 0.25,
                            },
                        ],
                        "other_count": 0,
                    },
                    "histogram": {"bins": []},
                },
                {
                    "name": "customer_name",
                    "duckdb_type": "VARCHAR",
                    "kind": "string",
                    "row_count": 4,
                    "null_count": 0,
                    "null_rate": 0.0,
                    "distinct_count": 3,
                    "numeric": None,
                    "frequency": {
                        "items": [
                            {
                                "value": {"type": "string", "value": "token:1234"},
                                "count": 2,
                                "rate_all": 0.5,
                            }
                        ],
                        "other_count": 2,
                    },
                    "histogram": None,
                },
            ],
            "target_distribution": {
                "status": "available",
                "column": "bad",
                "frequency": {
                    "items": [
                        {
                            "value": {"type": "int", "value": 0},
                            "count": 3,
                            "rate_all": 0.75,
                        },
                        {
                            "value": {"type": "int", "value": 1},
                            "count": 1,
                            "rate_all": 0.25,
                        },
                    ]
                },
            },
            "correlations": {
                "method": "pearson",
                "basis": "pairwise_finite",
                "columns": ["score", "bad"],
                "values": [[1.0, None], [None, None]],
                "pair_counts": [[3, 3], [3, 4]],
                "reasons": [["ok", "zero_variance_right"], ["zero_variance_left", "zero_variance_both"]],
            },
        },
    }


def test_profile_dataset_renderer_surfaces_report_ready_sections_without_recomputing():
    text, tables = render_tool_output("profile_dataset", _output())

    assert "全量扫描 4 行" in text
    assert "dataset-1" in text
    by_title = {table["title"]: table for table in tables}
    assert {"字段概览", "缺失分析", "Target 分布", "字段分布", "相关矩阵"} <= set(by_title)
    assert by_title["字段概览"]["rows"][0][0] == "score（模型分）"
    assert by_title["缺失分析"]["rows"][0][2] == "25.0%"
    assert by_title["Target 分布"]["rows"] == [["0", "3", "75.0%"], ["1", "1", "25.0%"]]
    correlation = by_title["相关矩阵"]
    assert correlation["rows"][0][2] == "n/a（右侧常量，n=3）"
    assert correlation["rows"][1][1] == "n/a（左侧常量，n=3）"
    assert "张三" not in str(tables)
    assert "token:1234" in str(tables)


def test_profile_dataset_renderer_honors_requested_sections():
    text, tables = render_tool_output(
        "profile_dataset",
        _output(sections=["missing"]),
    )

    assert "缺失分析" in {table["title"] for table in tables}
    assert {table["title"] for table in tables} == {"缺失分析"}
    assert "按请求展示：缺失" in text


def test_profile_dataset_renderer_explains_unsafe_numeric_correlation():
    output = _output(sections=["correlation"])
    output["result"]["correlations"]["reasons"][0][1] = (
        "unsafe_numeric_precision_right"
    )

    _text, tables = render_tool_output("profile_dataset", output)

    correlation = next(table for table in tables if table["title"] == "相关矩阵")
    assert correlation["rows"][0][2] == "n/a（右侧数值精度不安全，n=3）"
