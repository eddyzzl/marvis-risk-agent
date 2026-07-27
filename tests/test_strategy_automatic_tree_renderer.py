from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def test_automatic_tree_renderer_keeps_leaf_order_and_surfaces_all_deliveries() -> None:
    asset_id = "candidate-asset-" + "a" * 32
    output = {
        "summary": {
            "asset_id": asset_id,
            "asset_hash": "b" * 64,
            "tree_result_hash": "c" * 64,
            "candidate_stage": "development",
            "observation_stage": "backtested",
            "validation_status": "unvalidated",
            "leaf_count": 2,
        },
        "leaf_index": [
            {
                "leaf_id": "leaf-second",
                "rule_id": "rule-second",
                "metric_basis": {
                    "primary": "unweighted",
                    "sample_weight": {"status": "not_applicable"},
                },
                "measurements": {
                    "unweighted": {
                        "total": 8,
                        "share": 0.4,
                        "bad_rate": 0.25,
                        "bad_capture": 0.2,
                        "lift": 0.5,
                    },
                    "weighted": {"status": "not_applicable"},
                },
                "condition": {"feature": "score", "operator": ">", "value": 500},
            },
            {
                "leaf_id": "leaf-first",
                "rule_id": "rule-first",
                "metric_basis": {
                    "primary": "weighted",
                    "sample_weight": {"status": "available", "column": "weight"},
                },
                "measurements": {
                    "weighted": {
                        "total": 12.5,
                        "share": 0.6,
                        "bad_rate": 0.5,
                        "bad_capture": 0.8,
                        "lift": 1.5,
                    },
                    "unweighted": {
                        "total": 12,
                        "share": 0.6,
                        "bad_rate": 0.5,
                        "bad_capture": 0.8,
                        "lift": 1.5,
                    },
                },
                "condition": {"feature": "score", "operator": "<=", "value": 500},
            },
        ],
        "report_info_gaps": [
            {
                "code": "sample_weight_not_provided",
                "context": "sample_weight",
                "blocking": False,
            },
            {
                "code": "loan_amount_not_provided",
                "context": "loan_amount",
                "blocking": False,
            },
        ],
        "red_flags": [
            {
                "code": "direction_violation",
                "node_id": "node-root",
                "feature": "score",
                "expected_direction": "decreasing",
            },
            {
                "code": "direction_violation",
                "node_id": "node-right",
                "feature": "income",
                "expected_direction": "increasing",
            },
        ],
        "artifacts": [
            {
                "filename": f"tree.{extension}",
                "download_url": f"/api/tasks/t/task-artifacts/{index}/download",
            }
            for index, extension in enumerate(
                ("json", "py", "sql", "svg", "png", "xlsx"), start=1
            )
        ],
    }

    text, tables = render_tool_output("build_automatic_tree_candidate", output)

    assert asset_id in text
    assert "b" * 64 in text
    assert "c" * 64 in text
    assert "development / backtested / unvalidated" in text
    assert "尚未选叶" in text
    assert "未入池" in text
    assert "未采纳、未部署" in text
    assert "暂时没有可跳过，最终报告留空" in text
    assert "风险方向红旗" in text
    assert "诊断期望，不是强制分裂约束" in text
    assert all(
        f"tree.{extension}" in text
        for extension in ("json", "py", "sql", "svg", "png", "xlsx")
    )
    assert "最佳" not in text
    assert "最优" not in text
    assert [table["title"] for table in tables] == [
        "自动树风险方向红旗",
        "自动树完整叶节点清单",
    ]
    assert tables[0]["rows"] == [
        ["direction_violation", "node-root", "score", "递减"],
        ["direction_violation", "node-right", "income", "递增"],
    ]
    assert [row[0] for row in tables[1]["rows"]] == ["leaf-second", "leaf-first"]
    assert tables[1]["rows"][0][2:8] == [
        "unweighted",
        "8",
        "40.0%",
        "25.0%",
        "20.0%",
        "0.5000",
    ]


def test_automatic_tree_renderer_tolerates_missing_optional_leaf_measurements() -> None:
    text, tables = render_tool_output(
        "build_automatic_tree_candidate",
        {
            "summary": {
                "asset_id": "candidate-asset-" + "d" * 32,
                "candidate_stage": "development",
                "observation_stage": "backtested",
                "validation_status": "unvalidated",
            },
            "leaf_index": [
                {
                    "leaf_id": "leaf-only",
                    "rule_id": "rule-only",
                    "metric_basis": {"primary": "unweighted"},
                    "measurements": {"unweighted": {}},
                    "condition": {},
                }
            ],
            "report_info_gaps": [],
            "red_flags": [],
            "artifacts": [],
        },
    )

    assert "尚未选叶" in text
    assert tables[0]["rows"][0][3:8] == ["n/a", "n/a", "n/a", "n/a", "n/a"]
