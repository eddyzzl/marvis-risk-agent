from marvis.agent.renderers import render_tool_output


def test_profit_renderer_surfaces_economics_and_artifacts():
    text, tables = render_tool_output(
        "profit_calc",
        {
            "results": [
                {
                    "segment": "A",
                    "count": 2,
                    "revenue": 100,
                    "expected_loss": 20,
                    "funding_cost": 10,
                    "operating_cost": 5,
                    "net_profit": 65,
                    "roa": 0.065,
                }
            ],
            "quality_warnings": [],
            "artifacts": [
                    {
                        "artifact_id": "task-artifact-profit",
                        "kind": "profit_csv",
                        "filename": "profit.csv",
                }
            ],
        },
    )

    assert "利润分析完成" in text
    assert "可在策略产物卡下载" in text
    assert tables[0]["columns"] == [
        "分群", "样本数", "收入", "预期损失", "资金成本", "运营成本", "净利润", "ROA"
    ]


def test_roll_rate_renderer_surfaces_semantics_matrix_and_warnings():
    text, tables = render_tool_output(
        "roll_rate_matrix",
        {
            "states": ["C", "M1"],
            "matrix": [[0.8, 0.2], [0.1, 0.9]],
            "base_counts": {"C": 100, "M1": 20},
            "period": "month",
            "observation_semantics": "adjacent_observation",
            "data_quality_warnings": [{"code": "missing_month", "message": "存在跨月间隔"}],
            "artifacts": [
                    {
                        "artifact_id": "task-artifact-roll",
                        "kind": "roll_rate_csv",
                        "filename": "roll.csv",
                }
            ],
        },
    )

    assert "相邻观测" in text
    assert "1 条质量提示" in text
    assert "策略产物卡下载" in text
    assert tables[0]["columns"] == ["期初状态", "基数", "C", "M1"]


def test_standard_renderer_does_not_claim_unregistered_file_is_downloadable():
    text, _tables = render_tool_output(
        "profit_calc",
        {
            "results": [],
            "quality_warnings": [],
            "artifacts": [{"kind": "profit_csv", "filename": "unregistered.csv"}],
        },
    )

    assert "尚未登记下载" in text
    assert "可在策略产物卡下载" not in text
