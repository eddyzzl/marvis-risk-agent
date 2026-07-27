from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def test_univariate_renderer_surfaces_boundary_rankings_and_downloads():
    output = {
        "candidate_id": "candidate-" + "a" * 32,
        "feature_count": 2,
        "available_method_count": 5,
        "nan_labels_dropped": 3,
        "rankings": [
            {
                "feature": "score",
                "method": "tree",
                "iv": 0.42,
                "ks": 0.31,
                "auc": 0.73,
            }
        ],
        "red_flags": [
            "loan_amount_metrics_unavailable:column_not_configured",
            "income.equal_width:red_flag:min_bin_pct_not_achieved",
        ],
        "artifacts": [
            {
                "filename": "candidate.json",
                "download_url": "/api/tasks/t/task-artifacts/a/download",
            },
            {
                "filename": "candidate.xlsx",
                "download_url": "/api/tasks/t/task-artifacts/b/download",
            },
        ],
    }

    text, tables = render_tool_output("analyze_univariate_candidates", output)

    assert "development / unvalidated" in text
    assert "不代表独立验证、采纳或上线" in text
    assert "3 行空标签" in text
    assert "如能提供" in text
    assert "candidate.xlsx" in text
    assert [table["title"] for table in tables] == [
        "单变量候选排名（前20）",
        "候选分析提示",
    ]
    assert tables[0]["rows"] == [["score", "tree", "0.4200", "0.3100", "0.7300"]]


def test_univariate_renderer_does_not_claim_missing_amount_metrics():
    text, tables = render_tool_output(
        "analyze_univariate_candidates",
        {
            "candidate_id": "candidate-" + "b" * 32,
            "feature_count": 1,
            "available_method_count": 1,
            "rankings": [],
            "red_flags": [
                "loan_amount_metrics_unavailable:column_not_configured",
                "overdue_amount_metrics_unavailable:column_not_configured",
            ],
            "artifacts": [],
        },
    )

    assert "尚未配置放款金额列" in text
    assert "尚未配置逾期金额列" in text
    assert tables == []
