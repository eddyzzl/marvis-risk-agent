from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def test_model_score_comparison_has_a_nonselecting_evidence_presenter() -> None:
    text, tables = render_tool_output(
        "materialize_model_score_comparison_v2",
        {
            "population": "risk",
            "partition": "development",
            "comparison_id": "strategy-model-comparison-" + "a" * 24,
            "comparison": {
                "metrics": [
                    {
                        "metric_key": "auc",
                        "period": None,
                        "unit": "ratio",
                        "model_values": [
                            {
                                "model_evidence_ref": {
                                    "evidence_id": "model-a",
                                    "content_hash": "1" * 64,
                                },
                                "value": 0.71,
                            },
                            {
                                "model_evidence_ref": {
                                    "evidence_id": "model-b",
                                    "content_hash": "2" * 64,
                                },
                                "value": 0.74,
                            },
                        ],
                        "delta": 0.03,
                    }
                ],
                "selection": {"status": "no_selection"},
            },
            "governance": {
                "selection_status": "no_selection",
                "winner_selected": False,
                "not_adopted": True,
                "not_deployed": True,
            },
            "artifact": {
                "filename": "comparison.json",
                "download_url": "/api/tasks/task-1/task-artifacts/a/download",
            },
        },
    )

    assert "模型评分比较证据已生成" in text
    assert "risk / development" in text
    assert "未选择冠军" in text
    assert "未采纳、未部署" in text
    assert "[comparison.json](/api/tasks/task-1/task-artifacts/a/download)" in text
    assert tables == [
        {
            "title": "同样本模型评分指标",
            "columns": ["指标", "期间", "单位", "model-a", "model-b", "差值"],
            "rows": [["auc", "整体", "ratio", "0.7100", "0.7400", "0.0300"]],
        }
    ]


def test_model_score_comparison_fails_closed_when_payload_claims_a_winner() -> None:
    text, tables = render_tool_output(
        "materialize_model_score_comparison_v2",
        {
            "winner_selected": True,
            "comparison": {"selection": {"status": "winner_selected"}},
            "governance": {
                "selection_status": "winner_selected",
                "winner_selected": True,
                "not_adopted": False,
                "not_deployed": False,
            },
            "artifact": {"filename": "untrusted.json"},
        },
    )

    assert "模型评分比较证据完整性校验失败" in text
    assert "已完成" not in text
    assert "winner_selected" not in text
    assert "untrusted.json" not in text
    assert tables == []


def test_model_score_comparison_fails_closed_when_envelope_is_incomplete() -> None:
    text, tables = render_tool_output(
        "materialize_model_score_comparison_v2",
        {"winner_selected": True},
    )

    assert "模型评分比较证据完整性校验失败" in text
    assert "已完成" not in text
    assert "winner_selected" not in text
    assert tables == []
