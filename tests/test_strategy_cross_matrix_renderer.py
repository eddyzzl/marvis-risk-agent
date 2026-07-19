from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _cell(
    cell_id: str,
    row_bin_id: str,
    column_bin_id: str,
    *,
    count: int,
    good: int,
    bad: int,
    share: float,
    bad_rate: float | None,
    lift: float | None,
    woe: float | None,
    iv: float,
    amount_metrics: dict | None = None,
) -> dict:
    return {
        "cell_id": cell_id,
        "row_bin_id": row_bin_id,
        "column_bin_id": column_bin_id,
        "effect": {
            "count": count,
            "good": good,
            "bad": bad,
            "share": share,
            "bad_rate": bad_rate,
            "lift": lift,
            "woe": woe,
            "iv_contribution": iv,
            "amount_metrics": amount_metrics or {},
        },
    }


def _output() -> dict:
    return {
        "schema_version": "strategy.build-cross-matrix-candidate-tool.v1",
        "asset_id": "candidate-asset-" + "a" * 32,
        "asset_hash": "b" * 64,
        "candidate_id": "candidate-" + "e" * 32,
        "evidence_hash": "f" * 64,
        "parent_candidate_id": "candidate-" + "c" * 32,
        "parent_evidence_hash": "d" * 64,
        "dataset_id": "dataset-cross-source",
        "target_col": "bad",
        "population_count": 105,
        "labeled_count": 100,
        "drop_nan_labels": True,
        "nan_labels_dropped": 5,
        "row_axis": {
            "feature": "age",
            "method": "equal_frequency",
            "bin_count": 2,
        },
        "column_axis": {
            "feature": "score",
            "method": "equal_width",
            "bin_count": 2,
        },
        "cell_count": 4,
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
        "cross_matrix_candidate": {
            "axes": [
                {
                    "position": "row",
                    "bins": [
                        {"bin_id": "row-bin-0", "source_bin_id": "regular:0"},
                        {"bin_id": "row-bin-1", "source_bin_id": "regular:1"},
                    ],
                },
                {
                    "position": "column",
                    "bins": [
                        {"bin_id": "column-bin-0", "source_bin_id": "regular:0"},
                        {"bin_id": "column-bin-1", "source_bin_id": "regular:1"},
                    ],
                },
            ],
            "matrix": {
                "cells": [
                    _cell(
                        "cross-cell-1",
                        "row-bin-0",
                        "column-bin-0",
                        count=30,
                        share=0.3,
                        good=27,
                        bad=3,
                        bad_rate=0.1,
                        lift=0.5,
                        woe=-0.7,
                        iv=0.08,
                        amount_metrics={
                            "loan_amount": {
                                "status": "available",
                                "covered_count": 29,
                                "coverage_rate": 29 / 30,
                                "value": 120000.0,
                                "reason": None,
                            },
                            "overdue_amount": {
                                "status": "available",
                                "covered_count": 28,
                                "coverage_rate": 28 / 30,
                                "value": 6000.0,
                                "reason": None,
                            },
                            "overdue_rate": {
                                "status": "available",
                                "covered_count": 27,
                                "coverage_rate": 0.9,
                                "value": 0.05,
                                "reason": None,
                            },
                        },
                    ),
                    _cell(
                        "cross-cell-2",
                        "row-bin-0",
                        "column-bin-1",
                        count=0,
                        share=0.0,
                        good=0,
                        bad=0,
                        bad_rate=None,
                        lift=None,
                        woe=0.19105523676270922,
                        iv=0.0017487893525190772,
                        amount_metrics={
                            "loan_amount": {
                                "status": "available",
                                "covered_count": 0,
                                "coverage_rate": None,
                                "value": 0.0,
                                "reason": None,
                            },
                            "overdue_amount": {
                                "status": "unavailable",
                                "covered_count": None,
                                "coverage_rate": None,
                                "value": None,
                                "reason": "column_not_configured",
                            },
                            "overdue_rate": {
                                "status": "not_applicable",
                                "covered_count": 0,
                                "coverage_rate": None,
                                "value": None,
                                "reason": "no_observations",
                            },
                        },
                    ),
                    _cell(
                        "cross-cell-3",
                        "row-bin-1",
                        "column-bin-0",
                        count=35,
                        share=0.35,
                        good=25,
                        bad=10,
                        bad_rate=10 / 35,
                        lift=1.2,
                        woe=0.2,
                        iv=0.03,
                    ),
                    _cell(
                        "cross-cell-4",
                        "row-bin-1",
                        "column-bin-1",
                        count=35,
                        share=0.35,
                        good=20,
                        bad=15,
                        bad_rate=15 / 35,
                        lift=1.8,
                        woe=0.8,
                        iv=0.12,
                    ),
                ]
            }
        },
        "artifacts": [
            {
                "artifact_id": "artifact-cross-json",
                "kind": "strategy_cross_matrix_candidate_json",
                "format": "json",
                "filename": "cross-matrix.json",
                "content_hash": "e" * 64,
                "download_url": "/api/tasks/t/task-artifacts/a/download",
            }
        ],
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def test_cross_matrix_renderer_surfaces_exact_boundary_scope_and_download() -> None:
    output = _output()

    text, tables = render_tool_output("build_cross_matrix_candidate", output)

    assert output["asset_id"] in text
    assert output["asset_hash"] in text
    assert output["candidate_id"] in text
    assert output["evidence_hash"] in text
    assert output["parent_candidate_id"] in text
    assert output["parent_evidence_hash"] in text
    assert "age / equal_frequency / 2 bins" in text
    assert "score / equal_width / 2 bins" in text
    assert "4 个完整单元格" in text
    assert "development / backtested / unvalidated" in text
    assert "未选择格子" in text
    assert "未入池" in text
    assert "未应用写回" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert "绑定样本观测（未独立验证）" in text
    assert "dataset `dataset-cross-source`" in text
    assert "target `bad`" in text
    assert "population 105" in text
    assert "labeled 100" in text
    assert "drop_nan_labels `true`" in text
    assert "nan_labels_dropped 5" in text
    assert "cross-matrix.json" in text
    assert "/api/tasks/t/task-artifacts/a/download" in text
    assert [table["title"] for table in tables] == [
        "二维 Cross Matrix 全量单元格（保持 X/Y 分箱顺序）",
        "二维 Cross Matrix 金额观测",
    ]
    assert [row[0] for row in tables[0]["rows"]] == [
        "cross-cell-1",
        "cross-cell-2",
        "cross-cell-3",
        "cross-cell-4",
    ]
    assert tables[0]["rows"][1][7:] == ["n/a", "n/a", "0.1911", "0.0017"]


def test_cross_matrix_renderer_preserves_amount_statuses_without_deriving_values() -> None:
    text, tables = render_tool_output("build_cross_matrix_candidate", _output())

    amount = next(
        table for table in tables if table["title"] == "二维 Cross Matrix 金额观测"
    )
    assert amount["columns"] == [
        "Cell ID",
        "X source bin",
        "Y source bin",
        "维度",
        "状态",
        "覆盖样本",
        "覆盖率",
        "观测值",
        "原因",
    ]
    assert [row[3:8] for row in amount["rows"][:3]] == [
        ["放款金额", "available", "29", "96.7%", "120000.0000"],
        ["逾期金额", "available", "28", "93.3%", "6000.0000"],
        ["配对逾期率", "available", "27", "90.0%", "5.0%"],
    ]
    assert amount["rows"][2][-1] == ""
    unavailable = [row for row in amount["rows"] if row[4] == "unavailable"]
    assert unavailable
    assert all(row[7] == "n/a" for row in unavailable)
    assert unavailable[0][-1] == "column_not_configured"
    assert "推荐" not in text
    assert "最好" not in text
