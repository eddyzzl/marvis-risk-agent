"""Renderer contract for the thin Pool-impact Tool envelope."""

from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _accept_shape_fixture(monkeypatch) -> None:
    """Keep presentation fixtures focused; integration tests use real evidence."""

    monkeypatch.setattr(
        "marvis.packs.strategy.pool_impact_tools."
        "validate_measure_pool_impact_tool_output",
        lambda value: value,
    )


def _assessment() -> dict:
    return {
        "assessment_id": "strategy-impact-assessment-" + "a" * 24,
        "content_hash": "b" * 64,
        "identity": {
            "pool_id": "strategy-pool-1",
            "strategy_type": "approval",
            "revision": 3,
        },
        "population": {
            "population_count": 10,
            "labelled_count": 10,
            "unlabelled_count": 0,
            "label_coverage": 1.0,
        },
        "overall": {
            "effect": {
                "amounts": {
                    "loan_amount": {"status": "unavailable", "sum": None},
                    "overdue_amount": {"status": "available", "sum": 0.0},
                    "paired": {"status": "unavailable", "overdue_rate": None},
                }
            },
            "actions": {
                "metrics": {"overall_bad_rate": 0.2},
                "breakdown": [
                    {
                        "action": "approve",
                        "count": 0,
                        "rate": 0.0,
                        "labelled_count": 0,
                        "bad_count": 0,
                        "bad_rate": None,
                    },
                    {
                        "action": "reject",
                        "count": 10,
                        "rate": 1.0,
                        "labelled_count": 10,
                        "bad_count": 2,
                        "bad_rate": 0.2,
                    },
                ],
            },
        },
        "waterfall": [
            {
                "position": 1,
                "rule_id": "candidate-rule-1",
                "action": {"type": "reject"},
                "standalone": {"population_count": 5},
                "incremental": {
                    "population_count": 0,
                    "population_share": 0.0,
                    "bad_rate": None,
                },
                "shadowed": {"population_count": 5},
                "remaining_after": {"population_count": 5},
            }
        ],
        "default_unmatched": {
            "action": {"type": "approval"},
            "effect": {
                "population_count": 5,
                "population_share": 0.5,
                "bad_rate": 0.2,
            },
        },
        "monthly": {
            "status": "unavailable",
            "reason": "month_column_not_provided",
            "periods": [],
        },
        "baseline": {"status": "not_requested", "binding": None, "overall": None},
        "red_flags": [],
        "lifecycle": {
            "candidate_stage": "development",
            "observation_stage": "backtested",
            "validation_status": "unvalidated",
            "creates_strategy": False,
            "adopted": False,
            "deployed": False,
        },
    }


def _envelope(assessment: dict) -> dict:
    return {
        "schema_version": "strategy.measure-pool-impact-tool.v1",
        "nan_labels_excluded": 0,
        "assessment": assessment,
        "warnings": [],
        "artifacts": [
            {
                "artifact_id": "artifact-impact-1",
                "filename": "pool-impact.json",
                "download_url": "/api/tasks/t/task-artifacts/a/download",
            }
        ],
    }


def _amount_deltas(*, loan_sum: float, paired_overdue_rate: float) -> dict:
    unavailable = {
        "status": "unavailable",
        "coverage_count": None,
        "coverage_rate": None,
        "sum": None,
    }
    unavailable_paired = {
        "status": "unavailable",
        "coverage_count": None,
        "coverage_rate": None,
        "loan_amount_sum": None,
        "overdue_amount_sum": None,
        "overdue_rate": None,
    }
    return {
        "approve": {
            "loan_amount": {
                "status": "available",
                "coverage_count": 1,
                "coverage_rate": 0.1,
                "sum": loan_sum,
            },
            "overdue_amount": dict(unavailable),
            "paired": {
                "status": "available",
                "coverage_count": 0,
                "coverage_rate": 0.0,
                "loan_amount_sum": 10.0,
                "overdue_amount_sum": -5.0,
                "overdue_rate": paired_overdue_rate,
            },
        },
        "reject": {
            "loan_amount": dict(unavailable),
            "overdue_amount": dict(unavailable),
            "paired": dict(unavailable_paired),
        },
        "review": {
            "loan_amount": dict(unavailable),
            "overdue_amount": dict(unavailable),
            "paired": dict(unavailable_paired),
        },
    }


def test_pool_impact_renderer_never_uses_generic_fallback_on_validator_error(
    monkeypatch,
) -> None:
    def _unexpected(_value):
        raise RuntimeError("unexpected validator failure")

    monkeypatch.setattr(
        "marvis.packs.strategy.pool_impact_tools."
        "validate_measure_pool_impact_tool_output",
        _unexpected,
    )

    text, tables = render_tool_output(
        "measure_pool_impact",
        {"population_count": 999999, "labeled_count": 999998},
    )

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert tables == []


def test_pool_impact_renderer_distinguishes_zero_from_unavailable_and_no_deploy(
    monkeypatch,
) -> None:
    _accept_shape_fixture(monkeypatch)
    text, tables = render_tool_output("measure_pool_impact", _envelope(_assessment()))

    assert "放款金额 unavailable" in text
    assert "逐月结果 unavailable（month_column_not_provided）" in text
    assert "未创建或修改策略、未采纳、未部署" in text
    assert "/api/tasks/t/task-artifacts/a/download" in text
    waterfall = next(
        table for table in tables if table["title"] == "Strategy Pool 级联 Waterfall"
    )
    assert waterfall["rows"][0][4:7] == ["0", "0.0%", "n/a"]
    assert waterfall["rows"][-1] == [
        "default",
        "default_unmatched",
        "approval",
        "n/a",
        "5",
        "50.0%",
        "20.0%",
        "n/a",
        "n/a",
    ]
    actions = next(table for table in tables if table["title"] == "总体动作与风险影响")
    assert actions["rows"][0][1:3] == ["0", "0.0%"]
    amounts = next(table for table in tables if table["title"] == "整体金额影响")
    assert amounts["rows"] == [["逾期金额", "available", "n/a", "n/a", "0.0000"]]
    assert not any(table["title"] == "Pool Impact 红旗" for table in tables)
    assert not any(table["title"] == "Pool Impact Tool 警告" for table in tables)
    assert not any(
        table["title"] == "逐月相对基线指标变化（当前 - 基线）" for table in tables
    )
    assert not any(
        table["title"] == "相对基线金额变化（当前 - 基线）" for table in tables
    )
    assert not any(
        table["title"] == "逐月相对基线金额变化（当前 - 基线）" for table in tables
    )


def test_pool_impact_renderer_surfaces_label_exclusions_flags_and_warnings(
    monkeypatch,
) -> None:
    _accept_shape_fixture(monkeypatch)
    assessment = _assessment()
    assessment["population"] = {
        "population_count": 10,
        "labelled_count": 8,
        "unlabelled_count": 2,
        "label_coverage": 0.8,
    }
    assessment["red_flags"] = [
        {
            "level": "amber",
            "code": "incomplete_label_coverage",
            "message": "2 population rows are excluded from risk denominators",
        },
        {
            "level": "red",
            "code": "rule_fully_shadowed",
            "message": "candidate-rule-1 is fully shadowed",
        },
    ]
    envelope = _envelope(assessment)
    envelope["nan_labels_excluded"] = 3
    envelope["warnings"] = ["Tool envelope warning"]

    text, tables = render_tool_output("measure_pool_impact", envelope)

    assert "`unlabeled_rows` **2**" in text
    assert "`nan_labels_excluded` **3**" in text
    assert "Tool envelope warning" in text
    flags = next(table for table in tables if table["title"] == "Pool Impact 红旗")
    assert flags["rows"] == [
        [
            "amber",
            "incomplete_label_coverage",
            "2 population rows are excluded from risk denominators",
        ],
        ["red", "rule_fully_shadowed", "candidate-rule-1 is fully shadowed"],
    ]
    warnings = next(
        table for table in tables if table["title"] == "Pool Impact Tool 警告"
    )
    assert warnings["rows"] == [["Tool envelope warning"]]


def test_pool_impact_renderer_shows_tool_owned_monthly_and_baseline_deltas(
    monkeypatch,
) -> None:
    _accept_shape_fixture(monkeypatch)
    assessment = _assessment()
    assessment["monthly"] = {
        "status": "available",
        "reason": None,
        "periods": [
            {
                "period": "2026-01",
                "effect": {"population_count": 10, "label_coverage": 1.0},
                "actions": {
                    "metrics": {
                        "overall_bad_rate": 0.2,
                        "approve_rate": 0.4,
                        "reject_rate": 0.6,
                        "review_rate": 0.0,
                    }
                },
            }
        ],
    }
    assessment["baseline"] = {
        "status": "available",
        "binding": {"strategy_id": "strategy-baseline-1"},
        "overall": {
            "metric_deltas": {"approve_rate": 0.1, "reject_count": -2},
            "amount_deltas": _amount_deltas(
                loan_sum=125.5,
                paired_overdue_rate=-0.01,
            ),
        },
        "monthly": {
            "status": "available",
            "periods": [
                {
                    "period": "2026-01",
                    "metric_deltas": {
                        "approve_rate": 0.05,
                        "reject_count": -1,
                    },
                    "amount_deltas": _amount_deltas(
                        loan_sum=-25.0,
                        paired_overdue_rate=0.02,
                    ),
                }
            ],
        },
    }

    text, tables = render_tool_output("measure_pool_impact", _envelope(assessment))

    assert "基线对比 available" in text
    assert "strategy-baseline-1" in text
    monthly = next(table for table in tables if table["title"] == "逐月影响")
    assert monthly["rows"] == [
        ["2026-01", "10", "100.0%", "20.0%", "40.0%", "60.0%", "0.0%"]
    ]
    deltas = next(
        table for table in tables if table["title"] == "相对基线指标变化（当前 - 基线）"
    )
    assert deltas["rows"] == [["approve_rate", "10.0%"], ["reject_count", "-2"]]
    monthly_deltas = next(
        table
        for table in tables
        if table["title"] == "逐月相对基线指标变化（当前 - 基线）"
    )
    assert monthly_deltas["rows"] == [
        ["2026-01", "approve_rate", "5.0%"],
        ["2026-01", "reject_count", "-1"],
    ]
    overall_amount_deltas = next(
        table
        for table in tables
        if table["title"] == "相对基线金额变化（当前 - 基线）"
    )
    assert overall_amount_deltas["rows"] == [
        [
            "approve",
            "loan_amount",
            "available",
            "1",
            "10.0%",
            "125.5000",
            "n/a",
            "n/a",
            "n/a",
        ],
        [
            "approve",
            "paired",
            "available",
            "0",
            "0.0%",
            "n/a",
            "10.0000",
            "-5.0000",
            "-1.0%",
        ],
    ]
    monthly_amount_deltas = next(
        table
        for table in tables
        if table["title"] == "逐月相对基线金额变化（当前 - 基线）"
    )
    assert monthly_amount_deltas["rows"] == [
        [
            "2026-01",
            "approve",
            "loan_amount",
            "available",
            "1",
            "10.0%",
            "-25.0000",
            "n/a",
            "n/a",
            "n/a",
        ],
        [
            "2026-01",
            "approve",
            "paired",
            "available",
            "0",
            "0.0%",
            "n/a",
            "10.0000",
            "-5.0000",
            "2.0%",
        ],
    ]
