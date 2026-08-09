from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from marvis.agent.portfolio_setup import (
    PortfolioSetupError,
    build_portfolio_proposal,
)


@dataclass(frozen=True)
class _Dataset:
    id: str = "performance-1"
    task_id: str = "task-1"
    role: str = "performance"
    row_count: int = 2
    source_path: str = "performance.parquet"
    content_hash: str = "a" * 64


class _Registry:
    def list_for_task(self, task_id):
        return [_Dataset()]

    def resolve_path(self, dataset_id):
        return "/tmp/performance.parquet"


class _Backend:
    def __init__(self, columns, *, states=None):
        self._columns = list(columns)
        self._states = list(states or ["current", "M1"])

    def column_names(self, path):
        return list(self._columns)

    def read_frame(self, path, *, columns):
        return pd.DataFrame({columns[0]: self._states})


class _AuthenticatedRegistry(_Registry):
    def __init__(self):
        self.snapshot_reads = 0

    def read_authenticated_parquet_snapshot(self, dataset_id, *, columns=None):
        assert dataset_id == "performance-1"
        assert columns is None
        self.snapshot_reads += 1
        return pd.DataFrame(
            {
                "loan_id": ["a", "b"],
                "snapshot_month": ["2025-01", "2025-02"],
                "bucket": ["current", "M1"],
                "balance": [100.0, 80.0],
                "segment": ["A", "B"],
            }
        )


class _RejectSecondaryPathReadBackend:
    def column_names(self, path):
        raise AssertionError("columns must come from the authenticated snapshot")

    def read_frame(self, path, *, columns):
        raise AssertionError("states must come from the authenticated snapshot")


@pytest.mark.parametrize(
    ("columns", "missing_label"),
    [
        (
            ["loan_id", "snapshot_month", "bucket", "segment"],
            "余额/EAD",
        ),
        (
            ["loan_id", "snapshot_month", "bucket", "balance"],
            "业务分群",
        ),
        (
            ["loan_id", "snapshot_month", "bucket"],
            "余额/EAD、业务分群",
        ),
    ],
)
def test_portfolio_setup_rejects_incomplete_full_report_semantics(
    columns,
    missing_label,
):
    with pytest.raises(PortfolioSetupError, match=missing_label):
        build_portfolio_proposal(
            _Registry(),
            _Backend(columns),
            "task-1",
            None,
            loss_state="charged_off",
            lgd=0.45,
            horizon_months=18,
        )


def test_portfolio_setup_returns_complete_no_trend_contract():
    proposal = build_portfolio_proposal(
        _Registry(),
        _Backend(
            [
                "loan_id",
                "snapshot_month",
                "bucket",
                "balance",
                "segment",
            ]
        ),
        "task-1",
        None,
        loss_state="M1",
        lgd=0.45,
        horizon_months=18,
    )

    assert proposal.template_id == "portfolio_analysis_no_trend"
    assert proposal.balance_col == "balance"
    assert proposal.segment_col == "segment"
    assert proposal.template_slots(["current", "M1"]) == {
        "performance_dataset_id": "performance-1",
        "performance_dataset_content_hash": "a" * 64,
        "id_col": "loan_id",
        "snapshot_col": "snapshot_month",
        "bucket_col": "bucket",
        "states": ["current", "M1"],
        "balance_col": "balance",
        "segment_col": "segment",
        "loss_state": "M1",
        "lgd": 0.45,
        "horizon_months": 18,
    }


def test_portfolio_setup_uses_one_authenticated_snapshot_for_columns_and_states():
    registry = _AuthenticatedRegistry()

    proposal = build_portfolio_proposal(
        registry,
        _RejectSecondaryPathReadBackend(),
        "task-1",
        None,
        loss_state="M1",
        lgd=0.45,
        horizon_months=18,
    )

    assert registry.snapshot_reads == 1
    assert proposal.id_col == "loan_id"
    assert proposal.snapshot_col == "snapshot_month"
    assert proposal.bucket_col == "bucket"
    assert proposal.proposed_states == ["current", "M1"]


def test_portfolio_setup_keeps_charged_off_last_and_binds_explicit_el_contract():
    proposal = build_portfolio_proposal(
        _Registry(),
        _Backend(
            [
                "loan_id",
                "snapshot_month",
                "bucket",
                "balance",
                "segment",
            ],
            states=["current", "M1", "M2", "M3+", "charged_off"],
        ),
        "task-1",
        None,
        loss_state="charged_off",
        lgd=0.45,
        horizon_months=18,
    )

    assert proposal.proposed_states == [
        "current",
        "M1",
        "M2",
        "M3+",
        "charged_off",
    ]
    assert proposal.template_slots(proposal.proposed_states) == {
        "performance_dataset_id": "performance-1",
        "performance_dataset_content_hash": "a" * 64,
        "id_col": "loan_id",
        "snapshot_col": "snapshot_month",
        "bucket_col": "bucket",
        "states": ["current", "M1", "M2", "M3+", "charged_off"],
        "balance_col": "balance",
        "segment_col": "segment",
        "loss_state": "charged_off",
        "lgd": 0.45,
        "horizon_months": 18,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loss_state": None, "lgd": 0.45, "horizon_months": 18}, "损失态"),
        ({"loss_state": "M1", "lgd": None, "horizon_months": 18}, "LGD"),
        ({"loss_state": "M1", "lgd": 0.45, "horizon_months": None}, "预测期限"),
        ({"loss_state": "missing", "lgd": 0.45, "horizon_months": 18}, "不在逾期桶"),
        ({"loss_state": "M1", "lgd": 1.2, "horizon_months": 18}, "LGD"),
        ({"loss_state": "M1", "lgd": 0.45, "horizon_months": 0}, "预测期限"),
    ],
)
def test_portfolio_setup_rejects_missing_or_invalid_el_contract(kwargs, message):
    with pytest.raises(PortfolioSetupError, match=message):
        build_portfolio_proposal(
            _Registry(),
            _Backend(
                [
                    "loan_id",
                    "snapshot_month",
                    "bucket",
                    "balance",
                    "segment",
                ]
            ),
            "task-1",
            None,
            **kwargs,
        )
