import pandas as pd
import pytest

import marvis.packs.strategy as strategy_pack
from marvis.packs.strategy import (
    ProfitParams,
    TradeoffPoint,
    recommend_operating_point,
    tradeoff_view,
)


def _profit_params() -> ProfitParams:
    return ProfitParams(
        annual_rate=0.12,
        funding_rate=0.03,
        lgd=0.5,
        operating_cost_per_loan=10.0,
        term_months=6,
    )


def test_tradeoff_view_scans_cutoffs_and_calculates_platform_metrics():
    frame = pd.DataFrame({
        "score": [500, 620, 730, 760],
        "bad": [1, 0, 0, 1],
        "ead": [1000.0, 2000.0, 1000.0, 500.0],
        "pd": [0.20, 0.05, 0.02, 0.10],
    })

    points = tradeoff_view(
        frame,
        score_col="score",
        target_col="bad",
        cutoffs=[550, 650, 750],
        profit_params=_profit_params(),
        ead_col="ead",
        pd_col="pd",
    )

    assert points == [
        TradeoffPoint(cutoff=550.0, approval_rate=0.75, bad_rate=pytest.approx(1 / 3), expected_profit=42.5),
        TradeoffPoint(cutoff=650.0, approval_rate=0.5, bad_rate=0.5, expected_profit=12.5),
        TradeoffPoint(cutoff=750.0, approval_rate=0.25, bad_rate=1.0, expected_profit=-12.5),
    ]


def test_tradeoff_view_defaults_profit_to_zero_and_sorts_cutoffs():
    frame = pd.DataFrame({"score": [500, 620, 730, 760], "bad": [1, 0, 0, 1]})

    points = tradeoff_view(frame, score_col="score", target_col="bad", cutoffs=[750, 550])

    assert [point.cutoff for point in points] == [550.0, 750.0]
    assert [point.expected_profit for point in points] == [0.0, 0.0]


def test_recommend_operating_point_respects_objective_and_bad_rate_constraint():
    points = [
        TradeoffPoint(cutoff=500.0, approval_rate=0.9, bad_rate=0.08, expected_profit=100.0),
        TradeoffPoint(cutoff=600.0, approval_rate=0.7, bad_rate=0.04, expected_profit=120.0),
        TradeoffPoint(cutoff=700.0, approval_rate=0.5, bad_rate=0.02, expected_profit=90.0),
    ]

    assert recommend_operating_point(points, objective="max_profit", max_bad_rate=0.05).cutoff == 600.0
    assert recommend_operating_point(points, objective="max_approval", max_bad_rate=0.05).cutoff == 600.0
    assert recommend_operating_point(points, objective="max_profit", max_bad_rate=0.01).cutoff == 700.0
    with pytest.raises(ValueError, match="objective"):
        recommend_operating_point(points, objective="unknown")


def test_strategy_package_exports_tradeoff_surface():
    assert strategy_pack.tradeoff_view is tradeoff_view
    assert strategy_pack.recommend_operating_point is recommend_operating_point
