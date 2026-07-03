from dataclasses import asdict

import marvis.packs.strategy as strategy_contracts
from marvis.packs.strategy import (
    BacktestResult,
    ProfitResult,
    RollRateMatrix,
    Strategy,
    StrategyRule,
    TradeoffPoint,
    VintageCurve,
)


def test_vintage_curve_contract_round_trips():
    curve = VintageCurve(
        cohort_col="loan_month",
        mob_max=3,
        cohorts=("2026-01", "2026-02"),
        curves={"2026-01": [0.01, 0.02, 0.03], "2026-02": [0.02, 0.04, None]},
        counts={"2026-01": 100, "2026-02": 80},
    )

    payload = asdict(curve)

    assert VintageCurve(**payload) == curve
    assert payload["curves"]["2026-02"][2] is None
    assert payload["counts"]["2026-01"] == 100


def test_roll_rate_matrix_contract_keeps_state_order_and_counts():
    matrix = RollRateMatrix(
        states=("C", "M1", "M2"),
        matrix=((0.9, 0.1, 0.0), (0.2, 0.6, 0.2), (0.0, 0.0, 1.0)),
        period="month",
        base_counts={"C": 100, "M1": 20, "M2": 5},
    )

    payload = asdict(matrix)

    assert RollRateMatrix(**payload) == matrix
    assert payload["states"] == ("C", "M1", "M2")
    assert payload["matrix"][2] == (0.0, 0.0, 1.0)


def test_profit_result_contract_round_trips_business_numbers():
    result = ProfitResult(
        segment="low-risk",
        count=10,
        revenue=1200.0,
        expected_loss=180.0,
        funding_cost=300.0,
        operating_cost=50.0,
        net_profit=670.0,
        roa=0.067,
    )

    payload = asdict(result)

    assert ProfitResult(**payload) == result
    assert payload["net_profit"] == 670.0
    assert payload["roa"] == 0.067


def test_strategy_contract_round_trips_nested_rules():
    strategy = Strategy(
        id="strategy-1",
        strategy_type="approval",
        rules=(
            StrategyRule(condition="score < 600", decision="reject", value=None),
            StrategyRule(condition="score >= 720", decision="approve", value=None),
        ),
        score_col="score",
        default_decision="approve",
        description="baseline cutoff strategy",
    )

    payload = asdict(strategy)
    rebuilt = Strategy(
        id=payload["id"],
        strategy_type=payload["strategy_type"],
        rules=tuple(StrategyRule(**rule) for rule in payload["rules"]),
        score_col=payload["score_col"],
        default_decision=payload["default_decision"],
        description=payload["description"],
    )

    assert rebuilt == strategy
    assert payload["rules"][0]["condition"] == "score < 600"
    assert payload["rules"][1]["decision"] == "approve"


def test_backtest_result_contract_round_trips_swap_and_segments():
    result = BacktestResult(
        strategy_id="strategy-1",
        approval_rate=0.7,
        approved_count=70,
        approved_bad_rate=0.04,
        rejected_bad_rate=0.22,
        expected_profit=2300.0,
        swap_in_count=5,
        swap_out_count=8,
        swap_in_bad_rate=0.12,
        swap_out_bad_rate=0.01,
        by_segment=(
            {"segment": "approved", "count": 70, "bad_rate": 0.04},
            {"segment": "rejected", "count": 30, "bad_rate": 0.22},
        ),
    )

    payload = asdict(result)

    assert BacktestResult(**payload) == result
    assert payload["swap_in_count"] == 5
    assert payload["by_segment"][0]["segment"] == "approved"


def test_tradeoff_point_contract_round_trips():
    point = TradeoffPoint(
        cutoff=650.0,
        approval_rate=0.68,
        bad_rate=0.05,
        expected_profit=1800.0,
    )

    assert TradeoffPoint(**asdict(point)) == point


def test_strategy_package_exports_contract_surface():
    assert strategy_contracts.VintageCurve is VintageCurve
    assert strategy_contracts.RollRateMatrix is RollRateMatrix
    assert strategy_contracts.ProfitResult is ProfitResult
    assert strategy_contracts.StrategyRule is StrategyRule
    assert strategy_contracts.Strategy is Strategy
    assert strategy_contracts.BacktestResult is BacktestResult
    assert strategy_contracts.TradeoffPoint is TradeoffPoint
    assert "Strategy" in strategy_contracts.__all__
