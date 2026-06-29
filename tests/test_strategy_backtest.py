import pandas as pd
import pytest

import marvis.packs.strategy as strategy_pack
from marvis.packs.strategy import ProfitParams, apply_strategy, backtest_strategy, build_strategy
from marvis.packs.strategy.errors import StrategyError


def _profit_params() -> ProfitParams:
    return ProfitParams(
        annual_rate=0.12,
        funding_rate=0.03,
        lgd=0.5,
        operating_cost_per_loan=10.0,
        term_months=6,
    )


def test_backtest_strategy_calculates_rates_swap_and_profit():
    frame = pd.DataFrame({
        "score": [580, 620, 730, 760, 590],
        "bad": [1, 0, 0, 1, 0],
        "ead": [1000.0, 2000.0, 1000.0, 500.0, 1000.0],
        "pd": [0.20, 0.05, 0.02, 0.10, 0.15],
    })
    strategy = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )
    baseline = build_strategy(
        "approval",
        [{"condition": "score < 650", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )

    result = backtest_strategy(
        frame,
        strategy,
        target_col="bad",
        baseline=baseline,
        profit_params=_profit_params(),
        ead_col="ead",
        pd_col="pd",
    )

    assert result.strategy_id == strategy.id
    assert result.approval_rate == pytest.approx(3 / 5)
    assert result.approved_count == 3
    assert result.approved_bad_rate == pytest.approx(1 / 3)
    assert result.rejected_bad_rate == pytest.approx(1 / 2)
    assert result.expected_profit == pytest.approx(42.5)
    assert result.swap_in_count == 1
    assert result.swap_out_count == 0
    assert result.swap_in_bad_rate == 0.0
    assert result.swap_out_bad_rate == 0.0
    assert result.by_segment == (
        {"decision": "approve", "count": 3, "bad_count": 1, "bad_rate": pytest.approx(1 / 3)},
        {"decision": "reject", "count": 2, "bad_count": 1, "bad_rate": pytest.approx(1 / 2)},
    )


def test_backtest_strategy_defaults_swap_and_profit_when_optional_inputs_missing():
    frame = pd.DataFrame({"score": [700, 720], "bad": [0, 1]})
    strategy = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )

    result = backtest_strategy(frame, strategy, target_col="bad")

    assert result.approval_rate == 1.0
    assert result.approved_count == 2
    assert result.approved_bad_rate == 0.5
    assert result.rejected_bad_rate == 0.0
    assert result.expected_profit == 0.0
    assert result.swap_in_count == 0
    assert result.swap_out_count == 0
    assert result.swap_in_bad_rate == 0.0
    assert result.swap_out_bad_rate == 0.0


def test_strategy_conditions_coerce_numeric_literals_for_string_columns():
    frame = pd.DataFrame({"score": ["580", "620", "730"], "segment": ["1", "2", "3"]})
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score < 600", "decision": "reject"},
            {"condition": "segment in [2, 3]", "decision": "reject"},
        ],
        score_col="score",
        default_decision="approve",
    )

    decisions = apply_strategy(frame, strategy)

    assert decisions.tolist() == ["reject", "reject", "reject"]


def test_strategy_conditions_fail_loud_for_numeric_literal_on_non_numeric_column():
    frame = pd.DataFrame({"score": ["low", "high"]})
    strategy = build_strategy(
        "approval",
        [{"condition": "score == 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )

    with pytest.raises(StrategyError, match="non-numeric"):
        apply_strategy(frame, strategy)


def test_strategy_package_exports_backtest_strategy():
    assert strategy_pack.backtest_strategy is backtest_strategy
