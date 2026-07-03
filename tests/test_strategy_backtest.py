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
    # swap-out set is empty (baseline never approves anyone the new strategy rejects
    # here) -- an empty set has no defined bad rate, so it is None, not 0.0 (DOM-11).
    assert result.swap_out_bad_rate is None
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
    # No baseline supplied -> zero-swap path; both swap sets are empty, so their
    # bad rates are undefined (None), not the misleading 0.0 (DOM-11).
    assert result.swap_in_bad_rate is None
    assert result.swap_out_bad_rate is None


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


# -- FIN-3 #4: EL chain degrades gracefully when pd_col is missing ---------------
def test_backtest_strategy_degrades_expected_profit_when_pd_col_missing():
    frame = pd.DataFrame({
        "score": [580, 620, 730, 760, 590],
        "bad": [1, 0, 0, 1, 0],
        "ead": [1000.0, 2000.0, 1000.0, 500.0, 1000.0],
    })
    strategy = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )

    # A profit backtest is requested (profit_params given) but pd_col is absent, so the
    # expected-loss chain cannot run. It must degrade gracefully: no raise, no fake 0.0.
    result = backtest_strategy(
        frame, strategy, target_col="bad",
        profit_params=_profit_params(), ead_col="ead", pd_col=None,
    )

    assert result.expected_profit is None
    assert result.profit_note is not None
    assert "pd_col" in result.profit_note
    # The rest of the backtest still computes normally (score<600 rejects 580 & 590).
    assert result.approval_rate == pytest.approx(3 / 5)


def test_backtest_strategy_computes_expected_profit_when_pd_col_present():
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

    result = backtest_strategy(
        frame, strategy, target_col="bad",
        profit_params=_profit_params(), ead_col="ead", pd_col="pd",
    )

    # Real profit (float) and no degradation note -- unchanged behavior.
    assert isinstance(result.expected_profit, float)
    assert result.profit_note is None


def test_backtest_strategy_expected_profit_is_zero_without_profit_params():
    # Regression guard: 0.0 still means "no profit backtest requested", distinct from
    # the None graceful-degradation signal above.
    frame = pd.DataFrame({"score": [700, 720], "bad": [0, 1]})
    strategy = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )

    result = backtest_strategy(frame, strategy, target_col="bad")

    assert result.expected_profit == 0.0
    assert result.profit_note is None


def test_compare_strategies_profit_delta_none_when_pd_col_missing():
    from marvis.packs.strategy.compare import compare_strategies

    frame = pd.DataFrame({
        "score": [580, 620, 730, 760, 590],
        "bad": [1, 0, 0, 1, 0],
        "ead": [1000.0, 2000.0, 1000.0, 500.0, 1000.0],
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

    # Missing pd_col -> profit delta is undefined (None), summary says 不可用, no crash.
    result = compare_strategies(
        frame, strategy, baseline, target_col="bad",
        profit_params=_profit_params(), ead_col="ead", pd_col=None,
    )

    assert result.deltas["expected_profit"] is None
    assert "不可用" in result.summary_text
