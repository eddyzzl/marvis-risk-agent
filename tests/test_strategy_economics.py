import json

import pandas as pd
import pytest

from marvis.packs.strategy.economics import limit_metrics, pricing_metrics
from marvis.packs.strategy.errors import StrategyError


def test_limit_metrics_sorts_tiers_and_uses_only_labeled_bad_rate_denominator():
    result = limit_metrics(
        pd.Series([2000, 1000, 2000, 0]),
        pd.Series([1, 1, None, 0]),
        baseline=pd.Series([1500, 1000, 2500, 0]),
    )

    assert result == {
        "count": 4,
        "total_limit": 5000.0,
        "mean_limit": 1250.0,
        "min_limit": 0.0,
        "max_limit": 2000.0,
        "by_limit": [
            {
                "assigned_limit": 0.0,
                "count": 1,
                "share": 0.25,
                "labeled_count": 1,
                "bad_count": 0,
                "bad_rate": 0.0,
            },
            {
                "assigned_limit": 1000.0,
                "count": 1,
                "share": 0.25,
                "labeled_count": 1,
                "bad_count": 1,
                "bad_rate": 1.0,
            },
            {
                "assigned_limit": 2000.0,
                "count": 2,
                "share": 0.5,
                "labeled_count": 1,
                "bad_count": 1,
                "bad_rate": 1.0,
            },
        ],
        "baseline": {
            "up_count": 1,
            "down_count": 1,
            "unchanged_count": 2,
            "total_limit_delta": 0.0,
        },
        "economics": None,
    }


def test_limit_metrics_returns_none_for_a_tier_without_target_labels():
    result = limit_metrics(pd.Series([1000, 2000]), pd.Series([None, 1]))

    assert result["by_limit"][0]["labeled_count"] == 0
    assert result["by_limit"][0]["bad_count"] == 0
    assert result["by_limit"][0]["bad_rate"] is None


def test_limit_metrics_calculates_expected_ead_and_expected_loss_rowwise():
    result = limit_metrics(
        pd.Series([1000, 2000]),
        pd.Series([0, 1]),
        pd=pd.Series([0.1, 0.2]),
        lgd=0.5,
        utilization=pd.Series([0.5, 0.25]),
    )

    assert result["economics"] == {
        "expected_ead": 1000.0,
        "expected_loss": 75.0,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pd": pd.Series([0.1])},
        {"lgd": pd.Series([0.5]), "utilization": pd.Series([0.5])},
    ],
)
def test_limit_metrics_fails_closed_for_partial_economics(kwargs):
    with pytest.raises(StrategyError, match="requires all inputs"):
        limit_metrics(pd.Series([1000]), pd.Series([0]), **kwargs)


@pytest.mark.parametrize(
    "limits",
    [pd.Series([-1]), pd.Series([float("inf")]), pd.Series([None])],
)
def test_limit_metrics_rejects_invalid_assigned_limits(limits):
    with pytest.raises(StrategyError):
        limit_metrics(limits, pd.Series([0]))


def test_limit_metrics_empty_population_has_no_fabricated_summary_statistics():
    result = limit_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float))

    assert result["count"] == 0
    assert result["total_limit"] == 0.0
    assert result["mean_limit"] is None
    assert result["min_limit"] is None
    assert result["max_limit"] is None
    assert result["by_limit"] == []


def test_pricing_metrics_sorts_risk_tiers_and_compares_baseline():
    result = pricing_metrics(
        pd.Series([0.2, 0.1, 0.2]),
        pd.Series([1, None, 0]),
        baseline=pd.Series([0.15, 0.1, 0.25]),
    )

    assert result["mean_rate"] == pytest.approx(1 / 6)
    assert result["risk_tiers"] == [
        {
            "assigned_rate": 0.1,
            "count": 1,
            "share": pytest.approx(1 / 3),
            "labeled_count": 0,
            "bad_count": 0,
            "bad_rate": None,
        },
        {
            "assigned_rate": 0.2,
            "count": 2,
            "share": pytest.approx(2 / 3),
            "labeled_count": 2,
            "bad_count": 1,
            "bad_rate": 0.5,
        },
    ]
    assert result["baseline"] == {
        "repriced_up_count": 1,
        "repriced_down_count": 1,
        "unchanged_count": 1,
    }
    assert result["economics"] is None


def test_pricing_metrics_calculates_rowwise_economics_and_baseline_profit_delta():
    result = pricing_metrics(
        pd.Series([0.12, 0.18]),
        pd.Series([0, 1]),
        baseline=pd.Series([0.10, 0.16]),
        ead=pd.Series([1000, 2000]),
        pd=pd.Series([0.10, 0.05]),
        lgd=pd.Series([0.5, 0.4]),
        funding_rate=pd.Series([0.03, 0.04]),
        term_months=pd.Series([12, 6]),
        operating_cost_per_loan=pd.Series([10, 20]),
    )

    economics = result["economics"]
    assert economics == {
        "total_ead": 3000.0,
        "ead_weighted_rate": pytest.approx(0.16),
        "revenue": 300.0,
        "expected_loss": 90.0,
        "funding_cost": 70.0,
        "operating_cost": 30.0,
        "profit": 110.0,
        "roa": pytest.approx(110 / 3000),
        "baseline_profit": 70.0,
        "profit_delta_vs_baseline": 40.0,
        "by_row": [
            {
                "position": 0,
                "revenue": 120.0,
                "expected_loss": 50.0,
                "funding_cost": 30.0,
                "operating_cost": 10.0,
                "profit": 30.0,
                "roa": 0.03,
                "profit_delta_vs_baseline": 20.0,
            },
            {
                "position": 1,
                "revenue": 180.0,
                "expected_loss": 40.0,
                "funding_cost": 40.0,
                "operating_cost": 20.0,
                "profit": 80.0,
                "roa": 0.04,
                "profit_delta_vs_baseline": 20.0,
            },
        ],
    }


def test_pricing_economics_without_baseline_does_not_invent_a_profit_delta():
    result = pricing_metrics(
        pd.Series([0.1]),
        pd.Series([0]),
        ead=pd.Series([0]),
        pd=pd.Series([0.1]),
        lgd=0.5,
        funding_rate=0.03,
        term_months=12,
        operating_cost_per_loan=0,
    )

    economics = result["economics"]
    assert economics["roa"] is None
    assert economics["ead_weighted_rate"] is None
    assert economics["baseline_profit"] is None
    assert economics["profit_delta_vs_baseline"] is None
    assert economics["by_row"][0]["roa"] is None
    assert economics["by_row"][0]["profit_delta_vs_baseline"] is None


def test_pricing_metrics_fails_closed_for_partial_economics():
    with pytest.raises(StrategyError, match="requires all inputs"):
        pricing_metrics(
            pd.Series([0.1]),
            pd.Series([0]),
            ead=pd.Series([1000]),
            pd=pd.Series([0.1]),
        )


@pytest.mark.parametrize(
    "rates",
    [pd.Series([-0.01]), pd.Series([1.01]), pd.Series([float("nan")])],
)
def test_pricing_metrics_rejects_invalid_annual_decimal_rates(rates):
    with pytest.raises(StrategyError):
        pricing_metrics(rates, pd.Series([0]))


def test_metrics_reject_misaligned_indexes_instead_of_silently_realigning():
    assigned = pd.Series([1000], index=["loan-a"])
    target = pd.Series([0], index=["loan-b"])

    with pytest.raises(StrategyError, match="index must exactly match"):
        limit_metrics(assigned, target)


def test_metrics_reject_nonbinary_target_but_allow_missing_target():
    with pytest.raises(StrategyError, match="only 0, 1"):
        pricing_metrics(pd.Series([0.1, 0.2]), pd.Series([0, 2]))

    result = pricing_metrics(pd.Series([0.1]), pd.Series([None]))
    assert result["risk_tiers"][0]["bad_rate"] is None


def test_metric_results_are_strict_json_safe():
    results = [
        limit_metrics(pd.Series([1000]), pd.Series([None])),
        pricing_metrics(pd.Series([0.1]), pd.Series([None])),
    ]

    assert json.loads(json.dumps(results, allow_nan=False)) == results
