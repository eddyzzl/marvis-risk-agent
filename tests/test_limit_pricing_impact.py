"""B-10: typed limit/pricing impact measurement.

Every numeric expectation is hand-computed from the documented formulas in
``marvis/packs/strategy/limit_pricing_impact.py`` (never re-derived with the
implementation itself), so a regression in the arithmetic fails these goldens.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.limit_pricing_impact import (
    limit_impact,
    pricing_impact,
)


def test_limit_impact_hand_calculated_exposure_buckets_and_el_delta():
    result = limit_impact(
        assigned_limit=pd.Series([1000, 2000, 3000, 4000]),
        baseline_limit=pd.Series([1000, 1500, 3500, 4000]),
        exposure=pd.Series([500, 1000, 1500, 2000]),
        baseline_exposure=pd.Series([500, 750, 1750, 2000]),
        pd=pd.Series([0.1, 0.2, 0.3, 0.4]),
        lgd=0.5,
        target=pd.Series([0, 1, 0, 1]),
        segment=pd.Series(["A", "A", "B", "B"]),
        month=pd.Series(["202601", "202601", "202602", "202602"]),
    )

    assert result.strategy_type == "limit"
    assert result.count == 4
    assert result.labeled_count == 4
    assert result.total_limit == 10000.0
    assert result.total_exposure_after == 5000.0

    # Exposure before/after: before = 500 + 750 + 1750 + 2000 = 5000, delta 0.
    assert result.exposure["availability"] == "present"
    assert result.exposure["value"] == {
        "after": 5000.0,
        "before": 5000.0,
        "delta": 0.0,
    }

    # Action buckets (direction of exposure change).
    assert result.action_buckets["availability"] == "present"
    assert result.action_buckets["value"]["rows"] == [
        {"direction": "up", "count": 1, "exposure_delta": pytest.approx(250.0)},
        {"direction": "down", "count": 1, "exposure_delta": pytest.approx(-250.0)},
        {"direction": "unchanged", "count": 2, "exposure_delta": 0.0},
    ]

    # EL after = 25 + 100 + 225 + 400 = 750; before = 25 + 75 + 262.5 + 400 = 762.5.
    assert result.expected_loss["availability"] == "present"
    assert result.expected_loss["value"]["after"] == pytest.approx(750.0)
    assert result.expected_loss["value"]["before"] == pytest.approx(762.5)
    assert result.expected_loss["value"]["delta"] == pytest.approx(-12.5)
    assert result.expected_loss["value"]["pd_proxy_used"] is False

    # Segment x month cells, sorted deterministically by (segment, month).
    cells = result.by_segment_month["value"]["rows"]
    assert [(cell["segment"], cell["month"]) for cell in cells] == [
        ("A", "202601"),
        ("B", "202602"),
    ]
    cell_a = cells[0]
    assert cell_a["count"] == 2
    assert cell_a["labeled_count"] == 2
    assert cell_a["bad_count"] == 1
    assert cell_a["bad_rate"] == 0.5
    assert cell_a["exposure_delta"] == pytest.approx(250.0)
    assert cell_a["expected_loss_after"] == pytest.approx(125.0)
    assert cell_a["expected_loss_delta"] == pytest.approx(25.0)
    cell_b = cells[1]
    assert cell_b["exposure_delta"] == pytest.approx(-250.0)
    assert cell_b["expected_loss_after"] == pytest.approx(625.0)
    assert cell_b["expected_loss_delta"] == pytest.approx(-37.5)

    assert result.red_flags == ()


def test_limit_impact_derives_exposure_from_limit_times_utilization():
    result = limit_impact(
        assigned_limit=pd.Series([1000, 2000]),
        baseline_limit=pd.Series([500, 2000]),
        utilization=0.5,
        pd=0.1,
        lgd=0.5,
    )

    assert result.total_exposure_after == pytest.approx(1500.0)
    assert result.exposure["value"] == {
        "after": pytest.approx(1500.0),
        "before": pytest.approx(1250.0),
        "delta": pytest.approx(250.0),
    }
    assert result.expected_loss["value"]["after"] == pytest.approx(75.0)
    assert result.expected_loss["value"]["before"] == pytest.approx(62.5)
    assert result.expected_loss["value"]["delta"] == pytest.approx(12.5)


def test_limit_impact_marks_el_unavailable_when_lgd_missing():
    result = limit_impact(
        assigned_limit=pd.Series([1000, 2000]),
        exposure=pd.Series([500, 1000]),
        pd=pd.Series([0.1, 0.2]),
    )

    assert result.expected_loss == {
        "availability": "unavailable",
        "reason": "missing_economics_inputs:lgd",
        "value": None,
    }
    # Without a baseline the before/after evidence is also unavailable, never 0.
    assert result.exposure["availability"] == "unavailable"
    assert result.exposure["reason"] == "baseline_limit_not_bound"
    assert result.action_buckets["availability"] == "unavailable"
    assert any(flag["code"] == "economics_unavailable" for flag in result.red_flags)


def test_limit_impact_uses_empirical_bad_rate_as_pd_proxy():
    result = limit_impact(
        assigned_limit=pd.Series([1000, 2000, 3000]),
        baseline_limit=pd.Series([1000, 2000, 3000]),
        exposure=pd.Series([1000, 2000, 3000]),
        baseline_exposure=pd.Series([1000, 2000, 3000]),
        lgd=0.5,
        target=pd.Series([0, 1, 1]),
    )

    value = result.expected_loss["value"]
    assert value["pd_proxy_used"] is True
    # Bad-rate proxy = 2/3; EL = (1000+2000+3000) * (2/3) * 0.5 = 2000.
    assert value["after"] == pytest.approx(2000.0)
    assert value["before"] == pytest.approx(2000.0)
    assert value["delta"] == pytest.approx(0.0)
    assert any(flag["code"] == "pd_proxy_used" for flag in result.red_flags)


def test_limit_impact_without_baseline_marks_comparison_unavailable():
    result = limit_impact(
        assigned_limit=pd.Series([1000, 2000]),
        exposure=pd.Series([500, 1000]),
        pd=0.1,
        lgd=0.5,
    )

    assert result.exposure["availability"] == "unavailable"
    assert result.action_buckets["availability"] == "unavailable"
    # EL "after" is still computable; its delta is absent without a baseline.
    assert result.expected_loss["availability"] == "present"
    assert result.expected_loss["value"]["after"] == pytest.approx(75.0)
    assert result.expected_loss["value"]["before"] is None
    assert result.expected_loss["value"]["delta"] is None
    assert any(flag["code"] == "baseline_not_bound" for flag in result.red_flags)


def test_pricing_impact_hand_calculated_rate_delta_revenue_vs_bad_debt():
    result = pricing_impact(
        exposure=pd.Series([1000, 2000]),
        rate_delta=pd.Series([0.02, 0.01]),
        term_months=12,
        pd=pd.Series([0.1, 0.05]),
        lgd=0.5,
        target=pd.Series([0, 1]),
        segment=pd.Series(["A", "B"]),
        month=pd.Series(["202601", "202601"]),
    )

    assert result.strategy_type == "pricing"
    assert result.count == 2
    assert result.labeled_count == 2
    assert result.total_exposure == 3000.0
    assert result.revenue_source == "rate_delta"
    # Incremental revenue = 1000*0.02*1 + 2000*0.01*1 = 40.
    assert result.incremental_revenue == pytest.approx(40.0)
    # Expected bad-debt cost = 1000*0.1*0.5 + 2000*0.05*0.5 = 100.
    assert result.expected_loss["availability"] == "present"
    assert result.expected_loss["value"]["value"] == pytest.approx(100.0)
    assert result.expected_loss["value"]["pd_proxy_used"] is False
    assert result.net_impact["availability"] == "present"
    assert result.net_impact["value"] == {
        "incremental_revenue": pytest.approx(40.0),
        "expected_loss": pytest.approx(100.0),
        "net": pytest.approx(-60.0),
    }

    cells = result.by_segment_month["value"]["rows"]
    assert [cell["segment"] for cell in cells] == ["A", "B"]
    assert cells[0]["incremental_revenue"] == pytest.approx(20.0)
    assert cells[0]["expected_loss"] == pytest.approx(50.0)
    assert cells[0]["net_impact"] == pytest.approx(-30.0)
    assert cells[0]["bad_rate"] == 0.0
    assert cells[1]["bad_rate"] == 1.0


def test_pricing_impact_fee_delta_does_not_require_term_months():
    result = pricing_impact(
        exposure=pd.Series([1000, 2000]),
        fee_delta=pd.Series([10, 20]),
        pd=pd.Series([0.1, 0.05]),
        lgd=0.5,
    )

    assert result.revenue_source == "fee_delta"
    assert result.incremental_revenue == pytest.approx(30.0)
    assert result.net_impact["value"]["net"] == pytest.approx(-70.0)


def test_pricing_impact_marks_bad_debt_and_net_unavailable_when_lgd_missing():
    result = pricing_impact(
        exposure=pd.Series([1000, 2000]),
        rate_delta=pd.Series([0.02, 0.01]),
        term_months=12,
        pd=pd.Series([0.1, 0.05]),
    )

    assert result.incremental_revenue == pytest.approx(40.0)
    assert result.expected_loss == {
        "availability": "unavailable",
        "reason": "missing_economics_inputs:lgd",
        "value": None,
    }
    assert result.net_impact["availability"] == "unavailable"
    assert result.net_impact["reason"] == "expected_loss_unavailable"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"assigned_limit": pd.Series([-1])}, "assigned_limit"),
        ({"assigned_limit": pd.Series([1000])}, "requires exposure or utilization"),
        (
            {
                "assigned_limit": pd.Series([1000], index=["a"]),
                "exposure": pd.Series([500], index=["b"]),
            },
            "index must exactly match",
        ),
        (
            {"assigned_limit": pd.Series([1000]), "exposure": pd.Series([500]), "pd": 1.5},
            "pd must be <= 1",
        ),
    ],
)
def test_limit_impact_fails_closed_for_invalid_inputs(kwargs, match):
    with pytest.raises(StrategyError, match=match):
        limit_impact(**kwargs)


def test_limit_impact_rejects_nonbinary_target():
    with pytest.raises(StrategyError, match="only 0, 1"):
        limit_impact(
            assigned_limit=pd.Series([1000]),
            exposure=pd.Series([500]),
            target=pd.Series([2]),
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "requires rate_delta or fee_delta"),
        (
            {"rate_delta": 0.02, "fee_delta": 10.0},
            "exactly one of rate_delta or fee_delta",
        ),
        ({"rate_delta": 0.02}, "requires term_months"),
        (
            {
                "rate_delta": pd.Series([0.02], index=["a"]),
                "term_months": 12,
            },
            "index must exactly match",
        ),
        ({"fee_delta": 10.0, "pd": 1.5}, "pd must be <= 1"),
        ({"fee_delta": 10.0, "target": pd.Series([2])}, "only 0, 1"),
    ],
)
def test_pricing_impact_fails_closed_for_invalid_inputs(kwargs, match):
    with pytest.raises(StrategyError, match=match):
        pricing_impact(exposure=pd.Series([1000.0]), **kwargs)


def test_results_are_json_safe_and_declare_development_only_lifecycle():
    limit = limit_impact(
        assigned_limit=pd.Series([1000]),
        baseline_limit=pd.Series([500]),
        exposure=pd.Series([1000]),
        baseline_exposure=pd.Series([500]),
        pd=0.1,
        lgd=0.5,
    ).to_dict()
    pricing = pricing_impact(
        exposure=pd.Series([1000.0]),
        fee_delta=10.0,
        pd=0.1,
        lgd=0.5,
    ).to_dict()

    for payload in (limit, pricing):
        assert payload["effect_stage"] == "backtested"
        assert payload["validation_status"] == "unvalidated"
        assert payload["lifecycle"] == {
            "mutates_pool": False,
            "creates_strategy": False,
            "adopts_strategy": False,
            "promotes_strategy": False,
            "deploys_strategy": False,
        }
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload
