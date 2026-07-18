from __future__ import annotations

import json
import math

import pytest

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    build_monitoring_plan,
    canonical_economics_bindings_hash,
    canonical_monitoring_plan_hash,
    load_monitoring_plan,
    save_monitoring_plan,
)


def test_legacy_v1_monitoring_plan_still_loads_with_revision_defaults(tmp_path):
    path = tmp_path / "legacy-plan.json"
    path.write_text(
        json.dumps(
            {
                "plan_version": 1,
                "strategy_id": "strategy-1",
                "version": 3,
                "cadence_days": 30,
                "last_run_at": "2026-07-01T00:00:00+00:00",
                "thresholds": {"approval_rate": {"metric": "approval_rate"}},
                "expectation_baseline": {"approval_rate": 0.7},
            }
        ),
        encoding="utf-8",
    )

    plan = load_monitoring_plan(path)

    assert plan.plan_version == 1
    assert plan.monitoring_plan_id is None
    assert plan.revision == 1
    assert plan.supersedes_plan_id is None
    assert plan.economics_bindings == {}
    assert plan.last_run_at == "2026-07-01T00:00:00+00:00"


def test_v2_monitoring_plan_round_trips_revision_and_safe_economics_bindings(tmp_path):
    payload = build_monitoring_plan(
        strategy_id="strategy-1",
        version=3,
        approved_bad_rate=0.04,
        approval_rate=0.72,
        monitoring_plan_id="plan-2",
        revision=2,
        supersedes_plan_id="plan-1",
        economics_bindings={
            "lgd": {"kind": "scalar", "value": 0.45},
            "pd": {"kind": "column", "column": "predicted_pd"},
        },
    )
    path = save_monitoring_plan(tmp_path / "plan.json", payload)

    plan = load_monitoring_plan(path)

    assert plan.plan_version == 2
    assert plan.monitoring_plan_id == "plan-2"
    assert plan.revision == 2
    assert plan.supersedes_plan_id == "plan-1"
    assert plan.economics_bindings == {
        "lgd": {"kind": "scalar", "value": 0.45},
        "pd": {"kind": "column", "column": "predicted_pd"},
    }
    assert plan.to_dict() == payload


@pytest.mark.parametrize(
    "bindings, message",
    [
        ({"lgd": {"kind": "scalar", "value": math.nan}}, "finite"),
        ({"lgd": {"kind": "scalar", "value": math.inf}}, "finite"),
        (
            {"pd": {"kind": "column", "column": "pd", "rows": [0.1, 0.2]}},
            "only kind and column",
        ),
        ({"pd": {"kind": "column", "column": ""}}, "non-empty"),
        ({"pd": {"kind": "series", "name": "pd"}}, "unsupported kind"),
    ],
)
def test_v2_monitoring_plan_rejects_unsafe_economics_bindings(bindings, message):
    with pytest.raises(StrategyError, match=message):
        MonitoringPlan(
            strategy_id="strategy-1",
            version=1,
            economics_bindings=bindings,
        )


def test_monitoring_plan_and_economics_hashes_are_canonical():
    first = MonitoringPlan(
        strategy_id="strategy-1",
        version=1,
        thresholds={
            "b": {"metric": "b", "warn": 2},
            "a": {"metric": "a", "warn": 1},
        },
        economics_bindings={
            "pd": {"kind": "column", "column": "pd"},
            "lgd": {"kind": "scalar", "value": 0.5},
        },
    )
    reordered = MonitoringPlan(
        strategy_id="strategy-1",
        version=1,
        thresholds={
            "a": {"warn": 1, "metric": "a"},
            "b": {"warn": 2, "metric": "b"},
        },
        economics_bindings={
            "lgd": {"value": 0.5, "kind": "scalar"},
            "pd": {"column": "pd", "kind": "column"},
        },
    )

    assert canonical_monitoring_plan_hash(first) == canonical_monitoring_plan_hash(
        reordered
    )
    assert canonical_economics_bindings_hash(
        first.economics_bindings
    ) == canonical_economics_bindings_hash(reordered.economics_bindings)
    assert len(canonical_monitoring_plan_hash(first)) == 64
