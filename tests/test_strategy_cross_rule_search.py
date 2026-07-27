from __future__ import annotations

from copy import deepcopy
import json

import pytest

from marvis.packs.strategy.cross_rule_search import (
    CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION,
    CrossRuleSearchError,
    canonical_cross_rule_search_result_json,
    canonical_cross_rule_trial_prefix,
    parse_cross_rule_search_result_json,
    search_cross_threshold_rules,
    validate_cross_rule_search_request,
    validate_cross_rule_search_result,
)


def _features() -> list[dict]:
    return [
        {
            "feature": "age",
            "method": "tree",
            "risk_direction": "decreasing",
            "thresholds": [25.0, 35.0],
            "excluded_values": [],
            "missing_count": 2,
            "missing_bad": 1,
        },
        {
            "feature": "income",
            "method": "tree",
            "risk_direction": "increasing",
            "thresholds": [5_000.0, 10_000.0],
            "excluded_values": [],
            "missing_count": 0,
            "missing_bad": 0,
        },
        {
            "feature": "score",
            "method": "tree",
            "risk_direction": "non_monotonic",
            "thresholds": [500.0],
            "excluded_values": [],
            "missing_count": 1,
            "missing_bad": 1,
        },
    ]


def _request(*, dimension: int = 2, max_trials: int = 8) -> dict:
    features = _features()
    prefix = canonical_cross_rule_trial_prefix(
        features,
        dimension=dimension,
        max_trials=max_trials,
    )
    trials = []
    for index, conditions in enumerate(prefix):
        count = 10 + index
        bad = min(count, 4 + index)
        trials.append(
            {
                "conditions": conditions,
                "count": count,
                "good": count - bad,
                "bad": bad,
                "loan_amount_sum": float(count * 100),
                "overdue_amount_sum": float(bad * 20),
            }
        )
    return {
        "schema_version": CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION,
        "source": {
            "candidate_id": "candidate-" + "a" * 32,
            "evidence_hash": "b" * 64,
            "sample_context_hash": "c" * 64,
        },
        "population": {
            "row_count": 100,
            "good": 80,
            "bad": 20,
            "loan_amount_sum": 10_000.0,
            "overdue_amount_sum": 400.0,
        },
        "dimension": dimension,
        "features": features,
        "constraints": {
            "min_lift": 1.5,
            "min_bad_count": 5,
            "max_hit_share": 0.25,
            "min_amount_lift": 1.0,
        },
        "trials": trials,
        "max_trials": max_trials,
    }


def test_two_dimensional_search_is_canonical_budgeted_and_never_selects_a_rule() -> None:
    request = _request()
    reordered = deepcopy(request)
    reordered["features"].reverse()
    reordered["trials"].reverse()

    result = search_cross_threshold_rules(request)

    # Trial rows are bound by conditions, so caller ordering is irrelevant.
    assert result == search_cross_threshold_rules(reordered)
    assert result["configuration"]["dimension"] == 2
    assert result["configuration"]["enumeration_policy"] == (
        "canonical_feature_combo_round_robin_threshold_prefix.v1"
    )
    assert result["search_space"] > result["evaluated"] == 8
    assert result["truncated"] is True
    assert result["eligible"] == sum(
        1 for item in result["rules"] if item["eligible"]
    )
    assert [item["rank"] for item in result["rules"]] == list(
        range(1, len(result["rules"]) + 1)
    )
    assert result["lifecycle"] == {
        "selected": False,
        "admitted": False,
        "applied": False,
        "adopted": False,
        "deployed": False,
    }
    canonical = canonical_cross_rule_search_result_json(result)
    assert all(
        forbidden not in canonical
        for forbidden in (
            '"winner"',
            '"champion"',
            '"recommended"',
            '"selected_rule"',
            '"rows"',
            '"row_ids"',
            '"assignments"',
        )
    )
    assert validate_cross_rule_search_result(result) == result
    assert json.loads(canonical) == result


def test_three_dimensional_search_uses_all_three_conditions_and_explicit_missing_branches() -> None:
    request = _request(dimension=3, max_trials=10)

    result = search_cross_threshold_rules(request)

    assert result["configuration"]["dimension"] == 3
    assert all(len(item["conditions"]) == 3 for item in result["rules"])
    flattened = [
        condition
        for item in result["rules"]
        for condition in item["conditions"]
    ]
    assert {item["operator"] for item in flattened} == {"gte", "lt"}
    assert any(item["include_missing"] for item in flattened)
    assert any(not item["include_missing"] for item in flattened)
    assert result["trial_accounting"]["used"]["row_evaluations"] == 1_000


def test_metrics_and_constraints_are_derived_from_raw_aggregate_counts() -> None:
    request = _request(max_trials=1)
    request["trials"][0].update(
        {
            "count": 20,
            "good": 12,
            "bad": 8,
            "loan_amount_sum": 2_000.0,
            "overdue_amount_sum": 200.0,
        }
    )

    rule = search_cross_threshold_rules(request)["rules"][0]

    assert rule["metrics"] == {
        "count": 20,
        "good": 12,
        "bad": 8,
        "hit_share": 0.2,
        "bad_rate": 0.4,
        "lift": 2.0,
        "bad_capture_rate": 0.4,
        "loan_amount_sum": 2_000.0,
        "overdue_amount_sum": 200.0,
        "amount_overdue_rate": 0.1,
        "amount_lift": 2.5,
    }
    assert rule["eligible"] is True
    assert rule["constraint_failures"] == []

    request["constraints"]["min_lift"] = 2.1
    request["constraints"]["min_bad_count"] = 9
    request["constraints"]["max_hit_share"] = 0.1
    request["constraints"]["min_amount_lift"] = 3.0
    rule = search_cross_threshold_rules(request)["rules"][0]
    assert rule["eligible"] is False
    assert rule["constraint_failures"] == [
        "lift_below_minimum",
        "bad_count_below_minimum",
        "hit_share_above_maximum",
        "amount_lift_below_minimum",
    ]


def test_request_rejects_non_exact_prefix_duplicates_and_unbounded_work() -> None:
    missing = _request()
    missing["trials"].pop()
    with pytest.raises(CrossRuleSearchError, match="canonical trial prefix"):
        validate_cross_rule_search_request(missing)

    duplicate = _request()
    duplicate["trials"][1]["conditions"] = deepcopy(
        duplicate["trials"][0]["conditions"]
    )
    with pytest.raises(CrossRuleSearchError, match="duplicate trial"):
        validate_cross_rule_search_request(duplicate)

    bad_dimension = _request()
    bad_dimension["dimension"] = 4
    with pytest.raises(CrossRuleSearchError, match="dimension must be 2 or 3"):
        validate_cross_rule_search_request(bad_dimension)

    too_many_thresholds = _request()
    too_many_thresholds["features"][0]["thresholds"] = [
        float(index) for index in range(9)
    ]
    with pytest.raises(CrossRuleSearchError, match="1..8"):
        validate_cross_rule_search_request(too_many_thresholds)

    too_many_trials = _request()
    too_many_trials["max_trials"] = 5_001
    with pytest.raises(CrossRuleSearchError, match="between 1 and 5000"):
        validate_cross_rule_search_request(too_many_trials)

    row_budget = _request(max_trials=8)
    row_budget["population"] = {
        **row_budget["population"],
        "row_count": 10_000_000,
        "good": 8_000_000,
        "bad": 2_000_000,
    }
    with pytest.raises(CrossRuleSearchError, match="row evaluations"):
        validate_cross_rule_search_request(row_budget)


def test_amount_constraints_require_amount_evidence_and_result_hash_is_tamper_evident() -> None:
    missing_amount = _request()
    missing_amount["population"]["loan_amount_sum"] = None
    missing_amount["population"]["overdue_amount_sum"] = None
    for trial in missing_amount["trials"]:
        trial["loan_amount_sum"] = None
        trial["overdue_amount_sum"] = None
    with pytest.raises(CrossRuleSearchError, match="min_amount_lift"):
        validate_cross_rule_search_request(missing_amount)

    result = search_cross_threshold_rules(_request())
    tampered = deepcopy(result)
    tampered["rules"][0]["metrics"]["lift"] += 0.1
    with pytest.raises(
        CrossRuleSearchError,
        match="deterministically derived|content_hash",
    ):
        validate_cross_rule_search_result(tampered)

    with pytest.raises(CrossRuleSearchError, match="duplicate key: search_id"):
        parse_cross_rule_search_result_json(
            '{"search_id":"one","search_id":"two"}'
        )
