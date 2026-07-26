from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from marvis.packs.strategy.voting_candidate_search import (
    MAX_COMBINATIONS_BUDGET,
    MAX_RESULT_DISTRIBUTION_BINS,
    VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
    VotingCandidateSearchError,
    canonical_voting_candidate_search_request_json,
    canonical_voting_candidate_search_result_json,
    parse_voting_candidate_search_request_json,
    parse_voting_candidate_search_result_json,
    search_voting_candidate_combinations,
    validate_voting_candidate_search_request,
    validate_voting_candidate_search_result,
)


def _request() -> dict:
    return {
        "schema_version": VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "candidate_ids": ["rule-c", "rule-a", "rule-b"],
        "hit_matrix": [
            [False, False, True, True, True, False],
            [True, True, False, False, True, False],
            [False, True, True, False, True, False],
        ],
        "target": [0, 1, 1, 0, 1, 0],
        "weights": [1, 2, 3, 4, 5, 6],
        "amounts": [10, 20, 30, 40, 50, 60],
        "member_count": 2,
        "n": 2,
        "objective": {
            "metric": "bad_capture_rate",
            "direction": "maximize",
        },
        "constraints": [],
        "include": [],
        "exclude": [],
        "max_combinations": 10,
    }


def test_search_is_deterministic_and_returns_reconcilable_voting_metrics() -> None:
    request = _request()
    before = deepcopy(request)

    result = search_voting_candidate_combinations(request)

    assert request == before
    assert result["search_space"] == 3
    assert result["evaluated"] == 3
    assert result["truncated"] is False
    assert result["eligible"] == 3
    assert [item["member_ids"] for item in result["combinations"]] == [
        ["rule-a", "rule-b"],
        ["rule-b", "rule-c"],
        ["rule-a", "rule-c"],
    ]
    best = result["combinations"][0]
    assert best["rank"] == 1
    assert best["hit_count_distribution"] == [
        {"member_hits": 0, "row_count": 2, "row_share": 1 / 3},
        {"member_hits": 1, "row_count": 2, "row_share": 1 / 3},
        {"member_hits": 2, "row_count": 2, "row_share": 1 / 3},
    ]
    assert best["metrics"] == {
        "population_count": 6,
        "hit_count": 2,
        "hit_share": 1 / 3,
        "good_count": 0,
        "bad_count": 2,
        "bad_rate": 1.0,
        "base_bad_rate": 0.5,
        "lift": 2.0,
        "bad_capture_rate": 2 / 3,
        "weighted_hit_total": 7.0,
        "weighted_hit_share": 1 / 3,
        "weighted_good_total": 0.0,
        "weighted_bad_total": 7.0,
        "weighted_bad_rate": 1.0,
        "weighted_bad_capture_rate": 0.7,
        "hit_amount": 70.0,
        "hit_amount_share": 1 / 3,
        "good_amount": 0.0,
        "bad_amount": 70.0,
        "bad_amount_rate": 1.0,
        "bad_amount_capture_rate": 0.7,
    }
    assert validate_voting_candidate_search_result(result) == result
    assert canonical_voting_candidate_search_result_json(
        result
    ) == canonical_voting_candidate_search_result_json(
        search_voting_candidate_combinations(request)
    )


def test_hard_budget_include_exclude_and_constraints_are_explicit() -> None:
    request = _request()
    request["candidate_ids"].append("rule-d")
    request["hit_matrix"].append([True, False, False, True, False, True])
    request["include"] = ["rule-a"]
    request["exclude"] = ["rule-d"]
    request["objective"] = {"metric": "hit_share", "direction": "minimize"}
    request["constraints"] = [{"metric": "hit_share", "operator": "lte", "value": 0.2}]
    request["max_combinations"] = 1

    result = search_voting_candidate_combinations(request)

    assert result["search_space"] == 2
    assert result["evaluated"] == 1
    assert result["truncated"] is True
    assert result["eligible"] == 0
    assert result["combinations"][0]["member_ids"] == ["rule-a", "rule-b"]
    assert result["combinations"][0]["eligible"] is False
    assert result["combinations"][0]["constraint_failures"] == [
        {
            "metric": "hit_share",
            "operator": "lte",
            "threshold": 0.2,
            "actual": 1 / 3,
        }
    ]


def test_candidate_input_order_does_not_change_search_identity_or_ranking() -> None:
    request = _request()
    reversed_request = deepcopy(request)
    paired = list(zip(request["candidate_ids"], request["hit_matrix"], strict=True))
    paired.reverse()
    reversed_request["candidate_ids"] = [candidate_id for candidate_id, _ in paired]
    reversed_request["hit_matrix"] = [hits for _, hits in paired]

    assert search_voting_candidate_combinations(
        reversed_request
    ) == search_voting_candidate_combinations(request)


def test_optional_observations_are_null_and_result_has_no_raw_rows_or_adoption() -> (
    None
):
    request = _request()
    request["weights"] = None
    request["amounts"] = None

    result = search_voting_candidate_combinations(request)

    metrics = result["combinations"][0]["metrics"]
    assert result["population"]["weight"] == {
        "available": False,
        "total": None,
        "good_total": None,
        "bad_total": None,
    }
    assert result["population"]["amount"] == {
        "available": False,
        "total": None,
        "good_total": None,
        "bad_total": None,
    }
    assert all(
        metrics[field] is None
        for field in (
            "weighted_hit_total",
            "weighted_hit_share",
            "weighted_good_total",
            "weighted_bad_total",
            "weighted_bad_rate",
            "weighted_bad_capture_rate",
            "hit_amount",
            "hit_amount_share",
            "good_amount",
            "bad_amount",
            "bad_amount_rate",
            "bad_amount_capture_rate",
        )
    )
    assert {
        "target",
        "hit_matrix",
        "weights",
        "amounts",
        "winner",
        "champion",
        "selected",
        "pool",
        "action",
    }.isdisjoint(result)


def test_request_contract_rejects_ambiguous_nonfinite_or_unbudgeted_inputs() -> None:
    invalid_cases: list[tuple[dict, str]] = []

    extra = _request()
    extra["unexpected"] = True
    invalid_cases.append((extra, "unsupported fields"))

    numeric_hits = _request()
    numeric_hits["hit_matrix"][0][0] = 1
    invalid_cases.append((numeric_hits, "must be boolean"))

    boolean_target = _request()
    boolean_target["target"][0] = True
    invalid_cases.append((boolean_target, "integer 0 or 1"))

    nonfinite = _request()
    nonfinite["weights"][0] = float("inf")
    invalid_cases.append((nonfinite, "finite number"))

    missing_observation = _request()
    missing_observation["weights"] = None
    missing_observation["objective"] = {
        "metric": "weighted_bad_rate",
        "direction": "maximize",
    }
    invalid_cases.append((missing_observation, "requires weights"))

    overlap = _request()
    overlap["include"] = ["rule-a"]
    overlap["exclude"] = ["rule-a"]
    invalid_cases.append((overlap, "must be disjoint"))

    unbudgeted = _request()
    unbudgeted["max_combinations"] = MAX_COMBINATIONS_BUDGET + 1
    invalid_cases.append((unbudgeted, "must be between"))

    invalid_rate = _request()
    invalid_rate["constraints"] = [
        {"metric": "hit_share", "operator": "gte", "value": 1.1}
    ]
    invalid_cases.append((invalid_rate, "must be in \\[0, 1\\]"))

    for payload, message in invalid_cases:
        with pytest.raises(VotingCandidateSearchError, match=message):
            validate_voting_candidate_search_request(payload)


def test_request_rejects_a_budget_that_would_overflow_the_result_shape() -> None:
    candidate_count = 128
    request = {
        "schema_version": VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "candidate_ids": [f"rule-{index:03d}" for index in range(candidate_count)],
        "hit_matrix": [[False] for _ in range(candidate_count)],
        "target": [0],
        "weights": None,
        "amounts": None,
        "member_count": 100,
        "n": 50,
        "objective": {"metric": "bad_capture_rate", "direction": "maximize"},
        "constraints": [],
        "include": [],
        "exclude": [],
        "max_combinations": MAX_COMBINATIONS_BUDGET,
    }

    with pytest.raises(
        VotingCandidateSearchError,
        match=f"exceeds {MAX_RESULT_DISTRIBUTION_BINS} distribution bins",
    ):
        validate_voting_candidate_search_request(request)


def test_search_space_is_exact_even_when_it_exceeds_machine_integer_range() -> None:
    candidate_count = 128
    request = {
        "schema_version": VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "candidate_ids": [f"rule-{index:03d}" for index in range(candidate_count)],
        "hit_matrix": [[index % 2 == 0] for index in range(candidate_count)],
        "target": [1],
        "weights": None,
        "amounts": None,
        "member_count": 28,
        "n": 14,
        "objective": {"metric": "bad_capture_rate", "direction": "maximize"},
        "constraints": [],
        "include": [],
        "exclude": [],
        "max_combinations": 1,
    }

    result = search_voting_candidate_combinations(request)

    assert result["search_space"] > 2**63
    assert result["evaluated"] == 1
    assert result["truncated"] is True


def test_request_and_result_json_are_canonical_bounded_and_duplicate_safe() -> None:
    request = _request()
    canonical_request = canonical_voting_candidate_search_request_json(request)
    result = search_voting_candidate_combinations(request)
    canonical_result = canonical_voting_candidate_search_result_json(result)

    assert parse_voting_candidate_search_request_json(canonical_request) == (
        validate_voting_candidate_search_request(request)
    )
    assert parse_voting_candidate_search_result_json(canonical_result) == result
    with pytest.raises(VotingCandidateSearchError, match="duplicate JSON key"):
        parse_voting_candidate_search_request_json(
            '{"schema_version":"a","schema_version":"b"}'
        )
    with pytest.raises(VotingCandidateSearchError, match="duplicate JSON key"):
        parse_voting_candidate_search_result_json(
            '{"schema_version":"a","schema_version":"b"}'
        )


def test_result_rejects_rehashed_metric_conservation_tampering() -> None:
    result = search_voting_candidate_combinations(_request())
    tampered = deepcopy(result)
    tampered["combinations"][0]["metrics"]["good_count"] = 1
    body = {key: value for key, value in tampered.items() if key != "content_hash"}
    tampered["content_hash"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(
        VotingCandidateSearchError,
        match="good_count \\+ bad_count must equal hit_count",
    ):
        validate_voting_candidate_search_result(tampered)


@pytest.mark.parametrize(
    ("target", "forged_good_count", "expected_error"),
    [
        ([1, 1], 2, "good_count must not exceed population.good_count"),
        ([0, 0], 0, "bad_count must not exceed population.bad_count"),
    ],
)
def test_result_rejects_rehashed_class_population_tampering(
    target: list[int],
    forged_good_count: int,
    expected_error: str,
) -> None:
    request = {
        **_request(),
        "candidate_ids": ["rule-a", "rule-b"],
        "hit_matrix": [[True, True], [True, True]],
        "target": target,
        "weights": None,
        "amounts": None,
        "member_count": 2,
        "n": 1,
        "objective": {"metric": "hit_share", "direction": "maximize"},
    }
    result = search_voting_candidate_combinations(request)
    tampered = deepcopy(result)
    metrics = tampered["combinations"][0]["metrics"]
    metrics["good_count"] = forged_good_count
    metrics["bad_count"] = 2 - forged_good_count
    metrics["bad_rate"] = metrics["bad_count"] / 2
    metrics["lift"] = (
        metrics["bad_rate"] / tampered["population"]["bad_rate"]
        if tampered["population"]["bad_rate"]
        else 0.0
    )
    metrics["bad_capture_rate"] = (
        metrics["bad_count"] / tampered["population"]["bad_count"]
        if tampered["population"]["bad_count"]
        else 0.0
    )
    body = {key: value for key, value in tampered.items() if key != "content_hash"}
    tampered["content_hash"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(VotingCandidateSearchError, match=expected_error):
        validate_voting_candidate_search_result(tampered)


@pytest.mark.parametrize(
    ("observation", "good_metric", "bad_metric", "rate_metric", "capture_metric"),
    [
        (
            "weights",
            "weighted_good_total",
            "weighted_bad_total",
            "weighted_bad_rate",
            "weighted_bad_capture_rate",
        ),
        (
            "amounts",
            "good_amount",
            "bad_amount",
            "bad_amount_rate",
            "bad_amount_capture_rate",
        ),
    ],
)
def test_result_rejects_rehashed_good_observation_population_tampering(
    observation: str,
    good_metric: str,
    bad_metric: str,
    rate_metric: str,
    capture_metric: str,
) -> None:
    request = {
        **_request(),
        "candidate_ids": ["rule-a", "rule-b"],
        "hit_matrix": [[True, True], [True, True]],
        "target": [0, 1],
        "weights": [1, 100] if observation == "weights" else None,
        "amounts": [1, 100] if observation == "amounts" else None,
        "member_count": 2,
        "n": 1,
        "objective": {"metric": "hit_share", "direction": "maximize"},
    }
    result = search_voting_candidate_combinations(request)
    tampered = deepcopy(result)
    metrics = tampered["combinations"][0]["metrics"]
    metrics[good_metric] = 101.0
    metrics[bad_metric] = 0.0
    metrics[rate_metric] = 0.0
    metrics[capture_metric] = 0.0
    body = {key: value for key, value in tampered.items() if key != "content_hash"}
    tampered["content_hash"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(
        VotingCandidateSearchError,
        match=f"{good_metric} must not exceed population good total",
    ):
        validate_voting_candidate_search_result(tampered)


def test_result_rejects_objectives_without_the_declared_observation() -> None:
    request = _request()
    request["weights"] = None
    result = search_voting_candidate_combinations(request)
    tampered = deepcopy(result)
    tampered["configuration"]["objective"] = {
        "metric": "weighted_bad_rate",
        "direction": "maximize",
    }

    with pytest.raises(
        VotingCandidateSearchError,
        match="configuration objective requires weight observations",
    ):
        validate_voting_candidate_search_result(tampered)
