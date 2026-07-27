from __future__ import annotations

from copy import deepcopy
import json

import pytest

from marvis.packs.strategy.cross_candidate_search import (
    CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
    CrossCandidateSearchError,
    canonical_cross_candidate_search_result_json,
    parse_cross_candidate_search_result_json,
    search_cross_candidate_pairs,
    validate_cross_candidate_search_request,
    validate_cross_candidate_search_result,
)


def _fingerprint(seed: str) -> dict:
    digest = (seed * 64)[:64]
    return {
        "asset_id": f"candidate-asset-{digest[:32]}",
        "asset_hash": digest,
        "measurement_hash": digest,
        "matrix_hash": digest,
        "summary_hash": digest,
    }


def _request() -> dict:
    return {
        "schema_version": CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "source": {
            "candidate_id": "candidate-" + "a" * 32,
            "evidence_hash": "b" * 64,
            "sample_context_hash": "c" * 64,
        },
        "population": {
            "row_count": 100,
            "good": 80,
            "bad": 20,
        },
        "features": [
            {"feature": "score", "method": "tree", "axis_iv": 0.18, "bin_count": 3},
            {"feature": "age", "method": "equal_width", "axis_iv": 0.1, "bin_count": 2},
            {"feature": "income", "method": "equal_frequency", "axis_iv": 0.12, "bin_count": 2},
        ],
        "pair_trials": [
            {
                "x_feature": "age",
                "y_feature": "income",
                "cross_total_iv": 0.25,
                "cell_count": 4,
                "empty_cell_count": 0,
                "min_nonempty_cell_count": 10,
                "asset_fingerprint": _fingerprint("1"),
            },
            {
                "x_feature": "income",
                "y_feature": "score",
                "cross_total_iv": 0.28,
                "cell_count": 6,
                "empty_cell_count": 0,
                "min_nonempty_cell_count": 5,
                "asset_fingerprint": _fingerprint("2"),
            },
            {
                "x_feature": "score",
                "y_feature": "age",
                "cross_total_iv": 0.31,
                "cell_count": 6,
                "empty_cell_count": 1,
                "min_nonempty_cell_count": 3,
                "asset_fingerprint": _fingerprint("3"),
            },
        ],
        "max_pairs": 3,
    }


def test_search_is_order_independent_deduplicates_transposed_axes_and_ranks_evidence() -> None:
    request = _request()
    reordered = deepcopy(request)
    reordered["features"].reverse()
    reordered["pair_trials"].reverse()
    for trial in reordered["pair_trials"]:
        trial["x_feature"], trial["y_feature"] = (
            trial["y_feature"],
            trial["x_feature"],
        )

    result = search_cross_candidate_pairs(request)

    assert result == search_cross_candidate_pairs(reordered)
    assert result["search_space"] == 3
    assert result["evaluated"] == 3
    assert result["truncated"] is False
    assert [
        (pair["x_feature"], pair["y_feature"], pair["rank"])
        for pair in result["pairs"]
    ] == [
        ("age", "income", 1),
        ("income", "score", 2),
        ("age", "score", 3),
    ]
    assert result["pairs"][0]["interaction_gain_iv"] == pytest.approx(0.13)
    assert result["pairs"][1]["interaction_gain_iv"] == pytest.approx(0.1)
    assert result["pairs"][2]["eligible"] is False
    assert result["pairs"][2]["empty_cell_share"] == pytest.approx(1 / 6)
    assert result["lifecycle"] == {
        "selected": False,
        "admitted": False,
        "applied": False,
        "adopted": False,
        "deployed": False,
    }
    assert validate_cross_candidate_search_result(result) == result
    assert json.loads(canonical_cross_candidate_search_result_json(result)) == result


def test_round_robin_prefix_truncation_and_trial_accounting_are_explicit() -> None:
    request = _request()
    request["max_pairs"] = 2
    request["pair_trials"] = request["pair_trials"][:2]

    result = search_cross_candidate_pairs(request)

    assert result["configuration"]["enumeration_policy"] == (
        "canonical_round_robin_pair_prefix.v1"
    )
    assert result["search_space"] == 3
    assert result["evaluated"] == 2
    assert result["truncated"] is True
    assert result["trial_accounting"] == {
        "limits": {
            "max_features": 20,
            "max_pairs": 190,
            "max_cells_per_pair": 400,
            "max_pair_row_evaluations": 50_000_000,
            "max_axis_bin_row_evaluations": 50_000_000,
            "max_derived_cells": 50_000,
            "max_artifact_bytes": 67_108_864,
        },
        "used": {
            "features": 3,
            "pairs": 2,
            "pair_row_evaluations": 200,
            "axis_bin_row_evaluations": 700,
            "derived_cells": 10,
        },
    }


def test_sparse_pair_is_kept_as_aggregate_evidence_but_is_not_eligible() -> None:
    request = _request()
    sparse = request["pair_trials"][0]
    sparse["empty_cell_count"] = 3
    sparse["min_nonempty_cell_count"] = 100

    result = search_cross_candidate_pairs(request)

    pair = next(
        item
        for item in result["pairs"]
        if (item["x_feature"], item["y_feature"]) == ("age", "income")
    )
    assert pair["eligible"] is False
    assert pair["empty_cell_share"] == 0.75
    canonical = canonical_cross_candidate_search_result_json(result)
    assert all(
        forbidden not in canonical
        for forbidden in (
            '"rows"',
            '"row_ids"',
            '"target"',
            '"assignments"',
            '"winner"',
            '"champion"',
            '"recommended"',
        )
    )


def test_request_fails_closed_on_duplicate_pairs_unbounded_work_and_shape_drift() -> None:
    duplicate = _request()
    duplicate["pair_trials"][1] = deepcopy(duplicate["pair_trials"][0])
    duplicate["pair_trials"][1]["x_feature"], duplicate["pair_trials"][1][
        "y_feature"
    ] = (
        duplicate["pair_trials"][1]["y_feature"],
        duplicate["pair_trials"][1]["x_feature"],
    )
    with pytest.raises(CrossCandidateSearchError, match="duplicate canonical pair"):
        validate_cross_candidate_search_request(duplicate)

    too_many_features = _request()
    too_many_features["features"] = [
        {
            "feature": f"f{index:02d}",
            "method": "equal_width",
            "axis_iv": 0.1,
            "bin_count": 2,
        }
        for index in range(21)
    ]
    too_many_features["pair_trials"] = []
    with pytest.raises(CrossCandidateSearchError, match="2..20"):
        validate_cross_candidate_search_request(too_many_features)

    oversized_pair = _request()
    oversized_pair["features"][0]["bin_count"] = 201
    with pytest.raises(CrossCandidateSearchError, match="400 cells"):
        validate_cross_candidate_search_request(oversized_pair)

    missing_prefix = _request()
    missing_prefix["pair_trials"].pop()
    with pytest.raises(CrossCandidateSearchError, match="round-robin prefix"):
        validate_cross_candidate_search_request(missing_prefix)

    row_budget = _request()
    row_budget["population"] = {
        "row_count": 10_000_000,
        "good": 8_000_000,
        "bad": 2_000_000,
    }
    for trial in row_budget["pair_trials"]:
        trial["min_nonempty_cell_count"] = 1
    with pytest.raises(
        CrossCandidateSearchError,
        match="axis_bin_row_evaluations exceeds hard budget",
    ):
        validate_cross_candidate_search_request(row_budget)

    pair_budget = _request()
    pair_budget["max_pairs"] = 191
    with pytest.raises(CrossCandidateSearchError, match="between 1 and 190"):
        validate_cross_candidate_search_request(pair_budget)


def test_parser_rejects_duplicate_json_keys_and_result_hash_tampering() -> None:
    with pytest.raises(CrossCandidateSearchError, match="duplicate key: search_id"):
        parse_cross_candidate_search_result_json(
            '{"search_id":"one","search_id":"two"}'
        )

    result = search_cross_candidate_pairs(_request())
    tampered = deepcopy(result)
    tampered["pairs"][0]["cross_total_iv"] += 0.01
    with pytest.raises(CrossCandidateSearchError, match="content_hash"):
        validate_cross_candidate_search_result(tampered)
