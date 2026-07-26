"""Deterministic, budgeted search over authenticated Voting rule hit matrices.

This module is a pure domain seam.  Its caller must authenticate the candidate
rules and materialize their row-aligned hit matrix.  The seam only validates the
finite JSON contract, enumerates canonical ``n_of_k`` combinations within an
explicit hard budget, and returns reconciled measurements.  It has no authority
to register a candidate, mutate a Strategy Pool, select a winner, or deploy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
from itertools import combinations, islice
import json
import math
from numbers import Integral, Real
import re
from typing import Any

from marvis.packs.strategy.errors import StrategyError


VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION = (
    "strategy.voting-candidate-search-request.v1"
)
VOTING_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION = (
    "strategy.voting-candidate-search-result.v1"
)
VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION = "strategy.voting-candidate-search/1"

MAX_CANDIDATES = 128
MAX_SEARCH_SPACE = math.comb(MAX_CANDIDATES, MAX_CANDIDATES // 2)
MAX_ROWS = 1_000_000
MAX_MATRIX_CELLS = 5_000_000
MAX_COMBINATIONS_BUDGET = 10_000
MAX_EVALUATION_CELLS = 50_000_000
MAX_RESULT_DISTRIBUTION_BINS = 100_000
MAX_CONSTRAINTS = 32
MAX_JSON_BYTES = 64 * 1024 * 1024

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SEARCH_ID_RE = re.compile(r"^voting-search-[0-9a-f]{32}$")
_COMBO_ID_RE = re.compile(r"^voting-combo-[0-9a-f]{32}$")
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_ids",
        "hit_matrix",
        "target",
        "weights",
        "amounts",
        "member_count",
        "n",
        "objective",
        "constraints",
        "include",
        "exclude",
        "max_combinations",
    }
)
_OBJECTIVE_FIELDS = frozenset({"metric", "direction"})
_CONSTRAINT_FIELDS = frozenset({"metric", "operator", "value"})
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "search_id",
        "request_hash",
        "configuration",
        "population",
        "search_space",
        "evaluated",
        "truncated",
        "eligible",
        "combinations",
        "producer_version",
        "content_hash",
    }
)
_CONFIGURATION_FIELDS = frozenset(
    {
        "candidate_ids",
        "member_count",
        "n",
        "objective",
        "constraints",
        "include",
        "exclude",
        "max_combinations",
    }
)
_POPULATION_FIELDS = frozenset(
    {"row_count", "good_count", "bad_count", "bad_rate", "weight", "amount"}
)
_OBSERVATION_SUMMARY_FIELDS = frozenset(
    {"available", "total", "good_total", "bad_total"}
)
_COMBINATION_FIELDS = frozenset(
    {
        "combo_id",
        "member_ids",
        "n",
        "rank",
        "objective_value",
        "eligible",
        "constraint_failures",
        "hit_count_distribution",
        "metrics",
    }
)
_FAILURE_FIELDS = frozenset({"metric", "operator", "threshold", "actual"})
_DISTRIBUTION_FIELDS = frozenset({"member_hits", "row_count", "row_share"})
_METRIC_FIELDS = frozenset(
    {
        "population_count",
        "hit_count",
        "hit_share",
        "good_count",
        "bad_count",
        "bad_rate",
        "base_bad_rate",
        "lift",
        "bad_capture_rate",
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
    }
)
_BASE_METRICS = frozenset(
    {
        "hit_count",
        "hit_share",
        "good_count",
        "bad_count",
        "bad_rate",
        "lift",
        "bad_capture_rate",
    }
)
_WEIGHTED_METRICS = frozenset(
    {
        "weighted_hit_total",
        "weighted_hit_share",
        "weighted_good_total",
        "weighted_bad_total",
        "weighted_bad_rate",
        "weighted_bad_capture_rate",
    }
)
_AMOUNT_METRICS = frozenset(
    {
        "hit_amount",
        "hit_amount_share",
        "good_amount",
        "bad_amount",
        "bad_amount_rate",
        "bad_amount_capture_rate",
    }
)
_OBJECTIVE_METRICS = _BASE_METRICS | _WEIGHTED_METRICS | _AMOUNT_METRICS
_RATE_METRICS = frozenset(
    {
        "hit_share",
        "bad_rate",
        "bad_capture_rate",
        "weighted_hit_share",
        "weighted_bad_rate",
        "weighted_bad_capture_rate",
        "hit_amount_share",
        "bad_amount_rate",
        "bad_amount_capture_rate",
    }
)


class VotingCandidateSearchError(StrategyError):
    """The Voting candidate search contract or reconciliation was violated."""


def validate_voting_candidate_search_request(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical, row-aligned request after strict validation."""

    if not isinstance(payload, Mapping):
        raise VotingCandidateSearchError("Voting search request must be an object")
    _exact_fields(payload, _REQUEST_FIELDS, "Voting search request")
    if payload["schema_version"] != VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION:
        raise VotingCandidateSearchError(
            "schema_version must be " + VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION
        )

    raw_ids = _candidate_id_array(payload["candidate_ids"], "candidate_ids")
    if not raw_ids or len(raw_ids) > MAX_CANDIDATES:
        raise VotingCandidateSearchError(
            f"candidate_ids must contain 1..{MAX_CANDIDATES} ids"
        )
    if len(set(raw_ids)) != len(raw_ids):
        raise VotingCandidateSearchError("candidate_ids must be unique")

    target = _binary_array(payload["target"], "target")
    row_count = len(target)
    if row_count < 1 or row_count > MAX_ROWS:
        raise VotingCandidateSearchError(f"target must contain 1..{MAX_ROWS} rows")
    if len(raw_ids) * row_count > MAX_MATRIX_CELLS:
        raise VotingCandidateSearchError(
            f"candidate hit matrix exceeds {MAX_MATRIX_CELLS} cells"
        )
    hit_matrix = _hit_matrix(
        payload["hit_matrix"],
        candidate_count=len(raw_ids),
        row_count=row_count,
    )
    weights = _optional_observation_array(
        payload["weights"], "weights", row_count=row_count
    )
    amounts = _optional_observation_array(
        payload["amounts"], "amounts", row_count=row_count
    )
    member_count = _bounded_int(
        payload["member_count"],
        "member_count",
        minimum=2,
        maximum=len(raw_ids),
    )
    n = _bounded_int(payload["n"], "n", minimum=1, maximum=member_count)
    include = _candidate_id_array(payload["include"], "include")
    exclude = _candidate_id_array(payload["exclude"], "exclude")
    if len(set(include)) != len(include):
        raise VotingCandidateSearchError("include must not contain duplicate ids")
    if len(set(exclude)) != len(exclude):
        raise VotingCandidateSearchError("exclude must not contain duplicate ids")
    candidate_set = set(raw_ids)
    unknown = sorted((set(include) | set(exclude)) - candidate_set)
    if unknown:
        raise VotingCandidateSearchError(
            "include/exclude contains unknown candidate ids: " + ", ".join(unknown)
        )
    overlap = sorted(set(include) & set(exclude))
    if overlap:
        raise VotingCandidateSearchError(
            "include and exclude must be disjoint: " + ", ".join(overlap)
        )
    if len(include) > member_count:
        raise VotingCandidateSearchError(
            "include cannot contain more ids than member_count"
        )
    eligible_candidate_count = len(candidate_set - set(exclude))
    if member_count > eligible_candidate_count:
        raise VotingCandidateSearchError(
            "member_count exceeds candidates remaining after exclude"
        )

    objective = _normalize_objective(
        payload["objective"],
        weights_available=weights is not None,
        amounts_available=amounts is not None,
    )
    constraints = _normalize_constraints(
        payload["constraints"],
        weights_available=weights is not None,
        amounts_available=amounts is not None,
    )
    max_combinations = _bounded_int(
        payload["max_combinations"],
        "max_combinations",
        minimum=1,
        maximum=MAX_COMBINATIONS_BUDGET,
    )
    _validate_planned_work(
        candidate_ids=raw_ids,
        member_count=member_count,
        include=include,
        exclude=exclude,
        max_combinations=max_combinations,
        row_count=row_count,
    )

    order = sorted(range(len(raw_ids)), key=raw_ids.__getitem__)
    return {
        "schema_version": VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "candidate_ids": [raw_ids[index] for index in order],
        "hit_matrix": [hit_matrix[index] for index in order],
        "target": target,
        "weights": weights,
        "amounts": amounts,
        "member_count": member_count,
        "n": n,
        "objective": objective,
        "constraints": constraints,
        "include": sorted(include),
        "exclude": sorted(exclude),
        "max_combinations": max_combinations,
    }


def canonical_voting_candidate_search_request_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole byte-stable JSON representation of a valid request."""

    return _canonical_json(validate_voting_candidate_search_request(payload))


def parse_voting_candidate_search_request_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    """Parse a bounded JSON request while rejecting duplicate keys."""

    return validate_voting_candidate_search_request(
        _parse_json_object(raw, "Voting search request")
    )


def search_voting_candidate_combinations(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate canonical combinations up to the caller's explicit hard budget."""

    request = validate_voting_candidate_search_request(payload)
    request_hash = _sha256(_canonical_json(request))
    candidate_ids = request["candidate_ids"]
    include = request["include"]
    include_set = set(include)
    excluded = set(request["exclude"])
    optional = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in include_set and candidate_id not in excluded
    ]
    optional_count = request["member_count"] - len(include)
    search_space = math.comb(len(optional), optional_count)
    raw_combinations = list(
        islice(
            combinations(optional, optional_count),
            request["max_combinations"],
        )
    )
    hit_by_id = dict(zip(candidate_ids, request["hit_matrix"], strict=True))
    population = _population_summary(
        request["target"],
        weights=request["weights"],
        amounts=request["amounts"],
    )
    evaluated = [
        _evaluate_combination(
            sorted([*include, *members]),
            n=request["n"],
            hit_by_id=hit_by_id,
            target=request["target"],
            weights=request["weights"],
            amounts=request["amounts"],
            population=population,
            objective=request["objective"],
            constraints=request["constraints"],
        )
        for members in raw_combinations
    ]
    ranked = sorted(
        evaluated,
        key=lambda item: _result_sort_key(item, request["objective"]),
    )
    ranked = [{**item, "rank": index} for index, item in enumerate(ranked, 1)]
    configuration = _configuration_from_request(request)
    search_id = _search_id(request_hash, configuration)
    body = {
        "schema_version": VOTING_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION,
        "search_id": search_id,
        "request_hash": request_hash,
        "configuration": configuration,
        "population": population,
        "search_space": search_space,
        "evaluated": len(ranked),
        "truncated": len(ranked) < search_space,
        "eligible": sum(bool(item["eligible"]) for item in ranked),
        "combinations": ranked,
        "producer_version": VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION,
    }
    result = {**body, "content_hash": _sha256(_canonical_json(body))}
    return validate_voting_candidate_search_result(result)


def validate_voting_candidate_search_result(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate result schemas, identities, ranks, budgets, and conservation."""

    if not isinstance(payload, Mapping):
        raise VotingCandidateSearchError("Voting search result must be an object")
    _exact_fields(payload, _RESULT_FIELDS, "Voting search result")
    if payload["schema_version"] != VOTING_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION:
        raise VotingCandidateSearchError(
            "result.schema_version must be "
            + VOTING_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION
        )
    if payload["producer_version"] != VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION:
        raise VotingCandidateSearchError(
            "producer_version must be " + VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION
        )
    request_hash = _hash(payload["request_hash"], "request_hash")
    configuration = _normalize_configuration(payload["configuration"])
    population = _normalize_population(payload["population"])
    _validate_configuration_observation_availability(configuration, population)
    _validate_planned_work(
        candidate_ids=configuration["candidate_ids"],
        member_count=configuration["member_count"],
        include=configuration["include"],
        exclude=configuration["exclude"],
        max_combinations=configuration["max_combinations"],
        row_count=population["row_count"],
    )
    search_space = _bounded_int(
        payload["search_space"],
        "search_space",
        minimum=0,
        maximum=MAX_SEARCH_SPACE,
    )
    expected_space, expected_members = _expected_member_sets(configuration)
    if search_space != expected_space:
        raise VotingCandidateSearchError(
            "search_space does not match include/exclude/member_count"
        )
    expected_evaluated = min(search_space, configuration["max_combinations"])
    evaluated = _non_negative_int(payload["evaluated"], "evaluated")
    if evaluated != expected_evaluated:
        raise VotingCandidateSearchError(
            "evaluated must equal min(search_space, max_combinations)"
        )
    truncated = _strict_bool(payload["truncated"], "truncated")
    if truncated != (evaluated < search_space):
        raise VotingCandidateSearchError(
            "truncated must state whether the hard budget cut search_space"
        )
    combinations_value = _array(payload["combinations"], "combinations")
    if len(combinations_value) != evaluated:
        raise VotingCandidateSearchError("combinations length must equal evaluated")
    normalized_combinations = [
        _normalize_result_combination(
            value,
            configuration=configuration,
            population=population,
        )
        for value in combinations_value
    ]
    member_sets = [tuple(item["member_ids"]) for item in normalized_combinations]
    if len(set(member_sets)) != len(member_sets):
        raise VotingCandidateSearchError("combinations must not contain duplicates")
    if set(member_sets) != set(expected_members):
        raise VotingCandidateSearchError(
            "combinations must be the canonical budgeted enumeration"
        )
    expected_ranked = sorted(
        normalized_combinations,
        key=lambda item: _result_sort_key(item, configuration["objective"]),
    )
    if [item["combo_id"] for item in normalized_combinations] != [
        item["combo_id"] for item in expected_ranked
    ]:
        raise VotingCandidateSearchError(
            "combinations must use deterministic objective/tie-break ordering"
        )
    if [item["rank"] for item in normalized_combinations] != list(
        range(1, evaluated + 1)
    ):
        raise VotingCandidateSearchError("combination ranks must be consecutive")
    eligible = _non_negative_int(payload["eligible"], "eligible")
    if eligible != sum(bool(item["eligible"]) for item in normalized_combinations):
        raise VotingCandidateSearchError(
            "eligible must equal eligible combination count"
        )
    search_id = _text(payload["search_id"], "search_id")
    expected_search_id = _search_id(request_hash, configuration)
    if _SEARCH_ID_RE.fullmatch(search_id) is None or not hmac.compare_digest(
        search_id, expected_search_id
    ):
        raise VotingCandidateSearchError(
            "search_id does not match the canonical request/configuration"
        )
    body = {
        "schema_version": VOTING_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION,
        "search_id": search_id,
        "request_hash": request_hash,
        "configuration": configuration,
        "population": population,
        "search_space": search_space,
        "evaluated": evaluated,
        "truncated": truncated,
        "eligible": eligible,
        "combinations": normalized_combinations,
        "producer_version": VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION,
    }
    content_hash = _hash(payload["content_hash"], "content_hash")
    expected_hash = _sha256(_canonical_json(body))
    if not hmac.compare_digest(content_hash, expected_hash):
        raise VotingCandidateSearchError(
            "content_hash does not match canonical Voting search result"
        )
    return {**body, "content_hash": content_hash}


def canonical_voting_candidate_search_result_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole byte-stable JSON representation of a valid result."""

    return _canonical_json(validate_voting_candidate_search_result(payload))


def parse_voting_candidate_search_result_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    """Parse a bounded result JSON while rejecting duplicate keys."""

    return validate_voting_candidate_search_result(
        _parse_json_object(raw, "Voting search result")
    )


def _configuration_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_ids": list(request["candidate_ids"]),
        "member_count": request["member_count"],
        "n": request["n"],
        "objective": dict(request["objective"]),
        "constraints": [dict(item) for item in request["constraints"]],
        "include": list(request["include"]),
        "exclude": list(request["exclude"]),
        "max_combinations": request["max_combinations"],
    }


def _normalize_configuration(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateSearchError("configuration must be an object")
    _exact_fields(value, _CONFIGURATION_FIELDS, "configuration")
    candidate_ids = _candidate_id_array(
        value["candidate_ids"], "configuration.candidate_ids"
    )
    if (
        not candidate_ids
        or len(candidate_ids) > MAX_CANDIDATES
        or candidate_ids != sorted(candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise VotingCandidateSearchError(
            "configuration.candidate_ids must be unique and canonically sorted"
        )
    member_count = _bounded_int(
        value["member_count"],
        "configuration.member_count",
        minimum=2,
        maximum=len(candidate_ids),
    )
    n = _bounded_int(value["n"], "configuration.n", minimum=1, maximum=member_count)
    include = _candidate_id_array(value["include"], "configuration.include")
    exclude = _candidate_id_array(value["exclude"], "configuration.exclude")
    if include != sorted(include) or len(set(include)) != len(include):
        raise VotingCandidateSearchError(
            "configuration.include must be unique and sorted"
        )
    if exclude != sorted(exclude) or len(set(exclude)) != len(exclude):
        raise VotingCandidateSearchError(
            "configuration.exclude must be unique and sorted"
        )
    candidate_set = set(candidate_ids)
    if not set(include).issubset(candidate_set) or not set(exclude).issubset(
        candidate_set
    ):
        raise VotingCandidateSearchError(
            "configuration include/exclude must reference candidate_ids"
        )
    if set(include) & set(exclude):
        raise VotingCandidateSearchError(
            "configuration include/exclude must be disjoint"
        )
    if len(include) > member_count:
        raise VotingCandidateSearchError("configuration.include exceeds member_count")
    if member_count > len(candidate_set - set(exclude)):
        raise VotingCandidateSearchError(
            "configuration.member_count exceeds available candidates"
        )
    objective = _normalize_objective(
        value["objective"],
        weights_available=True,
        amounts_available=True,
    )
    constraints = _normalize_constraints(
        value["constraints"],
        weights_available=True,
        amounts_available=True,
    )
    max_combinations = _bounded_int(
        value["max_combinations"],
        "configuration.max_combinations",
        minimum=1,
        maximum=MAX_COMBINATIONS_BUDGET,
    )
    return {
        "candidate_ids": candidate_ids,
        "member_count": member_count,
        "n": n,
        "objective": objective,
        "constraints": constraints,
        "include": include,
        "exclude": exclude,
        "max_combinations": max_combinations,
    }


def _expected_member_sets(
    configuration: Mapping[str, Any],
) -> tuple[int, list[tuple[str, ...]]]:
    include = list(configuration["include"])
    include_set = set(include)
    excluded = set(configuration["exclude"])
    optional = [
        candidate_id
        for candidate_id in configuration["candidate_ids"]
        if candidate_id not in include_set and candidate_id not in excluded
    ]
    choose = configuration["member_count"] - len(include)
    search_space = math.comb(len(optional), choose)
    selected = islice(
        combinations(optional, choose),
        configuration["max_combinations"],
    )
    return search_space, [tuple(sorted([*include, *members])) for members in selected]


def _validate_configuration_observation_availability(
    configuration: Mapping[str, Any],
    population: Mapping[str, Any],
) -> None:
    objective_metric = configuration["objective"]["metric"]
    if objective_metric in _WEIGHTED_METRICS and not population["weight"]["available"]:
        raise VotingCandidateSearchError(
            "configuration objective requires weight observations"
        )
    if objective_metric in _AMOUNT_METRICS and not population["amount"]["available"]:
        raise VotingCandidateSearchError(
            "configuration objective requires amount observations"
        )
    for constraint in configuration["constraints"]:
        metric = constraint["metric"]
        if metric in _WEIGHTED_METRICS and not population["weight"]["available"]:
            raise VotingCandidateSearchError(
                "configuration constraint requires weight observations"
            )
        if metric in _AMOUNT_METRICS and not population["amount"]["available"]:
            raise VotingCandidateSearchError(
                "configuration constraint requires amount observations"
            )


def _validate_planned_work(
    *,
    candidate_ids: Sequence[str],
    member_count: int,
    include: Sequence[str],
    exclude: Sequence[str],
    max_combinations: int,
    row_count: int,
) -> None:
    include_set = set(include)
    excluded = set(exclude)
    optional_count = sum(
        candidate_id not in include_set and candidate_id not in excluded
        for candidate_id in candidate_ids
    )
    choose = member_count - len(include)
    search_space = math.comb(optional_count, choose)
    planned = min(search_space, max_combinations)
    evaluation_cells = planned * row_count * member_count
    if evaluation_cells > MAX_EVALUATION_CELLS:
        raise VotingCandidateSearchError(
            f"requested search exceeds {MAX_EVALUATION_CELLS} evaluation cells"
        )
    distribution_bins = planned * (member_count + 1)
    if distribution_bins > MAX_RESULT_DISTRIBUTION_BINS:
        raise VotingCandidateSearchError(
            f"requested search exceeds {MAX_RESULT_DISTRIBUTION_BINS} distribution bins"
        )


def _evaluate_combination(
    member_ids: list[str],
    *,
    n: int,
    hit_by_id: Mapping[str, Sequence[bool]],
    target: Sequence[int],
    weights: Sequence[float] | None,
    amounts: Sequence[float] | None,
    population: Mapping[str, Any],
    objective: Mapping[str, str],
    constraints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row_count = len(target)
    member_hits = [
        sum(1 for member_id in member_ids if hit_by_id[member_id][index])
        for index in range(row_count)
    ]
    hit = [count >= n for count in member_hits]
    distribution_counts = [0] * (len(member_ids) + 1)
    for count in member_hits:
        distribution_counts[count] += 1
    distribution = [
        {
            "member_hits": count,
            "row_count": count_rows,
            "row_share": _ratio(count_rows, row_count),
        }
        for count, count_rows in enumerate(distribution_counts)
    ]
    metrics = _combination_metrics(
        hit,
        target=target,
        weights=weights,
        amounts=amounts,
        population=population,
    )
    failures = _constraint_failures(metrics, constraints)
    return {
        "combo_id": _combo_id(member_ids, n),
        "member_ids": member_ids,
        "n": n,
        "rank": 0,
        "objective_value": float(metrics[objective["metric"]]),
        "eligible": not failures,
        "constraint_failures": failures,
        "hit_count_distribution": distribution,
        "metrics": metrics,
    }


def _combination_metrics(
    hit: Sequence[bool],
    *,
    target: Sequence[int],
    weights: Sequence[float] | None,
    amounts: Sequence[float] | None,
    population: Mapping[str, Any],
) -> dict[str, Any]:
    row_count = len(target)
    hit_count = sum(hit)
    bad_count = sum(
        1
        for selected, label in zip(hit, target, strict=True)
        if selected and label == 1
    )
    good_count = hit_count - bad_count
    base_bad_rate = float(population["bad_rate"])
    bad_rate = _ratio(bad_count, hit_count)
    metrics: dict[str, Any] = {
        "population_count": row_count,
        "hit_count": hit_count,
        "hit_share": _ratio(hit_count, row_count),
        "good_count": good_count,
        "bad_count": bad_count,
        "bad_rate": bad_rate,
        "base_bad_rate": base_bad_rate,
        "lift": _ratio(bad_rate, base_bad_rate),
        "bad_capture_rate": _ratio(bad_count, population["bad_count"]),
    }
    metrics.update(
        _observation_metrics(
            hit,
            target=target,
            values=weights,
            population=population["weight"],
            names=(
                "weighted_hit_total",
                "weighted_hit_share",
                "weighted_good_total",
                "weighted_bad_total",
                "weighted_bad_rate",
                "weighted_bad_capture_rate",
            ),
        )
    )
    metrics.update(
        _observation_metrics(
            hit,
            target=target,
            values=amounts,
            population=population["amount"],
            names=(
                "hit_amount",
                "hit_amount_share",
                "good_amount",
                "bad_amount",
                "bad_amount_rate",
                "bad_amount_capture_rate",
            ),
        )
    )
    return metrics


def _observation_metrics(
    hit: Sequence[bool],
    *,
    target: Sequence[int],
    values: Sequence[float] | None,
    population: Mapping[str, Any],
    names: tuple[str, str, str, str, str, str],
) -> dict[str, float | None]:
    if values is None:
        return dict.fromkeys(names)
    hit_total = _finite_sum(
        value for selected, value in zip(hit, values, strict=True) if selected
    )
    bad_total = _finite_sum(
        value
        for selected, label, value in zip(hit, target, values, strict=True)
        if selected and label == 1
    )
    good_total = hit_total - bad_total
    return {
        names[0]: hit_total,
        names[1]: _ratio(hit_total, population["total"]),
        names[2]: good_total,
        names[3]: bad_total,
        names[4]: _ratio(bad_total, hit_total),
        names[5]: _ratio(bad_total, population["bad_total"]),
    }


def _population_summary(
    target: Sequence[int],
    *,
    weights: Sequence[float] | None,
    amounts: Sequence[float] | None,
) -> dict[str, Any]:
    row_count = len(target)
    bad_count = sum(target)
    return {
        "row_count": row_count,
        "good_count": row_count - bad_count,
        "bad_count": bad_count,
        "bad_rate": _ratio(bad_count, row_count),
        "weight": _observation_summary(target, weights),
        "amount": _observation_summary(target, amounts),
    }


def _observation_summary(
    target: Sequence[int],
    values: Sequence[float] | None,
) -> dict[str, Any]:
    if values is None:
        return {
            "available": False,
            "total": None,
            "good_total": None,
            "bad_total": None,
        }
    total = _finite_sum(values)
    bad_total = _finite_sum(
        value for label, value in zip(target, values, strict=True) if label == 1
    )
    return {
        "available": True,
        "total": total,
        "good_total": total - bad_total,
        "bad_total": bad_total,
    }


def _normalize_population(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateSearchError("population must be an object")
    _exact_fields(value, _POPULATION_FIELDS, "population")
    row_count = _bounded_int(
        value["row_count"], "population.row_count", minimum=1, maximum=MAX_ROWS
    )
    good_count = _bounded_int(
        value["good_count"],
        "population.good_count",
        minimum=0,
        maximum=row_count,
    )
    bad_count = _bounded_int(
        value["bad_count"],
        "population.bad_count",
        minimum=0,
        maximum=row_count,
    )
    if good_count + bad_count != row_count:
        raise VotingCandidateSearchError(
            "population good_count + bad_count must equal row_count"
        )
    bad_rate = _derived_number(
        value["bad_rate"],
        _ratio(bad_count, row_count),
        "population.bad_rate",
    )
    return {
        "row_count": row_count,
        "good_count": good_count,
        "bad_count": bad_count,
        "bad_rate": bad_rate,
        "weight": _normalize_observation_summary(value["weight"], "weight"),
        "amount": _normalize_observation_summary(value["amount"], "amount"),
    }


def _normalize_observation_summary(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateSearchError(f"population.{name} must be an object")
    _exact_fields(value, _OBSERVATION_SUMMARY_FIELDS, f"population.{name}")
    available = _strict_bool(value["available"], f"population.{name}.available")
    if not available:
        if any(
            value[field] is not None for field in ("total", "good_total", "bad_total")
        ):
            raise VotingCandidateSearchError(
                f"population.{name} totals must be null when unavailable"
            )
        return {
            "available": False,
            "total": None,
            "good_total": None,
            "bad_total": None,
        }
    total = _non_negative_number(value["total"], f"population.{name}.total")
    good = _non_negative_number(value["good_total"], f"population.{name}.good_total")
    bad = _non_negative_number(value["bad_total"], f"population.{name}.bad_total")
    _require_close(
        good + bad,
        total,
        f"population.{name} good_total + bad_total must equal total",
    )
    return {
        "available": True,
        "total": total,
        "good_total": good,
        "bad_total": bad,
    }


def _normalize_result_combination(
    value: object,
    *,
    configuration: Mapping[str, Any],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateSearchError("combination must be an object")
    _exact_fields(value, _COMBINATION_FIELDS, "combination")
    member_ids = _candidate_id_array(value["member_ids"], "combination.member_ids")
    if (
        member_ids != sorted(member_ids)
        or len(set(member_ids)) != len(member_ids)
        or len(member_ids) != configuration["member_count"]
    ):
        raise VotingCandidateSearchError(
            "combination.member_ids must be unique, sorted, and match member_count"
        )
    if not set(configuration["include"]).issubset(member_ids):
        raise VotingCandidateSearchError(
            "combination.member_ids must contain every included candidate"
        )
    if set(configuration["exclude"]) & set(member_ids):
        raise VotingCandidateSearchError(
            "combination.member_ids must not contain excluded candidates"
        )
    if not set(member_ids).issubset(configuration["candidate_ids"]):
        raise VotingCandidateSearchError(
            "combination.member_ids must reference candidate_ids"
        )
    n = _bounded_int(value["n"], "combination.n", minimum=1, maximum=len(member_ids))
    if n != configuration["n"]:
        raise VotingCandidateSearchError("combination.n must match configuration.n")
    combo_id = _text(value["combo_id"], "combination.combo_id")
    expected_combo_id = _combo_id(member_ids, n)
    if _COMBO_ID_RE.fullmatch(combo_id) is None or not hmac.compare_digest(
        combo_id, expected_combo_id
    ):
        raise VotingCandidateSearchError(
            "combo_id does not match canonical member_ids and n"
        )
    rank = _positive_int(value["rank"], "combination.rank")
    distribution_value = _array(
        value["hit_count_distribution"], "combination.hit_count_distribution"
    )
    if len(distribution_value) != len(member_ids) + 1:
        raise VotingCandidateSearchError(
            "hit_count_distribution must cover 0..member_count"
        )
    distribution = [
        _normalize_distribution_bin(
            item,
            expected_hits=index,
            population_count=population["row_count"],
        )
        for index, item in enumerate(distribution_value)
    ]
    if sum(item["row_count"] for item in distribution) != population["row_count"]:
        raise VotingCandidateSearchError(
            "hit_count_distribution row_count must conserve population"
        )
    metrics = _normalize_metrics(
        value["metrics"],
        population=population,
        distribution=distribution,
        n=n,
    )
    objective_value = _derived_number(
        value["objective_value"],
        float(metrics[configuration["objective"]["metric"]]),
        "combination.objective_value",
    )
    failures = _normalize_constraint_failures(
        value["constraint_failures"],
        metrics=metrics,
        constraints=configuration["constraints"],
    )
    eligible = _strict_bool(value["eligible"], "combination.eligible")
    if eligible != (not failures):
        raise VotingCandidateSearchError(
            "combination.eligible must match constraint failures"
        )
    return {
        "combo_id": combo_id,
        "member_ids": member_ids,
        "n": n,
        "rank": rank,
        "objective_value": objective_value,
        "eligible": eligible,
        "constraint_failures": failures,
        "hit_count_distribution": distribution,
        "metrics": metrics,
    }


def _normalize_distribution_bin(
    value: object,
    *,
    expected_hits: int,
    population_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateSearchError("distribution bin must be an object")
    _exact_fields(value, _DISTRIBUTION_FIELDS, "distribution bin")
    member_hits = _non_negative_int(value["member_hits"], "distribution.member_hits")
    if member_hits != expected_hits:
        raise VotingCandidateSearchError(
            "distribution.member_hits must be consecutive from zero"
        )
    row_count = _bounded_int(
        value["row_count"],
        "distribution.row_count",
        minimum=0,
        maximum=population_count,
    )
    row_share = _derived_number(
        value["row_share"],
        _ratio(row_count, population_count),
        "distribution.row_share",
    )
    return {
        "member_hits": member_hits,
        "row_count": row_count,
        "row_share": row_share,
    }


def _normalize_metrics(
    value: object,
    *,
    population: Mapping[str, Any],
    distribution: Sequence[Mapping[str, Any]],
    n: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateSearchError("combination.metrics must be an object")
    _exact_fields(value, _METRIC_FIELDS, "combination.metrics")
    population_count = _positive_int(
        value["population_count"], "metrics.population_count"
    )
    if population_count != population["row_count"]:
        raise VotingCandidateSearchError(
            "metrics.population_count must match population.row_count"
        )
    hit_count = sum(
        item["row_count"] for item in distribution if item["member_hits"] >= n
    )
    supplied_hit = _bounded_int(
        value["hit_count"],
        "metrics.hit_count",
        minimum=0,
        maximum=population_count,
    )
    if supplied_hit != hit_count:
        raise VotingCandidateSearchError(
            "metrics.hit_count must reconcile to hit_count_distribution"
        )
    good_count = _bounded_int(
        value["good_count"],
        "metrics.good_count",
        minimum=0,
        maximum=hit_count,
    )
    bad_count = _bounded_int(
        value["bad_count"],
        "metrics.bad_count",
        minimum=0,
        maximum=hit_count,
    )
    if good_count + bad_count != hit_count:
        raise VotingCandidateSearchError(
            "metrics.good_count + bad_count must equal hit_count"
        )
    if good_count > population["good_count"]:
        raise VotingCandidateSearchError(
            "metrics.good_count must not exceed population.good_count"
        )
    if bad_count > population["bad_count"]:
        raise VotingCandidateSearchError(
            "metrics.bad_count must not exceed population.bad_count"
        )
    base_bad_rate = _derived_number(
        value["base_bad_rate"],
        population["bad_rate"],
        "metrics.base_bad_rate",
    )
    bad_rate = _derived_number(
        value["bad_rate"], _ratio(bad_count, hit_count), "metrics.bad_rate"
    )
    normalized: dict[str, Any] = {
        "population_count": population_count,
        "hit_count": hit_count,
        "hit_share": _derived_number(
            value["hit_share"],
            _ratio(hit_count, population_count),
            "metrics.hit_share",
        ),
        "good_count": good_count,
        "bad_count": bad_count,
        "bad_rate": bad_rate,
        "base_bad_rate": base_bad_rate,
        "lift": _derived_number(
            value["lift"], _ratio(bad_rate, base_bad_rate), "metrics.lift"
        ),
        "bad_capture_rate": _derived_number(
            value["bad_capture_rate"],
            _ratio(bad_count, population["bad_count"]),
            "metrics.bad_capture_rate",
        ),
    }
    normalized.update(
        _normalize_observation_metrics(
            value,
            population=population["weight"],
            names=(
                "weighted_hit_total",
                "weighted_hit_share",
                "weighted_good_total",
                "weighted_bad_total",
                "weighted_bad_rate",
                "weighted_bad_capture_rate",
            ),
        )
    )
    normalized.update(
        _normalize_observation_metrics(
            value,
            population=population["amount"],
            names=(
                "hit_amount",
                "hit_amount_share",
                "good_amount",
                "bad_amount",
                "bad_amount_rate",
                "bad_amount_capture_rate",
            ),
        )
    )
    return normalized


def _normalize_observation_metrics(
    value: Mapping[str, Any],
    *,
    population: Mapping[str, Any],
    names: tuple[str, str, str, str, str, str],
) -> dict[str, float | None]:
    supplied = [value[name] for name in names]
    if not population["available"]:
        if any(item is not None for item in supplied):
            raise VotingCandidateSearchError(
                f"{names[0]} metrics must be null when observation is unavailable"
            )
        return dict.fromkeys(names)
    hit_total = _non_negative_number(supplied[0], f"metrics.{names[0]}")
    good_total = _non_negative_number(supplied[2], f"metrics.{names[2]}")
    bad_total = _non_negative_number(supplied[3], f"metrics.{names[3]}")
    _require_close(
        good_total + bad_total,
        hit_total,
        f"metrics {names[2]} + {names[3]} must equal {names[0]}",
    )
    if hit_total > population["total"] and not math.isclose(
        hit_total, population["total"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise VotingCandidateSearchError(
            f"metrics.{names[0]} must not exceed population total"
        )
    if good_total > population["good_total"] and not math.isclose(
        good_total,
        population["good_total"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise VotingCandidateSearchError(
            f"metrics.{names[2]} must not exceed population good total"
        )
    if bad_total > population["bad_total"] and not math.isclose(
        bad_total, population["bad_total"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise VotingCandidateSearchError(
            f"metrics.{names[3]} must not exceed population bad total"
        )
    return {
        names[0]: hit_total,
        names[1]: _derived_number(
            supplied[1], _ratio(hit_total, population["total"]), f"metrics.{names[1]}"
        ),
        names[2]: good_total,
        names[3]: bad_total,
        names[4]: _derived_number(
            supplied[4], _ratio(bad_total, hit_total), f"metrics.{names[4]}"
        ),
        names[5]: _derived_number(
            supplied[5],
            _ratio(bad_total, population["bad_total"]),
            f"metrics.{names[5]}",
        ),
    }


def _normalize_constraint_failures(
    value: object,
    *,
    metrics: Mapping[str, Any],
    constraints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    supplied = _array(value, "constraint_failures")
    expected = _constraint_failures(metrics, constraints)
    normalized: list[dict[str, Any]] = []
    for item in supplied:
        if not isinstance(item, Mapping):
            raise VotingCandidateSearchError("constraint failure must be an object")
        _exact_fields(item, _FAILURE_FIELDS, "constraint failure")
        normalized.append(
            {
                "metric": _text(item["metric"], "failure.metric"),
                "operator": _text(item["operator"], "failure.operator"),
                "threshold": _finite_number(item["threshold"], "failure.threshold"),
                "actual": _finite_number(item["actual"], "failure.actual"),
            }
        )
    if normalized != expected:
        raise VotingCandidateSearchError(
            "constraint_failures do not match deterministic metric evaluation"
        )
    return normalized


def _constraint_failures(
    metrics: Mapping[str, Any],
    constraints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for constraint in constraints:
        actual = float(metrics[constraint["metric"]])
        threshold = float(constraint["value"])
        passed = (
            actual >= threshold
            if constraint["operator"] == "gte"
            else actual <= threshold
        )
        if not passed:
            failures.append(
                {
                    "metric": constraint["metric"],
                    "operator": constraint["operator"],
                    "threshold": threshold,
                    "actual": actual,
                }
            )
    return failures


def _result_sort_key(
    item: Mapping[str, Any],
    objective: Mapping[str, str],
) -> tuple[Any, ...]:
    objective_value = float(item["objective_value"])
    ordered_value = (
        -objective_value if objective["direction"] == "maximize" else objective_value
    )
    return (
        0 if item["eligible"] else 1,
        ordered_value,
        tuple(item["member_ids"]),
        item["combo_id"],
    )


def _normalize_objective(
    value: object,
    *,
    weights_available: bool,
    amounts_available: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise VotingCandidateSearchError("objective must be an object")
    _exact_fields(value, _OBJECTIVE_FIELDS, "objective")
    metric = _metric(
        value["metric"],
        "objective.metric",
        weights_available=weights_available,
        amounts_available=amounts_available,
    )
    direction = _text(value["direction"], "objective.direction")
    if direction not in {"maximize", "minimize"}:
        raise VotingCandidateSearchError(
            "objective.direction must be maximize or minimize"
        )
    return {"metric": metric, "direction": direction}


def _normalize_constraints(
    value: object,
    *,
    weights_available: bool,
    amounts_available: bool,
) -> list[dict[str, Any]]:
    items = _array(value, "constraints")
    if len(items) > MAX_CONSTRAINTS:
        raise VotingCandidateSearchError(
            f"constraints must contain at most {MAX_CONSTRAINTS} items"
        )
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise VotingCandidateSearchError(f"constraints[{index}] must be an object")
        _exact_fields(item, _CONSTRAINT_FIELDS, f"constraints[{index}]")
        metric = _metric(
            item["metric"],
            f"constraints[{index}].metric",
            weights_available=weights_available,
            amounts_available=amounts_available,
        )
        operator = _text(item["operator"], f"constraints[{index}].operator")
        if operator not in {"gte", "lte"}:
            raise VotingCandidateSearchError(
                f"constraints[{index}].operator must be gte or lte"
            )
        constraint_value = _non_negative_number(
            item["value"], f"constraints[{index}].value"
        )
        if metric in _RATE_METRICS and constraint_value > 1.0:
            raise VotingCandidateSearchError(
                f"constraints[{index}].value must be in [0, 1] for {metric}"
            )
        identity = (metric, operator)
        if identity in identities:
            raise VotingCandidateSearchError(
                "constraints must not repeat the same metric/operator"
            )
        identities.add(identity)
        normalized.append(
            {"metric": metric, "operator": operator, "value": constraint_value}
        )
    return sorted(
        normalized,
        key=lambda item: (item["metric"], item["operator"], item["value"]),
    )


def _metric(
    value: object,
    name: str,
    *,
    weights_available: bool,
    amounts_available: bool,
) -> str:
    metric = _text(value, name)
    if metric not in _OBJECTIVE_METRICS:
        raise VotingCandidateSearchError(f"{name} is unsupported")
    if metric in _WEIGHTED_METRICS and not weights_available:
        raise VotingCandidateSearchError(f"{name} requires weights")
    if metric in _AMOUNT_METRICS and not amounts_available:
        raise VotingCandidateSearchError(f"{name} requires amounts")
    return metric


def _candidate_id_array(value: object, name: str) -> list[str]:
    return [
        _candidate_id(item, f"{name}[{index}]")
        for index, item in enumerate(_array(value, name))
    ]


def _candidate_id(value: object, name: str) -> str:
    normalized = _text(value, name)
    if len(normalized.encode("utf-8")) > 256:
        raise VotingCandidateSearchError(f"{name} exceeds 256 UTF-8 bytes")
    return normalized


def _hit_matrix(
    value: object,
    *,
    candidate_count: int,
    row_count: int,
) -> list[list[bool]]:
    rows = _array(value, "hit_matrix")
    if len(rows) != candidate_count:
        raise VotingCandidateSearchError(
            "hit_matrix row count must equal candidate_ids length"
        )
    normalized: list[list[bool]] = []
    for candidate_index, candidate_hits in enumerate(rows):
        hits = _array(candidate_hits, f"hit_matrix[{candidate_index}]")
        if len(hits) != row_count:
            raise VotingCandidateSearchError(
                f"hit_matrix[{candidate_index}] length must equal target"
            )
        normalized.append(
            [
                _strict_bool(hit, f"hit_matrix[{candidate_index}][{row_index}]")
                for row_index, hit in enumerate(hits)
            ]
        )
    return normalized


def _binary_array(value: object, name: str) -> list[int]:
    output: list[int] = []
    for index, item in enumerate(_array(value, name)):
        if (
            isinstance(item, bool)
            or not isinstance(item, Integral)
            or int(item) not in {0, 1}
        ):
            raise VotingCandidateSearchError(f"{name}[{index}] must be integer 0 or 1")
        output.append(int(item))
    return output


def _optional_observation_array(
    value: object,
    name: str,
    *,
    row_count: int,
) -> list[float] | None:
    if value is None:
        return None
    items = _array(value, name)
    if len(items) != row_count:
        raise VotingCandidateSearchError(f"{name} length must equal target")
    return [
        _non_negative_number(item, f"{name}[{index}]")
        for index, item in enumerate(items)
    ]


def _combo_id(member_ids: Sequence[str], n: int) -> str:
    return (
        "voting-combo-"
        + _sha256(_canonical_json({"member_ids": list(member_ids), "n": n}))[:32]
    )


def _search_id(
    request_hash: str,
    configuration: Mapping[str, Any],
) -> str:
    return (
        "voting-search-"
        + _sha256(
            _canonical_json(
                {"request_hash": request_hash, "configuration": configuration}
            )
        )[:32]
    )


def _parse_json_object(
    raw: str | bytes | bytearray,
    name: str,
) -> dict[str, Any]:
    if not isinstance(raw, str | bytes | bytearray):
        raise VotingCandidateSearchError(f"{name} JSON must be text or bytes")
    if (
        len(raw if isinstance(raw, bytes | bytearray) else raw.encode("utf-8"))
        > MAX_JSON_BYTES
    ):
        raise VotingCandidateSearchError(f"{name} JSON exceeds {MAX_JSON_BYTES} bytes")
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except VotingCandidateSearchError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise VotingCandidateSearchError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VotingCandidateSearchError(f"{name} JSON must contain an object")
    return payload


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VotingCandidateSearchError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _array(value: object, name: str) -> list[Any]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise VotingCandidateSearchError(f"{name} must be an array")
    return list(value)


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise VotingCandidateSearchError(f"{name} must be boolean")
    return value


def _bounded_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise VotingCandidateSearchError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise VotingCandidateSearchError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return normalized


def _positive_int(value: object, name: str) -> int:
    return _bounded_int(value, name, minimum=1, maximum=2**63 - 1)


def _non_negative_int(value: object, name: str) -> int:
    return _bounded_int(value, name, minimum=0, maximum=2**63 - 1)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VotingCandidateSearchError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise VotingCandidateSearchError(f"{name} must be a finite number")
    return normalized


def _non_negative_number(value: object, name: str) -> float:
    normalized = _finite_number(value, name)
    if normalized < 0.0:
        raise VotingCandidateSearchError(f"{name} must be non-negative")
    return normalized


def _derived_number(value: object, expected: float, name: str) -> float:
    actual = _finite_number(value, name)
    _require_close(actual, expected, f"{name} is inconsistent")
    return float(expected)


def _require_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise VotingCandidateSearchError(message)


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    result = float(numerator) / float(denominator)
    if not math.isfinite(result):
        raise VotingCandidateSearchError("ratio calculation overflowed")
    return result


def _finite_sum(values: Sequence[float] | Any) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise VotingCandidateSearchError("observation sum overflowed") from exc
    if not math.isfinite(result):
        raise VotingCandidateSearchError("observation sum overflowed")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VotingCandidateSearchError(f"{name} must be non-empty canonical text")
    if "\x00" in value:
        raise VotingCandidateSearchError(f"{name} must not contain NUL")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise VotingCandidateSearchError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise VotingCandidateSearchError(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        raise VotingCandidateSearchError(f"invalid {name} ({'; '.join(details)})")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise VotingCandidateSearchError(
            "Voting search must contain finite canonical JSON"
        ) from exc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "MAX_CANDIDATES",
    "MAX_COMBINATIONS_BUDGET",
    "MAX_CONSTRAINTS",
    "MAX_EVALUATION_CELLS",
    "MAX_JSON_BYTES",
    "MAX_MATRIX_CELLS",
    "MAX_RESULT_DISTRIBUTION_BINS",
    "MAX_ROWS",
    "MAX_SEARCH_SPACE",
    "VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION",
    "VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION",
    "VOTING_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION",
    "VotingCandidateSearchError",
    "canonical_voting_candidate_search_request_json",
    "canonical_voting_candidate_search_result_json",
    "parse_voting_candidate_search_request_json",
    "parse_voting_candidate_search_result_json",
    "search_voting_candidate_combinations",
    "validate_voting_candidate_search_request",
    "validate_voting_candidate_search_result",
]
