"""Deterministic, bounded 2D/3D threshold-rule search over aggregate facts.

The governed Tool boundary owns dataset access and evaluates every canonical
condition against one authenticated risk/development sample.  This pure kernel
accepts only aggregate trial counts, derives all metrics and constraint
outcomes, and persists no row-level values.  Search results are evidence for a
later explicit selection; they never name a winner or mutate a Strategy Pool.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
from itertools import combinations, product
import json
import math
from numbers import Integral, Real
import re
from typing import Any

from marvis.packs.strategy.errors import StrategyError


CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION = (
    "strategy.cross-rule-search-request.v1"
)
CROSS_RULE_SEARCH_RESULT_SCHEMA_VERSION = (
    "strategy.cross-rule-search-result.v1"
)
CROSS_RULE_SEARCH_PRODUCER_VERSION = "strategy.cross-rule-search/1"

MIN_FEATURES = 2
MAX_FEATURES = 12
MAX_THRESHOLDS_PER_FEATURE = 8
MAX_TRIALS = 5_000
MAX_ROW_EVALUATIONS = 50_000_000
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
ENUMERATION_POLICY = (
    "canonical_feature_combo_round_robin_threshold_prefix.v1"
)

_METHODS = frozenset(
    {
        "equal_frequency",
        "equal_width",
        "chimerge",
        "tree",
        "manual",
    }
)
_RISK_DIRECTIONS = frozenset(
    {"increasing", "decreasing", "non_monotonic", "flat"}
)
_OPERATORS = frozenset({"gte", "lt"})
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "population",
        "dimension",
        "features",
        "constraints",
        "trials",
        "max_trials",
    }
)
_SOURCE_FIELDS = frozenset(
    {"candidate_id", "evidence_hash", "sample_context_hash"}
)
_POPULATION_FIELDS = frozenset(
    {
        "row_count",
        "good",
        "bad",
        "loan_amount_sum",
        "overdue_amount_sum",
    }
)
_FEATURE_FIELDS = frozenset(
    {
        "feature",
        "method",
        "risk_direction",
        "thresholds",
        "excluded_values",
        "missing_count",
        "missing_bad",
    }
)
_CONSTRAINT_FIELDS = frozenset(
    {
        "min_lift",
        "min_bad_count",
        "max_hit_share",
        "min_amount_lift",
    }
)
_CONDITION_FIELDS = frozenset(
    {"feature", "method", "operator", "threshold", "include_missing"}
)
_TRIAL_FIELDS = frozenset(
    {
        "conditions",
        "count",
        "good",
        "bad",
        "loan_amount_sum",
        "overdue_amount_sum",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "search_id",
        "request_hash",
        "source",
        "population",
        "configuration",
        "search_space",
        "evaluated",
        "truncated",
        "eligible",
        "rules",
        "trial_accounting",
        "lifecycle",
        "producer_version",
        "content_hash",
    }
)
_CONFIGURATION_FIELDS = frozenset(
    {
        "dimension",
        "features",
        "constraints",
        "max_trials",
        "enumeration_policy",
    }
)
_RULE_FIELDS = frozenset(
    {
        "rule_id",
        "conditions",
        "metrics",
        "constraint_failures",
        "eligible",
        "rank",
    }
)
_METRIC_FIELDS = frozenset(
    {
        "count",
        "good",
        "bad",
        "hit_share",
        "bad_rate",
        "lift",
        "bad_capture_rate",
        "loan_amount_sum",
        "overdue_amount_sum",
        "amount_overdue_rate",
        "amount_lift",
    }
)
_ACCOUNTING_FIELDS = frozenset({"limits", "used"})
_ACCOUNTING_LIMIT_FIELDS = frozenset(
    {
        "max_features",
        "max_thresholds_per_feature",
        "max_trials",
        "max_row_evaluations",
        "max_artifact_bytes",
    }
)
_ACCOUNTING_USED_FIELDS = frozenset(
    {"features", "thresholds", "trials", "row_evaluations"}
)
_LIFECYCLE = {
    "selected": False,
    "admitted": False,
    "applied": False,
    "adopted": False,
    "deployed": False,
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_SEARCH_ID_RE = re.compile(r"^cross-rule-search-[0-9a-f]{32}$")
_RULE_ID_RE = re.compile(r"^cross-rule-[0-9a-f]{32}$")


class CrossRuleSearchError(StrategyError):
    """The bounded Cross threshold-rule contract was violated."""


def canonical_cross_rule_trial_prefix(
    features: Sequence[Mapping[str, Any]],
    *,
    dimension: int,
    max_trials: int,
) -> list[list[dict[str, Any]]]:
    """Return the fair canonical trial prefix without evaluating any rows."""

    normalized_dimension = _dimension(dimension)
    normalized_max = _integer(
        max_trials,
        "max_trials",
        minimum=1,
        maximum=MAX_TRIALS,
    )
    feature_rows = sorted(
        (
            _feature(item, index=index, population_count=None)
            for index, item in enumerate(_array(features, "features"))
        ),
        key=lambda item: item["feature"],
    )
    if not MIN_FEATURES <= len(feature_rows) <= MAX_FEATURES:
        raise CrossRuleSearchError(
            "features must contain 2..12 explicit numeric features"
        )
    if len(feature_rows) < normalized_dimension:
        raise CrossRuleSearchError(
            "features must contain at least dimension values"
        )
    names = [item["feature"] for item in feature_rows]
    if len(set(names)) != len(names):
        raise CrossRuleSearchError("features must be unique")

    candidates = {
        item["feature"]: _condition_candidates(item) for item in feature_rows
    }
    iterators = [
        iter(product(*(candidates[name] for name in combo)))
        for combo in combinations(names, normalized_dimension)
    ]
    prefix: list[list[dict[str, Any]]] = []
    active = list(iterators)
    while active and len(prefix) < normalized_max:
        next_active = []
        for iterator in active:
            try:
                conditions = next(iterator)
            except StopIteration:
                continue
            prefix.append([dict(item) for item in conditions])
            next_active.append(iterator)
            if len(prefix) >= normalized_max:
                break
        active = next_active
    return prefix


def validate_cross_rule_search_request(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate aggregate trial facts against the exact canonical prefix."""

    obj = _object(payload, "Cross rule search request")
    _exact(obj, _REQUEST_FIELDS, "Cross rule search request")
    if obj["schema_version"] != CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION:
        raise CrossRuleSearchError(
            "schema_version must be "
            + CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION
        )
    source = _source(obj["source"])
    population = _population(obj["population"])
    dimension = _dimension(obj["dimension"])
    raw_features = _array(obj["features"], "features")
    if not MIN_FEATURES <= len(raw_features) <= MAX_FEATURES:
        raise CrossRuleSearchError(
            "features must contain 2..12 explicit numeric features"
        )
    if len(raw_features) < dimension:
        raise CrossRuleSearchError(
            "features must contain at least dimension values"
        )
    features = sorted(
        (
            _feature(
                item,
                index=index,
                population_count=population["row_count"],
            )
            for index, item in enumerate(raw_features)
        ),
        key=lambda item: item["feature"],
    )
    feature_names = [item["feature"] for item in features]
    if len(set(feature_names)) != len(feature_names):
        raise CrossRuleSearchError("features must be unique")
    constraints = _constraints(obj["constraints"])
    if (
        constraints["min_amount_lift"] is not None
        and population["loan_amount_sum"] is None
    ):
        raise CrossRuleSearchError(
            "min_amount_lift requires population amount evidence"
        )
    max_trials = _integer(
        obj["max_trials"],
        "max_trials",
        minimum=1,
        maximum=MAX_TRIALS,
    )
    prefix = canonical_cross_rule_trial_prefix(
        features,
        dimension=dimension,
        max_trials=max_trials,
    )
    expected = {_canonical_json(item): item for item in prefix}
    trials_by_key: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(_array(obj["trials"], "trials")):
        trial = _trial(
            item,
            index=index,
            feature_by_name={row["feature"]: row for row in features},
            dimension=dimension,
            population=population,
        )
        key = _canonical_json(trial["conditions"])
        if key in trials_by_key:
            raise CrossRuleSearchError("trials contains a duplicate trial")
        trials_by_key[key] = trial
    if set(trials_by_key) != set(expected):
        raise CrossRuleSearchError(
            "trials must exactly match the canonical trial prefix"
        )
    trials = [trials_by_key[_canonical_json(item)] for item in prefix]
    row_evaluations = population["row_count"] * len(trials)
    if row_evaluations > MAX_ROW_EVALUATIONS:
        raise CrossRuleSearchError(
            "Cross rule search row evaluations exceed hard budget"
        )
    normalized = {
        "schema_version": CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION,
        "source": source,
        "population": population,
        "dimension": dimension,
        "features": features,
        "constraints": constraints,
        "trials": trials,
        "max_trials": max_trials,
    }
    if len(_canonical_json(normalized).encode("utf-8")) > MAX_ARTIFACT_BYTES:
        raise CrossRuleSearchError("Cross rule search request exceeds byte budget")
    return normalized


def search_cross_threshold_rules(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Rank one bounded 2D/3D threshold prefix without selecting a rule."""

    normalized = validate_cross_rule_search_request(request)
    request_hash = _sha256(_canonical_json(normalized))
    search_id = _stable_id(
        "cross-rule-search",
        {"request_hash": request_hash},
    )
    constraints = normalized["constraints"]
    rules = []
    for trial in normalized["trials"]:
        metrics = _derived_metrics(
            population=normalized["population"],
            trial=trial,
        )
        failures = _constraint_failures(metrics, constraints)
        conditions = trial["conditions"]
        rules.append(
            {
                "rule_id": _stable_id(
                    "cross-rule",
                    {
                        "source": normalized["source"],
                        "conditions": conditions,
                    },
                ),
                "conditions": conditions,
                "metrics": metrics,
                "constraint_failures": failures,
                "eligible": not failures,
                "rank": 0,
            }
        )
    rules.sort(key=_rank_key)
    ranked = [
        {**item, "rank": index}
        for index, item in enumerate(rules, start=1)
    ]
    search_space = _search_space(
        normalized["features"],
        dimension=normalized["dimension"],
    )
    evaluated = len(ranked)
    body = {
        "schema_version": CROSS_RULE_SEARCH_RESULT_SCHEMA_VERSION,
        "search_id": search_id,
        "request_hash": request_hash,
        "source": normalized["source"],
        "population": normalized["population"],
        "configuration": {
            "dimension": normalized["dimension"],
            "features": normalized["features"],
            "constraints": constraints,
            "max_trials": normalized["max_trials"],
            "enumeration_policy": ENUMERATION_POLICY,
        },
        "search_space": search_space,
        "evaluated": evaluated,
        "truncated": evaluated < search_space,
        "eligible": sum(1 for item in ranked if item["eligible"]),
        "rules": ranked,
        "trial_accounting": {
            "limits": {
                "max_features": MAX_FEATURES,
                "max_thresholds_per_feature": MAX_THRESHOLDS_PER_FEATURE,
                "max_trials": MAX_TRIALS,
                "max_row_evaluations": MAX_ROW_EVALUATIONS,
                "max_artifact_bytes": MAX_ARTIFACT_BYTES,
            },
            "used": {
                "features": len(normalized["features"]),
                "thresholds": sum(
                    len(item["thresholds"])
                    for item in normalized["features"]
                ),
                "trials": evaluated,
                "row_evaluations": (
                    normalized["population"]["row_count"] * evaluated
                ),
            },
        },
        "lifecycle": dict(_LIFECYCLE),
        "producer_version": CROSS_RULE_SEARCH_PRODUCER_VERSION,
    }
    result = {
        **body,
        "content_hash": _sha256(_canonical_json(body)),
    }
    return validate_cross_rule_search_result(result)


def validate_cross_rule_search_result(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a detached search result and all deterministic derivations."""

    obj = _object(payload, "Cross rule search result")
    _exact(obj, _RESULT_FIELDS, "Cross rule search result")
    if obj["schema_version"] != CROSS_RULE_SEARCH_RESULT_SCHEMA_VERSION:
        raise CrossRuleSearchError(
            "Cross rule search result schema_version is invalid"
        )
    search_id = _id(
        obj["search_id"],
        "search_id",
        _SEARCH_ID_RE,
    )
    request_hash = _hash(obj["request_hash"], "request_hash")
    source = _source(obj["source"])
    population = _population(obj["population"])
    configuration = _configuration(
        obj["configuration"],
        population_count=population["row_count"],
    )
    search_space = _integer(
        obj["search_space"],
        "search_space",
        minimum=1,
    )
    evaluated = _integer(
        obj["evaluated"],
        "evaluated",
        minimum=1,
        maximum=MAX_TRIALS,
    )
    if evaluated > search_space:
        raise CrossRuleSearchError("evaluated cannot exceed search_space")
    truncated = _boolean(obj["truncated"], "truncated")
    if truncated is not (evaluated < search_space):
        raise CrossRuleSearchError("truncated does not match search accounting")
    rules_raw = _array(obj["rules"], "rules")
    if len(rules_raw) != evaluated:
        raise CrossRuleSearchError("rules length must equal evaluated")
    rules = [
        _result_rule(
            item,
            index=index,
            source=source,
            population=population,
            configuration=configuration,
        )
        for index, item in enumerate(rules_raw)
    ]
    if rules != sorted(rules, key=_rank_key):
        raise CrossRuleSearchError("rules are not in canonical rank order")
    eligible = _integer(
        obj["eligible"],
        "eligible",
        minimum=0,
        maximum=evaluated,
    )
    if eligible != sum(1 for item in rules if item["eligible"]):
        raise CrossRuleSearchError("eligible count does not match rules")
    accounting = _accounting(
        obj["trial_accounting"],
        features=configuration["features"],
        population=population,
        evaluated=evaluated,
    )
    lifecycle = _object(obj["lifecycle"], "lifecycle")
    if lifecycle != _LIFECYCLE:
        raise CrossRuleSearchError(
            "Cross rule search cannot claim selection or lifecycle changes"
        )
    if obj["producer_version"] != CROSS_RULE_SEARCH_PRODUCER_VERSION:
        raise CrossRuleSearchError(
            "Cross rule search producer_version is invalid"
        )
    content_hash = _hash(obj["content_hash"], "content_hash")
    normalized_without_hash = {
        "schema_version": CROSS_RULE_SEARCH_RESULT_SCHEMA_VERSION,
        "search_id": search_id,
        "request_hash": request_hash,
        "source": source,
        "population": population,
        "configuration": configuration,
        "search_space": search_space,
        "evaluated": evaluated,
        "truncated": truncated,
        "eligible": eligible,
        "rules": rules,
        "trial_accounting": accounting,
        "lifecycle": dict(_LIFECYCLE),
        "producer_version": CROSS_RULE_SEARCH_PRODUCER_VERSION,
    }
    expected_hash = _sha256(_canonical_json(normalized_without_hash))
    if not hmac.compare_digest(content_hash, expected_hash):
        raise CrossRuleSearchError(
            "content_hash does not match canonical Cross rule search result"
        )
    return {**normalized_without_hash, "content_hash": content_hash}


def canonical_cross_rule_search_result_json(
    payload: Mapping[str, Any],
) -> str:
    return _canonical_json(validate_cross_rule_search_result(payload))


def parse_cross_rule_search_result_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    if not isinstance(raw, (str, bytes, bytearray)):
        raise CrossRuleSearchError("Cross rule search JSON must be text")
    try:
        payload = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossRuleSearchError(
            f"Cross rule search JSON is invalid: {exc}"
        ) from exc
    return validate_cross_rule_search_result(
        _object(payload, "Cross rule search JSON")
    )


def _condition_candidates(feature: Mapping[str, Any]) -> list[dict[str, Any]]:
    directions = (
        ("gte",)
        if feature["risk_direction"] == "increasing"
        else (
            ("lt",)
            if feature["risk_direction"] == "decreasing"
            else ("gte", "lt")
        )
    )
    candidates: list[dict[str, Any]] = []
    for operator in directions:
        thresholds = (
            reversed(feature["thresholds"])
            if operator == "gte"
            else feature["thresholds"]
        )
        for threshold in thresholds:
            for include_missing in (
                (False, True)
                if feature["missing_count"] > 0
                else (False,)
            ):
                candidates.append(
                    {
                        "feature": feature["feature"],
                        "method": feature["method"],
                        "operator": operator,
                        "threshold": threshold,
                        "include_missing": include_missing,
                    }
                )
    return candidates


def _search_space(
    features: Sequence[Mapping[str, Any]],
    *,
    dimension: int,
) -> int:
    counts = {
        item["feature"]: len(_condition_candidates(item))
        for item in features
    }
    return sum(
        math.prod(counts[name] for name in combo)
        for combo in combinations(sorted(counts), dimension)
    )


def _feature(
    value: object,
    *,
    index: int,
    population_count: int | None,
) -> dict[str, Any]:
    obj = _object(value, f"features[{index}]")
    _exact(obj, _FEATURE_FIELDS, f"features[{index}]")
    feature = _text(obj["feature"], f"features[{index}].feature")
    method = _text(obj["method"], f"features[{index}].method")
    if method not in _METHODS:
        raise CrossRuleSearchError(
            f"features[{index}].method is unsupported"
        )
    direction = _text(
        obj["risk_direction"],
        f"features[{index}].risk_direction",
    )
    if direction not in _RISK_DIRECTIONS:
        raise CrossRuleSearchError(
            f"features[{index}].risk_direction is invalid"
        )
    raw_thresholds = _array(
        obj["thresholds"],
        f"features[{index}].thresholds",
    )
    if not 1 <= len(raw_thresholds) <= MAX_THRESHOLDS_PER_FEATURE:
        raise CrossRuleSearchError(
            f"features[{index}].thresholds must contain 1..8 values"
        )
    thresholds = sorted(
        _finite(item, f"features[{index}].thresholds[{position}]")
        for position, item in enumerate(raw_thresholds)
    )
    if len(set(thresholds)) != len(thresholds):
        raise CrossRuleSearchError(
            f"features[{index}].thresholds must be unique"
        )
    raw_excluded = _array(
        obj["excluded_values"],
        f"features[{index}].excluded_values",
        allow_empty=True,
    )
    if len(raw_excluded) > MAX_THRESHOLDS_PER_FEATURE:
        raise CrossRuleSearchError(
            f"features[{index}].excluded_values exceeds 8 values"
        )
    excluded_values = sorted(
        _finite(item, f"features[{index}].excluded_values[{position}]")
        for position, item in enumerate(raw_excluded)
    )
    if len(set(excluded_values)) != len(excluded_values):
        raise CrossRuleSearchError(
            f"features[{index}].excluded_values must be unique"
        )
    missing_count = _integer(
        obj["missing_count"],
        f"features[{index}].missing_count",
        minimum=0,
        maximum=population_count,
    )
    missing_bad = _integer(
        obj["missing_bad"],
        f"features[{index}].missing_bad",
        minimum=0,
        maximum=missing_count,
    )
    return {
        "feature": feature,
        "method": method,
        "risk_direction": direction,
        "thresholds": thresholds,
        "excluded_values": excluded_values,
        "missing_count": missing_count,
        "missing_bad": missing_bad,
    }


def _trial(
    value: object,
    *,
    index: int,
    feature_by_name: Mapping[str, Mapping[str, Any]],
    dimension: int,
    population: Mapping[str, Any],
) -> dict[str, Any]:
    obj = _object(value, f"trials[{index}]")
    _exact(obj, _TRIAL_FIELDS, f"trials[{index}]")
    conditions = [
        _condition(
            item,
            name=f"trials[{index}].conditions[{position}]",
            feature_by_name=feature_by_name,
        )
        for position, item in enumerate(
            _array(obj["conditions"], f"trials[{index}].conditions")
        )
    ]
    if len(conditions) != dimension:
        raise CrossRuleSearchError(
            f"trials[{index}].conditions must contain dimension values"
        )
    conditions.sort(key=lambda item: item["feature"])
    if len({item["feature"] for item in conditions}) != len(conditions):
        raise CrossRuleSearchError(
            f"trials[{index}].conditions features must be unique"
        )
    count = _integer(
        obj["count"],
        f"trials[{index}].count",
        minimum=0,
        maximum=population["row_count"],
    )
    good = _integer(
        obj["good"],
        f"trials[{index}].good",
        minimum=0,
        maximum=count,
    )
    bad = _integer(
        obj["bad"],
        f"trials[{index}].bad",
        minimum=0,
        maximum=count,
    )
    if good + bad != count:
        raise CrossRuleSearchError(
            f"trials[{index}] good + bad must equal count"
        )
    loan, overdue = _amount_pair(
        obj["loan_amount_sum"],
        obj["overdue_amount_sum"],
        name=f"trials[{index}]",
    )
    if (loan is None) != (population["loan_amount_sum"] is None):
        raise CrossRuleSearchError(
            f"trials[{index}] amount availability differs from population"
        )
    return {
        "conditions": conditions,
        "count": count,
        "good": good,
        "bad": bad,
        "loan_amount_sum": loan,
        "overdue_amount_sum": overdue,
    }


def _condition(
    value: object,
    *,
    name: str,
    feature_by_name: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    obj = _object(value, name)
    _exact(obj, _CONDITION_FIELDS, name)
    feature = _text(obj["feature"], f"{name}.feature")
    source = feature_by_name.get(feature)
    if source is None:
        raise CrossRuleSearchError(f"{name}.feature is not configured")
    method = _text(obj["method"], f"{name}.method")
    if method != source["method"]:
        raise CrossRuleSearchError(f"{name}.method differs from feature")
    operator = _text(obj["operator"], f"{name}.operator")
    if operator not in _OPERATORS:
        raise CrossRuleSearchError(f"{name}.operator is invalid")
    allowed = (
        {"gte"}
        if source["risk_direction"] == "increasing"
        else (
            {"lt"}
            if source["risk_direction"] == "decreasing"
            else set(_OPERATORS)
        )
    )
    if operator not in allowed:
        raise CrossRuleSearchError(
            f"{name}.operator disagrees with risk direction"
        )
    threshold = _finite(obj["threshold"], f"{name}.threshold")
    if threshold not in source["thresholds"]:
        raise CrossRuleSearchError(
            f"{name}.threshold is not a configured boundary"
        )
    include_missing = _boolean(
        obj["include_missing"],
        f"{name}.include_missing",
    )
    if include_missing and source["missing_count"] == 0:
        raise CrossRuleSearchError(
            f"{name}.include_missing requires observed missing rows"
        )
    return {
        "feature": feature,
        "method": method,
        "operator": operator,
        "threshold": threshold,
        "include_missing": include_missing,
    }


def _derived_metrics(
    *,
    population: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> dict[str, Any]:
    count = trial["count"]
    bad = trial["bad"]
    bad_rate = bad / count if count else 0.0
    overall_bad_rate = population["bad"] / population["row_count"]
    lift = bad_rate / overall_bad_rate if overall_bad_rate else 0.0
    bad_capture = bad / population["bad"] if population["bad"] else 0.0
    loan = trial["loan_amount_sum"]
    overdue = trial["overdue_amount_sum"]
    if loan is None:
        amount_rate = None
        amount_lift = None
    else:
        amount_rate = overdue / loan if loan else 0.0
        population_loan = population["loan_amount_sum"]
        population_overdue = population["overdue_amount_sum"]
        overall_amount_rate = (
            population_overdue / population_loan
            if population_loan
            else 0.0
        )
        amount_lift = (
            amount_rate / overall_amount_rate
            if overall_amount_rate
            else 0.0
        )
    return {
        "count": count,
        "good": trial["good"],
        "bad": bad,
        "hit_share": count / population["row_count"],
        "bad_rate": bad_rate,
        "lift": lift,
        "bad_capture_rate": bad_capture,
        "loan_amount_sum": loan,
        "overdue_amount_sum": overdue,
        "amount_overdue_rate": amount_rate,
        "amount_lift": amount_lift,
    }


def _constraint_failures(
    metrics: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> list[str]:
    failures = []
    if metrics["lift"] < constraints["min_lift"]:
        failures.append("lift_below_minimum")
    if metrics["bad"] < constraints["min_bad_count"]:
        failures.append("bad_count_below_minimum")
    if metrics["hit_share"] > constraints["max_hit_share"]:
        failures.append("hit_share_above_maximum")
    minimum_amount = constraints["min_amount_lift"]
    if minimum_amount is not None and (
        metrics["amount_lift"] is None
        or metrics["amount_lift"] < minimum_amount
    ):
        failures.append("amount_lift_below_minimum")
    return failures


def _rank_key(rule: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = rule["metrics"]
    amount_lift = metrics["amount_lift"]
    return (
        not rule["eligible"],
        -metrics["lift"],
        -(amount_lift if amount_lift is not None else -1.0),
        -metrics["bad_capture_rate"],
        metrics["hit_share"],
        rule["rule_id"],
    )


def _result_rule(
    value: object,
    *,
    index: int,
    source: Mapping[str, Any],
    population: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    name = f"rules[{index}]"
    obj = _object(value, name)
    _exact(obj, _RULE_FIELDS, name)
    feature_by_name = {
        item["feature"]: item for item in configuration["features"]
    }
    conditions = [
        _condition(
            item,
            name=f"{name}.conditions[{position}]",
            feature_by_name=feature_by_name,
        )
        for position, item in enumerate(
            _array(obj["conditions"], f"{name}.conditions")
        )
    ]
    if len(conditions) != configuration["dimension"]:
        raise CrossRuleSearchError(
            f"{name}.conditions must contain configured dimension"
        )
    conditions.sort(key=lambda item: item["feature"])
    if len({item["feature"] for item in conditions}) != len(conditions):
        raise CrossRuleSearchError(f"{name}.conditions features must be unique")
    rule_id = _id(obj["rule_id"], f"{name}.rule_id", _RULE_ID_RE)
    expected_rule_id = _stable_id(
        "cross-rule",
        {"source": source, "conditions": conditions},
    )
    if rule_id != expected_rule_id:
        raise CrossRuleSearchError(
            f"{name}.rule_id does not match canonical conditions"
        )
    metrics_obj = _object(obj["metrics"], f"{name}.metrics")
    _exact(metrics_obj, _METRIC_FIELDS, f"{name}.metrics")
    trial = {
        "count": _integer(
            metrics_obj["count"],
            f"{name}.metrics.count",
            minimum=0,
            maximum=population["row_count"],
        ),
        "good": _integer(
            metrics_obj["good"],
            f"{name}.metrics.good",
            minimum=0,
        ),
        "bad": _integer(
            metrics_obj["bad"],
            f"{name}.metrics.bad",
            minimum=0,
        ),
        "loan_amount_sum": _optional_finite(
            metrics_obj["loan_amount_sum"],
            f"{name}.metrics.loan_amount_sum",
            minimum=0.0,
        ),
        "overdue_amount_sum": _optional_finite(
            metrics_obj["overdue_amount_sum"],
            f"{name}.metrics.overdue_amount_sum",
            minimum=0.0,
        ),
    }
    if trial["good"] + trial["bad"] != trial["count"]:
        raise CrossRuleSearchError(
            f"{name}.metrics good + bad must equal count"
        )
    derived = _derived_metrics(population=population, trial=trial)
    for field in _METRIC_FIELDS:
        if field in {
            "count",
            "good",
            "bad",
            "loan_amount_sum",
            "overdue_amount_sum",
        }:
            continue
        actual = _optional_finite(
            metrics_obj[field],
            f"{name}.metrics.{field}",
            minimum=0.0,
        )
        expected = derived[field]
        if actual != expected:
            raise CrossRuleSearchError(
                f"{name}.metrics.{field} is not deterministically derived"
            )
    failures = [
        _text(item, f"{name}.constraint_failures[{position}]")
        for position, item in enumerate(
            _array(
                obj["constraint_failures"],
                f"{name}.constraint_failures",
                allow_empty=True,
            )
        )
    ]
    expected_failures = _constraint_failures(
        derived,
        configuration["constraints"],
    )
    if failures != expected_failures:
        raise CrossRuleSearchError(
            f"{name}.constraint_failures are inconsistent"
        )
    eligible = _boolean(obj["eligible"], f"{name}.eligible")
    if eligible is not (not failures):
        raise CrossRuleSearchError(f"{name}.eligible is inconsistent")
    rank = _integer(
        obj["rank"],
        f"{name}.rank",
        minimum=1,
    )
    if rank != index + 1:
        raise CrossRuleSearchError(f"{name}.rank is not contiguous")
    return {
        "rule_id": rule_id,
        "conditions": conditions,
        "metrics": derived,
        "constraint_failures": failures,
        "eligible": eligible,
        "rank": rank,
    }


def _configuration(
    value: object,
    *,
    population_count: int,
) -> dict[str, Any]:
    obj = _object(value, "configuration")
    _exact(obj, _CONFIGURATION_FIELDS, "configuration")
    dimension = _dimension(obj["dimension"])
    raw_features = _array(obj["features"], "configuration.features")
    if not MIN_FEATURES <= len(raw_features) <= MAX_FEATURES:
        raise CrossRuleSearchError(
            "configuration.features must contain 2..12 values"
        )
    features = [
        _feature(
            item,
            index=index,
            population_count=population_count,
        )
        for index, item in enumerate(raw_features)
    ]
    if features != sorted(features, key=lambda item: item["feature"]):
        raise CrossRuleSearchError(
            "configuration.features are not canonical"
        )
    if len({item["feature"] for item in features}) != len(features):
        raise CrossRuleSearchError("configuration.features must be unique")
    if len(features) < dimension:
        raise CrossRuleSearchError(
            "configuration.features cannot be smaller than dimension"
        )
    constraints = _constraints(obj["constraints"])
    max_trials = _integer(
        obj["max_trials"],
        "configuration.max_trials",
        minimum=1,
        maximum=MAX_TRIALS,
    )
    if obj["enumeration_policy"] != ENUMERATION_POLICY:
        raise CrossRuleSearchError(
            "configuration.enumeration_policy is invalid"
        )
    return {
        "dimension": dimension,
        "features": features,
        "constraints": constraints,
        "max_trials": max_trials,
        "enumeration_policy": ENUMERATION_POLICY,
    }


def _accounting(
    value: object,
    *,
    features: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    evaluated: int,
) -> dict[str, Any]:
    obj = _object(value, "trial_accounting")
    _exact(obj, _ACCOUNTING_FIELDS, "trial_accounting")
    limits = _object(obj["limits"], "trial_accounting.limits")
    _exact(
        limits,
        _ACCOUNTING_LIMIT_FIELDS,
        "trial_accounting.limits",
    )
    expected_limits = {
        "max_features": MAX_FEATURES,
        "max_thresholds_per_feature": MAX_THRESHOLDS_PER_FEATURE,
        "max_trials": MAX_TRIALS,
        "max_row_evaluations": MAX_ROW_EVALUATIONS,
        "max_artifact_bytes": MAX_ARTIFACT_BYTES,
    }
    if limits != expected_limits:
        raise CrossRuleSearchError("trial_accounting.limits changed")
    used = _object(obj["used"], "trial_accounting.used")
    _exact(used, _ACCOUNTING_USED_FIELDS, "trial_accounting.used")
    expected_used = {
        "features": len(features),
        "thresholds": sum(len(item["thresholds"]) for item in features),
        "trials": evaluated,
        "row_evaluations": population["row_count"] * evaluated,
    }
    if used != expected_used:
        raise CrossRuleSearchError("trial_accounting.used is inconsistent")
    return {"limits": expected_limits, "used": expected_used}


def _source(value: object) -> dict[str, str]:
    obj = _object(value, "source")
    _exact(obj, _SOURCE_FIELDS, "source")
    candidate_id = _id(
        obj["candidate_id"],
        "source.candidate_id",
        _CANDIDATE_ID_RE,
    )
    return {
        "candidate_id": candidate_id,
        "evidence_hash": _hash(
            obj["evidence_hash"],
            "source.evidence_hash",
        ),
        "sample_context_hash": _hash(
            obj["sample_context_hash"],
            "source.sample_context_hash",
        ),
    }


def _population(value: object) -> dict[str, Any]:
    obj = _object(value, "population")
    _exact(obj, _POPULATION_FIELDS, "population")
    row_count = _integer(
        obj["row_count"],
        "population.row_count",
        minimum=1,
    )
    good = _integer(
        obj["good"],
        "population.good",
        minimum=0,
        maximum=row_count,
    )
    bad = _integer(
        obj["bad"],
        "population.bad",
        minimum=0,
        maximum=row_count,
    )
    if good + bad != row_count:
        raise CrossRuleSearchError(
            "population good + bad must equal row_count"
        )
    loan, overdue = _amount_pair(
        obj["loan_amount_sum"],
        obj["overdue_amount_sum"],
        name="population",
    )
    return {
        "row_count": row_count,
        "good": good,
        "bad": bad,
        "loan_amount_sum": loan,
        "overdue_amount_sum": overdue,
    }


def _constraints(value: object) -> dict[str, Any]:
    obj = _object(value, "constraints")
    _exact(obj, _CONSTRAINT_FIELDS, "constraints")
    return {
        "min_lift": _finite(
            obj["min_lift"],
            "constraints.min_lift",
            minimum=0.0,
            maximum=1_000.0,
        ),
        "min_bad_count": _integer(
            obj["min_bad_count"],
            "constraints.min_bad_count",
            minimum=0,
        ),
        "max_hit_share": _finite(
            obj["max_hit_share"],
            "constraints.max_hit_share",
            minimum=0.0,
            maximum=1.0,
        ),
        "min_amount_lift": _optional_finite(
            obj["min_amount_lift"],
            "constraints.min_amount_lift",
            minimum=0.0,
            maximum=1_000.0,
        ),
    }


def _dimension(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) not in {2, 3}
    ):
        raise CrossRuleSearchError("dimension must be 2 or 3")
    return int(value)


def _amount_pair(
    loan_value: object,
    overdue_value: object,
    *,
    name: str,
) -> tuple[float | None, float | None]:
    loan = _optional_finite(
        loan_value,
        f"{name}.loan_amount_sum",
        minimum=0.0,
    )
    overdue = _optional_finite(
        overdue_value,
        f"{name}.overdue_amount_sum",
        minimum=0.0,
    )
    if (loan is None) != (overdue is None):
        raise CrossRuleSearchError(
            f"{name} amount sums must both be present or both be null"
        )
    return loan, overdue


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CrossRuleSearchError(f"{name} must be an object")
    return dict(value)


def _array(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or (not allow_empty and not value)
    ):
        suffix = "an array" if allow_empty else "a non-empty array"
        raise CrossRuleSearchError(f"{name} must be {suffix}")
    return list(value)


def _exact(
    obj: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(obj) != expected:
        raise CrossRuleSearchError(f"{name} fields are invalid")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise CrossRuleSearchError(f"{name} must be non-empty text")
    return value.strip()


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise CrossRuleSearchError(f"{name} must be boolean")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        upper = "" if maximum is None else f" and {maximum}"
        raise CrossRuleSearchError(
            f"{name} must be an integer between {minimum}{upper}"
        )
    normalized = int(value)
    if normalized < minimum or (
        maximum is not None and normalized > maximum
    ):
        upper = "" if maximum is None else f" and {maximum}"
        raise CrossRuleSearchError(
            f"{name} must be between {minimum}{upper}"
        )
    return normalized


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CrossRuleSearchError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CrossRuleSearchError(f"{name} must be a finite number")
    if minimum is not None and normalized < minimum:
        raise CrossRuleSearchError(f"{name} is below its minimum")
    if maximum is not None and normalized > maximum:
        raise CrossRuleSearchError(f"{name} exceeds its maximum")
    return normalized


def _optional_finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _finite(
        value,
        name,
        minimum=minimum,
        maximum=maximum,
    )


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise CrossRuleSearchError(
            f"{name} must be a lowercase SHA-256 hash"
        )
    return value


def _id(value: object, name: str, pattern: re.Pattern[str]) -> str:
    text = _text(value, name)
    if pattern.fullmatch(text) is None:
        raise CrossRuleSearchError(f"{name} has an invalid format")
    return text


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


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
        raise CrossRuleSearchError(
            "Cross rule search value is not canonical JSON"
        ) from exc


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CrossRuleSearchError(f"duplicate key: {key}")
        result[key] = value
    return result


__all__ = [
    "CROSS_RULE_SEARCH_PRODUCER_VERSION",
    "CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION",
    "CROSS_RULE_SEARCH_RESULT_SCHEMA_VERSION",
    "CrossRuleSearchError",
    "ENUMERATION_POLICY",
    "MAX_ARTIFACT_BYTES",
    "MAX_FEATURES",
    "MAX_ROW_EVALUATIONS",
    "MAX_THRESHOLDS_PER_FEATURE",
    "MAX_TRIALS",
    "canonical_cross_rule_search_result_json",
    "canonical_cross_rule_trial_prefix",
    "parse_cross_rule_search_result_json",
    "search_cross_threshold_rules",
    "validate_cross_rule_search_request",
    "validate_cross_rule_search_result",
]
