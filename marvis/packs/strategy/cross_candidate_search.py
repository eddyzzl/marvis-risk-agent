"""Pure deterministic search over aggregate two-dimensional Cross evidence.

The caller authenticates one univariate CandidateEvidence source and computes
one complete Cross Matrix asset per evaluated canonical feature pair.  This
kernel accepts only aggregate trial facts and immutable asset fingerprints.  It
does not read datasets, persist artifacts, select a pair, mutate a Pool, apply,
adopt, or deploy anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
from typing import Any

from marvis.packs.strategy.errors import StrategyError


CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION = (
    "strategy.cross-candidate-search-request.v1"
)
CROSS_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION = (
    "strategy.cross-candidate-search-result.v1"
)
CROSS_CANDIDATE_SEARCH_PRODUCER_VERSION = "strategy.cross-candidate-search/1"

MIN_FEATURES = 2
MAX_FEATURES = 20
MAX_PAIRS = 190
MAX_AXIS_BINS = 20
MAX_CELLS_PER_PAIR = 400
MAX_PAIR_ROW_EVALUATIONS = 50_000_000
MAX_AXIS_BIN_ROW_EVALUATIONS = 50_000_000
MAX_DERIVED_CELLS = 50_000
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
ENUMERATION_POLICY = "canonical_round_robin_pair_prefix.v1"

_METHODS = frozenset(
    {
        "equal_frequency",
        "equal_width",
        "chimerge",
        "tree",
        "manual",
        "categorical",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "population",
        "features",
        "pair_trials",
        "max_pairs",
    }
)
_SOURCE_FIELDS = frozenset(
    {"candidate_id", "evidence_hash", "sample_context_hash"}
)
_POPULATION_FIELDS = frozenset({"row_count", "good", "bad"})
_FEATURE_FIELDS = frozenset({"feature", "method", "axis_iv", "bin_count"})
_TRIAL_FIELDS = frozenset(
    {
        "x_feature",
        "y_feature",
        "cross_total_iv",
        "cell_count",
        "empty_cell_count",
        "min_nonempty_cell_count",
        "asset_fingerprint",
    }
)
_FINGERPRINT_FIELDS = frozenset(
    {
        "asset_id",
        "asset_hash",
        "measurement_hash",
        "matrix_hash",
        "summary_hash",
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
        "pairs",
        "trial_accounting",
        "lifecycle",
        "producer_version",
        "content_hash",
    }
)
_CONFIGURATION_FIELDS = frozenset(
    {"features", "max_pairs", "enumeration_policy"}
)
_PAIR_FIELDS = frozenset(
    {
        "pair_id",
        "x_feature",
        "x_method",
        "y_feature",
        "y_method",
        "x_axis_iv",
        "y_axis_iv",
        "cross_total_iv",
        "interaction_gain_iv",
        "cell_count",
        "empty_cell_count",
        "empty_cell_share",
        "min_nonempty_cell_count",
        "eligible",
        "rank",
        "asset_fingerprint",
    }
)
_ACCOUNTING_FIELDS = frozenset({"limits", "used"})
_LIMIT_FIELDS = frozenset(
    {
        "max_features",
        "max_pairs",
        "max_cells_per_pair",
        "max_pair_row_evaluations",
        "max_axis_bin_row_evaluations",
        "max_derived_cells",
        "max_artifact_bytes",
    }
)
_USED_FIELDS = frozenset(
    {
        "features",
        "pairs",
        "pair_row_evaluations",
        "axis_bin_row_evaluations",
        "derived_cells",
    }
)
_LIFECYCLE = {
    "selected": False,
    "admitted": False,
    "applied": False,
    "adopted": False,
    "deployed": False,
}
_LIFECYCLE_FIELDS = frozenset(_LIFECYCLE)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_SEARCH_ID_RE = re.compile(r"^cross-search-[0-9a-f]{32}$")
_PAIR_ID_RE = re.compile(r"^cross-pair-[0-9a-f]{32}$")


class CrossCandidateSearchError(StrategyError):
    """The bounded Cross candidate search contract was violated."""


def validate_cross_candidate_search_request(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize aggregate Cross pair search inputs."""

    if not isinstance(payload, Mapping):
        raise CrossCandidateSearchError("Cross search request must be an object")
    _exact(payload, _REQUEST_FIELDS, "Cross search request")
    if payload["schema_version"] != CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION:
        raise CrossCandidateSearchError(
            "schema_version must be "
            + CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION
        )
    source = _source(payload["source"])
    population = _population(payload["population"])
    features_raw = _array(payload["features"], "features")
    if not MIN_FEATURES <= len(features_raw) <= MAX_FEATURES:
        raise CrossCandidateSearchError("features must contain 2..20 explicit features")
    features = sorted(
        (_feature(item, index=index) for index, item in enumerate(features_raw)),
        key=lambda item: item["feature"],
    )
    feature_names = [item["feature"] for item in features]
    if len(set(feature_names)) != len(feature_names):
        raise CrossCandidateSearchError("features must be unique")
    max_pairs = _integer(
        payload["max_pairs"],
        "max_pairs",
        minimum=1,
        maximum=MAX_PAIRS,
    )
    canonical_pairs = _round_robin_pairs(feature_names)
    evaluated_count = min(max_pairs, len(canonical_pairs))
    expected_pairs = canonical_pairs[:evaluated_count]
    feature_by_name = {item["feature"]: item for item in features}

    trial_map: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(_array(payload["pair_trials"], "pair_trials")):
        trial = _trial(
            item,
            index=index,
            feature_by_name=feature_by_name,
            population_count=population["row_count"],
        )
        key = (trial["x_feature"], trial["y_feature"])
        if key in trial_map:
            raise CrossCandidateSearchError(
                "pair_trials contains a duplicate canonical pair"
            )
        trial_map[key] = trial
    if set(trial_map) != set(expected_pairs):
        raise CrossCandidateSearchError(
            "pair_trials must exactly match the canonical round-robin prefix"
        )
    trials = [trial_map[pair] for pair in expected_pairs]
    _require_budgets(
        population=population,
        features=features,
        trials=trials,
    )
    return {
        "schema_version": CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "source": source,
        "population": population,
        "features": features,
        "pair_trials": trials,
        "max_pairs": max_pairs,
    }


def search_cross_candidate_pairs(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Rank one bounded canonical pair prefix without choosing a pair."""

    normalized = validate_cross_candidate_search_request(request)
    request_hash = _sha256(_canonical_json(normalized))
    search_id = _stable_id("cross-search", {"request_hash": request_hash})
    features = normalized["features"]
    feature_by_name = {item["feature"]: item for item in features}
    ranked: list[dict[str, Any]] = []
    for trial in normalized["pair_trials"]:
        x = feature_by_name[trial["x_feature"]]
        y = feature_by_name[trial["y_feature"]]
        cross_total_iv = trial["cross_total_iv"]
        interaction_gain_iv = cross_total_iv - max(x["axis_iv"], y["axis_iv"])
        empty_share = trial["empty_cell_count"] / trial["cell_count"]
        eligible = trial["empty_cell_count"] == 0
        pair_core = {
            "x_feature": x["feature"],
            "x_method": x["method"],
            "y_feature": y["feature"],
            "y_method": y["method"],
        }
        ranked.append(
            {
                "pair_id": _stable_id(
                    "cross-pair",
                    {
                        "source": normalized["source"],
                        **pair_core,
                    },
                ),
                **pair_core,
                "x_axis_iv": x["axis_iv"],
                "y_axis_iv": y["axis_iv"],
                "cross_total_iv": cross_total_iv,
                "interaction_gain_iv": interaction_gain_iv,
                "cell_count": trial["cell_count"],
                "empty_cell_count": trial["empty_cell_count"],
                "empty_cell_share": empty_share,
                "min_nonempty_cell_count": trial["min_nonempty_cell_count"],
                "eligible": eligible,
                "rank": 0,
                "asset_fingerprint": trial["asset_fingerprint"],
            }
        )
    ranked.sort(
        key=lambda item: (
            not item["eligible"],
            -item["interaction_gain_iv"],
            -item["cross_total_iv"],
            -item["min_nonempty_cell_count"],
            item["empty_cell_share"],
            item["pair_id"],
        )
    )
    pairs = [
        {**item, "rank": rank}
        for rank, item in enumerate(ranked, start=1)
    ]
    search_space = len(_round_robin_pairs([item["feature"] for item in features]))
    accounting = _trial_accounting(
        population=normalized["population"],
        features=features,
        trials=normalized["pair_trials"],
    )
    body = {
        "schema_version": CROSS_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION,
        "search_id": search_id,
        "request_hash": request_hash,
        "source": normalized["source"],
        "population": normalized["population"],
        "configuration": {
            "features": features,
            "max_pairs": normalized["max_pairs"],
            "enumeration_policy": ENUMERATION_POLICY,
        },
        "search_space": search_space,
        "evaluated": len(pairs),
        "truncated": len(pairs) < search_space,
        "eligible": sum(1 for item in pairs if item["eligible"]),
        "pairs": pairs,
        "trial_accounting": accounting,
        "lifecycle": dict(_LIFECYCLE),
        "producer_version": CROSS_CANDIDATE_SEARCH_PRODUCER_VERSION,
    }
    content_hash = _sha256(_canonical_json(body))
    result = {**body, "content_hash": content_hash}
    if len(_canonical_json(result).encode("utf-8")) > MAX_ARTIFACT_BYTES:
        raise CrossCandidateSearchError(
            f"Cross search artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
        )
    return result


def validate_cross_candidate_search_result(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a stored result and independently rebuild every derived field."""

    if not isinstance(payload, Mapping):
        raise CrossCandidateSearchError("Cross search result must be an object")
    _exact(payload, _RESULT_FIELDS, "Cross search result")
    content_hash = _hash(payload["content_hash"], "content_hash")
    body = {key: payload[key] for key in payload if key != "content_hash"}
    expected_content_hash = _sha256(_canonical_json(body))
    if not hmac.compare_digest(content_hash, expected_content_hash):
        raise CrossCandidateSearchError("Cross search content_hash changed")

    if payload["schema_version"] != CROSS_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION:
        raise CrossCandidateSearchError("Cross search result schema_version is invalid")
    _search_id(payload["search_id"])
    request_hash = _hash(payload["request_hash"], "request_hash")
    source = _source(payload["source"])
    population = _population(payload["population"])
    configuration = _object(payload["configuration"], "configuration")
    _exact(configuration, _CONFIGURATION_FIELDS, "configuration")
    if configuration["enumeration_policy"] != ENUMERATION_POLICY:
        raise CrossCandidateSearchError("Cross search enumeration_policy changed")
    features = _array(configuration["features"], "configuration.features")
    pairs = _array(payload["pairs"], "pairs")
    trials: list[dict[str, Any]] = []
    for index, pair_value in enumerate(pairs):
        pair = _object(pair_value, f"pairs[{index}]")
        _exact(pair, _PAIR_FIELDS, f"pairs[{index}]")
        _pair_id(pair["pair_id"], f"pairs[{index}].pair_id")
        _method(pair["x_method"], f"pairs[{index}].x_method")
        _method(pair["y_method"], f"pairs[{index}].y_method")
        _finite(pair["x_axis_iv"], f"pairs[{index}].x_axis_iv", minimum=0.0)
        _finite(pair["y_axis_iv"], f"pairs[{index}].y_axis_iv", minimum=0.0)
        _finite(pair["interaction_gain_iv"], f"pairs[{index}].interaction_gain_iv")
        _finite(pair["empty_cell_share"], f"pairs[{index}].empty_cell_share")
        if not isinstance(pair["eligible"], bool):
            raise CrossCandidateSearchError(
                f"pairs[{index}].eligible must be boolean"
            )
        _integer(pair["rank"], f"pairs[{index}].rank", minimum=1, maximum=MAX_PAIRS)
        trials.append(
            {
                "x_feature": pair["x_feature"],
                "y_feature": pair["y_feature"],
                "cross_total_iv": pair["cross_total_iv"],
                "cell_count": pair["cell_count"],
                "empty_cell_count": pair["empty_cell_count"],
                "min_nonempty_cell_count": pair["min_nonempty_cell_count"],
                "asset_fingerprint": pair["asset_fingerprint"],
            }
        )
    request = {
        "schema_version": CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "source": source,
        "population": population,
        "features": features,
        "pair_trials": trials,
        "max_pairs": configuration["max_pairs"],
    }
    rebuilt = search_cross_candidate_pairs(request)
    if rebuilt != dict(payload):
        if not hmac.compare_digest(rebuilt["request_hash"], request_hash):
            raise CrossCandidateSearchError("Cross search request_hash changed")
        raise CrossCandidateSearchError(
            "Cross search result is not deterministically derived"
        )
    return rebuilt


def canonical_cross_candidate_search_request_json(
    payload: Mapping[str, Any],
) -> str:
    return _canonical_json(validate_cross_candidate_search_request(payload))


def canonical_cross_candidate_search_result_json(
    payload: Mapping[str, Any],
) -> str:
    return _canonical_json(validate_cross_candidate_search_result(payload))


def parse_cross_candidate_search_result_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    if not isinstance(raw, str | bytes | bytearray):
        raise CrossCandidateSearchError("Cross search result JSON must be text or bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_object_no_duplicates)
    except CrossCandidateSearchError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CrossCandidateSearchError(
            f"Cross search result is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CrossCandidateSearchError(
            "Cross search result JSON must contain an object"
        )
    return validate_cross_candidate_search_result(value)


def canonical_pair_prefix(
    features: Sequence[str],
    *,
    max_pairs: int,
) -> tuple[tuple[str, str], ...]:
    """Expose the stable bounded enumeration order to the governed Tool."""

    names = sorted(_text(item, "feature") for item in features)
    if len(set(names)) != len(names):
        raise CrossCandidateSearchError("features must be unique")
    if not MIN_FEATURES <= len(names) <= MAX_FEATURES:
        raise CrossCandidateSearchError("features must contain 2..20 explicit features")
    budget = _integer(
        max_pairs,
        "max_pairs",
        minimum=1,
        maximum=MAX_PAIRS,
    )
    return tuple(_round_robin_pairs(names)[:budget])


def asset_fingerprint(asset: Mapping[str, Any]) -> dict[str, str]:
    """Project the immutable full-asset identity used by from-search replay."""

    if not isinstance(asset, Mapping):
        raise CrossCandidateSearchError("Cross Matrix asset must be an object")
    try:
        fingerprint = {
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "measurement_hash": asset["measurement"]["measurement_hash"],
            "matrix_hash": asset["matrix"]["matrix_hash"],
            "summary_hash": asset["summary"]["summary_hash"],
        }
    except (KeyError, TypeError) as exc:
        raise CrossCandidateSearchError(
            "Cross Matrix asset fingerprint is incomplete"
        ) from exc
    return _fingerprint(fingerprint, "asset_fingerprint")


def _round_robin_pairs(features: Sequence[str]) -> list[tuple[str, str]]:
    names = list(features)
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for offset in range(1, len(names)):
        for index, left in enumerate(names):
            right = names[(index + offset) % len(names)]
            pair = tuple(sorted((left, right)))
            if pair[0] == pair[1] or pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    expected = len(names) * (len(names) - 1) // 2
    if len(pairs) != expected:
        raise CrossCandidateSearchError(
            "canonical round-robin pair enumeration did not conserve search space"
        )
    return pairs


def _source(value: object) -> dict[str, str]:
    obj = _object(value, "source")
    _exact(obj, _SOURCE_FIELDS, "source")
    candidate_id = _text(obj["candidate_id"], "source.candidate_id")
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise CrossCandidateSearchError("source.candidate_id is invalid")
    return {
        "candidate_id": candidate_id,
        "evidence_hash": _hash(obj["evidence_hash"], "source.evidence_hash"),
        "sample_context_hash": _hash(
            obj["sample_context_hash"],
            "source.sample_context_hash",
        ),
    }


def _population(value: object) -> dict[str, int]:
    obj = _object(value, "population")
    _exact(obj, _POPULATION_FIELDS, "population")
    row_count = _integer(obj["row_count"], "population.row_count", minimum=1)
    good = _integer(obj["good"], "population.good", minimum=0)
    bad = _integer(obj["bad"], "population.bad", minimum=0)
    if good + bad != row_count:
        raise CrossCandidateSearchError("population good + bad must equal row_count")
    return {"row_count": row_count, "good": good, "bad": bad}


def _feature(value: object, *, index: int) -> dict[str, Any]:
    obj = _object(value, f"features[{index}]")
    _exact(obj, _FEATURE_FIELDS, f"features[{index}]")
    bin_count = _integer(
        obj["bin_count"],
        f"features[{index}].bin_count",
        minimum=1,
        maximum=MAX_AXIS_BINS,
        maximum_message=(
            f"features[{index}].bin_count exceeds 20 axis bins / 400 cells"
        ),
    )
    return {
        "feature": _text(obj["feature"], f"features[{index}].feature"),
        "method": _method(obj["method"], f"features[{index}].method"),
        "axis_iv": _finite(
            obj["axis_iv"],
            f"features[{index}].axis_iv",
            minimum=0.0,
        ),
        "bin_count": bin_count,
    }


def _trial(
    value: object,
    *,
    index: int,
    feature_by_name: Mapping[str, Mapping[str, Any]],
    population_count: int,
) -> dict[str, Any]:
    label = f"pair_trials[{index}]"
    obj = _object(value, label)
    _exact(obj, _TRIAL_FIELDS, label)
    raw_x = _text(obj["x_feature"], f"{label}.x_feature")
    raw_y = _text(obj["y_feature"], f"{label}.y_feature")
    if raw_x == raw_y:
        raise CrossCandidateSearchError(f"{label} axes must be distinct")
    unknown = sorted({raw_x, raw_y} - set(feature_by_name))
    if unknown:
        raise CrossCandidateSearchError(
            f"{label} contains unknown features: " + ", ".join(unknown)
        )
    x_feature, y_feature = sorted((raw_x, raw_y))
    required_cells = (
        feature_by_name[x_feature]["bin_count"]
        * feature_by_name[y_feature]["bin_count"]
    )
    cell_count = _integer(
        obj["cell_count"],
        f"{label}.cell_count",
        minimum=1,
        maximum=MAX_CELLS_PER_PAIR,
    )
    if cell_count != required_cells:
        raise CrossCandidateSearchError(
            f"{label}.cell_count must equal the complete Cartesian axis product"
        )
    empty_count = _integer(
        obj["empty_cell_count"],
        f"{label}.empty_cell_count",
        minimum=0,
        maximum=cell_count,
    )
    minimum_nonempty = _integer(
        obj["min_nonempty_cell_count"],
        f"{label}.min_nonempty_cell_count",
        minimum=0,
        maximum=population_count,
    )
    if empty_count == cell_count:
        if minimum_nonempty != 0:
            raise CrossCandidateSearchError(
                f"{label}.min_nonempty_cell_count must be zero when all cells are empty"
            )
    elif minimum_nonempty < 1:
        raise CrossCandidateSearchError(
            f"{label}.min_nonempty_cell_count must be positive"
        )
    return {
        "x_feature": x_feature,
        "y_feature": y_feature,
        "cross_total_iv": _finite(
            obj["cross_total_iv"],
            f"{label}.cross_total_iv",
            minimum=0.0,
        ),
        "cell_count": cell_count,
        "empty_cell_count": empty_count,
        "min_nonempty_cell_count": minimum_nonempty,
        "asset_fingerprint": _fingerprint(
            obj["asset_fingerprint"],
            f"{label}.asset_fingerprint",
        ),
    }


def _fingerprint(value: object, label: str) -> dict[str, str]:
    obj = _object(value, label)
    _exact(obj, _FINGERPRINT_FIELDS, label)
    asset_id = _text(obj["asset_id"], f"{label}.asset_id")
    if _ASSET_ID_RE.fullmatch(asset_id) is None:
        raise CrossCandidateSearchError(f"{label}.asset_id is invalid")
    return {
        "asset_id": asset_id,
        "asset_hash": _hash(obj["asset_hash"], f"{label}.asset_hash"),
        "measurement_hash": _hash(
            obj["measurement_hash"],
            f"{label}.measurement_hash",
        ),
        "matrix_hash": _hash(obj["matrix_hash"], f"{label}.matrix_hash"),
        "summary_hash": _hash(obj["summary_hash"], f"{label}.summary_hash"),
    }


def _require_budgets(
    *,
    population: Mapping[str, int],
    features: Sequence[Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
) -> None:
    accounting = _trial_accounting(
        population=population,
        features=features,
        trials=trials,
    )["used"]
    for field, limit in (
        ("pair_row_evaluations", MAX_PAIR_ROW_EVALUATIONS),
        ("axis_bin_row_evaluations", MAX_AXIS_BIN_ROW_EVALUATIONS),
        ("derived_cells", MAX_DERIVED_CELLS),
    ):
        if accounting[field] > limit:
            raise CrossCandidateSearchError(
                f"Cross search {field} exceeds hard budget "
                f"({accounting[field]} > {limit})"
            )


def _trial_accounting(
    *,
    population: Mapping[str, int],
    features: Sequence[Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = population["row_count"]
    return {
        "limits": {
            "max_features": MAX_FEATURES,
            "max_pairs": MAX_PAIRS,
            "max_cells_per_pair": MAX_CELLS_PER_PAIR,
            "max_pair_row_evaluations": MAX_PAIR_ROW_EVALUATIONS,
            "max_axis_bin_row_evaluations": MAX_AXIS_BIN_ROW_EVALUATIONS,
            "max_derived_cells": MAX_DERIVED_CELLS,
            "max_artifact_bytes": MAX_ARTIFACT_BYTES,
        },
        "used": {
            "features": len(features),
            "pairs": len(trials),
            "pair_row_evaluations": rows * len(trials),
            "axis_bin_row_evaluations": rows
            * sum(item["bin_count"] for item in features),
            "derived_cells": sum(item["cell_count"] for item in trials),
        },
    }


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossCandidateSearchError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise CrossCandidateSearchError(f"{label} keys must be strings")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, str | bytes | bytearray
    ):
        raise CrossCandidateSearchError(f"{label} must be an array")
    return list(value)


def _exact(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    unsupported = sorted(set(value) - expected)
    if missing or unsupported:
        detail: list[str] = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unsupported:
            detail.append("unsupported: " + ", ".join(unsupported))
        raise CrossCandidateSearchError(
            f"{label} has invalid fields (" + "; ".join(detail) + ")"
        )


def _method(value: object, label: str) -> str:
    method = _text(value, label)
    if method not in _METHODS:
        raise CrossCandidateSearchError(f"{label} is unsupported")
    return method


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CrossCandidateSearchError(f"{label} must be non-empty text")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
    maximum_message: str | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CrossCandidateSearchError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < minimum or (
        maximum is not None and normalized > maximum
    ):
        if maximum_message is not None and normalized > maximum:
            raise CrossCandidateSearchError(maximum_message)
        if maximum is None:
            raise CrossCandidateSearchError(f"{label} must be at least {minimum}")
        raise CrossCandidateSearchError(
            f"{label} must be between {minimum} and {maximum}"
        )
    return normalized


def _finite(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CrossCandidateSearchError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CrossCandidateSearchError(f"{label} must be a finite number")
    if minimum is not None and normalized < minimum:
        raise CrossCandidateSearchError(f"{label} must be at least {minimum}")
    return normalized


def _hash(value: object, label: str) -> str:
    text = _text(value, label)
    if _HASH_RE.fullmatch(text) is None:
        raise CrossCandidateSearchError(f"{label} must be lowercase sha256")
    return text


def _search_id(value: object) -> str:
    text = _text(value, "search_id")
    if _SEARCH_ID_RE.fullmatch(text) is None:
        raise CrossCandidateSearchError("search_id is invalid")
    return text


def _pair_id(value: object, label: str) -> str:
    text = _text(value, label)
    if _PAIR_ID_RE.fullmatch(text) is None:
        raise CrossCandidateSearchError(f"{label} is invalid")
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
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrossCandidateSearchError(
            "Cross search payload must be finite JSON"
        ) from exc


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CrossCandidateSearchError(f"duplicate key: {key}")
        result[key] = value
    return result


__all__ = [
    "CROSS_CANDIDATE_SEARCH_PRODUCER_VERSION",
    "CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION",
    "CROSS_CANDIDATE_SEARCH_RESULT_SCHEMA_VERSION",
    "CrossCandidateSearchError",
    "asset_fingerprint",
    "canonical_cross_candidate_search_request_json",
    "canonical_cross_candidate_search_result_json",
    "canonical_pair_prefix",
    "parse_cross_candidate_search_result_json",
    "search_cross_candidate_pairs",
    "validate_cross_candidate_search_request",
    "validate_cross_candidate_search_result",
]
