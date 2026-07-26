"""Deterministic cross-partition stability evidence for one Strategy Pool.

The caller authenticates one immutable ImpactCube artifact and passes its exact
reference here.  This persistence-free kernel fixes development as the
baseline and compares validation and/or OOT without mutating, adopting,
promoting, or deploying the Pool.  Approval and risk populations remain
separate, and no stability result is represented as effect validation.
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

import numpy as np

from marvis.packs.strategy.dsl import StrategyAction
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube import validate_strategy_impact_cube
from marvis.validation.binning import compute_psi


POOL_STABILITY_SCHEMA_VERSION = "strategy.pool-stability.v1"
POOL_STABILITY_PRODUCER_VERSION = "marvis.strategy.pool-stability/1"
MAX_POOL_STABILITY_JSON_BYTES = 64 * 1024 * 1024
MAX_POOL_STABILITY_CATEGORIES = 201

_MAX_SAFE_JSON_INTEGER = 2**53 - 1
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_STABILITY_ID_RE = re.compile(r"^strategy-pool-stability-[0-9a-f]{24}$")
_CUBE_ID_RE = re.compile(r"^strategy-impact-cube-[0-9a-f]{24}$")
_PARTITION_ORDER = ("development", "validation", "oot")
_COMPARISON_PARTITIONS = ("validation", "oot")
_POPULATION_ROLES = ("approval", "risk")
_DISTRIBUTION_BASES = ("waterfall_incremental", "new_action")
_STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)
_STRATEGY_ACTION_TYPES = {
    "approval": frozenset({"approval", "reject", "review"}),
    "reject": frozenset({"approval", "reject", "review"}),
    "limit": frozenset({"limit"}),
    "pricing": frozenset({"pricing"}),
    "segmentation": frozenset({"segment"}),
}

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "stability_id",
        "identity",
        "source_bindings",
        "baseline_partition",
        "comparison_partitions",
        "populations",
        "lifecycle",
        "conservation",
        "content_hash",
    }
)
_BODY_FIELDS = _TOP_LEVEL_FIELDS - {"stability_id", "content_hash"}
_IDENTITY_FIELDS = frozenset(
    {
        "pool_id",
        "task_id",
        "strategy_type",
        "revision",
        "revision_id",
        "snapshot_hash",
        "design_hash",
        "strategy_spec_hash",
    }
)
_SOURCE_BINDING_FIELDS = frozenset(
    {"impact_cube", "sample_design_v2", "dataset"}
)
_IMPACT_CUBE_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_cube_id",
        "expected_cube_content_hash",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "membership_id",
        "membership_content_hash",
        "bundle_artifact_id",
        "bundle_artifact_content_hash",
        "bundle_id",
        "bundle_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "analysis_universe_row_count",
        "partition_counts",
        "population_partition_counts",
    }
)
_DATASET_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "dataset_source_path",
        "dataset_registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    }
)
_POPULATION_FIELDS = frozenset(
    {
        "population_role",
        "development_sample_count",
        "comparisons",
        "conservation",
    }
)
_COMPARISON_FIELDS = frozenset(
    {"partition", "sample_count", "distributions", "conservation"}
)
_DISTRIBUTION_FIELDS = frozenset(
    {
        "basis",
        "development_sample_count",
        "comparison_sample_count",
        "categories",
        "psi",
        "max_abs_share_delta",
        "severity",
        "conservation",
    }
)
_CATEGORY_ROW_FIELDS = frozenset(
    {
        "category",
        "development_count",
        "comparison_count",
        "development_share",
        "comparison_share",
        "share_delta",
    }
)
_WATERFALL_CATEGORY_FIELDS = frozenset(
    {"kind", "position", "entry_id", "rule_id"}
)
_ACTION_CATEGORY_FIELDS = frozenset({"kind", "action"})
_DISTRIBUTION_CONSERVATION = {
    "development_counts_equal_sample": True,
    "comparison_counts_equal_sample": True,
    "development_shares_sum_to_one": True,
    "comparison_shares_sum_to_one": True,
}
_COMPARISON_CONSERVATION = {
    "all_distribution_counts_equal_population": True,
}
_POPULATION_CONSERVATION = {
    "development_count_matches_source": True,
    "comparison_counts_match_source": True,
}
_LIFECYCLE = {
    "read_only": True,
    "effect_validation": False,
    "automatic_promotion": False,
    "mutates_pool": False,
    "creates_strategy": False,
    "adopts_strategy": False,
    "promotes_strategy": False,
    "deploys_strategy": False,
}
_CONSERVATION = {
    "development_baseline_present": True,
    "comparison_partition_present": True,
    "approval_and_risk_populations_present": True,
    "all_distribution_counts_conserve": True,
}


class PoolStabilityError(StrategyError):
    """Pool stability input or immutable evidence failed closed."""


def build_strategy_pool_stability(
    *,
    impact_cube: Mapping[str, Any],
    impact_cube_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Build aggregate Pool stability from one authenticated ImpactCube."""

    try:
        cube = validate_strategy_impact_cube(impact_cube)
    except StrategyError as exc:
        raise PoolStabilityError(
            f"Pool stability ImpactCube is invalid: {exc}"
        ) from exc
    normalized_ref = _impact_cube_ref(impact_cube_ref)
    if (
        normalized_ref["expected_cube_id"] != cube["cube_id"]
        or not hmac.compare_digest(
            normalized_ref["expected_cube_content_hash"],
            cube["content_hash"],
        )
    ):
        raise PoolStabilityError(
            "impact_cube_ref does not bind the supplied ImpactCube"
        )

    partition_names = tuple(
        row["name"]
        for row in cube["partitions"]
        if row["role"] == "risk"
    )
    if "development" not in partition_names:
        raise PoolStabilityError(
            "Pool stability requires a development baseline partition"
        )
    comparison_partitions = [
        name for name in _COMPARISON_PARTITIONS if name in partition_names
    ]
    if not comparison_partitions:
        raise PoolStabilityError(
            "Pool stability requires validation and/or OOT evidence"
        )

    populations = [
        _build_population(
            cube,
            population_role=role,
            comparison_partitions=comparison_partitions,
        )
        for role in _POPULATION_ROLES
    ]
    body = {
        "schema_version": POOL_STABILITY_SCHEMA_VERSION,
        "producer_version": POOL_STABILITY_PRODUCER_VERSION,
        "identity": _json_value(cube["identity"], "ImpactCube identity"),
        "source_bindings": {
            "impact_cube": normalized_ref,
            "sample_design_v2": _json_value(
                cube["source_bindings"]["sample_design_v2"],
                "ImpactCube sample design V2 binding",
            ),
            "dataset": _json_value(
                cube["source_bindings"]["dataset"],
                "ImpactCube dataset binding",
            ),
        },
        "baseline_partition": "development",
        "comparison_partitions": comparison_partitions,
        "populations": populations,
        "lifecycle": dict(_LIFECYCLE),
        "conservation": dict(_CONSERVATION),
    }
    stability_id = (
        "strategy-pool-stability-" + _sha256(_canonical_json(body))[:24]
    )
    document = {**body, "stability_id": stability_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    return validate_strategy_pool_stability(document)


def validate_strategy_pool_stability(
    payload: Mapping[str, Any],
    *,
    impact_cube: Mapping[str, Any] | None = None,
    impact_cube_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate exact fields, hashes, all derived metrics, and conservation.

    Supplying ``impact_cube`` and ``impact_cube_ref`` additionally rebinds the
    artifact to its authenticated source and rejects a coherently rehashed copy
    that no longer matches that source.
    """

    if (impact_cube is None) is not (impact_cube_ref is None):
        raise PoolStabilityError(
            "source verification requires both impact_cube and impact_cube_ref"
        )
    obj = _json_object(payload, "Pool stability artifact")
    _exact_fields(obj, _TOP_LEVEL_FIELDS, "Pool stability artifact")
    raw_canonical = _canonical_json(obj)
    if len(raw_canonical.encode("utf-8")) > MAX_POOL_STABILITY_JSON_BYTES:
        raise PoolStabilityError("Pool stability JSON byte budget exceeded")

    stability_id = _text(obj["stability_id"], "stability_id")
    if _STABILITY_ID_RE.fullmatch(stability_id) is None:
        raise PoolStabilityError("stability_id has an invalid format")
    supplied_hash = _hash(obj["content_hash"], "content_hash")
    body = _normalize_body(
        {
            key: obj[key]
            for key in obj
            if key not in {"stability_id", "content_hash"}
        }
    )
    expected_id = (
        "strategy-pool-stability-" + _sha256(_canonical_json(body))[:24]
    )
    if not hmac.compare_digest(stability_id, expected_id):
        raise PoolStabilityError(
            "stability_id does not match canonical Pool stability evidence"
        )
    without_hash = {**body, "stability_id": stability_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise PoolStabilityError(
            "content_hash does not match canonical Pool stability evidence"
        )
    result = {**without_hash, "content_hash": supplied_hash}
    canonical = _canonical_json(result)
    if len(canonical.encode("utf-8")) > MAX_POOL_STABILITY_JSON_BYTES:
        raise PoolStabilityError("Pool stability JSON byte budget exceeded")

    if impact_cube is not None and impact_cube_ref is not None:
        expected = build_strategy_pool_stability(
            impact_cube=impact_cube,
            impact_cube_ref=impact_cube_ref,
        )
        if result != expected:
            raise PoolStabilityError(
                "Pool stability evidence changed from its authenticated "
                "ImpactCube source"
            )
    return result


def canonical_strategy_pool_stability_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole byte-stable JSON representation."""

    return _canonical_json(validate_strategy_pool_stability(payload))


def strategy_pool_stability_content_hash(
    payload: Mapping[str, Any],
) -> str:
    """Return the verified embedded content hash."""

    return validate_strategy_pool_stability(payload)["content_hash"]


def _build_population(
    cube: Mapping[str, Any],
    *,
    population_role: str,
    comparison_partitions: Sequence[str],
) -> dict[str, Any]:
    development = _overall_slice(
        cube,
        population_role=population_role,
        partition="development",
    )
    development_count = int(development["population"]["value"]["count"])
    comparisons = [
        _build_comparison(
            cube,
            population_role=population_role,
            development=development,
            development_count=development_count,
            partition=partition,
        )
        for partition in comparison_partitions
    ]
    return {
        "population_role": population_role,
        "development_sample_count": development_count,
        "comparisons": comparisons,
        "conservation": dict(_POPULATION_CONSERVATION),
    }


def _build_comparison(
    cube: Mapping[str, Any],
    *,
    population_role: str,
    development: Mapping[str, Any],
    development_count: int,
    partition: str,
) -> dict[str, Any]:
    comparison = _overall_slice(
        cube,
        population_role=population_role,
        partition=partition,
    )
    comparison_count = int(comparison["population"]["value"]["count"])
    distributions = [
        _waterfall_distribution(
            development,
            comparison,
            development_count=development_count,
            comparison_count=comparison_count,
        ),
        _new_action_distribution(
            cube,
            population_role=population_role,
            comparison_partition=partition,
            development_count=development_count,
            comparison_count=comparison_count,
        ),
    ]
    return {
        "partition": partition,
        "sample_count": comparison_count,
        "distributions": distributions,
        "conservation": dict(_COMPARISON_CONSERVATION),
    }


def _overall_slice(
    cube: Mapping[str, Any],
    *,
    population_role: str,
    partition: str,
) -> Mapping[str, Any]:
    rows = [
        row
        for row in cube["slices"]
        if row["population_role"] == population_role
        and row["family"] == "overall"
        and row["dimensions"]["partition"]["value"] == partition
    ]
    if (
        len(rows) != 1
        or rows[0]["availability"] != "present"
        or rows[0]["waterfall"]["availability"] != "present"
    ):
        raise PoolStabilityError(
            f"Pool stability requires one present overall waterfall for "
            f"{population_role}/{partition}"
        )
    return rows[0]


def _waterfall_distribution(
    development: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    development_count: int,
    comparison_count: int,
) -> dict[str, Any]:
    development_counts = _waterfall_counts(development)
    comparison_counts = _waterfall_counts(comparison)
    combined = {**development_counts, **comparison_counts}
    categories = sorted(
        (
            category
            for category, _count_value in combined.values()
            if category["kind"] == "pool_entry_incremental"
        ),
        key=lambda row: (
            int(row["position"]),
            _canonical_json(row),
        ),
    )
    categories.append(
        {
            "kind": "default_unmatched",
            "position": None,
            "entry_id": None,
            "rule_id": None,
        }
    )
    return _distribution_document(
        basis="waterfall_incremental",
        categories=categories,
        development_counts={
            token: count for token, (_category, count) in development_counts.items()
        },
        comparison_counts={
            token: count for token, (_category, count) in comparison_counts.items()
        },
        development_count=development_count,
        comparison_count=comparison_count,
    )


def _waterfall_counts(
    overall: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], int]]:
    waterfall = overall["waterfall"]["value"]
    result: dict[str, tuple[dict[str, Any], int]] = {}
    for entry in waterfall["entries"]:
        category = {
            "kind": "pool_entry_incremental",
            "position": int(entry["position"]),
            "entry_id": entry["entry_id"],
            "rule_id": entry["rule_id"],
        }
        token = _canonical_json(category)
        if token in result:
            raise PoolStabilityError(
                "ImpactCube overall waterfall categories are duplicated"
            )
        result[token] = (category, int(entry["incremental"]["count"]))
    default = {
        "kind": "default_unmatched",
        "position": None,
        "entry_id": None,
        "rule_id": None,
    }
    result[_canonical_json(default)] = (
        default,
        int(waterfall["default_unmatched"]["effect"]["count"]),
    )
    return result
def _new_action_distribution(
    cube: Mapping[str, Any],
    *,
    population_role: str,
    comparison_partition: str,
    development_count: int,
    comparison_count: int,
) -> dict[str, Any]:
    development = _new_action_counts(
        cube,
        population_role=population_role,
        partition="development",
    )
    comparison = _new_action_counts(
        cube,
        population_role=population_role,
        partition=comparison_partition,
    )
    tokens = sorted(set(development) | set(comparison))
    categories = [
        {
            "kind": "typed_action",
            "action": (
                development[token][0]
                if token in development
                else comparison[token][0]
            ),
        }
        for token in tokens
    ]
    return _distribution_document(
        basis="new_action",
        categories=categories,
        development_counts={
            _canonical_json(
                {"kind": "typed_action", "action": action}
            ): count
            for action, count in development.values()
        },
        comparison_counts={
            _canonical_json(
                {"kind": "typed_action", "action": action}
            ): count
            for action, count in comparison.values()
        },
        development_count=development_count,
        comparison_count=comparison_count,
    )


def _new_action_counts(
    cube: Mapping[str, Any],
    *,
    population_role: str,
    partition: str,
) -> dict[str, tuple[dict[str, Any], int]]:
    rows = [
        row
        for row in cube["slices"]
        if row["population_role"] == population_role
        and row["family"] == "new_action"
        and row["dimensions"]["partition"]["value"] == partition
    ]
    if not rows or any(row["availability"] != "present" for row in rows):
        raise PoolStabilityError(
            f"Pool stability requires present new_action slices for "
            f"{population_role}/{partition}"
        )
    result: dict[str, tuple[dict[str, Any], int]] = {}
    for row in rows:
        action = _json_value(
            row["dimensions"]["new_action_bucket"]["value"],
            "ImpactCube typed action bucket",
        )
        token = _canonical_json(action)
        if token in result:
            raise PoolStabilityError(
                "ImpactCube new_action categories are duplicated"
            )
        result[token] = (action, int(row["population"]["value"]["count"]))
    return result


def _distribution_document(
    *,
    basis: str,
    categories: Sequence[Mapping[str, Any]],
    development_counts: Mapping[str, int],
    comparison_counts: Mapping[str, int],
    development_count: int,
    comparison_count: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for category in categories:
        token = _canonical_json(category)
        baseline = int(development_counts.get(token, 0))
        target = int(comparison_counts.get(token, 0))
        development_share = _ratio(baseline, development_count)
        comparison_share = _ratio(target, comparison_count)
        rows.append(
            {
                "category": dict(category),
                "development_count": baseline,
                "comparison_count": target,
                "development_share": development_share,
                "comparison_share": comparison_share,
                "share_delta": float(
                    comparison_share - development_share
                ),
            }
        )
    if sum(row["development_count"] for row in rows) != development_count:
        raise PoolStabilityError(
            f"{basis} development counts do not conserve"
        )
    if sum(row["comparison_count"] for row in rows) != comparison_count:
        raise PoolStabilityError(
            f"{basis} comparison counts do not conserve"
        )
    expected = np.asarray(
        [row["development_share"] for row in rows],
        dtype=float,
    )
    actual = np.asarray(
        [row["comparison_share"] for row in rows],
        dtype=float,
    )
    psi = float(compute_psi(expected, actual))
    max_delta = max(abs(float(row["share_delta"])) for row in rows)
    return {
        "basis": basis,
        "development_sample_count": development_count,
        "comparison_sample_count": comparison_count,
        "categories": rows,
        "psi": psi,
        "max_abs_share_delta": float(max_delta),
        "severity": _severity(psi),
        "conservation": dict(_DISTRIBUTION_CONSERVATION),
    }


def _normalize_body(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(value, _BODY_FIELDS, "Pool stability body")
    if value["schema_version"] != POOL_STABILITY_SCHEMA_VERSION:
        raise PoolStabilityError("Pool stability schema_version is invalid")
    if value["producer_version"] != POOL_STABILITY_PRODUCER_VERSION:
        raise PoolStabilityError("Pool stability producer_version is invalid")
    identity = _identity(value["identity"])
    comparison_partitions = _comparison_partitions(
        value["comparison_partitions"]
    )
    if value["baseline_partition"] != "development":
        raise PoolStabilityError(
            "Pool stability baseline_partition must be development"
        )
    sources = _source_bindings(
        value["source_bindings"],
        identity=identity,
        partitions=("development", *comparison_partitions),
    )
    populations = _populations(
        value["populations"],
        identity=identity,
        sources=sources,
        comparison_partitions=comparison_partitions,
    )
    if value["lifecycle"] != _LIFECYCLE:
        raise PoolStabilityError(
            "Pool stability lifecycle must remain read-only, unvalidated, "
            "and non-promoting"
        )
    if value["conservation"] != _CONSERVATION:
        raise PoolStabilityError(
            "Pool stability top-level conservation checks must all pass"
        )
    return {
        "schema_version": POOL_STABILITY_SCHEMA_VERSION,
        "producer_version": POOL_STABILITY_PRODUCER_VERSION,
        "identity": identity,
        "source_bindings": sources,
        "baseline_partition": "development",
        "comparison_partitions": comparison_partitions,
        "populations": populations,
        "lifecycle": dict(_LIFECYCLE),
        "conservation": dict(_CONSERVATION),
    }


def _identity(value: object) -> dict[str, Any]:
    obj = _json_object(value, "identity")
    _exact_fields(obj, _IDENTITY_FIELDS, "identity")
    strategy_type = _text(obj["strategy_type"], "identity.strategy_type")
    if strategy_type not in _STRATEGY_TYPES:
        raise PoolStabilityError("identity.strategy_type is invalid")
    return {
        "pool_id": _text(obj["pool_id"], "identity.pool_id"),
        "task_id": _text(obj["task_id"], "identity.task_id"),
        "strategy_type": strategy_type,
        "revision": _positive_int(obj["revision"], "identity.revision"),
        "revision_id": _text(obj["revision_id"], "identity.revision_id"),
        "snapshot_hash": _hash(
            obj["snapshot_hash"], "identity.snapshot_hash"
        ),
        "design_hash": _hash(obj["design_hash"], "identity.design_hash"),
        "strategy_spec_hash": _hash(
            obj["strategy_spec_hash"], "identity.strategy_spec_hash"
        ),
    }


def _source_bindings(
    value: object,
    *,
    identity: Mapping[str, Any],
    partitions: Sequence[str],
) -> dict[str, Any]:
    obj = _json_object(value, "source_bindings")
    _exact_fields(obj, _SOURCE_BINDING_FIELDS, "source_bindings")
    impact_ref = _impact_cube_ref(obj["impact_cube"])
    sample = _sample_design_v2(obj["sample_design_v2"], partitions=partitions)
    dataset = _dataset_binding(obj["dataset"])
    if dataset["task_id"] != identity["task_id"]:
        raise PoolStabilityError(
            "source dataset belongs to another Pool task"
        )
    return {
        "impact_cube": impact_ref,
        "sample_design_v2": sample,
        "dataset": dataset,
    }


def _impact_cube_ref(value: object) -> dict[str, str]:
    obj = _json_object(value, "impact_cube_ref")
    _exact_fields(obj, _IMPACT_CUBE_REF_FIELDS, "impact_cube_ref")
    cube_id = _text(obj["expected_cube_id"], "impact_cube_ref.expected_cube_id")
    if _CUBE_ID_RE.fullmatch(cube_id) is None:
        raise PoolStabilityError(
            "impact_cube_ref.expected_cube_id is not canonical"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"], "impact_cube_ref.artifact_id"
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "impact_cube_ref.expected_artifact_content_hash",
        ),
        "expected_cube_id": cube_id,
        "expected_cube_content_hash": _hash(
            obj["expected_cube_content_hash"],
            "impact_cube_ref.expected_cube_content_hash",
        ),
    }


def _sample_design_v2(
    value: object,
    *,
    partitions: Sequence[str],
) -> dict[str, Any]:
    obj = _json_object(value, "source_bindings.sample_design_v2")
    _exact_fields(obj, _SAMPLE_FIELDS, "source_bindings.sample_design_v2")
    result: dict[str, Any] = {}
    hash_fields = (
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "membership_content_hash",
        "bundle_artifact_id",
        "bundle_artifact_content_hash",
        "bundle_content_hash",
        "sample_design_content_hash",
    )
    text_fields = ("membership_id", "bundle_id", "sample_design_id")
    for field in hash_fields:
        result[field] = _hash(
            obj[field], f"source_bindings.sample_design_v2.{field}"
        )
    for field in text_fields:
        result[field] = _text(
            obj[field], f"source_bindings.sample_design_v2.{field}"
        )
    result["analysis_universe_row_count"] = _positive_int(
        obj["analysis_universe_row_count"],
        "source_bindings.sample_design_v2.analysis_universe_row_count",
    )
    partition_tuple = tuple(partitions)
    counts = _partition_count_map(
        obj["partition_counts"],
        name="source_bindings.sample_design_v2.partition_counts",
        partitions=partition_tuple,
    )
    populations = _json_object(
        obj["population_partition_counts"],
        "source_bindings.sample_design_v2.population_partition_counts",
    )
    _exact_fields(
        populations,
        frozenset(_POPULATION_ROLES),
        "source_bindings.sample_design_v2.population_partition_counts",
    )
    population_counts = {
        role: _partition_count_map(
            populations[role],
            name=(
                "source_bindings.sample_design_v2."
                f"population_partition_counts.{role}"
            ),
            partitions=partition_tuple,
        )
        for role in _POPULATION_ROLES
    }
    if counts != population_counts["risk"]:
        raise PoolStabilityError(
            "sample design risk partition counts changed"
        )
    for partition in partition_tuple:
        if (
            population_counts["risk"][partition]
            > population_counts["approval"][partition]
        ):
            raise PoolStabilityError(
                f"risk/{partition} exceeds approval population"
            )
    result["partition_counts"] = counts
    result["population_partition_counts"] = population_counts
    return result


def _partition_count_map(
    value: object,
    *,
    name: str,
    partitions: Sequence[str],
) -> dict[str, int]:
    obj = _json_object(value, name)
    if set(obj) != set(partitions):
        raise PoolStabilityError(
            f"{name} does not match stability partitions"
        )
    return {
        partition: _positive_int(obj[partition], f"{name}.{partition}")
        for partition in _PARTITION_ORDER
        if partition in obj
    }


def _dataset_binding(value: object) -> dict[str, Any]:
    obj = _json_object(value, "source_bindings.dataset")
    _exact_fields(obj, _DATASET_FIELDS, "source_bindings.dataset")
    return {
        "task_id": _text(obj["task_id"], "source_bindings.dataset.task_id"),
        "dataset_id": _text(
            obj["dataset_id"], "source_bindings.dataset.dataset_id"
        ),
        "dataset_content_hash": _hash(
            obj["dataset_content_hash"],
            "source_bindings.dataset.dataset_content_hash",
        ),
        "dataset_source_path": _text(
            obj["dataset_source_path"],
            "source_bindings.dataset.dataset_source_path",
        ),
        "dataset_registry_metadata_hash": _hash(
            obj["dataset_registry_metadata_hash"],
            "source_bindings.dataset.dataset_registry_metadata_hash",
        ),
        "workspace_revision": _nonnegative_int(
            obj["workspace_revision"],
            "source_bindings.dataset.workspace_revision",
        ),
        "workspace_generation": _nonnegative_int(
            obj["workspace_generation"],
            "source_bindings.dataset.workspace_generation",
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"],
            "source_bindings.dataset.semantic_mapping_hash",
        ),
    }


def _comparison_partitions(value: object) -> list[str]:
    if not _sequence(value):
        raise PoolStabilityError(
            "comparison_partitions must contain validation and/or OOT"
        )
    result = [
        _text(item, f"comparison_partitions[{index}]")
        for index, item in enumerate(value)
    ]
    expected = [
        partition
        for partition in _COMPARISON_PARTITIONS
        if partition in result
    ]
    if result != expected or len(set(result)) != len(result):
        raise PoolStabilityError(
            "comparison_partitions must be unique and ordered validation, OOT"
        )
    return result


def _populations(
    value: object,
    *,
    identity: Mapping[str, Any],
    sources: Mapping[str, Any],
    comparison_partitions: Sequence[str],
) -> list[dict[str, Any]]:
    if not _sequence(value) or len(value) != len(_POPULATION_ROLES):
        raise PoolStabilityError(
            "populations must contain separate approval and risk evidence"
        )
    result = [
        _population(
            raw,
            index=index,
            identity=identity,
            sources=sources,
            comparison_partitions=comparison_partitions,
        )
        for index, raw in enumerate(value)
    ]
    if [row["population_role"] for row in result] != list(_POPULATION_ROLES):
        raise PoolStabilityError(
            "populations must be ordered approval then risk"
        )
    waterfall_categories = [
        [
            row["category"]
            for row in population["comparisons"][0]["distributions"][0][
                "categories"
            ]
        ]
        for population in result
    ]
    if waterfall_categories[0] != waterfall_categories[1]:
        raise PoolStabilityError(
            "approval and risk waterfall category identities changed"
        )
    return result


def _population(
    value: object,
    *,
    index: int,
    identity: Mapping[str, Any],
    sources: Mapping[str, Any],
    comparison_partitions: Sequence[str],
) -> dict[str, Any]:
    name = f"populations[{index}]"
    obj = _json_object(value, name)
    _exact_fields(obj, _POPULATION_FIELDS, name)
    role = _text(obj["population_role"], f"{name}.population_role")
    if role not in _POPULATION_ROLES:
        raise PoolStabilityError(f"{name}.population_role is invalid")
    source_counts = sources["sample_design_v2"][
        "population_partition_counts"
    ][role]
    development_count = _positive_int(
        obj["development_sample_count"],
        f"{name}.development_sample_count",
    )
    if development_count != source_counts["development"]:
        raise PoolStabilityError(
            f"{name}.development_sample_count changed from sample design"
        )
    comparisons_value = obj["comparisons"]
    if (
        not _sequence(comparisons_value)
        or len(comparisons_value) != len(comparison_partitions)
    ):
        raise PoolStabilityError(
            f"{name}.comparisons do not match comparison_partitions"
        )
    comparisons = [
        _comparison(
            raw,
            name=f"{name}.comparisons[{comparison_index}]",
            expected_partition=partition,
            identity=identity,
            development_count=development_count,
            comparison_count=source_counts[partition],
        )
        for comparison_index, (raw, partition) in enumerate(
            zip(
                comparisons_value,
                comparison_partitions,
                strict=True,
            )
        )
    ]
    baseline_waterfalls = [
        [
            (row["category"], row["development_count"])
            for row in comparison["distributions"][0]["categories"]
        ]
        for comparison in comparisons
    ]
    if any(
        value != baseline_waterfalls[0]
        for value in baseline_waterfalls[1:]
    ):
        raise PoolStabilityError(
            f"{name} development waterfall changed across comparisons"
        )
    baseline_actions = [
        {
            _canonical_json(row["category"]["action"]): row[
                "development_count"
            ]
            for row in comparison["distributions"][1]["categories"]
            if row["development_count"] > 0
        }
        for comparison in comparisons
    ]
    if any(
        value != baseline_actions[0]
        for value in baseline_actions[1:]
    ):
        raise PoolStabilityError(
            f"{name} development typed actions changed across comparisons"
        )
    if obj["conservation"] != _POPULATION_CONSERVATION:
        raise PoolStabilityError(
            f"{name}.conservation checks must all pass"
        )
    return {
        "population_role": role,
        "development_sample_count": development_count,
        "comparisons": comparisons,
        "conservation": dict(_POPULATION_CONSERVATION),
    }


def _comparison(
    value: object,
    *,
    name: str,
    expected_partition: str,
    identity: Mapping[str, Any],
    development_count: int,
    comparison_count: int,
) -> dict[str, Any]:
    obj = _json_object(value, name)
    _exact_fields(obj, _COMPARISON_FIELDS, name)
    if obj["partition"] != expected_partition:
        raise PoolStabilityError(
            f"{name}.partition must remain {expected_partition}"
        )
    sample_count = _positive_int(
        obj["sample_count"], f"{name}.sample_count"
    )
    if sample_count != comparison_count:
        raise PoolStabilityError(
            f"{name}.sample_count changed from sample design"
        )
    distributions_value = obj["distributions"]
    if (
        not _sequence(distributions_value)
        or len(distributions_value) != len(_DISTRIBUTION_BASES)
    ):
        raise PoolStabilityError(
            f"{name}.distributions must contain waterfall and new_action"
        )
    distributions = [
        _distribution(
            raw,
            name=f"{name}.distributions[{index}]",
            expected_basis=basis,
            strategy_type=identity["strategy_type"],
            development_count=development_count,
            comparison_count=comparison_count,
        )
        for index, (raw, basis) in enumerate(
            zip(distributions_value, _DISTRIBUTION_BASES, strict=True)
        )
    ]
    if obj["conservation"] != _COMPARISON_CONSERVATION:
        raise PoolStabilityError(
            f"{name}.conservation checks must all pass"
        )
    return {
        "partition": expected_partition,
        "sample_count": sample_count,
        "distributions": distributions,
        "conservation": dict(_COMPARISON_CONSERVATION),
    }


def _distribution(
    value: object,
    *,
    name: str,
    expected_basis: str,
    strategy_type: str,
    development_count: int,
    comparison_count: int,
) -> dict[str, Any]:
    obj = _json_object(value, name)
    _exact_fields(obj, _DISTRIBUTION_FIELDS, name)
    if obj["basis"] != expected_basis:
        raise PoolStabilityError(f"{name}.basis must be {expected_basis}")
    development_sample_count = _positive_int(
        obj["development_sample_count"],
        f"{name}.development_sample_count",
    )
    comparison_sample_count = _positive_int(
        obj["comparison_sample_count"],
        f"{name}.comparison_sample_count",
    )
    if development_sample_count != development_count:
        raise PoolStabilityError(
            f"{name}.development_sample_count changed"
        )
    if comparison_sample_count != comparison_count:
        raise PoolStabilityError(
            f"{name}.comparison_sample_count changed"
        )
    categories_value = obj["categories"]
    if (
        not _sequence(categories_value)
        or len(categories_value) > MAX_POOL_STABILITY_CATEGORIES
    ):
        raise PoolStabilityError(
            f"{name}.categories must be a bounded non-empty array"
        )
    categories = [
        _category_row(
            raw,
            name=f"{name}.categories[{index}]",
            basis=expected_basis,
            strategy_type=strategy_type,
            development_count=development_count,
            comparison_count=comparison_count,
        )
        for index, raw in enumerate(categories_value)
    ]
    tokens = [_canonical_json(row["category"]) for row in categories]
    if len(tokens) != len(set(tokens)):
        raise PoolStabilityError(f"{name}.categories are duplicated")
    _require_category_order(
        categories,
        basis=expected_basis,
        name=name,
    )
    if sum(row["development_count"] for row in categories) != development_count:
        raise PoolStabilityError(
            f"{name} development counts do not conserve"
        )
    if sum(row["comparison_count"] for row in categories) != comparison_count:
        raise PoolStabilityError(
            f"{name} comparison counts do not conserve"
        )
    expected_distribution = np.asarray(
        [row["development_share"] for row in categories],
        dtype=float,
    )
    comparison_distribution = np.asarray(
        [row["comparison_share"] for row in categories],
        dtype=float,
    )
    if not math.isclose(
        float(expected_distribution.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(comparison_distribution.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PoolStabilityError(f"{name} shares do not conserve")
    psi = _nonnegative_number(obj["psi"], f"{name}.psi")
    expected_psi = float(
        compute_psi(expected_distribution, comparison_distribution)
    )
    _same_number(psi, expected_psi, f"{name}.psi")
    max_delta = _nonnegative_number(
        obj["max_abs_share_delta"],
        f"{name}.max_abs_share_delta",
    )
    expected_max_delta = max(
        abs(float(row["share_delta"])) for row in categories
    )
    _same_number(
        max_delta,
        expected_max_delta,
        f"{name}.max_abs_share_delta",
    )
    severity = _text(obj["severity"], f"{name}.severity")
    if severity != _severity(psi):
        raise PoolStabilityError(
            f"{name}.severity does not match PSI thresholds"
        )
    if obj["conservation"] != _DISTRIBUTION_CONSERVATION:
        raise PoolStabilityError(
            f"{name}.conservation checks must all pass"
        )
    return {
        "basis": expected_basis,
        "development_sample_count": development_sample_count,
        "comparison_sample_count": comparison_sample_count,
        "categories": categories,
        "psi": psi,
        "max_abs_share_delta": max_delta,
        "severity": severity,
        "conservation": dict(_DISTRIBUTION_CONSERVATION),
    }


def _category_row(
    value: object,
    *,
    name: str,
    basis: str,
    strategy_type: str,
    development_count: int,
    comparison_count: int,
) -> dict[str, Any]:
    obj = _json_object(value, name)
    _exact_fields(obj, _CATEGORY_ROW_FIELDS, name)
    category = _category(
        obj["category"],
        name=f"{name}.category",
        basis=basis,
        strategy_type=strategy_type,
    )
    baseline = _count(
        obj["development_count"],
        f"{name}.development_count",
        development_count,
    )
    target = _count(
        obj["comparison_count"],
        f"{name}.comparison_count",
        comparison_count,
    )
    development_share = _rate(
        obj["development_share"],
        f"{name}.development_share",
    )
    comparison_share = _rate(
        obj["comparison_share"],
        f"{name}.comparison_share",
    )
    share_delta = _number(obj["share_delta"], f"{name}.share_delta")
    _same_number(
        development_share,
        _ratio(baseline, development_count),
        f"{name}.development_share",
    )
    _same_number(
        comparison_share,
        _ratio(target, comparison_count),
        f"{name}.comparison_share",
    )
    _same_number(
        share_delta,
        comparison_share - development_share,
        f"{name}.share_delta",
    )
    return {
        "category": category,
        "development_count": baseline,
        "comparison_count": target,
        "development_share": development_share,
        "comparison_share": comparison_share,
        "share_delta": share_delta,
    }


def _category(
    value: object,
    *,
    name: str,
    basis: str,
    strategy_type: str,
) -> dict[str, Any]:
    obj = _json_object(value, name)
    if basis == "waterfall_incremental":
        _exact_fields(obj, _WATERFALL_CATEGORY_FIELDS, name)
        kind = obj["kind"]
        if kind == "pool_entry_incremental":
            return {
                "kind": kind,
                "position": _positive_int(
                    obj["position"], f"{name}.position"
                ),
                "entry_id": _text(obj["entry_id"], f"{name}.entry_id"),
                "rule_id": _text(obj["rule_id"], f"{name}.rule_id"),
            }
        if kind == "default_unmatched":
            if any(
                obj[field] is not None
                for field in ("position", "entry_id", "rule_id")
            ):
                raise PoolStabilityError(
                    f"{name} default_unmatched identity must be null"
                )
            return {
                "kind": kind,
                "position": None,
                "entry_id": None,
                "rule_id": None,
            }
        raise PoolStabilityError(f"{name}.kind is invalid")
    _exact_fields(obj, _ACTION_CATEGORY_FIELDS, name)
    if obj["kind"] != "typed_action":
        raise PoolStabilityError(f"{name}.kind must be typed_action")
    action_obj = _json_object(obj["action"], f"{name}.action")
    try:
        action = StrategyAction.from_dict(action_obj)
    except StrategyError as exc:
        raise PoolStabilityError(f"{name}.action is invalid: {exc}") from exc
    normalized = action.to_dict()
    if normalized != action_obj:
        raise PoolStabilityError(f"{name}.action is not canonical")
    if action.type not in _STRATEGY_ACTION_TYPES[strategy_type]:
        raise PoolStabilityError(
            f"{name}.action is invalid for {strategy_type}"
        )
    return {"kind": "typed_action", "action": normalized}


def _require_category_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    basis: str,
    name: str,
) -> None:
    categories = [row["category"] for row in rows]
    if basis == "new_action":
        tokens = [
            _canonical_json(category["action"]) for category in categories
        ]
        if tokens != sorted(tokens):
            raise PoolStabilityError(
                f"{name}.categories typed actions are not canonical"
            )
        return
    defaults = [
        index
        for index, category in enumerate(categories)
        if category["kind"] == "default_unmatched"
    ]
    if defaults != [len(categories) - 1]:
        raise PoolStabilityError(
            f"{name}.categories must end with one default_unmatched bucket"
        )
    entries = categories[:-1]
    positions = [int(category["position"]) for category in entries]
    if positions != list(range(1, len(entries) + 1)):
        raise PoolStabilityError(
            f"{name}.categories Pool entry positions are not canonical"
        )


def _severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "warning"
    return "material"


def _positive_int(value: object, name: str) -> int:
    normalized = _nonnegative_int(value, name)
    if normalized < 1:
        raise PoolStabilityError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_int(value: object, name: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 0
        or int(value) > _MAX_SAFE_JSON_INTEGER
    ):
        raise PoolStabilityError(
            f"{name} must be a non-negative JSON-safe integer"
        )
    return int(value)


def _count(value: object, name: str, maximum: int) -> int:
    normalized = _nonnegative_int(value, name)
    if normalized > maximum:
        raise PoolStabilityError(f"{name} exceeds its population")
    return normalized


def _number(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise PoolStabilityError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PoolStabilityError(f"{name} must be a finite number")
    return normalized


def _nonnegative_number(value: object, name: str) -> float:
    normalized = _number(value, name)
    if normalized < 0:
        raise PoolStabilityError(f"{name} must be non-negative")
    return normalized


def _rate(value: object, name: str) -> float:
    normalized = _number(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise PoolStabilityError(f"{name} must be between 0 and 1")
    return normalized


def _same_number(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise PoolStabilityError(f"{name} does not match derived evidence")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
    ):
        raise PoolStabilityError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise PoolStabilityError(f"{name} must be a lowercase SHA-256")
    return normalized


def _sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and len(value) > 0
    )


def _json_object(value: object, name: str) -> dict[str, Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise PoolStabilityError(f"{name} must be an object")
    return normalized


def _json_value(value: object, name: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        normalized = int(value)
        if abs(normalized) > _MAX_SAFE_JSON_INTEGER:
            raise PoolStabilityError(
                f"{name} must contain JSON-safe integers"
            )
        return normalized
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise PoolStabilityError(f"{name} must contain finite JSON")
        return normalized
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PoolStabilityError(f"{name} keys must be strings")
        return {
            key: _json_value(child, f"{name}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [
            _json_value(child, f"{name}[{index}]")
            for index, child in enumerate(value)
        ]
    raise PoolStabilityError(f"{name} must contain canonical JSON values")


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields " + ", ".join(unexpected))
        raise PoolStabilityError(f"{name} has " + "; ".join(details))


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _json_value(value, "Pool stability"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise PoolStabilityError(
            "Pool stability must contain finite canonical JSON"
        ) from exc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "MAX_POOL_STABILITY_CATEGORIES",
    "MAX_POOL_STABILITY_JSON_BYTES",
    "POOL_STABILITY_PRODUCER_VERSION",
    "POOL_STABILITY_SCHEMA_VERSION",
    "PoolStabilityError",
    "build_strategy_pool_stability",
    "canonical_strategy_pool_stability_json",
    "strategy_pool_stability_content_hash",
    "validate_strategy_pool_stability",
]
