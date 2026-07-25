"""Unified deterministic ImpactSlice and ImpactCube evidence.

The module is persistence-free.  Callers provide exact, already-authenticated
Pool/sample/dataset bindings and partition frames.  The core replays the
compiled Pool for all five Strategy DSL types and emits aggregate-only slices
for partition, month, external group, external segment, group-by-month,
segment-by-month, and the new strategy's typed action buckets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.dsl import (
    StrategyAction,
    StrategySpec,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import (
    evaluate_expression_frame,
    evaluate_strategy_frame,
)
from marvis.packs.strategy.pool import compile_strategy_pool, validate_strategy_pool
from marvis.packs.strategy.profit import ProfitParams
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef
from marvis.packs.strategy.typed_backtest import (
    ApprovalProfitInputs,
    StrategyBacktestResult,
    run_typed_backtest,
)
from marvis.validation.time_periods import month_key_series


STRATEGY_IMPACT_SLICE_SCHEMA_VERSION = "strategy.impact-slice.v2"
STRATEGY_IMPACT_CUBE_SCHEMA_VERSION = "strategy.impact-cube.v2"
STRATEGY_IMPACT_CUBE_PRODUCER_VERSION = "marvis.strategy.impact-cube/2"

MAX_IMPACT_CUBE_ROWS = 2_000_000
MAX_IMPACT_CUBE_RULES = 200
MAX_IMPACT_CUBE_SLICES = 512
MAX_IMPACT_CUBE_WORK = 50_000_000
MAX_IMPACT_CUBE_JSON_BYTES = 64 * 1024 * 1024
MAX_IMPACT_CUBE_DIMENSION_VALUES = 512
MIN_IMPACT_CUBE_GROUP_SIZE = 2
_HIGH_CARDINALITY_MIN_ROWS = 20
_FORBIDDEN_DIMENSION_ROLES = frozenset(
    {
        "id",
        "identifier",
        "idcard",
        "name",
        "phone",
        "personal",
        "pii",
        "sensitive",
        "ignore",
    }
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CUBE_ID_RE = re.compile(r"^strategy-impact-cube-[0-9a-f]{24}$")
_SLICE_ID_RE = re.compile(r"^strategy-impact-slice-[0-9a-f]{24}$")
_MONTH_RE = re.compile(r"^[0-9]{6}$")
_PARTITION_ORDER = ("development", "validation", "oot")
_PARTITIONS = frozenset(_PARTITION_ORDER)
_POPULATION_ROLE_ORDER = ("approval", "risk")
_POPULATION_ROLES = frozenset(_POPULATION_ROLE_ORDER)
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
_FAMILY_ORDER = (
    "overall",
    "month",
    "group",
    "segment",
    "group_month",
    "segment_month",
    "new_action",
)
_FAMILIES = frozenset(_FAMILY_ORDER)
_FAMILY_DIMENSIONS = {
    "overall": frozenset(),
    "month": frozenset({"month"}),
    "group": frozenset({"group"}),
    "segment": frozenset({"segment"}),
    "group_month": frozenset({"group", "month"}),
    "segment_month": frozenset({"segment", "month"}),
    "new_action": frozenset({"new_action_bucket"}),
}
_DIMENSION_FIELD_NAMES = {
    "month": "month_col",
    "group": "group_col",
    "segment": "segment_col",
}
_ACTION_ORDER = ("approve", "reject", "review")
_ACTION_METRIC_FIELDS = frozenset(
    {
        "overall_bad_count",
        "overall_bad_rate",
        *(
            f"{action}_{suffix}"
            for action in _ACTION_ORDER
            for suffix in (
                "count",
                "rate",
                "labeled_count",
                "bad_count",
                "bad_rate",
            )
        ),
    }
)
_ACTION_BREAKDOWN_FIELDS = frozenset(
    {
        "action",
        "count",
        "rate",
        "labeled_count",
        "bad_count",
        "bad_rate",
    }
)
_LIMIT_METRIC_FIELDS = frozenset(
    {
        "count",
        "min_limit",
        "max_limit",
        "mean_limit",
        "total_limit",
        "up_count",
        "down_count",
        "unchanged_count",
        "total_limit_delta",
    }
)
_LIMIT_BREAKDOWN_FIELDS = frozenset(
    {
        "assigned_limit",
        "count",
        "share",
        "labeled_count",
        "bad_count",
        "bad_rate",
    }
)
_PRICING_METRIC_FIELDS = frozenset(
    {
        "count",
        "mean_rate",
        "repriced_up_count",
        "repriced_down_count",
        "unchanged_count",
    }
)
_PRICING_BREAKDOWN_FIELDS = frozenset(
    {
        "assigned_rate",
        "count",
        "share",
        "labeled_count",
        "bad_count",
        "bad_rate",
    }
)
_SEGMENTATION_METRIC_FIELDS = frozenset(
    {"segment_count", "overall_bad_count", "overall_bad_rate"}
)
_SEGMENTATION_BREAKDOWN_FIELDS = frozenset(
    {
        "segment",
        "count",
        "share",
        "labeled_count",
        "bad_count",
        "bad_rate",
        "lift",
    }
)
_ECONOMICS_FIELDS = {
    "approval": frozenset(
        {
            "baseline_profit",
            "ead_weighted_rate",
            "expected_loss",
            "expected_profit",
            "funding_cost",
            "operating_cost",
            "profit",
            "profit_delta_vs_baseline",
            "profit_note",
            "revenue",
            "roa",
            "total_ead",
        }
    ),
    "reject": frozenset(
        {
            "baseline_profit",
            "ead_weighted_rate",
            "expected_loss",
            "expected_profit",
            "funding_cost",
            "operating_cost",
            "profit",
            "profit_delta_vs_baseline",
            "profit_note",
            "revenue",
            "roa",
            "total_ead",
        }
    ),
    "limit": frozenset({"expected_ead", "expected_loss"}),
    "pricing": frozenset(
        {
            "baseline_profit",
            "ead_weighted_rate",
            "expected_loss",
            "funding_cost",
            "operating_cost",
            "profit",
            "profit_delta_vs_baseline",
            "revenue",
            "roa",
            "total_ead",
        }
    ),
    "segmentation": frozenset(),
}
_ECONOMIC_COMPONENTS = {
    "approval": frozenset(
        {
            "ead",
            "pd",
            "annual_rate",
            "funding_rate",
            "lgd",
            "operating_cost_per_loan",
            "term_months",
        }
    ),
    "reject": frozenset(
        {
            "ead",
            "pd",
            "annual_rate",
            "funding_rate",
            "lgd",
            "operating_cost_per_loan",
            "term_months",
        }
    ),
    "limit": frozenset({"pd", "lgd", "utilization"}),
    "pricing": frozenset(
        {
            "ead",
            "pd",
            "lgd",
            "funding_rate",
            "term_months",
            "operating_cost_per_loan",
        }
    ),
    "segmentation": frozenset(),
}
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "cube_id",
        "identity",
        "source_bindings",
        "partitions",
        "slice_families",
        "slices",
        "lifecycle",
        "conservation",
        "red_flags",
        "content_hash",
    }
)
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
_SOURCE_FIELDS = frozenset(
    {
        "pool_artifact",
        "sample_design_v2",
        "dataset",
        "development_lineage",
        "target",
        "fields",
        "current_strategy",
        "economics",
        "privacy",
    }
)
_POOL_ARTIFACT_FIELDS = frozenset({"artifact_id", "artifact_content_hash"})
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
_TARGET_FIELDS = frozenset(
    {"column", "good_value", "bad_value", "missing_policy"}
)
_FIELD_FIELDS = frozenset(
    {
        "month_col",
        "group_col",
        "segment_col",
        "loan_amount_col",
        "overdue_amount_col",
    }
)
_PRIVACY_FIELDS = frozenset(
    {
        "policy_version",
        "minimum_group_size",
        "forbidden_semantic_roles",
        "bound_dimension_roles",
        "redacted_bucket_kind",
        "policy_hash",
    }
)
_CURRENT_REF_FIELDS = frozenset(
    {"strategy_id", "strategy_type", "strategy_spec_hash"}
)
_DEVELOPMENT_LINEAGE_FIELDS = frozenset(
    {"legacy_development_ref", "sample_binding"}
)
_SAMPLE_BINDING_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
    }
)
_PARTITION_FIELDS = frozenset(
    {
        "name",
        "role",
        "population_key",
        "row_count",
        "effect_stage",
        "validation_status",
    }
)
_SLICE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "slice_id",
        "family",
        "population_role",
        "availability",
        "unavailable_reason",
        "dimensions",
        "population",
        "new",
        "current",
        "transition",
        "waterfall",
        "economics",
        "conservation",
        "content_hash",
    }
)
_DIMENSIONS = frozenset(
    {"partition", "month", "group", "segment", "new_action_bucket"}
)
_TYPED_FIELD_FIELDS = frozenset({"availability", "reason", "value"})
_POPULATION_FIELDS = frozenset(
    {
        "count",
        "role",
        "labeled_count",
        "unlabeled_count",
        "label_coverage",
        "risk",
        "amounts",
    }
)
_RISK_FIELDS = frozenset(
    {"availability", "reason", "bad_count", "bad_rate"}
)
_PROJECTION_FIELDS = frozenset(
    {
        "strategy_id",
        "strategy_type",
        "population_role",
        "population_count",
        "labeled_count",
        "label_coverage",
        "metrics",
        "breakdown",
        "warnings",
    }
)
_CONSERVATION_FIELDS = frozenset(
    {
        "waterfall_incremental_plus_default_equals_population",
        "waterfall_standalone_equals_incremental_plus_shadowed",
        "transition_equals_population",
    }
)


@dataclass(frozen=True)
class _DimensionGroups:
    """Compact row-to-dimension mapping shared by all descriptors in a family."""

    values: tuple[dict[str, Any], ...]
    codes: np.ndarray


def build_strategy_impact_cube(
    *,
    pool: Mapping[str, Any],
    partition_frames: Mapping[str, pd.DataFrame],
    approval_partition_frames: Mapping[str, pd.DataFrame] | None = None,
    pool_artifact_ref: Mapping[str, Any],
    sample_design_v2_ref: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    legacy_development_ref: Mapping[str, Any],
    target_col: str,
    target_bad_value: int,
    month_col: str | None,
    group_col: str | None,
    segment_col: str | None,
    current_strategy_spec: Mapping[str, Any] | StrategySpec | None,
    current_strategy_ref: Mapping[str, Any] | None,
    economics_bindings: Mapping[str, Mapping[str, Any]] | None,
    semantic_field_roles: Mapping[str, str] | None = None,
    entity_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
) -> dict[str, Any]:
    """Build a canonical aggregate-only ImpactCube for exact partition rows."""

    current_pool = validate_strategy_pool(pool)
    if current_pool["strategy_type"] not in _STRATEGY_TYPES:
        raise StrategyError("ImpactCube strategy_type is unsupported")
    if not current_pool["entries"]:
        raise StrategyError("cannot build ImpactCube from an empty Strategy Pool")
    if len(current_pool["entries"]) > MAX_IMPACT_CUBE_RULES:
        raise StrategyError("ImpactCube rule budget exceeded")
    # The governed Tool boundary authenticates compiled requirements and
    # hydrates their virtual fields on the full row universe before slicing.
    # This pure kernel evaluates only the already materialized frames.
    compiled = compile_strategy_pool(current_pool)
    new_spec = parse_strategy_spec(compiled["strategy_spec"])

    risk_frames = _partition_frames(partition_frames)
    approval_frames = (
        {name: frame.copy() for name, frame in risk_frames.items()}
        if approval_partition_frames is None
        else _partition_frames(approval_partition_frames)
    )
    if tuple(approval_frames) != tuple(risk_frames):
        raise StrategyError(
            "ImpactCube approval and risk partitions must match"
        )
    population_frames = {
        "approval": approval_frames,
        "risk": risk_frames,
    }
    pool_artifact = _pool_artifact_ref(pool_artifact_ref)
    sample_v2 = _sample_design_v2_ref(
        sample_design_v2_ref,
        partitions=tuple(risk_frames),
    )
    dataset = _dataset_binding(dataset_binding)
    if dataset["task_id"] != current_pool["task_id"]:
        raise StrategyError("ImpactCube dataset belongs to another task")
    if (
        sum(
            len(frame)
            for frames in population_frames.values()
            for frame in frames.values()
        )
        > MAX_IMPACT_CUBE_ROWS
    ):
        raise StrategyError("ImpactCube row budget exceeded")
    for role, frames in population_frames.items():
        counts = sample_v2["population_partition_counts"][role]
        for name, frame in frames.items():
            if len(frame) != counts[name]:
                raise StrategyError(
                    f"ImpactCube {role}/{name} rows do not match "
                    "membership count"
                )
            if len(frame) < MIN_IMPACT_CUBE_GROUP_SIZE:
                raise StrategyError(
                    f"ImpactCube {role}/{name} is below the governed "
                    "minimum group size"
                )

    legacy_ref = StrategySampleDesignRef.from_value(
        legacy_development_ref
    ).to_ref_dict()
    sample_binding = _pool_sample_binding(current_pool)
    for field in (
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    ):
        if sample_binding[field] != dataset[field]:
            raise StrategyError(
                f"ImpactCube Pool lineage {field} does not match V2 dataset"
            )

    target = _target_binding(target_col, target_bad_value)
    fields = _field_bindings(
        month_col=month_col,
        group_col=group_col,
        segment_col=segment_col,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
        target_col=target["column"],
    )
    privacy = _privacy_binding(
        fields=fields,
        semantic_field_roles=semantic_field_roles,
        entity_col=entity_col,
    )
    current_spec, current_ref = _current_strategy(
        current_strategy_spec,
        current_strategy_ref,
        strategy_type=current_pool["strategy_type"],
    )
    economics = _economics_binding(
        economics_bindings,
        strategy_type=current_pool["strategy_type"],
    )

    required_columns = _strategy_fields(new_spec)
    if current_spec is not None:
        required_columns.update(_strategy_fields(current_spec))
    required_columns.add(target["column"])
    required_columns.update(
        column
        for column in (
            fields["month_col"],
            fields["group_col"],
            fields["segment_col"],
            fields["loan_amount_col"],
            fields["overdue_amount_col"],
        )
        if column is not None
    )
    required_columns.update(
        binding["column"]
        for binding in economics["bindings"].values()
        if binding["kind"] == "column"
    )
    for role, frames in population_frames.items():
        for name, frame in frames.items():
            missing = sorted(required_columns - set(frame.columns))
            if missing:
                raise StrategyError(
                    f"ImpactCube {role}/{name} partition is missing columns: "
                    + ", ".join(missing)
                )

    family_status = _slice_family_status(fields)
    present_family_count = sum(
        status["availability"] == "present"
        for status in family_status.values()
    )
    evaluation_multiplier = (
        len(new_spec.rules)
        + (0 if current_spec is None else len(current_spec.rules))
        + 2
    )
    work = (
        sum(
            len(frame)
            for frames in population_frames.values()
            for frame in frames.values()
        )
        * present_family_count
        * evaluation_multiplier
    )
    if work > MAX_IMPACT_CUBE_WORK:
        raise StrategyError("ImpactCube evaluation work budget exceeded")

    descriptors: list[dict[str, Any]] = []
    for role in _POPULATION_ROLE_ORDER:
        for partition in _PARTITION_ORDER:
            frame = population_frames[role].get(partition)
            if frame is None:
                continue
            partition_descriptors = _slice_descriptors(
                population_role=role,
                partition=partition,
                frame=frame,
                new_spec=new_spec,
                fields=fields,
                max_slices=MAX_IMPACT_CUBE_SLICES - len(descriptors),
            )
            descriptors.extend(partition_descriptors)
    descriptors.sort(key=_slice_sort_key)
    if len(descriptors) > MAX_IMPACT_CUBE_SLICES:
        raise StrategyError("ImpactCube slice budget exceeded")

    slices: list[dict[str, Any]] = []
    for descriptor in descriptors:
        role = descriptor["population_role"]
        frame = population_frames[role][descriptor["partition"]]
        if descriptor["availability"] == "unavailable":
            slices.append(
                _build_unavailable_slice(
                    population_role=role,
                    partition=descriptor["partition"],
                    family=descriptor["family"],
                    dimensions=descriptor["dimensions"],
                    reason=descriptor["reason"],
                )
            )
            continue
        selector = descriptor["selector"]
        if selector is None:
            selected = frame.reset_index(drop=True)
        else:
            positions = np.flatnonzero(
                selector["codes"] == selector["code"]
            )
            selected = frame.iloc[positions].reset_index(drop=True)
        if selected.empty:
            raise StrategyError("ImpactCube cannot persist empty present slices")
        slices.append(
            _build_present_slice(
                population_role=role,
                partition=descriptor["partition"],
                family=descriptor["family"],
                dimensions=descriptor["dimensions"],
                frame=selected,
                pool=current_pool,
                new_spec=new_spec,
                current_spec=current_spec,
                current_ref=current_ref,
                target=target,
                economics=economics,
                loan_amount_col=fields["loan_amount_col"],
                overdue_amount_col=fields["overdue_amount_col"],
            )
        )

    partition_rows = [
        {
            "role": role,
            "name": partition,
            "population_key": f"{role}/{partition}",
            "row_count": len(population_frames[role][partition]),
            **_partition_stage(partition),
        }
        for role in _POPULATION_ROLE_ORDER
        for partition in _PARTITION_ORDER
        if partition in population_frames[role]
    ]
    source_bindings = {
        "pool_artifact": pool_artifact,
        "sample_design_v2": sample_v2,
        "dataset": dataset,
        "development_lineage": {
            "legacy_development_ref": legacy_ref,
            "sample_binding": sample_binding,
        },
        "target": target,
        "fields": fields,
        "current_strategy": _typed_field(
            "unavailable" if current_ref is None else "present",
            current_ref,
            (
                "current_strategy_not_bound"
                if current_ref is None
                else None
            ),
        ),
        "economics": economics,
        "privacy": privacy,
    }
    identity = {
        "pool_id": current_pool["pool_id"],
        "task_id": current_pool["task_id"],
        "strategy_type": current_pool["strategy_type"],
        "revision": current_pool["revision"],
        "revision_id": current_pool["revision_id"],
        "snapshot_hash": current_pool["snapshot_hash"],
        "design_hash": compiled["design_hash"],
        "strategy_spec_hash": strategy_spec_hash(new_spec),
    }
    red_flags = _red_flags(
        family_status=family_status,
        current_ref=current_ref,
        economics=economics,
        slices=slices,
    )
    body = {
        "schema_version": STRATEGY_IMPACT_CUBE_SCHEMA_VERSION,
        "producer_version": STRATEGY_IMPACT_CUBE_PRODUCER_VERSION,
        "identity": identity,
        "source_bindings": source_bindings,
        "partitions": partition_rows,
        "slice_families": family_status,
        "slices": slices,
        "lifecycle": {
            "mutates_pool": False,
            "creates_strategy": False,
            "adopts_strategy": False,
            "promotes_strategy": False,
            "deploys_strategy": False,
        },
        "conservation": {
            "slice_family_rollups_equal_partition": True,
            "partition_counts_match_membership": True,
            "all_slice_documents_valid": True,
        },
        "red_flags": red_flags,
    }
    cube_id = "strategy-impact-cube-" + _sha256(_canonical_json(body))[:24]
    document = {**body, "cube_id": cube_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    canonical = _canonical_json(document)
    if len(canonical.encode("utf-8")) > MAX_IMPACT_CUBE_JSON_BYTES:
        raise StrategyError("ImpactCube JSON byte budget exceeded")
    return validate_strategy_impact_cube(document)


def validate_strategy_impact_cube(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate hashes, strict schema, typed availability, and conservation."""

    obj = _json_object(payload, "ImpactCube")
    _exact_fields(obj, _TOP_LEVEL_FIELDS, "ImpactCube")
    if obj["schema_version"] != STRATEGY_IMPACT_CUBE_SCHEMA_VERSION:
        raise StrategyError("ImpactCube schema_version is invalid")
    if obj["producer_version"] != STRATEGY_IMPACT_CUBE_PRODUCER_VERSION:
        raise StrategyError("ImpactCube producer_version is invalid")
    cube_id = _text(obj["cube_id"], "ImpactCube cube_id")
    if _CUBE_ID_RE.fullmatch(cube_id) is None:
        raise StrategyError("ImpactCube cube_id is invalid")
    content_hash = _hash(obj["content_hash"], "ImpactCube content_hash")
    without_hash = {
        key: value for key, value in obj.items() if key != "content_hash"
    }
    if not hmac.compare_digest(
        content_hash,
        _sha256(_canonical_json(without_hash)),
    ):
        raise StrategyError("ImpactCube content_hash does not match content")
    body = {
        key: value
        for key, value in obj.items()
        if key not in {"cube_id", "content_hash"}
    }
    expected_id = "strategy-impact-cube-" + _sha256(_canonical_json(body))[:24]
    if not hmac.compare_digest(cube_id, expected_id):
        raise StrategyError("ImpactCube cube_id does not match content")

    identity = _json_object(obj["identity"], "ImpactCube identity")
    _exact_fields(identity, _IDENTITY_FIELDS, "ImpactCube identity")
    if _text(identity["strategy_type"], "ImpactCube strategy_type") not in _STRATEGY_TYPES:
        raise StrategyError("ImpactCube strategy_type is invalid")
    _positive_int(identity["revision"], "ImpactCube revision")
    for field in ("snapshot_hash", "design_hash", "strategy_spec_hash"):
        _hash(identity[field], f"ImpactCube identity.{field}")
    for field in ("pool_id", "task_id", "revision_id"):
        _text(identity[field], f"ImpactCube identity.{field}")

    partitions = _validate_partitions(obj["partitions"])
    if [(row["role"], row["name"]) for row in partitions] != [
        (role, name)
        for role in _POPULATION_ROLE_ORDER
        for name in _PARTITION_ORDER
        if (role, name)
        in {(row["role"], row["name"]) for row in partitions}
    ]:
        raise StrategyError("ImpactCube partitions are not canonically ordered")
    selected_partition_names = tuple(
        row["name"] for row in partitions if row["role"] == "risk"
    )
    sources = _validate_source_bindings(
        obj["source_bindings"],
        partitions=selected_partition_names,
        strategy_type=identity["strategy_type"],
    )
    if sources["dataset"]["task_id"] != identity["task_id"]:
        raise StrategyError("ImpactCube source task does not match identity")
    sample_binding = sources["development_lineage"]["sample_binding"]
    if sample_binding["task_id"] != identity["task_id"]:
        raise StrategyError("ImpactCube lineage task does not match identity")
    for field in (
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    ):
        if sample_binding[field] != sources["dataset"][field]:
            raise StrategyError(
                f"ImpactCube lineage {field} does not match dataset binding"
            )
    if set(sources["sample_design_v2"]["partition_counts"]) != set(
        selected_partition_names
    ):
        raise StrategyError("ImpactCube partition bindings changed")
    for row in partitions:
        if (
            sources["sample_design_v2"]["population_partition_counts"][
                row["role"]
            ][row["name"]]
            != row["row_count"]
        ):
            raise StrategyError("ImpactCube partition population changed")

    families = _validate_family_status(obj["slice_families"])
    if families != _slice_family_status(sources["fields"]):
        raise StrategyError("ImpactCube slice families changed from bindings")
    raw_slices = _list(obj["slices"], "ImpactCube slices")
    if not raw_slices or len(raw_slices) > MAX_IMPACT_CUBE_SLICES:
        raise StrategyError("ImpactCube slice budget is invalid")
    slices = [
        _validate_slice(
            row,
            strategy_type=identity["strategy_type"],
        )
        for row in raw_slices
    ]
    for row in slices:
        _validate_slice_bindings(
            row,
            identity=identity,
            family_status=families[row["family"]],
            current=sources["current_strategy"],
            economics=sources["economics"],
            fields=sources["fields"],
        )
    declared_partitions = {
        (row["role"], row["name"]) for row in partitions
    }
    dimension_keys: set[tuple[str, str, str, str]] = set()
    for row in slices:
        partition = row["dimensions"]["partition"]["value"]
        if (row["population_role"], partition) not in declared_partitions:
            raise StrategyError(
                "ImpactCube slice uses an undeclared partition"
            )
        key = (
            row["population_role"],
            partition,
            row["family"],
            _canonical_json(row["dimensions"]),
        )
        if key in dimension_keys:
            raise StrategyError(
                "ImpactCube slice dimension buckets are duplicated"
            )
        dimension_keys.add(key)
    ids = [row["slice_id"] for row in slices]
    if len(ids) != len(set(ids)):
        raise StrategyError("ImpactCube slice ids are not unique")
    if slices != sorted(slices, key=_slice_sort_key):
        raise StrategyError("ImpactCube slices are not canonically ordered")
    _validate_rollups(
        partitions=partitions,
        families=families,
        slices=slices,
    )
    _validate_new_action_consistency(slices=slices)

    lifecycle = _json_object(obj["lifecycle"], "ImpactCube lifecycle")
    _exact_fields(
        lifecycle,
        frozenset(
            {
                "mutates_pool",
                "creates_strategy",
                "adopts_strategy",
                "promotes_strategy",
                "deploys_strategy",
            }
        ),
        "ImpactCube lifecycle",
    )
    if any(value is not False for value in lifecycle.values()):
        raise StrategyError("ImpactCube lifecycle cannot mutate strategy state")
    if obj["conservation"] != {
        "slice_family_rollups_equal_partition": True,
        "partition_counts_match_membership": True,
        "all_slice_documents_valid": True,
    }:
        raise StrategyError("ImpactCube conservation flags are invalid")
    red_flags = _list(obj["red_flags"], "ImpactCube red_flags")
    for index, flag in enumerate(red_flags):
        item = _json_object(flag, f"ImpactCube red_flags[{index}]")
        _exact_fields(
            item,
            frozenset({"code", "level", "message"}),
            f"ImpactCube red_flags[{index}]",
        )
        _text(item["code"], "ImpactCube red flag code")
        if item["level"] not in {"info", "amber", "red"}:
            raise StrategyError("ImpactCube red flag level is invalid")
        _text(item["message"], "ImpactCube red flag message")
    expected_red_flags = _red_flags(
        family_status=families,
        current_ref=(
            sources["current_strategy"]["value"]
            if sources["current_strategy"]["availability"] == "present"
            else None
        ),
        economics=sources["economics"],
        slices=slices,
    )
    if red_flags != expected_red_flags:
        raise StrategyError("ImpactCube red_flags changed from evidence")
    if len(_canonical_json(obj).encode("utf-8")) > MAX_IMPACT_CUBE_JSON_BYTES:
        raise StrategyError("ImpactCube JSON byte budget exceeded")
    return obj


def canonical_strategy_impact_cube_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON after full validation."""

    return _canonical_json(validate_strategy_impact_cube(payload))


def _build_present_slice(
    *,
    population_role: str,
    partition: str,
    family: str,
    dimensions: Mapping[str, Any],
    frame: pd.DataFrame,
    pool: Mapping[str, Any],
    new_spec: StrategySpec,
    current_spec: StrategySpec | None,
    current_ref: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    economics: Mapping[str, Any],
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> dict[str, Any]:
    analysis_frame = frame
    analysis_target = dict(target)
    if population_role == "approval":
        reserved = "__marvis_impact_cube_approval_target__"
        while reserved in analysis_frame.columns:
            reserved += "_"
        analysis_frame = frame.copy()
        analysis_frame[reserved] = np.nan
        analysis_target["column"] = reserved
    slice_economics = (
        economics
        if population_role == "approval"
        else {
            "availability": "not_applicable",
            "reason": "risk_population_economics_not_applicable",
            "bindings": {},
        }
    )
    economics_args = _economics_arguments(
        analysis_frame,
        strategy_type=new_spec.strategy_type,
        economics=slice_economics,
    )
    new_result = run_typed_backtest(
        analysis_frame,
        new_spec,
        target_col=analysis_target["column"],
        target_bad_value=analysis_target["bad_value"],
        strategy_id=f"pool-design-{strategy_spec_hash(new_spec)[:24]}",
        baseline=current_spec,
        economics_inputs=economics_args["economics_inputs"],
        approval_profit_inputs=economics_args["approval_profit_inputs"],
    )
    current_result = (
        None
        if current_spec is None
        else run_typed_backtest(
            analysis_frame,
            current_spec,
            target_col=analysis_target["column"],
            target_bad_value=analysis_target["bad_value"],
            strategy_id=current_ref["strategy_id"],
            economics_inputs=economics_args["economics_inputs"],
            approval_profit_inputs=economics_args["approval_profit_inputs"],
        )
    )
    normalized_target = _normalized_target(
        analysis_frame[analysis_target["column"]],
        bad_value=analysis_target["bad_value"],
    )
    amounts = _amount_series(
        analysis_frame,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
    )
    population = _population_summary(
        pd.Series(True, index=analysis_frame.index, dtype=bool),
        target=normalized_target,
        denominator=len(analysis_frame),
        population_role=population_role,
        amounts=amounts,
    )
    transition = _transition_field(
        frame=analysis_frame,
        target=normalized_target,
        new_spec=new_spec,
        current_spec=current_spec,
        population_role=population_role,
        amounts=amounts,
    )
    waterfall = _typed_field(
        "present",
        _waterfall(
            frame=analysis_frame,
            target=normalized_target,
            pool=pool,
            spec=new_spec,
            population_role=population_role,
            amounts=amounts,
        ),
        None,
    )
    economics_field = _economics_field(
        economics=slice_economics,
        strategy_type=new_spec.strategy_type,
        new_result=new_result,
        current_result=current_result,
    )
    transition_conserved = (
        None
        if transition["availability"] != "present"
        else sum(
            row["effect"]["count"]
            for row in transition["value"]["rows"]
        )
        == len(analysis_frame)
    )
    body = {
        "schema_version": STRATEGY_IMPACT_SLICE_SCHEMA_VERSION,
        "producer_version": STRATEGY_IMPACT_CUBE_PRODUCER_VERSION,
        "family": family,
        "population_role": population_role,
        "availability": "present",
        "unavailable_reason": None,
        "dimensions": dict(dimensions),
        "population": _typed_field("present", population, None),
        "new": _typed_field(
            "present",
            _projection(new_result, population_role=population_role),
            None,
        ),
        "current": (
            _typed_field(
                "unavailable",
                None,
                "current_strategy_not_bound",
            )
            if current_result is None
            else _typed_field(
                "present",
                _projection(
                    current_result,
                    population_role=population_role,
                ),
                None,
            )
        ),
        "transition": transition,
        "waterfall": waterfall,
        "economics": economics_field,
        "conservation": {
            "waterfall_incremental_plus_default_equals_population": True,
            "waterfall_standalone_equals_incremental_plus_shadowed": True,
            "transition_equals_population": transition_conserved,
        },
    }
    return _finalize_slice(body)


def _build_unavailable_slice(
    *,
    population_role: str,
    partition: str,
    family: str,
    dimensions: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    unavailable = _typed_field("unavailable", None, reason)
    body = {
        "schema_version": STRATEGY_IMPACT_SLICE_SCHEMA_VERSION,
        "producer_version": STRATEGY_IMPACT_CUBE_PRODUCER_VERSION,
        "family": family,
        "population_role": population_role,
        "availability": "unavailable",
        "unavailable_reason": reason,
        "dimensions": dict(dimensions),
        "population": dict(unavailable),
        "new": dict(unavailable),
        "current": dict(unavailable),
        "transition": dict(unavailable),
        "waterfall": dict(unavailable),
        "economics": dict(unavailable),
        "conservation": {
            "waterfall_incremental_plus_default_equals_population": None,
            "waterfall_standalone_equals_incremental_plus_shadowed": None,
            "transition_equals_population": None,
        },
    }
    return _finalize_slice(body)


def _finalize_slice(body: Mapping[str, Any]) -> dict[str, Any]:
    slice_id = "strategy-impact-slice-" + _sha256(_canonical_json(body))[:24]
    document = {**dict(body), "slice_id": slice_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    return document


def _slice_descriptors(
    *,
    population_role: str,
    partition: str,
    frame: pd.DataFrame,
    new_spec: StrategySpec,
    fields: Mapping[str, str | None],
    max_slices: int,
) -> list[dict[str, Any]]:
    if max_slices < 1:
        raise StrategyError("ImpactCube slice budget exceeded")
    descriptors = [
        _descriptor(
            population_role=population_role,
            partition=partition,
            family="overall",
            dimensions=_dimensions(partition=partition),
            codes=None,
            code=None,
            row_count=len(frame),
        )
    ]
    month_groups = _field_groups(
        frame,
        field=fields["month_col"],
        dimension="month",
    )
    group_groups = _field_groups(
        frame,
        field=fields["group_col"],
        dimension="group",
    )
    segment_groups = _field_groups(
        frame,
        field=fields["segment_col"],
        dimension="segment",
    )
    _extend_descriptors(
        descriptors,
        _single_dimension_descriptors(
            population_role=population_role,
            partition=partition,
            family="month",
            dimension="month",
            groups=month_groups,
            max_groups=max_slices - len(descriptors),
        ),
        max_slices=max_slices,
    )
    _extend_descriptors(
        descriptors,
        _single_dimension_descriptors(
            population_role=population_role,
            partition=partition,
            family="group",
            dimension="group",
            groups=group_groups,
            max_groups=max_slices - len(descriptors),
        ),
        max_slices=max_slices,
    )
    _extend_descriptors(
        descriptors,
        _single_dimension_descriptors(
            population_role=population_role,
            partition=partition,
            family="segment",
            dimension="segment",
            groups=segment_groups,
            max_groups=max_slices - len(descriptors),
        ),
        max_slices=max_slices,
    )
    _extend_descriptors(
        descriptors,
        _cross_dimension_descriptors(
            population_role=population_role,
            partition=partition,
            family="group_month",
            left_dimension="group",
            left_groups=group_groups,
            right_dimension="month",
            right_groups=month_groups,
            max_groups=max_slices - len(descriptors),
        ),
        max_slices=max_slices,
    )
    _extend_descriptors(
        descriptors,
        _cross_dimension_descriptors(
            population_role=population_role,
            partition=partition,
            family="segment_month",
            left_dimension="segment",
            left_groups=segment_groups,
            right_dimension="month",
            right_groups=month_groups,
            max_groups=max_slices - len(descriptors),
        ),
        max_slices=max_slices,
    )
    evaluation = evaluate_strategy_frame(frame, new_spec)
    action_groups = _action_groups(evaluation, new_spec)
    _extend_descriptors(
        descriptors,
        _single_dimension_descriptors(
            population_role=population_role,
            partition=partition,
            family="new_action",
            dimension="new_action_bucket",
            groups=action_groups,
            max_groups=max_slices - len(descriptors),
        ),
        max_slices=max_slices,
    )
    return descriptors


def _extend_descriptors(
    target: list[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    max_slices: int,
) -> None:
    if len(target) + len(rows) > max_slices:
        raise StrategyError("ImpactCube slice budget exceeded")
    target.extend(rows)


def _single_dimension_descriptors(
    *,
    population_role: str,
    partition: str,
    family: str,
    dimension: str,
    groups: _DimensionGroups | None,
    max_groups: int,
) -> list[dict[str, Any]]:
    if groups is None:
        if max_groups < 1:
            raise StrategyError("ImpactCube slice budget exceeded")
        reason = f"{dimension}_field_not_bound"
        return [
            _unavailable_descriptor(
                population_role=population_role,
                partition=partition,
                family=family,
                reason=reason,
                dimensions=_dimensions(
                    partition=partition,
                    **{dimension: _dimension("unavailable", None)},
                ),
            )
        ]
    if len(groups.values) > max_groups:
        raise StrategyError("ImpactCube slice budget exceeded")
    counts = np.bincount(groups.codes, minlength=len(groups.values))
    return [
        _descriptor(
            population_role=population_role,
            partition=partition,
            family=family,
            dimensions=_dimensions(
                partition=partition,
                **{dimension: value},
            ),
            codes=groups.codes,
            code=code,
            row_count=int(counts[code]),
        )
        for code, value in enumerate(groups.values)
    ]


def _cross_dimension_descriptors(
    *,
    population_role: str,
    partition: str,
    family: str,
    left_dimension: str,
    left_groups: _DimensionGroups | None,
    right_dimension: str,
    right_groups: _DimensionGroups | None,
    max_groups: int,
) -> list[dict[str, Any]]:
    missing = [
        dimension
        for dimension, groups in (
            (left_dimension, left_groups),
            (right_dimension, right_groups),
        )
        if groups is None
    ]
    if missing:
        if max_groups < 1:
            raise StrategyError("ImpactCube slice budget exceeded")
        reason = "_and_".join(missing) + "_field_not_bound"
        values = {
            dimension: _dimension(
                "unavailable" if dimension in missing else "all",
                None,
            )
            for dimension in (left_dimension, right_dimension)
        }
        return [
            _unavailable_descriptor(
                population_role=population_role,
                partition=partition,
                family=family,
                reason=reason,
                dimensions=_dimensions(partition=partition, **values),
            )
        ]
    assert left_groups is not None
    assert right_groups is not None
    right_count = len(right_groups.values)
    combined_codes = (
        left_groups.codes.astype(np.int64, copy=False) * right_count
        + right_groups.codes.astype(np.int64, copy=False)
    )
    observed_codes, counts = np.unique(combined_codes, return_counts=True)
    small_codes = {
        int(code)
        for code, count in zip(observed_codes, counts, strict=True)
        if int(count) < MIN_IMPACT_CUBE_GROUP_SIZE
    }
    redacted_count = sum(
        int(count)
        for code, count in zip(observed_codes, counts, strict=True)
        if int(code) in small_codes
    )
    if small_codes and redacted_count < MIN_IMPACT_CUBE_GROUP_SIZE:
        candidates = sorted(
            (
                int(count),
                int(code),
            )
            for code, count in zip(observed_codes, counts, strict=True)
            if int(code) not in small_codes
        )
        for count, code in candidates:
            small_codes.add(code)
            redacted_count += count
            if redacted_count >= MIN_IMPACT_CUBE_GROUP_SIZE:
                break
    if small_codes:
        combined_codes = np.where(
            np.isin(combined_codes, tuple(small_codes)),
            -1,
            combined_codes,
        )
        observed_codes, counts = np.unique(combined_codes, return_counts=True)
    if len(observed_codes) > max_groups:
        raise StrategyError("ImpactCube slice budget exceeded")
    rows: list[dict[str, Any]] = []
    for combined_code, count in zip(observed_codes, counts, strict=True):
        code = int(combined_code)
        if code == -1:
            left_value = _dimension("redacted", None)
            right_value = _dimension("redacted", None)
        else:
            left_code, right_code = divmod(code, right_count)
            left_value = left_groups.values[left_code]
            right_value = right_groups.values[right_code]
        rows.append(
            _descriptor(
                population_role=population_role,
                partition=partition,
                family=family,
                dimensions=_dimensions(
                    partition=partition,
                    **{
                        left_dimension: left_value,
                        right_dimension: right_value,
                    },
                ),
                codes=combined_codes,
                code=code,
                row_count=int(count),
            )
        )
    return rows


def _descriptor(
    *,
    population_role: str,
    partition: str,
    family: str,
    dimensions: Mapping[str, Any],
    codes: np.ndarray | None,
    code: int | None,
    row_count: int,
) -> dict[str, Any]:
    return {
        "population_role": population_role,
        "partition": partition,
        "family": family,
        "availability": "present",
        "reason": None,
        "dimensions": dict(dimensions),
        "selector": (
            None
            if codes is None
            else {
                "codes": codes,
                "code": code,
            }
        ),
        "row_count": row_count,
    }


def _unavailable_descriptor(
    *,
    population_role: str,
    partition: str,
    family: str,
    reason: str,
    dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "population_role": population_role,
        "partition": partition,
        "family": family,
        "availability": "unavailable",
        "reason": reason,
        "dimensions": dict(dimensions),
        "row_count": 0,
    }


def _field_groups(
    frame: pd.DataFrame,
    *,
    field: str | None,
    dimension: str,
) -> _DimensionGroups | None:
    if field is None:
        return None
    values = frame[field].reset_index(drop=True)
    if dimension == "month":
        normalized = _month_values(values, field=field)
    else:
        normalized = values.map(_external_dimension_value)
    if dimension in {"group", "segment"}:
        nonnull = values.dropna()
        distinct = int(nonnull.nunique(dropna=True))
        if (
            len(nonnull) >= _HIGH_CARDINALITY_MIN_ROWS
            and distinct >= _HIGH_CARDINALITY_MIN_ROWS
            and distinct / len(nonnull) > 0.5
        ):
            raise StrategyError(
                "ImpactCube high-cardinality raw identifier dimension "
                f"is forbidden: {field}"
            )
    normalized = _merge_small_dimension_cells(
        normalized,
        dimension=dimension,
    )
    tokens = normalized.map(_dimension_token).tolist()
    seen: dict[str, dict[str, Any]] = {}
    for token, value in zip(tokens, normalized, strict=True):
        seen.setdefault(token, value)
    if len(seen) > MAX_IMPACT_CUBE_DIMENSION_VALUES:
        raise StrategyError(
            "ImpactCube slice budget exceeded: "
            f"{dimension} cardinality is too high"
        )
    ordered_tokens = tuple(sorted(seen))
    code_by_token = {
        token: code for code, token in enumerate(ordered_tokens)
    }
    codes = np.fromiter(
        (code_by_token[token] for token in tokens),
        dtype=np.int32,
        count=len(tokens),
    )
    return _DimensionGroups(
        values=tuple(seen[token] for token in ordered_tokens),
        codes=codes,
    )


def _merge_small_dimension_cells(
    normalized: pd.Series,
    *,
    dimension: str,
) -> pd.Series:
    """Replace undersized cells with one governed bucket containing no values."""

    tokens = normalized.map(_dimension_token)
    counts = tokens.value_counts(sort=False).to_dict()
    small = {
        token
        for token, count in counts.items()
        if int(count) < MIN_IMPACT_CUBE_GROUP_SIZE
    }
    redacted_count = sum(int(counts[token]) for token in small)
    if small and redacted_count < MIN_IMPACT_CUBE_GROUP_SIZE:
        candidates = sorted(
            (
                (int(count), str(token))
                for token, count in counts.items()
                if token not in small
            )
        )
        for count, token in candidates:
            small.add(token)
            redacted_count += count
            if redacted_count >= MIN_IMPACT_CUBE_GROUP_SIZE:
                break
    if not small:
        return normalized
    redacted = _dimension("redacted", None)
    return pd.Series(
        [
            redacted if token in small else value
            for token, value in zip(tokens, normalized, strict=True)
        ],
        index=normalized.index,
        dtype="object",
    )


def _month_values(values: pd.Series, *, field: str) -> pd.Series:
    normalized: list[dict[str, Any]] = []
    nonnull = ~values.isna()
    parsed = pd.Series(index=values.index, dtype="object")
    if bool(nonnull.any()):
        try:
            parsed.loc[nonnull] = month_key_series(
                values.loc[nonnull],
                column_name=field,
            )
        except ValueError as exc:
            raise StrategyError(str(exc)) from exc
    for index, value in values.items():
        if pd.isna(value):
            normalized.append(_dimension("null", None))
        else:
            normalized.append(_dimension("value", str(parsed.loc[index])))
    return pd.Series(normalized, index=values.index, dtype="object")


def _external_dimension_value(value: object) -> dict[str, Any]:
    if _is_missing_scalar(value):
        return _dimension("null", None)
    return _dimension("value", _json_scalar(value, "dimension value"))


def _action_groups(
    evaluation,
    spec: StrategySpec,
) -> _DimensionGroups:
    dimensions = [
        _dimension("value", action)
        for action in _action_bucket_series(evaluation, spec)
    ]
    tokens = [_dimension_token(value) for value in dimensions]
    unique = {
        token: value for token, value in zip(tokens, dimensions, strict=True)
    }
    ordered_tokens = tuple(sorted(unique))
    code_by_token = {
        token: code for code, token in enumerate(ordered_tokens)
    }
    return _DimensionGroups(
        values=tuple(unique[token] for token in ordered_tokens),
        codes=np.fromiter(
            (code_by_token[token] for token in tokens),
            dtype=np.int32,
            count=len(tokens),
        ),
    )


def _dimensions(
    *,
    partition: str,
    month: Mapping[str, Any] | None = None,
    group: Mapping[str, Any] | None = None,
    segment: Mapping[str, Any] | None = None,
    new_action_bucket: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "partition": _dimension("value", partition),
        "month": dict(month or _dimension("all", None)),
        "group": dict(group or _dimension("all", None)),
        "segment": dict(segment or _dimension("all", None)),
        "new_action_bucket": dict(
            new_action_bucket or _dimension("all", None)
        ),
    }


def _dimension(kind: str, value: Any) -> dict[str, Any]:
    return {"kind": kind, "value": value}


def _dimension_token(value: Mapping[str, Any]) -> str:
    return _canonical_json(value)


def _transition_field(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    new_spec: StrategySpec,
    current_spec: StrategySpec | None,
    population_role: str,
    amounts: Mapping[str, Any],
) -> dict[str, Any]:
    if current_spec is None:
        return _typed_field(
            "unavailable",
            None,
            "current_strategy_not_bound",
        )
    current_eval = evaluate_strategy_frame(frame, current_spec)
    new_eval = evaluate_strategy_frame(frame, new_spec)
    current_buckets = _action_bucket_series(current_eval, current_spec)
    new_buckets = _action_bucket_series(new_eval, new_spec)
    pair_tokens = pd.Series(
        [
            _canonical_json({"from": old, "to": new})
            for old, new in zip(current_buckets, new_buckets, strict=True)
        ],
        dtype="object",
    )
    pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for token, old, new in zip(
        pair_tokens,
        current_buckets,
        new_buckets,
        strict=True,
    ):
        pairs.setdefault(token, (old, new))
    rows: list[dict[str, Any]] = []
    for token in sorted(pairs):
        old, new = pairs[token]
        mask = pair_tokens.eq(token)
        rows.append(
            {
                "from_bucket": old,
                "to_bucket": new,
                "direction": _transition_direction(
                    new_spec.strategy_type,
                    old,
                    new,
                ),
                "effect": _population_summary(
                    mask,
                    target=target,
                    denominator=len(frame),
                    population_role=population_role,
                    amounts=amounts,
                ),
            }
        )
    return _typed_field("present", {"rows": rows}, None)


def _action_bucket_series(
    evaluation,
    spec: StrategySpec,
) -> list[dict[str, Any]]:
    by_rule_id = {
        rule.rule_id: rule.action.to_dict()
        for rule in spec.rules
    }
    default_action = spec.default_action.to_dict()
    buckets: list[dict[str, Any]] = []
    for raw_rule_id in evaluation.matched_rule_id.reset_index(drop=True):
        if raw_rule_id is None:
            action = default_action
        else:
            rule_id = _text(raw_rule_id, "matched rule id")
            try:
                action = by_rule_id[rule_id]
            except KeyError as exc:
                raise StrategyError(
                    "ImpactCube evaluator returned an unknown rule id"
                ) from exc
        buckets.append(dict(action))
    return buckets


def _transition_direction(
    strategy_type: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> str:
    if old == new:
        return "unchanged"
    if strategy_type in {"limit", "pricing"}:
        old_value = float(old.get("output_value", old["value"]))
        new_value = float(new.get("output_value", new["value"]))
        return "increase" if new_value > old_value else "decrease"
    if strategy_type in {"approval", "reject"}:
        old_approved = old["type"] == "approval"
        new_approved = new["type"] == "approval"
        if new_approved and not old_approved:
            return "swap_in"
        if old_approved and not new_approved:
            return "swap_out"
    return "changed"


def _waterfall(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    pool: Mapping[str, Any],
    spec: StrategySpec,
    population_role: str,
    amounts: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = evaluate_strategy_frame(frame, spec)
    matched = evaluation.matched_rule_id.reset_index(drop=True)
    claimed = pd.Series(False, index=frame.index, dtype=bool)
    entries: list[dict[str, Any]] = []
    for position, (pool_entry, rule) in enumerate(
        zip(pool["entries"], spec.rules, strict=True),
        start=1,
    ):
        standalone = evaluate_expression_frame(
            frame,
            rule.condition,
        ).reset_index(drop=True)
        incremental = matched.eq(rule.rule_id)
        shadowed = standalone & claimed
        if not (standalone == (incremental | shadowed)).all():
            raise StrategyError(
                f"ImpactCube waterfall failed for rule_id {rule.rule_id}"
            )
        if bool((incremental & shadowed).any()):
            raise StrategyError(
                f"ImpactCube waterfall overlaps for rule_id {rule.rule_id}"
            )
        claimed |= standalone
        entries.append(
            {
                "position": position,
                "entry_id": pool_entry["entry_id"],
                "rule_id": rule.rule_id,
                "source_ref": {
                    key: pool_entry["source"][key]
                    for key in (
                        "artifact_id",
                        "artifact_content_hash",
                        "asset_id",
                        "asset_hash",
                        "fragment_id",
                    )
                },
                "action": rule.action.to_dict(),
                "standalone": _population_summary(
                    standalone,
                    target=target,
                    denominator=len(frame),
                    population_role=population_role,
                    amounts=amounts,
                ),
                "incremental": _population_summary(
                    incremental,
                    target=target,
                    denominator=len(frame),
                    population_role=population_role,
                    amounts=amounts,
                ),
                "shadowed": _population_summary(
                    shadowed,
                    target=target,
                    denominator=len(frame),
                    population_role=population_role,
                    amounts=amounts,
                ),
                "remaining_after": _population_summary(
                    ~claimed,
                    target=target,
                    denominator=len(frame),
                    population_role=population_role,
                    amounts=amounts,
                ),
            }
        )
    unmatched = matched.isna()
    if (
        sum(row["incremental"]["count"] for row in entries)
        + int(unmatched.sum())
        != len(frame)
    ):
        raise StrategyError("ImpactCube waterfall population does not conserve")
    return {
        "entries": entries,
        "default_unmatched": {
            "action": spec.default_action.to_dict(),
            "effect": _population_summary(
                unmatched,
                target=target,
                denominator=len(frame),
                population_role=population_role,
                amounts=amounts,
            ),
        },
    }


def _population_summary(
    mask: pd.Series,
    *,
    target: pd.Series,
    denominator: int,
    population_role: str,
    amounts: Mapping[str, Any],
) -> dict[str, Any]:
    selected = mask.reset_index(drop=True).astype(bool)
    labelled_mask = selected & target.notna()
    count = int(selected.sum())
    labeled = int(labelled_mask.sum())
    bad = int(target.loc[labelled_mask].eq(1).sum())
    if population_role == "approval":
        risk = {
            "availability": "not_applicable",
            "reason": "approval_population",
            "bad_count": 0,
            "bad_rate": None,
        }
    else:
        risk = (
        {
            "availability": "unavailable",
            "reason": "no_labeled_rows",
            "bad_count": 0,
            "bad_rate": None,
        }
        if labeled == 0
        else {
            "availability": "present",
            "reason": None,
            "bad_count": bad,
            "bad_rate": float(bad / labeled),
        }
        )
    return {
        "role": population_role,
        "count": count,
        "labeled_count": labeled,
        "unlabeled_count": count - labeled,
        "label_coverage": _ratio(labeled, count),
        "share": _ratio(count, denominator),
        "risk": risk,
        "amounts": _amount_observations(selected, amounts=amounts),
    }


def _projection(
    result: StrategyBacktestResult,
    *,
    population_role: str,
) -> dict[str, Any]:
    return {
        "strategy_id": result.strategy_id,
        "strategy_type": result.strategy_type,
        "population_role": population_role,
        "population_count": result.population_count,
        "labeled_count": result.labeled_count,
        "label_coverage": result.label_coverage,
        "metrics": _json_value(result.metrics, "strategy metrics"),
        "breakdown": [
            _json_value(row, "strategy breakdown")
            for row in result.breakdown
        ],
        "warnings": list(result.warnings),
    }


def _economics_field(
    *,
    economics: Mapping[str, Any],
    strategy_type: str,
    new_result: StrategyBacktestResult,
    current_result: StrategyBacktestResult | None,
) -> dict[str, Any]:
    if economics["availability"] != "present":
        return _typed_field(
            economics["availability"],
            None,
            economics["reason"],
        )
    new_values = _aggregate_economics(new_result.economics)
    current_values = (
        None
        if current_result is None
        else _aggregate_economics(current_result.economics)
    )
    if not new_values:
        raise StrategyError(
            f"ImpactCube {strategy_type} economics were bound but not computed"
        )
    delta: dict[str, float] = {}
    if current_values is not None:
        for key in sorted(set(new_values) & set(current_values)):
            new_value = new_values[key]
            old_value = current_values[key]
            if (
                isinstance(new_value, Real)
                and not isinstance(new_value, bool)
                and isinstance(old_value, Real)
                and not isinstance(old_value, bool)
            ):
                delta[key] = float(new_value) - float(old_value)
        if strategy_type in {"approval", "reject", "pricing"}:
            new_values["baseline_profit"] = current_values["profit"]
            new_values["profit_delta_vs_baseline"] = delta["profit"]
    return _typed_field(
        "present",
        {
            "new": new_values,
            "current": current_values,
            "delta": delta,
        },
        None,
    )


def _aggregate_economics(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError("ImpactCube economics result must be an object")
    return {
        key: _json_value(item, f"economics.{key}")
        for key, item in sorted(value.items())
        if key != "by_row"
    }


def _economics_arguments(
    frame: pd.DataFrame,
    *,
    strategy_type: str,
    economics: Mapping[str, Any],
) -> dict[str, Any]:
    if economics["availability"] != "present":
        return {
            "economics_inputs": None,
            "approval_profit_inputs": None,
        }
    resolved: dict[str, Any] = {}
    for name, binding in economics["bindings"].items():
        if binding["kind"] == "column":
            resolved[name] = frame[binding["column"]].reset_index(drop=True)
        else:
            resolved[name] = binding["value"]
    if strategy_type in {"approval", "reject"}:
        ead_col = economics["bindings"]["ead"]["column"]
        pd_col = economics["bindings"]["pd"]["column"]
        params = ProfitParams(
            annual_rate=float(resolved["annual_rate"]),
            funding_rate=float(resolved["funding_rate"]),
            lgd=float(resolved["lgd"]),
            operating_cost_per_loan=float(
                resolved["operating_cost_per_loan"]
            ),
            term_months=_positive_int(
                resolved["term_months"],
                "economics term_months",
            ),
        )
        return {
            "economics_inputs": None,
            "approval_profit_inputs": ApprovalProfitInputs(
                params=params,
                ead_col=ead_col,
                pd_col=pd_col,
            ),
        }
    return {
        "economics_inputs": resolved,
        "approval_profit_inputs": None,
    }


def _economics_binding(
    value: Mapping[str, Mapping[str, Any]] | None,
    *,
    strategy_type: str,
) -> dict[str, Any]:
    required = _ECONOMIC_COMPONENTS[strategy_type]
    if strategy_type == "segmentation":
        if value not in (None, {}):
            raise StrategyError(
                "segmentation ImpactCube does not accept economics inputs"
            )
        return {
            "availability": "not_applicable",
            "reason": "segmentation_has_no_economic_contract",
            "bindings": {},
        }
    if value is None:
        return {
            "availability": "unavailable",
            "reason": "economics_inputs_not_provided",
            "bindings": {},
        }
    obj = _json_object(value, "ImpactCube economics bindings")
    unsupported = sorted(set(obj) - required)
    if unsupported:
        raise StrategyError(
            "unsupported ImpactCube economics inputs: "
            + ", ".join(unsupported)
        )
    bindings = {
        key: _economic_component(obj[key], key)
        for key in sorted(obj)
    }
    missing = sorted(required - set(bindings))
    if missing:
        return {
            "availability": "unavailable",
            "reason": "missing_economics_inputs:" + ",".join(missing),
            "bindings": bindings,
        }
    if strategy_type in {"approval", "reject"}:
        for field in ("ead", "pd"):
            if bindings[field]["kind"] != "column":
                raise StrategyError(
                    f"ImpactCube approval economics {field} must bind a column"
                )
        for field in required - {"ead", "pd"}:
            if bindings[field]["kind"] != "scalar":
                raise StrategyError(
                    f"ImpactCube approval economics {field} must be scalar"
                )
    return {
        "availability": "present",
        "reason": None,
        "bindings": bindings,
    }


def _economic_component(value: object, name: str) -> dict[str, Any]:
    obj = _json_object(value, f"economics.{name}")
    kind = _text(obj.get("kind"), f"economics.{name}.kind")
    if kind == "column":
        _exact_fields(
            obj,
            frozenset({"kind", "column"}),
            f"economics.{name}",
        )
        return {
            "kind": "column",
            "column": _text(obj["column"], f"economics.{name}.column"),
        }
    if kind == "scalar":
        _exact_fields(
            obj,
            frozenset({"kind", "value"}),
            f"economics.{name}",
        )
        return {
            "kind": "scalar",
            "value": _finite_number(
                obj["value"],
                f"economics.{name}.value",
            ),
        }
    raise StrategyError(f"economics.{name}.kind must be column or scalar")


def _current_strategy(
    spec: Mapping[str, Any] | StrategySpec | None,
    ref: Mapping[str, Any] | None,
    *,
    strategy_type: str,
) -> tuple[StrategySpec | None, dict[str, Any] | None]:
    if spec is None and ref is None:
        return None, None
    if spec is None or ref is None:
        raise StrategyError(
            "ImpactCube current strategy spec and ref must be supplied together"
        )
    parsed = parse_strategy_spec(spec)
    if parsed.strategy_type != strategy_type:
        raise StrategyError("ImpactCube current strategy type must match Pool")
    obj = _json_object(ref, "ImpactCube current strategy ref")
    _exact_fields(obj, _CURRENT_REF_FIELDS, "ImpactCube current strategy ref")
    normalized = {
        "strategy_id": _text(
            obj["strategy_id"],
            "ImpactCube current strategy_id",
        ),
        "strategy_type": _text(
            obj["strategy_type"],
            "ImpactCube current strategy_type",
        ),
        "strategy_spec_hash": _hash(
            obj["strategy_spec_hash"],
            "ImpactCube current strategy_spec_hash",
        ),
    }
    calculated = strategy_spec_hash(parsed)
    if (
        normalized["strategy_type"] != strategy_type
        or not hmac.compare_digest(
            normalized["strategy_spec_hash"],
            calculated,
        )
    ):
        raise StrategyError("ImpactCube current strategy ref changed")
    return parsed, normalized


def _partition_frames(
    value: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if not isinstance(value, Mapping) or not value:
        raise StrategyError("ImpactCube partition_frames must be a non-empty object")
    unsupported = sorted(set(value) - _PARTITIONS)
    if unsupported:
        raise StrategyError(
            "ImpactCube has unsupported partitions: " + ", ".join(unsupported)
        )
    frames: dict[str, pd.DataFrame] = {}
    for partition in _PARTITION_ORDER:
        frame = value.get(partition)
        if frame is None:
            continue
        if not isinstance(frame, pd.DataFrame):
            raise StrategyError(
                f"ImpactCube {partition} partition must be a DataFrame"
            )
        if frame.empty:
            raise StrategyError(f"ImpactCube {partition} partition is empty")
        frames[partition] = frame.reset_index(drop=True).copy()
    return frames


def _pool_sample_binding(pool: Mapping[str, Any]) -> dict[str, Any]:
    identities = [
        _json_object(
            entry["source"]["evidence_identity"],
            "ImpactCube Pool evidence identity",
        )
        for entry in pool["entries"]
    ]
    if not identities:
        raise StrategyError("ImpactCube Pool has no lineage")
    first = identities[0]
    _exact_fields(
        first,
        _SAMPLE_BINDING_FIELDS - {"task_id"},
        "ImpactCube Pool evidence identity",
    )
    for identity in identities[1:]:
        if identity != first:
            raise StrategyError("ImpactCube Pool entries have mixed sample lineage")
    return {
        "task_id": pool["task_id"],
        **first,
    }


def _pool_artifact_ref(value: object) -> dict[str, str]:
    obj = _json_object(value, "ImpactCube pool artifact ref")
    _exact_fields(obj, _POOL_ARTIFACT_FIELDS, "ImpactCube pool artifact ref")
    return {
        field: _hash(obj[field], f"ImpactCube pool artifact {field}")
        for field in sorted(_POOL_ARTIFACT_FIELDS)
    }


def _sample_design_v2_ref(
    value: object,
    *,
    partitions: tuple[str, ...],
) -> dict[str, Any]:
    obj = _json_object(value, "ImpactCube sample design V2 ref")
    legacy_fields = _SAMPLE_FIELDS - {"population_partition_counts"}
    if set(obj) not in {legacy_fields, _SAMPLE_FIELDS}:
        raise StrategyError(
            "ImpactCube sample design V2 ref fields are invalid"
        )
    result = dict(obj)
    for field in (
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "membership_content_hash",
        "bundle_artifact_id",
        "bundle_artifact_content_hash",
        "bundle_content_hash",
        "sample_design_content_hash",
    ):
        result[field] = _hash(
            obj[field],
            f"ImpactCube sample design {field}",
        )
    for field in (
        "membership_id",
        "bundle_id",
        "sample_design_id",
    ):
        result[field] = _text(
            obj[field],
            f"ImpactCube sample design {field}",
        )
    result["analysis_universe_row_count"] = _positive_int(
        obj["analysis_universe_row_count"],
        "ImpactCube analysis universe row_count",
    )
    counts = _json_object(
        obj["partition_counts"],
        "ImpactCube partition counts",
    )
    if set(counts) != set(partitions):
        raise StrategyError(
            "ImpactCube partition counts do not match selected partitions"
        )
    result["partition_counts"] = {
        partition: _positive_int(
            counts[partition],
            f"ImpactCube {partition} partition count",
        )
        for partition in _PARTITION_ORDER
        if partition in counts
    }
    population_counts = (
        {
            "approval": dict(result["partition_counts"]),
            "risk": dict(result["partition_counts"]),
        }
        if "population_partition_counts" not in obj
        else _json_object(
            obj["population_partition_counts"],
            "ImpactCube population partition counts",
        )
    )
    _exact_fields(
        population_counts,
        _POPULATION_ROLES,
        "ImpactCube population partition counts",
    )
    result["population_partition_counts"] = {}
    for role in _POPULATION_ROLE_ORDER:
        role_counts = _json_object(
            population_counts[role],
            f"ImpactCube {role} partition counts",
        )
        if set(role_counts) != set(partitions):
            raise StrategyError(
                f"ImpactCube {role} partition counts do not match "
                "selected partitions"
            )
        result["population_partition_counts"][role] = {
            partition: _positive_int(
                role_counts[partition],
                f"ImpactCube {role}/{partition} partition count",
            )
            for partition in _PARTITION_ORDER
            if partition in role_counts
        }
    if result["partition_counts"] != result[
        "population_partition_counts"
    ]["risk"]:
        raise StrategyError(
            "ImpactCube legacy risk partition counts changed"
        )
    for partition in partitions:
        if (
            result["population_partition_counts"]["risk"][partition]
            > result["population_partition_counts"]["approval"][partition]
        ):
            raise StrategyError(
                f"ImpactCube risk/{partition} exceeds approval population"
            )
    if sum(result["partition_counts"].values()) > result[
        "analysis_universe_row_count"
    ]:
        raise StrategyError(
            "ImpactCube partition counts exceed analysis universe"
        )
    if sum(
        result["population_partition_counts"]["approval"].values()
    ) > result["analysis_universe_row_count"]:
        raise StrategyError(
            "ImpactCube approval partition counts exceed analysis universe"
        )
    return result


def _dataset_binding(value: object) -> dict[str, Any]:
    obj = _json_object(value, "ImpactCube dataset binding")
    _exact_fields(obj, _DATASET_FIELDS, "ImpactCube dataset binding")
    result = dict(obj)
    for field in (
        "dataset_content_hash",
        "dataset_registry_metadata_hash",
        "semantic_mapping_hash",
    ):
        result[field] = _hash(obj[field], f"ImpactCube dataset {field}")
    for field in (
        "task_id",
        "dataset_id",
        "dataset_source_path",
    ):
        result[field] = _text(obj[field], f"ImpactCube dataset {field}")
    for field in ("workspace_revision", "workspace_generation"):
        result[field] = _nonnegative_int(
            obj[field],
            f"ImpactCube dataset {field}",
        )
    return result


def _target_binding(column: object, bad_value: object) -> dict[str, Any]:
    target_bad_value = _target_bad_value(bad_value)
    return {
        "column": _text(column, "ImpactCube target column"),
        "good_value": 1 - target_bad_value,
        "bad_value": target_bad_value,
        "missing_policy": "retain_population_exclude_risk_denominator",
    }


def _field_bindings(
    *,
    month_col: object,
    group_col: object,
    segment_col: object,
    loan_amount_col: object,
    overdue_amount_col: object,
    target_col: str,
) -> dict[str, str | None]:
    result = {
        "month_col": _optional_text(month_col, "ImpactCube month_col"),
        "group_col": _optional_text(group_col, "ImpactCube group_col"),
        "segment_col": _optional_text(segment_col, "ImpactCube segment_col"),
        "loan_amount_col": _optional_text(
            loan_amount_col,
            "ImpactCube loan_amount_col",
        ),
        "overdue_amount_col": _optional_text(
            overdue_amount_col,
            "ImpactCube overdue_amount_col",
        ),
    }
    bound = [value for value in result.values() if value is not None]
    if len(bound) != len(set(bound)):
        raise StrategyError("ImpactCube dimension columns must be distinct")
    if target_col in bound:
        raise StrategyError("ImpactCube target cannot be a slice dimension")
    return result


def _privacy_binding(
    *,
    fields: Mapping[str, str | None],
    semantic_field_roles: Mapping[str, str] | None,
    entity_col: str | None,
) -> dict[str, Any]:
    roles_obj = {} if semantic_field_roles is None else dict(semantic_field_roles)
    roles: dict[str, str] = {}
    for column, role in roles_obj.items():
        normalized_column = _text(column, "ImpactCube semantic role column")
        normalized_role = _text(role, "ImpactCube semantic role").lower()
        roles[normalized_column] = normalized_role
    entity = _optional_text(entity_col, "ImpactCube entity_col")
    bound_roles: dict[str, str | None] = {}
    for dimension, field_name in _DIMENSION_FIELD_NAMES.items():
        column = fields[field_name]
        role = None if column is None else roles.get(column)
        bound_roles[dimension] = role
        if column is None:
            continue
        if column == entity or role in _FORBIDDEN_DIMENSION_ROLES:
            raise StrategyError(
                "ImpactCube identifier, personal, or sensitive dimension "
                f"is forbidden: {column}"
            )
    binding = {
        "policy_version": "strategy.dimension-privacy.v1",
        "minimum_group_size": MIN_IMPACT_CUBE_GROUP_SIZE,
        "forbidden_semantic_roles": sorted(_FORBIDDEN_DIMENSION_ROLES),
        "bound_dimension_roles": bound_roles,
        "redacted_bucket_kind": "redacted",
    }
    return {
        **binding,
        "policy_hash": _sha256(_canonical_json(binding)),
    }


def _slice_family_status(
    fields: Mapping[str, str | None],
) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    status["overall"] = {
        "availability": "present",
        "reason": None,
    }
    for family, field in (
        ("month", "month_col"),
        ("group", "group_col"),
        ("segment", "segment_col"),
    ):
        status[family] = {
            "availability": (
                "present" if fields[field] is not None else "unavailable"
            ),
            "reason": (
                None
                if fields[field] is not None
                else f"{family}_field_not_bound"
            ),
        }
    for family, left, right in (
        ("group_month", "group_col", "month_col"),
        ("segment_month", "segment_col", "month_col"),
    ):
        missing = [
            name.removesuffix("_col")
            for name in (left, right)
            if fields[name] is None
        ]
        status[family] = {
            "availability": "present" if not missing else "unavailable",
            "reason": (
                None
                if not missing
                else "_and_".join(missing) + "_field_not_bound"
            ),
        }
    status["new_action"] = {
        "availability": "present",
        "reason": None,
    }
    return status


def _partition_stage(partition: str) -> dict[str, str]:
    if partition == "development":
        return {
            "effect_stage": "backtested",
            "validation_status": "unvalidated",
        }
    return {
        "effect_stage": "oot_validated",
        "validation_status": "independent_evidence",
    }


def _normalized_target(values: pd.Series, *, bad_value: int) -> pd.Series:
    numeric = pd.to_numeric(values.reset_index(drop=True), errors="coerce")
    invalid = numeric.notna() & ~numeric.isin([0, 1])
    source_nonnull = values.reset_index(drop=True).notna()
    failed_coercion = source_nonnull & numeric.isna()
    if bool((invalid | failed_coercion).any()):
        raise StrategyError(
            "ImpactCube target must contain only 0, 1, or missing"
        )
    return numeric.map(
        lambda value: (
            float("nan")
            if pd.isna(value)
            else float(1 if int(value) == bad_value else 0)
        )
    )


def _amount_series(
    frame: pd.DataFrame,
    *,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, column in (
        ("loan_amount", loan_amount_col),
        ("overdue_amount", overdue_amount_col),
    ):
        if column is None:
            result[key] = {
                "column": None,
                "values": None,
            }
            continue
        raw = frame[column].reset_index(drop=True)
        unsupported = raw.notna() & raw.map(
            lambda item: isinstance(
                item,
                bool | np.bool_ | complex | np.complexfloating,
            )
            or not isinstance(item, str | Real | Decimal)
        )
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & (
            unsupported
            | numeric.isna()
            | ~np.isfinite(numeric)
        )
        if bool(invalid.any()) or bool((numeric.dropna() < 0).any()):
            raise StrategyError(
                f"ImpactCube {column} must contain non-negative finite "
                "numbers or missing"
            )
        result[key] = {
            "column": column,
            "values": numeric.astype(float),
        }
    return result


def _amount_observations(
    mask: pd.Series,
    *,
    amounts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = mask.reset_index(drop=True).astype(bool)
    count = int(selected.sum())
    result: dict[str, Any] = {}
    for key in ("loan_amount", "overdue_amount"):
        item = amounts[key]
        values = item["values"]
        if values is None:
            result[key] = _typed_field(
                "unavailable",
                None,
                f"{key}_field_not_bound",
            )
            continue
        covered = selected & values.notna()
        covered_count = int(covered.sum())
        result[key] = _typed_field(
            "present",
            {
                "column": item["column"],
                "covered_count": covered_count,
                "coverage": _ratio(covered_count, count),
                "sum": float(values.loc[covered].sum()),
            },
            None,
        )
    loan = amounts["loan_amount"]["values"]
    overdue = amounts["overdue_amount"]["values"]
    if loan is None or overdue is None:
        result["paired"] = _typed_field(
            "unavailable",
            None,
            "paired_amount_fields_not_bound",
        )
    else:
        paired = selected & loan.notna() & overdue.notna()
        covered_count = int(paired.sum())
        loan_sum = float(loan.loc[paired].sum())
        overdue_sum = float(overdue.loc[paired].sum())
        result["paired"] = _typed_field(
            "present",
            {
                "covered_count": covered_count,
                "coverage": _ratio(covered_count, count),
                "loan_amount_sum": loan_sum,
                "overdue_amount_sum": overdue_sum,
                "overdue_rate": (
                    None if loan_sum == 0 else overdue_sum / loan_sum
                ),
            },
            None,
        )
    return result


def _strategy_fields(spec: StrategySpec) -> set[str]:
    fields: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            field = value.get("field")
            if isinstance(field, str):
                fields.add(field)
            for item in value.values():
                visit(item)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            str | bytes | bytearray,
        ):
            for item in value:
                visit(item)

    for rule in spec.rules:
        visit(rule.condition)
    return fields


def _typed_field(
    availability: str,
    value: Any,
    reason: str | None,
) -> dict[str, Any]:
    if availability == "present":
        if value is None or reason is not None:
            raise StrategyError("present ImpactCube fields require a value")
    elif availability in {"unavailable", "not_applicable"}:
        if value is not None or not isinstance(reason, str) or not reason:
            raise StrategyError(
                "unavailable ImpactCube fields require a reason and null value"
            )
    else:
        raise StrategyError("ImpactCube availability is invalid")
    return {
        "availability": availability,
        "reason": reason,
        "value": value,
    }


def _red_flags(
    *,
    family_status: Mapping[str, Mapping[str, Any]],
    current_ref: Mapping[str, Any] | None,
    economics: Mapping[str, Any],
    slices: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    redacted_cells = sum(
        1
        for row in slices
        if row["availability"] == "present"
        and any(
            item["kind"] == "redacted"
            for name, item in row["dimensions"].items()
            if name != "partition"
        )
    )
    if redacted_cells:
        flags.append(
            {
                "code": "dimension_cells_redacted",
                "level": "amber",
                "message": (
                    f"{redacted_cells} dimension cells were merged into "
                    "governed redacted buckets to enforce minimum group size."
                ),
            }
        )
    for family in ("month", "group", "segment"):
        status = family_status[family]
        if status["availability"] == "unavailable":
            flags.append(
                {
                    "code": f"{family}_slices_unavailable",
                    "level": "amber",
                    "message": (
                        f"{family} slices are unavailable because no "
                        f"{family} field was bound."
                    ),
                }
            )
    if current_ref is None:
        flags.append(
            {
                "code": "current_strategy_unavailable",
                "level": "amber",
                "message": (
                    "Current strategy was not bound; transition evidence is "
                    "unavailable."
                ),
            }
        )
    if economics["availability"] == "unavailable":
        flags.append(
            {
                "code": "economics_unavailable",
                "level": "amber",
                "message": (
                    "Economic impact is unavailable because deterministic "
                    "inputs are incomplete."
                ),
            }
        )
    for partition in _PARTITION_ORDER:
        overall = next(
            (
                row
                for row in slices
                if row["family"] == "overall"
                and row["population_role"] == "risk"
                and row["dimensions"]["partition"]["value"] == partition
            ),
            None,
        )
        if (
            overall is not None
            and overall["population"]["value"]["labeled_count"] == 0
        ):
            flags.append(
                {
                    "code": f"{partition}_labels_unavailable",
                    "level": "red",
                    "message": (
                        f"{partition} has no labeled rows; risk metrics are "
                        "unavailable."
                    ),
                }
            )
    return flags


def _validate_source_bindings(
    value: object,
    *,
    partitions: tuple[str, ...],
    strategy_type: str,
) -> dict[str, Any]:
    obj = _json_object(value, "ImpactCube source bindings")
    _exact_fields(obj, _SOURCE_FIELDS, "ImpactCube source bindings")
    result = {
        "pool_artifact": _pool_artifact_ref(obj["pool_artifact"]),
        "sample_design_v2": _sample_design_v2_ref(
            obj["sample_design_v2"],
            partitions=partitions,
        ),
        "dataset": _dataset_binding(obj["dataset"]),
    }
    lineage = _json_object(
        obj["development_lineage"],
        "ImpactCube development lineage",
    )
    _exact_fields(
        lineage,
        _DEVELOPMENT_LINEAGE_FIELDS,
        "ImpactCube development lineage",
    )
    legacy = StrategySampleDesignRef.from_value(
        lineage["legacy_development_ref"]
    ).to_ref_dict()
    sample = _json_object(
        lineage["sample_binding"],
        "ImpactCube sample binding",
    )
    _exact_fields(
        sample,
        _SAMPLE_BINDING_FIELDS,
        "ImpactCube sample binding",
    )
    result["development_lineage"] = {
        "legacy_development_ref": legacy,
        "sample_binding": sample,
    }
    target = _json_object(obj["target"], "ImpactCube target binding")
    _exact_fields(target, _TARGET_FIELDS, "ImpactCube target binding")
    expected_target = _target_binding(target["column"], target["bad_value"])
    if target != expected_target:
        raise StrategyError("ImpactCube target binding changed")
    result["target"] = target
    fields = _json_object(obj["fields"], "ImpactCube field bindings")
    _exact_fields(fields, _FIELD_FIELDS, "ImpactCube field bindings")
    result["fields"] = _field_bindings(
        month_col=fields["month_col"],
        group_col=fields["group_col"],
        segment_col=fields["segment_col"],
        loan_amount_col=fields["loan_amount_col"],
        overdue_amount_col=fields["overdue_amount_col"],
        target_col=target["column"],
    )
    privacy = _json_object(obj["privacy"], "ImpactCube privacy binding")
    _exact_fields(privacy, _PRIVACY_FIELDS, "ImpactCube privacy binding")
    if (
        privacy["policy_version"] != "strategy.dimension-privacy.v1"
        or privacy["minimum_group_size"] != MIN_IMPACT_CUBE_GROUP_SIZE
        or privacy["redacted_bucket_kind"] != "redacted"
        or privacy["forbidden_semantic_roles"]
        != sorted(_FORBIDDEN_DIMENSION_ROLES)
    ):
        raise StrategyError("ImpactCube privacy policy changed")
    policy_hash = _hash(
        privacy["policy_hash"],
        "ImpactCube privacy policy_hash",
    )
    policy_body = {
        key: value
        for key, value in privacy.items()
        if key != "policy_hash"
    }
    if not hmac.compare_digest(
        policy_hash,
        _sha256(_canonical_json(policy_body)),
    ):
        raise StrategyError("ImpactCube privacy policy_hash changed")
    bound_roles = _json_object(
        privacy["bound_dimension_roles"],
        "ImpactCube privacy bound roles",
    )
    _exact_fields(
        bound_roles,
        frozenset(_DIMENSION_FIELD_NAMES),
        "ImpactCube privacy bound roles",
    )
    for dimension, role in bound_roles.items():
        if role is not None:
            _text(role, f"ImpactCube privacy {dimension} role")
        if (
            result["fields"][_DIMENSION_FIELD_NAMES[dimension]] is None
            and role is not None
        ):
            raise StrategyError(
                "ImpactCube privacy role exists for an unbound dimension"
            )
        if role in _FORBIDDEN_DIMENSION_ROLES:
            raise StrategyError(
                "ImpactCube privacy contains a forbidden dimension role"
            )
    result["privacy"] = privacy
    current = _validate_typed_field(
        obj["current_strategy"],
        "ImpactCube current strategy",
    )
    if current["availability"] == "present":
        ref = _json_object(
            current["value"],
            "ImpactCube current strategy value",
        )
        _exact_fields(
            ref,
            _CURRENT_REF_FIELDS,
            "ImpactCube current strategy value",
        )
        if ref["strategy_type"] != strategy_type:
            raise StrategyError(
                "ImpactCube current strategy type does not match cube"
            )
        _hash(
            ref["strategy_spec_hash"],
            "ImpactCube current strategy spec hash",
        )
        _text(ref["strategy_id"], "ImpactCube current strategy id")
    elif current != _typed_field(
        "unavailable",
        None,
        "current_strategy_not_bound",
    ):
        raise StrategyError(
            "ImpactCube current strategy unavailable binding changed"
        )
    result["current_strategy"] = current
    economics = _json_object(obj["economics"], "ImpactCube economics binding")
    _exact_fields(
        economics,
        frozenset({"availability", "reason", "bindings"}),
        "ImpactCube economics binding",
    )
    binding_obj = _json_object(
        economics["bindings"],
        "ImpactCube economics components",
    )
    rebuilt = _economics_binding(
        (
            None
            if economics["availability"] == "unavailable"
            and economics["reason"] == "economics_inputs_not_provided"
            else binding_obj
        ),
        strategy_type=strategy_type,
    )
    if economics != rebuilt:
        raise StrategyError("ImpactCube economics binding changed")
    result["economics"] = economics
    return result


def _validate_partitions(value: object) -> list[dict[str, Any]]:
    rows = _list(value, "ImpactCube partitions")
    if not rows or len(rows) > len(_PARTITION_ORDER) * len(
        _POPULATION_ROLE_ORDER
    ):
        raise StrategyError("ImpactCube partitions are invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        row = _json_object(raw, f"ImpactCube partitions[{index}]")
        _exact_fields(
            row,
            _PARTITION_FIELDS,
            f"ImpactCube partitions[{index}]",
        )
        partition = _text(row["name"], "ImpactCube partition name")
        role = _text(row["role"], "ImpactCube population role")
        key = (role, partition)
        if (
            role not in _POPULATION_ROLES
            or partition not in _PARTITIONS
            or key in seen
        ):
            raise StrategyError("ImpactCube partition name is invalid")
        seen.add(key)
        expected = {
            "role": role,
            "name": partition,
            "population_key": f"{role}/{partition}",
            "row_count": _positive_int(
                row["row_count"],
                "ImpactCube partition row_count",
            ),
            **_partition_stage(partition),
        }
        if row != expected:
            raise StrategyError("ImpactCube partition metadata changed")
        normalized.append(row)
    by_role = {
        role: {
            row["name"] for row in normalized if row["role"] == role
        }
        for role in _POPULATION_ROLE_ORDER
    }
    if (
        not by_role["approval"]
        or by_role["approval"] != by_role["risk"]
    ):
        raise StrategyError(
            "ImpactCube approval and risk partition metadata must match"
        )
    return normalized


def _validate_family_status(value: object) -> dict[str, dict[str, Any]]:
    obj = _json_object(value, "ImpactCube slice families")
    _exact_fields(obj, _FAMILIES, "ImpactCube slice families")
    normalized: dict[str, dict[str, Any]] = {}
    for family in _FAMILY_ORDER:
        field = _json_object(
            obj[family],
            f"ImpactCube family {family}",
        )
        _exact_fields(
            field,
            frozenset({"availability", "reason"}),
            f"ImpactCube family {family}",
        )
        availability = field["availability"]
        reason = field["reason"]
        if availability == "present":
            if reason is not None:
                raise StrategyError(
                    f"ImpactCube family {family} present reason must be null"
                )
        elif availability == "unavailable":
            _text(reason, f"ImpactCube family {family} reason")
        else:
            raise StrategyError(
                f"ImpactCube family {family} availability is invalid"
            )
        normalized[family] = field
    return normalized


def _validate_slice(
    value: object,
    *,
    strategy_type: str,
) -> dict[str, Any]:
    obj = _json_object(value, "ImpactSlice")
    _exact_fields(obj, _SLICE_FIELDS, "ImpactSlice")
    if obj["schema_version"] != STRATEGY_IMPACT_SLICE_SCHEMA_VERSION:
        raise StrategyError("ImpactSlice schema_version is invalid")
    if obj["producer_version"] != STRATEGY_IMPACT_CUBE_PRODUCER_VERSION:
        raise StrategyError("ImpactSlice producer_version is invalid")
    slice_id = _text(obj["slice_id"], "ImpactSlice slice_id")
    if _SLICE_ID_RE.fullmatch(slice_id) is None:
        raise StrategyError("ImpactSlice slice_id is invalid")
    content_hash = _hash(obj["content_hash"], "ImpactSlice content_hash")
    without_hash = {
        key: item for key, item in obj.items() if key != "content_hash"
    }
    if not hmac.compare_digest(
        content_hash,
        _sha256(_canonical_json(without_hash)),
    ):
        raise StrategyError("ImpactSlice content_hash does not match content")
    body = {
        key: item
        for key, item in obj.items()
        if key not in {"slice_id", "content_hash"}
    }
    expected_id = "strategy-impact-slice-" + _sha256(
        _canonical_json(body)
    )[:24]
    if not hmac.compare_digest(slice_id, expected_id):
        raise StrategyError("ImpactSlice slice_id does not match content")
    family = _text(obj["family"], "ImpactSlice family")
    if family not in _FAMILIES:
        raise StrategyError("ImpactSlice family is invalid")
    population_role = _text(
        obj["population_role"],
        "ImpactSlice population_role",
    )
    if population_role not in _POPULATION_ROLES:
        raise StrategyError("ImpactSlice population_role is invalid")
    dimensions = _validate_dimensions(obj["dimensions"])
    if dimensions["partition"]["value"] not in _PARTITIONS:
        raise StrategyError("ImpactSlice partition dimension is invalid")
    availability = obj["availability"]
    _validate_family_dimensions(
        family=family,
        dimensions=dimensions,
        availability=availability,
    )
    if family == "new_action" and availability == "present":
        _validate_action_bucket(
            dimensions["new_action_bucket"]["value"],
            name="ImpactSlice new_action dimension",
            strategy_type=strategy_type,
        )
    if availability == "unavailable":
        reason = _text(
            obj["unavailable_reason"],
            "ImpactSlice unavailable_reason",
        )
        for field in (
            "population",
            "new",
            "current",
            "transition",
            "waterfall",
            "economics",
        ):
            typed = _validate_typed_field(
                obj[field],
                f"ImpactSlice {field}",
            )
            if (
                typed["availability"] != "unavailable"
                or typed["value"] is not None
                or typed["reason"] != reason
            ):
                raise StrategyError(
                    f"ImpactSlice unavailable {field} is inconsistent"
                )
        conservation = _json_object(
            obj["conservation"],
            "ImpactSlice unavailable conservation",
        )
        _exact_fields(
            conservation,
            _CONSERVATION_FIELDS,
            "ImpactSlice unavailable conservation",
        )
        if any(item is not None for item in conservation.values()):
            raise StrategyError(
                "ImpactSlice unavailable conservation must be null"
            )
        return obj
    if availability != "present" or obj["unavailable_reason"] is not None:
        raise StrategyError("ImpactSlice availability is invalid")

    population_field = _validate_typed_field(
        obj["population"],
        "ImpactSlice population",
    )
    if population_field["availability"] != "present":
        raise StrategyError("ImpactSlice population must be present")
    population = _validate_population(
        population_field["value"],
        "ImpactSlice population value",
        expected_role=population_role,
    )
    if not math.isclose(
        population["share"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise StrategyError("ImpactSlice population share must equal 1")
    new_field = _validate_typed_field(obj["new"], "ImpactSlice new")
    if new_field["availability"] != "present":
        raise StrategyError("ImpactSlice new strategy must be present")
    new = _validate_projection(
        new_field["value"],
        name="ImpactSlice new strategy",
        expected_type=strategy_type,
        expected_role=population_role,
    )
    _require_projection_population(
        new,
        population,
        name="ImpactSlice new strategy",
    )
    current_field = _validate_typed_field(
        obj["current"],
        "ImpactSlice current",
    )
    transition_field = _validate_typed_field(
        obj["transition"],
        "ImpactSlice transition",
    )
    if current_field["availability"] == "present":
        current = _validate_projection(
            current_field["value"],
            name="ImpactSlice current strategy",
            expected_type=strategy_type,
            expected_role=population_role,
        )
        _require_projection_population(
            current,
            population,
            name="ImpactSlice current strategy",
        )
        if transition_field["availability"] != "present":
            raise StrategyError(
                "ImpactSlice transition must be present with current strategy"
            )
        _validate_transition(
            transition_field["value"],
            population=population,
            strategy_type=strategy_type,
            population_role=population_role,
        )
    elif transition_field["availability"] != "unavailable":
        raise StrategyError(
            "ImpactSlice transition must be unavailable without current"
        )
    waterfall_field = _validate_typed_field(
        obj["waterfall"],
        "ImpactSlice waterfall",
    )
    if waterfall_field["availability"] != "present":
        raise StrategyError("ImpactSlice waterfall must be present")
    _validate_waterfall(
        waterfall_field["value"],
        population=population,
        strategy_type=strategy_type,
        population_role=population_role,
    )
    economics = _validate_typed_field(
        obj["economics"],
        "ImpactSlice economics",
    )
    if population_role == "risk":
        if obj["economics"] != _typed_field(
            "not_applicable",
            None,
            "risk_population_economics_not_applicable",
        ):
            raise StrategyError(
                "ImpactSlice risk economics must be not applicable"
            )
    elif economics["availability"] == "present":
        _validate_economics_value(
            economics["value"],
            strategy_type=strategy_type,
        )
    conservation = _json_object(
        obj["conservation"],
        "ImpactSlice conservation",
    )
    _exact_fields(
        conservation,
        _CONSERVATION_FIELDS,
        "ImpactSlice conservation",
    )
    if (
        conservation[
            "waterfall_incremental_plus_default_equals_population"
        ]
        is not True
        or conservation[
            "waterfall_standalone_equals_incremental_plus_shadowed"
        ]
        is not True
    ):
        raise StrategyError("ImpactSlice waterfall conservation is invalid")
    expected_transition = (
        True if transition_field["availability"] == "present" else None
    )
    if conservation["transition_equals_population"] is not expected_transition:
        raise StrategyError("ImpactSlice transition conservation is invalid")
    return obj


def _validate_slice_bindings(
    row: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    family_status: Mapping[str, Any],
    current: Mapping[str, Any],
    economics: Mapping[str, Any],
    fields: Mapping[str, str | None],
) -> None:
    if row["availability"] == "unavailable":
        if (
            family_status["availability"] != "unavailable"
            or row["unavailable_reason"] != family_status["reason"]
        ):
            raise StrategyError(
                "ImpactSlice unavailable reason changed from slice families"
            )
        active = _FAMILY_DIMENSIONS[row["family"]]
        for dimension in _DIMENSIONS - {"partition"}:
            expected_kind = "all"
            field_name = _DIMENSION_FIELD_NAMES.get(dimension)
            if (
                dimension in active
                and field_name is not None
                and fields[field_name] is None
            ):
                expected_kind = "unavailable"
            if row["dimensions"][dimension] != _dimension(
                expected_kind,
                None,
            ):
                raise StrategyError(
                    "ImpactSlice unavailable dimensions changed from bindings"
                )
        return

    amount_values = row["population"]["value"]["amounts"]
    for key, field_name in (
        ("loan_amount", "loan_amount_col"),
        ("overdue_amount", "overdue_amount_col"),
    ):
        column = fields[field_name]
        typed = amount_values[key]
        if column is None:
            expected = _typed_field(
                "unavailable",
                None,
                f"{key}_field_not_bound",
            )
            if typed != expected:
                raise StrategyError(
                    "ImpactSlice amount availability changed from binding"
                )
        elif (
            typed["availability"] != "present"
            or typed["value"]["column"] != column
        ):
            raise StrategyError(
                "ImpactSlice amount column changed from binding"
            )
    paired_expected = (
        "present"
        if fields["loan_amount_col"] is not None
        and fields["overdue_amount_col"] is not None
        else "unavailable"
    )
    if amount_values["paired"]["availability"] != paired_expected:
        raise StrategyError(
            "ImpactSlice paired amount availability changed from binding"
        )

    expected_new_id = (
        f"pool-design-{identity['strategy_spec_hash'][:24]}"
    )
    if row["new"]["value"]["strategy_id"] != expected_new_id:
        raise StrategyError(
            "ImpactSlice new strategy identity changed from Pool"
        )
    if current["availability"] == "present":
        if (
            row["current"]["availability"] != "present"
            or row["transition"]["availability"] != "present"
            or row["current"]["value"]["strategy_id"]
            != current["value"]["strategy_id"]
        ):
            raise StrategyError(
                "ImpactSlice current strategy changed from binding"
            )
    elif (
        row["current"]
        != _typed_field(
            "unavailable",
            None,
            "current_strategy_not_bound",
        )
        or row["transition"]
        != _typed_field(
            "unavailable",
            None,
            "current_strategy_not_bound",
        )
    ):
        raise StrategyError(
            "ImpactSlice current strategy unavailable binding changed"
        )

    if row["population_role"] == "risk":
        if row["economics"] != _typed_field(
            "not_applicable",
            None,
            "risk_population_economics_not_applicable",
        ):
            raise StrategyError(
                "ImpactSlice risk economics must be not applicable"
            )
    elif economics["availability"] == "present":
        if row["economics"]["availability"] != "present":
            raise StrategyError(
                "ImpactSlice economics changed from binding"
            )
        has_current_economics = (
            row["economics"]["value"]["current"] is not None
        )
        current_is_bound = current["availability"] == "present"
        if has_current_economics is not current_is_bound:
            raise StrategyError(
                "ImpactSlice current economics changed from current binding"
            )
    else:
        expected_economics = _typed_field(
            economics["availability"],
            None,
            economics["reason"],
        )
        if row["economics"] != expected_economics:
            raise StrategyError(
                "ImpactSlice economics changed from binding"
            )


def _validate_dimensions(value: object) -> dict[str, Any]:
    obj = _json_object(value, "ImpactSlice dimensions")
    _exact_fields(obj, _DIMENSIONS, "ImpactSlice dimensions")
    result: dict[str, Any] = {}
    for name in (
        "partition",
        "month",
        "group",
        "segment",
        "new_action_bucket",
    ):
        item = _json_object(
            obj[name],
            f"ImpactSlice dimension {name}",
        )
        _exact_fields(
            item,
            frozenset({"kind", "value"}),
            f"ImpactSlice dimension {name}",
        )
        kind = item["kind"]
        if kind not in {
            "all",
            "value",
            "null",
            "redacted",
            "unavailable",
        }:
            raise StrategyError(
                f"ImpactSlice dimension {name} kind is invalid"
            )
        if (
            kind in {"all", "null", "redacted", "unavailable"}
            and item["value"] is not None
        ):
            raise StrategyError(
                f"ImpactSlice dimension {name} must have null value"
            )
        if kind == "value" and item["value"] is None:
            raise StrategyError(
                f"ImpactSlice dimension {name} value is missing"
            )
        result[name] = item
    if result["partition"]["kind"] != "value":
        raise StrategyError("ImpactSlice partition must be a value")
    return result


def _validate_family_dimensions(
    *,
    family: str,
    dimensions: Mapping[str, Mapping[str, Any]],
    availability: object,
) -> None:
    active = _FAMILY_DIMENSIONS[family]
    non_partition = _DIMENSIONS - {"partition"}
    if availability == "present":
        for dimension in non_partition:
            kind = dimensions[dimension]["kind"]
            if dimension in active:
                if kind not in {"value", "null", "redacted"}:
                    raise StrategyError(
                        f"ImpactSlice {family} dimensions are invalid"
                    )
                if dimension == "new_action_bucket" and kind != "value":
                    raise StrategyError(
                        f"ImpactSlice {family} dimensions are invalid"
                    )
                if kind == "value":
                    _validate_dimension_value(
                        dimension,
                        dimensions[dimension]["value"],
                        family=family,
                    )
            elif kind != "all":
                raise StrategyError(
                    f"ImpactSlice {family} dimensions are invalid"
                )
        return
    if availability == "unavailable":
        unavailable_count = 0
        for dimension in non_partition:
            kind = dimensions[dimension]["kind"]
            if dimension in active:
                if kind not in {"all", "unavailable"}:
                    raise StrategyError(
                        f"ImpactSlice {family} dimensions are invalid"
                    )
                unavailable_count += int(kind == "unavailable")
            elif kind != "all":
                raise StrategyError(
                    f"ImpactSlice {family} dimensions are invalid"
                )
        if not active or unavailable_count == 0:
            raise StrategyError(
                f"ImpactSlice {family} unavailable dimensions are invalid"
            )


def _validate_dimension_value(
    dimension: str,
    value: object,
    *,
    family: str,
) -> None:
    if dimension == "month":
        month = _text(value, f"ImpactSlice {family} month")
        if (
            _MONTH_RE.fullmatch(month) is None
            or not 1 <= int(month[-2:]) <= 12
        ):
            raise StrategyError(
                f"ImpactSlice {family} dimensions month is invalid"
            )
        return
    if dimension == "new_action_bucket":
        _validate_action_bucket(
            value,
            name=f"ImpactSlice {family} new_action_bucket",
        )
        return
    _json_scalar(
        value,
        f"ImpactSlice {family} dimensions {dimension}",
    )


def _validate_typed_field(value: object, name: str) -> dict[str, Any]:
    obj = _json_object(value, name)
    _exact_fields(obj, _TYPED_FIELD_FIELDS, name)
    availability = obj["availability"]
    if availability == "present":
        if obj["value"] is None or obj["reason"] is not None:
            raise StrategyError(f"{name} present value is invalid")
    elif availability in {"unavailable", "not_applicable"}:
        if obj["value"] is not None:
            raise StrategyError(f"{name} unavailable value must be null")
        _text(obj["reason"], f"{name} reason")
    else:
        raise StrategyError(f"{name} availability is invalid")
    return obj


def _validate_population(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
    denominator: int | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    obj = _json_object(value, name)
    expected_fields = _POPULATION_FIELDS | {"share"}
    _exact_fields(obj, expected_fields, name)
    role = _text(obj["role"], f"{name}.role")
    if (
        role not in _POPULATION_ROLES
        or (expected_role is not None and role != expected_role)
    ):
        raise StrategyError(f"{name} population role is invalid")
    count = (
        _nonnegative_int(obj["count"], f"{name}.count")
        if allow_zero
        else _positive_int(obj["count"], f"{name}.count")
    )
    labeled = _nonnegative_int(
        obj["labeled_count"],
        f"{name}.labeled_count",
    )
    unlabeled = _nonnegative_int(
        obj["unlabeled_count"],
        f"{name}.unlabeled_count",
    )
    if labeled + unlabeled != count:
        raise StrategyError(f"{name} population counts do not conserve")
    coverage = _ratio_number(obj["label_coverage"], f"{name}.label_coverage")
    expected_coverage = 0.0 if count == 0 else labeled / count
    if not math.isclose(
        coverage,
        expected_coverage,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise StrategyError(f"{name} label coverage is inconsistent")
    share = _ratio_number(obj["share"], f"{name}.share")
    if denominator is not None:
        expected_share = 0.0 if denominator == 0 else count / denominator
        if not math.isclose(
            share,
            expected_share,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise StrategyError(f"{name} share is inconsistent")
    risk = _json_object(obj["risk"], f"{name}.risk")
    _exact_fields(risk, _RISK_FIELDS, f"{name}.risk")
    bad = _nonnegative_int(risk["bad_count"], f"{name}.risk.bad_count")
    if bad > labeled:
        raise StrategyError(f"{name} bad count exceeds labeled count")
    if role == "approval":
        if (
            labeled != 0
            or unlabeled != count
            or risk["availability"] != "not_applicable"
            or risk["reason"] != "approval_population"
            or bad != 0
            or risk["bad_rate"] is not None
        ):
            raise StrategyError(
                f"{name} approval risk must be not applicable"
            )
    elif labeled == 0:
        if (
            risk["availability"] != "unavailable"
            or risk["bad_rate"] is not None
            or risk["reason"] != "no_labeled_rows"
        ):
            raise StrategyError(f"{name} unlabeled risk must be unavailable")
    else:
        if risk["availability"] != "present" or risk["reason"] is not None:
            raise StrategyError(f"{name} labeled risk must be present")
        bad_rate = _ratio_number(
            risk["bad_rate"],
            f"{name}.risk.bad_rate",
        )
        if not math.isclose(
            bad_rate,
            bad / labeled,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise StrategyError(f"{name} bad rate is inconsistent")
    _validate_amounts(
        obj["amounts"],
        population_count=count,
        name=f"{name}.amounts",
    )
    return obj


def _validate_amounts(
    value: object,
    *,
    population_count: int,
    name: str,
) -> dict[str, Any]:
    obj = _json_object(value, name)
    _exact_fields(
        obj,
        frozenset({"loan_amount", "overdue_amount", "paired"}),
        name,
    )
    singles: dict[str, Mapping[str, Any]] = {}
    for key in ("loan_amount", "overdue_amount"):
        typed = _validate_typed_field(obj[key], f"{name}.{key}")
        if typed["availability"] == "unavailable":
            if typed["reason"] != f"{key}_field_not_bound":
                raise StrategyError(f"{name}.{key} reason is invalid")
            continue
        item = _json_object(typed["value"], f"{name}.{key}.value")
        _exact_fields(
            item,
            frozenset({"column", "covered_count", "coverage", "sum"}),
            f"{name}.{key}.value",
        )
        _text(item["column"], f"{name}.{key}.column")
        covered = _nonnegative_int(
            item["covered_count"],
            f"{name}.{key}.covered_count",
        )
        if covered > population_count:
            raise StrategyError(f"{name}.{key} coverage exceeds population")
        _require_finite_equal(
            item["coverage"],
            _ratio(covered, population_count),
            name=f"{name}.{key}.coverage",
        )
        amount_sum = float(
            _finite_number(item["sum"], f"{name}.{key}.sum")
        )
        if amount_sum < 0 or (covered == 0 and amount_sum != 0):
            raise StrategyError(f"{name}.{key} sum is invalid")
        singles[key] = item
    paired = _validate_typed_field(obj["paired"], f"{name}.paired")
    if paired["availability"] == "unavailable":
        if paired["reason"] != "paired_amount_fields_not_bound":
            raise StrategyError(f"{name}.paired reason is invalid")
        if len(singles) == 2:
            raise StrategyError(
                f"{name}.paired contradicts available amount fields"
            )
        return obj
    if len(singles) != 2:
        raise StrategyError(f"{name}.paired requires both amount fields")
    pair = _json_object(paired["value"], f"{name}.paired.value")
    _exact_fields(
        pair,
        frozenset(
            {
                "covered_count",
                "coverage",
                "loan_amount_sum",
                "overdue_amount_sum",
                "overdue_rate",
            }
        ),
        f"{name}.paired.value",
    )
    covered = _nonnegative_int(
        pair["covered_count"],
        f"{name}.paired.covered_count",
    )
    if covered > population_count or any(
        covered > int(item["covered_count"])
        for item in singles.values()
    ):
        raise StrategyError(f"{name}.paired coverage is invalid")
    _require_finite_equal(
        pair["coverage"],
        _ratio(covered, population_count),
        name=f"{name}.paired.coverage",
    )
    loan_sum = float(
        _finite_number(
            pair["loan_amount_sum"],
            f"{name}.paired.loan_amount_sum",
        )
    )
    overdue_sum = float(
        _finite_number(
            pair["overdue_amount_sum"],
            f"{name}.paired.overdue_amount_sum",
        )
    )
    if (
        loan_sum < 0
        or overdue_sum < 0
        or (covered == 0 and (loan_sum != 0 or overdue_sum != 0))
        or loan_sum > float(singles["loan_amount"]["sum"])
        or overdue_sum > float(singles["overdue_amount"]["sum"])
    ):
        raise StrategyError(f"{name}.paired sums are invalid")
    _require_optional_ratio(
        pair["overdue_rate"],
        numerator=overdue_sum,
        denominator=loan_sum,
        name=f"{name}.paired.overdue_rate",
    )
    return obj


def _population_counts(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        "count": int(value["count"]),
        "labeled_count": int(value["labeled_count"]),
        "bad_count": int(value["risk"]["bad_count"]),
    }


def _require_amount_rollup(
    parts: Sequence[Mapping[str, Any]],
    total: Mapping[str, Any],
    *,
    name: str,
) -> None:
    for key in ("loan_amount", "overdue_amount", "paired"):
        expected = total[key]
        observations = [part[key] for part in parts]
        if any(
            item["availability"] != expected["availability"]
            or item["reason"] != expected["reason"]
            for item in observations
        ):
            raise StrategyError(f"{name} {key} availability does not roll up")
        if expected["availability"] != "present":
            continue
        expected_value = expected["value"]
        values = [item["value"] for item in observations]
        if key != "paired" and any(
            item["column"] != expected_value["column"] for item in values
        ):
            raise StrategyError(f"{name} {key} column changed")
        if sum(int(item["covered_count"]) for item in values) != int(
            expected_value["covered_count"]
        ):
            raise StrategyError(f"{name} {key} coverage does not roll up")
        sum_fields = (
            ("loan_amount_sum", "overdue_amount_sum")
            if key == "paired"
            else ("sum",)
        )
        for field in sum_fields:
            actual = sum(float(item[field]) for item in values)
            if not math.isclose(
                actual,
                float(expected_value[field]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise StrategyError(
                    f"{name} {key} {field} does not roll up"
                )


def _validate_projection(
    value: object,
    *,
    name: str,
    expected_type: str,
    expected_role: str,
) -> dict[str, Any]:
    obj = _json_object(value, name)
    _exact_fields(obj, _PROJECTION_FIELDS, name)
    _text(obj["strategy_id"], f"{name}.strategy_id")
    if obj["strategy_type"] != expected_type:
        raise StrategyError(f"{name} strategy_type changed")
    if obj["population_role"] != expected_role:
        raise StrategyError(f"{name} population_role changed")
    population = _positive_int(
        obj["population_count"],
        f"{name}.population_count",
    )
    labeled = _nonnegative_int(
        obj["labeled_count"],
        f"{name}.labeled_count",
    )
    if labeled > population:
        raise StrategyError(f"{name} labeled_count exceeds population")
    coverage = _ratio_number(
        obj["label_coverage"],
        f"{name}.label_coverage",
    )
    if not math.isclose(
        coverage,
        labeled / population,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise StrategyError(f"{name} label coverage is inconsistent")
    metrics = _json_object(obj["metrics"], f"{name}.metrics")
    breakdown = _list(obj["breakdown"], f"{name}.breakdown")
    metric_fields, breakdown_fields = _projection_fields(expected_type)
    _exact_fields(metrics, metric_fields, f"{name}.metrics")
    if not breakdown:
        raise StrategyError(f"{name} breakdown is empty")
    total = 0
    for index, raw in enumerate(breakdown):
        row = _json_object(raw, f"{name}.breakdown[{index}]")
        _exact_fields(
            row,
            breakdown_fields,
            f"{name}.breakdown[{index}]",
        )
        total += _nonnegative_int(
            row.get("count"),
            f"{name}.breakdown[{index}].count",
        )
    if expected_type in {"approval", "reject"} and [
        row["action"] for row in breakdown
    ] != list(_ACTION_ORDER):
        raise StrategyError(f"{name} breakdown actions are invalid")
    if total != population:
        raise StrategyError(f"{name} breakdown population is inconsistent")
    if expected_type in {"limit", "pricing"}:
        if _nonnegative_int(metrics["count"], f"{name}.metrics.count") != population:
            raise StrategyError(f"{name} metrics population is inconsistent")
    _validate_projection_values(
        metrics=metrics,
        breakdown=breakdown,
        strategy_type=expected_type,
        population=population,
        labeled=labeled,
        name=name,
    )
    warnings = _list(obj["warnings"], f"{name}.warnings")
    if any(not isinstance(item, str) for item in warnings):
        raise StrategyError(f"{name} warnings must contain strings")
    return obj


def _validate_projection_values(
    *,
    metrics: Mapping[str, Any],
    breakdown: Sequence[Mapping[str, Any]],
    strategy_type: str,
    population: int,
    labeled: int,
    name: str,
) -> None:
    if strategy_type in {"approval", "reject"}:
        _validate_action_projection(
            metrics=metrics,
            breakdown=breakdown,
            strategy_type=strategy_type,
            population=population,
            labeled=labeled,
            name=name,
        )
        return
    if strategy_type == "limit":
        _validate_limit_projection(
            metrics=metrics,
            breakdown=breakdown,
            population=population,
            labeled=labeled,
            name=name,
        )
        return
    if strategy_type == "pricing":
        _validate_pricing_projection(
            metrics=metrics,
            breakdown=breakdown,
            population=population,
            labeled=labeled,
            name=name,
        )
        return
    _validate_segmentation_projection(
        metrics=metrics,
        breakdown=breakdown,
        population=population,
        labeled=labeled,
        name=name,
    )


def _validate_projection_risk_row(
    row: Mapping[str, Any],
    *,
    population: int,
    name: str,
    share_field: str,
    allow_zero: bool,
) -> tuple[int, int, int]:
    count = (
        _nonnegative_int(row["count"], f"{name}.count")
        if allow_zero
        else _positive_int(row["count"], f"{name}.count")
    )
    labeled = _nonnegative_int(
        row["labeled_count"],
        f"{name}.labeled_count",
    )
    if labeled > count:
        raise StrategyError(f"{name} labeled_count exceeds count")
    bad = _nonnegative_int(row["bad_count"], f"{name}.bad_count")
    if bad > labeled:
        raise StrategyError(f"{name} bad_count exceeds labeled_count")
    _require_optional_ratio(
        row["bad_rate"],
        numerator=bad,
        denominator=labeled,
        name=f"{name}.bad_rate",
    )
    share = _ratio_number(row[share_field], f"{name}.{share_field}")
    expected_share = count / population
    if not math.isclose(
        share,
        expected_share,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise StrategyError(f"{name} {share_field} is inconsistent")
    return count, labeled, bad


def _validate_action_projection(
    *,
    metrics: Mapping[str, Any],
    breakdown: Sequence[Mapping[str, Any]],
    strategy_type: str,
    population: int,
    labeled: int,
    name: str,
) -> None:
    total_labeled = 0
    total_bad = 0
    by_action: dict[str, tuple[int, int, int]] = {}
    for index, row in enumerate(breakdown):
        row_name = f"{name}.breakdown[{index}]"
        action = _text(row["action"], f"{row_name}.action")
        counts = _validate_projection_risk_row(
            row,
            population=population,
            name=row_name,
            share_field="rate",
            allow_zero=True,
        )
        by_action[action] = counts
        count, action_labeled, bad = counts
        total_labeled += action_labeled
        total_bad += bad
        for suffix, expected in (
            ("count", count),
            ("labeled_count", action_labeled),
            ("bad_count", bad),
        ):
            if (
                _nonnegative_int(
                    metrics[f"{action}_{suffix}"],
                    f"{name}.metrics.{action}_{suffix}",
                )
                != expected
            ):
                raise StrategyError(
                    f"{name} metrics {action}_{suffix} is inconsistent"
                )
        _require_metric_ratio_equal(
            metrics[f"{action}_rate"],
            row["rate"],
            name=f"{name}.metrics.{action}_rate",
        )
        _require_metric_ratio_equal(
            metrics[f"{action}_bad_rate"],
            row["bad_rate"],
            name=f"{name}.metrics.{action}_bad_rate",
        )
    if total_labeled != labeled:
        raise StrategyError(f"{name} breakdown labeled population is inconsistent")
    if (
        _nonnegative_int(
            metrics["overall_bad_count"],
            f"{name}.metrics.overall_bad_count",
        )
        != total_bad
    ):
        raise StrategyError(f"{name} overall_bad_count is inconsistent")
    _require_optional_ratio(
        metrics["overall_bad_rate"],
        numerator=total_bad,
        denominator=labeled,
        name=f"{name}.metrics.overall_bad_rate",
    )
    if strategy_type == "reject":
        reject_count, reject_labeled, reject_bad = by_action["reject"]
        del reject_count
        _require_optional_ratio(
            metrics["bad_capture_rate"],
            numerator=reject_bad,
            denominator=total_bad,
            name=f"{name}.metrics.bad_capture_rate",
        )
        _require_optional_ratio(
            metrics["good_reject_rate"],
            numerator=reject_labeled - reject_bad,
            denominator=labeled - total_bad,
            name=f"{name}.metrics.good_reject_rate",
        )


def _validate_limit_projection(
    *,
    metrics: Mapping[str, Any],
    breakdown: Sequence[Mapping[str, Any]],
    population: int,
    labeled: int,
    name: str,
) -> None:
    assigned: list[float] = []
    total_labeled = 0
    weighted_total = 0.0
    for index, row in enumerate(breakdown):
        row_name = f"{name}.breakdown[{index}]"
        value = float(
            _finite_number(row["assigned_limit"], f"{row_name}.assigned_limit")
        )
        if value < 0:
            raise StrategyError(f"{row_name}.assigned_limit must be non-negative")
        count, row_labeled, _bad = _validate_projection_risk_row(
            row,
            population=population,
            name=row_name,
            share_field="share",
            allow_zero=False,
        )
        assigned.append(value)
        total_labeled += row_labeled
        weighted_total += value * count
    if assigned != sorted(set(assigned)):
        raise StrategyError(f"{name} limit breakdown is not canonical")
    if total_labeled != labeled:
        raise StrategyError(f"{name} breakdown labeled population is inconsistent")
    _require_finite_equal(
        metrics["total_limit"],
        weighted_total,
        name=f"{name}.metrics.total_limit",
    )
    _require_finite_equal(
        metrics["mean_limit"],
        weighted_total / population,
        name=f"{name}.metrics.mean_limit",
    )
    _require_finite_equal(
        metrics["min_limit"],
        assigned[0],
        name=f"{name}.metrics.min_limit",
    )
    _require_finite_equal(
        metrics["max_limit"],
        assigned[-1],
        name=f"{name}.metrics.max_limit",
    )
    _validate_optional_change_metrics(
        metrics,
        count_fields=("up_count", "down_count", "unchanged_count"),
        delta_field="total_limit_delta",
        population=population,
        name=name,
    )


def _validate_pricing_projection(
    *,
    metrics: Mapping[str, Any],
    breakdown: Sequence[Mapping[str, Any]],
    population: int,
    labeled: int,
    name: str,
) -> None:
    assigned: list[float] = []
    total_labeled = 0
    weighted_total = 0.0
    for index, row in enumerate(breakdown):
        row_name = f"{name}.breakdown[{index}]"
        value = _ratio_number(
            row["assigned_rate"],
            f"{row_name}.assigned_rate",
        )
        count, row_labeled, _bad = _validate_projection_risk_row(
            row,
            population=population,
            name=row_name,
            share_field="share",
            allow_zero=False,
        )
        assigned.append(value)
        total_labeled += row_labeled
        weighted_total += value * count
    if assigned != sorted(set(assigned)):
        raise StrategyError(f"{name} pricing breakdown is not canonical")
    if total_labeled != labeled:
        raise StrategyError(f"{name} breakdown labeled population is inconsistent")
    _require_finite_equal(
        metrics["mean_rate"],
        weighted_total / population,
        name=f"{name}.metrics.mean_rate",
    )
    _validate_optional_change_metrics(
        metrics,
        count_fields=(
            "repriced_up_count",
            "repriced_down_count",
            "unchanged_count",
        ),
        delta_field=None,
        population=population,
        name=name,
    )


def _validate_segmentation_projection(
    *,
    metrics: Mapping[str, Any],
    breakdown: Sequence[Mapping[str, Any]],
    population: int,
    labeled: int,
    name: str,
) -> None:
    tokens: list[str] = []
    total_labeled = 0
    total_bad = 0
    validated_rows: list[tuple[Mapping[str, Any], int, int]] = []
    for index, row in enumerate(breakdown):
        row_name = f"{name}.breakdown[{index}]"
        try:
            action = StrategyAction(type="segment", value=row["segment"])
        except StrategyError as exc:
            raise StrategyError(f"{row_name}.segment is invalid: {exc}") from exc
        token = _canonical_json(action.value)
        tokens.append(token)
        _count, row_labeled, bad = _validate_projection_risk_row(
            row,
            population=population,
            name=row_name,
            share_field="share",
            allow_zero=False,
        )
        total_labeled += row_labeled
        total_bad += bad
        validated_rows.append((row, row_labeled, bad))
    if tokens != sorted(set(tokens)):
        raise StrategyError(f"{name} segmentation breakdown is not canonical")
    if total_labeled != labeled:
        raise StrategyError(f"{name} breakdown labeled population is inconsistent")
    if (
        _positive_int(
            metrics["segment_count"],
            f"{name}.metrics.segment_count",
        )
        != len(breakdown)
        or _nonnegative_int(
            metrics["overall_bad_count"],
            f"{name}.metrics.overall_bad_count",
        )
        != total_bad
    ):
        raise StrategyError(f"{name} segmentation metrics are inconsistent")
    _require_optional_ratio(
        metrics["overall_bad_rate"],
        numerator=total_bad,
        denominator=labeled,
        name=f"{name}.metrics.overall_bad_rate",
    )
    overall_bad_rate = None if labeled == 0 else total_bad / labeled
    for index, (row, row_labeled, bad) in enumerate(validated_rows):
        bad_rate = None if row_labeled == 0 else bad / row_labeled
        expected_lift = (
            None
            if bad_rate is None or overall_bad_rate in {None, 0.0}
            else bad_rate / overall_bad_rate
        )
        _require_optional_finite_equal(
            row["lift"],
            expected_lift,
            name=f"{name}.breakdown[{index}].lift",
            nonnegative=True,
        )


def _validate_optional_change_metrics(
    metrics: Mapping[str, Any],
    *,
    count_fields: Sequence[str],
    delta_field: str | None,
    population: int,
    name: str,
) -> None:
    raw_counts = [metrics[field] for field in count_fields]
    if all(value is None for value in raw_counts):
        if delta_field is not None and metrics[delta_field] is not None:
            raise StrategyError(f"{name} baseline change metrics are inconsistent")
        return
    if any(value is None for value in raw_counts):
        raise StrategyError(f"{name} baseline change metrics are inconsistent")
    counts = [
        _nonnegative_int(value, f"{name}.metrics.{field}")
        for field, value in zip(count_fields, raw_counts, strict=True)
    ]
    if sum(counts) != population:
        raise StrategyError(f"{name} baseline change counts are inconsistent")
    if delta_field is not None:
        _finite_number(metrics[delta_field], f"{name}.metrics.{delta_field}")


def _require_optional_ratio(
    value: object,
    *,
    numerator: int,
    denominator: int,
    name: str,
) -> None:
    if denominator == 0:
        if value is not None:
            raise StrategyError(f"{name} must be null without a denominator")
        return
    actual = _ratio_number(value, name)
    expected = numerator / denominator
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise StrategyError(f"{name} is inconsistent")


def _require_metric_ratio_equal(
    value: object,
    expected: object,
    *,
    name: str,
) -> None:
    if expected is None:
        if value is not None:
            raise StrategyError(f"{name} is inconsistent")
        return
    actual = _ratio_number(value, name)
    if not math.isclose(
        actual,
        float(expected),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise StrategyError(f"{name} is inconsistent")


def _require_finite_equal(value: object, expected: float, *, name: str) -> None:
    actual = float(_finite_number(value, name))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise StrategyError(f"{name} is inconsistent")


def _require_optional_finite_equal(
    value: object,
    expected: float | None,
    *,
    name: str,
    nonnegative: bool = False,
) -> None:
    if expected is None:
        if value is not None:
            raise StrategyError(f"{name} is inconsistent")
        return
    actual = float(_finite_number(value, name))
    if nonnegative and actual < 0:
        raise StrategyError(f"{name} must be non-negative")
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise StrategyError(f"{name} is inconsistent")


def _projection_fields(
    strategy_type: str,
) -> tuple[frozenset[str], frozenset[str]]:
    if strategy_type == "approval":
        return _ACTION_METRIC_FIELDS, _ACTION_BREAKDOWN_FIELDS
    if strategy_type == "reject":
        return (
            _ACTION_METRIC_FIELDS
            | {"bad_capture_rate", "good_reject_rate"},
            _ACTION_BREAKDOWN_FIELDS,
        )
    if strategy_type == "limit":
        return _LIMIT_METRIC_FIELDS, _LIMIT_BREAKDOWN_FIELDS
    if strategy_type == "pricing":
        return _PRICING_METRIC_FIELDS, _PRICING_BREAKDOWN_FIELDS
    return _SEGMENTATION_METRIC_FIELDS, _SEGMENTATION_BREAKDOWN_FIELDS


def _require_projection_population(
    projection: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    name: str,
) -> None:
    if (
        projection["population_count"] != population["count"]
        or projection["labeled_count"] != population["labeled_count"]
        or not math.isclose(
            projection["label_coverage"],
            population["label_coverage"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise StrategyError(f"{name} population does not match slice population")
    projected_bad = sum(
        _nonnegative_int(
            row["bad_count"],
            f"{name}.breakdown bad_count",
        )
        for row in projection["breakdown"]
    )
    if projected_bad != population["risk"]["bad_count"]:
        raise StrategyError(
            f"{name} bad population does not match slice population"
        )


def _validate_transition(
    value: object,
    *,
    population: Mapping[str, Any],
    strategy_type: str,
    population_role: str,
) -> None:
    obj = _json_object(value, "ImpactSlice transition value")
    _exact_fields(
        obj,
        frozenset({"rows"}),
        "ImpactSlice transition value",
    )
    rows = _list(obj["rows"], "ImpactSlice transition rows")
    if not rows:
        raise StrategyError("ImpactSlice transition rows are empty")
    population_count = population["count"]
    totals = {"count": 0, "labeled_count": 0, "bad_count": 0}
    effects: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _json_object(raw, f"ImpactSlice transition rows[{index}]")
        _exact_fields(
            row,
            frozenset(
                {"from_bucket", "to_bucket", "direction", "effect"}
            ),
            f"ImpactSlice transition rows[{index}]",
        )
        old_bucket = _validate_action_bucket(
            row["from_bucket"],
            name=f"ImpactSlice transition rows[{index}].from_bucket",
            strategy_type=strategy_type,
        )
        new_bucket = _validate_action_bucket(
            row["to_bucket"],
            name=f"ImpactSlice transition rows[{index}].to_bucket",
            strategy_type=strategy_type,
        )
        token = _canonical_json(
            {
                "from": old_bucket,
                "to": new_bucket,
            }
        )
        if token in seen:
            raise StrategyError("ImpactSlice transition rows are duplicated")
        seen.add(token)
        direction = _text(
            row["direction"],
            "ImpactSlice transition direction",
        )
        if direction != _transition_direction(
            strategy_type,
            old_bucket,
            new_bucket,
        ):
            raise StrategyError(
                "ImpactSlice transition direction is inconsistent"
            )
        effect = _validate_population(
            row["effect"],
            f"ImpactSlice transition rows[{index}].effect",
            denominator=population_count,
            expected_role=population_role,
        )
        for field, value in _population_counts(effect).items():
            totals[field] += value
        effects.append(effect)
    if totals != _population_counts(population):
        raise StrategyError("ImpactSlice transition population is inconsistent")
    _require_amount_rollup(
        [effect["amounts"] for effect in effects],
        population["amounts"],
        name="ImpactSlice transition amounts",
    )


def _validate_action_bucket(
    value: object,
    *,
    name: str,
    strategy_type: str | None = None,
) -> dict[str, Any]:
    obj = _json_object(value, name)
    _validate_action(obj, name=name)
    if (
        strategy_type is not None
        and obj["type"] not in _STRATEGY_ACTION_TYPES[strategy_type]
    ):
        raise StrategyError(
            f"{name} action type is invalid for {strategy_type}"
        )
    return obj


def _validate_waterfall(
    value: object,
    *,
    population: Mapping[str, Any],
    strategy_type: str,
    population_role: str,
) -> None:
    population_count = population["count"]
    obj = _json_object(value, "ImpactSlice waterfall value")
    _exact_fields(
        obj,
        frozenset({"entries", "default_unmatched"}),
        "ImpactSlice waterfall value",
    )
    entries = _list(obj["entries"], "ImpactSlice waterfall entries")
    if not entries or len(entries) > MAX_IMPACT_CUBE_RULES:
        raise StrategyError("ImpactSlice waterfall rule budget is invalid")
    incremental_total = {"count": 0, "labeled_count": 0, "bad_count": 0}
    previous_remaining = _population_counts(population)
    previous_remaining_population = population
    incremental_effects: list[Mapping[str, Any]] = []
    for index, raw in enumerate(entries):
        row = _json_object(raw, f"ImpactSlice waterfall[{index}]")
        _exact_fields(
            row,
            frozenset(
                {
                    "position",
                    "entry_id",
                    "rule_id",
                    "source_ref",
                    "action",
                    "standalone",
                    "incremental",
                    "shadowed",
                    "remaining_after",
                }
            ),
            f"ImpactSlice waterfall[{index}]",
        )
        if row["position"] != index + 1:
            raise StrategyError("ImpactSlice waterfall position is invalid")
        _text(
            row["entry_id"],
            f"ImpactSlice waterfall[{index}].entry_id",
        )
        _text(
            row["rule_id"],
            f"ImpactSlice waterfall[{index}].rule_id",
        )
        _validate_waterfall_source_ref(
            row["source_ref"],
            name=f"ImpactSlice waterfall[{index}].source_ref",
        )
        _validate_action(
            row["action"],
            name=f"ImpactSlice waterfall[{index}].action",
        )
        _require_strategy_action_type(
            row["action"],
            strategy_type=strategy_type,
            name=f"ImpactSlice waterfall[{index}].action",
        )
        standalone = _validate_population(
            row["standalone"],
            f"ImpactSlice waterfall[{index}].standalone",
            allow_zero=True,
            denominator=population_count,
            expected_role=population_role,
        )
        incremental = _validate_population(
            row["incremental"],
            f"ImpactSlice waterfall[{index}].incremental",
            allow_zero=True,
            denominator=population_count,
            expected_role=population_role,
        )
        shadowed = _validate_population(
            row["shadowed"],
            f"ImpactSlice waterfall[{index}].shadowed",
            allow_zero=True,
            denominator=population_count,
            expected_role=population_role,
        )
        remaining = _validate_population(
            row["remaining_after"],
            f"ImpactSlice waterfall[{index}].remaining",
            allow_zero=True,
            denominator=population_count,
            expected_role=population_role,
        )
        standalone_counts = _population_counts(standalone)
        incremental_counts = _population_counts(incremental)
        shadowed_counts = _population_counts(shadowed)
        remaining_counts = _population_counts(remaining)
        if standalone_counts != {
            field: incremental_counts[field] + shadowed_counts[field]
            for field in standalone_counts
        }:
            raise StrategyError(
                "ImpactSlice waterfall standalone population is inconsistent"
            )
        _require_amount_rollup(
            [incremental["amounts"], shadowed["amounts"]],
            standalone["amounts"],
            name="ImpactSlice waterfall standalone amounts",
        )
        if remaining_counts != {
            field: previous_remaining[field] - incremental_counts[field]
            for field in previous_remaining
        }:
            raise StrategyError(
                "ImpactSlice waterfall remaining population is inconsistent"
            )
        _require_amount_rollup(
            [incremental["amounts"], remaining["amounts"]],
            previous_remaining_population["amounts"],
            name="ImpactSlice waterfall remaining amounts",
        )
        previous_remaining = remaining_counts
        previous_remaining_population = remaining
        incremental_effects.append(incremental)
        for field in incremental_total:
            incremental_total[field] += incremental_counts[field]
    default = _json_object(
        obj["default_unmatched"],
        "ImpactSlice default unmatched",
    )
    _exact_fields(
        default,
        frozenset({"action", "effect"}),
        "ImpactSlice default unmatched",
    )
    _validate_action(
        default["action"],
        name="ImpactSlice default unmatched action",
    )
    _require_strategy_action_type(
        default["action"],
        strategy_type=strategy_type,
        name="ImpactSlice default unmatched action",
    )
    default_effect = _validate_population(
        default["effect"],
        "ImpactSlice default unmatched effect",
        allow_zero=True,
        denominator=population_count,
        expected_role=population_role,
    )
    default_counts = _population_counts(default_effect)
    if (
        {
            field: incremental_total[field] + default_counts[field]
            for field in incremental_total
        }
        != _population_counts(population)
        or default_counts != previous_remaining
    ):
        raise StrategyError("ImpactSlice waterfall population is inconsistent")
    _require_amount_rollup(
        [
            *(effect["amounts"] for effect in incremental_effects),
            default_effect["amounts"],
        ],
        population["amounts"],
        name="ImpactSlice waterfall population amounts",
    )


def _validate_waterfall_source_ref(
    value: object,
    *,
    name: str,
) -> None:
    obj = _json_object(value, name)
    _exact_fields(
        obj,
        frozenset(
            {
                "artifact_id",
                "artifact_content_hash",
                "asset_id",
                "asset_hash",
                "fragment_id",
            }
        ),
        name,
    )
    for field in ("artifact_id", "asset_id", "fragment_id"):
        _text(obj[field], f"{name}.{field}")
    for field in ("artifact_content_hash", "asset_hash"):
        _hash(obj[field], f"{name}.{field}")


def _validate_action(value: object, *, name: str) -> None:
    obj = _json_object(value, name)
    try:
        action = StrategyAction.from_dict(obj)
    except StrategyError as exc:
        raise StrategyError(f"{name} is invalid: {exc}") from exc
    if action.to_dict() != obj:
        raise StrategyError(f"{name} is not canonical")


def _require_strategy_action_type(
    value: Mapping[str, Any],
    *,
    strategy_type: str,
    name: str,
) -> None:
    if value["type"] not in _STRATEGY_ACTION_TYPES[strategy_type]:
        raise StrategyError(
            f"{name} action type is invalid for {strategy_type}"
        )


def _validate_economics_value(
    value: object,
    *,
    strategy_type: str,
) -> None:
    obj = _json_object(value, "ImpactSlice economics value")
    _exact_fields(
        obj,
        frozenset({"new", "current", "delta"}),
        "ImpactSlice economics value",
    )
    new = _json_object(obj["new"], "ImpactSlice new economics")
    expected_fields = _ECONOMICS_FIELDS[strategy_type]
    _exact_fields(
        new,
        expected_fields,
        "ImpactSlice new economics",
    )
    _validate_economics_scalars(
        new,
        name="ImpactSlice new economics",
    )
    _validate_economics_identity(
        new,
        strategy_type=strategy_type,
        name="ImpactSlice new economics",
    )
    current = None
    if obj["current"] is not None:
        current = _json_object(
            obj["current"],
            "ImpactSlice current economics",
        )
        _exact_fields(
            current,
            expected_fields,
            "ImpactSlice current economics",
        )
        _validate_economics_scalars(
            current,
            name="ImpactSlice current economics",
        )
        _validate_economics_identity(
            current,
            strategy_type=strategy_type,
            name="ImpactSlice current economics",
        )
    delta = _json_object(obj["delta"], "ImpactSlice economics delta")
    expected_delta: dict[str, float] = {}
    if current is not None:
        for key in sorted(expected_fields):
            new_value = new[key]
            old_value = current[key]
            if (
                isinstance(new_value, Real)
                and not isinstance(new_value, bool)
                and isinstance(old_value, Real)
                and not isinstance(old_value, bool)
            ):
                expected_delta[key] = float(new_value) - float(old_value)
    if set(delta) != set(expected_delta):
        raise StrategyError("ImpactSlice economics delta fields changed")
    for key, expected in expected_delta.items():
        actual = float(
            _finite_number(
                delta[key],
                f"ImpactSlice economics delta {key}",
            )
        )
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise StrategyError(
                f"ImpactSlice economics delta {key} is inconsistent"
            )
    if (
        strategy_type in {"approval", "reject", "pricing"}
        and current is not None
    ):
        _require_finite_equal(
            new["baseline_profit"],
            float(current["profit"]),
            name=f"ImpactSlice {strategy_type} baseline_profit",
        )
        _require_finite_equal(
            new["profit_delta_vs_baseline"],
            float(delta["profit"]),
            name=(
                f"ImpactSlice {strategy_type} "
                "profit_delta_vs_baseline"
            ),
        )


def _validate_economics_identity(
    values: Mapping[str, Any],
    *,
    strategy_type: str,
    name: str,
) -> None:
    if strategy_type == "limit":
        for field in ("expected_ead", "expected_loss"):
            if float(values[field]) < 0:
                raise StrategyError(f"{name}.{field} must be non-negative")
        return
    if strategy_type == "segmentation":
        return
    profit = float(values["profit"])
    expected_profit = (
        float(values["revenue"])
        - float(values["expected_loss"])
        - float(values["funding_cost"])
        - float(values["operating_cost"])
    )
    if not math.isclose(
        profit,
        expected_profit,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise StrategyError(f"{name} profit identity is inconsistent")
    total_ead = float(values["total_ead"])
    if total_ead < 0:
        raise StrategyError(f"{name}.total_ead must be non-negative")
    if total_ead == 0:
        if values["roa"] is not None or values["ead_weighted_rate"] is not None:
            raise StrategyError(
                f"{name} zero-EAD derived economics are inconsistent"
            )
    else:
        _require_finite_equal(
            values["roa"],
            profit / total_ead,
            name=f"{name}.roa",
        )
        if values["ead_weighted_rate"] is None:
            raise StrategyError(
                f"{name}.ead_weighted_rate requires positive EAD"
            )
    baseline = values["baseline_profit"]
    baseline_delta = values["profit_delta_vs_baseline"]
    if (baseline is None) is not (baseline_delta is None):
        raise StrategyError(
            f"{name} baseline profit fields are inconsistent"
        )
    if baseline is not None:
        _require_finite_equal(
            baseline_delta,
            profit - float(baseline),
            name=f"{name}.profit_delta_vs_baseline",
        )
    if strategy_type in {"approval", "reject"}:
        _require_finite_equal(
            values["expected_profit"],
            profit,
            name=f"{name}.expected_profit",
        )
        if values["profit_note"] is not None:
            raise StrategyError(f"{name}.profit_note must be null")


def _validate_economics_scalars(
    values: Mapping[str, Any],
    *,
    name: str,
) -> None:
    for key, value in values.items():
        if key == "profit_note":
            if value is not None:
                _text(value, f"{name}.{key}")
            continue
        if value is not None:
            _finite_number(value, f"{name}.{key}")


def _validate_rollups(
    *,
    partitions: Sequence[Mapping[str, Any]],
    families: Mapping[str, Mapping[str, Any]],
    slices: Sequence[Mapping[str, Any]],
) -> None:
    for partition_row in partitions:
        partition = partition_row["name"]
        role = partition_row["role"]
        overall = [
            row
            for row in slices
            if row["family"] == "overall"
            and row["population_role"] == role
            and row["dimensions"]["partition"]["value"] == partition
        ]
        if len(overall) != 1 or overall[0]["availability"] != "present":
            raise StrategyError(
                f"ImpactCube {partition} overall slice is missing"
            )
        base = overall[0]["population"]["value"]
        if base["role"] != role:
            raise StrategyError(
                f"ImpactCube {role}/{partition} population role changed"
            )
        if base["count"] != partition_row["row_count"]:
            raise StrategyError(
                f"ImpactCube {partition} population does not match binding"
            )
        for family in _FAMILY_ORDER:
            rows = [
                row
                for row in slices
                if row["family"] == family
                and row["population_role"] == role
                and row["dimensions"]["partition"]["value"] == partition
            ]
            if families[family]["availability"] == "unavailable":
                if (
                    len(rows) != 1
                    or rows[0]["availability"] != "unavailable"
                ):
                    raise StrategyError(
                        f"ImpactCube {partition} {family} unavailable slice changed"
                    )
                continue
            if not rows or any(
                row["availability"] != "present" for row in rows
            ):
                raise StrategyError(
                    f"ImpactCube {partition} {family} slices are incomplete"
                )
            population = sum(
                row["population"]["value"]["count"] for row in rows
            )
            labeled = sum(
                row["population"]["value"]["labeled_count"] for row in rows
            )
            bad = sum(
                row["population"]["value"]["risk"]["bad_count"]
                for row in rows
            )
            if (
                population != base["count"]
                or labeled != base["labeled_count"]
                or bad != base["risk"]["bad_count"]
            ):
                raise StrategyError(
                    f"ImpactCube {partition} {family} population does not roll up"
                )
            _require_amount_rollup(
                [
                    row["population"]["value"]["amounts"]
                    for row in rows
                ],
                base["amounts"],
                name=(
                    f"ImpactCube {role}/{partition} {family} amounts"
                ),
            )
            _require_economics_rollup(
                rows,
                overall[0],
                name=(
                    f"ImpactCube {role}/{partition} {family} economics"
                ),
            )


def _require_economics_rollup(
    parts: Sequence[Mapping[str, Any]],
    total: Mapping[str, Any],
    *,
    name: str,
) -> None:
    expected = total["economics"]
    observations = [row["economics"] for row in parts]
    if any(
        item["availability"] != expected["availability"]
        or item["reason"] != expected["reason"]
        for item in observations
    ):
        raise StrategyError(f"{name} availability does not roll up")
    if expected["availability"] != "present":
        return
    if any(item["value"] is None for item in observations):
        raise StrategyError(f"{name} values are incomplete")
    strategy_type = total["new"]["value"]["strategy_type"]
    additive = {
        "approval": (
            "total_ead",
            "revenue",
            "expected_loss",
            "funding_cost",
            "operating_cost",
            "profit",
            "expected_profit",
            "baseline_profit",
            "profit_delta_vs_baseline",
        ),
        "reject": (
            "total_ead",
            "revenue",
            "expected_loss",
            "funding_cost",
            "operating_cost",
            "profit",
            "expected_profit",
            "baseline_profit",
            "profit_delta_vs_baseline",
        ),
        "limit": ("expected_ead", "expected_loss"),
        "pricing": (
            "total_ead",
            "revenue",
            "expected_loss",
            "funding_cost",
            "operating_cost",
            "profit",
            "baseline_profit",
            "profit_delta_vs_baseline",
        ),
        "segmentation": (),
    }[strategy_type]
    total_value = expected["value"]
    part_values = [item["value"] for item in observations]
    for side in ("new", "current"):
        expected_side = total_value[side]
        child_sides = [item[side] for item in part_values]
        if expected_side is None:
            if any(item is not None for item in child_sides):
                raise StrategyError(f"{name} {side} availability does not roll")
            continue
        if any(item is None for item in child_sides):
            raise StrategyError(f"{name} {side} values do not roll")
        for field in additive:
            expected_field = expected_side[field]
            child_fields = [item[field] for item in child_sides]
            if expected_field is None:
                if any(item is not None for item in child_fields):
                    raise StrategyError(
                        f"{name} {side}.{field} availability does not roll"
                    )
                continue
            if any(item is None for item in child_fields):
                raise StrategyError(
                    f"{name} {side}.{field} values do not roll"
                )
            actual = sum(float(item) for item in child_fields)
            if not math.isclose(
                actual,
                float(expected_field),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise StrategyError(
                    f"{name} {side}.{field} does not roll up"
                )
        if strategy_type in {"approval", "reject", "pricing"}:
            total_ead = float(expected_side["total_ead"])
            weighted = 0.0
            for child in child_sides:
                child_ead = float(child["total_ead"])
                rate = child["ead_weighted_rate"]
                if child_ead:
                    if rate is None:
                        raise StrategyError(
                            f"{name} {side}.ead_weighted_rate is missing"
                        )
                    weighted += float(rate) * child_ead
            expected_rate = (
                None if total_ead == 0 else weighted / total_ead
            )
            if expected_rate is None:
                if expected_side["ead_weighted_rate"] is not None:
                    raise StrategyError(
                        f"{name} {side}.ead_weighted_rate does not roll up"
                    )
            else:
                _require_finite_equal(
                    expected_side["ead_weighted_rate"],
                    expected_rate,
                    name=f"{name} {side}.ead_weighted_rate",
                )
    expected_delta = total_value["delta"]
    child_deltas = [item["delta"] for item in part_values]
    for field in additive:
        if field not in expected_delta:
            if any(field in item for item in child_deltas):
                raise StrategyError(
                    f"{name} delta.{field} availability does not roll"
                )
            continue
        if any(field not in item for item in child_deltas):
            raise StrategyError(f"{name} delta.{field} values do not roll")
        actual = sum(float(item[field]) for item in child_deltas)
        if not math.isclose(
            actual,
            float(expected_delta[field]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise StrategyError(f"{name} delta.{field} does not roll up")


def _validate_new_action_consistency(
    *,
    slices: Sequence[Mapping[str, Any]],
) -> None:
    for role in _POPULATION_ROLE_ORDER:
        for partition in _PARTITION_ORDER:
            overall = next(
                (
                    row
                    for row in slices
                    if row["population_role"] == role
                    and row["family"] == "overall"
                    and row["dimensions"]["partition"]["value"] == partition
                ),
                None,
            )
            if overall is None:
                continue
            waterfall = overall["waterfall"]["value"]
            expected: dict[str, list[Mapping[str, Any]]] = {}
            actions: dict[str, Mapping[str, Any]] = {}
            for entry in waterfall["entries"]:
                token = _canonical_json(entry["action"])
                actions[token] = entry["action"]
                expected.setdefault(token, []).append(entry["incremental"])
            default = waterfall["default_unmatched"]
            token = _canonical_json(default["action"])
            actions[token] = default["action"]
            expected.setdefault(token, []).append(default["effect"])
            expected = {
                action_token: effects
                for action_token, effects in expected.items()
                if sum(effect["count"] for effect in effects) > 0
            }
            rows = [
                row
                for row in slices
                if row["population_role"] == role
                and row["family"] == "new_action"
                and row["dimensions"]["partition"]["value"] == partition
            ]
            actual = {
                _canonical_json(
                    row["dimensions"]["new_action_bucket"]["value"]
                ): row
                for row in rows
            }
            if set(actual) != set(expected):
                raise StrategyError(
                    "ImpactCube new_action buckets changed from waterfall"
                )
            for action_token, effects in expected.items():
                row = actual[action_token]
                population = row["population"]["value"]
                if _population_counts(population) != {
                    field: sum(
                        _population_counts(effect)[field]
                        for effect in effects
                    )
                    for field in ("count", "labeled_count", "bad_count")
                }:
                    raise StrategyError(
                        "ImpactCube new_action population changed from "
                        "waterfall"
                    )
                _require_amount_rollup(
                    [effect["amounts"] for effect in effects],
                    population["amounts"],
                    name="ImpactCube new_action amounts",
                )
                _require_projection_matches_action(
                    row["new"]["value"],
                    action=actions[action_token],
                    name="ImpactCube new_action projection",
                )


def _require_projection_matches_action(
    projection: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    name: str,
) -> None:
    strategy_type = projection["strategy_type"]
    population = projection["population_count"]
    breakdown = projection["breakdown"]
    if strategy_type in {"approval", "reject"}:
        expected = action["value"]
        if expected == "approval":
            expected = "approve"
        by_action = {row["action"]: row["count"] for row in breakdown}
        if by_action.get(expected) != population or any(
            count != 0
            for key, count in by_action.items()
            if key != expected
        ):
            raise StrategyError(f"{name} action does not match bucket")
        return
    value = action.get("output_value", action["value"])
    field = {
        "limit": "assigned_limit",
        "pricing": "assigned_rate",
        "segmentation": "segment",
    }[strategy_type]
    if len(breakdown) != 1 or breakdown[0]["count"] != population:
        raise StrategyError(f"{name} breakdown is not homogeneous")
    actual = breakdown[0][field]
    if strategy_type in {"limit", "pricing"}:
        if not math.isclose(
            float(actual),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise StrategyError(f"{name} value does not match bucket")
    elif actual != value:
        raise StrategyError(f"{name} value does not match bucket")


def _slice_sort_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    partition = value["dimensions"]["partition"]["value"]
    return (
        _POPULATION_ROLE_ORDER.index(value["population_role"]),
        _PARTITION_ORDER.index(partition),
        _FAMILY_ORDER.index(value["family"]),
        _canonical_json(value["dimensions"]),
    )


def _json_scalar(value: object, name: str) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise StrategyError(f"{name} must be finite")
        return number
    if isinstance(value, str):
        if "\x00" in value:
            raise StrategyError(f"{name} contains a null byte")
        return value
    if isinstance(value, datetime | date | pd.Timestamp):
        return pd.Timestamp(value).isoformat()
    raise StrategyError(f"{name} must be a JSON scalar")


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _json_value(value: object, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError(f"{name} must be canonical JSON") from exc


def _json_object(value: object, name: str) -> dict[str, Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    if normalized != value:
        raise StrategyError(f"{name} contains non-canonical JSON values")
    return normalized


def _list(value: object, name: str) -> list[Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, list):
        raise StrategyError(f"{name} must be a list")
    if normalized != value:
        raise StrategyError(f"{name} contains non-canonical JSON values")
    return normalized


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        details: list[str] = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        raise StrategyError(f"{name} has invalid fields ({'; '.join(details)})")


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
        raise StrategyError("ImpactCube value must be canonical JSON") from exc


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise StrategyError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_number(value: object, name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategyError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise StrategyError(f"{name} must be a finite number")
    if isinstance(value, Integral):
        return int(value)
    return number


def _ratio_number(value: object, name: str) -> float:
    number = float(_finite_number(value, name))
    if not 0 <= number <= 1:
        raise StrategyError(f"{name} must be between 0 and 1")
    return number


def _target_bad_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in {
        0,
        1,
    }:
        raise StrategyError("ImpactCube target_bad_value must be 0 or 1")
    return int(value)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


__all__ = [
    "MAX_IMPACT_CUBE_DIMENSION_VALUES",
    "MAX_IMPACT_CUBE_JSON_BYTES",
    "MAX_IMPACT_CUBE_ROWS",
    "MAX_IMPACT_CUBE_RULES",
    "MAX_IMPACT_CUBE_SLICES",
    "MAX_IMPACT_CUBE_WORK",
    "MIN_IMPACT_CUBE_GROUP_SIZE",
    "STRATEGY_IMPACT_CUBE_PRODUCER_VERSION",
    "STRATEGY_IMPACT_CUBE_SCHEMA_VERSION",
    "STRATEGY_IMPACT_SLICE_SCHEMA_VERSION",
    "build_strategy_impact_cube",
    "canonical_strategy_impact_cube_json",
    "validate_strategy_impact_cube",
]
