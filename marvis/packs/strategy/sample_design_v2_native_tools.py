"""Native active-dataset execution boundary for StrategySampleDesign V2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import DatasetContentDriftError
from marvis.data.workspace import (
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.files import sha256_file
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_v2 import (
    MAX_SAMPLE_DESIGN_V2_JSON_BYTES,
    STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
    build_historical_score_v2,
    build_sample_design_policy_v2,
    build_sample_population_v2,
    build_strategy_sample_design_v2,
    build_strategy_sample_design_v2_bundle,
    build_target_selector_v2,
    canonical_strategy_sample_design_v2_bundle_json,
    strategy_sample_design_v2_bundle_from_json,
    validate_strategy_sample_design_v2_bundle,
)
from marvis.packs.strategy.sample_membership import (
    decode_sample_membership,
    encode_sample_membership,
    validate_sample_membership_header,
)
from marvis.packs.strategy import sample_design_v2_tools as common
from marvis.repositories.data_workspace import DataWorkspaceRepository


SAMPLE_DESIGN_V2_NATIVE_TOOL_SCHEMA_VERSION = (
    "strategy.materialize-sample-design-v2-native-tool.v1"
)
SAMPLE_DESIGN_V2_NATIVE_ARTIFACT_SCHEMA_VERSION = (
    "strategy.sample-design-v2-native-artifact.v1"
)
SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND = (
    "strategy_sample_membership_v2_native_binary"
)
SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL = (
    "strategy.materialize_sample_design_v2_native"
)
MAX_NATIVE_SAMPLE_DESIGN_V2_SOURCE_ROWS = 1_000_000
MAX_NATIVE_SAMPLE_DESIGN_V2_REQUIRED_COLUMNS = 500
MAX_NATIVE_SAMPLE_DESIGN_V2_REQUIRED_CELLS = 50_000_000
MAX_NATIVE_SAMPLE_DESIGN_V2_SOURCE_BYTES = 20 * 1024 * 1024 * 1024

_INPUT_FIELDS = frozenset(
    {
        "source_mode",
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "target_bad_value",
        "drop_nan_labels",
        "relationship",
        "scope",
        "approval_population",
        "risk_population",
        "partitioning",
        "maturity",
        "performance_window",
        "observation_window",
        "field_bindings",
        "historical_score",
        "policy",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "content_hash",
        "bundle_id",
        "sample_design_id",
        "sample_design_content_hash",
        "membership_id",
        "membership_content_hash",
        "bundle",
        "membership",
        "artifacts",
        "source_binding",
        "warnings",
        "not_created_strategy",
        "not_adopted",
        "not_deployed",
    }
)
_SOURCE_BINDING_FIELDS = frozenset(
    {
        "source_mode",
        "dataset_ref",
        "dataset_registry_ref",
        "workspace_ref",
        "target_selector",
        "membership_registry_identity_hash",
        "development_partition",
    }
)
_DATASET_REF_FIELDS = frozenset({"dataset_id", "content_hash"})
_DATASET_REGISTRY_REF_FIELDS = frozenset(
    {"source_path_hash", "metadata_hash"}
)
_WORKSPACE_REF_FIELDS = frozenset(
    {"revision", "generation", "semantic_mapping_hash"}
)
_TARGET_SELECTOR_FIELDS = frozenset(
    {"column", "bad_value", "drop_missing"}
)
_SOURCE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "source_mode",
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "dataset_source_path",
        "dataset_registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "target_bad_value",
        "drop_nan_labels",
    }
)
_MEMBERSHIP_REGISTRY_IDENTITY_FIELDS = (
    _SOURCE_PROVENANCE_FIELDS - {"dataset_source_path"}
) | {"dataset_source_path_hash"}
_MEMBERSHIP_PROVENANCE_FIELDS = _SOURCE_PROVENANCE_FIELDS | frozenset(
    {
        "format",
        "artifact_role",
        "membership_id",
        "membership_content_hash",
        "membership_artifact_content_hash",
    }
)
_BUNDLE_PROVENANCE_FIELDS = _SOURCE_PROVENANCE_FIELDS | frozenset(
    {
        "format",
        "artifact_role",
        "membership_id",
        "membership_content_hash",
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "bundle_id",
        "bundle_content_hash",
        "bundle_artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "request",
        "request_hash",
    }
)


@dataclass(frozen=True)
class NativeSampleDesignV2LiveBinding:
    task_id: str
    dataset_id: str
    dataset_content_hash: str
    dataset_source_path: str
    dataset_path: Path
    dataset_registry_metadata_hash: str
    row_count: int
    columns: tuple[str, ...]
    workspace_revision: int
    workspace_generation: int
    semantic_mapping_hash: str
    semantic_field_roles: tuple[tuple[str, str], ...]
    target_col: str
    target_bad_value: int
    drop_nan_labels: bool


@dataclass(frozen=True)
class StrategySampleDesignV2NativeArtifactBinding:
    """Authenticated native membership and V2 bundle artifact pair."""

    task_id: str
    membership_artifact_id: str
    membership_path: Path
    membership_artifact_content_hash: str
    bundle_artifact_id: str
    bundle_path: Path
    bundle_artifact_content_hash: str
    provenance: dict[str, Any]
    membership_provenance: dict[str, Any]
    membership: dict[str, Any]
    bundle: dict[str, Any]
    source_binding: NativeSampleDesignV2LiveBinding


@dataclass(frozen=True)
class AuthenticatedNativeSampleDesignV2BundleRecord:
    """One native bundle row authenticated without consulting workspace head."""

    task_id: str
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    bundle: dict[str, Any]
    provenance: dict[str, Any]
    source_provenance: dict[str, Any]


def run_materialize_sample_design_v2_native(
    inputs,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Materialize a governed V2 sample directly from the active dataset."""

    try:
        request = _validate_native_inputs(inputs)
        task_id = common._text(ctx.task_id, "task_id")
        binding = _load_native_live_binding(
            runtime,
            task_id=task_id,
            request=request,
        )
        normalized = common._normalize_request_against_columns(
            request,
            binding,
        )
        required_columns = _required_native_columns(normalized)
        _require_native_resource_preflight(
            binding,
            required_columns=required_columns,
        )
        frame = runtime.backend.read_frame(
            binding.dataset_path,
            columns=list(required_columns),
        )
        _require_frame(binding, frame, phase="before computation")
        masks, predicate_refs, partition_ref = common._resolve_masks(
            frame,
            request=normalized,
            binding=binding,
        )
        common._require_observation_window(
            frame,
            request=normalized,
            masks=masks,
        )
        membership_raw = encode_sample_membership(
            task_id=task_id,
            dataset_id=binding.dataset_id,
            dataset_content_hash=binding.dataset_content_hash,
            masks=masks,
        )
        membership = decode_sample_membership(membership_raw)
        components = _build_native_components(
            frame,
            request=normalized,
            binding=binding,
            membership=membership,
            masks=masks,
            predicate_refs=predicate_refs,
            partition_ref=partition_ref,
        )
        bundle = _build_native_bundle(
            frame=frame,
            request=normalized,
            binding=binding,
            membership=membership,
            masks=masks,
            components=components,
            predicate_refs=predicate_refs,
            partition_ref=partition_ref,
        )
        _require_native_live_binding(
            runtime,
            binding=binding,
            request=normalized,
        )
        return _persist_native_pair(
            runtime,
            task_id=task_id,
            request=normalized,
            binding=binding,
            membership_raw=membership_raw,
            membership=membership,
            bundle=bundle,
        )
    except StrategyError:
        raise
    except common._BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def _validate_native_inputs(value: object) -> dict[str, Any]:
    obj = common._json_object(
        value,
        "materialize_sample_design_v2_native inputs",
    )
    common._exact_fields(
        obj,
        _INPUT_FIELDS,
        "materialize_sample_design_v2_native inputs",
    )
    if obj["source_mode"] != "native_active_dataset":
        raise StrategyError(
            "native sample-design V2 source_mode must be native_active_dataset"
        )
    bad_value = obj["target_bad_value"]
    if isinstance(bad_value, bool) or bad_value not in {0, 1}:
        raise StrategyError("target_bad_value must be integer 0 or 1")
    if not isinstance(obj["drop_nan_labels"], bool):
        raise StrategyError("drop_nan_labels must be boolean")
    result: dict[str, Any] = {
        "source_mode": "native_active_dataset",
        "dataset_id": common._text(obj["dataset_id"], "dataset_id"),
        "expected_dataset_content_hash": common._hash(
            obj["expected_dataset_content_hash"],
            "expected_dataset_content_hash",
        ),
        "workspace_revision": common._non_negative_int(
            obj["workspace_revision"],
            "workspace_revision",
        ),
        "workspace_generation": common._non_negative_int(
            obj["workspace_generation"],
            "workspace_generation",
        ),
        "semantic_mapping_hash": common._hash(
            obj["semantic_mapping_hash"],
            "semantic_mapping_hash",
        ),
        "target_col": common._text(obj["target_col"], "target_col"),
        "target_bad_value": int(bad_value),
        "drop_nan_labels": obj["drop_nan_labels"],
        "relationship": common._enum(
            obj["relationship"],
            {"nested_same_cohort", "parallel_time_cohorts"},
            "relationship",
        ),
        "scope": common._enum(
            obj["scope"],
            {"strategy_development", "exploration_only"},
            "scope",
        ),
        "approval_population": common._population_request(
            obj["approval_population"],
            "approval",
        ),
        "risk_population": common._population_request(
            obj["risk_population"],
            "risk",
        ),
        "partitioning": common._partitioning_request(obj["partitioning"]),
        "maturity": common._maturity_request(obj["maturity"]),
        "performance_window": common._performance_window(
            obj["performance_window"]
        ),
        "observation_window": common._observation_window(
            obj["observation_window"]
        ),
        "field_bindings": common._field_bindings(obj["field_bindings"]),
        "historical_score": common._historical_score_request(
            obj["historical_score"]
        ),
        "policy": common._policy_request(obj["policy"]),
    }
    maturity = result["maturity"]
    performance = result["performance_window"]
    if maturity["status"] in {"confirmed_matured", "not_matured"}:
        if performance["status"] != "provided":
            raise StrategyError(
                "evaluated maturity requires a provided performance window"
            )
        if maturity["performance_window_days"] != performance["days"]:
            raise StrategyError(
                "maturity and performance window days must match"
            )
    expected_scope = (
        "strategy_development"
        if maturity["status"] == "confirmed_matured"
        and performance["status"] == "provided"
        and result["observation_window"]["status"] == "provided"
        else "exploration_only"
    )
    if result["scope"] != expected_scope:
        raise StrategyError("scope is inconsistent with windows and maturity")
    _require_native_target_separation(result)
    return result


def _load_native_live_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    verify_dataset_bytes: bool = True,
) -> NativeSampleDesignV2LiveBinding:
    workspace = DataWorkspaceRepository(
        runtime.settings.db_path
    ).get_or_default(task_id)
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    if (
        workspace.active_dataset_id != request["dataset_id"]
        or workspace.active_dataset_content_hash
        != request["expected_dataset_content_hash"]
        or workspace.revision != request["workspace_revision"]
        or workspace.analysis_generation != request["workspace_generation"]
        or not hmac.compare_digest(
            semantic_hash,
            request["semantic_mapping_hash"],
        )
        or workspace.semantic_mapping.target_col != request["target_col"]
    ):
        raise StrategyError(
            "native sample-design V2 DataWorkspace binding changed"
        )
    try:
        dataset = runtime.registry.get(request["dataset_id"])
        path = Path(runtime.registry.resolve_path(request["dataset_id"]))
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        DatasetContentDriftError,
    ) as exc:
        raise StrategyError(
            "native sample-design V2 active dataset is unavailable or drifted"
        ) from exc
    if (
        str(dataset.task_id) != task_id
        or str(dataset.id) != request["dataset_id"]
        or not common._matches_hash(
            dataset.content_hash,
            request["expected_dataset_content_hash"],
        )
    ):
        raise StrategyError(
            "native sample-design V2 dataset binding changed"
        )
    _require_native_static_source_preflight(
        row_count=int(dataset.row_count),
        dataset_path=path,
    )
    if verify_dataset_bytes:
        try:
            verified_path = Path(
                runtime.registry.resolve_verified_path(request["dataset_id"])
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            DatasetContentDriftError,
        ) as exc:
            raise StrategyError(
                "native sample-design V2 active dataset is unavailable or "
                "drifted"
            ) from exc
        if verified_path != path:
            raise StrategyError(
                "native sample-design V2 dataset registry path changed"
            )
    columns = tuple(str(column.name) for column in dataset.columns)
    if request["target_col"] not in columns:
        raise StrategyError(
            "native sample-design V2 target column is missing"
        )
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = common._dataset_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=str(dataset.id),
            expected_content_hash=str(dataset.content_hash),
        )
    return NativeSampleDesignV2LiveBinding(
        task_id=task_id,
        dataset_id=str(dataset.id),
        dataset_content_hash=str(dataset.content_hash),
        dataset_source_path=str(dataset.source_path),
        dataset_path=path,
        dataset_registry_metadata_hash=metadata_hash,
        row_count=int(dataset.row_count),
        columns=columns,
        workspace_revision=workspace.revision,
        workspace_generation=workspace.analysis_generation,
        semantic_mapping_hash=semantic_hash,
        semantic_field_roles=tuple(
            sorted(
                (str(column), str(role))
                for column, role in (
                    workspace.semantic_mapping.field_roles.items()
                )
            )
        ),
        target_col=request["target_col"],
        target_bad_value=request["target_bad_value"],
        drop_nan_labels=request["drop_nan_labels"],
    )


def _load_historical_native_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    source: Mapping[str, Any],
) -> NativeSampleDesignV2LiveBinding:
    """Recover immutable native source evidence without workspace head."""

    if source["task_id"] != task_id:
        raise StrategyError(
            "native sample-design V2 historical task binding changed"
        )
    expected_request_source = {
        "dataset_id": request["dataset_id"],
        "dataset_content_hash": request[
            "expected_dataset_content_hash"
        ],
        "workspace_revision": request["workspace_revision"],
        "workspace_generation": request["workspace_generation"],
        "semantic_mapping_hash": request["semantic_mapping_hash"],
        "target_col": request["target_col"],
        "target_bad_value": request["target_bad_value"],
        "drop_nan_labels": request["drop_nan_labels"],
    }
    for field, expected in expected_request_source.items():
        if source[field] != expected:
            raise StrategyError(
                "native sample-design V2 historical request "
                f"{field} changed"
            )
    try:
        dataset = runtime.registry.get(source["dataset_id"])
        path = Path(
            runtime.registry.resolve_verified_path(source["dataset_id"])
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        DatasetContentDriftError,
    ) as exc:
        raise StrategyError(
            "native sample-design V2 historical dataset is unavailable "
            "or drifted"
        ) from exc
    _require_native_static_source_preflight(
        row_count=int(dataset.row_count),
        dataset_path=path,
    )
    if (
        str(dataset.task_id) != task_id
        or str(dataset.id) != source["dataset_id"]
        or str(dataset.source_path) != source["dataset_source_path"]
        or not common._matches_hash(
            dataset.content_hash,
            source["dataset_content_hash"],
        )
        or not hmac.compare_digest(
            sha256_file(path),
            source["dataset_content_hash"],
        )
    ):
        raise StrategyError(
            "native sample-design V2 historical dataset binding changed"
        )
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = common._dataset_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=str(dataset.id),
            expected_content_hash=str(dataset.content_hash),
        )
    if not hmac.compare_digest(
        metadata_hash,
        source["dataset_registry_metadata_hash"],
    ):
        raise StrategyError(
            "native sample-design V2 historical dataset metadata changed"
        )
    columns = tuple(str(column.name) for column in dataset.columns)
    if source["target_col"] not in columns:
        raise StrategyError(
            "native sample-design V2 historical target column is missing"
        )
    time_field = request["field_bindings"]["time_field"]
    return NativeSampleDesignV2LiveBinding(
        task_id=task_id,
        dataset_id=str(dataset.id),
        dataset_content_hash=str(dataset.content_hash),
        dataset_source_path=str(dataset.source_path),
        dataset_path=path,
        dataset_registry_metadata_hash=metadata_hash,
        row_count=int(dataset.row_count),
        columns=columns,
        workspace_revision=source["workspace_revision"],
        workspace_generation=source["workspace_generation"],
        semantic_mapping_hash=source["semantic_mapping_hash"],
        semantic_field_roles=(
            () if time_field is None else ((str(time_field), "date"),)
        ),
        target_col=source["target_col"],
        target_bad_value=source["target_bad_value"],
        drop_nan_labels=source["drop_nan_labels"],
    )


def _predicate_columns(value: object) -> set[str]:
    columns: set[str] = set()
    if isinstance(value, Mapping):
        column = value.get("column")
        if isinstance(column, str):
            columns.add(column)
        for child in value.values():
            columns.update(_predicate_columns(child))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for child in value:
            columns.update(_predicate_columns(child))
    return columns


def _require_native_target_separation(
    request: Mapping[str, Any],
) -> None:
    target = str(request["target_col"])
    leaked_at: list[str] = []
    for role in ("approval_population", "risk_population"):
        for control in ("inclusion", "exclusion"):
            if target in _predicate_columns(request[role][control]):
                leaked_at.append(f"{role}.{control}")
    partitioning = request["partitioning"]
    if partitioning["method"] == "predicate_ast":
        for partition, predicate in partitioning["selectors"].items():
            if target in _predicate_columns(predicate):
                leaked_at.append(f"partitioning.selectors.{partition}")
    elif partitioning["column"] == target:
        leaked_at.append("partitioning.column")
    for field, column in request["field_bindings"].items():
        if column == target:
            leaked_at.append(f"field_bindings.{field}")
    historical = request["historical_score"]
    if historical["status"] == "available" and historical["column"] == target:
        leaked_at.append("historical_score.column")
    if leaked_at:
        raise StrategyError(
            "native sample-design V2 target column cannot be used by "
            + ", ".join(sorted(leaked_at))
        )
    if partitioning["method"] == "time_ranges":
        time_field = request["field_bindings"]["time_field"]
        if time_field is not None and partitioning["column"] != time_field:
            raise StrategyError(
                "native sample-design V2 time_ranges column must equal "
                "field_bindings.time_field"
            )


def _required_native_columns(
    request: Mapping[str, Any],
) -> tuple[str, ...]:
    required = {str(request["target_col"])}
    required.update(
        str(value)
        for value in request["field_bindings"].values()
        if value is not None
    )
    historical_column = request["historical_score"]["column"]
    if historical_column is not None:
        required.add(str(historical_column))
    partitioning = request["partitioning"]
    if partitioning["method"] == "time_ranges":
        required.add(str(partitioning["column"]))

    for role in ("approval_population", "risk_population"):
        controls = request[role]
        required.update(_predicate_columns(controls["inclusion"]))
        required.update(_predicate_columns(controls["exclusion"]))
    if partitioning["method"] == "predicate_ast":
        required.update(_predicate_columns(partitioning["selectors"]))
    return tuple(sorted(required))


def _require_native_resource_preflight(
    binding: NativeSampleDesignV2LiveBinding,
    *,
    required_columns: Sequence[str],
) -> None:
    _require_native_static_source_preflight(
        row_count=binding.row_count,
        dataset_path=binding.dataset_path,
    )
    required_count = len(required_columns)
    budgets = (
        (
            "required_columns",
            required_count,
            MAX_NATIVE_SAMPLE_DESIGN_V2_REQUIRED_COLUMNS,
        ),
        (
            "required_cells",
            binding.row_count * required_count,
            MAX_NATIVE_SAMPLE_DESIGN_V2_REQUIRED_CELLS,
        ),
    )
    for dimension, actual, limit in budgets:
        if actual > limit:
            raise StrategyError(
                "native sample-design V2 "
                f"{dimension} budget exceeded: actual={actual}, limit={limit}"
            )


def _require_native_static_source_preflight(
    *,
    row_count: int,
    dataset_path: Path,
) -> None:
    try:
        source_bytes = int(dataset_path.stat().st_size)
    except OSError as exc:
        raise StrategyError(
            "native sample-design V2 source dataset metadata is unavailable"
        ) from exc
    budgets = (
        (
            "source_rows",
            row_count,
            MAX_NATIVE_SAMPLE_DESIGN_V2_SOURCE_ROWS,
        ),
        (
            "source_bytes",
            source_bytes,
            MAX_NATIVE_SAMPLE_DESIGN_V2_SOURCE_BYTES,
        ),
    )
    for dimension, actual, limit in budgets:
        if actual > limit:
            raise StrategyError(
                "native sample-design V2 "
                f"{dimension} budget exceeded: actual={actual}, limit={limit}"
            )


def _target_source_ref(
    binding: NativeSampleDesignV2LiveBinding,
) -> dict[str, str]:
    return common._request_source_ref(
        "dataset_target_binding",
        {
            "dataset_id": binding.dataset_id,
            "dataset_content_hash": binding.dataset_content_hash,
            "workspace_revision": binding.workspace_revision,
            "workspace_generation": binding.workspace_generation,
            "semantic_mapping_hash": binding.semantic_mapping_hash,
            "column": binding.target_col,
            "bad_value": binding.target_bad_value,
            "drop_missing": binding.drop_nan_labels,
        },
    )


def _build_native_components(
    frame: pd.DataFrame,
    *,
    request: Mapping[str, Any],
    binding: NativeSampleDesignV2LiveBinding,
    membership: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    predicate_refs: Sequence[Mapping[str, str]],
    partition_ref: Mapping[str, str],
) -> dict[str, Any]:
    header = membership["header"]
    dataset_ref = common._dataset_source_ref(binding)
    target_ref = _target_source_ref(binding)
    population_refs = common._population_predicate_refs(request)
    target = build_target_selector_v2(
        status="resolved",
        column=binding.target_col,
        good_value=1 - binding.target_bad_value,
        bad_value=binding.target_bad_value,
        drop_missing=binding.drop_nan_labels,
        source_refs=[target_ref],
    )
    maturity, eligible_mask = common._maturity_evidence(
        frame,
        request=request,
        binding=binding,
        risk_mask=common._population_union(masks, "risk"),
        source_ref=dataset_ref,
    )
    common_sources = common._unique_refs(
        [dataset_ref, partition_ref, *predicate_refs]
    )
    approval = build_sample_population_v2(
        role="approval",
        membership_header=header,
        inclusion_predicate_ref=population_refs["approval"][0],
        exclusion_predicate_ref=population_refs["approval"][1],
        source_refs=common_sources,
    )
    risk = build_sample_population_v2(
        role="risk",
        membership_header=header,
        inclusion_predicate_ref=population_refs["risk"][0],
        exclusion_predicate_ref=population_refs["risk"][1],
        maturity_evidence=maturity,
        source_refs=common._unique_refs([*common_sources, target_ref]),
    )
    historical_request = request["historical_score"]
    historical = build_historical_score_v2(
        status=historical_request["status"],
        column=historical_request["column"],
        direction=historical_request["direction"],
        source_refs=(
            [
                common._field_source_ref(
                    binding,
                    kind="historical_score_field",
                    field=historical_request["column"],
                )
            ]
            if historical_request["status"] == "available"
            else []
        ),
        reason=historical_request["reason"],
    )
    policy = build_sample_design_policy_v2(**request["policy"])
    return {
        "target": target,
        "approval": approval,
        "risk": risk,
        "historical": historical,
        "policy": policy,
        "eligible_mask": eligible_mask,
    }


def _build_native_bundle(
    *,
    frame: pd.DataFrame,
    request: Mapping[str, Any],
    binding: NativeSampleDesignV2LiveBinding,
    membership: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    components: Mapping[str, Any],
    predicate_refs: Sequence[Mapping[str, str]],
    partition_ref: Mapping[str, str],
) -> dict[str, Any]:
    header = membership["header"]
    sources = common._unique_refs(
        [
            common._dataset_source_ref(binding),
            _target_source_ref(binding),
            partition_ref,
            *predicate_refs,
        ]
    )
    split = common._split_definition(
        request["partitioning"],
        partition_ref,
    )
    design = build_strategy_sample_design_v2(
        task_id=binding.task_id,
        membership_header=header,
        workspace_revision=binding.workspace_revision,
        workspace_generation=binding.workspace_generation,
        semantic_mapping_hash=binding.semantic_mapping_hash,
        relationship=request["relationship"],
        field_bindings=request["field_bindings"],
        scope=request["scope"],
        performance_window=request["performance_window"],
        observation_window=request["observation_window"],
        split_definition=split,
        target_selector=components["target"],
        approval_population=components["approval"],
        risk_population=components["risk"],
        historical_score=components["historical"],
        policy=components["policy"],
        source_mode="native_active_dataset",
        source_refs=sources,
    )
    statistics = common._diagnostic_statistics(
        frame=frame,
        binding=binding,
        request=request,
        masks=masks,
        eligible_mask=components["eligible_mask"],
    )
    observations = common._metric_observations(
        frame,
        binding=binding,
        masks=masks,
        eligible_mask=components["eligible_mask"],
        maturity_status=request["maturity"]["status"],
        membership_header=header,
        sample_design=design,
    )
    return build_strategy_sample_design_v2_bundle(
        task_id=binding.task_id,
        membership_header=header,
        membership_masks=masks,
        workspace_revision=binding.workspace_revision,
        workspace_generation=binding.workspace_generation,
        semantic_mapping_hash=binding.semantic_mapping_hash,
        relationship=request["relationship"],
        field_bindings=request["field_bindings"],
        scope=request["scope"],
        performance_window=request["performance_window"],
        observation_window=request["observation_window"],
        split_definition=split,
        target_selector=components["target"],
        approval_population=components["approval"],
        risk_population=components["risk"],
        historical_score=components["historical"],
        policy=components["policy"],
        source_mode="native_active_dataset",
        diagnostic_statistics=statistics,
        metric_observations=observations,
        source_refs=sources,
    )


def _require_frame(
    binding: NativeSampleDesignV2LiveBinding,
    frame: object,
    *,
    phase: str,
) -> None:
    if not isinstance(frame, pd.DataFrame) or len(frame) != binding.row_count:
        raise StrategyError(
            f"native sample-design V2 analysis universe row count changed {phase}"
        )
    if not hmac.compare_digest(
        sha256_file(binding.dataset_path),
        binding.dataset_content_hash,
    ):
        raise StrategyError(
            f"native sample-design V2 dataset bytes changed {phase}"
        )


def _require_native_live_binding(
    runtime,
    *,
    binding: NativeSampleDesignV2LiveBinding,
    request: Mapping[str, Any],
) -> None:
    current = _load_native_live_binding(
        runtime,
        task_id=binding.task_id,
        request=request,
        verify_dataset_bytes=False,
    )
    if current != binding:
        raise StrategyError(
            "native sample-design V2 live binding changed during computation"
        )


def _require_native_live_binding_on_connection(
    conn,
    binding: NativeSampleDesignV2LiveBinding,
) -> None:
    if not isinstance(binding, NativeSampleDesignV2LiveBinding):
        raise StrategyError(
            "native sample-design V2 live binding is invalid"
        )
    workspace = conn.execute(
        """
        SELECT revision, active_dataset_id, active_dataset_content_hash,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces
         WHERE task_id = ?
        """,
        (binding.task_id,),
    ).fetchone()
    if workspace is None:
        raise StrategyError(
            "native sample-design V2 DataWorkspace disappeared"
        )
    try:
        mapping = data_semantic_mapping_from_dict(
            json.loads(str(workspace["semantic_mapping_json"]))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(
            "native sample-design V2 semantic mapping is invalid"
        ) from exc
    if (
        int(workspace["revision"]) != binding.workspace_revision
        or int(workspace["analysis_generation"])
        != binding.workspace_generation
        or str(workspace["active_dataset_id"]) != binding.dataset_id
        or str(workspace["active_dataset_content_hash"])
        != binding.dataset_content_hash
        or not hmac.compare_digest(
            data_semantic_mapping_hash(mapping),
            binding.semantic_mapping_hash,
        )
        or mapping.target_col != binding.target_col
    ):
        raise StrategyError(
            "native sample-design V2 DataWorkspace binding changed"
        )
    metadata_hash = common._dataset_metadata_hash_on_connection(
        conn,
        task_id=binding.task_id,
        dataset_id=binding.dataset_id,
        expected_content_hash=binding.dataset_content_hash,
    )
    if not hmac.compare_digest(
        metadata_hash,
        binding.dataset_registry_metadata_hash,
    ):
        raise StrategyError(
            "native sample-design V2 dataset registry metadata changed"
        )
    dataset = conn.execute(
        "SELECT source_path FROM datasets WHERE task_id = ? AND id = ?",
        (binding.task_id, binding.dataset_id),
    ).fetchone()
    if (
        dataset is None
        or str(dataset["source_path"]) != binding.dataset_source_path
    ):
        raise StrategyError(
            "native sample-design V2 dataset registry path changed"
        )
    if not hmac.compare_digest(
        sha256_file(binding.dataset_path),
        binding.dataset_content_hash,
    ):
        raise StrategyError(
            "native sample-design V2 dataset bytes changed before registration"
        )


def _require_historical_native_binding_on_connection(
    conn,
    binding: NativeSampleDesignV2LiveBinding,
) -> None:
    """Recheck immutable dataset registry state without workspace head."""

    if not isinstance(binding, NativeSampleDesignV2LiveBinding):
        raise StrategyError(
            "native sample-design V2 historical binding is invalid"
        )
    metadata_hash = common._dataset_metadata_hash_on_connection(
        conn,
        task_id=binding.task_id,
        dataset_id=binding.dataset_id,
        expected_content_hash=binding.dataset_content_hash,
    )
    if not hmac.compare_digest(
        metadata_hash,
        binding.dataset_registry_metadata_hash,
    ):
        raise StrategyError(
            "native sample-design V2 historical dataset metadata changed"
        )
    dataset = conn.execute(
        """
        SELECT task_id, source_path, content_hash
          FROM datasets
         WHERE task_id = ? AND id = ?
        """,
        (binding.task_id, binding.dataset_id),
    ).fetchone()
    if (
        dataset is None
        or str(dataset["task_id"]) != binding.task_id
        or str(dataset["source_path"]) != binding.dataset_source_path
        or not hmac.compare_digest(
            str(dataset["content_hash"]),
            binding.dataset_content_hash,
        )
    ):
        raise StrategyError(
            "native sample-design V2 historical dataset binding changed"
        )
    try:
        dataset_hash = sha256_file(binding.dataset_path)
    except OSError as exc:
        raise StrategyError(
            "native sample-design V2 historical dataset bytes "
            "are unavailable"
        ) from exc
    if not hmac.compare_digest(
        dataset_hash,
        binding.dataset_content_hash,
    ):
        raise StrategyError(
            "native sample-design V2 historical dataset bytes changed"
        )


def _source_provenance(
    binding: NativeSampleDesignV2LiveBinding,
) -> dict[str, Any]:
    return {
        "schema_version": SAMPLE_DESIGN_V2_NATIVE_ARTIFACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
        "source_mode": "native_active_dataset",
        "task_id": binding.task_id,
        "dataset_id": binding.dataset_id,
        "dataset_content_hash": binding.dataset_content_hash,
        "dataset_source_path": binding.dataset_source_path,
        "dataset_registry_metadata_hash": (
            binding.dataset_registry_metadata_hash
        ),
        "workspace_revision": binding.workspace_revision,
        "workspace_generation": binding.workspace_generation,
        "semantic_mapping_hash": binding.semantic_mapping_hash,
        "target_col": binding.target_col,
        "target_bad_value": binding.target_bad_value,
        "drop_nan_labels": binding.drop_nan_labels,
    }


def native_sample_design_v2_membership_registry_identity_hash(
    value: Mapping[str, Any],
) -> str:
    """Hash the complete source identity used by the membership registry key."""

    obj = common._json_object(
        value,
        "native sample-design V2 source provenance",
    )
    fields = set(obj)
    if _SOURCE_PROVENANCE_FIELDS <= fields:
        source = {
            field: obj[field]
            for field in sorted(
                _SOURCE_PROVENANCE_FIELDS - {"dataset_source_path"}
            )
        }
        source["dataset_source_path_hash"] = hashlib.sha256(
            common._text(
                obj["dataset_source_path"],
                "native source dataset_source_path",
            ).encode("utf-8")
        ).hexdigest()
    elif _MEMBERSHIP_REGISTRY_IDENTITY_FIELDS <= fields:
        source = {
            field: obj[field]
            for field in sorted(_MEMBERSHIP_REGISTRY_IDENTITY_FIELDS)
        }
    else:
        missing = sorted(
            _MEMBERSHIP_REGISTRY_IDENTITY_FIELDS - fields
        )
        raise StrategyError(
            "native sample-design V2 source provenance is missing fields: "
            + ", ".join(missing)
        )
    common._hash(
        source["dataset_source_path_hash"],
        "native source dataset_source_path_hash",
    )
    try:
        canonical = common._canonical_json(source)
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "native sample-design V2 source provenance is not canonical JSON"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _native_membership_filename(
    membership_id: str,
    registry_identity_hash: str,
) -> str:
    normalized_id = common._text(membership_id, "membership_id")
    normalized_hash = common._hash(
        registry_identity_hash,
        "membership_registry_identity_hash",
    )
    return f"{normalized_id}-{normalized_hash[:24]}.bin"


def _persist_native_pair(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    binding: NativeSampleDesignV2LiveBinding,
    membership_raw: bytes,
    membership: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_raw = canonical_strategy_sample_design_v2_bundle_json(
        bundle
    ).encode("utf-8")
    membership_file_hash = hashlib.sha256(membership_raw).hexdigest()
    bundle_file_hash = hashlib.sha256(bundle_raw).hexdigest()
    header = membership["header"]
    design = bundle["sample_design"]
    source = _source_provenance(binding)
    registry_identity_hash = (
        native_sample_design_v2_membership_registry_identity_hash(source)
    )
    out_dir = common._prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    membership_path = out_dir / _native_membership_filename(
        header["membership_id"],
        registry_identity_hash,
    )
    bundle_path = out_dir / f"{bundle['bundle_id']}.json"
    request_evidence = common._json_object(
        request,
        "native sample-design V2 request evidence",
    )
    request_hash = hashlib.sha256(
        common._canonical_json(request_evidence).encode("utf-8")
    ).hexdigest()
    membership_provenance = {
        **source,
        "format": "binary",
        "artifact_role": "membership",
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "membership_artifact_content_hash": membership_file_hash,
    }
    uow = ArtifactUnitOfWork()
    db_committed = False
    rollback_attempted_under_lock = False
    staged_membership = uow.stage_file(out_dir, membership_path.name)
    staged_bundle = uow.stage_file(out_dir, bundle_path.name)
    try:
        staged_membership.path.write_bytes(membership_raw)
        staged_bundle.path.write_bytes(bundle_raw)
    except OSError as exc:
        uow.rollback()
        raise StrategyError(
            "native sample-design V2 artifacts could not be staged"
        ) from exc
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_native_live_binding_on_connection(conn, binding)
                membership_row = common._select_artifact_row(
                    conn,
                    task_id=task_id,
                    kind=(
                        SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
                    ),
                    path=membership_path,
                )
                common._prepare_one_artifact_under_lock(
                    row=membership_row,
                    staged=staged_membership,
                    task_id=task_id,
                    kind=(
                        SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
                    ),
                    path=membership_path,
                    canonical=membership_raw,
                    content_hash=membership_file_hash,
                    provenance=membership_provenance,
                    root=Path(runtime.settings.tasks_dir),
                    maximum_bytes=common._MAX_MEMBERSHIP_FILE_BYTES,
                    origin_tool=SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
                )
                membership_record = (
                    runtime.task_artifacts.register_on_connection(
                        conn,
                        task_id=task_id,
                        kind=(
                            SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
                        ),
                        path=str(membership_path),
                        content_hash=membership_file_hash,
                        origin_tool=SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
                        provenance=membership_provenance,
                    )
                )
                bundle_provenance = {
                    **source,
                    "format": "json",
                    "artifact_role": "bundle",
                    "membership_id": header["membership_id"],
                    "membership_content_hash": header["content_hash"],
                    "membership_artifact_id": membership_record["id"],
                    "membership_artifact_content_hash": (
                        membership_file_hash
                    ),
                    "bundle_id": bundle["bundle_id"],
                    "bundle_content_hash": bundle["content_hash"],
                    "bundle_artifact_content_hash": bundle_file_hash,
                    "sample_design_id": design["sample_design_id"],
                    "sample_design_content_hash": design["content_hash"],
                    "request": request_evidence,
                    "request_hash": request_hash,
                }
                bundle_row = common._select_artifact_row(
                    conn,
                    task_id=task_id,
                    kind=common.SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
                    path=bundle_path,
                )
                common._prepare_one_artifact_under_lock(
                    row=bundle_row,
                    staged=staged_bundle,
                    task_id=task_id,
                    kind=common.SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
                    path=bundle_path,
                    canonical=bundle_raw,
                    content_hash=bundle_file_hash,
                    provenance=bundle_provenance,
                    root=Path(runtime.settings.tasks_dir),
                    maximum_bytes=MAX_SAMPLE_DESIGN_V2_JSON_BYTES,
                    origin_tool=SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
                )
                bundle_record = (
                    runtime.task_artifacts.register_on_connection(
                        conn,
                        task_id=task_id,
                        kind=common.SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
                        path=str(bundle_path),
                        content_hash=bundle_file_hash,
                        origin_tool=SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
                        provenance=bundle_provenance,
                    )
                )
                conn.commit()
                db_committed = True
            except Exception:
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
        uow.commit()
    except Exception:
        if not db_committed and not rollback_attempted_under_lock:
            uow.rollback()
        raise
    return validate_materialize_sample_design_v2_native_tool_output(
        _native_tool_output(
            membership=membership,
            bundle=bundle,
            membership_record=membership_record,
            bundle_record=bundle_record,
            binding=binding,
        )
    )


def _native_source_binding_output(
    binding: NativeSampleDesignV2LiveBinding,
) -> dict[str, Any]:
    registry_identity_hash = (
        native_sample_design_v2_membership_registry_identity_hash(
            _source_provenance(binding)
        )
    )
    return {
        "source_mode": "native_active_dataset",
        "dataset_ref": {
            "dataset_id": binding.dataset_id,
            "content_hash": binding.dataset_content_hash,
        },
        "dataset_registry_ref": {
            "source_path_hash": hashlib.sha256(
                binding.dataset_source_path.encode("utf-8")
            ).hexdigest(),
            "metadata_hash": binding.dataset_registry_metadata_hash,
        },
        "workspace_ref": {
            "revision": binding.workspace_revision,
            "generation": binding.workspace_generation,
            "semantic_mapping_hash": binding.semantic_mapping_hash,
        },
        "target_selector": {
            "column": binding.target_col,
            "bad_value": binding.target_bad_value,
            "drop_missing": binding.drop_nan_labels,
        },
        "membership_registry_identity_hash": registry_identity_hash,
        "development_partition": "risk/development",
    }


def _native_source_binding_from_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_mode": provenance["source_mode"],
        "dataset_ref": {
            "dataset_id": provenance["dataset_id"],
            "content_hash": provenance["dataset_content_hash"],
        },
        "dataset_registry_ref": {
            "source_path_hash": hashlib.sha256(
                str(provenance["dataset_source_path"]).encode("utf-8")
            ).hexdigest(),
            "metadata_hash": provenance[
                "dataset_registry_metadata_hash"
            ],
        },
        "workspace_ref": {
            "revision": provenance["workspace_revision"],
            "generation": provenance["workspace_generation"],
            "semantic_mapping_hash": provenance["semantic_mapping_hash"],
        },
        "target_selector": {
            "column": provenance["target_col"],
            "bad_value": provenance["target_bad_value"],
            "drop_missing": provenance["drop_nan_labels"],
        },
        "membership_registry_identity_hash": (
            native_sample_design_v2_membership_registry_identity_hash(
                provenance
            )
        ),
        "development_partition": "risk/development",
    }


def _native_tool_output(
    *,
    membership: Mapping[str, Any],
    bundle: Mapping[str, Any],
    membership_record: Mapping[str, Any],
    bundle_record: Mapping[str, Any],
    binding: NativeSampleDesignV2LiveBinding,
) -> dict[str, Any]:
    header = membership["header"]
    design = bundle["sample_design"]
    body = {
        "schema_version": SAMPLE_DESIGN_V2_NATIVE_TOOL_SCHEMA_VERSION,
        "bundle_id": bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "bundle": dict(bundle),
        "membership": dict(header),
        "artifacts": {
            "membership": common._artifact_output(
                record=membership_record,
                artifact_format="binary",
                include_content_hash=False,
            ),
            "bundle": common._artifact_output(
                record=bundle_record,
                artifact_format="json",
                include_content_hash=True,
            ),
        },
        "source_binding": _native_source_binding_output(binding),
        "warnings": common._warnings(bundle),
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    return {
        **body,
        "content_hash": hashlib.sha256(
            common._canonical_json(body).encode("utf-8")
        ).hexdigest(),
    }


def validate_materialize_sample_design_v2_native_tool_output(
    value: object,
) -> dict[str, Any]:
    obj = common._json_object(
        value,
        "materialize_sample_design_v2_native output",
    )
    common._exact_fields(
        obj,
        _OUTPUT_FIELDS,
        "materialize_sample_design_v2_native output",
    )
    output_hash = common._hash(
        obj["content_hash"],
        "materialize_sample_design_v2_native output.content_hash",
    )
    bundle = validate_strategy_sample_design_v2_bundle(obj["bundle"])
    header = validate_sample_membership_header(obj["membership"])
    design = bundle["sample_design"]
    if (
        common.resolve_strategy_sample_design_v2_source_mode(
            design,
            capability="physical_v2",
        )
        != "native_active_dataset"
    ):
        raise StrategyError(
            "native sample-design V2 output source mode changed"
        )
    if bundle["membership"] != header:
        raise StrategyError(
            "native sample-design V2 output membership drifted from bundle"
        )
    expected_scalars = {
        "schema_version": SAMPLE_DESIGN_V2_NATIVE_TOOL_SCHEMA_VERSION,
        "bundle_id": bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
    }
    for field, expected in expected_scalars.items():
        if obj[field] != expected:
            raise StrategyError(
                f"native sample-design V2 output {field} drifted"
            )
    source = _validate_source_binding_output(obj["source_binding"])
    registry_identity = {
        "schema_version": SAMPLE_DESIGN_V2_NATIVE_ARTIFACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
        "source_mode": "native_active_dataset",
        "task_id": design["identity"]["task_id"],
        "dataset_id": design["identity"]["dataset_ref"]["dataset_id"],
        "dataset_content_hash": design["identity"]["dataset_ref"][
            "content_hash"
        ],
        "dataset_source_path_hash": source["dataset_registry_ref"][
            "source_path_hash"
        ],
        "dataset_registry_metadata_hash": source[
            "dataset_registry_ref"
        ]["metadata_hash"],
        "workspace_revision": design["identity"]["workspace_ref"][
            "revision"
        ],
        "workspace_generation": design["identity"]["workspace_ref"][
            "generation"
        ],
        "semantic_mapping_hash": design["identity"]["workspace_ref"][
            "semantic_mapping_hash"
        ],
        "target_col": design["target_selector"]["column"],
        "target_bad_value": design["target_selector"]["bad_value"],
        "drop_nan_labels": design["target_selector"]["drop_missing"],
    }
    expected_registry_identity_hash = (
        native_sample_design_v2_membership_registry_identity_hash(
            registry_identity
        )
    )
    if not hmac.compare_digest(
        source["membership_registry_identity_hash"],
        expected_registry_identity_hash,
    ):
        raise StrategyError(
            "native sample-design V2 source registry identity changed"
        )
    artifacts = common._json_object(
        obj["artifacts"],
        "native sample-design V2 artifacts",
    )
    common._exact_fields(
        artifacts,
        common._ARTIFACTS_FIELDS,
        "native sample-design V2 artifacts",
    )
    bundle_raw = canonical_strategy_sample_design_v2_bundle_json(
        bundle
    ).encode("utf-8")
    common._validate_output_artifact(
        artifacts["bundle"],
        kind=common.SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        artifact_format="json",
        filename=f"{bundle['bundle_id']}.json",
        expected_content_hash=hashlib.sha256(bundle_raw).hexdigest(),
    )
    common._validate_output_artifact(
        artifacts["membership"],
        kind=SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        artifact_format="binary",
        filename=_native_membership_filename(
            header["membership_id"],
            source["membership_registry_identity_hash"],
        ),
        expected_content_hash=None,
    )
    expected_source = {
        "source_mode": "native_active_dataset",
        "dataset_ref": design["identity"]["dataset_ref"]
        | {},
        "dataset_registry_ref": source["dataset_registry_ref"],
        "workspace_ref": design["identity"]["workspace_ref"],
        "target_selector": {
            "column": design["target_selector"]["column"],
            "bad_value": design["target_selector"]["bad_value"],
            "drop_missing": design["target_selector"]["drop_missing"],
        },
        "membership_registry_identity_hash": expected_registry_identity_hash,
        "development_partition": "risk/development",
    }
    expected_source["dataset_ref"].pop("role", None)
    if source != expected_source:
        raise StrategyError(
            "native sample-design V2 source binding drifted"
        )
    if obj["warnings"] != common._warnings(bundle):
        raise StrategyError("native sample-design V2 warnings drifted")
    for field in (
        "not_created_strategy",
        "not_adopted",
        "not_deployed",
    ):
        if obj[field] is not True:
            raise StrategyError(
                f"native sample-design V2 output {field} must be true"
            )
    obj["bundle"] = bundle
    obj["membership"] = header
    addressed = {
        key: item for key, item in obj.items() if key != "content_hash"
    }
    expected_hash = hashlib.sha256(
        common._canonical_json(addressed).encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(output_hash, expected_hash):
        raise StrategyError(
            "native sample-design V2 output content_hash does not match content"
        )
    return obj


def _validate_source_binding_output(value: object) -> dict[str, Any]:
    obj = common._json_object(
        value,
        "native sample-design V2 source_binding",
    )
    common._exact_fields(
        obj,
        _SOURCE_BINDING_FIELDS,
        "native sample-design V2 source_binding",
    )
    if (
        obj["source_mode"] != "native_active_dataset"
        or obj["development_partition"] != "risk/development"
    ):
        raise StrategyError(
            "native sample-design V2 source_binding mode is invalid"
        )
    dataset = common._json_object(
        obj["dataset_ref"],
        "native sample-design V2 source_binding.dataset_ref",
    )
    common._exact_fields(
        dataset,
        _DATASET_REF_FIELDS,
        "native sample-design V2 source_binding.dataset_ref",
    )
    registry = common._json_object(
        obj["dataset_registry_ref"],
        "native sample-design V2 source_binding.dataset_registry_ref",
    )
    common._exact_fields(
        registry,
        _DATASET_REGISTRY_REF_FIELDS,
        "native sample-design V2 source_binding.dataset_registry_ref",
    )
    workspace = common._json_object(
        obj["workspace_ref"],
        "native sample-design V2 source_binding.workspace_ref",
    )
    common._exact_fields(
        workspace,
        _WORKSPACE_REF_FIELDS,
        "native sample-design V2 source_binding.workspace_ref",
    )
    target = common._json_object(
        obj["target_selector"],
        "native sample-design V2 source_binding.target_selector",
    )
    common._exact_fields(
        target,
        _TARGET_SELECTOR_FIELDS,
        "native sample-design V2 source_binding.target_selector",
    )
    if (
        isinstance(target["bad_value"], bool)
        or target["bad_value"] not in {0, 1}
        or not isinstance(target["drop_missing"], bool)
    ):
        raise StrategyError(
            "native sample-design V2 source target selector is invalid"
        )
    return {
        "source_mode": "native_active_dataset",
        "dataset_ref": {
            "dataset_id": common._text(
                dataset["dataset_id"],
                "source dataset_id",
            ),
            "content_hash": common._hash(
                dataset["content_hash"],
                "source dataset content_hash",
            ),
        },
        "dataset_registry_ref": {
            "source_path_hash": common._hash(
                registry["source_path_hash"],
                "source dataset registry source_path_hash",
            ),
            "metadata_hash": common._hash(
                registry["metadata_hash"],
                "source dataset registry metadata_hash",
            ),
        },
        "workspace_ref": {
            "revision": common._non_negative_int(
                workspace["revision"],
                "source workspace revision",
            ),
            "generation": common._non_negative_int(
                workspace["generation"],
                "source workspace generation",
            ),
            "semantic_mapping_hash": common._hash(
                workspace["semantic_mapping_hash"],
                "source semantic_mapping_hash",
            ),
        },
        "target_selector": {
            "column": common._text(
                target["column"],
                "source target column",
            ),
            "bad_value": int(target["bad_value"]),
            "drop_missing": target["drop_missing"],
        },
        "membership_registry_identity_hash": common._hash(
            obj["membership_registry_identity_hash"],
            "source membership_registry_identity_hash",
        ),
        "development_partition": "risk/development",
    }


def authenticate_native_strategy_sample_design_v2_bundle_record(
    runtime,
    *,
    task_id: str,
    record: Mapping[str, Any],
) -> AuthenticatedNativeSampleDesignV2BundleRecord:
    """Authenticate one historical native bundle without reading workspace head."""

    try:
        normalized_task = common._text(task_id, "task_id")
        supplied = common._json_object(
            record,
            "native sample-design V2 bundle registry row",
        )
        common._exact_fields(
            supplied,
            common._RECORD_FIELDS,
            "native sample-design V2 bundle registry row",
        )
        artifact_id = common._hash(
            supplied["id"],
            "native bundle artifact_id",
        )
        artifact_hash = common._hash(
            supplied["content_hash"],
            "native bundle artifact content_hash",
        )
        registered = common._registered_record(
            runtime,
            task_id=normalized_task,
            artifact_id=artifact_id,
            kind=common.SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
            expected_content_hash=artifact_hash,
            origin_tool=SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        )
        if common._canonical_json(registered) != common._canonical_json(
            supplied
        ):
            raise StrategyError(
                "native sample-design V2 bundle registry row changed"
            )
        bundle_path = common._canonical_path(
            runtime.settings.tasks_dir,
            task_id=normalized_task,
            filename=Path(str(registered["path"])).name,
        )
        if Path(str(registered["path"])) != bundle_path:
            raise StrategyError(
                "native sample-design V2 bundle path is not canonical"
            )
        bundle_raw = common._read_verified(
            bundle_path,
            root=Path(runtime.settings.tasks_dir),
            expected_hash=artifact_hash,
            maximum_bytes=MAX_SAMPLE_DESIGN_V2_JSON_BYTES,
        )
        try:
            bundle = strategy_sample_design_v2_bundle_from_json(bundle_raw)
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise StrategyError(
                "native sample-design V2 bundle artifact is invalid"
            ) from exc
        canonical = canonical_strategy_sample_design_v2_bundle_json(
            bundle
        ).encode("utf-8")
        if canonical != bundle_raw:
            raise StrategyError(
                "native sample-design V2 bundle bytes are not canonical"
            )
        if bundle_path.name != f"{bundle['bundle_id']}.json":
            raise StrategyError(
                "native sample-design V2 bundle filename changed"
            )
        design = bundle["sample_design"]
        if (
            common.resolve_strategy_sample_design_v2_source_mode(
                design,
                capability="physical_v2",
            )
            != "native_active_dataset"
        ):
            raise StrategyError(
                "native sample-design V2 bundle source mode changed"
            )
        provenance = _validate_bundle_provenance(
            registered["provenance"]
        )
        membership = {"header": bundle["membership"]}
        _require_native_bundle_provenance(
            provenance,
            membership=membership,
            bundle=bundle,
            membership_artifact_id=provenance[
                "membership_artifact_id"
            ],
            membership_file_hash=provenance[
                "membership_artifact_content_hash"
            ],
            bundle_file_hash=artifact_hash,
        )
        source_provenance = {
            field: provenance[field]
            for field in sorted(_SOURCE_PROVENANCE_FIELDS)
        }
        return AuthenticatedNativeSampleDesignV2BundleRecord(
            task_id=normalized_task,
            artifact_id=artifact_id,
            artifact_path=bundle_path,
            artifact_content_hash=artifact_hash,
            bundle=bundle,
            provenance=provenance,
            source_provenance=source_provenance,
        )
    except StrategyError:
        raise
    except common._BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def load_native_strategy_sample_design_v2_artifacts(
    runtime,
    *,
    task_id: str,
    membership_artifact_id: str,
    expected_membership_artifact_content_hash: str,
    bundle_artifact_id: str,
    expected_bundle_artifact_content_hash: str,
    expected_bundle_id: str,
    expected_sample_design_id: str,
    expected_sample_design_content_hash: str,
) -> StrategySampleDesignV2NativeArtifactBinding:
    """Load and deterministically replay an active native V2 artifact pair."""

    return _load_native_strategy_sample_design_v2_artifacts(
        runtime,
        task_id=task_id,
        membership_artifact_id=membership_artifact_id,
        expected_membership_artifact_content_hash=(
            expected_membership_artifact_content_hash
        ),
        bundle_artifact_id=bundle_artifact_id,
        expected_bundle_artifact_content_hash=(
            expected_bundle_artifact_content_hash
        ),
        expected_bundle_id=expected_bundle_id,
        expected_sample_design_id=expected_sample_design_id,
        expected_sample_design_content_hash=(
            expected_sample_design_content_hash
        ),
        require_current_workspace=True,
    )


def load_historical_native_strategy_sample_design_v2_artifacts(
    runtime,
    *,
    task_id: str,
    membership_artifact_id: str,
    expected_membership_artifact_content_hash: str,
    bundle_artifact_id: str,
    expected_bundle_artifact_content_hash: str,
    expected_bundle_id: str,
    expected_sample_design_id: str,
    expected_sample_design_content_hash: str,
) -> StrategySampleDesignV2NativeArtifactBinding:
    """Replay immutable native V2 evidence without requiring workspace head."""

    return _load_native_strategy_sample_design_v2_artifacts(
        runtime,
        task_id=task_id,
        membership_artifact_id=membership_artifact_id,
        expected_membership_artifact_content_hash=(
            expected_membership_artifact_content_hash
        ),
        bundle_artifact_id=bundle_artifact_id,
        expected_bundle_artifact_content_hash=(
            expected_bundle_artifact_content_hash
        ),
        expected_bundle_id=expected_bundle_id,
        expected_sample_design_id=expected_sample_design_id,
        expected_sample_design_content_hash=(
            expected_sample_design_content_hash
        ),
        require_current_workspace=False,
    )


def _load_native_strategy_sample_design_v2_artifacts(
    runtime,
    *,
    task_id: str,
    membership_artifact_id: str,
    expected_membership_artifact_content_hash: str,
    bundle_artifact_id: str,
    expected_bundle_artifact_content_hash: str,
    expected_bundle_id: str,
    expected_sample_design_id: str,
    expected_sample_design_content_hash: str,
    require_current_workspace: bool,
) -> StrategySampleDesignV2NativeArtifactBinding:
    try:
        normalized_task = common._text(task_id, "task_id")
        membership_aid = common._hash(
            membership_artifact_id,
            "membership_artifact_id",
        )
        membership_file_hash = common._hash(
            expected_membership_artifact_content_hash,
            "expected_membership_artifact_content_hash",
        )
        bundle_aid = common._hash(
            bundle_artifact_id,
            "bundle_artifact_id",
        )
        bundle_file_hash = common._hash(
            expected_bundle_artifact_content_hash,
            "expected_bundle_artifact_content_hash",
        )
        bundle_id = common._text(
            expected_bundle_id,
            "expected_bundle_id",
        )
        design_id = common._text(
            expected_sample_design_id,
            "expected_sample_design_id",
        )
        design_hash = common._hash(
            expected_sample_design_content_hash,
            "expected_sample_design_content_hash",
        )
        membership_record = common._registered_record(
            runtime,
            task_id=normalized_task,
            artifact_id=membership_aid,
            kind=SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
            expected_content_hash=membership_file_hash,
            origin_tool=SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        )
        bundle_record = common._registered_record(
            runtime,
            task_id=normalized_task,
            artifact_id=bundle_aid,
            kind=common.SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
            expected_content_hash=bundle_file_hash,
            origin_tool=SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        )
        membership_path = common._canonical_path(
            runtime.settings.tasks_dir,
            task_id=normalized_task,
            filename=Path(str(membership_record["path"])).name,
        )
        bundle_path = common._canonical_path(
            runtime.settings.tasks_dir,
            task_id=normalized_task,
            filename=Path(str(bundle_record["path"])).name,
        )
        if (
            Path(str(membership_record["path"])) != membership_path
            or Path(str(bundle_record["path"])) != bundle_path
        ):
            raise StrategyError(
                "native sample-design V2 artifact path is not canonical"
            )
        membership_raw = common._read_verified(
            membership_path,
            root=Path(runtime.settings.tasks_dir),
            expected_hash=membership_file_hash,
            maximum_bytes=common._MAX_MEMBERSHIP_FILE_BYTES,
        )
        bundle_raw = common._read_verified(
            bundle_path,
            root=Path(runtime.settings.tasks_dir),
            expected_hash=bundle_file_hash,
            maximum_bytes=MAX_SAMPLE_DESIGN_V2_JSON_BYTES,
        )
        membership = decode_sample_membership(membership_raw)
        try:
            bundle = strategy_sample_design_v2_bundle_from_json(bundle_raw)
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise StrategyError(
                "native sample-design V2 bundle artifact is invalid"
            ) from exc
        if (
            canonical_strategy_sample_design_v2_bundle_json(
                bundle
            ).encode("utf-8")
            != bundle_raw
        ):
            raise StrategyError(
                "native sample-design V2 bundle bytes are not canonical"
            )
        if bundle["membership"] != membership["header"]:
            raise StrategyError(
                "native sample-design V2 artifact pair membership changed"
            )
        if bundle_path.name != f"{bundle_id}.json":
            raise StrategyError(
                "native sample-design V2 artifact filename changed"
            )
        design = bundle["sample_design"]
        if (
            bundle["bundle_id"] != bundle_id
            or design["sample_design_id"] != design_id
            or not hmac.compare_digest(
                design["content_hash"],
                design_hash,
            )
        ):
            raise StrategyError(
                "native sample-design V2 artifact identity changed"
            )
        if (
            common.resolve_strategy_sample_design_v2_source_mode(
                design,
                capability="physical_v2",
            )
            != "native_active_dataset"
        ):
            raise StrategyError(
                "native sample-design V2 artifact source mode changed"
            )
        membership_provenance = _validate_membership_provenance(
            membership_record["provenance"]
        )
        bundle_provenance = _validate_bundle_provenance(
            bundle_record["provenance"]
        )
        if membership_path.name != _native_membership_filename(
            membership["header"]["membership_id"],
            native_sample_design_v2_membership_registry_identity_hash(
                membership_provenance
            ),
        ):
            raise StrategyError(
                "native sample-design V2 membership filename changed"
            )
        _require_native_membership_provenance(
            membership_provenance,
            membership=membership,
            membership_file_hash=membership_file_hash,
        )
        _require_native_bundle_provenance(
            bundle_provenance,
            membership=membership,
            bundle=bundle,
            membership_artifact_id=membership_aid,
            membership_file_hash=membership_file_hash,
            bundle_file_hash=bundle_file_hash,
        )
        request = _validate_native_inputs(bundle_provenance["request"])
        binding = (
            _load_native_live_binding(
                runtime,
                task_id=normalized_task,
                request=request,
            )
            if require_current_workspace
            else _load_historical_native_binding(
                runtime,
                task_id=normalized_task,
                request=request,
                source=bundle_provenance,
            )
        )
        normalized_request = common._normalize_request_against_columns(
            request,
            binding,
        )
        required_columns = _required_native_columns(normalized_request)
        _require_native_resource_preflight(
            binding,
            required_columns=required_columns,
        )
        if common._canonical_json(
            normalized_request
        ) != common._canonical_json(bundle_provenance["request"]):
            raise StrategyError(
                "native sample-design V2 provenance request is not canonical"
            )
        frame = runtime.backend.read_frame(
            binding.dataset_path,
            columns=list(required_columns),
        )
        _require_frame(binding, frame, phase="during artifact load")
        masks, predicate_refs, partition_ref = common._resolve_masks(
            frame,
            request=normalized_request,
            binding=binding,
        )
        if any(
            not np.array_equal(
                masks[name],
                membership["masks"][name],
            )
            for name in masks
        ):
            raise StrategyError(
                "native sample-design V2 membership no longer matches request"
            )
        common._require_observation_window(
            frame,
            request=normalized_request,
            masks=masks,
        )
        components = _build_native_components(
            frame,
            request=normalized_request,
            binding=binding,
            membership=membership,
            masks=masks,
            predicate_refs=predicate_refs,
            partition_ref=partition_ref,
        )
        rebuilt = _build_native_bundle(
            frame=frame,
            request=normalized_request,
            binding=binding,
            membership=membership,
            masks=masks,
            components=components,
            predicate_refs=predicate_refs,
            partition_ref=partition_ref,
        )
        if rebuilt != bundle:
            raise StrategyError(
                "native sample-design V2 bundle no longer matches evidence"
            )
        _require_native_source_provenance(
            membership_provenance,
            binding=binding,
        )
        _require_native_source_provenance(
            bundle_provenance,
            binding=binding,
        )
        if require_current_workspace:
            _require_native_live_binding(
                runtime,
                binding=binding,
                request=normalized_request,
            )
        else:
            historical = _load_historical_native_binding(
                runtime,
                task_id=normalized_task,
                request=normalized_request,
                source=bundle_provenance,
            )
            if historical != binding:
                raise StrategyError(
                    "native sample-design V2 historical source changed "
                    "during replay"
                )
        if not hmac.compare_digest(
            sha256_file(binding.dataset_path),
            binding.dataset_content_hash,
        ):
            raise StrategyError(
                "native sample-design V2 dataset bytes changed "
                "after artifact replay"
            )
        return StrategySampleDesignV2NativeArtifactBinding(
            task_id=normalized_task,
            membership_artifact_id=membership_aid,
            membership_path=membership_path,
            membership_artifact_content_hash=membership_file_hash,
            bundle_artifact_id=bundle_aid,
            bundle_path=bundle_path,
            bundle_artifact_content_hash=bundle_file_hash,
            provenance=bundle_provenance,
            membership_provenance=membership_provenance,
            membership=membership,
            bundle=bundle,
            source_binding=binding,
        )
    except StrategyError:
        raise
    except common._BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def require_native_strategy_sample_design_v2_artifact_binding_on_connection(
    conn,
    binding: StrategySampleDesignV2NativeArtifactBinding,
) -> None:
    """Re-authenticate native evidence while a downstream writer holds a lock."""

    _require_native_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        binding,
        require_current_workspace=True,
    )


def require_historical_native_strategy_sample_design_v2_artifact_binding_on_connection(
    conn,
    binding: StrategySampleDesignV2NativeArtifactBinding,
) -> None:
    """Re-authenticate immutable native evidence without workspace head."""

    _require_native_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        binding,
        require_current_workspace=False,
    )


def _require_native_strategy_sample_design_v2_artifact_binding_on_connection(
    conn,
    binding: StrategySampleDesignV2NativeArtifactBinding,
    *,
    require_current_workspace: bool,
) -> None:
    if not isinstance(
        binding,
        StrategySampleDesignV2NativeArtifactBinding,
    ):
        raise StrategyError(
            "native sample-design V2 artifact binding is invalid"
        )
    expected_membership_filename = _native_membership_filename(
        binding.membership["header"]["membership_id"],
        native_sample_design_v2_membership_registry_identity_hash(
            _source_provenance(binding.source_binding)
        ),
    )
    if binding.membership_path.name != expected_membership_filename:
        raise StrategyError(
            "native sample-design V2 membership filename changed"
        )
    if require_current_workspace:
        _require_native_live_binding_on_connection(
            conn,
            binding.source_binding,
        )
    else:
        _require_historical_native_binding_on_connection(
            conn,
            binding.source_binding,
        )
    membership_raw = encode_sample_membership(
        task_id=binding.membership["header"]["task_id"],
        dataset_id=(
            binding.membership["header"]["dataset_ref"]["dataset_id"]
        ),
        dataset_content_hash=(
            binding.membership["header"]["dataset_ref"]["content_hash"]
        ),
        masks=binding.membership["masks"],
    )
    bundle_raw = canonical_strategy_sample_design_v2_bundle_json(
        binding.bundle
    ).encode("utf-8")
    checks = (
        (
            binding.membership_artifact_id,
            SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
            binding.membership_path,
            binding.membership_artifact_content_hash,
            binding.membership_provenance,
            membership_raw,
            common._MAX_MEMBERSHIP_FILE_BYTES,
        ),
        (
            binding.bundle_artifact_id,
            common.SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
            binding.bundle_path,
            binding.bundle_artifact_content_hash,
            binding.provenance,
            bundle_raw,
            MAX_SAMPLE_DESIGN_V2_JSON_BYTES,
        ),
    )
    for (
        artifact_id,
        kind,
        path,
        content_hash,
        provenance,
        canonical,
        maximum_bytes,
    ) in checks:
        row = common._select_artifact_row(
            conn,
            task_id=binding.task_id,
            kind=kind,
            path=path,
        )
        if row is None or str(row["id"]) != artifact_id:
            raise StrategyError(
                "native sample-design V2 artifact disappeared before write"
            )
        common._require_existing_row(
            row,
            task_id=binding.task_id,
            kind=kind,
            path=path,
            content_hash=content_hash,
            provenance=provenance,
            origin_tool=SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        )
        common._require_exact_file(
            path,
            root=binding.membership_path.parents[2],
            canonical=canonical,
            content_hash=content_hash,
            maximum_bytes=maximum_bytes,
        )


def _validate_source_provenance(
    value: Mapping[str, Any],
) -> None:
    if (
        value["schema_version"]
        != SAMPLE_DESIGN_V2_NATIVE_ARTIFACT_SCHEMA_VERSION
        or value["producer_version"]
        != STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION
        or value["source_mode"] != "native_active_dataset"
    ):
        raise StrategyError(
            "native sample-design V2 provenance version is invalid"
        )
    for field in (
        "dataset_content_hash",
        "dataset_registry_metadata_hash",
        "semantic_mapping_hash",
    ):
        common._hash(value[field], f"native provenance.{field}")
    for field in (
        "task_id",
        "dataset_id",
        "dataset_source_path",
        "target_col",
    ):
        common._text(value[field], f"native provenance.{field}")
    common._non_negative_int(
        value["workspace_revision"],
        "native provenance.workspace_revision",
    )
    common._non_negative_int(
        value["workspace_generation"],
        "native provenance.workspace_generation",
    )
    bad_value = value["target_bad_value"]
    if (
        isinstance(bad_value, bool)
        or bad_value not in {0, 1}
        or not isinstance(value["drop_nan_labels"], bool)
    ):
        raise StrategyError(
            "native sample-design V2 provenance target binding is invalid"
        )


def _validate_membership_provenance(value: object) -> dict[str, Any]:
    obj = common._json_object(
        value,
        "native sample-design V2 membership provenance",
    )
    common._exact_fields(
        obj,
        _MEMBERSHIP_PROVENANCE_FIELDS,
        "native sample-design V2 membership provenance",
    )
    _validate_source_provenance(obj)
    if obj["format"] != "binary" or obj["artifact_role"] != "membership":
        raise StrategyError(
            "native sample-design V2 membership provenance role is invalid"
        )
    for field in (
        "membership_content_hash",
        "membership_artifact_content_hash",
    ):
        common._hash(obj[field], f"native membership provenance.{field}")
    common._text(
        obj["membership_id"],
        "native membership provenance.membership_id",
    )
    return obj


def _validate_bundle_provenance(value: object) -> dict[str, Any]:
    obj = common._json_object(
        value,
        "native sample-design V2 bundle provenance",
    )
    common._exact_fields(
        obj,
        _BUNDLE_PROVENANCE_FIELDS,
        "native sample-design V2 bundle provenance",
    )
    _validate_source_provenance(obj)
    if obj["format"] != "json" or obj["artifact_role"] != "bundle":
        raise StrategyError(
            "native sample-design V2 bundle provenance role is invalid"
        )
    for field in (
        "membership_content_hash",
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "bundle_content_hash",
        "bundle_artifact_content_hash",
        "sample_design_content_hash",
        "request_hash",
    ):
        common._hash(obj[field], f"native bundle provenance.{field}")
    for field in ("membership_id", "bundle_id", "sample_design_id"):
        common._text(obj[field], f"native bundle provenance.{field}")
    request = _validate_native_inputs(obj["request"])
    expected_hash = hashlib.sha256(
        common._canonical_json(request).encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(obj["request_hash"], expected_hash):
        raise StrategyError(
            "native sample-design V2 provenance request_hash changed"
        )
    obj["request"] = request
    return obj


def _require_native_membership_provenance(
    provenance: Mapping[str, Any],
    *,
    membership: Mapping[str, Any],
    membership_file_hash: str,
) -> None:
    header = membership["header"]
    expected = {
        "artifact_role": "membership",
        "format": "binary",
        "task_id": header["task_id"],
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "membership_artifact_content_hash": membership_file_hash,
        "dataset_id": header["dataset_ref"]["dataset_id"],
        "dataset_content_hash": header["dataset_ref"]["content_hash"],
    }
    for field, expected_value in expected.items():
        if provenance[field] != expected_value:
            raise StrategyError(
                f"native sample-design V2 membership provenance {field} changed"
            )


def _require_native_bundle_provenance(
    provenance: Mapping[str, Any],
    *,
    membership: Mapping[str, Any],
    bundle: Mapping[str, Any],
    membership_artifact_id: str,
    membership_file_hash: str,
    bundle_file_hash: str,
) -> None:
    header = membership["header"]
    design = bundle["sample_design"]
    expected = {
        "artifact_role": "bundle",
        "format": "json",
        "task_id": design["identity"]["task_id"],
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "membership_artifact_id": membership_artifact_id,
        "membership_artifact_content_hash": membership_file_hash,
        "bundle_id": bundle["bundle_id"],
        "bundle_content_hash": bundle["content_hash"],
        "bundle_artifact_content_hash": bundle_file_hash,
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "dataset_id": design["identity"]["dataset_ref"]["dataset_id"],
        "dataset_content_hash": (
            design["identity"]["dataset_ref"]["content_hash"]
        ),
        "workspace_revision": (
            design["identity"]["workspace_ref"]["revision"]
        ),
        "workspace_generation": (
            design["identity"]["workspace_ref"]["generation"]
        ),
        "semantic_mapping_hash": (
            design["identity"]["workspace_ref"][
                "semantic_mapping_hash"
            ]
        ),
        "target_col": design["target_selector"]["column"],
        "target_bad_value": design["target_selector"]["bad_value"],
        "drop_nan_labels": design["target_selector"]["drop_missing"],
        "source_mode": "native_active_dataset",
    }
    for field, expected_value in expected.items():
        if provenance[field] != expected_value:
            raise StrategyError(
                f"native sample-design V2 bundle provenance {field} changed"
            )


def _require_native_source_provenance(
    provenance: Mapping[str, Any],
    *,
    binding: NativeSampleDesignV2LiveBinding,
) -> None:
    for field, expected in _source_provenance(binding).items():
        if provenance[field] != expected:
            raise StrategyError(
                f"native sample-design V2 source provenance {field} changed"
            )


__all__ = [
    "AuthenticatedNativeSampleDesignV2BundleRecord",
    "MAX_NATIVE_SAMPLE_DESIGN_V2_REQUIRED_CELLS",
    "MAX_NATIVE_SAMPLE_DESIGN_V2_REQUIRED_COLUMNS",
    "MAX_NATIVE_SAMPLE_DESIGN_V2_SOURCE_BYTES",
    "MAX_NATIVE_SAMPLE_DESIGN_V2_SOURCE_ROWS",
    "NativeSampleDesignV2LiveBinding",
    "SAMPLE_DESIGN_V2_NATIVE_ARTIFACT_SCHEMA_VERSION",
    "SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND",
    "SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL",
    "SAMPLE_DESIGN_V2_NATIVE_TOOL_SCHEMA_VERSION",
    "StrategySampleDesignV2NativeArtifactBinding",
    "authenticate_native_strategy_sample_design_v2_bundle_record",
    "load_historical_native_strategy_sample_design_v2_artifacts",
    "load_native_strategy_sample_design_v2_artifacts",
    "native_sample_design_v2_membership_registry_identity_hash",
    "require_historical_native_strategy_sample_design_v2_artifact_binding_on_connection",
    "require_native_strategy_sample_design_v2_artifact_binding_on_connection",
    "run_materialize_sample_design_v2_native",
    "validate_materialize_sample_design_v2_native_tool_output",
]
