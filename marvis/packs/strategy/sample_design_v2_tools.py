"""Governed execution boundary for StrategySampleDesign V2.

The V2 Tool freezes two populations and six partitions over the exact active
dataset row order.  Platform identity is never accepted from the request: it
is derived from the live DataWorkspace, dataset registry, and an authenticated
legacy ``StrategySampleDesignRef`` whose development membership must equal the
new ``risk/development`` mask row for row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import hmac
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.data.errors import DatasetContentDriftError
from marvis.data.predicate_ast import (
    PredicateAstError,
    canonicalize_predicate,
    evaluate_predicate,
)
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.files import sha256_file
from marvis.packs.strategy.errors import (
    StrategyError,
    StrategySampleDesignV2NativeSourceUnsupportedError,
)
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignExecutionBinding,
    StrategySampleDesignRef,
    bind_strategy_development_frame,
    load_historical_strategy_sample_design_execution_binding,
    load_strategy_sample_design_execution_binding,
    require_historical_strategy_sample_design_execution_binding_on_connection,
    require_strategy_sample_design_execution_binding_on_connection,
)
from marvis.packs.strategy.sample_design_tools import (
    load_historical_strategy_sample_design_artifact,
    load_strategy_sample_design_artifact,
)
from marvis.packs.strategy.sample_design_v2 import (
    MAX_SAMPLE_DESIGN_V2_JSON_BYTES,
    PARTITION_NAMES,
    STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
    build_historical_score_v2,
    build_metric_definitions_v2,
    build_metric_observation_v2,
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
    MAX_MEMBERSHIP_HEADER_BYTES,
    MAX_MEMBERSHIP_PAYLOAD_BYTES,
    decode_sample_membership,
    encode_sample_membership,
    validate_sample_membership_header,
)
from marvis.repositories.data_workspace import (
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DataWorkspaceRepository,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


SAMPLE_DESIGN_V2_TOOL_SCHEMA_VERSION = "strategy.materialize-sample-design-v2-tool.v2"
SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION = "strategy.sample-design-v2-artifact.v1"
SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND = "strategy_sample_membership_v2_binary"
SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND = "strategy_sample_design_v2_json"
SAMPLE_DESIGN_V2_ORIGIN_TOOL = "strategy.materialize_sample_design_v2"

_MAX_PREDICATE_NODES = 256
_MAX_PREDICATE_DEPTH = 12
_MAX_MEMBERSHIP_FILE_BYTES = (
    MAX_MEMBERSHIP_HEADER_BYTES + MAX_MEMBERSHIP_PAYLOAD_BYTES + 64
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_INPUT_FIELDS = frozenset(
    {
        "legacy_sample_design_ref",
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
_POPULATION_REQUEST_FIELDS = frozenset({"inclusion", "exclusion"})
_FIELD_BINDING_FIELDS = frozenset(
    {
        "entity_field",
        "time_field",
        "group_field",
        "month_field",
        "weight_field",
        "loan_amount_field",
        "overdue_amount_field",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "minimum_partition_count",
        "minimum_bad_count",
        "minimum_label_coverage",
        "minimum_historical_score_coverage",
        "maximum_group_coverage_gap",
        "diagnostic_severities",
    }
)
_SEVERITY_FIELDS = frozenset(
    {
        "entity_overlap",
        "temporal_oot",
        "risk_outside_approval",
        "maturity",
        "label_coverage",
        "historical_score_coverage",
        "group_coverage_gap",
        "sufficiency",
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
        "legacy_mapping",
        "warnings",
        "not_created_strategy",
        "not_adopted",
        "not_deployed",
    }
)
_MEMBERSHIP_ARTIFACT_OUTPUT_FIELDS = frozenset({"kind", "format", "filename"})
_BUNDLE_ARTIFACT_OUTPUT_FIELDS = frozenset(
    {"kind", "format", "filename", "content_hash"}
)
_ARTIFACTS_FIELDS = frozenset({"membership", "bundle"})
_LEGACY_MAPPING_FIELDS = frozenset(
    {"legacy_development_ref", "maps_to", "row_count", "row_equal"}
)
_MEMBERSHIP_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "format",
        "artifact_role",
        "task_id",
        "membership_id",
        "membership_content_hash",
        "membership_artifact_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "dataset_source_path",
        "dataset_registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "legacy_sample_design_ref",
    }
)
_BUNDLE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "format",
        "artifact_role",
        "task_id",
        "membership_id",
        "membership_content_hash",
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "bundle_id",
        "bundle_content_hash",
        "bundle_artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "dataset_source_path",
        "dataset_registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "legacy_sample_design_ref",
        "request",
        "request_hash",
    }
)
_RECORD_FIELDS = frozenset(
    {"id", "task_id", "kind", "path", "content_hash", "origin_tool", "provenance", "created_at"}
)
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DatasetContentDriftError,
    PredicateAstError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _LiveBinding:
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
    legacy: StrategySampleDesignExecutionBinding

    @property
    def target_col(self) -> str:
        return self.legacy.target_col

    @property
    def target_bad_value(self) -> int:
        return self.legacy.target_bad_value

    @property
    def drop_nan_labels(self) -> bool:
        return self.legacy.drop_nan_labels


@dataclass(frozen=True)
class StrategySampleDesignV2ArtifactBinding:
    """Strictly authenticated membership and V2 bundle artifact pair."""

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
    source_binding: _LiveBinding


def resolve_strategy_sample_design_v2_source_mode(
    sample_design: Mapping[str, Any],
    *,
    capability: str = "physical_v2",
    consumer: str | None = None,
) -> str:
    """Resolve authenticated V2 lineage without letting legacy consumers guess."""

    if capability not in {"physical_v2", "legacy_development"}:
        raise StrategyError("sample-design V2 source capability is invalid")
    design = _json_object(sample_design, "sample-design V2 source design")
    compatibility = _json_object(
        design.get("compatibility"),
        "sample-design V2 source compatibility",
    )
    fields = frozenset(compatibility)
    if fields == frozenset({"legacy_development_ref", "maps_to"}):
        if compatibility["maps_to"] != "risk/development":
            raise StrategyError(
                "sample-design V2 legacy compatibility mapping is invalid"
            )
        StrategySampleDesignRef.from_value(
            compatibility["legacy_development_ref"]
        )
        return "legacy_anchored"
    if fields == frozenset({"source_mode", "development_partition"}):
        if (
            compatibility["source_mode"] != "native_active_dataset"
            or compatibility["development_partition"] != "risk/development"
        ):
            raise StrategyError(
                "sample-design V2 native compatibility is invalid"
            )
        if capability == "legacy_development":
            raise StrategySampleDesignV2NativeSourceUnsupportedError(
                consumer=consumer or capability,
            )
        return "native_active_dataset"
    raise StrategyError(
        "sample-design V2 compatibility must use one exact supported shape"
    )


def run_materialize_sample_design_v2(inputs, ctx, runtime) -> dict[str, Any]:
    """Resolve, validate, and atomically publish one governed V2 sample design."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        binding = _load_live_binding(runtime, task_id=task_id, request=request)
        normalized = _normalize_request_against_columns(request, binding)
        frame = runtime.backend.read_frame(binding.dataset_path, columns=list(binding.columns))
        if not isinstance(frame, pd.DataFrame) or len(frame) != binding.row_count:
            raise StrategyError("sample-design V2 analysis universe row count changed")
        if sha256_file(binding.dataset_path) != binding.dataset_content_hash:
            raise StrategyError("sample-design V2 dataset bytes changed before computation")
        masks, predicate_refs, partition_ref = _resolve_masks(
            frame,
            request=normalized,
            binding=binding,
        )
        _require_observation_window(
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
        decoded = decode_sample_membership(membership_raw)
        _require_legacy_row_equality(frame, masks=masks, binding=binding.legacy)
        components = _build_components(
            frame,
            request=normalized,
            binding=binding,
            membership=decoded,
            masks=masks,
            predicate_refs=predicate_refs,
            partition_ref=partition_ref,
        )
        bundle = _build_bundle(
            frame=frame,
            request=normalized,
            binding=binding,
            membership=decoded,
            masks=masks,
            components=components,
            predicate_refs=predicate_refs,
            partition_ref=partition_ref,
        )
        _require_live_binding(runtime, binding=binding, request=normalized)
        return _persist_pair(
            runtime,
            task_id=task_id,
            request=normalized,
            binding=binding,
            membership_raw=membership_raw,
            membership=decoded,
            bundle=bundle,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_materialize_sample_design_v2_tool_output(value: object) -> dict[str, Any]:
    """Validate the complete cached Tool envelope without trusting display fields."""

    obj = _json_object(value, "materialize_sample_design_v2 output")
    _exact_fields(obj, _OUTPUT_FIELDS, "materialize_sample_design_v2 output")
    output_content_hash = _hash(
        obj["content_hash"], "materialize_sample_design_v2 output.content_hash"
    )
    bundle = validate_strategy_sample_design_v2_bundle(obj["bundle"])
    header = validate_sample_membership_header(obj["membership"])
    design = bundle["sample_design"]
    if bundle["membership"] != header:
        raise StrategyError("sample-design V2 output membership drifted from bundle")
    expected_scalars = {
        "schema_version": SAMPLE_DESIGN_V2_TOOL_SCHEMA_VERSION,
        "bundle_id": bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
    }
    for field, expected in expected_scalars.items():
        if obj[field] != expected:
            raise StrategyError(f"sample-design V2 output {field} drifted")
    artifacts = _json_object(obj["artifacts"], "sample-design V2 artifacts")
    _exact_fields(artifacts, _ARTIFACTS_FIELDS, "sample-design V2 artifacts")
    bundle_bytes = canonical_strategy_sample_design_v2_bundle_json(bundle).encode("utf-8")
    bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
    _validate_output_artifact(
        artifacts["bundle"],
        kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        artifact_format="json",
        filename=f"{bundle['bundle_id']}.json",
        expected_content_hash=bundle_hash,
    )
    _validate_output_artifact(
        artifacts["membership"],
        kind=SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
        artifact_format="binary",
        filename=f"{header['membership_id']}.bin",
        expected_content_hash=None,
    )
    mapping = _json_object(obj["legacy_mapping"], "sample-design V2 legacy_mapping")
    _exact_fields(mapping, _LEGACY_MAPPING_FIELDS, "sample-design V2 legacy_mapping")
    legacy_ref = StrategySampleDesignRef.from_value(
        design["compatibility"]["legacy_development_ref"]
    ).to_ref_dict()
    if mapping != {
        "legacy_development_ref": legacy_ref,
        "maps_to": "risk/development",
        "row_count": header["counts"]["risk"]["development"],
        "row_equal": True,
    }:
        raise StrategyError("sample-design V2 legacy mapping drifted")
    warnings = _warnings(bundle)
    if obj["warnings"] != warnings:
        raise StrategyError("sample-design V2 warnings drifted")
    for field in ("not_created_strategy", "not_adopted", "not_deployed"):
        if obj[field] is not True:
            raise StrategyError(f"sample-design V2 output {field} must be true")
    obj["bundle"] = bundle
    obj["membership"] = header
    addressed_body = {key: item for key, item in obj.items() if key != "content_hash"}
    expected_output_hash = hashlib.sha256(
        _canonical_json(addressed_body).encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(output_content_hash, expected_output_hash):
        raise StrategyError("sample-design V2 output content_hash does not match content")
    return obj


def load_strategy_sample_design_v2_artifacts(
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
) -> StrategySampleDesignV2ArtifactBinding:
    """Load and re-authenticate the inseparable membership/bundle artifact pair."""

    try:
        return _load_strategy_sample_design_v2_artifacts(
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
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def load_any_strategy_sample_design_v2_artifacts(
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
):
    """Dispatch exact V2 artifact refs by authenticated kind and origin."""

    normalized_task = _text(task_id, "task_id")
    membership_aid = _hash(
        membership_artifact_id,
        "membership_artifact_id",
    )
    bundle_aid = _hash(bundle_artifact_id, "bundle_artifact_id")
    membership_hash = _hash(
        expected_membership_artifact_content_hash,
        "expected_membership_artifact_content_hash",
    )
    bundle_hash = _hash(
        expected_bundle_artifact_content_hash,
        "expected_bundle_artifact_content_hash",
    )
    membership_record = _dispatch_record(
        runtime,
        task_id=normalized_task,
        artifact_id=membership_aid,
        expected_content_hash=membership_hash,
    )
    bundle_record = _dispatch_record(
        runtime,
        task_id=normalized_task,
        artifact_id=bundle_aid,
        expected_content_hash=bundle_hash,
    )
    pair = (
        membership_record["kind"],
        membership_record["origin_tool"],
        bundle_record["kind"],
        bundle_record["origin_tool"],
    )
    kwargs = {
        "task_id": normalized_task,
        "membership_artifact_id": membership_aid,
        "expected_membership_artifact_content_hash": membership_hash,
        "bundle_artifact_id": bundle_aid,
        "expected_bundle_artifact_content_hash": bundle_hash,
        "expected_bundle_id": expected_bundle_id,
        "expected_sample_design_id": expected_sample_design_id,
        "expected_sample_design_content_hash": (
            expected_sample_design_content_hash
        ),
    }
    legacy_pair = (
        SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_ORIGIN_TOOL,
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_ORIGIN_TOOL,
    )
    if pair == legacy_pair:
        return load_strategy_sample_design_v2_artifacts(
            runtime,
            **kwargs,
        )
    from marvis.packs.strategy.sample_design_v2_native_tools import (
        SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        load_native_strategy_sample_design_v2_artifacts,
    )

    native_pair = (
        SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    )
    if pair == native_pair:
        return load_native_strategy_sample_design_v2_artifacts(
            runtime,
            **kwargs,
        )
    raise StrategyError(
        "sample-design V2 artifact refs do not form one exact supported "
        "kind/origin pair"
    )


def load_historical_any_strategy_sample_design_v2_artifacts(
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
):
    """Dispatch immutable legacy or native V2 refs without requiring workspace head."""

    normalized_task = _text(task_id, "task_id")
    membership_aid = _hash(
        membership_artifact_id,
        "membership_artifact_id",
    )
    bundle_aid = _hash(bundle_artifact_id, "bundle_artifact_id")
    membership_hash = _hash(
        expected_membership_artifact_content_hash,
        "expected_membership_artifact_content_hash",
    )
    bundle_hash = _hash(
        expected_bundle_artifact_content_hash,
        "expected_bundle_artifact_content_hash",
    )
    membership_record = _dispatch_record(
        runtime,
        task_id=normalized_task,
        artifact_id=membership_aid,
        expected_content_hash=membership_hash,
    )
    bundle_record = _dispatch_record(
        runtime,
        task_id=normalized_task,
        artifact_id=bundle_aid,
        expected_content_hash=bundle_hash,
    )
    pair = (
        membership_record["kind"],
        membership_record["origin_tool"],
        bundle_record["kind"],
        bundle_record["origin_tool"],
    )
    kwargs = {
        "task_id": normalized_task,
        "membership_artifact_id": membership_aid,
        "expected_membership_artifact_content_hash": membership_hash,
        "bundle_artifact_id": bundle_aid,
        "expected_bundle_artifact_content_hash": bundle_hash,
        "expected_bundle_id": expected_bundle_id,
        "expected_sample_design_id": expected_sample_design_id,
        "expected_sample_design_content_hash": (
            expected_sample_design_content_hash
        ),
    }
    legacy_pair = (
        SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_ORIGIN_TOOL,
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_ORIGIN_TOOL,
    )
    if pair == legacy_pair:
        return load_historical_strategy_sample_design_v2_artifacts(
            runtime,
            **kwargs,
        )
    from marvis.packs.strategy.sample_design_v2_native_tools import (
        SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        load_historical_native_strategy_sample_design_v2_artifacts,
    )

    native_pair = (
        SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    )
    if pair == native_pair:
        return load_historical_native_strategy_sample_design_v2_artifacts(
            runtime,
            **kwargs,
        )
    raise StrategyError(
        "sample-design V2 artifact refs do not form one exact supported "
        "kind/origin pair"
    )


def _dispatch_record(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
) -> dict[str, Any]:
    try:
        record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    except (TaskArtifactDataError, TaskArtifactNotFoundError) as exc:
        raise StrategyError(str(exc)) from exc
    if (
        record is None
        or not isinstance(record, Mapping)
        or set(record) != _RECORD_FIELDS
        or record["id"] != artifact_id
        or record["task_id"] != task_id
        or not _matches_hash(
            record["content_hash"],
            expected_content_hash,
        )
    ):
        raise StrategyError(
            "sample-design V2 artifact dispatch binding changed"
        )
    return dict(record)


def load_historical_strategy_sample_design_v2_artifacts(
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
) -> StrategySampleDesignV2ArtifactBinding:
    """Load immutable V2 evidence without requiring its workspace as head."""

    try:
        return _load_strategy_sample_design_v2_artifacts(
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
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def require_strategy_sample_design_v2_artifact_binding_on_connection(
    conn,
    binding: StrategySampleDesignV2ArtifactBinding,
) -> None:
    """Revalidate a loaded V2 artifact pair while a downstream writer holds a lock."""

    _require_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        binding,
        require_current_workspace=True,
    )


def require_any_strategy_sample_design_v2_artifact_binding_on_connection(
    conn,
    binding,
) -> None:
    """Re-authenticate either exact V2 source mode without coercion."""

    if isinstance(binding, StrategySampleDesignV2ArtifactBinding):
        require_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            binding,
        )
        return
    from marvis.packs.strategy.sample_design_v2_native_tools import (
        StrategySampleDesignV2NativeArtifactBinding,
        require_native_strategy_sample_design_v2_artifact_binding_on_connection,
    )

    if isinstance(binding, StrategySampleDesignV2NativeArtifactBinding):
        require_native_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            binding,
        )
        return
    raise StrategyError("sample-design V2 artifact binding is invalid")


def require_historical_any_strategy_sample_design_v2_artifact_binding_on_connection(
    conn,
    binding,
) -> None:
    """Re-authenticate either immutable V2 source mode without workspace head."""

    if isinstance(binding, StrategySampleDesignV2ArtifactBinding):
        require_historical_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            binding,
        )
        return
    from marvis.packs.strategy.sample_design_v2_native_tools import (
        StrategySampleDesignV2NativeArtifactBinding,
        require_historical_native_strategy_sample_design_v2_artifact_binding_on_connection,
    )

    if isinstance(binding, StrategySampleDesignV2NativeArtifactBinding):
        require_historical_native_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            binding,
        )
        return
    raise StrategyError("sample-design V2 artifact binding is invalid")


def require_historical_strategy_sample_design_v2_artifact_binding_on_connection(
    conn,
    binding: StrategySampleDesignV2ArtifactBinding,
) -> None:
    """Revalidate immutable V2 evidence without requiring workspace head."""

    _require_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        binding,
        require_current_workspace=False,
    )


def _require_strategy_sample_design_v2_artifact_binding_on_connection(
    conn,
    binding: StrategySampleDesignV2ArtifactBinding,
    *,
    require_current_workspace: bool,
) -> None:
    if not isinstance(binding, StrategySampleDesignV2ArtifactBinding):
        raise StrategyError("sample-design V2 artifact binding is invalid")
    _require_live_binding_on_connection(
        conn,
        binding=binding.source_binding,
        require_current_workspace=require_current_workspace,
    )
    membership_raw = encode_sample_membership(
        task_id=binding.membership["header"]["task_id"],
        dataset_id=binding.membership["header"]["dataset_ref"]["dataset_id"],
        dataset_content_hash=binding.membership["header"]["dataset_ref"]["content_hash"],
        masks=binding.membership["masks"],
    )
    bundle_raw = canonical_strategy_sample_design_v2_bundle_json(
        binding.bundle
    ).encode("utf-8")
    checks = (
        (
            binding.membership_artifact_id,
            SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
            binding.membership_path,
            binding.membership_artifact_content_hash,
            binding.membership_provenance,
            membership_raw,
            _MAX_MEMBERSHIP_FILE_BYTES,
        ),
        (
            binding.bundle_artifact_id,
            SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
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
        row = _select_artifact_row(
            conn,
            task_id=binding.task_id,
            kind=kind,
            path=path,
        )
        if row is None:
            raise StrategyError("sample-design V2 artifact disappeared before write")
        if str(row["id"]) != artifact_id:
            raise StrategyError("sample-design V2 artifact identity changed before write")
        _require_existing_row(
            row,
            task_id=binding.task_id,
            kind=kind,
            path=path,
            content_hash=content_hash,
            provenance=provenance,
        )
        _require_exact_file(
            path,
            root=binding.membership_path.parents[2],
            canonical=canonical,
            content_hash=content_hash,
            maximum_bytes=maximum_bytes,
        )


def _load_strategy_sample_design_v2_artifacts(
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
) -> StrategySampleDesignV2ArtifactBinding:

    normalized_task = _text(task_id, "task_id")
    membership_aid = _hash(membership_artifact_id, "membership_artifact_id")
    membership_file_hash = _hash(
        expected_membership_artifact_content_hash,
        "expected_membership_artifact_content_hash",
    )
    bundle_aid = _hash(bundle_artifact_id, "bundle_artifact_id")
    bundle_file_hash = _hash(
        expected_bundle_artifact_content_hash,
        "expected_bundle_artifact_content_hash",
    )
    bundle_id = _text(expected_bundle_id, "expected_bundle_id")
    design_id = _text(expected_sample_design_id, "expected_sample_design_id")
    design_hash = _hash(
        expected_sample_design_content_hash,
        "expected_sample_design_content_hash",
    )
    membership_record = _registered_record(
        runtime,
        task_id=normalized_task,
        artifact_id=membership_aid,
        kind=SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
        expected_content_hash=membership_file_hash,
    )
    bundle_record = _registered_record(
        runtime,
        task_id=normalized_task,
        artifact_id=bundle_aid,
        kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        expected_content_hash=bundle_file_hash,
    )
    membership_path = _canonical_path(
        runtime.settings.tasks_dir,
        task_id=normalized_task,
        filename=Path(str(membership_record["path"])).name,
    )
    bundle_path = _canonical_path(
        runtime.settings.tasks_dir,
        task_id=normalized_task,
        filename=Path(str(bundle_record["path"])).name,
    )
    if Path(str(membership_record["path"])) != membership_path or Path(
        str(bundle_record["path"])
    ) != bundle_path:
        raise StrategyError("sample-design V2 artifact path is not canonical")
    membership_raw = _read_verified(
        membership_path,
        root=Path(runtime.settings.tasks_dir),
        expected_hash=membership_file_hash,
        maximum_bytes=_MAX_MEMBERSHIP_FILE_BYTES,
    )
    bundle_raw = _read_verified(
        bundle_path,
        root=Path(runtime.settings.tasks_dir),
        expected_hash=bundle_file_hash,
        maximum_bytes=MAX_SAMPLE_DESIGN_V2_JSON_BYTES,
    )
    membership = decode_sample_membership(membership_raw)
    try:
        bundle = strategy_sample_design_v2_bundle_from_json(bundle_raw)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise StrategyError("sample-design V2 bundle artifact is invalid") from exc
    canonical_bundle = canonical_strategy_sample_design_v2_bundle_json(bundle).encode("utf-8")
    if canonical_bundle != bundle_raw:
        raise StrategyError("sample-design V2 bundle bytes are not canonical")
    if bundle["membership"] != membership["header"]:
        raise StrategyError("sample-design V2 artifact pair membership changed")
    if membership_path.name != f"{membership['header']['membership_id']}.bin":
        raise StrategyError("sample-design V2 membership artifact filename changed")
    if bundle_path.name != f"{bundle_id}.json":
        raise StrategyError("sample-design V2 bundle artifact filename changed")
    design = bundle["sample_design"]
    if (
        bundle["bundle_id"] != bundle_id
        or design["sample_design_id"] != design_id
        or not hmac.compare_digest(design["content_hash"], design_hash)
    ):
        raise StrategyError("sample-design V2 artifact identity changed")
    membership_provenance = _validate_membership_provenance(
        membership_record["provenance"]
    )
    bundle_provenance = _validate_bundle_provenance(bundle_record["provenance"])
    _require_membership_provenance(
        membership_provenance,
        membership=membership,
        membership_file_hash=membership_file_hash,
    )
    _require_bundle_provenance(
        bundle_provenance,
        membership=membership,
        bundle=bundle,
        membership_artifact_id=membership_aid,
        membership_file_hash=membership_file_hash,
        bundle_file_hash=bundle_file_hash,
    )
    request = _validate_inputs(bundle_provenance["request"])
    binding = (
        _load_live_binding(
            runtime,
            task_id=normalized_task,
            request=request,
        )
        if require_current_workspace
        else _load_historical_live_binding(
            runtime,
            task_id=normalized_task,
            request=request,
            source=bundle_provenance,
        )
    )
    normalized_request = _normalize_request_against_columns(request, binding)
    if _canonical_json(normalized_request) != _canonical_json(bundle_provenance["request"]):
        raise StrategyError("sample-design V2 provenance request is not canonical")
    frame = runtime.backend.read_frame(binding.dataset_path, columns=list(binding.columns))
    if not isinstance(frame, pd.DataFrame) or len(frame) != binding.row_count:
        raise StrategyError("sample-design V2 loaded analysis universe row count changed")
    if sha256_file(binding.dataset_path) != binding.dataset_content_hash:
        raise StrategyError("sample-design V2 dataset bytes changed during artifact load")
    resolved_masks, predicate_refs, partition_ref = _resolve_masks(
        frame,
        request=normalized_request,
        binding=binding,
    )
    if any(
        not np.array_equal(resolved_masks[name], membership["masks"][name])
        for name in resolved_masks
    ):
        raise StrategyError("sample-design V2 persisted membership no longer matches its request")
    _require_observation_window(
        frame,
        request=normalized_request,
        masks=resolved_masks,
    )
    _require_legacy_row_equality(
        frame,
        masks=membership["masks"],
        binding=binding.legacy,
    )
    components = _build_components(
        frame,
        request=normalized_request,
        binding=binding,
        membership=membership,
        masks=resolved_masks,
        predicate_refs=predicate_refs,
        partition_ref=partition_ref,
    )
    rebuilt_bundle = _build_bundle(
        frame=frame,
        request=normalized_request,
        binding=binding,
        membership=membership,
        masks=resolved_masks,
        components=components,
        predicate_refs=predicate_refs,
        partition_ref=partition_ref,
    )
    if rebuilt_bundle != bundle:
        raise StrategyError("sample-design V2 bundle no longer matches deterministic evidence")
    _require_source_provenance(membership_provenance, binding=binding)
    _require_source_provenance(bundle_provenance, binding=binding)
    _require_live_binding(
        runtime,
        binding=binding,
        request=normalized_request,
        source=bundle_provenance,
        require_current_workspace=require_current_workspace,
    )
    return StrategySampleDesignV2ArtifactBinding(
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


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _json_object(value, "materialize_sample_design_v2 inputs")
    _exact_fields(obj, _INPUT_FIELDS, "materialize_sample_design_v2 inputs")
    result: dict[str, Any] = {
        "legacy_sample_design_ref": StrategySampleDesignRef.from_value(
            obj["legacy_sample_design_ref"]
        ).to_ref_dict(),
        "relationship": _enum(
            obj["relationship"],
            {"nested_same_cohort", "parallel_time_cohorts"},
            "relationship",
        ),
        "scope": _enum(
            obj["scope"], {"strategy_development", "exploration_only"}, "scope"
        ),
        "approval_population": _population_request(obj["approval_population"], "approval"),
        "risk_population": _population_request(obj["risk_population"], "risk"),
        "partitioning": _partitioning_request(obj["partitioning"]),
        "maturity": _maturity_request(obj["maturity"]),
        "performance_window": _performance_window(obj["performance_window"]),
        "observation_window": _observation_window(obj["observation_window"]),
        "field_bindings": _field_bindings(obj["field_bindings"]),
        "historical_score": _historical_score_request(obj["historical_score"]),
        "policy": _policy_request(obj["policy"]),
    }
    maturity = result["maturity"]
    performance = result["performance_window"]
    if maturity["status"] in {"confirmed_matured", "not_matured"}:
        if performance["status"] != "provided":
            raise StrategyError("evaluated maturity requires a provided performance window")
        if maturity["performance_window_days"] != performance["days"]:
            raise StrategyError("maturity and performance window days must match")
    expected_scope = (
        "strategy_development"
        if maturity["status"] == "confirmed_matured"
        and performance["status"] == "provided"
        and result["observation_window"]["status"] == "provided"
        else "exploration_only"
    )
    if result["scope"] != expected_scope:
        raise StrategyError("scope is inconsistent with windows and maturity")
    return result


def _population_request(value: object, role: str) -> dict[str, Any]:
    obj = _json_object(value, f"{role}_population")
    _exact_fields(obj, _POPULATION_REQUEST_FIELDS, f"{role}_population")
    return {"inclusion": obj["inclusion"], "exclusion": obj["exclusion"]}


def _partitioning_request(value: object) -> dict[str, Any]:
    obj = _json_object(value, "partitioning")
    method = _enum(obj.get("method"), {"predicate_ast", "time_ranges"}, "partitioning.method")
    if method == "predicate_ast":
        _exact_fields(obj, frozenset({"method", "selectors"}), "partitioning")
        selectors = _json_object(obj["selectors"], "partitioning.selectors")
        _exact_fields(selectors, frozenset(PARTITION_NAMES), "partitioning.selectors")
        return {"method": method, "selectors": selectors}
    _exact_fields(obj, frozenset({"method", "column", "ranges"}), "partitioning")
    ranges = _json_object(obj["ranges"], "partitioning.ranges")
    _exact_fields(ranges, frozenset(PARTITION_NAMES), "partitioning.ranges")
    normalized_ranges = {}
    for partition in PARTITION_NAMES:
        item = _json_object(ranges[partition], f"partitioning.ranges.{partition}")
        _exact_fields(item, frozenset({"start", "end"}), f"partitioning.ranges.{partition}")
        start = _optional_iso_date(item["start"], f"{partition}.start")
        end = _optional_iso_date(item["end"], f"{partition}.end")
        if start is None and end is None:
            raise StrategyError(f"partitioning range {partition} needs a bound")
        if start is not None and end is not None and start > end:
            raise StrategyError(f"partitioning range {partition} start exceeds end")
        normalized_ranges[partition] = {"start": start, "end": end}
    return {"method": method, "column": _text(obj["column"], "partitioning.column"), "ranges": normalized_ranges}


def _maturity_request(value: object) -> dict[str, Any]:
    obj = _json_object(value, "maturity")
    _exact_fields(
        obj,
        frozenset({"status", "performance_window_days", "cutoff_date", "reason"}),
        "maturity",
    )
    status = _enum(
        obj["status"],
        {"confirmed_matured", "not_matured", "unknown", "unavailable"},
        "maturity.status",
    )
    if status in {"confirmed_matured", "not_matured"}:
        days = _positive_int(obj["performance_window_days"], "maturity.performance_window_days")
        cutoff = _iso_date(obj["cutoff_date"], "maturity.cutoff_date")
        reason = None if status == "confirmed_matured" else _text(obj["reason"], "maturity.reason")
        if status == "confirmed_matured" and obj["reason"] is not None:
            raise StrategyError("confirmed_matured reason must be null")
    else:
        if obj["performance_window_days"] is not None or obj["cutoff_date"] is not None:
            raise StrategyError("unknown/unavailable maturity values must be null")
        days = None
        cutoff = None
        reason = _text(obj["reason"], "maturity.reason")
    return {"status": status, "performance_window_days": days, "cutoff_date": cutoff, "reason": reason}


def _performance_window(value: object) -> dict[str, Any]:
    obj = _json_object(value, "performance_window")
    _exact_fields(obj, frozenset({"status", "days"}), "performance_window")
    status = _enum(obj["status"], {"provided", "unavailable"}, "performance_window.status")
    if status == "provided":
        days = _positive_int(obj["days"], "performance_window.days")
    else:
        if obj["days"] is not None:
            raise StrategyError("unavailable performance_window.days must be null")
        days = None
    return {"status": status, "days": days}


def _observation_window(value: object) -> dict[str, Any]:
    obj = _json_object(value, "observation_window")
    _exact_fields(obj, frozenset({"status", "start", "end"}), "observation_window")
    status = _enum(obj["status"], {"provided", "unavailable"}, "observation_window.status")
    if status == "provided":
        start = _iso_date(obj["start"], "observation_window.start")
        end = _iso_date(obj["end"], "observation_window.end")
        if start > end:
            raise StrategyError("observation_window.start exceeds end")
    else:
        if obj["start"] is not None or obj["end"] is not None:
            raise StrategyError("unavailable observation window bounds must be null")
        start = end = None
    return {"status": status, "start": start, "end": end}


def _field_bindings(value: object) -> dict[str, str | None]:
    obj = _json_object(value, "field_bindings")
    _exact_fields(obj, _FIELD_BINDING_FIELDS, "field_bindings")
    return {field: _optional_text(obj[field], f"field_bindings.{field}") for field in sorted(_FIELD_BINDING_FIELDS)}


def _historical_score_request(value: object) -> dict[str, Any]:
    obj = _json_object(value, "historical_score")
    _exact_fields(obj, frozenset({"status", "column", "direction", "reason"}), "historical_score")
    status = _enum(obj["status"], {"available", "unavailable", "not_applicable"}, "historical_score.status")
    if status == "available":
        column = _text(obj["column"], "historical_score.column")
        direction = _enum(obj["direction"], {"higher_is_riskier", "lower_is_riskier"}, "historical_score.direction")
        if obj["reason"] is not None:
            raise StrategyError("available historical_score.reason must be null")
        reason = None
    else:
        if obj["column"] is not None or obj["direction"] is not None:
            raise StrategyError("non-available historical score fields must be null")
        column = direction = None
        reason = _text(obj["reason"], "historical_score.reason")
    return {"status": status, "column": column, "direction": direction, "reason": reason}


def _policy_request(value: object) -> dict[str, Any]:
    obj = _json_object(value, "policy")
    _exact_fields(obj, _POLICY_FIELDS, "policy")
    severities = _json_object(obj["diagnostic_severities"], "policy.diagnostic_severities")
    _exact_fields(severities, _SEVERITY_FIELDS, "policy.diagnostic_severities")
    return {
        "minimum_partition_count": _non_negative_int(obj["minimum_partition_count"], "minimum_partition_count"),
        "minimum_bad_count": _non_negative_int(obj["minimum_bad_count"], "minimum_bad_count"),
        "minimum_label_coverage": _ratio(obj["minimum_label_coverage"], "minimum_label_coverage"),
        "minimum_historical_score_coverage": _ratio(obj["minimum_historical_score_coverage"], "minimum_historical_score_coverage"),
        "maximum_group_coverage_gap": _ratio(obj["maximum_group_coverage_gap"], "maximum_group_coverage_gap"),
        "diagnostic_severities": {key: _enum(severities[key], {"warn", "fail"}, f"diagnostic_severities.{key}") for key in sorted(_SEVERITY_FIELDS)},
    }


def _load_live_binding(runtime, *, task_id: str, request: Mapping[str, Any]) -> _LiveBinding:
    workspace = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(task_id)
    if workspace.active_dataset_id is None or workspace.active_dataset_content_hash is None:
        raise StrategyError("sample-design V2 requires an active DataWorkspace dataset")
    ref = StrategySampleDesignRef.from_value(request["legacy_sample_design_ref"])
    legacy_artifact = load_strategy_sample_design_artifact(
        runtime,
        task_id=task_id,
        artifact_id=ref.artifact_id,
        expected_artifact_content_hash=ref.artifact_content_hash,
        expected_sample_design_id=ref.sample_design_id,
        expected_sample_design_content_hash=ref.sample_design_content_hash,
    )
    legacy_design = legacy_artifact.bundle["sample_design"]
    target_definition = legacy_design["target_definition"]
    optional = legacy_design["optional_fields"]
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    legacy = load_strategy_sample_design_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=ref.to_ref_dict(),
        dataset_id=workspace.active_dataset_id,
        dataset_content_hash=workspace.active_dataset_content_hash,
        workspace_revision=workspace.revision,
        workspace_generation=workspace.analysis_generation,
        semantic_mapping_hash=semantic_hash,
        target_col=target_definition["column"],
        drop_nan_labels=target_definition["drop_nan_labels"],
        month_col=optional["month_field"],
        weight_col=optional["weight_field"],
        loan_amount_col=optional["loan_amount_field"],
        overdue_amount_col=optional["overdue_amount_field"],
    )
    try:
        dataset = runtime.registry.get(workspace.active_dataset_id)
        path = Path(runtime.registry.resolve_verified_path(workspace.active_dataset_id))
    except (KeyError, OSError, TypeError, ValueError, DatasetContentDriftError) as exc:
        raise StrategyError("sample-design V2 active dataset is unavailable or drifted") from exc
    if str(dataset.task_id) != task_id:
        raise StrategyError("sample-design V2 dataset belongs to another task")
    if not _matches_hash(dataset.content_hash, workspace.active_dataset_content_hash):
        raise StrategyError("sample-design V2 dataset hash changed")
    if sha256_file(path) != workspace.active_dataset_content_hash:
        raise StrategyError("sample-design V2 dataset bytes changed")
    columns = tuple(str(column.name) for column in dataset.columns)
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = _dataset_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=str(dataset.id),
            expected_content_hash=str(dataset.content_hash),
        )
    return _LiveBinding(
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
                for column, role in workspace.semantic_mapping.field_roles.items()
            )
        ),
        legacy=legacy,
    )


def _load_historical_live_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    source: Mapping[str, Any],
) -> _LiveBinding:
    """Recover an immutable V2 source without consulting workspace head."""

    ref = StrategySampleDesignRef.from_value(request["legacy_sample_design_ref"])
    if source["task_id"] != task_id or source["legacy_sample_design_ref"] != (
        ref.to_ref_dict()
    ):
        raise StrategyError("sample-design V2 historical source binding changed")
    legacy_artifact = load_historical_strategy_sample_design_artifact(
        runtime,
        task_id=task_id,
        artifact_id=ref.artifact_id,
        expected_artifact_content_hash=ref.artifact_content_hash,
        expected_sample_design_id=ref.sample_design_id,
        expected_sample_design_content_hash=ref.sample_design_content_hash,
    )
    legacy_design = legacy_artifact.bundle["sample_design"]
    target_definition = legacy_design["target_definition"]
    optional = legacy_design["optional_fields"]
    legacy = load_historical_strategy_sample_design_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=ref.to_ref_dict(),
        dataset_id=source["dataset_id"],
        dataset_content_hash=source["dataset_content_hash"],
        workspace_revision=source["workspace_revision"],
        workspace_generation=source["workspace_generation"],
        semantic_mapping_hash=source["semantic_mapping_hash"],
        target_col=target_definition["column"],
        drop_nan_labels=target_definition["drop_nan_labels"],
        month_col=optional["month_field"],
        weight_col=optional["weight_field"],
        loan_amount_col=optional["loan_amount_field"],
        overdue_amount_col=optional["overdue_amount_field"],
    )
    try:
        dataset = runtime.registry.get(source["dataset_id"])
        path = Path(runtime.registry.resolve_verified_path(source["dataset_id"]))
    except (KeyError, OSError, TypeError, ValueError, DatasetContentDriftError) as exc:
        raise StrategyError(
            "sample-design V2 historical dataset is unavailable or drifted"
        ) from exc
    if (
        str(dataset.task_id) != task_id
        or str(dataset.id) != source["dataset_id"]
        or str(dataset.source_path) != source["dataset_source_path"]
        or not _matches_hash(
            dataset.content_hash,
            source["dataset_content_hash"],
        )
        or not hmac.compare_digest(
            sha256_file(path),
            source["dataset_content_hash"],
        )
    ):
        raise StrategyError("sample-design V2 historical dataset binding changed")
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = _dataset_metadata_hash_on_connection(
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
            "sample-design V2 historical dataset metadata changed"
        )
    time_field = request["field_bindings"]["time_field"]
    return _LiveBinding(
        task_id=task_id,
        dataset_id=str(dataset.id),
        dataset_content_hash=str(dataset.content_hash),
        dataset_source_path=str(dataset.source_path),
        dataset_path=path,
        dataset_registry_metadata_hash=metadata_hash,
        row_count=int(dataset.row_count),
        columns=tuple(str(column.name) for column in dataset.columns),
        workspace_revision=source["workspace_revision"],
        workspace_generation=source["workspace_generation"],
        semantic_mapping_hash=source["semantic_mapping_hash"],
        semantic_field_roles=(
            () if time_field is None else ((str(time_field), "date"),)
        ),
        legacy=legacy,
    )


def _normalize_request_against_columns(
    request: Mapping[str, Any], binding: _LiveBinding
) -> dict[str, Any]:
    allowed = set(binding.columns)
    result = _json_object(request, "sample-design V2 normalized request")
    field_bindings = result["field_bindings"]
    required = {value for value in field_bindings.values() if value is not None}
    historical = result["historical_score"]
    if historical["column"] is not None:
        required.add(historical["column"])
    if result["partitioning"]["method"] == "time_ranges":
        required.add(result["partitioning"]["column"])
    missing = sorted(required - allowed)
    if missing:
        raise StrategyError("sample-design V2 dataset is missing columns: " + ", ".join(missing))
    if result["maturity"]["status"] in {"confirmed_matured", "not_matured"} and field_bindings["time_field"] is None:
        raise StrategyError("evaluated maturity requires field_bindings.time_field")
    if (
        result["observation_window"]["status"] == "provided"
        and field_bindings["time_field"] is None
    ):
        raise StrategyError(
            "provided observation window requires field_bindings.time_field"
        )
    time_field = field_bindings["time_field"]
    if time_field is not None and dict(binding.semantic_field_roles).get(time_field) != "date":
        raise StrategyError(
            "field_bindings.time_field must be confirmed with the date semantic role"
        )
    if result["partitioning"]["method"] == "time_ranges" and result["partitioning"]["column"] != field_bindings["time_field"]:
        raise StrategyError("time_ranges column must equal field_bindings.time_field")
    for role in ("approval_population", "risk_population"):
        for key in ("inclusion", "exclusion"):
            predicate = result[role][key]
            if predicate is not None:
                    result[role][key] = canonicalize_predicate(
                        predicate,
                        binding.columns,
                    max_nodes=_MAX_PREDICATE_NODES,
                    max_depth=_MAX_PREDICATE_DEPTH,
                ).canonical
    if result["partitioning"]["method"] == "predicate_ast":
        result["partitioning"]["selectors"] = {
                partition: canonicalize_predicate(
                    result["partitioning"]["selectors"][partition],
                    binding.columns,
                max_nodes=_MAX_PREDICATE_NODES,
                max_depth=_MAX_PREDICATE_DEPTH,
            ).canonical
            for partition in PARTITION_NAMES
        }
    return _json_object(result, "sample-design V2 normalized request")


def _require_observation_window(
    frame: pd.DataFrame,
    *,
    request: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
) -> None:
    window = request["observation_window"]
    if window["status"] != "provided":
        return
    time_field = request["field_bindings"]["time_field"]
    if time_field is None:
        raise StrategyError(
            "provided observation window requires a bound time field"
        )
    dates = _date_series(frame[time_field], time_field)
    start = date.fromisoformat(window["start"])
    end = date.fromisoformat(window["end"])
    selected = _population_union(masks, "approval") | _population_union(
        masks, "risk"
    )
    outside = selected & np.array(
        [item < start or item > end for item in dates],
        dtype=np.bool_,
    )
    if bool(np.any(outside)):
        raise StrategyError(
            "sample-design V2 selected rows fall outside the observation window"
        )


def _resolve_masks(
    frame: pd.DataFrame,
    *,
    request: Mapping[str, Any],
    binding: _LiveBinding,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, str]]:
    cache: dict[str, np.ndarray] = {}
    predicate_refs: list[dict[str, str]] = []

    def evaluated(predicate: object | None, *, default: bool) -> np.ndarray:
        if predicate is None:
            return np.full(len(frame), default, dtype=np.bool_)
        canonical = _canonical_json(predicate)
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        ref = {"kind": "predicate_ast", "ref_id": f"predicate-{key[:24]}", "content_hash": key}
        if ref not in predicate_refs:
            predicate_refs.append(ref)
        if key not in cache:
            series = evaluate_predicate(
                frame,
                predicate,
                max_nodes=_MAX_PREDICATE_NODES,
                max_depth=_MAX_PREDICATE_DEPTH,
            )
            cache[key] = series.to_numpy(dtype=np.bool_, copy=True)
        return cache[key].copy()

    populations = {}
    population_refs = {}
    for role in ("approval", "risk"):
        controls = request[f"{role}_population"]
        inclusion = evaluated(controls["inclusion"], default=True)
        exclusion = evaluated(controls["exclusion"], default=False)
        populations[role] = inclusion & ~exclusion
        population_refs[role] = (
            _predicate_source_ref(controls["inclusion"]),
            _predicate_source_ref(controls["exclusion"]),
        )
    partitioning = request["partitioning"]
    partition_raw: dict[str, np.ndarray] = {}
    partition_ref = _request_source_ref("partition_definition", partitioning)
    if partitioning["method"] == "predicate_ast":
        for partition in PARTITION_NAMES:
            partition_raw[partition] = evaluated(
                partitioning["selectors"][partition], default=False
            )
    else:
        dates = _date_series(frame[partitioning["column"]], partitioning["column"])
        for partition in PARTITION_NAMES:
            bounds = partitioning["ranges"][partition]
            mask = np.ones(len(frame), dtype=np.bool_)
            if bounds["start"] is not None:
                mask &= np.array([item >= date.fromisoformat(bounds["start"]) for item in dates], dtype=np.bool_)
            if bounds["end"] is not None:
                mask &= np.array([item <= date.fromisoformat(bounds["end"]) for item in dates], dtype=np.bool_)
            partition_raw[partition] = mask
    masks: dict[str, np.ndarray] = {}
    for role in ("approval", "risk"):
        role_masks = [populations[role] & partition_raw[name] for name in PARTITION_NAMES]
        overlap = (role_masks[0] & role_masks[1]) | (role_masks[0] & role_masks[2]) | (role_masks[1] & role_masks[2])
        if bool(np.any(overlap)):
            raise StrategyError(f"sample-design V2 {role} partition selectors overlap")
        covered = np.logical_or.reduce(role_masks)
        if not np.array_equal(covered, populations[role]):
            raise StrategyError(f"sample-design V2 {role} partitions do not conserve the population")
        for partition, mask in zip(PARTITION_NAMES, role_masks, strict=True):
            masks[f"{role}/{partition}"] = np.ascontiguousarray(mask, dtype=np.bool_)
    if request["relationship"] == "nested_same_cohort":
        for partition in PARTITION_NAMES:
            if bool(np.any(masks[f"risk/{partition}"] & ~masks[f"approval/{partition}"])):
                raise StrategyError(f"nested_same_cohort risk is outside approval in {partition}")
    # Store resolved predicate refs on the normalized request only through
    # provenance refs; caller identity remains the canonical AST itself.
    del population_refs
    return masks, sorted(predicate_refs, key=lambda item: item["ref_id"]), partition_ref


def _require_legacy_row_equality(
    frame: pd.DataFrame,
    *,
    masks: Mapping[str, np.ndarray],
    binding: StrategySampleDesignExecutionBinding,
) -> None:
    ordinal = "__marvis_sample_v2_row_ordinal__"
    while ordinal in frame.columns:
        ordinal += "_"
    required = {binding.target_col}
    if binding.split_column is not None:
        required.add(binding.split_column)
    projection = frame.loc[:, [column for column in frame.columns if column in required]].copy()
    projection[ordinal] = np.arange(len(frame), dtype=np.int64)
    selected = bind_strategy_development_frame(
        projection,
        binding=binding,
        normalize_target=False,
    )
    expected = np.zeros(len(frame), dtype=np.bool_)
    expected[selected[ordinal].to_numpy(dtype=np.int64)] = True
    if not np.array_equal(expected, masks["risk/development"]):
        raise StrategyError(
            "sample-design V2 risk/development rows do not equal the exact legacy development membership"
        )


def _build_components(
    frame: pd.DataFrame,
    *,
    request: Mapping[str, Any],
    binding: _LiveBinding,
    membership: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    predicate_refs: Sequence[Mapping[str, str]],
    partition_ref: Mapping[str, str],
) -> dict[str, Any]:
    header = membership["header"]
    dataset_ref = _dataset_source_ref(binding)
    legacy_ref = _legacy_source_ref(binding.legacy)
    population_refs = _population_predicate_refs(request)
    target = build_target_selector_v2(
        status="resolved",
        column=binding.legacy.target_col,
        good_value=1 - binding.legacy.target_bad_value,
        bad_value=binding.legacy.target_bad_value,
        drop_missing=binding.legacy.drop_nan_labels,
        source_refs=[legacy_ref],
    )
    maturity, eligible_mask = _maturity_evidence(
        frame,
        request=request,
        binding=binding,
        risk_mask=_population_union(masks, "risk"),
        source_ref=dataset_ref,
    )
    approval = build_sample_population_v2(
        role="approval",
        membership_header=header,
        inclusion_predicate_ref=population_refs["approval"][0],
        exclusion_predicate_ref=population_refs["approval"][1],
        source_refs=_unique_refs([dataset_ref, partition_ref, *predicate_refs]),
    )
    risk = build_sample_population_v2(
        role="risk",
        membership_header=header,
        inclusion_predicate_ref=population_refs["risk"][0],
        exclusion_predicate_ref=population_refs["risk"][1],
        maturity_evidence=maturity,
        source_refs=_unique_refs([dataset_ref, legacy_ref, partition_ref, *predicate_refs]),
    )
    historical_request = request["historical_score"]
    historical = build_historical_score_v2(
        status=historical_request["status"],
        column=historical_request["column"],
        direction=historical_request["direction"],
        source_refs=(
            [
                _field_source_ref(
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


def _build_bundle(
    *,
    frame: pd.DataFrame,
    request: Mapping[str, Any],
    binding: _LiveBinding,
    membership: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    components: Mapping[str, Any],
    predicate_refs: Sequence[Mapping[str, str]],
    partition_ref: Mapping[str, str],
) -> dict[str, Any]:
    header = membership["header"]
    sources = _unique_refs(
        [_dataset_source_ref(binding), _legacy_source_ref(binding.legacy), partition_ref, *predicate_refs]
    )
    split = _split_definition(request["partitioning"], partition_ref)
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
        legacy_development_ref=binding.legacy.to_ref_dict(),
        source_refs=sources,
    )
    statistics = _diagnostic_statistics(
        frame=frame,
        binding=binding,
        request=request,
        masks=masks,
        eligible_mask=components["eligible_mask"],
    )
    observations = _metric_observations(
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
        legacy_development_ref=binding.legacy.to_ref_dict(),
        diagnostic_statistics=statistics,
        metric_observations=observations,
        source_refs=sources,
    )


def _maturity_evidence(
    frame: pd.DataFrame,
    *,
    request: Mapping[str, Any],
    binding: _LiveBinding,
    risk_mask: np.ndarray,
    source_ref: Mapping[str, str],
) -> tuple[dict[str, Any], np.ndarray | None]:
    control = request["maturity"]
    status = control["status"]
    if status not in {"confirmed_matured", "not_matured"}:
        return (
            {
                "status": status,
                "performance_window_days": None,
                "cutoff_date": None,
                "eligible_count": None,
                "labeled_count": None,
                "source_refs": [source_ref],
                "reason": control["reason"],
            },
            None,
        )
    time_field = request["field_bindings"]["time_field"]
    if time_field is None:
        raise StrategyError("evaluated maturity requires a bound time field")
    dates = _date_series(frame[time_field], time_field)
    maturity_date = date.fromisoformat(control["cutoff_date"]) - timedelta(
        days=control["performance_window_days"]
    )
    eligible = risk_mask & np.array(
        [item <= maturity_date for item in dates], dtype=np.bool_
    )
    target = frame[binding.target_col]
    labeled = _binary_label_mask(target)
    eligible_count = int(np.count_nonzero(eligible))
    labeled_count = int(np.count_nonzero(eligible & labeled))
    risk_count = int(np.count_nonzero(risk_mask))
    if status == "confirmed_matured" and eligible_count != risk_count:
        raise StrategyError(
            "confirmed_matured requires every risk row to satisfy the deterministic maturity cutoff"
        )
    if status == "not_matured" and (
        risk_count == 0 or eligible_count >= risk_count
    ):
        raise StrategyError(
            "not_matured requires at least one risk row outside the deterministic maturity cutoff"
        )
    return (
        {
            "status": status,
            "performance_window_days": control["performance_window_days"],
            "cutoff_date": control["cutoff_date"],
            "eligible_count": eligible_count,
            "labeled_count": labeled_count,
            "source_refs": [source_ref],
            "reason": control["reason"],
        },
        eligible,
    )


def _diagnostic_statistics(
    *,
    frame: pd.DataFrame,
    binding: _LiveBinding,
    request: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    eligible_mask: np.ndarray | None,
) -> dict[str, Any]:
    dataset_ref = _dataset_source_ref(binding)
    risk_union = _population_union(masks, "risk")
    fields = request["field_bindings"]
    entity_field = fields["entity_field"]
    if entity_field is None:
        entity = {
            "availability": "unavailable",
            "overlap_count": None,
            "compared_count": None,
            "source_refs": [],
        }
    else:
        partition_sets: list[set[str]] = []
        for partition in PARTITION_NAMES:
            values = frame.loc[masks[f"risk/{partition}"], entity_field].dropna()
            try:
                partition_sets.append(
                    {_canonical_json(_json_scalar(item)) for item in values.tolist()}
                )
            except (TypeError, ValueError) as exc:
                raise StrategyError(
                    "sample-design V2 entity field must contain JSON scalar values"
                ) from exc
        seen: set[str] = set()
        overlap: set[str] = set()
        for values in partition_sets:
            overlap.update(seen & values)
            seen.update(values)
        entity = {
            "availability": "available",
            "overlap_count": len(overlap),
            "compared_count": len(seen),
            "source_refs": [dataset_ref],
        }
    time_field = fields["time_field"]
    oot_union = masks["approval/oot"] | masks["risk/oot"]
    non_oot = np.logical_or.reduce(
        [
            masks["approval/development"],
            masks["approval/validation"],
            masks["risk/development"],
            masks["risk/validation"],
        ]
    )
    if not bool(np.any(oot_union)):
        temporal = {"availability": "not_applicable", "ordered": None, "source_refs": []}
    elif time_field is None:
        temporal = {"availability": "unavailable", "ordered": None, "source_refs": []}
    else:
        dates = _date_series(frame[time_field], time_field)
        oot_dates = [item for item, selected in zip(dates, oot_union, strict=True) if selected]
        prior_dates = [item for item, selected in zip(dates, non_oot, strict=True) if selected]
        ordered = not prior_dates or max(prior_dates) < min(oot_dates)
        temporal = {"availability": "available", "ordered": ordered, "source_refs": [dataset_ref]}
    historical = request["historical_score"]
    if historical["status"] == "available":
        values = frame.loc[risk_union, historical["column"]]
        score_coverage = {
            "availability": "available",
            "covered_count": int(values.notna().sum()),
            "eligible_count": int(np.count_nonzero(risk_union)),
            "source_refs": [dataset_ref],
        }
    else:
        score_coverage = {
            "availability": historical["status"],
            "covered_count": None,
            "eligible_count": None,
            "source_refs": [],
        }
    group_field = fields["group_field"]
    if group_field is None or not bool(np.any(risk_union)):
        group_gap = {
            "availability": "unavailable" if group_field is None else "not_applicable",
            "maximum_gap": None,
            "group_count": None,
            "source_refs": [],
        }
    else:
        labeled = _binary_label_mask(frame[binding.target_col])
        denominator_mask = risk_union if eligible_mask is None else risk_union & eligible_mask
        groups = frame.loc[denominator_mask, group_field]
        coverages: list[float] = []
        for group in groups.dropna().drop_duplicates().tolist():
            group_mask = denominator_mask & _scalar_equal_mask(frame[group_field], group)
            denominator = int(np.count_nonzero(group_mask))
            if denominator:
                coverages.append(int(np.count_nonzero(group_mask & labeled)) / denominator)
        maximum_gap = max(coverages) - min(coverages) if coverages else 0.0
        group_gap = {
            "availability": "available",
            "maximum_gap": maximum_gap,
            "group_count": len(coverages),
            "source_refs": [dataset_ref],
        }
    maturity_status = request["maturity"]["status"]
    if maturity_status != "confirmed_matured" or eligible_mask is None:
        sufficiency = {"availability": "unavailable", "bad_count": None, "source_refs": []}
    else:
        labels = _binary_label_mask(frame[binding.target_col])
        bad = _bad_label_mask(frame[binding.target_col], binding.target_bad_value)
        dev = masks["risk/development"] & eligible_mask & labels
        sufficiency = {
            "availability": "available",
            "bad_count": int(np.count_nonzero(dev & bad)),
            "source_refs": [dataset_ref],
        }
    return {
        "entity_overlap": entity,
        "temporal_oot": temporal,
        "historical_score_coverage": score_coverage,
        "group_coverage_gap": group_gap,
        "sufficiency": sufficiency,
    }


def _metric_observations(
    frame: pd.DataFrame,
    *,
    binding: _LiveBinding,
    masks: Mapping[str, np.ndarray],
    eligible_mask: np.ndarray | None,
    maturity_status: str,
    membership_header: Mapping[str, Any],
    sample_design: Mapping[str, Any],
) -> list[dict[str, Any]]:
    definitions = {item["metric_key"]: item for item in build_metric_definitions_v2()}
    design_ref = {
        "sample_design_id": sample_design["sample_design_id"],
        "content_hash": sample_design["content_hash"],
    }
    sources = [
        {
            "kind": "dataset",
            "ref_id": membership_header["dataset_ref"]["dataset_id"],
            "content_hash": membership_header["dataset_ref"]["content_hash"],
        },
        {
            "kind": "sample_membership",
            "ref_id": membership_header["membership_id"],
            "content_hash": membership_header["content_hash"],
        },
        {
            "kind": "sample_design",
            "ref_id": sample_design["sample_design_id"],
            "content_hash": sample_design["content_hash"],
        },
    ]
    labeled_all = _binary_label_mask(frame[binding.target_col])
    bad_all = _bad_label_mask(frame[binding.target_col], binding.target_bad_value)
    observations: list[dict[str, Any]] = []
    for role in ("approval", "risk"):
        slice_masks = {partition: masks[f"{role}/{partition}"] for partition in PARTITION_NAMES}
        slice_masks["overall"] = _population_union(masks, role)
        for partition in ("overall", *PARTITION_NAMES):
            population_mask = slice_masks[partition]
            population_count = int(np.count_nonzero(population_mask))
            values: dict[str, tuple[str, Any, Any, Any]] = {
                "population_count": ("present", population_count, population_count, population_count)
            }
            if role == "approval":
                for key in ("labeled_count", "label_coverage", "bad_count", "bad_rate"):
                    values[key] = ("not_applicable", None, None, None)
            else:
                evaluated_mask = population_mask.copy()
                if maturity_status in {"confirmed_matured", "not_matured"}:
                    if eligible_mask is None:
                        raise StrategyError(
                            "evaluated maturity is missing deterministic eligibility"
                        )
                    evaluated_mask &= eligible_mask
                labeled = int(np.count_nonzero(evaluated_mask & labeled_all))
                values["labeled_count"] = ("present", labeled, labeled, population_count)
                values["label_coverage"] = (
                    ("present", labeled / population_count, labeled, population_count)
                    if population_count
                    else ("insufficient_data", None, None, None)
                )
                if maturity_status == "not_matured":
                    bad_status = "not_matured"
                elif maturity_status in {"unknown", "unavailable"}:
                    bad_status = "unavailable"
                else:
                    bad_status = "present"
                if bad_status == "present":
                    bad = int(np.count_nonzero(evaluated_mask & labeled_all & bad_all))
                    values["bad_count"] = ("present", bad, bad, labeled)
                    values["bad_rate"] = (
                        ("present", bad / labeled, bad, labeled)
                        if labeled
                        else ("insufficient_data", None, None, None)
                    )
                else:
                    values["bad_count"] = (bad_status, None, None, None)
                    values["bad_rate"] = (bad_status, None, None, None)
            for metric_key, (status, value, numerator, denominator) in values.items():
                observations.append(
                    build_metric_observation_v2(
                        sample_design_ref=design_ref,
                        metric_definition=definitions[metric_key],
                        population=role,
                        partition=partition,
                        status=status,
                        value=value,
                        numerator=numerator,
                        denominator=denominator,
                        sample_count=population_count,
                        source_refs=sources,
                    )
                )
    return observations


def _split_definition(partitioning: Mapping[str, Any], source_ref: Mapping[str, str]) -> dict[str, Any]:
    if partitioning["method"] == "predicate_ast":
        return {
            "status": "available",
            "method": "precomputed_masks",
            "column": None,
            "development_values": [],
            "validation_values": [],
            "oot_values": [],
            "source_refs": [source_ref],
        }
    return {
        "status": "available",
        "method": "time_ranges",
        "column": partitioning["column"],
        **{
            f"{partition}_values": [
                _canonical_json(partitioning["ranges"][partition])
            ]
            for partition in PARTITION_NAMES
        },
        "source_refs": [source_ref],
    }


def _persist_pair(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    binding: _LiveBinding,
    membership_raw: bytes,
    membership: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_raw = canonical_strategy_sample_design_v2_bundle_json(bundle).encode("utf-8")
    membership_file_hash = hashlib.sha256(membership_raw).hexdigest()
    bundle_file_hash = hashlib.sha256(bundle_raw).hexdigest()
    header = membership["header"]
    design = bundle["sample_design"]
    out_dir = _prepare_output_directory(runtime.settings.tasks_dir, task_id=task_id)
    membership_path = out_dir / f"{header['membership_id']}.bin"
    bundle_path = out_dir / f"{bundle['bundle_id']}.json"
    request_evidence = _json_object(request, "sample-design V2 request evidence")
    request_hash = hashlib.sha256(
        _canonical_json(request_evidence).encode("utf-8")
    ).hexdigest()
    source_provenance = {
        "schema_version": SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION,
        "task_id": task_id,
        "dataset_id": binding.dataset_id,
        "dataset_content_hash": binding.dataset_content_hash,
        "dataset_source_path": binding.dataset_source_path,
        "dataset_registry_metadata_hash": binding.dataset_registry_metadata_hash,
        "workspace_revision": binding.workspace_revision,
        "workspace_generation": binding.workspace_generation,
        "semantic_mapping_hash": binding.semantic_mapping_hash,
        "legacy_sample_design_ref": binding.legacy.to_ref_dict(),
    }
    membership_provenance = {
        **source_provenance,
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
        raise StrategyError("sample-design V2 artifacts could not be staged") from exc
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_live_binding_on_connection(conn, binding=binding)
                membership_row = _select_artifact_row(
                    conn,
                    task_id=task_id,
                    kind=SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
                    path=membership_path,
                )
                _prepare_one_artifact_under_lock(
                    row=membership_row,
                    staged=staged_membership,
                    task_id=task_id,
                    kind=SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
                    path=membership_path,
                    canonical=membership_raw,
                    content_hash=membership_file_hash,
                    provenance=membership_provenance,
                    root=Path(runtime.settings.tasks_dir),
                    maximum_bytes=_MAX_MEMBERSHIP_FILE_BYTES,
                )
                membership_record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
                    path=str(membership_path),
                    content_hash=membership_file_hash,
                    origin_tool=SAMPLE_DESIGN_V2_ORIGIN_TOOL,
                    provenance=membership_provenance,
                )
                bundle_provenance = {
                    **source_provenance,
                    "format": "json",
                    "artifact_role": "bundle",
                    "membership_id": header["membership_id"],
                    "membership_content_hash": header["content_hash"],
                    "membership_artifact_id": membership_record["id"],
                    "membership_artifact_content_hash": membership_file_hash,
                    "bundle_id": bundle["bundle_id"],
                    "bundle_content_hash": bundle["content_hash"],
                    "bundle_artifact_content_hash": bundle_file_hash,
                    "sample_design_id": design["sample_design_id"],
                    "sample_design_content_hash": design["content_hash"],
                    "request": request_evidence,
                    "request_hash": request_hash,
                }
                bundle_row = _select_artifact_row(
                    conn,
                    task_id=task_id,
                    kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
                    path=bundle_path,
                )
                _prepare_one_artifact_under_lock(
                    row=bundle_row,
                    staged=staged_bundle,
                    task_id=task_id,
                    kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
                    path=bundle_path,
                    canonical=bundle_raw,
                    content_hash=bundle_file_hash,
                    provenance=bundle_provenance,
                    root=Path(runtime.settings.tasks_dir),
                    maximum_bytes=MAX_SAMPLE_DESIGN_V2_JSON_BYTES,
                )
                bundle_record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
                    path=str(bundle_path),
                    content_hash=bundle_file_hash,
                    origin_tool=SAMPLE_DESIGN_V2_ORIGIN_TOOL,
                    provenance=bundle_provenance,
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
    return validate_materialize_sample_design_v2_tool_output(
        _tool_output(
            membership=membership,
            bundle=bundle,
            membership_record=membership_record,
            bundle_record=bundle_record,
        )
    )


def _tool_output(
    *,
    membership: Mapping[str, Any],
    bundle: Mapping[str, Any],
    membership_record: Mapping[str, Any],
    bundle_record: Mapping[str, Any],
) -> dict[str, Any]:
    header = membership["header"]
    design = bundle["sample_design"]
    body = {
        "schema_version": SAMPLE_DESIGN_V2_TOOL_SCHEMA_VERSION,
        "bundle_id": bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "bundle": dict(bundle),
        "membership": dict(header),
        "artifacts": {
            "membership": _artifact_output(
                record=membership_record,
                artifact_format="binary",
                include_content_hash=False,
            ),
            "bundle": _artifact_output(
                record=bundle_record,
                artifact_format="json",
                include_content_hash=True,
            ),
        },
        "legacy_mapping": {
            "legacy_development_ref": design["compatibility"]["legacy_development_ref"],
            "maps_to": "risk/development",
            "row_count": header["counts"]["risk"]["development"],
            "row_equal": True,
        },
        "warnings": _warnings(bundle),
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    return {
        **body,
        "content_hash": hashlib.sha256(
            _canonical_json(body).encode("utf-8")
        ).hexdigest(),
    }


def _artifact_output(
    *,
    record: Mapping[str, Any],
    artifact_format: str,
    include_content_hash: bool,
) -> dict[str, str]:
    output = {
        "kind": str(record["kind"]),
        "format": artifact_format,
        "filename": Path(str(record["path"])).name,
    }
    if include_content_hash:
        output["content_hash"] = str(record["content_hash"])
    return output


def _warnings(bundle: Mapping[str, Any]) -> list[str]:
    return [
        str(item["message"])
        for item in bundle["diagnostics"]
        if item["status"] in {"warn", "fail", "unavailable"}
    ]


def _require_live_binding(
    runtime,
    *,
    binding: _LiveBinding,
    request: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
    require_current_workspace: bool = True,
) -> None:
    current = (
        _load_live_binding(
            runtime,
            task_id=binding.task_id,
            request=request,
        )
        if require_current_workspace
        else _load_historical_live_binding(
            runtime,
            task_id=binding.task_id,
            request=request,
            source=(
                source
                if source is not None
                else {
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
                    "legacy_sample_design_ref": (
                        binding.legacy.reference.to_ref_dict()
                    ),
                }
            ),
        )
    )
    if current != binding:
        raise StrategyError("sample-design V2 live binding changed during computation")


def _require_live_binding_on_connection(
    conn,
    *,
    binding: _LiveBinding,
    require_current_workspace: bool = True,
) -> None:
    if require_current_workspace:
        require_strategy_sample_design_execution_binding_on_connection(
            conn,
            binding.legacy,
        )
    else:
        require_historical_strategy_sample_design_execution_binding_on_connection(
            conn,
            binding.legacy,
        )
    metadata_hash = _dataset_metadata_hash_on_connection(
        conn,
        task_id=binding.task_id,
        dataset_id=binding.dataset_id,
        expected_content_hash=binding.dataset_content_hash,
    )
    if not hmac.compare_digest(metadata_hash, binding.dataset_registry_metadata_hash):
        raise StrategyError("sample-design V2 dataset registry metadata changed")
    row = conn.execute(
        "SELECT source_path FROM datasets WHERE task_id = ? AND id = ?",
        (binding.task_id, binding.dataset_id),
    ).fetchone()
    if row is None or str(row["source_path"]) != binding.dataset_source_path:
        raise StrategyError("sample-design V2 dataset registry path changed")
    if sha256_file(binding.dataset_path) != binding.dataset_content_hash:
        raise StrategyError("sample-design V2 dataset bytes changed before registration")


def _dataset_metadata_hash_on_connection(
    conn,
    *,
    task_id: str,
    dataset_id: str,
    expected_content_hash: str,
) -> str:
    row = conn.execute(
        """
        SELECT task_id, row_count, columns_json, has_target, target_col,
               content_hash, source_path
          FROM datasets WHERE id = ?
        """,
        (dataset_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError("sample-design V2 dataset is not task-owned")
    if not _matches_hash(row["content_hash"], expected_content_hash):
        raise StrategyError("sample-design V2 registered dataset hash changed")
    columns_json = row["columns_json"]
    if not isinstance(columns_json, str):
        raise StrategyError("sample-design V2 dataset schema is invalid")
    try:
        columns = json.loads(columns_json)
    except json.JSONDecodeError as exc:
        raise StrategyError("sample-design V2 dataset schema is invalid") from exc
    payload = {
        "row_count": int(row["row_count"]),
        "columns": columns,
        "has_target": int(row["has_target"]),
        "target_col": row["target_col"],
        "source_path": str(row["source_path"]),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _select_artifact_row(conn, *, task_id: str, kind: str, path: Path):
    return conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, kind, str(path)),
    ).fetchone()


def _prepare_one_artifact_under_lock(
    *,
    row,
    staged,
    task_id: str,
    kind: str,
    path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
    root: Path,
    maximum_bytes: int,
    origin_tool: str = SAMPLE_DESIGN_V2_ORIGIN_TOOL,
) -> None:
    file_exists = path.exists() or path.is_symlink()
    if row is not None:
        if not file_exists:
            raise StrategyError("sample-design V2 registry exists without artifact file")
        _require_existing_row(
            row,
            task_id=task_id,
            kind=kind,
            path=path,
            content_hash=content_hash,
            provenance=provenance,
            origin_tool=origin_tool,
        )
        _require_exact_file(
            path,
            root=root,
            canonical=canonical,
            content_hash=content_hash,
            maximum_bytes=maximum_bytes,
        )
        staged.rollback()
        return
    if file_exists:
        # Recover an exact content-addressed file left behind after promotion
        # but before its SQLite transaction committed.
        _require_exact_file(
            path,
            root=root,
            canonical=canonical,
            content_hash=content_hash,
            maximum_bytes=maximum_bytes,
        )
        staged.rollback()
        return
    staged.promote()
    _require_exact_file(
        path,
        root=root,
        canonical=canonical,
        content_hash=content_hash,
        maximum_bytes=maximum_bytes,
    )


def _require_existing_row(
    row,
    *,
    task_id: str,
    kind: str,
    path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
    origin_tool: str = SAMPLE_DESIGN_V2_ORIGIN_TOOL,
) -> None:
    expected = {
        "task_id": task_id,
        "kind": kind,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": origin_tool,
        "provenance_json": _canonical_json(provenance),
    }
    if any(str(row[field]) != expected[field] for field in expected):
        raise StrategyError("existing sample-design V2 artifact registry row changed")


def _registered_record(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    kind: str,
    expected_content_hash: str,
    origin_tool: str = SAMPLE_DESIGN_V2_ORIGIN_TOOL,
) -> dict[str, Any]:
    try:
        record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    except (TaskArtifactDataError, TaskArtifactNotFoundError) as exc:
        raise StrategyError(str(exc)) from exc
    if record is None or not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise StrategyError("sample-design V2 artifact registry row is invalid")
    if (
        record["id"] != artifact_id
        or record["task_id"] != task_id
        or record["kind"] != kind
        or record["origin_tool"] != origin_tool
        or not _matches_hash(record["content_hash"], expected_content_hash)
    ):
        raise StrategyError("sample-design V2 artifact registry binding changed")
    return dict(record)


def _validate_membership_provenance(value: object) -> dict[str, Any]:
    obj = _json_object(value, "sample-design V2 membership provenance")
    _exact_fields(
        obj,
        _MEMBERSHIP_PROVENANCE_FIELDS,
        "sample-design V2 membership provenance",
    )
    _validate_provenance_source_fields(obj)
    if obj["format"] != "binary" or obj["artifact_role"] != "membership":
        raise StrategyError("sample-design V2 membership provenance role is invalid")
    for field in (
        "membership_content_hash",
        "membership_artifact_content_hash",
    ):
        _hash(obj[field], f"membership provenance.{field}")
    _text(obj["membership_id"], "membership provenance.membership_id")
    return obj


def _validate_bundle_provenance(value: object) -> dict[str, Any]:
    obj = _json_object(value, "sample-design V2 bundle provenance")
    _exact_fields(
        obj,
        _BUNDLE_PROVENANCE_FIELDS,
        "sample-design V2 bundle provenance",
    )
    _validate_provenance_source_fields(obj)
    if obj["format"] != "json" or obj["artifact_role"] != "bundle":
        raise StrategyError("sample-design V2 bundle provenance role is invalid")
    for field in (
        "membership_content_hash",
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "bundle_content_hash",
        "bundle_artifact_content_hash",
        "sample_design_content_hash",
        "request_hash",
    ):
        _hash(obj[field], f"bundle provenance.{field}")
    for field in ("membership_id", "bundle_id", "sample_design_id"):
        _text(obj[field], f"bundle provenance.{field}")
    request = _validate_inputs(obj["request"])
    if not hmac.compare_digest(
        obj["request_hash"],
        hashlib.sha256(_canonical_json(request).encode("utf-8")).hexdigest(),
    ):
        raise StrategyError("sample-design V2 bundle provenance request_hash changed")
    obj["request"] = request
    return obj


def _validate_provenance_source_fields(obj: Mapping[str, Any]) -> None:
    if obj["schema_version"] != SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION:
        raise StrategyError("sample-design V2 provenance schema_version is invalid")
    if obj["producer_version"] != STRATEGY_SAMPLE_DESIGN_V2_PRODUCER_VERSION:
        raise StrategyError("sample-design V2 provenance producer_version is invalid")
    for field in (
        "dataset_content_hash",
        "dataset_registry_metadata_hash",
        "semantic_mapping_hash",
    ):
        _hash(obj[field], f"provenance.{field}")
    for field in ("task_id", "dataset_id", "dataset_source_path"):
        _text(obj[field], f"provenance.{field}")
    _non_negative_int(obj["workspace_revision"], "provenance.workspace_revision")
    _non_negative_int(obj["workspace_generation"], "provenance.workspace_generation")
    StrategySampleDesignRef.from_value(obj["legacy_sample_design_ref"])


def _require_membership_provenance(
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
            raise StrategyError(f"sample-design V2 membership provenance {field} changed")


def _require_bundle_provenance(
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
        "dataset_content_hash": design["identity"]["dataset_ref"]["content_hash"],
        "workspace_revision": design["identity"]["workspace_ref"]["revision"],
        "workspace_generation": design["identity"]["workspace_ref"]["generation"],
        "semantic_mapping_hash": design["identity"]["workspace_ref"]["semantic_mapping_hash"],
        "legacy_sample_design_ref": design["compatibility"]["legacy_development_ref"],
    }
    for field, expected_value in expected.items():
        if provenance[field] != expected_value:
            raise StrategyError(f"sample-design V2 bundle provenance {field} changed")


def _require_source_provenance(
    provenance: Mapping[str, Any], *, binding: _LiveBinding
) -> None:
    expected = {
        "task_id": binding.task_id,
        "dataset_id": binding.dataset_id,
        "dataset_content_hash": binding.dataset_content_hash,
        "dataset_source_path": binding.dataset_source_path,
        "dataset_registry_metadata_hash": binding.dataset_registry_metadata_hash,
        "workspace_revision": binding.workspace_revision,
        "workspace_generation": binding.workspace_generation,
        "semantic_mapping_hash": binding.semantic_mapping_hash,
        "legacy_sample_design_ref": binding.legacy.to_ref_dict(),
    }
    for field, expected_value in expected.items():
        if provenance[field] != expected_value:
            raise StrategyError(f"sample-design V2 source provenance {field} changed")


def _prepare_output_directory(tasks_dir: Path | str, *, task_id: str) -> Path:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise StrategyError("task_id cannot escape task storage")
    root = Path(tasks_dir).absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StrategyError("task artifact root must be a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / task_id
    if task_dir.exists() and (task_dir.is_symlink() or not task_dir.is_dir()):
        raise StrategyError("task artifact directory must be a regular directory")
    task_dir.mkdir(exist_ok=True)
    if task_dir.is_symlink() or task_dir.resolve(strict=True).parent != root.resolve(strict=True):
        raise StrategyError("sample-design V2 task directory escaped storage")
    out_dir = task_dir / "strategy_sample_designs_v2"
    if out_dir.exists() and (out_dir.is_symlink() or not out_dir.is_dir()):
        raise StrategyError("sample-design V2 output path must be a regular directory")
    out_dir.mkdir(exist_ok=True)
    if out_dir.is_symlink() or out_dir.resolve(strict=True).parent != task_dir.resolve(strict=True):
        raise StrategyError("sample-design V2 output directory escaped storage")
    return out_dir


def _canonical_path(tasks_dir: Path | str, *, task_id: str, filename: str) -> Path:
    if Path(task_id).name != task_id or Path(filename).name != filename:
        raise StrategyError("sample-design V2 artifact identity is not path-safe")
    return Path(tasks_dir).absolute() / task_id / "strategy_sample_designs_v2" / filename


def _read_verified(path: Path, *, root: Path, expected_hash: str, maximum_bytes: int) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise StrategyError("sample-design V2 artifact must be a regular file")
    root = root.absolute()
    current = path.parent
    while current != root:
        if current.is_symlink():
            raise StrategyError("sample-design V2 artifact path traverses a symlink")
        if current == current.parent:
            break
        current = current.parent
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError("sample-design V2 artifact escaped task storage") from exc
    try:
        with path.open("rb") as stream:
            raw = stream.read(maximum_bytes + 1)
            if len(raw) > maximum_bytes or stream.read(1):
                raise StrategyError("sample-design V2 artifact exceeds byte budget")
    except OSError as exc:
        raise StrategyError("sample-design V2 artifact could not be read") from exc
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_hash):
        raise StrategyError("sample-design V2 artifact content hash changed")
    return raw


def _require_exact_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
    maximum_bytes: int,
) -> None:
    if _read_verified(path, root=root, expected_hash=content_hash, maximum_bytes=maximum_bytes) != canonical:
        raise StrategyError("sample-design V2 artifact bytes changed")


def _validate_output_artifact(
    value: object,
    *,
    kind: str,
    artifact_format: str,
    filename: str,
    expected_content_hash: str | None,
) -> dict[str, Any]:
    obj = _json_object(value, "sample-design V2 output artifact")
    expected = {
        "kind": kind,
        "format": artifact_format,
        "filename": filename,
    }
    if expected_content_hash is not None:
        _exact_fields(
            obj,
            _BUNDLE_ARTIFACT_OUTPUT_FIELDS,
            "sample-design V2 output artifact",
        )
        _hash(obj["content_hash"], "artifact content_hash")
        expected["content_hash"] = expected_content_hash
    else:
        _exact_fields(
            obj,
            _MEMBERSHIP_ARTIFACT_OUTPUT_FIELDS,
            "sample-design V2 output artifact",
        )
    if obj != expected:
        raise StrategyError("sample-design V2 output artifact drifted")
    return expected


def _dataset_source_ref(binding: _LiveBinding) -> dict[str, str]:
    return {
        "kind": "dataset",
        "ref_id": binding.dataset_id,
        "content_hash": binding.dataset_content_hash,
    }


def _legacy_source_ref(binding: StrategySampleDesignExecutionBinding) -> dict[str, str]:
    return {
        "kind": "legacy_sample_design",
        "ref_id": binding.reference.sample_design_id,
        "content_hash": binding.reference.sample_design_content_hash,
    }


def _field_source_ref(binding: _LiveBinding, *, kind: str, field: str) -> dict[str, str]:
    identity = hashlib.sha256(
        _canonical_json(
            {
                "dataset_id": binding.dataset_id,
                "dataset_content_hash": binding.dataset_content_hash,
                "field": field,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {"kind": kind, "ref_id": f"{field}-{identity[:16]}", "content_hash": identity}


def _request_source_ref(kind: str, value: object) -> dict[str, str]:
    content_hash = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return {"kind": kind, "ref_id": f"{kind}-{content_hash[:24]}", "content_hash": content_hash}


def _predicate_source_ref(value: object | None) -> dict[str, str] | None:
    if value is None:
        return None
    content_hash = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return {
        "kind": "predicate_ast",
        "ref_id": f"predicate-{content_hash[:24]}",
        "content_hash": content_hash,
    }


def _population_predicate_refs(request: Mapping[str, Any]) -> dict[str, tuple[dict[str, str] | None, dict[str, str] | None]]:
    return {
        role: (
            _predicate_source_ref(request[f"{role}_population"]["inclusion"]),
            _predicate_source_ref(request[f"{role}_population"]["exclusion"]),
        )
        for role in ("approval", "risk")
    }


def _unique_refs(values: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    unique = {
        (item["kind"], item["ref_id"]): {
            "kind": item["kind"],
            "ref_id": item["ref_id"],
            "content_hash": item["content_hash"],
        }
        for item in values
    }
    return sorted(unique.values(), key=lambda item: (item["kind"], item["ref_id"]))


def _population_union(masks: Mapping[str, np.ndarray], role: str) -> np.ndarray:
    return np.logical_or.reduce([masks[f"{role}/{partition}"] for partition in PARTITION_NAMES])


def _date_series(series: pd.Series, field: str) -> list[date]:
    result: list[date] = []
    for raw in series.tolist():
        missing = pd.isna(raw)
        if raw is None or (
            isinstance(missing, (bool, np.bool_)) and bool(missing)
        ):
            raise StrategyError(
                f"sample-design V2 {field} contains missing or invalid dates"
            )
        if isinstance(raw, bool) or isinstance(raw, Real):
            raise StrategyError(
                f"sample-design V2 {field} rejects numeric epoch-style dates"
            )
        if isinstance(raw, pd.Timestamp):
            result.append(raw.date())
            continue
        if isinstance(raw, datetime):
            result.append(raw.date())
            continue
        if isinstance(raw, date):
            result.append(raw)
            continue
        if not isinstance(raw, str) or not raw.strip():
            raise StrategyError(
                f"sample-design V2 {field} is not a valid date field"
            )
        text = raw.strip()
        try:
            parsed = (
                datetime.fromisoformat(text.replace("Z", "+00:00")).date()
                if "T" in text or " " in text
                else date.fromisoformat(text)
            )
        except ValueError as exc:
            raise StrategyError(
                f"sample-design V2 {field} must contain ISO calendar dates"
            ) from exc
        result.append(parsed)
    return result


def _binary_label_mask(series: pd.Series) -> np.ndarray:
    result = np.zeros(len(series), dtype=np.bool_)
    for index, raw in enumerate(series.tolist()):
        if _is_missing(raw):
            continue
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
            raise StrategyError("sample-design V2 target must contain numeric 0/1 or missing")
        number = float(raw)
        if not math.isfinite(number) or number not in {0.0, 1.0}:
            raise StrategyError("sample-design V2 target must contain numeric 0/1 or missing")
        result[index] = True
    return result


def _bad_label_mask(series: pd.Series, bad_value: int) -> np.ndarray:
    labeled = _binary_label_mask(series)
    values = np.zeros(len(series), dtype=np.bool_)
    for index, raw in enumerate(series.tolist()):
        if labeled[index]:
            values[index] = int(float(raw)) == bad_value
    return values


def _scalar_equal_mask(series: pd.Series, value: object) -> np.ndarray:
    result = []
    for item in series.tolist():
        if _is_missing(item):
            result.append(False)
        else:
            try:
                result.append(bool(item == value))
            except (TypeError, ValueError):
                result.append(False)
    return np.asarray(result, dtype=np.bool_)


def _is_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _json_scalar(value: object) -> str | bool | int | float:
    if isinstance(value, np.generic):
        value = value.item()
    if _is_missing(value) or not isinstance(value, (str, bool, int, float)):
        raise TypeError("value is not a non-null JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("value is not finite")
    return value


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{name} must be an object with string keys")
    try:
        encoded = _canonical_json(dict(value))
        normalized = json.loads(encoded)
    except (TypeError, ValueError, RecursionError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} must be canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(f"{name} fields are invalid ({'; '.join(details)})")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be non-empty text")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _matches_hash(value: object, expected: str) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None and hmac.compare_digest(value, expected)


def _enum(value: object, allowed: set[str], name: str) -> str:
    normalized = _text(value, name)
    if normalized not in allowed:
        raise StrategyError(f"{name} must be one of: " + ", ".join(sorted(allowed)))
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _non_negative_int(value, name)
    if result < 1:
        raise StrategyError(f"{name} must be at least 1")
    return result


def _ratio(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyError(f"{name} must be a ratio")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or normalized > 1:
        raise StrategyError(f"{name} must be between 0 and 1")
    return normalized


def _iso_date(value: object, name: str) -> str:
    normalized = _text(value, name)
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as exc:
        raise StrategyError(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != normalized:
        raise StrategyError(f"{name} must be a canonical ISO date")
    return normalized


def _optional_iso_date(value: object, name: str) -> str | None:
    return None if value is None else _iso_date(value, name)


__all__ = [
    "SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION",
    "SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND",
    "SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND",
    "SAMPLE_DESIGN_V2_ORIGIN_TOOL",
    "SAMPLE_DESIGN_V2_TOOL_SCHEMA_VERSION",
    "StrategySampleDesignV2ArtifactBinding",
    "load_historical_any_strategy_sample_design_v2_artifacts",
    "load_historical_strategy_sample_design_v2_artifacts",
    "load_any_strategy_sample_design_v2_artifacts",
    "load_strategy_sample_design_v2_artifacts",
    "require_any_strategy_sample_design_v2_artifact_binding_on_connection",
    "require_historical_any_strategy_sample_design_v2_artifact_binding_on_connection",
    "require_historical_strategy_sample_design_v2_artifact_binding_on_connection",
    "require_strategy_sample_design_v2_artifact_binding_on_connection",
    "resolve_strategy_sample_design_v2_source_mode",
    "run_materialize_sample_design_v2",
    "validate_materialize_sample_design_v2_tool_output",
]
