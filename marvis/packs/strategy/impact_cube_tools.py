"""Governed Tool boundary for unified deterministic Strategy ImpactCube evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.packs.strategy.dsl import (
    StrategySpec,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube import (
    MAX_IMPACT_CUBE_JSON_BYTES,
    STRATEGY_IMPACT_CUBE_PRODUCER_VERSION,
    build_strategy_impact_cube,
    canonical_strategy_impact_cube_json,
    validate_strategy_impact_cube,
)
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
    StrategyPoolDevelopmentExecutionBinding,
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
    require_strategy_candidate_pool_artifact_binding_on_connection,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    ResolvedPoolRequirements,
    hydrate_requirement_fields,
    pool_requirement_bindings_provenance,
    project_pool_entry_requirements,
    require_resolved_pool_requirements_on_connection,
    resolve_pool_requirements,
    validate_pool_requirement_bindings_provenance,
)
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
    load_strategy_sample_design_v2_artifacts,
    require_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.repositories.strategy import (
    _strategy_from_row,
    _strategy_spec_hash_from_row,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
    _stable_artifact_id,
)


IMPACT_CUBE_TOOL_SCHEMA_VERSION = "strategy.measure-impact-cube-tool.v3"
IMPACT_CUBE_ARTIFACT_KIND = "strategy_impact_cube_json"
IMPACT_CUBE_ARTIFACT_SCHEMA_VERSION = "strategy.impact-cube-artifact.v3"
IMPACT_CUBE_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION = (
    "strategy.impact-cube-artifact.v4"
)
IMPACT_CUBE_ORIGIN_TOOL = "strategy.measure_strategy_impact_cube"
IMPACT_CUBE_MEASUREMENT_RUN_SCHEMA_VERSION = (
    "strategy.impact-cube-measurement-run.v1"
)
IMPACT_CUBE_MEASUREMENT_AUDIT_KIND = (
    "strategy.impact-cube.measurement.completed"
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_POOL_ID_RE = re.compile(r"^strategy-pool-[0-9a-f]{32}$")
_POOL_REVISION_ID_RE = re.compile(
    r"^strategy-pool-revision-[0-9a-f]{32}$"
)
_MEASUREMENT_RUN_ID_RE = re.compile(
    r"^strategy-impact-cube-run-[0-9a-f]{24}$"
)
_PARTITION_ORDER = ("development", "validation", "oot")
_PARTITIONS = frozenset(_PARTITION_ORDER)
_STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)
_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "pool_ref",
        "sample_design_ref",
        "partitions",
        "population",
        "dimension_bindings",
        "current_strategy_ref",
        "economics_inputs",
    }
)
_OPTIONAL_INPUT_FIELDS = frozenset(
    {"current_strategy_ref", "economics_inputs"}
)
_POOL_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_pool_id",
        "expected_revision",
        "expected_revision_id",
        "expected_snapshot_hash",
    }
)
_SAMPLE_DESIGN_REF_FIELDS = frozenset(
    {
        "membership_artifact_id",
        "expected_membership_artifact_content_hash",
        "bundle_artifact_id",
        "expected_bundle_artifact_content_hash",
        "expected_bundle_id",
        "expected_sample_design_id",
        "expected_sample_design_content_hash",
    }
)
_DIMENSION_FIELDS = frozenset(
    {"month_col", "group_col", "segment_col"}
)
_DATASET_BINDING_FIELDS = frozenset(
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
_TARGET_BINDING_FIELDS = frozenset(
    {"column", "good_value", "bad_value", "missing_policy"}
)
_CURRENT_REQUEST_FIELDS = frozenset(
    {"strategy_id", "expected_strategy_spec_hash"}
)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "cube_id",
        "content_hash",
        "pool_id",
        "pool_revision",
        "pool_snapshot_hash",
        "strategy_type",
        "partitions",
        "slice_count",
        "cube",
        "warnings",
        "artifact",
        "producer_run_ref",
        "not_mutated_pool",
        "not_created_strategy",
        "not_adopted",
        "not_promoted",
        "not_deployed",
    }
)
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "format",
        "filename",
        "content_hash",
        "download_url",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "cube_id",
        "cube_content_hash",
        "pool_ref",
        "sample_design_ref",
        "dataset_binding",
        "target_binding",
        "dimension_bindings",
        "current_strategy_ref",
        "economics_inputs",
        "partitions",
        "populations",
        "lifecycle",
        "producer_run",
    }
)
_REQUIREMENTS_PROVENANCE_FIELDS = _PROVENANCE_FIELDS | {
    "requirement_bindings"
}
_PRODUCER_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "content_hash",
        "input_hash",
        "task_id",
        "request",
        "tool_ref",
        "cube_ref",
        "artifact_ref",
    }
)
_PRODUCER_TOOL_REF_FIELDS = frozenset(
    {
        "plugin",
        "tool",
        "origin_tool",
        "tool_schema_version",
        "producer_version",
    }
)
_PRODUCER_CUBE_REF_FIELDS = frozenset({"cube_id", "content_hash"})
_PRODUCER_ARTIFACT_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "filename",
        "content_hash",
        "origin_tool",
    }
)
_PRODUCER_RUN_REF_FIELDS = frozenset(
    {"kind", "ref_id", "content_hash"}
)
_LIFECYCLE = {
    "mutates_pool": False,
    "creates_strategy": False,
    "adopts_strategy": False,
    "promotes_strategy": False,
    "deploys_strategy": False,
}
_TASK_ARTIFACT_ROW_FIELDS = (
    "id",
    "task_id",
    "kind",
    "path",
    "content_hash",
    "origin_tool",
    "provenance_json",
    "created_at",
)
_MAX_ECONOMIC_COMPONENTS = 16
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _CurrentStrategyBinding:
    strategy_id: str
    strategy_type: str
    spec: StrategySpec
    spec_hash: str


def run_measure_strategy_impact_cube(inputs, ctx, runtime) -> dict[str, Any]:
    """Build and atomically publish one exact, aggregate-only ImpactCube."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        pool = _load_pool_binding(
            runtime,
            task_id=task_id,
            request=request,
        )
        sample = _load_sample_design_binding(
            runtime,
            task_id=task_id,
            request=request,
        )
        development = bind_strategy_pool_development_execution(runtime, pool)
        resolved_requirements = resolve_pool_requirements(
            runtime,
            task_id=task_id,
            compiled_design={
                "requirements": list(
                    project_pool_entry_requirements(pool.pool["entries"])
                )
            },
            sample_design=sample,
        )
        semantics = _require_sample_contract(
            pool=pool,
            development=development,
            sample=sample,
            partitions=request["partitions"],
        )
        current = _load_current_strategy(
            runtime,
            task_id=task_id,
            strategy_type=request["strategy_type"],
            value=request["current_strategy_ref"],
        )
        _require_bindings_under_lock(
            runtime,
            pool=pool,
            sample=sample,
            current=current,
            task_id=task_id,
            resolved_requirements=resolved_requirements,
        )
        population_frames = _read_partition_frames(
            runtime,
            pool=pool,
            sample=sample,
            current=current,
            request=request,
            target_col=semantics["target_col"],
            loan_amount_col=semantics["loan_amount_col"],
            overdue_amount_col=semantics["overdue_amount_col"],
            resolved_requirements=resolved_requirements,
        )
        cube = build_strategy_impact_cube(
            pool=pool.pool,
            partition_frames=population_frames["risk"],
            approval_partition_frames=population_frames["approval"],
            pool_artifact_ref={
                "artifact_id": pool.artifact_id,
                "artifact_content_hash": pool.artifact_content_hash,
            },
            sample_design_v2_ref=_sample_design_evidence_ref(
                sample,
                partitions=request["partitions"],
            ),
            dataset_binding=_dataset_evidence_binding(sample),
            legacy_development_ref=semantics["legacy_development_ref"],
            target_col=semantics["target_col"],
            target_bad_value=semantics["target_bad_value"],
            month_col=request["dimension_bindings"]["month_col"],
            group_col=request["dimension_bindings"]["group_col"],
            segment_col=request["dimension_bindings"]["segment_col"],
            current_strategy_spec=(
                None if current is None else current.spec
            ),
            current_strategy_ref=(
                None
                if current is None
                else {
                    "strategy_id": current.strategy_id,
                    "strategy_type": current.strategy_type,
                    "strategy_spec_hash": current.spec_hash,
                }
            ),
            economics_bindings=request["economics_inputs"],
            semantic_field_roles=dict(
                sample.source_binding.semantic_field_roles
            ),
            entity_col=semantics["entity_col"],
            loan_amount_col=semantics["loan_amount_col"],
            overdue_amount_col=semantics["overdue_amount_col"],
        )
        _require_dataset_unchanged(sample)
        return _persist_cube(
            runtime,
            task_id=task_id,
            request=request,
            pool=pool,
            sample=sample,
            current=current,
            resolved_requirements=resolved_requirements,
            cube=cube,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_measure_strategy_impact_cube_tool_output(
    value: object,
    *,
    trusted_task_id: str,
    trusted_artifact_id: str,
    trusted_artifact_content_hash: str,
    trusted_producer_run_id: str,
    trusted_producer_run_content_hash: str,
) -> dict[str, Any]:
    """Validate every cached scalar against the canonical embedded cube."""

    obj = _json_object(value, "measure_strategy_impact_cube output")
    _exact_fields(
        obj,
        _OUTPUT_FIELDS,
        "measure_strategy_impact_cube output",
    )
    cube = validate_strategy_impact_cube(obj["cube"])
    identity = cube["identity"]
    task_id = _text(trusted_task_id, "trusted task_id")
    artifact_id = _hash(
        trusted_artifact_id,
        "trusted artifact_id",
    )
    artifact_content_hash = _hash(
        trusted_artifact_content_hash,
        "trusted artifact content_hash",
    )
    if identity["task_id"] != task_id:
        raise StrategyError(
            "measure_strategy_impact_cube trusted task_id drifted"
        )
    expected = {
        "schema_version": IMPACT_CUBE_TOOL_SCHEMA_VERSION,
        "cube_id": cube["cube_id"],
        "content_hash": cube["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "strategy_type": identity["strategy_type"],
        "partitions": [
            row["name"]
            for row in cube["partitions"]
            if row["role"] == "risk"
        ],
        "slice_count": len(cube["slices"]),
    }
    for field, expected_value in expected.items():
        if obj[field] != expected_value:
            raise StrategyError(
                f"measure_strategy_impact_cube output {field} drifted"
            )
    warnings = [
        str(flag["message"])
        for flag in cube["red_flags"]
        if flag["level"] in {"amber", "red"}
    ]
    if obj["warnings"] != warnings:
        raise StrategyError(
            "measure_strategy_impact_cube output warnings drifted"
        )
    for field in (
        "not_mutated_pool",
        "not_created_strategy",
        "not_adopted",
        "not_promoted",
        "not_deployed",
    ):
        if obj[field] is not True:
            raise StrategyError(
                f"measure_strategy_impact_cube output {field} must be true"
            )

    artifact = _json_object(
        obj["artifact"],
        "measure_strategy_impact_cube artifact",
    )
    _exact_fields(
        artifact,
        _OUTPUT_ARTIFACT_FIELDS,
        "measure_strategy_impact_cube artifact",
    )
    output_artifact_id = _hash(artifact["artifact_id"], "artifact_id")
    if not hmac.compare_digest(output_artifact_id, artifact_id):
        raise StrategyError(
            "measure_strategy_impact_cube trusted artifact_id drifted"
        )
    canonical = canonical_strategy_impact_cube_json(cube).encode("utf-8")
    canonical_hash = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(canonical_hash, artifact_content_hash):
        raise StrategyError(
            "measure_strategy_impact_cube trusted artifact content_hash "
            "drifted"
        )
    expected_artifact = {
        "artifact_id": artifact_id,
        "kind": IMPACT_CUBE_ARTIFACT_KIND,
        "format": "json",
        "filename": f"{cube['cube_id']}.json",
        "content_hash": artifact_content_hash,
        "download_url": (
            f"/api/tasks/{quote(identity['task_id'], safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
            f"?expected_content_hash={artifact_content_hash}"
        ),
    }
    if artifact != expected_artifact:
        raise StrategyError(
            "measure_strategy_impact_cube artifact binding drifted"
        )
    producer_run_ref = _json_object(
        obj["producer_run_ref"],
        "measure_strategy_impact_cube producer_run_ref",
    )
    _exact_fields(
        producer_run_ref,
        _PRODUCER_RUN_REF_FIELDS,
        "measure_strategy_impact_cube producer_run_ref",
    )
    expected_run_id = _text(
        trusted_producer_run_id,
        "trusted producer_run id",
    )
    if _MEASUREMENT_RUN_ID_RE.fullmatch(expected_run_id) is None:
        raise StrategyError("trusted producer_run id is invalid")
    expected_run_content_hash = _hash(
        trusted_producer_run_content_hash,
        "trusted producer_run content_hash",
    )
    if producer_run_ref != {
        "kind": "tool_run",
        "ref_id": expected_run_id,
        "content_hash": expected_run_content_hash,
    }:
        raise StrategyError(
            "measure_strategy_impact_cube producer_run_ref drifted"
        )
    obj["cube"] = cube
    return obj


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _json_object(value, "measure_strategy_impact_cube inputs")
    unknown = sorted(set(obj) - _INPUT_FIELDS)
    missing = sorted((_INPUT_FIELDS - _OPTIONAL_INPUT_FIELDS) - set(obj))
    if unknown or missing:
        raise StrategyError(
            "measure_strategy_impact_cube inputs fields are invalid"
        )
    strategy_type = _text(obj["strategy_type"], "strategy_type")
    if strategy_type not in _STRATEGY_TYPES:
        raise StrategyError("strategy_type is unsupported")
    partitions = _partition_list(obj["partitions"])
    if obj["population"] != "risk":
        raise StrategyError("population must be risk")
    return {
        "strategy_type": strategy_type,
        "pool_ref": _validate_pool_ref(obj["pool_ref"]),
        "sample_design_ref": _validate_sample_design_ref(
            obj["sample_design_ref"]
        ),
        "partitions": partitions,
        "population": "risk",
        "dimension_bindings": _validate_dimension_bindings(
            obj["dimension_bindings"]
        ),
        "current_strategy_ref": _validate_current_strategy_ref(
            obj.get("current_strategy_ref")
        ),
        "economics_inputs": _validate_economics_inputs(
            obj.get("economics_inputs")
        ),
    }


def _partition_list(value: object) -> list[str]:
    rows = _list(value, "partitions")
    if not rows or len(rows) > len(_PARTITION_ORDER):
        raise StrategyError("partitions must select one to three partitions")
    normalized = [_text(row, "partitions item") for row in rows]
    if any(row not in _PARTITIONS for row in normalized):
        raise StrategyError("partitions contain an unsupported partition")
    if len(normalized) != len(set(normalized)):
        raise StrategyError("partitions must not contain duplicates")
    return [
        partition
        for partition in _PARTITION_ORDER
        if partition in set(normalized)
    ]


def _validate_pool_ref(value: object) -> dict[str, Any]:
    obj = _json_object(value, "pool_ref")
    _exact_fields(obj, _POOL_REF_FIELDS, "pool_ref")
    pool_id = _text(obj["expected_pool_id"], "pool_ref.expected_pool_id")
    if _POOL_ID_RE.fullmatch(pool_id) is None:
        raise StrategyError("pool_ref.expected_pool_id is invalid")
    revision_id = _text(
        obj["expected_revision_id"],
        "pool_ref.expected_revision_id",
    )
    if _POOL_REVISION_ID_RE.fullmatch(revision_id) is None:
        raise StrategyError("pool_ref.expected_revision_id is invalid")
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "pool_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "pool_ref.expected_artifact_content_hash",
        ),
        "expected_pool_id": pool_id,
        "expected_revision": _positive_int(
            obj["expected_revision"],
            "pool_ref.expected_revision",
        ),
        "expected_revision_id": revision_id,
        "expected_snapshot_hash": _hash(
            obj["expected_snapshot_hash"],
            "pool_ref.expected_snapshot_hash",
        ),
    }


def _validate_sample_design_ref(value: object) -> dict[str, str]:
    obj = _json_object(value, "sample_design_ref")
    _exact_fields(
        obj,
        _SAMPLE_DESIGN_REF_FIELDS,
        "sample_design_ref",
    )
    result: dict[str, str] = {}
    for field in (
        "membership_artifact_id",
        "expected_membership_artifact_content_hash",
        "bundle_artifact_id",
        "expected_bundle_artifact_content_hash",
        "expected_sample_design_content_hash",
    ):
        result[field] = _hash(obj[field], f"sample_design_ref.{field}")
    for field in ("expected_bundle_id", "expected_sample_design_id"):
        result[field] = _text(obj[field], f"sample_design_ref.{field}")
    return result


def _validate_dimension_bindings(value: object) -> dict[str, str | None]:
    obj = _json_object(value, "dimension_bindings")
    _exact_fields(obj, _DIMENSION_FIELDS, "dimension_bindings")
    result = {
        field: _optional_text(obj[field], f"dimension_bindings.{field}")
        for field in sorted(_DIMENSION_FIELDS)
    }
    bound = [field for field in result.values() if field is not None]
    if len(bound) != len(set(bound)):
        raise StrategyError("dimension bindings must use distinct columns")
    return result


def _validate_current_strategy_ref(
    value: object,
) -> dict[str, str] | None:
    if value is None:
        return None
    obj = _json_object(value, "current_strategy_ref")
    _exact_fields(
        obj,
        _CURRENT_REQUEST_FIELDS,
        "current_strategy_ref",
    )
    return {
        "strategy_id": _text(
            obj["strategy_id"],
            "current_strategy_ref.strategy_id",
        ),
        "expected_strategy_spec_hash": _hash(
            obj["expected_strategy_spec_hash"],
            "current_strategy_ref.expected_strategy_spec_hash",
        ),
    }


def _validate_economics_inputs(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _json_object(value, "economics_inputs")
    if len(obj) > _MAX_ECONOMIC_COMPONENTS:
        raise StrategyError("economics_inputs component budget exceeded")
    result: dict[str, Any] = {}
    for raw_name, raw_binding in obj.items():
        name = _text(raw_name, "economics_inputs component")
        binding = _json_object(
            raw_binding,
            f"economics_inputs.{name}",
        )
        kind = _text(
            binding.get("kind"),
            f"economics_inputs.{name}.kind",
        )
        if kind == "column":
            _exact_fields(
                binding,
                frozenset({"kind", "column"}),
                f"economics_inputs.{name}",
            )
            result[name] = {
                "kind": "column",
                "column": _text(
                    binding["column"],
                    f"economics_inputs.{name}.column",
                ),
            }
        elif kind == "scalar":
            _exact_fields(
                binding,
                frozenset({"kind", "value"}),
                f"economics_inputs.{name}",
            )
            result[name] = {
                "kind": "scalar",
                "value": _finite_number(
                    binding["value"],
                    f"economics_inputs.{name}.value",
                ),
            }
        else:
            raise StrategyError(
                f"economics_inputs.{name}.kind must be column or scalar"
            )
    return {name: result[name] for name in sorted(result)}


def _load_pool_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> StrategyCandidatePoolArtifactBinding:
    ref = request["pool_ref"]
    binding = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        strategy_type=request["strategy_type"],
        expected_pool_revision=ref["expected_revision"],
        expected_pool_snapshot_hash=ref["expected_snapshot_hash"],
        expected_artifact_id=ref["artifact_id"],
        expected_artifact_content_hash=ref[
            "expected_artifact_content_hash"
        ],
    )
    if (
        binding.pool["pool_id"] != ref["expected_pool_id"]
        or binding.pool["revision_id"] != ref["expected_revision_id"]
    ):
        raise StrategyError("current Strategy Pool identity changed")
    if not binding.pool["entries"]:
        raise StrategyError("cannot measure an empty Strategy Pool")
    return binding


def _load_sample_design_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> StrategySampleDesignV2ArtifactBinding:
    ref = request["sample_design_ref"]
    return load_strategy_sample_design_v2_artifacts(
        runtime,
        task_id=task_id,
        membership_artifact_id=ref["membership_artifact_id"],
        expected_membership_artifact_content_hash=ref[
            "expected_membership_artifact_content_hash"
        ],
        bundle_artifact_id=ref["bundle_artifact_id"],
        expected_bundle_artifact_content_hash=ref[
            "expected_bundle_artifact_content_hash"
        ],
        expected_bundle_id=ref["expected_bundle_id"],
        expected_sample_design_id=ref["expected_sample_design_id"],
        expected_sample_design_content_hash=ref[
            "expected_sample_design_content_hash"
        ],
    )


def _require_sample_contract(
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    development: StrategyPoolDevelopmentExecutionBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    partitions: Sequence[str],
) -> dict[str, Any]:
    design = sample.bundle["sample_design"]
    target = design["target_selector"]
    if target["status"] != "resolved":
        raise StrategyError(
            "ImpactCube requires a resolved V2 target selector"
        )
    if design["sample_semantics"]["scope"] != "strategy_development":
        raise StrategyError("ImpactCube requires governed strategy scope")
    risk = next(
        item
        for item in sample.bundle["populations"]
        if item["role"] == "risk"
    )
    if risk["maturity_evidence"]["status"] != "confirmed_matured":
        raise StrategyError(
            "ImpactCube requires confirmed_matured risk outcomes"
        )
    counts = sample.membership["header"]["counts"]
    for role in ("approval", "risk"):
        for partition in partitions:
            if counts[role][partition] == 0:
                raise StrategyError(
                    f"ImpactCube {role}/{partition} partition is empty"
                )

    legacy_ref = StrategySampleDesignRef.from_value(
        design["compatibility"]["legacy_development_ref"]
    )
    if legacy_ref != sample.source_binding.legacy.reference:
        raise StrategyError(
            "StrategySampleDesign V2 legacy development mapping changed"
        )
    _require_pool_development_contract(
        pool=pool,
        development=development,
        sample=sample,
        legacy_ref=legacy_ref,
        target_col=target["column"],
    )
    if (
        sample.source_binding.legacy.target_col != target["column"]
        or sample.source_binding.legacy.target_bad_value
        != target["bad_value"]
    ):
        raise StrategyError(
            "StrategySampleDesign V2 target polarity changed from legacy lineage"
        )
    fields = design["sample_semantics"]["field_bindings"]
    return {
        "legacy_development_ref": legacy_ref.to_ref_dict(),
        "target_col": target["column"],
        "target_bad_value": target["bad_value"],
        "entity_col": fields["entity_field"],
        "loan_amount_col": fields["loan_amount_field"],
        "overdue_amount_col": fields["overdue_amount_field"],
    }


def _require_pool_development_contract(
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    development: StrategyPoolDevelopmentExecutionBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    legacy_ref: StrategySampleDesignRef,
    target_col: str,
) -> None:
    if (
        development.pool is not pool
        or development.task_id != pool.task_id
        or development.sample_design.reference != legacy_ref
        or development.target_col != target_col
    ):
        raise StrategyError(
            "Strategy Pool development binding does not match "
            "StrategySampleDesign V2"
        )
    dataset = development.dataset
    source = sample.source_binding
    if (
        dataset.task_id != source.task_id
        or dataset.dataset_id != source.dataset_id
        or dataset.source_path != source.dataset_source_path
        or dataset.path != source.dataset_path
        or not hmac.compare_digest(
            dataset.content_hash,
            source.dataset_content_hash,
        )
        or dataset.columns != source.columns
        or dataset.row_count != source.row_count
    ):
        raise StrategyError(
            "Strategy Pool development dataset does not match "
            "StrategySampleDesign V2"
        )
    bound_v2 = development.sample_design_v2
    if bound_v2 is not None and _sample_design_v2_identity(
        bound_v2
    ) != _sample_design_v2_identity(sample):
        raise StrategyError(
            "Strategy Pool development SampleDesign V2 binding changed"
        )


def _sample_design_v2_identity(
    binding: StrategySampleDesignV2ArtifactBinding,
) -> tuple[str, ...]:
    return (
        binding.task_id,
        binding.membership_artifact_id,
        binding.membership_artifact_content_hash,
        binding.bundle_artifact_id,
        binding.bundle_artifact_content_hash,
        str(binding.bundle["bundle_id"]),
        str(binding.bundle["sample_design"]["sample_design_id"]),
        str(binding.bundle["sample_design"]["content_hash"]),
    )


def _load_current_strategy(
    runtime,
    *,
    task_id: str,
    strategy_type: str,
    value: Mapping[str, str] | None,
) -> _CurrentStrategyBinding | None:
    if value is None:
        return None
    strategy_id = value["strategy_id"]
    snapshot = runtime.strategies.get_strategy_snapshot(strategy_id)
    strategy = None if snapshot is None else snapshot["strategy"]
    meta = None if snapshot is None else snapshot["metadata"]
    stored_hash = (
        None if snapshot is None else snapshot["strategy_spec_hash"]
    )
    if (
        strategy is None
        or meta is None
        or strategy.spec is None
        or not isinstance(stored_hash, str)
        or meta.get("task_id") != task_id
    ):
        raise StrategyError(
            "current strategy is not owned by the current task"
        )
    if (
        strategy.strategy_type != strategy_type
        or meta.get("strategy_type") != strategy_type
    ):
        raise StrategyError("current strategy type must match the Pool")
    calculated = strategy_spec_hash(strategy.spec)
    if (
        not hmac.compare_digest(stored_hash, calculated)
        or not hmac.compare_digest(
            stored_hash,
            value["expected_strategy_spec_hash"],
        )
    ):
        raise StrategyError("current strategy spec hash changed")
    return _CurrentStrategyBinding(
        strategy_id=strategy.id,
        strategy_type=strategy.strategy_type,
        spec=strategy.spec,
        spec_hash=stored_hash,
    )


def _require_bindings_under_lock(
    runtime,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    current: _CurrentStrategyBinding | None,
    task_id: str,
    resolved_requirements: ResolvedPoolRequirements,
) -> None:
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_bindings_on_connection(
            conn,
            pool=pool,
            sample=sample,
            current=current,
            task_id=task_id,
            resolved_requirements=resolved_requirements,
        )
        _require_dataset_unchanged(sample)
        conn.commit()


def _require_bindings_on_connection(
    conn,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    current: _CurrentStrategyBinding | None,
    task_id: str,
    resolved_requirements: ResolvedPoolRequirements,
) -> None:
    require_strategy_candidate_pool_artifact_binding_on_connection(conn, pool)
    require_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        sample,
    )
    _require_current_on_connection(
        conn,
        current=current,
        task_id=task_id,
    )
    require_resolved_pool_requirements_on_connection(
        conn,
        resolved_requirements,
    )


def _require_current_on_connection(
    conn,
    *,
    current: _CurrentStrategyBinding | None,
    task_id: str,
) -> None:
    if current is None:
        return
    row = conn.execute(
        """
        SELECT id, task_id, strategy_type, rules_json, score_col,
               default_decision_json, description, created_at,
               dsl_json, dsl_schema_version, dsl_content_hash
          FROM strategies
         WHERE id = ?
        """,
        (current.strategy_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError(
            "current strategy changed before ImpactCube publication"
        )
    strategy = _strategy_from_row(row)
    stored_hash = _strategy_spec_hash_from_row(row)
    if (
        strategy.strategy_type != current.strategy_type
        or strategy.spec is None
        or strategy.spec.to_dict() != current.spec.to_dict()
        or not hmac.compare_digest(stored_hash, current.spec_hash)
    ):
        raise StrategyError(
            "current strategy changed before ImpactCube publication"
        )


def _read_partition_frames(
    runtime,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    current: _CurrentStrategyBinding | None,
    request: Mapping[str, Any],
    target_col: str,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
    resolved_requirements: ResolvedPoolRequirements,
) -> dict[str, dict[str, pd.DataFrame]]:
    path = sample.source_binding.dataset_path
    _require_dataset_path(
        path,
        root=Path(runtime.settings.datasets_dir).absolute(),
    )
    fields = _expression_fields(
        pool.compiled_design["strategy_spec"]
    )
    if current is not None:
        fields.update(_expression_fields(current.spec.to_dict()))
    fields.add(target_col)
    fields.update(
        column
        for column in (loan_amount_col, overdue_amount_col)
        if column is not None
    )
    fields.update(
        field
        for field in request["dimension_bindings"].values()
        if field is not None
    )
    economics = request["economics_inputs"] or {}
    fields.update(
        binding["column"]
        for binding in economics.values()
        if binding["kind"] == "column"
    )
    virtual_fields = set(resolved_requirements.virtual_fields)
    physical_fields = fields - virtual_fields
    unknown = sorted(physical_fields - set(sample.source_binding.columns))
    if unknown:
        raise StrategyError(
            "ImpactCube dataset is missing columns: " + ", ".join(unknown)
        )
    frame = _read_authenticated_parquet_snapshot(
        path,
        root=Path(runtime.settings.datasets_dir).absolute(),
        expected_content_hash=sample.source_binding.dataset_content_hash,
        columns=sorted(physical_fields),
    )
    if not isinstance(frame, pd.DataFrame) or len(frame) != (
        sample.source_binding.row_count
    ):
        raise StrategyError(
            "ImpactCube analysis universe row count changed"
        )
    # Model score vectors are bound to raw row ordinals, not a user-provided
    # persisted Parquet index. Normalize only after the full universe has been
    # authenticated and before any partition mask is applied.
    frame = frame.reset_index(drop=True)
    frame = hydrate_requirement_fields(
        frame,
        resolved=resolved_requirements,
    )

    masks: dict[str, dict[str, np.ndarray]] = {
        "approval": {},
        "risk": {},
    }
    counts = sample.membership["header"]["counts"]
    for role in ("approval", "risk"):
        for partition in _PARTITION_ORDER:
            key = f"{role}/{partition}"
            if key not in sample.membership["masks"]:
                raise StrategyError(
                    f"ImpactCube membership is missing {key}"
                )
            mask = np.asarray(
                sample.membership["masks"][key],
                dtype=np.bool_,
            )
            if len(mask) != len(frame):
                raise StrategyError(
                    "StrategySampleDesign V2 membership row order changed"
                )
            if int(np.count_nonzero(mask)) != counts[role][partition]:
                raise StrategyError(
                    f"ImpactCube {role}/{partition} membership count changed"
                )
            masks[role][partition] = mask
        for index, left in enumerate(_PARTITION_ORDER):
            for right in _PARTITION_ORDER[index + 1 :]:
                if bool(np.any(masks[role][left] & masks[role][right])):
                    raise StrategyError(
                        f"ImpactCube {role} partitions overlap"
                    )
    for partition in _PARTITION_ORDER:
        if bool(
            np.any(
                masks["risk"][partition]
                & ~masks["approval"][partition]
            )
        ):
            raise StrategyError(
                f"ImpactCube risk/{partition} is outside approval population"
            )

    result: dict[str, dict[str, pd.DataFrame]] = {
        "approval": {},
        "risk": {},
    }
    for role in ("approval", "risk"):
        for partition in request["partitions"]:
            selected = frame.loc[
                pd.Series(
                    masks[role][partition],
                    index=frame.index,
                    dtype=bool,
                )
            ].reset_index(drop=True)
            if selected.empty or len(selected) != counts[role][partition]:
                raise StrategyError(
                    f"ImpactCube {role}/{partition} partition is empty "
                    "or changed"
                )
            result[role][partition] = selected
    _require_dataset_unchanged(sample)
    return result


def _read_authenticated_parquet_snapshot(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
    columns: list[str],
) -> pd.DataFrame:
    """Read only bytes copied from one authenticated, retained source fd."""

    _require_dataset_path(path, root=root)
    source_fd = -1
    snapshot = None
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise StrategyError(
                "ImpactCube dataset must be a regular file"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_fd = os.open(path, flags)
        opened = os.fstat(source_fd)
        after_open = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(after_open.st_mode)
            or _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(after_open)
            or _stable_file_stat(before) != _stable_file_stat(opened)
            or _stable_file_stat(opened) != _stable_file_stat(after_open)
        ):
            raise StrategyError("ImpactCube dataset changed while opening")

        snapshot = tempfile.TemporaryFile(mode="w+b", dir=root)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            snapshot.write(chunk)
        snapshot.flush()
        source_after_copy = os.fstat(source_fd)
        if (
            _stable_file_stat(source_after_copy)
            != _stable_file_stat(opened)
            or copied != int(opened.st_size)
            or not hmac.compare_digest(
                digest.hexdigest(),
                expected_content_hash,
            )
        ):
            raise StrategyError(
                "ImpactCube dataset bytes changed before replay"
            )

        snapshot_stat = os.fstat(snapshot.fileno())
        if int(snapshot_stat.st_size) != copied:
            raise StrategyError(
                "ImpactCube private dataset snapshot is incomplete"
            )
        snapshot.seek(0)
        frame = pd.read_parquet(snapshot, columns=columns)
        snapshot_after_read = os.fstat(snapshot.fileno())
        current = os.lstat(path)
        if (
            _stable_file_stat(snapshot_after_read)
            != _stable_file_stat(snapshot_stat)
            or _stable_file_stat(os.fstat(source_fd))
            != _stable_file_stat(opened)
            or stat.S_ISLNK(current.st_mode)
            or _stable_file_stat(current) != _stable_file_stat(opened)
        ):
            raise StrategyError("ImpactCube dataset changed during replay")
        return frame
    except StrategyError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StrategyError("ImpactCube dataset could not be read") from exc
    finally:
        if snapshot is not None:
            snapshot.close()
        if source_fd >= 0:
            os.close(source_fd)


def _file_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
    )


def _stable_file_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _expression_fields(value: object) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, Mapping):
        field = value.get("field")
        if isinstance(field, str):
            fields.add(field)
        for item in value.values():
            fields.update(_expression_fields(item))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        for item in value:
            fields.update(_expression_fields(item))
    return fields


def _sample_design_evidence_ref(
    sample: StrategySampleDesignV2ArtifactBinding,
    *,
    partitions: Sequence[str],
) -> dict[str, Any]:
    header = sample.membership["header"]
    bundle = sample.bundle
    design = bundle["sample_design"]
    return {
        "membership_artifact_id": sample.membership_artifact_id,
        "membership_artifact_content_hash": (
            sample.membership_artifact_content_hash
        ),
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "bundle_artifact_id": sample.bundle_artifact_id,
        "bundle_artifact_content_hash": (
            sample.bundle_artifact_content_hash
        ),
        "bundle_id": bundle["bundle_id"],
        "bundle_content_hash": bundle["content_hash"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "analysis_universe_row_count": header["row_count"],
        "partition_counts": {
            partition: header["counts"]["risk"][partition]
            for partition in partitions
        },
        "population_partition_counts": {
            role: {
                partition: header["counts"][role][partition]
                for partition in partitions
            }
            for role in ("approval", "risk")
        },
    }


def _dataset_evidence_binding(
    sample: StrategySampleDesignV2ArtifactBinding,
) -> dict[str, Any]:
    source = sample.source_binding
    return {
        "task_id": source.task_id,
        "dataset_id": source.dataset_id,
        "dataset_content_hash": source.dataset_content_hash,
        "dataset_source_path": source.dataset_source_path,
        "dataset_registry_metadata_hash": (
            source.dataset_registry_metadata_hash
        ),
        "workspace_revision": source.workspace_revision,
        "workspace_generation": source.workspace_generation,
        "semantic_mapping_hash": source.semantic_mapping_hash,
    }


def _require_dataset_unchanged(
    sample: StrategySampleDesignV2ArtifactBinding,
) -> None:
    path = sample.source_binding.dataset_path
    descriptor = -1
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise StrategyError(
                "ImpactCube dataset changed during computation"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        after_open = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(after_open.st_mode)
            or _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(after_open)
            or _stable_file_stat(before) != _stable_file_stat(opened)
            or _stable_file_stat(opened) != _stable_file_stat(after_open)
        ):
            raise StrategyError(
                "ImpactCube dataset changed during computation"
            )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            digest.update(chunk)
        after_read = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            copied != int(opened.st_size)
            or _stable_file_stat(after_read) != _stable_file_stat(opened)
            or stat.S_ISLNK(current.st_mode)
            or _stable_file_stat(current) != _stable_file_stat(opened)
        ):
            raise StrategyError(
                "ImpactCube dataset changed during computation"
            )
    except OSError as exc:
        raise StrategyError(
            "ImpactCube dataset changed during computation"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not hmac.compare_digest(
        digest.hexdigest(),
        sample.source_binding.dataset_content_hash,
    ):
        raise StrategyError("ImpactCube dataset changed during computation")


def build_impact_cube_producer_run(
    *,
    task_id: str,
    request: Mapping[str, Any],
    cube_id: str,
    cube_content_hash: str,
    artifact_id: str,
    artifact_filename: str,
    artifact_content_hash: str,
) -> dict[str, Any]:
    """Build one stable, content-addressed deterministic measurement run."""

    normalized_task_id = _text(task_id, "producer_run.task_id")
    normalized_request = _validate_inputs(request)
    tool_ref = {
        "plugin": "strategy",
        "tool": "measure_strategy_impact_cube",
        "origin_tool": IMPACT_CUBE_ORIGIN_TOOL,
        "tool_schema_version": IMPACT_CUBE_TOOL_SCHEMA_VERSION,
        "producer_version": STRATEGY_IMPACT_CUBE_PRODUCER_VERSION,
    }
    input_binding = {
        "task_id": normalized_task_id,
        "request": normalized_request,
        "producer": tool_ref,
    }
    input_hash = hashlib.sha256(
        _canonical_json(input_binding).encode("utf-8")
    ).hexdigest()
    body = {
        "schema_version": IMPACT_CUBE_MEASUREMENT_RUN_SCHEMA_VERSION,
        "run_id": f"strategy-impact-cube-run-{input_hash[:24]}",
        "input_hash": input_hash,
        "task_id": normalized_task_id,
        "request": normalized_request,
        "tool_ref": tool_ref,
        "cube_ref": {
            "cube_id": _text(cube_id, "producer_run.cube_ref.cube_id"),
            "content_hash": _hash(
                cube_content_hash,
                "producer_run.cube_ref.content_hash",
            ),
        },
        "artifact_ref": {
            "artifact_id": _hash(
                artifact_id,
                "producer_run.artifact_ref.artifact_id",
            ),
            "kind": IMPACT_CUBE_ARTIFACT_KIND,
            "filename": _text(
                artifact_filename,
                "producer_run.artifact_ref.filename",
            ),
            "content_hash": _hash(
                artifact_content_hash,
                "producer_run.artifact_ref.content_hash",
            ),
            "origin_tool": IMPACT_CUBE_ORIGIN_TOOL,
        },
    }
    content_hash = hashlib.sha256(
        _canonical_json(body).encode("utf-8")
    ).hexdigest()
    return validate_impact_cube_producer_run(
        {**body, "content_hash": content_hash},
        expected_task_id=normalized_task_id,
        expected_request=normalized_request,
        expected_cube_id=cube_id,
        expected_cube_content_hash=cube_content_hash,
        expected_artifact_id=artifact_id,
        expected_artifact_filename=artifact_filename,
        expected_artifact_content_hash=artifact_content_hash,
    )


def validate_impact_cube_producer_run(
    value: object,
    *,
    expected_task_id: str,
    expected_request: Mapping[str, Any],
    expected_cube_id: str,
    expected_cube_content_hash: str,
    expected_artifact_id: str,
    expected_artifact_filename: str,
    expected_artifact_content_hash: str,
) -> dict[str, Any]:
    """Authenticate a measurement run and all of its evidence bindings."""

    obj = _json_object(value, "ImpactCube producer_run")
    _exact_fields(obj, _PRODUCER_RUN_FIELDS, "ImpactCube producer_run")
    if obj["schema_version"] != IMPACT_CUBE_MEASUREMENT_RUN_SCHEMA_VERSION:
        raise StrategyError("ImpactCube producer_run schema_version is invalid")
    task_id = _text(expected_task_id, "expected producer_run task_id")
    request = _validate_inputs(expected_request)
    if obj["task_id"] != task_id or obj["request"] != request:
        raise StrategyError("ImpactCube producer_run input binding changed")
    tool_ref = _json_object(
        obj["tool_ref"],
        "ImpactCube producer_run.tool_ref",
    )
    _exact_fields(
        tool_ref,
        _PRODUCER_TOOL_REF_FIELDS,
        "ImpactCube producer_run.tool_ref",
    )
    expected_tool_ref = {
        "plugin": "strategy",
        "tool": "measure_strategy_impact_cube",
        "origin_tool": IMPACT_CUBE_ORIGIN_TOOL,
        "tool_schema_version": IMPACT_CUBE_TOOL_SCHEMA_VERSION,
        "producer_version": STRATEGY_IMPACT_CUBE_PRODUCER_VERSION,
    }
    if tool_ref != expected_tool_ref:
        raise StrategyError("ImpactCube producer_run tool_ref changed")
    expected_input_hash = hashlib.sha256(
        _canonical_json(
            {
                "task_id": task_id,
                "request": request,
                "producer": expected_tool_ref,
            }
        ).encode("utf-8")
    ).hexdigest()
    input_hash = _hash(
        obj["input_hash"],
        "ImpactCube producer_run.input_hash",
    )
    if not hmac.compare_digest(input_hash, expected_input_hash):
        raise StrategyError("ImpactCube producer_run input_hash changed")
    run_id = _text(obj["run_id"], "ImpactCube producer_run.run_id")
    if (
        _MEASUREMENT_RUN_ID_RE.fullmatch(run_id) is None
        or run_id != f"strategy-impact-cube-run-{input_hash[:24]}"
    ):
        raise StrategyError("ImpactCube producer_run run_id changed")

    cube_ref = _json_object(
        obj["cube_ref"],
        "ImpactCube producer_run.cube_ref",
    )
    _exact_fields(
        cube_ref,
        _PRODUCER_CUBE_REF_FIELDS,
        "ImpactCube producer_run.cube_ref",
    )
    expected_cube_ref = {
        "cube_id": _text(
            expected_cube_id,
            "expected producer_run cube_id",
        ),
        "content_hash": _hash(
            expected_cube_content_hash,
            "expected producer_run cube content_hash",
        ),
    }
    if cube_ref != expected_cube_ref:
        raise StrategyError("ImpactCube producer_run cube binding changed")

    artifact_ref = _json_object(
        obj["artifact_ref"],
        "ImpactCube producer_run.artifact_ref",
    )
    _exact_fields(
        artifact_ref,
        _PRODUCER_ARTIFACT_REF_FIELDS,
        "ImpactCube producer_run.artifact_ref",
    )
    expected_artifact_ref = {
        "artifact_id": _hash(
            expected_artifact_id,
            "expected producer_run artifact_id",
        ),
        "kind": IMPACT_CUBE_ARTIFACT_KIND,
        "filename": _text(
            expected_artifact_filename,
            "expected producer_run artifact filename",
        ),
        "content_hash": _hash(
            expected_artifact_content_hash,
            "expected producer_run artifact content_hash",
        ),
        "origin_tool": IMPACT_CUBE_ORIGIN_TOOL,
    }
    if artifact_ref != expected_artifact_ref:
        raise StrategyError("ImpactCube producer_run artifact binding changed")

    body = {key: obj[key] for key in obj if key != "content_hash"}
    expected_content_hash = hashlib.sha256(
        _canonical_json(body).encode("utf-8")
    ).hexdigest()
    content_hash = _hash(
        obj["content_hash"],
        "ImpactCube producer_run.content_hash",
    )
    if not hmac.compare_digest(content_hash, expected_content_hash):
        raise StrategyError("ImpactCube producer_run self hash changed")
    return obj


def impact_cube_producer_run_ref(
    producer_run: Mapping[str, Any],
) -> dict[str, str]:
    """Project an authenticated producer run to the public source-ref shape."""

    obj = _json_object(producer_run, "ImpactCube producer_run")
    run_id = _text(obj.get("run_id"), "ImpactCube producer_run.run_id")
    if _MEASUREMENT_RUN_ID_RE.fullmatch(run_id) is None:
        raise StrategyError("ImpactCube producer_run run_id is invalid")
    return {
        "kind": "tool_run",
        "ref_id": run_id,
        "content_hash": _hash(
            obj.get("content_hash"),
            "ImpactCube producer_run.content_hash",
        ),
    }


def _persist_cube(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    current: _CurrentStrategyBinding | None,
    resolved_requirements: ResolvedPoolRequirements,
    cube: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = canonical_strategy_impact_cube_json(cube).encode("utf-8")
    if len(canonical) > MAX_IMPACT_CUBE_JSON_BYTES:
        raise StrategyError("ImpactCube artifact exceeds byte budget")
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = out_dir / f"{cube['cube_id']}.json"
    artifact_id = _stable_artifact_id(
        task_id=task_id,
        kind=IMPACT_CUBE_ARTIFACT_KIND,
        path=str(final_path),
    )
    producer_run = build_impact_cube_producer_run(
        task_id=task_id,
        request=request,
        cube_id=str(cube["cube_id"]),
        cube_content_hash=str(cube["content_hash"]),
        artifact_id=artifact_id,
        artifact_filename=final_path.name,
        artifact_content_hash=artifact_hash,
    )
    sources = cube["source_bindings"]
    provenance = {
        "schema_version": (
            IMPACT_CUBE_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION
            if resolved_requirements.requirements
            else IMPACT_CUBE_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": STRATEGY_IMPACT_CUBE_PRODUCER_VERSION,
        "task_id": task_id,
        "cube_id": cube["cube_id"],
        "cube_content_hash": cube["content_hash"],
        "pool_ref": dict(request["pool_ref"]),
        "sample_design_ref": dict(request["sample_design_ref"]),
        "dataset_binding": dict(sources["dataset"]),
        "target_binding": dict(sources["target"]),
        "dimension_bindings": dict(request["dimension_bindings"]),
        "current_strategy_ref": (
            None
            if current is None
            else {
                "strategy_id": current.strategy_id,
                "strategy_type": current.strategy_type,
                "strategy_spec_hash": current.spec_hash,
            }
        ),
        "economics_inputs": (
            None
            if request["economics_inputs"] is None
            else dict(request["economics_inputs"])
        ),
        "partitions": list(request["partitions"]),
        "populations": ["approval", "risk"],
        "lifecycle": dict(_LIFECYCLE),
        "producer_run": producer_run,
    }
    if resolved_requirements.requirements:
        provenance["requirement_bindings"] = (
            pool_requirement_bindings_provenance(resolved_requirements)
        )
    _validate_provenance(provenance)
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError(
            "ImpactCube artifact could not be staged"
        ) from exc

    db_committed = False
    rollback_under_lock = False
    reused = False
    created_artifact_row: dict[str, Any] | None = None
    created_audit_row: dict[str, Any] | None = None
    retained_fd = -1
    retained_identity: tuple[int, ...] | None = None
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
                    current=current,
                    task_id=task_id,
                    resolved_requirements=resolved_requirements,
                )
                _require_dataset_unchanged(sample)
                row = _select_artifact_row(
                    conn,
                    task_id=task_id,
                    path=final_path,
                )
                artifact_row_existed = row is not None
                if row is not None:
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        path=final_path,
                        canonical=canonical,
                        content_hash=artifact_hash,
                        provenance=provenance,
                        root=Path(runtime.settings.tasks_dir).absolute(),
                    )
                    uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        _require_exact_file(
                            final_path,
                            root=Path(
                                runtime.settings.tasks_dir
                            ).absolute(),
                            canonical=canonical,
                            content_hash=artifact_hash,
                        )
                        uow.rollback()
                        reused = True
                    else:
                        uow.promote_all()
                        _require_exact_file(
                            final_path,
                            root=Path(
                                runtime.settings.tasks_dir
                            ).absolute(),
                            canonical=canonical,
                            content_hash=artifact_hash,
                        )
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
                    current=current,
                    task_id=task_id,
                    resolved_requirements=resolved_requirements,
                )
                _require_dataset_unchanged(sample)
                retained_fd, retained_identity = _open_retained_exact_file(
                    final_path,
                    root=Path(runtime.settings.tasks_dir).absolute(),
                    canonical=canonical,
                    content_hash=artifact_hash,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=IMPACT_CUBE_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=IMPACT_CUBE_ORIGIN_TOOL,
                    provenance=provenance,
                )
                if str(record["id"]) != artifact_id:
                    raise StrategyError(
                        "ImpactCube artifact stable identity changed"
                    )
                if not artifact_row_existed:
                    inserted_artifact_row = _select_artifact_row(
                        conn,
                        task_id=task_id,
                        path=final_path,
                    )
                    if inserted_artifact_row is None:
                        raise StrategyError(
                            "ImpactCube artifact registration disappeared"
                        )
                    created_artifact_row = {
                        field: inserted_artifact_row[field]
                        for field in _TASK_ARTIFACT_ROW_FIELDS
                    }
                created_audit_row = _write_or_require_measurement_audit(
                    conn,
                    runtime=runtime,
                    producer_run=producer_run,
                    artifact_row_existed=artifact_row_existed,
                )
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
                    current=current,
                    task_id=task_id,
                    resolved_requirements=resolved_requirements,
                )
                _require_dataset_unchanged(sample)
                _require_retained_exact_file(
                    retained_fd,
                    retained_identity=retained_identity,
                    path=final_path,
                    canonical=canonical,
                    content_hash=artifact_hash,
                )
                conn.commit()
                db_committed = True
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    _require_retained_exact_file(
                        retained_fd,
                        retained_identity=retained_identity,
                        path=final_path,
                        canonical=canonical,
                        content_hash=artifact_hash,
                    )
                    require_impact_cube_measurement_audit_on_connection(
                        conn,
                        producer_run,
                    )
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    if (
                        created_artifact_row is not None
                        or created_audit_row is not None
                    ):
                        _remove_committed_artifact_record(
                            conn,
                            created_artifact_row=created_artifact_row,
                            created_audit_row=created_audit_row,
                        )
                        disposition = (
                            "newly created registry and audit entries "
                            "were removed"
                        )
                    else:
                        disposition = (
                            "pre-existing registry and audit entries "
                            "were retained"
                        )
                    raise StrategyError(
                        "ImpactCube artifact or measurement audit changed "
                        f"after registration commit; {disposition}"
                    ) from exc
            except Exception:
                rollback_under_lock = True
                uow.rollback()
                raise
        if not reused:
            uow.commit()
    except Exception:
        if not db_committed and not rollback_under_lock:
            uow.rollback()
        raise
    finally:
        if retained_fd >= 0:
            os.close(retained_fd)
    return validate_measure_strategy_impact_cube_tool_output(
        _tool_output(
            cube=cube,
            record=record,
            task_id=task_id,
            producer_run=producer_run,
        ),
        trusted_task_id=task_id,
        trusted_artifact_id=str(record["id"]),
        trusted_artifact_content_hash=str(record["content_hash"]),
        trusted_producer_run_id=str(producer_run["run_id"]),
        trusted_producer_run_content_hash=str(
            producer_run["content_hash"]
        ),
    )


def _tool_output(
    *,
    cube: Mapping[str, Any],
    record: Mapping[str, Any],
    task_id: str,
    producer_run: Mapping[str, Any],
) -> dict[str, Any]:
    identity = cube["identity"]
    artifact_id = str(record["id"])
    return {
        "schema_version": IMPACT_CUBE_TOOL_SCHEMA_VERSION,
        "cube_id": cube["cube_id"],
        "content_hash": cube["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "strategy_type": identity["strategy_type"],
        "partitions": [
            row["name"]
            for row in cube["partitions"]
            if row["role"] == "risk"
        ],
        "slice_count": len(cube["slices"]),
        "cube": dict(cube),
        "warnings": [
            str(flag["message"])
            for flag in cube["red_flags"]
            if flag["level"] in {"amber", "red"}
        ],
        "artifact": {
            "artifact_id": artifact_id,
            "kind": IMPACT_CUBE_ARTIFACT_KIND,
            "format": "json",
            "filename": Path(str(record["path"])).name,
            "content_hash": str(record["content_hash"]),
            "download_url": (
                f"/api/tasks/{quote(task_id, safe='')}"
                f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                f"?expected_content_hash={quote(str(record['content_hash']), safe='')}"
            ),
        },
        "producer_run_ref": impact_cube_producer_run_ref(producer_run),
        "not_mutated_pool": True,
        "not_created_strategy": True,
        "not_adopted": True,
        "not_promoted": True,
        "not_deployed": True,
    }


def _validate_provenance(value: object) -> dict[str, Any]:
    obj = _json_object(value, "ImpactCube provenance")
    schema_version = obj.get("schema_version")
    if schema_version == IMPACT_CUBE_ARTIFACT_SCHEMA_VERSION:
        _exact_fields(obj, _PROVENANCE_FIELDS, "ImpactCube provenance")
    elif schema_version == IMPACT_CUBE_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION:
        _exact_fields(
            obj,
            _REQUIREMENTS_PROVENANCE_FIELDS,
            "ImpactCube provenance",
        )
        validate_pool_requirement_bindings_provenance(
            obj["requirement_bindings"]
        )
    else:
        raise StrategyError("ImpactCube provenance schema_version is invalid")
    if obj["producer_version"] != STRATEGY_IMPACT_CUBE_PRODUCER_VERSION:
        raise StrategyError("ImpactCube provenance producer_version is invalid")
    for field in ("task_id", "cube_id"):
        _text(obj[field], f"ImpactCube provenance.{field}")
    _hash(
        obj["cube_content_hash"],
        "ImpactCube provenance.cube_content_hash",
    )
    _validate_pool_ref(obj["pool_ref"])
    _validate_sample_design_ref(obj["sample_design_ref"])
    dataset = _json_object(
        obj["dataset_binding"],
        "ImpactCube dataset provenance",
    )
    _exact_fields(
        dataset,
        _DATASET_BINDING_FIELDS,
        "ImpactCube dataset provenance",
    )
    for field in ("task_id", "dataset_id", "dataset_source_path"):
        _text(dataset[field], f"ImpactCube dataset provenance.{field}")
    for field in (
        "dataset_content_hash",
        "dataset_registry_metadata_hash",
        "semantic_mapping_hash",
    ):
        _hash(dataset[field], f"ImpactCube dataset provenance.{field}")
    for field in ("workspace_revision", "workspace_generation"):
        _nonnegative_int(
            dataset[field],
            f"ImpactCube dataset provenance.{field}",
        )
    if dataset["task_id"] != obj["task_id"]:
        raise StrategyError("ImpactCube provenance task binding changed")
    target = _json_object(
        obj["target_binding"],
        "ImpactCube target provenance",
    )
    _exact_fields(
        target,
        _TARGET_BINDING_FIELDS,
        "ImpactCube target provenance",
    )
    _text(target["column"], "ImpactCube target provenance.column")
    bad_value = _binary_int(
        target["bad_value"],
        "ImpactCube target provenance.bad_value",
    )
    if (
        _binary_int(
            target["good_value"],
            "ImpactCube target provenance.good_value",
        )
        != 1 - bad_value
        or target["missing_policy"]
        != "retain_population_exclude_risk_denominator"
    ):
        raise StrategyError("ImpactCube target provenance changed")
    _validate_dimension_bindings(obj["dimension_bindings"])
    current = obj["current_strategy_ref"]
    if current is not None:
        current_obj = _json_object(
            current,
            "ImpactCube current strategy provenance",
        )
        _exact_fields(
            current_obj,
            frozenset(
                {
                    "strategy_id",
                    "strategy_type",
                    "strategy_spec_hash",
                }
            ),
            "ImpactCube current strategy provenance",
        )
        _text(current_obj["strategy_id"], "current strategy id")
        if current_obj["strategy_type"] not in _STRATEGY_TYPES:
            raise StrategyError(
                "ImpactCube current strategy type is invalid"
            )
        _hash(
            current_obj["strategy_spec_hash"],
            "current strategy spec hash",
        )
    _validate_economics_inputs(obj["economics_inputs"])
    _partition_list(obj["partitions"])
    if obj["populations"] != ["approval", "risk"]:
        raise StrategyError("ImpactCube provenance populations changed")
    if obj["lifecycle"] != _LIFECYCLE:
        raise StrategyError("ImpactCube provenance lifecycle changed")
    run_obj = _json_object(
        obj["producer_run"],
        "ImpactCube provenance.producer_run",
    )
    run = validate_impact_cube_producer_run(
        run_obj,
        expected_task_id=obj["task_id"],
        expected_request=run_obj.get("request"),
        expected_cube_id=obj["cube_id"],
        expected_cube_content_hash=obj["cube_content_hash"],
        expected_artifact_id=_json_object(
            run_obj.get("artifact_ref"),
            "ImpactCube producer_run.artifact_ref",
        ).get("artifact_id"),
        expected_artifact_filename=_json_object(
            run_obj.get("artifact_ref"),
            "ImpactCube producer_run.artifact_ref",
        ).get("filename"),
        expected_artifact_content_hash=_json_object(
            run_obj.get("artifact_ref"),
            "ImpactCube producer_run.artifact_ref",
        ).get("content_hash"),
    )
    request = run["request"]
    if (
        request["pool_ref"] != obj["pool_ref"]
        or request["sample_design_ref"] != obj["sample_design_ref"]
        or request["partitions"] != obj["partitions"]
        or request["dimension_bindings"] != obj["dimension_bindings"]
        or request["economics_inputs"] != obj["economics_inputs"]
    ):
        raise StrategyError(
            "ImpactCube producer_run request provenance changed"
        )
    expected_current_request = (
        None
        if obj["current_strategy_ref"] is None
        else {
            "strategy_id": obj["current_strategy_ref"]["strategy_id"],
            "expected_strategy_spec_hash": obj["current_strategy_ref"][
                "strategy_spec_hash"
            ],
        }
    )
    if request["current_strategy_ref"] != expected_current_request:
        raise StrategyError(
            "ImpactCube producer_run current strategy provenance changed"
        )
    return obj


def _prepare_output_directory(
    tasks_dir: Path | str,
    *,
    task_id: str,
) -> Path:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise StrategyError("task_id cannot escape task storage")
    root = Path(tasks_dir).absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StrategyError("task artifact root must be a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / task_id
    if task_dir.exists() and (
        task_dir.is_symlink() or not task_dir.is_dir()
    ):
        raise StrategyError(
            "task artifact directory must be a regular directory"
        )
    task_dir.mkdir(exist_ok=True)
    if (
        task_dir.is_symlink()
        or task_dir.resolve(strict=True).parent
        != root.resolve(strict=True)
    ):
        raise StrategyError("ImpactCube task directory escaped storage")
    out_dir = task_dir / "strategy_impact_cubes"
    if out_dir.exists() and (
        out_dir.is_symlink() or not out_dir.is_dir()
    ):
        raise StrategyError(
            "ImpactCube output path must be a regular directory"
        )
    out_dir.mkdir(exist_ok=True)
    if (
        out_dir.is_symlink()
        or out_dir.resolve(strict=True).parent
        != task_dir.resolve(strict=True)
    ):
        raise StrategyError("ImpactCube output directory escaped storage")
    return out_dir


def _select_artifact_row(conn, *, task_id: str, path: Path):
    return conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, IMPACT_CUBE_ARTIFACT_KIND, str(path)),
    ).fetchone()


def require_impact_cube_measurement_audit_on_connection(
    conn,
    producer_run: Mapping[str, Any],
) -> None:
    """Require the unique succeeded audit for one authenticated run record."""

    if not conn.in_transaction:
        raise StrategyError(
            "ImpactCube measurement audit requires a caller-owned transaction"
        )
    run = _json_object(producer_run, "ImpactCube producer_run")
    run_id = _text(run.get("run_id"), "ImpactCube producer_run.run_id")
    input_hash = _hash(
        run.get("input_hash"),
        "ImpactCube producer_run.input_hash",
    )
    rows = conn.execute(
        """
        SELECT actor, inputs_hash, outcome, detail_json
          FROM audit
         WHERE kind = ? AND target_ref = ?
         ORDER BY at, id
        """,
        (IMPACT_CUBE_MEASUREMENT_AUDIT_KIND, run_id),
    ).fetchall()
    if len(rows) != 1:
        state = "missing" if not rows else "duplicated"
        raise StrategyError(
            f"ImpactCube measurement audit is {state}"
        )
    row = rows[0]
    try:
        detail = json.loads(
            str(row["detail_json"]),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StrategyError(
            "ImpactCube measurement audit detail is invalid"
        ) from exc
    if (
        str(row["actor"]) != "system"
        or str(row["inputs_hash"]) != input_hash
        or str(row["outcome"]) != "succeeded"
        or detail != {"producer_run": run}
    ):
        raise StrategyError(
            "ImpactCube measurement audit binding changed"
        )


def _write_or_require_measurement_audit(
    conn,
    *,
    runtime,
    producer_run: Mapping[str, Any],
    artifact_row_existed: bool,
) -> dict[str, Any] | None:
    run = _json_object(producer_run, "ImpactCube producer_run")
    rows = conn.execute(
        """
        SELECT 1
          FROM audit
         WHERE kind = ? AND target_ref = ?
         ORDER BY at, id
        """,
        (IMPACT_CUBE_MEASUREMENT_AUDIT_KIND, run["run_id"]),
    ).fetchall()
    if artifact_row_existed:
        require_impact_cube_measurement_audit_on_connection(conn, run)
        return None
    if rows:
        raise StrategyError(
            "ImpactCube measurement audit exists without its artifact row"
        )
    runtime.repo.write_audit_on_connection(
        conn,
        kind=IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
        target_ref=run["run_id"],
        inputs_hash=run["input_hash"],
        outcome="succeeded",
        detail={"producer_run": run},
    )
    created_rows = conn.execute(
        """
        SELECT id, kind, actor, target_ref, inputs_hash, outcome,
               detail_json, at
          FROM audit
         WHERE kind = ? AND target_ref = ?
         ORDER BY at, id
        """,
        (IMPACT_CUBE_MEASUREMENT_AUDIT_KIND, run["run_id"]),
    ).fetchall()
    if len(created_rows) != 1:
        raise StrategyError(
            "ImpactCube created measurement audit is not unique"
        )
    created_row = dict(created_rows[0])
    require_impact_cube_measurement_audit_on_connection(conn, run)
    return created_row


def _remove_committed_artifact_record(
    conn,
    *,
    created_artifact_row: Mapping[str, Any] | None,
    created_audit_row: Mapping[str, Any] | None,
) -> None:
    artifact = (
        None
        if created_artifact_row is None
        else {
            field: _text(
                created_artifact_row.get(field),
                f"ImpactCube compensation artifact.{field}",
            )
            for field in _TASK_ARTIFACT_ROW_FIELDS
        }
    )
    if artifact is not None:
        artifact["id"] = _hash(
            artifact["id"],
            "ImpactCube compensation artifact.id",
        )
        artifact["content_hash"] = _hash(
            artifact["content_hash"],
            "ImpactCube compensation artifact.content_hash",
        )
    audit = (
        None
        if created_audit_row is None
        else {
            field: _text(
                created_audit_row.get(field),
                f"ImpactCube compensation audit.{field}",
            )
            for field in (
                "id",
                "kind",
                "actor",
                "target_ref",
                "inputs_hash",
                "outcome",
                "detail_json",
                "at",
            )
        }
    )
    if audit is not None:
        if audit["kind"] != IMPACT_CUBE_MEASUREMENT_AUDIT_KIND:
            raise StrategyError(
                "ImpactCube compensation audit kind is invalid"
            )
        if _MEASUREMENT_RUN_ID_RE.fullmatch(audit["target_ref"]) is None:
            raise StrategyError(
                "ImpactCube compensation producer_run_id is invalid"
            )
        audit["inputs_hash"] = _hash(
            audit["inputs_hash"],
            "ImpactCube compensation audit.inputs_hash",
        )
    conn.execute("BEGIN IMMEDIATE")
    artifact_deleted = 0
    audit_deleted = 0
    if artifact is not None:
        artifact_deleted = conn.execute(
            """
            DELETE FROM task_artifacts
             WHERE id = ? AND task_id = ? AND kind = ? AND path = ?
               AND content_hash = ? AND origin_tool = ?
               AND provenance_json = ? AND created_at = ?
            """,
            tuple(
                artifact[field]
                for field in _TASK_ARTIFACT_ROW_FIELDS
            ),
        ).rowcount
    if audit is not None:
        audit_deleted = conn.execute(
            """
            DELETE FROM audit
             WHERE id = ? AND kind = ? AND actor = ?
               AND target_ref = ? AND inputs_hash = ? AND outcome = ?
               AND detail_json = ? AND at = ?
            """,
            tuple(
                audit[field]
                for field in (
                    "id",
                    "kind",
                    "actor",
                    "target_ref",
                    "inputs_hash",
                    "outcome",
                    "detail_json",
                    "at",
                )
            ),
        ).rowcount
    if (
        (artifact is not None and artifact_deleted != 1)
        or (audit is not None and audit_deleted != 1)
    ):
        conn.rollback()
        raise StrategyError(
            "ImpactCube artifact registry compensation CAS failed"
        )
    conn.commit()


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
    root: Path,
) -> None:
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    expected = {
        "task_id": task_id,
        "kind": IMPACT_CUBE_ARTIFACT_KIND,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": IMPACT_CUBE_ORIGIN_TOOL,
    }
    if any(str(record[field]) != value for field, value in expected.items()):
        raise StrategyError(
            "existing ImpactCube artifact registry row changed"
        )
    provenance_json = _canonical_json(provenance)
    if str(record["provenance_json"]) != provenance_json:
        raise StrategyError("existing ImpactCube provenance changed")
    _require_exact_file(
        path,
        root=root,
        canonical=canonical,
        content_hash=content_hash,
    )


def _require_exact_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
) -> None:
    raw = _read_regular_nofollow(
        path,
        root=root,
        expected_content_hash=content_hash,
    )
    if raw != canonical:
        raise StrategyError("ImpactCube artifact bytes changed")


def _open_retained_exact_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
) -> tuple[int, tuple[int, ...]]:
    _require_exact_file(
        path,
        root=root,
        canonical=canonical,
        content_hash=content_hash,
    )
    descriptor = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        live = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(live.st_mode)
            or _file_identity(opened) != _file_identity(live)
        ):
            raise StrategyError(
                "ImpactCube artifact changed before registration"
            )
        identity = _stable_file_stat(opened)
        _require_retained_exact_file(
            descriptor,
            retained_identity=identity,
            path=path,
            canonical=canonical,
            content_hash=content_hash,
        )
        return descriptor, identity
    except OSError as exc:
        raise StrategyError(
            "ImpactCube artifact is unavailable before registration"
        ) from exc
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _require_retained_exact_file(
    descriptor: int,
    *,
    retained_identity: tuple[int, ...] | None,
    path: Path,
    canonical: bytes,
    content_hash: str,
) -> None:
    if descriptor < 0 or retained_identity is None:
        raise StrategyError("ImpactCube artifact verification fd is missing")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_IMPACT_CUBE_JSON_BYTES:
                raise StrategyError(
                    "ImpactCube artifact exceeds byte budget"
                )
            digest.update(chunk)
            chunks.append(chunk)
        opened = os.fstat(descriptor)
        live = os.lstat(path)
    except OSError as exc:
        raise StrategyError(
            "ImpactCube artifact changed during registration"
        ) from exc
    if (
        _stable_file_stat(opened) != retained_identity
        or not stat.S_ISREG(live.st_mode)
        or stat.S_ISLNK(live.st_mode)
        or _file_identity(opened) != _file_identity(live)
        or b"".join(chunks) != canonical
        or not hmac.compare_digest(digest.hexdigest(), content_hash)
    ):
        raise StrategyError(
            "ImpactCube artifact changed during registration"
        )


def _read_regular_nofollow(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise StrategyError("ImpactCube artifact must be a regular file")
    current = path.parent
    resolved_root = root.absolute()
    while current != resolved_root:
        if current.is_symlink():
            raise StrategyError(
                "ImpactCube artifact path traverses a symlink"
            )
        if current == current.parent:
            break
        current = current.parent
    try:
        path.resolve(strict=True).relative_to(
            resolved_root.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "ImpactCube artifact escaped task storage"
        ) from exc

    descriptor = -1
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    total = 0
    before = None
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StrategyError(
                "ImpactCube artifact must be a regular file"
            )
        if before.st_size < 0 or before.st_size > MAX_IMPACT_CUBE_JSON_BYTES:
            raise StrategyError("ImpactCube artifact exceeds byte budget")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_IMPACT_CUBE_JSON_BYTES:
                raise StrategyError(
                    "ImpactCube artifact exceeds byte budget"
                )
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise StrategyError(
                "ImpactCube artifact changed while read"
            )
        live = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(live.st_mode)
            or (live.st_dev, live.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise StrategyError(
                "ImpactCube artifact path changed while read"
            )
    except OSError as exc:
        raise StrategyError("ImpactCube artifact is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    assert before is not None
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or not hmac.compare_digest(
            digest.hexdigest(),
            expected_content_hash,
        )
    ):
        raise StrategyError("ImpactCube artifact bytes changed")
    return raw


def _require_dataset_path(path: Path, *, root: Path) -> None:
    resolved_root = root.absolute()
    if (
        not path.is_absolute()
        or resolved_root.is_symlink()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise StrategyError("ImpactCube dataset must be a regular file")
    current = path.parent
    while current != resolved_root:
        if current.is_symlink():
            raise StrategyError(
                "ImpactCube dataset path traverses a symlink"
            )
        if current == current.parent:
            break
        current = current.parent
    try:
        path.resolve(strict=True).relative_to(
            resolved_root.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "ImpactCube dataset escaped dataset storage"
        ) from exc


def _json_object(value: object, name: str) -> dict[str, Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    if normalized != value:
        raise StrategyError(f"{name} contains non-canonical JSON values")
    return normalized


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _json_value(value: object, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError(f"{name} must be canonical JSON") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        unexpected = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        details: list[str] = []
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        raise StrategyError(
            f"{name} has invalid fields ({'; '.join(details)})"
        )


def _list(value: object, name: str) -> list[Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, list):
        raise StrategyError(f"{name} must be an array")
    return normalized


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


def _binary_int(value: object, name: str) -> int:
    number = _nonnegative_int(value, name)
    if number not in {0, 1}:
        raise StrategyError(f"{name} must be 0 or 1")
    return number


def _finite_number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategyError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise StrategyError(f"{name} must be a finite number")
    if isinstance(value, Integral):
        return int(value)
    return number


__all__ = [
    "IMPACT_CUBE_ARTIFACT_KIND",
    "IMPACT_CUBE_ARTIFACT_SCHEMA_VERSION",
    "IMPACT_CUBE_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION",
    "IMPACT_CUBE_MEASUREMENT_AUDIT_KIND",
    "IMPACT_CUBE_MEASUREMENT_RUN_SCHEMA_VERSION",
    "IMPACT_CUBE_ORIGIN_TOOL",
    "IMPACT_CUBE_TOOL_SCHEMA_VERSION",
    "build_impact_cube_producer_run",
    "impact_cube_producer_run_ref",
    "require_impact_cube_measurement_audit_on_connection",
    "run_measure_strategy_impact_cube",
    "validate_impact_cube_producer_run",
    "validate_measure_strategy_impact_cube_tool_output",
]
