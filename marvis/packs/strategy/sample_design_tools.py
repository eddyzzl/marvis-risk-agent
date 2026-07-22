"""Governed Tool boundary for immutable strategy sample-design evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import hashlib
import hmac
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.data.errors import (
    DataLayerError,
    DatasetContentDriftError,
    NanLabelNotConfirmedError,
)
from marvis.data.labels import require_labels_confirmed
from marvis.data.workspace import (
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.files import sha256_file
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design import (
    MAX_SAMPLE_DESIGN_JSON_BYTES,
    MAX_SAMPLE_DESIGN_JSON_DEPTH,
    MAX_SAMPLE_DESIGN_JSON_NODES,
    MAX_SAMPLE_DESIGN_SPLIT_STRING_LENGTH,
    MAX_SAMPLE_DESIGN_SPLIT_VALUES,
    STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION,
    build_strategy_sample_design_bundle,
    canonical_strategy_sample_design_bundle_json,
    strategy_sample_design_bundle_from_json,
    validate_strategy_sample_design_bundle,
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


SAMPLE_DESIGN_TOOL_SCHEMA_VERSION = "strategy.materialize-sample-design-tool.v1"
SAMPLE_DESIGN_ARTIFACT_KIND = "strategy_sample_design_json"
SAMPLE_DESIGN_ARTIFACT_SCHEMA_VERSION = "strategy.sample-design-artifact.v1"
SAMPLE_DESIGN_ORIGIN_TOOL = "strategy.materialize_sample_design"

_INPUT_FIELDS = frozenset(
    {
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "target_bad_value",
        "performance_window_status",
        "performance_window_days",
        "observation_window_status",
        "observation_window_start",
        "observation_window_end",
        "maturity_status",
        "split_col",
        "development_values",
        "validation_values",
        "oot_values",
        "month_col",
        "weight_col",
        "loan_amount_col",
        "overdue_amount_col",
        "drop_nan_labels",
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "target_bad_value",
        "performance_window_status",
        "observation_window_status",
        "maturity_status",
        "drop_nan_labels",
    }
)
_SPLIT_FIELDS = frozenset(
    {"split_col", "development_values", "validation_values", "oot_values"}
)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "sample_design_id",
        "content_hash",
        "bundle",
        "warnings",
        "artifact",
        "development",
        "unvalidated",
        "not_created_strategy",
        "not_adopted",
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
        "format",
        "task_id",
        "bundle_id",
        "bundle_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "dataset_source_path",
        "registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "request",
        "request_hash",
    }
)
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
_TASK_ARTIFACT_RECORD_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "kind",
        "path",
        "content_hash",
        "origin_tool",
        "provenance",
        "created_at",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_SAFE_JSON_INTEGER = 2**53 - 1
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    DataLayerError,
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DatasetContentDriftError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _DatasetBinding:
    dataset_id: str
    task_id: str
    source_path: str
    path: Path
    content_hash: str
    registry_metadata_hash: str
    row_count: int
    columns: tuple[str, ...]
    workspace_revision: int
    workspace_generation: int
    semantic_mapping_hash: str
    target_col: str


@dataclass(frozen=True)
class StrategySampleDesignArtifactBinding:
    """Strictly verified immutable sample-design artifact and its bundle."""

    artifact_id: str
    task_id: str
    path: Path
    content_hash: str
    provenance: dict[str, Any]
    bundle: dict[str, Any]


def run_materialize_sample_design(inputs, ctx, runtime) -> dict[str, Any]:
    """Compute and atomically publish the exact active strategy sample boundary."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        binding = _load_dataset_binding(
            runtime,
            request=request,
            task_id=task_id,
        )
        columns = _projection_columns(request, binding=binding)
        frame = runtime.backend.read_frame(binding.path, columns=columns)
        if len(frame) != binding.row_count:
            raise StrategyError("sample-design dataset row count changed")
        if sha256_file(binding.path) != binding.content_hash:
            raise StrategyError("sample-design dataset bytes changed before computation")
        _require_binary_target(frame, binding.target_col)
        require_labels_confirmed(
            frame,
            binding.target_col,
            drop_nan_labels=request["drop_nan_labels"],
            scope="strategy sample-design active dataset",
        )
        bundle = build_strategy_sample_design_bundle(
            frame=frame,
            task_id=task_id,
            dataset_id=binding.dataset_id,
            dataset_content_hash=binding.content_hash,
            workspace_revision=binding.workspace_revision,
            workspace_generation=binding.workspace_generation,
            semantic_mapping_hash=binding.semantic_mapping_hash,
            target_col=binding.target_col,
            target_bad_value=request["target_bad_value"],
            drop_nan_labels=request["drop_nan_labels"],
            performance_window=request["performance_window"],
            observation_window=request["observation_window"],
            split_definition=request["split_definition"],
            maturity=request["maturity_status"],
            month_col=request.get("month_col"),
            weight_col=request.get("weight_col"),
            loan_amount_col=request.get("loan_amount_col"),
            overdue_amount_col=request.get("overdue_amount_col"),
            producer_version=STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION,
        )
        bundle = validate_strategy_sample_design_bundle(bundle)
        _require_live_binding(
            runtime,
            request=request,
            task_id=task_id,
            binding=binding,
        )
        return _persist_bundle(
            runtime,
            request=request,
            task_id=task_id,
            binding=binding,
            bundle=bundle,
        )
    except StrategyError:
        raise
    except NanLabelNotConfirmedError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_materialize_sample_design_tool_output(
    value: object,
) -> dict[str, Any]:
    """Fail closed when a cached Tool envelope drifts from its canonical bundle."""

    if not isinstance(value, Mapping) or set(value) != _OUTPUT_FIELDS:
        raise StrategyError("materialize_sample_design output envelope is invalid")
    normalized = _canonical_json_object(
        value,
        "materialize_sample_design output",
    )
    try:
        bundle = validate_strategy_sample_design_bundle(normalized["bundle"])
    except RecursionError as exc:
        raise StrategyError(
            "materialize_sample_design output must be canonical JSON"
        ) from exc
    design = bundle["sample_design"]
    expected = {
        "schema_version": SAMPLE_DESIGN_TOOL_SCHEMA_VERSION,
        "sample_design_id": design["sample_design_id"],
        "content_hash": design["content_hash"],
    }
    for field, expected_value in expected.items():
        if normalized[field] != expected_value:
            raise StrategyError(f"materialize_sample_design output {field} drifted")
    expected_warnings = _bundle_warnings(bundle)
    if normalized["warnings"] != expected_warnings:
        raise StrategyError("materialize_sample_design output warnings drifted")
    for field in (
        "development",
        "unvalidated",
        "not_created_strategy",
        "not_adopted",
        "not_deployed",
    ):
        if normalized[field] is not True:
            raise StrategyError(f"materialize_sample_design output {field} must be true")

    artifact = normalized["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != _OUTPUT_ARTIFACT_FIELDS:
        raise StrategyError("materialize_sample_design output artifact is invalid")
    artifact_id = _hash(artifact["artifact_id"], "artifact_id")
    task_id = _text(design["identity"]["task_id"], "sample design task_id")
    artifact_hash = hashlib.sha256(
        canonical_strategy_sample_design_bundle_json(bundle).encode("utf-8")
    ).hexdigest()
    expected_artifact = {
        "artifact_id": artifact_id,
        "kind": SAMPLE_DESIGN_ARTIFACT_KIND,
        "format": "json",
        "filename": f"{design['sample_design_id']}.json",
        "content_hash": artifact_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
            f"?expected_content_hash={artifact_hash}"
        ),
    }
    for field, expected_value in expected_artifact.items():
        if artifact[field] != expected_value:
            raise StrategyError(f"materialize_sample_design artifact {field} drifted")
    normalized["bundle"] = bundle
    return normalized


def load_strategy_sample_design_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_sample_design_id: str,
    expected_sample_design_content_hash: str,
) -> StrategySampleDesignArtifactBinding:
    """Load a task-owned sample design only after strict registry/byte validation."""

    normalized_task_id = _text(task_id, "task_id")
    normalized_artifact_id = _hash(artifact_id, "artifact_id")
    artifact_hash = _hash(
        expected_artifact_content_hash,
        "expected_artifact_content_hash",
    )
    sample_design_id = _text(expected_sample_design_id, "expected_sample_design_id")
    sample_design_hash = _hash(
        expected_sample_design_content_hash,
        "expected_sample_design_content_hash",
    )
    try:
        record = runtime.task_artifacts.get_for_task(
            normalized_task_id,
            normalized_artifact_id,
        )
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc
    if record is None:
        raise StrategyError("strategy sample-design artifact not found")
    if not isinstance(record, Mapping) or set(record) != _TASK_ARTIFACT_RECORD_FIELDS:
        raise StrategyError("strategy sample-design artifact registry row is invalid")
    if (
        record["id"] != normalized_artifact_id
        or record["task_id"] != normalized_task_id
        or record["kind"] != SAMPLE_DESIGN_ARTIFACT_KIND
        or record["origin_tool"] != SAMPLE_DESIGN_ORIGIN_TOOL
        or not _matches_hash(record["content_hash"], artifact_hash)
    ):
        raise StrategyError("strategy sample-design artifact registry binding changed")
    provenance = _canonical_json_object(
        record["provenance"],
        "strategy sample-design artifact provenance",
    )
    _require_exact_fields(
        provenance,
        _PROVENANCE_FIELDS,
        "strategy sample-design artifact provenance",
    )
    _validate_provenance_scalars(provenance)
    if (
        provenance["schema_version"] != SAMPLE_DESIGN_ARTIFACT_SCHEMA_VERSION
        or provenance["producer_version"]
        != STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION
        or provenance["format"] != "json"
        or provenance["task_id"] != normalized_task_id
        or provenance["sample_design_id"] != sample_design_id
        or not _matches_hash(
            provenance["sample_design_content_hash"], sample_design_hash
        )
    ):
        raise StrategyError("strategy sample-design artifact provenance changed")
    expected_path = _canonical_artifact_path(
        runtime.settings.tasks_dir,
        task_id=normalized_task_id,
        sample_design_id=sample_design_id,
    )
    path = Path(_text(record["path"], "artifact path"))
    if path != expected_path:
        raise StrategyError("strategy sample-design artifact path is not canonical")
    raw = _read_verified_artifact(
        path,
        root=Path(runtime.settings.tasks_dir),
        expected_content_hash=artifact_hash,
    )
    try:
        bundle = strategy_sample_design_bundle_from_json(raw.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("strategy sample-design artifact JSON is invalid") from exc
    bundle = validate_strategy_sample_design_bundle(bundle)
    canonical = canonical_strategy_sample_design_bundle_json(bundle).encode("utf-8")
    if len(canonical) > MAX_SAMPLE_DESIGN_JSON_BYTES:
        raise StrategyError("strategy sample-design artifact exceeds byte budget")
    if raw != canonical:
        raise StrategyError("strategy sample-design artifact bytes are not canonical")
    design = bundle["sample_design"]
    if (
        design["sample_design_id"] != sample_design_id
        or not _matches_hash(design["content_hash"], sample_design_hash)
        or bundle["bundle_id"] != provenance["bundle_id"]
        or not _matches_hash(bundle["content_hash"], provenance["bundle_content_hash"])
    ):
        raise StrategyError("strategy sample-design artifact content binding changed")
    _require_bundle_provenance(bundle, provenance)
    _require_loaded_source_live(runtime, provenance=provenance)
    return StrategySampleDesignArtifactBinding(
        artifact_id=normalized_artifact_id,
        task_id=normalized_task_id,
        path=path,
        content_hash=artifact_hash,
        provenance=provenance,
        bundle=bundle,
    )


def _validate_inputs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError("materialize_sample_design inputs must be an object")
    if any(not isinstance(key, str) for key in value):
        raise StrategyError("materialize_sample_design input keys must be strings")
    missing = sorted(_REQUIRED_FIELDS - set(value))
    unexpected = sorted(set(value) - _INPUT_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(
            "invalid materialize_sample_design inputs (" + "; ".join(details) + ")"
        )
    request: dict[str, Any] = {
        "dataset_id": _text(value["dataset_id"], "dataset_id"),
        "expected_dataset_content_hash": _hash(
            value["expected_dataset_content_hash"],
            "expected_dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"], "workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"], "workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "semantic_mapping_hash"
        ),
        "target_col": _text(value["target_col"], "target_col"),
        "target_bad_value": _binary_value(
            value["target_bad_value"], "target_bad_value"
        ),
        "maturity_status": _enum(
            value["maturity_status"],
            "maturity_status",
            {"confirmed_matured", "not_matured", "unknown"},
        ),
    }
    if not isinstance(value["drop_nan_labels"], bool):
        raise StrategyError("drop_nan_labels must be boolean")
    request["drop_nan_labels"] = value["drop_nan_labels"]
    request["performance_window"] = _performance_window(value)
    request["observation_window"] = _observation_window(value)
    request["split_definition"] = _split_definition(value)
    for field in (
        "month_col",
        "weight_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        if field in value:
            request[field] = _text(value[field], field)
    optional_columns = [
        request[field]
        for field in (
            "month_col",
            "weight_col",
            "loan_amount_col",
            "overdue_amount_col",
        )
        if field in request
    ]
    if request["target_col"] in optional_columns:
        raise StrategyError("sample-design target and optional fields must be distinct")
    if len(optional_columns) != len(set(optional_columns)):
        raise StrategyError("sample-design optional fields must be distinct")
    split_col = request["split_definition"]["column"]
    all_columns = [request["target_col"], *optional_columns]
    if split_col is not None:
        all_columns.append(split_col)
    if len(all_columns) != len(set(all_columns)):
        raise StrategyError("sample-design column bindings must be distinct")
    return request


def _performance_window(value: Mapping[str, Any]) -> dict[str, Any]:
    status = _enum(
        value["performance_window_status"],
        "performance_window_status",
        {"provided", "unavailable"},
    )
    present = "performance_window_days" in value
    if status == "provided":
        if not present:
            raise StrategyError(
                "provided performance window requires performance_window_days"
            )
        days = _positive_int(value["performance_window_days"], "performance_window_days")
        return {"status": "provided", "days": days}
    if present:
        raise StrategyError(
            "unavailable performance window forbids performance_window_days"
        )
    return {"status": "unavailable", "days": None}


def _observation_window(value: Mapping[str, Any]) -> dict[str, Any]:
    status = _enum(
        value["observation_window_status"],
        "observation_window_status",
        {"provided", "unavailable"},
    )
    fields = {"observation_window_start", "observation_window_end"}
    present = fields & set(value)
    if status == "provided":
        if present != fields:
            raise StrategyError(
                "provided observation window requires start and end"
            )
        start = _iso_date(value["observation_window_start"], "observation_window_start")
        end = _iso_date(value["observation_window_end"], "observation_window_end")
        if start > end:
            raise StrategyError("observation_window_start must not be after end")
        return {"status": "provided", "start": start, "end": end}
    if present:
        raise StrategyError("unavailable observation window forbids start and end")
    return {"status": "unavailable", "start": None, "end": None}


def _split_definition(value: Mapping[str, Any]) -> dict[str, Any]:
    present = _SPLIT_FIELDS & set(value)
    if not present:
        return {
            "status": "unavailable",
            "column": None,
            "development_values": [],
            "validation_values": [],
            "oot_values": [],
        }
    if present != _SPLIT_FIELDS:
        raise StrategyError(
            "split_col and development/validation/OOT values must be supplied together"
        )
    groups = {
        name: _split_values(value[name], name)
        for name in ("development_values", "validation_values", "oot_values")
    }
    if not groups["development_values"]:
        raise StrategyError("sample-design development_values must not be empty")
    seen: dict[str, str] = {}
    for name, items in groups.items():
        for item in items:
            key = _split_scalar_identity(item)
            previous = seen.get(key)
            if previous is not None:
                raise StrategyError(
                    f"sample-design split value appears in both {previous} and {name}"
                )
            seen[key] = name
    return {
        "status": "available",
        "column": _text(value["split_col"], "split_col"),
        **{
            name: sorted(items, key=_split_scalar_identity)
            for name, items in groups.items()
        },
    }


def _split_values(value: object, name: str) -> list[str | int | float | bool]:
    if not isinstance(value, list):
        raise StrategyError(f"{name} must be an array")
    if len(value) > MAX_SAMPLE_DESIGN_SPLIT_VALUES:
        raise StrategyError(f"{name} exceeds item budget")
    result: list[str | int | float | bool] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            if not item or "\x00" in item:
                raise StrategyError(f"{name} values must be non-empty JSON scalars")
            if len(item) > MAX_SAMPLE_DESIGN_SPLIT_STRING_LENGTH:
                raise StrategyError(f"{name} values exceed string length budget")
            normalized: str | int | float | bool = item
        elif isinstance(item, bool):
            normalized = item
        elif isinstance(item, int):
            if abs(item) > _MAX_SAFE_JSON_INTEGER:
                raise StrategyError(f"{name} values exceed exact JSON numeric range")
            normalized = item
        elif isinstance(item, float) and math.isfinite(item):
            if abs(item) > _MAX_SAFE_JSON_INTEGER:
                raise StrategyError(f"{name} values exceed exact JSON numeric range")
            normalized = int(item) if item == 0 or item.is_integer() else item
        else:
            raise StrategyError(f"{name} values must be finite JSON scalars")
        key = _split_scalar_identity(normalized)
        if key in seen:
            raise StrategyError(f"{name} values must be unique")
        seen.add(key)
        result.append(normalized)
    return result


def _split_scalar_identity(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    else:
        kind = "string"
    return _canonical_json([kind, value])


def _load_dataset_binding(
    runtime,
    *,
    request: Mapping[str, Any],
    task_id: str,
) -> _DatasetBinding:
    workspace = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(task_id)
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    if (
        workspace.active_dataset_id != request["dataset_id"]
        or workspace.active_dataset_content_hash
        != request["expected_dataset_content_hash"]
        or workspace.revision != request["workspace_revision"]
        or workspace.analysis_generation != request["workspace_generation"]
        or not hmac.compare_digest(semantic_hash, request["semantic_mapping_hash"])
        or workspace.semantic_mapping.target_col != request["target_col"]
    ):
        raise StrategyError("DataWorkspace binding changed before sample design")
    try:
        dataset = runtime.registry.get(request["dataset_id"])
        path = Path(runtime.registry.resolve_verified_path(request["dataset_id"]))
    except (DatasetContentDriftError, KeyError, OSError, TypeError, ValueError) as exc:
        raise StrategyError("sample-design dataset is unavailable or drifted") from exc
    if str(dataset.task_id) != task_id:
        raise StrategyError("sample-design dataset belongs to another task")
    content_hash = str(dataset.content_hash or "")
    if not _matches_hash(content_hash, request["expected_dataset_content_hash"]):
        raise StrategyError("sample-design dataset content hash changed")
    if sha256_file(path) != content_hash:
        raise StrategyError("sample-design dataset bytes changed")
    columns = tuple(str(column.name) for column in dataset.columns)
    required_columns = _requested_columns(request)
    missing = sorted(required_columns - set(columns))
    if missing:
        raise StrategyError(
            "sample-design dataset is missing columns: " + ", ".join(missing)
        )
    with runtime.task_artifacts.transaction() as conn:
        registry_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=str(dataset.id),
            expected_content_hash=content_hash,
        )
    return _DatasetBinding(
        dataset_id=str(dataset.id),
        task_id=task_id,
        source_path=str(dataset.source_path),
        path=path,
        content_hash=content_hash,
        registry_metadata_hash=registry_hash,
        row_count=int(dataset.row_count),
        columns=columns,
        workspace_revision=workspace.revision,
        workspace_generation=workspace.analysis_generation,
        semantic_mapping_hash=semantic_hash,
        target_col=request["target_col"],
    )


def _requested_columns(request: Mapping[str, Any]) -> set[str]:
    columns = {str(request["target_col"])}
    for field in ("month_col", "weight_col", "loan_amount_col", "overdue_amount_col"):
        if field in request:
            columns.add(str(request[field]))
    split_col = request["split_definition"]["column"]
    if split_col is not None:
        columns.add(str(split_col))
    return columns


def _projection_columns(
    request: Mapping[str, Any], *, binding: _DatasetBinding
) -> list[str]:
    requested = _requested_columns(request)
    return [column for column in binding.columns if column in requested]


def _require_binary_target(frame, target_col: str) -> None:
    observed = False
    for raw in frame[target_col].tolist():
        try:
            missing_value = pd.isna(raw)
        except (TypeError, ValueError):
            missing_value = False
        try:
            if bool(missing_value):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise StrategyError(
                "sample-design target must contain numeric 0/1 values or missing"
            )
        number = float(raw)
        if not math.isfinite(number) or number not in {0.0, 1.0}:
            raise StrategyError(
                "sample-design target must contain numeric 0/1 values or missing"
            )
        observed = True
    if not observed:
        raise StrategyError("sample-design target has no observed binary labels")


def _require_live_binding(
    runtime,
    *,
    request: Mapping[str, Any],
    task_id: str,
    binding: _DatasetBinding,
) -> None:
    current = _load_dataset_binding(runtime, request=request, task_id=task_id)
    if current != binding:
        raise StrategyError("sample-design dataset binding changed during computation")


def _persist_bundle(
    runtime,
    *,
    request: Mapping[str, Any],
    task_id: str,
    binding: _DatasetBinding,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = canonical_strategy_sample_design_bundle_json(bundle).encode("utf-8")
    artifact_content_hash = hashlib.sha256(canonical).hexdigest()
    design = bundle["sample_design"]
    sample_design_id = _text(design["sample_design_id"], "sample_design_id")
    out_dir = _prepare_output_directory(runtime.settings.tasks_dir, task_id=task_id)
    final_path = out_dir / f"{sample_design_id}.json"
    request_evidence = _request_evidence(request)
    provenance = {
        "schema_version": SAMPLE_DESIGN_ARTIFACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION,
        "format": "json",
        "task_id": task_id,
        "bundle_id": bundle["bundle_id"],
        "bundle_content_hash": bundle["content_hash"],
        "sample_design_id": sample_design_id,
        "sample_design_content_hash": design["content_hash"],
        "dataset_id": binding.dataset_id,
        "dataset_content_hash": binding.content_hash,
        "dataset_source_path": binding.source_path,
        "registry_metadata_hash": binding.registry_metadata_hash,
        "workspace_revision": binding.workspace_revision,
        "workspace_generation": binding.workspace_generation,
        "semantic_mapping_hash": binding.semantic_mapping_hash,
        "target_col": binding.target_col,
        "request": request_evidence,
        "request_hash": hashlib.sha256(
            _canonical_json(request_evidence).encode("utf-8")
        ).hexdigest(),
    }
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError("strategy sample-design artifact could not be staged") from exc
    db_committed = False
    rollback_under_lock = False
    reused = False
    record: Mapping[str, Any]
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_dataset_and_workspace_on_connection(
                    conn,
                    request=request,
                    task_id=task_id,
                    binding=binding,
                )
                row = conn.execute(
                    """
                    SELECT id, task_id, kind, path, content_hash, origin_tool,
                           provenance_json, created_at
                      FROM task_artifacts
                     WHERE task_id = ? AND kind = ? AND path = ?
                    """,
                    (task_id, SAMPLE_DESIGN_ARTIFACT_KIND, str(final_path)),
                ).fetchone()
                if row is not None:
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        final_path=final_path,
                        canonical=canonical,
                        content_hash=artifact_content_hash,
                        provenance=provenance,
                    )
                    uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        _verify_artifact_file(
                            final_path,
                            root=Path(runtime.settings.tasks_dir),
                            canonical=canonical,
                            content_hash=artifact_content_hash,
                        )
                        # Recover an exact content-addressed file left behind if
                        # the process died after promotion but before the DB
                        # transaction committed.  Any byte/path drift still
                        # fails closed in _verify_artifact_file.
                        uow.rollback()
                        reused = True
                    else:
                        uow.promote_all()
                        _verify_artifact_file(
                            final_path,
                            root=Path(runtime.settings.tasks_dir),
                            canonical=canonical,
                            content_hash=artifact_content_hash,
                        )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=SAMPLE_DESIGN_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_content_hash,
                    origin_tool=SAMPLE_DESIGN_ORIGIN_TOOL,
                    provenance=provenance,
                )
                conn.commit()
                db_committed = True
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
    return validate_materialize_sample_design_tool_output(
        _tool_output(bundle, record=record, task_id=task_id)
    )


def _request_evidence(request: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in request.items()
        if key not in {"performance_window", "observation_window", "split_definition"}
    }
    result.update(
        {
            "performance_window": request["performance_window"],
            "observation_window": request["observation_window"],
            "split_definition": request["split_definition"],
        }
    )
    return _canonical_json_object(result, "sample-design request evidence")


def _require_dataset_and_workspace_on_connection(
    conn,
    *,
    request: Mapping[str, Any],
    task_id: str,
    binding: _DatasetBinding,
) -> None:
    registry_hash = _registry_metadata_hash_on_connection(
        conn,
        task_id=task_id,
        dataset_id=binding.dataset_id,
        expected_content_hash=binding.content_hash,
    )
    if not hmac.compare_digest(registry_hash, binding.registry_metadata_hash):
        raise StrategyError("dataset registry metadata changed before registration")
    row = conn.execute(
        "SELECT source_path FROM datasets WHERE task_id = ? AND id = ?",
        (task_id, binding.dataset_id),
    ).fetchone()
    if row is None or str(row["source_path"]) != binding.source_path:
        raise StrategyError("dataset registry path changed before registration")
    if sha256_file(binding.path) != binding.content_hash:
        raise StrategyError("sample-design dataset bytes changed before registration")
    row = conn.execute(
        """
        SELECT revision, active_dataset_id, active_dataset_content_hash,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        raise StrategyError("DataWorkspace disappeared before registration")
    try:
        raw_mapping = str(row["semantic_mapping_json"])
        mapping = data_semantic_mapping_from_dict(json.loads(raw_mapping))
        canonical_mapping = _canonical_json(
            {
                "target_col": mapping.target_col,
                "field_roles": dict(mapping.field_roles),
                "business_names": dict(mapping.business_names),
            }
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("DataWorkspace semantic mapping is invalid") from exc
    if raw_mapping != canonical_mapping:
        raise StrategyError("DataWorkspace semantic mapping is not canonical")
    if (
        int(row["revision"]) != request["workspace_revision"]
        or str(row["active_dataset_id"]) != binding.dataset_id
        or str(row["active_dataset_content_hash"]) != binding.content_hash
        or int(row["analysis_generation"]) != request["workspace_generation"]
        or not hmac.compare_digest(
            data_semantic_mapping_hash(mapping), request["semantic_mapping_hash"]
        )
        or mapping.target_col != request["target_col"]
    ):
        raise StrategyError("DataWorkspace changed before sample-design registration")


def _registry_metadata_hash_on_connection(
    conn,
    *,
    task_id: str,
    dataset_id: str,
    expected_content_hash: str,
) -> str:
    row = conn.execute(
        """
        SELECT task_id, row_count, columns_json, has_target, target_col,
               content_hash
          FROM datasets WHERE id = ?
        """,
        (dataset_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError("sample-design dataset is not owned by the task")
    if not _matches_hash(row["content_hash"], expected_content_hash):
        raise StrategyError("sample-design dataset registered hash changed")
    columns_json = row["columns_json"]
    if not isinstance(columns_json, str):
        raise StrategyError("sample-design dataset schema is invalid")
    try:
        json.loads(columns_json)
    except json.JSONDecodeError as exc:
        raise StrategyError("sample-design dataset schema is invalid") from exc
    payload = {
        "row_count": int(row["row_count"]),
        "columns_json": columns_json,
        "has_target": int(row["has_target"]),
        "target_col": row["target_col"],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _prepare_output_directory(tasks_dir: Path | str, *, task_id: str) -> Path:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise StrategyError("task_id cannot address paths outside task storage")
    root = Path(tasks_dir).absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StrategyError("task artifact root must be a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / task_id
    if task_dir.exists() and (task_dir.is_symlink() or not task_dir.is_dir()):
        raise StrategyError("task artifact directory must be a regular directory")
    task_dir.mkdir(exist_ok=True)
    if task_dir.is_symlink() or task_dir.resolve(strict=True).parent != root.resolve(
        strict=True
    ):
        raise StrategyError("strategy sample-design directory escaped task storage")
    out_dir = task_dir / "strategy_sample_designs"
    if out_dir.exists() and (out_dir.is_symlink() or not out_dir.is_dir()):
        raise StrategyError("strategy sample-design path must be a regular directory")
    out_dir.mkdir(exist_ok=True)
    if out_dir.is_symlink() or out_dir.resolve(strict=True).parent != task_dir.resolve(
        strict=True
    ):
        raise StrategyError("strategy sample-design directory escaped task storage")
    return out_dir


def _canonical_artifact_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    sample_design_id: str,
) -> Path:
    if Path(task_id).name != task_id or Path(sample_design_id).name != sample_design_id:
        raise StrategyError("strategy sample-design artifact identity is not path-safe")
    return (
        Path(tasks_dir).absolute()
        / task_id
        / "strategy_sample_designs"
        / f"{sample_design_id}.json"
    )


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    final_path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    expected = {
        "task_id": task_id,
        "kind": SAMPLE_DESIGN_ARTIFACT_KIND,
        "path": str(final_path),
        "content_hash": content_hash,
        "origin_tool": SAMPLE_DESIGN_ORIGIN_TOOL,
    }
    if any(str(record[field]) != expected_value for field, expected_value in expected.items()):
        raise StrategyError("existing strategy sample-design registry row changed")
    if str(record["provenance_json"]) != _canonical_json(provenance):
        raise StrategyError("existing strategy sample-design provenance changed")
    _verify_artifact_file(
        final_path,
        root=final_path.parents[2],
        canonical=canonical,
        content_hash=content_hash,
    )


def _verify_artifact_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
) -> None:
    raw = _read_verified_artifact(
        path,
        root=root,
        expected_content_hash=content_hash,
    )
    if raw != canonical:
        raise StrategyError("strategy sample-design artifact bytes changed")


def _read_verified_artifact(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise StrategyError("strategy sample-design artifact must be a regular file")
    current = path.parent
    root_absolute = root.absolute()
    while current != root_absolute:
        if current.is_symlink():
            raise StrategyError(
                "strategy sample-design artifact path must not traverse symlinks"
            )
        if current == current.parent:
            break
        current = current.parent
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError("strategy sample-design artifact escaped task storage") from exc
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_SAMPLE_DESIGN_JSON_BYTES + 1)
            if len(raw) > MAX_SAMPLE_DESIGN_JSON_BYTES or stream.read(1):
                raise StrategyError(
                    "strategy sample-design artifact exceeds byte budget"
                )
    except OSError as exc:
        raise StrategyError(
            "strategy sample-design artifact could not be read"
        ) from exc
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_content_hash):
        raise StrategyError("strategy sample-design artifact content hash changed")
    return raw


def _require_bundle_provenance(
    bundle: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    if bundle["producer_version"] != provenance["producer_version"]:
        raise StrategyError(
            "strategy sample-design provenance producer_version does not match bundle"
        )
    design = bundle["sample_design"]
    identity = design["identity"]
    dataset_ref = identity["dataset_ref"]
    workspace_ref = identity["workspace_ref"]
    expected = {
        "task_id": identity["task_id"],
        "dataset_id": dataset_ref["dataset_id"],
        "dataset_content_hash": dataset_ref["content_hash"],
        "workspace_revision": workspace_ref["revision"],
        "workspace_generation": workspace_ref["generation"],
        "semantic_mapping_hash": workspace_ref["semantic_mapping_hash"],
        "target_col": design["target_definition"]["column"],
    }
    for field, expected_value in expected.items():
        if provenance[field] != expected_value:
            raise StrategyError(
                f"strategy sample-design provenance {field} does not match bundle"
            )
    request = _canonical_json_object(
        provenance["request"],
        "strategy sample-design provenance request",
    )
    calculated_request_hash = hashlib.sha256(
        _canonical_json(request).encode("utf-8")
    ).hexdigest()
    if not _matches_hash(provenance["request_hash"], calculated_request_hash):
        raise StrategyError("strategy sample-design provenance request_hash changed")
    request_expected = {
        "dataset_id": dataset_ref["dataset_id"],
        "expected_dataset_content_hash": dataset_ref["content_hash"],
        "workspace_revision": workspace_ref["revision"],
        "workspace_generation": workspace_ref["generation"],
        "semantic_mapping_hash": workspace_ref["semantic_mapping_hash"],
        "target_col": design["target_definition"]["column"],
        "target_bad_value": design["target_definition"]["bad_value"],
        "drop_nan_labels": design["target_definition"]["drop_nan_labels"],
        "performance_window": design["performance_window"],
        "observation_window": design["observation_window"],
        "split_definition": design["split_definition"],
        "maturity_status": design["maturity"],
    }
    optional_fields = design["optional_fields"]
    for request_field, design_field in (
        ("month_col", "month_field"),
        ("weight_col", "weight_field"),
        ("loan_amount_col", "loan_amount_field"),
        ("overdue_amount_col", "overdue_amount_field"),
    ):
        if optional_fields[design_field] is not None:
            request_expected[request_field] = optional_fields[design_field]
    if _canonical_json(request) != _canonical_json(request_expected):
        raise StrategyError(
            "strategy sample-design provenance request does not match bundle"
        )
    if design["active_dataset_boundary"] != {
        "status": "materialized_active_dataset",
        "population_count": design["active_dataset_boundary"]["population_count"],
        "inclusion_rules": ["all_rows_in_active_dataset"],
        "exclusion_rules": ["upstream_exclusions_already_materialized"],
        "applies_filters": False,
    }:
        raise StrategyError("strategy sample-design active boundary is not exact")


def _validate_provenance_scalars(provenance: Mapping[str, Any]) -> None:
    for field in (
        "schema_version",
        "producer_version",
        "format",
        "task_id",
        "bundle_id",
        "sample_design_id",
        "dataset_id",
        "dataset_source_path",
        "target_col",
    ):
        _text(provenance[field], f"sample-design provenance {field}")
    for field in (
        "bundle_content_hash",
        "sample_design_content_hash",
        "dataset_content_hash",
        "registry_metadata_hash",
        "semantic_mapping_hash",
        "request_hash",
    ):
        _hash(provenance[field], f"sample-design provenance {field}")
    _non_negative_int(
        provenance["workspace_revision"],
        "sample-design provenance workspace_revision",
    )
    _non_negative_int(
        provenance["workspace_generation"],
        "sample-design provenance workspace_generation",
    )


def _require_loaded_source_live(runtime, *, provenance: Mapping[str, Any]) -> None:
    """Re-authenticate the dataset registry, bytes, and current workspace binding."""

    task_id = provenance["task_id"]
    dataset_id = provenance["dataset_id"]
    try:
        dataset = runtime.registry.get(dataset_id)
        path = Path(runtime.registry.resolve_verified_path(dataset_id))
    except (DatasetContentDriftError, KeyError, OSError, TypeError, ValueError) as exc:
        raise StrategyError(
            "strategy sample-design source dataset is unavailable or drifted"
        ) from exc
    if (
        str(dataset.task_id) != task_id
        or str(dataset.source_path) != provenance["dataset_source_path"]
        or not _matches_hash(dataset.content_hash, provenance["dataset_content_hash"])
        or sha256_file(path) != provenance["dataset_content_hash"]
    ):
        raise StrategyError("strategy sample-design source dataset binding changed")
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=dataset_id,
            expected_content_hash=provenance["dataset_content_hash"],
        )
    if not _matches_hash(metadata_hash, provenance["registry_metadata_hash"]):
        raise StrategyError("strategy sample-design dataset metadata changed")
    try:
        workspace = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task_id
        )
    except (DataWorkspaceDataError, DataWorkspaceDatasetNotFound) as exc:
        raise StrategyError("strategy sample-design DataWorkspace is unavailable") from exc
    if (
        workspace.active_dataset_id != dataset_id
        or workspace.active_dataset_content_hash != provenance["dataset_content_hash"]
        or workspace.revision != provenance["workspace_revision"]
        or workspace.analysis_generation != provenance["workspace_generation"]
        or not hmac.compare_digest(
            data_semantic_mapping_hash(workspace.semantic_mapping),
            provenance["semantic_mapping_hash"],
        )
        or workspace.semantic_mapping.target_col != provenance["target_col"]
    ):
        raise StrategyError("strategy sample-design DataWorkspace binding changed")


def _tool_output(
    bundle: Mapping[str, Any], *, record: Mapping[str, Any], task_id: str
) -> dict[str, Any]:
    design = bundle["sample_design"]
    path = Path(str(record["path"]))
    artifact_id = str(record["id"])
    return {
        "schema_version": SAMPLE_DESIGN_TOOL_SCHEMA_VERSION,
        "sample_design_id": design["sample_design_id"],
        "content_hash": design["content_hash"],
        "bundle": dict(bundle),
        "warnings": _bundle_warnings(bundle),
        "artifact": {
            "artifact_id": artifact_id,
            "kind": SAMPLE_DESIGN_ARTIFACT_KIND,
            "format": "json",
            "filename": path.name,
            "content_hash": str(record["content_hash"]),
            "download_url": (
                f"/api/tasks/{quote(task_id, safe='')}"
                f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                f"?expected_content_hash={record['content_hash']}"
            ),
        },
        "development": True,
        "unvalidated": True,
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _bundle_warnings(bundle: Mapping[str, Any]) -> list[str]:
    try:
        flags = bundle["sample_design"]["red_flags"]
    except (KeyError, TypeError) as exc:
        raise StrategyError("strategy sample-design red_flags are invalid") from exc
    if not isinstance(flags, list):
        raise StrategyError("strategy sample-design red_flags are invalid")
    warnings: list[str] = []
    for flag in flags:
        if not isinstance(flag, Mapping):
            raise StrategyError("strategy sample-design red flag is invalid")
        warnings.append(_text(flag.get("message"), "sample-design red flag message"))
    return warnings


def _canonical_json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{name} must be an object")
    _preflight_json_value(value, name=name)
    try:
        encoded = _canonical_json(dict(value))
        if len(encoded.encode("utf-8")) > MAX_SAMPLE_DESIGN_JSON_BYTES:
            raise StrategyError(f"{name} exceeds byte budget")
        normalized = json.loads(encoded)
    except (TypeError, ValueError, RecursionError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} must be canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    return normalized


def _preflight_json_value(value: object, *, name: str) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    seen_containers: set[int] = set()
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_SAMPLE_DESIGN_JSON_NODES:
            raise StrategyError(f"{name} exceeds JSON node budget")
        if depth > MAX_SAMPLE_DESIGN_JSON_DEPTH:
            raise StrategyError(f"{name} exceeds JSON depth budget")
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen_containers:
                raise StrategyError(f"{name} contains a repeated or cyclic container")
            seen_containers.add(identity)
            if any(not isinstance(key, str) for key in current):
                raise StrategyError(f"{name} keys must be strings")
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            identity = id(current)
            if identity in seen_containers:
                raise StrategyError(f"{name} contains a repeated or cyclic container")
            seen_containers.add(identity)
            stack.extend((child, depth + 1) for child in current)
        elif current is None or isinstance(current, (str, bool, int, float)):
            if isinstance(current, float) and not math.isfinite(current):
                raise StrategyError(f"{name} contains a non-finite number")
        else:
            raise StrategyError(
                f"{name} contains unsupported JSON value {type(current).__name__}"
            )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    if set(value) != expected:
        raise StrategyError(f"{name} fields are invalid")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _matches_hash(value: object, expected: str) -> bool:
    return (
        isinstance(value, str)
        and _HASH_RE.fullmatch(value) is not None
        and hmac.compare_digest(value, expected)
    )


def _enum(value: object, name: str, allowed: set[str]) -> str:
    normalized = _text(value, name)
    if normalized not in allowed:
        raise StrategyError(f"{name} must be one of: " + ", ".join(sorted(allowed)))
    return normalized


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized < 1:
        raise StrategyError(f"{name} must be at least 1")
    return normalized


def _binary_value(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) not in {0.0, 1.0}
    ):
        raise StrategyError(f"{name} must be integer 0 or 1")
    return int(value)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _iso_date(value: object, name: str) -> str:
    normalized = _text(value, name)
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as exc:
        raise StrategyError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc
    if parsed.isoformat() != normalized:
        raise StrategyError(f"{name} must be an ISO date (YYYY-MM-DD)")
    return normalized


__all__ = [
    "SAMPLE_DESIGN_ARTIFACT_KIND",
    "SAMPLE_DESIGN_ARTIFACT_SCHEMA_VERSION",
    "SAMPLE_DESIGN_ORIGIN_TOOL",
    "SAMPLE_DESIGN_TOOL_SCHEMA_VERSION",
    "StrategySampleDesignArtifactBinding",
    "load_strategy_sample_design_artifact",
    "run_materialize_sample_design",
    "validate_materialize_sample_design_tool_output",
]
