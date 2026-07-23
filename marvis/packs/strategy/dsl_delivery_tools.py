"""Governed, downloadable delivery bundle for canonical Strategy DSL.

The Tool binds one immutable strategy definition and one task-owned dataset,
generates standalone Python, DuckDB SQL, and canonical JSON, then reconciles a
bounded row sample across the MARVIS evaluator and both generated engines.
Files, TaskArtifact rows, and the audit record share one SQLite writer
transaction and one rollback-capable filesystem unit of work.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import DatasetContentDriftError
from marvis.files import sha256_file
from marvis.packs.strategy.dsl import (
    canonical_strategy_json,
    parse_strategy_spec,
)
from marvis.packs.strategy.dsl_delivery import (
    MAX_EQUIVALENCE_ROWS,
    StrategyDeliveryError,
    generate_strategy_duckdb_sql_source,
    generate_strategy_python_source,
    validate_strategy_delivery_equivalence,
    verify_strategy_delivery_equivalence,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.legacy_adapter import legacy_strategy_to_spec
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.strategy import _strategy_spec_hash_from_row
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


DELIVERY_TOOL_SCHEMA_VERSION = "strategy.export-dsl-delivery-tool.v1"
DELIVERY_ARTIFACT_SCHEMA_VERSION = "strategy.dsl-delivery-artifact.v1"
DELIVERY_PRODUCER_VERSION = "strategy.dsl-delivery-producer.v1"
DELIVERY_ORIGIN_TOOL = "strategy.export_strategy_delivery"
DELIVERY_AUDIT_KIND = "strategy.delivery.exported"
DELIVERY_ARTIFACT_KINDS = {
    "python": "strategy_delivery_python",
    "sql": "strategy_delivery_sql",
    "strategy_json": "strategy_delivery_json",
    "equivalence_json": "strategy_delivery_equivalence_json",
}

_STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)
_INPUT_FIELDS = frozenset(
    {"strategy_ref", "dataset_ref", "maximum_equivalence_rows"}
)
_STRATEGY_REF_FIELDS = frozenset(
    {
        "strategy_id",
        "expected_strategy_type",
        "expected_version",
        "expected_spec_hash",
    }
)
_DATASET_REF_FIELDS = frozenset(
    {"dataset_id", "expected_content_hash"}
)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "delivery_id",
        "task_id",
        "strategy_id",
        "strategy_type",
        "strategy_version",
        "strategy_ref",
        "dataset_ref",
        "source_row_count",
        "maximum_equivalence_rows",
        "equivalence",
        "artifacts",
        "not_applied",
        "not_adopted",
        "not_deployed",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "format",
        "filename",
        "content_hash",
        "download_url",
    }
)
_FILE_CONTRACT = {
    "python": {
        "kind": DELIVERY_ARTIFACT_KINDS["python"],
        "format": "python",
        "filename": "strategy.py",
    },
    "sql": {
        "kind": DELIVERY_ARTIFACT_KINDS["sql"],
        "format": "sql",
        "filename": "strategy.sql",
    },
    "strategy_json": {
        "kind": DELIVERY_ARTIFACT_KINDS["strategy_json"],
        "format": "json",
        "filename": "strategy.json",
    },
    "equivalence_json": {
        "kind": DELIVERY_ARTIFACT_KINDS["equivalence_json"],
        "format": "json",
        "filename": "equivalence.json",
    },
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DELIVERY_ID_RE = re.compile(r"^strategy-delivery-[0-9a-f]{24}$")
_SAFE_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_INPUT_BYTES = 64 * 1024
_BOUNDARY_ERRORS = (
    DatasetContentDriftError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


class StrategyDeliveryToolError(StrategyDeliveryError):
    """The governed delivery request or one of its exact bindings is invalid."""


def run_export_strategy_delivery(inputs, ctx, runtime) -> dict[str, Any]:
    """Publish exact Python/SQL/JSON delivery files plus equivalence evidence."""

    try:
        request = _validate_inputs(inputs)
        task_id = _task_id(ctx.task_id)
        source = _load_exact_sources(
            runtime,
            task_id=task_id,
            request=request,
        )
        frame = runtime.backend.read_frame(source["dataset_path"])
        if sha256_file(source["dataset_path"]) != request["dataset_ref"][
            "expected_content_hash"
        ]:
            raise StrategyDeliveryToolError(
                "dataset no longer matches the exact delivery request"
            )
        spec = source["spec"]
        equivalence = verify_strategy_delivery_equivalence(
            spec,
            frame,
            maximum_rows=request["maximum_equivalence_rows"],
        )
        contents = _delivery_contents(spec=spec, equivalence=equivalence)
        content_hashes = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in contents.items()
        }
        delivery_id = _delivery_id(
            strategy_ref=request["strategy_ref"],
            dataset_ref=request["dataset_ref"],
            maximum_equivalence_rows=request["maximum_equivalence_rows"],
            equivalence=equivalence,
            content_hashes=content_hashes,
        )
        return _publish_delivery(
            runtime,
            task_id=task_id,
            request=request,
            request_hash=_sha256_json(request),
            source=source,
            delivery_id=delivery_id,
            equivalence=equivalence,
            contents=contents,
            content_hashes=content_hashes,
        )
    except StrategyDeliveryToolError:
        raise
    except (StrategyDeliveryError, StrategyError, *_BOUNDARY_ERRORS) as exc:
        raise StrategyDeliveryToolError(str(exc)) from exc


def validate_export_strategy_delivery_tool_output(
    value: object,
    *,
    expected_task_id: str,
    expected_strategy_ref: Mapping[str, Any],
    expected_dataset_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a Tool output against the exact refs held by its caller."""

    obj = _canonical_object(value, "export_strategy_delivery output")
    _exact_fields(obj, _OUTPUT_FIELDS, "export_strategy_delivery output")
    strategy_ref = _strategy_ref(obj["strategy_ref"])
    dataset_ref = _dataset_ref(obj["dataset_ref"])
    trusted_strategy = _strategy_ref(expected_strategy_ref)
    trusted_dataset = _dataset_ref(expected_dataset_ref)
    task_id = _task_id(expected_task_id)
    if strategy_ref != trusted_strategy:
        raise StrategyDeliveryToolError(
            "export_strategy_delivery strategy_ref does not match its "
            "authenticated request"
        )
    if dataset_ref != trusted_dataset:
        raise StrategyDeliveryToolError(
            "export_strategy_delivery dataset_ref does not match its "
            "authenticated request"
        )
    if (
        obj["schema_version"] != DELIVERY_TOOL_SCHEMA_VERSION
        or obj["task_id"] != task_id
        or obj["strategy_id"] != strategy_ref["strategy_id"]
        or obj["strategy_type"] != strategy_ref["expected_strategy_type"]
        or obj["strategy_version"] != strategy_ref["expected_version"]
    ):
        raise StrategyDeliveryToolError(
            "export_strategy_delivery strategy projection drifted"
        )
    maximum_rows = _bounded_rows(obj["maximum_equivalence_rows"])
    equivalence_raw = _canonical_object(
        obj["equivalence"],
        "export_strategy_delivery equivalence",
    )
    equivalence = validate_strategy_delivery_equivalence(
        equivalence_raw,
        expected_strategy_spec_hash=strategy_ref["expected_spec_hash"],
        expected_sample_hash=_hash(
            equivalence_raw.get("sample_hash"),
            "equivalence.sample_hash",
        ),
        expected_content_hash=_hash(
            equivalence_raw.get("content_hash"),
            "equivalence.content_hash",
        ),
    )
    source_row_count = _non_negative_int(
        obj["source_row_count"],
        "source_row_count",
    )
    if source_row_count != equivalence["source_row_count"]:
        raise StrategyDeliveryToolError(
            "export_strategy_delivery source_row_count drifted"
        )
    artifacts = _validate_artifacts(
        obj["artifacts"],
        task_id=task_id,
    )
    content_hashes = {
        name: artifacts[index]["content_hash"]
        for index, name in enumerate(_FILE_CONTRACT)
    }
    expected_id = _delivery_id(
        strategy_ref=strategy_ref,
        dataset_ref=dataset_ref,
        maximum_equivalence_rows=maximum_rows,
        equivalence=equivalence,
        content_hashes=content_hashes,
    )
    if (
        not isinstance(obj["delivery_id"], str)
        or _DELIVERY_ID_RE.fullmatch(obj["delivery_id"]) is None
        or obj["delivery_id"] != expected_id
    ):
        raise StrategyDeliveryToolError(
            "export_strategy_delivery delivery_id drifted"
        )
    for field in ("not_applied", "not_adopted", "not_deployed"):
        if obj[field] is not True:
            raise StrategyDeliveryToolError(
                f"export_strategy_delivery {field} must be true"
            )
    obj["strategy_ref"] = strategy_ref
    obj["dataset_ref"] = dataset_ref
    obj["equivalence"] = equivalence
    obj["artifacts"] = artifacts
    return obj


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _canonical_object(value, "export_strategy_delivery inputs")
    _exact_fields(obj, _INPUT_FIELDS, "export_strategy_delivery inputs")
    if len(_canonical_json(obj).encode("utf-8")) > _MAX_INPUT_BYTES:
        raise StrategyDeliveryToolError(
            "export_strategy_delivery inputs exceed byte budget"
        )
    return {
        "strategy_ref": _strategy_ref(obj["strategy_ref"]),
        "dataset_ref": _dataset_ref(obj["dataset_ref"]),
        "maximum_equivalence_rows": _bounded_rows(
            obj["maximum_equivalence_rows"]
        ),
    }


def _strategy_ref(value: object) -> dict[str, Any]:
    obj = _canonical_object(value, "strategy_ref")
    _exact_fields(obj, _STRATEGY_REF_FIELDS, "strategy_ref")
    strategy_type = _text(
        obj["expected_strategy_type"],
        "strategy_ref.expected_strategy_type",
    )
    if strategy_type not in _STRATEGY_TYPES:
        raise StrategyDeliveryToolError(
            "strategy_ref.expected_strategy_type is invalid"
        )
    return {
        "strategy_id": _text(obj["strategy_id"], "strategy_ref.strategy_id"),
        "expected_strategy_type": strategy_type,
        "expected_version": _positive_int(
            obj["expected_version"],
            "strategy_ref.expected_version",
        ),
        "expected_spec_hash": _hash(
            obj["expected_spec_hash"],
            "strategy_ref.expected_spec_hash",
        ),
    }


def _dataset_ref(value: object) -> dict[str, str]:
    obj = _canonical_object(value, "dataset_ref")
    _exact_fields(obj, _DATASET_REF_FIELDS, "dataset_ref")
    return {
        "dataset_id": _text(obj["dataset_id"], "dataset_ref.dataset_id"),
        "expected_content_hash": _hash(
            obj["expected_content_hash"],
            "dataset_ref.expected_content_hash",
        ),
    }


def _load_exact_sources(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    strategy_ref = request["strategy_ref"]
    strategy = runtime.strategies.get_strategy(strategy_ref["strategy_id"])
    meta = runtime.strategies.get_strategy_meta(strategy_ref["strategy_id"])
    spec_hash = runtime.strategies.get_strategy_spec_hash(
        strategy_ref["strategy_id"]
    )
    if strategy is None or meta is None or spec_hash is None:
        raise StrategyDeliveryToolError(
            "strategy does not match the exact delivery request"
        )
    if (
        meta["task_id"] != task_id
        or meta["strategy_type"] != strategy_ref["expected_strategy_type"]
        or meta["version"] != strategy_ref["expected_version"]
        or spec_hash != strategy_ref["expected_spec_hash"]
    ):
        raise StrategyDeliveryToolError(
            "strategy no longer matches the exact delivery request"
        )
    spec = parse_strategy_spec(
        strategy.spec or legacy_strategy_to_spec(strategy)
    ).to_dict()
    if spec["strategy_type"] != strategy_ref["expected_strategy_type"]:
        raise StrategyDeliveryToolError(
            "strategy no longer matches the exact delivery request"
        )

    dataset_ref = request["dataset_ref"]
    try:
        dataset = runtime.registry.get(dataset_ref["dataset_id"])
    except KeyError as exc:
        raise StrategyDeliveryToolError(
            "dataset does not match the exact delivery request"
        ) from exc
    if (
        str(dataset.task_id) != task_id
        or dataset.content_hash != dataset_ref["expected_content_hash"]
    ):
        raise StrategyDeliveryToolError(
            "dataset no longer matches the exact delivery request"
        )
    try:
        dataset_path = Path(
            runtime.registry.resolve_verified_path(dataset.id)
        )
    except (DatasetContentDriftError, KeyError, OSError, ValueError) as exc:
        raise StrategyDeliveryToolError(
            "dataset no longer matches the exact delivery request"
        ) from exc
    if sha256_file(dataset_path) != dataset_ref["expected_content_hash"]:
        raise StrategyDeliveryToolError(
            "dataset no longer matches the exact delivery request"
        )
    return {
        "spec": spec,
        "dataset_path": dataset_path,
        "dataset_source_path": str(dataset.source_path),
    }


def _delivery_contents(
    *,
    spec: Mapping[str, Any],
    equivalence: Mapping[str, Any],
) -> dict[str, bytes]:
    canonical_spec = parse_strategy_spec(spec)
    return {
        "python": generate_strategy_python_source(spec).encode("utf-8"),
        "sql": generate_strategy_duckdb_sql_source(spec).encode("utf-8"),
        "strategy_json": (
            canonical_strategy_json(
                canonical_spec,
                include_display_metadata=True,
            )
            + "\n"
        ).encode("utf-8"),
        "equivalence_json": (
            _canonical_json(equivalence) + "\n"
        ).encode("utf-8"),
    }


def _delivery_id(
    *,
    strategy_ref: Mapping[str, Any],
    dataset_ref: Mapping[str, Any],
    maximum_equivalence_rows: int,
    equivalence: Mapping[str, Any],
    content_hashes: Mapping[str, str],
) -> str:
    if set(content_hashes) != set(_FILE_CONTRACT):
        raise StrategyDeliveryToolError(
            "strategy delivery file content hashes are incomplete"
        )
    body = {
        "schema_version": DELIVERY_TOOL_SCHEMA_VERSION,
        "producer_version": DELIVERY_PRODUCER_VERSION,
        "strategy_ref": dict(strategy_ref),
        "dataset_ref": dict(dataset_ref),
        "maximum_equivalence_rows": maximum_equivalence_rows,
        "equivalence_ref": {
            "equivalence_id": equivalence["equivalence_id"],
            "content_hash": equivalence["content_hash"],
            "sample_hash": equivalence["sample_hash"],
            "source_row_count": equivalence["source_row_count"],
            "sample_count": equivalence["sample_count"],
        },
        "file_content_hashes": {
            name: _hash(content_hashes[name], f"content_hashes.{name}")
            for name in _FILE_CONTRACT
        },
    }
    return "strategy-delivery-" + _sha256_json(body)[:24]


def _publish_delivery(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    request_hash: str,
    source: Mapping[str, Any],
    delivery_id: str,
    equivalence: Mapping[str, Any],
    contents: Mapping[str, bytes],
    content_hashes: Mapping[str, str],
) -> dict[str, Any]:
    tasks_root = Path(runtime.settings.tasks_dir).resolve()
    output_dir = (
        tasks_root / task_id / "strategy_delivery" / delivery_id
    )
    try:
        output_dir.resolve(strict=False).relative_to(tasks_root)
    except ValueError as exc:
        raise StrategyDeliveryToolError(
            "strategy delivery output path escaped the task root"
        ) from exc
    uow = ArtifactUnitOfWork()
    staged = {
        name: uow.stage_file(
            output_dir,
            contract["filename"],
        )
        for name, contract in _FILE_CONTRACT.items()
    }
    provenance_base = {
        "schema_version": DELIVERY_ARTIFACT_SCHEMA_VERSION,
        "producer_version": DELIVERY_PRODUCER_VERSION,
        "task_id": task_id,
        "delivery_id": delivery_id,
        "strategy_ref": dict(request["strategy_ref"]),
        "dataset_ref": dict(request["dataset_ref"]),
        "maximum_equivalence_rows": request["maximum_equivalence_rows"],
        "equivalence_ref": {
            "equivalence_id": equivalence["equivalence_id"],
            "content_hash": equivalence["content_hash"],
            "sample_hash": equivalence["sample_hash"],
        },
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    try:
        for name, artifact in staged.items():
            artifact.path.write_bytes(contents[name])
            if sha256_file(artifact.path) != content_hashes[name]:
                raise StrategyDeliveryToolError(
                    f"staged strategy delivery {name} hash drifted"
                )
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _revalidate_sources_on_connection(
                conn,
                task_id=task_id,
                request=request,
                source=source,
            )
            uow.promote_all()
            records = []
            for name, contract in _FILE_CONTRACT.items():
                if sha256_file(staged[name].final_path) != content_hashes[name]:
                    raise StrategyDeliveryToolError(
                        f"promoted strategy delivery {name} hash drifted"
                    )
                records.append(
                    runtime.task_artifacts.register_on_connection(
                        conn,
                        task_id=task_id,
                        kind=contract["kind"],
                        path=str(staged[name].final_path),
                        content_hash=content_hashes[name],
                        origin_tool=DELIVERY_ORIGIN_TOOL,
                        provenance={
                            **provenance_base,
                            "format_key": name,
                            "artifact_kind": contract["kind"],
                            "artifact_content_hash": content_hashes[name],
                        },
                    )
                )
            if not _audit_exists(
                conn,
                delivery_id=delivery_id,
                request_hash=request_hash,
            ):
                _write_audit_row(
                    conn,
                    kind=DELIVERY_AUDIT_KIND,
                    target_ref=delivery_id,
                    inputs_hash=request_hash,
                    outcome="succeeded",
                    detail={
                        "task_id": task_id,
                        "strategy_ref": dict(request["strategy_ref"]),
                        "dataset_ref": dict(request["dataset_ref"]),
                        "equivalence_id": equivalence["equivalence_id"],
                        "artifact_ids": [record["id"] for record in records],
                        "not_applied": True,
                        "not_adopted": True,
                        "not_deployed": True,
                    },
                )
        uow.commit()
    except Exception:
        uow.rollback()
        raise

    output = {
        "schema_version": DELIVERY_TOOL_SCHEMA_VERSION,
        "delivery_id": delivery_id,
        "task_id": task_id,
        "strategy_id": request["strategy_ref"]["strategy_id"],
        "strategy_type": request["strategy_ref"]["expected_strategy_type"],
        "strategy_version": request["strategy_ref"]["expected_version"],
        "strategy_ref": dict(request["strategy_ref"]),
        "dataset_ref": dict(request["dataset_ref"]),
        "source_row_count": equivalence["source_row_count"],
        "maximum_equivalence_rows": request["maximum_equivalence_rows"],
        "equivalence": dict(equivalence),
        "artifacts": [
            _artifact_output(
                task_id=task_id,
                name=name,
                record=record,
            )
            for name, record in zip(
                _FILE_CONTRACT,
                records,
                strict=True,
            )
        ],
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    return validate_export_strategy_delivery_tool_output(
        output,
        expected_task_id=task_id,
        expected_strategy_ref=request["strategy_ref"],
        expected_dataset_ref=request["dataset_ref"],
    )


def _revalidate_sources_on_connection(
    conn,
    *,
    task_id: str,
    request: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    strategy_ref = request["strategy_ref"]
    row = conn.execute(
        """
        SELECT id, task_id, strategy_type, version, rules_json, score_col,
               default_decision_json, description, dsl_json,
               dsl_schema_version, dsl_content_hash
          FROM strategies
         WHERE id = ?
        """,
        (strategy_ref["strategy_id"],),
    ).fetchone()
    if row is None:
        raise StrategyDeliveryToolError(
            "strategy no longer matches the exact delivery request"
        )
    try:
        spec_hash = _strategy_spec_hash_from_row(row)
    except (StrategyError, TypeError, ValueError) as exc:
        raise StrategyDeliveryToolError(
            "strategy no longer matches the exact delivery request"
        ) from exc
    if (
        str(row["task_id"]) != task_id
        or str(row["strategy_type"])
        != strategy_ref["expected_strategy_type"]
        or int(row["version"]) != strategy_ref["expected_version"]
        or spec_hash != strategy_ref["expected_spec_hash"]
    ):
        raise StrategyDeliveryToolError(
            "strategy no longer matches the exact delivery request"
        )

    dataset_ref = request["dataset_ref"]
    dataset = conn.execute(
        """
        SELECT id, task_id, source_path, content_hash
          FROM datasets
         WHERE id = ?
        """,
        (dataset_ref["dataset_id"],),
    ).fetchone()
    if (
        dataset is None
        or str(dataset["task_id"]) != task_id
        or str(dataset["source_path"]) != source["dataset_source_path"]
        or str(dataset["content_hash"])
        != dataset_ref["expected_content_hash"]
        or sha256_file(source["dataset_path"])
        != dataset_ref["expected_content_hash"]
    ):
        raise StrategyDeliveryToolError(
            "dataset no longer matches the exact delivery request"
        )


def _artifact_output(
    *,
    task_id: str,
    name: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _FILE_CONTRACT[name]
    artifact_id = _hash(record["id"], f"artifacts.{name}.artifact_id")
    return {
        "artifact_id": artifact_id,
        "kind": contract["kind"],
        "format": contract["format"],
        "filename": contract["filename"],
        "content_hash": _hash(
            record["content_hash"],
            f"artifacts.{name}.content_hash",
        ),
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}/task-artifacts/"
            f"{quote(artifact_id, safe='')}/download"
        ),
    }


def _validate_artifacts(value: object, *, task_id: str | None) -> list[dict]:
    if not isinstance(value, list) or len(value) != len(_FILE_CONTRACT):
        raise StrategyDeliveryToolError(
            "export_strategy_delivery artifacts are invalid"
        )
    normalized: list[dict] = []
    for index, (name, contract) in enumerate(_FILE_CONTRACT.items()):
        artifact = _canonical_object(
            value[index],
            f"artifacts[{index}]",
        )
        _exact_fields(artifact, _ARTIFACT_FIELDS, f"artifacts[{index}]")
        artifact_id = _hash(
            artifact["artifact_id"],
            f"artifacts[{index}].artifact_id",
        )
        expected = {
            "artifact_id": artifact_id,
            "kind": contract["kind"],
            "format": contract["format"],
            "filename": contract["filename"],
            "content_hash": _hash(
                artifact["content_hash"],
                f"artifacts[{index}].content_hash",
            ),
            "download_url": artifact["download_url"],
        }
        if task_id is not None:
            expected["download_url"] = (
                f"/api/tasks/{quote(task_id, safe='')}/task-artifacts/"
                f"{quote(artifact_id, safe='')}/download"
            )
        if artifact != expected:
            raise StrategyDeliveryToolError(
                f"export_strategy_delivery artifacts[{index}] drifted"
            )
        if not isinstance(artifact["download_url"], str) or not artifact[
            "download_url"
        ].endswith(f"/{artifact_id}/download"):
            raise StrategyDeliveryToolError(
                f"export_strategy_delivery artifacts[{index}] download_url "
                "is invalid"
            )
        normalized.append(expected)
    return normalized


def _audit_exists(conn, *, delivery_id: str, request_hash: str) -> bool:
    return (
        conn.execute(
            """
            SELECT 1
              FROM audit
             WHERE kind = ? AND target_ref = ? AND inputs_hash = ?
             LIMIT 1
            """,
            (DELIVERY_AUDIT_KIND, delivery_id, request_hash),
        ).fetchone()
        is not None
    )


def _bounded_rows(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAX_EQUIVALENCE_ROWS
    ):
        raise StrategyDeliveryToolError(
            f"maximum_equivalence_rows must be between 1 and "
            f"{MAX_EQUIVALENCE_ROWS}"
        )
    return value


def _task_id(value: object) -> str:
    normalized = _text(value, "task_id")
    if _SAFE_TASK_ID_RE.fullmatch(normalized) is None:
        raise StrategyDeliveryToolError("task_id is not safe for artifact paths")
    return normalized


def _canonical_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyDeliveryToolError(f"{name} must be an object")
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyDeliveryToolError(
            f"{name} must contain canonical JSON values"
        ) from exc
    if not isinstance(normalized, dict):
        raise StrategyDeliveryToolError(f"{name} must be an object")
    return normalized


def _exact_fields(value: Mapping[str, Any], expected: set | frozenset, name: str) -> None:
    if set(value) != set(expected):
        raise StrategyDeliveryToolError(f"{name} fields are invalid")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StrategyDeliveryToolError(f"{name} must be canonical text")
    if "\x00" in value:
        raise StrategyDeliveryToolError(f"{name} must not contain NUL")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrategyDeliveryToolError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StrategyDeliveryToolError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyDeliveryToolError(
            f"{name} must be a non-negative integer"
        )
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "DELIVERY_ARTIFACT_KINDS",
    "DELIVERY_AUDIT_KIND",
    "DELIVERY_ORIGIN_TOOL",
    "DELIVERY_TOOL_SCHEMA_VERSION",
    "StrategyDeliveryToolError",
    "run_export_strategy_delivery",
    "validate_export_strategy_delivery_tool_output",
]
