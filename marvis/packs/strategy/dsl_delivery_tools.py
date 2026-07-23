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
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from urllib.parse import quote

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
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
_MAX_DELIVERY_ARTIFACT_BYTES = 64 * 1024 * 1024
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
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
        frame = _read_authenticated_parquet_snapshot(
            source["dataset_path"],
            root=source["dataset_root"],
            expected_content_hash=request["dataset_ref"][
                "expected_content_hash"
            ],
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
    expected_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a Tool output against the exact refs held by its caller."""

    obj = _canonical_object(value, "export_strategy_delivery output")
    _exact_fields(obj, _OUTPUT_FIELDS, "export_strategy_delivery output")
    strategy_ref = _strategy_ref(obj["strategy_ref"])
    dataset_ref = _dataset_ref(obj["dataset_ref"])
    trusted_strategy = _strategy_ref(expected_strategy_ref)
    trusted_dataset = _dataset_ref(expected_dataset_ref)
    trusted_artifacts = _artifact_projections(expected_artifacts)
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
    if equivalence["sample_count"] > maximum_rows:
        raise StrategyDeliveryToolError(
            "export_strategy_delivery equivalence sample_count exceeds its "
            "declared budget"
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
    for index, name in enumerate(_FILE_CONTRACT):
        artifact_projection = {
            "artifact_id": artifacts[index]["artifact_id"],
            "content_hash": artifacts[index]["content_hash"],
        }
        if artifact_projection != trusted_artifacts[name]:
            raise StrategyDeliveryToolError(
                f"export_strategy_delivery artifacts[{index}] {name} "
                "does not match its authenticated publication"
            )
    content_hashes = {
        name: artifacts[index]["content_hash"]
        for index, name in enumerate(_FILE_CONTRACT)
    }
    expected_equivalence_artifact_hash = hashlib.sha256(
        (_canonical_json(equivalence) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        content_hashes["equivalence_json"]
        != expected_equivalence_artifact_hash
    ):
        raise StrategyDeliveryToolError(
            "export_strategy_delivery equivalence artifact content does not "
            "match its canonical document bytes"
        )
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
    return {
        "spec": spec,
        "dataset_path": dataset_path,
        "dataset_root": Path(runtime.settings.datasets_dir).absolute(),
        "dataset_source_path": str(dataset.source_path),
    }


def _read_authenticated_parquet_snapshot(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> pd.DataFrame:
    """Read delivery rows only from one hash-authenticated private snapshot."""

    source_fd = -1
    snapshot = None
    try:
        resolved_root = root.resolve(strict=True)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve(strict=True).is_relative_to(resolved_root)
        ):
            raise StrategyDeliveryToolError(
                "dataset path escaped governed dataset storage"
            )
        before = os.lstat(path)
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
            raise StrategyDeliveryToolError(
                "dataset changed while opening the delivery snapshot"
            )

        snapshot = tempfile.TemporaryFile(mode="w+b", dir=resolved_root)
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
        if (
            _stable_file_stat(os.fstat(source_fd))
            != _stable_file_stat(opened)
            or copied != int(opened.st_size)
            or not hmac.compare_digest(
                digest.hexdigest(),
                expected_content_hash,
            )
        ):
            raise StrategyDeliveryToolError(
                "dataset bytes changed before delivery reconciliation"
            )

        snapshot_stat = os.fstat(snapshot.fileno())
        if int(snapshot_stat.st_size) != copied:
            raise StrategyDeliveryToolError(
                "private delivery dataset snapshot is incomplete"
            )
        snapshot.seek(0)
        frame = pd.read_parquet(snapshot)
        current = os.lstat(path)
        if (
            _stable_file_stat(os.fstat(snapshot.fileno()))
            != _stable_file_stat(snapshot_stat)
            or _stable_file_stat(os.fstat(source_fd))
            != _stable_file_stat(opened)
            or stat.S_ISLNK(current.st_mode)
            or _stable_file_stat(current) != _stable_file_stat(opened)
        ):
            raise StrategyDeliveryToolError(
                "dataset changed during delivery reconciliation"
            )
        return frame
    except StrategyDeliveryToolError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StrategyDeliveryToolError(
            "dataset could not be read for delivery reconciliation"
        ) from exc
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
    reused = False
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
            reused = _prepare_delivery_outputs_under_lock(
                conn,
                uow=uow,
                task_id=task_id,
                tasks_root=tasks_root,
                staged=staged,
                contents=contents,
                content_hashes=content_hashes,
                provenance_base=provenance_base,
            )
            records = []
            for name, contract in _FILE_CONTRACT.items():
                _require_exact_delivery_file(
                    staged[name].final_path,
                    root=tasks_root,
                    expected=contents[name],
                    expected_hash=content_hashes[name],
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
            _write_or_require_delivery_audit(
                conn,
                task_id=task_id,
                delivery_id=delivery_id,
                request_hash=request_hash,
                request=request,
                equivalence=equivalence,
                records=records,
            )
            for name in _FILE_CONTRACT:
                _require_exact_delivery_file(
                    staged[name].final_path,
                    root=tasks_root,
                    expected=contents[name],
                    expected_hash=content_hashes[name],
                )
        if not reused:
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
        expected_artifacts={
            name: {
                "artifact_id": record["id"],
                "content_hash": record["content_hash"],
            }
            for name, record in zip(
                _FILE_CONTRACT,
                records,
                strict=True,
            )
        },
    )


def _prepare_delivery_outputs_under_lock(
    conn,
    *,
    uow: ArtifactUnitOfWork,
    task_id: str,
    tasks_root: Path,
    staged: Mapping[str, Any],
    contents: Mapping[str, bytes],
    content_hashes: Mapping[str, str],
    provenance_base: Mapping[str, Any],
) -> bool:
    rows = {}
    for name, contract in _FILE_CONTRACT.items():
        rows[name] = conn.execute(
            """
            SELECT id, task_id, kind, path, content_hash, origin_tool,
                   provenance_json, created_at
              FROM task_artifacts
             WHERE task_id = ? AND kind = ? AND path = ?
            """,
            (
                task_id,
                contract["kind"],
                str(staged[name].final_path),
            ),
        ).fetchone()
    registered = [name for name, row in rows.items() if row is not None]
    if registered and len(registered) != len(_FILE_CONTRACT):
        raise StrategyDeliveryToolError(
            "existing strategy delivery registry set is incomplete"
        )

    if registered:
        for name, contract in _FILE_CONTRACT.items():
            provenance = {
                **dict(provenance_base),
                "format_key": name,
                "artifact_kind": contract["kind"],
                "artifact_content_hash": content_hashes[name],
            }
            _require_existing_delivery_row(
                rows[name],
                task_id=task_id,
                kind=contract["kind"],
                path=staged[name].final_path,
                content_hash=content_hashes[name],
                provenance=provenance,
            )
            _require_exact_delivery_file(
                staged[name].final_path,
                root=tasks_root,
                expected=contents[name],
                expected_hash=content_hashes[name],
            )
        uow.rollback()
        return True

    existing_files = [
        name
        for name in _FILE_CONTRACT
        if (
            staged[name].final_path.exists()
            or staged[name].final_path.is_symlink()
        )
    ]
    if existing_files:
        for name in existing_files:
            _require_exact_delivery_file(
                staged[name].final_path,
                root=tasks_root,
                expected=contents[name],
                expected_hash=content_hashes[name],
            )
        if len(existing_files) == len(_FILE_CONTRACT):
            uow.rollback()
            return True

    uow.promote_all()
    for name in _FILE_CONTRACT:
        _require_exact_delivery_file(
            staged[name].final_path,
            root=tasks_root,
            expected=contents[name],
            expected_hash=content_hashes[name],
        )
    return False


def _require_existing_delivery_row(
    row,
    *,
    task_id: str,
    kind: str,
    path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    expected_id = _stable_task_artifact_id(
        task_id=task_id,
        kind=kind,
        path=str(path),
    )
    expected = {
        "id": expected_id,
        "task_id": task_id,
        "kind": kind,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": DELIVERY_ORIGIN_TOOL,
        "provenance_json": _canonical_json(provenance),
    }
    if row is None or any(str(row[field]) != value for field, value in expected.items()):
        raise StrategyDeliveryToolError(
            "existing strategy delivery registry row changed"
        )


def _stable_task_artifact_id(
    *,
    task_id: str,
    kind: str,
    path: str,
) -> str:
    identity = json.dumps(
        [task_id, kind, path],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        f"marvis.task_artifact.v1:{identity}".encode("utf-8")
    ).hexdigest()


def _require_exact_delivery_file(
    path: Path,
    *,
    root: Path,
    expected: bytes,
    expected_hash: str,
) -> None:
    descriptor = -1
    try:
        if not path.is_absolute() or path.is_symlink():
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact must be a regular file"
            )
        resolved_root = root.resolve(strict=True)
        path.resolve(strict=True).relative_to(resolved_root)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_size) < 0
            or int(before.st_size) > _MAX_DELIVERY_ARTIFACT_BYTES
        ):
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact must be a bounded "
                "regular file"
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_DELIVERY_ARTIFACT_BYTES:
                raise StrategyDeliveryToolError(
                    "existing strategy delivery artifact exceeds byte budget"
                )
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            _stable_file_stat(after) != _stable_file_stat(before)
            or stat.S_ISLNK(current.st_mode)
            or _file_identity(current) != _file_identity(before)
            or total != int(before.st_size)
            or not hmac.compare_digest(digest.hexdigest(), expected_hash)
            or b"".join(chunks) != expected
        ):
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact bytes changed"
            )
    except StrategyDeliveryToolError:
        raise
    except (OSError, ValueError) as exc:
        raise StrategyDeliveryToolError(
            "existing strategy delivery artifact is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_authenticated_file_hash(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
) -> None:
    """Authenticate one live regular file without following its leaf path."""

    descriptor = -1
    try:
        resolved_root = root.resolve(strict=True)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(resolved_root)
        ):
            raise StrategyDeliveryToolError(
                "dataset path escaped governed dataset storage"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or _file_identity(current) != _file_identity(before)
        ):
            raise StrategyDeliveryToolError(
                "dataset must remain a regular governed file"
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            _stable_file_stat(after) != _stable_file_stat(before)
            or stat.S_ISLNK(current.st_mode)
            or _file_identity(current) != _file_identity(before)
            or total != int(before.st_size)
            or not hmac.compare_digest(digest.hexdigest(), expected_hash)
        ):
            raise StrategyDeliveryToolError(
                "dataset bytes changed during delivery publication"
            )
    except StrategyDeliveryToolError:
        raise
    except (OSError, ValueError) as exc:
        raise StrategyDeliveryToolError(
            "dataset is unavailable for delivery publication"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


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
    ):
        raise StrategyDeliveryToolError(
            "dataset no longer matches the exact delivery request"
        )
    _require_authenticated_file_hash(
        source["dataset_path"],
        root=source["dataset_root"],
        expected_hash=dataset_ref["expected_content_hash"],
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
            "?expected_content_hash="
            f"{quote(str(record['content_hash']), safe='')}"
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
                "?expected_content_hash="
                f"{quote(expected['content_hash'], safe='')}"
            )
        if artifact != expected:
            raise StrategyDeliveryToolError(
                f"export_strategy_delivery artifacts[{index}] drifted"
            )
        normalized.append(expected)
    return normalized


def _artifact_projections(
    value: object,
) -> dict[str, dict[str, str]]:
    obj = _canonical_object(value, "expected_artifacts")
    _exact_fields(
        obj,
        frozenset(_FILE_CONTRACT),
        "expected_artifacts",
    )
    projections: dict[str, dict[str, str]] = {}
    for name in _FILE_CONTRACT:
        projection = _canonical_object(
            obj[name],
            f"expected_artifacts.{name}",
        )
        _exact_fields(
            projection,
            frozenset({"artifact_id", "content_hash"}),
            f"expected_artifacts.{name}",
        )
        projections[name] = {
            "artifact_id": _hash(
                projection["artifact_id"],
                f"expected_artifacts.{name}.artifact_id",
            ),
            "content_hash": _hash(
                projection["content_hash"],
                f"expected_artifacts.{name}.content_hash",
            ),
        }
    return projections


def _write_or_require_delivery_audit(
    conn,
    *,
    task_id: str,
    delivery_id: str,
    request_hash: str,
    request: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> None:
    detail = {
        "task_id": task_id,
        "strategy_ref": dict(request["strategy_ref"]),
        "dataset_ref": dict(request["dataset_ref"]),
        "equivalence_id": equivalence["equivalence_id"],
        "artifact_ids": [record["id"] for record in records],
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    rows = conn.execute(
        """
        SELECT kind, target_ref, inputs_hash, actor, outcome, detail_json
          FROM audit
         WHERE target_ref = ?
            OR inputs_hash = ?
         ORDER BY at, id
        """,
        (delivery_id, request_hash),
    ).fetchall()
    if not rows:
        _write_audit_row(
            conn,
            kind=DELIVERY_AUDIT_KIND,
            target_ref=delivery_id,
            inputs_hash=request_hash,
            outcome="succeeded",
            detail=detail,
        )
        return
    try:
        persisted_detail = (
            json.loads(str(rows[0]["detail_json"]))
            if len(rows) == 1
            else None
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        persisted_detail = None
    if (
        len(rows) != 1
        or str(rows[0]["kind"]) != DELIVERY_AUDIT_KIND
        or str(rows[0]["target_ref"]) != delivery_id
        or str(rows[0]["inputs_hash"]) != request_hash
        or str(rows[0]["actor"]) != "system"
        or str(rows[0]["outcome"]) != "succeeded"
        or persisted_detail != detail
    ):
        raise StrategyDeliveryToolError(
            "existing strategy delivery audit changed"
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
