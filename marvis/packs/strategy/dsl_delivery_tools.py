"""Governed, downloadable delivery bundle for canonical Strategy DSL.

The Tool binds one immutable strategy definition and one task-owned dataset,
generates standalone Python, DuckDB SQL, and canonical JSON, then reconciles a
bounded row sample across the MARVIS evaluator and both generated engines.
Files, TaskArtifact rows, and the audit record share one SQLite writer
transaction and one rollback-capable filesystem unit of work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
import uuid

import pandas as pd

from marvis.data.errors import DatasetContentDriftError
from marvis.data.workspace import (
    DataSemanticMapping,
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.packs.strategy.dsl import (
    canonical_strategy_json,
    parse_strategy_spec,
    strategy_spec_hash,
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
from marvis.packs.strategy.materialized_runtime_requirements import (
    hydrate_materialized_strategy_runtime_requirements,
    load_materialized_strategy_runtime_requirements,
    materialized_runtime_requirements_provenance,
    require_materialized_strategy_runtime_requirements_on_connection,
    validate_materialized_runtime_requirements_provenance,
)
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
    {
        "strategy_ref",
        "dataset_ref",
        "workspace_ref",
        "maximum_equivalence_rows",
    }
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
_WORKSPACE_REF_FIELDS = frozenset(
    {
        "revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "active_dataset_id",
        "active_dataset_content_hash",
    }
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
        "workspace_ref",
        "source_row_count",
        "maximum_equivalence_rows",
        "equivalence",
        "artifacts",
        "not_applied",
        "not_adopted",
        "not_deployed",
    }
)
_OPTIONAL_OUTPUT_FIELDS = frozenset({"runtime_requirements"})
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
_ARTIFACT_RECORD_FIELDS = frozenset(
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
_ARTIFACT_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "delivery_id",
        "strategy_ref",
        "dataset_ref",
        "workspace_ref",
        "maximum_equivalence_rows",
        "equivalence_ref",
        "not_applied",
        "not_adopted",
        "not_deployed",
        "format_key",
        "artifact_kind",
        "artifact_content_hash",
    }
)
_OPTIONAL_ARTIFACT_PROVENANCE_FIELDS = frozenset(
    {"runtime_requirements"}
)
_UNSPECIFIED_RUNTIME_REQUIREMENTS = object()
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
    DatasetContentDriftError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


class StrategyDeliveryToolError(StrategyDeliveryError):
    """The governed delivery request or one of its exact bindings is invalid."""


@dataclass
class _SecureStagedDeliveryFile:
    """One delivery file staged and promoted only through held directory fds."""

    stage_name: str
    final_name: str
    stage_path: Path
    final_path: Path
    stage_identity: tuple[int, int]
    promoted_identity: tuple[int, int] | None = None
    final_created: bool = False
    discarded: bool = False

    @property
    def path(self) -> Path:
        return self.stage_path


class _SecureDeliveryUnitOfWork:
    """Directory-fd-relative delivery publication with rollback.

    The task, strategy-delivery, output, and staging directories stay open for
    the whole filesystem/SQLite boundary. Every create, read, link, unlink,
    and rollback is relative to those descriptors, so replacing any named
    parent after validation cannot redirect publication outside the directory
    chain that was authenticated.
    """

    def __init__(
        self,
        tasks_root: Path,
        *,
        task_id: str,
        delivery_id: str,
    ) -> None:
        self.tasks_root = Path(tasks_root).absolute()
        self.task_id = task_id
        self.delivery_id = delivery_id
        self.output_dir = (
            self.tasks_root
            / task_id
            / "strategy_delivery"
            / delivery_id
        )
        self._root_fd = -1
        self._task_fd = -1
        self._delivery_root_fd = -1
        self._output_fd = -1
        self._staging_fd = -1
        self._root_identity: tuple[int, int] | None = None
        self._task_identity: tuple[int, int] | None = None
        self._delivery_root_identity: tuple[int, int] | None = None
        self._output_identity: tuple[int, int] | None = None
        self._staging_identity: tuple[int, int] | None = None
        self._items: list[_SecureStagedDeliveryFile] = []
        self._committed = False
        self._closed = False
        try:
            self._open_chain()
        except Exception:
            self.close()
            raise

    def _open_chain(self) -> None:
        if (
            Path(self.task_id).name != self.task_id
            or self.task_id in {".", ".."}
            or _DELIVERY_ID_RE.fullmatch(self.delivery_id) is None
        ):
            raise StrategyDeliveryToolError(
                "strategy delivery output identity is unsafe"
            )
        try:
            self.tasks_root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "strategy delivery task root is unavailable"
            ) from exc
        self._root_fd, self._root_identity = _open_root_delivery_directory(
            self.tasks_root,
            "strategy delivery task root",
        )
        self._task_fd, self._task_identity = _open_or_create_delivery_directory(
            self._root_fd,
            self.task_id,
            "strategy delivery task directory",
        )
        (
            self._delivery_root_fd,
            self._delivery_root_identity,
        ) = _open_or_create_delivery_directory(
            self._task_fd,
            "strategy_delivery",
            "strategy delivery root",
        )
        self._output_fd, self._output_identity = (
            _open_or_create_delivery_directory(
                self._delivery_root_fd,
                self.delivery_id,
                "strategy delivery output directory",
            )
        )
        self._staging_fd, self._staging_identity = (
            _open_or_create_delivery_directory(
                self._output_fd,
                ".staging",
                "strategy delivery staging directory",
            )
        )
        self.assert_attached(include_staging=True)

    def assert_attached(self, *, include_staging: bool = False) -> None:
        if self._closed:
            raise StrategyDeliveryToolError(
                "strategy delivery directory chain is closed"
            )
        try:
            root_now = os.lstat(self.tasks_root)
            _require_directory_identity(
                root_now,
                self._root_identity,
                "strategy delivery task root",
            )
            _require_named_directory_identity(
                self._root_fd,
                self.task_id,
                self._task_identity,
                "strategy delivery task directory",
            )
            _require_named_directory_identity(
                self._task_fd,
                "strategy_delivery",
                self._delivery_root_identity,
                "strategy delivery root",
            )
            _require_named_directory_identity(
                self._delivery_root_fd,
                self.delivery_id,
                self._output_identity,
                "strategy delivery output directory",
            )
            if include_staging:
                _require_named_directory_identity(
                    self._output_fd,
                    ".staging",
                    self._staging_identity,
                    "strategy delivery staging directory",
                )
        except StrategyDeliveryToolError:
            raise
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "strategy delivery directory chain changed during publication"
            ) from exc

    def stage_file(
        self,
        final_name: str,
    ) -> _SecureStagedDeliveryFile:
        if Path(final_name).name != final_name or final_name in {".", ".."}:
            raise StrategyDeliveryToolError(
                "strategy delivery filename is unsafe"
            )
        self.assert_attached(include_staging=True)
        token = uuid.uuid4().hex
        stage_name = f"{Path(final_name).stem}.{token}{Path(final_name).suffix}"
        descriptor = -1
        try:
            descriptor = os.open(
                stage_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self._staging_fd,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise StrategyDeliveryToolError(
                    "strategy delivery staging target is not a regular file"
                )
            identity = _directory_entry_identity(metadata)
        except StrategyDeliveryToolError:
            raise
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "strategy delivery artifact could not be staged safely"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        artifact = _SecureStagedDeliveryFile(
            stage_name=stage_name,
            final_name=final_name,
            stage_path=self.output_dir / ".staging" / stage_name,
            final_path=self.output_dir / final_name,
            stage_identity=identity,
        )
        self._items.append(artifact)
        self.assert_attached(include_staging=True)
        return artifact

    def write_stage(
        self,
        artifact: _SecureStagedDeliveryFile,
        payload: bytes,
    ) -> None:
        self.assert_attached(include_staging=True)
        descriptor = -1
        try:
            descriptor = os.open(
                artifact.stage_name,
                os.O_WRONLY
                | os.O_TRUNC
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._staging_fd,
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _directory_entry_identity(metadata)
                != artifact.stage_identity
            ):
                raise StrategyDeliveryToolError(
                    "strategy delivery staging target changed"
                )
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("short strategy delivery artifact write")
                written += count
        except StrategyDeliveryToolError:
            raise
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "strategy delivery artifact could not be staged safely"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        _require_exact_delivery_entry(
            self._staging_fd,
            artifact.stage_name,
            expected=payload,
            expected_hash=hashlib.sha256(payload).hexdigest(),
            expected_identity=artifact.stage_identity,
        )
        self.assert_attached(include_staging=True)

    def final_exists(self, artifact: _SecureStagedDeliveryFile) -> bool:
        self.assert_attached()
        try:
            os.stat(
                artifact.final_name,
                dir_fd=self._output_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact is unavailable"
            ) from exc
        return True

    def require_final(
        self,
        artifact: _SecureStagedDeliveryFile,
        *,
        expected: bytes,
        expected_hash: str,
    ) -> None:
        self.assert_attached()
        _require_exact_delivery_entry(
            self._output_fd,
            artifact.final_name,
            expected=expected,
            expected_hash=expected_hash,
        )
        self.assert_attached()

    def promote_selected(
        self,
        *,
        names: list[str],
        contents: Mapping[str, bytes],
        content_hashes: Mapping[str, str],
        artifacts: Mapping[str, _SecureStagedDeliveryFile],
    ) -> None:
        for name in names:
            artifact = artifacts[name]
            self._promote_one(artifact)
            self.require_final(
                artifact,
                expected=contents[name],
                expected_hash=content_hashes[name],
            )

    def _promote_one(
        self,
        artifact: _SecureStagedDeliveryFile,
    ) -> None:
        try:
            self.assert_attached(include_staging=True)
            _require_regular_delivery_entry_identity(
                self._staging_fd,
                artifact.stage_name,
                artifact.stage_identity,
                "strategy delivery staging target",
            )
            if self.final_exists(artifact):
                raise StrategyDeliveryToolError(
                    "strategy delivery final appeared during no-overwrite "
                    "publication"
                )
            os.link(
                artifact.stage_name,
                artifact.final_name,
                src_dir_fd=self._staging_fd,
                dst_dir_fd=self._output_fd,
                follow_symlinks=False,
            )
            # Record the no-clobber link before any fallible unlink, stat,
            # identity, or attachment check so rollback owns this final.
            artifact.final_created = True
            artifact.promoted_identity = artifact.stage_identity
            _unlink_known_delivery_entry(
                self._staging_fd,
                artifact.stage_name,
                artifact.stage_identity,
            )
            artifact.discarded = True
            _require_promoted_delivery_entry(
                self._output_fd,
                artifact.final_name,
                artifact.stage_identity,
            )
            self.assert_attached(include_staging=True)
        except StrategyDeliveryToolError:
            raise
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "strategy delivery artifact promotion failed safely"
            ) from exc

    def discard_stage(self, artifact: _SecureStagedDeliveryFile) -> None:
        if artifact.final_created or artifact.discarded:
            return
        _unlink_known_delivery_entry(
            self._staging_fd,
            artifact.stage_name,
            artifact.stage_identity,
        )
        artifact.discarded = True

    def discard_staged(self) -> None:
        for artifact in self._items:
            self.discard_stage(artifact)

    def commit(self) -> None:
        """Make rollback impossible after SQLite commits the publication."""
        self._committed = True

    def rollback(self) -> None:
        if self._committed:
            return
        for artifact in reversed(self._items):
            try:
                if artifact.final_created:
                    if artifact.promoted_identity is not None:
                        _unlink_known_delivery_entry(
                            self._output_fd,
                            artifact.final_name,
                            artifact.promoted_identity,
                        )
                if not artifact.discarded:
                    _unlink_known_delivery_entry(
                        self._staging_fd,
                        artifact.stage_name,
                        artifact.stage_identity,
                    )
            except (OSError, StrategyDeliveryToolError):
                # Preserve the original publication error. Unknown/replaced
                # entries are deliberately not deleted during compensation.
                continue

    def close(self) -> None:
        if self._closed:
            return
        for descriptor in (
            self._staging_fd,
            self._output_fd,
            self._delivery_root_fd,
            self._task_fd,
            self._root_fd,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        self._closed = True


class _PathSafeDeliveryUnitOfWork:
    """Windows-compatible publication when Python lacks dir-fd operations.

    The Windows stdlib cannot hold an open directory chain for relative
    link/unlink. This backend therefore rejects symlinks, junctions, and
    other reparse points; records each directory/file identity; and rechecks
    containment plus identity before and after every filesystem mutation.
    """

    def __init__(
        self,
        tasks_root: Path,
        *,
        task_id: str,
        delivery_id: str,
    ) -> None:
        if (
            Path(task_id).name != task_id
            or task_id in {".", ".."}
            or _DELIVERY_ID_RE.fullmatch(delivery_id) is None
        ):
            raise StrategyDeliveryToolError(
                "strategy delivery output identity is unsafe"
            )
        self.tasks_root = Path(tasks_root).absolute()
        self.task_id = task_id
        self.delivery_id = delivery_id
        self.task_dir = self.tasks_root / task_id
        self.delivery_root = self.task_dir / "strategy_delivery"
        self.output_dir = self.delivery_root / delivery_id
        self.staging_dir = self.output_dir / ".staging"
        self._root_identity: tuple[int, int] | None = None
        self._task_identity: tuple[int, int] | None = None
        self._delivery_root_identity: tuple[int, int] | None = None
        self._output_identity: tuple[int, int] | None = None
        self._staging_identity: tuple[int, int] | None = None
        self._items: list[_SecureStagedDeliveryFile] = []
        self._committed = False
        self._closed = False
        self._open_chain()

    def _open_chain(self) -> None:
        self._root_identity = _open_or_create_delivery_path_directory(
            self.tasks_root,
            "strategy delivery task root",
        )
        resolved_root = _require_delivery_path_directory_identity(
            self.tasks_root,
            self._root_identity,
            "strategy delivery task root",
        )
        self._task_identity = _open_or_create_delivery_path_directory(
            self.task_dir,
            "strategy delivery task directory",
        )
        _require_delivery_child_directory(
            self.task_dir,
            self._task_identity,
            "strategy delivery task directory",
            resolved_root=resolved_root,
        )
        self._delivery_root_identity = (
            _open_or_create_delivery_path_directory(
                self.delivery_root,
                "strategy delivery root",
            )
        )
        _require_delivery_child_directory(
            self.delivery_root,
            self._delivery_root_identity,
            "strategy delivery root",
            resolved_root=resolved_root,
        )
        self._output_identity = _open_or_create_delivery_path_directory(
            self.output_dir,
            "strategy delivery output directory",
        )
        _require_delivery_child_directory(
            self.output_dir,
            self._output_identity,
            "strategy delivery output directory",
            resolved_root=resolved_root,
        )
        self._staging_identity = _open_or_create_delivery_path_directory(
            self.staging_dir,
            "strategy delivery staging directory",
        )
        _require_delivery_child_directory(
            self.staging_dir,
            self._staging_identity,
            "strategy delivery staging directory",
            resolved_root=resolved_root,
        )
        self.assert_attached(include_staging=True)

    def assert_attached(self, *, include_staging: bool = False) -> None:
        if self._closed:
            raise StrategyDeliveryToolError(
                "strategy delivery directory chain is closed"
            )
        resolved_root = _require_delivery_path_directory_identity(
            self.tasks_root,
            self._root_identity,
            "strategy delivery task root",
        )
        for path, identity, label in (
            (
                self.task_dir,
                self._task_identity,
                "strategy delivery task directory",
            ),
            (
                self.delivery_root,
                self._delivery_root_identity,
                "strategy delivery root",
            ),
            (
                self.output_dir,
                self._output_identity,
                "strategy delivery output directory",
            ),
        ):
            resolved = _require_delivery_path_directory_identity(
                path,
                identity,
                label,
            )
            try:
                resolved.relative_to(resolved_root)
            except ValueError as exc:
                raise StrategyDeliveryToolError(
                    f"{label} escaped its authenticated task root"
                ) from exc
        if include_staging:
            resolved_staging = _require_delivery_path_directory_identity(
                self.staging_dir,
                self._staging_identity,
                "strategy delivery staging directory",
            )
            try:
                resolved_staging.relative_to(resolved_root)
            except ValueError as exc:
                raise StrategyDeliveryToolError(
                    "strategy delivery staging directory escaped its "
                    "authenticated task root"
                ) from exc

    def stage_file(
        self,
        final_name: str,
    ) -> _SecureStagedDeliveryFile:
        if Path(final_name).name != final_name or final_name in {".", ".."}:
            raise StrategyDeliveryToolError(
                "strategy delivery filename is unsafe"
            )
        self.assert_attached(include_staging=True)
        token = uuid.uuid4().hex
        stage_name = f"{Path(final_name).stem}.{token}{Path(final_name).suffix}"
        stage_path = self.staging_dir / stage_name
        descriptor = -1
        try:
            descriptor = os.open(
                stage_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOINHERIT", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            opened = os.fstat(descriptor)
            declared = os.lstat(stage_path)
            if (
                _is_delivery_reparse_point(stage_path, declared)
                or not stat.S_ISREG(opened.st_mode)
                or _directory_entry_identity(opened)
                != _directory_entry_identity(declared)
            ):
                raise StrategyDeliveryToolError(
                    "strategy delivery staging target is not a regular file"
                )
            identity = _directory_entry_identity(opened)
        except StrategyDeliveryToolError:
            raise
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "strategy delivery artifact could not be staged safely"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        artifact = _SecureStagedDeliveryFile(
            stage_name=stage_name,
            final_name=final_name,
            stage_path=stage_path,
            final_path=self.output_dir / final_name,
            stage_identity=identity,
        )
        self._items.append(artifact)
        self.assert_attached(include_staging=True)
        return artifact

    def write_stage(
        self,
        artifact: _SecureStagedDeliveryFile,
        payload: bytes,
    ) -> None:
        self.assert_attached(include_staging=True)
        descriptor = -1
        try:
            declared = os.lstat(artifact.stage_path)
            if (
                _is_delivery_reparse_point(artifact.stage_path, declared)
                or not stat.S_ISREG(declared.st_mode)
                or _directory_entry_identity(declared)
                != artifact.stage_identity
            ):
                raise StrategyDeliveryToolError(
                    "strategy delivery staging target changed"
                )
            descriptor = os.open(
                artifact.stage_path,
                os.O_WRONLY
                | os.O_TRUNC
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOINHERIT", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _directory_entry_identity(opened)
                != artifact.stage_identity
            ):
                raise StrategyDeliveryToolError(
                    "strategy delivery staging target changed"
                )
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("short strategy delivery artifact write")
                written += count
        except StrategyDeliveryToolError:
            raise
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "strategy delivery artifact could not be staged safely"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        _require_exact_delivery_path(
            artifact.stage_path,
            expected=payload,
            expected_hash=hashlib.sha256(payload).hexdigest(),
            expected_identity=artifact.stage_identity,
        )
        self.assert_attached(include_staging=True)

    def final_exists(self, artifact: _SecureStagedDeliveryFile) -> bool:
        self.assert_attached()
        try:
            os.lstat(artifact.final_path)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact is unavailable"
            ) from exc
        return True

    def require_final(
        self,
        artifact: _SecureStagedDeliveryFile,
        *,
        expected: bytes,
        expected_hash: str,
    ) -> None:
        self.assert_attached()
        _require_exact_delivery_path(
            artifact.final_path,
            expected=expected,
            expected_hash=expected_hash,
        )
        self.assert_attached()

    def promote_selected(
        self,
        *,
        names: list[str],
        contents: Mapping[str, bytes],
        content_hashes: Mapping[str, str],
        artifacts: Mapping[str, _SecureStagedDeliveryFile],
    ) -> None:
        for name in names:
            artifact = artifacts[name]
            self._promote_one(artifact)
            self.require_final(
                artifact,
                expected=contents[name],
                expected_hash=content_hashes[name],
            )

    def _promote_one(
        self,
        artifact: _SecureStagedDeliveryFile,
    ) -> None:
        try:
            self.assert_attached(include_staging=True)
            _require_regular_delivery_path_identity(
                artifact.stage_path,
                artifact.stage_identity,
                "strategy delivery staging target",
            )
            if self.final_exists(artifact):
                raise StrategyDeliveryToolError(
                    "strategy delivery final appeared during no-overwrite "
                    "publication"
                )
            try:
                os.link(
                    artifact.stage_path,
                    artifact.final_path,
                    follow_symlinks=False,
                )
            except (NotImplementedError, TypeError):
                os.link(artifact.stage_path, artifact.final_path)
            # Record the no-clobber link before any fallible unlink, stat,
            # identity, or attachment check so rollback owns this final.
            artifact.final_created = True
            artifact.promoted_identity = artifact.stage_identity
            _unlink_known_delivery_path(
                artifact.stage_path,
                artifact.stage_identity,
            )
            artifact.discarded = True
            _require_promoted_delivery_path(
                artifact.final_path,
                artifact.stage_identity,
            )
            self.assert_attached(include_staging=True)
        except StrategyDeliveryToolError:
            raise
        except OSError as exc:
            raise StrategyDeliveryToolError(
                "strategy delivery artifact promotion failed safely"
            ) from exc

    def discard_stage(self, artifact: _SecureStagedDeliveryFile) -> None:
        if artifact.final_created or artifact.discarded:
            return
        self.assert_attached(include_staging=True)
        _unlink_known_delivery_path(
            artifact.stage_path,
            artifact.stage_identity,
        )
        artifact.discarded = True
        self.assert_attached(include_staging=True)

    def discard_staged(self) -> None:
        for artifact in self._items:
            self.discard_stage(artifact)

    def commit(self) -> None:
        self._committed = True

    def rollback(self) -> None:
        if self._committed:
            return
        for artifact in reversed(self._items):
            try:
                self.assert_attached(include_staging=True)
                if artifact.final_created:
                    if artifact.promoted_identity is not None:
                        _unlink_known_delivery_path(
                            artifact.final_path,
                            artifact.promoted_identity,
                        )
                if not artifact.discarded:
                    _unlink_known_delivery_path(
                        artifact.stage_path,
                        artifact.stage_identity,
                    )
            except (OSError, StrategyDeliveryToolError):
                continue

    def close(self) -> None:
        self._closed = True


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
        runtime_requirements = source["runtime_requirements"]
        frame = hydrate_materialized_strategy_runtime_requirements(
            frame,
            (runtime_requirements,),
        )
        requirements_provenance = source[
            "runtime_requirements_provenance"
        ]
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
            workspace_ref=request["workspace_ref"],
            maximum_equivalence_rows=request["maximum_equivalence_rows"],
            equivalence=equivalence,
            content_hashes=content_hashes,
            runtime_requirements=requirements_provenance,
        )
        request_hash_payload: object = request
        if requirements_provenance is not None:
            request_hash_payload = {
                "request": request,
                "runtime_requirements": requirements_provenance,
            }
        return _publish_delivery(
            runtime,
            task_id=task_id,
            request=request,
            request_hash=_sha256_json(request_hash_payload),
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
    expected_workspace_ref: Mapping[str, Any],
    expected_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a Tool output against the exact refs held by its caller."""

    obj = _canonical_object(value, "export_strategy_delivery output")
    output_fields = (
        _OUTPUT_FIELDS | _OPTIONAL_OUTPUT_FIELDS
        if "runtime_requirements" in obj
        else _OUTPUT_FIELDS
    )
    _exact_fields(obj, output_fields, "export_strategy_delivery output")
    runtime_requirements = (
        validate_materialized_runtime_requirements_provenance(
            obj["runtime_requirements"]
        )
        if "runtime_requirements" in obj
        else None
    )
    strategy_ref = _strategy_ref(obj["strategy_ref"])
    dataset_ref = _dataset_ref(obj["dataset_ref"])
    trusted_strategy = _strategy_ref(expected_strategy_ref)
    trusted_dataset = _dataset_ref(expected_dataset_ref)
    trusted_workspace = _workspace_ref(
        expected_workspace_ref,
        dataset_ref=trusted_dataset,
    )
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
    workspace_ref = _workspace_ref(
        obj["workspace_ref"],
        dataset_ref=dataset_ref,
    )
    if workspace_ref != trusted_workspace:
        raise StrategyDeliveryToolError(
            "export_strategy_delivery workspace_ref does not match its "
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
        workspace_ref=workspace_ref,
        maximum_equivalence_rows=maximum_rows,
        equivalence=equivalence,
        content_hashes=content_hashes,
        runtime_requirements=runtime_requirements,
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
    obj["workspace_ref"] = workspace_ref
    obj["equivalence"] = equivalence
    obj["artifacts"] = artifacts
    if runtime_requirements is not None:
        obj["runtime_requirements"] = runtime_requirements
    return obj


def validate_strategy_delivery_artifact_records(
    value: object,
    *,
    expected_task_id: str,
    expected_delivery_id: str,
    expected_strategy_ref: Mapping[str, Any],
    expected_dataset_ref: Mapping[str, Any],
    expected_workspace_ref: Mapping[str, Any],
    expected_maximum_equivalence_rows: int,
    expected_equivalence: Mapping[str, Any],
    expected_runtime_requirements: Mapping[str, Any] | None | object = (
        _UNSPECIFIED_RUNTIME_REQUIREMENTS
    ),
) -> dict[str, dict[str, str]]:
    """Authenticate the four registry rows behind a rendered delivery."""

    records = _canonical_object(value, "strategy delivery artifact records")
    _exact_fields(
        records,
        frozenset(_FILE_CONTRACT),
        "strategy delivery artifact records",
    )
    task_id = _task_id(expected_task_id)
    if (
        not isinstance(expected_delivery_id, str)
        or _DELIVERY_ID_RE.fullmatch(expected_delivery_id) is None
    ):
        raise StrategyDeliveryToolError(
            "strategy delivery artifact records have an invalid delivery id"
        )
    strategy_ref = _strategy_ref(expected_strategy_ref)
    dataset_ref = _dataset_ref(expected_dataset_ref)
    workspace_ref = _workspace_ref(
        expected_workspace_ref,
        dataset_ref=dataset_ref,
    )
    maximum_rows = _bounded_rows(expected_maximum_equivalence_rows)
    equivalence = _canonical_object(
        expected_equivalence,
        "strategy delivery expected equivalence",
    )
    equivalence_ref = {
        "equivalence_id": _text(
            equivalence.get("equivalence_id"),
            "equivalence.equivalence_id",
        ),
        "content_hash": _hash(
            equivalence.get("content_hash"),
            "equivalence.content_hash",
        ),
        "sample_hash": _hash(
            equivalence.get("sample_hash"),
            "equivalence.sample_hash",
        ),
    }
    infer_runtime_requirements = (
        expected_runtime_requirements
        is _UNSPECIFIED_RUNTIME_REQUIREMENTS
    )
    runtime_requirements = None
    if (
        not infer_runtime_requirements
        and expected_runtime_requirements is not None
    ):
        runtime_requirements = (
            validate_materialized_runtime_requirements_provenance(
                expected_runtime_requirements
            )
        )
    projections: dict[str, dict[str, str]] = {}
    seen_ids: set[str] = set()
    for name, contract in _FILE_CONTRACT.items():
        record = _canonical_object(
            records[name],
            f"strategy delivery artifact records.{name}",
        )
        _exact_fields(
            record,
            _ARTIFACT_RECORD_FIELDS,
            f"strategy delivery artifact records.{name}",
        )
        artifact_id = _hash(
            record["id"],
            f"strategy delivery artifact records.{name}.id",
        )
        if artifact_id in seen_ids:
            raise StrategyDeliveryToolError(
                "strategy delivery artifact records reuse an artifact id"
            )
        seen_ids.add(artifact_id)
        content_hash = _hash(
            record["content_hash"],
            f"strategy delivery artifact records.{name}.content_hash",
        )
        path = Path(
            _text(
                record["path"],
                f"strategy delivery artifact records.{name}.path",
            )
        )
        canonical_path = Path(os.path.abspath(path))
        if (
            record["task_id"] != task_id
            or record["kind"] != contract["kind"]
            or record["origin_tool"] != DELIVERY_ORIGIN_TOOL
            or not path.is_absolute()
            or path != canonical_path
            or tuple(canonical_path.parts[-4:])
            != (
                task_id,
                "strategy_delivery",
                expected_delivery_id,
                contract["filename"],
            )
        ):
            raise StrategyDeliveryToolError(
                f"strategy delivery artifact record {name} identity drifted"
            )
        expected_artifact_id = _stable_task_artifact_id(
            task_id=task_id,
            kind=contract["kind"],
            path=str(canonical_path),
        )
        if artifact_id != expected_artifact_id:
            raise StrategyDeliveryToolError(
                f"strategy delivery artifact record {name} stable identity "
                "drifted"
            )
        provenance = _canonical_object(
            record["provenance"],
            f"strategy delivery artifact records.{name}.provenance",
        )
        record_runtime_requirements = (
            validate_materialized_runtime_requirements_provenance(
                provenance["runtime_requirements"]
            )
            if "runtime_requirements" in provenance
            else None
        )
        if infer_runtime_requirements:
            if not projections:
                runtime_requirements = record_runtime_requirements
            elif record_runtime_requirements != runtime_requirements:
                raise StrategyDeliveryToolError(
                    "strategy delivery artifact runtime requirements drifted"
                )
        elif record_runtime_requirements != runtime_requirements:
            raise StrategyDeliveryToolError(
                "strategy delivery artifact runtime requirements drifted"
            )
        provenance_fields = (
            _ARTIFACT_PROVENANCE_FIELDS
            | _OPTIONAL_ARTIFACT_PROVENANCE_FIELDS
            if runtime_requirements is not None
            else _ARTIFACT_PROVENANCE_FIELDS
        )
        _exact_fields(
            provenance,
            provenance_fields,
            f"strategy delivery artifact records.{name}.provenance",
        )
        expected_provenance = {
            "schema_version": DELIVERY_ARTIFACT_SCHEMA_VERSION,
            "producer_version": DELIVERY_PRODUCER_VERSION,
            "task_id": task_id,
            "delivery_id": expected_delivery_id,
            "strategy_ref": strategy_ref,
            "dataset_ref": dataset_ref,
            "workspace_ref": workspace_ref,
            "maximum_equivalence_rows": maximum_rows,
            "equivalence_ref": equivalence_ref,
            "not_applied": True,
            "not_adopted": True,
            "not_deployed": True,
            "format_key": name,
            "artifact_kind": contract["kind"],
            "artifact_content_hash": content_hash,
        }
        if runtime_requirements is not None:
            expected_provenance["runtime_requirements"] = (
                runtime_requirements
            )
        if provenance != expected_provenance:
            raise StrategyDeliveryToolError(
                f"strategy delivery artifact record {name} provenance drifted"
            )
        projections[name] = {
            "artifact_id": artifact_id,
            "content_hash": content_hash,
        }
    expected_from_records = _delivery_id(
        strategy_ref=strategy_ref,
        dataset_ref=dataset_ref,
        workspace_ref=workspace_ref,
        maximum_equivalence_rows=maximum_rows,
        equivalence=equivalence,
        content_hashes={
            name: projections[name]["content_hash"]
            for name in _FILE_CONTRACT
        },
        runtime_requirements=runtime_requirements,
    )
    if expected_from_records != expected_delivery_id:
        raise StrategyDeliveryToolError(
            "strategy delivery artifact records do not bind the delivery id"
        )
    return projections


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
        "workspace_ref": _workspace_ref(
            obj["workspace_ref"],
            dataset_ref=obj["dataset_ref"],
        ),
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


def _workspace_ref(
    value: object,
    *,
    dataset_ref: object,
) -> dict[str, Any]:
    obj = _canonical_object(value, "workspace_ref")
    _exact_fields(obj, _WORKSPACE_REF_FIELDS, "workspace_ref")
    trusted_dataset = _dataset_ref(dataset_ref)
    active_dataset_id = obj["active_dataset_id"]
    active_content_hash = obj["active_dataset_content_hash"]
    if (active_dataset_id is None) != (active_content_hash is None):
        raise StrategyDeliveryToolError(
            "workspace_ref active dataset id/hash must both be null or present"
        )
    if active_dataset_id is not None:
        active_dataset_id = _text(
            active_dataset_id,
            "workspace_ref.active_dataset_id",
        )
        active_content_hash = _hash(
            active_content_hash,
            "workspace_ref.active_dataset_content_hash",
        )
        if (
            active_dataset_id != trusted_dataset["dataset_id"]
            or active_content_hash
            != trusted_dataset["expected_content_hash"]
        ):
            raise StrategyDeliveryToolError(
                "workspace_ref active dataset does not match dataset_ref"
            )
    return {
        "revision": _non_negative_int(
            obj["revision"],
            "workspace_ref.revision",
        ),
        "analysis_generation": _non_negative_int(
            obj["analysis_generation"],
            "workspace_ref.analysis_generation",
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"],
            "workspace_ref.semantic_mapping_hash",
        ),
        "active_dataset_id": active_dataset_id,
        "active_dataset_content_hash": active_content_hash,
    }


def _load_exact_sources(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    strategy_ref = request["strategy_ref"]
    snapshot = runtime.strategies.get_strategy_snapshot(
        strategy_ref["strategy_id"]
    )
    if snapshot is None:
        raise StrategyDeliveryToolError(
            "strategy does not match the exact delivery request"
        )
    strategy = snapshot["strategy"]
    meta = snapshot["metadata"]
    spec_hash = snapshot["strategy_spec_hash"]
    if (
        meta["task_id"] != task_id
        or meta["strategy_type"] != strategy_ref["expected_strategy_type"]
        or meta["version"] != strategy_ref["expected_version"]
        or spec_hash != strategy_ref["expected_spec_hash"]
    ):
        raise StrategyDeliveryToolError(
            "strategy no longer matches the exact delivery request"
        )
    if (
        strategy.spec is None
        or not hmac.compare_digest(
            strategy_spec_hash(strategy.spec),
            spec_hash,
        )
    ):
        raise StrategyDeliveryToolError(
            "strategy is not a canonical Strategy DSL snapshot; migrate the "
            "historical compatibility row before delivery"
        )
    spec = parse_strategy_spec(strategy.spec).to_dict()
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
    runtime_requirements = load_materialized_strategy_runtime_requirements(
        runtime,
        task_id=task_id,
        strategy_id=strategy_ref["strategy_id"],
        dataset_id=dataset_ref["dataset_id"],
        dataset_content_hash=dataset_ref["expected_content_hash"],
    )
    requirements_provenance = materialized_runtime_requirements_provenance(
        candidate=runtime_requirements,
        baseline=None,
    )
    return {
        "runtime": runtime,
        "spec": spec,
        "dataset_path": dataset_path,
        "dataset_root": Path(runtime.settings.datasets_dir).absolute(),
        "dataset_source_path": str(dataset.source_path),
        "runtime_requirements": runtime_requirements,
        "runtime_requirements_provenance": requirements_provenance,
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
    workspace_ref: Mapping[str, Any],
    maximum_equivalence_rows: int,
    equivalence: Mapping[str, Any],
    content_hashes: Mapping[str, str],
    runtime_requirements: Mapping[str, Any] | None = None,
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
        "workspace_ref": dict(workspace_ref),
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
    if runtime_requirements is not None:
        body["runtime_requirements"] = (
            validate_materialized_runtime_requirements_provenance(
                runtime_requirements
            )
        )
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
    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    uow = _open_delivery_directory_chain(
        tasks_root,
        task_id=task_id,
        delivery_id=delivery_id,
    )
    provenance_base = {
        "schema_version": DELIVERY_ARTIFACT_SCHEMA_VERSION,
        "producer_version": DELIVERY_PRODUCER_VERSION,
        "task_id": task_id,
        "delivery_id": delivery_id,
        "strategy_ref": dict(request["strategy_ref"]),
        "dataset_ref": dict(request["dataset_ref"]),
        "workspace_ref": dict(request["workspace_ref"]),
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
    requirements_provenance = source[
        "runtime_requirements_provenance"
    ]
    if requirements_provenance is not None:
        provenance_base["runtime_requirements"] = (
            requirements_provenance
        )
    registry_replay = False
    records: list[dict[str, Any]]
    validated_output: dict[str, Any]
    try:
        staged = {
            name: uow.stage_file(contract["filename"])
            for name, contract in _FILE_CONTRACT.items()
        }
        for name, artifact in staged.items():
            uow.write_stage(artifact, contents[name])
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _revalidate_sources_on_connection(
                conn,
                task_id=task_id,
                request=request,
                source=source,
            )
            registry_replay = _prepare_delivery_outputs_under_lock(
                conn,
                uow=uow,
                task_id=task_id,
                staged=staged,
                contents=contents,
                content_hashes=content_hashes,
                provenance_base=provenance_base,
            )
            records = []
            for name, contract in _FILE_CONTRACT.items():
                uow.require_final(
                    staged[name],
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
                registry_replay=registry_replay,
                runtime_requirements=requirements_provenance,
            )
            for name in _FILE_CONTRACT:
                uow.require_final(
                    staged[name],
                    expected=contents[name],
                    expected_hash=content_hashes[name],
                )
            uow.assert_attached()
            validated_output = _build_validated_delivery_output(
                task_id=task_id,
                request=request,
                delivery_id=delivery_id,
                equivalence=equivalence,
                records=records,
                runtime_requirements=requirements_provenance,
            )
    except Exception:
        uow.rollback()
        raise
    else:
        # SQLite committed only after the complete public Tool output contract
        # validated. No-overwrite publication has no post-commit filesystem
        # cleanup; this only disables rollback.
        uow.commit()
    finally:
        uow.close()
    return validated_output


def _build_validated_delivery_output(
    *,
    task_id: str,
    request: Mapping[str, Any],
    delivery_id: str,
    equivalence: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    runtime_requirements: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output = {
        "schema_version": DELIVERY_TOOL_SCHEMA_VERSION,
        "delivery_id": delivery_id,
        "task_id": task_id,
        "strategy_id": request["strategy_ref"]["strategy_id"],
        "strategy_type": request["strategy_ref"]["expected_strategy_type"],
        "strategy_version": request["strategy_ref"]["expected_version"],
        "strategy_ref": dict(request["strategy_ref"]),
        "dataset_ref": dict(request["dataset_ref"]),
        "workspace_ref": dict(request["workspace_ref"]),
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
    if runtime_requirements is not None:
        output["runtime_requirements"] = dict(runtime_requirements)
    return validate_export_strategy_delivery_tool_output(
        output,
        expected_task_id=task_id,
        expected_strategy_ref=request["strategy_ref"],
        expected_dataset_ref=request["dataset_ref"],
        expected_workspace_ref=request["workspace_ref"],
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


def _open_delivery_directory_chain(
    tasks_root: Path,
    *,
    task_id: str,
    delivery_id: str,
) -> _SecureDeliveryUnitOfWork | _PathSafeDeliveryUnitOfWork:
    backend = (
        _SecureDeliveryUnitOfWork
        if _supports_secure_delivery_dir_fds()
        else _PathSafeDeliveryUnitOfWork
    )
    return backend(
        tasks_root,
        task_id=task_id,
        delivery_id=delivery_id,
    )


def _supports_secure_delivery_dir_fds() -> bool:
    supported = getattr(os, "supports_dir_fd", ())
    follows = getattr(os, "supports_follow_symlinks", ())
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.link in follows
        and all(
            function in supported
            for function in (
                os.open,
                os.stat,
                os.mkdir,
                os.link,
                os.unlink,
            )
        )
    )


def _prepare_delivery_outputs_under_lock(
    conn,
    *,
    uow: _SecureDeliveryUnitOfWork | _PathSafeDeliveryUnitOfWork,
    task_id: str,
    staged: Mapping[str, _SecureStagedDeliveryFile],
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
            uow.require_final(
                staged[name],
                expected=contents[name],
                expected_hash=content_hashes[name],
            )
        uow.discard_staged()
        return True

    existing_files = [
        name
        for name in _FILE_CONTRACT
        if uow.final_exists(staged[name])
    ]
    if existing_files:
        for name in existing_files:
            uow.require_final(
                staged[name],
                expected=contents[name],
                expected_hash=content_hashes[name],
            )
            uow.discard_stage(staged[name])
        if len(existing_files) == len(_FILE_CONTRACT):
            # Exact files with no registry rows are a recoverable crash
            # boundary: replay rows and create the missing success audit in
            # this transaction. Only authenticated registry replay requires
            # an already-persisted audit.
            return False

    missing_files = [
        name for name in _FILE_CONTRACT if name not in existing_files
    ]
    uow.promote_selected(
        names=missing_files,
        contents=contents,
        content_hashes=content_hashes,
        artifacts=staged,
    )
    for name in _FILE_CONTRACT:
        uow.require_final(
            staged[name],
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


def _directory_entry_identity(
    value: os.stat_result,
) -> tuple[int, int]:
    return (int(value.st_dev), int(value.st_ino))


def _delivery_directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _require_directory_identity(
    metadata: os.stat_result,
    expected_identity: tuple[int, int] | None,
    label: str,
) -> None:
    if (
        expected_identity is None
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or _directory_entry_identity(metadata) != expected_identity
    ):
        raise StrategyDeliveryToolError(
            f"{label} changed or became a symlink during publication"
        )


def _open_root_delivery_directory(
    path: Path,
    label: str,
) -> tuple[int, tuple[int, int]]:
    descriptor = -1
    try:
        declared = os.lstat(path)
        if stat.S_ISLNK(declared.st_mode) or not stat.S_ISDIR(
            declared.st_mode
        ):
            raise StrategyDeliveryToolError(
                f"{label} must be a regular directory, not a symlink"
            )
        descriptor = os.open(path, _delivery_directory_open_flags())
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        identity = _directory_entry_identity(opened)
        _require_directory_identity(opened, identity, label)
        _require_directory_identity(declared, identity, label)
        _require_directory_identity(current, identity, label)
        return descriptor, identity
    except StrategyDeliveryToolError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise StrategyDeliveryToolError(
            f"{label} is unavailable or is a symlink"
        ) from exc


def _open_or_create_delivery_directory(
    parent_fd: int,
    name: str,
    label: str,
) -> tuple[int, tuple[int, int]]:
    if Path(name).name != name or name in {".", ".."}:
        raise StrategyDeliveryToolError(f"{label} has an unsafe name")
    descriptor = -1
    try:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        declared = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(declared.st_mode) or not stat.S_ISDIR(
            declared.st_mode
        ):
            raise StrategyDeliveryToolError(
                f"{label} must be a regular directory, not a symlink"
            )
        descriptor = os.open(
            name,
            _delivery_directory_open_flags(),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        identity = _directory_entry_identity(opened)
        _require_directory_identity(opened, identity, label)
        _require_directory_identity(declared, identity, label)
        _require_directory_identity(current, identity, label)
        return descriptor, identity
    except StrategyDeliveryToolError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise StrategyDeliveryToolError(
            f"{label} is unavailable or is a symlink"
        ) from exc


def _require_named_directory_identity(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int] | None,
    label: str,
) -> None:
    metadata = os.stat(
        name,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    _require_directory_identity(metadata, expected_identity, label)


def _require_regular_delivery_entry_identity(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    metadata = os.stat(
        name,
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _directory_entry_identity(metadata) != expected_identity
    ):
        raise StrategyDeliveryToolError(
            f"{label} changed or is not a regular file"
        )


def _require_promoted_delivery_entry(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    metadata = os.stat(
        name,
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _directory_entry_identity(metadata) != expected_identity
    ):
        raise StrategyDeliveryToolError(
            "strategy delivery promotion changed its staged file"
        )


def _remove_delivery_entry(directory_fd: int, name: str) -> None:
    """Remove one already-authenticated delivery entry relative to its fd."""

    os.unlink(name, dir_fd=directory_fd)


def _unlink_known_delivery_entry(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    try:
        metadata = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _directory_entry_identity(metadata) != expected_identity
    ):
        raise StrategyDeliveryToolError(
            "strategy delivery cleanup target changed"
        )
    _remove_delivery_entry(directory_fd, name)


def _is_delivery_reparse_point(
    path: Path,
    metadata: os.stat_result,
) -> bool:
    file_attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(
        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    is_junction = getattr(os.path, "isjunction", None)
    try:
        junction = bool(is_junction(path)) if is_junction is not None else False
    except OSError:
        junction = True
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(file_attributes & reparse_flag)
        or junction
    )


def _open_or_create_delivery_path_directory(
    path: Path,
    label: str,
) -> tuple[int, int]:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise StrategyDeliveryToolError(f"{label} is unavailable") from exc
    try:
        before = os.lstat(path)
        if (
            _is_delivery_reparse_point(path, before)
            or not stat.S_ISDIR(before.st_mode)
        ):
            raise StrategyDeliveryToolError(
                f"{label} must be a regular directory, not a symlink, "
                "junction, or reparse point"
            )
        path.resolve(strict=True)
        after = os.lstat(path)
        identity = _directory_entry_identity(before)
        if (
            _is_delivery_reparse_point(path, after)
            or not stat.S_ISDIR(after.st_mode)
            or _directory_entry_identity(after) != identity
        ):
            raise StrategyDeliveryToolError(
                f"{label} changed during publication"
            )
        return identity
    except StrategyDeliveryToolError:
        raise
    except OSError as exc:
        raise StrategyDeliveryToolError(f"{label} is unavailable") from exc


def _require_delivery_path_directory_identity(
    path: Path,
    expected_identity: tuple[int, int] | None,
    label: str,
) -> Path:
    try:
        before = os.lstat(path)
        if (
            expected_identity is None
            or _is_delivery_reparse_point(path, before)
            or not stat.S_ISDIR(before.st_mode)
            or _directory_entry_identity(before) != expected_identity
        ):
            raise StrategyDeliveryToolError(
                f"{label} changed or became a symlink, junction, or "
                "reparse point during publication"
            )
        resolved = path.resolve(strict=True)
        after = os.lstat(path)
        if (
            _is_delivery_reparse_point(path, after)
            or not stat.S_ISDIR(after.st_mode)
            or _directory_entry_identity(after) != expected_identity
        ):
            raise StrategyDeliveryToolError(
                f"{label} changed during publication"
            )
        return resolved
    except StrategyDeliveryToolError:
        raise
    except OSError as exc:
        raise StrategyDeliveryToolError(
            f"{label} changed during publication"
        ) from exc


def _require_delivery_child_directory(
    path: Path,
    expected_identity: tuple[int, int] | None,
    label: str,
    *,
    resolved_root: Path,
) -> None:
    resolved = _require_delivery_path_directory_identity(
        path,
        expected_identity,
        label,
    )
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise StrategyDeliveryToolError(
            f"{label} escaped its authenticated task root"
        ) from exc


def _require_regular_delivery_path_identity(
    path: Path,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise StrategyDeliveryToolError(f"{label} is unavailable") from exc
    if (
        _is_delivery_reparse_point(path, metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or _directory_entry_identity(metadata) != expected_identity
    ):
        raise StrategyDeliveryToolError(
            f"{label} changed or is not a regular file"
        )


def _require_promoted_delivery_path(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    metadata = os.lstat(path)
    if (
        _is_delivery_reparse_point(path, metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or _directory_entry_identity(metadata) != expected_identity
    ):
        raise StrategyDeliveryToolError(
            "strategy delivery promotion changed its staged file"
        )


def _remove_delivery_path(path: Path) -> None:
    path.unlink()


def _unlink_known_delivery_path(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        _is_delivery_reparse_point(path, metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or _directory_entry_identity(metadata) != expected_identity
    ):
        raise StrategyDeliveryToolError(
            "strategy delivery cleanup target changed"
        )
    _remove_delivery_path(path)


def _require_exact_delivery_path(
    path: Path,
    *,
    expected: bytes,
    expected_hash: str,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    descriptor = -1
    try:
        declared = os.lstat(path)
        if (
            _is_delivery_reparse_point(path, declared)
            or not stat.S_ISREG(declared.st_mode)
        ):
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact must be a regular file"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOINHERIT", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _directory_entry_identity(declared)
            != _directory_entry_identity(before)
            or (
                expected_identity is not None
                and _directory_entry_identity(before) != expected_identity
            )
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
            or _is_delivery_reparse_point(path, current)
            or _directory_entry_identity(current)
            != _directory_entry_identity(before)
            or total != int(before.st_size)
            or not hmac.compare_digest(digest.hexdigest(), expected_hash)
            or b"".join(chunks) != expected
        ):
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact bytes changed"
            )
    except StrategyDeliveryToolError:
        raise
    except OSError as exc:
        raise StrategyDeliveryToolError(
            "existing strategy delivery artifact is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_exact_delivery_entry(
    directory_fd: int,
    name: str,
    *,
    expected: bytes,
    expected_hash: str,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    descriptor = -1
    try:
        declared = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(declared.st_mode):
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact must be a regular file"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _directory_entry_identity(declared)
            != _directory_entry_identity(before)
            or (
                expected_identity is not None
                and _directory_entry_identity(before) != expected_identity
            )
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
        current = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            _stable_file_stat(after) != _stable_file_stat(before)
            or stat.S_ISLNK(current.st_mode)
            or _directory_entry_identity(current)
            != _directory_entry_identity(before)
            or total != int(before.st_size)
            or not hmac.compare_digest(digest.hexdigest(), expected_hash)
            or b"".join(chunks) != expected
        ):
            raise StrategyDeliveryToolError(
                "existing strategy delivery artifact bytes changed"
            )
    except StrategyDeliveryToolError:
        raise
    except OSError as exc:
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
    if any(
        row[field] is None
        for field in (
            "dsl_json",
            "dsl_schema_version",
            "dsl_content_hash",
        )
    ):
        raise StrategyDeliveryToolError(
            "strategy is not a canonical Strategy DSL snapshot; migrate the "
            "historical compatibility row before delivery"
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
    if _workspace_ref_on_connection(conn, task_id=task_id) != request[
        "workspace_ref"
    ]:
        raise StrategyDeliveryToolError(
            "DataWorkspace no longer matches the exact delivery request"
        )
    runtime_requirements = source["runtime_requirements"]
    if runtime_requirements is not None:
        require_materialized_strategy_runtime_requirements_on_connection(
            conn,
            source["runtime"],
            runtime_requirements,
        )
    _require_authenticated_file_hash(
        source["dataset_path"],
        root=source["dataset_root"],
        expected_hash=dataset_ref["expected_content_hash"],
    )


def _workspace_ref_on_connection(
    conn,
    *,
    task_id: str,
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT revision, active_dataset_id, active_dataset_content_hash,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces
         WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        return {
            "revision": 0,
            "analysis_generation": 0,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                DataSemanticMapping()
            ),
            "active_dataset_id": None,
            "active_dataset_content_hash": None,
        }
    try:
        raw_mapping = str(row["semantic_mapping_json"])
        mapping_payload = json.loads(raw_mapping)
        if raw_mapping != _canonical_json(mapping_payload):
            raise ValueError("semantic mapping is not canonical JSON")
        mapping = data_semantic_mapping_from_dict(mapping_payload)
        return _workspace_ref(
            {
                "revision": int(row["revision"]),
                "analysis_generation": int(row["analysis_generation"]),
                "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
                "active_dataset_id": row["active_dataset_id"],
                "active_dataset_content_hash": row[
                    "active_dataset_content_hash"
                ],
            },
            dataset_ref={
                "dataset_id": (
                    row["active_dataset_id"]
                    if row["active_dataset_id"] is not None
                    else "__unselected__"
                ),
                "expected_content_hash": (
                    row["active_dataset_content_hash"]
                    if row["active_dataset_content_hash"] is not None
                    else "0" * 64
                ),
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategyDeliveryToolError(
            "DataWorkspace no longer matches the exact delivery request"
        ) from exc


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
    artifact_ids: set[str] = set()
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
        if artifact_id in artifact_ids:
            raise StrategyDeliveryToolError(
                "export_strategy_delivery artifact ids must be unique"
            )
        artifact_ids.add(artifact_id)
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
    registry_replay: bool,
    runtime_requirements: Mapping[str, Any] | None,
) -> None:
    detail = {
        "task_id": task_id,
        "strategy_ref": dict(request["strategy_ref"]),
        "dataset_ref": dict(request["dataset_ref"]),
        "workspace_ref": dict(request["workspace_ref"]),
        "equivalence_id": equivalence["equivalence_id"],
        "artifact_ids": [record["id"] for record in records],
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    if runtime_requirements is not None:
        detail["runtime_requirements"] = (
            validate_materialized_runtime_requirements_provenance(
                runtime_requirements
            )
        )
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
    delivery_rows = [
        row
        for row in rows
        if not (
            str(row["kind"]) in {"tool.invoke.started", "tool.invoke"}
            and str(row["target_ref"]) == DELIVERY_ORIGIN_TOOL
        )
    ]
    if not delivery_rows:
        if registry_replay:
            raise StrategyDeliveryToolError(
                "existing strategy delivery audit changed"
            )
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
            json.loads(str(delivery_rows[0]["detail_json"]))
            if len(delivery_rows) == 1
            else None
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        persisted_detail = None
    if (
        len(delivery_rows) != 1
        or str(delivery_rows[0]["kind"]) != DELIVERY_AUDIT_KIND
        or str(delivery_rows[0]["target_ref"]) != delivery_id
        or str(delivery_rows[0]["inputs_hash"]) != request_hash
        or str(delivery_rows[0]["actor"]) != "system"
        or str(delivery_rows[0]["outcome"]) != "succeeded"
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
    "validate_strategy_delivery_artifact_records",
]
