"""Governed persistence for explicit interactive-tree frontier selections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.packs.strategy import interactive_tree_tools as revision_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION_V2,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
    INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION,
    INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION_V2,
    InteractiveTreeFrontierSelectionError,
    build_interactive_tree_frontier_selection,
    canonical_interactive_tree_frontier_selection_json,
    validate_interactive_tree_frontier_selection,
)
from marvis.packs.strategy.interactive_tree_revision import (
    INTERACTIVE_TREE_ASSET_TYPE,
    INTERACTIVE_TREE_REVISION_SCHEMA_VERSION,
    INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION,
)
from marvis.packs.strategy.interactive_tree_tools import (
    VerifiedInteractiveTreeRevision,
    load_verified_interactive_tree_revision,
    load_verified_interactive_tree_revision_on_connection,
)
from marvis.repositories.task_artifacts import stable_task_artifact_id


TOOL_SCHEMA_VERSION = (
    "strategy.materialize-interactive-tree-frontier-selection-tool.v1"
)
TOOL_SCHEMA_VERSION_V2 = (
    "strategy.materialize-interactive-tree-frontier-selection-tool.v2"
)
MAX_INTERACTIVE_TREE_FRONTIER_SELECTION_BYTES = 1024 * 1024
MAX_INTERACTIVE_TREE_FRONTIER_REVISION_BYTES = 64 * 1024 * 1024

_INPUT_FIELDS = frozenset(
    {"revision_id", "source_node_id", "selection_reason"}
)
_REQUIRED_INPUT_FIELDS = _INPUT_FIELDS - {"selection_reason"}
_TASK_ARTIFACT_ROW_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "kind",
        "path",
        "content_hash",
        "origin_tool",
        "provenance_json",
        "created_at",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "selection_id",
        "selection_hash",
        "revision_artifact_id",
        "revision_artifact_kind",
        "revision_artifact_schema_version",
        "revision_artifact_content_hash",
        "revision_artifact_origin_tool",
        "revision_artifact_path",
        "revision_artifact_provenance",
        "revision_schema_version",
        "revision_id",
        "revision_hash",
        "semantic_tree_id",
        "tree_hash",
        "asset_type",
        "source_node_id",
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_ID_RE = re.compile(
    r"^interactive-tree-revision-[0-9a-f]{32}$"
)
_SELECTION_ID_RE = re.compile(
    r"^interactive-tree-frontier-selection-[0-9a-f]{32}$"
)


@dataclass(frozen=True)
class VerifiedInteractiveTreeFrontierSelection:
    """One live selection plus its recursively authenticated revision."""

    artifact_id: str
    task_id: str
    path: Path
    content_hash: str
    provenance: dict[str, Any]
    canonical_bytes: bytes
    selection: dict[str, Any]
    revision: VerifiedInteractiveTreeRevision

    def artifact_binding(self) -> dict[str, Any]:
        """Project exact live facts for the pure Pool adapter."""

        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
            "artifact_schema_version": self.provenance["schema_version"],
            "content_hash": self.content_hash,
            "origin_tool": INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
            "path": str(self.path),
            "provenance": self.provenance,
            "canonical_bytes": self.canonical_bytes,
        }


def run_materialize_interactive_tree_frontier_selection(
    inputs: object,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Persist one explicit singleton pointer from a verified revision frontier."""

    request = _validate_inputs(inputs)
    task_id = _required_text(ctx.task_id, "task_id")
    revision = load_verified_interactive_tree_revision(
        runtime,
        task_id=task_id,
        revision_id=request["revision_id"],
        max_total_bytes=MAX_INTERACTIVE_TREE_FRONTIER_REVISION_BYTES,
    )
    selection = _build_selection(
        revision,
        source_node_id=request["source_node_id"],
        selection_reason=request.get("selection_reason"),
    )
    canonical = canonical_interactive_tree_frontier_selection_json(
        selection
    ).encode("utf-8")
    if len(canonical) > MAX_INTERACTIVE_TREE_FRONTIER_SELECTION_BYTES:
        raise StrategyError(
            "interactive-tree frontier selection exceeds the JSON byte budget"
        )
    content_hash = hashlib.sha256(canonical).hexdigest()
    provenance = interactive_tree_frontier_selection_provenance(selection)
    artifact = _persist_selection(
        runtime,
        task_id=task_id,
        request=request,
        revision=revision,
        selection=selection,
        canonical=canonical,
        content_hash=content_hash,
        provenance=provenance,
    )
    frontier = selection["frontier"]
    revision_ref = selection["revision"]
    return {
        "schema_version": (
            TOOL_SCHEMA_VERSION_V2
            if revision_ref["schema_version"]
            == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
            else TOOL_SCHEMA_VERSION
        ),
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "selection_reason": selection["selection_reason"],
        "revision_id": revision_ref["revision_id"],
        "semantic_tree_id": revision_ref["semantic_tree_id"],
        "tree_hash": revision_ref["tree_hash"],
        "source_node_id": frontier["source_node_id"],
        "leaf_id": frontier["leaf_id"],
        "fragment_id": frontier["fragment_id"],
        "fragment_hash": frontier["fragment_hash"],
        "rule_id": frontier["rule_id"],
        "effect_id": frontier["effect_id"],
        "artifacts": [artifact],
    }


def canonical_interactive_tree_frontier_selection_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    selection_id: str,
) -> Path:
    """Return the sole task-owned path for one immutable selection."""

    normalized_task = _safe_component(task_id, "task_id")
    normalized_selection = _required_text(selection_id, "selection_id")
    if _SELECTION_ID_RE.fullmatch(normalized_selection) is None:
        raise StrategyError(
            "interactive-tree frontier selection_id has an invalid format"
        )
    return (
        Path(tasks_dir).absolute()
        / normalized_task
        / "strategy_interactive_tree_frontier_selections"
        / f"{normalized_selection}.json"
    )


def interactive_tree_frontier_selection_provenance(
    selection_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact registry metadata from a canonical pointer."""

    try:
        selection = validate_interactive_tree_frontier_selection(
            selection_payload
        )
    except (InteractiveTreeFrontierSelectionError, TypeError, ValueError) as exc:
        raise StrategyError(
            "interactive-tree frontier selection failed validation"
        ) from exc
    artifact = selection["revision_artifact"]
    revision = selection["revision"]
    frontier = selection["frontier"]
    is_v2 = (
        revision["schema_version"]
        == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
    )
    provenance = {
        "schema_version": (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION_V2
            if is_v2
            else INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": (
            INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION_V2
            if is_v2
            else INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION
        ),
        "task_id": artifact["task_id"],
        "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "format": "json",
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "revision_artifact_id": artifact["artifact_id"],
        "revision_artifact_kind": artifact["kind"],
        "revision_artifact_schema_version": artifact[
            "artifact_schema_version"
        ],
        "revision_artifact_content_hash": artifact["content_hash"],
        "revision_artifact_origin_tool": artifact["origin_tool"],
        "revision_artifact_path": artifact["path"],
        "revision_artifact_provenance": artifact["provenance"],
        "revision_schema_version": revision["schema_version"],
        "revision_id": revision["revision_id"],
        "revision_hash": revision["revision_hash"],
        "semantic_tree_id": revision["semantic_tree_id"],
        "tree_hash": revision["tree_hash"],
        "asset_type": revision["asset_type"],
        "source_node_id": frontier["source_node_id"],
        "leaf_id": frontier["leaf_id"],
        "fragment_id": frontier["fragment_id"],
        "fragment_hash": frontier["fragment_hash"],
        "rule_id": frontier["rule_id"],
        "effect_id": frontier["effect_id"],
    }
    if set(provenance) != _PROVENANCE_FIELDS:
        raise StrategyError(
            "interactive-tree frontier selection provenance fields drifted"
        )
    return provenance


def load_verified_interactive_tree_frontier_selection_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    reserve_bytes: Callable[[int], None] | None = None,
    revision_cache: dict[str, VerifiedInteractiveTreeRevision] | None = None,
    automatic_source_cache: dict[tuple[str, str], Any] | None = None,
) -> VerifiedInteractiveTreeFrontierSelection:
    """Load a selection and recursively re-authenticate its revision chain."""

    with runtime.task_artifacts.transaction() as conn:
        return (
            load_verified_interactive_tree_frontier_selection_artifact_on_connection(
                conn,
                runtime=runtime,
                task_id=task_id,
                artifact_id=artifact_id,
                expected_content_hash=expected_content_hash,
                expected_asset_id=expected_asset_id,
                expected_asset_hash=expected_asset_hash,
                reserve_bytes=reserve_bytes,
                revision_cache=revision_cache,
                automatic_source_cache=automatic_source_cache,
            )
        )


def load_verified_interactive_tree_frontier_selection_artifact_on_connection(
    conn,
    *,
    runtime,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    reserve_bytes: Callable[[int], None] | None = None,
    revision_cache: dict[str, VerifiedInteractiveTreeRevision] | None = None,
    automatic_source_cache: dict[tuple[str, str], Any] | None = None,
) -> VerifiedInteractiveTreeFrontierSelection:
    """Connection-scoped verifier used under the Strategy Pool writer lock."""

    if reserve_bytes is not None and not callable(reserve_bytes):
        raise StrategyError("reserve_bytes must be callable")
    normalized_task = _required_text(task_id, "task_id")
    normalized_artifact = _required_text(artifact_id, "artifact_id")
    normalized_content_hash = _required_hash(
        expected_content_hash,
        "expected_artifact_content_hash",
    )
    normalized_asset_id = _required_text(expected_asset_id, "expected_asset_id")
    normalized_asset_hash = _required_hash(
        expected_asset_hash,
        "expected_asset_hash",
    )
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (normalized_task, normalized_artifact),
    ).fetchone()
    if row is None:
        raise StrategyError(
            "interactive-tree frontier selection artifact was not found"
        )
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _require_exact_fields(
        record,
        _TASK_ARTIFACT_ROW_FIELDS,
        "interactive-tree frontier selection registry row",
    )
    fixed_record = {
        "id": normalized_artifact,
        "task_id": normalized_task,
        "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "origin_tool": INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
        "content_hash": normalized_content_hash,
    }
    for field, expected in fixed_record.items():
        actual = record[field]
        matches = (
            hmac.compare_digest(str(actual), expected)
            if field == "content_hash"
            else actual == expected
        )
        if not matches:
            raise StrategyError(
                f"interactive-tree frontier selection registry {field} changed"
            )
    provenance = _strict_json_object_from_text(
        record["provenance_json"],
        "interactive-tree frontier selection provenance",
    )
    _require_exact_fields(
        provenance,
        _PROVENANCE_FIELDS,
        "interactive-tree frontier selection provenance",
    )
    revision_schema = provenance["revision_schema_version"]
    if revision_schema == INTERACTIVE_TREE_REVISION_SCHEMA_VERSION:
        selection_artifact_schema = (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
        )
        selection_producer = (
            INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION
        )
        revision_artifact_schema = (
            revision_tools.INTERACTIVE_TREE_REVISION_ARTIFACT_SCHEMA_VERSION
        )
    elif revision_schema == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION:
        selection_artifact_schema = (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION_V2
        )
        selection_producer = (
            INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION_V2
        )
        revision_artifact_schema = (
            revision_tools.INTERACTIVE_TREE_REVISION_ARTIFACT_SCHEMA_VERSION_V2
        )
    else:
        raise StrategyError(
            "interactive-tree frontier selection revision schema changed"
        )
    expected_fixed_provenance = {
        "schema_version": (
            selection_artifact_schema
        ),
        "producer_version": (
            selection_producer
        ),
        "task_id": normalized_task,
        "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "format": "json",
        "revision_artifact_kind": (
            revision_tools.INTERACTIVE_TREE_REVISION_ARTIFACT_KIND
        ),
        "revision_artifact_schema_version": (
            revision_artifact_schema
        ),
        "revision_artifact_origin_tool": (
            revision_tools.INTERACTIVE_TREE_REVISION_ORIGIN_TOOL
        ),
        "revision_schema_version": revision_schema,
        "semantic_tree_id": normalized_asset_id,
        "tree_hash": normalized_asset_hash,
        "asset_type": INTERACTIVE_TREE_ASSET_TYPE,
    }
    for field, expected in expected_fixed_provenance.items():
        actual = provenance[field]
        matches = (
            hmac.compare_digest(str(actual), expected)
            if field.endswith("_hash")
            else actual == expected
        )
        if not matches:
            raise StrategyError(
                f"interactive-tree frontier selection provenance {field} changed"
            )
    selection_id = _required_text(
        provenance["selection_id"],
        "selection provenance selection_id",
    )
    path = Path(_required_text(record["path"], "selection artifact path"))
    expected_path = canonical_interactive_tree_frontier_selection_path(
        runtime.settings.tasks_dir,
        task_id=normalized_task,
        selection_id=selection_id,
    )
    if path != expected_path:
        raise StrategyError(
            "interactive-tree frontier selection path is not canonical"
        )
    expected_artifact_id = stable_task_artifact_id(
        task_id=normalized_task,
        kind=INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        path=str(path),
    )
    if not hmac.compare_digest(normalized_artifact, expected_artifact_id):
        raise StrategyError(
            "interactive-tree frontier selection stable identity changed"
        )
    canonical_bytes = revision_tools._read_verified_regular_file(
        path,
        root=Path(runtime.settings.tasks_dir).absolute(),
        expected_hash=normalized_content_hash,
        max_bytes=MAX_INTERACTIVE_TREE_FRONTIER_SELECTION_BYTES,
    )
    if reserve_bytes is not None:
        reserve_bytes(len(canonical_bytes))
    selection = _strict_selection_from_bytes(canonical_bytes)
    canonical = canonical_interactive_tree_frontier_selection_json(
        selection
    ).encode("utf-8")
    if not hmac.compare_digest(canonical_bytes, canonical):
        raise StrategyError(
            "interactive-tree frontier selection is not canonical JSON"
        )
    expected_provenance = interactive_tree_frontier_selection_provenance(
        selection
    )
    if not hmac.compare_digest(
        _canonical_json(provenance),
        _canonical_json(expected_provenance),
    ):
        raise StrategyError(
            "interactive-tree frontier selection provenance changed"
        )
    revision_id = selection["revision"]["revision_id"]
    revision = load_verified_interactive_tree_revision_on_connection(
        conn,
        runtime=runtime,
        task_id=normalized_task,
        revision_id=revision_id,
        max_total_bytes=MAX_INTERACTIVE_TREE_FRONTIER_REVISION_BYTES,
        reserve_bytes=reserve_bytes,
        revision_cache=revision_cache,
        automatic_source_cache=automatic_source_cache,
    )
    rebuilt = _build_selection(
        revision,
        source_node_id=selection["frontier"]["source_node_id"],
        selection_reason=selection["selection_reason"],
    )
    if rebuilt != selection:
        raise StrategyError(
            "interactive-tree frontier selection changed from its revision"
        )
    return VerifiedInteractiveTreeFrontierSelection(
        artifact_id=normalized_artifact,
        task_id=normalized_task,
        path=path,
        content_hash=normalized_content_hash,
        provenance=provenance,
        canonical_bytes=canonical_bytes,
        selection=selection,
        revision=revision,
    )


def _build_selection(
    revision: VerifiedInteractiveTreeRevision,
    *,
    source_node_id: str,
    selection_reason: str | None,
) -> dict[str, Any]:
    ancestry = revision.ancestor_revisions
    try:
        return build_interactive_tree_frontier_selection(
            revision.revision,
            revision.automatic_source.asset,
            revision_artifact_binding=revision.builder_binding(),
            source_node_id=source_node_id,
            selection_reason=selection_reason,
            parent_revision=ancestry[0] if ancestry else None,
            ancestor_revisions=ancestry[1:],
        )
    except (
        InteractiveTreeFrontierSelectionError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError(
            "interactive-tree frontier selection is invalid"
        ) from exc


def _persist_selection(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    revision: VerifiedInteractiveTreeRevision,
    selection: Mapping[str, Any],
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    out_dir = _prepare_selection_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = canonical_interactive_tree_frontier_selection_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        selection_id=selection["selection_id"],
    )
    if final_path.parent != out_dir:
        raise StrategyError(
            "interactive-tree frontier selection output path drifted"
        )
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    staged.path.write_bytes(canonical)
    record: Mapping[str, Any]
    db_committed = False
    rollback_under_lock = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                locked_revision = (
                    load_verified_interactive_tree_revision_on_connection(
                        conn,
                        runtime=runtime,
                        task_id=task_id,
                        revision_id=request["revision_id"],
                        max_total_bytes=(
                            MAX_INTERACTIVE_TREE_FRONTIER_REVISION_BYTES
                        ),
                    )
                )
                locked_selection = _build_selection(
                    locked_revision,
                    source_node_id=request["source_node_id"],
                    selection_reason=request.get("selection_reason"),
                )
                locked_canonical = (
                    canonical_interactive_tree_frontier_selection_json(
                        locked_selection
                    ).encode("utf-8")
                )
                locked_provenance = (
                    interactive_tree_frontier_selection_provenance(
                        locked_selection
                    )
                )
                if locked_revision != revision:
                    raise StrategyError(
                        "interactive-tree revision changed before selection "
                        "registration"
                    )
                if locked_selection != selection or not hmac.compare_digest(
                    locked_canonical,
                    canonical,
                ):
                    raise StrategyError(
                        "interactive-tree frontier selection changed before "
                        "registration"
                    )
                if not hmac.compare_digest(
                    _canonical_json(locked_provenance),
                    _canonical_json(provenance),
                ):
                    raise StrategyError(
                        "interactive-tree frontier selection provenance changed "
                        "before registration"
                    )
                _require_existing_selection_consistent(
                    conn,
                    task_id=task_id,
                    final_path=final_path,
                    canonical=canonical,
                    content_hash=content_hash,
                    provenance=provenance,
                )
                uow.promote_all()
                _verify_selection_file(
                    final_path,
                    expected_content=canonical,
                    expected_hash=content_hash,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=content_hash,
                    origin_tool=INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
                    provenance=provenance,
                )
                _verify_registered_selection(
                    record,
                    task_id=task_id,
                    final_path=final_path,
                    content_hash=content_hash,
                    provenance=provenance,
                    canonical=canonical,
                    registry_record=True,
                )
                conn.commit()
                db_committed = True
            except Exception:
                rollback_under_lock = True
                uow.rollback()
                raise
        uow.commit()
    except Exception:
        if not db_committed and not rollback_under_lock:
            uow.rollback()
        raise
    artifact_id = _required_text(record["id"], "selection artifact id")
    return {
        "artifact_id": artifact_id,
        "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "format": "json",
        "filename": final_path.name,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        ),
    }


def _require_existing_selection_consistent(
    conn,
    *,
    task_id: str,
    final_path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (
            task_id,
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
            str(final_path),
        ),
    ).fetchone()
    if row is None:
        if final_path.exists() or final_path.is_symlink():
            raise StrategyError(
                "interactive-tree frontier selection path exists without a "
                "registry row"
            )
        return
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _verify_registered_selection(
        record,
        task_id=task_id,
        final_path=final_path,
        content_hash=content_hash,
        provenance=provenance,
        canonical=canonical,
    )


def _verify_registered_selection(
    record: Mapping[str, Any],
    *,
    task_id: str,
    final_path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
    canonical: bytes,
    registry_record: bool = False,
) -> None:
    expected_artifact_id = stable_task_artifact_id(
        task_id=task_id,
        kind=INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        path=str(final_path),
    )
    actual_artifact_id = record.get("id")
    if not isinstance(actual_artifact_id, str) or not hmac.compare_digest(
        actual_artifact_id,
        expected_artifact_id,
    ):
        raise StrategyError(
            "interactive-tree frontier selection stable identity changed"
        )
    expected = {
        "task_id": task_id,
        "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "path": str(final_path),
        "content_hash": content_hash,
        "origin_tool": INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
    }
    for field, value in expected.items():
        actual = record.get(field)
        matches = (
            hmac.compare_digest(str(actual), value)
            if field == "content_hash"
            else actual == value
        )
        if not matches:
            raise StrategyError(
                f"interactive-tree frontier selection registry {field} changed"
            )
    if registry_record:
        raw_provenance = record.get("provenance")
        if not isinstance(raw_provenance, Mapping):
            raise StrategyError(
                "interactive-tree frontier selection registry provenance is invalid"
            )
        actual_provenance = json.loads(_canonical_json(raw_provenance))
    else:
        actual_provenance = _strict_json_object_from_text(
            record.get("provenance_json"),
            "interactive-tree frontier selection provenance",
        )
    if not hmac.compare_digest(
        _canonical_json(actual_provenance),
        _canonical_json(provenance),
    ):
        raise StrategyError(
            "interactive-tree frontier selection registry provenance changed"
        )
    _verify_selection_file(
        final_path,
        expected_content=canonical,
        expected_hash=content_hash,
    )


def _verify_selection_file(
    path: Path,
    *,
    expected_content: bytes,
    expected_hash: str,
) -> None:
    persisted = revision_tools._read_verified_regular_file(
        path,
        root=path.parents[2],
        expected_hash=expected_hash,
        max_bytes=MAX_INTERACTIVE_TREE_FRONTIER_SELECTION_BYTES,
    )
    if not hmac.compare_digest(persisted, expected_content):
        raise StrategyError(
            "interactive-tree frontier selection artifact bytes changed"
        )


def _prepare_selection_directory(
    tasks_dir: Path | str,
    *,
    task_id: str,
) -> Path:
    normalized_task = _safe_component(task_id, "task_id")
    root = Path(tasks_dir).absolute()
    try:
        if root.is_symlink():
            raise StrategyError("task artifact root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve(strict=True)
        task_dir = root / normalized_task
        if task_dir.is_symlink():
            raise StrategyError("task artifact directory must not be a symlink")
        task_dir.mkdir(exist_ok=True)
        if task_dir.resolve(strict=True).parent != resolved_root:
            raise StrategyError("task artifact directory escaped task storage")
        out_dir = (
            task_dir / "strategy_interactive_tree_frontier_selections"
        )
        if out_dir.is_symlink():
            raise StrategyError(
                "interactive-tree frontier selection directory must not be a "
                "symlink"
            )
        out_dir.mkdir(exist_ok=True)
        if out_dir.resolve(strict=True).parent != task_dir.resolve(strict=True):
            raise StrategyError(
                "interactive-tree frontier selection directory escaped task "
                "storage"
            )
    except OSError as exc:
        raise StrategyError(
            "interactive-tree frontier selection directory is unavailable"
        ) from exc
    return out_dir


def _strict_selection_from_bytes(value: bytes) -> dict[str, Any]:
    parsed = revision_tools._strict_json_object_from_bytes(
        value,
        "interactive-tree frontier selection artifact",
    )
    try:
        return validate_interactive_tree_frontier_selection(parsed)
    except (
        InteractiveTreeFrontierSelectionError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError(
            "interactive-tree frontier selection artifact failed validation"
        ) from exc


def _validate_inputs(inputs: object) -> dict[str, Any]:
    if not isinstance(inputs, Mapping) or any(
        not isinstance(key, str) for key in inputs
    ):
        raise StrategyError(
            "interactive-tree frontier selection inputs must be an object"
        )
    actual = set(inputs)
    missing = sorted(_REQUIRED_INPUT_FIELDS - actual)
    unexpected = sorted(actual - _INPUT_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported " + ", ".join(unexpected))
        raise StrategyError(
            "invalid interactive-tree frontier selection inputs ("
            + "; ".join(details)
            + ")"
        )
    revision_id = _required_text(inputs["revision_id"], "revision_id")
    if _REVISION_ID_RE.fullmatch(revision_id) is None:
        raise StrategyError("revision_id has an invalid format")
    normalized = {
        "revision_id": revision_id,
        "source_node_id": _required_text(
            inputs["source_node_id"],
            "source_node_id",
        ),
    }
    if "selection_reason" in inputs:
        reason = inputs["selection_reason"]
        if reason is not None and not isinstance(reason, str):
            raise StrategyError("selection_reason must be text or null")
        normalized["selection_reason"] = reason
    return normalized


def _strict_json_object_from_text(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise StrategyError(f"{name} must be JSON text")
    return revision_tools._strict_json_object_from_text(value, name)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "interactive-tree frontier evidence must be finite canonical JSON"
        ) from exc


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"{name} must be non-empty text")
    return value.strip()


def _required_hash(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if _SHA256_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return normalized


def _safe_component(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if normalized in {".", ".."} or Path(normalized).name != normalized:
        raise StrategyError(f"{name} must be a safe path component")
    return normalized


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise StrategyError(f"{name} fields are invalid")


__all__ = [
    "INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND",
    "INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION",
    "INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION_V2",
    "INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL",
    "MAX_INTERACTIVE_TREE_FRONTIER_SELECTION_BYTES",
    "MAX_INTERACTIVE_TREE_FRONTIER_REVISION_BYTES",
    "TOOL_SCHEMA_VERSION",
    "TOOL_SCHEMA_VERSION_V2",
    "VerifiedInteractiveTreeFrontierSelection",
    "canonical_interactive_tree_frontier_selection_path",
    "interactive_tree_frontier_selection_provenance",
    "load_verified_interactive_tree_frontier_selection_artifact",
    "load_verified_interactive_tree_frontier_selection_artifact_on_connection",
    "run_materialize_interactive_tree_frontier_selection",
]
