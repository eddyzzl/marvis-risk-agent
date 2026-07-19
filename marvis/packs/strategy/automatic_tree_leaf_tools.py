"""Governed persistence for explicit automatic-tree leaf selections.

The full automatic tree remains the sole owner of topology, executable rules,
and measured effects.  This Tool loads one live task-owned tree artifact,
verifies its registry row, canonical path, provenance, hash, and bytes, then
persists only the pointer-only leaf selection produced by the pure projection
seam.  It never chooses a business action or admits a candidate to a Pool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
import stat
from typing import Any
import unicodedata
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.packs.strategy.automatic_tree_asset import (
    AutomaticTreeAssetError,
    canonical_automatic_tree_asset_json,
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION,
    AutomaticTreeLeafFragmentError,
    IndependentlyVerifiedAutomaticTreeArtifactBinding,
    build_automatic_tree_leaf_fragment,
    canonical_automatic_tree_leaf_fragment_json,
    validate_automatic_tree_leaf_fragment,
)
from marvis.packs.strategy.errors import StrategyError


TOOL_SCHEMA_VERSION = "strategy.materialize-automatic-tree-leaf-fragment-tool.v1"
MAX_SELECTION_REASON_LENGTH = 500

SOURCE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "asset_id",
        "asset_hash",
        "tree_result_hash",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "registry_metadata_hash",
        "sample_context_hash",
    }
)
SELECTION_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "selection_id",
        "selection_hash",
        "tree_artifact_id",
        "tree_artifact_kind",
        "tree_artifact_schema_version",
        "tree_artifact_content_hash",
        "tree_artifact_origin_tool",
        "tree_artifact_path",
        "tree_artifact_provenance",
        "tree_asset_schema_version",
        "tree_asset_id",
        "tree_asset_hash",
        "tree_result_hash",
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
    }
)

_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
        "leaf_id",
        "selection_reason",
    }
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
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_SELECTION_ID_RE = re.compile(r"^automatic-tree-leaf-selection-[0-9a-f]{32}$")


@dataclass(frozen=True)
class VerifiedAutomaticTreeSource:
    """One fully verified live full-tree row and its canonical bytes."""

    artifact_id: str
    task_id: str
    kind: str
    path: Path
    content_hash: str
    origin_tool: str
    provenance: dict[str, Any]
    canonical_bytes: bytes
    asset: dict[str, Any]

    def builder_binding(
        self,
    ) -> IndependentlyVerifiedAutomaticTreeArtifactBinding:
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "artifact_schema_version": self.provenance["schema_version"],
            "content_hash": self.content_hash,
            "origin_tool": self.origin_tool,
            "path": str(self.path),
            "provenance": self.provenance,
            "canonical_bytes": self.canonical_bytes,
        }


def run_materialize_automatic_tree_leaf_fragment(
    inputs: object,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Persist one explicit pointer to a verified full-tree leaf."""

    normalized = _validate_inputs(inputs)
    task_id = _required_text(ctx.task_id, "task_id")
    source = load_verified_automatic_tree_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=normalized["source_artifact_id"],
        expected_content_hash=normalized["expected_artifact_content_hash"],
        expected_asset_id=normalized["expected_asset_id"],
        expected_asset_hash=normalized["expected_asset_hash"],
        expected_tree_result_hash=normalized["expected_tree_result_hash"],
    )
    selection = _build_selection(
        source,
        leaf_id=normalized["leaf_id"],
        selection_reason=normalized.get("selection_reason"),
    )
    canonical_content = canonical_automatic_tree_leaf_fragment_json(selection).encode(
        "utf-8"
    )
    content_hash = _sha256_bytes(canonical_content)
    provenance = automatic_tree_leaf_selection_provenance(selection)
    artifact = _persist_selection(
        runtime,
        task_id=task_id,
        request=normalized,
        source=source,
        selection=selection,
        canonical_content=canonical_content,
        content_hash=content_hash,
        provenance=provenance,
    )
    tree_asset = selection["tree_asset"]
    leaf = selection["leaf"]
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "selection_reason": selection["selection_reason"],
        "tree_asset_id": tree_asset["asset_id"],
        "tree_asset_hash": tree_asset["asset_hash"],
        "tree_result_hash": tree_asset["tree_result_hash"],
        "leaf_id": leaf["leaf_id"],
        "fragment_id": leaf["fragment_id"],
        "fragment_hash": leaf["fragment_hash"],
        "rule_id": leaf["rule_id"],
        "effect_id": leaf["effect_id"],
        "artifacts": [artifact],
    }


def canonical_automatic_tree_source_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    asset_id: str,
) -> Path:
    """Return the sole full-tree JSON path without resolving symlinks."""

    normalized_task = _safe_component(task_id, "task_id")
    normalized_asset = _required_text(asset_id, "asset_id")
    if _ASSET_ID_RE.fullmatch(normalized_asset) is None:
        raise StrategyError("asset_id has an invalid format")
    return (
        Path(tasks_dir).absolute()
        / normalized_task
        / "strategy_automatic_trees"
        / f"{normalized_asset}.json"
    )


def canonical_automatic_tree_leaf_selection_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    selection_id: str,
) -> Path:
    """Return the sole leaf-selection JSON path without resolving symlinks."""

    normalized_task = _safe_component(task_id, "task_id")
    normalized_selection = _required_text(selection_id, "selection_id")
    if _SELECTION_ID_RE.fullmatch(normalized_selection) is None:
        raise StrategyError("selection_id has an invalid format")
    return (
        Path(tasks_dir).absolute()
        / normalized_task
        / "strategy_automatic_tree_leaf_fragments"
        / f"{normalized_selection}.json"
    )


def automatic_tree_source_provenance_from_asset(
    asset_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact full-tree TaskArtifact provenance contract."""

    try:
        asset = validate_automatic_tree_asset(asset_payload)
    except (AutomaticTreeAssetError, TypeError, ValueError) as exc:
        raise StrategyError("automatic-tree source asset failed validation") from exc
    identity = asset["identity"]
    provenance = {
        "schema_version": AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION,
        "producer_version": asset["producer_version"],
        "task_id": identity["task_id"],
        "kind": AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
        "format": "json",
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "tree_result_hash": asset["tree_result"]["result_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "registry_metadata_hash": identity["registry_metadata_hash"],
        "sample_context_hash": identity["sample_context_hash"],
    }
    if set(provenance) != SOURCE_PROVENANCE_FIELDS:
        raise StrategyError("automatic-tree source provenance fields drifted")
    return provenance


def verify_automatic_tree_source_provenance(
    provenance_payload: Mapping[str, Any],
    asset_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and detach exact full-tree provenance for Pool reuse."""

    actual = _canonical_json_object(provenance_payload, "source provenance")
    _require_exact_fields(actual, SOURCE_PROVENANCE_FIELDS, "source provenance")
    expected = automatic_tree_source_provenance_from_asset(asset_payload)
    if not hmac.compare_digest(_canonical_json(actual), _canonical_json(expected)):
        raise StrategyError(
            "automatic-tree source provenance does not match canonical asset"
        )
    return actual


def automatic_tree_leaf_selection_provenance(
    selection_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive exact selection TaskArtifact provenance from its pointer."""

    try:
        selection = validate_automatic_tree_leaf_fragment(selection_payload)
    except (AutomaticTreeLeafFragmentError, TypeError, ValueError) as exc:
        raise StrategyError("automatic-tree leaf selection failed validation") from exc
    tree_artifact = selection["tree_artifact"]
    tree_asset = selection["tree_asset"]
    leaf = selection["leaf"]
    provenance = {
        "schema_version": AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION,
        "producer_version": selection["producer_version"],
        "task_id": tree_artifact["task_id"],
        "kind": AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        "format": "json",
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "tree_artifact_id": tree_artifact["artifact_id"],
        "tree_artifact_kind": tree_artifact["kind"],
        "tree_artifact_schema_version": tree_artifact["artifact_schema_version"],
        "tree_artifact_content_hash": tree_artifact["content_hash"],
        "tree_artifact_origin_tool": tree_artifact["origin_tool"],
        "tree_artifact_path": tree_artifact["path"],
        "tree_artifact_provenance": tree_artifact["provenance"],
        "tree_asset_schema_version": tree_asset["schema_version"],
        "tree_asset_id": tree_asset["asset_id"],
        "tree_asset_hash": tree_asset["asset_hash"],
        "tree_result_hash": tree_asset["tree_result_hash"],
        "leaf_id": leaf["leaf_id"],
        "fragment_id": leaf["fragment_id"],
        "fragment_hash": leaf["fragment_hash"],
        "rule_id": leaf["rule_id"],
        "effect_id": leaf["effect_id"],
    }
    if set(provenance) != SELECTION_PROVENANCE_FIELDS:
        raise StrategyError("automatic-tree selection provenance fields drifted")
    return provenance


def verify_automatic_tree_leaf_selection_provenance(
    provenance_payload: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and detach exact selection provenance for Pool reuse."""

    actual = _canonical_json_object(provenance_payload, "selection provenance")
    _require_exact_fields(
        actual,
        SELECTION_PROVENANCE_FIELDS,
        "selection provenance",
    )
    expected = automatic_tree_leaf_selection_provenance(selection_payload)
    if not hmac.compare_digest(_canonical_json(actual), _canonical_json(expected)):
        raise StrategyError(
            "automatic-tree selection provenance does not match selection"
        )
    return actual


def load_verified_automatic_tree_source_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    expected_tree_result_hash: str,
) -> VerifiedAutomaticTreeSource:
    """Load and fully verify one current-task full-tree artifact."""

    with runtime.task_artifacts.transaction() as conn:
        return load_verified_automatic_tree_source_artifact_on_connection(
            conn,
            tasks_dir=runtime.settings.tasks_dir,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
            expected_tree_result_hash=expected_tree_result_hash,
        )


def load_verified_automatic_tree_source_artifact_on_connection(
    conn,
    *,
    tasks_dir: Path | str,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    expected_tree_result_hash: str,
) -> VerifiedAutomaticTreeSource:
    """Connection-scoped verifier used under the writer lock and by Pool."""

    normalized_task = _required_text(task_id, "task_id")
    normalized_artifact = _required_text(artifact_id, "source_artifact_id")
    normalized_content_hash = _required_sha256(
        expected_content_hash,
        "expected_artifact_content_hash",
    )
    normalized_asset_id = _required_asset_id(expected_asset_id)
    normalized_asset_hash = _required_sha256(
        expected_asset_hash,
        "expected_asset_hash",
    )
    normalized_tree_hash = _required_sha256(
        expected_tree_result_hash,
        "expected_tree_result_hash",
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
            f"automatic-tree source artifact not found: {normalized_artifact}"
        )
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _require_exact_fields(
        record,
        _TASK_ARTIFACT_ROW_FIELDS,
        "automatic-tree source artifact row",
    )
    if _required_text(record["id"], "source artifact id") != normalized_artifact:
        raise StrategyError("automatic-tree source artifact id changed")
    if _required_text(record["task_id"], "source artifact task_id") != normalized_task:
        raise StrategyError("automatic-tree source artifact belongs to another task")
    kind = _required_text(record["kind"], "source artifact kind")
    if kind != AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND:
        raise StrategyError("automatic-tree source artifact kind is invalid")
    origin = _required_text(record["origin_tool"], "source artifact origin_tool")
    if origin != AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL:
        raise StrategyError("automatic-tree source artifact origin_tool is invalid")
    registered_hash = _required_sha256(
        record["content_hash"],
        "source artifact content_hash",
    )
    if not hmac.compare_digest(registered_hash, normalized_content_hash):
        raise StrategyError("automatic-tree source artifact content hash changed")
    path = Path(_required_text(record["path"], "source artifact path"))
    expected_path = canonical_automatic_tree_source_path(
        tasks_dir,
        task_id=normalized_task,
        asset_id=normalized_asset_id,
    )
    if not path.is_absolute() or path != expected_path:
        raise StrategyError("automatic-tree source artifact path is not canonical")
    _require_regular_path(path, root=Path(tasks_dir).absolute())
    before = path.lstat()
    try:
        canonical_bytes = path.read_bytes()
    except OSError as exc:
        raise StrategyError("automatic-tree source artifact could not be read") from exc
    _require_regular_path(path, root=Path(tasks_dir).absolute())
    after = path.lstat()
    if _stat_identity(before) != _stat_identity(after):
        raise StrategyError("automatic-tree source artifact changed while read")
    actual_content_hash = _sha256_bytes(canonical_bytes)
    if not hmac.compare_digest(actual_content_hash, registered_hash):
        raise StrategyError("automatic-tree source artifact content hash drifted")
    asset = _strict_automatic_tree_asset_from_bytes(canonical_bytes)
    canonical_asset_bytes = canonical_automatic_tree_asset_json(asset).encode("utf-8")
    if not hmac.compare_digest(canonical_bytes, canonical_asset_bytes):
        raise StrategyError("automatic-tree source artifact is not canonical JSON")
    if asset["identity"]["task_id"] != normalized_task:
        raise StrategyError("automatic-tree source asset belongs to another task")
    comparisons = {
        "asset_id": (asset["asset_id"], normalized_asset_id),
        "asset_hash": (asset["asset_hash"], normalized_asset_hash),
        "tree_result_hash": (
            asset["tree_result"]["result_hash"],
            normalized_tree_hash,
        ),
    }
    for field, (actual, expected) in comparisons.items():
        matches = (
            hmac.compare_digest(actual, expected)
            if field.endswith("hash")
            else actual == expected
        )
        if not matches:
            raise StrategyError(f"automatic-tree source {field} changed")
    provenance_json = record["provenance_json"]
    if not isinstance(provenance_json, str):
        raise StrategyError("automatic-tree source provenance_json is invalid")
    provenance = _strict_json_object_from_text(
        provenance_json,
        "automatic-tree source provenance_json",
    )
    provenance = verify_automatic_tree_source_provenance(provenance, asset)
    return VerifiedAutomaticTreeSource(
        artifact_id=normalized_artifact,
        task_id=normalized_task,
        kind=kind,
        path=path,
        content_hash=registered_hash,
        origin_tool=origin,
        provenance=provenance,
        canonical_bytes=canonical_bytes,
        asset=asset,
    )


def _build_selection(
    source: VerifiedAutomaticTreeSource,
    *,
    leaf_id: str,
    selection_reason: str | None,
) -> dict[str, Any]:
    try:
        return build_automatic_tree_leaf_fragment(
            source.asset,
            tree_artifact_binding=source.builder_binding(),
            leaf_id=leaf_id,
            selection_reason=selection_reason,
        )
    except AutomaticTreeLeafFragmentError as exc:
        raise StrategyError("automatic-tree leaf selection failed validation") from exc


def _persist_selection(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    source: VerifiedAutomaticTreeSource,
    selection: Mapping[str, Any],
    canonical_content: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    selection_id = _required_text(selection["selection_id"], "selection_id")
    out_dir = _prepare_selection_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = canonical_automatic_tree_leaf_selection_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        selection_id=selection_id,
    )
    if final_path.parent != out_dir:
        raise StrategyError("automatic-tree selection output path drifted")
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    staged.path.write_bytes(canonical_content)
    record: Mapping[str, Any]
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                locked_source = (
                    load_verified_automatic_tree_source_artifact_on_connection(
                        conn,
                        tasks_dir=runtime.settings.tasks_dir,
                        task_id=task_id,
                        artifact_id=request["source_artifact_id"],
                        expected_content_hash=request["expected_artifact_content_hash"],
                        expected_asset_id=request["expected_asset_id"],
                        expected_asset_hash=request["expected_asset_hash"],
                        expected_tree_result_hash=request["expected_tree_result_hash"],
                    )
                )
                locked_selection = _build_selection(
                    locked_source,
                    leaf_id=request["leaf_id"],
                    selection_reason=request.get("selection_reason"),
                )
                locked_content = canonical_automatic_tree_leaf_fragment_json(
                    locked_selection
                ).encode("utf-8")
                locked_provenance = automatic_tree_leaf_selection_provenance(
                    locked_selection
                )
                if locked_source != source:
                    raise StrategyError(
                        "automatic-tree source binding changed before registration"
                    )
                if locked_selection != selection or not hmac.compare_digest(
                    locked_content,
                    canonical_content,
                ):
                    raise StrategyError(
                        "automatic-tree leaf selection changed before registration"
                    )
                if not hmac.compare_digest(
                    _canonical_json(locked_provenance),
                    _canonical_json(provenance),
                ):
                    raise StrategyError(
                        "automatic-tree selection provenance changed before registration"
                    )
                _prepare_selection_directory(
                    runtime.settings.tasks_dir,
                    task_id=task_id,
                )
                _require_existing_selection_consistent(
                    conn,
                    task_id=task_id,
                    final_path=final_path,
                    canonical_content=canonical_content,
                    content_hash=content_hash,
                    provenance=provenance,
                )
                uow.promote_all()
                _verify_selection_file(
                    final_path,
                    root=Path(runtime.settings.tasks_dir).absolute(),
                    expected_content=canonical_content,
                    expected_content_hash=content_hash,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=content_hash,
                    origin_tool=AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
                    provenance=provenance,
                )
                _require_registered_selection_record(
                    record,
                    task_id=task_id,
                    final_path=final_path,
                    content_hash=content_hash,
                    provenance=provenance,
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
    return {
        "artifact_id": str(record["id"]),
        "kind": AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        "format": "json",
        "filename": final_path.name,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
        ),
    }


def _require_existing_selection_consistent(
    conn,
    *,
    task_id: str,
    final_path: Path,
    canonical_content: bytes,
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
            AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
            str(final_path),
        ),
    ).fetchone()
    if row is None:
        if final_path.exists() or final_path.is_symlink():
            raise StrategyError(
                "automatic-tree selection path exists without a registry row"
            )
        return
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _require_registered_selection_record(
        record,
        task_id=task_id,
        final_path=final_path,
        content_hash=content_hash,
        provenance=provenance,
        raw_provenance=True,
    )
    _verify_selection_file(
        final_path,
        root=final_path.parents[2],
        expected_content=canonical_content,
        expected_content_hash=content_hash,
    )


def _require_registered_selection_record(
    record: Mapping[str, Any],
    *,
    task_id: str,
    final_path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
    raw_provenance: bool = False,
) -> None:
    expected = {
        "task_id": task_id,
        "kind": AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        "path": str(final_path),
        "content_hash": content_hash,
        "origin_tool": AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    }
    for field, expected_value in expected.items():
        actual = record.get(field)
        matches = (
            hmac.compare_digest(str(actual), expected_value)
            if field == "content_hash"
            else actual == expected_value
        )
        if not matches:
            raise StrategyError(f"automatic-tree selection registry {field} changed")
    if raw_provenance:
        provenance_json = record.get("provenance_json")
        if not isinstance(provenance_json, str):
            raise StrategyError("automatic-tree selection provenance_json is invalid")
        actual_provenance = _strict_json_object_from_text(
            provenance_json,
            "automatic-tree selection provenance_json",
        )
    else:
        actual_provenance = _canonical_json_object(
            record.get("provenance"),
            "automatic-tree selection registry provenance",
        )
    if not hmac.compare_digest(
        _canonical_json(actual_provenance),
        _canonical_json(provenance),
    ):
        raise StrategyError("automatic-tree selection registry provenance changed")


def _verify_selection_file(
    path: Path,
    *,
    root: Path,
    expected_content: bytes,
    expected_content_hash: str,
) -> None:
    _require_regular_path(path, root=root)
    before = path.lstat()
    try:
        persisted = path.read_bytes()
    except OSError as exc:
        raise StrategyError(
            "automatic-tree selection artifact could not be read"
        ) from exc
    _require_regular_path(path, root=root)
    after = path.lstat()
    if _stat_identity(before) != _stat_identity(after):
        raise StrategyError("automatic-tree selection artifact changed while read")
    if not hmac.compare_digest(persisted, expected_content):
        raise StrategyError("automatic-tree selection artifact bytes changed")
    if not hmac.compare_digest(_sha256_bytes(persisted), expected_content_hash):
        raise StrategyError("automatic-tree selection artifact hash changed")
    parsed = _strict_leaf_selection_from_bytes(persisted)
    canonical = canonical_automatic_tree_leaf_fragment_json(parsed).encode("utf-8")
    if not hmac.compare_digest(persisted, canonical):
        raise StrategyError("automatic-tree selection artifact is not canonical JSON")


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
        out_dir = task_dir / "strategy_automatic_tree_leaf_fragments"
        if out_dir.is_symlink():
            raise StrategyError(
                "automatic-tree selection directory must not be a symlink"
            )
        out_dir.mkdir(exist_ok=True)
        if out_dir.resolve(strict=True).parent != task_dir.resolve(strict=True):
            raise StrategyError(
                "automatic-tree selection directory escaped task storage"
            )
    except OSError as exc:
        raise StrategyError(
            "automatic-tree selection directory is unavailable"
        ) from exc
    return out_dir


def _require_regular_path(path: Path, *, root: Path) -> None:
    if not path.is_absolute():
        raise StrategyError("automatic-tree artifact path must be absolute")
    declared_root = root.absolute()
    try:
        relative = path.relative_to(declared_root)
    except ValueError as exc:
        raise StrategyError(
            "automatic-tree artifact path escapes task storage"
        ) from exc
    current = declared_root
    chain = [current]
    for part in relative.parts:
        current = current / part
        chain.append(current)
    try:
        for ancestor in chain[:-1]:
            metadata = ancestor.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise StrategyError(
                    "automatic-tree artifact path has a symlink ancestor"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise StrategyError(
                    "automatic-tree artifact path ancestor is not a directory"
                )
        leaf_metadata = chain[-1].lstat()
        if stat.S_ISLNK(leaf_metadata.st_mode):
            raise StrategyError("automatic-tree artifact path must not be a symlink")
        if not stat.S_ISREG(leaf_metadata.st_mode):
            raise StrategyError("automatic-tree artifact path is not a regular file")
        resolved_root = declared_root.resolve(strict=True)
        path.resolve(strict=True).relative_to(resolved_root)
    except StrategyError:
        raise
    except FileNotFoundError as exc:
        raise StrategyError(
            "automatic-tree artifact path is not a regular file"
        ) from exc
    except OSError as exc:
        raise StrategyError("automatic-tree artifact path is unavailable") from exc
    except ValueError as exc:
        raise StrategyError(
            "automatic-tree artifact path escapes task storage"
        ) from exc


def _strict_automatic_tree_asset_from_bytes(value: bytes) -> dict[str, Any]:
    parsed = _strict_json_object_from_bytes(value, "automatic-tree source artifact")
    try:
        return validate_automatic_tree_asset(parsed)
    except (AutomaticTreeAssetError, TypeError, ValueError) as exc:
        raise StrategyError(
            "automatic-tree source artifact failed strict asset validation"
        ) from exc


def _strict_leaf_selection_from_bytes(value: bytes) -> dict[str, Any]:
    parsed = _strict_json_object_from_bytes(value, "automatic-tree leaf selection")
    try:
        return validate_automatic_tree_leaf_fragment(parsed)
    except (AutomaticTreeLeafFragmentError, TypeError, ValueError) as exc:
        raise StrategyError(
            "automatic-tree leaf selection failed strict validation"
        ) from exc


def _strict_json_object_from_bytes(value: bytes, name: str) -> dict[str, Any]:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StrategyError(f"{name} must be strict UTF-8 JSON") from exc
    return _strict_json_object_from_text(text, name)


def _strict_json_object_from_text(value: str, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise StrategyError(f"{name} contains a duplicate JSON key: {key}")
            result[key] = child
        return result

    def reject_constant(constant: str):
        raise StrategyError(f"{name} contains non-finite JSON: {constant}")

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except StrategyError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise StrategyError(f"{name} must be a JSON object")
    return parsed


def _validate_inputs(inputs: object) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise StrategyError(
            "materialize_automatic_tree_leaf_fragment inputs must be an object"
        )
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyError(
            "materialize_automatic_tree_leaf_fragment input keys must be strings"
        )
    actual_fields = set(inputs)
    missing = sorted(_REQUIRED_INPUT_FIELDS - actual_fields)
    unexpected = sorted(actual_fields - _INPUT_FIELDS)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(
            "invalid materialize_automatic_tree_leaf_fragment inputs ("
            + "; ".join(details)
            + ")"
        )
    reason = inputs.get("selection_reason")
    if reason is not None:
        if not isinstance(reason, str):
            raise StrategyError("selection_reason must be a string or null")
        if len(reason) > MAX_SELECTION_REASON_LENGTH:
            raise StrategyError("selection_reason must be at most 500 characters")
    return {
        "source_artifact_id": _required_text(
            inputs["source_artifact_id"],
            "source_artifact_id",
        ),
        "expected_artifact_content_hash": _required_sha256(
            inputs["expected_artifact_content_hash"],
            "expected_artifact_content_hash",
        ),
        "expected_asset_id": _required_asset_id(inputs["expected_asset_id"]),
        "expected_asset_hash": _required_sha256(
            inputs["expected_asset_hash"],
            "expected_asset_hash",
        ),
        "expected_tree_result_hash": _required_sha256(
            inputs["expected_tree_result_hash"],
            "expected_tree_result_hash",
        ),
        "leaf_id": _required_text(inputs["leaf_id"], "leaf_id"),
        **({"selection_reason": reason} if "selection_reason" in inputs else {}),
    }


def _required_asset_id(value: object) -> str:
    normalized = _required_text(value, "expected_asset_id")
    if _ASSET_ID_RE.fullmatch(normalized) is None:
        raise StrategyError("expected_asset_id has an invalid format")
    return normalized


def _safe_component(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if Path(normalized).name != normalized or normalized in {".", ".."}:
        raise StrategyError(f"{name} is unsafe for artifact storage")
    return normalized


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StrategyError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise StrategyError(f"{name} must not contain NUL")
    normalized = unicodedata.normalize("NFC", value)
    if value != normalized or value != value.strip():
        raise StrategyError(f"{name} must be canonical text")
    return value


def _required_sha256(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256")
    return normalized


def _canonical_json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{name} must be a JSON object")
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} must contain finite JSON") from exc


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(str(field) for field in actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported " + ", ".join(unexpected))
        raise StrategyError(f"{name} has " + "; ".join(details))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stat_identity(value) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


__all__ = [
    "MAX_SELECTION_REASON_LENGTH",
    "SELECTION_PROVENANCE_FIELDS",
    "SOURCE_PROVENANCE_FIELDS",
    "TOOL_SCHEMA_VERSION",
    "VerifiedAutomaticTreeSource",
    "automatic_tree_leaf_selection_provenance",
    "automatic_tree_source_provenance_from_asset",
    "canonical_automatic_tree_leaf_selection_path",
    "canonical_automatic_tree_source_path",
    "load_verified_automatic_tree_source_artifact",
    "load_verified_automatic_tree_source_artifact_on_connection",
    "run_materialize_automatic_tree_leaf_fragment",
    "verify_automatic_tree_leaf_selection_provenance",
    "verify_automatic_tree_source_provenance",
]
