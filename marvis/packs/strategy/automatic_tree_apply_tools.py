"""Governed full-tree writeback Tool for automatic strategy candidates.

The pure kernel owns deterministic Parquet bytes.  This module binds those
bytes to the current task's canonical tree asset and original data workspace,
then commits the derived dataset, evidence artifact, immutable run, audit and
optional workspace activation as one SQLite/file unit of work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceSnapshot,
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.files import sha256_file
from marvis.packs.strategy.automatic_tree_apply import (
    AUTOMATIC_TREE_APPLY_PRODUCER_VERSION,
    AutomaticTreeApplyResult,
    _writer_contract,
    apply_automatic_tree_to_parquet,
)
from marvis.packs.strategy.automatic_tree_leaf_tools import (
    VerifiedAutomaticTreeSource,
    load_verified_automatic_tree_source_artifact,
    load_verified_automatic_tree_source_artifact_on_connection,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.repositories.automatic_tree_apply import (
    AUTOMATIC_TREE_APPLY_EVIDENCE_ARTIFACT_KIND,
    AUTOMATIC_TREE_APPLY_ORIGIN_TOOL,
    AutomaticTreeApplyCommittedFacts,
    AutomaticTreeApplyIdentity,
    AutomaticTreeApplyRecord,
)


TOOL_SCHEMA_VERSION = "strategy.apply-automatic-tree-tool.v1"
EVIDENCE_PROVENANCE_SCHEMA_VERSION = (
    "strategy.automatic-tree-apply-evidence-artifact.v1"
)
DEFAULT_LEAF_ID_COLUMN = "automatic_tree_leaf_id"
DEFAULT_RULE_ID_COLUMN = "automatic_tree_rule_id"
RESULT_DATASET_ROLE = "strategy.automatic_tree.applied"

_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "leaf_id_column",
        "rule_id_column",
        "activate_result",
    }
)
_REQUIRED_INPUT_FIELDS = _INPUT_FIELDS - {"leaf_id_column", "rule_id_column"}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_OUTPUT_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_EVIDENCE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "run_id",
        "input_hash",
        "source_tree_artifact_id",
        "source_tree_artifact_hash",
        "asset_id",
        "asset_hash",
        "tree_result_hash",
        "source_dataset_id",
        "source_dataset_hash",
        "result_dataset_id",
        "result_dataset_hash",
        "output_leaf_column",
        "output_rule_column",
        "result_hash",
    }
)


@dataclass(frozen=True)
class _LockedContext:
    source: VerifiedAutomaticTreeSource
    dataset: dict[str, Any]
    source_path: Path
    workspace: DataWorkspaceSnapshot


def run_apply_automatic_tree(inputs: object, ctx, runtime) -> dict[str, Any]:
    """Write a canonical automatic tree to a governed derived dataset."""

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
    _require_asset_input_binding(source, normalized, task_id=task_id)
    dataset = _load_owned_dataset(runtime, normalized, task_id=task_id)
    source_path = runtime.registry.resolve_verified_path(dataset.id)
    _require_output_columns_absent(
        runtime.backend.column_names(source_path),
        leaf_id_column=normalized["leaf_id_column"],
        rule_id_column=normalized["rule_id_column"],
    )
    writer = _writer_contract()
    identity = AutomaticTreeApplyIdentity(
        task_id=task_id,
        source_tree_artifact_id=source.artifact_id,
        source_tree_artifact_hash=source.content_hash,
        asset_id=source.asset["asset_id"],
        asset_hash=source.asset["asset_hash"],
        tree_result_hash=source.asset["tree_result"]["result_hash"],
        source_dataset_id=dataset.id,
        source_dataset_hash=normalized["expected_content_hash"],
        output_leaf_column=normalized["leaf_id_column"],
        output_rule_column=normalized["rule_id_column"],
        writer_contract=str(writer["contract"]),
        writer_version=str(writer["engine_version"]),
    )
    result_dir = _safe_child_directory(
        runtime.datasets_root,
        task_id=task_id,
        child="strategy_automatic_tree_applies",
    )
    evidence_dir = _safe_child_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
        child="strategy_automatic_tree_applies",
    )
    result_path = result_dir / f"{identity.run_id}.parquet"
    evidence_path = evidence_dir / f"{identity.run_id}.evidence.json"

    cached = _try_cached_run(
        runtime,
        normalized=normalized,
        identity=identity,
        expected_result_path=result_path,
        expected_evidence_path=evidence_path,
    )
    if cached is not None:
        record, workspace = cached
        return _tool_payload(
            record,
            workspace=workspace,
            normalized=normalized,
            cached=True,
        )

    uow = ArtifactUnitOfWork()
    commit_state = {"db_committed": False}
    staged_result = uow.stage_file(result_dir, result_path.name)
    staged_evidence = None
    try:
        # The pure kernel deliberately rejects a pre-existing output, including
        # the reservation created by stage_file.
        staged_result.path.unlink(missing_ok=True)
        kernel_result = apply_automatic_tree_to_parquet(
            source.asset,
            source_path,
            staged_result.path,
            leaf_id_column=identity.output_leaf_column,
            rule_id_column=identity.output_rule_column,
        )
        _require_kernel_result(kernel_result, identity=identity, dataset=dataset)
        evidence_bytes = _canonical_json(kernel_result.to_dict()).encode("utf-8")
        staged_evidence = uow.stage_file(evidence_dir, evidence_path.name)
        staged_evidence.path.write_bytes(evidence_bytes)

        record, workspace, replayed = _commit_computed_run(
            runtime,
            normalized=normalized,
            identity=identity,
            source=source,
            dataset=dataset,
            source_path=source_path,
            kernel_result=kernel_result,
            evidence_bytes=evidence_bytes,
            staged_result=staged_result,
            staged_evidence=staged_evidence,
            uow=uow,
            commit_state=commit_state,
        )
    except Exception:
        if not commit_state["db_committed"]:
            uow.rollback()
        raise
    return _tool_payload(
        record,
        workspace=workspace,
        normalized=normalized,
        cached=replayed,
    )


def _try_cached_run(
    runtime,
    *,
    normalized: Mapping[str, Any],
    identity: AutomaticTreeApplyIdentity,
    expected_result_path: Path,
    expected_evidence_path: Path,
) -> tuple[AutomaticTreeApplyRecord, DataWorkspaceSnapshot] | None:
    with runtime.automatic_tree_apply_runs.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        locked = _load_locked_context(
            conn,
            runtime,
            normalized=normalized,
            identity=identity,
        )
        record = runtime.automatic_tree_apply_runs.get_by_input_on_connection(
            conn, identity
        )
        if record is None:
            _require_original_workspace(locked.workspace, normalized)
            conn.commit()
            return None
        _verify_cached_record(
            conn,
            runtime,
            record=record,
            identity=identity,
            source_dataset=locked.dataset,
            expected_result_path=expected_result_path,
            expected_evidence_path=expected_evidence_path,
        )
        workspace, changed = _resolve_workspace_on_replay(
            conn,
            runtime,
            record=record,
            current=locked.workspace,
            normalized=normalized,
        )
        if changed:
            _write_activation_audit(conn, runtime, record=record, workspace=workspace)
        conn.commit()
        return record, workspace


def _commit_computed_run(
    runtime,
    *,
    normalized: Mapping[str, Any],
    identity: AutomaticTreeApplyIdentity,
    source: VerifiedAutomaticTreeSource,
    dataset,
    source_path: Path,
    kernel_result: AutomaticTreeApplyResult,
    evidence_bytes: bytes,
    staged_result,
    staged_evidence,
    uow: ArtifactUnitOfWork,
    commit_state: dict[str, bool],
) -> tuple[AutomaticTreeApplyRecord, DataWorkspaceSnapshot, bool]:
    with runtime.automatic_tree_apply_runs.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            locked = _load_locked_context(
                conn,
                runtime,
                normalized=normalized,
                identity=identity,
            )
            existing = runtime.automatic_tree_apply_runs.get_by_input_on_connection(
                conn, identity
            )
            if existing is not None:
                uow.rollback()
                _verify_cached_record(
                    conn,
                    runtime,
                    record=existing,
                    identity=identity,
                    source_dataset=locked.dataset,
                    expected_result_path=staged_result.final_path,
                    expected_evidence_path=staged_evidence.final_path,
                )
                workspace, changed = _resolve_workspace_on_replay(
                    conn,
                    runtime,
                    record=existing,
                    current=locked.workspace,
                    normalized=normalized,
                )
                if changed:
                    _write_activation_audit(
                        conn, runtime, record=existing, workspace=workspace
                    )
                conn.commit()
                return existing, workspace, True

            _require_original_workspace(locked.workspace, normalized)
            if locked.source != source:
                raise StrategyError(
                    "automatic-tree source binding changed before apply commit"
                )
            if locked.source_path != source_path or not hmac.compare_digest(
                sha256_file(locked.source_path), identity.source_dataset_hash
            ):
                raise StrategyError(
                    "automatic-tree source dataset changed before apply commit"
                )
            if (
                staged_result.final_path.exists()
                or staged_result.final_path.is_symlink()
            ):
                raise StrategyError("automatic-tree result path already exists")
            if (
                staged_evidence.final_path.exists()
                or staged_evidence.final_path.is_symlink()
            ):
                raise StrategyError("automatic-tree evidence path already exists")
            uow.promote_all()
            if not hmac.compare_digest(
                sha256_file(staged_result.final_path),
                kernel_result.output_content_hash,
            ):
                raise StrategyError("automatic-tree promoted result hash changed")
            if not hmac.compare_digest(
                sha256_file(staged_evidence.final_path), _sha256_bytes(evidence_bytes)
            ):
                raise StrategyError("automatic-tree promoted evidence hash changed")

            result_dataset = runtime.registry.register_existing_on_connection(
                conn,
                staged_result.final_path,
                task_id=identity.task_id,
                role=RESULT_DATASET_ROLE,
                target_col_override=(
                    dataset.target_col if dataset.has_target else None
                ),
                seed=0,
            )
            _require_registered_result(
                result_dataset,
                kernel_result=kernel_result,
                expected_path=staged_result.final_path,
                runtime=runtime,
            )
            evidence_hash = _sha256_bytes(evidence_bytes)
            provenance = automatic_tree_apply_evidence_provenance(
                identity,
                result_dataset_id=result_dataset.id,
                result_dataset_hash=result_dataset.content_hash,
                result_hash=kernel_result.result_hash,
            )
            artifact = runtime.task_artifacts.register_on_connection(
                conn,
                task_id=identity.task_id,
                kind=AUTOMATIC_TREE_APPLY_EVIDENCE_ARTIFACT_KIND,
                path=str(staged_evidence.final_path),
                content_hash=evidence_hash,
                origin_tool=AUTOMATIC_TREE_APPLY_ORIGIN_TOOL,
                provenance=provenance,
            )
            committed = AutomaticTreeApplyCommittedFacts(
                result_dataset_id=result_dataset.id,
                result_dataset_hash=result_dataset.content_hash,
                result_dataset_path=result_dataset.source_path,
                evidence_artifact_id=artifact["id"],
                evidence_artifact_hash=evidence_hash,
                evidence_artifact_path=artifact["path"],
            )
            record = runtime.automatic_tree_apply_runs.record_succeeded_on_connection(
                conn,
                identity,
                committed,
                result_payload=kernel_result.to_dict(),
            )
            workspace = locked.workspace
            if normalized["activate_result"]:
                workspace = _activate_result(
                    conn,
                    runtime,
                    record=record,
                    source_workspace=locked.workspace,
                )
            runtime.repo.write_audit_on_connection(
                conn,
                kind="strategy.automatic_tree.apply.completed",
                target_ref=identity.run_id,
                actor="agent:strategy-automatic-tree-apply",
                inputs_hash=identity.input_hash,
                outcome="succeeded",
                detail={
                    "task_id": identity.task_id,
                    "source_tree_artifact_id": identity.source_tree_artifact_id,
                    "source_dataset_id": identity.source_dataset_id,
                    "result_dataset_id": result_dataset.id,
                    "result_dataset_hash": result_dataset.content_hash,
                    "evidence_artifact_id": artifact["id"],
                    "activated": bool(normalized["activate_result"]),
                },
            )
            conn.commit()
            commit_state["db_committed"] = True
        except Exception:
            conn.rollback()
            # Restore/remove promoted files while this writer still owns the
            # BEGIN IMMEDIATE lock boundary.
            uow.rollback()
            raise
    uow.commit()
    return record, workspace, False


def automatic_tree_apply_evidence_provenance(
    identity: AutomaticTreeApplyIdentity,
    *,
    result_dataset_id: str,
    result_dataset_hash: str,
    result_hash: str,
) -> dict[str, Any]:
    provenance = {
        "schema_version": EVIDENCE_PROVENANCE_SCHEMA_VERSION,
        "producer_version": AUTOMATIC_TREE_APPLY_PRODUCER_VERSION,
        "task_id": identity.task_id,
        "kind": AUTOMATIC_TREE_APPLY_EVIDENCE_ARTIFACT_KIND,
        "format": "json",
        "run_id": identity.run_id,
        "input_hash": identity.input_hash,
        "source_tree_artifact_id": identity.source_tree_artifact_id,
        "source_tree_artifact_hash": identity.source_tree_artifact_hash,
        "asset_id": identity.asset_id,
        "asset_hash": identity.asset_hash,
        "tree_result_hash": identity.tree_result_hash,
        "source_dataset_id": identity.source_dataset_id,
        "source_dataset_hash": identity.source_dataset_hash,
        "result_dataset_id": _required_text(result_dataset_id, "result_dataset_id"),
        "result_dataset_hash": _required_hash(
            result_dataset_hash, "result_dataset_hash"
        ),
        "output_leaf_column": identity.output_leaf_column,
        "output_rule_column": identity.output_rule_column,
        "result_hash": _required_hash(result_hash, "result_hash"),
    }
    if set(provenance) != _EVIDENCE_PROVENANCE_FIELDS:
        raise StrategyError("automatic-tree apply evidence provenance fields drifted")
    return provenance


def _resolve_workspace_on_replay(
    conn,
    runtime,
    *,
    record: AutomaticTreeApplyRecord,
    current: DataWorkspaceSnapshot,
    normalized: Mapping[str, Any],
) -> tuple[DataWorkspaceSnapshot, bool]:
    if not normalized["activate_result"]:
        if not _is_original_workspace(current, normalized):
            _require_exact_activated_workspace(
                current,
                record=record,
                normalized=normalized,
            )
        return current, False
    if _is_original_workspace(current, normalized):
        return (
            _activate_result(
                conn,
                runtime,
                record=record,
                source_workspace=current,
            ),
            True,
        )
    _require_exact_activated_workspace(current, record=record, normalized=normalized)
    return current, False


def _activate_result(
    conn,
    runtime,
    *,
    record: AutomaticTreeApplyRecord,
    source_workspace: DataWorkspaceSnapshot,
) -> DataWorkspaceSnapshot:
    mapping = _derived_semantic_mapping(
        source_workspace.semantic_mapping,
        leaf_column=record.identity.output_leaf_column,
        rule_column=record.identity.output_rule_column,
    )
    return runtime.data_workspaces.activate_derived_on_connection(
        conn,
        record.task_id,
        expected_revision=source_workspace.revision,
        source_dataset_id=record.identity.source_dataset_id,
        source_dataset_content_hash=record.identity.source_dataset_hash,
        result_dataset_id=record.committed.result_dataset_id,
        result_dataset_content_hash=record.committed.result_dataset_hash,
        semantic_mapping=mapping,
        page="history",
        selected_field=source_workspace.selected_field,
        audit={
            "actor": "agent:strategy-automatic-tree-apply",
            "detail": {"automatic_tree_apply_run_id": record.run_id},
        },
    )


def _derived_semantic_mapping(
    mapping: DataSemanticMapping,
    *,
    leaf_column: str,
    rule_column: str,
) -> DataSemanticMapping:
    roles = dict(mapping.field_roles)
    if leaf_column in roles or rule_column in roles:
        raise StrategyError("automatic-tree output semantic fields already exist")
    roles[leaf_column] = "rule_node"
    roles[rule_column] = "segment"
    return DataSemanticMapping(
        target_col=mapping.target_col,
        field_roles=roles,
        business_names=dict(mapping.business_names),
    )


def _write_activation_audit(conn, runtime, *, record, workspace) -> None:
    runtime.repo.write_audit_on_connection(
        conn,
        kind="strategy.automatic_tree.apply.activated",
        target_ref=record.run_id,
        actor="agent:strategy-automatic-tree-apply",
        inputs_hash=record.input_hash,
        outcome="succeeded",
        detail={
            "task_id": record.task_id,
            "result_dataset_id": record.committed.result_dataset_id,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
        },
    )


def _verify_cached_record(
    conn,
    runtime,
    *,
    record: AutomaticTreeApplyRecord,
    identity: AutomaticTreeApplyIdentity,
    source_dataset: Mapping[str, Any],
    expected_result_path: Path,
    expected_evidence_path: Path,
) -> None:
    if record.identity != identity:
        raise StrategyError("cached automatic-tree apply identity changed")
    result = _dataset_row_on_connection(
        conn,
        dataset_id=record.committed.result_dataset_id,
        task_id=record.task_id,
        label="cached automatic-tree result dataset",
    )
    if (
        result["role"] != RESULT_DATASET_ROLE
        or result["format"] != "parquet"
        or result["content_hash"] != record.committed.result_dataset_hash
        or result["source_path"] != record.committed.result_dataset_path
        or int(result["row_count"]) != int(record.result_payload["output"]["row_count"])
        or bool(result["has_target"]) != bool(source_dataset["has_target"])
        or result["target_col"] != source_dataset["target_col"]
    ):
        raise StrategyError("cached automatic-tree result dataset binding changed")
    result_path = _verified_dataset_path(
        runtime,
        result,
        expected_hash=record.committed.result_dataset_hash,
    )
    if result_path != expected_result_path:
        raise StrategyError("cached automatic-tree result path changed")
    expected_columns = [
        field["name"] for field in record.result_payload["output"]["schema"]["fields"]
    ]
    if _dataset_column_names(result) != expected_columns:
        raise StrategyError("cached automatic-tree result columns changed")

    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool, provenance_json
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (record.task_id, record.committed.evidence_artifact_id),
    ).fetchone()
    if row is None:
        raise StrategyError("cached automatic-tree evidence artifact is missing")
    try:
        provenance = json.loads(row["provenance_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise StrategyError(
            "cached automatic-tree evidence provenance is invalid"
        ) from exc
    expected_provenance = automatic_tree_apply_evidence_provenance(
        identity,
        result_dataset_id=record.committed.result_dataset_id,
        result_dataset_hash=record.committed.result_dataset_hash,
        result_hash=record.result_payload["result_hash"],
    )
    if (
        row["kind"] != AUTOMATIC_TREE_APPLY_EVIDENCE_ARTIFACT_KIND
        or row["origin_tool"] != AUTOMATIC_TREE_APPLY_ORIGIN_TOOL
        or row["path"] != record.committed.evidence_artifact_path
        or row["content_hash"] != record.committed.evidence_artifact_hash
        or provenance != expected_provenance
    ):
        raise StrategyError("cached automatic-tree evidence binding changed")
    evidence_path = _verified_artifact_path(
        runtime,
        task_id=record.task_id,
        path=Path(row["path"]),
        expected_hash=record.committed.evidence_artifact_hash,
    )
    if evidence_path != expected_evidence_path:
        raise StrategyError("cached automatic-tree evidence path changed")
    if evidence_path.read_bytes() != record.result_json.encode("utf-8"):
        raise StrategyError("cached automatic-tree evidence content changed")


def _load_locked_context(
    conn,
    runtime,
    *,
    normalized: Mapping[str, Any],
    identity: AutomaticTreeApplyIdentity,
) -> _LockedContext:
    source = load_verified_automatic_tree_source_artifact_on_connection(
        conn,
        tasks_dir=runtime.settings.tasks_dir,
        task_id=identity.task_id,
        artifact_id=normalized["source_artifact_id"],
        expected_content_hash=normalized["expected_artifact_content_hash"],
        expected_asset_id=normalized["expected_asset_id"],
        expected_asset_hash=normalized["expected_asset_hash"],
        expected_tree_result_hash=normalized["expected_tree_result_hash"],
    )
    _require_asset_input_binding(source, normalized, task_id=identity.task_id)
    dataset = _dataset_row_on_connection(
        conn,
        dataset_id=identity.source_dataset_id,
        task_id=identity.task_id,
        label="automatic-tree source dataset",
    )
    if dataset["content_hash"] != identity.source_dataset_hash:
        raise StrategyError("automatic-tree source dataset content hash changed")
    source_path = _verified_dataset_path(
        runtime,
        dataset,
        expected_hash=identity.source_dataset_hash,
    )
    _require_output_columns_absent(
        _dataset_column_names(dataset),
        leaf_id_column=identity.output_leaf_column,
        rule_id_column=identity.output_rule_column,
    )
    workspace = _workspace_on_connection(conn, task_id=identity.task_id)
    return _LockedContext(
        source=source,
        dataset=dataset,
        source_path=source_path,
        workspace=workspace,
    )


def _load_owned_dataset(runtime, normalized, *, task_id: str):
    try:
        dataset = runtime.registry.get(normalized["dataset_id"])
    except KeyError as exc:
        raise StrategyError(
            f"automatic-tree source dataset not found: {normalized['dataset_id']}"
        ) from exc
    if str(dataset.task_id) != task_id:
        raise StrategyError(
            f"automatic-tree source dataset not found: {normalized['dataset_id']}"
        )
    if not isinstance(dataset.content_hash, str) or not hmac.compare_digest(
        dataset.content_hash, normalized["expected_content_hash"]
    ):
        raise StrategyError("automatic-tree source dataset content hash changed")
    return dataset


def _require_asset_input_binding(
    source: VerifiedAutomaticTreeSource,
    normalized: Mapping[str, Any],
    *,
    task_id: str,
) -> None:
    identity = source.asset["identity"]
    expected = {
        "task_id": task_id,
        "dataset_id": normalized["dataset_id"],
        "dataset_content_hash": normalized["expected_content_hash"],
        "workspace_revision": normalized["workspace_revision"],
        "workspace_generation": normalized["analysis_generation"],
        "semantic_mapping_hash": normalized["semantic_mapping_hash"],
    }
    for field, value in expected.items():
        actual = identity[field]
        matches = (
            hmac.compare_digest(str(actual), str(value))
            if field.endswith("hash")
            else actual == value
        )
        if not matches:
            raise StrategyError(
                f"automatic-tree asset {field} does not match apply input"
            )


def _require_kernel_result(
    result: AutomaticTreeApplyResult,
    *,
    identity: AutomaticTreeApplyIdentity,
    dataset,
) -> None:
    if (
        result.source_content_hash != identity.source_dataset_hash
        or result.source_row_count != dataset.row_count
        or result.asset_id != identity.asset_id
        or result.asset_hash != identity.asset_hash
        or result.tree_result_hash != identity.tree_result_hash
        or result.output_columns
        != {
            "leaf_id": identity.output_leaf_column,
            "rule_id": identity.output_rule_column,
        }
        or result.writer_contract != _writer_contract()
    ):
        raise StrategyError("automatic-tree apply kernel result binding changed")


def _require_registered_result(
    dataset,
    *,
    kernel_result: AutomaticTreeApplyResult,
    expected_path: Path,
    runtime,
) -> None:
    expected_relative = (
        expected_path.resolve().relative_to(runtime.datasets_root.resolve()).as_posix()
    )
    if (
        dataset.task_id is None
        or dataset.role != RESULT_DATASET_ROLE
        or dataset.source_path != expected_relative
        or dataset.format != "parquet"
        or dataset.row_count != kernel_result.source_row_count
        or dataset.content_hash != kernel_result.output_content_hash
        or [profile.name for profile in dataset.columns]
        != [field["name"] for field in kernel_result.output_schema["fields"]]
    ):
        raise StrategyError("registered automatic-tree result dataset changed")


def _require_original_workspace(
    snapshot: DataWorkspaceSnapshot,
    normalized: Mapping[str, Any],
) -> None:
    if not _is_original_workspace(snapshot, normalized):
        raise StrategyError(
            "automatic-tree apply requires the original active data workspace"
        )


def _is_original_workspace(
    snapshot: DataWorkspaceSnapshot,
    normalized: Mapping[str, Any],
) -> bool:
    return bool(
        snapshot.revision == normalized["workspace_revision"]
        and snapshot.analysis_generation == normalized["analysis_generation"]
        and snapshot.active_dataset_id == normalized["dataset_id"]
        and snapshot.active_dataset_content_hash == normalized["expected_content_hash"]
        and hmac.compare_digest(
            data_semantic_mapping_hash(snapshot.semantic_mapping),
            normalized["semantic_mapping_hash"],
        )
    )


def _require_exact_activated_workspace(
    snapshot: DataWorkspaceSnapshot,
    *,
    record: AutomaticTreeApplyRecord,
    normalized: Mapping[str, Any],
) -> None:
    roles = dict(snapshot.semantic_mapping.field_roles)
    if (
        roles.pop(record.identity.output_leaf_column, None) != "rule_node"
        or roles.pop(record.identity.output_rule_column, None) != "segment"
    ):
        raise StrategyError("automatic-tree activated workspace semantics changed")
    reconstructed = DataSemanticMapping(
        target_col=snapshot.semantic_mapping.target_col,
        field_roles=roles,
        business_names=dict(snapshot.semantic_mapping.business_names),
    )
    if not (
        snapshot.revision == normalized["workspace_revision"] + 1
        and snapshot.analysis_generation == normalized["analysis_generation"] + 1
        and snapshot.active_dataset_id == record.committed.result_dataset_id
        and snapshot.active_dataset_content_hash == record.committed.result_dataset_hash
        and snapshot.page == "history"
        and hmac.compare_digest(
            data_semantic_mapping_hash(reconstructed),
            normalized["semantic_mapping_hash"],
        )
    ):
        raise StrategyError("automatic-tree activated workspace binding changed")


def _workspace_on_connection(conn, *, task_id: str) -> DataWorkspaceSnapshot:
    row = conn.execute(
        "SELECT * FROM data_workspaces WHERE task_id = ?", (task_id,)
    ).fetchone()
    if row is None:
        raise StrategyError("automatic-tree apply requires a persisted data workspace")
    try:
        mapping = data_semantic_mapping_from_dict(
            json.loads(row["semantic_mapping_json"])
        )
        return DataWorkspaceSnapshot(
            task_id=task_id,
            schema_version=row["schema_version"],
            revision=int(row["revision"]),
            active_dataset_id=row["active_dataset_id"],
            active_dataset_content_hash=row["active_dataset_content_hash"],
            analysis_generation=int(row["analysis_generation"]),
            page=row["page"],
            selected_field=row["selected_field"],
            semantic_mapping=mapping,
            updated_at=row["updated_at"],
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("automatic-tree data workspace is invalid") from exc


def _dataset_row_on_connection(
    conn,
    *,
    dataset_id: str,
    task_id: str,
    label: str,
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, task_id, role, source_path, format, sheet, row_count,
               columns_json, has_target, target_col, created_at, content_hash
          FROM datasets
         WHERE task_id = ? AND id = ?
        """,
        (task_id, dataset_id),
    ).fetchone()
    if row is None:
        raise StrategyError(f"{label} not found: {dataset_id}")
    return {key: row[key] for key in row.keys()}


def _dataset_column_names(dataset: Mapping[str, Any]) -> list[str]:
    try:
        columns = json.loads(dataset["columns_json"])
        names = [column["name"] for column in columns]
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise StrategyError(
            "automatic-tree dataset column metadata is invalid"
        ) from exc
    if not all(isinstance(name, str) and name for name in names):
        raise StrategyError("automatic-tree dataset column metadata is invalid")
    return names


def _verified_dataset_path(runtime, dataset, *, expected_hash: str) -> Path:
    declared_root = Path(runtime.datasets_root).absolute()
    candidate = declared_root / str(dataset["source_path"])
    try:
        resolved_root = declared_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
        if candidate.is_symlink() or not resolved.is_file():
            raise OSError("dataset path is not a regular file")
    except (OSError, RuntimeError, ValueError) as exc:
        raise StrategyError(
            "automatic-tree dataset path is unavailable or unsafe"
        ) from exc
    actual_hash = sha256_file(resolved)
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise StrategyError("automatic-tree dataset physical bytes changed")
    return resolved


def _verified_artifact_path(
    runtime,
    *,
    task_id: str,
    path: Path,
    expected_hash: str,
) -> Path:
    task_root = (Path(runtime.settings.tasks_dir).absolute() / task_id).resolve(
        strict=True
    )
    try:
        if not path.is_absolute() or path.is_symlink():
            raise OSError("artifact path is not absolute and regular")
        resolved = path.resolve(strict=True)
        resolved.relative_to(task_root)
        if not resolved.is_file():
            raise OSError("artifact path is not a regular file")
    except (OSError, RuntimeError, ValueError) as exc:
        raise StrategyError(
            "automatic-tree evidence path is unavailable or unsafe"
        ) from exc
    if not hmac.compare_digest(sha256_file(resolved), expected_hash):
        raise StrategyError("automatic-tree evidence physical bytes changed")
    return resolved


def _require_output_columns_absent(
    source_columns,
    *,
    leaf_id_column: str,
    rule_id_column: str,
) -> None:
    folded = {str(column).casefold() for column in source_columns}
    collisions = [
        column
        for column in (leaf_id_column, rule_id_column)
        if column.casefold() in folded
    ]
    if collisions:
        raise StrategyError(
            "automatic-tree output columns collide with source columns: "
            + ", ".join(collisions)
        )


def _safe_child_directory(root: Path, *, task_id: str, child: str) -> Path:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise StrategyError("task id is unsafe for automatic-tree output")
    declared_root = Path(root).absolute()
    if declared_root.is_symlink():
        raise StrategyError("automatic-tree output root must not be a symlink")
    declared_root.mkdir(parents=True, exist_ok=True)
    resolved_root = declared_root.resolve(strict=True)
    task_root = declared_root / task_id
    if task_root.is_symlink():
        raise StrategyError("automatic-tree task output directory is unsafe")
    task_root.mkdir(exist_ok=True)
    if task_root.resolve(strict=True).parent != resolved_root:
        raise StrategyError("automatic-tree task output escaped its root")
    output = task_root / child
    if output.is_symlink():
        raise StrategyError("automatic-tree output directory is unsafe")
    output.mkdir(exist_ok=True)
    if output.resolve(strict=True).parent != task_root.resolve(strict=True):
        raise StrategyError("automatic-tree output directory escaped its task")
    return output


def _tool_payload(
    record: AutomaticTreeApplyRecord,
    *,
    workspace: DataWorkspaceSnapshot,
    normalized: Mapping[str, Any],
    cached: bool,
) -> dict[str, Any]:
    payload = record.result_payload
    activated = workspace.active_dataset_id == record.committed.result_dataset_id
    result_revision = workspace.revision if activated else None
    result_generation = workspace.analysis_generation if activated else None
    if activated:
        result_semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    else:
        result_semantic_hash = data_semantic_mapping_hash(
            _derived_semantic_mapping(
                workspace.semantic_mapping,
                leaf_column=record.identity.output_leaf_column,
                rule_column=record.identity.output_rule_column,
            )
        )
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "run_id": record.run_id,
        "input_hash": record.input_hash,
        "cached": bool(cached),
        "activated": activated,
        "source": {
            "tree_artifact_id": record.identity.source_tree_artifact_id,
            "tree_artifact_content_hash": record.identity.source_tree_artifact_hash,
            "asset_id": record.identity.asset_id,
            "asset_hash": record.identity.asset_hash,
            "tree_result_hash": record.identity.tree_result_hash,
            "dataset_id": record.identity.source_dataset_id,
            "dataset_content_hash": record.identity.source_dataset_hash,
            "row_count": int(payload["source"]["row_count"]),
        },
        "result": {
            "dataset_id": record.committed.result_dataset_id,
            "dataset_content_hash": record.committed.result_dataset_hash,
            "row_count": int(payload["output"]["row_count"]),
            "result_hash": payload["result_hash"],
        },
        "columns": dict(payload["output"]["columns"]),
        "leaf_distribution": [
            dict(item) for item in payload["output"]["leaf_distribution"]
        ],
        "workspace": {
            "source_revision": int(normalized["workspace_revision"]),
            "source_analysis_generation": int(normalized["analysis_generation"]),
            "source_semantic_mapping_hash": normalized["semantic_mapping_hash"],
            "result_revision": result_revision,
            "result_analysis_generation": result_generation,
            "result_semantic_mapping_hash": result_semantic_hash,
            "active_dataset_id": workspace.active_dataset_id,
        },
        "evidence": {
            "artifact_id": record.committed.evidence_artifact_id,
            "content_hash": record.committed.evidence_artifact_hash,
            "download_url": (
                f"/api/tasks/{quote(record.task_id, safe='')}/task-artifacts/"
                f"{quote(record.committed.evidence_artifact_id, safe='')}/download"
            ),
        },
    }


def _validate_inputs(inputs: object) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise StrategyError("automatic-tree apply inputs must be an object")
    actual = set(inputs)
    missing = sorted(_REQUIRED_INPUT_FIELDS - actual)
    unexpected = sorted(actual - _INPUT_FIELDS)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected: " + ", ".join(unexpected))
        raise StrategyError(
            "invalid automatic-tree apply inputs (" + "; ".join(detail) + ")"
        )
    normalized = {
        "source_artifact_id": _required_text(
            inputs["source_artifact_id"], "source_artifact_id"
        ),
        "expected_artifact_content_hash": _required_hash(
            inputs["expected_artifact_content_hash"],
            "expected_artifact_content_hash",
        ),
        "expected_asset_id": _required_asset_id(inputs["expected_asset_id"]),
        "expected_asset_hash": _required_hash(
            inputs["expected_asset_hash"], "expected_asset_hash"
        ),
        "expected_tree_result_hash": _required_hash(
            inputs["expected_tree_result_hash"], "expected_tree_result_hash"
        ),
        "dataset_id": _required_text(inputs["dataset_id"], "dataset_id"),
        "expected_content_hash": _required_hash(
            inputs["expected_content_hash"], "expected_content_hash"
        ),
        "workspace_revision": _non_negative_int(
            inputs["workspace_revision"], "workspace_revision"
        ),
        "analysis_generation": _non_negative_int(
            inputs["analysis_generation"], "analysis_generation"
        ),
        "semantic_mapping_hash": _required_hash(
            inputs["semantic_mapping_hash"], "semantic_mapping_hash"
        ),
        "leaf_id_column": _output_column(
            inputs.get("leaf_id_column", DEFAULT_LEAF_ID_COLUMN), "leaf_id_column"
        ),
        "rule_id_column": _output_column(
            inputs.get("rule_id_column", DEFAULT_RULE_ID_COLUMN), "rule_id_column"
        ),
    }
    activate_result = inputs["activate_result"]
    if not isinstance(activate_result, bool):
        raise StrategyError("activate_result must be boolean")
    normalized["activate_result"] = activate_result
    if (
        normalized["leaf_id_column"].casefold()
        == normalized["rule_id_column"].casefold()
    ):
        raise StrategyError(
            "automatic-tree leaf and rule output columns must be distinct"
        )
    return normalized


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{field} must be a non-empty string")
    return value.strip()


def _required_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{field} must be a lowercase SHA-256 hash")
    return value


def _required_asset_id(value: object) -> str:
    normalized = _required_text(value, "expected_asset_id")
    if _ASSET_ID_RE.fullmatch(normalized) is None:
        raise StrategyError("expected_asset_id is invalid")
    return normalized


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{field} must be a non-negative integer")
    return value


def _output_column(value: object, field: str) -> str:
    normalized = _required_text(value, field)
    if _OUTPUT_COLUMN_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{field} is invalid")
    return normalized


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


__all__ = [
    "DEFAULT_LEAF_ID_COLUMN",
    "DEFAULT_RULE_ID_COLUMN",
    "EVIDENCE_PROVENANCE_SCHEMA_VERSION",
    "RESULT_DATASET_ROLE",
    "TOOL_SCHEMA_VERSION",
    "automatic_tree_apply_evidence_provenance",
    "run_apply_automatic_tree",
]
