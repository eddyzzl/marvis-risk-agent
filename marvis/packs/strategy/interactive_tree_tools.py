"""Governed Tool boundary for immutable interactive-tree prune revisions.

Callers identify only one task-owned automatic tree or prior interactive
revision, one exact visible split node, and an optional audit reason.  The
platform resolves every artifact/hash/sample binding, derives the new frontier
from the authenticated automatic topology, replays it against the live
development population, and publishes one canonical immutable JSON artifact.
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

import numpy as np

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.labels import resolve_labeled_frame
from marvis.feature.weighted_rule_tree import (
    _metrics_bundle,
    _strict_amounts,
    _strict_binary_target,
    _strict_positive_weights,
)
from marvis.files import sha256_file
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
)
from marvis.packs.strategy.automatic_tree_leaf_tools import (
    VerifiedAutomaticTreeSource,
    canonical_automatic_tree_source_path,
    load_verified_automatic_tree_source_artifact_on_connection,
)
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.automatic_tree_tools import (
    _load_binding,
    _require_binding_on_connection,
    automatic_tree_sample_context_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression_frame
from marvis.packs.strategy.interactive_tree_revision import (
    INTERACTIVE_TREE_REVISION_PRODUCER_VERSION,
    InteractiveTreeRevisionError,
    build_interactive_tree_revision,
    canonical_interactive_tree_revision_json,
    validate_interactive_tree_revision,
)
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignExecutionBinding,
    StrategySampleDesignRef,
    bind_strategy_development_frame,
    load_strategy_sample_design_execution_binding,
    require_strategy_sample_design_execution_binding_on_connection,
    revalidate_strategy_sample_design_execution_binding,
)
from marvis.packs.strategy.sample_design_tools import (
    load_strategy_sample_design_artifact,
)


TOOL_SCHEMA_VERSION = "strategy.revise-interactive-tree-tool.v1"
INTERACTIVE_TREE_REVISION_ARTIFACT_KIND = (
    "strategy_interactive_tree_revision_json"
)
INTERACTIVE_TREE_REVISION_ARTIFACT_SCHEMA_VERSION = (
    "strategy.interactive-tree-revision-artifact.v1"
)
INTERACTIVE_TREE_REVISION_ORIGIN_TOOL = "strategy.revise_interactive_tree"
MAX_INTERACTIVE_TREE_REVISION_BYTES = 4 * 1024 * 1024
MAX_INTERACTIVE_TREE_REVISION_CHAIN = 511

_INPUT_FIELDS = frozenset(
    {"source_tree_id", "node_id", "operation", "reason"}
)
_REQUIRED_INPUT_FIELDS = _INPUT_FIELDS - {"reason"}
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "revision_id",
        "revision_hash",
        "semantic_tree_id",
        "tree_hash",
        "base_asset_id",
        "base_asset_hash",
        "base_tree_result_hash",
        "parent_revision_id",
        "source_tree_id",
        "edit_operation",
        "edit_node_id",
        "sample_design_ref",
    }
)
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
_REVISION_ID_RE = re.compile(
    r"^interactive-tree-revision-[0-9a-f]{32}$"
)


@dataclass(frozen=True)
class VerifiedInteractiveTreeRevision:
    """One fully authenticated task-owned interactive revision artifact."""

    artifact_id: str
    task_id: str
    path: Path
    content_hash: str
    provenance: dict[str, Any]
    canonical_bytes: bytes
    revision: dict[str, Any]
    ancestor_revisions: tuple[dict[str, Any], ...]
    automatic_source: VerifiedAutomaticTreeSource


@dataclass(frozen=True)
class _ResolvedRevisionSource:
    automatic_source: VerifiedAutomaticTreeSource
    parent_revision: dict[str, Any] | None
    ancestor_revisions: tuple[dict[str, Any], ...]
    source_tree_id: str


@dataclass(frozen=True)
class _ReplayBinding:
    data_binding: Any
    sample_design: StrategySampleDesignExecutionBinding
    evidence: dict[str, Any]


def run_revise_interactive_tree(inputs: object, ctx, runtime) -> dict[str, Any]:
    """Build, replay, and atomically publish one prune revision."""

    request = _validate_inputs(inputs)
    task_id = _required_text(ctx.task_id, "task_id")
    with runtime.task_artifacts.transaction() as conn:
        source = _resolve_revision_source_on_connection(
            conn,
            runtime=runtime,
            task_id=task_id,
            source_tree_id=request["source_tree_id"],
        )
    revision = _build_revision(source, request=request)
    canonical = canonical_interactive_tree_revision_json(
        revision,
        source.automatic_source.asset,
        parent_revision=source.parent_revision,
        ancestor_revisions=source.ancestor_revisions,
    ).encode("utf-8")
    if len(canonical) > MAX_INTERACTIVE_TREE_REVISION_BYTES:
        raise StrategyError("interactive-tree revision exceeds the JSON byte budget")
    replay = _replay_revision(
        runtime,
        task_id=task_id,
        source=source.automatic_source,
        revision=revision,
    )
    content_hash = hashlib.sha256(canonical).hexdigest()
    provenance = interactive_tree_revision_provenance(
        revision,
        automatic_source=source.automatic_source,
        parent_revision=source.parent_revision,
        ancestor_revisions=source.ancestor_revisions,
    )
    artifact = _persist_revision(
        runtime,
        task_id=task_id,
        request=request,
        source=source,
        revision=revision,
        canonical=canonical,
        content_hash=content_hash,
        provenance=provenance,
        replay=replay,
    )
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "revision_id": revision["revision_id"],
        "revision_hash": revision["revision_hash"],
        "semantic_tree_id": revision["semantic_tree_id"],
        "tree_hash": revision["tree"]["tree_hash"],
        "source_tree_id": source.source_tree_id,
        "base_asset_id": revision["base_tree"]["asset_id"],
        "parent_revision_id": (
            None
            if revision["parent_revision"] is None
            else revision["parent_revision"]["revision_id"]
        ),
        "edit": revision["edit"],
        "visible_node_count": len(revision["tree"]["visible_node_ids"]),
        "frontier_node_count": len(revision["tree"]["frontier_node_ids"]),
        "replay": replay.evidence,
        "artifacts": [artifact],
    }


def canonical_interactive_tree_revision_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    revision_id: str,
) -> Path:
    """Return the sole canonical JSON path without resolving symlinks."""

    normalized_task = _safe_component(task_id, "task_id")
    normalized_revision = _required_text(revision_id, "revision_id")
    if _REVISION_ID_RE.fullmatch(normalized_revision) is None:
        raise StrategyError("revision_id has an invalid format")
    return (
        Path(tasks_dir).absolute()
        / normalized_task
        / "strategy_interactive_tree_revisions"
        / f"{normalized_revision}.json"
    )


def interactive_tree_revision_provenance(
    revision_payload: Mapping[str, Any],
    *,
    automatic_source: VerifiedAutomaticTreeSource,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    """Derive the exact registry provenance from verified evidence."""

    try:
        revision = validate_interactive_tree_revision(
            revision_payload,
            automatic_source.asset,
            parent_revision=parent_revision,
            ancestor_revisions=ancestor_revisions,
        )
    except (InteractiveTreeRevisionError, TypeError, ValueError) as exc:
        raise StrategyError("interactive-tree revision failed validation") from exc
    sample_ref = sample_design_ref_from_automatic_tree_source_refs(
        automatic_source.asset["source_refs"]
    )
    base_tree = revision["base_tree"]
    parent_ref = revision["parent_revision"]
    parent_revision_id = (
        None if parent_ref is None else parent_ref["revision_id"]
    )
    provenance = {
        "schema_version": INTERACTIVE_TREE_REVISION_ARTIFACT_SCHEMA_VERSION,
        "producer_version": INTERACTIVE_TREE_REVISION_PRODUCER_VERSION,
        "task_id": revision["identity"]["task_id"],
        "kind": INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
        "format": "json",
        "revision_id": revision["revision_id"],
        "revision_hash": revision["revision_hash"],
        "semantic_tree_id": revision["semantic_tree_id"],
        "tree_hash": revision["tree"]["tree_hash"],
        "base_asset_id": base_tree["asset_id"],
        "base_asset_hash": base_tree["asset_hash"],
        "base_tree_result_hash": base_tree["tree_result_hash"],
        "parent_revision_id": parent_revision_id,
        "source_tree_id": (
            base_tree["asset_id"]
            if parent_revision_id is None
            else parent_revision_id
        ),
        "edit_operation": revision["edit"]["operation"],
        "edit_node_id": revision["edit"]["node_id"],
        "sample_design_ref": sample_ref,
    }
    if set(provenance) != _PROVENANCE_FIELDS:
        raise StrategyError("interactive-tree revision provenance fields drifted")
    return _canonical_json_object(
        provenance,
        "interactive-tree revision provenance",
    )


def load_verified_interactive_tree_revision(
    runtime,
    *,
    task_id: str,
    revision_id: str,
) -> VerifiedInteractiveTreeRevision:
    """Load one exact revision and recursively authenticate its ancestry."""

    with runtime.task_artifacts.transaction() as conn:
        return _load_verified_interactive_tree_revision_on_connection(
            conn,
            runtime=runtime,
            task_id=task_id,
            revision_id=revision_id,
            visited=set(),
            depth=0,
        )


def _resolve_revision_source_on_connection(
    conn,
    *,
    runtime,
    task_id: str,
    source_tree_id: str,
) -> _ResolvedRevisionSource:
    if _ASSET_ID_RE.fullmatch(source_tree_id):
        source = _load_automatic_source_by_asset_on_connection(
            conn,
            runtime=runtime,
            task_id=task_id,
            asset_id=source_tree_id,
        )
        return _ResolvedRevisionSource(
            automatic_source=source,
            parent_revision=None,
            ancestor_revisions=(),
            source_tree_id=source_tree_id,
        )
    if _REVISION_ID_RE.fullmatch(source_tree_id):
        parent = _load_verified_interactive_tree_revision_on_connection(
            conn,
            runtime=runtime,
            task_id=task_id,
            revision_id=source_tree_id,
            visited=set(),
            depth=0,
        )
        return _ResolvedRevisionSource(
            automatic_source=parent.automatic_source,
            parent_revision=parent.revision,
            ancestor_revisions=parent.ancestor_revisions,
            source_tree_id=source_tree_id,
        )
    raise StrategyError(
        "source_tree_id must identify an automatic tree or interactive revision"
    )


def _load_automatic_source_by_asset_on_connection(
    conn,
    *,
    runtime,
    task_id: str,
    asset_id: str,
) -> VerifiedAutomaticTreeSource:
    path = canonical_automatic_tree_source_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        asset_id=asset_id,
    )
    row = _lookup_artifact_row(
        conn,
        task_id=task_id,
        kind=AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
        path=path,
    )
    provenance = _strict_provenance(row, "automatic-tree source")
    for field in ("asset_id", "asset_hash", "tree_result_hash"):
        if field not in provenance:
            raise StrategyError(
                f"automatic-tree source provenance is missing {field}"
            )
    return load_verified_automatic_tree_source_artifact_on_connection(
        conn,
        tasks_dir=runtime.settings.tasks_dir,
        task_id=task_id,
        artifact_id=_required_text(row["id"], "automatic-tree artifact id"),
        expected_content_hash=_required_hash(
            row["content_hash"],
            "automatic-tree artifact content_hash",
        ),
        expected_asset_id=_required_text(
            provenance["asset_id"],
            "automatic-tree provenance asset_id",
        ),
        expected_asset_hash=_required_hash(
            provenance["asset_hash"],
            "automatic-tree provenance asset_hash",
        ),
        expected_tree_result_hash=_required_hash(
            provenance["tree_result_hash"],
            "automatic-tree provenance tree_result_hash",
        ),
    )


def _load_verified_interactive_tree_revision_on_connection(
    conn,
    *,
    runtime,
    task_id: str,
    revision_id: str,
    visited: set[str],
    depth: int,
) -> VerifiedInteractiveTreeRevision:
    normalized_revision = _required_text(revision_id, "revision_id")
    if _REVISION_ID_RE.fullmatch(normalized_revision) is None:
        raise StrategyError("interactive-tree revision_id has an invalid format")
    if depth >= MAX_INTERACTIVE_TREE_REVISION_CHAIN:
        raise StrategyError("interactive-tree revision chain exceeds the depth budget")
    if normalized_revision in visited:
        raise StrategyError("interactive-tree revision chain contains a cycle")
    visited.add(normalized_revision)
    try:
        path = canonical_interactive_tree_revision_path(
            runtime.settings.tasks_dir,
            task_id=task_id,
            revision_id=normalized_revision,
        )
        row = _lookup_artifact_row(
            conn,
            task_id=task_id,
            kind=INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
            path=path,
        )
        if row["origin_tool"] != INTERACTIVE_TREE_REVISION_ORIGIN_TOOL:
            raise StrategyError(
                "interactive-tree revision artifact origin_tool is invalid"
            )
        provenance = _strict_provenance(row, "interactive-tree revision")
        _require_exact_fields(
            provenance,
            _PROVENANCE_FIELDS,
            "interactive-tree revision provenance",
        )
        _require_revision_provenance_scalars(
            provenance,
            task_id=task_id,
            revision_id=normalized_revision,
        )
        source = _load_automatic_source_by_asset_on_connection(
            conn,
            runtime=runtime,
            task_id=task_id,
            asset_id=provenance["base_asset_id"],
        )
        parent_id = provenance["parent_revision_id"]
        parent_binding = (
            None
            if parent_id is None
            else _load_verified_interactive_tree_revision_on_connection(
                conn,
                runtime=runtime,
                task_id=task_id,
                revision_id=parent_id,
                visited=visited,
                depth=depth + 1,
            )
        )
        if (
            parent_binding is not None
            and parent_binding.automatic_source.asset["asset_id"]
            != source.asset["asset_id"]
        ):
            raise StrategyError(
                "interactive-tree revision parent uses another automatic base"
            )
        registered_hash = _required_hash(
            row["content_hash"],
            "interactive-tree revision content_hash",
        )
        canonical_bytes = _read_verified_regular_file(
            path,
            root=Path(runtime.settings.tasks_dir).absolute(),
            expected_hash=registered_hash,
            max_bytes=MAX_INTERACTIVE_TREE_REVISION_BYTES,
        )
        payload = _strict_json_object_from_bytes(
            canonical_bytes,
            "interactive-tree revision artifact",
        )
        try:
            revision = validate_interactive_tree_revision(
                payload,
                source.asset,
                parent_revision=(
                    None if parent_binding is None else parent_binding.revision
                ),
                ancestor_revisions=(
                    ()
                    if parent_binding is None
                    else parent_binding.ancestor_revisions
                ),
            )
            canonical = canonical_interactive_tree_revision_json(
                revision,
                source.asset,
                parent_revision=(
                    None if parent_binding is None else parent_binding.revision
                ),
                ancestor_revisions=(
                    ()
                    if parent_binding is None
                    else parent_binding.ancestor_revisions
                ),
            ).encode("utf-8")
        except (InteractiveTreeRevisionError, TypeError, ValueError) as exc:
            raise StrategyError(
                "interactive-tree revision artifact failed validation"
            ) from exc
        if not hmac.compare_digest(canonical_bytes, canonical):
            raise StrategyError(
                "interactive-tree revision artifact is not canonical JSON"
            )
        expected_provenance = interactive_tree_revision_provenance(
            revision,
            automatic_source=source,
            parent_revision=(
                None if parent_binding is None else parent_binding.revision
            ),
            ancestor_revisions=(
                ()
                if parent_binding is None
                else parent_binding.ancestor_revisions
            ),
        )
        if not hmac.compare_digest(
            _canonical_json(provenance),
            _canonical_json(expected_provenance),
        ):
            raise StrategyError(
                "interactive-tree revision artifact provenance changed"
            )
        if revision["revision_id"] != normalized_revision:
            raise StrategyError("interactive-tree revision identity changed")
        return VerifiedInteractiveTreeRevision(
            artifact_id=_required_text(row["id"], "revision artifact id"),
            task_id=task_id,
            path=path,
            content_hash=registered_hash,
            provenance=provenance,
            canonical_bytes=canonical_bytes,
            revision=revision,
            ancestor_revisions=(
                ()
                if parent_binding is None
                else (
                    parent_binding.revision,
                    *parent_binding.ancestor_revisions,
                )
            ),
            automatic_source=source,
        )
    finally:
        visited.remove(normalized_revision)


def _build_revision(
    source: _ResolvedRevisionSource,
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return build_interactive_tree_revision(
            source.automatic_source.asset,
            node_id=request["node_id"],
            reason=request.get("reason"),
            parent_revision=source.parent_revision,
            ancestor_revisions=source.ancestor_revisions,
        )
    except (InteractiveTreeRevisionError, TypeError, ValueError) as exc:
        raise StrategyError("interactive-tree revision edit is invalid") from exc


def _replay_revision(
    runtime,
    *,
    task_id: str,
    source: VerifiedAutomaticTreeSource,
    revision: Mapping[str, Any],
) -> _ReplayBinding:
    asset = source.asset
    identity = asset["identity"]
    training = asset["tree_result"]["training"]
    sample_ref = StrategySampleDesignRef.from_value(
        sample_design_ref_from_automatic_tree_source_refs(asset["source_refs"])
    )
    sample_artifact = load_strategy_sample_design_artifact(
        runtime,
        task_id=task_id,
        artifact_id=sample_ref.artifact_id,
        expected_artifact_content_hash=sample_ref.artifact_content_hash,
        expected_sample_design_id=sample_ref.sample_design_id,
        expected_sample_design_content_hash=sample_ref.sample_design_content_hash,
    )
    design = sample_artifact.bundle["sample_design"]
    target_definition = design["target_definition"]
    optional = design["optional_fields"]
    weight_col = (
        None
        if training["sample_weight"]["status"] == "not_applicable"
        else training["sample_weight"]["column"]
    )
    expected_optional = {
        "weight_field": weight_col,
        "loan_amount_field": training["loan_amount_col"],
        "overdue_amount_field": training["overdue_amount_col"],
    }
    for field, expected in expected_optional.items():
        if optional[field] != expected:
            raise StrategyError(
                f"interactive-tree sample-design {field} changed from the base tree"
            )
    data_binding = _load_binding(
        runtime,
        task_id=task_id,
        dataset_id=identity["dataset_id"],
        expected_content_hash=identity["dataset_content_hash"],
        expected_revision=identity["workspace_revision"],
        expected_generation=identity["workspace_generation"],
        expected_semantic_hash=identity["semantic_mapping_hash"],
    )
    if not hmac.compare_digest(
        data_binding.registry_metadata_hash,
        identity["registry_metadata_hash"],
    ):
        raise StrategyError(
            "interactive-tree dataset registry metadata changed from the base tree"
        )
    sample_design = load_strategy_sample_design_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=sample_ref.to_ref_dict(),
        dataset_id=identity["dataset_id"],
        dataset_content_hash=identity["dataset_content_hash"],
        workspace_revision=identity["workspace_revision"],
        workspace_generation=identity["workspace_generation"],
        semantic_mapping_hash=identity["semantic_mapping_hash"],
        target_col=training["target_col"],
        drop_nan_labels=target_definition["drop_nan_labels"],
        month_col=optional["month_field"],
        weight_col=weight_col,
        loan_amount_col=training["loan_amount_col"],
        overdue_amount_col=training["overdue_amount_col"],
    )
    projected = list(training["feature_order"])
    for column in (
        training["target_col"],
        weight_col,
        training["loan_amount_col"],
        training["overdue_amount_col"],
        sample_design.split_column,
    ):
        if column is not None and column not in projected:
            projected.append(column)
    frame = runtime.backend.read_frame(data_binding.path, columns=projected)
    revalidate_strategy_sample_design_execution_binding(runtime, sample_design)
    if not hmac.compare_digest(
        sha256_file(data_binding.path),
        data_binding.content_hash,
    ):
        raise StrategyError("interactive-tree source dataset bytes changed")
    frame = bind_strategy_development_frame(frame, binding=sample_design)
    labeled, dropped = resolve_labeled_frame(
        frame,
        training["target_col"],
        drop_nan_labels=sample_design.drop_nan_labels,
        scope="interactive-tree development sample",
    )
    labeled = labeled.reset_index(drop=True)
    if len(labeled) != int(training["row_count"]):
        raise StrategyError(
            "interactive-tree development row count changed from the base tree"
        )
    context_hash = automatic_tree_sample_context_hash(
        task_id=task_id,
        binding=data_binding,
        target_col=training["target_col"],
        labeled_row_count=len(labeled),
        drop_nan_labels=sample_design.drop_nan_labels,
        nan_labels_dropped=dropped,
        loan_amount_col=training["loan_amount_col"],
        overdue_amount_col=training["overdue_amount_col"],
        sample_design_ref=sample_design.to_ref_dict(),
    )
    if not hmac.compare_digest(context_hash, identity["sample_context_hash"]):
        raise StrategyError(
            "interactive-tree development sample context changed from the base tree"
        )

    masks = [
        evaluate_expression_frame(labeled, fragment["condition"]).to_numpy(
            dtype=bool,
            copy=False,
        )
        for fragment in revision["fragments"]
    ]
    assignment_count = np.zeros(len(labeled), dtype=np.int16)
    for mask in masks:
        assignment_count += mask.astype(np.int16, copy=False)
    if len(labeled) == 0 or not np.all(assignment_count == 1):
        raise StrategyError(
            "interactive-tree frontier must assign every development row exactly once"
        )

    target = _strict_binary_target(
        labeled[training["target_col"]],
        column=training["target_col"],
    )
    weights = (
        None
        if weight_col is None
        else _strict_positive_weights(labeled[weight_col], column=weight_col)
    )
    loan_values = (
        None
        if training["loan_amount_col"] is None
        else _strict_amounts(
            labeled[training["loan_amount_col"]],
            column=training["loan_amount_col"],
        )
    )
    overdue_values = (
        None
        if training["overdue_amount_col"] is None
        else _strict_amounts(
            labeled[training["overdue_amount_col"]],
            column=training["overdue_amount_col"],
        )
    )
    root_mask = np.ones(len(labeled), dtype=bool)
    assigned_leaf_ids: list[str | None] = [None] * len(labeled)
    for fragment, mask in zip(revision["fragments"], masks, strict=True):
        replayed_metrics = _metrics_bundle(
            mask,
            target,
            weights=weights,
            root_mask=root_mask,
            loan_values=loan_values,
            overdue_values=overdue_values,
        )
        if not hmac.compare_digest(
            _canonical_json(replayed_metrics),
            _canonical_json(fragment["metrics"]),
        ):
            raise StrategyError(
                "interactive-tree fragment metrics do not replay against the "
                "development sample"
            )
        for index in np.flatnonzero(mask):
            assigned_leaf_ids[int(index)] = fragment["leaf_id"]
    evidence_body = {
        "schema_version": "strategy.interactive-tree-replay.v1",
        "partition": "development",
        "source_row_count": len(labeled),
        "frontier_count": len(masks),
        "exactly_once": True,
        "metrics_matched": True,
        "assignment_hash": _canonical_sha256(assigned_leaf_ids),
        "sample_context_hash": context_hash,
    }
    evidence = {
        **evidence_body,
        "result_hash": _canonical_sha256(evidence_body),
    }
    return _ReplayBinding(
        data_binding=data_binding,
        sample_design=sample_design,
        evidence=evidence,
    )


def _persist_revision(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    source: _ResolvedRevisionSource,
    revision: Mapping[str, Any],
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
    replay: _ReplayBinding,
) -> dict[str, Any]:
    revision_id = _required_text(revision["revision_id"], "revision_id")
    out_dir = _prepare_revision_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = canonical_interactive_tree_revision_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        revision_id=revision_id,
    )
    if final_path.parent != out_dir:
        raise StrategyError("interactive-tree revision output path drifted")
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
                locked_source = _resolve_revision_source_on_connection(
                    conn,
                    runtime=runtime,
                    task_id=task_id,
                    source_tree_id=request["source_tree_id"],
                )
                locked_revision = _build_revision(
                    locked_source,
                    request=request,
                )
                locked_canonical = canonical_interactive_tree_revision_json(
                    locked_revision,
                    locked_source.automatic_source.asset,
                    parent_revision=locked_source.parent_revision,
                    ancestor_revisions=locked_source.ancestor_revisions,
                ).encode("utf-8")
                locked_provenance = interactive_tree_revision_provenance(
                    locked_revision,
                    automatic_source=locked_source.automatic_source,
                    parent_revision=locked_source.parent_revision,
                    ancestor_revisions=locked_source.ancestor_revisions,
                )
                if (
                    locked_source != source
                    or locked_revision != revision
                    or not hmac.compare_digest(locked_canonical, canonical)
                    or not hmac.compare_digest(
                        _canonical_json(locked_provenance),
                        _canonical_json(provenance),
                    )
                ):
                    raise StrategyError(
                        "interactive-tree revision evidence changed before registration"
                    )
                _require_binding_on_connection(
                    conn,
                    task_id=task_id,
                    binding=replay.data_binding,
                )
                require_strategy_sample_design_execution_binding_on_connection(
                    conn,
                    replay.sample_design,
                )
                if not hmac.compare_digest(
                    sha256_file(replay.data_binding.path),
                    replay.data_binding.content_hash,
                ):
                    raise StrategyError(
                        "interactive-tree source dataset changed before registration"
                    )
                _prepare_revision_directory(
                    runtime.settings.tasks_dir,
                    task_id=task_id,
                )
                existing = _lookup_optional_artifact_row(
                    conn,
                    task_id=task_id,
                    kind=INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
                    path=final_path,
                )
                if existing is not None:
                    _verify_registered_revision(
                        existing,
                        task_id=task_id,
                        final_path=final_path,
                        content_hash=content_hash,
                        provenance=provenance,
                        canonical=canonical,
                    )
                    record = existing
                    uow.rollback()
                else:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "interactive-tree revision path exists without a registry row"
                        )
                    uow.promote_all()
                    _verify_revision_file(
                        final_path,
                        root=Path(runtime.settings.tasks_dir).absolute(),
                        expected_content=canonical,
                        expected_hash=content_hash,
                    )
                    record = runtime.task_artifacts.register_on_connection(
                        conn,
                        task_id=task_id,
                        kind=INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
                        path=str(final_path),
                        content_hash=content_hash,
                        origin_tool=INTERACTIVE_TREE_REVISION_ORIGIN_TOOL,
                        provenance=provenance,
                    )
                    _verify_registered_revision(
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
    artifact_id = _required_text(record["id"], "revision artifact id")
    return {
        "artifact_id": artifact_id,
        "kind": INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
        "format": "json",
        "filename": final_path.name,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        ),
    }


def _lookup_artifact_row(
    conn,
    *,
    task_id: str,
    kind: str,
    path: Path,
) -> dict[str, Any]:
    row = _lookup_optional_artifact_row(
        conn,
        task_id=task_id,
        kind=kind,
        path=path,
    )
    if row is None:
        raise StrategyError(f"{kind} artifact was not found for this task")
    return row


def _lookup_optional_artifact_row(
    conn,
    *,
    task_id: str,
    kind: str,
    path: Path,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, kind, str(path)),
    ).fetchone()
    if row is None:
        return None
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _require_exact_fields(
        record,
        _TASK_ARTIFACT_ROW_FIELDS,
        "task artifact registry row",
    )
    return record


def _strict_provenance(
    row: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    raw = row.get("provenance_json")
    if not isinstance(raw, str):
        raise StrategyError(f"{name} provenance_json is invalid")
    return _strict_json_object_from_text(raw, f"{name} provenance_json")


def _require_revision_provenance_scalars(
    provenance: Mapping[str, Any],
    *,
    task_id: str,
    revision_id: str,
) -> None:
    expected = {
        "schema_version": INTERACTIVE_TREE_REVISION_ARTIFACT_SCHEMA_VERSION,
        "producer_version": INTERACTIVE_TREE_REVISION_PRODUCER_VERSION,
        "task_id": task_id,
        "kind": INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
        "format": "json",
        "revision_id": revision_id,
        "edit_operation": "prune_subtree",
    }
    for field, value in expected.items():
        if provenance[field] != value:
            raise StrategyError(
                f"interactive-tree revision provenance {field} changed"
            )
    for field in (
        "revision_hash",
        "tree_hash",
        "base_asset_hash",
        "base_tree_result_hash",
    ):
        _required_hash(provenance[field], f"revision provenance {field}")
    if _ASSET_ID_RE.fullmatch(
        _required_text(provenance["base_asset_id"], "base_asset_id")
    ) is None:
        raise StrategyError(
            "interactive-tree revision provenance base_asset_id is invalid"
        )
    _required_text(provenance["semantic_tree_id"], "semantic_tree_id")
    _required_text(provenance["source_tree_id"], "source_tree_id")
    _required_text(provenance["edit_node_id"], "edit_node_id")
    parent = provenance["parent_revision_id"]
    if parent is not None and (
        not isinstance(parent, str) or _REVISION_ID_RE.fullmatch(parent) is None
    ):
        raise StrategyError(
            "interactive-tree revision provenance parent_revision_id is invalid"
        )
    StrategySampleDesignRef.from_value(provenance["sample_design_ref"])


def _verify_registered_revision(
    record: Mapping[str, Any],
    *,
    task_id: str,
    final_path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
    canonical: bytes,
    registry_record: bool = False,
) -> None:
    expected = {
        "task_id": task_id,
        "kind": INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
        "path": str(final_path),
        "content_hash": content_hash,
        "origin_tool": INTERACTIVE_TREE_REVISION_ORIGIN_TOOL,
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
                f"interactive-tree revision registry {field} changed"
            )
    if registry_record:
        actual_provenance = _canonical_json_object(
            record.get("provenance"),
            "interactive-tree revision registry provenance",
        )
    else:
        actual_provenance = _strict_provenance(
            record,
            "interactive-tree revision",
        )
    if not hmac.compare_digest(
        _canonical_json(actual_provenance),
        _canonical_json(provenance),
    ):
        raise StrategyError(
            "interactive-tree revision registry provenance changed"
        )
    _verify_revision_file(
        final_path,
        root=final_path.parents[2],
        expected_content=canonical,
        expected_hash=content_hash,
    )


def _prepare_revision_directory(
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
        out_dir = task_dir / "strategy_interactive_tree_revisions"
        if out_dir.is_symlink():
            raise StrategyError(
                "interactive-tree revision directory must not be a symlink"
            )
        out_dir.mkdir(exist_ok=True)
        if out_dir.resolve(strict=True).parent != task_dir.resolve(strict=True):
            raise StrategyError(
                "interactive-tree revision directory escaped task storage"
            )
    except OSError as exc:
        raise StrategyError(
            "interactive-tree revision directory is unavailable"
        ) from exc
    return out_dir


def _verify_revision_file(
    path: Path,
    *,
    root: Path,
    expected_content: bytes,
    expected_hash: str,
) -> None:
    persisted = _read_verified_regular_file(
        path,
        root=root,
        expected_hash=expected_hash,
        max_bytes=MAX_INTERACTIVE_TREE_REVISION_BYTES,
    )
    if not hmac.compare_digest(persisted, expected_content):
        raise StrategyError("interactive-tree revision artifact bytes changed")


def _read_verified_regular_file(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
    max_bytes: int,
) -> bytes:
    _require_regular_path(path, root=root)
    before = path.lstat()
    if int(before.st_size) > max_bytes:
        raise StrategyError("interactive-tree revision artifact exceeds byte budget")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise StrategyError(
            "interactive-tree revision artifact could not be read"
        ) from exc
    _require_regular_path(path, root=root)
    after = path.lstat()
    if _stat_identity(before) != _stat_identity(after):
        raise StrategyError(
            "interactive-tree revision artifact changed while read"
        )
    if len(payload) > max_bytes:
        raise StrategyError("interactive-tree revision artifact exceeds byte budget")
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_hash):
        raise StrategyError(
            "interactive-tree revision artifact content hash drifted"
        )
    return payload


def _require_regular_path(path: Path, *, root: Path) -> None:
    if not path.is_absolute():
        raise StrategyError("interactive-tree artifact path must be absolute")
    declared_root = root.absolute()
    try:
        relative = path.relative_to(declared_root)
    except ValueError as exc:
        raise StrategyError(
            "interactive-tree artifact path escapes task storage"
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
                    "interactive-tree artifact path has a symlink ancestor"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise StrategyError(
                    "interactive-tree artifact path ancestor is not a directory"
                )
        metadata = chain[-1].lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise StrategyError(
                "interactive-tree artifact path must not be a symlink"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise StrategyError(
                "interactive-tree artifact path is not a regular file"
            )
        resolved_root = declared_root.resolve(strict=True)
        path.resolve(strict=True).relative_to(resolved_root)
    except StrategyError:
        raise
    except FileNotFoundError as exc:
        raise StrategyError(
            "interactive-tree artifact path is not a regular file"
        ) from exc
    except OSError as exc:
        raise StrategyError(
            "interactive-tree artifact path is unavailable"
        ) from exc
    except ValueError as exc:
        raise StrategyError(
            "interactive-tree artifact path escapes task storage"
        ) from exc


def _validate_inputs(inputs: object) -> dict[str, Any]:
    if not isinstance(inputs, Mapping) or any(
        not isinstance(key, str) for key in inputs
    ):
        raise StrategyError("revise_interactive_tree inputs must be an object")
    actual = set(inputs)
    missing = sorted(_REQUIRED_INPUT_FIELDS - actual)
    unexpected = sorted(actual - _INPUT_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(
            "invalid revise_interactive_tree inputs ("
            + "; ".join(details)
            + ")"
        )
    source_tree_id = _required_text(inputs["source_tree_id"], "source_tree_id")
    node_id = _required_text(inputs["node_id"], "node_id")
    operation = _required_text(inputs["operation"], "operation")
    if operation != "prune_subtree":
        raise StrategyError("operation must be prune_subtree")
    normalized: dict[str, Any] = {
        "source_tree_id": source_tree_id,
        "node_id": node_id,
        "operation": operation,
    }
    if "reason" in inputs:
        reason = inputs["reason"]
        if reason is not None:
            if not isinstance(reason, str) or "\x00" in reason:
                raise StrategyError("reason must be text or null")
            reason = " ".join(reason.split())
            if not reason:
                raise StrategyError("reason must not be blank")
            if len(reason) > 500:
                raise StrategyError("reason must be at most 500 characters")
        normalized["reason"] = reason
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


def _required_hash(value: object, name: str) -> str:
    normalized = _required_text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256")
    return normalized


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(str(field) for field in actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported " + ", ".join(unexpected))
        raise StrategyError(f"{name} has " + "; ".join(details))


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
                raise StrategyError(
                    f"{name} contains a duplicate JSON key: {key}"
                )
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


def _canonical_json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise StrategyError(f"{name} must be a JSON object")
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} must contain finite JSON") from exc


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
            "interactive-tree evidence must be finite canonical JSON"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stat_identity(value) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


__all__ = [
    "INTERACTIVE_TREE_REVISION_ARTIFACT_KIND",
    "INTERACTIVE_TREE_REVISION_ARTIFACT_SCHEMA_VERSION",
    "INTERACTIVE_TREE_REVISION_ORIGIN_TOOL",
    "MAX_INTERACTIVE_TREE_REVISION_BYTES",
    "MAX_INTERACTIVE_TREE_REVISION_CHAIN",
    "TOOL_SCHEMA_VERSION",
    "VerifiedInteractiveTreeRevision",
    "canonical_interactive_tree_revision_path",
    "interactive_tree_revision_provenance",
    "load_verified_interactive_tree_revision",
    "run_revise_interactive_tree",
]
