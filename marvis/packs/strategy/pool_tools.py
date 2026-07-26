"""Governed tool boundary for task-owned Strategy Candidate Pools.

The pure pool kernel owns membership, ordering, and typed actions.  This module
owns immutable artifact lineage, optimistic compare-and-swap, and the single
SQLite/file unit of work used to advance a pool revision.  Compilation is a
read-only design projection: it never evaluates a dataset or creates a
``strategies`` row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import DatasetContentDriftError
from marvis.output.strategy_candidate_report import (
    canonical_strategy_candidate_report_json,
    strategy_candidate_report_from_json,
)
from marvis.packs.strategy.candidate_asset import (
    canonical_candidate_asset_json,
    validate_candidate_asset,
)
from marvis.packs.strategy.candidate_asset_tools import (
    ASSET_ARTIFACT_KIND,
    ASSET_ARTIFACT_SCHEMA_VERSION,
    ORIGIN_TOOL as ASSET_ORIGIN_TOOL,
    _load_dataset_binding,
    _load_source_artifact,
    _normalize_source_record,
    _require_asset_binding,
    _require_dataset_on_connection,
    _require_file_content_hash,
    _require_regular_artifact_path,
    _require_report_binding,
    _registry_metadata_hash_on_connection,
    _require_source_on_connection,
)
from marvis.packs.strategy.candidate_evidence import validate_candidate_evidence
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
    univariate_asset_to_verified_fragment,
    verified_fragment_pool_parts,
)
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    automatic_tree_leaf_fragment_to_verified_candidate_fragment,
)
from marvis.packs.strategy.automatic_tree_leaf_tools import (
    VerifiedAutomaticTreeLeafSelection,
    VerifiedAutomaticTreeSource,
    load_verified_automatic_tree_leaf_selection_artifact,
    load_verified_automatic_tree_leaf_selection_artifact_on_connection,
    load_verified_automatic_tree_source_artifact,
    load_verified_automatic_tree_source_artifact_on_connection,
)
from marvis.packs.strategy.cross_matrix_candidate import (
    rebuild_cross_matrix_candidate_asset,
)
from marvis.packs.strategy.cross_matrix_candidate_tools import (
    ASSET_ARTIFACT_KIND as CROSS_MATRIX_ASSET_ARTIFACT_KIND,
    ASSET_ARTIFACT_SCHEMA_VERSION as CROSS_MATRIX_ASSET_ARTIFACT_SCHEMA_VERSION,
    ASSET_ARTIFACT_V2_SCHEMA_VERSION as CROSS_MATRIX_ASSET_ARTIFACT_V2_SCHEMA_VERSION,
    ORIGIN_TOOL as CROSS_MATRIX_ASSET_ORIGIN_TOOL,
)
from marvis.packs.strategy.cross_matrix_cell_selection import (
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION,
    CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
    cross_matrix_cell_selection_to_verified_candidate_fragment,
)
from marvis.packs.strategy.cross_matrix_cell_selection_tools import (
    VerifiedCrossMatrixCellSelection,
    VerifiedCrossMatrixSource,
    load_verified_cross_matrix_cell_selection_artifact,
    load_verified_cross_matrix_cell_selection_artifact_on_connection,
    load_verified_cross_matrix_source_artifact,
    load_verified_cross_matrix_source_artifact_on_connection,
)
from marvis.packs.strategy.scorecard_candidate import (
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    SCORECARD_BAND_ASSET_TYPE,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    scorecard_cutoff_selection_to_verified_candidate_fragment,
)
from marvis.packs.strategy.scorecard_candidate_tools import (
    ScorecardBandAssetArtifactBinding,
    ScorecardCutoffSelectionArtifactBinding,
    load_scorecard_cutoff_selection_artifact,
    require_scorecard_cutoff_selection_artifact_binding_on_connection,
)
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_TYPE,
    verify_voting_candidate_asset_against_pool,
)
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1,
    VOTING_CANDIDATE_ORIGIN_TOOL,
    voting_candidate_to_verified_fragment,
)
from marvis.packs.strategy.voting_candidate_tools import (
    VerifiedVotingCandidateArtifact,
    canonical_voting_candidate_artifact_json,
    load_verified_voting_candidate_artifact,
    load_verified_voting_candidate_artifact_on_connection,
    require_voting_snapshot_marginal_reachability,
    validate_voting_candidate_artifact_document,
)
from marvis.packs.strategy.errors import (
    StrategyError,
    StrategyPoolLegacyDraftNeedsRebuildError,
)
from marvis.packs.strategy.pool import (
    APPEND_PLACEMENT,
    BEFORE_SELECTED_MEMBERS_PLACEMENT,
    POOL_PRODUCER_VERSION,
    REPLACE_SELECTED_MEMBERS_PLACEMENT,
    add_verified_candidate_fragment,
    canonical_strategy_pool_json,
    compile_strategy_pool,
    remove_pool_entry,
    reorder_strategy_pool,
    set_pool_entry_action,
    validate_strategy_pool,
)
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignExecutionBinding,
    StrategySampleDesignRef,
    load_strategy_sample_design_execution_binding,
    require_strategy_sample_design_execution_binding_on_connection,
)
from marvis.packs.strategy.sample_design_tools import (
    load_strategy_sample_design_artifact,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
    require_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.repositories.strategy_pool import (
    ABSENT_POOL_REVISION,
    ABSENT_POOL_SNAPSHOT_HASH,
    POOL_ARTIFACT_KIND,
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    StrategyCandidatePoolRepository,
    canonical_strategy_pool_snapshot_json,
    strategy_pool_artifact_content_hash,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


POOL_ARTIFACT_SCHEMA_VERSION = "strategy.candidate-pool-artifact.v2"
POOL_MUTATION_TOOL_SCHEMA_VERSION = "strategy.candidate-pool-mutation-tool.v2"
POOL_COMPILE_TOOL_SCHEMA_VERSION = "strategy.compile-candidate-pool-tool.v2"
SCORECARD_REPORT_PROJECTION_SCHEMA_VERSION = (
    "strategy.scorecard-report-projection.v1"
)
MAX_SCORECARD_REPORT_MODELS = 256
MAX_SCORECARD_REPORT_USAGES = 4_096
MAX_SCORECARD_REPORT_DETAIL_ROWS = 120_000
# The final V2 report has global 100k-field / 20k-ref / 16 MiB caps and
# contains six other sections.  These scorecard-only reservations make the
# public projection fail before an adapter can construct an undeliverable
# report; they never trim or sample evidence.
MAX_SCORECARD_REPORT_TABLE_FIELDS = 50_000
MAX_SCORECARD_REPORT_TABLE_REFS = 10_000
MAX_SCORECARD_REPORT_PROJECTION_JSON_BYTES = 4 * 1024 * 1024
_LEGACY_ARCHIVE_WARNING = (
    "A draft Strategy Pool v1 ledger was archived unchanged; this v2 Pool is "
    "a separate rebuild and does not claim v1 revision continuity."
)

_ADD_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "strategy_type",
        "default_action",
        "action",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "reason",
        "placement_mode",
    }
)
_ADD_REQUIRED_FIELDS = _ADD_INPUT_FIELDS - {"reason", "placement_mode"}
_REMOVE_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "rule_id",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "reason",
    }
)
_REMOVE_REQUIRED_FIELDS = _REMOVE_INPUT_FIELDS - {"reason"}
_SET_ACTION_INPUT_FIELDS = _REMOVE_INPUT_FIELDS | {"action"}
_SET_ACTION_REQUIRED_FIELDS = _SET_ACTION_INPUT_FIELDS - {"reason"}
_REORDER_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "ordered_rule_ids",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "reason",
    }
)
_REORDER_REQUIRED_FIELDS = _REORDER_INPUT_FIELDS - {"reason"}
_COMPILE_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
    }
)
_ASSET_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "asset_id",
        "asset_hash",
        "candidate_id",
        "evidence_hash",
        "source_artifact_id",
        "source_artifact_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "feature",
        "method",
    }
)
_POOL_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "pool_id",
        "strategy_type",
        "revision",
        "revision_id",
        "parent_revision_id",
        "snapshot_hash",
        "operation_kind",
        "source_artifact_ids",
        "evidence_identity",
    }
)
_ORIGIN_BY_OPERATION = {
    "add_candidate": "strategy.add_candidate_to_pool",
    "insert_candidate_before_entries": "strategy.add_candidate_to_pool",
    "replace_entries_with_candidate": "strategy.add_candidate_to_pool",
    "remove_entry": "strategy.remove_pool_entry",
    "set_entry_action": "strategy.set_pool_entry_action",
    "reorder_entries": "strategy.reorder_strategy_pool",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_VOTING_ANCESTRY_DEPTH = 16
_MAX_VOTING_ANCESTRY_NODES = 256
_MAX_POOL_ARTIFACT_BYTES = 64 * 1024 * 1024
_BOUNDARY_ERRORS = (
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _UnivariateCandidateLineage:
    asset_record: Any
    asset: dict[str, Any]
    parent_record: Any
    evidence: dict[str, Any]
    dataset: Any
    verified_fragment: dict[str, Any]
    source_binding: dict[str, Any]


@dataclass(frozen=True)
class _AutomaticTreeDatasetBinding:
    dataset_id: str
    task_id: str
    source_path: str
    path: Path
    content_hash: str
    registry_metadata_hash: str
    columns: tuple[str, ...]
    row_count: int


@dataclass(frozen=True)
class _AutomaticTreeCandidateLineage:
    selection: VerifiedAutomaticTreeLeafSelection
    tree: VerifiedAutomaticTreeSource
    dataset: _AutomaticTreeDatasetBinding
    verified_fragment: dict[str, Any]
    source_binding: dict[str, Any]


@dataclass(frozen=True)
class _CrossMatrixCandidateLineage:
    selection: VerifiedCrossMatrixCellSelection
    matrix: VerifiedCrossMatrixSource
    parent_record: Any
    evidence: dict[str, Any]
    dataset: Any
    verified_fragment: dict[str, Any]
    source_binding: dict[str, Any]


@dataclass(frozen=True)
class _ScorecardDatasetBinding:
    dataset_id: str
    task_id: str
    source_path: str
    path: Path
    content_hash: str
    registry_metadata_hash: str
    columns: tuple[str, ...]
    row_count: int


@dataclass(frozen=True)
class _ScorecardCandidateLineage:
    selection: ScorecardCutoffSelectionArtifactBinding
    asset: ScorecardBandAssetArtifactBinding
    dataset: _ScorecardDatasetBinding
    verified_fragment: dict[str, Any]
    source_binding: dict[str, Any]


@dataclass(frozen=True)
class _VotingCandidateLineage:
    candidate: VerifiedVotingCandidateArtifact
    parent_pool: dict[str, Any]
    parent_pool_artifact: Any
    parent_lineages: tuple[_CandidateLineage, ...]
    verified_fragment: dict[str, Any]
    source_binding: dict[str, Any]


_CandidateLineage = (
    _UnivariateCandidateLineage
    | _AutomaticTreeCandidateLineage
    | _CrossMatrixCandidateLineage
    | _ScorecardCandidateLineage
    | _VotingCandidateLineage
)


@dataclass(frozen=True)
class StrategyCandidatePoolArtifactBinding:
    """Authenticated current Pool, compiled design, and all source lineages.

    The nested domain payloads remain ordinary canonical JSON objects so the
    deterministic kernels can consume them directly.  Downstream writers must
    call :func:`require_strategy_candidate_pool_artifact_binding_on_connection`
    while holding their own transaction; that seam detects any in-memory,
    registry, file, head, or upstream-lineage drift.
    """

    task_id: str
    strategy_type: str
    pool: dict[str, Any]
    compiled_design: dict[str, Any]
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    artifact_origin_tool: str
    artifact_provenance: dict[str, Any]
    artifact_provenance_json: str
    lineages: tuple[_CandidateLineage, ...]
    tasks_root: Path
    datasets_root: Path
    db_path: Path


@dataclass(frozen=True)
class StrategyPoolDevelopmentDatasetBinding:
    """Public immutable dataset projection shared by every Pool lineage."""

    task_id: str
    dataset_id: str
    source_path: str
    path: Path
    content_hash: str
    registry_metadata_hash: str
    columns: tuple[str, ...]
    row_count: int


@dataclass(frozen=True)
class StrategyPoolDevelopmentExecutionBinding:
    """One exact governed development universe for an authenticated Pool.

    Consumers intentionally receive no private candidate-lineage classes.
    The Pool boundary recursively authenticates concrete univariate, tree,
    Cross, scorecard, and Voting sources, then proves that all of them share
    this dataset, target, sample design, and evidence identity.
    """

    task_id: str
    pool: StrategyCandidatePoolArtifactBinding
    dataset: StrategyPoolDevelopmentDatasetBinding
    sample_design: StrategySampleDesignExecutionBinding
    sample_design_v2: StrategySampleDesignV2ArtifactBinding | None
    evidence_identity: dict[str, Any]
    target_col: str
    month_col: str | None


@dataclass(frozen=True)
class _PoolDevelopmentLineageFacts:
    dataset: StrategyPoolDevelopmentDatasetBinding
    sample_ref: StrategySampleDesignRef
    sample_design_v2: StrategySampleDesignV2ArtifactBinding | None
    evidence_identity: dict[str, Any]
    target_col: str


def project_scorecard_report_evidence(
    binding: StrategyCandidatePoolArtifactBinding,
) -> dict[str, Any]:
    """Project governed scorecard detail without exposing executable vectors.

    The projection follows the current Pool order and recursively follows only
    the parent entries frozen by Voting candidates.  Every scorecard selection
    and Voting wrapper is replayed against its authenticated parent before any
    aggregate value is copied.  Full-band assets and selections are
    de-duplicated by their canonical domain identities; their authenticated
    TaskArtifact refs and every distinct Pool/Voting usage path remain visible
    for audit.  Resource limits fail closed and never slice the evidence.
    """

    if not isinstance(binding, StrategyCandidatePoolArtifactBinding):
        raise StrategyError(
            "scorecard report projection requires an authenticated "
            "StrategyCandidatePoolArtifactBinding"
        )
    pool = validate_strategy_pool(binding.pool)
    if (
        pool != binding.pool
        or pool["task_id"] != binding.task_id
        or pool["strategy_type"] != binding.strategy_type
        or compile_strategy_pool(pool) != binding.compiled_design
    ):
        raise StrategyError("scorecard report Pool binding changed")
    canonical_pool = canonical_strategy_pool_json(pool).encode("utf-8")
    if not hmac.compare_digest(
        hashlib.sha256(canonical_pool).hexdigest(),
        _required_hash(
            binding.artifact_content_hash,
            "scorecard report Pool artifact content hash",
        ),
    ):
        raise StrategyError("scorecard report Pool canonical evidence changed")

    entries = pool["entries"]
    relevant_types = {SCORECARD_BAND_ASSET_TYPE, VOTING_CANDIDATE_ASSET_TYPE}
    if not any(
        entry["source"]["asset_type"] in relevant_types for entry in entries
    ):
        return {
            "schema_version": SCORECARD_REPORT_PROJECTION_SCHEMA_VERSION,
            "models": [],
            "usages": [],
            "artifact_refs": [],
        }
    if len(binding.lineages) != len(entries):
        raise StrategyError(
            "scorecard report Pool lineage order is incomplete"
        )

    models_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    model_order: list[tuple[str, str]] = []
    usages_by_identity: dict[
        tuple[str, str, str, str],
        dict[str, Any],
    ] = {}
    usage_order: list[tuple[str, str, str, str]] = []
    artifact_refs: dict[tuple[str, str], dict[str, str]] = {}
    active_voting: set[tuple[str, str]] = set()
    traversal_nodes = 0
    detail_rows = 0
    usage_path_count = 0

    def add_artifact_ref(
        *,
        kind: str,
        ref_id: str,
        content_hash: str,
    ) -> dict[str, str]:
        ref = {
            "kind": _required_text(kind, "scorecard report artifact kind"),
            "ref_id": _required_text(ref_id, "scorecard report artifact id"),
            "content_hash": _required_hash(
                content_hash,
                "scorecard report artifact content hash",
            ),
        }
        identity = (ref["kind"], ref["ref_id"])
        existing = artifact_refs.get(identity)
        if existing is not None and not hmac.compare_digest(
            existing["content_hash"],
            ref["content_hash"],
        ):
            raise StrategyError(
                "scorecard report artifact reference identity drifted"
            )
        artifact_refs[identity] = ref
        return ref

    def dedupe_artifact_refs(
        refs: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        by_identity: dict[tuple[str, str], dict[str, str]] = {}
        for ref in refs:
            normalized = add_artifact_ref(
                kind=ref["kind"],
                ref_id=ref["ref_id"],
                content_hash=ref["content_hash"],
            )
            by_identity[
                (normalized["kind"], normalized["ref_id"])
            ] = normalized
        return [
            by_identity[key] for key in sorted(by_identity)
        ]

    def walk(
        entry: Mapping[str, Any],
        lineage: _CandidateLineage,
        *,
        path: list[dict[str, Any]],
        lineage_refs: list[dict[str, str]],
        depth: int,
    ) -> None:
        nonlocal detail_rows, traversal_nodes, usage_path_count
        traversal_nodes += 1
        if traversal_nodes > _MAX_VOTING_ANCESTRY_NODES:
            raise StrategyError(
                "scorecard report lineage exceeds node budget"
            )
        if depth > _MAX_VOTING_ANCESTRY_DEPTH:
            raise StrategyError(
                "scorecard report lineage exceeds depth budget"
            )
        if lineage.source_binding != entry["source"]:
            raise StrategyError(
                "scorecard report lineage changed from Pool entry"
            )
        source_asset_type = entry["source"]["asset_type"]
        if (
            source_asset_type == SCORECARD_BAND_ASSET_TYPE
            and not isinstance(lineage, _ScorecardCandidateLineage)
        ) or (
            source_asset_type == VOTING_CANDIDATE_ASSET_TYPE
            and not isinstance(lineage, _VotingCandidateLineage)
        ):
            raise StrategyError(
                "scorecard report relevant lineage type changed"
            )

        if isinstance(lineage, _ScorecardCandidateLineage):
            replayed = scorecard_cutoff_selection_to_verified_candidate_fragment(
                lineage.selection.selection,
                lineage.asset.asset,
                selection_artifact_binding=lineage.selection.to_domain_binding(),
                source_artifact_binding=lineage.asset.to_domain_binding(),
            )
            if (
                replayed != lineage.verified_fragment
                or replayed["artifact"]["artifact_id"]
                != entry["source"]["artifact_id"]
                or replayed["asset"]["asset_id"]
                != entry["source"]["asset_id"]
            ):
                raise StrategyError(
                    "scorecard report cutoff selection replay changed"
                )
            asset_ref = add_artifact_ref(
                kind="strategy_scorecard_band_asset",
                ref_id=lineage.asset.artifact_id,
                content_hash=lineage.asset.content_hash,
            )
            selection_ref = add_artifact_ref(
                kind="strategy_scorecard_cutoff_selection",
                ref_id=lineage.selection.artifact_id,
                content_hash=lineage.selection.content_hash,
            )
            model_key = (
                lineage.asset.asset["asset_id"],
                lineage.asset.asset["asset_hash"],
            )
            model = _scorecard_report_model(
                lineage.asset.asset,
                artifact_ref=asset_ref,
            )
            existing = models_by_identity.get(model_key)
            if existing is None:
                if len(model_order) >= MAX_SCORECARD_REPORT_MODELS:
                    raise StrategyError(
                        "scorecard report models exceed budget"
                    )
                model_order.append(model_key)
                models_by_identity[model_key] = model
                detail_rows += (
                    len(model["scorecard_points"])
                    + len(model["bands"])
                    + len(model["cutoffs"])
                )
                if detail_rows > MAX_SCORECARD_REPORT_DETAIL_ROWS:
                    raise StrategyError(
                        "scorecard report detail rows exceed budget"
                    )
            elif existing != model:
                raise StrategyError(
                    "scorecard report band artifact projection drifted"
                )
            usage_path_count += 1
            if usage_path_count > MAX_SCORECARD_REPORT_USAGES:
                raise StrategyError(
                    "scorecard report usage paths exceed budget"
                )
            selection_value = lineage.selection.selection
            usage_key = (
                *model_key,
                selection_value["selection_id"],
                selection_value["selection_hash"],
            )
            usage = {
                "model_ref": model_key,
                "band_artifact_ref": dict(asset_ref),
                "selection_artifact_ref": dict(selection_ref),
                "usage_artifact_refs": dedupe_artifact_refs(
                    [*lineage_refs, asset_ref, selection_ref]
                ),
                "cutoff_id": _required_text(
                    selection_value["cutoff_id"],
                    "scorecard report cutoff_id",
                ),
                "selection_reason": selection_value["selection_reason"],
            }
            usage_path = {
                "path": [dict(node) for node in path],
                "artifact_refs": [
                    dict(ref) for ref in usage["usage_artifact_refs"]
                ],
            }
            existing_usage = usages_by_identity.get(usage_key)
            if existing_usage is None:
                usage_order.append(usage_key)
                usages_by_identity[usage_key] = {
                    **usage,
                    "usage_paths": [usage_path],
                }
            else:
                if {
                    key: value
                    for key, value in existing_usage.items()
                    if key not in {"usage_paths", "usage_artifact_refs"}
                } != {
                    key: value
                    for key, value in usage.items()
                    if key != "usage_artifact_refs"
                }:
                    raise StrategyError(
                        "scorecard report selection projection drifted"
                    )
                existing_usage["usage_artifact_refs"] = (
                    dedupe_artifact_refs(
                        [
                            *existing_usage["usage_artifact_refs"],
                            *usage["usage_artifact_refs"],
                        ]
                    )
                )
                if any(
                    usage_path["path"] == item["path"]
                    for item in existing_usage["usage_paths"]
                ):
                    raise StrategyError(
                        "scorecard report contains a duplicate usage path"
                    )
                existing_usage["usage_paths"].append(usage_path)
            return

        if not isinstance(lineage, _VotingCandidateLineage):
            return
        candidate = lineage.candidate
        voting_key = (candidate.artifact_id, candidate.content_hash)
        if voting_key in active_voting:
            raise StrategyError(
                "scorecard report Voting ancestry contains a cycle"
            )
        if len(active_voting) >= _MAX_VOTING_ANCESTRY_DEPTH:
            raise StrategyError(
                "scorecard report Voting ancestry exceeds depth budget"
            )
        active_voting.add(voting_key)
        try:
            document = validate_voting_candidate_artifact_document(
                candidate.document
            )
            canonical = canonical_voting_candidate_artifact_json(
                document
            ).encode("utf-8")
            if (
                document != candidate.document
                or document["asset"] != candidate.asset
                or canonical != candidate.canonical_bytes
                or not hmac.compare_digest(
                    hashlib.sha256(canonical).hexdigest(),
                    _required_hash(
                        candidate.content_hash,
                        "scorecard report Voting artifact content hash",
                    ),
                )
            ):
                raise StrategyError(
                    "scorecard report Voting artifact canonical evidence changed"
                )
            parent = validate_strategy_pool(lineage.parent_pool)
            verify_voting_candidate_asset_against_pool(candidate.asset, parent)
            voting_ref = add_artifact_ref(
                kind="strategy_voting_candidate",
                ref_id=candidate.artifact_id,
                content_hash=candidate.content_hash,
            )
            parent_artifact = lineage.parent_pool_artifact
            expected_parent_hash = strategy_pool_artifact_content_hash(parent)
            expected_parent_origin = _ORIGIN_BY_OPERATION[
                parent["operation"]["kind"]
            ]
            if (
                getattr(parent_artifact, "task_id", None) != binding.task_id
                or getattr(parent_artifact, "kind", None) != POOL_ARTIFACT_KIND
                or getattr(parent_artifact, "origin_tool", None)
                != expected_parent_origin
                or not hmac.compare_digest(
                    _required_hash(
                        getattr(parent_artifact, "content_hash", None),
                        "scorecard report parent Pool artifact content hash",
                    ),
                    expected_parent_hash,
                )
                or candidate.provenance.get("pool_artifact_id")
                != getattr(parent_artifact, "artifact_id", None)
                or not hmac.compare_digest(
                    _required_hash(
                        candidate.provenance.get(
                            "pool_artifact_content_hash"
                        ),
                        "scorecard report Voting parent Pool content hash",
                    ),
                    expected_parent_hash,
                )
            ):
                raise StrategyError(
                    "scorecard report Voting parent Pool artifact changed"
                )
            parent_pool_ref = add_artifact_ref(
                kind="strategy_candidate_pool",
                ref_id=_required_text(
                    parent_artifact.artifact_id,
                    "scorecard report parent Pool artifact id",
                ),
                content_hash=expected_parent_hash,
            )
            child_lineage_refs = dedupe_artifact_refs(
                [*lineage_refs, voting_ref, parent_pool_ref]
            )
            replayed = voting_candidate_to_verified_fragment(
                candidate.asset,
                artifact_binding=candidate.artifact_binding(),
            )
            if (
                replayed != lineage.verified_fragment
                or replayed["artifact"]["artifact_id"]
                != entry["source"]["artifact_id"]
                or replayed["asset"]["asset_id"]
                != entry["source"]["asset_id"]
            ):
                raise StrategyError(
                    "scorecard report Voting fragment replay changed"
                )
            selected = candidate.asset["selected_entries"]
            if len(selected) != len(lineage.parent_lineages):
                raise StrategyError(
                    "scorecard report Voting parent lineage order changed"
                )
            parent_by_id = {
                parent_entry["entry_id"]: parent_entry
                for parent_entry in parent["entries"]
            }
            for selected_entry, parent_lineage in zip(
                selected,
                lineage.parent_lineages,
                strict=True,
            ):
                parent_entry = parent_by_id.get(selected_entry["entry_id"])
                if (
                    parent_entry is None
                    or parent_entry["position"]
                    != selected_entry["pool_position"]
                    or parent_entry["rule_id"] != selected_entry["rule_id"]
                    or parent_lineage.source_binding != parent_entry["source"]
                ):
                    raise StrategyError(
                        "scorecard report Voting parent entry changed"
                    )
                walk(
                    parent_entry,
                    parent_lineage,
                    path=[
                        *path,
                        _scorecard_report_path_node(
                            parent_entry,
                            scope="voting_parent_entry",
                            lineage=parent_lineage,
                        ),
                    ],
                    lineage_refs=child_lineage_refs,
                    depth=depth + 1,
                )
        finally:
            active_voting.discard(voting_key)

    for entry, lineage in zip(entries, binding.lineages, strict=True):
        walk(
            entry,
            lineage,
            path=[
                _scorecard_report_path_node(
                    entry,
                    scope="current_pool_entry",
                    lineage=lineage,
                )
            ],
            lineage_refs=[],
            depth=0,
        )

    model_index = {
        key: index for index, key in enumerate(model_order, start=1)
    }
    reported_artifact_ref_keys = {
        (ref["kind"], ref["ref_id"])
        for usage in (
            usages_by_identity[key] for key in usage_order
        )
        for ref in usage["usage_artifact_refs"]
    }
    projection = {
        "schema_version": SCORECARD_REPORT_PROJECTION_SCHEMA_VERSION,
        "models": [
            {
                **models_by_identity[key],
                "model_index": model_index[key],
            }
            for key in model_order
        ],
        "usages": [
            {
                **{
                    key: value
                    for key, value in usage.items()
                    if key != "model_ref"
                },
                "model_index": model_index[usage["model_ref"]],
            }
            for usage in (
                usages_by_identity[key] for key in usage_order
            )
        ],
        "artifact_refs": [
            artifact_refs[key]
            for key in sorted(reported_artifact_ref_keys)
        ],
    }
    _enforce_scorecard_projection_report_budget(projection)
    return projection


def _enforce_scorecard_projection_report_budget(
    projection: Mapping[str, Any],
) -> None:
    """Reserve enough of the final report budget for the other six sections."""

    models = projection["models"]
    usages = projection["usages"]
    usages_by_model: dict[int, list[Mapping[str, Any]]] = {
        int(model["model_index"]): [] for model in models
    }
    for usage in usages:
        usages_by_model[int(usage["model_index"])].append(usage)

    field_footprint = 0
    # Each governed artifact appears in the top-level inventory, candidate
    # section and four scorecard table inventories.  The current Pool adds one
    # more identity.  Each model's frozen/result role refs and repeated
    # development dataset binding are budgeted separately.
    artifact_ref_count = len(projection["artifact_refs"]) + 1
    ref_footprint = (
        artifact_ref_count * 6
        + len(models) * 11
        + (2 if models else 0)
    )
    for model in models:
        model_index = int(model["model_index"])
        model_ref_identities = {
            ("strategy_candidate_pool", "current_pool"),
            (
                model["band_artifact_ref"]["kind"],
                model["band_artifact_ref"]["ref_id"],
            ),
            ("backtest", model["band_artifact_ref"]["ref_id"]),
            *(
                (ref["kind"], ref["ref_id"])
                for usage in usages_by_model[model_index]
                for ref in usage["usage_artifact_refs"]
            ),
        }
        model_ref_count = len(model_ref_identities)
        point_count = len(model["scorecard_points"])
        band_count = len(model["bands"])
        cutoff_count = len(model["cutoffs"])
        field_footprint += (
            22
            + point_count * 15
            + band_count * 13
            + cutoff_count * 16
        )
        # This is an upper bound for the current four-table adapter: optional
        # cells can use fewer refs, while selected cutoff refs are subsets of
        # the complete model lineage.
        ref_footprint += (
            19
            + model_ref_count * 3
            + point_count * (14 + model_ref_count)
            + band_count * (12 + model_ref_count)
            + cutoff_count * (12 + model_ref_count * 4)
        )

    if field_footprint > MAX_SCORECARD_REPORT_TABLE_FIELDS:
        raise StrategyError(
            "scorecard report field footprint exceeds budget"
        )
    if ref_footprint > MAX_SCORECARD_REPORT_TABLE_REFS:
        raise StrategyError(
            "scorecard report reference footprint exceeds budget"
        )
    if (
        len(_canonical_json(projection).encode("utf-8"))
        > MAX_SCORECARD_REPORT_PROJECTION_JSON_BYTES
    ):
        raise StrategyError(
            "scorecard report JSON footprint exceeds budget"
        )


def _scorecard_report_path_node(
    entry: Mapping[str, Any],
    *,
    scope: str,
    lineage: _CandidateLineage,
) -> dict[str, Any]:
    voting = (
        lineage.candidate.asset["voting"]
        if isinstance(lineage, _VotingCandidateLineage)
        else None
    )
    return {
        "scope": scope,
        "position": int(entry["position"]),
        "entry_id": _required_text(
            entry["entry_id"],
            "scorecard report path entry_id",
        ),
        "rule_id": _required_text(
            entry["rule_id"],
            "scorecard report path rule_id",
        ),
        "asset_type": _required_text(
            entry["source"]["asset_type"],
            "scorecard report path asset_type",
        ),
        "voting_n": None if voting is None else int(voting["n"]),
        "voting_k": None if voting is None else int(voting["k"]),
    }


def _scorecard_report_model(
    asset: Mapping[str, Any],
    *,
    artifact_ref: Mapping[str, str],
) -> dict[str, Any]:
    contract = asset["score_contract"]
    vector = asset["score_vector"]
    return {
        "band_artifact_ref": dict(artifact_ref),
        "score_product": contract["score_product"],
        "score_direction": contract["score_direction"],
        "points_direction": contract["points_direction"],
        "scale": {
            key: contract["scale"][key]
            for key in (
                "base_score",
                "pdo",
                "base_odds",
                "factor",
                "offset",
            )
        },
        "sample_summary": {
            key: vector[key]
            for key in (
                "row_count",
                "development_count",
                "labeled_count",
                "bad_count",
            )
        },
        "performance": {
            key: asset["performance"][key] for key in ("auc", "ks")
        },
        "lifecycle": {
            key: asset["lifecycle"][key]
            for key in (
                "candidate_stage",
                "observation_stage",
                "validation_status",
            )
        },
        "scorecard_points": [
            {
                key: row[key]
                for key in (
                    "feature",
                    "bin_index",
                    "bin_label",
                    "lower",
                    "upper",
                    "count",
                    "bad_count",
                    "good_count",
                    "bad_rate",
                    "woe",
                    "iv_contribution",
                    "coefficient",
                    "monotonic_direction",
                    "points",
                )
            }
            for row in contract["scorecard_table"]
        ],
        "bands": [
            {
                key: band[key]
                for key in (
                    "ordinal",
                    "bin_id",
                    "lower_bound",
                    "upper_bound",
                    "lower_inclusive",
                    "upper_inclusive",
                    "count",
                    "share",
                    "labeled_count",
                    "bad_count",
                    "bad_rate",
                    "average_pd",
                )
            }
            for band in asset["bands"]
        ],
        "cutoffs": [
            {
                "ordinal": cutoff["ordinal"],
                "cutoff_id": cutoff["cutoff_id"],
                "execution_pd": cutoff["execution_pd"],
                "display_points": cutoff["display_points"],
                "lower_risk": {
                    key: cutoff["lower_risk"][key]
                    for key in (
                        "count",
                        "labeled_count",
                        "bad_count",
                        "bad_rate",
                    )
                },
                "higher_risk": {
                    key: cutoff["higher_risk"][key]
                    for key in (
                        "count",
                        "labeled_count",
                        "bad_count",
                        "bad_rate",
                    )
                },
            }
            for cutoff in asset["cutoffs"]
        ],
    }


@dataclass(frozen=True)
class VerifiedUnivariateCandidateLineageBinding:
    """Authenticated task-owned univariate asset and its complete source lineage.

    Downstream deterministic evidence tools may consume the canonical asset,
    evidence, dataset binding, and verified fragment exposed here.  They must
    re-authenticate this binding with
    :func:`require_verified_univariate_candidate_lineage_on_connection` while
    holding their artifact-registration transaction.
    """

    task_id: str
    lineage: _UnivariateCandidateLineage
    tasks_root: Path
    datasets_root: Path
    db_path: Path

    @property
    def asset(self) -> dict[str, Any]:
        return self.lineage.asset

    @property
    def evidence(self) -> dict[str, Any]:
        return self.lineage.evidence

    @property
    def dataset(self) -> Any:
        return self.lineage.dataset

    @property
    def verified_fragment(self) -> dict[str, Any]:
        return self.lineage.verified_fragment

    @property
    def source_binding(self) -> dict[str, Any]:
        return self.lineage.source_binding


@dataclass(frozen=True)
class _PoolArtifactSourceBinding:
    artifact_id: str
    task_id: str
    kind: str
    path: Path
    content_hash: str
    origin_tool: str
    provenance: dict[str, Any]
    provenance_json: str


@dataclass
class _LineageCache:
    trees: dict[tuple[str, str], VerifiedAutomaticTreeSource]
    matrices: dict[tuple[str, str], VerifiedCrossMatrixSource]
    datasets: dict[tuple[str, str], _AutomaticTreeDatasetBinding]
    univariate_datasets: dict[tuple[str, str], Any]
    datasets_verified_on_connection: set[tuple[str, str]]
    voting: dict[tuple[str, str], _VotingCandidateLineage]
    voting_in_progress: set[tuple[str, str]]
    voting_verified: set[tuple[str, str]]

    @classmethod
    def empty(cls) -> _LineageCache:
        return cls(
            trees={},
            matrices={},
            datasets={},
            univariate_datasets={},
            datasets_verified_on_connection=set(),
            voting={},
            voting_in_progress=set(),
            voting_verified=set(),
        )


def _require_snapshot_voting_marginals(
    runtime,
    *,
    snapshot: Mapping[str, Any],
    lineages: Sequence[_CandidateLineage],
) -> None:
    """Prove every Voting entry remains reachable after an ordering mutation."""

    lineage_by_source = {
        _canonical_json(lineage.source_binding): lineage for lineage in lineages
    }
    voting_candidates: dict[int, VerifiedVotingCandidateArtifact] = {}
    anchor_lineage: _CandidateLineage | None = None
    anchor_entry: Mapping[str, Any] | None = None
    for position, entry in enumerate(snapshot["entries"]):
        if entry["source"]["asset_type"] != VOTING_CANDIDATE_ASSET_TYPE:
            continue
        lineage = lineage_by_source.get(_canonical_json(entry["source"]))
        if not isinstance(lineage, _VotingCandidateLineage):
            raise StrategyError("Voting Pool entry is missing its verified lineage")
        voting_candidates[position] = lineage.candidate
        if anchor_lineage is None:
            selected = lineage.candidate.asset["selected_entries"][0]
            selected_parent_by_id = {
                parent["entry_id"]: parent
                for parent in lineage.parent_pool["entries"]
            }
            anchor_entry = selected_parent_by_id.get(selected["entry_id"])
            if anchor_entry is None:
                raise StrategyError(
                    "Voting candidate selected parent entry no longer exists"
                )
            anchor_lineage = lineage.parent_lineages[0]
    if not voting_candidates:
        return
    if anchor_lineage is None or anchor_entry is None:
        raise StrategyError("Voting Pool replay is missing its sample anchor")
    require_voting_snapshot_marginal_reachability(
        runtime,
        entries=snapshot["entries"],
        voting_candidates=voting_candidates,
        anchor_lineage=anchor_lineage,
        anchor_entry=anchor_entry,
    )


def run_add_candidate_to_pool(inputs, ctx, runtime) -> dict[str, Any]:
    """Add one verified concrete candidate through the generic Pool seam."""

    try:
        normalized = _validate_add_inputs(inputs)
        task_id = _required_text(ctx.task_id, "task_id")
        repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
        legacy_archive = repository.get_archived_legacy_draft(
            task_id, normalized["strategy_type"]
        )
        base = _expected_base_pool(
            repository,
            task_id=task_id,
            strategy_type=normalized["strategy_type"],
            expected_revision=normalized["expected_pool_revision"],
            expected_snapshot_hash=normalized["expected_pool_snapshot_hash"],
        )
        lineage_cache = _LineageCache.empty()
        prior_lineages = _load_pool_lineages(
            runtime,
            task_id=task_id,
            pool=base,
            cache=lineage_cache,
        )
        candidate = _load_candidate_lineage(
            runtime,
            task_id=task_id,
            artifact_id=normalized["source_artifact_id"],
            expected_content_hash=normalized["expected_artifact_content_hash"],
            expected_asset_id=normalized["expected_asset_id"],
            expected_asset_hash=normalized["expected_asset_hash"],
            cache=lineage_cache,
        )
        if isinstance(candidate, _VotingCandidateLineage):
            if base is None:
                raise StrategyError(
                    "Voting candidate requires its existing source Strategy Pool"
                )
            if candidate.parent_pool["strategy_type"] != normalized["strategy_type"]:
                raise StrategyError(
                    "Voting candidate strategy_type differs from the target Pool"
                )
            if candidate.parent_pool["revision"] > base["revision"]:
                raise StrategyError(
                    "Voting candidate parent Pool is newer than the target revision"
                )
            placement_mode = normalized.get("placement_mode")
            if placement_mode not in {
                BEFORE_SELECTED_MEMBERS_PLACEMENT,
                REPLACE_SELECTED_MEMBERS_PLACEMENT,
            }:
                raise StrategyError(
                    "Voting candidate admission requires explicit "
                    "before_selected_members or replace_selected_members placement"
                )
            selected_entry_ids = [
                str(entry["entry_id"])
                for entry in candidate.candidate.asset["selected_entries"]
            ]
        else:
            placement_mode = normalized.get("placement_mode", APPEND_PLACEMENT)
            if placement_mode != APPEND_PLACEMENT:
                raise StrategyError(
                    "non-Voting candidates only support append placement"
                )
            selected_entry_ids = []
        snapshot = add_verified_candidate_fragment(
            base,
            task_id=task_id,
            strategy_type=normalized["strategy_type"],
            default_action=normalized["default_action"],
            verified_candidate_fragment=candidate.verified_fragment,
            action=normalized["action"],
            placement_mode=placement_mode,
            selected_entry_ids=selected_entry_ids,
            reason=normalized.get("reason"),
        )
        surviving_sources = {
            (
                entry["source"]["artifact_id"],
                entry["source"]["artifact_content_hash"],
            )
            for entry in snapshot["entries"]
        }
        persisted_lineages = [
            lineage
            for lineage in prior_lineages
            if (
                lineage.source_binding["artifact_id"],
                lineage.source_binding["artifact_content_hash"],
            )
            in surviving_sources
        ]
        mutation_lineages = [*persisted_lineages, candidate]
        _require_cross_matrix_groups_disjoint(mutation_lineages)
        _require_snapshot_voting_marginals(
            runtime,
            snapshot=snapshot,
            lineages=mutation_lineages,
        )
        return _persist_mutation(
            runtime,
            repository=repository,
            snapshot=snapshot,
            expected_revision=normalized["expected_pool_revision"],
            expected_snapshot_hash=normalized["expected_pool_snapshot_hash"],
            lineages=mutation_lineages,
            inputs=normalized,
            legacy_archive=legacy_archive,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def run_remove_pool_entry(inputs, ctx, runtime) -> dict[str, Any]:
    """Remove the entry addressed by its external stable ``rule_id``."""

    try:
        normalized = _validate_inputs(
            inputs,
            allowed=_REMOVE_INPUT_FIELDS,
            required=_REMOVE_REQUIRED_FIELDS,
            tool_name="remove_pool_entry",
        )
        normalized = _normalize_common_mutation_inputs(normalized, include_rule=True)
        task_id, repository, base, legacy_archive = _mutation_base(
            runtime, ctx, normalized
        )
        lineages = _load_pool_lineages(runtime, task_id=task_id, pool=base)
        entry_id = _entry_id_for_rule(base, normalized["rule_id"])
        snapshot = remove_pool_entry(
            base,
            entry_id,
            reason=normalized.get("reason"),
        )
        return _persist_mutation(
            runtime,
            repository=repository,
            snapshot=snapshot,
            expected_revision=normalized["expected_pool_revision"],
            expected_snapshot_hash=normalized["expected_pool_snapshot_hash"],
            lineages=lineages,
            inputs=normalized,
            legacy_archive=legacy_archive,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def run_set_pool_entry_action(inputs, ctx, runtime) -> dict[str, Any]:
    """Set a Pool-owned typed action for the entry addressed by ``rule_id``."""

    try:
        normalized = _validate_inputs(
            inputs,
            allowed=_SET_ACTION_INPUT_FIELDS,
            required=_SET_ACTION_REQUIRED_FIELDS,
            tool_name="set_pool_entry_action",
        )
        normalized = _normalize_common_mutation_inputs(normalized, include_rule=True)
        if not isinstance(normalized["action"], Mapping):
            raise StrategyError("action must be an object")
        normalized["action"] = _json_object(normalized["action"], "action")
        task_id, repository, base, legacy_archive = _mutation_base(
            runtime, ctx, normalized
        )
        entry_id = _entry_id_for_rule(base, normalized["rule_id"])
        snapshot = set_pool_entry_action(
            base,
            entry_id,
            normalized["action"],
            reason=normalized.get("reason"),
        )
        lineages = _load_pool_lineages(runtime, task_id=task_id, pool=snapshot)
        return _persist_mutation(
            runtime,
            repository=repository,
            snapshot=snapshot,
            expected_revision=normalized["expected_pool_revision"],
            expected_snapshot_hash=normalized["expected_pool_snapshot_hash"],
            lineages=lineages,
            inputs=normalized,
            legacy_archive=legacy_archive,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def run_reorder_strategy_pool(inputs, ctx, runtime) -> dict[str, Any]:
    """Apply one complete external ``rule_id`` permutation to a pool."""

    try:
        normalized = _validate_inputs(
            inputs,
            allowed=_REORDER_INPUT_FIELDS,
            required=_REORDER_REQUIRED_FIELDS,
            tool_name="reorder_strategy_pool",
        )
        normalized = _normalize_common_mutation_inputs(normalized)
        ordered = _text_list(normalized["ordered_rule_ids"], "ordered_rule_ids")
        if len(set(ordered)) != len(ordered):
            raise StrategyError("ordered_rule_ids must not contain duplicate rule_ids")
        normalized["ordered_rule_ids"] = ordered
        task_id, repository, base, legacy_archive = _mutation_base(
            runtime, ctx, normalized
        )
        entry_ids = [_entry_id_for_rule(base, rule_id) for rule_id in ordered]
        snapshot = reorder_strategy_pool(
            base,
            entry_ids,
            reason=normalized.get("reason"),
        )
        lineages = _load_pool_lineages(runtime, task_id=task_id, pool=snapshot)
        _require_snapshot_voting_marginals(
            runtime,
            snapshot=snapshot,
            lineages=lineages,
        )
        return _persist_mutation(
            runtime,
            repository=repository,
            snapshot=snapshot,
            expected_revision=normalized["expected_pool_revision"],
            expected_snapshot_hash=normalized["expected_pool_snapshot_hash"],
            lineages=lineages,
            inputs=normalized,
            legacy_archive=legacy_archive,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def run_compile_strategy_pool(inputs, ctx, runtime) -> dict[str, Any]:
    """Compile the exact current pool to a canonical design without execution."""

    try:
        normalized = _validate_inputs(
            inputs,
            allowed=_COMPILE_INPUT_FIELDS,
            required=_COMPILE_INPUT_FIELDS,
            tool_name="compile_strategy_pool",
        )
        normalized = _normalize_cas_inputs(normalized)
        task_id = _required_text(ctx.task_id, "task_id")
        repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
        legacy_archive = repository.get_archived_legacy_draft(
            task_id, normalized["strategy_type"]
        )
        current = repository.get_current(task_id, normalized["strategy_type"])
        if current is None:
            if legacy_archive is not None:
                raise StrategyPoolLegacyDraftNeedsRebuildError(legacy_archive)
            raise StrategyError("strategy candidate pool not found")
        pool = validate_strategy_pool(current)
        if pool["revision"] != normalized[
            "expected_pool_revision"
        ] or not hmac.compare_digest(
            pool["snapshot_hash"],
            normalized["expected_pool_snapshot_hash"],
        ):
            raise StrategyError(
                "stale strategy candidate pool revision or snapshot hash"
            )
        _load_pool_lineages(runtime, task_id=task_id, pool=pool)
        artifact = _load_pool_artifact(runtime, task_id=task_id, snapshot=pool)
        selected = compile_strategy_pool(pool)
        return {
            "schema_version": POOL_COMPILE_TOOL_SCHEMA_VERSION,
            "pool_id": pool["pool_id"],
            "revision": pool["revision"],
            "snapshot_hash": pool["snapshot_hash"],
            "requirements": selected["requirements"],
            "strategy_spec": selected["strategy_spec"],
            "source_entry_refs": selected["source_entry_refs"],
            "design_hash": selected["design_hash"],
            "selected_strategy_design": selected,
            "artifacts": [_artifact_output(artifact, task_id=task_id)],
            "archived_legacy_draft": legacy_archive,
            "warnings": ([] if legacy_archive is None else [_LEGACY_ARCHIVE_WARNING]),
        }
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def load_current_strategy_candidate_pool_artifact(
    runtime,
    *,
    task_id: str,
    strategy_type: str,
    expected_pool_revision: int | None = None,
    expected_pool_snapshot_hash: str | None = None,
    expected_artifact_id: str | None = None,
    expected_artifact_content_hash: str | None = None,
) -> StrategyCandidatePoolArtifactBinding:
    """Load the exact current V2 Pool as a downstream-safe immutable binding."""

    try:
        normalized_task = _required_text(task_id, "task_id")
        normalized_type = _required_text(strategy_type, "strategy_type")
        revision = (
            None
            if expected_pool_revision is None
            else _non_negative_int(
                expected_pool_revision,
                "expected_pool_revision",
            )
        )
        snapshot_hash = (
            None
            if expected_pool_snapshot_hash is None
            else _required_hash(
                expected_pool_snapshot_hash,
                "expected_pool_snapshot_hash",
            )
        )
        artifact_id = (
            None
            if expected_artifact_id is None
            else _required_hash(expected_artifact_id, "expected_artifact_id")
        )
        artifact_content_hash = (
            None
            if expected_artifact_content_hash is None
            else _required_hash(
                expected_artifact_content_hash,
                "expected_artifact_content_hash",
            )
        )
        tasks_root = Path(runtime.settings.tasks_dir).absolute()
        datasets_root = Path(runtime.settings.datasets_dir).absolute()
        db_path = Path(runtime.settings.db_path).absolute()
        repository = StrategyCandidatePoolRepository(db_path)
        current = repository.get_current(normalized_task, normalized_type)
        if current is None:
            raise StrategyError("strategy candidate pool not found")
        pool = validate_strategy_pool(current)
        if revision is not None and pool["revision"] != revision:
            raise StrategyError("stale strategy candidate pool revision")
        if snapshot_hash is not None and not hmac.compare_digest(
            pool["snapshot_hash"],
            snapshot_hash,
        ):
            raise StrategyError("stale strategy candidate pool snapshot hash")

        lineages = tuple(
            _load_pool_lineages(
                runtime,
                task_id=normalized_task,
                pool=pool,
            )
        )
        artifact_record = _normalize_source_record(
            _load_pool_artifact(
                runtime,
                task_id=normalized_task,
                snapshot=pool,
            )
        )
        if artifact_id is not None and artifact_record.artifact_id != artifact_id:
            raise StrategyError("current strategy pool artifact id changed")
        if artifact_content_hash is not None and not hmac.compare_digest(
            artifact_record.content_hash,
            artifact_content_hash,
        ):
            raise StrategyError("current strategy pool artifact content hash changed")
        compiled_design = compile_strategy_pool(pool)
        binding = StrategyCandidatePoolArtifactBinding(
            task_id=normalized_task,
            strategy_type=normalized_type,
            pool=pool,
            compiled_design=compiled_design,
            artifact_id=artifact_record.artifact_id,
            artifact_path=artifact_record.path,
            artifact_content_hash=artifact_record.content_hash,
            artifact_origin_tool=artifact_record.origin_tool,
            artifact_provenance=artifact_record.provenance,
            artifact_provenance_json=artifact_record.provenance_json,
            lineages=lineages,
            tasks_root=tasks_root,
            datasets_root=datasets_root,
            db_path=db_path,
        )
        with repository.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_strategy_candidate_pool_artifact_binding_on_connection(
                conn,
                binding,
            )
            conn.commit()
        return binding
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def bind_strategy_pool_development_execution(
    runtime,
    pool_binding: StrategyCandidatePoolArtifactBinding,
) -> StrategyPoolDevelopmentExecutionBinding:
    """Resolve one public development/sample binding for an exact Pool."""

    if not isinstance(pool_binding, StrategyCandidatePoolArtifactBinding):
        raise StrategyError(
            "strategy Pool development requires an authenticated Pool binding"
        )
    facts = _project_pool_development_lineages(pool_binding)
    reference = facts.sample_ref
    sample_artifact = load_strategy_sample_design_artifact(
        runtime,
        task_id=pool_binding.task_id,
        artifact_id=reference.artifact_id,
        expected_artifact_content_hash=reference.artifact_content_hash,
        expected_sample_design_id=reference.sample_design_id,
        expected_sample_design_content_hash=reference.sample_design_content_hash,
    )
    design = sample_artifact.bundle["sample_design"]
    drop_nan_labels = design["target_definition"]["drop_nan_labels"]
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError(
            "strategy Pool sample-design drop_nan_labels binding is invalid"
        )
    identity = facts.evidence_identity
    sample = load_strategy_sample_design_execution_binding(
        runtime,
        task_id=pool_binding.task_id,
        sample_design_ref=reference.to_ref_dict(),
        dataset_id=identity["dataset_id"],
        dataset_content_hash=identity["dataset_content_hash"],
        workspace_revision=identity["workspace_revision"],
        workspace_generation=identity["workspace_generation"],
        semantic_mapping_hash=identity["semantic_mapping_hash"],
        target_col=facts.target_col,
        drop_nan_labels=drop_nan_labels,
    )
    if (
        sample.reference != reference
        or sample.dataset_id != facts.dataset.dataset_id
        or not hmac.compare_digest(
            sample.dataset_content_hash,
            facts.dataset.content_hash,
        )
        or sample.target_col != facts.target_col
    ):
        raise StrategyError(
            "strategy Pool development sample changed from candidate lineage"
        )
    if facts.sample_design_v2 is not None:
        source = facts.sample_design_v2.source_binding
        if source.legacy != sample:
            raise StrategyError(
                "strategy Pool SampleDesign V2 legacy mapping changed"
            )
    binding = StrategyPoolDevelopmentExecutionBinding(
        task_id=pool_binding.task_id,
        pool=pool_binding,
        dataset=facts.dataset,
        sample_design=sample,
        sample_design_v2=facts.sample_design_v2,
        evidence_identity=dict(facts.evidence_identity),
        target_col=facts.target_col,
        month_col=sample.month_col,
    )
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_pool_development_execution_binding_on_connection(
            conn,
            binding,
        )
        conn.commit()
    return binding


def require_strategy_pool_development_execution_binding_on_connection(
    conn,
    binding: StrategyPoolDevelopmentExecutionBinding,
) -> None:
    """Re-authenticate a Pool and its exact development universe under lock."""

    if not isinstance(binding, StrategyPoolDevelopmentExecutionBinding):
        raise StrategyError("strategy Pool development binding is invalid")
    if binding.task_id != binding.pool.task_id:
        raise StrategyError("strategy Pool development task binding changed")
    require_strategy_candidate_pool_artifact_binding_on_connection(
        conn,
        binding.pool,
    )
    facts = _project_pool_development_lineages(binding.pool)
    if (
        facts.dataset != binding.dataset
        or facts.sample_ref != binding.sample_design.reference
        or facts.evidence_identity != binding.evidence_identity
        or facts.target_col != binding.target_col
        or binding.sample_design.task_id != binding.task_id
        or binding.sample_design.dataset_id != binding.dataset.dataset_id
        or not hmac.compare_digest(
            binding.sample_design.dataset_content_hash,
            binding.dataset.content_hash,
        )
        or binding.sample_design.target_col != binding.target_col
        or binding.sample_design.month_col != binding.month_col
        or _sample_design_v2_identity(facts.sample_design_v2)
        != _sample_design_v2_identity(binding.sample_design_v2)
    ):
        raise StrategyError("strategy Pool development binding changed")
    require_strategy_sample_design_execution_binding_on_connection(
        conn,
        binding.sample_design,
    )
    if binding.sample_design_v2 is not None:
        require_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            binding.sample_design_v2,
        )
        if binding.sample_design_v2.source_binding.legacy != binding.sample_design:
            raise StrategyError(
                "strategy Pool SampleDesign V2 legacy mapping changed"
            )


def _project_pool_development_lineages(
    binding: StrategyCandidatePoolArtifactBinding,
) -> _PoolDevelopmentLineageFacts:
    pool = validate_strategy_pool(binding.pool)
    if (
        pool != binding.pool
        or pool["task_id"] != binding.task_id
        or pool["strategy_type"] != binding.strategy_type
        or compile_strategy_pool(pool) != binding.compiled_design
    ):
        raise StrategyError("strategy Pool development Pool binding changed")
    entries = pool["entries"]
    if not entries or len(entries) != len(binding.lineages):
        raise StrategyError(
            "strategy Pool development requires complete non-empty lineages"
    )
    projected: list[_PoolDevelopmentLineageFacts] = []
    active_voting: set[str] = set()
    voting_nodes: set[str] = set()
    facts_by_lineage: dict[str, _PoolDevelopmentLineageFacts] = {}
    for entry, lineage in zip(entries, binding.lineages, strict=True):
        if lineage.source_binding != entry["source"]:
            raise StrategyError(
                "strategy Pool development entry lineage changed"
            )
        facts = _pool_lineage_development_facts(
            lineage,
            active_voting=active_voting,
            voting_nodes=voting_nodes,
            facts_by_lineage=facts_by_lineage,
        )
        if facts.evidence_identity != entry["source"]["evidence_identity"]:
            raise StrategyError(
                "strategy Pool development evidence identity changed"
            )
        if (
            facts.dataset.task_id != binding.task_id
            or facts.dataset.dataset_id
            != facts.evidence_identity["dataset_id"]
            or not hmac.compare_digest(
                facts.dataset.content_hash,
                facts.evidence_identity["dataset_content_hash"],
            )
        ):
            raise StrategyError(
                "strategy Pool development dataset identity changed"
            )
        projected.append(facts)
    first = projected[0]
    selected_v2 = first.sample_design_v2
    for facts in projected[1:]:
        _require_same_pool_development_facts(first, facts)
        if facts.sample_design_v2 is not None:
            if selected_v2 is None:
                selected_v2 = facts.sample_design_v2
            elif _sample_design_v2_identity(
                selected_v2
            ) != _sample_design_v2_identity(facts.sample_design_v2):
                raise StrategyError(
                    "strategy Pool candidates resolve different SampleDesign V2"
                )
    return _PoolDevelopmentLineageFacts(
        dataset=first.dataset,
        sample_ref=first.sample_ref,
        sample_design_v2=selected_v2,
        evidence_identity=dict(first.evidence_identity),
        target_col=first.target_col,
    )


def _pool_lineage_development_facts(
    lineage: _CandidateLineage,
    *,
    active_voting: set[str],
    voting_nodes: set[str],
    facts_by_lineage: dict[str, _PoolDevelopmentLineageFacts],
) -> _PoolDevelopmentLineageFacts:
    cache_identity = _canonical_json(lineage.source_binding)
    cached = facts_by_lineage.get(cache_identity)
    if cached is not None:
        return cached

    def remember(
        facts: _PoolDevelopmentLineageFacts,
    ) -> _PoolDevelopmentLineageFacts:
        facts_by_lineage[cache_identity] = facts
        return facts

    if isinstance(lineage, (_UnivariateCandidateLineage, _CrossMatrixCandidateLineage)):
        evidence = lineage.evidence
        parameters = evidence["generation"]["parameters"]
        target_col = _required_text(
            evidence["analysis"]["target"],
            "strategy Pool candidate target",
        )
        if parameters.get("target_col") != target_col:
            raise StrategyError(
                "strategy Pool candidate target binding is inconsistent"
            )
        verified_identity = _verified_lineage_evidence_identity(lineage)
        if not hmac.compare_digest(
            sample_context_hash_from_candidate_evidence(evidence),
            verified_identity["sample_context_hash"],
        ):
            raise StrategyError(
                "strategy Pool candidate sample context changed"
            )
        return remember(_PoolDevelopmentLineageFacts(
            dataset=_pool_development_dataset(lineage.dataset),
            sample_ref=StrategySampleDesignRef.from_value(
                parameters.get("sample_design_ref")
            ),
            sample_design_v2=None,
            evidence_identity=verified_identity,
            target_col=target_col,
        ))
    if isinstance(lineage, _AutomaticTreeCandidateLineage):
        asset = lineage.tree.asset
        training = asset["tree_result"]["training"]
        asset_identity = asset["identity"]
        verified_identity = _verified_lineage_evidence_identity(lineage)
        if _evidence_identity_from_concrete(asset_identity) != verified_identity:
            raise StrategyError(
                "automatic-tree strategy Pool evidence identity changed"
            )
        sample_ref = sample_design_ref_from_automatic_tree_source_refs(
            asset["source_refs"]
        )
        if StrategySampleDesignRef.from_value(
            lineage.tree.provenance.get("sample_design_ref")
        ).to_ref_dict() != sample_ref:
            raise StrategyError(
                "automatic-tree sample-design asset and provenance disagree"
            )
        return remember(_PoolDevelopmentLineageFacts(
            dataset=_pool_development_dataset(lineage.dataset),
            sample_ref=StrategySampleDesignRef.from_value(sample_ref),
            sample_design_v2=None,
            evidence_identity=verified_identity,
            target_col=_required_text(
                training["target_col"],
                "automatic-tree strategy Pool target",
            ),
        ))
    if isinstance(lineage, _ScorecardCandidateLineage):
        sample_v2 = lineage.asset.sample_design
        source = sample_v2.source_binding
        design = sample_v2.bundle["sample_design"]
        target = design["target_selector"]
        if target["status"] != "resolved":
            raise StrategyError(
                "scorecard strategy Pool target selector is unresolved"
            )
        target_col = _required_text(
            target["column"],
            "scorecard strategy Pool target",
        )
        if (
            source.legacy.target_col != target_col
            or source.legacy.target_bad_value != target["bad_value"]
            or source.legacy.drop_nan_labels is not target["drop_missing"]
        ):
            raise StrategyError(
                "scorecard strategy Pool target semantics changed"
            )
        verified_identity = _verified_lineage_evidence_identity(lineage)
        if _evidence_identity_from_concrete(
            lineage.asset.asset["identity"]
        ) != verified_identity:
            raise StrategyError(
                "scorecard strategy Pool evidence identity changed"
            )
        return remember(_PoolDevelopmentLineageFacts(
            dataset=_pool_development_dataset(lineage.dataset),
            sample_ref=source.legacy.reference,
            sample_design_v2=sample_v2,
            evidence_identity=verified_identity,
            target_col=target_col,
        ))
    if isinstance(lineage, _VotingCandidateLineage):
        if cache_identity in active_voting:
            raise StrategyError(
                "strategy Pool development Voting ancestry contains a cycle"
            )
        voting_nodes.add(cache_identity)
        if len(voting_nodes) > _MAX_VOTING_ANCESTRY_NODES:
            raise StrategyError(
                "strategy Pool development Voting ancestry is too large"
            )
        active_voting.add(cache_identity)
        try:
            selected = lineage.candidate.asset["selected_entries"]
            if not selected or len(selected) != len(lineage.parent_lineages):
                raise StrategyError(
                    "strategy Pool development Voting parents are incomplete"
                )
            parents = [
                _pool_lineage_development_facts(
                    parent,
                    active_voting=active_voting,
                    voting_nodes=voting_nodes,
                    facts_by_lineage=facts_by_lineage,
                )
                for parent in lineage.parent_lineages
            ]
        finally:
            active_voting.remove(cache_identity)
        first = parents[0]
        selected_v2 = first.sample_design_v2
        for facts in parents[1:]:
            _require_same_pool_development_facts(first, facts)
            if facts.sample_design_v2 is not None:
                if selected_v2 is None:
                    selected_v2 = facts.sample_design_v2
                elif _sample_design_v2_identity(
                    selected_v2
                ) != _sample_design_v2_identity(facts.sample_design_v2):
                    raise StrategyError(
                        "Voting parents resolve different SampleDesign V2"
                    )
        asset = lineage.candidate.asset
        if "sample_design_ref" not in asset:
            raise StrategyError(
                "legacy Voting candidate is not bound to a governed sample "
                "design; regenerate it before stability measurement"
            )
        sample_ref = StrategySampleDesignRef.from_value(
            asset["sample_design_ref"]
        )
        verified_identity = _verified_lineage_evidence_identity(lineage)
        target_col = _required_text(
            asset["measurement_context"]["target_col"],
            "Voting strategy Pool target",
        )
        if (
            sample_ref != first.sample_ref
            or asset["evidence_identity"] != verified_identity
            or asset["measurement_context"]["sample_context_hash"]
            != verified_identity["sample_context_hash"]
            or first.evidence_identity != verified_identity
            or first.target_col != target_col
        ):
            raise StrategyError(
                "Voting strategy Pool development binding changed"
            )
        return remember(_PoolDevelopmentLineageFacts(
            dataset=first.dataset,
            sample_ref=sample_ref,
            sample_design_v2=selected_v2,
            evidence_identity=verified_identity,
            target_col=target_col,
        ))
    raise StrategyError("unsupported strategy Pool development lineage type")


def _verified_lineage_evidence_identity(
    lineage: _CandidateLineage,
) -> dict[str, Any]:
    try:
        identity = lineage.verified_fragment["evidence"]["identity"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise StrategyError(
            "strategy Pool development evidence binding is invalid"
        ) from exc
    return _evidence_identity_from_concrete(identity)


def _evidence_identity_from_concrete(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError("strategy Pool evidence identity must be an object")
    expected = {
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
    }
    if not expected.issubset(set(value)):
        raise StrategyError("strategy Pool evidence identity is incomplete")
    return {
        "dataset_id": _required_text(
            value["dataset_id"],
            "strategy Pool evidence dataset_id",
        ),
        "dataset_content_hash": _required_hash(
            value["dataset_content_hash"],
            "strategy Pool evidence dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"],
            "strategy Pool evidence workspace_revision",
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"],
            "strategy Pool evidence workspace_generation",
        ),
        "semantic_mapping_hash": _required_hash(
            value["semantic_mapping_hash"],
            "strategy Pool evidence semantic_mapping_hash",
        ),
        "sample_context_hash": _required_hash(
            value["sample_context_hash"],
            "strategy Pool evidence sample_context_hash",
        ),
    }


def _pool_development_dataset(
    dataset: object,
) -> StrategyPoolDevelopmentDatasetBinding:
    try:
        columns = tuple(
            _required_text(column, "strategy Pool dataset column")
            for column in dataset.columns
        )
        path = dataset.path
        binding = StrategyPoolDevelopmentDatasetBinding(
            task_id=_required_text(
                dataset.task_id,
                "strategy Pool dataset task_id",
            ),
            dataset_id=_required_text(
                dataset.dataset_id,
                "strategy Pool dataset_id",
            ),
            source_path=_required_text(
                dataset.source_path,
                "strategy Pool dataset source_path",
            ),
            path=path,
            content_hash=_required_hash(
                dataset.content_hash,
                "strategy Pool dataset content_hash",
            ),
            registry_metadata_hash=_required_hash(
                dataset.registry_metadata_hash,
                "strategy Pool dataset registry_metadata_hash",
            ),
            columns=columns,
            row_count=_non_negative_int(
                dataset.row_count,
                "strategy Pool dataset row_count",
            ),
        )
    except (AttributeError, TypeError) as exc:
        raise StrategyError(
            "strategy Pool development dataset binding is invalid"
        ) from exc
    if not isinstance(path, Path) or not path.is_absolute():
        raise StrategyError(
            "strategy Pool development dataset path changed"
        )
    return binding


def _require_same_pool_development_facts(
    first: _PoolDevelopmentLineageFacts,
    other: _PoolDevelopmentLineageFacts,
) -> None:
    if (
        first.dataset != other.dataset
        or first.sample_ref != other.sample_ref
        or first.evidence_identity != other.evidence_identity
        or first.target_col != other.target_col
    ):
        raise StrategyError(
            "strategy Pool candidates do not share one exact development sample"
        )


def _sample_design_v2_identity(
    binding: StrategySampleDesignV2ArtifactBinding | None,
) -> tuple[object, ...] | None:
    if binding is None:
        return None
    design = binding.bundle["sample_design"]
    header = binding.membership["header"]
    source = binding.source_binding
    return (
        binding.task_id,
        binding.membership_artifact_id,
        binding.membership_artifact_content_hash,
        binding.bundle_artifact_id,
        binding.bundle_artifact_content_hash,
        binding.bundle["bundle_id"],
        design["sample_design_id"],
        design["content_hash"],
        header["membership_id"],
        header["content_hash"],
        header["payload_hash"],
        header["row_count"],
        source.dataset_id,
        source.dataset_content_hash,
        source.workspace_revision,
        source.workspace_generation,
        source.semantic_mapping_hash,
        source.legacy.reference,
    )


def load_verified_univariate_candidate_lineage(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> VerifiedUnivariateCandidateLineageBinding:
    """Load one exact task-owned refined univariate asset for downstream use."""

    try:
        normalized_task = _required_text(task_id, "task_id")
        lineage = _load_candidate_lineage(
            runtime,
            task_id=normalized_task,
            artifact_id=_required_hash(artifact_id, "artifact_id"),
            expected_content_hash=_required_hash(
                expected_content_hash,
                "expected_content_hash",
            ),
            expected_asset_id=_required_text(
                expected_asset_id,
                "expected_asset_id",
            ),
            expected_asset_hash=_required_hash(
                expected_asset_hash,
                "expected_asset_hash",
            ),
        )
        if not isinstance(lineage, _UnivariateCandidateLineage):
            raise StrategyError(
                "candidate source must be a refined univariate asset"
            )
        binding = VerifiedUnivariateCandidateLineageBinding(
            task_id=normalized_task,
            lineage=lineage,
            tasks_root=Path(runtime.settings.tasks_dir).absolute(),
            datasets_root=Path(runtime.settings.datasets_dir).absolute(),
            db_path=Path(runtime.settings.db_path).absolute(),
        )
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_verified_univariate_candidate_lineage_on_connection(
                conn,
                binding,
            )
            conn.commit()
        return binding
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def require_verified_univariate_candidate_lineage_on_connection(
    conn,
    binding: VerifiedUnivariateCandidateLineageBinding,
) -> None:
    """Re-authenticate one refined candidate lineage under a caller-owned lock."""

    if not isinstance(binding, VerifiedUnivariateCandidateLineageBinding):
        raise StrategyError("verified univariate candidate binding is invalid")
    _require_binding_connection(
        conn,
        db_path=binding.db_path,
        name="verified univariate candidate",
    )
    task_id = _required_text(binding.task_id, "candidate binding.task_id")
    task = conn.execute(
        "SELECT id, task_type FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if (
        task is None
        or str(task["id"]) != task_id
        or str(task["task_type"]) != "strategy"
    ):
        raise StrategyError("verified univariate candidate task ownership changed")
    if (
        binding.tasks_root != binding.tasks_root.absolute()
        or binding.datasets_root != binding.datasets_root.absolute()
        or binding.lineage.asset_record.task_id != task_id
    ):
        raise StrategyError("verified univariate candidate binding changed")
    cache = _LineageCache.empty()
    _require_lineage_on_connection(
        conn,
        binding.lineage,
        tasks_root=binding.tasks_root,
        cache=cache,
    )
    _require_lineage_dataset_paths(
        binding.lineage,
        datasets_root=binding.datasets_root,
    )


def require_strategy_candidate_pool_artifact_binding_on_connection(
    conn,
    binding: StrategyCandidatePoolArtifactBinding,
) -> None:
    """Re-authenticate one current Pool while a downstream writer owns the lock."""

    if not isinstance(binding, StrategyCandidatePoolArtifactBinding):
        raise StrategyError("strategy candidate pool artifact binding is invalid")
    _require_binding_connection(
        conn,
        db_path=binding.db_path,
        name="strategy candidate pool",
    )
    task_id = _required_text(binding.task_id, "pool binding.task_id")
    strategy_type = _required_text(
        binding.strategy_type,
        "pool binding.strategy_type",
    )
    task = conn.execute(
        "SELECT id, task_type FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if (
        task is None
        or str(task["id"]) != task_id
        or str(task["task_type"]) != "strategy"
    ):
        raise StrategyError("strategy candidate pool task ownership changed")

    repository = StrategyCandidatePoolRepository(binding.db_path)
    current = repository.get_current_on_connection(
        conn,
        task_id,
        strategy_type,
    )
    if current is None:
        raise StrategyError("strategy candidate pool is no longer current")
    pool = validate_strategy_pool(current)
    bound_pool = validate_strategy_pool(binding.pool)
    if pool != bound_pool:
        raise StrategyError("strategy candidate pool revision is no longer current")
    if bound_pool["task_id"] != task_id or bound_pool["strategy_type"] != strategy_type:
        raise StrategyError("strategy candidate pool binding identity changed")
    if not isinstance(binding.lineages, tuple) or len(binding.lineages) != len(
        bound_pool["entries"]
    ):
        raise StrategyError("strategy candidate pool lineage binding changed")
    if binding.tasks_root != binding.tasks_root.absolute():
        raise StrategyError("strategy candidate pool task root changed")
    if binding.datasets_root != binding.datasets_root.absolute():
        raise StrategyError("strategy candidate pool dataset root changed")

    artifact_binding = _pool_artifact_source_binding(binding)
    expected_origin = _ORIGIN_BY_OPERATION[bound_pool["operation"]["kind"]]
    expected_provenance = _pool_provenance(bound_pool)
    expected_path = (
        binding.tasks_root
        / task_id
        / "strategy_candidate_pools"
        / _pool_filename(bound_pool)
    )
    expected_content_hash = strategy_pool_artifact_content_hash(bound_pool)
    if (
        binding.artifact_path != expected_path
        or binding.artifact_origin_tool != expected_origin
        or binding.artifact_provenance != expected_provenance
        or binding.artifact_provenance_json != _canonical_json(expected_provenance)
        or not hmac.compare_digest(
            binding.artifact_content_hash,
            expected_content_hash,
        )
    ):
        raise StrategyError("strategy candidate pool artifact binding changed")
    _require_pool_revision_artifact_link_on_connection(
        conn,
        pool=bound_pool,
        artifact_id=binding.artifact_id,
        artifact_content_hash=binding.artifact_content_hash,
    )
    _require_parent_pool_artifact_on_connection(
        conn,
        artifact_binding,
        snapshot=bound_pool,
        tasks_root=binding.tasks_root,
    )

    cache = _LineageCache.empty()
    for entry, lineage in zip(
        bound_pool["entries"],
        binding.lineages,
        strict=True,
    ):
        if lineage.source_binding != entry["source"]:
            raise StrategyError(
                f"pool source binding drifted for rule_id: {entry['rule_id']}"
            )
        _require_lineage_on_connection(
            conn,
            lineage,
            tasks_root=binding.tasks_root,
            cache=cache,
        )
        _require_lineage_dataset_paths(
            lineage,
            datasets_root=binding.datasets_root,
        )
    _require_cross_matrix_groups_disjoint(binding.lineages)
    compiled = compile_strategy_pool(bound_pool)
    if compiled != binding.compiled_design:
        raise StrategyError("strategy candidate pool compiled design changed")


def _validate_add_inputs(inputs: object) -> dict[str, Any]:
    normalized = _validate_inputs(
        inputs,
        allowed=_ADD_INPUT_FIELDS,
        required=_ADD_REQUIRED_FIELDS,
        tool_name="add_candidate_to_pool",
    )
    normalized = _normalize_cas_inputs(normalized)
    normalized.update(
        {
            "source_artifact_id": _required_text(
                normalized["source_artifact_id"], "source_artifact_id"
            ),
            "expected_artifact_content_hash": _required_hash(
                normalized["expected_artifact_content_hash"],
                "expected_artifact_content_hash",
            ),
            "expected_asset_id": _required_text(
                normalized["expected_asset_id"], "expected_asset_id"
            ),
            "expected_asset_hash": _required_hash(
                normalized["expected_asset_hash"], "expected_asset_hash"
            ),
        }
    )
    for field in ("default_action", "action"):
        if not isinstance(normalized[field], Mapping):
            raise StrategyError(f"{field} must be an object")
        normalized[field] = _json_object(normalized[field], field)
    if "reason" in normalized:
        normalized["reason"] = _optional_text(normalized["reason"], "reason")
    if "placement_mode" in normalized:
        placement_mode = _required_text(
            normalized["placement_mode"], "placement_mode"
        )
        if placement_mode not in {
            APPEND_PLACEMENT,
            BEFORE_SELECTED_MEMBERS_PLACEMENT,
            REPLACE_SELECTED_MEMBERS_PLACEMENT,
        }:
            raise StrategyError("unsupported placement_mode")
        normalized["placement_mode"] = placement_mode
    return normalized


def _validate_inputs(
    inputs: object,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    tool_name: str,
) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise StrategyError(f"{tool_name} inputs must be an object")
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyError(f"{tool_name} input keys must be strings")
    missing = sorted(required - set(inputs))
    unexpected = sorted(set(inputs) - allowed)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(f"invalid {tool_name} inputs ({'; '.join(details)})")
    return _json_object(inputs, f"{tool_name} inputs")


def _normalize_common_mutation_inputs(
    inputs: dict[str, Any], *, include_rule: bool = False
) -> dict[str, Any]:
    normalized = _normalize_cas_inputs(inputs)
    if include_rule:
        normalized["rule_id"] = _required_text(normalized["rule_id"], "rule_id")
    if "reason" in normalized:
        normalized["reason"] = _optional_text(normalized["reason"], "reason")
    return normalized


def _normalize_cas_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    normalized["strategy_type"] = _required_text(
        normalized["strategy_type"], "strategy_type"
    )
    normalized["expected_pool_revision"] = _non_negative_int(
        normalized["expected_pool_revision"], "expected_pool_revision"
    )
    normalized["expected_pool_snapshot_hash"] = _required_hash(
        normalized["expected_pool_snapshot_hash"],
        "expected_pool_snapshot_hash",
    )
    return normalized


def _mutation_base(runtime, ctx, inputs):
    task_id = _required_text(ctx.task_id, "task_id")
    repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
    legacy_archive = repository.get_archived_legacy_draft(
        task_id, inputs["strategy_type"]
    )
    base = _expected_base_pool(
        repository,
        task_id=task_id,
        strategy_type=inputs["strategy_type"],
        expected_revision=inputs["expected_pool_revision"],
        expected_snapshot_hash=inputs["expected_pool_snapshot_hash"],
    )
    if base is None:
        if legacy_archive is not None:
            raise StrategyPoolLegacyDraftNeedsRebuildError(legacy_archive)
        raise StrategyError("strategy candidate pool not found at expected revision")
    return task_id, repository, base, legacy_archive


def _expected_base_pool(
    repository: StrategyCandidatePoolRepository,
    *,
    task_id: str,
    strategy_type: str,
    expected_revision: int,
    expected_snapshot_hash: str,
) -> dict[str, Any] | None:
    if expected_revision == ABSENT_POOL_REVISION:
        if not hmac.compare_digest(expected_snapshot_hash, ABSENT_POOL_SNAPSHOT_HASH):
            raise StrategyError(
                "pool revision 0 requires the canonical absent snapshot hash"
            )
        return None
    persisted = repository.get_revision(task_id, strategy_type, expected_revision)
    if persisted is None:
        raise StrategyError("stale strategy candidate pool revision")
    pool = validate_strategy_pool(persisted)
    if not hmac.compare_digest(pool["snapshot_hash"], expected_snapshot_hash):
        raise StrategyError("stale strategy candidate pool snapshot hash")
    return pool


def _entry_id_for_rule(pool: Mapping[str, Any], rule_id: str) -> str:
    matches = [entry for entry in pool["entries"] if entry["rule_id"] == rule_id]
    if len(matches) != 1:
        raise StrategyError(f"unknown rule_id in strategy pool: {rule_id}")
    return str(matches[0]["entry_id"])


def _load_pool_lineages(
    runtime,
    *,
    task_id: str,
    pool: Mapping[str, Any] | None,
    cache: _LineageCache | None = None,
) -> list[_CandidateLineage]:
    if pool is None:
        return []
    normalized = validate_strategy_pool(pool)
    lineage_cache = cache if cache is not None else _LineageCache.empty()
    lineages: list[_CandidateLineage] = []
    for entry in normalized["entries"]:
        source = entry["source"]
        lineage = _load_candidate_lineage(
            runtime,
            task_id=task_id,
            artifact_id=source["artifact_id"],
            expected_content_hash=source["artifact_content_hash"],
            expected_asset_id=source["asset_id"],
            expected_asset_hash=source["asset_hash"],
            cache=lineage_cache,
        )
        if lineage.source_binding != source:
            raise StrategyError(
                f"pool source binding drifted for rule_id: {entry['rule_id']}"
            )
        if (
            isinstance(lineage, _VotingCandidateLineage)
            and lineage.parent_pool["revision"] >= normalized["revision"]
        ):
            raise StrategyError(
                "Voting candidate must originate from an earlier Pool revision"
            )
        lineages.append(lineage)
    _require_cross_matrix_groups_disjoint(lineages)
    return lineages


def _load_candidate_lineage(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    cache: _LineageCache | None = None,
) -> _CandidateLineage:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if record is None:
        raise StrategyError(f"candidate source artifact not found: {artifact_id}")
    live = _normalize_source_record(record)
    triple = (
        live.kind,
        live.origin_tool,
        live.provenance.get("schema_version"),
    )
    univariate_triple = (
        ASSET_ARTIFACT_KIND,
        ASSET_ORIGIN_TOOL,
        ASSET_ARTIFACT_SCHEMA_VERSION,
    )
    automatic_leaf_triple = (
        AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
        AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION,
    )
    voting_triple = (
        VOTING_CANDIDATE_ARTIFACT_KIND,
        VOTING_CANDIDATE_ORIGIN_TOOL,
        VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    )
    legacy_voting_triple = (
        VOTING_CANDIDATE_ARTIFACT_KIND,
        VOTING_CANDIDATE_ORIGIN_TOOL,
        VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1,
    )
    cross_matrix_selection_triple = (
        CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
        CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION,
    )
    cross_matrix_asset_triple = (
        CROSS_MATRIX_ASSET_ARTIFACT_KIND,
        CROSS_MATRIX_ASSET_ORIGIN_TOOL,
        CROSS_MATRIX_ASSET_ARTIFACT_SCHEMA_VERSION,
    )
    cross_matrix_v2_asset_triple = (
        CROSS_MATRIX_ASSET_ARTIFACT_KIND,
        CROSS_MATRIX_ASSET_ORIGIN_TOOL,
        CROSS_MATRIX_ASSET_ARTIFACT_V2_SCHEMA_VERSION,
    )
    scorecard_selection_triple = (
        SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
    )
    scorecard_asset_triple = (
        SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    )
    if triple == univariate_triple:
        return _load_univariate_candidate_lineage(
            runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
    if triple == automatic_leaf_triple:
        return _load_automatic_tree_candidate_lineage(
            runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
    if triple == cross_matrix_selection_triple:
        return _load_cross_matrix_candidate_lineage(
            runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
    if triple in {cross_matrix_asset_triple, cross_matrix_v2_asset_triple}:
        raise StrategyError(
            "complete Cross Matrix assets cannot be admitted directly; "
            "materialize a cell selection first"
        )
    if triple == scorecard_selection_triple:
        return _load_scorecard_candidate_lineage(
            runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
        )
    if triple == scorecard_asset_triple:
        raise StrategyError(
            "complete scorecard band assets cannot be admitted directly; "
            "materialize a cutoff selection first"
        )
    if triple in {voting_triple, legacy_voting_triple}:
        return _load_voting_candidate_lineage(
            runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
    raise StrategyError(
        "strategy pool contains an unsupported candidate fragment adapter triple"
    )


def _load_univariate_candidate_lineage(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    cache: _LineageCache,
) -> _UnivariateCandidateLineage:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if record is None:
        raise StrategyError(f"candidate asset artifact not found: {artifact_id}")
    asset_record = _normalize_source_record(record)
    if asset_record.task_id != task_id:
        raise StrategyError("candidate asset artifact belongs to another task")
    if asset_record.kind != ASSET_ARTIFACT_KIND:
        raise StrategyError("source artifact must be strategy_candidate_asset_json")
    if asset_record.origin_tool != ASSET_ORIGIN_TOOL:
        raise StrategyError("candidate asset artifact origin_tool is invalid")
    if not hmac.compare_digest(asset_record.content_hash, expected_content_hash):
        raise StrategyError("candidate asset artifact content hash changed")
    _require_exact_fields(
        asset_record.provenance,
        _ASSET_PROVENANCE_FIELDS,
        "candidate asset artifact provenance",
    )
    expected_path = (
        Path(runtime.settings.tasks_dir)
        / task_id
        / "strategy_candidate_assets"
        / f"{expected_asset_id}_{expected_content_hash[:12]}.json"
    )
    if asset_record.path != expected_path:
        raise StrategyError("candidate asset artifact path is not canonical")
    _require_regular_artifact_path(
        asset_record.path, root=Path(runtime.settings.tasks_dir)
    )
    _require_file_content_hash(
        asset_record.path,
        asset_record.content_hash,
        "candidate asset artifact content hash drifted",
    )
    asset = _read_canonical_asset(asset_record.path)
    if asset["asset_id"] != expected_asset_id:
        raise StrategyError("candidate asset artifact asset_id does not match")
    if not hmac.compare_digest(asset["asset_hash"], expected_asset_hash):
        raise StrategyError("candidate asset artifact asset_hash does not match")
    provenance = asset_record.provenance
    if (
        provenance["schema_version"] != ASSET_ARTIFACT_SCHEMA_VERSION
        or provenance["producer_version"] != asset["producer_version"]
    ):
        raise StrategyError("candidate asset artifact provenance contract is invalid")
    parent = asset["parent"]
    source_evidence = parent["source_evidence"]
    comparisons = {
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_id": parent["candidate_id"],
        "evidence_hash": parent["evidence_hash"],
        "source_artifact_id": source_evidence["artifact_id"],
        "source_artifact_content_hash": source_evidence["content_hash"],
        "feature": asset["feature"],
        "method": asset["method"],
    }
    for field, expected in comparisons.items():
        if provenance[field] != expected:
            raise StrategyError(
                f"candidate asset artifact provenance {field} does not match asset"
            )
    parent_record = _load_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=source_evidence["artifact_id"],
        expected_content_hash=source_evidence["content_hash"],
        expected_candidate_id=parent["candidate_id"],
        expected_evidence_hash=parent["evidence_hash"],
    )
    evidence = _read_canonical_parent_evidence(parent_record.path)
    _require_report_binding(
        evidence,
        source=parent_record,
        task_id=task_id,
        expected_candidate_id=parent["candidate_id"],
        expected_evidence_hash=parent["evidence_hash"],
    )
    _require_asset_binding(
        asset,
        evidence=evidence,
        source=parent_record,
        feature=asset["feature"],
        method=asset["method"],
    )
    identity = evidence["identity"]
    dataset_key = (identity["dataset_id"], identity["dataset_content_hash"])
    dataset = cache.univariate_datasets.get(dataset_key)
    if dataset is None:
        dataset = _load_dataset_binding(
            runtime,
            evidence=evidence,
            source=parent_record,
        )
        cache.univariate_datasets[dataset_key] = dataset
    else:
        parameters = evidence["generation"]["parameters"]
        expected_metadata_hash = parameters.get("registry_metadata_hash")
        if (
            dataset.task_id != identity["task_id"]
            or not hmac.compare_digest(
                dataset.registry_metadata_hash,
                str(expected_metadata_hash or ""),
            )
            or not hmac.compare_digest(
                dataset.registry_metadata_hash,
                str(parent_record.provenance.get("registry_metadata_hash") or ""),
            )
        ):
            raise StrategyError(
                "candidate source dataset registry metadata changed"
            )
    if identity["task_id"] != task_id:
        raise StrategyError("candidate evidence belongs to another task")
    if provenance["dataset_id"] != identity["dataset_id"] or not hmac.compare_digest(
        provenance["dataset_content_hash"],
        identity["dataset_content_hash"],
    ):
        raise StrategyError(
            "candidate asset artifact dataset provenance does not match evidence"
        )
    effect = asset["effect"]
    source_binding = {
        "artifact_id": asset_record.artifact_id,
        "kind": asset_record.kind,
        "content_hash": asset_record.content_hash,
        "origin_tool": asset_record.origin_tool,
        "artifact_schema_version": provenance["schema_version"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_kind": asset["asset_type"],
        "fragment_id": asset["rule"]["rule_id"],
        "effect_id": effect["effect_id"],
        "effect_stage": asset["effect_stage"],
        "validation_status": asset["validation_status"],
        "parent_candidate_id": parent["candidate_id"],
        "parent_evidence_hash": parent["evidence_hash"],
        "evidence_identity": {
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
        },
    }
    verified_fragment = univariate_asset_to_verified_fragment(
        asset,
        source_binding=source_binding,
        candidate_evidence=evidence,
    )
    generic_source, _rule_id, _execution = verified_fragment_pool_parts(
        verified_fragment
    )
    return _UnivariateCandidateLineage(
        asset_record=asset_record,
        asset=asset,
        parent_record=parent_record,
        evidence=evidence,
        dataset=dataset,
        verified_fragment=verified_fragment,
        source_binding=generic_source,
    )


def _load_automatic_tree_candidate_lineage(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    cache: _LineageCache,
) -> _AutomaticTreeCandidateLineage:
    selection = load_verified_automatic_tree_leaf_selection_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_content_hash=expected_content_hash,
        expected_asset_id=expected_asset_id,
        expected_asset_hash=expected_asset_hash,
    )
    tree_pointer = selection.selection["tree_artifact"]
    tree_key = (
        tree_pointer["artifact_id"],
        tree_pointer["content_hash"],
    )
    tree = cache.trees.get(tree_key)
    if tree is None:
        tree_asset = selection.selection["tree_asset"]
        tree = load_verified_automatic_tree_source_artifact(
            runtime,
            task_id=task_id,
            artifact_id=tree_pointer["artifact_id"],
            expected_content_hash=tree_pointer["content_hash"],
            expected_asset_id=tree_asset["asset_id"],
            expected_asset_hash=tree_asset["asset_hash"],
            expected_tree_result_hash=tree_asset["tree_result_hash"],
        )
        cache.trees[tree_key] = tree
    dataset = _load_automatic_tree_dataset(
        runtime,
        task_id=task_id,
        tree=tree,
        cache=cache,
    )
    verified_fragment, source_binding = _replay_automatic_tree_lineage(
        selection,
        tree,
    )
    return _AutomaticTreeCandidateLineage(
        selection=selection,
        tree=tree,
        dataset=dataset,
        verified_fragment=verified_fragment,
        source_binding=source_binding,
    )


def _load_automatic_tree_dataset(
    runtime,
    *,
    task_id: str,
    tree: VerifiedAutomaticTreeSource,
    cache: _LineageCache,
) -> _AutomaticTreeDatasetBinding:
    identity = tree.asset["identity"]
    dataset_id = str(identity["dataset_id"])
    content_hash = str(identity["dataset_content_hash"])
    cache_key = (dataset_id, content_hash)
    cached = cache.datasets.get(cache_key)
    if cached is not None:
        if cached.task_id != task_id or not hmac.compare_digest(
            cached.registry_metadata_hash,
            str(identity["registry_metadata_hash"]),
        ):
            raise StrategyError("automatic-tree source dataset identity changed")
        return cached
    try:
        dataset = runtime.registry.get(dataset_id)
        path = Path(runtime.registry.resolve_verified_path(dataset_id))
    except (DatasetContentDriftError, KeyError, OSError, TypeError, ValueError) as exc:
        raise StrategyError(
            f"automatic-tree source dataset not found or drifted: {dataset_id}"
        ) from exc
    if str(dataset.task_id) != task_id:
        raise StrategyError("automatic-tree source dataset belongs to another task")
    registered_hash = str(dataset.content_hash or "")
    if not hmac.compare_digest(registered_hash, content_hash):
        raise StrategyError("automatic-tree source dataset content hash changed")
    _require_file_content_hash(
        path,
        content_hash,
        "automatic-tree source dataset content hash drifted",
    )
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=dataset_id,
            expected_content_hash=content_hash,
        )
    if not hmac.compare_digest(
        metadata_hash,
        str(identity["registry_metadata_hash"]),
    ):
        raise StrategyError("automatic-tree source dataset registry metadata changed")
    binding = _AutomaticTreeDatasetBinding(
        dataset_id=dataset_id,
        task_id=task_id,
        source_path=str(dataset.source_path),
        path=path,
        content_hash=content_hash,
        registry_metadata_hash=metadata_hash,
        columns=tuple(str(profile.name) for profile in dataset.columns),
        row_count=int(dataset.row_count),
    )
    cache.datasets[cache_key] = binding
    return binding


def _replay_automatic_tree_lineage(
    selection: VerifiedAutomaticTreeLeafSelection,
    tree: VerifiedAutomaticTreeSource,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verified_fragment = automatic_tree_leaf_fragment_to_verified_candidate_fragment(
        selection.selection,
        tree.asset,
        selection_artifact_binding=selection.replay_binding(),
        tree_artifact_binding=tree.builder_binding(),
    )
    source_binding, _rule_id, _execution = verified_fragment_pool_parts(
        verified_fragment
    )
    return verified_fragment, source_binding


def _load_cross_matrix_candidate_lineage(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    cache: _LineageCache,
) -> _CrossMatrixCandidateLineage:
    selection = load_verified_cross_matrix_cell_selection_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_content_hash=expected_content_hash,
        expected_asset_id=expected_asset_id,
        expected_asset_hash=expected_asset_hash,
    )
    source_pointer = selection.selection["source_artifact"]
    source_asset = selection.selection["source_asset"]
    source_candidate = selection.selection["source_candidate"]
    matrix_key = (
        source_pointer["artifact_id"],
        source_pointer["content_hash"],
    )
    matrix = cache.matrices.get(matrix_key)
    if matrix is None:
        matrix = load_verified_cross_matrix_source_artifact(
            runtime,
            task_id=task_id,
            artifact_id=source_pointer["artifact_id"],
            expected_content_hash=source_pointer["content_hash"],
            expected_asset_id=source_asset["asset_id"],
            expected_asset_hash=source_asset["asset_hash"],
            expected_candidate_id=source_candidate["candidate_id"],
            expected_evidence_hash=source_candidate["evidence_hash"],
        )
        cache.matrices[matrix_key] = matrix
    elif (
        matrix.asset["asset_id"] != source_asset["asset_id"]
        or not hmac.compare_digest(
            matrix.asset["asset_hash"], source_asset["asset_hash"]
        )
        or matrix.asset["candidate_evidence"]["candidate_id"]
        != source_candidate["candidate_id"]
        or not hmac.compare_digest(
            matrix.asset["candidate_evidence"]["evidence_hash"],
            source_candidate["evidence_hash"],
        )
    ):
        raise StrategyError("cached Cross Matrix source binding changed")

    parent = matrix.asset["parent"]
    provenance = matrix.provenance
    parent_record = _load_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=provenance["source_artifact_id"],
        expected_content_hash=provenance["source_artifact_content_hash"],
        expected_candidate_id=parent["candidate_id"],
        expected_evidence_hash=parent["evidence_hash"],
    )
    evidence = _read_canonical_parent_evidence(parent_record.path)
    _require_report_binding(
        evidence,
        source=parent_record,
        task_id=task_id,
        expected_candidate_id=parent["candidate_id"],
        expected_evidence_hash=parent["evidence_hash"],
    )
    if rebuild_cross_matrix_candidate_asset(matrix.asset, evidence) != matrix.asset:
        raise StrategyError("Cross Matrix source does not replay its parent evidence")
    identity = evidence["identity"]
    dataset_key = (identity["dataset_id"], identity["dataset_content_hash"])
    dataset = cache.univariate_datasets.get(dataset_key)
    if dataset is None:
        dataset = _load_dataset_binding(
            runtime,
            evidence=evidence,
            source=parent_record,
        )
        cache.univariate_datasets[dataset_key] = dataset
    else:
        parameters = evidence["generation"]["parameters"]
        expected_metadata_hash = parameters.get("registry_metadata_hash")
        if (
            dataset.task_id != identity["task_id"]
            or not hmac.compare_digest(
                dataset.registry_metadata_hash,
                str(expected_metadata_hash or ""),
            )
            or not hmac.compare_digest(
                dataset.registry_metadata_hash,
                str(parent_record.provenance.get("registry_metadata_hash") or ""),
            )
        ):
            raise StrategyError("Cross Matrix source dataset identity changed")
    verified_fragment = cross_matrix_cell_selection_to_verified_candidate_fragment(
        selection.selection,
        matrix.asset,
        selection_artifact_binding=selection.replay_binding(),
        source_artifact_binding=matrix.source_binding(),
    )
    source_binding, _rule_id, _execution = verified_fragment_pool_parts(
        verified_fragment
    )
    return _CrossMatrixCandidateLineage(
        selection=selection,
        matrix=matrix,
        parent_record=parent_record,
        evidence=evidence,
        dataset=dataset,
        verified_fragment=verified_fragment,
        source_binding=source_binding,
    )


def _require_cross_matrix_groups_disjoint(
    lineages: Sequence[_CandidateLineage],
) -> None:
    seen_by_matrix: dict[tuple[str, str], set[str]] = {}
    overlaps: set[str] = set()
    for lineage in lineages:
        if not isinstance(lineage, _CrossMatrixCandidateLineage):
            continue
        key = (lineage.matrix.asset["asset_id"], lineage.matrix.asset["asset_hash"])
        seen = seen_by_matrix.setdefault(key, set())
        selected = set(lineage.selection.selection["cell_ids"])
        overlaps.update(seen & selected)
        seen.update(selected)
    if overlaps:
        raise StrategyError(
            "Cross Matrix cell groups from the same matrix must be disjoint; "
            "overlapping cell_ids: " + ", ".join(sorted(overlaps))
        )


def _load_scorecard_candidate_lineage(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> _ScorecardCandidateLineage:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if record is None:
        raise StrategyError("scorecard cutoff selection artifact not found")
    source_record = _normalize_source_record(record)
    provenance = source_record.provenance
    selection = load_scorecard_cutoff_selection_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_artifact_content_hash=expected_content_hash,
        expected_selection_id=_required_text(
            provenance.get("selection_id"),
            "scorecard selection provenance.selection_id",
        ),
        expected_selection_hash=_required_hash(
            provenance.get("selection_hash"),
            "scorecard selection provenance.selection_hash",
        ),
    )
    asset = selection.source_asset_binding
    if asset.asset["asset_id"] != expected_asset_id or not hmac.compare_digest(
        asset.asset["asset_hash"],
        expected_asset_hash,
    ):
        raise StrategyError(
            "scorecard cutoff selection source asset identity changed"
        )
    source = asset.sample_design.source_binding
    identity = asset.asset["identity"]
    if (
        selection.task_id != task_id
        or asset.task_id != task_id
        or identity["task_id"] != task_id
        or source.task_id != task_id
        or source.dataset_id != identity["dataset_id"]
        or not hmac.compare_digest(
            source.dataset_content_hash,
            identity["dataset_content_hash"],
        )
        or source.workspace_revision != identity["workspace_revision"]
        or source.workspace_generation != identity["workspace_generation"]
        or not hmac.compare_digest(
            source.semantic_mapping_hash,
            identity["semantic_mapping_hash"],
        )
    ):
        raise StrategyError(
            "scorecard cutoff selection sample or dataset identity changed"
        )
    verified_fragment = scorecard_cutoff_selection_to_verified_candidate_fragment(
        selection.selection,
        asset.asset,
        selection_artifact_binding=selection.to_domain_binding(),
        source_artifact_binding=asset.to_domain_binding(),
    )
    source_binding, _rule_id, _execution = verified_fragment_pool_parts(
        verified_fragment
    )
    return _ScorecardCandidateLineage(
        selection=selection,
        asset=asset,
        dataset=_ScorecardDatasetBinding(
            dataset_id=source.dataset_id,
            task_id=source.task_id,
            source_path=source.dataset_source_path,
            path=source.dataset_path,
            content_hash=source.dataset_content_hash,
            registry_metadata_hash=source.dataset_registry_metadata_hash,
            columns=source.columns,
            row_count=source.row_count,
        ),
        verified_fragment=verified_fragment,
        source_binding=source_binding,
    )


def _load_voting_candidate_lineage(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    cache: _LineageCache,
) -> _VotingCandidateLineage:
    key = (artifact_id, expected_content_hash)
    cached = cache.voting.get(key)
    if cached is not None:
        if (
            cached.candidate.asset["asset_id"] != expected_asset_id
            or not hmac.compare_digest(
                cached.candidate.asset["asset_hash"], expected_asset_hash
            )
        ):
            raise StrategyError("cached Voting candidate binding changed")
        return cached
    if key in cache.voting_in_progress:
        raise StrategyError("Voting candidate Pool ancestry contains a cycle")
    if len(cache.voting_in_progress) >= _MAX_VOTING_ANCESTRY_DEPTH:
        raise StrategyError("Voting candidate Pool ancestry exceeds depth budget")
    if len(cache.voting) >= _MAX_VOTING_ANCESTRY_NODES:
        raise StrategyError("Voting candidate Pool ancestry exceeds node budget")
    cache.voting_in_progress.add(key)
    try:
        candidate = load_verified_voting_candidate_artifact(
            runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=expected_content_hash,
            expected_asset_id=expected_asset_id,
            expected_asset_hash=expected_asset_hash,
        )
        pool_ref = candidate.asset["pool_ref"]
        if pool_ref["task_id"] != task_id:
            raise StrategyError("Voting candidate parent Pool belongs to another task")
        repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
        parent = repository.get_revision_by_id(
            task_id,
            pool_ref["strategy_type"],
            pool_ref["revision_id"],
        )
        if parent is None:
            raise StrategyError("Voting candidate parent Pool revision not found")
        parent = validate_strategy_pool(parent)
        verify_voting_candidate_asset_against_pool(candidate.asset, parent)
        parent_artifact = _normalize_source_record(
            _load_pool_artifact(runtime, task_id=task_id, snapshot=parent)
        )
        if (
            candidate.provenance["pool_artifact_id"]
            != parent_artifact.artifact_id
            or not hmac.compare_digest(
                candidate.provenance["pool_artifact_content_hash"],
                parent_artifact.content_hash,
            )
        ):
            raise StrategyError("Voting candidate parent Pool artifact changed")
        parent_by_id = {entry["entry_id"]: entry for entry in parent["entries"]}
        selected_parent_lineages: list[_CandidateLineage] = []
        for selected in candidate.asset["selected_entries"]:
            parent_entry = parent_by_id.get(selected["entry_id"])
            if parent_entry is None:
                raise StrategyError(
                    "Voting candidate selected parent entry no longer exists"
                )
            source = parent_entry["source"]
            selected_lineage = _load_candidate_lineage(
                runtime,
                task_id=task_id,
                artifact_id=source["artifact_id"],
                expected_content_hash=source["artifact_content_hash"],
                expected_asset_id=source["asset_id"],
                expected_asset_hash=source["asset_hash"],
                cache=cache,
            )
            if selected_lineage.source_binding != source:
                raise StrategyError(
                    "Voting selected parent source binding changed"
                )
            selected_parent_lineages.append(selected_lineage)
        parent_lineages = tuple(selected_parent_lineages)
        verified_fragment = voting_candidate_to_verified_fragment(
            candidate.asset,
            artifact_binding=candidate.artifact_binding(),
        )
        source_binding, _rule_id, _execution = verified_fragment_pool_parts(
            verified_fragment
        )
        lineage = _VotingCandidateLineage(
            candidate=candidate,
            parent_pool=parent,
            parent_pool_artifact=parent_artifact,
            parent_lineages=parent_lineages,
            verified_fragment=verified_fragment,
            source_binding=source_binding,
        )
        cache.voting[key] = lineage
        return lineage
    finally:
        cache.voting_in_progress.discard(key)


def _read_canonical_asset(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
        asset = validate_candidate_asset(parsed)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError(
            "candidate asset artifact failed strict validation"
        ) from exc
    canonical = canonical_candidate_asset_json(asset).encode("utf-8")
    if canonical != raw:
        raise StrategyError("candidate asset artifact is not canonical JSON")
    return asset


def _read_canonical_parent_evidence(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        report = strategy_candidate_report_from_json(raw)
        evidence = validate_candidate_evidence(report["candidate_evidence"])
        canonical = canonical_strategy_candidate_report_json(
            evidence,
            report["univariate_analysis"],
        )
    except (OSError, TypeError, ValueError, StrategyError) as exc:
        raise StrategyError("parent candidate report failed strict validation") from exc
    if canonical != raw:
        raise StrategyError("parent candidate report is not canonical JSON")
    return evidence


def _persist_mutation(
    runtime,
    *,
    repository: StrategyCandidatePoolRepository,
    snapshot: Mapping[str, Any],
    expected_revision: int,
    expected_snapshot_hash: str,
    lineages: Sequence[_CandidateLineage],
    inputs: Mapping[str, Any],
    legacy_archive: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = validate_strategy_pool(snapshot)
    canonical = canonical_strategy_pool_snapshot_json(normalized)
    content = canonical.encode("utf-8")
    content_hash = strategy_pool_artifact_content_hash(normalized)
    if not hmac.compare_digest(_sha256(content), content_hash):
        raise StrategyError("canonical pool artifact content hash is inconsistent")
    task_id = normalized["task_id"]
    parent_snapshot: dict[str, Any] | None = None
    parent_artifact_binding = None
    if expected_revision != ABSENT_POOL_REVISION:
        persisted_parent = repository.get_revision(
            task_id,
            normalized["strategy_type"],
            expected_revision,
        )
        if persisted_parent is None:
            raise StrategyError("parent strategy pool revision not found")
        parent_snapshot = validate_strategy_pool(persisted_parent)
        if not hmac.compare_digest(
            parent_snapshot["snapshot_hash"], expected_snapshot_hash
        ):
            raise StrategyError("parent strategy pool snapshot hash changed")
        if normalized["parent_revision_id"] != parent_snapshot["revision_id"]:
            raise StrategyError("new strategy pool revision has a mismatched parent")
        parent_artifact_binding = _normalize_source_record(
            _load_pool_artifact(
                runtime,
                task_id=task_id,
                snapshot=parent_snapshot,
            )
        )
    out_dir = Path(runtime.settings.tasks_dir) / task_id / "strategy_candidate_pools"
    _require_output_directory(out_dir, root=Path(runtime.settings.tasks_dir))
    filename = _pool_filename(normalized)
    provenance = _pool_provenance(normalized)
    origin = _ORIGIN_BY_OPERATION[normalized["operation"]["kind"]]
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, filename)
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        staged.path.write_bytes(content)
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if parent_snapshot is not None and parent_artifact_binding is not None:
                    _require_parent_pool_artifact_on_connection(
                        conn,
                        parent_artifact_binding,
                        snapshot=parent_snapshot,
                        tasks_root=Path(runtime.settings.tasks_dir),
                    )
                lineage_cache = _LineageCache.empty()
                for lineage in lineages:
                    _require_lineage_on_connection(
                        conn,
                        lineage,
                        tasks_root=Path(runtime.settings.tasks_dir),
                        cache=lineage_cache,
                    )
                uow.promote_all()
                _require_file_content_hash(
                    staged.final_path,
                    content_hash,
                    "strategy pool artifact changed before registration",
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=POOL_ARTIFACT_KIND,
                    path=str(staged.final_path),
                    content_hash=content_hash,
                    origin_tool=origin,
                    provenance=provenance,
                )
                result = repository.apply_snapshot_on_connection(
                    conn,
                    snapshot=normalized,
                    expected_revision=expected_revision,
                    expected_snapshot_hash=expected_snapshot_hash,
                    artifact_id=str(record["id"]),
                    artifact_content_hash=content_hash,
                    audit={
                        "kind": f"strategy.pool.{normalized['operation']['kind']}",
                        "target_ref": normalized["revision_id"],
                        "actor": "system",
                        "inputs_hash": _sha256(_canonical_json(inputs).encode("utf-8")),
                        "outcome": "succeeded",
                        "detail": {
                            "entry_count": len(normalized["entries"]),
                            "archived_legacy_draft": legacy_archive,
                            "warnings": (
                                []
                                if legacy_archive is None
                                else [_LEGACY_ARCHIVE_WARNING]
                            ),
                        },
                    },
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
    persisted = validate_strategy_pool(result["snapshot"])
    return {
        "schema_version": POOL_MUTATION_TOOL_SCHEMA_VERSION,
        "operation": persisted["operation"]["kind"],
        "pool_id": persisted["pool_id"],
        "revision": persisted["revision"],
        "snapshot_hash": persisted["snapshot_hash"],
        "status": persisted["status"],
        "validation_status": persisted["validation_status"],
        "entry_count": len(persisted["entries"]),
        "entries": persisted["entries"],
        "pool": persisted,
        "artifacts": [_artifact_output(record, task_id=task_id)],
        "archived_legacy_draft": legacy_archive,
        "warnings": [] if legacy_archive is None else [_LEGACY_ARCHIVE_WARNING],
    }


def _require_lineage_on_connection(
    conn,
    lineage: _CandidateLineage,
    *,
    tasks_root: Path,
    cache: _LineageCache | None = None,
) -> None:
    if isinstance(lineage, _UnivariateCandidateLineage):
        _require_univariate_lineage_on_connection(
            conn,
            lineage,
            tasks_root=tasks_root,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
        return
    if isinstance(lineage, _AutomaticTreeCandidateLineage):
        _require_automatic_tree_lineage_on_connection(
            conn,
            lineage,
            tasks_root=tasks_root,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
        return
    if isinstance(lineage, _CrossMatrixCandidateLineage):
        _require_cross_matrix_lineage_on_connection(
            conn,
            lineage,
            tasks_root=tasks_root,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
        return
    if isinstance(lineage, _ScorecardCandidateLineage):
        _require_scorecard_lineage_on_connection(
            conn,
            lineage,
        )
        return
    if isinstance(lineage, _VotingCandidateLineage):
        _require_voting_lineage_on_connection(
            conn,
            lineage,
            tasks_root=tasks_root,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
        return
    raise StrategyError("unsupported candidate lineage type")


def _require_scorecard_lineage_on_connection(
    conn,
    lineage: _ScorecardCandidateLineage,
) -> None:
    require_scorecard_cutoff_selection_artifact_binding_on_connection(
        conn,
        lineage.selection,
    )
    asset = lineage.selection.source_asset_binding
    if asset is not lineage.asset:
        raise StrategyError(
            "scorecard cutoff selection source binding changed before Pool persistence"
        )
    source = asset.sample_design.source_binding
    dataset = _ScorecardDatasetBinding(
        dataset_id=source.dataset_id,
        task_id=source.task_id,
        source_path=source.dataset_source_path,
        path=source.dataset_path,
        content_hash=source.dataset_content_hash,
        registry_metadata_hash=source.dataset_registry_metadata_hash,
        columns=source.columns,
        row_count=source.row_count,
    )
    verified_fragment = scorecard_cutoff_selection_to_verified_candidate_fragment(
        lineage.selection.selection,
        asset.asset,
        selection_artifact_binding=lineage.selection.to_domain_binding(),
        source_artifact_binding=asset.to_domain_binding(),
    )
    source_binding, _rule_id, _execution = verified_fragment_pool_parts(
        verified_fragment
    )
    if (
        dataset != lineage.dataset
        or verified_fragment != lineage.verified_fragment
        or source_binding != lineage.source_binding
    ):
        raise StrategyError(
            "scorecard candidate lineage changed before Pool persistence"
        )


def _require_univariate_lineage_on_connection(
    conn,
    lineage: _UnivariateCandidateLineage,
    *,
    tasks_root: Path,
    cache: _LineageCache,
) -> None:
    _require_source_on_connection(conn, lineage.asset_record)
    _require_source_on_connection(conn, lineage.parent_record)
    dataset_key = (lineage.dataset.dataset_id, lineage.dataset.content_hash)
    if dataset_key not in cache.datasets_verified_on_connection:
        _require_dataset_on_connection(conn, lineage.dataset)
    for binding, message in (
        (lineage.asset_record, "candidate asset artifact content hash drifted"),
        (lineage.parent_record, "parent candidate report content hash drifted"),
    ):
        _require_regular_artifact_path(binding.path, root=tasks_root)
        _require_file_content_hash(binding.path, binding.content_hash, message)
    live_asset = _read_canonical_asset(lineage.asset_record.path)
    live_evidence = _read_canonical_parent_evidence(lineage.parent_record.path)
    if live_asset != lineage.asset or live_evidence != lineage.evidence:
        raise StrategyError("candidate lineage changed before pool persistence")
    _require_asset_binding(
        live_asset,
        evidence=live_evidence,
        source=lineage.parent_record,
        feature=live_asset["feature"],
        method=live_asset["method"],
    )
    if dataset_key not in cache.datasets_verified_on_connection:
        _require_file_content_hash(
            lineage.dataset.path,
            lineage.dataset.content_hash,
            "candidate source dataset content hash drifted",
        )
        cache.datasets_verified_on_connection.add(dataset_key)


def _require_automatic_tree_lineage_on_connection(
    conn,
    lineage: _AutomaticTreeCandidateLineage,
    *,
    tasks_root: Path,
    cache: _LineageCache,
) -> None:
    selection = load_verified_automatic_tree_leaf_selection_artifact_on_connection(
        conn,
        tasks_dir=tasks_root,
        task_id=lineage.selection.task_id,
        artifact_id=lineage.selection.artifact_id,
        expected_content_hash=lineage.selection.content_hash,
        expected_asset_id=lineage.tree.asset["asset_id"],
        expected_asset_hash=lineage.tree.asset["asset_hash"],
    )
    tree_pointer = selection.selection["tree_artifact"]
    tree_key = (
        tree_pointer["artifact_id"],
        tree_pointer["content_hash"],
    )
    tree = cache.trees.get(tree_key)
    if tree is None:
        tree_asset = selection.selection["tree_asset"]
        tree = load_verified_automatic_tree_source_artifact_on_connection(
            conn,
            tasks_dir=tasks_root,
            task_id=selection.task_id,
            artifact_id=tree_pointer["artifact_id"],
            expected_content_hash=tree_pointer["content_hash"],
            expected_asset_id=tree_asset["asset_id"],
            expected_asset_hash=tree_asset["asset_hash"],
            expected_tree_result_hash=tree_asset["tree_result_hash"],
        )
        cache.trees[tree_key] = tree
    dataset_key = (lineage.dataset.dataset_id, lineage.dataset.content_hash)
    dataset = cache.datasets.get(dataset_key)
    if dataset is None:
        if dataset_key not in cache.datasets_verified_on_connection:
            _require_dataset_on_connection(conn, lineage.dataset)
            _require_file_content_hash(
                lineage.dataset.path,
                lineage.dataset.content_hash,
                "automatic-tree source dataset content hash drifted",
            )
            cache.datasets_verified_on_connection.add(dataset_key)
        dataset = lineage.dataset
        cache.datasets[dataset_key] = dataset
    verified_fragment, source_binding = _replay_automatic_tree_lineage(
        selection,
        tree,
    )
    if (
        selection != lineage.selection
        or tree != lineage.tree
        or dataset != lineage.dataset
        or verified_fragment != lineage.verified_fragment
        or source_binding != lineage.source_binding
    ):
        raise StrategyError(
            "automatic-tree candidate lineage changed before pool persistence"
        )


def _require_cross_matrix_lineage_on_connection(
    conn,
    lineage: _CrossMatrixCandidateLineage,
    *,
    tasks_root: Path,
    cache: _LineageCache,
) -> None:
    """Replay one persisted cell group from live rows while holding the Pool lock."""

    selection = load_verified_cross_matrix_cell_selection_artifact_on_connection(
        conn,
        tasks_dir=tasks_root,
        task_id=lineage.selection.task_id,
        artifact_id=lineage.selection.artifact_id,
        expected_content_hash=lineage.selection.content_hash,
        expected_asset_id=lineage.matrix.asset["asset_id"],
        expected_asset_hash=lineage.matrix.asset["asset_hash"],
    )
    source_pointer = selection.selection["source_artifact"]
    source_asset = selection.selection["source_asset"]
    source_candidate = selection.selection["source_candidate"]
    matrix_key = (
        source_pointer["artifact_id"],
        source_pointer["content_hash"],
    )
    matrix = cache.matrices.get(matrix_key)
    if matrix is None:
        matrix = load_verified_cross_matrix_source_artifact_on_connection(
            conn,
            tasks_dir=tasks_root,
            task_id=selection.task_id,
            artifact_id=source_pointer["artifact_id"],
            expected_content_hash=source_pointer["content_hash"],
            expected_asset_id=source_asset["asset_id"],
            expected_asset_hash=source_asset["asset_hash"],
            expected_candidate_id=source_candidate["candidate_id"],
            expected_evidence_hash=source_candidate["evidence_hash"],
        )
        cache.matrices[matrix_key] = matrix
    elif (
        matrix.asset["asset_id"] != source_asset["asset_id"]
        or not hmac.compare_digest(
            matrix.asset["asset_hash"], source_asset["asset_hash"]
        )
        or matrix.asset["candidate_evidence"]["candidate_id"]
        != source_candidate["candidate_id"]
        or not hmac.compare_digest(
            matrix.asset["candidate_evidence"]["evidence_hash"],
            source_candidate["evidence_hash"],
        )
    ):
        raise StrategyError("cached Cross Matrix source binding changed")

    _require_source_on_connection(conn, lineage.parent_record)
    _require_regular_artifact_path(lineage.parent_record.path, root=tasks_root)
    _require_file_content_hash(
        lineage.parent_record.path,
        lineage.parent_record.content_hash,
        "Cross Matrix parent candidate report content hash drifted",
    )
    evidence = _read_canonical_parent_evidence(lineage.parent_record.path)
    parent = matrix.asset["parent"]
    _require_report_binding(
        evidence,
        source=lineage.parent_record,
        task_id=selection.task_id,
        expected_candidate_id=parent["candidate_id"],
        expected_evidence_hash=parent["evidence_hash"],
    )
    if rebuild_cross_matrix_candidate_asset(matrix.asset, evidence) != matrix.asset:
        raise StrategyError("Cross Matrix source does not replay its parent evidence")

    dataset_key = (lineage.dataset.dataset_id, lineage.dataset.content_hash)
    dataset = cache.univariate_datasets.get(dataset_key)
    if dataset is None:
        _require_dataset_on_connection(conn, lineage.dataset)
        _require_file_content_hash(
            lineage.dataset.path,
            lineage.dataset.content_hash,
            "Cross Matrix source dataset content hash drifted",
        )
        cache.datasets_verified_on_connection.add(dataset_key)
        dataset = lineage.dataset
        cache.univariate_datasets[dataset_key] = dataset

    verified_fragment = cross_matrix_cell_selection_to_verified_candidate_fragment(
        selection.selection,
        matrix.asset,
        selection_artifact_binding=selection.replay_binding(),
        source_artifact_binding=matrix.source_binding(),
    )
    source_binding, _rule_id, _execution = verified_fragment_pool_parts(
        verified_fragment
    )
    if (
        selection != lineage.selection
        or matrix != lineage.matrix
        or evidence != lineage.evidence
        or dataset != lineage.dataset
        or verified_fragment != lineage.verified_fragment
        or source_binding != lineage.source_binding
    ):
        raise StrategyError(
            "Cross Matrix candidate lineage changed before pool persistence"
        )


def _require_voting_lineage_on_connection(
    conn,
    lineage: _VotingCandidateLineage,
    *,
    tasks_root: Path,
    cache: _LineageCache,
) -> None:
    key = (lineage.candidate.artifact_id, lineage.candidate.content_hash)
    if key in cache.voting_verified:
        return
    if key in cache.voting_in_progress:
        raise StrategyError("Voting candidate Pool ancestry contains a cycle")
    if len(cache.voting_in_progress) >= _MAX_VOTING_ANCESTRY_DEPTH:
        raise StrategyError("Voting candidate Pool ancestry exceeds depth budget")
    if (
        len(cache.voting_verified) + len(cache.voting_in_progress)
        >= _MAX_VOTING_ANCESTRY_NODES
    ):
        raise StrategyError("Voting candidate Pool ancestry exceeds node budget")
    cache.voting_in_progress.add(key)
    try:
        candidate = load_verified_voting_candidate_artifact_on_connection(
            conn,
            tasks_dir=tasks_root,
            task_id=lineage.candidate.task_id,
            artifact_id=lineage.candidate.artifact_id,
            expected_content_hash=lineage.candidate.content_hash,
            expected_asset_id=lineage.candidate.asset["asset_id"],
            expected_asset_hash=lineage.candidate.asset["asset_hash"],
        )
        pool_ref = candidate.asset["pool_ref"]
        parent = StrategyCandidatePoolRepository.get_revision_by_id_on_connection(
            conn,
            candidate.task_id,
            pool_ref["strategy_type"],
            pool_ref["revision_id"],
        )
        if parent is None:
            raise StrategyError("Voting candidate parent Pool revision not found")
        parent = validate_strategy_pool(parent)
        verify_voting_candidate_asset_against_pool(candidate.asset, parent)
        _require_parent_pool_artifact_on_connection(
            conn,
            lineage.parent_pool_artifact,
            snapshot=parent,
            tasks_root=tasks_root,
        )
        if (
            candidate.provenance["pool_artifact_id"]
            != lineage.parent_pool_artifact.artifact_id
            or not hmac.compare_digest(
                candidate.provenance["pool_artifact_content_hash"],
                lineage.parent_pool_artifact.content_hash,
            )
        ):
            raise StrategyError("Voting candidate parent Pool artifact changed")
        for parent_lineage in lineage.parent_lineages:
            _require_lineage_on_connection(
                conn,
                parent_lineage,
                tasks_root=tasks_root,
                cache=cache,
            )
        verified_fragment = voting_candidate_to_verified_fragment(
            candidate.asset,
            artifact_binding=candidate.artifact_binding(),
        )
        source_binding, _rule_id, _execution = verified_fragment_pool_parts(
            verified_fragment
        )
        if (
            candidate != lineage.candidate
            or parent != lineage.parent_pool
            or verified_fragment != lineage.verified_fragment
            or source_binding != lineage.source_binding
        ):
            raise StrategyError(
                "Voting candidate lineage changed before Pool persistence"
            )
        cache.voting_verified.add(key)
    finally:
        cache.voting_in_progress.discard(key)


def _require_parent_pool_artifact_on_connection(
    conn,
    binding,
    *,
    snapshot: Mapping[str, Any],
    tasks_root: Path,
) -> None:
    """Replay the immutable parent Pool artifact under the mutation DB lock."""

    _require_source_on_connection(conn, binding)
    raw_bytes = _read_verified_pool_file(
        binding.path,
        root=tasks_root,
        expected_content_hash=binding.content_hash,
        error_message="parent strategy pool artifact content hash drifted",
    )
    try:
        raw = raw_bytes.decode("utf-8")
        persisted = validate_strategy_pool(
            json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError("parent strategy pool artifact is invalid") from exc
    if canonical_strategy_pool_snapshot_json(persisted) != raw or persisted != snapshot:
        raise StrategyError("parent strategy pool artifact is not canonical")


def _load_pool_artifact(runtime, *, task_id: str, snapshot: Mapping[str, Any]):
    expected_path = (
        Path(runtime.settings.tasks_dir)
        / task_id
        / "strategy_candidate_pools"
        / _pool_filename(snapshot)
    )
    matches = [
        record
        for record in runtime.task_artifacts.list_for_task(task_id)
        if record["kind"] == POOL_ARTIFACT_KIND
        and Path(record["path"]) == expected_path
    ]
    if len(matches) != 1:
        raise StrategyError("current strategy pool artifact not found")
    record = matches[0]
    expected_hash = strategy_pool_artifact_content_hash(snapshot)
    if not hmac.compare_digest(record["content_hash"], expected_hash):
        raise StrategyError("current strategy pool artifact content hash changed")
    expected_origin = _ORIGIN_BY_OPERATION[snapshot["operation"]["kind"]]
    if record["origin_tool"] != expected_origin:
        raise StrategyError("current strategy pool artifact origin_tool is invalid")
    _require_exact_fields(
        record["provenance"],
        _POOL_PROVENANCE_FIELDS,
        "strategy pool artifact provenance",
    )
    if record["provenance"] != _pool_provenance(snapshot):
        raise StrategyError("current strategy pool artifact provenance changed")
    path = Path(record["path"])
    raw_bytes = _read_verified_pool_file(
        path,
        root=Path(runtime.settings.tasks_dir),
        expected_content_hash=expected_hash,
        error_message="current strategy pool artifact content hash drifted",
    )
    try:
        raw = raw_bytes.decode("utf-8")
        persisted = validate_strategy_pool(
            json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError("current strategy pool artifact is invalid") from exc
    if canonical_strategy_pool_snapshot_json(persisted) != raw or persisted != snapshot:
        raise StrategyError("current strategy pool artifact is not canonical")
    return record


def _pool_provenance(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    entries = snapshot["entries"]
    identity = entries[0]["source"]["evidence_identity"] if entries else None
    return {
        "schema_version": POOL_ARTIFACT_SCHEMA_VERSION,
        "producer_version": POOL_PRODUCER_VERSION,
        "pool_id": snapshot["pool_id"],
        "strategy_type": snapshot["strategy_type"],
        "revision": snapshot["revision"],
        "revision_id": snapshot["revision_id"],
        "parent_revision_id": snapshot["parent_revision_id"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "operation_kind": snapshot["operation"]["kind"],
        "source_artifact_ids": [entry["source"]["artifact_id"] for entry in entries],
        "evidence_identity": identity,
    }


def _pool_filename(snapshot: Mapping[str, Any]) -> str:
    return (
        f"{snapshot['pool_id']}_r{snapshot['revision']}_"
        f"{snapshot['snapshot_hash'][:12]}.json"
    )


def _artifact_output(record: Mapping[str, Any], *, task_id: str) -> dict[str, Any]:
    path = Path(str(record["path"]))
    return {
        "artifact_id": str(record["id"]),
        "kind": str(record["kind"]),
        "filename": path.name,
        "content_hash": str(record["content_hash"]),
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
        ),
    }


def _require_output_directory(path: Path, *, root: Path) -> None:
    if not path.is_absolute() or not root.is_absolute():
        raise StrategyError("strategy pool directory must use absolute task storage")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StrategyError("strategy pool directory escapes task storage") from exc
    current = path
    while True:
        if current.is_symlink():
            raise StrategyError("strategy pool directory must not use symlinks")
        if current.exists() and not current.is_dir():
            raise StrategyError("strategy pool directory must be a directory")
        if current == root:
            return
        if current == current.parent:
            raise StrategyError("strategy pool directory escapes task storage")
        current = current.parent


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{name} must be an object")
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(f"{name} fields are invalid ({'; '.join(details)})")


def _json_object(value: object, name: str) -> dict[str, Any]:
    try:
        result = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} must be a finite JSON object") from exc
    if not isinstance(result, dict):
        raise StrategyError(f"{name} must be an object")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _required_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrategyError(f"{name} must be non-empty canonical text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _required_text(value, name)


def _required_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _text_list(value: object, name: str) -> list[str]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise StrategyError(f"{name} must be a list")
    return [_required_text(item, f"{name} item") for item in value]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_binding_connection(conn, *, db_path: Path, name: str) -> None:
    if not conn.in_transaction:
        raise StrategyError(f"{name} binding requires a caller-owned transaction")
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != db_path
    ):
        raise StrategyError(f"{name} binding database changed")


def _pool_artifact_source_binding(
    binding: StrategyCandidatePoolArtifactBinding,
) -> _PoolArtifactSourceBinding:
    return _PoolArtifactSourceBinding(
        artifact_id=_required_hash(binding.artifact_id, "pool binding.artifact_id"),
        task_id=_required_text(binding.task_id, "pool binding.task_id"),
        kind=POOL_ARTIFACT_KIND,
        path=binding.artifact_path,
        content_hash=_required_hash(
            binding.artifact_content_hash,
            "pool binding.artifact_content_hash",
        ),
        origin_tool=_required_text(
            binding.artifact_origin_tool,
            "pool binding.artifact_origin_tool",
        ),
        provenance=_json_object(
            binding.artifact_provenance,
            "pool binding.artifact_provenance",
        ),
        provenance_json=_required_text(
            binding.artifact_provenance_json,
            "pool binding.artifact_provenance_json",
        ),
    )


def _require_pool_revision_artifact_link_on_connection(
    conn,
    *,
    pool: Mapping[str, Any],
    artifact_id: str,
    artifact_content_hash: str,
) -> None:
    row = conn.execute(
        """
        SELECT revision.artifact_id, revision.artifact_content_hash
          FROM strategy_candidate_pool_revisions AS revision
          JOIN strategy_candidate_pools AS head
            ON head.id = revision.pool_id
         WHERE head.task_id = ?
           AND head.strategy_type = ?
           AND revision.id = ?
           AND revision.revision = ?
        """,
        (
            pool["task_id"],
            pool["strategy_type"],
            pool["revision_id"],
            pool["revision"],
        ),
    ).fetchone()
    if (
        row is None
        or str(row["artifact_id"]) != artifact_id
        or not hmac.compare_digest(
            str(row["artifact_content_hash"]),
            artifact_content_hash,
        )
    ):
        raise StrategyError("strategy candidate pool revision artifact link changed")


def _require_lineage_dataset_paths(
    lineage: _CandidateLineage,
    *,
    datasets_root: Path,
) -> None:
    stack: list[_CandidateLineage] = [lineage]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        identity = id(current)
        if identity in seen:
            raise StrategyError("strategy candidate pool lineage graph changed")
        seen.add(identity)
        if len(seen) > _MAX_VOTING_ANCESTRY_NODES:
            raise StrategyError("strategy candidate pool lineage graph is too large")
        if isinstance(current, _VotingCandidateLineage):
            stack.extend(current.parent_lineages)
            continue
        dataset = current.dataset
        path = getattr(dataset, "path", None)
        source_path = getattr(dataset, "source_path", None)
        if not isinstance(path, Path) or not isinstance(source_path, str):
            raise StrategyError("strategy candidate pool dataset binding changed")
        _require_regular_dataset_path(path, root=datasets_root)
        try:
            expected_path = (datasets_root / source_path).resolve(strict=True)
        except OSError as exc:
            raise StrategyError(
                "strategy candidate pool dataset binding changed"
            ) from exc
        if path != expected_path:
            raise StrategyError("strategy candidate pool dataset path changed")


def _require_regular_dataset_path(path: Path, *, root: Path) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise StrategyError("strategy candidate pool dataset path changed")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "strategy candidate pool dataset escaped dataset storage"
        ) from exc


def _read_verified_pool_file(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
    error_message: str,
) -> bytes:
    """Read a bounded Pool snapshot without following or racing a path swap."""

    _require_regular_artifact_path(path, root=root)
    descriptor = -1
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    total = 0
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StrategyError(error_message)
        if before.st_size < 0 or before.st_size > _MAX_POOL_ARTIFACT_BYTES:
            raise StrategyError("strategy candidate pool artifact exceeds byte budget")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_POOL_ARTIFACT_BYTES:
                raise StrategyError(
                    "strategy candidate pool artifact exceeds byte budget"
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
            raise StrategyError(error_message)
        live_path = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(live_path.st_mode)
            or (live_path.st_dev, live_path.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise StrategyError(error_message)
    except OSError as exc:
        raise StrategyError(error_message) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or not hmac.compare_digest(digest.hexdigest(), expected_content_hash)
    ):
        raise StrategyError(error_message)
    return raw


__all__ = [
    "POOL_ARTIFACT_KIND",
    "POOL_ARTIFACT_SCHEMA_VERSION",
    "SCORECARD_REPORT_PROJECTION_SCHEMA_VERSION",
    "StrategyCandidatePoolArtifactBinding",
    "StrategyPoolDevelopmentDatasetBinding",
    "StrategyPoolDevelopmentExecutionBinding",
    "VerifiedUnivariateCandidateLineageBinding",
    "bind_strategy_pool_development_execution",
    "load_current_strategy_candidate_pool_artifact",
    "load_verified_univariate_candidate_lineage",
    "project_scorecard_report_evidence",
    "require_strategy_candidate_pool_artifact_binding_on_connection",
    "require_strategy_pool_development_execution_binding_on_connection",
    "require_verified_univariate_candidate_lineage_on_connection",
    "run_add_candidate_to_pool",
    "run_compile_strategy_pool",
    "run_remove_pool_entry",
    "run_reorder_strategy_pool",
    "run_set_pool_entry_action",
]
