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
from pathlib import Path
import re
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
    univariate_asset_to_verified_fragment,
    verified_fragment_pool_parts,
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
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_TYPE,
    verify_voting_candidate_asset_against_pool,
)
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    VOTING_CANDIDATE_ORIGIN_TOOL,
    voting_candidate_to_verified_fragment,
)
from marvis.packs.strategy.voting_candidate_tools import (
    VerifiedVotingCandidateArtifact,
    load_verified_voting_candidate_artifact,
    load_verified_voting_candidate_artifact_on_connection,
    require_voting_snapshot_marginal_reachability,
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
    compile_strategy_pool,
    remove_pool_entry,
    reorder_strategy_pool,
    set_pool_entry_action,
    validate_strategy_pool,
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
    | _VotingCandidateLineage
)


@dataclass
class _LineageCache:
    trees: dict[tuple[str, str], VerifiedAutomaticTreeSource]
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
    """Add one allowlisted asset with governed Voting placement semantics."""

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
    if triple == voting_triple:
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
    if isinstance(lineage, _VotingCandidateLineage):
        _require_voting_lineage_on_connection(
            conn,
            lineage,
            tasks_root=tasks_root,
            cache=cache if cache is not None else _LineageCache.empty(),
        )
        return
    raise StrategyError("unsupported candidate lineage type")


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
    _require_regular_artifact_path(binding.path, root=tasks_root)
    _require_file_content_hash(
        binding.path,
        binding.content_hash,
        "parent strategy pool artifact content hash drifted",
    )
    try:
        raw = binding.path.read_text("utf-8")
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
    _require_regular_artifact_path(path, root=Path(runtime.settings.tasks_dir))
    _require_file_content_hash(
        path,
        expected_hash,
        "current strategy pool artifact content hash drifted",
    )
    try:
        raw = path.read_text("utf-8")
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


__all__ = [
    "POOL_ARTIFACT_KIND",
    "POOL_ARTIFACT_SCHEMA_VERSION",
    "run_add_candidate_to_pool",
    "run_compile_strategy_pool",
    "run_remove_pool_entry",
    "run_reorder_strategy_pool",
    "run_set_pool_entry_action",
]
