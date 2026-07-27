"""Fail-closed, UI-safe projection of Strategy Candidate Lab evidence.

This module is deliberately read-only.  It does not calculate strategy
performance, mutate candidate state, or infer recommendations.  Every value in
the projection is copied from an independently revalidated immutable artifact
or from the current task/plan state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import ModelingRepository, TaskRepository
from marvis.domain import STRATEGY_TYPES
from marvis.output.strategy_candidate_report import (
    canonical_strategy_candidate_report_json,
    strategy_candidate_report_from_json,
)
from marvis.packs.modeling.evidence import (
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
)
from marvis.packs.modeling.evidence_tools import (
    TRAINING_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL,
)
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.score_evidence import (
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
    MODEL_SCORE_VECTOR_ARTIFACT_KIND,
)
from marvis.packs.modeling.score_evidence_tools import (
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
)
from marvis.packs.strategy.automatic_tree_asset import (
    canonical_automatic_tree_asset_json,
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
    AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION,
    AUTOMATIC_TREE_ASSET_ORIGIN_TOOL,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    automatic_tree_leaf_fragment_to_verified_candidate_fragment,
    canonical_automatic_tree_leaf_fragment_json,
    validate_automatic_tree_leaf_fragment,
)
from marvis.packs.strategy.automatic_tree_leaf_tools import (
    automatic_tree_source_provenance_from_asset,
    canonical_automatic_tree_leaf_selection_path,
    canonical_automatic_tree_source_path,
    verify_automatic_tree_leaf_selection_provenance,
    verify_automatic_tree_source_provenance,
)
from marvis.packs.strategy.candidate_asset import (
    canonical_candidate_asset_json,
    validate_candidate_asset,
)
from marvis.packs.strategy.candidate_asset_tools import (
    ASSET_ARTIFACT_KIND,
    ASSET_ARTIFACT_SCHEMA_VERSION,
    ORIGIN_TOOL as CANDIDATE_ASSET_ORIGIN_TOOL,
)
from marvis.packs.strategy.candidate_fragment import (
    univariate_asset_to_verified_fragment,
    verified_fragment_pool_parts,
)
from marvis.packs.strategy.interactive_tree_revision import (
    interactive_tree_topology_evidence,
)
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
    interactive_tree_frontier_selection_to_verified_candidate_fragment,
)
from marvis.packs.strategy.interactive_tree_frontier_tools import (
    load_verified_interactive_tree_frontier_selection_artifact,
)
from marvis.packs.strategy.interactive_tree_frontier_group_selection import (
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL,
    interactive_tree_frontier_group_selection_to_verified_candidate_fragment,
)
from marvis.packs.strategy.interactive_tree_frontier_group_tools import (
    load_verified_interactive_tree_frontier_group_selection_artifact,
)
from marvis.packs.strategy.interactive_tree_tools import (
    INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
    INTERACTIVE_TREE_REVISION_ORIGIN_TOOL,
    VerifiedInteractiveTreeRevision,
    load_verified_interactive_tree_revisions,
)
from marvis.packs.strategy.impact_cube_binding import (
    load_strategy_impact_cube_artifact,
)
from marvis.packs.strategy.impact_cube_tools import (
    IMPACT_CUBE_ARTIFACT_KIND,
    IMPACT_CUBE_ORIGIN_TOOL,
)
from marvis.packs.strategy.cross_matrix_cell_selection import (
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
    CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
    canonical_cross_matrix_cell_selection_json,
    cross_matrix_cell_selection_to_verified_candidate_fragment,
    validate_cross_matrix_cell_selection,
)
from marvis.packs.strategy.cross_matrix_cell_selection_tools import (
    canonical_cross_matrix_cell_selection_path,
    canonical_cross_matrix_source_path,
    verify_cross_matrix_cell_selection_provenance,
    verify_cross_matrix_source_provenance,
)
from marvis.packs.strategy.cross_matrix_candidate import (
    CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION,
    canonical_cross_matrix_candidate_asset_json,
    parse_cross_matrix_candidate_asset_json,
)
from marvis.packs.strategy.cross_candidate_search_tools import (
    CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
    CROSS_CANDIDATE_SEARCH_ORIGIN_TOOL,
    load_cross_candidate_search_artifact,
)
from marvis.packs.strategy.cross_rule_candidate import (
    cross_rule_candidate_to_verified_fragment,
)
from marvis.packs.strategy.cross_rule_search_tools import (
    CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
    CROSS_RULE_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    CROSS_RULE_CANDIDATE_ORIGIN_TOOL,
    CROSS_RULE_SEARCH_ARTIFACT_KIND,
    CROSS_RULE_SEARCH_ORIGIN_TOOL,
    load_cross_rule_candidate_artifact,
    load_cross_rule_search_artifact,
    replay_cross_rule_candidate_binding,
)
from marvis.packs.strategy.pool import (
    POOL_PRODUCER_VERSION,
    validate_strategy_pool,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    normalize_pool_requirements,
)
from marvis.packs.strategy.pool_impact_tools import (
    POOL_IMPACT_ARTIFACT_KIND,
    POOL_IMPACT_ORIGIN_TOOL,
    load_historical_strategy_pool_impact_artifact,
)
from marvis.packs.strategy.pool_stability_tools import (
    POOL_STABILITY_ARTIFACT_KIND,
    POOL_STABILITY_ORIGIN_TOOL,
    authenticate_strategy_pool_stability_artifact_record,
)
from marvis.packs.strategy.pool_validation_tools import (
    POOL_VALIDATION_ARTIFACT_KIND,
    POOL_VALIDATION_ORIGIN_TOOL,
    authenticate_strategy_pool_validation_artifact_record,
)
from marvis.packs.strategy.project_context_tools import (
    load_current_strategy_project_context_artifact,
)
from marvis.packs.strategy.scorecard_candidate import (
    MAX_SCORECARD_BANDS,
    MAX_SCORECARD_TABLE_ROWS,
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    canonical_scorecard_band_asset_json,
    canonical_scorecard_cutoff_selection_json,
    scorecard_cutoff_selection_to_verified_candidate_fragment,
    validate_scorecard_band_asset,
    validate_scorecard_cutoff_selection,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION,
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_ORIGIN_TOOL,
    load_any_strategy_sample_design_v2_artifacts,
    resolve_strategy_sample_design_v2_source_mode,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
)
from marvis.packs.strategy.scorecard_candidate_tools import (
    load_scorecard_band_asset_artifact,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ORIGIN_TOOL,
    voting_candidate_to_verified_fragment,
)
from marvis.packs.strategy.voting_candidate import (
    verify_voting_candidate_asset_against_pool,
)
from marvis.packs.strategy.voting_candidate_tools import (
    canonical_voting_candidate_artifact_json,
    canonical_voting_candidate_path,
    validate_voting_candidate_artifact_document,
    voting_candidate_artifact_provenance,
)
from marvis.packs.strategy.voting_candidate_search_tools import (
    VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
    VOTING_CANDIDATE_SEARCH_ORIGIN_TOOL,
    load_historical_voting_candidate_search_artifact,
)
from marvis.repositories.plans import PlanRepository
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_pool import (
    POOL_ARTIFACT_KIND,
    StrategyCandidatePoolRepository,
    canonical_strategy_pool_snapshot_json,
    strategy_pool_artifact_content_hash,
)
from marvis.repositories.strategy_reports import (
    STRATEGY_REPORT_ORIGIN_TOOL,
    STRATEGY_REPORT_OUTPUT_KINDS,
    StrategyReportRepository,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.strategy_lifecycle import is_locally_adopted


SCHEMA_VERSION = "strategy.candidate-lab-projection.v8"

UNIVARIATE_ARTIFACT_KIND = "strategy_candidate_json"
UNIVARIATE_ORIGIN_TOOL = "strategy.analyze_univariate_candidates"
CROSS_MATRIX_ARTIFACT_KIND = "strategy_cross_matrix_candidate_json"
CROSS_MATRIX_ORIGIN_TOOL = "strategy.build_cross_matrix_candidate"

_UNIVARIATE_VERSION_CONTRACTS = {
    "univariate-analysis-result.v1": (
        "strategy.univariate-candidate-artifact.v1",
        "strategy.univariate-candidate/1",
    ),
    "univariate-analysis-result.v2": (
        "strategy.univariate-candidate-artifact.v2",
        "strategy.univariate-candidate/2",
    ),
}
_CROSS_MATRIX_ARTIFACT_SCHEMAS = {
    "strategy.cross-matrix-candidate-asset.v1": (
        "strategy.cross-matrix-candidate-artifact.v1"
    ),
    CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION: (
        "strategy.cross-matrix-candidate-artifact.v2"
    ),
}
_POOL_ARTIFACT_SCHEMA_VERSION = "strategy.candidate-pool-artifact.v2"
_POOL_ORIGIN_BY_OPERATION = {
    "add_candidate": "strategy.add_candidate_to_pool",
    "insert_candidate_before_entries": "strategy.add_candidate_to_pool",
    "replace_entries_with_candidate": "strategy.add_candidate_to_pool",
    "remove_entry": "strategy.remove_pool_entry",
    "set_entry_action": "strategy.set_pool_entry_action",
    "reorder_entries": "strategy.reorder_strategy_pool",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAMPLE_DESIGN_V2_BUNDLE_ID_RE = re.compile(
    r"^strategy-sample-design-bundle-[0-9a-f]{24}$"
)
_MODEL_SCORE_VIRTUAL_FIELD_RE = re.compile(
    r"^__marvis_model_pd_[0-9a-f]{16}$"
)
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_PROJECTION_BYTES = 64 * 1024 * 1024
_MAX_CANDIDATES_PER_KIND = 20
_MAX_INTERACTIVE_TREE_REVISIONS = 20
_MAX_SCORECARD_CANDIDATES_PER_KIND = 3
_MAX_VOTING_SEARCHES = 20
_MAX_VOTING_SEARCH_COMBINATIONS = 20
_MAX_CROSS_SEARCHES = 20
_MAX_CROSS_SEARCH_PAIRS = 20
_MAX_CROSS_RULE_SEARCHES = 20
_MAX_CROSS_RULES = 50
_MAX_CROSS_RULE_CANDIDATES = 20
_MAX_POOL_ADD_SOURCES_PER_KIND = 20
_MAX_RANKINGS = 50
_MAX_METRICS = 100
_MAX_BIN_POINTERS = 200
_MAX_CELL_POINTERS = 400
_MAX_LEAF_POINTERS = 256
_MAX_SCORECARD_BAND_POINTERS = MAX_SCORECARD_BANDS
_MAX_SCORECARD_CUTOFF_POINTERS = MAX_SCORECARD_BANDS - 1
_MAX_SCORECARD_POINT_POINTERS = min(MAX_SCORECARD_TABLE_ROWS, 2_000)
_MAX_POOL_ENTRIES = 200
_MAX_RISKS = 50
_MAX_WORKFLOW_EVIDENCE_PER_KIND = 40
_MAX_REPORT_REVISIONS = 20
_MAX_STRATEGIES = 100
_MAX_STRATEGY_ARTIFACTS = 50
_STRATEGY_ARTIFACT_SUFFIXES = frozenset(
    {
        ".csv",
        ".docx",
        ".json",
        ".md",
        ".pdf",
        ".png",
        ".py",
        ".sql",
        ".svg",
        ".xlsx",
    }
)
_UNIVARIATE_PARAMETER_FIELDS = (
    "analysis_schema_version",
    "features",
    "feature_types",
    "methods",
    "method_mode",
    "bin_count",
    "min_bin_pct",
    "loan_amount_col",
    "overdue_amount_col",
    "sentinel_values",
    "manual_breakpoints",
    "estimated_evaluated_cells",
    "budget_unit",
    "drop_nan_labels",
    "nan_labels_dropped",
)
_SCORECARD_POINT_FIELDS = (
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
    "base_score",
    "pdo",
    "base_odds",
    "factor",
    "offset",
)


class CandidateLabProjectionError(RuntimeError):
    """Candidate Lab evidence could not be proven safe for projection."""


@dataclass
class _ProjectionBudget:
    limit: int
    used: int = 0

    def reserve(self, byte_count: int) -> None:
        if byte_count < 0 or self.used + byte_count > self.limit:
            raise CandidateLabProjectionError(
                "strategy candidate lab aggregate byte budget exceeded"
            )
        self.used += byte_count


@dataclass
class _ProjectionContext:
    settings: Any
    task_id: str
    artifact_repository: TaskArtifactRepository
    budget: _ProjectionBudget
    raw_cache: dict[tuple[str, str], bytes] = field(default_factory=dict)
    verified_cache: dict[tuple[str, str], Any] = field(default_factory=dict)
    scorecard_source_cache: set[str] = field(default_factory=set)
    scorecard_runtime: Any | None = None
    verified_pool_entry_replays: set[tuple[str, str]] = field(
        default_factory=set
    )
    pool_entry_replays_in_progress: set[tuple[str, str]] = field(
        default_factory=set
    )


def build_strategy_candidate_lab_projection(settings, task_id: str) -> dict[str, Any]:
    """Build a bounded projection from task-owned, revalidated evidence."""

    try:
        return _build_projection(settings, task_id)
    except CandidateLabProjectionError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise CandidateLabProjectionError(
            "strategy candidate lab evidence verification failed"
        ) from exc


def _build_projection(settings, task_id: str) -> dict[str, Any]:
    artifact_repository = TaskArtifactRepository(settings.db_path)
    context = _ProjectionContext(
        settings=settings,
        task_id=task_id,
        artifact_repository=artifact_repository,
        budget=_ProjectionBudget(_MAX_PROJECTION_BYTES),
    )

    univariate_records, univariate_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            UNIVARIATE_ARTIFACT_KIND,
            limit=_MAX_CANDIDATES_PER_KIND,
        )
    )
    univariate_records = _candidate_record_window(
        settings,
        task_id,
        univariate_records,
        kind=UNIVARIATE_ARTIFACT_KIND,
        origin_tool=UNIVARIATE_ORIGIN_TOOL,
        directory_name="strategy_candidates",
        filename_pattern=re.compile(
            r"^candidate-[0-9a-f]{32}_[0-9a-f]{12}\.json$"
        ),
    )
    cross_matrix_records, cross_matrix_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            CROSS_MATRIX_ARTIFACT_KIND,
            limit=_MAX_CANDIDATES_PER_KIND,
        )
    )
    cross_matrix_records = _candidate_record_window(
        settings,
        task_id,
        cross_matrix_records,
        kind=CROSS_MATRIX_ARTIFACT_KIND,
        origin_tool=CROSS_MATRIX_ORIGIN_TOOL,
        directory_name="strategy_cross_matrix_candidates",
        filename_pattern=re.compile(
            r"^candidate-asset-[0-9a-f]{32}_[0-9a-f]{12}\.json$"
        ),
    )
    cross_search_records, cross_search_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
            limit=_MAX_CROSS_SEARCHES,
        )
    )
    cross_search_records = _candidate_record_window(
        settings,
        task_id,
        cross_search_records,
        kind=CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
        origin_tool=CROSS_CANDIDATE_SEARCH_ORIGIN_TOOL,
        directory_name="strategy_cross_candidate_searches",
        filename_pattern=re.compile(
            r"^cross-search-[0-9a-f]{32}_[0-9a-f]{12}\.json$"
        ),
    )
    cross_rule_search_records, cross_rule_search_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            CROSS_RULE_SEARCH_ARTIFACT_KIND,
            limit=_MAX_CROSS_RULE_SEARCHES,
        )
    )
    cross_rule_search_records = _candidate_record_window(
        settings,
        task_id,
        cross_rule_search_records,
        kind=CROSS_RULE_SEARCH_ARTIFACT_KIND,
        origin_tool=CROSS_RULE_SEARCH_ORIGIN_TOOL,
        directory_name="strategy_cross_rule_searches",
        filename_pattern=re.compile(
            r"^cross-rule-search-[0-9a-f]{32}_[0-9a-f]{12}\.json$"
        ),
    )
    cross_rule_candidate_records, cross_rule_candidate_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
            limit=_MAX_CROSS_RULE_CANDIDATES,
        )
    )
    cross_rule_candidate_records = _candidate_record_window(
        settings,
        task_id,
        cross_rule_candidate_records,
        kind=CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
        origin_tool=CROSS_RULE_CANDIDATE_ORIGIN_TOOL,
        directory_name="strategy_cross_rule_candidates",
        filename_pattern=re.compile(
            r"^cross-rule-asset-[0-9a-f]{32}_[0-9a-f]{12}\.json$"
        ),
    )
    automatic_tree_records, automatic_tree_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
            limit=_MAX_CANDIDATES_PER_KIND,
        )
    )
    automatic_tree_records = _candidate_record_window(
        settings,
        task_id,
        automatic_tree_records,
        kind=AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
        origin_tool=AUTOMATIC_TREE_ASSET_ORIGIN_TOOL,
        directory_name="strategy_automatic_trees",
        filename_pattern=re.compile(r"^candidate-asset-[0-9a-f]{32}\.json$"),
    )
    interactive_tree_records, interactive_tree_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
            limit=_MAX_INTERACTIVE_TREE_REVISIONS,
        )
    )
    interactive_tree_records = _candidate_record_window(
        settings,
        task_id,
        interactive_tree_records,
        kind=INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
        origin_tool=INTERACTIVE_TREE_REVISION_ORIGIN_TOOL,
        directory_name="strategy_interactive_tree_revisions",
        filename_pattern=re.compile(
            r"^interactive-tree-revision-[0-9a-f]{32}\.json$"
        ),
    )
    scorecard_band_records, scorecard_band_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            SCORECARD_BAND_ASSET_ARTIFACT_KIND,
            limit=_MAX_SCORECARD_CANDIDATES_PER_KIND,
        )
    )
    scorecard_band_records = _candidate_record_window(
        settings,
        task_id,
        scorecard_band_records,
        kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        origin_tool=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        directory_name="strategy_scorecard_candidates",
        filename_pattern=re.compile(
            r"^scorecard-band-asset-[0-9a-f]{32}\.json$"
        ),
    )
    scorecard_selection_records, scorecard_selection_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
            limit=_MAX_SCORECARD_CANDIDATES_PER_KIND,
        )
    )
    scorecard_selection_records = _candidate_record_window(
        settings,
        task_id,
        scorecard_selection_records,
        kind=SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        origin_tool=SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        directory_name="strategy_scorecard_candidates",
        filename_pattern=re.compile(
            r"^scorecard-cutoff-selection-[0-9a-f]{32}\.json$"
        ),
    )
    voting_search_records, voting_search_total = (
        artifact_repository.list_recent_for_task_kind_with_count(
            task_id,
            VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
            limit=_MAX_VOTING_SEARCHES,
        )
    )
    voting_search_records = _candidate_record_window(
        settings,
        task_id,
        voting_search_records,
        kind=VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
        origin_tool=VOTING_CANDIDATE_SEARCH_ORIGIN_TOOL,
        directory_name="strategy_voting_candidate_searches",
        filename_pattern=re.compile(
            r"^voting-search-[0-9a-f]{32}-[0-9a-f]{16}\.json$"
        ),
    )
    univariate = [
        _project_univariate(context, record)
        for record in univariate_records
    ]
    cross_matrix = [
        _project_cross_matrix(
            context,
            record,
        )
        for record in cross_matrix_records
    ]
    cross_search = [
        _project_cross_search(context, record)
        for record in cross_search_records
    ]
    cross_rule_search = [
        _project_cross_rule_search(context, record)
        for record in cross_rule_search_records
    ]
    cross_rule_candidate = [
        _project_cross_rule_candidate(context, record)
        for record in cross_rule_candidate_records
    ]
    automatic_tree = [
        _project_automatic_tree(context, record)
        for record in automatic_tree_records
    ]
    interactive_tree_revision = _project_interactive_tree_revisions(
        context,
        interactive_tree_records,
    )
    scorecard_band = [
        _project_scorecard_band(context, record)
        for record in scorecard_band_records
    ]
    scorecard_cutoff_selection = [
        _project_scorecard_cutoff_selection(context, record)
        for record in scorecard_selection_records
    ]
    voting_search = [
        _project_voting_search(context, record)
        for record in voting_search_records
    ]
    pool_add_sources = _project_pool_add_sources(context)
    pools = _project_current_pools(
        context,
    )
    strategies = _project_strategy_history(context)
    project_context = _project_current_project_context(context)
    sample_design = _project_active_sample_design(context)
    latest_evidence = _project_latest_workflow_evidence(
        context,
        pools=pools,
    )
    report = _project_latest_strategy_report(context)
    workflow = _workflow_projection(
        project_context=project_context,
        sample_design=sample_design,
        candidates={
            "univariate": univariate,
            "cross_matrix": cross_matrix,
            "cross_search": cross_search,
            "cross_rule_search": cross_rule_search,
            "cross_rule_candidate": cross_rule_candidate,
            "automatic_tree": automatic_tree,
            "interactive_tree_revision": interactive_tree_revision,
            "scorecard_band": scorecard_band,
            "scorecard_cutoff_selection": scorecard_cutoff_selection,
            "voting_search": voting_search,
        },
        pools=pools,
        latest_evidence=latest_evidence,
        report=report,
    )
    workflow["strategy_history_status"] = (
        "complete" if strategies["total"] > 0 else "missing"
    )
    active_plan = _active_plan_projection(settings, task_id)
    open_gate = _open_gate_projection(settings, task_id)
    if active_plan is not None:
        blocked_reason = "active_plan"
    elif open_gate is not None:
        blocked_reason = "open_gate"
    else:
        blocked_reason = None

    projection = {
        "schema_version": SCHEMA_VERSION,
        "task_id": context.task_id,
        "can_start": blocked_reason is None,
        "blocked_reason": blocked_reason,
        "active_plan": active_plan,
        "open_gate": open_gate,
        "candidates": {
            "univariate": _collection(
                univariate,
                _MAX_CANDIDATES_PER_KIND,
                total=univariate_total,
            ),
            "cross_matrix": _collection(
                cross_matrix,
                _MAX_CANDIDATES_PER_KIND,
                total=cross_matrix_total,
            ),
            "cross_search": _collection(
                cross_search,
                _MAX_CROSS_SEARCHES,
                total=cross_search_total,
            ),
            "cross_rule_search": _collection(
                cross_rule_search,
                _MAX_CROSS_RULE_SEARCHES,
                total=cross_rule_search_total,
            ),
            "cross_rule_candidate": _collection(
                cross_rule_candidate,
                _MAX_CROSS_RULE_CANDIDATES,
                total=cross_rule_candidate_total,
            ),
            "automatic_tree": _collection(
                automatic_tree,
                _MAX_CANDIDATES_PER_KIND,
                total=automatic_tree_total,
            ),
            "interactive_tree_revision": _collection(
                interactive_tree_revision,
                _MAX_INTERACTIVE_TREE_REVISIONS,
                total=interactive_tree_total,
            ),
            "scorecard_band": _collection(
                scorecard_band,
                _MAX_SCORECARD_CANDIDATES_PER_KIND,
                total=scorecard_band_total,
            ),
            "scorecard_cutoff_selection": _collection(
                scorecard_cutoff_selection,
                _MAX_SCORECARD_CANDIDATES_PER_KIND,
                total=scorecard_selection_total,
            ),
            "voting_search": _collection(
                voting_search,
                _MAX_VOTING_SEARCHES,
                total=voting_search_total,
            ),
        },
        "pool_add_sources": pool_add_sources,
        "pools": _collection(pools, len(STRATEGY_TYPES)),
        "strategies": strategies,
        "workflow": workflow,
    }
    _require_projection_payload_budget(projection)
    return projection


def _project_active_sample_design(
    context: _ProjectionContext,
) -> dict[str, Any] | None:
    """Project the newest real, current, fully authenticated V2 sample pair."""

    records, _total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
            limit=_MAX_CANDIDATES_PER_KIND,
        )
    )
    candidates = []
    for record in records:
        _require_record_identity(record, task_id=context.task_id)
        if record["origin_tool"] not in {
            SAMPLE_DESIGN_V2_ORIGIN_TOOL,
            SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        }:
            raise CandidateLabProjectionError(
                "sample-design V2 bundle origin drifted"
            )
        provenance = _mapping(
            record["provenance"],
            "sample-design V2 bundle provenance",
        )
        bundle_id = provenance.get("bundle_id")
        # Scorecard compatibility fixtures and pre-V2 placeholders may use the
        # same kind without being valid SampleDesign V2 bundles.  They are not
        # eligible to become the active workbench sample.
        if (
            not isinstance(bundle_id, str)
            or _SAMPLE_DESIGN_V2_BUNDLE_ID_RE.fullmatch(bundle_id) is None
        ):
            continue
        candidates.append(record)
    if not candidates:
        return None
    record = max(
        candidates,
        key=lambda item: (item["created_at"], item["id"]),
    )
    provenance = _mapping(
        record["provenance"],
        "active sample-design V2 provenance",
    )
    try:
        binding = load_any_strategy_sample_design_v2_artifacts(
            _scorecard_live_runtime(context),
            task_id=context.task_id,
            membership_artifact_id=_text(
                provenance.get("membership_artifact_id"),
                "active sample membership artifact_id",
            ),
            expected_membership_artifact_content_hash=_sha256(
                provenance.get("membership_artifact_content_hash"),
                "active sample membership artifact content_hash",
            ),
            bundle_artifact_id=_text(
                record.get("id"),
                "active sample bundle artifact_id",
            ),
            expected_bundle_artifact_content_hash=_sha256(
                record.get("content_hash"),
                "active sample bundle artifact content_hash",
            ),
            expected_bundle_id=_text(
                provenance.get("bundle_id"),
                "active sample bundle_id",
            ),
            expected_sample_design_id=_text(
                provenance.get("sample_design_id"),
                "active sample design_id",
            ),
            expected_sample_design_content_hash=_sha256(
                provenance.get("sample_design_content_hash"),
                "active sample design content_hash",
            ),
        )
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "active sample-design V2 failed authoritative replay"
        ) from exc
    bundle = binding.bundle
    design = _mapping(bundle.get("sample_design"), "active sample design")
    populations = {
        _text(item.get("role"), "sample population role"): item
        for item in _sequence(
            bundle.get("populations"),
            "active sample populations",
        )
    }
    if set(populations) != {"approval", "risk"}:
        raise CandidateLabProjectionError(
            "active sample must contain approval and risk populations"
        )
    header = _mapping(bundle.get("membership"), "active sample membership")
    counts = _mapping(header.get("counts"), "active sample membership counts")
    relationship_counts = _mapping(
        counts.get("relationship"),
        "active sample relationship counts",
    )
    within = _mapping(
        relationship_counts.get("risk_within_approval"),
        "active sample within relationship",
    )
    outside = _mapping(
        relationship_counts.get("risk_outside_approval"),
        "active sample outside relationship",
    )
    universe = int(counts.get("analysis_universe"))
    approval_total = int(_mapping(counts["approval"], "approval counts")["total"])
    risk_total = int(_mapping(counts["risk"], "risk counts")["total"])
    overlap = int(within["total"])
    diagnostics = list(
        _sequence(bundle.get("diagnostics"), "active sample diagnostics")
    )
    diagnostic_counts = {
        status: sum(1 for item in diagnostics if item.get("status") == status)
        for status in ("pass", "warn", "fail", "unavailable", "not_applicable")
    }
    overall_status = next(
        (
            status
            for status in ("fail", "warn", "unavailable", "pass")
            if diagnostic_counts[status]
        ),
        "unavailable",
    )
    target = _mapping(
        design.get("target_selector"),
        "active sample target selector",
    )
    return {
        "artifact": _artifact_projection(record, context.task_id),
        "sample_design_id": design["sample_design_id"],
        "bundle_id": bundle["bundle_id"],
        "source_mode": resolve_strategy_sample_design_v2_source_mode(
            design,
            capability="physical_v2",
            consumer="strategy_candidate_lab_projection",
        ),
        "freshness": "current",
        "relationship": design["relationship"],
        "analysis_universe_count": universe,
        "target": {
            "column": target["column"],
            "good_value": target["good_value"],
            "bad_value": target["bad_value"],
            "missing_policy": (
                "drop" if target["drop_missing"] else "keep"
            ),
        },
        "populations": {
            role: _sample_population_projection(populations[role])
            for role in ("approval", "risk")
        },
        "relationship_counts": {
            "approval_and_risk": overlap,
            "approval_only": approval_total - overlap,
            "risk_only": int(outside["total"]),
            "neither": universe - (approval_total + risk_total - overlap),
        },
        "diagnostics": {
            "overall_status": overall_status,
            "counts": diagnostic_counts,
        },
    }


def _sample_population_projection(
    population: Mapping[str, Any],
) -> dict[str, Any]:
    partitions = {
        _text(item.get("name"), "sample partition name"): int(
            item.get("row_count")
        )
        for item in _sequence(
            population.get("partitions"),
            "sample population partitions",
        )
    }
    if set(partitions) != {"development", "validation", "oot"}:
        raise CandidateLabProjectionError(
            "sample population partitions are incomplete"
        )
    maturity = _mapping(
        population.get("maturity_evidence"),
        "sample population maturity",
    )
    return {
        "total_count": int(population["total_count"]),
        "partitions": {
            name: partitions[name]
            for name in ("development", "validation", "oot")
        },
        "maturity": {
            key: maturity[key]
            for key in (
                "status",
                "performance_window_days",
                "cutoff_date",
                "eligible_count",
                "labeled_count",
                "reason",
            )
        },
    }


def _project_latest_workflow_evidence(
    context: _ProjectionContext,
    *,
    pools: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Project newest authenticated evidence without treating it as current."""

    return {
        "pool_stability": _project_latest_pool_stability(
            context,
            pools=pools,
        ),
        "pool_impact": _project_latest_pool_impact(
            context,
            pools=pools,
        ),
        "impact_cube": _project_latest_impact_cube(
            context,
            pools=pools,
        ),
        "pool_validation": _project_latest_pool_validations(
            context,
            pools=pools,
        ),
    }


def _latest_record_for_kind(
    context: _ProjectionContext,
    kind: str,
) -> Mapping[str, Any] | None:
    records, _total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            kind,
            limit=_MAX_WORKFLOW_EVIDENCE_PER_KIND,
        )
    )
    return None if not records else records[0]


def _project_latest_pool_stability(
    context: _ProjectionContext,
    *,
    pools: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    record = _latest_record_for_kind(context, POOL_STABILITY_ARTIFACT_KIND)
    if record is None:
        return None
    raw = _read_candidate_record(
        context,
        record,
        kind=POOL_STABILITY_ARTIFACT_KIND,
        origin_tool=POOL_STABILITY_ORIGIN_TOOL,
        directory_name="strategy_pool_stabilities",
    )
    stability = _json_object(raw, "strategy Pool stability")
    try:
        authenticated = authenticate_strategy_pool_stability_artifact_record(
            task_id=context.task_id,
            record=record,
            stability=stability,
            tasks_root=context.settings.tasks_dir,
        )
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "latest Pool stability failed authoritative replay"
        ) from exc
    normalized = _mapping(
        authenticated.get("stability"),
        "authenticated Pool stability",
    )
    identity = _mapping(
        normalized.get("identity"),
        "Pool stability identity",
    )
    return {
        "kind": "pool_stability",
        "artifact": _artifact_projection(record, context.task_id),
        "stability_id": normalized["stability_id"],
        "strategy_type": identity["strategy_type"],
        "pool_revision": identity["revision"],
        "baseline_partition": normalized["baseline_partition"],
        "comparison_partitions": list(normalized["comparison_partitions"]),
        "lifecycle": dict(
            _mapping(
                normalized.get("lifecycle"),
                "Pool stability lifecycle",
            )
        ),
        "freshness": _pool_identity_freshness(identity, pools),
    }


def _project_latest_pool_impact(
    context: _ProjectionContext,
    *,
    pools: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    record = _latest_record_for_kind(context, POOL_IMPACT_ARTIFACT_KIND)
    if record is None:
        return None
    _require_record_identity(record, task_id=context.task_id)
    if (
        record["kind"] != POOL_IMPACT_ARTIFACT_KIND
        or record["origin_tool"] != POOL_IMPACT_ORIGIN_TOOL
    ):
        raise CandidateLabProjectionError(
            "latest Pool impact registry identity drifted"
        )
    provenance = _mapping(
        record.get("provenance"),
        "latest Pool impact provenance",
    )
    try:
        binding = load_historical_strategy_pool_impact_artifact(
            _scorecard_live_runtime(context),
            task_id=context.task_id,
            artifact_id=_sha256(
                record.get("id"),
                "latest Pool impact artifact_id",
            ),
            expected_artifact_content_hash=_sha256(
                record.get("content_hash"),
                "latest Pool impact artifact content_hash",
            ),
            expected_assessment_id=_text(
                provenance.get("assessment_id"),
                "latest Pool impact assessment_id",
            ),
            expected_assessment_content_hash=_sha256(
                provenance.get("assessment_content_hash"),
                "latest Pool impact assessment content_hash",
            ),
        )
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "latest Pool impact failed authoritative replay"
        ) from exc
    assessment = _mapping(
        binding.assessment,
        "authenticated Pool impact",
    )
    identity = _mapping(
        assessment.get("identity"),
        "Pool impact identity",
    )
    population = _mapping(
        assessment.get("population"),
        "Pool impact population",
    )
    lifecycle = _mapping(
        assessment.get("lifecycle"),
        "Pool impact lifecycle",
    )
    monthly = _mapping(
        assessment.get("monthly"),
        "Pool impact monthly",
    )
    return {
        "kind": "pool_impact",
        "artifact": _artifact_projection(record, context.task_id),
        "assessment_id": assessment["assessment_id"],
        "strategy_type": identity["strategy_type"],
        "pool_revision": identity["revision"],
        "population_count": population["population_count"],
        "labeled_count": population["labelled_count"],
        "monthly_status": monthly["status"],
        "lifecycle": {
            key: lifecycle[key]
            for key in (
                "candidate_stage",
                "observation_stage",
                "validation_status",
            )
        },
        "freshness": _pool_identity_freshness(identity, pools),
    }


def _project_latest_impact_cube(
    context: _ProjectionContext,
    *,
    pools: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    record = _latest_record_for_kind(context, IMPACT_CUBE_ARTIFACT_KIND)
    if record is None:
        return None
    _require_record_identity(record, task_id=context.task_id)
    if (
        record["kind"] != IMPACT_CUBE_ARTIFACT_KIND
        or record["origin_tool"] != IMPACT_CUBE_ORIGIN_TOOL
    ):
        raise CandidateLabProjectionError(
            "latest ImpactCube registry identity drifted"
        )
    provenance = _mapping(
        record.get("provenance"),
        "latest ImpactCube provenance",
    )
    try:
        binding = load_strategy_impact_cube_artifact(
            _scorecard_live_runtime(context),
            task_id=context.task_id,
            artifact_id=_sha256(
                record.get("id"),
                "latest ImpactCube artifact_id",
            ),
            expected_artifact_content_hash=_sha256(
                record.get("content_hash"),
                "latest ImpactCube artifact content_hash",
            ),
            expected_cube_id=_text(
                provenance.get("cube_id"),
                "latest ImpactCube cube_id",
            ),
            expected_cube_content_hash=_sha256(
                provenance.get("cube_content_hash"),
                "latest ImpactCube content_hash",
            ),
        )
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "latest ImpactCube failed authoritative replay"
        ) from exc
    cube = _mapping(binding.cube, "authenticated ImpactCube")
    identity = _mapping(cube.get("identity"), "ImpactCube identity")
    partitions = []
    for row in _sequence(cube.get("partitions"), "ImpactCube partitions"):
        if row.get("role") != "risk":
            continue
        name = _text(row.get("name"), "ImpactCube partition name")
        if name not in partitions:
            partitions.append(name)
    return {
        "kind": "impact_cube",
        "artifact": _artifact_projection(record, context.task_id),
        "cube_id": cube["cube_id"],
        "strategy_type": identity["strategy_type"],
        "pool_revision": identity["revision"],
        "partitions": partitions,
        "slice_families": dict(
            _mapping(
                cube.get("slice_families"),
                "ImpactCube slice families",
            )
        ),
        "freshness": _pool_identity_freshness(identity, pools),
    }


def _project_latest_pool_validations(
    context: _ProjectionContext,
    *,
    pools: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    records, _total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            POOL_VALIDATION_ARTIFACT_KIND,
            limit=_MAX_WORKFLOW_EVIDENCE_PER_KIND,
        )
    )
    projected: dict[str, dict[str, Any] | None] = {
        "validation": None,
        "oot": None,
    }
    for record in records:
        _require_record_identity(record, task_id=context.task_id)
        if (
            record["kind"] != POOL_VALIDATION_ARTIFACT_KIND
            or record["origin_tool"] != POOL_VALIDATION_ORIGIN_TOOL
        ):
            raise CandidateLabProjectionError(
                "latest Pool validation registry identity drifted"
            )
        provenance = _mapping(
            record.get("provenance"),
            "Pool validation provenance",
        )
        partition = provenance.get("partition")
        if partition not in projected:
            raise CandidateLabProjectionError(
                "Pool validation partition is unsupported"
            )
        if projected[partition] is not None:
            continue
        raw = _read_candidate_record(
            context,
            record,
            kind=POOL_VALIDATION_ARTIFACT_KIND,
            origin_tool=POOL_VALIDATION_ORIGIN_TOOL,
            directory_name="strategy_pool_validations",
        )
        try:
            evidence = (
                authenticate_strategy_pool_validation_artifact_record(
                    task_id=context.task_id,
                    record=record,
                    evidence=_json_object(
                        raw,
                        "strategy Pool validation",
                    ),
                    tasks_root=context.settings.tasks_dir,
                )
            )
        except StrategyError as exc:
            raise CandidateLabProjectionError(
                "latest Pool validation failed authoritative replay"
            ) from exc
        identity = _mapping(
            evidence.get("identity"),
            "Pool validation identity",
        )
        population = _mapping(
            evidence.get("population_metrics"),
            "Pool validation population metrics",
        )
        projected[partition] = {
            "kind": "pool_validation",
            "artifact": _artifact_projection(record, context.task_id),
            "evidence_id": evidence["evidence_id"],
            "strategy_type": identity["strategy_type"],
            "pool_revision": identity["revision"],
            "partition": evidence["partition"],
            "population_count": population["population_count"],
            "labeled_count": population["labelled_count"],
            "lifecycle": dict(
                _mapping(
                    evidence.get("lifecycle"),
                    "Pool validation lifecycle",
                )
            ),
            "freshness": _pool_identity_freshness(identity, pools),
        }
        if all(projected.values()):
            break
    return projected


def _pool_identity_freshness(
    identity: Mapping[str, Any],
    pools: Sequence[Mapping[str, Any]],
) -> str:
    strategy_type = _text(
        identity.get("strategy_type"),
        "workflow evidence strategy_type",
    )
    expected = {
        "pool_id": _text(
            identity.get("pool_id"),
            "workflow evidence pool_id",
        ),
        "revision": int(identity.get("revision")),
    }
    current = next(
        (
            pool
            for pool in pools
            if pool.get("strategy_type") == strategy_type
        ),
        None,
    )
    if current is None:
        return "stale"
    return (
        "current"
        if all(current.get(key) == value for key, value in expected.items())
        else "stale"
    )


def _project_latest_strategy_report(
    context: _ProjectionContext,
) -> dict[str, Any] | None:
    json_kind = STRATEGY_REPORT_OUTPUT_KINDS["json"]
    records, _total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            json_kind,
            limit=_MAX_REPORT_REVISIONS,
        )
    )
    if not records:
        return None
    record = records[0]
    _require_record_identity(record, task_id=context.task_id)
    if (
        record["kind"] != json_kind
        or record["origin_tool"] != STRATEGY_REPORT_ORIGIN_TOOL
    ):
        raise CandidateLabProjectionError(
            "latest Strategy report registry identity drifted"
        )
    provenance = _mapping(
        record.get("provenance"),
        "latest Strategy report provenance",
    )
    report_id = _text(
        provenance.get("report_id"),
        "latest Strategy report_id",
    )
    report = StrategyReportRepository(context.settings.db_path).get_by_id(
        task_id=context.task_id,
        report_id=report_id,
    )
    if report is None:
        raise CandidateLabProjectionError(
            "latest Strategy report revision is missing"
        )
    bundle = _mapping(report.get("bundle"), "authenticated Strategy report")
    artifacts = _mapping(
        report.get("artifacts"),
        "authenticated Strategy report artifacts",
    )
    json_artifact = _mapping(
        artifacts.get("json"),
        "authenticated Strategy JSON report",
    )
    if (
        json_artifact.get("id") != record.get("id")
        or json_artifact.get("content_hash") != record.get("content_hash")
    ):
        raise CandidateLabProjectionError(
            "latest Strategy report JSON binding drifted"
        )
    title = _mapping(bundle.get("title"), "Strategy report title")
    return {
        "report_id": bundle["report_id"],
        "revision": bundle["report_revision"],
        "status": bundle["status"],
        "title": (
            title.get("value")
            if title.get("availability") == "present"
            else None
        ),
        "created_at": report["created_at"],
        "freshness": "current",
        "artifacts": {
            output_format: (
                None
                if output_format not in artifacts
                else _artifact_projection(
                    _mapping(
                        artifacts[output_format],
                        f"Strategy report {output_format} artifact",
                    ),
                    context.task_id,
                )
            )
            for output_format in ("json", "markdown", "xlsx", "docx")
        },
    }


def _project_strategy_history(
    context: _ProjectionContext,
) -> dict[str, Any]:
    repository = StrategyRepository(context.settings.db_path)
    refs, total = repository.list_recent_strategy_refs_for_task_with_count(
        context.task_id,
        limit=_MAX_STRATEGIES,
    )
    if total < len(refs):
        raise CandidateLabProjectionError(
            "Strategy history total is inconsistent"
        )

    projected = []
    for ref in refs:
        binding = _authenticate_strategy_snapshot(
            repository,
            task_id=context.task_id,
            ref=ref,
        )
        projected.append(
            _project_strategy_history_item(
                context,
                repository=repository,
                binding=binding,
            )
        )

    champion_refs, champion_total = (
        repository.list_current_local_champion_refs_for_task_with_count(
            context.task_id,
            limit=len(STRATEGY_TYPES) + 1,
        )
    )
    if champion_total != len(champion_refs) or champion_total > len(
        STRATEGY_TYPES
    ):
        raise CandidateLabProjectionError(
            "current local Strategy champions are inconsistent"
        )
    champions = []
    champion_types: set[str] = set()
    for ref in champion_refs:
        binding = _authenticate_strategy_snapshot(
            repository,
            task_id=context.task_id,
            ref=ref,
        )
        metadata = binding["metadata"]
        if (
            not is_locally_adopted(
                metadata["status"],
                metadata["asset_status"],
            )
            or metadata["strategy_type"] in champion_types
        ):
            raise CandidateLabProjectionError(
                "current local Strategy champion lifecycle drifted"
            )
        champion_types.add(metadata["strategy_type"])
        champions.append(
            {
                "strategy_id": metadata["id"],
                "strategy_type": metadata["strategy_type"],
                "version": metadata["version"],
            }
        )

    return {
        "latest": projected[0] if projected else None,
        "all": projected,
        "total": total,
        "truncated": total > len(projected),
        "current_local_champions": champions,
    }


def _authenticate_strategy_snapshot(
    repository: StrategyRepository,
    *,
    task_id: str,
    ref: Mapping[str, Any],
) -> dict[str, Any]:
    strategy_id = _text(ref.get("id"), "Strategy id")
    snapshot = repository.get_strategy_snapshot(strategy_id)
    metadata = repository.get_strategy_meta(strategy_id)
    spec_hash = repository.get_strategy_spec_hash(strategy_id)
    if snapshot is None or metadata is None or spec_hash is None:
        raise CandidateLabProjectionError("Strategy snapshot disappeared")
    snapshot_metadata = _mapping(
        snapshot.get("metadata"),
        "Strategy snapshot metadata",
    )
    snapshot_hash = _sha256(
        snapshot.get("strategy_spec_hash"),
        "Strategy snapshot spec hash",
    )
    normalized_hash = _sha256(spec_hash, "Strategy spec hash")
    if (
        dict(ref) != metadata
        or dict(snapshot_metadata) != metadata
        or not hmac.compare_digest(snapshot_hash, normalized_hash)
        or metadata["task_id"] != task_id
        or metadata["strategy_type"] not in STRATEGY_TYPES
    ):
        raise CandidateLabProjectionError(
            "Strategy snapshot, metadata, or spec hash drifted"
        )
    strategy = snapshot.get("strategy")
    if (
        strategy is None
        or strategy.id != strategy_id
        or strategy.strategy_type != metadata["strategy_type"]
        or strategy.spec is None
        or len(strategy.rules) != len(strategy.spec.rules)
    ):
        raise CandidateLabProjectionError(
            "Strategy definition binding drifted"
        )
    version = metadata["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise CandidateLabProjectionError("Strategy version is invalid")
    _text(metadata["created_at"], "Strategy created_at")
    if metadata["adopted_at"] is not None:
        _text(metadata["adopted_at"], "Strategy adopted_at")
    if metadata["parent_strategy_id"] is not None:
        _text(metadata["parent_strategy_id"], "Strategy parent id")
    return {
        "strategy": strategy,
        "metadata": metadata,
        "strategy_spec_hash": normalized_hash,
    }


def _project_strategy_history_item(
    context: _ProjectionContext,
    *,
    repository: StrategyRepository,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    strategy = binding["strategy"]
    metadata = binding["metadata"]
    strategy_id = metadata["id"]
    materialization = repository.get_pool_materialization_for_strategy(
        strategy_id
    )
    projected_materialization = None
    if materialization is not None:
        projected_materialization = _project_strategy_materialization(
            materialization,
            task_id=context.task_id,
            strategy_id=strategy_id,
            strategy_type=metadata["strategy_type"],
            strategy_version=metadata["version"],
            strategy_spec_hash=binding["strategy_spec_hash"],
        )
    return {
        "strategy_id": strategy_id,
        "strategy_type": metadata["strategy_type"],
        "version": metadata["version"],
        "status": metadata["status"],
        "asset_status": metadata["asset_status"],
        "created_at": metadata["created_at"],
        "adopted_at": metadata["adopted_at"],
        "parent_strategy_id": metadata["parent_strategy_id"],
        "rule_count": len(strategy.rules),
        "strategy_spec_hash": binding["strategy_spec_hash"],
        "materialization": projected_materialization,
        "artifacts": _project_strategy_artifacts(
            context,
            repository=repository,
            strategy_id=strategy_id,
        ),
    }


def _project_strategy_materialization(
    value: Mapping[str, Any],
    *,
    task_id: str,
    strategy_id: str,
    strategy_type: str,
    strategy_version: int,
    strategy_spec_hash: str,
) -> dict[str, Any]:
    materialization = _mapping(value, "Strategy materialization")
    materialization_spec_hash = _sha256(
        materialization.get("strategy_spec_hash"),
        "materialized Strategy spec hash",
    )
    if (
        materialization.get("task_id") != task_id
        or materialization.get("strategy_id") != strategy_id
        or materialization.get("strategy_type") != strategy_type
        or materialization.get("strategy_version") != strategy_version
        or not hmac.compare_digest(
            materialization_spec_hash,
            strategy_spec_hash,
        )
    ):
        raise CandidateLabProjectionError(
            "Strategy materialization binding drifted"
        )
    requirements = normalize_pool_requirements(
        materialization.get("requirements")
    )
    return {
        "materialization_id": _text(
            materialization.get("id"),
            "Strategy materialization id",
        ),
        "pool_id": _text(
            materialization.get("pool_id"),
            "Strategy materialization Pool id",
        ),
        "pool_revision_id": _text(
            materialization.get("pool_revision_id"),
            "Strategy materialization Pool revision id",
        ),
        "pool_revision": materialization["pool_revision"],
        "pool_artifact_id": _text(
            materialization.get("pool_artifact_id"),
            "Strategy materialization Pool artifact id",
        ),
        "design_hash": _sha256(
            materialization.get("selected_design_hash"),
            "Strategy materialization design hash",
        ),
        "requirements_count": len(requirements),
        "runtime_blockers": [],
    }


def _project_strategy_artifacts(
    context: _ProjectionContext,
    *,
    repository: StrategyRepository,
    strategy_id: str,
) -> dict[str, Any]:
    records, total = (
        repository.list_recent_strategy_artifacts_for_task_with_count(
            context.task_id,
            strategy_id,
            limit=_MAX_STRATEGY_ARTIFACTS,
        )
    )
    if total < len(records):
        raise CandidateLabProjectionError(
            "Strategy artifact total is inconsistent"
        )
    projected = [
        _project_strategy_artifact(
            context,
            record=record,
            strategy_id=strategy_id,
        )
        for record in records
    ]
    return {
        "all": projected,
        "total": total,
        "truncated": total > len(projected),
    }


def _project_strategy_artifact(
    context: _ProjectionContext,
    *,
    record: Mapping[str, Any],
    strategy_id: str,
) -> dict[str, Any]:
    artifact_id = _text(record.get("id"), "Strategy artifact id")
    if (
        record.get("strategy_id") != strategy_id
        or record.get("integrity_status") != "verified"
    ):
        raise CandidateLabProjectionError(
            "Strategy artifact ownership or integrity metadata drifted"
        )
    content_size = record.get("content_size")
    if (
        isinstance(content_size, bool)
        or not isinstance(content_size, int)
        or content_size < 0
    ):
        raise CandidateLabProjectionError(
            "Strategy artifact content size is invalid"
        )
    expected_hash = _sha256(
        record.get("content_hash"),
        "Strategy artifact content hash",
    )
    path = Path(_text(record.get("path"), "Strategy artifact path"))
    if path.suffix.lower() not in _STRATEGY_ARTIFACT_SUFFIXES:
        raise CandidateLabProjectionError(
            "Strategy artifact type is not downloadable"
        )
    raw = _read_regular_file(
        path,
        root=Path(context.settings.tasks_dir) / context.task_id,
        max_bytes=_MAX_ARTIFACT_BYTES,
        budget=context.budget,
    )
    if (
        len(raw) != content_size
        or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(),
            expected_hash,
        )
    ):
        raise CandidateLabProjectionError(
            "Strategy artifact physical bytes drifted"
        )
    provenance = _mapping(
        record.get("provenance"),
        "Strategy artifact provenance",
    )
    if provenance.get("task_id") not in {None, context.task_id} or provenance.get(
        "strategy_id"
    ) not in {None, strategy_id}:
        raise CandidateLabProjectionError(
            "Strategy artifact provenance ownership drifted"
        )
    return {
        "artifact_id": artifact_id,
        "kind": _text(record.get("kind"), "Strategy artifact kind"),
        "filename": path.name,
        "created_at": _text(
            record.get("created_at"),
            "Strategy artifact created_at",
        ),
        "content_size": content_size,
        "download_url": (
            f"/api/tasks/{quote(context.task_id, safe='')}"
            f"/strategy-artifacts/{quote(artifact_id, safe='')}/download"
        ),
    }


def _project_current_project_context(
    context: _ProjectionContext,
) -> dict[str, Any] | None:
    runtime = SimpleNamespace(settings=context.settings)
    binding = load_current_strategy_project_context_artifact(
        runtime,
        task_id=context.task_id,
    )
    if binding is None:
        return None
    record = context.artifact_repository.get_for_task(
        context.task_id,
        binding.artifact_id,
    )
    if record is None:
        raise CandidateLabProjectionError(
            "current Strategy project context artifact is missing"
        )
    _require_record_identity(record, task_id=context.task_id)
    if (
        Path(record["path"]) != binding.artifact_path
        or record["content_hash"] != binding.artifact_content_hash
        or record["provenance"] != binding.provenance
    ):
        raise CandidateLabProjectionError(
            "current Strategy project context artifact binding drifted"
        )
    try:
        context.budget.reserve(binding.artifact_path.stat().st_size)
    except OSError as exc:
        raise CandidateLabProjectionError(
            "current Strategy project context artifact is unavailable"
        ) from exc

    revision = binding.revision
    state = _mapping(
        revision.get("state"),
        "current Strategy project context state",
    )
    snapshot = _mapping(
        state.get("current_project_snapshot"),
        "current Strategy project snapshot",
    )
    histories = [
        _project_strategy_history_review(item)
        for item in _sequence(
            state.get("historical_strategy_reviews"),
            "historical Strategy reviews",
        )
    ]
    missing = [
        _project_missing_information(item)
        for item in _sequence(
            state.get("missing_information_records"),
            "Strategy missing-information records",
        )
    ]
    history_resolution = _project_history_resolution(
        histories=histories,
        missing=missing,
    )
    return {
        "revision_id": revision["revision_id"],
        "revision": revision["revision"],
        "as_of": state["as_of"],
        "freshness": "current",
        "scope": _project_report_field(snapshot["scope"]),
        "current": {
            "snapshot_id": snapshot["snapshot_id"],
            "status_fields": {
                key: _project_report_field(snapshot["status_fields"][key])
                for key in ("volume", "approval", "risk", "economics")
            },
            "maturity_summary": _project_report_field(
                snapshot["maturity_summary"]
            ),
            "red_flags": _project_red_flags(snapshot["red_flags"]),
        },
        "historical_versions": histories,
        "history_resolution": history_resolution,
        "missing_information": missing,
        "red_flags": _project_red_flags(state["red_flags"]),
        "artifact": _artifact_projection(record, context.task_id),
    }


def _project_strategy_history_review(value: object) -> dict[str, Any]:
    review = _mapping(value, "historical Strategy review")
    effect_refs = _mapping(
        review.get("observation_refs_by_effect_stage"),
        "historical Strategy effect refs",
    )
    return {
        "review_id": review["review_id"],
        "version": review["version"],
        "effective_period": _project_report_field(review["effective_period"]),
        "asset_status": _project_report_field(review["asset_status"]),
        "scope": _project_report_field(review["scope"]),
        "traffic_allocation": _project_report_field(
            review["traffic_allocation"]
        ),
        "availability": review["availability"],
        "effect_stages": [
            stage
            for stage in (
                "estimated",
                "backtested",
                "oot_validated",
                "post_launch_observed",
            )
            if effect_refs.get(stage)
        ],
        "external_source_count": len(
            _sequence(
                review.get("external_source_refs"),
                "historical Strategy external refs",
            )
        ),
        "red_flags": _project_red_flags(review["red_flags"]),
    }


def _project_missing_information(value: object) -> dict[str, Any]:
    record = _mapping(value, "Strategy missing-information record")
    return {
        "field_path": record["field_path"],
        "status": record["status"],
        "blocking": record["blocking"],
        "question": record["question"],
        "reason": record["reason"],
        "asked_count": record["asked_count"],
    }


def _project_history_resolution(
    *,
    histories: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
) -> str:
    if histories:
        return "present"
    record = next(
        (
            item
            for item in missing
            if item.get("field_path") == "historical_strategy_reviews"
        ),
        None,
    )
    if record is None:
        return "pending"
    status = record.get("status")
    return "unavailable" if status == "unavailable" else "pending"


def _project_report_field(value: object) -> dict[str, Any]:
    field = _mapping(value, "Strategy report field")
    return {
        "availability": field["availability"],
        "value": field["value"],
    }


def _project_red_flags(value: object) -> list[dict[str, Any]]:
    return [
        {
            "code": flag["code"],
            "level": flag["level"],
            "message": flag["message"],
        }
        for flag in (
            _mapping(item, "Strategy project-context red flag")
            for item in _sequence(value, "Strategy project-context red flags")
        )
    ]


def _workflow_projection(
    *,
    project_context: dict[str, Any] | None,
    sample_design: dict[str, Any] | None,
    candidates: Mapping[str, Sequence[dict[str, Any]]],
    pools: Sequence[dict[str, Any]],
    latest_evidence: dict[str, Any],
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_count = sum(len(items) for items in candidates.values())
    has_voting = bool(candidates.get("voting_search"))
    evidence_items = [
        latest_evidence.get(key)
        for key in ("pool_stability", "pool_impact", "impact_cube")
    ] + list(latest_evidence["pool_validation"].values())
    present_evidence = [item for item in evidence_items if item is not None]
    impact_status = (
        "missing"
        if not present_evidence
        else (
            "stale"
            if all(item.get("freshness") == "stale" for item in present_evidence)
            else "complete"
        )
    )
    report_status = (
        "missing"
        if report is None
        else (
            "stale"
            if report.get("freshness") == "stale"
            else "complete"
        )
    )
    stages = (
        (
            "current_context",
            "项目现状",
            "complete" if project_context is not None else "missing",
        ),
        (
            "history",
            "历史版本",
            (
                "complete"
                if project_context is not None
                and project_context["history_resolution"]
                in {"present", "unavailable"}
                else "missing"
            ),
        ),
        (
            "sample_design",
            "样本设计",
            "complete" if sample_design is not None else "missing",
        ),
        (
            "candidate_analysis",
            "单变量/模型",
            "complete" if candidate_count > 0 else "missing",
        ),
        (
            "strategy_combination",
            "交叉组合/策略",
            "complete" if bool(pools) or has_voting else "missing",
        ),
        ("impact", "影响测算", impact_status),
        ("report", "形成报告", report_status),
    )
    return {
        "project_context": project_context,
        "sample_design": sample_design,
        "latest_evidence": latest_evidence,
        "report": report,
        "stages": [
            {
                "id": stage_id,
                "label": label,
                "status": status,
            }
            for stage_id, label, status in stages
        ],
    }


def _project_pool_add_sources(
    context: _ProjectionContext,
) -> dict[str, Any]:
    """Project only independently replayed, already materialized Pool sources."""

    univariate_records, univariate_total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            ASSET_ARTIFACT_KIND,
            limit=_MAX_POOL_ADD_SOURCES_PER_KIND,
        )
    )
    automatic_records, automatic_total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
            limit=_MAX_POOL_ADD_SOURCES_PER_KIND,
        )
    )
    cross_records, cross_total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
            limit=_MAX_POOL_ADD_SOURCES_PER_KIND,
        )
    )
    interactive_records, interactive_total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
            limit=_MAX_POOL_ADD_SOURCES_PER_KIND,
        )
    )
    interactive_group_records, interactive_group_total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
            limit=_MAX_POOL_ADD_SOURCES_PER_KIND,
        )
    )
    scorecard_records, scorecard_total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
            limit=_MAX_SCORECARD_CANDIDATES_PER_KIND,
        )
    )
    voting_records, voting_total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            VOTING_CANDIDATE_ARTIFACT_KIND,
            limit=_MAX_POOL_ADD_SOURCES_PER_KIND,
        )
    )
    cross_rule_records, cross_rule_total = (
        context.artifact_repository.list_recent_for_task_kind_with_count(
            context.task_id,
            CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
            limit=_MAX_POOL_ADD_SOURCES_PER_KIND,
        )
    )
    projected = []
    for record in univariate_records:
        fragment = _verified_univariate_asset_fragment(context, record)
        projected.append(
            (
                record,
                _pool_add_source_from_fragment(
                    fragment,
                    source_kind="univariate_asset",
                    candidate_asset_id=_text(
                        _mapping(
                            fragment.get("asset"),
                            "univariate Pool source asset",
                        ).get("asset_id"),
                        "univariate Pool source asset_id",
                    ),
                ),
            )
        )
    for record in automatic_records:
        fragment = _verified_automatic_tree_selection_fragment(
            context,
            record,
        )
        provenance = _mapping(
            record.get("provenance"),
            "automatic-tree Pool source provenance",
        )
        projected.append(
            (
                record,
                _pool_add_source_from_fragment(
                    fragment,
                    source_kind="automatic_tree_leaf_selection",
                    selection_id=_text(
                        provenance.get("selection_id"),
                        "automatic-tree Pool source selection_id",
                    ),
                ),
            )
        )
    for record in cross_records:
        fragment = _verified_cross_matrix_selection_fragment(
            context,
            record,
        )
        provenance = _mapping(
            record.get("provenance"),
            "cross-matrix Pool source provenance",
        )
        projected.append(
            (
                record,
                _pool_add_source_from_fragment(
                    fragment,
                    source_kind="cross_matrix_cell_selection",
                    selection_id=_text(
                        provenance.get("selection_id"),
                        "cross-matrix Pool source selection_id",
                    ),
                ),
            )
        )
    for records, source_kind, verifier in (
        (
            interactive_records,
            "interactive_tree_frontier_selection",
            _verified_interactive_tree_selection_fragment,
        ),
        (
            interactive_group_records,
            "interactive_tree_frontier_group_selection",
            _verified_interactive_tree_group_selection_fragment,
        ),
    ):
        for record in records:
            provenance = _mapping(
                record.get("provenance"),
                f"{source_kind} Pool source provenance",
            )
            source_binding = {
                "artifact_id": record.get("id"),
                "artifact_content_hash": record.get("content_hash"),
                "asset_id": provenance.get("semantic_tree_id"),
                "asset_hash": provenance.get("tree_hash"),
            }
            fragment = verifier(context, source_binding, record)
            projected.append(
                (
                    record,
                    _pool_add_source_from_fragment(
                        fragment,
                        source_kind=source_kind,
                        selection_id=_text(
                            provenance.get("selection_id"),
                            f"{source_kind} Pool source selection_id",
                        ),
                    ),
                )
            )
    for record in scorecard_records:
        verified = _verified_scorecard_cutoff_selection(context, record)
        selection = _mapping(
            verified.get("selection"),
            "scorecard Pool source selection",
        )
        projected.append(
            (
                record,
                _pool_add_source_from_fragment(
                    _mapping(
                        verified.get("fragment"),
                        "scorecard Pool source fragment",
                    ),
                    source_kind="scorecard_cutoff_selection",
                    selection_id=_text(
                        selection.get("selection_id"),
                        "scorecard Pool source selection_id",
                    ),
                ),
            )
        )
    for record in voting_records:
        fragment = _verified_voting_candidate_fragment(context, record)
        provenance = _mapping(
            record.get("provenance"),
            "Voting Pool source provenance",
        )
        asset = _mapping(
            fragment.get("asset"),
            "Voting Pool source asset",
        )
        strategy_type = _text(
            provenance.get("strategy_type"),
            "Voting Pool source strategy_type",
        )
        if strategy_type not in STRATEGY_TYPES:
            raise CandidateLabProjectionError(
                "Voting Pool source strategy_type is unsupported"
            )
        projected.append(
            (
                record,
                _pool_add_source_from_fragment(
                    fragment,
                    source_kind="voting_candidate",
                    candidate_asset_id=_text(
                        asset.get("asset_id"),
                        "Voting Pool source asset_id",
                    ),
                    strategy_type=strategy_type,
                ),
            )
        )
    for record in cross_rule_records:
        fragment = _verified_cross_rule_candidate_fragment(context, record)
        asset = _mapping(
            fragment.get("asset"),
            "Cross rule Pool source asset",
        )
        projected.append(
            (
                record,
                _pool_add_source_from_fragment(
                    fragment,
                    source_kind="cross_threshold_rule",
                    candidate_asset_id=_text(
                        asset.get("asset_id"),
                        "Cross rule Pool source asset_id",
                    ),
                ),
            )
        )
    return _pool_add_source_collection(
        projected,
        total=(
            univariate_total
            + automatic_total
            + cross_total
            + interactive_total
            + interactive_group_total
            + scorecard_total
            + voting_total
            + cross_rule_total
        ),
        limit=_MAX_POOL_ADD_SOURCES_PER_KIND * 8,
    )


def _pool_add_source_from_fragment(
    fragment: Mapping[str, Any],
    *,
    source_kind: str,
    candidate_asset_id: str | None = None,
    selection_id: str | None = None,
    strategy_type: str | None = None,
) -> dict[str, Any]:
    if (candidate_asset_id is None) == (selection_id is None):
        raise CandidateLabProjectionError(
            "Pool add source must expose exactly one candidate or selection id"
        )
    return {
        "source_kind": _text(source_kind, "Pool add source kind"),
        **(
            {"candidate_asset_id": candidate_asset_id}
            if candidate_asset_id is not None
            else {"selection_id": selection_id}
        ),
        "strategy_type": strategy_type,
        "candidate_stage": _text(
            fragment.get("candidate_stage"),
            "Pool add source candidate_stage",
        ),
        "validation_status": _text(
            fragment.get("validation_status"),
            "Pool add source validation_status",
        ),
    }


def _pool_add_source_collection(
    projected: Sequence[tuple[Mapping[str, Any], dict[str, Any]]],
    *,
    total: int,
    limit: int,
) -> dict[str, Any]:
    identities: set[str] = set()
    for _record, source in projected:
        identity = source.get("candidate_asset_id") or source.get("selection_id")
        if not isinstance(identity, str) or identity in identities:
            raise CandidateLabProjectionError(
                "Pool add source identity is missing or duplicated"
            )
        identities.add(identity)
    ordered = sorted(
        projected,
        key=lambda pair: (
            pair[0]["created_at"],
            pair[0]["id"],
        ),
        reverse=True,
    )
    visible = [source for _record, source in ordered[:limit]]
    if total < len(projected):
        raise CandidateLabProjectionError(
            "Pool add source total is inconsistent"
        )
    return {
        "latest": visible[0] if visible else None,
        "all": visible,
        "total": total,
        "truncated": total > len(visible),
    }


def _project_univariate(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    verified = _verified_univariate_source(context, record)
    evidence = verified["evidence"]
    analysis = verified["analysis"]
    candidate_id = evidence["candidate_id"]
    analysis_schema = analysis["schema_version"]
    generation = evidence["generation"]

    rankings, rankings_truncated = _bounded_list(
        analysis.get("rankings", []),
        _MAX_RANKINGS,
    )
    metrics, metrics_truncated = _bounded_list(
        evidence.get("metrics", []),
        _MAX_METRICS,
    )
    bin_pointers = []
    for feature in _sequence(analysis.get("features", []), "analysis.features"):
        feature_name = feature["feature"]
        for method in _sequence(feature.get("methods", []), "feature.methods"):
            if method.get("status") != "available":
                continue
            for bin_row in _sequence(
                method.get("bins", []),
                "feature method bins",
            ):
                bin_pointers.append(
                    {
                        "feature": feature_name,
                        "method": method["method"],
                        "bin_id": bin_row["id"],
                        "condition": bin_row["condition"],
                        "metrics": _univariate_bin_metrics(bin_row),
                    }
                )
    total = len(bin_pointers)
    projected_bins = bin_pointers[:_MAX_BIN_POINTERS]
    item_truncated = bool(
        generation.get("truncated")
        or rankings_truncated
        or metrics_truncated
        or total > len(projected_bins)
    )
    return {
        "kind": "univariate",
        "artifact": _artifact_projection(record, context.task_id),
        "candidate_id": candidate_id,
        "lifecycle": {
            "candidate_stage": evidence["effect_stage"],
            "observation_stage": None,
            "validation_status": evidence["validation_status"],
        },
        "detail": {
            "analysis_schema_version": analysis_schema,
            "rankings": rankings,
            "metrics": metrics,
            "generation": {
                "seed": generation["seed"],
                "budget": generation["budget"],
                "truncated": generation["truncated"],
                "parameters": {
                    key: generation["parameters"][key]
                    for key in _UNIVARIATE_PARAMETER_FIELDS
                    if key in generation["parameters"]
                },
            },
        },
        "risks": {
            "red_flags": list(evidence.get("red_flags", []))[:_MAX_RISKS],
            "report_info_gaps": [],
        },
        "pointers": {"bins": projected_bins},
        "total": total,
        "truncated": item_truncated,
    }


def _project_voting_search(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    _read_candidate_record(
        context,
        record,
        kind=VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
        origin_tool=VOTING_CANDIDATE_SEARCH_ORIGIN_TOOL,
        directory_name="strategy_voting_candidate_searches",
    )
    binding = load_historical_voting_candidate_search_artifact(
        _scorecard_live_runtime(context),
        task_id=context.task_id,
        artifact_id=_text(record.get("id"), "Voting search artifact id"),
        expected_artifact_content_hash=_sha256(
            record.get("content_hash"),
            "Voting search artifact content hash",
        ),
    )
    result = binding.result
    configuration = result["configuration"]
    pool = binding.pool_development.pool.pool
    combinations = [
        {
            "combo_id": item["combo_id"],
            "members": list(item["member_ids"]),
            "eligible": item["eligible"],
            "failures": [dict(failure) for failure in item["constraint_failures"]],
            "metrics": dict(item["metrics"]),
        }
        for item in result["combinations"][:_MAX_VOTING_SEARCH_COMBINATIONS]
    ]
    return {
        "search_id": result["search_id"],
        "strategy_type": pool["strategy_type"],
        "pool_revision": pool["revision"],
        "member_count": configuration["member_count"],
        "n": configuration["n"],
        "objective": dict(configuration["objective"]),
        "constraints": [
            dict(constraint) for constraint in configuration["constraints"]
        ],
        "include_rule_ids": list(configuration["include"]),
        "exclude_rule_ids": list(configuration["exclude"]),
        "max_combinations": configuration["max_combinations"],
        "search_space": result["search_space"],
        "evaluated": result["evaluated"],
        "eligible": result["eligible"],
        "truncated": result["truncated"],
        "combinations": combinations,
        "artifact": _artifact_projection(record, context.task_id),
    }


def _project_cross_search(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    _read_candidate_record(
        context,
        record,
        kind=CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
        origin_tool=CROSS_CANDIDATE_SEARCH_ORIGIN_TOOL,
        directory_name="strategy_cross_candidate_searches",
    )
    binding = load_cross_candidate_search_artifact(
        _scorecard_live_runtime(context),
        task_id=context.task_id,
        artifact_id=_text(record.get("id"), "Cross search artifact id"),
        expected_artifact_content_hash=_sha256(
            record.get("content_hash"),
            "Cross search artifact content hash",
        ),
    )
    result = binding.result
    configuration = result["configuration"]
    pairs = [
        {
            key: item[key]
            for key in (
                "pair_id",
                "x_feature",
                "x_method",
                "y_feature",
                "y_method",
                "x_axis_iv",
                "y_axis_iv",
                "cross_total_iv",
                "interaction_gain_iv",
                "cell_count",
                "empty_cell_count",
                "empty_cell_share",
                "min_nonempty_cell_count",
                "eligible",
                "rank",
            )
        }
        for item in result["pairs"][:_MAX_CROSS_SEARCH_PAIRS]
    ]
    return {
        "search_id": result["search_id"],
        "features": [
            {
                key: item[key]
                for key in ("feature", "method", "axis_iv", "bin_count")
            }
            for item in configuration["features"]
        ],
        "max_pairs": configuration["max_pairs"],
        "search_space": result["search_space"],
        "evaluated": result["evaluated"],
        "eligible": result["eligible"],
        "truncated": result["truncated"],
        "pairs": pairs,
        "artifact": _artifact_projection(record, context.task_id),
    }


def _project_cross_rule_search(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    _read_candidate_record(
        context,
        record,
        kind=CROSS_RULE_SEARCH_ARTIFACT_KIND,
        origin_tool=CROSS_RULE_SEARCH_ORIGIN_TOOL,
        directory_name="strategy_cross_rule_searches",
    )
    binding = load_cross_rule_search_artifact(
        _scorecard_live_runtime(context),
        task_id=context.task_id,
        artifact_id=_text(record.get("id"), "Cross rule search artifact id"),
        expected_artifact_content_hash=_sha256(
            record.get("content_hash"),
            "Cross rule search artifact content hash",
        ),
    )
    result = binding.result
    configuration = result["configuration"]
    rules = [
        {
            "rule_id": item["rule_id"],
            "rank": item["rank"],
            "conditions": [
                dict(condition) for condition in item["conditions"]
            ],
            "metrics": dict(item["metrics"]),
            "eligible": item["eligible"],
            "constraint_failures": list(item["constraint_failures"]),
        }
        for item in result["rules"][:_MAX_CROSS_RULES]
    ]
    return {
        "search_id": result["search_id"],
        "dimension": configuration["dimension"],
        "features": [
            {
                key: item[key]
                for key in (
                    "feature",
                    "method",
                    "risk_direction",
                    "thresholds",
                    "excluded_values",
                    "missing_count",
                    "missing_bad",
                )
            }
            for item in configuration["features"]
        ],
        "constraints": dict(configuration["constraints"]),
        "max_trials": configuration["max_trials"],
        "search_space": result["search_space"],
        "evaluated": result["evaluated"],
        "eligible": result["eligible"],
        "truncated": result["truncated"],
        "rules": rules,
        "rules_truncated": len(result["rules"]) > len(rules),
        "artifact": _artifact_projection(record, context.task_id),
    }


def _load_cross_rule_candidate_binding(
    context: _ProjectionContext,
    record: Mapping[str, Any],
):
    _read_candidate_record(
        context,
        record,
        kind=CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
        origin_tool=CROSS_RULE_CANDIDATE_ORIGIN_TOOL,
        directory_name="strategy_cross_rule_candidates",
    )
    provenance = _mapping(
        record.get("provenance"),
        "Cross rule candidate provenance",
    )
    binding = load_cross_rule_candidate_artifact(
        _scorecard_live_runtime(context),
        task_id=context.task_id,
        artifact_id=_text(record.get("id"), "Cross rule candidate artifact id"),
        expected_artifact_content_hash=_sha256(
            record.get("content_hash"),
            "Cross rule candidate artifact content hash",
        ),
        expected_asset_id=_text(
            provenance.get("asset_id"),
            "Cross rule candidate asset_id",
        ),
        expected_asset_hash=_sha256(
            provenance.get("asset_hash"),
            "Cross rule candidate asset_hash",
        ),
    )
    replay_cross_rule_candidate_binding(
        _scorecard_live_runtime(context),
        binding,
    )
    return binding


def _project_cross_rule_candidate(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _load_cross_rule_candidate_binding(context, record)
    candidate = binding.candidate
    selection = candidate["source_selection"]
    return {
        "kind": "cross_rule_candidate",
        "artifact": _artifact_projection(record, context.task_id),
        "candidate_id": candidate["asset_id"],
        "lifecycle": dict(candidate["lifecycle"]),
        "detail": {
            "asset_id": candidate["asset_id"],
            "asset_type": candidate["asset_type"],
            "search_id": selection["search_id"],
            "rule_id": selection["rule_id"],
            "rule_rank": selection["rule_rank"],
            "eligible": selection["eligible"],
            "constraint_failures": list(
                selection["constraint_failures"]
            ),
            "dimension": candidate["dimension"],
            "conditions": [
                dict(item) for item in candidate["condition"]["args"]
            ],
            "metrics": dict(candidate["metrics"]),
            "selection_reason": candidate["selection_reason"],
            "effect_stage": candidate["effect_stage"],
            "validation_status": candidate["validation_status"],
        },
        "risks": {
            "red_flags": [],
            "report_info_gaps": (
                []
                if selection["eligible"]
                else [
                    "该规则未满足搜索约束；仍允许按精确 ID 物化，"
                    "但入池前应说明评审依据。"
                ]
            ),
        },
        "pointers": {},
        "total": 1,
        "truncated": False,
    }


def _verified_univariate_source(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = (
        _text(record.get("id"), "univariate artifact id"),
        _sha256(record.get("content_hash"), "univariate content hash"),
    )
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        if not isinstance(cached, dict) or cached.get("kind") != "univariate":
            raise CandidateLabProjectionError("artifact verification cache drifted")
        return cached
    raw = _read_candidate_record(
        context,
        record,
        kind=UNIVARIATE_ARTIFACT_KIND,
        origin_tool=UNIVARIATE_ORIGIN_TOOL,
        directory_name="strategy_candidates",
    )
    report = strategy_candidate_report_from_json(raw)
    evidence = report["candidate_evidence"]
    analysis = report["univariate_analysis"]
    canonical = canonical_strategy_candidate_report_json(evidence, analysis)
    _require_bytes_equal(canonical, raw)

    candidate_id = evidence["candidate_id"]
    expected_path = (
        Path(context.settings.tasks_dir)
        / context.task_id
        / "strategy_candidates"
        / f"{candidate_id}_{record['content_hash'][:12]}.json"
    )
    _require_exact_path(record, expected_path)

    analysis_schema = analysis["schema_version"]
    version_contract = _UNIVARIATE_VERSION_CONTRACTS.get(analysis_schema)
    if version_contract is None:
        raise CandidateLabProjectionError("unsupported univariate schema")
    artifact_schema, producer_version = version_contract
    identity = evidence["identity"]
    generation = evidence["generation"]
    provenance = _mapping(record["provenance"], "univariate provenance")
    registry_metadata_hash = _sha256(
        provenance.get("registry_metadata_hash"),
        "registry_metadata_hash",
    )
    expected_provenance = {
        "schema_version": artifact_schema,
        "producer_version": producer_version,
        "candidate_id": candidate_id,
        "evidence_hash": evidence["evidence_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "registry_metadata_hash": registry_metadata_hash,
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "generation_parameters": generation["parameters"],
        "format": "json",
    }
    if (
        identity["task_id"] != context.task_id
        or evidence["producer_version"] != producer_version
        or provenance != expected_provenance
    ):
        raise CandidateLabProjectionError("univariate provenance drifted")
    verified = {
        "kind": "univariate",
        "evidence": evidence,
        "analysis": analysis,
        "artifact_schema_version": artifact_schema,
        "registry_metadata_hash": registry_metadata_hash,
    }
    context.verified_cache[cache_key] = verified
    return verified


def _project_cross_matrix(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    verified = _verified_cross_matrix_source(context, record)
    asset = verified["asset"]
    lifecycle = asset["lifecycle"]

    cells = asset["matrix"]["cells"]
    projected_cells = [
        {
            "cell_id": cell["cell_id"],
            "row_bin_id": cell["row_bin_id"],
            "column_bin_id": cell["column_bin_id"],
            "effect": _cross_effect_projection(cell["effect"]),
        }
        for cell in cells[:_MAX_CELL_POINTERS]
    ]
    return {
        "kind": "cross_matrix",
        "artifact": _artifact_projection(record, context.task_id),
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "lifecycle": lifecycle,
        "detail": {
            "asset_id": asset["asset_id"],
            "axes": [
                {
                    "position": axis["position"],
                    "feature": axis["feature"],
                    "method": axis["method"],
                    "bin_count": len(axis["bins"]),
                }
                for axis in asset["axes"]
            ],
            "summary": _cross_summary_projection(asset["summary"]),
        },
        "risks": {"red_flags": [], "report_info_gaps": []},
        "pointers": {"cells": projected_cells},
        "total": len(cells),
        "truncated": bool(
            asset["budget"]["truncated"] or len(cells) > len(projected_cells)
        ),
    }


def _cross_effect_projection(effect: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "effect_id",
        "count",
        "good",
        "bad",
        "share",
        "bad_rate",
        "lift",
        "woe",
        "iv_contribution",
        "amount_metrics",
    )
    return {key: effect[key] for key in fields}


def _cross_summary_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "count",
        "good",
        "bad",
        "bad_rate",
        "total_iv",
        "amount_metrics",
    )
    return {key: summary[key] for key in fields}


def _verified_cross_matrix_source(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = (
        _text(record.get("id"), "cross-matrix artifact id"),
        _sha256(record.get("content_hash"), "cross-matrix content hash"),
    )
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        if not isinstance(cached, dict) or cached.get("kind") != "cross_matrix":
            raise CandidateLabProjectionError("artifact verification cache drifted")
        return cached
    raw = _read_candidate_record(
        context,
        record,
        kind=CROSS_MATRIX_ARTIFACT_KIND,
        origin_tool=CROSS_MATRIX_ORIGIN_TOOL,
        directory_name="strategy_cross_matrix_candidates",
    )
    asset = parse_cross_matrix_candidate_asset_json(raw)
    canonical = canonical_cross_matrix_candidate_asset_json(asset).encode("utf-8")
    _require_bytes_equal(canonical, raw)
    expected_path = canonical_cross_matrix_source_path(
        context.settings.tasks_dir,
        task_id=context.task_id,
        asset_id=asset["asset_id"],
        content_hash=record["content_hash"],
    )
    _require_exact_path(record, expected_path)

    provenance = _mapping(record["provenance"], "cross-matrix provenance")
    source = _require_source_artifact(
        context,
        artifact_id=provenance.get("source_artifact_id"),
        content_hash=provenance.get("source_artifact_content_hash"),
        kind=UNIVARIATE_ARTIFACT_KIND,
        origin_tool=UNIVARIATE_ORIGIN_TOOL,
    )
    verified_source = _verified_univariate_source(context, source)
    if asset["parent"] != _cross_parent_from_univariate_evidence(
        verified_source["evidence"]
    ):
        raise CandidateLabProjectionError(
            "cross-matrix parent does not match source report"
        )
    identity = asset["parent"]["identity"]
    sample = asset["sample_identity"]
    lifecycle = asset["lifecycle"]
    expected_provenance = {
        "schema_version": _CROSS_MATRIX_ARTIFACT_SCHEMAS[asset["schema_version"]],
        "producer_version": asset["producer_version"],
        "asset_schema_version": asset["schema_version"],
        "asset_type": asset["asset_type"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "parent_candidate_id": asset["parent"]["candidate_id"],
        "parent_evidence_hash": asset["parent"]["evidence_hash"],
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "source_artifact_id": provenance["source_artifact_id"],
        "source_artifact_content_hash": provenance[
            "source_artifact_content_hash"
        ],
        "task_id": identity["task_id"],
        "dataset_id": sample["dataset_id"],
        "dataset_content_hash": sample["dataset_content_hash"],
        "registry_metadata_hash": verified_source["registry_metadata_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "sample_context_hash": sample["sample_context_hash"],
        "target_col": sample["target_col"],
        "labeled_row_count": sample["row_count"],
        "row_axis": _cross_axis_provenance(asset["axes"][0]),
        "column_axis": _cross_axis_provenance(asset["axes"][1]),
        "cell_count": asset["matrix"]["cell_count"],
        "candidate_stage": lifecycle["candidate_stage"],
        "observation_stage": lifecycle["observation_stage"],
        "validation_status": lifecycle["validation_status"],
        "budget": asset["budget"]["limit"],
        "truncated": asset["budget"]["truncated"],
    }
    if identity["task_id"] != context.task_id or provenance != expected_provenance:
        raise CandidateLabProjectionError("cross-matrix provenance drifted")
    artifact_schema = expected_provenance["schema_version"]
    verify_cross_matrix_source_provenance(
        provenance,
        source_binding={
            "artifact_id": record["id"],
            "task_id": context.task_id,
            "kind": CROSS_MATRIX_ARTIFACT_KIND,
            "artifact_schema_version": artifact_schema,
            "content_hash": record["content_hash"],
            "origin_tool": CROSS_MATRIX_ORIGIN_TOOL,
            "path": record["path"],
            "canonical_bytes": raw,
        },
        asset_payload=asset,
    )
    verified = {
        "kind": "cross_matrix",
        "asset": asset,
        "provenance": dict(provenance),
        "artifact_schema_version": artifact_schema,
        "raw": raw,
    }
    context.verified_cache[cache_key] = verified
    return verified


def _cross_parent_from_univariate_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = _mapping(evidence.get("analysis"), "univariate analysis")
    parameters = _mapping(analysis.get("parameters"), "analysis parameters")
    identity = _mapping(evidence.get("identity"), "candidate identity")
    parent = {
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "identity": {
            key: identity[key]
            for key in (
                "task_id",
                "dataset_id",
                "dataset_content_hash",
                "workspace_revision",
                "workspace_generation",
                "semantic_mapping_hash",
            )
        },
        "target_col": analysis["target"],
        "row_count": analysis["row_count"],
        "target_definition": {"good": 0, "bad": 1},
        "smoothing": parameters["smoothing"],
    }
    if analysis["schema_version"] == "univariate-analysis-result.v2":
        parent["analysis_schema_version"] = analysis["schema_version"]
    return parent


def _project_automatic_tree(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    verified = _verified_automatic_tree_source(context, record)
    asset = verified["asset"]
    tree = asset["tree_result"]

    fragments = asset["fragments"]
    projected_leaves = [
        {
            "leaf_id": fragment["leaf_id"],
            "fragment_id": fragment["fragment_id"],
            "rule_id": fragment["rule_id"],
            "effect_id": fragment["effect_id"],
            "condition": fragment["condition"],
            "metrics": fragment["metrics"],
        }
        for fragment in fragments[:_MAX_LEAF_POINTERS]
    ]
    tree_info = tree["tree"]
    diagnostics = asset["diagnostics"]
    topology = interactive_tree_topology_evidence(asset)
    return {
        "kind": "automatic_tree",
        "artifact": _artifact_projection(record, context.task_id),
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "lifecycle": asset["lifecycle"],
        "detail": {
            "asset_id": asset["asset_id"],
            "tree_id": tree_info["tree_id"],
            "source_tree_id": asset["asset_id"],
            "summary": {
                "node_count": tree_info["node_count"],
                "leaf_count": tree_info["leaf_count"],
                "features": tree["training"]["feature_order"],
                "max_depth": tree["training"]["cart"]["max_depth"],
            },
        },
        "risks": {
            "red_flags": list(diagnostics["red_flags"])[:_MAX_RISKS],
            "report_info_gaps": [],
        },
        "pointers": {
            "leaves": projected_leaves,
            **_interactive_tree_topology_pointers(
                source_tree_id=asset["asset_id"],
                topology=topology,
                feature_universe=tree["training"]["feature_order"],
            ),
        },
        "total": len(fragments),
        "truncated": len(fragments) > len(projected_leaves),
    }


def _project_interactive_tree_revisions(
    context: _ProjectionContext,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        return []
    revision_ids = tuple(
        _text(
            _mapping(record.get("provenance"), "interactive-tree provenance").get(
                "revision_id"
            ),
            "interactive-tree revision_id",
        )
        for record in records
    )
    runtime = SimpleNamespace(
        settings=context.settings,
        task_artifacts=context.artifact_repository,
    )
    try:
        verified = load_verified_interactive_tree_revisions(
            runtime,
            task_id=context.task_id,
            revision_ids=revision_ids,
            reserve_bytes=context.budget.reserve,
        )
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "interactive-tree revision verification failed"
        ) from exc
    projected = []
    for record, revision_id in zip(records, revision_ids, strict=True):
        binding = verified[revision_id]
        if (
            binding.artifact_id != record["id"]
            or binding.path != Path(record["path"])
            or not hmac.compare_digest(
                binding.content_hash,
                _sha256(
                    record.get("content_hash"),
                    "interactive-tree content hash",
                ),
            )
            or binding.provenance
            != _mapping(
                record.get("provenance"),
                "interactive-tree provenance",
            )
        ):
            raise CandidateLabProjectionError(
                "interactive-tree registry binding drifted"
            )
        projected.append(
            _project_interactive_tree_revision(
                context,
                record,
                binding=binding,
            )
        )
    return projected


def _project_interactive_tree_revision(
    context: _ProjectionContext,
    record: Mapping[str, Any],
    *,
    binding: VerifiedInteractiveTreeRevision,
) -> dict[str, Any]:
    revision = binding.revision
    ancestry = binding.ancestor_revisions
    topology = interactive_tree_topology_evidence(
        binding.automatic_source.asset,
        revision_payload=revision,
        parent_revision=ancestry[0] if ancestry else None,
        ancestor_revisions=ancestry[1:],
    )
    parent_ref = revision["parent_revision"]
    source_tree_id = revision["revision_id"]
    projected_frontier = [
        {
            key: fragment[key]
            for key in (
                "source_node_id",
                "leaf_id",
                "fragment_id",
                "rule_id",
                "effect_id",
                "condition",
                "metrics",
            )
        }
        for fragment in revision["fragments"]
    ]
    history = [
        {
            "revision_id": item["revision_id"],
            "semantic_tree_id": item["semantic_tree_id"],
            "parent_revision_id": (
                None
                if item["parent_revision"] is None
                else item["parent_revision"]["revision_id"]
            ),
            "edit": item["edit"],
        }
        for item in (revision, *ancestry)
    ]
    return {
        "kind": "interactive_tree_revision",
        "artifact": _artifact_projection(record, context.task_id),
        "candidate_id": revision["candidate_evidence"]["candidate_id"],
        "lifecycle": revision["lifecycle"],
        "detail": {
            "revision_id": revision["revision_id"],
            "semantic_tree_id": revision["semantic_tree_id"],
            "source_tree_id": source_tree_id,
            "derived_from_source_tree_id": binding.provenance["source_tree_id"],
            "base_asset_id": revision["base_tree"]["asset_id"],
            "parent_revision_id": (
                None if parent_ref is None else parent_ref["revision_id"]
            ),
            "edit": revision["edit"],
            "summary": {
                "node_count": len(topology["nodes"]),
                "visible_node_count": len(topology["visible_node_ids"]),
                "frontier_node_count": len(topology["frontier_node_ids"]),
            },
        },
        "risks": {
            "red_flags": [],
            "report_info_gaps": [],
        },
        "history": history,
        "pointers": {
            "frontier": projected_frontier,
            **_interactive_tree_topology_pointers(
                source_tree_id=source_tree_id,
                topology=topology,
                feature_universe=binding.automatic_source.asset[
                    "tree_result"
                ]["training"]["feature_order"],
            ),
        },
        "total": len(topology["nodes"]),
        "truncated": False,
    }


def _interactive_tree_topology_pointers(
    *,
    source_tree_id: str,
    topology: Mapping[str, Any],
    feature_universe: Sequence[str],
) -> dict[str, Any]:
    nodes = [
        dict(node)
        for node in _sequence(topology.get("nodes"), "interactive-tree nodes")
    ]
    eligible_prunes = [
        {
            "source_tree_id": source_tree_id,
            "node_id": node["node_id"],
            "operation": "prune_subtree",
        }
        for node in nodes
        if node.get("can_prune") is True
    ]
    eligible_threshold_adjustments = [
        {
            "source_tree_id": source_tree_id,
            "node_id": _text(
                node.get("node_id"),
                "interactive-tree threshold node_id",
            ),
            "operation": "adjust_split_threshold",
            "feature": _text(
                node.get("feature"),
                "interactive-tree threshold feature",
            ),
            "current_threshold": float(node["threshold"]),
        }
        for node in nodes
        if node.get("can_prune") is True
    ]
    authenticated_features = list(
        _text_sequence(
            feature_universe,
            "interactive-tree feature universe",
        )
    )
    eligible_feature_replacements = [
        {
            "source_tree_id": source_tree_id,
            "node_id": _text(
                node.get("node_id"),
                "interactive-tree replacement node_id",
            ),
            "operation": "replace_split_feature",
            "current_feature": _text(
                node.get("feature"),
                "interactive-tree current feature",
            ),
            "current_threshold": float(node["threshold"]),
        }
        for node in nodes
        if (
            node.get("can_prune") is True
            and any(
                feature != node.get("feature")
                for feature in authenticated_features
            )
        )
    ]
    return {
        "root_node_id": _text(
            topology.get("root_node_id"),
            "interactive-tree root_node_id",
        ),
        "nodes": nodes,
        "visible_node_ids": list(
            _text_sequence(
                topology.get("visible_node_ids"),
                "interactive-tree visible_node_ids",
            )
        ),
        "frontier_node_ids": list(
            _text_sequence(
                topology.get("frontier_node_ids"),
                "interactive-tree frontier_node_ids",
            )
        ),
        "eligible_prunes": eligible_prunes,
        "eligible_threshold_adjustments": eligible_threshold_adjustments,
        "eligible_feature_replacements": eligible_feature_replacements,
        "feature_universe": authenticated_features,
    }


def _verified_automatic_tree_source(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = (
        _text(record.get("id"), "automatic-tree artifact id"),
        _sha256(record.get("content_hash"), "automatic-tree content hash"),
    )
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        if not isinstance(cached, dict) or cached.get("kind") != "automatic_tree":
            raise CandidateLabProjectionError("artifact verification cache drifted")
        return cached
    raw = _read_candidate_record(
        context,
        record,
        kind=AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
        origin_tool=AUTOMATIC_TREE_ASSET_ORIGIN_TOOL,
        directory_name="strategy_automatic_trees",
    )
    asset = validate_automatic_tree_asset(_json_object(raw, "automatic tree"))
    canonical = canonical_automatic_tree_asset_json(asset).encode("utf-8")
    _require_bytes_equal(canonical, raw)
    expected_path = canonical_automatic_tree_source_path(
        context.settings.tasks_dir,
        task_id=context.task_id,
        asset_id=asset["asset_id"],
    )
    _require_exact_path(record, expected_path)

    provenance = _mapping(record["provenance"], "automatic-tree provenance")
    identity = asset["identity"]
    expected_provenance = automatic_tree_source_provenance_from_asset(asset)
    if identity["task_id"] != context.task_id or provenance != expected_provenance:
        raise CandidateLabProjectionError("automatic-tree provenance drifted")
    verify_automatic_tree_source_provenance(provenance, asset)
    verified = {
        "kind": "automatic_tree",
        "asset": asset,
        "provenance": dict(provenance),
        "artifact_schema_version": AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION,
        "raw": raw,
    }
    context.verified_cache[cache_key] = verified
    return verified


def _scorecard_directions_projection() -> dict[str, dict[str, str]]:
    return {
        "raw_pd": {
            "direction": "higher_is_riskier",
            "meaning": "higher_raw_pd_means_higher_risk",
        },
        "scorecard_points": {
            "direction": "higher_is_better",
            "meaning": "higher_points_mean_safer",
        },
    }


def _scorecard_cutoff_projection(
    cutoff: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: cutoff[key]
        for key in (
            "ordinal",
            "cutoff_id",
            "execution_pd",
            "display_points",
            "lower_risk",
            "higher_risk",
        )
    }


def _project_scorecard_band(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    verified = _verified_scorecard_band_source(context, record)
    asset = verified["asset"]
    bands = [
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
        for band in asset["bands"][:_MAX_SCORECARD_BAND_POINTERS]
    ]
    cutoffs = [
        _scorecard_cutoff_projection(cutoff)
        for cutoff in asset["cutoffs"][:_MAX_SCORECARD_CUTOFF_POINTERS]
    ]
    scorecard_table = asset["score_contract"]["scorecard_table"]
    scorecard_points = [
        {
            key: row[key]
            for key in _SCORECARD_POINT_FIELDS
            if key in row
        }
        for row in scorecard_table[:_MAX_SCORECARD_POINT_POINTERS]
    ]
    score_vector = asset["score_vector"]
    total = (
        len(asset["bands"])
        + len(asset["cutoffs"])
        + len(scorecard_table)
    )
    projected_total = len(bands) + len(cutoffs) + len(scorecard_points)
    return {
        "kind": "scorecard_band",
        "artifact": _artifact_projection(record, context.task_id),
        "candidate_id": asset["asset_id"],
        "lifecycle": asset["lifecycle"],
        "detail": {
            "asset_id": asset["asset_id"],
            "performance": asset["performance"],
            "sample": {
                key: score_vector[key]
                for key in (
                    "row_count",
                    "development_count",
                    "labeled_count",
                    "bad_count",
                )
            },
            "directions": _scorecard_directions_projection(),
        },
        "risks": {"red_flags": [], "report_info_gaps": []},
        "pointers": {
            "bands": bands,
            "cutoffs": cutoffs,
            "scorecard_points": scorecard_points,
        },
        "total": total,
        "truncated": total > projected_total,
    }


def _project_scorecard_cutoff_selection(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    verified = _verified_scorecard_cutoff_selection(context, record)
    selection = verified["selection"]
    asset = verified["asset"]
    cutoff = verified["cutoff"]
    return {
        "kind": "scorecard_cutoff_selection",
        "artifact": _artifact_projection(record, context.task_id),
        "candidate_id": selection["selection_id"],
        "lifecycle": asset["lifecycle"],
        "detail": {
            "selection_id": selection["selection_id"],
            "asset_id": asset["asset_id"],
            "cutoff_id": cutoff["cutoff_id"],
            "reason": selection["selection_reason"],
            "directions": _scorecard_directions_projection(),
            "effect": _scorecard_cutoff_projection(cutoff),
        },
        "risks": {"red_flags": [], "report_info_gaps": []},
        "pointers": {},
        "total": 1,
        "truncated": False,
    }


def _verified_scorecard_band_source(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        if not isinstance(cached, dict) or cached.get("kind") != "scorecard_band":
            raise CandidateLabProjectionError("artifact verification cache drifted")
        return cached
    raw = _read_candidate_record(
        context,
        record,
        kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        origin_tool=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        directory_name="strategy_scorecard_candidates",
    )
    asset = validate_scorecard_band_asset(
        _json_object(raw, "scorecard band asset")
    )
    _require_bytes_equal(
        canonical_scorecard_band_asset_json(asset).encode("utf-8"),
        raw,
    )
    expected_path = (
        Path(context.settings.tasks_dir)
        / context.task_id
        / "strategy_scorecard_candidates"
        / f"{asset['asset_id']}.json"
    )
    _require_exact_path(record, expected_path)
    provenance = _mapping(record["provenance"], "scorecard band provenance")
    if (
        asset["identity"]["task_id"] != context.task_id
        or provenance != _scorecard_band_provenance(asset)
    ):
        raise CandidateLabProjectionError("scorecard band provenance drifted")
    _verify_scorecard_direct_sources(context, asset)
    try:
        live = load_scorecard_band_asset_artifact(
            _scorecard_live_runtime(context),
            task_id=context.task_id,
            artifact_id=record["id"],
            expected_artifact_content_hash=record["content_hash"],
            expected_asset_id=asset["asset_id"],
            expected_asset_hash=asset["asset_hash"],
        )
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "scorecard band failed authoritative source replay"
        ) from exc
    if live.asset != asset or live.canonical_bytes != raw:
        raise CandidateLabProjectionError(
            "scorecard authoritative replay disagrees with projected asset"
        )
    verified = {
        "kind": "scorecard_band",
        "asset": asset,
        "provenance": dict(provenance),
        "raw": raw,
    }
    context.verified_cache[cache_key] = verified
    return verified


def _scorecard_live_runtime(context: _ProjectionContext) -> Any:
    cached = context.scorecard_runtime
    if cached is not None:
        return cached
    settings = context.settings
    datasets_root = getattr(
        settings,
        "datasets_dir",
        settings.workspace / "datasets",
    )
    backend = DataBackend(datasets_root)
    runtime = SimpleNamespace(
        settings=settings,
        backend=backend,
        registry=DatasetRegistry(
            DatasetRepository(settings.db_path),
            backend,
            datasets_root,
        ),
        task_artifacts=context.artifact_repository,
        experiments=ExperimentStore(settings.db_path),
        modeling_repo=ModelingRepository(settings.db_path),
    )
    context.scorecard_runtime = runtime
    return runtime


def _verified_scorecard_cutoff_selection(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        if (
            not isinstance(cached, dict)
            or cached.get("kind") != "scorecard_cutoff_selection"
        ):
            raise CandidateLabProjectionError("artifact verification cache drifted")
        return cached
    raw = _read_candidate_record(
        context,
        record,
        kind=SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        origin_tool=SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        directory_name="strategy_scorecard_candidates",
    )
    selection = validate_scorecard_cutoff_selection(
        _json_object(raw, "scorecard cutoff selection")
    )
    _require_bytes_equal(
        canonical_scorecard_cutoff_selection_json(selection).encode("utf-8"),
        raw,
    )
    expected_path = (
        Path(context.settings.tasks_dir)
        / context.task_id
        / "strategy_scorecard_candidates"
        / f"{selection['selection_id']}.json"
    )
    _require_exact_path(record, expected_path)
    provenance = _mapping(
        record["provenance"],
        "scorecard cutoff selection provenance",
    )
    if provenance != _scorecard_selection_provenance(selection):
        raise CandidateLabProjectionError(
            "scorecard cutoff selection provenance drifted"
        )
    source_pointer = selection["source_asset_ref"]
    source_record = _require_source_artifact(
        context,
        artifact_id=source_pointer["artifact_id"],
        content_hash=source_pointer["artifact_content_hash"],
        kind=source_pointer["kind"],
        origin_tool=source_pointer["origin_tool"],
    )
    source = _verified_scorecard_band_source(context, source_record)
    selection_binding = _scorecard_artifact_binding(
        record,
        artifact_schema_version=(
            SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        canonical_bytes=raw,
    )
    source_binding = _scorecard_artifact_binding(
        source_record,
        artifact_schema_version=SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
        canonical_bytes=source["raw"],
    )
    fragment = scorecard_cutoff_selection_to_verified_candidate_fragment(
        selection,
        source["asset"],
        selection_artifact_binding=selection_binding,
        source_artifact_binding=source_binding,
    )
    cutoff = next(
        (
            cutoff
            for cutoff in source["asset"]["cutoffs"]
            if cutoff["cutoff_id"] == selection["cutoff_id"]
        ),
        None,
    )
    if cutoff is None:
        raise CandidateLabProjectionError(
            "scorecard cutoff selection source no longer contains cutoff"
        )
    verified = {
        "kind": "scorecard_cutoff_selection",
        "selection": selection,
        "asset": source["asset"],
        "cutoff": cutoff,
        "fragment": fragment,
    }
    context.verified_cache[cache_key] = verified
    return verified


def _scorecard_artifact_binding(
    record: Mapping[str, Any],
    *,
    artifact_schema_version: str,
    canonical_bytes: bytes,
) -> dict[str, Any]:
    return {
        "artifact_id": record["id"],
        "task_id": record["task_id"],
        "kind": record["kind"],
        "artifact_schema_version": artifact_schema_version,
        "content_hash": record["content_hash"],
        "origin_tool": record["origin_tool"],
        "canonical_bytes": canonical_bytes,
    }


def _scorecard_band_provenance(asset: Mapping[str, Any]) -> dict[str, Any]:
    identity = asset["identity"]
    refs = asset["source_refs"]
    return {
        "schema_version": SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
        "asset_schema_version": asset["schema_version"],
        "producer_version": asset["producer_version"],
        "task_id": identity["task_id"],
        "asset_type": asset["asset_type"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "sample_context_hash": identity["sample_context_hash"],
        "sample_design_ref": asset["sample_design_ref"],
        "training_evidence_ref": refs["training_evidence"],
        "score_evidence_ref": refs["score_evidence"],
        "score_vector_ref": refs["score_vector"],
        "score_product": asset["score_contract"]["score_product"],
        "scorecard_table_hash": asset["score_contract"]["scorecard_table_hash"],
        "raw_pd_internal_edges": [
            band["upper_bound"] for band in asset["bands"][:-1]
        ],
        "band_count": len(asset["bands"]),
        "cutoff_count": len(asset["cutoffs"]),
    }


def _scorecard_selection_provenance(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    source = selection["source_asset_ref"]
    return {
        "schema_version": SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
        "selection_schema_version": selection["schema_version"],
        "producer_version": selection["producer_version"],
        "task_id": source["task_id"],
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "cutoff_id": selection["cutoff_id"],
        "selection_reason": selection["selection_reason"],
        "source_artifact_id": source["artifact_id"],
        "source_artifact_content_hash": source["artifact_content_hash"],
        "source_asset_id": source["asset_id"],
        "source_asset_hash": source["asset_hash"],
    }


def _verify_scorecard_direct_sources(
    context: _ProjectionContext,
    asset: Mapping[str, Any],
) -> None:
    cache_payload = {
        "identity": asset["identity"],
        "sample_design_ref": asset["sample_design_ref"],
        "source_refs": asset["source_refs"],
    }
    cache_key = _canonical_mapping_hash(cache_payload)
    if cache_key in context.scorecard_source_cache:
        return

    task_root = Path(context.settings.tasks_dir) / context.task_id
    identity = asset["identity"]
    sample_ref = asset["sample_design_ref"]
    refs = asset["source_refs"]

    membership_record = _require_source_artifact(
        context,
        artifact_id=sample_ref["membership_artifact_id"],
        content_hash=sample_ref[
            "expected_membership_artifact_content_hash"
        ],
        kind=SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
        origin_tool=SAMPLE_DESIGN_V2_ORIGIN_TOOL,
    )
    membership_provenance = _mapping(
        membership_record["provenance"],
        "scorecard membership provenance",
    )
    membership_id = _source_path_component(
        membership_provenance.get("membership_id"),
        "scorecard membership id",
    )
    _require_exact_path(
        membership_record,
        task_root
        / "strategy_sample_designs_v2"
        / f"{membership_id}.bin",
    )
    _require_provenance_subset(
        membership_provenance,
        {
            "schema_version": SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION,
            "task_id": context.task_id,
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "format": "binary",
            "artifact_role": "membership",
            "membership_artifact_content_hash": membership_record[
                "content_hash"
            ],
        },
        "scorecard membership",
    )
    membership_content_hash = _sha256(
        membership_provenance.get("membership_content_hash"),
        "scorecard membership content hash",
    )

    bundle_record = _require_source_artifact(
        context,
        artifact_id=sample_ref["bundle_artifact_id"],
        content_hash=sample_ref["expected_bundle_artifact_content_hash"],
        kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        origin_tool=SAMPLE_DESIGN_V2_ORIGIN_TOOL,
    )
    bundle_id = _source_path_component(
        sample_ref["expected_bundle_id"],
        "scorecard bundle id",
    )
    _require_exact_path(
        bundle_record,
        task_root / "strategy_sample_designs_v2" / f"{bundle_id}.json",
    )
    bundle_provenance = _mapping(
        bundle_record["provenance"],
        "scorecard sample bundle provenance",
    )
    _require_provenance_subset(
        bundle_provenance,
        {
            "schema_version": SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION,
            "task_id": context.task_id,
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "format": "json",
            "artifact_role": "bundle",
            "membership_id": membership_id,
            "membership_content_hash": membership_content_hash,
            "membership_artifact_id": membership_record["id"],
            "membership_artifact_content_hash": membership_record[
                "content_hash"
            ],
            "bundle_id": bundle_id,
            "bundle_artifact_content_hash": bundle_record["content_hash"],
            "sample_design_id": sample_ref["expected_sample_design_id"],
            "sample_design_content_hash": sample_ref[
                "expected_sample_design_content_hash"
            ],
        },
        "scorecard sample bundle",
    )

    training_ref = refs["training_evidence"]
    training_record = _require_source_artifact(
        context,
        artifact_id=training_ref["artifact_id"],
        content_hash=training_ref["artifact_content_hash"],
        kind=MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        origin_tool=TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL,
    )
    training_evidence_id = _source_path_component(
        training_ref["evidence_id"],
        "scorecard training evidence id",
    )
    _require_exact_path(
        training_record,
        task_root
        / "modeling_artifacts"
        / f"{training_evidence_id}.training_evidence.json",
    )
    training_provenance = _mapping(
        training_record["provenance"],
        "scorecard training evidence provenance",
    )
    _require_provenance_subset(
        training_provenance,
        {
            "schema_version": TRAINING_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
            "format": "json",
            "artifact_role": "training_evidence",
            "task_id": context.task_id,
            "evidence_id": training_evidence_id,
            "evidence_content_hash": training_ref[
                "evidence_content_hash"
            ],
            "evidence_artifact_content_hash": training_record[
                "content_hash"
            ],
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "sample_design_id": sample_ref["expected_sample_design_id"],
            "sample_design_content_hash": sample_ref[
                "expected_sample_design_content_hash"
            ],
            "sample_membership_id": membership_id,
            "sample_membership_content_hash": membership_content_hash,
            "sample_membership_artifact_id": membership_record["id"],
            "sample_membership_artifact_content_hash": membership_record[
                "content_hash"
            ],
            "sample_bundle_artifact_id": bundle_record["id"],
            "sample_bundle_artifact_content_hash": bundle_record[
                "content_hash"
            ],
        },
        "scorecard training evidence",
    )

    score_ref = refs["score_evidence"]
    score_record = _require_source_artifact(
        context,
        artifact_id=score_ref["artifact_id"],
        content_hash=score_ref["artifact_content_hash"],
        kind=MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        origin_tool=MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
    )
    vector_ref = refs["score_vector"]
    vector_record = _require_source_artifact(
        context,
        artifact_id=vector_ref["artifact_id"],
        content_hash=vector_ref["artifact_content_hash"],
        kind=MODEL_SCORE_VECTOR_ARTIFACT_KIND,
        origin_tool=MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
    )
    score_provenance = _mapping(
        score_record["provenance"],
        "scorecard score evidence provenance",
    )
    vector_provenance = _mapping(
        vector_record["provenance"],
        "scorecard score vector provenance",
    )
    request_hash = _sha256(
        score_provenance.get("request_hash"),
        "scorecard score request hash",
    )
    if vector_provenance.get("request_hash") != request_hash:
        raise CandidateLabProjectionError(
            "scorecard score evidence and vector request drifted"
        )
    _require_exact_path(
        score_record,
        task_root
        / "model_score_evidence"
        / f"{request_hash}.model_score_evidence.json",
    )
    _require_exact_path(
        vector_record,
        task_root
        / "model_score_evidence"
        / f"{request_hash}.scores.parquet",
    )
    score_lineage = {
        "schema_version": (
            MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION
        ),
        "task_id": context.task_id,
        "request_hash": request_hash,
        "training_evidence_id": training_evidence_id,
        "training_evidence_content_hash": training_ref[
            "evidence_content_hash"
        ],
        "training_evidence_artifact_id": training_record["id"],
        "training_evidence_artifact_content_hash": training_record[
            "content_hash"
        ],
        "sample_design_id": sample_ref["expected_sample_design_id"],
        "sample_design_content_hash": sample_ref[
            "expected_sample_design_content_hash"
        ],
        "sample_membership_id": membership_id,
        "sample_membership_content_hash": membership_content_hash,
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "score_product": asset["score_contract"]["score_product"],
    }
    _require_provenance_subset(
        vector_provenance,
        {
            **score_lineage,
            "format": "parquet",
            "artifact_role": "model_score_vector",
            "row_count": asset["score_vector"]["row_count"],
        },
        "scorecard score vector",
    )
    _require_provenance_subset(
        score_provenance,
        {
            **score_lineage,
            "format": "json",
            "artifact_role": "model_score_evidence",
            "score_vector_artifact_id": vector_record["id"],
            "score_vector_artifact_content_hash": vector_record[
                "content_hash"
            ],
            "score_evidence_id": score_ref["evidence_id"],
            "score_evidence_content_hash": score_ref[
                "evidence_content_hash"
            ],
            "score_evidence_artifact_content_hash": score_record[
                "content_hash"
            ],
        },
        "scorecard score evidence",
    )
    context.scorecard_source_cache.add(cache_key)


def _require_provenance_subset(
    provenance: Mapping[str, Any],
    expected: Mapping[str, Any],
    name: str,
) -> None:
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise CandidateLabProjectionError(f"{name} provenance drifted")


def _source_path_component(value: object, name: str) -> str:
    text = _text(value, name)
    if Path(text).name != text:
        raise CandidateLabProjectionError(f"{name} is not path-safe")
    return text


def _project_current_pools(
    context: _ProjectionContext,
) -> list[dict[str, Any]]:
    pool_repository = StrategyCandidatePoolRepository(context.settings.db_path)
    projected = []
    for strategy_type in sorted(STRATEGY_TYPES):
        snapshot = pool_repository.get_current(context.task_id, strategy_type)
        if snapshot is None:
            continue
        pool = validate_strategy_pool(snapshot)
        if pool["task_id"] != context.task_id:
            raise CandidateLabProjectionError("pool task ownership drifted")
        expected_path = (
            Path(context.settings.tasks_dir)
            / context.task_id
            / "strategy_candidate_pools"
            / (
                f"{pool['pool_id']}_r{pool['revision']}_"
                f"{pool['snapshot_hash'][:12]}.json"
            )
        )
        record = context.artifact_repository.get_for_task_kind_path(
            context.task_id,
            POOL_ARTIFACT_KIND,
            str(expected_path),
        )
        if record is None:
            raise CandidateLabProjectionError("current pool artifact is missing")
        expected_hash = strategy_pool_artifact_content_hash(pool)
        if record.get("content_hash") != expected_hash:
            raise CandidateLabProjectionError("pool registry hash drifted")
        raw = _read_candidate_record(
            context,
            record,
            kind=POOL_ARTIFACT_KIND,
            origin_tool=_POOL_ORIGIN_BY_OPERATION[pool["operation"]["kind"]],
            directory_name="strategy_candidate_pools",
        )
        persisted = validate_strategy_pool(_json_object(raw, "strategy pool"))
        canonical = canonical_strategy_pool_snapshot_json(persisted).encode("utf-8")
        _require_bytes_equal(canonical, raw)
        if persisted != pool:
            raise CandidateLabProjectionError("current pool artifact drifted")
        entries = pool["entries"]
        if len(entries) > _MAX_POOL_ENTRIES:
            raise CandidateLabProjectionError("current pool entry cap exceeded")
        _verify_pool_entries_against_sources(context, pool)
        evidence_identity = (
            entries[0]["source"]["evidence_identity"] if entries else None
        )
        expected_provenance = {
            "schema_version": _POOL_ARTIFACT_SCHEMA_VERSION,
            "producer_version": POOL_PRODUCER_VERSION,
            "pool_id": pool["pool_id"],
            "strategy_type": pool["strategy_type"],
            "revision": pool["revision"],
            "revision_id": pool["revision_id"],
            "parent_revision_id": pool["parent_revision_id"],
            "snapshot_hash": pool["snapshot_hash"],
            "operation_kind": pool["operation"]["kind"],
            "source_artifact_ids": [
                entry["source"]["artifact_id"] for entry in entries
            ],
            "evidence_identity": evidence_identity,
        }
        if record["provenance"] != expected_provenance:
            raise CandidateLabProjectionError("pool provenance drifted")
        projected_entries = [
            {
                "entry_id": entry["entry_id"],
                "rule_id": entry["rule_id"],
                "position": entry["position"],
                "source": _pool_source_projection(entry["source"]),
                "execution": _pool_execution_projection(entry["execution"]),
                "action": entry["action"],
                "enabled": entry["enabled"],
            }
            for entry in entries[:_MAX_POOL_ENTRIES]
        ]
        projected.append(
            {
                "kind": "candidate_pool",
                "artifact": _artifact_projection(record, context.task_id),
                "pool_id": pool["pool_id"],
                "strategy_type": pool["strategy_type"],
                "revision": pool["revision"],
                "status": pool["status"],
                "validation_status": pool["validation_status"],
                "operation": {
                    key: pool["operation"][key]
                    for key in ("kind", "reason")
                    if key in pool["operation"]
                },
                "default_action": pool["default_action"],
                "entries": projected_entries,
                "total": len(entries),
                "truncated": len(entries) > len(projected_entries),
            }
        )
    return projected


def _verify_pool_entries_against_sources(
    context: _ProjectionContext,
    pool: Mapping[str, Any],
) -> None:
    replay_key = (
        _text(pool.get("revision_id"), "pool revision_id"),
        _sha256(pool.get("snapshot_hash"), "pool snapshot_hash"),
    )
    if replay_key in context.verified_pool_entry_replays:
        return
    if replay_key in context.pool_entry_replays_in_progress:
        raise CandidateLabProjectionError("voting parent pool cycle detected")
    context.pool_entry_replays_in_progress.add(replay_key)
    try:
        entries = _sequence(pool.get("entries"), "pool entries")
        if len(entries) > _MAX_POOL_ENTRIES:
            raise CandidateLabProjectionError("pool artifact entry cap exceeded")
        for entry in entries:
            source = _mapping(entry.get("source"), "pool entry source")
            fragment = _verified_pool_source_fragment(context, source)
            replayed_source, replayed_rule_id, replayed_execution = (
                verified_fragment_pool_parts(fragment)
            )
            if (
                source != replayed_source
                or entry.get("rule_id") != replayed_rule_id
                or entry.get("execution") != replayed_execution
            ):
                raise CandidateLabProjectionError(
                    "pool entry does not match replayed source fragment"
                )
    finally:
        context.pool_entry_replays_in_progress.discard(replay_key)
    context.verified_pool_entry_replays.add(replay_key)


def _pool_source_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    visible_fields = (
        "asset_id",
        "asset_type",
        "fragment_id",
        "fragment_type",
        "effect_id",
        "evidence_id",
        "candidate_stage",
        "observation_stage",
        "validation_status",
    )
    return {key: source[key] for key in visible_fields if key in source}


def _pool_execution_projection(
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    requirements = _sequence(
        execution.get("requirements", []),
        "pool execution requirements",
    )
    return {
        "condition": _ui_safe_pool_condition(execution["condition"]),
        "requirement_types": [
            _text(requirement.get("type"), "pool requirement type")
            for requirement in requirements
        ],
    }


def _ui_safe_pool_condition(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _ui_safe_pool_condition(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_ui_safe_pool_condition(item) for item in value]
    if (
        isinstance(value, str)
        and _MODEL_SCORE_VIRTUAL_FIELD_RE.fullmatch(value) is not None
    ):
        return "scorecard_raw_pd"
    return value


def _verified_pool_source_fragment(
    context: _ProjectionContext,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    record = _require_source_artifact(
        context,
        artifact_id=source.get("artifact_id"),
        content_hash=source.get("artifact_content_hash"),
        kind=source.get("artifact_kind"),
        origin_tool=source.get("origin_tool"),
    )
    dispatch = (record["kind"], record["origin_tool"])
    if dispatch == (ASSET_ARTIFACT_KIND, CANDIDATE_ASSET_ORIGIN_TOOL):
        return _verified_univariate_asset_fragment(context, record)
    if dispatch == (
        CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
    ):
        return _verified_cross_matrix_selection_fragment(context, record)
    if dispatch == (
        AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    ):
        return _verified_automatic_tree_selection_fragment(context, record)
    if dispatch == (
        INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
    ):
        return _verified_interactive_tree_selection_fragment(
            context,
            source,
            record,
        )
    if dispatch == (
        INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
        INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL,
    ):
        return _verified_interactive_tree_group_selection_fragment(
            context,
            source,
            record,
        )
    if dispatch == (
        VOTING_CANDIDATE_ARTIFACT_KIND,
        VOTING_CANDIDATE_ORIGIN_TOOL,
    ):
        return _verified_voting_candidate_fragment(context, record)
    if dispatch == (
        SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    ):
        return _verified_scorecard_cutoff_selection(context, record)[
            "fragment"
        ]
    if dispatch == (
        CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
        CROSS_RULE_CANDIDATE_ORIGIN_TOOL,
    ):
        return _verified_cross_rule_candidate_fragment(context, record)
    raise CandidateLabProjectionError("pool source artifact contract is unsupported")


def _verified_cross_rule_candidate_fragment(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        return _cached_fragment(cached, "cross_rule_candidate_fragment")
    binding = _load_cross_rule_candidate_binding(context, record)
    identity = binding.search.evidence["identity"]
    fragment = cross_rule_candidate_to_verified_fragment(
        binding.candidate,
        artifact_binding={
            "artifact_id": binding.artifact_id,
            "artifact_kind": CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
            "artifact_schema_version": (
                CROSS_RULE_CANDIDATE_ARTIFACT_SCHEMA_VERSION
            ),
            "artifact_content_hash": binding.artifact_content_hash,
            "origin_tool": CROSS_RULE_CANDIDATE_ORIGIN_TOOL,
        },
        evidence_identity={
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "sample_context_hash": binding.search.result["source"][
                "sample_context_hash"
            ],
        },
    )
    context.verified_cache[cache_key] = {
        "kind": "cross_rule_candidate_fragment",
        "fragment": fragment,
    }
    return fragment


def _verified_univariate_asset_fragment(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        return _cached_fragment(cached, "univariate_asset_fragment")
    raw = _read_candidate_record(
        context,
        record,
        kind=ASSET_ARTIFACT_KIND,
        origin_tool=CANDIDATE_ASSET_ORIGIN_TOOL,
        directory_name="strategy_candidate_assets",
    )
    asset = validate_candidate_asset(_json_object(raw, "candidate asset"))
    _require_bytes_equal(
        canonical_candidate_asset_json(asset).encode("utf-8"),
        raw,
    )
    expected_path = (
        Path(context.settings.tasks_dir)
        / context.task_id
        / "strategy_candidate_assets"
        / f"{asset['asset_id']}_{record['content_hash'][:12]}.json"
    )
    _require_exact_path(record, expected_path)
    provenance = _mapping(record["provenance"], "candidate asset provenance")
    parent_record = _require_source_artifact(
        context,
        artifact_id=provenance.get("source_artifact_id"),
        content_hash=provenance.get("source_artifact_content_hash"),
        kind=UNIVARIATE_ARTIFACT_KIND,
        origin_tool=UNIVARIATE_ORIGIN_TOOL,
    )
    parent = _verified_univariate_source(context, parent_record)
    evidence = parent["evidence"]
    identity = evidence["identity"]
    expected_provenance = {
        "schema_version": ASSET_ARTIFACT_SCHEMA_VERSION,
        "producer_version": asset["producer_version"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "source_artifact_id": parent_record["id"],
        "source_artifact_content_hash": parent_record["content_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "feature": asset["feature"],
        "method": asset["method"],
    }
    if provenance != expected_provenance:
        raise CandidateLabProjectionError("candidate asset provenance drifted")
    source_binding = {
        "artifact_id": record["id"],
        "kind": record["kind"],
        "content_hash": record["content_hash"],
        "origin_tool": record["origin_tool"],
        "artifact_schema_version": ASSET_ARTIFACT_SCHEMA_VERSION,
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_kind": asset["asset_type"],
        "fragment_id": asset["rule"]["rule_id"],
        "effect_id": asset["effect"]["effect_id"],
        "effect_stage": asset["effect_stage"],
        "validation_status": asset["validation_status"],
        "parent_candidate_id": evidence["candidate_id"],
        "parent_evidence_hash": evidence["evidence_hash"],
        "evidence_identity": {
            key: identity[key]
            for key in (
                "dataset_id",
                "dataset_content_hash",
                "workspace_revision",
                "workspace_generation",
                "semantic_mapping_hash",
            )
        },
    }
    fragment = univariate_asset_to_verified_fragment(
        asset,
        source_binding=source_binding,
        candidate_evidence=evidence,
    )
    context.verified_cache[cache_key] = {
        "kind": "univariate_asset_fragment",
        "fragment": fragment,
    }
    return fragment


def _verified_cross_matrix_selection_fragment(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        return _cached_fragment(cached, "cross_matrix_selection_fragment")
    raw = _read_candidate_record(
        context,
        record,
        kind=CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        origin_tool=CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
        directory_name="strategy_cross_matrix_cell_selections",
    )
    selection = validate_cross_matrix_cell_selection(
        _json_object(raw, "cross-matrix cell selection")
    )
    _require_bytes_equal(
        canonical_cross_matrix_cell_selection_json(selection).encode("utf-8"),
        raw,
    )
    expected_path = canonical_cross_matrix_cell_selection_path(
        context.settings.tasks_dir,
        task_id=context.task_id,
        selection_id=selection["selection_id"],
    )
    _require_exact_path(record, expected_path)
    provenance = verify_cross_matrix_cell_selection_provenance(
        _mapping(record["provenance"], "cross-matrix selection provenance"),
        selection,
    )
    source_pointer = selection["source_artifact"]
    source_record = _require_source_artifact(
        context,
        artifact_id=source_pointer["artifact_id"],
        content_hash=source_pointer["content_hash"],
        kind=source_pointer["kind"],
        origin_tool=source_pointer["origin_tool"],
    )
    source = _verified_cross_matrix_source(context, source_record)
    source_binding = {
        "artifact_id": source_record["id"],
        "task_id": context.task_id,
        "kind": source_record["kind"],
        "artifact_schema_version": source["artifact_schema_version"],
        "content_hash": source_record["content_hash"],
        "origin_tool": source_record["origin_tool"],
        "path": source_record["path"],
        "provenance_hash": _canonical_mapping_hash(source["provenance"]),
        "provenance": source["provenance"],
        "canonical_bytes": source["raw"],
    }
    fragment = cross_matrix_cell_selection_to_verified_candidate_fragment(
        selection,
        source["asset"],
        selection_artifact_binding=_selection_binding(record, provenance),
        source_artifact_binding=source_binding,
    )
    context.verified_cache[cache_key] = {
        "kind": "cross_matrix_selection_fragment",
        "fragment": fragment,
    }
    return fragment


def _verified_automatic_tree_selection_fragment(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        return _cached_fragment(cached, "automatic_tree_selection_fragment")
    raw = _read_candidate_record(
        context,
        record,
        kind=AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        origin_tool=AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
        directory_name="strategy_automatic_tree_leaf_fragments",
    )
    selection = validate_automatic_tree_leaf_fragment(
        _json_object(raw, "automatic-tree leaf selection")
    )
    _require_bytes_equal(
        canonical_automatic_tree_leaf_fragment_json(selection).encode("utf-8"),
        raw,
    )
    expected_path = canonical_automatic_tree_leaf_selection_path(
        context.settings.tasks_dir,
        task_id=context.task_id,
        selection_id=selection["selection_id"],
    )
    _require_exact_path(record, expected_path)
    provenance = verify_automatic_tree_leaf_selection_provenance(
        _mapping(record["provenance"], "automatic-tree selection provenance"),
        selection,
    )
    source_pointer = selection["tree_artifact"]
    source_record = _require_source_artifact(
        context,
        artifact_id=source_pointer["artifact_id"],
        content_hash=source_pointer["content_hash"],
        kind=source_pointer["kind"],
        origin_tool=source_pointer["origin_tool"],
    )
    source = _verified_automatic_tree_source(context, source_record)
    source_binding = {
        "artifact_id": source_record["id"],
        "task_id": context.task_id,
        "kind": source_record["kind"],
        "artifact_schema_version": source["artifact_schema_version"],
        "content_hash": source_record["content_hash"],
        "origin_tool": source_record["origin_tool"],
        "path": source_record["path"],
        "provenance": source["provenance"],
        "canonical_bytes": source["raw"],
    }
    fragment = automatic_tree_leaf_fragment_to_verified_candidate_fragment(
        selection,
        source["asset"],
        selection_artifact_binding=_selection_binding(record, provenance),
        tree_artifact_binding=source_binding,
    )
    context.verified_cache[cache_key] = {
        "kind": "automatic_tree_selection_fragment",
        "fragment": fragment,
    }
    return fragment


def _verified_interactive_tree_selection_fragment(
    context: _ProjectionContext,
    source: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        return _cached_fragment(cached, "interactive_tree_selection_fragment")
    runtime = SimpleNamespace(
        settings=context.settings,
        task_artifacts=context.artifact_repository,
    )
    try:
        selection = (
            load_verified_interactive_tree_frontier_selection_artifact(
                runtime,
                task_id=context.task_id,
                artifact_id=_text(
                    source.get("artifact_id"),
                    "interactive-tree selection artifact_id",
                ),
                expected_content_hash=_sha256(
                    source.get("artifact_content_hash"),
                    "interactive-tree selection content hash",
                ),
                expected_asset_id=_text(
                    source.get("asset_id"),
                    "interactive-tree semantic asset_id",
                ),
                expected_asset_hash=_sha256(
                    source.get("asset_hash"),
                    "interactive-tree semantic asset hash",
                ),
                reserve_bytes=context.budget.reserve,
            )
        )
        revision = selection.revision
        ancestry = revision.ancestor_revisions
        fragment = (
            interactive_tree_frontier_selection_to_verified_candidate_fragment(
                selection.selection,
                revision.revision,
                revision.automatic_source.asset,
                selection_artifact_binding=selection.artifact_binding(),
                revision_artifact_binding=revision.builder_binding(),
                parent_revision=ancestry[0] if ancestry else None,
                ancestor_revisions=ancestry[1:],
            )
        )
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "interactive-tree frontier selection verification failed"
        ) from exc
    context.verified_cache[cache_key] = {
        "kind": "interactive_tree_selection_fragment",
        "fragment": fragment,
    }
    return fragment


def _verified_interactive_tree_group_selection_fragment(
    context: _ProjectionContext,
    source: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        return _cached_fragment(
            cached,
            "interactive_tree_group_selection_fragment",
        )
    runtime = SimpleNamespace(
        settings=context.settings,
        task_artifacts=context.artifact_repository,
    )
    try:
        selection = (
            load_verified_interactive_tree_frontier_group_selection_artifact(
                runtime,
                task_id=context.task_id,
                artifact_id=_text(
                    source.get("artifact_id"),
                    "interactive-tree group selection artifact_id",
                ),
                expected_content_hash=_sha256(
                    source.get("artifact_content_hash"),
                    "interactive-tree group selection content hash",
                ),
                expected_asset_id=_text(
                    source.get("asset_id"),
                    "interactive-tree group semantic asset_id",
                ),
                expected_asset_hash=_sha256(
                    source.get("asset_hash"),
                    "interactive-tree group semantic asset hash",
                ),
                reserve_bytes=context.budget.reserve,
            )
        )
        revision = selection.revision
        ancestry = revision.ancestor_revisions
        fragment = (
            interactive_tree_frontier_group_selection_to_verified_candidate_fragment(
                selection.selection,
                revision.revision,
                revision.automatic_source.asset,
                selection_artifact_binding=selection.artifact_binding(),
                revision_artifact_binding=revision.builder_binding(),
                parent_revision=ancestry[0] if ancestry else None,
                ancestor_revisions=ancestry[1:],
            )
        )
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "interactive-tree frontier group selection verification failed"
        ) from exc
    context.verified_cache[cache_key] = {
        "kind": "interactive_tree_group_selection_fragment",
        "fragment": fragment,
    }
    return fragment


def _verified_voting_candidate_fragment(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        return _cached_fragment(cached, "voting_candidate_fragment")
    raw = _read_candidate_record(
        context,
        record,
        kind=VOTING_CANDIDATE_ARTIFACT_KIND,
        origin_tool=VOTING_CANDIDATE_ORIGIN_TOOL,
        directory_name="strategy_voting_candidates",
    )
    document = validate_voting_candidate_artifact_document(
        _json_object(raw, "voting candidate")
    )
    _require_bytes_equal(
        canonical_voting_candidate_artifact_json(document).encode("utf-8"),
        raw,
    )
    asset = document["asset"]
    expected_path = canonical_voting_candidate_path(
        context.settings.tasks_dir,
        task_id=context.task_id,
        asset_id=asset["asset_id"],
    )
    _require_exact_path(record, expected_path)
    provenance = _mapping(record["provenance"], "voting candidate provenance")
    parent_record = _require_source_artifact(
        context,
        artifact_id=provenance.get("pool_artifact_id"),
        content_hash=provenance.get("pool_artifact_content_hash"),
        kind=POOL_ARTIFACT_KIND,
        origin_tool=_pool_artifact_origin(context, provenance),
    )
    parent_pool = _verified_pool_snapshot_record(context, parent_record)
    _verify_pool_entries_against_sources(context, parent_pool)
    pool_ref = asset["pool_ref"]
    for key, expected in (
        ("pool_id", parent_pool["pool_id"]),
        ("strategy_type", parent_pool["strategy_type"]),
        ("revision", parent_pool["revision"]),
        ("revision_id", parent_pool["revision_id"]),
        ("snapshot_hash", parent_pool["snapshot_hash"]),
    ):
        if pool_ref[key] != expected:
            raise CandidateLabProjectionError(
                "voting candidate parent pool binding drifted"
            )
    try:
        verify_voting_candidate_asset_against_pool(asset, parent_pool)
    except StrategyError as exc:
        raise CandidateLabProjectionError(
            "voting candidate no longer replays against its parent pool"
        ) from exc
    expected_provenance = voting_candidate_artifact_provenance(
        document,
        task_id=context.task_id,
        pool_artifact={
            "id": parent_record["id"],
            "content_hash": parent_record["content_hash"],
        },
    )
    if provenance != expected_provenance:
        raise CandidateLabProjectionError("voting candidate provenance drifted")
    fragment = voting_candidate_to_verified_fragment(
        asset,
        artifact_binding={
            "artifact_id": record["id"],
            "task_id": context.task_id,
            "kind": record["kind"],
            "content_hash": record["content_hash"],
            "origin_tool": record["origin_tool"],
            "artifact_schema_version": document["schema_version"],
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
        },
    )
    context.verified_cache[cache_key] = {
        "kind": "voting_candidate_fragment",
        "fragment": fragment,
    }
    return fragment


def _verified_pool_snapshot_record(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = _record_cache_key(record)
    cached = context.verified_cache.get(cache_key)
    if cached is not None:
        if not isinstance(cached, dict) or cached.get("kind") != "pool_snapshot":
            raise CandidateLabProjectionError("artifact verification cache drifted")
        return cached["pool"]
    raw = _read_candidate_record(
        context,
        record,
        kind=POOL_ARTIFACT_KIND,
        origin_tool=record["origin_tool"],
        directory_name="strategy_candidate_pools",
    )
    pool = validate_strategy_pool(_json_object(raw, "strategy pool"))
    _require_bytes_equal(
        canonical_strategy_pool_snapshot_json(pool).encode("utf-8"),
        raw,
    )
    expected_path = (
        Path(context.settings.tasks_dir)
        / context.task_id
        / "strategy_candidate_pools"
        / (
            f"{pool['pool_id']}_r{pool['revision']}_"
            f"{pool['snapshot_hash'][:12]}.json"
        )
    )
    _require_exact_path(record, expected_path)
    expected_origin = _POOL_ORIGIN_BY_OPERATION.get(pool["operation"]["kind"])
    if record["origin_tool"] != expected_origin:
        raise CandidateLabProjectionError("pool artifact origin drifted")
    entries = pool["entries"]
    if len(entries) > _MAX_POOL_ENTRIES:
        raise CandidateLabProjectionError("pool artifact entry cap exceeded")
    evidence_identity = (
        entries[0]["source"]["evidence_identity"] if entries else None
    )
    expected_provenance = {
        "schema_version": _POOL_ARTIFACT_SCHEMA_VERSION,
        "producer_version": POOL_PRODUCER_VERSION,
        "pool_id": pool["pool_id"],
        "strategy_type": pool["strategy_type"],
        "revision": pool["revision"],
        "revision_id": pool["revision_id"],
        "parent_revision_id": pool["parent_revision_id"],
        "snapshot_hash": pool["snapshot_hash"],
        "operation_kind": pool["operation"]["kind"],
        "source_artifact_ids": [
            entry["source"]["artifact_id"] for entry in entries
        ],
        "evidence_identity": evidence_identity,
    }
    if record["provenance"] != expected_provenance:
        raise CandidateLabProjectionError("pool artifact provenance drifted")
    context.verified_cache[cache_key] = {
        "kind": "pool_snapshot",
        "pool": pool,
    }
    return pool


def _pool_artifact_origin(
    context: _ProjectionContext,
    provenance: Mapping[str, Any],
) -> str:
    artifact_id = _text(provenance.get("pool_artifact_id"), "pool artifact id")
    record = context.artifact_repository.get_for_task(context.task_id, artifact_id)
    if record is None:
        raise CandidateLabProjectionError("voting parent pool is missing")
    origin = _text(record.get("origin_tool"), "pool artifact origin")
    if origin not in set(_POOL_ORIGIN_BY_OPERATION.values()):
        raise CandidateLabProjectionError("voting parent pool origin is unsupported")
    return origin


def _selection_binding(
    record: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    excluded = {"schema_version", "task_id", "kind", "format"}
    return {
        "artifact_id": record["id"],
        "task_id": record["task_id"],
        "kind": record["kind"],
        "content_hash": record["content_hash"],
        "origin_tool": record["origin_tool"],
        "artifact_schema_version": provenance["schema_version"],
        **{
            key: value
            for key, value in provenance.items()
            if key not in excluded
        },
    }


def _record_cache_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _text(record.get("id"), "artifact id"),
        _sha256(record.get("content_hash"), "artifact content hash"),
    )


def _cached_fragment(value: object, kind: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("kind") != kind
        or not isinstance(value.get("fragment"), dict)
    ):
        raise CandidateLabProjectionError("artifact verification cache drifted")
    return value["fragment"]


def _canonical_mapping_hash(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_candidate_record(
    context: _ProjectionContext,
    record: Mapping[str, Any],
    *,
    kind: str,
    origin_tool: str,
    directory_name: str,
) -> bytes:
    _require_record_identity(record, task_id=context.task_id)
    if record["kind"] != kind or record["origin_tool"] != origin_tool:
        raise CandidateLabProjectionError("artifact kind or origin drifted")
    path = Path(record["path"])
    expected_directory = (
        Path(context.settings.tasks_dir) / context.task_id / directory_name
    )
    if path.parent != expected_directory or path.suffix != ".json":
        raise CandidateLabProjectionError("artifact path is not canonical")
    raw = _read_record_bytes(context, record)
    actual_hash = hashlib.sha256(raw).hexdigest()
    expected_hash = _sha256(record["content_hash"], "artifact content_hash")
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise CandidateLabProjectionError("artifact physical bytes drifted")
    return raw


def _require_source_artifact(
    context: _ProjectionContext,
    *,
    artifact_id: object,
    content_hash: object,
    kind: object,
    origin_tool: object,
) -> Mapping[str, Any]:
    artifact_id_text = _text(artifact_id, "source artifact_id")
    record = context.artifact_repository.get_for_task(
        context.task_id,
        artifact_id_text,
    )
    if record is None:
        raise CandidateLabProjectionError("source artifact is not task-owned")
    _require_record_identity(record, task_id=context.task_id)
    expected_hash = _sha256(content_hash, "source artifact content_hash")
    if (
        record["kind"] != _text(kind, "source artifact kind")
        or record["origin_tool"] != _text(origin_tool, "source artifact origin")
        or not hmac.compare_digest(record["content_hash"], expected_hash)
    ):
        raise CandidateLabProjectionError("source artifact binding drifted")
    raw = _read_record_bytes(context, record)
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_hash):
        raise CandidateLabProjectionError("source artifact physical bytes drifted")
    return record


def _read_record_bytes(
    context: _ProjectionContext,
    record: Mapping[str, Any],
) -> bytes:
    artifact_id = _text(record.get("id"), "artifact id")
    content_hash = _sha256(record.get("content_hash"), "artifact content_hash")
    cache_key = (artifact_id, content_hash)
    cached = context.raw_cache.get(cache_key)
    if cached is not None:
        return cached
    raw = _read_regular_file(
        Path(_text(record.get("path"), "artifact path")),
        root=Path(context.settings.tasks_dir) / context.task_id,
        max_bytes=_MAX_ARTIFACT_BYTES,
        budget=context.budget,
    )
    context.raw_cache[cache_key] = raw
    return raw


def _require_record_identity(
    record: Mapping[str, Any],
    *,
    task_id: str,
) -> None:
    if record.get("task_id") != task_id:
        raise CandidateLabProjectionError("artifact task ownership drifted")
    kind = _text(record.get("kind"), "artifact kind")
    path = _text(record.get("path"), "artifact path")
    expected_id = _stable_artifact_id(task_id=task_id, kind=kind, path=path)
    if record.get("id") != expected_id:
        raise CandidateLabProjectionError("artifact stable identity drifted")
    _sha256(record.get("content_hash"), "artifact content_hash")
    _text(record.get("origin_tool"), "artifact origin")
    _mapping(record.get("provenance"), "artifact provenance")
    _text(record.get("created_at"), "artifact created_at")


def _read_regular_file(
    path: Path,
    *,
    root: Path,
    max_bytes: int,
    budget: _ProjectionBudget,
) -> bytes:
    if not path.is_absolute() or not root.is_absolute():
        raise CandidateLabProjectionError("artifact path must be absolute")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CandidateLabProjectionError("artifact path escapes task storage") from exc
    current = path
    while True:
        try:
            if current.is_symlink():
                raise CandidateLabProjectionError("artifact path uses a symlink")
        except OSError as exc:
            raise CandidateLabProjectionError("artifact path is unreadable") from exc
        if current == root:
            break
        if current == current.parent:
            raise CandidateLabProjectionError("artifact path escapes task storage")
        current = current.parent

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateLabProjectionError("artifact file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise CandidateLabProjectionError("artifact file is not bounded")
        budget.reserve(before.st_size)
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) > max_bytes
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or len(raw) != after.st_size
        ):
            raise CandidateLabProjectionError("artifact changed while reading")
        return raw
    finally:
        os.close(descriptor)


def _require_exact_path(record: Mapping[str, Any], expected: Path) -> None:
    if record["path"] != str(expected):
        raise CandidateLabProjectionError("artifact canonical path drifted")


def _artifact_projection(
    record: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    artifact_id = str(record["id"])
    return {
        "artifact_id": artifact_id,
        "created_at": str(record["created_at"]),
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        ),
    }


def _candidate_record_window(
    settings,
    task_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    origin_tool: str,
    directory_name: str,
    filename_pattern: re.Pattern[str],
) -> list[Mapping[str, Any]]:
    expected_directory = Path(settings.tasks_dir) / task_id / directory_name
    for record in records:
        _require_record_identity(record, task_id=task_id)
        path = Path(record["path"])
        if (
            record["kind"] != kind
            or record["origin_tool"] != origin_tool
            or path.parent != expected_directory
            or filename_pattern.fullmatch(path.name) is None
        ):
            raise CandidateLabProjectionError(
                "candidate artifact registry identity drifted"
            )
    ordered = sorted(
        records,
        key=lambda record: (record["created_at"], record["id"]),
        reverse=True,
    )
    return ordered[:_MAX_CANDIDATES_PER_KIND]


def _collection(
    items: Sequence[dict[str, Any]],
    limit: int,
    *,
    total: int | None = None,
) -> dict[str, Any]:
    ordered = sorted(
        items,
        key=lambda item: (
            item["artifact"]["created_at"],
            item["artifact"]["artifact_id"],
        ),
        reverse=True,
    )
    projected = ordered[:limit]
    actual_total = len(ordered) if total is None else total
    if actual_total < len(ordered):
        raise CandidateLabProjectionError("projection total is inconsistent")
    return {
        "latest": projected[0] if projected else None,
        "all": projected,
        "total": actual_total,
        "truncated": actual_total > len(projected),
    }


def _require_projection_payload_budget(payload: Mapping[str, Any]) -> None:
    try:
        byte_count = len(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise CandidateLabProjectionError(
            "strategy candidate lab projection is not canonical JSON"
        ) from exc
    if byte_count > _MAX_PROJECTION_BYTES:
        raise CandidateLabProjectionError(
            "strategy candidate lab response byte budget exceeded"
        )


def _active_plan_projection(settings, task_id: str) -> dict[str, Any] | None:
    return PlanRepository(settings.db_path).latest_nonterminal_summary_for_task(task_id)


def _open_gate_projection(settings, task_id: str) -> dict[str, Any] | None:
    last_assistant = TaskRepository(settings.db_path).get_latest_assistant_message(
        task_id
    )
    if last_assistant is None:
        return None
    metadata = last_assistant.get("metadata") or {}
    if metadata.get("error") or metadata.get("join_skip"):
        return None
    kind = metadata.get("kind")
    if kind not in ("gate", "plan_overview") and "join_c1" not in metadata:
        return None
    return {
        "message_id": last_assistant.get("id"),
        "kind": kind if isinstance(kind, str) else "join_c1",
        "step_id": metadata.get("step_id"),
    }


def _univariate_bin_metrics(bin_row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "count",
        "share",
        "good",
        "bad",
        "bad_rate",
        "woe",
        "iv_contribution",
        "lift",
        "cumulative_ks",
        "amount_metrics",
    )
    return {key: bin_row[key] for key in keys if key in bin_row}


def _cross_axis_provenance(axis: Mapping[str, Any]) -> dict[str, Any]:
    result = {"feature": axis["feature"], "method": axis["method"]}
    if "parent_evidence_hash" in axis or "manual_breakpoints" in axis:
        result.update(
            {
                "bin_count": len(axis["bins"]),
                "manual_breakpoints": axis.get("manual_breakpoints"),
                "parent_evidence_hash": axis.get("parent_evidence_hash"),
            }
        )
    return result


def _json_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CandidateLabProjectionError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CandidateLabProjectionError(f"{name} must contain an object")
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CandidateLabProjectionError("artifact JSON has duplicate keys")
        value[key] = item
    return value


def _stable_artifact_id(*, task_id: str, kind: str, path: str) -> str:
    identity_json = json.dumps(
        [task_id, kind, path],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        f"marvis.task_artifact.v1:{identity_json}".encode("utf-8")
    ).hexdigest()


def _bounded_list(value: object, limit: int) -> tuple[list[Any], bool]:
    items = list(_sequence(value, "projection list"))
    return items[:limit], len(items) > limit


def _sequence(value: object, name: str) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CandidateLabProjectionError(f"{name} must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise CandidateLabProjectionError(f"{name} entries must be objects")
    return value


def _text_sequence(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise CandidateLabProjectionError(f"{name} must be an array")
    return tuple(_text(item, f"{name} item") for item in value)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateLabProjectionError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise CandidateLabProjectionError(f"{name} must be non-empty text")
    return value.strip()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CandidateLabProjectionError(f"{name} must be a SHA-256 hash")
    return value


def _require_bytes_equal(left: bytes, right: bytes) -> None:
    if not hmac.compare_digest(left, right):
        raise CandidateLabProjectionError("artifact is not canonical JSON")


__all__ = [
    "CandidateLabProjectionError",
    "SCHEMA_VERSION",
    "build_strategy_candidate_lab_projection",
]
