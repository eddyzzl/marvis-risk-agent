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
from typing import Any
from urllib.parse import quote

from marvis.db import TaskRepository
from marvis.domain import STRATEGY_TYPES
from marvis.output.strategy_candidate_report import (
    canonical_strategy_candidate_report_json,
    strategy_candidate_report_from_json,
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
from marvis.packs.strategy.pool import (
    POOL_PRODUCER_VERSION,
    validate_strategy_pool,
)
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ORIGIN_TOOL,
    voting_candidate_to_verified_fragment,
)
from marvis.packs.strategy.voting_candidate_tools import (
    canonical_voting_candidate_artifact_json,
    canonical_voting_candidate_path,
    validate_voting_candidate_artifact_document,
    voting_candidate_artifact_provenance,
)
from marvis.repositories.plans import PlanRepository
from marvis.repositories.strategy_pool import (
    POOL_ARTIFACT_KIND,
    StrategyCandidatePoolRepository,
    canonical_strategy_pool_snapshot_json,
    strategy_pool_artifact_content_hash,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository


SCHEMA_VERSION = "strategy.candidate-lab-projection.v1"

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
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_PROJECTION_BYTES = 64 * 1024 * 1024
_MAX_CANDIDATES_PER_KIND = 20
_MAX_RANKINGS = 50
_MAX_METRICS = 100
_MAX_BIN_POINTERS = 200
_MAX_CELL_POINTERS = 400
_MAX_LEAF_POINTERS = 256
_MAX_POOL_ENTRIES = 200
_MAX_RISKS = 50
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
    automatic_tree = [
        _project_automatic_tree(context, record)
        for record in automatic_tree_records
    ]
    pools = _project_current_pools(
        context,
    )
    active_plan = _active_plan_projection(settings, task_id)
    open_gate = _open_gate_projection(settings, task_id)
    if active_plan is not None:
        blocked_reason = "active_plan"
    elif open_gate is not None:
        blocked_reason = "open_gate"
    else:
        blocked_reason = None

    return {
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
            "automatic_tree": _collection(
                automatic_tree,
                _MAX_CANDIDATES_PER_KIND,
                total=automatic_tree_total,
            ),
        },
        "pools": _collection(pools, len(STRATEGY_TYPES)),
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
    return {
        "kind": "automatic_tree",
        "artifact": _artifact_projection(record, context.task_id),
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "lifecycle": asset["lifecycle"],
        "detail": {
            "asset_id": asset["asset_id"],
            "tree_id": tree_info["tree_id"],
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
        "pointers": {"leaves": projected_leaves},
        "total": len(fragments),
        "truncated": len(fragments) > len(projected_leaves),
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
        for entry in entries:
            source = entry["source"]
            fragment = _verified_pool_source_fragment(context, source)
            replayed_source, replayed_rule_id, replayed_execution = (
                verified_fragment_pool_parts(fragment)
            )
            if (
                source != replayed_source
                or entry["rule_id"] != replayed_rule_id
                or entry["execution"] != replayed_execution
            ):
                raise CandidateLabProjectionError(
                    "pool entry does not match replayed source fragment"
                )
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
                "execution": entry["execution"],
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
        VOTING_CANDIDATE_ARTIFACT_KIND,
        VOTING_CANDIDATE_ORIGIN_TOOL,
    ):
        return _verified_voting_candidate_fragment(context, record)
    raise CandidateLabProjectionError("pool source artifact contract is unsupported")


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
