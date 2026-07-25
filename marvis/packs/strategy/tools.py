from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import quote
import uuid

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.direction import check_score_direction, normalize_score_direction
from marvis.data.errors import DatasetContentDriftError, LabelSemanticsNotDeclaredError
from marvis.data.labels import require_labels_confirmed, resolve_labeled_frame
from marvis.data.workspace import (
    DataSemanticMapping,
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.feature.univariate import (
    MANUAL_SCHEMA_VERSION as UNIVARIATE_MANUAL_ANALYSIS_SCHEMA_VERSION,
    SCHEMA_VERSION as UNIVARIATE_ANALYSIS_SCHEMA_VERSION,
    analyze_univariate,
)
from marvis.files import sha256_file
from marvis.packs.strategy.backtest_compat import (
    BacktestRecord,
    approval_backtest_projection,
    backtest_record_payload,
)
from marvis.packs.strategy.bands import design_cutoff_bands
from marvis.packs.strategy.candidate_design import (
    CANDIDATE_POLICY_VERSION,
    design_strategy_candidate,
    normalize_candidate_design,
    normalize_candidate_economics_inputs,
)
from marvis.packs.strategy.candidate_evidence import (
    MetricObservation,
    build_candidate_evidence,
)
from marvis.packs.strategy.candidate_asset_tools import (
    run_refine_univariate_candidate,
)
from marvis.packs.strategy.automatic_tree_tools import (
    run_build_automatic_tree_candidate,
)
from marvis.packs.strategy.automatic_tree_apply_tools import (
    run_apply_automatic_tree,
)
from marvis.packs.strategy.automatic_tree_leaf_tools import (
    run_materialize_automatic_tree_leaf_fragment,
)
from marvis.packs.strategy.voting_candidate_tools import (
    run_build_voting_candidate,
)
from marvis.packs.strategy.cross_matrix_candidate_tools import (
    run_build_cross_matrix_candidate,
)
from marvis.packs.strategy.cross_matrix_cell_selection_tools import (
    run_materialize_cross_matrix_cell_selection,
)
from marvis.packs.strategy.scorecard_candidate_tools import (
    run_build_scorecard_band_asset,
    run_materialize_scorecard_cutoff_selection,
)
from marvis.packs.strategy.pool_tools import (
    run_add_candidate_to_pool,
    run_compile_strategy_pool,
    run_remove_pool_entry,
    run_reorder_strategy_pool,
    run_set_pool_entry_action,
)
from marvis.packs.strategy.pool_impact_tools import run_measure_pool_impact
from marvis.packs.strategy.candidate_stability_tools import (
    run_measure_candidate_monthly_stability,
)
from marvis.packs.strategy.pool_validation_tools import (
    run_measure_strategy_pool_validation,
)
from marvis.packs.strategy.impact_cube_tools import (
    run_measure_strategy_impact_cube,
)
from marvis.packs.strategy.dsl_delivery_tools import (
    run_export_strategy_delivery,
)
from marvis.packs.strategy.project_context_tools import run_materialize_project_context
from marvis.packs.strategy.report_bundle_tools import (
    run_build_strategy_report_bundle_v2,
)
from marvis.packs.strategy.model_evidence_tools import (
    run_materialize_model_evidence_v2,
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
    run_materialize_sample_design,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    run_materialize_sample_design_v2,
)
from marvis.packs.strategy.compare import compare_strategies
from marvis.packs.strategy.contracts import Strategy
from marvis.packs.strategy.deliverables import decision_table_csv
from marvis.packs.strategy.dsl import (
    canonical_strategy_json,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.doc import render_strategy_doc_markdown
from marvis.packs.strategy.monitor_tools import (  # noqa: F401
    tool_render_monitoring_report,
    tool_run_strategy_monitoring,
)
from marvis.packs.strategy.monitoring_plan import (
    DEFAULT_CADENCE_DAYS,
    MonitoringPlan,
    PLAN_VERSION,
    build_monitoring_plan,
    canonical_monitoring_plan_hash,
    monitoring_plan_from_dict,
    save_monitoring_plan,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pricing import (
    LimitPricingResult,
    PricingParams,
    limit_pricing_matrix,
)
from marvis.packs.strategy.profit import ProfitParams, profit_calc
from marvis.packs.strategy.roll_rate import roll_rate_matrix
from marvis.packs.strategy.rules import (
    DEFAULT_MINE_SEED,
    evaluate_rule_set,
    mine_rules,
)
from marvis.packs.strategy.legacy_adapter import legacy_strategy_to_spec
from marvis.packs.strategy.strategy import (
    build_strategy,
    build_strategy_from_spec,
    infer_strategy_rule_direction,
)
from marvis.packs.strategy.tradeoff import (
    recommend_operating_point,
    tradeoff_feasible_flags,
    tradeoff_view,
)
from marvis.packs.strategy.typed_backtest import (
    ApprovalProfitInputs,
    StrategyBacktestResult,
    run_typed_backtest,
)
from marvis.packs.strategy.vintage import vintage_curve, vintage_summary
from marvis.output.strategy_candidate_report import render_strategy_candidate_bundle
from marvis.plugins.sdk import PackRuntime
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.strategy_handoff import StrategyHandoffRepository
from marvis.repositories.strategy_monitoring import (
    StrategyMonitoringDataError,
    StrategyMonitoringRepository,
    validate_monitoring_run_result,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.automatic_tree_apply import AutomaticTreeApplyRepository
from marvis.repositories.data_workspace import (
    DataWorkspaceDataError,
    DataWorkspaceRepository,
)
from marvis.strategy_adoption import AdoptionReasonError, normalize_adoption_reason
from marvis.strategy_lifecycle import is_locally_adopted
from marvis.validation.vintage import compute_vintage_curve


_TASK_ARTIFACT_PROVENANCE_SCHEMA_VERSION = "task-artifact-provenance.v1"
# These versions describe the producer algorithm/output contract, not the DB
# schema. Any material calculation or rendered-output change must bump the
# corresponding value so a new immutable artifact identity is produced.
_TASK_ANALYSIS_PRODUCER_VERSIONS = {
    "profit": "strategy.profit_calc.v1",
    "roll_rate": "strategy.roll_rate_matrix.v1",
    "limit_pricing": "strategy.limit_pricing_matrix.v2",
}
_STRATEGY_ARTIFACT_PROVENANCE_SCHEMA_VERSION = "strategy-artifact-provenance.v1"
_STRATEGY_ADOPTION_ARTIFACT_PRODUCER_VERSIONS = {
    "decision_table_csv": "strategy.adopt.decision_table.v1",
    "monitoring_plan_json": "strategy.adopt.monitoring_plan.v1",
}
_STRATEGY_REPORT_PRODUCER_VERSIONS = {
    "strategy_doc_md": "strategy.render_strategy_doc.v2",
    "challenger_report_md": "strategy.render_challenger_report.v2",
    "monitoring_report_md": "strategy.render_monitoring_report.v2",
}
_CANDIDATE_TOOL_SCHEMA_VERSION = "strategy.candidate_tool.v1"
_CANDIDATE_TOOL_INPUT_FIELDS = frozenset(
    {
        "dataset_id",
        "target_col",
        "sample_design_ref",
        "drop_nan_labels",
        "strategy_type",
        "candidate_design",
        "economics_inputs",
        "candidate_policy_version",
    }
)
_CALLER_RESULT_FIELDS = frozenset(
    {
        "action",
        "actions",
        "default_action",
        "design_evidence",
        "metrics",
        "recommendation",
        "recommended",
        "recommended_value",
        "rules",
        "selected_action",
        "strategy_effect_hash",
        "strategy_spec",
    }
)
_UNIVARIATE_TOOL_INPUT_FIELDS = frozenset(
    {
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
        "drop_nan_labels",
        "features",
        "methods",
        "manual_breakpoints",
        "bin_count",
        "min_bin_pct",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
    }
)
_UNIVARIATE_AUTO_EXCLUDED_ROLES = frozenset(
    {
        "target",
        "id",
        "phone",
        "idcard",
        "name",
        "date",
        "month",
        "weight",
        "loan_amount",
        "overdue_amount",
        "ignore",
    }
)
_UNIVARIATE_FORBIDDEN_EXPLICIT_ROLES = frozenset(
    {"id", "phone", "idcard", "name", "ignore"}
)
_UNIVARIATE_NUMERIC_ROLES = frozenset(
    {"numeric", "score", "amount", "loan_amount", "overdue_amount", "weight"}
)
_UNIVARIATE_CATEGORICAL_ROLES = frozenset({"categorical", "segment", "rule_node"})
_UNIVARIATE_CANDIDATE_TOOL_SCHEMA_VERSION = "strategy.univariate-candidate-tool.v1"
_UNIVARIATE_CANDIDATE_PRODUCER_VERSION = "strategy.univariate-candidate/1"
_UNIVARIATE_CANDIDATE_ARTIFACT_SCHEMA_VERSION = (
    "strategy.univariate-candidate-artifact.v1"
)
_UNIVARIATE_CANDIDATE_V2_TOOL_SCHEMA_VERSION = "strategy.univariate-candidate-tool.v2"
_UNIVARIATE_CANDIDATE_V2_PRODUCER_VERSION = "strategy.univariate-candidate/2"
_UNIVARIATE_CANDIDATE_V2_ARTIFACT_SCHEMA_VERSION = (
    "strategy.univariate-candidate-artifact.v2"
)
_UNIVARIATE_MAX_ROWS = 1_000_000
_UNIVARIATE_MAX_FEATURES = 50
_UNIVARIATE_MAX_BINS = 20
_UNIVARIATE_MAX_CATEGORIES = 100
_UNIVARIATE_MAX_EVALUATED_CELLS = 50_000_000


def tool_vintage_curve(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    cohort_col = str(inputs["cohort_col"])
    mob_col = str(inputs["mob_col"])
    bad_col = str(inputs["bad_col"])
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        task_id=str(ctx.task_id),
        columns=[cohort_col, mob_col, bad_col],
    )
    # NaN-label gate runs FIRST (an unusable label is a harder problem than an
    # undeclared cumulation basis); label_semantics is checked on the resolved frame.
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        bad_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    # A1: the strategy path must NOT guess the cumulation basis. The kernel always
    # accumulates the target across MOBs; on a snapshot/ever-bad flag that double-counts
    # silently. When the caller has not declared label_semantics, stop at a gate and hand
    # the two concrete semantics to the user (mirrors the NaN-label gate).
    label_semantics = _optional_str(inputs.get("label_semantics"))
    if label_semantics is None:
        raise LabelSemanticsNotDeclaredError(
            target_col=bad_col,
            n_cohorts=_vintage_cohort_count(frame, cohort_col),
            monotone_heuristic=_vintage_looks_like_snapshot(
                frame, cohort_col, mob_col, bad_col
            ),
        )
    curve = vintage_curve(
        frame,
        cohort_col=cohort_col,
        mob_col=mob_col,
        bad_col=bad_col,
        mob_max=int(inputs.get("mob_max", 12)),
        label_semantics=label_semantics,
    )
    return {
        "cohorts": list(curve.cohorts),
        "mob_axis": list(curve.mob_axis),
        "curves": _jsonable(curve.curves),
        "counts": _jsonable(curve.counts),
        "summary": vintage_summary(curve, ref_mob=int(inputs.get("ref_mob", 6))),
        "nan_labels_dropped": nan_labels_dropped,
        "warnings": list(curve.warnings),
    }


def _vintage_cohort_count(frame, cohort_col: str) -> int:
    try:
        return int(frame[cohort_col].nunique(dropna=True))
    except Exception:
        return 0


def _vintage_looks_like_snapshot(
    frame, cohort_col: str, mob_col: str, bad_col: str
) -> bool:
    """Reuse the kernel's own conservative snapshot heuristic (single source of truth):
    the incremental path attaches a snapshot red flag exactly when the data looks
    cumulative. If any point carries it, the data looks snapshot-shaped."""
    try:
        points = compute_vintage_curve(
            frame,
            cohort_col=cohort_col,
            mob_col=mob_col,
            target_col=bad_col,
            label_semantics="incremental",
        )
    except Exception:
        return False
    return any(
        "snapshot" in warning.lower() or "快照" in warning
        for point in points
        for warning in point.data_quality_warnings
    )


def tool_roll_rate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    observation_semantics = str(
        inputs.get("observation_semantics") or "adjacent_observation"
    )
    if observation_semantics != "adjacent_observation":
        raise StrategyError(
            "roll_rate_matrix only supports observation_semantics="
            "'adjacent_observation'; use bucket_migration for month-end snapshots"
        )
    balance_col = _optional_str(inputs.get("balance_col"))
    columns = _unique(
        [
            str(inputs["id_col"]),
            str(inputs["time_col"]),
            str(inputs["status_col"]),
            balance_col,
        ]
    )
    frame, source_evidence, source_path = _task_dataset_frame_with_evidence(
        runtime,
        dataset_id,
        task_id=task_id,
        columns=columns,
    )
    matrix = roll_rate_matrix(
        frame,
        id_col=str(inputs["id_col"]),
        time_col=str(inputs["time_col"]),
        status_col=str(inputs["status_col"]),
        states=[str(item) for item in inputs["states"]],
        balance_col=balance_col,
    )
    warnings = [dict(warning) for warning in matrix.data_quality_warnings]
    assumptions = {
        "dataset_id": dataset_id,
        "id_col": str(inputs["id_col"]),
        "time_col": str(inputs["time_col"]),
        "status_col": str(inputs["status_col"]),
        "states": list(matrix.states),
        "balance_col": balance_col,
        "period": matrix.period,
        "observation_semantics": observation_semantics,
    }
    csv_text = _roll_rate_csv(matrix)
    markdown_text = _roll_rate_markdown(
        matrix=matrix,
        assumptions=assumptions,
        source_evidence=source_evidence,
        warnings=warnings,
    )
    _assert_source_unchanged(source_path, str(source_evidence["dataset_content_hash"]))
    artifacts = _write_task_analysis_artifacts(
        runtime,
        task_id=task_id,
        analysis_kind="roll_rate",
        source_hash=str(source_evidence["dataset_content_hash"]),
        assumptions=assumptions,
        files=(
            ("roll_rate_csv", "csv", csv_text),
            ("roll_rate_markdown", "md", markdown_text),
        ),
    )
    return {
        "states": list(matrix.states),
        "matrix": [list(row) for row in matrix.matrix],
        "base_counts": dict(matrix.base_counts),
        "period": matrix.period,
        "observation_semantics": observation_semantics,
        "assumptions": assumptions,
        "source_evidence": source_evidence,
        "data_quality_warnings": warnings,
        "artifacts": artifacts,
    }


def tool_profit_calc(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    segment_col = _optional_str(inputs.get("segment_col"))
    columns = _unique([segment_col, str(inputs["ead_col"]), str(inputs["pd_col"])])
    frame, source_evidence, source_path = _task_dataset_frame_with_evidence(
        runtime,
        dataset_id,
        task_id=task_id,
        columns=columns,
    )
    params = _profit_params(inputs["params"])
    results = profit_calc(
        frame,
        segment_col=segment_col,
        ead_col=str(inputs["ead_col"]),
        pd_col=str(inputs["pd_col"]),
        params=params,
    )
    result_rows = [_jsonable(result) for result in results]
    warnings = _profit_quality_warnings(
        frame,
        segment_col=segment_col,
        ead_col=str(inputs["ead_col"]),
    )
    assumptions = {
        "dataset_id": dataset_id,
        "segment_col": segment_col,
        "ead_col": str(inputs["ead_col"]),
        "pd_col": str(inputs["pd_col"]),
        "params": _jsonable(params),
        "formula": (
            "net_profit = EAD*annual_rate*term/12 - EAD*PD*LGD "
            "- EAD*funding_rate*term/12 - operating_cost_per_loan"
        ),
    }
    csv_text = pd.DataFrame(result_rows).to_csv(index=False)
    markdown_text = _profit_markdown(
        result_rows=result_rows,
        assumptions=assumptions,
        source_evidence=source_evidence,
        warnings=warnings,
    )
    _assert_source_unchanged(source_path, str(source_evidence["dataset_content_hash"]))
    artifacts = _write_task_analysis_artifacts(
        runtime,
        task_id=task_id,
        analysis_kind="profit",
        source_hash=str(source_evidence["dataset_content_hash"]),
        assumptions=assumptions,
        files=(
            ("profit_csv", "csv", csv_text),
            ("profit_markdown", "md", markdown_text),
        ),
    )
    return {
        "results": result_rows,
        "assumptions": assumptions,
        "source_evidence": source_evidence,
        "quality_warnings": warnings,
        "artifacts": artifacts,
    }


@dataclass(frozen=True)
class _UnivariateWorkspaceBinding:
    persisted: bool
    revision: int
    generation: int
    active_dataset_id: str | None
    active_dataset_content_hash: str | None
    semantic_mapping: DataSemanticMapping
    semantic_mapping_hash: str


@dataclass(frozen=True)
class _UnivariateDatasetBinding:
    dataset: Any
    path: Path
    content_hash: str
    registry_metadata_hash: str
    workspace: _UnivariateWorkspaceBinding


def tool_analyze_univariate_candidates(inputs: dict, ctx) -> dict:
    """Generate task-owned, development-only single-variable candidate evidence."""

    unexpected = sorted(set(inputs) - _UNIVARIATE_TOOL_INPUT_FIELDS)
    caller_results = sorted(set(inputs) & _CALLER_RESULT_FIELDS)
    if caller_results:
        raise StrategyError(
            "caller cannot supply univariate candidate results: "
            + ", ".join(caller_results)
        )
    if unexpected:
        raise StrategyError(
            "unsupported analyze_univariate_candidates inputs: " + ", ".join(unexpected)
        )
    if "sample_design_ref" not in inputs:
        raise StrategyError("sample_design_ref is required")

    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    binding = _univariate_dataset_binding(
        runtime,
        task_id=task_id,
        dataset_id=dataset_id,
    )
    _require_expected_univariate_binding(inputs, binding)
    methods = _univariate_methods(inputs.get("methods"))
    requested_sentinels = _normalize_univariate_sentinel_values(
        inputs.get("sentinel_values")
    )
    dataset_columns = [str(profile.name) for profile in binding.dataset.columns]
    if target_col not in dataset_columns:
        raise StrategyError(f"unknown target column: {target_col}")
    resolved_roles = _univariate_field_roles(
        binding.dataset,
        binding.workspace.semantic_mapping,
        target_col=target_col,
    )
    loan_amount_col = _resolve_univariate_amount_column(
        inputs.get("loan_amount_col"),
        role="loan_amount",
        columns=dataset_columns,
    )
    overdue_amount_col = _resolve_univariate_amount_column(
        inputs.get("overdue_amount_col"),
        role="overdue_amount",
        columns=dataset_columns,
    )
    if target_col in {loan_amount_col, overdue_amount_col}:
        raise StrategyError("amount columns cannot use the target column")
    if loan_amount_col is not None and loan_amount_col == overdue_amount_col:
        raise StrategyError(
            "loan_amount_col and overdue_amount_col must be different columns"
        )
    sample_binding = load_strategy_sample_design_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=inputs["sample_design_ref"],
        dataset_id=dataset_id,
        dataset_content_hash=binding.content_hash,
        workspace_revision=binding.workspace.revision,
        workspace_generation=binding.workspace.generation,
        semantic_mapping_hash=binding.workspace.semantic_mapping_hash,
        target_col=target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
    )
    # The authenticated sample design owns optional analysis fields. An omitted
    # caller value inherits the frozen design; semantic-role inference must not
    # silently introduce an amount field outside that evidence boundary.
    loan_amount_col = loan_amount_col or sample_binding.loan_amount_col
    overdue_amount_col = overdue_amount_col or sample_binding.overdue_amount_col
    if target_col in {loan_amount_col, overdue_amount_col}:
        raise StrategyError("amount columns cannot use the target column")
    if loan_amount_col is not None and loan_amount_col == overdue_amount_col:
        raise StrategyError(
            "loan_amount_col and overdue_amount_col must be different columns"
        )
    _preflight_univariate_work_budget(
        inputs,
        binding=binding,
        target_col=target_col,
        methods=methods,
        sentinel_count=len(requested_sentinels),
        row_count=sample_binding.development_population_count,
        sample_split_col=sample_binding.split_column,
    )
    features = _resolve_univariate_features(
        inputs.get("features"),
        columns=dataset_columns,
        target_col=target_col,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
        sample_split_col=sample_binding.split_column,
        field_roles=resolved_roles,
    )
    required_columns = {
        target_col,
        *features,
        *(
            column
            for column in (
                loan_amount_col,
                overdue_amount_col,
                sample_binding.split_column,
            )
            if column is not None
        ),
    }
    projected_columns = [
        column for column in dataset_columns if column in required_columns
    ]
    frame = runtime.backend.read_frame(
        binding.path,
        columns=projected_columns,
    )
    if sha256_file(binding.path) != binding.content_hash:
        raise StrategyError(
            "source dataset changed while univariate analysis was loading"
        )
    frame = bind_strategy_development_frame(frame, binding=sample_binding)
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    feature_types = _univariate_feature_types(
        features,
        frame=frame,
        field_roles=resolved_roles,
    )
    manual_breakpoints = _univariate_manual_breakpoints(
        inputs.get("manual_breakpoints"),
        features=features,
        feature_types=feature_types,
        methods=methods,
    )
    sentinel_mapping, sentinel_red_flags = _univariate_sentinel_mapping(
        requested_sentinels,
        features=features,
        frame=frame,
        feature_types=feature_types,
    )
    estimated_evaluated_cells = _univariate_estimated_evaluated_cells(
        frame,
        features=features,
        feature_types=feature_types,
        methods=methods,
        bin_count=int(inputs.get("bin_count", 10)),
        manual_breakpoints=manual_breakpoints,
        sentinel_mapping=sentinel_mapping,
        sentinel_value_count=len(requested_sentinels),
    )
    if estimated_evaluated_cells > _UNIVARIATE_MAX_EVALUATED_CELLS:
        raise StrategyError(
            "univariate candidate analysis exceeds the combined row/bin work "
            f"budget ({estimated_evaluated_cells} > "
            f"{_UNIVARIATE_MAX_EVALUATED_CELLS}); select fewer features or bins"
        )
    seed = int(ctx.seed or 0)
    analysis = analyze_univariate(
        frame,
        features=features,
        target=target_col,
        methods=(None if not methods else methods),
        feature_types=feature_types,
        manual_breakpoints=manual_breakpoints,
        bin_count=int(inputs.get("bin_count", 10)),
        sentinel_values=sentinel_mapping,
        loan_amount=loan_amount_col,
        overdue_amount=overdue_amount_col,
        max_rows=_UNIVARIATE_MAX_ROWS,
        max_features=_UNIVARIATE_MAX_FEATURES,
        max_bins=_UNIVARIATE_MAX_BINS,
        max_categories=_UNIVARIATE_MAX_CATEGORIES,
        min_bin_pct=float(inputs.get("min_bin_pct", 0.02)),
        seed=seed,
    )
    available_method_count = sum(
        method["status"] == "available"
        for feature in analysis["features"]
        for method in feature["methods"]
    )
    if available_method_count == 0:
        raise StrategyError(
            "univariate analysis produced no available candidate method; "
            "review feature types, values, and requested methods"
        )
    red_flags = _univariate_red_flags(
        analysis,
        initial=sentinel_red_flags,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
    )
    version_contract = _univariate_candidate_version_contract(
        analysis["schema_version"]
    )
    generation_parameters = {
        "analysis_schema_version": analysis["schema_version"],
        "target_col": target_col,
        "drop_nan_labels": bool(inputs.get("drop_nan_labels")),
        "nan_labels_dropped": int(nan_labels_dropped),
        "features": list(features),
        "feature_types": dict(feature_types),
        "methods": list(methods),
        "method_mode": "type_aware_auto" if not methods else "explicit",
        "bin_count": int(inputs.get("bin_count", 10)),
        "min_bin_pct": float(inputs.get("min_bin_pct", 0.02)),
        "loan_amount_col": loan_amount_col,
        "overdue_amount_col": overdue_amount_col,
        "sentinel_values": {
            feature: list(values) for feature, values in sentinel_mapping.items()
        },
        "registry_metadata_hash": binding.registry_metadata_hash,
        "estimated_evaluated_cells": int(estimated_evaluated_cells),
        "budget_unit": "row_bin_evaluations",
        "sample_design_ref": sample_binding.to_ref_dict(),
        **(
            {
                "manual_breakpoints": {
                    feature: list(points)
                    for feature, points in manual_breakpoints.items()
                }
            }
            if manual_breakpoints
            else {}
        ),
    }
    candidate_evidence = build_candidate_evidence(
        task_id=task_id,
        dataset_id=dataset_id,
        dataset_content_hash=binding.content_hash,
        workspace_revision=binding.workspace.revision,
        workspace_generation=binding.workspace.generation,
        semantic_mapping_hash=binding.workspace.semantic_mapping_hash,
        generation_parameters=generation_parameters,
        seed=seed,
        budget=_UNIVARIATE_MAX_EVALUATED_CELLS,
        truncated=bool(analysis["resource_budget"]["truncated"]),
        analysis=analysis,
        metrics=_univariate_candidate_metrics(analysis),
        source_refs=(
            f"dataset:{dataset_id}@sha256:{binding.content_hash}",
            (
                f"data-workspace:{task_id}@revision:{binding.workspace.revision}"
                f":generation:{binding.workspace.generation}"
            ),
            sample_binding.source_ref_token,
        ),
        red_flags=red_flags,
        producer_version=version_contract["producer_version"],
    )
    bundle = render_strategy_candidate_bundle(candidate_evidence, analysis)
    _assert_source_unchanged(binding.path, binding.content_hash)
    artifacts = _write_univariate_candidate_artifacts(
        runtime,
        task_id=task_id,
        binding=binding,
        sample_design_binding=sample_binding,
        candidate_evidence=candidate_evidence,
        generation_parameters=generation_parameters,
        bundle=bundle,
    )
    return {
        "schema_version": version_contract["tool_schema_version"],
        "candidate_id": candidate_evidence["candidate_id"],
        "evidence_hash": candidate_evidence["evidence_hash"],
        "validation_status": candidate_evidence["validation_status"],
        "dataset_id": dataset_id,
        "dataset_content_hash": binding.content_hash,
        "workspace_revision": binding.workspace.revision,
        "workspace_generation": binding.workspace.generation,
        "semantic_mapping_hash": binding.workspace.semantic_mapping_hash,
        "target_col": target_col,
        "nan_labels_dropped": int(nan_labels_dropped),
        "feature_count": len(features),
        "available_method_count": int(available_method_count),
        "rankings": list(analysis["rankings"]),
        "red_flags": list(candidate_evidence["red_flags"]),
        "candidate_evidence": candidate_evidence,
        "artifacts": artifacts,
    }


def tool_refine_univariate_candidate(inputs: dict, ctx) -> dict:
    """Refine task-owned evidence into an immutable development candidate asset."""

    return run_refine_univariate_candidate(inputs, ctx, _runtime(ctx))


def tool_build_automatic_tree_candidate(inputs: dict, ctx) -> dict:
    """Build one complete governed automatic weighted rule-tree candidate."""

    return run_build_automatic_tree_candidate(inputs, ctx, _runtime(ctx))


def tool_apply_automatic_tree(inputs: dict, ctx) -> dict:
    """Apply one canonical automatic tree to its original governed dataset."""

    return run_apply_automatic_tree(inputs, ctx, _runtime(ctx))


def tool_materialize_automatic_tree_leaf_fragment(inputs: dict, ctx) -> dict:
    """Persist one explicit pointer to a verified automatic-tree leaf."""

    return run_materialize_automatic_tree_leaf_fragment(inputs, ctx, _runtime(ctx))


def tool_build_voting_candidate(inputs: dict, ctx) -> dict:
    """Build one immutable n-of-k candidate from an exact Pool revision."""

    return run_build_voting_candidate(inputs, ctx, _runtime(ctx))


def tool_build_cross_matrix_candidate(inputs: dict, ctx) -> dict:
    """Build one complete immutable two-dimensional Cross Matrix candidate."""

    return run_build_cross_matrix_candidate(inputs, ctx, _runtime(ctx))


def tool_materialize_cross_matrix_cell_selection(inputs: dict, ctx) -> dict:
    """Persist one explicit pointer to a verified Cross Matrix cell group."""

    return run_materialize_cross_matrix_cell_selection(inputs, ctx, _runtime(ctx))


def tool_build_scorecard_band_asset(inputs: dict, ctx) -> dict:
    """Build one complete immutable scorecard band asset."""

    return run_build_scorecard_band_asset(inputs, ctx, _runtime(ctx))


def tool_materialize_scorecard_cutoff_selection(inputs: dict, ctx) -> dict:
    """Persist one explicit pointer to a verified scorecard cutoff."""

    return run_materialize_scorecard_cutoff_selection(
        inputs,
        ctx,
        _runtime(ctx),
    )


def tool_add_candidate_to_pool(inputs: dict, ctx) -> dict:
    """Add one verified candidate source.

    Supports a univariate asset, automatic-tree leaf, Cross Matrix cell selection,
    or Voting candidate.
    """

    return run_add_candidate_to_pool(inputs, ctx, _runtime(ctx))


def tool_remove_pool_entry(inputs: dict, ctx) -> dict:
    """Remove one pool entry addressed by its external stable rule id."""

    return run_remove_pool_entry(inputs, ctx, _runtime(ctx))


def tool_set_pool_entry_action(inputs: dict, ctx) -> dict:
    """Set the Pool-owned typed action for one stable rule id."""

    return run_set_pool_entry_action(inputs, ctx, _runtime(ctx))


def tool_reorder_strategy_pool(inputs: dict, ctx) -> dict:
    """Persist one complete rule-id ordering as a new draft pool revision."""

    return run_reorder_strategy_pool(inputs, ctx, _runtime(ctx))


def tool_compile_strategy_pool(inputs: dict, ctx) -> dict:
    """Compile an exact pool revision to a canonical, unexecuted design."""

    return run_compile_strategy_pool(inputs, ctx, _runtime(ctx))


def tool_measure_pool_impact(inputs: dict, ctx) -> dict:
    """Measure governed first-match and monthly impact for the current Pool."""

    return run_measure_pool_impact(inputs, ctx, _runtime(ctx))


def tool_measure_candidate_monthly_stability(inputs: dict, ctx) -> dict:
    """Publish governed monthly hit-distribution stability for one candidate."""

    return run_measure_candidate_monthly_stability(inputs, ctx, _runtime(ctx))


def tool_measure_strategy_pool_validation(inputs: dict, ctx) -> dict:
    """Replay the exact current Pool on governed validation or OOT rows."""

    return run_measure_strategy_pool_validation(inputs, ctx, _runtime(ctx))


def tool_measure_strategy_impact_cube(inputs: dict, ctx) -> dict:
    """Publish unified deterministic impact slices for an exact current Pool."""

    return run_measure_strategy_impact_cube(inputs, ctx, _runtime(ctx))


def tool_export_strategy_delivery(inputs: dict, ctx) -> dict:
    """Publish offline Strategy DSL code plus bounded equivalence evidence."""

    return run_export_strategy_delivery(inputs, ctx, _runtime(ctx))


def tool_build_report_bundle_v2(inputs: dict, ctx) -> dict:
    """Build and publish the governed StrategyReportBundle projections."""

    return run_build_strategy_report_bundle_v2(inputs, ctx, _runtime(ctx))


def tool_materialize_project_context(inputs: dict, ctx) -> dict:
    """Refresh governed current-project, history and missing-information evidence."""

    # Template slots intentionally omit ``None``. Restore the three nullable
    # values here so the governed Tool receives its exact closed contract.
    normalized = {
        **inputs,
        "expected_revision_id": inputs.get("expected_revision_id"),
        "expected_state_hash": inputs.get("expected_state_hash"),
        "scope": inputs.get("scope"),
    }
    return run_materialize_project_context(normalized, ctx, _runtime(ctx))


def tool_materialize_sample_design(inputs: dict, ctx) -> dict:
    """Freeze the exact active strategy sample boundary as immutable evidence."""

    return run_materialize_sample_design(inputs, ctx, _runtime(ctx))


def tool_materialize_sample_design_v2(inputs: dict, ctx) -> dict:
    """Freeze governed dual-population V2 sample evidence."""

    return run_materialize_sample_design_v2(inputs, ctx, _runtime(ctx))


def tool_materialize_model_evidence_v2(inputs: dict, ctx) -> dict:
    """Materialize governed V2 analysis evidence from authenticated sources."""

    return run_materialize_model_evidence_v2(inputs, ctx, _runtime(ctx))


def tool_design_strategy_candidate(inputs: dict, ctx) -> dict:
    """Design one non-approval candidate from task-owned evidence only."""

    unexpected = sorted(set(inputs) - _CANDIDATE_TOOL_INPUT_FIELDS)
    caller_results = sorted(set(inputs) & _CALLER_RESULT_FIELDS)
    if caller_results:
        raise StrategyError(
            "caller cannot supply candidate results: " + ", ".join(caller_results)
        )
    if unexpected:
        raise StrategyError(
            "unsupported design_strategy_candidate inputs: " + ", ".join(unexpected)
        )
    if "sample_design_ref" not in inputs:
        raise StrategyError("sample_design_ref is required")

    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    strategy_type = str(inputs["strategy_type"])
    policy_version = str(inputs["candidate_policy_version"])
    if policy_version != CANDIDATE_POLICY_VERSION:
        raise StrategyError(
            f"candidate_policy_version must be {CANDIDATE_POLICY_VERSION}"
        )

    candidate_design = normalize_candidate_design(
        strategy_type,
        inputs.get("candidate_design"),
    )
    economics_inputs = normalize_candidate_economics_inputs(
        strategy_type,
        inputs.get("economics_inputs"),
    )
    design_col = candidate_design[
        "feature_col" if strategy_type == "segmentation" else "score_col"
    ]
    economic_columns = [
        str(value)
        for key, value in (economics_inputs or {}).items()
        if key.endswith("_col")
    ]
    columns = _unique([target_col, str(design_col), *economic_columns])
    frame, source_evidence, source_path, sample_binding = (
        _strategy_development_frame_with_evidence(
            runtime,
            dataset_id,
            task_id=task_id,
            target_col=target_col,
            sample_design_ref=inputs["sample_design_ref"],
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
            columns=columns,
        )
    )
    if design_col == sample_binding.split_column:
        raise StrategyError(
            "candidate design column cannot use the sample-design split column"
        )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    source_hash = str(source_evidence["dataset_content_hash"])
    result = design_strategy_candidate(
        frame,
        strategy_type=strategy_type,
        target_col=target_col,
        candidate_design=candidate_design,
        economics_inputs=economics_inputs,
        dataset_id=dataset_id,
        source_dataset_content_hash=source_hash,
        candidate_policy_version=policy_version,
    )
    _assert_source_unchanged(source_path, source_hash)
    revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
    payload = result.to_dict()
    sample_ref = sample_binding.to_ref_dict()
    lineage = payload["strategy_spec"]["metadata"]["lineage"]
    lineage["sample_design_ref"] = sample_ref
    payload["strategy_effect_hash"] = strategy_spec_hash(
        parse_strategy_spec(payload["strategy_spec"])
    )
    payload["design_evidence"] = {
        **payload["design_evidence"],
        "sample_design_ref": sample_ref,
        "sample_design_source_ref": sample_binding.source_ref,
        "development_population_count": sample_binding.development_population_count,
        "nan_labels_dropped": int(nan_labels_dropped),
    }
    return {
        "schema_version": _CANDIDATE_TOOL_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_dataset_content_hash": source_hash,
        "source_evidence": {
            **source_evidence,
            "sample_design_ref": sample_ref,
            "sample_design_source_ref": sample_binding.source_ref,
        },
        "sample_design_ref": sample_ref,
        "strategy_type": strategy_type,
        "target_col": target_col,
        "candidate_policy_version": policy_version,
        **payload,
    }


def tool_build_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    has_spec = inputs.get("strategy_spec") is not None
    has_rules = inputs.get("rules") is not None
    if has_spec == has_rules:
        raise StrategyError(
            "build_strategy requires exactly one of rules or strategy_spec"
        )
    if has_spec:
        strategy = build_strategy_from_spec(
            dict(inputs["strategy_spec"]),
            score_col=_optional_str(inputs.get("score_col")),
            description=str(inputs.get("description") or ""),
        )
    else:
        strategy = build_strategy(
            str(inputs["strategy_type"]),
            list(inputs["rules"]),
            score_col=_optional_str(inputs.get("score_col")),
            default_decision=inputs.get("default_decision"),
            description=str(inputs.get("description") or ""),
        )
    strategy = replace(
        strategy,
        id=_strategy_instance_id(str(ctx.task_id), strategy),
    )
    persisted = runtime.strategies.get_strategy(strategy.id)
    if persisted is None:
        runtime.strategies.create_strategy_with_audit(
            ctx.task_id,
            strategy,
            audit={
                "kind": "strategy.create",
                "target_ref": strategy.id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": str(ctx.task_id),
                    "strategy_type": strategy.strategy_type,
                    "rule_count": len(strategy.rules),
                },
            },
        )
    else:
        metadata = runtime.strategies.get_strategy_meta(strategy.id)
        if (
            metadata is None
            or str(metadata["task_id"]) != str(ctx.task_id)
            or not _same_strategy_payload(persisted, strategy)
        ):
            raise StrategyError(
                "strategy instance identity collision; refusing to reuse another "
                "task or a different persisted payload"
            )
        # Idempotent retries must report the row that actually exists, not a
        # transient object with display metadata the database never stored.
        strategy = persisted
    return {
        "strategy_id": strategy.id,
        "strategy_type": strategy.strategy_type,
        "score_col": strategy.score_col,
        "default_decision": strategy.default_decision,
        "description": strategy.description,
        "rules": [_jsonable(rule) for rule in strategy.rules],
        "dsl_schema_version": strategy.spec.schema_version,
        "strategy_spec": strategy.spec.to_dict(),
        "inferred_score_direction": (
            infer_strategy_rule_direction(list(strategy.rules), strategy.score_col)
            if not has_spec
            else None
        ),
    }


def _strategy_instance_id(task_id: str, strategy: Strategy) -> str:
    if strategy.spec is None:
        raise StrategyError("canonical strategy spec is required for persistence")
    semantic_digest = strategy_spec_hash(strategy.spec)
    payload = {
        "task_id": str(task_id),
        "dsl": json.loads(
            canonical_strategy_json(strategy.spec, include_display_metadata=True)
        ),
        "score_col": strategy.score_col,
        "description": strategy.description,
    }
    instance_digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"strategy-{semantic_digest[:12]}-{instance_digest[:12]}"


def _same_strategy_payload(left: Strategy, right: Strategy) -> bool:
    if left.spec is None or right.spec is None:
        return False
    return (
        left.strategy_type == right.strategy_type
        and left.score_col == right.score_col
        and left.default_decision == right.default_decision
        and left.description == right.description
        and canonical_strategy_json(left.spec, include_display_metadata=True)
        == canonical_strategy_json(right.spec, include_display_metadata=True)
    )


_APPLY_SCHEMA_VERSION = "strategy.apply.v1"
_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_APPLY_OUTPUT_SUFFIXES = {
    "action": "action",
    "value": "value",
    "value_type": "value_type",
    "rule_id": "rule_id",
    "reason_code": "reason_code",
}


def tool_apply_strategy(inputs: dict, ctx) -> dict:
    """Apply one persisted canonical Strategy DSL to a task-owned dataset.

    Execution delegates all condition and first-match semantics to the canonical
    vectorized evaluator.  This layer only projects its typed actions into new
    columns, atomically registers the derived parquet, and records evidence.
    """

    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    strategy = _strategy(runtime, str(inputs["strategy_id"]), task_id=task_id)
    spec = parse_strategy_spec(strategy.spec or legacy_strategy_to_spec(strategy))
    dataset = _owned_dataset(runtime, dataset_id, task_id=task_id)
    source_path = runtime.registry.resolve_path(dataset.id)
    source_hash = sha256_file(source_path)
    frame = runtime.backend.read_frame(source_path)
    if sha256_file(source_path) != source_hash:
        raise StrategyError(
            "source dataset changed while the strategy was being applied"
        )
    output_columns = _strategy_apply_output_columns(inputs, frame)

    evaluation = evaluate_strategy_frame(frame, spec)
    action_values, action_value_types = _strategy_apply_values(
        evaluation.decisions,
        strategy_type=spec.strategy_type,
    )
    derived = frame.copy()
    derived[output_columns["action"]] = evaluation.action_type
    derived[output_columns["value"]] = action_values
    derived[output_columns["value_type"]] = action_value_types
    derived[output_columns["rule_id"]] = evaluation.matched_rule_id
    derived[output_columns["reason_code"]] = evaluation.reason_code

    action_counts = _string_counts(evaluation.action_type)
    rule_counts = _rule_counts(evaluation.matched_rule_id)
    default_count = int(evaluation.matched_rule_id.isna().sum())
    strategy_hash = strategy_spec_hash(spec)
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(
        runtime.datasets_root / task_id / "strategy_apply",
        f"applied_{strategy_hash[:12]}_{uuid.uuid4().hex}.parquet",
    )
    try:
        derived.to_parquet(staged.path, index=False)
        result_hash = sha256_file(staged.path)
        evidence = {
            "source_dataset_content_hash": source_hash,
            "strategy_effect_hash": strategy_hash,
            "result_dataset_content_hash": result_hash,
        }

        def audit_factory(registered_dataset):
            return {
                "kind": "strategy.apply",
                "target_ref": registered_dataset.id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": task_id,
                    "source_dataset_id": dataset.id,
                    "strategy_id": strategy.id,
                    "strategy_type": spec.strategy_type,
                    "population_count": int(len(frame)),
                    "action_counts": action_counts,
                    "rule_counts": rule_counts,
                    "default_count": default_count,
                    "output_columns": output_columns,
                    "evidence": evidence,
                },
            }

        registered = uow.finalize_with_connection(
            runtime.repo.transaction,
            lambda conn: runtime.registry.register_existing_with_audit_on_connection(
                conn,
                staged.final_path,
                audit_factory=audit_factory,
                task_id=task_id,
                role="strategy.applied",
                anchor_target=dataset.id,
                seed=int(ctx.seed or 0),
            ),
        )
    except Exception:
        uow.rollback()
        raise

    return {
        "schema_version": _APPLY_SCHEMA_VERSION,
        "strategy_id": strategy.id,
        "strategy_type": spec.strategy_type,
        "source_dataset_id": dataset.id,
        "result_dataset_id": registered.id,
        "population_count": int(len(frame)),
        "action_counts": action_counts,
        "rule_counts": rule_counts,
        "default_count": default_count,
        "output_columns": output_columns,
        "evidence": evidence,
    }


def _strategy_apply_output_columns(inputs: dict, frame: pd.DataFrame) -> dict[str, str]:
    raw_prefix = inputs.get("output_prefix")
    raw_columns = inputs.get("output_columns")
    if raw_prefix is not None and raw_columns is not None:
        raise StrategyError(
            "apply_strategy accepts output_prefix or output_columns, not both"
        )
    if raw_columns is not None and not isinstance(raw_columns, dict):
        raise StrategyError("output_columns must be an object")

    if raw_columns is None:
        if raw_prefix is not None and not isinstance(raw_prefix, str):
            raise StrategyError("output_prefix must be a string")
        prefix = "strategy_" if raw_prefix is None else raw_prefix
        _require_safe_output_name(prefix, name="output_prefix", is_prefix=True)
        columns = {
            key: f"{prefix}{suffix}" for key, suffix in _APPLY_OUTPUT_SUFFIXES.items()
        }
    else:
        unsupported = sorted(set(raw_columns) - set(_APPLY_OUTPUT_SUFFIXES))
        if unsupported:
            raise StrategyError(
                "output_columns has unsupported fields: " + ", ".join(unsupported)
            )
        columns = {}
        for key, suffix in _APPLY_OUTPUT_SUFFIXES.items():
            value = raw_columns.get(key)
            if value is None:
                columns[key] = f"strategy_{suffix}"
            elif not isinstance(value, str):
                raise StrategyError(f"output_columns.{key} must be a string")
            else:
                columns[key] = value

    for key, column in columns.items():
        _require_safe_output_name(column, name=f"output_columns.{key}")
    normalized_outputs = [column.casefold() for column in columns.values()]
    if len(set(normalized_outputs)) != len(normalized_outputs):
        raise StrategyError(
            "strategy output column names must be case-insensitively unique"
        )
    source_columns = {str(column).casefold() for column in frame.columns}
    collisions = sorted(
        column for column in columns.values() if column.casefold() in source_columns
    )
    if collisions:
        raise StrategyError(
            "strategy output columns already exist (case-insensitive): "
            + ", ".join(collisions)
        )
    return columns


def _require_safe_output_name(
    value: str,
    *,
    name: str,
    is_prefix: bool = False,
) -> None:
    limit = 48 if is_prefix else 64
    if not isinstance(value, str) or not value or len(value) > limit:
        raise StrategyError(f"{name} must be a non-empty safe identifier")
    if _SAFE_OUTPUT_NAME.fullmatch(value) is None:
        raise StrategyError(
            f"{name} must contain only ASCII letters, digits, and underscores "
            "and cannot start with a digit"
        )


def _strategy_apply_values(
    decisions: pd.Series,
    *,
    strategy_type: str,
) -> tuple[pd.Series, pd.Series]:
    decision_values = decisions.tolist()
    value_types = [_strategy_value_type(value) for value in decision_values]
    numeric_storage = strategy_type in {"limit", "pricing"} and all(
        value_type in {"integer", "number"} for value_type in value_types
    )
    values: list[object] = []
    for value, value_type in zip(decision_values, value_types, strict=True):
        values.append(
            _strategy_storage_value(
                value,
                value_type=value_type,
                numeric_storage=numeric_storage,
            )
        )
    return (
        pd.Series(values, index=decisions.index, dtype="object"),
        pd.Series(value_types, index=decisions.index, dtype="object"),
    )


def _strategy_storage_value(value, *, value_type: str, numeric_storage: bool):
    # Segment ids and legacy approval/reject output aliases may legally mix JSON
    # scalar types; parquet has no union column. Preserve their exact type in the
    # adjacent value_type column and use deterministic text storage. Canonical
    # limit/pricing decision values remain numeric for immediate downstream use.
    if numeric_storage:
        return value
    if value_type == "string":
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strategy_value_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    raise StrategyError("strategy decision value must be JSON serializable")


def _string_counts(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts(dropna=False).to_dict()
    return {
        str(key): int(counts[key]) for key in sorted(counts, key=lambda item: str(item))
    }


def _rule_counts(values: pd.Series) -> dict[str, int]:
    return _string_counts(values.loc[values.notna()].map(str))


def tool_backtest_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    strategy = _strategy(runtime, str(inputs["strategy_id"]), task_id=str(ctx.task_id))
    baseline_id = _optional_str(inputs.get("baseline_strategy_id"))
    baseline = (
        _strategy(runtime, baseline_id, task_id=str(ctx.task_id))
        if baseline_id
        else None
    )
    dataset_id = str(inputs["dataset_id"])
    dataset = _owned_dataset(runtime, dataset_id, task_id=task_id)
    source_path = runtime.registry.resolve_path(dataset.id)
    source_dataset_content_hash = sha256_file(source_path)
    frame = runtime.backend.read_frame(source_path)
    if sha256_file(source_path) != source_dataset_content_hash:
        raise StrategyError(
            "source dataset changed while the strategy backtest was running"
        )
    target_col = str(inputs["target_col"])
    sample_binding = None
    if inputs.get("sample_design_ref") is not None:
        try:
            workspace = DataWorkspaceRepository(
                runtime.settings.db_path
            ).get_or_default(task_id)
        except DataWorkspaceDataError as exc:
            raise StrategyError("strategy backtest DataWorkspace is invalid") from exc
        if (
            workspace.active_dataset_id != dataset.id
            or workspace.active_dataset_content_hash
            != source_dataset_content_hash
        ):
            raise StrategyError(
                "strategy backtest dataset must be the exact active DataWorkspace dataset"
            )
        if workspace.semantic_mapping.target_col != target_col:
            raise StrategyError(
                "strategy backtest target must match the confirmed DataWorkspace target"
            )
        sample_binding = load_strategy_sample_design_execution_binding(
            runtime,
            task_id=task_id,
            sample_design_ref=inputs["sample_design_ref"],
            dataset_id=dataset.id,
            dataset_content_hash=source_dataset_content_hash,
            workspace_revision=workspace.revision,
            workspace_generation=workspace.analysis_generation,
            semantic_mapping_hash=data_semantic_mapping_hash(
                workspace.semantic_mapping
            ),
            target_col=target_col,
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        )
        frame = bind_strategy_development_frame(
            frame,
            binding=sample_binding,
            normalize_target=False,
        )
    # Keep the full population in the typed envelope while still requiring an
    # explicit confirmation before label metrics exclude missing supervision.
    nan_labels_dropped = require_labels_confirmed(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    economics_inputs = _typed_economics_inputs(
        frame,
        strategy_type=strategy.strategy_type,
        payload=inputs.get("economics_inputs"),
    )
    precomputed_economics, approval_profit_inputs = _approval_profit_inputs(
        strategy_type=strategy.strategy_type,
        profit_params=inputs.get("profit_params"),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    result = run_typed_backtest(
        frame,
        strategy.spec or legacy_strategy_to_spec(strategy),
        target_col=target_col,
        target_bad_value=(1 if sample_binding is None else sample_binding.target_bad_value),
        sample_design_ref=(
            None if sample_binding is None else sample_binding.to_ref_dict()
        ),
        strategy_id=strategy.id,
        baseline=(
            None
            if baseline is None
            else baseline.spec or legacy_strategy_to_spec(baseline)
        ),
        economics=precomputed_economics,
        economics_inputs=economics_inputs,
        approval_profit_inputs=approval_profit_inputs,
    )
    if sha256_file(source_path) != source_dataset_content_hash:
        raise StrategyError(
            "source dataset changed while the strategy backtest was running"
        )
    if sample_binding is not None:
        revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
    backtest_id = _backtest_id(
        dataset_id,
        result,
        source_dataset_content_hash=source_dataset_content_hash,
    )
    audit = {
        "kind": "strategy.backtest",
        "target_ref": backtest_id,
        "outcome": "succeeded",
        "detail": {
            "task_id": str(ctx.task_id),
            "strategy_id": strategy.id,
            "dataset_id": dataset_id,
            "source_dataset_content_hash": source_dataset_content_hash,
            "schema_version": result.schema_version,
            "sample_design_ref": (
                None if sample_binding is None else sample_binding.to_ref_dict()
            ),
            "strategy_type": result.strategy_type,
            "population_count": result.population_count,
            "labeled_count": result.labeled_count,
            **_backtest_audit_summary(result),
        },
    }
    if sample_binding is None:
        # The unbound branch is the explicit legacy/direct-tool compatibility
        # boundary. V2 workflows always supply a governed sample reference.
        existing = runtime.strategies.get_backtest(backtest_id)
        if existing is None:
            runtime.strategies.save_backtest_with_audit(
                backtest_id,
                strategy.id,
                dataset_id,
                result,
                audit=audit,
            )
        elif backtest_record_payload(existing) != result.to_dict():
            raise StrategyError(
                "backtest identity collision; refusing to reuse different evidence"
            )
    else:
        # Hold the SQLite writer lock while the governed sample/workspace
        # binding is re-authenticated and the evidence + audit are inserted.
        # This closes the gap where a sample artifact could be invalidated
        # between the prior read-side validation and durable persistence.
        with runtime.strategies.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_strategy_sample_design_execution_binding_on_connection(
                conn,
                sample_binding,
            )
            _assert_source_unchanged(source_path, source_dataset_content_hash)
            existing = runtime.strategies.get_backtest_on_connection(
                conn,
                backtest_id,
            )
            if existing is None:
                runtime.strategies.save_backtest_with_audit_on_connection(
                    conn,
                    backtest_id,
                    strategy.id,
                    dataset_id,
                    result,
                    audit=audit,
                )
            elif backtest_record_payload(existing) != result.to_dict():
                raise StrategyError(
                    "backtest identity collision; refusing to reuse different evidence"
                )
    payload = result.to_dict()
    payload["backtest_id"] = backtest_id
    payload["source_dataset_content_hash"] = source_dataset_content_hash
    payload["nan_labels_dropped"] = nan_labels_dropped
    if result.strategy_type in {"approval", "reject"}:
        payload.update(approval_backtest_projection(result))
    profit_note = result.economics.get("profit_note")
    if profit_note:
        # FIN-3 #4: a profit backtest was requested but the EL chain inputs
        # (pd_col/ead_col) were missing, so expected_profit is None rather than a
        # fabricated 0.0. Surface the reason as a red flag instead of failing silently.
        payload["red_flags"] = [
            *payload.get("red_flags", []),
            {
                "code": "expected_profit_unavailable",
                "level": "amber",
                "message": profit_note,
            },
        ]
    return payload


def tool_tradeoff_view(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    if "sample_design_ref" not in inputs:
        raise StrategyError("sample_design_ref is required")
    score_col = str(inputs["score_col"])
    frame, source_evidence, source_path, sample_binding = (
        _strategy_development_frame_with_evidence(
            runtime,
            dataset_id,
            task_id=task_id,
            target_col=target_col,
            sample_design_ref=inputs["sample_design_ref"],
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
            columns=_unique(
                [
                    score_col,
                    target_col,
                    _optional_str(inputs.get("ead_col")),
                    _optional_str(inputs.get("pd_col")),
                ]
            ),
        )
    )
    if score_col == sample_binding.split_column:
        raise StrategyError("score_col cannot use the sample-design split column")
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    score_direction = normalize_score_direction(
        _optional_str(inputs.get("score_direction"))
    )
    effective_direction = score_direction or "higher_is_better"
    points = tradeoff_view(
        frame,
        score_col=score_col,
        target_col=target_col,
        cutoffs=[float(item) for item in inputs["cutoffs"]]
        if inputs.get("cutoffs") is not None
        else None,
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
        score_direction=score_direction,
        confirm_direction_conflict=bool(inputs.get("confirm_direction_conflict")),
    )
    max_bad_rate = _optional_float(inputs.get("max_bad_rate"))
    min_approval_rate = _optional_float(inputs.get("min_approval_rate"))
    feasible_flags = tradeoff_feasible_flags(
        points, max_bad_rate=max_bad_rate, min_approval_rate=min_approval_rate
    )
    red_flags: list[dict] = []
    recommended = None
    if points and any(feasible_flags):
        recommended = recommend_operating_point(
            [point for point, ok in zip(points, feasible_flags, strict=True) if ok],
            objective=str(inputs.get("objective") or "max_profit"),
            max_bad_rate=max_bad_rate,
        )
    elif points and (max_bad_rate is not None or min_approval_rate is not None):
        red_flags.append(
            {
                "code": "infeasible_constraints",
                "level": "red",
                "message": "在给定 max_bad_rate/min_approval_rate 约束下没有可行 cutoff。",
            }
        )
    direction_check = check_score_direction(
        pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float),
        declared_direction=effective_direction,
    )
    point_rows = []
    for point, feasible in zip(points, feasible_flags, strict=True):
        row = _jsonable(point)
        row["feasible"] = bool(feasible)
        point_rows.append(row)
    result = {
        "points": point_rows,
        "recommended": _jsonable(recommended),
        "nan_labels_dropped": nan_labels_dropped,
        "score_direction": effective_direction,
        "red_flags": red_flags,
    }
    if direction_check.status != "skipped":
        result["direction_diagnostics"] = _jsonable(direction_check)
    _assert_source_unchanged(
        source_path, str(source_evidence["dataset_content_hash"])
    )
    revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
    result["sample_design_ref"] = sample_binding.to_ref_dict()
    return result


def tool_design_cutoff_bands(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    if "sample_design_ref" not in inputs:
        raise StrategyError("sample_design_ref is required")
    score_col = str(inputs["score_col"])
    frame, source_evidence, source_path, sample_binding = (
        _strategy_development_frame_with_evidence(
            runtime,
            dataset_id,
            task_id=task_id,
            target_col=target_col,
            sample_design_ref=inputs["sample_design_ref"],
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
            columns=_unique(
                [
                    score_col,
                    target_col,
                    _optional_str(inputs.get("ead_col")),
                    _optional_str(inputs.get("pd_col")),
                ]
            ),
        )
    )
    if score_col == sample_binding.split_column:
        raise StrategyError("score_col cannot use the sample-design split column")
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    score_direction = normalize_score_direction(
        _optional_str(inputs.get("score_direction"))
    )
    effective_direction = score_direction or "higher_is_better"
    red_flags: list[dict] = []
    # Direction self-check (S1a): a conflict is a red flag and blocks unless the
    # caller confirms, mirroring tradeoff_view's confirm_direction_conflict gate.
    direction_check = check_score_direction(
        pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float),
        declared_direction=effective_direction,
    )
    if direction_check.status == "conflict" and not bool(
        inputs.get("confirm_direction_conflict")
    ):
        from marvis.data.errors import ScoreDirectionConflictError

        raise ScoreDirectionConflictError(
            tool="design_cutoff_bands",
            score_col=score_col,
            target_col=target_col,
            declared_direction=effective_direction,
            implied_direction=direction_check.implied_direction,
            corr=direction_check.corr,
            n_labeled=direction_check.n,
        )
    if direction_check.status == "conflict":
        red_flags.append(
            {
                "code": "direction_conflict",
                "level": "red",
                "message": (
                    f"分数方向自检冲突：声明 {effective_direction}，数据隐含 "
                    f"{direction_check.implied_direction}（corr={direction_check.corr:.3f}）。"
                ),
            }
        )
    result = design_cutoff_bands(
        frame,
        score_col=score_col,
        target_col=target_col,
        score_direction=effective_direction,
        n_bands=int(inputs.get("n_bands", 5)),
        band_edges=[float(edge) for edge in inputs["band_edges"]]
        if inputs.get("band_edges") is not None
        else None,
        objective=str(inputs.get("objective") or "max_profit"),
        max_bad_rate=_optional_float(inputs.get("max_bad_rate")),
        min_approval_rate=_optional_float(inputs.get("min_approval_rate")),
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    red_flags.extend(_jsonable(flag) for flag in result.red_flags)
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    _assert_source_unchanged(
        source_path, str(source_evidence["dataset_content_hash"])
    )
    revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
    return {
        "bands": [_jsonable(band) for band in result.bands],
        "band_edges": [float(edge) for edge in result.band_edges],
        "recommended_rules": [dict(rule) for rule in result.recommended_rules],
        "red_flags": red_flags,
        "score_direction": effective_direction,
        "nan_labels_dropped": nan_labels_dropped,
        "sample_design_ref": sample_binding.to_ref_dict(),
    }


def tool_compare_strategies(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    if "sample_design_ref" not in inputs:
        raise StrategyError("sample_design_ref is required")
    frame, source_evidence, source_path, sample_binding = (
        _strategy_development_frame_with_evidence(
            runtime,
            dataset_id,
            task_id=task_id,
            target_col=target_col,
            sample_design_ref=inputs["sample_design_ref"],
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        )
    )
    strategy = _strategy(runtime, str(inputs["strategy_id"]), task_id=task_id)
    baseline_id = _optional_str(inputs.get("baseline_strategy_id"))
    if baseline_id is None:
        # No baseline means there is no comparison population or delta. Keep
        # every affected value explicitly unavailable; zero would be a false
        # measured result and label_coverage=1 would invent evidence.
        _assert_source_unchanged(
            source_path, str(source_evidence["dataset_content_hash"])
        )
        revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
        return {
            "status": "no_baseline",
            "matrix_2x2": None,
            "deltas": None,
            "summary_text": "未提供基线策略，跳过对比。",
            "red_flags": [],
            "nan_labels_dropped": 0,
            "label_coverage": None,
            "sample_design_ref": sample_binding.to_ref_dict(),
        }
    baseline = _strategy(runtime, baseline_id, task_id=task_id)
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    result = compare_strategies(
        frame,
        strategy,
        baseline,
        target_col=target_col,
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    payload = _jsonable(result)
    payload["status"] = "compared"
    payload["nan_labels_dropped"] = nan_labels_dropped
    payload["label_coverage"] = _label_coverage(
        len(frame) + nan_labels_dropped, nan_labels_dropped
    )
    _assert_source_unchanged(
        source_path, str(source_evidence["dataset_content_hash"])
    )
    revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
    payload["sample_design_ref"] = sample_binding.to_ref_dict()
    return payload


def tool_limit_pricing_matrix(inputs: dict, ctx) -> dict:
    """S6 (A3): a band x limit x rate expected-profit grid with an EL simulation.

    Always computes and returns the full matrix + per-band recommended feasible cell.
    CSV and Markdown deliverables are written ONLY when ``confirm`` is true -- the
    trusted workflow calls this after its explicit matrix confirmation gate. Files
    always live under the task root; when ``strategy_id`` is supplied they are also
    transactionally registered against a task-owned limit/pricing strategy.
    """
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    if "sample_design_ref" not in inputs:
        raise StrategyError("sample_design_ref is required")
    score_col = str(inputs["score_col"])
    target_col = _optional_str(inputs.get("target_col"))
    pd_col = _optional_str(inputs.get("pd_col"))
    if (target_col is None) == (pd_col is None):
        raise StrategyError(
            "limit_pricing_matrix requires exactly one risk source: pd_col or target_col"
        )
    strategy_id = _optional_str(inputs.get("strategy_id"))
    if strategy_id is not None:
        strategy = _strategy(runtime, strategy_id, task_id=task_id)
        if strategy.strategy_type not in {"limit", "pricing"}:
            raise StrategyError(
                "limit_pricing_matrix artifacts may attach only to a limit or pricing strategy"
            )
    limit_grid, rate_grid, params, band_edges, n_bands = _validated_pricing_inputs(
        inputs
    )
    columns = _unique([score_col, target_col, pd_col])
    frame, source_evidence, source_path, sample_binding = (
        _strategy_development_frame_with_evidence(
            runtime,
            dataset_id,
            task_id=task_id,
            target_col=target_col,
            sample_design_ref=inputs["sample_design_ref"],
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
            columns=columns,
            normalize_target=False,
        )
    )
    if score_col == sample_binding.split_column:
        raise StrategyError("score_col cannot use the sample-design split column")
    if pd_col is not None and pd_col == sample_binding.split_column:
        raise StrategyError("pd_col cannot use the sample-design split column")
    expected_source_hash = _optional_str(inputs.get("expected_source_hash"))
    if (
        expected_source_hash is not None
        and expected_source_hash != source_evidence["dataset_content_hash"]
    ):
        raise StrategyError(
            "source dataset changed since preview; recompute and review the matrix before export"
        )
    if target_col:
        frame, nan_labels_dropped = resolve_labeled_frame(
            frame,
            target_col,
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        )
    else:
        nan_labels_dropped = 0
    _validate_pricing_frame(
        frame,
        score_col=score_col,
        target_col=target_col,
        pd_col=pd_col,
    )
    result = limit_pricing_matrix(
        frame,
        score_col=score_col,
        limit_grid=limit_grid,
        rate_grid=rate_grid,
        params=params,
        target_col=target_col,
        target_bad_value=sample_binding.target_bad_value,
        pd_col=pd_col,
        band_edges=band_edges,
        n_bands=n_bands,
    )
    red_flags = [dict(flag) for flag in result.red_flags]
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )

    assumptions = {
        "dataset_id": dataset_id,
        "score_col": score_col,
        "target_col": target_col,
        "target_bad_value": sample_binding.target_bad_value,
        "pd_col": pd_col,
        "lgd": params.lgd,
        "funding_rate": params.funding_rate,
        "term_months": params.term_months,
        "cost_per_loan": params.cost_per_loan,
        "el_ead_max": params.el_ead_max,
        "risk_source": "pd_col" if pd_col else "target_col",
        "population_scope": "development",
        "sample_design_ref": sample_binding.to_ref_dict(),
        "limit_grid": limit_grid,
        "rate_grid": rate_grid,
        "band_edges": [float(edge) for edge in result.band_edges],
        "n_bands": n_bands,
    }

    payload = {
        "matrix": [_jsonable(cell) for cell in result.matrix],
        "recommended": [dict(item) for item in result.recommended],
        "band_edges": [float(edge) for edge in result.band_edges],
        "assumptions": assumptions,
        "source_evidence": source_evidence,
        "source_dataset_content_hash": source_evidence["dataset_content_hash"],
        "sample_design_ref": sample_binding.to_ref_dict(),
        "red_flags": red_flags,
        "nan_labels_dropped": nan_labels_dropped,
    }

    _assert_source_unchanged(
        source_path, str(source_evidence["dataset_content_hash"])
    )
    revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)

    artifacts: list[dict] = []
    # The first workflow step is a computation/decision view. Only the explicit
    # accept/export step calls again with confirm=true and persists deliverables.
    if bool(inputs.get("confirm")):
        _assert_source_unchanged(
            source_path, str(source_evidence["dataset_content_hash"])
        )
        artifacts = _write_limit_pricing_artifacts(
            runtime,
            ctx,
            result=result,
            strategy_id=strategy_id,
            assumptions=assumptions,
            source_evidence=source_evidence,
            red_flags=red_flags,
            source_path=source_path,
            sample_design_binding=sample_binding,
        )
    payload["artifacts"] = artifacts
    return payload


def _limit_pricing_csv(result: LimitPricingResult) -> str:
    recommended = {
        (item["band"], float(item["limit"]), float(item["rate"]))
        for item in result.recommended
    }
    rows = []
    for cell in result.matrix:
        row = _jsonable(cell)
        row["recommended"] = (
            cell.band,
            float(cell.limit),
            float(cell.rate),
        ) in recommended
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


def _validated_pricing_inputs(
    inputs: dict,
) -> tuple[list[float], list[float], PricingParams, list[float] | None, int]:
    limit_grid = _bounded_numeric_grid(
        inputs["limit_grid"],
        name="limit_grid",
        minimum=0.0,
        maximum=1_000_000_000_000.0,
        strictly_greater_than_minimum=True,
    )
    rate_grid = _bounded_numeric_grid(
        inputs["rate_grid"],
        name="rate_grid",
        minimum=0.0,
        maximum=1.0,
    )
    n_bands = int(inputs.get("n_bands", 5))
    if not 1 <= n_bands <= 100:
        raise StrategyError("n_bands must be between 1 and 100")
    band_edges_raw = inputs.get("band_edges")
    band_edges = None
    if band_edges_raw is not None:
        band_edges = _bounded_numeric_grid(
            band_edges_raw,
            name="band_edges",
            minimum=-1_000_000_000_000.0,
            maximum=1_000_000_000_000.0,
            max_items=101,
            require_unique=False,
        )
        if len(band_edges) < 2 or any(
            right <= left for left, right in zip(band_edges, band_edges[1:])
        ):
            raise StrategyError(
                "band_edges must contain at least two strictly increasing values"
            )
    band_count = len(band_edges) - 1 if band_edges is not None else n_bands
    cell_count = band_count * len(limit_grid) * len(rate_grid)
    if cell_count > 10_000:
        raise StrategyError(
            f"pricing grid is too large ({cell_count} cells); maximum is 10000"
        )
    lgd = _bounded_finite_number(inputs.get("lgd", 0.6), "lgd", 0.0, 1.0)
    funding_rate = _bounded_finite_number(
        inputs["funding_rate"], "funding_rate", 0.0, 1.0
    )
    term_months = int(inputs["term_months"])
    if not 1 <= term_months <= 1_200:
        raise StrategyError("term_months must be between 1 and 1200")
    cost_per_loan = _bounded_finite_number(
        inputs["cost_per_loan"], "cost_per_loan", 0.0, 1_000_000_000_000.0
    )
    el_ead_max = _bounded_finite_number(
        inputs.get("el_ead_max", 0.20), "el_ead_max", 0.0, 1.0
    )
    return (
        limit_grid,
        rate_grid,
        PricingParams(
            lgd=lgd,
            funding_rate=funding_rate,
            term_months=term_months,
            cost_per_loan=cost_per_loan,
            el_ead_max=el_ead_max,
        ),
        band_edges,
        n_bands,
    )


def _bounded_numeric_grid(
    raw_values,
    *,
    name: str,
    minimum: float,
    maximum: float,
    max_items: int = 100,
    strictly_greater_than_minimum: bool = False,
    require_unique: bool = True,
) -> list[float]:
    try:
        values = [float(item) for item in raw_values]
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"{name} must contain only finite numbers") from exc
    if not values:
        raise StrategyError(f"{name} must not be empty")
    if len(values) > max_items:
        raise StrategyError(f"{name} may contain at most {max_items} values")
    if any(not math.isfinite(value) for value in values):
        raise StrategyError(f"{name} must contain only finite numbers")
    if strictly_greater_than_minimum:
        invalid = any(not minimum < value <= maximum for value in values)
    else:
        invalid = any(not minimum <= value <= maximum for value in values)
    if invalid:
        operator = ">" if strictly_greater_than_minimum else ">="
        raise StrategyError(
            f"{name} values must be {operator} {minimum:g} and <= {maximum:g}"
        )
    if require_unique and len(set(values)) != len(values):
        raise StrategyError(f"{name} must not contain duplicate values")
    return values


def _bounded_finite_number(raw, name: str, minimum: float, maximum: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise StrategyError(f"{name} must be a finite number")
    if not minimum <= value <= maximum:
        raise StrategyError(f"{name} must be between {minimum:g} and {maximum:g}")
    return value


def _validate_pricing_frame(
    frame: pd.DataFrame,
    *,
    score_col: str,
    target_col: str | None,
    pd_col: str | None,
) -> None:
    if frame.empty:
        raise StrategyError("limit_pricing_matrix requires a non-empty dataset")
    try:
        scores = pd.to_numeric(frame[score_col], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise StrategyError("score_col must contain finite numeric values") from exc
    if scores.isna().any() or not scores.map(math.isfinite).all():
        raise StrategyError("score_col must contain finite numeric values")
    risk_col = pd_col or target_col
    try:
        risk = pd.to_numeric(frame[risk_col], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"{risk_col} must contain finite numeric values") from exc
    if risk.isna().any() or not risk.map(math.isfinite).all():
        raise StrategyError(f"{risk_col} must contain finite numeric values")
    if pd_col is not None:
        if ((risk < 0.0) | (risk > 1.0)).any():
            raise StrategyError("pd_col values must be between 0 and 1")
    elif not set(risk.unique()).issubset({0.0, 1.0}):
        raise StrategyError("target_col must be binary 0/1")


def _write_limit_pricing_artifacts(
    runtime,
    ctx,
    *,
    result: LimitPricingResult,
    strategy_id: str | None,
    assumptions: dict,
    source_evidence: dict,
    red_flags: list[dict],
    source_path: Path,
    sample_design_binding: StrategySampleDesignExecutionBinding,
) -> list[dict]:
    task_id = str(ctx.task_id)
    analysis_dir = Path(runtime.settings.tasks_dir) / task_id / "strategy_analysis"
    stem = _analysis_artifact_stem(
        "limit_pricing",
        str(source_evidence["dataset_content_hash"]),
        assumptions,
    )
    uow = ArtifactUnitOfWork()
    staged_csv = uow.stage_file(analysis_dir, f"{stem}.csv")
    staged_markdown = uow.stage_file(analysis_dir, f"{stem}.md")
    specs = (
        ("limit_pricing_csv", staged_csv),
        ("limit_pricing_markdown", staged_markdown),
    )
    provenance = _task_analysis_artifact_provenance(
        analysis_kind="limit_pricing",
        source_hash=str(source_evidence["dataset_content_hash"]),
        assumptions=assumptions,
    )
    records = []
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        staged_csv.path.write_text(_limit_pricing_csv(result), encoding="utf-8")
        staged_markdown.path.write_text(
            _limit_pricing_markdown(
                result=result,
                assumptions=assumptions,
                source_evidence=source_evidence,
                red_flags=red_flags,
            ),
            encoding="utf-8",
        )

        # Deterministic report paths mean identical invocations can target the
        # same files. Acquire the cross-process writer lock before promotion so
        # a failed writer always restores its files before a peer may promote.
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                require_strategy_sample_design_execution_binding_on_connection(
                    conn,
                    sample_design_binding,
                )
                _assert_source_unchanged(
                    source_path,
                    str(source_evidence["dataset_content_hash"]),
                )
                uow.promote_all()
                for kind, staged in specs:
                    path = str(staged.final_path)
                    task_record = runtime.task_artifacts.register_on_connection(
                        conn,
                        task_id=task_id,
                        kind=kind,
                        path=path,
                        content_hash=sha256_file(staged.final_path),
                        origin_tool="strategy.limit_pricing_matrix",
                        provenance=provenance,
                    )
                    strategy_artifact_id = None
                    if strategy_id is not None:
                        existing = conn.execute(
                            """
                            SELECT id
                              FROM strategy_artifacts
                             WHERE strategy_id = ? AND kind = ? AND path = ?
                             ORDER BY created_at, id
                             LIMIT 1
                            """,
                            (strategy_id, kind, path),
                        ).fetchone()
                        if existing is not None:
                            strategy_artifact_id = str(existing["id"])
                        else:
                            strategy_artifact_id = runtime.strategies.save_strategy_artifact_with_audit_on_connection(
                                conn,
                                strategy_id,
                                kind=kind,
                                path=path,
                                audit={
                                    "kind": "strategy.artifact",
                                    "target_ref": strategy_id,
                                    "outcome": "succeeded",
                                    "detail": {
                                        "task_id": task_id,
                                        "kind": kind,
                                        "path": path,
                                        "source_dataset_content_hash": (
                                            source_evidence[
                                                "dataset_content_hash"
                                            ]
                                        ),
                                    },
                                },
                            )
                    records.append(
                        {
                            "task_record": task_record,
                            "strategy_artifact_id": strategy_artifact_id,
                        }
                    )
                # Keep any DB commit failure inside the writer-lock boundary so
                # promoted files can be restored before another writer starts.
                conn.commit()
                db_committed = True
            except Exception:
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
        uow.commit()
    except Exception:
        # Once the DB rows commit, a later backup-cleanup error must not delete
        # an identical peer's durable files. Failed pre-commit paths were
        # already rolled back while holding the SQLite writer lock.
        if not db_committed and not rollback_attempted_under_lock:
            uow.rollback()
        raise
    return [
        {
            "artifact_id": str(record["task_record"]["id"]),
            "kind": kind,
            "filename": staged.final_path.name,
            "content_hash": str(record["task_record"]["content_hash"]),
            **(
                {"strategy_artifact_id": str(record["strategy_artifact_id"])}
                if record["strategy_artifact_id"] is not None
                else {}
            ),
        }
        for (kind, staged), record in zip(specs, records, strict=True)
    ]


def _limit_pricing_markdown(
    *,
    result: LimitPricingResult,
    assumptions: dict,
    source_evidence: dict,
    red_flags: list[dict],
) -> str:
    lines = [
        "# 额度与定价矩阵",
        "",
        f"- 数据集：`{source_evidence['dataset_id']}`",
        f"- 数据哈希：`{source_evidence['dataset_content_hash']}`",
        f"- 矩阵单元：{len(result.matrix)}",
        f"- 推荐档位：{len(result.recommended)}",
        "",
        "## 假设",
        "",
        "```json",
        json.dumps(assumptions, ensure_ascii=False, indent=2, allow_nan=True),
        "```",
        "",
        "## 推荐档位",
        "",
        _markdown_table(
            ["分数带", "额度", "年化利率"],
            [
                [item.get("band"), item.get("limit"), item.get("rate")]
                for item in result.recommended
            ],
        ),
    ]
    if red_flags:
        lines.extend(
            [
                "",
                "## 风险提示",
                "",
                *[f"- {flag.get('message') or flag.get('code')}" for flag in red_flags],
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


_ADOPTION_EVIDENCE_SCHEMA_VERSION = "strategy.adoption-evidence.v1"
_LEGACY_BACKTEST_SCHEMA_VERSION = "strategy.backtest.v1"


def tool_adopt_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id, task_id=task_id)
    backtest_id = str(inputs["backtest_id"])
    backtest = runtime.strategies.get_backtest(backtest_id)
    if backtest is None or backtest.strategy_id != strategy_id:
        raise StrategyError(
            f"backtest {backtest_id} does not belong to strategy {strategy_id}"
        )
    if isinstance(backtest, StrategyBacktestResult):
        if backtest.strategy_type != strategy.strategy_type:
            raise StrategyError(
                "backtest strategy_type does not match the persisted strategy"
            )
    elif strategy.strategy_type not in {"approval", "reject"}:
        raise StrategyError(
            f"{strategy.strategy_type} adoption requires a typed StrategyBacktestResult"
        )

    adoption_evidence, approval_metrics = _strategy_adoption_evidence(
        runtime,
        strategy=strategy,
        backtest=backtest,
        backtest_id=backtest_id,
        task_id=task_id,
    )
    experiment_id = _strategy_monitoring_experiment_id(
        runtime,
        inputs.get("experiment_id"),
        task_id=task_id,
    )
    try:
        adoption_reason = normalize_adoption_reason(inputs.get("adoption_reason"))
    except AdoptionReasonError as exc:
        raise StrategyError(str(exc)) from exc
    effect_execution_id = _optional_str(getattr(ctx, "effect_execution_id", None))
    runtime_generation = _optional_str(getattr(ctx, "runtime_generation", None))
    if (effect_execution_id is None) != (runtime_generation is None):
        raise StrategyError("治理执行元数据不完整，拒绝采纳策略")
    strategy_meta = runtime.strategies.get_strategy_meta(strategy_id)
    if strategy_meta is None:
        raise StrategyError(f"strategy not found: {strategy_id}")
    version = int(strategy_meta["version"])
    strategy_dir = Path(runtime.settings.tasks_dir) / task_id / "strategy"
    stem = f"{strategy_id}_v{version}"

    # band_stats is retained in the manifest only for stored-plan compatibility.
    # It is caller-supplied and not bound to this backtest, so adoption artifacts
    # must not present it as verified evidence. The decision table comes only
    # from the persisted canonical strategy and includes unmatched/default flow.
    rules = _adoption_decision_table_rules(strategy)
    csv_text = decision_table_csv(rules, [])
    monitoring_plan = _build_adoption_monitoring_plan(
        strategy_id=strategy_id,
        strategy_type=strategy.strategy_type,
        version=version,
        evidence=adoption_evidence,
        approval_metrics=approval_metrics,
        experiment_id=experiment_id,
    )
    monitoring_plan_hash = canonical_monitoring_plan_hash(monitoring_plan)
    monitoring_repo = StrategyMonitoringRepository(runtime.settings.db_path)

    # Adoption is one multi-resource commit: the two required deliverables are
    # staged first, then promoted and recorded together with lifecycle/effect
    # state on one caller-owned SQLite transaction. Any filesystem, artifact,
    # audit, or commit failure restores both files and every database mutation.
    uow = ArtifactUnitOfWork()
    staged_csv = uow.stage_file(strategy_dir, f"decision_table_{stem}.csv")
    staged_json = uow.stage_file(strategy_dir, f"monitoring_plan_{stem}.json")
    artifact_specs = (
        ("decision_table_csv", staged_csv),
        ("monitoring_plan_json", staged_json),
    )
    try:
        staged_csv.path.write_text(csv_text, encoding="utf-8")
        save_monitoring_plan(staged_json.path, monitoring_plan)

        def finalize_adoption(conn):
            adopt_result = runtime.strategies.adopt_strategy_with_audit_on_connection(
                conn,
                strategy_id,
                reason=adoption_reason,
                audit={
                    "kind": "strategy.adopt",
                    "target_ref": strategy_id,
                    "outcome": "succeeded",
                    "detail": {
                        "task_id": task_id,
                        "backtest_id": backtest_id,
                        "strategy_type": strategy.strategy_type,
                        "experiment_id": experiment_id,
                        "adoption_reason": adoption_reason,
                        "adoption_evidence": adoption_evidence,
                        "monitoring_plan_id": monitoring_plan.monitoring_plan_id,
                        "monitoring_plan_revision": monitoring_plan.revision,
                        "monitoring_plan_hash": monitoring_plan_hash,
                        **_approval_adoption_audit_summary(approval_metrics),
                    },
                },
                effect_execution_id=effect_execution_id,
                runtime_generation=runtime_generation,
            )
            if int(adopt_result["version"]) != version:
                raise StrategyError("strategy version changed during adoption")
            plan_record = monitoring_repo.create_plan_on_connection(
                conn,
                monitoring_plan,
                expected_revision=0,
                plan_id=monitoring_plan.monitoring_plan_id,
            )
            if plan_record.payload_hash != monitoring_plan_hash:
                raise StrategyError(
                    "monitoring plan ledger hash does not match the adoption artifact"
                )
            artifact_records = []
            for kind, staged in artifact_specs:
                final_path = str(staged.final_path)
                content_hash = sha256_file(staged.final_path)
                content_size = staged.final_path.stat().st_size
                producer_version = _STRATEGY_ADOPTION_ARTIFACT_PRODUCER_VERSIONS[kind]
                provenance = _adoption_artifact_provenance(
                    task_id=task_id,
                    strategy_id=strategy_id,
                    strategy_type=strategy.strategy_type,
                    strategy_version=version,
                    kind=kind,
                    producer_version=producer_version,
                    backtest_id=backtest_id,
                    adoption_evidence=adoption_evidence,
                    monitoring_plan=monitoring_plan,
                    monitoring_plan_hash=monitoring_plan_hash,
                )
                artifact_records.append(
                    runtime.strategies.register_verified_strategy_artifact_with_audit_on_connection(
                        conn,
                        strategy_id,
                        kind=kind,
                        path=final_path,
                        content_hash=content_hash,
                        content_size=content_size,
                        provenance=provenance,
                        audit={
                            "kind": "strategy.artifact",
                            "target_ref": strategy_id,
                            "outcome": "succeeded",
                            "detail": {
                                "task_id": task_id,
                                "kind": kind,
                                "path": final_path,
                                "content_hash": content_hash,
                                "content_size": content_size,
                                "producer_version": producer_version,
                            },
                        },
                    )
                )
            return adopt_result, plan_record, artifact_records

        adopt_result, plan_record, artifact_records = uow.finalize_with_connection(
            runtime.strategies.transaction,
            finalize_adoption,
        )
    except Exception:
        uow.rollback()
        raise

    artifacts = [
        {
            "artifact_id": str(record["id"]),
            "kind": str(record["kind"]),
            "path": str(record["path"]),
            "content_hash": str(record["content_hash"]),
            "content_size": int(record["content_size"]),
        }
        for record in artifact_records
    ]

    return {
        "strategy_id": strategy_id,
        "strategy_type": strategy.strategy_type,
        "backtest_id": backtest_id,
        "version": version,
        "status": "adopted",
        "asset_status": "adopted_local",
        "lifecycle_notice": "本地已采纳，不代表生产上线。",
        "retired_strategy_ids": list(adopt_result["retired_strategy_ids"]),
        "adoption_evidence": adoption_evidence,
        "monitoring_plan_id": plan_record.id,
        "monitoring_plan_revision": plan_record.revision,
        "monitoring_plan_hash": plan_record.payload_hash,
        "artifacts": artifacts,
    }


def _adoption_artifact_provenance(
    *,
    task_id: str,
    strategy_id: str,
    strategy_type: str,
    strategy_version: int,
    kind: str,
    producer_version: str,
    backtest_id: str,
    adoption_evidence: Mapping[str, Any],
    monitoring_plan: MonitoringPlan,
    monitoring_plan_hash: str,
) -> dict[str, Any]:
    evidence = {
        "operation": "strategy.adopt",
        "backtest_id": backtest_id,
        "strategy_type": strategy_type,
        "strategy_version": strategy_version,
        "strategy_effect_hash": adoption_evidence["strategy_effect_hash"],
        "source_dataset_id": adoption_evidence.get("source_dataset_id"),
        "source_dataset_content_hash": adoption_evidence.get(
            "source_dataset_content_hash"
        ),
    }
    if kind == "monitoring_plan_json":
        evidence.update(
            {
                "monitoring_plan_id": monitoring_plan.monitoring_plan_id,
                "monitoring_plan_revision": monitoring_plan.revision,
                "monitoring_plan_hash": monitoring_plan_hash,
            }
        )
    return {
        "schema_version": _STRATEGY_ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "producer_version": producer_version,
        "task_id": task_id,
        "strategy_id": strategy_id,
        "kind": kind,
        "evidence": _jsonable(evidence),
    }


def _strategy_adoption_evidence(
    runtime,
    *,
    strategy: Strategy,
    backtest: BacktestRecord,
    backtest_id: str,
    task_id: str,
) -> tuple[dict, dict | None]:
    if isinstance(backtest, StrategyBacktestResult):
        _require_typed_adoption_quality(backtest)
        expected_effect_hash = strategy_spec_hash(
            strategy.spec or legacy_strategy_to_spec(strategy)
        )
        actual_effect_hash = str(
            backtest.normalized_input.get("strategy_effect_hash") or ""
        )
        if actual_effect_hash != expected_effect_hash:
            raise StrategyError(
                "backtest strategy effect hash does not match the persisted strategy"
            )
        binding = _backtest_binding(runtime, backtest_id)
        if binding["strategy_id"] != strategy.id:
            raise StrategyError(
                f"backtest {backtest_id} does not belong to strategy {strategy.id}"
            )
        if binding["dataset_task_id"] is None:
            raise StrategyError(
                "typed backtest source dataset is not registered; rerun the backtest"
            )
        if binding["dataset_task_id"] != task_id:
            raise StrategyError(
                "typed backtest source dataset must belong to the same task as the strategy"
            )
        if binding["dataset_content_hash"] is None:
            raise StrategyError(
                "typed backtest source dataset file is unavailable; rerun the backtest"
            )
        if binding["backtest_dataset_content_hash"] is None:
            raise StrategyError(
                "typed backtest is missing backtest-time source dataset hash evidence; "
                "rerun the backtest"
            )
        if binding["dataset_content_hash"] != binding["backtest_dataset_content_hash"]:
            raise StrategyError(
                "source dataset content hash no longer matches the backtest evidence"
            )
        evidence = {
            "schema_version": _ADOPTION_EVIDENCE_SCHEMA_VERSION,
            "backtest_schema_version": backtest.schema_version,
            "backtest_id": backtest_id,
            "strategy_id": strategy.id,
            "strategy_type": strategy.strategy_type,
            "source_dataset_id": binding["dataset_id"],
            "source_dataset_content_hash": binding["backtest_dataset_content_hash"],
            "strategy_effect_hash": expected_effect_hash,
            "baseline_effect_hash": backtest.normalized_input["baseline_effect_hash"],
            "target_col": str(backtest.normalized_input["target_col"]),
            "population_count": int(backtest.population_count),
            "labeled_count": int(backtest.labeled_count),
            "label_coverage": float(backtest.label_coverage),
            "metrics": _jsonable(dict(backtest.metrics)),
            "breakdown": [_jsonable(dict(row)) for row in backtest.breakdown],
            "transitions": [_jsonable(dict(row)) for row in backtest.transitions],
            # Per-row pricing economics is deliberately excluded from adoption
            # evidence and audit. The typed envelope has already reconciled it to
            # these aggregates, which are sufficient for governance decisions.
            "economics": _aggregate_economics(backtest.economics),
            "economics_input_evidence": _jsonable(
                dict(backtest.normalized_input["economics_input_evidence"])
            ),
            "warnings": list(backtest.warnings),
        }
        approval_metrics = (
            approval_backtest_projection(
                backtest,
                preserve_undefined_rates=True,
            )
            if strategy.strategy_type in {"approval", "reject"}
            else None
        )
        return evidence, approval_metrics

    approval_metrics = approval_backtest_projection(
        backtest,
        preserve_undefined_rates=True,
    )
    if approval_metrics["approved_bad_rate"] is None:
        raise StrategyError(
            "cannot adopt strategy because approved bad rate is undefined; "
            "provide labeled approved observations and rerun the backtest"
        )
    binding = _backtest_binding(runtime, backtest_id)
    if binding["dataset_task_id"] is not None and binding["dataset_task_id"] != task_id:
        raise StrategyError(
            "legacy backtest source dataset must belong to the same task as the strategy"
        )
    effect_hash = strategy_spec_hash(strategy.spec or legacy_strategy_to_spec(strategy))
    metrics = {
        key: _jsonable(value)
        for key, value in approval_metrics.items()
        if key
        not in {
            "strategy_id",
            "by_segment",
            "expected_profit",
            "profit_note",
        }
    }
    return {
        "schema_version": _ADOPTION_EVIDENCE_SCHEMA_VERSION,
        "backtest_schema_version": _LEGACY_BACKTEST_SCHEMA_VERSION,
        "backtest_id": backtest_id,
        "strategy_id": strategy.id,
        "strategy_type": strategy.strategy_type,
        "source_dataset_id": binding["dataset_id"],
        "source_dataset_content_hash": binding["dataset_content_hash"],
        "strategy_effect_hash": effect_hash,
        "baseline_effect_hash": None,
        "target_col": None,
        # Legacy rows never carried population/label provenance. Keep that gap
        # explicit rather than reconstructing counts from rounded rates.
        "population_count": None,
        "labeled_count": None,
        "label_coverage": None,
        "metrics": metrics,
        "breakdown": [
            _jsonable(dict(row)) for row in approval_metrics.get("by_segment", [])
        ],
        "transitions": [],
        "economics": {
            "expected_profit": _jsonable(approval_metrics.get("expected_profit")),
            "profit_note": _jsonable(approval_metrics.get("profit_note")),
        },
        "economics_input_evidence": {},
        "warnings": [
            "legacy backtest has no task-bound dataset or label-provenance contract"
        ],
    }, approval_metrics


def _adoption_decision_table_rules(strategy: Strategy) -> list[dict]:
    spec = parse_strategy_spec(strategy.spec or legacy_strategy_to_spec(strategy))
    rows = [_jsonable(rule) for rule in strategy.rules]
    default_action = spec.default_action
    default_decision = {
        "approval": "approve",
        "reject": "reject",
        "review": "review",
        "limit": "limit",
        "pricing": "price",
        "segment": "segment",
    }[default_action.type]
    default_value = (
        default_action.value
        if default_action.type in {"limit", "pricing", "segment"}
        else default_action.output_value
    )
    rows.append(
        {
            "condition": "未命中任何规则（默认动作）",
            "decision": default_decision,
            "value": _jsonable(default_value),
            "rule_id": "__default__",
            "priority": None,
            "reason_code": default_action.reason_code,
        }
    )
    return rows


def _require_typed_adoption_quality(result: StrategyBacktestResult) -> None:
    if result.population_count <= 0:
        raise StrategyError("cannot adopt strategy from an empty backtest population")
    if result.labeled_count <= 0:
        raise StrategyError(
            "cannot adopt strategy without labeled backtest observations"
        )

    metrics = result.metrics
    if result.strategy_type == "approval":
        if metrics.get("approve_bad_rate") is None:
            raise StrategyError(
                "cannot adopt strategy because approved bad rate is undefined; "
                "provide labeled approved observations and rerun the backtest"
            )
        return
    if result.strategy_type == "reject":
        if metrics.get("bad_capture_rate") is None:
            raise StrategyError(
                "cannot adopt reject strategy because bad capture rate is undefined"
            )
        if metrics.get("good_reject_rate") is None:
            raise StrategyError(
                "cannot adopt reject strategy because good reject rate is undefined"
            )
        return
    if result.strategy_type == "limit":
        if metrics.get("mean_limit") is None or set(result.economics) != {
            "expected_ead",
            "expected_loss",
        }:
            raise StrategyError(
                "limit adoption requires complete limit economics evidence"
            )
        return
    if result.strategy_type == "pricing":
        required = {
            "total_ead",
            "ead_weighted_rate",
            "revenue",
            "expected_loss",
            "funding_cost",
            "operating_cost",
            "profit",
            "roa",
            "baseline_profit",
            "profit_delta_vs_baseline",
            "by_row",
        }
        if (
            metrics.get("mean_rate") is None
            or set(result.economics) != required
            or result.economics.get("total_ead") in {None, 0.0}
            or result.economics.get("profit") is None
            or result.economics.get("roa") is None
        ):
            raise StrategyError(
                "pricing adoption requires complete pricing economics evidence"
            )
        return
    if metrics.get("segment_count", 0) <= 0 or metrics.get("overall_bad_rate") is None:
        raise StrategyError(
            "segmentation adoption requires non-empty labeled segment evidence"
        )


def _backtest_binding(runtime, backtest_id: str) -> dict[str, str | None]:
    from marvis.db_schema import connect

    with connect(runtime.settings.db_path) as conn:
        row = conn.execute(
            """
            SELECT b.strategy_id, b.dataset_id, d.task_id AS dataset_task_id
              FROM backtests b
              LEFT JOIN datasets d ON d.id = b.dataset_id
             WHERE b.id = ?
            """,
            (backtest_id,),
        ).fetchone()
        audit_row = conn.execute(
            """
            SELECT detail_json
              FROM audit
             WHERE kind = 'strategy.backtest'
               AND target_ref = ?
               AND outcome = 'succeeded'
             ORDER BY at DESC, id DESC
             LIMIT 1
            """,
            (backtest_id,),
        ).fetchone()
    if row is None:
        raise StrategyError(f"backtest not found: {backtest_id}")
    dataset_id = str(row["dataset_id"])
    dataset_content_hash = None
    if row["dataset_task_id"] is not None:
        try:
            dataset_content_hash = sha256_file(
                runtime.registry.resolve_path(dataset_id)
            )
        except (KeyError, OSError):
            # Typed adoption turns this into a hard failure. Legacy rows keep a
            # nullable provenance field because historical fixtures and migrated
            # databases may no longer have the original registered file.
            dataset_content_hash = None
    backtest_dataset_content_hash = None
    if audit_row is not None:
        try:
            audit_detail = json.loads(str(audit_row["detail_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            audit_detail = None
        if isinstance(audit_detail, dict):
            candidate = audit_detail.get("source_dataset_content_hash")
            if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate):
                backtest_dataset_content_hash = candidate
    return {
        "strategy_id": str(row["strategy_id"]),
        "dataset_id": dataset_id,
        "dataset_task_id": (
            None if row["dataset_task_id"] is None else str(row["dataset_task_id"])
        ),
        "dataset_content_hash": dataset_content_hash,
        "backtest_dataset_content_hash": backtest_dataset_content_hash,
    }


def _strategy_monitoring_experiment_id(
    runtime,
    value,
    *,
    task_id: str,
) -> str | None:
    experiment_id = _optional_str(value)
    if experiment_id is None:
        return None
    from marvis.db_schema import connect

    with connect(runtime.settings.db_path) as conn:
        row = conn.execute(
            "SELECT task_id FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError(
            "monitoring experiment must exist and belong to the same task as the strategy"
        )
    return experiment_id


def _aggregate_economics(economics) -> dict:
    return {
        str(key): _jsonable(value)
        for key, value in economics.items()
        if key != "by_row"
    }


def _approval_adoption_audit_summary(approval_metrics: dict | None) -> dict:
    if approval_metrics is None:
        return {}
    return {
        "approval_rate": approval_metrics["approval_rate"],
        "approved_bad_rate": approval_metrics["approved_bad_rate"],
        "expected_profit": approval_metrics["expected_profit"],
    }


def _build_adoption_monitoring_plan(
    *,
    strategy_id: str,
    strategy_type: str,
    version: int,
    evidence: dict,
    approval_metrics: dict | None,
    experiment_id: str | None,
) -> MonitoringPlan:
    monitoring_plan_id = uuid.uuid4().hex
    economics_bindings = _adoption_monitoring_economics_bindings(
        strategy_type,
        evidence=evidence,
    )
    baseline = {
        "strategy_type": strategy_type,
        "backtest_schema_version": evidence["backtest_schema_version"],
        "strategy_effect_hash": evidence["strategy_effect_hash"],
        "baseline_effect_hash": evidence["baseline_effect_hash"],
        "source_dataset_id": evidence["source_dataset_id"],
        "source_dataset_content_hash": evidence["source_dataset_content_hash"],
        "source_backtest_id": evidence["backtest_id"],
        "population_count": evidence["population_count"],
        "labeled_count": evidence["labeled_count"],
        "label_coverage": evidence["label_coverage"],
        "metrics": dict(evidence["metrics"]),
        "economics": dict(evidence["economics"]),
        "breakdown": [dict(row) for row in evidence["breakdown"]],
        "transitions": [dict(row) for row in evidence["transitions"]],
    }
    if (
        strategy_type == "approval"
        or evidence["backtest_schema_version"] == _LEGACY_BACKTEST_SCHEMA_VERSION
    ):
        assert approval_metrics is not None
        plan = build_monitoring_plan(
            strategy_id=strategy_id,
            version=version,
            approved_bad_rate=float(approval_metrics["approved_bad_rate"]),
            approval_rate=float(approval_metrics["approval_rate"]),
            experiment_id=experiment_id,
            source_backtest_id=evidence["backtest_id"],
            monitoring_plan_id=monitoring_plan_id,
            revision=1,
            supersedes_plan_id=None,
            economics_bindings=economics_bindings,
        )
        plan["thresholds"] = _with_model_monitoring_thresholds(
            plan["thresholds"],
            experiment_id=experiment_id,
        )
        plan["expectation_baseline"].update(baseline)
        return monitoring_plan_from_dict(plan, source="adoption")

    thresholds = _typed_monitoring_thresholds(
        strategy_type,
        evidence=evidence,
        approval_metrics=approval_metrics,
    )
    if approval_metrics is not None:
        baseline.update(
            {
                "approval_rate": approval_metrics["approval_rate"],
                "approved_bad_rate": approval_metrics["approved_bad_rate"],
            }
        )
    return monitoring_plan_from_dict(
        {
            "plan_version": PLAN_VERSION,
            "monitoring_plan_id": monitoring_plan_id,
            "strategy_id": strategy_id,
            "version": int(version),
            "revision": 1,
            "supersedes_plan_id": None,
            "cadence_days": DEFAULT_CADENCE_DAYS,
            "experiment_id": experiment_id,
            "last_run_at": None,
            "thresholds": _with_model_monitoring_thresholds(
                thresholds,
                experiment_id=experiment_id,
            ),
            "expectation_baseline": baseline,
            "economics_bindings": economics_bindings,
        },
        source="adoption",
    )


def _with_model_monitoring_thresholds(
    strategy_thresholds: Mapping[str, Any],
    *,
    experiment_id: str | None,
) -> dict[str, dict]:
    combined = {
        str(check_id): dict(spec) for check_id, spec in strategy_thresholds.items()
    }
    if experiment_id is None:
        return combined
    from marvis.packs.modeling.monitor_tools import MONITOR_RUN_THRESHOLDS

    collisions = sorted(set(combined) & set(MONITOR_RUN_THRESHOLDS))
    if collisions:
        raise StrategyError(
            "strategy and model monitoring threshold ids collide: "
            + ", ".join(collisions)
        )
    combined.update(
        {str(check_id): dict(spec) for check_id, spec in MONITOR_RUN_THRESHOLDS.items()}
    )
    return combined


def _adoption_monitoring_economics_bindings(
    strategy_type: str,
    *,
    evidence: dict,
) -> dict[str, dict]:
    """Project typed backtest economics identity into safe rerun bindings.

    Scalar assumptions can be frozen directly. Series evidence may only retain
    its stable dataset column name: row counts, row-level hashes, and values are
    deliberately excluded from the monitoring plan. A nameless series cannot be
    reproduced against a fresh monitoring dataset, so adoption fails closed.
    """

    if strategy_type not in {"limit", "pricing"}:
        return {}
    raw_bindings = evidence.get("economics_input_evidence")
    if not isinstance(raw_bindings, dict):
        raise StrategyError(
            f"{strategy_type} adoption requires economics input evidence"
        )
    bindings: dict[str, dict] = {}
    for raw_name, raw_evidence in sorted(raw_bindings.items()):
        name = str(raw_name)
        if not isinstance(raw_evidence, dict):
            raise StrategyError(f"economics input evidence {name} must be an object")
        kind = raw_evidence.get("kind")
        if kind == "scalar":
            bindings[name] = {
                "kind": "scalar",
                "value": raw_evidence.get("value"),
            }
            continue
        if kind == "series":
            column = raw_evidence.get("name")
            if not isinstance(column, str) or not column.strip():
                raise StrategyError(
                    f"{strategy_type} economics input {name} requires a stable column name "
                    "before adoption"
                )
            bindings[name] = {"kind": "column", "column": column.strip()}
            continue
        raise StrategyError(
            f"economics input evidence {name} has unsupported kind {kind!r}"
        )
    return bindings


_MONITORING_DISPOSITION_AUDIT_KIND = "strategy.monitoring.disposition"
_MONITORING_DISPOSITIONS = {"observe", "adjust_threshold", "new_version"}


@dataclass(frozen=True)
class _MonitoringDispositionContext:
    task_id: str
    strategy_id: str
    monitoring_plan: MonitoringPlan
    monitoring_plan_id: str
    monitoring_plan_revision: int
    monitoring_plan_hash: str
    monitoring_run_id: str
    monitoring_run_result: dict
    monitoring_run_result_hash: str
    overall_level: str
    checks: list[dict]


def tool_apply_monitoring_disposition(inputs: dict, ctx) -> dict:
    """Apply one governed response to the latest immutable monitoring result.

    Callers identify the already-computed run and its latest plan receipt. They
    never supply metrics or a verdict. Observe/acknowledge and new-version
    handoff commit their disposition receipt in one writer transaction. Threshold
    adjustment delegates the candidate plan to the monitoring runtime, which
    must rerun and persist the new plan/run/receipt atomically.
    """

    if "metrics" in inputs:
        raise StrategyError(
            "metrics must not be supplied to apply_monitoring_disposition"
        )
    task_id = str(ctx.task_id)
    strategy_id = _required_disposition_text(
        inputs.get("strategy_id"), field="strategy_id"
    )
    monitoring_run_id = _required_disposition_text(
        inputs.get("monitoring_run_id"), field="monitoring_run_id"
    )
    expected_plan_id = _required_disposition_text(
        inputs.get("expected_plan_id"), field="expected_plan_id"
    )
    expected_plan_revision = _positive_disposition_integer(
        inputs.get("expected_plan_revision"), field="expected_plan_revision"
    )
    expected_plan_hash = _required_disposition_sha256(
        inputs.get("expected_plan_hash"), field="expected_plan_hash"
    )
    disposition = _monitoring_disposition(inputs.get("disposition"))
    reason = _required_disposition_text(inputs.get("reason"), field="reason")
    raw_threshold_patch = inputs.get("threshold_patch")
    if disposition != "adjust_threshold" and raw_threshold_patch is not None:
        raise StrategyError(
            "threshold_patch is only valid for adjust_threshold disposition"
        )

    runtime = _runtime(ctx)
    with connect(runtime.settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        disposition_context = _monitoring_disposition_context_on_connection(
            conn,
            task_id=task_id,
            strategy_id=strategy_id,
            monitoring_run_id=monitoring_run_id,
            expected_plan_id=expected_plan_id,
            expected_plan_revision=expected_plan_revision,
            expected_plan_hash=expected_plan_hash,
        )
        _validate_monitoring_disposition_level(
            disposition_context.overall_level,
            disposition=disposition,
        )
        _reject_monitoring_disposition_replay(
            conn,
            monitoring_run_id=monitoring_run_id,
        )

        if disposition == "adjust_threshold":
            normalized_patch, candidate_plan = _monitoring_threshold_candidate(
                disposition_context,
                raw_threshold_patch,
            )
        else:
            normalized_patch = None
            candidate_plan = None

        if disposition == "new_version":
            handoff = StrategyHandoffRepository(
                runtime.settings.db_path,
                runtime.datasets_root,
            ).create_new_version_from_red_run_on_connection(
                conn,
                source_task_id=task_id,
                parent_strategy_id=strategy_id,
                monitoring_run_id=monitoring_run_id,
            )
            status = "new_version_created"
        elif disposition in {None, "observe"}:
            handoff = None
            status = "acknowledged" if disposition is None else "observed"
        else:
            handoff = None
            status = "threshold_adjusted"

        if disposition != "adjust_threshold":
            _write_monitoring_disposition_audit(
                conn,
                disposition_context=disposition_context,
                disposition=disposition,
                status=status,
                reason=reason,
                handoff=handoff,
            )

    if disposition == "adjust_threshold":
        assert candidate_plan is not None
        assert normalized_patch is not None
        adjusted = _rerun_monitoring_candidate_atomically(
            ctx=ctx,
            strategy_id=strategy_id,
            source_monitoring_run_id=monitoring_run_id,
            expected_latest_plan_id=expected_plan_id,
            expected_latest_plan_revision=expected_plan_revision,
            expected_latest_plan_hash=expected_plan_hash,
            candidate_plan=candidate_plan,
            reason=reason,
            threshold_patch=normalized_patch,
        )
        return _adjusted_monitoring_disposition_output(
            strategy_id=strategy_id,
            source_monitoring_run_id=monitoring_run_id,
            reason=reason,
            adjusted=adjusted,
        )

    return _monitoring_disposition_output(
        disposition_context,
        disposition=disposition,
        status=status,
        reason=reason,
        handoff=handoff,
    )


def _monitoring_disposition_context_on_connection(
    conn,
    *,
    task_id: str,
    strategy_id: str,
    monitoring_run_id: str,
    expected_plan_id: str,
    expected_plan_revision: int,
    expected_plan_hash: str,
) -> _MonitoringDispositionContext:
    strategy_row = conn.execute(
        "SELECT id, task_id, version, status, asset_status FROM strategies WHERE id = ?",
        (strategy_id,),
    ).fetchone()
    if strategy_row is None or str(strategy_row["task_id"]) != task_id:
        raise StrategyError(f"strategy not found: {strategy_id}")
    if not is_locally_adopted(strategy_row["status"], strategy_row["asset_status"]):
        raise StrategyError(
            "monitoring disposition requires an adopted strategy with "
            "asset_status=adopted_local; "
            "local adoption is not production deployment"
        )

    plan_row = conn.execute(
        """
        SELECT * FROM strategy_monitoring_plans
         WHERE strategy_id = ?
         ORDER BY revision DESC, created_at DESC, id DESC
         LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
    if plan_row is None:
        raise StrategyError("strategy has no immutable monitoring plan")
    actual_plan_id = str(plan_row["id"])
    actual_plan_revision = int(plan_row["revision"])
    actual_plan_hash = _required_disposition_sha256(
        plan_row["payload_hash"], field="stored monitoring plan hash"
    )
    if (
        actual_plan_id != expected_plan_id
        or actual_plan_revision != expected_plan_revision
        or not hmac.compare_digest(actual_plan_hash, expected_plan_hash)
    ):
        raise StrategyError(
            "monitoring plan CAS mismatch: expected id/revision/hash is stale"
        )
    monitoring_plan = _monitoring_plan_from_ledger_row(plan_row)
    if monitoring_plan.version != int(strategy_row["version"]):
        raise StrategyError(
            "monitoring plan does not bind the adopted strategy version"
        )

    run_row = conn.execute(
        "SELECT * FROM strategy_monitoring_runs WHERE id = ?",
        (monitoring_run_id,),
    ).fetchone()
    if run_row is None or str(run_row["strategy_id"]) != strategy_id:
        raise StrategyError(f"monitoring run not found: {monitoring_run_id}")
    latest_run = conn.execute(
        """
        SELECT id FROM strategy_monitoring_runs
         WHERE strategy_id = ?
         ORDER BY created_at DESC, id DESC
         LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
    if latest_run is None or str(latest_run["id"]) != monitoring_run_id:
        raise StrategyError("disposition requires the latest monitoring run")
    if str(run_row["monitoring_plan_id"]) != actual_plan_id:
        raise StrategyError("latest monitoring run does not bind the latest plan")
    dataset_row = conn.execute(
        "SELECT task_id FROM datasets WHERE id = ?",
        (str(run_row["dataset_id"]),),
    ).fetchone()
    if dataset_row is None or str(dataset_row["task_id"]) != task_id:
        raise StrategyError(
            "monitoring run dataset does not belong to the strategy task"
        )
    result, result_hash = _validated_monitoring_run_result(run_row)
    level = str(run_row["overall_level"])
    if result.get("overall_level") != level:
        raise StrategyError("monitoring run result level does not match its ledger row")
    checks = result.get("checks")
    if not isinstance(checks, list) or any(
        not isinstance(check, dict) for check in checks
    ):
        raise StrategyError("monitoring run result must contain a checks array")
    return _MonitoringDispositionContext(
        task_id=task_id,
        strategy_id=strategy_id,
        monitoring_plan=monitoring_plan,
        monitoring_plan_id=actual_plan_id,
        monitoring_plan_revision=actual_plan_revision,
        monitoring_plan_hash=actual_plan_hash,
        monitoring_run_id=monitoring_run_id,
        monitoring_run_result=result,
        monitoring_run_result_hash=result_hash,
        overall_level=level,
        checks=[dict(check) for check in checks],
    )


def _monitoring_plan_from_ledger_row(row) -> MonitoringPlan:
    try:
        payload = json.loads(str(row["payload_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("stored monitoring plan payload is invalid") from exc
    if not isinstance(payload, dict):
        raise StrategyError("stored monitoring plan payload must be an object")
    plan = monitoring_plan_from_dict(payload, source=f"database:{row['id']}")
    calculated_hash = canonical_monitoring_plan_hash(plan)
    stored_hash = _required_disposition_sha256(
        row["payload_hash"], field="stored monitoring plan hash"
    )
    if not hmac.compare_digest(calculated_hash, stored_hash):
        raise StrategyError("stored monitoring plan payload hash is invalid")
    if (
        plan.monitoring_plan_id != str(row["id"])
        or plan.strategy_id != str(row["strategy_id"])
        or plan.version != int(row["strategy_version"])
        or plan.revision != int(row["revision"])
        or plan.supersedes_plan_id
        != (
            None
            if row["supersedes_plan_id"] is None
            else str(row["supersedes_plan_id"])
        )
    ):
        raise StrategyError("stored monitoring plan columns do not match its payload")
    return plan


def _validated_monitoring_run_result(row) -> tuple[dict, str]:
    try:
        result = json.loads(str(row["result_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("stored monitoring run result is invalid") from exc
    if not isinstance(result, dict):
        raise StrategyError("stored monitoring run result must be an object")
    try:
        canonical = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "stored monitoring run result is not canonical JSON"
        ) from exc
    calculated_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    stored_hash = _required_disposition_sha256(
        row["result_hash"], field="stored monitoring run result hash"
    )
    if not hmac.compare_digest(calculated_hash, stored_hash):
        raise StrategyError("stored monitoring run result hash is invalid")
    try:
        validate_monitoring_run_result(
            result,
            overall_level=row["overall_level"],
        )
    except StrategyMonitoringDataError as exc:
        raise StrategyError(
            "stored monitoring run result violates the semantic contract"
        ) from exc
    return result, stored_hash


def _validate_monitoring_disposition_level(
    overall_level: str,
    *,
    disposition: str | None,
) -> None:
    if overall_level == "red":
        if disposition is None:
            raise StrategyError(
                "red monitoring run requires an explicit disposition; null is not observe"
            )
        return
    if disposition is not None:
        raise StrategyError(
            "only red monitoring runs accept observe, adjust_threshold, or new_version"
        )


def _reject_monitoring_disposition_replay(
    conn,
    *,
    monitoring_run_id: str,
) -> None:
    replay = conn.execute(
        "SELECT 1 FROM audit WHERE kind = ? AND target_ref = ? LIMIT 1",
        (_MONITORING_DISPOSITION_AUDIT_KIND, monitoring_run_id),
    ).fetchone()
    if replay is not None:
        raise StrategyError(
            f"monitoring run {monitoring_run_id} already has a disposition"
        )


def _monitoring_threshold_candidate(
    disposition_context: _MonitoringDispositionContext,
    raw_patch,
) -> tuple[dict[str, dict[str, float]], MonitoringPlan]:
    if not isinstance(raw_patch, dict) or not raw_patch:
        raise StrategyError(
            "adjust_threshold disposition requires a non-empty threshold_patch"
        )
    thresholds = {
        str(check_id): dict(spec)
        for check_id, spec in disposition_context.monitoring_plan.thresholds.items()
        if isinstance(spec, dict)
    }
    normalized_patch: dict[str, dict[str, float]] = {}
    for raw_check_id, raw_changes in raw_patch.items():
        check_id = str(raw_check_id)
        if check_id not in thresholds:
            raise StrategyError(
                f"unknown monitoring check in threshold_patch: {check_id}"
            )
        if not isinstance(raw_changes, dict) or not raw_changes:
            raise StrategyError(
                f"threshold_patch.{check_id} must be a non-empty object"
            )
        unsupported = sorted(set(raw_changes) - {"warn", "fail"})
        if unsupported:
            raise StrategyError(
                "threshold_patch may only change warn/fail; unsupported fields: "
                + ", ".join(unsupported)
            )
        changes = {
            field: _finite_disposition_number(
                value,
                field=f"threshold_patch.{check_id}.{field}",
            )
            for field, value in raw_changes.items()
        }
        candidate_spec = {**thresholds[check_id], **changes}
        direction = candidate_spec.get("direction")
        if direction not in {"min", "max"}:
            raise StrategyError(
                f"monitoring check {check_id} has unsupported direction"
            )
        warn = _finite_disposition_number(
            candidate_spec.get("warn"), field=f"monitoring check {check_id}.warn"
        )
        fail = _finite_disposition_number(
            candidate_spec.get("fail"), field=f"monitoring check {check_id}.fail"
        )
        if direction == "min" and warn < fail:
            raise StrategyError(
                f"monitoring check {check_id} min direction requires warn >= fail"
            )
        if direction == "max" and warn > fail:
            raise StrategyError(
                f"monitoring check {check_id} max direction requires warn <= fail"
            )
        normalized_candidate = {**candidate_spec, "warn": warn, "fail": fail}
        effective_changes = {
            field: normalized_candidate[field]
            for field in changes
            if thresholds[check_id].get(field) != normalized_candidate[field]
        }
        if effective_changes:
            thresholds[check_id] = normalized_candidate
            normalized_patch[check_id] = effective_changes

    if not normalized_patch:
        raise StrategyError("threshold_patch must change at least one warn/fail value")

    candidate = replace(
        disposition_context.monitoring_plan,
        monitoring_plan_id=uuid.uuid4().hex,
        revision=disposition_context.monitoring_plan_revision + 1,
        supersedes_plan_id=disposition_context.monitoring_plan_id,
        last_run_at=None,
        thresholds=thresholds,
    )
    return normalized_patch, candidate


def _rerun_monitoring_candidate_atomically(**kwargs) -> dict:
    from marvis.packs.strategy.monitor_tools import (
        rerun_strategy_monitoring_with_candidate_plan,
    )

    return rerun_strategy_monitoring_with_candidate_plan(**kwargs)


def _write_monitoring_disposition_audit(
    conn,
    *,
    disposition_context: _MonitoringDispositionContext,
    disposition: str | None,
    status: str,
    reason: str,
    handoff: dict | None,
) -> None:
    detail = {
        "receipt_schema_version": "strategy.monitoring-disposition.v1",
        "task_id": disposition_context.task_id,
        "strategy_id": disposition_context.strategy_id,
        "source_monitoring_run_id": disposition_context.monitoring_run_id,
        "resolved_monitoring_run_id": disposition_context.monitoring_run_id,
        "monitoring_run_result_hash": disposition_context.monitoring_run_result_hash,
        "monitoring_plan_id": disposition_context.monitoring_plan_id,
        "monitoring_plan_revision": disposition_context.monitoring_plan_revision,
        "monitoring_plan_hash": disposition_context.monitoring_plan_hash,
        "disposition": disposition,
        "status": status,
        "reason": reason,
    }
    if handoff is not None:
        detail.update(
            {
                "new_task_id": handoff["new_task_id"],
                "new_strategy_id": handoff["new_strategy_id"],
                "new_dataset_id": handoff["new_dataset_id"],
            }
        )
    _write_audit_row(
        conn,
        kind=_MONITORING_DISPOSITION_AUDIT_KIND,
        target_ref=disposition_context.monitoring_run_id,
        inputs_hash=_monitoring_disposition_inputs_hash(detail),
        outcome="succeeded",
        detail=detail,
    )


def _monitoring_disposition_inputs_hash(detail: dict) -> str:
    raw = json.dumps(
        detail,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _monitoring_disposition_output(
    disposition_context: _MonitoringDispositionContext,
    *,
    disposition: str | None,
    status: str,
    reason: str,
    handoff: dict | None,
) -> dict:
    return {
        "strategy_id": disposition_context.strategy_id,
        "source_monitoring_run_id": disposition_context.monitoring_run_id,
        "disposition": disposition,
        "status": status,
        "reason": reason,
        "overall_level": disposition_context.overall_level,
        "checks": list(disposition_context.checks),
        "monitoring_plan_id": disposition_context.monitoring_plan_id,
        "monitoring_plan_revision": disposition_context.monitoring_plan_revision,
        "monitoring_plan_hash": disposition_context.monitoring_plan_hash,
        "resolved_monitoring_run_id": disposition_context.monitoring_run_id,
        "new_task_id": None if handoff is None else handoff["new_task_id"],
        "new_strategy_id": None if handoff is None else handoff["new_strategy_id"],
        "new_dataset_id": None if handoff is None else handoff["new_dataset_id"],
        "plan_artifact_path": None,
    }


def _adjusted_monitoring_disposition_output(
    *,
    strategy_id: str,
    source_monitoring_run_id: str,
    reason: str,
    adjusted: dict,
) -> dict:
    return {
        "strategy_id": strategy_id,
        "source_monitoring_run_id": source_monitoring_run_id,
        "disposition": "adjust_threshold",
        "status": "threshold_adjusted",
        "reason": reason,
        "overall_level": str(adjusted["overall_level"]),
        "checks": [dict(check) for check in adjusted["checks"]],
        "monitoring_plan_id": str(adjusted["monitoring_plan_id"]),
        "monitoring_plan_revision": int(adjusted["monitoring_plan_revision"]),
        "monitoring_plan_hash": str(adjusted["monitoring_plan_hash"]),
        "resolved_monitoring_run_id": str(adjusted["monitoring_run_id"]),
        "new_task_id": None,
        "new_strategy_id": None,
        "new_dataset_id": None,
        "plan_artifact_path": (
            None
            if adjusted.get("plan_artifact_path") is None
            else str(adjusted["plan_artifact_path"])
        ),
    }


def _monitoring_disposition(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise StrategyError("disposition must be null or a supported value")
    normalized = value.strip()
    if normalized not in _MONITORING_DISPOSITIONS:
        allowed = ", ".join(sorted(_MONITORING_DISPOSITIONS))
        raise StrategyError(f"unsupported monitoring disposition; expected {allowed}")
    return normalized


def _required_disposition_text(value, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"{field} must be non-empty")
    return value.strip()


def _positive_disposition_integer(value, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StrategyError(f"{field} must be a positive integer")
    return value


def _required_disposition_sha256(value, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise StrategyError(f"{field} must be a sha256 hash")
    return value.lower()


def _finite_disposition_number(value, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise StrategyError(f"{field} must be a finite number")
    return number


def _typed_monitoring_thresholds(
    strategy_type: str,
    *,
    evidence: dict,
    approval_metrics: dict | None,
) -> dict:
    metrics = evidence["metrics"]
    economics = evidence["economics"]
    if strategy_type == "reject":
        assert approval_metrics is not None
        approval_rate = float(approval_metrics["approval_rate"])
        bad_capture_rate = float(metrics["bad_capture_rate"])
        good_reject_rate = float(metrics["good_reject_rate"])
        thresholds = {
            "approval_rate": _monitor_threshold(
                "审批率下滑",
                "approval_rate",
                "min",
                max(0.0, approval_rate - 0.05),
                max(0.0, approval_rate - 0.10),
            ),
            "bad_capture_rate": _monitor_threshold(
                "坏客户捕获率下滑",
                "bad_capture_rate",
                "min",
                max(0.0, bad_capture_rate - 0.05),
                max(0.0, bad_capture_rate - 0.10),
            ),
            "good_reject_rate": _monitor_threshold(
                "好客户误拒率上升",
                "good_reject_rate",
                "max",
                min(1.0, good_reject_rate + 0.02),
                min(1.0, good_reject_rate + 0.05),
            ),
        }
        approved_bad_rate = approval_metrics["approved_bad_rate"]
        if approved_bad_rate is not None:
            value = float(approved_bad_rate)
            thresholds["approved_bad_rate"] = _monitor_threshold(
                "通过客群坏率漂移",
                "approved_bad_rate",
                "max",
                min(1.0, value + 0.02),
                min(1.0, value + 0.05),
            )
        return thresholds

    if strategy_type == "limit":
        mean_limit = float(metrics["mean_limit"])
        expected_loss = float(economics["expected_loss"])
        return {
            "mean_limit": _monitor_threshold(
                "户均额度上升",
                "mean_limit",
                "max",
                mean_limit * 1.10,
                mean_limit * 1.20,
            ),
            "expected_loss": _monitor_threshold(
                "额度策略预期损失上升",
                "expected_loss",
                "max",
                expected_loss * 1.10,
                expected_loss * 1.20,
            ),
        }

    if strategy_type == "pricing":
        mean_rate = float(metrics["mean_rate"])
        expected_loss = float(economics["expected_loss"])
        profit = float(economics["profit"])
        roa = float(economics["roa"])
        return {
            "mean_rate": _monitor_threshold(
                "平均利率上升",
                "mean_rate",
                "max",
                min(1.0, mean_rate + 0.02),
                min(1.0, mean_rate + 0.05),
            ),
            "expected_loss": _monitor_threshold(
                "定价策略预期损失上升",
                "expected_loss",
                "max",
                expected_loss * 1.10,
                expected_loss * 1.20,
            ),
            "profit": _monitor_threshold(
                "利润下滑",
                "profit",
                "min",
                profit - abs(profit) * 0.10,
                profit - abs(profit) * 0.20,
            ),
            "roa": _monitor_threshold(
                "ROA 下滑",
                "roa",
                "min",
                roa - 0.01,
                roa - 0.02,
            ),
        }

    overall_bad_rate = float(metrics["overall_bad_rate"])
    return {
        "overall_bad_rate": _monitor_threshold(
            "分群总体坏率上升",
            "overall_bad_rate",
            "max",
            min(1.0, overall_bad_rate + 0.02),
            min(1.0, overall_bad_rate + 0.05),
        ),
        "segment_share_psi": _monitor_threshold(
            "分群占比漂移",
            "segment_share_psi",
            "max",
            0.10,
            0.25,
        ),
    }


def _monitor_threshold(
    label: str,
    metric: str,
    direction: str,
    warn: float,
    fail: float,
) -> dict:
    return {
        "label": label,
        "metric": metric,
        "direction": direction,
        "warn": float(warn),
        "fail": float(fail),
    }


def tool_render_challenger_report(inputs: dict, ctx) -> dict:
    """Render a challenger report only from task-owned persisted evidence."""
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id, task_id=task_id)
    strategy_meta = runtime.strategies.get_strategy_meta(strategy_id)
    assert strategy_meta is not None
    champion_id = _optional_str(inputs.get("champion_strategy_id"))

    if not champion_id:
        markdown = "# 挑战者对比报告\n\n未提供基线（champion）策略，跳过对比报告。\n"
        return {
            "status": "no_baseline",
            "report_md": markdown,
            "artifacts": [],
        }
    champion = _strategy(runtime, champion_id, task_id=task_id)
    if champion.strategy_type != strategy.strategy_type:
        raise StrategyError("challenger and champion strategy types must match")
    if strategy.strategy_type not in {"approval", "reject"}:
        raise StrategyError(
            "challenger report currently requires approval or reject strategies"
        )
    trusted = _trusted_challenger_report_evidence(
        runtime,
        task_id=task_id,
        strategy=strategy,
        champion=champion,
        challenger_backtest_carrier=inputs.get("challenger_backtest"),
    )
    adopted = is_locally_adopted(strategy_meta["status"], strategy_meta["asset_status"])
    markdown = _challenger_report_markdown(
        strategy_id=strategy_id,
        champion_id=champion_id,
        compare=trusted["compare"],
        challenger_backtest=trusted["challenger_backtest"],
        champion_backtest=trusted["champion_backtest"],
        adopted=adopted,
    )
    artifact = _persist_verified_strategy_markdown(
        runtime,
        ctx,
        strategy_id=strategy_id,
        kind="challenger_report_md",
        filename_prefix=f"challenger_report_{strategy_id}_vs_{champion_id}",
        markdown=markdown,
        evidence=trusted["provenance"],
    )
    return {
        "status": "rendered",
        "report_md": markdown,
        "report_path": artifact["path"],
        "artifacts": [artifact],
    }


def _trusted_challenger_report_evidence(
    runtime,
    *,
    task_id: str,
    strategy: Strategy,
    champion: Strategy,
    challenger_backtest_carrier: object,
) -> dict[str, Any]:
    carrier = _as_dict(challenger_backtest_carrier)
    backtest_id = _optional_str(carrier.get("backtest_id"))
    if backtest_id is None:
        raise StrategyError(
            "challenger report requires a persisted challenger backtest_id"
        )
    persisted = runtime.strategies.get_backtest(backtest_id)
    if not isinstance(persisted, StrategyBacktestResult):
        raise StrategyError(f"typed backtest not found: {backtest_id}")
    binding = _backtest_binding(runtime, backtest_id)
    if binding["strategy_id"] != strategy.id:
        raise StrategyError(
            "challenger backtest does not belong to challenger strategy"
        )
    if binding["dataset_task_id"] != task_id:
        raise StrategyError(
            "challenger backtest dataset does not belong to strategy task"
        )
    current_hash = binding["dataset_content_hash"]
    evidence_hash = binding["backtest_dataset_content_hash"]
    if (
        current_hash is None
        or evidence_hash is None
        or not hmac.compare_digest(current_hash, evidence_hash)
    ):
        raise StrategyError(
            "challenger backtest dataset evidence is missing or changed"
        )
    normalized = dict(persisted.normalized_input)
    challenger_effect_hash = runtime.strategies.get_strategy_spec_hash(strategy.id)
    champion_effect_hash = runtime.strategies.get_strategy_spec_hash(champion.id)
    if challenger_effect_hash is None or champion_effect_hash is None:
        raise StrategyError("challenger report strategy evidence is missing")
    if normalized.get("strategy_effect_hash") != challenger_effect_hash:
        raise StrategyError("challenger backtest strategy effect is no longer current")
    if normalized.get("baseline_effect_hash") != champion_effect_hash:
        raise StrategyError("challenger backtest is not bound to the supplied champion")

    dataset_id = str(binding["dataset_id"])
    dataset = _owned_dataset(runtime, dataset_id, task_id=task_id)
    source_path = runtime.registry.resolve_path(dataset.id)
    before_hash = sha256_file(source_path)
    if not hmac.compare_digest(before_hash, evidence_hash):
        raise StrategyError("challenger backtest dataset content no longer matches")
    frame = runtime.backend.read_frame(source_path)
    if not hmac.compare_digest(sha256_file(source_path), evidence_hash):
        raise StrategyError("challenger backtest dataset changed while reading")
    target_col = str(normalized.get("target_col") or "")
    if not target_col:
        raise StrategyError("challenger backtest target binding is missing")
    target_bad_value = int(normalized.get("target_encoding", {}).get("bad", 1))
    sample_design_ref = normalized.get("sample_design_ref")
    sample_binding = None
    if sample_design_ref is not None:
        reference = StrategySampleDesignRef.from_value(sample_design_ref)
        artifact = load_strategy_sample_design_artifact(
            runtime,
            task_id=task_id,
            artifact_id=reference.artifact_id,
            expected_artifact_content_hash=reference.artifact_content_hash,
            expected_sample_design_id=reference.sample_design_id,
            expected_sample_design_content_hash=(
                reference.sample_design_content_hash
            ),
        )
        design = artifact.bundle["sample_design"]
        workspace = DataWorkspaceRepository(
            runtime.settings.db_path
        ).get_or_default(task_id)
        sample_binding = load_strategy_sample_design_execution_binding(
            runtime,
            task_id=task_id,
            sample_design_ref=reference.to_ref_dict(),
            dataset_id=dataset.id,
            dataset_content_hash=evidence_hash,
            workspace_revision=workspace.revision,
            workspace_generation=workspace.analysis_generation,
            semantic_mapping_hash=data_semantic_mapping_hash(
                workspace.semantic_mapping
            ),
            target_col=target_col,
            drop_nan_labels=bool(
                design["target_definition"]["drop_nan_labels"]
            ),
        )
        frame = bind_strategy_development_frame(
            frame,
            binding=sample_binding,
            normalize_target=False,
        )
        target_bad_value = sample_binding.target_bad_value
    approval_profit_inputs = _approval_profit_inputs_from_backtest(normalized)
    challenger_result = run_typed_backtest(
        frame,
        strategy.spec or legacy_strategy_to_spec(strategy),
        target_col=target_col,
        target_bad_value=target_bad_value,
        sample_design_ref=(
            None if sample_binding is None else sample_binding.to_ref_dict()
        ),
        strategy_id=strategy.id,
        baseline=champion.spec or legacy_strategy_to_spec(champion),
        approval_profit_inputs=approval_profit_inputs,
    )
    if challenger_result.to_dict() != persisted.to_dict():
        raise StrategyError(
            "persisted challenger backtest no longer recomputes exactly"
        )
    champion_result = run_typed_backtest(
        frame,
        champion.spec or legacy_strategy_to_spec(champion),
        target_col=target_col,
        target_bad_value=target_bad_value,
        sample_design_ref=(
            None if sample_binding is None else sample_binding.to_ref_dict()
        ),
        strategy_id=champion.id,
        approval_profit_inputs=approval_profit_inputs,
    )
    comparison = _trusted_approval_comparison(challenger_result, champion_result)
    if sample_binding is not None:
        revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
    return {
        "compare": comparison,
        "challenger_backtest": _approval_report_projection(challenger_result),
        "champion_backtest": _approval_report_projection(champion_result),
        "provenance": {
            "challenger_strategy_effect_hash": challenger_effect_hash,
            "champion_strategy_id": champion.id,
            "champion_strategy_effect_hash": champion_effect_hash,
            "challenger_backtest_id": backtest_id,
            "challenger_backtest_payload_hash": hashlib.sha256(
                json.dumps(
                    persisted.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "dataset_id": dataset_id,
            "dataset_content_hash": evidence_hash,
            "sample_design_ref": (
                None if sample_binding is None else sample_binding.to_ref_dict()
            ),
        },
    }


def _approval_profit_inputs_from_backtest(
    normalized_input: Mapping[str, Any],
) -> ApprovalProfitInputs | None:
    raw = normalized_input.get("approval_profit_input")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise StrategyError("backtest approval profit input is invalid")
    params = raw.get("params")
    if not isinstance(params, Mapping):
        raise StrategyError("backtest approval profit parameters are invalid")
    return ApprovalProfitInputs(
        params=_profit_params(dict(params)),
        ead_col=str(raw.get("ead_col") or ""),
        pd_col=str(raw.get("pd_col") or ""),
    )


def _approval_report_projection(result: StrategyBacktestResult) -> dict[str, Any]:
    return {
        "approval_rate": result.metrics.get("approve_rate"),
        "approved_bad_rate": result.metrics.get("approve_bad_rate"),
        "expected_profit": result.economics.get("expected_profit"),
    }


def _trusted_approval_comparison(
    challenger: StrategyBacktestResult,
    champion: StrategyBacktestResult,
) -> dict[str, Any]:
    challenger_view = _approval_report_projection(challenger)
    champion_view = _approval_report_projection(champion)
    deltas: dict[str, float | None] = {}
    for key in ("approval_rate", "approved_bad_rate", "expected_profit"):
        left = challenger_view[key]
        right = champion_view[key]
        deltas[key] = (
            None if left is None or right is None else float(left) - float(right)
        )
    red_flags: list[dict] = []
    swap_in = _approval_transition_group(
        challenger.transitions,
        lambda row: (
            row.get("from_action") != "approve" and row.get("to_action") == "approve"
        ),
    )
    swap_out = _approval_transition_group(
        challenger.transitions,
        lambda row: (
            row.get("from_action") == "approve" and row.get("to_action") != "approve"
        ),
    )
    if (
        swap_in["count"]
        and swap_out["count"]
        and swap_in["bad_rate"] is not None
        and swap_out["bad_rate"] is not None
        and swap_in["bad_rate"] > swap_out["bad_rate"]
    ):
        red_flags.append(
            {
                "code": "swap_in_worse",
                "level": "red",
                "message": "挑战者换入客群坏率高于换出客群。",
            }
        )
    if deltas["expected_profit"] is not None and deltas["expected_profit"] < 0:
        red_flags.append(
            {
                "code": "profit_negative_delta",
                "level": "amber",
                "message": "挑战者预期利润低于基线。",
            }
        )
    return {
        "deltas": deltas,
        "summary_text": _trusted_challenger_summary(deltas),
        "red_flags": red_flags,
    }


def _approval_transition_group(transitions, predicate) -> dict[str, Any]:
    selected = [row for row in transitions if predicate(row)]
    count = sum(int(row.get("count") or 0) for row in selected)
    labeled = sum(int(row.get("labeled_count") or 0) for row in selected)
    bad = sum(int(row.get("bad_count") or 0) for row in selected)
    return {
        "count": count,
        "bad_rate": None if labeled == 0 else float(bad / labeled),
    }


def _trusted_challenger_summary(deltas: Mapping[str, float | None]) -> str:
    approval = deltas.get("approval_rate")
    bad_rate = deltas.get("approved_bad_rate")
    profit = deltas.get("expected_profit")
    profit_text = (
        "预期利润不可用。"
        if profit is None
        else f"预期利润{_report_delta_word(profit)}{abs(profit):.2f}。"
    )
    return (
        "挑战者审批率较基线"
        + _report_delta_text(approval, percentage_points=True)
        + "，通过客群坏率"
        + _report_delta_text(bad_rate, percentage_points=True, decimals=2)
        + f"，{profit_text}"
    )


def _report_delta_text(
    value: float | None,
    *,
    percentage_points: bool,
    decimals: int = 1,
) -> str:
    if value is None:
        return "不可用"
    number = abs(float(value)) * (100 if percentage_points else 1)
    suffix = "pp" if percentage_points else ""
    return f"{_report_delta_word(value)}{number:.{decimals}f}{suffix}"


def _report_delta_word(value: float) -> str:
    if value > 0:
        return "上升"
    if value < 0:
        return "下降"
    return "持平"


def _challenger_report_markdown(
    *,
    strategy_id: str,
    champion_id: str,
    compare: dict,
    challenger_backtest: dict,
    champion_backtest: dict,
    adopted: bool,
) -> str:
    deltas = _as_dict(compare.get("deltas"))
    lines = [
        "# 挑战者对比报告",
        "",
        f"- 挑战者策略：`{strategy_id}`",
        f"- 基线（champion）策略：`{champion_id}`",
        "- 采纳状态："
        + (
            "已在本地采纳挑战者；本地采纳不代表生产环境已上线"
            if adopted
            else "未采纳（仍以基线为准）"
        ),
        "",
        "## 关键指标并排",
        "",
        "| 指标 | 挑战者 | 基线 | 挑战者−基线 |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in (
        ("审批率", "approval_rate"),
        ("通过客群坏率", "approved_bad_rate"),
        ("预期利润", "expected_profit"),
    ):
        lines.append(
            f"| {label} | {_report_num(challenger_backtest.get(key))} | "
            f"{_report_num(champion_backtest.get(key))} | {_report_num(deltas.get(key))} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            str(compare.get("summary_text") or ""),
            "",
        ]
    )
    red_flags = [
        flag for flag in (compare.get("red_flags") or []) if isinstance(flag, dict)
    ]
    if red_flags:
        lines.append("## 红旗")
        lines.append("")
        for flag in red_flags:
            lines.append(
                f"- [{flag.get('level', '')}] {flag.get('code', '')}: {flag.get('message', '')}"
            )
        lines.append("")
    return "\n".join(lines)


def _report_num(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _as_dict(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def tool_render_strategy_doc(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id, task_id=str(ctx.task_id))
    meta = runtime.strategies.get_strategy_meta(strategy_id)
    backtests = [
        _jsonable(result) for result in runtime.strategies.list_backtests(strategy_id)
    ]
    artifacts = [
        artifact
        for artifact in runtime.strategies.list_strategy_artifacts(strategy_id)
        if artifact.get("kind") != "strategy_doc_md"
    ]
    band_stats = _band_stats_from_inputs(inputs.get("band_stats"))
    markdown, sections = render_strategy_doc_markdown(
        strategy=_jsonable(strategy),
        meta=meta or {},
        backtests=backtests,
        artifacts=artifacts,
        band_stats=band_stats,
    )
    version = int((meta or {}).get("version", 1))
    artifact = _persist_verified_strategy_markdown(
        runtime,
        ctx,
        strategy_id=strategy_id,
        kind="strategy_doc_md",
        filename_prefix=f"strategy_doc_{strategy_id}_v{version}",
        markdown=markdown,
        evidence={
            "strategy_effect_hash": runtime.strategies.get_strategy_spec_hash(
                strategy_id
            ),
            "strategy_version": version,
            "status": (meta or {}).get("status"),
            "asset_status": (meta or {}).get("asset_status"),
            "source_artifact_ids": [item["id"] for item in artifacts],
        },
    )
    return {"doc_path": artifact["path"], "sections": list(sections)}


# ---------------------------------------------------------------------------
# S4 rule strategy: mining, evaluation, and the rule-set selection gate helper.
# ---------------------------------------------------------------------------
# A single-rule lift this high (or a hit bad rate this high) usually means a
# leakage/near-target feature slipped into the candidate set, not a genuine
# reject rule -- surfaced so a reviewer can drop it before adoption.
_SUSPECT_LEAKAGE_LIFT = 10.0
_SUSPECT_LEAKAGE_BAD_RATE = 0.9
# Two rules co-hitting more than this share (Jaccard) are largely redundant.
_HIGH_OVERLAP_THRESHOLD = 0.8
# An included rule whose population share is below this fixed floor is flagged
# low_support (mirrors bands.py's _SPARSE_BAND_THRESHOLD). Distinct from the
# caller's min_support MINING filter: a caller may mine at a looser min_support
# (e.g. 0.01) yet still want a warning on any sub-2% rule before adoption.
_LOW_SUPPORT_FLOOR = 0.02


def tool_mine_rules(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    if "sample_design_ref" not in inputs:
        raise StrategyError("sample_design_ref is required")
    feature_cols = _optional_str_list(inputs.get("feature_cols"))
    columns = _unique([*(feature_cols or []), target_col]) if feature_cols else None
    frame, source_evidence, source_path, sample_binding = (
        _strategy_development_frame_with_evidence(
            runtime,
            dataset_id,
            task_id=task_id,
            target_col=target_col,
            sample_design_ref=inputs["sample_design_ref"],
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
            columns=columns,
        )
    )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    if feature_cols and sample_binding.split_column in feature_cols:
        raise StrategyError(
            "feature_cols cannot include the sample-design split column"
        )
    resolved_features = feature_cols or [
        feature
        for feature in _default_feature_cols(frame, target_col)
        if feature != sample_binding.split_column
    ]
    min_support = _float_or(inputs.get("min_support"), 0.02)
    min_lift = _float_or(inputs.get("min_lift"), 1.5)
    candidates = mine_rules(
        frame,
        feature_cols=resolved_features,
        target_col=target_col,
        max_depth=int(inputs.get("max_depth", 3)),
        min_support=min_support,
        min_lift=min_lift,
        top_k=int(inputs.get("top_k", 20)),
        seed=int(inputs.get("seed", DEFAULT_MINE_SEED)),
    )
    candidate_rules = [rule.as_dict() for rule in candidates]
    red_flags = _mine_red_flags(candidate_rules, nan_labels_dropped)
    _assert_source_unchanged(
        source_path, str(source_evidence["dataset_content_hash"])
    )
    revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
    return {
        "candidate_rules": candidate_rules,
        "n_rows": int(len(frame)),
        "feature_cols": list(resolved_features),
        "red_flags": red_flags,
        "nan_labels_dropped": nan_labels_dropped,
        "sample_design_ref": sample_binding.to_ref_dict(),
    }


def tool_evaluate_rule_set(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    if "sample_design_ref" not in inputs:
        raise StrategyError("sample_design_ref is required")
    rules_ordered = [
        dict(rule) for rule in (inputs.get("rules") or []) if isinstance(rule, dict)
    ]
    frame, source_evidence, source_path, sample_binding = (
        _strategy_development_frame_with_evidence(
            runtime,
            dataset_id,
            task_id=task_id,
            target_col=target_col,
            sample_design_ref=inputs["sample_design_ref"],
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        )
    )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    result = evaluate_rule_set(
        frame,
        rules_ordered,
        target_col=target_col,
        decision=str(inputs.get("decision") or "reject"),
    )
    red_flags = _evaluate_red_flags(result, rules_ordered, nan_labels_dropped)
    result["red_flags"] = red_flags
    result["nan_labels_dropped"] = nan_labels_dropped
    _assert_source_unchanged(
        source_path, str(source_evidence["dataset_content_hash"])
    )
    revalidate_strategy_sample_design_execution_binding(runtime, sample_binding)
    result["sample_design_ref"] = sample_binding.to_ref_dict()
    return result


def tool_select_rule_set(inputs: dict, ctx) -> dict:
    """Lightweight rule-set selection gate helper (S4).

    Assembles the user-selected ordered subset of the mined candidate rules into
    a gate payload and passes it through unchanged. ``selection`` is a literal
    ``None`` default in the template step's inputs so the generic apply_adjust
    gate-override channel (agent/gate_execution_adapter.py) can overwrite it with
    the parsed 「选 1,3,5」/「全选」/「去掉 2」 instruction -- exactly the band_edges
    precedent. A ``None`` selection means "keep all candidates" (no filter yet).
    """
    candidate_rules = [
        dict(rule)
        for rule in (inputs.get("candidate_rules") or [])
        if isinstance(rule, dict)
    ]
    selection = inputs.get("selection")
    decision = str(inputs.get("decision") or "reject")
    selected = [
        _build_ready_rule(rule, decision)
        for rule in _apply_rule_selection(candidate_rules, selection)
    ]
    return {
        "selected_rules": selected,
        "selected_count": len(selected),
        "candidate_count": len(candidate_rules),
    }


def _build_ready_rule(rule: dict, decision: str) -> dict:
    """Shape a mined candidate into a build_strategy-ready rule dict.

    build_strategy needs {condition, decision(, value)}; a mined CandidateRule
    carries only condition + display stats (lift/support/source/hit_bad_rate).
    Attach the reject decision and keep the display fields (build_strategy reads
    only condition/decision/value and ignores the rest, so they ride along for
    the renderer/waterfall without affecting the strategy)."""
    ready = dict(rule)
    ready["condition"] = str(rule.get("condition", ""))
    ready["decision"] = decision
    return ready


def _apply_rule_selection(candidate_rules: list[dict], selection) -> list[dict]:
    """Resolve a parsed selection into an ordered subset of candidate_rules.

    ``selection`` is None (keep all) or a list of 1-based indices in the display
    order the user chose (e.g. [1, 3, 5]); the returned order follows the
    selection order, not the candidate order, so the user can also reorder.
    Out-of-range/duplicate indices are dropped defensively -- the gate reply
    parser already validated them, this is belt-and-braces."""
    if selection is None:
        return [dict(rule) for rule in candidate_rules]
    ordered: list[dict] = []
    seen: set[int] = set()
    for raw in selection:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if index < 1 or index > len(candidate_rules) or index in seen:
            continue
        seen.add(index)
        ordered.append(dict(candidate_rules[index - 1]))
    return ordered


def _mine_red_flags(candidate_rules: list[dict], nan_labels_dropped: int) -> list[dict]:
    red_flags: list[dict] = []
    for rule in candidate_rules:
        lift = _finite(rule.get("lift"))
        hit_bad_rate = _finite(rule.get("hit_bad_rate"))
        if (lift is not None and lift > _SUSPECT_LEAKAGE_LIFT) or (
            hit_bad_rate is not None and hit_bad_rate > _SUSPECT_LEAKAGE_BAD_RATE
        ):
            red_flags.append(
                {
                    "code": "suspect_leakage",
                    "level": "red",
                    "message": (
                        f"规则 {rule.get('rule_id')}（{rule.get('condition')}）lift="
                        f"{_fmt_num(lift)}、命中坏率={_fmt_pct(hit_bad_rate)}，疑似泄漏/近目标特征入选，请核查。"
                    ),
                }
            )
        support = _finite(rule.get("support"))
        if support is not None and support < _LOW_SUPPORT_FLOOR:
            red_flags.append(
                {
                    "code": "low_support",
                    "level": "amber",
                    "message": (
                        f"规则 {rule.get('rule_id')}（{rule.get('condition')}）支持度 "
                        f"{_fmt_pct(support)} 低于 {_fmt_pct(_LOW_SUPPORT_FLOOR)} 底线，样本量偏小。"
                    ),
                }
            )
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    return red_flags


def _evaluate_red_flags(
    result: dict, rules_ordered: list[dict], nan_labels_dropped: int
) -> list[dict]:
    red_flags: list[dict] = []
    waterfall = result.get("waterfall") or []
    for row in waterfall:
        if int(row.get("incremental_hits") or 0) == 0:
            red_flags.append(
                {
                    "code": "rule_shadowed",
                    "level": "amber",
                    "message": (
                        f"规则 {row.get('rule_id')} 在瀑布中零增量命中（被前序规则完全覆盖），可考虑移除。"
                    ),
                }
            )
    overlap = result.get("overlap_matrix") or []
    for i in range(len(overlap)):
        for j in range(i + 1, len(overlap)):
            share = _finite(overlap[i][j])
            if share is not None and share > _HIGH_OVERLAP_THRESHOLD:
                red_flags.append(
                    {
                        "code": "high_overlap",
                        "level": "amber",
                        "message": (
                            f"规则 {waterfall[i].get('rule_id')} 与 "
                            f"{waterfall[j].get('rule_id')} 重叠 {_fmt_pct(share)} "
                            f"(>{_fmt_pct(_HIGH_OVERLAP_THRESHOLD)})，高度冗余。"
                        ),
                    }
                )
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    return red_flags


def _default_feature_cols(frame: pd.DataFrame, target_col: str) -> list[str]:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    return [column for column in numeric if column != target_col]


def _optional_str_list(value) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        cleaned = [str(item) for item in value if str(item).strip()]
        return cleaned or None
    return None


def _float_or(value, default: float) -> float:
    number = _optional_float(value)
    return default if number is None else number


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt_num(value) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.2f}"


def _fmt_pct(value) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:.1f}%"


def _persist_verified_strategy_markdown(
    runtime,
    ctx,
    *,
    strategy_id: str,
    kind: str,
    filename_prefix: str,
    markdown: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist one immutable, content-addressed Markdown strategy artifact."""

    try:
        producer_version = _STRATEGY_REPORT_PRODUCER_VERSIONS[kind]
    except KeyError as exc:
        raise StrategyError(f"unsupported verified report kind: {kind}") from exc
    content = markdown.encode("utf-8")
    content_hash = hashlib.sha256(content).hexdigest()
    content_size = len(content)
    strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    if strategy_dir.is_symlink():
        raise StrategyError("strategy artifact directory must not be a symlink")
    token = _content_addressed_artifact_token(filename_prefix)
    final_path = strategy_dir / f"{token}_{content_hash[:20]}.md"
    provenance = {
        "schema_version": _STRATEGY_ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "producer_version": producer_version,
        "strategy_id": strategy_id,
        "task_id": str(ctx.task_id),
        "kind": kind,
        "evidence": _jsonable(dict(evidence)),
    }
    audit = {
        "kind": "strategy.artifact",
        "target_ref": strategy_id,
        "outcome": "succeeded",
        "detail": {
            "task_id": str(ctx.task_id),
            "kind": kind,
            "path": str(final_path),
            "content_hash": content_hash,
            "content_size": content_size,
            "producer_version": producer_version,
        },
    }

    def register(conn):
        return runtime.strategies.register_verified_strategy_artifact_with_audit_on_connection(
            conn,
            strategy_id,
            kind=kind,
            path=str(final_path),
            content_hash=content_hash,
            content_size=content_size,
            provenance=provenance,
            audit=audit,
        )

    if final_path.exists() or final_path.is_symlink():
        if (
            final_path.is_symlink()
            or not final_path.is_file()
            or final_path.stat().st_size != content_size
            or not hmac.compare_digest(sha256_file(final_path), content_hash)
        ):
            raise StrategyError(
                "content-addressed strategy artifact path contains different bytes"
            )
        with runtime.strategies.transaction() as conn:
            record = register(conn)
    else:
        uow = ArtifactUnitOfWork()
        staged = uow.stage_file(strategy_dir, final_path.name)
        try:
            staged.path.write_bytes(content)
            record = uow.finalize_with_connection(
                runtime.strategies.transaction,
                register,
            )
        except Exception:
            uow.rollback()
            raise
    return {
        "artifact_id": str(record["id"]),
        "kind": kind,
        "filename": final_path.name,
        "path": str(final_path),
        "content_hash": content_hash,
        "content_size": content_size,
    }


def _content_addressed_artifact_token(value: object) -> str:
    token = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value)
    ).strip("_")
    return token[:96] or "strategy_artifact"


def _band_stats_from_inputs(value) -> list[dict]:
    if value in (None, ""):
        return []
    if isinstance(value, dict):
        bands = value.get("bands")
        if isinstance(bands, list):
            return [dict(band) for band in bands if isinstance(band, dict)]
        return []
    if isinstance(value, list):
        return [dict(band) for band in value if isinstance(band, dict)]
    return []


def _univariate_dataset_binding(
    runtime: "_Runtime",
    *,
    task_id: str,
    dataset_id: str,
) -> _UnivariateDatasetBinding:
    dataset = _owned_dataset(runtime, dataset_id, task_id=task_id)
    try:
        path = runtime.registry.resolve_verified_path(dataset_id)
    except (DatasetContentDriftError, KeyError, OSError, ValueError) as exc:
        raise StrategyError(
            "univariate candidate source dataset failed immutable hash verification"
        ) from exc
    content_hash = str(dataset.content_hash or "")
    if sha256_file(path) != content_hash:
        raise StrategyError(
            "univariate candidate source dataset content hash is invalid"
        )
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task_id
        )
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategyError(
            "univariate candidate data workspace binding is invalid"
        ) from exc
    if snapshot.active_dataset_id is not None:
        if snapshot.active_dataset_id != dataset_id:
            raise StrategyError(
                "univariate candidate dataset is not the active task workspace dataset"
            )
        if snapshot.active_dataset_content_hash != content_hash:
            raise StrategyError(
                "univariate candidate workspace content hash does not match the dataset"
            )
    semantic_hash = data_semantic_mapping_hash(snapshot.semantic_mapping)
    with connect(runtime.settings.db_path) as conn:
        persisted = (
            conn.execute(
                "SELECT 1 FROM data_workspaces WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            is not None
        )
        registry_metadata_hash = _univariate_registry_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=dataset_id,
            expected_content_hash=content_hash,
        )
    workspace = _UnivariateWorkspaceBinding(
        persisted=persisted,
        revision=snapshot.revision,
        generation=snapshot.analysis_generation,
        active_dataset_id=snapshot.active_dataset_id,
        active_dataset_content_hash=snapshot.active_dataset_content_hash,
        semantic_mapping=snapshot.semantic_mapping,
        semantic_mapping_hash=semantic_hash,
    )
    return _UnivariateDatasetBinding(
        dataset=dataset,
        path=Path(path),
        content_hash=content_hash,
        registry_metadata_hash=registry_metadata_hash,
        workspace=workspace,
    )


def _require_expected_univariate_binding(
    inputs: Mapping[str, Any],
    binding: _UnivariateDatasetBinding,
) -> None:
    expected_hash = inputs.get("expected_content_hash")
    expected_revision = inputs.get("workspace_revision")
    expected_generation = inputs.get("analysis_generation")
    expected_semantic_hash = inputs.get("semantic_mapping_hash")
    if (
        not isinstance(expected_hash, str)
        or not hmac.compare_digest(expected_hash, binding.content_hash)
        or isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision != binding.workspace.revision
        or isinstance(expected_generation, bool)
        or not isinstance(expected_generation, int)
        or expected_generation != binding.workspace.generation
        or not isinstance(expected_semantic_hash, str)
        or not hmac.compare_digest(
            expected_semantic_hash,
            binding.workspace.semantic_mapping_hash,
        )
    ):
        raise StrategyError(
            "univariate candidate data binding changed after user confirmation"
        )


def _univariate_registry_metadata_hash_on_connection(
    conn,
    *,
    task_id: str,
    dataset_id: str,
    expected_content_hash: str,
) -> str:
    row = conn.execute(
        """
        SELECT task_id, role, row_count, columns_json, has_target, target_col,
               content_hash
          FROM datasets
         WHERE id = ?
        """,
        (dataset_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError(f"dataset not found: {dataset_id}")
    registered_hash = row["content_hash"]
    if not isinstance(registered_hash, str) or not hmac.compare_digest(
        registered_hash,
        expected_content_hash,
    ):
        raise StrategyError("univariate candidate registered dataset hash changed")
    columns_json = row["columns_json"]
    if not isinstance(columns_json, str):
        raise StrategyError("univariate candidate dataset schema is invalid")
    try:
        json.loads(columns_json)
    except json.JSONDecodeError as exc:
        raise StrategyError("univariate candidate dataset schema is invalid") from exc
    payload = {
        "role": str(row["role"]),
        "row_count": int(row["row_count"]),
        "columns_json": columns_json,
        "has_target": int(row["has_target"]),
        "target_col": row["target_col"],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _univariate_field_roles(
    dataset,
    semantic_mapping: DataSemanticMapping,
    *,
    target_col: str,
) -> dict[str, str]:
    roles = {
        str(column): str(role) for column, role in semantic_mapping.field_roles.items()
    }
    inferred_sensitive: dict[str, str] = {}
    for profile in getattr(dataset, "columns", ()):
        column = str(getattr(profile, "name", ""))
        role = str(getattr(profile, "semantic_role", "") or "")
        if (
            column
            and column != target_col
            and role in _UNIVARIATE_FORBIDDEN_EXPLICIT_ROLES
        ):
            inferred_sensitive[column] = role
        if role == "target" and column != target_col:
            continue
        if column and role:
            roles.setdefault(column, role)
    # A user mapping may refine ordinary business semantics, but it cannot
    # downgrade registry-inferred identifiers or personal data into a reportable
    # categorical feature.
    roles.update(inferred_sensitive)
    for column in [column for column, role in roles.items() if role == "target"]:
        if column != target_col:
            del roles[column]
    roles[target_col] = "target"
    return roles


def _resolve_univariate_amount_column(
    raw_value,
    *,
    role: str,
    columns: list[str],
) -> str | None:
    if raw_value not in (None, ""):
        if not isinstance(raw_value, str):
            raise StrategyError(f"{role}_col must be a column name")
        if raw_value not in columns:
            raise StrategyError(f"unknown {role} column: {raw_value}")
        return raw_value
    return None


def _resolve_univariate_features(
    raw_features,
    *,
    columns: list[str],
    target_col: str,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
    sample_split_col: str | None,
    field_roles: Mapping[str, str],
) -> list[str]:
    if raw_features is None:
        requested: list[str] = []
    elif isinstance(raw_features, (str, bytes, bytearray)) or not isinstance(
        raw_features,
        (list, tuple),
    ):
        raise StrategyError("features must be an ordered array")
    else:
        requested = list(raw_features)
    if any(not isinstance(feature, str) or not feature for feature in requested):
        raise StrategyError("features must contain non-empty column names")
    if len(requested) != len(set(requested)):
        raise StrategyError("features must not contain duplicates")
    missing = sorted(set(requested) - set(columns))
    if missing:
        raise StrategyError("unknown feature columns: " + ", ".join(missing))
    if target_col in requested:
        raise StrategyError("target cannot also be a univariate feature")
    if sample_split_col is not None and sample_split_col in requested:
        raise StrategyError(
            "sample-design split column cannot be a univariate feature"
        )
    forbidden = sorted(
        feature
        for feature in requested
        if field_roles.get(feature) in _UNIVARIATE_FORBIDDEN_EXPLICIT_ROLES
    )
    if forbidden:
        raise StrategyError(
            "identifier, personal-data, or ignored fields cannot be univariate "
            "strategy candidates: " + ", ".join(forbidden)
        )
    if requested:
        features = requested
    else:
        excluded_columns = {
            target_col,
            *(
                column
                for column in (loan_amount_col, overdue_amount_col)
                if column is not None
            ),
        }
        if sample_split_col is not None:
            excluded_columns.add(sample_split_col)
        features = [
            str(column)
            for column in columns
            if str(column) not in excluded_columns
            and field_roles.get(str(column)) not in _UNIVARIATE_AUTO_EXCLUDED_ROLES
        ]
    if not features:
        raise StrategyError(
            "no eligible univariate features remain; provide an explicit feature list"
        )
    if len(features) > _UNIVARIATE_MAX_FEATURES:
        raise StrategyError(
            "univariate candidate analysis accepts at most "
            f"{_UNIVARIATE_MAX_FEATURES} features per bounded run; "
            "select a smaller explicit feature set"
        )
    return features


def _univariate_feature_types(
    features: list[str],
    *,
    frame: pd.DataFrame,
    field_roles: Mapping[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for feature in features:
        role = field_roles.get(feature)
        if role in _UNIVARIATE_NUMERIC_ROLES:
            result[feature] = "numeric"
        elif role in _UNIVARIATE_CATEGORICAL_ROLES:
            result[feature] = "categorical"
        else:
            result[feature] = (
                "numeric"
                if pd.api.types.is_numeric_dtype(frame[feature].dtype)
                and not pd.api.types.is_bool_dtype(frame[feature].dtype)
                else "categorical"
            )
    return result


def _univariate_methods(raw_methods) -> tuple[str, ...]:
    if raw_methods is None:
        return ()
    if isinstance(raw_methods, (str, bytes, bytearray)) or not isinstance(
        raw_methods,
        (list, tuple),
    ):
        raise StrategyError("methods must be an ordered array")
    methods = tuple(raw_methods)
    supported = {"equal_frequency", "equal_width", "chimerge", "tree", "manual"}
    if len(methods) != len(set(methods)) or any(
        not isinstance(method, str) or method not in supported for method in methods
    ):
        raise StrategyError(
            "methods must be unique and selected from equal_frequency, "
            "equal_width, chimerge, tree, manual"
        )
    order = {
        "equal_frequency": 0,
        "equal_width": 1,
        "chimerge": 2,
        "tree": 3,
        "manual": 4,
    }
    return tuple(sorted(methods, key=order.__getitem__))


def _univariate_manual_breakpoints(
    raw_breakpoints,
    *,
    features: list[str],
    feature_types: Mapping[str, str],
    methods: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    manual_requested = "manual" in methods
    if raw_breakpoints is not None and not isinstance(raw_breakpoints, Mapping):
        raise StrategyError(
            "manual_breakpoints must be a feature-to-array mapping"
        )
    provided = {} if raw_breakpoints is None else dict(raw_breakpoints)
    if any(not isinstance(feature, str) or not feature for feature in provided):
        raise StrategyError(
            "manual_breakpoints keys must be requested numeric feature names"
        )
    if not manual_requested:
        if provided:
            raise StrategyError(
                "manual_breakpoints are only allowed when manual is requested"
            )
        return {}

    numeric_features = tuple(
        feature for feature in features if feature_types[feature] == "numeric"
    )
    if not numeric_features:
        if provided:
            raise StrategyError(
                "manual_breakpoints contains an unknown or non-numeric feature"
            )
        return {}
    if not provided:
        raise StrategyError(
            "manual_breakpoints must provide at least one requested numeric feature"
        )
    extras = sorted(set(provided) - set(numeric_features))
    if extras:
        raise StrategyError(
            "manual_breakpoints contains an unknown or non-numeric feature: "
            + ", ".join(extras)
        )
    return {
        feature: _univariate_manual_breakpoint_values(
            provided[feature],
            feature=feature,
        )
        for feature in numeric_features
        if feature in provided
    }


def _univariate_manual_breakpoint_values(
    raw_values,
    *,
    feature: str,
) -> tuple[float, ...]:
    if isinstance(raw_values, (str, bytes, bytearray)) or not isinstance(
        raw_values,
        Sequence,
    ):
        raise StrategyError(
            f"manual_breakpoints for {feature} must be an ordered array"
        )
    if not raw_values:
        raise StrategyError(
            f"manual_breakpoints for {feature} requires at least one breakpoint"
        )
    if len(raw_values) + 1 > _UNIVARIATE_MAX_BINS:
        raise StrategyError(
            f"manual_breakpoints for {feature} exceed the configured bin budget"
        )
    normalized: list[float] = []
    for item in raw_values:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise StrategyError(
                f"manual_breakpoints for {feature} must contain finite numbers"
            )
        if isinstance(item, Integral) and abs(int(item)) > 2**53 - 1:
            raise StrategyError(
                f"manual_breakpoints for {feature} exceed exact numeric precision"
            )
        number = float(item)
        if not math.isfinite(number):
            raise StrategyError(
                f"manual_breakpoints for {feature} must contain finite numbers"
            )
        normalized.append(number)
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise StrategyError(
            f"manual_breakpoints for {feature} must be strictly increasing and unique"
        )
    return tuple(normalized)


def _univariate_candidate_version_contract(
    analysis_schema_version: object,
) -> dict[str, str]:
    if analysis_schema_version == UNIVARIATE_ANALYSIS_SCHEMA_VERSION:
        return {
            "tool_schema_version": _UNIVARIATE_CANDIDATE_TOOL_SCHEMA_VERSION,
            "producer_version": _UNIVARIATE_CANDIDATE_PRODUCER_VERSION,
            "artifact_schema_version": (
                _UNIVARIATE_CANDIDATE_ARTIFACT_SCHEMA_VERSION
            ),
        }
    if analysis_schema_version == UNIVARIATE_MANUAL_ANALYSIS_SCHEMA_VERSION:
        return {
            "tool_schema_version": _UNIVARIATE_CANDIDATE_V2_TOOL_SCHEMA_VERSION,
            "producer_version": _UNIVARIATE_CANDIDATE_V2_PRODUCER_VERSION,
            "artifact_schema_version": (
                _UNIVARIATE_CANDIDATE_V2_ARTIFACT_SCHEMA_VERSION
            ),
        }
    raise StrategyError("univariate analysis schema_version is unsupported")


def _preflight_univariate_work_budget(
    inputs: Mapping[str, Any],
    *,
    binding: _UnivariateDatasetBinding,
    target_col: str,
    methods: tuple[str, ...],
    sentinel_count: int,
    row_count: int | None = None,
    sample_split_col: str | None = None,
) -> None:
    row_count = int(binding.dataset.row_count if row_count is None else row_count)
    if not 1 <= row_count <= _UNIVARIATE_MAX_ROWS:
        raise StrategyError(
            "univariate candidate source row count exceeds the configured budget"
        )
    raw_features = inputs.get("features")
    if isinstance(raw_features, (list, tuple)) and raw_features:
        feature_count = len(raw_features)
    else:
        roles = _univariate_field_roles(
            binding.dataset,
            binding.workspace.semantic_mapping,
            target_col=target_col,
        )
        explicitly_excluded = {
            target_col,
            *(
                value
                for value in (
                    inputs.get("loan_amount_col"),
                    inputs.get("overdue_amount_col"),
                    sample_split_col,
                )
                if isinstance(value, str) and value
            ),
        }
        feature_count = sum(
            str(profile.name) not in explicitly_excluded
            and roles.get(str(profile.name)) not in _UNIVARIATE_AUTO_EXCLUDED_ROLES
            for profile in binding.dataset.columns
        )
    if feature_count > _UNIVARIATE_MAX_FEATURES:
        raise StrategyError(
            "univariate candidate analysis accepts at most "
            f"{_UNIVARIATE_MAX_FEATURES} features per bounded run; "
            "select a smaller explicit feature set"
        )
    method_count = len(methods) if methods else 4
    bin_count = int(inputs.get("bin_count", 10))
    estimated = (
        row_count
        * feature_count
        * (method_count * (bin_count + sentinel_count + 1) + sentinel_count)
    )
    if estimated > _UNIVARIATE_MAX_EVALUATED_CELLS:
        raise StrategyError(
            "univariate candidate analysis exceeds the combined row/bin work "
            f"budget ({estimated} > {_UNIVARIATE_MAX_EVALUATED_CELLS}); "
            "select fewer features, bins, or sentinel values"
        )


def _univariate_estimated_evaluated_cells(
    frame: pd.DataFrame,
    *,
    features: list[str],
    feature_types: Mapping[str, str],
    methods: tuple[str, ...],
    bin_count: int,
    manual_breakpoints: Mapping[str, Sequence[float]],
    sentinel_mapping: Mapping[str, list[str | int | float]],
    sentinel_value_count: int,
) -> int:
    groups = 0
    for feature in features:
        sentinel_count = len(sentinel_mapping.get(feature, ()))
        if feature_types[feature] == "numeric":
            method_names = (
                methods
                if methods
                else ("equal_frequency", "equal_width", "chimerge", "tree")
            )
            for method in method_names:
                if method == "manual" and feature not in manual_breakpoints:
                    continue
                regular_bin_count = (
                    len(manual_breakpoints[feature]) + 1
                    if method == "manual"
                    else bin_count
                )
                groups += regular_bin_count + sentinel_count + 1
            continue
        category_count = int(frame[feature].nunique(dropna=True))
        missing_count = int(bool(frame[feature].isna().any()))
        groups += max(1, category_count + sentinel_count + missing_count)
    sentinel_discovery = len(features) * sentinel_value_count
    return int(len(frame) * (groups + sentinel_discovery))


def _normalize_univariate_sentinel_values(
    raw_values,
) -> list[str | int | float]:
    if raw_values is None:
        values: list[str | int | float] = []
    elif isinstance(raw_values, (str, bytes, bytearray)) or not isinstance(
        raw_values,
        (list, tuple),
    ):
        raise StrategyError("sentinel_values must be an ordered array")
    else:
        values = list(raw_values)
    if len(values) > 20:
        raise StrategyError("sentinel_values accepts at most 20 values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (str, int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        for value in values
    ):
        raise StrategyError("sentinel_values must contain finite numbers or text")
    identities = [
        json.dumps(
            [type(value).__name__, value],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        for value in values
    ]
    if len(identities) != len(set(identities)):
        raise StrategyError("sentinel_values must not contain duplicates")
    return values


def _univariate_sentinel_mapping(
    raw_values,
    *,
    features: list[str],
    frame: pd.DataFrame,
    feature_types: Mapping[str, str],
) -> tuple[dict[str, list[str | int | float]], list[str]]:
    values = _normalize_univariate_sentinel_values(raw_values)
    mapping: dict[str, list[str | int | float]] = {}
    red_flags: list[str] = []
    for feature in features:
        compatible: list[str | int | float] = []
        compatible_keys: set[tuple[str, str]] = set()
        ignored: list[str] = []
        for value in values:
            if feature_types[feature] == "numeric" and isinstance(value, str):
                ignored.append(repr(value))
                continue
            if _univariate_sentinel_observed(
                frame[feature],
                value,
                feature_type=feature_types[feature],
            ):
                identity = _univariate_sentinel_identity(value)
                if identity in compatible_keys:
                    red_flags.append(
                        f"sentinel_equivalent_duplicate:{feature}:{value!r}"
                    )
                    continue
                compatible_keys.add(identity)
                compatible.append(value)
        if compatible:
            mapping[feature] = compatible
        if ignored:
            red_flags.append(f"sentinel_incompatible:{feature}:" + ",".join(ignored))
    return mapping, red_flags


def _univariate_sentinel_identity(
    value: str | int | float,
) -> tuple[str, str]:
    if isinstance(value, (int, float)):
        return ("number", repr(float(value)))
    return ("string", value)


def _univariate_sentinel_observed(
    series: pd.Series,
    value: str | int | float,
    *,
    feature_type: str,
) -> bool:
    if feature_type == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        return bool((numeric == float(value)).fillna(False).any())
    try:
        return bool((series == value).fillna(False).any())
    except (TypeError, ValueError):
        return False


def _univariate_candidate_metrics(
    analysis: Mapping[str, Any],
) -> list[MetricObservation]:
    observations: list[MetricObservation] = []
    for feature in analysis["features"]:
        feature_name = str(feature["feature"])
        for method in feature["methods"]:
            method_name = str(method["method"])
            prefix = f"{feature_name}.{method_name}"
            metrics = method.get("metrics")
            available = method.get("status") == "available" and isinstance(
                metrics,
                Mapping,
            )
            for metric_key in ("iv", "ks", "auc"):
                observations.extend(
                    _metric_observations(
                        f"{prefix}.{metric_key}",
                        count=(
                            ("observed", metrics[metric_key])
                            if available
                            else ("unavailable", None)
                        ),
                        loan_amount=("unavailable", None),
                        overdue_amount=("unavailable", None),
                    )
                )
            if not available:
                continue
            total_amounts = metrics["amount_metrics"]
            for bin_row in method["bins"]:
                bin_prefix = f"{prefix}.bin.{bin_row['id']}"
                observations.extend(
                    _metric_observations(
                        f"{bin_prefix}.hit_rate",
                        count=("observed", bin_row["share"]),
                        loan_amount=_amount_share_observation(
                            bin_row["amount_metrics"]["loan_amount"],
                            total_amounts["loan_amount"],
                        ),
                        overdue_amount=_amount_share_observation(
                            bin_row["amount_metrics"]["overdue_amount"],
                            total_amounts["overdue_amount"],
                        ),
                    )
                )
                observations.extend(
                    _metric_observations(
                        f"{bin_prefix}.overdue_rate",
                        count=("not_applicable", None),
                        loan_amount=_kernel_rate_observation(
                            bin_row["amount_metrics"]["overdue_rate"]
                        ),
                        overdue_amount=("not_applicable", None),
                    )
                )
    return observations


def _metric_observations(
    metric_name: str,
    *,
    count: tuple[str, int | float | None],
    loan_amount: tuple[str, int | float | None],
    overdue_amount: tuple[str, int | float | None],
) -> list[MetricObservation]:
    return [
        MetricObservation(metric_name, "count", *count),
        MetricObservation(metric_name, "loan_amount", *loan_amount),
        MetricObservation(metric_name, "overdue_amount", *overdue_amount),
    ]


def _amount_share_observation(
    selected: Mapping[str, Any],
    total: Mapping[str, Any],
) -> tuple[str, float | None]:
    if selected.get("status") != "available" or total.get("status") != "available":
        return ("unavailable", None)
    denominator = float(total["sum"])
    if denominator == 0:
        return ("not_applicable", None)
    return ("observed", float(selected["sum"]) / denominator)


def _kernel_rate_observation(
    rate: Mapping[str, Any],
) -> tuple[str, float | None]:
    status = rate.get("status")
    if status == "available":
        return ("observed", float(rate["value"]))
    if status == "not_applicable":
        return ("not_applicable", None)
    return ("unavailable", None)


def _univariate_red_flags(
    analysis: Mapping[str, Any],
    *,
    initial: list[str],
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> list[str]:
    red_flags = list(initial)
    if loan_amount_col is None:
        red_flags.append("loan_amount_metrics_unavailable:column_not_configured")
    if overdue_amount_col is None:
        red_flags.append("overdue_amount_metrics_unavailable:column_not_configured")
    available_count = 0
    for feature in analysis["features"]:
        for method in feature["methods"]:
            prefix = f"{feature['feature']}.{method['method']}"
            if method["status"] != "available":
                evidence = method.get("evidence") or {}
                kind = (
                    evidence.get("kind", "unknown")
                    if isinstance(evidence, dict)
                    else "unknown"
                )
                red_flags.append(f"{prefix}:unavailable:{kind}")
                continue
            available_count += 1
            method_metrics = method.get("metrics") or {}
            amount_metrics = method_metrics.get("amount_metrics") or {}
            for dimension in ("loan_amount", "overdue_amount"):
                measure = amount_metrics.get(dimension) or {}
                if (
                    measure.get("status") == "available"
                    and float(measure.get("coverage_rate", 0.0)) < 1.0
                ):
                    red_flags.append(f"{prefix}:{dimension}_partial_coverage")
            for evidence in method.get("evidence") or []:
                if isinstance(evidence, Mapping):
                    kind = str(evidence.get("kind") or "diagnostic")
                    severity = str(evidence.get("severity") or "warning")
                    red_flags.append(f"{prefix}:{severity}:{kind}")
    if available_count == 0:
        red_flags.append("no_available_univariate_methods")
    return sorted(set(red_flags))


def _write_univariate_candidate_artifacts(
    runtime: "_Runtime",
    *,
    task_id: str,
    binding: _UnivariateDatasetBinding,
    sample_design_binding: StrategySampleDesignExecutionBinding,
    candidate_evidence: Mapping[str, Any],
    generation_parameters: Mapping[str, Any],
    bundle: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    if set(bundle) != {"json", "xlsx"} or any(
        not isinstance(content, bytes) or not content for content in bundle.values()
    ):
        raise StrategyError(
            "strategy candidate report bundle must contain non-empty JSON and XLSX bytes"
        )
    analysis = candidate_evidence.get("analysis")
    if not isinstance(analysis, Mapping):
        raise StrategyError("univariate candidate analysis must be an object")
    version_contract = _univariate_candidate_version_contract(
        analysis.get("schema_version")
    )
    if candidate_evidence.get("producer_version") != version_contract[
        "producer_version"
    ]:
        raise StrategyError(
            "univariate candidate analysis and producer versions do not match"
        )
    evidence_generation = candidate_evidence.get("generation")
    if not isinstance(evidence_generation, Mapping) or not isinstance(
        evidence_generation.get("parameters"),
        Mapping,
    ):
        raise StrategyError("univariate candidate generation must be an object")
    if dict(evidence_generation["parameters"]) != dict(generation_parameters):
        raise StrategyError(
            "univariate candidate generation parameters changed before persistence"
        )
    if generation_parameters.get("analysis_schema_version") != analysis.get(
        "schema_version"
    ):
        raise StrategyError(
            "univariate candidate analysis schema binding is inconsistent"
        )
    revalidate_strategy_sample_design_execution_binding(
        runtime,
        sample_design_binding,
    )
    candidate_id = str(candidate_evidence["candidate_id"])
    evidence_hash = str(candidate_evidence["evidence_hash"])
    out_dir = Path(runtime.settings.tasks_dir) / task_id / "strategy_candidates"
    kinds = {
        "json": "strategy_candidate_json",
        "xlsx": "strategy_candidate_xlsx",
    }
    uow = ArtifactUnitOfWork()
    staged_specs = []
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        for report_format in ("json", "xlsx"):
            content = bundle[report_format]
            content_hash = hashlib.sha256(content).hexdigest()
            staged = uow.stage_file(
                out_dir,
                f"{candidate_id}_{content_hash[:12]}.{report_format}",
            )
            staged.path.write_bytes(content)
            staged_specs.append(
                (report_format, kinds[report_format], staged, content_hash)
            )

        # Acquire the SQLite writer lock before promoting deterministic final
        # paths.  Otherwise two identical invocations can promote over one
        # another and a late rollback can delete the peer's committed file.
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_univariate_candidate_binding_on_connection(
                    conn,
                    task_id=task_id,
                    binding=binding,
                )
                require_strategy_sample_design_execution_binding_on_connection(
                    conn,
                    sample_design_binding,
                )
                _assert_source_unchanged(binding.path, binding.content_hash)
                uow.promote_all()
                records = []
                for report_format, kind, staged, content_hash in staged_specs:
                    if sha256_file(staged.final_path) != content_hash:
                        raise StrategyError(
                            "strategy candidate report changed before registration"
                        )
                    provenance = {
                        "schema_version": version_contract[
                            "artifact_schema_version"
                        ],
                        "producer_version": version_contract["producer_version"],
                        "candidate_id": candidate_id,
                        "evidence_hash": evidence_hash,
                        "dataset_id": str(binding.dataset.id),
                        "dataset_content_hash": binding.content_hash,
                        "registry_metadata_hash": binding.registry_metadata_hash,
                        "workspace_revision": binding.workspace.revision,
                        "workspace_generation": binding.workspace.generation,
                        "semantic_mapping_hash": binding.workspace.semantic_mapping_hash,
                        "generation_parameters": dict(generation_parameters),
                        "format": report_format,
                    }
                    records.append(
                        runtime.task_artifacts.register_on_connection(
                            conn,
                            task_id=task_id,
                            kind=kind,
                            path=str(staged.final_path),
                            content_hash=content_hash,
                            origin_tool="strategy.analyze_univariate_candidates",
                            provenance=provenance,
                        )
                    )
                # Commit explicitly while the connection is still in this try
                # block.  Any registration/commit failure can therefore restore
                # promoted files before the SQLite writer lock is released.
                conn.commit()
                db_committed = True
            except Exception:
                # Do not retry a partially failed promoted-file rollback after
                # this transaction releases its cross-process writer lock.
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
        uow.commit()
    except Exception:
        # Once the DB commit succeeds the registered final paths are durable.
        # Never let a later backup-cleanup error delete a peer's identical file.
        if not db_committed and not rollback_attempted_under_lock:
            uow.rollback()
        raise
    return [
        {
            "artifact_id": str(record["id"]),
            "kind": kind,
            "filename": staged.final_path.name,
            "content_hash": content_hash,
            "download_url": (
                f"/api/tasks/{quote(task_id, safe='')}"
                f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
            ),
        }
        for (_report_format, kind, staged, content_hash), record in zip(
            staged_specs,
            records,
            strict=True,
        )
    ]


def _require_univariate_candidate_binding_on_connection(
    conn,
    *,
    task_id: str,
    binding: _UnivariateDatasetBinding,
) -> None:
    live_metadata_hash = _univariate_registry_metadata_hash_on_connection(
        conn,
        task_id=task_id,
        dataset_id=str(binding.dataset.id),
        expected_content_hash=binding.content_hash,
    )
    if not hmac.compare_digest(
        live_metadata_hash,
        binding.registry_metadata_hash,
    ):
        raise StrategyError(
            "univariate candidate dataset metadata changed during analysis"
        )
    row = conn.execute(
        """
        SELECT revision, active_dataset_id, active_dataset_content_hash,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces
         WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    expected = binding.workspace
    if not expected.persisted:
        if row is not None:
            raise StrategyError(
                "univariate candidate data workspace changed during analysis"
            )
        return
    if row is None:
        raise StrategyError(
            "univariate candidate data workspace disappeared during analysis"
        )
    mapping_json = row["semantic_mapping_json"]
    if not isinstance(mapping_json, str):
        raise StrategyError("univariate candidate semantic mapping is invalid")
    try:
        mapping = data_semantic_mapping_from_dict(json.loads(mapping_json))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StrategyError("univariate candidate semantic mapping is invalid") from exc
    live_semantic_hash = data_semantic_mapping_hash(mapping)
    active_hash = row["active_dataset_content_hash"]
    if (
        int(row["revision"]) != expected.revision
        or int(row["analysis_generation"]) != expected.generation
        or row["active_dataset_id"] != expected.active_dataset_id
        or row["active_dataset_content_hash"] != expected.active_dataset_content_hash
        or not hmac.compare_digest(
            live_semantic_hash,
            expected.semantic_mapping_hash,
        )
        or (
            active_hash is not None
            and not hmac.compare_digest(str(active_hash), binding.content_hash)
        )
    ):
        raise StrategyError(
            "univariate candidate data workspace changed during analysis"
        )


class _Runtime(PackRuntime):
    def _extend(self, ctx) -> None:
        self.strategies = StrategyRepository(self.settings.db_path)
        self.task_artifacts = TaskArtifactRepository(self.settings.db_path)
        self.automatic_tree_apply_runs = AutomaticTreeApplyRepository(
            self.settings.db_path
        )
        self.data_workspaces = DataWorkspaceRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _strategy_development_frame_with_evidence(
    runtime: _Runtime,
    dataset_id: str,
    *,
    task_id: str,
    target_col: str | None,
    sample_design_ref: object,
    drop_nan_labels: bool,
    columns: list[str] | None = None,
    normalize_target: bool = True,
) -> tuple[
    pd.DataFrame,
    dict,
    Path,
    StrategySampleDesignExecutionBinding,
]:
    """Load one exact active development population under governed target semantics.

    The legacy strategy tools historically read a task-owned dataset directly.
    V2 development workflows instead share this fail-closed boundary so their
    candidate metrics cannot silently mix development, validation, or OOT rows
    or reinterpret the target polarity.
    """

    if sample_design_ref is None:
        raise StrategyError("sample_design_ref is required")
    dataset = _owned_dataset(runtime, dataset_id, task_id=task_id)
    try:
        source_path = Path(runtime.registry.resolve_verified_path(dataset.id))
    except (DatasetContentDriftError, KeyError, OSError, ValueError) as exc:
        raise StrategyError(
            "strategy development source dataset failed immutable hash verification"
        ) from exc
    source_hash = str(dataset.content_hash or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise StrategyError(
            "strategy development source dataset has no immutable content hash"
        )
    if sha256_file(source_path) != source_hash:
        raise StrategyError(
            "strategy development source dataset content hash is invalid"
        )
    try:
        workspace = DataWorkspaceRepository(
            runtime.settings.db_path
        ).get_or_default(task_id)
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategyError("strategy development DataWorkspace is invalid") from exc
    if (
        workspace.active_dataset_id != dataset.id
        or workspace.active_dataset_content_hash != source_hash
    ):
        raise StrategyError(
            "strategy development dataset must be the exact active DataWorkspace dataset"
        )
    resolved_target_col = target_col or workspace.semantic_mapping.target_col
    if not isinstance(resolved_target_col, str) or not resolved_target_col:
        raise StrategyError(
            "strategy development requires a confirmed DataWorkspace target"
        )
    if workspace.semantic_mapping.target_col != resolved_target_col:
        raise StrategyError(
            "strategy development target must match the confirmed DataWorkspace target"
        )

    sample_binding = load_strategy_sample_design_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=sample_design_ref,
        dataset_id=dataset.id,
        dataset_content_hash=source_hash,
        workspace_revision=workspace.revision,
        workspace_generation=workspace.analysis_generation,
        semantic_mapping_hash=data_semantic_mapping_hash(
            workspace.semantic_mapping
        ),
        target_col=resolved_target_col,
        drop_nan_labels=drop_nan_labels,
    )
    projected_columns = None
    if columns is not None:
        projected_columns = _unique(
            [*columns, resolved_target_col, sample_binding.split_column]
        )
    frame = runtime.backend.read_frame(source_path, columns=projected_columns)
    _assert_source_unchanged(source_path, source_hash)
    frame = bind_strategy_development_frame(
        frame,
        binding=sample_binding,
        normalize_target=normalize_target,
    )
    return (
        frame,
        {
            "schema_version": "strategy.analysis-source.v2",
            "dataset_id": dataset.id,
            "dataset_role": dataset.role,
            "dataset_content_hash": source_hash,
            "registered_content_hash": dataset.content_hash,
            "registered_row_count": int(dataset.row_count),
            "active_population_count": sample_binding.active_population_count,
            "analyzed_row_count": int(len(frame)),
            "columns": list(projected_columns or frame.columns),
            "sample_design_ref": sample_binding.to_ref_dict(),
            "sample_design_source_ref": sample_binding.source_ref,
            "sample_design_partition": "development",
            "target_col": sample_binding.target_col,
            "target_bad_value": sample_binding.target_bad_value,
        },
        source_path,
        sample_binding,
    )


def _dataset_frame(
    runtime: _Runtime,
    dataset_id: str,
    *,
    task_id: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    dataset = runtime.registry.get(dataset_id)
    if str(dataset.task_id) != str(task_id):
        raise StrategyError(f"dataset not found: {dataset_id}")
    return runtime.backend.read_frame(
        runtime.registry.resolve_path(dataset.id), columns=columns
    )


def _task_dataset_frame_with_evidence(
    runtime: _Runtime,
    dataset_id: str,
    *,
    task_id: str,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict, Path]:
    dataset = _owned_dataset(runtime, dataset_id, task_id=task_id)
    source_path = runtime.registry.resolve_path(dataset.id)
    source_hash = sha256_file(source_path)
    frame = runtime.backend.read_frame(source_path, columns=columns)
    _assert_source_unchanged(source_path, source_hash)
    return (
        frame,
        {
            "schema_version": "strategy.analysis-source.v1",
            "dataset_id": dataset.id,
            "dataset_role": dataset.role,
            "dataset_content_hash": source_hash,
            "registered_content_hash": dataset.content_hash,
            "registered_row_count": int(dataset.row_count),
            "analyzed_row_count": int(len(frame)),
            "columns": list(columns or frame.columns),
        },
        Path(source_path),
    )


def _assert_source_unchanged(source_path: Path, expected_hash: str) -> None:
    if sha256_file(source_path) != expected_hash:
        raise StrategyError("source dataset changed while analysis was running")


def _owned_dataset(runtime: _Runtime, dataset_id: str, *, task_id: str):
    try:
        dataset = runtime.registry.get(dataset_id)
    except KeyError:
        raise StrategyError(f"dataset not found: {dataset_id}") from None
    if str(dataset.task_id) != str(task_id):
        raise StrategyError(f"dataset not found: {dataset_id}")
    return dataset


def _strategy(runtime: _Runtime, strategy_id: str, *, task_id: str) -> Strategy:
    strategy = runtime.strategies.get_strategy(strategy_id)
    metadata = runtime.strategies.get_strategy_meta(strategy_id)
    if strategy is None or metadata is None or str(metadata["task_id"]) != str(task_id):
        raise StrategyError(f"strategy not found: {strategy_id}")
    return strategy


def _profit_params(payload: dict) -> ProfitParams:
    return ProfitParams(
        annual_rate=float(payload["annual_rate"]),
        funding_rate=float(payload["funding_rate"]),
        lgd=float(payload["lgd"]),
        operating_cost_per_loan=float(payload["operating_cost_per_loan"]),
        term_months=int(payload["term_months"]),
    )


def _profit_quality_warnings(
    frame: pd.DataFrame,
    *,
    segment_col: str | None,
    ead_col: str,
) -> list[dict]:
    warnings: list[dict] = []
    zero_ead_rows = int((pd.to_numeric(frame[ead_col], errors="raise") == 0).sum())
    if zero_ead_rows:
        warnings.append(
            {
                "code": "zero_ead_rows",
                "level": "amber",
                "count": zero_ead_rows,
                "message": (
                    f"{zero_ead_rows} 行 EAD 为 0；这些记录仍计入运营成本，"
                    "但不贡献收入、预期损失或资金成本。"
                ),
            }
        )
    if segment_col:
        null_segments = int(frame[segment_col].isna().sum())
        if null_segments:
            warnings.append(
                {
                    "code": "null_segment",
                    "level": "amber",
                    "count": null_segments,
                    "message": f"{null_segments} 行分群为空，已作为独立空值分群汇总。",
                }
            )
    return warnings


def _write_task_analysis_artifacts(
    runtime,
    *,
    task_id: str,
    analysis_kind: str,
    source_hash: str,
    assumptions: dict,
    files: tuple[tuple[str, str, str], ...],
) -> list[dict]:
    out_dir = Path(runtime.settings.tasks_dir) / task_id / "strategy_analysis"
    stem = _analysis_artifact_stem(analysis_kind, source_hash, assumptions)
    uow = ArtifactUnitOfWork()
    staged_specs = [
        (kind, uow.stage_file(out_dir, f"{stem}.{suffix}"), text)
        for kind, suffix, text in files
    ]
    origin_tool = {
        "profit": "strategy.profit_calc",
        "roll_rate": "strategy.roll_rate_matrix",
    }.get(analysis_kind, f"strategy.{analysis_kind}")
    provenance = _task_analysis_artifact_provenance(
        analysis_kind=analysis_kind,
        source_hash=source_hash,
        assumptions=assumptions,
    )
    try:
        for _kind, staged, text in staged_specs:
            staged.path.write_text(text, encoding="utf-8")

        def register(conn):
            return [
                runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=kind,
                    path=str(staged.final_path),
                    content_hash=sha256_file(staged.final_path),
                    origin_tool=origin_tool,
                    provenance=provenance,
                )
                for kind, staged, _text in staged_specs
            ]

        records = uow.finalize_with_connection(
            runtime.task_artifacts.transaction,
            register,
        )
    except Exception:
        uow.rollback()
        raise
    return [
        {
            "artifact_id": str(record["id"]),
            "kind": kind,
            "filename": staged.final_path.name,
            "content_hash": str(record["content_hash"]),
        }
        for (kind, staged, _text), record in zip(staged_specs, records, strict=True)
    ]


def _task_analysis_artifact_provenance(
    *,
    analysis_kind: str,
    source_hash: str,
    assumptions: dict,
) -> dict:
    producer_version = _task_analysis_producer_version(analysis_kind)
    assumptions_payload = _jsonable(assumptions)
    assumptions_hash = hashlib.sha256(
        json.dumps(
            assumptions_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": _TASK_ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "producer_version": producer_version,
        "analysis_kind": str(analysis_kind),
        "source_dataset_id": str(assumptions.get("dataset_id") or ""),
        "source_dataset_content_hash": str(source_hash),
        "assumptions_hash": assumptions_hash,
        "assumptions": assumptions_payload,
    }


def _analysis_artifact_stem(
    analysis_kind: str,
    source_hash: str,
    assumptions: dict,
) -> str:
    producer_version = _task_analysis_producer_version(analysis_kind)
    digest = hashlib.sha256(
        json.dumps(
            {
                "producer_version": producer_version,
                "source_hash": source_hash,
                "assumptions": assumptions,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        ).encode("utf-8")
    ).hexdigest()
    return f"{analysis_kind}_{digest[:16]}"


def _task_analysis_producer_version(analysis_kind: str) -> str:
    try:
        return _TASK_ANALYSIS_PRODUCER_VERSIONS[str(analysis_kind)]
    except KeyError as exc:
        raise StrategyError(
            f"unsupported task analysis artifact kind: {analysis_kind}"
        ) from exc


def _profit_markdown(
    *,
    result_rows: list[dict],
    assumptions: dict,
    source_evidence: dict,
    warnings: list[dict],
) -> str:
    columns = [
        "segment",
        "count",
        "revenue",
        "expected_loss",
        "funding_cost",
        "operating_cost",
        "net_profit",
        "roa",
    ]
    lines = [
        "# 分群利润分析",
        "",
        f"- 数据集：`{source_evidence['dataset_id']}`",
        f"- 数据哈希：`{source_evidence['dataset_content_hash']}`",
        f"- 分群数：{len(result_rows)}",
        "",
        "## 假设与公式",
        "",
        "```json",
        json.dumps(assumptions, ensure_ascii=False, indent=2, allow_nan=False),
        "```",
        "",
        "## 结果",
        "",
        _markdown_table(
            columns, [[row.get(column) for column in columns] for row in result_rows]
        ),
    ]
    if warnings:
        lines.extend(
            [
                "",
                "## 数据质量提示",
                "",
                *[f"- {warning['message']}" for warning in warnings],
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _roll_rate_csv(matrix) -> str:
    rows = []
    for index, state in enumerate(matrix.states):
        row = {"from_state": state, "base_count": matrix.base_counts[state]}
        row.update(
            {
                to_state: matrix.matrix[index][to_index]
                for to_index, to_state in enumerate(matrix.states)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


def _roll_rate_markdown(
    *,
    matrix,
    assumptions: dict,
    source_evidence: dict,
    warnings: list[dict],
) -> str:
    columns = ["期初状态", "基数", *matrix.states]
    rows = [
        [
            state,
            matrix.base_counts[state],
            *matrix.matrix[index],
        ]
        for index, state in enumerate(matrix.states)
    ]
    lines = [
        "# Roll-rate 转移矩阵",
        "",
        "口径：同一主体按时间排序后的相邻观测；不补齐缺失月份，也不代表月末快照迁徙。",
        "",
        f"- 数据集：`{source_evidence['dataset_id']}`",
        f"- 数据哈希：`{source_evidence['dataset_content_hash']}`",
        f"- 周期标签：{matrix.period}",
        "",
        "## 假设",
        "",
        "```json",
        json.dumps(assumptions, ensure_ascii=False, indent=2, allow_nan=False),
        "```",
        "",
        "## 转移矩阵",
        "",
        _markdown_table(columns, rows),
    ]
    if warnings:
        lines.extend(
            [
                "",
                "## 数据质量提示",
                "",
                *[
                    f"- {warning.get('message') or warning.get('code')}"
                    for warning in warnings
                ],
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _markdown_table(columns: list, rows: list[list]) -> str:
    def cell(value) -> str:
        return (
            str(value if value is not None else "")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    header = "| " + " | ".join(cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join(
            (header, separator, "| " + " | ".join("" for _ in columns) + " |")
        )
    return "\n".join(
        [
            header,
            separator,
            *["| " + " | ".join(cell(value) for value in row) + " |" for row in rows],
        ]
    )


def _optional_profit_params(payload) -> ProfitParams | None:
    return None if payload in (None, "") else _profit_params(dict(payload))


def _approval_profit_inputs(
    *,
    strategy_type: str,
    profit_params,
    ead_col: str | None,
    pd_col: str | None,
) -> tuple[dict | None, ApprovalProfitInputs | None]:
    if strategy_type not in {"approval", "reject"}:
        if profit_params not in (None, "") or ead_col is not None or pd_col is not None:
            raise StrategyError(
                "profit_params/ead_col/pd_col are only valid for approval/reject; "
                "use economics_inputs for limit or pricing"
            )
        return None, None
    if profit_params in (None, ""):
        # No economics was requested: the canonical envelope must stay empty.
        # The historical flat Tool projection alone retains expected_profit=0.0.
        return None, None
    if not ead_col or not pd_col:
        return {
            "expected_profit": None,
            "profit_note": (
                "已请求利润回测，但缺少 pd_col/ead_col，无法计算预期损失链，"
                "expected_profit 记为不可用（未用 0 冒充）。"
            ),
        }, None
    return None, ApprovalProfitInputs(
        params=_profit_params(dict(profit_params)),
        ead_col=ead_col,
        pd_col=pd_col,
    )


def _typed_economics_inputs(
    frame: pd.DataFrame,
    *,
    strategy_type: str,
    payload,
) -> dict | None:
    if payload in (None, ""):
        return None
    if strategy_type not in {"limit", "pricing"}:
        raise StrategyError(
            "economics_inputs are only valid for limit or pricing strategies"
        )
    values = dict(payload)
    required = (
        ("pd", "lgd", "utilization")
        if strategy_type == "limit"
        else (
            "ead",
            "pd",
            "lgd",
            "funding_rate",
            "term_months",
            "operating_cost_per_loan",
        )
    )
    allowed = {key for name in required for key in (f"{name}_col", f"{name}_value")}
    unsupported = sorted(set(values) - allowed)
    if unsupported:
        raise StrategyError(
            f"unsupported {strategy_type} economics_inputs: " + ", ".join(unsupported)
        )
    normalized: dict = {}
    missing: list[str] = []
    for name in required:
        column_key = f"{name}_col"
        value_key = f"{name}_value"
        has_column = values.get(column_key) not in (None, "")
        has_value = values.get(value_key) not in (None, "")
        if has_column == has_value:
            if has_column:
                raise StrategyError(
                    f"economics_inputs requires exactly one of {column_key} or "
                    f"{value_key}"
                )
            missing.append(f"{column_key}/{value_key}")
            continue
        if has_column:
            column = str(values[column_key])
            if column not in frame.columns:
                raise StrategyError(f"missing columns: {column}")
            normalized[name] = frame[column]
        else:
            if isinstance(values[value_key], bool):
                raise StrategyError(f"{value_key} must be numeric, not boolean")
            normalized[name] = float(values[value_key])
    if missing:
        raise StrategyError(
            f"{strategy_type} economics_inputs is incomplete; missing "
            + ", ".join(missing)
        )
    return normalized


def _label_coverage(total_rows: int, n_dropped: int) -> float:
    # drop_nan_labels semantics: coverage = labeled rows / total rows (DOM-11), so
    # callers see how much of the sample actually carried supervision signal.
    if total_rows <= 0:
        return 0.0
    return float((total_rows - n_dropped) / total_rows)


def _backtest_audit_summary(result: StrategyBacktestResult) -> dict[str, object]:
    """Keep audit rows useful without flattening typed result semantics."""

    if result.strategy_type in {"approval", "reject"}:
        return {
            "approve_rate": result.metrics.get("approve_rate"),
            "approve_bad_rate": result.metrics.get("approve_bad_rate"),
            "expected_profit": result.economics.get("expected_profit"),
        }
    if result.strategy_type == "limit":
        return {
            "total_limit": result.metrics.get("total_limit"),
            "mean_limit": result.metrics.get("mean_limit"),
            "expected_loss": result.economics.get("expected_loss"),
        }
    if result.strategy_type == "pricing":
        return {
            "mean_rate": result.metrics.get("mean_rate"),
            "profit": result.economics.get("profit"),
        }
    return {"segment_count": result.metrics.get("segment_count")}


def _backtest_id(
    dataset_id: str,
    result: BacktestRecord,
    *,
    source_dataset_content_hash: str | None = None,
) -> str:
    payload = {"dataset_id": dataset_id, "result": backtest_record_payload(result)}
    if not isinstance(result, StrategyBacktestResult):
        # Preserve historical legacy IDs byte-for-byte.  Typed envelopes use the
        # stricter canonical JSON path below; old rows and external references do
        # not change merely because V2 introduced a versioned result contract.
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"backtest-{digest[:12]}"
    if source_dataset_content_hash is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", source_dataset_content_hash):
            raise StrategyError(
                "source_dataset_content_hash must be a lowercase SHA256 digest"
            )
        payload["source_dataset_content_hash"] = source_dataset_content_hash
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"backtest-{digest[:12]}"


def _jsonable(value):
    if value is None:
        return None
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _unique(values: list[str | None]) -> list[str]:
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
