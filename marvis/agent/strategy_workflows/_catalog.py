from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ._analytics import (
    limit_pricing_confirmation,
    prepare_limit_pricing,
    prepare_profit,
    prepare_roll_rate,
    profit_confirmation,
    roll_rate_confirmation,
    validate_limit_pricing_inputs,
    validate_profit_inputs,
    validate_roll_rate_inputs,
)
from ._cross_voting import CROSS_VOTING_WORKFLOW_SPECS
from ._foundation_delivery import (
    FOUNDATION_DELIVERY_SPECS,
    validate_sample_design_v2_replay_inputs,
)
from ._interactive_tree_frontier import (
    interactive_tree_frontier_confirmation,
    interactive_tree_frontier_group_confirmation,
    prepare_interactive_tree_frontier,
    prepare_interactive_tree_frontier_group,
    validate_interactive_tree_frontier_group_inputs,
    validate_interactive_tree_frontier_inputs,
)
from ._model_score_comparison import (
    model_score_comparison_confirmation,
    prepare_model_score_comparison,
    validate_model_score_comparison_inputs,
)
from ._pool_workflows import POOL_WORKFLOW_SPECS
from ._tree_workflows import TREE_WORKFLOW_SPECS
from ._univariate_scorecard import (
    candidate_monthly_stability_confirmation,
    prepare_candidate_monthly_stability,
    prepare_scorecard_band_build,
    prepare_scorecard_cutoff_selection,
    prepare_scorecard_model_score_evidence,
    prepare_univariate_analysis,
    prepare_univariate_refinement,
    scorecard_band_build_confirmation,
    scorecard_cutoff_selection_confirmation,
    scorecard_model_score_evidence_confirmation,
    univariate_analysis_confirmation,
    univariate_refinement_confirmation,
    univariate_refinement_requirements,
    validate_candidate_monthly_stability_inputs,
    validate_scorecard_band_build_inputs,
    validate_scorecard_cutoff_selection_inputs,
    validate_scorecard_model_score_evidence_inputs,
    validate_univariate_analysis_inputs,
    validate_univariate_refinement_inputs,
)
from .contracts import (
    Confirmation,
    StrategyWorkflowRequirements,
    StrategyWorkflowSpec,
)


_FRESH_IDS = (
    "strategy_project_context",
    "strategy_sample_design_v2",
    "strategy_model_evidence_v2",
    "strategy_model_score_comparison_v2",
    "profit_calc",
    "roll_rate_matrix",
    "limit_pricing_matrix",
    "univariate_candidate_analysis",
    "univariate_candidate_refinement",
    "candidate_monthly_stability",
    "scorecard_model_score_evidence_build",
    "scorecard_band_build",
    "scorecard_cutoff_selection",
    "automatic_tree_candidate_build",
    "automatic_tree_apply",
    "automatic_tree_leaf_materialization",
    "interactive_tree_split_search",
    "interactive_tree_auto_continuation",
    "interactive_tree_revision",
    "interactive_tree_frontier_group_materialization",
    "interactive_tree_frontier_materialization",
    "voting_candidate_search",
    "voting_candidate_build_from_search",
    "voting_candidate_build",
    "cross_matrix_candidate_search",
    "cross_matrix_candidate_build_from_search",
    "cross_rule_search",
    "cross_rule_candidate_build_from_search",
    "cross_matrix_analysis",
    "cross_matrix_cell_selection",
    "strategy_pool_add_candidate",
    "strategy_pool_remove_entry",
    "strategy_pool_set_action",
    "strategy_pool_reorder",
    "strategy_pool_compile",
    "strategy_pool_materialize",
    "strategy_pool_apply",
    "strategy_pool_validation",
    "strategy_pool_impact",
    "strategy_impact_cube",
    "strategy_pool_stability",
    "strategy_dsl_delivery",
    "strategy_report_bundle_v2",
)

_MANUAL_IDS = frozenset(
    {
        "strategy_project_context",
        "strategy_sample_design_v2",
        "strategy_model_score_comparison_v2",
        "univariate_candidate_analysis",
        "cross_matrix_analysis",
        "automatic_tree_candidate_build",
        "univariate_candidate_refinement",
        "scorecard_model_score_evidence_build",
        "scorecard_band_build",
        "scorecard_cutoff_selection",
        "candidate_monthly_stability",
        "voting_candidate_search",
        "voting_candidate_build_from_search",
        "cross_matrix_candidate_search",
        "cross_matrix_candidate_build_from_search",
        "cross_rule_search",
        "cross_rule_candidate_build_from_search",
        "interactive_tree_split_search",
        "interactive_tree_auto_continuation",
        "interactive_tree_revision",
        "interactive_tree_frontier_group_materialization",
        "interactive_tree_frontier_materialization",
        "strategy_pool_add_candidate",
        "strategy_pool_compile",
        "strategy_pool_materialize",
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
        "strategy_pool_reorder",
        "strategy_pool_apply",
        "strategy_pool_validation",
        "strategy_pool_stability",
        "strategy_pool_impact",
        "strategy_impact_cube",
        "strategy_dsl_delivery",
        "strategy_report_bundle_v2",
    }
)

_LEGACY_REPLAY_IDS = ("strategy_sample_design",)
_NO_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=False,
    target=False,
    complete_labels=False,
)
_UNKNOWN_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=None,
    target=None,
    complete_labels=None,
)
_COMMON_CONFIRMATION_SUFFIX = (
    "请确认以上口径。确认后 Agent 只编排受信任工具；"
    "所有数字由平台确定性计算。"
)
_TREE_WORKFLOW_IDS = frozenset(
    spec.workflow_id for spec in TREE_WORKFLOW_SPECS
)
_PRESENTER_REFS_BY_WORKFLOW: Mapping[str, tuple[str, ...]] = {
    "strategy_project_context": ("materialize_project_context",),
    "strategy_sample_design_v2": (
        "materialize_sample_design",
        "materialize_sample_design_v2",
        "materialize_sample_design_v2_native",
    ),
    "strategy_model_evidence_v2": ("materialize_model_evidence_v2",),
    "strategy_model_score_comparison_v2": (
        "materialize_model_score_comparison_v2",
    ),
    "profit_calc": ("profit_calc",),
    "roll_rate_matrix": ("roll_rate_matrix",),
    "limit_pricing_matrix": ("limit_pricing_matrix",),
    "univariate_candidate_analysis": ("analyze_univariate_candidates",),
    "univariate_candidate_refinement": (
        "analyze_univariate_candidates",
        "refine_univariate_candidate",
    ),
    "candidate_monthly_stability": ("measure_candidate_monthly_stability",),
    "scorecard_model_score_evidence_build": (
        "train_model_with_evidence_v2",
        "materialize_model_score_evidence_v2",
    ),
    "scorecard_band_build": ("build_scorecard_band_asset",),
    "scorecard_cutoff_selection": (
        "materialize_scorecard_cutoff_selection",
    ),
    "automatic_tree_candidate_build": ("build_automatic_tree_candidate",),
    "automatic_tree_apply": ("apply_automatic_tree",),
    "automatic_tree_leaf_materialization": (
        "materialize_automatic_tree_leaf_fragment",
    ),
    "interactive_tree_split_search": (
        "search_interactive_tree_split_candidates",
    ),
    "interactive_tree_auto_continuation": (
        "auto_continue_interactive_tree",
    ),
    "interactive_tree_revision": ("revise_interactive_tree",),
    "interactive_tree_frontier_group_materialization": (
        "materialize_interactive_tree_frontier_group_selection",
    ),
    "interactive_tree_frontier_materialization": (
        "materialize_interactive_tree_frontier_selection",
    ),
    "voting_candidate_search": ("search_voting_candidates",),
    "voting_candidate_build_from_search": (
        "build_voting_candidate_from_search",
    ),
    "voting_candidate_build": ("build_voting_candidate",),
    "cross_matrix_candidate_search": ("search_cross_matrix_candidates",),
    "cross_matrix_candidate_build_from_search": (
        "build_cross_matrix_candidate_from_search",
    ),
    "cross_rule_search": ("search_cross_threshold_rules",),
    "cross_rule_candidate_build_from_search": (
        "build_cross_rule_candidate_from_search",
    ),
    "cross_matrix_analysis": (
        "analyze_univariate_candidates",
        "build_cross_matrix_candidate",
    ),
    "cross_matrix_cell_selection": (
        "materialize_cross_matrix_cell_selection",
    ),
    "strategy_pool_add_candidate": ("add_candidate_to_pool",),
    "strategy_pool_remove_entry": ("remove_pool_entry",),
    "strategy_pool_set_action": ("set_pool_entry_action",),
    "strategy_pool_reorder": ("reorder_strategy_pool",),
    "strategy_pool_compile": ("compile_strategy_pool",),
    "strategy_pool_materialize": ("materialize_strategy_from_pool",),
    "strategy_pool_apply": ("apply_strategy_pool",),
    "strategy_pool_validation": ("measure_strategy_pool_validation",),
    "strategy_pool_impact": ("measure_pool_impact",),
    "strategy_impact_cube": ("measure_strategy_impact_cube",),
    "strategy_pool_stability": (
        "measure_strategy_impact_cube",
        "measure_strategy_pool_stability",
    ),
    "strategy_dsl_delivery": ("export_strategy_delivery",),
    "strategy_report_bundle_v2": ("build_report_bundle_v2",),
    "strategy_sample_design": ("materialize_sample_design",),
}


def _with_common_confirmation_suffix(
    confirmation: Confirmation,
) -> Confirmation:
    def wrapped(inputs: Mapping[str, Any]) -> str:
        text = confirmation(inputs)
        if text.endswith(_COMMON_CONFIRMATION_SUFFIX):
            return text
        return f"{text}；{_COMMON_CONFIRMATION_SUFFIX}"

    return wrapped


def _canonicalize_imported_spec(
    spec: StrategyWorkflowSpec,
) -> StrategyWorkflowSpec:
    updates: dict[str, object] = {}
    if spec.workflow_id == "strategy_sample_design_v2":
        updates["replay_validator"] = validate_sample_design_v2_replay_inputs
    if spec.workflow_id in _TREE_WORKFLOW_IDS:
        assert spec.confirmation is not None
        updates["confirmation"] = _with_common_confirmation_suffix(
            spec.confirmation
        )
    return replace(spec, **updates) if updates else spec


def _with_presenter_refs(spec: StrategyWorkflowSpec) -> StrategyWorkflowSpec:
    if not spec.migrated:
        return spec
    try:
        presenter_refs = _PRESENTER_REFS_BY_WORKFLOW[spec.workflow_id]
    except KeyError as exc:  # pragma: no cover - import-time invariant
        raise RuntimeError(
            f"missing presenter refs for {spec.workflow_id}"
        ) from exc
    return replace(spec, presenter_refs=presenter_refs)


def build_catalog() -> tuple[StrategyWorkflowSpec, ...]:
    overrides = {
        "profit_calc": StrategyWorkflowSpec(
            workflow_id="profit_calc",
            fresh=True,
            replayable=True,
            manual=False,
            requirements=StrategyWorkflowRequirements(
                dataset=True,
                target=False,
                complete_labels=False,
            ),
            template_ids=("strategy_profit_analysis",),
            validator=validate_profit_inputs,
            confirmation=profit_confirmation,
            preparer=prepare_profit,
        ),
        "roll_rate_matrix": StrategyWorkflowSpec(
            workflow_id="roll_rate_matrix",
            fresh=True,
            replayable=True,
            manual=False,
            requirements=StrategyWorkflowRequirements(
                dataset=True,
                target=False,
                complete_labels=False,
            ),
            template_ids=("strategy_roll_rate_analysis",),
            validator=validate_roll_rate_inputs,
            confirmation=roll_rate_confirmation,
            preparer=prepare_roll_rate,
        ),
        "limit_pricing_matrix": StrategyWorkflowSpec(
            workflow_id="limit_pricing_matrix",
            fresh=True,
            replayable=True,
            manual=False,
            requirements=StrategyWorkflowRequirements(
                dataset=True,
                target=True,
                complete_labels=True,
            ),
            template_ids=("strategy_limit_pricing_analysis",),
            validator=validate_limit_pricing_inputs,
            confirmation=limit_pricing_confirmation,
            preparer=prepare_limit_pricing,
        ),
        "strategy_model_score_comparison_v2": StrategyWorkflowSpec(
            workflow_id="strategy_model_score_comparison_v2",
            fresh=True,
            replayable=True,
            manual=True,
            requirements=_NO_REQUIREMENTS,
            template_ids=("strategy_model_score_comparison_v2",),
            validator=validate_model_score_comparison_inputs,
            confirmation=model_score_comparison_confirmation,
            preparer=prepare_model_score_comparison,
        ),
        "univariate_candidate_analysis": StrategyWorkflowSpec(
            workflow_id="univariate_candidate_analysis",
            fresh=True,
            replayable=True,
            manual=True,
            requirements=StrategyWorkflowRequirements(
                dataset=True,
                target=True,
                complete_labels=True,
            ),
            template_ids=("strategy_univariate_candidate_analysis",),
            validator=validate_univariate_analysis_inputs,
            confirmation=univariate_analysis_confirmation,
            preparer=prepare_univariate_analysis,
        ),
        "univariate_candidate_refinement": StrategyWorkflowSpec(
            workflow_id="univariate_candidate_refinement",
            fresh=True,
            replayable=True,
            manual=True,
            requirements=univariate_refinement_requirements,
            template_ids=(
                "strategy_univariate_candidate_refinement",
                "strategy_univariate_candidate_refinement_existing",
            ),
            validator=validate_univariate_refinement_inputs,
            confirmation=univariate_refinement_confirmation,
            preparer=prepare_univariate_refinement,
        ),
        "candidate_monthly_stability": StrategyWorkflowSpec(
            workflow_id="candidate_monthly_stability",
            fresh=True,
            replayable=True,
            manual=True,
            requirements=_NO_REQUIREMENTS,
            template_ids=("strategy_candidate_monthly_stability",),
            validator=validate_candidate_monthly_stability_inputs,
            confirmation=candidate_monthly_stability_confirmation,
            preparer=prepare_candidate_monthly_stability,
        ),
        "scorecard_model_score_evidence_build": StrategyWorkflowSpec(
            workflow_id="scorecard_model_score_evidence_build",
            fresh=True,
            replayable=True,
            manual=True,
            requirements=_NO_REQUIREMENTS,
            template_ids=("strategy_scorecard_model_score_evidence_build",),
            validator=validate_scorecard_model_score_evidence_inputs,
            confirmation=scorecard_model_score_evidence_confirmation,
            preparer=prepare_scorecard_model_score_evidence,
        ),
        "scorecard_band_build": StrategyWorkflowSpec(
            workflow_id="scorecard_band_build",
            fresh=True,
            replayable=True,
            manual=True,
            requirements=_NO_REQUIREMENTS,
            template_ids=("strategy_scorecard_band_build",),
            validator=validate_scorecard_band_build_inputs,
            confirmation=scorecard_band_build_confirmation,
            preparer=prepare_scorecard_band_build,
        ),
        "scorecard_cutoff_selection": StrategyWorkflowSpec(
            workflow_id="scorecard_cutoff_selection",
            fresh=True,
            replayable=True,
            manual=True,
            requirements=_NO_REQUIREMENTS,
            template_ids=("strategy_scorecard_cutoff_selection",),
            validator=validate_scorecard_cutoff_selection_inputs,
            confirmation=scorecard_cutoff_selection_confirmation,
            preparer=prepare_scorecard_cutoff_selection,
        ),
        "interactive_tree_frontier_materialization": StrategyWorkflowSpec(
            workflow_id="interactive_tree_frontier_materialization",
            fresh=True,
            replayable=True,
            manual=True,
            requirements=_NO_REQUIREMENTS,
            template_ids=(
                "strategy_interactive_tree_frontier_materialization",
            ),
            validator=validate_interactive_tree_frontier_inputs,
            confirmation=interactive_tree_frontier_confirmation,
            preparer=prepare_interactive_tree_frontier,
        ),
        "interactive_tree_frontier_group_materialization": (
            StrategyWorkflowSpec(
                workflow_id=(
                    "interactive_tree_frontier_group_materialization"
                ),
                fresh=True,
                replayable=True,
                manual=True,
                requirements=_NO_REQUIREMENTS,
                template_ids=(
                    "strategy_interactive_tree_frontier_group_materialization",
                ),
                validator=validate_interactive_tree_frontier_group_inputs,
                confirmation=interactive_tree_frontier_group_confirmation,
                preparer=prepare_interactive_tree_frontier_group,
            )
        ),
    }
    for imported_spec in (
        *FOUNDATION_DELIVERY_SPECS,
        *TREE_WORKFLOW_SPECS,
        *CROSS_VOTING_WORKFLOW_SPECS,
        *POOL_WORKFLOW_SPECS,
    ):
        if imported_spec.workflow_id in overrides:
            raise RuntimeError(
                f"duplicate strategy workflow override: {imported_spec.workflow_id}"
            )
        overrides[imported_spec.workflow_id] = _canonicalize_imported_spec(
            imported_spec
        )
    fresh_specs = tuple(
        overrides.get(
            workflow_id,
            StrategyWorkflowSpec(
                workflow_id=workflow_id,
                fresh=True,
                replayable=True,
                manual=workflow_id in _MANUAL_IDS,
                requirements=_UNKNOWN_REQUIREMENTS,
            ),
        )
        for workflow_id in _FRESH_IDS
    )
    legacy_specs = tuple(
        overrides.get(
            workflow_id,
            StrategyWorkflowSpec(
                workflow_id=workflow_id,
                fresh=False,
                replayable=True,
                manual=False,
                requirements=_UNKNOWN_REQUIREMENTS,
            ),
        )
        for workflow_id in _LEGACY_REPLAY_IDS
    )
    return tuple(
        _with_presenter_refs(spec) for spec in (*fresh_specs, *legacy_specs)
    )
