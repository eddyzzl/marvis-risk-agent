"""Backward-compatible namespace for marvis.agent.turn_handlers.

Physical implementation is organized across lane files; this package executes
them, in dependency order, into the single module namespace below so that
name resolution and monkeypatching behave exactly like the original
monolithic module.
"""
from __future__ import annotations

from pathlib import Path

_LANES = [
    'core',
    'cross',
    'impact',
    'model_evidence',
    'pool',
    'refinement_report',
    'sample_design',
    'scorecard',
    'tree',
    'voting',
]

_ns = globals()
_pkg = Path(__file__).parent
for _lane in _LANES:
    _lane_path = _pkg / f"{_lane}.py"
    _code = compile(_lane_path.read_text(encoding="utf-8"), str(_lane_path), "exec")
    exec(_code, _ns, _ns)  # nosec B102 - executes only our own lane files into one namespace
del _lane, _lane_path, _code, _pkg, _ns

__all__ = [
    'CompiledStrategyRequestDraft',
    'FRESH_STANDARD_STRATEGY_WORKFLOWS',
    'LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS',
    'REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS',
    'STANDARD_STRATEGY_WORKFLOWS',
    'UNIVARIATE_BINNING_METHODS',
    'UNIVARIATE_REFINEMENT_METHODS',
    'STRATEGY_REQUEST_KINDS',
    'STRATEGY_OPERATIONS',
    'STRATEGY_REQUEST_JSON_SCHEMA',
    'STRATEGY_TYPES',
    'StrategyRequestCompilation',
    'StrategyRequestDraft',
    'StandardWorkflowRequestDraft',
    'compile_strategy_request',
    'strategy_request_confirmation_text',
    'utterance_targets_candidate_monthly_stability',
    'utterance_targets_interactive_tree_frontier_group_materialization',
    'utterance_targets_interactive_tree_frontier_materialization',
    'utterance_targets_model_score_comparison_v2',
    'utterance_targets_scorecard_band_build',
    'utterance_targets_scorecard_cutoff_selection',
    'utterance_targets_strategy_dsl_delivery',
    'utterance_targets_strategy_pool_materialize',
    'utterance_targets_strategy_pool_stability',
    'utterance_targets_strategy_project_context',
    'utterance_targets_strategy_report_bundle_v2',
    'utterance_targets_strategy_sample_design',
    'validate_strategy_request',
]
