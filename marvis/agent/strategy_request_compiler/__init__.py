"""Stable public interface for strategy request compilation.

Grammar modules own their functions and imports. This facade preserves the
published request types and entry points without executing source files or
sharing implementation globals.
"""

from marvis.agent.strategy_workflows import (
    FRESH_STANDARD_STRATEGY_WORKFLOWS as FRESH_STANDARD_STRATEGY_WORKFLOWS,
)
from marvis.agent.strategy_workflows import (
    LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS as LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS,
)
from marvis.agent.strategy_workflows import (
    REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS as REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS,
)
from marvis.agent.strategy_workflows._univariate_scorecard import (
    UNIVARIATE_BINNING_METHODS as UNIVARIATE_BINNING_METHODS,
)
from marvis.agent.strategy_workflows._univariate_scorecard import (
    UNIVARIATE_REFINEMENT_METHODS as UNIVARIATE_REFINEMENT_METHODS,
)

from .contracts import (
    CompiledStrategyRequestDraft as CompiledStrategyRequestDraft,
)
from .contracts import (
    StandardWorkflowRequestDraft as StandardWorkflowRequestDraft,
)
from .contracts import (
    StrategyRequestCompilation as StrategyRequestCompilation,
)
from .contracts import (
    StrategyRequestDraft as StrategyRequestDraft,
)
from .core import (
    _SYSTEM as _SYSTEM,
)
from .core import (
    STANDARD_STRATEGY_WORKFLOWS as STANDARD_STRATEGY_WORKFLOWS,
)
from .core import (
    STRATEGY_OPERATIONS as STRATEGY_OPERATIONS,
)
from .core import (
    STRATEGY_REQUEST_JSON_SCHEMA as STRATEGY_REQUEST_JSON_SCHEMA,
)
from .core import (
    STRATEGY_REQUEST_KINDS as STRATEGY_REQUEST_KINDS,
)
from .core import (
    STRATEGY_TYPES as STRATEGY_TYPES,
)
from .core import (
    compile_strategy_request as compile_strategy_request,
)
from .core import (
    strategy_request_confirmation_text as strategy_request_confirmation_text,
)
from .core import (
    validate_strategy_request as validate_strategy_request,
)
from .impact import (
    utterance_targets_strategy_impact_cube as utterance_targets_strategy_impact_cube,
)
from .model_evidence import (
    utterance_targets_model_score_comparison_v2 as utterance_targets_model_score_comparison_v2,
)
from .pool import (
    utterance_targets_strategy_pool_materialize as utterance_targets_strategy_pool_materialize,
)
from .pool import (
    utterance_targets_strategy_pool_stability as utterance_targets_strategy_pool_stability,
)
from .refinement_report import (
    utterance_targets_strategy_dsl_delivery as utterance_targets_strategy_dsl_delivery,
)
from .refinement_report import (
    utterance_targets_strategy_project_context as utterance_targets_strategy_project_context,
)
from .refinement_report import (
    utterance_targets_strategy_report_bundle_v2 as utterance_targets_strategy_report_bundle_v2,
)
from .sample_design import (
    _sample_design_v2_has_chained_operation as _sample_design_v2_has_chained_operation,
)
from .sample_design import (
    _sample_v2_predicate_grounded as _sample_v2_predicate_grounded,
)
from .sample_design import (
    _sample_v2_predicate_semantics_grounded as _sample_v2_predicate_semantics_grounded,
)
from .sample_design import (
    utterance_targets_strategy_sample_design as utterance_targets_strategy_sample_design,
)
from .scorecard import (
    utterance_targets_candidate_monthly_stability as utterance_targets_candidate_monthly_stability,
)
from .scorecard import (
    utterance_targets_scorecard_band_build as utterance_targets_scorecard_band_build,
)
from .scorecard import (
    utterance_targets_scorecard_cutoff_selection as utterance_targets_scorecard_cutoff_selection,
)
from .tree import (
    utterance_targets_interactive_tree_frontier_group_materialization as utterance_targets_interactive_tree_frontier_group_materialization,
)
from .tree import (
    utterance_targets_interactive_tree_frontier_materialization as utterance_targets_interactive_tree_frontier_materialization,
)

__all__ = [
    "CompiledStrategyRequestDraft",
    "FRESH_STANDARD_STRATEGY_WORKFLOWS",
    "LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS",
    "REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS",
    "STANDARD_STRATEGY_WORKFLOWS",
    "UNIVARIATE_BINNING_METHODS",
    "UNIVARIATE_REFINEMENT_METHODS",
    "STRATEGY_REQUEST_KINDS",
    "STRATEGY_OPERATIONS",
    "STRATEGY_REQUEST_JSON_SCHEMA",
    "STRATEGY_TYPES",
    "StrategyRequestCompilation",
    "StrategyRequestDraft",
    "StandardWorkflowRequestDraft",
    "compile_strategy_request",
    "strategy_request_confirmation_text",
    "utterance_targets_candidate_monthly_stability",
    "utterance_targets_interactive_tree_frontier_group_materialization",
    "utterance_targets_interactive_tree_frontier_materialization",
    "utterance_targets_model_score_comparison_v2",
    "utterance_targets_scorecard_band_build",
    "utterance_targets_scorecard_cutoff_selection",
    "utterance_targets_strategy_dsl_delivery",
    "utterance_targets_strategy_pool_materialize",
    "utterance_targets_strategy_pool_stability",
    "utterance_targets_strategy_project_context",
    "utterance_targets_strategy_report_bundle_v2",
    "utterance_targets_strategy_sample_design",
    "validate_strategy_request",
]
