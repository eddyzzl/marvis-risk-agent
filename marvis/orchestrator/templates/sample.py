from __future__ import annotations

from marvis.orchestrator.templates import _register_builtin_template
from marvis.orchestrator.templates.data import DATASET_DESCRIPTIVE_ANALYSIS
from marvis.orchestrator.templates.data_export import DATASET_EXPORT
from marvis.orchestrator.templates.data_transform import DATASET_TRANSFORM
from marvis.orchestrator.templates.feature import (
    FEATURE_ANALYSIS,
    FEATURE_ANALYSIS_WITH_JOIN,
    FEATURE_DERIVATION,
)
from marvis.orchestrator.templates.join import DATA_JOIN
from marvis.orchestrator.templates.labeling import LABEL_CONSTRUCTION
from marvis.orchestrator.templates.modeling import (
    MODELING,
    MODELING_WITH_JOIN,
    STANDARD_MODELING,
)
from marvis.orchestrator.templates.monitoring import MONITORING_RUN, STRATEGY_MONITORING
from marvis.orchestrator.templates.portfolio import (
    PORTFOLIO_ANALYSIS,
    PORTFOLIO_ANALYSIS_NO_TREND,
)
from marvis.orchestrator.templates.sample_echo import SAMPLE_ECHO
from marvis.orchestrator.templates.strategy import (
    DETERMINISTIC_STRATEGY_CANDIDATE_DEVELOPMENT,
    RULE_STRATEGY,
    SLICE_AGGREGATE,
    STORED_STRATEGY_ADOPTION,
    STORED_STRATEGY_APPLY,
    STORED_STRATEGY_EVALUATION,
    STORED_STRATEGY_REPORT,
    STRATEGY_ANALYSIS,
    STRATEGY_DEVELOPMENT,
    STRATEGY_LIMIT_PRICING_ANALYSIS,
    STRATEGY_PROFIT_ANALYSIS,
    STRATEGY_ROLL_RATE_ANALYSIS,
    STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS,
    STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT,
    STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT_EXISTING,
    TYPED_STRATEGY_APPLY,
    TYPED_STRATEGY_BUILD,
    TYPED_STRATEGY_EVALUATION,
    VINTAGE_ANALYSIS,
)
from marvis.orchestrator.templates.validation import MODEL_VALIDATION

# This module is the builtin-template aggregation facade: the template
# definitions themselves live in per-domain modules below; this file only wires
# them into the registry in the product display order.
BUILTIN_TEMPLATES = (
    SAMPLE_ECHO,
    DATASET_DESCRIPTIVE_ANALYSIS,
    DATASET_TRANSFORM,
    DATASET_EXPORT,
    MODEL_VALIDATION,
    STANDARD_MODELING,
    DATA_JOIN,
    LABEL_CONSTRUCTION,
    MODELING,
    MODELING_WITH_JOIN,
    FEATURE_ANALYSIS,
    FEATURE_ANALYSIS_WITH_JOIN,
    FEATURE_DERIVATION,
    STRATEGY_ANALYSIS,
    STRATEGY_DEVELOPMENT,
    STRATEGY_PROFIT_ANALYSIS,
    STRATEGY_ROLL_RATE_ANALYSIS,
    STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS,
    STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT,
    STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT_EXISTING,
    STRATEGY_LIMIT_PRICING_ANALYSIS,
    DETERMINISTIC_STRATEGY_CANDIDATE_DEVELOPMENT,
    TYPED_STRATEGY_APPLY,
    TYPED_STRATEGY_BUILD,
    TYPED_STRATEGY_EVALUATION,
    STORED_STRATEGY_EVALUATION,
    STORED_STRATEGY_REPORT,
    STORED_STRATEGY_APPLY,
    STORED_STRATEGY_ADOPTION,
    RULE_STRATEGY,
    VINTAGE_ANALYSIS,
    SLICE_AGGREGATE,
    MONITORING_RUN,
    STRATEGY_MONITORING,
    PORTFOLIO_ANALYSIS,
    PORTFOLIO_ANALYSIS_NO_TREND,
)


def register_all_builtin_templates() -> None:
    for template in BUILTIN_TEMPLATES:
        _register_builtin_template(template)


register_all_builtin_templates()
