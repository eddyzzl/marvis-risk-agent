from __future__ import annotations

from marvis.orchestrator.templates import _register_builtin_template
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
    RULE_STRATEGY,
    SLICE_AGGREGATE,
    STRATEGY_ANALYSIS,
    STRATEGY_DEVELOPMENT,
    VINTAGE_ANALYSIS,
)
from marvis.orchestrator.templates.validation import MODEL_VALIDATION

# This module is the builtin-template aggregation facade: the template
# definitions themselves live in per-domain modules below; this file only wires
# them into the registry in the product display order.
BUILTIN_TEMPLATES = (
    SAMPLE_ECHO,
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
