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

# This module is the builtin-template aggregation facade: marvis.orchestrator.
# templates.load_builtin_templates() imports it BY NAME, so importing it must
# register the full builtin template set, in this exact order, as a side
# effect. The template definitions themselves live in per-domain modules
# below; this file only wires them into the registry.
_register_builtin_template(SAMPLE_ECHO)
_register_builtin_template(MODEL_VALIDATION)
_register_builtin_template(STANDARD_MODELING)
_register_builtin_template(DATA_JOIN)
_register_builtin_template(LABEL_CONSTRUCTION)
_register_builtin_template(MODELING)
_register_builtin_template(MODELING_WITH_JOIN)
_register_builtin_template(FEATURE_ANALYSIS)
_register_builtin_template(FEATURE_ANALYSIS_WITH_JOIN)
_register_builtin_template(FEATURE_DERIVATION)
_register_builtin_template(STRATEGY_ANALYSIS)
_register_builtin_template(STRATEGY_DEVELOPMENT)
_register_builtin_template(RULE_STRATEGY)
_register_builtin_template(VINTAGE_ANALYSIS)
_register_builtin_template(SLICE_AGGREGATE)
_register_builtin_template(MONITORING_RUN)
_register_builtin_template(STRATEGY_MONITORING)
_register_builtin_template(PORTFOLIO_ANALYSIS)
_register_builtin_template(PORTFOLIO_ANALYSIS_NO_TREND)
