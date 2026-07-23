from marvis.packs.strategy.backtest import backtest_strategy
from marvis.packs.strategy.backtest_compat import (
    BacktestRecord,
    approval_backtest_projection,
    backtest_record_payload,
)
from marvis.packs.strategy.bands import (
    CutoffBandsResult,
    RedFlag,
    ScoreBand,
    design_cutoff_bands,
)
from marvis.packs.strategy.compare import (
    CompareCell,
    CompareResult,
    compare_strategies,
)
from marvis.packs.strategy.contracts import (
    BacktestResult,
    ProfitResult,
    RollRateMatrix,
    Strategy,
    StrategyRule,
    TradeoffPoint,
    VintageCurve,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.dsl import (
    STRATEGY_DSL_SCHEMA_VERSION,
    StrategyAction,
    StrategyRuleSpec,
    StrategySpec,
    canonical_strategy_json,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.evaluator import (
    FrameEvaluation,
    RowEvaluation,
    evaluate_expression,
    evaluate_expression_frame,
    evaluate_strategy_frame,
    evaluate_strategy_row,
    evaluate_strategy_rows,
)
from marvis.packs.strategy.economics import limit_metrics, pricing_metrics
from marvis.packs.strategy.pricing import (
    LimitPricingResult,
    PricingCell,
    PricingParams,
    limit_pricing_matrix,
)
from marvis.packs.strategy.pool_impact import (
    STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
    STRATEGY_POOL_IMPACT_SCHEMA_VERSION,
    build_strategy_pool_impact_assessment,
    canonical_strategy_pool_impact_json,
    validate_strategy_pool_impact_assessment,
)
from marvis.packs.strategy.pool_validation import (
    STRATEGY_POOL_VALIDATION_PRODUCER_VERSION,
    STRATEGY_POOL_VALIDATION_SCHEMA_VERSION,
    build_strategy_pool_validation_evidence,
    canonical_strategy_pool_validation_json,
    validate_strategy_pool_validation_evidence,
)
from marvis.packs.strategy.profit import ProfitParams, profit_calc, vintage_profit
from marvis.packs.strategy.roll_rate import roll_rate_matrix
from marvis.packs.strategy.rules import CandidateRule, evaluate_rule_set, mine_rules
from marvis.packs.strategy.strategy import (
    apply_strategy,
    build_strategy,
    build_strategy_from_spec,
    evaluate_condition_mask,
)
from marvis.packs.strategy.tradeoff import (
    recommend_operating_point,
    tradeoff_feasible_flags,
    tradeoff_view,
)
from marvis.packs.strategy.typed_backtest import (
    STRATEGY_BACKTEST_SCHEMA_VERSION,
    ApprovalProfitInputs,
    StrategyBacktestResult,
    run_typed_backtest,
)
from marvis.packs.strategy.vintage import vintage_curve, vintage_summary

__all__ = [
    "BacktestResult",
    "BacktestRecord",
    "CompareCell",
    "CompareResult",
    "CutoffBandsResult",
    "FrameEvaluation",
    "ProfitParams",
    "ProfitResult",
    "RedFlag",
    "RollRateMatrix",
    "ScoreBand",
    "STRATEGY_DSL_SCHEMA_VERSION",
    "STRATEGY_BACKTEST_SCHEMA_VERSION",
    "STRATEGY_POOL_IMPACT_PRODUCER_VERSION",
    "STRATEGY_POOL_IMPACT_SCHEMA_VERSION",
    "STRATEGY_POOL_VALIDATION_PRODUCER_VERSION",
    "STRATEGY_POOL_VALIDATION_SCHEMA_VERSION",
    "Strategy",
    "StrategyAction",
    "StrategyBacktestResult",
    "StrategyError",
    "StrategyRule",
    "StrategyRuleSpec",
    "StrategySpec",
    "TradeoffPoint",
    "VintageCurve",
    "ApprovalProfitInputs",
    "approval_backtest_projection",
    "apply_strategy",
    "backtest_strategy",
    "backtest_record_payload",
    "build_strategy",
    "build_strategy_from_spec",
    "build_strategy_pool_impact_assessment",
    "build_strategy_pool_validation_evidence",
    "canonical_strategy_json",
    "canonical_strategy_pool_impact_json",
    "canonical_strategy_pool_validation_json",
    "compare_strategies",
    "limit_pricing_matrix",
    "limit_metrics",
    "LimitPricingResult",
    "PricingCell",
    "PricingParams",
    "design_cutoff_bands",
    "profit_calc",
    "pricing_metrics",
    "recommend_operating_point",
    "tradeoff_feasible_flags",
    "roll_rate_matrix",
    "run_typed_backtest",
    "tradeoff_view",
    "validate_strategy_pool_impact_assessment",
    "validate_strategy_pool_validation_evidence",
    "vintage_curve",
    "vintage_summary",
    "CandidateRule",
    "evaluate_condition_mask",
    "evaluate_expression",
    "evaluate_expression_frame",
    "evaluate_rule_set",
    "evaluate_strategy_row",
    "evaluate_strategy_rows",
    "evaluate_strategy_frame",
    "mine_rules",
    "parse_strategy_spec",
    "RowEvaluation",
    "strategy_spec_hash",
    "vintage_profit",
]
