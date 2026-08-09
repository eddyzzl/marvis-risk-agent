"""Public Strategy pack interface with lazy compatibility exports.

Importing one Strategy leaf module must not initialize the full workflow graph.
The package keeps its historical ``from marvis.packs.strategy import ...``
interface, while resolving each public name only when a caller first uses it.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "backtest_strategy": "marvis.packs.strategy.backtest",
    "BacktestRecord": "marvis.packs.strategy.backtest_compat",
    "approval_backtest_projection": "marvis.packs.strategy.backtest_compat",
    "backtest_record_payload": "marvis.packs.strategy.backtest_compat",
    "CutoffBandsResult": "marvis.packs.strategy.bands",
    "RedFlag": "marvis.packs.strategy.bands",
    "ScoreBand": "marvis.packs.strategy.bands",
    "design_cutoff_bands": "marvis.packs.strategy.bands",
    "CompareCell": "marvis.packs.strategy.compare",
    "CompareResult": "marvis.packs.strategy.compare",
    "compare_strategies": "marvis.packs.strategy.compare",
    "BacktestResult": "marvis.packs.strategy.contracts",
    "ProfitResult": "marvis.packs.strategy.contracts",
    "RollRateMatrix": "marvis.packs.strategy.contracts",
    "Strategy": "marvis.packs.strategy.contracts",
    "StrategyRule": "marvis.packs.strategy.contracts",
    "TradeoffPoint": "marvis.packs.strategy.contracts",
    "VintageCurve": "marvis.packs.strategy.contracts",
    "StrategyError": "marvis.packs.strategy.errors",
    "STRATEGY_DSL_SCHEMA_VERSION": "marvis.packs.strategy.dsl",
    "StrategyAction": "marvis.packs.strategy.dsl",
    "StrategyRuleSpec": "marvis.packs.strategy.dsl",
    "StrategySpec": "marvis.packs.strategy.dsl",
    "canonical_strategy_json": "marvis.packs.strategy.dsl",
    "parse_strategy_spec": "marvis.packs.strategy.dsl",
    "strategy_spec_hash": "marvis.packs.strategy.dsl",
    "FrameEvaluation": "marvis.packs.strategy.evaluator",
    "RowEvaluation": "marvis.packs.strategy.evaluator",
    "evaluate_expression": "marvis.packs.strategy.evaluator",
    "evaluate_expression_frame": "marvis.packs.strategy.evaluator",
    "evaluate_strategy_frame": "marvis.packs.strategy.evaluator",
    "evaluate_strategy_row": "marvis.packs.strategy.evaluator",
    "evaluate_strategy_rows": "marvis.packs.strategy.evaluator",
    "limit_metrics": "marvis.packs.strategy.economics",
    "pricing_metrics": "marvis.packs.strategy.economics",
    "STRATEGY_IMPACT_CUBE_PRODUCER_VERSION": "marvis.packs.strategy.impact_cube",
    "STRATEGY_IMPACT_CUBE_SCHEMA_VERSION": "marvis.packs.strategy.impact_cube",
    "STRATEGY_IMPACT_SLICE_SCHEMA_VERSION": "marvis.packs.strategy.impact_cube",
    "build_strategy_impact_cube": "marvis.packs.strategy.impact_cube",
    "canonical_strategy_impact_cube_json": "marvis.packs.strategy.impact_cube",
    "validate_strategy_impact_cube": "marvis.packs.strategy.impact_cube",
    "LimitPricingResult": "marvis.packs.strategy.pricing",
    "PricingCell": "marvis.packs.strategy.pricing",
    "PricingParams": "marvis.packs.strategy.pricing",
    "limit_pricing_matrix": "marvis.packs.strategy.pricing",
    "STRATEGY_POOL_IMPACT_PRODUCER_VERSION": "marvis.packs.strategy.pool_impact",
    "STRATEGY_POOL_IMPACT_SCHEMA_VERSION": "marvis.packs.strategy.pool_impact",
    "build_strategy_pool_impact_assessment": "marvis.packs.strategy.pool_impact",
    "canonical_strategy_pool_impact_json": "marvis.packs.strategy.pool_impact",
    "validate_strategy_pool_impact_assessment": "marvis.packs.strategy.pool_impact",
    "STRATEGY_POOL_VALIDATION_PRODUCER_VERSION": (
        "marvis.packs.strategy.pool_validation"
    ),
    "STRATEGY_POOL_VALIDATION_SCHEMA_VERSION": "marvis.packs.strategy.pool_validation",
    "build_strategy_pool_validation_evidence": "marvis.packs.strategy.pool_validation",
    "canonical_strategy_pool_validation_json": "marvis.packs.strategy.pool_validation",
    "validate_strategy_pool_validation_evidence": "marvis.packs.strategy.pool_validation",
    "ProfitParams": "marvis.packs.strategy.profit",
    "profit_calc": "marvis.packs.strategy.profit",
    "vintage_profit": "marvis.packs.strategy.profit",
    "roll_rate_matrix": "marvis.packs.strategy.roll_rate",
    "CandidateRule": "marvis.packs.strategy.rules",
    "evaluate_rule_set": "marvis.packs.strategy.rules",
    "mine_rules": "marvis.packs.strategy.rules",
    "apply_strategy": "marvis.packs.strategy.strategy",
    "build_strategy": "marvis.packs.strategy.strategy",
    "build_strategy_from_spec": "marvis.packs.strategy.strategy",
    "evaluate_condition_mask": "marvis.packs.strategy.strategy",
    "recommend_operating_point": "marvis.packs.strategy.tradeoff",
    "tradeoff_feasible_flags": "marvis.packs.strategy.tradeoff",
    "tradeoff_view": "marvis.packs.strategy.tradeoff",
    "STRATEGY_BACKTEST_SCHEMA_VERSION": "marvis.packs.strategy.typed_backtest",
    "ApprovalProfitInputs": "marvis.packs.strategy.typed_backtest",
    "StrategyBacktestResult": "marvis.packs.strategy.typed_backtest",
    "run_typed_backtest": "marvis.packs.strategy.typed_backtest",
    "vintage_curve": "marvis.packs.strategy.vintage",
    "vintage_summary": "marvis.packs.strategy.vintage",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
