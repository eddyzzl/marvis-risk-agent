"""Task-type -> driver-turn registry (executed last by __init__.py)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes
    from . import TASK_TYPE_DATA_JOIN
    from . import TASK_TYPE_FEATURE_ANALYSIS
    from . import TASK_TYPE_MODELING
    from . import TASK_TYPE_PORTFOLIO
    from . import TASK_TYPE_STRATEGY
    from . import TASK_TYPE_VINTAGE
    from . import run_feature_driver_turn
    from . import run_join_driver_turn
    from . import run_modeling_driver_turn
    from . import run_portfolio_driver_turn
    from . import run_strategy_driver_turn
    from . import run_vintage_driver_turn

DRIVER_TURN_FUNCS = {
    TASK_TYPE_MODELING: run_modeling_driver_turn,
    TASK_TYPE_DATA_JOIN: run_join_driver_turn,
    TASK_TYPE_FEATURE_ANALYSIS: run_feature_driver_turn,
    TASK_TYPE_STRATEGY: run_strategy_driver_turn,
    TASK_TYPE_VINTAGE: run_vintage_driver_turn,
    TASK_TYPE_PORTFOLIO: run_portfolio_driver_turn,
}
