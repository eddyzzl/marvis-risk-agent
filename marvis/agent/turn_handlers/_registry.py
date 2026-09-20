"""registry for governed Agent turns."""

from __future__ import annotations

from marvis.domain import TASK_TYPE_DATA_JOIN
from marvis.domain import TASK_TYPE_FEATURE_ANALYSIS
from marvis.domain import TASK_TYPE_MODELING
from marvis.domain import TASK_TYPE_PORTFOLIO
from marvis.domain import TASK_TYPE_STRATEGY
from marvis.domain import TASK_TYPE_VINTAGE
from . import feature as feature_lane
from . import join as join_lane
from . import modeling as modeling_lane
from . import portfolio as portfolio_lane
from . import strategy_turns as strategy_turns_lane
from . import vintage as vintage_lane


DRIVER_TURN_FUNCS = {
    TASK_TYPE_MODELING: modeling_lane.run_modeling_driver_turn,
    TASK_TYPE_DATA_JOIN: join_lane.run_join_driver_turn,
    TASK_TYPE_FEATURE_ANALYSIS: feature_lane.run_feature_driver_turn,
    TASK_TYPE_STRATEGY: strategy_turns_lane.run_strategy_driver_turn,
    TASK_TYPE_VINTAGE: vintage_lane.run_vintage_driver_turn,
    TASK_TYPE_PORTFOLIO: portfolio_lane.run_portfolio_driver_turn,
}
