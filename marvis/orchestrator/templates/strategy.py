from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.plugins.manifest import ToolRef


STRATEGY_ANALYSIS = WorkflowTemplate(
    id="strategy_analysis",
    title="策略分析与回测",
    goal_patterns=("策略分析", "策略回测", "策略权衡", "strategy analysis", "strategy backtest"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("score_col", True, "task_context", "Score column"),
        SlotSpec("strategy_type", True, "user", "Strategy type"),
        SlotSpec("rules", True, "user", "Ordered strategy rules"),
        SlotSpec("default_decision", True, "user", "Fallback decision"),
    ),
    steps=(
        StepTemplate(
            title="构造策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_type": "{slot:strategy_type}",
                "rules": "{slot:rules}",
                "score_col": "{slot:score_col}",
                "default_decision": "{slot:default_decision}",
                "description": "Workflow generated strategy candidate",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="回测策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "target_col": "{slot:target_col}",
            },
            depends_on_titles=("构造策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("range", {"field": "approval_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "approved_bad_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "rejected_bad_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "expected_profit"}),
            ),
            decision_point=True,
            needs_confirmation=True,
        ),
        StepTemplate(
            title="生成策略权衡视图",
            tool_ref=ToolRef("strategy", "tradeoff_view"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "score_col": "{slot:score_col}",
                "target_col": "{slot:target_col}",
            },
            depends_on_titles=("回测策略",),
            post_checks=(PostCheck("nonempty", {"field": "points"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

VINTAGE_ANALYSIS = WorkflowTemplate(
    id="vintage_analysis",
    title="Vintage 风险分析",
    goal_patterns=("风险分析", "vintage", "vintage analysis", "账龄分析"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered vintage dataset id"),
        SlotSpec("cohort_col", True, "task_context", "Cohort/month column"),
        SlotSpec("mob_col", True, "task_context", "Month-on-book column"),
        SlotSpec("bad_col", True, "task_context", "Binary bad/default target column"),
        SlotSpec("mob_max", False, "task_context", "Maximum MOB to render"),
        SlotSpec("ref_mob", False, "task_context", "Reference MOB for trend summary"),
    ),
    steps=(
        StepTemplate(
            title="计算 Vintage 曲线",
            tool_ref=ToolRef("strategy", "vintage_curve"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "cohort_col": "{slot:cohort_col}",
                "mob_col": "{slot:mob_col}",
                "bad_col": "{slot:bad_col}",
                "mob_max": "{slot:mob_max}",
                "ref_mob": "{slot:ref_mob}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "cohorts"}),),
            decision_point=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)
