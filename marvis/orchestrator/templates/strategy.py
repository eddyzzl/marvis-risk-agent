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


STRATEGY_DEVELOPMENT = WorkflowTemplate(
    # S2 conversational strategy-development template (new id; the lightweight
    # strategy_analysis entry stays untouched). Flow: tradeoff scan (direction
    # self-check) -> [confirm] design cutoff bands -> build strategy from the
    # recommended rules -> [confirm] backtest -> [optional] compare vs baseline
    # -> [mandatory confirm] adopt -> render doc. goal_patterns are disjoint from
    # strategy_analysis so keyword routing never crosses the two.
    id="strategy_development",
    title="策略开发",
    goal_patterns=("策略开发", "开发策略", "设计cutoff", "分数带策略", "strategy development"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("score_col", True, "task_context", "Score column"),
        SlotSpec("score_direction", False, "task_context", "Score direction if a model artifact injected one"),
        SlotSpec("objective", False, "user", "max_profit or max_approval"),
        SlotSpec("max_bad_rate", False, "user", "Max approved bad rate constraint"),
        SlotSpec("min_approval_rate", False, "user", "Min approval rate constraint"),
        SlotSpec("profit_params", False, "user", "Profit parameters for expected-profit"),
        SlotSpec("strategy_type", False, "user", "Strategy type (default approval)"),
        SlotSpec("baseline_strategy_id", False, "user", "Baseline strategy id for the optional compare step"),
        SlotSpec("adoption_reason", True, "user", "Reason recorded when the strategy is adopted"),
    ),
    steps=(
        StepTemplate(
            title="权衡扫描",
            tool_ref=ToolRef("strategy", "tradeoff_view"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "score_col": "{slot:score_col}",
                "target_col": "{slot:target_col}",
                "score_direction": "{slot:score_direction}",
                "objective": "{slot:objective}",
                "max_bad_rate": "{slot:max_bad_rate}",
                "min_approval_rate": "{slot:min_approval_rate}",
                "profit_params": "{slot:profit_params}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "points"}),),
            decision_point=True,
        ),
        StepTemplate(
            title="设计分数带",
            tool_ref=ToolRef("strategy", "design_cutoff_bands"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "score_col": "{slot:score_col}",
                "target_col": "{slot:target_col}",
                "score_direction": "{slot:score_direction}",
                "objective": "{slot:objective}",
                "max_bad_rate": "{slot:max_bad_rate}",
                "min_approval_rate": "{slot:min_approval_rate}",
                "profit_params": "{slot:profit_params}",
                # Literal default (not {slot:band_edges}): apply_adjust's generic
                # gate override mechanism (agent/gate_execution_adapter.py) only
                # picks up a key that already exists in the step's instantiated
                # inputs -- omitted-slot keys get dropped entirely by
                # planner._fill_inputs. Baking the key in with a null default
                # (mirroring modeling's split_config passthrough default at
                # templates/modeling.py:72) is what makes the manual band_edges=[...]
                # structured gate override actually reach this step.
                "band_edges": None,
            },
            depends_on_titles=("权衡扫描",),
            post_checks=(PostCheck("nonempty", {"field": "bands"}),),
            needs_confirmation=True,
        ),
        StepTemplate(
            title="构造策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                # SlotSpec has no default-value mechanism (unlike the spec's "user,
                # default approval" phrasing implies): the strategy_type slot stays
                # optional/informational, and this step pins the literal default the
                # spec calls for so build_strategy's required input is always filled
                # even when the slot is omitted.
                "strategy_type": "approval",
                "rules": "$ref:设计分数带.output.recommended_rules",
                "score_col": "{slot:score_col}",
                "default_decision": "approve",
                "description": "Strategy development generated candidate",
            },
            depends_on_titles=("设计分数带",),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="回测策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "target_col": "{slot:target_col}",
                "profit_params": "{slot:profit_params}",
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
            title="对比基线",
            tool_ref=ToolRef("strategy", "compare_strategies"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                "profit_params": "{slot:profit_params}",
            },
            depends_on_titles=("构造策略", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "matrix_2x2"}),),
            decision_point=True,
        ),
        StepTemplate(
            title="采纳策略",
            tool_ref=ToolRef("strategy", "adopt_strategy"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "backtest_id": "$ref:回测策略.output.backtest_id",
                "adoption_reason": "{slot:adoption_reason}",
                "band_stats": "$ref:设计分数带.output",
            },
            depends_on_titles=("设计分数带", "构造策略", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
            # Mandatory adoption gate: auto-accept must not pass it through
            # (delivery-gate precedent), so the driver always pauses here.
            needs_confirmation=True,
        ),
        StepTemplate(
            title="策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "band_stats": "$ref:设计分数带.output",
            },
            depends_on_titles=("设计分数带", "构造策略", "采纳策略"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)
