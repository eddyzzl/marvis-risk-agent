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
    title="快速策略分析与回测",
    goal_patterns=(
        "快速策略分析",
        "快速策略回测",
        "quick strategy analysis",
        "quick strategy backtest",
    ),
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
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),  # FIN-3 #4: None when profit requested w/o pd_col (graceful EL degradation)
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


TYPED_STRATEGY_BUILD = WorkflowTemplate(
    id="typed_strategy_build",
    title="类型化策略开发",
    goal_patterns=("类型化策略开发", "typed strategy build"),
    slots=(
        SlotSpec("strategy_spec", True, "user", "Validated canonical Strategy DSL"),
    ),
    steps=(
        StepTemplate(
            title="构造类型化策略草案",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_spec": "{slot:strategy_spec}",
                "description": "Natural-language compiled typed strategy",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="生成类型化策略草案文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造类型化策略草案.output.strategy_id",
            },
            depends_on_titles=("构造类型化策略草案",),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


TYPED_STRATEGY_EVALUATION = WorkflowTemplate(
    # Natural-language requests compile into a canonical StrategySpec before this
    # workflow is instantiated. The shared build/backtest/doc chain therefore
    # evaluates all five strategy types without copying approval-only cutoff or
    # tradeoff steps into limit, pricing or segmentation flows.
    id="typed_strategy_evaluation",
    title="类型化策略评估",
    goal_patterns=(
        "类型化策略评估",
        "typed strategy evaluation",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("strategy_spec", True, "user", "Validated canonical Strategy DSL"),
        SlotSpec("baseline_strategy_id", False, "user", "Optional baseline strategy id"),
        SlotSpec("economics_inputs", False, "user", "Typed limit/pricing economics inputs"),
        SlotSpec("profit_params", False, "user", "Approval/reject profit parameters"),
        SlotSpec("ead_col", False, "user", "Approval/reject EAD column"),
        SlotSpec("pd_col", False, "user", "Approval/reject PD column"),
    ),
    steps=(
        StepTemplate(
            title="构造类型化策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_spec": "{slot:strategy_spec}",
                "description": "Natural-language compiled typed strategy",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="回测类型化策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造类型化策略.output.strategy_id",
                "target_col": "{slot:target_col}",
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                "economics_inputs": "{slot:economics_inputs}",
                "profit_params": "{slot:profit_params}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
            },
            depends_on_titles=("构造类型化策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "metrics"}),
                # The shared Tool retains flat approval aliases only for
                # approval/reject compatibility. Missing aliases on the other
                # three typed envelopes are valid, hence ``allow_null``.
                PostCheck(
                    "range",
                    {"field": "approval_rate", "min": 0.0, "max": 1.0, "allow_null": True},
                ),
                PostCheck(
                    "range",
                    {
                        "field": "approved_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "rejected_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="生成类型化策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造类型化策略.output.strategy_id",
            },
            depends_on_titles=("构造类型化策略", "回测类型化策略"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


TYPED_STRATEGY_APPLY = WorkflowTemplate(
    id="typed_strategy_apply",
    title="构造并应用类型化策略",
    goal_patterns=("构造并应用策略", "build and apply typed strategy"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Task-owned input dataset id"),
        SlotSpec("strategy_spec", True, "user", "Validated canonical Strategy DSL"),
    ),
    steps=(
        StepTemplate(
            title="构造待应用策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_spec": "{slot:strategy_spec}",
                "description": "Natural-language compiled strategy for application",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="应用类型化策略并生成逐行结果",
            tool_ref=ToolRef("strategy", "apply_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造待应用策略.output.strategy_id",
            },
            depends_on_titles=("构造待应用策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("range", {"field": "population_count", "min": 0}),
                PostCheck("nonempty", {"field": "evidence"}),
            ),
        ),
        StepTemplate(
            title="生成已应用策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造待应用策略.output.strategy_id",
            },
            depends_on_titles=("构造待应用策略", "应用类型化策略并生成逐行结果"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STORED_STRATEGY_EVALUATION = WorkflowTemplate(
    id="stored_strategy_evaluation",
    title="已有策略评估",
    goal_patterns=(
        "回测已有策略",
        "分析已有策略",
        "对比已有策略",
        "stored strategy evaluation",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("strategy_id", True, "user", "Task-owned strategy id"),
        SlotSpec("baseline_strategy_id", False, "user", "Optional same-type baseline id"),
        SlotSpec("economics_inputs", False, "user", "Typed limit/pricing economics inputs"),
        SlotSpec("profit_params", False, "user", "Approval/reject profit parameters"),
        SlotSpec("ead_col", False, "user", "Approval/reject EAD column"),
        SlotSpec("pd_col", False, "user", "Approval/reject PD column"),
    ),
    steps=(
        StepTemplate(
            title="回测已有策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "{slot:strategy_id}",
                "target_col": "{slot:target_col}",
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                "economics_inputs": "{slot:economics_inputs}",
                "profit_params": "{slot:profit_params}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "metrics"}),
                PostCheck(
                    "range",
                    {
                        "field": "approval_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "approved_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "rejected_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="生成已有策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={"strategy_id": "{slot:strategy_id}"},
            depends_on_titles=("回测已有策略",),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STORED_STRATEGY_REPORT = WorkflowTemplate(
    id="stored_strategy_report",
    title="已有策略报告",
    goal_patterns=("生成已有策略报告", "stored strategy report"),
    slots=(
        SlotSpec("strategy_id", True, "user", "Task-owned strategy id"),
    ),
    steps=(
        StepTemplate(
            title="生成已有策略报告",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={"strategy_id": "{slot:strategy_id}"},
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STORED_STRATEGY_APPLY = WorkflowTemplate(
    id="stored_strategy_apply",
    title="应用已有策略",
    goal_patterns=("应用已有策略", "执行已有策略", "apply stored strategy"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Task-owned input dataset id"),
        SlotSpec("strategy_id", True, "user", "Task-owned persisted strategy id"),
    ),
    steps=(
        StepTemplate(
            title="应用已有策略并生成逐行结果",
            tool_ref=ToolRef("strategy", "apply_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "{slot:strategy_id}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("range", {"field": "population_count", "min": 0}),
                PostCheck("nonempty", {"field": "output_columns"}),
                PostCheck("nonempty", {"field": "evidence"}),
            ),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STORED_STRATEGY_ADOPTION = WorkflowTemplate(
    id="stored_strategy_adoption",
    title="已有策略采纳",
    goal_patterns=("采纳已有策略", "adopt stored strategy"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("strategy_id", True, "user", "Task-owned draft strategy id"),
        SlotSpec("adoption_reason", True, "user", "Human supplied adoption reason"),
        SlotSpec("economics_inputs", False, "user", "Typed limit/pricing economics inputs"),
        SlotSpec("profit_params", False, "user", "Approval/reject profit parameters"),
        SlotSpec("ead_col", False, "user", "Approval/reject EAD column"),
        SlotSpec("pd_col", False, "user", "Approval/reject PD column"),
    ),
    steps=(
        StepTemplate(
            title="采纳前回测",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "{slot:strategy_id}",
                "target_col": "{slot:target_col}",
                "economics_inputs": "{slot:economics_inputs}",
                "profit_params": "{slot:profit_params}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "metrics"}),
                PostCheck(
                    "range",
                    {
                        "field": "approval_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "approved_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "rejected_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="采纳已有策略",
            tool_ref=ToolRef("strategy", "adopt_strategy"),
            inputs_template={
                "strategy_id": "{slot:strategy_id}",
                "backtest_id": "$ref:采纳前回测.output.backtest_id",
                "adoption_reason": "{slot:adoption_reason}",
            },
            depends_on_titles=("采纳前回测",),
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
            needs_confirmation=True,
        ),
        StepTemplate(
            title="生成采纳策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={"strategy_id": "{slot:strategy_id}"},
            depends_on_titles=("采纳已有策略",),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
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
        # A1: label_semantics has no default slot value on purpose -- an undeclared
        # basis makes tool_vintage_curve raise LabelSemanticsNotDeclaredError so the
        # user is forced to pick incremental vs snapshot; drop_nan_labels threads the
        # NaN-label confirmation through the same gate.
        SlotSpec("label_semantics", False, "user", "Bad-column cumulation basis: incremental or snapshot"),
        SlotSpec("drop_nan_labels", False, "user", "Confirm dropping NaN-label rows"),
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
                # Literal null default (not {slot:label_semantics}), mirroring
                # design_cutoff_bands' band_edges: apply_adjust's gate override only
                # reaches a key already present in the instantiated inputs, and an
                # omitted slot would be dropped by planner._fill_inputs. Baking null
                # here (a valid ["string","null"] per the manifest) is what lets the
                # gate write the user's incremental/snapshot choice onto this step;
                # an unanswered gate leaves null -> the tool raises the semantics gate.
                "label_semantics": None,
                # Baked False (a valid boolean per the manifest) for the same reason,
                # so a "drop the NaN rows" confirmation can be written onto the step.
                "drop_nan_labels": False,
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "cohorts"}),),
            decision_point=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

SLICE_AGGREGATE = WorkflowTemplate(
    # S6 ad-hoc 问数 entry: a single deterministic group-by aggregate over a
    # ready dataset. The LLM only produced a validated SliceSpec (INV-1); the
    # 口径确认门 is handled turn-side (turn_handlers) BEFORE this plan is built,
    # so — like vintage_analysis — the one step just runs to DONE and renders its
    # table (no needs_confirmation gate). Every slice_aggregate input is an
    # optional slot; turn_handlers fills exactly SliceSpec.tool_inputs(dataset_id),
    # and omitted slots drop out via _fill_inputs' _OMIT handling.
    id="slice_aggregate",
    title="即席问数",
    goal_patterns=("问数", "即席分析", "slice aggregate", "ad-hoc query"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Ready dataset id to aggregate"),
        SlotSpec("metrics", True, "task_context", "Validated aggregate metrics (op/col)"),
        SlotSpec("group_by", False, "task_context", "Optional group-by columns"),
        SlotSpec("filters", False, "task_context", "Optional filter conditions"),
        SlotSpec("month_col", False, "task_context", "Optional month column"),
        SlotSpec("months", False, "task_context", "Optional month range"),
        SlotSpec("sort_by", False, "task_context", "Optional sort column/metric label"),
    ),
    steps=(
        StepTemplate(
            title="计算聚合结果",
            tool_ref=ToolRef("data_ops", "slice_aggregate"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "metrics": "{slot:metrics}",
                "group_by": "{slot:group_by}",
                "filters": "{slot:filters}",
                "month_col": "{slot:month_col}",
                "months": "{slot:months}",
                "sort_by": "{slot:sort_by}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "columns"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_DEVELOPMENT = WorkflowTemplate(
    # S2 conversational strategy-development template. It is the standard entry;
    # strategy_analysis remains only as an explicit quick-analysis compatibility path.
    # Flow: tradeoff scan (direction
    # self-check) -> [confirm] design cutoff bands -> build strategy from the
    # recommended rules -> [confirm] backtest -> [optional] compare vs baseline
    # -> [mandatory confirm] adopt -> render doc. goal_patterns are disjoint from
    # strategy_analysis so keyword routing never crosses the two.
    id="strategy_development",
    title="策略开发",
    goal_patterns=(
        "策略开发",
        "开发策略",
        "策略分析",
        "策略回测",
        "策略权衡",
        "设计cutoff",
        "分数带策略",
        "strategy development",
        "strategy analysis",
        "strategy backtest",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("score_col", True, "task_context", "Score column"),
        SlotSpec("score_direction", False, "task_context", "Score direction if a model artifact injected one"),
        SlotSpec("objective", False, "user", "max_profit or max_approval"),
        SlotSpec("max_bad_rate", False, "user", "Max approved bad rate constraint"),
        SlotSpec("min_approval_rate", False, "user", "Min approval rate constraint"),
        SlotSpec("ead_col", False, "user", "Exposure-at-default column for profit evaluation"),
        SlotSpec("pd_col", False, "user", "Probability-of-default column for profit evaluation"),
        SlotSpec("profit_params", False, "user", "Profit parameters for expected-profit"),
        SlotSpec("strategy_type", True, "task_context", "Approval or reject strategy type"),
        SlotSpec("baseline_strategy_id", False, "user", "Baseline strategy id for the optional compare step"),
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
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
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
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
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
                "strategy_type": "{slot:strategy_type}",
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
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
                "profit_params": "{slot:profit_params}",
            },
            depends_on_titles=("构造策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("range", {"field": "approval_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "approved_bad_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "rejected_bad_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),  # FIN-3 #4: None when profit requested w/o pd_col (graceful EL degradation)
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
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
                "profit_params": "{slot:profit_params}",
            },
            depends_on_titles=("构造策略", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "matrix_2x2"}),),
            decision_point=True,
        ),
        StepTemplate(
            # S6: optional challenger report step, sits right after the compare step.
            # planner has no step-pruning, so degradation is at the TOOL level: with no
            # champion (baseline_strategy_id slot omitted) render_challenger_report
            # returns status='no_baseline' + a 「未提供基线」 markdown and writes NO
            # artifact -- exactly the compare_strategies no-op precedent, so the step
            # never fails the plan. The report numbers all come from the compare +
            # backtest outputs referenced here (report follows tool output).
            title="挑战者报告",
            tool_ref=ToolRef("strategy", "render_challenger_report"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "champion_strategy_id": "{slot:baseline_strategy_id}",
                "compare": "$ref:对比基线.output",
                "challenger_backtest": "$ref:回测策略.output",
            },
            depends_on_titles=("构造策略", "对比基线", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "status"}),),
        ),
        StepTemplate(
            title="采纳策略",
            tool_ref=ToolRef("strategy", "adopt_strategy"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "backtest_id": "$ref:回测策略.output.backtest_id",
                # The evidence-bound final gate writes the operator's reason into this
                # explicit override target.  Task setup must never pre-authorize adoption.
                "adoption_reason": "",
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


RULE_STRATEGY = WorkflowTemplate(
    # S4 conversational rule-mining strategy template (new id; disjoint
    # goal_patterns from strategy_analysis/strategy_development so keyword routing
    # never crosses them). Flow: mine candidate reject rules -> [confirm] pick an
    # ordered subset (rule-set selection gate) -> evaluate the chosen set
    # (waterfall/overlap) -> build a reject strategy from the selected rules ->
    # [confirm] backtest -> [mandatory confirm] adopt (S2 forced gate reused) ->
    # render doc. Adoption/doc/memory all reuse the S2 surface unchanged.
    id="rule_strategy",
    title="规则策略开发",
    goal_patterns=("规则挖掘", "拒绝规则", "规则策略", "rule mining", "rule strategy"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("feature_cols", False, "user", "Candidate feature columns (default: numeric columns)"),
        SlotSpec("score_col", False, "user", "Score column, if the rules should carry score-band rules"),
        SlotSpec("max_depth", False, "user", "Decision-tree depth for rule mining"),
        SlotSpec("min_support", False, "user", "Minimum rule support"),
        SlotSpec("min_lift", False, "user", "Minimum rule lift"),
        SlotSpec("top_k", False, "user", "Maximum candidate rules to return"),
    ),
    steps=(
        StepTemplate(
            title="挖掘规则",
            tool_ref=ToolRef("strategy", "mine_rules"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "feature_cols": "{slot:feature_cols}",
                "max_depth": "{slot:max_depth}",
                "min_support": "{slot:min_support}",
                "min_lift": "{slot:min_lift}",
                "top_k": "{slot:top_k}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "candidate_rules"}),),
        ),
        StepTemplate(
            title="规则集确认",
            tool_ref=ToolRef("strategy", "select_rule_set"),
            inputs_template={
                "candidate_rules": "$ref:挖掘规则.output.candidate_rules",
                # Literal None default (not {slot:selection}): SlotSpec has no
                # default-value mechanism and an omitted slot key gets dropped by
                # planner._fill_inputs, so the apply_adjust gate-override channel
                # (which only overwrites keys already present in the step's
                # instantiated inputs) needs the key baked in. The rule-set gate
                # reply parser turns 「选 1,3,5」/「全选」/「去掉 2」 into a selection
                # list that apply_adjust writes here -- exactly the band_edges
                # precedent (templates/strategy.py STRATEGY_DEVELOPMENT 设计分数带).
                # None == keep all candidates.
                "selection": None,
            },
            depends_on_titles=("挖掘规则",),
            post_checks=(PostCheck("nonempty", {"field": "selected_rules"}),),
            # Rule-set selection gate: pause so the user can pick/reorder/drop
            # rules before they are evaluated and built into a strategy.
            needs_confirmation=True,
        ),
        StepTemplate(
            title="评估规则集",
            tool_ref=ToolRef("strategy", "evaluate_rule_set"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "rules": "$ref:规则集确认.output.selected_rules",
            },
            depends_on_titles=("规则集确认",),
            post_checks=(PostCheck("nonempty", {"field": "waterfall"}),),
            decision_point=True,
        ),
        StepTemplate(
            title="构造策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                # Pin the literal reject-strategy defaults (SlotSpec has no
                # default-value mechanism): a rule strategy is an approval-type
                # strategy whose selected rules reject, defaulting to approve.
                "strategy_type": "approval",
                "rules": "$ref:规则集确认.output.selected_rules",
                # score_col flows only when the optional slot is filled; when a
                # score column is present build_strategy's rule-direction
                # self-check (S1a) fires automatically on any score-band rules.
                "score_col": "{slot:score_col}",
                "default_decision": "approve",
                "description": "Rule strategy generated candidate",
            },
            depends_on_titles=("规则集确认",),
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
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),  # FIN-3 #4: None when profit requested w/o pd_col (graceful EL degradation)
            ),
            decision_point=True,
            needs_confirmation=True,
        ),
        StepTemplate(
            title="采纳策略",
            tool_ref=ToolRef("strategy", "adopt_strategy"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "backtest_id": "$ref:回测策略.output.backtest_id",
                "adoption_reason": "",
            },
            depends_on_titles=("构造策略", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
            # Mandatory adoption gate (S2 forced-gate precedent): auto-accept must
            # not pass it through, so the driver always pauses here.
            needs_confirmation=True,
        ),
        StepTemplate(
            title="策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
            },
            depends_on_titles=("构造策略", "采纳策略"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)
