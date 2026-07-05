from pathlib import Path

import pytest

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
    builtin_template_ids,
    clear_user_templates,
    get_template,
    list_templates,
    load_builtin_templates,
    register_template,
    register_user_template,
)
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _template(template_id: str, *, source: str = "builtin") -> WorkflowTemplate:
    return WorkflowTemplate(
        id=template_id,
        title=f"Template {template_id}",
        goal_patterns=(template_id,),
        slots=(SlotSpec("task_id", True, "task_context", "Current task"),),
        steps=(
            StepTemplate(
                title="Echo",
                tool_ref=ToolRef("_sample", "echo"),
                inputs_template={"message": "{slot:task_id}"},
                depends_on_titles=(),
                post_checks=(PostCheck("nonempty", {"field": "echoed"}),),
            ),
        ),
        source=source,
    )


def test_register_get_and_list_templates():
    template = _template("test_builtin_template")

    register_template(template)

    assert get_template("test_builtin_template") == template
    assert template in list_templates()
    with pytest.raises(ValueError, match="duplicate"):
        register_template(template)


def test_load_builtin_templates_registers_sample_echo_idempotently():
    load_builtin_templates()
    load_builtin_templates()

    template = get_template("sample_echo")
    assert template.source == "builtin"
    assert template.steps[0].tool_ref == ToolRef("_sample", "echo")
    assert template.slots[0].name == "message"
    assert template.success_criteria == ()
    assert "sample_echo" in builtin_template_ids()
    model_validation = get_template("model_validation")
    assert model_validation.steps[0].tool_ref == ToolRef("v1_compat", "scan_materials")
    assert model_validation.steps[-1].needs_confirmation is True
    assert model_validation.success_criteria == ()
    assert "model_validation" in builtin_template_ids()
    standard_modeling = get_template("standard_modeling")
    assert standard_modeling.steps[-2].tool_ref == ToolRef("modeling", "generate_model_report")
    assert standard_modeling.steps[-1].tool_ref == ToolRef("modeling", "post_training_action")
    assert standard_modeling.steps[-1].needs_confirmation is True
    for template_id in ("standard_modeling", "modeling", "modeling_with_join"):
        assert "champion_reference" in {slot.name for slot in get_template(template_id).slots}
    assert not any(step.decision_point for step in standard_modeling.steps)
    assert standard_modeling.success_criteria == ()
    assert "standard_modeling" in builtin_template_ids()


def test_standard_modeling_template_instantiates_valid_report_plan(tmp_path):
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("standard_modeling"),
        {
            "dataset_id": "dataset-1",
            "target_col": "bad_flag",
            "feature_cols": ["income", "age"],
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "recipe": "lr",
            "seed": 7,
            "business_columns": {"loan_month_col": "loan_month", "interest_rate_col": "rate"},
            "feature_dictionary_id": "dict-1",
            "project_meta": {"项目名称": "A卡模型"},
            "champion_reference": {"experiment_id": "exp-current-champion"},
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("modeling", "check_data_quality"),
        ToolRef("modeling", "modeling_readiness"),
        ToolRef("modeling", "prepare_modeling_frame"),
        ToolRef("modeling", "select_features"),
        ToolRef("modeling", "train_model"),
        ToolRef("modeling", "compare_experiments"),
        ToolRef("modeling", "select_experiment"),
        ToolRef("modeling", "generate_model_report"),
        ToolRef("modeling", "post_training_action"),
    ]
    train_step = plan.steps[4]
    compare_step = plan.steps[5]
    select_step = plan.steps[6]
    report_step = plan.steps[7]
    delivery_step = plan.steps[8]
    assert compare_step.inputs == {"experiment_ids": [f"$ref:{train_step.id}.output.experiment_id"]}
    assert select_step.inputs["experiment_ids"] == [f"$ref:{train_step.id}.output.experiment_id"]
    assert select_step.inputs["selection_policy"] == {"require_pmml": True, "require_handoff": True}
    assert report_step.inputs["experiment_id"] == f"$ref:{select_step.id}.output.selected_experiment_id"
    assert report_step.inputs["dataset_id"] == "dataset-1"
    assert report_step.inputs["business_columns"] == {"loan_month_col": "loan_month", "interest_rate_col": "rate"}
    assert report_step.inputs["feature_dictionary_id"] == "dict-1"
    assert report_step.inputs["project_meta"] == {"项目名称": "A卡模型"}
    assert report_step.needs_confirmation is True
    assert delivery_step.inputs["experiment_id"] == f"$ref:{select_step.id}.output.selected_experiment_id"
    assert delivery_step.inputs["sample_dataset_id"] == "dataset-1"
    assert delivery_step.inputs["actions"] == [
        "export_pmml",
        "handoff_to_validation",
        "create_challenger_backtest",
    ]
    assert delivery_step.inputs["selection_policy_decision"] == f"$ref:{select_step.id}.output.policy_decision"
    assert delivery_step.inputs["champion_reference"] == {"experiment_id": "exp-current-champion"}
    assert delivery_step.needs_confirmation is True
    assert plan.success_criteria == []


def test_modeling_template_phases_gates_and_refs(tmp_path):
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("modeling"),
        {
            "dataset_id": "dataset-1",
            "target_col": "long_y",
            "feature_cols": ["sig1", "sig2", "sig3"],
            "split_col": "model_flag",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "recipe": "lgb",
            "recipes": ["lgb"],
            "seed": 23,
            "holdout_values": ["oot"],
            "business_columns": {"loan_month_col": "loan_month"},
            "feature_dictionary_id": "dict-1",
            "project_meta": {"项目名称": "通用A卡"},
            "selection_policy": {"require_pmml": True, "require_handoff": True},
        },
        task_id="task-1",
    )

    # valid against the real modeling pack tool catalog
    assert PlanValidator(tool_registry).validate(plan) == []
    # step order: G1 make_split -> G2 spec -> screen -> refine (FS-1) -> configure -> tune
    # -> train -> compare -> select -> report -> delivery
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("modeling", "make_split"),
        ToolRef("modeling", "choose_modeling_spec"),
        ToolRef("modeling", "screen_features"),
        ToolRef("modeling", "select_features"),
        ToolRef("modeling", "configure_tuning"),
        ToolRef("modeling", "tune_hyperparameters"),
        ToolRef("modeling", "train_models"),
        ToolRef("modeling", "compare_experiments"),
        ToolRef("modeling", "select_experiment"),
        ToolRef("modeling", "generate_model_report"),
        ToolRef("modeling", "post_training_action"),
    ]
    # phase tags for right-rail big-step grouping
    assert [step.phase for step in plan.steps] == [
        "特征", "建模", "特征", "特征", "建模", "建模", "建模", "建模", "建模", "报告", "交付"
    ]
    # gates: confirm split/features/refined-features/tuning config, select final
    # experiment, approve report and delivery.
    assert [step.needs_confirmation for step in plan.steps] == [
        False, False, True, True, True, True, False, False, True, True, True
    ]
    assert not any(step.decision_point for step in plan.steps)

    make_split, spec, screen, refine, tuning_config, tune, train, compare, select, report, delivery = plan.steps
    # screen/tune/train run on the split frame produced by the G1 gate
    split_ref = f"$ref:{make_split.id}.output.result_dataset_id"
    assert spec.inputs["features"] == f"$ref:{make_split.id}.output.feature_cols"
    assert spec.inputs["recipes"] == ["lgb"]
    assert spec.inputs["n_trials"] == 40
    assert screen.inputs["dataset_id"] == split_ref
    assert screen.inputs["features"] == f"$ref:{spec.id}.output.feature_cols"
    assert screen.inputs["target_type"] == f"$ref:{spec.id}.output.target_type"
    assert screen.inputs["leakage_ks"] == 0.4
    assert screen.inputs["max_missing_rate"] == 0.95
    assert screen.inputs["top_k"] == 200
    # FS-1: multivariate refinement funnel sits between screen and tuning config —
    # IV floor + correlation dedup on the screen's clean candidate set.
    assert refine.inputs["dataset_id"] == split_ref
    assert refine.inputs["features"] == f"$ref:{screen.id}.output.selected"
    assert refine.inputs["target_type"] == f"$ref:{spec.id}.output.target_type"
    assert refine.inputs["space"] == "raw"
    assert refine.inputs["iv_min"] == 0.02
    assert refine.inputs["corr_max"] == 0.95
    assert refine.inputs["vif_max"] == 1e9  # VIF off by default (tree recipes don't need it)
    assert refine.needs_confirmation is True
    assert tune.inputs["dataset_id"] == split_ref
    assert train.inputs["dataset_id"] == split_ref
    # tune + train consume the REFINED feature set (not the raw screen output); train
    # consumes tuned params
    assert tuning_config.inputs["recipe"] == f"$ref:{spec.id}.output.recipe"
    assert tuning_config.inputs["recipes"] == f"$ref:{spec.id}.output.recipes"
    assert tuning_config.inputs["n_trials_by_recipe"] == f"$ref:{spec.id}.output.n_trials_by_recipe"
    assert tune.inputs["features"] == f"$ref:{refine.id}.output.selected"
    assert tune.inputs["recipe"] == f"$ref:{tuning_config.id}.output.recipe"
    assert tune.inputs["recipes"] == f"$ref:{tuning_config.id}.output.recipes"
    assert tune.inputs["n_trials_by_recipe"] == f"$ref:{tuning_config.id}.output.n_trials_by_recipe"
    assert tune.inputs["params"] == f"$ref:{tuning_config.id}.output.params"
    assert train.inputs["features"] == f"$ref:{refine.id}.output.selected"
    assert train.inputs["params"] == f"$ref:{tune.id}.output.best_params"
    assert train.inputs["recipes"] == f"$ref:{spec.id}.output.recipes"
    assert train.inputs["target_type"] == f"$ref:{spec.id}.output.target_type"
    assert compare.inputs == {"experiment_ids": f"$ref:{train.id}.output.experiment_ids"}
    assert select.inputs["experiment_ids"] == f"$ref:{train.id}.output.experiment_ids"
    assert select.inputs["target_type"] == f"$ref:{spec.id}.output.target_type"
    assert select.inputs["selection_policy"] == {"require_pmml": True, "require_handoff": True}
    assert report.inputs["experiment_id"] == f"$ref:{select.id}.output.selected_experiment_id"
    assert report.inputs["dataset_id"] == "dataset-1"
    assert delivery.inputs["experiment_id"] == f"$ref:{select.id}.output.selected_experiment_id"
    assert delivery.inputs["sample_dataset_id"] == "dataset-1"
    assert delivery.inputs["selection_policy_decision"] == f"$ref:{select.id}.output.policy_decision"
    assert plan.success_criteria == []
    assert "modeling" in builtin_template_ids()


def test_modeling_template_validates_with_optional_slots_omitted(tmp_path):
    """Driver may not always have holdout_values / report business metadata; the
    optional slots must drop cleanly without breaking tool input-schema validation."""
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("modeling"),
        {
            "dataset_id": "dataset-1",
            "target_col": "long_y",
            "feature_cols": ["sig1", "sig2"],
            "split_col": "model_flag",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "recipe": "lgb",
            "recipes": ["lgb"],
            "seed": 23,
            "selection_policy": {"require_pmml": True, "require_handoff": True},
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    spec = plan.steps[1]
    screen = plan.steps[2]
    refine = plan.steps[3]
    assert "holdout_values" not in screen.inputs  # omitted optional dropped, not None
    assert "holdout_values" not in refine.inputs
    assert "sample_weight_candidates" not in spec.inputs
    assert "params" not in spec.inputs
    tuning_config = plan.steps[4]
    assert tuning_config.inputs["sample_weight_col"] == f"$ref:{spec.id}.output.sample_weight_col"
    assert tuning_config.inputs["params"] == f"$ref:{spec.id}.output.params"
    report = plan.steps[-2]
    assert "business_columns" not in report.inputs


def test_modeling_template_does_not_shadow_standard_modeling_goal_routing(tmp_path):
    """The new template must keep narrow goal patterns so common modeling goals
    still route to the legacy standard_modeling template (pinned by intent tests)."""
    load_builtin_templates()
    modeling = get_template("modeling")
    standard = get_template("standard_modeling")
    assert set(modeling.goal_patterns).isdisjoint(set(standard.goal_patterns))


def test_modeling_template_select_step_does_not_inherit_screen_holdout(tmp_path):
    """D11/FS-2: when an OOT split exists (holdout_values slot = ['oot']), the screen
    step must still hold out only OOT (train+test pooled as dev), but the 精选特征
    (select_features) step must NOT inherit that ['oot'] holdout — otherwise IV/corr/
    VIF/top_k fit on train+test and leak the test-split labels. Select must fall back
    to its own safe ('test','oot') default, so the resolved step carries no
    holdout_values=['oot'] key."""
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("modeling"),
        {
            "dataset_id": "dataset-1",
            "target_col": "long_y",
            "feature_cols": ["sig1", "sig2", "sig3"],
            "split_col": "model_flag",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "recipe": "lgb",
            "recipes": ["lgb"],
            "seed": 23,
            "holdout_values": ["oot"],
            "selection_policy": {"require_pmml": True, "require_handoff": True},
        },
        task_id="task-1",
    )

    screen = plan.steps[2]
    refine = plan.steps[3]
    # screen still holds out OOT only (pools train+test as dev) — untouched
    assert screen.inputs["holdout_values"] == ["oot"]
    # select must not receive the screen's ['oot'] holdout (would leak test labels)
    assert refine.inputs.get("holdout_values") != ["oot"]
    assert "holdout_values" not in refine.inputs


def test_modeling_templates_select_step_never_binds_holdout_values(tmp_path):
    """D11 re-introduction guard: neither the single-table `modeling` nor the
    multi-table `modeling_with_join` 精选特征 (select_features) step template may bind
    a `holdout_values` input — that binding is the screen's holdout and forwarding it
    to select re-opens the FS-2 test-label leak. select_features applies its own safe
    ('test','oot') default when the key is absent."""
    load_builtin_templates()
    for template_id in ("modeling", "modeling_with_join"):
        template = get_template(template_id)
        select_steps = [
            step
            for step in template.steps
            if step.tool_ref == ToolRef("modeling", "select_features")
        ]
        assert select_steps, f"{template_id} has no select_features step"
        for step in select_steps:
            assert "holdout_values" not in step.inputs_template, (
                f"{template_id} {step.title} must not bind holdout_values into select"
            )
        # regression guard: the screen step MUST still bind holdout_values
        screen_steps = [
            step
            for step in template.steps
            if step.tool_ref == ToolRef("modeling", "screen_features")
        ]
        assert screen_steps, f"{template_id} has no screen_features step"
        for step in screen_steps:
            assert "holdout_values" in step.inputs_template, (
                f"{template_id} {step.title} must keep its holdout_values binding"
            )


def test_data_join_template_phases_gate_and_refs(tmp_path):
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("data_join"),
        {
            "anchor_id": "ds-anchor",
            "feature_ids": ["ds-f1", "ds-f2"],
            "dedup_strategies": {"ds-f1": "first"},
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("data_ops", "propose_join"),
        ToolRef("data_ops", "confirm_join"),
        ToolRef("data_ops", "execute_join"),
    ]
    # single phase; the forced-confirm human gate sits on execute_join (INV-3)
    assert [step.phase for step in plan.steps] == ["数据准备", "数据准备", "数据准备"]
    assert [step.needs_confirmation for step in plan.steps] == [False, False, True]
    # propose_join is a decision point (spec §2/§10): agent mode may adapt from diagnostics.
    # The execute_join INV-3 gate + engine backstop keep the 1:1 invariant regardless.
    assert [step.decision_point for step in plan.steps] == [True, False, False]

    propose, confirm, execute = plan.steps
    # confirm + execute both operate on the join plan id produced by propose
    assert confirm.inputs["join_plan_id"] == f"$ref:{propose.id}.output.join_plan_id"
    assert execute.inputs["join_plan_id"] == f"$ref:{propose.id}.output.join_plan_id"
    assert confirm.inputs["dedup_strategies"] == {"ds-f1": "first"}
    # execute_join must directly depend on propose (it refs its output) and on confirm (ordering)
    assert set(execute.depends_on) == {propose.id, confirm.id}
    assert {check.kind for check in execute.post_checks} == {"nonempty", "rowcount", "invariant"}
    assert any(
        check.kind == "invariant" and check.spec["rule"] == "joined_rows<=anchor_rows"
        for check in execute.post_checks
    )
    assert "data_join" in builtin_template_ids()


def test_from_template_step_ids_globally_unique_across_plans(tmp_path):
    """Regression: instantiating the same template twice must yield disjoint step
    ids (plan_steps.id is a primary key) and both must persist to one repo.
    Previously every plan reused step-1/step-2/... so the second insert hit a
    UNIQUE constraint failure — only ever exercised in fresh-workspace tests."""
    from marvis.db import PlanRepository, init_db as _init_db

    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))
    slots = {"anchor_id": "a", "feature_ids": ["f1"]}

    plan1 = planner.from_template(get_template("data_join"), slots, task_id="t1")
    plan2 = planner.from_template(get_template("data_join"), slots, task_id="t2")

    ids1 = {step.id for step in plan1.steps}
    ids2 = {step.id for step in plan2.steps}
    assert ids1.isdisjoint(ids2)
    assert all(step.id.startswith(plan1.id) for step in plan1.steps)

    # both plans persist to the same repo without a primary-key collision
    db_path = tmp_path / "plans.sqlite"
    _init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(plan1)
    repo.create_plan(plan2)
    assert {p.id for p in repo.list_plans_for_task("t1")} == {plan1.id}
    assert {p.id for p in repo.list_plans_for_task("t2")} == {plan2.id}


def test_feature_derivation_template_marks_adaptive_decision_point(tmp_path):
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("feature_derivation"),
        {
            "dataset_id": "dataset-1",
            "target_col": "bad_flag",
            "feature_cols": ["income", "age"],
            "derivation_recipe": [{"kind": "ratio", "num": "income", "den": "age"}],
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("feature", "compute_feature_metrics"),
        ToolRef("feature", "cross_features"),
        ToolRef("feature", "compute_feature_metrics"),
        ToolRef("feature", "screen_features"),  # FEAT-3: derivation now ends in a screening step
    ]
    assert plan.steps[-1].title == "特征筛选"
    assert [step.title for step in plan.steps if step.decision_point] == ["衍生特征"]
    assert not get_template("model_validation").steps[-1].decision_point
    assert not any(step.decision_point for step in get_template("standard_modeling").steps)


def test_strategy_analysis_template_marks_backtest_decision_point(tmp_path):
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("strategy_analysis"),
        {
            "dataset_id": "dataset-1",
            "target_col": "bad_flag",
            "score_col": "score",
            "strategy_type": "approval",
            "rules": [{"condition": "score < 600", "decision": "reject"}],
            "default_decision": "approve",
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("strategy", "build_strategy"),
        ToolRef("strategy", "backtest_strategy"),
        ToolRef("strategy", "tradeoff_view"),
    ]
    assert [step.title for step in plan.steps if step.decision_point] == ["回测策略"]
    assert [step.title for step in plan.steps if step.needs_confirmation] == ["回测策略"]


def test_strategy_development_template_instantiates_and_validates(tmp_path):
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("strategy_development"),
        {
            "dataset_id": "dataset-1",
            "target_col": "bad_flag",
            "score_col": "score",
            "adoption_reason": "committee approved",
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("strategy", "tradeoff_view"),
        ToolRef("strategy", "design_cutoff_bands"),
        ToolRef("strategy", "build_strategy"),
        ToolRef("strategy", "backtest_strategy"),
        ToolRef("strategy", "compare_strategies"),
        ToolRef("strategy", "render_challenger_report"),
        ToolRef("strategy", "adopt_strategy"),
        ToolRef("strategy", "render_strategy_doc"),
    ]
    bands_step = plan.steps[1]
    build_step = plan.steps[2]
    backtest_step = plan.steps[3]
    compare_step = plan.steps[4]
    report_step = plan.steps[5]
    adopt_step = plan.steps[6]
    doc_step = plan.steps[7]
    assert bands_step.needs_confirmation is True
    assert backtest_step.needs_confirmation is True
    assert backtest_step.decision_point is True
    # Mandatory adoption gate: auto-accept must not skip it (delivery-gate precedent).
    assert adopt_step.needs_confirmation is True
    assert build_step.inputs["rules"] == f"$ref:{bands_step.id}.output.recommended_rules"
    assert adopt_step.inputs["backtest_id"] == f"$ref:{backtest_step.id}.output.backtest_id"
    assert adopt_step.inputs["band_stats"] == f"$ref:{bands_step.id}.output"
    assert doc_step.inputs["strategy_id"] == f"$ref:{build_step.id}.output.strategy_id"
    # S6: the optional challenger report step sits after compare, before adopt. Its
    # numbers come from the compare + backtest outputs (report follows tool output); with
    # baseline_strategy_id omitted the champion slot drops and the tool degrades to a
    # no-baseline no-op (工具级优雅降级) rather than the plan failing.
    assert report_step.inputs["compare"] == f"$ref:{compare_step.id}.output"
    assert report_step.inputs["challenger_backtest"] == f"$ref:{backtest_step.id}.output"
    assert report_step.inputs["strategy_id"] == f"$ref:{build_step.id}.output.strategy_id"
    assert "champion_strategy_id" not in report_step.inputs  # omitted baseline slot dropped


def test_strategy_development_goal_patterns_do_not_cross_strategy_analysis(tmp_path):
    load_builtin_templates()
    development = get_template("strategy_development")
    analysis = get_template("strategy_analysis")
    assert set(development.goal_patterns).isdisjoint(set(analysis.goal_patterns))


def test_rule_strategy_template_instantiates_and_validates(tmp_path):
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("rule_strategy"),
        {
            "dataset_id": "dataset-1",
            "target_col": "bad_flag",
            "adoption_reason": "committee approved",
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("strategy", "mine_rules"),
        ToolRef("strategy", "select_rule_set"),
        ToolRef("strategy", "evaluate_rule_set"),
        ToolRef("strategy", "build_strategy"),
        ToolRef("strategy", "backtest_strategy"),
        ToolRef("strategy", "adopt_strategy"),
        ToolRef("strategy", "render_strategy_doc"),
    ]
    mine, select, evaluate, build, backtest, adopt, doc = plan.steps
    # rule-set selection gate + backtest + mandatory adopt are the confirm gates.
    assert [step.needs_confirmation for step in plan.steps] == [
        False, True, False, False, True, True, False
    ]
    assert [step.title for step in plan.steps if step.decision_point] == ["评估规则集", "回测策略"]
    # $ref wiring: select consumes the mined candidates; evaluate + build consume
    # the SELECTED subset (same $ref, so the two never diverge from the gate).
    assert select.inputs["candidate_rules"] == f"$ref:{mine.id}.output.candidate_rules"
    # selection default is a literal None (the apply_adjust override slot).
    assert select.inputs["selection"] is None
    assert evaluate.inputs["rules"] == f"$ref:{select.id}.output.selected_rules"
    assert build.inputs["rules"] == f"$ref:{select.id}.output.selected_rules"
    assert adopt.inputs["backtest_id"] == f"$ref:{backtest.id}.output.backtest_id"
    assert doc.inputs["strategy_id"] == f"$ref:{build.id}.output.strategy_id"
    # optional score_col slot omitted -> dropped, not None (build_strategy skips
    # the direction self-check for arbitrary-feature rules).
    assert "score_col" not in build.inputs


def test_rule_strategy_goal_patterns_disjoint_from_other_strategy_templates(tmp_path):
    load_builtin_templates()
    rule = get_template("rule_strategy")
    analysis = get_template("strategy_analysis")
    development = get_template("strategy_development")
    assert set(rule.goal_patterns).isdisjoint(set(analysis.goal_patterns))
    assert set(rule.goal_patterns).isdisjoint(set(development.goal_patterns))
    assert "rule_strategy" in builtin_template_ids()


def test_vintage_analysis_template_runs_vintage_curve(tmp_path):
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("vintage_analysis"),
        {
            "dataset_id": "dataset-1",
            "cohort_col": "cohort",
            "mob_col": "mob",
            "bad_col": "bad",
            "mob_max": 12,
            "ref_mob": 6,
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [ToolRef("strategy", "vintage_curve")]
    assert [step.title for step in plan.steps if step.decision_point] == ["计算 Vintage 曲线"]


def test_vintage_template_threads_label_semantics_and_drop_nan_labels(tmp_path):
    # A1: the vintage step must carry label_semantics (baked literal-null so the
    # gate override reaches it) and drop_nan_labels so the confirmation choices
    # thread through to tool_vintage_curve.
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("vintage_analysis"),
        {"dataset_id": "dataset-1", "cohort_col": "cohort", "mob_col": "mob", "bad_col": "bad"},
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    step = next(step for step in plan.steps if step.tool_ref == ToolRef("strategy", "vintage_curve"))
    # label_semantics is baked as a literal null default (mirrors band_edges) so
    # the apply_adjust gate override can write the user's choice onto the step.
    assert "label_semantics" in step.inputs
    assert step.inputs["label_semantics"] is None
    assert "drop_nan_labels" in step.inputs
    assert step.inputs["drop_nan_labels"] is False


def test_monitoring_run_template_chains_score_then_monitor_with_alert_gate(tmp_path):
    """(f) S1b MONITORING_RUN: score_dataset -> monitor_run, with monitor_run as
    the sole confirmation/decision_point gate (the alert-confirmation gate whose
    copy is rendered from tool_monitor_run's own red/amber/green output)."""
    load_builtin_templates()
    tool_registry = _tool_registry(tmp_path)
    planner = Planner(tool_registry, lambda: None, PlanValidator(tool_registry))

    plan = planner.from_template(
        get_template("monitoring_run"),
        {
            "experiment_id": "experiment-1",
            "dataset_id": "dataset-1",
        },
        task_id="task-1",
    )

    assert PlanValidator(tool_registry).validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("modeling", "score_dataset"),
        ToolRef("modeling", "monitor_run"),
    ]
    assert [step.title for step in plan.steps if step.decision_point] == ["监控运行"]
    assert [step.title for step in plan.steps if step.needs_confirmation] == ["监控运行"]
    score_step = next(step for step in plan.steps if step.title == "打分")
    monitor_step = next(step for step in plan.steps if step.title == "监控运行")
    assert monitor_step.inputs["scored_dataset_id"] == f"$ref:{score_step.id}.output.result_dataset_id"
    assert monitor_step.inputs["score_col"] == f"$ref:{score_step.id}.output.score_col"
    assert score_step.id in monitor_step.depends_on
    assert "monitoring_run" in builtin_template_ids()


def test_user_template_registration_cannot_shadow_builtin_and_can_reload():
    load_builtin_templates()
    clear_user_templates()
    user_v1 = _template("user_echo", source="user")
    user_v2 = _template("user_echo", source="user")

    register_user_template(user_v1)
    register_user_template(user_v2)

    assert get_template("user_echo") == user_v2
    with pytest.raises(ValueError, match="builtin"):
        register_user_template(_template("sample_echo", source="user"))

    clear_user_templates()
    with pytest.raises(KeyError):
        get_template("user_echo")
    assert get_template("sample_echo").source == "builtin"


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    load_builtin_packs(registry, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(registry)
