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
    assert "sample_echo" in builtin_template_ids()
    model_validation = get_template("model_validation")
    assert model_validation.steps[0].tool_ref == ToolRef("v1_compat", "scan_materials")
    assert model_validation.steps[-1].needs_confirmation is True
    assert "model_validation" in builtin_template_ids()
    standard_modeling = get_template("standard_modeling")
    assert standard_modeling.steps[-1].tool_ref == ToolRef("modeling", "generate_model_report")
    assert standard_modeling.steps[-1].needs_confirmation is True
    assert not any(step.decision_point for step in standard_modeling.steps)
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
        ToolRef("modeling", "generate_model_report"),
    ]
    train_step = plan.steps[4]
    compare_step = plan.steps[5]
    report_step = plan.steps[6]
    assert compare_step.inputs == {"experiment_ids": [f"$ref:{train_step.id}.output.experiment_id"]}
    assert report_step.inputs["experiment_id"] == f"$ref:{train_step.id}.output.experiment_id"
    assert report_step.inputs["dataset_id"] == "dataset-1"
    assert report_step.inputs["project_meta"] == {}
    assert report_step.needs_confirmation is True


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
    ]
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
