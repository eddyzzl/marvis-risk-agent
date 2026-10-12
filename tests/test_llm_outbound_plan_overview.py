from copy import deepcopy
from pathlib import Path
import json

import pytest

from marvis.agent.auto_drive import decide_gate
from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.db import init_db
from marvis.domain import TaskCreate
from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_plan_overview import (
    ApprovedPlanOverview,
    approved_plan_overview,
)
from marvis.orchestrator.contracts import Plan, PlanStep, PlanStatus
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.repositories.plans import PlanRepository
from marvis.repositories.plugins import PluginRepository
from marvis.repositories.tasks import TaskRepository
from tests.test_llm_outbound import Wire


def _fixture(tmp_path):
    db = tmp_path / "marvis.sqlite"
    init_db(db)
    plugins = PluginRegistry(PluginRepository(db))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis/packs")
    tools = ToolRegistry(plugins)
    repo = PlanRepository(db)
    tasks = TaskRepository(db)
    task = tasks.create_task(
        TaskCreate(
            model_name="public model",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
        )
    )
    tasks.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content="请建立二分类模型，只使用逻辑回归，调参1轮。",
    )
    first_ref = ToolRef("modeling", "choose_modeling_spec")
    first = PlanStep(
        id="setup",
        plan_id="plan-owned",
        index=0,
        title="PRIVATE_REPORT_TITLE",
        tool_ref=first_ref,
        inputs={
            "target_col": "y",
            "target_type": "binary",
            "recipe": "lr",
            "features": ["signal", "noise"],
            "recipes": ["lr"],
            "n_trials": 1,
            "seed": 7,
        },
        depends_on=[],
        post_checks=[],
        needs_confirmation=True,
        policy=tools.resolve(first_ref).policy,
    )
    second = PlanStep(
        id="finish",
        plan_id="plan-owned",
        index=1,
        title="PRIVATE_REPORT_TITLE",
        tool_ref=ToolRef("_sample", "echo"),
        inputs={"message": "PRIVATE_RAW_CODE_OR_REPORT"},
        depends_on=["setup"],
        post_checks=[],
        needs_confirmation=True,
    )
    plan = Plan(
        id="plan-owned",
        task_id=task.id,
        goal="PRIVATE_REPORT_GOAL",
        source="template",
        template_id=None,
        steps=[first, second],
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
    )
    assert not PlanValidator(tools).validate_problems(plan)
    repo.create_plan(plan)
    plan = repo.load_plan(plan.id)
    gate = PlanMessageComposer(load_output=repo.load_step_output).plan_overview_message(
        plan
    )
    return (
        repo,
        PlanValidator(tools),
        task,
        {"content": gate.content, "metadata": gate.metadata},
    )


def test_actual_validated_plan_graph_controls_and_origin_reach_protocol(
    monkeypatch, tmp_path
):
    repo, validator, task, gate = _fixture(tmp_path)
    context = approved_plan_overview(
        repo, validator, task_id=task.id, gate_metadata=gate["metadata"]
    )
    wire = Wire(monkeypatch)
    wire.reply = {
        "action": "confirm",
        "reason": "启动平台已验证计划，后续按节点治理",
        "confidence": 0.9,
    }
    assert (
        decide_gate(wire.client, gate=gate, plan_overview=context)["action"]
        == "confirm"
    )
    sent = wire.last_context()["plan_overview"]
    assert sent["purpose"] == "review_validated_plan_before_execution"
    assert sent["status"] == "validated" and sent["step_count"] == 2
    assert [row["id"] for row in sent["steps"]] == ["setup", "finish"]
    assert sent["steps"][1]["depends_on"] == ["setup"]
    assert sent["steps"][0]["inputs"] == {
        "target_col": "y",
        "target_type": "binary",
        "recipe": "lr",
        "features": ["signal", "noise"],
        "recipes": ["lr"],
        "n_trials": 1,
        "seed": 7,
    }
    assert sent["steps"][0]["tool"]["tool"] == "choose_modeling_spec"
    assert all(row["needs_confirmation"] for row in sent["steps"])
    assert all(len(row["complete_inputs_sha256"]) == 64 for row in sent["steps"])
    assert "active_user_instruction" not in sent
    assert "PRIVATE_" not in json.dumps(wire.requests)


def test_native_modeling_plan_keeps_typed_upstream_field_references(monkeypatch, tmp_path):
    from marvis.orchestrator.planner import Planner
    from marvis.orchestrator.templates import get_template, load_builtin_templates

    repo, validator, task, _ = _fixture(tmp_path)
    load_builtin_templates()
    plan = Planner(validator._tools, lambda: None, validator).from_template(
        get_template("modeling"),
        {"dataset_id": "dataset-1", "target_col": "y",
         "feature_cols": ["signal", "affordability", "noise"],
         "split_col": "split", "split_values": {"train": "train", "test": "test", "oot": "oot"},
         "recipe": "lr", "recipes": ["lr"], "seed": 7, "holdout_values": ["oot"],
         "sample_weight_col": "", "n_trials": 1},
        task_id=task.id,
    )
    assert not validator.validate_problems(plan)
    plan.status = PlanStatus.VALIDATED
    repo.create_plan(plan)
    plan = repo.load_plan(plan.id)
    gate = PlanMessageComposer(load_output=repo.load_step_output).plan_overview_message(plan)
    context = approved_plan_overview(repo, validator, task_id=task.id, gate_metadata=gate.metadata)
    wire = Wire(monkeypatch)
    decide_gate(wire.client, gate={"content": gate.content, "metadata": gate.metadata}, plan_overview=context)
    sent = wire.last_context()["plan_overview"]["steps"]
    by_id = {step["id"]: step for step in sent}
    references = 0
    for step in plan.steps:
        for key in ("split_col", "sample_weight_col", "features"):
            value = step.inputs.get(key)
            if isinstance(value, str) and value.startswith("$ref:"):
                assert by_id[step.id]["inputs"][key] == value
                references += 1
    assert references > 10
    assert len(sent) == 12


def test_stale_wrong_task_unbound_or_forged_plan_has_zero_requests(
    monkeypatch, tmp_path
):
    repo, validator, task, gate = _fixture(tmp_path)
    wire = Wire(monkeypatch)
    bad = deepcopy(gate["metadata"])
    bad["confirmation_snapshot"]["expected_plan_fingerprint"] = "b" * 64
    for repository, owner, metadata in [
        (repo, task.id, bad),
        (repo, "other-task", gate["metadata"]),
        ({}, task.id, gate["metadata"]),
    ]:
        with pytest.raises(OutboundContextError):
            approved_plan_overview(
                repository, validator, task_id=owner, gate_metadata=metadata
            )
    for value in [
        {"steps": []},
        ApprovedPlanOverview('{"private":"PRIVATE_BODY"}', object()),
    ]:
        with pytest.raises(OutboundContextError):
            decide_gate(wire.client, gate=gate, plan_overview=value)
    assert wire.requests == []


def test_sensitive_field_name_prevents_plan_context_and_requests(monkeypatch, tmp_path):
    repo, validator, task, gate = _fixture(tmp_path)
    from marvis.db_schema import connect

    with connect(repo.db_path) as conn:
        row = conn.execute(
            "SELECT inputs_json FROM plan_steps WHERE id='setup'"
        ).fetchone()
        inputs = json.loads(row[0])
        inputs["target_col"] = "api_key=synthetic-private-key"
        conn.execute(
            "UPDATE plan_steps SET inputs_json=? WHERE id='setup'",
            (json.dumps(inputs),),
        )
    plan = repo.load_plan("plan-owned")
    gate = PlanMessageComposer(load_output=repo.load_step_output).plan_overview_message(
        plan
    )
    wire = Wire(monkeypatch)
    with pytest.raises(OutboundContextError, match="字段名称"):
        approved_plan_overview(
            repo, validator, task_id=task.id, gate_metadata=gate.metadata
        )
    assert wire.requests == []


def test_native_per_recipe_budgets_and_full_field_names_reach_protocol(
    monkeypatch, tmp_path
):
    repo, validator, task, _ = _fixture(tmp_path)
    from marvis.db_schema import connect

    inputs = {
        "recipe": "lr",
        "recipes": ["lr", "xgb"],
        "seed": 7,
        "n_trials_by_recipe": {"lr": 1, "xgb": 17},
    }
    with connect(repo.db_path) as conn:
        row = conn.execute(
            "SELECT inputs_json FROM plan_steps WHERE id='setup'"
        ).fetchone()
        setup = json.loads(row[0])
        setup.update(target_col="风险 标签", features=["收入(月)", "申请 金额"])
        conn.execute(
            "UPDATE plan_steps SET inputs_json=? WHERE id='setup'", (json.dumps(setup),)
        )
        conn.execute(
            "UPDATE plan_steps SET tool_plugin='modeling',tool_name='configure_tuning',inputs_json=? WHERE id='finish'",
            (json.dumps(inputs),),
        )
    plan = repo.load_plan("plan-owned")
    gate = PlanMessageComposer(load_output=repo.load_step_output).plan_overview_message(
        plan
    )
    context = approved_plan_overview(
        repo, validator, task_id=task.id, gate_metadata=gate.metadata
    )
    wire = Wire(monkeypatch)
    decide_gate(
        wire.client,
        gate={"content": gate.content, "metadata": gate.metadata},
        plan_overview=context,
    )
    steps = wire.last_context()["plan_overview"]["steps"]
    assert steps[0]["inputs"]["target_col"] == "风险 标签"
    assert steps[0]["inputs"]["features"] == ["收入(月)", "申请 金额"]
    assert steps[1]["inputs"]["n_trials_by_recipe"] == {"lr": 1, "xgb": 17}


@pytest.mark.parametrize(
    "budgets", [{"unregistered_recipe": 1}, {"lr": True}, {"lr": -1}]
)
def test_per_recipe_controls_never_silently_drop_invalid_bindings(
    monkeypatch, tmp_path, budgets
):
    from marvis.llm_outbound_plan_overview import _modeling_recipe_controls

    wire = Wire(monkeypatch)
    with pytest.raises(OutboundContextError):
        _modeling_recipe_controls({"n_trials_by_recipe": budgets})
    assert wire.requests == []
