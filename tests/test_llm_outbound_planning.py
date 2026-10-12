"""Planning controls are exact or explicitly refused, never silently reduced."""

import json
from copy import copy, deepcopy

import pytest

from marvis.llm_outbound import OutboundContextError, planning_prompt
from marvis.llm_outbound_catalog import restore_planning_catalog
from marvis.orchestrator.contracts import Plan, PlanStep, StepStatus
from marvis.orchestrator.planner import (
    PLAN_SYS, PlannerConstraints, PlannerOutboundPolicyError, RequiredLiteralInput,
    _fit_planner_request, build_plan_prompt, build_replan_prompt,
)
from marvis.plugins.manifest import ToolRef
from tests.test_llm_outbound import Wire
from tests.test_llm_outbound_catalog import _dynamic_manifest, _registry, _send


def test_copy_preserves_existing_prompt_authority_without_minting_it():
    from marvis.llm_outbound import StructuralPrompt
    from marvis.llm_outbound_instructions import user_instruction

    prompt = planning_prompt({"goal": "核验当前绑定"})
    instruction = user_instruction("核验当前绑定")
    for operation in (copy, deepcopy):
        assert operation(prompt) is prompt
        assert operation(instruction) is instruction
    with pytest.raises(OutboundContextError):
        StructuralPrompt(str(prompt))


def _catalog(tmp_path, schema=None):
    plugins, registry = _registry(tmp_path)
    plugins.register(_dynamic_manifest(schema))
    return registry.catalog_for_planner()


def test_exact_task_revision_roles_time_split_and_fields_reach_protocol(tmp_path, monkeypatch):
    wire = Wire(monkeypatch)
    context = {
        "workflow_family": "fixed", "entrypoint": "vintage", "analysis_kind": "vtg_terminal",
        "dataset_id": "ds-24b05", "active_dataset_id": "ds-24b05", "active_dataset_revision": 23,
        "target_col": "bad_flag", "score_col": "score", "time_col": "event_date",
        "features": ["mob", "balance"], "split": {"train": "2024-01", "test": "2024-02"},
        "column_map": {"cohort": "loan_month", "mob": "mob", "bad": "bad_flag"},
    }
    payload = _send(wire, build_plan_prompt("按当前样本和口径生成Vintage报告", _catalog(tmp_path), {}, context, None))
    assert payload["task_context"] == context


@pytest.mark.parametrize("context", [
    {"dataset_id": "fixture://risk-analysis/current-snapshot"},
    {"active_dataset_revision": -1},
    {"entrypoint": "private_source_kind"},
    {"split": {"train": "RAW_SAMPLE_VALUE"}},
])
def test_invalid_execution_bindings_refuse_before_any_request(tmp_path, monkeypatch, context):
    wire = Wire(monkeypatch)
    with pytest.raises(PlannerOutboundPolicyError):
        build_plan_prompt("核验当前绑定", _catalog(tmp_path), {}, context, None)
    assert wire.requests == []


def test_required_conditional_literals_and_full_remaining_inputs_preserve_exact_contract(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path)
    ref = ToolRef("approved_dynamic", "bounded_tool")
    values = {"mode": "frozen", "threshold": .4}
    constraints = PlannerConstraints(required_literal_inputs=(RequiredLiteralInput(ref, values),))
    wire = Wire(monkeypatch)
    prompt = build_plan_prompt("如果核验通过再用冻结模式，阈值0.4", catalog, {}, {}, None, constraints=constraints)
    first = _send(wire, prompt)
    assert first["planner_constraints"]["required_literal_inputs"][0]["inputs"] == values
    plan = Plan(id="plan-1", task_id="task-1", goal="如果核验通过再用冻结模式，阈值0.4",
                source="llm", template_id=None, autonomy_level=1, steps=[
                    PlanStep(id="bounded-step", plan_id="plan-1", index=0, title="结构步骤", tool_ref=ref,
                             inputs=values, depends_on=[], post_checks=[], status=StepStatus.PENDING),
                ])
    second_prompt = build_replan_prompt(plan, catalog, [], observation={}, reason="修改阈值", last_error=None)
    second = _send(wire, second_prompt)
    assert second["remaining_steps"][0]["inputs"] == values
    # Budget replay reuses only the original immutable control authority.
    replayed = planning_prompt(restore_planning_catalog(json.loads(second_prompt), second_prompt))
    assert json.loads(replayed)["remaining_steps"][0]["inputs"] == values
    altered = json.loads(second_prompt)
    altered["remaining_steps"][0]["inputs"]["threshold"] = .8
    with pytest.raises(OutboundContextError):
        restore_planning_catalog(altered, second_prompt)


@pytest.mark.parametrize("values", [
    {"mode": "private_injected_mode"}, {"threshold": .9}, {"unexpected": .4},
    {"description": "UNAPPROVED_LOCAL_TEXT"},
])
def test_schema_strings_cannot_bless_local_prose_or_invalid_literals(tmp_path, monkeypatch, values):
    wire = Wire(monkeypatch)
    constraints = PlannerConstraints(required_literal_inputs=(
        RequiredLiteralInput(ToolRef("approved_dynamic", "bounded_tool"), values),
    ))
    with pytest.raises(PlannerOutboundPolicyError):
        build_plan_prompt("使用当前冻结方案", _catalog(tmp_path), {}, {}, None, constraints=constraints)
    assert wire.requests == []


def test_exact_user_goal_quote_can_supply_free_string_but_memory_prose_cannot(tmp_path, monkeypatch):
    schema = {"type": "object", "properties": {"message": {"type": "string", "maxLength": 80}},
              "required": ["message"], "additionalProperties": False}
    catalog = _catalog(tmp_path, schema)
    values = {"message": "current"}
    constraints = PlannerConstraints(required_literal_inputs=(
        RequiredLiteralInput(ToolRef("approved_dynamic", "bounded_tool"), values),
    ))
    wire = Wire(monkeypatch)
    prompt = build_plan_prompt("echo current", catalog, {"user_constraint": "UNAPPROVED_MEMORY_TEXT"}, {}, None,
                               constraints=constraints)
    payload = _send(wire, prompt)
    assert payload["planner_constraints"]["required_literal_inputs"][0]["inputs"] == values
    assert "UNAPPROVED_MEMORY_TEXT" not in json.dumps(wire.requests)
    with pytest.raises(PlannerOutboundPolicyError):
        build_plan_prompt("echo once", catalog, {}, {}, None, constraints=constraints)


def test_budget_cannot_drop_installed_schema_for_remaining_step_or_required_input(tmp_path, monkeypatch):
    schema = {"type": "object", "properties": {
        "mode": {"enum": ["frozen"]},
        **{f"field_{index}": {"type": "number", "minimum": .1, "maximum": .8} for index in range(1000)},
    }, "additionalProperties": False}
    catalog = _catalog(tmp_path, schema)
    ref = ToolRef("approved_dynamic", "bounded_tool")
    values = {"mode": "frozen"}
    plan = Plan(id="plan-1", task_id="task-1", goal="保持冻结模式，其余不变。", source="llm",
                template_id=None, autonomy_level=1, steps=[
                    PlanStep(id="bounded-step", plan_id="plan-1", index=0, title="结构步骤", tool_ref=ref,
                             inputs=values, depends_on=[], post_checks=[]),
                ])
    wire = Wire(monkeypatch)
    wire.client.profile.update(context_window=8192, max_output_tokens=4096)
    prompt = build_replan_prompt(plan, catalog, [], observation={}, reason="保持完整契约", last_error=None)
    with pytest.raises(PlannerOutboundPolicyError, match="未截断或发送"):
        _fit_planner_request(wire.client, system_prompt=PLAN_SYS, user_prompt=prompt)
    assert wire.requests == []
    assert json.loads(prompt)["remaining_steps"][0]["inputs"] == values
    assert len(json.loads(prompt)["available_tools"][0]["input_schema"]["properties"]) == 1001
