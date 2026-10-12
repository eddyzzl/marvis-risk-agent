"""Intent routing through the real client protocol boundary, without providers."""

import json

import pytest

from marvis.agent.semantic_intent import (
    ALL_SEMANTIC_INTENTS, INTENT_ADHOC_QUERY, INTENT_NONE, route_semantic_intent,
)
from marvis.llm_client import LLMClientError, LLMClientErrorKind
from marvis.llm_outbound import application_schema
from marvis.llm_outbound_intent import semantic_intent_context, workflow_intent_prompt
from marvis.orchestrator.intent import IntentRouter
from marvis.orchestrator.templates import load_builtin_templates
from tests.test_llm_outbound import Wire


def _reply(instruction, **fields):
    return {"intent": INTENT_ADHOC_QUERY, "evidence_quote": instruction, "reason": "用户请求问数",
            "confidence": "high", "is_question": False, "is_conditional": False,
            "requests_change": False, "withholds_action": False, **fields}


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "none"])
def test_semantic_intent_normal_route_retains_user_goal_and_exact_enum_contract(monkeypatch, mode):
    wire = Wire(monkeypatch)
    wire.client.profile["structured_output"] = mode
    instruction = "按 age 分组计算坏率，只看当前数据，预算最多 20 次计算。"
    wire.replies = [_reply(instruction), _reply(instruction)]
    result = route_semantic_intent(wire.client, task_type="feature_analysis", instruction=instruction,
        context={"has_ready_dataset": True, "task_id": "task-1", "rows": [{"customer": "PRIVATE_BODY"}],
                 "pending_question": "PRIVATE_SOURCE_PROSE", "available_join_tables": ["PRIVATE_NAME.csv"]},
        allowed_intents=tuple(sorted(ALL_SEMANTIC_INTENTS)))
    assert result.accepted is True and result.intent == INTENT_ADHOC_QUERY
    assert len(wire.requests) == 2
    for body in wire.requests:
        request = json.loads(body["messages"][1]["content"])
        assert request["instruction"] == instruction
        assert request["current_context"] == {"has_ready_dataset": True, "task_id": "task-1"}
        assert set(request["allowed_intents"]) == ALL_SEMANTIC_INTENTS
        assert set(request["intent_definitions"]) == ALL_SEMANTIC_INTENTS
        assert "PRIVATE" not in json.dumps(body)
        if mode == "json_schema":
            schema = body["response_format"]["json_schema"]
        else:
            schema = json.loads(body["messages"][0]["content"].split("\n")[-1])
        assert set(schema["schema"]["properties"]["intent"]["enum"]) == ALL_SEMANTIC_INTENTS


def test_negation_conditions_thresholds_and_budget_survive_and_remain_unauthorized(monkeypatch):
    wire = Wire(monkeypatch)
    instruction = "只做分组统计，不训练；坏率阈值 0.08，如果材料核验通过再输出报告，最多 20 次计算。"
    wire.replies = [_reply(instruction, is_conditional=True, withholds_action=True)] * 2
    result = route_semantic_intent(wire.client, task_type="feature_analysis", instruction=instruction,
        context={"has_ready_dataset": True}, allowed_intents=(INTENT_ADHOC_QUERY, INTENT_NONE))
    assert result.accepted is False
    assert len(wire.requests) == 2
    assert all(json.loads(body["messages"][1]["content"])["instruction"] == instruction for body in wire.requests)


def test_repair_preserves_frozen_user_request_and_drops_model_reason_and_extra_fields(monkeypatch):
    wire = Wire(monkeypatch)
    instruction = "按 age 分组看坏率，阈值 0.08，不训练模型。"
    bad = _reply(instruction, reason="PRIVATE_RESPONSE_BODY", unexpected={"raw": "PRIVATE_EXTRA"})
    wire.replies = [bad, _reply(instruction), _reply(instruction)]
    result = route_semantic_intent(wire.client, task_type="feature_analysis", instruction=instruction,
        context={"has_ready_dataset": True}, allowed_intents=(INTENT_ADHOC_QUERY, INTENT_NONE))
    assert result.accepted is True
    assert len(wire.requests) == 3
    repair = json.loads(wire.requests[1]["messages"][1]["content"])
    assert repair["original_request"] == json.loads(wire.requests[0]["messages"][1]["content"])
    assert repair["first_pass_fields"]["evidence_quote"] == instruction
    assert "reason" not in repair["first_pass_fields"]
    assert "PRIVATE_RESPONSE_BODY" not in json.dumps(wire.requests)
    assert "PRIVATE_EXTRA" not in json.dumps(wire.requests)


@pytest.mark.parametrize("instruction", [
    "请按样本统计，api_key=sk-synthetic-sensitive-12345",
    "读取 /Users/test/customer.csv 然后训练",
    '检查 {"sample_rows":[{"customer":"PRIVATE_BODY"}]}',
    "客户姓名：PRIVATE_BODY，然后分析",
    "分析 " + "x" * 16384,
], ids=["credential", "path", "raw_samples", "personal_record", "length"])
def test_forbidden_user_material_refuses_before_either_semantic_pass(monkeypatch, instruction):
    wire = Wire(monkeypatch)
    result = route_semantic_intent(wire.client, task_type="feature_analysis", instruction=instruction,
        context={"has_ready_dataset": True}, allowed_intents=(INTENT_ADHOC_QUERY, INTENT_NONE))
    assert result.accepted is False
    assert wire.requests == []
    assert "PRIVATE_BODY" not in result.reason
    assert "未发送" in result.reason or "未截断" in result.reason


def test_known_risk_intake_question_is_kept_but_arbitrary_context_prose_is_withheld():
    question = ("平台正在等待用户确认材料已上传，并补充样本范围和字段口径；"
                "回答用于检查材料并准备已选分析。后续执行仍受原流程校验与确认门约束。")
    context = {"risk_setup_phase": "await_materials", "analysis_kind": "standard_vintage",
               "label_semantics": "incremental", "pending_question": question,
               "description": "PRIVATE_SOURCE_PROSE"}
    assert semantic_intent_context(context) == {key: value for key, value in context.items() if key != "description"}
    context["pending_question"] = "PRIVATE_SOURCE_PROSE"
    assert "pending_question" not in semantic_intent_context(context)


def test_workflow_classifier_preserves_full_goal_and_installed_candidates(monkeypatch):
    load_builtin_templates()
    wire = Wire(monkeypatch)
    wire.reply = {"choice": "sample_echo"}
    router = IntentRouter(lambda: wire.client, tool_registry=None)
    goal = "do the thing; do not train; stop if a prerequisite is missing; budget 20"
    choice = router._llm_classify(goal, {"task_id": "task-1", "rows": ["PRIVATE_BODY"]}, ["sample_echo", "novel"])
    assert choice == "sample_echo"
    context = json.loads(wire.requests[0]["messages"][1]["content"])
    assert context["goal"] == goal
    assert context["task_context"] == {"task_id": "task-1"}
    assert context["candidates"] == ["sample_echo", "novel"]
    assert wire.requests[0]["response_format"]["json_schema"]["schema"]["properties"]["choice"]["enum"] == context["candidates"]


def test_workflow_classifier_policy_failure_is_visible_and_has_zero_requests(monkeypatch):
    load_builtin_templates()
    wire = Wire(monkeypatch)
    router = IntentRouter(lambda: wire.client, tool_registry=None, propagate_llm_errors=True)
    with pytest.raises(LLMClientError) as caught:
        router._llm_classify("密码=PRIVATE_BODY", {}, ["sample_echo", "novel"])
    assert caught.value.error_kind == LLMClientErrorKind.OUTBOUND_POLICY
    assert wire.requests == []


def test_unknown_workflow_cannot_be_declared_as_installed_schema_or_prompt(monkeypatch):
    wire = Wire(monkeypatch)
    for build in (lambda: application_schema("workflow_intent", candidates=["PRIVATE_UNKNOWN"]),
                  lambda: workflow_intent_prompt("do the thing", {}, ["PRIVATE_UNKNOWN"])):
        with pytest.raises(ValueError):
            build()
    assert wire.requests == []
