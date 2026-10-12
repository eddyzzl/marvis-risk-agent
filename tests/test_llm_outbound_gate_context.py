import json

import pytest

from marvis.agent.semantic_authorization import review_semantic_authorization
from marvis.agent.instruction_router import route_instruction
from marvis.llm_outbound import OutboundContextError, gate_instruction_prompt
from marvis.llm_outbound_gate_context import ApprovedGateContext, authorization_gate
from tests.test_llm_outbound import Wire


@pytest.mark.parametrize("purpose", ["portfolio_order", "labeling_proposal", "execute_plan", "current_step", "role_assignment", "validation_stage"])
def test_installed_confirmation_purpose_reaches_actual_protocol_without_local_prose(monkeypatch, purpose):
    wire = Wire(monkeypatch)
    instruction = "立即按当前已展示的口径执行，并保留已有风险说明。"
    wire.reply = dict(verdict="authorize", evidence_quote=instruction, reason="明确接受当前动作",
                      confidence="high", is_question=False, is_conditional=False,
                      requests_change=False, withholds_authorization=False)
    context = authorization_gate(purpose, bindings={"proposal_hash": "a" * 64, "local_report": "PRIVATE_REPORT_BODY"})
    review = review_semantic_authorization(wire.client, gate_context=context,
        instruction=instruction, proposed_params={})
    assert review.authorized
    request = wire.last_context()
    assert request["instruction"] == instruction
    assert request["gate_context"]["purpose"] == purpose
    assert request["gate_context"]["application_contract"]
    assert "PRIVATE_REPORT_BODY" not in json.dumps(wire.requests)
    assert len(request["gate_context"]["binding"]["complete_binding_sha256"]) == 64


def test_unknown_or_forged_gate_contract_cannot_bless_arbitrary_ui_body():
    with pytest.raises(OutboundContextError):
        authorization_gate("PRIVATE_REPORT_BODY")
    with pytest.raises(OutboundContextError):
        gate_instruction_prompt(ApprovedGateContext("current_step", '{"body":"PRIVATE_REPORT_BODY"}', object()), "继续")


@pytest.mark.parametrize("family", ["modeling", "feature_analysis"])
def test_installed_workflow_family_reaches_both_actual_protocol_passes(monkeypatch, family):
    wire = Wire(monkeypatch)
    instruction = "请为已选择的工作流创建可审核计划。"
    context = authorization_gate("create_workflow_plan", bindings={
        "workflow_family": family, "local_report": "PRIVATE_REPORT_BODY",
    })
    wire.reply = dict(action="confirm", params={}, constraint="", reason="仅创建计划",
        confidence="high", explicit_authorization=True)
    route = route_instruction(wire.client, gate_context=context, instruction=instruction,
        strict_contract=True)
    assert route["action"] == "confirm"
    wire.reply = dict(verdict="authorize", evidence_quote=instruction, reason="明确创建计划",
        confidence="high", is_question=False, is_conditional=False,
        requests_change=False, withholds_authorization=False)
    assert review_semantic_authorization(wire.client, gate_context=context,
        instruction=instruction, proposed_params={}).authorized
    assert len(wire.requests) == 2
    for request in wire.requests:
        payload = json.loads(request["messages"][1]["content"])
        assert payload["instruction"] == instruction
        assert payload["gate_context"]["purpose"] == "create_workflow_plan"
        assert payload["gate_context"]["binding"]["workflow_family"] == family
        assert "PRIVATE_REPORT_BODY" not in json.dumps(request)


@pytest.mark.parametrize("family", [None, "", "strategy", "modeling PRIVATE_REPORT_BODY",
    {"body": "PRIVATE_REPORT_BODY"}, ["modeling"], True])
def test_unsupported_workflow_family_rejects_before_any_request(monkeypatch, family):
    wire = Wire(monkeypatch)
    with pytest.raises(OutboundContextError):
        context = authorization_gate("create_workflow_plan", bindings={"workflow_family": family})
        route_instruction(wire.client, gate_context=context, instruction="请创建计划。")
    assert wire.requests == []
