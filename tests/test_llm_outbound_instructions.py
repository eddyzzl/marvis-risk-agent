import json

import pytest

from marvis.agent.service import answer_chat_message
from marvis.agent.strategy_request_compiler import compile_strategy_request
from marvis.llm_outbound import OutboundContextError, planning_prompt
from marvis.llm_outbound_instructions import MAX_INSTRUCTION_CHARS, column_identifiers, user_instruction
from tests.test_agent_service import _task
from tests.test_llm_outbound import Wire


@pytest.mark.parametrize("instruction", [
    "只做 JOIN，不训练模型；如果核验通过再生成报告。预算最多 20 次。",
    "run roll rate with id_col=account_id, time_col=mob, status_col=bad, balance_col=balance",
    "按申请月分组，阈值 0.08；不要修改参数，保留其余结论。",
    "试验 lr,xgb,lgb，调参轮数 15，最多生成 5 个候选。",
    "对 score 做单变量分析，score manual 切点 [200, 400]",
    "使用 raw PD 分带边界 [0, 0.2, 0.6, 1] 构建分数带。",
    "Pool 默认输出值：[1,2]；命中动作：reject",
])
def test_complete_user_controls_preserve_meaning_without_topic_rewriting(instruction):
    assert user_instruction(instruction) == instruction
    context = json.loads(planning_prompt({"goal": instruction, "user_constraint": instruction}))
    assert context["goal"] == context["user_constraint"] == instruction
    assert "request_topics" not in context


@pytest.mark.parametrize("instruction", [
    "api_key=TEST_ONLY_VALUE", "密码是 TEST_ONLY_VALUE", "API key is TEST_ONLY_VALUE",
    "OPENAI_API_KEY=TEST_ONLY_VALUE", "DB_PASSWORD=TEST_ONLY_VALUE",
    '客户记录 {"客户姓名":"测试姓名","授信余额":42}',
    '样本数据 [["测试客户",42],["测试客户乙",84]]',
    '客户姓名：测试姓名', '读取 /Users/example/private/data.csv',
    '访问 https://private.invalid/data', '身份证号：123456789012345678',
    '手机号为13812345678', '邮箱 a@example.invalid',
    'customer_id,balance,label\n001,42,1',
    'customer_name,balance\nTEST_ONLY_PERSON,42',
    'customer_name: TEST_ONLY_PERSON\ncustomer_id: customer-739\nbalance: 42',
    '```python\nx = 42\nprint(x)\n```',
    'import pandas as pd\nRMC_SAMPLE_DF = private_data',
    '{"cells":[],"nbformat":4}', '按 age 分组\u200b',
    '切点 [[1,2],[3,4]]', '切点 ["TEST_ONLY_PERSON", 42]',
    '样本明细：[200, 400]', '[200, 400]',
    '切点 [9007199254740993]', '切点 [NaN]',
])
def test_forbidden_user_material_has_zero_requests_and_no_sensitive_error_echo(monkeypatch, instruction):
    wire = Wire(monkeypatch)
    with pytest.raises(OutboundContextError):
        user_instruction(instruction)
    result = compile_strategy_request(instruction, allowed_columns=["age"], llm=wire.client)
    assert result.draft is None and result.clarification_code == "outbound_policy"
    assert instruction not in result.clarification
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    _, metadata = answer_chat_message(task=_task(), user_message=instruction,
        conversation=[], evidence={}, memory_context=None, model_profile={})
    assert metadata["fallback"] is True
    assert wire.requests == []


def test_oversize_instruction_is_refused_without_truncation():
    with pytest.raises(OutboundContextError, match="未截断"):
        user_instruction("x" * (MAX_INSTRUCTION_CHARS + 1))


def test_real_chat_protocol_uses_user_turns_and_aggregates_without_source_or_assistant_prose(monkeypatch):
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "当前 KS 为 0.31；历史比较证据尚未提供。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    message, metadata = answer_chat_message(task=_task(), user_message="解释当前 KS，不训练；预算最多 2 次。",
        conversation=[{"role":"user","content":"只讨论效果"},
                      {"role":"assistant","content":"PRIVATE_REPORT_PROSE"},
                      {"role":"user","content":"密码是 TEST_ONLY_VALUE"}],
        evidence={"metrics":{"ks":.31},"raw":"PRIVATE_SAMPLE_ROWS","saved_report_narrative":{"text":"PRIVATE_REPORT_PROSE"}},
        memory_context={"memories":[{"summary":"PRIVATE_MEMORY_PROSE"}]}, model_profile={})
    assert metadata["fallback"] is False and "0.31" in message
    assert len(wire.requests) == 1
    request=json.loads(wire.requests[0]["messages"][1]["content"])
    assert request["user_message"] == "解释当前 KS，不训练；预算最多 2 次。"
    assert request["conversation_memory"]["user_turns"] == [{"role":"user","instruction":"只讨论效果"}]
    assert request["conversation_memory"]["withheld_user_turn_count"] == 1
    assert request["evidence"]["metrics"]["ks"] == .31
    assert request["history_comparison_available"] is False
    assert "PRIVATE" not in json.dumps(wire.requests)
    assert "TEST_ONLY_VALUE" not in json.dumps(wire.requests)


def test_strategy_protocol_preserves_full_user_goal_and_registered_contract(monkeypatch):
    wire=Wire(monkeypatch)
    wire.reply={"operation":"backtest","strategy_type":"approval"}
    instruction="回测当前审批策略，不采纳、不部署，最多 20 次计算。"
    result=compile_strategy_request(instruction,allowed_columns=["申请月","score"],llm=wire.client)
    assert result.draft is not None
    request=json.loads(wire.requests[0]["messages"][1]["content"])
    assert request["instruction"] == instruction
    assert request["allowed_columns"] == ["score","申请月"]
    assert wire.requests[0]["response_format"]["json_schema"]["schema"]["type"] == "object"


def test_conversational_previous_question_is_not_mistaken_for_historical_model_comparison(monkeypatch):
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "选择字段后先确认口径。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    message, metadata = answer_chat_message(task=_task(), user_message="解释之前的字段确认流程",
        conversation=[{"role":"user","content":"如何确认字段？"}], evidence={},
        memory_context=None, model_profile={})
    assert message == wire.reply and not metadata["fallback"]


def test_unicode_column_roles_are_preserved_and_material_labels_are_refused():
    assert column_identifiers(["客户ID","申请月","balance"]) == ["客户ID","申请月","balance"]
    with pytest.raises(OutboundContextError):
        column_identifiers(["/Users/example/customer.csv"])
