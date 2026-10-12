import json
from pathlib import Path
import subprocess

import pytest

from marvis.agent.strategy_request_compiler import compile_strategy_request
from marvis.data.field_aliases import FieldAliasError, alias_grounding, field_alias_bindings, grounding_fields
from marvis.agent.strategy_request_compiler.grounding import _automatic_tree_column_mention_resolution


class LLM:
    def __init__(self, *replies):
        self.replies = replies
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


def test_aliases_preserve_wire_instruction_and_ground_physical_role_bindings():
    utterance = "客户 ID 字段 客户标识；时间字段 账龄；状态字段 逾期状态；使用 余额 作为余额权重。"
    columns = ("account_id", "mob", "bad", "balance")
    aliases = field_alias_bindings(columns, dict(zip(columns, ("客户标识", "账龄", "逾期状态", "余额"))))
    llm = LLM({"request_kind": "standard_workflow", "workflow": "roll_rate_matrix", "workflow_inputs": {
        "id_col": "account_id", "time_col": "mob", "status_col": "bad", "states": ["0", "1"],
        "balance_col": "balance", "observation_semantics": "adjacent_observation",
    }})
    result = compile_strategy_request(utterance, allowed_columns=columns, field_aliases=aliases, llm=llm)
    assert result.draft is not None, result.clarification
    payload = json.loads(llm.calls[0]["user_prompt"])
    assert payload["instruction"] == utterance
    assert set(payload["allowed_columns"]) == set(columns)
    assert {item["physical"] for item in payload["field_aliases"]} == set(columns)
    assert result.draft.workflow_inputs["time_col"] == "mob"


@pytest.mark.parametrize("names", [
    {"a": "风险分", "b": "风险分"},
    {"a": "b"},
    {"a": "ｂ"},
    {"a": "Income", "b": "income"},
])
def test_ambiguous_business_name_refuses_before_any_wire_request(names):
    llm = LLM({"clarification": "unused"})
    instruction = "字段 INCOME" if "Income" in names.values() else "用风险分分析" if "风险分" in names.values() else "用 b 分析"
    result = compile_strategy_request(instruction, allowed_columns=("a", "b"),
                                      field_aliases=field_alias_bindings(("a", "b"), names), llm=llm)
    assert result.clarification_code == "field_alias_binding_invalid"
    assert llm.calls == []


def test_original_whitespace_is_preserved_on_initial_and_format_repair_wire():
    utterance = "  请生成当前策略迭代评审报告。 \n"
    llm = LLM("invalid JSON", {"request_kind": "standard_workflow", "workflow": "strategy_report_bundle_v2", "workflow_inputs": {}})
    result = compile_strategy_request(utterance, allowed_columns=("x",), llm=llm)
    assert result.draft is not None
    assert len(llm.calls) == 2
    assert all(json.loads(call["user_prompt"])["instruction"] == utterance for call in llm.calls)


def test_only_explicit_field_spans_resolve_without_rewriting_any_control_text():
    utterance = '字段“风险分”；字段“风险分数”；不要生成报告。'
    bindings = (("a", "风险分"), ("b", "风险分数"))
    assert alias_grounding(utterance, ("a", "b"), bindings)[0] == utterance
    mentions, ambiguous = _automatic_tree_column_mention_resolution(utterance, grounding_fields(("a", "b"), bindings))
    assert not ambiguous
    assert [(utterance[start:end], str(column)) for start, end, column in mentions] == [("风险分", "a"), ("风险分数", "b")]
    assert alias_grounding("income_index income", ("a",), (("a", "income"),))[0] == "income_index income"


def test_alias_does_not_erase_explicit_balance_role_or_allow_another_valid_column():
    columns = ("account_id", "mob", "bad", "balance", "other_balance")
    aliases = field_alias_bindings(columns, {"balance": "余额"})
    utterance = "客户 ID 字段 account_id；时间字段 mob；状态字段 bad；使用 余额 作为余额权重。"
    grounded, _ = alias_grounding(utterance, columns, aliases)
    assert grounded == utterance
    reply = {"request_kind": "standard_workflow", "workflow": "roll_rate_matrix", "workflow_inputs": {
        "id_col": "account_id", "time_col": "mob", "status_col": "bad", "states": ["0", "1"],
        "balance_col": "other_balance", "observation_semantics": "adjacent_observation",
    }}
    result = compile_strategy_request(utterance, allowed_columns=columns, field_aliases=aliases, llm=LLM(reply))
    assert result.draft is None
    assert result.clarification_code == "roll_rate_column_binding_not_grounded"


@pytest.mark.parametrize("alias,utterance,workflow,inputs,code", [
    ("不要", "不要 生成当前策略迭代评审报告。", "strategy_report_bundle_v2", {}, "strategy_report_bundle_v2_intent_negated"),
    ("部署", "请生成当前策略迭代评审报告；部署 x。", "strategy_report_bundle_v2", {}, "strategy_report_bundle_v2_single_operation_required"),
    ("not", "do not export strategy-current-1 to Python, SQL, JSON and equivalence evidence", "strategy_dsl_delivery", {"strategy_id": "strategy-current-1"}, "strategy_dsl_delivery_intent_negated"),
])
def test_alias_never_changes_negation_or_authorizes_chained_side_effect(alias, utterance, workflow, inputs, code):
    llm = LLM({"request_kind": "standard_workflow", "workflow": workflow, "workflow_inputs": inputs})
    result = compile_strategy_request(utterance, allowed_columns=("x",), field_aliases=(("x", alias),), llm=llm)
    assert result.draft is None
    assert result.clarification_code == code
    assert all(json.loads(call["user_prompt"])["instruction"] == utterance for call in llm.calls)
    mentions, _ = _automatic_tree_column_mention_resolution(utterance, grounding_fields(("x",), (("x", alias),)))
    assert not any(utterance[start:end] == alias for start, end, _ in mentions)


def test_unsafe_label_never_leaves_workspace_and_unknown_binding_refuses():
    assert field_alias_bindings(("a",), {"a": "密码=secret-material"}) == ()
    with pytest.raises(FieldAliasError):
        field_alias_bindings(("a",), {"unknown": "收入"})
    llm = LLM({"clarification": "unused"})
    result = compile_strategy_request("密码=secret-material", allowed_columns=("a",), field_aliases=(("a", "收入"),), llm=llm)
    assert result.clarification_code == "outbound_policy"
    assert not llm.calls


def test_field_display_modes_leave_submitted_values_and_nonfield_choices_exact():
    subprocess.run(["node", "--input-type=module", "-e", """
      import assert from 'node:assert/strict';
      import {applyStrategyFieldLabels} from './marvis/static/js/v2/strategy_field_labels.js';
      const preference={value:'both'};
      const field={value:'income',textContent:'income · 3 种受认证分箱方法',dataset:{},closest:()=>({dataset:{candidateLabField:'cross_search_features'}})};
      const strategy={value:'strategy-1',textContent:'业务策略'};
      const root={querySelector:()=>preference,querySelectorAll:()=>[field,strategy]};
      const payload={field_labels:{fields:[{physical:'income',business_name:'月收入'}]}};
      applyStrategyFieldLabels(root,payload);assert.equal(field.textContent,'月收入 (income) · 3 种受认证分箱方法');
      preference.value='business';applyStrategyFieldLabels(root,payload);assert.equal(field.textContent,'月收入 · 3 种受认证分箱方法');
      preference.value='physical';applyStrategyFieldLabels(root,payload);assert.equal(field.textContent,'income · 3 种受认证分箱方法');
      assert.equal(field.value,'income');assert.equal(strategy.textContent,'业务策略');
      payload.field_labels.fields[0].business_name='<img src=x onerror=bad>';
      preference.value='business';applyStrategyFieldLabels(root,payload);assert.equal(field.textContent,'<img src=x onerror=bad> · 3 种受认证分箱方法');
    """], cwd=Path(__file__).parents[1], check=True, capture_output=True, text=True)
