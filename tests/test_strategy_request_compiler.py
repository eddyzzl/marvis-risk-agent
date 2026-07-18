"""Pure natural-language strategy request compiler contract tests."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STRATEGY_OPERATIONS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    STRATEGY_TYPES,
    compile_strategy_request,
    validate_strategy_request,
)


class _SequencedLLM:
    def __init__(self, *replies: object) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        reply_index = min(len(self.calls) - 1, len(self.replies) - 1)
        reply = self.replies[reply_index]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _pricing_spec(*, field: str = "score", strategy_type: str = "pricing") -> dict:
    action = (
        {"type": "pricing", "value": 0.18}
        if strategy_type == "pricing"
        else {"type": "approval"}
    )
    rule_action = (
        {"type": "pricing", "value": 0.12}
        if strategy_type == "pricing"
        else {"type": "reject"}
    )
    return {
        "schema_version": "strategy.dsl.v1",
        "strategy_type": strategy_type,
        "match_policy": "first_match",
        "default_action": action,
        "rules": [
            {
                "rule_id": "risk-rule",
                "priority": 1,
                "condition": {
                    "op": "and",
                    "args": [
                        {
                            "op": "compare",
                            "field": field,
                            "operator": ">=",
                            "value": 700,
                        },
                        {
                            "op": "not",
                            "arg": {"op": "is_null", "field": "income"},
                        },
                    ],
                },
                "action": rule_action,
            }
        ],
    }


def _limit_economics() -> dict:
    return {
        "pd_col": "pd_12m",
        "lgd_value": 0.45,
        "utilization_col": "utilization",
    }


def _pricing_economics() -> dict:
    return {
        "ead_col": "ead",
        "pd_col": "pd_12m",
        "lgd_value": 0.5,
        "funding_rate_value": 0.04,
        "term_months_value": 12,
        "operating_cost_per_loan_value": 12,
    }


def test_operation_and_strategy_type_are_orthogonal() -> None:
    for operation in STRATEGY_OPERATIONS:
        for strategy_type in STRATEGY_TYPES:
            result = validate_strategy_request(
                {"operation": operation, "strategy_type": strategy_type},
                allowed_columns=(),
            )
            assert result.clarification is None
            assert result.draft is not None
            assert result.draft.operation == operation
            assert result.draft.strategy_type == strategy_type


def test_compile_uses_deterministic_json_schema_call_and_returns_confirmation() -> None:
    llm = _SequencedLLM(
        json.dumps(
            {
                "operation": "backtest",
                "strategy_type": "approval",
                "strategy_id": "strategy-7",
            }
        )
    )

    result = compile_strategy_request(
        "回测 strategy-7 的审批策略",
        allowed_columns=["score", "bad"],
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft["operation"] == "backtest"
    assert result.draft["strategy_type"] == "approval"
    assert result.clarification is None
    assert "审批策略" in result.confirmation
    assert "回测" in result.confirmation
    assert "strategy-7" in result.confirmation
    assert "指标由平台" in result.confirmation
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["temperature"] == 0.0
    assert call["response_format"] == {"type": "json_object"}
    assert call["json_schema"] == STRATEGY_REQUEST_JSON_SCHEMA
    assert call["stream"] is False
    assert call["caller"] == "strategy_request_compiler"
    assert call["prompt_name"] == "STRATEGY_REQUEST_COMPILER_SYS"
    assert call["prompt_version"] == 2


def test_all_optional_inputs_are_canonicalized_and_echoed_in_chinese() -> None:
    payload = {
        "operation": "develop",
        "strategy_type": "pricing",
        "objective": "在风险约束内提升收益",
        "max_bad_rate": 0.08,
        "min_approval_rate": 0.55,
        "baseline_strategy_id": "baseline-1",
        "strategy_id": "candidate-1",
        "adoption_reason": "收益改善且风险约束满足",
        "economics_inputs": _pricing_economics(),
        "strategy_spec": _pricing_spec(),
    }

    result = validate_strategy_request(
        payload,
        allowed_columns=["score", "income", "ead", "pd_12m"],
    )

    assert result.draft is not None
    assert result.draft.max_bad_rate == 0.08
    assert result.draft.min_approval_rate == 0.55
    assert result.draft.economics_inputs == payload["economics_inputs"]
    assert result.draft.strategy_spec["strategy_type"] == "pricing"
    assert result.draft.strategy_spec["rules"][0]["condition"]["op"] == "and"
    assert result.draft.to_dict()["strategy_id"] == "candidate-1"
    assert "定价策略" in result.confirmation
    assert "最大坏账率 8.00%" in result.confirmation
    assert "最低通过率 55.00%" in result.confirmation
    assert "EAD 取数据列 ead" in result.confirmation
    assert "PD 取数据列 pd_12m" in result.confirmation
    assert "资金成本率 取固定值 4.00%" in result.confirmation
    assert "LGD 取固定值 50.00%" in result.confirmation
    assert "单笔运营成本 取固定值 12" in result.confirmation
    assert "期限 取固定值 12 个月" in result.confirmation
    assert "1 条规则" in result.confirmation
    assert "默认动作" in result.confirmation
    assert "risk-rule" in result.confirmation
    assert '"field":"score"' in result.confirmation
    assert '"type":"pricing","value":0.12' in result.confirmation
    assert '"type":"pricing","value":0.18' in result.confirmation


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"operation": "backtest", "strategy_type": "approval", "metrics": {"ks": 0.5}},
            "不支持的字段",
        ),
        ({"operation": "ship", "strategy_type": "approval"}, "不支持的策略操作"),
        ({"operation": "analyze", "strategy_type": "scorecard"}, "不支持的策略类型"),
        (
            {"operation": "develop", "strategy_type": "approval", "max_bad_rate": True},
            "max_bad_rate",
        ),
        (
            {"operation": "develop", "strategy_type": "approval", "max_bad_rate": 1.1},
            "max_bad_rate",
        ),
        (
            {"operation": "develop", "strategy_type": "approval", "strategy_id": ""},
            "strategy_id",
        ),
    ],
)
def test_unknown_fields_types_and_ranges_become_chinese_clarification(
    payload: dict,
    message: str,
) -> None:
    result = validate_strategy_request(payload, allowed_columns=["score"])

    assert result.draft is None
    assert message in result.clarification
    assert result.confirmation is None


def test_strategy_spec_must_match_type_and_recursively_use_whitelisted_columns() -> None:
    mismatch = validate_strategy_request(
        {
            "operation": "apply",
            "strategy_type": "approval",
            "strategy_spec": _pricing_spec(),
        },
        allowed_columns=["score", "income"],
    )
    assert mismatch.draft is None
    assert "strategy_type" in mismatch.clarification
    assert "一致" in mismatch.clarification

    hallucinated = validate_strategy_request(
        {
            "operation": "apply",
            "strategy_type": "pricing",
            "strategy_spec": _pricing_spec(field="ghost_score"),
        },
        allowed_columns=["score", "income"],
    )
    assert hallucinated.draft is None
    assert "ghost_score" in hallucinated.clarification
    assert "不存在" in hallucinated.clarification


def test_invalid_dsl_and_profit_contract_fail_to_clarification() -> None:
    invalid_dsl = _pricing_spec()
    invalid_dsl["rules"][0]["condition"]["args"][0]["operator"] = "python_eval"
    dsl_result = validate_strategy_request(
        {
            "operation": "develop",
            "strategy_type": "pricing",
            "strategy_spec": invalid_dsl,
        },
        allowed_columns=["score", "income"],
    )
    assert dsl_result.draft is None
    assert "规则草案" in dsl_result.clarification

    profit_result = validate_strategy_request(
        {
            "operation": "develop",
            "strategy_type": "approval",
            "profit": {
                "ead_col": "missing_ead",
                "pd_col": "pd_12m",
                "annual_rate": 0.12,
                "funding_rate": 0.03,
                "lgd": 0.5,
                "operating_cost_per_loan": 10,
                "term_months": 12,
            },
        },
        allowed_columns=["pd_12m"],
    )
    assert profit_result.draft is None
    assert "missing_ead" in profit_result.clarification


def test_profit_contract_rejects_unknown_missing_and_invalid_fields() -> None:
    base_profit = {
        "ead_col": "ead",
        "pd_col": "pd_12m",
        "annual_rate": 0.12,
        "funding_rate": 0.03,
        "lgd": 0.5,
        "operating_cost_per_loan": 10,
        "term_months": 12,
    }
    cases = [
        {**base_profit, "invented": 1},
        {key: value for key, value in base_profit.items() if key != "lgd"},
        {**base_profit, "term_months": 0},
        {**base_profit, "lgd": "0.5"},
    ]
    for profit in cases:
        result = validate_strategy_request(
            {
                "operation": "develop",
                "strategy_type": "approval",
                "profit": profit,
            },
            allowed_columns=["ead", "pd_12m"],
        )
        assert result.draft is None
        assert "利润" in result.clarification


@pytest.mark.parametrize(
    ("strategy_type", "economics_inputs", "allowed_columns"),
    [
        ("limit", _limit_economics(), ["pd_12m", "utilization"]),
        (
            "pricing",
            _pricing_economics(),
            ["ead", "pd_12m"],
        ),
    ],
)
def test_limit_and_pricing_economics_match_backtest_contract(
    strategy_type: str,
    economics_inputs: dict,
    allowed_columns: list[str],
) -> None:
    result = validate_strategy_request(
        {
            "operation": "backtest",
            "strategy_type": strategy_type,
            "economics_inputs": economics_inputs,
        },
        allowed_columns=allowed_columns,
    )

    assert result.draft is not None
    assert result.draft.economics_inputs == {
        key: float(value) if key.endswith("_value") else value
        for key, value in economics_inputs.items()
    }
    assert f"{strategy_type}" not in result.confirmation
    assert "经济参数" in result.confirmation
    assert "指标由平台" in result.confirmation


@pytest.mark.parametrize("strategy_type", ["approval", "reject", "segmentation"])
def test_non_economic_strategy_types_reject_economics_inputs(
    strategy_type: str,
) -> None:
    result = validate_strategy_request(
        {
            "operation": "backtest",
            "strategy_type": strategy_type,
            "economics_inputs": _limit_economics(),
        },
        allowed_columns=["pd_12m", "utilization"],
    )

    assert result.draft is None
    assert "economics_inputs 只适用于额度或定价策略" in result.clarification


@pytest.mark.parametrize("strategy_type", ["limit", "pricing", "segmentation"])
def test_old_profit_contract_is_rejected_outside_approval_and_reject(
    strategy_type: str,
) -> None:
    result = validate_strategy_request(
        {
            "operation": "backtest",
            "strategy_type": strategy_type,
            "profit": {
                "ead_col": "ead",
                "pd_col": "pd_12m",
                "annual_rate": 0.18,
                "funding_rate": 0.04,
                "lgd": 0.5,
                "operating_cost_per_loan": 12,
                "term_months": 12,
            },
        },
        allowed_columns=["ead", "pd_12m"],
    )

    assert result.draft is None
    assert "profit 只适用于审批或拒绝策略" in result.clarification


@pytest.mark.parametrize(
    ("strategy_type", "economics_inputs", "message"),
    [
        ("limit", {**_limit_economics(), "invented": 1}, "不支持的字段"),
        (
            "limit",
            {key: value for key, value in _limit_economics().items() if key != "lgd_value"},
            "不完整",
        ),
        (
            "limit",
            {**_limit_economics(), "pd_value": 0.1},
            "二选一",
        ),
        (
            "limit",
            {**_limit_economics(), "lgd_value": True},
            "有限数字",
        ),
        (
            "limit",
            {**_limit_economics(), "utilization_col": "bad"},
            "不存在或不可用于策略",
        ),
        (
            "pricing",
            {**_pricing_economics(), "pd_value": 1.01, "pd_col": None},
            "二选一",
        ),
        (
            "pricing",
            {**_pricing_economics(), "term_months_value": 0},
            "大于 0",
        ),
        (
            "pricing",
            {**_pricing_economics(), "operating_cost_per_loan_value": -1},
            "大于等于 0",
        ),
    ],
)
def test_economics_inputs_fail_closed_on_unknown_missing_ambiguous_or_invalid_values(
    strategy_type: str,
    economics_inputs: dict,
    message: str,
) -> None:
    result = validate_strategy_request(
        {
            "operation": "backtest",
            "strategy_type": strategy_type,
            "economics_inputs": economics_inputs,
        },
        allowed_columns=["ead", "pd_12m", "utilization"],
    )

    assert result.draft is None
    assert message in result.clarification


def test_economics_column_references_are_limited_to_upstream_safe_whitelist() -> None:
    result = validate_strategy_request(
        {
            "operation": "backtest",
            "strategy_type": "limit",
            "economics_inputs": {
                **_limit_economics(),
                "pd_col": "bad",
            },
        },
        # The caller deliberately excludes the observed target from this list.
        allowed_columns=["utilization"],
    )

    assert result.draft is None
    assert "bad" in result.clarification
    assert "不可用于策略" in result.clarification


def test_explicit_llm_clarification_is_returned_without_repair() -> None:
    llm = _SequencedLLM({"clarification": "请说明要分析哪一个策略版本。"})

    result = compile_strategy_request(
        "帮我看看策略",
        allowed_columns=["score"],
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification == "请说明要分析哪一个策略版本。"
    assert len(llm.calls) == 1


def test_invalid_reply_gets_exactly_one_repair() -> None:
    llm = _SequencedLLM(
        {"operation": "backtest", "strategy_type": "approval", "ks": 0.5},
        {"operation": "backtest", "strategy_type": "approval"},
    )

    result = compile_strategy_request(
        "回测审批策略",
        allowed_columns=["score"],
        llm=llm,
    )

    assert result.draft is not None
    assert len(llm.calls) == 2
    assert "上一次输出未通过平台校验" in llm.calls[1]["user_prompt"]
    assert "ks" in llm.calls[1]["user_prompt"]
    assert all(call["temperature"] == 0.0 for call in llm.calls)
    assert all(call["json_schema"] == STRATEGY_REQUEST_JSON_SCHEMA for call in llm.calls)


def test_failed_repair_never_leaks_metric_results_as_a_draft() -> None:
    metric_reply = {
        "operation": "analyze",
        "strategy_type": "approval",
        "metrics": {"approval_rate": 0.7, "bad_rate": 0.04},
    }
    llm = _SequencedLLM(metric_reply, metric_reply, metric_reply)

    result = compile_strategy_request(
        "分析审批策略",
        allowed_columns=["score"],
        llm=llm,
    )

    assert result.draft is None
    assert "不支持的字段" in result.clarification
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    "metadata",
    [
        {"gini": 0.42},
        {"roi": 0.18},
        {"profit": 9000},
        {"lineage": {"model_ks": 0.9}},
    ],
)
def test_strategy_metadata_cannot_hide_llm_metric_results(metadata: dict) -> None:
    strategy_spec = _pricing_spec()
    strategy_spec["metadata"] = metadata

    result = validate_strategy_request(
        {
            "operation": "develop",
            "strategy_type": "pricing",
            "strategy_spec": strategy_spec,
        },
        allowed_columns=["score", "income"],
    )

    assert result.draft is None
    assert "metadata" in result.clarification
    assert "LLM 不得写入指标结果" in result.clarification


def test_validated_nested_draft_is_immutable_and_to_dict_is_defensive() -> None:
    result = validate_strategy_request(
        {
            "operation": "develop",
            "strategy_type": "pricing",
            "economics_inputs": _pricing_economics(),
            "strategy_spec": _pricing_spec(),
        },
        allowed_columns=["score", "income", "ead", "pd_12m"],
    )
    assert result.draft is not None

    with pytest.raises(TypeError):
        result.draft.economics_inputs["lgd_value"] = 2
    with pytest.raises(TypeError):
        result.draft.strategy_spec["rules"][0]["condition"]["args"][0][
            "field"
        ] = "ghost"

    detached = result.draft.to_dict()
    detached["economics_inputs"]["lgd_value"] = 2
    detached["strategy_spec"]["rules"][0]["condition"]["args"][0][
        "field"
    ] = "ghost"
    assert result.draft.economics_inputs["lgd_value"] == 0.5
    assert (
        result.draft.strategy_spec["rules"][0]["condition"]["args"][0]["field"]
        == "score"
    )


def test_load_json_object_wrappers_are_supported_and_transport_failure_clarifies() -> None:
    wrapped = _SequencedLLM(
        "模型解析如下：\n```json\n"
        '{"operation":"report","strategy_type":"segmentation"}\n```'
    )
    parsed = compile_strategy_request(
        "生成分群策略报告",
        allowed_columns=[],
        llm=wrapped,
    )
    assert parsed.draft is not None
    assert parsed.draft.operation == "report"

    broken = _SequencedLLM(RuntimeError("offline"))
    failed = compile_strategy_request(
        "生成报告",
        allowed_columns=[],
        llm=broken,
    )
    assert failed.draft is None
    assert "暂时无法解析" in failed.clarification
    assert len(broken.calls) == 1


def test_json_schema_closes_top_level_profit_and_economics_fields() -> None:
    schema = STRATEGY_REQUEST_JSON_SCHEMA["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["operation"]["enum"] == list(STRATEGY_OPERATIONS)
    assert schema["properties"]["strategy_type"]["enum"] == list(STRATEGY_TYPES)
    assert schema["properties"]["profit"]["additionalProperties"] is False
    economics = schema["properties"]["economics_inputs"]
    assert economics["additionalProperties"] is False
    assert len(economics["oneOf"]) == 2
    assert economics["properties"]["pd_value"]["maximum"] == 1
    assert economics["properties"]["term_months_value"]["exclusiveMinimum"] == 0
