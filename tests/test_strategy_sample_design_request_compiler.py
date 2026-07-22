"""Natural-language compiler contract for governed strategy sample design."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    compile_strategy_request,
    utterance_targets_strategy_sample_design,
    validate_strategy_request,
)


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


@pytest.mark.parametrize(
    "utterance",
    [
        "固化策略样本设计",
        "请创建样本设计",
        "把策略样本边界冻结下来",
        "materialize the strategy sample design",
    ],
)
def test_sample_design_command_router_accepts_only_direct_build_intent(utterance):
    assert utterance_targets_strategy_sample_design(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "基于已固化的样本设计，用 age、income 构建自动树候选",
        "使用策略样本设计做单变量分析",
        "参考样本设计，测算当前审批策略池影响",
        "不要固化样本设计，只构建自动树",
    ],
)
def test_sample_design_command_router_does_not_hijack_downstream_workflows(utterance):
    assert utterance_targets_strategy_sample_design(utterance) is False


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "target_bad_value": 1,
        "performance_window_status": "provided",
        "performance_window_days": 90,
        "observation_window_status": "provided",
        "observation_start": "2025-01-01",
        "observation_end": "2025-12-31",
        "maturity_status": "confirmed_matured",
    }
    inputs.update(overrides)
    if inputs.get("performance_window_status") == "unavailable":
        inputs.pop("performance_window_days", None)
    if inputs.get("observation_window_status") == "unavailable":
        inputs.pop("observation_start", None)
        inputs.pop("observation_end", None)
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design",
        "workflow_inputs": inputs,
    }


def test_sample_design_validates_full_user_contract_and_echoes_no_side_effects() -> None:
    payload = _payload(
        split_col="sample_role",
        development_values=["dev-a", "dev-b"],
        validation_values=["validation"],
        oot_values=["oot"],
        month_col="month",
        weight_col="weight",
        loan_amount_col="loan_amount",
        overdue_amount_col="overdue_amount",
        drop_nan_labels=True,
    )

    result = validate_strategy_request(
        payload,
        allowed_columns=[
            "sample_role",
            "month",
            "weight",
            "loan_amount",
            "overdue_amount",
        ],
        target_col="bad",
    )

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert "strategy_sample_design" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "strategy_sample_design"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert "策略样本设计 Workflow" in result.confirmation
    assert "表现窗：已提供 90 天" in result.confirmation
    assert "观察窗：2025-01-01 至 2025-12-31" in result.confirmation
    assert "成熟度：已确认成熟" in result.confirmation
    assert "坏样本值：1；好样本值：0" in result.confirmation
    assert "开发样本值 dev-a、dev-b" in result.confirmation
    assert "保留总体样本行，仅从好坏/风险分母排除空标签" in result.confirmation
    assert "不建模、不建树、不入池、不采纳、不部署" in result.confirmation


def test_sample_design_allows_explicit_exploration_only_downgrade() -> None:
    result = validate_strategy_request(
        _payload(
            performance_window_status="unavailable",
            observation_window_status="unavailable",
            maturity_status="unknown",
        ),
        allowed_columns=[],
        target_col="bad",
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert "performance_window_days" not in inputs
    assert "observation_start" not in inputs
    assert "observation_end" not in inputs
    assert "exploration-only" in result.confirmation
    assert "不能据此声称样本已成熟" in result.confirmation


def test_sample_design_missing_observation_window_is_also_exploration_only() -> None:
    result = validate_strategy_request(
        _payload(observation_window_status="unavailable"),
        allowed_columns=[],
        target_col="bad",
    )

    assert result.draft is not None
    assert "exploration-only" in result.confirmation


def test_sample_design_keeps_missing_validation_and_oot_explicitly_unavailable() -> None:
    payload = _payload(
        split_col="sample_role",
        development_values=["dev"],
        validation_values=[],
        oot_values=[],
    )
    validated = validate_strategy_request(
        payload,
        allowed_columns=["sample_role"],
        target_col="bad",
    )
    compiled = compile_strategy_request(
        "固化策略样本设计；表现窗 90 天；观察窗 2025-01-01 至 2025-12-31；"
        "成熟度已确认成熟；1 代表坏样本；切分列 sample_role；开发值 dev；"
        "验证值暂无；OOT 值暂无。",
        allowed_columns=["sample_role"],
        target_col="bad",
        llm=_FakeLLM(payload),
    )

    assert validated.draft is not None
    assert "验证样本值 unavailable" in validated.confirmation
    assert "OOT 样本值 unavailable" in validated.confirmation
    assert compiled.draft is not None


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({}, "缺少必需口径"),
        (
            {
                "performance_window_status": "provided",
                "observation_window_status": "unavailable",
                "maturity_status": "confirmed_matured",
            },
            "正整数",
        ),
        (
            {
                "performance_window_status": "unavailable",
                "performance_window_days": 90,
                "observation_window_status": "unavailable",
                "maturity_status": "unknown",
            },
            "不能填写 performance_window_days",
        ),
        (
            {
                "performance_window_status": "provided",
                "performance_window_days": 90,
                "observation_window_status": "provided",
                "observation_start": "2025-13-01",
                "observation_end": "2025-12-31",
                "maturity_status": "confirmed_matured",
            },
            "ISO 日期",
        ),
        (
            {
                "performance_window_status": "provided",
                "performance_window_days": 90,
                "observation_window_status": "provided",
                "observation_start": "2025-12-31",
                "observation_end": "2025-01-01",
                "maturity_status": "confirmed_matured",
            },
            "不能晚于",
        ),
        (
            {
                "performance_window_status": "provided",
                "performance_window_days": 90,
                "observation_window_status": "unavailable",
                "maturity_status": "confirmed_matured",
                "split_col": "sample_role",
            },
            "必须同时提供",
        ),
        (
            {
                "performance_window_status": "provided",
                "performance_window_days": 90,
                "observation_window_status": "unavailable",
                "maturity_status": "confirmed_matured",
                "split_col": "sample_role",
                "development_values": [1],
                "validation_values": [1.0],
                "oot_values": [2],
            },
            "互不重叠",
        ),
        (
            {
                "performance_window_status": "provided",
                "performance_window_days": 90,
                "observation_window_status": "unavailable",
                "maturity_status": "confirmed_matured",
                "month_col": "ghost",
            },
            "ghost",
        ),
        (
            {
                "performance_window_status": "provided",
                "performance_window_days": 90,
                "observation_window_status": "unavailable",
                "maturity_status": "confirmed_matured",
                "month_col": "month",
                "weight_col": "month",
            },
            "必须彼此不同",
        ),
        (
            {
                "performance_window_status": "provided",
                "performance_window_days": 90,
                "observation_window_status": "unavailable",
                "maturity_status": "confirmed_matured",
                "dataset_id": "llm-owned",
            },
            "不支持的字段",
        ),
    ],
)
def test_sample_design_rejects_incomplete_invalid_or_platform_owned_inputs(
    inputs: dict,
    message: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design",
            "workflow_inputs": {"target_bad_value": 1, **inputs},
        },
        allowed_columns=["sample_role", "month"],
        target_col="bad",
    )

    assert result.draft is None
    assert message in result.clarification


def test_sample_design_compiler_grounds_every_explicit_control() -> None:
    reply = _payload(
        split_col="sample_role",
        development_values=["dev-a", "dev-b"],
        validation_values=["validation"],
        oot_values=["oot"],
        month_col="month",
        weight_col="weight",
        loan_amount_col="loan_amount",
        overdue_amount_col="overdue_amount",
        drop_nan_labels=True,
    )
    llm = _FakeLLM(reply)

    result = compile_strategy_request(
        "固化策略样本设计；表现窗 90 天；观察窗 2025-01-01 至 2025-12-31；"
        "成熟度已确认成熟；1 代表坏样本；切分列 sample_role；开发值 dev-a、dev-b；"
        "验证值 validation；OOT 值 oot；月份列 month；权重列 weight；"
        "放款金额列 loan_amount；逾期金额列 overdue_amount；确认丢弃空标签。",
        allowed_columns=[
            "sample_role",
            "month",
            "weight",
            "loan_amount",
            "overdue_amount",
        ],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == reply
    assert len(llm.calls) == 1
    assert llm.calls[0]["prompt_version"] == 27
    assert "strategy_sample_design" in llm.calls[0]["system_prompt"]
    assert "平台字段" in llm.calls[0]["system_prompt"]


def test_sample_design_compiler_accepts_only_explicit_exploration_downgrade() -> None:
    reply = _payload(
        performance_window_status="unavailable",
        observation_window_status="unavailable",
        maturity_status="unknown",
    )

    accepted = compile_strategy_request(
        "先探索并固化策略样本设计；表现窗暂时没有；观察窗暂时没有；"
        "成熟度未知；1 代表坏样本。",
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(reply),
    )
    rejected = compile_strategy_request(
        "固化策略样本设计；1 代表坏样本。",
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(reply),
    )

    assert accepted.draft is not None
    assert rejected.draft is None
    assert rejected.clarification_code == "strategy_sample_design_controls_not_grounded"
    assert "exploration_only" in rejected.clarification_fields


def test_sample_design_compiler_marks_explicit_not_matured_as_exploration_only() -> None:
    result = compile_strategy_request(
        "先探索并固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
        "样本尚未成熟；1 代表坏样本。",
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                observation_window_status="unavailable",
                maturity_status="not_matured",
            )
        ),
    )

    assert result.draft is not None
    assert "exploration-only" in result.confirmation
    assert "尚未成熟" in result.confirmation


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            "能否固化策略样本设计？表现窗 90 天，观察窗暂时没有，成熟度已确认成熟。",
            "strategy_sample_design_positive_command_required",
        ),
        (
            "不要固化策略样本设计；表现窗 90 天，观察窗暂时没有，成熟度已确认成熟。",
            "strategy_sample_design_intent_negated",
        ),
        (
            "昨天固化样本设计时表现窗 90 天，观察窗暂时没有，成熟度已确认成熟。",
            "strategy_sample_design_positive_command_required",
        ),
        (
            "固化策略样本设计后训练模型；表现窗 90 天，观察窗暂时没有，成熟度已确认成熟。",
            "strategy_sample_design_single_step_required",
        ),
        (
            "固化策略样本设计，dataset_id=dataset-1；表现窗 90 天，观察窗暂时没有，成熟度已确认成熟。",
            "strategy_sample_design_platform_binding_forbidden",
        ),
    ],
)
def test_sample_design_compiler_rejects_noncommand_chains_and_platform_injection(
    utterance: str,
    code: str,
) -> None:
    result = compile_strategy_request(
        utterance + "；1 代表坏样本。",
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                observation_window_status="unavailable",
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code == code


def test_sample_design_compiler_rejects_omitted_explicit_optional_column() -> None:
    result = compile_strategy_request(
        "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
        "成熟度已确认成熟；1 代表坏样本；月份列 month。",
        allowed_columns=["month"],
        target_col="bad",
        llm=_FakeLLM(_payload(observation_window_status="unavailable")),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_controls_not_grounded"
    assert "month_col=month" in result.clarification_fields


def test_sample_design_compiler_rejects_omitted_explicit_nan_policy() -> None:
    result = compile_strategy_request(
        "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
        "成熟度已确认成熟；1 代表坏样本；确认丢弃空标签。",
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(_payload(observation_window_status="unavailable")),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_controls_not_grounded"
    assert "drop_nan_labels=true" in result.clarification_fields


@pytest.mark.parametrize("bad_value", [True, False, -1, 2, "1", 0.5, float("nan")])
def test_sample_design_requires_integer_binary_target_bad_value(
    bad_value: object,
) -> None:
    result = validate_strategy_request(
        _payload(target_bad_value=bad_value),
        allowed_columns=[],
        target_col="bad",
    )

    assert result.draft is None
    assert "target_bad_value" in result.clarification


def test_sample_design_normalizes_json_integral_target_bad_value() -> None:
    result = validate_strategy_request(
        _payload(target_bad_value=1.0),
        allowed_columns=[],
        target_col="bad",
    )
    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"]["target_bad_value"] == 1


def test_sample_design_requires_target_bad_value() -> None:
    payload = _payload()
    del payload["workflow_inputs"]["target_bad_value"]

    result = validate_strategy_request(
        payload,
        allowed_columns=[],
        target_col="bad",
    )

    assert result.draft is None
    assert "target_bad_value" in result.clarification


def test_sample_design_compiler_supports_explicit_reverse_target_encoding() -> None:
    result = compile_strategy_request(
        "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
        "成熟度已确认成熟；0 代表坏样本。",
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                target_bad_value=0,
                observation_window_status="unavailable",
            )
        ),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"]["target_bad_value"] == 0
    assert "坏样本值：0；好样本值：1" in result.confirmation


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
            "成熟度已确认成熟。"
        ),
        (
            "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
            "成熟度已确认成熟；0 代表坏样本；1 也代表坏样本。"
        ),
    ],
)
def test_sample_design_compiler_rejects_missing_or_conflicting_bad_semantics(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(_payload(observation_window_status="unavailable")),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_controls_not_grounded"
    assert "target_bad_value=1" in result.clarification_fields


@pytest.mark.parametrize(
    ("utterance", "reply"),
    [
        (
            "固化策略样本设计；表现窗不是 90 天而是 60 天；观察窗暂时没有；"
            "成熟度已确认成熟；1 代表坏样本。",
            _payload(observation_window_status="unavailable"),
        ),
        (
            "固化策略样本设计；表现窗不是 90 天；观察窗暂时没有；"
            "成熟度已确认成熟；1 代表坏样本。",
            _payload(observation_window_status="unavailable"),
        ),
        (
            "固化策略样本设计；表现窗 90 天；观察窗结束 2025-01-01、"
            "开始 2025-12-31；成熟度已确认成熟；1 代表坏样本。",
            _payload(),
        ),
        (
            "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
            "成熟度不是已成熟，而是未知；1 代表坏样本。",
            _payload(observation_window_status="unavailable"),
        ),
        (
            "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
            "成熟度未确认成熟；1 代表坏样本。",
            _payload(observation_window_status="unavailable"),
        ),
        (
            "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
            "成熟度已确认成熟；1 代表坏样本；切分列 sample_role；"
            "开发值 dev-a、dev-b；验证值 validation；OOT 值 oot。",
            _payload(
                observation_window_status="unavailable",
                split_col="sample_role",
                development_values=["dev-a"],
                validation_values=["validation"],
                oot_values=["oot"],
            ),
        ),
    ],
)
def test_sample_design_compiler_rejects_rewrites_swaps_conflicts_and_omissions(
    utterance: str,
    reply: dict,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=["sample_role"],
        target_col="bad",
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_controls_not_grounded"


@pytest.mark.parametrize(
    "operation",
    [
        "先筛选 age 大于 30 的样本",
        "仅保留已放款样本",
        "删除异常行",
        "先清洗样本",
        "派生列 risk_band",
        "按渠道过滤样本",
    ],
)
def test_sample_design_compiler_rejects_out_of_scope_data_mutations(
    operation: str,
) -> None:
    result = compile_strategy_request(
        "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
        f"成熟度已确认成熟；1 代表坏样本；{operation}。",
        allowed_columns=["age", "risk_band"],
        target_col="bad",
        llm=_FakeLLM(_payload(observation_window_status="unavailable")),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_single_step_required"


def test_sample_design_compiler_allows_explicitly_negated_downstream_operations() -> None:
    result = compile_strategy_request(
        "只固化策略样本设计，不建模、不要生成报告；表现窗 90 天；"
        "观察窗暂时没有；成熟度已确认成熟；1 代表坏样本。",
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(_payload(observation_window_status="unavailable")),
    )

    assert result.draft is not None


def test_sample_design_compiler_rejects_conflicting_explicit_good_semantics() -> None:
    result = compile_strategy_request(
        "固化策略样本设计；表现窗 90 天；观察窗暂时没有；"
        "成熟度已确认成熟；1 代表坏样本，1 也代表好样本。",
        allowed_columns=[],
        target_col="bad",
        llm=_FakeLLM(_payload(observation_window_status="unavailable")),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_controls_not_grounded"
    assert "target_bad_value=1" in result.clarification_fields
