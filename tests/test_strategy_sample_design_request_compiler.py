"""Fresh V2 and legacy-replay compiler contracts for strategy samples."""

from __future__ import annotations

from copy import deepcopy

import pytest

from marvis.agent.strategy_request_compiler import (
    FRESH_STANDARD_STRATEGY_WORKFLOWS,
    LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS,
    REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS,
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    _sample_v2_predicate_grounded,
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


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _v2_inputs() -> dict:
    return {
        "target_bad_value": 1,
        "drop_nan_labels": True,
        "approval_population": {"inclusion": None, "exclusion": None},
        "risk_population": {"inclusion": None, "exclusion": None},
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                "development": _eq("sample_role", "dev"),
                "validation": _eq("sample_role", "valid"),
                "oot": _eq("sample_role", "oot"),
            },
        },
        "maturity": {
            "status": "confirmed_matured",
            "performance_window_days": 30,
            "cutoff_date": "2026-04-30",
            "reason": None,
        },
        "performance_window": {"status": "provided", "days": 30},
        "observation_window": {
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-04-30",
        },
        "field_bindings": {
            "entity_field": "customer_id",
            "time_field": "apply_date",
            "group_field": None,
            "month_field": "apply_month",
            "weight_field": "weight",
            "loan_amount_field": "loan_amount",
            "overdue_amount_field": "overdue_amount",
        },
        "historical_score": {
            "status": "available",
            "column": "legacy_score",
            "direction": "higher_is_riskier",
            "reason": None,
        },
    }


def _v2_payload(inputs: dict | None = None) -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design_v2",
        "workflow_inputs": _v2_inputs() if inputs is None else inputs,
    }


def _legacy_payload() -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design",
        "workflow_inputs": {
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_start": "2026-01-01",
            "observation_end": "2026-04-30",
            "maturity_status": "confirmed_matured",
            "split_col": "sample_role",
            "development_values": ["dev"],
            "validation_values": ["valid"],
            "oot_values": ["oot"],
            "drop_nan_labels": True,
        },
    }


_COLUMNS = [
    "sample_role",
    "customer_id",
    "apply_date",
    "apply_month",
    "weight",
    "loan_amount",
    "overdue_amount",
    "legacy_score",
    "channel",
]


_GROUNDED_UTTERANCE = (
    "固化 V2 策略样本设计；1 代表坏样本；丢弃缺失标签；"
    "审批总体无纳排条件；风险总体无纳排条件；切分列 sample_role；"
    "开发值 dev；验证值 valid；OOT 值 oot；表现窗 30 天；"
    "观察窗 2026-01-01 至 2026-04-30；成熟度已确认成熟；"
    "成熟表现窗 30 天；成熟度截止日 2026-04-30；实体字段 customer_id；"
    "时间字段 apply_date；分组字段暂无；月份字段 apply_month；"
    "权重字段 weight；放款金额字段 loan_amount；逾期金额字段 overdue_amount；"
    "历史分 legacy_score，越高越风险。"
)


@pytest.mark.parametrize(
    "utterance",
    [
        "固化策略样本设计",
        "请创建样本设计",
        "把策略样本边界冻结下来",
        "materialize the strategy sample design",
    ],
)
def test_sample_design_command_router_accepts_direct_build_intent(utterance: str) -> None:
    assert utterance_targets_strategy_sample_design(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "基于已固化的样本设计，用 age 构建自动树候选",
        "使用策略样本设计做单变量分析",
        "参考样本设计，测算当前审批策略池影响",
        "不要固化样本设计，只构建自动树",
    ],
)
def test_sample_design_router_does_not_hijack_downstream_workflows(utterance: str) -> None:
    assert utterance_targets_strategy_sample_design(utterance) is False


def test_fresh_schema_exposes_v2_ids_and_hides_legacy_sample_id() -> None:
    schema_ids = STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]

    assert STANDARD_STRATEGY_WORKFLOWS == FRESH_STANDARD_STRATEGY_WORKFLOWS
    assert "strategy_sample_design_v2" in schema_ids
    assert "strategy_model_evidence_v2" in schema_ids
    assert "strategy_sample_design" not in schema_ids
    assert LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS == ("strategy_sample_design",)
    assert set(REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS) == {
        *FRESH_STANDARD_STRATEGY_WORKFLOWS,
        "strategy_sample_design",
    }


def test_legacy_sample_request_is_rejected_by_default_and_allowed_only_for_replay() -> None:
    rejected = validate_strategy_request(
        _legacy_payload(),
        allowed_columns=_COLUMNS,
        target_col="bad",
    )
    replayed = validate_strategy_request(
        _legacy_payload(),
        allowed_columns=_COLUMNS,
        target_col="bad",
        allow_legacy_replay=True,
    )

    assert rejected.draft is None
    assert "strategy_sample_design_v2" in rejected.clarification
    assert replayed.draft is not None
    assert replayed.draft.to_dict() == _legacy_payload()


def test_v2_sample_validates_and_compiler_grounds_every_user_control() -> None:
    payload = _v2_payload()
    validated = validate_strategy_request(
        payload,
        allowed_columns=_COLUMNS,
        target_col="bad",
    )
    llm = _FakeLLM(payload)
    compiled = compile_strategy_request(
        _GROUNDED_UTTERANCE,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=llm,
    )

    assert validated.draft is not None
    assert validated.draft.to_dict() == payload
    assert "V2 双总体策略样本设计" in validated.confirmation
    assert "两者均无纳排" in validated.confirmation
    assert compiled.draft is not None
    assert compiled.draft.to_dict() == payload
    assert len(llm.calls) == 1
    assert llm.calls[0]["prompt_version"] == 43
    assert "strategy_sample_design_v2" in llm.calls[0]["system_prompt"]
    assert "strategy_model_evidence_v2" in llm.calls[0]["system_prompt"]


def test_v2_sample_confirmation_echoes_historical_score_binding() -> None:
    result = validate_strategy_request(
        _v2_payload(),
        allowed_columns=_COLUMNS,
        target_col="bad",
    )

    assert result.draft is not None
    assert (
        "历史分：available；字段：legacy_score；方向：higher_is_riskier"
        in result.confirmation
    )


@pytest.mark.parametrize(
    "field",
    [
        "legacy_sample_design_ref",
        "relationship",
        "scope",
        "policy",
        "dataset_id",
        "workspace_revision",
        "artifact_id",
        "content_hash",
    ],
)
def test_v2_sample_rejects_every_platform_owned_input(field: str) -> None:
    inputs = _v2_inputs()
    inputs[field] = "injected"

    result = validate_strategy_request(
        _v2_payload(inputs),
        allowed_columns=_COLUMNS,
        target_col="bad",
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_platform_binding_forbidden"
    assert field in result.clarification_fields


def test_v2_sample_rejects_platform_identity_in_the_utterance() -> None:
    result = compile_strategy_request(
        _GROUNDED_UTTERANCE + " dataset_id=dataset-1。",
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_platform_binding_forbidden"


def test_v2_sample_rejects_ungrounded_field_binding() -> None:
    payload = _v2_payload()
    payload["workflow_inputs"]["field_bindings"]["group_field"] = "channel"

    result = compile_strategy_request(
        _GROUNDED_UTTERANCE,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(payload),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_controls_not_grounded"
    assert "field_bindings.group_field" in result.clarification_fields


def test_v2_sample_requires_explicit_null_population_and_field_controls() -> None:
    vague = _GROUNDED_UTTERANCE.replace("审批总体无纳排条件；", "").replace("分组字段暂无；", "")

    result = compile_strategy_request(
        vague,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_controls_not_grounded"
    assert "approval_population" in result.clarification_fields
    assert "field_bindings.group_field" in result.clarification_fields


def test_v2_sample_time_ranges_returns_typed_native_bootstrap_clarification() -> None:
    inputs = _v2_inputs()
    inputs["partitioning"] = {
        "method": "time_ranges",
        "column": "apply_date",
        "ranges": {
            "development": {"start": "2026-01-01", "end": "2026-02-28"},
            "validation": {"start": "2026-03-01", "end": "2026-03-31"},
            "oot": {"start": "2026-04-01", "end": "2026-04-30"},
        },
    }

    result = validate_strategy_request(
        _v2_payload(inputs),
        allowed_columns=_COLUMNS,
        target_col="bad",
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_native_bootstrap_required"
    assert result.clarification_fields == ("partitioning.method",)


@pytest.mark.parametrize(
    ("population", "role_text"),
    [
        ("approval_population", "审批总体纳入 channel 等于 app。"),
        ("risk_population", "风险总体纳入 channel 等于 app。"),
    ],
)
def test_v2_sample_any_population_filter_requires_native_bootstrap_without_repair(
    population: str,
    role_text: str,
) -> None:
    payload = _v2_payload()
    payload["workflow_inputs"][population]["inclusion"] = _eq("channel", "app")
    llm = _FakeLLM(payload)

    result = compile_strategy_request(
        _GROUNDED_UTTERANCE + role_text,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_native_bootstrap_required"
    assert result.clarification_fields == (population,)
    assert len(llm.calls) == 1


def test_v2_sample_no_filter_grounding_cannot_cross_population_roles() -> None:
    utterance = _GROUNDED_UTTERANCE.replace(
        "风险总体无纳排条件；",
        "风险总体纳入 channel 等于 app；",
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_controls_not_grounded"
    assert "risk_population" in result.clarification_fields


def test_v2_sample_unqualified_no_condition_text_is_bound_to_each_role() -> None:
    utterance = _GROUNDED_UTTERANCE.replace(
        "审批总体无纳排条件；风险总体无纳排条件；",
        "审批总体无条件；风险总体无条件；",
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is not None


def test_v2_population_predicate_grounding_cannot_swap_inclusion_and_exclusion() -> None:
    predicate = _eq("channel", "app")
    utterance = "审批总体排除 channel 等于 app。"

    assert _sample_v2_predicate_grounded(
        utterance,
        predicate,
        role_labels=("审批总体", "审批样本", "approval population"),
        direction="exclusion",
    )
    assert not _sample_v2_predicate_grounded(
        utterance,
        predicate,
        role_labels=("审批总体", "审批样本", "approval population"),
        direction="inclusion",
    )


def test_v2_population_predicate_grounding_rejects_operator_rewrite() -> None:
    predicate = _eq("channel", "app")

    assert not _sample_v2_predicate_grounded(
        "审批总体纳入 channel 不等于 app。",
        predicate,
        role_labels=("审批总体", "审批样本", "approval population"),
        direction="inclusion",
    )


def test_v2_sample_partition_operator_cannot_be_rewritten() -> None:
    utterance = _GROUNDED_UTTERANCE.replace(
        "开发值 dev；",
        "开发值不等于 dev；",
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_controls_not_grounded"
    assert "partitioning.selectors.development" in result.clarification_fields


def test_v2_sample_maturity_cutoff_cannot_borrow_observation_end_date() -> None:
    utterance = _GROUNDED_UTTERANCE.replace(
        "成熟表现窗 30 天；成熟度截止日 2026-04-30；",
        "成熟表现窗 30 天；",
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_controls_not_grounded"
    assert "maturity.cutoff_date" in result.clarification_fields


def test_v2_sample_maturity_cutoff_requires_maturity_qualifier() -> None:
    utterance = _GROUNDED_UTTERANCE.replace(
        "成熟度截止日 2026-04-30；",
        "数据截止日 2026-04-30；",
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_controls_not_grounded"
    assert "maturity.cutoff_date" in result.clarification_fields


def test_v2_sample_performance_window_cannot_borrow_maturity_window() -> None:
    utterance = _GROUNDED_UTTERANCE.replace("表现窗 30 天；", "", 1)

    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_controls_not_grounded"
    assert "performance_window" in result.clarification_fields


def test_v2_sample_historical_direction_cannot_borrow_another_field_direction() -> None:
    utterance = _GROUNDED_UTTERANCE.replace(
        "历史分 legacy_score，越高越风险。",
        "历史分 legacy_score，越低越风险；channel 越高越风险。",
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        target_col="bad",
        llm=_FakeLLM(_v2_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_controls_not_grounded"
    assert "historical_score" in result.clarification_fields


@pytest.mark.parametrize(
    "partitioning",
    [
        {
            "method": "predicate_ast",
            "selectors": {
                "development": _eq("sample_role", "dev"),
                "validation": _eq("channel", "valid"),
                "oot": _eq("sample_role", "oot"),
            },
        },
        {
            "method": "predicate_ast",
            "selectors": {
                "development": {
                    "op": "and",
                    "args": [_eq("sample_role", "dev"), _eq("channel", "app")],
                },
                "validation": _eq("sample_role", "valid"),
                "oot": _eq("sample_role", "oot"),
            },
        },
    ],
)
def test_v2_sample_requires_lossless_simple_same_column_partitioning(partitioning: dict) -> None:
    inputs = _v2_inputs()
    inputs["partitioning"] = partitioning

    result = validate_strategy_request(
        _v2_payload(inputs),
        allowed_columns=_COLUMNS,
        target_col="bad",
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_sample_design_v2_native_bootstrap_required"


def test_model_evidence_v2_fresh_inputs_are_exactly_empty() -> None:
    accepted = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_model_evidence_v2",
            "workflow_inputs": {},
        },
        allowed_columns=_COLUMNS,
    )
    rejected = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_model_evidence_v2",
            "workflow_inputs": {"artifact_id": "a" * 64},
        },
        allowed_columns=_COLUMNS,
    )

    assert accepted.draft is not None
    assert accepted.draft.to_dict()["workflow_inputs"] == {}
    assert "当前 task" in accepted.confirmation
    assert rejected.draft is None
    assert rejected.clarification_code == "strategy_model_evidence_v2_platform_binding_forbidden"


@pytest.mark.parametrize(
    "utterance",
    [
        "汇总已有认证单变量候选为模型证据，然后训练模型",
        "生成 ModelEvidence 并比较模型",
        "汇总认证单变量证据并生成月度 OOT 报告",
        "生成模型证据后部署上线",
    ],
)
def test_model_evidence_v2_rejects_unsupported_chains(utterance: str) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        llm=_FakeLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_model_evidence_v2",
                "workflow_inputs": {},
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_model_evidence_v2_univariate_only"


def test_model_evidence_v2_compiles_only_existing_authenticated_univariate_summary() -> None:
    payload = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_model_evidence_v2",
        "workflow_inputs": {},
    }
    llm = _FakeLLM(payload)

    result = compile_strategy_request(
        "汇总当前任务已有认证单变量候选证据为 ModelEvidence V2。",
        allowed_columns=_COLUMNS,
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert llm.calls[0]["prompt_version"] == 43


@pytest.mark.parametrize(
    "utterance",
    [
        "汇总此前已有已认证单变量候选证据为 ModelEvidence V2。",
        (
            "只归集此前已有已认证单变量候选证据为 ModelEvidence V2，"
            "不训练模型、不比较模型、不生成报告、不采纳、不部署。"
        ),
    ],
)
def test_model_evidence_v2_accepts_source_history_and_explicitly_negated_chains(
    utterance: str,
) -> None:
    payload = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_model_evidence_v2",
        "workflow_inputs": {},
    }

    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        llm=_FakeLLM(payload),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == payload


def test_model_evidence_v2_accepts_extended_negated_downstream_actions() -> None:
    payload = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_model_evidence_v2",
        "workflow_inputs": {},
    }

    result = compile_strategy_request(
        "只归集已有认证单变量候选证据为 ModelEvidence V2，"
        "不需要训练模型，不用比较模型，暂不生成报告，先不部署。",
        allowed_columns=_COLUMNS,
        llm=_FakeLLM(payload),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == payload


def test_model_evidence_v2_rejects_a_truly_historical_aggregation_action() -> None:
    result = compile_strategy_request(
        "此前已汇总已有认证单变量候选证据为 ModelEvidence V2。",
        allowed_columns=_COLUMNS,
        llm=_FakeLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_model_evidence_v2",
                "workflow_inputs": {},
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_model_evidence_v2_positive_command_required"


@pytest.mark.parametrize(
    "utterance",
    [
        "此前未汇总已有认证单变量候选证据为 ModelEvidence V2。",
        "此前没有汇总已有认证单变量候选证据为 ModelEvidence V2。",
        "当前没有汇总已有认证单变量候选证据为 ModelEvidence V2。",
        "目前有没有汇总已有认证单变量候选证据为 ModelEvidence V2",
    ],
)
def test_model_evidence_v2_rejects_negated_history_and_unmarked_questions(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=_COLUMNS,
        llm=_FakeLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_model_evidence_v2",
                "workflow_inputs": {},
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_model_evidence_v2_positive_command_required"


def test_v2_sample_payload_copy_is_not_mutated_by_validation() -> None:
    payload = _v2_payload()
    before = deepcopy(payload)

    validate_strategy_request(payload, allowed_columns=_COLUMNS, target_col="bad")

    assert payload == before
