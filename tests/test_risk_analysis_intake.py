from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import marvis.agent.risk_analysis_setup as risk_setup
import pandas as pd
from fastapi.testclient import TestClient

from marvis.agent.turn_handlers import _maybe_handle_adhoc_turn
from marvis.app import create_app
from marvis.domain import TASK_TYPE_VINTAGE


@dataclass(frozen=True)
class _Dataset:
    id: str
    source_path: str
    columns: tuple = ()
    role: str = "sample"
    row_count: int = 1
    has_target: bool = False


class _Registry:
    def __init__(self, datasets=()):
        self.datasets = list(datasets)
        self.list_calls = 0

    def list_for_task(self, task_id):
        self.list_calls += 1
        return list(self.datasets)

    def resolve_path(self, dataset_id):
        return Path(f"/{dataset_id}.parquet")


class _Backend:
    def __init__(self, columns_by_dataset=None):
        self.columns_by_dataset = dict(columns_by_dataset or {})

    def column_names(self, path):
        return list(self.columns_by_dataset[path.stem])


def _assistant_state(
    phase: str,
    analysis_kind: str | None = None,
    *,
    analysis_scope: str | None = None,
) -> dict:
    state = {"phase": phase}
    if analysis_kind is not None:
        state["analysis_kind"] = analysis_kind
    if analysis_scope is not None:
        state["analysis_scope"] = analysis_scope
    return {
        "role": "assistant",
        "metadata": {risk_setup.RISK_ANALYSIS_INTAKE_META_KEY: state},
    }


def _advance(registry, backend, *, user_text=None, conversation=()):
    return risk_setup.advance_risk_analysis_setup(
        registry,
        backend,
        "task-risk",
        None,
        user_text=user_text,
        conversation=list(conversation),
    )


def test_first_turn_asks_goal_before_inspecting_materials():
    registry = _Registry()

    result = _advance(registry, _Backend())

    assert result.template_id is None
    assert "想分析什么" in result.content
    assert "VTG终值与年化不良" in result.content
    assert "收益测算" in result.content
    assert "标准 Vintage" in result.content
    assert result.intake_state == {"phase": "ask_goal"}
    assert registry.list_calls == 0


def test_vtg_goal_moves_to_material_request_with_explicit_contract():
    registry = _Registry()
    conversation = [
        _assistant_state("ask_goal"),
        # _run_driver_turn has already persisted the current user message. It
        # must not hide the preceding assistant state.
        {"role": "user", "content": "做VTG终值与年化不良"},
    ]

    result = _advance(
        registry,
        _Backend(),
        user_text="做VTG终值与年化不良",
        conversation=conversation,
    )

    assert result.intake_state == {
        "phase": "request_materials",
        "analysis_kind": "vtg_terminal",
        "analysis_scope": "做VTG终值与年化不良",
    }
    assert "canonical VTG input" in result.content
    assert "每个 product × cohort 一行" in result.content
    assert "product × cohort × MOB" in result.content
    for column in (
        "product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "turnover",
        "mob_days",
        "day_count_basis",
        "mob_balance_rate",
        "mob_balance_amount",
        "terminal_bad_rate",
        "long_term_recovery_rate",
        "auxiliary_terminal_bad_rate",
    ):
        assert column in result.content
    assert "百分比用小数" in result.content
    assert "年周转" in result.content
    assert "day_count_basis" in result.content
    assert "selection_rule=min_auxiliary_recovery" in result.content
    assert "一个 as_of_date 和一个 scenario" in result.content
    assert "不会由系统猜测" in result.content
    assert registry.list_calls == 0


def test_profitability_goal_lists_row_units_weights_and_tax_basis():
    result = _advance(
        _Registry(),
        _Backend(),
        user_text="我要做收益测算",
        conversation=[_assistant_state("ask_goal")],
    )

    assert result.intake_state["analysis_kind"] == "profitability"
    assert result.intake_state["analysis_scope"] == "我要做收益测算"
    assert "表：canonical economics" in result.content
    assert "product × as_of_period × scenario × asset_class" in result.content
    for column in (
        "weight",
        "weight_basis",
        "customer_rate",
        "risk_cost_rate",
        "funding_cost_rate",
        "interest_loss_rate",
        "other_cost_rate",
    ):
        assert column in result.content
    assert "每行" in result.content
    assert "权重" in result.content
    assert "税费" in result.content
    assert "不能直接填合同分润比例" in result.content
    assert "独立获客成本" in result.content
    assert "显式填 0" in result.content
    assert "不会把缺失成本静默当成 0" in result.content


def test_no_uploaded_data_waits_in_chat_instead_of_raising():
    result = _advance(
        _Registry(),
        _Backend(),
        user_text="材料已上传",
        conversation=[_assistant_state("request_materials", "vtg_terminal")],
    )

    assert result.template_id is None
    assert result.intake_state == {
        "phase": "await_materials",
        "analysis_kind": "vtg_terminal",
    }
    assert "还没有检测到已登记的数据表" in result.content


def test_confirmed_analysis_scope_survives_material_wait():
    result = _advance(
        _Registry(),
        _Backend(),
        user_text="材料已上传",
        conversation=[
            _assistant_state(
                "request_materials",
                "profitability",
                analysis_scope="白条 2025-12 基准与压力方案收益测算",
            )
        ],
    )

    assert (
        result.intake_state["analysis_scope"] == "白条 2025-12 基准与压力方案收益测算"
    )


def test_uploaded_filename_cannot_silently_switch_selected_analysis_kind():
    result = _advance(
        _Registry(),
        _Backend(),
        user_text="已上传材料：白条收益测算参考.xlsx，请检查后继续。",
        conversation=[
            _assistant_state(
                "request_materials",
                "vtg_terminal",
                analysis_scope="做VTG终值与年化不良",
            )
        ],
    )

    assert result.intake_state["analysis_kind"] == "vtg_terminal"
    assert result.intake_state["analysis_scope"] == "做VTG终值与年化不良"


def test_selected_analysis_kind_changes_only_on_explicit_switch_intent():
    result = _advance(
        _Registry(),
        _Backend(),
        user_text="改做收益测算，比较基准与压力方案",
        conversation=[_assistant_state("request_materials", "vtg_terminal")],
    )

    assert result.intake_state == {
        "phase": "request_materials",
        "analysis_kind": "profitability",
        "analysis_scope": "改做收益测算，比较基准与压力方案",
    }


def test_vtg_missing_columns_are_named_and_plan_is_not_started():
    dataset = _Dataset("vtg", "task-risk/vtg.parquet")
    result = _advance(
        _Registry([dataset]),
        _Backend(
            {
                "vtg": [
                    "product",
                    "as_of_date",
                    "cohort",
                    "amount_unit",
                    "disbursement_amount",
                    "mob14_bad_rate",
                ]
            }
        ),
        user_text="检查材料",
        conversation=[_assistant_state("await_materials", "vtg_terminal")],
    )

    assert result.template_id is None
    assert result.intake_state["phase"] == "await_materials"
    assert result.intake_state["missing_columns"] == [
        "terminal_bad_rate 或 long_term_recovery_rate 或 auxiliary_terminal_bad_rate（至少一个）",
        "turnover 或 (mob + mob_days + day_count_basis + mob_balance_rate) 或 "
        "(mob + mob_days + day_count_basis + mob_balance_amount)（满足一组）",
    ]
    assert "turnover" in result.content
    assert "terminal_bad_rate 或 long_term_recovery_rate" in result.content


def test_vtg_complete_contract_returns_report_template_and_column_map():
    dataset = _Dataset("vtg", "task-risk/vtg.parquet")
    columns = [
        "Product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "turnover",
        "long_term_recovery_rate",
        "previous_annualized_bad_rate",
    ]
    result = _advance(
        _Registry([dataset]),
        _Backend({"vtg": columns}),
        user_text="开始分析",
        conversation=[_assistant_state("request_materials", "vtg_terminal")],
    )

    assert result.template_id == "risk_analysis_report"
    assert result.slots == {
        "analysis_kind": "vtg_terminal",
        "dataset_id": "vtg",
        "column_map": {
            "product": "Product",
            "as_of_date": "as_of_date",
            "cohort": "cohort",
            "amount_unit": "amount_unit",
            "disbursement_amount": "disbursement_amount",
            "mob14_bad_rate": "mob14_bad_rate",
            "turnover": "turnover",
            "long_term_recovery_rate": "long_term_recovery_rate",
            "previous_annualized_bad_rate": "previous_annualized_bad_rate",
        },
    }
    assert result.intake_state["phase"] == "ready"


def test_normalized_column_name_collision_blocks_plan_instead_of_guessing():
    dataset = _Dataset("vtg", "task-risk/vtg.parquet")
    columns = [
        "product",
        "Product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "turnover",
        "terminal_bad_rate",
    ]

    result = _advance(
        _Registry([dataset]),
        _Backend({"vtg": columns}),
        user_text="材料已上传",
        conversation=[_assistant_state("await_materials", "vtg_terminal")],
    )

    assert result.template_id is None
    assert result.intake_state["phase"] == "await_materials"
    assert result.intake_state["missing_columns"] == [
        "product（归一化后匹配多个源列）",
    ]
    assert result.intake_state["ambiguous_columns"] == [
        {
            "canonical": "product",
            "source_columns": ["product", "Product"],
        }
    ]
    assert "字段名冲突" in result.content
    assert "product 匹配到 product, Product" in result.content


def test_equal_contract_reupload_selects_latest_registered_dataset():
    old = _Dataset("vtg_old", "task-risk/vtg_old.parquet", row_count=2)
    corrected = _Dataset(
        "vtg_corrected", "task-risk/vtg_corrected.parquet", row_count=2
    )
    columns = [
        "product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "turnover",
        "terminal_bad_rate",
    ]

    result = _advance(
        _Registry([old, corrected]),
        _Backend({"vtg_old": columns, "vtg_corrected": columns}),
        user_text="请用我刚重新上传的修正版",
        conversation=[_assistant_state("await_materials", "vtg_terminal")],
    )

    assert result.template_id == "risk_analysis_report"
    assert result.slots["dataset_id"] == "vtg_corrected"


def test_complete_corrected_reupload_with_fewer_rows_still_selects_latest():
    old = _Dataset("vtg_old", "task-risk/vtg_old.parquet", row_count=100)
    corrected = _Dataset(
        "vtg_corrected", "task-risk/vtg_corrected.parquet", row_count=2
    )
    columns = [
        "product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "turnover",
        "terminal_bad_rate",
    ]

    result = _advance(
        _Registry([old, corrected]),
        _Backend({"vtg_old": columns, "vtg_corrected": columns}),
        user_text="请用我刚重新上传的修正版",
        conversation=[_assistant_state("await_materials", "vtg_terminal")],
    )

    assert result.template_id == "risk_analysis_report"
    assert result.slots["dataset_id"] == "vtg_corrected"


def test_complete_corrected_reupload_with_fewer_optional_columns_selects_latest():
    old = _Dataset("vtg_old", "task-risk/vtg_old.parquet", row_count=2)
    corrected = _Dataset(
        "vtg_corrected", "task-risk/vtg_corrected.parquet", row_count=2
    )
    required_columns = [
        "product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "turnover",
        "terminal_bad_rate",
    ]
    old_columns = [*required_columns, "previous_annualized_bad_rate", "channel"]

    result = _advance(
        _Registry([old, corrected]),
        _Backend({"vtg_old": old_columns, "vtg_corrected": required_columns}),
        user_text="请用我刚重新上传的修正版",
        conversation=[_assistant_state("await_materials", "vtg_terminal")],
    )

    assert result.template_id == "risk_analysis_report"
    assert result.slots["dataset_id"] == "vtg_corrected"


def test_vtg_raw_balance_curve_contract_can_replace_precomputed_turnover():
    dataset = _Dataset("vtg_curve", "task-risk/vtg_curve.parquet")
    columns = [
        "product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "long_term_recovery_rate",
        "mob",
        "mob_days",
        "day_count_basis",
        "mob_balance_rate",
    ]

    result = _advance(
        _Registry([dataset]),
        _Backend({"vtg_curve": columns}),
        user_text="材料已上传",
        conversation=[_assistant_state("request_materials", "vtg_terminal")],
    )

    assert result.template_id == "risk_analysis_report"
    assert "turnover" not in result.slots["column_map"]
    assert result.slots["column_map"]["mob_balance_rate"] == "mob_balance_rate"
    assert result.slots["column_map"]["mob_days"] == "mob_days"


def test_vtg_auxiliary_terminal_requires_explicit_terminal_method():
    dataset = _Dataset("vtg_aux", "task-risk/vtg_aux.parquet")
    base_columns = [
        "product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "turnover",
        "auxiliary_terminal_bad_rate",
    ]
    missing_method = _advance(
        _Registry([dataset]),
        _Backend({"vtg_aux": base_columns}),
        user_text="材料已上传",
        conversation=[_assistant_state("request_materials", "vtg_terminal")],
    )
    assert missing_method.template_id is None
    assert any(
        "terminal_method" in item
        for item in missing_method.intake_state["missing_columns"]
    )

    ready = _advance(
        _Registry([dataset]),
        _Backend({"vtg_aux": [*base_columns, "terminal_method"]}),
        user_text="已补口径",
        conversation=[_assistant_state("await_materials", "vtg_terminal")],
    )
    assert ready.template_id == "risk_analysis_report"


def test_vtg_recovery_and_auxiliary_columns_require_explicit_selection_rule():
    dataset = _Dataset("vtg_guardrail", "task-risk/vtg_guardrail.parquet")
    columns = [
        "product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
        "turnover",
        "long_term_recovery_rate",
        "auxiliary_terminal_bad_rate",
    ]

    missing_rule = _advance(
        _Registry([dataset]),
        _Backend({"vtg_guardrail": columns}),
        user_text="材料已上传",
        conversation=[_assistant_state("request_materials", "vtg_terminal")],
    )

    assert missing_rule.template_id is None
    assert any(
        "selection_rule" in item
        for item in missing_rule.intake_state["missing_columns"]
    )

    ready = _advance(
        _Registry([dataset]),
        _Backend({"vtg_guardrail": [*columns, "selection_rule"]}),
        user_text="已补充选择规则",
        conversation=[_assistant_state("await_materials", "vtg_terminal")],
    )
    assert ready.template_id == "risk_analysis_report"


def test_vtg_real_business_summary_headers_map_to_canonical_contract():
    dataset = _Dataset("vtg_cn", "task-risk/vtg_cn.parquet")
    columns = [
        "产品名称",
        "截面日",
        "放款月份",
        "金额单位",
        "放款额",
        "MOB14 vtg30+",
        "vtg终值",
        "周转次数",
        "月日均余额",
    ]

    result = _advance(
        _Registry([dataset]),
        _Backend({"vtg_cn": columns}),
        user_text="材料已上传",
        conversation=[_assistant_state("request_materials", "vtg_terminal")],
    )

    assert result.template_id == "risk_analysis_report"
    assert result.slots["column_map"] == {
        "product": "产品名称",
        "as_of_date": "截面日",
        "cohort": "放款月份",
        "amount_unit": "金额单位",
        "disbursement_amount": "放款额",
        "mob14_bad_rate": "MOB14 vtg30+",
        "turnover": "周转次数",
        "terminal_bad_rate": "vtg终值",
        "avg_daily_balance": "月日均余额",
    }


def test_profitability_complete_contract_returns_report_template():
    dataset = _Dataset("econ", "task-risk/economics.parquet")
    required = [
        "product",
        "as_of_period",
        "asset_class",
        "weight",
        "weight_basis",
        "customer_rate",
        "interest_loss_rate",
        "revenue_share_rate",
        "risk_cost_rate",
        "acquisition_cost_rate",
        "data_cost_rate",
        "payment_cost_rate",
        "collection_cost_rate",
        "funding_cost_rate",
        "other_cost_rate",
        "tax_rate",
    ]
    result = _advance(
        _Registry([dataset]),
        _Backend({"econ": required}),
        user_text="材料齐了",
        conversation=[_assistant_state("request_materials", "profitability")],
    )

    assert result.template_id == "risk_analysis_report"
    assert result.slots["analysis_kind"] == "profitability"
    assert result.slots["dataset_id"] == "econ"
    assert result.slots["column_map"] == {column: column for column in required}


def test_profitability_does_not_treat_raw_share_or_tax_labels_as_asset_yield_costs():
    dataset = _Dataset("econ_raw_labels", "task-risk/economics.parquet")
    columns = [
        "product",
        "as_of_period",
        "asset_class",
        "weight",
        "weight_basis",
        "customer_rate",
        "interest_loss_rate",
        "分润率",
        "risk_cost_rate",
        "acquisition_cost_rate",
        "data_cost_rate",
        "payment_cost_rate",
        "collection_cost_rate",
        "funding_cost_rate",
        "other_cost_rate",
        "税率",
    ]

    result = _advance(
        _Registry([dataset]),
        _Backend({"econ_raw_labels": columns}),
        user_text="材料已上传",
        conversation=[_assistant_state("request_materials", "profitability")],
    )

    assert result.template_id is None
    assert any(
        "revenue_share_rate" in item for item in result.intake_state["missing_columns"]
    )
    assert any("tax_rate" in item for item in result.intake_state["missing_columns"])


def test_profitability_raw_driver_groups_satisfy_material_contract():
    dataset = _Dataset("econ_drivers", "task-risk/economics.parquet")
    columns = [
        "product",
        "as_of_period",
        "asset_class",
        "weight",
        "weight_basis",
        "customer_rate",
        "acquisition_cost_rate",
        "payment_cost_rate",
        "collection_cost_rate",
        "funding_cost_rate",
        "other_cost_rate",
        "terminal_vintage_rate",
        "risk_turnover",
        "loss_timing_factor",
        "profit_share_ratio",
        "per_application_cost",
        "credit_approval_rate",
        "draw_initiation_rate",
        "draw_approval_rate",
        "average_ticket",
        "data_annualization_factor",
        "amount_unit",
        "tax_method",
        "tax_inclusive_divisor",
        "tax_combined_rate",
    ]

    result = _advance(
        _Registry([dataset]),
        _Backend({"econ_drivers": columns}),
        user_text="材料已上传",
        conversation=[_assistant_state("request_materials", "profitability")],
    )

    assert result.template_id == "risk_analysis_report"
    assert result.slots["column_map"] == {column: column for column in columns}


def test_profitability_raw_data_cost_requires_amount_unit():
    dataset = _Dataset("econ_drivers", "task-risk/economics.parquet")
    columns = [
        "product",
        "as_of_period",
        "asset_class",
        "weight",
        "weight_basis",
        "customer_rate",
        "revenue_share_rate",
        "payment_cost_rate",
        "collection_cost_rate",
        "funding_cost_rate",
        "other_cost_rate",
        "risk_cost_rate",
        "interest_loss_rate",
        "acquisition_cost_rate",
        "per_application_cost",
        "credit_approval_rate",
        "draw_initiation_rate",
        "draw_approval_rate",
        "average_ticket",
        "data_annualization_factor",
        "customer_stage",
        "tax_rate",
    ]

    result = _advance(
        _Registry([dataset]),
        _Backend({"econ_drivers": columns}),
        user_text="材料已上传",
        conversation=[_assistant_state("request_materials", "profitability")],
    )

    assert result.template_id is None
    assert any(
        "数据成本" in item and "amount_unit" in item
        for item in result.intake_state["missing_columns"]
    )
    assert any(
        "transaction_weight" in item for item in result.intake_state["missing_columns"]
    )


def test_standard_vintage_waits_for_material_stage_then_reuses_existing_proposal(
    monkeypatch,
):
    class _Proposal:
        template_id = "vintage_analysis"
        dataset_id = "panel"
        dataset_name = "panel.parquet"
        cohort_col = "cohort"
        mob_col = "mob"
        bad_col = "bad"

        def template_slots(self):
            return {
                "dataset_id": "panel",
                "cohort_col": "cohort",
                "mob_col": "mob",
                "bad_col": "bad",
            }

    monkeypatch.setattr(
        risk_setup, "build_vintage_proposal", lambda *args, **kwargs: _Proposal()
    )
    registry = _Registry([_Dataset("panel", "task-risk/panel.parquet")])

    requested = _advance(
        registry,
        _Backend(),
        user_text="标准Vintage",
        conversation=[_assistant_state("ask_goal")],
    )
    assert requested.template_id is None
    assert requested.intake_state["phase"] == "request_materials"
    assert (
        "cohort" in requested.content
        and "mob" in requested.content
        and "bad" in requested.content
    )

    ready = _advance(
        registry,
        _Backend(),
        user_text="开始",
        conversation=[_assistant_state("request_materials", "standard_vintage")],
    )
    assert ready.template_id == "vintage_analysis"
    assert ready.slots["dataset_id"] == "panel"
    assert ready.intake_state["phase"] == "ready"


def test_question_form_goal_is_not_hijacked_by_adhoc_routing():
    class _Repo:
        def list_agent_messages(self, task_id):
            return [_assistant_state("ask_goal")]

    class _Task:
        id = "task-risk"
        task_type = TASK_TYPE_VINTAGE

    # ``runtime`` intentionally has no plan/data attributes. The risk-intake
    # guard must return before generic ad-hoc routing inspects either of them.
    assert (
        _maybe_handle_adhoc_turn(
            object(),
            _Repo(),
            _Task(),
            user_text="可以做收益测算吗？",
        )
        is None
    )


def test_vintage_http_flow_persists_intake_before_building_standard_plan(tmp_path):
    workspace = tmp_path / "workspace"
    client = TestClient(create_app(workspace))
    source = workspace / "vintage"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "cohort": ["202601", "202601", "202602", "202602"],
            "mob": [0, 1, 0, 1],
            "bad": [0, 1, 0, 0],
        }
    ).to_csv(source / "vintage.csv", index=False)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "Vintage 分析",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "vintage",
            "run_mode": "manual",
            "target_col": "bad",
            "time_col": "cohort",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    first = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert first.status_code == 202, first.text
    assert first.json()["messages"][-1]["metadata"][
        risk_setup.RISK_ANALYSIS_INTAKE_META_KEY
    ] == {"phase": "ask_goal"}

    second = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "标准 Vintage"},
    )
    assert second.status_code == 202, second.text
    assert second.json()["messages"][-1]["metadata"][
        risk_setup.RISK_ANALYSIS_INTAKE_META_KEY
    ] == {
        "phase": "request_materials",
        "analysis_kind": "standard_vintage",
        "analysis_scope": "标准 Vintage",
    }

    third = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "材料已上传"},
    )
    assert third.status_code == 202, third.text
    messages = third.json()["messages"]
    ready = next(
        message
        for message in messages
        if (message.get("metadata") or {})
        .get(risk_setup.RISK_ANALYSIS_INTAKE_META_KEY, {})
        .get("phase")
        == "ready"
    )
    assert (
        ready["metadata"][risk_setup.RISK_ANALYSIS_INTAKE_META_KEY]["analysis_kind"]
        == "standard_vintage"
    )
    assert messages[-1]["metadata"]["kind"] == "plan_overview"
