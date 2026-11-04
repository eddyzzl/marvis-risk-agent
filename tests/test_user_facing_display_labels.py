"""What users read carries no ids, server paths, Python reprs or English codes.

2026-10-19 Agent eval round 1 (P2 "内部信息外露"). Machine values stay in the
tool outputs / metadata; these tests pin only the displayed text.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from marvis.agent.gate_payloads import build_screen_payload
from marvis.agent.presenters.display_labels import (
    metric_label,
    metric_value_text,
    money_text,
    readable_selection_reason,
    screening_reason_label,
    scrub_internal_ids,
)
from marvis.agent.renderers import (
    _render_choose_modeling_spec,
    _render_el_estimate,
    _render_execute_join,
    _render_portfolio_gate_summary,
    _render_post_training_action,
    _render_screen,
    _render_train,
    render_tool_output,
)
from marvis.orchestrator.templates.portfolio import PORTFOLIO_ANALYSIS, PORTFOLIO_ANALYSIS_NO_TREND
from marvis.packs.data_ops.tools import _friendly_name

SERVER_PATH = "/Users/someone/.omc/eval-ws/tasks/task-1/feature_reports/feature_report_979d57b70d641774.xlsx"
EXPERIMENT = "experiment_11bd9106" + "0" * 24
ARTIFACT = "artifact_599560" + "f" * 26


def _visible(text, tables) -> str:
    return text + json.dumps(tables, ensure_ascii=False)


def test_report_completion_messages_show_no_server_path():
    cases = {
        "generate_feature_report": {"metrics": [], "report_path": SERVER_PATH},
        "portfolio_report": {"report_path": SERVER_PATH, "sheets": ["组合概览", "数据质量红旗"]},
        "generate_model_report": {"report_path": SERVER_PATH, "section_status": []},
        "generate_model_reports": {
            "report_path": SERVER_PATH,
            "reports": [{"experiment_id": EXPERIMENT, "recipe": "lr", "report_path": SERVER_PATH}],
        },
    }
    for tool, output in cases.items():
        shown = _visible(*render_tool_output(tool, output))
        assert "/Users/" not in shown, tool
        assert "feature_report_979d" not in shown, tool
        assert EXPERIMENT not in shown, tool
        assert "报告已生成" in shown, tool


def test_special_value_and_generic_completions_hide_ids():
    text, _ = render_tool_output("resolve_special_values", {
        "result_dataset_id": "ds_046d1234",
        "selected": ["age", "income"],
        "policy_fingerprint": "",
        "masked": ["income"],
        "retained": [],
        "dropped": [],
    })
    assert "result_dataset_id" not in text and "ds_046d" not in text
    assert "policy_fingerprint" not in text
    assert "income" in text and "2 个特征" in text

    generic, _ = render_tool_output("unknown_tool_for_display", {
        "result_dataset_id": "ds_1", "policy_fingerprint": "sha256:abc", "oot_ks": 0.41234, "feature_count": 6.0,
    })
    assert "ds_1" not in generic and "fingerprint" not in generic
    assert "OOT KS 0.4123" in generic and "特征数 6" in generic


def test_screen_text_lists_names_and_reasons_in_chinese():
    text, tables = _render_screen({
        "selected": ["sig1"],
        "leakage": [["score", 0.448, "univariate KS 0.448 >= 0.4 — suspected target leakage"]],
        "suspected": [],
        "unusable": [["const_col", "constant"]],
        "excluded_categorical": [{"column": "channel", "cardinality": 3}],
        "scores": {"sig1": {"ks": 0.2, "iv": 0.1, "missing_rate": 0.0}},
        "n_screened": 4,
    })
    shown = _visible(text, tables)
    assert "['score']" not in shown and "（如 score）" in text
    assert "woe_encode_categorical" not in shown and "WOE 编码" in text
    assert "suspected target leakage" not in shown
    assert "单变量 KS 0.448 ≥ 0.4，疑似目标泄漏" in shown
    assert "取值单一（常量）" in shown


def test_screen_gate_payload_carries_chinese_reason_labels_beside_raw_reasons():
    output = {
        "leakage": [["score", 0.448, "univariate KS 0.448 >= 0.4 — suspected target leakage"]],
        "suspected": [["model_score", 0.31, "name looks like a model output/score — confirm it is an allowed input"]],
        "unusable": [["cust_id", "identity/row key: application/entity sequence identifier; excluded by name contract (not by high cardinality)"]],
    }
    payload = build_screen_payload(output, SimpleNamespace(id="screen", title="筛选", inputs={}))
    # raw reasons kept for code and audit
    assert payload["leakage"][0][2].endswith("suspected target leakage")
    assert payload["reason_labels"] == {
        "score": "单变量 KS 0.448 ≥ 0.4，疑似目标泄漏",
        "model_score": "列名像模型输出/评分，请确认是否允许作为入模特征",
        "cust_id": "主键/流水号类标识列，按命名规则排除",
    }


def test_modeling_spec_reads_in_business_words():
    text, tables = _render_choose_modeling_spec({
        "recipes": ["lr"], "recipe": "lr", "target_type": "binary", "feature_count": 6,
        "n_trials": 20,
        "metric_policy": "higher overfit-penalized test KS; OOT reported only, not used for selection",
        "eligible_algorithms": ["lr"],
        "disabled_algorithms": [{"recipe": "lgb_regressor", "reason": "recipe target family does not match `binary`"}],
    })
    shown = _visible(text, tables)
    assert "overfit-penalized" not in shown and "recipe target family" not in shown
    assert "按测试集 KS（扣除过拟合惩罚）择优；OOT 只用于展示，不参与选择" in shown
    assert "该算法不适用于二分类目标" in shown
    assert "二分类" in text and "逻辑回归" in text


def test_metric_tables_use_business_labels_and_integer_counts():
    _, tables = _render_train({"metrics": {"oot_ks": 0.5681, "feature_count": 6.0, "oot_ks_ci_high": 0.61}})
    rows = dict(tables[0]["rows"])
    assert rows == {"OOT KS": "0.5681", "特征数": "6", "OOT KS 置信上限": "0.6100"}


def test_selection_reason_and_ids_read_as_sentences():
    assert readable_selection_reason(f"用户指定实验（平台推荐为 `{EXPERIMENT}`）。") == "用户指定实验（平台推荐的是另一个实验）。"
    assert readable_selection_reason("平台推荐候选（按 test_ks(overfit-penalized) 选优），用户已确认。") == (
        "平台推荐候选（按 测试集 KS（扣除过拟合惩罚） 选优），用户已确认。"
    )
    assert scrub_internal_ids(f"评估指标来自所选实验 `{EXPERIMENT}` 的留出集；交付模型 `{ARTIFACT}` 用相同超参") == (
        "评估指标来自所选实验的留出集；交付模型用相同超参"
    )
    assert metric_label("weighted_oot_ks") == "加权OOT KS"
    assert metric_value_text("feature_count", 6.0) == "6" and metric_value_text("oot_ks", None) == "—"


def test_delivery_card_names_deliverables_without_internal_files():
    root = "/Users/someone/workspace/tasks/task-1/modeling_artifacts"
    text, tables = _render_post_training_action({
        "native_model_path": f"{root}/{ARTIFACT}.joblib",
        "approval_package_path": f"{root}/{ARTIFACT}.approval_package.json",
        "approval_package_markdown_path": f"{root}/{ARTIFACT}.approval_package.md",
        "model_card_path": f"{root}/{ARTIFACT}.model_card.json",
        "model_card_markdown_path": f"{root}/{ARTIFACT}.model_card.md",
        "actions": [
            {"action": "export_pmml", "status": "succeeded", "pmml_path": f"{root}/{ARTIFACT}.pmml"},
            {"action": "handoff_to_validation", "status": "succeeded", "validation_task_id": "task-9"},
        ],
    })
    shown = _visible(text, tables)
    assert ARTIFACT not in shown and "/Users/" not in shown and "task-9" not in shown
    assert "succeeded" not in shown and "export_pmml" not in shown
    rows = {row[0]: row for row in tables[0]["rows"]}
    assert rows["模型文件"][1:3] == ["已完成", "模型文件"]
    assert rows["审批材料包"][2] == "Markdown 文档"
    assert rows["模型卡"][2] == "Markdown 文档"
    assert rows["导出 PMML"][2] == "PMML 文件"
    assert rows["移交验证"][2] == "已创建验证任务"


class _Registry:
    def source_identity(self, dataset_id):
        return {"original_name": "bureau_features.csv"} if dataset_id == "ds_feat" else None

    def get(self, dataset_id):
        return SimpleNamespace(source_path="578b547bde361.parquet")


def test_join_tables_show_uploaded_file_names():
    assert _friendly_name(_Registry(), "ds_feat") == "bureau_features.csv"
    # legacy/derived datasets without an upload identity keep the stored name
    assert _friendly_name(_Registry(), "ds_other") == "578b547bde361.parquet"

    text, tables = _render_execute_join({
        "result_dataset_id": "ds_e98c", "anchor_rows": 240, "joined_rows": 240, "warnings": [],
        "per_table": [{"feature_id": "ds_acd103c", "feature_name": "bureau_features.csv", "match_rate": 1.0,
                       "new_columns": 5, "new_columns_null_rate": 0.0, "dedup_strategy": "无"}],
    })
    assert "ds_e98c" not in text and "锚行 240 → 拼接后 240 行" in text
    assert tables[0]["rows"][0][0] == "bureau_features.csv"


def test_portfolio_money_and_gate_highlights_read_in_business_terms():
    text, tables = _render_el_estimate({
        "loss_state": "charged_off", "total_el": 8300.0,
        "el_by_month": [{"month": "2026-03", "balance": 8300.0, "expected_loss": 1234.5, "is_reference": True}],
        "chain": [], "red_flags": [], "assumptions": {},
    })
    assert "8,300.00" in text
    assert tables[0]["rows"][0][1:] == ["8,300.00", "1,234.50"]
    assert money_text(None) == "n/a"

    _, gate_tables = _render_portfolio_gate_summary({
        "red_flag_count": 1, "checklist": ["月 2026-01 对齐月对仅 3 对"],
        "highlights": {"total_el": 8300.0, "total_el_basis": "reference_snapshot", "reference_snapshot": "2026-03",
                       "hhi": 0.5, "top1_pct": 0.5, "migration_months": ["2026-01", "2026-02"]},
    })
    rows = dict(gate_tables[0]["rows"])
    assert rows == {
        "合计预期损失（参考快照口径）": "8,300.00",
        "参考快照月": "2026-03",
        "细分集中度 HHI": "0.5000",
        "最大细分占比": "50.0%",
        "迁徙窗口月份": "2026-01、2026-02",
    }


def test_portfolio_plan_rail_stages_have_names():
    for template in (PORTFOLIO_ANALYSIS, PORTFOLIO_ANALYSIS_NO_TREND):
        phases = [step.phase for step in template.steps]
        assert None not in phases
        assert phases[-1] == "报告" and set(phases[:-1]) == {"组合分析"}


def test_screening_reason_labels_cover_the_screen_vocabulary():
    assert screening_reason_label("missing rate 0.97 >= 0.95") == "缺失率 0.97 ≥ 0.95"
    assert screening_reason_label("only 1 distinct non-null value(s)") == "非空取值只有 1 种"
    assert screening_reason_label(
        "semantic/temporal target leakage（语义/时序目标泄漏）：贷后 FPD 表现字段，申请/评分时点不可得"
    ) == "语义/时序目标泄漏：贷后 FPD 表现字段，申请/评分时点不可得"
    assert screening_reason_label("业务自定义原因") == "业务自定义原因"
