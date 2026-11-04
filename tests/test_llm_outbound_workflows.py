import json

from marvis.agent.workflow_insights import build_workflow_insight_context, render_workflow_insight
from marvis.domain import TASK_TYPE_DATA_JOIN, TASK_TYPE_FEATURE_ANALYSIS, TASK_TYPE_MODELING
from tests.test_llm_outbound import Wire

PRIVATE = "PRIVATE_FILENAME_OR_REPORT_347"


def _render(monkeypatch, context):
    wire = Wire(monkeypatch)
    wire.reply = {"summary": "依据本次平台指标分析。", "findings": [], "risks": [], "recommendations": []}
    context["facts"].append(PRIVATE)
    context["risks"].append(PRIVATE)
    result = render_workflow_insight(context, client=wire.client, memory_context=[{"summary": PRIVATE}])
    assert result["generated_by"] == "llm"
    assert len(wire.requests) == 1
    assert PRIVATE not in json.dumps(wire.requests)
    return wire.last_context()["measured_evidence"]


def test_join_producer_to_wire_preserves_rates_and_counts_without_file_names(monkeypatch):
    context = build_workflow_insight_context(TASK_TYPE_DATA_JOIN, stage="done", metadata={"tables": [{
        "title": "各特征表贡献", "columns": ["特征表", "命中率", "新列缺失率", "键唯一", "膨胀"],
        "rows": [[PRIVATE, "62%", "0.38", "否", "是"]],
    }]}, content="锚行 100 → 拼接后 125 行")
    measured = _render(monkeypatch, context)
    assert measured == {"anchor_rows": 100, "joined_rows": 125, "join_tables": [{
        "table_index": 1, "match_rate": .62, "missing_rate": .38,
        "key_unique": False, "row_inflation": True,
    }]}


def test_feature_producer_to_wire_keeps_platform_states_and_quality(monkeypatch):
    context = build_workflow_insight_context(TASK_TYPE_FEATURE_ANALYSIS, stage="gate", metadata={"tables": [{
        "title": "Agent 特征建议", "columns": ["特征", "Agent建议", "推荐原因", "建议状态"],
        "rows": [["x", "推荐", PRIVATE, "not_recommended"], ["y", "待评估", PRIVATE, "unevaluated"]],
    }, {"title": "数据质量", "columns": ["特征", "缺失率", "单一值率", "零值率"],
        "rows": [["x", ".6", ".96", "0"]]}]})
    measured = _render(monkeypatch, context)
    assert measured["features"] == [{"feature": "x", "state": "not_recommended", "state_label": "不推荐"},
                                    {"feature": "y", "state": "unevaluated", "state_label": "未评估"}]
    assert measured["quality"] == [{"feature": "x", "missing_rate": .6, "single_value_rate": .96, "zero_rate": 0.}]


def test_model_producer_to_wire_keeps_selected_metrics_and_known_parameters(monkeypatch):
    context = build_workflow_insight_context(TASK_TYPE_MODELING, stage="done", metadata={
        "model_delivery": {"recipe": "lgb", "metrics": {"train_ks": .52, "test_ks": .36, "test_auc": .72,
                           "rows": [[PRIVATE]], "coef_": [.23456789]}},
        "tables": [{"title": "最优超参", "columns": ["参数", "值"],
                    "rows": [["num_leaves", "31"], ["learning_rate", ".05"], ["secret", PRIVATE]]}],
    })
    measured = _render(monkeypatch, context)
    assert measured["recipe"] == "lgb"
    assert measured["metrics"]["train_ks"] == .52
    assert measured["metrics"]["oot_ks"] is None
    assert measured["parameters"] == {"num_leaves": 31., "learning_rate": .05}
    assert measured["risk_flags"] == ["train_test_ks_gap", "oot_ks_unavailable"]


def test_model_wire_rounds_measured_values_to_display_precision(monkeypatch):
    context = build_workflow_insight_context(TASK_TYPE_MODELING, stage="done", metadata={
        "model_delivery": {"recipe": "lr", "metrics": {"train_ks": 0.5087659574468084,
                                                       "test_ks": 0.40096692502494047}},
        "tables": [{"title": "最优超参", "columns": ["参数", "值"],
                    "rows": [["learning_rate", "0.022631"]]}],
    })
    measured = _render(monkeypatch, context)
    assert measured["metrics"]["train_ks"] == 0.5088
    assert measured["metrics"]["test_ks"] == 0.401
    # Hyperparameters are settings, not measurements: exact values travel.
    assert measured["parameters"] == {"learning_rate": 0.022631}


def test_unstructured_legacy_context_falls_back_without_http(monkeypatch):
    wire = Wire(monkeypatch)
    result = render_workflow_insight({"milestone": "model_completed", "facts": [PRIVATE]}, client=wire.client)
    assert result["generated_by"] == "deterministic_fallback"
    assert wire.requests == []


def test_each_workflow_gets_only_its_own_reading_rules(monkeypatch):
    # Eval S3: one shared instruction made a feature analysis say the platform
    # gave no OOT KS goal, overfit flag or duplicate-key check.
    def instruction(task_type, metadata, content=""):
        wire = Wire(monkeypatch)
        wire.reply = {"summary": "依据本次平台指标分析。", "findings": [], "risks": [], "recommendations": []}
        context = build_workflow_insight_context(task_type, stage="gate", metadata=metadata, content=content)
        render_workflow_insight(context, client=wire.client, memory_context=[])
        return str(wire.last_context()["instruction"])

    feature = instruction(TASK_TYPE_FEATURE_ANALYSIS, {"tables": [{
        "title": "数据质量", "columns": ["特征", "缺失率", "最常见值占比", "零值率"], "rows": [["x", ".1", ".2", "0"]],
    }]})
    assert "不涉及模型训练、OOT 目标" in feature
    assert "overfit_flag=true 必须" not in feature
    assert "重复键" not in feature
    assert "最常见取值所占比例" in feature
    join = instruction(TASK_TYPE_DATA_JOIN, {"tables": [{
        "title": "各特征表贡献", "columns": ["特征表", "命中率", "新列缺失率", "键唯一", "膨胀"],
        "rows": [["t", "62%", "0.38", "否", "是"]],
    }]}, content="锚行 100 → 拼接后 125 行")
    assert "重复键" in join and "overfit_flag" not in join


def test_feature_and_model_readings_get_the_platform_psi_bands(monkeypatch):
    # Eval S4: PSI 0.0672 was read as 「分布存在一定差异」; the platform calls it stable.
    def instruction(task_type, metadata):
        wire = Wire(monkeypatch)
        wire.reply = {"summary": "依据本次平台指标分析。", "findings": [], "risks": [], "recommendations": []}
        context = build_workflow_insight_context(task_type, stage="gate", metadata=metadata)
        render_workflow_insight(context, client=wire.client, memory_context=[])
        return str(wire.last_context()["instruction"])

    model = instruction(TASK_TYPE_MODELING, {"model_delivery": {"recipe": "lr", "metrics": {"train_ks": .5, "test_ks": .48}}})
    assert "低于 0.10 为稳定" in model
    assert "overfit_train_oot_gap" in model and "pre_refit_heldout 等代码" in model
    assert "不要声称做过历史对比" in model and "未提供，不能声称" not in model


def test_join_reading_does_not_speculate_about_other_tables(monkeypatch):
    # Eval S6: a one-table join was read as 「其他拼接表未给出度量」 three times.
    wire = Wire(monkeypatch)
    wire.reply = {"summary": "依据本次平台指标分析。", "findings": [], "risks": [], "recommendations": []}
    context = build_workflow_insight_context(TASK_TYPE_DATA_JOIN, stage="done", metadata={"tables": [{
        "title": "各特征表贡献", "columns": ["特征表", "命中率", "新增列", "新列缺失率", "去重策略"],
        "rows": [["bureau_features.csv", "1.0", "7", "0.0", "无"]],
    }]}, content="锚行 240 → 拼接后 240 行")
    render_workflow_insight(context, client=wire.client, memory_context=[])
    instruction = str(wire.last_context()["instruction"])
    assert "不要推测还有其他表" in instruction
    assert "拼接后行数等于锚点行数即说明没有行数膨胀" in instruction
