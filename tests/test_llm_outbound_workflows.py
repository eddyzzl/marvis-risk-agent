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
    assert measured["features"] == [{"feature": "x", "state": "not_recommended"},
                                    {"feature": "y", "state": "unevaluated"}]
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


def test_unstructured_legacy_context_falls_back_without_http(monkeypatch):
    wire = Wire(monkeypatch)
    result = render_workflow_insight({"milestone": "model_completed", "facts": [PRIVATE]}, client=wire.client)
    assert result["generated_by"] == "deterministic_fallback"
    assert wire.requests == []
