from __future__ import annotations

import json

from marvis.agent.workflow_insights import (
    build_workflow_insight_context,
    render_workflow_insight,
)
from marvis.domain import (
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
)


def test_join_insight_calls_out_low_match_and_row_inflation():
    context = build_workflow_insight_context(
        TASK_TYPE_DATA_JOIN,
        stage="done",
        metadata={
            "tables": [{
                "title": "各特征表贡献",
                "columns": ["特征表", "命中率", "新增列", "新列缺失率", "去重策略"],
                "rows": [
                    ["vars_a", "0.9600", "3", "0.0100", "无"],
                    ["vars_b", "0.6200", "2", "0.3800", "无"],
                ],
            }],
            "result_dataset": {"dataset_id": "ds-result"},
        },
        content="锚行 100 → 拼接后 125 行",
    )

    assert context is not None
    assert context["milestone"] == "join_completed"
    assert any("vars_b" in item and "命中率" in item for item in context["risks"])
    assert any("100" in item and "125" in item for item in context["risks"])


def test_feature_insight_distinguishes_recommended_and_risky_features():
    context = build_workflow_insight_context(
        TASK_TYPE_FEATURE_ANALYSIS,
        stage="gate",
        metadata={
            "feature_binning": {"features": [{"feature": "x_good"}, {"feature": "x_bad"}]},
            "tables": [
                {
                    "title": "Agent 特征建议",
                    "columns": ["特征", "Agent建议", "推荐原因"],
                    "rows": [
                        ["x_good", "推荐", "区分力较好"],
                        ["x_bad", "不推荐", "缺失率过高"],
                    ],
                },
                {
                    "title": "数据质量",
                    "columns": ["特征", "有效样本", "缺失率", "单一值率", "零值率"],
                    "rows": [
                        ["x_good", "100", "0.01", "0.20", "0.10"],
                        ["x_bad", "40", "0.60", "0.96", "0.00"],
                    ],
                },
            ],
        },
        content="单变量分析已完成",
    )

    assert context is not None
    assert "x_good" in context["recommended_features"]
    assert "x_bad" in context["avoid_features"]
    assert any("x_bad" in item and "缺失率" in item for item in context["risks"])


def test_feature_insight_keeps_unevaluated_and_empty_labels_neutral():
    context = build_workflow_insight_context(
        TASK_TYPE_FEATURE_ANALYSIS,
        stage="done",
        metadata={
            "tables": [{
                "title": "Agent 特征建议",
                "columns": [
                    "特征",
                    "Agent建议",
                    "推荐原因",
                    "建议状态",
                    "证据置信度",
                    "支持指标",
                ],
                "rows": [
                    ["x_candidate", "候选", "有信号", "candidate", "medium", "ks=0.2"],
                    ["x_pending", "待评估", "未选择信号指标", "unevaluated", "none", "-"],
                    ["x_empty", "", "", "", "none", "-"],
                    [
                        "x_conflict",
                        "推荐",
                        "质量冲突，以状态为准",
                        "not_recommended",
                        "medium",
                        "missing_rate=0.9",
                    ],
                ],
            }],
        },
    )

    assert context is not None
    assert context["recommended_features"] == ["x_candidate"]
    assert context["avoid_features"] == ["x_conflict"]
    assert "x_pending" not in context["recommended_features"]
    assert "x_empty" not in context["recommended_features"]
    assert any("x_pending：待评估" in fact for fact in context["facts"])


def test_model_insight_exposes_metrics_params_and_overfit_warning():
    context = build_workflow_insight_context(
        TASK_TYPE_MODELING,
        stage="done",
        metadata={
            "model_delivery": {
                "recipe": "lgb",
                "metrics": {
                    "train_ks": 0.52,
                    "test_ks": 0.36,
                    "oot_ks": 0.31,
                    "test_auc": 0.72,
                },
            },
            "tables": [{
                "title": "最优超参",
                "columns": ["参数", "值"],
                "rows": [["num_leaves", "31"], ["learning_rate", "0.05"]],
            }],
        },
        content="模型训练完成",
    )

    assert context is not None
    assert context["recommended_params"] == {"num_leaves": "31", "learning_rate": "0.05"}
    assert any("过拟合" in item for item in context["risks"])
    assert any("OOT KS" in item for item in context["facts"])


def test_model_insight_uses_selected_candidate_metrics_before_selection_output_exists():
    context = build_workflow_insight_context(
        TASK_TYPE_MODELING,
        stage="gate",
        metadata={
            "model_delivery": {
                "recipe": "",
                "metrics": {},
                "candidates": [
                    {
                        "experiment_id": "exp-lgb",
                        "recipe": "lgb",
                        "selected": False,
                        "metrics": {
                            "train_ks": 0.79,
                            "test_ks": 0.58,
                            "oot_ks": 0.42,
                            "oot_auc": 0.77,
                        },
                    },
                    {
                        "experiment_id": "exp-xgb",
                        "recipe": "xgb",
                        "selected": False,
                        "metrics": {
                            "train_ks": 0.91,
                            "test_ks": 0.59,
                            "oot_ks": 0.40,
                        },
                    },
                ],
            },
            "tables": [{
                "title": "候选模型对比",
                "columns": ["算法", "Test KS", "OOT KS"],
                "rows": [
                    ["lgb ★", "0.58", "0.42"],
                    ["xgb", "0.59", "0.40"],
                ],
            }],
        },
        content="请确认冠军实验",
    )

    assert context is not None
    assert any("当前推荐模型为 lgb" in item for item in context["facts"])
    assert any("OOT KS=0.4200" in item for item in context["facts"])
    assert any("Test/OOT KS" in item for item in context["risks"])
    assert not any("当前没有 OOT KS" in item for item in context["risks"])


def test_render_workflow_insight_uses_llm_but_keeps_grounded_context():
    class FakeLLM:
        profile = {"model_name": "fake"}

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            assert prompt["facts"] == ["已分析 3 个特征"]
            assert prompt["memory_context"][0]["summary"] == "历史任务曾发现 x2 不稳定"
            return json.dumps({
                "summary": "整体可用，但应排除 x2。",
                "findings": ["x1 区分力较好"],
                "risks": ["x2 历史稳定性较差"],
                "recommendations": ["保留 x1，复核 x2"],
            }, ensure_ascii=False)

    result = render_workflow_insight(
        {
            "title": "特征分析解读",
            "milestone": "feature_analyzed",
            "facts": ["已分析 3 个特征"],
            "risks": [],
            "recommendations": [],
            "recommended_features": ["x1"],
            "avoid_features": ["x2"],
            "recommended_params": {},
            "evidence": ["Agent 特征建议"],
        },
        client=FakeLLM(),
        memory_context=[{"summary": "历史任务曾发现 x2 不稳定"}],
    )

    assert result["generated_by"] == "llm"
    assert "整体可用" in result["content"]
    assert "x2 历史稳定性较差" in result["content"]
