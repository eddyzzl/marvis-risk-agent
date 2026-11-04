"""The selected experiment's held-out metrics stay the evaluation evidence.

A champion refit delivers a new model fitted on train+test (the native
development refit is also never scored on OOT). Selection card, delivery
payload, approval, report and Agent interpretation must keep evaluating the
selected experiment's held-out metrics, label the refit, and never print a
literal ``None``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from marvis.agent.gate_payloads import _policy_signals, build_model_delivery_payload
from marvis.agent.renderers import _render_select_experiment
from marvis.agent.workflow_insights import build_workflow_insight_context, render_workflow_insight
from marvis.domain import TASK_TYPE_MODELING
from marvis.packs.modeling.development_refit_reporting import dataset_split_rows, model_narrative
from marvis.packs.modeling.select_tools import _platform_recommendation
from marvis.plugins.manifest import ToolRef
from tests.test_llm_outbound import Wire
from tests.test_modeling_pack import _register_modeling_sample, _runtime

# Real experiment ids are ``experiment_<uuid hex>``; the card must not print them.
PARENT = "experiment_" + "a1" * 16
REFIT = "experiment_" + "b2" * 16
PARENT_METRICS = {
    "train_ks": 0.4647, "test_ks": 0.5714, "oot_ks": 0.5681,
    "train_auc": 0.8126, "test_auc": 0.7844, "oot_auc": 0.8200,
    "psi_oot_vs_train": 0.1091, "overfit_train_test_gap": 0.2296, "overfit_flag": True,
}
CAPS = {"pmml_supported": True, "handoff_supported": True, "native_model_supported": True}


def _step(tool):
    return SimpleNamespace(id=f"step-{tool}", title=tool, tool_ref=ToolRef("modeling", tool))


def _dev_refit_select_output():
    """Shape of the eval task's select_experiment output (native development refit)."""
    return {
        "selected_experiment_id": REFIT,
        "selection_origin": "platform_recommended_user_confirmed",
        "artifact_id": "artifact_refit",
        "recipe": "lr",
        "target_type": "binary",
        "selection_metric": "test_ks(overfit-penalized)",
        "selection_reason": "平台推荐候选（按 test_ks(overfit-penalized) 选优），用户已确认。",
        "metrics": {
            "train_ks": 0.5088, "train_auc": 0.8115, "test_ks": None, "oot_ks": None,
            "test_auc": None, "oot_auc": None, "overfit_flag": None,
        },
        "evaluation_evidence": {
            "basis": "pre_refit_heldout",
            "experiment_id": PARENT,
            "artifact_id": "artifact_parent",
            "delivered_experiment_id": REFIT,
            "delivered_artifact_id": "artifact_refit",
            "refit_applied": True,
            "metrics": dict(PARENT_METRICS),
            "note": f"评估指标来自所选实验 `{PARENT}` 的留出集；交付模型 `{REFIT}` 用相同超参在 train+test 上重新拟合。",
        },
        "refit": {"applied": True, "requested": True, "oot_evaluation": "not_performed"},
        "capabilities": dict(CAPS),
        "policy_decision": {"status": "accepted", "warnings": [
            {"code": "overfit_warning", "message": "train-test KS 差距为 0.2296,超过警戒阈值 0.1。"},
        ]},
        "experiments": [{"id": PARENT, "recipe": "lr", "artifact_id": "artifact_parent",
                         "capabilities": dict(CAPS), "feature_count": 6, **PARENT_METRICS}],
    }


def test_selection_card_labels_refit_and_never_prints_none():
    text, tables = _render_select_experiment(_dev_refit_select_output())

    assert "逻辑回归" in text.split("\n")[0]
    assert "重新拟合" in text
    assert PARENT not in text and REFIT not in text
    assert "评估指标来自所选实验的留出集" in text
    assert "用户指定" not in text
    assert "None" not in json.dumps(tables, ensure_ascii=False)
    evaluation = next(table for table in tables if table["title"].startswith("最终模型指标"))
    assert "留出集" in evaluation["title"]
    assert PARENT not in evaluation["title"]
    assert dict(evaluation["rows"])["OOT KS"] == "0.5681"
    refit = dict(next(table for table in tables if table["title"].startswith("交付模型指标"))["rows"])
    assert refit["训练集 KS"] == "0.5088"
    assert refit["测试集 KS"].startswith("—（未计算")
    assert "OOT" in refit["OOT KS"]
    capability = dict(tables[0]["rows"])
    assert "过拟合" not in capability.get("策略门控", "")
    assert "0.2296" in capability["质量提醒"]


def test_delivery_approval_uses_holdout_evidence_not_refit_fit_metrics():
    delivery = build_model_delivery_payload(_dev_refit_select_output(), _step("select_experiment"))

    assert delivery["evaluation_evidence"]["experiment_id"] == PARENT
    assert delivery["policy_signals"]["approval"] != "建议可审批"
    assert delivery["policy_signals"]["approval"] == "需业务复核"
    assert any("留出集" in reason for reason in delivery["policy_signals"]["reasons"])
    assert delivery["business_signals"]["stability"] == "需复核"
    assert delivery["business_signals"]["feature_count"] == 6
    assert [row["selected"] for row in delivery["candidates"]] == [True]
    assert "重新拟合" in delivery["selection_reason"]


def test_fit_only_metrics_never_read_as_approvable():
    signals = _policy_signals({"metrics": {"train_ks": 0.51, "train_auc": 0.81, "test_ks": None},
                               "capabilities": dict(CAPS)})
    assert signals["approval"] == "待评估"
    assert any("留出集" in reason for reason in signals["reasons"])


def test_compare_summary_matches_the_deliverable_candidate_row():
    compare_output = {"experiments": _dev_refit_select_output()["experiments"], "eval_metric": "ks_auc"}
    delivery = build_model_delivery_payload(
        compare_output, _step("compare_experiments"), recommended_experiment_id=PARENT
    )

    row = delivery["candidates"][0]
    assert row["business_signals"]["delivery"] == "可移交"
    assert delivery["business_signals"]["delivery"] == row["business_signals"]["delivery"]
    assert delivery["capabilities"]["pmml_supported"] is True
    readiness = {item["id"]: item["status"] for item in delivery["readiness"]}
    assert readiness["native_model"] == "ready"
    assert readiness["pmml"] == "ready"
    assert "缺少可交付模型产物" not in delivery["policy_signals"]["reasons"]
    assert delivery["policy_signals"]["approval"] == row["policy_signals"]["approval"]


def test_interpretation_reads_selected_holdout_metrics_goal_and_overfit():
    delivery = build_model_delivery_payload(_dev_refit_select_output(), _step("select_experiment"))
    task = SimpleNamespace(oot_ks_min=0.3, business_objective=None)
    context = build_workflow_insight_context(
        TASK_TYPE_MODELING, stage="gate", metadata={"model_delivery": delivery}, task=task
    )

    measured = context["measured_evidence"]
    assert measured["evidence_basis"] == "pre_refit_heldout"
    assert measured["metrics"]["oot_ks"] == pytest.approx(0.5681)
    assert measured["metrics"]["test_ks"] == pytest.approx(0.5714)
    assert measured["overfit_flag"] is True
    assert measured["goal"] == {"metric": "oot_ks", "min": 0.3, "met": True}
    assert "overfit_flag" in measured["risk_flags"]
    assert "oot_ks_unavailable" not in measured["risk_flags"]
    content = render_workflow_insight(context)["content"]
    assert "平台标记了过拟合风险" in content and "overfit_flag=" not in content
    assert "已达成" in content
    assert "留出集" in content


def test_interpretation_goal_from_business_objective_and_unverifiable_without_oot():
    criterion = SimpleNamespace(metric="oot_ks", minimum=0.3, comparison="absolute")
    task = SimpleNamespace(oot_ks_min=None, business_objective=SimpleNamespace(criteria=(criterion,)))
    context = build_workflow_insight_context(TASK_TYPE_MODELING, stage="done", metadata={
        "model_delivery": {"recipe": "lr", "metrics": {"train_ks": 0.5, "test_ks": 0.4}},
    }, task=task)
    assert context["measured_evidence"]["goal"]["met"] is None
    assert "oot_ks_goal_unverifiable" in context["measured_evidence"]["risk_flags"]


def test_compare_step_interpretation_reads_the_candidate_metrics():
    compare_output = {"experiments": _dev_refit_select_output()["experiments"]}
    delivery = build_model_delivery_payload(compare_output, _step("compare_experiments"))
    context = build_workflow_insight_context(
        TASK_TYPE_MODELING, stage="gate", metadata={"model_delivery": delivery}
    )
    assert context["measured_evidence"]["metrics"]["train_ks"] == pytest.approx(0.4647)
    assert "Train KS=0.4647" in context["facts"]


def test_outbound_prompt_carries_evidence_basis_goal_and_overfit(monkeypatch):
    delivery = build_model_delivery_payload(_dev_refit_select_output(), _step("select_experiment"))
    context = build_workflow_insight_context(
        TASK_TYPE_MODELING, stage="gate", metadata={"model_delivery": delivery},
        task=SimpleNamespace(oot_ks_min=0.3, business_objective=None),
    )
    wire = Wire(monkeypatch)
    wire.reply = {"summary": "依据平台指标。", "findings": [], "risks": [], "recommendations": []}
    render_workflow_insight(context, client=wire.client)
    measured = wire.last_context()["measured_evidence"]
    assert measured["evidence_basis"] == "pre_refit_heldout"
    assert measured["overfit_flag"] is True
    assert measured["goal"] == {"metric": "oot_ks", "min": 0.3, "met": True}
    assert measured["metrics"]["oot_ks"] == pytest.approx(0.5681)
    assert "overfit_flag" in measured["risk_flags"]
    assert PARENT not in json.dumps(measured)


def test_platform_recommendation_matches_automatic_pick_without_mutating_rows():
    rows = [
        {"id": "a", "recipe": "lr", "train_ks": 0.40, "test_ks": 0.38, "capabilities": dict(CAPS)},
        {"id": "b", "recipe": "mlp", "train_ks": 0.60, "test_ks": 0.45,
         "capabilities": {"native_model_supported": True}},
    ]
    recommended, metric = _platform_recommendation(rows, target_type="binary", policy={}, eval_metric="ks_auc")
    assert recommended == "a"
    assert metric == "test_ks(overfit-penalized)"
    assert "delivery_excluded" not in rows[1]


def test_refit_report_lists_parent_oot_and_holdout_narrative():
    scope = {
        "experiment_id": REFIT,
        "artifact_id": "artifact_refit",
        "parent_metrics": dict(PARENT_METRICS),
        "proof": {
            "parent": {"experiment_id": PARENT, "artifact_id": "artifact_parent"},
            "memberships": {"development_fit": {"rows": 900}, "original_train": {"rows": 700},
                            "original_test": {"rows": 200}},
        },
    }
    rows = {row["split"]: row for row in dataset_split_rows(scope, SimpleNamespace(train_ks=0.5, train_auc=0.8))}
    assert rows["pre_refit_oot"]["ks"] == PARENT_METRICS["oot_ks"]
    assert rows["pre_refit_oot"]["experiment_id"] == PARENT
    narrative = model_narrative(scope)
    assert "所选实验（重新拟合前）的留出集" in narrative and PARENT not in narrative
    assert "OOT KS=0.5681" in narrative
    assert "该实验被平台标记了过拟合风险" in narrative and "overfit_flag" not in narrative


@pytest.fixture(scope="module")
def confirmed_refit(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("confirmed-refit")
    runner, _pr, registry, _backend, settings, task = _runtime(tmp_path)
    dataset = _register_modeling_sample(registry, tmp_path, task.id)
    prepared = runner.invoke(ToolRef("modeling", "prepare_modeling_frame"), {
        "dataset_id": dataset.id, "target_col": "y", "feature_cols": ["x1", "x2"],
        "split_col": "split", "seed": 11,
    }, task_id=task.id)
    assert prepared.ok is True, prepared.error

    def train(recipe, params):
        result = runner.invoke(ToolRef("modeling", "train_model"), {
            "dataset_id": prepared.output["result_dataset_id"], "recipe": recipe,
            "features": ["x1", "x2"], "target_col": "y", "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": params, "seed": 23,
        }, task_id=task.id)
        assert result.ok is True, result.error
        return result.output["experiment_id"]

    lgb = train("lgb", {"num_boost_round": 5, "learning_rate": 0.1, "num_leaves": 4})
    lr = train("lr", {})
    candidates = [lgb, lr]
    compared = runner.invoke(ToolRef("modeling", "compare_experiments"),
                             {"experiment_ids": candidates}, task_id=task.id)
    assert compared.ok is True, compared.error
    recommended, _metric = _platform_recommendation(
        compared.output["experiments"], target_type="binary", policy={}, eval_metric="ks_auc"
    )
    other = next(item for item in candidates if item != recommended)

    def select(selected_id, **extra):
        result = runner.invoke(ToolRef("modeling", "select_experiment"), {
            "experiment_ids": candidates, "target_type": "binary",
            "selected_experiment_id": selected_id, **extra,
        }, task_id=task.id)
        assert result.ok is True, result.error
        return result.output

    override = select(other, refit_on_train_plus_test=False)
    confirmed = select(recommended)
    post = runner.invoke(ToolRef("modeling", "post_training_action"), {
        "experiment_id": confirmed["selected_experiment_id"], "sample_dataset_id": dataset.id,
        "actions": [], "selection_policy_decision": confirmed["policy_decision"],
    }, task_id=task.id)
    assert post.ok is True, post.error
    rows = {row["id"]: row for row in compared.output["experiments"]}
    return SimpleNamespace(recommended=recommended, other=other, rows=rows,
                           override=override, confirmed=confirmed, post=post.output)


@pytest.mark.slow
def test_confirmed_recommendation_is_not_a_user_override(confirmed_refit):
    confirmed = confirmed_refit.confirmed
    assert confirmed["selection_origin"] == "platform_recommended_user_confirmed"
    assert "用户指定" not in confirmed["selection_reason"]
    assert "平台推荐" in confirmed["selection_reason"]
    override = confirmed_refit.override
    assert override["selection_origin"] == "user_override"
    assert override["selection_metric"] == "manual"
    assert f"平台推荐为 `{confirmed_refit.recommended}`" in override["selection_reason"]
    assert override["evaluation_evidence"]["basis"] == "selected_experiment"
    assert override["evaluation_evidence"]["experiment_id"] == confirmed_refit.other


@pytest.mark.slow
def test_refit_keeps_selected_holdout_metrics_as_evaluation_evidence(confirmed_refit):
    confirmed = confirmed_refit.confirmed
    assert confirmed["refit"]["applied"] is True
    evidence = confirmed["evaluation_evidence"]
    assert evidence["basis"] == "pre_refit_heldout"
    assert evidence["experiment_id"] == confirmed_refit.recommended
    assert evidence["delivered_experiment_id"] == confirmed["selected_experiment_id"]
    assert evidence["metrics"]["test_ks"] == confirmed_refit.rows[confirmed_refit.recommended]["test_ks"]
    assert confirmed_refit.recommended in evidence["note"] and "train+test" in evidence["note"]
    # The delivered refit's own metrics keep the D14 contract.
    assert "test_ks" not in confirmed["metrics"]
    text, tables = _render_select_experiment(confirmed)
    assert "None" not in json.dumps(tables, ensure_ascii=False)
    assert "重新拟合" in text

    post_evidence = confirmed_refit.post["evaluation_evidence"]
    assert post_evidence["basis"] == "delivered_experiment"
    assert post_evidence["refit_applied"] is True
    assert "test_ks" not in post_evidence["metrics"]
