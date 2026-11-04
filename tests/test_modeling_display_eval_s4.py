"""Modeling result cards read as business text (agent eval round 1, S4).

The tuning card listed the platform's evidence records as raw dicts under
「最优超参」, metric tables kept keys like ``overfit_train_oot_gap``, and the
model reading wrote 「overfit_flag=true」 without saying which gap set it.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from marvis.agent.presenters.display_labels import metric_label, metric_value_text
from marvis.agent.renderers import _render_tune
from marvis.agent.workflow_insights import build_workflow_insight_context
from marvis.domain import TASK_TYPE_MODELING


def test_tuning_card_lists_only_estimator_settings_with_readable_metric_names():
    _text, tables = _render_tune({
        "n_trials": 1,
        "best_params": {
            "C": 0.0635,
            "selection_evidence": {"schema_version": "marvis.selection_consumption.v1"},
            "preprocessing_assurance": "unknown",
            "feature_time_evidence": {"assurance": "unknown"},
            "fitted_input_time_evidence": None,
        },
        "best_metrics": {
            "search_stage": "coarse",
            "lift_head_5": 1.6164,
            "test_ks_std": 0.1268,
            "oot_stability_gap": None,
        },
        "trials": [{"score": 0.31, "train_ks": 0.37, "test_ks": 0.38}],
    })
    by_title = {table["title"]: table for table in tables}

    assert by_title["最优超参"]["rows"] == [["C", "0.0635"]]
    metric_rows = dict(by_title["最优 trial 指标"]["rows"])
    assert metric_rows["搜索阶段"] == "粗搜"
    assert metric_rows["头部 5% Lift"] == "1.6164"
    assert "各折测试集 KS 标准差" in metric_rows and "OOT 稳定性差距" in metric_rows
    [ranking] = [table for table in tables if table["title"].startswith("trials 排行")]
    assert "in-time" not in ranking["title"]
    assert "过拟合差（训练-测试）" in ranking["columns"] and "头部 5% Lift" in ranking["columns"]
    assert not any("gap（" in column or "lift5%" in column for column in ranking["columns"])


def test_leftover_metric_keys_have_business_names():
    assert metric_label("overfit_train_oot_gap") == "训练/OOT KS 差"
    assert metric_label("cv_fold_test_ks") == "交叉验证各折测试集 KS"
    assert metric_value_text("search_stage", "fine") == "细搜"


def test_model_reading_says_which_gap_set_the_overfit_flag():
    # Hold-out evidence without stored gaps: OOT KS beats train KS by 0.0959,
    # over the platform's 0.05 train/OOT threshold.
    context = build_workflow_insight_context(TASK_TYPE_MODELING, stage="gate", metadata={
        "model_delivery": {
            "recipe": "lr",
            "evaluation_evidence": {
                "basis": "pre_refit_heldout",
                "metrics": {"train_ks": 0.4893, "test_ks": 0.4710, "oot_ks": 0.5852, "overfit_flag": True},
            },
        },
    })

    assert context is not None
    assert "当前推荐模型为 逻辑回归。" in context["facts"]
    risks = "\n".join(context["risks"])
    assert "训练/OOT KS 相差 0.0959，超过 0.05" in risks
    assert "这不是典型的过拟合" in risks
    assert "overfit_flag" not in risks + "\n".join(context.get("headline") or [])
    measured = context["measured_evidence"]["metrics"]
    assert round(measured["overfit_train_oot_gap"], 4) == 0.0959
    assert round(measured["overfit_train_test_gap"], 4) == 0.0374


def test_refit_reports_say_which_one_is_the_delivered_model():
    from marvis.api_report_helpers import driver_report_download_metadata

    final, parent = "experiment_" + "a" * 32, "experiment_" + "b" * 32
    output = {
        "reports": [
            {"experiment_id": final, "recipe": "lr", "report_path": "/tmp/a.xlsx"},
            {"experiment_id": parent, "recipe": "lr", "report_path": "/tmp/b.xlsx"},
        ],
        "development_refit_scope": {"final_experiment_id": final, "parent_selection_experiment_id": parent},
    }
    items = driver_report_download_metadata(
        plan_id="plan-1", task_id="task-1", step_id="reports", output=output,
        default_label="下载模型开发报告",
    )
    assert [item["detail_label"] for item in items] == [
        "实验 1 · 逻辑回归（交付模型）",
        "实验 2 · 逻辑回归（重新拟合前的所选实验）",
    ]

    from marvis.agent.renderers import _render_reports

    _text, [table] = _render_reports(output)
    assert [row[0] for row in table["rows"]] == ["实验 1（交付模型）", "实验 2（重新拟合前的所选实验）"]


def test_report_readiness_names_missing_inputs_and_sections_in_chinese():
    from marvis.packs.modeling.report_compute import resolve_report_sections

    reasons = {status.section: status.reason for status in resolve_report_sections(None, None)}
    assert reasons["vintage"] == "缺少放款月份列、MOB 观察列"
    assert reasons["product_list"] == "缺少特征字典"

    root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { renderModelDeliveryPanel } from "./marvis/static/js/v2/model_delivery_panel.js";
        const html = renderModelDeliveryPanel({
          task_id: "task-1",
          metadata: { model_delivery: {
            source_tool: "post_training_action",
            report: { total_sections: 2, available_sections: 1, skipped_sections: 1, sections: [
              { section: "sample_analysis", available: true },
              { section: "low_pricing", available: false, reason: "缺少利率列" },
            ] },
          } },
        });
        assert.equal(html.includes(">样本分析<"), true);
        assert.equal(html.includes(">低定价压力测试<"), true);
        assert.equal(html.includes("sample_analysis"), false);
        process.stdout.write("ok");
        """
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=root, check=True, capture_output=True, text=True,
    )
    assert result.stdout == "ok"


def test_delivery_card_takes_algorithm_and_feature_count_from_the_model_card():
    from types import SimpleNamespace

    from marvis.agent.gate_payloads import build_model_delivery_payload
    from marvis.agent.renderers import _delivery_status
    from marvis.packs.modeling.delivery_tools import _challenger_comparison_recommendation

    dep = SimpleNamespace(id="step-delivery", title="模型交付动作",
                          tool_ref=SimpleNamespace(tool="post_training_action"))
    payload = build_model_delivery_payload({
        "artifact_id": "artifact-1",
        "actions": [],
        "model_card": {"recipe": "lr", "target_type": "binary", "feature_count": 6},
    }, dep)

    assert payload["recipe"] == "lr" and payload["target_type"] == "binary"
    assert payload["business_signals"]["feature_count"] == 6
    assert _delivery_status("warn") == "需复核" and _delivery_status("warning") == "需复核"
    text = _challenger_comparison_recommendation("warn", {"declined_count": 2})
    assert text == "挑战者模型有 2 项指标弱于冠军模型，需业务复核差异"


def test_delivery_panel_reads_warning_status_and_hides_empty_chips():
    root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { renderModelDeliveryPanel } from "./marvis/static/js/v2/model_delivery_panel.js";
        const html = renderModelDeliveryPanel({
          task_id: "task-1",
          metadata: { model_delivery: {
            source_tool: "post_training_action",
            recipe: "lr",
            readiness: [{ id: "approval_policy", label: "审批策略", status: "warning", reason: "需业务复核" }],
          } },
        });
        assert.equal(html.includes(">warning<"), false);
        assert.equal(html.includes("需复核"), true);
        assert.equal(html.includes("<span>目标</span><strong>-</strong>"), false);
        assert.equal(html.includes("<span>算法</span><strong>逻辑回归</strong>"), true);
        process.stdout.write("ok");
        """
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=root, check=True, capture_output=True, text=True,
    )
    assert result.stdout == "ok"



def test_reading_never_contradicts_the_platform_overfit_flag():
    # Review M2: an ad-hoc 0.05 absolute rule said 「存在过拟合迹象」 next to
    # 「平台未标记过拟合风险」.
    def reading(flag):
        metrics = {"train_ks": 0.60, "test_ks": 0.545, "oot_ks": 0.54}
        if flag is not None:
            metrics["overfit_flag"] = flag
        return build_workflow_insight_context(TASK_TYPE_MODELING, stage="done", metadata={
            "model_delivery": {"recipe": "lgb", "metrics": metrics},
        })

    clear = reading(False)
    assert "平台未标记过拟合风险。" in clear["headline"]
    assert not any("过拟合迹象" in risk for risk in clear["risks"])
    assert "train_test_ks_gap" not in clear["measured_evidence"]["risk_flags"]
    # No platform flag: the platform's relative rule (0.0917 ≤ 0.10) finds nothing either.
    assert not any("过拟合迹象" in risk for risk in reading(None)["risks"])


def test_refit_delivery_card_titles_fit_only_metrics_as_the_delivered_model():
    # Review M5: after a refit the grid's fit-only numbers were also titled
    # 「最终模型指标」 next to the hold-out evidence table.
    root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { renderModelDeliveryPanel } from "./marvis/static/js/v2/model_delivery_panel.js";
        const render = (refit) => renderModelDeliveryPanel({ task_id: "t", metadata: { model_delivery: {
          source_tool: "select_experiment", metrics: { train_ks: 0.4768, train_auc: 0.8166 },
          evaluation_evidence: { refit_applied: refit, basis: "pre_refit_heldout" },
        } } });
        assert.equal(render(true).includes("交付模型指标（train+test 重新拟合）"), true);
        assert.equal(render(true).includes(">最终模型指标<"), false);
        assert.equal(render(false).includes(">最终模型指标<"), true);
        process.stdout.write("ok");
        """
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=root, check=True, capture_output=True, text=True,
    )
    assert result.stdout == "ok"


def test_refit_drops_overfit_fields_measured_on_the_training_slice():
    # Review M3: the flag and train/test gap of a TUNE-4 refit come from the
    # same in-distribution slice as its dropped test metrics.
    from marvis.packs.modeling.delivery_tools import _is_refit_holdout_tainted_metric
    from marvis.packs.modeling.select_tools import _is_refit_holdout_tainted_key

    for check in (_is_refit_holdout_tainted_metric, _is_refit_holdout_tainted_key):
        assert check("overfit_flag") and check("overfit_train_test_gap") and check("test_ks")
        assert not check("oot_ks") and not check("overfit_train_oot_gap") and not check("train_ks")


def test_tuning_config_explanation_names_algorithms():
    # Review L3: the 「说明」 row read 「lr 使用有界随机搜索…按算法预算:lr=1」.
    from marvis.agent.presenters.display_labels import readable_recipe_text
    from marvis.agent.renderers import render_tool_output

    _text, tables = render_tool_output("configure_tuning", {
        "recipe": "lr", "recipes": ["lr"], "target_type": "binary", "tune_enabled": True, "n_trials": 1,
        "reason": "lr 使用有界随机搜索(预算>1时自动进入粗搜+细搜;按算法预算:lr=1;多算法总预算=Σ各配方预算=1 轮)。",
    })
    [config] = [table for table in tables if table["title"] == "调参配置"]
    explanation = dict(config["rows"])["说明"]
    assert explanation.startswith("逻辑回归 使用有界随机搜索") and "逻辑回归 1 轮" in explanation
    assert "lr" not in explanation
    # Codes inside longer identifiers stay as written.
    assert readable_recipe_text("flr=1 lgb_regressor") == "flr=1 LightGBM 回归"
