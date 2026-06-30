"""Static checks for the §4 interactive feature-screening selection table (FEAT-1).

The table is a thin consumer of the backend ``metadata.screen`` contract: it renders
the screened features with metric columns + checkboxes (pre-checked = the screen's
proposed set) and, on confirm, posts ``{content:"确认", selection:[...]}`` so the
backend overrides the screen step's selected set. These assertions pin the wiring so
a frontend rewrite can't silently drop it. (Kept in its own file, away from the
brand-treatment tests, to stay decoupled from unrelated frontend churn.)
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

_STATIC = Path(__file__).resolve().parent.parent / "marvis" / "static"


def _read(rel: str) -> str:
    return (_STATIC / rel).read_text(encoding="utf-8")


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=_STATIC.parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _app_slice(start: str, end: str) -> str:
    app_js = _read("app.js")
    start_index = app_js.index(start)
    end_index = app_js.index(end, start_index)
    return app_js[start_index:end_index]


def test_screen_table_renderer_and_manual_branch_are_wired():
    app_js = _read("app.js")
    # the interactive renderer exists and is dispatched for screen gate messages
    assert "function agentMessageScreenTableHtml(message, options = {})" in app_js
    assert "if (meta.screen)" in app_js
    assert "agentMessageScreenTableHtml(message, { interactive })" in app_js
    assert "latestInteractiveScreenMessageId(messages)" in app_js
    # it reads the structured screen payload the backend attaches
    assert "message?.metadata?.screen" in app_js
    # checkbox per feature, pre-checked from the proposed selected set
    assert 'class="screen-pick"' in app_js
    assert "screen.selected" in app_js


def test_screen_confirm_posts_edited_selection():
    app_js = _read("app.js")
    assert "function submitScreenSelection(button)" in app_js
    assert "data-screen-confirm" in app_js
    # collects checked, non-disabled features and posts them as `selection`
    # with the rendered gate token so stale tabs cannot confirm a newer gate.
    assert ".screen-pick:checked" in app_js
    assert '"content": "确认"' in app_js or 'content: "确认"' in app_js
    assert "selection" in app_js
    assert "expected_step_id" in app_js
    # a delegated document click handler drives it (mirrors the C1 form pattern)
    assert "handleScreenConfirmClick" in app_js


def test_screen_threshold_adjust_posts_structured_params():
    app_js = _read("app.js")
    assert 'class="screen-threshold-input"' in app_js
    assert "data-screen-threshold=\"leakage_ks\"" in app_js
    assert "data-screen-threshold=\"max_missing_rate\"" in app_js
    assert "function submitScreenThresholdAdjust(button)" in app_js
    assert "adjust_params" in app_js
    assert "handleScreenAdjustClick" in app_js
    assert 'class="screen-num"' in app_js
    assert "阈值不能为空" in app_js


def test_modeling_setup_weight_picker_renderer_and_branch_are_wired():
    app_js = _read("app.js")
    module_js = _read("js/v2/modeling_setup_panel.js")
    css = _read("css/v2-workbench.css")
    assert "submitModelingWeightAdjustController" in app_js
    assert "handleModelingWeightAdjustClickController" in app_js
    assert "function agentMessageModelingSetupHtml(message, options = {})" in app_js
    assert "return renderModelingSetupPanel(message, options);" in app_js
    assert "modelingSetupControllerContext()" in app_js
    assert "if (meta.modeling_setup)" in app_js
    assert "agentMessageModelingSetupHtml(message, { interactive })" in app_js
    assert "export function renderModelingSetupPanel(message, options = {})" in module_js
    assert "export async function submitModelingWeightAdjust(button, context = {})" in module_js
    assert "export function handleModelingWeightAdjustClick(event, context = {})" in module_js
    assert 'class="modeling-weight-pick"' in module_js
    assert "data-modeling-gate-step-id" in module_js
    assert "function submitModelingWeightAdjust(button)" in app_js
    assert "collectModelingSetupAdjustParams" in module_js
    assert "params.sample_weight_col" in module_js
    assert "modeling-target-select" in module_js
    assert "modeling-recipe-pick" in module_js
    assert "modeling-n-trials-input" in module_js
    assert "modeling-override-reason-input" in module_js
    assert "sample_weight_diagnostics" in module_js
    assert "override_guidance" in module_js
    assert "modeling-guidance-list" in module_js
    assert "modeling-weight-diagnostic" in module_js
    assert "modeling-spec-grid" in module_js
    assert "modeling-algorithm-grid" in module_js
    assert "modeling-split-summary" in module_js
    assert "handleModelingWeightAdjustClick" in app_js
    assert ".modeling-setup-panel" in css
    assert ".modeling-spec-grid" in css
    assert ".modeling-algorithm-grid" in css
    assert ".modeling-split-grid" in css
    assert ".modeling-setup-controls" in css
    assert ".modeling-guidance-list" in css
    assert ".modeling-guidance-item" in css
    assert ".modeling-recipe-options" in css
    assert ".modeling-weight-options" in css
    assert ".modeling-weight-diagnostics" in css


def test_modeling_setup_weight_picker_renders_candidates():
    output = _run_node(
        f"""
        {""}
        import assert from "node:assert/strict";
        import {{ renderModelingSetupPanel }} from "./marvis/static/js/v2/modeling_setup_panel.js";
        const html = renderModelingSetupPanel({{
          id: "m1",
          metadata: {{
            step_id: "gate-1",
            modeling_setup: {{
              target_type: "binary",
              recipe: "lgb",
              recipes: ["lgb", "xgb"],
              feature_count: 12,
              n_trials: 24,
              metric_policy: "oot_ks",
              eligible_algorithms: ["lgb", "xgb"],
              disabled_algorithms: [{{ recipe: "lgb_regressor", reason: "target mismatch" }}],
              pmml_supported_algorithms: ["lgb", "xgb"],
              warnings: ["样本权重列已从入模特征中移除。"],
              override_guidance: [
                {{ id: "target_type", label: "目标类型", level: "info", message: "二分类适合 0/1 风控标签。" }},
                {{ id: "sample_weight", label: "样本权重", level: "review", message: "权重列会改变拟合目标。" }},
              ],
              split_summary: {{
                split_col: "split",
                split_counts: {{ train: 80, test: 10, oot: 10 }},
                total_rows: 100,
                warnings: [],
              }},
              sample_weight_col: "weight",
              sample_weight_candidates: ["weight", "sample_weight"],
              sample_weight_diagnostics: [
                {{
                  column: "weight",
                  valid: true,
                  missing_rate: 0,
                  min: 1,
                  max: 2,
                  mean: 1.25,
                  reason: "",
                }},
              ],
            }},
          }},
        }});
        assert.equal(html.includes('data-modeling-gate-step-id="gate-1"'), true);
        assert.equal(html.includes('value="weight" checked'), true);
        assert.equal(html.includes("sample_weight"), true);
        assert.equal(html.includes("lgb/xgb"), true);
        assert.equal(html.includes("modeling-target-select"), true);
        assert.equal(html.includes("modeling-guidance-list"), true);
        assert.equal(html.includes("二分类适合 0/1 风控标签。"), true);
        assert.equal(html.includes('data-level="review"'), true);
        assert.equal(html.includes("modeling-recipe-pick"), true);
        assert.equal(html.includes("modeling-n-trials-input"), true);
        assert.equal(html.includes("变更原因"), true);
        assert.equal(html.includes("候选特征"), true);
        assert.equal(html.includes("24"), true);
        assert.equal(html.includes("PMML 可导出"), true);
        assert.equal(html.includes("target mismatch"), true);
        assert.equal(html.includes("样本切分 · split"), true);
        assert.equal(html.includes("TRAIN"), true);
        assert.equal(html.includes("样本权重列已从入模特征中移除。"), true);
        assert.equal(html.includes("modeling-weight-diagnostic"), true);
        assert.equal(html.includes("缺失 0.0%"), true);
        assert.equal(html.includes("范围 1-2"), true);
        assert.equal(html.includes("均值 1.25"), true);
        const readonly = renderModelingSetupPanel({{
          id: "m2",
          metadata: {{ step_id: "gate-2", modeling_setup: {{ sample_weight_candidates: ["weight"] }} }},
        }}, {{ interactive: false }});
        assert.equal(readonly.includes('data-modeling-readonly="true"'), true);
        assert.equal(readonly.includes("历史规格"), true);
        const noWeight = renderModelingSetupPanel({{
          id: "m3",
          metadata: {{
            step_id: "gate-3",
            modeling_setup: {{
              target_type: "continuous",
              recipe: "lgb_regressor",
              recipes: ["lgb_regressor"],
              feature_count: 5,
              n_trials: 6,
              metric_policy: "oot_rmse",
              eligible_algorithms: ["lgb_regressor"],
              pmml_supported_algorithms: [],
              sample_weight_candidates: [],
            }},
          }},
        }});
        assert.equal(noWeight.includes("建模规格"), true);
        assert.equal(noWeight.includes("continuous"), true);
        assert.equal(noWeight.includes("lgb_regressor"), true);
        assert.equal(noWeight.includes("不使用权重"), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_model_delivery_panel_renderer_and_branch_are_wired():
    app_js = _read("app.js")
    module_js = _read("js/v2/model_delivery_panel.js")
    css = _read("css/v2-workbench.css")
    assert 'import { renderModelDeliveryPanel } from "./js/v2/model_delivery_panel.js";' in app_js
    assert "function agentMessageModelDeliveryHtml(message, options = {})" in app_js
    assert "return renderModelDeliveryPanel(message, options);" in app_js
    assert "if (meta.model_delivery)" in app_js
    assert "agentMessageModelDeliveryHtml(message)" in app_js
    assert "export function renderModelDeliveryPanel(message, options = {})" in module_js
    assert "model-delivery-readiness-grid" in module_js
    assert "candidateTable(delivery.candidates)" in module_js
    assert "actionTable(delivery.actions)" in module_js
    assert "reportSummary(delivery.report)" in module_js
    assert "businessSignalSummary(delivery.business_signals)" in module_js
    assert "policySignalSummary(delivery.policy_signals)" in module_js
    assert "policyDecisionSummary(delivery.policy_decision)" in module_js
    assert "business_signals" in module_js
    assert "policy_signals" in module_js
    assert "policy_decision" in module_js
    assert ".model-delivery-panel" in css
    assert ".model-delivery-readiness-grid" in css
    assert ".model-delivery-business-grid" in css
    assert ".model-delivery-policy-grid" in css
    assert ".model-delivery-policy-card" in css
    assert ".model-delivery-table" in css
    assert ".model-delivery-status" in css
    assert ".model-delivery-report-summary" in css


def test_modeling_panels_use_semantic_model_visual_tokens():
    styles_css = _read("styles.css")
    v2_css = _read("css/v2-workbench.css")
    start = v2_css.index("/* Modeling setup gate controls. */")
    end = v2_css.index("/* §4 interactive feature-screening selection table", start)
    modeling_panel_css = v2_css[start:end]

    for token in [
        "--model-panel-border",
        "--model-panel-border-soft",
        "--model-panel-surface",
        "--model-panel-surface-soft",
        "--model-panel-text",
        "--model-panel-text-muted",
        "--model-signal-info",
        "--model-signal-ready",
        "--model-signal-warning",
        "--model-signal-error",
    ]:
        assert token in styles_css
        assert token in modeling_panel_css

    assert "#" not in modeling_panel_css
    assert "var(--success," not in modeling_panel_css
    assert "var(--warning," not in modeling_panel_css
    assert "var(--danger," not in modeling_panel_css


def test_model_delivery_panel_renders_selection_and_actions():
    output = _run_node(
        f"""
        {""}
        import assert from "node:assert/strict";
        import {{ renderModelDeliveryPanel }} from "./marvis/static/js/v2/model_delivery_panel.js";
        const html = renderModelDeliveryPanel({{
          id: "m-delivery",
          metadata: {{
            model_delivery: {{
              source_tool: "post_training_action",
              selected_experiment_id: "exp-lgb",
              artifact_id: "art-lgb",
              recipe: "lgb",
              target_type: "binary",
              selection_metric: "oot_ks",
              selection_reason: "按 oot_ks 在 PMML/验证移交可用候选中自动选择。",
              metrics: {{ oot_ks: 0.3123, test_ks: 0.2876, oot_auc: 0.721 }},
              business_signals: {{
                stability: "稳定",
                feature_count: 18,
                calibration: "已校准(PMML不含)",
                delivery: "可移交",
              }},
              policy_signals: {{
                scorecard: "评分卡",
                scorecard_status: "ready",
                monotonicity: "已约束",
                monotonicity_status: "ready",
                approval: "建议可审批",
                approval_status: "ready",
                reasons: ["评分卡表 12 行"],
              }},
              policy_decision: {{
                status: "overridden",
                policy: {{ require_pmml: true, require_handoff: true }},
                profile: {{ pmml_supported: false, handoff_supported: false, monotonicity_declared: true }},
                violations: [
                  {{ code: "require_pmml", message: "要求最终模型支持 PMML 导出,但该候选不支持。" }},
                ],
                override_reason: "本轮只验收原生模型。",
              }},
              readiness: [
                {{ id: "native_model", label: "原生模型", status: "ready", artifact: "/tmp/model.pkl" }},
                {{ id: "model_report", label: "模型报告", status: "partial", artifact: "/tmp/model_report.xlsx", reason: "报告章节 1/2 可生成" }},
                {{ id: "pmml", label: "PMML", status: "succeeded", artifact: "/tmp/model.pmml" }},
                {{ id: "validation_handoff", label: "验证移交", status: "succeeded", artifact: "task-validation" }},
                {{ id: "model_card", label: "模型卡", status: "ready", artifact: "/tmp/art-lgb.model_card.md", reason: "最终模型卡已生成" }},
                {{ id: "monitoring_policy", label: "监控策略", status: "pass", artifact: "/tmp/model.monitoring_policy.md", reason: "可进入常规监控" }},
                {{ id: "challenger_comparison", label: "Champion对比", status: "warn", artifact: "/tmp/art-lgb.champion_comparison.md", reason: "需业务复核差异" }},
                {{ id: "challenger_backtest", label: "Challenger/Backtest", status: "succeeded", artifact: "task-challenger" }},
                {{ id: "approval_policy", label: "审批策略", status: "ready", reason: "建议可审批" }},
              ],
              candidates: [
                {{
                  id: "exp-lgb",
                  recipe: "lgb",
                  selected: true,
                  metrics: {{ oot_ks: 0.3123, test_ks: 0.2876 }},
                  business_signals: {{ stability: "稳定", feature_count: 18, calibration: "已校准(PMML不含)", delivery: "可移交" }},
                  policy_signals: {{ scorecard: "评分卡", monotonicity: "已约束", approval: "建议可审批" }},
                  capabilities: {{ pmml_supported: true, handoff_supported: true, native_model_supported: true }},
                }},
                {{
                  id: "exp-mlp",
                  recipe: "mlp",
                  selected: false,
                  metrics: {{ oot_ks: 0.3321 }},
                  business_signals: {{ stability: "高风险", feature_count: 120, calibration: "未校准", delivery: "仅原生" }},
                  policy_signals: {{ scorecard: "非评分卡", monotonicity: "未声明", approval: "需业务复核", approval_status: "warning" }},
                  capabilities: {{ pmml_supported: false, handoff_supported: false, native_model_supported: true, reason: "仅原生模型" }},
                }},
              ],
              actions: [
                {{ action: "export_pmml", status: "succeeded", pmml_path: "/tmp/model.pmml" }},
                {{ action: "handoff_to_validation", status: "succeeded", validation_task_id: "task-validation" }},
                {{
                  action: "create_challenger_backtest",
                  status: "succeeded",
                  challenger_task_id: "task-challenger",
                  markdown_path: "/tmp/challenger_backtest_plan.md",
                }},
              ],
              native_model_path: "/tmp/model.pkl",
              pmml_path: "/tmp/model.pmml",
              validation_task_id: "task-validation",
              challenger_task_id: "task-challenger",
              challenger_package_path: "/tmp/challenger_backtest_plan.json",
              challenger_package_markdown_path: "/tmp/challenger_backtest_plan.md",
              approval_package_path: "/tmp/art-lgb.approval_package.json",
              approval_package_markdown_path: "/tmp/art-lgb.approval_package.md",
              model_card_path: "/tmp/art-lgb.model_card.json",
              model_card_markdown_path: "/tmp/art-lgb.model_card.md",
              model_card: {{ card_version: "model_card_v1" }},
              monitoring_policy_path: "/tmp/model.monitoring_policy.json",
              monitoring_policy_markdown_path: "/tmp/model.monitoring_policy.md",
              monitoring_policy: {{ status: "pass", recommendation: "可进入常规监控" }},
              challenger_comparison_path: "/tmp/art-lgb.champion_comparison.json",
              challenger_comparison_markdown_path: "/tmp/art-lgb.champion_comparison.md",
              challenger_comparison: {{ status: "warn", recommendation: "需业务复核差异" }},
              report: {{
                report_path: "/tmp/model_report.xlsx",
                available_sections: 1,
                total_sections: 2,
                skipped_sections: 1,
                status: "partial",
                sections: [
                  {{ section: "汇总", available: true }},
                  {{ section: "Vintage", available: false, reason: "缺少 MOB 列" }},
                ],
              }},
            }},
          }},
        }});
        assert.equal(html.includes("训练后交付"), true);
        assert.equal(html.includes("2 项需处理/不支持"), true);
        assert.equal(html.includes("exp-lgb"), true);
        assert.equal(html.includes("已选"), true);
        assert.equal(html.includes("0.3123"), true);
        assert.equal(html.includes("稳定性"), true);
        assert.equal(html.includes("特征数"), true);
        assert.equal(html.includes("校准"), true);
        assert.equal(html.includes("模型策略"), true);
        assert.equal(html.includes("策略执行"), true);
        assert.equal(html.includes("已人工放行"), true);
        assert.equal(html.includes("require_pmml: true"), true);
        assert.equal(html.includes("要求最终模型支持 PMML 导出"), true);
        assert.equal(html.includes("放行原因: 本轮只验收原生模型。"), true);
        assert.equal(html.includes("评分卡"), true);
        assert.equal(html.includes("单调性"), true);
        assert.equal(html.includes("审批建议"), true);
        assert.equal(html.includes("建议可审批"), true);
        assert.equal(html.includes("评分卡表 12 行"), true);
        assert.equal(html.includes("需业务复核"), true);
        assert.equal(html.includes("稳定"), true);
        assert.equal(html.includes("高风险"), true);
        assert.equal(html.includes("已校准(PMML不含)"), true);
        assert.equal(html.includes("仅原生"), true);
        assert.equal(html.includes("120.00"), true);
        assert.equal(html.includes("PMML"), true);
        assert.equal(html.includes("验证移交"), true);
        assert.equal(html.includes("Challenger/Backtest"), true);
        assert.equal(html.includes("创建Challenger/Backtest"), true);
        assert.equal(html.includes("模型卡"), true);
        assert.equal(html.includes("art-lgb.model_card.md"), true);
        assert.equal(html.includes("监控策略"), true);
        assert.equal(html.includes("model.monitoring_policy.md"), true);
        assert.equal(html.includes("可进入常规监控"), true);
        assert.equal(html.includes("Champion对比"), true);
        assert.equal(html.includes("art-lgb.champion_comparison.md"), true);
        assert.equal(html.includes("需复核"), true);
        assert.equal(html.includes("model.pmml"), true);
        assert.equal(html.includes("task-challenger"), true);
        assert.equal(html.includes("challenger_backtest_plan.md"), true);
        assert.equal(html.includes("审批包"), true);
        assert.equal(html.includes("approval_package.md"), true);
        assert.equal(html.includes("审批包JSON"), true);
        assert.equal(html.includes("approval_package.json"), true);
        assert.equal(html.includes("model_report.xlsx"), true);
        assert.equal(html.includes("报告就绪度"), true);
        assert.equal(html.includes("1/2 章节可生成"), true);
        assert.equal(html.includes("缺少 MOB 列"), true);
        assert.equal(html.includes("task-validation"), true);
        assert.equal(renderModelDeliveryPanel({{ metadata: {{}} }}), "");
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_modeling_panels_combined_dom_smoke_contract():
    output = _run_node(
        f"""
        {""}
        import assert from "node:assert/strict";
        import {{ renderModelingSetupPanel }} from "./marvis/static/js/v2/modeling_setup_panel.js";
        import {{ renderModelDeliveryPanel }} from "./marvis/static/js/v2/model_delivery_panel.js";
        const longPath = "/tmp/" + "very-long-artifact-name-".repeat(12) + "model.pmml";
        const setupHtml = renderModelingSetupPanel({{
          id: "setup-smoke",
          metadata: {{
            step_id: "gate-setup",
            modeling_setup: {{
              target_type: "binary",
              recipe: "lgb",
              recipes: ["lgb", "xgb"],
              feature_count: 128,
              n_trials: 32,
              metric_policy: "oot_ks",
              eligible_algorithms: ["lgb", "xgb", "lr"],
              disabled_algorithms: [
                {{ recipe: "lgb_regressor", reason: "recipe target family does not match `binary`" }},
                {{ recipe: "lgb_multiclass", reason: "recipe target family does not match `binary`" }},
              ],
              pmml_supported_algorithms: ["lgb", "xgb", "lr"],
              override_guidance: [
                {{ id: "recipes", label: "算法组合", level: "info", message: "当前算法均可交付。" }},
                {{ id: "n_trials", label: "调参预算", level: "warning", message: "调参轮数较高。" }},
              ],
              sample_weight_col: "",
              sample_weight_candidates: ["weight"],
              sample_weight_diagnostics: [
                {{ column: "weight", valid: true, missing_rate: 0, min: 0.5, max: 2.0, mean: 1.0 }},
              ],
            }},
          }},
        }});
        const deliveryHtml = renderModelDeliveryPanel({{
          metadata: {{
            model_delivery: {{
              source_tool: "select_experiment",
              selected_experiment_id: "exp-lgb",
              artifact_id: "art-lgb",
              recipe: "lgb",
              target_type: "binary",
              selection_metric: "oot_ks",
              business_signals: {{ stability: "关注", feature_count: 128, calibration: "需说明", delivery: "可移交" }},
              policy_signals: {{ scorecard: "非评分卡", monotonicity: "未声明", approval: "仅实验候选" }},
              policy_decision: {{
                status: "accepted",
                policy: {{ require_pmml: true, require_handoff: true }},
                profile: {{ pmml_supported: true, handoff_supported: true, monotonicity_declared: false }},
                violations: [],
              }},
              readiness: [
                {{ id: "native_model", label: "原生模型", status: "ready", artifact: "/tmp/model.pkl" }},
                {{ id: "pmml", label: "PMML", status: "succeeded", artifact: longPath }},
              ],
              metrics: {{ oot_ks: 0.31, test_ks: 0.29, psi_oot_vs_train: 0.12 }},
              candidates: [
                {{
                  id: "exp-lgb",
                  recipe: "lgb",
                  selected: true,
                  metrics: {{ oot_ks: 0.31, test_ks: 0.29, psi_oot_vs_train: 0.12 }},
                  business_signals: {{ stability: "关注", feature_count: 128, calibration: "需说明", delivery: "可移交" }},
                  policy_signals: {{ scorecard: "非评分卡", monotonicity: "未声明", approval: "仅实验候选" }},
                  capabilities: {{ pmml_supported: true, handoff_supported: true, native_model_supported: true }},
                }},
              ],
              pmml_path: longPath,
              challenger_task_id: "task-challenger",
              challenger_package_markdown_path: "/tmp/challenger_backtest_plan.md",
              monitoring_policy_markdown_path: "/tmp/model.monitoring_policy.md",
              approval_package_path: "/tmp/art-lgb.approval_package.json",
              approval_package_markdown_path: "/tmp/art-lgb.approval_package.md",
            }},
          }},
        }});
        const html = setupHtml + deliveryHtml;
        for (const fragment of [
          "modeling-setup-panel",
          "modeling-guidance-list",
          "modeling-target-select",
          "modeling-recipe-pick",
          "modeling-override-reason-input",
          "model-delivery-panel",
          "model-delivery-business-grid",
          "model-delivery-policy-grid",
          "data-policy-decision-status",
          "model-delivery-table-wrap",
          "model-delivery-artifacts",
        ]) {{
          assert.equal(html.includes(fragment), true, fragment);
        }}
        assert.equal(html.includes("undefined"), false);
        assert.equal(html.includes("NaN"), false);
        assert.equal(html.includes(longPath.slice(-69)), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_modeling_setup_weight_adjust_posts_structured_params():
    output = _run_node(
        f"""
        {""}
        import assert from "node:assert/strict";
        let agentMessages = [];
        let rendered = 0;
        const statuses = [];
        const calls = [];
        import {{
          handleModelingWeightAdjustClick,
          submitModelingWeightAdjust,
        }} from "./marvis/static/js/v2/modeling_setup_panel.js";
        const context = {{
          getSelectedTaskId: () => "task-1",
          agentAcceptanceModeValue: () => "manual",
          setActionStatus: (message, kind) => statuses.push([message, kind]),
          setAgentMessages: (messages) => {{ agentMessages = messages || agentMessages; }},
          renderAgentConversation: () => {{ rendered += 1; }},
          api: async (url, options) => {{
            calls.push([url, JSON.parse(options.body)]);
            return {{ messages: [{{ id: "m2" }}] }};
          }},
        }};
        function makeForm({{
          currentWeight = "",
          pickedWeight = "",
          currentTargetType = "binary",
          targetType = "binary",
          currentTrials = "12",
          trials = "12",
          currentRecipes = "lgb",
          selectedRecipes = ["lgb"],
          reason = "",
        }} = {{}}) {{
          const recipeInputs = selectedRecipes.map((value) => ({{ value }}));
          return {{
            dataset: {{ modelingGateStepId: "gate-modeling", modelingCurrentWeight: currentWeight }},
            querySelector: (selector) => {{
              if (selector === ".modeling-target-select") {{
                return {{ value: targetType, getAttribute: (name) => name === "data-current-target-type" ? currentTargetType : "" }};
              }}
              if (selector === ".modeling-n-trials-input") {{
                return {{ value: trials, getAttribute: (name) => name === "data-current-n-trials" ? currentTrials : "" }};
              }}
              if (selector === ".modeling-recipe-control") {{
                return {{
                  dataset: {{ currentRecipes }},
                  querySelectorAll: (innerSelector) => innerSelector === ".modeling-recipe-pick:checked" ? recipeInputs : [],
                }};
              }}
              if (selector === ".modeling-weight-pick:checked") return {{ value: pickedWeight }};
              if (selector === ".modeling-override-reason-input") return {{ value: reason }};
              return null;
            }},
          }};
        }}
        await submitModelingWeightAdjust({{ disabled: false, closest: () => makeForm({{ pickedWeight: "weight" }}) }}, context);
        assert.equal(calls[0][0], "/api/tasks/task-1/agent/messages");
        assert.deepEqual(calls[0][1].adjust_params, {{ sample_weight_col: "weight" }});
        assert.equal(calls[0][1].expected_step_id, "gate-modeling");
        assert.equal(calls[0][1].acceptance_mode, "manual");
        assert.deepEqual(agentMessages, [{{ id: "m2" }}]);
        assert.equal(rendered, 1);

        await submitModelingWeightAdjust({{ disabled: false, closest: () => makeForm({{ currentWeight: "weight", pickedWeight: "weight" }}) }}, context);
        assert.deepEqual(statuses.at(-1), ["建模设置未变化。", "info"]);
        await submitModelingWeightAdjust({{ disabled: false, closest: () => makeForm({{
          currentRecipes: "lgb",
          selectedRecipes: [],
          reason: "清空算法",
        }}) }}, context);
        assert.deepEqual(statuses.at(-1), ["请至少选择一个训练算法。", "error"]);
        await submitModelingWeightAdjust({{ disabled: false, closest: () => makeForm({{
          targetType: "continuous",
          selectedRecipes: ["lgb"],
          currentRecipes: "lgb",
          reason: "目标改为连续",
        }}) }}, context);
        assert.deepEqual(statuses.at(-1), ["目标类型 continuous 与算法 lgb 不匹配,请选择同一目标类型的算法。", "error"]);
        await submitModelingWeightAdjust({{ disabled: false, closest: () => makeForm({{
          targetType: "continuous",
          selectedRecipes: ["lgb_regressor"],
          currentRecipes: "lgb",
          trials: "20",
          reason: "",
        }}) }}, context);
        assert.deepEqual(statuses.at(-1), ["调整目标类型、算法或调参轮数时请填写变更原因。", "error"]);
        await submitModelingWeightAdjust({{ disabled: false, closest: () => makeForm({{
          targetType: "continuous",
          selectedRecipes: ["lgb_regressor"],
          currentRecipes: "lgb",
          trials: "20",
          reason: "目标是连续金额预测",
        }}) }}, context);
        assert.deepEqual(calls.at(-1)[1].adjust_params, {{
          target_type: "continuous",
          n_trials: 20,
          recipes: ["lgb_regressor"],
        }});
        assert.equal(calls.at(-1)[1].content, "调整建模规格：目标是连续金额预测");
        const eventCalls = [];
        const eventContext = {{ ...context, api: async (url, options) => {{
          eventCalls.push([url, JSON.parse(options.body)]);
          return {{ messages: [] }};
        }} }};
        const eventForm = {{
          dataset: {{ modelingGateStepId: "gate-modeling", modelingCurrentWeight: "" }},
          querySelector: (selector) => {{
            if (selector === ".modeling-target-select") return {{ value: "binary", getAttribute: () => "binary" }};
            if (selector === ".modeling-n-trials-input") return {{ value: "12", getAttribute: () => "12" }};
            if (selector === ".modeling-recipe-control") return {{
              dataset: {{ currentRecipes: "lgb" }},
              querySelectorAll: () => [{{ value: "lgb" }}],
            }};
            if (selector === ".modeling-weight-pick:checked") return {{ value: "sample_weight" }};
            if (selector === ".modeling-override-reason-input") return {{ value: "" }};
            return null;
          }},
        }};
        await handleModelingWeightAdjustClick({{
          target: {{
            closest: () => ({{
              disabled: false,
              closest: () => eventForm,
            }}),
          }},
          preventDefault: () => statuses.push(["prevented", "event"]),
        }}, eventContext);
        assert.deepEqual(statuses.at(-1), ["prevented", "event"]);
        assert.deepEqual(eventCalls[0][1].adjust_params, {{ sample_weight_col: "sample_weight" }});
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_has_hardcut_coloring_styles():
    css = _read("css/v2-workbench.css")
    assert ".screen-table" in css
    # hard-cut buckets are visually distinguished (leakage / suspected / unusable)
    assert ".screen-row.screen-leakage" in css
    assert ".screen-row.screen-suspected" in css
    assert ".screen-badge" in css
    assert ".screen-threshold-controls" in css
    assert ".screen-table td.screen-num" in css
    assert '.screen-table-wrap[data-screen-readonly="true"]' in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "text-align: right" in css


def test_screen_table_only_latest_gate_is_interactive():
    screen_slice = _app_slice("function screenNum", "async function submitScreenThresholdAdjust")
    manual_slice = _app_slice("function stripChatInstructions", "function renderDriverManualAnalysis")
    output = _run_node(
        f"""
        import assert from "node:assert/strict";
        function escapeHtml(value) {{
          return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
        }}
        function renderAgentMarkdown(value) {{ return String(value || ""); }}
        function agentMessageC1FormHtml() {{ return ""; }}
        function agentMessageDedupPickerHtml() {{ return ""; }}
        function agentMessageModelingSetupHtml() {{ return ""; }}
        function agentMessageTablesHtml() {{ return ""; }}
        {screen_slice}
        {manual_slice}
        const messages = [
          {{
            id: "old-screen",
            role: "assistant",
            content: "old",
            metadata: {{ kind: "gate", step_id: "gate-old", screen: {{ selected: ["x1"], thresholds: {{ leakage_ks: 0.4, max_missing_rate: 0.95 }} }} }},
          }},
          {{
            id: "latest-screen",
            role: "assistant",
            content: "latest",
            metadata: {{ kind: "gate", step_id: "gate-new", screen: {{ selected: ["x2"], thresholds: {{ leakage_ks: 0.35, max_missing_rate: 0.9 }} }} }},
          }},
        ];
        const html = driverManualAnalysisHtml(messages);
        assert.equal(html.includes('data-screen-step-id="gate-new"'), true);
        assert.equal((html.match(/data-screen-readonly/g) || []).length, 1);
        assert.equal((html.match(/data-screen-adjust/g) || []).length, 1);
        assert.equal(html.includes("已归档"), true);
        assert.equal(html.includes("确认所选特征"), true);
        const laterGateHtml = driverManualAnalysisHtml([
          ...messages,
          {{ id: "later-gate", role: "assistant", content: "later", metadata: {{ kind: "gate", tables: [] }} }},
        ]);
        assert.equal((laterGateHtml.match(/data-screen-readonly/g) || []).length, 2);
        assert.equal((laterGateHtml.match(/data-screen-adjust/g) || []).length, 0);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_threshold_adjust_rejects_empty_and_posts_valid_payload():
    adjust_slice = _app_slice("async function submitScreenThresholdAdjust", "async function submitScreenSelection")
    output = _run_node(
        f"""
        import assert from "node:assert/strict";
        let selectedTaskId = "task-1";
        let agentMessages = [];
        const statuses = [];
        const calls = [];
        function setActionStatus(message, kind) {{ statuses.push([message, kind]); }}
        function agentAcceptanceModeValue() {{ return "manual"; }}
        function renderAgentConversation() {{}}
        async function api(url, options) {{
          calls.push([url, JSON.parse(options.body)]);
          return {{ messages: [{{ id: "m2" }}] }};
        }}
        {adjust_slice}
        const emptyWrap = {{
          dataset: {{}},
          querySelectorAll: () => [
            {{ getAttribute: (name) => name === "data-screen-threshold" ? "leakage_ks" : null, value: "" }},
          ],
        }};
        await submitScreenThresholdAdjust({{ disabled: false, closest: () => emptyWrap }});
        assert.deepEqual(calls, []);
        assert.deepEqual(statuses.at(-1), ["阈值不能为空。", "error"]);

        const validButton = {{ disabled: false, closest: () => ({{
          dataset: {{ screenStepId: "gate-screen" }},
          querySelectorAll: () => [
            {{ getAttribute: (name) => name === "data-screen-threshold" ? "leakage_ks" : null, value: "0.33" }},
            {{ getAttribute: (name) => name === "data-screen-threshold" ? "max_missing_rate" : null, value: "0.91" }},
          ],
        }}) }};
        await submitScreenThresholdAdjust(validButton);
        assert.equal(validButton.disabled, true);
        assert.equal(calls[0][0], "/api/tasks/task-1/agent/messages");
        assert.deepEqual(calls[0][1].adjust_params, {{ leakage_ks: 0.33, max_missing_rate: 0.91 }});
        assert.equal(calls[0][1].expected_step_id, "gate-screen");
        assert.equal(calls[0][1].acceptance_mode, "manual");
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_dedup_picker_renderer_and_branch_are_wired():
    app_js = _read("app.js")
    assert "function agentMessageDedupPickerHtml(message)" in app_js
    assert "if (meta.dedup)" in app_js
    assert "message?.metadata?.dedup" in app_js
    # a first/last strategy <select> per conflicting feature
    assert 'class="dedup-strategy"' in app_js
    assert "data-dedup-feature" in app_js


def test_dedup_picker_posts_strategies():
    app_js = _read("app.js")
    assert "function submitDedupStrategies(button)" in app_js
    assert "data-dedup-confirm" in app_js
    assert "dedup_strategies" in app_js
    assert "data-dedup-gate-step-id" in app_js
    assert "expected_step_id: expectedStepId" in app_js
    assert "handleDedupConfirmClick" in app_js
    css = _read("css/v2-workbench.css")
    assert ".dedup-picker" in css and ".dedup-table" in css


def test_capability_tier_picker_is_wired():
    """TIER-IA (spec §5.1): the create dialog exposes a per-task capability-tier
    selector (the previously-missing entry point), collected into payload."""
    index_html = _read("index.html")
    app_js = _read("app.js")
    assert 'id="createTaskTier"' in index_html
    for tier in ("conservative", "balanced", "autonomous"):
        assert f'value="{tier}"' in index_html
    assert "createTaskTierField" in app_js
    assert "tierField" in app_js  # gated to driver task types
    assert "syncCreateTaskTierDefault" in app_js
    assert "getSelectedTier()" in app_js
    assert "payload.capability_tier" in app_js
