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
    module_js = _read("js/v2/screen_gate_controller.js")
    manual_module_js = _read("js/v2/driver_manual_analysis.js")
    # the interactive renderer exists and is dispatched for screen gate messages
    assert 'renderScreenGateTable' in app_js
    assert "function agentMessageScreenTableHtml(message, options = {})" in app_js
    assert "return renderScreenGateTable(message, options);" in app_js
    assert "if (meta.screen)" in manual_module_js
    assert "renderScreenTable(message, { interactive })" in manual_module_js
    assert "latestInteractiveScreenMessageIdController(messages)" in app_js
    assert "export function latestInteractiveScreenMessageId(messages = [])" in manual_module_js
    # it reads the structured screen payload the backend attaches
    assert "export function renderScreenGateTable(message, options = {})" in module_js
    assert "message?.metadata?.screen" in module_js
    # checkbox per feature, pre-checked from the proposed selected set
    assert 'class="screen-pick"' in module_js
    assert "screen.selected" in module_js


def test_screen_confirm_posts_edited_selection():
    app_js = _read("app.js")
    module_js = _read("js/v2/screen_gate_controller.js")
    assert "function submitScreenSelection(button)" in app_js
    assert "submitScreenSelectionController(button, screenGateControllerContext())" in app_js
    assert "export async function submitScreenSelection(button, rawContext = {})" in module_js
    assert "data-screen-confirm" in module_js
    # collects checked, non-disabled features and posts them as `selection`
    # with the rendered gate token so stale tabs cannot confirm a newer gate.
    assert ".screen-pick:checked" in module_js
    assert '"确认"' in module_js
    assert "selection" in module_js
    assert "expected_step_id" in module_js
    # UX-4: a leakage/suspected pick requires a written override reason, folded
    # into the confirm content (no backend schema change) before submission.
    assert "screen-leakage-reason-input" in module_js
    assert "泄漏/疑似列覆盖理由" in module_js
    # a delegated document click handler drives it (mirrors the C1 form pattern)
    assert "handleScreenConfirmClick" in app_js
    # UX-1/REL-1: submission now shows immediate busy feedback and keeps the
    # agent-message stream + plan rail live while the driver turn (job-wrapped)
    # runs, instead of freezing until the request resolves.
    assert '"正在执行下一步…", "busy"' in module_js
    assert "pollAgentMessagesUntilSettled" in module_js
    assert "resetFetchThrottle" in module_js
    assert "renderWorkflowStepper" in module_js


def test_screen_threshold_adjust_posts_structured_params():
    app_js = _read("app.js")
    module_js = _read("js/v2/screen_gate_controller.js")
    assert 'class="screen-threshold-input"' in module_js
    assert "data-screen-threshold=\"leakage_ks\"" in module_js
    assert "data-screen-threshold=\"max_missing_rate\"" in module_js
    assert "function submitScreenThresholdAdjust(button)" in app_js
    assert "submitScreenThresholdAdjustController(button, screenGateControllerContext())" in app_js
    assert "export async function submitScreenThresholdAdjust(button, rawContext = {})" in module_js
    assert "adjust_params" in module_js
    assert "handleScreenAdjustClick" in app_js
    assert 'class="screen-num"' in module_js
    assert "阈值不能为空" in module_js
    # UX-1/REL-1: adjust submission shares the same busy-feedback contract as
    # the confirm path (immediate busy pill + streamed poll + plan rail ticks).
    assert '"正在执行下一步…", "busy"' in module_js
    assert "pollAgentMessagesUntilSettled" in module_js
    assert "resetFetchThrottle" in module_js
    assert "renderWorkflowStepper" in module_js


def test_modeling_setup_weight_picker_renderer_and_branch_are_wired():
    app_js = _read("app.js")
    module_js = _read("js/v2/modeling_setup_panel.js")
    manual_module_js = _read("js/v2/driver_manual_analysis.js")
    css = _read("css/v2-workbench.css")
    assert "submitModelingWeightAdjustController" in app_js
    assert "handleModelingWeightAdjustClickController" in app_js
    assert "function agentMessageModelingSetupHtml(message, options = {})" in app_js
    assert "return renderModelingSetupPanel(message, options);" in app_js
    assert "modelingSetupControllerContext()" in app_js
    assert "if (meta.modeling_setup)" in manual_module_js
    assert "renderModelingSetup(message, { interactive })" in manual_module_js
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
    manual_module_js = _read("js/v2/driver_manual_analysis.js")
    css = _read("css/v2-workbench.css")
    assert 'import { renderModelDeliveryPanel } from "./js/v2/model_delivery_panel.js";' in app_js
    assert "function agentMessageModelDeliveryHtml(message, options = {})" in app_js
    assert "return renderModelDeliveryPanel(message, options);" in app_js
    assert "if (meta.model_delivery)" in manual_module_js
    assert "renderModelDelivery(message)" in manual_module_js
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
        // UX-1/REL-1: context supplies minimal, call-counting stubs for the
        // busy-feedback capabilities (poll/reset/stepper) so the plan-rail
        // "finally" refresh can be asserted alongside the success payload.
        let pollCalls = 0;
        let resetCalls = 0;
        let stepperCalls = 0;
        context.pollAgentMessagesUntilSettled = async () => {{ pollCalls += 1; }};
        context.resetFetchThrottle = () => {{ resetCalls += 1; }};
        context.renderWorkflowStepper = () => {{ stepperCalls += 1; }};

        const firstButton = {{ disabled: false, closest: () => makeForm({{ pickedWeight: "weight" }}) }};
        await submitModelingWeightAdjust(firstButton, context);
        // busy state is pushed synchronously before the request settles
        assert.deepEqual(statuses[0], ["正在执行下一步…", "busy"]);
        assert.equal(calls[0][0], "/api/tasks/task-1/agent/messages");
        assert.deepEqual(calls[0][1].adjust_params, {{ sample_weight_col: "weight" }});
        assert.equal(calls[0][1].expected_step_id, "gate-modeling");
        assert.equal(calls[0][1].acceptance_mode, "manual");
        assert.deepEqual(agentMessages, [{{ id: "m2" }}]);
        assert.equal(rendered, 1);
        // success path: message list + conversation render are updated, and
        // the plan rail is force-refreshed at least once via the finally path
        assert.equal(pollCalls, 1);
        assert.equal(resetCalls >= 1, true);
        assert.equal(stepperCalls >= 1, true);
        // success path leaves the button disabled (no re-enable on success)
        assert.equal(firstButton.disabled, true);

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
        const statusesBeforeStructuralAdjust = statuses.length;
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
        // this successful submission pushes the busy status before resolving
        assert.deepEqual(statuses[statusesBeforeStructuralAdjust], ["正在执行下一步…", "busy"]);

        const eventCalls = [];
        let eventPollCalls = 0;
        let eventResetCalls = 0;
        let eventStepperCalls = 0;
        const eventContext = {{
          ...context,
          api: async (url, options) => {{
            eventCalls.push([url, JSON.parse(options.body)]);
            return {{ messages: [] }};
          }},
          pollAgentMessagesUntilSettled: async () => {{ eventPollCalls += 1; }},
          resetFetchThrottle: () => {{ eventResetCalls += 1; }},
          renderWorkflowStepper: () => {{ eventStepperCalls += 1; }},
        }};
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
        // handleModelingWeightAdjustClick returns the underlying submit promise,
        // so awaiting it here lets us observe the full busy -> settle sequence.
        await handleModelingWeightAdjustClick({{
          target: {{
            closest: () => ({{
              disabled: false,
              closest: () => eventForm,
            }}),
          }},
          preventDefault: () => statuses.push(["prevented", "event"]),
        }}, eventContext);
        // preventDefault fires synchronously before the busy status is pushed
        const preventedIndex = statuses.findIndex((entry) => entry[0] === "prevented");
        const busyIndex = statuses.findIndex((entry, index) => index > preventedIndex && entry[1] === "busy");
        assert.equal(preventedIndex >= 0, true);
        assert.equal(busyIndex > preventedIndex, true);
        assert.deepEqual(eventCalls[0][1].adjust_params, {{ sample_weight_col: "sample_weight" }});
        // finally-path plan-rail refresh fires at least once for the event-driven submit too
        assert.equal(eventPollCalls, 1);
        assert.equal(eventResetCalls >= 1, true);
        assert.equal(eventStepperCalls >= 1, true);
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
    output = _run_node(
        f"""
        {""}
        import assert from "node:assert/strict";
        import {{ driverManualAnalysisHtml }} from "./marvis/static/js/v2/driver_manual_analysis.js";
        import {{ renderScreenGateTable }} from "./marvis/static/js/v2/screen_gate_controller.js";
        function renderAgentMarkdown(value) {{ return String(value || ""); }}
        function agentMessageC1FormHtml() {{ return ""; }}
        function agentMessageDedupPickerHtml() {{ return ""; }}
        function agentMessageModelingSetupHtml() {{ return ""; }}
        function agentMessageTablesHtml() {{ return ""; }}
        function agentMessageScreenTableHtml(message, options = {{}}) {{
          return renderScreenGateTable(message, options);
        }}
        const renderers = {{
          renderAgentMarkdown,
          renderC1Form: agentMessageC1FormHtml,
          renderDedupPicker: agentMessageDedupPickerHtml,
          renderModelingSetup: agentMessageModelingSetupHtml,
          renderScreenTable: agentMessageScreenTableHtml,
          renderTables: agentMessageTablesHtml,
          renderModelDelivery: () => "",
        }};
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
        const html = driverManualAnalysisHtml(messages, renderers);
        assert.equal(html.includes('data-screen-step-id="gate-new"'), true);
        assert.equal((html.match(/data-screen-readonly/g) || []).length, 1);
        assert.equal((html.match(/data-screen-adjust/g) || []).length, 1);
        assert.equal(html.includes("已归档"), true);
        assert.equal(html.includes("确认所选特征"), true);
        const laterGateHtml = driverManualAnalysisHtml([
          ...messages,
          {{ id: "later-gate", role: "assistant", content: "later", metadata: {{ kind: "gate", tables: [] }} }},
        ], renderers);
        assert.equal((laterGateHtml.match(/data-screen-readonly/g) || []).length, 2);
        assert.equal((laterGateHtml.match(/data-screen-adjust/g) || []).length, 0);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_threshold_adjust_rejects_empty_and_posts_valid_payload():
    output = _run_node(
        f"""
        {""}
        import assert from "node:assert/strict";
        import {{ submitScreenThresholdAdjust }} from "./marvis/static/js/v2/screen_gate_controller.js";
        let agentMessages = [];
        let rendered = 0;
        const statuses = [];
        const calls = [];
        const context = {{
          selectedTaskId: "task-1",
          agentAcceptanceModeValue: () => "manual",
          setActionStatus: (message, kind) => statuses.push([message, kind]),
          setAgentMessages: (messages) => {{ agentMessages = messages || agentMessages; }},
          renderAgentConversation: () => {{ rendered += 1; }},
          api: async (url, options) => {{
            calls.push([url, JSON.parse(options.body)]);
            return {{ messages: [{{ id: "m2" }}] }};
          }},
        }};
        const emptyWrap = {{
          dataset: {{}},
          querySelectorAll: () => [
            {{ getAttribute: (name) => name === "data-screen-threshold" ? "leakage_ks" : null, value: "" }},
          ],
        }};
        await submitScreenThresholdAdjust({{ disabled: false, closest: () => emptyWrap }}, context);
        assert.deepEqual(calls, []);
        assert.deepEqual(statuses.at(-1), ["阈值不能为空。", "error"]);

        const validButton = {{ disabled: false, closest: () => ({{
          dataset: {{ screenStepId: "gate-screen" }},
          querySelectorAll: () => [
            {{ getAttribute: (name) => name === "data-screen-threshold" ? "leakage_ks" : null, value: "0.33" }},
            {{ getAttribute: (name) => name === "data-screen-threshold" ? "max_missing_rate" : null, value: "0.91" }},
          ],
        }}) }};
        await submitScreenThresholdAdjust(validButton, context);
        assert.equal(validButton.disabled, true);
        assert.equal(calls[0][0], "/api/tasks/task-1/agent/messages");
        assert.deepEqual(calls[0][1].adjust_params, {{ leakage_ks: 0.33, max_missing_rate: 0.91 }});
        assert.equal(calls[0][1].expected_step_id, "gate-screen");
        assert.equal(calls[0][1].acceptance_mode, "manual");
        assert.deepEqual(agentMessages, [{{ id: "m2" }}]);
        assert.equal(rendered, 1);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_gains_search_sort_chips_bulk_pagination_wiring():
    app_js = _read("app.js")
    module_js = _read("js/v2/screen_gate_controller.js")
    css = _read("css/v2-workbench.css")
    # UX-4: search box, sortable headers, category chips, bulk ops, pagination.
    assert "screen-search-input" in module_js
    assert "data-screen-search" in module_js
    assert "data-screen-sort" in module_js
    assert "data-screen-chip" in module_js
    assert "data-screen-bulk" in module_js
    assert "data-screen-page-prev" in module_js
    assert "data-screen-page-next" in module_js
    assert "screen-selected-count" in module_js
    assert "export function handleScreenSearchInput(event, context = {})" in module_js
    assert "export function handleScreenSortClick(event, context = {})" in module_js
    assert "export function handleScreenChipClick(event, context = {})" in module_js
    assert "export function handleScreenPageClick(event, context = {})" in module_js
    assert "export function handleScreenBulkClick(event, context = {})" in module_js
    assert "export function handleScreenPickChange(event, context = {})" in module_js
    # wired into the app shell via delegated document listeners
    assert 'document.addEventListener("click", handleScreenSortClick)' in app_js
    assert 'document.addEventListener("click", handleScreenChipClick)' in app_js
    assert 'document.addEventListener("click", handleScreenPageClick)' in app_js
    assert 'document.addEventListener("click", handleScreenBulkClick)' in app_js
    assert 'document.addEventListener("input", handleScreenSearchInput)' in app_js
    assert 'document.addEventListener("change", handleScreenPickChange)' in app_js
    assert "getAgentMessages: () => agentMessages" in app_js
    # CSS for the new toolbar/summary/bulk/pagination affordances
    assert ".screen-toolbar" in css
    assert ".screen-chip" in css
    assert ".screen-summary" in css
    assert ".screen-bulk-actions" in css
    assert ".screen-pagination" in css
    assert ".screen-sort-btn" in css


def test_screen_table_search_filters_by_feature_name():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { handleScreenSearchInput } from "./marvis/static/js/v2/screen_gate_controller.js";

        function makeScreen() {
          const scores = {};
          const selected = [];
          for (let i = 0; i < 5; i++) {
            const name = "alpha_feature_" + i;
            selected.push(name);
            scores[name] = { ks: 0.1 * i, iv: 0.05 * i };
          }
          selected.push("beta_other");
          scores.beta_other = { ks: 0.2, iv: 0.1 };
          return {
            selected,
            leakage: [],
            suspected: [],
            unusable: [],
            scores,
            n_screened: selected.length,
            thresholds: { leakage_ks: 0.4, max_missing_rate: 0.95 },
          };
        }
        const message = { id: "search-msg", metadata: { step_id: "gate-search", screen: makeScreen() } };
        let lastHtml = null;
        const context = {
          getAgentMessages: () => [message],
          applyRerender: (wrap, html) => { lastHtml = html; },
        };
        const fakeInput = { value: "alpha" };
        const fakeWrap = {
          dataset: { screenForm: "search-msg", screenReadonly: "false" },
          querySelector: (sel) => (sel === ".screen-search-input" ? fakeInput : null),
          querySelectorAll: () => [],
        };
        fakeInput.closest = (sel) => (sel === "[data-screen-search]" ? fakeInput : fakeWrap);
        const handled = handleScreenSearchInput({ target: fakeInput }, context);
        assert.equal(handled, true);
        assert.equal((lastHtml.match(/class="screen-row/g) || []).length, 5);
        assert.equal(lastHtml.includes("beta_other"), false);
        assert.equal(lastHtml.includes("alpha_feature_0"), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_sort_toggles_direction_and_reorders_rows():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { handleScreenSortClick } from "./marvis/static/js/v2/screen_gate_controller.js";

        const screen = {
          selected: ["low_ks", "mid_ks", "high_ks"],
          leakage: [],
          suspected: [],
          unusable: [],
          scores: {
            low_ks: { ks: 0.1, iv: 0.02 },
            mid_ks: { ks: 0.3, iv: 0.15 },
            high_ks: { ks: 0.5, iv: 0.4 },
          },
          n_screened: 3,
          thresholds: { leakage_ks: 0.6, max_missing_rate: 0.95 },
        };
        const message = { id: "sort-msg", metadata: { step_id: "gate-sort", screen } };
        let lastHtml = null;
        const context = {
          getAgentMessages: () => [message],
          applyRerender: (wrap, html) => { lastHtml = html; },
        };
        const wrap = {
          dataset: { screenForm: "sort-msg", screenReadonly: "false" },
          querySelectorAll: () => [],
        };
        const button = {
          closest: (sel) => (sel === "[data-screen-sort]" ? button : wrap),
          getAttribute: () => "ks",
        };
        handleScreenSortClick({ target: button, preventDefault: () => {} }, context);
        // first click: descending by KS
        let order = [...lastHtml.matchAll(/data-screen-feature="([^"]+)"/g)].map((m) => m[1]);
        assert.deepEqual(order, ["high_ks", "mid_ks", "low_ks"]);
        assert.equal(lastHtml.includes("KS \\u25bc"), true);
        // second click on the same column: flips to ascending
        handleScreenSortClick({ target: button, preventDefault: () => {} }, context);
        order = [...lastHtml.matchAll(/data-screen-feature="([^"]+)"/g)].map((m) => m[1]);
        assert.deepEqual(order, ["low_ks", "mid_ks", "high_ks"]);
        assert.equal(lastHtml.includes("KS \\u25b2"), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_category_chip_filters_rows():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { handleScreenChipClick } from "./marvis/static/js/v2/screen_gate_controller.js";

        const screen = {
          selected: ["keep_a"],
          leakage: [["leak_col", 0.81, "reason"]],
          suspected: [["susp_col", 0.42, "reason"]],
          unusable: [["dead_col", "reason"]],
          scores: {
            keep_a: { ks: 0.1, iv: 0.05, coverage: 0.9 },
            leak_col: { ks: 0.81, iv: 0.9, coverage: 1 },
            susp_col: { ks: 0.42, iv: 0.35, coverage: 0.2 },
          },
          n_screened: 4,
          thresholds: { leakage_ks: 0.4, max_missing_rate: 0.95 },
        };
        const message = { id: "chip-msg", metadata: { step_id: "gate-chip", screen } };
        let lastHtml = null;
        const context = {
          getAgentMessages: () => [message],
          applyRerender: (wrap, html) => { lastHtml = html; },
        };
        function clickChip(chipKey) {
          const wrap = {
            dataset: { screenForm: "chip-msg", screenReadonly: "false" },
            querySelectorAll: () => [],
          };
          const button = {
            closest: (sel) => (sel === "[data-screen-chip]" ? button : wrap),
            getAttribute: () => chipKey,
          };
          return handleScreenChipClick({ target: button, preventDefault: () => {} }, context);
        }
        assert.equal(clickChip("leakage"), true);
        let rows = [...lastHtml.matchAll(/data-screen-feature="([^"]+)"/g)].map((m) => m[1]);
        assert.deepEqual(rows.sort(), ["leak_col", "susp_col"]);

        assert.equal(clickChip("low_coverage"), true);
        rows = [...lastHtml.matchAll(/data-screen-feature="([^"]+)"/g)].map((m) => m[1]);
        assert.deepEqual(rows, ["susp_col"]);

        assert.equal(clickChip("all"), true);
        rows = [...lastHtml.matchAll(/data-screen-feature="([^"]+)"/g)].map((m) => m[1]);
        assert.deepEqual(rows.sort(), ["dead_col", "keep_a", "leak_col", "susp_col"]);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_bulk_select_clear_invert_visible():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { handleScreenBulkClick } from "./marvis/static/js/v2/screen_gate_controller.js";

        const screen = {
          selected: [],
          leakage: [],
          suspected: [],
          unusable: [],
          scores: {
            f1: { ks: 0.1, iv: 0.05 },
            f2: { ks: 0.2, iv: 0.1 },
            f3: { ks: 0.3, iv: 0.15 },
          },
          ranked: [["f1", 0.1], ["f2", 0.2], ["f3", 0.3]],
          n_screened: 3,
          thresholds: { leakage_ks: 0.9, max_missing_rate: 0.95 },
        };
        // none of f1/f2/f3 are in `selected`, so build rows via the `keep` bucket
        // by putting them in `selected` instead (bulk ops operate on whatever is
        // rendered, category is not the point of this test).
        screen.selected = ["f1", "f2", "f3"];
        const message = { id: "bulk-msg", metadata: { step_id: "gate-bulk", screen } };
        let lastHtml = null;
        const context = {
          getAgentMessages: () => [message],
          applyRerender: (wrap, html) => { lastHtml = html; },
        };
        function clickBulk(action, checkboxState) {
          const boxes = Object.entries(checkboxState).map(([value, checked]) => ({ value, checked, disabled: false }));
          const wrap = {
            dataset: { screenForm: "bulk-msg", screenReadonly: "false" },
            querySelectorAll: (sel) => {
              if (sel === ".screen-pick") return boxes;
              if (sel.includes("screen-leakage") || sel.includes("screen-suspected")) return [];
              return [];
            },
          };
          const button = {
            closest: (sel) => (sel === "[data-screen-bulk]" ? button : wrap),
            getAttribute: () => action,
          };
          return handleScreenBulkClick({ target: button, preventDefault: () => {} }, context);
        }
        clickBulk("select_visible", { f1: false, f2: false, f3: false });
        let checkedCount = (lastHtml.match(/class="screen-pick" value="f\\d" checked/g) || []).length;
        assert.equal(checkedCount, 3);
        assert.equal(lastHtml.includes("已选 3/3"), true);

        clickBulk("clear_visible", { f1: true, f2: true, f3: true });
        checkedCount = (lastHtml.match(/class="screen-pick" value="f\\d" checked/g) || []).length;
        assert.equal(checkedCount, 0);
        assert.equal(lastHtml.includes("已选 0/3"), true);

        clickBulk("invert_visible", { f1: true, f2: false, f3: false });
        checkedCount = (lastHtml.match(/class="screen-pick" value="f\\d" checked/g) || []).length;
        assert.equal(checkedCount, 2);
        assert.equal(lastHtml.includes("已选 2/3"), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_paginates_at_fifty_rows_per_page():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { renderScreenGateTable, handleScreenPageClick } from "./marvis/static/js/v2/screen_gate_controller.js";

        const selected = [];
        const scores = {};
        for (let i = 0; i < 120; i++) {
          const name = "f" + String(i).padStart(3, "0");
          selected.push(name);
          scores[name] = { ks: 0.01 * i, iv: 0.005 * i };
        }
        const screen = {
          selected,
          leakage: [],
          suspected: [],
          unusable: [],
          scores,
          n_screened: 120,
          thresholds: { leakage_ks: 0.9, max_missing_rate: 0.95 },
        };
        const message = { id: "page-msg", metadata: { step_id: "gate-page", screen } };
        const firstPageHtml = renderScreenGateTable(message, { interactive: true });
        assert.equal((firstPageHtml.match(/class="screen-row/g) || []).length, 50);
        assert.equal(firstPageHtml.includes("第 1 / 3 页"), true);

        let lastHtml = null;
        const context = {
          getAgentMessages: () => [message],
          applyRerender: (wrap, html) => { lastHtml = html; },
        };
        const wrap = {
          dataset: { screenForm: "page-msg", screenReadonly: "false" },
          querySelectorAll: () => [],
        };
        const nextButton = {
          closest: (sel) => {
            if (sel === "[data-screen-page-next]") return nextButton;
            if (sel === ".screen-table-wrap") return wrap;
            return null;
          },
          getAttribute: () => "1",
        };
        handleScreenPageClick({ target: nextButton, preventDefault: () => {} }, context);
        assert.equal((lastHtml.match(/class="screen-row/g) || []).length, 50);
        assert.equal(lastHtml.includes("第 2 / 3 页"), true);
        assert.equal(lastHtml.includes("f000"), false);
        assert.equal(lastHtml.includes("f050"), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_leakage_pick_requires_override_reason_before_confirm():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { submitScreenSelection } from "./marvis/static/js/v2/screen_gate_controller.js";

        function makeWrap(reasonValue) {
          const leakageRow = { classList: { contains: (c) => c === "screen-leakage" } };
          const checkbox = {
            value: "leak_col",
            checked: true,
            disabled: false,
            closest: (sel) => (sel === ".screen-row" ? leakageRow : null),
          };
          return {
            dataset: { screenForm: "leak-msg", screenReadonly: "false", screenStepId: "gate-leak" },
            querySelectorAll: (sel) => (sel === ".screen-pick:checked" ? [checkbox] : []),
            querySelector: (sel) => (sel === ".screen-leakage-reason-input" ? { value: reasonValue } : null),
          };
        }
        const statuses = [];
        const calls = [];
        const context = {
          getSelectedTaskId: () => "task-1",
          agentAcceptanceModeValue: () => "manual",
          setActionStatus: (message, kind) => statuses.push([message, kind]),
          setAgentMessages: () => {},
          renderAgentConversation: () => {},
          api: async (url, options) => {
            calls.push([url, JSON.parse(options.body)]);
            return { messages: [] };
          },
        };
        // empty reason: rejected, nothing posted
        await submitScreenSelection({ disabled: false, closest: () => makeWrap("") }, context);
        assert.equal(calls.length, 0);
        assert.deepEqual(statuses.at(-1), ["勾选了泄漏/疑似列，请先填写覆盖理由（至少4个字）。", "error"]);

        // too-short reason: also rejected
        await submitScreenSelection({ disabled: false, closest: () => makeWrap("ok") }, context);
        assert.equal(calls.length, 0);
        assert.deepEqual(statuses.at(-1), ["勾选了泄漏/疑似列，请先填写覆盖理由（至少4个字）。", "error"]);

        // sufficient reason: posts with the reason folded into content
        await submitScreenSelection({ disabled: false, closest: () => makeWrap("已核实非未来信息") }, context);
        assert.equal(calls.length, 1);
        assert.equal(calls[0][1].content, "确认（泄漏/疑似列覆盖理由：已核实非未来信息）");
        assert.deepEqual(calls[0][1].selection, ["leak_col"]);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_databar_widths_are_proportional_to_values():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { renderScreenGateTable } from "./marvis/static/js/v2/screen_gate_controller.js";

        const screen = {
          selected: ["low_feat", "high_feat"],
          leakage: [],
          suspected: [],
          unusable: [],
          scores: {
            low_feat: { ks: 0.1, iv: 0.05 },
            high_feat: { ks: 0.4, iv: 0.2 },
          },
          n_screened: 2,
          thresholds: { leakage_ks: 0.9, max_missing_rate: 0.95 },
        };
        const message = { id: "bar-msg", metadata: { step_id: "gate-bar", screen } };
        const html = renderScreenGateTable(message, { interactive: true });
        const fractions = [...html.matchAll(/--fraction:([0-9.]+)/g)].map((m) => Number(m[1]));
        // KS column: 0.1/0.4 = 0.25, 0.4/0.4 = 1; IV column: 0.05/0.2 = 0.25, 0.2/0.2 = 1
        assert.deepEqual(fractions, [0.25, 0.25, 1, 1]);
        assert.equal(html.includes("screen-databar"), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_iv_tier_badges_and_visual_hierarchy_css():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { ivTier, ivTooltipText, renderScreenGateTable } from "./marvis/static/js/v2/screen_gate_controller.js";

        assert.equal(ivTier(0.01).tier, "none");
        assert.equal(ivTier(0.05).tier, "weak");
        assert.equal(ivTier(0.2).tier, "medium");
        assert.equal(ivTier(0.35).tier, "strong");
        assert.equal(ivTier(null), null);
        assert.equal(ivTooltipText(0.6).includes("建议结合 KS/业务口径复核是否存在泄漏"), true);

        const screen = {
          selected: ["strong_feat"],
          leakage: [],
          suspected: [],
          unusable: [],
          scores: { strong_feat: { ks: 0.3, iv: 0.35 } },
          n_screened: 1,
          thresholds: { leakage_ks: 0.9, max_missing_rate: 0.95 },
        };
        const message = { id: "tier-msg", metadata: { step_id: "gate-tier", screen } };
        const html = renderScreenGateTable(message, { interactive: true });
        assert.equal(html.includes('data-tier="strong"'), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_screen_table_visual_hierarchy_css_tokens_only():
    css = _read("css/v2-workbench.css")
    start = css.index("/* §4 interactive feature-screening selection table")
    end = css.index("/* §4 join dedup picker", start)
    screen_css = css[start:end]
    assert ".iv-tier-badge" in screen_css
    assert ".screen-row.screen-leakage" in screen_css
    assert "border-left: 3px solid var(--danger)" in screen_css
    assert "border-left: 3px solid var(--warning)" in screen_css
    assert ".screen-databar" in screen_css
    assert "font-variant-numeric: tabular-nums" in screen_css
    assert "text-align: right" in screen_css
    # no ad-hoc hex/rgb colors — everything routes through existing --* tokens
    assert "#" not in screen_css


def test_dedup_picker_renderer_and_branch_are_wired():
    app_js = _read("app.js")
    module_js = _read("js/v2/join_gate_controller.js")
    manual_module_js = _read("js/v2/driver_manual_analysis.js")
    assert "function agentMessageDedupPickerHtml(message, options = {})" in app_js
    assert "return renderDedupPicker(message, options);" in app_js
    assert "if (meta.dedup)" in manual_module_js
    # UX-2: both modes mount the picker through the shared driverGateBodyHtml,
    # so the render branch decision lives in driver_manual_analysis.js, not a
    # per-mode copy.
    assert "if (meta.dedup) return" in manual_module_js
    assert "export function renderDedupPicker(message, options = {})" in module_js
    assert "message?.metadata?.dedup" in module_js
    # a first/last strategy <select> per conflicting feature
    assert 'class="dedup-strategy"' in module_js
    assert "data-dedup-feature" in module_js
    # UX-2: an earlier (non-latest) dedup gate renders read-only so a stale tab
    # cannot re-submit strategies against an already-advanced step.
    assert 'data-dedup-readonly="true"' in module_js
    assert "form.dataset.dedupReadonly" in module_js


def test_dedup_picker_posts_strategies():
    app_js = _read("app.js")
    module_js = _read("js/v2/join_gate_controller.js")
    assert "function submitDedupStrategies(button)" in app_js
    assert "submitDedupStrategiesController(button, joinGateControllerContext())" in app_js
    assert "export async function submitDedupStrategies(button, rawContext = {})" in module_js
    assert "data-dedup-confirm" in module_js
    assert "dedup_strategies" in module_js
    assert "data-dedup-gate-step-id" in module_js
    assert "expected_step_id: expectedStepId" in module_js
    assert "handleDedupConfirmClick" in app_js
    css = _read("css/v2-workbench.css")
    assert ".dedup-picker" in css and ".dedup-table" in css
    # UX-1/REL-1: dedup submission shares the same busy-feedback contract
    # (immediate busy pill + streamed poll + plan rail ticks) as the other
    # v2 gate controllers, since it also reruns the driver turn.
    assert '"正在执行下一步…", "busy"' in module_js
    assert "pollAgentMessagesUntilSettled" in module_js
    assert "resetFetchThrottle" in module_js
    assert "renderWorkflowStepper" in module_js


def test_join_c1_form_renderer_and_submit_are_wired():
    app_js = _read("app.js")
    module_js = _read("js/v2/join_gate_controller.js")
    manual_module_js = _read("js/v2/driver_manual_analysis.js")
    assert "function agentMessageC1FormHtml(message, options = {})" in app_js
    assert "return renderJoinC1Form(message, options);" in app_js
    assert "if (meta.join_c1)" in manual_module_js
    # UX-2: both modes mount the form through the shared driverGateBodyHtml.
    assert "if (meta.join_c1) return" in manual_module_js
    assert "export function renderJoinC1Form(message, options = {})" in module_js
    assert 'class="c1-role"' in module_js
    assert "data-c1-dataset" in module_js
    assert "function submitC1Assignment(button)" in app_js
    assert "submitC1AssignmentController(button, joinGateControllerContext())" in app_js
    assert "export async function submitC1Assignment(button, rawContext = {})" in module_js
    assert '"[C1]"' in module_js
    assert "anchor_id" in module_js
    assert "feature_ids" in module_js
    assert "target_col" in module_js
    assert "handleC1ConfirmClick" in app_js
    # UX-1/REL-1: C1 submission shares the same busy-feedback contract
    # (immediate busy pill + streamed poll + plan rail ticks) as the other
    # v2 gate controllers, since it also reruns the driver turn (execute_join).
    assert '"正在执行下一步…", "busy"' in module_js
    assert "pollAgentMessagesUntilSettled" in module_js
    assert "resetFetchThrottle" in module_js
    assert "renderWorkflowStepper" in module_js
    # UX-2: an earlier (non-latest) C1 form renders read-only so a stale tab
    # cannot re-submit role assignments against an already-advanced step.
    assert 'data-c1-readonly="true"' in module_js
    assert "form.dataset.c1Readonly" in module_js


def test_join_gate_controller_posts_c1_and_dedup_payloads():
    output = _run_node(
        f"""
        {""}
        import assert from "node:assert/strict";
        import {{
          renderDedupPicker,
          renderJoinC1Form,
          submitC1Assignment,
          submitDedupStrategies,
        }} from "./marvis/static/js/v2/join_gate_controller.js";
        let agentMessages = [];
        let rendered = 0;
        const statuses = [];
        const calls = [];
        const context = {{
          selectedTaskId: "task-1",
          agentAcceptanceModeValue: () => "manual",
          setActionStatus: (message, kind) => statuses.push([message, kind]),
          setAgentMessages: (messages) => {{ agentMessages = messages || agentMessages; }},
          renderAgentConversation: () => {{ rendered += 1; }},
          api: async (url, options) => {{
            calls.push([url, JSON.parse(options.body)]);
            return {{ messages: [{{ id: "m2" }}] }};
          }},
        }};
        const c1Html = renderJoinC1Form({{
          id: "c1-msg",
          metadata: {{
            join_c1: {{
              target_col: "bad",
              files: [
                {{ dataset_id: "main", name: "main.csv", row_count: 10, n_cols: 3, has_target: true, proposed_role: "anchor", columns: ["id", "bad"] }},
                {{ dataset_id: "feat", name: "feat.csv", row_count: 10, n_cols: 2, has_target: false, proposed_role: "feature", columns: ["id", "score"] }},
              ],
            }},
          }},
        }});
        assert.equal(c1Html.includes('data-c1-form="c1-msg"'), true);
        assert.equal(c1Html.includes('data-c1-dataset="main"'), true);
        assert.equal(c1Html.includes('value="bad" selected'), true);
        const c1Form = {{
          dataset: {{}},
          querySelectorAll: (selector) => selector === ".c1-role" ? [
            {{ getAttribute: (name) => name === "data-c1-dataset" ? "main" : null, value: "anchor" }},
            {{ getAttribute: (name) => name === "data-c1-dataset" ? "feat" : null, value: "feature" }},
          ] : [],
          querySelector: (selector) => selector === ".c1-target" ? {{ value: "bad" }} : null,
        }};
        const c1Button = {{ disabled: false, closest: () => c1Form }};
        await submitC1Assignment(c1Button, context);
        assert.equal(c1Button.disabled, true);
        assert.equal(calls[0][0], "/api/tasks/task-1/agent/messages");
        assert.equal(calls[0][1].acceptance_mode, "manual");
        assert.equal(calls[0][1].content.startsWith("[C1]"), true);
        assert.deepEqual(JSON.parse(calls[0][1].content.slice(4)), {{
          anchor_id: "main",
          anchor_ids: ["main"],
          feature_ids: ["feat"],
          target_col: "bad",
        }});

        const noAnchorForm = {{
          dataset: {{}},
          querySelectorAll: () => [{{ getAttribute: () => "feat", value: "feature" }}],
          querySelector: () => null,
        }};
        await submitC1Assignment({{ disabled: false, closest: () => noAnchorForm }}, context);
        assert.deepEqual(statuses.at(-1), ["请先把一张表选为「样本主表」。", "error"]);

        // UX-7: two datasets marked "样本主表" must be rejected client-side,
        // not silently collapsed to a single anchor with the second dropped.
        const callsBeforeDuplicateAttempt = calls.length;
        const duplicateAnchorForm = {{
          dataset: {{}},
          querySelectorAll: (selector) => selector === ".c1-role" ? [
            {{ getAttribute: (name) => name === "data-c1-dataset" ? "main" : null, value: "anchor" }},
            {{ getAttribute: (name) => name === "data-c1-dataset" ? "feat" : null, value: "anchor" }},
          ] : [],
          querySelector: () => null,
        }};
        await submitC1Assignment({{ disabled: false, closest: () => duplicateAnchorForm }}, context);
        assert.deepEqual(statuses.at(-1), ["只能有一张样本主表，请把其余表改为「特征表」或「忽略」。", "error"]);
        assert.equal(calls.length, callsBeforeDuplicateAttempt);

        // UX-2: a read-only (stale) C1 form must refuse to submit, matching the
        // screen/modeling-setup/dedup readonly-guard convention.
        const readonlyC1Form = {{ dataset: {{ c1Readonly: "true" }} }};
        await submitC1Assignment({{ disabled: false, closest: () => readonlyC1Form }}, context);
        assert.deepEqual(statuses.at(-1), ["这是历史拼接角色结果,请使用最新待确认步骤确认。", "error"]);
        assert.equal(calls.length, callsBeforeDuplicateAttempt);

        const dedupHtml = renderDedupPicker({{
          id: "dedup-msg",
          metadata: {{
            step_id: "gate-dedup",
            dedup: {{
              strategies: ["first", "last"],
              features: [{{ feature_id: "feat", conflict_keys: 3 }}],
            }},
          }},
        }});
        assert.equal(dedupHtml.includes('data-dedup-gate-step-id="gate-dedup"'), true);
        assert.equal(dedupHtml.includes('data-dedup-feature="feat"'), true);
        const readonlyDedupHtml = renderDedupPicker({{
          id: "dedup-msg-old",
          metadata: {{
            step_id: "gate-dedup-old",
            dedup: {{
              strategies: ["first", "last"],
              features: [{{ feature_id: "feat", conflict_keys: 3 }}],
            }},
          }},
        }}, {{ interactive: false }});
        assert.equal(readonlyDedupHtml.includes('data-dedup-readonly="true"'), true);
        assert.equal(readonlyDedupHtml.includes("历史结果"), true);
        const dedupForm = {{
          dataset: {{ dedupGateStepId: "gate-dedup" }},
          querySelectorAll: (selector) => selector === ".dedup-strategy" ? [
            {{ getAttribute: (name) => name === "data-dedup-feature" ? "feat" : null, value: "last" }},
          ] : [],
        }};
        const dedupButton = {{ disabled: false, closest: () => dedupForm }};
        await submitDedupStrategies(dedupButton, context);
        assert.equal(dedupButton.disabled, true);
        assert.deepEqual(calls[1][1].dedup_strategies, {{ feat: "last" }});
        assert.equal(calls[1][1].expected_step_id, "gate-dedup");
        assert.deepEqual(agentMessages, [{{ id: "m2" }}]);
        assert.equal(rendered, 2);

        // UX-2: a read-only (stale) dedup picker must refuse to submit.
        const callsBeforeReadonlyDedup = calls.length;
        const readonlyDedupForm = {{ dataset: {{ dedupReadonly: "true" }} }};
        await submitDedupStrategies({{ disabled: false, closest: () => readonlyDedupForm }}, context);
        assert.deepEqual(statuses.at(-1), ["这是历史去重结果,请使用最新待确认步骤确认。", "error"]);
        assert.equal(calls.length, callsBeforeReadonlyDedup);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_driver_gate_confirm_controller_renders_and_posts_confirm():
    output = _run_node(
        f"""
        {""}
        import assert from "node:assert/strict";
        import {{
          handleDriverConfirmClick,
          renderDriverGateButton,
          submitDriverConfirm,
        }} from "./marvis/static/js/v2/driver_gate_confirm.js";
        const renderedButton = renderDriverGateButton({{ metadata: {{ kind: "gate", step_id: "gate<1>" }} }});
        assert.equal(renderedButton.includes("data-driver-confirm"), true);
        assert.equal(renderedButton.includes('data-expected-step-id="gate&lt;1&gt;"'), true);
        // UX-2: the plain confirm button now renders identically in agent mode
        // too (no more isAgentMode short-circuit) — it only steps aside when the
        // gate carries a structured widget (screen/dedup/modeling_setup/join_c1),
        // since that widget already owns the primary confirm action.
        assert.equal(renderDriverGateButton({{ metadata: {{ kind: "gate" }} }}).includes("data-driver-confirm"), true);
        assert.equal(renderDriverGateButton({{ metadata: {{ kind: "gate", screen: {{ selected: ["x"] }} }} }}), "");
        assert.equal(renderDriverGateButton({{ metadata: {{ kind: "gate", dedup: {{ features: [{{ feature_id: "f" }}] }} }} }}), "");
        assert.equal(renderDriverGateButton({{ metadata: {{ kind: "gate", modeling_setup: {{}} }} }}), "");
        assert.equal(renderDriverGateButton({{ metadata: {{ kind: "done" }} }}), "");

        let agentMessages = [];
        let rendered = 0;
        const statuses = [];
        const calls = [];
        // UX-1/REL-1: minimal, call-counting stubs for the busy-feedback
        // capabilities (poll/reset/stepper) so the immediate-busy-state and
        // plan-rail "finally" refresh can be asserted alongside the existing
        // success-path payload assertions.
        let pollCalls = 0;
        let resetCalls = 0;
        let stepperCalls = 0;
        const context = {{
          selectedTaskId: "task-1",
          setActionStatus: (message, kind) => statuses.push([message, kind]),
          setAgentMessages: (messages) => {{ agentMessages = messages || agentMessages; }},
          renderAgentConversation: () => {{ rendered += 1; }},
          api: async (url, options) => {{
            calls.push([url, JSON.parse(options.body)]);
            return {{ messages: [{{ id: "m2" }}] }};
          }},
          pollAgentMessagesUntilSettled: async () => {{ pollCalls += 1; }},
          resetFetchThrottle: () => {{ resetCalls += 1; }},
          renderWorkflowStepper: () => {{ stepperCalls += 1; }},
        }};
        const button = {{
          disabled: false,
          getAttribute: (name) => name === "data-expected-step-id" ? "gate-1" : null,
        }};
        await submitDriverConfirm(button, context);
        // busy state is pushed synchronously before the request settles
        assert.deepEqual(statuses[0], ["正在执行下一步…", "busy"]);
        // success path leaves the button disabled (no re-enable on success)
        assert.equal(button.disabled, true);
        assert.equal(calls[0][0], "/api/tasks/task-1/agent/messages");
        assert.deepEqual(calls[0][1], {{ content: "确认", expected_step_id: "gate-1" }});
        assert.deepEqual(agentMessages, [{{ id: "m2" }}]);
        assert.equal(rendered, 1);
        // finally-path plan-rail refresh fires at least once on the success path
        assert.equal(pollCalls, 1);
        assert.equal(resetCalls >= 1, true);
        assert.equal(stepperCalls >= 1, true);

        // failure path: api rejects, button is re-enabled, and an error status
        // (not just the transient busy status) is surfaced to the user.
        let failResetCalls = 0;
        let failStepperCalls = 0;
        const failContext = {{
          ...context,
          api: async () => {{ throw new Error("网络错误"); }},
          resetFetchThrottle: () => {{ failResetCalls += 1; }},
          renderWorkflowStepper: () => {{ failStepperCalls += 1; }},
        }};
        const failButton = {{
          disabled: false,
          getAttribute: (name) => name === "data-expected-step-id" ? "gate-1" : null,
        }};
        await submitDriverConfirm(failButton, failContext);
        assert.equal(failButton.disabled, false);
        assert.deepEqual(statuses.at(-1), ["网络错误", "error"]);
        // the plan rail is still force-refreshed via the finally path on failure
        assert.equal(failResetCalls >= 1, true);
        assert.equal(failStepperCalls >= 1, true);

        const eventButton = {{ disabled: false }};
        const event = {{
          target: {{ closest: (selector) => selector === "[data-driver-confirm]" ? eventButton : null }},
          preventDefault: () => statuses.push(["prevented", "event"]),
        }};
        assert.equal(handleDriverConfirmClick(event, context), true);
        // preventDefault fires synchronously (before the async submit's busy push)
        const preventedIndex = statuses.findIndex((entry) => entry[0] === "prevented");
        assert.equal(preventedIndex >= 0, true);
        await new Promise((resolve) => setTimeout(resolve, 0));
        const busyIndexAfterEvent = statuses.findIndex((entry, index) => index > preventedIndex && entry[1] === "busy");
        assert.equal(busyIndexAfterEvent > preventedIndex, true);
        assert.equal(calls.length, 2);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_capability_tier_picker_is_wired():
    """TIER-IA (spec §5.1): the create dialog exposes a per-task capability-tier
    selector (the previously-missing entry point), collected into payload."""
    index_html = _read("index.html")
    app_js = _read("app.js")
    create_dialog_js = _read("js/create-task-dialog.js")
    task_types_js = _read("js/task-types.js")
    assert 'id="createTaskTier"' in index_html
    for tier in ("conservative", "balanced", "autonomous"):
        assert f'value="{tier}"' in index_html
    assert "createTaskTierField" in create_dialog_js
    assert "tierField" in task_types_js  # gated to driver task types
    assert "syncCreateTaskTierDefault" in app_js
    assert "getSelectedTier" in create_dialog_js
    assert "payload.capability_tier" in create_dialog_js
