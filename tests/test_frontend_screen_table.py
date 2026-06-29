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
    css = _read("css/v2-workbench.css")
    assert "function agentMessageModelingSetupHtml(message, options = {})" in app_js
    assert "if (meta.modeling_setup)" in app_js
    assert "agentMessageModelingSetupHtml(message, { interactive })" in app_js
    assert 'class="modeling-weight-pick"' in app_js
    assert "data-modeling-gate-step-id" in app_js
    assert "function submitModelingWeightAdjust(button)" in app_js
    assert "sample_weight_col: sampleWeightCol" in app_js
    assert "sample_weight_diagnostics" in app_js
    assert "modeling-weight-diagnostic" in app_js
    assert "modeling-spec-grid" in app_js
    assert "modeling-algorithm-grid" in app_js
    assert "modeling-split-summary" in app_js
    assert "handleModelingWeightAdjustClick" in app_js
    assert ".modeling-setup-panel" in css
    assert ".modeling-spec-grid" in css
    assert ".modeling-algorithm-grid" in css
    assert ".modeling-split-grid" in css
    assert ".modeling-weight-options" in css
    assert ".modeling-weight-diagnostics" in css


def test_modeling_setup_weight_picker_renders_candidates():
    render_slice = _app_slice("function agentMessageModelingSetupHtml", "async function submitModelingWeightAdjust")
    output = _run_node(
        f"""
        import assert from "node:assert/strict";
        function escapeHtml(value) {{
          return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
        }}
        {render_slice}
        const html = agentMessageModelingSetupHtml({{
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
              disabled_algorithms: [{{ recipe: "regressor", reason: "target mismatch" }}],
              pmml_supported_algorithms: ["lgb", "xgb"],
              warnings: ["样本权重列已从入模特征中移除。"],
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
        const readonly = agentMessageModelingSetupHtml({{
          id: "m2",
          metadata: {{ step_id: "gate-2", modeling_setup: {{ sample_weight_candidates: ["weight"] }} }},
        }}, {{ interactive: false }});
        assert.equal(readonly.includes('data-modeling-readonly="true"'), true);
        assert.equal(readonly.includes("历史规格"), true);
        const noWeight = agentMessageModelingSetupHtml({{
          id: "m3",
          metadata: {{
            step_id: "gate-3",
            modeling_setup: {{
              target_type: "continuous",
              recipe: "regressor",
              recipes: ["regressor"],
              feature_count: 5,
              n_trials: 6,
              metric_policy: "oot_rmse",
              eligible_algorithms: ["regressor"],
              pmml_supported_algorithms: [],
              sample_weight_candidates: [],
            }},
          }},
        }});
        assert.equal(noWeight.includes("建模规格"), true);
        assert.equal(noWeight.includes("continuous"), true);
        assert.equal(noWeight.includes("regressor"), true);
        assert.equal(noWeight.includes("不使用权重"), true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"


def test_modeling_setup_weight_adjust_posts_structured_params():
    adjust_slice = _app_slice("async function submitModelingWeightAdjust", "function handleModelingWeightAdjustClick")
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
        await submitModelingWeightAdjust({{ disabled: false, closest: () => ({{
          dataset: {{ modelingGateStepId: "gate-modeling", modelingCurrentWeight: "" }},
          querySelector: () => ({{ value: "weight" }}),
        }}) }});
        assert.equal(calls[0][0], "/api/tasks/task-1/agent/messages");
        assert.deepEqual(calls[0][1].adjust_params, {{ sample_weight_col: "weight" }});
        assert.equal(calls[0][1].expected_step_id, "gate-modeling");
        assert.equal(calls[0][1].acceptance_mode, "manual");

        await submitModelingWeightAdjust({{ disabled: false, closest: () => ({{
          dataset: {{ modelingGateStepId: "gate-modeling", modelingCurrentWeight: "weight" }},
          querySelector: () => ({{ value: "weight" }}),
        }}) }});
        assert.deepEqual(statuses.at(-1), ["样本权重设置未变化。", "info"]);
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
