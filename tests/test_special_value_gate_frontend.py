"""Directed frontend contract tests for the modeling special-value HITL gate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "marvis" / "static"
SPECIAL_VALUE_MODULE = STATIC_DIR / "js" / "v2" / "special_value_gate.js"
MANUAL_ANALYSIS_MODULE = STATIC_DIR / "js" / "v2" / "driver_manual_analysis.js"
GATE_CONFIRM_MODULE = STATIC_DIR / "js" / "v2" / "driver_gate_confirm.js"
APP_JS = STATIC_DIR / "app.js"


def _run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )


def _module_url(path: Path) -> str:
    return json.dumps(path.as_uri())


def _slice_function(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"\nfunction {next_name}", start)
    return source[start:end]


def test_special_value_gate_renders_evidence_actions_and_one_atomic_button():
    _run_node(
        f"""
        import assert from "node:assert/strict";
        import {{ renderSpecialValueGate }} from {_module_url(SPECIAL_VALUE_MODULE)};

        const message = {{
          metadata: {{
            step_id: "model-special-values",
            special_values: {{
              columns: [
                {{
                  column: "balance",
                  values: [
                    {{ value: -999, share: 0.125 }},
                    {{ value: 999999, share: 0.03 }},
                  ],
                }},
                {{
                  column: "channel",
                  values: [{{ value: "<UNKNOWN>", share: 0.5 }}],
                }},
              ],
            }},
          }},
        }};

        const html = renderSpecialValueGate(message);
        assert.match(html, /balance/);
        assert.match(html, /channel/);
        assert.match(html, /-999/);
        assert.match(html, /999999/);
        assert.match(html, /12\\.5%/);
        assert.match(html, /3\\.00%/);
        assert.match(html, /&lt;UNKNOWN&gt;/);
        assert.match(html, /50\\.0%/);
        assert.match(html, /<option value="mask">/);
        assert.match(html, /<option value="retain">/);
        assert.match(html, /<option value="drop">/);

        const buttons = html.match(/<button\\b[^>]*>/g) || [];
        assert.equal(buttons.length, 1);
        assert.match(buttons[0], /data-special-value-submit/);
        assert.match(html, /class="special-value-actions gate-action-bar"/);
        assert.match(html, /确认治理策略并继续/);
        """
    )


def test_special_value_gate_read_only_render_and_agent_conversation_wiring():
    app_js = APP_JS.read_text(encoding="utf-8")
    strip_buttons = _slice_function(
        app_js,
        "stripGateButtonsHtml",
        "normalizeAgentConversationGateContent",
    )

    _run_node(
        f"""
        import assert from "node:assert/strict";
        import {{ renderSpecialValueGate }} from {_module_url(SPECIAL_VALUE_MODULE)};

        const message = {{
          metadata: {{
            step_id: "model-special-values",
            special_values: {{
              columns: [{{
                column: "balance",
                values: [{{ value: -999, share: 0.1 }}],
              }}],
            }},
          }},
        }};
        const readOnlyHtml = renderSpecialValueGate(
          message,
          {{ interactive: false }},
        );
        const controls = readOnlyHtml.match(/<(?:select|input|button)\\b[^>]*>/g) || [];
        assert.equal(controls.length, 3);
        for (const control of controls) assert.match(control, /\\bdisabled\\b/);

        {strip_buttons}
        const conversationHtml = stripGateButtonsHtml(readOnlyHtml);
        assert.doesNotMatch(conversationHtml, /<button\\b/i);
        assert.match(conversationHtml, /balance/);
        assert.match(conversationHtml, /-999/);
        """
    )

    # The Agent timeline always requests conversation-only rendering. The shared
    # gate body remains visible as evidence, but is forced read-only and then has
    # every workflow-advancing button removed. Manual mode injects the same
    # special-value renderer without that conversation-only override.
    assert "{ ...options, conversationOnly: true }" in app_js
    assert "{ interactive: conversationOnly ? false : interactive }" in app_js
    assert "return conversationOnly ? stripGateButtonsHtml(html) : html;" in app_js
    assert app_js.count(
        "renderSpecialValues: agentMessageSpecialValuesHtml"
    ) >= 2


def test_special_value_submit_posts_atomic_decisions_without_evidence_values():
    _run_node(
        f"""
        import assert from "node:assert/strict";
        import {{ submitSpecialValueDecisions }} from {_module_url(SPECIAL_VALUE_MODULE)};

        function makeRow(column, action, reason = "") {{
          const actionControl = {{ value: action, disabled: false }};
          const reasonControl = {{ value: reason, disabled: false }};
          return {{
            dataset: {{
              specialValueColumn: column,
              // Even hostile DOM metadata must never be copied into the request.
              values: JSON.stringify([-999, 999999]),
            }},
            controls: [actionControl, reasonControl],
            querySelector(selector) {{
              if (selector === "[data-special-value-action]") return actionControl;
              if (selector === "[data-special-value-reason]") return reasonControl;
              return null;
            }},
          }};
        }}

        const rows = [
          makeRow("balance", "mask"),
          makeRow("channel", "retain", "业务约定的有效占位码"),
          makeRow("obsolete_feature", "drop"),
        ];
        const submitButton = {{ disabled: false }};
        const controls = [submitButton, ...rows.flatMap((row) => row.controls)];
        const wrap = {{
          dataset: {{ specialValueStepId: "model-special-values" }},
          querySelectorAll(selector) {{
            if (selector === "[data-special-value-row]") return rows;
            if (selector === "button, select, input") return controls;
            return [];
          }},
        }};
        submitButton.closest = (selector) => (
          selector === "[data-special-value-step-id]" ? wrap : null
        );

        const calls = [];
        let renderedMessages = null;
        await submitSpecialValueDecisions(submitButton, {{
          getSelectedTaskId: () => "task-17",
          api: async (url, options) => {{
            calls.push({{ url, options }});
            return {{ messages: [{{ role: "assistant", content: "已接收" }}] }};
          }},
          pollAgentMessagesUntilSettled: async () => undefined,
          setAgentMessages: (messages) => {{ renderedMessages = messages; }},
        }});

        assert.equal(calls.length, 1);
        assert.equal(calls[0].url, "/api/tasks/task-17/agent/messages");
        assert.equal(calls[0].options.method, "POST");
        const body = JSON.parse(calls[0].options.body);
        assert.deepEqual(body, {{
          content: "确认",
          ui_action: "confirm_gate",
          expected_step_id: "model-special-values",
          adjust_params: {{
            decisions: {{
              balance: {{ action: "mask" }},
              channel: {{
                action: "retain",
                confirmed: true,
                reason: "业务约定的有效占位码",
              }},
              obsolete_feature: {{ action: "drop" }},
            }},
          }},
        }});
        assert.equal(JSON.stringify(body).includes('"values"'), false);
        assert.deepEqual(
          renderedMessages,
          [{{ role: "assistant", content: "已接收" }}],
        );
        assert.equal(controls.every((control) => control.disabled), true);
        """
    )


def test_special_value_retain_without_reason_is_rejected_before_post():
    _run_node(
        f"""
        import assert from "node:assert/strict";
        import {{ submitSpecialValueDecisions }} from {_module_url(SPECIAL_VALUE_MODULE)};

        const actionControl = {{ value: "retain", disabled: false }};
        const reasonControl = {{ value: "   ", disabled: false }};
        const row = {{
          dataset: {{ specialValueColumn: "channel" }},
          querySelector(selector) {{
            if (selector === "[data-special-value-action]") return actionControl;
            if (selector === "[data-special-value-reason]") return reasonControl;
            return null;
          }},
        }};
        const submitButton = {{ disabled: false }};
        const wrap = {{
          dataset: {{ specialValueStepId: "model-special-values" }},
          querySelectorAll(selector) {{
            if (selector === "[data-special-value-row]") return [row];
            if (selector === "button, select, input") {{
              return [submitButton, actionControl, reasonControl];
            }}
            return [];
          }},
        }};
        submitButton.closest = () => wrap;

        let apiCalls = 0;
        const statuses = [];
        await submitSpecialValueDecisions(submitButton, {{
          selectedTaskId: "task-17",
          api: async () => {{
            apiCalls += 1;
            return {{ messages: [] }};
          }},
          setActionStatus: (message, tone) => statuses.push([message, tone]),
        }});

        assert.equal(apiCalls, 0);
        assert.deepEqual(
          statuses,
          [["channel 选择保留时必须填写理由", "error"]],
        );
        assert.equal(submitButton.disabled, false);
        assert.equal(actionControl.disabled, false);
        assert.equal(reasonControl.disabled, false);
        """
    )


def test_special_value_widget_suppresses_generic_gate_confirm():
    _run_node(
        f"""
        import assert from "node:assert/strict";
        import {{
          driverGateHasWidget,
          driverManualAnalysisHtml,
        }} from {_module_url(MANUAL_ANALYSIS_MODULE)};
        import {{ renderDriverGateButton }} from {_module_url(GATE_CONFIRM_MODULE)};

        const message = {{
          id: "gate-message-1",
          role: "assistant",
          content: "请确认特殊值治理方式。",
          metadata: {{
            kind: "gate",
            step_id: "model-special-values",
            special_values: {{
              columns: [{{
                column: "balance",
                values: [{{ value: -999, share: 0.1 }}],
              }}],
            }},
          }},
        }};

        assert.equal(driverGateHasWidget(message), true);
        assert.equal(
          renderDriverGateButton(
            message,
            {{ gateStepTool: "resolve_special_values" }},
          ),
          "",
        );

        let specialRenderCalls = 0;
        let genericConfirmCalls = 0;
        const html = driverManualAnalysisHtml([message], {{
          renderAgentMarkdown: (content) => content,
          renderSpecialValues: (_message, options) => {{
            specialRenderCalls += 1;
            assert.equal(options.interactive, true);
            return '<div data-special-value-widget>special values</div>';
          }},
          renderGateConfirm: () => {{
            genericConfirmCalls += 1;
            return '<button data-driver-confirm>generic confirm</button>';
          }},
        }});

        assert.equal(specialRenderCalls, 1);
        assert.equal(genericConfirmCalls, 0);
        assert.match(html, /data-special-value-widget/);
        assert.doesNotMatch(html, /data-driver-confirm/);
        """
    )
