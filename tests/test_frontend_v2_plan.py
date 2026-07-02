from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_plan_html_renders_ordered_steps_confirmation_and_review_verdicts():
    run_node(
        """
        import assert from "node:assert/strict";
        import { planHtml } from "./marvis/static/js/v2/plan_view.js";

        const html = planHtml({
          id: "plan-1",
          goal: "<img onerror=alert(1)>",
          status: "running",
          tier: "balanced",
          novel_mode: "explore",
          steps: [
            {
              id: "step-2",
              index: 1,
              title: "Run score",
              status: "done",
              tool_ref: { plugin: "score", tool: "run" },
              depends_on: ["step-1"],
              output_ref: "metrics:step-2<script>",
              review_verdicts: [],
            },
            {
              id: "step-1",
              index: 0,
              title: "Check <sample>",
              status: "awaiting_confirm",
              tool_ref: { plugin: "data<ops>", tool: "join" },
              depends_on: [],
              decision_point: true,
              review_verdicts: [
                {
                  reviewer: "deterministic",
                  passed: false,
                  reasons: ["KS drift < threshold failed"],
                  at: "2026-06-20T00:00:00Z",
                },
                {
                  reviewer: "llm_critic",
                  passed: false,
                  reasons: ["needs human review"],
                  at: "2026-06-20T00:00:01Z",
                },
              ],
            },
          ],
        });

        assert.equal(html.includes("<img onerror"), false);
        assert.ok(html.includes("&lt;img onerror=alert(1)&gt;"));
        assert.ok(html.includes("plan-status-running"));
        assert.ok(html.includes("novel-explore"));
        assert.ok(html.includes('aria-valuenow="50"'));
        assert.ok(html.indexOf('data-step="step-1"') < html.indexOf('data-step="step-2"'));
        assert.ok(html.includes("Check &lt;sample&gt;"));
        assert.ok(html.includes("data&lt;ops&gt;.join"));
        assert.equal(html.includes("metrics:step-2<script>"), false);
        assert.ok(html.includes('data-artifact="metrics:step-2&lt;script&gt;"'));
        assert.ok(html.includes("查看输出"));
        assert.ok(html.includes('class="dp-mark"'));
        assert.ok(html.includes('data-confirm-step="step-1"'));
        assert.equal(html.includes('data-confirm-step="step-2"'), false);
        assert.ok(html.includes("review-verdict reviewer-deterministic failed hard-fail"));
        assert.ok(html.includes("KS drift &lt; threshold failed"));
        assert.ok(html.includes("review-verdict reviewer-llm_critic failed soft-warning"));
        assert.ok(html.includes("<span class=\\"verdict\\">警告</span>"));
        assert.ok(html.includes("needs human review"));
        """
    )


def test_plan_html_renders_no_warning_for_skipped_llm_critique():
    # AGT-6: manual mode (no LLM configured) previously rendered a "警告" line for
    # every step's llm_critic verdict. A status="skipped" verdict must render
    # nothing at all — no pass badge, no warning badge — so an all-manual-mode
    # plan shows zero warning noise even though every step still carries a
    # deterministic verdict.
    run_node(
        """
        import assert from "node:assert/strict";
        import { planHtml } from "./marvis/static/js/v2/plan_view.js";

        const html = planHtml({
          id: "plan-1",
          goal: "manual mode plan",
          status: "done",
          tier: "balanced",
          novel_mode: "plan_ahead",
          steps: [
            {
              id: "step-1",
              index: 0,
              title: "Train model",
              status: "done",
              tool_ref: { plugin: "modeling", tool: "train_model" },
              depends_on: [],
              review_verdicts: [
                {
                  reviewer: "deterministic",
                  passed: true,
                  reasons: [],
                  at: "2026-07-02T00:00:00Z",
                },
                {
                  reviewer: "llm_critic",
                  passed: true,
                  reasons: ["skipped: no LLM configured"],
                  at: "2026-07-02T00:00:01Z",
                  status: "skipped",
                },
              ],
            },
          ],
        });

        assert.equal(html.includes("警告"), false);
        assert.equal(html.includes("skipped: no LLM configured"), false);
        assert.equal(html.includes("reviewer-llm_critic"), false);
        """
    )


def test_artifact_view_fetches_versioned_metric_refs():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderArtifact } from "./marvis/static/js/v2/artifact_view.js";

        const calls = [];
        const container = { innerHTML: "", dataset: {} };
        const result = await renderArtifact(container, "metrics:screen:v2", {
          fetchMetrics: async (id) => {
            calls.push(id);
            return { selected: ["x1"] };
          },
        });

        assert.deepEqual(calls, ["screen:v2"]);
        assert.deepEqual(result, { selected: ["x1"] });
        assert.equal(container.innerHTML.includes("x1"), true);
        """
    )


def test_render_plan_view_updates_from_state_and_can_unsubscribe():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderPlanView } from "./marvis/static/js/v2/plan_view.js";
        import { resetV2State, setPlan } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const container = { innerHTML: "", dataset: {} };
        const unsubscribe = renderPlanView(container);

        assert.ok(container.innerHTML.includes('data-v2-empty="plan"'));
        assert.equal(container.dataset.v2PlanView, "true");

        setPlan({
          id: "plan-1",
          goal: "Assemble a validation workflow",
          status: "validated",
          tier: "conservative",
          steps: [],
        });
        assert.ok(container.innerHTML.includes("Assemble a validation workflow"));
        assert.ok(container.innerHTML.includes("plan-status-validated"));

        const rendered = container.innerHTML;
        unsubscribe();
        setPlan({ id: "plan-2", goal: "Should not render", status: "draft", steps: [] });
        assert.equal(container.innerHTML, rendered);
        """
    )


def test_render_plan_view_stops_active_polling_on_unsubscribe():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderPlanView, startPlanPolling } from "./marvis/static/js/v2/plan_view.js";
        import { resetV2State, setPlan } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        setPlan({ id: "plan-1", goal: "poll", status: "running", steps: [] });
        const cleared = [];
        const poll = startPlanPolling("plan-1", {
          autoStart: false,
          setTimeoutFn: () => 11,
          clearTimeoutFn: (timerId) => cleared.push(timerId),
        });
        poll.timer = 11;

        const container = { innerHTML: "", dataset: {} };
        const unsubscribe = renderPlanView(container);
        unsubscribe();

        assert.deepEqual(cleared, [11]);
        const next = startPlanPolling("plan-1", { autoStart: false });
        assert.notEqual(next, poll);
        """
    )


def test_plan_html_renders_validated_plan_actions_and_hides_terminal_actions():
    run_node(
        """
        import assert from "node:assert/strict";
        import { planHtml } from "./marvis/static/js/v2/plan_view.js";

        const ready = planHtml({
          id: "plan-1",
          goal: "Ready plan",
          status: "validated",
          steps: [],
        });
        assert.ok(ready.includes('class="plan-actions"'));
        assert.ok(ready.includes('data-confirm-plan="plan-1"'));
        assert.ok(ready.includes('data-cancel-plan="plan-1"'));
        assert.ok(ready.includes("确认并运行"));
        assert.ok(ready.includes("取消"));

        const done = planHtml({
          id: "plan-2",
          goal: "Done plan",
          status: "done",
          steps: [],
        });
        assert.equal(done.includes("data-confirm-plan"), false);
        assert.equal(done.includes("data-cancel-plan"), false);
        """
    )


def test_plan_html_renders_failed_step_retry_action():
    run_node(
        """
        import assert from "node:assert/strict";
        import { planHtml } from "./marvis/static/js/v2/plan_view.js";

        const html = planHtml({
          id: "plan-1",
          goal: "Retry failed step",
          status: "failed",
          steps: [
            {
              id: "step-1",
              index: 0,
              title: "Run risky step",
              status: "failed",
              tool_ref: { plugin: "_sample", tool: "echo" },
              inputs: { message: "try again <safe>" },
              failure_envelope: {
                editable_input_schema: {
                  type: "object",
                  properties: {
                    message: { type: "string", default: "retry from envelope <safe>" },
                    threshold: { type: "number", default: 0.42 },
                    enabled: { type: "boolean", default: true },
                    mode: { type: "string", enum: ["fast", "safe"], default: "safe" },
                  },
                },
                downstream_reset_steps: ["step-1", "step-2"],
              },
              depends_on: [],
              review_verdicts: [],
            },
          ],
        });

        assert.ok(html.includes('data-retry-step="step-1"'));
        assert.ok(html.includes('data-retry-inputs-for="step-1"'));
        assert.ok(html.includes('data-retry-input-key="message"'));
        assert.ok(html.includes('data-retry-input-type="number"'));
        assert.ok(html.includes('value="0.42"'));
        assert.ok(html.includes('<option value="true" selected>true</option>'));
        assert.ok(html.includes('<option value="safe" selected>safe</option>'));
        assert.equal(html.includes("try again &lt;safe&gt;"), false);
        assert.ok(html.includes('&quot;message&quot;: &quot;retry from envelope &lt;safe&gt;&quot;'));
        assert.ok(html.includes("将重置"));
        assert.ok(html.includes("<code>step-1</code>"));
        assert.ok(html.includes("<code>step-2</code>"));
        assert.ok(html.includes("重试步骤"));
        assert.equal(html.includes('data-confirm-step="step-1"'), false);
        """
    )


def test_start_plan_polling_dedupes_and_stops_at_terminal_status():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          startPlanPolling,
          stopPlanPolling,
        } from "./marvis/static/js/v2/plan_view.js";
        import {
          getPlan,
          resetV2State,
        } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const scheduled = [];
        const cleared = [];
        const responses = [
          { plan: { id: "plan-1", status: "running", goal: "poll", steps: [] } },
          { plan: { id: "plan-1", status: "done", goal: "poll", steps: [] } },
        ];
        globalThis.fetch = async () => {
          const payload = responses.shift();
          return {
            ok: true,
            status: 200,
            headers: { get: () => "application/json" },
            json: async () => payload,
            text: async () => "",
          };
        };

        const poll = startPlanPolling("plan-1", {
          autoStart: false,
          intervalMs: 25,
          setTimeoutFn: (fn, ms) => {
            scheduled.push({ fn, ms });
            return scheduled.length;
          },
          clearTimeoutFn: (timerId) => cleared.push(timerId),
        });
        const duplicate = startPlanPolling("plan-1", { autoStart: false });
        assert.equal(duplicate, poll);

        await poll.tick();
        assert.equal(getPlan().status, "running");
        assert.equal(scheduled.length, 1);
        assert.equal(scheduled[0].ms, 25);

        await scheduled.shift().fn();
        assert.equal(getPlan().status, "done");
        assert.equal(scheduled.length, 0);
        assert.deepEqual(cleared, [1]);

        const nextPoll = startPlanPolling("plan-1", { autoStart: false });
        assert.notEqual(nextPoll, poll);
        stopPlanPolling("plan-1");
        """
    )


def test_start_plan_polling_ignores_inflight_response_after_stop():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          startPlanPolling,
          stopPlanPolling,
        } from "./marvis/static/js/v2/plan_view.js";
        import {
          getPlan,
          resetV2State,
          setPlan,
        } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        setPlan({ id: "plan-1", status: "cancelled", goal: "stopped", steps: [] });
        let resolveFetch;
        globalThis.fetch = async () => new Promise((resolve) => {
          resolveFetch = resolve;
        });
        const scheduled = [];
        const poll = startPlanPolling("plan-1", {
          autoStart: false,
          setTimeoutFn: (fn, ms) => {
            scheduled.push({ fn, ms });
            return scheduled.length;
          },
        });

        const pending = poll.tick();
        stopPlanPolling("plan-1");
        resolveFetch({
          ok: true,
          status: 200,
          headers: { get: () => "application/json" },
          json: async () => ({ plan: { id: "plan-1", status: "running", goal: "late", steps: [] } }),
          text: async () => "",
        });
        const result = await pending;

        assert.equal(result, null);
        assert.equal(getPlan().status, "cancelled");
        assert.equal(getPlan().goal, "stopped");
        assert.equal(scheduled.length, 0);
        """
    )


def test_start_plan_polling_surfaces_autostart_errors_without_unhandled_rejection():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          startPlanPolling,
          stopPlanPolling,
        } from "./marvis/static/js/v2/plan_view.js";

        const messages = [];
        const unhandled = [];
        const onUnhandled = (error) => unhandled.push(error?.message || String(error));
        process.on("unhandledRejection", onUnhandled);
        globalThis.fetch = async () => ({
          ok: false,
          status: 503,
          headers: { get: () => "application/json" },
          json: async () => ({ detail: "poll failed" }),
          text: async () => "",
        });

        startPlanPolling("plan-error", {
          showError: (message) => messages.push(message),
        });
        await new Promise((resolve) => setTimeout(resolve, 0));

        process.off("unhandledRejection", onUnhandled);
        stopPlanPolling("plan-error");
        assert.deepEqual(messages, ["poll failed"]);
        assert.deepEqual(unhandled, []);
        """
    )
