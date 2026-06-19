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
        assert.ok(html.includes('class="dp-mark"'));
        assert.ok(html.includes('data-confirm-step="step-1"'));
        assert.equal(html.includes('data-confirm-step="step-2"'), false);
        assert.ok(html.includes("review-verdict reviewer-deterministic failed hard-fail"));
        assert.ok(html.includes("KS drift &lt; threshold failed"));
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
