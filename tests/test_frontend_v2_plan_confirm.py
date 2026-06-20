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


def test_plan_confirm_handlers_sequence_plan_step_and_cancel_actions():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachPlanConfirmHandlers } from "./marvis/static/js/v2/plan_confirm.js";
        import { getPlan, resetV2State, setPlan } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        setPlan({ id: "plan-1", status: "awaiting_confirm", steps: [] });
        const calls = [];
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
        };

        const detach = attachPlanConfirmHandlers(root, {
          confirmPlan: async (planId) => calls.push(["confirmPlan", planId]),
          runPlan: async (planId) => calls.push(["runPlan", planId]),
          confirmStep: async (planId, stepId) => calls.push(["confirmStep", planId, stepId]),
          cancelPlan: async (planId) => {
            calls.push(["cancelPlan", planId]);
            return { plan: { id: planId, status: "cancelled", steps: [] } };
          },
          startPlanPolling: (planId) => calls.push(["startPlanPolling", planId]),
          stopPlanPolling: (planId) => calls.push(["stopPlanPolling", planId]),
        });

        const planTarget = {
          closest(selector) {
            return selector === "[data-confirm-plan]"
              ? { dataset: { confirmPlan: "plan-1" } }
              : null;
          },
        };
        await listeners.click({ target: planTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [
          ["confirmPlan", "plan-1"],
          ["runPlan", "plan-1"],
          ["startPlanPolling", "plan-1"],
        ]);

        const stepTarget = {
          closest(selector) {
            return selector === "[data-confirm-step]"
              ? { dataset: { confirmStep: "step-1" } }
              : null;
          },
        };
        await listeners.click({ target: stepTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [
          ["confirmStep", "plan-1", "step-1"],
          ["startPlanPolling", "plan-1"],
        ]);

        const cancelTarget = {
          closest(selector) {
            return selector === "[data-cancel-plan]"
              ? { dataset: { cancelPlan: "plan-1" } }
              : null;
          },
        };
        await listeners.click({ target: cancelTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [
          ["cancelPlan", "plan-1"],
          ["stopPlanPolling", "plan-1"],
        ]);
        assert.equal(getPlan().status, "cancelled");

        detach();
        assert.equal(listeners.click, undefined);
        """
    )


def test_plan_confirm_handlers_ignore_step_without_current_plan():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachPlanConfirmHandlers } from "./marvis/static/js/v2/plan_confirm.js";
        import { resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const calls = [];
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
        };
        attachPlanConfirmHandlers(root, {
          confirmStep: async (...args) => calls.push(args),
          startPlanPolling: (planId) => calls.push(["poll", planId]),
        });
        const stepTarget = {
          closest(selector) {
            return selector === "[data-confirm-step]"
              ? { dataset: { confirmStep: "step-1" } }
              : null;
          },
        };

        await listeners.click({ target: stepTarget, preventDefault() {} });
        assert.deepEqual(calls, []);
        """
    )


def test_plan_confirm_handlers_surface_run_errors_without_polling():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachPlanConfirmHandlers } from "./marvis/static/js/v2/plan_confirm.js";
        import { getPlan, resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const calls = [];
        const messages = [];
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
        };
        attachPlanConfirmHandlers(root, {
          confirmPlan: async (planId) => {
            calls.push(["confirmPlan", planId]);
            return { plan: { id: planId, status: "confirmed", steps: [] } };
          },
          runPlan: async (planId) => {
            calls.push(["runPlan", planId]);
            throw new Error("task already has an active job");
          },
          startPlanPolling: (planId) => calls.push(["startPlanPolling", planId]),
          showError: (message) => messages.push(message),
        });

        const planTarget = {
          closest(selector) {
            return selector === "[data-confirm-plan]"
              ? { dataset: { confirmPlan: "plan-1" } }
              : null;
          },
        };
        await listeners.click({ target: planTarget, preventDefault() {} });

        assert.deepEqual(calls, [
          ["confirmPlan", "plan-1"],
          ["runPlan", "plan-1"],
        ]);
        assert.equal(getPlan().status, "confirmed");
        assert.deepEqual(messages, ["task already has an active job"]);
        """
    )


def test_plan_confirm_handlers_surface_step_errors_without_polling():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachPlanConfirmHandlers } from "./marvis/static/js/v2/plan_confirm.js";
        import { resetV2State, setPlan } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        setPlan({ id: "plan-1", status: "awaiting_confirm", steps: [] });
        const calls = [];
        const messages = [];
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
        };
        attachPlanConfirmHandlers(root, {
          confirmStep: async (planId, stepId) => {
            calls.push(["confirmStep", planId, stepId]);
            throw new Error("step not found");
          },
          startPlanPolling: (planId) => calls.push(["startPlanPolling", planId]),
          showError: (message) => messages.push(message),
        });

        const stepTarget = {
          closest(selector) {
            return selector === "[data-confirm-step]"
              ? { dataset: { confirmStep: "step-1" } }
              : null;
          },
        };
        await listeners.click({ target: stepTarget, preventDefault() {} });

        assert.deepEqual(calls, [["confirmStep", "plan-1", "step-1"]]);
        assert.deepEqual(messages, ["step not found"]);
        """
    )


def test_plan_confirm_handlers_surface_cancel_errors_without_stopping_polling():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachPlanConfirmHandlers } from "./marvis/static/js/v2/plan_confirm.js";

        const calls = [];
        const messages = [];
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
        };
        attachPlanConfirmHandlers(root, {
          cancelPlan: async (planId) => {
            calls.push(["cancelPlan", planId]);
            throw new Error("plan not found");
          },
          stopPlanPolling: (planId) => calls.push(["stopPlanPolling", planId]),
          showError: (message) => messages.push(message),
        });

        const cancelTarget = {
          closest(selector) {
            return selector === "[data-cancel-plan]"
              ? { dataset: { cancelPlan: "plan-1" } }
              : null;
          },
        };
        await listeners.click({ target: cancelTarget, preventDefault() {} });

        assert.deepEqual(calls, [["cancelPlan", "plan-1"]]);
        assert.deepEqual(messages, ["plan not found"]);
        """
    )


def test_render_plan_validation_problems_escapes_structured_problem_text():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          renderPlanValidationProblems,
        } from "./marvis/static/js/v2/plan_confirm.js";

        const container = { innerHTML: "", dataset: {} };
        renderPlanValidationProblems(container, [
          "missing tool <img onerror=alert(1)>",
          { message: "schema mismatch <script>" },
        ]);

        assert.equal(container.dataset.v2PlanProblems, "true");
        assert.ok(container.innerHTML.includes("plan-problems"));
        assert.equal(container.innerHTML.includes("<img onerror"), false);
        assert.equal(container.innerHTML.includes("<script>"), false);
        assert.ok(container.innerHTML.includes("&lt;img onerror=alert(1)&gt;"));
        assert.ok(container.innerHTML.includes("schema mismatch &lt;script&gt;"));

        renderPlanValidationProblems(container, []);
        assert.ok(container.innerHTML.includes('data-v2-empty="plan-problems"'));
        """
    )
