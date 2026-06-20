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


def test_goal_composer_html_renders_goal_tier_and_novel_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import { goalComposerHtml } from "./marvis/static/js/v2/workflow_create.js";

        const html = goalComposerHtml({
          tiers: [
            { name: "conservative", summary: "Guarded <mode>" },
            { name: "balanced", summary: "Default" },
          ],
          defaultTier: "balanced",
        });

        assert.ok(html.includes('id="goalInput"'));
        assert.ok(html.includes('id="tierSelect"'));
        assert.ok(html.includes('value="balanced" selected'));
        assert.equal(html.includes("Guarded <mode>"), false);
        assert.ok(html.includes("Guarded &lt;mode&gt;"));
        assert.ok(html.includes('id="novelMode"'));
        assert.ok(html.includes('id="createPlanBtn"'));
        """
    )


def test_goal_composer_tracks_capability_tier_state_without_losing_inputs():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderGoalComposer } from "./marvis/static/js/v2/workflow_create.js";
        import {
          resetV2State,
          setCapabilityTiers,
          setSelectedTier,
        } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const controls = {
          "#goalInput": { value: "Keep <goal>" },
          "#tierSelect": { value: "balanced" },
          "#novelMode": { value: "explore" },
        };
        const container = {
          dataset: {},
          innerHTML: "",
          querySelector(selector) {
            return controls[selector] || null;
          },
        };

        const cleanup = renderGoalComposer(container);
        setCapabilityTiers([
          { name: "conservative", summary: "Guarded <mode>" },
          { name: "autonomous", summary: "Higher autonomy" },
        ]);
        controls["#tierSelect"].value = "autonomous";
        setSelectedTier("autonomous");

        assert.equal(container.dataset.v2GoalComposer, "true");
        assert.equal(container.innerHTML.includes("Guarded <mode>"), false);
        assert.ok(container.innerHTML.includes("Guarded &lt;mode&gt;"));
        assert.ok(container.innerHTML.includes('value="autonomous" selected'));
        assert.ok(container.innerHTML.includes("Keep &lt;goal&gt;"));
        assert.ok(container.innerHTML.includes('<option value="explore" selected'));

        cleanup();
        setSelectedTier("conservative");
        assert.ok(container.innerHTML.includes('value="autonomous" selected'));
        """
    )


def test_goal_handlers_create_plan_and_update_state():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachGoalHandlers } from "./marvis/static/js/v2/workflow_create.js";
        import { getPlan, resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const calls = [];
        const listeners = {};
        const controls = {
          "#goalInput": { value: "Join these tables" },
          "#tierSelect": { value: "balanced" },
          "#novelMode": { value: "explore" },
          "[data-plan-problems]": { innerHTML: "", dataset: {} },
        };
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector(selector) { return controls[selector] || null; },
        };

        const detach = attachGoalHandlers(root, "task-1", {
          createPlan: async (taskId, body) => {
            calls.push(["createPlan", taskId, body]);
            return { plan: { id: "plan-1", goal: body.goal, status: "validated", steps: [] } };
          },
        });

        const createTarget = {
          closest(selector) {
            return selector === "#createPlanBtn" ? this : null;
          },
        };
        await listeners.click({ target: createTarget, preventDefault() {} });

        assert.deepEqual(calls, [[
          "createPlan",
          "task-1",
          { goal: "Join these tables", tier: "balanced", novel_mode: "explore" },
        ]]);
        assert.equal(getPlan().id, "plan-1");
        detach();
        assert.equal(listeners.click, undefined);
        """
    )


def test_goal_handlers_resolve_dynamic_task_id():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachGoalHandlers } from "./marvis/static/js/v2/workflow_create.js";

        const calls = [];
        const listeners = {};
        const controls = {
          "#goalInput": { value: "Build a validation workflow" },
          "#tierSelect": { value: "conservative" },
          "#novelMode": { value: "" },
          "[data-plan-problems]": { innerHTML: "", dataset: {} },
        };
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
          querySelector(selector) { return controls[selector] || null; },
        };
        let taskId = "task-a";
        attachGoalHandlers(root, () => taskId, {
          createPlan: async (resolvedTaskId, body) => {
            calls.push([resolvedTaskId, body.goal]);
            return { id: "plan-dynamic", goal: body.goal, status: "validated", steps: [] };
          },
        });

        const createTarget = {
          closest(selector) {
            return selector === "#createPlanBtn" ? this : null;
          },
        };
        taskId = "task-b";
        await listeners.click({ target: createTarget, preventDefault() {} });

        assert.deepEqual(calls, [["task-b", "Build a validation workflow"]]);
        """
    )


def test_goal_handlers_require_task_before_create_plan():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachGoalHandlers } from "./marvis/static/js/v2/workflow_create.js";
        import { resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const problemsSlot = { innerHTML: "", dataset: {} };
        const controls = {
          "#goalInput": { value: "No task yet" },
          "#tierSelect": { value: "balanced" },
          "#novelMode": { value: "" },
          "[data-plan-problems]": problemsSlot,
        };
        const listeners = {};
        let createCalls = 0;
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
          querySelector(selector) { return controls[selector] || null; },
        };
        attachGoalHandlers(root, () => "", {
          createPlan: async () => {
            createCalls += 1;
            return { id: "should-not-create" };
          },
        });

        const createTarget = {
          closest(selector) {
            return selector === "#createPlanBtn" ? this : null;
          },
        };
        await listeners.click({ target: createTarget, preventDefault() {} });

        assert.equal(createCalls, 0);
        assert.equal(problemsSlot.dataset.v2PlanProblems, "true");
        assert.ok(problemsSlot.innerHTML.includes("select or create a task"));
        """
    )


def test_goal_handlers_render_422_plan_validation_problems():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachGoalHandlers } from "./marvis/static/js/v2/workflow_create.js";
        import { resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const problemsSlot = { innerHTML: "", dataset: {} };
        const controls = {
          "#goalInput": { value: "Bad plan" },
          "#tierSelect": { value: "" },
          "#novelMode": { value: "" },
          "[data-plan-problems]": problemsSlot,
        };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
          querySelector(selector) { return controls[selector] || null; },
        };
        const error = new Error("invalid plan");
        error.status = 422;
        error.detail = { problems: ["missing tool <bad>"] };
        attachGoalHandlers(root, "task-1", {
          createPlan: async () => { throw error; },
        });

        const createTarget = {
          closest(selector) {
            return selector === "#createPlanBtn" ? this : null;
          },
        };
        await listeners.click({ target: createTarget, preventDefault() {} });

        assert.ok(problemsSlot.innerHTML.includes("missing tool &lt;bad&gt;"));
        assert.equal(problemsSlot.innerHTML.includes("<bad>"), false);
        """
    )
