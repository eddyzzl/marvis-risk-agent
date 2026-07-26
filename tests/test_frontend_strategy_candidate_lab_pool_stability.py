"""Wide-desktop Candidate Lab controls for current-Pool stability evidence."""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

from tests.test_frontend_strategy_candidate_lab_interactive_tree import (
    NODE_HARNESS,
)
from tests.test_frontend_strategy_candidate_lab_pool_operations import (
    POOL_OPERATIONS_HARNESS,
)


ROOT = Path(__file__).parents[1]

POOL_STABILITY_HARNESS = r"""
function installPoolStabilityForm(harness) {
  const type = new FakeSelect("pool_stability_strategy_type");
  const submit = simpleField();
  const error = { textContent: "" };
  const help = { textContent: "" };
  const form = {
    dataset: { candidateLabWorkflow: "strategy_pool_stability" },
    querySelector(selector) {
      if (selector === "[data-candidate-lab-form-error]") return error;
      if (selector === "[data-candidate-lab-pool-stability-help]") {
        return help;
      }
      return selector.includes("pool_stability_strategy_type")
        ? type
        : null;
    },
    querySelectorAll() { return [type, submit]; },
    reset() {
      type.value = "";
      error.textContent = "";
    },
    closest() { return null; },
  };
  type.form = form;
  const originalOne = harness.panel.querySelector.bind(harness.panel);
  const originalAll = harness.panel.querySelectorAll.bind(harness.panel);
  harness.panel.querySelector = (selector) => (
    selector === '[data-candidate-lab-workflow="strategy_pool_stability"]'
      ? form
      : originalOne(selector)
  );
  harness.panel.querySelectorAll = (selector) => {
    const existing = Array.from(originalAll(selector) || []);
    if (selector === "[data-candidate-lab-form]") {
      return [...existing, form];
    }
    if (selector === "[data-candidate-lab-retry]") return existing;
    return [...existing, type, submit];
  };
  return { error, form, help, submit, type };
}
"""


def run_node(body: str) -> None:
    script = f"""
        import assert from "node:assert/strict";
        import {{ readFileSync }} from "node:fs";
        import {{
          STRATEGY_CANDIDATE_LAB_WORKFLOWS,
          collectStrategyCandidateLabRequest,
          createStrategyCandidateLabController,
        }} from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        {NODE_HARNESS}
        {POOL_OPERATIONS_HARNESS}
        {POOL_STABILITY_HARNESS}

        {textwrap.dedent(body)}
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Node contract failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_pool_stability_launcher_collects_only_current_pool_type() -> None:
    run_node(
        r"""
        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        const start = indexHtml.indexOf(
          'data-candidate-lab-workflow="strategy_pool_stability"',
        );
        assert.ok(start >= 0);
        const end = indexHtml.indexOf("</form>", start);
        const formHtml = indexHtml.slice(start, end);
        for (const marker of [
          'data-candidate-lab-field="pool_stability_strategy_type"',
          "Strategy Pool 稳定性",
          "development",
          "validation",
          "OOT",
          "PSI",
          "分布漂移",
          "只读",
        ]) {
          assert.ok(formHtml.includes(marker), marker);
        }
        for (const forbidden of [
          "partition",
          "pool_ref",
          "sample_design_ref",
          "expected_pool_revision",
          "dataset_id",
          "target_col",
          "threshold",
          "metric",
        ]) {
          assert.ok(!formHtml.includes(forbidden), forbidden);
        }

        assert.ok(
          STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes(
            "strategy_pool_stability",
          ),
        );
        for (const strategyType of [
          "approval",
          "reject",
          "limit",
          "pricing",
          "segmentation",
        ]) {
          const type = new FakeSelect("pool_stability_strategy_type");
          installSelectedOption(type, strategyType, {
            candidateLabProjection: "1",
            strategyType,
          });
          const form = {
            dataset: { candidateLabWorkflow: "strategy_pool_stability" },
            querySelector(selector) {
              return selector.includes("pool_stability_strategy_type")
                ? type
                : null;
            },
          };
          assert.deepEqual(
            collectStrategyCandidateLabRequest(form),
            {
              request_kind: "standard_workflow",
              workflow: "strategy_pool_stability",
              workflow_inputs: { strategy_type: strategyType },
            },
          );
        }
        """
    )


def test_pool_stability_type_uses_only_complete_unique_pool_projection() -> None:
    run_node(
        r"""
        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        const start = indexHtml.indexOf(
          'data-candidate-lab-workflow="strategy_pool_stability"',
        );
        const formHtml = indexHtml.slice(
          start,
          indexHtml.indexOf("</form>", start),
        );
        for (const strategyType of [
          "approval",
          "reject",
          "limit",
          "pricing",
          "segmentation",
        ]) {
          assert.ok(
            !formHtml.includes(`<option value="${strategyType}"`),
            strategyType,
          );
        }
        assert.match(formHtml, /等待当前.*Pool.*投影/);

        const uniqueHarness = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload("strategy-a", [
              operationPool(
                "pricing",
                [operationEntry("a", 0, { type: "pricing", value: 0.08 })],
              ),
              operationPool("approval", []),
            ]),
          ]]),
        });
        const unique = installPoolStabilityForm(uniqueHarness);
        await uniqueHarness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          unique.type.options.map((option) => option.value),
          ["", "pricing"],
        );
        assert.equal(unique.type.value, "pricing");
        assert.match(unique.help.textContent, /唯一|pricing/);
        assert.equal(
          unique.type.selectedOptions[0].dataset.candidateLabProjection,
          "1",
        );

        const allTypes = [
          "approval",
          "reject",
          "limit",
          "pricing",
          "segmentation",
        ];
        const multipleHarness = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload(
              "strategy-a",
              allTypes.map((strategyType, index) => operationPool(
                strategyType,
                [operationEntry(
                  String(index + 1),
                  0,
                  strategyType === "limit"
                    ? { type: "limit", value: 1000 }
                    : strategyType === "pricing"
                      ? { type: "pricing", value: 0.1 }
                      : strategyType === "segmentation"
                        ? { type: "segment", value: "A" }
                        : { type: strategyType },
                )],
              )),
            ),
          ]]),
        });
        const multiple = installPoolStabilityForm(multipleHarness);
        await multipleHarness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          multiple.type.options.map((option) => option.value),
          ["", ...allTypes],
        );
        assert.equal(multiple.type.value, "");
        assert.match(multiple.help.textContent, /多个|明确选择/);
        multiple.type.value = "limit";
        multipleHarness.controller.render();
        assert.equal(multiple.type.value, "limit");

        const duplicate = operationPool(
          "reject",
          [operationEntry("c", 0, { type: "reject" })],
        );
        const incomplete = operationPool(
          "approval",
          [operationEntry("d", 0, { type: "approval" })],
        );
        incomplete.truncated = true;
        const partial = operationPool(
          "limit",
          [operationEntry("e", 0, { type: "limit", value: 500 })],
        );
        partial.total = 2;
        const unavailableHarness = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload("strategy-a", [
              duplicate,
              { ...duplicate },
              incomplete,
              partial,
              operationPool("pricing", []),
            ]),
          ]]),
        });
        const unavailable = installPoolStabilityForm(unavailableHarness);
        await unavailableHarness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          unavailable.type.options.map((option) => option.value),
          [""],
        );
        assert.equal(unavailable.type.value, "");
        assert.match(unavailable.help.textContent, /尚无|非空|受认证/);
        """
    )


def test_pool_stability_rechecks_current_projection_before_submit() -> None:
    run_node(
        r"""
        function setup() {
          const pool = operationPool(
            "approval",
            [operationEntry("a", 0, { type: "approval" })],
          );
          const payload = operationPayload("strategy-a", [pool]);
          const harness = makeHarness({
            payloads: new Map([["strategy-a", payload]]),
          });
          const stability = installPoolStabilityForm(harness);
          return { harness, payload, pool, stability };
        }

        const stale = setup();
        await stale.harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.equal(stale.stability.type.value, "approval");
        stale.pool.entries = [];
        stale.pool.total = 0;
        assert.equal(
          await stale.harness.controller.submit(stale.stability.form),
          null,
        );
        assert.equal(stale.harness.submitCalls.length, 0);
        assert.match(
          stale.stability.error.textContent,
          /过期|为空|非空|受认证|刷新/,
        );

        const forged = setup();
        await forged.harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        installSelectedOption(forged.stability.type, "pricing", {
          candidateLabProjection: "1",
          strategyType: "pricing",
        });
        assert.equal(
          await forged.harness.controller.submit(forged.stability.form),
          null,
        );
        assert.equal(forged.harness.submitCalls.length, 0);
        assert.match(
          forged.stability.error.textContent,
          /过期|为空|非空|受认证|刷新/,
        );

        const duplicate = setup();
        await duplicate.harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        duplicate.payload.pools.all.push({ ...duplicate.pool });
        duplicate.payload.pools.total = 2;
        assert.equal(
          await duplicate.harness.controller.submit(duplicate.stability.form),
          null,
        );
        assert.equal(duplicate.harness.submitCalls.length, 0);
        assert.match(
          duplicate.stability.error.textContent,
          /过期|为空|非空|受认证|刷新/,
        );
        """
    )


def test_pool_stability_is_single_flight_and_refreshes_only_after_settle() -> None:
    run_node(
        r"""
        const payload = operationPayload("strategy-a", [
          operationPool(
            "segmentation",
            [operationEntry("b", 0, { type: "segment", value: "A" })],
          ),
        ]);
        const events = [];
        let releaseSubmission;
        const pendingSubmission = new Promise((resolve) => {
          releaseSubmission = resolve;
        });
        const harness = makeHarness({
          getStrategyCandidateLab: async () => {
            events.push("fetch");
            return payload;
          },
          submitStrategyCandidateLabRequest: async () => {
            events.push("submit");
            await pendingSubmission;
            return { status: "accepted", messages: [] };
          },
          settleCandidateLabSubmission: async () => {
            events.push("settle");
          },
        });
        const stability = installPoolStabilityForm(harness);
        await harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        const poolsBefore = JSON.stringify(payload.pools);

        const first = harness.controller.submit(stability.form);
        await Promise.resolve();
        assert.equal(harness.controller.getState().submitting, true);
        assert.equal(stability.type.disabled, true);
        assert.equal(stability.submit.disabled, true);

        const second = await harness.controller.submit(stability.form);
        assert.equal(second, null);
        assert.equal(harness.submitCalls.length, 1);
        assert.match(stability.error.textContent, /正在提交|等待/);

        releaseSubmission();
        const accepted = await first;
        assert.equal(accepted.status, "accepted");
        assert.deepEqual(
          harness.submitCalls[0][1],
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_stability",
            workflow_inputs: { strategy_type: "segmentation" },
          },
        );
        assert.deepEqual(
          events,
          ["fetch", "submit", "settle", "fetch"],
          "stability refresh must happen only after app settlement",
        );
        assert.equal(harness.fetchCalls.length, 2);
        assert.equal(JSON.stringify(payload.pools), poolsBefore);
        assert.equal(harness.controller.getState().submitting, false);
        assert.equal(stability.type.value, "segmentation");
        """
    )
