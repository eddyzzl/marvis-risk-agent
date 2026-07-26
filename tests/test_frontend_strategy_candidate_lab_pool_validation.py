"""Wide-desktop Candidate Lab controls for independent Pool replay evidence."""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

from tests.test_frontend_strategy_candidate_lab_interactive_tree import (
    NODE_HARNESS,
)


ROOT = Path(__file__).parents[1]


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

def test_pool_validation_form_collects_only_type_and_partition() -> None:
    run_node(
        r"""
        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        const start = indexHtml.indexOf(
          'data-candidate-lab-workflow="strategy_pool_validation"',
        );
        assert.ok(start >= 0);
        const end = indexHtml.indexOf("</form>", start);
        const formHtml = indexHtml.slice(start, end);
        for (const marker of [
          'data-candidate-lab-field="pool_validation_strategy_type"',
          'data-candidate-lab-field="pool_validation_partition"',
          "独立样本回放验证",
          "independent replay evidence",
        ]) {
          assert.ok(formHtml.includes(marker), marker);
        }
        for (const forbidden of [
          "pool_ref",
          "sample_design_ref",
          "expected_pool_revision",
          "dataset_id",
          "target_col",
          "population",
          "comparison_mode",
          "PSI",
          "稳定性",
        ]) {
          assert.ok(!formHtml.includes(forbidden), forbidden);
        }

        const type = new FakeSelect("pool_validation_strategy_type");
        installSelectedOption(type, "approval", {
          candidateLabProjection: "1",
          strategyType: "approval",
        });
        const fields = new Map([
          ["pool_validation_strategy_type", type],
          ["pool_validation_partition", { value: "validation" }],
        ]);
        const form = {
          dataset: { candidateLabWorkflow: "strategy_pool_validation" },
          querySelector(selector) {
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? fields.get(match[1]) || null : null;
          },
        };
        assert.ok(
          STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes("strategy_pool_validation"),
        );
        assert.deepEqual(
          collectStrategyCandidateLabRequest(form),
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_validation",
            workflow_inputs: {
              strategy_type: "approval",
              partition: "validation",
            },
          },
        );
        installSelectedOption(type, "limit", {
          candidateLabProjection: "1",
          strategyType: "limit",
        });
        assert.throws(
          () => collectStrategyCandidateLabRequest(form),
          /approval|reject/,
        );
        installSelectedOption(type, "reject", {
          candidateLabProjection: "1",
          strategyType: "reject",
        });
        fields.get("pool_validation_partition").value = "development";
        assert.throws(
          () => collectStrategyCandidateLabRequest(form),
          /validation|oot/,
        );
        """
    )


def test_pool_validation_form_reuses_blocking_and_single_flight_submission() -> None:
    run_node(
        r"""
        function validationPayload() {
          const payload = payloadFor("strategy-a", { empty: true });
          const pool = {
            kind: "candidate_pool",
            strategy_type: "reject",
            revision: 2,
            entries: [{
              entry_id: `pool-entry-${"a".repeat(32)}`,
              position: 0,
              action: { type: "reject" },
            }],
            total: 1,
            truncated: false,
          };
          payload.pools = {
            latest: pool,
            all: [pool],
            total: 1,
            truncated: false,
          };
          return payload;
        }

        function installValidationForm(harness) {
          const type = new FakeSelect("pool_validation_strategy_type");
          const partition = new FakeSelect("pool_validation_partition");
          partition.innerHTML = [
            '<option value="validation">Validation</option>',
            '<option value="oot">OOT</option>',
          ].join("");
          partition.value = "oot";
          const submit = { disabled: false, closest() { return null; } };
          const error = { textContent: "" };
          const help = { textContent: "" };
          const fields = new Map([
            ["pool_validation_strategy_type", type],
            ["pool_validation_partition", partition],
          ]);
          const form = {
            dataset: { candidateLabWorkflow: "strategy_pool_validation" },
            querySelector(selector) {
              if (selector === "[data-candidate-lab-form-error]") return error;
              if (
                selector === "[data-candidate-lab-pool-validation-help]"
              ) {
                return help;
              }
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? fields.get(match[1]) || null : null;
            },
            querySelectorAll() { return [type, partition, submit]; },
            reset() {
              type.value = "";
              partition.value = "oot";
              error.textContent = "";
            },
            closest() { return null; },
          };
          type.form = form;
          partition.form = form;
          const originalOne = harness.panel.querySelector.bind(harness.panel);
          const originalAll = harness.panel.querySelectorAll.bind(harness.panel);
          harness.panel.querySelector = (selector) => (
            selector === '[data-candidate-lab-workflow="strategy_pool_validation"]'
              ? form
              : originalOne(selector)
          );
          harness.panel.querySelectorAll = (selector) => {
            if (selector === "[data-candidate-lab-form]") {
              return [...originalAll(selector), form];
            }
            return [...originalAll(selector), type, partition, submit];
          };
          return { error, form };
        }

        let release;
        const pending = new Promise((resolve) => { release = resolve; });
        const harness = makeHarness({
          payloads: new Map([["strategy-a", validationPayload()]]),
          submitStrategyCandidateLabRequest: async () => {
            await pending;
            return { status: "accepted", messages: [] };
          },
        });
        const validation = installValidationForm(harness);
        await harness.controller.selectTask({ id: "strategy-a", task_type: "strategy" });
        const first = harness.controller.submit(validation.form);
        const second = await harness.controller.submit(validation.form);
        assert.equal(second, null);
        assert.equal(harness.submitCalls.length, 1);
        release();
        await first;
        assert.deepEqual(
          harness.submitCalls[0][1],
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_validation",
            workflow_inputs: {
              strategy_type: "reject",
              partition: "oot",
            },
          },
        );

        const blocked = makeHarness({
          getBlockedReason: () => "active_plan",
          payloads: new Map([["strategy-a", validationPayload()]]),
        });
        const blockedValidation = installValidationForm(blocked);
        await blocked.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.equal(await blocked.controller.submit(blockedValidation.form), null);
        assert.equal(blocked.submitCalls.length, 0);
        assert.match(blockedValidation.error.textContent, /已有策略计划/);
        """
    )
