"""Wide-desktop Candidate Lab controls for governed Strategy Pool apply."""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

from tests.test_frontend_strategy_candidate_lab_interactive_tree import (
    NODE_HARNESS,
)


ROOT = Path(__file__).parents[1]

POOL_APPLY_HARNESS = r"""
function poolProjection(strategyType, entryCount = 1) {
  return {
    kind: "candidate_pool",
    strategy_type: strategyType,
    revision: 3,
    entries: Array.from({ length: entryCount }, (_unused, index) => ({
      entry_id: `pool-entry-${String(index + 1).padStart(32, "0")}`,
      rule_id: `candidate-rule-${String(index + 1).padStart(32, "a")}`,
      enabled: true,
    })),
    total: entryCount,
    truncated: false,
  };
}

function projectedPools(items) {
  return {
    latest: items[0] || null,
    all: items,
    total: items.length,
    truncated: false,
  };
}

function poolApplyPayload(taskId, pools) {
  const payload = payloadFor(taskId, { empty: true });
  payload.pools = projectedPools(pools);
  return payload;
}

function installPoolApplyForm(harness) {
  const strategyType = new FakeSelect("pool_apply_strategy_type");
  const prefix = {
    value: "",
    disabled: false,
    closest() { return null; },
  };
  const submitButton = {
    disabled: false,
    closest() { return null; },
  };
  const error = { textContent: "" };
  const help = { textContent: "" };
  const fields = new Map([
    ["pool_apply_strategy_type", strategyType],
    ["pool_apply_output_prefix", prefix],
  ]);
  const form = {
    dataset: { candidateLabWorkflow: "strategy_pool_apply" },
    querySelector(selector) {
      if (selector === "[data-candidate-lab-form-error]") return error;
      if (selector === "[data-candidate-lab-pool-apply-empty]") return help;
      const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
      return match ? fields.get(match[1]) || null : null;
    },
    querySelectorAll() {
      return [strategyType, prefix, submitButton];
    },
    reset() {
      strategyType.value = "";
      prefix.value = "";
      error.textContent = "";
    },
    closest() { return null; },
  };
  strategyType.form = form;

  const originalOne = harness.panel.querySelector.bind(harness.panel);
  const originalAll = harness.panel.querySelectorAll.bind(harness.panel);
  harness.panel.querySelector = (selector) => (
    selector === '[data-candidate-lab-workflow="strategy_pool_apply"]'
      ? form
      : originalOne(selector)
  );
  harness.panel.querySelectorAll = (selector) => {
    const existing = Array.from(originalAll(selector) || []);
    if (selector === "[data-candidate-lab-form]") {
      return [...existing, form];
    }
    if (selector === "[data-candidate-lab-retry]") return existing;
    return [...existing, strategyType, prefix, submitButton];
  };
  return { error, form, help, prefix, strategyType, submitButton };
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
        {POOL_APPLY_HARNESS}

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


def test_pool_apply_form_collects_only_current_type_and_safe_optional_prefix() -> None:
    run_node(
        r"""
        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        const start = indexHtml.indexOf(
          'data-candidate-lab-workflow="strategy_pool_apply"',
        );
        assert.ok(start >= 0);
        const end = indexHtml.indexOf("</form>", start);
        const formHtml = indexHtml.slice(start, end);
        for (const marker of [
          'data-candidate-lab-field="pool_apply_strategy_type"',
          'data-candidate-lab-field="pool_apply_output_prefix"',
          "应用当前 Strategy Pool",
          "task-owned 派生数据集",
          "不会修改 Pool",
          "不切换工作区数据集",
          "不采纳、发布或部署",
        ]) {
          assert.ok(formHtml.includes(marker), marker);
        }
        for (const forbidden of [
          "expected_pool_revision",
          "expected_pool_snapshot_hash",
          "pool_ref",
          "artifact_id",
          "dataset_id",
          "sample_design",
          "sample_membership",
          "requirements",
          "workspace_revision",
          "workspace_generation",
        ]) {
          assert.ok(!formHtml.includes(forbidden), forbidden);
        }

        const strategyType = new FakeSelect("pool_apply_strategy_type");
        installSelectedOption(
          strategyType,
          "limit",
          { candidateLabProjection: "1", strategyType: "limit" },
        );
        const prefix = { value: "risk_run_01" };
        const fields = new Map([
          ["pool_apply_strategy_type", strategyType],
          ["pool_apply_output_prefix", prefix],
        ]);
        const form = {
          dataset: { candidateLabWorkflow: "strategy_pool_apply" },
          querySelector(selector) {
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? fields.get(match[1]) || null : null;
          },
        };
        assert.ok(
          STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes("strategy_pool_apply"),
        );
        assert.deepEqual(
          collectStrategyCandidateLabRequest(form),
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_apply",
            workflow_inputs: {
              strategy_type: "limit",
              output_prefix: "risk_run_01",
            },
          },
        );

        prefix.value = "   ";
        assert.deepEqual(
          collectStrategyCandidateLabRequest(form).workflow_inputs,
          { strategy_type: "limit" },
        );
        for (const unsafe of [
          "9risk",
          "bad-prefix",
          "../risk",
          "风险策略",
          "with space",
          "a".repeat(49),
        ]) {
          prefix.value = unsafe;
          assert.throws(
            () => collectStrategyCandidateLabRequest(form),
            /ASCII|prefix|前缀/,
            unsafe,
          );
        }

        prefix.value = "safe";
        for (const strategy of [
          "approval",
          "reject",
          "limit",
          "pricing",
          "segmentation",
        ]) {
          installSelectedOption(
            strategyType,
            strategy,
            { candidateLabProjection: "1", strategyType: strategy },
          );
          assert.equal(
            collectStrategyCandidateLabRequest(form).workflow_inputs.strategy_type,
            strategy,
          );
        }
        installSelectedOption(
          strategyType,
          "approval",
          { candidateLabProjection: "1", strategyType: "reject" },
        );
        assert.throws(
          () => collectStrategyCandidateLabRequest(form),
          /受认证|Strategy Pool/,
        );
        """
    )


def test_pool_apply_select_tracks_only_nonempty_projected_pools() -> None:
    run_node(
        r"""
        const uniquePayloads = new Map([[
          "strategy-a",
          poolApplyPayload("strategy-a", [
            poolProjection("approval", 2),
            poolProjection("reject", 0),
          ]),
        ]]);
        const unique = makeHarness({ payloads: uniquePayloads });
        const uniqueApply = installPoolApplyForm(unique);
        await unique.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          uniqueApply.strategyType.options.map((option) => option.value),
          ["", "approval"],
          "empty Pools must not be offered",
        );
        assert.equal(
          uniqueApply.strategyType.value,
          "approval",
          "the only nonempty authenticated Pool may be selected by default",
        );
        assert.match(uniqueApply.help.textContent, /当前受认证|非空/);

        const allTypes = [
          "approval",
          "reject",
          "limit",
          "pricing",
          "segmentation",
        ];
        const multiplePayloads = new Map([[
          "strategy-a",
          poolApplyPayload(
            "strategy-a",
            allTypes.map((strategyType) => poolProjection(strategyType, 1)),
          ),
        ]]);
        const multiple = makeHarness({ payloads: multiplePayloads });
        const multipleApply = installPoolApplyForm(multiple);
        await multiple.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          multipleApply.strategyType.options.map((option) => option.value),
          ["", ...allTypes],
        );
        assert.equal(
          multipleApply.strategyType.value,
          "",
          "multiple nonempty Pools require an explicit operator selection",
        );
        assert.match(multipleApply.help.textContent, /多个|明确选择/);
        for (const option of multipleApply.strategyType.options.slice(1)) {
          assert.equal(option.dataset.candidateLabProjection, "1");
          assert.equal(option.dataset.strategyType, option.value);
        }

        const emptyPayloads = new Map([[
          "strategy-a",
          poolApplyPayload("strategy-a", [
            poolProjection("approval", 0),
            poolProjection("reject", 0),
          ]),
        ]]);
        const empty = makeHarness({ payloads: emptyPayloads });
        const emptyApply = installPoolApplyForm(empty);
        await empty.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          emptyApply.strategyType.options.map((option) => option.value),
          [""],
        );
        assert.equal(emptyApply.strategyType.value, "");
        assert.match(emptyApply.help.textContent, /尚无|非空/);
        """
    )


def test_pool_apply_rechecks_current_nonempty_pool_membership_before_submit() -> None:
    run_node(
        r"""
        const payload = poolApplyPayload(
          "strategy-a",
          [poolProjection("approval", 1)],
        );
        const harness = makeHarness({
          payloads: new Map([["strategy-a", payload]]),
        });
        const apply = installPoolApplyForm(harness);
        await harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.equal(apply.strategyType.value, "approval");

        payload.pools.latest.entries = [];
        payload.pools.latest.total = 0;
        const staleEmpty = await harness.controller.submit(apply.form);
        assert.equal(staleEmpty, null);
        assert.equal(harness.submitCalls.length, 0);
        assert.match(apply.error.textContent, /过期|非空|受认证/);

        const foreignPayload = poolApplyPayload(
          "strategy-a",
          [poolProjection("approval", 1)],
        );
        const foreign = makeHarness({
          payloads: new Map([["strategy-a", foreignPayload]]),
        });
        const foreignApply = installPoolApplyForm(foreign);
        await foreign.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        installSelectedOption(
          foreignApply.strategyType,
          "pricing",
          { candidateLabProjection: "1", strategyType: "pricing" },
        );
        const foreignResult = await foreign.controller.submit(foreignApply.form);
        assert.equal(foreignResult, null);
        assert.equal(foreign.submitCalls.length, 0);
        assert.match(foreignApply.error.textContent, /过期|非空|受认证/);
        """
    )


def test_pool_apply_is_single_flight_and_silently_refreshes_after_settle() -> None:
    run_node(
        r"""
        const payload = poolApplyPayload(
          "strategy-a",
          [poolProjection("pricing", 2)],
        );
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
        const apply = installPoolApplyForm(harness);
        await harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        apply.prefix.value = "price_apply";
        const poolsBefore = JSON.stringify(payload.pools);

        const first = harness.controller.submit(apply.form);
        await Promise.resolve();
        assert.equal(harness.controller.getState().submitting, true);
        assert.equal(apply.strategyType.disabled, true);
        assert.equal(apply.prefix.disabled, true);
        assert.equal(apply.submitButton.disabled, true);

        const second = await harness.controller.submit(apply.form);
        assert.equal(second, null);
        assert.equal(harness.submitCalls.length, 1);
        assert.match(apply.error.textContent, /正在提交|等待/);

        releaseSubmission();
        const accepted = await first;
        assert.equal(accepted.status, "accepted");
        assert.deepEqual(
          harness.submitCalls[0][1],
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_apply",
            workflow_inputs: {
              strategy_type: "pricing",
              output_prefix: "price_apply",
            },
          },
        );
        assert.deepEqual(
          events,
          ["fetch", "submit", "settle", "fetch"],
          "apply must silently refresh Candidate Lab only after app settlement",
        );
        assert.equal(harness.fetchCalls.length, 2);
        assert.equal(JSON.stringify(payload.pools), poolsBefore);
        assert.equal(harness.controller.getState().submitting, false);
        assert.equal(apply.strategyType.value, "pricing");

        const controllerSource = readFileSync(
          "./marvis/static/js/v2/strategy_candidate_lab_controller.js",
          "utf8",
        );
        assert.ok(
          controllerSource.includes(
            'strategyRequest.workflow === "strategy_pool_apply"',
          ),
        );
        assert.ok(
          controllerSource.includes(
            "await refresh(requestTaskId, { silent: true });",
          ),
          "the post-settle refresh must use the silent Candidate Lab path",
        );
        """
    )


def test_pool_apply_reuses_active_plan_and_open_gate_blockers() -> None:
    run_node(
        r"""
        for (const [reason, expected] of [
          ["active_plan", /已有策略计划/],
          ["open_gate", /待处理确认门/],
        ]) {
          const harness = makeHarness({
            getBlockedReason: () => reason,
            payloads: new Map([[
              "strategy-a",
              poolApplyPayload(
                "strategy-a",
                [poolProjection("segmentation", 1)],
              ),
            ]]),
          });
          const apply = installPoolApplyForm(harness);
          await harness.controller.selectTask(
            { id: "strategy-a", task_type: "strategy" },
          );
          assert.equal(apply.strategyType.disabled, true);
          assert.equal(apply.prefix.disabled, true);
          assert.equal(apply.submitButton.disabled, true);
          assert.equal(await harness.controller.submit(apply.form), null);
          assert.equal(harness.submitCalls.length, 0);
          assert.match(apply.error.textContent, expected);
        }
        """
    )
