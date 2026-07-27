"""Wide-desktop Candidate Lab controls for authenticated Strategy Pool admission."""

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

POOL_ADD_HARNESS = r"""
const addAsset = `candidate-asset-${"1".repeat(32)}`;
const votingAsset = `candidate-asset-${"2".repeat(32)}`;
const autoSelection = `automatic-tree-leaf-selection-${"3".repeat(32)}`;
const singletonSelection =
  `interactive-tree-frontier-selection-${"4".repeat(32)}`;
const groupSelection =
  `interactive-tree-frontier-group-selection-${"5".repeat(32)}`;
const crossSelection = `cross-matrix-cell-selection-${"6".repeat(32)}`;
const scorecardSelection = `scorecard-cutoff-selection-${"7".repeat(32)}`;

function addSource(sourceKind, sourceId, strategyType = null) {
  const source = {
    source_kind: sourceKind,
    strategy_type: strategyType,
    candidate_stage: "development",
    validation_status: "unvalidated",
  };
  if (["univariate_asset", "voting_candidate"].includes(sourceKind)) {
    source.candidate_asset_id = sourceId;
  } else {
    source.selection_id = sourceId;
  }
  return source;
}

function addSourceCollection(sources, total = sources.length) {
  return {
    latest: sources[0] || null,
    all: sources,
    total,
    truncated: total > sources.length,
  };
}

function addPool(strategyType, defaultAction, entries = [
  operationEntry("a", 0),
]) {
  return {
    ...operationPool(strategyType, entries),
    default_action: defaultAction,
  };
}

function addPayload(taskId, sources, pools = []) {
  const payload = operationPayload(taskId, pools);
  payload.pool_add_sources = addSourceCollection(sources);
  return payload;
}

function installPoolAddForm(harness) {
  const fields = {
    strategyType: new FakeSelect("pool_add_strategy_type"),
    source: new FakeSelect("pool_add_source_id"),
    defaultType: new FakeSelect("pool_add_default_action_type"),
    defaultValue: simpleField(),
    actionType: new FakeSelect("pool_add_action_type"),
    actionValue: simpleField(),
    placement: new FakeSelect("pool_add_placement_mode"),
    reason: simpleField(),
  };
  fields.strategyType.innerHTML = [
    '<option value="">请选择 Strategy Pool 类型</option>',
    '<option value="approval">approval</option>',
    '<option value="reject">reject</option>',
    '<option value="limit">limit</option>',
    '<option value="pricing">pricing</option>',
    '<option value="segmentation">segmentation</option>',
  ].join("");
  fields.placement.innerHTML = [
    '<option value="">请选择 Voting 放置方式</option>',
    '<option value="before_selected_members">before_selected_members</option>',
    '<option value="replace_selected_members">replace_selected_members</option>',
  ].join("");
  const panels = {
    defaultValue: {
      classList: fakeClassList(true),
      setAttribute() {},
      querySelectorAll() { return [fields.defaultValue]; },
    },
    actionValue: {
      classList: fakeClassList(true),
      setAttribute() {},
      querySelectorAll() { return [fields.actionValue]; },
    },
    placement: {
      classList: fakeClassList(true),
      setAttribute() {},
      querySelectorAll() { return [fields.placement]; },
    },
  };
  fields.defaultValue.closest = (selector) => (
    selector === "[data-candidate-lab-pool-add-default-value-panel]"
      ? panels.defaultValue
      : null
  );
  fields.actionValue.closest = (selector) => (
    selector === "[data-candidate-lab-pool-add-action-value-panel]"
      ? panels.actionValue
      : null
  );
  fields.placement.closest = (selector) => (
    selector === "[data-candidate-lab-pool-add-placement-panel]"
      ? panels.placement
      : null
  );
  const help = { textContent: "" };
  const error = { textContent: "" };
  const submit = simpleField();
  const fieldMap = new Map([
    ["pool_add_strategy_type", fields.strategyType],
    ["pool_add_source_id", fields.source],
    ["pool_add_default_action_type", fields.defaultType],
    ["pool_add_default_action_value", fields.defaultValue],
    ["pool_add_action_type", fields.actionType],
    ["pool_add_action_value", fields.actionValue],
    ["pool_add_placement_mode", fields.placement],
    ["pool_add_reason", fields.reason],
  ]);
  const form = {
    dataset: { candidateLabWorkflow: "strategy_pool_add_candidate" },
    querySelector(selector) {
      if (selector === "[data-candidate-lab-form-error]") return error;
      if (selector === "[data-candidate-lab-pool-add-help]") return help;
      if (
        selector === "[data-candidate-lab-pool-add-default-value-panel]"
      ) return panels.defaultValue;
      if (
        selector === "[data-candidate-lab-pool-add-action-value-panel]"
      ) return panels.actionValue;
      if (
        selector === "[data-candidate-lab-pool-add-placement-panel]"
      ) return panels.placement;
      const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
      return match ? fieldMap.get(match[1]) || null : null;
    },
    querySelectorAll() {
      return [...fieldMap.values()];
    },
    reset() {
      for (const field of fieldMap.values()) field.value = "";
      error.textContent = "";
    },
    closest() { return null; },
  };
  for (const field of [
    fields.strategyType,
    fields.source,
    fields.defaultType,
    fields.actionType,
    fields.placement,
  ]) {
    field.form = form;
  }
  const originalOne = harness.panel.querySelector.bind(harness.panel);
  const originalAll = harness.panel.querySelectorAll.bind(harness.panel);
  harness.panel.querySelector = (selector) => (
    selector === '[data-candidate-lab-workflow="strategy_pool_add_candidate"]'
      ? form
      : originalOne(selector)
  );
  harness.panel.querySelectorAll = (selector) => {
    const existing = Array.from(originalAll(selector) || []);
    if (selector === "[data-candidate-lab-form]") {
      return [...existing, form];
    }
    if (selector === "[data-candidate-lab-retry]") return existing;
    return [...existing, ...fieldMap.values(), submit];
  };
  return { error, fields, form, help, panels, submit };
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
        {POOL_ADD_HARNESS}

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


def test_pool_add_form_collects_only_authenticated_minimal_user_controls() -> None:
    run_node(
        r"""
        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        const start = indexHtml.indexOf(
          'data-candidate-lab-workflow="strategy_pool_add_candidate"',
        );
        assert.ok(start >= 0);
        assert.ok(STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes(
          "strategy_pool_add_candidate",
        ));
        const formHtml = indexHtml.slice(
          start,
          indexHtml.indexOf("</form>", start),
        );
        for (const field of [
          "pool_add_strategy_type",
          "pool_add_source_id",
          "pool_add_default_action_type",
          "pool_add_default_action_value",
          "pool_add_action_type",
          "pool_add_action_value",
          "pool_add_placement_mode",
          "pool_add_reason",
        ]) {
          assert.ok(formHtml.includes(`data-candidate-lab-field="${field}"`));
        }
        assert.ok(formHtml.includes(
          '<select data-candidate-lab-field="pool_add_source_id"',
        ));
        for (const forbidden of [
          "artifact_id",
          "expected_artifact_content_hash",
          "expected_asset_hash",
          "dataset_id",
          "sample_design_ref",
          "workspace_revision",
          "pool_revision",
          "snapshot_hash",
          "requirements",
          "rule_id",
        ]) {
          assert.ok(!formHtml.includes(forbidden), forbidden);
        }

        const harness = makeHarness();
        const controls = installPoolAddForm(harness);
        controls.fields.strategyType.value = "approval";
        installSelectedOption(controls.fields.source, autoSelection, {
          candidateLabProjection: "1",
          sourceKind: "automatic_tree_leaf_selection",
          sourceId: autoSelection,
          pointerKind: "selection_id",
          strategyType: "",
        });
        controls.fields.defaultType.value = "approval";
        controls.fields.actionType.value = "reject";
        controls.fields.reason.value = "人工确认入池";
        assert.deepEqual(
          collectStrategyCandidateLabRequest(controls.form),
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_add_candidate",
            workflow_inputs: {
              strategy_type: "approval",
              selection_id: autoSelection,
              default_action: { type: "approval" },
              action: { type: "reject" },
              reason: "人工确认入池",
            },
          },
        );

        controls.fields.strategyType.value = "limit";
        installSelectedOption(controls.fields.source, addAsset, {
          candidateLabProjection: "1",
          sourceKind: "univariate_asset",
          sourceId: addAsset,
          pointerKind: "candidate_asset_id",
          strategyType: "",
        });
        controls.fields.defaultType.value = "limit";
        controls.fields.defaultValue.value = "0";
        controls.fields.actionType.value = "limit";
        controls.fields.actionValue.value = "25000.5";
        controls.fields.reason.value = "";
        assert.deepEqual(
          collectStrategyCandidateLabRequest(controls.form).workflow_inputs,
          {
            strategy_type: "limit",
            candidate_asset_id: addAsset,
            default_action: { type: "limit", value: 0 },
            action: { type: "limit", value: 25000.5 },
          },
        );

        controls.fields.strategyType.value = "approval";
        installSelectedOption(controls.fields.source, votingAsset, {
          candidateLabProjection: "1",
          sourceKind: "voting_candidate",
          sourceId: votingAsset,
          pointerKind: "candidate_asset_id",
          strategyType: "approval",
        });
        controls.fields.defaultType.value = "approval";
        controls.fields.actionType.value = "review";
        controls.fields.placement.value = "before_selected_members";
        assert.deepEqual(
          collectStrategyCandidateLabRequest(controls.form).workflow_inputs,
          {
            strategy_type: "approval",
            candidate_asset_id: votingAsset,
            default_action: { type: "approval" },
            action: { type: "review" },
            placement_mode: "before_selected_members",
          },
        );
        controls.fields.placement.value = "";
        assert.throws(
          () => collectStrategyCandidateLabRequest(controls.form),
          /Voting|放置/,
        );

        installSelectedOption(controls.fields.source, addAsset, {
          candidateLabProjection: "1",
          sourceKind: "univariate_asset",
          sourceId: addAsset,
          pointerKind: "candidate_asset_id",
          strategyType: "",
        });
        controls.fields.placement.value = "replace_selected_members";
        assert.throws(
          () => collectStrategyCandidateLabRequest(controls.form),
          /Voting|普通候选|placement/,
        );
        controls.fields.placement.value = "";
        controls.fields.source.options[0].dataset.candidateLabProjection = "0";
        assert.throws(
          () => collectStrategyCandidateLabRequest(controls.form),
          /受认证|投影|来源/,
        );
        """
    )


def test_pool_add_controls_use_all_materialized_sources_and_restore_pool_default() -> None:
    run_node(
        r"""
        const sources = [
          addSource("univariate_asset", addAsset),
          addSource("automatic_tree_leaf_selection", autoSelection),
          addSource(
            "interactive_tree_frontier_selection",
            singletonSelection,
          ),
          addSource(
            "interactive_tree_frontier_group_selection",
            groupSelection,
          ),
          addSource("cross_matrix_cell_selection", crossSelection),
          addSource("scorecard_cutoff_selection", scorecardSelection),
          addSource("voting_candidate", votingAsset, "approval"),
        ];
        const payload = addPayload(
          "strategy-a",
          sources,
          [addPool("approval", { type: "approval" })],
        );
        const harness = makeHarness({
          payloads: new Map([["strategy-a", payload]]),
        });
        const controls = installPoolAddForm(harness);
        await harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );

        assert.deepEqual(
          controls.fields.strategyType.options.map((option) => option.value),
          ["", "approval", "reject", "limit", "pricing", "segmentation"],
        );
        controls.fields.strategyType.value = "approval";
        harness.controller.handleChange({ target: controls.fields.strategyType });
        assert.deepEqual(
          controls.fields.source.options.map((option) => option.value),
          [
            "",
            addAsset,
            autoSelection,
            singletonSelection,
            groupSelection,
            crossSelection,
            scorecardSelection,
            votingAsset,
          ],
        );
        assert.equal(controls.fields.defaultType.value, "approval");
        assert.equal(controls.fields.defaultType.disabled, true);
        assert.equal(controls.fields.defaultType.dataset.candidateLabPoolAddLocked, "1");
        assert.equal(controls.fields.actionType.disabled, false);
        assert.match(controls.help.textContent, /当前.*Pool|默认动作|不会修改/);

        controls.fields.strategyType.value = "limit";
        harness.controller.handleChange({ target: controls.fields.strategyType });
        assert.deepEqual(
          controls.fields.source.options.map((option) => option.value),
          [
            "",
            addAsset,
            autoSelection,
            singletonSelection,
            groupSelection,
            crossSelection,
            scorecardSelection,
          ],
          "Voting candidate is bound to its approval parent Pool",
        );
        assert.equal(controls.fields.defaultType.value, "limit");
        assert.equal(controls.fields.defaultType.disabled, false);
        assert.equal(
          controls.fields.defaultType.dataset.candidateLabPoolAddLocked,
          undefined,
        );
        assert.equal(
          controls.panels.defaultValue.classList.contains("hidden"),
          false,
        );
        assert.equal(controls.fields.actionType.value, "limit");

        controls.fields.strategyType.value = "approval";
        harness.controller.handleChange({ target: controls.fields.strategyType });
        controls.fields.source.value = votingAsset;
        harness.controller.handleChange({ target: controls.fields.source });
        assert.equal(
          controls.panels.placement.classList.contains("hidden"),
          false,
        );
        controls.fields.source.value = addAsset;
        harness.controller.handleChange({ target: controls.fields.source });
        assert.equal(
          controls.panels.placement.classList.contains("hidden"),
          true,
        );
        assert.equal(controls.fields.placement.value, "");

        const numericDefaultPayload = addPayload(
          "strategy-a",
          [addSource("univariate_asset", addAsset)],
          [addPool("segmentation", { type: "segment", value: 7 })],
        );
        const numericHarness = makeHarness({
          payloads: new Map([["strategy-a", numericDefaultPayload]]),
        });
        const numeric = installPoolAddForm(numericHarness);
        await numericHarness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        numeric.fields.strategyType.value = "segmentation";
        numericHarness.controller.handleChange({
          target: numeric.fields.strategyType,
        });
        numeric.fields.source.value = addAsset;
        numericHarness.controller.handleChange({
          target: numeric.fields.source,
        });
        numeric.fields.actionType.value = "segment";
        numeric.fields.actionValue.value = "tier-a";
        assert.deepEqual(
          collectStrategyCandidateLabRequest(numeric.form)
            .workflow_inputs.default_action,
          { type: "segment", value: 7 },
          "a locked numeric segment default must preserve its typed projection",
        );
        """
    )


def test_pool_add_rechecks_source_and_default_action_before_submission() -> None:
    run_node(
        r"""
        const source = addSource("univariate_asset", addAsset);
        const pool = addPool("approval", { type: "approval" });
        const payload = addPayload("strategy-a", [source], [pool]);
        const harness = makeHarness({
          payloads: new Map([["strategy-a", payload]]),
        });
        const controls = installPoolAddForm(harness);
        await harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        controls.fields.strategyType.value = "approval";
        harness.controller.handleChange({ target: controls.fields.strategyType });
        controls.fields.source.value = addAsset;
        harness.controller.handleChange({ target: controls.fields.source });
        controls.fields.actionType.value = "reject";

        source.candidate_asset_id = `candidate-asset-${"9".repeat(32)}`;
        assert.equal(await harness.controller.submit(controls.form), null);
        assert.equal(harness.submitCalls.length, 0);
        assert.match(controls.error.textContent, /过期|受认证|刷新|来源/);

        source.candidate_asset_id = addAsset;
        pool.default_action = { type: "review" };
        assert.equal(await harness.controller.submit(controls.form), null);
        assert.equal(harness.submitCalls.length, 0);
        assert.match(controls.error.textContent, /默认动作|过期|刷新/);
        """
    )


def test_pool_add_is_single_flight_and_refreshes_only_after_settlement() -> None:
    run_node(
        r"""
        const source = addSource("univariate_asset", addAsset);
        const payload = addPayload("strategy-a", [source], []);
        const events = [];
        const harness = makeHarness({
          getStrategyCandidateLab: async () => {
            events.push("fetch");
            return payload;
          },
          submitStrategyCandidateLabRequest: async () => {
            events.push("submit");
            return { status: "accepted", messages: [] };
          },
          settleCandidateLabSubmission: async () => {
            events.push("settle");
          },
        });
        const controls = installPoolAddForm(harness);
        await harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        controls.fields.strategyType.value = "approval";
        harness.controller.handleChange({ target: controls.fields.strategyType });
        controls.fields.source.value = addAsset;
        harness.controller.handleChange({ target: controls.fields.source });
        controls.fields.defaultType.value = "approval";
        controls.fields.actionType.value = "reject";
        const sourcesBefore = JSON.stringify(payload.pool_add_sources);
        const poolsBefore = JSON.stringify(payload.pools);

        const result = await harness.controller.submit(controls.form);

        assert.equal(result.status, "accepted");
        assert.deepEqual(events, ["fetch", "submit", "settle", "fetch"]);
        assert.equal(harness.fetchCalls.length, 2);
        assert.equal(harness.submitCalls.length, 1);
        assert.deepEqual(harness.submitCalls[0][1], {
          request_kind: "standard_workflow",
          workflow: "strategy_pool_add_candidate",
          workflow_inputs: {
            strategy_type: "approval",
            candidate_asset_id: addAsset,
            default_action: { type: "approval" },
            action: { type: "reject" },
          },
        });
        assert.equal(JSON.stringify(payload.pool_add_sources), sourcesBefore);
        assert.equal(JSON.stringify(payload.pools), poolsBefore);

        let release;
        const pending = new Promise((resolve) => { release = resolve; });
        const blocked = makeHarness({
          getStrategyCandidateLab: async () => payload,
          submitStrategyCandidateLabRequest: async () => {
            await pending;
            return { status: "accepted", messages: [] };
          },
        });
        const blockedControls = installPoolAddForm(blocked);
        await blocked.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        blockedControls.fields.strategyType.value = "approval";
        blocked.controller.handleChange({
          target: blockedControls.fields.strategyType,
        });
        blockedControls.fields.source.value = addAsset;
        blocked.controller.handleChange({
          target: blockedControls.fields.source,
        });
        blockedControls.fields.defaultType.value = "approval";
        blockedControls.fields.actionType.value = "reject";
        const first = blocked.controller.submit(blockedControls.form);
        await Promise.resolve();
        assert.equal(blocked.controller.getState().submitting, true);
        assert.equal(
          await blocked.controller.submit(blockedControls.form),
          null,
        );
        assert.equal(blocked.submitCalls.length, 1);
        assert.match(blockedControls.error.textContent, /正在提交|等待/);
        release();
        await first;
        """
    )
