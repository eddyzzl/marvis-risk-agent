from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


NODE_HARNESS = r"""
class FakeSelect {
  constructor(fieldName) {
    this.dataset = { candidateLabField: fieldName };
    this.options = [];
    this._value = "";
    this.disabled = false;
    this.form = null;
  }

  set innerHTML(html) {
    const decode = (value) => String(value)
      .replaceAll("&quot;", '"')
      .replaceAll("&#039;", "'")
      .replaceAll("&lt;", "<")
      .replaceAll("&gt;", ">")
      .replaceAll("&amp;", "&");
    this.options = Array.from(
      html.matchAll(/<option value="([^"]*)"([^>]*)>(.*?)<\/option>/g),
    ).map((match) => {
      const dataset = {};
      for (const attribute of match[2].matchAll(
        /data-([a-z0-9-]+)="([^"]*)"/g,
      )) {
        const key = attribute[1].replace(
          /-([a-z])/g,
          (_whole, letter) => letter.toUpperCase(),
        );
        dataset[key] = decode(attribute[2]);
      }
      return {
        value: decode(match[1]),
        selected: false,
        dataset,
      };
    });
    this.value = this.options[0]?.value || "";
  }

  get innerHTML() {
    return "";
  }

  set value(value) {
    this._value = String(value);
    for (const option of this.options) {
      option.selected = option.value === this._value;
    }
  }

  get value() {
    return this._value;
  }

  get selectedOptions() {
    return this.options.filter((option) => option.selected);
  }

  closest(selector) {
    if (selector === "[data-candidate-lab-field]") return this;
    if (
      selector
      === '[data-candidate-lab-workflow="interactive_tree_revision"]'
    ) {
      return this.form;
    }
    return null;
  }
}

const sourceA = `candidate-asset-${"a".repeat(32)}`;
const sourceB = `interactive-tree-revision-${"b".repeat(32)}`;
const historyAncestor = `interactive-tree-revision-${"c".repeat(32)}`;
const foreignSource = `candidate-asset-${"d".repeat(32)}`;
const rootA = `node-${"1".repeat(20)}`;
const otherA = `node-${"2".repeat(20)}`;
const leafA = `node-${"3".repeat(20)}`;
const rootB = `node-${"4".repeat(20)}`;
const leafB = `node-${"5".repeat(20)}`;
const orphanNode = `node-${"e".repeat(20)}`;
const foreignNode = `node-${"f".repeat(20)}`;
const unsafeNodeId = 'node-"><img src=x onerror=alert(1)>';

function node(
  nodeId,
  {
    kind = "split",
    depth = 0,
    visible = true,
    frontier = false,
    canPrune = false,
    feature = "score",
    condition = { field: "score", operator: "<=", value: 600 },
  } = {},
) {
  return {
    node_id: nodeId,
    kind,
    depth,
    feature,
    threshold: 600,
    missing_child: "left",
    condition,
    metrics: { count: 20, bad_rate: 0.2 },
    is_visible: visible,
    is_frontier: frontier,
    can_prune: canPrune,
  };
}

function automaticTreeItem() {
  return {
    kind: "automatic_tree",
    candidate_id: `candidate-${"6".repeat(32)}`,
    detail: {
      asset_id: sourceA,
      source_tree_id: sourceA,
      tree_id: `tree-${"7".repeat(32)}`,
      summary: {
        node_count: 4,
        visible_node_count: 4,
        frontier_node_count: 2,
      },
    },
    lifecycle: {
      candidate_stage: "development",
      validation_status: "unvalidated",
    },
    risks: { red_flags: [], report_info_gaps: [] },
    pointers: {
      root_node_id: rootA,
      nodes: [
        node(rootA, {
          canPrune: true,
          feature: 'income<script>alert("condition")</script>',
        }),
        node(otherA, { depth: 1, canPrune: true, feature: "age" }),
        node(leafA, {
          kind: "leaf",
          depth: 2,
          frontier: true,
          condition: {
            field: 'score<svg onload=alert("condition")>',
            operator: ">",
            value: 600,
          },
        }),
        node(unsafeNodeId, {
          kind: "leaf",
          depth: 2,
          frontier: true,
          condition: { field: "age", operator: "<=", value: 30 },
        }),
      ],
      visible_node_ids: [rootA, otherA, leafA, unsafeNodeId],
      frontier_node_ids: [leafA, unsafeNodeId],
      eligible_prunes: [
        {
          source_tree_id: sourceA,
          node_id: rootA,
          operation: "prune_subtree",
        },
        {
          source_tree_id: sourceA,
          node_id: orphanNode,
          operation: "prune_subtree",
        },
      ],
    },
    total: 4,
    truncated: false,
  };
}

function revisionTreeItem() {
  return {
    kind: "interactive_tree_revision",
    candidate_id: `candidate-${"8".repeat(32)}`,
    detail: {
      revision_id: sourceB,
      source_tree_id: sourceB,
      derived_from_source_tree_id: sourceA,
      parent_revision_id: historyAncestor,
      base_asset_id: sourceA,
      edit: {
        operation: "prune_subtree",
        node_id: otherA,
        reason: 'audit<svg onload=alert("history")>',
      },
      summary: {
        node_count: 2,
        visible_node_count: 2,
        frontier_node_count: 1,
      },
    },
    lifecycle: {
      candidate_stage: "development",
      validation_status: "unvalidated",
    },
    history: [
      {
        revision_id: sourceB,
        parent_revision_id: historyAncestor,
        semantic_tree_id: `semantic-tree-${"9".repeat(32)}`,
        edit: {
          operation: "prune_subtree",
          node_id: otherA,
          reason: 'audit<svg onload=alert("history")>',
        },
      },
      {
        revision_id: historyAncestor,
        parent_revision_id: null,
        semantic_tree_id: `semantic-tree-${"0".repeat(32)}`,
        edit: {
          operation: "prune_subtree",
          node_id: rootA,
          reason: "ancestor",
        },
      },
    ],
    risks: { red_flags: [], report_info_gaps: [] },
    pointers: {
      root_node_id: rootA,
      nodes: [
        node(rootB, { canPrune: true, feature: "utilization" }),
        node(leafB, {
          kind: "leaf",
          depth: 1,
          frontier: true,
          condition: { field: "utilization", operator: "<=", value: 0.8 },
        }),
      ],
      visible_node_ids: [rootB, leafB],
      frontier_node_ids: [leafB],
      eligible_prunes: [{
        source_tree_id: sourceB,
        node_id: rootB,
        operation: "prune_subtree",
      }],
      frontier: [{
        source_node_id: leafB,
        leaf_id: "leaf-b",
        fragment_id: "fragment-b",
        rule_id: "rule-b",
        effect_id: "effect-b",
        condition: { field: "utilization", operator: "<=", value: 0.8 },
        metrics: { count: 10, bad_rate: 0.1 },
      }],
    },
    total: 2,
    truncated: false,
  };
}

function collection(item) {
  return {
    latest: item,
    all: [item],
    total: 1,
    truncated: false,
  };
}

function emptyCollection() {
  return {
    latest: null,
    all: [],
    total: 0,
    truncated: false,
  };
}

function payloadFor(
  taskId,
  {
    empty = false,
    blockedReason = null,
  } = {},
) {
  return {
    schema_version: "strategy.candidate-lab-projection.v3",
    task_id: taskId,
    can_start: !blockedReason,
    blocked_reason: blockedReason,
    active_plan: blockedReason === "active_plan" ? { plan_id: "plan-1" } : null,
    open_gate: blockedReason === "open_gate" ? { gate_id: "gate-1" } : null,
    candidates: {
      automatic_tree: empty
        ? emptyCollection()
        : collection(automaticTreeItem()),
      interactive_tree_revision: empty
        ? emptyCollection()
        : collection(revisionTreeItem()),
    },
    pools: emptyCollection(),
  };
}

function pruneButton(sourceTreeId, nodeId) {
  return {
    disabled: false,
    dataset: { sourceTreeId, nodeId },
    closest(selector) {
      return selector === "[data-candidate-lab-interactive-tree-prune]"
        ? this
        : null;
    },
  };
}

function forgedOption(value, dataset) {
  return { value, selected: true, dataset };
}

function installSelectedOption(select, value, dataset) {
  select.options = [forgedOption(value, dataset)];
  select._value = value;
}

function makeHarness(options = {}) {
  let selectedTask = { id: "strategy-a", task_type: "strategy" };
  const sourceSelect = new FakeSelect("interactive_tree_source_id");
  const nodeSelect = new FakeSelect("interactive_tree_node_id");
  const reasonField = {
    value: "",
    disabled: false,
    closest() {
      return null;
    },
  };
  const submitButton = {
    disabled: false,
    closest() {
      return null;
    },
  };
  const errorTarget = { textContent: "" };
  const help = { textContent: "" };
  const launcher = { open: false };
  const fields = new Map([
    ["interactive_tree_source_id", sourceSelect],
    ["interactive_tree_node_id", nodeSelect],
    ["interactive_tree_reason", reasonField],
  ]);
  const form = {
    dataset: { candidateLabWorkflow: "interactive_tree_revision" },
    querySelector(selector) {
      if (selector === "[data-candidate-lab-form-error]") return errorTarget;
      if (selector === "[data-candidate-lab-tree-help]") return help;
      const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
      return match ? fields.get(match[1]) || null : null;
    },
    querySelectorAll() {
      return [];
    },
    reset() {
      sourceSelect.value = "";
      nodeSelect.value = "";
      reasonField.value = "";
      errorTarget.textContent = "";
    },
    closest(selector) {
      return selector === ".candidate-lab-launcher" ? launcher : null;
    },
  };
  sourceSelect.form = form;
  nodeSelect.form = form;

  const actionButtons = [];
  const panel = {
    classList: { toggle() {} },
    dataset: {},
    setAttribute() {},
    querySelector(selector) {
      if (
        selector
        === '[data-candidate-lab-workflow="interactive_tree_revision"]'
      ) {
        return form;
      }
      return null;
    },
    querySelectorAll(selector) {
      if (selector === "[data-candidate-lab-form]") return [form];
      if (selector === "[data-candidate-lab-retry]") return [];
      return [
        sourceSelect,
        nodeSelect,
        reasonField,
        submitButton,
        ...actionButtons,
      ];
    },
  };
  const results = { innerHTML: "" };
  const status = { textContent: "", dataset: {} };
  const ids = {
    strategyCandidateLabPanel: panel,
    strategyCandidateLabResults: results,
    strategyCandidateLabStatus: status,
  };
  const fetchCalls = [];
  const submitCalls = [];
  const statuses = [];
  const payloads = options.payloads || new Map([
    ["strategy-a", payloadFor("strategy-a")],
    ["strategy-b", payloadFor("strategy-b", { empty: true })],
  ]);
  const controller = createStrategyCandidateLabController({
    $: (id) => ids[id] || null,
    getSelectedTask: () => selectedTask,
    getSelectedTaskId: () => selectedTask?.id || "",
    getBlockedReason: () => options.getBlockedReason?.() || "",
    getStrategyCandidateLab: async (taskId, fetchOptions = {}) => {
      fetchCalls.push({ taskId, fetchOptions });
      if (options.getStrategyCandidateLab) {
        return options.getStrategyCandidateLab(taskId, fetchOptions);
      }
      return payloads.get(taskId);
    },
    submitStrategyCandidateLabRequest: async (...args) => {
      submitCalls.push(args);
      if (options.submitStrategyCandidateLabRequest) {
        return options.submitStrategyCandidateLabRequest(...args);
      }
      return { status: "accepted", messages: [] };
    },
    pollAgentMessagesUntilSettled:
      options.pollAgentMessagesUntilSettled,
    settleCandidateLabSubmission:
      options.settleCandidateLabSubmission,
    refreshAgentMessages: async () => {},
    setActionStatus: (...args) => statuses.push(args),
  });
  return {
    actionButtons,
    controller,
    errorTarget,
    fetchCalls,
    form,
    help,
    launcher,
    nodeSelect,
    panel,
    payloads,
    reasonField,
    registerButton(button) {
      actionButtons.push(button);
      return button;
    },
    results,
    selectTask(task) {
      selectedTask = task;
      return controller.selectTask(task);
    },
    sourceSelect,
    status,
    statuses,
    submitButton,
    submitCalls,
  };
}
"""


def run_node(body: str) -> None:
    script = f"""
        import assert from "node:assert/strict";
        import {{ readFileSync }} from "node:fs";
        import {{
          collectStrategyCandidateLabRequest,
          createStrategyCandidateLabController,
          strategyCandidateLabResultsHtml,
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


def test_interactive_tree_renders_full_topology_history_and_only_exact_prunes():
    run_node(
        r"""
        const indexHtml = readFileSync(
          "./marvis/static/index.html",
          "utf8",
        );
        for (const marker of [
          'data-candidate-lab-workflow="interactive_tree_revision"',
          "data-candidate-lab-interactive-tree-operation",
          'data-candidate-lab-field="interactive_tree_source_id"',
          'data-candidate-lab-field="interactive_tree_node_id"',
          'data-candidate-lab-field="interactive_tree_reason"',
        ]) {
          assert.ok(indexHtml.includes(marker), marker);
        }

        const html = strategyCandidateLabResultsHtml(payloadFor("strategy-a"));
        for (const expected of [
          "自动规则树",
          "交互式树修订",
          "完整节点拓扑",
          "frontier",
          sourceA,
          sourceB,
          rootA,
          otherA,
          leafA,
          rootB,
          leafB,
          historyAncestor,
        ]) {
          assert.ok(html.includes(expected), expected);
        }
        assert.ok(
          (html.match(/完整节点拓扑/g) || []).length >= 2,
          "base tree and immutable revision must each show their topology",
        );
        assert.ok(
          (html.match(/data-candidate-lab-interactive-tree-prune="1"/g)
            || []).length === 2,
          "only the two eligible nodes present in topology get buttons",
        );
        assert.ok(
          html.includes(`data-source-tree-id="${sourceA}"`)
          && html.includes(`data-node-id="${rootA}"`),
        );
        assert.ok(
          html.includes(`data-source-tree-id="${sourceB}"`)
          && html.includes(`data-node-id="${rootB}"`),
        );
        assert.equal(html.includes(`data-node-id="${otherA}"`), false);
        assert.equal(html.includes(`data-node-id="${leafA}"`), false);
        assert.equal(html.includes(`data-node-id="${orphanNode}"`), false);

        assert.ok(
          html.includes("income&lt;script&gt;alert(&quot;condition&quot;)&lt;/script&gt;"),
        );
        assert.ok(html.includes("node-&quot;&gt;&lt;img"));
        assert.ok(html.includes("audit&lt;svg"));
        for (const unsafe of ["<script", "<img src=", "<svg onload="]) {
          assert.equal(html.includes(unsafe), false, unsafe);
        }
        assert.equal(
          /(best|champion|最好|冠军|唯一)/i.test(html),
          false,
          "tree UI must not invent a best, champion, or unique current branch",
        );
        """
    )


def test_tree_row_click_only_selects_then_submit_emits_exact_typed_envelope():
    run_node(
        r"""
        const harness = makeHarness();
        const button = harness.registerButton(pruneButton(sourceA, rootA));
        await harness.selectTask({ id: "strategy-a", task_type: "strategy" });
        harness.sourceSelect.value = "";
        harness.nodeSelect.value = "";

        let prevented = 0;
        assert.equal(
          harness.controller.handleClick({
            target: button,
            preventDefault() { prevented += 1; },
          }),
          true,
        );
        assert.equal(prevented, 1);
        assert.equal(harness.submitCalls.length, 0);
        assert.equal(harness.launcher.open, true);
        assert.equal(harness.sourceSelect.value, sourceA);
        assert.equal(harness.nodeSelect.value, rootA);
        assert.ok(harness.statuses.some(([, tone]) => tone === "info"));

        harness.reasonField.value = "  Remove unstable branch.  ";
        await harness.controller.submit(harness.form);
        assert.equal(harness.submitCalls.length, 1);
        assert.deepEqual(harness.submitCalls[0], [
          "strategy-a",
          {
            request_kind: "standard_workflow",
            workflow: "interactive_tree_revision",
            workflow_inputs: {
              source_tree_id: sourceA,
              node_id: rootA,
              operation: "prune_subtree",
              reason: "Remove unstable branch.",
            },
          },
          "创建不可变交互式树修订",
        ]);

        harness.reasonField.value = "   ";
        assert.deepEqual(
          collectStrategyCandidateLabRequest(harness.form),
          {
            request_kind: "standard_workflow",
            workflow: "interactive_tree_revision",
            workflow_inputs: {
              source_tree_id: sourceA,
              node_id: rootA,
              operation: "prune_subtree",
            },
          },
        );
        """
    )


def test_interactive_tree_revalidates_click_and_submit_against_current_payload():
    run_node(
        r"""
        const clickHarness = makeHarness();
        const orphan = clickHarness.registerButton(
          pruneButton(sourceA, orphanNode),
        );
        await clickHarness.selectTask({
          id: "strategy-a",
          task_type: "strategy",
        });
        clickHarness.sourceSelect.value = "";
        clickHarness.nodeSelect.value = "";
        assert.equal(
          clickHarness.controller.handleClick({
            target: orphan,
            preventDefault() {},
          }),
          true,
        );
        assert.equal(clickHarness.sourceSelect.value, "");
        assert.equal(clickHarness.nodeSelect.value, "");
        assert.equal(clickHarness.submitCalls.length, 0);
        assert.ok(
          clickHarness.statuses.some(([message, tone]) => (
            tone === "error" && message.includes("受认证")
          )),
        );

        for (const forged of [
          { sourceTreeId: foreignSource, nodeId: foreignNode },
          { sourceTreeId: sourceA, nodeId: orphanNode },
        ]) {
          const harness = makeHarness();
          await harness.selectTask({
            id: "strategy-a",
            task_type: "strategy",
          });
          installSelectedOption(
            harness.sourceSelect,
            forged.sourceTreeId,
            {
              candidateLabProjection: "1",
              sourceTreeId: forged.sourceTreeId,
            },
          );
          installSelectedOption(
            harness.nodeSelect,
            forged.nodeId,
            {
              candidateLabProjection: "1",
              sourceTreeId: forged.sourceTreeId,
              operation: "prune_subtree",
            },
          );

          const result = await harness.controller.submit(harness.form);
          assert.equal(result, null);
          assert.equal(
            harness.submitCalls.length,
            0,
            `forged pointer ${forged.sourceTreeId}/${forged.nodeId}`,
          );
          assert.ok(harness.errorTarget.textContent.includes("受认证"));
        }

        const operationHarness = makeHarness();
        await operationHarness.selectTask({
          id: "strategy-a",
          task_type: "strategy",
        });
        installSelectedOption(
          operationHarness.sourceSelect,
          sourceA,
          { candidateLabProjection: "1", sourceTreeId: sourceA },
        );
        installSelectedOption(
          operationHarness.nodeSelect,
          rootA,
          {
            candidateLabProjection: "1",
            sourceTreeId: sourceA,
            operation: "delete_tree",
          },
        );
        assert.equal(
          await operationHarness.controller.submit(operationHarness.form),
          null,
        );
        assert.equal(operationHarness.submitCalls.length, 0);
        """
    )


def test_tree_actions_disable_while_blocked_or_submitting_and_switch_clears_pending():
    run_node(
        r"""
        for (const blockedReason of ["active_plan", "open_gate"]) {
          const payloads = new Map([[
            "strategy-a",
            payloadFor("strategy-a", { blockedReason }),
          ]]);
          const harness = makeHarness({ payloads });
          const button = harness.registerButton(pruneButton(sourceA, rootA));
          await harness.selectTask({
            id: "strategy-a",
            task_type: "strategy",
          });
          assert.equal(button.disabled, true, blockedReason);
          assert.equal(harness.sourceSelect.disabled, true, blockedReason);
          assert.equal(harness.nodeSelect.disabled, true, blockedReason);
          assert.equal(harness.reasonField.disabled, true, blockedReason);
          assert.equal(harness.submitButton.disabled, true, blockedReason);

          harness.sourceSelect.value = "";
          harness.nodeSelect.value = "";
          harness.controller.handleClick({
            target: button,
            preventDefault() {},
          });
          assert.equal(harness.sourceSelect.value, "");
          assert.equal(harness.nodeSelect.value, "");
          assert.equal(harness.submitCalls.length, 0);
        }

        let resolveSubmission;
        const pendingSubmission = new Promise((resolve) => {
          resolveSubmission = resolve;
        });
        const harness = makeHarness({
          submitStrategyCandidateLabRequest: () => pendingSubmission,
        });
        const button = harness.registerButton(pruneButton(sourceA, rootA));
        await harness.selectTask({
          id: "strategy-a",
          task_type: "strategy",
        });
        harness.controller.handleClick({
          target: button,
          preventDefault() {},
        });
        harness.reasonField.value = "pending task A reason";

        const submitPromise = harness.controller.submit(harness.form);
        await Promise.resolve();
        assert.equal(harness.controller.getState().submitting, true);
        for (const control of [
          button,
          harness.sourceSelect,
          harness.nodeSelect,
          harness.reasonField,
          harness.submitButton,
        ]) {
          assert.equal(control.disabled, true);
        }

        await harness.selectTask({
          id: "strategy-b",
          task_type: "strategy",
        });
        assert.equal(harness.controller.getState().taskId, "strategy-b");
        assert.equal(harness.controller.getState().submitting, false);
        assert.equal(harness.sourceSelect.value, "");
        assert.equal(harness.nodeSelect.value, "");
        assert.equal(harness.reasonField.value, "");

        resolveSubmission({ status: "accepted", messages: [] });
        await submitPromise;
        assert.equal(harness.controller.getState().taskId, "strategy-b");
        assert.equal(harness.controller.getState().submitting, false);
        assert.deepEqual(
          harness.fetchCalls.map(({ taskId }) => taskId),
          ["strategy-a", "strategy-b"],
        );
        """
    )


def test_successful_tree_submit_reloads_once_only_from_external_settle():
    run_node(
        r"""
        const settleCalls = [];
        const harness = makeHarness({
          settleCandidateLabSubmission: async (taskId) => {
            settleCalls.push(taskId);
          },
        });
        const button = harness.registerButton(pruneButton(sourceA, rootA));
        await harness.selectTask({
          id: "strategy-a",
          task_type: "strategy",
        });
        assert.equal(harness.fetchCalls.length, 1);
        harness.controller.handleClick({
          target: button,
          preventDefault() {},
        });

        const result = await harness.controller.submit(harness.form);
        assert.equal(result.status, "accepted");
        assert.equal(harness.submitCalls.length, 1);
        assert.deepEqual(
          settleCalls,
          ["strategy-a"],
          "accepted tree submit must enter the app-owned task settle path once",
        );
        assert.equal(
          harness.fetchCalls.length,
          1,
          "accepted submit must not directly reload Candidate Lab",
        );

        await harness.controller.refresh("strategy-a", { silent: true });
        assert.equal(
          harness.fetchCalls.length,
          2,
          "the app settle hook owns exactly one post-settle reload",
        );

        const appJs = readFileSync("./marvis/static/app.js", "utf8");
        assert.ok(appJs.includes("settleCandidateLabSubmission:"));
        const wiring = appJs.slice(
          appJs.indexOf("settleCandidateLabSubmission:"),
          appJs.indexOf("settleCandidateLabSubmission:") + 500,
        );
        assert.ok(wiring.includes("pollValidationProgress("));
        assert.ok(wiring.includes("settleWhenServerIdle: true"));
        """
    )


def test_app_wires_candidate_lab_submit_to_the_existing_task_settle_poller():
    app_js = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")

    assert "settleCandidateLabSubmission: (taskId) => pollValidationProgress(" in app_js
    assert "terminalTaskStatuses,\n    taskId,\n    { settleWhenServerIdle: true }" in app_js
    assert "await refreshStrategyCandidateLabAfterSettled(taskId, polledTask)" in app_js
