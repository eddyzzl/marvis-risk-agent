"""Wide-desktop Candidate Lab controls for governed current-Pool operations."""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

from tests.test_frontend_strategy_candidate_lab_interactive_tree import (
    NODE_HARNESS,
)


ROOT = Path(__file__).parents[1]

POOL_OPERATIONS_HARNESS = r"""
function operationEntry(hex, position, action = { type: "approval" }) {
  return {
    entry_id: `pool-entry-${hex.repeat(32)}`,
    rule_id: `candidate-rule-${hex.repeat(32)}`,
    position,
    action,
    enabled: true,
  };
}

function operationPool(
  strategyType,
  entries = [operationEntry("a", 0)],
) {
  return {
    kind: "candidate_pool",
    strategy_type: strategyType,
    revision: 4,
    entries,
    total: entries.length,
    truncated: false,
  };
}

function operationPayload(taskId, pools) {
  const payload = payloadFor(taskId, { empty: true });
  payload.pools = {
    latest: pools[0] || null,
    all: pools,
    total: pools.length,
    truncated: false,
  };
  return payload;
}

function simpleField(value = "") {
  return {
    value,
    disabled: false,
    closest() { return null; },
  };
}

function reorderMoveButton(direction) {
  return {
    disabled: false,
    dataset: { candidateLabPoolReorderMove: direction },
    closest(selector) {
      return selector === "[data-candidate-lab-pool-reorder-move]"
        ? this
        : null;
    },
  };
}

function fakeClassList(initialHidden = false) {
  const values = new Set(initialHidden ? ["hidden"] : []);
  return {
    contains(value) { return values.has(value); },
    toggle(value, force) {
      if (force) values.add(value);
      else values.delete(value);
    },
  };
}

function installPoolOperationForms(harness) {
  const fields = {
    compileType: new FakeSelect("pool_compile_strategy_type"),
    removeType: new FakeSelect("pool_remove_strategy_type"),
    removeEntry: new FakeSelect("pool_remove_entry_id"),
    removeReason: simpleField(),
    actionTypePool: new FakeSelect("pool_action_strategy_type"),
    actionEntry: new FakeSelect("pool_action_entry_id"),
    actionType: new FakeSelect("pool_action_type"),
    actionValue: simpleField(),
    actionReason: simpleField(),
    reorderType: new FakeSelect("pool_reorder_strategy_type"),
    reorderOrder: new FakeSelect("pool_reorder_ordered_ids"),
    reorderReason: simpleField(),
  };
  const helps = {
    compile: { textContent: "" },
    remove: { textContent: "" },
    action: { textContent: "" },
    reorder: { textContent: "" },
  };
  const errors = {
    compile: { textContent: "" },
    remove: { textContent: "" },
    action: { textContent: "" },
    reorder: { textContent: "" },
  };
  const actionPanel = {
    classList: fakeClassList(true),
    setAttribute() {},
    querySelectorAll() { return [fields.actionValue]; },
  };
  const buttons = {
    compile: simpleField(),
    remove: simpleField(),
    action: simpleField(),
    reorder: simpleField(),
    reorderUp: reorderMoveButton("up"),
    reorderDown: reorderMoveButton("down"),
  };

  function makeForm(workflow, fieldMap, error, helpSelector, help) {
    const form = {
      dataset: { candidateLabWorkflow: workflow },
      querySelector(selector) {
        if (selector === "[data-candidate-lab-form-error]") return error;
        if (selector === helpSelector) return help;
        if (
          workflow === "strategy_pool_set_action"
          && selector === "[data-candidate-lab-pool-action-value-panel]"
        ) {
          return actionPanel;
        }
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
    for (const field of fieldMap.values()) {
      if (field instanceof FakeSelect) field.form = form;
    }
    return form;
  }

  const forms = {
    compile: makeForm(
      "strategy_pool_compile",
      new Map([["pool_compile_strategy_type", fields.compileType]]),
      errors.compile,
      "[data-candidate-lab-pool-compile-help]",
      helps.compile,
    ),
    remove: makeForm(
      "strategy_pool_remove_entry",
      new Map([
        ["pool_remove_strategy_type", fields.removeType],
        ["pool_remove_entry_id", fields.removeEntry],
        ["pool_remove_reason", fields.removeReason],
      ]),
      errors.remove,
      "[data-candidate-lab-pool-remove-help]",
      helps.remove,
    ),
    action: makeForm(
      "strategy_pool_set_action",
      new Map([
        ["pool_action_strategy_type", fields.actionTypePool],
        ["pool_action_entry_id", fields.actionEntry],
        ["pool_action_type", fields.actionType],
        ["pool_action_value", fields.actionValue],
        ["pool_action_reason", fields.actionReason],
      ]),
      errors.action,
      "[data-candidate-lab-pool-action-help]",
      helps.action,
    ),
    reorder: makeForm(
      "strategy_pool_reorder",
      new Map([
        ["pool_reorder_strategy_type", fields.reorderType],
        ["pool_reorder_ordered_ids", fields.reorderOrder],
        ["pool_reorder_reason", fields.reorderReason],
      ]),
      errors.reorder,
      "[data-candidate-lab-pool-reorder-help]",
      helps.reorder,
    ),
  };
  const byWorkflow = new Map(
    Object.values(forms).map((form) => [
      form.dataset.candidateLabWorkflow,
      form,
    ]),
  );
  const originalOne = harness.panel.querySelector.bind(harness.panel);
  const originalAll = harness.panel.querySelectorAll.bind(harness.panel);
  harness.panel.querySelector = (selector) => {
    const match = selector.match(/data-candidate-lab-workflow="([^"]+)"/);
    return match && byWorkflow.has(match[1])
      ? byWorkflow.get(match[1])
      : originalOne(selector);
  };
  harness.panel.querySelectorAll = (selector) => {
    const existing = Array.from(originalAll(selector) || []);
    if (selector === "[data-candidate-lab-form]") {
      return [...existing, ...Object.values(forms)];
    }
    if (selector === "[data-candidate-lab-retry]") return existing;
    return [
      ...existing,
      ...Object.values(fields),
      ...Object.values(buttons),
    ];
  };
  return {
    actionPanel,
    buttons,
    errors,
    fields,
    forms,
    helps,
  };
}

function installProjectedPoolValidationForm(harness) {
  const type = new FakeSelect("pool_validation_strategy_type");
  const partition = new FakeSelect("pool_validation_partition");
  partition.innerHTML = [
    '<option value="validation">Validation</option>',
    '<option value="oot">OOT</option>',
  ].join("");
  partition.value = "validation";
  const submit = simpleField();
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
      if (selector === "[data-candidate-lab-pool-validation-help]") {
        return help;
      }
      const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
      return match ? fields.get(match[1]) || null : null;
    },
    querySelectorAll() {
      return [type, partition, submit];
    },
    reset() {
      type.value = "";
      partition.value = "validation";
      error.textContent = "";
    },
    closest() { return null; },
  };
  type.form = form;
  partition.form = form;
  const originalOne = harness.panel.querySelector.bind(harness.panel);
  const originalAll = harness.panel.querySelectorAll.bind(harness.panel);
  harness.panel.querySelector = (selector) => (
    selector
      === '[data-candidate-lab-workflow="strategy_pool_validation"]'
      ? form
      : originalOne(selector)
  );
  harness.panel.querySelectorAll = (selector) => {
    const existing = Array.from(originalAll(selector) || []);
    if (selector === "[data-candidate-lab-form]") {
      return [...existing, form];
    }
    if (selector === "[data-candidate-lab-retry]") return existing;
    return [...existing, type, partition, submit];
  };
  return {
    error,
    fields,
    form,
    help,
    partition,
    submit,
    type,
  };
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


def test_pool_operation_forms_collect_only_minimal_user_owned_controls() -> None:
    run_node(
        r"""
        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        const requiredFields = {
          strategy_pool_compile: ["pool_compile_strategy_type"],
          strategy_pool_remove_entry: [
            "pool_remove_strategy_type",
            "pool_remove_entry_id",
            "pool_remove_reason",
          ],
          strategy_pool_set_action: [
            "pool_action_strategy_type",
            "pool_action_entry_id",
            "pool_action_type",
            "pool_action_value",
            "pool_action_reason",
          ],
          strategy_pool_reorder: [
            "pool_reorder_strategy_type",
            "pool_reorder_ordered_ids",
            "pool_reorder_reason",
          ],
        };
        const forbidden = [
          "rule_id",
          "expected_pool_revision",
          "expected_pool_snapshot_hash",
          "revision_id",
          "snapshot_hash",
          "artifact_id",
          "dataset_id",
          "sample_design",
          "sample_membership",
          "requirements",
          "workspace_revision",
          "workspace_generation",
        ];
        for (const [workflow, fields] of Object.entries(requiredFields)) {
          assert.ok(STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes(workflow), workflow);
          const start = indexHtml.indexOf(
            `data-candidate-lab-workflow="${workflow}"`,
          );
          assert.ok(start >= 0, workflow);
          const end = indexHtml.indexOf("</form>", start);
          const formHtml = indexHtml.slice(start, end);
          for (const field of fields) {
            assert.ok(
              formHtml.includes(`data-candidate-lab-field="${field}"`),
              `${workflow}:${field}`,
            );
          }
          for (const field of forbidden) {
            assert.ok(!formHtml.includes(field), `${workflow}:${field}`);
          }
        }
        const reorderStart = indexHtml.indexOf(
          'data-candidate-lab-workflow="strategy_pool_reorder"',
        );
        const reorderHtml = indexHtml.slice(
          reorderStart,
          indexHtml.indexOf("</form>", reorderStart),
        );
        assert.ok(
          reorderHtml.includes(
            '<select data-candidate-lab-field="pool_reorder_ordered_ids"',
          ),
        );
        assert.ok(!reorderHtml.includes(
          '<input data-candidate-lab-field="pool_reorder_ordered_ids"',
        ));
        assert.ok(!reorderHtml.includes(
          '<textarea data-candidate-lab-field="pool_reorder_ordered_ids"',
        ));
        assert.ok(reorderHtml.includes(
          'data-candidate-lab-pool-reorder-move="up"',
        ));
        assert.ok(reorderHtml.includes(
          'data-candidate-lab-pool-reorder-move="down"',
        ));

        function projectionSelect(field, value, extra = {}) {
          const select = new FakeSelect(field);
          installSelectedOption(select, value, {
            candidateLabProjection: "1",
            strategyType: value,
            ...extra,
          });
          return select;
        }
        function form(workflow, fields) {
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? fields.get(match[1]) || null : null;
            },
          };
        }

        const compileType = projectionSelect(
          "pool_compile_strategy_type",
          "approval",
        );
        assert.deepEqual(
          collectStrategyCandidateLabRequest(form(
            "strategy_pool_compile",
            new Map([["pool_compile_strategy_type", compileType]]),
          )),
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_compile",
            workflow_inputs: { strategy_type: "approval" },
          },
        );

        const entryA = `pool-entry-${"a".repeat(32)}`;
        const entryB = `pool-entry-${"b".repeat(32)}`;
        const removeType = projectionSelect(
          "pool_remove_strategy_type",
          "reject",
        );
        const removeEntry = projectionSelect(
          "pool_remove_entry_id",
          entryA,
          { strategyType: "reject", entryId: entryA },
        );
        const removeReason = { value: "业务复核后移除" };
        assert.deepEqual(
          collectStrategyCandidateLabRequest(form(
            "strategy_pool_remove_entry",
            new Map([
              ["pool_remove_strategy_type", removeType],
              ["pool_remove_entry_id", removeEntry],
              ["pool_remove_reason", removeReason],
            ]),
          )),
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_remove_entry",
            workflow_inputs: {
              strategy_type: "reject",
              entry_id: entryA,
              reason: "业务复核后移除",
            },
          },
        );
        removeReason.value = "x".repeat(501);
        assert.throws(
          () => collectStrategyCandidateLabRequest(form(
            "strategy_pool_remove_entry",
            new Map([
              ["pool_remove_strategy_type", removeType],
              ["pool_remove_entry_id", removeEntry],
              ["pool_remove_reason", removeReason],
            ]),
          )),
          /500/,
        );

        function setActionRequest(strategyType, actionType, actionValue) {
          const pool = projectionSelect(
            "pool_action_strategy_type",
            strategyType,
          );
          const entry = projectionSelect(
            "pool_action_entry_id",
            entryA,
            { strategyType, entryId: entryA },
          );
          return collectStrategyCandidateLabRequest(form(
            "strategy_pool_set_action",
            new Map([
              ["pool_action_strategy_type", pool],
              ["pool_action_entry_id", entry],
              ["pool_action_type", { value: actionType }],
              ["pool_action_value", { value: actionValue }],
              ["pool_action_reason", { value: "" }],
            ]),
          ));
        }
        assert.deepEqual(
          setActionRequest("approval", "review", "must-not-leak")
            .workflow_inputs,
          {
            strategy_type: "approval",
            entry_id: entryA,
            action: { type: "review" },
          },
        );
        assert.deepEqual(
          setActionRequest("reject", "approval", "").workflow_inputs.action,
          { type: "approval" },
        );
        assert.deepEqual(
          setActionRequest("limit", "limit", "2500.5").workflow_inputs.action,
          { type: "limit", value: 2500.5 },
        );
        assert.deepEqual(
          setActionRequest("pricing", "pricing", "0.125")
            .workflow_inputs.action,
          { type: "pricing", value: 0.125 },
        );
        assert.deepEqual(
          setActionRequest("segmentation", "segment", "VIP-01")
            .workflow_inputs.action,
          { type: "segment", value: "VIP-01" },
        );
        assert.throws(
          () => setActionRequest("pricing", "reject", ""),
          /不适用|动作/,
        );
        assert.throws(
          () => setActionRequest("limit", "limit", "-1"),
          /非负|额度/,
        );
        assert.throws(
          () => setActionRequest("pricing", "pricing", "1.01"),
          /0.*1|定价/,
        );

        const reorderType = projectionSelect(
          "pool_reorder_strategy_type",
          "limit",
        );
        const ordered = new FakeSelect("pool_reorder_ordered_ids");
        ordered.options = [
          {
            value: entryB,
            selected: false,
            dataset: {
              candidateLabProjection: "1",
              strategyType: "limit",
              entryId: entryB,
            },
          },
          {
            value: entryA,
            selected: true,
            dataset: {
              candidateLabProjection: "1",
              strategyType: "limit",
              entryId: entryA,
            },
          },
        ];
        ordered._value = entryA;
        assert.deepEqual(
          collectStrategyCandidateLabRequest(form(
            "strategy_pool_reorder",
            new Map([
              ["pool_reorder_strategy_type", reorderType],
              ["pool_reorder_ordered_ids", ordered],
              ["pool_reorder_reason", { value: "调整瀑布顺序" }],
            ]),
          )),
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_reorder",
            workflow_inputs: {
              strategy_type: "limit",
              ordered_ids: [entryB, entryA],
              reason: "调整瀑布顺序",
            },
          },
        );
        ordered.options.push({ ...ordered.options[0] });
        assert.throws(
          () => collectStrategyCandidateLabRequest(form(
            "strategy_pool_reorder",
            new Map([
              ["pool_reorder_strategy_type", reorderType],
              ["pool_reorder_ordered_ids", ordered],
              ["pool_reorder_reason", { value: "" }],
            ]),
          )),
          /重复|完整/,
        );
        """
    )


def test_pool_operation_controls_follow_complete_authenticated_pool_projection() -> None:
    run_node(
        r"""
        const approvalEntries = [
          operationEntry("a", 0, { type: "approval" }),
          operationEntry("b", 1, { type: "review" }),
        ];
        const allTypes = [
          operationPool("approval", approvalEntries),
          operationPool("reject", [operationEntry("c", 0, { type: "reject" })]),
          operationPool("limit", [operationEntry("d", 0, { type: "limit", value: 1000 })]),
          operationPool("pricing", [operationEntry("e", 0, { type: "pricing", value: 0.1 })]),
          operationPool("segmentation", [operationEntry("f", 0, { type: "segment", value: "A" })]),
        ];
        const multiple = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload("strategy-a", allTypes),
          ]]),
        });
        const controls = installPoolOperationForms(multiple);
        await multiple.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        const expectedTypes = [
          "",
          "approval",
          "reject",
          "limit",
          "pricing",
          "segmentation",
        ];
        for (const select of [
          controls.fields.compileType,
          controls.fields.removeType,
          controls.fields.actionTypePool,
          controls.fields.reorderType,
        ]) {
          assert.deepEqual(
            select.options.map((option) => option.value),
            expectedTypes,
          );
          assert.equal(select.value, "", "multiple Pools require explicit choice");
        }
        assert.deepEqual(
          controls.fields.removeEntry.options.map((option) => option.value),
          [""],
        );
        assert.deepEqual(
          controls.fields.actionEntry.options.map((option) => option.value),
          [""],
        );
        assert.deepEqual(
          controls.fields.reorderOrder.options.map((option) => option.value),
          [""],
        );

        controls.fields.removeType.value = "approval";
        multiple.controller.handleChange({ target: controls.fields.removeType });
        assert.deepEqual(
          controls.fields.removeEntry.options.map((option) => option.value),
          ["", approvalEntries[0].entry_id, approvalEntries[1].entry_id],
        );
        assert.equal(controls.fields.removeEntry.value, "");

        controls.fields.actionTypePool.value = "approval";
        multiple.controller.handleChange({ target: controls.fields.actionTypePool });
        assert.deepEqual(
          controls.fields.actionEntry.options.map((option) => option.value),
          ["", approvalEntries[0].entry_id, approvalEntries[1].entry_id],
        );
        assert.deepEqual(
          controls.fields.actionType.options.map((option) => option.value),
          ["", "approval", "reject", "review"],
        );
        assert.equal(controls.fields.actionType.value, "");
        assert.equal(controls.actionPanel.classList.contains("hidden"), true);

        controls.fields.actionTypePool.value = "pricing";
        multiple.controller.handleChange({ target: controls.fields.actionTypePool });
        assert.deepEqual(
          controls.fields.actionType.options.map((option) => option.value),
          ["", "pricing"],
        );
        assert.equal(controls.fields.actionType.value, "pricing");
        assert.equal(controls.actionPanel.classList.contains("hidden"), false);
        assert.equal(controls.fields.actionValue.disabled, false);

        controls.fields.reorderType.value = "approval";
        multiple.controller.handleChange({ target: controls.fields.reorderType });
        assert.deepEqual(
          controls.fields.reorderOrder.options.map((option) => option.value),
          approvalEntries.map((entry) => entry.entry_id),
        );
        for (const option of controls.fields.reorderOrder.options) {
          assert.equal(option.dataset.candidateLabProjection, "1");
          assert.equal(option.dataset.strategyType, "approval");
          assert.equal(option.dataset.entryId, option.value);
        }

        const unique = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload(
              "strategy-a",
              [operationPool("approval", approvalEntries)],
            ),
          ]]),
        });
        const uniqueControls = installPoolOperationForms(unique);
        await unique.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        for (const select of [
          uniqueControls.fields.compileType,
          uniqueControls.fields.removeType,
          uniqueControls.fields.actionTypePool,
          uniqueControls.fields.reorderType,
        ]) {
          assert.equal(select.value, "approval");
        }
        assert.equal(uniqueControls.fields.removeEntry.value, "");
        assert.equal(uniqueControls.fields.actionEntry.value, "");
        assert.deepEqual(
          uniqueControls.fields.reorderOrder.options.map((option) => option.value),
          approvalEntries.map((entry) => entry.entry_id),
        );

        const empty = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload(
              "strategy-a",
              [operationPool("approval", [])],
            ),
          ]]),
        });
        const emptyControls = installPoolOperationForms(empty);
        await empty.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        for (const select of [
          emptyControls.fields.compileType,
          emptyControls.fields.removeType,
          emptyControls.fields.actionTypePool,
          emptyControls.fields.reorderType,
        ]) {
          assert.deepEqual(select.options.map((option) => option.value), [""]);
          assert.equal(select.value, "");
        }
        assert.match(emptyControls.helps.compile.textContent, /非空|不能编译/);
        """
    )


def test_pool_validation_type_follows_all_supported_current_pool_projections() -> None:
    run_node(
        r"""
        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        const start = indexHtml.indexOf(
          'data-candidate-lab-workflow="strategy_pool_validation"',
        );
        const formHtml = indexHtml.slice(
          start,
          indexHtml.indexOf("</form>", start),
        );
        assert.ok(
          formHtml.includes("data-candidate-lab-pool-validation-help"),
        );
        assert.ok(!formHtml.includes('<option value="approval"'));
        assert.ok(!formHtml.includes('<option value="reject"'));
        assert.match(formHtml, /等待当前.*Pool.*投影/);

        const uniqueApproval = operationPool(
          "approval",
          [operationEntry("a", 0, { type: "approval" })],
        );
        const uniqueHarness = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload("strategy-a", [
              uniqueApproval,
              operationPool("reject", []),
              operationPool(
                "limit",
                [],
              ),
            ]),
          ]]),
        });
        const unique = installProjectedPoolValidationForm(uniqueHarness);
        await uniqueHarness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          unique.type.options.map((option) => option.value),
          ["", "approval"],
        );
        assert.equal(unique.type.value, "approval");
        assert.match(unique.help.textContent, /唯一|approval|审批/);

        const multipleHarness = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload("strategy-a", [
              operationPool(
                "approval",
                [operationEntry("c", 0, { type: "review" })],
              ),
              operationPool(
                "reject",
                [operationEntry("d", 0, { type: "reject" })],
              ),
              operationPool(
                "pricing",
                [operationEntry("e", 0, { type: "pricing", value: 0.1 })],
              ),
            ]),
          ]]),
        });
        const multiple = installProjectedPoolValidationForm(multipleHarness);
        await multipleHarness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          multiple.type.options.map((option) => option.value),
          ["", "approval", "reject", "pricing"],
        );
        assert.equal(multiple.type.value, "");
        assert.match(multiple.help.textContent, /多个|明确选择/);

        const unavailableHarness = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload("strategy-a", [
              operationPool("approval", []),
              operationPool(
                "segmentation",
                [],
              ),
            ]),
          ]]),
        });
        const unavailable = installProjectedPoolValidationForm(
          unavailableHarness,
        );
        await unavailableHarness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        assert.deepEqual(
          unavailable.type.options.map((option) => option.value),
          [""],
        );
        assert.equal(unavailable.type.value, "");
        assert.match(unavailable.help.textContent, /没有|尚无|非空/);
        assert.throws(
          () => collectStrategyCandidateLabRequest(unavailable.form),
          /受认证|approval|reject|选择/,
        );

        uniqueApproval.entries = [];
        uniqueApproval.total = 0;
        assert.equal(
          await uniqueHarness.controller.submit(unique.form),
          null,
        );
        assert.equal(uniqueHarness.submitCalls.length, 0);
        assert.match(unique.error.textContent, /过期|为空|受认证|刷新/);

        const currentHarness = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload("strategy-a", [
              operationPool(
                "reject",
                [operationEntry("1", 0, { type: "reject" })],
              ),
            ]),
          ]]),
        });
        const current = installProjectedPoolValidationForm(currentHarness);
        await currentHarness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        current.partition.value = "oot";
        const result = await currentHarness.controller.submit(current.form);
        assert.equal(result.status, "accepted");
        assert.deepEqual(
          currentHarness.submitCalls[0][1],
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_validation",
            workflow_inputs: {
              strategy_type: "reject",
              partition: "oot",
            },
          },
        );
        """
    )


def test_pool_reorder_buttons_form_complete_order_without_free_text() -> None:
    run_node(
        r"""
        const entries = [
          operationEntry("a", 0, { type: "approval" }),
          operationEntry("b", 1, { type: "review" }),
          operationEntry("c", 2, { type: "reject" }),
        ];
        const harness = makeHarness({
          payloads: new Map([[
            "strategy-a",
            operationPayload(
              "strategy-a",
              [operationPool("approval", entries)],
            ),
          ]]),
        });
        const controls = installPoolOperationForms(harness);
        await harness.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        const order = controls.fields.reorderOrder;
        assert.deepEqual(
          order.options.map((option) => option.value),
          entries.map((entry) => entry.entry_id),
        );

        order.value = entries[1].entry_id;
        assert.equal(
          harness.controller.handleClick({
            target: controls.buttons.reorderUp,
            preventDefault() {},
          }),
          true,
        );
        assert.deepEqual(
          order.options.map((option) => option.value),
          [
            entries[1].entry_id,
            entries[0].entry_id,
            entries[2].entry_id,
          ],
        );
        assert.equal(order.value, entries[1].entry_id);

        harness.controller.handleClick({
          target: controls.buttons.reorderDown,
          preventDefault() {},
        });
        harness.controller.handleClick({
          target: controls.buttons.reorderDown,
          preventDefault() {},
        });
        assert.deepEqual(
          order.options.map((option) => option.value),
          [
            entries[0].entry_id,
            entries[2].entry_id,
            entries[1].entry_id,
          ],
        );
        assert.equal(order.value, entries[1].entry_id);
        harness.controller.handleClick({
          target: controls.buttons.reorderDown,
          preventDefault() {},
        });
        assert.deepEqual(
          order.options.map((option) => option.value),
          [
            entries[0].entry_id,
            entries[2].entry_id,
            entries[1].entry_id,
          ],
          "moving the last Entry down is a no-op",
        );

        controls.fields.reorderReason.value = "人工调整瀑布顺序";
        assert.deepEqual(
          collectStrategyCandidateLabRequest(controls.forms.reorder),
          {
            request_kind: "standard_workflow",
            workflow: "strategy_pool_reorder",
            workflow_inputs: {
              strategy_type: "approval",
              ordered_ids: [
                entries[0].entry_id,
                entries[2].entry_id,
                entries[1].entry_id,
              ],
              reason: "人工调整瀑布顺序",
            },
          },
        );
        """
    )


def test_pool_operations_recheck_type_entry_and_complete_set_before_submit() -> None:
    run_node(
        r"""
        function setup(entries) {
          const pool = operationPool("approval", entries);
          const payload = operationPayload("strategy-a", [pool]);
          const harness = makeHarness({
            payloads: new Map([["strategy-a", payload]]),
          });
          const controls = installPoolOperationForms(harness);
          return { controls, harness, payload, pool };
        }
        async function selectCurrent(harness) {
          await harness.controller.selectTask(
            { id: "strategy-a", task_type: "strategy" },
          );
        }
        const entryA = operationEntry("a", 0, { type: "approval" });
        const entryB = operationEntry("b", 1, { type: "review" });
        const replacement = operationEntry("c", 0, { type: "reject" });

        const compile = setup([entryA]);
        await selectCurrent(compile.harness);
        compile.pool.entries = [];
        compile.pool.total = 0;
        assert.equal(
          await compile.harness.controller.submit(compile.controls.forms.compile),
          null,
        );
        assert.equal(compile.harness.submitCalls.length, 0);
        assert.match(compile.controls.errors.compile.textContent, /过期|非空|受认证/);

        const remove = setup([entryA]);
        await selectCurrent(remove.harness);
        remove.controls.fields.removeEntry.value = entryA.entry_id;
        remove.pool.entries = [replacement];
        remove.pool.total = 1;
        assert.equal(
          await remove.harness.controller.submit(remove.controls.forms.remove),
          null,
        );
        assert.equal(remove.harness.submitCalls.length, 0);
        assert.match(remove.controls.errors.remove.textContent, /过期|Entry|受认证/);

        const action = setup([entryA]);
        await selectCurrent(action.harness);
        action.controls.fields.actionEntry.value = entryA.entry_id;
        action.controls.fields.actionType.value = "review";
        action.pool.entries = [replacement];
        action.pool.total = 1;
        assert.equal(
          await action.harness.controller.submit(action.controls.forms.action),
          null,
        );
        assert.equal(action.harness.submitCalls.length, 0);
        assert.match(action.controls.errors.action.textContent, /过期|Entry|受认证/);

        const reorder = setup([entryA, entryB]);
        await selectCurrent(reorder.harness);
        reorder.pool.entries = [
          entryA,
          operationEntry("c", 1, { type: "reject" }),
        ];
        reorder.pool.total = 2;
        assert.equal(
          await reorder.harness.controller.submit(reorder.controls.forms.reorder),
          null,
        );
        assert.equal(reorder.harness.submitCalls.length, 0);
        assert.match(
          reorder.controls.errors.reorder.textContent,
          /过期|完整|Entry|受认证/,
        );
        """
    )


def test_pool_operations_reuse_active_plan_and_open_gate_blockers() -> None:
    run_node(
        r"""
        const cases = [
          ["active_plan", /已有策略计划/],
          ["open_gate", /待确认|确认门/],
        ];
        for (const [reason, expectedCopy] of cases) {
          const harness = makeHarness({
            getBlockedReason: () => reason,
            payloads: new Map([[
              "strategy-a",
              operationPayload("strategy-a", [
                operationPool(
                  "approval",
                  [operationEntry("a", 0, { type: "approval" })],
                ),
              ]),
            ]]),
          });
          const controls = installPoolOperationForms(harness);
          await harness.controller.selectTask(
            { id: "strategy-a", task_type: "strategy" },
          );
          for (const field of Object.values(controls.fields)) {
            assert.equal(field.disabled, true, `${reason}: control disabled`);
          }
          for (const form of Object.values(controls.forms)) {
            assert.equal(await harness.controller.submit(form), null);
          }
          assert.equal(harness.submitCalls.length, 0);
          for (const error of Object.values(controls.errors)) {
            assert.match(error.textContent, expectedCopy);
          }
        }
        """
    )


def test_pool_operations_are_single_flight_and_refresh_only_after_settle() -> None:
    run_node(
        r"""
        const entryA = operationEntry("a", 0, { type: "approval" });
        const entryB = operationEntry("b", 1, { type: "review" });
        const cases = [
          ["strategy_pool_compile", "compile"],
          ["strategy_pool_remove_entry", "remove"],
          ["strategy_pool_set_action", "action"],
          ["strategy_pool_reorder", "reorder"],
        ];
        for (const [workflow, formKey] of cases) {
          const payload = operationPayload(
            "strategy-a",
            [operationPool("approval", [entryA, entryB])],
          );
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
          const controls = installPoolOperationForms(harness);
          await harness.controller.selectTask(
            { id: "strategy-a", task_type: "strategy" },
          );
          controls.fields.removeEntry.value = entryA.entry_id;
          controls.fields.actionEntry.value = entryA.entry_id;
          controls.fields.actionType.value = "review";
          const poolsBefore = JSON.stringify(payload.pools);
          const result = await harness.controller.submit(controls.forms[formKey]);
          assert.equal(result.status, "accepted", workflow);
          assert.equal(harness.submitCalls.length, 1, workflow);
          assert.equal(harness.submitCalls[0][1].workflow, workflow);
          assert.deepEqual(
            events,
            ["fetch", "submit", "settle", "fetch"],
            `${workflow} must refresh only after settlement`,
          );
          assert.equal(harness.fetchCalls.length, 2, workflow);
          assert.equal(JSON.stringify(payload.pools), poolsBefore, workflow);
        }

        const payload = operationPayload(
          "strategy-a",
          [operationPool("approval", [entryA])],
        );
        let release;
        const pending = new Promise((resolve) => { release = resolve; });
        const singleFlight = makeHarness({
          getStrategyCandidateLab: async () => payload,
          submitStrategyCandidateLabRequest: async () => {
            await pending;
            return { status: "accepted", messages: [] };
          },
        });
        const controls = installPoolOperationForms(singleFlight);
        await singleFlight.controller.selectTask(
          { id: "strategy-a", task_type: "strategy" },
        );
        controls.fields.removeEntry.value = entryA.entry_id;
        const first = singleFlight.controller.submit(controls.forms.remove);
        await Promise.resolve();
        assert.equal(singleFlight.controller.getState().submitting, true);
        const second = await singleFlight.controller.submit(controls.forms.remove);
        assert.equal(second, null);
        assert.equal(singleFlight.submitCalls.length, 1);
        assert.match(controls.errors.remove.textContent, /正在提交|等待/);
        for (const field of Object.values(controls.fields)) {
          assert.equal(field.disabled, true);
        }
        release();
        await first;
        assert.equal(singleFlight.controller.getState().submitting, false);
        """
    )
