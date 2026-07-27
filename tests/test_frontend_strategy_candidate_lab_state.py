"""Task-scoped Candidate Lab view state stays bounded and non-sensitive."""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).parents[1]


def run_node(body: str) -> None:
    script = f"""
        import assert from "node:assert/strict";
        import {{
          captureStrategyCandidateLabViewState,
          loadStrategyCandidateLabViewState,
          persistStrategyCandidateLabViewState,
          restoreStrategyCandidateLabViewState,
          strategyCandidateLabStateStorageKey,
        }} from "./marvis/static/js/v2/strategy_candidate_lab_state.js";
        import {{
          createStrategyCandidateLabController,
        }} from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function option(value, selected = false) {{
          return {{ value, selected }};
        }}

        function select(fieldName, values, selected, multiple = false) {{
          const field = {{
            dataset: {{ candidateLabField: fieldName }},
            tagName: "SELECT",
            type: "select-one",
            multiple,
            options: values.map((value) => option(
              value,
              Array.isArray(selected)
                ? selected.includes(value)
                : selected === value,
            )),
            _value: Array.isArray(selected) ? "" : selected,
          }};
          Object.defineProperty(field, "selectedOptions", {{
            get() {{ return field.options.filter((item) => item.selected); }},
          }});
          Object.defineProperty(field, "innerHTML", {{
            set(html) {{
              field.options = Array.from(
                String(html).matchAll(
                  /<option value="([^"]*)"([^>]*)>(.*?)<\\/option>/g,
                ),
              ).map((match) => ({{
                value: match[1],
                selected: false,
                dataset: Object.fromEntries(
                  Array.from(
                    match[2].matchAll(/data-([a-z0-9-]+)="([^"]*)"/g),
                  ).map((item) => [
                    item[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase()),
                    item[2],
                  ]),
                ),
              }}));
              field.value = "";
            }},
          }});
          Object.defineProperty(field, "value", {{
            get() {{ return field._value; }},
            set(value) {{
              field._value = String(value);
              for (const item of field.options) {{
                item.selected = item.value === field._value;
              }}
            }},
          }});
          return field;
        }}

        function input(fieldName, value, type = "text") {{
          return {{
            dataset: {{ candidateLabField: fieldName }},
            tagName: "INPUT",
            type,
            value,
          }};
        }}

        function checkbox(fieldName, value, checked) {{
          return {{
            dataset: {{ candidateLabField: fieldName }},
            tagName: "INPUT",
            type: "checkbox",
            value,
            checked,
          }};
        }}

        function form(workflow, fields, launcher) {{
          return {{
            dataset: {{ candidateLabWorkflow: workflow }},
            querySelectorAll(selector) {{
              return selector === "[data-candidate-lab-field]" ? fields : [];
            }},
            closest(selector) {{
              return selector === ".candidate-lab-launcher" ? launcher : null;
            }},
          }};
        }}

        function root(forms) {{
          return {{
            querySelectorAll(selector) {{
              return selector === "[data-candidate-lab-form]" ? forms : [];
            }},
          }};
        }}

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


def test_state_captures_only_gated_selections_and_safe_numeric_cutpoints() -> None:
    run_node(
        r"""
        const launcher = { open: true };
        const fields = [
          select("x_method", ["equal_frequency", "manual"], "manual"),
          select("selected_cell_ids", ["cell-a", "cell-b"], ["cell-b"], true),
          checkbox("methods", "equal_frequency", false),
          checkbox("methods", "manual", true),
          input("x_manual_breakpoints", "30, 50"),
          input("max_trials", "500", "number"),
          input("x_feature", "sensitive_column_name"),
          input("selection_reason", "sensitive business explanation"),
        ];
        const snapshot = captureStrategyCandidateLabViewState(root([
          form("cross_matrix_analysis", fields, launcher),
        ]));

        assert.deepEqual(snapshot.open_workflows, ["cross_matrix_analysis"]);
        const byField = new Map(snapshot.fields.map((item) => [item.field, item]));
        assert.deepEqual(byField.get("x_method").values, ["manual"]);
        assert.deepEqual(byField.get("selected_cell_ids").values, ["cell-b"]);
        assert.deepEqual(byField.get("methods").values, ["manual"]);
        assert.deepEqual(
          byField.get("x_manual_breakpoints").values,
          ["30, 50"],
        );
        assert.deepEqual(byField.get("max_trials").values, ["500"]);
        assert.equal(byField.has("x_feature"), false);
        assert.equal(byField.has("selection_reason"), false);
        assert.ok(!JSON.stringify(snapshot).includes("sensitive"));
        """
    )


def test_state_restore_rejects_stale_selectors_and_restores_dependencies() -> None:
    run_node(
        r"""
        const launcher = { open: false };
        const method = select(
          "x_method",
          ["equal_frequency", "manual"],
          "equal_frequency",
        );
        const cells = select(
          "selected_cell_ids",
          ["cell-a", "cell-b"],
          [],
          true,
        );
        const points = input("x_manual_breakpoints", "");
        const reason = input("selection_reason", "");
        const currentRoot = root([
          form(
            "cross_matrix_analysis",
            [method, cells, points, reason],
            launcher,
          ),
        ]);
        const restored = restoreStrategyCandidateLabViewState(currentRoot, {
          schema_version: "strategy.candidate-lab-view-state.v1",
          open_workflows: ["cross_matrix_analysis"],
          fields: [
            {
              workflow: "cross_matrix_analysis",
              field: "x_method",
              kind: "select",
              values: ["manual"],
            },
            {
              workflow: "cross_matrix_analysis",
              field: "selected_cell_ids",
              kind: "select",
              values: ["cell-b", "stale-cell"],
            },
            {
              workflow: "cross_matrix_analysis",
              field: "x_manual_breakpoints",
              kind: "scalar",
              values: ["30,50"],
            },
            {
              workflow: "cross_matrix_analysis",
              field: "selection_reason",
              kind: "scalar",
              values: ["must-not-restore"],
            },
          ],
        });

        assert.equal(restored, true);
        assert.equal(method.value, "manual");
        assert.deepEqual(
          cells.selectedOptions.map((item) => item.value),
          ["cell-b"],
        );
        assert.equal(points.value, "30,50");
        assert.equal(reason.value, "");
        assert.equal(launcher.open, true);
        """
    )


def test_state_storage_is_task_scoped_and_malformed_data_fails_closed() -> None:
    run_node(
        r"""
        const data = new Map();
        const storage = {
          getItem(key) { return data.has(key) ? data.get(key) : null; },
          setItem(key, value) { data.set(key, String(value)); },
          removeItem(key) { data.delete(key); },
        };
        const launcher = { open: true };
        const currentRoot = root([
          form(
            "strategy_pool_validation",
            [select(
              "pool_validation_strategy_type",
              ["", "approval", "limit"],
              "limit",
            )],
            launcher,
          ),
        ]);

        assert.equal(
          persistStrategyCandidateLabViewState(
            "task/a",
            currentRoot,
            storage,
          ),
          true,
        );
        assert.notEqual(
          strategyCandidateLabStateStorageKey("task/a"),
          strategyCandidateLabStateStorageKey("task/b"),
        );
        const loaded = loadStrategyCandidateLabViewState("task/a", storage);
        assert.equal(loaded.fields[0].values[0], "limit");
        assert.equal(loadStrategyCandidateLabViewState("task/b", storage), null);

        data.set(
          strategyCandidateLabStateStorageKey("task/a"),
          '{"schema_version":"wrong","fields":[],"open_workflows":[]}',
        );
        assert.equal(loadStrategyCandidateLabViewState("task/a", storage), null);
        data.set(strategyCandidateLabStateStorageKey("task/a"), "{broken");
        assert.equal(loadStrategyCandidateLabViewState("task/a", storage), null);
        """
    )


def test_controller_restores_once_and_persists_before_task_switch() -> None:
    run_node(
        r"""
        const data = new Map();
        const storage = {
          getItem(key) { return data.has(key) ? data.get(key) : null; },
          setItem(key, value) { data.set(key, String(value)); },
          removeItem(key) { data.delete(key); },
        };
        data.set(
          strategyCandidateLabStateStorageKey("task-a"),
          JSON.stringify({
            schema_version: "strategy.candidate-lab-view-state.v1",
            open_workflows: ["strategy_pool_validation"],
            fields: [{
              workflow: "strategy_pool_validation",
              field: "pool_validation_strategy_type",
              kind: "select",
              values: ["limit"],
            }],
          }),
        );

        const launcher = { open: false };
        const strategyType = select(
          "pool_validation_strategy_type",
          [""],
          "",
        );
        strategyType.setAttribute = () => {};
        strategyType.closest = (selector) => (
          selector === "[data-candidate-lab-field]" ? strategyType : null
        );
        const validationForm = form(
          "strategy_pool_validation",
          [strategyType],
          launcher,
        );
        validationForm.querySelector = (selector) => {
          if (
            selector
            === '[data-candidate-lab-field="pool_validation_strategy_type"]'
          ) {
            return strategyType;
          }
          if (selector === "[data-candidate-lab-pool-validation-help]") {
            return { textContent: "" };
          }
          if (selector === "[data-candidate-lab-form-error]") {
            return { textContent: "" };
          }
          return null;
        };
        validationForm.reset = () => { strategyType.value = ""; };
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector(selector) {
            return selector
              === '[data-candidate-lab-workflow="strategy_pool_validation"]'
              ? validationForm
              : null;
          },
          querySelectorAll(selector) {
            if (selector === "[data-candidate-lab-form]") {
              return [validationForm];
            }
            if (selector === "[data-candidate-lab-retry]") return [];
            return [strategyType];
          },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const pool = {
          kind: "candidate_pool",
          strategy_type: "limit",
          pool_id: `strategy-pool-${"a".repeat(32)}`,
          revision: 1,
          entries: [{
            entry_id: `pool-entry-${"b".repeat(32)}`,
            rule_id: `candidate-rule-${"c".repeat(32)}`,
            position: 0,
            action: { type: "limit", value: 1000 },
            enabled: true,
          }],
          total: 1,
          truncated: false,
        };
        const approvalPool = {
          ...pool,
          strategy_type: "approval",
          pool_id: `strategy-pool-${"d".repeat(32)}`,
          entries: [{
            entry_id: `pool-entry-${"e".repeat(32)}`,
            rule_id: `candidate-rule-${"f".repeat(32)}`,
            position: 0,
            action: { type: "approval" },
            enabled: true,
          }],
        };
        const payload = (taskId, pools) => ({
          task_id: taskId,
          can_start: true,
          blocked_reason: null,
          workflow: {},
          strategies: {},
          candidates: {},
          pools: {
            latest: pools[0] || null,
            all: pools,
            total: pools.length,
            truncated: false,
          },
        });
        const payloads = new Map([
          ["task-a", payload("task-a", [approvalPool, pool])],
          ["task-b", payload("task-b", [])],
        ]);
        let selectedTask = { id: "task-a", task_type: "strategy" };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          storage,
          getSelectedTask: () => selectedTask,
          getSelectedTaskId: () => selectedTask.id,
          getBlockedReason: () => "",
          getStrategyCandidateLab: async (taskId) => payloads.get(taskId),
        });

        await controller.selectTask(selectedTask);
        assert.equal(strategyType.value, "limit");
        assert.equal(launcher.open, true);

        strategyType.value = "approval";
        selectedTask = { id: "task-b", task_type: "strategy" };
        await controller.selectTask(selectedTask);
        const stored = JSON.parse(
          data.get(strategyCandidateLabStateStorageKey("task-a")),
        );
        assert.equal(stored.fields[0].values[0], "approval");
        """
    )
