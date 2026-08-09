from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

from tests.static_stylesheets import read_browser_stylesheets


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_data_workspace_panel_renders_and_saves_semantic_mapping():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";
        import { createDataWorkspacePanel } from "./marvis/static/js/v2/data_workspace_panel.js";

        function element(initial = {}) {
          const listeners = new Map();
          return {
            hidden: false,
            className: "",
            textContent: "",
            innerHTML: "",
            value: "",
            disabled: false,
            title: "",
            dataset: {},
            ...initial,
            addEventListener(type, listener) { listeners.set(type, listener); },
            dispatch(type, target = this) {
              return listeners.get(type)?.({ target, preventDefault() {}, stopPropagation() {} });
            },
          };
        }

        const elements = Object.fromEntries([
          "dataWorkspacePanel",
          "dataWorkspaceDataset",
          "dataWorkspaceDatasetHash",
          "dataWorkspaceRevision",
          "dataWorkspaceState",
          "dataWorkspaceTarget",
          "dataWorkspaceFieldCount",
          "dataWorkspaceFields",
          "dataWorkspaceStatus",
          "dataWorkspaceRefreshButton",
          "dataWorkspaceDiscardButton",
          "dataWorkspaceSaveButton",
        ].map((id) => [id, element()]));

        const initial = {
          schema_version: "data-workspace.v1",
          task_id: "task-1",
          revision: 3,
          active_dataset_id: "dataset-1",
          active_dataset_content_hash: "a".repeat(64),
          analysis_generation: 2,
          page: "semantics",
          selected_field: "loan_amount",
          semantic_mapping: {
            target_col: "bad_flag",
            field_roles: { bad_flag: "target", loan_amount: "amount" },
            business_names: { loan_amount: "贷款金额" },
          },
          updated_at: "2026-08-01T00:00:00Z",
        };
        const writes = [];
        const controller = createDataWorkspaceController({
          getDataWorkspace: async () => initial,
          putDataWorkspace: async (taskId, body, revision) => {
            writes.push({ taskId, body, revision });
            return { ...initial, ...body, revision: revision + 1 };
          },
        });
        const panel = createDataWorkspacePanel({
          getElementById: (id) => elements[id],
          controller,
          previewTaskDataset: async () => ({
            columns: ["customer_id", "bad_flag", "loan_amount"],
            rows: [],
          }),
        });

        await panel.selectTask("task-1");

        assert.equal(elements.dataWorkspacePanel.hidden, false);
        assert.equal(elements.dataWorkspaceDataset.textContent, "dataset-1");
        assert.equal(elements.dataWorkspaceDatasetHash.textContent, "aaaaaaaaaaaa…");
        assert.equal(elements.dataWorkspaceRevision.textContent, "R3");
        assert.equal(elements.dataWorkspaceState.textContent, "已同步");
        assert.equal(elements.dataWorkspaceTarget.value, "bad_flag");
        assert.match(elements.dataWorkspaceFields.innerHTML, /loan_amount/);
        assert.match(elements.dataWorkspaceFields.innerHTML, /贷款金额/);

        elements.dataWorkspaceTarget.value = "loan_amount";
        await elements.dataWorkspaceTarget.dispatch("change");
        assert.equal(controller.getDraft().semantic_mapping.target_col, "loan_amount");
        assert.equal(controller.getDraft().semantic_mapping.field_roles.loan_amount, "target");
        assert.equal(controller.getDraft().semantic_mapping.field_roles.bad_flag, undefined);
        assert.equal(elements.dataWorkspaceState.textContent, "未保存");

        await elements.dataWorkspaceFields.dispatch("change", {
          dataset: { workspaceRole: "customer_id" },
          value: "id",
        });
        await elements.dataWorkspaceFields.dispatch("change", {
          dataset: { workspaceBusinessName: "customer_id" },
          value: "客户编号",
        });

        await elements.dataWorkspaceSaveButton.dispatch("click");
        assert.equal(writes.length, 1);
        assert.equal(writes[0].taskId, "task-1");
        assert.equal(writes[0].revision, 3);
        assert.equal(writes[0].body.semantic_mapping.target_col, "loan_amount");
        assert.equal(writes[0].body.semantic_mapping.field_roles.customer_id, "id");
        assert.equal(writes[0].body.semantic_mapping.business_names.customer_id, "客户编号");
        assert.equal(controller.isDirty(), false);
        assert.equal(elements.dataWorkspaceRevision.textContent, "R4");
        assert.equal(elements.dataWorkspaceState.textContent, "已同步");
        """
    )


def test_data_workspace_panel_guards_navigation_and_surfaces_revision_conflict():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";
        import { createDataWorkspacePanel } from "./marvis/static/js/v2/data_workspace_panel.js";

        function element() {
          const listeners = new Map();
          return {
            hidden: false,
            className: "",
            textContent: "",
            innerHTML: "",
            value: "",
            disabled: false,
            title: "",
            dataset: {},
            addEventListener(type, listener) { listeners.set(type, listener); },
            dispatch(type, target = this) {
              return listeners.get(type)?.({ target, preventDefault() {}, stopPropagation() {} });
            },
          };
        }

        const elements = Object.fromEntries([
          "dataWorkspacePanel",
          "dataWorkspaceDataset",
          "dataWorkspaceDatasetHash",
          "dataWorkspaceRevision",
          "dataWorkspaceState",
          "dataWorkspaceTarget",
          "dataWorkspaceFieldCount",
          "dataWorkspaceFields",
          "dataWorkspaceStatus",
          "dataWorkspaceRefreshButton",
          "dataWorkspaceDiscardButton",
          "dataWorkspaceSaveButton",
        ].map((id) => [id, element()]));
        const snapshot = {
          schema_version: "data-workspace.v1",
          task_id: "task-1",
          revision: 8,
          active_dataset_id: "dataset-1",
          active_dataset_content_hash: "b".repeat(64),
          analysis_generation: 1,
          page: "overview",
          selected_field: null,
          semantic_mapping: { target_col: null, field_roles: {}, business_names: {} },
          updated_at: "2026-08-01T00:00:00Z",
        };
        let loads = 0;
        const conflict = Object.assign(new Error("stale data workspace revision"), { status: 412 });
        const controller = createDataWorkspaceController({
          getDataWorkspace: async () => { loads += 1; return snapshot; },
          putDataWorkspace: async () => { throw conflict; },
        });
        let choice = "cancel";
        const errors = [];
        const panel = createDataWorkspacePanel({
          getElementById: (id) => elements[id],
          controller,
          previewTaskDataset: async () => ({ columns: ["bad_flag"], rows: [] }),
          resolveNavigationChoice: () => choice,
          onError: (error) => errors.push(error),
        });

        await panel.selectTask("task-1");
        elements.dataWorkspaceTarget.value = "bad_flag";
        await elements.dataWorkspaceTarget.dispatch("change");

        let navigations = 0;
        assert.equal(await panel.requestNavigation(() => { navigations += 1; }), false);
        assert.equal(navigations, 0);
        assert.equal(controller.isDirty(), true);
        assert.equal(await panel.reload("task-1"), false);
        assert.equal(loads, 1);
        assert.match(elements.dataWorkspaceStatus.textContent, /保存或丢弃/);

        choice = "discard";
        assert.equal(await panel.requestNavigation(() => { navigations += 1; }), true);
        assert.equal(navigations, 1);
        assert.equal(controller.isDirty(), false);
        assert.equal(await panel.reload("task-1"), true);
        assert.equal(loads, 2);

        elements.dataWorkspaceTarget.value = "bad_flag";
        await elements.dataWorkspaceTarget.dispatch("change");
        assert.equal(await panel.save(), false);
        assert.equal(errors.at(-1), conflict);
        assert.equal(controller.isDirty(), true);
        assert.equal(elements.dataWorkspaceState.textContent, "版本冲突");
        assert.match(elements.dataWorkspaceStatus.textContent, /stale data workspace revision/);
        """
    )


def test_data_workspace_panel_is_mounted_in_the_task_workspace():
    app_js = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "marvis/static/index.html").read_text(encoding="utf-8")
    css = read_browser_stylesheets(ROOT / "marvis/static")

    assert 'import { createDataWorkspaceController } from "./js/v2/data_workspace_controller.js";' in app_js
    assert 'import { createDataWorkspacePanel } from "./js/v2/data_workspace_panel.js";' in app_js
    assert "const dataWorkspaceController = createDataWorkspaceController();" in app_js
    assert "const dataWorkspacePanel = createDataWorkspacePanel({" in app_js
    assert "async function requestTaskSelection(task)" in app_js
    assert "await new Promise((resolve) => requestAnimationFrame(() => resolve()));" in app_js
    assert app_js.count("row.onclick = () => requestTaskSelection(task);") == 2
    assert "row.onclick = () => selectTask(task);" not in app_js
    assert "async function reloadDataWorkspace(" in app_js
    assert app_js.count("await reloadDataWorkspace(taskId, { silent: true });") >= 2
    assert "await reloadDataWorkspace(selectedTaskId, { silent: true });" in app_js

    for element_id in (
        "dataWorkspacePanel",
        "dataWorkspaceDataset",
        "dataWorkspaceDatasetHash",
        "dataWorkspaceRevision",
        "dataWorkspaceState",
        "dataWorkspaceTarget",
        "dataWorkspaceFieldCount",
        "dataWorkspaceFields",
        "dataWorkspaceStatus",
        "dataWorkspaceRefreshButton",
        "dataWorkspaceDiscardButton",
        "dataWorkspaceSaveButton",
    ):
        assert f'id="{element_id}"' in index_html

    assert ".data-workspace-panel" in css
    assert ".data-workspace-field-row" in css
    assert ".data-workspace-state-conflict" in css
    assert ".data-workspace-state-dirty" in css
