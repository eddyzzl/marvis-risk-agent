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


def test_data_workspace_api_urls_headers_and_task_owned_preview():
    run_node(
        """
        import assert from "node:assert/strict";

        globalThis.document = {
          body: { dataset: { marvisLocalToken: "local-token" } },
        };
        const { apiPut } = await import("./marvis/static/js/api.js");
        const {
          getDataWorkspace,
          previewDataset,
          previewTaskDataset,
          putDataWorkspace,
        } = await import("./marvis/static/js/v2/api_v2.js");

        const calls = [];
        globalThis.fetch = async (url, options = {}) => {
          calls.push({ url, options });
          return {
            ok: true,
            status: 200,
            headers: { get: () => "application/json" },
            json: async () => ({ ok: true }),
            text: async () => "",
          };
        };

        await apiPut("api/example", { enabled: true }, {
          headers: { "If-Match": "3", "X-Caller": "kept" },
        });
        assert.equal(calls.at(-1).url, "/api/example");
        assert.equal(calls.at(-1).options.method, "PUT");
        assert.equal(calls.at(-1).options.headers["If-Match"], "3");
        assert.equal(calls.at(-1).options.headers["X-Caller"], "kept");
        assert.equal(calls.at(-1).options.headers["X-Marvis-Token"], "local-token");
        assert.equal(calls.at(-1).options.headers["Content-Type"], "application/json");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), { enabled: true });

        await getDataWorkspace("task / 1");
        assert.equal(calls.at(-1).url, "/api/tasks/task%20%2F%201/data-workspace");
        assert.equal(calls.at(-1).options.method, "GET");

        const body = {
          active_dataset_id: "dataset/1",
          active_dataset_content_hash: "c".repeat(64),
          page: "overview",
          selected_field: null,
          semantic_mapping: {
            target_col: null,
            field_roles: {},
            business_names: {},
          },
        };
        await putDataWorkspace("task / 1", body, 7);
        assert.equal(calls.at(-1).url, "/api/tasks/task%20%2F%201/data-workspace");
        assert.equal(calls.at(-1).options.method, "PUT");
        assert.equal(calls.at(-1).options.headers["If-Match"], "7");
        assert.equal(calls.at(-1).options.headers["X-Marvis-Token"], "local-token");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), body);

        await previewTaskDataset("task / 1", "dataset/1", 25);
        assert.equal(
          calls.at(-1).url,
          "/api/tasks/task%20%2F%201/datasets/dataset%2F1/preview?rows=25",
        );

        // The task-owned route is additive; existing callers keep the legacy route.
        await previewDataset("dataset/1", 25);
        assert.equal(calls.at(-1).url, "/api/datasets/dataset%2F1/preview?rows=25");
        """
    )


def test_workspace_session_save_clears_dirty_and_uses_saved_revision():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";

        const saved = [];
        const initial = {
          schema_version: "data-workspace.v1",
          task_id: "task-1",
          revision: 3,
          active_dataset_id: "dataset-1",
          active_dataset_content_hash: "a".repeat(64),
          analysis_generation: 4,
          page: "overview",
          selected_field: null,
          semantic_mapping: {
            target_col: "bad",
            field_roles: { income: "feature", bad: "target" },
            business_names: { income: "Monthly income" },
          },
          updated_at: "2026-07-19T00:00:00Z",
        };
        const controller = createDataWorkspaceController({
          getDataWorkspace: async () => initial,
          putDataWorkspace: async (taskId, body, revision) => {
            saved.push({ taskId, body, revision });
            return {
              ...initial,
              ...body,
              revision: revision + 1,
              updated_at: "2026-07-19T00:01:00Z",
            };
          },
        });

        await controller.load("task-1");
        assert.equal(controller.isDirty(), false);

        // Object key insertion order is not a semantic edit.
        controller.edit({
          semantic_mapping: {
            field_roles: { bad: "target", income: "feature" },
          },
        });
        assert.equal(controller.isDirty(), false);

        controller.edit({ page: "fields", selected_field: "income" });
        assert.equal(controller.isDirty(), true);
        await controller.save();

        assert.equal(saved.length, 1);
        assert.equal(saved[0].taskId, "task-1");
        assert.equal(saved[0].revision, 3);
        assert.deepEqual(Object.keys(saved[0].body).sort(), [
          "active_dataset_content_hash",
          "active_dataset_id",
          "page",
          "selected_field",
          "semantic_mapping",
        ]);
        assert.equal(saved[0].body.page, "fields");
        assert.equal(saved[0].body.selected_field, "income");
        assert.equal(controller.getServerSnapshot().revision, 4);
        assert.equal(controller.isDirty(), false);
        """
    )


def test_workspace_session_discard_and_navigation_choices():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";

        const snapshot = {
          schema_version: "data-workspace.v1",
          task_id: "task-1",
          revision: 5,
          active_dataset_id: "dataset-1",
          active_dataset_content_hash: "a".repeat(64),
          analysis_generation: 2,
          page: "overview",
          selected_field: null,
          semantic_mapping: {
            target_col: null,
            field_roles: {},
            business_names: {},
          },
          updated_at: "2026-07-19T00:00:00Z",
        };
        let saveCount = 0;
        const controller = createDataWorkspaceController({
          getDataWorkspace: async () => snapshot,
          putDataWorkspace: async (_taskId, body, revision) => {
            saveCount += 1;
            return { ...snapshot, ...body, revision: revision + 1 };
          },
        });
        await controller.load("task-1");

        controller.edit({ page: "fields", selected_field: "income" });
        controller.discard();
        assert.equal(controller.getDraft().page, "overview");
        assert.equal(controller.getDraft().selected_field, null);
        assert.equal(controller.isDirty(), false);

        let navigations = 0;
        controller.edit({ page: "fields" });
        assert.equal(
          await controller.guardNavigation("cancel", () => { navigations += 1; }),
          false,
        );
        assert.equal(navigations, 0);
        assert.equal(controller.isDirty(), true);

        assert.equal(
          await controller.guardNavigation("discard", () => { navigations += 1; }),
          true,
        );
        assert.equal(navigations, 1);
        assert.equal(controller.isDirty(), false);

        controller.edit({ page: "statistics" });
        assert.equal(
          await controller.guardNavigation(
            async (state) => state.dirty ? "save" : "cancel",
            () => { navigations += 1; },
          ),
          true,
        );
        assert.equal(saveCount, 1);
        assert.equal(navigations, 2);
        assert.equal(controller.isDirty(), false);
        """
    )


def test_workspace_session_conflict_preserves_draft_and_dirty_state():
    run_node(
        """
        import assert from "node:assert/strict";
        import { ApiError } from "./marvis/static/js/api.js";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";

        const snapshot = {
          schema_version: "data-workspace.v1",
          task_id: "task-1",
          revision: 8,
          active_dataset_id: "dataset-1",
          active_dataset_content_hash: "a".repeat(64),
          analysis_generation: 3,
          page: "overview",
          selected_field: null,
          semantic_mapping: {
            target_col: null,
            field_roles: {},
            business_names: {},
          },
          updated_at: "2026-07-19T00:00:00Z",
        };
        const conflict = new ApiError("stale data workspace revision", { status: 412 });
        const controller = createDataWorkspaceController({
          getDataWorkspace: async () => snapshot,
          putDataWorkspace: () => { throw conflict; },
        });

        await controller.load("task-1");
        controller.edit({
          page: "fields",
          selected_field: "income",
          semantic_mapping: {
            target_col: "bad",
            field_roles: { income: "feature", bad: "target" },
          },
        });
        const beforeSave = controller.getDraft();

        await assert.rejects(() => controller.save(), conflict);
        assert.deepEqual(controller.getDraft(), beforeSave);
        assert.equal(controller.getServerSnapshot().revision, 8);
        assert.equal(controller.isDirty(), true);
        assert.equal(controller.getState().error, conflict);
        """
    )


def test_activate_dataset_and_task_switch_clear_semantic_choices():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";

        const snapshots = {
          "task-1": {
            schema_version: "data-workspace.v1",
            task_id: "task-1",
            revision: 1,
            active_dataset_id: "dataset-1",
            active_dataset_content_hash: "a".repeat(64),
            analysis_generation: 1,
            page: "fields",
            selected_field: "income",
            semantic_mapping: {
              target_col: "bad",
              field_roles: { income: "feature", bad: "target" },
              business_names: { income: "Monthly income" },
            },
            updated_at: "2026-07-19T00:00:00Z",
          },
          "task-2": {
            schema_version: "data-workspace.v1",
            task_id: "task-2",
            revision: 0,
            active_dataset_id: null,
            active_dataset_content_hash: null,
            analysis_generation: 0,
            page: "overview",
            selected_field: null,
            semantic_mapping: {
              target_col: null,
              field_roles: {},
              business_names: {},
            },
            updated_at: "2026-07-19T00:00:01Z",
          },
        };
        const controller = createDataWorkspaceController({
          getDataWorkspace: async (taskId) => snapshots[taskId],
          putDataWorkspace: async () => { throw new Error("not used"); },
        });

        await controller.load("task-1");
        assert.throws(
          () => controller.activateDataset({ id: "dataset-2" }),
          /both be null or non-null/,
        );
        assert.throws(
          () => controller.activateDataset({ id: "dataset-2", content_hash: "not-a-hash" }),
          /SHA-256/,
        );
        controller.activateDataset({ id: "dataset-2", content_hash: "b".repeat(64) });
        assert.deepEqual(controller.getDraft(), {
          active_dataset_id: "dataset-2",
          active_dataset_content_hash: "b".repeat(64),
          page: "overview",
          selected_field: null,
          semantic_mapping: {
            target_col: null,
            field_roles: {},
            business_names: {},
          },
        });
        assert.equal(controller.isDirty(), true);

        await assert.rejects(
          () => controller.load("task-1"),
          /saved or discarded before loading a workspace/,
        );
        await assert.rejects(
          () => controller.load("task-2"),
          /saved or discarded before loading a workspace/,
        );
        assert.equal(controller.getState().taskId, "task-1");
        assert.equal(
          await controller.guardNavigation("discard", () => controller.load("task-2")),
          true,
        );
        assert.equal(controller.getState().taskId, "task-2");
        assert.deepEqual(controller.getDraft().semantic_mapping, {
          target_col: null,
          field_roles: {},
          business_names: {},
        });
        assert.equal(controller.getDraft().selected_field, null);
        assert.equal(controller.isDirty(), false);
        """
    )


def test_workspace_session_rejects_missing_task_or_invalid_revision_snapshot():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";

        const base = {
          schema_version: "data-workspace.v1",
          task_id: "task-1",
          revision: 0,
          active_dataset_id: null,
          active_dataset_content_hash: null,
          analysis_generation: 0,
          page: "overview",
          selected_field: null,
          semantic_mapping: {
            target_col: null,
            field_roles: {},
            business_names: {},
          },
          updated_at: "2026-07-19T00:00:00Z",
        };

        const missingTask = createDataWorkspaceController({
          getDataWorkspace: async () => {
            const snapshot = { ...base };
            delete snapshot.task_id;
            return snapshot;
          },
        });
        await assert.rejects(() => missingTask.load("task-1"), /different task/);

        const invalidRevision = createDataWorkspaceController({
          getDataWorkspace: async () => ({ ...base, revision: -1 }),
        });
        await assert.rejects(
          () => invalidRevision.load("task-1"),
          /non-negative integer/,
        );
        """
    )


def test_workspace_navigation_and_discard_are_blocked_while_save_is_in_flight():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";

        const snapshot = {
          schema_version: "data-workspace.v1",
          task_id: "task-1",
          revision: 2,
          active_dataset_id: null,
          active_dataset_content_hash: null,
          analysis_generation: 0,
          page: "overview",
          selected_field: null,
          semantic_mapping: {
            target_col: null,
            field_roles: {},
            business_names: {},
          },
          updated_at: "2026-07-19T00:00:00Z",
        };
        let releaseSave;
        const pendingSave = new Promise((resolve) => { releaseSave = resolve; });
        const controller = createDataWorkspaceController({
          getDataWorkspace: async () => snapshot,
          putDataWorkspace: async (_taskId, body, revision) => {
            await pendingSave;
            return { ...snapshot, ...body, revision: revision + 1 };
          },
        });
        await controller.load("task-1");
        controller.edit({ page: "fields" });
        const save = controller.save();
        controller.edit({ page: "overview" });
        assert.equal(controller.isDirty(), false);
        await assert.rejects(
          () => controller.load("task-2"),
          /save is in progress/,
        );
        assert.equal(controller.getState().taskId, "task-1");
        controller.edit({ page: "statistics" });

        let navigations = 0;
        assert.equal(
          await controller.guardNavigation("discard", () => { navigations += 1; }),
          false,
        );
        assert.equal(navigations, 0);
        assert.equal(controller.getDraft().page, "statistics");
        assert.throws(() => controller.discard(), /save is in progress/);

        releaseSave();
        await save;
        assert.equal(controller.getServerSnapshot().page, "fields");
        assert.equal(controller.getDraft().page, "statistics");
        assert.equal(controller.isDirty(), true);

        assert.equal(
          await controller.guardNavigation("discard", () => { navigations += 1; }),
          true,
        );
        assert.equal(navigations, 1);
        assert.equal(controller.getDraft().page, "fields");
        assert.equal(controller.isDirty(), false);
        """
    )
