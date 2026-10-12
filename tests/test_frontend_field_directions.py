from tests.test_frontend_data_workspace import run_node


def test_actual_controller_confirmation_payload_dirty_task_change_and_no_alias_inference():
    run_node('''
        import assert from "node:assert/strict";
        import { createDataWorkspaceController } from "./marvis/static/js/v2/data_workspace_controller.js";
        import { createFieldDirectionPanel, fieldDirectionHtml } from "./marvis/static/js/v2/field_direction_panel.js";
        const snapshot = { task_id: "task-1", revision: 3, active_dataset_id: "ds-1",
          active_dataset_content_hash: "a".repeat(64), page: "semantics", selected_field: null,
          semantic_mapping: { target_col: null, field_roles: {}, business_names: { x: "低值低风险" } } };
        const controller = createDataWorkspaceController({getDataWorkspace: async () => snapshot});
        // Real controller reload emits an empty draft before its GET. The
        // actual renderer must keep this loading state available.
        for (const draft of [null, undefined, {semantic_mapping:{}}]) {
          assert.ok(fieldDirectionHtml("x", draft).includes('value="unknown" selected'));
        }
        await controller.load("task-1");
        const html = fieldDirectionHtml("x", snapshot);
        assert.ok(html.includes('value="unknown" selected'));
        assert.ok(!html.includes('value="lower_is_riskier" selected'));
        let calls = []; let errors = [];
        const projection = { revision: 3, semantic_mapping_hash: "b".repeat(64), dataset_id: "ds-1",
          dataset_content_hash: "a".repeat(64), fields: [{column: "x"}] };
        const panel = createFieldDirectionPanel({element: {addEventListener(){}}, controller,
          getDirections: async () => projection,
          confirmDirections: async (taskId, body) => { calls.push({taskId, body}); },
          onError: (error) => errors.push(error) });
        assert.equal(await panel.confirm("x", "lower_is_riskier"), true);
        assert.deepEqual(calls[0], { taskId: "task-1", body: { expected_revision: 3,
          expected_semantic_mapping_hash: "b".repeat(64), dataset_id: "ds-1",
          dataset_content_hash: "a".repeat(64), directions: { x: "lower_is_riskier" } } });
        controller.edit({page: "fields"});
        assert.equal(await panel.confirm("x", "higher_is_riskier"), false);
        assert.equal(calls.length, 1);
        controller.discard();
        const stale = createFieldDirectionPanel({element: {addEventListener(){}}, controller,
          getDirections: async () => { await controller.load("task-2"); return projection; },
          confirmDirections: async () => { throw Error("must not submit stale task"); }, onError: error => errors.push(error) });
        assert.equal(await stale.confirm("x", "higher_is_riskier"), false);
        assert.equal(calls.length, 1); assert.equal(errors.length, 1);
        // Registered provenance survives ordinary controller edits and serialization.
        snapshot.semantic_mapping.risk_directions = {x: {direction: "non_monotonic", source: "user_confirmed",
          dataset_id: "ds-1", dataset_content_hash: "a".repeat(64), actor_id: "server-maker"}};
        await controller.load("task-1");
        controller.edit({semantic_mapping: {business_names: {x: "opposite meaning"}}});
        assert.equal(controller.getDraft().semantic_mapping.risk_directions.x.direction, "non_monotonic");
        assert.ok(fieldDirectionHtml("x", controller.getDraft(), {disabled:true}).includes('value="non_monotonic" selected'));
        assert.ok(fieldDirectionHtml("x", {...snapshot, active_dataset_id:"ds-2"}).includes("新数据待确认"));
    ''')
