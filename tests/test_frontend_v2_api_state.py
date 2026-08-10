from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_v2_static_modules_are_packaged_and_present():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["marvis"]

    assert "static/js/v2/*" in package_data

    static_v2 = Path("marvis/static/js/v2")
    for module_name in (
        "api_v2.js",
        "data_workspace_controller.js",
        "state_v2.js",
        "governance_extensions.js",
        "plugin_manager.js",
        "schema_table.js",
        "skill_manager.js",
        "artifact_view.js",
        "model_delivery_panel.js",
        "modeling_setup_panel.js",
        "capability.js",
    ):
        assert (static_v2 / module_name).is_file()


def test_api_wrappers_keep_formdata_boundary_under_fetch_control():
    run_node(
        """
        import assert from "node:assert/strict";
        import { apiDelete, apiGet, apiPost, apiPut } from "./marvis/static/js/api.js";

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
        globalThis.document = {
          body: { dataset: { marvisLocalToken: "shared-host-token" } },
          baseURI: "https://marvis.local/user/alice/proxy/8000/",
        };
        globalThis.location = new URL("https://marvis.local/user/alice/proxy/8000/");

        await apiGet("api/tasks");
        assert.equal(calls.at(-1).url, "https://marvis.local/user/alice/proxy/8000/api/tasks");
        assert.equal(calls.at(-1).options.method, "GET");
        assert.equal(calls.at(-1).options.headers["X-Marvis-Token"], "shared-host-token");

        await apiGet("https://outside.example.test/data");
        assert.equal(calls.at(-1).options.headers["X-Marvis-Token"], undefined);

        await apiGet("https://marvis.local/api/tasks", {
          headers: { "x-marvis-token": "caller-provided" },
        });
        assert.equal(calls.at(-1).options.headers["x-marvis-token"], "caller-provided");
        assert.equal(calls.at(-1).options.headers["X-Marvis-Token"], undefined);

        await apiPost("/api/plans/p1/confirm", { approved: true });
        assert.equal(calls.at(-1).options.method, "POST");
        assert.equal(calls.at(-1).options.headers["Content-Type"], "application/json");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), { approved: true });

        await apiPut("/api/tasks/t1/data-workspace", { page: "overview" }, {
          headers: { "If-Match": "0", "X-Caller": "kept" },
        });
        assert.equal(calls.at(-1).options.method, "PUT");
        assert.equal(calls.at(-1).options.headers["If-Match"], "0");
        assert.equal(calls.at(-1).options.headers["X-Caller"], "kept");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), { page: "overview" });

        const formData = new FormData();
        formData.append("file", new Blob(["zip"]), "plugin.zip");
        await apiPost("/api/plugins", formData);
        assert.equal(calls.at(-1).options.method, "POST");
        assert.ok(calls.at(-1).options.body instanceof FormData);
        assert.equal(
          Object.prototype.hasOwnProperty.call(calls.at(-1).options.headers ?? {}, "Content-Type"),
          false,
        );

        await apiDelete("/api/plugins/demo");
        assert.equal(calls.at(-1).url, "https://marvis.local/user/alice/proxy/8000/api/plugins/demo");
        assert.equal(calls.at(-1).options.method, "DELETE");

        const callCount = calls.length;
        await assert.rejects(apiGet("//outside.example.test/api/tasks"), /invalid API endpoint/);
        assert.equal(calls.length, callCount);
        """
    )


def test_driver_gate_conflicts_are_classified_without_treating_every_409_as_stale():
    run_node(
        """
        import assert from "node:assert/strict";
        import { ApiError, apiConflictKind } from "./marvis/static/js/api.js";

        const active = new ApiError("该任务正在执行上一步，请等待完成", {
          status: 409,
          detail: "该任务正在执行上一步，请等待完成",
          payload: { detail: "该任务正在执行上一步，请等待完成" },
        });
        const stale = new ApiError("该操作对应的计划已变化，请刷新页面后重试。", {
          status: 409,
          detail: "该操作对应的计划已变化，请刷新页面后重试。",
        });
        const unrelated = new ApiError("cannot generate report in status created", {
          status: 409,
          detail: "cannot generate report in status created",
        });

        assert.equal(apiConflictKind(active), "active_driver_job");
        assert.equal(apiConflictKind(stale), "confirmation_snapshot_stale");
        assert.equal(apiConflictKind(new ApiError("step gate-1 fingerprint changed while confirming", {
          status: 409,
        })), "confirmation_snapshot_stale");
        assert.equal(apiConflictKind(unrelated), "other_conflict");
        assert.equal(apiConflictKind(new Error("network failed")), "not_conflict");
        """
    )


def test_driver_gate_pending_claim_survives_rerender_until_authoritative_state_changes():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          claimDriverGateSubmission,
          confirmationSnapshotsEqual,
          driverGateActionable,
          markDriverGateSubmission,
          reconcileDriverGateSubmissions,
        } from "./marvis/static/js/v2/driver_gate_confirm.js";

        const snapshot1 = {
          expected_plan_status: "awaiting_confirm",
          expected_plan_revision: 1,
          expected_plan_fingerprint: "a".repeat(64),
          expected_step_fingerprint: "b".repeat(64),
        };
        const snapshot2 = {
          ...snapshot1,
          expected_plan_revision: 2,
          expected_plan_fingerprint: "c".repeat(64),
          expected_step_fingerprint: "d".repeat(64),
        };
        const gate = {
          taskId: "task-1",
          planId: "plan-1",
          stepId: "step-1",
          stepStatus: "awaiting_confirm",
          snapshot: snapshot1,
          localBusy: false,
          serverBusy: false,
        };

        assert.equal(driverGateActionable(gate), true);
        assert.equal(confirmationSnapshotsEqual(snapshot1, snapshot1, { requireStep: true }), true);
        assert.equal(confirmationSnapshotsEqual(snapshot1, snapshot2, { requireStep: true }), false);
        assert.equal(driverGateActionable({ ...gate, snapshot: {} }), false);
        claimDriverGateSubmission({ ...gate, state: "submitting" });
        assert.equal(driverGateActionable(gate), false);
        // A DOM rebuild with the same task+plan+step must remain disabled.
        assert.equal(driverGateActionable({ ...gate }), false);
        // A new authoritative CAS snapshot is a new action and releases the old claim.
        assert.equal(driverGateActionable({ ...gate, snapshot: snapshot2 }), true);

        claimDriverGateSubmission({ ...gate, state: "submitting" });
        markDriverGateSubmission({ ...gate, state: "active_driver_job" });
        assert.equal(driverGateActionable({ ...gate, serverBusy: true }), false);
        // The duplicate request was never accepted; once task polling proves the
        // existing server job is idle, the still-current gate may be used again.
        reconcileDriverGateSubmissions("task-1", { serverBusy: false });
        assert.equal(driverGateActionable({ ...gate, serverBusy: false }), true);
        assert.equal(driverGateActionable({ ...gate, stepStatus: "done" }), false);
        """
    )


def test_driver_gate_request_binding_only_claims_typed_agent_confirmation_posts():
    run_node(
        """
        import assert from "node:assert/strict";
        import { driverGateRequestBinding } from "./marvis/static/js/v2/driver_gate_confirm.js";

        const snapshot = {
          expected_plan_status: "awaiting_confirm",
          expected_plan_revision: 3,
          expected_plan_fingerprint: "a".repeat(64),
          expected_step_fingerprint: "b".repeat(64),
        };
        const binding = driverGateRequestBinding("/api/tasks/task-1/agent/messages", {
          method: "POST",
          body: JSON.stringify({
            content: "确认",
            ui_action: "confirm_features",
            expected_plan_id: "plan-1",
            expected_step_id: "step-1",
            ...snapshot,
          }),
        });
        assert.deepEqual(binding, {
          taskId: "task-1",
          planId: "plan-1",
          stepId: "step-1",
          snapshot,
        });
        assert.equal(driverGateRequestBinding("/api/tasks/task-1/agent/messages", {
          method: "POST",
          body: JSON.stringify({ content: "普通问题" }),
        }), null);
        assert.equal(driverGateRequestBinding("/api/tasks/task-1/agent/messages", {
          method: "GET",
        }), null);
        """
    )


def test_shared_driver_gate_api_claims_before_request_and_keeps_accepted_gate_disabled():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createDriverGateApi,
          driverGateActionable,
          reconcileDriverGateSubmissions,
        } from "./marvis/static/js/v2/driver_gate_confirm.js";

        const snapshot = {
          expected_plan_status: "awaiting_confirm",
          expected_plan_revision: 1,
          expected_plan_fingerprint: "a".repeat(64),
          expected_step_fingerprint: "b".repeat(64),
        };
        const gate = {
          taskId: "task-1",
          planId: "plan-1",
          stepId: "step-1",
          stepStatus: "awaiting_confirm",
          snapshot,
          localBusy: false,
          serverBusy: false,
        };
        let resolveRequest;
        const busy = [];
        const states = [];
        const request = new Promise((resolve) => { resolveRequest = resolve; });
        const gateApi = createDriverGateApi({
          api: async () => request,
          getLocalBusyAction: () => "",
          setDriverExecutionBusy: (active, taskId) => busy.push([active, taskId]),
          onSubmissionStateChange: (binding, state) => states.push([binding.stepId, state]),
        });
        const pending = gateApi("/api/tasks/task-1/agent/messages", {
          method: "POST",
          body: JSON.stringify({
            content: "确认",
            ui_action: "confirm_gate",
            expected_plan_id: "plan-1",
            expected_step_id: "step-1",
            ...snapshot,
          }),
        });
        assert.equal(driverGateActionable(gate), false);
        resolveRequest({ messages: [] });
        await pending;
        assert.equal(driverGateActionable(gate), false);
        assert.deepEqual(busy, [[true, "task-1"], [false, "task-1"]]);
        assert.deepEqual(states, [["step-1", "submitting"], ["step-1", "accepted"], ["step-1", "settled"]]);

        const activeApi = createDriverGateApi({
          api: async () => { throw Object.assign(new Error("该任务正在执行上一步，请等待完成"), { status: 409 }); },
        });
        await assert.rejects(() => activeApi("/api/tasks/task-2/agent/messages", {
          method: "POST",
          body: JSON.stringify({
            content: "确认",
            ui_action: "confirm_gate",
            expected_plan_id: "plan-2",
            expected_step_id: "step-2",
            ...snapshot,
          }),
        }));
        const activeGate = { ...gate, taskId: "task-2", planId: "plan-2", stepId: "step-2" };
        assert.equal(driverGateActionable(activeGate), false);
        reconcileDriverGateSubmissions("task-2", { serverBusy: false });
        assert.equal(driverGateActionable(activeGate), true);
        """
    )


def test_plugin_tools_render_contract_tables_and_runtime_metadata():
    run_node(
        """
        import assert from "node:assert/strict";
        import { pluginToolsHtml } from "./marvis/static/js/v2/plugin_manager.js";

        const html = pluginToolsHtml({
          module: "marvis.packs.feature.tools",
          hooks: [{ event: "step.completed", tool: "screen_features" }],
          tools: [{
            name: "screen_features",
            summary: "Leakage-aware feature screening.",
            entrypoint: "tool_screen_features",
            determinism: "deterministic",
            timeout_seconds: 300,
            failure_policy: "fail",
            memory_limit_mb: 4096,
            side_effects: ["read:dataset"],
            input_schema: {
              type: "object",
              properties: {
                dataset_id: { type: "string", minLength: 1 },
                top_k: { type: "integer", minimum: 1 },
                mode: { type: "string", enum: ["fast", "full"] },
              },
              required: ["dataset_id"],
            },
            output_schema: {
              type: "object",
              properties: {
                selected: { type: "array", items: { type: "string" } },
                n_screened: { type: "integer", minimum: 0 },
              },
              required: ["selected"],
            },
          }],
        });

        assert.ok(html.includes("plugin-tool-impl"));
        assert.ok(html.includes("marvis.packs.feature.tools.tool_screen_features"));
        assert.ok(html.includes("Hook: step.completed"));
        assert.ok(html.includes("<table class=\\"plugin-schema-table\\">"));
        assert.ok(html.includes("<caption>输入</caption>"));
        assert.ok(html.includes("<caption>输出</caption>"));
        assert.ok(html.includes("<code>dataset_id</code>"));
        assert.ok(html.includes("最短 1"));
        assert.ok(html.includes("可选：fast / full"));
        assert.ok(!html.includes("<pre><code>"));
      """
    )


def test_plugin_and_workflow_managers_show_upload_format_examples():
    run_node(
        """
        import assert from "node:assert/strict";
        import { pluginManagerHtml } from "./marvis/static/js/v2/plugin_manager.js";
        import { skillManagerHtml } from "./marvis/static/js/v2/skill_manager.js";

        const pluginHtml = pluginManagerHtml({ plugins: [] });
        assert.ok(pluginHtml.includes("插件包格式示例"));
        assert.ok(pluginHtml.includes("sample_pack.zip"));
        assert.ok(pluginHtml.includes("manifest.json"));
        assert.ok(pluginHtml.includes("tools.py"));
        assert.ok(pluginHtml.includes("&quot;entrypoint&quot;"));
        assert.ok(pluginHtml.includes("def tool_echo"));
        assert.ok(pluginHtml.includes("data-upload-plugin"));

        const workflowHtml = skillManagerHtml({ active: [], disabled: [], rejected: [] });
        assert.ok(workflowHtml.includes("模板 JSON 示例"));
        assert.ok(workflowHtml.includes("workspace/skills/*.json"));
        assert.ok(workflowHtml.includes("custom_echo_review.json"));
        assert.ok(workflowHtml.includes("&quot;slots&quot;"));
        assert.ok(workflowHtml.includes("&quot;steps&quot;"));
        assert.ok(workflowHtml.includes("&quot;tool&quot;"));
      """
    )


def test_draft_tool_detail_renders_schema_tables_instead_of_raw_json():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createDraftToolsPanelController } from "./marvis/static/js/draft-tools-panel.js";

        class Element {
          constructor() {
            this.dataset = {};
            this.disabled = false;
            this.innerHTML = "";
            this.textContent = "";
            this.value = "";
            this.classes = new Set();
            this.classList = {
              add: (...names) => names.forEach((name) => this.classes.add(name)),
              remove: (...names) => names.forEach((name) => this.classes.delete(name)),
            };
          }
        }

        const ids = Object.fromEntries([
          "draftToolBody",
          "draftToolEmpty",
          "draftToolsList",
          "draftToolName",
          "draftToolSummary",
          "draftToolStatus",
          "draftToolMeta",
          "draftToolCode",
          "draftInputSchema",
          "draftOutputSchema",
          "draftLearningNote",
          "draftRunHistory",
          "draftRunInputs",
          "draftPromotionTestCases",
          "runDraftButton",
          "promoteDraftButton",
          "rejectDraftButton",
        ].map((id) => [id, new Element()]));

        const controller = createDraftToolsPanelController({
          $: (id) => ids[id],
          api: async () => ({}),
          runAction: (fn) => fn(),
          showPlatformConfirm: async () => true,
        });

        controller.renderDetail({
          draft: {
            id: "draft-1",
            task_id: "task-1",
            name: "calc_margin",
            summary: "Calculate margin.",
            code: "def calc_margin(inputs, ctx):\\n    return {'margin': inputs['revenue'] - inputs['cost']}\\n",
            input_schema: {
              type: "object",
              properties: {
                revenue: { type: "number", minimum: 0 },
                cost: { type: "number" },
              },
              required: ["revenue", "cost"],
            },
            output_schema: {
              type: "object",
              properties: {
                margin: { type: "number", description: "Revenue minus cost." },
              },
              required: ["margin"],
            },
            determinism: "deterministic",
            source: "hand_written",
            status: "tested",
            created_at: "2026-06-19T00:00:00Z",
          },
          learning_note: null,
          runs: [],
        });

        assert.ok(ids.draftInputSchema.innerHTML.includes("<table class=\\"plugin-schema-table\\">"));
        assert.ok(ids.draftOutputSchema.innerHTML.includes("<caption>输出</caption>"));
        assert.ok(ids.draftInputSchema.innerHTML.includes("<code>revenue</code>"));
        assert.ok(ids.draftInputSchema.innerHTML.includes("最小 0"));
        assert.ok(ids.draftOutputSchema.innerHTML.includes("Revenue minus cost."));
        assert.equal(ids.draftInputSchema.innerHTML.includes('"properties"'), false);
        assert.equal(ids.draftOutputSchema.innerHTML.includes('"properties"'), false);
      """
    )


def test_v2_api_routes_and_multipart_helpers_match_backend_contracts():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          cancelPlan,
          authorDraftTool,
          confirmJoinSpec,
          confirmPlan,
          confirmStep,
          decideStep,
          createPlan,
          distillDraftLearning,
          executeJoin,
          fetchDraftUrl,
          getJoinPlan,
          getLatestTaskJob,
          getTask,
          getMemoryDistillation,
          getPlan,
          listCapabilityTiers,
          listDatasets,
          listMemoryDistillations,
          listPluginTools,
          listPlugins,
          listSkills,
          previewDataset,
          proposeJoin,
          reloadSkills,
          removePlugin,
          rollbackMemoryDistillation,
          retryStep,
          runPlan,
          searchDraftWeb,
          setPluginEnabled,
          consolidateMemory,
          uploadDataset,
          uploadPlugin,
          validateSkill,
        } from "./marvis/static/js/v2/api_v2.js";

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

        await createPlan("task id", { goal: "build plan" });
        assert.equal(calls.at(-1).url, "/api/tasks/task%20id/plans");
        assert.equal(calls.at(-1).options.method, "POST");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), { goal: "build plan" });
        await getTask("task id");
        assert.equal(calls.at(-1).url, "/api/tasks/task%20id");
        await getLatestTaskJob("task id", "join");
        assert.equal(calls.at(-1).url, "/api/tasks/task%20id/jobs/latest?kind=join");
        await getLatestTaskJob("task id");
        assert.equal(calls.at(-1).url, "/api/tasks/task%20id/jobs/latest");

        await getPlan("plan/1");
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1");
        const planSnapshot = {
          expected_plan_status: "validated",
          expected_plan_revision: 2,
          expected_plan_fingerprint: "a".repeat(64),
        };
        const stepSnapshot = {
          ...planSnapshot,
          expected_step_fingerprint: "b".repeat(64),
        };
        await confirmPlan("plan/1", planSnapshot);
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/confirm");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), planSnapshot);
        await runPlan("plan/1");
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/run");
        await confirmStep("plan/1", "step/a", stepSnapshot);
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/steps/step%2Fa/confirm");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), stepSnapshot);
        await decideStep("plan/1", "step/a", "reject", "needs revision", stepSnapshot);
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/steps/step%2Fa/decisions");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), {
          decision: "reject",
          reason: "needs revision",
          ...stepSnapshot,
        });
        await retryStep("plan/1", "step/a", { message: "new" });
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/steps/step%2Fa/retry");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), { inputs: { message: "new" } });
        await cancelPlan("plan/1");
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/cancel");

        await listPlugins(true);
        assert.equal(calls.at(-1).url, "/api/plugins?include_disabled=true");
        await uploadPlugin(new Blob(["zip"]));
        assert.equal(calls.at(-1).url, "/api/plugins");
        assert.ok(calls.at(-1).options.body instanceof FormData);
        assert.equal(
          Object.prototype.hasOwnProperty.call(calls.at(-1).options.headers ?? {}, "Content-Type"),
          false,
        );
        await setPluginEnabled("plugin/demo", false);
        assert.equal(calls.at(-1).url, "/api/plugins/plugin%2Fdemo/disable");
        await removePlugin("plugin/demo");
        assert.equal(calls.at(-1).url, "/api/plugins/plugin%2Fdemo");
        assert.equal(calls.at(-1).options.method, "DELETE");
        await listPluginTools("plugin/demo");
        assert.equal(calls.at(-1).url, "/api/plugins/plugin%2Fdemo/tools");

        await listSkills();
        assert.equal(calls.at(-1).url, "/api/skills");
        await reloadSkills();
        assert.equal(calls.at(-1).url, "/api/skills/reload");
        await validateSkill({ id: "workflow_template" });
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), {
          skill: { id: "workflow_template" },
        });

        await listDatasets("task id");
        assert.equal(calls.at(-1).url, "/api/tasks/task%20id/datasets");
        await uploadDataset("task id", new Blob(["csv"]), { role: "sample", sheet: "Sheet 1" });
        assert.equal(calls.at(-1).url, "/api/tasks/task%20id/datasets/upload");
        assert.ok(calls.at(-1).options.body instanceof FormData);
        assert.equal(calls.at(-1).options.body.get("role"), "sample");
        assert.equal(calls.at(-1).options.body.get("sheet"), "Sheet 1");
        assert.equal(
          Object.prototype.hasOwnProperty.call(calls.at(-1).options.headers ?? {}, "Content-Type"),
          false,
        );
        await previewDataset("dataset/1", 25);
        assert.equal(calls.at(-1).url, "/api/datasets/dataset%2F1/preview?rows=25");
        await proposeJoin("task id", { anchor_dataset_id: "sample" });
        assert.equal(calls.at(-1).url, "/api/tasks/task%20id/joins/propose");
        await getJoinPlan("join/1");
        assert.equal(calls.at(-1).url, "/api/joins/join%2F1");
        await confirmJoinSpec("join/1", { feature_dataset_id: "feature" });
        assert.equal(calls.at(-1).url, "/api/joins/join%2F1/confirm");
        await executeJoin("join/1");
        assert.equal(calls.at(-1).url, "/api/joins/join%2F1/execute");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), { async_execute: true });
        await executeJoin("join/1", { async_execute: false });
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), { async_execute: false });

        await listCapabilityTiers();
        assert.equal(calls.at(-1).url, "/api/capability-tiers");

        await listMemoryDistillations({ category: "field_convention", includeSuperseded: true });
        assert.equal(calls.at(-1).url, "/api/agent-memory/distillations?category=field_convention&include_superseded=true");
        await getMemoryDistillation("distill/1");
        assert.equal(calls.at(-1).url, "/api/agent-memory/distillations/distill%2F1");
        await rollbackMemoryDistillation("distill/1");
        assert.equal(calls.at(-1).url, "/api/agent-memory/distillations/distill%2F1/rollback");
        await consolidateMemory("model_experience");
        assert.equal(calls.at(-1).url, "/api/agent-memory/consolidate?category=model_experience");
        await searchDraftWeb("learn joins", 3);
        assert.equal(calls.at(-1).url, "/api/drafts/web-search");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), {
          query: "learn joins",
          max_results: 3,
        });
        await fetchDraftUrl("https://example.test/a", 1200);
        assert.equal(calls.at(-1).url, "/api/drafts/fetch-url");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), {
          url: "https://example.test/a",
          max_bytes: 1200,
        });
        await distillDraftLearning({
          query: "learn joins",
          contents: ["bounded page contents"],
          sources: ["https://example.test/a"],
          model_id: "m1",
        });
        assert.equal(calls.at(-1).url, "/api/drafts/learning-notes");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), {
          query: "learn joins",
          contents: ["bounded page contents"],
          sources: ["https://example.test/a"],
          model_id: "m1",
        });
        await authorDraftTool({
          task_id: "task-1",
          goal: "build helper",
          learning_note_id: "note-1",
          model_id: "m1",
        });
        assert.equal(calls.at(-1).url, "/api/drafts/author");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), {
          task_id: "task-1",
          goal: "build helper",
          learning_note_id: "note-1",
          model_id: "m1",
        });
        """
    )


def test_v2_api_preserves_structured_plan_validation_errors():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createPlan } from "./marvis/static/js/v2/api_v2.js";

        globalThis.fetch = async () => ({
          ok: false,
          status: 422,
          headers: { get: () => "application/json" },
          json: async () => ({ detail: { problems: ["missing tool <bad>"] } }),
          text: async () => "",
        });

        await assert.rejects(
          () => createPlan("task-1", { goal: "bad plan" }),
          (error) => {
            assert.equal(error.name, "ApiError");
            assert.equal(error.status, 422);
            assert.deepEqual(error.detail, { problems: ["missing tool <bad>"] });
            assert.ok(error.message.includes("missing tool"));
            return true;
          },
        );
        """
    )


def test_v2_state_store_is_keyed_subscribable_and_resettable():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          getCapabilityTiers,
          getCurrentJoin,
          getDatasets,
          getLoopEvents,
          getPlan,
          getPlugins,
          getSelectedStepId,
          getSelectedTier,
          getState,
          onPlanChange,
          resetV2State,
          setCapabilityTiers,
          setCurrentJoin,
          setDatasets,
          setLoopEvents,
          setPlan,
          setPlugins,
          setSelectedStepId,
          setSelectedTier,
          setState,
          subscribe,
        } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const planEvents = [];
        const unsubscribePlan = onPlanChange((next, previous) => {
          planEvents.push({ next, previous });
        });

        setPlan({ id: "p1", status: "draft" });
        assert.equal(getPlan().id, "p1");
        assert.equal(planEvents.length, 1);
        assert.equal(planEvents[0].next.id, "p1");
        assert.equal(planEvents[0].previous, null);

        unsubscribePlan();
        setPlan({ id: "p2" });
        assert.equal(planEvents.length, 1);

        const stepEvents = [];
        const unsubscribeStep = subscribe("v2.selectedStepId", (next) => stepEvents.push(next));
        setSelectedStepId("s1");
        assert.equal(getSelectedStepId(), "s1");
        assert.deepEqual(stepEvents, ["s1"]);
        unsubscribeStep();

        setPlugins([{ name: "demo" }]);
        setDatasets([{ id: "dataset-1" }]);
        setCurrentJoin({ id: "join-1" });
        setCapabilityTiers([{ name: "balanced" }]);
        setSelectedTier("balanced");
        setLoopEvents([{ type: "replan" }]);

        assert.equal(getPlugins()[0].name, "demo");
        assert.equal(getDatasets()[0].id, "dataset-1");
        assert.equal(getCurrentJoin().id, "join-1");
        assert.equal(getCapabilityTiers()[0].name, "balanced");
        assert.equal(getSelectedTier(), "balanced");
        assert.equal(getLoopEvents()[0].type, "replan");
        assert.equal(getState("v2.selectedTier"), "balanced");

        assert.throws(() => setState("v1.currentPlan", {}), /Unknown v2 state key/);
        resetV2State();
        assert.equal(getPlan(), null);
        assert.deepEqual(getPlugins(), []);
        """
    )


def test_governance_extension_mount_creates_stable_panels_idempotently():
    run_node(
        """
        import assert from "node:assert/strict";
        import { mountGovernanceExtensionPanels } from "./marvis/static/js/v2/governance_extensions.js";
        import { resetV2State } from "./marvis/static/js/v2/state_v2.js";

        function makeElement(tagName) {
          return {
            tagName: tagName.toUpperCase(),
            id: "",
            innerHTML: "",
            className: "",
            dataset: {},
            attributes: {},
            children: [],
            setAttribute(name, value) {
              this.attributes[name] = String(value);
            },
            appendChild(child) {
              this.children.push(child);
              return child;
            },
          };
        }

        resetV2State();
        const root = makeElement("div");
        root.ownerDocument = { createElement: makeElement };
        root.querySelector = (selector) => {
          const id = selector.startsWith("#") ? selector.slice(1) : selector;
          return root.children.find((child) => child.id === id) ?? null;
        };

        const first = mountGovernanceExtensionPanels(root);
        const second = mountGovernanceExtensionPanels(root);

        assert.deepEqual(Object.keys(first.panels), [
          "pluginPanel",
          "skillPanel",
          "capabilityPanel",
        ]);
        assert.equal(first.panels.pluginPanel, second.panels.pluginPanel);
        assert.equal(first.panels.skillPanel, second.panels.skillPanel);
        assert.equal(second.panels.capabilityPanel, first.panels.capabilityPanel);
        assert.equal(root.children.length, 3);
        assert.deepEqual(root.children.map((child) => child.id), [
          "pluginPanel",
          "skillPanel",
          "capabilityPanel",
        ]);
        assert.equal(root.dataset.governanceExtensionsMounted, "true");
        assert.equal(first.panels.pluginPanel.dataset.v2PluginManager, "true");
        assert.equal(first.panels.skillPanel.dataset.v2SkillManager, "true");
        assert.equal(first.panels.capabilityPanel.dataset.v2TierSettings, "true");
        assert.equal(first.panels.pluginPanel.dataset.panelTitle, "插件");
        assert.ok(first.panels.pluginPanel.innerHTML.includes('data-upload-plugin'));
        assert.ok(first.panels.skillPanel.innerHTML.includes('id="reloadSkills"'));
        assert.ok(first.panels.skillPanel.innerHTML.includes('data-validate-skill'));
        assert.ok(first.panels.capabilityPanel.innerHTML.includes('安全护栏保持一致'));
        """
    )


def test_governance_extension_mount_registers_delegated_handlers_once_and_cleans_up():
    run_node(
        """
        import assert from "node:assert/strict";
        import { mountGovernanceExtensionPanels } from "./marvis/static/js/v2/governance_extensions.js";
        import { resetV2State } from "./marvis/static/js/v2/state_v2.js";

        function makeElement(tagName) {
          return {
            tagName: tagName.toUpperCase(),
            id: "",
            innerHTML: "",
            className: "",
            dataset: {},
            attributes: {},
            children: [],
            setAttribute(name, value) {
              this.attributes[name] = String(value);
            },
            appendChild(child) {
              this.children.push(child);
              return child;
            },
          };
        }

        resetV2State();
        const listeners = {};
        const root = makeElement("div");
        root.ownerDocument = { createElement: makeElement };
        root.querySelector = (selector) => {
          const id = selector.startsWith("#") ? selector.slice(1) : selector;
          return root.children.find((child) => child.id === id) ?? null;
        };
        root.addEventListener = (type, handler) => {
          listeners[type] = [...(listeners[type] || []), handler];
        };
        root.removeEventListener = (type, handler) => {
          listeners[type] = (listeners[type] || []).filter((candidate) => candidate !== handler);
        };

        const mounted = mountGovernanceExtensionPanels(root);
        mountGovernanceExtensionPanels(root);

        assert.equal((listeners.click || []).length, 2);
        assert.equal((listeners.change || []).length, 2);
        assert.equal((listeners.input || []).length, 1);

        mounted.unmount();

        assert.equal((listeners.click || []).length, 0);
        assert.equal((listeners.change || []).length, 0);
        assert.equal((listeners.input || []).length, 0);
        """
    )


def test_governance_extension_mount_initially_loads_governance_panels():
    run_node(
        """
        import assert from "node:assert/strict";
        import { mountGovernanceExtensionPanels } from "./marvis/static/js/v2/governance_extensions.js";
        import { resetV2State } from "./marvis/static/js/v2/state_v2.js";

        function makeElement(tagName) {
          return {
            tagName: tagName.toUpperCase(),
            id: "",
            innerHTML: "",
            className: "",
            dataset: {},
            attributes: {},
            children: [],
            setAttribute(name, value) {
              this.attributes[name] = String(value);
            },
            appendChild(child) {
              this.children.push(child);
              return child;
            },
          };
        }

        resetV2State();
        const calls = [];
        const root = makeElement("div");
        root.ownerDocument = { createElement: makeElement };
        root.querySelector = (selector) => {
          const id = selector.startsWith("#") ? selector.slice(1) : selector;
          return root.children.find((child) => child.id === id) ?? null;
        };

        const mounted = mountGovernanceExtensionPanels(root, {
          pluginActions: {
            listPlugins: async (includeDisabled) => {
              calls.push(["listPlugins", includeDisabled]);
              return {
                plugins: [
                  {
                    name: "demo",
                    display_name: "Demo Plugin",
                    version: "1.0",
                    enabled: true,
                    builtin: false,
                    tool_count: 1,
                  },
                ],
              };
            },
          },
          skillActions: {
            listSkills: async () => {
              calls.push(["listSkills"]);
              return { active: ["demo_skill"], disabled: [], rejected: [] };
            },
          },
          capabilityActions: {
            listCapabilityTiers: async () => {
              calls.push(["listCapabilityTiers"]);
              return { default: "reviewed", tiers: [{ name: "reviewed", summary: "Reviewed" }] };
            },
          },
        });
        await Promise.resolve();
        await Promise.resolve();

        assert.deepEqual(calls, [
          ["listPlugins", true],
          ["listSkills"],
          ["listCapabilityTiers"],
        ]);
        assert.ok(mounted.panels.pluginPanel.innerHTML.includes("Demo Plugin"));
        assert.ok(mounted.panels.skillPanel.innerHTML.includes("demo_skill"));
        assert.ok(mounted.panels.capabilityPanel.innerHTML.includes("Reviewed"));
        """
    )


def test_governance_extension_mount_wires_plugin_and_skill_refresh_actions():
    run_node(
        """
        import assert from "node:assert/strict";
        import { mountGovernanceExtensionPanels } from "./marvis/static/js/v2/governance_extensions.js";
        import { resetV2State } from "./marvis/static/js/v2/state_v2.js";

        function makeElement(tagName) {
          return {
            tagName: tagName.toUpperCase(),
            id: "",
            innerHTML: "",
            className: "",
            dataset: {},
            attributes: {},
            children: [],
            setAttribute(name, value) {
              this.attributes[name] = String(value);
            },
            appendChild(child) {
              this.children.push(child);
              return child;
            },
          };
        }

        resetV2State();
        const calls = [];
        const listeners = {};
        const root = makeElement("div");
        root.ownerDocument = { createElement: makeElement };
        root.querySelector = (selector) => {
          const id = selector.startsWith("#") ? selector.slice(1) : selector;
          return root.children.find((child) => child.id === id) ?? null;
        };
        root.addEventListener = (type, handler) => {
          listeners[type] = [...(listeners[type] || []), handler];
        };
        root.removeEventListener = (type, handler) => {
          listeners[type] = (listeners[type] || []).filter((candidate) => candidate !== handler);
        };

        const mounted = mountGovernanceExtensionPanels(root, {
          pluginActions: {
            uploadPlugin: async (file) => calls.push(["uploadPlugin", file.name]),
            listPlugins: async (includeDisabled) => {
              calls.push(["listPlugins", includeDisabled]);
              return {
                plugins: [
                  {
                    name: "demo",
                    display_name: "Demo Plugin",
                    version: "1.0",
                    enabled: true,
                    builtin: false,
                    tool_count: 1,
                  },
                ],
              };
            },
          },
          skillActions: {
            reloadSkills: async () => calls.push(["reloadSkills"]),
            listSkills: async () => {
              calls.push(["listSkills"]);
              return { active: ["demo_skill"], disabled: [], rejected: [] };
            },
          },
        });
        await Promise.resolve();
        await Promise.resolve();
        assert.deepEqual(calls.splice(0), [
          ["listPlugins", true],
          ["listSkills"],
        ]);

        const uploadTarget = {
          files: [{ name: "demo.zip" }],
          closest(selector) {
            return selector === "[data-upload-plugin]" ? this : null;
          },
        };
        for (const handler of listeners.change || []) {
          await handler({ target: uploadTarget });
        }
        assert.deepEqual(calls.splice(0), [
          ["uploadPlugin", "demo.zip"],
          ["listPlugins", true],
        ]);
        assert.ok(mounted.panels.pluginPanel.innerHTML.includes("Demo Plugin"));

        const reloadTarget = {
          closest(selector) {
            return selector === "#reloadSkills" || selector === "[data-reload-skills]" ? this : null;
          },
        };
        for (const handler of listeners.click || []) {
          await handler({ target: reloadTarget, preventDefault() {} });
        }
        assert.deepEqual(calls.splice(0), [
          ["reloadSkills"],
          ["listSkills"],
        ]);
        assert.ok(mounted.panels.skillPanel.innerHTML.includes("demo_skill"));
        """
    )


def test_governance_extension_mount_fetches_capability_tiers_into_panel_and_state():
    run_node(
        """
        import assert from "node:assert/strict";
        import { mountGovernanceExtensionPanels } from "./marvis/static/js/v2/governance_extensions.js";
        import {
          getCapabilityTiers,
          getSelectedTier,
          resetV2State,
        } from "./marvis/static/js/v2/state_v2.js";

        function makeElement(tagName) {
          return {
            tagName: tagName.toUpperCase(),
            id: "",
            innerHTML: "",
            className: "",
            dataset: {},
            attributes: {},
            children: [],
            setAttribute(name, value) {
              this.attributes[name] = String(value);
            },
            appendChild(child) {
              this.children.push(child);
              return child;
            },
          };
        }

        resetV2State();
        const calls = [];
        const root = makeElement("div");
        root.ownerDocument = { createElement: makeElement };
        root.querySelector = (selector) => {
          const id = selector.startsWith("#") ? selector.slice(1) : selector;
          return root.children.find((child) => child.id === id) ?? null;
        };

        const mounted = mountGovernanceExtensionPanels(root, {
          capabilityActions: {
            listCapabilityTiers: async () => {
              calls.push(["listCapabilityTiers"]);
              return {
                default: "autonomous",
                tiers: [
                  { name: "autonomous", summary: "Auto <mode>", max_replans: 8 },
                ],
              };
            },
          },
        });
        await Promise.resolve();
        await Promise.resolve();

        assert.deepEqual(calls, [["listCapabilityTiers"]]);
        assert.equal(getSelectedTier(), "autonomous");
        assert.equal(getCapabilityTiers()[0].name, "autonomous");
        assert.equal(mounted.panels.capabilityPanel.innerHTML.includes("Auto <mode>"), false);
        assert.ok(mounted.panels.capabilityPanel.innerHTML.includes("Auto &lt;mode&gt;"));
        assert.ok(mounted.panels.capabilityPanel.innerHTML.includes("最大重规划"));
        """
    )
