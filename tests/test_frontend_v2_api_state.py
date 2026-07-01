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
        "state_v2.js",
        "governance_extensions.js",
        "plan_view.js",
        "plan_confirm.js",
        "join_review.js",
        "plugin_manager.js",
        "skill_manager.js",
        "workflow_create.js",
        "artifact_view.js",
        "model_delivery_panel.js",
        "modeling_setup_panel.js",
        "capability.js",
        "memory_manager.js",
        "subagent_view.js",
        "loop_progress.js",
    ):
        assert (static_v2 / module_name).is_file()


def test_api_wrappers_keep_formdata_boundary_under_fetch_control():
    run_node(
        """
        import assert from "node:assert/strict";
        import { apiDelete, apiGet, apiPost } from "./marvis/static/js/api.js";

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

        await apiGet("api/tasks");
        assert.equal(calls.at(-1).url, "/api/tasks");
        assert.equal(calls.at(-1).options.method, "GET");

        await apiPost("/api/plans/p1/confirm", { approved: true });
        assert.equal(calls.at(-1).options.method, "POST");
        assert.equal(calls.at(-1).options.headers["Content-Type"], "application/json");
        assert.deepEqual(JSON.parse(calls.at(-1).options.body), { approved: true });

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
        assert.equal(calls.at(-1).url, "/api/plugins/demo");
        assert.equal(calls.at(-1).options.method, "DELETE");
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
        await confirmPlan("plan/1");
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/confirm");
        await runPlan("plan/1");
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/run");
        await confirmStep("plan/1", "step/a");
        assert.equal(calls.at(-1).url, "/api/plans/plan%2F1/steps/step%2Fa/confirm");
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
