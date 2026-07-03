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


def test_draft_tools_panel_controller_wires_api_and_render_state():
    run_node(
        """
        import assert from "node:assert/strict";
        // The plugin-admin header is read from <body data-marvis-plugin-admin-token>
        // (server-injected for local clients); stub it so the panel echoes it.
        globalThis.document = { body: { dataset: { marvisPluginAdminToken: "test-admin-token" } } };
        const { createDraftToolsPanelController } = await import("./marvis/static/js/draft-tools-panel.js");

        function makeElement(value = "") {
          const classes = new Set();
          return {
            value,
            textContent: "",
            innerHTML: "",
            dataset: {},
            disabled: false,
            classList: {
              add: (name) => classes.add(name),
              remove: (name) => classes.delete(name),
              contains: (name) => classes.has(name),
            },
          };
        }

        const elements = new Map();
        const ids = [
          "draftToolsStatus",
          "draftStatusFilter",
          "draftToolsList",
          "draftToolBody",
          "draftToolEmpty",
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
        ];
        for (const id of ids) elements.set(id, makeElement());
        elements.get("draftStatusFilter").value = "draft";

        const calls = [];
        const api = async (url, options = {}) => {
          calls.push([url, options]);
          if (url === "/api/drafts?status=draft") {
            return {
              drafts: [{
                id: "draft-1",
                name: "calc<script>",
                summary: "Run <margin>",
                source: "web_learning",
                status: "draft",
                task_id: "task-1",
              }],
            };
          }
          if (url === "/api/drafts/draft-1") {
            return {
              draft: {
                id: "draft-1",
                name: "calc_margin",
                summary: "Summary",
                code: "return '<unsafe>'",
                input_schema: { type: "object" },
                output_schema: { type: "object" },
                source: "web_learning",
                status: "tested",
                task_id: "task-1",
              },
              learning_note: {
                distilled: "Use revenue < cost.",
                sources: ["https://example.test/a?<x>"],
              },
              runs: [{ ok: false, error: "boom <script>", at: "now" }],
            };
          }
          if (url === "/api/drafts/draft-1/run") return { ok: true };
          if (url === "/api/drafts/draft-1/promote") return { plugin: { name: "calc_margin" } };
          throw new Error(`unexpected api call ${url}`);
        };
        const confirmed = [];
        const controller = createDraftToolsPanelController({
          $: (id) => elements.get(id),
          api,
          runAction: (action) => action(),
          showPlatformConfirm: async (payload) => {
            confirmed.push(payload.title);
            return true;
          },
        });

        assert.equal(controller.hasLoaded(), false);
        await controller.load();
        assert.equal(controller.hasLoaded(), true);
        assert.ok(elements.get("draftToolsList").innerHTML.includes("calc&lt;script&gt;"));
        assert.equal(elements.get("draftToolsList").innerHTML.includes("calc<script>"), false);

        await controller.inspect("draft-1");
        assert.equal(elements.get("draftToolName").textContent, "calc_margin");
        assert.ok(elements.get("draftLearningNote").innerHTML.includes("Use revenue &lt; cost."));
        assert.ok(elements.get("draftRunHistory").innerHTML.includes("boom &lt;script&gt;"));

        elements.get("draftRunInputs").value = '{"revenue": 10}';
        await controller.run();
        const runCall = calls.find(([url]) => url === "/api/drafts/draft-1/run");
        assert.deepEqual(JSON.parse(runCall[1].body), { inputs: { revenue: 10 } });

        elements.get("draftPromotionTestCases").value = "[]";
        await controller.promote();
        assert.equal(elements.get("draftToolsStatus").textContent, "请填写转正测试用例。");
        assert.equal(confirmed.length, 0);

        elements.get("draftPromotionTestCases").value = '[{"inputs":{"revenue":10},"expect":{"margin":7}}]';
        await controller.promote();
        const promoteCall = calls.find(([url]) => url === "/api/drafts/draft-1/promote");
        assert.equal(promoteCall[1].headers["X-MARVIS-Plugin-Admin"], "test-admin-token");
        assert.deepEqual(JSON.parse(promoteCall[1].body), {
          test_cases: [{ inputs: { revenue: 10 }, expect: { margin: 7 } }],
        });
        assert.deepEqual(confirmed, ["转正草稿工具"]);
      """
    )
