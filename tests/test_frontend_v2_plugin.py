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


def test_plugin_manager_html_escapes_rows_and_hides_builtin_delete():
    run_node(
        """
        import assert from "node:assert/strict";
        import { pluginManagerHtml } from "./marvis/static/js/v2/plugin_manager.js";

        const html = pluginManagerHtml({
          plugins: [
            {
              name: "core",
              display_name: "Core <pack>",
              version: "1.0",
              builtin: true,
              enabled: true,
              tool_count: 2,
            },
            {
              name: "user-pack",
              display_name: "User Pack",
              version: "0.2",
              builtin: false,
              enabled: false,
              tool_count: 1,
            },
          ],
        });

        assert.equal(html.includes("Core <pack>"), false);
        assert.ok(html.includes("Core &lt;pack&gt;"));
        assert.ok(html.includes('data-toggle-plugin="core" checked'));
        assert.equal(html.includes('data-remove-plugin="core"'), false);
        assert.ok(html.includes('data-toggle-plugin="user-pack"'));
        assert.ok(html.includes('data-remove-plugin="user-pack"'));
        assert.ok(html.includes('data-show-tools="user-pack"'));
        assert.ok(html.includes('data-plugin-tools="user-pack"'));
        """
    )


def test_plugin_tools_html_escapes_schema_payloads():
    run_node(
        """
        import assert from "node:assert/strict";
        import { pluginToolsHtml } from "./marvis/static/js/v2/plugin_manager.js";

        const html = pluginToolsHtml({
          tools: [
            {
              name: "join<script>",
              description: "Run <join>",
              // schema is rendered as a structured field table (schemaTableHtml),
              // so inject payloads into the fields it actually shows: property
              // name, type, and enum/description constraints.
              input_schema: {
                type: "object",
                properties: {
                  "field<script>": {
                    type: "string<x>",
                    description: "danger <b>bold</b>",
                    enum: ["a<script>", "b"],
                  },
                },
                required: ["field<script>"],
              },
              output_schema: {
                type: "object",
                properties: { "out<img>": { type: "number" } },
              },
            },
          ],
        });

        // No raw markup from any user-controlled field may reach the DOM.
        assert.equal(html.includes("<script>"), false);
        assert.equal(html.includes("<b>bold</b>"), false);
        assert.equal(html.includes("<img>"), false);
        assert.equal(html.includes("string<x>"), false);

        // Tool name + description stay escaped.
        assert.ok(html.includes("join&lt;script&gt;"));
        assert.ok(html.includes("Run &lt;join&gt;"));

        // Schema field name / type / constraint text render escaped in the table.
        assert.ok(html.includes("field&lt;script&gt;"));
        assert.ok(html.includes("string&lt;x&gt;"));
        assert.ok(html.includes("a&lt;script&gt;"));
        assert.ok(html.includes("danger &lt;b&gt;bold&lt;/b&gt;"));
        assert.ok(html.includes("out&lt;img&gt;"));
        """
    )


def test_plugin_handlers_upload_toggle_remove_and_show_tools():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachPluginHandlers } from "./marvis/static/js/v2/plugin_manager.js";

        const calls = [];
        const messages = [];
        const toolSlot = { innerHTML: "" };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector(selector) {
            return selector === '[data-plugin-tools="demo"]' ? toolSlot : null;
          },
        };

        const detach = attachPluginHandlers(root, {
          uploadPlugin: async (file) => calls.push(["uploadPlugin", file.name]),
          setPluginEnabled: async (name, on) => calls.push(["setPluginEnabled", name, on]),
          removePlugin: async (name) => calls.push(["removePlugin", name]),
          listPluginTools: async (name) => {
            calls.push(["listPluginTools", name]);
            return { tools: [{ name: "tool", description: "desc", input_schema: {}, output_schema: {} }] };
          },
          refreshPlugins: async () => calls.push(["refreshPlugins"]),
          confirmRemove: async () => true,
          showError: (message) => messages.push(message),
        });

        const uploadTarget = {
          files: [{ name: "plugin.zip" }],
          closest(selector) {
            return selector === "[data-upload-plugin]" ? this : null;
          },
        };
        await listeners.change({ target: uploadTarget });
        assert.deepEqual(calls.splice(0), [["uploadPlugin", "plugin.zip"], ["refreshPlugins"]]);

        const toggleTarget = {
          checked: false,
          closest(selector) {
            return selector === "[data-toggle-plugin]"
              ? { dataset: { togglePlugin: "demo" }, checked: this.checked }
              : null;
          },
        };
        await listeners.change({ target: toggleTarget });
        assert.deepEqual(calls.splice(0), [["setPluginEnabled", "demo", false], ["refreshPlugins"]]);

        const removeTarget = {
          closest(selector) {
            return selector === "[data-remove-plugin]"
              ? { dataset: { removePlugin: "demo" } }
              : null;
          },
        };
        await listeners.click({ target: removeTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [["removePlugin", "demo"], ["refreshPlugins"]]);

        const toolsButton = { dataset: { showTools: "demo" } };
        const toolsTarget = {
          closest(selector) {
            return selector === "[data-show-tools]" ? toolsButton : null;
          },
        };
        // First click fetches + shows the tools and flips the button to 收起工具.
        await listeners.click({ target: toolsTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [["listPluginTools", "demo"]]);
        assert.ok(toolSlot.innerHTML.includes("tool"));
        assert.equal(toolsButton.dataset.expanded, "true");
        assert.equal(toolsButton.textContent, "收起工具");
        // Second click folds them back up without re-fetching (was stuck open before).
        await listeners.click({ target: toolsTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), []);
        assert.equal(toolSlot.innerHTML, "");
        assert.equal(toolsButton.dataset.expanded, "false");
        assert.equal(toolsButton.textContent, "查看工具");

        const emptyUploadTarget = {
          files: [],
          closest(selector) {
            return selector === "[data-upload-plugin]" ? this : null;
          },
        };
        await listeners.change({ target: emptyUploadTarget });
        assert.deepEqual(messages, []);

        detach();
        assert.equal(listeners.click, undefined);
        assert.equal(listeners.change, undefined);
        """
    )


def test_plugin_handlers_show_status_specific_upload_conflict_error():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachPluginHandlers } from "./marvis/static/js/v2/plugin_manager.js";

        const messages = [];
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector() { return null; },
        };

        attachPluginHandlers(root, {
          uploadPlugin: async () => {
            const error = new Error("duplicate plugin uploaded_pack");
            error.status = 409;
            throw error;
          },
          showError: (message) => messages.push(message),
        });

        const uploadTarget = {
          files: [{ name: "plugin.zip" }],
          closest(selector) {
            return selector === "[data-upload-plugin]" ? this : null;
          },
        };
        await listeners.change({ target: uploadTarget });

        assert.equal(messages.length, 1);
        assert.ok(messages[0].includes("插件已安装"));
        assert.ok(messages[0].includes("新版本"));
        assert.equal(messages[0].includes("duplicate plugin"), false);
        """
    )


def test_plugin_handlers_show_status_specific_upload_manifest_error():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachPluginHandlers } from "./marvis/static/js/v2/plugin_manager.js";

        const messages = [];
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector() { return null; },
        };

        attachPluginHandlers(root, {
          uploadPlugin: async () => {
            const error = new Error("unknown hook <script>");
            error.status = 422;
            throw error;
          },
          showError: (message) => messages.push(message),
        });

        const uploadTarget = {
          files: [{ name: "plugin.zip" }],
          closest(selector) {
            return selector === "[data-upload-plugin]" ? this : null;
          },
        };
        await listeners.change({ target: uploadTarget });

        assert.equal(messages.length, 1);
        assert.ok(messages[0].includes("manifest"));
        assert.ok(messages[0].includes("重新上传"));
        assert.equal(messages[0].includes("unknown hook"), false);
        assert.equal(messages[0].includes("<script>"), false);
        """
    )
