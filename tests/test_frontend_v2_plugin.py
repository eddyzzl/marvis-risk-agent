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
              input_schema: { type: "object", title: "<input>" },
              output_schema: { type: "object", title: "<output>" },
            },
          ],
        });

        assert.equal(html.includes("<script>"), false);
        assert.ok(html.includes("join&lt;script&gt;"));
        assert.ok(html.includes("Run &lt;join&gt;"));
        assert.ok(html.includes("&quot;&lt;input&gt;&quot;"));
        assert.ok(html.includes("&quot;&lt;output&gt;&quot;"));
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
          confirmRemove: () => true,
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

        const toolsTarget = {
          closest(selector) {
            return selector === "[data-show-tools]"
              ? { dataset: { showTools: "demo" } }
              : null;
          },
        };
        await listeners.click({ target: toolsTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [["listPluginTools", "demo"]]);
        assert.ok(toolSlot.innerHTML.includes("tool"));

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
