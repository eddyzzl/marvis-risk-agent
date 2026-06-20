import { escapeHtml } from "../ui-utils.js";
import {
  listPluginTools as listPluginToolsApi,
  listPlugins as listPluginsApi,
  removePlugin as removePluginApi,
  setPluginEnabled as setPluginEnabledApi,
  uploadPlugin as uploadPluginApi,
} from "./api_v2.js";
import { setPlugins } from "./state_v2.js";

function cssEscape(value) {
  if (globalThis.CSS?.escape) {
    return globalThis.CSS.escape(String(value));
  }
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function schemaText(schema) {
  return escapeHtml(JSON.stringify(schema || {}, null, 2));
}

export function pluginToolsHtml(data = {}) {
  const tools = data.tools || [];
  if (!tools.length) {
    return '<div class="v2-empty" data-v2-empty="plugin-tools">No tools</div>';
  }
  return tools.map((tool) => (
    `<section class="plugin-tool">
      <header>
        <strong>${escapeHtml(tool.name || "")}</strong>
        <span>${escapeHtml(tool.description || "")}</span>
      </header>
      <div class="plugin-tool-schema">
        <pre><code>${schemaText(tool.input_schema)}</code></pre>
        <pre><code>${schemaText(tool.output_schema)}</code></pre>
      </div>
    </section>`
  )).join("");
}

export function pluginRowHtml(plugin) {
  const name = plugin?.name || "";
  const displayName = plugin?.display_name || name;
  const checked = plugin?.enabled ? " checked" : "";
  const builtin = Boolean(plugin?.builtin);
  const remove = builtin
    ? ""
    : `<button type="button" data-remove-plugin="${escapeHtml(name)}">Remove</button>`;
  return `<section class="plugin-row${builtin ? " plugin-builtin" : ""}" data-plugin-row="${escapeHtml(name)}">
    <header>
      <strong>${escapeHtml(displayName)}</strong>
      <span class="plugin-version">v${escapeHtml(plugin?.version || "")}</span>
      ${builtin ? '<span class="plugin-builtin-badge">builtin</span>' : ""}
    </header>
    <label>
      <input type="checkbox" data-toggle-plugin="${escapeHtml(name)}"${checked}>
      enabled
    </label>
    <span class="plugin-tool-count">${escapeHtml(plugin?.tool_count ?? 0)} tools</span>
    ${remove}
    <button type="button" data-show-tools="${escapeHtml(name)}">Show tools</button>
    <div class="plugin-tools" data-plugin-tools="${escapeHtml(name)}"></div>
  </section>`;
}

export function pluginManagerHtml(data = {}) {
  const plugins = data.plugins || [];
  const rows = plugins.length
    ? plugins.map(pluginRowHtml).join("")
    : '<div class="v2-empty" data-v2-empty="plugins">No plugins</div>';
  return `<section class="plugin-manager">
    <label class="plugin-upload">
      <input type="file" data-upload-plugin accept=".zip,.tar,.gz,.tgz">
      Upload plugin
    </label>
    <div class="plugin-list">${rows}</div>
  </section>`;
}

export function renderPluginManagerShell(container, data = {}) {
  if (!container) {
    throw new Error("renderPluginManagerShell requires a container");
  }
  if (container.dataset) {
    container.dataset.v2PluginManager = "true";
  }
  container.innerHTML = pluginManagerHtml(data);
  return () => {};
}

export async function renderPluginManager(container, deps = {}) {
  if (!container) {
    throw new Error("renderPluginManager requires a container");
  }
  if (container.dataset) {
    container.dataset.v2PluginManager = "true";
  }
  const actions = {
    listPlugins: listPluginsApi,
    ...deps,
  };
  const data = await actions.listPlugins(true);
  setPlugins(data.plugins || []);
  container.innerHTML = pluginManagerHtml(data);
  return data;
}

function defaultConfirmRemove(name) {
  if (typeof confirm === "function") {
    return confirm(`Remove plugin ${name}?`);
  }
  return true;
}

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

export function attachPluginHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachPluginHandlers requires a stable event root");
  }
  const actions = {
    confirmRemove: defaultConfirmRemove,
    listPluginTools: listPluginToolsApi,
    refreshPlugins: async () => {},
    removePlugin: removePluginApi,
    setPluginEnabled: setPluginEnabledApi,
    showError: defaultShowError,
    uploadPlugin: uploadPluginApi,
    ...deps,
  };

  const changeHandler = async (event) => {
    const target = event.target;
    const uploadInput = closest(target, "[data-upload-plugin]");
    if (uploadInput) {
      const file = uploadInput.files?.[0];
      if (!file) {
        return;
      }
      try {
        await actions.uploadPlugin(file);
        await actions.refreshPlugins();
      } catch (error) {
        actions.showError(error?.message || "plugin upload failed");
      }
      return;
    }

    const toggle = closest(target, "[data-toggle-plugin]");
    if (toggle?.dataset?.togglePlugin) {
      try {
        await actions.setPluginEnabled(toggle.dataset.togglePlugin, Boolean(toggle.checked));
        await actions.refreshPlugins();
      } catch (error) {
        actions.showError(error?.message || "plugin toggle failed");
      }
    }
  };

  const clickHandler = async (event) => {
    const target = event.target;
    const removeButton = closest(target, "[data-remove-plugin]");
    if (removeButton?.dataset?.removePlugin) {
      event.preventDefault?.();
      const name = removeButton.dataset.removePlugin;
      if (!actions.confirmRemove(name)) {
        return;
      }
      try {
        await actions.removePlugin(name);
        await actions.refreshPlugins();
      } catch (error) {
        actions.showError(error?.message || "plugin remove failed");
      }
      return;
    }

    const toolsButton = closest(target, "[data-show-tools]");
    if (toolsButton?.dataset?.showTools) {
      event.preventDefault?.();
      const name = toolsButton.dataset.showTools;
      try {
        const data = await actions.listPluginTools(name);
        const slot = root.querySelector?.(`[data-plugin-tools="${cssEscape(name)}"]`);
        if (slot) {
          slot.innerHTML = pluginToolsHtml(data);
        }
      } catch (error) {
        actions.showError(error?.message || "plugin tools failed");
      }
    }
  };

  root.addEventListener("change", changeHandler);
  root.addEventListener("click", clickHandler);
  return () => {
    root.removeEventListener?.("change", changeHandler);
    root.removeEventListener?.("click", clickHandler);
  };
}
