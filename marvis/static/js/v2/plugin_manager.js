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

// Keeps the 查看工具 / 收起工具 toggle button's label, aria-expanded and
// data-expanded state in sync. Guards setAttribute so unit-test mock buttons
// (plain objects with only a dataset) don't throw.
function setToolsButtonExpanded(button, expanded) {
  button.dataset.expanded = expanded ? "true" : "false";
  if (typeof button.setAttribute === "function") {
    button.setAttribute("aria-expanded", expanded ? "true" : "false");
  }
  button.textContent = expanded ? "收起工具" : "查看工具";
}

export function pluginToolsHtml(data = {}) {
  const tools = data.tools || [];
  if (!tools.length) {
    return '<div class="v2-empty" data-v2-empty="plugin-tools">暂无工具</div>';
  }
  return tools.map((tool) => (
    `<section class="plugin-tool">
      <div class="plugin-tool-head">
        <strong>${escapeHtml(tool.name || "")}</strong>
        ${tool.description ? `<span>${escapeHtml(tool.description)}</span>` : ""}
      </div>
      <details class="plugin-tool-schemas">
        <summary>输入 / 输出 schema</summary>
        <div class="plugin-tool-schema">
          <div><span class="plugin-schema-label">输入</span><pre><code>${schemaText(tool.input_schema)}</code></pre></div>
          <div><span class="plugin-schema-label">输出</span><pre><code>${schemaText(tool.output_schema)}</code></pre></div>
        </div>
      </details>
    </section>`
  )).join("");
}

export function pluginRowHtml(plugin) {
  const name = plugin?.name || "";
  const displayName = plugin?.display_name || name;
  const checked = plugin?.enabled ? " checked" : "";
  const builtin = Boolean(plugin?.builtin);
  const description = plugin?.description || "";
  const remove = builtin
    ? ""
    : `<button type="button" class="button secondary compact danger" data-remove-plugin="${escapeHtml(name)}">移除</button>`;
  return `<section class="plugin-row${builtin ? " plugin-builtin" : ""}" data-plugin-row="${escapeHtml(name)}">
    <div class="plugin-row-head">
      <div class="plugin-row-id">
        <strong>${escapeHtml(displayName)}</strong>
        <span class="plugin-version">v${escapeHtml(plugin?.version || "")}</span>
        ${builtin ? '<span class="plugin-builtin-badge">内置</span>' : ""}
      </div>
      <input type="checkbox" class="plugin-toggle" data-toggle-plugin="${escapeHtml(name)}"${checked} aria-label="启用 ${escapeHtml(displayName)}">
    </div>
    ${description ? `<p class="plugin-row-desc">${escapeHtml(description)}</p>` : ""}
    <div class="plugin-row-foot">
      <span class="plugin-tool-count">${escapeHtml(plugin?.tool_count ?? 0)} 个工具</span>
      <div class="plugin-row-actions">
        <button type="button" class="button secondary compact" data-show-tools="${escapeHtml(name)}" data-expanded="false" aria-expanded="false">查看工具</button>
        ${remove}
      </div>
    </div>
    <div class="plugin-tools" data-plugin-tools="${escapeHtml(name)}"></div>
  </section>`;
}

export function pluginManagerHtml(data = {}) {
  const plugins = data.plugins || [];
  const rows = plugins.length
    ? plugins.map(pluginRowHtml).join("")
    : '<div class="v2-empty" data-v2-empty="plugins">暂无插件</div>';
  return `<section class="plugin-manager">
    <div class="plugin-upload">
      <span class="plugin-upload-text">
        <strong>上传插件</strong>
        <span>仅支持 .zip 插件包，安装后出现在下方列表，可随时启停或移除。</span>
      </span>
      <label class="button secondary compact plugin-upload-button">
        选择文件
        <input type="file" data-upload-plugin accept=".zip">
      </label>
    </div>
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
    return confirm(`确定移除插件 ${name}？`);
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

function pluginUploadErrorMessage(error) {
  if (error?.status === 409) {
    return "插件已安装。请先移除旧版本，或上传新版本包。";
  }
  if (error?.status === 422) {
    return "插件 manifest 无效。请修复 manifest.json 后重新上传。";
  }
  if (error?.status === 403) {
    return "插件变更需要本地插件管理员确认。";
  }
  return error?.message || "插件上传失败";
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
        actions.showError(pluginUploadErrorMessage(error));
      }
      return;
    }

    const toggle = closest(target, "[data-toggle-plugin]");
    if (toggle?.dataset?.togglePlugin) {
      try {
        await actions.setPluginEnabled(toggle.dataset.togglePlugin, Boolean(toggle.checked));
        await actions.refreshPlugins();
      } catch (error) {
        actions.showError(error?.message || "插件启用状态更新失败");
      }
    }
  };

  const clickHandler = async (event) => {
    const target = event.target;
    const removeButton = closest(target, "[data-remove-plugin]");
    if (removeButton?.dataset?.removePlugin) {
      event.preventDefault?.();
      const name = removeButton.dataset.removePlugin;
      if (!(await actions.confirmRemove(name))) {
        return;
      }
      try {
        await actions.removePlugin(name);
        await actions.refreshPlugins();
      } catch (error) {
        actions.showError(error?.message || "插件移除失败");
      }
      return;
    }

    const toolsButton = closest(target, "[data-show-tools]");
    if (toolsButton?.dataset?.showTools) {
      event.preventDefault?.();
      const name = toolsButton.dataset.showTools;
      const slot = root.querySelector?.(`[data-plugin-tools="${cssEscape(name)}"]`);
      // 查看工具 is a toggle: a second click folds the tool list back up instead
      // of re-fetching and leaving it stuck open (the old one-way behavior).
      if (toolsButton.dataset.expanded === "true") {
        if (slot) slot.innerHTML = "";
        setToolsButtonExpanded(toolsButton, false);
        return;
      }
      try {
        const data = await actions.listPluginTools(name);
        if (slot) {
          slot.innerHTML = pluginToolsHtml(data);
        }
        setToolsButtonExpanded(toolsButton, true);
      } catch (error) {
        actions.showError(error?.message || "插件工具读取失败");
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
