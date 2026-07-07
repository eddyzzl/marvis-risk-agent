import { escapeHtml } from "../ui-utils.js";
import { schemaTableHtml } from "./schema_table.js";
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

function failurePolicyLabel(value) {
  return {
    fail: "失败即中止",
    retry: "失败可重试",
    skip: "失败可跳过",
  }[value] || value || "未声明";
}

function determinismLabel(value) {
  return {
    deterministic: "确定性",
    stochastic: "非确定性",
  }[value] || value || "未声明";
}

function toolTriggerText(tool, hooks = []) {
  const events = hooks
    .filter((hook) => hook?.tool === tool.name)
    .map((hook) => hook.event)
    .filter(Boolean);
  return events.length ? `Hook: ${events.join(" / ")}` : "手动/Planner 调用";
}

function toolImplementationRows(tool = {}, data = {}) {
  const moduleName = data.module || "";
  const entrypoint = tool.entrypoint ? [moduleName, tool.entrypoint].filter(Boolean).join(".") : moduleName;
  return [
    ["实现", entrypoint || "未声明"],
    ["触发", toolTriggerText(tool, data.hooks || [])],
    ["确定性", determinismLabel(tool.determinism)],
    ["失败策略", failurePolicyLabel(tool.failure_policy)],
    ["超时", tool.timeout_seconds ? `${tool.timeout_seconds}s` : "未声明"],
    ["内存上限", tool.memory_limit_mb ? `${tool.memory_limit_mb} MB` : "未声明"],
    ["副作用", Array.isArray(tool.side_effects) && tool.side_effects.length ? tool.side_effects.join(" / ") : "无"],
  ];
}

// Monochrome leading glyphs per implementation field — same faint-tile language
// as the static settings rows (.settings-row-ico).
const TOOL_IMPL_ICONS = {
  "实现": '<path d="M8 6l-5 6 5 6"/><path d="M16 6l5 6-5 6"/>',
  "触发": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
  "确定性": '<path d="M12 3l7 3v5c0 4-3 7-7 8-4-1-7-4-7-8V6z"/><path d="M9 12l2 2 4-4"/>',
  "失败策略": '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/>',
  "超时": '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>',
  "内存上限": '<rect x="4" y="6" width="16" height="12" rx="2"/><path d="M8 6V3M12 6V3M16 6V3M8 21v-3M12 21v-3M16 21v-3"/>',
  "副作用": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
};

function toolImplIconHtml(label) {
  const path = TOOL_IMPL_ICONS[label] || "";
  return `<span class="settings-row-ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${path}</svg></span>`;
}

function toolImplementationHtml(tool, data) {
  return [
    '<dl class="plugin-tool-impl">',
    ...toolImplementationRows(tool, data).map(([label, value]) => (
      `<div>${toolImplIconHtml(label)}<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`
    )),
    "</dl>",
  ].join("");
}

const PLUGIN_ZIP_TREE_EXAMPLE = `sample_pack.zip
└── sample_pack/
    ├── manifest.json
    └── tools.py`;

const PLUGIN_MANIFEST_EXAMPLE = {
  name: "sample_pack",
  version: "0.1.0",
  display_name: "Sample Pack",
  description: "Demo plugin with one deterministic tool.",
  module: "sample_pack.tools",
  python_requires: ">=3.10,<3.14",
  permissions: ["read:input"],
  tools: [
    {
      name: "echo",
      summary: "Echo a message.",
      entrypoint: "tool_echo",
      determinism: "deterministic",
      timeout_seconds: 10,
      failure_policy: "fail",
      side_effects: ["read:input"],
      input_schema: {
        type: "object",
        properties: { message: { type: "string", minLength: 1 } },
        required: ["message"],
        additionalProperties: false,
      },
      output_schema: {
        type: "object",
        properties: { echoed: { type: "string" } },
        required: ["echoed"],
        additionalProperties: false,
      },
    },
  ],
  hooks: [],
};

const PLUGIN_TOOL_EXAMPLE = `def tool_echo(inputs, ctx):
    return {"echoed": inputs["message"]}`;

function codeBlockHtml(code) {
  return `<pre><code>${escapeHtml(code)}</code></pre>`;
}

function pluginFormatGuideHtml() {
  return `<details class="extension-format-guide plugin-format-guide">
    <summary>
      <span>
        <strong>插件包格式示例</strong>
        <span>上传 zip；包内放 manifest.json 和实现模块，manifest 里声明工具、入口函数、权限和输入输出 schema。</span>
      </span>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
    </summary>
    <div class="extension-format-guide-body">
      <div>
        <strong>zip 结构</strong>
        ${codeBlockHtml(PLUGIN_ZIP_TREE_EXAMPLE)}
      </div>
      <div>
        <strong>manifest.json</strong>
        ${codeBlockHtml(JSON.stringify(PLUGIN_MANIFEST_EXAMPLE, null, 2))}
      </div>
      <div>
        <strong>tools.py</strong>
        ${codeBlockHtml(PLUGIN_TOOL_EXAMPLE)}
      </div>
    </div>
  </details>`;
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
        ${tool.summary || tool.description ? `<span>${escapeHtml(tool.summary || tool.description)}</span>` : ""}
      </div>
      ${toolImplementationHtml(tool, data)}
      <div class="plugin-tool-schema">
        ${schemaTableHtml(tool.input_schema, "输入")}
        ${schemaTableHtml(tool.output_schema, "输出")}
      </div>
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
    ${pluginFormatGuideHtml()}
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
