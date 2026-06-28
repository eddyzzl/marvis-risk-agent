import { attachCapabilityHandlers, renderTierSettings, renderTierSettingsShell } from "./capability.js";
import { attachPluginHandlers, renderPluginManager, renderPluginManagerShell } from "./plugin_manager.js";
import { attachSkillHandlers, renderSkillManager, renderSkillManagerShell } from "./skill_manager.js";

const panelDefinitions = [
  { id: "pluginPanel", className: "v2-plugin-panel", label: "V2 插件", title: "插件", description: "管理可调用工具包。" },
  { id: "skillPanel", className: "v2-skill-panel", label: "V2 Workflow 模板", title: "Workflow 模板", description: "加载和校验用户可编写模板。" },
  { id: "capabilityPanel", className: "v2-capability-panel", label: "V2 能力档位", title: "能力档位", description: "控制自治程度，不改变证据和安全护栏。" },
];

const mountStateKey = "__marvisV2MountState";

function documentFor(root) {
  if (root?.ownerDocument?.createElement) {
    return root.ownerDocument;
  }
  if (typeof document !== "undefined") {
    return document;
  }
  throw new Error("mountV2 requires a DOM root with an ownerDocument");
}

function ensurePanel(root, definition) {
  const existing = root.querySelector(`#${definition.id}`);
  if (existing) {
    return existing;
  }
  const panel = documentFor(root).createElement("section");
  panel.id = definition.id;
  panel.className = `v2-panel ${definition.className}`;
  panel.dataset.v2Panel = definition.id;
  panel.dataset.panelTitle = definition.title || definition.label;
  panel.dataset.panelDescription = definition.description || "";
  panel.setAttribute("aria-label", definition.label);
  panel.setAttribute("aria-live", "polite");
  root.appendChild(panel);
  return panel;
}

export function mountV2(root, options = {}) {
  if (!root || typeof root.querySelector !== "function" || typeof root.appendChild !== "function") {
    throw new Error("mountV2 requires a stable root element");
  }
  const panels = {};
  for (const definition of panelDefinitions) {
    panels[definition.id] = ensurePanel(root, definition);
  }
  if (!root[mountStateKey]) {
    const pluginActions = options.pluginActions || {};
    const skillActions = options.skillActions || {};
    const capabilityActions = options.capabilityActions || {};
    const refreshPlugins = () => renderPluginManager(panels.pluginPanel, pluginActions);
    const refreshSkills = () => renderSkillManager(panels.skillPanel, skillActions);
    const refreshCapabilities = async () => {
      try {
        await renderTierSettings(panels.capabilityPanel, capabilityActions);
      } catch (_error) {
        renderTierSettingsShell(panels.capabilityPanel);
      }
    };
    const quietlyRefresh = (refresh) => {
      void refresh().catch(() => {});
    };
    const cleanups = [
      renderPluginManagerShell(panels.pluginPanel),
      renderSkillManagerShell(panels.skillPanel),
      renderTierSettingsShell(panels.capabilityPanel),
    ];
    quietlyRefresh(refreshPlugins);
    quietlyRefresh(refreshSkills);
    void refreshCapabilities();
    if (typeof root.addEventListener === "function") {
      cleanups.push(
        attachCapabilityHandlers(root),
        attachPluginHandlers(root, { ...pluginActions, refreshPlugins }),
        attachSkillHandlers(root, { ...skillActions, refreshSkills }),
      );
    }
    root[mountStateKey] = { cleanups };
  }
  if (root.dataset) {
    root.dataset.v2Mounted = "true";
  }
  return { root, panels, unmount: () => unmountV2(root) };
}

export function unmountV2(root) {
  const state = root?.[mountStateKey];
  if (!state) {
    return;
  }
  for (const cleanup of state.cleanups || []) {
    cleanup();
  }
  delete root[mountStateKey];
  if (root.dataset) {
    delete root.dataset.v2Mounted;
  }
}

export const v2PanelDefinitions = panelDefinitions.map((definition) => ({ ...definition }));
