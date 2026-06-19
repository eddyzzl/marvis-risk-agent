import { attachCapabilityHandlers } from "./capability.js";
import { attachJoinHandlers } from "./join_review.js";
import { attachPlanConfirmHandlers } from "./plan_confirm.js";
import { renderPlanView } from "./plan_view.js";
import { attachPluginHandlers } from "./plugin_manager.js";
import { attachSkillHandlers } from "./skill_manager.js";
import { renderSubAgentView } from "./subagent_view.js";

const panelDefinitions = [
  { id: "planPanel", className: "v2-plan-panel", label: "V2 plan" },
  { id: "subAgentPanel", className: "v2-subagent-panel", label: "V2 sub agents" },
  { id: "pluginPanel", className: "v2-plugin-panel", label: "V2 plugins" },
  { id: "artifactPanel", className: "v2-artifact-panel", label: "V2 artifacts" },
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
  panel.setAttribute("aria-label", definition.label);
  panel.setAttribute("aria-live", "polite");
  root.appendChild(panel);
  return panel;
}

export function mountV2(root) {
  if (!root || typeof root.querySelector !== "function" || typeof root.appendChild !== "function") {
    throw new Error("mountV2 requires a stable root element");
  }
  const panels = {};
  for (const definition of panelDefinitions) {
    panels[definition.id] = ensurePanel(root, definition);
  }
  if (!root[mountStateKey]) {
    const cleanups = [
      renderPlanView(panels.planPanel),
      renderSubAgentView(panels.subAgentPanel),
    ];
    if (typeof root.addEventListener === "function") {
      cleanups.push(
        attachCapabilityHandlers(root),
        attachJoinHandlers(root),
        attachPlanConfirmHandlers(root),
        attachPluginHandlers(root),
        attachSkillHandlers(root),
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
