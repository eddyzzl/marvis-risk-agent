import { attachArtifactHandlers } from "./artifact_view.js";
import { attachCapabilityHandlers, renderTierSettings, renderTierSettingsShell } from "./capability.js";
import { listDrafts as listDraftsApi } from "./api_v2.js";
import { attachDraftHandlers, renderDraftManager, renderDraftManagerShell } from "./draft_manager.js";
import { attachJoinHandlers, renderJoinReview } from "./join_review.js";
import { renderLoopEvents } from "./loop_progress.js";
import { attachMemoryHandlers, renderMemoryManager, renderMemoryManagerShell } from "./memory_manager.js";
import { attachPlanConfirmHandlers } from "./plan_confirm.js";
import { renderPlanView } from "./plan_view.js";
import { attachPluginHandlers, renderPluginManager, renderPluginManagerShell } from "./plugin_manager.js";
import { attachSkillHandlers, renderSkillManager, renderSkillManagerShell } from "./skill_manager.js";
import { renderSubAgentView } from "./subagent_view.js";
import { attachGoalHandlers, renderGoalComposer } from "./workflow_create.js";

const panelDefinitions = [
  { id: "goalPanel", className: "v2-goal-panel", label: "V2 目标编排", title: "计划生成", description: "对当前任务生成可校验的 Workflow 计划。" },
  { id: "planPanel", className: "v2-plan-panel", label: "V2 执行计划", title: "执行计划", description: "查看步骤、确认门、状态和输出引用。" },
  { id: "joinPanel", className: "v2-join-panel", label: "V2 数据处理复核", title: "数据处理", description: "选择主表和特征表，复核键匹配与去重策略。" },
  { id: "subAgentPanel", className: "v2-subagent-panel", label: "V2 子 Agent", title: "子 Agent", description: "查看并行执行分支和授权工具。" },
  { id: "pluginPanel", className: "v2-plugin-panel", label: "V2 插件", title: "插件", description: "管理可调用工具包。" },
  { id: "skillPanel", className: "v2-skill-panel", label: "V2 Workflow 模板", title: "Workflow 模板", description: "加载和校验用户可编写模板。" },
  { id: "draftPanel", className: "v2-draft-panel", label: "V2 草稿工具", title: "草稿工具", description: "从学习材料生成、试运行并晋升工具草稿。" },
  { id: "capabilityPanel", className: "v2-capability-panel", label: "V2 能力档位", title: "能力档位", description: "控制自治程度，不改变证据和安全护栏。" },
  { id: "memoryPanel", className: "v2-memory-panel", label: "V2 记忆审计", title: "记忆审计", description: "查看沉淀、来源记忆和回滚记录。" },
  { id: "loopPanel", className: "v2-loop-panel", label: "V2 循环进展", title: "循环进展", description: "跟踪重规划、探索分支和无进展事件。" },
  { id: "artifactPanel", className: "v2-artifact-panel", label: "V2 工件", title: "工件预览", description: "查看步骤输出、数据集预览和报告工件。" },
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

function renderEmptyPanel(container, key, text) {
  if (!container) {
    return () => {};
  }
  container.innerHTML = `<div class="v2-empty" data-v2-empty="${key}">${text}</div>`;
  return () => {};
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
    const draftActions = options.draftActions || {};
    const capabilityActions = options.capabilityActions || {};
    const memoryActions = options.memoryActions || {};
    const refreshPlugins = () => renderPluginManager(panels.pluginPanel, pluginActions);
    const refreshSkills = () => renderSkillManager(panels.skillPanel, skillActions);
    const refreshDrafts = (query = {}) => renderDraftManager(panels.draftPanel, {
      ...draftActions,
      listDrafts: () => (draftActions.listDrafts || listDraftsApi)(query),
    });
    const refreshMemories = () => renderMemoryManager(panels.memoryPanel, memoryActions);
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
      renderGoalComposer(panels.goalPanel),
      renderPlanView(panels.planPanel),
      renderJoinReview(panels.joinPanel),
      renderSubAgentView(panels.subAgentPanel),
      renderPluginManagerShell(panels.pluginPanel),
      renderSkillManagerShell(panels.skillPanel),
      renderDraftManagerShell(panels.draftPanel),
      renderTierSettingsShell(panels.capabilityPanel),
      renderMemoryManagerShell(panels.memoryPanel),
      renderLoopEvents(panels.loopPanel),
      renderEmptyPanel(panels.artifactPanel, "artifact", "暂无工件预览"),
    ];
    quietlyRefresh(refreshPlugins);
    quietlyRefresh(refreshSkills);
    quietlyRefresh(refreshDrafts);
    quietlyRefresh(refreshMemories);
    void refreshCapabilities();
    if (typeof root.addEventListener === "function") {
      cleanups.push(
        attachCapabilityHandlers(root),
        attachArtifactHandlers(root, () => panels.artifactPanel),
        attachJoinHandlers(root, options.taskId || ""),
        attachGoalHandlers(root, options.taskId || ""),
        attachPlanConfirmHandlers(root),
        attachPluginHandlers(root, { ...pluginActions, refreshPlugins }),
        attachSkillHandlers(root, { ...skillActions, refreshSkills }),
        attachDraftHandlers(root, { ...draftActions, refreshDrafts }),
        attachMemoryHandlers(root, { ...memoryActions, refreshMemories }),
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
