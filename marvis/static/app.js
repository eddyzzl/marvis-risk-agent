import { api, sleep } from "./js/api.js";
import {
  createAgentMemoryPanelController,
  formatMemoryConfidence,
} from "./js/agent-memory-panel.js";
import { createDraftToolsPanelController } from "./js/draft-tools-panel.js";
import {
  agentMessageContent,
  agentMessageIsAdvanceIntent,
  agentMessageIsContinuePrompt,
  agentReportMessagesForDisplay,
  agentRerunMessageFingerprint,
  agentTimelineStageDefinitions,
} from "./js/agent-conversation-view.js";
import {
  removeAgentTimelineBuckets as removeAgentTimelineBucketsDom,
  renderAgentTimeline as renderAgentTimelineDom,
  restoreResultScrollDefaultOrder as restoreResultScrollDefaultOrderDom,
  updateAgentMessageContentsInPlace as updateAgentMessageContentsInPlaceDom,
} from "./js/agent-conversation-mount.js";
import { applyBranding, normalizeBranding } from "./js/branding.js";
import { createCreateTaskDialogController } from "./js/create-task-dialog.js";
import { createMaterialSourceController } from "./js/dialogs.js";
import { createPlatformConfirmController } from "./js/platform-confirm.js";
import { claimProgressPoll, createProgressPollRegistry, releaseProgressPoll } from "./js/polling.js";
import { renderAgentMarkdown } from "./js/render-agent.js";
import {
  loadResultScrollPositions as loadStoredResultScrollPositions,
  persistResultScrollPositions as persistStoredResultScrollPositions,
  rememberSelectedTaskId as rememberStoredSelectedTaskId,
  storedSelectedTaskId as readStoredSelectedTaskId,
} from "./js/task-workspace-state.js";
import {
  renderCurrentTaskWorkspace,
  renderTaskSnapshot as renderTaskSnapshotView,
  updateWorkspaceGreeting as updateWorkspaceGreetingView,
} from "./js/task-workspace-view.js";
import { defaultTaskType, taskTypeDisplayOrder } from "./js/task-types.js";
import { createThemeController } from "./js/theme.js";
import { renderTierSettings, selectedTierStorageKey } from "./js/v2/capability.js";
import {
  driverManualAnalysisHtml as driverManualAnalysisHtmlController,
  latestInteractiveScreenMessageId as latestInteractiveScreenMessageIdController,
  stripChatInstructions as stripChatInstructionsController,
} from "./js/v2/driver_manual_analysis.js";
import {
  handleDriverConfirmClick as handleDriverConfirmClickController,
  renderDriverGateButton,
  submitDriverConfirm as submitDriverConfirmController,
} from "./js/v2/driver_gate_confirm.js";
import { mountGovernanceExtensionPanels } from "./js/v2/governance_extensions.js";
import {
  handleC1ConfirmClick as handleC1ConfirmClickController,
  handleDedupConfirmClick as handleDedupConfirmClickController,
  renderDedupPicker,
  renderJoinC1Form,
  submitC1Assignment as submitC1AssignmentController,
  submitDedupStrategies as submitDedupStrategiesController,
} from "./js/v2/join_gate_controller.js";
import {
  handleModelingWeightAdjustClick as handleModelingWeightAdjustClickController,
  renderModelingSetupPanel,
  submitModelingWeightAdjust as submitModelingWeightAdjustController,
} from "./js/v2/modeling_setup_panel.js";
import { renderModelDeliveryPanel } from "./js/v2/model_delivery_panel.js";
import { createPlanRailController, taskUsesPlanRail } from "./js/v2/plan_rail_controller.js";
import { renderPluginManager } from "./js/v2/plugin_manager.js";
import {
  handleScreenAdjustClick as handleScreenAdjustClickController,
  handleScreenConfirmClick as handleScreenConfirmClickController,
  renderScreenGateTable,
  submitScreenSelection as submitScreenSelectionController,
  submitScreenThresholdAdjust as submitScreenThresholdAdjustController,
} from "./js/v2/screen_gate_controller.js";
import { renderSkillManager } from "./js/v2/skill_manager.js";
import { getSelectedTier, onSelectedTierChange } from "./js/v2/state_v2.js";
import {
  columnFractions,
  columnHeatColors,
  columnRanks,
  parseNumeric,
  psiTier,
  psiTooltipText,
} from "./js/render-metrics.js";
import {
  activeValidationStatuses,
  agentComposerPreferenceStorageKey,
  agentTaskComposerStorageKey,
  createRenderSignatures,
  defaultBranding,
  defaultExecutionEnvironment,
  defaultPetPreference,
  explicitPetNoneStorageKey,
  metricOverviewCompleteStatuses,
  notebookReproducibilityCompleteStatuses,
  requiredMaterialRoles,
  resultScrollPositionsStorageKey,
  roleLabels,
  scanFailurePrefix,
  selectedTaskStorageKey,
  statusLabels,
  terminalTaskStatuses,
  workflowSteps,
} from "./js/state.js";
import {
  $,
  clamp,
  escapeHtml,
  fileName,
  signatureFromParts,
  splitListInput,
} from "./js/ui-utils.js";

let selectedTaskId = null;
let selectedTask = null;
let taskCache = [];
let lastMetricValues = {};
let lastMetricValuesTaskId = null;
let lastMetricTableSections = [];
const taskBusyActions = new Map();
const progressPolls = createProgressPollRegistry();
const resultScrollPositionsByTask = new Map();
let globalBusyAction = null;
let actionStatusOverride = null;
const themeController = createThemeController({
  onChange: () => renderSettingsState(),
});
let taskSearchQuery = "";
let taskSortMode = "created_desc";
let taskGroupMode = "none";
let lastPointerDownControl = null;
let lastPointerDownAt = 0;
let executionEnvironmentOptions = [];
let executionEnvironmentSettings = null;
let llmSettings = { default_model_id: "", models: [], enabled_models: [] };
let llmEditingIndex = null;
let agentMessages = [];
const agentComposerPreferences = restoreAgentComposerPreferences();
let agentSelectedModelId = agentComposerPreferences.model_id || "";
let agentSelectedEffort = agentComposerPreferences.effort || "high";
let agentAcceptanceMode = agentComposerPreferences.acceptance_mode || "normal";
let lastAgentRenderSignature = null;
let lastAgentStructuralSignature = null;
// Cached render-input signatures so the per-second polling loop can skip
// rewriting DOM regions whose visible inputs have not changed. Reset only
// when task selection, validation run, or filter/sort/search state changes.
const renderSignatures = createRenderSignatures();
const agentTypingState = new Map();
// messageId -> content as it appeared when the typewriter caught up and the
// server stopped streaming. Lets a later streaming-resumed render seed
// visible with the bytes the user already saw, instead of replaying from 0.
const agentTypingCompleted = new Map();
let agentTypingTimer = null;
let agentAutoScrollFrame = null;
// taskId -> [{triggerMessageId, stage, sectionId, headingHtml, label,
//             contentClassName, contentHtml}, ...]
// One entry per rerun event: the previous live section's preview is frozen
// at the moment the rerun is requested and rendered inline above the rerun
// user message so chart history persists alongside the chat history.
const taskFrozenSectionSnapshots = new Map();
let pendingResultScrollRestoreTaskId = null;
let resultScrollRestoreFrame = null;
let resultScrollPersistFrame = null;
let suppressAgentAutoScrollTaskId = null;
let pendingTaskContentLoadTaskId = null;
let taskContentSettleTimer = null;
let latestNotebookSteps = [];
let sidebarCollapsed = false;
let taskSearchActive = false;
let sidebarSlideTimer = null;
let scanAbortController = null;
let petPreference = defaultPetPreference;
let petDragState = null;
let petReactionMood = null;
let petReactionKey = "";
let petReactionTimer = null;
let taskHeroGlassFrame = null;
let taskHeroGlassActive = null;
let taskHeroCanScroll = false;
const platformConfirm = createPlatformConfirmController({ getElementById: $ });
const showPlatformConfirm = platformConfirm.showPlatformConfirm;
const bindPlatformConfirmDialog = platformConfirm.bindPlatformConfirmDialog;
const materialSourceController = createMaterialSourceController({
  $,
  onFilesChanged: renderMaterialUploadSelection,
});
const createTaskDialog = createCreateTaskDialogController({
  $,
  materialSourceController,
  getSelectedTier,
  selectedTierStorageKey,
  onUnavailableTaskType: (message) => {
    showComingSoonToast(message);
    setActionStatus(message, "info", "这个入口会继续展示在任务启动页，但当前不会打开创建弹窗。");
  },
});
const agentMemoryPanel = createAgentMemoryPanelController({
  $,
  api,
  runAction,
  showPlatformConfirm,
  openMemorySettings: (navKey) => openGovernanceSettingsCenter(navKey),
  openMemoryDetails: () => {
    const details = $("memoryManageDetails");
    if (details) details.open = true;
  },
});
const draftToolsPanel = createDraftToolsPanelController({
  $,
  api,
  runAction,
  showPlatformConfirm,
});
const planRailController = createPlanRailController({
  $,
  stepCheckerHtml,
  getSelectedTask: () => selectedTask,
  getSelectedTaskId: () => selectedTaskId,
  getAgentMessages: () => agentMessages,
  isAgentMode: selectedTaskIsAgentMode,
  renderWorkflowStepper,
  setActionStatus,
  refreshTasks,
  loadAgentMessages,
  renderAll,
});

const PET_REACTION_DURATION_MS = 6500;
const AGENT_STREAM_POLL_INTERVAL_MS = 180;
const AGENT_TYPEWRITER_INTERVAL_MS = 12;
const AGENT_TYPEWRITER_CHARS_PER_TICK = 2;
// When the typewriter falls far behind a streamed message, drain the backlog
// across at most this many ticks so big late chunks still feel like a reveal
// instead of a dump, but finish in well under a second.
const AGENT_TYPEWRITER_CATCHUP_TICKS = 15;
const AGENT_NO_ENABLED_MODEL_MESSAGE = "请先在设置中配置并启用大模型，再发送 Agent 消息。";
const AGENT_NO_SELECTED_MODEL_MESSAGE = "请先选择一个可用大模型，再发送 Agent 消息。";
// Follow-mode state machine: the typewriter only pulls the viewport to the
// bottom while agentAutoScrollFollows is true. recomputeAgentAutoScrollFollow
// runs on scroll events that arrive within AGENT_USER_SCROLL_INPUT_WINDOW_MS
// of a real wheel/touch — programmatic scrollTo() calls (typewriter snap-to-
// bottom, saved-position restore) reach the handler with no recent input, so
// they leave the flag alone and cannot override a still-fresh user scroll-up.
const AGENT_AUTO_SCROLL_BOTTOM_TOLERANCE_PX = 2;
const AGENT_USER_SCROLL_INPUT_WINDOW_MS = 250;
let agentAutoScrollFollows = true;
let lastUserScrollInputAt = 0;
const SIDEBAR_WIDTH_MIN = 314;
const SIDEBAR_WIDTH_MAX = 520;
const PROGRESS_WIDTH_MIN = 314;
const PROGRESS_WIDTH_MAX = 560;
const taskSortModes = new Set(["created_desc", "created_asc", "name_asc", "name_desc"]);
const taskGroupModes = new Set(["none", "task_type", "validator", "created_month"]);
const petReactionMoods = new Set(["success", "failed", "complete", "review"]);

const petDefinitions = {
  naitang: {
    name: "蛋黄",
    label: "奶油色长毛蓝眼猫，黑色领结",
    kind: "spritesheet",
    asset: "static/pets/naitang/spritesheet.webp",
  },
  xiaojiu: {
    name: "小九",
    label: "贪吃、呆萌、胆小的小猫",
    kind: "spritesheet",
    asset: "static/pets/xiaojiu/spritesheet.webp?v=c078ec6f",
  },
  auditbot: {
    name: "MARVIS",
    label: "3D 玩具审计机器人，青色护目镜眼睛和铜色耳机",
    kind: "spritesheet",
    asset: "static/pets/auditbot/spritesheet.webp",
  },
  "auditbot-pro": {
    name: "MARVIS Pro",
    label: "专业风格 3D 审计机器人",
    kind: "spritesheet",
    asset: "static/pets/auditbot-pro/spritesheet.webp",
  },
  "auditbot-poly": {
    name: "MARVIS Poly",
    label: "低多边形硬表面审计机器人",
    kind: "spritesheet",
    asset: "static/pets/auditbot-poly/spritesheet.webp",
  },
  "auditbot-ink": {
    name: "MARVIS Ink",
    label: "技术线稿风格审计机器人",
    kind: "spritesheet",
    asset: "static/pets/auditbot-ink/spritesheet.webp",
  },
  "auditbot-clay": {
    name: "MARVIS Clay",
    label: "黏土与乙烯基质感审计机器人",
    kind: "spritesheet",
    asset: "static/pets/auditbot-clay/spritesheet.webp",
  },
  "auditbot-comic": {
    name: "MARVIS Comic",
    label: "漫画描边风格审计机器人",
    kind: "spritesheet",
    asset: "static/pets/auditbot-comic/spritesheet.webp",
  },
  "auditbot-pixel": {
    name: "MARVIS Pixel",
    label: "像素风审计机器人",
    kind: "spritesheet",
    asset: "static/pets/auditbot-pixel/spritesheet.webp",
  },
};

const legacyPetPreferences = {
  danhuang: "naitang",
  buou: "xiaojiu",
  "ragdoll-cat": "xiaojiu",
};

executionEnvironmentSettings = { ...defaultExecutionEnvironment };

function taskStopped(task = selectedTask) {
  return task?.stopped === true;
}

function taskBusyAction(taskId = selectedTaskId) {
  if (!taskId) return globalBusyAction;
  const localBusyAction = taskBusyActions.get(taskId);
  if (localBusyAction) return localBusyAction;
  if (taskId === selectedTaskId) return taskServerBusyAction();
  return null;
}

function taskServerBusyAction(task = selectedTask) {
  const kind = task?.active_job_kind || "";
  if (kind === "agent") return "agent";
  if (kind === "pipeline" || kind === "notebook") return "notebook";
  if (kind === "metrics") return "metrics";
  if (kind === "report") return "report";
  if (taskStopped(task)) return null;
  return null;
}

function selectedTaskIsBusy() {
  return Boolean(taskBusyAction());
}

// Real validator name -> display alias, populated from the workspace brand.json
// via GET api/branding. Empty by default so real names never ship in this bundle.
let agentValidatorAliases = {};

async function loadBranding() {
  try {
    const response = await fetch("api/branding");
    const payload = response.ok ? await response.json() : {};
    const branding = normalizeBranding(payload);
    agentValidatorAliases = branding.validatorAliases || {};
    applyBranding(branding);
  } catch (_error) {
    applyBranding(defaultBranding);
  }
}

function currentTaskSignature(task) {
  if (!task) return "empty";
  return signatureFromParts([
    task.id || "",
    task.name || "",
    task.status || "",
    task.active_job_kind || "",
    task.status_message || "",
    task.report_available ? 1 : 0,
    taskStopped(task) ? 1 : 0,
  ]);
}

function stepFingerprint(steps) {
  // Backend-driven progress fields ONLY. Wall-clock-derived elapsed must
  // stay out; clock ticks belong in a separate text-only refresher
  // (refreshWorkflowStepperElapsedTimes), not in the structural signature.
  return Array.isArray(steps)
    ? steps.map((step) => [
        step?.id || "",
        step?.status || "",
        step?.started_at || "",
        step?.ended_at || "",
        Number.isFinite(step?.elapsed_seconds) ? Number(step.elapsed_seconds) : "",
        Number.isFinite(step?.cell_count) ? Number(step.cell_count) : "",
      ])
    : [];
}

function workflowStepperSignature(task) {
  if (!task) return "empty";
  return signatureFromParts([
    task.id || "",
    task.status || "",
    task.status_message || "",
    task.active_job_kind || "",
    task.report_available ? 1 : 0,
    taskStopped(task) ? 1 : 0,
    taskBusyAction(task.id) || "",
    stepFingerprint(notebookStepsForRail()),
    stepFingerprint(metricStepsForRail()),
  ]);
}

function taskListSignature(tasks, totalTaskCount) {
  const list = Array.isArray(tasks) ? tasks : [];
  return signatureFromParts([
    list.map((task) => [
      task.id || "",
      task.name || "",
      task.task_type || "",
      task.status || "",
      task.updated_at || "",
      task.active_job_kind || "",
      task.validator || "",
    ]),
    Number.isFinite(totalTaskCount) ? totalTaskCount : 0,
    taskSearchQuery || "",
    taskSortMode || "",
    taskGroupMode || "",
    selectedTaskId || "",
  ]);
}

function metricPreviewSignature(taskId, metricValues, tableSections) {
  return signatureFromParts([
    taskId || "",
    metricValues || {},
    tableSections || [],
  ]);
}

function resetMetricPreviewRenderSignature() {
  renderSignatures.metricPreview = "";
  renderSignatures.metricPreviewTaskId = "";
}

function resetReproducibilityRenderSignatures() {
  renderSignatures.reproducibilityEvidence = "";
  renderSignatures.reproducibilityTaskId = "";
  renderSignatures.reproducibilityAnimatedTaskId = "";
}

function resetValidationRenderSignatures() {
  renderSignatures.actionStatus = "";
  renderSignatures.currentTask = "";
  renderSignatures.workflowStepper = "";
  renderSignatures.taskList = "";
  resetMetricPreviewRenderSignature();
}

function taskTypeDefinition(taskType = createTaskDialog.activeTaskType()) {
  return createTaskDialog.taskTypeDefinition(taskType);
}

function taskTypeLabel(taskOrType = selectedTask) {
  const taskType = typeof taskOrType === "string" ? taskOrType : taskOrType?.task_type;
  return taskTypeDefinition(taskType).label;
}

function syncCreateTaskTierDefault() {
  createTaskDialog.syncCreateTaskTierDefault();
}

function openTaskDialog(taskType = defaultTaskType) {
  createTaskDialog.openTaskDialog(taskType);
}

function openTaskDialogFromCard(event) {
  createTaskDialog.openTaskDialogFromCard(event);
}

let comingSoonToastTimer = null;
function showComingSoonToast(message) {
  let toast = $("comingSoonToast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "comingSoonToast";
    toast.className = "coming-soon-toast";
    toast.setAttribute("role", "status");
    toast.setAttribute("aria-live", "polite");
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.add("is-visible");
  if (comingSoonToastTimer) clearTimeout(comingSoonToastTimer);
  comingSoonToastTimer = setTimeout(() => toast.classList.remove("is-visible"), 2400);
}

function openTaskTypeWelcome() {
  const taskDialog = $("taskDialog");
  if (taskDialog?.open) closeTaskDialog();
  if (selectedTaskId || selectedTask) {
    deselectCurrentTask();
    return;
  }
  rememberSelectedTaskId(null);
  setActionStatus("");
  renderCurrentTask({ force: true });
  renderTaskList();
}

function closeTaskDialog() {
  createTaskDialog.closeTaskDialog();
}

function closeDialogOnBackdropClick(event) {
  const dialog = event.currentTarget;
  if (!(dialog instanceof HTMLDialogElement)) return;
  if (event.target !== dialog || !dialog.open) return;
  dialog.close();
}

function bindDialogBackdropDismissal() {
  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", closeDialogOnBackdropClick);
  });
}

function renderMaterialUploadSelection(files = materialSourceController.selectedFiles()) {
  const status = $("materialUploadStatus");
  if (!status) return;
  if (files.length === 0) {
    status.textContent = "请选择文件或文件夹。";
    return;
  }
  const names = files
    .slice(0, 3)
    .map((file) => file.name)
    .join("、");
  const suffix = files.length > 3 ? ` 等 ${files.length} 个文件` : "";
  const folderCount = new Set(
    files
      .map((file) => (file.relativePath || "").split("/").slice(0, -1).join("/"))
      .filter(Boolean),
  ).size;
  const folderText = folderCount > 0 ? `，包含 ${folderCount} 个目录` : "";
  status.textContent = `已选择 ${names}${suffix}${folderText}。`;
}

function bindRunModeDeselectableCards() {
  createTaskDialog.bindRunModeDeselectableCards();
}

function openExecutionEnvironmentDialog() {
  $("executionEnvironmentStatus").textContent = "正在读取执行环境...";
  $("executionEnvironmentStatus").className = "status";
  openGovernanceSettingsCenter("execution-environment");
}

function closeExecutionEnvironmentDialog() {
  closeGovernanceSettingsDialog();
}

function openLLMSettingsDialog() {
  setLLMSettingsStatus("正在读取大模型配置...");
  openGovernanceSettingsCenter("llm");
}

function closeLLMSettingsDialog() {
  closeGovernanceSettingsDialog();
}

const governanceSettingsCopy = {
  "execution-environment": {
    title: "执行环境",
    subtitle: "选择 Notebook 和工具运行使用的 Python 环境。",
  },
  llm: {
    title: "模型引擎",
    subtitle: "配置 Agent 会话可调用的大模型连接信息。",
  },
  "memory-policy": {
    title: "记忆",
    subtitle: "控制 Agent 记忆的引用范围、沉淀规则；展开下方可查看与管理记忆。",
  },
  plugins: {
    title: "插件",
    subtitle: "管理可调用工具包，启停插件并查看插件暴露的工具。",
    extensionTitle: "插件",
    extensionDescription: "管理可调用工具包，启停插件并查看插件暴露的工具。",
  },
  workflows: {
    title: "Workflow 模板",
    subtitle: "加载、校验和复用用户可编写的 Workflow 模板。",
    extensionTitle: "Workflow 模板",
    extensionDescription: "加载、校验和复用用户可编写的 Workflow 模板。",
  },
  capabilities: {
    title: "能力档位",
    subtitle: "选择 Agent 自治程度；证据、确认门和安全护栏保持不变。",
    extensionTitle: "能力档位",
    extensionDescription: "选择 Agent 自治程度；证据、确认门和安全护栏保持不变。",
  },
};

let activeGovernanceNav = "execution-environment";

function governanceNavButton(navKey) {
  return document.querySelector(`[data-governance-nav="${navKey}"]`);
}

function activeGovernanceButton(navKey = activeGovernanceNav) {
  return governanceNavButton(navKey) || governanceNavButton("execution-environment");
}

function setGovernanceCopy(navKey, button) {
  const copy = governanceSettingsCopy[navKey] || governanceSettingsCopy["execution-environment"];
  $("governanceSettingsTitle").textContent = copy.title;
  $("governanceSettingsSubtitle").textContent = copy.subtitle;
  if (button?.dataset?.extensionView) {
    $("governanceExtensionTitle").textContent = copy.extensionTitle || copy.title;
    $("governanceExtensionDescription").textContent = copy.extensionDescription || copy.subtitle;
  }
}

// Single, context-aware refresh for the dialog title bar. Only panels that load
// remote data appear here; execution-environment keeps its own 扫描环境 action.
const governanceRefreshActions = {
  plugins: () => runGovernanceExtensionAction(refreshGovernancePlugins),
  workflows: () => runGovernanceExtensionAction(refreshGovernanceSkills),
  capabilities: () => runGovernanceExtensionAction(refreshGovernanceCapability),
};

function syncGovernanceRefreshButton(navKey = activeGovernanceNav) {
  const button = $("governanceRefreshButton");
  if (!button) return;
  const unavailable = !governanceRefreshActions[navKey];
  button.classList.toggle("is-unavailable", unavailable);
  button.disabled = unavailable;
  button.setAttribute("aria-hidden", unavailable ? "true" : "false");
}

function refreshActiveGovernancePanel() {
  const action = governanceRefreshActions[activeGovernanceNav];
  if (!action) return;
  const button = $("governanceRefreshButton");
  if (button) {
    button.classList.add("is-spinning");
    window.setTimeout(() => button.classList.remove("is-spinning"), 700);
  }
  action();
}

function setGovernanceSettingsPanel(navKey = "execution-environment", options = {}) {
  const button = activeGovernanceButton(navKey);
  const normalizedNav = button?.dataset?.governanceNav || "execution-environment";
  const panel = button?.dataset?.governancePanel || "execution-environment";
  activeGovernanceNav = normalizedNav;
  syncGovernanceRefreshButton(normalizedNav);
  for (const item of document.querySelectorAll("[data-governance-nav]")) {
    const selected = item === button;
    item.classList.toggle("selected", selected);
    item.setAttribute("aria-selected", selected ? "true" : "false");
  }
  for (const section of document.querySelectorAll("[data-governance-panel-content]")) {
    section.classList.toggle("selected", section.dataset.governancePanelContent === panel);
  }
  const dialog = $("governanceSettingsDialog");
  dialog.dataset.governanceActive = normalizedNav;
  dialog.dataset.extensionView = button?.dataset?.extensionView || "";
  setGovernanceCopy(normalizedNav, button);
  if (panel === "extensions") {
    mountGovernanceExtensions();
    setGovernanceExtensionStatus("");
  }
}

function refreshGovernancePanel(navKey = activeGovernanceNav, options = {}) {
  const button = activeGovernanceButton(navKey);
  if (button?.dataset?.governancePanel === "execution-environment" && options.load !== false) {
    runAction(loadExecutionEnvironmentSettings, {
      actionId: "executionEnvironment",
      busyText: "正在读取执行环境...",
    });
  }
  if (button?.dataset?.governancePanel === "llm" && options.load !== false) {
    runAction(loadLLMSettings, { actionId: "llmSettings", busyText: "正在读取大模型配置..." });
  }
  if (button?.dataset?.governancePanel === "memory-policy" && options.load !== false) {
    runAction(loadMemoryPolicySettings, { actionId: "memoryPolicy", busyText: "正在读取记忆策略..." });
  }
}

function openGovernanceSettingsCenter(navKey = "execution-environment", options = {}) {
  closeSidebarSettingsMenu();
  setGovernanceSettingsPanel(navKey, { reloadMemory: false });
  const dialog = $("governanceSettingsDialog");
  if (!dialog.open) {
    dialog.showModal();
  }
  refreshGovernancePanel(navKey, options);
}

function closeGovernanceSettingsDialog() {
  $("governanceSettingsDialog").close();
}

function closeSidebarSettingsMenu() {
  const settings = $("sidebarSettings");
  if (!settings) return;
  settings.open = false;
}

let sidebarSettingsOpenFrame = 0;

function scheduleGovernanceSettingsFromSidebar() {
  if (sidebarSettingsOpenFrame || $("governanceSettingsDialog")?.open) return;
  sidebarSettingsOpenFrame = window.requestAnimationFrame(() => {
    sidebarSettingsOpenFrame = 0;
    openGovernanceSettingsCenter("execution-environment");
  });
}

function handleGovernanceSettingsNavClick(event) {
  const viewTab = event.target.closest("[data-agent-memory-view]");
  if (viewTab) {
    setAgentMemoryViewMode(viewTab.dataset.agentMemoryView, { reload: true });
    return;
  }
  const jump = event.target.closest("[data-governance-jump]");
  const navKey = jump
    ? jump.dataset.governanceJump
    : event.target.closest("[data-governance-nav]")?.dataset.governanceNav;
  if (!navKey) return;
  setGovernanceSettingsPanel(navKey, { reloadMemory: false });
  refreshGovernancePanel(navKey);
}

function handleGovernanceSettingsSearch(event) {
  const query = String(event.target.value || "").trim().toLowerCase();
  let visibleCount = 0;
  for (const item of document.querySelectorAll("[data-governance-nav]")) {
    const hidden = Boolean(query && !item.textContent.toLowerCase().includes(query));
    item.classList.toggle("hidden", hidden);
    if (!hidden) visibleCount += 1;
  }
  for (const group of document.querySelectorAll(".governance-nav-group")) {
    const visibleItems = group.querySelectorAll("[data-governance-nav]:not(.hidden)");
    group.classList.toggle("hidden", visibleItems.length === 0);
  }
  const empty = $("governanceSettingsNavEmpty");
  if (empty) empty.hidden = visibleCount !== 0;
}

function syncAgentMemoryViewControls() {
  agentMemoryPanel.syncViewControls();
}

function setAgentMemoryViewMode(mode, { reload = true } = {}) {
  agentMemoryPanel.setViewMode(mode, { reload });
}

function openWordPreviewDialog() {
  if (!selectedTaskId) return;
  const frame = $("wordPreviewFrame");
  const title = selectedTask ? reportTitleForTask(selectedTask) : "Word 报告预览";
  $("wordPreviewTitle").textContent = `${title} · Word 报告预览`;
  frame.src = `api/tasks/${selectedTaskId}/report/preview?t=${Date.now()}`;
  $("wordPreviewDialog").showModal();
  setActionStatus("Word 报告预览已打开。", "success");
}

function closeWordPreviewDialog() {
  $("wordPreviewDialog").close();
  $("wordPreviewFrame").src = "about:blank";
}

function setCssNumber(name, value) {
  document.documentElement.style.setProperty(name, `${Math.round(value)}px`);
}

function formControlFocusTarget(target) {
  return target?.closest?.("input, textarea, select") || null;
}

function installFormControlFocusRingGuard() {
  function handleFormControlPointerDown(event) {
    const control = formControlFocusTarget(event.target);
    lastPointerDownControl = control;
    lastPointerDownAt = performance.now();
    if (control) control.classList.remove("suppress-pointer-focus-ring");
  }

  function handleFormControlFocusIn(event) {
    const control = formControlFocusTarget(event.target);
    if (!control) return;
    const pointerFocusPending = performance.now() - lastPointerDownAt < 750;
    control.classList.toggle(
      "suppress-pointer-focus-ring",
      pointerFocusPending && lastPointerDownControl !== control
    );
    lastPointerDownControl = null;
    lastPointerDownAt = 0;
  }

  function handleFormControlFocusOut(event) {
    const control = formControlFocusTarget(event.target);
    if (control) control.classList.remove("suppress-pointer-focus-ring");
  }

  function handleFormControlLabelClick(event) {
    const clickedControl = formControlFocusTarget(event.target);
    if (clickedControl) {
      clickedControl.classList.remove("suppress-pointer-focus-ring");
      return;
    }
    const label = event.target.closest?.("label");
    if (!label) return;
    setTimeout(() => {
      const focused = formControlFocusTarget(document.activeElement);
      if (!focused) return;
      const labelTargetsFocusedControl =
        label.contains(focused) || Boolean(label.htmlFor && focused.id === label.htmlFor);
      if (labelTargetsFocusedControl) focused.classList.add("suppress-pointer-focus-ring");
    }, 0);
  }

  document.addEventListener("pointerdown", handleFormControlPointerDown, true);
  document.addEventListener("mousedown", handleFormControlPointerDown, true);
  document.addEventListener("touchstart", handleFormControlPointerDown, true);
  document.addEventListener("click", handleFormControlLabelClick, true);
  document.addEventListener("focusin", handleFormControlFocusIn, true);
  document.addEventListener("focusout", handleFormControlFocusOut, true);
}

function saveLayoutWidths() {
  try {
    localStorage.setItem(
      "marvis_layout",
      JSON.stringify({
        sidebar: parseInt(getComputedStyle(document.documentElement).getPropertyValue("--sidebar-width"), 10),
        progress: parseInt(getComputedStyle(document.documentElement).getPropertyValue("--progress-width"), 10),
      })
    );
  } catch (_) {
    // Layout persistence is optional in restricted notebook browsers.
  }
}

function restoreLayoutWidths() {
  try {
    const stored = JSON.parse(localStorage.getItem("marvis_layout") || "{}");
    if (stored.sidebar) {
      setCssNumber(
        "--sidebar-width",
        clamp(stored.sidebar === 320 ? SIDEBAR_WIDTH_MIN : stored.sidebar, SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX)
      );
    }
    if (stored.progress) setCssNumber("--progress-width", clamp(stored.progress, PROGRESS_WIDTH_MIN, PROGRESS_WIDTH_MAX));
  } catch (_) {
    // Keep CSS defaults when storage is unavailable or invalid.
  }
}

function applySidebarCollapsed(collapsed) {
  const shouldKeepPetOnLeftEdge = petIsPinnedToWorkspaceLeftEdge();
  sidebarCollapsed = Boolean(collapsed);
  const shell = $("appShell");
  // Keep expanded text laid out at the expanded width while the grid column slides away.
  const expandedWidth =
    parseInt(getComputedStyle(document.documentElement).getPropertyValue("--sidebar-width"), 10) || 314;
  document.documentElement.style.setProperty("--rail-content-width", `${expandedWidth}px`);
  if (document.body.classList.contains("anim-ready")) {
    document.body.classList.add("sidebar-sliding");
    clearTimeout(sidebarSlideTimer);
    sidebarSlideTimer = setTimeout(() => document.body.classList.remove("sidebar-sliding"), 340);
  }
  shell.classList.toggle("sidebar-collapsed", sidebarCollapsed);
  window.requestAnimationFrame(() => {
    if (shouldKeepPetOnLeftEdge) {
      pinPetToWorkspaceLeftEdge({ persist: true });
    } else {
      ensurePetWithinViewport({ persist: true });
    }
    if (document.body.classList.contains("anim-ready")) {
      window.setTimeout(() => {
        if (shouldKeepPetOnLeftEdge) {
          pinPetToWorkspaceLeftEdge({ persist: true });
        } else {
          ensurePetWithinViewport({ persist: true });
        }
      }, 340);
    }
  });
  const button = $("sidebarCollapseButton");
  button.setAttribute("aria-expanded", String(!sidebarCollapsed));
  button.setAttribute("aria-label", sidebarCollapsed ? "展开侧栏" : "收起侧栏");
  button.title = sidebarCollapsed ? "展开侧栏" : "收起侧栏";
  const brandTrigger = $("sidebarBrandTrigger");
  brandTrigger.classList.toggle("is-collapse-trigger", sidebarCollapsed);
  brandTrigger.tabIndex = sidebarCollapsed ? 0 : -1;
  if (sidebarCollapsed) {
    brandTrigger.setAttribute("role", "button");
    brandTrigger.setAttribute("aria-label", "展开侧栏");
    brandTrigger.title = "展开侧栏";
  } else {
    brandTrigger.removeAttribute("role");
    brandTrigger.removeAttribute("aria-label");
    brandTrigger.removeAttribute("title");
  }
}

function toggleSidebarCollapsed() {
  applySidebarCollapsed(!sidebarCollapsed);
  try {
    localStorage.setItem("sidebarCollapsed", sidebarCollapsed ? "1" : "0");
  } catch (_) {
    // Sidebar persistence is optional in restricted notebook browsers.
  }
}

function restoreSidebarCollapsed() {
  try {
    applySidebarCollapsed(localStorage.getItem("sidebarCollapsed") === "1");
  } catch (_) {
    applySidebarCollapsed(false);
  }
}

function expandSidebarFromBrand(event) {
  if (!sidebarCollapsed) return;
  event?.preventDefault();
  toggleSidebarCollapsed();
}

function handleSidebarBrandKeydown(event) {
  if (!sidebarCollapsed || !["Enter", " "].includes(event.key)) return;
  expandSidebarFromBrand(event);
}

function startResizeDrag(side, event) {
  event.preventDefault();
  const rootStyle = getComputedStyle(document.documentElement);
  const startX = event.clientX;
  const startSidebar = parseInt(rootStyle.getPropertyValue("--sidebar-width"), 10);
  const startProgress = parseInt(rootStyle.getPropertyValue("--progress-width"), 10);

  function onPointerMove(moveEvent) {
    const deltaX = moveEvent.clientX - startX;
    if (side === "left") {
      setCssNumber("--sidebar-width", clamp(startSidebar + deltaX, SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX));
    } else {
      setCssNumber("--progress-width", clamp(startProgress - deltaX, PROGRESS_WIDTH_MIN, PROGRESS_WIDTH_MAX));
    }
  }

  function onPointerUp() {
    document.body.classList.remove("is-resizing");
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    saveLayoutWidths();
  }

  document.body.classList.add("is-resizing");
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
}

function handleResizeKey(side, event) {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  const rootStyle = getComputedStyle(document.documentElement);
  const step = event.shiftKey ? 32 : 12;
  const direction = event.key === "ArrowRight" ? 1 : -1;
  if (side === "left") {
    const current = parseInt(rootStyle.getPropertyValue("--sidebar-width"), 10);
    setCssNumber("--sidebar-width", clamp(current + direction * step, SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX));
  } else {
    const current = parseInt(rootStyle.getPropertyValue("--progress-width"), 10);
    setCssNumber("--progress-width", clamp(current - direction * step, PROGRESS_WIDTH_MIN, PROGRESS_WIDTH_MAX));
  }
  saveLayoutWidths();
}

function openTaskSearch() {
  if (taskSearchActive) return;
  taskSearchActive = true;
  document.body.classList.add("search-active");
  $("taskSearchToggle").setAttribute("aria-expanded", "true");
  const input = $("taskSearchInput");
  window.requestAnimationFrame(() => {
    input.focus();
    input.select();
  });
}

function closeTaskSearch({ focusToggle = false } = {}) {
  if (!taskSearchActive) return;
  taskSearchActive = false;
  document.body.classList.remove("search-active");
  $("taskSearchToggle").setAttribute("aria-expanded", "false");
  const input = $("taskSearchInput");
  if (input.value) {
    input.value = "";
    taskSearchQuery = "";
    renderTaskList();
  }
  if (focusToggle) $("taskSearchToggle").focus();
}

function toggleTaskSearch() {
  if (taskSearchActive) {
    closeTaskSearch({ focusToggle: true });
  } else {
    openTaskSearch();
  }
}

function basePetMoodFromTask() {
  const status = selectedTask?.status || "";
  if (taskStopped(selectedTask)) return "idle";
  if (selectedTaskIsBusy()) return "running";
  if (status === "succeeded") return "success";
  if (status === "failed") return "failed";
  if (status === "review_required") return "review";
  if (["running", "computing_metrics"].includes(status)) return "running";
  if (["scanned", "executed", "writing_artifacts"].includes(status)) return "complete";
  return "idle";
}

function clearPetReactionTimer() {
  if (!petReactionTimer) return;
  clearTimeout(petReactionTimer);
  petReactionTimer = null;
}

function petReactionKeyForMood(mood) {
  if (!petReactionMoods.has(mood)) return "";
  const task = selectedTask;
  return [
    task?.id || "",
    task?.status || "",
    task?.updated_at || "",
    task?.status_message || "",
    mood,
  ].join("|");
}

function schedulePetReactionReset(key) {
  clearPetReactionTimer();
  petReactionTimer = setTimeout(() => {
    if (petReactionKey !== key) return;
    petReactionMood = null;
    renderPetState();
  }, PET_REACTION_DURATION_MS);
}

function petMoodFromTask() {
  const mood = basePetMoodFromTask();
  if (!petReactionMoods.has(mood)) {
    petReactionMood = null;
    petReactionKey = "";
    clearPetReactionTimer();
    return mood;
  }

  const key = petReactionKeyForMood(mood);
  if (petReactionKey !== key) {
    petReactionMood = mood;
    petReactionKey = key;
    schedulePetReactionReset(key);
  }
  return petReactionMood || "idle";
}

function normalizePetPreference(value) {
  if (value === "none") return "none";
  const normalized = legacyPetPreferences[value] || value;
  return petDefinitions[normalized] ? normalized : defaultPetPreference;
}

function persistPetPreference(value, explicitNone = false) {
  try {
    localStorage.setItem("marvis_pet", value);
    if (value === "none" && explicitNone) {
      localStorage.setItem(explicitPetNoneStorageKey, "1");
    } else {
      localStorage.removeItem(explicitPetNoneStorageKey);
    }
  } catch (_) {
    // Pet preference is optional in restricted notebook browsers.
  }
}

function applyPetPreference(value, options = {}) {
  const { persist = true, explicit = false } = options;
  const normalized = normalizePetPreference(value);
  petPreference = normalized;
  if ($("settingsPetSelect")) $("settingsPetSelect").value = petPreference;
  if (persist) {
    persistPetPreference(petPreference, explicit);
  }
  renderPetState();
  ensurePetWithinViewport({ persist });
}

function restorePetPreference() {
  try {
    const stored = localStorage.getItem("marvis_pet");
    const explicitNone = localStorage.getItem(explicitPetNoneStorageKey) === "1";
    if (!stored || (stored === "none" && !explicitNone)) {
      applyPetPreference(defaultPetPreference, { persist: stored === "none" });
      return;
    }
    if (stored === "none") {
      applyPetPreference("none", { persist: false });
      return;
    }
    const normalized = normalizePetPreference(stored);
    applyPetPreference(normalized, { persist: normalized !== stored });
  } catch (_) {
    applyPetPreference(defaultPetPreference, { persist: false });
  }
}

function applyPetPosition(left, top) {
  const pet = $("petCompanion");
  if (!pet) return;
  const workspace = $("validationWorkspace")?.getBoundingClientRect();
  const offsetLeft = workspace ? left - workspace.left : left;
  pet.style.setProperty("--pet-offset-left", `${Math.round(offsetLeft)}px`);
  pet.style.left = "";
  pet.style.top = `${Math.round(top)}px`;
  pet.style.right = "auto";
  pet.style.bottom = "auto";
}

function savePetPosition(left, top) {
  try {
    const workspace = $("validationWorkspace")?.getBoundingClientRect();
    const payload = { left, top };
    if (workspace) payload.workspaceOffsetLeft = left - workspace.left;
    localStorage.setItem("marvis_pet_position", JSON.stringify(payload));
  } catch (_) {
    // Drag position persistence is optional in restricted notebook browsers.
  }
}

function restorePetPosition() {
  try {
    const stored = JSON.parse(localStorage.getItem("marvis_pet_position") || "{}");
    const workspace = $("validationWorkspace")?.getBoundingClientRect();
    const storedLeft =
      workspace && Number.isFinite(stored.workspaceOffsetLeft)
        ? workspace.left + stored.workspaceOffsetLeft
        : stored.left;
    if (Number.isFinite(storedLeft) && Number.isFinite(stored.top)) {
      const next = clampPetPosition(storedLeft, stored.top);
      applyPetPosition(next.left, next.top);
      if (
        next.left !== stored.left ||
        next.top !== stored.top ||
        !Number.isFinite(stored.workspaceOffsetLeft)
      ) {
        savePetPosition(next.left, next.top);
      }
    }
  } catch (_) {
    // Keep the default fixed bottom-left workspace position.
  }
}

function petCssPx(name, fallback) {
  const pet = $("petCompanion");
  const host = pet || $("appShell") || document.documentElement;
  const value = getComputedStyle(host).getPropertyValue(name);
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function petDragBounds() {
  const pet = $("petCompanion");
  const workspace = $("validationWorkspace")?.getBoundingClientRect();
  const padding = 14;
  const minWorkspaceOffset = petCssPx("--pet-min-workspace-offset", padding);
  const minLeft = Math.max(
    padding,
    workspace ? workspace.left + minWorkspaceOffset : minWorkspaceOffset,
  );
  const minTop = Math.max(padding, workspace ? workspace.top + padding : padding);
  const maxLeft = Math.max(minLeft, window.innerWidth - (pet?.offsetWidth || 104) - padding);
  const maxTop = Math.max(minTop, window.innerHeight - (pet?.offsetHeight || 116) - padding);
  return { minLeft, minTop, maxLeft, maxTop };
}

function petWorkspaceOffset() {
  const pet = $("petCompanion");
  const workspace = $("validationWorkspace")?.getBoundingClientRect();
  if (!pet || !workspace) return null;
  return pet.getBoundingClientRect().left - workspace.left;
}

function petIsPinnedToWorkspaceLeftEdge() {
  const pet = $("petCompanion");
  if (!pet || pet.classList.contains("hidden")) return false;
  const offset = petWorkspaceOffset();
  if (!Number.isFinite(offset)) return false;
  const minWorkspaceOffset = petCssPx("--pet-min-workspace-offset", 14);
  return Math.abs(offset - minWorkspaceOffset) <= 2;
}

function pinPetToWorkspaceLeftEdge(options = {}) {
  const { persist = false } = options;
  const pet = $("petCompanion");
  const workspace = $("validationWorkspace")?.getBoundingClientRect();
  if (!pet || !workspace || pet.classList.contains("hidden")) return;
  const minWorkspaceOffset = petCssPx("--pet-min-workspace-offset", 14);
  const rect = pet.getBoundingClientRect();
  const next = clampPetPosition(workspace.left + minWorkspaceOffset, rect.top);
  applyPetPosition(next.left, next.top);
  if (persist) savePetPosition(next.left, next.top);
}

function clampPetPosition(left, top) {
  const bounds = petDragBounds();
  return {
    left: clamp(left, bounds.minLeft, bounds.maxLeft),
    top: clamp(top, bounds.minTop, bounds.maxTop),
  };
}

function ensurePetWithinViewport(options = {}) {
  const { persist = false } = options;
  const pet = $("petCompanion");
  if (!pet || pet.classList.contains("hidden")) return;
  const rect = pet.getBoundingClientRect();
  const next = clampPetPosition(rect.left, rect.top);
  if (Math.round(next.left) === Math.round(rect.left) && Math.round(next.top) === Math.round(rect.top)) return;
  applyPetPosition(next.left, next.top);
  if (persist) savePetPosition(next.left, next.top);
}

function renderPetState() {
  const pet = $("petCompanion");
  const sticker = $("petSticker");
  if (!pet || !sticker) return;

  const definition = petDefinitions[petPreference];
  pet.classList.toggle("hidden", !definition);
  pet.dataset.petId = petPreference;
  if (!definition) {
    sticker.replaceChildren();
    delete sticker.dataset.petId;
    pet.dataset.petMood = "idle";
    pet.setAttribute("aria-label", "宠物未显示");
    return;
  }

  if (sticker.dataset.petId !== petPreference) {
    sticker.replaceChildren();
    if (definition.kind === "spritesheet") {
      const sprite = document.createElement("div");
      sprite.className = "pet-sprite";
      sprite.style.backgroundImage = `url("${definition.asset}")`;
      sprite.setAttribute("aria-hidden", "true");
      sticker.appendChild(sprite);
    } else {
      const image = document.createElement("img");
      image.className = "pet-image";
      image.src = definition.asset;
      image.alt = "";
      image.decoding = "async";
      image.draggable = false;
      sticker.appendChild(image);
    }
    sticker.dataset.petId = petPreference;
  }
  const mood = petMoodFromTask();
  pet.dataset.petMood = mood;
  pet.setAttribute("aria-label", `${definition.name}，${definition.label}，当前状态：${mood}`);
  ensurePetWithinViewport({ persist: false });
}

function startPetDrag(event) {
  const pet = $("petCompanion");
  if (!pet || pet.classList.contains("hidden")) return;
  if (event.button !== undefined && event.button !== 0) return;
  event.preventDefault();
  const rect = pet.getBoundingClientRect();
  petDragState = {
    offsetX: event.clientX - rect.left,
    offsetY: event.clientY - rect.top,
  };
  pet.classList.add("dragging");

  function onPointerMove(moveEvent) {
    if (!petDragState) return;
    const next = clampPetPosition(moveEvent.clientX - petDragState.offsetX, moveEvent.clientY - petDragState.offsetY);
    const current = pet.getBoundingClientRect();
    pet.dataset.petMood = next.left >= current.left ? "running-right" : "running-left";
    applyPetPosition(next.left, next.top);
  }

  function onPointerUp() {
    const current = pet.getBoundingClientRect();
    petDragState = null;
    pet.classList.remove("dragging");
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    savePetPosition(current.left, current.top);
    renderPetState();
  }

  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
}

function renderSettingsState() {
  if ($("settingsSortSelect")) $("settingsSortSelect").value = taskSortMode;
  if ($("settingsGroupSelect")) $("settingsGroupSelect").value = taskGroupMode;
  if ($("settingsThemeSelect")) $("settingsThemeSelect").value = themeController.preference;
  if ($("settingsPetSelect")) $("settingsPetSelect").value = petPreference;
  renderExecutionEnvironmentSummary();
  renderLLMSettingsSummary();
}

function normalizeTaskSortMode(value) {
  return taskSortModes.has(value) ? value : "created_desc";
}

function normalizeTaskGroupMode(value) {
  return taskGroupModes.has(value) ? value : "none";
}

function saveTaskListSettings() {
  try {
    localStorage.setItem("marvis_task_list_settings", JSON.stringify({
      sort: taskSortMode,
      group: taskGroupMode,
    }));
  } catch (_) {
    // Sidebar list preferences are optional in restricted notebook browsers.
  }
}

function restoreTaskListSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem("marvis_task_list_settings") || "{}");
    taskSortMode = normalizeTaskSortMode(stored.sort);
    taskGroupMode = normalizeTaskGroupMode(stored.group);
  } catch (_) {
    taskSortMode = "created_desc";
    taskGroupMode = "none";
  }
}

function handleSettingsMenuChange(event) {
  const target = event.target;
  if (target.id === "settingsSortSelect") {
    taskSortMode = normalizeTaskSortMode(target.value);
    saveTaskListSettings();
    renderTaskList();
    renderSettingsState();
    return;
  }
  if (target.id === "settingsGroupSelect") {
    taskGroupMode = normalizeTaskGroupMode(target.value);
    saveTaskListSettings();
    renderTaskList();
    renderSettingsState();
    return;
  }
  if (target.id === "settingsThemeSelect") {
    themeController.applyTheme(target.value);
  }
  if (target.id === "settingsPetSelect") {
    applyPetPreference(target.value, { explicit: true });
  }
}

function taskDisplayName(task) {
  if (!task) return "";
  const name = String(task.model_name || "").trim();
  const version = String(task.model_version || "").trim();
  return version ? `${name} · ${version}` : name;
}

function reportTitleForTask(task) {
  const displayName = taskDisplayName(task);
  return displayName ? `${displayName}模型验证文档` : "未选择任务";
}

function setCreateStatus(message, kind = "info") {
  createTaskDialog.setCreateStatus(message, kind);
}

function setExecutionEnvironmentStatus(message, kind = "info") {
  const status = $("executionEnvironmentStatus");
  status.textContent = message;
  status.className = `status ${kind}`;
}

function actionStatusPill(message, kind) {
  if (!message) return null;
  if (kind === "error") {
    return /复核/.test(message)
      ? { label: "需复核", tone: "ok" }
      : { label: "验证失败", tone: "fail" };
  }
  if (kind === "stopped") return { label: "停止", tone: "neutral" };
  if (kind === "busy") return { label: "进行中", tone: "run" };
  if (kind === "success") return { label: "已完成", tone: "ok" };
  return { label: "待处理", tone: "neutral" };
}

function describeActionStatus(message, kind, detail) {
  if (!message) return "";
  if (kind === "error" && detail && detail !== message) return `${message} · ${detail}`;
  return message;
}

function setActionErrorDetail(message = "", kind = "info") {
  const detail = $("actionErrorDetail");
  if (!detail) return;
  detail.textContent = message || "";
  detail.setAttribute("role", kind === "error" ? "alert" : "status");
  detail.setAttribute("aria-live", kind === "error" ? "assertive" : "polite");
  detail.className = `action-error-detail ${kind === "error" ? "error" : ""}`.trim();
}

function setActionStatus(message, kind = "info", detail = "") {
  const nextSignature = signatureFromParts([message || "", kind || "info", detail || ""]);
  if (renderSignatures.actionStatus === nextSignature) return;
  renderSignatures.actionStatus = nextSignature;

  const pill = $("actionStatus");
  const info = actionStatusPill(message, kind);
  if (pill) {
    pill.textContent = info ? info.label : "";
    pill.className = `task-pill ${info ? info.tone : ""}`.trim();
    const hero = pill.closest(".task-hero");
    if (hero) hero.dataset.tone = info ? info.tone : "";
  }
  setActionErrorDetail(describeActionStatus(message, kind, detail), kind);
  requestAnimationFrame(syncTaskHeroGlassLayout);
}

function setActionStatusOverride(message, kind = "info", detail = "") {
  if (!selectedTaskId) {
    setActionStatus(message, kind, detail);
    return;
  }
  actionStatusOverride = { taskId: selectedTaskId, message, kind, detail };
  setActionStatus(message, kind, detail);
}

function clearActionStatusOverride(taskId = selectedTaskId) {
  if (!actionStatusOverride) return;
  if (!taskId || actionStatusOverride.taskId === taskId) actionStatusOverride = null;
}

function taskFailureActionStatusMessage(task = selectedTask) {
  if (!task || !["failed", "review_required"].includes(task.status)) return "";
  if (task.status === "review_required") return "全部流程已完成，请查看右侧报告并进行人工复核。";
  if (task.status_message) return task.status_message;
  return "验证失败。";
}

function taskStoppedActionStatusMessage(task = selectedTask) {
  if (!taskStopped(task)) return "";
  return "已停止当前动作，请问有什么指示？";
}

function taskFailedDuringScan(task = selectedTask) {
  return task?.status === "failed" && normalizedFailureStage(task.failure_stage) === "scan";
}

function taskFailureWasRestartReclaim(task = selectedTask) {
  return task?.status === "failed" && task?.failure_reason_code === "server_restart_while_running";
}

function normalizedFailureStage(stage) {
  const value = String(stage || "");
  return ["scan", "notebook", "metrics", "report"].includes(value) ? value : null;
}

function taskFailureStage(task = selectedTask) {
  if (!task || task.status !== "failed") return null;
  const structuredStage = normalizedFailureStage(task.failure_stage);
  if (structuredStage) return structuredStage;
  return null;
}

function taskFailedDuringMetrics(task = selectedTask) {
  return taskFailureStage(task) === "metrics";
}

function taskFailedDuringReport(task = selectedTask) {
  return taskFailureStage(task) === "report";
}

function taskFailedDuringNotebook(task = selectedTask) {
  return taskFailureStage(task) === "notebook";
}

function taskFailureActionStatusTitle(task = selectedTask) {
  if (!task || !["failed", "review_required"].includes(task.status)) return "";
  if (task.status === "review_required") return "验证已完成，需复核报告。";
  const stage = taskFailureStage(task);
  if (stage === "scan") return "材料识别失败。";
  if (stage === "metrics") return "模型效果&稳定性验证失败。";
  if (stage === "report") return "报告输出失败。";
  if (stage === "notebook") return "模型可复现性验证失败。";
  return "任务执行失败。";
}

function taskStoppedActionStatusTitle(task = selectedTask) {
  if (!taskStopped(task)) return "";
  return "已停止当前动作。";
}

function setTaskFailureActionStatus(task = selectedTask) {
  if (taskStopped(task)) {
    setActionStatus(
      taskStoppedActionStatusTitle(task),
      "stopped",
      taskStoppedActionStatusMessage(task),
    );
    return true;
  }
  const message = taskFailureActionStatusMessage(task);
  if (!message) {
    setActionErrorDetail("");
    return false;
  }
  const kind = task.status === "review_required" ? "success" : "error";
  setActionStatus(taskFailureActionStatusTitle(task), kind, message);
  return true;
}

function actionFailureStatusTitle(actionId) {
  switch (actionId) {
    case "agent":
      return "Agent 执行失败。";
    case "scan":
      return "材料识别失败。";
    case "notebook":
      return "模型可复现性验证失败。";
    case "metrics":
      return "指标概览失败。";
    case "report":
      return "报告输出失败。";
    case "delete":
      return "任务删除失败。";
    default:
      return "操作失败。";
  }
}

function actionCancelledStatusTitle(actionId) {
  switch (actionId) {
    case "scan":
      return "材料扫描已停止。";
    case "notebook":
    case "cancelNotebook":
      return "Notebook 已停止，可重新运行。";
    case "metrics":
    case "cancelMetrics":
      return "指标生成已停止，可重新生成。";
    case "report":
    case "cancelReport":
      return "报告生成已停止，可重新生成。";
    default:
      return "操作已停止。";
  }
}

function taskActionStatusSnapshot(task = selectedTask) {
  if (!task) return { message: "", kind: "info" };
  if (taskStopped(task)) return { message: "已停止当前动作。", kind: "stopped" };
  switch (task.status) {
    case "created":
      return { message: "任务已创建。", kind: "info" };
    case "scanned":
    case "configured":
      return { message: "材料识别完成。", kind: "success" };
    case "running":
      return { message: "模型可复现性验证进行中。", kind: "busy" };
    case "executed":
      return { message: "模型可复现性验证完成。", kind: "success" };
    case "computing_metrics":
      return { message: "指标概览进行中。", kind: "busy" };
    case "writing_artifacts":
      // writing_artifacts is dual-meaning: backend flips here the moment
      // metrics finishes (idle, awaiting "生成 Word") and stays here while
      // the report job actually runs. Only the second case is in-progress.
      if (task.active_job_kind === "report") {
        return { message: "报告输出进行中。", kind: "busy" };
      }
      return { message: "模型效果&稳定性验证完成。", kind: "success" };
    case "succeeded":
      return { message: "验证完成。", kind: "success" };
    case "review_required":
      return { message: "验证完成，需人工复核。", kind: "success" };
    default:
      return { message: "", kind: "info" };
  }
}

function clearStatus() {
  setCreateStatus("");
  setActionStatus("");
}

function statusLabel(status) {
  return statusLabels[status] || status || "未知";
}

function taskStatusLabel(task) {
  if (taskStopped(task)) return "停止";
  return statusLabel(task?.status);
}

function statusTone(status) {
  if (status === "failed") return "danger";
  if (status === "review_required") return "success";
  if (status === "succeeded" || status === "executed") return "success";
  if (status === "running" || status === "computing_metrics") return "run";
  return "";
}

function taskStatusTone(task) {
  if (taskStopped(task)) return "";
  if (task?.status === "writing_artifacts") {
    return task.active_job_kind === "report" ? "run" : "success";
  }
  return statusTone(task?.status);
}

function notebookReproducibilityComplete(task = selectedTask) {
  return (
    notebookReproducibilityCompleteStatuses.has(task?.status || "") ||
    taskFailedDuringMetrics(task) ||
    taskFailedDuringReport(task) ||
    (taskFailureWasRestartReclaim(task) && workflowStageCompleteFromEvidence("notebook"))
  );
}

function shouldShowReproducibilitySection() {
  return Boolean(selectedTaskId && notebookReproducibilityComplete(selectedTask));
}

function renderReproducibilitySectionVisibility() {
  // Driver tasks (data_join / feature / modeling) have no validation notebook
  // section — they run through the conversation + plan rail.
  if (taskUsesPlanRail(selectedTask)) {
    $("notebookSection")?.classList.add("hidden");
    return;
  }
  $("notebookSection")?.classList.toggle("hidden", !shouldShowReproducibilitySection());
}

function metricOverviewComplete(task = selectedTask) {
  return (
    metricOverviewCompleteStatuses.has(task?.status || "") ||
    taskFailedDuringReport(task) ||
    (taskFailureWasRestartReclaim(task) && workflowStageCompleteFromEvidence("metrics"))
  );
}

function shouldShowMetricSection() {
  return Boolean(selectedTaskId && metricOverviewComplete(selectedTask));
}

function renderMetricSectionVisibility() {
  // Driver tasks render metrics inline in the conversation, not in the validation
  // metric section.
  if (taskUsesPlanRail(selectedTask)) {
    $("metricSection")?.classList.add("hidden");
    return;
  }
  $("metricSection")?.classList.toggle("hidden", !shouldShowMetricSection());
}

function workflowIndex(status) {
  if (!selectedTaskId) return -1;
  if (taskFailedDuringScan(selectedTask)) return 0;
  if (taskFailedDuringMetrics(selectedTask)) return 2;
  if (taskFailedDuringReport(selectedTask)) return 3;
  if (status === "succeeded" || status === "review_required") return 3;
  if (status === "writing_artifacts") return 3;
  if (status === "computing_metrics" || status === "executed") return 2;
  if (status === "running" || status === "failed" || status === "scanned" || status === "configured") return 1;
  return 0;
}

function taskFailureStepId(task = selectedTask) {
  return taskFailureStage(task);
}

function taskRunningStepId(status = selectedTask?.status) {
  const selectedBusyAction = taskBusyAction();
  if (selectedBusyAction === "scan") return "scan";
  if (selectedBusyAction === "notebook" || selectedBusyAction === "cancelNotebook") return "notebook";
  if (selectedBusyAction === "metrics" || selectedBusyAction === "cancelMetrics") return "metrics";
  if (selectedBusyAction === "report" || selectedBusyAction === "cancelReport") return "report";
  if (status === "running") return "notebook";
  if (status === "computing_metrics") return "metrics";
  return null;
}

function recommendedAction() {
  if (!selectedTaskId || selectedTaskIsBusy()) return null;
  const status = selectedTask?.status;
  if (status === "created" || taskFailedDuringScan(selectedTask)) return "scan";
  if (taskFailedDuringMetrics(selectedTask)) return "metrics";
  if (taskFailedDuringReport(selectedTask)) return "report";
  if (taskFailedDuringNotebook(selectedTask)) return "notebook";
  if (status === "scanned" || status === "configured") {
    return "notebook";
  }
  if (status === "executed") return "metrics";
  if (status === "writing_artifacts") return "report";
  return null;
}

function canRunStepAction(actionId) {
  if (!selectedTaskId) return false;
  const status = selectedTask?.status;
  if (actionId === "notebook" && status === "running") return true;
  switch (actionId) {
    case "scan":
      return ["created", "scanned", "failed", "executed", "writing_artifacts", "succeeded", "review_required"].includes(status);
    case "notebook":
      if (taskFailedDuringScan(selectedTask)) return false;
      return ["scanned", "configured", "executed", "writing_artifacts", "succeeded", "review_required"].includes(status) || taskFailedDuringNotebook(selectedTask);
    case "metrics":
      return status === "executed" || taskFailedDuringMetrics(selectedTask);
    case "report":
      return ["writing_artifacts", "review_required"].includes(status) || taskFailedDuringReport(selectedTask);
    default:
      return false;
  }
}

function setBusy(actionId, message = "", taskId = selectedTaskId) {
  if (taskId) {
    if (actionId) taskBusyActions.set(taskId, actionId);
    else taskBusyActions.delete(taskId);
  } else {
    globalBusyAction = actionId;
  }
  if (actionId && (!taskId || selectedTaskId === taskId)) {
    setActionStatus(message || "正在处理...", "busy");
  }
  renderWorkflowStepper();
  renderPetState();
  updateAgentSendDisabled();
}

function setAgentMemoryStatus(message = "", kind = "") {
  agentMemoryPanel.setStatus(message, kind);
}

function setGovernanceExtensionStatus(message = "", kind = "") {
  const status = $("governanceExtensionStatus");
  if (!status) return;
  status.textContent = message;
  status.className = ["status", kind].filter(Boolean).join(" ");
}

function governanceExtensionActions() {
  const showExtensionError = (message) => {
    setGovernanceExtensionStatus(message || "操作失败", "error");
  };
  return {
    pluginActions: {
      showError: showExtensionError,
      confirmRemove: (name) => showPlatformConfirm({
        title: "移除插件",
        message: `确定移除插件「${name}」？移除后该插件提供的工具将不可用。`,
        confirmText: "移除",
        cancelText: "取消",
        tone: "danger",
      }),
    },
    skillActions: {
      showError: showExtensionError,
    },
    capabilityActions: {
      showError: showExtensionError,
    },
  };
}

function mountGovernanceExtensions() {
  const root = $("governanceExtensionMount");
  return root ? mountGovernanceExtensionPanels(root, governanceExtensionActions()) : null;
}

async function refreshGovernancePlugins() {
  const mounted = mountGovernanceExtensions();
  if (!mounted) return;
  const actions = governanceExtensionActions();
  setGovernanceExtensionStatus("正在读取插件...");
  await renderPluginManager(mounted.panels.pluginPanel, actions.pluginActions);
  setGovernanceExtensionStatus("插件已更新。", "success");
}

async function refreshGovernanceSkills() {
  const mounted = mountGovernanceExtensions();
  if (!mounted) return;
  const actions = governanceExtensionActions();
  setGovernanceExtensionStatus("正在读取 Workflow 模板...");
  await renderSkillManager(mounted.panels.skillPanel, actions.skillActions);
  setGovernanceExtensionStatus("Workflow 模板已更新。", "success");
}

async function refreshGovernanceCapability() {
  const mounted = mountGovernanceExtensions();
  if (!mounted) return;
  const actions = governanceExtensionActions();
  setGovernanceExtensionStatus("正在读取能力档位...");
  await renderTierSettings(mounted.panels.capabilityPanel, actions.capabilityActions);
  setGovernanceExtensionStatus("能力档位已更新。", "success");
}

function runGovernanceExtensionAction(action) {
  action().catch((error) => {
    setGovernanceExtensionStatus(error?.message || "扩展设置操作失败", "error");
  });
}

function renderAgentMemoryItems() {
  agentMemoryPanel.renderItems();
}

function renderAgentMemoryDetail(memory = null, events = [], detailOptions = {}) {
  agentMemoryPanel.renderDetail(memory, events, detailOptions);
}

async function loadAgentMemoryItems() {
  return agentMemoryPanel.loadItems();
}

async function inspectAgentMemory(memoryId) {
  return agentMemoryPanel.inspect(memoryId);
}

async function disableAgentMemory(memoryId) {
  return agentMemoryPanel.disable(memoryId);
}

async function enableAgentMemory(memoryId) {
  return agentMemoryPanel.enable(memoryId);
}

async function deleteAgentMemory(memoryId) {
  return agentMemoryPanel.remove(memoryId);
}

async function rollbackAgentMemoryDistillation(memoryId) {
  return agentMemoryPanel.rollbackDistillation(memoryId);
}

async function loadAgentMessageMemoryReferences(taskId, messageId) {
  if (!taskId || !messageId) return [];
  const payload = await api(`api/tasks/${encodeURIComponent(taskId)}/agent/messages/${encodeURIComponent(messageId)}/memory-references`);
  return Array.isArray(payload?.memory_references) ? payload.memory_references : [];
}

function handleAgentMemoryListClick(event) {
  agentMemoryPanel.handleListClick(event);
}

function handleAgentMemoryInlineInspect(event) {
  agentMemoryPanel.handleInlineInspect(event);
}

function setDraftToolsStatus(message = "", kind = "") {
  draftToolsPanel.setStatus(message, kind);
}

function renderDraftToolsList() {
  draftToolsPanel.renderList();
}

function renderDraftToolDetail(payload = null) {
  draftToolsPanel.renderDetail(payload);
}

async function loadDraftTools({ preserveSelection = false } = {}) {
  return draftToolsPanel.load({ preserveSelection });
}

async function inspectDraftTool(draftId) {
  return draftToolsPanel.inspect(draftId);
}

async function runDraftTool() {
  return draftToolsPanel.run();
}

async function promoteDraftTool() {
  return draftToolsPanel.promote();
}

async function rejectDraftTool() {
  return draftToolsPanel.reject();
}

function handleDraftToolsListClick(event) {
  draftToolsPanel.handleListClick(event);
}

function handleDraftToolsListKeydown(event) {
  draftToolsPanel.handleListKeydown(event);
}

function requireTaskId(taskId, actionName = "当前操作") {
  const normalizedTaskId = String(taskId || "").trim();
  if (!normalizedTaskId) {
    throw new Error(`${actionName}缺少任务 ID，请刷新任务列表后重试。`);
  }
  return normalizedTaskId;
}

function normalizeExecutionEnvironment(settings = {}) {
  return {
    ...defaultExecutionEnvironment,
    ...(settings || {}),
  };
}

function executionEnvironmentSettingsFromOption(option = {}) {
  return {
    execution_mode: option.execution_mode || "jupyter_kernel",
    kernel_name: option.kernel_name || "",
    conda_env_name: option.conda_env_name || "",
    python_executable: option.python_executable || "",
  };
}

function executionEnvironmentSettingsMatch(option = {}, settings = {}) {
  const normalized = normalizeExecutionEnvironment(settings);
  if ((option.execution_mode || "") !== normalized.execution_mode) return false;
  if ((option.kernel_name || "") !== (normalized.kernel_name || "")) return false;
  if (normalized.execution_mode === "conda_env") {
    return (option.conda_env_name || "") === (normalized.conda_env_name || "");
  }
  if (normalized.execution_mode === "python_executable") {
    return (option.python_executable || "") === (normalized.python_executable || "");
  }
  return true;
}

function executionEnvironmentSettingsLabel(settings = executionEnvironmentSettings, options = executionEnvironmentOptions) {
  const normalized = normalizeExecutionEnvironment(settings);
  const matchedOption = (options || []).find((option) => executionEnvironmentSettingsMatch(option, normalized));
  if (matchedOption?.label) return matchedOption.label;
  if (normalized.execution_mode === "conda_env") {
    return `Conda · ${normalized.conda_env_name || "未选择"}`;
  }
  if (normalized.execution_mode === "python_executable") {
    return `Python · ${fileName(normalized.python_executable) || "未选择"}`;
  }
  return `Jupyter Kernel · ${normalized.kernel_name || "python3"}`;
}

function renderExecutionEnvironmentSummary() {
  const label = executionEnvironmentSettingsLabel();
  const systemButton = $("openGovernanceSettingsButton");
  if (systemButton) systemButton.title = `打开系统设置，当前执行环境：${label}`;
}

function addExecutionEnvironmentRow(list, option, selected) {
  const settings = executionEnvironmentSettingsFromOption(option);
  const unavailable = option.available === false;
  const row = document.createElement("button");
  row.type = "button";
  row.className = "exec-env-row" + (selected ? " selected" : "");
  row.setAttribute("role", "radio");
  row.setAttribute("aria-checked", selected ? "true" : "false");
  row.tabIndex = -1; // roving tabindex; the active row is promoted after render
  row.disabled = unavailable;
  row.dataset.settings = JSON.stringify(settings);
  const title = option.label || option.id || "未命名环境";
  const subParts = [];
  if (option.note) subParts.push(option.note);
  if (unavailable) subParts.push("不可用");
  const sub = subParts.join(" · ");
  row.innerHTML =
    '<span class="exec-env-check" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M5 12.5l4.2 4.2L19 7"></path></svg></span>' +
    '<span class="exec-env-row-text">' +
    `<span class="exec-env-row-title">${escapeHtml(title)}</span>` +
    (sub ? `<span class="exec-env-row-sub">${escapeHtml(sub)}</span>` : "") +
    "</span>";
  list.appendChild(row);
}

function renderExecutionEnvironmentOptions(options = [], settings = {}) {
  executionEnvironmentOptions = Array.isArray(options) ? options : [];
  const list = $("executionEnvironmentList");
  if (!list) return;
  const normalized = normalizeExecutionEnvironment(settings);
  list.innerHTML = "";

  const rows = [];
  let selected = false;
  for (const option of executionEnvironmentOptions) {
    // Only the first match is marked selected, so at most one row is checked.
    const matches = !selected && executionEnvironmentSettingsMatch(option, normalized);
    rows.push({ option, selected: matches });
    selected = selected || matches;
  }

  if (!selected && (normalized.kernel_name || normalized.conda_env_name || normalized.python_executable)) {
    rows.push({
      option: {
        id: "saved-current",
        label: "当前保存配置",
        ...normalized,
        note: "未在本次扫描结果中匹配",
        available: true,
      },
      selected: true,
    });
    selected = true;
  }

  if (rows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "exec-env-empty";
    empty.textContent = "未扫描到可用 Python 环境";
    list.appendChild(empty);
    return;
  }

  if (!selected) {
    const firstAvailable = rows.find((row) => row.option.available !== false);
    if (firstAvailable) firstAvailable.selected = true;
  }

  for (const row of rows) addExecutionEnvironmentRow(list, row.option, row.selected);

  // Promote one row to the group's tab stop (roving tabindex): the selected
  // row, else the first selectable row.
  const focusTarget =
    list.querySelector(".exec-env-row.selected:not(:disabled)") ||
    list.querySelector(".exec-env-row:not(:disabled)");
  if (focusTarget) focusTarget.tabIndex = 0;
}

function handleExecutionEnvironmentListKeydown(event) {
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
  const rows = [...$("executionEnvironmentList").querySelectorAll(".exec-env-row:not(:disabled)")];
  if (!rows.length) return;
  event.preventDefault();
  const current = event.target.closest(".exec-env-row");
  let idx = rows.indexOf(current);
  if (event.key === "Home") idx = 0;
  else if (event.key === "End") idx = rows.length - 1;
  else if (event.key === "ArrowDown") idx = idx < 0 ? 0 : (idx + 1) % rows.length;
  else idx = idx < 0 ? rows.length - 1 : (idx - 1 + rows.length) % rows.length;
  const next = rows[idx];
  for (const row of rows) row.tabIndex = row === next ? 0 : -1;
  next.focus();
}

function populateExecutionEnvironmentForm(settings = {}, options = executionEnvironmentOptions) {
  const normalized = normalizeExecutionEnvironment(settings);
  executionEnvironmentSettings = normalized;
  renderExecutionEnvironmentOptions(options, normalized);
  renderExecutionEnvironmentSummary();
}

function handleExecutionEnvironmentListClick(event) {
  const row = event.target.closest(".exec-env-row");
  if (!row || row.disabled) return;
  let settings;
  try {
    settings = { ...defaultExecutionEnvironment, ...JSON.parse(row.dataset.settings || "{}") };
  } catch (_) {
    setExecutionEnvironmentStatus("执行环境配置解析失败，请重新扫描后再选择。", "error");
    return;
  }
  // Optimistically move the checkmark; saveExecutionEnvironmentSettings reverts on failure.
  for (const item of $("executionEnvironmentList").querySelectorAll(".exec-env-row")) {
    const on = item === row;
    item.classList.toggle("selected", on);
    item.setAttribute("aria-checked", on ? "true" : "false");
  }
  saveExecutionEnvironmentSettings(settings);
}

function renderExecutionEnvironmentValidation(validation = {}) {
  if (!validation || Object.keys(validation).length === 0) return;
  const parts = [
    validation.message,
    validation.kernel_name ? `Kernel: ${validation.kernel_name}` : "",
    validation.python_version ? `Python: ${validation.python_version}` : "",
  ].filter(Boolean);
  setExecutionEnvironmentStatus(parts.join(" · ") || "执行环境已保存。", validation.ok === false ? "error" : "success");
}

async function loadExecutionEnvironmentSettings({ silent = false } = {}) {
  try {
    const payload = await api("/api/settings/execution-environment/options");
    populateExecutionEnvironmentForm(payload.settings, payload.options || []);
    if (!silent) {
      renderExecutionEnvironmentValidation(payload.validation);
      if (!payload.validation) setExecutionEnvironmentStatus("执行环境已加载。", "success");
    }
  } catch (error) {
    populateExecutionEnvironmentForm(defaultExecutionEnvironment, []);
    if (!silent) setExecutionEnvironmentStatus(error.message || "执行环境读取失败。", "error");
  }
}

async function refreshExecutionEnvironmentOptions() {
  setExecutionEnvironmentStatus("正在扫描 Python 环境...");
  await loadExecutionEnvironmentSettings();
}

async function saveExecutionEnvironmentSettings(settings) {
  const list = $("executionEnvironmentList");
  try {
    setExecutionEnvironmentStatus("正在验证并保存环境...");
    if (list) list.classList.add("is-saving");
    const payload = await api("/api/settings/execution-environment", {
      method: "PUT",
      body: JSON.stringify({ ...defaultExecutionEnvironment, ...(settings || {}) }),
    });
    populateExecutionEnvironmentForm(payload.settings, executionEnvironmentOptions);
    renderExecutionEnvironmentValidation(payload.validation);
    if (!payload.validation) setExecutionEnvironmentStatus("执行环境已保存。", "success");
  } catch (error) {
    // Revert the optimistic checkmark to the last known-good selection.
    populateExecutionEnvironmentForm(executionEnvironmentSettings, executionEnvironmentOptions);
    setExecutionEnvironmentStatus(error.message || "执行环境保存失败。", "error");
  } finally {
    if (list) list.classList.remove("is-saving");
  }
}

function setLLMSettingsStatus(message, kind = "info") {
  const status = $("llmSettingsStatus");
  if (!status) return;
  status.textContent = message;
  status.className = `status ${kind}`.trim();
}

function setLLMEngineEditStatus(message, kind = "info") {
  const status = $("llmEngineEditStatus");
  if (!status) return;
  status.textContent = message;
  status.className = `status ${kind}`.trim();
}

function normalizeLLMSettings(payload = {}) {
  return {
    default_model_id: payload.default_model_id || "",
    models: Array.isArray(payload.models) ? payload.models : [],
    enabled_models: Array.isArray(payload.enabled_models) ? payload.enabled_models : [],
  };
}

function normalizeAgentEffort(value) {
  return ["low", "medium", "high"].includes(value) ? value : "high";
}

function normalizeAgentAcceptanceMode(value) {
  return value === "auto_accept" ? "auto_accept" : "normal";
}

function restoreAgentComposerPreferences() {
  try {
    const stored = JSON.parse(localStorage.getItem(agentComposerPreferenceStorageKey) || "{}");
    return {
      model_id: typeof stored.model_id === "string" ? stored.model_id : "",
      effort: normalizeAgentEffort(stored.effort),
      acceptance_mode: normalizeAgentAcceptanceMode(stored.acceptance_mode),
    };
  } catch (_) {
    return { model_id: "", effort: "high", acceptance_mode: "normal" };
  }
}

function saveAgentComposerPreferences() {
  localStorage.setItem(agentComposerPreferenceStorageKey, JSON.stringify({
    model_id: agentSelectedModelId || "",
    effort: normalizeAgentEffort(agentSelectedEffort),
    acceptance_mode: normalizeAgentAcceptanceMode(agentAcceptanceMode),
  }));
}

function loadAgentTaskComposerOverrides() {
  try {
    const stored = JSON.parse(localStorage.getItem(agentTaskComposerStorageKey) || "{}");
    if (stored && typeof stored === "object" && !Array.isArray(stored)) return stored;
  } catch (_) {
    /* fall through */
  }
  return {};
}

let agentTaskComposerOverrides = loadAgentTaskComposerOverrides();

function persistAgentTaskComposerOverrides() {
  try {
    localStorage.setItem(
      agentTaskComposerStorageKey,
      JSON.stringify(agentTaskComposerOverrides),
    );
  } catch (_) {
    /* swallow quota errors */
  }
}

function getAgentTaskComposerOverride(taskId) {
  if (!taskId) return null;
  const entry = agentTaskComposerOverrides[taskId];
  if (!entry || typeof entry !== "object") return null;
  const result = {};
  if (typeof entry.model_id === "string") result.model_id = entry.model_id;
  if (entry.effort) result.effort = normalizeAgentEffort(entry.effort);
  if (entry.acceptance_mode) {
    result.acceptance_mode = normalizeAgentAcceptanceMode(entry.acceptance_mode);
  }
  return result;
}

function updateAgentTaskComposerOverride(taskId, patch) {
  if (!taskId || !patch) return;
  // Re-read from localStorage so a sibling tab's overrides for OTHER tasks
  // are not silently clobbered. We still own the entry for `taskId`.
  const latest = loadAgentTaskComposerOverrides();
  const current = latest[taskId] || {};
  agentTaskComposerOverrides = { ...latest, [taskId]: { ...current, ...patch } };
  persistAgentTaskComposerOverrides();
}

function applyAgentTaskComposerPreferences(taskId) {
  // Called whenever a task becomes the selected one. Falls back to the
  // global seed preferences when no override exists so the composer always
  // shows a coherent state.
  const fallback = {
    model_id: agentComposerPreferences.model_id || "",
    effort: normalizeAgentEffort(agentComposerPreferences.effort),
    acceptance_mode: normalizeAgentAcceptanceMode(agentComposerPreferences.acceptance_mode),
  };
  const override = getAgentTaskComposerOverride(taskId) || {};
  agentSelectedModelId = override.model_id !== undefined
    ? override.model_id
    : fallback.model_id;
  agentSelectedEffort = override.effort !== undefined
    ? override.effort
    : fallback.effort;
  agentAcceptanceMode = override.acceptance_mode !== undefined
    ? override.acceptance_mode
    : fallback.acceptance_mode;
}

function resetAgentComposerToGlobalDefaults() {
  agentSelectedModelId = agentComposerPreferences.model_id || "";
  agentSelectedEffort = normalizeAgentEffort(agentComposerPreferences.effort);
  agentAcceptanceMode = normalizeAgentAcceptanceMode(agentComposerPreferences.acceptance_mode);
}

function renderLLMSettingsSummary() {
  const models = llmSettings.models || [];
  const systemButton = $("openGovernanceSettingsButton");
  if (models.length === 0) {
    if (systemButton) systemButton.dataset.llmSummary = "未配置";
    return;
  }
  const primary = models.find((model) => model.model_id === llmSettings.default_model_id) || models[0];
  const name = llmModelDisplayName(primary);
  if (systemButton) {
    systemButton.dataset.llmSummary = models.length > 1 ? `${name} 等 ${models.length} 个` : name;
  }
}

function llmModelDisplayName(model = {}) {
  return model.display_name || model.model_name || model.model_id || "未命名模型";
}

function renderLLMModelProfiles() {
  const list = $("llmModelProfiles");
  if (!list) return;
  const models = llmSettings.models || [];
  if (models.length === 0) {
    list.innerHTML = '<div class="llm-engine-empty">还没有配置模型，点击下方「添加模型」。</div>';
    return;
  }
  list.innerHTML = models.map((model, index) => {
    const name = llmModelDisplayName(model);
    const meta = [model.model_name, model.api_base_url].filter(Boolean).join(" · ") || "未填写连接信息";
    return [
      `<div class="llm-engine-item" data-llm-edit="${index}" role="button" tabindex="0">`,
      '<div class="llm-engine-item-info">',
      `<div class="llm-engine-item-name">${escapeHtml(name)}</div>`,
      `<div class="llm-engine-item-url">${escapeHtml(meta)}</div>`,
      "</div>",
      `<button class="engine-del-btn" type="button" data-llm-remove="${index}" title="删除模型" aria-label="删除模型">×</button>`,
      "</div>",
    ].join("");
  }).join("");
}

function collectLLMSettings() {
  const models = (llmSettings.models || []).map((model) => {
    const payload = {
      model_id: model.model_id || "",
      display_name: (model.display_name || "").trim(),
      provider: model.provider || "OpenAI Compatible",
      model_name: (model.model_name || "").trim(),
      api_base_url: (model.api_base_url || "").trim(),
      enabled: model.enabled !== false,
      enable_thinking: Boolean(model.enable_thinking),
      timeout_seconds: Number(model.timeout_seconds || 60),
    };
    if (typeof model.api_key === "string" && model.api_key.trim()) {
      payload.api_key = model.api_key.trim();
    } else if (model.has_api_key) {
      payload.has_api_key = true;
    }
    return payload;
  });
  // default_model_id is left for the server to derive — model selection happens
  // in the composer, not here.
  return { default_model_id: "", models };
}

async function loadLLMSettings({ silent = false } = {}) {
  try {
    const payload = await api("/api/settings/llm");
    llmSettings = normalizeLLMSettings(payload);
    renderLLMModelProfiles();
    renderLLMSettingsSummary();
    renderAgentModelOptions();
    if (!silent) setLLMSettingsStatus("");
  } catch (error) {
    llmSettings = normalizeLLMSettings();
    renderLLMModelProfiles();
    renderLLMSettingsSummary();
    renderAgentModelOptions();
    if (!silent) setLLMSettingsStatus(error.message || "大模型配置读取失败。", "error");
  }
}

// Persists the current in-memory engine list. Throws on failure so callers can
// roll back; status reporting is left to the caller's dialog.
async function saveLLMSettings() {
  const payload = await api("/api/settings/llm", {
    method: "PUT",
    body: JSON.stringify(collectLLMSettings()),
  });
  llmSettings = normalizeLLMSettings(payload);
  renderLLMModelProfiles();
  renderLLMSettingsSummary();
  renderAgentModelOptions();
}

function setMemoryPolicyStatus(message, kind = "info") {
  const status = $("memoryPolicyStatus");
  if (!status) return;
  status.textContent = message || "";
  status.className = `status ${kind}`;
}

function applyMemoryPolicy(settings = {}) {
  for (const input of document.querySelectorAll(".memory-policy-switch")) {
    const key = input.dataset.memoryPolicy;
    if (key && key in settings) input.checked = Boolean(settings[key]);
  }
}

function collectMemoryPolicy() {
  const out = {};
  for (const input of document.querySelectorAll(".memory-policy-switch")) {
    if (input.dataset.memoryPolicy) out[input.dataset.memoryPolicy] = Boolean(input.checked);
  }
  return out;
}

async function loadMemoryPolicySettings({ silent = false } = {}) {
  try {
    const payload = await api("/api/settings/memory-policy");
    applyMemoryPolicy(payload.settings || {});
    if (!silent) setMemoryPolicyStatus("");
  } catch (error) {
    if (!silent) setMemoryPolicyStatus(error.message || "记忆策略读取失败。", "error");
  }
}

async function saveMemoryPolicySettings() {
  try {
    setMemoryPolicyStatus("正在保存记忆策略...");
    const payload = await api("/api/settings/memory-policy", {
      method: "PUT",
      body: JSON.stringify(collectMemoryPolicy()),
    });
    applyMemoryPolicy(payload.settings || {});
    setMemoryPolicyStatus("记忆策略已保存。", "success");
  } catch (error) {
    setMemoryPolicyStatus(error.message || "记忆策略保存失败。", "error");
    loadMemoryPolicySettings({ silent: true });
  }
}

function handleMemoryPolicyChange(event) {
  if (event.target.closest(".memory-policy-switch")) saveMemoryPolicySettings();
}

function addLLMModelProfile() {
  openLLMEngineEdit(null);
}

async function removeLLMModelProfile(index) {
  const previous = llmSettings.models || [];
  llmSettings = { ...llmSettings, models: previous.filter((_, i) => i !== index) };
  renderLLMModelProfiles();
  try {
    await saveLLMSettings();
    setLLMSettingsStatus("模型已删除。", "success");
  } catch (error) {
    llmSettings = { ...llmSettings, models: previous };
    renderLLMModelProfiles();
    setLLMSettingsStatus(error.message || "删除失败。", "error");
  }
}

function openLLMEngineEdit(index) {
  llmEditingIndex = index;
  const model = index === null ? {} : (llmSettings.models[index] || {});
  $("llmEngineEditTitle").textContent = index === null ? "添加模型" : "编辑模型";
  $("llmEngineDisplayName").value = model.display_name || "";
  $("llmEngineModelName").value = model.model_name || "";
  $("llmEngineBaseUrl").value = model.api_base_url || "";
  $("llmEngineEnableThinking").checked = Boolean(model.enable_thinking);
  const keyInput = $("llmEngineApiKey");
  keyInput.value = "";
  keyInput.placeholder = model.has_api_key ? "留空保持不变" : "sk-...";
  setLLMEngineEditStatus("");
  $("llmEngineEditDialog").showModal();
  $("llmEngineDisplayName").focus();
}

function closeLLMEngineEdit() {
  $("llmEngineEditDialog").close();
  llmEditingIndex = null;
}

async function saveLLMEngineEdit() {
  const displayName = $("llmEngineDisplayName").value.trim();
  const modelName = $("llmEngineModelName").value.trim();
  const baseUrl = $("llmEngineBaseUrl").value.trim();
  const apiKey = $("llmEngineApiKey").value.trim();
  const editing = llmEditingIndex !== null ? (llmSettings.models[llmEditingIndex] || {}) : null;
  if (!modelName) return setLLMEngineEditStatus("请填写模型名称。", "error");
  if (!baseUrl) return setLLMEngineEditStatus("请填写 API 地址。", "error");
  if (!apiKey && !(editing && editing.has_api_key)) {
    return setLLMEngineEditStatus("请填写 API 密钥。", "error");
  }

  const model = editing
    ? { ...editing }
    : { model_id: "", provider: "OpenAI Compatible", timeout_seconds: 60, enabled: true, has_api_key: false };
  model.display_name = displayName;
  model.model_name = modelName;
  model.api_base_url = baseUrl;
  model.enable_thinking = $("llmEngineEnableThinking").checked;
  if (apiKey) {
    model.api_key = apiKey;
    model.has_api_key = true;
  }

  const previous = llmSettings.models || [];
  const models = editing
    ? previous.map((item, i) => (i === llmEditingIndex ? model : item))
    : [...previous, model];
  llmSettings = { ...llmSettings, models };

  try {
    setLLMEngineEditStatus("正在保存...");
    await saveLLMSettings();
    closeLLMEngineEdit();
    setLLMSettingsStatus("模型已保存。", "success");
  } catch (error) {
    llmSettings = { ...llmSettings, models: previous };
    renderLLMModelProfiles();
    setLLMEngineEditStatus(error.message || "保存失败。", "error");
  }
}

function rememberSelectedTaskId(taskId) {
  rememberStoredSelectedTaskId(selectedTaskStorageKey, taskId);
}

function storedSelectedTaskId() {
  return readStoredSelectedTaskId(selectedTaskStorageKey);
}

function loadResultScrollPositions() {
  loadStoredResultScrollPositions(resultScrollPositionsStorageKey, resultScrollPositionsByTask);
}

function persistResultScrollPositions() {
  persistStoredResultScrollPositions(resultScrollPositionsStorageKey, resultScrollPositionsByTask);
}

function scheduleResultScrollPositionsPersist() {
  if (resultScrollPersistFrame !== null) return;
  resultScrollPersistFrame = window.requestAnimationFrame(() => {
    resultScrollPersistFrame = null;
    persistResultScrollPositions();
  });
}

function restoreSelectedTaskPlaceholder() {
  if (selectedTaskId) return;
  const storedTaskId = storedSelectedTaskId();
  if (!storedTaskId) return;
  selectedTaskId = storedTaskId;
  selectedTask = null;
}

function syncSelectedTaskFromCache() {
  if (!selectedTaskId) {
    const storedTaskId = storedSelectedTaskId();
    if (storedTaskId) {
      const restored = taskCache.find((task) => task.id === storedTaskId);
      if (restored) {
        selectedTaskId = restored.id;
        selectedTask = restored;
        applyAgentTaskComposerPreferences(restored.id);
        prepareResultScrollRestoreForTask(restored.id);
        return;
      }
      rememberSelectedTaskId(null);
    }
    selectedTask = null;
    return;
  }
  const current = taskCache.find((task) => task.id === selectedTaskId);
  if (current) {
    const wasPlaceholder = !selectedTask;
    selectedTask = current;
    rememberSelectedTaskId(current.id);
    if (wasPlaceholder) {
      applyAgentTaskComposerPreferences(current.id);
      prepareResultScrollRestoreForTask(current.id);
    }
    return;
  }
  selectedTaskId = null;
  selectedTask = null;
  rememberSelectedTaskId(null);
}

function findTaskInCache(taskId) {
  return taskCache.find((task) => task.id === taskId) || null;
}

function ensureActiveTaskProgressPolling(task = selectedTask) {
  const taskId = task?.id || selectedTaskId;
  if (!taskId || !taskServerBusyAction(task)) return;
  if (progressPolls.has(taskId)) return;
  pollValidationProgress(terminalTaskStatuses, taskId, { background: true }).catch(() => null);
}

function runModeLabel(mode) {
  return mode === "agent" ? "Agent 模式" : "手动模式";
}

function selectedTaskIsAgentMode(task = selectedTask) {
  return task?.run_mode === "agent";
}

function updateWorkspaceGreeting(now = new Date()) {
  updateWorkspaceGreetingView({ now, getElementById: $ });
}

function setTaskHeroGlassActive(hero, workspace, glassActive) {
  if (taskHeroGlassActive === glassActive) return;
  taskHeroGlassActive = glassActive;
  hero.classList.toggle("is-glass-active", glassActive);
  workspace.classList.toggle("is-glass-active", glassActive);
}

function updateTaskHeroGlassState({ measureScroll = false } = {}) {
  const scrollContent = $("resultScrollContent");
  const hero = $("taskHero");
  const workspace = $("resultWorkspace");
  if (!scrollContent || !hero || !workspace) return;
  if (measureScroll) {
    taskHeroCanScroll = scrollContent.scrollHeight > scrollContent.clientHeight + 1;
  }
  const glassActive = taskHeroCanScroll && scrollContent.scrollTop > 6;
  setTaskHeroGlassActive(hero, workspace, glassActive);
}

function beginTaskContentLoad(taskId) {
  if (taskContentSettleTimer !== null) {
    window.clearTimeout(taskContentSettleTimer);
    taskContentSettleTimer = null;
  }
  pendingTaskContentLoadTaskId = taskId || null;
  const workspace = $("validationWorkspace");
  workspace?.classList.remove("is-task-content-settling");
  workspace?.classList.toggle("is-task-content-loading", Boolean(taskId));
}

function finishTaskContentLoad(taskId = pendingTaskContentLoadTaskId) {
  if (taskId && pendingTaskContentLoadTaskId !== taskId) return;
  pendingTaskContentLoadTaskId = null;
  const workspace = $("validationWorkspace");
  if (!workspace) return;
  workspace.classList.remove("is-task-content-loading");
  if (!taskId) {
    workspace.classList.remove("is-task-content-settling");
    return;
  }
  workspace.classList.add("is-task-content-settling");
  taskContentSettleTimer = window.setTimeout(() => {
    taskContentSettleTimer = null;
    workspace.classList.remove("is-task-content-settling");
  }, 220);
}

function clearTaskContentLoad() {
  if (taskContentSettleTimer !== null) {
    window.clearTimeout(taskContentSettleTimer);
    taskContentSettleTimer = null;
  }
  pendingTaskContentLoadTaskId = null;
  const workspace = $("validationWorkspace");
  workspace?.classList.remove("is-task-content-loading");
  workspace?.classList.remove("is-task-content-settling");
}

function rememberResultScrollPosition(taskId = selectedTaskId) {
  const scrollContent = $("resultScrollContent");
  if (!scrollContent || !taskId) return;
  resultScrollPositionsByTask.set(taskId, scrollContent.scrollTop);
  scheduleResultScrollPositionsPersist();
}

function cancelResultScrollRestoreFrame() {
  if (resultScrollRestoreFrame === null) return;
  window.cancelAnimationFrame(resultScrollRestoreFrame);
  resultScrollRestoreFrame = null;
}

function prepareResultScrollRestoreForTask(taskId) {
  if (!taskId) return;
  pendingResultScrollRestoreTaskId = taskId;
  suppressAgentAutoScrollTaskId = taskId;
  // Reset on every task switch so a stale `false` from the previous task
  // does not stop the next task's typewriter from auto-following.
  agentAutoScrollFollows = true;
  if (agentAutoScrollFrame !== null) {
    window.cancelAnimationFrame(agentAutoScrollFrame);
    agentAutoScrollFrame = null;
  }
}

function applyResultScrollPosition(taskId = selectedTaskId) {
  const scrollContent = $("resultScrollContent");
  if (!scrollContent || !taskId) return;
  const savedTop = resultScrollPositionsByTask.get(taskId) || 0;
  const maxTop = Math.max(0, scrollContent.scrollHeight - scrollContent.clientHeight);
  scrollContent.scrollTop = Math.min(savedTop, maxTop);
  updateTaskHeroGlassState({ measureScroll: true });
}

function syncAgentAutoScrollFollowFromCurrentPosition(taskId = selectedTaskId) {
  if (taskId !== selectedTaskId || !selectedTaskIsAgentMode()) return;
  const scrollContent = $("resultScrollContent");
  if (!scrollContent) return;
  if (scrollContent.scrollHeight <= scrollContent.clientHeight) {
    agentAutoScrollFollows = true;
    return;
  }
  const distance = scrollContent.scrollHeight - scrollContent.scrollTop - scrollContent.clientHeight;
  agentAutoScrollFollows = distance <= AGENT_AUTO_SCROLL_BOTTOM_TOLERANCE_PX;
}

function nextAnimationFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(resolve));
}

async function restoreResultScrollPositionAfterRender(taskId = selectedTaskId) {
  if (!taskId) return;
  cancelResultScrollRestoreFrame();
  await nextAnimationFrame();
  await nextAnimationFrame();
  if (selectedTaskId !== taskId) {
    if (suppressAgentAutoScrollTaskId === taskId) suppressAgentAutoScrollTaskId = null;
    return;
  }
  applyResultScrollPosition(taskId);
  if (pendingResultScrollRestoreTaskId === taskId) pendingResultScrollRestoreTaskId = null;
  if (suppressAgentAutoScrollTaskId === taskId) suppressAgentAutoScrollTaskId = null;
  syncAgentAutoScrollFollowFromCurrentPosition(taskId);
}

function scheduleResultScrollRestore(taskId = selectedTaskId) {
  if (!taskId) return;
  prepareResultScrollRestoreForTask(taskId);
  cancelResultScrollRestoreFrame();
  resultScrollRestoreFrame = window.requestAnimationFrame(() => {
    resultScrollRestoreFrame = window.requestAnimationFrame(() => {
      resultScrollRestoreFrame = null;
      if (pendingResultScrollRestoreTaskId !== taskId || selectedTaskId !== taskId) {
        if (suppressAgentAutoScrollTaskId === taskId) suppressAgentAutoScrollTaskId = null;
        return;
      }
      applyResultScrollPosition(taskId);
      pendingResultScrollRestoreTaskId = null;
      if (suppressAgentAutoScrollTaskId === taskId) suppressAgentAutoScrollTaskId = null;
      syncAgentAutoScrollFollowFromCurrentPosition(taskId);
    });
  });
}

function scheduleTaskHeroGlassState() {
  if (taskHeroGlassFrame !== null) return;
  taskHeroGlassFrame = requestAnimationFrame(() => {
    taskHeroGlassFrame = null;
    updateTaskHeroGlassState();
  });
}

function handleResultScroll() {
  if (pendingResultScrollRestoreTaskId !== selectedTaskId) {
    rememberResultScrollPosition();
  }
  scheduleTaskHeroGlassState();
  recomputeAgentAutoScrollFollow();
}

function recomputeAgentAutoScrollFollow() {
  if (!selectedTaskIsAgentMode()) return;
  const scrollContent = $("resultScrollContent");
  if (!scrollContent) return;
  // Programmatic scrolls reach this handler without a preceding wheel/touch
  // event. Treat them as no-ops so the typewriter's own snap-to-bottom cannot
  // re-enable follow-mode the user just disengaged a few milliseconds ago.
  if (performance.now() - lastUserScrollInputAt > AGENT_USER_SCROLL_INPUT_WINDOW_MS) return;
  if (scrollContent.scrollHeight <= scrollContent.clientHeight) return;
  const distance =
    scrollContent.scrollHeight - scrollContent.scrollTop - scrollContent.clientHeight;
  if (distance < 0) return;
  agentAutoScrollFollows = distance <= AGENT_AUTO_SCROLL_BOTTOM_TOLERANCE_PX;
}

function noteAgentUserScrollInput() {
  lastUserScrollInputAt = performance.now();
}

function routeWorkspaceWheelToResult(event) {
  const scrollContent = $("resultScrollContent");
  const appShell = $("appShell");
  const target = event.target instanceof Element ? event.target : null;
  if (!scrollContent || !appShell || !target) return;
  if (event.defaultPrevented || event.ctrlKey) return;
  if (!appShell.contains(target)) return;
  if (scrollTargetIsWithin(target, "#taskSidebar, #progressRail, #workflowStepper")) return;
  if (scrollTargetIsWithin(target, "#resultScrollContent")) return;
  if (scrollTargetIsWithin(target, "dialog, textarea, select, input, .metric-table-scroll")) return;

  const previousTop = scrollContent.scrollTop;
  const previousLeft = scrollContent.scrollLeft;
  scrollContent.scrollTop += event.deltaY;
  scrollContent.scrollLeft += event.deltaX;
  if (scrollContent.scrollTop !== previousTop || scrollContent.scrollLeft !== previousLeft) {
    event.preventDefault();
  }
}

function scrollTargetIsWithin(target, selector) {
  return target instanceof Element && Boolean(target.closest(selector));
}

function syncTaskHeroGlassLayout() {
  const workspace = $("resultWorkspace");
  const head = document.querySelector("#resultWorkspace .workspace-head");
  if (!workspace || !head) return;
  const headHeight = head.getBoundingClientRect().height;
  if (Number.isFinite(headHeight) && headHeight > 0) {
    workspace.style.setProperty("--workspace-head-space", `${Math.ceil(headHeight)}px`);
  }
  syncAgentComposerClearance();
  updateTaskHeroGlassState({ measureScroll: true });
}

function syncAgentComposerClearance() {
  const workspace = $("resultWorkspace");
  const composer = $("agentComposer");
  if (!workspace || !composer || composer.classList.contains("hidden")) return;
  const composerHeight = composer.getBoundingClientRect().height;
  const composerGap = parseFloat(getComputedStyle(workspace).getPropertyValue("--agent-composer-gap")) || 28;
  if (!Number.isFinite(composerHeight) || composerHeight <= 0) return;
  workspace.style.setProperty("--agent-composer-clearance", `${Math.ceil(composerHeight + composerGap)}px`);
}

function renderCurrentTask({ force = false } = {}) {
  const nextSignature = currentTaskSignature(selectedTask);
  if (!force && renderSignatures.currentTask === nextSignature) return;
  renderSignatures.currentTask = nextSignature;

  renderCurrentTaskWorkspace({
    selectedTask,
    selectedTaskId,
    getElementById: $,
    taskDisplayName,
    renderTaskSnapshot,
    setActionStatus,
    updateGreeting: updateWorkspaceGreetingView,
    statusOverride: actionStatusOverride?.taskId === selectedTaskId ? actionStatusOverride : null,
    setTaskFailureActionStatus,
    taskActionStatusSnapshot,
    syncTaskHeroGlassLayout,
  });
}

function workflowStepStatus(index, activeIndex) {
  if (!selectedTaskId) return "pending";
  const status = selectedTask?.status || "";
  const step = workflowSteps[index];
  const runningStepId = taskRunningStepId(status);
  if (runningStepId && step.id === runningStepId) return "running";
  if (taskFailureWasRestartReclaim(selectedTask) && workflowStageCompleteFromEvidence(step.id)) return "succeeded";
  const failedStepId = taskFailureStepId(selectedTask);
  if (failedStepId) {
    const failedIndex = workflowSteps.findIndex((candidate) => candidate.id === failedStepId);
    if (step.id === failedStepId) return "failed";
    if (failedIndex >= 0) return index < failedIndex ? "succeeded" : "pending";
  }
  if (status === "created") return "pending";
  if (status === "scanned" || status === "configured") {
    return index < 1 ? "succeeded" : "pending";
  }
  if (status === "running") {
    return index < 1 ? "succeeded" : index === 1 ? "running" : "pending";
  }
  if (status === "executed") {
    return index < 2 ? "succeeded" : "pending";
  }
  if (status === "computing_metrics") {
    return index < 2 ? "succeeded" : index === 2 ? "running" : "pending";
  }
  if (status === "writing_artifacts") {
    return index < 3 ? "succeeded" : index === 3 && taskServerBusyAction() === "report" ? "running" : "pending";
  }
  if (status === "review_required") return "succeeded";
  if (status === "succeeded") return "succeeded";
  return "pending";
}

function workflowStepStatusLabel(status, actionId) {
  if (status === "succeeded") return "已完成";
  if (status === "review") return "需复核";
  if (status === "failed") return "失败";
  if (status === "running" && taskBusyAction() === actionId) return "执行中";
  if (status === "running") return "当前";
  return "未开始";
}

function stepStopAction(step) {
  const status = selectedTask?.status || "";
  const selectedBusyAction = taskBusyAction();
  if (step.action === "notebook" && status === "running") return "cancelNotebook";
  if (step.action === "metrics" && status === "computing_metrics") return "cancelMetrics";
  if (step.action === "report" && (selectedBusyAction === "report" || taskServerBusyAction() === "report")) return "cancelReport";
  return null;
}

function completedReportReadyForDownloads(step) {
  const selectedBusyAction = taskBusyAction();
  return (
    step.action === "report" &&
    selectedBusyAction !== "report" &&
    selectedTask?.report_available === true &&
    ["succeeded", "review_required"].includes(selectedTask?.status)
  );
}

function stepDownloadActionsHtml(step) {
  if (!completedReportReadyForDownloads(step)) return "";
  return [
    '<div class="step-download-actions">',
    '<button class="button compact step-action-button secondary" type="button" data-step-action="previewWordReport">',
    "预览",
    "</button>",
    '<button class="button compact step-action-button primary word" type="button" data-step-action="downloadWordReport">',
    "下载Word",
    "</button>",
    '<button class="button compact step-action-button excel" type="button" data-step-action="downloadExcelAnalysis">',
    "下载Excel",
    "</button>",
    "</div>",
  ].join("");
}

function stepActionButtonHtml(step) {
  if (selectedTaskIsAgentMode()) return "";
  if (!step.action || completedReportReadyForDownloads(step)) return "";
  const selectedBusy = selectedTaskIsBusy();
  const stopAction = stepStopAction(step);
  const isStopAction = Boolean(stopAction);
  const action = stopAction || step.action;
  const canRunAction = isStopAction || canRunStepAction(step.action);
  const disabled = !selectedTaskId || (selectedBusy && !isStopAction) || !canRunAction;
  const recommended = !isStopAction && recommendedAction() === step.action;
  const tone = isStopAction ? "danger" : recommended ? "primary" : "secondary";
  const label = isStopAction ? "停止" : step.actionLabel || "执行";
  const title = !selectedTaskId
    ? "请先选择任务"
    : (selectedBusy && !isStopAction)
      ? "当前任务正在执行"
      : !canRunAction
        ? "请先完成上一步"
      : isStopAction
        ? "停止当前执行"
        : "";
  return [
    `<button class="button compact step-action-button ${tone}" type="button" data-step-action="${escapeHtml(action)}"${disabled ? " disabled" : ""}${title ? ` title="${escapeHtml(title)}"` : ""}>`,
    escapeHtml(label),
    "</button>",
  ].join("");
}

// Status checker shown before every step / sub-step: a hollow ring (pending),
// a spinning arc (running), a filled green tick (succeeded), or a red cross
// (failed). SVG marks use currentColor so CSS controls the glyph color.
function stepCheckerHtml(state) {
  if (state === "succeeded") {
    return (
      '<span class="check-icon succeeded" aria-hidden="true">' +
      '<svg viewBox="0 0 16 16" width="11" height="11"><path d="M3 8.4l3 3 7-7" fill="none" ' +
      'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
      "</span>"
    );
  }
  if (state === "failed") {
    return (
      '<span class="check-icon failed" aria-hidden="true">' +
      '<svg viewBox="0 0 16 16" width="10" height="10"><path d="M4 4l8 8M12 4l-8 8" fill="none" ' +
      'stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>' +
      "</span>"
    );
  }
  if (state === "stopped") {
    return '<span class="check-icon stopped" aria-hidden="true"></span>';
  }
  if (state === "review") {
    return (
      '<span class="check-icon review" aria-hidden="true">' +
      '<svg viewBox="0 0 16 16" width="10" height="10"><path d="M8 3v6M8 12.5h.01" fill="none" ' +
      'stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>' +
      "</span>"
    );
  }
  if (state === "running") {
    // Sync the spin phase to a global clock so the ring keeps rotating smoothly
    // even though the stepper is rebuilt from scratch on every poll tick.
    return `<span class="check-icon running" aria-hidden="true" style="animation-delay: -${Date.now() % 800}ms"></span>`;
  }
  return '<span class="check-icon pending" aria-hidden="true"></span>';
}

function notebookStepTone(status) {
  const value = String(status || "").toLowerCase();
  if (["success", "succeeded", "done", "completed", "passed"].includes(value)) return "succeeded";
  if (taskStopped(selectedTask) && ["running", "executing", "active"].includes(value)) return "stopped";
  if (["running", "executing", "active"].includes(value)) return "running";
  if (["failed", "error", "exception"].includes(value)) return "failed";
  return "pending";
}

function stepWorkflowStage(step) {
  const id = String(step?.id || "");
  if (id.startsWith("system-metrics-")) return "metrics";
  return "notebook";
}

function notebookStepsForRail() {
  return latestNotebookSteps.filter((step) => stepWorkflowStage(step) === "notebook");
}

function metricStepsForRail() {
  return latestNotebookSteps.filter((step) => stepWorkflowStage(step) === "metrics");
}

function workflowStageCompleteFromEvidence(stepId) {
  if (stepId === "scan") return latestNotebookSteps.length > 0;
  const stageSteps = stepId === "notebook"
    ? notebookStepsForRail()
    : stepId === "metrics"
      ? metricStepsForRail()
      : [];
  return stageSteps.length > 0 && stageSteps.every((step) => notebookStepTone(step.status) === "succeeded");
}

function plannedReproducibilitySteps() {
  return [
    {
      id: "system-repro-pmml",
      title: "PMML 打分",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
    {
      id: "system-repro-compare",
      title: "分数一致性对比",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
  ];
}

function plannedMetricSteps() {
  return [
    {
      id: "system-metrics-prepare",
      title: "指标数据准备",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
    {
      id: "system-metrics-score",
      title: "RMC_SCORE_FN 全量打分",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
    {
      id: "system-metrics-basic",
      title: "样本与变量概览",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
    {
      id: "system-metrics-ks",
      title: "KS 计算",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
    {
      id: "system-metrics-psi",
      title: "PSI 计算",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
    {
      id: "system-metrics-binning",
      title: "分箱计算",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
    {
      id: "system-metrics-stress",
      title: "压力测试",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
    {
      id: "system-metrics-output",
      title: "写入指标产物",
      status: "pending",
      cell_count: 1,
      cell_indexes: [],
      source_previews: [],
      system: true,
    },
  ];
}

function shouldPrimeReproducibilitySteps() {
  return taskBusyAction() === "notebook" || selectedTask?.status === "running";
}

function shouldPrimeMetricSteps() {
  return taskBusyAction() === "metrics" || selectedTask?.status === "computing_metrics";
}

function mergePendingSystemSteps(notebookSteps = []) {
  const steps = Array.isArray(notebookSteps) ? [...notebookSteps] : [];
  const existingIds = new Set(steps.map((step) => step?.id).filter(Boolean));
  const plannedSteps = [
    ...(shouldPrimeReproducibilitySteps() ? plannedReproducibilitySteps() : []),
    ...(shouldPrimeMetricSteps() ? plannedMetricSteps() : []),
  ];
  plannedSteps.forEach((step) => {
    if (!existingIds.has(step.id)) steps.push(step);
  });
  return steps;
}

function appendPendingReproducibilitySteps() {
  latestNotebookSteps = mergePendingSystemSteps(latestNotebookSteps);
  renderWorkflowStepper();
}

function appendPendingMetricSteps() {
  latestNotebookSteps = mergePendingSystemSteps(latestNotebookSteps);
  renderWorkflowStepper();
}

function stepElapsedSeconds(step, nextStep = null) {
  const startedAt = Date.parse(step?.started_at || "");
  if (step?.status !== "running" && Number.isFinite(step?.elapsed_seconds)) {
    const elapsed = Number(step.elapsed_seconds);
    const nextStartedAt = Date.parse(nextStep?.started_at || "");
    if (
      elapsed < 1 &&
      Number.isFinite(startedAt) &&
      Number.isFinite(nextStartedAt) &&
      nextStartedAt > startedAt
    ) {
      return Math.max(elapsed, (nextStartedAt - startedAt) / 1000);
    }
    return elapsed;
  }
  if (!Number.isFinite(startedAt)) return null;
  const endedAt = Date.parse(step?.ended_at || "");
  const endMs = Number.isFinite(endedAt) ? endedAt : Date.now();
  return Math.max(0, (endMs - startedAt) / 1000);
}

function formatStepElapsed(step, nextStep = null) {
  const seconds = stepElapsedSeconds(step, nextStep);
  if (!Number.isFinite(seconds)) return "";
  const totalSeconds = Math.max(0, Math.round(seconds));
  if (totalSeconds === 0 && seconds >= 0 && step?.status !== "pending" && step?.started_at) return "0s";
  const minutes = Math.floor(totalSeconds / 60);
  const remainder = totalSeconds % 60;
  if (minutes <= 0) return `${remainder}s`;
  return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
}

function stepAfterInLatestNotebookSteps(step) {
  if (!step || !Array.isArray(latestNotebookSteps)) return null;
  const index = latestNotebookSteps.findIndex((candidate) => (
    candidate === step || (step.id && candidate.id === step.id)
  ));
  return index >= 0 ? latestNotebookSteps[index + 1] || null : null;
}

function renderNotebookStepRail(
  notebookSteps = latestNotebookSteps,
  title = "分段进度",
  parentNumber = "",
  parentStatus = "",
  stageId = "",
) {
  if (!Array.isArray(notebookSteps) || notebookSteps.length === 0) {
    return "";
  }
  const tones = notebookSteps.map((step) => notebookStepTone(step.status));
  // When the parent stage is running but the backend has not flagged a specific
  // sub-step as running yet, spin the first unfinished sub-step so it stays in
  // sync with the parent's spinner instead of sitting on a hollow circle.
  const hasRunning = tones.includes("running");
  const activeIndex = parentStatus === "running" && !hasRunning
    ? tones.findIndex((tone) => tone !== "succeeded" && tone !== "failed")
    : -1;
  return [
    '<section class="notebook-step-group">',
    `<h4>${escapeHtml(title)} · ${notebookSteps.length}</h4>`,
    ...notebookSteps.map((step, index) => {
      const title = step.title || step.heading || step.name || `步骤 ${step.step_order ?? index + 1}`;
      const tone = index === activeIndex ? "running" : tones[index];
      const cells = Number.isFinite(step.cell_count) ? `${step.cell_count} cells` : "";
      const elapsed = formatStepElapsed(step, notebookSteps[index + 1] || stepAfterInLatestNotebookSteps(step));
      const number = parentNumber ? `${parentNumber}.${index + 1}` : `${index + 1}`;
      // Two separate spans so the per-second elapsed updater can rewrite just
      // the elapsed text without touching the rest of the substep DOM.
      const cellsHtml = cells ? `<span class="step-cells">${escapeHtml(cells)}</span>` : "";
      const elapsedKey = stageId && step?.id ? `${stageId}:${step.id}` : "";
      const elapsedHtml = elapsedKey
        ? `<span class="step-elapsed" data-step-elapsed-key="${escapeHtml(elapsedKey)}">${escapeHtml(elapsed)}</span>`
        : (elapsed ? `<span class="step-elapsed">${escapeHtml(elapsed)}</span>` : "");
      const separator = cellsHtml && elapsedHtml && elapsed ? '<span class="step-sep"> · </span>' : "";
      const metaHtml = cellsHtml || elapsedHtml ? `<small>${cellsHtml}${separator}${elapsedHtml}</small>` : "";
      return [
        `<div class="notebook-step ${tone}">`,
        stepCheckerHtml(tone),
        `<span class="notebook-step-no">${escapeHtml(number)}</span>`,
        "<strong>",
        escapeHtml(title),
        "</strong>",
        metaHtml,
        "</div>",
      ].join("");
    }),
    "</section>",
  ].join("");
}

function refreshWorkflowStepperElapsedTimes() {
  const stepper = $("workflowStepper");
  if (!stepper) return;
  const notebookSteps = notebookStepsForRail();
  const metricSteps = metricStepsForRail();
  const lookup = new Map();
  const fillLookup = (stageId, steps) => {
    steps.forEach((step, index, arr) => {
      if (!step?.id) return;
      lookup.set(`${stageId}:${step.id}`, {
        step,
        next: arr[index + 1] || stepAfterInLatestNotebookSteps(step),
      });
    });
  };
  fillLookup("notebook", notebookSteps);
  fillLookup("metrics", metricSteps);
  stepper.querySelectorAll("[data-step-elapsed-key]").forEach((node) => {
    const entry = lookup.get(node.dataset.stepElapsedKey || "");
    if (!entry) return;
    const elapsed = formatStepElapsed(entry.step, entry.next);
    if (node.textContent !== elapsed) node.textContent = elapsed;
  });
}

function renderWorkflowStepper({ force = false } = {}) {
  const progressRail = $("progressRail");
  const railTitle = document.querySelector("#progressRail .step-rail-head h3");
  if (planRailController.render({ force, renderSignatures })) {
    return;
  }
  progressRail?.setAttribute("aria-label", "验证步骤");
  planRailController.clearArtifactPanel();
  if (railTitle) railTitle.textContent = "验证步骤";
  const nextSignature = workflowStepperSignature(selectedTask);
  if (!force && renderSignatures.workflowStepper === nextSignature) {
    // Structure unchanged; still tick elapsed-seconds spans so running steps
    // do not freeze at the value captured during the last structural render.
    refreshWorkflowStepperElapsedTimes();
    return;
  }
  renderSignatures.workflowStepper = nextSignature;

  const stepper = $("workflowStepper");
  const activeIndex = workflowIndex(selectedTask?.status);
  const stepActionIds = ["scan", "notebook", "metrics", "report"];
  const renderTaskId = selectedTaskId || "";
  const previousScrollTop = stepper.dataset.taskId === renderTaskId ? stepper.scrollTop : 0;
  stepper.innerHTML = "";
  workflowSteps.forEach((step, index) => {
    if (step.action && !stepActionIds.includes(step.action)) return;
    const item = document.createElement("div");
    const classes = ["step"];
    const stepStatus = workflowStepStatus(index, activeIndex);
    if (stepStatus === "succeeded") {
      classes.push("succeeded");
    } else if (stepStatus === "running") {
      classes.push("running");
    } else if (stepStatus === "failed") {
      classes.push("failed");
    } else if (stepStatus === "stopped") {
      classes.push("stopped");
    } else if (stepStatus === "review") {
      classes.push("review");
    } else {
      classes.push("pending");
    }
    item.className = classes.join(" ");
    item.dataset.stepTarget = step.target;
    item.tabIndex = 0;
    item.setAttribute("role", "group");
    item.innerHTML = [
      '<div class="step-head">',
      stepCheckerHtml(stepStatus),
      `<span class="step-number">${index + 1}</span>`,
      '<span class="step-copy">',
      `<strong class="step-title">${escapeHtml(step.title)}</strong>`,
      `<small class="step-hint">${escapeHtml(step.hint)}</small>`,
      "</span>",
      stepActionButtonHtml(step),
      "</div>",
      stepDownloadActionsHtml(step),
      step.id === "notebook" ? renderNotebookStepRail(notebookStepsForRail(), "分段进度", index + 1, stepStatus, "notebook") : "",
      step.id === "metrics" ? renderNotebookStepRail(metricStepsForRail(), "计算进度", index + 1, stepStatus, "metrics") : "",
    ].join("");
    stepper.appendChild(item);
  });
  stepper.dataset.taskId = renderTaskId;
  stepper.scrollTop = previousScrollTop;
  refreshWorkflowStepperElapsedTimes();
}

function formatDate(value) {
  if (!value) return "";
  try {
    return new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(value));
  } catch (_) {
    return value;
  }
}

function taskCreatedMonth(task) {
  const rawDate = task.created_at || task.updated_at || "";
  const date = new Date(rawDate);
  if (Number.isNaN(date.getTime())) return "未知创建月份";
  return `${date.getFullYear()}年${String(date.getMonth() + 1).padStart(2, "0")}月`;
}

function sortMonthGroups([left], [right]) {
  if (left === right) return 0;
  if (left === "未知创建月份") return 1;
  if (right === "未知创建月份") return -1;
  const direction = taskSortMode === "created_asc" ? 1 : -1;
  return left.localeCompare(right, "zh-CN") * direction;
}

function sortTaskTypeGroups([left], [right]) {
  const leftType = left || defaultTaskType;
  const rightType = right || defaultTaskType;
  const leftRank = taskTypeDisplayOrder.indexOf(leftType);
  const rightRank = taskTypeDisplayOrder.indexOf(rightType);
  if (leftRank >= 0 && rightRank >= 0) return leftRank - rightRank;
  if (leftRank >= 0) return -1;
  if (rightRank >= 0) return 1;
  return taskTypeLabel(leftType).localeCompare(taskTypeLabel(rightType), "zh-CN");
}

function compareTasks(left, right) {
  if (taskSortMode === "name_asc") {
    return left.model_name.localeCompare(right.model_name, "zh-CN");
  }
  if (taskSortMode === "name_desc") {
    return right.model_name.localeCompare(left.model_name, "zh-CN");
  }
  const leftDate = Date.parse(left.created_at || left.updated_at || "") || 0;
  const rightDate = Date.parse(right.created_at || right.updated_at || "") || 0;
  return taskSortMode === "created_asc" ? leftDate - rightDate : rightDate - leftDate;
}

function applyTaskFilters(tasks = taskCache) {
  const query = taskSearchQuery.trim().toLowerCase();
  return tasks
    .filter((task) => {
      if (!query) return true;
      return [
        task.model_name,
        task.model_version,
        task.validator,
        task.status_message,
        task.source_dir,
      ].some((value) => String(value || "").toLowerCase().includes(query));
    })
    .sort(compareTasks);
}

// Layered, multi-tone glyphs for the six task kinds — one shared source used by
// the sidebar rows and the task-hero snapshot. Mirrors the welcome-card icons
// in index.html; classes (back/mid/cut/cs/cst/ln) are themed in styles.css.
const TASK_KIND_GLYPHS = {
  data_join:
    '<rect class="back" x="3" y="8" width="10.5" height="10.5" rx="3"></rect><rect class="mid" x="6.75" y="6.75" width="10.5" height="10.5" rx="3"></rect><rect x="10.5" y="5.5" width="10.5" height="10.5" rx="3"></rect><rect class="cut" x="13" y="8.7" width="5.5" height="1.3" rx="0.65"></rect><rect class="cut" x="13" y="11.2" width="3.8" height="1.3" rx="0.65"></rect>',
  feature_analysis:
    '<rect class="back" x="4.4" y="16.6" width="17" height="3.4" rx="1.6"></rect><rect x="5" y="10.5" width="3.2" height="6.6" rx="1"></rect><rect x="9.2" y="7.5" width="3.2" height="9.6" rx="1"></rect><rect x="13.4" y="5" width="3.2" height="12.1" rx="1"></rect><rect x="17.6" y="9" width="3.2" height="8.1" rx="1"></rect>',
  vintage:
    '<rect class="back" x="3.5" y="6" width="17" height="12.5" rx="2"></rect><path class="ln vintage-calendar-binding" d="M7.2 4.8v2.8M16.8 4.8v2.8"></path><path class="ln" d="M6.3 15.5 9.6 12.7 13 14.1 17.8 10"></path>',
  modeling:
    '<rect x="2.6" y="4.6" width="18.8" height="14.8" rx="2.6"></rect><path class="mid" d="M2.6 8 V7 Q2.6 4.6 5 4.6 H19 Q21.4 4.6 21.4 7 V8 Z"></path><circle class="cut" cx="5.5" cy="6.2" r="0.82"></circle><circle class="cut" cx="7.7" cy="6.2" r="0.82"></circle><circle class="cut" cx="9.9" cy="6.2" r="0.82"></circle><path class="cs" d="M8.2 11.2 11 13.8 8.2 16.4"></path><rect class="cut" x="12" y="14.9" width="4" height="1.5" rx="0.75"></rect>',
  validation:
    '<rect class="back" x="7" y="3.5" width="11.5" height="16" rx="2.2"></rect><rect x="5" y="5" width="11.5" height="15.5" rx="2.2"></rect><rect class="mid" x="7.75" y="3.7" width="6" height="2.2" rx="1.1"></rect><rect class="cut" x="7.4" y="9" width="6.6" height="1.2" rx="0.6"></rect><rect class="cut" x="7.4" y="12" width="6.6" height="1.2" rx="0.6"></rect><rect class="cut" x="7.4" y="15" width="4.4" height="1.2" rx="0.6"></rect><circle class="cut" cx="16.6" cy="17.6" r="4.9"></circle><circle cx="16.6" cy="17.6" r="4"></circle><path class="cst" d="M14.8 17.7 16 18.9 18.4 16.4"></path>',
  strategy:
    '<rect class="back" x="4" y="13.8" width="16" height="4.6" rx="1.8"></rect><rect class="mid" x="4" y="9.6" width="16" height="4.6" rx="1.8"></rect><rect x="4" y="5" width="16" height="5.6" rx="1.8"></rect><rect class="cut" x="6.6" y="6.2" width="7.2" height="1.3" rx="0.65"></rect><rect class="cut" x="6.6" y="8.1" width="4.6" height="1.3" rx="0.65"></rect>',
};

function taskKindIconHtml(taskOrType = selectedTask, extraClass = "") {
  const kind = typeof taskOrType === "string" ? taskOrType : taskOrType?.task_type;
  const safeKind = TASK_KIND_GLYPHS[kind] ? kind : defaultTaskType;
  const cls = "task-kind-icon" + (extraClass ? ` ${extraClass}` : "");
  return `<svg class="${cls}" data-kind="${escapeHtml(safeKind)}" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${TASK_KIND_GLYPHS[safeKind] || ""}</svg>`;
}

function appendTaskRow(list, task) {
  const item = document.createElement("div");
  item.className = "task-row-shell";
  item.setAttribute("role", "listitem");

  const row = document.createElement("button");
  row.type = "button";
  row.className = "task-row" + (task.id === selectedTaskId ? " selected" : "");
  row.setAttribute("aria-current", task.id === selectedTaskId ? "true" : "false");
  const tone = taskStatusTone(task);
  const validatorName = escapeHtml(task.validator || "-");
  row.innerHTML = [
    '<span class="task-row-top">',
    '<span class="task-row-title">',
    taskKindIconHtml(task),
    `<strong class="task-row-name">${escapeHtml(task.model_name)}</strong>`,
    "</span>",
    `<span class="task-row-badges"><span class="pill ${tone}">${escapeHtml(taskStatusLabel(task))}</span></span>`,
    "</span>",
    '<span class="task-row-meta">',
    `<small class="task-row-validator" aria-label="验证人员：${validatorName}">`,
    '<svg class="task-row-validator-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">',
    '<circle cx="12" cy="8" r="3.2"></circle>',
    '<path d="M5.5 19c0.9-3.5 3.2-5.4 6.5-5.4s5.6 1.9 6.5 5.4"></path>',
    "</svg>",
    `<span class="task-row-validator-text">${validatorName}</span>`,
    "</small>",
    `<small class="task-row-date">${escapeHtml(formatDate(task.updated_at))}</small>`,
    "</span>",
  ].join("");
  row.onclick = () => selectTask(task);

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "delete-task-button";
  deleteButton.title = "删除任务";
  deleteButton.setAttribute("aria-label", `删除任务 ${task.model_name}`);
  deleteButton.innerHTML = [
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">',
    '<path d="M5 7h14"></path>',
    '<path d="M9.5 7V5.5h5V7"></path>',
    '<path d="M6.5 7 7.4 19h9.2L17.5 7"></path>',
    '<path d="M10 10.5v5"></path>',
    '<path d="M14 10.5v5"></path>',
    "</svg>",
  ].join("");
  deleteButton.onclick = (event) => {
    event.stopPropagation();
    deleteTask(task);
  };

  item.appendChild(row);
  item.appendChild(deleteButton);
  list.appendChild(item);
}

function appendTaskGroup(list, groupName, groupTasks) {
  const heading = document.createElement("div");
  heading.className = "task-group-title";
  heading.textContent = groupName;
  list.appendChild(heading);
  groupTasks.forEach((task) => appendTaskRow(list, task));
}

function renderTaskSnapshot() {
  renderTaskSnapshotView({
    selectedTask,
    getElementById: $,
    taskTypeLabel,
    taskKindIconHtml,
    runModeLabel,
  });
}

function renderTaskList(tasks = applyTaskFilters(taskCache), { force = false } = {}) {
  const nextSignature = taskListSignature(tasks, taskCache.length);
  if (!force && renderSignatures.taskList === nextSignature) return;
  renderSignatures.taskList = nextSignature;

  const list = $("taskList");
  list.innerHTML = "";
  if (taskCache.length === 0) {
    list.innerHTML = '<div class="empty-state">暂无任务</div>';
    return;
  }
  if (tasks.length === 0) {
    list.innerHTML = '<div class="empty-state">没有匹配的任务。</div>';
    return;
  }

  if (taskGroupMode === "validator") {
    const groups = new Map();
    for (const task of tasks) {
      const key = task.validator || "未填写验证人员";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(task);
    }
    [...groups.entries()]
      .sort(([left], [right]) => left.localeCompare(right, "zh-CN"))
      .forEach(([groupName, groupTasks]) => appendTaskGroup(list, groupName, groupTasks));
    return;
  }

  if (taskGroupMode === "task_type") {
    const groups = new Map();
    for (const task of tasks) {
      const key = task.task_type || defaultTaskType;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(task);
    }
    [...groups.entries()]
      .sort(sortTaskTypeGroups)
      .forEach(([taskType, groupTasks]) => appendTaskGroup(list, taskTypeLabel(taskType), groupTasks));
    return;
  }

  if (taskGroupMode === "created_month") {
    const groups = new Map();
    for (const task of tasks) {
      const key = taskCreatedMonth(task);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(task);
    }
    [...groups.entries()]
      .sort(sortMonthGroups)
      .forEach(([groupName, groupTasks]) => appendTaskGroup(list, groupName, groupTasks));
    return;
  }

  tasks.forEach((task) => appendTaskRow(list, task));
}

function selectTask(task) {
  rememberResultScrollPosition();
  if (selectedTaskId === task.id && selectedTask) {
    selectedTask = task;
    rememberSelectedTaskId(task.id);
    renderCurrentTask();
    renderTaskList();
    return;
  }
  // Task identity is changing — drop any in-flight typewriter state so a
  // still-revealing message from the previous task can't re-reveal (or worse,
  // leak its visible-prefix via shared messageId) on the new task's panel.
  resetAgentTypingState();
  selectedTaskId = task.id;
  selectedTask = task;
  rememberSelectedTaskId(task.id);
  applyAgentTaskComposerPreferences(task.id);
  beginTaskContentLoad(task.id);
  prepareResultScrollRestoreForTask(task.id);
  ensureActiveTaskProgressPolling(task);
  renderMetricPreview({});
  renderStoredStateSummaries();
  runAction(async () => {
    try {
      renderTaskList();
      await loadTaskEvidence();
      await loadReportFields();
      await loadAgentMessages(task.id);
    } finally {
      renderAll();
      await restoreResultScrollPositionAfterRender(task.id);
      finishTaskContentLoad(task.id);
    }
  }, { renderAfter: false });
}

function deselectCurrentTask() {
  rememberResultScrollPosition();
  clearTaskContentLoad();
  selectedTaskId = null;
  selectedTask = null;
  rememberSelectedTaskId(null);
  latestNotebookSteps = [];
  agentMessages = [];
  resetAgentComposerToGlobalDefaults();
  renderMetricPreview({});
  setActionStatus("");
  renderStoredStateSummaries();
  renderAll();
}

function renderMetricPreview(
  metricValues = lastMetricValues,
  workbookSource = null,
  sections = lastMetricTableSections,
) {
  renderMetricSectionVisibility();
  lastMetricValues = metricValues || {};
  lastMetricValuesTaskId = selectedTaskId || null;
  lastMetricTableSections = Array.isArray(sections) ? sections : [];

  // Identical metric payloads (same taskId + same values + same sections) must
  // leave the existing DOM intact so charts and KPI cards do not replay their
  // animations or drop hover state during the per-second polling loop.
  const previewTaskId = lastMetricValuesTaskId || "";
  const nextSignature = metricPreviewSignature(
    previewTaskId,
    lastMetricValues,
    lastMetricTableSections,
  );
  if (
    renderSignatures.metricPreviewTaskId === previewTaskId
    && renderSignatures.metricPreview === nextSignature
  ) {
    return;
  }
  renderSignatures.metricPreviewTaskId = previewTaskId;
  renderSignatures.metricPreview = nextSignature;

  // Extract the standalone ROC&KS section so each curve can sit beneath
  // its matching KPI card. The original section is dropped from the
  // visible list (we render 6 sections, not 7).
  let rocCurves = null;
  const visibleSections = lastMetricTableSections.filter((section) => {
    const tables = Array.isArray(section && section.tables) ? section.tables : [];
    const isRocSection =
      (tables[0] && tables[0].layout === "roc_ks_curve")
      || (section && section.title === "ROC&KS 曲线");
    if (isRocSection) {
      rocCurves = (tables[0] && tables[0].curves) || null;
      return false;
    }
    return true;
  });

  if (visibleSections.length === 0) {
    $("metricPreview").innerHTML =
      '<div class="result-summary empty">效果&稳定性验证完成后展示</div>';
    return;
  }
  const sectionHtml = visibleSections
    .map((section, index) => renderMetricTableSection(section, index, { rocCurves }))
    .join("");
  $("metricPreview").innerHTML = sectionHtml;
  attachRocInteractions($("metricPreview"));
  attachMetricTooltip($("metricPreview"));
}

function renderMetricTableSection(section = {}, index = 0, options = {}) {
  const tables = Array.isArray(section.tables) ? section.tables : [];
  const theme = section.section_theme || "cool-blue";
  const isOverallEffect =
    (tables[0] && tables[0].layout === "kpi_cards")
    || section.title === "整体效果&稳定性";
  const sectionIndex = String(index + 1).padStart(2, "0");
  const title = section.title || "";
  return [
    `<section class="metric-table-section" data-theme="${escapeHtml(theme)}" data-section-index="${escapeHtml(sectionIndex)}">`,
    `<h4 class="metric-section-title">${escapeHtml(title)}</h4>`,
    '<div class="metric-table-stack">',
    ...tables.map((table) => {
      if (isOverallEffect && table.layout === "kpi_cards") {
        return renderKpiCards(table, { curves: options.rocCurves || null });
      }
      return renderMetricTable(table);
    }),
    "</div>",
    "</section>",
  ].join("");
}

// ====== Metric overview cell helpers ======

function renderCellByKind(spec, value, context) {
  const kind = (spec && spec.kind) || "text";
  if (kind === "trend-spark" && spec && spec.__localHtml === true) {
    return { cls: "cell-sparkline", html: String(value ?? "") };  // value is raw <svg>
  }
  const headerLabel = context.headerLabel ?? "";
  switch (kind) {
    case "split-badge":
      return {
        cls: "cell-split",
        html: `<span class="split-badge">${escapeHtml(String(value ?? "").toUpperCase())}</span>`,
      };
    case "period":
      return {
        cls: "cell-period",
        html: `<span class="period-text">${escapeHtml(String(value ?? ""))}</span>`,
      };
    case "databar":
    case "databar-primary": {
      const fraction = context.fractions.get(context.rowIndex);
      if (fraction === undefined) {
        return { cls: "cell-text", html: escapeHtml(String(value ?? "")) };
      }
      const color = (spec && spec.color) || "primary";
      const emphasize = kind === "databar-primary" ? "primary" : "normal";
      const rank = context.ranks.get(context.rowIndex) ?? "";
      const tip = `${headerLabel} ${value} · ${rank}`;
      return {
        cls: "cell-databar",
        html: `<span class="databar" data-color="${color}" data-emphasize="${emphasize}" data-tip="${escapeHtml(tip)}" style="--fraction:${fraction.toFixed(4)}">`
          + `<span class="databar-fill"></span>`
          + `<span class="databar-label">${escapeHtml(String(value ?? ""))}</span>`
          + `</span>`,
      };
    }
    case "percent-heat": {
      const heat = context.heatColors.get(context.rowIndex);
      if (heat === undefined) {
        return { cls: "cell-text", html: escapeHtml(String(value ?? "")) };
      }
      const tip = `${headerLabel} ${value}`;
      return {
        cls: "cell-heat",
        html: `<span class="heat-chip" data-tip="${escapeHtml(tip)}" style="--heat:${heat}">${escapeHtml(String(value ?? ""))}</span>`,
      };
    }
    case "psi": {
      const numeric = parseNumeric(value);
      const thresholds = (spec && spec.thresholds) || [0.02, 0.10];
      const tier = psiTier(numeric, thresholds);
      const displayText = value === "BASE" || value === "-" || numeric === null
        ? String(value ?? "")
        : String(value);
      const stripMarker = numeric === null
        ? ""
        : `<i class="psi-marker" style="left:${Math.min(Math.abs(numeric) / 0.20, 1) * 100}%"></i>`;
      const tip = psiTooltipText(numeric, thresholds);
      return {
        cls: "cell-psi",
        html: `<span class="psi-cell" data-tip="${escapeHtml(tip)}">`
          + `<span class="psi-value" data-tier="${tier}">${escapeHtml(displayText)}</span>`
          + `<span class="psi-strip"><span></span><span></span><span></span>${stripMarker}</span>`
          + `</span>`,
      };
    }
    case "text":
    default:
      if (metricHeaderShouldRightAlign(headerLabel) && parseNumeric(value) !== null) {
        return { cls: "cell-number", html: escapeHtml(String(value ?? "")) };
      }
      return { cls: "cell-text", html: escapeHtml(String(value ?? "")) };
  }
}

function metricHeaderShouldRightAlign(headerLabel) {
  const label = String(headerLabel || "").trim();
  if (!label) return false;
  if (/(^id$|id$|编号|月份|日期|时间|参考月|特征|变量|字段|数据集|样本集|分组|类别|等级|区间|范围)/i.test(label)) {
    return false;
  }
  return /^(KS|KS\(%\)|AUC|AUC\(%\)|PSI|IV|样本量|坏样本量|好样本量|逾期率|坏账率|通过率|命中率|缺失率|占比|比例|分数|评分|重要性|Gain|Split|Coverage|Lift|5%头部lift|5%尾部lift)$/i.test(label);
}

function renderMetricTable(table = {}) {
  const layout = table.layout || "table";
  switch (layout) {
    case "kpi_cards":
      return renderKpiCards(table);
    case "trend_table":
      return renderTrendTable(table);
    case "roc_ks_curve":
      return renderRocKsCurve(table);
    case "table":
    default:
      return renderEnhancedTable(table);
  }
}

function renderKpiCards(table = {}, options = {}) {
  const headers = Array.isArray(table.headers) ? table.headers : [];
  const rows = Array.isArray(table.rows) ? table.rows : [];
  const specs = Array.isArray(table.column_specs) ? table.column_specs : [];
  const curves = (options && options.curves) || null;

  const idx = (label) => headers.indexOf(label);
  const idxAny = (...labels) => labels.map(idx).find((index) => index >= 0) ?? -1;
  const splitIdx = idx("数据集");
  const periodIdx = idx("时间范围");
  const ksIdx = idxAny("KS(%)", "KS");
  const aucIdx = idxAny("AUC(%)", "AUC");
  const headLiftIdx = idx("5%头部lift");
  const tailLiftIdx = idx("5%尾部lift");
  const psiIdx = idx("PSI");
  const sampleIdx = idx("样本量");
  const badRateIdx = idx("逾期率");
  const badCountIdx = idx("坏样本量");

  const ksFractions = columnFractions(rows, ksIdx);
  const aucFractions = columnFractions(rows, aucIdx);
  const headLiftFractions = columnFractions(rows, headLiftIdx);
  const tailLiftFractions = columnFractions(rows, tailLiftIdx);

  const psiThresholds = (specs[psiIdx] && specs[psiIdx].thresholds) || [0.02, 0.10];

  const cardHtml = rows.map((row, rowIndex) => {
    const cell = (i) => Array.isArray(row) ? row[i] : "";
    const psiNumeric = parseNumeric(cell(psiIdx));
    const psiDisplay = cell(psiIdx);
    const splitName = String(cell(splitIdx) || "").toLowerCase();
    const curveForSplit = curves ? curves[splitName] : null;
    const rocHtml = curveForSplit
      ? renderRocCard(splitName, curveForSplit)
      : "";
    return [
      `<div class="kpi-card-column">`,
      `<article class="kpi-card">`,
      `  <header class="kpi-card-header">`,
      `    <span class="kpi-card-split">${escapeHtml(String(cell(splitIdx) || "").toUpperCase())}</span>`,
      `    <span class="kpi-card-period">${escapeHtml(String(cell(periodIdx) ?? ""))}</span>`,
      `  </header>`,
      `  <div class="kpi-card-primary" data-tip="${escapeHtml(`KS ${cell(ksIdx)} · ${columnRanks(rows, ksIdx).get(rowIndex) || ""}`)}">`,
      `    <span class="kpi-card-primary-label">KS</span>`,
      `    <span class="kpi-card-primary-value">${escapeHtml(String(cell(ksIdx) ?? ""))}</span>`,
      `    <span class="kpi-card-primary-bar" style="--fraction:${(ksFractions.get(rowIndex) ?? 0).toFixed(4)}"><i></i></span>`,
      `  </div>`,
      `  <div class="kpi-card-rule"></div>`,
      kpiCardRow("AUC", cell(aucIdx), aucFractions.get(rowIndex), rowIndex, aucIdx, rows, "var(--accent)"),
      kpiCardRow("5%头部lift", cell(headLiftIdx), headLiftFractions.get(rowIndex), rowIndex, headLiftIdx, rows, "var(--metric-databar-accent)"),
      kpiCardRow("5%尾部lift", cell(tailLiftIdx), tailLiftFractions.get(rowIndex), rowIndex, tailLiftIdx, rows, "var(--metric-databar-accent)"),
      kpiPsiRow(psiDisplay, psiNumeric, psiThresholds),
      `  <footer class="kpi-card-footer">`,
      `    <span class="kpi-card-footer-cell"><span class="kpi-card-footer-label">样本量</span><span class="kpi-card-footer-value">${escapeHtml(String(cell(sampleIdx) ?? ""))}</span></span>`,
      `    <span class="kpi-card-footer-cell"><span class="kpi-card-footer-label">逾期率</span><span class="kpi-card-footer-value">${escapeHtml(String(cell(badRateIdx) ?? ""))}</span></span>`,
      `    <span class="kpi-card-footer-cell"><span class="kpi-card-footer-label">坏样本</span><span class="kpi-card-footer-value">${escapeHtml(String(cell(badCountIdx) ?? ""))}</span></span>`,
      `  </footer>`,
      `</article>`,
      rocHtml,
      `</div>`,
    ].join("\n");
  }).join("");

  return [
    `<div class="metric-table-wrap" data-metric-key="${escapeHtml(table.key || "")}">`,
    `<div class="kpi-cards" style="--kpi-count:${rows.length}">`,
    cardHtml,
    `</div>`,
    `</div>`,
  ].join("");
}

function kpiCardRow(label, displayValue, fraction, rowIndex, columnIndex, rows, color) {
  const tip = `${label} ${displayValue} · ${columnRanks(rows, columnIndex).get(rowIndex) || ""}`;
  return [
    `<div class="kpi-card-row" data-tip="${escapeHtml(tip)}">`,
    `  <span class="kpi-card-row-label">${escapeHtml(label)}</span>`,
    `  <span class="kpi-card-row-bar" style="--fraction:${(fraction ?? 0).toFixed(4)};--bar-color:${color}"><i></i></span>`,
    `  <span class="kpi-card-row-value">${escapeHtml(String(displayValue ?? ""))}</span>`,
    `</div>`,
  ].join("");
}

function kpiPsiRow(displayValue, numeric, thresholds) {
  const tier = psiTier(numeric, thresholds);
  const tip = psiTooltipText(numeric, thresholds);
  return [
    `<div class="kpi-card-row" data-tip="${escapeHtml(tip)}">`,
    `  <span class="kpi-card-row-label">PSI</span>`,
    `  <span class="psi-strip"><span></span><span></span><span></span>`,
    numeric === null ? "" : `<i class="psi-marker" style="left:${Math.min(Math.abs(numeric) / 0.20, 1) * 100}%"></i>`,
    `  </span>`,
    `  <span class="psi-value kpi-card-row-value" data-tier="${tier}">${escapeHtml(String(displayValue ?? ""))}</span>`,
    `</div>`,
  ].join("");
}

function renderTrendTable(table = {}) {
  const baseHeaders = Array.isArray(table.headers) ? [...table.headers] : [];
  const baseRows = Array.isArray(table.rows) ? table.rows.map((row) => Array.isArray(row) ? [...row] : []) : [];
  const baseSpecs = Array.isArray(table.column_specs) ? [...table.column_specs] : [];

  const ksIdx = baseHeaders.indexOf("KS(%)") >= 0
    ? baseHeaders.indexOf("KS(%)")
    : baseHeaders.indexOf("KS");
  const ksSeries = ksIdx >= 0
    ? baseRows.map((row) => parseNumeric(row[ksIdx])).filter((v) => v !== null)
    : [];
  const ksAll = ksIdx >= 0
    ? baseRows.map((row) => parseNumeric(row[ksIdx]))
    : [];

  const sampleAtIdx = baseHeaders.indexOf("样本量");
  const insertAt = sampleAtIdx >= 0 ? sampleAtIdx + 1 : baseHeaders.length;
  const trendHeaders = [...baseHeaders];
  const trendSpecs = [...baseSpecs];
  trendHeaders.splice(insertAt, 0, "KS 趋势");
  trendSpecs.splice(insertAt, 0, { kind: "trend-spark", __localHtml: true });
  const trendRows = baseRows.map((row, rowIndex) => {
    const copy = [...row];
    const sparkHtml = renderSparklineSvg(ksSeries, ksAll[rowIndex]);
    copy.splice(insertAt, 0, sparkHtml);
    return copy;
  });

  return renderEnhancedTableExplicit({
    key: table.key,
    title: table.title,
    headers: trendHeaders,
    column_specs: trendSpecs,
    rows: trendRows,
  });
}

function renderSparklineSvg(series, currentValue) {
  if (!series || series.length < 2) return "";
  const minV = Math.min(...series);
  const maxV = Math.max(...series);
  const range = Math.max(maxV - minV, 1e-6);
  const W = 120, H = 24, PAD_X = 4, PAD_Y = 4;
  const stepX = (W - 2 * PAD_X) / (series.length - 1);
  const yOf = (v) => H - PAD_Y - ((v - minV) / range) * (H - 2 * PAD_Y);
  const linePath = series.map((v, i) => `${i === 0 ? "M" : "L"}${PAD_X + i * stepX},${yOf(v).toFixed(2)}`).join(" ");
  const baseY = H / 2;
  const points = series.map((v, i) => {
    const cx = PAD_X + i * stepX;
    const isCurrent = currentValue !== null && Math.abs(v - currentValue) < 1e-9;
    return `<circle class="metric-sparkline-point${isCurrent ? " current" : ""}" cx="${cx.toFixed(2)}" cy="${yOf(v).toFixed(2)}" r="${isCurrent ? 3.2 : 2}" data-tip="KS ${v.toFixed(4)}"></circle>`;
  }).join("");
  return `<svg class="metric-sparkline" viewBox="0 0 ${W} ${H}" role="img" aria-label="KS 趋势">`
    + `<line class="metric-sparkline-baseline" x1="${PAD_X}" y1="${baseY}" x2="${W - PAD_X}" y2="${baseY}"></line>`
    + `<path class="metric-sparkline-line" d="${linePath}"></path>`
    + points
    + `</svg>`;
}

function renderRocKsCurve(table = {}) {
  const curves = table.curves || {};
  const splits = ["train", "test", "oot"].filter((split) => curves[split]);
  if (splits.length === 0) {
    return `<div class="metric-table-wrap"><div class="result-summary empty">暂无 ROC&KS 曲线数据</div></div>`;
  }
  const cards = splits.map((split) => renderRocCard(split, curves[split])).join("");
  return [
    `<div class="metric-table-wrap" data-metric-key="${escapeHtml(table.key || "ROC_KS_CURVES")}">`,
    `<div class="roc-grid" style="--roc-count:${splits.length}">${cards}</div>`,
    `</div>`,
  ].join("");
}

function renderRocCard(split, curve) {
  const W = 280, H = 240, PAD = 28;
  const plot = { x: PAD, y: PAD - 4, w: W - PAD - 8, h: H - PAD - 14 };
  const fpr = curve.fpr || [];
  const tpr = curve.tpr || [];
  const ks = curve.ks_curve || [];
  if (fpr.length < 2) {
    return `<div class="roc-card"><div class="roc-card-header"><span class="roc-card-split">${escapeHtml(split)}</span></div><div class="result-summary empty">无曲线数据</div></div>`;
  }
  const xOf = (v) => plot.x + Math.max(0, Math.min(1, v)) * plot.w;
  const yOf = (v) => plot.y + (1 - Math.max(0, Math.min(1, v))) * plot.h;
  const buildPath = (xs, ys) => xs.map((x, i) => `${i === 0 ? "M" : "L"}${xOf(x).toFixed(2)},${yOf(ys[i]).toFixed(2)}`).join(" ");

  const tprPath = buildPath(fpr, tpr);
  const diagonalPath = `M${xOf(0)},${yOf(0)} L${xOf(1)},${yOf(1)}`;
  const ksPath = ks.length === fpr.length ? buildPath(fpr, ks) : "";
  // KS marker sits on the FPR axis: anchor it at fpr[argmax(|ks_curve|)]. Using
  // population_at_ks (a different axis) misplaces the line on imbalanced data.
  const ksArgmax = ks.length
    ? ks.reduce((best, value, i) => (Math.abs(value) > Math.abs(ks[best]) ? i : best), 0)
    : 0;
  const ksMarkerX = xOf(fpr[ksArgmax] ?? 0);

  const gridLines = [0.25, 0.5, 0.75].map((t) =>
    `<line class="roc-grid-line" x1="${xOf(t).toFixed(2)}" y1="${plot.y}" x2="${xOf(t).toFixed(2)}" y2="${plot.y + plot.h}"></line>`
    + `<line class="roc-grid-line" x1="${plot.x}" y1="${yOf(t).toFixed(2)}" x2="${plot.x + plot.w}" y2="${yOf(t).toFixed(2)}"></line>`
  ).join("");

  const xLabels = [0, 0.5, 1].map((t) =>
    `<text class="roc-axis-label" x="${xOf(t).toFixed(2)}" y="${H - 2}" text-anchor="middle" font-size="9">${t}</text>`
  ).join("");
  const yLabels = [0, 0.5, 1].map((t) =>
    `<text class="roc-axis-label" x="${plot.x - 4}" y="${(yOf(t) + 3).toFixed(2)}" text-anchor="end" font-size="9">${t}</text>`
  ).join("");

  return [
    `<div class="roc-card" data-split="${escapeHtml(split)}">`,
    `  <div class="roc-card-header">`,
    `    <span class="roc-card-split">${escapeHtml(split)}</span>`,
    `    <span class="roc-card-ks" data-tip="KS=${(curve.ks ?? 0).toFixed(4)} at population=${(curve.population_at_ks ?? 0).toFixed(2)}">KS ${(curve.ks ?? 0).toFixed(4)}</span>`,
    `  </div>`,
    `  <svg class="roc-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="ROC and KS curves for ${escapeHtml(split)}"`,
    `       data-roc-fpr="${escapeHtml(JSON.stringify(fpr))}" data-roc-tpr="${escapeHtml(JSON.stringify(tpr))}" data-roc-ks="${escapeHtml(JSON.stringify(ks))}"`,
    `       data-roc-plot-x="${plot.x}" data-roc-plot-y="${plot.y}" data-roc-plot-w="${plot.w}" data-roc-plot-h="${plot.h}">`,
    `    ${gridLines}`,
    `    <line class="roc-axis" x1="${plot.x}" y1="${plot.y + plot.h}" x2="${plot.x + plot.w}" y2="${plot.y + plot.h}"></line>`,
    `    <line class="roc-axis" x1="${plot.x}" y1="${plot.y}" x2="${plot.x}" y2="${plot.y + plot.h}"></line>`,
    `    <path class="roc-curve roc-curve-baseline" data-series="baseline" d="${diagonalPath}"></path>`,
    `    <path class="roc-curve roc-curve-tpr" data-series="tpr" d="${tprPath}"></path>`,
    ksPath ? `    <path class="roc-curve roc-curve-ks" data-series="ks" d="${ksPath}"></path>` : "",
    `    <line class="roc-ks-marker" data-series="ks-marker" x1="${ksMarkerX.toFixed(2)}" y1="${plot.y}" x2="${ksMarkerX.toFixed(2)}" y2="${plot.y + plot.h}" data-tip="KS=${(curve.ks ?? 0).toFixed(4)} at population=${(curve.population_at_ks ?? 0).toFixed(2)}"></line>`,
    `    <line class="roc-crosshair roc-crosshair-x" x1="0" y1="0" x2="0" y2="0" style="display:none"></line>`,
    `    <line class="roc-crosshair roc-crosshair-y" x1="0" y1="0" x2="0" y2="0" style="display:none"></line>`,
    `    ${xLabels}`,
    `    ${yLabels}`,
    `  </svg>`,
    `  <div class="roc-legend" role="group" aria-label="切换曲线显示">`,
    `    <button type="button" class="roc-legend-tpr" data-roc-toggle="tpr" aria-pressed="true"><i></i>TPR</button>`,
    `    <button type="button" class="roc-legend-baseline" data-roc-toggle="baseline" aria-pressed="true"><i></i>Random Baseline</button>`,
    `    <button type="button" class="roc-legend-ks" data-roc-toggle="ks" aria-pressed="true"><i></i>KS Curve</button>`,
    `  </div>`,
    `  <div class="roc-readout" data-roc-readout>移动鼠标到图上查看 FPR / TPR / KS</div>`,
    `</div>`,
  ].join("");
}

function attachRocInteractions(rootEl) {
  if (!rootEl) return;
  rootEl.querySelectorAll(".roc-card").forEach((card) => {
    const svg = card.querySelector(".roc-svg");
    const readout = card.querySelector("[data-roc-readout]");
    if (!svg) return;
    const fpr = JSON.parse(svg.getAttribute("data-roc-fpr") || "[]");
    const tpr = JSON.parse(svg.getAttribute("data-roc-tpr") || "[]");
    const ks  = JSON.parse(svg.getAttribute("data-roc-ks")  || "[]");
    const px = Number(svg.getAttribute("data-roc-plot-x"));
    const py = Number(svg.getAttribute("data-roc-plot-y"));
    const pw = Number(svg.getAttribute("data-roc-plot-w"));
    const ph = Number(svg.getAttribute("data-roc-plot-h"));
    const xLine = svg.querySelector(".roc-crosshair-x");
    const yLine = svg.querySelector(".roc-crosshair-y");

    const hideCrosshair = () => {
      xLine.style.display = "none";
      yLine.style.display = "none";
      readout.textContent = "移动鼠标到图上查看 FPR / TPR / KS";
    };

    const onMove = (event) => {
      const rect = svg.getBoundingClientRect();
      const viewBox = svg.viewBox.baseVal;
      const xViewbox = ((event.clientX - rect.left) / rect.width) * viewBox.width;
      const fprVal = (xViewbox - px) / pw;
      if (fprVal < 0 || fprVal > 1) { hideCrosshair(); return; }
      let nearestIdx = 0;
      let nearestDist = Infinity;
      for (let i = 0; i < fpr.length; i++) {
        const d = Math.abs(fpr[i] - fprVal);
        if (d < nearestDist) { nearestDist = d; nearestIdx = i; }
      }
      const xPos = px + fpr[nearestIdx] * pw;
      const yPosTpr = py + (1 - tpr[nearestIdx]) * ph;
      xLine.setAttribute("x1", xPos); xLine.setAttribute("x2", xPos);
      xLine.setAttribute("y1", py); xLine.setAttribute("y2", py + ph);
      xLine.style.display = "";
      yLine.setAttribute("x1", px); yLine.setAttribute("x2", px + pw);
      yLine.setAttribute("y1", yPosTpr); yLine.setAttribute("y2", yPosTpr);
      yLine.style.display = "";
      const ksHere = ks[nearestIdx] ?? Math.abs(tpr[nearestIdx] - fpr[nearestIdx]);
      readout.textContent = `FPR ${fpr[nearestIdx].toFixed(3)}  ·  TPR ${tpr[nearestIdx].toFixed(3)}  ·  KS ${ksHere.toFixed(3)}`;
    };

    svg.addEventListener("mousemove", onMove);
    svg.addEventListener("mouseleave", hideCrosshair);

    card.querySelectorAll("[data-roc-toggle]").forEach((button) => {
      button.addEventListener("click", () => {
        const series = button.getAttribute("data-roc-toggle");
        const pressed = button.getAttribute("aria-pressed") === "true";
        const next = !pressed;
        button.setAttribute("aria-pressed", String(next));
        svg.querySelectorAll(`[data-series="${series}"], [data-series="${series}-marker"]`).forEach((el) => {
          el.style.display = next ? "" : "none";
        });
      });
    });
  });
}

let metricTooltipAttached = false;

function attachMetricTooltip(rootEl) {
  if (metricTooltipAttached || !rootEl) return;
  const tooltip = document.getElementById("metricTooltip");
  if (!tooltip) return;
  metricTooltipAttached = true;
  let currentTarget = null;
  const positionTooltip = (el, event) => {
    const pad = 12;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    const rect = el.getBoundingClientRect();
    if (x + rect.width > window.innerWidth) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight) y = event.clientY - rect.height - pad;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
  };
  const show = (target, event) => {
    const tip = target.getAttribute("data-tip");
    if (!tip) return;
    currentTarget = target;
    tooltip.textContent = tip;
    tooltip.hidden = false;
    positionTooltip(tooltip, event);
  };
  const hide = () => {
    currentTarget = null;
    tooltip.hidden = true;
  };
  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest("#metricPreview [data-tip]");
    if (target) show(target, event);
  });
  document.addEventListener("mousemove", (event) => {
    if (currentTarget && currentTarget.contains(event.target)) {
      positionTooltip(tooltip, event);
    }
  });
  document.addEventListener("mouseout", (event) => {
    if (!currentTarget) return;
    const next = event.relatedTarget;
    if (!next || !currentTarget.contains(next)) hide();
  });
  document.addEventListener("scroll", hide, { passive: true, capture: true });
}

function renderEnhancedTable(table = {}) {
  return renderEnhancedTableExplicit(table);
}

function renderEnhancedTableExplicit(table) {
  const headers = Array.isArray(table.headers) ? table.headers : [];
  const rows = Array.isArray(table.rows) ? table.rows : [];
  const specs = Array.isArray(table.column_specs) ? table.column_specs : [];
  const columnCount = headers.length;

  const fractionsByColumn = new Map();
  const heatByColumn = new Map();
  const ranksByColumn = new Map();
  for (let i = 0; i < columnCount; i++) {
    const kind = (specs[i] && specs[i].kind) || "text";
    if (kind === "databar" || kind === "databar-primary") {
      fractionsByColumn.set(i, columnFractions(rows, i));
      ranksByColumn.set(i, columnRanks(rows, i));
    }
    if (kind === "percent-heat") {
      heatByColumn.set(i, columnHeatColors(rows, i));
    }
  }

  // Columns whose text-only cells should hug the left edge (e.g.
  // 特征 in 特征重要性). Numeric primitives stay centered regardless.
  const leftAlignCols = new Set();
  headers.forEach((header, i) => {
    if (header === "特征") leftAlignCols.add(i);
  });

  const bodyRows = rows.length === 0
    ? `<tr class="metric-table-empty"><td colspan="${Math.max(columnCount, 1)}">暂无数据</td></tr>`
    : rows.map((row, rowIndex) => {
        const cells = [];
        for (let columnIndex = 0; columnIndex < columnCount; columnIndex++) {
          const value = Array.isArray(row) ? row[columnIndex] : "";
          const spec = specs[columnIndex] || { kind: "text" };
          const cell = renderCellByKind(spec, value, {
            rowIndex,
            columnIndex,
            headerLabel: headers[columnIndex] ?? "",
            fractions: fractionsByColumn.get(columnIndex) || new Map(),
            heatColors: heatByColumn.get(columnIndex) || new Map(),
            ranks: ranksByColumn.get(columnIndex) || new Map(),
          });
          const alignAttr = leftAlignCols.has(columnIndex) && (spec.kind || "text") === "text"
            ? ' data-align="left"'
            : "";
          cells.push(`<td class="${cell.cls}"${alignAttr}>${cell.html}</td>`);
        }
        return `<tr>${cells.join("")}</tr>`;
      }).join("");

  return [
    `<div class="metric-table-wrap" data-metric-key="${escapeHtml(table.key || "")}">`,
    '<div class="metric-table-scroll">',
    '<table class="metric-table metric-table-hoverable">',
    '<thead><tr>',
    ...headers.map((header, i) => {
      const alignAttr = leftAlignCols.has(i) ? ' data-align="left"' : "";
      return `<th${alignAttr}>${escapeHtml(header)}</th>`;
    }),
    '</tr></thead>',
    `<tbody>${bodyRows}</tbody>`,
    "</table>",
    "</div>",
    "</div>",
  ].join("");
}

function renderLegacyTable(table = {}) {
  const headers = Array.isArray(table.headers) ? table.headers : [];
  const rows = Array.isArray(table.rows) ? table.rows : [];
  const columnCount = Math.max(
    headers.length,
    ...rows.map((row) => (Array.isArray(row) ? row.length : 0)),
    1,
  );
  const bodyRows = rows.length > 0
    ? rows.map((row) => renderMetricTableRow(row, columnCount))
    : [`<tr class="metric-table-empty"><td colspan="${columnCount}">暂无数据</td></tr>`];
  return [
    `<div class="metric-table-wrap" data-metric-key="${escapeHtml(table.key || "")}">`,
    `<div class="metric-table-caption">${escapeHtml(table.title || "指标明细")}</div>`,
    '<div class="metric-table-scroll">',
    '<table class="metric-table">',
    "<thead><tr>",
    ...Array.from(
      { length: columnCount },
      (_, index) => `<th>${escapeHtml(headers[index] ?? "")}</th>`,
    ),
    "</tr></thead>",
    `<tbody>${bodyRows.join("")}</tbody>`,
    "</table>",
    "</div>",
    "</div>",
  ].join("");
}

function renderMetricTableRow(row, columnCount) {
  const cells = Array.from({ length: columnCount }, (_, index) => {
    const value = Array.isArray(row) ? row[index] : "";
    return `<td>${escapeHtml(value === null || value === undefined ? "" : String(value))}</td>`;
  });
  return `<tr>${cells.join("")}</tr>`;
}

function currentMetricPreviewHasValues(taskId = selectedTaskId) {
  return lastMetricValuesTaskId === taskId && lastMetricTableSections.length > 0;
}

function roleCounts(artifacts) {
  return artifacts.reduce((counts, artifact) => {
    counts[artifact.role] = (counts[artifact.role] || 0) + 1;
    return counts;
  }, {});
}

function scanCheckTone(status) {
  if (status === "success") return "success";
  if (status === "warning") return "warning";
  if (status === "error") return "danger";
  return "";
}

function renderScanResult(result, notebookCells = []) {
  const artifacts = result.artifacts || [];
  const checks = result.checks || [];
  const counts = roleCounts(artifacts);
  const materialChecks = requiredMaterialRoles.map(({ role, label }) => {
    const found = counts[role] || 0;
    const tone = found === 0 ? "danger" : found > 1 ? "warning" : "success";
    const text = found === 0 ? "缺失" : found > 1 ? `${found} 个候选` : "已识别";
    return `<span class="pill ${tone}">${escapeHtml(label)} · ${escapeHtml(text)}</span>`;
  }).join("");
  const preflightChecks = checks.length
    ? [
        '<div class="preflight-check-list" aria-label="扫描前置检查">',
        ...checks.map((check) => {
          const tone = scanCheckTone(check.status);
          const contractClass = check.id === "notebook_contract" ? " notebook-contract-check" : "";
          const statusText = check.status === "error"
            ? "异常"
            : check.status === "warning"
              ? "提示"
              : "通过";
          return [
            `<div class="preflight-check-item ${tone}${contractClass}" data-check-id="${escapeHtml(check.id || "")}">`,
            `<span class="pill ${tone}">${escapeHtml(statusText)}</span>`,
            `<strong>${escapeHtml(check.label || check.id || "检查项")}</strong>`,
            `<small>${escapeHtml(check.message || "")}</small>`,
            "</div>",
          ].join("");
        }),
        "</div>",
      ].join("")
    : "";
  $("scanSummary").className = "result-summary";
  $("scanSummary").innerHTML = [
    `<strong>识别到 ${artifacts.length} 个材料文件。</strong>`,
    `<div class="chip-row">${materialChecks}</div>`,
    preflightChecks,
  ].join("");
  updateAgentScanSectionVisibility();
  renderNotebookSteps(result.notebook_steps || [], result.notebook_cells || notebookCells);
}

function renderValidationResult(result) {
  if (result?.status) setActionStatus("Notebook 已提交执行。", "busy");
}

function evidenceEmpty(id, message) {
  const element = $(id);
  if (!element) return;
  element.className = "result-summary empty";
  element.textContent = message;
}

function resetEvidenceSummaries() {
  latestNotebookSteps = [];
  evidenceEmpty("reproducibilitySummary", "暂无分数一致性证据，运行完建模代码后展示结果");
  resetReproducibilityRenderSignatures();
  renderWorkflowStepper();
}

function normalizeNotebookSteps(notebookSteps = [], notebookCells = []) {
  const steps = Array.isArray(notebookSteps) ? notebookSteps : [];
  const cells = Array.isArray(notebookCells) ? notebookCells : [];
  if (!cells.length) return steps;
  const cellsByIndex = new Map(cells.map((cell) => [Number(cell?.cell_index), cell]));
  return steps.map((step) => {
    const id = String(step?.id || "");
    const cellIndexes = Array.isArray(step?.cell_indexes) ? step.cell_indexes : [];
    if (!step?.system || !id.startsWith("system-") || cellIndexes.length <= 1) return step;

    let latestCell = null;
    let latestCellIndex = null;
    let latestSourcePreview = null;
    cellIndexes.forEach((cellIndex, index) => {
      const numericIndex = Number(cellIndex);
      if (!Number.isFinite(numericIndex)) return;
      const cell = cellsByIndex.get(numericIndex);
      if (!cell || (cell.step_id && cell.step_id !== id)) return;
      if (latestCellIndex !== null && numericIndex < latestCellIndex) return;
      latestCell = cell;
      latestCellIndex = numericIndex;
      latestSourcePreview = Array.isArray(step.source_previews) ? step.source_previews[index] : null;
    });
    if (!latestCell || latestCellIndex === null) return step;

    const normalized = {
      ...step,
      status: latestCell.status || step.status,
      started_at: latestCell.started_at ?? step.started_at,
      ended_at: latestCell.ended_at ?? step.ended_at,
      elapsed_seconds: latestCell.elapsed_seconds ?? null,
      cell_count: 1,
      cell_indexes: [latestCellIndex],
    };
    if (latestSourcePreview !== null && latestSourcePreview !== undefined) {
      normalized.source_previews = [latestSourcePreview];
    }
    return normalized;
  });
}

function renderNotebookSteps(notebookSteps = [], notebookCells = []) {
  latestNotebookSteps = mergePendingSystemSteps(normalizeNotebookSteps(notebookSteps, notebookCells));
  renderReproducibilitySectionVisibility();
  renderMetricSectionVisibility();
  renderWorkflowStepper();
}

function formatScoreValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toFixed(6);
}

function reproducibilityStatusLabel(status) {
  if (status === "pass") return "一致";
  if (status === "fail") return "不一致";
  if (status === "review") return "需复核";
  return status || "未知";
}

function reproducibilityStatusClass(status) {
  if (status === "pass") return "repro-status-pass";
  if (status === "fail") return "repro-status-fail";
  if (status === "review") return "repro-status-review";
  return "";
}

function roundedScoresMatch(left, right, decimals) {
  const leftNumber = Number(left);
  const rightNumber = Number(right);
  if (!Number.isFinite(leftNumber) || !Number.isFinite(rightNumber)) return false;
  return leftNumber.toFixed(decimals) === rightNumber.toFixed(decimals);
}

function precisionConsistencyTier(rate) {
  if (rate >= 99.5) return "exact";
  if (rate >= 90) return "strong";
  if (rate >= 50) return "fair";
  return "weak";
}

function buildPrecisionConsistencyBars(rows = []) {
  const total = rows.length;
  const bars = [];
  for (let decimals = 1; decimals <= 6; decimals += 1) {
    const matchCount = rows.filter((row) => (
      roundedScoresMatch(row.score_code_model, row.score_submitted_pmml, decimals)
    )).length;
    const rate = total > 0 ? (matchCount / total) * 100 : 0;
    bars.push({
      decimals,
      matchCount,
      total,
      rate,
      tier: precisionConsistencyTier(rate),
    });
  }
  return bars;
}

function formatPercentValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0%";
  return `${number.toFixed(1)}%`;
}

// Compact label for the bar tops: drop the decimal at 100% so it fits a narrow
// column; the exact one-decimal value still shows in the hover tooltip.
function formatPrecisionLabel(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0%";
  if (number >= 99.95) return "100%";
  return `${number.toFixed(1)}%`;
}

function renderPrecisionConsistencyChart(rows = [], options = {}) {
  if (!rows.length) return "";
  const animationAttribute = options.animate === false ? ' data-animation="none"' : "";
  const bars = buildPrecisionConsistencyBars(rows);
  const barItems = bars.map((bar, index) => {
    const height = bar.rate > 0 ? Math.max(2, Math.min(100, bar.rate)) : 0;
    const title = `${bar.decimals} 位小数：${bar.matchCount}/${bar.total} 行一致（${formatPercentValue(bar.rate)}）`;
    return [
      `<div class="score-precision-bar-item" data-tier="${bar.tier}"${bar.rate > 0 ? "" : ' data-empty="true"'} title="${escapeHtml(title)}">`,
      `<span class="score-precision-value">${escapeHtml(formatPrecisionLabel(bar.rate))}</span>`,
      '<span class="score-precision-bar-track" aria-hidden="true">',
      `<span class="score-precision-bar" style="height: ${height}%; --bar-index: ${index}"></span>`,
      "</span>",
      `<span class="score-precision-label">${bar.decimals}位</span>`,
      "</div>",
    ].join("");
  });
  return [
    `<div class="score-precision-chart"${animationAttribute}>`,
    '<div class="score-precision-chart-head">',
    "<strong>四舍五入一致率</strong>",
    `<span class="score-precision-meta">${escapeHtml(rows.length)} 行 · 保留小数位越多越严格</span>`,
    "</div>",
    '<div class="score-precision-plot">',
    '<div class="score-precision-axis" aria-hidden="true">',
    "<span>100%</span>",
    "<span>75%</span>",
    "<span>50%</span>",
    "<span>25%</span>",
    "<span>0%</span>",
    "</div>",
    `<div class="score-precision-bars">${barItems.join("")}</div>`,
    "</div>",
    "</div>",
  ].join("");
}

function reproducibilityEvidenceSignature(reproducibility = {}, summary = {}, rows = []) {
  return JSON.stringify({
    status: summary.status || "",
    sample_size: reproducibility.sample_size ?? null,
    seed: reproducibility.seed ?? null,
    mismatch_count: summary.mismatch_count ?? null,
    max_abs_diff: summary.max_abs_diff ?? null,
    rows: rows.map((row) => ({
      row_index: row.row_index ?? null,
      score_code_model: row.score_code_model ?? null,
      score_submitted_pmml: row.score_submitted_pmml ?? null,
      abs_diff: row.abs_diff ?? null,
      matched: row.matched ?? null,
    })),
  });
}

function currentReproducibilityTaskId() {
  return selectedTaskId || "unselected";
}

function renderReproducibilityEvidence(reproducibility = {}) {
  const summary = reproducibility?.summary || {};
  const rows = Array.isArray(reproducibility?.rows) ? reproducibility.rows : [];
  const element = $("reproducibilitySummary");
  const taskId = currentReproducibilityTaskId();
  if (!summary || Object.keys(summary).length === 0) {
    // While polling an active run, evidence payloads may transiently arrive
    // empty between populated ones (different backend writers update at
    // different times). Don't clobber a chart we've already rendered for
    // THIS task — that would let the next populated poll re-trigger the
    // precision-bar CSS entry animation, causing the bars to "keep
    // bouncing" each second.
    if (
      renderSignatures.reproducibilityEvidence
      && renderSignatures.reproducibilityTaskId === taskId
    ) {
      return;
    }
    evidenceEmpty("reproducibilitySummary", "暂无分数一致性证据，运行完建模代码后展示结果");
    resetReproducibilityRenderSignatures();
    return;
  }
  const evidenceSignature = reproducibilityEvidenceSignature(reproducibility, summary, rows);
  if (
    renderSignatures.reproducibilityTaskId === taskId
    && renderSignatures.reproducibilityEvidence === evidenceSignature
  ) {
    return;
  }
  // Animation policy: play the precision-bar entry animation only on the
  // FIRST populated render for a given task. Subsequent rebuilds caused by
  // real data drift (rows changed, summary updated) still rebuild the DOM
  // but with `data-animation="none"` so the bars do not visually replay.
  const shouldAnimatePrecisionChart = renderSignatures.reproducibilityAnimatedTaskId !== taskId;
  const maxDiff = rows.reduce((current, row) => {
    const diff = row.abs_diff === null || row.abs_diff === undefined ? Number.NaN : Number(row.abs_diff);
    return Number.isFinite(diff) ? Math.max(current, diff) : current;
  }, 0);
  const rowLimit = 10;
  const rowItems = rows.slice(0, rowLimit).map((row) => {
    const diff = row.abs_diff === null || row.abs_diff === undefined ? Number.NaN : Number(row.abs_diff);
    const diffWidth = maxDiff > 0 && Number.isFinite(diff)
      ? Math.max(2, Math.min(100, (diff / maxDiff) * 100))
      : 0;
    return [
      `<div class="score-compare-row ${row.matched ? "matched" : "mismatched"}">`,
      `<span>${escapeHtml(row.row_index ?? "-")}</span>`,
      `<strong>${escapeHtml(formatScoreValue(row.score_code_model))}</strong>`,
      `<strong>${escapeHtml(formatScoreValue(row.score_submitted_pmml))}</strong>`,
      '<span class="score-diff-cell">',
      `<span>${escapeHtml(formatScoreValue(row.abs_diff))}</span>`,
      '<span class="score-diff-track" aria-hidden="true">',
      `<span class="score-diff-bar" style="width: ${diffWidth}%"></span>`,
      "</span>",
      "</span>",
      "</div>",
    ].join("");
  });
  const rowsHtml = rows.length
    ? [
        '<div class="score-compare-list">',
        '<div class="score-compare-row score-compare-head">',
        "<span>行号</span>",
        "<span>代码模型分</span>",
        "<span>PMML 分</span>",
        "<span>绝对差</span>",
        "</div>",
        ...rowItems,
        rows.length > rowLimit ? `<small>仅展示前 ${rowLimit} 行，共 ${rows.length} 行。</small>` : "",
        "</div>",
      ].join("")
    : '<div class="result-summary empty">暂无明细行。</div>';
  const precisionChartHtml = renderPrecisionConsistencyChart(rows, {
    animate: shouldAnimatePrecisionChart,
  });
  const statusClass = ["summary-item", reproducibilityStatusClass(summary.status)]
    .filter(Boolean)
    .join(" ");
  element.className = "result-summary";
  element.innerHTML = [
    '<div class="summary-grid">',
    `<div class="${statusClass}"><span>状态</span><strong>${escapeHtml(reproducibilityStatusLabel(summary.status))}</strong></div>`,
    `<div class="summary-item"><span>抽样行数</span><strong>${escapeHtml(reproducibility.sample_size ?? "-")}</strong></div>`,
    `<div class="summary-item"><span>6位小数不一致条数</span><strong>${escapeHtml(summary.mismatch_count ?? 0)}</strong></div>`,
    `<div class="summary-item"><span>最大绝对差</span><strong>${escapeHtml(formatScoreValue(summary.max_abs_diff))}</strong></div>`,
    `<div class="summary-item"><span>随机种子</span><strong>${escapeHtml(reproducibility.seed ?? "-")}</strong></div>`,
    "</div>",
    precisionChartHtml,
    rowsHtml,
  ].join("");
  renderSignatures.reproducibilityTaskId = taskId;
  renderSignatures.reproducibilityEvidence = evidenceSignature;
  if (shouldAnimatePrecisionChart) {
    renderSignatures.reproducibilityAnimatedTaskId = taskId;
  }
}

function renderEvidence(evidence = {}) {
  renderReproducibilitySectionVisibility();
  if (evidence.scan && Object.keys(evidence.scan).length > 0) {
    renderScanResult(evidence.scan, evidence.notebook_cells || []);
  } else {
    renderNotebookSteps(evidence.notebook_steps || [], evidence.notebook_cells || []);
  }
  renderReproducibilityEvidence(evidence.reproducibility || {});
  if (selectedTaskIsAgentMode()) {
    lastAgentRenderSignature = null;
    renderAgentConversation();
  }
}

async function loadTaskEvidence(taskId = selectedTaskId) {
  if (!taskId) {
    resetEvidenceSummaries();
    return;
  }
  try {
    const evidence = await api(`/api/tasks/${taskId}/evidence`);
    if (selectedTaskId !== taskId) return;
    renderEvidence(evidence || {});
  } catch (_) {
    if (selectedTaskId === taskId && !notebookReproducibilityComplete(selectedTask)) {
      resetEvidenceSummaries();
    }
  }
}

function renderActionError(actionId, message) {
  const summaryId = {
    scan: "scanSummary",
    notebook: "reproducibilitySummary",
  }[actionId];
  if (!summaryId) return;
  $(summaryId).className = "result-summary error";
  $(summaryId).innerHTML = `<strong>操作失败。</strong><span>${escapeHtml(message)}</span>`;
  if (actionId === "scan") updateAgentScanSectionVisibility();
}

function scanSummaryHasResult() {
  const scanSummary = $("scanSummary");
  return Boolean(scanSummary && !scanSummary.classList.contains("empty") && scanSummary.textContent.trim());
}

function updateAgentScanSectionVisibility() {
  const scanSection = $("scanSection");
  if (!scanSection) return;
  // Driver tasks (data_join / feature / modeling) never use the validation
  // scan→notebook→metrics flow — they drive everything through the conversation +
  // plan rail. Hide the scan section entirely so a manual driver task doesn't show
  // a dead "点击扫描材料开始" prompt with no scan button to click.
  if (taskUsesPlanRail(selectedTask)) {
    scanSection.classList.add("hidden");
    return;
  }
  if (!selectedTaskIsAgentMode()) {
    scanSection.classList.remove("hidden");
    return;
  }
  const hasScanResult = scanSummaryHasResult();
  scanSection.classList.toggle("hidden", !hasScanResult);
}

function updateAgentReportSectionVisibility() {
  const reportSection = $("reportSection");
  if (!reportSection) return;
  const hasReportMessages = ["agentReportLeadMessages", "agentReportMessages"]
    .some((targetId) => Boolean($(targetId)?.children.length));
  reportSection.setAttribute("aria-hidden", hasReportMessages ? "false" : "true");
}

function renderStoredStateSummaries() {
  renderReproducibilitySectionVisibility();
  renderMetricSectionVisibility();
  const scanEmptyText = selectedTaskId ? "点击\"扫描材料\"开始" : "选择任务后点击\"扫描材料\"开始";
  $("scanSummary").className = "result-summary empty";
  $("scanSummary").textContent = selectedTaskIsAgentMode() ? "" : scanEmptyText;
  updateAgentScanSectionVisibility();
  resetEvidenceSummaries();
  updateAgentReportSectionVisibility();
  renderTaskSnapshot();
}

function renderAll() {
  renderCurrentTask({ force: true });
  renderReproducibilitySectionVisibility();
  renderMetricSectionVisibility();
  renderWorkflowStepper({ force: true });
  renderTaskList();
  renderSettingsState();
  renderAgentConversation();
  renderPetState();
  updateAgentSendDisabled();
}

// Lighter-weight repaint for the per-second polling loop: each renderer's
// own signature guard decides whether to touch the DOM, so unchanged regions
// keep their existing nodes (and animations) intact.
function renderChangedValidationViews() {
  renderCurrentTask();
  renderReproducibilitySectionVisibility();
  renderMetricSectionVisibility();
  renderWorkflowStepper();
  renderTaskList();
  renderSettingsState();
  renderAgentConversation();
  renderPetState();
  updateAgentSendDisabled();
}

function renderAgentModelOptions() {
  const select = $("agentModelSelect");
  if (!select) return;
  const enabledModels = llmSettings.enabled_models || [];
  const preferred = agentPreferredModelId(enabledModels);
  const signature = JSON.stringify({
    default_model_id: llmSettings.default_model_id || "",
    models: enabledModels.map((model) => ({
      model_id: model.model_id || "",
      display_name: model.display_name || "",
      model_name: model.model_name || "",
    })),
  });
  if (select.dataset.agentModelOptionsSignature === signature) {
    select.disabled = enabledModels.length === 0;
    const preferredStillAvailable = Array.from(select.options).some((option) => option.value === preferred);
    if (document.activeElement !== select && preferred && preferredStillAvailable && select.value !== preferred) {
      select.value = preferred;
    }
    return;
  }
  select.dataset.agentModelOptionsSignature = signature;
  select.innerHTML = "";
  if (enabledModels.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "未配置大模型";
    select.appendChild(option);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  enabledModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.display_name || model.model_name || model.model_id;
    if (option.value === preferred) option.selected = true;
    select.appendChild(option);
  });
}

function agentPreferredModelId(enabledModels = llmSettings.enabled_models || []) {
  const enabledIds = new Set(
    enabledModels
      .map((model) => model.model_id || "")
      .filter(Boolean),
  );
  if (agentSelectedModelId && enabledIds.has(agentSelectedModelId)) return agentSelectedModelId;
  if (llmSettings.default_model_id && enabledIds.has(llmSettings.default_model_id)) {
    return llmSettings.default_model_id;
  }
  return enabledModels.find((model) => model.model_id)?.model_id || "";
}

function setAgentComposerNotice(message = "", kind = "info") {
  const notice = $("agentComposerNotice");
  if (!notice) return;
  if (!message) clearActionStatusOverride();
  notice.textContent = message || "";
  notice.className = `agent-composer-notice ${message ? kind : ""}`.trim();
  notice.setAttribute("role", kind === "error" ? "alert" : "status");
  notice.setAttribute("aria-live", kind === "error" ? "assertive" : "polite");
  requestAnimationFrame(syncAgentComposerClearance);
}

function agentModelUnavailableMessage() {
  const enabledModels = llmSettings.enabled_models || [];
  if (enabledModels.length === 0) return AGENT_NO_ENABLED_MODEL_MESSAGE;
  const selectedId = $("agentModelSelect")?.value || "";
  if (!selectedId) return AGENT_NO_SELECTED_MODEL_MESSAGE;
  return "";
}

function agentModelConfigurationErrorMessage(error) {
  const message = String(error?.message || error || "");
  if (!message) return "";
  if (message.includes("请先在设置中配置至少一个启用的大模型")) {
    return AGENT_NO_ENABLED_MODEL_MESSAGE;
  }
  if (message.includes("当前选择的模型不可用")) {
    return "当前选择的模型不可用，请重新选择或到设置中检查配置。";
  }
  if (message.includes("当前选择的模型缺少 API Base URL 或模型名")) {
    return "当前选择的模型缺少 API Base URL 或模型名，请到设置中补全配置。";
  }
  return "";
}

function showAgentModelGuidance(message) {
  if (!message) return false;
  setAgentComposerNotice(message, "error");
  setActionStatusOverride(message, "error");
  $("agentModelSelect")?.focus();
  return true;
}

function renderAgentEffortPreference() {
  const select = $("agentEffortSelect");
  if (!select) return;
  agentSelectedEffort = normalizeAgentEffort(agentSelectedEffort);
  if (select.value !== agentSelectedEffort) select.value = agentSelectedEffort;
}

function renderAgentAcceptanceModePreference() {
  const select = $("agentAcceptanceModeSelect");
  if (!select) return;
  agentAcceptanceMode = normalizeAgentAcceptanceMode(agentAcceptanceMode);
  if (select.value !== agentAcceptanceMode) select.value = agentAcceptanceMode;
  const chip = select.closest(".agent-composer-acceptance");
  if (chip) chip.dataset.acceptanceMode = agentAcceptanceMode;
  // Relabel the auto-accept option per task type so the chip reads naturally for the
  // current flow (自动拼接/分析/建模) instead of always "自动审查".
  const autoOption = select.querySelector('option[value="auto_accept"]');
  if (autoOption) autoOption.textContent = autoAcceptLabel(selectedTask?.task_type);
}

function autoAcceptLabel(taskType) {
  switch (taskType) {
    case "data_join":
      return "自动拼接";
    case "feature_analysis":
      return "自动分析";
    case "modeling":
      return "自动建模";
    default:
      return "自动审查";
  }
}

function requestAgentConversationScrollToLatest() {
  if (!selectedTaskIsAgentMode()) return;
  if (suppressAgentAutoScrollTaskId === selectedTaskId) return;
  if (!agentAutoScrollFollows) return;
  const scrollContent = $("resultScrollContent");
  if (!scrollContent) return;
  if (agentAutoScrollFrame !== null) {
    window.cancelAnimationFrame(agentAutoScrollFrame);
  }
  agentAutoScrollFrame = window.requestAnimationFrame(() => {
    agentAutoScrollFrame = null;
    scrollContent.scrollTo({ top: scrollContent.scrollHeight, behavior: "auto" });
    if (typeof scheduleTaskHeroGlassState === "function") scheduleTaskHeroGlassState();
  });
}

function renderAgentConversation() {
  const panel = $("agentConversationPanel");
  const composer = $("agentComposer");
  const workspace = $("resultWorkspace");
  if (!panel) return;
  const isAgent = selectedTaskIsAgentMode();
  // Driver tasks (data_join / feature / modeling) show the same conversation +
  // controls in BOTH modes. Manual = the user operates the controls (no free-text
  // composer, no LLM); agent = an LLM operates them + free-text composer.
  const showConversation = isAgent || taskUsesPlanRail(selectedTask);
  panel.classList.toggle("hidden", !showConversation);
  panel.setAttribute("aria-hidden", showConversation ? "false" : "true");
  composer?.classList.toggle("hidden", !isAgent);
  composer?.setAttribute("aria-hidden", isAgent ? "false" : "true");
  workspace?.classList.toggle("agent-composer-active", isAgent);
  renderAgentAcceptanceModePreference();
  renderAgentModelOptions();
  renderAgentEffortPreference();
  requestAnimationFrame(syncAgentComposerClearance);
  panel.classList.remove("driver-analysis-mode");
  panel.setAttribute("aria-label", "Agent 对话");
  // Manual mode for a driver task is a TOOL, not a conversation: render the step
  // outputs as analysis panels (no speaker labels / chat bubbles) and put the gate
  // confirm controls in the step rail — exactly like 模型验证 manual mode. Only agent
  // mode is a genuine LLM conversation; keeping manual mode conversation-free is what
  // proves the agent-mode dialogue isn't pre-written.
  if (showConversation && !isAgent && taskUsesPlanRail(selectedTask)) {
    renderDriverManualAnalysis(agentMessages);
    planRailController.resetFetchThrottle(selectedTaskId);
    renderWorkflowStepper({ force: true });
    return;
  }
  if (!showConversation) {
    agentMessages = [];
    lastAgentRenderSignature = null;
    lastAgentStructuralSignature = null;
    resetAgentTypingState();
    clearAgentStageMessages();
    restoreResultScrollDefaultOrder();
    return;
  }
  // Polling re-renders the whole app every second. Only rebuild the transcript
  // DOM when the messages actually changed, so the entry animation does not
  // re-fire each tick and in-progress draft edits are never wiped.
  const visibleStages = agentTimelineVisibleStages();
  const displayedMessages = agentReportMessagesForDisplay(agentMessages);
  const structuralSignature = agentStructuralSignature(displayedMessages, visibleStages);
  const signature = JSON.stringify({
    messages: agentMessages,
    visibleStages,
  });
  if (signature === lastAgentRenderSignature) return;
  if (
    lastAgentStructuralSignature !== null
    && structuralSignature === lastAgentStructuralSignature
    && updateAgentMessageContentsInPlace(displayedMessages)
  ) {
    // Fast path: structural layout unchanged (typewriter tick / streaming
    // delta). We only patched the content of the affected messages, so the
    // metric-section bars and other animated descendants are not moved or
    // re-rendered, which avoids the flicker observed during agent streaming.
    lastAgentRenderSignature = signature;
    requestAgentConversationScrollToLatest();
    return;
  }
  lastAgentRenderSignature = signature;
  lastAgentStructuralSignature = structuralSignature;
  // Snapshot the live preview HTML for any new rerun trigger BEFORE the
  // upcoming new run overwrites #metricPreview / #scanSummary / etc. This
  // keeps every previous run's chart visible at its chronological position.
  freezeAgentSectionSnapshotsForReruns();
  clearAgentStageMessages();
  renderAgentTimeline(displayedMessages);
  requestAgentConversationScrollToLatest();
  // The conversation just changed (a driver turn likely created/advanced the
  // plan). Plan-rail tasks have no validation poll tick to refresh the right
  // rail, so force a fresh plan fetch + re-render here (only on real changes,
  // since this is the post-signature full-rebuild path).
  if (taskUsesPlanRail(selectedTask)) {
    planRailController.resetFetchThrottle(selectedTaskId);
    renderWorkflowStepper({ force: true });
  }
}

function agentStructuralSignature(messages = [], visibleStages = []) {
  // Anything that changes message COUNT, ORDER, stage assignment, role,
  // streaming/thinking state, or label visibility forces a full timeline
  // rebuild. Pure content edits (typewriter tick) leave this signature
  // unchanged and take the fast path.
  let previousAssistantLabel = "";
  const skeleton = messages.map((message) => {
    const role = message?.role === "user" ? "user" : "assistant";
    const label = role === "user" ? "" : agentStageLabel(message?.stage);
    const hideMeta = Boolean(label && label === previousAssistantLabel);
    previousAssistantLabel = label || previousAssistantLabel;
    const metadata = message?.metadata || {};
    return {
      id: message?.id || "",
      role,
      stage: message?.stage || "",
      label,
      hideMeta,
      streaming: agentMessageIsStreaming(message),
      thinking: agentMessageIsThinking(message),
      // Optimistic placeholders and chat metadata flags can change the bucket
      // structure (e.g. report confirmation), so include them.
      flags: {
        optimistic: Boolean(metadata.optimistic),
        awaiting_confirmation: Boolean(metadata.awaiting_confirmation),
        awaiting_next_stage: metadata.awaiting_next_stage || "",
        intent: metadata.intent || "",
        tool_call_name: metadata.tool_call?.name || "",
        memory_references: Array.isArray(metadata.memory_references)
          ? metadata.memory_references.map((reference) => [
            reference.id || "",
            reference.memory_type || "",
            reference.source_task_id || "",
            reference.confidence ?? "",
            reference.use_reason || "",
          ])
          : [],
      },
    };
  });
  return JSON.stringify({ skeleton, visibleStages });
}

function updateAgentMessageContentsInPlace(messages = []) {
  return updateAgentMessageContentsInPlaceDom(messages, {
    getElementById: $,
    isStreaming: agentMessageIsStreaming,
    isThinking: agentMessageIsThinking,
    thinkingHtml: agentThinkingHtml,
    visibleContent: agentVisibleContent,
    formatMessageContent: formatAgentMessageContent,
    memoryReferencesHtml: agentMemoryReferencesHtml,
  });
}

function agentFrozenStageConfig(stage) {
  if (stage === "scan") {
    return {
      sectionId: "scanSection",
      contentId: "scanSummary",
      headingHtml: "<h3>材料识别</h3>",
      label: "材料识别（历史）",
    };
  }
  if (stage === "reproducibility") {
    return {
      sectionId: "notebookSection",
      contentId: "reproducibilitySummary",
      headingHtml: "<h3>分数一致性</h3>",
      label: "分数一致性（历史）",
    };
  }
  if (stage === "metrics") {
    return {
      sectionId: "metricSection",
      contentId: "metricPreview",
      headingHtml: "<h3>指标概览</h3>",
      label: "指标概览（历史）",
    };
  }
  return null;
}

function freezeAgentSectionSnapshotsForReruns() {
  // Capture the live preview HTML for any rerun message we have not yet
  // frozen. Must run BEFORE the new run's data overwrites the live section,
  // so we call it on every render pass — captures are idempotent per
  // triggerMessageId.
  if (!selectedTaskId) return;
  const stored = taskFrozenSectionSnapshots.get(selectedTaskId) || [];
  const frozenIds = new Set(stored.map((entry) => entry.triggerMessageId));
  // Optimistic rerun ids get replaced by server ids on the next poll. Track
  // fingerprints so we do not double-freeze the same rerun once the real id
  // arrives.
  const frozenFingerprints = new Set(
    stored.map((entry) => entry.triggerFingerprint).filter(Boolean),
  );
  let updated = false;
  for (const message of agentMessages) {
    const fingerprint = agentRerunMessageFingerprint(message);
    if (!fingerprint) continue;
    const stage = message?.metadata?.target_stage;
    const config = agentFrozenStageConfig(stage);
    if (!config) continue;
    const messageId = message?.id ? String(message.id) : "";
    if (!messageId) continue;
    if (frozenIds.has(messageId)) continue;
    if (frozenFingerprints.has(fingerprint)) continue;
    const contentNode = $(config.contentId);
    if (!contentNode) continue;
    if (contentNode.classList.contains("empty")) continue;
    const html = String(contentNode.innerHTML || "").trim();
    if (!html) continue;
    stored.push({
      triggerMessageId: messageId,
      triggerFingerprint: fingerprint,
      stage,
      sectionId: config.sectionId,
      headingHtml: config.headingHtml,
      label: config.label,
      contentClassName: contentNode.className || "",
      contentHtml: contentNode.innerHTML,
    });
    frozenIds.add(messageId);
    frozenFingerprints.add(fingerprint);
    updated = true;
  }
  if (updated) taskFrozenSectionSnapshots.set(selectedTaskId, stored);
}

function stripIdsFromHtml(html) {
  // Sanitize a frozen HTML fragment before re-inserting it:
  //  - remove id attributes so we never produce duplicate ids (e.g. two
  //    #metricPreview) that make getElementById/querySelector resolve to a stale
  //    frozen element;
  //  - as defense-in-depth, drop <script> elements and inline on* event handlers
  //    so a snapshot can never reintroduce active content (the live data source is
  //    already escaped, but frozen snapshots must stay inert).
  const template = document.createElement("template");
  template.innerHTML = String(html || "");
  template.content.querySelectorAll("script").forEach((el) => el.remove());
  template.content.querySelectorAll("*").forEach((el) => {
    el.removeAttribute("id");
    for (const attr of [...el.attributes]) {
      if (/^on/i.test(attr.name)) el.removeAttribute(attr.name);
    }
  });
  return template.innerHTML;
}

function createAgentFrozenSnapshotElement(snapshot) {
  const wrap = document.createElement("section");
  wrap.className = "progress-panel agent-frozen-snapshot";
  wrap.dataset.agentFrozenSnapshot = "true";
  wrap.dataset.frozenStage = snapshot.stage || "";
  wrap.dataset.frozenTrigger = snapshot.triggerMessageId || "";
  // Strip any id attributes from the snapshot HTML so we never end up with
  // duplicate ids (e.g. multiple #metricPreview) in the document.
  const innerWrapClass = String(snapshot.contentClassName || "").trim();
  wrap.innerHTML = [
    `<div class="agent-frozen-snapshot-label">${escapeHtml(snapshot.label || "历史")}</div>`,
    snapshot.headingHtml || "",
    `<div class="${escapeHtml(innerWrapClass)}" data-frozen-snapshot-content="true">${stripIdsFromHtml(snapshot.contentHtml)}</div>`,
  ].join("");
  return wrap;
}

function agentPersistentTimelineElementIds() {
  return [
    "scanSection",
    "notebookSection",
    "metricSection",
    "reportSection",
    "agentConversationPanel",
  ];
}

function restoreResultScrollDefaultOrder() {
  restoreResultScrollDefaultOrderDom({
    getElementById: $,
    persistentElementIds: agentPersistentTimelineElementIds(),
  });
}

function appendOptimisticAgentUserMessage(content, modelId = "") {
  const metadata = { optimistic: true };
  if (modelId) metadata.model_id = modelId;
  if (agentMessageIsAdvanceIntent({ role: "user", stage: "chat", content, metadata })) {
    metadata.intent = "advance";
  }
  const message = {
    id: `optimistic-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    role: "user",
    stage: "chat",
    content,
    metadata,
  };
  agentMessages = [...agentMessages, message];
  lastAgentRenderSignature = null;
  renderAgentConversation();
  return message;
}

function appendOptimisticAgentThinkingMessage(modelId = "") {
  const metadata = { optimistic: true, streaming: true };
  if (modelId) metadata.model_id = modelId;
  const message = {
    id: `optimistic-thinking-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    role: "assistant",
    stage: "chat",
    content: "",
    metadata,
  };
  agentMessages = [...agentMessages, message];
  lastAgentRenderSignature = null;
  renderAgentConversation();
  return message;
}

function removeOptimisticAgentMessage(messageId) {
  if (!messageId) return;
  agentMessages = agentMessages.filter((message) => message.id !== messageId);
  lastAgentRenderSignature = null;
  renderAgentConversation();
}

function clearAgentStageMessages() {
  const stageMessageIds = [
    "agentScanLeadMessages",
    "agentScanBeforeMessages",
    "agentScanMessages",
    "agentReproducibilityLeadMessages",
    "agentReproducibilityMessages",
    "agentMetricLeadMessages",
    "agentMetricMessages",
    "agentReportLeadMessages",
    "agentReportMessages",
  ];
  for (const targetId of stageMessageIds) {
    const target = $(targetId);
    if (!target) continue;
    target.innerHTML = "";
    target.classList.add("hidden");
  }
  updateAgentReportSectionVisibility();
}

function removeAgentTimelineBuckets() {
  removeAgentTimelineBucketsDom(document);
}

function agentTimelineVisibleStages() {
  return agentTimelineStageDefinitions()
    .filter(({ sectionId }) => {
      const section = $(sectionId);
      return section && !section.classList.contains("hidden") && section.getAttribute("aria-hidden") !== "true";
    })
    .map(({ stage }) => stage);
}

function renderAgentTimeline(messages = []) {
  renderAgentTimelineDom(messages, {
    getElementById: $,
    visibleStages: agentTimelineVisibleStages(),
    selectedTaskId,
    taskFrozenSectionSnapshots,
    agentMessages,
    createFrozenSnapshotElement: createAgentFrozenSnapshotElement,
    persistentElementIds: agentPersistentTimelineElementIds(),
    agentStageLabel,
    agentMessageHtml,
  });
}

function stripChatInstructions(content) {
  return stripChatInstructionsController(content);
}

function driverManualAnalysisHtml(messages) {
  return driverManualAnalysisHtmlController(messages, {
    renderAgentMarkdown,
    renderC1Form: agentMessageC1FormHtml,
    renderDedupPicker: agentMessageDedupPickerHtml,
    renderModelingSetup: agentMessageModelingSetupHtml,
    renderScreenTable: agentMessageScreenTableHtml,
    renderTables: agentMessageTablesHtml,
    renderModelDelivery: agentMessageModelDeliveryHtml,
  });
}

function latestInteractiveScreenMessageId(messages = []) {
  return latestInteractiveScreenMessageIdController(messages);
}

function renderDriverManualAnalysis(messages) {
  const panel = $("agentConversationPanel");
  const container = $("agentMessages");
  if (!panel || !container) return;
  removeAgentTimelineBuckets();
  resetAgentTypingState();
  panel.classList.remove("hidden");
  panel.classList.add("driver-analysis-mode");
  panel.setAttribute("aria-hidden", "false");
  panel.setAttribute("aria-label", "分析结果");
  container.innerHTML = driverManualAnalysisHtml(messages);
  // Keep the (hidden-for-driver) validation sections ordered after the analysis
  // panel so a later switch to a validation task restores cleanly.
  const scrollContent = $("resultScrollContent");
  if (scrollContent) {
    scrollContent.appendChild(panel);
    for (const elementId of agentPersistentTimelineElementIds()) {
      if (elementId === "agentConversationPanel") continue;
      const element = $(elementId);
      if (element) scrollContent.appendChild(element);
    }
  }
}

function resetAgentTypingState() {
  agentTypingState.clear();
  agentTypingCompleted.clear();
  if (agentTypingTimer !== null) {
    window.clearTimeout(agentTypingTimer);
    agentTypingTimer = null;
  }
}

function agentMessageIsStreaming(message) {
  const metadata = message?.metadata || {};
  return message?.role !== "user" && metadata.streaming === true;
}

function agentMessageIsThinking(message) {
  return agentMessageIsStreaming(message) && !String(message?.content || "").trim();
}

function agentVisibleContent(message) {
  const content = String(message?.content || "");
  const messageId = message?.id || "";
  if (!messageId) return content;
  let typing = agentTypingState.get(messageId);
  const streaming = agentMessageIsStreaming(message);
  if (!typing) {
    if (!streaming) return content;
    // A previously-completed id flipping back to streaming = server resumed
    // delta delivery. Seed visible with the bytes the user already saw so
    // the new tail appends, instead of visually clearing the message and
    // re-typing from byte 0. The startsWith guard below resets to empty if
    // the server actually rewrote the message instead of appending.
    const seedVisible = agentTypingCompleted.get(messageId) || "";
    typing = { visible: seedVisible, target: content };
    agentTypingState.set(messageId, typing);
  }
  if (!content.startsWith(typing.visible)) {
    typing.visible = "";
  }
  typing.target = content;
  if (typing.visible.length < typing.target.length) {
    scheduleAgentTyping();
    return typing.visible;
  }
  // Caught up. Only drop the state once the server has also signaled that
  // no further chunks are coming; remember the completion so a later resume
  // takes the seeded-visible path above instead of replaying from empty.
  if (!streaming) {
    agentTypingState.delete(messageId);
    agentTypingCompleted.set(messageId, content);
  }
  return content;
}

function scheduleAgentTyping() {
  if (agentTypingTimer !== null) return;
  agentTypingTimer = window.setTimeout(tickAgentTyping, AGENT_TYPEWRITER_INTERVAL_MS);
}

function tickAgentTyping() {
  agentTypingTimer = null;
  let changed = false;
  let pending = false;
  for (const typing of agentTypingState.values()) {
    if (typing.visible.length < typing.target.length) {
      const backlog = typing.target.length - typing.visible.length;
      const chunkSize = Math.max(
        AGENT_TYPEWRITER_CHARS_PER_TICK,
        Math.ceil(backlog / AGENT_TYPEWRITER_CATCHUP_TICKS),
      );
      const nextLength = typing.visible.length + chunkSize;
      typing.visible += typing.target.slice(typing.visible.length, nextLength);
      changed = true;
    }
    if (typing.visible.length < typing.target.length) {
      pending = true;
    }
  }
  if (changed) {
    lastAgentRenderSignature = null;
    renderAgentConversation();
  }
  if (pending) scheduleAgentTyping();
}

function agentMemoryReferencesHtml(references = []) {
  if (!Array.isArray(references) || references.length === 0) return "";
  const rows = references.map((reference) => {
    const memoryId = String(reference.id || reference.memory_id || "");
    const kind = reference.kind || "raw";
    const type = reference.memory_type || "memory";
    const sourceTask = reference.source_task_id || "";
    const confidence = reference.confidence !== undefined ? formatMemoryConfidence(reference.confidence) : "";
    const reason = reference.use_reason || reference.reason || "";
    const sourceCount = Array.isArray(reference.source_memory_ids) ? reference.source_memory_ids.length : 0;
    const meta = [
      kind === "distillation" ? "进化沉淀" : "",
      type,
      sourceTask ? `来源 ${sourceTask}` : "",
      sourceCount ? `来源记忆 ${sourceCount}` : "",
      reference.support_count !== undefined ? `支持 ${reference.support_count}` : "",
      confidence ? `置信度 ${confidence}` : "",
    ].filter(Boolean).map(escapeHtml).join(" · ");
    return [
      '<li class="agent-memory-reference">',
      '<span class="agent-memory-reference-main">',
      `<strong>${escapeHtml(memoryId || type)}</strong>`,
      meta ? `<small>${meta}</small>` : "",
      reason ? `<span>${escapeHtml(reason)}</span>` : "",
      "</span>",
      memoryId
        ? `<button class="agent-memory-reference-action" type="button" data-agent-memory-inline-inspect="${escapeHtml(memoryId)}" data-agent-memory-inline-kind="${escapeHtml(kind)}">查看</button>`
        : "",
      "</li>",
    ].join("");
  }).join("");
  return [
    '<details class="agent-memory-references">',
    `<summary>引用记忆 ${references.length}</summary>`,
    `<ul>${rows}</ul>`,
    "</details>",
  ].join("");
}

// Inline rich tables carried by the generic plan driver (data_join / future
// feature / modeling). Format is the driver's simple {title, columns, rows};
// validation metric tables use a different path (metadata.sections) and are
// untouched. Each driver message is appended whole, so this renders once on the
// full timeline rebuild — no streaming fast-path interaction.
function agentMessageTablesHtml(message) {
  const tables = message?.metadata?.tables;
  if (!Array.isArray(tables) || !tables.length) return "";
  const blocks = tables
    .map((table) => {
      const columns = Array.isArray(table?.columns) ? table.columns : [];
      const rows = Array.isArray(table?.rows) ? table.rows : [];
      if (!columns.length && !rows.length) return "";
      const head = columns.length
        ? `<thead><tr>${columns.map((col) => `<th>${escapeHtml(String(col))}</th>`).join("")}</tr></thead>`
        : "";
      const body = `<tbody>${rows
        .map((row) => {
          const cells = Array.isArray(row) ? row : [row];
          return `<tr>${cells.map((cell) => `<td>${escapeHtml(String(cell ?? ""))}</td>`).join("")}</tr>`;
        })
        .join("")}</tbody>`;
      const caption = table?.title
        ? `<div class="agent-inline-table-title">${escapeHtml(String(table.title))}</div>`
        : "";
      return `<div class="agent-inline-table">${caption}<div class="agent-inline-table-scroll"><table>${head}${body}</table></div></div>`;
    })
    .join("");
  return blocks ? `<div class="agent-message-tables">${blocks}</div>` : "";
}

function agentMessageModelingSetupHtml(message, options = {}) {
  return renderModelingSetupPanel(message, options);
}

function agentMessageModelDeliveryHtml(message, options = {}) {
  return renderModelDeliveryPanel(message, options);
}

async function submitModelingWeightAdjust(button) {
  return submitModelingWeightAdjustController(button, modelingSetupControllerContext());
}

function handleModelingWeightAdjustClick(event) {
  return handleModelingWeightAdjustClickController(event, modelingSetupControllerContext());
}

function modelingSetupControllerContext() {
  return {
    getSelectedTaskId: () => selectedTaskId,
    api,
    agentAcceptanceModeValue,
    setActionStatus,
    setAgentMessages: (messages) => {
      agentMessages = messages || agentMessages;
    },
    renderAgentConversation,
  };
}
if (typeof document !== "undefined") {
  document.addEventListener("click", handleModelingWeightAdjustClick);
}

function agentMessageC1FormHtml(message) {
  return renderJoinC1Form(message);
}

async function submitC1Assignment(button) {
  return submitC1AssignmentController(button, joinGateControllerContext());
}

function handleC1ConfirmClick(event) {
  return handleC1ConfirmClickController(event, joinGateControllerContext());
}

function joinGateControllerContext() {
  return {
    getSelectedTaskId: () => selectedTaskId,
    api,
    agentAcceptanceModeValue,
    setActionStatus,
    setAgentMessages: (messages) => {
      agentMessages = messages || agentMessages;
    },
    renderAgentConversation,
  };
}
if (typeof document !== "undefined") {
  document.addEventListener("click", handleC1ConfirmClick);
}

function agentMessageScreenTableHtml(message, options = {}) {
  return renderScreenGateTable(message, options);
}

async function submitScreenThresholdAdjust(button) {
  return submitScreenThresholdAdjustController(button, screenGateControllerContext());
}

async function submitScreenSelection(button) {
  return submitScreenSelectionController(button, screenGateControllerContext());
}

function handleScreenAdjustClick(event) {
  return handleScreenAdjustClickController(event, screenGateControllerContext());
}

function handleScreenConfirmClick(event) {
  return handleScreenConfirmClickController(event, screenGateControllerContext());
}

function screenGateControllerContext() {
  return {
    getSelectedTaskId: () => selectedTaskId,
    api,
    agentAcceptanceModeValue,
    setActionStatus,
    setAgentMessages: (messages) => {
      agentMessages = messages || agentMessages;
    },
    renderAgentConversation,
  };
}
if (typeof document !== "undefined") {
  document.addEventListener("click", handleScreenAdjustClick);
  document.addEventListener("click", handleScreenConfirmClick);
}

function agentMessageDedupPickerHtml(message) {
  return renderDedupPicker(message);
}

async function submitDedupStrategies(button) {
  return submitDedupStrategiesController(button, joinGateControllerContext());
}

function handleDedupConfirmClick(event) {
  return handleDedupConfirmClickController(event, joinGateControllerContext());
}
if (typeof document !== "undefined") {
  document.addEventListener("click", handleDedupConfirmClick);
}

function agentMessageGateButtonHtml(message) {
  return renderDriverGateButton(message, { isAgentMode: selectedTaskIsAgentMode });
}

async function submitDriverConfirm(button) {
  return submitDriverConfirmController(button, driverConfirmControllerContext());
}

function handleDriverConfirmClick(event) {
  return handleDriverConfirmClickController(event, driverConfirmControllerContext());
}

function driverConfirmControllerContext() {
  return {
    getSelectedTaskId: () => selectedTaskId,
    api,
    setActionStatus,
    setAgentMessages: (messages) => {
      agentMessages = messages || agentMessages;
    },
    renderAgentConversation,
  };
}
if (typeof document !== "undefined") {
  document.addEventListener("click", handleDriverConfirmClick);
}

function handleDriverReportDownloadClick(event) {
  const button = event.target?.closest?.("[data-driver-report-download]");
  if (!button || !selectedTaskId) return;
  event.preventDefault();
  window.location.href = `/api/tasks/${encodeURIComponent(selectedTaskId)}/driver-report/download`;
}
if (typeof document !== "undefined") {
  document.addEventListener("click", handleDriverReportDownloadClick);
  planRailController.installArtifactHandlers(document);
}

function agentMessageHtml(message, labelStage = message?.stage, options = {}) {
  const role = message.role === "user" ? "user" : "assistant";
  const className = role === "user" ? "agent-message user" : "agent-message assistant";
  const streaming = agentMessageIsStreaming(message);
  const thinking = agentMessageIsThinking(message);
  const contentHtml = thinking
    ? agentThinkingHtml()
    : formatAgentMessageContent(agentVisibleContent(message), { markdown: role === "assistant" });
  const memoryReferencesHtml = role === "assistant"
    ? agentMemoryReferencesHtml(message?.metadata?.memory_references)
    : "";
  const messageId = message?.id ? String(message.id) : "";
  const idAttr = messageId ? ` data-agent-message-id="${escapeHtml(messageId)}"` : "";
  return [
    `<article class="${className}"${idAttr}>`,
    role === "assistant" && !options.hideMeta ? `<div class="agent-message-meta">${escapeHtml(agentMessageMetaLabel(message, labelStage))}</div>` : "",
    `<div class="agent-message-content" data-agent-streaming="${streaming ? "true" : "false"}" data-agent-thinking="${thinking ? "true" : "false"}">${contentHtml}</div>`,
    role === "assistant"
      ? (message?.metadata?.join_c1 ? agentMessageC1FormHtml(message) : `${agentMessageModelDeliveryHtml(message)}${agentMessageTablesHtml(message)}`)
      : "",
    role === "assistant" ? agentMessageGateButtonHtml(message) : "",
    memoryReferencesHtml,
    "</article>",
  ].join("");
}

function agentThinkingHtml() {
  return [
    '<span class="agent-thinking" role="status" aria-live="polite">',
    '<span class="agent-thinking-text">正在思考</span>',
    '<span class="agent-thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span>',
    "</span>",
  ].join("");
}

function agentValidatorAlias(validator) {
  return agentValidatorAliases[String(validator || "").trim()] || "";
}

function agentStageLabel(_stage) {
  return agentValidatorAlias(selectedTask?.validator) || "Agent";
}

function agentMessageMetaLabel(message, labelStage = message?.stage) {
  const pieces = [agentStageLabel(labelStage)];
  const metadata = message?.metadata || {};
  const step = agentMessagePlanStep(metadata);
  const phase = metadata.phase || step?.phase || "";
  const stepTitle = metadata.step_title || step?.title || "";
  const runSeq = Number(metadata.run_seq);
  if (phase) pieces.push(String(phase));
  if (stepTitle) pieces.push(String(stepTitle));
  if (Number.isFinite(runSeq) && runSeq > 0) pieces.push(`第 ${runSeq} 轮`);
  return pieces.filter(Boolean).join(" · ");
}

function agentMessagePlanStep(metadata = {}) {
  return planRailController.planStep(metadata, selectedTaskId);
}

function formatAgentMessageContent(content, { markdown = false } = {}) {
  if (markdown) return renderAgentMarkdown(content);
  return escapeHtml(content).replaceAll("\n", "<br>");
}

function shouldPreserveOptimisticAgentMessages(nextMessages = []) {
  const optimisticCount = agentMessages.filter((message) => message?.metadata?.optimistic).length;
  return optimisticCount > 0 && nextMessages.length < agentMessages.length;
}

function agentMessageCanPollIncrementally({ preserveOptimistic = false } = {}) {
  if (preserveOptimistic || !agentMessages.length) return false;
  return !agentMessages.some((message) => message?.metadata?.optimistic || message?.metadata?.streaming);
}

function mergeIncrementalAgentMessages(nextMessages = []) {
  if (!nextMessages.length) return false;
  const seen = new Set(agentMessages.map((message) => message.id).filter(Boolean));
  const additions = nextMessages.filter((message) => !seen.has(message.id));
  if (!additions.length) return false;
  agentMessages = [...agentMessages, ...additions];
  return true;
}

async function loadAgentMessages(taskId = selectedTaskId, { preserveOptimistic = false } = {}) {
  const messageTask = findTaskInCache(taskId) || selectedTask;
  // Driver tasks have a conversation in manual mode too (controls, no LLM).
  const hasConversation = selectedTaskIsAgentMode(messageTask) || taskUsesPlanRail(messageTask);
  if (!taskId || !hasConversation) {
    agentMessages = [];
    renderAgentConversation();
    return;
  }
  const useIncremental = agentMessageCanPollIncrementally({ preserveOptimistic });
  const lastMessageId = useIncremental ? agentMessages[agentMessages.length - 1]?.id : "";
  const suffix = lastMessageId ? `?after_id=${encodeURIComponent(lastMessageId)}` : "";
  const payload = await api(`api/tasks/${taskId}/agent/messages${suffix}`);
  if (selectedTaskId !== taskId) return;
  const nextMessages = payload.messages || [];
  if (payload.incremental) {
    if (mergeIncrementalAgentMessages(nextMessages)) renderAgentConversation();
    return;
  }
  if (preserveOptimistic && shouldPreserveOptimisticAgentMessages(nextMessages)) return;
  agentMessages = nextMessages;
  renderAgentConversation();
}

async function pollAgentMessagesUntilSettled(taskId, pendingPromise, { preserveOptimistic = false } = {}) {
  let settled = false;
  pendingPromise.then(
    () => { settled = true; },
    () => { settled = true; },
  );
  while (!settled && selectedTaskId === taskId) {
    await sleep(AGENT_STREAM_POLL_INTERVAL_MS);
    if (settled || selectedTaskId !== taskId) break;
    try {
      await loadAgentMessages(taskId, { preserveOptimistic });
    } catch (_error) {
      // The primary request path owns user-visible errors.
    }
  }
}

async function startAgentValidation() {
  const taskId = selectedTaskId;
  if (!taskId) return;
  const input = $("agentComposerInput");
  const originalValue = input.value;
  const content = input.value.trim();
  if (!content) {
    setActionStatus("请输入要交给 Agent 的任务。", "error");
    return;
  }
  // Agent mode is, by definition, "manual mode with the operator's decisions made
  // by an LLM" — so it always requires a configured LLM. Without one, error out and
  // prompt the user to configure a model (no canned/default agent conversation).
  // The deterministic, no-LLM flow is the *manual* mode, reached a different way.
  const unavailableModelMessage = agentModelUnavailableMessage();
  if (showAgentModelGuidance(unavailableModelMessage)) return;
  setAgentComposerNotice("");
  const modelId = $("agentModelSelect").value || "";
  input.value = "";
  autoGrowComposerInput();
  updateAgentSendDisabled();
  const optimisticMessage = appendOptimisticAgentUserMessage(content, modelId);
  const optimisticThinkingMessage = appendOptimisticAgentThinkingMessage(modelId);
  let result;
  try {
    const requestPromise = api(`api/tasks/${taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify({
        content,
        model_id: modelId || null,
        effort: agentEffort(),
        acceptance_mode: agentAcceptanceModeValue(),
      }),
    });
    const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
    result = await requestPromise;
    await streamPollPromise;
  } catch (error) {
    removeOptimisticAgentMessage(optimisticMessage.id);
    removeOptimisticAgentMessage(optimisticThinkingMessage.id);
    input.value = originalValue;
    autoGrowComposerInput();
    updateAgentSendDisabled();
    if (showAgentModelGuidance(agentModelConfigurationErrorMessage(error))) return;
    throw error;
  }
  agentMessages = result.messages || agentMessages;
  renderAgentConversation();
  if (result.status === "cancel_requested") {
    await waitForAgentValidation(taskId, { stopping: true });
    return;
  }
  if (result.status !== "accepted") return;
  await waitForAgentValidation(taskId);
}

async function dispatchAgentValidation(taskId = selectedTaskId) {
  const normalizedTaskId = requireTaskId(taskId || selectedTaskId, "Agent 初始化");
  const modelId = $("agentModelSelect").value || "";
  const result = await api(`/api/tasks/${normalizedTaskId}/agent/start`, {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId || null,
      effort: agentEffort(),
      acceptance_mode: agentAcceptanceModeValue(),
    }),
  });
  agentMessages = result.messages || agentMessages;
  renderAgentConversation();
  if (result.status !== "accepted") return;
  await waitForAgentValidation(normalizedTaskId);
}

async function stopAgentValidation(taskId = selectedTaskId) {
  const normalizedTaskId = requireTaskId(taskId || selectedTaskId, "Agent 停止");
  const result = await api(`api/tasks/${normalizedTaskId}/agent/stop`, {
    method: "POST",
  });
  agentMessages = result.messages || agentMessages;
  renderAgentConversation();
  updateAgentSendDisabled();
  if (result.status === "cancel_requested") {
    await waitForAgentValidation(normalizedTaskId, { stopping: true });
    return;
  }
  setActionStatus(result.message || "已停止当前动作，请问有什么指示？", "success");
}

async function waitForAgentValidation(taskId, { stopping = false } = {}) {
  const busyText = stopping ? "Agent 正在停止..." : "Agent 正在执行验证...";
  setBusy("agent", busyText, taskId);
  setActionStatus(busyText, "busy");
  const progressPromise = pollValidationProgress(
    new Set(["scanned", "executed", "writing_artifacts", "failed", "succeeded", "review_required"]),
    taskId,
    { stopping },
  );
  const streamPollPromise = pollAgentMessagesUntilSettled(taskId, progressPromise);
  const finalTask = await progressPromise;
  await streamPollPromise;
  if (selectedTaskId !== taskId) return;
  await loadAgentMessages(taskId);
  await loadReportFields(taskId);
  if (stopping || agentValidationStopped(finalTask || selectedTask)) {
    setActionStatus("Agent 已停止，可根据当前阶段结果重新发起或继续下一步。", "success");
    return;
  }
  if (agentValidationPaused(finalTask || selectedTask)) {
    setActionStatus("当前阶段已完成，等待你的下一步指令。", "success");
    return;
  }
  if (finalTask?.status === "failed" || selectedTask?.status === "failed") {
    setTaskFailureActionStatus(finalTask || selectedTask);
  } else {
    setActionStatus("Agent 已完成当前处理。", "success");
  }
}

function agentValidationStopped(task) {
  return task?.stopped === true;
}

function agentValidationPaused(task) {
  const status = task?.status || "";
  return ["scanned", "executed", "writing_artifacts", "review_required"].includes(status);
}

function prefillAgentTaskInstruction(task) {
  if (task?.run_mode !== "agent") return;
  const input = $("agentComposerInput");
  if (!input || input.value.trim()) return;
  const definition = taskTypeDefinition(task.task_type || createTaskDialog.activeTaskType());
  input.value = definition.initialGoal;
  autoGrowComposerInput();
  updateAgentSendDisabled();
}

async function createTask() {
  const task = await createTaskDialog.createTask();
  if (!task) return null;
  selectedTaskId = task.id;
  selectedTask = task;
  rememberSelectedTaskId(task.id);
  renderStoredStateSummaries();
  await refreshTasks();
  await loadReportFields();
  setCreateStatus("任务已创建。");
  closeTaskDialog();
  prefillAgentTaskInstruction(task);
  return task;
}

async function refreshTasks() {
  taskCache = await api("api/tasks");
  syncSelectedTaskFromCache();
  ensureActiveTaskProgressPolling();
}

async function scanCurrentTask() {
  const taskId = selectedTaskId;
  if (!taskId) return;
  const controller = new AbortController();
  scanAbortController = controller;
  try {
    const result = await api(`api/tasks/${taskId}/scan`, {
      method: "POST",
      signal: controller.signal,
    });
    if (selectedTaskId === taskId) renderScanResult(result);
    await refreshTasks();
    if (selectedTaskId === taskId) {
      if (selectedTask?.status === "failed") {
        setTaskFailureActionStatus(selectedTask);
        return;
      }
      setActionStatus(
        selectedTaskIsAgentMode(selectedTask) ? "材料完备性识别完成。" : "材料扫描完成。",
        "success",
      );
      scrollToManualWorkflowSection("scan");
    }
  } finally {
    if (scanAbortController === controller) scanAbortController = null;
  }
}

async function createTaskAndScan() {
  const task = await createTask();
  if (!task) return;
  if (task.run_mode === "agent") {
    const taskId = task.id || selectedTaskId;
    const activeDialogTaskType = createTaskDialog.activeTaskType();
    const definition = taskTypeDefinition(task.task_type || activeDialogTaskType);
    const isValidationTask = (task.task_type || activeDialogTaskType || defaultTaskType) === "validation";
    setBusy(null, "", taskId);
    await loadAgentMessages(taskId);
    renderAll();
    if (!isValidationTask && definition.initialGoal) {
      // createTask() already seeded the conversation composer via
      // prefillAgentTaskInstruction; just focus it (the V2 plan dialog is retired).
      $("agentComposerInput")?.focus?.();
      setActionStatus(`${definition.label}任务已创建，已填入建议目标，确认后发送即可。`, "success");
      return;
    }
    setActionStatus("Agent 任务已创建，等待你的下一条指令。", "success");
    return;
  }
  // Manual mode for a driver task (data_join / feature / modeling): start the
  // deterministic, control-driven flow (no LLM). Validation manual still scans.
  if (taskUsesPlanRail(task)) {
    const taskId = task.id || selectedTaskId;
    setBusy(null, "", taskId);
    await dispatchDriverStart(taskId);
    renderAll();
    setActionStatus(`${taskTypeDefinition(task.task_type).label}任务已创建，请在下方逐步确认。`, "success");
    return;
  }
  setBusy(null, "", null);
  setBusy("scan", "任务已创建，正在自动扫描材料...", task.id);
  setActionStatus("任务已创建，正在自动扫描材料...", "busy");
  try {
    await scanCurrentTask();
    await loadTaskEvidence(task.id);
  } finally {
    setBusy(null, "", task.id);
  }
}

// Start a driver-based task's deterministic flow (manual mode, no LLM): POST the
// agent-start endpoint, which routes to the plan-conversation driver.
async function dispatchDriverStart(taskId = selectedTaskId) {
  const normalizedTaskId = requireTaskId(taskId || selectedTaskId, "启动");
  const result = await api(`/api/tasks/${normalizedTaskId}/agent/start`, {
    method: "POST",
    body: JSON.stringify({}),
  });
  agentMessages = result.messages || agentMessages;
  renderAgentConversation();
}

async function pollValidationProgress(
  doneStatuses = terminalTaskStatuses,
  taskId = selectedTaskId,
  { stopping = false, background = false } = {},
) {
  if (!taskId) return null;
  const claim = claimProgressPoll(progressPolls, taskId, { background });
  if (!claim.claimed) return claim.existing.promise;
  const pollState = claim.pollState;
  const promise = (async () => {
    const startedAt = Date.now();
    const timeoutMs = 1000 * 60 * 60;
    while (true) {
      if (pollState.cancelled) return null;
      await sleep(1000);
      if (pollState.cancelled) return null;
      await refreshTasks();
      const polledTask = findTaskInCache(taskId);
      if (!polledTask) return null;
      if (selectedTaskId === taskId) {
        await loadTaskEvidence(taskId);
        if (metricOverviewComplete(polledTask) && !currentMetricPreviewHasValues(taskId)) {
          await loadReportFields(taskId);
        }
        if (selectedTaskIsAgentMode(polledTask)) await loadAgentMessages(taskId);
        renderChangedValidationViews();
      } else {
        renderTaskList();
      }

      const status = polledTask.status || "";
      const serverBusyAction = taskServerBusyAction(polledTask);
      if (stopping && !serverBusyAction) {
        if (selectedTaskId === taskId && !background) {
          setActionStatus("Agent 已停止，可根据当前阶段结果重新发起或继续下一步。", "success");
        }
        return polledTask;
      }
      if (doneStatuses.has(status) && !serverBusyAction) {
        if (selectedTaskId === taskId && !background) {
          if (status === "failed" || status === "review_required") {
            setTaskFailureActionStatus(polledTask);
          } else {
            setActionStatus("验证完成。", "success");
          }
        }
        return polledTask;
      }

      // Status copy for in-flight polling is owned by taskActionStatusSnapshot()
      // via renderCurrentTask(); writing here too would alternate the pill text
      // between two sources every second.

      if (Date.now() - startedAt > timeoutMs) {
        if (selectedTaskId === taskId && !background) {
          setActionStatus("验证仍在后台运行，请稍后刷新查看结果。", "error");
        }
        return polledTask;
      }
    }
  })().finally(() => releaseProgressPoll(progressPolls, taskId, pollState));
  pollState.promise = promise;
  return promise;
}

async function validateCurrentTask(options = {}) {
  const taskId = selectedTaskId;
  if (!taskId) return;
  // A fresh notebook run produces a fresh reproducibility result; the entry
  // animation is allowed to play once for the new run.
  resetReproducibilityRenderSignatures();
  const result = await api(`api/tasks/${taskId}/notebook`, {
    method: "POST",
    body: JSON.stringify({}),
  });
  if (selectedTaskId === taskId) {
    renderValidationResult(result);
    appendPendingReproducibilitySteps();
  }
  const finalTask = await pollValidationProgress(new Set(["executed", "failed", "scanned"]), taskId);
  if (selectedTaskId !== taskId) return;
  if (finalTask?.status === "scanned" || selectedTask?.status === "scanned") {
    setActionStatus("Notebook 已停止，可重新运行。", "success");
    return;
  }
  await loadReportFields(taskId);
  await loadTaskEvidence(taskId);
  if (selectedTask?.status === "failed" || selectedTask?.status === "review_required") {
    setTaskFailureActionStatus(selectedTask || finalTask);
  } else {
    setActionStatus("验证完成。", "success");
    scrollToManualWorkflowSection("notebook");
  }
}

async function cancelCurrentNotebook() {
  const taskId = selectedTaskId;
  if (!taskId) return;
  await api(`api/tasks/${taskId}/notebook/cancel`, { method: "POST" });
  if (selectedTaskId === taskId) setActionStatus("正在停止 Notebook...", "busy");
  const finalTask = await pollValidationProgress(new Set(["scanned", "failed"]), taskId);
  if (selectedTaskId !== taskId) return;
  await loadTaskEvidence(taskId);
  if (finalTask?.status === "failed" || selectedTask?.status === "failed") {
    setTaskFailureActionStatus(finalTask || selectedTask);
  } else {
    setActionStatus("Notebook 已停止，可重新运行。", "success");
  }
}

async function cancelCurrentMetrics() {
  const taskId = selectedTaskId;
  if (!taskId) return;
  await api(`api/tasks/${taskId}/metrics/cancel`, { method: "POST" });
  if (selectedTaskId === taskId) setActionStatus("正在停止指标生成...", "busy");
  const finalTask = await pollValidationProgress(new Set(["executed", "writing_artifacts", "failed"]), taskId);
  if (selectedTaskId !== taskId) return;
  await loadTaskEvidence(taskId);
  if (finalTask?.status === "failed" || selectedTask?.status === "failed") {
    setTaskFailureActionStatus(finalTask || selectedTask);
  } else if (finalTask?.status === "writing_artifacts" || selectedTask?.status === "writing_artifacts") {
    setActionStatus("指标与 Excel 已生成。", "success");
  } else {
    setActionStatus("指标生成已停止，可重新生成。", "success");
  }
}

async function cancelCurrentReport() {
  const taskId = selectedTaskId;
  if (!taskId) return;
  await api(`api/tasks/${taskId}/report/cancel`, { method: "POST" });
  if (selectedTaskId === taskId) setActionStatus("正在停止报告生成...", "busy");
  const finalTask = await pollValidationProgress(new Set(["writing_artifacts", "succeeded", "review_required", "failed"]), taskId);
  if (selectedTaskId !== taskId) return;
  await loadTaskEvidence(taskId);
  if (finalTask?.status === "failed" || selectedTask?.status === "failed") {
    setTaskFailureActionStatus(finalTask || selectedTask);
  } else if ((finalTask?.report_available || selectedTask?.report_available) === true) {
    setActionStatus("Word 报告已生成，可下载。", "success");
  } else {
    setActionStatus("报告生成已停止，可重新生成。", "success");
  }
}

async function generateMetrics() {
  const taskId = selectedTaskId;
  if (!taskId) return;
  await api(`api/tasks/${taskId}/metrics`, { method: "POST" });
  if (selectedTaskId === taskId) {
    appendPendingMetricSteps();
    $("metricPreview").innerHTML =
      '<div class="result-summary empty">指标与 Excel 正在生成...</div>';
  }
  const finalTask = await pollValidationProgress(new Set(["executed", "writing_artifacts", "failed"]), taskId);
  if (selectedTaskId !== taskId) return;
  await loadReportFields(taskId);
  await loadTaskEvidence(taskId);
  if (selectedTask?.status === "failed") {
    setTaskFailureActionStatus(selectedTask);
  } else if (finalTask?.status === "executed" || selectedTask?.status === "executed") {
    setActionStatus("指标生成已停止，可重新生成。", "success");
  } else {
    setActionStatus("指标与 Excel 已生成。", "success");
    scrollToManualWorkflowSection("metrics");
  }
}

async function loadReportFields(taskId = selectedTaskId) {
  if (!taskId) {
    renderMetricPreview({});
    return;
  }
  const payload = await api(`api/tasks/${taskId}/report-fields`);
  if (selectedTaskId !== taskId) return;
  renderMetricPreview(
    payload.metric_values || {},
    payload.workbook_source,
    payload.metric_table_sections || [],
  );
}

async function generateReport() {
  const taskId = selectedTaskId;
  if (!taskId) return;
  await api(`api/tasks/${taskId}/report`, { method: "POST" });
  await pollValidationProgress(terminalTaskStatuses, taskId);
  if (selectedTaskId !== taskId) return;
  await loadReportFields(taskId);
  if (selectedTask?.status === "failed") {
    setTaskFailureActionStatus(selectedTask);
  } else {
    setActionStatus("Word 报告已生成，可下载。", "success");
    scrollToManualWorkflowSection("report");
  }
}

function downloadWordReport() {
  if (!selectedTaskId) return;
  window.location.href = `api/tasks/${selectedTaskId}/report/download`;
}

function downloadExcelAnalysis() {
  if (!selectedTaskId) return;
  window.location.href = `api/tasks/${selectedTaskId}/analysis/download`;
}

function previewWordReport() {
  openWordPreviewDialog();
}

async function deleteTask(task) {
  if (!task || taskBusyAction(task.id)) return;
  if (taskServerBusyAction(task)) {
    setActionStatus("运行中的任务不能删除。", "error");
    return;
  }
  const confirmed = await showPlatformConfirm({
    title: "删除任务",
    message: `确认删除任务「${taskDisplayName(task)}」？删除后将移除任务记录和本地输出文件，不能撤销。`,
    confirmText: "删除",
    cancelText: "取消",
    tone: "danger",
  });
  if (!confirmed) {
    setActionStatus("已取消删除。");
    return;
  }

  try {
    setBusy("delete", "正在删除任务...", task.id);
    setActionStatus("正在删除任务...", "busy");
    renderAll();
    await api(`api/tasks/${task.id}`, { method: "DELETE" });
    if (selectedTaskId === task.id) {
      selectedTaskId = null;
      selectedTask = null;
      rememberSelectedTaskId(null);
    }
    resultScrollPositionsByTask.delete(task.id);
    persistResultScrollPositions();
    await refreshTasks();
    renderStoredStateSummaries();
    await loadReportFields();
    setActionStatus("任务已删除。", "success");
  } catch (error) {
    setActionStatus(error.message || "删除任务失败。", "error");
  } finally {
    setBusy(null, "", task.id);
    renderAll();
  }
}

async function runAction(action, options = {}) {
  const actionId = options.actionId || null;
  const taskId = options.taskId || selectedTaskId;
  let shouldRenderAfter = options.renderAfter !== false;
  try {
    if (actionId) setBusy(actionId, options.busyText || "正在处理...", taskId);
    await action();
  } catch (error) {
    shouldRenderAfter = true;
    if (error?.name === "AbortError") {
      if (actionId) setActionStatus(actionCancelledStatusTitle(actionId), "success");
      return;
    }
    if (selectedTaskId) {
      try {
        await refreshTasks();
      } catch (_) {
        // Keep the original action error visible when status refresh also fails.
      }
    }
    const message = error.message || "操作失败";
    if (actionId === "agentMemory") setAgentMemoryStatus(message, "error");
    if (actionId === "draftTools") setDraftToolsStatus(message, "error");
    if (actionId) renderActionError(actionId, message);
    if (actionId) setActionStatus(actionFailureStatusTitle(actionId), "error", message);
    else setCreateStatus(message, "error");
  } finally {
    if (actionId) setBusy(null, "", taskId);
    if (shouldRenderAfter) renderAll();
  }
}

function handleTaskListKeydown(event) {
  if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
  const rows = Array.from(document.querySelectorAll(".task-row"));
  if (rows.length === 0) return;
  event.preventDefault();
  const currentIndex = rows.indexOf(document.activeElement);
  const nextIndex = event.key === "ArrowDown"
    ? Math.min(rows.length - 1, currentIndex + 1)
    : Math.max(0, currentIndex - 1);
  rows[nextIndex < 0 ? 0 : nextIndex].focus();
}

async function copyText(text) {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    setActionStatus("路径已复制。", "success");
  } catch (_) {
    setActionStatus("浏览器不允许自动复制，请手动选择路径。", "error");
  }
}

function closeSidebarSettingsOnOutsideClick(event) {
  const settings = $("sidebarSettings");
  if (!settings?.open) return;
  const target = event.target;
  if (target instanceof Element && target.closest("#sidebarSettings")) return;
  settings.open = false;
}

function openGovernanceSettingsFromSidebar() {
  closeSidebarSettingsMenu();
  scheduleGovernanceSettingsFromSidebar();
}

function handleGovernanceSettingsPointerDown(event) {
  event.preventDefault();
  event.stopPropagation();
  closeSidebarSettingsMenu();
  scheduleGovernanceSettingsFromSidebar();
}

function workflowActionConfig(actionId) {
  if (actionId === "scan") {
    return { action: scanCurrentTask, busyText: "正在扫描材料..." };
  }
  if (actionId === "notebook") {
    return { action: validateCurrentTask, busyText: "正在运行 v2 验证..." };
  }
  if (actionId === "cancelNotebook") {
    return { action: cancelCurrentNotebook, busyText: "正在停止 Notebook..." };
  }
  if (actionId === "metrics") {
    return { action: generateMetrics, busyText: "正在生成指标与 Excel..." };
  }
  if (actionId === "cancelMetrics") {
    return { action: cancelCurrentMetrics, busyText: "正在停止指标生成..." };
  }
  if (actionId === "report") {
    return { action: generateReport, busyText: "正在生成 Word 报告..." };
  }
  if (actionId === "cancelReport") {
    return { action: cancelCurrentReport, busyText: "正在停止报告生成..." };
  }
  if (actionId === "downloadWordReport") {
    return { action: downloadWordReport, busyText: "" };
  }
  if (actionId === "downloadExcelAnalysis") {
    return { action: downloadExcelAnalysis, busyText: "" };
  }
  if (actionId === "previewWordReport") {
    return { action: previewWordReport, busyText: "正在打开 Word 预览..." };
  }
  return null;
}

function scrollStepTarget(targetId) {
  if (!targetId) return;
  $(targetId)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function scrollToManualWorkflowSection(stepId) {
  if (!stepId) return;
  // Agent mode owns its own scroll behavior (follow-the-stream); reusing
  // the manual jump there would fight the typewriter auto-scroll.
  if (selectedTaskIsAgentMode()) return;
  const step = workflowSteps.find((candidate) => candidate.id === stepId);
  if (!step?.target) return;
  // Capture the task id at scheduling time so a deferred-frame scroll does
  // not jump the panel to the wrong section after the user switched tasks
  // mid-action.
  const targetTaskId = selectedTaskId;
  window.requestAnimationFrame(() => {
    if (selectedTaskId !== targetTaskId) return;
    scrollStepTarget(step.target);
  });
}

function handleWorkflowStepperClick(event) {
  if (planRailController.handleClick(event)) return;
  const actionButton = event.target.closest("[data-step-action]");
  if (actionButton) {
    event.preventDefault();
    event.stopPropagation();
    const actionId = actionButton.dataset.stepAction;
    const config = workflowActionConfig(actionId);
    if (config) runAction(config.action, { actionId, busyText: config.busyText });
    return;
  }
  const step = event.target.closest(".step[data-step-target]");
  if (step) scrollStepTarget(step.dataset.stepTarget);
}

function handleWorkflowStepperKeydown(event) {
  if (!["Enter", " "].includes(event.key)) return;
  const step = event.target.closest(".step[data-step-target]");
  if (!step || event.target.closest("[data-step-action]")) return;
  event.preventDefault();
  scrollStepTarget(step.dataset.stepTarget);
}

$("createTaskOpenButton").onclick = openTaskTypeWelcome;
$("collapsedCreateTaskButton").onclick = openTaskTypeWelcome;
$("welcomeTaskCards").onclick = openTaskDialogFromCard;
$("closeTaskDialogButton").onclick = closeTaskDialog;
$("openGovernanceSettingsButton").addEventListener("pointerdown", handleGovernanceSettingsPointerDown, true);
$("openGovernanceSettingsButton").onclick = openGovernanceSettingsFromSidebar;
$("closeGovernanceSettingsButton").onclick = closeGovernanceSettingsDialog;
$("governanceSettingsDialog").addEventListener("click", handleGovernanceSettingsNavClick);
$("governanceSettingsDialog").addEventListener("change", handleMemoryPolicyChange);
$("governanceSettingsSearch").oninput = handleGovernanceSettingsSearch;
$("governanceRefreshButton").onclick = refreshActiveGovernancePanel;
$("closeWordPreviewButton").onclick = closeWordPreviewDialog;
$("refreshExecutionEnvironmentOptionsButton").onclick = refreshExecutionEnvironmentOptions;
$("executionEnvironmentList").addEventListener("click", handleExecutionEnvironmentListClick);
$("executionEnvironmentList").addEventListener("keydown", handleExecutionEnvironmentListKeydown);
$("addLLMModelButton").onclick = addLLMModelProfile;
$("closeLLMEngineEditButton").onclick = closeLLMEngineEdit;
$("cancelLLMEngineEditButton").onclick = closeLLMEngineEdit;
$("saveLLMEngineEditButton").onclick = () =>
  runAction(saveLLMEngineEdit, { actionId: "llmSettings", busyText: "正在保存模型..." });
$("sidebarCollapseButton").onclick = toggleSidebarCollapsed;
$("sidebarBrandTrigger").onclick = expandSidebarFromBrand;
$("sidebarBrandTrigger").onkeydown = handleSidebarBrandKeydown;
$("createTaskButton").onclick = () =>
  runAction(createTaskAndScan);
$("workflowStepper").onclick = handleWorkflowStepperClick;
$("workflowStepper").onkeydown = handleWorkflowStepperKeydown;
$("taskSearchInput").oninput = (event) => {
  taskSearchQuery = event.target.value;
  renderTaskList();
};
$("taskSearchToggle").onclick = toggleTaskSearch;
$("taskSearchClose").onclick = () => closeTaskSearch({ focusToggle: true });
$("searchScrim").onclick = () => closeTaskSearch({ focusToggle: true });
$("taskList").addEventListener("click", () => closeTaskSearch());
$("settingsMenu").onchange = handleSettingsMenuChange;
$("agentMemoryList").addEventListener("click", handleAgentMemoryListClick);
document.addEventListener("click", handleAgentMemoryInlineInspect);
$("refreshAgentMemoryButton").onclick = () =>
  runAction(loadAgentMemoryItems, { actionId: "agentMemory", busyText: "正在读取 Agent 记忆..." });
$("memoryManageDetails").addEventListener("toggle", (event) => {
  if (event.target.open && !agentMemoryPanel.hasItems()) {
    runAction(loadAgentMemoryItems, { actionId: "agentMemory", busyText: "正在读取 Agent 记忆..." });
  }
});
$("draftManageDetails").addEventListener("toggle", (event) => {
  if (event.target.open && !draftToolsPanel.hasLoaded()) {
    runAction(loadDraftTools, { actionId: "draftTools", busyText: "正在读取草稿工具..." });
  }
});
$("draftStatusFilter").onchange = () =>
  runAction(loadDraftTools, { actionId: "draftTools", busyText: "正在读取草稿工具..." });
$("draftToolsList").addEventListener("click", handleDraftToolsListClick);
$("draftToolsList").addEventListener("keydown", handleDraftToolsListKeydown);
$("runDraftButton").onclick = () =>
  runAction(runDraftTool, { actionId: "draftTools", busyText: "正在试运行草稿..." });
$("promoteDraftButton").onclick = () =>
  runAction(promoteDraftTool, { actionId: "draftTools", busyText: "正在转正草稿..." });
$("rejectDraftButton").onclick = () =>
  runAction(rejectDraftTool, { actionId: "draftTools", busyText: "正在拒绝草稿..." });
$("llmModelProfiles").addEventListener("click", (event) => {
  const removeButton = event.target.closest("[data-llm-remove]");
  if (removeButton) {
    event.preventDefault();
    event.stopPropagation();
    runAction(() => removeLLMModelProfile(Number(removeButton.dataset.llmRemove)), {
      actionId: "llmSettings",
      busyText: "正在删除模型...",
    });
    return;
  }
  const editItem = event.target.closest("[data-llm-edit]");
  if (editItem) openLLMEngineEdit(Number(editItem.dataset.llmEdit));
});
$("llmModelProfiles").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const editItem = event.target.closest("[data-llm-edit]");
  if (!editItem) return;
  event.preventDefault();
  openLLMEngineEdit(Number(editItem.dataset.llmEdit));
});
$("agentModelSelect").onchange = (event) => {
  agentSelectedModelId = event.target.value;
  setAgentComposerNotice("");
  persistCurrentAgentComposerPreference({ model_id: agentSelectedModelId });
  event.target.blur();
};
$("agentEffortSelect").onchange = (event) => {
  agentSelectedEffort = normalizeAgentEffort(event.target.value);
  event.target.value = agentSelectedEffort;
  persistCurrentAgentComposerPreference({ effort: agentSelectedEffort });
  event.target.blur();
};
$("agentAcceptanceModeSelect").onchange = (event) => {
  agentAcceptanceMode = normalizeAgentAcceptanceMode(event.target.value);
  event.target.value = agentAcceptanceMode;
  renderAgentAcceptanceModePreference();
  persistCurrentAgentComposerPreference({ acceptance_mode: agentAcceptanceMode });
  event.target.blur();
};

// Per-task overrides take precedence; without a selected task the change
// belongs in the global preference store (used as the seed for new tasks).
function persistCurrentAgentComposerPreference(patch) {
  if (selectedTaskId) {
    updateAgentTaskComposerOverride(selectedTaskId, patch);
  } else {
    saveAgentComposerPreferences();
  }
}
function blurChipSelectIfFocused() {
  const focused = document.activeElement;
  if (!focused) return;
  if (!agentComposerSelectIds.includes(focused.id)) return;
  focused.blur();
}
const agentComposerSelectIds = ["agentAcceptanceModeSelect", "agentModelSelect", "agentEffortSelect"];
document.addEventListener(
  "mousedown",
  (event) => {
    const focused = document.activeElement;
    if (!focused) return;
    if (!agentComposerSelectIds.includes(focused.id)) return;
    const chip = focused.closest(".agent-composer-chip");
    if (chip && !chip.contains(event.target)) focused.blur();
  },
  true,
);
window.addEventListener("focus", () => {
  setTimeout(blurChipSelectIfFocused, 0);
});
for (const id of agentComposerSelectIds) {
  $(id).addEventListener("keyup", (event) => {
    if (event.key === "Escape") event.currentTarget.blur();
  });
}
$("sendAgentMessageButton").onclick = () => {
  if (agentSendIsStopMode()) {
    runAction(stopAgentValidation, { actionId: "agent", busyText: "Agent 正在停止..." });
    return;
  }
  runAction(startAgentValidation, { actionId: "agent", busyText: "Agent 正在处理..." });
};
$("agentComposerInput").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  if (agentSendIsStopMode()) return;
  if ($("sendAgentMessageButton")?.disabled) return;
  runAction(startAgentValidation, { actionId: "agent", busyText: "Agent 正在处理..." });
});
$("agentComposerInput").addEventListener("input", () => {
  autoGrowComposerInput();
  updateAgentSendDisabled();
});

function autoGrowComposerInput() {
  const input = $("agentComposerInput");
  if (!input) return;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  requestAnimationFrame(syncAgentComposerClearance);
}

function agentSendIsStopMode() {
  return Boolean(selectedTaskIsAgentMode() && taskBusyAction(selectedTaskId) === "agent");
}

function renderAgentSendButtonState() {
  const button = $("sendAgentMessageButton");
  if (!button) return false;
  const stopMode = agentSendIsStopMode();
  button.dataset.agentSendState = stopMode ? "stop" : "send";
  button.setAttribute("aria-label", stopMode ? "停止当前 Agent 动作" : "发送消息");
  button.title = stopMode ? "停止当前 Agent 动作" : "";
  return stopMode;
}

// Send is disabled until the user has typed something; while Agent is running,
// the same control becomes an always-enabled stop button.
function updateAgentSendDisabled() {
  const input = $("agentComposerInput");
  const button = $("sendAgentMessageButton");
  if (!input || !button) return;
  const stopMode = renderAgentSendButtonState();
  button.disabled = stopMode ? false : !input.value.trim();
}

updateAgentSendDisabled();

function agentEffort() {
  agentSelectedEffort = normalizeAgentEffort($("agentEffortSelect")?.value || agentSelectedEffort);
  return agentSelectedEffort;
}

function agentAcceptanceModeValue() {
  agentAcceptanceMode = normalizeAgentAcceptanceMode($("agentAcceptanceModeSelect")?.value || agentAcceptanceMode);
  return agentAcceptanceMode;
}
bindRunModeDeselectableCards();
bindDialogBackdropDismissal();
bindPlatformConfirmDialog();
mountGovernanceExtensions();
onSelectedTierChange(syncCreateTaskTierDefault);
createTaskDialog.bindMaterialSourceControls();
const pet = $("petCompanion");
if (pet) pet.addEventListener("pointerdown", startPetDrag);
$("leftResizeHandle").onpointerdown = (event) => startResizeDrag("left", event);
$("rightResizeHandle").onpointerdown = (event) => startResizeDrag("right", event);
$("leftResizeHandle").onkeydown = (event) => handleResizeKey("left", event);
$("rightResizeHandle").onkeydown = (event) => handleResizeKey("right", event);
$("resultScrollContent").addEventListener("scroll", handleResultScroll, { passive: true });
document.addEventListener("wheel", routeWorkspaceWheelToResult, { passive: false });
// Note real user-driven scroll inputs so recomputeAgentAutoScrollFollow can
// distinguish them from typewriter/restore-driven programmatic scrolls.
document.addEventListener("wheel", noteAgentUserScrollInput, { passive: true });
document.addEventListener("touchstart", noteAgentUserScrollInput, { passive: true });
document.addEventListener("touchmove", noteAgentUserScrollInput, { passive: true });
window.addEventListener("resize", syncTaskHeroGlassLayout);

$("taskList").onkeydown = handleTaskListKeydown;

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && taskSearchActive) {
    closeTaskSearch({ focusToggle: true });
    return;
  }
  if (
    event.key === "Enter" &&
    event.target.closest("#taskDialog") &&
    event.target.tagName !== "TEXTAREA" &&
    !event.isComposing
  ) {
    event.preventDefault();
    runAction(createTaskAndScan);
  }
});

document.addEventListener("click", (event) => {
  const copyButton = event.target.closest("[data-copy]");
  if (copyButton) {
    event.preventDefault();
    copyText(copyButton.dataset.copy);
  }
});
document.addEventListener("click", closeSidebarSettingsOnOutsideClick);

installFormControlFocusRingGuard();
themeController.restoreTheme();
themeController.watchSystemTheme();
restoreTaskListSettings();
restorePetPreference();
restorePetPosition();
restoreLayoutWidths();
restoreSidebarCollapsed();
updateWorkspaceGreeting();
setInterval(updateWorkspaceGreeting, 60 * 1000);
renderSettingsState();
loadBranding();
loadExecutionEnvironmentSettings({ silent: true });
loadLLMSettings({ silent: true });
loadResultScrollPositions();
restoreSelectedTaskPlaceholder();
renderCurrentTask({ force: true });
renderMetricPreview({});
renderStoredStateSummaries();
initializeApp();

function enableAppAnimationsAfterBoot() {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.body.classList.add("anim-ready");
    });
  });
}

function finishAppBoot() {
  document.body.classList.remove("app-booting");
  enableAppAnimationsAfterBoot();
}

async function initializeApp() {
  try {
    await refreshTasks();
    renderStoredStateSummaries();
    await loadReportFields();
    await loadTaskEvidence();
    await loadAgentMessages();
  } catch (error) {
    const detail = error?.message || "";
    setActionStatus("服务连接失败，请检查后端是否运行。", "error", detail);
    setCreateStatus(detail || "服务连接失败，请检查后端是否运行。", "error");
  } finally {
    renderAll();
    await restoreResultScrollPositionAfterRender(selectedTaskId);
    finishAppBoot();
  }
}
