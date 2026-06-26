import { escapeHtml } from "../ui-utils.js";
import { createPlan as createPlanApi } from "./api_v2.js";
import { renderPlanValidationProblems } from "./plan_confirm.js";
import {
  getCapabilityTiers,
  getSelectedTier,
  onCapabilityTiersChange,
  onSelectedTierChange,
  setPlan,
} from "./state_v2.js";

const defaultTiers = [
  { name: "conservative", summary: "保守执行" },
  { name: "balanced", summary: "默认自治" },
  { name: "autonomous", summary: "更高自治" },
];

const tierLabels = {
  deterministic_only: "仅确定性",
  guarded: "受控 Agent",
  conservative: "稳健",
  balanced: "均衡",
  explorer: "探索",
  autonomous: "自治",
};

const novelModeLabels = {
  "": "自动选择",
  plan_ahead: "先规划",
  explore: "先探索",
  exploratory: "探索式",
  reactive: "边执行边调整",
};

function tierLabel(tier = {}) {
  const name = String(tier.name || "");
  const label = tierLabels[name] || name || "未命名档位";
  const summary = String(tier.summary || "").trim();
  return summary ? `${label} - ${summary}` : label;
}

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function controlValue(root, selector) {
  return String(root.querySelector?.(selector)?.value || "").trim();
}

function resolveTaskId(taskId) {
  const value = typeof taskId === "function" ? taskId() : taskId;
  return String(value || "").trim();
}

function tierOptionsHtml(tiers, defaultTier) {
  return tiers.map((tier) => {
    const selected = tier.name === defaultTier ? " selected" : "";
    return `<option value="${escapeHtml(tier.name)}"${selected}>${escapeHtml(tierLabel(tier))}</option>`;
  }).join("");
}

function novelOptionHtml(value, label, selectedValue) {
  const selected = value === selectedValue ? " selected" : "";
  return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(novelModeLabels[value] || label)}</option>`;
}

export function goalComposerHtml(options = {}) {
  const tiers = options.tiers?.length ? options.tiers : defaultTiers;
  const defaultTier = options.defaultTier || "balanced";
  const goal = String(options.goal || "");
  const novelMode = String(options.novelMode || "");
  return `<section class="goal-composer">
    <textarea id="goalInput" placeholder="描述本次 Workflow 目标">${escapeHtml(goal)}</textarea>
    <label>
      能力档位
      <select id="tierSelect">${tierOptionsHtml(tiers, defaultTier)}</select>
    </label>
    <label>
      规划方式
      <select id="novelMode">
        ${novelOptionHtml("", "auto", novelMode)}
        ${novelOptionHtml("plan_ahead", "plan_ahead", novelMode)}
        ${novelOptionHtml("explore", "explore", novelMode)}
      </select>
    </label>
    <button id="createPlanBtn" type="button" data-create-plan>生成执行计划</button>
    <div data-plan-problems></div>
  </section>`;
}

function currentComposerValues(container) {
  return {
    goal: controlValue(container, "#goalInput"),
    tier: controlValue(container, "#tierSelect"),
    novelMode: controlValue(container, "#novelMode"),
  };
}

export function renderGoalComposer(container, options = {}) {
  if (!container) {
    throw new Error("renderGoalComposer requires a container");
  }
  if (container.dataset) {
    container.dataset.v2GoalComposer = "true";
  }
  const render = () => {
    const current = currentComposerValues(container);
    const tiers = getCapabilityTiers();
    container.innerHTML = goalComposerHtml({
      ...options,
      goal: current.goal,
      tiers: tiers.length ? tiers : options.tiers,
      defaultTier: getSelectedTier() || current.tier || options.defaultTier,
      novelMode: current.novelMode,
    });
  };
  render();
  const cleanups = [
    onCapabilityTiersChange(() => render()),
    onSelectedTierChange(() => render()),
  ];
  return () => cleanups.forEach((cleanup) => cleanup());
}

function validationProblems(error) {
  if (Array.isArray(error?.detail?.problems)) {
    return error.detail.problems;
  }
  if (Array.isArray(error?.detail)) {
    return error.detail;
  }
  return [error?.message || "计划校验失败"];
}

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

export function attachGoalHandlers(root, taskId, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachGoalHandlers requires a stable event root");
  }
  const actions = {
    createPlan: createPlanApi,
    showError: defaultShowError,
    ...deps,
  };

  const handler = async (event) => {
    const createButton = closest(event.target, "#createPlanBtn")
      || closest(event.target, "[data-create-plan]");
    if (!createButton) {
      return;
    }
    event.preventDefault?.();
    const resolvedTaskId = resolveTaskId(taskId);
    if (!resolvedTaskId) {
      const problemSlot = root.querySelector?.("[data-plan-problems]");
      const message = "请先选择或创建任务，再生成 V2 计划。";
      if (problemSlot) {
        renderPlanValidationProblems(problemSlot, [message]);
      } else {
        actions.showError(message);
      }
      return;
    }
    const goal = controlValue(root, "#goalInput");
    const tier = controlValue(root, "#tierSelect");
    const novelMode = controlValue(root, "#novelMode");
    const body = { goal };
    if (tier) body.tier = tier;
    if (novelMode) body.novel_mode = novelMode;
    try {
      const payload = await actions.createPlan(resolvedTaskId, body);
      const plan = payload?.plan || payload;
      if (plan) {
        setPlan(plan);
      }
      const problemSlot = root.querySelector?.("[data-plan-problems]");
      if (problemSlot) {
        renderPlanValidationProblems(problemSlot, []);
      }
    } catch (error) {
      if (error?.status === 422) {
        const problemSlot = root.querySelector?.("[data-plan-problems]");
        if (problemSlot) {
          renderPlanValidationProblems(problemSlot, validationProblems(error));
          return;
        }
      }
      actions.showError(error?.message || "创建计划失败");
    }
  };

  root.addEventListener("click", handler);
  return () => root.removeEventListener?.("click", handler);
}
