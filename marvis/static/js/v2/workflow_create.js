import { escapeHtml } from "../ui-utils.js";
import { createPlan as createPlanApi } from "./api_v2.js";
import { renderPlanValidationProblems } from "./plan_confirm.js";
import { setPlan } from "./state_v2.js";

const defaultTiers = [
  { name: "conservative", summary: "Guarded execution" },
  { name: "balanced", summary: "Default autonomy" },
  { name: "autonomous", summary: "Higher autonomy" },
];

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
    return `<option value="${escapeHtml(tier.name)}"${selected}>${escapeHtml(tier.name)} - ${escapeHtml(tier.summary || "")}</option>`;
  }).join("");
}

export function goalComposerHtml(options = {}) {
  const tiers = options.tiers?.length ? options.tiers : defaultTiers;
  const defaultTier = options.defaultTier || "balanced";
  return `<section class="goal-composer">
    <textarea id="goalInput" placeholder="Describe the workflow goal"></textarea>
    <label>
      Capability tier
      <select id="tierSelect">${tierOptionsHtml(tiers, defaultTier)}</select>
    </label>
    <label>
      Novel mode
      <select id="novelMode">
        <option value="">auto</option>
        <option value="plan_ahead">plan_ahead</option>
        <option value="explore">explore</option>
      </select>
    </label>
    <button id="createPlanBtn" type="button" data-create-plan>Create plan</button>
    <div data-plan-problems></div>
  </section>`;
}

export function renderGoalComposer(container, options = {}) {
  if (!container) {
    throw new Error("renderGoalComposer requires a container");
  }
  if (container.dataset) {
    container.dataset.v2GoalComposer = "true";
  }
  container.innerHTML = goalComposerHtml(options);
  return () => {};
}

function validationProblems(error) {
  if (Array.isArray(error?.detail?.problems)) {
    return error.detail.problems;
  }
  if (Array.isArray(error?.detail)) {
    return error.detail;
  }
  return [error?.message || "plan validation failed"];
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
      const message = "select or create a task before creating a V2 plan";
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
      actions.showError(error?.message || "create plan failed");
    }
  };

  root.addEventListener("click", handler);
  return () => root.removeEventListener?.("click", handler);
}
