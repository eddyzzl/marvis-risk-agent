import { escapeHtml } from "../ui-utils.js";
import {
  cancelPlan as cancelPlanApi,
  confirmPlan as confirmPlanApi,
  confirmStep as confirmStepApi,
  retryStep as retryStepApi,
  runPlan as runPlanApi,
} from "./api_v2.js";
import { startPlanPolling, stopPlanPolling } from "./plan_view.js";
import { getPlan as getCurrentPlan, setPlan } from "./state_v2.js";

function problemText(problem) {
  if (typeof problem === "string") {
    return problem;
  }
  if (problem && typeof problem === "object") {
    return problem.message || problem.msg || JSON.stringify(problem);
  }
  return String(problem ?? "");
}

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function syncPlanFromPayload(payload) {
  const plan = payload?.plan || payload;
  if (plan?.id) {
    setPlan(plan);
  }
}

function parseJsonObject(text) {
  let value;
  try {
    value = JSON.parse(text);
  } catch (_error) {
    throw new Error("重试参数必须是合法 JSON");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("重试参数必须是 JSON 对象");
  }
  return value;
}

function parseRetryFieldValue(field) {
  const type = String(field?.dataset?.retryInputType || "string");
  const raw = String(field?.value ?? "");
  if (type === "boolean") {
    return raw === "true";
  }
  if (type === "integer") {
    const value = Number.parseInt(raw, 10);
    if (!Number.isFinite(value)) {
      throw new Error("整数重试参数无效");
    }
    return value;
  }
  if (type === "number") {
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      throw new Error("数值重试参数无效");
    }
    return value;
  }
  if (type === "array") {
    let value;
    try {
      value = JSON.parse(raw || "[]");
    } catch (_error) {
      throw new Error("数组重试参数无效");
    }
    if (!Array.isArray(value)) {
      throw new Error("数组重试参数无效");
    }
    return value;
  }
  if (type === "object") {
    try {
      return parseJsonObject(raw || "{}");
    } catch (_error) {
      throw new Error("对象重试参数无效");
    }
  }
  if (type === "null") {
    return null;
  }
  return raw;
}

function readStructuredRetryInputs(panel) {
  const fields = Array.from(panel?.querySelectorAll?.("[data-retry-input-key]") || []);
  if (!fields.length) {
    return null;
  }
  const inputs = {};
  fields.forEach((field) => {
    const key = String(field?.dataset?.retryInputKey || "");
    if (!key) {
      return;
    }
    inputs[key] = parseRetryFieldValue(field);
  });
  return inputs;
}

function defaultReadRetryInputs(root, retryButton, stepId) {
  const panel = closest(retryButton, "[data-retry-panel]");
  const structured = readStructuredRetryInputs(panel);
  if (structured) {
    return structured;
  }
  const panelField = panel?.querySelector?.("[data-retry-inputs-for]");
  const field = panelField || Array.from(root?.querySelectorAll?.("[data-retry-inputs-for]") || [])
    .find((candidate) => candidate?.dataset?.retryInputsFor === stepId);
  if (!field) {
    return undefined;
  }
  return parseJsonObject(String(field.value || "{}"));
}

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

async function withButtonLock(button, key, inflight, fn) {
  if (inflight.has(key)) {
    return;
  }
  inflight.add(key);
  const previousDisabled = button.disabled;
  button.disabled = true;
  button.setAttribute?.("aria-busy", "true");
  try {
    await fn();
  } finally {
    inflight.delete(key);
    button.disabled = previousDisabled;
    button.removeAttribute?.("aria-busy");
  }
}

export function attachPlanConfirmHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachPlanConfirmHandlers requires a stable event root");
  }
  const inflight = new Set();
  const actions = {
    cancelPlan: cancelPlanApi,
    confirmPlan: confirmPlanApi,
    confirmStep: confirmStepApi,
    getCurrentPlan,
    readRetryInputs: defaultReadRetryInputs,
    retryStep: retryStepApi,
    runPlan: runPlanApi,
    showError: defaultShowError,
    startPlanPolling,
    stopPlanPolling,
    ...deps,
  };

  const handler = async (event) => {
    const target = event.target;
    const planButton = closest(target, "[data-confirm-plan]");
    if (planButton?.dataset?.confirmPlan) {
      event.preventDefault?.();
      const planId = planButton.dataset.confirmPlan;
      await withButtonLock(planButton, `confirm-plan:${planId}`, inflight, async () => {
        try {
          syncPlanFromPayload(await actions.confirmPlan(planId));
          await actions.runPlan(planId);
          actions.startPlanPolling(planId);
        } catch (error) {
          actions.showError(error?.message || "计划确认失败");
        }
      });
      return;
    }

    const stepButton = closest(target, "[data-confirm-step]");
    if (stepButton?.dataset?.confirmStep) {
      event.preventDefault?.();
      const plan = actions.getCurrentPlan();
      if (!plan?.id) {
        return;
      }
      const stepId = stepButton.dataset.confirmStep;
      await withButtonLock(stepButton, `confirm-step:${plan.id}:${stepId}`, inflight, async () => {
        try {
          await actions.confirmStep(plan.id, stepId);
          actions.startPlanPolling(plan.id);
        } catch (error) {
          actions.showError(error?.message || "步骤确认失败");
        }
      });
      return;
    }

    const retryButton = closest(target, "[data-retry-step]");
    if (retryButton?.dataset?.retryStep) {
      event.preventDefault?.();
      const plan = actions.getCurrentPlan();
      if (!plan?.id) {
        return;
      }
      const stepId = retryButton.dataset.retryStep;
      await withButtonLock(retryButton, `retry-step:${plan.id}:${stepId}`, inflight, async () => {
        try {
          const inputs = actions.readRetryInputs(root, retryButton, stepId, plan);
          await actions.retryStep(plan.id, stepId, inputs);
          actions.startPlanPolling(plan.id);
        } catch (error) {
          actions.showError(error?.message || "步骤重试失败");
        }
      });
      return;
    }

    const cancelButton = closest(target, "[data-cancel-plan]");
    if (cancelButton?.dataset?.cancelPlan) {
      event.preventDefault?.();
      const planId = cancelButton.dataset.cancelPlan;
      await withButtonLock(cancelButton, `cancel-plan:${planId}`, inflight, async () => {
        try {
          syncPlanFromPayload(await actions.cancelPlan(planId));
          actions.stopPlanPolling(planId);
        } catch (error) {
          actions.showError(error?.message || "计划取消失败");
        }
      });
    }
  };

  root.addEventListener("click", handler);
  return () => root.removeEventListener?.("click", handler);
}

export function renderPlanValidationProblems(container, problems = []) {
  if (!container) {
    throw new Error("renderPlanValidationProblems requires a container");
  }
  if (container.dataset) {
    container.dataset.v2PlanProblems = "true";
  }
  if (!problems.length) {
    container.innerHTML = '<div class="v2-empty" data-v2-empty="plan-problems">暂无计划校验问题</div>';
    return;
  }
  const items = problems
    .map((problem) => `<li>${escapeHtml(problemText(problem))}</li>`)
    .join("");
  container.innerHTML = `<ul class="plan-problems">${items}</ul>`;
}
