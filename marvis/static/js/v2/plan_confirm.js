import { escapeHtml } from "../ui-utils.js";
import {
  cancelPlan as cancelPlanApi,
  confirmPlan as confirmPlanApi,
  confirmStep as confirmStepApi,
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

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

export function attachPlanConfirmHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachPlanConfirmHandlers requires a stable event root");
  }
  const actions = {
    cancelPlan: cancelPlanApi,
    confirmPlan: confirmPlanApi,
    confirmStep: confirmStepApi,
    getCurrentPlan,
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
      try {
        syncPlanFromPayload(await actions.confirmPlan(planId));
        await actions.runPlan(planId);
        actions.startPlanPolling(planId);
      } catch (error) {
        actions.showError(error?.message || "confirm plan failed");
      }
      return;
    }

    const stepButton = closest(target, "[data-confirm-step]");
    if (stepButton?.dataset?.confirmStep) {
      event.preventDefault?.();
      const plan = actions.getCurrentPlan();
      if (!plan?.id) {
        return;
      }
      try {
        await actions.confirmStep(plan.id, stepButton.dataset.confirmStep);
        actions.startPlanPolling(plan.id);
      } catch (error) {
        actions.showError(error?.message || "confirm step failed");
      }
      return;
    }

    const cancelButton = closest(target, "[data-cancel-plan]");
    if (cancelButton?.dataset?.cancelPlan) {
      event.preventDefault?.();
      const planId = cancelButton.dataset.cancelPlan;
      try {
        syncPlanFromPayload(await actions.cancelPlan(planId));
        actions.stopPlanPolling(planId);
      } catch (error) {
        actions.showError(error?.message || "cancel plan failed");
      }
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
    container.innerHTML = '<div class="v2-empty" data-v2-empty="plan-problems">No plan validation problems</div>';
    return;
  }
  const items = problems
    .map((problem) => `<li>${escapeHtml(problemText(problem))}</li>`)
    .join("");
  container.innerHTML = `<ul class="plan-problems">${items}</ul>`;
}
