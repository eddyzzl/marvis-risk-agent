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
      syncPlanFromPayload(await actions.confirmPlan(planId));
      await actions.runPlan(planId);
      actions.startPlanPolling(planId);
      return;
    }

    const stepButton = closest(target, "[data-confirm-step]");
    if (stepButton?.dataset?.confirmStep) {
      event.preventDefault?.();
      const plan = actions.getCurrentPlan();
      if (!plan?.id) {
        return;
      }
      await actions.confirmStep(plan.id, stepButton.dataset.confirmStep);
      actions.startPlanPolling(plan.id);
      return;
    }

    const cancelButton = closest(target, "[data-cancel-plan]");
    if (cancelButton?.dataset?.cancelPlan) {
      event.preventDefault?.();
      const planId = cancelButton.dataset.cancelPlan;
      syncPlanFromPayload(await actions.cancelPlan(planId));
      actions.stopPlanPolling(planId);
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
