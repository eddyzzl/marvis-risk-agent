import { escapeHtml } from "../ui-utils.js";
import { getPlan as fetchPlan } from "./api_v2.js";
import {
  getPlan as getCurrentPlan,
  onPlanChange,
  setPlan,
} from "./state_v2.js";

export const terminalPlanStatuses = new Set(["done", "failed", "cancelled", "review"]);

const planPolls = new Map();

const planStatusLabels = {
  draft: "Draft",
  validated: "Validated",
  confirmed: "Confirmed",
  running: "Running",
  awaiting_confirm: "Awaiting confirm",
  review: "Review",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

const stepStatusLabels = {
  pending: "Pending",
  blocked: "Blocked",
  awaiting_confirm: "Awaiting confirm",
  running: "Running",
  checking: "Checking",
  done: "Done",
  failed: "Failed",
  skipped: "Skipped",
};

function classToken(value, fallback = "unknown") {
  return String(value || fallback)
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "") || fallback;
}

function sortedSteps(plan) {
  return [...(plan?.steps || [])].sort((left, right) => {
    const leftIndex = Number.isFinite(Number(left?.index)) ? Number(left.index) : 0;
    const rightIndex = Number.isFinite(Number(right?.index)) ? Number(right.index) : 0;
    if (leftIndex !== rightIndex) {
      return leftIndex - rightIndex;
    }
    return String(left?.id || "").localeCompare(String(right?.id || ""));
  });
}

function toolLabel(step) {
  const ref = step?.tool_ref || {};
  if (!ref.plugin || !ref.tool) {
    return "tool pending";
  }
  return `${ref.plugin}.${ref.tool}`;
}

export function planProgress(plan) {
  const steps = sortedSteps(plan);
  if (!steps.length) {
    return 0;
  }
  const complete = steps.filter((step) => ["done", "skipped"].includes(step.status)).length;
  return Math.round((complete / steps.length) * 100);
}

export function planStatusBadge(status) {
  const normalized = classToken(status);
  const label = planStatusLabels[status] || status || "Unknown";
  return `<span class="plan-status-badge plan-status-${normalized}">${escapeHtml(label)}</span>`;
}

export function stepStatusBadge(status) {
  const normalized = classToken(status);
  const label = stepStatusLabels[status] || status || "Unknown";
  return `<span class="step-status step-status-${normalized}">${escapeHtml(label)}</span>`;
}

export function reviewVerdictHtml(verdicts = []) {
  if (!verdicts.length) {
    return "";
  }
  const items = verdicts.map((verdict) => {
    const reviewer = classToken(verdict.reviewer, "reviewer");
    const passed = Boolean(verdict.passed);
    const hardFail = reviewer === "deterministic" && !passed;
    const reasons = (verdict.reasons || [])
      .map((reason) => `<li>${escapeHtml(reason)}</li>`)
      .join("");
    return `<li class="review-verdict reviewer-${reviewer} ${passed ? "passed" : "failed"}${hardFail ? " hard-fail" : ""}">
      <span class="reviewer">${escapeHtml(verdict.reviewer || "reviewer")}</span>
      <span class="verdict">${passed ? "Passed" : "Failed"}</span>
      ${reasons ? `<ul class="review-reasons">${reasons}</ul>` : ""}
    </li>`;
  }).join("");
  return `<ul class="review-verdicts">${items}</ul>`;
}

export function stepRowHtml(step) {
  const status = step?.status || "pending";
  const index = Number.isFinite(Number(step?.index)) ? Number(step.index) + 1 : "";
  const depends = (step?.depends_on || []).map((item) => escapeHtml(item)).join(", ");
  const decisionPoint = step?.decision_point
    ? '<span class="dp-mark" data-tip="Decision point">◆</span>'
    : "";
  const confirm = status === "awaiting_confirm"
    ? `<button type="button" data-confirm-step="${escapeHtml(step.id)}">Confirm</button>`
    : "";
  const verdicts = reviewVerdictHtml(step?.review_verdicts || []);
  return `<li data-step="${escapeHtml(step?.id || "")}" class="plan-step step-${classToken(status)}">
    <span class="idx">${escapeHtml(index)}</span>
    <span class="title">${escapeHtml(step?.title || "Untitled step")}</span>
    <span class="tool">${escapeHtml(toolLabel(step))}</span>
    ${depends ? `<span class="depends">after ${depends}</span>` : ""}
    ${decisionPoint}
    ${stepStatusBadge(status)}
    ${confirm}
    ${verdicts}
  </li>`;
}

export function progressBarHtml(plan) {
  const progress = planProgress(plan);
  return `<div class="plan-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}">
    <span class="plan-progress-bar" style="width: ${progress}%"></span>
  </div>`;
}

export function planHtml(plan) {
  if (!plan) {
    return emptyPlanHtml();
  }
  const status = plan.status || "draft";
  const novelMode = plan.novel_mode || "plan_ahead";
  const steps = sortedSteps(plan).map(stepRowHtml).join("");
  return `<section class="v2-plan plan-status-${classToken(status)} novel-${classToken(novelMode)}" data-plan-id="${escapeHtml(plan.id || "")}">
    <header class="plan-header">
      <div class="plan-goal">${escapeHtml(plan.goal || "Untitled plan")}</div>
      <div class="plan-meta">
        ${planStatusBadge(status)}
        <span class="plan-tier">${escapeHtml(plan.tier || "balanced")}</span>
        <span class="plan-mode">${escapeHtml(novelMode)}</span>
      </div>
    </header>
    <ol class="plan-steps">${steps}</ol>
    ${progressBarHtml(plan)}
  </section>`;
}

export function emptyPlanHtml() {
  return '<div class="v2-empty" data-v2-empty="plan">No active V2 plan</div>';
}

export function renderPlanView(container) {
  if (!container) {
    throw new Error("renderPlanView requires a container");
  }
  if (container.dataset) {
    container.dataset.v2PlanView = "true";
  }
  const render = (plan) => {
    container.innerHTML = planHtml(plan);
  };
  render(getCurrentPlan());
  return onPlanChange((plan) => render(plan));
}

function deactivatePlanPoll(planId, pollState) {
  if (!pollState || pollState.stopped) {
    return;
  }
  pollState.stopped = true;
  if (pollState.timer !== null && pollState.timer !== undefined) {
    pollState.clearTimeoutFn(pollState.timer);
  }
  if (planPolls.get(planId) === pollState) {
    planPolls.delete(planId);
  }
}

export function startPlanPolling(planId, options = {}) {
  const key = String(planId);
  const existing = planPolls.get(key);
  if (existing) {
    return existing;
  }
  const intervalMs = Number.isFinite(Number(options.intervalMs))
    ? Number(options.intervalMs)
    : 1000;
  const pollState = {
    planId: key,
    stopped: false,
    timer: null,
    setTimeoutFn: options.setTimeoutFn || ((fn, ms) => setTimeout(fn, ms)),
    clearTimeoutFn: options.clearTimeoutFn || ((timerId) => clearTimeout(timerId)),
    stop() {
      stopPlanPolling(key);
    },
    async tick() {
      if (pollState.stopped) {
        return null;
      }
      try {
        const payload = await fetchPlan(key);
        const plan = payload?.plan || payload;
        if (plan) {
          setPlan(plan);
        }
        if (terminalPlanStatuses.has(plan?.status)) {
          deactivatePlanPoll(key, pollState);
          return plan;
        }
        pollState.timer = pollState.setTimeoutFn(pollState.tick, intervalMs);
        return plan;
      } catch (error) {
        deactivatePlanPoll(key, pollState);
        throw error;
      }
    },
  };
  planPolls.set(key, pollState);
  if (options.autoStart !== false) {
    void pollState.tick();
  }
  return pollState;
}

export function stopPlanPolling(planId) {
  const key = String(planId);
  deactivatePlanPoll(key, planPolls.get(key));
}
