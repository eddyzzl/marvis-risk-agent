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
  draft: "草稿",
  validated: "已校验",
  confirmed: "已确认",
  running: "运行中",
  awaiting_confirm: "待确认",
  review: "待复核",
  done: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

const stepStatusLabels = {
  pending: "待执行",
  blocked: "已阻塞",
  awaiting_confirm: "待确认",
  running: "运行中",
  checking: "检查中",
  done: "已完成",
  failed: "失败",
  skipped: "已跳过",
};

const tierLabels = {
  deterministic_only: "仅确定性",
  guarded: "受控 Agent",
  balanced: "均衡",
  explorer: "探索",
  autonomous: "自动化",
};

const novelModeLabels = {
  plan_ahead: "先规划",
  reactive: "边执行边调整",
  exploratory: "探索式",
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
    return "工具待定";
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
  const label = planStatusLabels[status] || status || "未知";
  return `<span class="plan-status-badge plan-status-${normalized}">${escapeHtml(label)}</span>`;
}

export function stepStatusBadge(status) {
  const normalized = classToken(status);
  const label = stepStatusLabels[status] || status || "未知";
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
    const softWarning = !passed && !hardFail;
    const reasons = (verdict.reasons || [])
      .map((reason) => `<li>${escapeHtml(reason)}</li>`)
      .join("");
    return `<li class="review-verdict reviewer-${reviewer} ${passed ? "passed" : "failed"}${hardFail ? " hard-fail" : ""}${softWarning ? " soft-warning" : ""}">
      <span class="reviewer">${escapeHtml(verdict.reviewer || "审查器")}</span>
      <span class="verdict">${passed ? "通过" : softWarning ? "警告" : "失败"}</span>
      ${reasons ? `<ul class="review-reasons">${reasons}</ul>` : ""}
    </li>`;
  }).join("");
  return `<ul class="review-verdicts">${items}</ul>`;
}

function outputRefHtml(step) {
  const outputRef = String(step?.output_ref || "");
  if (!outputRef) {
    return "";
  }
  return `<button type="button" class="step-output-button" data-artifact="${escapeHtml(outputRef)}">查看输出</button>`;
}

function retryInputText(step) {
  const schema = step?.failure_envelope?.editable_input_schema;
  const properties = schema && typeof schema === "object" ? schema.properties : null;
  if (properties && typeof properties === "object") {
    const inputs = {};
    Object.entries(properties).forEach(([key, spec]) => {
      if (spec && typeof spec === "object" && Object.prototype.hasOwnProperty.call(spec, "default")) {
        inputs[key] = spec.default;
      }
    });
    if (Object.keys(inputs).length) {
      return JSON.stringify(inputs, null, 2);
    }
  }
  return JSON.stringify(step?.inputs || {}, null, 2);
}

function retryScopeHtml(step) {
  const resetSteps = Array.isArray(step?.failure_envelope?.downstream_reset_steps)
    ? step.failure_envelope.downstream_reset_steps.filter(Boolean)
    : [];
  if (!resetSteps.length) {
    return "";
  }
  return `<p class="retry-step-scope">将重置 ${resetSteps.map((item) => `<code>${escapeHtml(item)}</code>`).join("、")}</p>`;
}

function retryPanelHtml(step) {
  const stepId = String(step?.id || "");
  const inputText = retryInputText(step);
  return `<details class="retry-step-panel" data-retry-panel="${escapeHtml(stepId)}">
    <summary>重试步骤</summary>
    ${retryScopeHtml(step)}
    <label>
      参数 JSON
      <textarea data-retry-inputs-for="${escapeHtml(stepId)}" rows="5" spellcheck="false">${escapeHtml(inputText)}</textarea>
    </label>
    <button type="button" data-retry-step="${escapeHtml(stepId)}">使用这些参数重试</button>
  </details>`;
}

export function stepRowHtml(step) {
  const status = step?.status || "pending";
  const index = Number.isFinite(Number(step?.index)) ? Number(step.index) + 1 : "";
  const depends = (step?.depends_on || []).map((item) => escapeHtml(item)).join(", ");
  const decisionPoint = step?.decision_point
    ? '<span class="dp-mark" data-tip="决策点">◆</span>'
    : "";
  const confirm = status === "awaiting_confirm"
    ? `<button type="button" data-confirm-step="${escapeHtml(step.id)}">确认步骤</button>`
    : "";
  const retry = status === "failed"
    ? retryPanelHtml(step)
    : "";
  const verdicts = reviewVerdictHtml(step?.review_verdicts || []);
  const output = outputRefHtml(step);
  return `<li data-step="${escapeHtml(step?.id || "")}" class="plan-step step-${classToken(status)}">
    <span class="idx">${escapeHtml(index)}</span>
    <span class="title">${escapeHtml(step?.title || "未命名步骤")}</span>
    <span class="tool">${escapeHtml(toolLabel(step))}</span>
    ${depends ? `<span class="depends">依赖 ${depends}</span>` : ""}
    ${decisionPoint}
    ${stepStatusBadge(status)}
    ${confirm}
    ${retry}
    ${output}
    ${verdicts}
  </li>`;
}

export function progressBarHtml(plan) {
  const progress = planProgress(plan);
  return `<div class="plan-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}">
    <span class="plan-progress-bar" style="width: ${progress}%"></span>
  </div>`;
}

export function planActionsHtml(plan) {
  const planId = String(plan?.id || "");
  if (!planId) {
    return "";
  }
  const status = plan?.status || "draft";
  const confirm = status === "validated"
    ? `<button type="button" data-confirm-plan="${escapeHtml(planId)}">确认并运行</button>`
    : "";
  const cancel = terminalPlanStatuses.has(status)
    ? ""
    : `<button type="button" data-cancel-plan="${escapeHtml(planId)}">取消</button>`;
  if (!confirm && !cancel) {
    return "";
  }
  return `<div class="plan-actions">${confirm}${cancel}</div>`;
}

export function planHtml(plan) {
  if (!plan) {
    return emptyPlanHtml();
  }
  const status = plan.status || "draft";
  const novelMode = plan.novel_mode || "plan_ahead";
  const tierLabel = tierLabels[plan.tier] || plan.tier || "均衡";
  const novelModeLabel = novelModeLabels[novelMode] || novelMode;
  const steps = sortedSteps(plan).map(stepRowHtml).join("");
  return `<section class="v2-plan plan-status-${classToken(status)} novel-${classToken(novelMode)}" data-plan-id="${escapeHtml(plan.id || "")}">
    <header class="plan-header">
      <div class="plan-goal">${escapeHtml(plan.goal || "未命名计划")}</div>
      <div class="plan-meta">
        ${planStatusBadge(status)}
        <span class="plan-tier">${escapeHtml(tierLabel)}</span>
        <span class="plan-mode">${escapeHtml(novelModeLabel)}</span>
      </div>
      ${planActionsHtml(plan)}
    </header>
    <ol class="plan-steps">${steps}</ol>
    ${progressBarHtml(plan)}
  </section>`;
}

export function emptyPlanHtml() {
  return '<div class="v2-empty" data-v2-empty="plan">暂无活动 V2 计划</div>';
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
  const unsubscribe = onPlanChange((plan) => render(plan));
  return () => {
    const plan = getCurrentPlan();
    if (plan?.id) {
      stopPlanPolling(plan.id);
    }
    unsubscribe();
  };
}

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
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
    showError: options.showError || defaultShowError,
    stop() {
      stopPlanPolling(key);
    },
    async tick() {
      if (pollState.stopped) {
        return null;
      }
      try {
        const payload = await fetchPlan(key);
        if (pollState.stopped) {
          return null;
        }
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
    void pollState.tick().catch((error) => {
      pollState.showError(error?.message || "计划轮询失败");
    });
  }
  return pollState;
}

export function stopPlanPolling(planId) {
  const key = String(planId);
  deactivatePlanPoll(key, planPolls.get(key));
}
