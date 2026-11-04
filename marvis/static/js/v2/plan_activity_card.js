// Activity list (2026-10-14 redesign, D11): while a plan runs, the conversation
// shows what the Agent is doing in the current phase — each step's state, the
// platform-recorded duration of finished steps, and the running step's elapsed
// time and progress. Everything comes from the plan payload
// (status / last_run_duration_ms / started_at / progress); nothing is estimated.
import { escapeHtml } from "../ui-utils.js";
import { resolvePlanStepStatus, statusPresentation } from "../task-status.js";
import { renderModelTuningProgress } from "./model_tuning_progress.js";

const ACTIVE_STATUSES = new Set(["running", "checking"]);

export function formatStepDuration(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value < 0) return "";
  if (value < 1000) return "不到 1 秒";
  const seconds = value / 1000;
  if (seconds < 10) return `${seconds.toFixed(1)} 秒`;
  if (seconds < 60) return `${Math.round(seconds)} 秒`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return rest ? `${minutes} 分 ${rest} 秒` : `${minutes} 分钟`;
}

// Row state and words come from the shared status vocabulary (D16, batch 4),
// so a blocked or halted step reads the same here as in the rail and title.
const ACTIVITY_STATES = {
  succeeded: "done",
  skipped: "skipped",
  running: "running",
  failed: "failed",
  review: "review",
  stopped: "stopped",
  pending: "pending",
};

function stepState(step) {
  return statusPresentation(resolvePlanStepStatus(step));
}

function stepDetail(step, status, now) {
  if (status.key === "done") return formatStepDuration(step?.last_run_duration_ms);
  if (status.key === "running") {
    const started = Date.parse(String(step?.started_at || ""));
    if (Number.isFinite(started) && Number.isFinite(now) && now >= started) {
      return `${status.label} · 已用 ${formatStepDuration(now - started)}`;
    }
    return status.label;
  }
  return status.checker === "pending" ? "" : status.label;
}

export function planActivityCardHtml(plan, { now = Date.now() } = {}) {
  const steps = Array.isArray(plan?.steps) ? plan.steps : [];
  const running = steps.find((step) => ACTIVE_STATUSES.has(String(step?.status || "")));
  if (!running) return "";
  const phase = String(running.phase || "");
  const phaseSteps = phase ? steps.filter((step) => String(step?.phase || "") === phase) : [running];
  const rows = phaseSteps.map((step) => {
    const status = stepState(step);
    const state = ACTIVITY_STATES[status.checker] || "pending";
    const detail = stepDetail(step, status, now);
    const progress = state === "running" && step?.progress
      ? renderModelTuningProgress(step.progress, { compact: true })
      : "";
    return [
      `<li class="plan-activity-row" data-activity-state="${state}">`,
      '<span class="plan-activity-mark" aria-hidden="true"></span>',
      '<div class="plan-activity-copy">',
      `<span class="plan-activity-title">${escapeHtml(String(step?.title || "步骤"))}</span>`,
      detail ? `<span class="plan-activity-detail">${escapeHtml(detail)}</span>` : "",
      progress,
      "</div>",
      "</li>",
    ].join("");
  }).join("");
  const heading = phase ? `正在执行「${phase}」` : "正在执行";
  return [
    '<section class="plan-driver-action-card plan-activity-card" data-driver-action="activity" aria-live="polite">',
    `<header class="plan-activity-head">${escapeHtml(heading)}</header>`,
    `<ol class="plan-activity-list">${rows}</ol>`,
    "</section>",
  ].join("");
}
