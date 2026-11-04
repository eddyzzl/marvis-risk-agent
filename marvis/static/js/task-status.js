// One status vocabulary for every surface (2026-10-14 redesign, D16 / batch 4).
// The sidebar row, the title pill, the multi-model chip and the right rail all
// resolve a record here: a resolver reads the structured fields of a task,
// plan, plan step, batch or batch item and returns one canonical key, and
// statusPresentation(key) turns that key into the word, tone, mascot face,
// sidebar dot and rail checker. Free-text status messages are never parsed.
// Pure module: no DOM access, so node test harnesses can import it.

// tone: ok | fail | run | review | neutral (title pill, chip, capsule colour)
// face: idle | busy | wait | done | fail (mascot avatar expression)
// dot: run | wait | bad | none (sidebar row dot)
// checker: the rail step glyph vocabulary of step-checker.js
export const STATUS = Object.freeze({
  not_started: { label: "待开始", tone: "neutral", face: "idle", dot: "none", checker: "pending" },
  queued: { label: "排队中", tone: "neutral", face: "idle", dot: "none", checker: "pending" },
  planning: { label: "规划中", tone: "run", face: "busy", dot: "run", checker: "running" },
  running: { label: "执行中", tone: "run", face: "busy", dot: "run", checker: "running" },
  awaiting_confirm: { label: "待确认", tone: "review", face: "wait", dot: "wait", checker: "review" },
  awaiting_next: { label: "待继续", tone: "review", face: "wait", dot: "wait", checker: "review" },
  needs_review: { label: "需复核", tone: "review", face: "wait", dot: "wait", checker: "review" },
  done: { label: "已完成", tone: "ok", face: "done", dot: "none", checker: "succeeded" },
  partial_failure: { label: "部分失败", tone: "fail", face: "fail", dot: "bad", checker: "failed" },
  failed: { label: "失败", tone: "fail", face: "fail", dot: "bad", checker: "failed" },
  needs_reconciliation: { label: "待核对", tone: "fail", face: "fail", dot: "bad", checker: "failed" },
  paused: { label: "已暂停", tone: "neutral", face: "idle", dot: "none", checker: "stopped" },
  stopped: { label: "已停止", tone: "neutral", face: "idle", dot: "none", checker: "stopped" },
  cancelled: { label: "已取消", tone: "neutral", face: "idle", dot: "none", checker: "stopped" },
  skipped: { label: "已跳过", tone: "neutral", face: "done", dot: "none", checker: "skipped" },
  // Any code not listed above. The raw code is never rendered.
  unknown: { label: "状态未知", tone: "neutral", face: "idle", dot: "none", checker: "pending" },
});

export function statusPresentation(key) {
  const known = typeof key === "string" && Object.prototype.hasOwnProperty.call(STATUS, key);
  const resolved = known ? key : "unknown";
  return { key: resolved, ...STATUS[resolved] };
}

function lower(value) {
  return value === null || value === undefined ? "" : String(value).trim().toLowerCase();
}

function planSteps(plan) {
  return Array.isArray(plan?.steps) ? plan.steps : [];
}

function findStep(steps, status) {
  return steps.find((step) => String(step?.status || "") === status);
}

// ---------- Plan family ----------

const PLAN_RECONCILIATION_MESSAGE = "执行结果尚未核对，已暂停重试。请先核对执行记录与实际产物，完成核对后再继续。";

export function planStepNeedsReconciliation(step) {
  if ((step?.execution_halt === "paused" && step?.failure_envelope?.error_kind === "search_paused")
    || (step?.execution_halt === "cancelled" && step?.failure_envelope?.error_kind === "search_cancelled")) return false;
  return step?.status === "failed" && step?.failure_envelope?.retryable === false;
}

export function planNeedsReconciliation(plan) {
  return plan?.status === "failed" && plan?.failure_envelope?.retryable === false;
}

const PLAN_STEP_KEYS = {
  pending: "not_started",
  blocked: "stopped",
  awaiting_confirm: "awaiting_confirm",
  running: "running",
  checking: "running",
  done: "done",
  skipped: "skipped",
};

export function resolvePlanStepStatus(step) {
  const status = String(step?.status || "");
  if (status === "failed") {
    if (planStepNeedsReconciliation(step)) return "needs_reconciliation";
    // A native search halt keeps a resumable checkpoint; it is not a failure.
    if (step?.execution_halt === "paused") return "paused";
    if (step?.execution_halt === "cancelled") return "cancelled";
    return "failed";
  }
  if (!status) return "not_started";
  return PLAN_STEP_KEYS[status] || "unknown";
}

// task.workflow_status carries the latest plan status when no plan is cached.
const WORKFLOW_KEYS = {
  draft: "planning",
  validated: "not_started",
  confirmed: "running",
  running: "running",
  awaiting_confirm: "awaiting_confirm",
  review: "needs_review",
  done: "done",
  failed: "failed",
  cancelled: "stopped",
};

export function resolveWorkflowStatus(workflowStatus) {
  return WORKFLOW_KEYS[String(workflowStatus || "")] || null;
}

// Precedence (shared with the task title): reconciliation > plan stopped >
// native halt > failed > awaiting confirmation (plan or any step) >
// continuation > running > the plan's own status. A cancelled plan that still
// needs reconciliation stays 待核对 so the reconciliation card keeps the lead.
export function resolvePlanStatus(plan) {
  if (!plan) return null;
  const steps = planSteps(plan);
  const status = String(plan.status || "");
  if (planNeedsReconciliation(plan) || steps.some(planStepNeedsReconciliation)) return "needs_reconciliation";
  if (status === "cancelled") return "stopped";
  const failedStep = findStep(steps, "failed");
  if (status === "failed" || failedStep) {
    // A failed step is authoritative even when the plan moved to
    // awaiting_confirm only to ask whether it should be retried.
    if (failedStep?.execution_halt === "paused") return "paused";
    if (failedStep?.execution_halt === "cancelled") return "cancelled";
    return "failed";
  }
  if (status === "awaiting_confirm" || findStep(steps, "awaiting_confirm")) return "awaiting_confirm";
  if (status === "running" && plan?.reconciliation?.continuation) return "awaiting_next";
  return resolveWorkflowStatus(status);
}

// Roll several step keys up into one (a rail phase), with the same precedence.
const ROLLUP_ORDER = [
  "needs_reconciliation",
  "paused",
  "cancelled",
  "failed",
  "awaiting_confirm",
  "awaiting_next",
  "needs_review",
  "running",
  "planning",
];

export function rollupStatus(keys = []) {
  const list = keys.filter(Boolean);
  if (!list.length) return "not_started";
  const worst = ROLLUP_ORDER.find((key) => list.includes(key));
  if (worst) return worst;
  if (list.every((key) => key === "done" || key === "skipped")) return "done";
  if (list.includes("stopped")) return "stopped";
  return "not_started";
}

const WORKFLOW_COPY = {
  failed: { message: "计划执行失败。", kind: "error", detail: "请查看中间信息流中的诊断与恢复方案。" },
  awaiting_confirm: { message: "等待你的确认。", kind: "info", detail: "当前步骤：计划确认。" },
  running: { message: "计划执行进行中。", kind: "busy", detail: "当前步骤：执行计划。" },
  not_started: { message: "计划已生成。", kind: "info", detail: "当前步骤：等待开始执行。" },
  done: { message: "计划执行完成。", kind: "success", detail: "所有计划步骤均已完成。" },
  needs_review: { message: "计划执行完成，等待复核。", kind: "review", detail: "请在中间信息流中查看结果。" },
  stopped: { message: "计划已停止。", kind: "stopped", detail: "可在中间信息流中重新发起。" },
  planning: { message: "正在生成计划。", kind: "busy", detail: "当前步骤：生成执行计划。" },
};

function statusSnapshot(key, copy = WORKFLOW_COPY[key] || {}) {
  return { ...statusPresentation(key), message: "", kind: "info", detail: "", ...copy };
}

// Title/sidebar snapshot of a bare workflow status: {key, label, tone, face,
// dot, checker, message, kind, detail}, or null for a status it does not know.
export function workflowStatusSnapshot(workflowStatus) {
  const key = resolveWorkflowStatus(workflowStatus);
  return key ? statusSnapshot(key) : null;
}

export function planWorkflowStatus(plan) {
  const key = resolvePlanStatus(plan);
  if (!key) return null;
  const steps = planSteps(plan);
  if (key === "needs_reconciliation") {
    const reconciliationStep = steps.find(planStepNeedsReconciliation);
    return statusSnapshot(key, {
      message: PLAN_RECONCILIATION_MESSAGE,
      kind: "error",
      detail: planNeedsReconciliation(plan)
        ? `计划完成结果待核对。${PLAN_RECONCILIATION_MESSAGE}`
        : `待核对步骤：${reconciliationStep?.title || "未命名步骤"}。${PLAN_RECONCILIATION_MESSAGE}`,
    });
  }
  if (key === "paused") {
    return statusSnapshot(key, {
      message: "搜索已暂停，检查点已保存。", kind: "stopped",
      detail: "在策略工作台核对检查点后继续搜索。",
    });
  }
  if (key === "cancelled") {
    return statusSnapshot(key, {
      message: "搜索已取消。", kind: "stopped",
      detail: "在策略工作台明确发起新的搜索。",
    });
  }
  if (key === "failed") {
    const failedStep = findStep(steps, "failed");
    return statusSnapshot(key, {
      ...WORKFLOW_COPY.failed,
      detail: failedStep
        ? `失败步骤：${failedStep.title || "未命名步骤"}。请在中间信息流中查看诊断与恢复方案。`
        : WORKFLOW_COPY.failed.detail,
    });
  }
  if (key === "awaiting_confirm") {
    const step = findStep(steps, "awaiting_confirm");
    return statusSnapshot(key, { ...WORKFLOW_COPY.awaiting_confirm, detail: `当前步骤：${step?.title || "计划确认"}。` });
  }
  if (key === "awaiting_next") {
    return statusSnapshot(key, {
      message: "已核对原执行结果，剩余步骤等待继续。", kind: "info",
      detail: "继续后按原计划执行剩余步骤，审批要求保持有效。",
    });
  }
  if (key === "running") {
    const step = findStep(steps, "running") || findStep(steps, "checking") || findStep(steps, "pending");
    return statusSnapshot(key, { ...WORKFLOW_COPY.running, detail: `当前步骤：${step?.title || "执行计划"}。` });
  }
  return statusSnapshot(key);
}

// ---------- Validation batch ----------

export function resolveBatchItemStatus(item, { batchStatus = "" } = {}) {
  const status = lower(item?.status);
  switch (status) {
    case "cancelled": return "cancelled";
    case "failed": return "failed";
    case "awaiting_confirmation":
    case "pending_confirmation":
    case "pending": return "awaiting_confirm";
    case "running": return "running";
    case "review_required": return "needs_review";
    case "succeeded":
    case "completed": return "done";
    case "queued": return "queued";
    case "created":
    case "batch_created": return lower(batchStatus) === "running" ? "queued" : "not_started";
    default: break;
  }
  // Older payloads may carry only the verdict. It decides the state word; the
  // verdict itself (通过 / 需复核) belongs in result cards, not in the chip.
  const verdict = lower(item?.outcome) || status;
  if (verdict === "fail" || verdict === "failed") return "failed";
  if (verdict === "manual_review") return "needs_review";
  if (verdict === "pass") return "done";
  return "unknown";
}

const BATCH_KEYS = {
  created: "not_started",
  batch_created: "not_started",
  queued: "queued",
  running: "running",
  awaiting_confirmation: "awaiting_confirm",
  pending_confirmation: "awaiting_confirm",
  partial_failure: "partial_failure",
  failed: "failed",
  cancelled: "cancelled",
  completed: "done",
  succeeded: "done",
  review_required: "needs_review",
};

// Batch parent: worst first — failed / partial_failure > awaiting_confirm >
// needs_review > running > done (2026-10-15 batch-4 decision).
export function resolveBatchStatus(batch) {
  const status = lower(batch?.status);
  const own = BATCH_KEYS[status] || (status ? "unknown" : "not_started");
  if (own === "cancelled" || own === "failed" || own === "partial_failure") return own;
  const items = Array.isArray(batch?.items) ? batch.items : [];
  const keys = items.map((item) => resolveBatchItemStatus(item, { batchStatus: status }));
  const total = Math.max(items.length, Number(batch?.counts?.total) || 0);
  const failed = keys.filter((key) => key === "failed").length;
  if (failed) return failed === total ? "failed" : "partial_failure";
  if (own === "awaiting_confirm" || keys.includes("awaiting_confirm")) return "awaiting_confirm";
  if (own === "needs_review" || keys.includes("needs_review")) return "needs_review";
  if (own === "running" || keys.includes("running")) return "running";
  if (own === "done") return "done";
  if (keys.length && keys.length === total && keys.every((key) => key === "done" || key === "skipped")) return "done";
  return own;
}

// ---------- Task ----------

const V1_TASK_KEYS = {
  created: "not_started",
  scanned: "awaiting_next",
  configured: "awaiting_next",
  executed: "awaiting_next",
  running: "running",
  computing_metrics: "running",
  // writing_artifacts is dual-meaning: metrics finished and the user starts
  // the report (待继续), or the report job runs (执行中, split below).
  writing_artifacts: "awaiting_next",
  succeeded: "done",
  failed: "failed",
  review_required: "needs_review",
};

// A plan-rail task record can look terminal after a setup sub-stage; until a
// plan or workflow_status proves completion it is waiting on the user.
const PLAN_COMPLETION_UNPROVEN = new Set([
  "scanned",
  "configured",
  "executed",
  "writing_artifacts",
  "succeeded",
  "review_required",
]);

// Precedence: stopped > plan / workflow state > batch aggregation (or, for a
// batch child, its batch item, so the rail matches that model's chip) > active
// job or local busy action > planning > V1 task status.
export function resolveTaskStatus(task, {
  usesPlanWorkflow = false,
  planStatus = null,
  busy = false,
  batch = null,
  batchItem = null,
  batchStatus = "",
  openGate = false,
} = {}) {
  if (!task) return statusPresentation("unknown");
  if (task.stopped === true) return statusPresentation("stopped");
  const planKey = typeof planStatus === "string" ? planStatus : planStatus?.key;
  if (planKey) return statusPresentation(planKey);
  const active = Boolean(busy || task.active_job_kind);
  if (batch || batchItem) {
    const key = batch ? resolveBatchStatus(batch) : resolveBatchItemStatus(batchItem, { batchStatus });
    return statusPresentation(active && ["not_started", "queued", "unknown"].includes(key) ? "running" : key);
  }
  const status = String(task.status || "");
  if (usesPlanWorkflow) {
    // A running job is answering the latest gate, so it outranks the gate.
    if (active) return statusPresentation("planning");
    if (openGate || PLAN_COMPLETION_UNPROVEN.has(status)) return statusPresentation("awaiting_confirm");
    return statusPresentation("not_started");
  }
  if (active) return statusPresentation("running");
  if (status === "writing_artifacts" && task.active_job_kind === "report") return statusPresentation("running");
  const key = V1_TASK_KEYS[status] || "unknown";
  // The task list flags a validation input contract still waiting on the
  // field check; a batch item in that state reads 待确认, so the single task
  // does too instead of the generic 待继续.
  // Only a scanned task: one reset to CREATED needs a rescan first.
  if (task.awaiting_input_confirmation === true && key === "awaiting_next") {
    return statusPresentation("awaiting_confirm");
  }
  return statusPresentation(key);
}

// Title pill adapter. The record's resolved status supplies the word; the
// action kind only selects a transient overlay while an action is in flight,
// failed or was cancelled. Stage wording ("PMML打分测试失败。") stays in the
// message/detail line, never in the pill.
export function actionStatusPresentation(message, kind, {
  task = null,
  stopped = false,
  usesPlanWorkflow = false,
  workflowSnapshot = null,
  status = null,
} = {}) {
  if (!message) return null;
  if (stopped) return statusPresentation("stopped");
  // A structured plan state wins over action copy that may disagree with it.
  if (usesPlanWorkflow && workflowSnapshot?.key) return statusPresentation(workflowSnapshot.key);
  const resolved = () => status || resolveTaskStatus(task, { usesPlanWorkflow, planStatus: workflowSnapshot });
  // An overlay picks the family; when the record already resolves inside that
  // family (规划中 while busy, 部分失败 on error) its more precise word stays.
  const overlay = (key, keepWhen) => {
    if (!task) return statusPresentation(key);
    const current = resolved();
    return keepWhen(current) ? current : statusPresentation(key);
  };
  if (kind === "stopped") return overlay("stopped", (current) => ["stopped", "paused", "cancelled"].includes(current.key));
  if (kind === "busy") return overlay("running", (current) => current.tone === "run");
  if (kind === "error") {
    return overlay("failed", (current) => current.tone === "fail" || task.status === "review_required");
  }
  if (!task) return kind === "success" ? statusPresentation("done") : null;
  return resolved();
}
