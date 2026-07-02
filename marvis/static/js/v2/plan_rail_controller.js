import { api } from "../api.js";
import { escapeHtml } from "../ui-utils.js";
import { attachArtifactHandlers, renderArtifact } from "./artifact_view.js";

// Wired driver task types drive the plan rail / analysis flow.
export const PLAN_RAIL_TASK_TYPES = new Set(["data_join", "feature_analysis", "modeling", "strategy", "vintage"]);

export function taskUsesPlanRail(task) {
  return PLAN_RAIL_TASK_TYPES.has(task?.task_type);
}

// Short human subtitle per tool, mirroring the validation stepper's step hints.
const PLAN_STEP_HINTS = {
  "data_ops.propose_join": "诊断匹配键 / 命中率 / 膨胀",
  "data_ops.confirm_join": "确认拼接规格",
  "data_ops.execute_join": "左连接生成锚样本",
  "modeling.screen_features": "泄漏感知特征筛选",
  "modeling.select_features": "IV/相关性多变量精选",
  "modeling.tune_hyperparameters": "超参搜索调优",
  "modeling.train_model": "训练模型",
  "modeling.compare_experiments": "对比候选实验",
  "modeling.generate_model_report": "生成模型开发报告",
};

// Map a plan step's status to the validation stepper's status vocabulary so it
// reuses stepCheckerHtml() (the checkmark / ring / etc.) and the .step CSS.
function planStepToCheckerStatus(status) {
  switch (status) {
    case "done":
    case "skipped":
      return "succeeded";
    case "running":
    case "checking":
      return "running";
    case "failed":
      return "failed";
    case "awaiting_confirm":
      return "review";
    default:
      return "pending";
  }
}

function planPhaseStatus(steps = []) {
  const statuses = steps.map((step) => planStepToCheckerStatus(step.status || "pending"));
  if (statuses.includes("failed")) return "failed";
  if (statuses.includes("review")) return "review";
  if (statuses.includes("running")) return "running";
  if (statuses.length && statuses.every((status) => status === "succeeded")) return "succeeded";
  return "pending";
}

function planPhaseHint(phase, steps = []) {
  const titles = steps
    .map((step) => String(step?.title || "").trim())
    .filter(Boolean);
  if (!titles.length) return `${phase}任务`;
  if (titles.length <= 3) return titles.join("、");
  return `${titles.slice(0, 3).join("、")}等 ${titles.length} 个子任务`;
}

function planRetryInputsText(step) {
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
      try {
        return JSON.stringify(inputs, null, 2);
      } catch (_) {
        return "{}";
      }
    }
  }
  try {
    return JSON.stringify(step?.inputs || {}, null, 2);
  } catch (_) {
    return "{}";
  }
}

function planRetrySchemaProperties(step) {
  const schema = step?.failure_envelope?.editable_input_schema;
  const properties = schema && typeof schema === "object" ? schema.properties : null;
  return properties && typeof properties === "object" ? properties : {};
}

function planRetryFieldValue(value) {
  if (value && typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch (_) {
      return "";
    }
  }
  return value == null ? "" : String(value);
}

function planRetryFieldType(spec) {
  const type = Array.isArray(spec?.type) ? spec.type[0] : spec?.type;
  return String(type || "string");
}

function planRetrySchemaFieldsHtml(step) {
  const properties = planRetrySchemaProperties(step);
  const fields = Object.entries(properties).map(([key, spec]) => {
    const fieldSpec = spec && typeof spec === "object" ? spec : {};
    const type = planRetryFieldType(fieldSpec);
    const defaultValue = Object.prototype.hasOwnProperty.call(fieldSpec, "default") ? fieldSpec.default : "";
    const encodedKey = escapeHtml(key);
    const label = escapeHtml(fieldSpec.title || key);
    const typeLabel = escapeHtml(type);
    const baseAttrs = `data-plan-retry-input-key="${encodedKey}" data-plan-retry-input-type="${typeLabel}"`;
    if (Array.isArray(fieldSpec.enum) && fieldSpec.enum.length) {
      const current = planRetryFieldValue(defaultValue);
      const options = fieldSpec.enum.map((item) => {
        const value = planRetryFieldValue(item);
        const selected = value === current ? " selected" : "";
        return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(value)}</option>`;
      }).join("");
      return `<label class="plan-retry-schema-field"><span>${label}<em>${typeLabel}</em></span><select ${baseAttrs}>${options}</select></label>`;
    }
    if (type === "boolean") {
      const selected = Boolean(defaultValue);
      return `<label class="plan-retry-schema-field"><span>${label}<em>${typeLabel}</em></span><select ${baseAttrs}><option value="true"${selected ? " selected" : ""}>true</option><option value="false"${selected ? "" : " selected"}>false</option></select></label>`;
    }
    const inputType = type === "number" || type === "integer" ? "number" : "text";
    return `<label class="plan-retry-schema-field"><span>${label}<em>${typeLabel}</em></span><input ${baseAttrs} type="${inputType}" value="${escapeHtml(planRetryFieldValue(defaultValue))}"></label>`;
  });
  if (!fields.length) return "";
  return `<div class="plan-retry-schema-fields">${fields.join("")}</div>`;
}

function planRetryScopeHtml(step) {
  const envelope = step?.failure_envelope;
  const resetSteps = Array.isArray(envelope?.downstream_reset_steps)
    ? envelope.downstream_reset_steps.filter(Boolean)
    : [];
  if (!resetSteps.length) return "";
  return `<p class="plan-retry-scope">将重置 ${resetSteps.map((item) => `<code>${escapeHtml(item)}</code>`).join("、")}</p>`;
}

function planRetryControlHtml(step) {
  const stepId = String(step?.id || "");
  return `<details class="plan-step-retry" data-plan-step-retry="${escapeHtml(stepId)}">
    <summary>编辑参数后重试</summary>
    ${planRetryScopeHtml(step)}
    ${planRetrySchemaFieldsHtml(step)}
    <label>
      参数 JSON
      <textarea class="plan-retry-inputs" data-plan-retry-inputs="${escapeHtml(stepId)}" rows="5" spellcheck="false">${escapeHtml(planRetryInputsText(step))}</textarea>
    </label>
    <button type="button" class="button compact primary" data-plan-retry-step="${escapeHtml(stepId)}">使用这些参数重试</button>
  </details>`;
}

function planOutputButtonHtml(step) {
  const outputRef = String(step?.output_ref || "");
  if (!outputRef) return "";
  return `<button type="button" class="button compact secondary plan-step-output" data-artifact="${escapeHtml(outputRef)}">查看输出</button>`;
}

function parsePlanRetryStructuredValue(field) {
  const type = String(field?.dataset?.planRetryInputType || "string");
  const raw = String(field?.value ?? "");
  if (type === "boolean") {
    return raw === "true";
  }
  if (type === "integer") {
    const value = Number.parseInt(raw, 10);
    if (!Number.isFinite(value)) throw new Error("整数重试参数无效。");
    return value;
  }
  if (type === "number") {
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error("数值重试参数无效。");
    return value;
  }
  if (type === "array") {
    let value;
    try {
      value = JSON.parse(raw || "[]");
    } catch (_) {
      throw new Error("数组重试参数无效。");
    }
    if (!Array.isArray(value)) throw new Error("数组重试参数无效。");
    return value;
  }
  if (type === "object") {
    let value;
    try {
      value = JSON.parse(raw || "{}");
    } catch (_) {
      throw new Error("对象重试参数无效。");
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("对象重试参数无效。");
    }
    return value;
  }
  if (type === "null") {
    return null;
  }
  return raw;
}

function collectPlanRetryStructuredInputs(form) {
  const fields = Array.from(form?.querySelectorAll?.("[data-plan-retry-input-key]") || []);
  if (!fields.length) return null;
  const inputs = {};
  fields.forEach((field) => {
    const key = String(field?.dataset?.planRetryInputKey || "");
    if (!key) return;
    inputs[key] = parsePlanRetryStructuredValue(field);
  });
  return inputs;
}

function parsePlanRetryInputs(form) {
  const structured = collectPlanRetryStructuredInputs(form);
  if (structured) return structured;
  const field = form?.querySelector?.(".plan-retry-inputs");
  let value;
  try {
    value = JSON.parse(String(field?.value || "{}"));
  } catch (_) {
    throw new Error("重试参数必须是合法 JSON。");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("重试参数必须是 JSON 对象。");
  }
  return value;
}

export function createPlanRailController({
  $,
  stepCheckerHtml,
  getSelectedTask,
  getSelectedTaskId,
  getAgentMessages,
  isAgentMode,
  renderWorkflowStepper,
  setActionStatus,
  refreshTasks,
  loadAgentMessages,
  renderAll,
  apiClient = api,
  artifactRenderer = renderArtifact,
} = {}) {
  const v2PlanCache = new Map();
  const v2PlanLastFetch = new Map();
  const v2PlanFetchErrors = new Map();
  const renderStepChecker = typeof stepCheckerHtml === "function" ? stepCheckerHtml : () => "";
  let artifactHandlersInstalled = false;

  function selectedTaskId() {
    return String(getSelectedTaskId?.() || "");
  }

  function selectedTask() {
    return getSelectedTask?.() || null;
  }

  function setArtifactPanelVisible(visible) {
    const panel = $("artifactPanel");
    if (!panel) return null;
    panel.hidden = !visible;
    panel.classList.toggle("hidden", !visible);
    return panel;
  }

  function clearArtifactPanel() {
    const body = $("artifactPanelBody");
    if (body) body.innerHTML = "";
    setArtifactPanelVisible(false);
  }

  function artifactPreviewContainer() {
    setArtifactPanelVisible(true);
    return $("artifactPanelBody") || $("artifactPanel");
  }

  async function renderRightRailArtifact(container, artifactRef) {
    const target = artifactPreviewContainer() || container;
    if (target) {
      target.innerHTML = '<div class="artifact-loading">正在加载输出...</div>';
    }
    return artifactRenderer(target, artifactRef);
  }

  function handleArtifactPanelCloseClick(event) {
    const button = event.target?.closest?.("[data-artifact-panel-close]");
    if (!button) return;
    event.preventDefault();
    clearArtifactPanel();
  }

  function installArtifactHandlers(root = typeof document !== "undefined" ? document : null) {
    if (!root || artifactHandlersInstalled) return;
    artifactHandlersInstalled = true;
    root.addEventListener("click", handleArtifactPanelCloseClick);
    attachArtifactHandlers(root, artifactPreviewContainer, {
      renderArtifact: renderRightRailArtifact,
      showError: (message) => setActionStatus?.(message, "error"),
    });
  }

  function planSubstepGroupHtml(steps = [], parentNumber = "") {
    if (!steps.length) return "";
    return [
      '<section class="notebook-step-group plan-rail-substeps">',
      `<h4>子任务 · ${steps.length}</h4>`,
      ...steps.map((step, index) => {
        const status = step.status || "pending";
        const checkerStatus = planStepToCheckerStatus(status);
        const ref = step.tool_ref || {};
        const description = step.description || step.summary || PLAN_STEP_HINTS[`${ref.plugin}.${ref.tool}`] || "";
        const subNumber = parentNumber ? `${parentNumber}.${index + 1}` : `${index + 1}`;
        // Manual mode confirms each gate from the rail (the middle is analysis
        // only); agent mode shows a read-only "待确认" badge because the LLM
        // operates the gate. The button reuses the document-level
        // data-driver-confirm handler.
        const awaiting = status === "awaiting_confirm"
          ? (isAgentMode?.()
            ? '<span class="plan-step-await">待确认</span>'
            : '<button type="button" class="button compact primary plan-step-confirm driver-confirm" data-driver-confirm="1">确认</button>')
          : "";
        // Download sits inline on the producing report step (spec §9: like validation's
        // step-action-button), not floating at the rail bottom.
        const isReportDone = (ref.tool === "generate_model_report" || ref.tool === "generate_feature_report")
          && status === "done";
        const download = isReportDone
          ? '<button type="button" class="button compact secondary plan-step-download" data-driver-report-download="1">下载报告</button>'
          : "";
        const output = planOutputButtonHtml(step);
        const retry = status === "failed" ? planRetryControlHtml(step) : "";
        const descriptionHtml = description ? `<small>${escapeHtml(description)}</small>` : "";
        return [
          `<div class="notebook-step ${escapeHtml(checkerStatus)}">`,
          renderStepChecker(checkerStatus),
          `<span class="notebook-step-no">${escapeHtml(subNumber)}</span>`,
          '<span class="plan-substep-copy">',
          `<strong>${escapeHtml(step.title || "未命名步骤")}</strong>`,
          descriptionHtml,
          "</span>",
          awaiting,
          download,
          output,
          retry,
          "</div>",
        ].join("");
      }),
      "</section>",
    ].join("");
  }

  // True when the driver's latest assistant message is a blocking error (e.g. a
  // setup failure that prevented any plan from being built). Used to give the plan
  // rail an honest empty state instead of a perpetual "计划生成中…".
  function driverHasBlockingError() {
    const messages = getAgentMessages?.() || [];
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const message = messages[i];
      if (message?.role !== "assistant") continue;
      return Boolean((message.metadata || {}).error);
    }
    return false;
  }

  function maybeFetchPlan(taskId = selectedTaskId()) {
    if (!taskId) return;
    // Note: we intentionally do NOT short-circuit on a terminal cached plan. Re-engaging
    // a finished driver task now builds a FRESH plan (see _active_plan in api.py), so the
    // rail must be able to pick that new plan up. Driver tasks aren't on a polling loop,
    // so this only fetches on render events (throttled below), not continuously.
    const now = Date.now();
    if (now - (v2PlanLastFetch.get(taskId) || 0) < 900) return;
    v2PlanLastFetch.set(taskId, now);
    fetch(`/api/tasks/${encodeURIComponent(taskId)}/plans`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => {
        const plans = (data && data.plans) || [];
        const next = plans.length ? plans[plans.length - 1] : null;
        const hadError = v2PlanFetchErrors.delete(taskId);
        const changed = hadError || JSON.stringify(v2PlanCache.get(taskId)) !== JSON.stringify(next);
        v2PlanCache.set(taskId, next);
        if (changed && selectedTaskId() === taskId) renderWorkflowStepper?.({ force: true });
      })
      .catch((error) => {
        v2PlanFetchErrors.set(taskId, error?.message || "network");
        if (selectedTaskId() === taskId) renderWorkflowStepper?.({ force: true });
      });
  }

  function retryFetch(taskId = selectedTaskId()) {
    if (!taskId) return;
    v2PlanLastFetch.delete(taskId);
    v2PlanFetchErrors.delete(taskId);
    maybeFetchPlan(taskId);
    renderWorkflowStepper?.({ force: true });
  }

  function resetFetchThrottle(taskId = selectedTaskId()) {
    if (!taskId) return;
    v2PlanLastFetch.delete(taskId);
  }

  async function retryPlanStep(button) {
    const taskId = selectedTaskId();
    const plan = v2PlanCache.get(taskId);
    const stepId = button?.dataset?.planRetryStep || "";
    if (!taskId || !plan?.id || !stepId) {
      setActionStatus?.("缺少可重试的计划步骤，请刷新后重试。", "error");
      return;
    }
    let inputs;
    try {
      inputs = parsePlanRetryInputs(button.closest("[data-plan-step-retry]"));
    } catch (error) {
      setActionStatus?.(error?.message || "重试参数无效。", "error");
      return;
    }
    button.disabled = true;
    try {
      await apiClient(`/api/plans/${encodeURIComponent(plan.id)}/steps/${encodeURIComponent(stepId)}/retry`, {
        method: "POST",
        body: JSON.stringify({ inputs }),
      });
      setActionStatus?.("正在重试步骤...", "busy");
      v2PlanLastFetch.delete(taskId);
      v2PlanCache.delete(taskId);
      renderWorkflowStepper?.({ force: true });
      await refreshTasks?.();
      await loadAgentMessages?.(taskId, { preserveOptimistic: true });
      if (selectedTaskId() === taskId) {
        renderAll?.();
        maybeFetchPlan(taskId);
        window.setTimeout(() => retryFetch(taskId), 1000);
      }
    } catch (error) {
      button.disabled = false;
      setActionStatus?.(error?.message || "重试步骤失败。", "error");
    }
  }

  function planRailHtml(plan, { blocked = false, fetchError = "" } = {}) {
    const fetchErrorBanner = fetchError
      ? '<div class="plan-rail-fetch-error" role="status">'
        + '<span>计划读取失败，当前显示的是上次缓存的计划。</span>'
        + '<button type="button" class="button compact secondary" data-plan-rail-retry="1">重试</button>'
        + "</div>"
      : "";
    if (!plan || !(plan.steps || []).length) {
      // A driver task can fail setup before any plan is built (e.g. modeling with no
      // train/test/oot split column). Don't claim a plan is "生成中" forever — point
      // the user at the conversation message that explains what to fix.
      if (fetchError) {
        return '<div class="plan-rail-empty plan-rail-error">'
          + '<strong>计划读取失败</strong>'
          + '<button type="button" class="button compact secondary" data-plan-rail-retry="1">重试</button>'
          + "</div>";
      }
      return blocked
        ? '<div class="plan-rail-empty">尚未生成计划。请按对话中的提示处理后重新发起。</div>'
        : '<div class="plan-rail-empty">计划生成中…</div>';
    }
    const steps = [...plan.steps].sort(
      (left, right) => (Number(left.index) || 0) - (Number(right.index) || 0),
    );
    const order = [];
    const byPhase = new Map();
    for (const step of steps) {
      const phase = step.phase || "步骤";
      if (!byPhase.has(phase)) {
        byPhase.set(phase, []);
        order.push(phase);
      }
      byPhase.get(phase).push(step);
    }
    const phasesHtml = order
      .map((phase, phaseIndex) => {
        const phaseSteps = byPhase.get(phase) || [];
        const phaseNumber = phaseIndex + 1;
        const phaseStatus = planPhaseStatus(phaseSteps);
        return [
          `<div class="step plan-rail-step ${escapeHtml(phaseStatus)}" role="group" aria-label="${phaseNumber}. ${escapeHtml(phase)}">`,
          '<div class="step-head">',
          renderStepChecker(phaseStatus),
          `<span class="step-number">${phaseNumber}</span>`,
          '<span class="step-copy">',
          `<strong class="step-title">${escapeHtml(phase)}</strong>`,
          `<small class="step-hint">${escapeHtml(planPhaseHint(phase, phaseSteps))}</small>`,
          "</span>",
          "</div>",
          planSubstepGroupHtml(phaseSteps, phaseNumber),
          "</div>",
        ].join("");
      })
      .join("");
    // Plan-level overview gate: the plan is built but has not started (status
    // "validated"). In manual mode the user confirms 开始 from the rail; agent mode
    // auto-confirms (AUTO) or uses the composer (NORMAL), so no button.
    const awaitingStart = plan.status === "validated" && !isAgentMode?.();
    const startControl = awaitingStart
      ? '<div class="plan-rail-start"><button type="button" class="button compact primary plan-step-confirm driver-confirm" data-driver-confirm="1">开始执行</button></div>'
      : "";
    // The report download now lives inline on the producing step row (see
    // planSubstepGroupHtml), not as a floating button at the rail bottom.
    return fetchErrorBanner + phasesHtml + startControl;
  }

  function render({ force = false, renderSignatures = {} } = {}) {
    const task = selectedTask();
    if (!taskUsesPlanRail(task)) return false;
    const taskId = selectedTaskId();
    const progressRail = $("progressRail");
    const railTitle = document.querySelector("#progressRail .step-rail-head h3");
    progressRail?.setAttribute("aria-label", "计划步骤");
    if (railTitle) railTitle.textContent = "计划步骤";
    maybeFetchPlan(taskId);
    const plan = v2PlanCache.get(taskId);
    const blocked = driverHasBlockingError();
    const fetchError = v2PlanFetchErrors.get(taskId) || "";
    const planSignature = JSON.stringify({ task: taskId, plan, blocked, fetchError });
    if (force || renderSignatures.workflowStepper !== planSignature) {
      renderSignatures.workflowStepper = planSignature;
      const planStepper = $("workflowStepper");
      if (planStepper) planStepper.innerHTML = planRailHtml(plan, { blocked, fetchError });
    }
    return true;
  }

  function handleClick(event) {
    const planRetryButton = event.target?.closest?.("[data-plan-retry-step]");
    if (planRetryButton) {
      event.preventDefault();
      event.stopPropagation();
      void retryPlanStep(planRetryButton);
      return true;
    }
    const retryButton = event.target?.closest?.("[data-plan-rail-retry]");
    if (retryButton) {
      event.preventDefault();
      event.stopPropagation();
      retryFetch();
      return true;
    }
    return false;
  }

  function planStep(metadata = {}, taskId = selectedTaskId()) {
    const stepId = metadata.step_id ? String(metadata.step_id) : "";
    if (!stepId) return null;
    const plan = v2PlanCache.get(taskId);
    const steps = Array.isArray(plan?.steps) ? plan.steps : [];
    return steps.find((step) => String(step?.id || "") === stepId) || null;
  }

  return {
    artifactPreviewContainer,
    clearArtifactPanel,
    handleClick,
    installArtifactHandlers,
    maybeFetchPlan,
    planStep,
    render,
    resetFetchThrottle,
    retryFetch,
  };
}
