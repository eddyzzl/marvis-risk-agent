import { api } from "../api.js";
import { escapeHtml } from "../ui-utils.js";
import { skeletonRowsHtml } from "../skeleton.js";
import { listPluginTools, listStrategyArtifacts, listTaskArtifacts } from "./api_v2.js";
import { gateConfirmLabel } from "./driver_gate_confirm.js";
import {
  driverGateHasWidget,
  gateMessageForCurrentTool,
} from "./driver_manual_analysis.js";
import { renderModelTuningProgress } from "./model_tuning_progress.js";

// Wired driver task types drive the plan rail / analysis flow.
export const PLAN_RAIL_TASK_TYPES = new Set(["data_join", "feature_analysis", "modeling", "strategy", "vintage"]);
const PLAN_RETRY_REFRESH_MAX_ATTEMPTS = 300;
const PLAN_RETRY_REFRESH_INTERVAL_MS = 1000;

export function taskUsesPlanRail(task) {
  return PLAN_RAIL_TASK_TYPES.has(task?.task_type);
}

export function workflowStatusSnapshot(workflowStatus) {
  const status = String(workflowStatus || "");
  if (status === "failed") {
    return {
      label: "失败", message: "计划执行失败。", kind: "error", tone: "danger",
      detail: "请查看中间信息流中的诊断与恢复方案。",
    };
  }
  if (status === "awaiting_confirm") {
    return {
      label: "待确认", message: "等待你的确认。", kind: "info", tone: "",
      detail: "当前步骤：计划确认。",
    };
  }
  if (["running", "confirmed"].includes(status)) {
    return {
      label: "执行中", message: "计划执行进行中。", kind: "busy", tone: "run",
      detail: "当前步骤：执行计划。",
    };
  }
  if (status === "validated") {
    return { label: "待开始", message: "计划已生成。", kind: "info", tone: "", detail: "当前步骤：等待开始执行。" };
  }
  if (status === "done") {
    return { label: "已完成", message: "计划执行完成。", kind: "success", tone: "success", detail: "所有计划步骤均已完成。" };
  }
  if (status === "review") {
    return { label: "待复核", message: "计划执行完成，等待复核。", kind: "success", tone: "success", detail: "请在中间信息流中查看结果。" };
  }
  if (status === "cancelled") {
    return { label: "已停止", message: "计划已停止。", kind: "stopped", tone: "", detail: "可在中间信息流中重新发起。" };
  }
  if (status === "draft") {
    return { label: "规划中", message: "正在生成计划。", kind: "busy", tone: "run", detail: "当前步骤：生成执行计划。" };
  }
  return null;
}

export function planWorkflowStatus(plan) {
  if (!plan) return null;
  const steps = Array.isArray(plan.steps) ? plan.steps : [];
  const findStep = (status) => steps.find((step) => String(step?.status || "") === status);
  const status = String(plan.status || "");
  const failedStep = findStep("failed");
  // A failed step is the authoritative execution state even if the plan-level
  // status was moved to `awaiting_confirm` solely to ask whether it should be
  // retried.  The recovery question remains in the middle conversation, but it
  // must not turn the task header/sidebar back into a healthy confirmation
  // gate while the failed step is still unresolved.
  if (status === "failed" || failedStep) {
    return {
      ...workflowStatusSnapshot("failed"),
      detail: failedStep
        ? `失败步骤：${failedStep.title || "未命名步骤"}。请在中间信息流中查看诊断与恢复方案。`
        : "请查看中间信息流中的诊断与恢复方案。",
    };
  }
  if (status === "awaiting_confirm") {
    const step = findStep("awaiting_confirm");
    return {
      ...workflowStatusSnapshot(status),
      detail: `当前步骤：${step?.title || "计划确认"}。`,
    };
  }
  if (["running", "confirmed"].includes(status)) {
    const step = findStep("running") || findStep("checking") || findStep("pending");
    return {
      ...workflowStatusSnapshot(status),
      detail: `当前步骤：${step?.title || "执行计划"}。`,
    };
  }
  return workflowStatusSnapshot(status);
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
  "modeling.generate_model_reports": "生成各候选模型开发报告",
};

// Map a plan step's status to the validation stepper's status vocabulary so it
// reuses stepCheckerHtml() (the checkmark / ring / etc.) and the .step CSS.
export function planStepToCheckerStatus(status) {
  switch (status) {
    case "done":
      return "succeeded";
    case "skipped":
      return "skipped";
    case "running":
    case "checking":
      return "running";
    case "failed":
      return "failed";
    case "awaiting_confirm":
      return "review";
    case "blocked":
      return "stopped";
    default:
      return "pending";
  }
}

function planPhaseStatus(steps = []) {
  const statuses = steps.map((step) => planStepToCheckerStatus(step?.status || "pending"));
  if (statuses.includes("failed")) return "failed";
  if (statuses.includes("review")) return "review";
  if (statuses.includes("running")) return "running";
  if (statuses.length && statuses.every((status) => ["succeeded", "skipped"].includes(status))) return "succeeded";
  return "pending";
}

function planStepDisplayPhase(step) {
  // choose_modeling_spec executes before feature screening and supplies its
  // feature universe / target type. Older persisted plans label it 建模, which
  // split one chronological feature phase and made its completed check appear
  // below a later failed feature step. Treat it as feature preparation here;
  // new templates carry the corrected phase as well.
  if (String(step?.tool_ref?.tool || "") === "choose_modeling_spec") return "特征";
  return String(step?.phase || "步骤");
}

export function planRailPhaseRows(plan) {
  const steps = Array.isArray(plan?.steps) ? [...plan.steps] : [];
  const sorted = steps.sort((left, right) => {
      const byIndex = (Number(left?.index) || 0) - (Number(right?.index) || 0);
      if (byIndex) return byIndex;
      return String(left?.id || "").localeCompare(String(right?.id || ""));
    });
  const groups = [];
  for (const step of sorted) {
    const phase = planStepDisplayPhase(step);
    const current = groups[groups.length - 1];
    if (!current || current.phase !== phase) groups.push({ phase, steps: [] });
    groups[groups.length - 1].steps.push(step);
  }
  return groups.map((group, index) => ({
    ...group,
    number: index + 1,
    checkerStatus: planPhaseStatus(group.steps),
  }));
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

// LT-4: the failure_envelope's editable_input_schema (above) is inferred from
// the current step inputs' Python types (marvis/agent/gates/contracts.py
// _editable_input_schema) -- it never carries `required`, `enum`, or a real
// `title`, because those only exist on the tool's authored input_schema in
// its pack manifest.json. planRetryRealProperties()/planRetryRequiredKeys()
// read that real schema once maybeFetchToolSchema() (below) has resolved
// it, so rendered fields can upgrade (enum -> select, required -> marked)
// without a backend change -- reusing GET /api/plugins/{name}/tools (already
// exposes input_schema; see marvis/routers/plugins.py) instead of a new
// endpoint.
function planRetryRequiredKeys(realSchema) {
  const required = realSchema && typeof realSchema === "object" ? realSchema.required : null;
  return Array.isArray(required) ? new Set(required.map((key) => String(key))) : new Set();
}

function planRetryRealProperties(realSchema) {
  const properties = realSchema && typeof realSchema === "object" ? realSchema.properties : null;
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
  const types = Array.isArray(spec?.type) ? spec.type : [spec?.type];
  const type = types.find((item) => item && item !== "null") || types[0];
  return String(type || "string");
}

function planRetryFieldNullable(spec, defaultValue) {
  const types = Array.isArray(spec?.type) ? spec.type : [spec?.type];
  return defaultValue === null || types.includes("null");
}

function planRetrySchemaFieldsHtml(step, realSchema = null) {
  const properties = planRetrySchemaProperties(step);
  const realProperties = planRetryRealProperties(realSchema);
  const requiredKeys = planRetryRequiredKeys(realSchema);
  const fields = Object.entries(properties).map(([key, spec]) => {
    const fieldSpec = spec && typeof spec === "object" ? spec : {};
    // Merge in the real tool input_schema's property (enum/title/type) when
    // it has resolved -- the inferred failure_envelope schema never carries
    // those, only a value-derived `type` and `default` (see the LT-4 note
    // above planRetryRequiredKeys()).
    const realSpec = realProperties[key] && typeof realProperties[key] === "object" ? realProperties[key] : {};
    const mergedSpec = { ...fieldSpec, ...realSpec };
    const type = planRetryFieldType(mergedSpec);
    const defaultValue = Object.prototype.hasOwnProperty.call(fieldSpec, "default") ? fieldSpec.default : "";
    const encodedKey = escapeHtml(key);
    const required = requiredKeys.has(key);
    const label = escapeHtml(mergedSpec.title || key) + (required ? '<span class="plan-retry-required">*</span>' : "");
    const typeLabel = escapeHtml(type);
    const nullable = planRetryFieldNullable(mergedSpec, defaultValue);
    const baseAttrs = `data-plan-retry-input-key="${encodedKey}" data-plan-retry-input-type="${typeLabel}" data-plan-retry-input-nullable="${nullable ? "true" : "false"}"`;
    if (Array.isArray(mergedSpec.enum) && mergedSpec.enum.length) {
      const current = planRetryFieldValue(defaultValue);
      const options = mergedSpec.enum.map((item) => {
        const value = planRetryFieldValue(item);
        const selected = value === current ? " selected" : "";
        return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(value)}</option>`;
      }).join("");
      return `<label class="plan-retry-schema-field${required ? " required" : ""}"><span>${label}<em>${typeLabel}</em></span><select ${baseAttrs}>${options}</select></label>`;
    }
    if (type === "boolean") {
      const selected = Boolean(defaultValue);
      return `<label class="plan-retry-schema-field${required ? " required" : ""}"><span>${label}<em>${typeLabel}</em></span><select ${baseAttrs}><option value="true"${selected ? " selected" : ""}>true</option><option value="false"${selected ? "" : " selected"}>false</option></select></label>`;
    }
    const inputType = type === "number" || type === "integer" ? "number" : "text";
    return `<label class="plan-retry-schema-field${required ? " required" : ""}"><span>${label}<em>${typeLabel}</em></span><input ${baseAttrs} type="${inputType}" value="${escapeHtml(planRetryFieldValue(defaultValue))}"></label>`;
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

// LT-4: a smoke pass on the retry flow found the endpoint semantics are a
// full REPLACE of the step's inputs_json (marvis/repositories/plans.py
// retry_failed_step UPDATE ... SET inputs_json = ?), not a merge with the
// step's existing inputs -- any field left out of what gets submitted here
// is silently dropped for that step. The JSON editor pre-fills current
// values (planRetryInputsText) so a naive "just tweak one field" edit
// mostly survives, but a user who clears the textarea and retypes a partial
// object loses the rest. Spell that out inline so it isn't discovered via a
// failed rerun.
function planRetryReplaceWarningHtml() {
  return '<p class="plan-retry-warning">'
    + '此处提交将<strong>整体替换</strong>该步骤输入（非合并）——未填字段将丢失，请基于当前值修改。'
    + "</p>";
}

// The full retry form, rendered into the middle workspace panel (not the rail).
// Markup below the <form> is byte-identical to the previous rail form so the
// submit path (retryPlanStep / parsePlanRetryInputs, scoped by
// [data-plan-step-retry]) is unchanged — only the mount location and the outer
// card shell differ.
function planRetryCardHtml(step, realSchema = null) {
  const stepId = String(step?.id || "");
  const stepTitle = String(step?.title || "未命名步骤");
  return `<section class="plan-retry-card" data-plan-step-retry="${escapeHtml(stepId)}" data-plan-retry-card="${escapeHtml(stepId)}">
    <header class="plan-retry-card-head">
      <span class="plan-retry-card-pill">编辑参数后重试</span>
      <span class="plan-retry-card-title">${escapeHtml(stepTitle)}</span>
    </header>
    <div class="plan-retry-card-body">
      ${planRetryScopeHtml(step)}
      ${planRetrySchemaFieldsHtml(step, realSchema)}
      ${planRetryReplaceWarningHtml()}
      <label class="plan-retry-json-label">
        参数 JSON
        <textarea class="plan-retry-inputs" data-plan-retry-inputs="${escapeHtml(stepId)}" rows="5" spellcheck="false">${escapeHtml(planRetryInputsText(step))}</textarea>
      </label>
      <button type="button" class="button compact primary" data-plan-retry-step="${escapeHtml(stepId)}">使用这些参数重试</button>
    </div>
  </section>`;
}

function parsePlanRetryStructuredValue(field) {
  const type = String(field?.dataset?.planRetryInputType || "string");
  const raw = String(field?.value ?? "");
  const nullable = field?.dataset?.planRetryInputNullable === "true";
  if (nullable && raw.trim() === "") {
    return null;
  }
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

function parsePlanRetryJson(field) {
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

function parsePlanRetryInputs(form) {
  const field = form?.querySelector?.(".plan-retry-inputs");
  if (
    field
    && typeof field.defaultValue === "string"
    && String(field.value ?? "") !== field.defaultValue
  ) {
    return parsePlanRetryJson(field);
  }
  const structured = collectPlanRetryStructuredInputs(form);
  if (structured) return structured;
  return parsePlanRetryJson(field);
}

export function createPlanRailController({
  $,
  stepCheckerHtml,
  getSelectedTask,
  getSelectedTaskId,
  getTaskBusyAction,
  setDriverExecutionBusy,
  getAgentMessages,
  isAgentMode,
  renderWorkflowStepper,
  setActionStatus,
  refreshTasks,
  loadAgentMessages,
  renderAll,
  apiClient = api,
  // LT-4: defaults to the already-existing GET /api/plugins/{name}/tools
  // client (marvis/static/js/v2/api_v2.js) so the retry form can progressively
  // upgrade from the inferred failure_envelope schema to the tool's real
  // authored input_schema (required/enum/title) -- optional so tests can
  // stub it without a network dependency.
  listPluginToolsClient = listPluginTools,
  // UX-5: fills the agent composer so "发消息介入" on a no_progress event can
  // hand the user straight into typing a steering instruction. Optional so
  // callers that don't wire the composer (tests) don't need a stub.
  fillComposer,
  // Strategy artifacts stay off the high-frequency task payload.  They are
  // fetched only after the latest strategy plan reaches done and then cached
  // per task + plan revision.
  listStrategyArtifactsClient = listStrategyArtifacts,
  listTaskArtifactsClient = listTaskArtifacts,
} = {}) {
  const v2PlanCache = new Map();
  const v2PlanLastFetch = new Map();
  const v2PlanFetchErrors = new Map();
  // LT-4: real tool input_schema, keyed by "plugin:tool", fetched lazily the
  // first time a failed step renders its retry control. Cached for the life
  // of the controller (schemas are static per pack) and merged into the
  // schema-form fields once resolved. A failed/absent fetch simply leaves
  // the entry unset, and planRetrySchemaFieldsHtml() keeps rendering from
  // the inferred failure_envelope schema -- the defensive fallback the spec
  // asks for, with no behavior regression.
  const v2ToolSchemaCache = new Map();
  const v2ToolSchemaFetching = new Set();
  const strategyArtifactsCache = new Map();
  const strategyArtifactsFetching = new Set();
  const renderStepChecker = typeof stepCheckerHtml === "function" ? stepCheckerHtml : () => "";

  function selectedTaskId() {
    return String(getSelectedTaskId?.() || "");
  }

  function selectedTask() {
    return getSelectedTask?.() || null;
  }

  function planForRail(plan, task = selectedTask()) {
    if (!plan || !Array.isArray(plan.steps)) return plan;
    // A driver job also wraps ordinary Agent questions and diagnostic replies,
    // so neither active_job_kind="driver" nor the generic local "agent" busy
    // flag proves that a workflow step is executing. Only structured
    // execute/confirm/retry actions set the dedicated local hint. Authoritative
    // server step states ("running"/"checking") already render directly and do
    // not need inference here.
    const localDriverRunning = getTaskBusyAction?.() === "driver_execute";
    if (!localDriverRunning) return plan;
    const alreadyRunning = plan.steps.some((step) => ["running", "checking"].includes(String(step?.status || "")));
    if (alreadyRunning) return plan;
    const current = plan.steps.find((step) => String(step?.status || "") === "awaiting_confirm")
      || plan.steps.find((step) => String(step?.status || "") === "pending");
    if (!current) return plan;
    return {
      ...plan,
      steps: plan.steps.map((step) => (
        String(step?.id || "") === String(current?.id || "")
          ? { ...step, status: "running", running_inferred_from_explicit_action: true }
          : step
      )),
    };
  }

  function toolSchemaKey(ref) {
    const plugin = String(ref?.plugin || "");
    const tool = String(ref?.tool || "");
    if (!plugin || !tool) return "";
    return `${plugin}:${tool}`;
  }

  // LT-4: lazily fetches the failed step's tool's real input_schema (via the
  // already-existing plugin tools endpoint) the first time its retry control
  // renders, then forces a re-render so planRetrySchemaFieldsHtml() can pick
  // up required/enum/title. Mirrors maybeFetchPlan()'s fetch-then-force-
  // rerender shape below. Errors (network, tool not found in the plugin's
  // tool list) are swallowed -- the schema-form stays on the inferred
  // failure_envelope schema, which is always available.
  function maybeFetchToolSchema(ref) {
    const key = toolSchemaKey(ref);
    if (!key || v2ToolSchemaCache.has(key) || v2ToolSchemaFetching.has(key)) return;
    if (typeof listPluginToolsClient !== "function") return;
    v2ToolSchemaFetching.add(key);
    Promise.resolve(listPluginToolsClient(ref.plugin))
      .then((data) => {
        const tools = (data && data.tools) || [];
        const tool = tools.find((item) => String(item?.name || "") === String(ref.tool));
        if (tool && tool.input_schema && typeof tool.input_schema === "object") {
          v2ToolSchemaCache.set(key, tool.input_schema);
          renderWorkflowStepper?.({ force: true });
        }
      })
      .catch(() => {})
      .finally(() => {
        v2ToolSchemaFetching.delete(key);
      });
  }

  function toolSchemaFor(ref) {
    const key = toolSchemaKey(ref);
    return key ? v2ToolSchemaCache.get(key) || null : null;
  }

  // Child steps contain only their checker, number and copy. All interactive
  // controls and status tags remain in the middle stream.
  function planSubstepHtml(step, subNumber) {
    const checkerStatus = planStepToCheckerStatus(step?.status || "pending");
    const ref = step.tool_ref || {};
    const description = step.description || step.summary || PLAN_STEP_HINTS[`${ref.plugin}.${ref.tool}`] || "";
    const tuningProgress = ["running", "checking"].includes(String(step?.status || ""))
      ? renderModelTuningProgress(step?.progress, { compact: true })
      : "";
    return [
      `<div class="notebook-step ${escapeHtml(checkerStatus)}" data-step-key="${escapeHtml(String(step.id || ""))}" data-plan-step-id="${escapeHtml(String(step.id || ""))}">`,
      renderStepChecker(checkerStatus),
      `<span class="notebook-step-no">${escapeHtml(subNumber)}</span>`,
      '<span class="plan-substep-copy">',
      `<strong>${escapeHtml(step.title || "未命名步骤")}</strong>`,
      description ? `<small>${escapeHtml(description)}</small>` : "",
      tuningProgress,
      "</span>",
      "</div>",
    ].join("");
  }

  function planSubstepGroupHtml(steps, parentNumber) {
    return [
      '<section class="notebook-step-group plan-rail-substeps">',
      `<h4>子任务 · ${steps.length}</h4>`,
      ...steps.map((step, index) => planSubstepHtml(step, `${parentNumber}.${index + 1}`)),
      "</section>",
    ].join("");
  }

  function planPhaseHtml({ phase, steps, number, checkerStatus }) {
    const titles = steps.map((step) => step?.title).filter(Boolean);
    const hint = titles.length <= 3 ? titles.join("、") : `${titles.slice(0, 3).join("、")}等 ${titles.length} 个子任务`;
    return [
      `<div class="step plan-rail-step ${escapeHtml(checkerStatus)}" data-plan-phase-key="${escapeHtml(`${phase}:${steps[0]?.id || number}`)}">`,
      '<div class="step-head">',
      renderStepChecker(checkerStatus),
      `<span class="step-number">${number}</span>`,
      '<span class="step-copy">',
      `<strong class="step-title">${escapeHtml(phase)}</strong>`,
      `<small class="step-hint">${escapeHtml(hint)}</small>`,
      "</span>",
      "</div>",
      planSubstepGroupHtml(steps, number),
      "</div>",
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

  // UX-10: mirrors the backend's latest_open_gate() predicate (turn_handlers.py) so
  // the plan rail can tell "the system is waiting on YOU" (a gate message with no
  // plan yet, e.g. the C1 role-assignment stage before confirm_join builds the plan)
  // apart from "the system is still generating" — the two were both rendered as
  // "计划生成中…" before, misattributing who the wait is on.
  function latestOpenGateStepName() {
    const messages = getAgentMessages?.() || [];
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const message = messages[i];
      if (message?.role !== "assistant") continue;
      const meta = message.metadata || {};
      if (meta.error || meta.join_skip) return null;
      const isGateShaped = meta.kind === "gate" || meta.kind === "plan_overview" || "join_c1" in meta;
      if (!isGateShaped) return null;
      if (meta.kind === "plan_overview") return "开始执行";
      if ("join_c1" in meta) return "文件角色与目标列";
      const step = planStep(meta);
      return step?.title || "当前步骤";
    }
    return null;
  }

  function maybeFetchPlan(taskId = selectedTaskId()) {
    if (!taskId) return Promise.resolve(null);
    // Note: we intentionally do NOT short-circuit on a terminal cached plan. Re-engaging
    // a finished driver task now builds a FRESH plan (see _active_plan in api.py), so the
    // rail must be able to pick that new plan up. Driver tasks aren't on a polling loop,
    // so this only fetches on render events (throttled below), not continuously.
    const now = Date.now();
    if (now - (v2PlanLastFetch.get(taskId) || 0) < 900) {
      return Promise.resolve(v2PlanCache.get(taskId) || null);
    }
    v2PlanLastFetch.set(taskId, now);
    return fetch(`/api/tasks/${encodeURIComponent(taskId)}/plans`)
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
        if (changed && selectedTaskId() === taskId) renderAll?.();
        return next;
      })
      .catch((error) => {
        v2PlanFetchErrors.set(taskId, error?.message || "network");
        if (selectedTaskId() === taskId) renderWorkflowStepper?.({ force: true });
        return null;
      });
  }

  async function retryFetch(taskId = selectedTaskId(), attempt = 0) {
    if (!taskId || selectedTaskId() !== taskId) return;
    v2PlanLastFetch.delete(taskId);
    v2PlanFetchErrors.delete(taskId);
    const [planResult] = await Promise.allSettled([
      maybeFetchPlan(taskId),
      refreshTasks?.(),
      loadAgentMessages?.(taskId, { preserveOptimistic: true }),
    ]);
    const nextPlan = planResult.status === "fulfilled" ? planResult.value : null;
    if (selectedTaskId() === taskId) renderAll?.();
    renderWorkflowStepper?.({ force: true });
    const status = String(nextPlan?.status || "");
    const stillRunning = !nextPlan || ["confirmed", "running"].includes(status);
    if (stillRunning && attempt < PLAN_RETRY_REFRESH_MAX_ATTEMPTS) {
      window.setTimeout(
        () => { void retryFetch(taskId, attempt + 1); },
        PLAN_RETRY_REFRESH_INTERVAL_MS,
      );
    }
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
    setDriverExecutionBusy?.(true, taskId);
    renderWorkflowStepper?.({ force: true });
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
        window.setTimeout(
          () => { void retryFetch(taskId); },
          PLAN_RETRY_REFRESH_INTERVAL_MS,
        );
      }
    } catch (error) {
      button.disabled = false;
      setActionStatus?.(error?.message || "重试步骤失败。", "error");
    } finally {
      setDriverExecutionBusy?.(false, taskId);
    }
  }

  // VD-3: three stand-in phase rows (checker + title-bar shimmer), matching the
  // shape of the real plan-rail phase rows below so the skeleton-to-content
  // swap doesn't jump in height.
  function planRailSkeletonHtml() {
    return [
      '<div class="plan-rail-skeleton" aria-hidden="true" data-skeleton="plan-rail">',
      skeletonRowsHtml({ rows: 3, height: 34 }),
      "</div>",
    ].join("");
  }

  function planRailHtml(plan, { blocked = false, fetchError = "", firstLoad = false } = {}) {
    if (!plan || !(plan.steps || []).length) {
      // A driver task can fail setup before any plan is built (e.g. modeling with no
      // train/test/oot split column). Don't claim a plan is "生成中" forever — point
      // the user at the conversation message that explains what to fix.
      if (fetchError) {
        return '<div class="plan-rail-empty plan-rail-error">'
          + '<strong>计划读取失败</strong>'
          + "</div>";
      }
      if (blocked) {
        return '<div class="plan-rail-empty">尚未生成计划。请按对话中的提示处理后重新发起。</div>';
      }
      // UX-10: the system is not "生成中" here — it is waiting on the user (e.g. the
      // C1 role-assignment gate runs before confirm_join has built a plan at all).
      // Distinguish that from a genuine still-generating wait so the two don't read
      // as the same two-way "who's waiting on whom" deadlock.
      const openGateStep = latestOpenGateStepName();
      if (openGateStep) {
        return `<div class="plan-rail-empty">等待确认：${escapeHtml(openGateStep)}</div>`;
      }
      // VD-3: the genuine first fetch (no cached response yet, successful or
      // not) shows a skeleton instead of blank-then-text, so a slow first
      // plan build doesn't read as a hang. Once a response has landed at
      // least once, fall back to the plain "计划生成中…" text for any later
      // still-empty state (this should be rare after the first response).
      return firstLoad ? planRailSkeletonHtml() : '<div class="plan-rail-empty">计划生成中…</div>';
    }
    return planRailPhaseRows(plan).map((row) => planPhaseHtml(row)).join("");
  }

  // Failed steps in plan order, so the middle retry panel lists them the same
  // way the rail shows them.
  function failedPlanSteps(plan) {
    const steps = Array.isArray(plan?.steps) ? plan.steps : [];
    return [...steps]
      .filter((step) => (step?.status || "pending") === "failed")
      .sort((left, right) => (Number(left.index) || 0) - (Number(right.index) || 0));
  }

  // Builds the middle-workspace retry panel body: one editable card per failed
  // step. Returns "" when there is nothing to retry (the caller then hides the
  // panel entirely). The tool schema for each failed step is fetched lazily via
  // the same maybeFetchToolSchema() path the rail uses, so enum/required upgrades
  // apply here too.
  function planRetryPanelHtml(plan) {
    const failed = failedPlanSteps(plan);
    if (!failed.length) return "";
    const cards = failed.map((step) => {
      const ref = step?.tool_ref || {};
      maybeFetchToolSchema(ref);
      return planRetryCardHtml(step, toolSchemaFor(ref));
    });
    return [
      '<header class="plan-retry-panel-head">',
      '<h3>编辑参数后重试</h3>',
      '<p class="plan-retry-panel-sub">修改失败步骤的输入后重新执行。此处提交将整体替换该步骤输入（非合并）。</p>',
      "</header>",
      `<div class="plan-retry-panel-body">${cards.join("")}</div>`,
    ].join("");
  }

  // The done report step whose output the middle 下载报告 button drives, if any.
  function doneReportStep(plan) {
    const steps = Array.isArray(plan?.steps) ? plan.steps : [];
    return steps.find((step) => {
      const ref = step?.tool_ref || {};
      const tool = ref.tool;
      return (
        tool === "generate_model_report"
        || tool === "generate_model_reports"
        || tool === "generate_feature_report"
        || tool === "generate_risk_analysis_report"
      )
        && (step?.status || "pending") === "done";
    }) || null;
  }

  function strategyArtifactPlanKey(plan) {
    if (!plan?.id || plan?.status !== "done") return "";
    return `${String(plan.id)}:${String(plan.revision || 1)}`;
  }

  function maybeFetchStrategyArtifacts(plan, taskId = selectedTaskId()) {
    if (
      selectedTask()?.task_type !== "strategy"
      || typeof listStrategyArtifactsClient !== "function"
      || typeof listTaskArtifactsClient !== "function"
    ) {
      return null;
    }
    const planKey = strategyArtifactPlanKey(plan);
    if (!taskId || !planKey) return null;
    const cached = strategyArtifactsCache.get(taskId);
    if (cached?.planKey === planKey || strategyArtifactsFetching.has(`${taskId}:${planKey}`)) {
      return cached || { planKey, state: "loading", artifacts: [] };
    }

    const fetchKey = `${taskId}:${planKey}`;
    const loading = { planKey, state: "loading", artifacts: [] };
    strategyArtifactsCache.set(taskId, loading);
    strategyArtifactsFetching.add(fetchKey);
    Promise.all([
      Promise.resolve(listStrategyArtifactsClient(taskId)),
      Promise.resolve(listTaskArtifactsClient(taskId)),
    ])
      .then(([strategyPayload, taskPayload]) => {
        if (strategyArtifactsCache.get(taskId)?.planKey !== planKey) return;
        const strategyArtifacts = Array.isArray(strategyPayload?.artifacts)
          ? strategyPayload.artifacts.map((item) => ({ ...item, artifact_scope: "strategy" }))
          : [];
        const taskArtifacts = Array.isArray(taskPayload?.artifacts)
          ? taskPayload.artifacts.map((item) => ({ ...item, artifact_scope: "task_analysis" }))
          : [];
        const seen = new Set(
          strategyArtifacts.map((item) => `${String(item?.kind || "")}\u0000${String(item?.filename || "")}`),
        );
        const newestTaskArtifactByKey = new Map();
        taskArtifacts.forEach((item, index) => {
          const key = `${String(item?.kind || "")}\u0000${String(item?.filename || "")}`;
          const createdAt = Date.parse(String(item?.created_at || ""));
          const candidate = {
            item,
            index,
            createdAt: Number.isFinite(createdAt) ? createdAt : Number.NEGATIVE_INFINITY,
          };
          const current = newestTaskArtifactByKey.get(key);
          if (
            !current
            || candidate.createdAt > current.createdAt
            || (
              candidate.createdAt === current.createdAt
              && candidate.index > current.index
            )
          ) {
            newestTaskArtifactByKey.set(key, candidate);
          }
        });
        const artifacts = [
          ...strategyArtifacts,
          ...taskArtifacts.filter((item, index) => {
            const key = `${String(item?.kind || "")}\u0000${String(item?.filename || "")}`;
            if (newestTaskArtifactByKey.get(key)?.index !== index) return false;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          }),
        ];
        strategyArtifactsCache.set(taskId, { planKey, state: "ready", artifacts });
      })
      .catch((error) => {
        if (strategyArtifactsCache.get(taskId)?.planKey !== planKey) return;
        strategyArtifactsCache.set(taskId, {
          planKey,
          state: "error",
          artifacts: [],
          error: String(error?.message || "策略产物读取失败"),
        });
      })
      .finally(() => {
        strategyArtifactsFetching.delete(fetchKey);
        if (selectedTaskId() === taskId) renderWorkflowStepper?.({ force: true });
      });
    return loading;
  }

  function strategyArtifactStatusLabel(status) {
    const labels = {
      adopted_local: "本地采纳",
      validated: "已验证",
      draft: "草稿",
      retired: "已退役",
    };
    return labels[String(status || "")] || "状态未标注";
  }

  const strategyDeliveryKindLabels = {
    strategy_delivery_python: "Python",
    strategy_delivery_sql: "DuckDB SQL",
    strategy_delivery_json: "Strategy JSON",
    strategy_delivery_equivalence_json: "Equivalence JSON",
  };
  const strategyReportFormatByKind = {
    strategy_report_bundle_json: "JSON",
    strategy_report_markdown: "Markdown",
    strategy_report_xlsx: "Excel",
    strategy_report_docx: "Word",
  };

  function strategyReportArtifactGroupHtml(artifacts) {
    const byFormat = new Map();
    for (const artifact of artifacts) {
      const format = strategyReportFormatByKind[String(artifact?.kind || "")];
      if (format) byFormat.set(format, artifact);
    }
    if (!byFormat.size) return "";
    const actions = ["JSON", "Markdown", "Excel", "Word"].map((format) => {
      const artifact = byFormat.get(format);
      const downloadUrl = String(artifact?.download_url || "");
      const available = Boolean(artifact?.available && downloadUrl);
      return available
        ? `<a class="button compact secondary strategy-artifact-download" href="${escapeHtml(downloadUrl)}" download>${escapeHtml(format)}</a>`
        : `<span class="strategy-artifact-unavailable" aria-label="${escapeHtml(format)} 文件不可用">${escapeHtml(format)} 不可用</span>`;
    }).join("");
    return [
      '<li class="strategy-artifact-row strategy-report-artifact-group">',
      '<span class="strategy-artifact-copy">',
      "<strong>策略报告</strong>",
      "<small>同一最新修订 · JSON · Markdown · Excel · Word</small>",
      "</span>",
      `<span class="strategy-report-artifact-actions">${actions}</span>`,
      "</li>",
    ].join("");
  }

  function strategyArtifactRowsHtml(artifacts) {
    if (!artifacts.length) {
      return '<p class="strategy-artifacts-empty">计划已完成，当前没有可下载的策略产物。</p>';
    }
    const reportGroup = strategyReportArtifactGroupHtml(artifacts);
    const rows = artifacts
      .filter((artifact) => !strategyReportFormatByKind[String(artifact?.kind || "")])
      .map((artifact) => {
      const filename = String(artifact?.filename || "策略产物");
      const kind = String(artifact?.kind || "artifact");
      const deliveryKind = strategyDeliveryKindLabels[kind] || "";
      const version = artifact?.version == null ? "" : `v${String(artifact.version)}`;
      const taskAnalysis = artifact?.artifact_scope === "task_analysis";
      let scope = version;
      let status = strategyArtifactStatusLabel(artifact?.asset_status);
      if (taskAnalysis) {
        scope = "任务分析";
        status = String(artifact?.origin_tool || "任务分析产物");
      }
      if (deliveryKind) {
        scope = "策略交付";
        status = "离线交付";
      }
      const downloadUrl = String(artifact?.download_url || "");
      const available = Boolean(artifact?.available && downloadUrl);
      const action = available
        ? `<a class="button compact secondary strategy-artifact-download" href="${escapeHtml(downloadUrl)}" download>下载</a>`
        : '<span class="strategy-artifact-unavailable" aria-label="文件不可用">不可用</span>';
      return [
        '<li class="strategy-artifact-row">',
        '<span class="strategy-artifact-copy">',
        `<strong>${escapeHtml(filename)}</strong>`,
        `<small>${escapeHtml([deliveryKind || kind, scope, status].filter(Boolean).join(" · "))}</small>`,
        "</span>",
        action,
        "</li>",
      ].join("");
      });
    return `<ul class="strategy-artifact-list">${reportGroup}${rows.join("")}</ul>`;
  }

  function strategyArtifactsCardHtml(state) {
    if (!state) return "";
    if (state.state === "loading") {
      return [
        '<section class="plan-driver-action-card" data-driver-action="strategy-artifacts">',
        '<header class="plan-driver-action-head">',
        '<span class="plan-driver-action-pill">策略产物</span>',
        '<span class="plan-driver-action-title">正在读取本地策略产物…</span>',
        "</header>",
        "</section>",
      ].join("");
    }
    if (state.state === "error") {
      return [
        '<section class="plan-driver-action-card" data-driver-action="strategy-artifacts">',
        '<header class="plan-driver-action-head">',
        '<span class="plan-driver-action-pill">策略产物</span>',
        `<span class="plan-driver-action-title">${escapeHtml(state.error || "策略产物读取失败")}</span>`,
        "</header>",
        '<button type="button" class="button compact secondary" data-strategy-artifacts-retry="1">重新读取</button>',
        "</section>",
      ].join("");
    }
    const artifacts = Array.isArray(state.artifacts) ? state.artifacts : [];
    const hasAdopted = artifacts.some((item) => item?.asset_status === "adopted_local");
    const title = hasAdopted
      ? `策略产物已在当前工作区本地采纳，共 ${artifacts.length} 个。`
      : `策略计划已完成，共 ${artifacts.length} 个产物。`;
    return [
      '<section class="plan-driver-action-card strategy-artifacts-card" data-driver-action="strategy-artifacts">',
      '<header class="plan-driver-action-head">',
      `<span class="plan-driver-action-pill">${hasAdopted ? "本地采纳" : "策略产物"}</span>`,
      `<span class="plan-driver-action-title">${escapeHtml(title)}</span>`,
      "</header>",
      hasAdopted
        ? '<p class="strategy-artifacts-notice">本地采纳仅代表当前 MARVIS 工作区已确认，不代表生产环境已上线。</p>'
        : '<p class="strategy-artifacts-notice">这些文件是当前任务的本地分析产物，不代表策略已采纳或生产环境已上线。</p>',
      strategyArtifactRowsHtml(artifacts),
      "</section>",
    ].join("");
  }

  // Report completion messages render the richer, report-specific download
  // card in the timeline. Keep the generic plan action only as a fallback for
  // older/incomplete message histories, otherwise the same artifact appears
  // twice in the middle workspace.
  function hasConversationReportDownload() {
    const messages = getAgentMessages?.();
    if (!Array.isArray(messages)) return false;
    return messages.some((message) => Boolean(
      String(message?.metadata?.report_download?.download_url || "").trim()
      || (Array.isArray(message?.metadata?.report_downloads)
        && message.metadata.report_downloads.some((report) => (
          String(report?.download_url || "").trim()
        ))),
    ));
  }

  function matchingAgentGateMessage(plan, gate) {
    const messages = getAgentMessages?.();
    if (!Array.isArray(messages) || !gate) return null;
    const planId = String(plan?.id || "");
    const stepId = String(gate?.id || "");
    for (let index = messages.length - 1; index >= 0; index--) {
      const message = messages[index];
      const meta = message?.metadata || {};
      if (message?.role !== "assistant" || meta.kind !== "gate") continue;
      if (String(meta.plan_id || "") !== planId) continue;
      if (String(meta.step_id || "") !== stepId) continue;
      return message;
    }
    return null;
  }

  function agentGateNeedsStructuredInput(message) {
    if (!message) return true;
    const currentMessage = gateMessageForCurrentTool(message);
    const meta = currentMessage?.metadata || {};
    if (String(meta.gate_source_tool || "") === "apply_monitoring_disposition") {
      // Monitoring always exposes a disposition schema, but only a
      // deterministic red verdict requires that structured choice. Green and
      // amber runs are explicitly acknowledgeable with a plain confirmation.
      // Missing legacy evidence fails closed.
      const monitoringContract = meta.monitoring_disposition || {};
      const overallLevel = String(monitoringContract.overall_level || "").toLowerCase();
      return !(
        ["green", "amber"].includes(overallLevel)
        && monitoringContract.requires_structured_input === false
      );
    }
    const properties = meta.editable_input_schema?.properties;
    return driverGateHasWidget(currentMessage) || Boolean(
      properties
      && typeof properties === "object"
      && Object.keys(properties).length,
    );
  }

  // Builds the middle-workspace driver-actions panel body: explicit human
  // authorization for the latest Agent gate, the 开始执行 control (plan built but
  // not started), and/or the 下载报告 control (a report step has completed).
  // They reuse the existing document-level handlers
  // (data-driver-confirm / data-driver-report-download) — only the mount moves
  // out of the narrow rail into the roomy middle region. Returns "" when there is
  // no driver action to surface (the caller then hides the panel).
  function planDriverActionsHtml(plan, strategyArtifactState = null) {
    const cards = [];
    const agentMode = Boolean(isAgentMode?.());
    const activeJobKind = String(selectedTask()?.active_job_kind || "").trim();
    const authorizationBusy = Boolean(activeJobKind);
    const authorizationBusyAttrs = authorizationBusy
      ? ' disabled aria-disabled="true" title="上一步正在收尾，完成后可继续授权"'
      : "";
    const agentGate = agentMode && Array.isArray(plan?.steps)
      ? plan.steps.find((step) => String(step?.status || "") === "awaiting_confirm")
      : null;
    const agentGateMessage = matchingAgentGateMessage(plan, agentGate);
    // Agent-mode structured gates already explain the exact inputs required in
    // the conversation (special-value decisions, adoption reason, screening,
    // etc.). Their widgets are evidence-only in Agent mode and the composer is
    // the single governed input channel. Never add a generic confirm button
    // that cannot supply those required values. Also wait for the matching gate
    // message before exposing a plain action, avoiding a transient unsafe
    // button while plan polling is ahead of message polling.
    const agentGateHasStructuredInput = agentGateNeedsStructuredInput(agentGateMessage);
    if (agentGate && !agentGateHasStructuredInput) {
      const planId = String(plan?.id || "");
      const stepId = String(agentGate?.id || "");
      const toolName = String(agentGate?.tool_ref?.tool || "");
      cards.push([
        '<section class="plan-driver-action-card" data-driver-action="agent-gate">',
        '<header class="plan-driver-action-head">',
        `<span class="plan-driver-action-pill">${authorizationBusy ? "正在收尾" : "需要人工授权"}</span>`,
        `<span class="plan-driver-action-title">${authorizationBusy ? "上一步仍在收尾，完成后即可授权" : "Agent 已理解你的意图；请复核并授权"}「${escapeHtml(agentGate?.title || "当前步骤")}」。</span>`,
        "</header>",
        `<button type="button" class="button compact primary plan-step-confirm driver-confirm" data-driver-confirm="1" data-expected-plan-id="${escapeHtml(planId)}" data-expected-step-id="${escapeHtml(stepId)}"${authorizationBusyAttrs}>${escapeHtml(gateConfirmLabel(toolName))}</button>`,
        "</section>",
      ].join(""));
    }
    const awaitingStart = plan?.status === "validated";
    if (awaitingStart) {
      cards.push([
        '<section class="plan-driver-action-card" data-driver-action="start">',
        '<header class="plan-driver-action-head">',
        `<span class="plan-driver-action-pill">${agentMode ? "需要人工授权" : "开始执行"}</span>`,
        `<span class="plan-driver-action-title">${agentMode ? "Agent 已生成执行计划；请复核后授权开始。" : "计划已生成，确认后开始逐步执行。"}</span>`,
        "</header>",
        `<button type="button" class="button compact primary plan-step-confirm driver-confirm" data-driver-confirm="1" data-expected-plan-id="${escapeHtml(String(plan?.id || ""))}"${authorizationBusyAttrs}>开始执行</button>`,
        "</section>",
      ].join(""));
    }
    const reportStep = doneReportStep(plan);
    if (reportStep && !hasConversationReportDownload()) {
      const reportTool = reportStep?.tool_ref?.tool;
      const reportLabel = reportTool === "generate_feature_report"
        ? "特征分析报告"
        : reportTool === "generate_risk_analysis_report"
          ? "风险分析报告"
          : "模型开发报告";
      cards.push([
        '<section class="plan-driver-action-card" data-driver-action="report-download">',
        '<header class="plan-driver-action-head">',
        '<span class="plan-driver-action-pill">报告已就绪</span>',
        `<span class="plan-driver-action-title">${reportLabel}已生成，可下载查看。</span>`,
        "</header>",
        '<button type="button" class="button compact secondary plan-step-download" data-driver-report-download="1">下载报告</button>',
        "</section>",
      ].join(""));
    }
    const strategyArtifactsCard = strategyArtifactsCardHtml(strategyArtifactState);
    if (strategyArtifactsCard) cards.push(strategyArtifactsCard);
    if (!cards.length) return "";
    return `<div class="plan-driver-actions-body">${cards.join("")}</div>`;
  }

  // A stable signature of the driver-actions panel state, so an unchanged panel
  // is not rebuilt on every poll tick (which would drop focus / restart flashes).
  function planDriverActionsSignature(plan, strategyArtifactState = null) {
    const report = doneReportStep(plan);
    const conversationReport = hasConversationReportDownload();
    const agentGate = isAgentMode?.() && Array.isArray(plan?.steps)
      ? plan.steps.find((step) => String(step?.status || "") === "awaiting_confirm")
      : null;
    const agentGateMessage = matchingAgentGateMessage(plan, agentGate);
    const agentGateControl = agentGate
      ? (
        agentGateMessage
          ? (agentGateNeedsStructuredInput(agentGateMessage) ? "structured" : "plain")
          : "pending-message"
      )
      : "";
    return JSON.stringify({
      plan_id: String(plan?.id || ""),
      start: plan?.status === "validated",
      authorization_busy: String(selectedTask()?.active_job_kind || ""),
      agent_gate: agentGate
        ? `${String(agentGate?.id || "")}:${String(agentGate?.tool_ref?.tool || "")}:${String(agentGateMessage?.id || "")}:${agentGateControl}`
        : "",
      report: report && !conversationReport ? String(report.id || report.output_ref || "1") : "",
      conversationReport,
      strategy_artifacts: strategyArtifactState,
    });
  }

  // Mounts the driver-actions panel into the middle workspace (#planDriverActions).
  // Shows it only when there is at least one driver action (授权 / 开始执行 / 下载报告);
  // otherwise clears and hides it so a healthy in-progress plan never leaves a
  // stale action card in the middle region.
  function renderDriverActionsPanel(plan) {
    const panel = $("planDriverActions");
    if (!panel) return;
    const strategyArtifactState = maybeFetchStrategyArtifacts(plan);
    const html = planDriverActionsHtml(plan, strategyArtifactState);
    if (!html) {
      if (panel.dataset.driverActionsSignature !== "") {
        panel.dataset.driverActionsSignature = "";
        panel.innerHTML = "";
      }
      panel.classList.add("hidden");
      panel.classList.remove("is-open");
      panel.setAttribute("aria-hidden", "true");
      return;
    }
    const signature = planDriverActionsSignature(plan, strategyArtifactState);
    if (panel.dataset.driverActionsSignature !== signature) {
      panel.dataset.driverActionsSignature = signature;
      panel.innerHTML = html;
    }
    panel.classList.remove("hidden");
    panel.setAttribute("aria-hidden", "false");
  }

  // Hides and empties the middle driver-actions panel. Called alongside
  // clearRetryPanel when leaving a plan-rail task.
  function clearDriverActionsPanel() {
    const panel = $("planDriverActions");
    if (!panel) return;
    panel.dataset.driverActionsSignature = "";
    panel.innerHTML = "";
    panel.classList.add("hidden");
    panel.classList.remove("is-open");
    panel.setAttribute("aria-hidden", "true");
  }

  // Mounts the retry panel into the middle workspace (#planRetryPanel). Shows it
  // only when there is at least one failed step to retry; otherwise clears and
  // hides it so it never occupies the middle region on a healthy plan. Cards are
  // only rebuilt when their content signature changes, so an open panel with an
  // in-progress edit is not wiped on every poll tick.
  function renderRetryPanel(plan) {
    const panel = $("planRetryPanel");
    if (!panel) return;
    if (isAgentMode?.()) {
      clearRetryPanel();
      return;
    }
    const html = planRetryPanelHtml(plan);
    if (!html) {
      if (panel.dataset.planRetrySignature !== "") {
        panel.dataset.planRetrySignature = "";
        panel.innerHTML = "";
      }
      panel.classList.add("hidden");
      panel.setAttribute("aria-hidden", "true");
      panel.classList.remove("is-open");
      return;
    }
    const failed = failedPlanSteps(plan);
    const signature = JSON.stringify(failed.map((step) => {
      const ref = step?.tool_ref || {};
      return { id: step?.id, inputs: planRetryInputsText(step), schema: toolSchemaFor(ref) };
    }));
    if (panel.dataset.planRetrySignature !== signature) {
      panel.dataset.planRetrySignature = signature;
      panel.innerHTML = html;
    }
    panel.classList.remove("hidden");
    panel.setAttribute("aria-hidden", "false");
  }

  // Hides and empties the middle retry panel. Called when leaving a plan-rail
  // task (e.g. switching to a validation task) so a leftover retry form never
  // lingers in the middle workspace of an unrelated task.
  function clearRetryPanel() {
    const panel = $("planRetryPanel");
    if (!panel) return;
    panel.dataset.planRetrySignature = "";
    panel.innerHTML = "";
    panel.classList.add("hidden");
    panel.classList.remove("is-open");
    panel.setAttribute("aria-hidden", "true");
  }

  // True only for a real DOM element that supports the operations the keyed
  // reconciler needs. The static tests pass a bare `{ innerHTML: '' }` mock;
  // those exercise the innerHTML fallback below (which still lets them assert on
  // the produced markup), while a live browser gets node-preserving patching.
  function supportsReconciliation(el) {
    return Boolean(
      el
      && typeof el.insertBefore === "function"
      && typeof el.querySelector === "function"
      && typeof el.appendChild === "function"
      && typeof document !== "undefined"
      && typeof document.createElement === "function"
      && el.children,
    );
  }

  // Builds a detached node from an HTML string via a throwaway container. Used
  // to mint fresh keyed nodes (phase cards, substeps, chrome blocks) that the
  // reconciler then splices into the live rail.
  function nodeFromHtml(html) {
    const holder = document.createElement("div");
    holder.innerHTML = html;
    return holder.firstElementChild;
  }

  function reconcileSubsteps(section, steps, parentNumber) {
    const heading = section.querySelector("h4");
    if (heading) heading.textContent = `子任务 · ${steps.length}`;
    const existing = new Map();
    section.querySelectorAll(":scope > .notebook-step").forEach((node) => {
      if (node.dataset.stepKey) existing.set(node.dataset.stepKey, node);
    });
    let cursor = heading || null;
    steps.forEach((step, index) => {
      const key = String(step?.id || `idx:${index}`);
      const html = planSubstepHtml(step, `${parentNumber}.${index + 1}`);
      let node = existing.get(key);
      if (node) {
        existing.delete(key);
        if (node.dataset.stepSignature !== html) {
          const fresh = nodeFromHtml(html);
          if (fresh) {
            node.className = fresh.className;
            node.innerHTML = fresh.innerHTML;
          }
          node.dataset.stepSignature = html;
        }
      } else {
        node = nodeFromHtml(html);
        if (!node) return;
        node.dataset.stepKey = key;
        node.dataset.stepSignature = html;
      }
      const desiredNext = cursor ? cursor.nextSibling : section.firstChild;
      if (node !== desiredNext) section.insertBefore(node, desiredNext);
      cursor = node;
    });
    for (const node of existing.values()) node.remove();
  }

  // Keyed reconciliation preserves both parent phase cards and child step nodes
  // across polling updates while keeping the rail free of controls and tags.
  function reconcilePlanRail(container, plan) {
    if (!supportsReconciliation(container)) return false;
    const existing = new Map();
    for (const node of Array.from(container.children)) {
      if (node.dataset && node.dataset.phaseKey) {
        existing.set(node.dataset.phaseKey, node);
      } else {
        node.remove();
      }
    }
    let cursor = null;
    for (const row of planRailPhaseRows(plan)) {
      const key = `${row.phase}:${row.steps[0]?.id || row.number}`;
      const html = planPhaseHtml(row);
      let node = existing.get(key);
      if (node) {
        existing.delete(key);
        node.className = `step plan-rail-step ${row.checkerStatus}`;
        const fresh = nodeFromHtml(html);
        const head = node.querySelector(":scope > .step-head");
        const freshHead = fresh?.querySelector(":scope > .step-head");
        if (head && freshHead && head.innerHTML !== freshHead.innerHTML) {
          head.innerHTML = freshHead.innerHTML;
        }
      } else {
        node = nodeFromHtml(html);
        if (!node) continue;
        node.dataset.phaseKey = key;
      }
      const section = node.querySelector(":scope > .plan-rail-substeps");
      if (section) reconcileSubsteps(section, row.steps, row.number);
      const desiredNext = cursor ? cursor.nextSibling : container.firstChild;
      if (node !== desiredNext) container.insertBefore(node, desiredNext);
      cursor = node;
    }
    for (const node of existing.values()) node.remove();
    return true;
  }

  function render({ force = false, renderSignatures = {} } = {}) {
    const task = selectedTask();
    if (!taskUsesPlanRail(task)) return false;
    const taskId = selectedTaskId();
    const progressRail = $("progressRail");
    const railTitle = document.querySelector("#progressRail .step-rail-head h3");
    progressRail?.setAttribute("aria-label", "计划步骤");
    if (railTitle) railTitle.textContent = "计划步骤";
    // VD-3: "no response has landed for this task yet" — captured before
    // maybeFetchPlan's async .then can populate v2PlanCache — is the genuine
    // first-load moment that gets the skeleton treatment below.
    const firstLoad = !v2PlanCache.has(taskId);
    maybeFetchPlan(taskId);
    const plan = v2PlanCache.get(taskId);
    const railPlan = planForRail(plan, task);
    const blocked = driverHasBlockingError();
    const fetchError = v2PlanFetchErrors.get(taskId) || "";
    const planSignature = JSON.stringify({
      task: taskId,
      activeJobKind: task?.active_job_kind || "",
      plan: railPlan,
      blocked,
      fetchError,
      firstLoad,
    });
    if (force || renderSignatures.workflowStepper !== planSignature) {
      renderSignatures.workflowStepper = planSignature;
      const planStepper = $("workflowStepper");
      if (planStepper) {
        // A populated plan gets node-preserving keyed reconciliation so hovering
        // a step card during the per-second poll does not rebuild the node under
        // the cursor (the flicker fix). Empty/error/skeleton states are single
        // transient blocks with no hover target, and the static test harness
        // passes a bare innerHTML mock — both fall back to a plain innerHTML set.
        const hasSteps = Boolean(railPlan && (railPlan.steps || []).length);
        const reconciled = hasSteps
          && !fetchError
          && reconcilePlanRail(planStepper, railPlan);
        if (!reconciled) {
          planStepper.innerHTML = planRailHtml(railPlan, { blocked, fetchError, firstLoad });
          // Leaving the keyed path (e.g. plan emptied out) invalidates any slot
          // bookkeeping so the next populated render rebuilds slots cleanly.
          if (planStepper.dataset) delete planStepper.dataset.railReconciled;
        } else if (planStepper.dataset) {
          planStepper.dataset.railReconciled = "1";
        }
      }
    }
    // The editable retry form(s) and the driver actions (开始执行 / 下载报告)
    // render into the middle workspace, not the rail.
    renderRetryPanel(plan);
    renderDriverActionsPanel(plan);
    return true;
  }

  function handleClick(event) {
    const strategyArtifactsRetry = event.target?.closest?.("[data-strategy-artifacts-retry]");
    if (strategyArtifactsRetry) {
      event.preventDefault();
      event.stopPropagation();
      const taskId = selectedTaskId();
      strategyArtifactsCache.delete(taskId);
      maybeFetchStrategyArtifacts(v2PlanCache.get(taskId), taskId);
      renderWorkflowStepper?.({ force: true });
      return true;
    }
    const planRetryButton = event.target?.closest?.("[data-plan-retry-step]");
    if (planRetryButton) {
      event.preventDefault();
      event.stopPropagation();
      void retryPlanStep(planRetryButton);
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

  function planId(taskId = selectedTaskId()) {
    return String(v2PlanCache.get(taskId)?.id || "");
  }

  // VD-2: the gate card's consequence line ("确认后将执行:<下一步>") reads the
  // step that depends on the gate step, so it can name what happens next
  // without the caller re-deriving plan topology.
  function nextStepAfter(metadata = {}, taskId = selectedTaskId()) {
    const gate = planStep(metadata, taskId);
    if (!gate) return null;
    const plan = v2PlanCache.get(taskId);
    const steps = Array.isArray(plan?.steps) ? plan.steps : [];
    const downstream = steps.filter((step) => (step?.depends_on || []).includes(gate.id));
    if (!downstream.length) return null;
    return downstream.reduce((earliest, step) => (
      earliest === null || (step?.index ?? Infinity) < (earliest?.index ?? Infinity) ? step : earliest
    ), null);
  }

  function statusSnapshot(taskId = selectedTaskId()) {
    return planWorkflowStatus(v2PlanCache.get(taskId));
  }

  return {
    clearDriverActionsPanel,
    clearRetryPanel,
    handleClick,
    maybeFetchPlan,
    nextStepAfter,
    planId,
    planStep,
    render,
    resetFetchThrottle,
    retryFetch,
    statusSnapshot,
  };
}
