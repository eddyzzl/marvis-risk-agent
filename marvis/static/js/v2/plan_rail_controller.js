import { api } from "../api.js";
import { escapeHtml } from "../ui-utils.js";
import { skeletonRowsHtml, skeletonTableHtml } from "../skeleton.js";
import { listPluginTools } from "./api_v2.js";
import { attachArtifactHandlers, renderArtifact } from "./artifact_view.js";
import { gateConfirmLabel } from "./driver_gate_confirm.js";

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

// UX-5: human copy for loop_event reason codes (marvis/orchestrator/executor.py
// reason= call sites: failure/decision_point/final_review/user_instruction).
const LOOP_EVENT_REASON_LABELS = {
  failure: "步骤执行失败",
  decision_point: "决策点复核",
  final_review: "终审未通过",
  user_instruction: "用户指令",
};

function loopEventReasonText(event) {
  if (event?.type === "replan" && event?.reason === "user_instruction" && event?.instruction) {
    return String(event.instruction);
  }
  return LOOP_EVENT_REASON_LABELS[event?.reason] || String(event?.reason || "");
}

// UX-5: mirrors loop_progress.js's eventLabel() copy (重新规划：<reason> /
// 暂无进展：<reason>) so the same event reads the same way if that dead module
// is later deleted per UX-8 — but rendered inline in the live plan rail.
function loopEventLabel(event) {
  const reason = loopEventReasonText(event);
  if (event?.type === "replan") return `已重规划：${reason}`;
  if (event?.type === "no_progress") return `暂无进展：${reason}`;
  return `${event?.type || "事件"}：${reason}`;
}

// UX-5: last-3 loop_events strip at the top of the plan rail, so replan/
// no_progress branches (invisible before this) surface without digging into
// dev tools. no_progress gets the attention tone plus an intervene shortcut
// that hands the user straight to the composer instead of leaving them to
// guess why the plan looks stuck.
function loopEventStripHtml(plan) {
  const events = Array.isArray(plan?.loop_events) ? plan.loop_events : [];
  if (!events.length) return "";
  const recent = events.slice(-3);
  const rows = recent.map((event) => {
    const attention = event?.type === "no_progress" ? " attention" : "";
    const intervene = event?.type === "no_progress"
      ? '<button type="button" class="button compact secondary plan-rail-intervene" data-plan-rail-intervene="1">发消息介入</button>'
      : "";
    return `<div class="plan-rail-event${attention}">`
      + `<span>${escapeHtml(loopEventLabel(event))}</span>`
      + intervene
      + "</div>";
  });
  return `<div class="plan-rail-events" data-plan-rail-events="1">${rows.join("")}</div>`;
}

// UX-5: 已重规划 N 次 badge next to the plan title once at least one replan
// has happened, so a returning user can tell at a glance the plan already
// deviated from its original shape.
function replanBadgeHtml(plan) {
  const count = Number(plan?.replan_count) || 0;
  if (count <= 0) return "";
  return `<span class="plan-rail-replan-badge" title="该计划已重新规划 ${count} 次">已重规划 ${count} 次</span>`;
}

// UX-5: sub-agent activity rows, ported from subagent_view.js's status
// vocabulary (pending/running/done/failed/cancelled) into the live rail.
const SUB_AGENT_STATUS_LABELS = {
  spawned: "待执行",
  running: "运行中",
  returned: "已完成",
  failed: "失败",
  killed: "已终止",
};

function subAgentStatusLabel(status) {
  return SUB_AGENT_STATUS_LABELS[status] || String(status || "未知");
}

function subAgentRowsHtml(plan) {
  const subAgents = Array.isArray(plan?.sub_agents) ? plan.sub_agents : [];
  const active = subAgents.filter((sub) => sub?.status === "spawned" || sub?.status === "running");
  if (!active.length) return "";
  const rows = active.map((sub) => {
    const scope = escapeHtml(sub?.scope || "子任务");
    const status = escapeHtml(subAgentStatusLabel(sub?.status));
    return `<div class="plan-rail-subagent" data-plan-rail-subagent="${escapeHtml(String(sub?.id || ""))}">`
      + '<span class="plan-rail-subagent-badge">子任务运行中</span>'
      + `<span class="plan-rail-subagent-scope">${scope}</span>`
      + `<span class="plan-rail-subagent-status">${status}</span>`
      + "</div>";
  });
  return `<div class="plan-rail-subagents" data-plan-rail-subagents="1">${rows.join("")}</div>`;
}

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
  const type = Array.isArray(spec?.type) ? spec.type[0] : spec?.type;
  return String(type || "string");
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
    const baseAttrs = `data-plan-retry-input-key="${encodedKey}" data-plan-retry-input-type="${typeLabel}"`;
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

// The right rail only carries a lightweight entry now: the full "编辑参数后重试"
// form lives in the middle workspace (#planRetryPanel) so filling/selecting is
// done in the roomy middle region, not squeezed into the narrow rail. Clicking
// this opens the middle panel and scrolls its matching step card into view.
function planRetryRailEntryHtml(step) {
  const stepId = String(step?.id || "");
  return `<button type="button" class="button compact secondary plan-step-retry-open" data-plan-retry-open="${escapeHtml(stepId)}">编辑参数后重试</button>`;
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
  getAgentMessages,
  isAgentMode,
  renderWorkflowStepper,
  setActionStatus,
  refreshTasks,
  loadAgentMessages,
  renderAll,
  apiClient = api,
  artifactRenderer = renderArtifact,
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

  // VD-3: a gate's evidence table (JOIN diagnostics / feature metrics / model
  // compare) is exactly what the artifact panel loads here, so this is the
  // "门表格数据加载中" wait — swap the old plain-text placeholder for a table
  // skeleton so a slow fetch doesn't read as a stalled/hung panel.
  async function renderRightRailArtifact(container, artifactRef) {
    const target = artifactPreviewContainer() || container;
    if (target) {
      target.innerHTML = `<div class="artifact-loading" data-skeleton="artifact">${skeletonTableHtml({ rows: 4, columns: 4 })}</div>`;
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

  // Single substep `.notebook-step` block. Factored out of planSubstepGroupHtml
  // so the keyed reconciler can rebuild one substep's markup in place (keyed by
  // step id) without touching its sibling substep nodes — that keeps a hovered
  // substep card from being destroyed on a poll tick.
  function planSubstepHtml(step, subNumber) {
    const status = step.status || "pending";
    const checkerStatus = planStepToCheckerStatus(status);
    const ref = step.tool_ref || {};
    const description = step.description || step.summary || PLAN_STEP_HINTS[`${ref.plugin}.${ref.tool}`] || "";
    // Interactive gate confirm no longer lives in the rail. The middle
    // conversation area already renders the pending gate section (plain confirm
    // button or a structured widget); the rail keeps only a "待确认" status badge
    // plus a lightweight locate entry that scrolls to (and flashes) that middle
    // gate section — no confirm control is rendered here in either mode.
    const stepId = String(step?.id || "");
    const awaiting = status === "awaiting_confirm"
      ? (isAgentMode?.()
        ? '<span class="plan-step-await">待确认</span>'
        : '<span class="plan-step-await">待确认</span>'
          + `<button type="button" class="button compact plan-step-locate" data-plan-gate-locate="${escapeHtml(stepId)}" title="跳到中间的确认卡片">定位</button>`)
      : "";
    // Report download no longer sits inline on the rail step row: the actual
    // 下载报告 button lives in the middle driver-actions panel (renderDriverActionsPanel
    // below). The rail step row only marks that the report is ready plus a locate
    // entry that scrolls to (and flashes) the middle download card.
    const isReportDone = (ref.tool === "generate_model_report" || ref.tool === "generate_feature_report")
      && status === "done";
    const download = isReportDone
      ? '<span class="plan-step-ready">报告已就绪</span>'
        + '<button type="button" class="button compact plan-step-locate" data-plan-report-locate="1" title="跳到中间的下载卡片">定位</button>'
      : "";
    const output = planOutputButtonHtml(step);
    if (status === "failed") maybeFetchToolSchema(ref);
    // Rail keeps only a lightweight entry; the editable form itself renders
    // in the middle workspace (#planRetryPanel via renderRetryPanel below).
    const retry = status === "failed" ? planRetryRailEntryHtml(step) : "";
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
  }

  function planSubstepGroupHtml(steps = [], parentNumber = "") {
    if (!steps.length) return "";
    return [
      '<section class="notebook-step-group plan-rail-substeps">',
      `<h4>子任务 · ${steps.length}</h4>`,
      ...steps.map((step, index) => {
        const subNumber = parentNumber ? `${parentNumber}.${index + 1}` : `${index + 1}`;
        return planSubstepHtml(step, subNumber);
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

  // Ordered phase plan: groups sorted steps by their `phase`, preserving first-
  // seen phase order. Shared by planRailHtml (string build) and the keyed
  // reconciler so both agree on phase identity/order.
  function planPhasePlan(plan) {
    const steps = [...(plan?.steps || [])].sort(
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
    return order.map((phase, phaseIndex) => ({
      phase,
      phaseSteps: byPhase.get(phase) || [],
      phaseNumber: phaseIndex + 1,
    }));
  }

  // The `.step-head` block for a phase card (checker + number + title + hint).
  // Extracted so the reconciler can refresh a persisted phase node's head in
  // place without rebuilding the whole phase card (which holds the substeps).
  function planPhaseHeadHtml(phase, phaseSteps, phaseNumber) {
    const phaseStatus = planPhaseStatus(phaseSteps);
    return [
      '<div class="step-head">',
      renderStepChecker(phaseStatus),
      `<span class="step-number">${phaseNumber}</span>`,
      '<span class="step-copy">',
      `<strong class="step-title">${escapeHtml(phase)}</strong>`,
      `<small class="step-hint">${escapeHtml(planPhaseHint(phase, phaseSteps))}</small>`,
      "</span>",
      "</div>",
    ].join("");
  }

  function planRailHtml(plan, { blocked = false, fetchError = "", firstLoad = false } = {}) {
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
    const phases = planPhasePlan(plan);
    const phasesHtml = phases
      .map(({ phase, phaseSteps, phaseNumber }) => {
        const phaseStatus = planPhaseStatus(phaseSteps);
        return [
          `<div class="step plan-rail-step ${escapeHtml(phaseStatus)}" role="group" aria-label="${phaseNumber}. ${escapeHtml(phase)}">`,
          planPhaseHeadHtml(phase, phaseSteps, phaseNumber),
          planSubstepGroupHtml(phaseSteps, phaseNumber),
          "</div>",
        ].join("");
      })
      .join("");
    // Plan-level overview gate: the plan is built but has not started (status
    // "validated"). The interactive 开始执行 button now lives in the middle
    // driver-actions panel (renderDriverActionsPanel); the rail keeps only a
    // status line plus a locate entry that scrolls to (and flashes) it. Agent
    // mode auto-confirms (AUTO) or uses the composer (NORMAL), so no entry.
    const awaitingStart = plan.status === "validated" && !isAgentMode?.();
    const startControl = awaitingStart
      ? '<div class="plan-rail-start">'
        + '<span class="plan-rail-start-status">等待开始执行</span>'
        + '<button type="button" class="button compact plan-step-locate" data-plan-start-locate="1" title="跳到中间的开始执行卡片">定位</button>'
        + "</div>"
      : "";
    // The report download now lives inline on the producing step row (see
    // planSubstepGroupHtml), not as a floating button at the rail bottom.
    // UX-5: replan badge + last-3 loop_events + active sub-agent rows, kept
    // above the phase list and rendered only when there is something to show
    // (no chrome on the common, uneventful plan).
    const replanBadge = replanBadgeHtml(plan);
    const headerBadge = replanBadge ? `<div class="plan-rail-header">${replanBadge}</div>` : "";
    const eventStrip = loopEventStripHtml(plan);
    const subAgentRows = subAgentRowsHtml(plan);
    return fetchErrorBanner + headerBadge + eventStrip + subAgentRows + phasesHtml + startControl;
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

  // The done report step whose output the 下载报告 button drives, if any. Mirrors
  // planSubstepHtml's isReportDone predicate so the middle download card appears
  // exactly when the rail marks a report ready.
  function doneReportStep(plan) {
    const steps = Array.isArray(plan?.steps) ? plan.steps : [];
    return steps.find((step) => {
      const ref = step?.tool_ref || {};
      const tool = ref.tool;
      return (tool === "generate_model_report" || tool === "generate_feature_report")
        && (step?.status || "pending") === "done";
    }) || null;
  }

  // Builds the middle-workspace driver-actions panel body: the 开始执行 control
  // (plan built but not started, manual mode) and/or the 下载报告 control (a
  // report step has completed). Both reuse the existing document-level handlers
  // (data-driver-confirm / data-driver-report-download) — only the mount moves
  // out of the narrow rail into the roomy middle region. Returns "" when there is
  // no driver action to surface (the caller then hides the panel).
  function planDriverActionsHtml(plan) {
    const cards = [];
    const awaitingStart = plan?.status === "validated" && !isAgentMode?.();
    if (awaitingStart) {
      cards.push([
        '<section class="plan-driver-action-card" data-driver-action="start">',
        '<header class="plan-driver-action-head">',
        '<span class="plan-driver-action-pill">开始执行</span>',
        '<span class="plan-driver-action-title">计划已生成，确认后开始逐步执行。</span>',
        "</header>",
        '<button type="button" class="button compact primary plan-step-confirm driver-confirm" data-driver-confirm="1">开始执行</button>',
        "</section>",
      ].join(""));
    }
    if (doneReportStep(plan)) {
      cards.push([
        '<section class="plan-driver-action-card" data-driver-action="report-download">',
        '<header class="plan-driver-action-head">',
        '<span class="plan-driver-action-pill">报告已就绪</span>',
        '<span class="plan-driver-action-title">模型开发报告已生成，可下载查看。</span>',
        "</header>",
        '<button type="button" class="button compact secondary plan-step-download" data-driver-report-download="1">下载报告</button>',
        "</section>",
      ].join(""));
    }
    if (!cards.length) return "";
    return `<div class="plan-driver-actions-body">${cards.join("")}</div>`;
  }

  // A stable signature of the driver-actions panel state, so an unchanged panel
  // is not rebuilt on every poll tick (which would drop focus / restart flashes).
  function planDriverActionsSignature(plan) {
    const report = doneReportStep(plan);
    return JSON.stringify({
      start: plan?.status === "validated" && !isAgentMode?.(),
      report: report ? String(report.id || report.output_ref || "1") : "",
    });
  }

  // Mounts the driver-actions panel into the middle workspace (#planDriverActions).
  // Shows it only when there is at least one driver action (开始执行 / 下载报告);
  // otherwise clears and hides it so a healthy in-progress plan never leaves a
  // stale action card in the middle region.
  function renderDriverActionsPanel(plan) {
    const panel = $("planDriverActions");
    if (!panel) return;
    const html = planDriverActionsHtml(plan);
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
    const signature = planDriverActionsSignature(plan);
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

  // Reveals the middle driver-actions panel, scrolls the requested action card
  // into view, and flashes it — the locate-and-flash bridge from the rail's
  // lightweight 开始执行 / 下载报告 status entries (mirrors openRetryCard).
  function openDriverActionCard(action) {
    const panel = $("planDriverActions");
    if (!panel) return;
    panel.classList.remove("hidden");
    panel.classList.add("is-open");
    panel.setAttribute("aria-hidden", "false");
    const card = action
      ? panel.querySelector(`[data-driver-action="${cssEscape(action)}"]`)
      : null;
    flashLocatedCard(panel, card);
  }

  // Scrolls the middle conversation gate section into view and flashes it — the
  // locate-and-flash bridge from the rail's lightweight "待确认" gate entry. The
  // actual confirm control (plain button or a structured widget) already lives in
  // that middle gate section; this only brings the user's eye to it.
  function openGateCard(stepId) {
    const container = $("agentMessages");
    if (!container) return;
    const escapedStep = stepId ? cssEscape(String(stepId)) : "";
    const bySection = escapedStep
      ? container.querySelector(`[data-driver-gate-section="${escapedStep}"]`)
      : null;
    // Fall back to the single pending gate section (only ever one) when the
    // section carries no per-step id.
    const card = bySection || container.querySelector(".driver-analysis-section.is-gate-pending");
    flashLocatedCard(container, card);
  }

  // Shared scroll-into-view + restart-flash routine for the middle locate
  // entries. Scrolls the card (or the container as a fallback) into view, then
  // restarts the flash highlight on the card via a reflow, and focuses the first
  // actionable control so keyboard users land on it.
  function flashLocatedCard(container, card) {
    const target = card || container;
    if (target && typeof target.scrollIntoView === "function") {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (card) {
      card.classList.remove("is-flash");
      void card.offsetWidth;
      card.classList.add("is-flash");
      const focusable = card.querySelector("button, input, select, textarea");
      if (focusable && typeof focusable.focus === "function") {
        try { focusable.focus({ preventScroll: true }); } catch (_) { focusable.focus(); }
      }
    }
  }

  // Mounts the retry panel into the middle workspace (#planRetryPanel). Shows it
  // only when there is at least one failed step to retry; otherwise clears and
  // hides it so it never occupies the middle region on a healthy plan. Cards are
  // only rebuilt when their content signature changes, so an open panel with an
  // in-progress edit is not wiped on every poll tick.
  function renderRetryPanel(plan) {
    const panel = $("planRetryPanel");
    if (!panel) return;
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

  // Reveals the middle retry panel and scrolls the requested step's card into
  // view. Called when the user clicks the rail's lightweight "编辑参数后重试"
  // entry — the heavy form lives in the middle, so this is the bridge.
  function openRetryCard(stepId) {
    const panel = $("planRetryPanel");
    if (!panel) return;
    panel.classList.remove("hidden");
    panel.classList.add("is-open");
    panel.setAttribute("aria-hidden", "false");
    const card = panel.querySelector(`[data-plan-retry-card="${cssEscape(stepId)}"]`);
    const target = card || panel;
    if (typeof target.scrollIntoView === "function") {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (card) {
      card.classList.remove("is-flash");
      // Reflow so re-adding the class restarts the highlight animation.
      void card.offsetWidth;
      card.classList.add("is-flash");
      const focusable = card.querySelector("input, select, textarea");
      if (focusable && typeof focusable.focus === "function") {
        try { focusable.focus({ preventScroll: true }); } catch (_) { focusable.focus(); }
      }
    }
  }

  // Minimal CSS.escape fallback for attribute selectors (step ids are backend
  // slugs, but guard against special chars so querySelector never throws).
  function cssEscape(value) {
    const raw = String(value == null ? "" : value);
    if (typeof CSS !== "undefined" && typeof CSS.escape === "function") return CSS.escape(raw);
    return raw.replace(/["\\\]]/g, "\\$&");
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

  // Reconciles the substeps inside a phase card's `.plan-rail-substeps` section.
  // Each `.notebook-step` is keyed by its step id; a persisting step keeps its
  // node object (so a :hover on that substep survives the tick) and only has its
  // innerHTML/class refreshed when the step's own markup changed.
  function reconcileSubsteps(section, phaseSteps, phaseNumber) {
    const heading = section.querySelector("h4");
    if (heading) {
      const headingText = `子任务 · ${phaseSteps.length}`;
      if (heading.textContent !== headingText) heading.textContent = headingText;
    }
    const existing = new Map();
    section.querySelectorAll(":scope > .notebook-step").forEach((node) => {
      if (node.dataset.stepKey) existing.set(node.dataset.stepKey, node);
    });
    let cursor = heading || null;
    phaseSteps.forEach((step, index) => {
      const subNumber = phaseNumber ? `${phaseNumber}.${index + 1}` : `${index + 1}`;
      const key = String(step?.id || `idx:${index}`);
      const html = planSubstepHtml(step, subNumber);
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

  // Keyed reconciliation of the whole plan rail. Chrome blocks (fetch error /
  // header badge / event strip / sub-agent rows / start control) are keyed by a
  // fixed slot id and their node is reused-or-replaced only when their markup
  // changed. Phase cards are keyed by phase name so a persisting phase keeps its
  // node (and its hovered substeps) across ticks. Returns false when the target
  // cannot be reconciled so the caller can fall back to innerHTML.
  function reconcilePlanRail(container, plan, opts) {
    if (!supportsReconciliation(container)) return false;
    const phases = planPhasePlan(plan);
    // Ordered slot descriptors. `phase` slots recurse into substep keying; all
    // other slots are simple keyed HTML blocks (may be empty -> absent).
    const slots = [
      { key: "fetch-error", html: opts.fetchError
        ? '<div class="plan-rail-fetch-error" role="status">'
          + '<span>计划读取失败，当前显示的是上次缓存的计划。</span>'
          + '<button type="button" class="button compact secondary" data-plan-rail-retry="1">重试</button>'
          + "</div>"
        : "" },
      { key: "header-badge", html: (() => {
        const replanBadge = replanBadgeHtml(plan);
        return replanBadge ? `<div class="plan-rail-header">${replanBadge}</div>` : "";
      })() },
      { key: "event-strip", html: loopEventStripHtml(plan) },
      { key: "subagent-rows", html: subAgentRowsHtml(plan) },
      ...phases.map((entry) => ({ key: `phase:${entry.phase}`, phase: entry })),
      { key: "start-control", html: (plan.status === "validated" && !isAgentMode?.())
        ? '<div class="plan-rail-start">'
          + '<span class="plan-rail-start-status">等待开始执行</span>'
          + '<button type="button" class="button compact plan-step-locate" data-plan-start-locate="1" title="跳到中间的开始执行卡片">定位</button>'
          + "</div>"
        : "" },
    ];
    // Index current top-level children by their data-rail-slot key. Any child
    // without a slot key is leftover from a previous innerHTML-fallback render
    // (empty/error/skeleton state) and must be cleared so the keyed slots start
    // from a clean container.
    const existing = new Map();
    for (const node of Array.from(container.children)) {
      if (node.dataset && node.dataset.railSlot) {
        existing.set(node.dataset.railSlot, node);
      } else {
        node.remove();
      }
    }
    let cursor = null;
    for (const slot of slots) {
      if (slot.phase) {
        const { phase, phaseSteps, phaseNumber } = slot.phase;
        const phaseStatus = planPhaseStatus(phaseSteps);
        let node = existing.get(slot.key);
        if (!node) {
          const shell = [
            `<div class="step plan-rail-step ${escapeHtml(phaseStatus)}" role="group" aria-label="${phaseNumber}. ${escapeHtml(phase)}">`,
            planPhaseHeadHtml(phase, phaseSteps, phaseNumber),
            '<section class="notebook-step-group plan-rail-substeps"><h4></h4></section>',
            "</div>",
          ].join("");
          node = nodeFromHtml(shell);
          if (!node) continue;
          node.dataset.railSlot = slot.key;
        } else {
          // Refresh phase card class + head in place (node preserved).
          node.className = `step plan-rail-step ${phaseStatus}`;
          node.setAttribute("aria-label", `${phaseNumber}. ${phase}`);
          const head = node.querySelector(":scope > .step-head");
          const headHtml = planPhaseHeadHtml(phase, phaseSteps, phaseNumber);
          if (head && head.dataset.headSignature !== headHtml) {
            const fresh = nodeFromHtml(headHtml);
            if (fresh) head.innerHTML = fresh.innerHTML;
            head.dataset.headSignature = headHtml;
          } else if (!head) {
            const fresh = nodeFromHtml(headHtml);
            if (fresh) node.insertBefore(fresh, node.firstChild);
          }
        }
        existing.delete(slot.key);
        // Ensure a substeps section exists, then key its substeps.
        let section = node.querySelector(":scope > .plan-rail-substeps");
        if (phaseSteps.length && !section) {
          section = nodeFromHtml('<section class="notebook-step-group plan-rail-substeps"><h4></h4></section>');
          if (section) node.appendChild(section);
        }
        if (section) {
          if (!phaseSteps.length) {
            section.remove();
          } else {
            reconcileSubsteps(section, phaseSteps, phaseNumber);
          }
        }
        const desiredNext = cursor ? cursor.nextSibling : container.firstChild;
        if (node !== desiredNext) container.insertBefore(node, desiredNext);
        cursor = node;
        continue;
      }
      // Plain keyed HTML slot.
      let node = existing.get(slot.key);
      if (!slot.html) {
        if (node) node.remove();
        existing.delete(slot.key);
        continue;
      }
      if (node) {
        existing.delete(slot.key);
        if (node.dataset.slotSignature !== slot.html) {
          const fresh = nodeFromHtml(slot.html);
          if (fresh) {
            node.className = fresh.className;
            node.innerHTML = fresh.innerHTML;
          }
          node.dataset.slotSignature = slot.html;
        }
      } else {
        node = nodeFromHtml(slot.html);
        if (!node) continue;
        node.dataset.railSlot = slot.key;
        node.dataset.slotSignature = slot.html;
      }
      const desiredNext = cursor ? cursor.nextSibling : container.firstChild;
      if (node !== desiredNext) container.insertBefore(node, desiredNext);
      cursor = node;
    }
    // Drop any slot node that no longer has a descriptor (removed phase/chrome).
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
    const blocked = driverHasBlockingError();
    const fetchError = v2PlanFetchErrors.get(taskId) || "";
    const planSignature = JSON.stringify({ task: taskId, plan, blocked, fetchError, firstLoad });
    if (force || renderSignatures.workflowStepper !== planSignature) {
      renderSignatures.workflowStepper = planSignature;
      const planStepper = $("workflowStepper");
      if (planStepper) {
        // A populated plan gets node-preserving keyed reconciliation so hovering
        // a step card during the per-second poll does not rebuild the node under
        // the cursor (the flicker fix). Empty/error/skeleton states are single
        // transient blocks with no hover target, and the static test harness
        // passes a bare innerHTML mock — both fall back to a plain innerHTML set.
        const hasSteps = Boolean(plan && (plan.steps || []).length);
        const reconciled = hasSteps
          && !fetchError
          && reconcilePlanRail(planStepper, plan, { fetchError });
        if (!reconciled) {
          planStepper.innerHTML = planRailHtml(plan, { blocked, fetchError, firstLoad });
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
    // Rail's lightweight entry: open the middle retry panel and scroll to the
    // step's card. The heavy form itself lives in the middle workspace.
    const planRetryOpen = event.target?.closest?.("[data-plan-retry-open]");
    if (planRetryOpen) {
      event.preventDefault();
      event.stopPropagation();
      openRetryCard(planRetryOpen.dataset.planRetryOpen || "");
      return true;
    }
    const planRetryButton = event.target?.closest?.("[data-plan-retry-step]");
    if (planRetryButton) {
      event.preventDefault();
      event.stopPropagation();
      void retryPlanStep(planRetryButton);
      return true;
    }
    // Rail's lightweight "待确认" gate entry: scroll the middle gate section into
    // view and flash it (the confirm control itself already lives there).
    const gateLocate = event.target?.closest?.("[data-plan-gate-locate]");
    if (gateLocate) {
      event.preventDefault();
      event.stopPropagation();
      openGateCard(gateLocate.dataset.planGateLocate || "");
      return true;
    }
    // Rail's lightweight 开始执行 / 下载报告 entries: reveal the middle
    // driver-actions panel and flash the matching action card.
    const startLocate = event.target?.closest?.("[data-plan-start-locate]");
    if (startLocate) {
      event.preventDefault();
      event.stopPropagation();
      openDriverActionCard("start");
      return true;
    }
    const reportLocate = event.target?.closest?.("[data-plan-report-locate]");
    if (reportLocate) {
      event.preventDefault();
      event.stopPropagation();
      openDriverActionCard("report-download");
      return true;
    }
    const retryButton = event.target?.closest?.("[data-plan-rail-retry]");
    if (retryButton) {
      event.preventDefault();
      event.stopPropagation();
      retryFetch();
      return true;
    }
    // UX-5: "发消息介入" on a no_progress event strip row — hands the user
    // straight into the composer instead of leaving them to hunt for it.
    const interveneButton = event.target?.closest?.("[data-plan-rail-intervene]");
    if (interveneButton) {
      event.preventDefault();
      event.stopPropagation();
      fillComposer?.();
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

  return {
    artifactPreviewContainer,
    clearArtifactPanel,
    clearDriverActionsPanel,
    clearRetryPanel,
    handleClick,
    installArtifactHandlers,
    maybeFetchPlan,
    nextStepAfter,
    planStep,
    render,
    resetFetchThrottle,
    retryFetch,
  };
}
