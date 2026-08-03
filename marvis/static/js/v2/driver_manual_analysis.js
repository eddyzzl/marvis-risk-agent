import { isStrategyClarificationMessage } from "./strategy_clarification_controller.js";
import {
  hasWorkflowErrorDiagnostic,
  workflowMessageContentHtml,
} from "./workflow_error_card.js";

const emptyRenderer = () => "";
const markdownRenderer = (value) => String(value || "");

// Lines that only make sense in a chat ("回复「确认」继续…"). In manual mode the
// confirm is a step-rail button, so these instructions are stripped from the
// analysis text; what remains is the factual statistical summary.
export function stripChatInstructions(content) {
  return String(content || "")
    .split("\n")
    .filter((line) => !/(回复「|确认请回复|Agent 模式请回复|要调整可|可直接说明|请确认.*回复)/.test(line))
    .join("\n")
    .trim();
}

export function latestInteractiveScreenMessageId(messages = []) {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message?.role !== "assistant") continue;
    const meta = message.metadata || {};
    if (meta.kind === "gate") return meta.screen ? String(message.id || "") : "";
  }
  return "";
}

// VD-2: only the LAST assistant message can still be an actionable pending
// gate (the step-rail confirm button only ever targets the latest step) —
// earlier "kind: gate" messages are resolved history. Manual mode keeps its
// "tool, not conversation" section styling (no bubble/card shell) but still
// needs to mark the one section that is genuinely awaiting confirmation, so
// it gets the same tone bar language as the agent-mode gate card.
export function lastAssistantMessageId(messages = []) {
  for (let index = messages.length - 1; index >= 0; index--) {
    if (messages[index]?.role === "assistant") return String(messages[index].id || "");
  }
  return "";
}

export function filterRecoveredWorkflowFailures(messages = [], stepStatus = () => "") {
  return (messages || []).filter((message) => {
    if (message?.role !== "assistant") return true;
    const meta = message.metadata || {};
    if (!meta.error && !hasWorkflowErrorDiagnostic(meta)) return true;
    const currentStatus = String(stepStatus(message) || "");
    // A failure from a superseded plan is still audit evidence. Keep it in the
    // transcript; the manual renderer presents it as collapsed, read-only
    // history and excludes it from current-gate interactivity below.
    return !currentStatus || currentStatus === "failed" || currentStatus === "historical";
  });
}

function historicalWorkflowFailureHtml(message, body) {
  const meta = message?.metadata || {};
  const planId = String(meta.plan_id || meta.failure_envelope?.plan_id || "").trim();
  const stepId = String(meta.step_id || meta.failure_envelope?.failed_step_id || "").trim();
  const context = [
    planId ? `计划 ${planId}` : "",
    stepId ? `步骤 ${stepId}` : "",
  ].filter(Boolean).map(escapeAttr).join(" · ");
  const contextHtml = context
    ? `<div class="driver-analysis-history__context">${context}</div>`
    : "";
  return (
    '<details class="driver-analysis-history is-error">'
    + '<summary>历史计划失败（只读）</summary>'
    + contextHtml
    + body
    + "</details>"
  );
}

export function gateMessageForCurrentTool(message) {
  const meta = message?.metadata || {};
  const gateTool = String(meta.gate_source_tool || "");
  // A screen_features output can remain a direct dependency of later modeling
  // gates (notably tune_hyperparameters). It is useful as evidence, but its
  // checkbox table is only the action component for the select_features gate.
  // Strip the stale widget marker for later gates so they render their own
  // plain, consequence-labelled confirmation action.
  if (meta.kind !== "gate" || !gateTool) return message;
  const metadata = { ...meta };
  let changed = false;
  if (meta.screen && gateTool !== "select_features") {
    delete metadata.screen;
    changed = true;
  }
  // The setup workbench is actionable only while accepting the split/model
  // specification or configuring the tuning budget. Later gates can still
  // depend on that output for evidence, but must not resurrect its controls.
  if (
    meta.modeling_setup
    && !["screen_features", "configure_tuning"].includes(gateTool)
  ) {
    delete metadata.modeling_setup;
    changed = true;
  }
  return changed ? { ...message, metadata } : message;
}

// UX-2: does this gate message carry a structured widget payload at all
// (screening table / dedup picker / modeling setup panel / C1 role form)?
// Shared by both modes so the "does this gate need a widget" decision lives
// in exactly one place.
export function driverGateHasWidget(message) {
  const meta = gateMessageForCurrentTool(message)?.metadata || {};
  return Boolean(
    meta.join_c1
    || meta.screen
    || meta.modeling_setup
    || meta.dedup
    || meta.join_keys
    || meta.feature_binning
    || meta.special_values
    || meta.editable_input_schema?.properties?.adoption_reason
  );
}

// UX-2: mounts the FULL body (structured widget(s) + any accompanying
// diagnostics tables) for a single gate message, exactly matching the
// widget/table placement manual mode has always used per gate kind (tables
// alongside modeling_setup/dedup, no separate tables for join_c1/screen since
// their widgets already surface the relevant data). The widget components
// themselves are mode-agnostic (they only read message.metadata and post
// through the shared /agent/messages endpoint), so this is the ONE place that
// decides which widget(s) a given gate message's metadata calls for. Both
// manual mode (driverManualAnalysisHtml, below) and agent mode (app.js's
// agentMessageHtml) call this instead of re-deciding the branch themselves,
// so a gate always gets the same controls in both modes.
export function driverGateBodyHtml(message, renderers = {}, options = {}) {
  message = gateMessageForCurrentTool(message);
  const renderC1Form = renderers.renderC1Form || emptyRenderer;
  const renderDedupPicker = renderers.renderDedupPicker || emptyRenderer;
  const renderJoinKeyPicker = renderers.renderJoinKeyPicker || emptyRenderer;
  const renderModelingSetup = renderers.renderModelingSetup || emptyRenderer;
  const renderScreenTable = renderers.renderScreenTable || emptyRenderer;
  const renderAdoptionGate = renderers.renderAdoptionGate || emptyRenderer;
  const renderFeatureBinning = renderers.renderFeatureBinning || emptyRenderer;
  const renderSpecialValues = renderers.renderSpecialValues || emptyRenderer;
  const renderTables = renderers.renderTables || emptyRenderer;
  const meta = message?.metadata || {};
  const interactive = options.interactive !== false;
  if (meta.join_c1) return renderC1Form(message, { interactive });
  if (meta.screen) return `${renderModelingSetup(message, { interactive })}${renderScreenTable(message, { interactive })}`;
  if (meta.modeling_setup) return `${renderModelingSetup(message, { interactive })}${renderTables(message)}`;
  if (meta.join_keys) return `${renderTables(message)}${renderJoinKeyPicker(message, { interactive })}`;
  if (meta.dedup) return `${renderTables(message)}${renderDedupPicker(message, { interactive })}`;
  if (meta.feature_binning) return `${renderTables(message)}${renderFeatureBinning(message, { interactive })}`;
  if (meta.special_values) return `${renderTables(message)}${renderSpecialValues(message, { interactive })}`;
  if (meta.editable_input_schema?.properties?.adoption_reason) {
    return `${renderTables(message)}${renderAdoptionGate(message, { interactive })}`;
  }
  return "";
}

// Manual mode for driver tasks (data_join / feature / modeling): render each
// step's output as a plain analysis section. The plan overview is omitted because
// the step rail already shows the plan, and gate-specific renderers are injected
// so this module does not own individual gate UI implementations.
export function driverManualAnalysisHtml(messages, renderers = {}) {
  const renderMarkdown = renderers.renderAgentMarkdown || markdownRenderer;
  const renderTables = renderers.renderTables || emptyRenderer;
  const renderModelDelivery = renderers.renderModelDelivery || emptyRenderer;
  const renderResultDataset = renderers.renderResultDataset || emptyRenderer;
  const renderReportDownload = renderers.renderReportDownload || emptyRenderer;
  const renderStrategyClarification = renderers.renderStrategyClarification || emptyRenderer;
  // The plain-gate confirm control (a gate with no structured widget). In manual
  // mode ALL interactive controls live in this middle region now, so the pending
  // gate section renders its own confirm button here instead of the rail. Widget
  // gates (join_c1 / screen / modeling_setup / dedup) already carry their own
  // primary action inside the widget, so this renders nothing for them.
  const renderGateConfirm = renderers.renderGateConfirm || emptyRenderer;
  const stepStatus = typeof renderers.stepStatus === "function"
    ? renderers.stepStatus
    : () => "";

  const sections = [];
  // A retry keeps the original failure message for audit history. Once the
  // authoritative plan says that failed step has recovered, the stale message
  // must no longer win the "latest assistant" slot or cover the newly-opened
  // confirmation gate. Unknown step status keeps the legacy behavior so setup
  // failures without a plan remain visible.
  const visibleMessages = filterRecoveredWorkflowFailures(messages, stepStatus)
    .map(gateMessageForCurrentTool);
  const currentPlanMessages = visibleMessages.filter(
    (message) => String(stepStatus(message) || "") !== "historical",
  );
  const latestScreenMessageId = latestInteractiveScreenMessageId(currentPlanMessages);
  const lastMessageId = lastAssistantMessageId(currentPlanMessages);
  for (const message of visibleMessages) {
    if (message?.role !== "assistant") continue;
    const meta = message.metadata || {};
    const messageStatus = String(stepStatus(message) || "");
    const isHistoricalFailure = messageStatus === "historical"
      && (Boolean(meta.error) || hasWorkflowErrorDiagnostic(meta));
    if (meta.kind === "overview" || meta.kind === "plan_overview") continue;
    // Canonical C1 role proposals are emitted before PlanDriver creates a plan
    // and intentionally carry `join_c1` without `kind: "gate"`. Treat that
    // structured pre-plan proposal as the pending gate too; otherwise Manual
    // mode renders its only actionable role/target form as historical/disabled.
    const isPendingGate = (
      meta.kind === "gate"
      || Boolean(meta.join_c1)
    ) && String(message.id || "") === lastMessageId;
    // A stable per-step anchor so the rail's lightweight "待确认" locate entry can
    // scroll to (and flash) exactly this middle gate section.
    const stepId = meta.step_id ? String(meta.step_id) : "";
    const gateAttr = isPendingGate
      ? ` data-driver-gate-section="${escapeAttr(stepId)}"`
      : "";
    const sectionClass = isPendingGate ? "driver-analysis-section is-gate-pending" : "driver-analysis-section";
    if (hasWorkflowErrorDiagnostic(meta)) {
      const diagnostic = workflowMessageContentHtml(
        message,
        (content) => renderMarkdown(content),
      );
      const content = isHistoricalFailure
        ? historicalWorkflowFailureHtml(message, diagnostic)
        : diagnostic;
      sections.push(
        `<section class="driver-analysis-section is-error has-workflow-error${isHistoricalFailure ? " is-historical" : ""}">${content}</section>`,
      );
      continue;
    }
    if (meta.error) {
      const legacyError = workflowMessageContentHtml(
        message,
        (content) => renderMarkdown(content),
      );
      const content = isHistoricalFailure
        ? historicalWorkflowFailureHtml(message, legacyError)
        : legacyError;
      sections.push(
        `<section class="driver-analysis-section is-error${isHistoricalFailure ? " is-historical" : ""}">${content}</section>`,
      );
      continue;
    }
    const intro = workflowMessageContentHtml(
      message,
      (content) => renderMarkdown(stripChatInstructions(content)),
    );
    if (isStrategyClarificationMessage(message)) {
      const interactive = String(message.id || "") === lastMessageId;
      const clarificationClass = interactive
        ? "driver-analysis-section is-clarification is-clarification-pending"
        : "driver-analysis-section is-clarification";
      sections.push(
        `<section class="${clarificationClass}">${intro}${renderStrategyClarification(message, { interactive })}</section>`,
      );
      continue;
    }
    if (meta.join_c1) {
      sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${driverGateBodyHtml(message, renderers, { interactive: isPendingGate })}</section>`);
      continue;
    }
    if (meta.screen) {
      const interactive = String(message.id || "") === latestScreenMessageId;
      sections.push(
        `<section class="${sectionClass}"${gateAttr}>${intro}${driverGateBodyHtml(message, renderers, { interactive })}</section>`,
      );
      continue;
    }
    if (meta.modeling_setup) {
      sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${driverGateBodyHtml(message, renderers, { interactive: isPendingGate })}</section>`);
      continue;
    }
    if (meta.join_keys) {
      sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${driverGateBodyHtml(message, renderers, { interactive: isPendingGate })}</section>`);
      continue;
    }
    if (meta.dedup) {
      sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${driverGateBodyHtml(message, renderers, { interactive: isPendingGate })}</section>`);
      continue;
    }
    if (meta.feature_binning) {
      sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${driverGateBodyHtml(message, renderers, { interactive: isPendingGate })}</section>`);
      continue;
    }
    if (meta.special_values) {
      sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${driverGateBodyHtml(message, renderers, { interactive: isPendingGate })}</section>`);
      continue;
    }
    if (meta.editable_input_schema?.properties?.adoption_reason) {
      sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${driverGateBodyHtml(message, renderers, { interactive: isPendingGate })}</section>`);
      continue;
    }
    if (meta.model_delivery) {
      const confirm = isPendingGate ? renderGateConfirm(message) : "";
      sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${renderModelDelivery(message)}${renderTables(message)}${renderReportDownload(message)}${confirm}</section>`);
      continue;
    }
    const tables = renderTables(message);
    // A plain gate (no widget) still needs its confirm control; render it in this
    // middle section. Non-gate plain sections with no text and no tables are
    // skipped as before.
    const confirm = isPendingGate ? renderGateConfirm(message) : "";
    const hasIngestNotices = Array.isArray(meta.ingest_notices) && meta.ingest_notices.length > 0;
    if (!String(message.content || "").trim() && !tables && !confirm && !hasIngestNotices) continue;
    sections.push(`<section class="${sectionClass}"${gateAttr}>${intro}${tables}${renderResultDataset(message)}${renderReportDownload(message)}${confirm}</section>`);
  }
  return sections.join("") || '<div class="plan-rail-empty">尚无分析结果，请在右侧步骤栏操作。</div>';
}

// Minimal attribute escaper for the anchor id (backend step slugs are safe, but
// guard the attribute value so a stray quote can't break the section markup).
function escapeAttr(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
