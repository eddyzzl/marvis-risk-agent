const emptyRenderer = () => "";
const markdownRenderer = (value) => String(value || "");

// Lines that only make sense in a chat ("回复「确认」继续…"). In manual mode the
// confirm is a step-rail button, so these instructions are stripped from the
// analysis text; what remains is the factual statistical summary.
export function stripChatInstructions(content) {
  return String(content || "")
    .split("\n")
    .filter((line) => !/(回复「确认」|确认请回复|要调整可|可直接说明|请确认.*回复)/.test(line))
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

// UX-2: does this gate message carry a structured widget payload at all
// (screening table / dedup picker / modeling setup panel / C1 role form)?
// Shared by both modes so the "does this gate need a widget" decision lives
// in exactly one place.
export function driverGateHasWidget(message) {
  const meta = message?.metadata || {};
  return Boolean(meta.join_c1 || meta.screen || meta.modeling_setup || meta.dedup);
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
  const renderC1Form = renderers.renderC1Form || emptyRenderer;
  const renderDedupPicker = renderers.renderDedupPicker || emptyRenderer;
  const renderModelingSetup = renderers.renderModelingSetup || emptyRenderer;
  const renderScreenTable = renderers.renderScreenTable || emptyRenderer;
  const renderTables = renderers.renderTables || emptyRenderer;
  const meta = message?.metadata || {};
  const interactive = options.interactive !== false;
  if (meta.join_c1) return renderC1Form(message, { interactive });
  if (meta.screen) return `${renderModelingSetup(message, { interactive })}${renderScreenTable(message, { interactive })}`;
  if (meta.modeling_setup) return `${renderModelingSetup(message, { interactive })}${renderTables(message)}`;
  if (meta.dedup) return `${renderTables(message)}${renderDedupPicker(message, { interactive })}`;
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

  const sections = [];
  const latestScreenMessageId = latestInteractiveScreenMessageId(messages);
  const lastMessageId = lastAssistantMessageId(messages);
  for (const message of messages || []) {
    if (message?.role !== "assistant") continue;
    const meta = message.metadata || {};
    if (meta.kind === "overview" || meta.kind === "plan_overview") continue;
    const isPendingGate = meta.kind === "gate" && String(message.id || "") === lastMessageId;
    const sectionClass = isPendingGate ? "driver-analysis-section is-gate-pending" : "driver-analysis-section";
    if (meta.error) {
      sections.push(
        `<section class="driver-analysis-section is-error">${renderMarkdown(message.content || "")}</section>`,
      );
      continue;
    }
    const intro = renderMarkdown(stripChatInstructions(message.content || ""));
    if (meta.join_c1) {
      sections.push(`<section class="${sectionClass}">${intro}${driverGateBodyHtml(message, renderers)}</section>`);
      continue;
    }
    if (meta.screen) {
      const interactive = String(message.id || "") === latestScreenMessageId;
      sections.push(
        `<section class="${sectionClass}">${intro}${driverGateBodyHtml(message, renderers, { interactive })}</section>`,
      );
      continue;
    }
    if (meta.modeling_setup) {
      const interactive = meta.kind === "gate";
      sections.push(`<section class="${sectionClass}">${intro}${driverGateBodyHtml(message, renderers, { interactive })}</section>`);
      continue;
    }
    if (meta.dedup) {
      sections.push(`<section class="${sectionClass}">${intro}${driverGateBodyHtml(message, renderers)}</section>`);
      continue;
    }
    if (meta.model_delivery) {
      sections.push(`<section class="${sectionClass}">${intro}${renderModelDelivery(message)}${renderTables(message)}</section>`);
      continue;
    }
    const tables = renderTables(message);
    if (!String(message.content || "").trim() && !tables) continue;
    sections.push(`<section class="${sectionClass}">${intro}${tables}</section>`);
  }
  return sections.join("") || '<div class="plan-rail-empty">尚无分析结果，请在右侧步骤栏操作。</div>';
}
