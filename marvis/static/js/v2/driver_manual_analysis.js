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

// Manual mode for driver tasks (data_join / feature / modeling): render each
// step's output as a plain analysis section. The plan overview is omitted because
// the step rail already shows the plan, and gate-specific renderers are injected
// so this module does not own individual gate UI implementations.
export function driverManualAnalysisHtml(messages, renderers = {}) {
  const renderMarkdown = renderers.renderAgentMarkdown || markdownRenderer;
  const renderC1Form = renderers.renderC1Form || emptyRenderer;
  const renderDedupPicker = renderers.renderDedupPicker || emptyRenderer;
  const renderModelingSetup = renderers.renderModelingSetup || emptyRenderer;
  const renderScreenTable = renderers.renderScreenTable || emptyRenderer;
  const renderTables = renderers.renderTables || emptyRenderer;
  const renderModelDelivery = renderers.renderModelDelivery || emptyRenderer;

  const sections = [];
  const latestScreenMessageId = latestInteractiveScreenMessageId(messages);
  for (const message of messages || []) {
    if (message?.role !== "assistant") continue;
    const meta = message.metadata || {};
    if (meta.kind === "overview" || meta.kind === "plan_overview") continue;
    if (meta.error) {
      sections.push(
        `<section class="driver-analysis-section is-error">${renderMarkdown(message.content || "")}</section>`,
      );
      continue;
    }
    const intro = renderMarkdown(stripChatInstructions(message.content || ""));
    if (meta.join_c1) {
      sections.push(`<section class="driver-analysis-section">${intro}${renderC1Form(message)}</section>`);
      continue;
    }
    if (meta.screen) {
      const interactive = String(message.id || "") === latestScreenMessageId;
      sections.push(
        `<section class="driver-analysis-section">${intro}${renderModelingSetup(message, { interactive })}${renderScreenTable(message, { interactive })}</section>`,
      );
      continue;
    }
    if (meta.modeling_setup) {
      const interactive = meta.kind === "gate";
      sections.push(`<section class="driver-analysis-section">${intro}${renderModelingSetup(message, { interactive })}${renderTables(message)}</section>`);
      continue;
    }
    if (meta.dedup) {
      const diagTables = renderTables(message);
      sections.push(`<section class="driver-analysis-section">${intro}${diagTables}${renderDedupPicker(message)}</section>`);
      continue;
    }
    if (meta.model_delivery) {
      sections.push(`<section class="driver-analysis-section">${intro}${renderModelDelivery(message)}${renderTables(message)}</section>`);
      continue;
    }
    const tables = renderTables(message);
    if (!String(message.content || "").trim() && !tables) continue;
    sections.push(`<section class="driver-analysis-section">${intro}${tables}</section>`);
  }
  return sections.join("") || '<div class="plan-rail-empty">尚无分析结果，请在右侧步骤栏操作。</div>';
}
