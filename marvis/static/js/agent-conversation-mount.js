import {
  agentFrozenSnapshotsByTriggerId,
  agentMessagesHtml,
  agentTimelineItems,
} from "./agent-conversation-view.js";
import { latestPendingReportDraftMessageId, latestReportDraftMessageId } from "./report-draft-table.js";
import { localizeHtml } from "./i18n.js";

// UX-2: same "only the latest assistant gate is interactive" rule manual mode
// uses (driver_manual_analysis.js's lastAssistantMessageId), narrowed to
// actual gate messages the same way the backend's own latest_open_gate()
// does (turn_handlers.py): kind === "gate", or a join_c1 turn (which carries
// no explicit kind — see plan_message_composer.py). Kept as a local copy
// here since this module has no dependency on the v2 driver modules, and the
// timeline bucketing needs it computed once over the FULL message list,
// before messages are split into per-bucket chunks below.
// Answers to questions asked while a gate waits stay on the gate's step and
// carry no kind (or are plain chat with no plan). Acks, results and progress
// mean the plan moved on.
function gateConversationReply(meta, stepId) {
  if (meta.kind || meta.ui_action || meta.execution_completed || meta.tool_call) {
    return false;
  }
  const replyStep = String(meta.step_id || "");
  return replyStep ? replyStep === stepId : !meta.plan_id;
}

// The gate whose controls stay live: the newest gate message, also when the
// user asked questions after it (eval round 1: two questions at the binning
// gate removed its buttons). Whether it can still act is decided against the
// current plan step and confirmation snapshot, not here.
export function lastGateMessageId(messages = []) {
  const later = [];
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message?.role !== "assistant") continue;
    const meta = message?.metadata || {};
    if (meta.kind === "gate" || meta.join_c1) {
      const stepId = String(meta.step_id || "");
      return later.every((other) => gateConversationReply(other, stepId)) ? String(message.id || "") : "";
    }
    later.push(meta);
  }
  return "";
}

function defaultDocument() {
  return globalThis.document || null;
}

function elementGetter(root = defaultDocument()) {
  return (id) => root?.getElementById?.(id) || null;
}

export function cssEscapeAttr(value, cssApi = globalThis.CSS) {
  if (cssApi && typeof cssApi.escape === "function") {
    return cssApi.escape(value);
  }
  return String(value).replace(/["\\]/g, (ch) => `\\${ch}`);
}

export function updateAgentMessageContentsInPlace(messages = [], deps = {}) {
  const root = deps.root || defaultDocument();
  const getElementById = deps.getElementById || elementGetter(root);
  const scrollContent = deps.scrollContent || getElementById("resultScrollContent");
  if (!scrollContent) return false;
  for (const message of messages) {
    const messageId = message?.id ? String(message.id) : "";
    if (!messageId) return false;
    const article = scrollContent.querySelector(
      `article[data-agent-message-id="${cssEscapeAttr(messageId, deps.cssApi)}"]`,
    );
    if (!article) return false;
    const contentNode = article.querySelector(".agent-message-content");
    if (!contentNode) return false;
    // Structured cards (failure / repair / ingest notices) are rebuilt by the
    // structural render; patching them with Markdown would flatten the card.
    if (deps.hasStructuredContent?.(message)) continue;
    const streaming = Boolean(deps.isStreaming?.(message));
    const thinking = Boolean(deps.isThinking?.(message));
    const nextHtml = thinking
      ? String(deps.thinkingHtml?.() || "")
      : String(deps.formatMessageContent?.(
        deps.visibleContent?.(message) ?? String(message?.content || ""),
        { markdown: message?.role !== "user" },
      ) || "");
    if (contentNode.innerHTML !== localizeHtml(nextHtml, { context: contentNode })) contentNode.innerHTML = nextHtml;
    contentNode.dataset.agentStreaming = streaming ? "true" : "false";
    contentNode.dataset.agentThinking = thinking ? "true" : "false";
    const referencesNode = article.querySelector(".agent-memory-references");
    const referencesHtml = message?.role === "user"
      ? ""
      : String(deps.memoryReferencesHtml?.(message?.metadata?.memory_references) || "");
    if (referencesNode) {
      if (referencesHtml) referencesNode.outerHTML = referencesHtml;
      else referencesNode.remove();
    } else if (referencesHtml) {
      contentNode.insertAdjacentHTML("afterend", referencesHtml);
    }
  }
  return true;
}

export function removeAgentTimelineBuckets(root = defaultDocument()) {
  root?.querySelectorAll?.("[data-agent-timeline-bucket]")?.forEach((bucket) => bucket.remove());
  root?.querySelectorAll?.("[data-agent-frozen-snapshot]")?.forEach((node) => node.remove());
}

export function restoreResultScrollDefaultOrder(deps = {}) {
  const root = deps.root || defaultDocument();
  const getElementById = deps.getElementById || elementGetter(root);
  const scrollContent = deps.scrollContent || getElementById("resultScrollContent");
  if (!scrollContent) return;
  removeAgentTimelineBuckets(root);
  for (const elementId of deps.persistentElementIds || []) {
    const element = getElementById(elementId);
    if (element) scrollContent.appendChild(element);
  }
}

export function createAgentTimelineBucket(root = defaultDocument()) {
  const bucket = root.createElement("section");
  bucket.className = "agent-conversation agent-timeline-bucket";
  bucket.dataset.agentTimelineBucket = "true";
  bucket.setAttribute("aria-label", "Agent 对话片段");
  const messages = root.createElement("div");
  messages.className = "agent-messages";
  bucket.appendChild(messages);
  return bucket;
}

export function renderAgentTimeline(messages = [], deps = {}) {
  const root = deps.root || defaultDocument();
  const getElementById = deps.getElementById || elementGetter(root);
  const scrollContent = deps.scrollContent || getElementById("resultScrollContent");
  const basePanel = deps.basePanel || getElementById("agentConversationPanel");
  const baseMessages = deps.baseMessages || getElementById("agentMessages");
  if (!root || !scrollContent || !basePanel || !baseMessages) return;

  removeAgentTimelineBuckets(root);
  baseMessages.innerHTML = "";
  basePanel.classList.add("hidden");
  basePanel.setAttribute("aria-hidden", "true");

  const items = agentTimelineItems(messages, deps.visibleStages || [], {
    snapshotsByTrigger: deps.snapshotsByTrigger || agentFrozenSnapshotsByTriggerId({
      selectedTaskId: deps.selectedTaskId,
      taskFrozenSectionSnapshots: deps.taskFrozenSectionSnapshots,
      agentMessages: deps.agentMessages,
    }),
  });
  const latestGateMessageId = deps.latestGateMessageId ?? lastGateMessageId(messages);
  const pendingReportDraftMessageId = deps.latestPendingReportDraftMessageId
    ?? latestPendingReportDraftMessageId(messages);
  const reportDraftMessageId = deps.latestReportDraftMessageId
    ?? latestReportDraftMessageId(messages);
  const appendedSections = new Set();
  let basePanelUsed = false;

  for (const item of items) {
    if (item.type === "stage") {
      const section = getElementById(item.sectionId);
      if (!section) continue;
      scrollContent.appendChild(section);
      appendedSections.add(item.sectionId);
      continue;
    }
    if (item.type === "frozen") {
      const frozen = deps.createFrozenSnapshotElement?.(item.snapshot);
      if (frozen) scrollContent.appendChild(frozen);
      continue;
    }
    if (item.type !== "messages" || !item.messages.length) continue;
    const bucket = basePanelUsed ? createAgentTimelineBucket(root) : basePanel;
    const bucketMessages = basePanelUsed ? bucket.querySelector(".agent-messages") : baseMessages;
    basePanelUsed = true;
    bucket.classList.remove("hidden");
    bucket.setAttribute("aria-hidden", "false");
    bucketMessages.innerHTML = agentMessagesHtml(item.messages, undefined, {
      agentStageLabel: deps.agentStageLabel,
      agentMessageHtml: deps.agentMessageHtml,
      latestGateMessageId,
      latestPendingReportDraftMessageId: pendingReportDraftMessageId,
      latestReportDraftMessageId: reportDraftMessageId,
    });
    scrollContent.appendChild(bucket);
  }

  if (!basePanelUsed) {
    scrollContent.appendChild(basePanel);
  }
  for (const elementId of deps.persistentElementIds || []) {
    if (elementId === "agentConversationPanel" || appendedSections.has(elementId)) continue;
    const element = getElementById(elementId);
    if (element) scrollContent.appendChild(element);
  }
}
