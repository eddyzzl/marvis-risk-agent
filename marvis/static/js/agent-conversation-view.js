export function agentTimelineStageDefinitions() {
  return [
    { stage: "scan", sectionId: "scanSection", order: 1 },
    { stage: "reproducibility", sectionId: "notebookSection", order: 2 },
    { stage: "metrics", sectionId: "metricSection", order: 3 },
  ];
}

export function agentRerunMessageFingerprint(message) {
  if (message?.role !== "user") return "";
  const intent = message?.metadata?.intent;
  if (intent !== "rerun_stage" && intent !== "regenerate_report_draft") return "";
  const stage = message?.metadata?.target_stage || "";
  const content = String(message?.content || "").trim();
  return `${intent}|${stage}|${content}`;
}

export function agentFrozenSnapshotsByTriggerId({
  selectedTaskId,
  taskFrozenSectionSnapshots,
  agentMessages,
} = {}) {
  if (!selectedTaskId) return new Map();
  const stored = taskFrozenSectionSnapshots?.get?.(selectedTaskId) || [];
  if (stored.length === 0) return new Map();
  if (!Array.isArray(agentMessages)) return new Map();
  const messagesById = new Map();
  const messagesByFingerprint = new Map();
  for (const message of agentMessages) {
    const id = message?.id ? String(message.id) : "";
    if (id) messagesById.set(id, message);
    const fingerprint = agentRerunMessageFingerprint(message);
    if (fingerprint && !messagesByFingerprint.has(fingerprint)) {
      messagesByFingerprint.set(fingerprint, message);
    }
  }
  const result = new Map();
  for (const snapshot of stored) {
    let anchorMessage = null;
    if (snapshot.triggerMessageId && messagesById.has(snapshot.triggerMessageId)) {
      anchorMessage = messagesById.get(snapshot.triggerMessageId);
    } else if (snapshot.triggerFingerprint && messagesByFingerprint.has(snapshot.triggerFingerprint)) {
      anchorMessage = messagesByFingerprint.get(snapshot.triggerFingerprint);
    }
    if (!anchorMessage) continue;
    const anchorId = String(anchorMessage.id);
    // Re-anchor: keep snapshot pointed at the current real id so future
    // renders (and the structural signature) stay stable.
    snapshot.triggerMessageId = anchorId;
    result.set(anchorId, snapshot);
  }
  return result;
}

export function agentMessageIsContinuePrompt(message) {
  const metadata = message?.metadata || {};
  return Boolean(metadata.awaiting_next_stage);
}

export function agentMessageContent(message) {
  return String(message?.content || "").trim();
}

export function stripAgentAdvanceIntentAffixes(value) {
  let content = value;
  const prefixes = ["好的", "好", "那", "请", "麻烦", "帮我", "先", "可以", "确认"];
  const suffixes = ["一下", "下", "吧", "了"];
  let changed = true;
  while (changed) {
    changed = false;
    for (const prefix of prefixes) {
      if (content.startsWith(prefix) && content.length > prefix.length) {
        content = content.slice(prefix.length);
        changed = true;
      }
    }
    for (const suffix of suffixes) {
      if (content.endsWith(suffix) && content.length > suffix.length) {
        content = content.slice(0, -suffix.length);
        changed = true;
      }
    }
  }
  return content;
}

export function agentMessageIsAdvanceIntent(message) {
  const metadata = message?.metadata || {};
  if (metadata.intent === "advance") return true;
  const content = agentMessageContent(message).replace(/\s+/gu, "").replace(/[。.!！?？]+$/u, "");
  if (["不继续", "不要继续", "先不继续", "暂不继续", "不用继续", "别继续", "无需继续"].some((marker) => content.includes(marker))) {
    return false;
  }
  const phrases = ["开始", "开始验证", "继续", "继续吧", "继续执行", "继续下一步", "下一步"];
  if (phrases.includes(content)) return true;
  return phrases.includes(stripAgentAdvanceIntentAffixes(content));
}

export function agentMessageIsScanLead(message) {
  const metadata = message?.metadata || {};
  return metadata.tool_call?.name === "scan_materials";
}

export function agentReportMessagesForDisplay(messages = []) {
  const latestConfirmationIndex = messages.reduce(
    (latestIndex, message, index) => (
      message?.stage === "word_conclusion_confirmed" ? index : latestIndex
    ),
    -1,
  );
  if (latestConfirmationIndex < 0) return messages;
  return messages.filter((message, index) => {
    if (index > latestConfirmationIndex) return true;
    return !(message?.stage === "chat" && message?.metadata?.awaiting_confirmation);
  });
}

export function agentMessagesHtml(messages = [], labelStage, deps = {}) {
  const stageLabel = deps.agentStageLabel || (() => "Agent");
  const messageHtml = deps.agentMessageHtml || ((message) => String(message?.content || ""));
  let previousAssistantLabel = "";
  return messages.map((message) => {
    const resolvedLabelStage = labelStage === undefined ? message?.stage : labelStage;
    const label = message?.role === "user" ? "" : stageLabel(resolvedLabelStage);
    const hideMeta = Boolean(label && label === previousAssistantLabel);
    previousAssistantLabel = label || "";
    return messageHtml(message, resolvedLabelStage, { hideMeta });
  }).join("");
}

function findLastMessageIndex(messages, predicate) {
  if (!Array.isArray(messages)) return -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (predicate(messages[index], index)) return index;
  }
  return -1;
}

export function agentTimelineInsertionIndex(messages = [], stage = "", deps = {}) {
  if (!Array.isArray(messages) || messages.length === 0) return 0;
  const isScanLead = deps.agentMessageIsScanLead || agentMessageIsScanLead;
  if (stage === "scan") {
    const scanLeadIndex = findLastMessageIndex(messages, (message) => (
      message?.stage === "scan" && isScanLead(message)
    ));
    if (scanLeadIndex >= 0) return scanLeadIndex + 1;
    const scanIndex = findLastMessageIndex(messages, (message) => message?.stage === "scan");
    return scanIndex >= 0 ? scanIndex : 0;
  }
  if (stage === "reproducibility") {
    const reproducibilityIndex = findLastMessageIndex(
      messages,
      (message) => message?.stage === "reproducibility",
    );
    return reproducibilityIndex >= 0 ? reproducibilityIndex : messages.length;
  }
  if (stage === "metrics") {
    const metricIndex = findLastMessageIndex(
      messages,
      (message) => message?.stage === "metrics" || message?.stage === "summary",
    );
    return metricIndex >= 0 ? metricIndex : messages.length;
  }
  return messages.length;
}

export function agentTimelineItems(messages = [], visibleStages = [], deps = {}) {
  const definitions = deps.stageDefinitions || agentTimelineStageDefinitions();
  const snapshotsByTrigger = deps.snapshotsByTrigger || new Map();
  const visibleStageSet = new Set(visibleStages || []);
  const insertions = definitions
    .filter(({ stage }) => visibleStageSet.has(stage))
    .map((definition) => ({
      ...definition,
      insertionIndex: agentTimelineInsertionIndex(messages, definition.stage, deps),
    }))
    .sort((left, right) => (
      left.insertionIndex - right.insertionIndex ||
      left.order - right.order
    ));
  const insertionsByIndex = new Map();
  for (const insertion of insertions) {
    const stageInsertions = insertionsByIndex.get(insertion.insertionIndex) || [];
    stageInsertions.push(insertion);
    insertionsByIndex.set(insertion.insertionIndex, stageInsertions);
  }

  const items = [];
  let messageBucket = [];
  const flushMessages = () => {
    if (!messageBucket.length) return;
    items.push({ type: "messages", messages: messageBucket });
    messageBucket = [];
  };

  for (let index = 0; index <= messages.length; index += 1) {
    const stageInsertions = insertionsByIndex.get(index) || [];
    if (stageInsertions.length) {
      flushMessages();
      for (const insertion of stageInsertions) {
        items.push({ type: "stage", stage: insertion.stage, sectionId: insertion.sectionId });
      }
    }
    if (index < messages.length) {
      const message = messages[index];
      const messageId = message?.id ? String(message.id) : "";
      const snapshot = messageId ? snapshotsByTrigger.get(messageId) : null;
      if (snapshot) {
        // Flush so the frozen section lands between the previous bucket and
        // the rerun user message that triggered it.
        flushMessages();
        items.push({ type: "frozen", snapshot });
      }
      messageBucket.push(message);
    }
  }
  flushMessages();
  return items;
}
