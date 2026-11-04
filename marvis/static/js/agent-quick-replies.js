// Quick replies (2026-10-14 redesign, D11): glass chips above the composer that
// turn the latest Agent turn's structured next steps into one click. A chip
// sends exactly the command the user could have typed, through the same send
// path, so the audit trail and backend routing do not change.
//
// Only structured metadata is read (never the free-text body):
// - quick_replies[]                    -> {label, command} declared by the backend
//                                         (marvis/agent/quick_replies.py)
// - error_diagnostic.recovery_actions[] -> {label, command}
// - awaiting_next_stage                -> "继续" for V1 hand-offs stored before
//                                         the backend declared quick_replies
// - inspector_actions[]                -> {label, item}: opens that Inspector
//                                         panel; display only, sends nothing
import { escapeHtml } from "./ui-utils.js";

const MAX_QUICK_REPLIES = 4;

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function cleanText(value) {
  return value === undefined || value === null ? "" : String(value).trim();
}

function latestTimelineMessage(messages) {
  if (!Array.isArray(messages)) return null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!isRecord(message)) continue;
    if (message?.metadata?.display_in_timeline === false) continue;
    return message;
  }
  return null;
}

export function agentQuickReplies(messages = []) {
  const message = latestTimelineMessage(messages);
  // A newer user turn means the Agent owes the next move.
  if (!message || message.role !== "assistant") return [];
  const metadata = isRecord(message.metadata) ? message.metadata : {};
  if (metadata.streaming === true || metadata.optimistic === true) return [];
  const replies = [];
  const add = (label, command, kind, consequence = "") => {
    const text = cleanText(command);
    if (!text || replies.some((reply) => reply.command === text)) return;
    const reply = { label: cleanText(label) || text, command: text, kind };
    // B-line: a recovery option's stated consequence rides along as a tooltip.
    if (cleanText(consequence)) reply.consequence = cleanText(consequence);
    replies.push(reply);
  };
  if (Array.isArray(metadata.quick_replies)) {
    metadata.quick_replies.filter(isRecord).forEach((reply) => {
      add(reply.label, reply.command, "reply");
    });
  }
  const diagnostic = isRecord(metadata.error_diagnostic) ? metadata.error_diagnostic : null;
  if (diagnostic && Array.isArray(diagnostic.recovery_actions)) {
    diagnostic.recovery_actions.filter(isRecord).forEach((action) => {
      add(action.label, action.command, "recovery", action.consequence);
    });
  }
  if (!replies.length && cleanText(metadata.awaiting_next_stage)) add("继续", "继续", "advance");
  // V3 D-7: an action that only opens an Inspector panel (no command, no send).
  if (Array.isArray(metadata.inspector_actions)) {
    metadata.inspector_actions.filter(isRecord).forEach((action) => {
      const item = cleanText(action.item);
      if (!item || replies.some((reply) => reply.item === item)) return;
      replies.push({ label: cleanText(action.label) || item, command: "", kind: "inspector", item });
    });
  }
  return replies.slice(0, MAX_QUICK_REPLIES);
}

// D-10: a chip sends its Chinese command (routing stays unchanged), so the
// user's bubble would show Chinese in the English UI. When a user turn is
// exactly a command some earlier Agent turn offered as a chip, the bubble
// shows that chip's label instead. Display only; the stored text is untouched.
export function quickReplyLabelsByCommand(messages = []) {
  const labels = new Map([["继续", "继续"]]);
  if (!Array.isArray(messages)) return labels;
  for (const message of messages) {
    if (!isRecord(message) || message.role === "user") continue;
    const metadata = isRecord(message.metadata) ? message.metadata : {};
    const offered = [
      ...(Array.isArray(metadata.quick_replies) ? metadata.quick_replies : []),
      ...(isRecord(metadata.error_diagnostic) && Array.isArray(metadata.error_diagnostic.recovery_actions)
        ? metadata.error_diagnostic.recovery_actions
        : []),
    ];
    for (const reply of offered.filter(isRecord)) {
      const command = cleanText(reply.command);
      if (command) labels.set(command, cleanText(reply.label) || command);
    }
  }
  return labels;
}

export function quickReplyBubbleText(message, messages = [], translate = (text) => text) {
  const content = String(message?.content ?? "");
  if (message?.role !== "user") return content;
  const label = quickReplyLabelsByCommand(messages).get(content.trim());
  return label === undefined ? content : translate(label);
}

export function agentQuickRepliesHtml(replies = []) {
  return replies.map((reply) => (reply.kind === "inspector" ? [
    `<button type="button" class="agent-quick-reply ui-glass" data-agent-inspector-item="${escapeHtml(reply.item)}"`,
    ' data-quick-reply-kind="inspector">',
    escapeHtml(reply.label),
    "</button>",
  ] : [
    `<button type="button" class="agent-quick-reply ui-glass" data-agent-quick-reply="${escapeHtml(reply.command)}"`,
    ` data-quick-reply-kind="${escapeHtml(reply.kind || "")}"`,
    reply.consequence ? ` title="${escapeHtml(reply.consequence)}"` : "",
    ">",
    escapeHtml(reply.label),
    "</button>",
  ]).join("")).join("");
}
