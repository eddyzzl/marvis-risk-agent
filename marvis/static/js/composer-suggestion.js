// V3 UI-6 (2026-10-23 user request): the composer's grey suggestion can be
// taken with the right-arrow key while the input is empty. On the home page
// the suggestion is the placeholder example; in a task it is the reply the
// Agent suggested for this turn (message metadata.suggested_reply). Taking it
// only fills the input; nothing is sent.

export const SUGGESTED_REPLY_MAX = 40;

export function suggestionFromPlaceholder(text) {
  return String(text || "").replace(/^(?:例如[:：]|e\.g\.)\s*/i, "").trim();
}

// The latest message's suggestion, only when the Agent spoke last.
export function latestSuggestedReply(messages) {
  const list = Array.isArray(messages) ? messages : [];
  const last = list[list.length - 1];
  if (!last || last.role === "user") return "";
  const text = String(last.metadata?.suggested_reply || "").trim();
  if (!text || text.length > SUGGESTED_REPLY_MAX) return "";
  return text;
}

export function bindSuggestionAccept(input, getSuggestion) {
  if (!input?.addEventListener) return;
  input.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" || event.isComposing) return;
    if (event.shiftKey || event.altKey || event.metaKey || event.ctrlKey) return;
    if (input.value !== "") return;
    const text = String(getSuggestion?.() || "").trim();
    if (!text) return;
    event.preventDefault();
    input.value = text;
    input.setSelectionRange?.(text.length, text.length);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
