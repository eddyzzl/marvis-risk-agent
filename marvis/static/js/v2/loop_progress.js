import { escapeHtml } from "../ui-utils.js";
import {
  getLoopEvents,
  onLoopEventsChange,
} from "./state_v2.js";

function classToken(value, fallback = "event") {
  return String(value || fallback)
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "") || fallback;
}

function sortedEvents(events) {
  return [...(events || [])].sort((left, right) => String(left.at || "").localeCompare(String(right.at || "")));
}

function eventLabel(event) {
  const reason = event.reason || event.detail || "";
  if (event.type === "replan") {
    return `Replan: ${reason}`;
  }
  if (event.type === "explore_segment") {
    return `Explore segment: ${reason}`;
  }
  if (event.type === "no_progress") {
    return `No progress: ${reason}`;
  }
  return `${event.type || "event"}: ${reason}`;
}

export function loopEventHtml(event) {
  const typeClass = classToken(event?.type);
  const attention = event?.type === "no_progress" ? " attention" : "";
  return `<div class="loop-evt ${typeClass}${attention}">
    <span>${escapeHtml(eventLabel(event || {}))}</span>
    <time>${escapeHtml(event?.at || "")}</time>
  </div>`;
}

export function loopEventsHtml(events = []) {
  if (!events.length) {
    return '<div class="v2-empty" data-v2-empty="loop-events">No loop events</div>';
  }
  return `<section class="loop-events">${sortedEvents(events).map(loopEventHtml).join("")}</section>`;
}

export function renderLoopEvents(container) {
  if (!container) {
    throw new Error("renderLoopEvents requires a container");
  }
  if (container.dataset) {
    container.dataset.v2LoopEvents = "true";
  }
  const render = (events) => {
    container.innerHTML = loopEventsHtml(events);
  };
  render(getLoopEvents());
  return onLoopEventsChange((events) => render(events));
}
