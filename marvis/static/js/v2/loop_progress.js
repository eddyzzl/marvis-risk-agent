import { escapeHtml } from "../ui-utils.js";
import {
  getPlan,
  getLoopEvents,
  onLoopEventsChange,
  onPlanChange,
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
    return `重新规划：${reason}`;
  }
  if (event.type === "explore_segment") {
    return `探索分支：${reason}`;
  }
  if (event.type === "no_progress") {
    return `暂无进展：${reason}`;
  }
  return `${event.type || "事件"}：${reason}`;
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
    return '<div class="v2-empty" data-v2-empty="loop-events">暂无循环事件</div>';
  }
  return `<section class="loop-events">${sortedEvents(events).map(loopEventHtml).join("")}</section>`;
}

function currentLoopEvents() {
  const planEvents = getPlan()?.loop_events;
  if (Array.isArray(planEvents)) {
    return planEvents;
  }
  return getLoopEvents();
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
  render(currentLoopEvents());
  const unsubLoopEvents = onLoopEventsChange(() => render(currentLoopEvents()));
  const unsubPlan = onPlanChange(() => render(currentLoopEvents()));
  return () => {
    unsubLoopEvents();
    unsubPlan();
  };
}
