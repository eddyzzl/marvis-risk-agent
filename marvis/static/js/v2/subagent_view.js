import { escapeHtml } from "../ui-utils.js";
import {
  getPlan,
  onPlanChange,
} from "./state_v2.js";

function classToken(value, fallback = "unknown") {
  return String(value || fallback)
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "") || fallback;
}

function toolLabel(ref) {
  if (!ref?.plugin || !ref?.tool) {
    return "tool pending";
  }
  return `${ref.plugin}.${ref.tool}`;
}

export function subAgentStatusBadge(status) {
  const normalized = classToken(status);
  return `<span class="subagent-status subagent-status-${normalized}">${escapeHtml(status || "unknown")}</span>`;
}

export function subAgentRowHtml(subAgent) {
  const tools = (subAgent?.granted_tools || [])
    .map((ref) => escapeHtml(toolLabel(ref)))
    .join(", ");
  const resultRef = subAgent?.result_ref
    ? `<button type="button" data-artifact="${escapeHtml(subAgent.result_ref)}">Open result</button>`
    : "";
  return `<section class="subagent-row subagent-${classToken(subAgent?.status)}">
    <span class="subagent-scope">${escapeHtml(subAgent?.scope || "sub agent")}</span>
    ${subAgentStatusBadge(subAgent?.status)}
    <span class="subagent-grants">${tools}</span>
    ${resultRef}
  </section>`;
}

export function subAgentListHtml(plan) {
  const subAgents = plan?.sub_agents || [];
  if (!subAgents.length) {
    return '<div class="v2-empty" data-v2-empty="subagents">No sub agents</div>';
  }
  return `<section class="subagent-list">${subAgents.map(subAgentRowHtml).join("")}</section>`;
}

export function renderSubAgentView(container) {
  if (!container) {
    throw new Error("renderSubAgentView requires a container");
  }
  if (container.dataset) {
    container.dataset.v2SubAgentView = "true";
  }
  const render = (plan) => {
    container.innerHTML = subAgentListHtml(plan);
  };
  render(getPlan());
  return onPlanChange((plan) => render(plan));
}
