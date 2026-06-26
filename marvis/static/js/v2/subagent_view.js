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
    return "工具待定";
  }
  return `${ref.plugin}.${ref.tool}`;
}

function subAgentStatusLabel(status) {
  return {
    pending: "待执行",
    running: "运行中",
    done: "已完成",
    failed: "失败",
    cancelled: "已取消",
  }[status] || String(status || "未知");
}

export function subAgentStatusBadge(status) {
  const normalized = classToken(status);
  return `<span class="subagent-status subagent-status-${normalized}">${escapeHtml(subAgentStatusLabel(status))}</span>`;
}

export function subAgentRowHtml(subAgent) {
  const tools = (subAgent?.granted_tools || [])
    .map((ref) => escapeHtml(toolLabel(ref)))
    .join(", ");
  const resultRef = subAgent?.result_ref
    ? `<button type="button" data-artifact="${escapeHtml(subAgent.result_ref)}">查看结果</button>`
    : "";
  return `<section class="subagent-row subagent-${classToken(subAgent?.status)}">
    <span class="subagent-scope">${escapeHtml(subAgent?.scope || "子 Agent")}</span>
    ${subAgentStatusBadge(subAgent?.status)}
    <span class="subagent-grants">${tools}</span>
    ${resultRef}
  </section>`;
}

export function subAgentListHtml(plan) {
  const subAgents = plan?.sub_agents || [];
  if (!subAgents.length) {
    return '<div class="v2-empty" data-v2-empty="subagents">暂无子 Agent</div>';
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
