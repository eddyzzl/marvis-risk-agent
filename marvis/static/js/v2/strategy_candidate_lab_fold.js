/** Candidate Lab fold: the collapsed summary card and per-task open memory.
 *
 * 2026-10-14 redesign (batch 4): the strategy workbench renders above the
 * conversation, so it folds into one summary card and the first screen shows
 * the conversation. Every fact comes from the structured Candidate Lab payload
 * the controller already holds; no message text is read. Pure module: no DOM
 * access, so node harnesses import it directly.
 */

import { escapeHtml } from "../ui-utils.js";
import { collectionItems, isRecord, nonEmptyText } from "./strategy_candidate_lab_contracts.js";
import {
  COLLECTION_DEFINITIONS,
  WORKFLOW_STAGE_STATUS_LABELS,
} from "./strategy_candidate_lab_presenters.js";

export const STRATEGY_CANDIDATE_LAB_EXPANDED_STORAGE_PREFIX = "marvis_strategy_candidate_lab_expanded_v1:";

const BLOCKED_SUMMARY_COPY = Object.freeze({
  active_plan: "策略计划执行中，暂不能启动新分析",
  open_gate: "有待处理的确认，暂不能启动新分析",
  loading: "正在核验候选证据",
  submitting: "分析请求提交中",
});

const LATEST_EVIDENCE_LABELS = Object.freeze([
  ["pool_stability", "策略池稳定性"],
  ["pool_impact", "策略池影响"],
  ["impact_cube", "综合影响测算"],
]);

const POOL_VALIDATION_LABELS = Object.freeze([
  ["validation", "验证样本"],
  ["oot", "时间外样本（OOT）"],
]);

function countOf(collection) {
  if (Number.isInteger(collection?.total) && collection.total >= 0) return collection.total;
  return collectionItems(collection).length;
}

function createdAtMs(item) {
  const text = nonEmptyText(item?.artifact?.created_at || item?.created_at);
  const value = text ? Date.parse(text) : Number.NaN;
  return Number.isFinite(value) ? value : null;
}

function latestStep(payload) {
  const steps = [];
  const candidates = isRecord(payload.candidates) ? payload.candidates : {};
  for (const definition of COLLECTION_DEFINITIONS) {
    const latest = candidates[definition.key]?.latest;
    if (isRecord(latest)) steps.push([definition.title, createdAtMs(latest)]);
  }
  for (const pool of collectionItems(payload.pools)) steps.push(["策略池", createdAtMs(pool)]);
  const workflow = isRecord(payload.workflow) ? payload.workflow : {};
  const evidence = isRecord(workflow.latest_evidence) ? workflow.latest_evidence : {};
  for (const [key, label] of LATEST_EVIDENCE_LABELS) {
    if (isRecord(evidence[key])) steps.push([label, createdAtMs(evidence[key])]);
  }
  const validations = isRecord(evidence.pool_validation) ? evidence.pool_validation : {};
  for (const [key, label] of POOL_VALIDATION_LABELS) {
    if (isRecord(validations[key])) steps.push([label, createdAtMs(validations[key])]);
  }
  if (isRecord(workflow.report)) steps.push(["策略报告", createdAtMs(workflow.report)]);
  const dated = steps.filter(([, at]) => at !== null);
  if (!dated.length) return null;
  const [title, at] = dated.reduce((best, step) => (step[1] > best[1] ? step : best));
  return { title, createdAt: new Date(at).toISOString() };
}

// {state, notice, stage, completed, stageTotal, candidates, poolReady, pools,
//  rules, latest, redFlags}. state is error | loading | blocked | ready; the
// counts are null when there is no payload yet.
export function strategyCandidateLabSummaryFacts(payload, { blockedReason = "", loading = false, error = "" } = {}) {
  const value = isRecord(payload) ? payload : null;
  const reason = nonEmptyText(blockedReason);
  const state = error
    ? "error"
    : reason === "loading" || (loading && !value) ? "loading" : reason ? "blocked" : "ready";
  const notice = state === "error"
    ? "候选证据读取失败"
    : state === "ready" ? "" : BLOCKED_SUMMARY_COPY[reason || "loading"] || "暂不能启动新分析";
  const facts = {
    state,
    notice,
    stage: null,
    completed: null,
    stageTotal: null,
    candidates: null,
    poolReady: null,
    pools: null,
    rules: null,
    latest: null,
    redFlags: null,
  };
  if (!value) return facts;
  const workflow = isRecord(value.workflow) ? value.workflow : {};
  const stages = Array.isArray(workflow.stages) ? workflow.stages.filter(isRecord).slice(0, 7) : [];
  if (stages.length) {
    const statusOf = (stage) => (["complete", "stale", "missing"].includes(stage.status) ? stage.status : "missing");
    // Same rule as the stage directory: the first stage that is not complete.
    const index = stages.findIndex((stage) => statusOf(stage) !== "complete");
    facts.completed = stages.filter((stage) => statusOf(stage) === "complete").length;
    facts.stageTotal = stages.length;
    if (index >= 0) {
      const status = statusOf(stages[index]);
      facts.stage = {
        number: index + 1,
        label: nonEmptyText(stages[index].label) || `阶段 ${index + 1}`,
        status,
        statusLabel: WORKFLOW_STAGE_STATUS_LABELS[status],
      };
    }
  }
  const candidates = isRecord(value.candidates) ? value.candidates : {};
  facts.candidates = COLLECTION_DEFINITIONS.reduce((total, definition) => total + countOf(candidates[definition.key]), 0);
  facts.poolReady = countOf(value.pool_add_sources);
  facts.pools = countOf(value.pools);
  facts.rules = collectionItems(value.pools).reduce(
    (total, pool) => total + (Number.isInteger(pool.total) && pool.total >= 0
      ? pool.total
      : (Array.isArray(pool.entries) ? pool.entries.length : 0)),
    0,
  );
  facts.latest = latestStep(value);
  const redFlags = isRecord(value.evidence_drawer) ? value.evidence_drawer.red_flags : null;
  facts.redFlags = countOf(redFlags);
  return facts;
}

function pad(number) {
  return String(number).padStart(2, "0");
}

function shortTime(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function factHtml(name, label, value, tone = "") {
  return [
    `<div class="candidate-lab-summary-fact" data-fact="${name}"${tone ? ` data-tone="${tone}"` : ""}>`,
    `<dt>${escapeHtml(label)}</dt>`,
    `<dd>${escapeHtml(value)}</dd>`,
    "</div>",
  ].join("");
}

export function strategyCandidateLabSummaryHtml(facts) {
  if (!isRecord(facts)) return "";
  const notice = facts.notice
    ? `<p class="candidate-lab-summary-state" data-state="${escapeHtml(facts.state)}">${escapeHtml(facts.notice)}</p>`
    : "";
  if (facts.candidates === null) return notice;
  const rows = [];
  if (facts.stageTotal) {
    rows.push(factHtml(
      "stage",
      "当前阶段",
      facts.stage
        ? `第 ${facts.stage.number} 步 ${facts.stage.label} · ${facts.stage.statusLabel}`
        : "全部阶段已完成",
      facts.stage?.status === "stale" ? "warn" : "",
    ));
    rows.push(factHtml("completed", "已完成", `${facts.completed}/${facts.stageTotal}`));
  }
  rows.push(factHtml(
    "candidates",
    "候选",
    facts.poolReady ? `${facts.candidates} 个 · 可入池 ${facts.poolReady}` : `${facts.candidates} 个`,
  ));
  rows.push(factHtml("pools", "策略池", facts.pools ? `${facts.pools} 个 · ${facts.rules} 条规则` : "暂无规则"));
  if (facts.latest) {
    const time = shortTime(facts.latest.createdAt);
    rows.push(factHtml("latest", "最近一步", time ? `${facts.latest.title} · ${time}` : facts.latest.title));
  }
  if (facts.redFlags) rows.push(factHtml("red_flags", "风险提示", `${facts.redFlags} 条`, "warn"));
  return `${notice}<dl class="candidate-lab-summary-facts">${rows.join("")}</dl>`;
}

export function strategyCandidateLabExpandedStorageKey(taskId) {
  return `${STRATEGY_CANDIDATE_LAB_EXPANDED_STORAGE_PREFIX}${encodeURIComponent(nonEmptyText(taskId))}`;
}

// Per-viewer convenience: an unreadable or blocked storage reads as collapsed.
export function loadStrategyCandidateLabExpanded(taskId, storage) {
  if (!nonEmptyText(taskId)) return false;
  try {
    return storage?.getItem?.(strategyCandidateLabExpandedStorageKey(taskId)) === "1";
  } catch (_) {
    return false;
  }
}

export function persistStrategyCandidateLabExpanded(taskId, expanded, storage) {
  if (!nonEmptyText(taskId)) return false;
  const key = strategyCandidateLabExpandedStorageKey(taskId);
  try {
    if (expanded) storage?.setItem?.(key, "1");
    else storage?.removeItem?.(key);
    return true;
  } catch (_) {
    return false;
  }
}
