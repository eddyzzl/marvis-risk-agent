import { escapeHtml as esc } from "./ui-utils.js";
import { field, select } from "./historical-replay-form.js";
const value = (v) =>
  v == null
    ? "未知"
    : esc(
        typeof v === "number"
          ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 }).format(
              v,
            )
          : v,
      );
const fact = (label, v) =>
  `<div><dt>${esc(label)}</dt><dd>${value(v)}</dd></div>`;
const details = (title, data) =>
  `<details class="history-json"><summary>${esc(title)}</summary><pre>${esc(JSON.stringify(data, null, 2))}</pre></details>`;
const boundary =
  '<p class="history-boundary">历史回放只比较给定人群和冻结方案。尚未证明方案当时已上线，收益增量的因果关系也未识别；结果不会自动发布或调整策略。</p>';
const state = (status) =>
  ({
    passed: "约束满足",
    failed: "约束未满足",
    unknown: "证据不足",
    not_configured: "未配置约束",
    not_declared: "未配置约束",
    estimated: "固定条件估算",
    measured: "描述性测量",
    reconciled: "完成对账",
    partial_comparability: "部分可比",
    partial: "部分可测",
    insufficient_evidence: "证据不足",
  })[status] ||
  status ||
  "未知";
export function proposalHtml(proposal, kind = "replay") {
  return `<section class="history-proposal"><h4>请复核本次提案</h4>${boundary}<dl class="history-facts">${fact("人群记录数", proposal.population_count)}${fact("人群指纹", proposal.population_hash)}${fact("提案指纹", proposal.proposal_hash)}${fact("数据来源", proposal.contract.source_ref)}</dl>${temporalPopulationHtml(proposal.temporal_population)}${details("查看完整字段、时点与方案合同", proposal.contract)}<button type="button" class="button compact primary" data-history-action="plan" data-history-kind="${kind}">提交到执行计划</button></section>`;
}
export function receiptListHtml(records) {
  return records.length
    ? records
        .map(
          (r) =>
            `<button type="button" class="button compact secondary history-receipt" data-history-action="receipt" data-history-id="${esc(r.artifact_id)}" ${r.status !== "available" ? "disabled" : ""}>${r.kind === "decision_twin_batch_replay" ? "历史回放" : "现金流对账"} · ${esc(r.artifact_id.slice(0, 12))} · ${r.status === "available" ? "可查看" : "完整性校验失败"}</button>`,
        )
        .join("")
    : '<p class="history-note">尚无已完成的回放或对账回执。</p>';
}
export function receiptHtml(receipt) {
  const p = receipt.payload,
    replay = p.schema_version === "decision_twin.batch_receipt.v2";
  if (!replay && p.schema_version !== "decision_twin.batch_reconciliation.v2")
    return '<p class="history-error">此回执版本不受支持，请更新后查看。</p>';
  let body;
  if (replay) {
    body = `<dl class="history-facts">${fact("目标人群", p.source?.population)}${fact("逐笔记录数", p.source?.population_count)}${fact("来源快照", p.source?.dataset_id)}${fact("人群指纹", p.source?.population_hash)}${fact("历史动作", p.observed_actions?.status === "imported" ? "已导入外部历史记录（未认证平台执行）" : "未导入，不能称为实际历史冠军方案")}${fact("特征时点来源", "导入快照已认证；可用时点由来源声明")}</dl><div class="history-table-scroll"><table class="history-table"><thead><tr><th>方案</th><th>批准 / 全人群</th><th>批准率</th><th>相对基线变化</th><th>决策变化数</th><th>约束</th></tr></thead><tbody>${p.scenarios.map((s) => `<tr><th>${esc(s.name)}<small>${esc(s.package_hash.slice(0, 12))}</small></th><td>${value(s.metrics.approval_count)} / ${value(s.metrics.count)}</td><td>${value(s.metrics.approval_rate)}</td><td>${value(s.comparison.approval_rate_delta)}</td><td>${value(s.comparison.decision_change_count)}</td><td>${esc(state(s.constraints.status))}</td></tr>`).join("")}</tbody></table></div>${p.scenarios.map((s) => `<details class="history-scenario"><summary>${esc(s.name)} · 指标、假设和未知项</summary><dl class="history-facts">${fact("风险暴露金额", s.metrics.economics.value?.ead)}${fact("预期损失", s.metrics.economics.value?.expected_loss)}${fact("估算利润", s.metrics.economics.value?.profit)}${fact("币种", s.metrics.economics.currency)}${fact("收益口径", s.metrics.economics.status === "unknown" ? s.metrics.economics.reason : "固定条件下批准人群的估算，不是实际现金流")}${fact("客群批准率差异", s.metrics.protected_groups.value)}${fact("运营处理量", s.metrics.operations_capacity.value)}${fact("时间稳定性", s.metrics.stability?.status === "unknown" ? s.metrics.stability.reason : state(s.metrics.stability?.status))}</dl>${temporalResultHtml(s.metrics.stability)}${details("约束核对", s.constraints)}${details("全部指标与比较", { metrics: s.metrics, comparison: s.comparison })}</details>`).join("")}`;
  } else
    body = `<dl class="history-facts">${fact("对账状态", state(p.status))}${fact("历史批准人群", p.approved_denominator)}${fact("可比记录数", p.comparable_denominator)}${fact("币种", p.currency)}${fact("实际损失", p.actual_loss)}${fact("实际利润", p.actual_profit)}${fact("可比损失差", p.comparable_loss_delta)}${fact("可比利润差", p.comparable_profit_delta)}</dl><p class="history-note">实际结果来自外部历史导入，只对成熟且可比的已观察批准人群核对，不认定未批准人群的结果。</p>`;
  return `<section data-history-receipt="${esc(receipt.artifact_id)}"><h4>${replay ? "历史回放结果" : "历史现金流对账结果"}</h4>${boundary}${body}<div class="history-actions">${["json", "xlsx", "docx"].map((format) => `<button class="button compact secondary" type="button" data-history-action="download" data-history-format="${format}" data-history-id="${esc(receipt.artifact_id)}">下载 ${format.toUpperCase()}</button>`).join("")}${replay ? `<button class="button compact secondary" type="button" data-history-action="reconcile" data-history-id="${esc(receipt.artifact_id)}">导入成熟现金流对账</button>` : ""}</div>${details("冻结回执与来源合同", { artifact_id: receipt.artifact_id, kind: receipt.kind, contract_hash: p.contract_hash, contract: p.contract })}</section>`;
}
export function reconciliationFormHtml(datasets, artifactId) {
  return `<form data-history-form="reconciliation" data-history-id="${esc(artifactId)}"><h4>导入成熟现金流对账</h4><p class="history-note">绑定上方回放回执。申请编号用于对齐历史批准人群；完整成熟窗口、损失与利润均需明确来源。</p><div class="history-grid">${select(
    "dataset_id",
    "现金流数据快照",
    datasets.map((d) => ({
      id: d.id,
      label: `${d.source_name || d.id} · ${d.row_count} 行`,
    })),
    "required",
  )}${select("record_id_col", "申请编号列", [], "required")}${select("observed_at_col", "观察时间列", [], "required")}${select("actual_loss_col", "实际损失列", [], "required")}${select("actual_profit_col", "实际利润列", [], "required")}${field("currency", "币种", "text", 'required pattern="[A-Z]{3}" maxlength="3"')}${field("maturity_days", "成熟天数", "number", 'required min="1" max="36500" step="1"')}${field("maturity_source_ref", "成熟口径来源", "text", "required")}${field("source_ref", "现金流数据来源", "text", "required")}${field("reconciled_at", "本次对账截止时间（UTC）", "datetime-local", 'required step="1"')}</div><p class="history-note" data-history-source></p><button type="submit" class="button compact primary">校验并生成对账提案</button></form>`;
}
export function replayGateHtml(plan) {
  const step = plan?.steps?.find(
    (s) =>
      s.status === "awaiting_confirm" && s.tool_ref?.plugin === "decision_twin",
  );
  if (!step) return "";
  return `<section class="history-gate" data-history-step="${esc(step.id)}"><h4>人工复核 · ${esc(step.title)}</h4><p class="history-note">请核对计划中的人群、时点、冻结包和未知边界。授权只适用于当前计划快照；平台会记录本次身份和复核理由。</p>${details("查看本次待执行合同", step.inputs)}${field("decision_reason", "复核理由", "text", 'required maxlength="4000"')}<div class="history-actions"><button type="button" class="button compact primary" data-history-action="approve">授权本次执行</button><button type="button" class="button compact secondary" data-history-action="reject">拒绝本次执行</button></div></section>`;
}

export function temporalPopulationHtml(population) {
  if (!population) return "";
  return `<h4>时间窗口成员核对</h4><p class="history-note">来源 ${value(population.source_count)} 条 · 窗口外排除 ${value(population.excluded_count)} 条</p><div class="history-table-scroll"><table class="history-table"><thead><tr><th>窗口</th><th>起点（含）</th><th>终点（不含）</th><th>记录数</th><th>成员指纹</th></tr></thead><tbody>${[population.reference, ...population.comparisons].map((w) => `<tr><th>${esc(w.name)}</th><td>${esc(w.start)}</td><td>${esc(w.end)}</td><td>${value(w.sample_count)}</td><td>${esc(w.members_hash)}</td></tr>`).join("")}</tbody></table></div>`;
}
export function temporalResultHtml(stability) {
  if (stability?.schema_version !== "decision_twin.temporal_stability.v1")
    return "";
  const labels = {
    score_psi: "分数 PSI",
    action_psi: "动作 PSI",
    absolute_approval_rate_delta: "批准率绝对变化",
  };
  return `<h4>时间稳定性 · ${esc(state(stability.verdict))}</h4><p class="history-note">仅描述回放分布的变化。分数分箱只由参考窗拟合；缺乏证据的检查保持未知。</p><div class="history-table-scroll"><table class="history-table"><thead><tr><th>对照窗口</th><th>检查项</th><th>测量值</th><th>人工阈值</th><th>判断</th></tr></thead><tbody>${stability.comparisons.map((w) => w.checks.map((c) => `<tr><th>${esc(w.window.name)}</th><td>${esc(labels[c.metric] || c.metric)}</td><td>${value(c.value)}</td><td>${value(c.threshold)}</td><td>${c.passed == null ? `未知：${esc(c.reason || c.status)}` : c.passed ? "满足" : "未满足"}</td></tr>`).join("")).join("")}</tbody></table></div>${details("参考窗分箱与人工稳定性合同", { reference: stability.reference, policy: stability.policy })}`;
}
