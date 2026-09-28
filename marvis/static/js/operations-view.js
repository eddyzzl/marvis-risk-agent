import { escapeHtml as esc } from "./ui-utils.js";

export const operationTime = value => value ? String(value).replace("T", " ").replace(/(?:\.\d+)?(?:Z|\+00:00)$/, " UTC") : "—";
const stamp = value => esc(operationTime(value));
const pair = (label, value) => `<div><dt>${esc(label)}</dt><dd>${esc(value ?? "—")}</dd></div>`;
const json = (label, value) => `<details class="ops-json"><summary>${esc(label)}</summary><pre>${esc(JSON.stringify(value, null, 2))}</pre></details>`;
const levelText = {green:"正常", amber:"需关注", red:"高风险", not_available:"无可用结论"};
export function levelBadge(level) {
  return `<span class="ops-badge ${["green", "amber", "red"].includes(level) ? level : "neutral"}">${esc(levelText[level] || "待评估")}</span>`;
}
export function runtimeHtml(capabilities, principal) {
  const r = capabilities?.runtime;
  return `<span>${esc(principal?.display_name || "")} · ${esc({maker:"配置人员", checker:"复核人员", admin:"管理员"}[principal?.role] || "")}</span>
    <span class="ops-runtime ${r?.alive && !r.stopping && !r.last_error_code ? "" : "ops-error"}">${!r ? "后台状态未知" : r.stopping ? "后台正在停止" : r.alive ? r.last_error_code ? "后台运行异常" : "后台运行中" : "后台未运行"}</span>
    <span>最近巡检 ${stamp(r?.last_finished_at)}</span>${r?.last_error_code ? `<span>${esc(r.last_error_code)} · 请检查服务日志并重新启动服务</span>` : ""}`;
}
export function inboxHtml(notifications) {
  if (!notifications.length) return `<p class="ops-empty">暂无已投递通知。监控完成并投递到应用后会显示在这里。</p>`;
  return `<p class="ops-note">最近 ${notifications.length} 条投递（最多 100 条）。打开证据不会自动标为已读。</p><div class="ops-inbox-list">${notifications.map(n => `<article class="ops-receipt" data-notification-id="${esc(n.notification_id)}">
    <div class="ops-section-title"><strong>${esc(n.schedule_id)}</strong>${levelBadge(n.payload?.level)}</div>
    <p class="ops-note">${esc(n.period_key)}</p><div class="ops-receipt-state"><span>已投递 ${stamp(n.delivered_at)}</span><span class="${n.read_at ? "" : "ops-unread"}">${n.read_at ? `已读 ${stamp(n.read_at)}` : "未读"}</span></div>
    ${n.read_by ? `<p class="ops-note">阅读人 ${esc(n.read_by)}</p>` : ""}
    <div class="ops-actions"><button class="secondary" data-ops-action="notification" data-id="${esc(n.notification_id)}">查看周期与证据</button>${n.read_at ? "" : `<button class="secondary" data-ops-action="ack" data-id="${esc(n.notification_id)}">标为已读</button>`}</div>
  </article>`).join("")}</div>`;
}
export function schedulesHtml(schedules, selectedId) {
  if (!schedules.length) return `<p class="ops-empty">暂无监控配置。新建监控后可查看运行周期与诊断。</p>`;
  return schedules.map(({contract:c}) => `<button class="ops-schedule ${c.schedule_id === selectedId ? "selected" : ""}" data-ops-action="schedule" data-id="${esc(c.schedule_id)}">
    <strong>${esc(c.schedule_id)}</strong><span>v${esc(c.revision)} · ${c.enabled ? "启用" : "已停用"}</span><small>${esc(c.monitoring_ref)}</small>${c.recheck_of ? "<small>独立重新评估</small>" : ""}</button>`).join("");
}
export function scheduleHtml(record, periods, canWrite) {
  const c = record.contract, b = c.monitoring_binding;
  return `<div class="ops-section-title"><h3>${esc(c.schedule_id)} <small>v${esc(c.revision)}</small></h3>${canWrite && !c.recheck_of ? `<button class="secondary" data-ops-action="edit">修改配置</button>` : ""}</div>
    ${!c.enabled ? '<p class="ops-callout">监控已停用：不会创建新的周期；历史结果保留。可通过“修改配置”发布新版本恢复。</p>' : ""}
    <dl class="ops-facts">${pair("监控类型", c.monitoring_ref)}${pair("运行状态", c.enabled ? "启用" : "停用")}${pair("任务", b?.task_id)}${pair("目标", b?.target_id)}${pair("周期长度", `${c.calendar.interval_seconds} 秒`)}${pair("数据完整覆盖至（发布者声明）", operationTime(b?.complete_through))}</dl>
    ${c.recheck_of ? `<p class="ops-note">重新评估来源 ${esc(c.recheck_of.schedule_id)} · ${esc(c.recheck_of.period_key)}。此配置保留原始关联，需要补充数据时可对其周期再次重新评估。</p>` : ""}
    <h4>最近运行周期</h4>${periods.length ? `<div class="ops-periods">${periods.map(({period:p, diagnostic:d}) => `<button class="ops-period" data-ops-action="period" data-key="${esc(p.period.key)}">
      <span>${stamp(p.period.starts_at)} → ${stamp(p.period.ends_at)}</span><strong>${esc(d.summary)}</strong><small>${esc(d.next_action?.label || "")} · ${d.retry_scheduled ? "已安排重试" : "未安排自动重试"}</small></button>`).join("")}</div>` : `<p class="ops-empty">暂无运行周期。后台按周期结束时间与启用范围调度；确认时间范围、数据声明与后台状态后刷新。</p>`}
    ${json("查看配置与内容指纹", record)}`;
}
export function periodHtml(detail, canWrite) {
  const {period:p, diagnostic:d, outcome, runs, events, notifications} = detail;
  const recheckable = ["succeeded", "terminal_failed"].includes(p.state);
  return `<section class="ops-period-detail"><h4>周期诊断</h4><p class="ops-callout">${esc(d.summary)}</p>
    <dl class="ops-facts">${pair("周期", p.period.key)}${pair("执行状态", p.state)}${pair("使用配置", `v${p.schedule_revision}`)}${pair("执行次数", p.attempt_count)}${pair("下一步", d.next_action?.label)}${pair("自动重试", d.retry_scheduled ? `已安排 · ${operationTime(p.next_attempt_at)}` : "未安排")}</dl>
    <p class="ops-note">${esc(d.code)} · 通知只供人工查看，不代表审批通过或自动调整策略。</p>
    <div class="ops-actions">${outcome ? '<button class="secondary" data-ops-action="evidence">查看已校验的监控证据</button>' : '<span class="ops-note">尚无结果证据</span>'}${canWrite && recheckable ? '<button class="secondary" data-ops-action="recheck">重新评估此周期</button>' : ""}</div>
    <div data-ops-evidence></div>
    <h4>投递处理</h4>${notifications.length ? notifications.map(n => `<p class="ops-note">${esc({pending:"等待投递", sending:"投递中", sent:"发送处理完成", exhausted:"投递重试已耗尽"}[n.status] || n.status)} · 尝试 ${esc(n.attempt_count)} / ${esc(n.max_attempts)}${n.next_attempt_at ? ` · 下次 ${stamp(n.next_attempt_at)}` : ""}</p>`).join("") : '<p class="ops-note">尚未产生通知投递记录。</p>'}
    <p class="ops-note">实际投递与已读时间以通知页的应用回执为准。</p>
    ${json(`执行记录（${runs.length}）`, runs)}${json(`事件记录（${events.length}）`, events)}${outcome ? json("结果指纹", {outcome_id:outcome.outcome_id,outcome_hash:outcome.outcome_hash,recorded_at:outcome.recorded_at}) : ""}
  </section>`;
}
export function evidenceHtml(evidence) {
  return `<section class="ops-evidence"><h4>监控证据 · 已由服务端校验</h4>
    <dl class="ops-facts">${pair("数据快照", evidence.source_dataset_id)}${pair("来源内容指纹", evidence.source_content_hash)}${pair("完整覆盖至", operationTime(evidence.declared_complete_through))}${pair("覆盖保证", evidence.coverage_assurance === "publisher_declared" ? "发布者声明，未独立认证" : evidence.coverage_assurance)}${pair("标签要求", evidence.label_mode === "required" ? "需要真实标签" : evidence.label_mode === "not_applicable" ? "不适用标签" : "未知")}${pair("标签等待（秒）", evidence.label_maturity_seconds)}</dl>
    ${evidence.tool_output ? monitoringOutputHtml(evidence.tool_output) : '<p class="ops-callout">当前无监控工具输出，不能视作零风险或正常。</p>'}
    ${json("展开完整确定性证据", evidence)}</section>`;
}

function monitoringOutputHtml(output) {
  const checks = Array.isArray(output.checks) ? output.checks : [];
  return `<div class="ops-section-title"><h4>本期指标${output.row_count == null ? "" : ` · ${esc(output.row_count)} 行`}</h4>${levelBadge(output.overall_level)}</div>
    ${output.recommendation ? `<p class="ops-note">${esc(output.recommendation)}</p>` : ""}
    ${checks.length ? `<div class="ops-table-scroll"><table class="ops-metrics"><thead><tr><th>检查</th><th>实际值</th><th>状态</th><th>说明</th></tr></thead><tbody>${checks.map(check => `<tr><td>${esc(check.label || check.metric || check.id)}</td><td>${check.value == null ? "无可用值" : esc(check.value)}</td><td>${levelBadge(check.level)}</td><td>${esc(check.message || "—")}</td></tr>`).join("")}</tbody></table></div>` : '<p class="ops-note">此工具未提供统一检查表，详见完整证据。</p>'}`;
}
