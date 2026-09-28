import { escapeHtml as esc } from "./ui-utils.js";
const roles = { maker: "配置人员", checker: "复核人员", admin: "管理员" };
export const statuses = {
  pending: "待审批",
  pending_checker: "待独立复核",
  awaiting_admin: "待管理员审批",
  approved: "已审批 · 待激活",
  promoted: "已激活",
  expired: "已过期",
  rejected: "已拒绝",
};
export const jsonDetails = (label, data) =>
  `<details class="production-json"><summary>${esc(label)}</summary><pre>${esc(JSON.stringify(data, null, 2))}</pre></details>`;
export const input = (name, label, type = "text", extra = "", value = "") =>
  `<label>${esc(label)}<input name="${esc(name)}" type="${type}" value="${esc(value)}" ${extra}></label>`;
export const select = (name, label, items, extra = "") =>
  `<label>${esc(label)}<select name="${name}" ${extra}><option value="">请选择</option>${items.map((i) => `<option value="${esc(i.id)}">${esc(i.label || i.id)}</option>`).join("")}</select></label>`;
export const button = (action, label, id = "", extra = "") =>
  `<button type="button" class="button compact secondary" data-production-action="${action}" data-production-id="${esc(id)}" ${extra}>${esc(label)}</button>`;
const submit = (label) =>
  `<button type="submit" class="button compact primary">${esc(label)}</button>`;
export function shellHtml() {
  return `<p class="production-boundary">当前环境为本地参考决策服务。机构系统适配和业务 SLO 尚未声明；冻结包、安装探针与本地请求回执分别记录，不代表已连接真实信贷生产系统。</p><div data-production-identity></div><p class="production-error" data-production-error role="alert"></p><div data-production-heads></div><div class="production-actions">${button("build", "构建冻结包")}${button("refresh", "刷新状态")}</div><div data-production-editor></div><h3>冻结方案</h3><p class="production-note">目录记录只用于发现。打开详情后重新核对冻结包内容。</p><div data-production-packages></div><h3>发布申请与审批</h3><div data-production-requests></div><h3>安装记录</h3><p class="production-note">安装记录不等于当前健康状态；打开后执行回执读回校验。</p><div data-production-installations></div><div data-production-detail></div>`;
}
export const identityHtml = (p) =>
  p
    ? `<p class="production-identity">${esc(p.display_name)} · ${roles[p.role] || esc(p.role)} · 当前工作区会话</p>`
    : `<p class="production-note">请先绑定工作区会话身份。配置、审批、安装和激活分别按角色授权。</p>${button("identity", "绑定会话身份")}`;
export function headsHtml(heads, environment, role) {
  return `<div class="production-head-grid">${heads
    .map((h, index) => {
      const slot = index ? "shadow" : "production",
        serving = h.state === "serving",
        deployment =
          slot === "production" ? environment?.active : environment?.shadow;
      return `<section class="production-card"><h3>${slot === "production" ? "本地正式服务" : "本地影子服务"}</h3><p class="production-state">${serving ? "正在服务" : "未提供服务"}</p><p class="production-note">${serving ? `治理版本 ${esc(h.revision)} · 冻结包 ${esc(h.package_hash.slice(0, 12))}` : esc(h.next_action || h.error_code || "尚未激活")}</p>${serving ? button("probe", "发起单笔验证", slot) : ""}${serving && slot === "production" && role === "admin" ? (deployment?.predecessor_deployment_id ? button("rollback", "回滚到上一版本", h.deployment_id) : '<p class="production-note">首个版本没有可恢复的前驱版本。</p>') : ""}${jsonDetails("查看当前治理指针", h)}</section>`;
    })
    .join("")}</div>`;
}
export const packagesHtml = (rows) =>
  rows.length
    ? rows
        .map(
          (p) =>
            `<button class="production-row" type="button" data-production-action="package" data-production-id="${esc(p.package_hash)}"><strong>${esc(p.strategy_id)} · v${esc(p.strategy_version)}</strong><span>${esc(p.decision_node)} · ${esc(p.package_hash.slice(0, 16))}</span></button>`,
        )
        .join("")
    : '<p class="production-note">尚无已登记冻结包。</p>';
export const requestsHtml = (rows) =>
  rows.length
    ? rows
        .map(
          (r) =>
            `<button class="production-row" type="button" data-production-action="request" data-production-id="${esc(r.id)}"><strong>${esc(r.strategy_id)} · v${r.strategy_version} · ${r.deployment_slot === "shadow" ? "影子" : "正式"}</strong><span>${esc(statuses[r.status] || r.status)} · ${esc(r.id)}</span></button>`,
        )
        .join("")
    : '<p class="production-note">尚无发布申请。</p>';
export const installationsHtml = (rows) =>
  rows.length
    ? rows
        .map(
          (r) =>
            `<button class="production-row" type="button" data-production-action="installation" data-production-id="${esc(r.promotion_request_id)}"><strong>${esc(r.package_hash.slice(0, 16))} · ${r.deployment_slot === "shadow" ? "影子" : "正式"}</strong><span>安装于 ${esc(r.installed_at)} · 待读回验证</span></button>`,
        )
        .join("")
    : '<p class="production-note">尚无安装记录。</p>';
export function packageHtml(p, role) {
  const c = p.manifest.configuration;
  return `<section><h3>冻结包详情</h3><p class="production-note">本次读取已校验包内容。审批与激活仍需独立完成。</p>${jsonDetails("字段与冻结内容", p)}${
    role === "maker"
      ? `<form data-production-form="promotion"><h4>提请发布</h4><p class="production-note">${esc(c.strategy_id)} · v${c.strategy_version} · 本地参考环境</p><div class="production-grid">${select(
          "deployment_slot",
          "服务位置",
          [
            { id: "shadow", label: "本地影子服务" },
            { id: "production", label: "本地正式服务" },
          ],
          "required",
        )}${input("expires_in_seconds", "审批有效期（秒）", "number", 'required min="1" max="86400" step="1"')}${input("reason", "发布理由", "text", 'required maxlength="4000"')}</div>${submit("提交发布申请")}</form>`
      : ""
  }</section>`;
}
export function requestHtml(r, principal, installation = null) {
  const role = principal.role,
    stage = role === "admin" ? "admin" : "checker",
    already = r.approvals.some((a) => a.stage === stage),
    independent = r.maker_principal_id !== principal.id;
  return `<section><h3>发布申请 · ${esc(statuses[r.status] || r.status)}</h3><p class="production-note">${esc(r.strategy_id)} v${r.strategy_version} · ${r.deployment_slot === "shadow" ? "影子" : "正式"} · 到期 ${esc(r.expires_at)}</p><p>${esc(r.reason)}</p><p class="production-note">已完成审批：${r.approvals.map((a) => roles[a.role]).join("、") || "暂无"}</p>${jsonDetails("冻结绑定与审批身份", r)}${independent && !already && ((role === "checker" && r.status === "pending_checker") || (role === "admin" && r.status === "awaiting_admin")) ? `<form data-production-form="approve">${input("reason", "本次复核理由", "text", 'required maxlength="4000"')}${submit("同意本次发布")}</form>` : ""}${role === "admin" && r.status === "approved" ? (installation ? `<p class="production-note">已读回核对安装回执；激活将切换相应的治理指针。</p>${jsonDetails("安装探针与证据", installation)}<form data-production-form="activate">${input("reason", "激活理由", "text", 'required maxlength="4000"')}${submit("激活已安装方案")}</form>` : button("prepare-install", "准备安装探针", r.id)) : ""}</section>`;
}
export function featuresFormHtml(
  kind,
  schema,
  { slot = "", requestId = "" } = {},
) {
  return `<form data-production-form="${kind}"><h3>${kind === "install" ? "安装并验证探针" : "单笔本地决策验证"}</h3><p class="production-note">按冻结包的原始字段合同输入测试数据。提交会生成持久回执；空值仅在合同允许时使用。</p>${kind === "decision" ? input("request_id", "请求唯一标识（相同标识用于幂等重试）", "text", 'required pattern="[A-Za-z0-9_.:-]{1,128}"', requestId) : ""}<div class="production-grid">${schema
    .map(
      (f, i) =>
        `<div data-production-feature="${esc(f.name)}">${
          f.type === "boolean"
            ? select(
                `feature_${i}`,
                `${f.name} · boolean`,
                [
                  { id: "true", label: "true" },
                  { id: "false", label: "false" },
                ],
                f.nullable ? "" : "required",
              )
            : input(
                `feature_${i}`,
                `${f.name} · ${f.type}`,
                f.type === "number" || f.type === "integer" ? "number" : "text",
                `${f.nullable ? "" : "required"} ${f.type === "number" ? 'step="any"' : f.type === "integer" ? 'step="1"' : ""}`,
              )
        }${f.nullable ? "<small>允许空值（null）</small>" : ""}</div>`,
    )
    .join(
      "",
    )}</div>${submit(kind === "install" ? "安装并运行探针" : "提交本地决策请求")}</form>`;
}
export function rollbackHtml(head) {
  return `<form data-production-form="rollback"><h3>回滚本地正式服务</h3><p class="production-note">将当前部署 ${esc(head.deployment_id)} 恢复到其已验证前驱。若治理指针已变化，平台拒绝此旧请求。</p>${input("reason", "回滚理由", "text", 'required maxlength="4000"')}${submit("确认回滚到上一版本")}</form>`;
}
export function decisionHtml(result) {
  return `<h3>本地决策回执</h3>${result.status === "fallback" ? `<p class="production-error">执行不可用，已按冻结合同使用失败动作。${esc(result.error_code || "")} · ${esc(result.next_action || "")}</p>` : ""}<p>动作：${esc(result.action?.type || result.action || "未知")} · 分数：${result.score == null ? "未知" : esc(result.score)}</p><p class="production-note">此结果来自本地参考执行器。回执反映本次请求，不代表机构实际放款或收益。</p>${jsonDetails("查看本次请求、治理绑定与完整结果", result)}`;
}
