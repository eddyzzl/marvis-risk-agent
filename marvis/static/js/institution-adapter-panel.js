import { api as defaultApi } from "./api.js";
import { createContractForm } from "./contract-form.js";
import { taskUsesPlanRail } from "./v2/plan_rail_controller.js";

const key = encodeURIComponent;
const actions = {
  registry_upload: ["提交原生模型登记", "registry/uploads", ["maker"]],
  registry_transition: ["晋级已登记模型", "registry/promotions", ["admin"]],
  registry_revoke: ["撤销已登记模型", "registry/revocations", ["admin"]],
  deployment_install: ["安装已批准方案并检查健康状态", "deployments/install", ["admin"]],
  deployment_activate: ["激活已检查的方案", "deployments/activate", ["admin"]],
  deployment_rollback: ["核对当前版本并回滚", "deployments/rollback", ["admin"]],
  source_grant: ["登记外部数据授权", "sources/grants", ["admin"]],
  source_query: ["执行外部数据查询", "sources/requests", ["maker"]],
  source_revoke: ["撤销外部数据授权", "sources/grants", ["admin"]],
};
export function createInstitutionAdapterPanel({ getElement, getTask, captureView, isCurrentView, api = defaultApi, openIdentity = () => {}, isBusy = () => false }) {
  let state = null;
  const current = (owner) => owner === state && getTask()?.id === owner.taskId && isCurrentView(owner.view);
  const make = (document, tag, text) => { const element = document.createElement(tag); if (text != null) element.textContent = text; return element; };
  function button(owner, label, action, parent = owner.buttons) {
    const element = make(owner.document, "button", label); element.type = "button"; element.className = "button compact secondary";
    element.addEventListener("click", async () => {
      if (!current(owner) || owner.pending || isBusy()) return;
      owner.pending = true; element.disabled = true;
      try { await action(); }
      catch (error) { if (current(owner)) owner.status.textContent = error?.detail?.next_action || error?.detail?.code || error.message || "操作未完成，请读取原状态。"; }
      finally { owner.pending = false; element.disabled = false; }
    }); parent.append(element);
  }
  function display(owner, payload) {
    if (!current(owner)) return;
    owner.status.textContent = `状态：${payload.status || payload.state || "已登记"}。${payload.next_action || "可读取原状态核对证据。"}`;
    const details = make(owner.document, "details"), summary = make(owner.document, "summary", "查看通用协议证据回执"), pre = make(owner.document, "pre", JSON.stringify(payload, null, 2));
    pre.className = "contract-receipt"; details.append(summary, pre); owner.result.replaceChildren(details);
  }
  function render() {
    const panel = getElement(), task = getTask(); if (!panel) return;
    panel.hidden = !taskUsesPlanRail(task);
    if (panel.hidden) { state = null; panel.replaceChildren(); return; }
    if (state && current(state)) return;
    const document = panel.ownerDocument, details = make(document, "details"), body = make(document, "div"), buttons = make(document, "div"), status = make(document, "p", "尚未读取机构服务配置。");
    body.className = "risk-context-body"; buttons.className = "history-actions"; status.className = "history-note"; status.setAttribute("role", "status");
    body.append(make(document, "p", "通过固定协议登记模型、部署已批准方案或查询授权数据。目标服务地址和凭据由管理员在本机配置；网络结果不明确时，先读取原操作状态并恢复本地登记。"), buttons, status);
    const editor = make(document, "div"), result = make(document, "div"); body.append(editor, result); details.append(make(document, "summary", "机构服务接入"), body); panel.replaceChildren(details);
    const owner = state = { document, taskId: task.id, view: captureView(), buttons, status, editor, result, pending: false, edit: 0 };
    button(owner, "读取服务配置", () => load(owner));
    button(owner, "配置会话身份", openIdentity);
  }
  async function load(owner) {
    const base = `/api/tasks/${key(owner.taskId)}/institution-adapters`;
    const [capabilities, principal] = await Promise.all([api(base + "/capabilities"), api("/api/production-governance/me")]);
    if (!current(owner)) return;
    owner.capabilities = capabilities; owner.principal = principal;
    owner.editor.replaceChildren();
    if (capabilities.status === "unconfigured") { owner.status.textContent = "目前未配置目标系统。通用接入协议已安装；请由管理员按接入手册配置后再检查。"; return; }
    owner.status.textContent = "已读取固定目标服务。原生模型登记仅保管模型材料；参考部署当前支持规则方案，生产机构验收须另行完成。";
    const alias = make(owner.document, "select"); alias.setAttribute("aria-label", "目标服务");
    capabilities.endpoints.forEach((endpoint) => { const option = make(owner.document, "option", `${endpoint.alias} · ${endpoint.environment}`); option.value = endpoint.alias; alias.append(option); });
    const operation = make(owner.document, "select"); operation.setAttribute("aria-label", "机构服务操作");
    const placeholder = make(owner.document, "option", "请选择操作"); placeholder.value = ""; operation.append(placeholder);
    Object.entries(actions).filter(([, action]) => action[2].includes(principal.role)).forEach(([value, action]) => { const option = make(owner.document, "option", action[0]); option.value = value; operation.append(option); });
    const slot = make(owner.document, "div"), recovery = make(owner.document, "div"); recovery.className = "history-actions";
    const identity = make(owner.document, "input"); identity.placeholder = "已登记操作编号"; identity.setAttribute("aria-label", "已登记操作编号"); recovery.append(identity);
    const ownedPath = () => { if (!/^[A-Za-z0-9_.:-]{1,128}$/.test(identity.value)) throw new Error("请填写有效的已登记操作编号"); return `${base}/${key(alias.value)}/operations/${key(identity.value)}`; };
    button(owner, "读取原操作状态", async () => display(owner, await api(ownedPath())), recovery);
    button(owner, "读回远端并恢复本地登记", async () => display(owner, await api(ownedPath() + "/recover", { method: "POST", body: "{}" })), recovery);
    owner.editor.append(alias, operation, slot, recovery);
    const renderForm = () => {
      owner.edit++; slot.replaceChildren(); if (!operation.value) return;
      const selected = operation.value, selectedAlias = alias.value;
      if (!Object.hasOwn(actions, selected) || !actions[selected][2].includes(principal.role)
          || !capabilities.endpoints.some((endpoint) => endpoint.alias === selectedAlias)) {
        owner.status.textContent = "请选择当前身份可用的操作和已配置目标服务。"; return;
      }
      const schema = capabilities.schemas[selected === "registry_revoke" ? "registry_transition" : selected];
      if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
        owner.status.textContent = "当前服务未安装此操作合同，请读取配置后再试。"; return;
      }
      const form = createContractForm({ document: owner.document, schema, title: actions[selected][0], initial: selected === "source_grant" ? { task_id: owner.taskId } : {}, submitLabel: "确认提交本次操作", onSubmit: async (payload) => {
        if (!current(owner) || owner.pending || isBusy()) throw new Error("当前任务操作尚未完成，请稍后提交。");
        const edit = owner.edit; owner.pending = true;
        try {
          const suffix = selected === "source_revoke" ? `sources/grants/${key(payload.grant_id)}/revoke` : actions[selected][1];
          const result = await api(`${base}/${key(selectedAlias)}/${suffix}`, { method: "POST", body: JSON.stringify(selected === "source_revoke" ? {} : payload) });
          if (!current(owner) || edit !== owner.edit) return;
          identity.value = payload.operation_id || ""; display(owner, result);
        } finally { owner.pending = false; }
      } });
      form.element.addEventListener("input", () => owner.edit++); form.element.addEventListener("change", () => owner.edit++); slot.append(form.element);
    };
    alias.addEventListener("change", renderForm); operation.addEventListener("change", renderForm);
  }
  return { render, destroy() { state = null; getElement()?.replaceChildren(); } };
}
