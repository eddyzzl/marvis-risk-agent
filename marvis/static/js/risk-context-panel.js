import { api as defaultApi } from "./api.js";
import { createContractForm } from "./contract-form.js";
import { taskUsesPlanRail } from "./v2/plan_rail_controller.js";

const id = encodeURIComponent;
const stateLabel = (value) => ({ prepared: "已准备", running: "执行中", completed: "已完成", failed: "执行失败", unknown_effect: "待核对" }[value] || value);
const node = (document, tag, text, className = "") => {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  element.className = className;
  return element;
};
const errorText = (error) => error?.detail?.next_action ? `${error.detail.next_action}（${error.detail.code || "请求失败"}）` : error?.detail?.code || error.message || "请求失败";

export function createRiskContextPanel({ getElement, getTask, getPlan, captureView, isCurrentView, api = defaultApi, onPlanCreated = async () => {}, refreshPlan = async () => {}, openIdentity = () => {}, isBusy = () => false }) {
  let state = null;
  const current = (owner) => owner === state && getTask()?.id === owner.taskId && isCurrentView(owner.view);
  const post = (path, body = {}) => api(path, { method: "POST", body: JSON.stringify(body) });
  function button(document, label, action, parent) {
    const element = node(document, "button", label, "button compact secondary"); element.type = "button";
    element.addEventListener("click", async () => { element.disabled = true; try { await action(); } finally { if (element.isConnected) element.disabled = false; } });
    parent.append(element); return element;
  }
  function show(owner, payload, title = "服务器回执") {
    if (!current(owner)) return;
    owner.result.replaceChildren(node(owner.document, "h4", title));
    const facts = node(owner.document, "p", [stateLabel(payload.status), stateLabel(payload.state), payload.artifact_id, payload.request_id].filter(Boolean).join(" · "), "history-note"); owner.result.append(facts);
    const details = node(owner.document, "details"); details.append(node(owner.document, "summary", "查看完整证据回执"));
    const pre = node(owner.document, "pre", JSON.stringify(payload, null, 2)); pre.className = "contract-receipt"; details.append(pre); owner.result.append(details);
  }
  function message(owner, text) { if (current(owner)) owner.status.textContent = text; }
  async function request(owner, callback, awaitingGate = false) {
    if (!current(owner) || owner.pending || (isBusy() && !awaitingGate)) { message(owner, "当前操作尚未完成，请稍后再试。"); return; }
    owner.pending = true; message(owner, "正在读取原始状态…");
    try { await callback(); if (current(owner) && owner.status.textContent === "正在读取原始状态…") message(owner, "状态已更新。"); }
    catch (error) { message(owner, errorText(error)); }
    finally { owner.pending = false; }
  }
  function render() {
    const panel = getElement(), task = getTask(); if (!panel) return;
    panel.hidden = !taskUsesPlanRail(task);
    if (panel.hidden) { state = null; panel.replaceChildren(); return; }
    if (state && current(state)) {
      const plan = getPlan();
      renderGate(state, plan);
      if (state.loaded && plan?.status === "done" && plan.id !== state.completedPlan && plan.steps?.some((step) => step.tool_ref?.plugin === "risk_context")) { state.completedPlan = plan.id; void load(state, true); }
      return;
    }
    const document = panel.ownerDocument;
    const details = node(document, "details"), summary = node(document, "summary", "外部数据与事件特征");
    const body = node(document, "div", null, "risk-context-body");
    body.append(node(document, "p", "登记脱敏来源及授权，查询历史记录，按决策时点冻结事件窗口和关系特征。覆盖不足、未来信息和无结果会明确保留。", "history-note"));
    const actions = node(document, "div", null, "history-actions"), status = node(document, "p", "准备数据服务后可查看来源、授权和执行结果。", "history-note"); status.setAttribute("role", "status");
    const editor = node(document, "div"), listing = node(document, "div"), result = node(document, "div"), gate = node(document, "div");
    body.append(actions, status, editor, gate, listing, result); details.append(summary, body); panel.replaceChildren(details);
    const owner = state = { document, taskId: task.id, view: captureView(), actions, status, editor, listing, result, gate, pending: false, loaded: false, edit: 0 };
    button(document, "准备与刷新来源", () => load(owner), actions);
    button(document, "配置会话身份", openIdentity, actions);
    renderGate(owner, getPlan());
  }
  function renderGate(owner, plan) {
    const step = plan?.steps?.find((step) => step.status === "awaiting_confirm" && step.tool_ref?.plugin === "risk_context" && step.tool_ref?.tool === "replay_events");
    const signature = JSON.stringify([plan?.id, step?.id, step?.confirmation_snapshot]);
    if (signature === owner.gateSignature) return;
    owner.gateSignature = signature; owner.gate.replaceChildren();
    if (!step) return;
    const { document } = owner;
    owner.gate.append(node(document, "h4", "人工复核事件回放"), node(document, "p", "请核对原生来源、可得截止时间、窗口、关系与覆盖未知项。本次授权只适用于当前计划快照。", "history-note"));
    const detail = node(document, "details"); detail.append(node(document, "summary", "查看本次原生合同"), node(document, "pre", JSON.stringify(step.inputs, null, 2), "contract-receipt")); owner.gate.append(detail);
    const reason = node(document, "input"); reason.placeholder = "填写本次复核理由"; reason.maxLength = 4000; reason.setAttribute("aria-label", "事件回放复核理由"); owner.gate.append(reason);
    const actions = node(document, "div", null, "history-actions"); owner.gate.append(actions);
    for (const [decision, label] of [["approve", "授权本次事件回放"], ["reject", "拒绝本次事件回放"]]) button(document, label, () => request(owner, async () => {
      if (!reason.value.trim()) throw new Error("请填写本次复核理由。");
      if (!step.confirmation_snapshot?.expected_step_fingerprint) throw new Error("计划快照缺失，请刷新后复核。");
      await post(`/api/plans/${id(plan.id)}/steps/${id(step.id)}/decisions`, { decision, reason: reason.value.trim(), ...step.confirmation_snapshot });
      if (current(owner)) await refreshPlan(owner.taskId);
      message(owner, decision === "approve" ? "已记录授权，完成后可读取冻结证据。" : "已记录拒绝决定。");
    }, true), actions);
  }
  async function load(owner, readOnly = false) {
    if (!current(owner) || owner.pending) return;
    owner.pending = true; message(owner, "正在校验当前身份、来源和授权…");
    try {
      const principal = await api("/api/production-governance/me");
      if (!current(owner)) return;
      const base = `/api/tasks/${id(owner.taskId)}`;
      const [source, event, profiles, access, data, artifacts, sourceRequests] = await Promise.all([
        api("/api/risk-sources/capabilities"), api("/api/risk-events/capabilities"), api("/api/risk-sources/profiles"), api(base + "/risk-events"), api(base + "/datasets"), api(base + "/task-artifacts"), principal.role === "checker" ? Promise.resolve({ requests: [] }) : api(base + "/risk-sources/requests"),
      ]);
      const grants = principal.role === "checker" ? { grants: [] } : await api(base + "/risk-sources/grants");
      if (!current(owner)) return;
      Object.assign(owner, { principal, source, event, profiles: profiles.profiles, eventAccess: access, sourceGrants: grants.grants, datasets: data.datasets, artifacts: artifacts.artifacts, sourceRequests: sourceRequests.requests, loaded: true });
      renderListing(owner);
      if (!readOnly) renderEditor(owner);
      message(owner, `身份 ${principal.display_name || principal.id} · ${{ maker: "发起人", checker: "复核人", admin: "管理员" }[principal.role] || principal.role}。来源声明与覆盖情况将作为证据展示。`);
    } catch (error) { message(owner, error.status === 403 ? "当前会话尚无可用治理身份，请先配置身份。" : errorText(error)); }
    finally { owner.pending = false; }
  }
  function renderEditor(owner) {
    const { document, principal } = owner;
    const select = node(document, "select"); select.setAttribute("aria-label", "数据服务操作");
    const options = [
      ...(principal.role === "admin" ? [["source-profile", "登记数据服务"], ["basis", "登记授权依据"], ["source-grant", "授予数据查询权限"], ["event-source", "登记事件来源"], ["event-grant", "授予事件权限"]] : []),
      ...(principal.role === "maker" ? [["source-query", "准备历史或服务查询"]] : []),
      ...(["maker", "admin"].includes(principal.role) ? [["event-import", "导入已登记事件快照"], ["event-query", "准备事件特征回放"]] : []),
    ];
    select.append(node(document, "option", "请选择操作"));
    options.forEach(([value, label]) => { const option = node(document, "option", label); option.value = value; select.append(option); });
    const slot = node(document, "div"); owner.editor.replaceChildren(select, slot);
    select.addEventListener("change", () => {
      owner.edit++; slot.replaceChildren();
      if (!select.value) return;
      const base = `/api/tasks/${id(owner.taskId)}`, operation = select.value;
      const activeGrant = owner.eventAccess.grants.find((grant) => grant.owned && grant.state === "active" && grant.permissions.includes("write") && grant.permissions.includes("read"));
      const source = owner.eventAccess.sources.find((source) => source.source_id === activeGrant?.source_id);
      const dataset = owner.datasets[0];
      const routes = {
        "source-profile": [owner.source.profile_schema, "/api/risk-sources/profiles", {}],
        basis: [owner.source.basis_schema, base + "/risk-sources/authorization-bases", {}],
        "source-grant": [owner.source.grant_schema, "/api/risk-sources/grants", { task_id: owner.taskId }],
        "source-query": [owner.source.query_schema, base + "/risk-sources/requests", {}],
        "event-source": [owner.event.source_schema, "/api/risk-events/sources", { task_id: owner.taskId }],
        "event-grant": [owner.event.grant_schema, "/api/risk-events/grants", { task_id: owner.taskId, source_id: source?.source_id, source_contract_hash: source?.contract_hash }],
        "event-import": [owner.event.import_schema, base + "/risk-events/imports", { grant_id: activeGrant?.grant_id, dataset: { source_id: source?.source_id, source_contract_hash: source?.contract_hash, dataset_id: dataset?.id, expected_content_hash: dataset?.content_hash } }],
        "event-query": [owner.event.request_schema, base + "/risk-events/requests", { grant_id: activeGrant?.grant_id, contract: { source_id: source?.source_id, source_contract_hash: source?.contract_hash, event_types: source?.event_types, window_features: [], relation_features: [] } }],
      };
      const [schema, route, initial] = routes[operation];
      const choices = {
        dataset_id: owner.datasets.map((d) => ({ value: d.id, label: `${d.source_name || d.id} · ${d.row_count} 行` })),
        basis_artifact_id: owner.artifacts.filter((a) => a.available).map((a) => ({ value: a.id, label: `${a.filename} · ${a.id}` })),
        profile_id: owner.profiles.map((p) => ({ value: p.profile_id, label: p.profile_id })),
        source_id: owner.eventAccess.sources.map((s) => ({ value: s.source_id, label: s.source_id })),
        grant_id: (operation === "source-query" ? owner.sourceGrants : owner.eventAccess.grants).filter((g) => g.state === "active" && (operation === "source-query" || g.owned)).map((g) => ({ value: g.grant_id, label: `${g.grant_id} · ${g.profile_id || g.source_id}` })),
      };
      try {
        const form = createContractForm({ document, schema, initial, choices, title: options.find(([value]) => value === operation)[1], submitLabel: operation.includes("query") ? "校验并准备请求" : "确认登记", onSubmit: async (payload) => {
          if (!current(owner) || isBusy() || owner.pending) throw new Error("当前任务正在执行，请稍后提交。");
          const edit = owner.edit; owner.pending = true;
          try {
            const result = await post(route, payload);
            if (!current(owner) || edit !== owner.edit) return;
            show(owner, result);
            if (operation === "event-query") showEventProposal(owner, result);
            if (operation === "source-query") sourceActions(owner, result.request_id || payload.request_id);
            message(owner, "请求已登记；修改输入后须使用新请求编号重新准备。可刷新查看已登记状态。");
          } finally { owner.pending = false; }
        } });
        form.element.addEventListener("input", () => owner.edit++);
        form.element.addEventListener("change", (event) => {
          owner.edit++;
          const path = event.target.dataset.contractInput;
          const bind = (suffix, value) => { const target = form.element.querySelector(`[data-contract-input="${suffix}"]`); if (target && value != null) target.value = value; };
          if (path?.endsWith("dataset_id")) {
            const selected = owner.datasets.find((d) => d.id === event.target.value);
            const prefix = path.slice(0, -"dataset_id".length);
            bind(prefix + "expected_content_hash", selected?.content_hash);
          }
          if (path?.endsWith("source_id")) {
            const selected = owner.eventAccess.sources.find((s) => s.source_id === event.target.value);
            bind(path.slice(0, -"source_id".length) + "source_contract_hash", selected?.contract_hash);
          }
        });
        slot.append(form.element);
      } catch (error) { message(owner, errorText(error)); }
    });
  }
  function sourceActions(owner, requestId, parent = owner.result) {
    const base = `/api/tasks/${id(owner.taskId)}/risk-sources/requests/${id(requestId)}`;
    button(owner.document, "读取状态与结果", () => request(owner, async () => show(owner, await api(base))), parent);
    if (owner.principal.role === "maker") button(owner.document, "执行或恢复本次查询", () => request(owner, async () => { const result = await post(base + "/execute"); show(owner, result); sourceActions(owner, requestId); }), parent);
    button(owner.document, "读取原始证据回执", () => request(owner, async () => show(owner, await api(base + "/evidence"))), parent);
  }
  function showEventProposal(owner, proposal) {
    if (!current(owner)) return;
    const edit = owner.edit;
    button(owner.document, "提交到事件回放执行计划", () => request(owner, async () => {
      if (edit !== owner.edit) throw new Error("输入已改变，请重新准备事件请求。");
      const result = await post(`/api/tasks/${id(owner.taskId)}/plans`, { goal: "事件窗口特征回放", slots: { event_request_id: proposal.request_id, event_proposal_hash: proposal.proposal_hash, event_contract: proposal.contract } });
      if (!current(owner)) return;
      await onPlanCreated(result.plan, owner.view);
      message(owner, "计划已生成，请在执行计划中开始并复核授权，完成后刷新查看冻结证据。");
    }), owner.result);
  }
  function renderListing(owner) {
    const { document } = owner;
    owner.listing.replaceChildren(node(document, "h4", "来源与授权"));
    for (const profile of owner.profiles) owner.listing.append(node(document, "p", `${profile.profile_id} · ${profile.mode === "historical_file" ? "历史快照" : "本机参考服务"}`, "history-note"));
    for (const grant of [...owner.sourceGrants.map((g) => ({ ...g, domain: "source" })), ...owner.eventAccess.grants.map((g) => ({ ...g, domain: "event" }))]) {
      const row = node(document, "div", null, "risk-context-row");
      row.append(node(document, "span", `${grant.grant_id} · ${grant.profile_id || grant.source_id} · ${{ active: "有效", revoked: "已撤销", not_current: "未生效或已过期" }[grant.state]}`));
      if (owner.principal.role === "admin" && grant.state !== "revoked") button(document, "撤销授权", () => request(owner, async () => {
        const path = grant.domain === "source" ? `/api/risk-sources/grants/${id(grant.grant_id)}/revoke` : `/api/tasks/${id(owner.taskId)}/risk-events/grants/${id(grant.grant_id)}/revoke`;
        show(owner, await post(path)); grant.state = "revoked"; renderListing(owner);
      }), row);
      owner.listing.append(row);
    }
    owner.listing.append(node(document, "h4", "已登记请求"));
    for (const record of owner.sourceRequests) {
      const row = node(document, "div", null, "risk-context-row"); row.append(node(document, "span", `${record.request_id || record.id} · ${stateLabel(record.state || record.status) || "已登记"}`)); sourceActions(owner, record.request_id || record.id, row); owner.listing.append(row);
    }
    for (const record of owner.eventAccess.requests) {
      const row = node(document, "div", null, "risk-context-row"); row.append(node(document, "span", `${record.request_id} · ${record.decision_at}`));
      const base = `/api/tasks/${id(owner.taskId)}/risk-events/requests/${id(record.request_id)}`;
      button(document, "读取冻结证据", () => request(owner, async () => show(owner, await api(base + `/evidence?grant_id=${id(record.grant_id)}`))), row);
      button(document, "读取原提案", () => request(owner, async () => { const proposal = await api(base + `/proposal?grant_id=${id(record.grant_id)}`); show(owner, proposal); showEventProposal(owner, proposal); }), row);
      owner.listing.append(row);
    }
    if (!owner.sourceRequests.length && !owner.eventAccess.requests.length) owner.listing.append(node(document, "p", "尚无当前身份可读取的请求。", "history-note"));
  }
  return { render, refresh: () => state ? load(state) : Promise.resolve(), destroy() { state = null; getElement()?.replaceChildren(); } };
}
