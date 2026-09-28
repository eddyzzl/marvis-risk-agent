import { api } from "./api.js";
import { operationsFormHtml, operationsPayload, selectOptions } from "./operations-form.js";
import { runtimeHtml, inboxHtml, schedulesHtml, scheduleHtml, periodHtml, evidenceHtml } from "./operations-view.js";

const path = value => encodeURIComponent(value);
const errorText = error => Number(error?.status) === 409 ? `配置版本或证据已变化，操作未被接受。请返回并刷新后重试。${error.message || ""}` : error?.message || "读取失败，请重试";

// Operations is a global inbox, not a second task session. Every view and form
// source request is fenced so A → B → A cannot bind an old upload or target.
export function createOperationsController({ root, openButton, apiClient = api,
  adminToken = () => document.body?.dataset?.marvisPluginAdminToken || "",
  setIntervalFn = setInterval, clearIntervalFn = clearInterval } = {}) {
  let epoch = 0, sourceEpoch = 0, detailEpoch = 0, inboxEpoch = 0, timer = null;
  let principal = null, capabilities = null, notifications = [], schedules = [];
  let tab = "inbox", selected = null, period = null, formState = null, busy = false;
  let refreshing = false;
  const q = selector => root.querySelector(selector);
  const current = ticket => root.open && ticket === epoch;
  const writable = () => ["maker", "admin"].includes(principal?.role);
  const message = (text, informational = false) => {
    const el = q("[data-ops-error]"); el.textContent = text; el.classList.toggle("ops-info", informational);
  };
  const content = html => { q("[data-ops-body]").innerHTML = html; };
  const post = (url, body, headers) => apiClient(url, {method:"POST", ...(body === undefined ? {} : {body:JSON.stringify(body)}), ...(headers ? {headers} : {})});
  function badge() {
    const unread = notifications.filter(n => !n.read_at).length;
    const el = openButton.querySelector("[data-ops-unread]");
    el.hidden = !unread; el.textContent = String(unread);
    openButton.title = unread ? `最近 100 条通知中 ${unread} 条未读` : "监控通知与周期诊断";
  }
  function toolbar() {
    q("[data-ops-runtime]").innerHTML = runtimeHtml(capabilities, principal);
    q('[data-ops-action="new"]').hidden = !writable();
    root.querySelectorAll("[data-ops-tab]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.opsTab === tab)));
  }
  async function inbox({ render = false, valid = () => true } = {}) {
    const version = ++inboxEpoch, ticket = epoch;
    const payload = await apiClient("/api/operations/inbox?limit=100");
    if (version !== inboxEpoch || !valid()) return;
    const changed = JSON.stringify(notifications) !== JSON.stringify(payload.notifications);
    notifications = payload.notifications; badge();
    if (changed && render && current(ticket) && tab === "inbox" && !formState) content(inboxHtml(notifications));
  }
  function scheduleShell() {
    content(`<div class="ops-layout"><nav class="ops-schedules" aria-label="监控配置">${schedulesHtml(schedules, selected?.contract.schedule_id)}</nav><div class="ops-detail" data-ops-detail>${selected ? "正在读取周期…" : '<p class="ops-empty">选择监控查看周期与证据。</p>'}</div></div>`);
  }
  async function showSchedule(id, periodKey = null) {
    const ticket = ++epoch, detailTicket = ++detailEpoch;
    formState = null; tab = "schedules"; period = null; selected = null;
    toolbar(); scheduleShell(); message("");
    try {
      const [record, history] = await Promise.all([
        apiClient(`/api/operations/schedules/${path(id)}`),
        apiClient(`/api/operations/schedules/${path(id)}/periods?limit=100`),
      ]);
      if (!current(ticket) || detailTicket !== detailEpoch) return;
      selected = record; scheduleShell();
      q("[data-ops-detail]").innerHTML = scheduleHtml(record, history.periods, writable()) + '<div data-ops-period></div>';
      if (periodKey) await showPeriod(periodKey);
    } catch (error) { if (current(ticket)) { message(errorText(error)); q("[data-ops-detail]").textContent = "监控读取失败，请刷新重试。"; } }
  }
  async function showPeriod(key) {
    const ticket = epoch, version = ++detailEpoch, id = selected?.contract.schedule_id;
    if (!id) return;
    period = null; q("[data-ops-period]").textContent = "正在读取诊断…"; message("");
    try {
      const result = await apiClient(`/api/operations/schedules/${path(id)}/period?period_key=${path(key)}`);
      if (!current(ticket) || version !== detailEpoch) return;
      period = result; q("[data-ops-period]").innerHTML = periodHtml(result, writable());
      q("[data-ops-period]").scrollIntoView({block:"nearest"});
    } catch (error) { if (current(ticket) && version === detailEpoch) { message(errorText(error)); q("[data-ops-period]").textContent = "周期读取失败，请重试。"; } }
  }
  function claimHtml() {
    return `<form data-ops-form="claim" class="ops-claim"><h3>绑定当前会话身份</h3><p class="ops-note">监控配置、阅读回执会记录当前会话身份。请选择工作区授权的角色。</p>
      <label>姓名<input name="display_name" maxlength="120" required autocomplete="name"></label>
      <label>角色<select name="role"><option value="maker">配置人员（配置、查看和已读）</option><option value="checker">复核人员（查看和已读）</option><option value="admin">管理员</option></select></label>
      <button class="primary" type="submit">绑定当前会话</button></form>`;
  }
  async function load() {
    const ticket = ++epoch;
    formState = null; selected = null; period = null; message(""); content('<p class="ops-empty">正在读取监控…</p>');
    try {
      const identity = await apiClient("/api/production-governance/me");
      if (!current(ticket)) return;
      principal = identity;
    } catch (error) {
      if (!current(ticket)) return;
      principal = null; capabilities = null; toolbar();
      if (error.status === 403) content(claimHtml());
      else { content('<p class="ops-empty">身份读取失败</p>'); message(errorText(error)); }
      return;
    }
    try {
      const [caps, list, receipt] = await Promise.all([
        apiClient("/api/operations/capabilities"), apiClient("/api/operations/schedules?include_disabled=true"), apiClient("/api/operations/inbox?limit=100"),
      ]);
      if (!current(ticket)) return;
      capabilities = caps; schedules = list.schedules; notifications = receipt.notifications; ++inboxEpoch;
      toolbar(); badge();
      if (tab === "inbox") content(inboxHtml(notifications)); else scheduleShell();
    } catch (error) { if (current(ticket)) { message(errorText(error)); content('<p class="ops-empty">监控读取失败，请刷新重试。</p>'); } }
  }
  function values(form) {
    return Object.fromEntries(Array.from(form.querySelectorAll("[name]")).map(input => [input.name, input.type === "checkbox" ? input.checked : input.value]));
  }
  function labels(form) {
    const required = form.elements.label_mode.value === "required";
    for (const name of ["target_col", "label_maturity_seconds"]) {
      form.elements[name].disabled = !required; form.elements[name].required = required;
    }
  }
  function datasetChanged(preferred = {}) {
    const form = q('[data-ops-form="schedule"]');
    if (!form || !formState) return;
    const dataset = formState.datasets.find(d => d.id === form.elements.dataset_id.value);
    const columns = (dataset?.columns || []).map(c => ({id:typeof c === "string" ? c : c.name, label:typeof c === "string" ? c : c.name}));
    for (const name of ["time_col", "target_col", "score_col"]) form.elements[name].innerHTML = selectOptions(columns, preferred[name] || "");
    q("[data-ops-source-hash]").textContent = dataset ? `快照 ${dataset.id} · ${dataset.row_count ?? "未知"} 行 · SHA256 ${dataset.content_hash || "缺失"}` : "尚未选择数据快照";
    labels(form);
  }
  async function sources(preferred = {}) {
    const ticket = epoch, version = ++sourceEpoch, state = formState;
    const form = q('[data-ops-form="schedule"]');
    if (!form || !state) return;
    const taskId = form.elements.task_id.value;
    const ref = state.recheck ? state.record.contract.monitoring_ref : form.elements.monitoring_ref.value;
    state.datasets = []; state.targets = []; state.ready = false;
    form.elements.dataset_id.innerHTML = selectOptions([]); form.elements.target_id.innerHTML = selectOptions([]);
    datasetChanged(); q("[data-ops-publish]").disabled = true; q("[data-ops-upload]").disabled = true;
    q("[data-ops-source-message]").textContent = taskId && ref ? "正在读取任务数据和目标…" : "先选择任务与监控类型。";
    if (!taskId || !ref) return;
    try {
      const targetUrl = ref === "strategy.run_strategy_monitoring" ? `/api/tasks/${path(taskId)}/strategy-candidate-lab` : `/api/tasks/${path(taskId)}/experiments`;
      const [data, targetData] = await Promise.all([apiClient(`/api/tasks/${path(taskId)}/datasets`), apiClient(targetUrl)]);
      if (!current(ticket) || version !== sourceEpoch || formState !== state) return;
      state.datasets = data.datasets;
      state.targets = ref === "strategy.run_strategy_monitoring" ? (targetData.strategies || []).map(s => ({id:s.strategy_id, label:`${s.strategy_id} · v${s.version} · ${s.adopted_at ? "已采纳" : s.status}`})) : targetData.experiments.map(e => ({id:e.id,label:`${e.recipe_id} · ${e.id} · ${e.status}`}));
      form.elements.dataset_id.innerHTML = selectOptions(state.datasets.map(d => ({id:d.id,label:`${d.source_name || d.id} · ${d.row_count ?? "未知"} 行`})), preferred.dataset_id);
      form.elements.target_id.innerHTML = selectOptions(state.targets, preferred.target_id);
      // A retained recheck target must still be present in authoritative history.
      state.ready = true; q("[data-ops-publish]").disabled = false; q("[data-ops-upload]").disabled = false;
      q("[data-ops-source-message]").textContent = `${state.datasets.length} 个快照 · ${state.targets.length} 个目标。请选择明确的监控对象。`;
      datasetChanged(preferred);
    } catch (error) { if (current(ticket) && version === sourceEpoch) { q("[data-ops-source-message]").textContent = `任务数据读取失败：${errorText(error)}。重新选择任务可重试。`; } }
  }
  async function edit(record = null, recheck = null) {
    if (!writable()) { message("当前角色仅可查看监控和标记已读。"); return; }
    if (record?.contract.recheck_of && !recheck) { message("独立重新评估配置保留来源关联，请从周期发起下一次重新评估。"); return; }
    const ticket = ++epoch; message(""); content('<p class="ops-empty">正在读取可选任务…</p>');
    try {
      const tasks = await apiClient("/api/tasks");
      if (!current(ticket)) return;
      formState = {record, recheck, datasets:[], targets:[], ready:false, idempotencyKey:crypto.randomUUID()};
      content(operationsFormHtml({tasks, refs:capabilities.monitoring_refs, record, recheck}));
      labels(q('[data-ops-form="schedule"]'));
      if (record) await sources(record.contract.monitoring_binding || {});
    } catch (error) { if (current(ticket)) { message(errorText(error)); content('<p class="ops-empty">配置表单读取失败，请返回通知页重试。</p>'); } }
  }
  async function upload(input) {
    const file = input.files?.[0], state = formState, form = q('[data-ops-form="schedule"]');
    if (!file || !state || !form || busy) return;
    const ticket = epoch, version = ++sourceEpoch, taskId = form.elements.task_id.value;
    state.ready = false; q("[data-ops-publish]").disabled = true; input.disabled = true;
    q("[data-ops-source-message]").textContent = "正在上传快照；切换任务后此文件不会绑定到新任务。";
    const selectedValues = values(form), body = new FormData(); body.append("file", file);
    try {
      const result = await apiClient(`/api/tasks/${path(taskId)}/datasets/upload`, {method:"POST", body});
      if (!current(ticket) || version !== sourceEpoch || state !== formState) return;
      const datasets = result.datasets || [result.dataset];
      await sources({...selectedValues, dataset_id:datasets[0]?.id});
    } catch (error) { if (current(ticket) && version === sourceEpoch) { state.ready = true; input.disabled = false; q("[data-ops-publish]").disabled = false; q("[data-ops-source-message]").textContent = `上传失败：${errorText(error)}`; } }
  }
  async function submit(event) {
    const form = event.target.closest("[data-ops-form]");
    if (!form) return;
    event.preventDefault(); if (busy) return;
    const ticket = epoch, state = formState;
    try {
      let url, payload, headers;
      if (form.dataset.opsForm === "claim") {
        const token = adminToken(); if (!token) throw new Error("当前页面缺少工作区身份管理授权，请由工作区管理员绑定身份。");
        url = "/api/production-governance/principals/claim"; payload = values(form); headers = {"X-Marvis-Governance-Admin":token};
      } else {
        if (!state?.ready) throw new Error("任务数据尚未读取完成，请重新选择任务后重试。");
        payload = operationsPayload(values(form), state);
        url = state.recheck ? `/api/operations/schedules/${path(state.record.contract.schedule_id)}/rechecks` : "/api/operations/schedules";
      }
      busy = true;
      const controls = Array.from(form.querySelectorAll("button,input,select")).map(control => ({control, disabled:control.disabled}));
      controls.forEach(({control}) => { control.disabled = true; });
      message("正在提交，关闭窗口不会取消服务端操作。", true);
      try {
        const result = await post(url, payload, headers);
        if (!current(ticket)) return;
        if (form.dataset.opsForm === "claim") { await load(); }
        else {
          tab = "schedules";
          const loading = load(), loadTicket = epoch;
          await loading;
          if (current(loadTicket)) await showSchedule(result.contract.schedule_id);
        }
      } finally { busy = false; if (current(ticket)) controls.forEach(({control, disabled}) => { control.disabled = disabled; }); }
    } catch (error) { if (current(ticket)) { message(errorText(error)); const el = q("[data-ops-form-error]"); if (el) el.textContent = errorText(error); } }
  }
  async function click(event) {
    const button = event.target.closest("[data-ops-action], [data-ops-tab]");
    if (!button) return;
    const action = button.dataset.opsAction, ticket = epoch;
    if (action === "close") { root.close(); return; }
    if (busy) { message("操作提交中，请等待完成。"); return; }
    if (button.dataset.opsTab) { tab = button.dataset.opsTab; await load(); return; }
    try {
      if (action === "refresh" || action === "cancel-form") { await load(); return; }
      if (action === "new") { await edit(); return; }
      if (action === "schedule") { await showSchedule(button.dataset.id); return; }
      if (action === "notification") { const n = notifications.find(n => n.notification_id === button.dataset.id); if (n) await showSchedule(n.schedule_id, n.period_key); return; }
      if (action === "period") { await showPeriod(button.dataset.key); return; }
      if (action === "edit") { await edit(selected); return; }
      if (action === "recheck") { if (period) await edit(selected, period.period.period.key); return; }
      if (action === "evidence") {
        const periodSnapshot = period, version = detailEpoch;
        button.disabled = true; message("");
        const data = await apiClient(`/api/operations/schedules/${path(period.period.schedule_id)}/evidence?period_key=${path(period.period.period.key)}`);
        if (current(ticket) && version === detailEpoch && period === periodSnapshot) q("[data-ops-evidence]").innerHTML = evidenceHtml(data);
      }
      if (action === "ack") {
        busy = true; button.disabled = true; message("");
        try { await post(`/api/operations/inbox/${path(button.dataset.id)}/acknowledge`); await inbox({render:true}); }
        finally { busy = false; }
      }
    } catch (error) { if (current(ticket)) message(errorText(error)); }
    finally { if (current(ticket)) button.disabled = false; }
  }
  async function poll() {
    if (!principal || refreshing || busy) return;
    const ticket = epoch, identity = principal;
    // A closed dialog may refresh its badge, but an obsolete visit or identity
    // cannot change a newer view, including clearing it after an old 403.
    const valid = () => epoch === ticket && principal === identity;
    refreshing = true;
    try {
      await inbox({render:root.open && tab === "inbox", valid});
      if (!valid()) return;
      if (root.open) {
        const result = await apiClient("/api/operations/capabilities");
        if (!valid()) return;
        capabilities = result; toolbar();
      }
    } catch (error) {
      if (!valid()) return;
      if (root.open) message(`自动刷新失败：${errorText(error)}`);
      openButton.title = "监控通知刷新失败，请打开后刷新";
      if ([401,403].includes(error.status)) principal = null;
    } finally { refreshing = false; }
  }
  function bind() {
    openButton.addEventListener("click", () => { if (!root.open) root.showModal(); load(); });
    root.addEventListener("close", () => { ++epoch; ++sourceEpoch; ++detailEpoch; formState = null; });
    root.addEventListener("click", click); root.addEventListener("submit", submit);
    root.addEventListener("change", event => {
      if (event.target.matches("[data-ops-upload]")) { upload(event.target); return; }
      if (["task_id", "monitoring_ref"].includes(event.target.name)) { message("任务或类型已改变，请重新选择数据快照和目标。", true); sources(); }
      if (event.target.name === "dataset_id") datasetChanged();
      if (event.target.name === "label_mode") labels(q('[data-ops-form="schedule"]'));
    });
    // Existing sessions receive unread badges without having to open the modal.
    const bootstrapTicket = epoch;
    apiClient("/api/production-governance/me").then(p => {
      if (bootstrapTicket !== epoch) return;
      principal = p; return poll();
    }).catch(() => {});
    timer = setIntervalFn(poll, 15000);
  }
  return {bind, open:() => { if (!root.open) root.showModal(); return load(); }, destroy:() => { ++epoch; if (timer) clearIntervalFn(timer); }};
}
