import { apiGet, apiPost } from "../api.js";

const metrics = ["volume", "approval_rate", "bad_rate", "loan_amount", "delinquent_amount", "ks", "auc", "psi", "lift", "net_yield"];
const units = ["unknown", "count", "ratio", "percent", "currency", "score"];
const formats = ["unknown", "date", "month", "date_range"];
const kinds = { field: "字段", metric: "指标", period: "期间", strategy_version: "策略版本" };
const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const errorMessage = (error) => error?.detail?.message || error?.payload?.detail?.message || error?.message || "历史映射未完成，请重新读取材料。";
const requestKeys = ["binding_key", "source_id", "source_hash", "locator", "cell_hash", "target", "period_key", "version_key"];
const requestOnly = (binding) => Object.fromEntries(requestKeys.map((key) => [key, binding[key]]));

function location(locator) {
  if (locator.format === "csv") return `CSV 第 ${locator.row} 行、第 ${locator.column} 列`;
  if (locator.format === "xlsx") return `${locator.sheet}!${locator.cell}`;
  return `JSON ${locator.pointer || "根值"}`;
}

export function createStrategyHistoryMappingPanel({ root, get = apiGet, post = apiPost }) {
  root.innerHTML = `<details><summary>确认历史材料中的字段和指标</summary>
    <p>来源值留在本地。历史报告值未经平台复算，不能据此宣称验证通过；缺项保持未知。</p>
    <label>已登记材料 <select data-hm-source aria-label="历史映射材料"></select></label>
    <button type="button" class="button compact secondary" data-hm-refresh>重新读取材料</button>
    <label>输入方式 <select data-hm-mode><option value="form">选择来源位置</option><option value="instruction">完整位置指令</option></select></label>
    <div data-hm-form>
      <label>来源单元格 <select data-hm-cell aria-label="历史来源单元格"></select></label>
      <button type="button" class="button compact secondary" data-hm-more>下一页位置</button>
      <label>类型 <select data-hm-kind>${Object.entries(kinds).map(([key, label]) => `<option value="${key}">${label}</option>`).join("")}</select></label>
      <label>口径名称 <input data-hm-name maxlength="128" placeholder="输入明确的字段口径名称"></label>
      <label>指标名称 <select data-hm-metric><option value="">请明确选择</option>${metrics.map((name) => `<option>${name}</option>`).join("")}</select></label>
      <label>指标单位 <select data-hm-unit>${units.map((name) => `<option>${name}</option>`).join("")}</select></label>
      <label>期间格式 <select data-hm-format>${formats.map((name) => `<option>${name}</option>`).join("")}</select></label>
      <label>明确关联期间 <select data-hm-period><option value="">未知 / 不关联</option></select></label>
      <label>明确关联版本 <select data-hm-version><option value="">未知 / 不关联</option></select></label>
      <button type="button" class="button compact secondary" data-hm-add>加入待确认映射</button>
    </div>
    <label data-hm-instruction-label hidden>完整位置指令（确认后替换当前全部映射）<textarea data-hm-instruction rows="3" placeholder="将 CSV 第2行第3列映射为指标 bad_rate，单位 percent；将 CSV 第2行第1列映射为期间 effective_period，格式 month"></textarea></label>
    <div data-hm-draft></div>
    <button type="button" class="button compact secondary" data-hm-propose>核对映射</button>
    <div data-hm-preview hidden></div>
    <button type="button" class="button compact primary" data-hm-confirm disabled>确认并保存来源映射</button>
    <p data-hm-status role="status" aria-live="polite">请选择策略任务。</p>
  </details>`;
  const field = (key) => root.querySelector(`[data-hm-${key}]`);
  let taskId = "", epoch = 0, sources = [], cells = [], sourceHash = "", nextOffset = null, draft = [], proposal = null;
  let head = { revision: 0, artifact_id: null, content_hash: null }, busy = false;
  const endpoint = () => `/api/tasks/${encodeURIComponent(taskId)}/strategy/history-mapping`;
  function invalidate() { proposal = null; field("preview").hidden = true; field("preview").innerHTML = ""; field("confirm").disabled = true; }
  function sync() {
    const kind = field("kind").value || "field";
    field("form").hidden = field("mode").value === "instruction";
    field("instruction-label").hidden = field("mode").value !== "instruction";
    field("metric").parentElement.hidden = kind !== "metric";
    field("unit").parentElement.hidden = kind !== "metric";
    field("format").parentElement.hidden = kind !== "period";
    field("name").parentElement.hidden = kind === "metric" || kind === "strategy_version";
    const ready = sources.some((source) => source.artifact_id === field("source").value && source.status === "ready");
    for (const name of ["source", "mode", "cell", "kind", "name", "metric", "unit", "format", "period", "version", "instruction", "refresh"]) field(name).disabled = busy;
    field("add").disabled = busy || !ready || !cells.length;
    field("more").disabled = busy || nextOffset === null;
    field("propose").disabled = busy || !taskId || (field("mode").value === "instruction" ? !ready : !draft.length && head.revision === 0);
    field("confirm").disabled = busy || !proposal;
  }
  function renderDraft() {
    field("draft").innerHTML = draft.length ? `<table><thead><tr><th>类型 / 口径</th><th>来源位置</th><th>关联</th><th></th></tr></thead><tbody>${draft.map((binding, index) => `<tr><td>${escape(kinds[binding.target.kind])} / ${escape(binding.target.name)}</td><td>${escape(binding.source_hash.slice(0, 12))} · ${escape(location(binding.locator))}</td><td>期间：${escape(binding.period_key || "未知")}；版本：${escape(binding.version_key || "未知")}</td><td><button type="button" class="button compact secondary" data-hm-remove="${index}">移除</button></td></tr>`).join("")}</tbody></table>` : "尚未选择映射。";
    for (const [name, kind] of [["period", "period"], ["version", "strategy_version"]]) {
      field(name).innerHTML = `<option value="">未知 / 不关联</option>${draft.filter((binding) => binding.target.kind === kind).map((binding) => `<option value="${escape(binding.binding_key)}">${escape(binding.target.name)} · ${escape(location(binding.locator))}</option>`).join("")}`;
    }
  }
  async function loadCells(offset = 0) {
    const task = taskId, sourceId = field("source").value, operation = ++epoch;
    invalidate(); busy = true; sync();
    try {
      const response = await get(`${endpoint()}/sources/${encodeURIComponent(sourceId)}/cells?offset=${offset}`);
      if (operation !== epoch || task !== taskId) return;
      cells = response.cells; sourceHash = response.source_hash; nextOffset = response.next_offset;
      field("cell").innerHTML = cells.map((cell, index) => `<option value="${index}">${escape(location(cell.locator))} · ${escape(cell.state === "missing" ? "未知（空白）" : String(cell.value))}</option>`).join("");
      field("status").textContent = cells.length ? "选择明确位置和类型，再核对并确认。" : "该材料没有可定位的单元值，请提供静态汇总表。";
    } catch (error) { if (operation === epoch && task === taskId) { cells = []; nextOffset = null; field("cell").innerHTML = ""; field("status").textContent = errorMessage(error); } }
    finally { if (operation === epoch && task === taskId) { busy = false; sync(); } }
  }
  async function refresh() {
    const task = taskId, operation = ++epoch;
    invalidate(); cells = []; busy = true; sync();
    if (!task) { busy = false; sync(); return; }
    try {
      const [current, available] = await Promise.all([get(endpoint()), get(endpoint() + "/sources")]);
      if (operation !== epoch || task !== taskId) return;
      head = current.head; draft = (current.mapping?.bindings || []).map(requestOnly); sources = available.sources;
      field("source").innerHTML = sources.map((source) => `<option value="${escape(source.artifact_id)}">${escape(source.name)}${source.status === "ready" ? "" : " · 无法定位"}</option>`).join("");
      renderDraft();
      busy = false; sync();
      if (sources[0]?.status === "ready") await loadCells();
      else field("status").textContent = sources[0]?.next_action || "请先在“项目现状与历史材料”中登记本地材料，再重新读取。";
    } catch (error) { if (operation === epoch && task === taskId) { busy = false; field("status").textContent = errorMessage(error); sync(); } }
  }
  field("source").addEventListener("change", () => {
    const source = sources.find((item) => item.artifact_id === field("source").value);
    if (source?.status === "ready") loadCells();
    else { epoch += 1; cells = []; nextOffset = null; invalidate(); field("status").textContent = source?.next_action || "来源不可用。"; sync(); }
  });
  field("refresh").addEventListener("click", refresh);
  field("more").addEventListener("click", () => loadCells(nextOffset));
  for (const name of ["mode", "kind", "name", "metric", "unit", "format", "period", "version", "cell", "instruction"]) field(name).addEventListener(name === "instruction" || name === "name" ? "input" : "change", () => { invalidate(); sync(); });
  field("add").addEventListener("click", () => {
    if (field("add").disabled) return;
    const cell = cells[Number(field("cell").value)], kind = field("kind").value || "field";
    const name = kind === "metric" ? field("metric").value : (kind === "strategy_version" ? "strategy_version" : field("name").value);
    if (!cell || !name.trim()) { field("status").textContent = "请明确选择口径名称，不能根据材料自动猜测。"; return; }
    let ordinal = 1;
    while (draft.some((binding) => binding.binding_key === `b${ordinal}`)) ordinal += 1;
    draft.push({binding_key:`b${ordinal}`, source_id:field("source").value, source_hash:sourceHash, locator:cell.locator,
      cell_hash:cell.cell_hash, target:{kind, name, unit:kind === "metric" ? field("unit").value : "unknown", period_format:kind === "period" ? field("format").value : "unknown"},
      period_key:field("period").value || null, version_key:field("version").value || null});
    invalidate(); renderDraft(); sync();
  });
  field("draft").addEventListener("click", (event) => {
    const button = event.target.closest?.("[data-hm-remove]");
    if (!button || busy) return;
    const removed = draft.splice(Number(button.dataset.hmRemove), 1)[0];
    for (const binding of draft) { if (binding.period_key === removed.binding_key) binding.period_key = null; if (binding.version_key === removed.binding_key) binding.version_key = null; }
    invalidate(); renderDraft(); sync();
  });
  field("propose").addEventListener("click", async () => {
    if (field("propose").disabled) return;
    const task = taskId, operation = ++epoch;
    const body = field("mode").value === "instruction" ? {expected_head:head, source_id:field("source").value, instruction:field("instruction").value} : {expected_head:head, bindings:draft};
    busy = true; invalidate(); sync();
    try {
      const response = await post(endpoint() + "/proposals", body);
      if (operation !== epoch || task !== taskId) return;
      proposal = response;
      field("preview").innerHTML = `<p>${escape(response.historical_declaration)}</p><table><thead><tr><th>口径</th><th>来源位置</th><th>确认值</th><th>关联期间 / 版本</th><th>状态</th></tr></thead><tbody>${response.bindings.map((binding) => `<tr><td>${escape(binding.target.name)}</td><td>${escape(binding.source_hash.slice(0, 12))} · ${escape(location(binding.locator))}</td><td>${escape(binding.availability === "unknown" ? "未知" : typeof binding.value === "object" ? JSON.stringify(binding.value) : binding.value)}</td><td>${escape(binding.period_key || "未知")} / ${escape(binding.version_key || "未知")}</td><td>${escape(binding.target.unit)} · 历史报告值，未验证</td></tr>`).join("")}</tbody></table>`;
      field("preview").hidden = false;
      field("status").textContent = response.bindings.length ? "请核对全部来源、值和关联。确认后，上表将成为完整映射的新版本；尚未保存。" : "本次确认将清空当前全部映射，原修订仍保留；尚未保存。";
    } catch (error) { if (operation === epoch && task === taskId) field("status").textContent = errorMessage(error); }
    finally { if (operation === epoch && task === taskId) { busy = false; sync(); } }
  });
  field("confirm").addEventListener("click", async () => {
    if (field("confirm").disabled) return;
    const task = taskId, frozen = proposal, operation = ++epoch;
    busy = true; sync();
    try {
      const response = await post(endpoint() + "/confirm", {proposal_token:frozen.proposal_token});
      if (operation !== epoch || task !== taskId) return;
      head = response.head; draft = response.mapping.bindings.map(requestOnly); invalidate(); renderDraft();
      field("status").textContent = response.next_action;
    } catch (error) { if (operation === epoch && task === taskId) { invalidate(); field("status").textContent = errorMessage(error); } }
    finally { if (operation === epoch && task === taskId) { busy = false; sync(); } }
  });
  sync();
  return {refresh, selectTask(task) {
    const selected = task?.task_type === "strategy" ? task.id : "";
    if (selected === taskId) return Promise.resolve();
    epoch += 1; taskId = selected; busy = false; draft = []; cells = []; sources = []; nextOffset = null;
    head = {revision:0, artifact_id:null, content_hash:null}; field("instruction").value = ""; field("name").value = "";
    field("source").innerHTML = ""; field("cell").innerHTML = ""; field("metric").value = "";
    field("unit").value = "unknown"; field("format").value = "unknown"; invalidate(); renderDraft(); sync();
    return refresh();
  }};
}
