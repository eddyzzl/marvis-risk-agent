import { apiGet, apiPost } from "../api.js";
import { matchesSqlResultWorkspace, readSqlResultDownload, readSqlResultPreview,
  saveSqlResultDownload, sqlResultBinding } from "./sql_import_controller.js";

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function message(error) {
  const detail = error?.detail || error?.payload?.detail;
  const text = detail?.message || error?.message || "SQL 请求未完成，请重试。";
  return detail?.next_action && !text.includes(detail.next_action) ? `${text} ${detail.next_action}` : text;
}

export function createSqlImportPanel({ controller, getElementById, onImported = () => {},
  get = apiGet, post = apiPost, saveDownload = saveSqlResultDownload }) {
  const root = getElementById("sqlImportPanel");
  if (!root) throw new Error("SQL import panel is missing");
  root.innerHTML = `<details><summary>从 SQL 数据源导入</summary>
    <p>连接信息留在本地。先核对查询，再将结果作为当前数据集导入。</p>
    <label>已配置数据源 <select data-sql-source aria-label="SQL 数据源"></select></label>
    <button class="button compact secondary" type="button" data-sql-refresh>重新读取来源</button>
    <p data-sql-schema></p>
    <label>编写方式 <select data-sql-mode aria-label="SQL 编写方式"><option value="question">自然语言</option><option value="query">手动 SELECT</option></select></label>
    <label>查询目标<textarea data-sql-input aria-label="SQL 查询目标" rows="3" placeholder="选择哪些字段、如何筛选或分组；不会自动执行。"></textarea></label>
    <small>支持单表 SELECT、字段选择、COUNT/SUM/AVG/MIN/MAX、AND 筛选、分组、排序和 LIMIT；其他语法会明确提示。</small>
    <div><button class="button compact secondary" type="button" data-sql-propose>生成待确认查询</button></div>
    <pre data-sql-preview hidden></pre>
    <button class="button compact primary" type="button" data-sql-confirm disabled>确认查询并导入</button>
    <p data-sql-status role="status" aria-live="polite">请先选择当前任务。</p>
    <section data-sql-result hidden aria-label="SQL 导入结果" style="padding:12px;border-top:1px solid var(--border)">
      <strong>导入结果预览</strong><p data-sql-result-meta style="overflow-wrap:anywhere"></p>
      <small>最多预览 50 行；预览按平台隐私规则脱敏，下载保留原数据。文件保存前核对完整内容指纹。</small>
      <div data-sql-result-table style="max-width:100%;overflow-x:auto"></div>
      <p data-sql-result-status role="status" aria-live="polite"></p>
      <button class="button compact secondary" type="button" data-sql-result-refresh>重新读取结果</button>
      <button class="button compact primary" type="button" data-sql-download disabled>下载结果数据集</button>
      <button class="button compact secondary" type="button" data-sql-cancel-read hidden>取消结果读取</button>
    </section>
  </details>`;
  const sourceSelect = root.querySelector("[data-sql-source]");
  const mode = root.querySelector("[data-sql-mode]");
  const input = root.querySelector("[data-sql-input]");
  const preview = root.querySelector("[data-sql-preview]");
  const confirm = root.querySelector("[data-sql-confirm]");
  const propose = root.querySelector("[data-sql-propose]");
  const status = root.querySelector("[data-sql-status]");
  const schema = root.querySelector("[data-sql-schema]");
  const sourceRefresh = root.querySelector("[data-sql-refresh]");
  const resultRoot = root.querySelector("[data-sql-result]");
  const resultMeta = root.querySelector("[data-sql-result-meta]");
  const resultTable = root.querySelector("[data-sql-result-table]");
  const resultStatus = root.querySelector("[data-sql-result-status]");
  const resultRefresh = root.querySelector("[data-sql-result-refresh]");
  const download = root.querySelector("[data-sql-download]");
  const cancelRead = root.querySelector("[data-sql-cancel-read]");
  let taskId = "", revision = null, epoch = 0, sources = [], proposal = null, busy = false;
  let result = null, resultEpoch = 0, readAbort = null, readBusy = false, resultVerified = false;

  function clearResult() {
    resultEpoch += 1; readAbort?.abort(); readAbort = null;
    result = null; readBusy = false; resultVerified = false;
    resultRoot.hidden = true; resultTable.innerHTML = "";
    resultMeta.textContent = ""; resultStatus.textContent = "";
  }
  function currentResult(binding, operation) {
    return operation === resultEpoch && result === binding
      && matchesSqlResultWorkspace(binding, controller.getState());
  }
  async function previewResult() {
    if (!result || readBusy) return;
    const binding = result, operation = ++resultEpoch;
    readAbort?.abort(); readAbort = new AbortController(); readBusy = true;
    resultVerified = false; resultTable.innerHTML = "";
    resultStatus.textContent = "正在读取已导入数据的预览…"; sync();
    try {
      const data = await readSqlResultPreview(binding, { get, getState: controller.getState, signal: readAbort.signal });
      if (!currentResult(binding, operation)) return;
      const cellStyle = "padding:8px 12px;text-align:left;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums";
      const headings = data.columns.map(column => `<th scope="col" style="${cellStyle};background:var(--surface-soft)">${escapeHtml(column)}</th>`).join("");
      const rows = data.rows.map(row => `<tr>${data.columns.map(column => `<td style="${cellStyle}">${escapeHtml(row[column] ?? "—")}</td>`).join("")}</tr>`).join("");
      resultTable.innerHTML = `<table style="width:100%;border-collapse:collapse"><thead><tr>${headings}</tr></thead><tbody>${rows}</tbody></table>`;
      resultVerified = true;
      resultStatus.textContent = `已读取 ${data.rows.length} 行${data.truncated ? "（其余行在下载文件中）" : ""}；已核对当前数据集和工作区版本。`;
    } catch (error) {
      if (currentResult(binding, operation)) {
        if (Number(error?.status) === 403) { clearResult(); status.textContent = message(error); sync(); }
        else resultStatus.textContent = message(error);
      }
    } finally {
      if (currentResult(binding, operation)) { readBusy = false; readAbort = null; sync(); }
    }
  }
  resultRefresh.addEventListener("click", previewResult);
  cancelRead.addEventListener("click", () => { clearResult(); status.textContent = "已取消结果读取；已导入的数据保留，未再次执行查询。"; sync(); });
  download.addEventListener("click", async () => {
    if (download.disabled) return;
    const binding = result, operation = ++resultEpoch;
    readAbort?.abort(); readAbort = new AbortController(); readBusy = true;
    resultStatus.textContent = "正在读取文件并核对内容指纹…"; sync();
    try {
      const blob = await readSqlResultDownload(binding, { get, getState: controller.getState, signal: readAbort.signal });
      if (!currentResult(binding, operation)) return;
      saveDownload(blob, binding);
      resultStatus.textContent = "下载文件指纹与确认结果一致，已保存结果数据集。";
    } catch (error) {
      if (currentResult(binding, operation)) {
        if (Number(error?.status) === 403) { clearResult(); status.textContent = message(error); sync(); }
        else { resultVerified = false; resultStatus.textContent = message(error); }
      }
    } finally {
      if (currentResult(binding, operation)) { readBusy = false; readAbort = null; sync(); }
    }
  });

  function invalidate() {
    proposal = null;
    preview.hidden = true;
    preview.textContent = "";
    confirm.disabled = true;
  }
  function currentSource() { return sources.find((item) => item.source_id === sourceSelect.value); }
  function sync() {
    const state = controller.getState();
    const source = currentSource();
    propose.disabled = busy || !taskId || source?.status !== "ready" || state.dirty || !state.serverSnapshot;
    confirm.disabled = busy || !proposal || state.dirty || state.serverSnapshot?.revision !== proposal.workspace_revision;
    sourceSelect.disabled = busy; mode.disabled = busy; input.disabled = busy;
    sourceRefresh.disabled = busy;
    const resultCurrent = matchesSqlResultWorkspace(result, state);
    download.disabled = busy || readBusy || !resultVerified || !resultCurrent;
    resultRefresh.disabled = busy || readBusy || !resultCurrent;
    cancelRead.hidden = !readBusy;
    schema.textContent = source?.status === "ready"
      ? source.tables.map((table) => `${table.table}：${table.columns.map((column) => `${column.name} (${column.dtype})`).join("、")}`).join("；")
      : source?.next_action || "请由管理员在本地配置数据源，然后重新读取。";
  }
  async function loadSources() {
    if (busy) return;
    const task = taskId, operation = ++epoch;
    invalidate(); clearResult();
    if (!task) { sources = []; sourceSelect.innerHTML = ""; sync(); return; }
    status.textContent = "正在读取已配置来源的字段结构…";
    try {
      const response = await get(`/api/tasks/${encodeURIComponent(task)}/sql/sources`);
      if (operation !== epoch || task !== taskId) return;
      sources = response.sources || [];
      sourceSelect.innerHTML = sources.map((item) => `<option value="${escapeHtml(item.source_id)}">${escapeHtml(item.name)}${item.status === "ready" ? "" : " · 不可用"}</option>`).join("");
      status.textContent = sources.length ? "来源已读取。核对字段后生成查询。" : "没有已配置来源，请由管理员在本地配置后刷新。";
    } catch (error) {
      if (operation !== epoch || task !== taskId) return;
      sources = []; sourceSelect.innerHTML = ""; status.textContent = message(error);
    }
    sync();
  }
  function selectionChanged() { epoch += 1; busy = false; invalidate(); clearResult(); sync(); }
  sourceSelect.addEventListener("change", selectionChanged);
  mode.addEventListener("change", selectionChanged);
  input.addEventListener("input", selectionChanged);
  sourceRefresh.addEventListener("click", loadSources);
  propose.addEventListener("click", async () => {
    const state = controller.getState();
    if (propose.disabled || !input.value.trim()) { status.textContent = "请填写查询目标。"; return; }
    const task = taskId, operation = ++epoch;
    busy = true; invalidate(); clearResult(); sync(); status.textContent = "正在生成查询；尚未执行…";
    try {
      const response = await post(`/api/tasks/${encodeURIComponent(task)}/sql/proposals`, {
        source_id: sourceSelect.value, expected_revision: state.serverSnapshot.revision, [mode.value]: input.value,
      });
      if (operation !== epoch || task !== taskId) return;
      if (controller.getState().serverSnapshot?.revision !== response.workspace_revision) throw new Error("工作区版本已变化，请重新生成查询。");
      proposal = response; preview.textContent = response.query; preview.hidden = false;
      status.textContent = `尚未执行。最多 ${response.max_rows} 行、${response.timeout_seconds} 秒；确认后导入并重置当前字段语义。`;
    } catch (error) { if (operation === epoch && task === taskId) status.textContent = message(error); }
    finally { if (operation === epoch && task === taskId) { busy = false; sync(); } }
  });
  confirm.addEventListener("click", async () => {
    if (confirm.disabled) return;
    const task = taskId, frozen = proposal, operation = ++epoch;
    busy = true; sync(); status.textContent = "正在执行已确认查询并导入…";
    try {
      const response = await post(`/api/tasks/${encodeURIComponent(task)}/sql/confirm`, { proposal_token: frozen.proposal_token });
      if (operation !== epoch || task !== taskId) return;
      const binding = sqlResultBinding(response, task);
      invalidate(); await controller.load(task);
      if (operation !== epoch || task !== taskId) return;
      if (!matchesSqlResultWorkspace(binding, controller.getState())) throw new Error("当前数据或工作区版本已变化，请刷新后核对导入结果。");
      result = binding; resultRoot.hidden = false;
      resultMeta.textContent = `${binding.rowCount} 行 · 工作区 R${binding.revision} · 数据集 ${binding.datasetId} · SHA-256 ${binding.contentHash}`;
      status.textContent = `已导入 ${response.evidence.row_count} 行；内容指纹 ${response.evidence.content_hash.slice(0, 12)}。`;
      await previewResult();
      if (operation !== epoch || task !== taskId) return;
      await onImported(response);
    } catch (error) { if (operation === epoch && task === taskId) { invalidate(); status.textContent = message(error); } }
    finally { if (operation === epoch && task === taskId) { busy = false; sync(); } }
  });
  const unsubscribe = controller.subscribe((state) => {
    if (result && !matchesSqlResultWorkspace(result, state)) clearResult();
    if (state.taskId !== taskId) {
      taskId = state.taskId; revision = state.serverSnapshot?.revision ?? null;
      busy = false; input.value = ""; loadSources();
    } else if (state.serverSnapshot?.revision !== revision) {
      revision = state.serverSnapshot?.revision ?? null; invalidate(); sync();
    } else sync();
  });
  sync();
  return { refresh: loadSources, destroy: () => { epoch += 1; clearResult(); unsubscribe(); } };
}
