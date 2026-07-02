import { escapeHtml } from "../ui-utils.js";
import { columnFractions } from "../render-metrics.js";

export function screenNum(value) {
  const n = Number(value);
  return value === null || value === undefined || Number.isNaN(n) ? "n/a" : n.toFixed(4);
}

export function screenPct(value) {
  const n = Number(value);
  return value === null || value === undefined || Number.isNaN(n) ? "n/a" : (n * 100).toFixed(1) + "%";
}

// VD-7: IV tier convention — <0.02 no predictive power, 0.02-0.1 weak, 0.1-0.3
// medium, >0.3 strong (and worth a leakage double-check at very high values).
// Presentation-only reference bands; the backend screen result (INV-1) remains
// the sole source of truth for which features are actually dropped.
const IV_TIERS = [
  { max: 0.02, tier: "none", label: "无区分力" },
  { max: 0.1, tier: "weak", label: "弱" },
  { max: 0.3, tier: "medium", label: "中等" },
  { max: Infinity, tier: "strong", label: "强" },
];

export function ivTier(value) {
  const n = Number(value);
  if (value === null || value === undefined || Number.isNaN(n)) return null;
  return IV_TIERS.find((band) => n < band.max) || IV_TIERS[IV_TIERS.length - 1];
}

export function ivTooltipText(value) {
  const n = Number(value);
  if (value === null || value === undefined || Number.isNaN(n)) {
    return "IV · 无数据\n参考分档：<0.02 无区分力 · 0.02-0.1 弱 · 0.1-0.3 中等 · >0.3 强（>0.5 建议复核是否泄漏）";
  }
  const band = ivTier(n);
  const highNote = n > 0.5 ? "\n△ 数值偏高，建议结合 KS/业务口径复核是否存在泄漏" : "";
  return `IV ${n.toFixed(4)}\n当前：${band.label}\n参考分档：<0.02 无区分力 · 0.02-0.1 弱 · 0.1-0.3 中等 · >0.3 强${highNote}`;
}

// UX-4: page size for the simple pagination fallback at real credit-data scale
// (hundreds to thousands of candidate columns) — chosen over virtual scrolling
// for implementation simplicity/reliability per the review's guidance.
const PAGE_SIZE = 50;

const badges = {
  keep: '<span class="screen-badge keep">入选</span>',
  leakage: '<span class="screen-badge leak">泄漏</span>',
  suspected: '<span class="screen-badge susp">疑似</span>',
  unusable: '<span class="screen-badge unusable">不可用</span>',
};

// Per-message client-side table state (search/sort/filter/pagination + the
// user's edited checkbox picks). Keyed by message id so switching between
// gate messages (or re-rendering the same one) does not bleed state across
// tables. Pure UI state — never sent to the backend; confirm still POSTs the
// checked feature set exactly as before.
const tableState = new Map();

function defaultState() {
  return {
    query: "",
    sortKey: null,
    sortDir: "desc",
    chip: "all",
    page: 1,
    checkedOverrides: new Map(),
    leakageReason: "",
  };
}

function getState(messageId) {
  let state = tableState.get(messageId);
  if (!state) {
    state = defaultState();
    tableState.set(messageId, state);
  }
  return state;
}

function buildRows(screen, interactive) {
  const scores = screen.scores && typeof screen.scores === "object" ? screen.scores : {};
  const selectedSet = new Set((screen.selected || []).map((value) => String(value)));
  const watchSet = new Set();
  for (const key of ["leakage_watch", "ks_decay_watch", "psi_watch", "split_shift"]) {
    for (const item of screen[key] || []) {
      const name = Array.isArray(item) ? item[0] : item;
      if (name !== undefined) watchSet.add(String(name));
    }
  }
  const categoricalSet = new Set();
  for (const key of ["excluded_categorical", "suspected_categorical"]) {
    for (const item of screen[key] || []) {
      const name = item && typeof item === "object" ? item.column : item;
      if (name !== undefined) categoricalSet.add(String(name));
    }
  }
  const sentinelColumns = screen.sentinel_columns && typeof screen.sentinel_columns === "object"
    ? screen.sentinel_columns
    : {};
  for (const name of Object.keys(sentinelColumns)) categoricalSet.add(String(name));

  const tuple = (item) => (Array.isArray(item) ? item : [item]);
  const rows = [];
  const seen = new Set();
  const pushRow = (feature, ks, category) => {
    const name = String(feature);
    if (seen.has(name)) return;
    seen.add(name);
    const stats = scores[name] && typeof scores[name] === "object" ? scores[name] : {};
    const ksValue = ks === undefined || ks === null ? stats.ks : ks;
    rows.push({
      name,
      category,
      ks: ksValue === undefined ? null : ksValue,
      iv: stats.iv === undefined ? null : stats.iv,
      missingRate: stats.missing_rate === undefined ? null : stats.missing_rate,
      coverage: stats.coverage === undefined ? null : stats.coverage,
      ksDecay: stats.ks_decay === undefined ? null : stats.ks_decay,
      psiSplit: stats.psi_split === undefined ? null : stats.psi_split,
      checked: selectedSet.has(name),
      disabled: category === "unusable" || !interactive, // constant/sparse: no signal to select; also disabled when read-only
      isWatch: watchSet.has(name),
      isCategorical: categoricalSet.has(name),
    });
  };
  for (const feature of screen.selected || []) pushRow(feature, undefined, "keep");
  for (const item of screen.leakage || []) pushRow(tuple(item)[0], tuple(item)[1], "leakage");
  for (const item of screen.suspected || []) pushRow(tuple(item)[0], tuple(item)[1], "suspected");
  for (const item of screen.unusable || []) pushRow(tuple(item)[0], null, "unusable");
  return rows;
}

const CHIP_DEFS = [
  { key: "all", label: "全部" },
  { key: "selected", label: "已选" },
  { key: "leakage", label: "泄漏嫌疑" },
  { key: "watch", label: "watch" },
  { key: "low_coverage", label: "低覆盖" },
  { key: "categorical", label: "类别列" },
];

const _LOW_COVERAGE = 0.5;

function matchesChip(row, chip) {
  switch (chip) {
    case "selected":
      return row.checked;
    case "leakage":
      return row.category === "leakage" || row.category === "suspected";
    case "watch":
      return row.isWatch;
    case "low_coverage":
      return row.coverage !== null && row.coverage < _LOW_COVERAGE;
    case "categorical":
      return row.isCategorical;
    case "all":
    default:
      return true;
  }
}

function applyCheckedOverrides(rows, overrides) {
  return rows.map((row) => (
    overrides.has(row.name) ? { ...row, checked: overrides.get(row.name) } : row
  ));
}

function filterRows(rows, state) {
  const query = state.query.trim().toLowerCase();
  return rows.filter((row) => {
    if (query && !row.name.toLowerCase().includes(query)) return false;
    return matchesChip(row, state.chip);
  });
}

const SORT_ACCESSORS = {
  name: (row) => row.name,
  ks: (row) => row.ks,
  iv: (row) => row.iv,
  missing_rate: (row) => row.missingRate,
  coverage: (row) => row.coverage,
  ks_decay: (row) => row.ksDecay,
  psi_split: (row) => row.psiSplit,
};

function sortRows(rows, state) {
  if (!state.sortKey || !SORT_ACCESSORS[state.sortKey]) return rows;
  const accessor = SORT_ACCESSORS[state.sortKey];
  const dir = state.sortDir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const av = accessor(a);
    const bv = accessor(b);
    const aMissing = av === null || av === undefined || (Number.isNaN(Number(av)) && typeof av !== "string");
    const bMissing = bv === null || bv === undefined || (Number.isNaN(Number(bv)) && typeof bv !== "string");
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1; // n/a values sink to the bottom regardless of direction
    if (bMissing) return -1;
    if (typeof av === "string" || typeof bv === "string") {
      return dir * String(av).localeCompare(String(bv));
    }
    return dir * (Number(av) - Number(bv));
  });
}

function categoryCounts(rows) {
  const counts = { keep: 0, leakage: 0, suspected: 0, unusable: 0 };
  for (const row of rows) counts[row.category] = (counts[row.category] || 0) + 1;
  return counts;
}

function sortIndicator(state, key) {
  if (state.sortKey !== key) return "";
  return state.sortDir === "asc" ? " ▲" : " ▼";
}

function sortableHeader(label, key, state) {
  return `<th><button type="button" class="screen-sort-btn" data-screen-sort="${escapeHtml(key)}" aria-label="按${escapeHtml(label)}排序">${escapeHtml(label)}${sortIndicator(state, key)}</button></th>`;
}

function databarCell(value, fraction, tip) {
  const text = screenNum(value);
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) {
    return `<span class="screen-num-plain">${escapeHtml(text)}</span>`;
  }
  return `<span class="databar screen-databar" data-tip="${escapeHtml(tip)}" style="--fraction:${Math.max(0, Math.min(1, fraction)).toFixed(4)}">`
    + `<span class="databar-fill"></span>`
    + `<span class="databar-label">${escapeHtml(text)}</span>`
    + `</span>`;
}

function ivTierBadge(value) {
  const band = ivTier(value);
  if (!band) return "";
  return `<span class="iv-tier-badge" data-tier="${escapeHtml(band.tier)}" title="${escapeHtml(ivTooltipText(value))}">${escapeHtml(band.label)}</span>`;
}

function renderRow(row, fractions, index) {
  const rowClasses = [`screen-row`, `screen-${row.category}`];
  if (row.isWatch) rowClasses.push("screen-watch");
  const ksTip = `KS ${screenNum(row.ks)}`;
  const ivTip = ivTooltipText(row.iv);
  return `<tr class="${rowClasses.join(" ")}" data-screen-feature="${escapeHtml(row.name)}">
      <td class="screen-pick-cell"><input type="checkbox" class="screen-pick" value="${escapeHtml(row.name)}"${row.checked ? " checked" : ""}${row.disabled ? " disabled" : ""} /></td>
      <td class="screen-feat">${escapeHtml(row.name)}</td>
      <td class="screen-num">${databarCell(row.ks, fractions.ks.get(index), ksTip)}</td>
      <td class="screen-num">${databarCell(row.iv, fractions.iv.get(index), ivTip)}${ivTierBadge(row.iv)}</td>
      <td class="screen-num">${escapeHtml(screenPct(row.missingRate))}</td>
      <td class="screen-num">${escapeHtml(screenPct(row.coverage))}</td>
      <td class="screen-num">${escapeHtml(screenNum(row.ksDecay))}</td>
      <td class="screen-num">${escapeHtml(screenNum(row.psiSplit))}</td>
      <td>${badges[row.category] || ""}${row.isWatch ? '<span class="screen-badge watch">watch</span>' : ""}</td>
    </tr>`;
}

function paginationHtml(page, totalPages) {
  if (totalPages <= 1) return "";
  return `<div class="screen-pagination">
    <button type="button" class="button compact secondary screen-page-prev" data-screen-page-prev="1"${page <= 1 ? " disabled" : ""}>上一页</button>
    <span class="screen-page-status">第 ${page} / ${totalPages} 页</span>
    <button type="button" class="button compact secondary screen-page-next" data-screen-page-next="1"${page >= totalPages ? " disabled" : ""}>下一页</button>
  </div>`;
}

function chipsHtml(rows, state) {
  const counts = {
    all: rows.length,
    selected: rows.filter((row) => row.checked).length,
    leakage: rows.filter((row) => row.category === "leakage" || row.category === "suspected").length,
    watch: rows.filter((row) => row.isWatch).length,
    low_coverage: rows.filter((row) => row.coverage !== null && row.coverage < _LOW_COVERAGE).length,
    categorical: rows.filter((row) => row.isCategorical).length,
  };
  return CHIP_DEFS.map(({ key, label }) => (
    `<button type="button" class="screen-chip${state.chip === key ? " active" : ""}" data-screen-chip="${escapeHtml(key)}" aria-pressed="${state.chip === key ? "true" : "false"}">${escapeHtml(label)} <b>${counts[key]}</b></button>`
  )).join("");
}

function tableBodyHtml(message, options = {}) {
  const screen = message?.metadata?.screen;
  if (!screen || typeof screen !== "object") return null;
  const messageId = message?.id ? String(message.id) : "";
  const interactive = options.interactive !== false;
  const state = getState(messageId);
  const allRows = applyCheckedOverrides(buildRows(screen, interactive), state.checkedOverrides);
  const totalCounts = categoryCounts(allRows);
  const filtered = sortRows(filterRows(allRows, state), state);
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const page = Math.min(Math.max(1, state.page), totalPages);
  state.page = page;
  const pageRows = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  // Databar fractions are normalized across the *filtered* set that is actually
  // visible, using the shared render-metrics.js column-max normalization so the
  // bar language matches the report tables (VD-7/VD-1).
  const ksRows = filtered.map((row) => [row.ks]);
  const ivRows = filtered.map((row) => [row.iv]);
  const fractions = { ks: columnFractions(ksRows, 0), iv: columnFractions(ivRows, 0) };
  const pageStartIndex = (page - 1) * PAGE_SIZE;
  const rowsHtml = pageRows.map((row, i) => renderRow(row, fractions, pageStartIndex + i)).join("");
  const selectedCount = allRows.filter((row) => row.checked).length;
  const checkedLeakage = allRows.some((row) => row.checked && (row.category === "leakage" || row.category === "suspected"));
  const disabledAttr = interactive ? "" : " disabled aria-disabled=\"true\"";
  return {
    screen,
    messageId,
    interactive,
    state,
    allRows,
    filtered,
    totalCounts,
    totalPages,
    page,
    rowsHtml,
    selectedCount,
    checkedLeakage,
    disabledAttr,
  };
}

export function renderScreenGateTable(message, options = {}) {
  const built = tableBodyHtml(message, options);
  if (!built) return "";
  const { screen, messageId, interactive, state, totalCounts, totalPages, page, rowsHtml, selectedCount, checkedLeakage, disabledAttr } = built;
  const gateStepId = message?.metadata?.step_id ? String(message.metadata.step_id) : "";
  const thresholds = screen.thresholds && typeof screen.thresholds === "object" ? screen.thresholds : {};
  const leakageKs = thresholds.leakage_ks ?? 0.4;
  const maxMissingRate = thresholds.max_missing_rate ?? 0.95;
  const note = interactive
    ? `共筛 ${screen.n_screened ?? built.allRows.length} 列；泄漏阈值 KS≥${leakageKs}。勾选=入选，可硬选泄漏/疑似列；确认后用所选特征训练。`
    : `共筛 ${screen.n_screened ?? built.allRows.length} 列；泄漏阈值 KS≥${leakageKs}。这是历史筛选结果，如需调整请使用最新待确认步骤。`;
  const thresholdControls = `<div class="screen-threshold-controls">
    <label>泄漏KS <input type="number" class="screen-threshold-input" data-screen-threshold="leakage_ks" min="0" max="1" step="0.01" value="${escapeHtml(String(leakageKs))}"${disabledAttr} required /></label>
    <label>最大缺失率 <input type="number" class="screen-threshold-input" data-screen-threshold="max_missing_rate" min="0" max="1" step="0.01" value="${escapeHtml(String(maxMissingRate))}"${disabledAttr} required /></label>
    <button type="button" class="button compact secondary screen-adjust"${interactive ? ` data-screen-adjust="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "重算" : "已归档"}</button>
  </div>`;
  const toolbarHtml = `<div class="screen-toolbar">
    <input type="search" class="screen-search-input" data-screen-search="${escapeHtml(messageId)}" placeholder="按特征名搜索…" value="${escapeHtml(state.query)}"${interactive ? "" : " disabled"} aria-label="按特征名筛选" />
    <div class="screen-chips" role="group" aria-label="按类别筛选" data-screen-chip-group="${escapeHtml(messageId)}">${chipsHtml(built.allRows, state)}</div>
  </div>`;
  const summaryHtml = `<div class="screen-summary">
    <span class="screen-summary-item">入选 ${totalCounts.keep}</span>
    <span class="screen-summary-item">泄漏 ${totalCounts.leakage}</span>
    <span class="screen-summary-item">疑似 ${totalCounts.suspected}</span>
    <span class="screen-summary-item">不可用 ${totalCounts.unusable}</span>
    <span class="screen-summary-item screen-selected-count">已选 ${selectedCount}/${built.allRows.length}</span>
  </div>`;
  const bulkHtml = `<div class="screen-bulk-actions" data-screen-bulk-group="${escapeHtml(messageId)}">
    <button type="button" class="button compact secondary screen-bulk-select-visible" data-screen-bulk="select_visible"${interactive ? "" : " disabled"}>全选可见</button>
    <button type="button" class="button compact secondary screen-bulk-clear-visible" data-screen-bulk="clear_visible"${interactive ? "" : " disabled"}>清空可见</button>
    <button type="button" class="button compact secondary screen-bulk-invert-visible" data-screen-bulk="invert_visible"${interactive ? "" : " disabled"}>反选可见</button>
  </div>`;
  const leakageReasonHtml = interactive ? `<div class="screen-leakage-reason"${checkedLeakage ? "" : ' hidden'}>
    <label>泄漏/疑似列覆盖理由（勾选泄漏或疑似列时必填）
      <textarea class="screen-leakage-reason-input" rows="2" placeholder="说明为何仍需强选该列（例如：已核实非未来信息、口径已确认）">${escapeHtml(state.leakageReason)}</textarea>
    </label>
  </div>` : "";
  return `<div class="screen-table-wrap" data-screen-form="${escapeHtml(messageId)}" data-screen-step-id="${escapeHtml(gateStepId)}"${interactive ? "" : ' data-screen-readonly="true"'}>
    ${thresholdControls}
    ${toolbarHtml}
    ${summaryHtml}
    ${bulkHtml}
    <div class="screen-table-scroll">
      <table class="screen-table">
        <thead><tr><th>选</th>${sortableHeader("特征", "name", state)}${sortableHeader("KS", "ks", state)}${sortableHeader("IV", "iv", state)}${sortableHeader("缺失率", "missing_rate", state)}${sortableHeader("覆盖率", "coverage", state)}${sortableHeader("KS衰减", "ks_decay", state)}${sortableHeader("PSI", "psi_split", state)}<th>类别</th></tr></thead>
        <tbody>${rowsHtml || '<tr class="screen-empty-row"><td colspan="9">没有匹配的特征</td></tr>'}</tbody>
      </table>
    </div>
    ${paginationHtml(page, totalPages)}
    ${leakageReasonHtml}
    <div class="screen-table-foot">
      <span class="screen-note">${escapeHtml(note)}</span>
      <button type="button" class="button compact primary screen-confirm"${interactive ? ` data-screen-confirm="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "确认所选特征" : "历史结果"}</button>
    </div>
  </div>`;
}

// Re-renders just the wrap's markup in place (search/sort/chip/page/bulk
// interactions never touch the backend — only Confirm/Adjust do) via a plain
// `outerHTML` replace, and restores focus to the search box when it was
// focused, since search-as-you-type would otherwise lose focus on every
// keystroke's re-render. The actual DOM write goes through `applyRerender`
// (default: `wrap.outerHTML = html`) so tests can inject a capture-only stub
// instead of needing a full browser DOM.
function defaultApplyRerender(wrap, html) {
  wrap.outerHTML = html;
}

function rerenderWrap(wrap, message, options, context = {}) {
  if (!wrap) return;
  const searchInput = typeof wrap.querySelector === "function" ? wrap.querySelector(".screen-search-input") : null;
  const searchHadFocus = typeof document !== "undefined" && searchInput && document.activeElement === searchInput;
  const selectionStart = searchHadFocus ? searchInput.selectionStart : null;
  const html = renderScreenGateTable(message, options);
  const apply = typeof context.applyRerender === "function" ? context.applyRerender : defaultApplyRerender;
  apply(wrap, html);
  if (searchHadFocus && wrap.parentNode) {
    const input = wrap.parentNode.querySelector(".screen-search-input");
    if (input) {
      input.focus();
      if (selectionStart !== null && typeof input.setSelectionRange === "function") {
        input.setSelectionRange(selectionStart, selectionStart);
      }
    }
  }
}

function findMessage(messageId, context) {
  const messages = typeof context.getAgentMessages === "function" ? context.getAgentMessages() : context.agentMessages;
  return (messages || []).find((message) => String(message?.id || "") === messageId) || null;
}

// Before a re-render collapses/rebuilds rows, capture the current DOM checkbox
// state into the per-message override map so search/sort/filter/page changes
// never silently discard picks the user already made on a different page/filter.
function captureCheckedState(wrap, state) {
  for (const box of wrap.querySelectorAll(".screen-pick")) {
    state.checkedOverrides.set(box.value, box.checked);
  }
}

export function handleScreenSearchInput(event, context = {}) {
  const input = event.target?.closest?.("[data-screen-search]");
  if (!input) return false;
  const wrap = input.closest(".screen-table-wrap");
  if (!wrap) return false;
  const messageId = wrap.dataset.screenForm || "";
  const message = findMessage(messageId, context);
  if (!message) return false;
  const state = getState(messageId);
  captureCheckedState(wrap, state);
  state.query = input.value || "";
  state.page = 1;
  rerenderWrap(wrap, message, { interactive: wrap.dataset.screenReadonly !== "true" }, context);
  return true;
}

export function handleScreenSortClick(event, context = {}) {
  const button = event.target?.closest?.("[data-screen-sort]");
  if (!button) return false;
  event.preventDefault();
  const wrap = button.closest(".screen-table-wrap");
  if (!wrap) return false;
  const messageId = wrap.dataset.screenForm || "";
  const message = findMessage(messageId, context);
  if (!message) return false;
  const state = getState(messageId);
  captureCheckedState(wrap, state);
  const key = button.getAttribute("data-screen-sort");
  if (state.sortKey === key) {
    state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
  } else {
    state.sortKey = key;
    state.sortDir = "desc";
  }
  rerenderWrap(wrap, message, { interactive: wrap.dataset.screenReadonly !== "true" }, context);
  return true;
}

export function handleScreenChipClick(event, context = {}) {
  const button = event.target?.closest?.("[data-screen-chip]");
  if (!button) return false;
  event.preventDefault();
  const wrap = button.closest(".screen-table-wrap");
  if (!wrap) return false;
  const messageId = wrap.dataset.screenForm || "";
  const message = findMessage(messageId, context);
  if (!message) return false;
  const state = getState(messageId);
  captureCheckedState(wrap, state);
  state.chip = button.getAttribute("data-screen-chip") || "all";
  state.page = 1;
  rerenderWrap(wrap, message, { interactive: wrap.dataset.screenReadonly !== "true" }, context);
  return true;
}

export function handleScreenPageClick(event, context = {}) {
  const prevButton = event.target?.closest?.("[data-screen-page-prev]");
  const nextButton = event.target?.closest?.("[data-screen-page-next]");
  const button = prevButton || nextButton;
  if (!button) return false;
  event.preventDefault();
  const wrap = button.closest(".screen-table-wrap");
  if (!wrap) return false;
  const messageId = wrap.dataset.screenForm || "";
  const message = findMessage(messageId, context);
  if (!message) return false;
  const state = getState(messageId);
  captureCheckedState(wrap, state);
  state.page += prevButton ? -1 : 1;
  rerenderWrap(wrap, message, { interactive: wrap.dataset.screenReadonly !== "true" }, context);
  return true;
}

export function handleScreenBulkClick(event, context = {}) {
  const button = event.target?.closest?.("[data-screen-bulk]");
  if (!button) return false;
  event.preventDefault();
  const wrap = button.closest(".screen-table-wrap");
  if (!wrap) return false;
  const action = button.getAttribute("data-screen-bulk");
  // Bulk-selecting a leakage/suspected feature is a deliberate hard-select the
  // review calls for a one-time confirmation on; window.confirm keeps this a
  // pure frontend change with no new backend contract.
  if (action === "select_visible") {
    const hasLeakage = Array.from(wrap.querySelectorAll(".screen-row.screen-leakage, .screen-row.screen-suspected")).length > 0;
    if (hasLeakage && typeof window !== "undefined" && typeof window.confirm === "function") {
      const proceed = window.confirm("当前可见范围包含泄漏/疑似列，全选将把它们一并选中，是否继续？");
      if (!proceed) return true;
    }
  }
  const messageId = wrap.dataset.screenForm || "";
  const message = findMessage(messageId, context);
  if (!message) return false;
  const state = getState(messageId);
  captureCheckedState(wrap, state);
  for (const box of wrap.querySelectorAll(".screen-pick")) {
    if (box.disabled) continue;
    if (action === "select_visible") state.checkedOverrides.set(box.value, true);
    else if (action === "clear_visible") state.checkedOverrides.set(box.value, false);
    else if (action === "invert_visible") state.checkedOverrides.set(box.value, !box.checked);
  }
  rerenderWrap(wrap, message, { interactive: wrap.dataset.screenReadonly !== "true" }, context);
  return true;
}

// Keeps the "已选 M/N" summary and the leakage-override-reason visibility live
// as the user ticks/unticks individual checkboxes, without a full re-render
// (a full re-render on every checkbox click would be disruptive while
// scanning a long page of rows).
export function handleScreenPickChange(event, context = {}) {
  const box = event.target?.closest?.(".screen-pick");
  if (!box) return false;
  const wrap = box.closest(".screen-table-wrap");
  if (!wrap) return false;
  const messageId = wrap.dataset.screenForm || "";
  const state = getState(messageId);
  state.checkedOverrides.set(box.value, box.checked);
  const total = wrap.querySelectorAll(".screen-pick").length;
  const checkedCount = wrap.querySelectorAll(".screen-pick:checked").length;
  const summary = wrap.querySelector(".screen-selected-count");
  if (summary) summary.textContent = `已选 ${checkedCount}/${total}`;
  const row = box.closest(".screen-row");
  const isLeakageRow = row && (row.classList.contains("screen-leakage") || row.classList.contains("screen-suspected"));
  if (isLeakageRow) {
    const anyLeakageChecked = Array.from(wrap.querySelectorAll(".screen-row.screen-leakage .screen-pick, .screen-row.screen-suspected .screen-pick"))
      .some((input) => input.checked);
    const reasonBlock = wrap.querySelector(".screen-leakage-reason");
    if (reasonBlock) reasonBlock.hidden = !anyLeakageChecked;
  }
  return true;
}

function screenGateContext(context = {}) {
  return {
    taskId: typeof context.getSelectedTaskId === "function"
      ? context.getSelectedTaskId()
      : context.selectedTaskId,
    api: context.api,
    acceptanceMode: typeof context.agentAcceptanceModeValue === "function"
      ? context.agentAcceptanceModeValue()
      : context.acceptanceMode,
    setActionStatus: context.setActionStatus || (() => {}),
    setAgentMessages: context.setAgentMessages || (() => {}),
    renderAgentConversation: context.renderAgentConversation || (() => {}),
    pollAgentMessagesUntilSettled: context.pollAgentMessagesUntilSettled || (() => Promise.resolve()),
    resetFetchThrottle: context.resetFetchThrottle || (() => {}),
    renderWorkflowStepper: context.renderWorkflowStepper || (() => {}),
  };
}

// UX-1: screen-gate submissions rerun the driver turn (now job-wrapped, REL-1)
// and can take a while (recompute screening / retrain downstream). Give
// immediate busy feedback, poll agent messages so intermediate step content
// streams in, and force the plan rail to re-fetch on a short interval.
function withDriverTurnBusyFeedback(taskId, context, run) {
  const { setActionStatus, pollAgentMessagesUntilSettled, resetFetchThrottle, renderWorkflowStepper } = context;
  setActionStatus("正在执行下一步…", "busy");
  let planRailTimer = null;
  if (typeof setInterval === "function") {
    planRailTimer = setInterval(() => {
      resetFetchThrottle(taskId);
      renderWorkflowStepper({ force: true });
    }, 1500);
  }
  const stopPlanRailTicker = () => {
    if (planRailTimer !== null) clearInterval(planRailTimer);
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  };
  return run(pollAgentMessagesUntilSettled).finally(stopPlanRailTicker);
}

export async function submitScreenThresholdAdjust(button, rawContext = {}) {
  const wrap = button.closest(".screen-table-wrap");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = screenGateContext(rawContext);
  if (!wrap || !taskId || typeof api !== "function") return;
  if (wrap.dataset.screenReadonly === "true") {
    setActionStatus("这是历史筛选结果，请使用最新待确认步骤调整。", "error");
    return;
  }
  const adjustParams = {};
  for (const input of wrap.querySelectorAll(".screen-threshold-input")) {
    const key = input.getAttribute("data-screen-threshold");
    if (!key) continue;
    const rawValue = String(input.value || "").trim();
    if (!rawValue) {
      setActionStatus("阈值不能为空。", "error");
      return;
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value) || value < 0 || value > 1) {
      setActionStatus("阈值需在 0 到 1 之间。", "error");
      return;
    }
    adjustParams[key] = value;
  }
  if (!Object.keys(adjustParams).length) return;
  const expectedStepId = wrap.dataset.screenStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息，请刷新后重试。", "error");
    return;
  }
  button.disabled = true;
  const context = screenGateContext(rawContext);
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: "调整筛选阈值",
          adjust_params: adjustParams,
          expected_step_id: expectedStepId,
          acceptance_mode: acceptanceMode,
        }),
      });
      const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
      const result = await requestPromise;
      await streamPollPromise;
      setAgentMessages(result.messages);
      renderAgentConversation();
    });
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "重算特征筛选失败", "error");
  }
}

export async function submitScreenSelection(button, rawContext = {}) {
  const wrap = button.closest(".screen-table-wrap");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = screenGateContext(rawContext);
  if (!wrap || !taskId || typeof api !== "function") return;
  if (wrap.dataset.screenReadonly === "true") {
    setActionStatus("这是历史筛选结果，请使用最新待确认步骤确认。", "error");
    return;
  }
  const selection = [];
  let hasLeakagePick = false;
  for (const box of wrap.querySelectorAll(".screen-pick:checked")) {
    if (box.disabled) continue;
    selection.push(box.value);
    const row = box.closest(".screen-row");
    if (row && (row.classList.contains("screen-leakage") || row.classList.contains("screen-suspected"))) {
      hasLeakagePick = true;
    }
  }
  if (!selection.length) {
    setActionStatus("请至少勾选一个特征。", "error");
    return;
  }
  // UX-4: forcing a leakage/suspected column into the model is a deliberate
  // override of the screen's own hard-cut classification, so it requires a
  // written reason — folded into the confirm message text since the backend
  // confirm contract (content/selection/expected_step_id) already carries
  // free-form content and needs no schema change for this.
  let leakageReason = "";
  if (hasLeakagePick) {
    const reasonInput = wrap.querySelector(".screen-leakage-reason-input");
    leakageReason = String(reasonInput?.value || "").trim();
    if (leakageReason.length < 4) {
      setActionStatus("勾选了泄漏/疑似列，请先填写覆盖理由（至少4个字）。", "error");
      return;
    }
  }
  const expectedStepId = wrap.dataset.screenStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息，请刷新后重试。", "error");
    return;
  }
  button.disabled = true;
  const context = screenGateContext(rawContext);
  const content = leakageReason ? `确认（泄漏/疑似列覆盖理由：${leakageReason}）` : "确认";
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content,
          selection,
          expected_step_id: expectedStepId,
          acceptance_mode: acceptanceMode,
        }),
      });
      const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
      const result = await requestPromise;
      await streamPollPromise;
      setAgentMessages(result.messages);
      renderAgentConversation();
    });
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "确认所选特征失败", "error");
  }
}

export function handleScreenAdjustClick(event, context = {}) {
  const button = event.target?.closest?.("[data-screen-adjust]");
  if (!button) return false;
  event.preventDefault();
  void submitScreenThresholdAdjust(button, context);
  return true;
}

export function handleScreenConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-screen-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitScreenSelection(button, context);
  return true;
}
