// Result summary (2026-10-19, plan batch 6 follow-up): the headline of a
// workflow result card — metric tiles and one bar chart — taken from the
// message's own platform tables, in the mockup's evidence-card form.
//
// Rules:
// - Only tables named in the registry below are summarised; their titles are
//   platform-owned, so the mapping is deterministic.
// - Every value is the table cell's own text; bar lengths only compare cells
//   within one column. Nothing is recomputed, rounded or judged here — the
//   full tables stay below the summary.

import { escapeHtml } from "./ui-utils.js";

const MAX_TILES = 4;
const MAX_BARS = 10;
const FIELD_COLUMNS = /^(特征|特征A|特征B|字段|变量|列|feature|field|column)$/i;

// Metric tables (two columns: name, value). `keys` orders the rows that make
// the best headline; tables without a listed key use their first rows.
// Each metric is listed by its machine key and by the business label the
// platform renderers now print in the name column (``oot_ks`` / ``OOT KS``).
const MODEL_METRIC_KEYS = [
  "oot_ks", "OOT KS", "oot_auc", "OOT AUC", "test_ks", "测试集 KS", "test_auc", "测试集 AUC",
  "oot_macro_auc", "OOT Macro-AUC", "test_macro_auc", "测试集 Macro-AUC",
  "oot_rmse", "OOT RMSE", "test_rmse", "测试集 RMSE", "oot_mae", "OOT MAE", "test_mae", "测试集 MAE",
  "train_ks", "训练集 KS", "train_auc", "训练集 AUC",
];

// Checked in this order: the final model's metrics win over a trial's.
const TILE_TABLES = [
  { title: /^最终模型指标/, keys: MODEL_METRIC_KEYS },
  { title: /^模型指标/, keys: MODEL_METRIC_KEYS },
  { title: /^最优 trial 指标/, keys: MODEL_METRIC_KEYS },
  { title: /^策略回测摘要/, keys: ["审批率", "通过坏率", "拒绝率", "通过人数", "拒绝人数"] },
  { title: /^核心指标/, keys: [] },
  { title: /^关键数字/, keys: [] },
];

// Bar tables: `label` and `value` name columns (header text, or an index);
// `note` adds a second cell beside the value.
const BAR_TABLES = [
  { title: /^切分计数/, label: 0, value: "行数", note: "占比" },
  { title: /^特征指标$/, label: "特征", value: "IV" },
  { title: /^(入选特征|最终清单)/, label: "特征", value: ["IV", "KS"] },
  { title: /^候选模型对比/, label: "算法", value: ["oot_ks", "OOT KS"] },
  { title: /^参考 MOB 坏账率/, label: "cohort", value: "坏账率" },
  { title: /^(按决策分组|独立回放总体动作)/, label: 0, value: "样本数", note: "坏率" },
  { title: /^总体动作与风险影响/, label: "action", value: "count", note: "bad_rate" },
  { title: /^各特征表贡献/, label: "特征表", value: "命中率" },
];

const METRIC_LABELS = {
  oot_ks: "KS · OOT",
  test_ks: "KS · 测试",
  train_ks: "KS · 训练",
  oot_auc: "AUC · OOT",
  test_auc: "AUC · 测试",
  train_auc: "AUC · 训练",
  oot_macro_auc: "Macro AUC · OOT",
  test_macro_auc: "Macro AUC · 测试",
  oot_rmse: "RMSE · OOT",
  test_rmse: "RMSE · 测试",
  oot_mae: "MAE · OOT",
  test_mae: "MAE · 测试",
  product_count: "产品数",
  analysis_slice_count: "分析切片数",
};

function cellText(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return String(value.text ?? value.value ?? "");
  return String(value);
}

function headerText(column) {
  if (column && typeof column === "object") return String(column.label ?? column.key ?? "");
  return String(column ?? "");
}

// "6.0%" -> 6, "1,234" -> 1234, "0.9044" -> 0.9044; "n/a" and words -> null.
export function numericCell(value) {
  const text = cellText(value).trim().replace(/,/g, "");
  if (!/^-?\d+(\.\d+)?%?$/.test(text)) return null;
  const number = Number.parseFloat(text);
  return Number.isFinite(number) ? number : null;
}

function columnIndex(columns, spec) {
  if (typeof spec === "number") return spec < columns.length ? spec : -1;
  const wanted = Array.isArray(spec) ? spec : [spec];
  for (const name of wanted) {
    const index = columns.findIndex((column) => headerText(column) === name);
    if (index >= 0) return index;
  }
  return -1;
}

function tableRows(table) {
  return Array.isArray(table?.rows) ? table.rows.map((row) => (Array.isArray(row) ? row : [row])) : [];
}

// A tile shows one number or a short word; a cell holding a whole record (a
// serialised dict, a sentence) is left to the table below.
function tileWorthy(value) {
  const text = cellText(value).trim();
  if (!text) return false;
  if (numericCell(text) !== null) return true;
  return text.length <= 12 && !/[{}[\]]/.test(text);
}

function summaryTiles(table, spec) {
  const rows = tableRows(table).filter((row) => row.length >= 2 && tileWorthy(row[1]));
  if (!rows.length) return [];
  const byKey = new Map(rows.map((row) => [cellText(row[0]).trim(), row]));
  const ordered = spec.keys.map((key) => byKey.get(key)).filter(Boolean);
  const picked = (ordered.length ? ordered : rows).slice(0, MAX_TILES);
  return picked.map((row) => {
    const key = cellText(row[0]).trim();
    return { label: METRIC_LABELS[key] || key, value: cellText(row[1]).trim() };
  });
}

function summaryBars(table, spec) {
  const columns = Array.isArray(table?.columns) ? table.columns : [];
  const labelAt = columnIndex(columns, spec.label);
  const valueAt = columnIndex(columns, spec.value);
  if (labelAt < 0 || valueAt < 0) return null;
  const noteAt = spec.note === undefined ? -1 : columnIndex(columns, spec.note);
  // V3 5.7: a field column's codes carry the table's confirmed business names
  // (structured `field_labels`; display only, the label stays the code).
  const labels = FIELD_COLUMNS.test(headerText(columns[labelAt]).trim())
    && table?.field_labels && typeof table.field_labels === "object" ? table.field_labels : {};
  const aliasOf = (code) => {
    const value = Object.prototype.hasOwnProperty.call(labels, code) ? labels[code] : "";
    return typeof value === "string" ? value : "";
  };
  const items = tableRows(table)
    .slice(0, MAX_BARS)
    .map((row) => ({
      label: cellText(row[labelAt]).trim(),
      text: cellText(row[valueAt]).trim(),
      number: numericCell(row[valueAt]),
      note: noteAt >= 0 ? cellText(row[noteAt]).trim() : "",
      alias: aliasOf(cellText(row[labelAt]).trim()),
    }))
    .filter((item) => item.label);
  const max = Math.max(0, ...items.map((item) => item.number ?? 0));
  // One bar compares nothing; the table already says it.
  const measured = items.filter((item) => item.number !== null);
  if (measured.length < 2 || max <= 0) return null;
  return {
    title: String(table?.title || ""),
    valueLabel: headerText(columns[valueAt]),
    noteLabel: noteAt >= 0 ? headerText(columns[noteAt]) : "",
    items: items.map((item) => ({
      ...item,
      ratio: item.number === null || item.number <= 0 ? 0 : item.number / max,
    })),
  };
}

export function resultSummaryParts(tables = []) {
  const list = Array.isArray(tables) ? tables : [];
  let tiles = [];
  let tileSource = "";
  for (const spec of TILE_TABLES) {
    const table = list.find((item) => spec.title.test(String(item?.title || "")));
    if (!table) continue;
    tiles = summaryTiles(table, spec);
    if (tiles.length) {
      tileSource = String(table.title || "");
      break;
    }
  }
  let bars = null;
  for (const table of list) {
    const spec = BAR_TABLES.find((item) => item.title.test(String(table?.title || "")));
    if (!spec) continue;
    bars = summaryBars(table, spec);
    if (bars) break;
  }
  return { tiles, tileSource, bars };
}

export function resultSummaryHtml(tables = []) {
  const { tiles, tileSource, bars } = resultSummaryParts(tables);
  if (!tiles.length && !bars) return "";
  const tilesHtml = tiles.length
    ? [
      // Say which table the numbers are from (a trial's metrics vs the final model's).
      `<p class="result-headline-source">${escapeHtml(tileSource)}</p>`,
      `<div class="result-headline-tiles" role="list" aria-label="${escapeHtml(tileSource)}">`,
      ...tiles.map((tile) => [
        '<div class="result-headline-tile" role="listitem">',
        `<span title="${escapeHtml(tile.label)}">${escapeHtml(tile.label)}</span>`,
        `<b class="num" title="${escapeHtml(tile.value)}">${escapeHtml(tile.value)}</b>`,
        "</div>",
      ].join("")),
      "</div>",
    ].join("")
    : "";
  const barsHtml = bars
    ? [
      '<figure class="result-headline-bars">',
      `<figcaption><b>${escapeHtml(bars.title)}</b><span>${escapeHtml(bars.valueLabel)}${bars.noteLabel ? ` · ${escapeHtml(bars.noteLabel)}` : ""}</span></figcaption>`,
      '<div class="result-headline-bar-list" role="list">',
      ...bars.items.map((item) => [
        `<div class="result-headline-bar" role="listitem" aria-label="${escapeHtml(`${item.label}：${item.text}${item.note ? `，${bars.noteLabel} ${item.note}` : ""}`)}">`,
        `<span class="result-headline-bar-label" title="${escapeHtml(item.alias ? `${item.label} · ${item.alias}` : item.label)}">${escapeHtml(item.label)}${item.alias ? `<small class="field-alias">${escapeHtml(item.alias)}</small>` : ""}</span>`,
        `<i aria-hidden="true" style="--w:${(item.ratio * 100).toFixed(1)}%"></i>`,
        `<b class="num">${escapeHtml(item.text)}</b>`,
        item.note ? `<small class="num">${escapeHtml(item.note)}</small>` : "",
        "</div>",
      ].join("")),
      "</div>",
      "</figure>",
    ].join("")
    : "";
  return `<div class="result-headline">${tilesHtml}${barsHtml}</div>`;
}
