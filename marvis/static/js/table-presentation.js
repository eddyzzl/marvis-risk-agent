// Readable platform tables in the conversation (2026-10-19 user feedback):
// enum codes read as Chinese tags, "iv=0.90438；ks=0.4009…" evidence strings
// as compact metric chips, over-long decimals are trimmed for display, and a
// two-column field/value table reads as a field list. Display only: values
// are never recomputed or judged here; the platform's text stays in the
// tooltip so nothing is lost.
// Pure module: no DOM access, so node test harnesses load it.
import { escapeHtml } from "./ui-utils.js";
import { localizeText } from "./i18n.js";

// value -> [label, tone]; tones: good | warn | bad | info | muted.
const ENUM_TAGS = Object.freeze({
  candidate: ["候选", "info"],
  recommended: ["推荐", "good"],
  recommend: ["推荐", "good"],
  not_recommended: ["不推荐", "muted"],
  caution: ["慎用", "warn"],
  watch: ["关注", "warn"],
  not_evaluated: ["未评估", "muted"],
  leakage: ["泄漏", "bad"],
  suspected: ["疑似", "warn"],
  pass: ["通过", "good"],
  passed: ["通过", "good"],
  fail: ["未通过", "bad"],
  failed: ["失败", "bad"],
  warning: ["警告", "warn"],
  blocked: ["阻断", "bad"],
  completed: ["完成", "good"],
  success: ["成功", "good"],
  deteriorating: ["恶化", "bad"],
  improving: ["改善", "good"],
  stable: ["稳定", "good"],
  monotonic: ["单调", "good"],
  high: ["高", "muted"],
  medium: ["中", "muted"],
  low: ["低", "muted"],
});

const METRIC_LABELS = Object.freeze({
  iv: "IV",
  ks: "KS",
  auc: "AUC",
  psi: "PSI",
  vif: "VIF",
  missing_rate: "缺失率",
  mode_rate: "最常见值占比",
  zero_rate: "零值率",
  unique_count: "唯一值数",
  coverage: "覆盖率",
  hit_rate: "命中率",
});
const RATE_METRICS = new Set(["missing_rate", "mode_rate", "zero_rate", "coverage", "hit_rate"]);

// Field/value tables (验证字段候选, 最优 trial 指标 …): machine keys read as
// their business names.
const FIELD_LABELS = Object.freeze({
  algorithm: "算法",
  target_col: "目标字段",
  split_col: "样本划分字段",
  time_col: "时间字段",
  time_granularity: "时间粒度",
  pmml_output_field: "PMML 分数字段",
  model_params: "模型参数",
  feature_metadata_selection: "特征元数据列",
  positive_label: "坏样本值",
  negative_label: "好样本值",
  feature_col: "特征列",
  category_col: "类别列",
  importance_col: "重要性列",
  metadata_sheet: "元数据工作表",
  intercept: "截距",
  selection_metric: "选优指标",
  search_stage: "搜索阶段",
  score: "得分",
  train_ks: "训练集 KS",
  test_ks: "测试集 KS",
  oot_ks: "OOT KS",
  train_auc: "训练集 AUC",
  test_auc: "测试集 AUC",
  oot_auc: "OOT AUC",
});

const KV_FIRST_HEADER = /^(验证字段|字段|项目|参数|指标|名称|配置项|Validation fields?|Fields?|Items?|Parameters?|Metrics?|Name|Settings?)$/i;
const PLATFORM_KEY_HEADER = /^(项目|参数|指标|配置项|Items?|Parameters?|Metrics?|Settings?)$/i;
const LONG_DECIMAL = /^-?\d+\.\d{5,}$/;
const METRIC_PAIR = /([A-Za-z_]+)\s*=\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)/g;

function text(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return String(value.text ?? value.value ?? JSON.stringify(value));
  return String(value);
}

// 0.40096692502494047 -> 0.4010; integers and short decimals stay as written.
export function displayNumberText(value) {
  const raw = text(value).trim();
  if (!LONG_DECIMAL.test(raw)) return raw;
  const number = Number(raw);
  return Number.isFinite(number) ? number.toFixed(4) : raw;
}

export function enumTag(value) {
  const key = text(value).trim().toLowerCase();
  return Object.prototype.hasOwnProperty.call(ENUM_TAGS, key) ? ENUM_TAGS[key] : null;
}

// "iv=0.90438；ks=0.4009…；missing_rate=0.0" -> [{label, value}] when the whole
// cell is such pairs; otherwise null.
export function metricPairs(value) {
  const raw = text(value).trim();
  if (!raw || !raw.includes("=")) return null;
  const pairs = [];
  for (const match of raw.matchAll(METRIC_PAIR)) {
    const key = match[1].toLowerCase();
    const number = Number(match[2]);
    if (!Number.isFinite(number)) return null;
    const shown = RATE_METRICS.has(key)
      ? `${Number((number * 100).toFixed(2))}%`
      : key === "unique_count"
        ? String(Math.trunc(number))
        : Number(number.toFixed(4)).toString();
    pairs.push({ label: METRIC_LABELS[key] || match[1], value: shown });
  }
  // Every non-separator character must belong to a pair.
  const rest = raw.replace(METRIC_PAIR, "").replace(/[；;，,\s|]/g, "");
  return pairs.length >= 2 && !rest ? pairs : null;
}

// Presented cell HTML for enum tags and metric pairs; null keeps the default.
export function presentedCellHtml(value) {
  const tag = enumTag(value);
  if (tag) {
    return {
      cls: "cell-tag",
      html: `<span class="result-tag" data-tone="${tag[1]}" title="${escapeHtml(text(value))}">${escapeHtml(tag[0])}</span>`,
    };
  }
  const pairs = metricPairs(value);
  if (pairs) {
    return {
      cls: "cell-metrics",
      html: `<span class="result-metric-chips" title="${escapeHtml(text(value))}">${pairs.map((pair) => (
        `<span class="result-metric-chip"><i>${escapeHtml(pair.label)}</i>${escapeHtml(pair.value)}</span>`
      )).join("")}</span>`,
    };
  }
  return null;
}

function parsedJson(value) {
  const raw = text(value).trim();
  if (!raw || !/^[[{"]/.test(raw)) return value;
  try {
    return JSON.parse(raw);
  } catch (_) {
    return value;
  }
}

function fieldValueText(value) {
  const parsed = parsedJson(value);
  if (parsed === null || parsed === undefined || parsed === "") return "—";
  if (Array.isArray(parsed)) return parsed.map((item) => fieldValueText(item)).join("、");
  if (typeof parsed === "object") {
    return Object.entries(parsed)
      .filter(([, item]) => item !== null && item !== undefined && item !== "")
      .map(([key, item]) => `${FIELD_LABELS[key] ? localizeText(FIELD_LABELS[key]) : key} ${fieldValueText(item)}`)
      .join(" · ") || "—";
  }
  return displayNumberText(parsed);
}

// A structured candidate (模型参数 / 特征元数据列) read as business words for
// a select option; an empty object means nothing extra was declared.
export function readableValueText(value) {
  if (value && typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length) {
    return localizeText("无额外参数");
  }
  return fieldValueText(value);
}

// A two-column name/value table reads as a field list: business names,
// readable values, identical rows once.
export function fieldListModel(table) {
  const columns = Array.isArray(table?.columns) ? table.columns.map(text) : [];
  const rows = Array.isArray(table?.rows) ? table.rows : [];
  if (columns.length !== 2 || rows.length < 2 || !KV_FIRST_HEADER.test(columns[0].trim())) return null;
  const seen = new Set();
  const items = [];
  for (const row of rows) {
    const cells = Array.isArray(row) ? row : [row];
    const key = text(cells[0]).trim();
    const value = fieldValueText(cells[1]);
    const identity = `${key}\u0000${value}`;
    if (!key || seen.has(identity)) continue;
    seen.add(identity);
    const sourceLabel = FIELD_LABELS[key] || key;
    const platformLabel = Boolean(FIELD_LABELS[key]) || PLATFORM_KEY_HEADER.test(columns[0].trim());
    items.push({ key, label: platformLabel ? localizeText(sourceLabel) : sourceLabel, value });
  }
  return items.length ? { items } : null;
}

export function fieldListHtml(model) {
  return [
    '<dl class="result-field-list" translate="no">',
    ...model.items.map((item) => [
      `<div class="result-field" title="${escapeHtml(item.key)}">`,
      `<dt>${escapeHtml(item.label)}</dt>`,
      `<dd>${escapeHtml(item.value)}</dd>`,
      "</div>",
    ].join("")),
    "</dl>",
  ].join("");
}
