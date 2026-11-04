// Result cards (2026-10-14 redesign, batch 6a, D12): a step's evidence is a card
// in the conversation instead of a separate evidence view. The card only
// rearranges the platform's structured payload (the same metric table sections
// the evidence view rendered); it never computes a metric or invents a
// pass/fail line. Status words come from thresholds carried in the payload
// (column_specs), so KS and AUC, which carry none, get no verdict.
// Pure module: no DOM access at import time.
import { escapeHtml } from "./ui-utils.js";
import { parseNumeric, psiTier } from "./render-metrics.js";

const SPLIT_LABELS = { train: "训练", test: "测试", oot: "OOT" };
const SPLIT_ORDER = ["train", "test", "oot"];
const PSI_TIER_WORDS = { stable: "稳定", warn: "可接受", critical: "漂移显著" };
const PSI_TIER_TONES = { stable: "good", warn: "warn", critical: "bad" };
const DELTA_TIER_WORDS = { stable: "KS 变化小", warn: "KS 变化中", critical: "KS 变化大" };
// The platform's own result enums (marvis/validation/results.py): PMML scoring
// is pass | failed; Notebook-vs-PMML consistency is pass | review | fail.
const PMML_STATUS = { pass: { word: "通过", tone: "good" }, failed: { word: "未通过", tone: "bad" } };
const CONSISTENCY_STATUS = {
  pass: { word: "通过", tone: "good" },
  review: { word: "需复核", tone: "warn" },
  fail: { word: "未通过", tone: "bad" },
};

function tablesOf(section) {
  return Array.isArray(section?.tables) ? section.tables : [];
}

function headerIndex(table, test) {
  const headers = Array.isArray(table?.headers) ? table.headers : [];
  return headers.findIndex((header) => test(String(header || "")));
}

function cell(row, index) {
  return index >= 0 && Array.isArray(row) ? row[index] : undefined;
}

function specKind(table, index) {
  const spec = Array.isArray(table?.column_specs) ? table.column_specs[index] : null;
  return String(spec?.kind || "");
}

// PSI columns are the ones the payload types as "psi" (a header such as
// "PSI参考月" is a month label, not a PSI).
function isPsiColumn(table, index) {
  const kind = specKind(table, index);
  return kind ? kind === "psi" : /^PSI/i.test(String(table?.headers?.[index] || ""));
}

function specThresholds(table, index) {
  const spec = Array.isArray(table?.column_specs) ? table.column_specs[index] : null;
  return Array.isArray(spec?.thresholds) && spec.thresholds.length === 2 ? spec.thresholds : null;
}

function splitKey(value) {
  return String(value || "").trim().toLowerCase();
}

function findTable(sections, predicate) {
  for (const section of Array.isArray(sections) ? sections : []) {
    const table = tablesOf(section).find((item) => predicate(item, section));
    if (table) return { section, table };
  }
  return null;
}

function kpiTiles(kpi) {
  if (!kpi) return [];
  const rows = Array.isArray(kpi.rows) ? kpi.rows : [];
  const bySplit = new Map(rows.map((row) => [splitKey(row?.[0]), row]));
  const focusKey = bySplit.has("oot") ? "oot" : splitKey(rows[rows.length - 1]?.[0]);
  const focus = bySplit.get(focusKey);
  if (!focus) return [];
  const focusLabel = SPLIT_LABELS[focusKey] || String(focus[0] || "");
  const tile = (test) => {
    const index = headerIndex(kpi, test);
    if (index < 0) return null;
    const value = cell(focus, index);
    if (value === undefined || value === null || value === "") return null;
    const others = SPLIT_ORDER
      .filter((key) => key !== focusKey && bySplit.has(key))
      .map((key) => `${SPLIT_LABELS[key]} ${cell(bySplit.get(key), index)}`);
    return { label: `${kpi.headers[index]} · ${focusLabel}`, value: String(value), sub: others.join(" · "), index };
  };
  const tiles = [];
  const ks = tile((header) => /^KS/i.test(header));
  if (ks) tiles.push({ key: "ks", ...ks });
  const auc = tile((header) => /^AUC/i.test(header));
  if (auc) tiles.push({ key: "auc", ...auc });
  const psiIndex = (kpi.headers || []).findIndex((_, index) => isPsiColumn(kpi, index));
  const psi = psiIndex >= 0 ? tile((header) => header === String(kpi.headers[psiIndex])) : null;
  if (psi) {
    const thresholds = specThresholds(kpi, psi.index);
    const numeric = parseNumeric(psi.value);
    const tier = thresholds && numeric !== null ? psiTier(numeric, thresholds) : "";
    tiles.push({
      key: "psi",
      ...psi,
      sub: "",
      tone: PSI_TIER_TONES[tier] || "",
      status: PSI_TIER_WORDS[tier] || "",
    });
  }
  return tiles;
}

function monthlyModel(trend) {
  if (!trend) return null;
  const rows = Array.isArray(trend.rows) ? trend.rows : [];
  const ksIndex = headerIndex(trend, (header) => /^KS/i.test(header));
  const psiColumns = (trend.headers || [])
    .map((header, index) => ({ header: String(header || ""), index }))
    .filter(({ index }) => isPsiColumn(trend, index))
    .map(({ header, index }) => ({ header, index, thresholds: specThresholds(trend, index) }));
  if (!rows.length || !psiColumns.length) return null;
  const items = rows.map((row) => {
    const psi = psiColumns.map((column) => {
      const value = cell(row, column.index);
      const numeric = parseNumeric(value);
      return {
        header: column.header,
        value: String(value ?? ""),
        tier: numeric === null || !column.thresholds ? "" : psiTier(numeric, column.thresholds),
        numeric,
      };
    });
    return { month: String(row?.[0] ?? ""), ks: ksIndex >= 0 ? String(cell(row, ksIndex) ?? "") : "", psi };
  });
  const flagged = (tier) => items.filter((item) => item.psi.some((entry) => entry.tier === tier)).length;
  const first = psiColumns[0];
  // A verdict needs at least one computed PSI judged by a payload threshold.
  const judged = items.some((item) => item.psi.some((entry) => entry.tier));
  let peak = null;
  for (const item of items) {
    const entry = item.psi[0];
    if (entry?.numeric === null || entry?.numeric === undefined) continue;
    if (!peak || Math.abs(entry.numeric) > Math.abs(peak.numeric)) peak = { ...entry, month: item.month };
  }
  return {
    ksHeader: ksIndex >= 0 ? String(trend.headers[ksIndex]) : "",
    psiHeaders: psiColumns.map((column) => column.header),
    criticalAt: first.thresholds ? first.thresholds[1] : null,
    warnAt: first.thresholds ? first.thresholds[0] : null,
    judged,
    items,
    critical: flagged("critical"),
    warn: flagged("warn"),
    peak,
  };
}

function stressModel(stress) {
  if (!stress) return null;
  const rows = Array.isArray(stress.rows) ? stress.rows : [];
  const baseIndex = headerIndex(stress, (header) => /baseline/i.test(header) && /^KS/i.test(header));
  const afterIndex = headerIndex(stress, (header) => /after/i.test(header));
  const deltaIndex = headerIndex(stress, (header) => /delta/i.test(header));
  const psiIndex = (stress.headers || []).findIndex((header, index) => /^PSI/i.test(String(header || "")) && isPsiColumn(stress, index));
  if (!rows.length || afterIndex < 0 || deltaIndex < 0) return null;
  const deltaThresholds = specThresholds(stress, deltaIndex);
  const psiThresholds = psiIndex >= 0 ? specThresholds(stress, psiIndex) : null;
  const items = rows.map((row) => {
    const delta = parseNumeric(cell(row, deltaIndex));
    const psi = psiIndex >= 0 ? parseNumeric(cell(row, psiIndex)) : null;
    const deltaTier = deltaThresholds && delta !== null ? psiTier(delta, deltaThresholds) : "";
    return {
      name: String(row?.[0] ?? ""),
      base: String(cell(row, baseIndex) ?? ""),
      after: String(cell(row, afterIndex) ?? ""),
      delta: String(cell(row, deltaIndex) ?? ""),
      deltaNumeric: delta,
      deltaTier,
      psi: psiIndex >= 0 ? String(cell(row, psiIndex) ?? "") : "",
      psiTier: psiThresholds && psi !== null ? psiTier(psi, psiThresholds) : "",
    };
  });
  const drops = items.filter((item) => item.deltaNumeric !== null && item.deltaNumeric < 0);
  const largestDrop = drops.sort((left, right) => left.deltaNumeric - right.deltaNumeric)[0] || null;
  return { items, largestDrop, psiHeader: psiIndex >= 0 ? String(stress.headers[psiIndex]) : "" };
}

// 分箱排序性 (train bins) and 独立10等分分箱 (each split's own deciles) are
// separate sections; each table is one split under one mode, titled like
// "OOT(按照train分箱)". The line is the bin's bad rate (逾期率, not 累计), and
// every split keeps its own interval labels.
function binningModel(sections) {
  const modes = new Map();
  for (const table of sections.flatMap(tablesOf)) {
    const title = String(table?.title || "");
    const match = /^(.*?)[(（](.*)[)）]\s*$/.exec(title);
    const name = (match ? match[1] : title).trim();
    const mode = (match ? match[2] : "").trim() || "分箱";
    const rateIndex = headerIndex(table, (header) => header.trim() === "逾期率");
    const rows = Array.isArray(table?.rows) ? table.rows : [];
    if (rateIndex < 0 || !rows.length) continue;
    if (!modes.has(mode)) modes.set(mode, []);
    modes.get(mode).push({
      name,
      labels: rows.map((row) => String(row?.[0] ?? "")),
      raw: rows.map((row) => String(row?.[rateIndex] ?? "")),
      values: rows.map((row) => parseNumeric(row?.[rateIndex])),
    });
  }
  return [...modes.entries()].map(([label, series]) => ({ label, series }));
}

export function validationMetricCardModel(sections = []) {
  const kpi = findTable(sections, (table) => table?.layout === "kpi_cards")?.table || null;
  const trend = findTable(sections, (table) => table?.layout === "trend_table")?.table || null;
  const stress = findTable(sections, (table, section) => String(section?.title || "").includes("压力测试")
    && headerIndex(table, (header) => /delta/i.test(header)) >= 0)?.table || null;
  const binningSections = (Array.isArray(sections) ? sections : []).filter((section) => {
    const title = String(section?.title || "");
    return title.includes("分箱排序") || title.includes("独立10等分");
  });
  const tiles = kpiTiles(kpi);
  const monthly = monthlyModel(trend);
  const psiTile = tiles.find((tile) => tile.key === "psi");
  if (psiTile && monthly?.peak) {
    psiTile.sub = `逐月最高 ${monthly.peak.value} · ${monthly.peak.month}`;
    psiTile.title = `逐月最高 ${monthly.peak.value}（${monthly.peak.month}，${monthly.peak.header}）`;
  }
  return {
    tiles,
    monthly,
    stress: stressModel(stress),
    binning: binningModel(binningSections),
  };
}

// The fourth tile: Notebook-vs-PMML consistency when that check ran, else the
// PMML scoring run. Status words are the platform's own pass/fail.
export function scoreEvidenceTile(evidence = null) {
  if (!evidence?.data || !Object.keys(evidence.data).length) return null;
  const data = evidence.data;
  if (evidence.kind === "pmml") {
    const count = (value) => (value === null || value === undefined || value === "" ? null : Number(value));
    const total = count(data.input_row_count);
    const ok = count(data.success_count);
    if (!Number.isFinite(total) || !Number.isFinite(ok)) return null;
    const verdict = PMML_STATUS[String(data.status || "")] || null;
    return {
      key: "score",
      label: "PMML 打分，成功行数",
      value: ok.toLocaleString("en-US"),
      sub: `共 ${total.toLocaleString("en-US")} 行 · 失败 ${Number(data.failure_count || 0)} · 空值 ${Number(data.null_count || 0)}`,
      tone: verdict?.tone || "",
      status: verdict?.word || "",
    };
  }
  const summary = data.summary || {};
  const verdict = CONSISTENCY_STATUS[String(summary.status || "")] || null;
  if (!verdict) return null;
  const mismatches = summary.mismatch_count;
  return {
    key: "score",
    label: "分数一致性",
    value: mismatches === null || mismatches === undefined ? verdict.word : `${Number(mismatches)} 条不一致`,
    sub: data.sample_size ? `抽样 ${data.sample_size} 行` : "",
    tone: verdict.tone,
    status: verdict.word,
  };
}

function statusHtml(tone, text) {
  if (!text) return "";
  return `<em class="result-status" data-tone="${escapeHtml(tone || "neutral")}">${escapeHtml(text)}</em>`;
}

function tileHtml(tile) {
  return [
    `<div class="result-tile" data-tile="${escapeHtml(tile.key)}">`,
    `<span class="result-tile-label">${escapeHtml(tile.label)}</span>`,
    `<b class="result-tile-value">${escapeHtml(tile.value)}</b>`,
    tile.sub ? `<small class="result-tile-sub" title="${escapeHtml(tile.title || tile.sub)}">${escapeHtml(tile.sub)}</small>` : "",
    statusHtml(tile.tone, tile.status),
    "</div>",
  ].join("");
}

const CHART = { width: 640, height: 200, left: 46, right: 12, top: 14, bottom: 30 };

function niceMax(value) {
  if (!(value > 0)) return 1;
  const exponent = 10 ** Math.floor(Math.log10(value));
  const step = [1, 2, 2.5, 5, 10].find((candidate) => candidate * exponent >= value / 4) * exponent;
  return Math.ceil((value * 1.08) / step) * step;
}

function percentText(value, digits = 1) {
  return value === null || value === undefined ? "-" : `${(value * 100).toFixed(digits)}%`;
}

export function binChartSvg(mode) {
  const series = mode?.series || [];
  const count = Math.max(0, ...series.map((item) => item.values.length));
  if (!count) return "";
  const values = series.flatMap((item) => item.values).filter((value) => value !== null);
  const yMax = niceMax(Math.max(0, ...values));
  const plotWidth = CHART.width - CHART.left - CHART.right;
  const plotHeight = CHART.height - CHART.top - CHART.bottom;
  const x = (index) => CHART.left + (count === 1 ? plotWidth / 2 : (plotWidth * index) / (count - 1));
  const y = (value) => CHART.top + plotHeight - (plotHeight * value) / yMax;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => fraction * yMax);
  const grid = ticks.map((tick) => [
    `<line class="chart-grid" x1="${CHART.left}" x2="${CHART.width - CHART.right}" y1="${y(tick).toFixed(1)}" y2="${y(tick).toFixed(1)}" />`,
    `<text class="chart-axis" x="${CHART.left - 6}" y="${(y(tick) + 4).toFixed(1)}" text-anchor="end">${escapeHtml(percentText(tick, yMax < 0.1 ? 1 : 0))}</text>`,
  ].join("")).join("");
  const xLabels = Array.from({ length: count }, (_, index) => (
    `<text class="chart-axis" x="${x(index).toFixed(1)}" y="${CHART.height - 10}" text-anchor="middle">${index + 1}</text>`
  )).join("");
  const lines = series.map((item, seriesIndex) => {
    const points = item.values
      .map((value, index) => (value === null ? null : `${x(index).toFixed(1)},${y(value).toFixed(1)}`))
      .filter(Boolean);
    const dots = item.values.map((value, index) => (value === null ? "" : (
      `<circle class="chart-dot" cx="${x(index).toFixed(1)}" cy="${y(value).toFixed(1)}" r="3" />`
    ))).join("");
    return `<g class="chart-series" data-series="${seriesIndex}"><polyline points="${points.join(" ")}" />${dots}</g>`;
  }).join("");
  const columns = Array.from({ length: count }, (_, index) => {
    const start = index === 0 ? CHART.left : (x(index - 1) + x(index)) / 2;
    const end = index === count - 1 ? CHART.width - CHART.right : (x(index) + x(index + 1)) / 2;
    return `<rect class="chart-hit" data-bin="${index}" x="${start.toFixed(1)}" y="${CHART.top}" width="${(end - start).toFixed(1)}" height="${plotHeight}" />`;
  }).join("");
  return [
    `<svg class="result-chart-svg" viewBox="0 0 ${CHART.width} ${CHART.height}" role="presentation" focusable="false">`,
    grid,
    xLabels,
    `<line class="chart-guide" x1="0" x2="0" y1="${CHART.top}" y2="${CHART.top + plotHeight}" />`,
    lines,
    columns,
    "</svg>",
  ].join("");
}

function binningVizHtml(binning) {
  if (!binning?.length) return "";
  const modeButtons = binning.length > 1
    ? [
      '<div class="result-seg" role="group" aria-label="分箱方式">',
      binning.map((mode, index) => (
        `<button type="button" data-result-bin-mode="${index}" aria-pressed="${index === 0}">${escapeHtml(mode.label)}</button>`
      )).join(""),
      "</div>",
    ].join("")
    : `<span class="result-chart-mode">${escapeHtml(binning[0].label)}</span>`;
  const panels = binning.map((mode, index) => [
    `<div class="result-chart-mode-panel" data-result-bin-panel="${index}"${index === 0 ? "" : " hidden"}>`,
    `<div class="result-chart-legend">${mode.series.map((item, seriesIndex) => (
      `<span data-series="${seriesIndex}"><i aria-hidden="true"></i>${escapeHtml(item.name)}</span>`
    )).join("")}</div>`,
    `<div class="result-chart-plot" tabindex="0" role="group" aria-roledescription="图表" aria-label="${escapeHtml(`各分箱逾期率，${mode.series.map((item) => item.name).join("、")}，左右方向键查看各分箱`)}" data-result-chart='${escapeHtml(JSON.stringify(mode.series.map((item) => ({ name: item.name, labels: item.labels, raw: item.raw }))))}'>`,
    binChartSvg(mode),
    '<div class="result-chart-readout" aria-live="polite" hidden></div>',
    "</div>",
    "</div>",
  ].join("")).join("");
  return [
    '<div class="result-viz" data-result-viz>',
    '<div class="result-viz-head">',
    "<div><h4>分箱排序性</h4><p>各分箱的逾期率，分数从低到高</p></div>",
    '<span class="result-grow"></span>',
    modeButtons,
    "</div>",
    panels,
    "</div>",
  ].join("");
}

function monthlyDetailsHtml(monthly) {
  if (!monthly) return "";
  const summary = !monthly.judged
    ? ""
    : monthly.critical
      ? statusHtml("bad", `${monthly.critical} 个月漂移显著（PSI ≥ ${monthly.criticalAt}）`)
      : monthly.warn
        ? statusHtml("warn", `${monthly.warn} 个月 PSI 在可接受区间`)
        : statusHtml("good", "各月稳定");
  const head = [
    '<tr><th scope="col">月份</th>',
    monthly.ksHeader ? `<th scope="col">${escapeHtml(monthly.ksHeader)}</th>` : "",
    monthly.psiHeaders.map((header) => `<th scope="col">${escapeHtml(header)}</th>`).join(""),
    "</tr>",
  ].join("");
  const rows = monthly.items.map((item) => [
    `<tr><th scope="row">${escapeHtml(item.month)}</th>`,
    monthly.ksHeader ? `<td>${escapeHtml(item.ks)}</td>` : "",
    item.psi.map((entry) => (
      `<td data-tone="${escapeHtml(PSI_TIER_TONES[entry.tier] || "")}">${escapeHtml(entry.value)}</td>`
    )).join(""),
    "</tr>",
  ].join("")).join("");
  return [
    `<section class="result-section"><div class="result-section-head"><h4 class="result-grow">逐月稳定性</h4>${summary}</div>`,
    `<div class="result-table-wrap"><table class="result-mini-table"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`,
    monthly.judged
      ? `<p class="result-note">PSI 阈值来自平台：低于 ${escapeHtml(String(monthly.warnAt))} 稳定，${escapeHtml(String(monthly.criticalAt))} 及以上漂移显著。</p>`
      : "",
    "</section>",
  ].join("");
}

function stressDetailsHtml(stress) {
  if (!stress) return "";
  const drop = stress.largestDrop;
  const summary = drop
    ? statusHtml(PSI_TIER_TONES[drop.deltaTier] || "neutral", `剔除「${drop.name}」KS 降幅最大：${drop.base} → ${drop.after}`)
    : "";
  const rows = stress.items.map((item) => [
    '<div class="result-row">',
    `<b class="result-grow">${escapeHtml(item.name)}</b>`,
    `<span class="result-row-value">KS ${escapeHtml(item.base)} → ${escapeHtml(item.after)}（${escapeHtml(item.delta)}）</span>`,
    item.deltaTier ? statusHtml(PSI_TIER_TONES[item.deltaTier], DELTA_TIER_WORDS[item.deltaTier]) : "",
    item.psi ? `<span class="result-row-value" data-tone="${escapeHtml(PSI_TIER_TONES[item.psiTier] || "")}">${escapeHtml(stress.psiHeader)} ${escapeHtml(item.psi)}</span>` : "",
    "</div>",
  ].join("")).join("");
  return `<section class="result-section"><div class="result-section-head"><h4 class="result-grow">压力测试，逐个剔除数据源</h4>${summary}</div><div class="result-rows">${rows}</div></section>`;
}

// detailsHtml: the original metric sections the summary above does not
// already show (sample, full bin tables, feature importance, stress coverage,
// ROC/KS), laid out directly under it — nothing folded, nothing twice
// (2026-10-19 user feedback).
export function validationMetricCardHtml(model, { title = "", scoreTile = null, detailsHtml = "" } = {}) {
  const tiles = [...(model?.tiles || []), ...(scoreTile ? [scoreTile] : [])];
  return [
    '<article class="result-card" data-result-card="validation-metrics" data-agent-result="true">',
    '<header class="result-card-head">',
    `<b title="${escapeHtml(title ? `${title} 的效果与稳定性` : "效果与稳定性")}">${escapeHtml(title ? `${title} 的效果与稳定性` : "效果与稳定性")}</b>`,
    '<span class="result-card-source">平台计算</span>',
    "</header>",
    tiles.length ? `<div class="result-tiles">${tiles.map(tileHtml).join("")}</div>` : "",
    binningVizHtml(model?.binning),
    monthlyDetailsHtml(model?.monthly),
    stressDetailsHtml(model?.stress),
    detailsHtml ? `<div class="result-sections">${detailsHtml}</div>` : "",
    "</article>",
  ].join("");
}

// ---------- interactions (hover / keyboard readout, chart-table switch) ----------

function readoutHtml(series, index) {
  const shared = sharedLabels(series);
  const label = shared ? series[0]?.labels?.[index] || "" : "";
  return [
    `<b>第 ${index + 1} 箱</b>`,
    label ? `<small>${escapeHtml(label)}</small>` : "",
    series.map((item, seriesIndex) => (
      `<span data-series="${seriesIndex}"><i aria-hidden="true"></i>${escapeHtml(item.name)} ${escapeHtml(item.raw[index] || "-")}${shared ? "" : ` <small>${escapeHtml(item.labels[index] || "")}</small>`}</span>`
    )).join(""),
  ].join("");
}

function showBin(plot, index) {
  let series = [];
  try {
    series = JSON.parse(plot.dataset.resultChart || "[]");
  } catch (_) {
    return;
  }
  const count = Math.max(0, ...series.map((item) => item.raw.length));
  if (!count) return;
  const bin = Math.max(0, Math.min(count - 1, index));
  plot.dataset.activeBin = String(bin);
  const hit = plot.querySelector(`.chart-hit[data-bin="${bin}"]`);
  const guide = plot.querySelector(".chart-guide");
  const readout = plot.querySelector(".result-chart-readout");
  if (hit && guide) {
    const center = Number(hit.getAttribute("x")) + Number(hit.getAttribute("width")) / 2;
    guide.setAttribute("x1", String(center));
    guide.setAttribute("x2", String(center));
    guide.classList.add("is-on");
    if (readout) {
      readout.innerHTML = readoutHtml(series, bin);
      readout.hidden = false;
      readout.style.left = `${Math.min(70, Math.max(0, (center / CHART.width) * 100 - 8))}%`;
    }
  }
}

function hideBin(plot) {
  delete plot.dataset.activeBin;
  plot.querySelector(".chart-guide")?.classList.remove("is-on");
  const readout = plot.querySelector(".result-chart-readout");
  if (readout) readout.hidden = true;
}

export function attachResultCardInteractions(root) {
  if (!root || root.dataset?.resultCardsBound === "true") return;
  if (root.dataset) root.dataset.resultCardsBound = "true";
  root.addEventListener("pointermove", (event) => {
    const hit = event.target?.closest?.(".chart-hit");
    if (hit) showBin(hit.closest(".result-chart-plot"), Number(hit.dataset.bin));
  });
  root.addEventListener("pointerleave", (event) => {
    if (event.target?.matches?.(".result-chart-plot")) hideBin(event.target);
  }, true);
  root.addEventListener("keydown", (event) => {
    const plot = event.target?.closest?.(".result-chart-plot");
    if (!plot || !["ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Escape") {
      hideBin(plot);
      return;
    }
    const current = plot.dataset.activeBin === undefined ? -1 : Number(plot.dataset.activeBin);
    const next = event.key === "Home" ? 0
      : event.key === "End" ? Number.MAX_SAFE_INTEGER
        : event.key === "ArrowLeft" ? Math.max(0, current - 1) : current + 1;
    showBin(plot, next);
  });
  root.addEventListener("focusout", (event) => {
    if (event.target?.matches?.(".result-chart-plot")) hideBin(event.target);
  });
  root.addEventListener("click", (event) => {
    const viewButton = event.target?.closest?.("[data-result-view]");
    const modeButton = event.target?.closest?.("[data-result-bin-mode]");
    const viz = (viewButton || modeButton)?.closest?.("[data-result-viz]");
    if (!viz) return;
    if (modeButton) {
      viz.querySelectorAll("[data-result-bin-mode]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button === modeButton));
      });
      viz.querySelectorAll("[data-result-bin-panel]").forEach((panel) => {
        panel.hidden = panel.dataset.resultBinPanel !== modeButton.dataset.resultBinMode;
      });
    }
    if (viewButton) {
      viz.querySelectorAll("[data-result-view]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button === viewButton));
      });
    }
    const table = viz.querySelector('[data-result-view="table"]')?.getAttribute("aria-pressed") === "true";
    viz.querySelectorAll(".result-chart-plot").forEach((plot) => { plot.hidden = table; });
    viz.querySelectorAll(".result-chart-legend").forEach((legend) => { legend.hidden = table; });
    viz.querySelectorAll(".result-chart-table-wrap").forEach((wrap) => { wrap.hidden = !table; });
  });
}
