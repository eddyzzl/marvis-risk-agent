import { escapeHtml } from "./ui-utils.js";
import {
  columnFractions,
  columnHeatColors,
  columnRanks,
  parseNumeric,
  psiTier,
  psiTooltipText,
} from "./render-metrics.js";

export function renderMetricTableSection(section = {}, index = 0, options = {}) {
  const tables = Array.isArray(section.tables) ? section.tables : [];
  const theme = section.section_theme || "cool-blue";
  const isOverallEffect =
    (tables[0] && tables[0].layout === "kpi_cards")
    || section.title === "整体效果&稳定性";
  const sectionIndex = String(index + 1).padStart(2, "0");
  const title = section.title || "";
  return [
    `<section class="metric-table-section" data-theme="${escapeHtml(theme)}" data-section-index="${escapeHtml(sectionIndex)}">`,
    `<h4 class="metric-section-title">${escapeHtml(title)}</h4>`,
    '<div class="metric-table-stack">',
    ...tables.map((table) => {
      if (isOverallEffect && table.layout === "kpi_cards") {
        return renderKpiCards(table, { curves: options.rocCurves || null, animate: options.animate });
      }
      return renderMetricTable(table, { animate: options.animate });
    }),
    "</div>",
    "</section>",
  ].join("");
}

// ====== Metric overview cell helpers ======

export function renderCellByKind(spec, value, context) {
  const kind = (spec && spec.kind) || "text";
  if (kind === "trend-spark" && spec && spec.__localHtml === true) {
    return { cls: "cell-sparkline", html: String(value ?? "") };  // value is raw <svg>
  }
  const headerLabel = context.headerLabel ?? "";
  switch (kind) {
    case "split-badge":
      return {
        cls: "cell-split",
        html: `<span class="split-badge">${escapeHtml(String(value ?? "").toUpperCase())}</span>`,
      };
    case "period":
      return {
        cls: "cell-period",
        html: `<span class="period-text">${escapeHtml(String(value ?? ""))}</span>`,
      };
    case "databar":
    case "databar-primary": {
      const fraction = context.fractions.get(context.rowIndex);
      if (fraction === undefined) {
        return { cls: "cell-text", html: escapeHtml(String(value ?? "")) };
      }
      const color = (spec && spec.color) || "primary";
      const emphasize = kind === "databar-primary" ? "primary" : "normal";
      const rank = context.ranks.get(context.rowIndex) ?? "";
      const tip = `${headerLabel} ${value} · ${rank}`;
      return {
        cls: "cell-databar",
        html: `<span class="databar" data-color="${color}" data-emphasize="${emphasize}" data-tip="${escapeHtml(tip)}" style="--fraction:${fraction.toFixed(4)};--bar-index:${context.rowIndex ?? 0}">`
          + `<span class="databar-fill"></span>`
          + `<span class="databar-label">${escapeHtml(String(value ?? ""))}</span>`
          + `</span>`,
      };
    }
    case "percent-heat": {
      const heat = context.heatColors.get(context.rowIndex);
      if (heat === undefined) {
        return { cls: "cell-text", html: escapeHtml(String(value ?? "")) };
      }
      const tip = `${headerLabel} ${value}`;
      return {
        cls: "cell-heat",
        html: `<span class="heat-chip" data-tip="${escapeHtml(tip)}" style="--heat:${heat}">${escapeHtml(String(value ?? ""))}</span>`,
      };
    }
    case "matrix-heat": {
      // S3: NxN migration/flow matrix cell. Colors from the cell's own 0..1
      // value (a transition/migration rate), reusing the percent-heat chip skin
      // and color scale. Self-contained -- no precomputed per-row heatColors,
      // since a matrix column's heat is the cell value itself.
      const numeric = parseNumeric(value);
      if (numeric === null) {
        return { cls: "cell-text", html: escapeHtml(String(value ?? "")) };
      }
      const heat = Math.max(0, Math.min(1, numeric));
      const display = (numeric >= 0 && numeric <= 1) ? `${(numeric * 100).toFixed(1)}%` : String(value);
      const tip = `${headerLabel} ${display}`;
      return {
        cls: "cell-heat",
        html: `<span class="heat-chip" data-tip="${escapeHtml(tip)}" style="--heat:${heat}">${escapeHtml(display)}</span>`,
      };
    }
    case "psi": {
      const numeric = parseNumeric(value);
      const thresholds = (spec && spec.thresholds) || [0.02, 0.10];
      const tier = psiTier(numeric, thresholds);
      const displayText = value === "BASE" || value === "-" || numeric === null
        ? String(value ?? "")
        : String(value);
      const stripMarker = numeric === null
        ? ""
        : `<i class="psi-marker" style="left:${Math.min(Math.abs(numeric) / 0.20, 1) * 100}%"></i>`;
      const tip = psiTooltipText(numeric, thresholds);
      return {
        cls: "cell-psi",
        html: `<span class="psi-cell" data-tip="${escapeHtml(tip)}">`
          + `<span class="psi-value" data-tier="${tier}">${escapeHtml(displayText)}</span>`
          + `<span class="psi-strip"><span></span><span></span><span></span>${stripMarker}</span>`
          + `</span>`,
      };
    }
    case "text":
    default:
      if (metricHeaderShouldRightAlign(headerLabel) && parseNumeric(value) !== null) {
        return { cls: "cell-number", html: escapeHtml(String(value ?? "")) };
      }
      return { cls: "cell-text", html: escapeHtml(String(value ?? "")) };
  }
}

export function metricHeaderShouldRightAlign(headerLabel) {
  const label = String(headerLabel || "").trim();
  if (!label) return false;
  if (/(^id$|id$|编号|月份|日期|时间|参考月|特征|变量|字段|数据集|样本集|分组|类别|等级|区间|范围)/i.test(label)) {
    return false;
  }
  return /^(KS|KS\(%\)|AUC|AUC\(%\)|PSI|IV|样本量|坏样本量|好样本量|逾期率|坏账率|通过率|命中率|缺失率|占比|比例|分数|评分|重要性|Gain|Split|Coverage|Lift|5%头部lift|5%尾部lift)$/i.test(label);
}

export function renderMetricTable(table = {}, options = {}) {
  const layout = table.layout || "table";
  switch (layout) {
    case "kpi_cards":
      return renderKpiCards(table, options);
    case "trend_table":
      return renderTrendTable(table);
    case "roc_ks_curve":
      return renderRocKsCurve(table);
    case "table":
    default:
      return renderEnhancedTable(table, options);
  }
}

export function renderKpiCards(table = {}, options = {}) {
  const headers = Array.isArray(table.headers) ? table.headers : [];
  const rows = Array.isArray(table.rows) ? table.rows : [];
  const specs = Array.isArray(table.column_specs) ? table.column_specs : [];
  const curves = (options && options.curves) || null;
  // VD-9: default to animating (e.g. standalone callers/tests) unless a
  // caller explicitly opts out via options.animate === false.
  const animate = options.animate !== false;
  const animationAttribute = animate ? "" : ' data-animation="none"';

  const idx = (label) => headers.indexOf(label);
  const idxAny = (...labels) => labels.map(idx).find((index) => index >= 0) ?? -1;
  const splitIdx = idx("数据集");
  const periodIdx = idx("时间范围");
  const ksIdx = idxAny("KS(%)", "KS");
  const aucIdx = idxAny("AUC(%)", "AUC");
  const headLiftIdx = idx("5%头部lift");
  const tailLiftIdx = idx("5%尾部lift");
  const psiIdx = idx("PSI");
  const sampleIdx = idx("样本量");
  const badRateIdx = idx("逾期率");
  const badCountIdx = idx("坏样本量");

  const ksFractions = columnFractions(rows, ksIdx);
  const aucFractions = columnFractions(rows, aucIdx);
  const headLiftFractions = columnFractions(rows, headLiftIdx);
  const tailLiftFractions = columnFractions(rows, tailLiftIdx);

  const psiThresholds = (specs[psiIdx] && specs[psiIdx].thresholds) || [0.02, 0.10];

  const cardHtml = rows.map((row, rowIndex) => {
    const cell = (i) => Array.isArray(row) ? row[i] : "";
    const psiNumeric = parseNumeric(cell(psiIdx));
    const psiDisplay = cell(psiIdx);
    const splitName = String(cell(splitIdx) || "").toLowerCase();
    const curveForSplit = curves ? curves[splitName] : null;
    const rocHtml = curveForSplit
      ? renderRocCard(splitName, curveForSplit)
      : "";
    return [
      `<div class="kpi-card-column">`,
      `<article class="kpi-card">`,
      `  <header class="kpi-card-header">`,
      `    <span class="kpi-card-split">${escapeHtml(String(cell(splitIdx) || "").toUpperCase())}</span>`,
      `    <span class="kpi-card-period">${escapeHtml(String(cell(periodIdx) ?? ""))}</span>`,
      `  </header>`,
      `  <div class="kpi-card-primary" data-tip="${escapeHtml(`KS ${cell(ksIdx)} · ${columnRanks(rows, ksIdx).get(rowIndex) || ""}`)}">`,
      `    <span class="kpi-card-primary-label">KS</span>`,
      `    <span class="kpi-card-primary-value">${escapeHtml(String(cell(ksIdx) ?? ""))}</span>`,
      `    <span class="kpi-card-primary-bar" style="--fraction:${(ksFractions.get(rowIndex) ?? 0).toFixed(4)};--bar-index:0"><i></i></span>`,
      `  </div>`,
      `  <div class="kpi-card-rule"></div>`,
      kpiCardRow("AUC", cell(aucIdx), aucFractions.get(rowIndex), rowIndex, aucIdx, rows, "var(--accent)", 1),
      kpiCardRow("5%头部lift", cell(headLiftIdx), headLiftFractions.get(rowIndex), rowIndex, headLiftIdx, rows, "var(--metric-databar-accent)", 2),
      kpiCardRow("5%尾部lift", cell(tailLiftIdx), tailLiftFractions.get(rowIndex), rowIndex, tailLiftIdx, rows, "var(--metric-databar-accent)", 3),
      kpiPsiRow(psiDisplay, psiNumeric, psiThresholds),
      `  <footer class="kpi-card-footer">`,
      `    <span class="kpi-card-footer-cell"><span class="kpi-card-footer-label">样本量</span><span class="kpi-card-footer-value">${escapeHtml(String(cell(sampleIdx) ?? ""))}</span></span>`,
      `    <span class="kpi-card-footer-cell"><span class="kpi-card-footer-label">逾期率</span><span class="kpi-card-footer-value">${escapeHtml(String(cell(badRateIdx) ?? ""))}</span></span>`,
      `    <span class="kpi-card-footer-cell"><span class="kpi-card-footer-label">坏样本</span><span class="kpi-card-footer-value">${escapeHtml(String(cell(badCountIdx) ?? ""))}</span></span>`,
      `  </footer>`,
      `</article>`,
      rocHtml,
      `</div>`,
    ].join("\n");
  }).join("");

  return [
    `<div class="metric-table-wrap"${animationAttribute} data-metric-key="${escapeHtml(table.key || "")}">`,
    `<div class="kpi-cards" style="--kpi-count:${rows.length}">`,
    cardHtml,
    `</div>`,
    `</div>`,
  ].join("");
}

export function kpiCardRow(label, displayValue, fraction, rowIndex, columnIndex, rows, color, barIndex = 0) {
  const tip = `${label} ${displayValue} · ${columnRanks(rows, columnIndex).get(rowIndex) || ""}`;
  return [
    `<div class="kpi-card-row" data-tip="${escapeHtml(tip)}">`,
    `  <span class="kpi-card-row-label">${escapeHtml(label)}</span>`,
    `  <span class="kpi-card-row-bar" style="--fraction:${(fraction ?? 0).toFixed(4)};--bar-color:${color};--bar-index:${barIndex}"><i></i></span>`,
    `  <span class="kpi-card-row-value">${escapeHtml(String(displayValue ?? ""))}</span>`,
    `</div>`,
  ].join("");
}

export function kpiPsiRow(displayValue, numeric, thresholds) {
  const tier = psiTier(numeric, thresholds);
  const tip = psiTooltipText(numeric, thresholds);
  return [
    `<div class="kpi-card-row" data-tip="${escapeHtml(tip)}">`,
    `  <span class="kpi-card-row-label">PSI</span>`,
    `  <span class="psi-strip"><span></span><span></span><span></span>`,
    numeric === null ? "" : `<i class="psi-marker" style="left:${Math.min(Math.abs(numeric) / 0.20, 1) * 100}%"></i>`,
    `  </span>`,
    `  <span class="psi-value kpi-card-row-value" data-tier="${tier}">${escapeHtml(String(displayValue ?? ""))}</span>`,
    `</div>`,
  ].join("");
}

export function renderTrendTable(table = {}) {
  const baseHeaders = Array.isArray(table.headers) ? [...table.headers] : [];
  const baseRows = Array.isArray(table.rows) ? table.rows.map((row) => Array.isArray(row) ? [...row] : []) : [];
  const baseSpecs = Array.isArray(table.column_specs) ? [...table.column_specs] : [];

  const ksIdx = baseHeaders.indexOf("KS(%)") >= 0
    ? baseHeaders.indexOf("KS(%)")
    : baseHeaders.indexOf("KS");
  const ksSeries = ksIdx >= 0
    ? baseRows.map((row) => parseNumeric(row[ksIdx])).filter((v) => v !== null)
    : [];
  const ksAll = ksIdx >= 0
    ? baseRows.map((row) => parseNumeric(row[ksIdx]))
    : [];

  const sampleAtIdx = baseHeaders.indexOf("样本量");
  const insertAt = sampleAtIdx >= 0 ? sampleAtIdx + 1 : baseHeaders.length;
  const trendHeaders = [...baseHeaders];
  const trendSpecs = [...baseSpecs];
  trendHeaders.splice(insertAt, 0, "KS 趋势");
  trendSpecs.splice(insertAt, 0, { kind: "trend-spark", __localHtml: true });
  const trendRows = baseRows.map((row, rowIndex) => {
    const copy = [...row];
    const sparkHtml = renderSparklineSvg(ksSeries, ksAll[rowIndex]);
    copy.splice(insertAt, 0, sparkHtml);
    return copy;
  });

  return renderEnhancedTableExplicit({
    key: table.key,
    title: table.title,
    headers: trendHeaders,
    column_specs: trendSpecs,
    rows: trendRows,
  });
}

export function renderSparklineSvg(series, currentValue) {
  if (!series || series.length < 2) return "";
  const minV = Math.min(...series);
  const maxV = Math.max(...series);
  const range = Math.max(maxV - minV, 1e-6);
  const W = 120, H = 24, PAD_X = 4, PAD_Y = 4;
  const stepX = (W - 2 * PAD_X) / (series.length - 1);
  const yOf = (v) => H - PAD_Y - ((v - minV) / range) * (H - 2 * PAD_Y);
  const linePath = series.map((v, i) => `${i === 0 ? "M" : "L"}${PAD_X + i * stepX},${yOf(v).toFixed(2)}`).join(" ");
  const baseY = H / 2;
  const points = series.map((v, i) => {
    const cx = PAD_X + i * stepX;
    const isCurrent = currentValue !== null && Math.abs(v - currentValue) < 1e-9;
    return `<circle class="metric-sparkline-point${isCurrent ? " current" : ""}" cx="${cx.toFixed(2)}" cy="${yOf(v).toFixed(2)}" r="${isCurrent ? 3.2 : 2}" data-tip="KS ${v.toFixed(4)}"></circle>`;
  }).join("");
  return `<svg class="metric-sparkline" viewBox="0 0 ${W} ${H}" role="img" aria-label="KS 趋势">`
    + `<line class="metric-sparkline-baseline" x1="${PAD_X}" y1="${baseY}" x2="${W - PAD_X}" y2="${baseY}"></line>`
    + `<path class="metric-sparkline-line" d="${linePath}"></path>`
    + points
    + `</svg>`;
}

// VD-4: probability-calibration reliability curve. Reuses the roc-card plot
// framework (fixed viewBox, PAD-based plot rect, diagonal reference line,
// hover readout) so the same visual language covers ROC/KS and calibration.
// Renders only the "raw" reliability points (pre-calibration, the version a
// reviewer checks against the diagonal) -- points already carry sample_count
// so bubble size communicates bin weight without a second data pass.
export function renderCalibrationCard(chart) {
  const points = (chart && chart.points) || [];
  const W = 280, H = 240, PAD = 28;
  const plot = { x: PAD, y: PAD - 4, w: W - PAD - 8, h: H - PAD - 14 };
  if (points.length === 0) {
    return `<div class="calibration-card"><div class="result-summary empty">暂无校准数据</div></div>`;
  }
  const xOf = (v) => plot.x + Math.max(0, Math.min(1, v)) * plot.w;
  const yOf = (v) => plot.y + (1 - Math.max(0, Math.min(1, v))) * plot.h;
  const sorted = [...points].sort((a, b) => a.avg_predicted_pd - b.avg_predicted_pd);
  const xs = sorted.map((p) => p.avg_predicted_pd);
  const ys = sorted.map((p) => p.observed_bad_rate);
  const linePath = xs.map((x, i) => `${i === 0 ? "M" : "L"}${xOf(x).toFixed(2)},${yOf(ys[i]).toFixed(2)}`).join(" ");
  const diagonalPath = `M${xOf(0)},${yOf(0)} L${xOf(1)},${yOf(1)}`;

  const gridLines = [0.25, 0.5, 0.75].map((t) =>
    `<line class="roc-grid-line" x1="${xOf(t).toFixed(2)}" y1="${plot.y}" x2="${xOf(t).toFixed(2)}" y2="${plot.y + plot.h}"></line>`
    + `<line class="roc-grid-line" x1="${plot.x}" y1="${yOf(t).toFixed(2)}" x2="${plot.x + plot.w}" y2="${yOf(t).toFixed(2)}"></line>`
  ).join("");
  const xLabels = [0, 0.5, 1].map((t) =>
    `<text class="roc-axis-label" x="${xOf(t).toFixed(2)}" y="${H - 2}" text-anchor="middle" font-size="9">${t}</text>`
  ).join("");
  const yLabels = [0, 0.5, 1].map((t) =>
    `<text class="roc-axis-label" x="${plot.x - 4}" y="${(yOf(t) + 3).toFixed(2)}" text-anchor="end" font-size="9">${t}</text>`
  ).join("");

  // Points that deviate from perfect calibration (predicted != actual) are
  // emphasized with the PSI warn/critical palette instead of the neutral
  // accent -- a point exactly on the diagonal needs no visual call-out.
  const markers = sorted.map((p, i) => {
    const gap = Math.abs(p.avg_predicted_pd - p.observed_bad_rate);
    const tier = gap >= 0.1 ? "critical" : gap >= 0.05 ? "warn" : "stable";
    const tip = `预测 ${(p.avg_predicted_pd * 100).toFixed(1)}% · 实际 ${(p.observed_bad_rate * 100).toFixed(1)}% · n=${p.sample_count ?? "-"}`;
    return `<circle class="calibration-point" data-tier="${tier}" data-index="${i}" cx="${xOf(p.avg_predicted_pd).toFixed(2)}" cy="${yOf(p.observed_bad_rate).toFixed(2)}" r="3.6" data-tip="${escapeHtml(tip)}"></circle>`;
  }).join("");

  const brierRaw = chart.brier_raw;
  const eceRaw = chart.ece_raw;
  const summaryTip = `Brier=${brierRaw === null || brierRaw === undefined ? "-" : Number(brierRaw).toFixed(4)}`
    + ` · ECE=${eceRaw === null || eceRaw === undefined ? "-" : Number(eceRaw).toFixed(4)}`;

  return [
    `<div class="calibration-card">`,
    `  <div class="roc-card-header">`,
    `    <span class="roc-card-split">可靠性曲线</span>`,
    `    <span class="roc-card-ks" data-tip="${escapeHtml(summaryTip)}">${escapeHtml(summaryTip)}</span>`,
    `  </div>`,
    `  <svg class="roc-svg calibration-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="概率校准可靠性曲线"`,
    `       data-cal-x="${escapeHtml(JSON.stringify(xs))}" data-cal-y="${escapeHtml(JSON.stringify(ys))}"`,
    `       data-roc-plot-x="${plot.x}" data-roc-plot-y="${plot.y}" data-roc-plot-w="${plot.w}" data-roc-plot-h="${plot.h}">`,
    `    ${gridLines}`,
    `    <line class="roc-axis" x1="${plot.x}" y1="${plot.y + plot.h}" x2="${plot.x + plot.w}" y2="${plot.y + plot.h}"></line>`,
    `    <line class="roc-axis" x1="${plot.x}" y1="${plot.y}" x2="${plot.x}" y2="${plot.y + plot.h}"></line>`,
    `    <path class="roc-curve roc-curve-baseline" data-series="baseline" d="${diagonalPath}"></path>`,
    `    <path class="roc-curve calibration-curve-line" d="${linePath}"></path>`,
    markers,
    `    <line class="roc-crosshair roc-crosshair-x" x1="0" y1="0" x2="0" y2="0" style="display:none"></line>`,
    `    ${xLabels}`,
    `    ${yLabels}`,
    `  </svg>`,
    `  <div class="roc-readout calibration-readout" data-cal-readout>移动鼠标到图上查看预测概率 / 实际坏率</div>`,
    `</div>`,
  ].join("");
}

// VD-4: score-band distribution -- grouped bars for sample_count (left axis)
// with a bad-rate line overlaid on a normalized right axis (bad_rate's own
// max), following the sparkline's simple polyline style rather than the
// ROC card's crosshair interaction (bars carry their own per-band hover).
export function renderScoreBandCard(chart) {
  const bands = (chart && chart.bands) || [];
  const W = 320, H = 220, PAD_L = 34, PAD_R = 30, PAD_T = 14, PAD_B = 34;
  const plot = { x: PAD_L, y: PAD_T, w: W - PAD_L - PAD_R, h: H - PAD_T - PAD_B };
  if (bands.length === 0) {
    return `<div class="score-band-card"><div class="result-summary empty">暂无分段数据</div></div>`;
  }
  const counts = bands.map((b) => Number(b.sample_count) || 0);
  const rates = bands.map((b) => (b.bad_rate === null || b.bad_rate === undefined ? 0 : Number(b.bad_rate)));
  const maxCount = Math.max(...counts, 1);
  const maxRate = Math.max(...rates, 1e-6);

  const bandWidth = plot.w / bands.length;
  const barGap = Math.min(6, bandWidth * 0.18);
  const barWidth = Math.max(2, bandWidth - barGap);
  const yCount = (v) => plot.y + plot.h - (v / maxCount) * plot.h;
  const yRate = (v) => plot.y + plot.h - (v / maxRate) * plot.h;
  const xCenter = (i) => plot.x + i * bandWidth + bandWidth / 2;

  const bars = bands.map((b, i) => {
    const count = counts[i];
    const barX = xCenter(i) - barWidth / 2;
    const barY = yCount(count);
    const barH = Math.max(0, plot.y + plot.h - barY);
    const tip = `分箱${b.bin ?? i + 1} · 样本量 ${count} · 坏率 ${(rates[i] * 100).toFixed(2)}%`;
    return `<rect class="score-band-bar" data-index="${i}" x="${barX.toFixed(2)}" y="${barY.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${barH.toFixed(2)}" data-tip="${escapeHtml(tip)}"></rect>`;
  }).join("");

  const linePath = bands.map((b, i) => `${i === 0 ? "M" : "L"}${xCenter(i).toFixed(2)},${yRate(rates[i]).toFixed(2)}`).join(" ");
  const linePoints = bands.map((b, i) =>
    `<circle class="score-band-rate-point" data-index="${i}" cx="${xCenter(i).toFixed(2)}" cy="${yRate(rates[i]).toFixed(2)}" r="2.6" data-tip="坏率 ${(rates[i] * 100).toFixed(2)}%"></circle>`
  ).join("");

  const xLabels = bands.map((b, i) =>
    `<text class="roc-axis-label score-band-x-label" x="${xCenter(i).toFixed(2)}" y="${H - PAD_B + 12}" text-anchor="middle" font-size="8">${escapeHtml(String(b.bin ?? i + 1))}</text>`
  ).join("");
  const yCountLabels = [0, 0.5, 1].map((t) =>
    `<text class="roc-axis-label" x="${plot.x - 4}" y="${(yCount(t * maxCount) + 3).toFixed(2)}" text-anchor="end" font-size="8">${Math.round(t * maxCount)}</text>`
  ).join("");
  const yRateLabels = [0, 0.5, 1].map((t) =>
    `<text class="roc-axis-label" x="${plot.x + plot.w + 4}" y="${(yRate(t * maxRate) + 3).toFixed(2)}" text-anchor="start" font-size="8">${(t * maxRate * 100).toFixed(1)}%</text>`
  ).join("");

  return [
    `<div class="score-band-card" data-split="${escapeHtml(chart.split || "")}">`,
    `  <div class="roc-card-header">`,
    `    <span class="roc-card-split">${escapeHtml(chart.split || "")} 分段分布</span>`,
    `    <span class="score-band-legend">`,
    `      <span class="score-band-legend-item"><i class="score-band-legend-bar"></i>样本量</span>`,
    `      <span class="score-band-legend-item"><i class="score-band-legend-line"></i>坏率</span>`,
    `    </span>`,
    `  </div>`,
    `  <svg class="roc-svg score-band-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="分数分段样本量与坏率">`,
    `    <line class="roc-axis" x1="${plot.x}" y1="${plot.y + plot.h}" x2="${plot.x + plot.w}" y2="${plot.y + plot.h}"></line>`,
    `    <line class="roc-axis" x1="${plot.x}" y1="${plot.y}" x2="${plot.x}" y2="${plot.y + plot.h}"></line>`,
    bars,
    `    <path class="score-band-rate-line" d="${linePath}"></path>`,
    linePoints,
    xLabels,
    yCountLabels,
    yRateLabels,
    `    <text class="roc-axis-label" x="${plot.x}" y="${PAD_T - 4}" text-anchor="start" font-size="8">样本量</text>`,
    `    <text class="roc-axis-label" x="${plot.x + plot.w}" y="${PAD_T - 4}" text-anchor="end" font-size="8">坏率</text>`,
    `  </svg>`,
    `</div>`,
  ].join("");
}

// Shared hover readout for the calibration curve, mirroring attachRocInteractions'
// crosshair pattern but simplified to a single-series x-axis (predicted PD).
export function attachCalibrationInteractions(rootEl) {
  if (!rootEl) return;
  rootEl.querySelectorAll(".calibration-card").forEach((card) => {
    const svg = card.querySelector(".calibration-svg");
    const readout = card.querySelector("[data-cal-readout]");
    if (!svg) return;
    const xs = JSON.parse(svg.getAttribute("data-cal-x") || "[]");
    const ys = JSON.parse(svg.getAttribute("data-cal-y") || "[]");
    const px = Number(svg.getAttribute("data-roc-plot-x"));
    const pw = Number(svg.getAttribute("data-roc-plot-w"));
    const xLine = svg.querySelector(".roc-crosshair-x");

    const hide = () => {
      if (xLine) xLine.style.display = "none";
      if (readout) readout.textContent = "移动鼠标到图上查看预测概率 / 实际坏率";
    };

    svg.addEventListener("mousemove", (event) => {
      const rect = svg.getBoundingClientRect();
      const viewBox = svg.viewBox.baseVal;
      const xViewbox = ((event.clientX - rect.left) / rect.width) * viewBox.width;
      const predicted = (xViewbox - px) / pw;
      if (predicted < 0 || predicted > 1 || xs.length === 0) { hide(); return; }
      let nearestIdx = 0;
      let nearestDist = Infinity;
      for (let i = 0; i < xs.length; i++) {
        const d = Math.abs(xs[i] - predicted);
        if (d < nearestDist) { nearestDist = d; nearestIdx = i; }
      }
      if (readout) {
        readout.textContent = `预测 ${(xs[nearestIdx] * 100).toFixed(1)}%  ·  实际 ${(ys[nearestIdx] * 100).toFixed(1)}%`;
      }
    });
    svg.addEventListener("mouseleave", hide);
  });
}

export function renderRocKsCurve(table = {}) {
  const curves = table.curves || {};
  const splits = ["train", "test", "oot"].filter((split) => curves[split]);
  if (splits.length === 0) {
    return `<div class="metric-table-wrap"><div class="result-summary empty">暂无 ROC&KS 曲线数据</div></div>`;
  }
  const cards = splits.map((split) => renderRocCard(split, curves[split])).join("");
  return [
    `<div class="metric-table-wrap" data-metric-key="${escapeHtml(table.key || "ROC_KS_CURVES")}">`,
    `<div class="roc-grid" style="--roc-count:${splits.length}">${cards}</div>`,
    `</div>`,
  ].join("");
}

export function renderRocCard(split, curve) {
  const W = 280, H = 240, PAD = 28;
  const plot = { x: PAD, y: PAD - 4, w: W - PAD - 8, h: H - PAD - 14 };
  const fpr = curve.fpr || [];
  const tpr = curve.tpr || [];
  const ks = curve.ks_curve || [];
  if (fpr.length < 2) {
    return `<div class="roc-card"><div class="roc-card-header"><span class="roc-card-split">${escapeHtml(split)}</span></div><div class="result-summary empty">无曲线数据</div></div>`;
  }
  const xOf = (v) => plot.x + Math.max(0, Math.min(1, v)) * plot.w;
  const yOf = (v) => plot.y + (1 - Math.max(0, Math.min(1, v))) * plot.h;
  const buildPath = (xs, ys) => xs.map((x, i) => `${i === 0 ? "M" : "L"}${xOf(x).toFixed(2)},${yOf(ys[i]).toFixed(2)}`).join(" ");

  const tprPath = buildPath(fpr, tpr);
  const diagonalPath = `M${xOf(0)},${yOf(0)} L${xOf(1)},${yOf(1)}`;
  const ksPath = ks.length === fpr.length ? buildPath(fpr, ks) : "";
  // KS marker sits on the FPR axis: anchor it at fpr[argmax(|ks_curve|)]. Using
  // population_at_ks (a different axis) misplaces the line on imbalanced data.
  const ksArgmax = ks.length
    ? ks.reduce((best, value, i) => (Math.abs(value) > Math.abs(ks[best]) ? i : best), 0)
    : 0;
  const ksMarkerX = xOf(fpr[ksArgmax] ?? 0);

  const gridLines = [0.25, 0.5, 0.75].map((t) =>
    `<line class="roc-grid-line" x1="${xOf(t).toFixed(2)}" y1="${plot.y}" x2="${xOf(t).toFixed(2)}" y2="${plot.y + plot.h}"></line>`
    + `<line class="roc-grid-line" x1="${plot.x}" y1="${yOf(t).toFixed(2)}" x2="${plot.x + plot.w}" y2="${yOf(t).toFixed(2)}"></line>`
  ).join("");

  const xLabels = [0, 0.5, 1].map((t) =>
    `<text class="roc-axis-label" x="${xOf(t).toFixed(2)}" y="${H - 2}" text-anchor="middle" font-size="9">${t}</text>`
  ).join("");
  const yLabels = [0, 0.5, 1].map((t) =>
    `<text class="roc-axis-label" x="${plot.x - 4}" y="${(yOf(t) + 3).toFixed(2)}" text-anchor="end" font-size="9">${t}</text>`
  ).join("");

  return [
    `<div class="roc-card" data-split="${escapeHtml(split)}">`,
    `  <div class="roc-card-header">`,
    `    <span class="roc-card-split">${escapeHtml(split)}</span>`,
    `    <span class="roc-card-ks" data-tip="KS=${(curve.ks ?? 0).toFixed(4)} at population=${(curve.population_at_ks ?? 0).toFixed(2)}">KS ${(curve.ks ?? 0).toFixed(4)}</span>`,
    `  </div>`,
    `  <svg class="roc-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="ROC and KS curves for ${escapeHtml(split)}"`,
    `       data-roc-fpr="${escapeHtml(JSON.stringify(fpr))}" data-roc-tpr="${escapeHtml(JSON.stringify(tpr))}" data-roc-ks="${escapeHtml(JSON.stringify(ks))}"`,
    `       data-roc-plot-x="${plot.x}" data-roc-plot-y="${plot.y}" data-roc-plot-w="${plot.w}" data-roc-plot-h="${plot.h}">`,
    `    ${gridLines}`,
    `    <line class="roc-axis" x1="${plot.x}" y1="${plot.y + plot.h}" x2="${plot.x + plot.w}" y2="${plot.y + plot.h}"></line>`,
    `    <line class="roc-axis" x1="${plot.x}" y1="${plot.y}" x2="${plot.x}" y2="${plot.y + plot.h}"></line>`,
    `    <path class="roc-curve roc-curve-baseline" data-series="baseline" d="${diagonalPath}"></path>`,
    `    <path class="roc-curve roc-curve-tpr" data-series="tpr" d="${tprPath}"></path>`,
    ksPath ? `    <path class="roc-curve roc-curve-ks" data-series="ks" d="${ksPath}"></path>` : "",
    `    <line class="roc-ks-marker" data-series="ks-marker" x1="${ksMarkerX.toFixed(2)}" y1="${plot.y}" x2="${ksMarkerX.toFixed(2)}" y2="${plot.y + plot.h}" data-tip="KS=${(curve.ks ?? 0).toFixed(4)} at population=${(curve.population_at_ks ?? 0).toFixed(2)}"></line>`,
    `    <line class="roc-crosshair roc-crosshair-x" x1="0" y1="0" x2="0" y2="0" style="display:none"></line>`,
    `    <line class="roc-crosshair roc-crosshair-y" x1="0" y1="0" x2="0" y2="0" style="display:none"></line>`,
    `    ${xLabels}`,
    `    ${yLabels}`,
    `  </svg>`,
    `  <div class="roc-legend" role="group" aria-label="切换曲线显示">`,
    `    <button type="button" class="roc-legend-tpr" data-roc-toggle="tpr" aria-pressed="true"><i></i>TPR</button>`,
    `    <button type="button" class="roc-legend-baseline" data-roc-toggle="baseline" aria-pressed="true"><i></i>Random Baseline</button>`,
    `    <button type="button" class="roc-legend-ks" data-roc-toggle="ks" aria-pressed="true"><i></i>KS Curve</button>`,
    `  </div>`,
    `  <div class="roc-readout" data-roc-readout>移动鼠标到图上查看 FPR / TPR / KS</div>`,
    `</div>`,
  ].join("");
}

export function attachRocInteractions(rootEl) {
  if (!rootEl) return;
  rootEl.querySelectorAll(".roc-card").forEach((card) => {
    const svg = card.querySelector(".roc-svg");
    const readout = card.querySelector("[data-roc-readout]");
    if (!svg) return;
    const fpr = JSON.parse(svg.getAttribute("data-roc-fpr") || "[]");
    const tpr = JSON.parse(svg.getAttribute("data-roc-tpr") || "[]");
    const ks  = JSON.parse(svg.getAttribute("data-roc-ks")  || "[]");
    const px = Number(svg.getAttribute("data-roc-plot-x"));
    const py = Number(svg.getAttribute("data-roc-plot-y"));
    const pw = Number(svg.getAttribute("data-roc-plot-w"));
    const ph = Number(svg.getAttribute("data-roc-plot-h"));
    const xLine = svg.querySelector(".roc-crosshair-x");
    const yLine = svg.querySelector(".roc-crosshair-y");

    const hideCrosshair = () => {
      xLine.style.display = "none";
      yLine.style.display = "none";
      readout.textContent = "移动鼠标到图上查看 FPR / TPR / KS";
    };

    const onMove = (event) => {
      const rect = svg.getBoundingClientRect();
      const viewBox = svg.viewBox.baseVal;
      const xViewbox = ((event.clientX - rect.left) / rect.width) * viewBox.width;
      const fprVal = (xViewbox - px) / pw;
      if (fprVal < 0 || fprVal > 1) { hideCrosshair(); return; }
      let nearestIdx = 0;
      let nearestDist = Infinity;
      for (let i = 0; i < fpr.length; i++) {
        const d = Math.abs(fpr[i] - fprVal);
        if (d < nearestDist) { nearestDist = d; nearestIdx = i; }
      }
      const xPos = px + fpr[nearestIdx] * pw;
      const yPosTpr = py + (1 - tpr[nearestIdx]) * ph;
      xLine.setAttribute("x1", xPos); xLine.setAttribute("x2", xPos);
      xLine.setAttribute("y1", py); xLine.setAttribute("y2", py + ph);
      xLine.style.display = "";
      yLine.setAttribute("x1", px); yLine.setAttribute("x2", px + pw);
      yLine.setAttribute("y1", yPosTpr); yLine.setAttribute("y2", yPosTpr);
      yLine.style.display = "";
      const ksHere = ks[nearestIdx] ?? Math.abs(tpr[nearestIdx] - fpr[nearestIdx]);
      readout.textContent = `FPR ${fpr[nearestIdx].toFixed(3)}  ·  TPR ${tpr[nearestIdx].toFixed(3)}  ·  KS ${ksHere.toFixed(3)}`;
    };

    svg.addEventListener("mousemove", onMove);
    svg.addEventListener("mouseleave", hideCrosshair);

    card.querySelectorAll("[data-roc-toggle]").forEach((button) => {
      button.addEventListener("click", () => {
        const series = button.getAttribute("data-roc-toggle");
        const pressed = button.getAttribute("aria-pressed") === "true";
        const next = !pressed;
        button.setAttribute("aria-pressed", String(next));
        svg.querySelectorAll(`[data-series="${series}"], [data-series="${series}-marker"]`).forEach((el) => {
          el.style.display = next ? "" : "none";
        });
      });
    });
  });
}

export let metricTooltipAttached = false;

export function attachMetricTooltip(rootEl) {
  if (metricTooltipAttached || !rootEl) return;
  const tooltip = document.getElementById("metricTooltip");
  if (!tooltip) return;
  metricTooltipAttached = true;
  let currentTarget = null;
  const positionTooltip = (el, event) => {
    const pad = 12;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    const rect = el.getBoundingClientRect();
    if (x + rect.width > window.innerWidth) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight) y = event.clientY - rect.height - pad;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
  };
  const show = (target, event) => {
    const tip = target.getAttribute("data-tip");
    if (!tip) return;
    currentTarget = target;
    tooltip.textContent = tip;
    tooltip.hidden = false;
    positionTooltip(tooltip, event);
  };
  const hide = () => {
    currentTarget = null;
    tooltip.hidden = true;
  };
  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest("#metricPreview [data-tip]");
    if (target) show(target, event);
  });
  document.addEventListener("mousemove", (event) => {
    if (currentTarget && currentTarget.contains(event.target)) {
      positionTooltip(tooltip, event);
    }
  });
  document.addEventListener("mouseout", (event) => {
    if (!currentTarget) return;
    const next = event.relatedTarget;
    if (!next || !currentTarget.contains(next)) hide();
  });
  document.addEventListener("scroll", hide, { passive: true, capture: true });
}

export function renderEnhancedTable(table = {}, options = {}) {
  return renderEnhancedTableExplicit(table, options);
}

export function renderEnhancedTableExplicit(table, options = {}) {
  const headers = Array.isArray(table.headers) ? table.headers : [];
  const rows = Array.isArray(table.rows) ? table.rows : [];
  const specs = Array.isArray(table.column_specs) ? table.column_specs : [];
  const columnCount = headers.length;
  // VD-9: default to animating unless a caller explicitly opts out.
  const animationAttribute = options.animate === false ? ' data-animation="none"' : "";

  const fractionsByColumn = new Map();
  const heatByColumn = new Map();
  const ranksByColumn = new Map();
  for (let i = 0; i < columnCount; i++) {
    const kind = (specs[i] && specs[i].kind) || "text";
    if (kind === "databar" || kind === "databar-primary") {
      fractionsByColumn.set(i, columnFractions(rows, i));
      ranksByColumn.set(i, columnRanks(rows, i));
    }
    if (kind === "percent-heat") {
      heatByColumn.set(i, columnHeatColors(rows, i));
    }
  }

  // Columns whose text-only cells should hug the left edge (e.g.
  // 特征 in 特征重要性). Numeric primitives stay centered regardless.
  const leftAlignCols = new Set();
  headers.forEach((header, i) => {
    if (header === "特征") leftAlignCols.add(i);
  });

  const bodyRows = rows.length === 0
    ? `<tr class="metric-table-empty"><td colspan="${Math.max(columnCount, 1)}">暂无数据</td></tr>`
    : rows.map((row, rowIndex) => {
        const cells = [];
        for (let columnIndex = 0; columnIndex < columnCount; columnIndex++) {
          const value = Array.isArray(row) ? row[columnIndex] : "";
          const spec = specs[columnIndex] || { kind: "text" };
          const cell = renderCellByKind(spec, value, {
            rowIndex,
            columnIndex,
            headerLabel: headers[columnIndex] ?? "",
            fractions: fractionsByColumn.get(columnIndex) || new Map(),
            heatColors: heatByColumn.get(columnIndex) || new Map(),
            ranks: ranksByColumn.get(columnIndex) || new Map(),
          });
          const alignAttr = leftAlignCols.has(columnIndex) && (spec.kind || "text") === "text"
            ? ' data-align="left"'
            : "";
          cells.push(`<td class="${cell.cls}"${alignAttr}>${cell.html}</td>`);
        }
        return `<tr>${cells.join("")}</tr>`;
      }).join("");

  return [
    `<div class="metric-table-wrap"${animationAttribute} data-metric-key="${escapeHtml(table.key || "")}">`,
    '<div class="metric-table-scroll">',
    '<table class="metric-table metric-table-hoverable">',
    '<thead><tr>',
    ...headers.map((header, i) => {
      const alignAttr = leftAlignCols.has(i) ? ' data-align="left"' : "";
      return `<th${alignAttr}>${escapeHtml(header)}</th>`;
    }),
    '</tr></thead>',
    `<tbody>${bodyRows}</tbody>`,
    "</table>",
    "</div>",
    "</div>",
  ].join("");
}

export function renderLegacyTable(table = {}) {
  const headers = Array.isArray(table.headers) ? table.headers : [];
  const rows = Array.isArray(table.rows) ? table.rows : [];
  const columnCount = Math.max(
    headers.length,
    ...rows.map((row) => (Array.isArray(row) ? row.length : 0)),
    1,
  );
  const bodyRows = rows.length > 0
    ? rows.map((row) => renderMetricTableRow(row, columnCount))
    : [`<tr class="metric-table-empty"><td colspan="${columnCount}">暂无数据</td></tr>`];
  return [
    `<div class="metric-table-wrap" data-metric-key="${escapeHtml(table.key || "")}">`,
    `<div class="metric-table-caption">${escapeHtml(table.title || "指标明细")}</div>`,
    '<div class="metric-table-scroll">',
    '<table class="metric-table">',
    "<thead><tr>",
    ...Array.from(
      { length: columnCount },
      (_, index) => `<th>${escapeHtml(headers[index] ?? "")}</th>`,
    ),
    "</tr></thead>",
    `<tbody>${bodyRows.join("")}</tbody>`,
    "</table>",
    "</div>",
    "</div>",
  ].join("");
}

export function renderMetricTableRow(row, columnCount) {
  const cells = Array.from({ length: columnCount }, (_, index) => {
    const value = Array.isArray(row) ? row[index] : "";
    return `<td>${escapeHtml(value === null || value === undefined ? "" : String(value))}</td>`;
  });
  return `<tr>${cells.join("")}</tr>`;
}
