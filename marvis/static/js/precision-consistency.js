import { escapeHtml } from "./ui-utils.js";

export function roundedScoresMatch(left, right, decimals) {
  const leftNumber = Number(left);
  const rightNumber = Number(right);
  if (!Number.isFinite(leftNumber) || !Number.isFinite(rightNumber)) return false;
  return leftNumber.toFixed(decimals) === rightNumber.toFixed(decimals);
}

export function precisionConsistencyTier(rate) {
  if (rate >= 99.5) return "exact";
  if (rate >= 90) return "strong";
  if (rate >= 50) return "fair";
  return "weak";
}

export function buildPrecisionConsistencyBars(rows = []) {
  const total = rows.length;
  const bars = [];
  for (let decimals = 1; decimals <= 6; decimals += 1) {
    const matchCount = rows.filter((row) => (
      roundedScoresMatch(row.score_code_model, row.score_submitted_pmml, decimals)
    )).length;
    const rate = total > 0 ? (matchCount / total) * 100 : 0;
    bars.push({
      decimals,
      matchCount,
      total,
      rate,
      tier: precisionConsistencyTier(rate),
    });
  }
  return bars;
}

export function formatPercentValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0%";
  return `${number.toFixed(1)}%`;
}

// Compact label for the bar tops: drop the decimal at 100% so it fits a narrow
// column; the exact one-decimal value still shows in the hover tooltip.
export function formatPrecisionLabel(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0%";
  if (number >= 99.95) return "100%";
  return `${number.toFixed(1)}%`;
}

export function renderPrecisionConsistencyChart(rows = [], options = {}) {
  if (!rows.length) return "";
  const animationAttribute = options.animate === false ? ' data-animation="none"' : "";
  const bars = buildPrecisionConsistencyBars(rows);
  const barItems = bars.map((bar, index) => {
    const height = bar.rate > 0 ? Math.max(2, Math.min(100, bar.rate)) : 0;
    const title = `${bar.decimals} 位小数：${bar.matchCount}/${bar.total} 行一致（${formatPercentValue(bar.rate)}）`;
    return [
      `<div class="score-precision-bar-item" data-tier="${bar.tier}"${bar.rate > 0 ? "" : ' data-empty="true"'} title="${escapeHtml(title)}">`,
      `<span class="score-precision-value">${escapeHtml(formatPrecisionLabel(bar.rate))}</span>`,
      '<span class="score-precision-bar-track" aria-hidden="true">',
      `<span class="score-precision-bar" style="height: ${height}%; --bar-index: ${index}"></span>`,
      "</span>",
      `<span class="score-precision-label">${bar.decimals}位</span>`,
      "</div>",
    ].join("");
  });
  return [
    `<div class="score-precision-chart"${animationAttribute}>`,
    '<div class="score-precision-chart-head">',
    "<strong>四舍五入一致率</strong>",
    `<span class="score-precision-meta">${escapeHtml(rows.length)} 行 · 保留小数位越多越严格</span>`,
    "</div>",
    '<div class="score-precision-plot">',
    '<div class="score-precision-axis" aria-hidden="true">',
    "<span>100%</span>",
    "<span>75%</span>",
    "<span>50%</span>",
    "<span>25%</span>",
    "<span>0%</span>",
    "</div>",
    `<div class="score-precision-bars">${barItems.join("")}</div>`,
    "</div>",
    "</div>",
  ].join("");
}
