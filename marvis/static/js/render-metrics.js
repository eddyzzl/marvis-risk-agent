export function parseNumeric(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const text = String(value).trim().replace(/,/g, "");
  if (!text || text === "-") return null;
  if (text.endsWith("%")) {
    const inner = Number(text.slice(0, -1));
    return Number.isFinite(inner) ? inner / 100 : null;
  }
  const num = Number(text);
  return Number.isFinite(num) ? num : null;
}

export function columnNumerics(rows, columnIndex) {
  return rows
    .map((row, index) => [index, parseNumeric(Array.isArray(row) ? row[columnIndex] : null)])
    .filter(([, value]) => value !== null);
}

export function columnFractions(rows, columnIndex) {
  const indexed = columnNumerics(rows, columnIndex);
  if (indexed.length === 0) return new Map();
  const values = indexed.map(([, v]) => v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const result = new Map();
  for (const [rowIndex, value] of indexed) {
    let fraction;
    if (max === min) fraction = value ? 1 : 0;
    else if (min >= 0 && max > 0) fraction = value / max;
    else fraction = (value - min) / (max - min);
    result.set(rowIndex, fraction);
  }
  return result;
}

export function heatColor(value, min, mid, max) {
  if (max === min) return "#FFEB84";
  const lerp = (a, b, t) => Math.round(a + (b - a) * Math.max(0, Math.min(1, t)));
  const GREEN = [0x63, 0xBE, 0x7B];
  const YELLOW = [0xFF, 0xEB, 0x84];
  const RED = [0xF8, 0x69, 0x6B];
  let r, g, b;
  if (value <= mid) {
    const t = mid === min ? 0 : (value - min) / (mid - min);
    [r, g, b] = [lerp(GREEN[0], YELLOW[0], t), lerp(GREEN[1], YELLOW[1], t), lerp(GREEN[2], YELLOW[2], t)];
  } else {
    const t = max === mid ? 1 : (value - mid) / (max - mid);
    [r, g, b] = [lerp(YELLOW[0], RED[0], t), lerp(YELLOW[1], RED[1], t), lerp(YELLOW[2], RED[2], t)];
  }
  return `rgb(${r}, ${g}, ${b})`;
}

export function columnHeatColors(rows, columnIndex) {
  const indexed = columnNumerics(rows, columnIndex);
  if (indexed.length === 0) return new Map();
  const sorted = [...indexed].map(([, v]) => v).sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const mid = sorted[Math.floor(sorted.length / 2)];
  const result = new Map();
  for (const [rowIndex, value] of indexed) {
    result.set(rowIndex, heatColor(value, min, mid, max));
  }
  return result;
}

export function columnRanks(rows, columnIndex) {
  const indexed = columnNumerics(rows, columnIndex);
  if (indexed.length === 0) return new Map();
  const sorted = [...indexed].sort((a, b) => b[1] - a[1]);
  const ranks = new Map();
  let lastValue = null;
  let lastRank = 0;
  sorted.forEach(([rowIndex, value], i) => {
    const rank = value === lastValue ? lastRank : i + 1;
    ranks.set(rowIndex, `#${rank} of ${indexed.length}`);
    lastValue = value;
    lastRank = rank;
  });
  return ranks;
}

export function psiTier(value, thresholds) {
  if (value === null) return "base";
  const [warnAt, critAt] = thresholds;
  if (Math.abs(value) >= critAt) return "critical";
  if (Math.abs(value) >= warnAt) return "warn";
  return "stable";
}

export function psiTooltipText(value, thresholds) {
  const [warnAt, critAt] = thresholds;
  if (value === null) return `PSI · 基线\n阈值：稳定 <${warnAt} · 可接受 <${critAt} · 漂移 ≥${critAt}`;
  const tier = psiTier(value, thresholds);
  const tierLabel = { stable: "稳定", warn: "可接受", critical: "漂移显著", base: "基线" }[tier];
  return `PSI ${value.toFixed(4)}\n当前：${tierLabel}\n阈值：稳定 <${warnAt} · 可接受 <${critAt} · 漂移 ≥${critAt}`;
}
