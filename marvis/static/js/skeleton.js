// VD-3: shared loading-skeleton templates. Three shapes only — block, rows,
// table — reusing the `.skeleton` shimmer primitive in styles.css. Callers
// swap these in for a blank/spinner-only container on first load / genuine
// state transitions; polling updates should keep diffing in place instead of
// re-flashing a skeleton every tick (see call sites in app.js / plan_rail_controller.js).

export function skeletonBlockHtml({ height = 16, width = "100%" } = {}) {
  const widthCss = typeof width === "number" ? `${width}px` : String(width);
  return `<div class="skeleton skeleton-block" style="height:${height}px;width:${widthCss}"></div>`;
}

export function skeletonRowsHtml({ rows = 3, height = 14 } = {}) {
  return [
    '<div class="skeleton-rows">',
    ...Array.from({ length: Math.max(1, rows) }, (_, index) => {
      const width = index === Math.max(1, rows) - 1 ? "60%" : "100%";
      return `<div class="skeleton skeleton-row" style="height:${height}px;width:${width}"></div>`;
    }),
    "</div>",
  ].join("");
}

export function skeletonTableHtml({ rows = 4, columns = 4 } = {}) {
  const columnCells = Array.from({ length: Math.max(1, columns) }, () => (
    '<span class="skeleton skeleton-table-cell"></span>'
  )).join("");
  return [
    '<div class="skeleton-table">',
    '<div class="skeleton-table-header">',
    Array.from({ length: Math.max(1, columns) }, () => (
      '<span class="skeleton skeleton-table-cell skeleton-table-cell-head"></span>'
    )).join(""),
    "</div>",
    ...Array.from({ length: Math.max(1, rows) }, () => (
      `<div class="skeleton-table-row">${columnCells}</div>`
    )),
    "</div>",
  ].join("");
}
