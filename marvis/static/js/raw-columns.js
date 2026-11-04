// Table headers whose cells are user data (feature, field, file names): the UI
// never translates those cells. Single source for the browser (metric-tables.js)
// and the server (marvis/ui_locale.py reads the JSON array below), so the two
// layers cannot drift. Matching is case-insensitive on the trimmed header.
export const RAW_COLUMN_HEADERS = Object.freeze(/* raw-columns:start */[
  "特征", "特征名", "特征名称", "变量", "字段", "字段名", "验证字段", "类别", "未分类特征",
  "名称", "文件", "文件名", "路径",
  "feature", "feature name", "variable", "field", "field name", "validation field",
  "validation fields", "category", "name", "file", "file name", "filename", "path",
  "column", "column name"
]/* raw-columns:end */);

const RAW_COLUMN_SET = new Set(RAW_COLUMN_HEADERS.map((label) => label.toLowerCase()));

export function isRawColumnHeader(header) {
  return RAW_COLUMN_SET.has(String(header ?? "").trim().toLowerCase());
}
