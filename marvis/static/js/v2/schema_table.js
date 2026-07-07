import { escapeHtml } from "../ui-utils.js";

function schemaProperties(schema) {
  const properties = schema?.properties;
  return properties && typeof properties === "object" && !Array.isArray(properties) ? properties : {};
}

function schemaRequired(schema) {
  return new Set(Array.isArray(schema?.required) ? schema.required.map(String) : []);
}

export function schemaType(schema) {
  const type = schema?.type;
  if (Array.isArray(type)) return type.join(" | ");
  if (type) return String(type);
  if (schema?.enum) return "enum";
  if (schema?.items) return "array";
  if (schema?.properties) return "object";
  return "any";
}

function schemaConstraintText(schema = {}) {
  const parts = [];
  if (Array.isArray(schema.enum) && schema.enum.length) {
    parts.push(`可选：${schema.enum.map((item) => String(item)).join(" / ")}`);
  }
  if (schema.minimum !== undefined) parts.push(`最小 ${schema.minimum}`);
  if (schema.maximum !== undefined) parts.push(`最大 ${schema.maximum}`);
  if (schema.minLength !== undefined) parts.push(`最短 ${schema.minLength}`);
  if (schema.maxLength !== undefined) parts.push(`最长 ${schema.maxLength}`);
  if (schema.minItems !== undefined) parts.push(`至少 ${schema.minItems} 项`);
  if (schema.maxItems !== undefined) parts.push(`最多 ${schema.maxItems} 项`);
  if (schema.items) parts.push(`元素 ${schemaType(schema.items)}`);
  if (schema.description) parts.push(String(schema.description));
  return parts.join("；") || "—";
}

function schemaFieldRows(schema = {}) {
  const properties = schemaProperties(schema);
  const required = schemaRequired(schema);
  return Object.entries(properties).map(([name, fieldSchema]) => ({
    name,
    type: schemaType(fieldSchema || {}),
    required: required.has(name),
    detail: schemaConstraintText(fieldSchema || {}),
  }));
}

export function schemaTableHtml(schema, label) {
  const rows = schemaFieldRows(schema);
  if (!rows.length) {
    return `<div class="plugin-schema-empty">${escapeHtml(label)}无字段要求。</div>`;
  }
  return [
    '<div class="plugin-schema-table-wrap">',
    '<table class="plugin-schema-table">',
    `<caption>${escapeHtml(label)}</caption>`,
    "<thead><tr><th>字段</th><th>类型</th><th>必填</th><th>约束 / 说明</th></tr></thead>",
    "<tbody>",
    rows
      .map((row) => (
        `<tr>
          <td><code>${escapeHtml(row.name)}</code></td>
          <td>${escapeHtml(row.type)}</td>
          <td>${row.required ? '<span class="plugin-required">是</span>' : "否"}</td>
          <td>${escapeHtml(row.detail)}</td>
        </tr>`
      ))
      .join(""),
    "</tbody>",
    "</table>",
    "</div>",
  ].join("");
}
