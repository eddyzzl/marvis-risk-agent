export function $(id) {
  return document.getElementById(id);
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function fileName(path) {
  return String(path || "").split(/[\\/]/).filter(Boolean).pop() || path || "";
}

export function formatDateInput(value = new Date()) {
  const date = value instanceof Date ? value : new Date(value);
  return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
}

export function splitListInput(value) {
  return String(value || "")
    .split(/[\n,，]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function signatureFromParts(parts) {
  return JSON.stringify(parts.map((part) => (part === undefined ? null : part)));
}

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
