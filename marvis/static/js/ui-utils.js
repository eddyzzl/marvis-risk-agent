import { localizeHtml, localizeText } from "./i18n.js";

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

// Polling re-renders unchanged markup every second or so. Replacing equal
// nodes restarts animations and image decoding, which reads as the page
// flashing (2026-10-20 user feedback), so these write only on a real change.
export function setHtmlIfChanged(element, html) {
  if (!element) return false;
  const markup = String(html ?? "");
  const probe = (element.ownerDocument || globalThis.document)?.createElement?.("template");
  if (probe && "innerHTML" in probe) {
    probe.innerHTML = markup;
    // D-10: compare with what the markup looks like in the current UI locale,
    // so English mode does not rewrite unchanged markup on every poll.
    if (localizeHtml(probe.innerHTML, { context: element }) === element.innerHTML) return false;
  }
  element.innerHTML = markup;
  return true;
}

export function setTextIfChanged(element, text) {
  if (!element) return false;
  const value = String(text ?? "");
  if (element.textContent === value || element.textContent === localizeText(value)) return false;
  element.textContent = value;
  return true;
}
