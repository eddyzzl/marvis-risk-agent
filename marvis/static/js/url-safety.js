const SAME_ORIGIN_BASE = "https://marvis.invalid";
const UNSAFE_URL_CODEPOINTS = /[\u0000-\u001f\u007f\\]/;

function normalizedHref(value) {
  const href = String(value || "").trim();
  if (!href || UNSAFE_URL_CODEPOINTS.test(href)) return "";
  return href;
}

export function isSafeMarkdownHrefValue(value) {
  const href = normalizedHref(value);
  if (!href) return false;
  if (href.startsWith("#")) return true;
  try {
    const parsed = new URL(href, SAME_ORIGIN_BASE);
    if (!["http:", "https:"].includes(parsed.protocol)) return false;
    if (href.startsWith("/")) return parsed.origin === SAME_ORIGIN_BASE;
    return /^https?:\/\//i.test(href);
  } catch (_error) {
    return false;
  }
}

export function safeSameOriginApiHref(value) {
  const href = normalizedHref(value);
  if (!href || !href.startsWith("/api/tasks/")) return "";
  try {
    const parsed = new URL(href, SAME_ORIGIN_BASE);
    if (parsed.origin !== SAME_ORIGIN_BASE || !parsed.pathname.startsWith("/api/tasks/")) return "";
    // Keep downloads below a reverse-proxy mount (for example JupyterHub's
    // `/proxy/<port>/`) rather than jumping to the origin-root `/api/...`.
    return href.slice(1);
  } catch (_error) {
    return "";
  }
}
