import { defaultBranding } from "./state.js";
import { $ } from "./ui-utils.js";


export function normalizeBranding(payload = {}) {
  const branding = { ...defaultBranding };
  if (typeof payload.platformName === "string" && payload.platformName.trim()) {
    branding.platformName = payload.platformName.trim();
  }
  if (typeof payload.browserTitle === "string" && payload.browserTitle.trim()) {
    branding.browserTitle = payload.browserTitle.trim();
  }
  if (typeof payload.primaryColor === "string" && /^#[0-9a-fA-F]{6}$/.test(payload.primaryColor.trim())) {
    branding.primaryColor = payload.primaryColor.trim().toLowerCase();
  }
  if (typeof payload.logoUrl === "string" && isSafeAssetUrl(payload.logoUrl)) {
    branding.logoUrl = payload.logoUrl.trim();
  }
  if (typeof payload.faviconUrl === "string" && isSafeAssetUrl(payload.faviconUrl)) {
    branding.faviconUrl = payload.faviconUrl.trim();
  }
  branding.validatorAliases = normalizeValidatorAliases(payload.validatorAliases);
  return branding;
}

export function normalizeValidatorAliases(value) {
  const aliases = {};
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [name, alias] of Object.entries(value)) {
      if (typeof name === "string" && typeof alias === "string" && name.trim() && alias.trim()) {
        aliases[name.trim()] = alias.trim();
      }
    }
  }
  return aliases;
}

export function isSafeAssetUrl(url) {
  // Branding asset URLs are assigned to img.src / link.href. Allow explicit
  // http(s) and same-origin paths (absolute "/branding/assets/..." or relative
  // "static/brand/...", the default format) but reject protocol-relative URLs and
  // any other scheme (javascript:/data:/vbscript:/...). Unsafe values fall back to
  // the safe defaults.
  const value = String(url || "").trim();
  if (!value) return false;
  if (value.startsWith("//")) return false; // protocol-relative → external origin
  if (/^https?:\/\//i.test(value)) return true; // explicit http(s)
  if (/^[a-z][a-z0-9+.-]*:/i.test(value)) return false; // any other URL scheme
  return true; // same-origin absolute or relative asset path
}

export function brandHoverColor(color) {
  if (color === "#000000") return "#1f1f1f";
  const parts = [1, 3, 5].map((index) => parseInt(color.slice(index, index + 2), 16));
  return `#${parts.map((value) => Math.max(0, Math.round(value * 0.86)).toString(16).padStart(2, "0")).join("")}`;
}

export function imageMimeType(url) {
  const path = url.split("?")[0].toLowerCase();
  if (path.endsWith(".svg")) return "image/svg+xml";
  if (path.endsWith(".ico")) return "image/x-icon";
  if (path.endsWith(".webp")) return "image/webp";
  return "image/png";
}

export function applyBranding(branding) {
  document.title = branding.browserTitle;
  if ($("platformName")) $("platformName").textContent = branding.platformName;
  if ($("brandLogo")) {
    $("brandLogo").src = branding.logoUrl;
    $("brandLogo").alt = `${branding.platformName} logo`;
  }
  if ($("workspaceBrandLogo")) {
    $("workspaceBrandLogo").src = branding.logoUrl;
    $("workspaceBrandLogo").alt = `${branding.platformName} logo`;
  }
  const favicon = $("brandFavicon") || document.querySelector('link[rel="icon"]');
  if (favicon) {
    favicon.href = branding.faviconUrl;
    favicon.type = imageMimeType(branding.faviconUrl);
  }
  document.documentElement.style.setProperty("--brand-primary", branding.primaryColor);
  document.documentElement.style.setProperty("--brand-primary-hover", brandHoverColor(branding.primaryColor));
}
