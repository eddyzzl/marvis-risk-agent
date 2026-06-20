import { escapeHtml } from "../ui-utils.js";
import { listCapabilityTiers as listCapabilityTiersApi } from "./api_v2.js";
import {
  setCapabilityTiers,
  setSelectedTier,
} from "./state_v2.js";

export const selectedTierStorageKey = "marvis_v2_selected_tier";

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function tierOptionHtml(tier, defaultTier) {
  const selected = tier.name === defaultTier ? " selected" : "";
  return `<option value="${escapeHtml(tier.name)}"${selected}>${escapeHtml(tier.name)} - ${escapeHtml(tier.summary || "")}</option>`;
}

export function capabilitySelectHtmlFromData(data = {}) {
  const tiers = data.tiers || [];
  const options = tiers.map((tier) => tierOptionHtml(tier, data.default)).join("");
  return `<label>
    Capability tier
    <select id="tierSelect">${options}</select>
  </label>`;
}

export async function capabilitySelectHtml(deps = {}) {
  const actions = {
    listCapabilityTiers: listCapabilityTiersApi,
    ...deps,
  };
  const data = await actions.listCapabilityTiers();
  return capabilitySelectHtmlFromData(data);
}

function tierLimitsHtml(tier) {
  const entries = Object.entries(tier)
    .filter(([key]) => !["name", "summary"].includes(key))
    .map(([key, value]) => `<span class="tier-limit"><b>${escapeHtml(key)}</b>: ${escapeHtml(value)}</span>`)
    .join("");
  return entries || '<span class="tier-limit">no limits published</span>';
}

export function tierSettingsHtml(data = {}) {
  const tiers = data.tiers || [];
  const rows = tiers.map((tier) => (
    `<section class="tier-row${tier.name === data.default ? " default-tier" : ""}">
      <h4>${escapeHtml(tier.name)}</h4>
      <p>${escapeHtml(tier.summary || "")}</p>
      <div class="tier-limits">${tierLimitsHtml(tier)}</div>
    </section>`
  )).join("");
  return `<section class="tier-settings">
    <p class="tier-guardrail-note">Guardrails remain constant across capability tiers.</p>
    ${rows}
  </section>`;
}

export function renderTierSettingsShell(container, data = {}) {
  if (!container) {
    throw new Error("renderTierSettingsShell requires a container");
  }
  if (container.dataset) {
    container.dataset.v2TierSettings = "true";
  }
  container.innerHTML = tierSettingsHtml(data);
  return () => {};
}

export async function renderTierSettings(container, deps = {}) {
  if (!container) {
    throw new Error("renderTierSettings requires a container");
  }
  if (container.dataset) {
    container.dataset.v2TierSettings = "true";
  }
  const actions = {
    listCapabilityTiers: listCapabilityTiersApi,
    ...deps,
  };
  const data = await actions.listCapabilityTiers();
  setCapabilityTiers(data.tiers || []);
  setSelectedTier(data.default || "");
  container.innerHTML = tierSettingsHtml(data);
  return data;
}

export function attachCapabilityHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachCapabilityHandlers requires a stable event root");
  }
  const storage = deps.storage
    || (typeof localStorage !== "undefined" ? localStorage : null);
  const handler = async (event) => {
    const select = closest(event.target, "#tierSelect");
    if (!select) {
      return;
    }
    const tier = String(select.value || "");
    setSelectedTier(tier);
    storage?.setItem?.(selectedTierStorageKey, tier);
  };
  root.addEventListener("change", handler);
  return () => root.removeEventListener?.("change", handler);
}
