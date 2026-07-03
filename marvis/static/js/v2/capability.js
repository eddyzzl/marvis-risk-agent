import { escapeHtml } from "../ui-utils.js";
import { listCapabilityTiers as listCapabilityTiersApi } from "./api_v2.js";
import {
  setCapabilityTiers,
  setSelectedTier,
} from "./state_v2.js";

export const selectedTierStorageKey = "marvis_v2_selected_tier";

const tierLabels = {
  deterministic_only: "仅确定性",
  guarded: "受控 Agent",
  conservative: "稳健",
  balanced: "均衡",
  explorer: "探索",
  autonomous: "自治",
};

const tierLimitLabels = {
  name: "名称",
  summary: "说明",
  max_steps: "最大步骤",
  max_replans: "最大重规划",
  allow_parallel: "允许并行",
  allow_network: "允许联网",
};

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function tierDisplayName(tier = {}) {
  return tierLabels[tier.name] || tier.name || "未命名档位";
}

function tierSummary(tier = {}) {
  const raw = String(tier.summary || "");
  return {
    "Guarded execution": "保守执行，适合高风险材料和首次试跑。",
    "Default autonomy": "默认自治，适合常规分析和建模任务。",
    "Higher autonomy": "更高自治，适合探索性任务和多步计划。",
  }[raw] || raw;
}

function tierOptionHtml(tier, defaultTier) {
  const selected = tier.name === defaultTier ? " selected" : "";
  const summary = tierSummary(tier);
  return `<option value="${escapeHtml(tier.name)}"${selected}>${escapeHtml(tierDisplayName(tier))}${summary ? ` - ${escapeHtml(summary)}` : ""}</option>`;
}

export function capabilitySelectHtmlFromData(data = {}) {
  const tiers = data.tiers || [];
  const options = tiers.map((tier) => tierOptionHtml(tier, data.default)).join("");
  return `<label>
    能力档位
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
    .map(([key, value]) => `<span class="tier-limit"><b>${escapeHtml(tierLimitLabels[key] || key)}</b>: ${escapeHtml(value)}</span>`)
    .join("");
  return entries || '<span class="tier-limit">暂无公开限制</span>';
}

export function tierSettingsHtml(data = {}) {
  const tiers = data.tiers || [];
  const selected = data.selected || data.default || "";
  const rows = tiers.map((tier) => {
    const isSelected = tier.name === selected;
    return `<label class="tier-row${isSelected ? " is-selected" : ""}">
      <input class="tier-row-radio" type="radio" name="capabilityTier" value="${escapeHtml(tier.name)}"${isSelected ? " checked" : ""} />
      <h4>${escapeHtml(tierDisplayName(tier))}</h4>
      <p>${escapeHtml(tierSummary(tier))}</p>
      <div class="tier-limits">${tierLimitsHtml(tier)}</div>
    </label>`;
  }).join("");
  return `<section class="tier-settings">
    <p class="tier-guardrail-note">能力档位只影响自治程度；证据、确认门和安全护栏保持一致。</p>
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
  const storage = deps.storage
    || (typeof localStorage !== "undefined" ? localStorage : null);
  const data = await actions.listCapabilityTiers();
  const persistedTier = String(storage?.getItem?.(selectedTierStorageKey) || "");
  const tierNames = new Set((data.tiers || []).map((tier) => String(tier.name || "")));
  const selectedTier = tierNames.has(persistedTier) ? persistedTier : data.default || "";
  setCapabilityTiers(data.tiers || []);
  setSelectedTier(selectedTier);
  container.innerHTML = tierSettingsHtml({ ...data, selected: selectedTier });
  return data;
}

export function attachCapabilityHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachCapabilityHandlers requires a stable event root");
  }
  const storage = deps.storage
    || (typeof localStorage !== "undefined" ? localStorage : null);
  const handler = async (event) => {
    const source = closest(event.target, "#tierSelect")
      || closest(event.target, 'input[name="capabilityTier"]');
    if (!source) {
      return;
    }
    const tier = String(source.value || "");
    setSelectedTier(tier);
    storage?.setItem?.(selectedTierStorageKey, tier);
  };
  root.addEventListener("change", handler);
  return () => root.removeEventListener?.("change", handler);
}
