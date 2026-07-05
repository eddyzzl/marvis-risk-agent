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
  default_autonomy_level: "默认自治级别",
  max_replan_iterations: "最大重规划",
  max_plan_depth: "计划步数上限",
  allow_explore_mode: "探索模式",
  max_auto_gates: "单轮自动审查上限",
};

// Internal orchestrator knobs that mean nothing to a user reading a settings
// page; name/summary already render as the card title/description.
const hiddenTierLimitKeys = new Set([
  "name",
  "summary",
  "failure_driven_replan",
  "decision_point_replan",
  "explore_segment_size",
]);

// 档位只调节自治预算（重规划次数、计划步数、探索模式），确认门和安全护栏在所有档位下都一致
// （见 marvis/domain.py 的 capability_tier 说明）。摘要只描述预算差异，具体数值在下方标签里。
const tierNameSummaries = {
  conservative: "自治预算最小：重规划次数少、计划步数浅、关闭探索模式，适合高风险材料或首次试跑。",
  balanced: "默认档位：自治预算适中、允许探索模式，适合常规分析和建模任务。",
  autonomous: "自治预算最大：重规划次数多、计划步数深，适合熟悉的数据和探索性任务。",
};

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function tierDisplayName(tier = {}) {
  return tierLabels[tier.name] || tier.name || "未命名档位";
}

function tierSummary(tier = {}) {
  const raw = String(tier.summary || "");
  const translated = {
    "Guarded execution": "保守执行，适合高风险材料和首次试跑。",
    "Default autonomy": "默认自治，适合常规分析和建模任务。",
    "Higher autonomy": "更高自治，适合探索性任务和多步计划。",
  }[raw] || raw;
  return translated || tierNameSummaries[tier.name] || "";
}

function tierLimitValue(key, value) {
  if (typeof value === "boolean") return value ? "允许" : "关闭";
  if (key === "default_autonomy_level") return `L${value}`;
  if (key === "max_replan_iterations" || key === "max_replans") return `${value} 次`;
  if (key === "max_plan_depth" || key === "max_steps") return `${value} 步`;
  if (key === "max_auto_gates") return `${value} 个确认门`;
  return String(value);
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
    .filter(([key]) => !hiddenTierLimitKeys.has(key))
    .map(([key, value]) => `<span class="tier-limit"><b>${escapeHtml(tierLimitLabels[key] || key)}</b>${escapeHtml(tierLimitLabels[key] ? "：" : ": ")}${escapeHtml(tierLimitValue(key, value))}</span>`)
    .join("");
  return entries || '<span class="tier-limit">暂无公开限制</span>';
}

export function tierSettingsHtml(data = {}) {
  const tiers = data.tiers || [];
  const selected = data.selected || data.default || "";
  const rows = tiers.map((tier) => {
    const isSelected = tier.name === selected;
    const summary = tierSummary(tier);
    return `<label class="tier-row${isSelected ? " is-selected" : ""}">
      <input class="tier-row-radio" type="radio" name="capabilityTier" value="${escapeHtml(tier.name)}"${isSelected ? " checked" : ""} />
      <span class="tier-check" aria-hidden="true">
        <svg viewBox="0 0 24 24" focusable="false"><path d="M5 13l4 4L19 7"></path></svg>
      </span>
      <span class="tier-row-body">
        <h4>${escapeHtml(tierDisplayName(tier))}</h4>
        ${summary ? `<p>${escapeHtml(summary)}</p>` : ""}
        <div class="tier-limits">${tierLimitsHtml(tier)}</div>
      </span>
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
