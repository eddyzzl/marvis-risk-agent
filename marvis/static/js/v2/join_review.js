import { escapeHtml } from "../ui-utils.js";
import {
  confirmJoinSpec as confirmJoinSpecApi,
  executeJoin as executeJoinApi,
  getJoinPlan,
} from "./api_v2.js";
import {
  getCurrentJoin,
  onCurrentJoinChange,
  setCurrentJoin,
} from "./state_v2.js";

function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "-";
}

function cssEscape(value) {
  if (globalThis.CSS?.escape) {
    return globalThis.CSS.escape(String(value));
  }
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function diagnostics(spec) {
  return spec?.diagnostics || {};
}

function keyPairsHtml(spec) {
  const pairs = spec?.key_pairs || [];
  if (!pairs.length) {
    return '<span class="join-key-empty">No key pairs proposed</span>';
  }
  return pairs.map((pair) => (
    `<span class="join-key-pair">
      <span class="anchor-key">${escapeHtml(pair.anchor_col)}</span>
      <span class="join-arrow">&harr;</span>
      <span class="feature-key">${escapeHtml(pair.feature_col)}</span>
      <span class="match-method">${escapeHtml(pair.match_method || "match")}</span>
      <span class="match-rate">${pct(pair.match_rate)}</span>
    </span>`
  )).join("");
}

function warningHtml(spec) {
  const d = diagnostics(spec);
  const warnings = [];
  if (d.fan_out_detected) {
    warnings.push(`<div class="join-warning fan-out">fan-out risk: joined ${escapeHtml(d.joined_rows_preview)} rows &gt; anchor ${escapeHtml(d.anchor_rows)} rows</div>`);
  }
  if (d.shrink_detected) {
    warnings.push(`<div class="join-warning shrink">low match rate: ${pct(d.match_rate)}</div>`);
  }
  return warnings.join("");
}

function dedupHtml(spec) {
  const d = diagnostics(spec);
  if (d.feature_key_unique) {
    return '<span class="join-key-unique">Feature key unique</span>';
  }
  const featureId = escapeHtml(spec.feature_dataset_id);
  return `<select data-dedup="${featureId}" aria-label="Dedup strategy">
    <option value="">dedup required</option>
    <option value="first">first</option>
    <option value="last">last</option>
    <option value="agg_mean">agg_mean</option>
    <option value="agg_max">agg_max</option>
    <option value="abort">abort</option>
  </select>`;
}

export function joinSpecCardHtml(spec) {
  const d = diagnostics(spec);
  const warned = d.fan_out_detected || d.shrink_detected;
  const confirmed = Boolean(spec?.confirmed);
  return `<section class="join-card${warned ? " has-warn" : ""}${confirmed ? " join-confirmed" : ""}" data-feature-dataset="${escapeHtml(spec?.feature_dataset_id || "")}">
    <header class="join-card-header">
      <strong class="join-feature">${escapeHtml(spec?.feature_dataset_id || "feature dataset")}</strong>
      ${confirmed ? '<span class="join-confirmed">Confirmed</span>' : ""}
    </header>
    <div class="join-keys">${keyPairsHtml(spec)}</div>
    <div class="join-diagnostics">
      matched ${escapeHtml(d.matched_rows)} / ${escapeHtml(d.anchor_rows)} (${pct(d.match_rate)})
      | new columns ${escapeHtml(d.new_columns)} | null rate ${pct(d.new_columns_null_rate)}
    </div>
    ${warningHtml(spec)}
    ${dedupHtml(spec)}
    ${confirmed ? "" : `<button type="button" data-confirm-join="${escapeHtml(spec?.feature_dataset_id || "")}">Confirm table</button>`}
  </section>`;
}

export function joinReviewHtml(joinPlan) {
  if (!joinPlan) {
    return '<div class="v2-empty" data-v2-empty="join">No join plan selected</div>';
  }
  const joins = joinPlan.joins || [];
  const canExecute = joins.length > 0 && joins.every((spec) => spec.confirmed);
  const cards = joins.map(joinSpecCardHtml).join("");
  return `<section class="join-review" data-join-id="${escapeHtml(joinPlan.id || "")}">
    <div class="join-anchor">Anchor: ${escapeHtml(joinPlan.anchor_dataset_id || "")}</div>
    ${cards}
    <button type="button" data-exec-join="${escapeHtml(joinPlan.id || "")}"${canExecute ? "" : " disabled"}>Execute join</button>
  </section>`;
}

export function renderJoinReview(container, joinPlan = getCurrentJoin()) {
  if (!container) {
    throw new Error("renderJoinReview requires a container");
  }
  if (container.dataset) {
    container.dataset.v2JoinReview = "true";
  }
  const render = (nextJoin) => {
    container.innerHTML = joinReviewHtml(nextJoin);
  };
  render(joinPlan);
  return onCurrentJoinChange((nextJoin) => render(nextJoin));
}

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

async function defaultRefreshJoin(joinId) {
  const payload = await getJoinPlan(joinId);
  const joinPlan = payload?.join_plan || payload?.join || payload;
  if (joinPlan) {
    setCurrentJoin(joinPlan);
  }
  return joinPlan;
}

export function attachJoinHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachJoinHandlers requires a stable event root");
  }
  const actions = {
    confirmJoinSpec: confirmJoinSpecApi,
    executeJoin: executeJoinApi,
    getCurrentJoin,
    refreshJoin: defaultRefreshJoin,
    showError: defaultShowError,
    showResult: () => {},
    ...deps,
  };

  const handler = async (event) => {
    const target = event.target;
    const confirmButton = closest(target, "[data-confirm-join]");
    if (confirmButton?.dataset?.confirmJoin) {
      event.preventDefault?.();
      const join = actions.getCurrentJoin();
      if (!join?.id) {
        return;
      }
      const featureDatasetId = confirmButton.dataset.confirmJoin;
      const dedupSelect = root.querySelector?.(`[data-dedup="${cssEscape(featureDatasetId)}"]`);
      const dedupStrategy = dedupSelect ? dedupSelect.value : null;
      if (dedupSelect && !dedupStrategy) {
        actions.showError("dedup strategy is required before confirming this join");
        return;
      }
      await actions.confirmJoinSpec(join.id, {
        feature_dataset_id: featureDatasetId,
        dedup_strategy: dedupStrategy,
      });
      await actions.refreshJoin(join.id);
      return;
    }

    const executeButton = closest(target, "[data-exec-join]");
    if (executeButton?.dataset?.execJoin) {
      event.preventDefault?.();
      const result = await actions.executeJoin(executeButton.dataset.execJoin);
      if (result?.fan_out) {
        actions.showError("fan-out detected; join execution was stopped");
        return;
      }
      actions.showResult(result);
    }
  };

  root.addEventListener("click", handler);
  return () => root.removeEventListener?.("click", handler);
}
