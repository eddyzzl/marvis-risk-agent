import { escapeHtml } from "../ui-utils.js";
import {
  confirmJoinSpec as confirmJoinSpecApi,
  executeJoin as executeJoinApi,
  getJoinPlan,
  listDatasets as listDatasetsApi,
  proposeJoin as proposeJoinApi,
} from "./api_v2.js";
import {
  getDatasets,
  getCurrentJoin,
  onDatasetsChange,
  onCurrentJoinChange,
  setDatasets,
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

function joinPlanId(joinPlan) {
  return String(joinPlan?.id || joinPlan?.join_plan_id || "");
}

function anchorDatasetId(joinPlan) {
  return String(joinPlan?.anchor_dataset_id || joinPlan?.anchor_id || "");
}

function featureDatasetId(spec) {
  return String(spec?.feature_dataset_id || spec?.feature_id || "");
}

function datasetId(dataset) {
  return String(dataset?.id || dataset?.dataset_id || "");
}

function datasetLabel(dataset) {
  const source = String(dataset?.source_name || dataset?.name || datasetId(dataset) || "dataset");
  const role = String(dataset?.role || "dataset");
  const rowCount = Number(dataset?.row_count);
  const rows = Number.isFinite(rowCount) ? `${rowCount} rows` : "rows unknown";
  return `${source} - ${role} | ${rows}`;
}

function datasetOptionsHtml(datasets) {
  return datasets.map((dataset) => {
    const id = datasetId(dataset);
    return `<option value="${escapeHtml(id)}">${escapeHtml(datasetLabel(dataset))}</option>`;
  }).join("");
}

function joinProblemHtml(problems) {
  if (!problems.length) {
    return "";
  }
  return `<div class="join-problems" data-join-problem>${problems
    .map((problem) => `<div>${escapeHtml(problem)}</div>`)
    .join("")}</div>`;
}

function renderJoinProblems(root, problems = []) {
  const slot = root.querySelector?.("[data-join-problems]");
  if (!slot) {
    return false;
  }
  slot.innerHTML = joinProblemHtml(problems);
  return true;
}

function showJoinProblem(root, actions, message) {
  if (!renderJoinProblems(root, [message])) {
    actions.showError(message);
  }
}

function clearJoinProblems(root) {
  renderJoinProblems(root, []);
}

function resolveTaskId(taskId) {
  const value = typeof taskId === "function" ? taskId() : taskId;
  return String(value || "").trim();
}

function controlValue(root, selector) {
  return String(root.querySelector?.(selector)?.value || "").trim();
}

function selectedValues(root, selector) {
  const control = root.querySelector?.(selector);
  if (!control) {
    return [];
  }
  const selectedOptions = control.selectedOptions
    ? Array.from(control.selectedOptions)
    : Array.from(control.options || []).filter((option) => option.selected);
  if (selectedOptions.length) {
    return selectedOptions.map((option) => String(option.value || "").trim()).filter(Boolean);
  }
  const value = String(control.value || "").trim();
  return value ? [value] : [];
}

function uniqueValues(values) {
  return [...new Set(values.filter(Boolean))];
}

function normalizeAttachArgs(taskId, deps) {
  if (taskId && typeof taskId === "object" && !Array.isArray(taskId)) {
    return { taskId: "", deps: taskId };
  }
  return { taskId, deps };
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
  const featureId = escapeHtml(featureDatasetId(spec));
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
  const featureId = featureDatasetId(spec);
  return `<section class="join-card${warned ? " has-warn" : ""}${confirmed ? " join-confirmed" : ""}" data-feature-dataset="${escapeHtml(featureId)}">
    <header class="join-card-header">
      <strong class="join-feature">${escapeHtml(featureId || "feature dataset")}</strong>
      ${confirmed ? '<span class="join-confirmed">Confirmed</span>' : ""}
    </header>
    <div class="join-keys">${keyPairsHtml(spec)}</div>
    <div class="join-diagnostics">
      matched ${escapeHtml(d.matched_rows)} / ${escapeHtml(d.anchor_rows)} (${pct(d.match_rate)})
      | new columns ${escapeHtml(d.new_columns)} | null rate ${pct(d.new_columns_null_rate)}
    </div>
    ${warningHtml(spec)}
    ${dedupHtml(spec)}
    ${confirmed ? "" : `<button type="button" data-confirm-join="${escapeHtml(featureId)}">Confirm table</button>`}
  </section>`;
}

export function joinReviewHtml(joinPlan) {
  if (!joinPlan) {
    return '<div class="v2-empty" data-v2-empty="join">No join plan selected</div>';
  }
  const joins = joinPlan.joins || [];
  const canExecute = joins.length > 0 && joins.every((spec) => spec.confirmed);
  const cards = joins.map(joinSpecCardHtml).join("");
  const planId = joinPlanId(joinPlan);
  return `<section class="join-review" data-join-id="${escapeHtml(planId)}">
    <div class="join-anchor">Anchor: ${escapeHtml(anchorDatasetId(joinPlan))}</div>
    ${cards}
    <button type="button" data-exec-join="${escapeHtml(planId)}"${canExecute ? "" : " disabled"}>Execute join</button>
  </section>`;
}

export function joinProposalHtml(datasets = getDatasets()) {
  const items = Array.isArray(datasets) ? datasets : [];
  const options = datasetOptionsHtml(items);
  return `<section class="join-proposal" data-join-proposal>
    <div class="join-proposal-toolbar">
      <button type="button" data-refresh-datasets>Refresh datasets</button>
    </div>
    ${items.length
    ? `<label>
        Anchor dataset
        <select data-join-anchor>${options}</select>
      </label>
      <label>
        Feature datasets
        <select data-join-features multiple>${options}</select>
      </label>
      <button type="button" data-propose-join>Propose join</button>`
    : '<div class="v2-empty" data-v2-empty="datasets">No datasets loaded</div>'}
    <div data-join-problems></div>
  </section>`;
}

export function joinPanelHtml(joinPlan = getCurrentJoin(), datasets = getDatasets()) {
  return `${joinProposalHtml(datasets)}${joinReviewHtml(joinPlan)}`;
}

export function renderJoinReview(container, joinPlan = getCurrentJoin()) {
  if (!container) {
    throw new Error("renderJoinReview requires a container");
  }
  if (container.dataset) {
    container.dataset.v2JoinReview = "true";
  }
  const render = () => {
    container.innerHTML = joinPanelHtml(getCurrentJoin(), getDatasets());
  };
  if (joinPlan !== getCurrentJoin()) {
    setCurrentJoin(joinPlan);
  }
  render();
  const cleanups = [
    onCurrentJoinChange(() => render()),
    onDatasetsChange(() => render()),
  ];
  return () => cleanups.forEach((cleanup) => cleanup());
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

export function attachJoinHandlers(root, taskId = "", deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachJoinHandlers requires a stable event root");
  }
  const normalized = normalizeAttachArgs(taskId, deps);
  const actions = {
    confirmJoinSpec: confirmJoinSpecApi,
    executeJoin: executeJoinApi,
    listDatasets: listDatasetsApi,
    proposeJoin: proposeJoinApi,
    getCurrentJoin,
    refreshJoin: defaultRefreshJoin,
    setCurrentJoin,
    setDatasets,
    showError: defaultShowError,
    showResult: () => {},
    ...normalized.deps,
  };

  const handler = async (event) => {
    const target = event.target;
    const refreshButton = closest(target, "[data-refresh-datasets]");
    if (refreshButton) {
      event.preventDefault?.();
      const resolvedTaskId = resolveTaskId(normalized.taskId);
      if (!resolvedTaskId) {
        showJoinProblem(root, actions, "select or create a task before loading V2 datasets");
        return;
      }
      try {
        const payload = await actions.listDatasets(resolvedTaskId);
        const datasets = Array.isArray(payload?.datasets) ? payload.datasets : [];
        actions.setDatasets(datasets);
        clearJoinProblems(root);
      } catch (error) {
        showJoinProblem(root, actions, error?.message || "load datasets failed");
      }
      return;
    }

    const proposeButton = closest(target, "[data-propose-join]");
    if (proposeButton) {
      event.preventDefault?.();
      const resolvedTaskId = resolveTaskId(normalized.taskId);
      if (!resolvedTaskId) {
        showJoinProblem(root, actions, "select or create a task before proposing a join");
        return;
      }
      const anchorId = controlValue(root, "[data-join-anchor]");
      const featureIds = uniqueValues(selectedValues(root, "[data-join-features]"))
        .filter((featureId) => featureId !== anchorId);
      if (!anchorId || !featureIds.length) {
        showJoinProblem(root, actions, "select one anchor dataset and at least one feature dataset");
        return;
      }
      try {
        const payload = await actions.proposeJoin(resolvedTaskId, {
          anchor_dataset_id: anchorId,
          feature_dataset_ids: featureIds,
        });
        const joinPlan = payload?.join_plan || payload?.join || payload;
        if (joinPlan) {
          actions.setCurrentJoin(joinPlan);
        }
        clearJoinProblems(root);
      } catch (error) {
        showJoinProblem(root, actions, error?.message || "propose join failed");
      }
      return;
    }

    const confirmButton = closest(target, "[data-confirm-join]");
    if (confirmButton?.dataset?.confirmJoin) {
      event.preventDefault?.();
      const join = actions.getCurrentJoin();
      const joinId = joinPlanId(join);
      if (!joinId) {
        return;
      }
      const featureDatasetId = confirmButton.dataset.confirmJoin;
      const dedupSelect = root.querySelector?.(`[data-dedup="${cssEscape(featureDatasetId)}"]`);
      const dedupStrategy = dedupSelect ? dedupSelect.value : null;
      if (dedupSelect && !dedupStrategy) {
        actions.showError("dedup strategy is required before confirming this join");
        return;
      }
      await actions.confirmJoinSpec(joinId, {
        feature_id: featureDatasetId,
        feature_dataset_id: featureDatasetId,
        dedup_strategy: dedupStrategy,
      });
      await actions.refreshJoin(joinId);
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
