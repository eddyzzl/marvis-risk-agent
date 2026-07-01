import { sleep } from "../api.js";
import { escapeHtml } from "../ui-utils.js";
import {
  confirmJoinSpec as confirmJoinSpecApi,
  executeJoin as executeJoinApi,
  getLatestTaskJob as getLatestTaskJobApi,
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
  const source = String(dataset?.source_name || dataset?.name || datasetId(dataset) || "数据集");
  const role = String(dataset?.role || "数据集");
  const rowCount = Number(dataset?.row_count);
  const rows = Number.isFinite(rowCount) ? `${rowCount} 行` : "行数未知";
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
    return '<span class="join-key-empty">暂无建议关联键</span>';
  }
  return pairs.map((pair) => (
    `<span class="join-key-pair">
      <span class="anchor-key">${escapeHtml(pair.anchor_col)}</span>
      <span class="join-arrow">&harr;</span>
      <span class="feature-key">${escapeHtml(pair.feature_col)}</span>
      <span class="match-method">${escapeHtml(pair.match_method || "匹配")}</span>
      <span class="match-rate">${pct(pair.match_rate)}</span>
    </span>`
  )).join("");
}

function warningHtml(spec) {
  const d = diagnostics(spec);
  const warnings = [];
  if (d.fan_out_detected) {
    warnings.push(`<div class="join-warning fan-out">fan-out 风险：拼接后 ${escapeHtml(d.joined_rows_preview)} 行 &gt; 主表 ${escapeHtml(d.anchor_rows)} 行</div>`);
  }
  if (d.shrink_detected) {
    warnings.push(`<div class="join-warning shrink">匹配率偏低：${pct(d.match_rate)}</div>`);
  }
  if (spec?.dedup_strategy_warning) {
    warnings.push(`<div class="join-warning synthetic-dedup">${escapeHtml(spec.dedup_strategy_warning)}</div>`);
  }
  return warnings.join("");
}

function dedupHtml(spec) {
  const d = diagnostics(spec);
  if (d.feature_key_unique) {
    return '<span class="join-key-unique">特征表键唯一</span>';
  }
  const featureId = escapeHtml(featureDatasetId(spec));
  return `<select data-dedup="${featureId}" aria-label="去重策略">
    <option value="">需要选择去重策略</option>
    <option value="first">保留首条</option>
    <option value="last">保留末条</option>
    <option value="agg_mean">数值取均值（合成聚合行）</option>
    <option value="agg_max">数值取最大（合成聚合行）</option>
    <option value="abort">终止拼接</option>
  </select>
  <div class="join-warning synthetic-dedup">聚合去重会基于同键冲突生成合成特征行，结果可能不对应任何一条原始特征记录。</div>`;
}

export function joinSpecCardHtml(spec) {
  const d = diagnostics(spec);
  const warned = d.fan_out_detected || d.shrink_detected;
  const confirmed = Boolean(spec?.confirmed);
  const featureId = featureDatasetId(spec);
  return `<section class="join-card${warned ? " has-warn" : ""}${confirmed ? " join-confirmed" : ""}" data-feature-dataset="${escapeHtml(featureId)}">
    <header class="join-card-header">
      <strong class="join-feature">${escapeHtml(featureId || "特征数据集")}</strong>
      ${confirmed ? '<span class="join-confirmed">已确认</span>' : ""}
    </header>
    <div class="join-keys">${keyPairsHtml(spec)}</div>
    <div class="join-diagnostics">
      匹配 ${escapeHtml(d.matched_rows)} / ${escapeHtml(d.anchor_rows)} (${pct(d.match_rate)})
      | 新增字段 ${escapeHtml(d.new_columns)} | 空值率 ${pct(d.new_columns_null_rate)}
    </div>
    ${warningHtml(spec)}
    ${dedupHtml(spec)}
    ${confirmed ? "" : `<button type="button" data-confirm-join="${escapeHtml(featureId)}">确认该表</button>`}
  </section>`;
}

export function joinReviewHtml(joinPlan) {
  if (!joinPlan) {
    return '<div class="v2-empty" data-v2-empty="join">暂无选中的拼接计划</div>';
  }
  const joins = joinPlan.joins || [];
  const canExecute = joins.length > 0 && joins.every((spec) => spec.confirmed);
  const cards = joins.map(joinSpecCardHtml).join("");
  const planId = joinPlanId(joinPlan);
  return `<section class="join-review" data-join-id="${escapeHtml(planId)}">
    <div class="join-anchor">主表：${escapeHtml(anchorDatasetId(joinPlan))}</div>
    ${cards}
    <button type="button" data-exec-join="${escapeHtml(planId)}"${canExecute ? "" : " disabled"}>执行拼接</button>
  </section>`;
}

export function joinProposalHtml(datasets = getDatasets()) {
  const items = Array.isArray(datasets) ? datasets : [];
  const options = datasetOptionsHtml(items);
  return `<section class="join-proposal" data-join-proposal>
    <div class="join-proposal-toolbar">
      <button type="button" data-refresh-datasets>刷新数据集</button>
    </div>
    ${items.length
    ? `<label>
        主数据集
        <select data-join-anchor>${options}</select>
      </label>
      <label>
        特征数据集
        <select data-join-features multiple>${options}</select>
      </label>
      <button type="button" data-propose-join>生成拼接建议</button>`
    : '<div class="v2-empty" data-v2-empty="datasets">暂无已加载数据集</div>'}
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

function isFanOutError(error) {
  const text = String(error?.message || error?.detail || "").toLowerCase();
  return error?.status === 409 || text.includes("fan-out") || text.includes("fanout");
}

function joinExecutionAccepted(result) {
  return Boolean(result?.job_id || result?.status === "accepted");
}

function joinExecutionComplete(result) {
  const joinPlan = result?.join_plan || result?.join || result;
  return Boolean(result?.result_dataset_id || joinPlan?.result_dataset_id || joinPlan?.status === "executed");
}

async function defaultRefreshJoin(joinId) {
  const payload = await getJoinPlan(joinId);
  const joinPlan = payload?.join_plan || payload?.join || payload;
  if (joinPlan) {
    setCurrentJoin(joinPlan);
  }
  return joinPlan;
}

async function defaultPollJoinExecution({
  joinId,
  taskId = "",
  refreshJoin = defaultRefreshJoin,
  getLatestTaskJob = getLatestTaskJobApi,
  intervalMs = 1000,
  maxAttempts = 120,
  sleepFn = sleep,
} = {}) {
  if (!joinId) {
    return null;
  }
  const attempts = Math.max(1, Number(maxAttempts) || 1);
  const resolvedTaskId = String(taskId || "").trim();
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const joinPlan = await refreshJoin(joinId);
    if (joinExecutionComplete(joinPlan)) {
      return { status: "executed", join_plan: joinPlan, result_dataset_id: joinPlan?.result_dataset_id };
    }
    if (!resolvedTaskId) {
      return joinPlan;
    }
    const payload = await getLatestTaskJob(resolvedTaskId, "join");
    const job = payload?.job || null;
    if (job?.status === "failed") {
      throw new Error(job.error_value || job.error_name || "后台拼接失败。");
    }
    if (!job || !["queued", "running"].includes(String(job.status || ""))) {
      break;
    }
    if (attempt < attempts - 1) {
      await sleepFn(intervalMs);
    }
  }
  throw new Error("后台拼接任务已结束，但拼接计划未生成结果，请刷新后重试。");
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
    pollJoinExecution: defaultPollJoinExecution,
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
        showJoinProblem(root, actions, "请先选择或创建任务，再加载 V2 数据集。");
        return;
      }
      try {
        const payload = await actions.listDatasets(resolvedTaskId);
        const datasets = Array.isArray(payload?.datasets) ? payload.datasets : [];
        actions.setDatasets(datasets);
        clearJoinProblems(root);
      } catch (error) {
        showJoinProblem(root, actions, error?.message || "数据集加载失败");
      }
      return;
    }

    const proposeButton = closest(target, "[data-propose-join]");
    if (proposeButton) {
      event.preventDefault?.();
      const resolvedTaskId = resolveTaskId(normalized.taskId);
      if (!resolvedTaskId) {
        showJoinProblem(root, actions, "请先选择或创建任务，再生成拼接建议。");
        return;
      }
      const anchorId = controlValue(root, "[data-join-anchor]");
      const featureIds = uniqueValues(selectedValues(root, "[data-join-features]"))
        .filter((featureId) => featureId !== anchorId);
      if (!anchorId || !featureIds.length) {
        showJoinProblem(root, actions, "请选择一个主数据集和至少一个特征数据集。");
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
        showJoinProblem(root, actions, error?.message || "拼接建议生成失败");
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
        actions.showError("确认该拼接前必须选择去重策略。");
        return;
      }
      try {
        await actions.confirmJoinSpec(joinId, {
          feature_id: featureDatasetId,
          feature_dataset_id: featureDatasetId,
          dedup_strategy: dedupStrategy,
        });
        await actions.refreshJoin(joinId);
      } catch (error) {
        actions.showError(error?.message || "拼接确认失败");
      }
      return;
    }

    const executeButton = closest(target, "[data-exec-join]");
    if (executeButton?.dataset?.execJoin) {
      event.preventDefault?.();
      try {
        const joinId = executeButton.dataset.execJoin;
        const result = await actions.executeJoin(joinId);
        if (joinExecutionAccepted(result)) {
          actions.showResult(result);
          const finalResult = await actions.pollJoinExecution({
            accepted: result,
            joinId,
            taskId: resolveTaskId(normalized.taskId),
            refreshJoin: actions.refreshJoin,
          });
          if (finalResult) {
            actions.showResult(finalResult);
          }
          return;
        }
        if (result?.fan_out) {
          actions.showError("检测到 fan-out 风险，已停止执行拼接。");
          return;
        }
        actions.showResult(result);
      } catch (error) {
        if (isFanOutError(error)) {
          actions.showError("检测到 fan-out 风险，已停止执行拼接。");
          return;
        }
        actions.showError(error?.message || "拼接执行失败");
      }
    }
  };

  root.addEventListener("click", handler);
  return () => root.removeEventListener?.("click", handler);
}
