import { escapeHtml } from "../ui-utils.js";

export function renderModelingSetupPanel(message, options = {}) {
  const setup = message?.metadata?.modeling_setup;
  if (!setup || typeof setup !== "object") return "";
  const messageId = message?.id ? String(message.id) : "";
  const gateStepId = message?.metadata?.step_id ? String(message.metadata.step_id) : "";
  const candidates = Array.isArray(setup.sample_weight_candidates)
    ? setup.sample_weight_candidates.map((value) => String(value)).filter(Boolean)
    : [];
  const selected = String(setup.sample_weight_col || "");
  const currentTargetType = String(setup.target_type || "binary");
  const interactive = options.interactive !== false;
  const disabledAttr = interactive ? "" : " disabled aria-disabled=\"true\"";
  const uniqueCandidates = [...new Set(selected ? [selected, ...candidates] : candidates)];
  const recipeText = Array.isArray(setup.recipes) && setup.recipes.length
    ? setup.recipes.map((recipe) => String(recipe)).join("/")
    : "-";
  const primaryRecipe = String(setup.recipe || (Array.isArray(setup.recipes) ? setup.recipes[0] : "") || "-");
  const featureCount = Number.isFinite(Number(setup.feature_count)) ? String(Number(setup.feature_count)) : "-";
  const nTrials = Number.isFinite(Number(setup.n_trials)) ? String(Number(setup.n_trials)) : "-";
  const metricPolicy = String(setup.metric_policy || "-");
  const supportedPmml = new Set(Array.isArray(setup.pmml_supported_algorithms)
    ? setup.pmml_supported_algorithms.map((item) => String(item))
    : []);
  const setupWarnings = Array.isArray(setup.warnings)
    ? setup.warnings.map((item) => String(item)).filter(Boolean)
    : [];
  const splitSummary = setup.split_summary && typeof setup.split_summary === "object" ? setup.split_summary : null;
  const splitCounts = splitSummary && splitSummary.split_counts && typeof splitSummary.split_counts === "object"
    ? Object.entries(splitSummary.split_counts)
    : [];
  const splitWarnings = splitSummary && Array.isArray(splitSummary.warnings)
    ? splitSummary.warnings.map((item) => String(item)).filter(Boolean)
    : [];
  const specChips = [
    ["目标", String(setup.target_type || "binary")],
    ["算法", recipeText],
    ["主调参", primaryRecipe],
    ["候选特征", featureCount],
    ["调参轮数", nTrials],
    ["选择指标", metricPolicy],
  ].map(([label, value]) => `<div class="modeling-spec-chip"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  const eligibleAlgorithms = Array.isArray(setup.eligible_algorithms)
    ? setup.eligible_algorithms.map((item) => String(item)).filter(Boolean)
    : [];
  const selectedRecipes = Array.isArray(setup.recipes)
    ? setup.recipes.map((item) => String(item)).filter(Boolean)
    : [];
  const disabledAlgorithms = Array.isArray(setup.disabled_algorithms)
    ? setup.disabled_algorithms.filter((item) => item && typeof item === "object")
    : [];
  const algorithmChoices = uniqueRecipeChoices([
    ...eligibleAlgorithms.map((recipe) => ({
    recipe,
    state: "可用",
    reason: supportedPmml.has(recipe) ? "PMML 可导出" : "仅原生模型",
    enabled: true,
  })), ...disabledAlgorithms.map((item) => ({
    recipe: String(item.recipe || ""),
    state: "不可用",
    reason: String(item.reason || ""),
    enabled: false,
  }))]);
  const algorithmHtml = algorithmChoices.filter((item) => item.recipe).map((item) => `<div class="modeling-algorithm-chip" data-enabled="${item.enabled ? "true" : "false"}">
      <strong>${escapeHtml(item.recipe)}</strong>
      <span>${escapeHtml(item.state)} · ${escapeHtml(item.reason || "-")}</span>
    </div>`).join("");
  const splitCountsHtml = splitCounts.map(([split, count]) => {
    const total = Number(splitSummary?.total_rows || 0);
    const n = Number(count);
    const pct = total > 0 && Number.isFinite(n) ? `${((n / total) * 100).toFixed(1)}%` : "n/a";
    return `<div class="modeling-split-chip"><span>${escapeHtml(String(split).toUpperCase())}</span><strong>${escapeHtml(String(count))}</strong><small>${escapeHtml(pct)}</small></div>`;
  }).join("");
  const warningHtml = [...setupWarnings, ...splitWarnings].map((warning) => (
    `<div class="modeling-setup-warning">${escapeHtml(warning)}</div>`
  )).join("");
  const guidance = Array.isArray(setup.override_guidance)
    ? setup.override_guidance.filter((item) => item && typeof item === "object")
    : [];
  const guidanceHtml = guidance.map((item) => {
    const level = ["info", "review", "warning"].includes(String(item.level || ""))
      ? String(item.level)
      : "info";
    return `<div class="modeling-guidance-item" data-level="${escapeHtml(level)}">
      <strong>${escapeHtml(String(item.label || "业务提示"))}</strong>
      <span>${escapeHtml(String(item.message || ""))}</span>
    </div>`;
  }).join("");
  const targetOptions = ["binary", "continuous", "multiclass"].map((value) => (
    `<option value="${escapeHtml(value)}"${value === currentTargetType ? " selected" : ""}>${escapeHtml(value)}</option>`
  )).join("");
  const recipeOptions = algorithmChoices.filter((item) => item.recipe).map((item) => {
    const recipe = item.recipe;
    const checked = selectedRecipes.includes(recipe);
    const pmmlText = supportedPmml.has(recipe) ? "PMML" : "原生";
    return `<label class="modeling-recipe-option">
      <input type="checkbox" class="modeling-recipe-pick" value="${escapeHtml(recipe)}"${checked ? " checked" : ""}${disabledAttr} />
      <span>${escapeHtml(recipe)}</span>
      <small>${escapeHtml(recipeFamily(recipe))} · ${escapeHtml(pmmlText)}</small>
    </label>`;
  }).join("");
  const setupControlsHtml = `<div class="modeling-setup-controls">
    <label>目标类型
      <select class="modeling-target-select"${disabledAttr} data-current-target-type="${escapeHtml(currentTargetType)}">${targetOptions}</select>
    </label>
    <label>调参轮数
      <input type="number" class="modeling-n-trials-input" min="1" max="200" step="1" value="${escapeHtml(nTrials === "-" ? "" : nTrials)}"${disabledAttr} data-current-n-trials="${escapeHtml(nTrials === "-" ? "" : nTrials)}" />
    </label>
    ${recipeOptions ? `<div class="modeling-recipe-control" data-current-recipes="${escapeHtml(selectedRecipes.join(","))}">
      <span>训练算法</span>
      <div class="modeling-recipe-options">${recipeOptions}</div>
    </div>` : ""}
    <label class="modeling-override-reason">变更原因
      <textarea class="modeling-override-reason-input" rows="2" placeholder="调整目标类型、算法或调参轮数时必填"${disabledAttr}></textarea>
    </label>
  </div>`;
  const optionRows = [
    { value: "", label: "不使用权重" },
    ...uniqueCandidates.map((value) => ({ value, label: value })),
  ].map((option) => {
    const checked = option.value === selected || (!selected && option.value === "");
    return `<label class="modeling-weight-option">
      <input type="radio" name="modelingWeight-${escapeHtml(messageId)}" class="modeling-weight-pick" value="${escapeHtml(option.value)}"${checked ? " checked" : ""}${disabledAttr} />
      <span>${escapeHtml(option.label)}</span>
    </label>`;
  }).join("");
  const diagnostics = Array.isArray(setup.sample_weight_diagnostics)
    ? setup.sample_weight_diagnostics.filter((item) => item && typeof item === "object")
    : [];
  const diagnosticsByColumn = new Map(diagnostics.map((item) => [String(item.column || ""), item]));
  const diagnosticsHtml = uniqueCandidates
    .map((column) => {
      const item = diagnosticsByColumn.get(column);
      if (!item) return "";
      const missing = Number.isFinite(Number(item.missing_rate))
        ? `${(Number(item.missing_rate) * 100).toFixed(1)}%`
        : "n/a";
      const min = item.min ?? "n/a";
      const max = item.max ?? "n/a";
      const mean = item.mean ?? "n/a";
      const state = item.valid ? "可用" : "需检查";
      const reason = item.reason || "已排除出入模特征";
      return `<div class="modeling-weight-diagnostic" data-valid="${item.valid ? "true" : "false"}">
        <strong>${escapeHtml(column)}</strong>
        <span>${escapeHtml(state)} · 缺失 ${escapeHtml(missing)} · 范围 ${escapeHtml(min)}-${escapeHtml(max)} · 均值 ${escapeHtml(mean)}</span>
        <small>${escapeHtml(reason)}</small>
      </div>`;
    })
    .filter(Boolean)
    .join("");
  return `<div class="modeling-setup-panel" data-modeling-weight-form="${escapeHtml(messageId)}" data-modeling-gate-step-id="${escapeHtml(gateStepId)}" data-modeling-current-weight="${escapeHtml(selected)}"${interactive ? "" : ' data-modeling-readonly="true"'}>
    <div class="modeling-setup-head">
      <span>建模规格</span>
      <small>${escapeHtml(String(setup.target_type || "binary"))} · ${escapeHtml(recipeText)}</small>
    </div>
    <div class="modeling-spec-grid">${specChips}</div>
    ${guidanceHtml ? `<div class="modeling-guidance-list">${guidanceHtml}</div>` : ""}
    ${setupControlsHtml}
    ${algorithmHtml ? `<div class="modeling-algorithm-grid">${algorithmHtml}</div>` : ""}
    ${splitCountsHtml ? `<div class="modeling-split-summary">
      <div class="modeling-section-label">样本切分 · ${escapeHtml(String(splitSummary?.split_col || "split"))}</div>
      <div class="modeling-split-grid">${splitCountsHtml}</div>
    </div>` : ""}
    ${warningHtml ? `<div class="modeling-setup-warnings">${warningHtml}</div>` : ""}
    <div class="modeling-weight-options" role="radiogroup" aria-label="样本权重列">${optionRows}</div>
    ${diagnosticsHtml ? `<div class="modeling-weight-diagnostics">${diagnosticsHtml}</div>` : ""}
    <div class="modeling-setup-foot">
      <span>权重列不进入特征;目标/算法/调参调整会重算后续步骤。</span>
      <button type="button" class="button compact secondary modeling-weight-adjust"${interactive ? ` data-modeling-weight-adjust="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "应用建模设置" : "历史规格"}</button>
    </div>
  </div>`;
}

export async function submitModelingWeightAdjust(button, context = {}) {
  const form = button.closest(".modeling-setup-panel");
  const taskId = typeof context.getSelectedTaskId === "function"
    ? context.getSelectedTaskId()
    : context.selectedTaskId;
  const api = context.api;
  const setActionStatus = context.setActionStatus || (() => {});
  if (!form || !taskId || typeof api !== "function") return;
  if (form.dataset.modelingReadonly === "true") {
    setActionStatus("这是历史建模规格,请使用最新待确认步骤调整。", "error");
    return;
  }
  const adjustParams = collectModelingSetupAdjustParams(form);
  if (!Object.keys(adjustParams).length) {
    setActionStatus("建模设置未变化。", "info");
    return;
  }
  const reason = String(form.querySelector(".modeling-override-reason-input")?.value || "").trim();
  const structuralKeys = ["target_type", "recipes", "n_trials"];
  if (Array.isArray(adjustParams.recipes) && !adjustParams.recipes.length) {
    setActionStatus("请至少选择一个训练算法。", "error");
    return;
  }
  const targetType = selectedModelingTargetType(form);
  const selectedRecipes = selectedModelingRecipes(form);
  const mismatchedRecipe = selectedRecipes.find((recipe) => recipeFamily(recipe) !== targetType);
  if (mismatchedRecipe) {
    setActionStatus(`目标类型 ${targetType} 与算法 ${mismatchedRecipe} 不匹配,请选择同一目标类型的算法。`, "error");
    return;
  }
  if (structuralKeys.some((key) => Object.prototype.hasOwnProperty.call(adjustParams, key)) && reason.length < 4) {
    setActionStatus("调整目标类型、算法或调参轮数时请填写变更原因。", "error");
    return;
  }
  const expectedStepId = form.dataset.modelingGateStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息,请刷新后重试。", "error");
    return;
  }
  button.disabled = true;
  // UX-1: this adjust reruns the driver turn (now job-wrapped, REL-1) and can
  // rerun screen/tune/train downstream, so give immediate busy feedback, poll
  // agent messages so intermediate step content streams in, and force the plan
  // rail to re-fetch on a short interval so the running step doesn't look frozen.
  const pollAgentMessagesUntilSettled = context.pollAgentMessagesUntilSettled || (() => Promise.resolve());
  const resetFetchThrottle = context.resetFetchThrottle || (() => {});
  const renderWorkflowStepper = context.renderWorkflowStepper || (() => {});
  setActionStatus("正在执行下一步…", "busy");
  let planRailTimer = null;
  if (typeof setInterval === "function") {
    planRailTimer = setInterval(() => {
      resetFetchThrottle(taskId);
      renderWorkflowStepper({ force: true });
    }, 1500);
  }
  try {
    const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify({
        content: reason ? `调整建模规格：${reason}` : "调整建模规格",
        adjust_params: adjustParams,
        expected_step_id: expectedStepId,
        acceptance_mode: typeof context.agentAcceptanceModeValue === "function"
          ? context.agentAcceptanceModeValue()
          : "manual",
      }),
    });
    const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
    const result = await requestPromise;
    await streamPollPromise;
    if (typeof context.setAgentMessages === "function") {
      context.setAgentMessages(result.messages);
    }
    if (typeof context.renderAgentConversation === "function") {
      context.renderAgentConversation();
    }
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "调整样本权重失败", "error");
  } finally {
    if (planRailTimer !== null) clearInterval(planRailTimer);
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  }
}

function collectModelingSetupAdjustParams(form) {
  const params = {};
  const target = form.querySelector(".modeling-target-select");
  if (target) {
    const value = String(target.value || "").trim();
    const current = String(target.getAttribute("data-current-target-type") || "").trim();
    if (value && value !== current) params.target_type = value;
  }
  const nTrials = form.querySelector(".modeling-n-trials-input");
  if (nTrials) {
    const value = Number(nTrials.value);
    const current = Number(nTrials.getAttribute("data-current-n-trials") || NaN);
    if (Number.isFinite(value) && value !== current) params.n_trials = value;
  }
  const recipeControl = form.querySelector(".modeling-recipe-control");
  if (recipeControl) {
    const selected = [...recipeControl.querySelectorAll(".modeling-recipe-pick:checked")]
      .map((input) => String(input.value || "").trim())
      .filter(Boolean);
    const current = String(recipeControl.dataset.currentRecipes || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    if (selected.join(",") !== current.join(",")) params.recipes = selected;
  }
  const picked = form.querySelector(".modeling-weight-pick:checked");
  const sampleWeightCol = picked ? String(picked.value || "").trim() : "";
  const currentWeight = String(form.dataset.modelingCurrentWeight || "").trim();
  if (sampleWeightCol !== currentWeight) params.sample_weight_col = sampleWeightCol;
  return params;
}

function selectedModelingTargetType(form) {
  const target = form.querySelector(".modeling-target-select");
  return String(target?.value || "binary").trim() || "binary";
}

function selectedModelingRecipes(form) {
  const recipeControl = form.querySelector(".modeling-recipe-control");
  if (!recipeControl) return [];
  return [...recipeControl.querySelectorAll(".modeling-recipe-pick:checked")]
    .map((input) => String(input.value || "").trim())
    .filter(Boolean);
}

function uniqueRecipeChoices(items) {
  const seen = new Set();
  const choices = [];
  for (const item of items) {
    const recipe = String(item.recipe || "");
    if (!recipe || seen.has(recipe)) continue;
    seen.add(recipe);
    choices.push(item);
  }
  return choices;
}

function recipeFamily(recipe) {
  if (recipe === "lgb_regressor") return "continuous";
  if (recipe === "lgb_multiclass") return "multiclass";
  return "binary";
}

export function handleModelingWeightAdjustClick(event, context = {}) {
  const button = event.target?.closest?.("[data-modeling-weight-adjust]");
  if (!button) return;
  event.preventDefault();
  return submitModelingWeightAdjust(button, context);
}
