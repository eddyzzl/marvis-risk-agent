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
  const candidateFeatureCount = Number.isFinite(Number(setup.candidate_feature_count))
    ? String(Number(setup.candidate_feature_count))
    : "";
  const rawNTrials = setup.n_trials;
  const nTrials = rawNTrials !== null
    && rawNTrials !== undefined
    && String(rawNTrials).trim() !== ""
    && Number.isFinite(Number(rawNTrials))
    ? String(Number(rawNTrials))
    : "-";
  const metricPolicy = String(setup.metric_policy || "-");
  const metricPolicyLabel = humanMetricPolicy(metricPolicy);
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
  const splitConfig = splitSummary?.split_config && typeof splitSummary.split_config === "object"
    ? splitSummary.split_config
    : {};
  const splitColumns = Array.isArray(splitSummary?.available_columns)
    ? splitSummary.available_columns.map(String).filter(Boolean)
    : [];
  const splitMode = splitConfig.oot_by_time ? "time" : splitConfig.random_oot ? "random" : "none";
  const testPercent = Number.isFinite(Number(splitConfig.test_size)) ? Number(splitConfig.test_size) * 100 : 25;
  const ootPercent = Number.isFinite(Number(splitConfig.oot_size)) ? Number(splitConfig.oot_size) * 100 : 20;
  const specChips = [
    ["目标", String(setup.target_type || "binary")],
    ["算法", recipeText],
    ["主调参", primaryRecipe],
    [
      candidateFeatureCount ? "精选后特征" : "候选特征",
      featureCount,
      candidateFeatureCount ? `原始候选 ${candidateFeatureCount}，精选后 ${featureCount}` : featureCount,
    ],
    ["调参轮数", nTrials],
    ["选择指标", metricPolicyLabel, metricPolicy],
  ].map(([label, value, detail = value]) => `<div class="modeling-spec-chip" title="${escapeHtml(detail)}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
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
  const splitModeOptions = [
    ["none", "不设置 OOT", "只切分 train / test"],
    ["time", "按时间字段切分 OOT（推荐）", "用较新的时间段做外推验证"],
    ["random", "随机留出 OOT", "没有可靠时间字段时才使用"],
  ].map(([value, label, note]) => `<label class="modeling-split-mode-option">
    <input type="radio" name="modelingSplitMode-${escapeHtml(messageId)}" class="modeling-split-mode" value="${value}"${splitMode === value ? " checked" : ""}${disabledAttr}>
    <span><strong>${escapeHtml(label)}</strong><small>${escapeHtml(note)}</small></span>
  </label>`).join("");
  const timeColumnOptions = ['<option value="">请选择时间字段</option>'].concat(splitColumns.map((column) => (
    `<option value="${escapeHtml(column)}"${String(splitConfig.oot_by_time || "") === column ? " selected" : ""}>${escapeHtml(column)}</option>`
  ))).join("");
  const splitControlsHtml = splitSummary ? `<section class="modeling-split-controls">
    <div class="modeling-section-heading"><strong>数据集切分设置</strong><span>系统当前展示的是切分预览，确认后才会进入特征筛选和训练。</span></div>
    <div class="modeling-split-mode-list" role="radiogroup" aria-label="OOT 切分方式">${splitModeOptions}</div>
    <div class="modeling-split-fields">
      <label>测试集占剩余样本比例（%）<input type="number" class="modeling-test-size-input" min="1" max="50" step="1" value="${escapeHtml(String(testPercent))}"${disabledAttr}></label>
      <label>OOT 占比（%）<input type="number" class="modeling-oot-size-input" min="1" max="50" step="1" value="${escapeHtml(String(ootPercent))}"${disabledAttr}></label>
      <label>OOT 时间字段<select class="modeling-time-column-select"${disabledAttr}>${timeColumnOptions}</select></label>
    </div>
    <p class="modeling-split-help">训练集 / 测试集 / OOT 会按同一分组键整体分配，避免同一客户跨集合造成泄漏。选择“不设置 OOT”时，仅保留 train / test。</p>
  </section>` : "";
  const targetAlgorithmControlsHtml = `<div class="modeling-setup-controls modeling-target-controls">
    <label>目标类型
      <select class="modeling-target-select"${disabledAttr} data-current-target-type="${escapeHtml(currentTargetType)}">${targetOptions}</select>
    </label>
    ${recipeOptions ? `<div class="modeling-recipe-control" data-current-recipes="${escapeHtml(selectedRecipes.join(","))}">
      <span>训练算法</span>
      <div class="modeling-recipe-options">${recipeOptions}</div>
    </div>` : ""}
  </div>`;
  const reasonControlHtml = `<label class="modeling-override-reason">调整说明（变更原因）
    <textarea class="modeling-override-reason-input" rows="2" placeholder="如果修改了 Agent 建议，请说明业务原因"${disabledAttr}></textarea>
  </label>`;
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
  const tuningControlsHtml = `<div class="modeling-tuning-layout">
    <label class="modeling-trial-control">
      <span>调参轮数</span>
      <strong data-modeling-live-value="trials">${escapeHtml(nTrials)}</strong>
      <input type="number" class="modeling-n-trials-input" min="1" max="200" step="1" value="${escapeHtml(nTrials === "-" ? "" : nTrials)}"${disabledAttr} data-current-n-trials="${escapeHtml(nTrials === "-" ? "" : nTrials)}" />
      <small>轮数越高，搜索更充分，但执行时间也会增加。</small>
    </label>
    <div class="modeling-weight-control">
      <span>样本权重</span>
      <div class="modeling-weight-options" role="radiogroup" aria-label="样本权重列">${optionRows}</div>
    </div>
  </div>`;
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
  const journey = [
    ["split", "样本方案", "确认 Train / Test / OOT"],
    ["model", "模型候选", "目标类型与算法"],
    ["tuning", "训练策略", "调参与样本权重"],
    ["review", "确认执行", "复核 Agent 建议"],
  ];
  const journeyNav = journey.map(([id, label, note], index) => `<button type="button" class="modeling-journey-node${index === 0 ? " is-active" : ""}" data-modeling-step-jump="${id}" data-step-index="${index + 1}" aria-current="${index === 0 ? "step" : "false"}"${disabledAttr}>
    <span>${index + 1}</span><strong>${label}</strong><small>${note}</small>
  </button>`).join("");
  return `<div class="modeling-setup-panel" data-modeling-weight-form="${escapeHtml(messageId)}" data-modeling-gate-step-id="${escapeHtml(gateStepId)}" data-modeling-current-weight="${escapeHtml(selected)}" data-current-split-config="${escapeHtml(JSON.stringify(splitConfig))}" data-modeling-active-step="split"${interactive ? "" : ' data-modeling-readonly="true"'}>
    <div class="modeling-setup-head">
      <span class="modeling-setup-title"><small>Agent 建模协作台</small><strong>一起确定这次模型怎么做</strong></span>
      <span class="modeling-setup-status"><i></i>${interactive ? "等待你逐步确认" : "历史规格"}</span>
    </div>
    <div class="modeling-agent-note"><span aria-hidden="true">✦</span><p><strong>Agent 已准备一版方案</strong>按四个小步骤复核即可；每次选择都会保留，最后统一执行。</p></div>
    <nav class="modeling-journey" aria-label="建模规格步骤">${journeyNav}</nav>
    <div class="modeling-step-progress" aria-hidden="true"><i></i></div>
    <div class="modeling-step-error" role="status" aria-live="polite"></div>
    <section class="modeling-journey-stage is-active" data-modeling-stage="split">
      <div class="modeling-stage-heading"><span>01</span><div><strong>先确定样本怎么切</strong><small>Agent 不会擅自决定 OOT；这里确认时间外推与测试集比例。</small></div></div>
      ${splitControlsHtml || '<div class="modeling-stage-empty">当前数据没有可配置的切分预览，将沿用现有样本口径。</div>'}
      ${splitCountsHtml ? `<div class="modeling-split-summary"><div class="modeling-section-label">样本切分 · ${escapeHtml(String(splitSummary?.split_col || "split"))} · 预览</div><div class="modeling-split-grid">${splitCountsHtml}</div></div>` : ""}
      <div class="modeling-stage-actions"><button type="button" class="button compact secondary" data-modeling-step-next="model"${disabledAttr}>下一步：模型候选</button></div>
    </section>
    <section class="modeling-journey-stage" data-modeling-stage="model">
      <div class="modeling-stage-heading"><span>02</span><div><strong>选择目标与候选算法</strong><small>不可用算法会说明原因，选中的算法才进入后续对比实验。</small></div></div>
      ${targetAlgorithmControlsHtml}
      ${algorithmHtml ? `<div class="modeling-algorithm-grid">${algorithmHtml}</div>` : ""}
      ${guidanceHtml ? `<div class="modeling-guidance-list">${guidanceHtml}</div>` : ""}
      <div class="modeling-stage-actions"><button type="button" class="button compact ghost" data-modeling-step-back="split"${disabledAttr}>返回样本方案</button><button type="button" class="button compact secondary" data-modeling-step-next="tuning"${disabledAttr}>下一步：训练策略</button></div>
    </section>
    <section class="modeling-journey-stage" data-modeling-stage="tuning">
      <div class="modeling-stage-heading"><span>03</span><div><strong>确定训练投入</strong><small>调参轮数控制搜索预算；权重列只参与训练，不作为特征。</small></div></div>
      ${tuningControlsHtml}
      ${diagnosticsHtml ? `<div class="modeling-weight-diagnostics">${diagnosticsHtml}</div>` : ""}
      <div class="modeling-stage-actions"><button type="button" class="button compact ghost" data-modeling-step-back="model"${disabledAttr}>返回模型候选</button><button type="button" class="button compact secondary" data-modeling-step-next="review"${disabledAttr}>下一步：确认执行</button></div>
    </section>
    <section class="modeling-journey-stage" data-modeling-stage="review">
      <div class="modeling-stage-heading"><span>04</span><div><strong>复核 Agent 建议</strong><small>确认后才会开始特征筛选、调参、训练与实验对比。</small></div></div>
      <div class="modeling-spec-grid">${specChips}</div>
      ${warningHtml ? `<div class="modeling-setup-warnings">${warningHtml}</div>` : ""}
      ${reasonControlHtml}
      <div class="modeling-setup-foot gate-action-bar">
        <button type="button" class="button compact ghost" data-modeling-step-back="tuning"${disabledAttr}>返回训练策略</button>
        <span>目标、算法或调参调整会安全地重算后续步骤。</span>
        <button type="button" class="button compact secondary modeling-weight-adjust"${interactive ? ` data-modeling-weight-adjust="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "确认方案，开始执行" : "历史规格"}</button>
      </div>
    </section>
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
    setActionStatus("这是历史建模规格，请使用最新待确认步骤调整。", "error");
    return;
  }
  const splitError = modelingSplitInputError(form);
  if (splitError) {
    setActionStatus(splitError, "error");
    return;
  }
  const adjustParams = collectModelingSetupAdjustParams(form);
  const hasAdjustments = Object.keys(adjustParams).length > 0;
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
    setActionStatus(`目标类型 ${targetType} 与算法 ${mismatchedRecipe} 不匹配，请选择同一目标类型的算法。`, "error");
    return;
  }
  if (structuralKeys.some((key) => Object.prototype.hasOwnProperty.call(adjustParams, key)) && reason.length < 4) {
    setActionStatus("调整目标类型、算法或调参轮数时请填写变更原因。", "error");
    return;
  }
  const expectedStepId = form.dataset.modelingGateStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息，请刷新后重试。", "error");
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
  setActionStatus("正在应用建模设置并重新计算…", "busy");
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
        content: hasAdjustments ? (reason ? `调整建模规格：${reason}` : "调整建模规格") : "确认建模设置",
        ui_action: hasAdjustments ? "apply_modeling_setup" : "confirm_gate",
        ...(hasAdjustments ? { adjust_params: adjustParams } : {}),
        expected_step_id: expectedStepId,
        acceptance_mode: typeof context.agentAcceptanceModeValue === "function"
          ? context.agentAcceptanceModeValue()
          : (context.acceptanceMode || "manual"),
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
    setActionStatus(error?.message || "应用建模设置失败", "error");
  } finally {
    if (planRailTimer !== null) clearInterval(planRailTimer);
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  }
}

function collectModelingSetupAdjustParams(form) {
  const params = {};
  const splitMode = form.querySelector(".modeling-split-mode:checked");
  if (splitMode) {
    const currentSplit = normalizeSplitConfig(parseSplitConfig(form.dataset.currentSplitConfig));
    const selectedSplit = selectedSplitConfig(form, currentSplit);
    if (JSON.stringify(selectedSplit) !== JSON.stringify(currentSplit)) {
      params.split_config = selectedSplit;
    }
  }
  const target = form.querySelector(".modeling-target-select");
  if (target) {
    const value = String(target.value || "").trim();
    const current = String(target.getAttribute("data-current-target-type") || "").trim();
    if (value && value !== current) params.target_type = value;
  }
  const nTrials = form.querySelector(".modeling-n-trials-input");
  if (nTrials) {
    const rawValue = String(nTrials.value ?? "").trim();
    const rawCurrent = String(nTrials.getAttribute("data-current-n-trials") ?? "").trim();
    if (rawValue) {
      const value = Number(rawValue);
      const current = rawCurrent ? Number(rawCurrent) : null;
      if (Number.isFinite(value) && value !== current) params.n_trials = value;
    }
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
    const selectedSet = [...new Set(selected)].sort();
    const currentSet = [...new Set(current)].sort();
    if (selectedSet.join(",") !== currentSet.join(",")) params.recipes = selected;
  }
  const picked = form.querySelector(".modeling-weight-pick:checked");
  const sampleWeightCol = picked ? String(picked.value || "").trim() : "";
  const currentWeight = String(form.dataset.modelingCurrentWeight || "").trim();
  if (sampleWeightCol !== currentWeight) params.sample_weight_col = sampleWeightCol;
  return params;
}

function parseSplitConfig(value) {
  try {
    const parsed = JSON.parse(String(value || "{}"));
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function normalizeSplitConfig(config = {}) {
  const normalized = {};
  if (Number.isFinite(Number(config.test_size))) normalized.test_size = Number(config.test_size);
  if (config.oot_by_time) normalized.oot_by_time = String(config.oot_by_time);
  if (config.random_oot) normalized.random_oot = true;
  if ((config.oot_by_time || config.random_oot) && Number.isFinite(Number(config.oot_size))) {
    normalized.oot_size = Number(config.oot_size);
  }
  if (Array.isArray(config.group_cols) && config.group_cols.length) {
    normalized.group_cols = config.group_cols.map(String);
  }
  if (Array.isArray(config.rules) && config.rules.length) normalized.rules = config.rules;
  return normalized;
}

function selectedSplitConfig(form, current) {
  const mode = String(form.querySelector(".modeling-split-mode:checked")?.value || "none");
  const testPercent = Number(form.querySelector(".modeling-test-size-input")?.value || 25);
  const ootPercent = Number(form.querySelector(".modeling-oot-size-input")?.value || 20);
  const timeColumn = String(form.querySelector(".modeling-time-column-select")?.value || "").trim();
  const selected = {
    ...current,
    test_size: testPercent / 100,
  };
  delete selected.oot_by_time;
  delete selected.random_oot;
  delete selected.oot_size;
  if (mode === "time") {
    selected.oot_by_time = timeColumn;
    selected.oot_size = ootPercent / 100;
  } else if (mode === "random") {
    selected.random_oot = true;
    selected.oot_size = ootPercent / 100;
  }
  return normalizeSplitConfig(selected);
}

function modelingSplitInputError(form) {
  const modeControl = form.querySelector(".modeling-split-mode:checked");
  if (!modeControl) return "";
  const testPercent = Number(form.querySelector(".modeling-test-size-input")?.value);
  if (!Number.isFinite(testPercent) || testPercent <= 0 || testPercent > 50) {
    return "测试集比例必须大于 0% 且不超过 50%。";
  }
  const mode = String(modeControl.value || "none");
  if (mode !== "none") {
    const ootPercent = Number(form.querySelector(".modeling-oot-size-input")?.value);
    if (!Number.isFinite(ootPercent) || ootPercent <= 0 || ootPercent > 50) {
      return "OOT 比例必须大于 0% 且不超过 50%。";
    }
  }
  if (mode === "time" && !String(form.querySelector(".modeling-time-column-select")?.value || "").trim()) {
    return "按时间切分 OOT 时，请先选择时间字段。";
  }
  return "";
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
  const normalized = String(recipe || "").trim().toLowerCase();
  if (normalized.endsWith("_regressor")) return "continuous";
  if (normalized.endsWith("_multiclass")) return "multiclass";
  return "binary";
}

function humanMetricPolicy(policy) {
  const value = String(policy || "-").trim();
  const normalized = value.toLocaleLowerCase();
  if (normalized.includes("overfit") && normalized.includes("test") && normalized.includes("ks")) {
    return "测试集 KS（过拟合惩罚）";
  }
  if (normalized === "oot_ks" || normalized.includes("oot ks")) return "OOT KS";
  if (normalized === "test_ks" || normalized.includes("test ks")) return "测试集 KS";
  if (normalized.includes("auc")) return normalized.includes("oot") ? "OOT AUC" : "AUC";
  return value;
}

export function activateModelingJourneyStep(form, requestedStep) {
  if (!form || form.dataset.modelingReadonly === "true") return false;
  const stages = [...form.querySelectorAll("[data-modeling-stage]")];
  const nodes = [...form.querySelectorAll("[data-modeling-step-jump]")];
  const targetIndex = stages.findIndex((stage) => stage.dataset.modelingStage === requestedStep);
  if (targetIndex < 0) return false;
  stages.forEach((stage, index) => stage.classList.toggle("is-active", index === targetIndex));
  nodes.forEach((node, index) => {
    node.classList.toggle("is-active", index === targetIndex);
    node.classList.toggle("is-complete", index < targetIndex);
    node.setAttribute("aria-current", index === targetIndex ? "step" : "false");
  });
  form.dataset.modelingActiveStep = requestedStep;
  const progress = form.querySelector(".modeling-step-progress > i");
  if (progress) progress.style.setProperty("--modeling-progress", `${targetIndex / Math.max(stages.length - 1, 1) * 100}%`);
  form.querySelector("[data-modeling-stage].is-active")?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  return true;
}

export function handleModelingSetupInteraction(event) {
  const form = event.target?.closest?.(".modeling-setup-panel");
  if (!form || form.dataset.modelingReadonly === "true") return;
  if (event.type === "input" && event.target?.matches?.(".modeling-n-trials-input")) {
    const value = String(event.target.value || "-");
    const liveValue = form.querySelector('[data-modeling-live-value="trials"]');
    if (liveValue) liveValue.textContent = value;
    return;
  }
  const control = event.target?.closest?.("[data-modeling-step-next], [data-modeling-step-back], [data-modeling-step-jump]");
  if (!control) return;
  event.preventDefault();
  const target = control.dataset.modelingStepNext
    || control.dataset.modelingStepBack
    || control.dataset.modelingStepJump;
  const error = form.querySelector(".modeling-step-error");
  if (control.dataset.modelingStepNext && form.dataset.modelingActiveStep === "split") {
    const detail = modelingSplitInputError(form);
    if (detail) {
      if (error) error.textContent = detail;
      return;
    }
  }
  if (control.dataset.modelingStepNext && form.dataset.modelingActiveStep === "model") {
    const recipes = selectedModelingRecipes(form);
    const targetType = selectedModelingTargetType(form);
    const mismatch = recipes.find((recipe) => recipeFamily(recipe) !== targetType);
    const detail = !recipes.length
      ? "请先选择至少一个候选算法。"
      : mismatch
        ? `算法 ${mismatch} 与目标类型 ${targetType} 不匹配。`
        : "";
    if (detail) {
      if (error) error.textContent = detail;
      return;
    }
  }
  if (error) error.textContent = "";
  activateModelingJourneyStep(form, target);
}

export function handleModelingWeightAdjustClick(event, context = {}) {
  const button = event.target?.closest?.("[data-modeling-weight-adjust]");
  if (!button) return;
  event.preventDefault();
  return submitModelingWeightAdjust(button, context);
}
