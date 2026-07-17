const STRATEGY_CLARIFICATION_CODE = "strategy_business_inputs_required";

const MISSING_FIELD_LABELS = {
  objective: "业务目标",
  constraints: "至少一项策略约束",
  max_bad_rate_or_min_approval_rate: "至少一项策略约束",
  "profit.ead_col": "EAD 列",
  "profit.pd_col": "PD 列",
  "profit.annual_rate": "年化利率",
  "profit.funding_rate": "资金成本率",
  "profit.lgd": "LGD",
  "profit.operating_cost_per_loan": "单笔运营成本",
  "profit.term_months": "期限（月）",
};

export function isStrategyClarificationMessage(message) {
  const metadata = message?.metadata || {};
  return metadata.kind === "clarification"
    && metadata.clarification?.code === STRATEGY_CLARIFICATION_CODE;
}

function missingFieldLabels(message) {
  const fields = Array.isArray(message?.metadata?.clarification?.missing_fields)
    ? message.metadata.clarification.missing_fields
    : [];
  const labels = [];
  for (const field of fields) {
    const label = MISSING_FIELD_LABELS[String(field)] || String(field);
    if (label && !labels.includes(label)) labels.push(label);
  }
  return labels;
}

function option(value, label, selectedValue) {
  const selected = value === selectedValue ? " selected" : "";
  return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(label)}</option>`;
}

function inputValue(value) {
  return value === null || value === undefined ? "" : escapeHtml(value);
}

export function renderStrategyClarification(message, options = {}) {
  if (!isStrategyClarificationMessage(message)) return "";
  const interactive = options.interactive !== false;
  const readonly = interactive ? "false" : "true";
  const disabled = interactive ? "" : " disabled";
  const readonlyTitle = interactive
    ? ""
    : ' title="此为历史澄清请求，只能查看，不能再次提交"';
  const clarification = message?.metadata?.clarification || {};
  const missingFields = clarification.missing_fields || [];
  const currentInput = clarification.current_input
    && typeof clarification.current_input === "object"
    ? clarification.current_input
    : {};
  const currentProfit = currentInput.profit && typeof currentInput.profit === "object"
    ? currentInput.profit
    : {};
  const currentObjective = String(currentInput.objective || "").trim();
  const inferredObjective = !currentObjective
    && Array.isArray(missingFields)
    && missingFields.some((field) => String(field).startsWith("profit."))
    ? "max_profit"
    : "";
  const selectedObjective = currentObjective || inferredObjective;
  const missingLabels = missingFieldLabels(message);
  const missingHtml = missingLabels.length
    ? `<p class="strategy-clarification-missing">缺少：${missingLabels.map(escapeHtml).join("、")}</p>`
    : "";
  const profitHidden = selectedObjective === "max_profit" ? "" : " hidden";
  const actionLabel = interactive ? "保存口径并开始策略开发" : "历史请求（只读）";

  return [
    `<div class="strategy-clarification-card" data-strategy-clarification="1" data-strategy-clarification-readonly="${readonly}"${readonlyTitle}>`,
    '<header class="strategy-clarification-head">',
    '<div><span class="strategy-clarification-eyebrow">需要业务确认</span><h4>补充策略业务口径</h4></div>',
    '<p>平台不会用技术默认值代替经营目标。请提交完整目标与约束后再生成策略开发计划。</p>',
    "</header>",
    missingHtml,
    '<div class="strategy-clarification-grid">',
    '<label class="strategy-clarification-field"><span>优化目标 <b aria-hidden="true">*</b></span>',
    `<select data-strategy-objective${disabled}>`,
    option("", "请选择业务目标", selectedObjective),
    option("max_approval", "约束坏率下最大化通过率", selectedObjective),
    option("max_profit", "最大化预期利润", selectedObjective),
    "</select></label>",
    '<label class="strategy-clarification-field"><span>审批后坏率上限（0-1）</span>',
    `<input data-strategy-max-bad-rate type="number" min="0" max="1" step="0.001" inputmode="decimal" value="${inputValue(currentInput.max_bad_rate)}" placeholder="例如：0.05"${disabled}></label>`,
    '<label class="strategy-clarification-field"><span>通过率下限（0-1）</span>',
    `<input data-strategy-min-approval-rate type="number" min="0" max="1" step="0.001" inputmode="decimal" value="${inputValue(currentInput.min_approval_rate)}" placeholder="例如：0.60"${disabled}></label>`,
    '<label class="strategy-clarification-field is-wide"><span>基线策略 ID（可选）</span>',
    `<input data-strategy-baseline-id autocomplete="off" value="${inputValue(currentInput.baseline_strategy_id)}" placeholder="用于 champion/challenger 对比"${disabled}></label>`,
    "</div>",
    `<fieldset class="strategy-clarification-profit" data-strategy-profit-fields${profitHidden}${disabled}>`,
    '<legend>利润参数</legend>',
    '<p>利润目标必须同时提供 EAD、PD 与完整收益成本参数，缺失值不会静默按 0 处理。</p>',
    '<div class="strategy-clarification-grid">',
    '<label class="strategy-clarification-field"><span>EAD 列</span>',
    `<input data-strategy-ead-col autocomplete="off" value="${inputValue(currentProfit.ead_col)}" placeholder="例如：ead"${disabled}></label>`,
    '<label class="strategy-clarification-field"><span>PD 列</span>',
    `<input data-strategy-pd-col autocomplete="off" value="${inputValue(currentProfit.pd_col)}" placeholder="例如：pd"${disabled}></label>`,
    '<label class="strategy-clarification-field"><span>年化利率</span>',
    `<input data-strategy-annual-rate type="number" min="0" max="1" step="0.001" inputmode="decimal" value="${inputValue(currentProfit.annual_rate)}" placeholder="例如：0.18"${disabled}></label>`,
    '<label class="strategy-clarification-field"><span>资金成本率</span>',
    `<input data-strategy-funding-rate type="number" min="0" max="1" step="0.001" inputmode="decimal" value="${inputValue(currentProfit.funding_rate)}" placeholder="例如：0.04"${disabled}></label>`,
    '<label class="strategy-clarification-field"><span>LGD</span>',
    `<input data-strategy-lgd type="number" min="0" max="1" step="0.001" inputmode="decimal" value="${inputValue(currentProfit.lgd)}" placeholder="例如：0.60"${disabled}></label>`,
    '<label class="strategy-clarification-field"><span>单笔运营成本</span>',
    `<input data-strategy-operating-cost type="number" min="0" step="0.01" inputmode="decimal" value="${inputValue(currentProfit.operating_cost_per_loan)}" placeholder="例如：20"${disabled}></label>`,
    '<label class="strategy-clarification-field"><span>期限（月）</span>',
    `<input data-strategy-term-months type="number" min="1" step="1" inputmode="numeric" value="${inputValue(currentProfit.term_months)}" placeholder="例如：12"${disabled}></label>`,
    "</div>",
    "</fieldset>",
    '<p class="strategy-clarification-error" data-strategy-clarification-error role="alert" aria-live="polite"></p>',
    '<div class="strategy-clarification-actions">',
    `<button type="button" class="button compact primary" data-strategy-clarification-submit="1"${disabled}${readonlyTitle}>${actionLabel}</button>`,
    "</div>",
    "</div>",
  ].join("");
}

function fieldValue(wrap, selector) {
  return String(wrap?.querySelector?.(selector)?.value ?? "").trim();
}

function optionalNumber(wrap, selector) {
  const value = fieldValue(wrap, selector);
  return value === "" ? null : Number(value);
}

function collectStrategyInput(wrap) {
  const objective = fieldValue(wrap, "[data-strategy-objective]");
  const input = {
    entry_mode: "strategy_development",
    objective,
    max_bad_rate: optionalNumber(wrap, "[data-strategy-max-bad-rate]"),
    min_approval_rate: optionalNumber(wrap, "[data-strategy-min-approval-rate]"),
    baseline_strategy_id: fieldValue(wrap, "[data-strategy-baseline-id]") || null,
    profit: null,
  };
  if (objective === "max_profit") {
    input.profit = {
      ead_col: fieldValue(wrap, "[data-strategy-ead-col]"),
      pd_col: fieldValue(wrap, "[data-strategy-pd-col]"),
      annual_rate: optionalNumber(wrap, "[data-strategy-annual-rate]"),
      funding_rate: optionalNumber(wrap, "[data-strategy-funding-rate]"),
      lgd: optionalNumber(wrap, "[data-strategy-lgd]"),
      operating_cost_per_loan: optionalNumber(wrap, "[data-strategy-operating-cost]"),
      term_months: optionalNumber(wrap, "[data-strategy-term-months]"),
    };
  }
  return input;
}

export function strategyClarificationInputError(input) {
  if (!input || !["max_approval", "max_profit"].includes(input.objective)) {
    return "请选择完整策略开发的业务目标。";
  }
  for (const [label, value] of [
    ["审批后坏率上限", input.max_bad_rate],
    ["通过率下限", input.min_approval_rate],
  ]) {
    if (value !== null && (!Number.isFinite(value) || value < 0 || value > 1)) {
      return `${label}必须是 0 到 1 之间的数字。`;
    }
  }
  if (input.max_bad_rate === null && input.min_approval_rate === null) {
    return "请至少填写一个审批后坏率上限或通过率下限。";
  }
  if (input.objective !== "max_profit") return "";
  const profit = input.profit || {};
  const requiredNumbers = [
    profit.annual_rate,
    profit.funding_rate,
    profit.lgd,
    profit.operating_cost_per_loan,
    profit.term_months,
  ];
  if (
    !profit.ead_col
    || !profit.pd_col
    || requiredNumbers.some((value) => value === null || !Number.isFinite(value))
  ) {
    return "利润最大化需要填写 EAD/PD 列和完整收益参数。";
  }
  if (
    profit.annual_rate < 0 || profit.annual_rate > 1
    || profit.funding_rate < 0 || profit.funding_rate > 1
    || profit.lgd < 0 || profit.lgd > 1
    || profit.operating_cost_per_loan < 0
    || !Number.isInteger(profit.term_months) || profit.term_months < 1
  ) {
    return "利润参数范围无效：率和 LGD 需在 0-1，成本不得为负，期限必须是正整数。";
  }
  return "";
}

function controllerContext(context = {}) {
  return {
    taskId: typeof context.getSelectedTaskId === "function"
      ? context.getSelectedTaskId()
      : context.selectedTaskId,
    api: context.api,
    setActionStatus: context.setActionStatus || (() => {}),
    setAgentMessages: context.setAgentMessages || (() => {}),
    renderAgentConversation: context.renderAgentConversation || (() => {}),
    pollAgentMessagesUntilSettled: context.pollAgentMessagesUntilSettled
      || (() => Promise.resolve()),
    refreshAgentMessages: context.refreshAgentMessages || (() => Promise.resolve()),
    resetFetchThrottle: context.resetFetchThrottle || (() => {}),
    renderWorkflowStepper: context.renderWorkflowStepper || (() => {}),
  };
}

function setFormError(wrap, message) {
  const error = wrap?.querySelector?.("[data-strategy-clarification-error]");
  if (error) error.textContent = String(message || "");
}

export async function submitStrategyClarification(button, context = {}) {
  const wrap = button?.closest?.("[data-strategy-clarification]");
  const {
    taskId,
    api,
    setActionStatus,
    setAgentMessages,
    renderAgentConversation,
    pollAgentMessagesUntilSettled,
    refreshAgentMessages,
    resetFetchThrottle,
    renderWorkflowStepper,
  } = controllerContext(context);
  if (!wrap) return;
  if (wrap?.dataset?.strategyClarificationReadonly === "true") return;
  if (!taskId || typeof api !== "function") {
    const message = "缺少当前策略任务，请刷新后重试。";
    setFormError(wrap, message);
    setActionStatus(message, "error");
    return;
  }

  const strategyInput = collectStrategyInput(wrap);
  const validationError = strategyClarificationInputError(strategyInput);
  if (validationError) {
    setFormError(wrap, validationError);
    setActionStatus(validationError, "error");
    return;
  }

  setFormError(wrap, "");
  button.disabled = true;
  setActionStatus("正在保存业务口径并生成策略计划…", "busy");
  try {
    const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify({
        content: "补充策略业务口径",
        strategy_input: strategyInput,
      }),
    });
    const pollPromise = Promise.resolve(
      pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true }),
    ).catch(() => {});
    const result = await requestPromise;
    await pollPromise;
    setAgentMessages(result?.messages);
    renderAgentConversation();
    await refreshAgentMessages(taskId);
    setActionStatus("策略业务口径已保存，开发计划已生成。", "success");
  } catch (error) {
    button.disabled = false;
    const message = error?.message || "策略业务口径提交失败";
    setFormError(wrap, message);
    setActionStatus(message, "error");
  } finally {
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  }
}

export function handleStrategyClarificationSubmit(event, context = {}) {
  const button = event.target?.closest?.("[data-strategy-clarification-submit]");
  if (!button) return false;
  event.preventDefault();
  void submitStrategyClarification(button, context);
  return true;
}

export function handleStrategyClarificationChange(event) {
  const select = event.target?.closest?.("[data-strategy-objective]");
  if (!select) return false;
  const wrap = select.closest?.("[data-strategy-clarification]");
  const profitFields = wrap?.querySelector?.("[data-strategy-profit-fields]");
  if (profitFields) profitFields.hidden = select.value !== "max_profit";
  return true;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
