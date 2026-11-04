function requiredElement(getElementById, id) {
  const element = getElementById(id);
  if (!element) throw new Error(`portfolio setup panel element is missing: ${id}`);
  return element;
}

function trimmedValue(element) {
  return String(element?.value || "").trim();
}

function portfolioGateExists(messages = []) {
  return (Array.isArray(messages) ? messages : []).some((message) => (
    message?.role === "assistant"
    && message?.metadata
    && typeof message.metadata.portfolio_states === "object"
  ));
}

export function portfolioTurnUsesDeterministicRoute(task, messages = []) {
  if (task?.task_type !== "portfolio") return false;
  for (let index = (Array.isArray(messages) ? messages.length : 0) - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role !== "assistant") continue;
    const metadata = message?.metadata || {};
    return metadata.kind === "portfolio_setup_required"
      || metadata.kind === "gate"
      || Boolean(metadata.portfolio_states);
  }
  return false;
}

// Eval S9: the column boxes were empty while their grey placeholders were the
// sample's real column names, so the form looked filled and failed on
// submit. Columns whose name matches exactly one alias are prefilled for the
// user to check; the platform still validates everything on submit.
const PORTFOLIO_COLUMN_ALIASES = Object.freeze({
  id_col: ["loan_id", "loanid", "loan_no", "contract_id", "account_id", "acct_id", "借据号", "贷款编号", "合同号"],
  snapshot_col: ["snapshot_month", "snapshot", "snapshot_date", "stat_month", "data_month", "as_of_month", "month", "快照月", "统计月"],
  bucket_col: ["bucket", "dpd_bucket", "delinquency_bucket", "overdue_bucket", "逾期桶", "逾期状态"],
  balance_col: ["balance", "ead", "outstanding", "os_balance", "principal_balance", "余额", "在贷余额"],
  segment_col: ["segment", "segment_name", "product", "channel", "分群", "客群"],
});

function normalizedColumn(name) {
  return String(name || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
}

export function portfolioColumnSuggestions(columns = []) {
  const names = columns.map((column) => String(column?.name ?? column ?? "")).filter(Boolean);
  const suggestions = {};
  for (const [field, aliases] of Object.entries(PORTFOLIO_COLUMN_ALIASES)) {
    const matches = names.filter((name) => aliases.includes(normalizedColumn(name)));
    if (matches.length === 1) suggestions[field] = matches[0];
  }
  return suggestions;
}

export function createPortfolioSetupPanel(dependencies = {}) {
  const getElementById = dependencies.getElementById || ((id) => document.getElementById(id));
  const request = dependencies.api;
  const getSelectedTask = dependencies.getSelectedTask || (() => null);
  const getAgentMessages = dependencies.getAgentMessages || (() => []);
  const captureView = dependencies.captureView || (() => getSelectedTask()?.id);
  const isCurrentView = dependencies.isCurrentView || (view => view === getSelectedTask()?.id);
  const onMessages = dependencies.onMessages || (() => {});
  const onSubmitted = dependencies.onSubmitted || (() => {});
  const onError = dependencies.onError || (() => {});
  if (typeof request !== "function") throw new TypeError("portfolio setup panel requires api");

  const elements = {
    root: requiredElement(getElementById, "portfolioSetupPanel"),
    form: requiredElement(getElementById, "portfolioSetupForm"),
    status: requiredElement(getElementById, "portfolioSetupStatus"),
    submit: requiredElement(getElementById, "portfolioSetupSubmit"),
    idCol: requiredElement(getElementById, "portfolioIdCol"),
    snapshotCol: requiredElement(getElementById, "portfolioSnapshotCol"),
    bucketCol: requiredElement(getElementById, "portfolioBucketCol"),
    balanceCol: requiredElement(getElementById, "portfolioBalanceCol"),
    segmentCol: requiredElement(getElementById, "portfolioSegmentCol"),
    lossState: requiredElement(getElementById, "portfolioLossState"),
    lgd: requiredElement(getElementById, "portfolioLgd"),
    horizonMonths: requiredElement(getElementById, "portfolioHorizonMonths"),
    scoreCol: requiredElement(getElementById, "portfolioScoreCol"),
    experimentId: requiredElement(getElementById, "portfolioExperimentId"),
  };

  let activeSubmission = null;
  const isSubmitting = () => Boolean(activeSubmission && isCurrentView(activeSubmission.view));
  const beginActivity = dependencies.beginActivity || (() => ({}));
  const endActivity = dependencies.endActivity || (() => {});

  function setStatus(message = "", kind = "") {
    elements.status.textContent = message;
    elements.status.className = `portfolio-setup-status${kind ? ` ${kind}` : ""}`;
  }

  let prefilledTaskId = "";
  const listDatasets = dependencies.listDatasets;

  async function prefillColumns(task) {
    if (typeof listDatasets !== "function" || !task?.id || prefilledTaskId === task.id) return;
    prefilledTaskId = task.id;
    const fields = {
      id_col: elements.idCol,
      snapshot_col: elements.snapshotCol,
      bucket_col: elements.bucketCol,
      balance_col: elements.balanceCol,
      segment_col: elements.segmentCol,
    };
    let payload;
    try {
      payload = await listDatasets(task.id);
    } catch (_error) {
      return; // No datasets yet: the user fills the form as before.
    }
    if (getSelectedTask()?.id !== task.id) return;
    const datasets = Array.isArray(payload?.datasets) ? payload.datasets : [];
    if (datasets.length !== 1) return; // Several tables: which one is the panel is the user's call.
    const suggestions = portfolioColumnSuggestions(datasets[0].columns || []);
    let filled = 0;
    for (const [field, element] of Object.entries(fields)) {
      if (!suggestions[field] || trimmedValue(element)) continue;
      element.value = suggestions[field];
      element.dataset.prefilled = "true";
      filled += 1;
    }
    if (filled && !isSubmitting()) {
      setStatus(`已按上传表的列名预填 ${filled} 项，请核对后提交。`, "info");
    }
  }

  function renderAvailability() {
    const task = getSelectedTask();
    const visible = task?.task_type === "portfolio"
      && !portfolioGateExists(getAgentMessages());
    elements.root.hidden = !visible;
    elements.submit.disabled = isSubmitting() || !visible;
    if (visible) void prefillColumns(task);
    return visible;
  }

  function readContract() {
    const stringFields = [
      ["贷款 ID 列", "id_col", elements.idCol],
      ["快照月列", "snapshot_col", elements.snapshotCol],
      ["逾期桶列", "bucket_col", elements.bucketCol],
      ["余额/EAD 列", "balance_col", elements.balanceCol],
      ["业务分群列", "segment_col", elements.segmentCol],
      ["损失态", "loss_state", elements.lossState],
    ];
    const contract = {};
    for (const [label, key, element] of stringFields) {
      const value = trimmedValue(element);
      if (!value) throw new Error(`请填写${label}。`);
      contract[key] = value;
    }

    const lgd = Number(trimmedValue(elements.lgd));
    if (!Number.isFinite(lgd) || lgd < 0 || lgd > 1) {
      throw new Error("LGD 必须是 0 到 1 之间的数字。");
    }
    const horizonMonths = Number(trimmedValue(elements.horizonMonths));
    if (!Number.isInteger(horizonMonths) || horizonMonths <= 0) {
      throw new Error("预测期限必须是正整数月数。");
    }
    contract.lgd = lgd;
    contract.horizon_months = horizonMonths;

    const scoreCol = trimmedValue(elements.scoreCol);
    const experimentId = trimmedValue(elements.experimentId);
    if (Boolean(scoreCol) !== Boolean(experimentId)) {
      throw new Error("趋势分析的分数列和实验 ID 必须同时填写，或同时留空。");
    }
    if (scoreCol) {
      contract.score_col = scoreCol;
      contract.experiment_id = experimentId;
    }
    return contract;
  }

  async function submit(event) {
    event?.preventDefault?.();
    if (isSubmitting()) return false;
    const task = getSelectedTask();
    if (!task?.id || task.task_type !== "portfolio") {
      setStatus("请选择组合分析任务后再提交。", "error");
      return false;
    }

    let portfolioRequest;
    try {
      portfolioRequest = readContract();
    } catch (error) {
      setStatus(error.message || "组合分析口径不完整。", "error");
      return false;
    }

    const view = captureView();
    const lease = beginActivity("panel:portfolio", task.id);
    if (!lease) return false;
    const submission = { view, lease };
    activeSubmission = submission;
    elements.submit.disabled = true;
    setStatus("正在校验字段和组合口径…", "busy");
    try {
      const result = await request(`api/tasks/${task.id}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: "提交已确认的组合分析口径",
          portfolio_request: portfolioRequest,
        }),
      });
      if (!isCurrentView(view)) return false;
      if (Array.isArray(result?.messages)) onMessages(result.messages, view);
      setStatus("口径已校验，请确认逾期桶顺序。", "success");
      await onSubmitted(result, view);
      return true;
    } catch (error) {
      if (!isCurrentView(view)) return false;
      setStatus(error?.message || "组合分析口径提交失败。", "error");
      onError(error);
      return false;
    } finally {
      if (activeSubmission === submission) activeSubmission = null;
      endActivity(lease);
      renderAvailability();
    }
  }

  elements.form.addEventListener("submit", submit);
  renderAvailability();

  return {
    renderAvailability,
    submit,
  };
}
