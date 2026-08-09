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

export function createPortfolioSetupPanel(dependencies = {}) {
  const getElementById = dependencies.getElementById || ((id) => document.getElementById(id));
  const request = dependencies.api;
  const getSelectedTask = dependencies.getSelectedTask || (() => null);
  const getAgentMessages = dependencies.getAgentMessages || (() => []);
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

  let submitting = false;

  function setStatus(message = "", kind = "") {
    elements.status.textContent = message;
    elements.status.className = `portfolio-setup-status${kind ? ` ${kind}` : ""}`;
  }

  function renderAvailability() {
    const task = getSelectedTask();
    const visible = task?.task_type === "portfolio"
      && !portfolioGateExists(getAgentMessages());
    elements.root.hidden = !visible;
    elements.submit.disabled = submitting || !visible;
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
    if (submitting) return false;
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

    submitting = true;
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
      if (Array.isArray(result?.messages)) onMessages(result.messages);
      setStatus("口径已校验，请确认逾期桶顺序。", "success");
      await onSubmitted(result);
      return true;
    } catch (error) {
      setStatus(error?.message || "组合分析口径提交失败。", "error");
      onError(error);
      return false;
    } finally {
      submitting = false;
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
