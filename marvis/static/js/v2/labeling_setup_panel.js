const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const ISO_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

function requiredElement(getElementById, id) {
  const element = getElementById(id);
  if (!element) throw new Error(`labeling setup panel element is missing: ${id}`);
  return element;
}

function trimmedValue(element) {
  return String(element?.value || "").trim();
}

function nonNegativeInteger(element, label, { positive = false } = {}) {
  const value = Number(trimmedValue(element));
  if (!Number.isInteger(value) || value < (positive ? 1 : 0)) {
    throw new Error(`${label}必须是${positive ? "正" : "非负"}整数。`);
  }
  return value;
}

function validIsoDate(value) {
  if (!ISO_DATE_PATTERN.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

function labelingUiState(messages = []) {
  const items = Array.isArray(messages) ? messages : [];
  for (let index = items.length - 1; index >= 0; index -= 1) {
    const message = items[index];
    if (message?.role !== "assistant") continue;
    const metadata = message?.metadata || {};
    const resolution = metadata.labeling_proposal_resolution;
    if (resolution === "confirmed") return { mode: "complete", proposal: null };
    if (resolution === "stale" || resolution === "cancelled") {
      return { mode: "setup", proposal: null };
    }
    if (
      metadata.kind === "labeling_preplan_confirmation"
      && metadata.labeling_proposal
      && typeof metadata.labeling_proposal === "object"
    ) {
      return { mode: "confirm", proposal: metadata.labeling_proposal };
    }
  }
  return { mode: "setup", proposal: null };
}

function workspaceAvailability(workspaceController, task) {
  const state = workspaceController?.getState?.() || {};
  const snapshot = state.serverSnapshot;
  if (state.loading || state.saving) {
    return { ready: false, message: "数据工作区正在同步，请稍候。", state, snapshot };
  }
  if (state.dirty) {
    return {
      ready: false,
      message: "请先保存或丢弃数据工作区修改，再提交标签口径。",
      state,
      snapshot,
    };
  }
  if (!snapshot || snapshot.task_id !== task?.id) {
    return { ready: false, message: "请先加载当前任务的数据工作区。", state, snapshot };
  }
  if (
    !String(snapshot.active_dataset_id || "").trim()
    || !SHA256_PATTERN.test(String(snapshot.active_dataset_content_hash || ""))
  ) {
    return { ready: false, message: "请先绑定数据集并保存数据工作区。", state, snapshot };
  }
  if (
    !Number.isInteger(snapshot.revision)
    || snapshot.revision < 0
    || !Number.isInteger(snapshot.analysis_generation)
    || snapshot.analysis_generation < 0
  ) {
    return { ready: false, message: "数据工作区版本信息无效，请刷新后重试。", state, snapshot };
  }
  return { ready: true, message: "填写经业务确认的观察窗、表现窗和坏样本规则。", state, snapshot };
}

function proposalSummary(proposal = {}) {
  const maturity = proposal.maturity && typeof proposal.maturity === "object"
    ? proposal.maturity
    : {};
  const immature = Array.isArray(maturity.immature_cohorts)
    ? maturity.immature_cohorts
    : [];
  const maturityText = maturity.all_matured
    ? "全部 cohort 已达到观察成熟度"
    : `${immature.length} 个 cohort 尚未成熟：${immature.join("、") || "请查看明细"}`;
  return [
    `来源：${proposal.source_dataset_name || "已绑定数据集"}`,
    `截至切点保留 ${Number(proposal.rows_at_as_of || 0)} 行`,
    `排除切点后 ${Number(proposal.rows_excluded_after_as_of || 0)} 行`,
    `规则：${proposal.rule_summary || "未提供"}`,
    maturityText,
  ].join(" · ");
}

export function createLabelingSetupPanel(dependencies = {}) {
  const getElementById = dependencies.getElementById || ((id) => document.getElementById(id));
  const request = dependencies.api;
  const workspaceController = dependencies.workspaceController;
  const getSelectedTask = dependencies.getSelectedTask || (() => null);
  const getAgentMessages = dependencies.getAgentMessages || (() => []);
  const onMessages = dependencies.onMessages || (() => {});
  const onSubmitted = dependencies.onSubmitted || (() => {});
  const onError = dependencies.onError || (() => {});
  if (typeof request !== "function") throw new TypeError("labeling setup panel requires api");
  if (!workspaceController || typeof workspaceController.getState !== "function") {
    throw new TypeError("labeling setup panel requires a data workspace controller");
  }

  const elements = {
    root: requiredElement(getElementById, "labelingSetupPanel"),
    form: requiredElement(getElementById, "labelingSetupForm"),
    proposalPanel: requiredElement(getElementById, "labelingProposalPanel"),
    status: requiredElement(getElementById, "labelingSetupStatus"),
    submit: requiredElement(getElementById, "labelingSetupSubmit"),
    proposalSummary: requiredElement(getElementById, "labelingProposalSummary"),
    confirm: requiredElement(getElementById, "labelingProposalConfirm"),
    idCol: requiredElement(getElementById, "labelingIdCol"),
    mobCol: requiredElement(getElementById, "labelingMobCol"),
    cohortCol: requiredElement(getElementById, "labelingCohortCol"),
    dateCol: requiredElement(getElementById, "labelingDateCol"),
    asOfDate: requiredElement(getElementById, "labelingAsOfDate"),
    targetCol: requiredElement(getElementById, "labelingTargetCol"),
    observationWindow: requiredElement(getElementById, "labelingObservationWindow"),
    performanceWindow: requiredElement(getElementById, "labelingPerformanceWindow"),
    atMob: requiredElement(getElementById, "labelingAtMob"),
    ruleKind: requiredElement(getElementById, "labelingRuleKind"),
    dpdFields: requiredElement(getElementById, "labelingDpdFields"),
    dpdCol: requiredElement(getElementById, "labelingDpdCol"),
    thresholdDpd: requiredElement(getElementById, "labelingThresholdDpd"),
    statusFields: requiredElement(getElementById, "labelingStatusFields"),
    statusCol: requiredElement(getElementById, "labelingStatusCol"),
    thresholdStatus: requiredElement(getElementById, "labelingThresholdStatus"),
    states: requiredElement(getElementById, "labelingStates"),
  };

  let submitting = false;

  function setStatus(message = "", kind = "") {
    elements.status.textContent = message;
    elements.status.className = `labeling-setup-status${kind ? ` ${kind}` : ""}`;
  }

  function renderRuleFields() {
    const usesStatus = trimmedValue(elements.ruleKind).toLowerCase() === "status";
    elements.dpdFields.hidden = usesStatus;
    elements.statusFields.hidden = !usesStatus;
  }

  function renderAvailability() {
    const task = getSelectedTask();
    const uiState = labelingUiState(getAgentMessages());
    const taskVisible = task?.task_type === "data_join";
    const visible = taskVisible && uiState.mode !== "complete";
    const availability = workspaceAvailability(workspaceController, task);

    elements.root.hidden = !visible;
    elements.form.hidden = !visible || uiState.mode !== "setup";
    elements.proposalPanel.hidden = !visible || uiState.mode !== "confirm";
    renderRuleFields();

    if (!visible) {
      elements.submit.disabled = true;
      elements.confirm.disabled = true;
      return false;
    }
    if (uiState.mode === "confirm") {
      elements.proposalSummary.textContent = proposalSummary(uiState.proposal);
      elements.submit.disabled = true;
      elements.confirm.disabled = submitting || !availability.ready;
      setStatus(
        availability.ready
          ? "平台已完成只读核验；确认后才会生成并执行标签构造计划。"
          : availability.message,
        availability.ready ? "success" : "warning",
      );
      return true;
    }
    elements.confirm.disabled = true;
    elements.submit.disabled = submitting || !availability.ready;
    setStatus(availability.message, availability.ready ? "" : "warning");
    return true;
  }

  function readContract(snapshot) {
    const contract = {
      dataset_id: String(snapshot.active_dataset_id),
      expected_content_hash: String(snapshot.active_dataset_content_hash),
      workspace_revision: snapshot.revision,
      analysis_generation: snapshot.analysis_generation,
    };
    const stringFields = [
      ["贷款 ID 列", "id_col", elements.idCol],
      ["账龄 MOB 列", "mob_col", elements.mobCol],
      ["cohort 列", "cohort_col", elements.cohortCol],
      ["快照日期列", "date_col", elements.dateCol],
      ["目标标签列", "target_col", elements.targetCol],
    ];
    for (const [label, key, element] of stringFields) {
      const value = trimmedValue(element);
      if (!value) throw new Error(`请填写${label}。`);
      contract[key] = value;
    }

    const ruleKind = trimmedValue(elements.ruleKind).toLowerCase();
    if (ruleKind !== "dpd" && ruleKind !== "status") {
      throw new Error("请选择 DPD 或有序状态坏样本规则。");
    }
    contract.rule_kind = ruleKind;
    if (ruleKind === "dpd") {
      const dpdCol = trimmedValue(elements.dpdCol);
      const thresholdDpd = Number(trimmedValue(elements.thresholdDpd));
      if (!dpdCol) throw new Error("请填写 DPD 列。");
      if (!Number.isFinite(thresholdDpd) || thresholdDpd < 0) {
        throw new Error("DPD 阈值必须是非负数字。");
      }
      contract.dpd_col = dpdCol;
      contract.threshold_dpd = thresholdDpd;
    } else {
      const statusCol = trimmedValue(elements.statusCol);
      const thresholdStatus = trimmedValue(elements.thresholdStatus);
      const states = trimmedValue(elements.states)
        .split(/[,，、\n]+/)
        .map((value) => value.trim())
        .filter(Boolean);
      if (!statusCol) throw new Error("请填写状态列。");
      if (!thresholdStatus) throw new Error("请填写坏样本起始状态。");
      if (states.length < 2) throw new Error("请按从好到坏顺序填写至少两个状态。");
      if (new Set(states).size !== states.length) throw new Error("状态顺序中存在重复值。");
      if (!states.includes(thresholdStatus)) throw new Error("坏样本起始状态必须出现在状态顺序中。");
      contract.status_col = statusCol;
      contract.threshold_status = thresholdStatus;
      contract.states = states;
    }

    const asOfDate = trimmedValue(elements.asOfDate);
    if (!validIsoDate(asOfDate)) throw new Error("数据切点必须是有效的 YYYY-MM-DD 日期。");
    contract.as_of_date = asOfDate;
    contract.observation_window = nonNegativeInteger(elements.observationWindow, "观察窗");
    contract.performance_window = nonNegativeInteger(
      elements.performanceWindow,
      "表现窗",
      { positive: true },
    );
    contract.at_mob = nonNegativeInteger(elements.atMob, "标签判定 MOB", { positive: true });
    if (contract.at_mob <= contract.observation_window) {
      throw new Error("标签判定 MOB 必须大于观察窗。");
    }
    if (contract.at_mob > contract.observation_window + contract.performance_window) {
      throw new Error("标签判定 MOB 不能超过观察窗与表现窗之和。");
    }
    return contract;
  }

  async function submit(event) {
    event?.preventDefault?.();
    if (submitting) return false;
    const task = getSelectedTask();
    if (!task?.id || task.task_type !== "data_join") {
      setStatus("请选择数据处理任务后再提交标签口径。", "error");
      return false;
    }
    const availability = workspaceAvailability(workspaceController, task);
    if (!availability.ready) {
      setStatus(availability.message, "warning");
      return false;
    }

    let labelingRequest;
    try {
      labelingRequest = readContract(availability.snapshot);
    } catch (error) {
      setStatus(error.message || "标签构造口径不完整。", "error");
      return false;
    }

    submitting = true;
    elements.submit.disabled = true;
    setStatus("正在核验数据切点、字段和 cohort 成熟度…", "busy");
    try {
      const result = await request(`api/tasks/${task.id}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: "提交已确认的标签构造口径",
          labeling_request: labelingRequest,
        }),
      });
      if (Array.isArray(result?.messages)) onMessages(result.messages);
      await onSubmitted(result);
      return true;
    } catch (error) {
      setStatus(error?.message || "标签构造口径提交失败。", "error");
      onError(error);
      return false;
    } finally {
      submitting = false;
      renderAvailability();
    }
  }

  async function confirm() {
    if (submitting) return false;
    const task = getSelectedTask();
    const uiState = labelingUiState(getAgentMessages());
    const availability = workspaceAvailability(workspaceController, task);
    if (!task?.id || task.task_type !== "data_join" || uiState.mode !== "confirm") {
      setStatus("当前没有可确认的标签构造提案。", "error");
      return false;
    }
    if (!availability.ready) {
      setStatus(availability.message, "warning");
      return false;
    }
    submitting = true;
    elements.confirm.disabled = true;
    setStatus("正在确认并生成受治理的标签构造计划…", "busy");
    try {
      const result = await request(`api/tasks/${task.id}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({ content: "确认" }),
      });
      if (Array.isArray(result?.messages)) onMessages(result.messages);
      await onSubmitted(result);
      return true;
    } catch (error) {
      setStatus(error?.message || "标签构造提案确认失败。", "error");
      onError(error);
      return false;
    } finally {
      submitting = false;
      renderAvailability();
    }
  }

  elements.ruleKind.addEventListener("change", renderRuleFields);
  elements.form.addEventListener("submit", submit);
  elements.confirm.addEventListener("click", confirm);
  workspaceController.subscribe?.(() => renderAvailability());
  renderAvailability();

  return { confirm, renderAvailability, submit };
}
