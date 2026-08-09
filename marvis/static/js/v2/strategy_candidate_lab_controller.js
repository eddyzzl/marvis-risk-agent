/** Candidate Lab state, network, event, and lifecycle coordination. */

import { escapeHtml } from "../ui-utils.js";

import {
  getStrategyCandidateLab,
  submitStrategyCandidateLabRequest,
} from "./api_v2.js";

import {
  loadStrategyCandidateLabViewState,
  persistStrategyCandidateLabViewState,
  restoreStrategyCandidateLabViewState,
} from "./strategy_candidate_lab_state.js";

import {
  STRATEGY_POOL_OPERATION_WORKFLOWS,
  formField,
  formValue,
  isRecord,
  nonEmptyText,
  stablePrimitiveText,
} from "./strategy_candidate_lab_contracts.js";

import {
  strategyCandidateLabResultsHtml,
} from "./strategy_candidate_lab_presenters.js";

import {
  STRATEGY_CANDIDATE_LAB_WORKFLOWS,
  collectStrategyCandidateLabRequest,
} from "./strategy_candidate_lab_requests.js";

import {
  candidateStabilityForm,
  crossCandidateBuildForm,
  crossCandidateRequestIsCurrent,
  crossCandidateSearchForm,
  crossRuleBuildForm,
  crossRuleRequestIsCurrent,
  crossRuleSearchForm,
  interactiveTreeForm,
  interactiveTreeFrontierGroupMaterializationForm,
  interactiveTreeFrontierGroupPointers,
  interactiveTreeFrontierMaterializationForm,
  interactiveTreeFrontierPointer,
  interactiveTreePointer,
  interactiveTreeRevisionPointer,
  interactiveTreeRevisionRequestIsCurrent,
  interactiveTreeSplitCandidatePointer,
  interactiveTreeSplitSearchForm,
  interactiveTreeSplitSearchRequestIsCurrent,
  renderStrategyPoolReorderOrder,
  selectContainsValue,
  setStrategyPoolActionValuePanelVisible,
  setStrategyPoolAddPanelVisible,
  strategyDslDeliveryForm,
  strategyImpactCubeForm,
  strategyLifecycleAdoptionForm,
  strategyMeasurementRequestIsCurrent,
  strategyPoolAddForm,
  strategyPoolAddRequestIsCurrent,
  strategyPoolApplyForm,
  strategyPoolApplyOptions,
  strategyPoolImpactForm,
  strategyPoolMaterializeForm,
  strategyPoolOperationPools,
  strategyPoolOperationRequestIsCurrent,
  strategyPoolOrderMatches,
  strategyPoolReorderForm,
  strategyPoolStabilityForm,
  strategyPoolStabilityRequestIsCurrent,
  strategyPoolValidationForm,
  strategyPoolValidationRequestIsCurrent,
  strategyProjectContextForm,
  strategyWorkbenchRequestIsCurrent,
  syncCandidateStabilityControls,
  syncCrossCandidateBuildControls,
  syncCrossCandidateSearchControls,
  syncCrossRuleBuildControls,
  syncCrossRuleSearchControls,
  syncInteractiveTreeFrontierGroupMaterializationControls,
  syncInteractiveTreeFrontierMaterializationControls,
  syncInteractiveTreeRevisionControls,
  syncInteractiveTreeSplitSearchControls,
  syncRefinementCandidateControls,
  syncRefinementForm,
  syncRefinementMode,
  syncSampleDesignV2StatusControls,
  syncScorecardBandingMode,
  syncScorecardCutoffControls,
  syncScorecardForms,
  syncStrategyDslDeliveryControls,
  syncStrategyImpactCubeControls,
  syncStrategyLifecycleAdoptionControls,
  syncStrategyLifecycleAdoptionEconomics,
  syncStrategyPoolAddControls,
  syncStrategyPoolAddPlacement,
  syncStrategyPoolApplyControls,
  syncStrategyPoolImpactControls,
  syncStrategyPoolMaterializeControls,
  syncStrategyPoolOperationForms,
  syncStrategyPoolRemoveEntryControls,
  syncStrategyPoolReorderControls,
  syncStrategyPoolSetActionControls,
  syncStrategyPoolStabilityControls,
  syncStrategyPoolValidationControls,
  syncStrategyProjectContextControls,
  syncVotingBuildControls,
  syncVotingForms,
  syncVotingSearchControls,
} from "./strategy_candidate_lab_projection.js";

export {
  STRATEGY_CANDIDATE_LAB_WORKFLOWS,
  collectStrategyCandidateLabRequest,
  strategyCandidateLabResultsHtml,
  syncSampleDesignV2StatusControls,
};

const WORKFLOW_LABELS = Object.freeze({
  strategy_project_context: "固化当前项目现状与历史材料",
  strategy_sample_design_v2: "创建双人群 SampleDesign V2",
  univariate_candidate_analysis: "启动单变量候选分析",
  univariate_candidate_refinement: "启动单变量候选细化",
  cross_matrix_analysis: "启动二维 Cross Matrix",
  cross_matrix_candidate_search: "搜索 Cross Matrix 字段组合",
  cross_matrix_candidate_build_from_search: "从搜索结果构建 Cross Matrix 候选",
  cross_rule_search: "搜索 2D/3D Cross 阈值规则",
  cross_rule_candidate_build_from_search: "构建指定 Cross 阈值规则候选",
  automatic_tree_candidate_build: "启动自动规则树",
  scorecard_model_score_evidence_build: "训练评分卡并生成模型评分证据",
  scorecard_band_build: "生成评分卡分档证据",
  scorecard_cutoff_selection: "记录评分卡 Cutoff 选择",
  candidate_monthly_stability: "测算候选逐月稳定性",
  strategy_pool_add_candidate: "把已物化候选加入 Strategy Pool",
  strategy_pool_compile: "编译预览当前 Strategy Pool",
  strategy_pool_remove_entry: "从当前 Strategy Pool 移除条目",
  strategy_pool_set_action: "修改当前 Strategy Pool 条目动作",
  strategy_pool_reorder: "完整重排当前 Strategy Pool",
  strategy_pool_apply: "应用当前 Strategy Pool",
  strategy_pool_validation: "执行 Strategy Pool 独立样本回放验证",
  strategy_pool_stability: "测算 Strategy Pool 稳定性",
  strategy_pool_impact: "测算 Strategy Pool 影响",
  strategy_impact_cube: "生成统一策略 ImpactCube",
  strategy_pool_materialize: "把当前 Strategy Pool 物化为草稿策略",
  strategy_lifecycle_adopt: "提交策略本地采纳确认",
  strategy_dsl_delivery: "生成策略等价代码交付包",
  strategy_report_bundle_v2: "形成策略迭代评审报告",
  voting_candidate_search: "搜索 Voting 组合",
  voting_candidate_build_from_search: "从搜索结果构建 Voting 候选",
  interactive_tree_split_search: "搜索交互树节点分裂候选",
  interactive_tree_auto_continuation: "从明确候选受控续建交互树",
  interactive_tree_revision: "创建不可变交互式树修订",
  interactive_tree_frontier_group_materialization: "物化交互树前沿 OR 分组",
  interactive_tree_frontier_materialization: "物化交互树前沿节点",
});

const BLOCKED_REASON_COPY = Object.freeze({
  active_plan: "当前已有策略计划执行中。完成或停止该计划后，才能启动新的 Candidate Lab 分析。",
  open_gate: "当前策略任务有待处理确认门。请先完成确认，再启动新的 Candidate Lab 分析。",
  loading: "正在核验 Candidate Lab 状态，完成前暂不能启动新分析。",
  submitting: "Candidate Lab 请求正在提交，请等待当前动作完成。",
});

function setFormError(form, message) {
  const target = form?.querySelector?.("[data-candidate-lab-form-error]");
  if (target) target.textContent = String(message || "");
}

function localBlockedReason(dependencies) {
  const value = dependencies.getBlockedReason?.();
  return ["active_plan", "open_gate"].includes(value) ? value : "";
}

function blockedReason(state, dependencies) {
  if (state.submitting) return "submitting";
  const local = localBlockedReason(dependencies);
  if (local) return local;
  if (state.loading && !state.payload) return "loading";
  if (state.payload?.can_start === false) {
    return nonEmptyText(state.payload.blocked_reason) || "active_plan";
  }
  return "";
}

function panelStatusText(state, dependencies) {
  if (state.error) return state.error;
  const reason = blockedReason(state, dependencies);
  if (reason) return BLOCKED_REASON_COPY[reason] || "当前 Candidate Lab 暂不可启动新分析。";
  if (state.loading) return "正在刷新受认证候选证据…";
  return "只展示平台已经登记并重新验真的候选；所有启动动作复用 Agent 的确定性治理链。";
}

function stateSnapshot(state) {
  return {
    taskId: state.taskId,
    payload: state.payload,
    loading: state.loading,
    submitting: state.submitting,
    error: state.error,
  };
}

function submissionClarificationText(result) {
  const direct = nonEmptyText(result?.message);
  if (direct) return direct;
  const messages = Array.isArray(result?.messages) ? result.messages : [];
  const assistant = [...messages].reverse().find((message) => (
    message?.role === "assistant" && nonEmptyText(message.content)
  ));
  if (assistant) return nonEmptyText(assistant.content);
  const code = nonEmptyText(result?.code);
  return code ? `平台需要补充信息（${code}）。` : "平台需要补充信息后才能启动该分析。";
}

function submissionWasAccepted(result) {
  return ["accepted", "ok", "plan_started"].includes(nonEmptyText(result?.status));
}

export function createStrategyCandidateLabController(dependencies = {}) {
  const $ = dependencies.$ || ((id) => document.getElementById(id));
  const fetchCandidateLab = dependencies.getStrategyCandidateLab || getStrategyCandidateLab;
  const submitCandidateLab = dependencies.submitStrategyCandidateLabRequest
    || submitStrategyCandidateLabRequest;
  let storage = dependencies.storage;
  if (storage === undefined) {
    try {
      storage = typeof localStorage === "undefined" ? null : localStorage;
    } catch (_) {
      storage = null;
    }
  }
  const state = {
    taskId: "",
    payload: null,
    loading: false,
    submitting: false,
    error: "",
  };
  let operation = 0;
  let boundRoot = null;
  let activeRefresh = null;
  let restoredTaskId = "";

  function selectedTask() {
    return dependencies.getSelectedTask?.() || null;
  }

  function selectedTaskId() {
    return nonEmptyText(dependencies.getSelectedTaskId?.());
  }

  function panel() {
    return $("strategyCandidateLabPanel");
  }

  function persistCurrentView(taskId = state.taskId) {
    const currentTaskId = nonEmptyText(taskId);
    if (!currentTaskId) return false;
    return persistStrategyCandidateLabViewState(
      currentTaskId,
      panel(),
      storage,
    );
  }

  function renderAvailability() {
    const root = panel();
    if (!root) return;
    const reason = blockedReason(state, dependencies);
    root.dataset.candidateLabBlockedReason = reason;
    const controls = root.querySelectorAll?.(
      "[data-candidate-lab-form] input, "
      + "[data-candidate-lab-form] select, "
      + "[data-candidate-lab-form] textarea, "
      + "[data-candidate-lab-form] button, "
      + "[data-candidate-lab-interactive-tree-prune], "
      + "[data-candidate-lab-interactive-tree-threshold], "
      + "[data-candidate-lab-interactive-tree-split-candidate], "
      + "[data-candidate-lab-interactive-tree-auto-continuation], "
      + "[data-candidate-lab-interactive-tree-frontier-materialize]",
    ) || [];
    for (const control of controls) {
      const refinementPanel = control.closest?.(
        "[data-candidate-lab-refinement-panel]",
      );
      const scorecardPanel = control.closest?.(
        "[data-candidate-lab-scorecard-banding-panel]",
      );
      const stabilityPanel = control.closest?.(
        "[data-candidate-lab-stability-panel]",
      );
      const poolActionValuePanel = control.closest?.(
        "[data-candidate-lab-pool-action-value-panel]",
      );
      const poolAddDefaultValuePanel = control.closest?.(
        "[data-candidate-lab-pool-add-default-value-panel]",
      );
      const poolAddActionValuePanel = control.closest?.(
        "[data-candidate-lab-pool-add-action-value-panel]",
      );
      const poolAddPlacementPanel = control.closest?.(
        "[data-candidate-lab-pool-add-placement-panel]",
      );
      const adoptionEconomicsPanel = control.closest?.(
        "[data-candidate-lab-adoption-economics]",
      );
      const adoptionComponent = control.closest?.(
        "[data-candidate-lab-adoption-component]",
      );
      const adoptionBinding = control.closest?.(
        "[data-candidate-lab-adoption-binding]",
      );
      const treeThresholdPanel = control.closest?.(
        "[data-candidate-lab-tree-threshold-panel]",
      );
      const hiddenByMode = Boolean(
        refinementPanel?.classList?.contains?.("hidden")
        || scorecardPanel?.classList?.contains?.("hidden")
        || stabilityPanel?.classList?.contains?.("hidden")
        || poolActionValuePanel?.classList?.contains?.("hidden")
        || poolAddDefaultValuePanel?.classList?.contains?.("hidden")
        || poolAddActionValuePanel?.classList?.contains?.("hidden")
        || poolAddPlacementPanel?.classList?.contains?.("hidden")
        || adoptionEconomicsPanel?.classList?.contains?.("hidden")
        || adoptionComponent?.classList?.contains?.("hidden")
        || adoptionBinding?.classList?.contains?.("hidden")
        || treeThresholdPanel?.classList?.contains?.("hidden"),
      );
      const locked = (
        control.dataset?.candidateLabPoolAddLocked === "1"
      );
      control.disabled = Boolean(reason) || hiddenByMode || locked;
    }
    const status = $("strategyCandidateLabStatus");
    if (status) {
      status.textContent = panelStatusText(state, dependencies);
      status.dataset.tone = state.error ? "error" : reason ? "blocked" : state.loading ? "loading" : "ready";
    }
    const retries = root.querySelectorAll?.("[data-candidate-lab-retry]") || [];
    for (const retry of retries) retry.disabled = state.loading;
  }

  function render() {
    const root = panel();
    if (!root) return;
    const task = selectedTask();
    const visible = Boolean(task && task.task_type === "strategy" && state.taskId);
    root.classList.toggle("hidden", !visible);
    root.setAttribute("aria-hidden", visible ? "false" : "true");
    if (!visible) return;
    const results = $("strategyCandidateLabResults");
    if (results) {
      if (state.payload) {
        results.innerHTML = strategyCandidateLabResultsHtml(state.payload);
      } else if (state.error) {
        results.innerHTML = [
          '<div class="candidate-lab-load-state" data-tone="error">',
          "<strong>Candidate Lab 读取失败</strong>",
          `<p>${escapeHtml(state.error)}</p>`,
          '<button type="button" class="button compact secondary" data-candidate-lab-retry="1">重新读取</button>',
          "</div>",
        ].join("");
      } else {
        results.innerHTML = [
          '<div class="candidate-lab-load-state">',
          "<strong>正在核验候选证据</strong>",
          "<p>平台正在读取 task-owned artifact 与当前 Strategy Pool。</p>",
          "</div>",
        ].join("");
      }
    }
    syncStrategyProjectContextControls(
      strategyProjectContextForm(root),
    );
    syncRefinementForm(root, state.payload);
    syncScorecardForms(root, state.payload);
    syncCandidateStabilityControls(
      candidateStabilityForm(root),
      state.payload,
    );
    syncStrategyPoolAddControls(
      strategyPoolAddForm(root),
      state.payload,
    );
    syncStrategyPoolValidationControls(
      strategyPoolValidationForm(root),
      state.payload,
    );
    syncStrategyPoolStabilityControls(
      strategyPoolStabilityForm(root),
      state.payload,
    );
    syncStrategyPoolImpactControls(
      strategyPoolImpactForm(root),
      state.payload,
    );
    syncStrategyImpactCubeControls(
      strategyImpactCubeForm(root),
      state.payload,
    );
    syncStrategyPoolMaterializeControls(
      strategyPoolMaterializeForm(root),
      state.payload,
    );
    syncStrategyLifecycleAdoptionControls(
      strategyLifecycleAdoptionForm(root),
      state.payload,
    );
    syncStrategyDslDeliveryControls(
      strategyDslDeliveryForm(root),
      state.payload,
    );
    syncStrategyPoolApplyControls(
      strategyPoolApplyForm(root),
      state.payload,
    );
    syncStrategyPoolOperationForms(root, state.payload);
    syncCrossCandidateSearchControls(
      crossCandidateSearchForm(root),
      state.payload,
    );
    syncCrossCandidateBuildControls(
      crossCandidateBuildForm(root),
      state.payload,
    );
    syncCrossRuleSearchControls(
      crossRuleSearchForm(root),
      state.payload,
    );
    syncCrossRuleBuildControls(
      crossRuleBuildForm(root),
      state.payload,
    );
    syncVotingForms(root, state.payload);
    syncInteractiveTreeSplitSearchControls(
      interactiveTreeSplitSearchForm(root),
      state.payload,
    );
    syncInteractiveTreeRevisionControls(
      interactiveTreeForm(root),
      state.payload,
    );
    syncInteractiveTreeFrontierGroupMaterializationControls(
      interactiveTreeFrontierGroupMaterializationForm(root),
      state.payload,
    );
    syncInteractiveTreeFrontierMaterializationControls(
      interactiveTreeFrontierMaterializationForm(root),
      state.payload,
    );
    if (state.payload && restoredTaskId !== state.taskId) {
      const snapshot = loadStrategyCandidateLabViewState(
        state.taskId,
        storage,
      );
      restoredTaskId = state.taskId;
      if (snapshot && restoreStrategyCandidateLabViewState(root, snapshot)) {
        // Re-run the projection sync once with restored parent selectors, then
        // restore dependent node/cell/cutpoint selectors against those options.
        render();
        restoreStrategyCandidateLabViewState(root, snapshot);
      }
    }
    renderAvailability();
  }

  function resetForms() {
    const root = panel();
    const forms = root?.querySelectorAll?.("[data-candidate-lab-form]") || [];
    for (const form of forms) {
      form.reset?.();
      setFormError(form, "");
    }
    syncStrategyProjectContextControls(
      strategyProjectContextForm(root),
    );
    syncRefinementForm(root, state.payload, { preserveBins: false });
    syncScorecardForms(root, state.payload);
    syncCandidateStabilityControls(
      candidateStabilityForm(root),
      state.payload,
    );
    syncStrategyPoolAddControls(
      strategyPoolAddForm(root),
      state.payload,
      { preserveSource: false },
    );
    syncStrategyPoolValidationControls(
      strategyPoolValidationForm(root),
      state.payload,
    );
    syncStrategyPoolStabilityControls(
      strategyPoolStabilityForm(root),
      state.payload,
    );
    syncStrategyPoolImpactControls(
      strategyPoolImpactForm(root),
      state.payload,
    );
    syncStrategyImpactCubeControls(
      strategyImpactCubeForm(root),
      state.payload,
    );
    syncStrategyPoolMaterializeControls(
      strategyPoolMaterializeForm(root),
      state.payload,
    );
    syncStrategyLifecycleAdoptionControls(
      strategyLifecycleAdoptionForm(root),
      state.payload,
    );
    syncStrategyDslDeliveryControls(
      strategyDslDeliveryForm(root),
      state.payload,
    );
    syncStrategyPoolApplyControls(
      strategyPoolApplyForm(root),
      state.payload,
    );
    syncStrategyPoolOperationForms(root, state.payload);
    syncCrossCandidateSearchControls(
      crossCandidateSearchForm(root),
      state.payload,
    );
    syncCrossCandidateBuildControls(
      crossCandidateBuildForm(root),
      state.payload,
      { preservePair: false },
    );
    syncCrossRuleSearchControls(
      crossRuleSearchForm(root),
      state.payload,
    );
    syncCrossRuleBuildControls(
      crossRuleBuildForm(root),
      state.payload,
      { preserveRule: false },
    );
    syncVotingForms(root, state.payload);
    syncInteractiveTreeSplitSearchControls(
      interactiveTreeSplitSearchForm(root),
      state.payload,
      { preserveNode: false },
    );
    syncInteractiveTreeRevisionControls(
      interactiveTreeForm(root),
      state.payload,
      { preserveNode: false },
    );
    syncInteractiveTreeFrontierGroupMaterializationControls(
      interactiveTreeFrontierGroupMaterializationForm(root),
      state.payload,
      { preserveNodes: false },
    );
    syncInteractiveTreeFrontierMaterializationControls(
      interactiveTreeFrontierMaterializationForm(root),
      state.payload,
      { preserveNode: false },
    );
  }

  function clear() {
    persistCurrentView();
    activeRefresh?.controller?.abort?.();
    activeRefresh = null;
    operation += 1;
    state.taskId = "";
    state.payload = null;
    state.loading = false;
    state.submitting = false;
    state.error = "";
    restoredTaskId = "";
    render();
    return stateSnapshot(state);
  }

  async function refresh(taskId = selectedTaskId(), { silent = false } = {}) {
    const requestedTaskId = nonEmptyText(taskId);
    if (
      !requestedTaskId
      || selectedTask()?.task_type !== "strategy"
      || selectedTaskId() !== requestedTaskId
    ) {
      return stateSnapshot(state);
    }
    if (activeRefresh?.taskId === requestedTaskId) {
      return activeRefresh.promise;
    }
    if (activeRefresh) {
      activeRefresh.controller?.abort?.();
      activeRefresh = null;
    }
    if (state.taskId !== requestedTaskId) {
      persistCurrentView();
      state.taskId = requestedTaskId;
      state.payload = null;
      state.error = "";
      restoredTaskId = "";
      resetForms();
    }
    const refreshOperation = ++operation;
    state.loading = true;
    if (!silent) state.error = "";
    render();
    const controller = typeof AbortController === "undefined"
      ? null
      : new AbortController();
    const refreshToken = {
      taskId: requestedTaskId,
      controller,
      promise: null,
    };
    const promise = (async () => {
      try {
        const payload = await fetchCandidateLab(
          requestedTaskId,
          controller ? { signal: controller.signal } : {},
        );
        if (refreshOperation !== operation || selectedTaskId() !== requestedTaskId) {
          return stateSnapshot(state);
        }
        if (!isRecord(payload) || payload.task_id !== requestedTaskId) {
          throw new Error("Candidate Lab 响应不属于当前任务。");
        }
        state.payload = payload;
        state.error = "";
        state.loading = false;
        render();
      } catch (error) {
        if (refreshOperation !== operation || selectedTaskId() !== requestedTaskId) {
          return stateSnapshot(state);
        }
        state.loading = false;
        if (controller?.signal?.aborted || error?.name === "AbortError") {
          render();
          return stateSnapshot(state);
        }
        state.error = error?.message || "Candidate Lab 读取失败。";
        render();
      } finally {
        if (activeRefresh === refreshToken) activeRefresh = null;
      }
      return stateSnapshot(state);
    })();
    refreshToken.promise = promise;
    activeRefresh = refreshToken;
    return promise;
  }

  async function selectTask(task) {
    const taskId = nonEmptyText(task?.id);
    if (!taskId || task?.task_type !== "strategy") {
      return clear();
    }
    if (state.taskId !== taskId) {
      persistCurrentView();
      activeRefresh?.controller?.abort?.();
      activeRefresh = null;
      operation += 1;
      state.taskId = taskId;
      state.payload = null;
      state.loading = true;
      state.submitting = false;
      state.error = "";
      restoredTaskId = "";
      resetForms();
      render();
    }
    return refresh(taskId);
  }

  async function submit(form) {
    const taskId = selectedTaskId();
    if (
      !taskId
      || selectedTask()?.task_type !== "strategy"
      || state.taskId !== taskId
    ) {
      const message = "缺少当前策略任务，请重新选择任务后再试。";
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      return null;
    }
    const reason = blockedReason(state, dependencies);
    if (reason) {
      const message = BLOCKED_REASON_COPY[reason] || "当前 Candidate Lab 暂不可启动新分析。";
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      renderAvailability();
      return null;
    }

    let strategyRequest;
    try {
      strategyRequest = collectStrategyCandidateLabRequest(form);
      if (!strategyPoolAddRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选候选来源或当前 Pool 默认动作已过期、不完整或不属于受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyPoolOperationRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "当前 Strategy Pool 类型或 Entry 集合已过期、不完整或不属于受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyPoolValidationRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选 Strategy Pool 已过期、为空或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyPoolStabilityRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选 Strategy Pool 已过期、为空或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyMeasurementRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "影响测算所选 Strategy Pool 已过期、为空或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyWorkbenchRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选 Pool 或策略版本已过期、不完整或不属于当前任务受认证投影，请刷新 Strategy Workbench 后重选。",
        );
      }
      if (!crossCandidateRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选单变量字段、Cross 搜索或 Pair 已过期或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!crossRuleRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选单变量字段、Cross 阈值搜索或规则已过期或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow === "strategy_pool_apply"
        && !strategyPoolApplyOptions(state.payload).some(
          (pool) => (
            pool.strategyType === strategyRequest.workflow_inputs.strategy_type
          ),
        )
      ) {
        throw new Error(
          "所选 Strategy Pool 已过期、为空或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow === "interactive_tree_split_search"
        && !interactiveTreeSplitSearchRequestIsCurrent(
          state.payload,
          strategyRequest.workflow_inputs,
        )
      ) {
        throw new Error(
          "树节点搜索来源、可见节点或认证特征全集已过期，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow === "interactive_tree_revision"
        && !interactiveTreeRevisionRequestIsCurrent(
          state.payload,
          strategyRequest.workflow_inputs,
        )
      ) {
        throw new Error(
          "交互式树修订指针或当前阈值已过期，或不属于当前任务的受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow === "interactive_tree_auto_continuation"
        && (
          !interactiveTreeSplitCandidatePointer(
            state.payload,
            strategyRequest.workflow_inputs.search_id,
            strategyRequest.workflow_inputs.candidate_id,
          )
          || interactiveTreeSplitCandidatePointer(
            state.payload,
            strategyRequest.workflow_inputs.search_id,
            strategyRequest.workflow_inputs.candidate_id,
          )?.search?.source_node?.is_frontier !== true
        )
      ) {
        throw new Error(
          "自动续建的搜索或候选已过期，或不再指向受认证 frontier，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow
          === "interactive_tree_frontier_group_materialization"
        && interactiveTreeFrontierGroupPointers(
          state.payload,
          strategyRequest.workflow_inputs.revision_id,
          strategyRequest.workflow_inputs.source_node_ids,
        ).length !== strategyRequest.workflow_inputs.source_node_ids.length
      ) {
        throw new Error(
          "交互树前沿 OR 分组指针已过期或不属于当前任务的受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow
          === "interactive_tree_frontier_materialization"
        && !interactiveTreeFrontierPointer(
          state.payload,
          strategyRequest.workflow_inputs.revision_id,
          strategyRequest.workflow_inputs.source_node_id,
        )
      ) {
        throw new Error(
          "交互树前沿指针已过期或不属于当前任务的受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
    } catch (error) {
      const message = error?.message || "Candidate Lab 表单输入无效。";
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      return null;
    }

    const workflow = strategyRequest.workflow;
    const content = strategyRequest.request_kind === "strategy_lifecycle"
      ? WORKFLOW_LABELS.strategy_lifecycle_adopt
      : WORKFLOW_LABELS[workflow] || "从 Candidate Lab 启动策略分析";
    setFormError(form, "");
    state.submitting = true;
    renderAvailability();
    dependencies.setActionStatus?.(`${content}…`, "busy");
    const requestTaskId = taskId;
    try {
      const requestPromise = submitCandidateLab(requestTaskId, strategyRequest, content);
      const pollPromise = Promise.resolve(
        dependencies.pollAgentMessagesUntilSettled?.(
          requestTaskId,
          requestPromise,
          { preserveOptimistic: false },
        ),
      ).catch(() => {});
      const result = await requestPromise;
      await pollPromise;
      if (selectedTaskId() !== requestTaskId) return result;
      if (Array.isArray(result?.messages)) {
        dependencies.setAgentMessages?.(result.messages);
      }
      dependencies.renderAgentConversation?.();
      dependencies.resetPlanFetchThrottle?.(requestTaskId);
      dependencies.renderWorkflowStepper?.({ force: true });
      if (result?.status === "clarification_required") {
        state.submitting = false;
        const message = submissionClarificationText(result);
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "info");
        renderAvailability();
        return result;
      }
      if (!submissionWasAccepted(result)) {
        throw new Error(
          submissionClarificationText(result)
          || `${content}未被平台接受。`,
        );
      }
      await dependencies.refreshAgentMessages?.(requestTaskId);
      await dependencies.settleCandidateLabSubmission?.(requestTaskId);
      if (selectedTaskId() !== requestTaskId) return result;
      if (
        strategyRequest.request_kind === "strategy_lifecycle"
        || strategyRequest.workflow === "cross_matrix_candidate_search"
        || strategyRequest.workflow
          === "cross_matrix_candidate_build_from_search"
        || strategyRequest.workflow === "strategy_pool_apply"
        || strategyRequest.workflow === "strategy_pool_stability"
        || strategyRequest.workflow === "strategy_pool_impact"
        || strategyRequest.workflow === "strategy_impact_cube"
        || strategyRequest.workflow === "strategy_project_context"
        || strategyRequest.workflow === "strategy_sample_design_v2"
        || strategyRequest.workflow === "strategy_pool_materialize"
        || strategyRequest.workflow === "strategy_dsl_delivery"
        || strategyRequest.workflow === "strategy_report_bundle_v2"
        || strategyRequest.workflow === "strategy_pool_add_candidate"
        || STRATEGY_POOL_OPERATION_WORKFLOWS.includes(strategyRequest.workflow)
      ) {
        await refresh(requestTaskId, { silent: true });
        if (selectedTaskId() !== requestTaskId) return result;
      }
      state.submitting = false;
      dependencies.setActionStatus?.(`${content}已提交。`, "success");
      return result;
    } catch (error) {
      if (selectedTaskId() !== requestTaskId) return null;
      state.submitting = false;
      const message = error?.message || `${content}失败。`;
      // Do not reset or re-render the static form: every operator input stays
      // available for correction and retry after a failed request.
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      renderAvailability();
      return null;
    } finally {
      if (selectedTaskId() === requestTaskId) {
        state.submitting = false;
        dependencies.resetPlanFetchThrottle?.(requestTaskId);
        dependencies.renderWorkflowStepper?.({ force: true });
        renderAvailability();
      }
    }
  }

  function handleSubmit(event) {
    const form = event.target?.closest?.("[data-candidate-lab-form]");
    if (!form) return false;
    event.preventDefault?.();
    void submit(form);
    return true;
  }

  function handleClick(event) {
    const reorderMove = event.target?.closest?.(
      "[data-candidate-lab-pool-reorder-move]",
    );
    if (reorderMove) {
      event.preventDefault?.();
      const form = strategyPoolReorderForm(panel());
      const reason = blockedReason(state, dependencies);
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可调整 Pool 顺序。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      try {
        const request = collectStrategyCandidateLabRequest(form);
        const strategyType = request.workflow_inputs.strategy_type;
        const orderedIds = request.workflow_inputs.ordered_ids;
        const pool = strategyPoolOperationPools(state.payload).find(
          (item) => item.strategyType === strategyType,
        );
        if (!strategyPoolOrderMatches(pool, orderedIds)) {
          throw new Error(
            "当前 Pool Entry 集合已过期，请刷新 Candidate Lab 后重试。",
          );
        }
        const orderSelect = formField(form, "pool_reorder_ordered_ids");
        const selected = Array.from(orderSelect?.selectedOptions || [])[0];
        const selectedEntryId = nonEmptyText(selected?.value);
        if (
          !selected
          || selected.dataset?.candidateLabProjection !== "1"
          || nonEmptyText(selected.dataset?.strategyType) !== strategyType
          || nonEmptyText(selected.dataset?.entryId) !== selectedEntryId
        ) {
          throw new Error("请先选择一个当前受认证 Pool Entry 再调整顺序。");
        }
        const direction = nonEmptyText(
          reorderMove.dataset?.candidateLabPoolReorderMove,
        );
        if (!["up", "down"].includes(direction)) {
          throw new Error("Pool Entry 调整方向无效。");
        }
        const selectedIndex = orderedIds.indexOf(selectedEntryId);
        const targetIndex = direction === "up"
          ? selectedIndex - 1
          : selectedIndex + 1;
        if (targetIndex >= 0 && targetIndex < orderedIds.length) {
          [orderedIds[selectedIndex], orderedIds[targetIndex]] = [
            orderedIds[targetIndex],
            orderedIds[selectedIndex],
          ];
          renderStrategyPoolReorderOrder(
            form,
            pool,
            orderedIds,
            selectedEntryId,
          );
        }
        setFormError(form, "");
        renderAvailability();
      } catch (error) {
        const message = error?.message || "无法调整当前 Pool Entry 顺序。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
      }
      return true;
    }
    const materialize = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-frontier-materialize]",
    );
    if (materialize) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = interactiveTreeFrontierMaterializationForm(panel());
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const revisionId = nonEmptyText(materialize.dataset?.revisionId);
      const sourceNodeId = nonEmptyText(materialize.dataset?.sourceNodeId);
      const pointer = interactiveTreeFrontierPointer(
        state.payload,
        revisionId,
        sourceNodeId,
      );
      if (!pointer || !form) {
        const message = "该前沿指针不属于当前任务的受认证 Candidate Lab 投影。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      syncInteractiveTreeFrontierMaterializationControls(
        form,
        state.payload,
        {
          requestedRevisionId: revisionId,
          requestedSourceNodeId: sourceNodeId,
          preserveNode: false,
        },
      );
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已带入受认证 revision 与前沿节点；确认后只物化该节点，不会自动入池。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const continuationCandidate = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-auto-continuation]",
    );
    if (continuationCandidate) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = panel()?.querySelector?.(
        '[data-candidate-lab-workflow="interactive_tree_auto_continuation"]',
      );
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const searchId = nonEmptyText(
        continuationCandidate.dataset?.searchId,
      );
      const candidateId = nonEmptyText(
        continuationCandidate.dataset?.candidateId,
      );
      const pointer = interactiveTreeSplitCandidatePointer(
        state.payload,
        searchId,
        candidateId,
      );
      if (
        !form
        || !pointer
        || pointer.search?.source_node?.is_frontier !== true
      ) {
        const message = "该候选不属于当前受认证 frontier 搜索，请刷新后重试。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const searchField = formField(
        form,
        "interactive_tree_continuation_search_id",
      );
      const candidateField = formField(
        form,
        "interactive_tree_continuation_candidate_id",
      );
      if (searchField) searchField.value = searchId;
      if (candidateField) candidateField.value = candidateId;
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已带入人工明确选择的 eligible 候选；请检查全部硬预算后再提交续建。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const splitCandidate = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-split-candidate]",
    );
    if (splitCandidate) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = interactiveTreeForm(panel());
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const searchId = nonEmptyText(splitCandidate.dataset?.searchId);
      const candidateId = nonEmptyText(splitCandidate.dataset?.candidateId);
      const pointer = interactiveTreeSplitCandidatePointer(
        state.payload,
        searchId,
        candidateId,
      );
      const sourceTreeId = nonEmptyText(
        splitCandidate.dataset?.sourceTreeId,
      );
      const nodeId = nonEmptyText(splitCandidate.dataset?.nodeId);
      const feature = nonEmptyText(splitCandidate.dataset?.feature);
      const threshold = Number(splitCandidate.dataset?.threshold);
      if (
        !pointer
        || !form
        || pointer.search.source_tree_id !== sourceTreeId
        || pointer.search.node_id !== nodeId
        || pointer.candidate.feature !== feature
        || Number(pointer.candidate.threshold) !== threshold
      ) {
        const message = "该分裂候选不属于当前任务的受认证搜索证据。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const currentFeature = nonEmptyText(
        pointer.search?.source_node?.feature,
      );
      const operation = feature === currentFeature
        ? "adjust_split_threshold"
        : "replace_split_feature";
      const revisionPointer = interactiveTreeRevisionPointer(
        state.payload,
        sourceTreeId,
        nodeId,
        operation,
      );
      if (!revisionPointer) {
        const message = "该搜索来源节点当前已不可编辑，请刷新树投影后重新搜索。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      syncInteractiveTreeRevisionControls(
        form,
        state.payload,
        {
          requestedOperation: operation,
          requestedSourceTreeId: sourceTreeId,
          requestedNodeId: nodeId,
          preserveNode: false,
        },
      );
      const thresholdField = formField(form, "interactive_tree_threshold");
      if (thresholdField) thresholdField.value = stablePrimitiveText(threshold);
      if (operation === "replace_split_feature") {
        const featureField = formField(form, "interactive_tree_feature");
        if (!selectContainsValue(featureField, feature)) {
          const message = "候选字段已不在当前来源树的认证特征全集中。";
          setFormError(form, message);
          dependencies.setActionStatus?.(message, "error");
          return true;
        }
        featureField.value = feature;
      }
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已回填受认证候选字段与阈值；尚未修改树，请检查理由并手动确认创建不可变 revision。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const thresholdAdjustment = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-threshold]",
    );
    if (thresholdAdjustment) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = interactiveTreeForm(panel());
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const sourceTreeId = nonEmptyText(
        thresholdAdjustment.dataset?.sourceTreeId,
      );
      const nodeId = nonEmptyText(thresholdAdjustment.dataset?.nodeId);
      const pointer = interactiveTreeRevisionPointer(
        state.payload,
        sourceTreeId,
        nodeId,
        "adjust_split_threshold",
      );
      if (
        !pointer
        || !form
        || nonEmptyText(thresholdAdjustment.dataset?.feature)
          !== pointer.feature
        || nonEmptyText(thresholdAdjustment.dataset?.currentThreshold)
          !== stablePrimitiveText(pointer.current_threshold)
      ) {
        const message = "该阈值调整指针不属于当前任务的受认证 Candidate Lab 投影。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      syncInteractiveTreeRevisionControls(
        form,
        state.payload,
        {
          requestedOperation: "adjust_split_threshold",
          requestedSourceTreeId: sourceTreeId,
          requestedNodeId: nodeId,
          preserveNode: false,
        },
      );
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已带入受认证分支、字段与当前阈值；请明确填写不同的新阈值后创建不可变 revision。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const prune = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-prune]",
    );
    if (prune) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = interactiveTreeForm(panel());
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const sourceTreeId = nonEmptyText(prune.dataset?.sourceTreeId);
      const nodeId = nonEmptyText(prune.dataset?.nodeId);
      const pointer = interactiveTreePointer(
        state.payload,
        sourceTreeId,
        nodeId,
      );
      if (!pointer || !form) {
        const message = "该剪枝指针不属于当前任务的受认证 Candidate Lab 投影。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      syncInteractiveTreeRevisionControls(
        form,
        state.payload,
        {
          requestedOperation: "prune_subtree",
          requestedSourceTreeId: sourceTreeId,
          requestedNodeId: nodeId,
          preserveNode: false,
        },
      );
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已带入受认证分支和节点；填写可选理由后确认创建不可变 revision。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const retry = event.target?.closest?.("[data-candidate-lab-retry]");
    if (!retry) return false;
    event.preventDefault?.();
    void refresh();
    return true;
  }

  function handleChange(event) {
    const field = event.target?.closest?.("[data-candidate-lab-field]");
    if (!field) return false;
    const fieldName = field.dataset?.candidateLabField;
    const sampleDesign = field.closest?.(
      '[data-candidate-lab-workflow="strategy_sample_design_v2"]',
    );
    if (
      sampleDesign
      && syncSampleDesignV2StatusControls(sampleDesign, fieldName)
    ) {
      setFormError(sampleDesign, "");
      renderAvailability();
      return true;
    }
    const adoption = field.closest?.(
      '[data-candidate-lab-workflow="strategy_lifecycle_adopt"]',
    );
    if (adoption && fieldName === "lifecycle_adopt_strategy_id") {
      syncStrategyLifecycleAdoptionControls(adoption, state.payload);
      renderAvailability();
      return true;
    }
    if (
      adoption
      && /^lifecycle_adopt_[a-z_]+_mode$/.test(fieldName || "")
    ) {
      const selected = Array.from(
        formField(adoption, "lifecycle_adopt_strategy_id")
          ?.selectedOptions || [],
      )[0] || null;
      syncStrategyLifecycleAdoptionEconomics(
        adoption,
        nonEmptyText(selected?.dataset?.strategyType),
      );
      renderAvailability();
      return true;
    }
    const crossSearch = field.closest?.(
      '[data-candidate-lab-workflow="cross_matrix_candidate_search"]',
    );
    if (
      crossSearch
      && (
        fieldName === "cross_search_features"
        || fieldName === "cross_search_max_pairs"
      )
    ) {
      syncCrossCandidateSearchControls(crossSearch, state.payload);
      renderAvailability();
      return true;
    }
    const crossBuild = field.closest?.(
      '[data-candidate-lab-workflow="cross_matrix_candidate_build_from_search"]',
    );
    if (crossBuild && fieldName === "cross_build_search_id") {
      syncCrossCandidateBuildControls(
        crossBuild,
        state.payload,
        { preservePair: false },
      );
      renderAvailability();
      return true;
    }
    const crossRuleSearch = field.closest?.(
      '[data-candidate-lab-workflow="cross_rule_search"]',
    );
    if (
      crossRuleSearch
      && (
        fieldName === "cross_rule_features"
        || fieldName === "cross_rule_dimension"
        || fieldName === "cross_rule_max_trials"
      )
    ) {
      syncCrossRuleSearchControls(crossRuleSearch, state.payload);
      renderAvailability();
      return true;
    }
    const crossRuleBuild = field.closest?.(
      '[data-candidate-lab-workflow="cross_rule_candidate_build_from_search"]',
    );
    if (
      crossRuleBuild
      && fieldName === "cross_rule_build_search_id"
    ) {
      syncCrossRuleBuildControls(
        crossRuleBuild,
        state.payload,
        { preserveRule: false },
      );
      renderAvailability();
      return true;
    }
    const refinement = field.closest?.(
      '[data-candidate-lab-workflow="univariate_candidate_refinement"]',
    );
    if (refinement) {
      if (fieldName === "refinement_mode") {
        syncRefinementMode(refinement);
      } else if (
        fieldName === "source_candidate_id"
        || fieldName === "source_feature_method"
      ) {
        syncRefinementCandidateControls(
          refinement,
          state.payload,
          { preserveBins: false },
        );
      } else {
        return false;
      }
      renderAvailability();
      return true;
    }
    const scorecardBand = field.closest?.(
      '[data-candidate-lab-workflow="scorecard_band_build"]',
    );
    if (scorecardBand && fieldName === "scorecard_banding_mode") {
      syncScorecardBandingMode(scorecardBand);
      renderAvailability();
      return true;
    }
    const scorecardSelection = field.closest?.(
      '[data-candidate-lab-workflow="scorecard_cutoff_selection"]',
    );
    if (scorecardSelection && fieldName === "scorecard_asset_id") {
      syncScorecardCutoffControls(
        scorecardSelection,
        state.payload,
        { preserveCutoff: false },
      );
      renderAvailability();
      return true;
    }
    const candidateStability = field.closest?.(
      '[data-candidate-lab-workflow="candidate_monthly_stability"]',
    );
    if (
      candidateStability
      && fieldName === "stability_source_mode"
    ) {
      syncCandidateStabilityControls(
        candidateStability,
        state.payload,
      );
      renderAvailability();
      return true;
    }
    const poolAdd = field.closest?.(
      '[data-candidate-lab-workflow="strategy_pool_add_candidate"]',
    );
    if (poolAdd && fieldName === "pool_add_strategy_type") {
      syncStrategyPoolAddControls(
        poolAdd,
        state.payload,
        { preserveSource: false },
      );
      renderAvailability();
      return true;
    }
    if (poolAdd && fieldName === "pool_add_source_id") {
      syncStrategyPoolAddPlacement(poolAdd);
      renderAvailability();
      return true;
    }
    if (
      poolAdd
      && (
        fieldName === "pool_add_default_action_type"
        || fieldName === "pool_add_action_type"
      )
    ) {
      const isDefault = fieldName === "pool_add_default_action_type";
      const actionType = formValue(poolAdd, fieldName);
      setStrategyPoolAddPanelVisible(
        poolAdd,
        isDefault
          ? "[data-candidate-lab-pool-add-default-value-panel]"
          : "[data-candidate-lab-pool-add-action-value-panel]",
        ["limit", "pricing", "segment"].includes(actionType),
      );
      renderAvailability();
      return true;
    }
    const poolRemove = field.closest?.(
      '[data-candidate-lab-workflow="strategy_pool_remove_entry"]',
    );
    if (poolRemove && fieldName === "pool_remove_strategy_type") {
      syncStrategyPoolRemoveEntryControls(
        poolRemove,
        strategyPoolOperationPools(state.payload),
        { preserveEntry: false },
      );
      renderAvailability();
      return true;
    }
    const poolAction = field.closest?.(
      '[data-candidate-lab-workflow="strategy_pool_set_action"]',
    );
    if (poolAction && fieldName === "pool_action_strategy_type") {
      syncStrategyPoolSetActionControls(
        poolAction,
        strategyPoolOperationPools(state.payload),
        { preserveEntry: false, preserveAction: false },
      );
      renderAvailability();
      return true;
    }
    if (poolAction && fieldName === "pool_action_type") {
      setStrategyPoolActionValuePanelVisible(
        poolAction,
        ["limit", "pricing", "segment"].includes(formValue(
          poolAction,
          "pool_action_type",
        )),
      );
      renderAvailability();
      return true;
    }
    const poolReorder = field.closest?.(
      '[data-candidate-lab-workflow="strategy_pool_reorder"]',
    );
    if (poolReorder && fieldName === "pool_reorder_strategy_type") {
      syncStrategyPoolReorderControls(
        poolReorder,
        strategyPoolOperationPools(state.payload),
      );
      renderAvailability();
      return true;
    }
    const votingSearch = field.closest?.(
      '[data-candidate-lab-workflow="voting_candidate_search"]',
    );
    if (votingSearch && fieldName === "voting_strategy_type") {
      syncVotingSearchControls(votingSearch, state.payload);
      renderAvailability();
      return true;
    }
    const votingBuild = field.closest?.(
      '[data-candidate-lab-workflow="voting_candidate_build_from_search"]',
    );
    if (votingBuild && fieldName === "voting_search_id") {
      syncVotingBuildControls(
        votingBuild,
        state.payload,
        { preserveCombo: false },
      );
      renderAvailability();
      return true;
    }
    const interactiveTree = field.closest?.(
      '[data-candidate-lab-workflow="interactive_tree_revision"]',
    );
    const interactiveTreeSearch = field.closest?.(
      '[data-candidate-lab-workflow="interactive_tree_split_search"]',
    );
    if (
      interactiveTreeSearch
      && [
        "interactive_tree_search_source_id",
        "interactive_tree_search_node_id",
        "interactive_tree_search_mode",
      ].includes(fieldName)
    ) {
      syncInteractiveTreeSplitSearchControls(
        interactiveTreeSearch,
        state.payload,
        {
          preserveNode: fieldName === "interactive_tree_search_node_id",
        },
      );
      renderAvailability();
      return true;
    }
    if (
      interactiveTree
      && [
        "interactive_tree_operation",
        "interactive_tree_source_id",
        "interactive_tree_node_id",
      ].includes(fieldName)
    ) {
      syncInteractiveTreeRevisionControls(
        interactiveTree,
        state.payload,
        {
          preserveNode: fieldName === "interactive_tree_node_id",
        },
      );
      renderAvailability();
      return true;
    }
    const interactiveTreeFrontier = field.closest?.(
      '[data-candidate-lab-workflow="interactive_tree_frontier_materialization"]',
    );
    if (
      interactiveTreeFrontier
      && fieldName === "interactive_tree_frontier_revision_id"
    ) {
      syncInteractiveTreeFrontierMaterializationControls(
        interactiveTreeFrontier,
        state.payload,
        { preserveNode: false },
      );
      renderAvailability();
      return true;
    }
    const interactiveTreeFrontierGroup = field.closest?.(
      '[data-candidate-lab-workflow="interactive_tree_frontier_group_materialization"]',
    );
    if (
      interactiveTreeFrontierGroup
      && fieldName === "interactive_tree_frontier_group_revision_id"
    ) {
      syncInteractiveTreeFrontierGroupMaterializationControls(
        interactiveTreeFrontierGroup,
        state.payload,
        { preserveNodes: false },
      );
      renderAvailability();
      return true;
    }
    return false;
  }

  function handlePersistedViewMutation() {
    persistCurrentView();
  }

  function bind(root = document) {
    if (!root || boundRoot === root || typeof root.addEventListener !== "function") return;
    if (boundRoot) unbind();
    boundRoot = root;
    root.addEventListener("submit", handleSubmit);
    root.addEventListener("click", handleClick);
    root.addEventListener("change", handleChange);
    root.addEventListener("change", handlePersistedViewMutation);
    root.addEventListener("toggle", handlePersistedViewMutation, true);
  }

  function unbind() {
    if (!boundRoot) return;
    boundRoot.removeEventListener?.("submit", handleSubmit);
    boundRoot.removeEventListener?.("click", handleClick);
    boundRoot.removeEventListener?.("change", handleChange);
    boundRoot.removeEventListener?.("change", handlePersistedViewMutation);
    boundRoot.removeEventListener?.("toggle", handlePersistedViewMutation, true);
    boundRoot = null;
  }

  return {
    bind,
    clear,
    getState: () => stateSnapshot(state),
    handleChange,
    handleClick,
    handleSubmit,
    refresh,
    render,
    renderAvailability,
    selectTask,
    submit,
    unbind,
  };
}
