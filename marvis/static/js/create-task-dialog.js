import { api, localToken } from "./api.js";
import { defaultTaskType, taskTypeDefinitions } from "./task-types.js";
import { formatDateInput } from "./ui-utils.js";
import { materialUploadSelectionText } from "./dialogs.js";
import {
  MAX_VALIDATION_BATCH_ROWS,
  buildValidationBatchUploadFormData,
  classifyValidationBatchFiles,
  submitValidationBatchDraft,
} from "./validation-batch-create.js";
import { formatValidationBatchTaskName } from "./validation-batch.js";

// UX-12: below this total upload size, plain "正在上传材料..." is enough — a
// percentage readout for a few small files would jump straight to 100% and
// add noise, not signal. Large credit-sample/feature-table files (the actual
// case this is for) clear this easily.
const MATERIAL_UPLOAD_PERCENT_THRESHOLD_BYTES = 10 * 1024 * 1024;

// Keep these editable form seeds aligned with validation_report_copy.py.
export function validationNarrativeDefaults(modelName) {
  const name = String(modelName || "").trim() || "本模型";
  const displayName = name.endsWith("模型") ? name.slice(0, -2) : name;
  const hasT = /t卡/i.test(name);
  const hasA = /a卡/i.test(name);
  const kind = hasT !== hasA ? (hasT ? "T" : "A") : "";
  const boundary = /[AT]卡|MOB\s*\d+/i.exec(name);
  let channel = boundary ? name.slice(0, boundary.index).replace(/^[ _\-/：:]+|[ _\-/：:]+$/g, "") : "";
  if (["自营", "自营通用"].includes(channel)) channel = "自营通用";
  else channel = channel.replace(/^自营/, "").replace(/^[ _\-/：:]+|[ _\-/：:]+$/g, "");
  const cohort = channel ? `${channel}${kind ? `${kind}卡` : ""}` : "xx";
  const stage = kind === "T" ? "支用" : "授信";
  const audience = kind && cohort !== "xx" ? `${cohort}用户` : "xx用户";
  const window = /MOB\s*([36])/i.exec(name);
  return {
    "TEXT:model_overview": `为了更好的对${audience}进行${stage}环节风险管控，现开发${displayName}模型，对${kind ? cohort : "xx"}客群做前置风险拦截，从${stage}申请阶段做好风险防范。`,
    "TEXT:model_scope": `本模型适用于${cohort}渠道用户。`,
    "TEXT:bad_sample_definition": window ? `MOB${window[1]} 逾期 >= 30 天` : "xx逾期 >= xx天",
    "TEXT:good_sample_definition": window ? `MOB${window[1]} 未逾期` : "xx未逾期",
    "TEXT:sample_audience": `申请${stage}的用户`,
  };
}

export function updateAutoReportValue(input, nextValue) {
  if (!input) return;
  if (!input.value.trim() || input.value === input.dataset.createReportSeed) {
    input.value = nextValue;
  }
  input.dataset.createReportSeed = nextValue;
}

export function validationExtraModelCardMarkup(rowId, ordinal) {
  const pathTabId = `${rowId}-path-tab`;
  const uploadTabId = `${rowId}-upload-tab`;
  const pathPanelId = `${rowId}-path-panel`;
  const uploadPanelId = `${rowId}-upload-panel`;
  const uploadStatusId = `${rowId}-upload-status`;
  return [
    `<article class="task-form-section validation-create-model-card" data-validation-extra-row-id="${rowId}">`,
    `<header class="validation-create-model-head">`,
    `<h3>模型 ${ordinal}</h3>`,
    `<button type="button" class="button compact secondary" data-remove-validation-extra-row="${rowId}">移除</button>`,
    `</header>`,
    `<label><span>模型名称</span>`,
    `<input data-extra-model-field="name" placeholder="例如：贷前评分卡 MOB3 v202604" autocomplete="off" /></label>`,
    `<label class="wide-field"><span>模型概述</span>`,
    `<textarea data-extra-model-field="overview"></textarea></label>`,
    `<label class="wide-field"><span>适用范围</span>`,
    `<input data-extra-model-field="scope" autocomplete="off" /></label>`,
    `<label><span>坏样本定义</span>`,
    `<input data-extra-model-field="bad-sample" autocomplete="off" /></label>`,
    `<label><span>好样本定义</span>`,
    `<input data-extra-model-field="good-sample" autocomplete="off" /></label>`,
    `<div class="material-source-section">`,
    `<div class="material-source-segment" role="tablist" aria-label="材料来源">`,
    `<button class="material-source-tab selected" type="button" role="tab" aria-selected="true"`,
    ` data-extra-material-tab="path" id="${pathTabId}" aria-controls="${pathPanelId}">文件路径</button>`,
    `<button class="material-source-tab" type="button" role="tab" aria-selected="false"`,
    ` data-extra-material-tab="upload" id="${uploadTabId}" aria-controls="${uploadPanelId}">文件上传</button>`,
    `</div>`,
    `<div class="material-source-panel" role="tabpanel" data-extra-material-panel="path"`,
    ` id="${pathPanelId}" aria-labelledby="${pathTabId}">`,
    `<label class="wide-field"><span>材料目录</span>`,
    `<input data-extra-model-field="source-dir" placeholder="/path/to/project" autocomplete="off" /></label>`,
    `</div>`,
    `<div class="material-source-panel material-upload-panel" hidden role="tabpanel"`,
    ` data-extra-material-panel="upload" id="${uploadPanelId}" aria-labelledby="${uploadTabId}">`,
    `<input class="visually-hidden" type="file" multiple data-extra-upload-input />`,
    `<div class="material-upload-dropzone" role="button" tabindex="0" aria-describedby="${uploadStatusId}">`,
    `<svg class="material-upload-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">`,
    `<path d="M12 15V4"></path><path d="M7.5 8.5 12 4l4.5 4.5"></path>`,
    `<path d="M5 15.5v2.2A2.3 2.3 0 0 0 7.3 20h9.4a2.3 2.3 0 0 0 2.3-2.3v-2.2"></path>`,
    `</svg>`,
    `<strong>点击或拖拽上传</strong>`,
    `<span id="${uploadStatusId}" data-extra-upload-status>请选择文件或文件夹。</span>`,
    `</div></div></div>`,
    `</article>`,
  ].join("");
}

export function modelRecipeFamily(recipe) {
  const normalized = String(recipe || "").trim().toLowerCase();
  if (normalized.endsWith("_regressor")) return "continuous";
  if (normalized.endsWith("_multiclass")) return "multiclass";
  return "binary";
}

export function modelTargetTypeForRecipes(recipes = []) {
  const families = new Set((recipes || []).map(modelRecipeFamily));
  return families.size === 1 ? [...families][0] : null;
}

export function resetCreateTaskSpecificInputs({
  $,
  root = document,
} = {}) {
  const getElement = typeof $ === "function" ? $ : () => null;
  for (const id of [
    "modelName",
    "validator",
    "sourceDir",
    "modelOotKsMin",
    "materialUploadInput",
  ]) {
    const input = getElement(id);
    if (input) input.value = "";
  }
  root?.querySelectorAll?.("[data-create-report-key]").forEach((input) => {
    input.value = "";
  });
}

export function createCreateTaskDialogController({
  $,
  materialSourceController,
  getSelectedTier,
  selectedTierStorageKey,
  onUnavailableTaskType,
} = {}) {
  let activeTaskType = defaultTaskType;
  let extraValidationModelRows = [];
  let extraValidationRowSequence = 0;

  function taskTypeDefinition(taskType = activeTaskType) {
    return taskTypeDefinitions[taskType] || taskTypeDefinitions[defaultTaskType];
  }

  function getActiveTaskType() {
    return activeTaskType;
  }

  function setRunModeCardState(mode, { disabled = false, checked = false } = {}) {
    const input = document.querySelector(`input[name="runMode"][value="${mode}"]`);
    if (!input) return;
    input.disabled = disabled;
    input.checked = checked;
    const card = input.closest(".run-mode-card");
    card?.classList.toggle("disabled", disabled);
    card?.setAttribute("aria-disabled", disabled ? "true" : "false");
    if (!disabled) {
      card?.removeAttribute("aria-disabled");
    }
  }

  function setRunModeDescription(mode, description = "") {
    const descriptionElement = document.querySelector(`[data-run-mode-description="${mode}"]`);
    if (!descriptionElement) return;
    descriptionElement.textContent = description;
  }

  function applyTaskTypeToDialog(taskType = defaultTaskType) {
    activeTaskType = taskTypeDefinition(taskType) === taskTypeDefinitions[defaultTaskType]
      ? defaultTaskType
      : taskType;
    const definition = taskTypeDefinition(activeTaskType);
    $("taskType").value = activeTaskType;
    $("taskDialogTitle").textContent = definition.dialogTitle;
    $("taskDialogSubtitle").textContent = definition.dialogSubtitle;
    $("modelNameLabel").textContent = definition.nameLabel;
    $("modelName").placeholder = definition.namePlaceholder;
    $("validatorLabel").textContent = definition.validatorLabel;
    $("validator").placeholder = definition.validatorPlaceholder;
    $("sourceDirLabel").textContent = definition.sourceLabel;
    $("sourceDir").placeholder = definition.sourcePlaceholder;
    const primaryHeading = $("createTaskPrimaryModelHeading");
    if (primaryHeading) {
      primaryHeading.textContent = activeTaskType === "validation" ? "模型 1" : "任务信息";
    }
    $("createTaskReportFields").hidden = !definition.reportFields;
    $("createTaskReportFields").classList.toggle("hidden", !definition.reportFields);
    toggleConditionalField("createTaskStrategyField", Boolean(definition.strategyField));
    setRunModeCardState("manual", {
      disabled: !definition.manualEnabled,
      checked: false,
    });
    setRunModeDescription("manual", definition.manualModeDescription);
    setRunModeCardState("agent", {
      disabled: false,
      checked: false,
    });
    setRunModeDescription("agent", definition.agentModeDescription);
    toggleConditionalField("validationCreateExtraModelsSection", activeTaskType === "validation");
    updateAlgorithmFieldVisibility();
  }

  function updateAlgorithmFieldVisibility() {
    const definition = taskTypeDefinition($("taskType")?.value || activeTaskType || defaultTaskType);
    const runMode = document.querySelector('input[name="runMode"]:checked')?.value;
    toggleConditionalField("createTaskAlgorithmField", Boolean(definition.algorithmField) && runMode === "manual");
    // Feature metric selection is one contract in both modes. Agent mode may
    // explain/suggest, but it must not silently replace the user's checked set.
    toggleConditionalField("createTaskMetricField", Boolean(definition.metricField));
    const meaningMetric = document.querySelector(
      'input[name="featureMetric"][value="meaning_consistency"]',
    );
    if (meaningMetric) {
      meaningMetric.disabled = runMode !== "agent";
      if (meaningMetric.disabled) meaningMetric.checked = false;
    }
    toggleConditionalField("createTaskTierField", Boolean(definition.tierField) && runMode === "agent");
    toggleConditionalField("createTaskStrategyField", Boolean(definition.strategyField));
  }

  function syncCreateTaskTierDefault() {
    const select = $("createTaskTier");
    if (!select) return;
    const selected = getSelectedTier?.()
      || (typeof localStorage !== "undefined" ? String(localStorage.getItem(selectedTierStorageKey) || "") : "");
    if (selected && [...select.options].some((option) => option.value === selected)) {
      select.value = selected;
    }
  }

  function toggleConditionalField(id, show) {
    const field = $(id);
    if (!field) return;
    field.hidden = !show;
    field.classList.toggle("hidden", !show);
  }

  function resetModelAlgorithmChoices() {
    document.querySelectorAll('input[name="modelAlgorithm"]').forEach((input) => {
      input.checked = false;
    });
    document.querySelectorAll('input[name="featureMetric"]').forEach((input) => {
      // Reset to the product defaults declared in the markup. Using
      // ``defaultChecked`` means reopening the dialog does not accidentally
      // turn an explicit all-metric choice into an empty payload.
      input.checked = Boolean(input.defaultChecked);
    });
    const weightPolicy = $("modelSampleWeightPolicy");
    if (weightPolicy) weightPolicy.value = "none";
    const weightInput = $("modelSampleWeightCol");
    if (weightInput) weightInput.value = "";
    updateSampleWeightCreateState();
  }

  function resetStrategyTaskInput() {
    const defaults = {
      strategyEntryMode: "strategy_development",
      strategyObjective: "",
      strategyMaxBadRate: "",
      strategyMinApprovalRate: "",
      strategyBaselineId: "",
      strategyEadCol: "",
      strategyPdCol: "",
      strategyAnnualRate: "",
      strategyFundingRate: "",
      strategyLgd: "",
      strategyOperatingCost: "",
      strategyTermMonths: "",
    };
    for (const [id, value] of Object.entries(defaults)) {
      const input = $(id);
      if (input) input.value = value;
    }
    updateStrategyProfitVisibility();
  }

  function updateStrategyProfitVisibility() {
    const show = $("strategyEntryMode")?.value === "strategy_development"
      && $("strategyObjective")?.value === "max_profit";
    toggleConditionalField("strategyProfitFields", show);
  }

  function optionalNumber(id) {
    const raw = $(id)?.value?.trim?.() || "";
    return raw === "" ? null : Number(raw);
  }

  function collectStrategyTaskInput() {
    const entryMode = $("strategyEntryMode").value;
    const objective = $("strategyObjective").value;
    const input = {
      entry_mode: entryMode,
      objective,
      max_bad_rate: optionalNumber("strategyMaxBadRate"),
      min_approval_rate: optionalNumber("strategyMinApprovalRate"),
      baseline_strategy_id: $("strategyBaselineId").value.trim() || null,
      profit: null,
    };
    if (entryMode === "strategy_development" && objective === "max_profit") {
      input.profit = {
        ead_col: $("strategyEadCol").value.trim(),
        pd_col: $("strategyPdCol").value.trim(),
        annual_rate: optionalNumber("strategyAnnualRate"),
        funding_rate: optionalNumber("strategyFundingRate"),
        lgd: optionalNumber("strategyLgd"),
        operating_cost_per_loan: optionalNumber("strategyOperatingCost"),
        term_months: optionalNumber("strategyTermMonths"),
      };
    }
    return input;
  }

  function strategyInputError(input) {
    if (input.entry_mode === "strategy_analysis") return "";
    if (!input.objective) return "请选择完整策略开发的业务目标。";
    for (const [label, value] of [
      ["审批后坏率上限", input.max_bad_rate],
      ["通过率下限", input.min_approval_rate],
    ]) {
      if (value !== null && (!Number.isFinite(value) || value < 0 || value > 1)) {
        return `${label}必须是 0 到 1 之间的数字。`;
      }
    }
    if (input.max_bad_rate === null && input.min_approval_rate === null) {
      return "完整策略开发至少需要一个坏率上限或通过率下限。";
    }
    if (input.objective !== "max_profit") return "";
    const profit = input.profit || {};
    const numeric = [
      profit.annual_rate,
      profit.funding_rate,
      profit.lgd,
      profit.operating_cost_per_loan,
      profit.term_months,
    ];
    if (!profit.ead_col || !profit.pd_col || numeric.some((value) => value === null || !Number.isFinite(value))) {
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

  function updateSampleWeightCreateState() {
    const policy = $("modelSampleWeightPolicy")?.value || "none";
    const weightInput = $("modelSampleWeightCol");
    if (!weightInput) return;
    const explicit = policy === "explicit";
    weightInput.disabled = !explicit;
    weightInput.classList.toggle("is-disabled", !explicit);
    if (!explicit) weightInput.value = "";
  }

  function normalizeModelAlgorithmFamilies(changedInput = null) {
    const checked = [...document.querySelectorAll('input[name="modelAlgorithm"]:checked')];
    if (!checked.length) return;
    const activeFamily = changedInput?.checked
      ? (changedInput.dataset.recipeFamily || modelRecipeFamily(changedInput.value))
      : (checked[0].dataset.recipeFamily || modelRecipeFamily(checked[0].value));
    for (const input of document.querySelectorAll('input[name="modelAlgorithm"]')) {
      const family = input.dataset.recipeFamily || modelRecipeFamily(input.value);
      if (family !== activeFamily) input.checked = false;
    }
  }

  function openTaskDialog(taskType = defaultTaskType) {
    applyTaskTypeToDialog(taskType);
    resetCreateTaskSpecificInputs({ $ });
    document.querySelectorAll('input[name="runMode"]').forEach((input) => {
      input.checked = false;
    });
    resetModelAlgorithmChoices();
    resetStrategyTaskInput();
    syncCreateTaskTierDefault();
    updateAlgorithmFieldVisibility();
    document.querySelectorAll(".run-mode-card").forEach((card) => {
      delete card.dataset.wasChecked;
    });
    setCreateStatus("");
    materialSourceController.reset();
    resetValidationExtraModels();
    prefillCreateTaskReportFields();
    $("taskDialog").showModal();
    $("modelName").focus();
  }

  function openTaskDialogFromCard(event) {
    const card = event.target.closest("[data-task-kind]");
    if (!card) return;
    const definition = taskTypeDefinition(card.dataset.taskKind || defaultTaskType);
    if (definition.available === false) {
      const message = definition.unavailableMessage || "新功能开发中，敬请期待";
      if (typeof onUnavailableTaskType === "function") onUnavailableTaskType(message);
      return;
    }
    openTaskDialog(card.dataset.taskKind || defaultTaskType);
  }

  function closeTaskDialog() {
    $("taskDialog").close();
  }

  function handleRunModeCardPointerDown(event) {
    const card = event.target.closest(".run-mode-card");
    if (!card) return;
    const input = card.querySelector('input[name="runMode"]');
    if (!input) return;
    card.dataset.wasChecked = input.checked ? "true" : "false";
  }

  function handleRunModeCardClick(event) {
    const card = event.target.closest(".run-mode-card");
    if (!card) return;
    const input = card.querySelector('input[name="runMode"]');
    if (!input) return;
    if (card.dataset.wasChecked !== "true") return;
    event.preventDefault();
    input.checked = false;
    card.dataset.wasChecked = "false";
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function bindRunModeDeselectableCards() {
    document.querySelectorAll(".run-mode-card").forEach((card) => {
      card.addEventListener("pointerdown", handleRunModeCardPointerDown);
      card.addEventListener("click", handleRunModeCardClick);
    });
    document.querySelectorAll('input[name="runMode"]').forEach((input) => {
      input.addEventListener("change", updateAlgorithmFieldVisibility);
    });
    document.querySelectorAll('input[name="modelAlgorithm"]').forEach((input) => {
      input.addEventListener("change", () => normalizeModelAlgorithmFamilies(input));
    });
    $("modelSampleWeightPolicy")?.addEventListener("change", updateSampleWeightCreateState);
    $("strategyEntryMode")?.addEventListener("change", updateStrategyProfitVisibility);
    $("strategyObjective")?.addEventListener("change", updateStrategyProfitVisibility);
  }

  function taskTextSeed() {
    const modelName = $("modelName").value.trim() || "本模型";
    const validator = $("validator").value.trim();
    return {
      modelName,
      validator,
      reportTitle: `${modelName.endsWith("模型") ? modelName : `${modelName}模型`}验证文档`,
    };
  }

  function defaultCreateReportValues() {
    const seed = taskTextSeed();
    const today = formatDateInput();
    return {
      "TEXT:report_title": seed.reportTitle,
      "TEXT:drafter": seed.validator,
      "TEXT:draft_date": today,
      "TEXT:revision_version": "V1",
      "TEXT:revision_date": today,
      "TEXT:revision_author": seed.validator,
      "TEXT:revision_description": "初稿",
      ...validationNarrativeDefaults(seed.modelName),
    };
  }

  function prefillCreateTaskReportFields() {
    const defaults = defaultCreateReportValues();
    for (const input of document.querySelectorAll("[data-create-report-key]")) {
      const key = input.dataset.createReportKey;
      if (defaults[key] !== undefined) updateAutoReportValue(input, defaults[key]);
    }
  }

  function collectCreateTaskReportValues() {
    const values = defaultCreateReportValues();
    for (const input of document.querySelectorAll("[data-create-report-key]")) {
      values[input.dataset.createReportKey] = input.value.trim();
    }
    values["TEXT:report_title"] = values["TEXT:report_title"] || taskTextSeed().reportTitle;
    values["TEXT:drafter"] = values["TEXT:drafter"] || $("validator").value.trim();
    values["TEXT:revision_author"] = values["TEXT:revision_author"] || $("validator").value.trim();
    return values;
  }

  function setCreateStatus(message, kind = "info") {
    const status = $("statusMessage");
    status.textContent = message;
    status.className = `status ${kind}`;
  }

  // UX-12: XMLHttpRequest (not fetch/api()) because only XHR exposes upload
  // progress events. onProgress receives (loadedBytes, totalBytes) so the
  // caller can render "正在上传 N 个文件 (P%)" — large credit-data files can
  // take real time even on localhost, and fetch gives no signal at all
  // during that wait.
  function uploadMaterialFiles(files, { onProgress } = {}) {
    if (!files.length) {
      return Promise.reject(new Error("请先选择要上传的材料文件。"));
    }
    const formData = new FormData();
    files.forEach((item) => {
      formData.append("files", item.file, item.name);
      formData.append("relative_paths", item.relativePath || item.name);
    });
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "api/material-uploads");
      const token = localToken();
      if (token) xhr.setRequestHeader("X-Marvis-Token", token);
      if (xhr.upload && typeof onProgress === "function") {
        xhr.upload.onprogress = (event) => {
          if (!event.lengthComputable) return;
          onProgress(event.loaded, event.total);
        };
      }
      xhr.onload = () => {
        let payload = null;
        try {
          payload = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        } catch (_) {
          payload = null;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(payload);
          return;
        }
        const detail = payload?.detail;
        const message = typeof detail === "string" ? detail : (detail ? JSON.stringify(detail) : "材料上传失败");
        reject(new Error(message));
      };
      xhr.onerror = () => reject(new Error("材料上传失败：网络错误。"));
      xhr.onabort = () => reject(new Error("材料上传已取消。"));
      xhr.send(formData);
    });
  }

  async function createTask() {
    setCreateStatus("");
    const selectedRunMode = document.querySelector('input[name="runMode"]:checked')?.value;
    if (!selectedRunMode) {
      setCreateStatus("请选择执行模式。", "error");
      return null;
    }
    const taskType = $("taskType")?.value || activeTaskType || defaultTaskType;
    const definition = taskTypeDefinition(taskType);
    const allowDeferredMaterials = Boolean(definition.deferredMaterials)
      && selectedRunMode === "agent";
    const payload = {
      task_type: taskType,
      model_name: $("modelName").value.trim(),
      model_version: "",
      validator: $("validator").value.trim(),
      source_dir: $("sourceDir").value.trim(),
      run_mode: selectedRunMode,
      report_values: definition.reportFields ? collectCreateTaskReportValues() : {},
    };
    if (definition.strategyField) {
      const strategyInput = collectStrategyTaskInput();
      const error = strategyInputError(strategyInput);
      if (error) {
        setCreateStatus(error, "error");
        return null;
      }
      payload.strategy_input = strategyInput;
    }
    if (definition.algorithmField && selectedRunMode === "manual") {
      normalizeModelAlgorithmFamilies();
      payload.recipes = [...document.querySelectorAll('input[name="modelAlgorithm"]:checked')].map((box) => box.value);
      if (payload.recipes.length === 0) {
        setCreateStatus("请至少选择一个建模算法。", "error");
        return null;
      }
      const targetType = modelTargetTypeForRecipes(payload.recipes);
      if (!targetType) {
        setCreateStatus("二分类、回归与多分类算法不能混选。", "error");
        return null;
      }
      payload.target_type = targetType;
      const sampleWeightPolicy = $("modelSampleWeightPolicy")?.value || "none";
      if (sampleWeightPolicy === "explicit") {
        const sampleWeightCol = $("modelSampleWeightCol")?.value.trim();
        if (!sampleWeightCol) {
          setCreateStatus("请填写样本权重列，或改选不使用样本权重。", "error");
          return null;
        }
        payload.sample_weight_col = sampleWeightCol;
      }
      // AGT-4: optional minimum OOT KS success criterion. Left blank by default —
      // never defaulted to a platform-chosen number. Only meaningful for binary
      // targets (KS is not computed for continuous/multiclass recipes).
      const ootKsMinRaw = $("modelOotKsMin")?.value.trim();
      if (ootKsMinRaw) {
        const ootKsMin = Number(ootKsMinRaw);
        if (!Number.isFinite(ootKsMin) || ootKsMin < 0 || ootKsMin > 1) {
          setCreateStatus("成功标准（OOT KS 下限）必须是 0 到 1 之间的数字。", "error");
          return null;
        }
        if (payload.target_type !== "binary") {
          setCreateStatus("成功标准（OOT KS 下限）仅适用于二分类算法。", "error");
          return null;
        }
        payload.oot_ks_min = ootKsMin;
      }
    }
    if (definition.metricField) {
      payload.metrics = [...document.querySelectorAll('input[name="featureMetric"]:checked')].map((box) => box.value);
    }
    if (definition.tierField && selectedRunMode === "agent") {
      const tier = $("createTaskTier")?.value;
      if (tier) payload.capability_tier = tier;
    }
    if (taskType === "validation") {
      const extraRows = collectValidationExtraModels();
      if (extraRows.length > 0) {
        if (!payload.model_name || !payload.validator) {
          setCreateStatus("请先填写模型名称和验证人员。", "error");
          return null;
        }
        const primaryRow = collectPrimaryValidationModel(payload);
        if (!primaryRow || extraRows.some((row) => !validationModelMaterialsReady(row))) {
          setCreateStatus(
            "添加多个模型时，请为每个模型填写材料目录，或上传 Notebook、样本、PMML 和数据字典。",
            "error",
          );
          return null;
        }
        return await createMultiModelValidationTask(
          payload,
          [primaryRow, ...extraRows],
          selectedRunMode,
        );
      }
    }
    if (materialSourceController.mode() === "upload") {
      const files = materialSourceController.selectedFiles();
      if (files.length === 0 && !allowDeferredMaterials) {
        setCreateStatus("请先选择要上传的材料文件。", "error");
        return null;
      }
      if (!payload.model_name || !payload.validator) {
        setCreateStatus(
          definition.reportFields ? "请先填写模型名称和验证人员。" : "请先填写任务名称和负责人。",
          "error",
        );
        return null;
      }
      if (files.length > 0) {
        // UX-12: percentage only kicks in once there is something worth showing a
        // number for (>10MB total) — for a handful of small files the plain
        // "正在上传材料..." text is enough and a jumpy 0%→100% readout would be
        // noise, not signal.
        const totalBytes = files.reduce((sum, item) => sum + (Number(item.size) || 0), 0);
        const showPercent = totalBytes > MATERIAL_UPLOAD_PERCENT_THRESHOLD_BYTES;
        setCreateStatus(`正在上传材料...${showPercent ? " (0%)" : ""}`, "busy");
        const upload = await uploadMaterialFiles(files, {
          onProgress: showPercent
            ? (loaded, total) => {
                const percent = total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
                setCreateStatus(`正在上传材料...共 ${files.length} 个文件 (${percent}%)`, "busy");
              }
            : undefined,
        });
        payload.source_dir = upload.source_dir;
      }
    }
    if (!payload.model_name || !payload.validator || (!payload.source_dir && !allowDeferredMaterials)) {
      setCreateStatus(
        definition.reportFields ? "请先填写模型名称、验证人员和材料目录。" : "请先填写任务名称、负责人和材料目录。",
        "error",
      );
      return null;
    }
    return await api("api/tasks", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  function resetValidationExtraModels() {
    extraValidationModelRows = [];
    extraValidationRowSequence = 0;
    const container = $("validationCreateExtraModels");
    if (container) container.innerHTML = "";
    const addButton = $("addValidationModelRowButton");
    if (addButton) addButton.disabled = false;
  }

  function extraNarrativeDefaults(modelName) {
    const defaults = validationNarrativeDefaults(modelName);
    return {
      overview: defaults["TEXT:model_overview"],
      scope: defaults["TEXT:model_scope"],
      "bad-sample": defaults["TEXT:bad_sample_definition"],
      "good-sample": defaults["TEXT:good_sample_definition"],
    };
  }

  function classifiedFilesFromSelection(files) {
    const raw = (files || []).map((item) => item?.file || item).filter(Boolean);
    return classifyValidationBatchFiles(raw);
  }

  function validationModelMaterialsReady(row) {
    if (String(row?.sourceDir || "").trim()) return true;
    return Boolean(row?.files?.notebook && row?.files?.sample && row?.files?.pmml && row?.files?.dictionary);
  }

  function bindExtraModelMaterialSource(root) {
    const state = { mode: "path", files: [] };
    const pathTab = root.querySelector('[data-extra-material-tab="path"]');
    const uploadTab = root.querySelector('[data-extra-material-tab="upload"]');
    const pathPanel = root.querySelector('[data-extra-material-panel="path"]');
    const uploadPanel = root.querySelector('[data-extra-material-panel="upload"]');
    const input = root.querySelector("[data-extra-upload-input]");
    const dropzone = root.querySelector(".material-upload-dropzone");
    const status = root.querySelector("[data-extra-upload-status]");

    function setMode(nextMode) {
      state.mode = nextMode === "upload" ? "upload" : "path";
      const isPath = state.mode === "path";
      pathTab?.classList.toggle("selected", isPath);
      uploadTab?.classList.toggle("selected", !isPath);
      pathTab?.setAttribute("aria-selected", isPath ? "true" : "false");
      uploadTab?.setAttribute("aria-selected", isPath ? "false" : "true");
      if (pathPanel) {
        pathPanel.hidden = !isPath;
        pathPanel.classList.toggle("hidden", !isPath);
      }
      if (uploadPanel) {
        uploadPanel.hidden = isPath;
        uploadPanel.classList.toggle("hidden", isPath);
      }
    }

    function renderStatus() {
      if (!status) return;
      status.textContent = materialUploadSelectionText(state.files);
    }

    pathTab?.addEventListener("click", () => setMode("path"));
    uploadTab?.addEventListener("click", () => setMode("upload"));
    dropzone?.addEventListener("click", () => input?.click());
    dropzone?.addEventListener("keydown", (event) => {
      if (!["Enter", " "].includes(event.key)) return;
      event.preventDefault();
      input?.click();
    });
    if (input) {
      input.onchange = () => {
        state.files = Array.from(input.files || []);
        renderStatus();
      };
    }
    if (dropzone) {
      ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
          event.preventDefault();
          dropzone.classList.add("is-dragover");
        });
      });
      ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, () => {
          dropzone.classList.remove("is-dragover");
        });
      });
      dropzone.ondrop = (event) => {
        event.preventDefault();
        state.files = Array.from(event.dataTransfer?.files || []);
        renderStatus();
      };
    }
    return {
      mode: () => state.mode,
      selectedFiles: () => [...state.files],
      sourceDir: () => root.querySelector('[data-extra-model-field="source-dir"]')?.value.trim() || "",
    };
  }

  function extraFieldValue(root, field) {
    return root?.querySelector(`[data-extra-model-field="${field}"]`)?.value || "";
  }

  function renumberValidationExtraModels() {
    extraValidationModelRows.forEach((row, index) => {
      const heading = document.querySelector(
        `[data-validation-extra-row-id="${row.id}"] .validation-create-model-head h3`,
      );
      if (heading) heading.textContent = `模型 ${index + 2}`;
    });
  }

  function addValidationModelRow() {
    if (extraValidationModelRows.length + 1 >= MAX_VALIDATION_BATCH_ROWS) {
      setCreateStatus("一次最多验证 10 个模型。", "error");
      return;
    }
    extraValidationRowSequence += 1;
    const rowId = `validation-extra-${extraValidationRowSequence}`;
    const container = $("validationCreateExtraModels");
    if (!container) return;
    const ordinal = extraValidationModelRows.length + 2;
    container.insertAdjacentHTML("beforeend", validationExtraModelCardMarkup(rowId, ordinal));
    const article = container.querySelector(`[data-validation-extra-row-id="${rowId}"]`);
    if (!article) return;
    const nameInput = article.querySelector('[data-extra-model-field="name"]');
    const updateNarrativeDefaults = () => {
      for (const [field, value] of Object.entries(extraNarrativeDefaults(nameInput?.value))) {
        updateAutoReportValue(article.querySelector(`[data-extra-model-field="${field}"]`), value);
      }
    };
    updateNarrativeDefaults();
    nameInput?.addEventListener("input", updateNarrativeDefaults);
    const material = bindExtraModelMaterialSource(article);
    extraValidationModelRows.push({ id: rowId, material });
    article.querySelector("[data-remove-validation-extra-row]")?.addEventListener("click", () => {
      extraValidationModelRows = extraValidationModelRows.filter((row) => row.id !== rowId);
      article.remove();
      renumberValidationExtraModels();
    });
  }

  function collectPrimaryValidationModel(payload) {
    const uploadMode = materialSourceController.mode() === "upload";
    return {
      id: "validation-primary",
      modelName: payload.model_name,
      modelVersion: payload.model_version || "v1",
      sourceDir: uploadMode ? "" : payload.source_dir,
      files: uploadMode
        ? classifiedFilesFromSelection(materialSourceController.selectedFiles()) || {}
        : {},
    };
  }

  function collectValidationExtraModels() {
    return extraValidationModelRows.map((row) => {
      const element = document.querySelector(`[data-validation-extra-row-id="${row.id}"]`);
      const uploadMode = row.material?.mode() === "upload";
      return {
        id: row.id,
        modelName: extraFieldValue(element, "name"),
        modelVersion: "v1",
        sourceDir: uploadMode ? "" : (row.material?.sourceDir() || extraFieldValue(element, "source-dir")),
        files: uploadMode
          ? classifiedFilesFromSelection(row.material?.selectedFiles()) || {}
          : {},
      };
    });
  }

  async function createMultiModelValidationTask(firstPayload, rows, runMode) {
    if (!Array.isArray(rows) || rows.length < 2) {
      setCreateStatus(
        "添加多个模型时，请为每个模型填写材料目录，或上传 Notebook、样本、PMML 和数据字典。",
        "error",
      );
      return null;
    }
    void runMode;
    setCreateStatus("正在创建多个模型的验证任务...", "busy");
    try {
      const created = await submitValidationBatchDraft(
        {
          batchName: formatValidationBatchTaskName(new Date(), rows.length),
          validator: firstPayload.validator,
          rows,
        },
        {
          uploadMaterials: async ({ row }) => api("/api/validation-batches/material-uploads", {
            method: "POST",
            body: buildValidationBatchUploadFormData(row),
          }),
          createBatch: async (body) => api("/api/validation-batches", {
            method: "POST",
            body: JSON.stringify(body),
          }),
          cleanupMaterials: async (token) => api(
            `/api/validation-batches/material-uploads/${encodeURIComponent(token)}`,
            { method: "DELETE" },
          ),
        },
      );
      const parentTask = created?.parent_task;
      const parentTaskId = parentTask?.id || created?.batch?.parent_task_id;
      if (!parentTaskId) throw new Error("多个模型创建响应缺少父任务 ID。");
      return parentTask || {
        id: parentTaskId,
        task_type: "validation_batch",
        run_mode: "agent",
      };
    } catch (error) {
      setCreateStatus(error?.message || "多个模型创建失败。", "error");
      return null;
    }
  }

  function bindMaterialSourceControls() {
    materialSourceController.bindTabs();
    materialSourceController.bindDropzone();
    $("modelName")?.addEventListener("input", prefillCreateTaskReportFields);
    $("validator")?.addEventListener("input", prefillCreateTaskReportFields);
    $("addValidationModelRowButton")?.addEventListener("click", () => {
      addValidationModelRow();
    });
  }

  return {
    activeTaskType: getActiveTaskType,
    bindMaterialSourceControls,
    bindRunModeDeselectableCards,
    closeTaskDialog,
    createTask,
    openTaskDialog,
    openTaskDialogFromCard,
    setCreateStatus,
    syncCreateTaskTierDefault,
    taskTypeDefinition,
  };
}
