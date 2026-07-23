import { api } from "./api.js";
import { defaultTaskType, taskTypeDefinitions } from "./task-types.js";
import { formatDateInput } from "./ui-utils.js";

// UX-12: below this total upload size, plain "正在上传材料..." is enough — a
// percentage readout for a few small files would jump straight to 100% and
// add noise, not signal. Large credit-sample/feature-table files (the actual
// case this is for) clear this easily.
const MATERIAL_UPLOAD_PERCENT_THRESHOLD_BYTES = 10 * 1024 * 1024;

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

export function createCreateTaskDialogController({
  $,
  materialSourceController,
  getSelectedTier,
  selectedTierStorageKey,
  onUnavailableTaskType,
} = {}) {
  let activeTaskType = defaultTaskType;

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
      reportTitle: `${modelName}模型验证文档`,
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
      "TEXT:model_overview": `为了更好的对xx用户进行授信环节风险管控，现开发${seed.modelName}模型，对xx客群做前置风险拦截，从授信申请阶段做好风险防范。`,
      "TEXT:model_scope": "本模型适用于xx渠道用户。",
      "TEXT:bad_sample_definition": "xx逾期 >= xx天",
      "TEXT:good_sample_definition": "xx未逾期",
    };
  }

  function prefillCreateTaskReportFields() {
    const defaults = defaultCreateReportValues();
    for (const input of document.querySelectorAll("[data-create-report-key]")) {
      const key = input.dataset.createReportKey;
      if (!input.value.trim() && defaults[key]) input.value = defaults[key];
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
      xhr.open("POST", "/api/material-uploads");
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

  function bindMaterialSourceControls() {
    materialSourceController.bindTabs();
    materialSourceController.bindDropzone();
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
