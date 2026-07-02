import { api } from "./api.js";
import { defaultTaskType, taskTypeDefinitions } from "./task-types.js";
import { formatDateInput } from "./ui-utils.js";

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
    toggleConditionalField("createTaskMetricField", Boolean(definition.metricField) && runMode === "manual");
    toggleConditionalField("createTaskTierField", Boolean(definition.tierField) && runMode === "agent");
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
    document.querySelectorAll('input[name="modelAlgorithm"], input[name="featureMetric"]').forEach((input) => {
      input.checked = false;
    });
    const weightPolicy = $("modelSampleWeightPolicy");
    if (weightPolicy) weightPolicy.value = "none";
    const weightInput = $("modelSampleWeightCol");
    if (weightInput) weightInput.value = "";
    updateSampleWeightCreateState();
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

  function modelRecipeFamily(recipe) {
    if (recipe === "lgb_regressor") return "continuous";
    if (recipe === "lgb_multiclass") return "multiclass";
    return "binary";
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

  async function uploadMaterialFiles(files) {
    if (!files.length) {
      throw new Error("请先选择要上传的材料文件。");
    }
    const formData = new FormData();
    files.forEach((item) => {
      formData.append("files", item.file, item.name);
      formData.append("relative_paths", item.relativePath || item.name);
    });
    return await api("api/material-uploads", {
      method: "POST",
      body: formData,
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
    const payload = {
      task_type: taskType,
      model_name: $("modelName").value.trim(),
      model_version: "",
      validator: $("validator").value.trim(),
      source_dir: $("sourceDir").value.trim(),
      run_mode: selectedRunMode,
      report_values: definition.reportFields ? collectCreateTaskReportValues() : {},
    };
    if (definition.algorithmField && selectedRunMode === "manual") {
      normalizeModelAlgorithmFamilies();
      payload.recipes = [...document.querySelectorAll('input[name="modelAlgorithm"]:checked')].map((box) => box.value);
      if (payload.recipes.length === 0) {
        setCreateStatus("请至少选择一个建模算法。", "error");
        return null;
      }
      const families = new Set(payload.recipes.map(modelRecipeFamily));
      if (families.size > 1) {
        setCreateStatus("二分类、回归与多分类算法不能混选。", "error");
        return null;
      }
      payload.target_type = [...families][0] || "binary";
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
    if (definition.metricField && selectedRunMode === "manual") {
      payload.metrics = [...document.querySelectorAll('input[name="featureMetric"]:checked')].map((box) => box.value);
    }
    if (definition.tierField && selectedRunMode === "agent") {
      const tier = $("createTaskTier")?.value;
      if (tier) payload.capability_tier = tier;
    }
    if (materialSourceController.mode() === "upload") {
      const files = materialSourceController.selectedFiles();
      if (files.length === 0) {
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
      setCreateStatus("正在上传材料...");
      const upload = await uploadMaterialFiles(files);
      payload.source_dir = upload.source_dir;
    }
    if (!payload.model_name || !payload.validator || !payload.source_dir) {
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
