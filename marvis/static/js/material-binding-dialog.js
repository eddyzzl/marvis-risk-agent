const MATERIAL_BINDING_ROLES = [
  { role: "notebook", field: "notebook_path", label: "Notebook", caption: "Notebook 文件" },
  { role: "sample", field: "sample_path", label: "Sample", caption: "样本数据" },
  { role: "model_pmml", field: "pmml_path", label: "PMML", caption: "PMML 模型" },
  { role: "data_dictionary", field: "dictionary_path", label: "Metadata", caption: "数据字典 / 特征元数据" },
];

const MATERIAL_BINDING_ROLE_ICONS = {
  notebook: '<path d="M6 4.5h9.8A2.2 2.2 0 0 1 18 6.7v12.8H7.4A2.4 2.4 0 0 1 5 17.1V5.5a1 1 0 0 1 1-1Z"></path><path d="M9 8h5.5M9 11h4"></path><path d="M7.4 19.5A2.4 2.4 0 0 1 5 17.1"></path>',
  sample: '<ellipse cx="12" cy="6.5" rx="6.5" ry="2.7"></ellipse><path d="M5.5 6.5v10.8c0 1.5 2.9 2.7 6.5 2.7s6.5-1.2 6.5-2.7V6.5"></path><path d="M5.5 12c0 1.5 2.9 2.7 6.5 2.7s6.5-1.2 6.5-2.7"></path>',
  model_pmml: '<path d="M12 3.5 19 7.5v8.8l-7 4-7-4V7.5Z"></path><path d="M5.3 7.7 12 11.5l6.7-3.8"></path><path d="M12 11.6v8.1"></path>',
  data_dictionary: '<path d="M5.5 4.5h9.2A3.8 3.8 0 0 1 18.5 8.3v11.2H7.7a2.2 2.2 0 0 1-2.2-2.2Z"></path><path d="M8.5 8.5h6M8.5 12h5M8.5 15.5h3.5"></path>',
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatBytes(value) {
  const size = Number(value || 0);
  if (!Number.isFinite(size) || size <= 0) return "";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function candidateText(candidate) {
  const size = formatBytes(candidate?.size_bytes);
  const metadataStatus = candidate?.metadata_compatibility?.status;
  const compatibility = candidate?.recommended
    ? " · 推荐：与 PMML 完整匹配"
    : metadataStatus === "compatible"
      ? " · 与 PMML 匹配"
      : metadataStatus === "incompatible"
        ? " · 元数据不完整"
        : metadataStatus === "not_evaluated"
          ? " · 需人工确认"
          : "";
  return `${candidate?.relative_path || candidate?.name || "未命名文件"}${size ? ` · ${size}` : ""}${compatibility}`;
}

function defaultSelectionForRole(role, candidates, selection) {
  const selected = String(selection?.[role.field] || "");
  if (selected) return selected;
  const recommendedCandidates = candidates.filter((candidate) => candidate.recommended);
  if (recommendedCandidates.length === 1) return recommendedCandidates[0].relative_path || "";
  const hasMetadataAssessment = candidates.some((candidate) => candidate.metadata_compatibility);
  if (role.role === "data_dictionary" && (hasMetadataAssessment || candidates.length > 1)) return "";
  const exactRoleCandidates = candidates.filter((candidate) => candidate.role === role.role);
  if (exactRoleCandidates.length === 1) return exactRoleCandidates[0].relative_path || "";
  if (candidates.length === 1) return candidates[0].relative_path || "";
  return "";
}

function completeSelection(selection = {}) {
  return MATERIAL_BINDING_ROLES.every((role) => String(selection[role.field] || "").trim());
}

function renderRoleRow(role, candidates, value) {
  const disabled = candidates.length === 0 ? " disabled" : "";
  const icon = MATERIAL_BINDING_ROLE_ICONS[role.role] || "";
  const options = candidates.length
    ? [
        '<option value="">请选择</option>',
        ...candidates.map((candidate) => {
          const relativePath = candidate.relative_path || "";
          const selected = relativePath === value ? " selected" : "";
          return `<option value="${escapeHtml(relativePath)}"${selected}>${escapeHtml(candidateText(candidate))}</option>`;
        }),
      ].join("")
    : '<option value="">未找到可选文件</option>';
  return [
    '<label class="material-binding-row">',
    '<span class="material-binding-role">',
    `<span class="material-binding-role-icon" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false">${icon}</svg></span>`,
    '<span class="material-binding-role-text">',
    `<strong>${escapeHtml(role.label)}</strong>`,
    `<small>${escapeHtml(role.caption)}</small>`,
    "</span>",
    "</span>",
    `<select data-material-binding-field="${escapeHtml(role.field)}" aria-label="选择${escapeHtml(role.caption)}"${disabled}>${options}</select>`,
    "</label>",
  ].join("");
}

export function createMaterialBindingDialogController({ $, api } = {}) {
  let pendingResolve = null;
  let activeTask = null;
  let previewRequestSequence = 0;

  function setStatus(message, kind = "info") {
    const status = $("materialBindingStatus");
    if (!status) return;
    status.textContent = message;
    status.className = `status ${kind}`;
  }

  function render(payload) {
    const rows = $("materialBindingRows");
    if (!rows) return;
    const selection = payload?.selection || {};
    rows.innerHTML = MATERIAL_BINDING_ROLES.map((role) => {
      const candidates = Array.isArray(payload?.candidates?.[role.role])
        ? payload.candidates[role.role]
        : [];
      const selected = defaultSelectionForRole(role, candidates, selection);
      return renderRoleRow(role, candidates, selected);
    }).join("");
  }

  function collectSelection() {
    const selection = {};
    document.querySelectorAll("[data-material-binding-field]").forEach((select) => {
      selection[select.dataset.materialBindingField] = select.value;
    });
    return selection;
  }

  async function refreshMetadataRecommendation(pmmlPath) {
    if (!activeTask?.id) return;
    const requestSequence = ++previewRequestSequence;
    const dictionarySelect = document.querySelector(
      '[data-material-binding-field="dictionary_path"]',
    );
    if (dictionarySelect) dictionarySelect.value = "";
    const confirmButton = $("materialBindingConfirmButton");
    if (confirmButton) confirmButton.disabled = true;
    setStatus("正在检查所选 PMML 与特征元数据...", "busy");
    try {
      const payload = await api(
        `/api/tasks/${activeTask.id}/materials?pmml_path=${encodeURIComponent(pmmlPath)}`,
      );
      if (requestSequence !== previewRequestSequence) return;
      const currentSelection = collectSelection();
      render({
        ...payload,
        selection: {
          ...(payload.selection || {}),
          ...currentSelection,
          pmml_path: pmmlPath,
        },
      });
      setStatus(
        payload.recommendation
          ? "已推荐与所选 PMML 完整匹配的特征元数据，请确认。"
          : "已按所选 PMML 更新元数据检查结果，请确认。",
        "info",
      );
    } catch (error) {
      if (requestSequence !== previewRequestSequence) return;
      setStatus(error?.message || "特征元数据检查失败。", "error");
    } finally {
      if (requestSequence === previewRequestSequence && confirmButton) {
        confirmButton.disabled = false;
      }
    }
  }

  function resolvePending(value) {
    if (!pendingResolve) return;
    const resolve = pendingResolve;
    pendingResolve = null;
    resolve(value);
  }

  function closeWith(value) {
    previewRequestSequence += 1;
    activeTask = null;
    resolvePending(value);
    const dialog = $("materialBindingDialog");
    if (dialog?.open) dialog.close(value ? "confirm" : "cancel");
  }

  async function confirmSelection() {
    if (!activeTask?.id) return;
    const selection = collectSelection();
    if (!completeSelection(selection)) {
      setStatus("请为四类材料都选择对应文件。", "error");
      return;
    }
    setStatus("正在保存材料选择...", "busy");
    try {
      const result = await api(`/api/tasks/${activeTask.id}/materials`, {
        method: "PUT",
        body: JSON.stringify(selection),
      });
      setStatus("材料选择已保存。", "success");
      closeWith(result.task || activeTask);
    } catch (error) {
      setStatus(error?.message || "材料选择保存失败。", "error");
    }
  }

  async function ensureMaterialSelection(task, { force = false } = {}) {
    if (!task || (task.task_type || "validation") !== "validation") return task;
    const payload = await api(`/api/tasks/${task.id}/materials`);
    if (!force && completeSelection(payload.selection)) return task;
    previewRequestSequence += 1;
    activeTask = task;
    render(payload);
    const confirmButton = $("materialBindingConfirmButton");
    if (confirmButton) confirmButton.disabled = false;
    setStatus("");
    const dialog = $("materialBindingDialog");
    if (!dialog) return task;
    dialog.showModal();
    return await new Promise((resolve) => {
      pendingResolve = resolve;
    });
  }

  function bind() {
    $("materialBindingRows")?.addEventListener("change", (event) => {
      const field = event.target?.dataset?.materialBindingField;
      if (field === "pmml_path") {
        refreshMetadataRecommendation(event.target.value || "");
      }
    });
    $("materialBindingConfirmButton")?.addEventListener("click", confirmSelection);
    $("materialBindingCancelButton")?.addEventListener("click", () => closeWith(null));
    $("closeMaterialBindingDialogButton")?.addEventListener("click", () => closeWith(null));
    $("materialBindingDialog")?.addEventListener("cancel", (event) => {
      event.preventDefault();
      closeWith(null);
    });
    $("materialBindingDialog")?.addEventListener("close", () => {
      resolvePending(null);
    });
  }

  return {
    bind,
    ensureMaterialSelection,
  };
}
