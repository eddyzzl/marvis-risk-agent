const MATERIAL_BINDING_ROLES = [
  { role: "notebook", field: "notebook_path", label: "Notebook", caption: "Notebook 文件" },
  { role: "sample", field: "sample_path", label: "样本数据", caption: "样本数据" },
  { role: "model_pmml", field: "pmml_path", label: "PMML 模型", caption: "PMML 模型" },
  { role: "data_dictionary", field: "dictionary_path", label: "数据字典", caption: "数据字典 / 特征元数据" },
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

// Mirrors marvis.validation_report_copy.metadata_reason_codes: the backend's
// English blocking errors become one Chinese reason the user can act on.
const METADATA_REASON_LABELS = {
  missing_feature_column: "缺少 feature（特征名）列",
  missing_category_column: "缺少 category（特征类别/数据源）列",
  missing_importance_column: "缺少 importance（特征重要性）列",
  sample_file: "这是样本数据文件，不是数据字典",
  not_evaluated: "文件过大或候选过多，平台未能自动检查",
  no_table: "没有读到可用的表格",
  pmml_mismatch: "没有覆盖所选 PMML 的入模特征，或类别/重要性取值无效",
};

function metadataReasonCodes(blockingErrors) {
  const codes = [];
  (Array.isArray(blockingErrors) ? blockingErrors : []).forEach((error) => {
    const text = String(error || "").toLowerCase();
    let found = ["feature", "category", "importance"]
      .filter((column) => new RegExp(`missing\\b[^;]*\\b${column} column alias`).test(text))
      .map((column) => `missing_${column}_column`);
    if (!found.length) {
      if (text.includes("sample data file")) found = ["sample_file"];
      else if (text.includes("limit")) found = ["not_evaluated"];
      else if (text.includes("no candidate table")) found = ["no_table"];
      else found = ["pmml_mismatch"];
    }
    found.forEach((code) => { if (!codes.includes(code)) codes.push(code); });
  });
  return codes;
}

export function metadataUnavailableReason(candidate) {
  const codes = metadataReasonCodes(candidate?.metadata_compatibility?.blocking_errors);
  if (!codes.length) return "";
  const reasons = codes.map((code) => METADATA_REASON_LABELS[code]).join("；");
  return codes.some((code) => code.startsWith("missing_"))
    ? `${reasons}，请补充后重新上传`
    : reasons;
}

export function materialCandidateSelectable(role, candidate) {
  if (candidate?.available === false) return false;
  if (role?.role !== "data_dictionary") return true;
  return candidate?.metadata_compatibility?.status !== "incompatible";
}

function incompatibleMetadataText(candidate) {
  const codes = metadataReasonCodes(candidate?.metadata_compatibility?.blocking_errors);
  if (codes.length === 1 && codes[0] === "sample_file") return " · 样本数据，不能作数据字典";
  const reason = metadataUnavailableReason(candidate);
  return ` · 元数据不完整${reason ? `：${reason}` : ""}`;
}

function candidateText(candidate) {
  const size = formatBytes(candidate?.size_bytes);
  const metadataStatus = candidate?.metadata_compatibility?.status;
  const compatibility = candidate?.recommended
    ? " · 推荐：与 PMML 完整匹配"
    : metadataStatus === "compatible"
      ? " · 与 PMML 匹配"
      : metadataStatus === "incompatible"
        ? incompatibleMetadataText(candidate)
        : metadataStatus === "not_evaluated"
          ? " · 需人工确认"
          : "";
  const unavailable = metadataStatus === "incompatible" || candidate?.available === false
    ? " · 不可选"
    : "";
  return `${candidate?.relative_path || candidate?.name || "未命名文件"}${size ? ` · ${size}` : ""}${compatibility}${unavailable}`;
}

function defaultSelectionForRole(role, candidates, selection) {
  const selected = String(selection?.[role.field] || "");
  const selectableCandidates = candidates.filter((candidate) => (
    materialCandidateSelectable(role, candidate)
  ));
  const selectedCandidate = selectableCandidates.find((candidate) => (
    String(candidate?.relative_path || "") === selected
  ));
  if (selected && selectedCandidate) return selected;
  const recommendedCandidates = selectableCandidates.filter((candidate) => candidate.recommended);
  if (recommendedCandidates.length === 1) return recommendedCandidates[0].relative_path || "";
  const hasMetadataAssessment = candidates.some((candidate) => candidate.metadata_compatibility);
  if (role.role === "data_dictionary" && (hasMetadataAssessment || candidates.length > 1)) return "";
  const exactRoleCandidates = selectableCandidates.filter((candidate) => candidate.role === role.role);
  if (exactRoleCandidates.length === 1) return exactRoleCandidates[0].relative_path || "";
  if (selectableCandidates.length === 1) return selectableCandidates[0].relative_path || "";
  return "";
}

function completeSelection(selection = {}) {
  return MATERIAL_BINDING_ROLES.every((role) => String(selection[role.field] || "").trim());
}

// Eval S2: all four materials matched (the dictionary even marked 推荐) and
// the dialog still asked for one more click. When every role has the single
// choice the dialog itself would preselect, and that choice raises no issue,
// the selection is saved without asking.
export function autoMaterialSelection(payload = {}) {
  const selection = {};
  for (const role of MATERIAL_BINDING_ROLES) {
    const candidates = Array.isArray(payload?.candidates?.[role.role]) ? payload.candidates[role.role] : [];
    const value = defaultSelectionForRole(role, candidates, payload?.selection || {});
    if (!value) return null;
    selection[role.field] = value;
  }
  return materialSelectionIssue(selection, payload) ? null : selection;
}

export function materialSelectionIssue(selection = {}, payload = {}) {
  const role = MATERIAL_BINDING_ROLES.find((item) => item.role === "data_dictionary");
  const candidates = Array.isArray(payload?.candidates?.data_dictionary)
    ? payload.candidates.data_dictionary
    : [];
  const selected = String(selection?.dictionary_path || "");
  const selectedCandidate = candidates.find((candidate) => (
    String(candidate?.relative_path || "") === selected
  ));
  const guidance = "请更换 PMML 或补齐 feature/category/importance 列后重新扫描。";
  if (selectedCandidate && !materialCandidateSelectable(role, selectedCandidate)) {
    const reason = metadataUnavailableReason(selectedCandidate);
    return `当前特征元数据不可用${reason ? `（${reason}）` : ""}；${guidance}`;
  }
  if (candidates.length && !candidates.some((candidate) => materialCandidateSelectable(role, candidate))) {
    return `所选 PMML 没有可用的特征元数据；${guidance}`;
  }
  return "";
}

export function renderRoleRow(role, candidates, value) {
  const selectableCandidates = candidates.filter((candidate) => (
    materialCandidateSelectable(role, candidate)
  ));
  const disabled = selectableCandidates.length === 0 ? " disabled" : "";
  const icon = MATERIAL_BINDING_ROLE_ICONS[role.role] || "";
  const options = candidates.length
    ? [
        '<option value="">请选择</option>',
        ...candidates.map((candidate) => {
          const relativePath = candidate.relative_path || "";
          const selectable = materialCandidateSelectable(role, candidate);
          const selected = selectable && relativePath === value ? " selected" : "";
          const unavailable = selectable ? "" : ' disabled aria-disabled="true"';
          return `<option value="${escapeHtml(relativePath)}"${selected}${unavailable}>${escapeHtml(candidateText(candidate))}</option>`;
        }),
      ].join("")
    : '<option value="">未找到可选文件</option>';
  const reasons = role.role === "data_dictionary"
    ? candidates
      .filter((candidate) => !materialCandidateSelectable(role, candidate)
        && metadataReasonCodes(candidate?.metadata_compatibility?.blocking_errors).some((code) => code !== "sample_file"))
      .map((candidate) => {
        const reason = metadataUnavailableReason(candidate);
        return reason ? `${candidate.relative_path || candidate.name || "数据字典"}：${reason}。` : "";
      })
      .filter(Boolean)
    : [];
  const unavailableHint = candidates.length && !selectableCandidates.length
    ? `<small class="material-binding-unavailable">${escapeHtml(reasons.join(" "))}${reasons.length ? " " : ""}所选 PMML 没有可用的特征元数据；请更换 PMML 或补齐 feature/category/importance 列后重新扫描。</small>`
    : reasons.length
      ? `<small class="material-binding-unavailable">${escapeHtml(reasons.join(" "))}</small>`
      : "";
  return [
    '<label class="material-binding-row">',
    '<span class="material-binding-role">',
    `<span class="material-binding-role-icon" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false">${icon}</svg></span>`,
    '<span class="material-binding-role-text">',
    `<strong>${escapeHtml(role.label)}</strong>`,
    "</span>",
    "</span>",
    `<select data-material-binding-field="${escapeHtml(role.field)}" aria-label="选择${escapeHtml(role.caption)}"${disabled}>${options}</select>`,
    unavailableHint,
    "</label>",
  ].join("");
}

// A freshly uploaded file is chosen for one role: the dictionary when the
// platform found it usable as one, otherwise the role it was recognised as
// (a CSV is offered under every CSV role, so "offered" alone is not enough).
// The rest of the user's choices stay.
function uploadedRole(payload, path) {
  const candidatesFor = (role) => (
    Array.isArray(payload?.candidates?.[role.role]) ? payload.candidates[role.role] : []
  );
  const dictionary = MATERIAL_BINDING_ROLES.find((role) => role.role === "data_dictionary");
  const asDictionary = candidatesFor(dictionary).find((candidate) => candidate.relative_path === path);
  if (asDictionary?.metadata_compatibility?.status === "compatible") return dictionary;
  return MATERIAL_BINDING_ROLES.find((role) => candidatesFor(role).some((candidate) => (
    candidate.relative_path === path
    && candidate.role === role.role
    && materialCandidateSelectable(role, candidate)
  ))) || null;
}

export function uploadedMaterialSelection(payload, uploadedPaths) {
  const selection = {};
  const chosenRoles = [];
  uploadedPaths.forEach((path) => {
    const role = uploadedRole(payload, path);
    if (!role || selection[role.field]) return;
    selection[role.field] = path;
    chosenRoles.push(role.caption);
  });
  return { selection, chosenRoles };
}

export function createMaterialBindingDialogController({ $, api } = {}) {
  let pendingResolve = null;
  let activeTask = null;
  let activePayload = null;
  let previewRequestSequence = 0;

  function setStatus(message, kind = "info") {
    const status = $("materialBindingStatus");
    if (!status) return;
    status.textContent = message;
    status.className = `status ${kind}`;
  }

  function renderSupplement(payload) {
    const supported = payload?.supplement_upload_supported !== false;
    const label = $("materialBindingUploadLabel");
    if (label) label.hidden = !supported;
    const hint = $("materialBindingSupplementHint");
    if (hint) {
      hint.textContent = supported
        ? "补好的数据字典等文件可以直接传进这个任务，原文件保留。"
        : "这个任务的材料来自本地目录：把补好的文件放进该目录，再点「刷新列表」。";
    }
  }

  function render(payload) {
    const rows = $("materialBindingRows");
    if (!rows) return;
    renderSupplement(payload);
    activePayload = payload || {};
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

  // The candidate list for the PMML currently chosen in the dialog.
  async function fetchMaterials() {
    const pmmlPath = collectSelection().pmml_path || "";
    const query = pmmlPath ? `?pmml_path=${encodeURIComponent(pmmlPath)}` : "";
    return api(`/api/tasks/${activeTask.id}/materials${query}`);
  }

  async function reloadCandidates({ uploadedPaths = [] } = {}) {
    const requestSequence = ++previewRequestSequence;
    const payload = await fetchMaterials();
    if (requestSequence !== previewRequestSequence || !activeTask) return null;
    const { selection, chosenRoles } = uploadedMaterialSelection(payload, uploadedPaths);
    render({
      ...payload,
      selection: { ...(payload.selection || {}), ...collectSelection(), ...selection },
    });
    return chosenRoles;
  }

  async function uploadSupplementFiles(fileList) {
    const selected = Array.from(fileList || []);
    if (!activeTask?.id || !selected.length) return;
    const form = new FormData();
    selected.forEach((file) => form.append("files", file, file.name));
    const confirmButton = $("materialBindingConfirmButton");
    if (confirmButton) confirmButton.disabled = true;
    setStatus(`正在上传 ${selected.length} 个文件...`, "busy");
    try {
      const result = await api(`/api/tasks/${activeTask.id}/materials/files`, {
        method: "POST",
        body: form,
      });
      const uploadedPaths = (result?.files || []).map((item) => String(item.relative_path || ""));
      const chosenRoles = await reloadCandidates({ uploadedPaths });
      if (chosenRoles === null) return;
      const names = selected.map((file) => file.name).join("、");
      const next = activeTask?.run_mode === "agent" ? "确认后 Agent 会重新扫描" : "确认后请点「重新扫描」";
      setStatus(
        chosenRoles.length
          ? `已上传 ${names}，并为${chosenRoles.join("、")}选中新文件；${next}。`
          : `已上传 ${names}；请在上面的列表里选择它。`,
        chosenRoles.length ? "success" : "info",
      );
    } catch (error) {
      setStatus(error?.message || "补充材料上传失败。", "error");
    } finally {
      if (confirmButton) confirmButton.disabled = false;
    }
  }

  async function refreshCandidates() {
    if (!activeTask?.id) return;
    setStatus("正在刷新材料列表...", "busy");
    try {
      const chosenRoles = await reloadCandidates();
      if (chosenRoles !== null) setStatus("材料列表已刷新。", "info");
    } catch (error) {
      setStatus(error?.message || "材料列表刷新失败。", "error");
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
    activePayload = null;
    resolvePending(value);
    const dialog = $("materialBindingDialog");
    if (dialog?.open) dialog.close(value ? "confirm" : "cancel");
  }

  async function confirmSelection() {
    if (!activeTask?.id) return;
    const selection = collectSelection();
    const issue = materialSelectionIssue(selection, activePayload || {});
    if (issue) {
      setStatus(issue, "error");
      return;
    }
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
    const selectionIssue = materialSelectionIssue(payload.selection || {}, payload);
    if (!force && completeSelection(payload.selection) && !selectionIssue) return task;
    let autoSaveError = "";
    const automatic = force ? null : autoMaterialSelection(payload);
    if (automatic) {
      try {
        const result = await api(`/api/tasks/${task.id}/materials`, {
          method: "PUT",
          body: JSON.stringify(automatic),
        });
        return result?.task || task;
      } catch (error) {
        // Fall back to the dialog and say why the automatic choice failed.
        autoSaveError = error?.message || "材料选择保存失败。";
      }
    }
    previewRequestSequence += 1;
    activeTask = task;
    render(payload);
    const confirmButton = $("materialBindingConfirmButton");
    if (confirmButton) confirmButton.disabled = false;
    const opening = autoSaveError || selectionIssue;
    setStatus(opening, opening ? "error" : "info");
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
    $("materialBindingFileInput")?.addEventListener("change", async (event) => {
      const input = event.target;
      await uploadSupplementFiles(input.files);
      input.value = "";
    });
    $("materialBindingRefreshButton")?.addEventListener("click", refreshCandidates);
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
