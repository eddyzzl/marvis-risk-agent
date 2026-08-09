import { previewTaskDataset as fetchTaskDatasetPreview } from "./api_v2.js";

export const DATA_WORKSPACE_FIELD_ROLES = Object.freeze([
  "phone",
  "idcard",
  "id",
  "date",
  "target",
  "score",
  "amount",
  "name",
  "feature",
  "numeric",
  "categorical",
  "loan_amount",
  "overdue_amount",
  "month",
  "rule_node",
  "segment",
  "weight",
  "ignore",
]);

const FIELD_ROLE_LABELS = Object.freeze({
  phone: "手机号",
  idcard: "证件号",
  id: "标识字段",
  date: "日期",
  target: "目标字段",
  score: "分数",
  amount: "金额",
  name: "名称",
  feature: "特征",
  numeric: "数值",
  categorical: "类别",
  loan_amount: "贷款金额",
  overdue_amount: "逾期金额",
  month: "月份",
  rule_node: "规则节点",
  segment: "客群",
  weight: "权重",
  ignore: "忽略",
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function requiredElement(getElementById, id) {
  const element = getElementById(id);
  if (!element) throw new Error(`data workspace panel element is missing: ${id}`);
  return element;
}

function canonicalColumns(values) {
  const seen = new Set();
  const columns = [];
  for (const value of Array.isArray(values) ? values : []) {
    const column = String(value ?? "").trim();
    if (!column || seen.has(column)) continue;
    seen.add(column);
    columns.push(column);
  }
  return columns;
}

function mappedColumns(columns, mapping = {}) {
  return canonicalColumns([
    ...columns,
    mapping.target_col,
    ...Object.keys(mapping.field_roles || {}),
    ...Object.keys(mapping.business_names || {}),
  ]);
}

function roleOptions(currentRole = "") {
  const roles = DATA_WORKSPACE_FIELD_ROLES.includes(currentRole)
    ? DATA_WORKSPACE_FIELD_ROLES
    : [currentRole, ...DATA_WORKSPACE_FIELD_ROLES].filter(Boolean);
  return [
    '<option value="">未标注</option>',
    ...roles.map((role) => (
      `<option value="${escapeHtml(role)}"${role === currentRole ? " selected" : ""}>${escapeHtml(FIELD_ROLE_LABELS[role] || role)} · ${escapeHtml(role)}</option>`
    )),
  ].join("");
}

function fieldRowsHtml(columns, mapping = {}, { disabled = false } = {}) {
  if (!columns.length) {
    return '<p class="data-workspace-empty">绑定数据集后显示字段语义。</p>';
  }
  const roles = mapping.field_roles || {};
  const businessNames = mapping.business_names || {};
  return columns.map((column) => [
    '<div class="data-workspace-field-row">',
    `<code title="${escapeHtml(column)}">${escapeHtml(column)}</code>`,
    `<label><span class="visually-hidden">${escapeHtml(column)} 字段角色</span>`,
    `<select data-workspace-role="${escapeHtml(column)}" aria-label="${escapeHtml(column)} 字段角色"${disabled ? " disabled" : ""}>`,
    roleOptions(String(roles[column] || "")),
    "</select></label>",
    `<label><span class="visually-hidden">${escapeHtml(column)} 业务名称</span>`,
    `<input data-workspace-business-name="${escapeHtml(column)}" value="${escapeHtml(businessNames[column] || "")}" placeholder="业务名称" aria-label="${escapeHtml(column)} 业务名称"${disabled ? " disabled" : ""} />`,
    "</label>",
    "</div>",
  ].join("")).join("");
}

function statusPresentation(state) {
  if (state.loading) return { label: "读取中", tone: "busy" };
  if (state.saving) return { label: "保存中", tone: "busy" };
  if (Number(state.error?.status) === 412) return { label: "版本冲突", tone: "conflict" };
  if (state.error) return { label: "同步失败", tone: "error" };
  if (state.dirty) return { label: "未保存", tone: "dirty" };
  return { label: "已同步", tone: "clean" };
}

function nextMappingForTarget(draft, nextTarget) {
  const mapping = draft.semantic_mapping || {};
  const fieldRoles = { ...(mapping.field_roles || {}) };
  for (const [column, role] of Object.entries(fieldRoles)) {
    if (role === "target") delete fieldRoles[column];
  }
  if (nextTarget) fieldRoles[nextTarget] = "target";
  return {
    ...mapping,
    target_col: nextTarget || null,
    field_roles: fieldRoles,
  };
}

export function createDataWorkspacePanel(dependencies = {}) {
  const getElementById = dependencies.getElementById || ((id) => document.getElementById(id));
  const controller = dependencies.controller;
  if (!controller) throw new TypeError("data workspace panel requires a controller");
  const previewTaskDataset = dependencies.previewTaskDataset || fetchTaskDatasetPreview;
  const resolveNavigationChoice = dependencies.resolveNavigationChoice || (() => "cancel");
  const onError = dependencies.onError || (() => {});

  const elements = {
    root: requiredElement(getElementById, "dataWorkspacePanel"),
    dataset: requiredElement(getElementById, "dataWorkspaceDataset"),
    datasetHash: requiredElement(getElementById, "dataWorkspaceDatasetHash"),
    revision: requiredElement(getElementById, "dataWorkspaceRevision"),
    state: requiredElement(getElementById, "dataWorkspaceState"),
    target: requiredElement(getElementById, "dataWorkspaceTarget"),
    fieldCount: requiredElement(getElementById, "dataWorkspaceFieldCount"),
    fields: requiredElement(getElementById, "dataWorkspaceFields"),
    status: requiredElement(getElementById, "dataWorkspaceStatus"),
    refresh: requiredElement(getElementById, "dataWorkspaceRefreshButton"),
    discard: requiredElement(getElementById, "dataWorkspaceDiscardButton"),
    save: requiredElement(getElementById, "dataWorkspaceSaveButton"),
  };

  let activeTaskId = "";
  let columns = [];
  let previewError = null;
  let notice = "";
  let previewOperation = 0;

  function render() {
    const state = controller.getState();
    const snapshot = state.serverSnapshot;
    const draft = state.draft;
    const mapping = draft?.semantic_mapping || {};
    const datasetId = draft?.active_dataset_id || "";
    const contentHash = draft?.active_dataset_content_hash || "";
    const visibleColumns = mappedColumns(columns, mapping);
    const presentation = statusPresentation(state);
    const inputsDisabled = !draft || !datasetId || state.loading || state.saving;

    elements.root.hidden = !activeTaskId;
    elements.dataset.textContent = datasetId || "尚未绑定数据集";
    elements.dataset.title = datasetId;
    elements.datasetHash.textContent = contentHash ? `${contentHash.slice(0, 12)}…` : "无内容指纹";
    elements.datasetHash.title = contentHash;
    elements.revision.textContent = snapshot ? `R${snapshot.revision}` : "R–";
    elements.revision.title = snapshot
      ? `服务端 revision ${snapshot.revision} · analysis generation ${snapshot.analysis_generation}`
      : "尚未读取服务端版本";
    elements.state.textContent = presentation.label;
    elements.state.className = `data-workspace-state data-workspace-state-${presentation.tone}`;

    elements.target.disabled = inputsDisabled;
    elements.target.innerHTML = [
      '<option value="">未设置目标字段</option>',
      ...visibleColumns.map((column) => (
        `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`
      )),
    ].join("");
    elements.target.value = mapping.target_col || "";

    const mappedCount = new Set([
      ...Object.keys(mapping.field_roles || {}),
      ...Object.keys(mapping.business_names || {}),
    ]).size;
    elements.fieldCount.textContent = `${visibleColumns.length} 个字段 · ${mappedCount} 个已标注`;
    elements.fields.innerHTML = fieldRowsHtml(visibleColumns, mapping, {
      disabled: inputsDisabled,
    });

    elements.refresh.disabled = !activeTaskId || state.loading || state.saving;
    elements.discard.disabled = !state.dirty || state.loading || state.saving;
    elements.save.disabled = !state.dirty || state.loading || state.saving;

    const error = state.error || previewError;
    if (error) {
      elements.status.textContent = error.message || "数据工作区同步失败。";
      elements.status.className = "data-workspace-status error";
    } else if (notice) {
      elements.status.textContent = notice;
      elements.status.className = "data-workspace-status info";
    } else if (state.dirty) {
      elements.status.textContent = "语义修改尚未保存。";
      elements.status.className = "data-workspace-status warning";
    } else {
      elements.status.textContent = datasetId
        ? "工作区与服务端一致。"
        : "上传或由 Agent 绑定数据集后，可在这里维护字段语义。";
      elements.status.className = "data-workspace-status";
    }
  }

  async function loadColumnsForCurrentWorkspace() {
    const state = controller.getState();
    const taskId = state.taskId;
    const datasetId = state.draft?.active_dataset_id;
    const operation = ++previewOperation;
    columns = [];
    previewError = null;
    render();
    if (!taskId || !datasetId) return;
    try {
      const preview = await previewTaskDataset(taskId, datasetId, 1);
      if (operation !== previewOperation || controller.getState().taskId !== taskId) return;
      columns = canonicalColumns(preview?.columns);
    } catch (error) {
      if (operation !== previewOperation) return;
      previewError = error;
    }
    render();
  }

  async function selectTask(taskId) {
    const normalizedTaskId = String(taskId ?? "").trim();
    if (!normalizedTaskId) {
      clear();
      return controller.getState();
    }
    activeTaskId = normalizedTaskId;
    columns = [];
    previewError = null;
    notice = "";
    render();
    await controller.load(normalizedTaskId);
    await loadColumnsForCurrentWorkspace();
    return controller.getState();
  }

  async function reload(taskId = activeTaskId, options = {}) {
    const normalizedTaskId = String(taskId ?? "").trim();
    if (!normalizedTaskId) return false;
    if (controller.isDirty()) {
      notice = "请先保存或丢弃修改，再刷新服务端工作区。";
      render();
      return false;
    }
    try {
      await selectTask(normalizedTaskId);
      return true;
    } catch (error) {
      if (!options.silent) onError(error);
      return false;
    }
  }

  async function save() {
    notice = "";
    previewError = null;
    try {
      await controller.save();
      return true;
    } catch (error) {
      onError(error);
      return false;
    }
  }

  function discard() {
    notice = "";
    previewError = null;
    controller.discard();
    return true;
  }

  async function requestNavigation(navigate) {
    return controller.guardNavigation(
      (state) => resolveNavigationChoice(state),
      async () => {
        if (typeof navigate === "function") await navigate();
      },
    );
  }

  function clear() {
    activeTaskId = "";
    columns = [];
    previewError = null;
    notice = "";
    previewOperation += 1;
    render();
  }

  function editTarget() {
    const draft = controller.getDraft();
    if (!draft) return;
    const target = String(elements.target.value || "").trim() || null;
    controller.edit({
      page: "semantics",
      selected_field: target,
      semantic_mapping: nextMappingForTarget(draft, target),
    });
  }

  function editField(event) {
    const target = event?.target;
    const roleColumn = String(target?.dataset?.workspaceRole || "").trim();
    const businessNameColumn = String(target?.dataset?.workspaceBusinessName || "").trim();
    const column = roleColumn || businessNameColumn;
    const draft = controller.getDraft();
    if (!column || !draft) return;
    const mapping = draft.semantic_mapping || {};
    const fieldRoles = { ...(mapping.field_roles || {}) };
    const businessNames = { ...(mapping.business_names || {}) };
    let targetColumn = mapping.target_col || null;

    if (roleColumn) {
      const role = String(target.value || "").trim();
      if (role === "target") {
        for (const [name, currentRole] of Object.entries(fieldRoles)) {
          if (currentRole === "target") delete fieldRoles[name];
        }
        fieldRoles[column] = "target";
        targetColumn = column;
      } else {
        if (role) fieldRoles[column] = role;
        else delete fieldRoles[column];
        if (targetColumn === column) targetColumn = null;
      }
    } else {
      const businessName = String(target.value || "").trim();
      if (businessName) businessNames[column] = businessName;
      else delete businessNames[column];
    }

    controller.edit({
      page: "semantics",
      selected_field: column,
      semantic_mapping: {
        ...mapping,
        target_col: targetColumn,
        field_roles: fieldRoles,
        business_names: businessNames,
      },
    });
  }

  elements.target.addEventListener("change", editTarget);
  elements.fields.addEventListener("change", editField);
  elements.refresh.addEventListener("click", () => reload());
  elements.discard.addEventListener("click", discard);
  elements.save.addEventListener("click", save);
  const unsubscribe = controller.subscribe(render);
  render();

  return {
    clear,
    destroy: unsubscribe,
    discard,
    getState: () => ({ ...controller.getState(), activeTaskId, columns: [...columns] }),
    reload,
    requestNavigation,
    save,
    selectTask,
  };
}

export default createDataWorkspacePanel;
