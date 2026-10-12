import { apiGet, apiPost } from "../api.js";

const DIRECTIONS = Object.freeze({
  unknown: "未知 / 无法确认", higher_is_riskier: "越高风险越高",
  lower_is_riskier: "越低风险越高", non_monotonic: "非单调",
});
const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const base = (taskId) => `/api/tasks/${encodeURIComponent(taskId)}/data-workspace/field-directions`;

export function fieldDirectionHtml(column, draft, { disabled = false } = {}) {
  const declaration = draft?.semantic_mapping?.risk_directions?.[column];
  const current = Boolean(declaration && draft?.active_dataset_id && draft?.active_dataset_content_hash
    && declaration.dataset_id === draft.active_dataset_id
    && declaration.dataset_content_hash === draft.active_dataset_content_hash);
  const direction = current ? declaration.direction : "unknown";
  const source = current ? "用户已确认" : declaration ? "新数据待确认" : "未声明";
  return `<div class="data-workspace-direction-cell"><label><span>风险方向 · ${escapeHtml(source)}</span><select data-field-risk-direction="${escapeHtml(column)}" aria-label="${escapeHtml(column)} 风险方向"${disabled ? " disabled" : ""}>`
    + Object.entries(DIRECTIONS).map(([key, label]) => `<option value="${key}"${key === direction ? " selected" : ""}>${label}</option>`).join("")
    + `</select></label><button type="button" data-confirm-field-direction="${escapeHtml(column)}"${disabled ? " disabled" : ""}>确认方向</button></div>`;
}

export function createFieldDirectionPanel({ element, controller, onChange = () => {}, onError = () => {},
  getDirections = (taskId) => apiGet(base(taskId)),
  confirmDirections = (taskId, body) => apiPost(`${base(taskId)}/confirm`, body) }) {
  let busy = false;
  async function confirm(column, direction) {
    const state = controller.getState();
    if (busy || state.dirty || state.loading || state.saving || !state.taskId) return false;
    if (!Object.hasOwn(DIRECTIONS, direction)) throw new Error("请明确选择风险方向。");
    const taskId = state.taskId;
    const revision = state.serverSnapshot?.revision;
    busy = true;
    onChange();
    try {
      const projection = await getDirections(taskId);
      const latest = controller.getState();
      if (latest.taskId !== taskId || latest.dirty || latest.serverSnapshot?.revision !== revision
        || projection.revision !== revision || projection.dataset_id !== latest.draft?.active_dataset_id
        || projection.dataset_content_hash !== latest.draft?.active_dataset_content_hash) {
        throw new Error("字段语义或活动任务已变化，请刷新后重新确认方向。");
      }
      if (!projection.fields.some((field) => field.column === column)) throw new Error("请选择当前数据的物理字段。");
      await confirmDirections(taskId, { expected_revision: projection.revision,
        expected_semantic_mapping_hash: projection.semantic_mapping_hash,
        dataset_id: projection.dataset_id, dataset_content_hash: projection.dataset_content_hash,
        directions: { [column]: direction } });
      if (controller.getState().taskId === taskId && !controller.isDirty()) await controller.load(taskId);
      return true;
    } catch (error) {
      onError(error);
      return false;
    } finally {
      busy = false;
      onChange();
    }
  }
  function click(event) {
    const column = event?.target?.dataset?.confirmFieldDirection;
    if (!column) return;
    const select = [...(element.querySelectorAll?.("[data-field-risk-direction]") || [])]
      .find((item) => item.dataset.fieldRiskDirection === column);
    if (select) void confirm(column, select.value);
  }
  element.addEventListener("click", click);
  return { confirm, isBusy: () => busy, destroy: () => element.removeEventListener?.("click", click) };
}
