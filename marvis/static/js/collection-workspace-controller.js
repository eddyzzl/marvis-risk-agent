import { createCollectionIntakeController } from "./collection-intake-controller.js";
import { api } from "./api.js";
import { escapeHtml as esc } from "./ui-utils.js";
import { taskUsesPlanRail } from "./v2/plan_rail_controller.js";
import { safeSameOriginApiHref } from "./url-safety.js";
import {
  shellHtml,
  listsHtml,
  batchHtml,
  gateHtml,
  details,
  button,
} from "./collection-workspace-view.js";
const path = encodeURIComponent;
const error = (e) =>
  e?.detail?.code
    ? `${e.detail.code} · 请核对来源与当前会话权限`
    : e?.message || "读取失败，请重试";
const goals = {
  queue: "催收参考排队",
  execute: "催收参考执行",
  cancel: "取消催收参考批次",
};
export const batchPlanPayload = (batch, operation) => {
  if (
    !goals[operation] ||
    !(operation === "queue"
      ? batch.status === "proposed"
      : operation === "execute"
        ? batch.status === "queued"
        : ["proposed", "queued"].includes(batch.status))
  )
    throw new Error("当前批次状态已变化，请重新核对");
  return {
    goal: goals[operation],
    slots: {
      collection_batch_id: batch.batch_id,
      collection_request_hash: batch.request_hash,
      collection_preview_hash: batch.preview_hash,
    },
  };
};
export function createCollectionWorkspaceController({
  getElement,
  getTask,
  getPlan,
  captureView,
  isCurrentView,
  apiClient = api,
  onPlanCreated = async () => {},
  refreshPlan = async () => {},
  beginActivity = () => null,
  endActivity = () => {},
  isBusy = () => false,
  openIdentity = () => {},
  downloadBlob = (blob, name) => {
    const href = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = href;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
  },
} = {}) {
  let state = null,
    serial = 0;
  const q = (s) => getElement()?.querySelector(s),
    current = (o) =>
      o === state && isCurrentView(o.view) && getTask()?.id === o.taskId;
  const base = (o) => `/api/tasks/${path(o.taskId)}/collection`;
  const message = (text = "") => {
    const el = q("[data-collection-message]");
    if (el) el.textContent = text;
  };
  const post = (url, body) =>
    apiClient(url, { method: "POST", body: JSON.stringify(body) });
  const lock = () => {
    getElement()
      ?.querySelectorAll(
        '[data-collection-action="queue"], [data-collection-action="execute"], [data-collection-action="cancel"], [data-collection-action="approve"], [data-collection-action="reject"]',
      )
      .forEach((b) => (b.disabled = !!state?.pending || isBusy()));
  };
  function render() {
    const panel = getElement(),
      task = getTask();
    if (!panel) return;
    panel.hidden = !taskUsesPlanRail(task) || task?.task_type !== "strategy";
    if (panel.hidden) {
      state = null;
      panel.innerHTML = "";
      return;
    }
    if (!state || !current(state)) {
      state = {
        serial: ++serial,
        taskId: task.id,
        view: captureView(),
        principal: null,
        cases: [],
        pending: false,
        detailVersion: 0,
        loadVersion: 0,
        batch: null,
        gateSignature: "",
        completed: "",
      };
      panel.innerHTML = shellHtml();
    }
    const plan = getPlan(),
      signature = JSON.stringify([
        plan?.id,
        plan?.status,
        plan?.steps?.map((s) => [s.id, s.status, s.confirmation_snapshot]),
      ]);
    if (signature !== state.gateSignature) {
      state.gateSignature = signature;
      q("[data-collection-gate]").innerHTML = gateHtml(plan);
    }
    lock();
    const completion =
      plan?.status === "done" &&
      plan.steps?.some((s) => s.tool_ref?.plugin === "collection")
        ? plan.id
        : "";
    if (completion && completion !== state.completed) {
      state.completed = completion;
      void load({ preserveDetail: true });
    }
  }
  async function load({ preserveDetail = false } = {}) {
    const owner = state;
    if (!owner) return;
    const version = ++owner.loadVersion,
      detailVersion = owner.detailVersion,
      batchId = owner.batch?.batch_id;
    try {
      const principal = await apiClient("/api/production-governance/me");
      if (!current(owner) || version !== owner.loadVersion) return;
      owner.principal = principal;
      q("[data-collection-identity]").textContent =
        `${principal.display_name} · ${principal.role === "maker" ? "配置人员" : principal.role === "checker" ? "复核人员" : "管理员"}`;
      const [cap, cases, batches] = await Promise.all([
        apiClient(base(owner) + "/capabilities"),
        apiClient(base(owner) + "/cases"),
        apiClient(base(owner) + "/batches"),
      ]);
      if (!current(owner) || version !== owner.loadVersion) return;
      owner.capabilities = cap;
      owner.cases = cases.cases;
      q("[data-collection-intake-actions]").innerHTML =
        principal.role === "maker"
          ? [
              ["new-case", "登记历史案件"],
              ["new-schedule", "登记还款排期"],
              ["new-cashflow", "登记资金观测"],
              ["new-reconcile", "核对资金"],
              ["new-batch", "配置参考批次"],
              ["import-csv", "批量资料映射"],
              ["import-batch", "高级合同导入"],
            ]
              .map(([action, label]) => button(action, label))
              .join("")
          : "";
      q("[data-collection-lists]").innerHTML = listsHtml(
        cases.cases,
        batches.batches,
      );
      message("");
      if (preserveDetail && batchId && detailVersion === owner.detailVersion)
        await read("batch", batchId);
    } catch (e) {
      if (current(owner) && version === owner.loadVersion) {
        if ([401, 403].includes(e.status)) {
          owner.principal = null;
          owner.batch = null;
          ++owner.detailVersion;
          q("[data-collection-identity]").textContent =
            "请绑定可用会话身份后重试";
          q("[data-collection-detail]").innerHTML = "";
          q("[data-collection-lists]").innerHTML = "";
        }
        message(error(e));
      }
    }
  }
  async function read(kind, id) {
    const owner = state;
    if (!owner) return;
    const version = ++owner.detailVersion;
    owner.batch = null;
    q("[data-collection-detail]").innerHTML =
      '<p class="collection-note">正在重新核对当前记录…</p>';
    q("[data-collection-receipt]").innerHTML = "";
    try {
      const result = await apiClient(
        base(owner) + `/${kind === "batch" ? "batches" : "cases"}/${path(id)}`,
      );
      if (!current(owner) || version !== owner.detailVersion) return;
      if (kind === "batch") {
        owner.batch = result;
        q("[data-collection-detail]").innerHTML = batchHtml(
          result,
          owner.principal?.role,
        );
      } else
        q("[data-collection-detail]").innerHTML =
          `<h4>案件 · ${esc(id)}</h4>${details("案件与来源合同", result)}`;
      lock();
      message("");
    } catch (e) {
      if (current(owner) && version === owner.detailVersion) message(error(e));
    }
  }
  async function createPlan(operation) {
    const owner = state;
    if (!owner?.batch) {
      message("请等待当前批次读取完成");
      return;
    }
    if (owner.pending || isBusy()) {
      message("上一步正在收尾，完成后即可创建下一计划");
      return;
    }
    let payload;
    try {
      payload = batchPlanPayload(owner.batch, operation);
    } catch (e) {
      message(error(e));
      return;
    }
    if (owner.principal?.role !== "maker") {
      message("请使用原批次配置人员的会话创建计划");
      return;
    }
    owner.pending = true;
    const activity = beginActivity("collection_plan", owner.taskId);
    lock();
    message("正在生成受治理的执行计划…");
    try {
      const result = await post(
        `/api/tasks/${path(owner.taskId)}/plans`,
        payload,
      );
      if (!current(owner)) return;
      await onPlanCreated(result.plan, owner.view);
      if (current(owner))
        message(
          "计划已生成，请在现有执行计划中开始，再由独立复核人员授权本次步骤。",
        );
    } catch (e) {
      if (current(owner)) message(error(e));
    } finally {
      owner.pending = false;
      endActivity(activity);
      if (current(owner)) lock();
    }
  }
  async function review(decision) {
    const owner = state,
      plan = getPlan(),
      step = plan?.steps?.find(
        (s) =>
          s.status === "awaiting_confirm" &&
          s.tool_ref?.plugin === "collection",
      );
    if (!owner || !step || owner.pending || isBusy()) return;
    const reason = q("[name=collection_decision_reason]")?.value.trim();
    if (!reason) {
      message("请填写本次人工复核理由");
      return;
    }
    if (!step.confirmation_snapshot?.expected_step_fingerprint) {
      message("当前计划快照缺失，请刷新后重试");
      return;
    }
    owner.pending = true;
    const activity = beginActivity("collection_review", owner.taskId);
    lock();
    try {
      await post(
        `/api/plans/${path(plan.id)}/steps/${path(step.id)}/decisions`,
        { decision, reason, ...step.confirmation_snapshot },
      );
      if (!current(owner)) return;
      await refreshPlan(owner.taskId);
      if (current(owner)) {
        await load({ preserveDetail: true });
        if (current(owner))
          message(
            decision === "approve"
              ? "本次授权已记录，请核对平台执行回执。"
              : "本次拒绝已记录。",
          );
      }
    } catch (e) {
      if (current(owner)) message(error(e));
    } finally {
      owner.pending = false;
      endActivity(activity);
      if (current(owner)) lock();
    }
  }
  async function evidence(raw, download = false) {
    const owner = state;
    if (!owner) return;
    const safe = safeSameOriginApiHref(raw);
    const url = safe
      ? new URL(`/${safe}`, "https://marvis.invalid").pathname
      : null;
    if (!url?.startsWith(base(owner) + "/batches/")) {
      message("回执地址不属于当前任务");
      return;
    }
    const version = ++owner.detailVersion;
    try {
      if (download) {
        const blob = await apiClient(url + "/export", { responseType: "blob" });
        if (current(owner) && version === owner.detailVersion)
          downloadBlob(blob, "collection-reference-receipt.json");
      } else {
        const result = await apiClient(url);
        if (current(owner) && version === owner.detailVersion)
          q("[data-collection-receipt]").innerHTML =
            `<h4>已校验原生执行回执</h4><p class="collection-note">本地参考模拟；执行记录不表示联系了客户或产生实际回收。</p>${button("export", "导出同一冻结回执", url)}${details("查看执行事实与认证来源", result)}`;
      }
    } catch (e) {
      if (current(owner) && version === owner.detailVersion) message(error(e));
    }
  }
  const intake = createCollectionIntakeController({
    getRoot: getElement,
    getOwner: () => state,
    isCurrent: current,
    apiClient,
    message,
    downloadBlob,
    setPending: (owner, pending) => {
      owner.pending = pending;
      if (current(owner)) {
        if (pending) {
          owner.intakeDisabled = new Map();
          getElement()
            ?.querySelectorAll(
              "[data-collection-form] input, [data-collection-form] select, [data-collection-form] button",
            )
            .forEach((el) => {
              owner.intakeDisabled.set(el, el.disabled);
              el.disabled = true;
            });
        } else {
          owner.intakeDisabled?.forEach((disabled, el) => {
            el.disabled = disabled;
          });
          owner.intakeDisabled = null;
        }
        lock();
      }
    },
    refresh: () => load(),
    onBatch: (id) => read("batch", id),
  });
  function handle(event) {
    if (!getElement()?.contains(event.target)) return false;
    if (intake.handle(event)) return true;
    if (event.type !== "click") return false;
    const control = event.target.closest("[data-collection-action]");
    if (!control) return false;
    event.preventDefault();
    const { collectionAction: action, collectionId: id } = control.dataset;
    if (action === "identity") {
      openIdentity();
      return true;
    }
    if (state?.pending) {
      message("正在提交，请等待当前操作完成");
      return true;
    }
    if (action === "load") void load({ preserveDetail: true });
    else if (["case", "batch"].includes(action)) void read(action, id);
    else if (goals[action]) void createPlan(action);
    else if (["approve", "reject"].includes(action)) void review(action);
    else if (["evidence", "export"].includes(action))
      void evidence(id, action === "export");
    return true;
  }
  return { render, handle, load };
}
