import { api } from "./api.js";
import { escapeHtml as esc } from "./ui-utils.js";
import { taskUsesPlanRail } from "./v2/plan_rail_controller.js";
import {
  replayFormHtml,
  featureFieldsHtml,
  constraintHtml,
  temporalWindowHtml,
  collectReplay,
  collectReconciliation,
  options,
} from "./historical-replay-form.js";
import {
  proposalHtml,
  receiptListHtml,
  receiptHtml,
  reconciliationFormHtml,
  replayGateHtml,
} from "./historical-replay-view.js";

const path = encodeURIComponent;
const formValues = (form) =>
  Object.fromEntries(
    Array.from(form.querySelectorAll("[name]")).map((input) => [
      input.name,
      input.value,
    ]),
  );
const errorText = (e) =>
  e?.detail?.next_action
    ? `${e.detail.next_action}（${e.detail.code || "请求失败"}）`
    : e?.message || "请求失败，请重试";
export function createHistoricalReplayController({
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
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
  },
} = {}) {
  let state = null,
    serial = 0;
  const q = (selector) => getElement()?.querySelector(selector);
  const current = (owner, ticket = owner?.serial) =>
    owner === state &&
    owner?.serial === ticket &&
    isCurrentView(owner.view) &&
    getTask()?.id === owner.taskId;
  const message = (text = "") => {
    const el = q("[data-history-error]");
    if (el) el.textContent = text;
  };
  const post = (url, body) =>
    apiClient(url, { method: "POST", body: JSON.stringify(body) });
  function invalidate() {
    if (!state) return;
    const wasPrepared = state.proposal || state.pending;
    state.editVersion++;
    state.proposal = null;
    const el = q("[data-history-proposal]");
    if (el) el.innerHTML = "";
    if (wasPrepared) message("口径已改变，请重新校验并生成提案。");
  }
  function render() {
    const panel = getElement(),
      task = getTask();
    if (!panel) return;
    panel.hidden = !taskUsesPlanRail(task);
    if (panel.hidden) {
      state = null;
      panel.innerHTML = "";
      return;
    }
    if (!state || !current(state)) {
      state = {
        taskId: task.id,
        view: captureView(),
        serial: ++serial,
        loaded: false,
        loading: false,
        datasets: [],
        packages: [],
        records: [],
        featureNames: [],
        packageSignature: "",
        proposal: null,
        editVersion: 0,
        pending: false,
        gateSignature: "",
        completedPlan: "",
      };
      panel.innerHTML =
        '<details data-history-panel><summary>历史决策回放 <span>比较冻结方案，核对成熟现金流</span></summary><div class="history-body"><p class="history-note">准备历史快照后，可校验时点、复核提案并查看冻结回执。</p><div class="history-actions"><button type="button" class="button compact secondary" data-history-action="load">准备回放</button><button type="button" class="button compact secondary" data-history-action="refresh">刷新回执</button></div><p data-history-error class="history-error" role="alert"></p><div data-history-editor></div><div data-history-proposal></div><div data-history-gate></div><h4>已完成的冻结回执</h4><div data-history-receipts></div><div data-history-detail></div></div></details>';
    }
    const plan = getPlan();
    const signature = JSON.stringify([
      plan?.id,
      plan?.status,
      plan?.steps?.map((s) => [s.id, s.status, s.confirmation_snapshot]),
    ]);
    if (signature !== state.gateSignature) {
      state.gateSignature = signature;
      q("[data-history-gate]").innerHTML = replayGateHtml(plan);
    }
    q("[data-history-gate]")
      .querySelectorAll("button")
      .forEach((button) => {
        button.disabled = state.pending || isBusy();
      });
    const completion =
      plan?.status === "done" &&
      plan.steps?.some((s) => s.tool_ref?.plugin === "decision_twin")
        ? plan.id
        : "";
    if (completion && completion !== state.completedPlan) {
      state.completedPlan = completion;
      void receipts();
    }
  }
  async function receipts() {
    const owner = state;
    if (!owner) return;
    const version = (owner.receiptVersion = (owner.receiptVersion || 0) + 1);
    try {
      const result = await apiClient(
        `/api/tasks/${path(owner.taskId)}/decision-twin`,
      );
      if (!current(owner) || version !== owner.receiptVersion) return;
      owner.records = result.artifacts;
      q("[data-history-receipts]").innerHTML = receiptListHtml(
        result.artifacts,
      );
    } catch (e) {
      if (current(owner) && version === owner.receiptVersion)
        message(errorText(e));
    }
  }
  async function packageList() {
    let items = [],
      offset = 0;
    do {
      const result = await apiClient(
        `/api/reference-decision/packages?limit=100&offset=${offset}`,
      );
      items.push(...result.packages);
      offset = result.next_offset;
    } while (offset != null);
    return items;
  }
  async function load() {
    const owner = state;
    if (!owner || owner.loading || owner.pending) return;
    owner.loading = true;
    message("正在读取当前任务快照和冻结方案…");
    try {
      const [data, packages, caps] = await Promise.all([
        apiClient(`/api/tasks/${path(owner.taskId)}/datasets`),
        packageList(),
        apiClient("/api/decision-twin/capabilities"),
      ]);
      if (!current(owner)) return;
      if (caps.schema_version !== "decision_twin.capabilities.v2")
        throw new Error("当前历史回放能力版本不受支持");
      owner.datasets = data.datasets;
      owner.packages = packages;
      owner.loaded = true;
      owner.featureNames = [];
      owner.packageSignature = "";
      invalidate();
      q("[data-history-editor]").innerHTML = replayFormHtml(
        owner.datasets,
        owner.packages,
      );
      message("");
      await receipts();
    } catch (e) {
      if (current(owner)) {
        message(errorText(e));
        if (e.status === 403)
          q("[data-history-editor]").innerHTML =
            '<p class="history-note">查看冻结包需要工作区会话身份。</p><button type="button" class="button compact secondary" data-history-action="identity">绑定会话身份</button>';
      }
    } finally {
      if (current(owner)) owner.loading = false;
    }
  }
  function datasetChanged(form) {
    const dataset = state.datasets.find(
      (d) => d.id === form.elements.dataset_id.value,
    );
    const columns = (dataset?.columns || []).map((c) => ({
      id: typeof c === "string" ? c : c.name,
    }));
    const columnNames = new Set([
      "record_id_col",
      "decision_at_col",
      "action_col",
      "recorded_at_col",
      "ead_col",
      "economics_available_at_col",
      "group_column",
      "group_available_at_col",
      "observed_at_col",
      "actual_loss_col",
      "actual_profit_col",
    ]);
    form.querySelectorAll("select[name]").forEach((input) => {
      if (columnNames.has(input.name)) input.innerHTML = options(columns);
    });
    const source = form.querySelector("[data-history-source]");
    source.textContent = dataset
      ? `快照 ${dataset.id} · ${dataset.row_count} 行 · 内容指纹 ${dataset.content_hash}`
      : "请选择当前任务快照";
    const features = form.querySelector("[data-history-feature-fields]");
    if (features)
      features.innerHTML = state.featureNames.length
        ? featureFieldsHtml(state.featureNames, columns)
        : "";
  }
  const packageSignature = (form) =>
    JSON.stringify(
      ["baseline", "challenger", "counterfactual"].map(
        (name) => form.elements[name].value,
      ),
    );
  async function features() {
    const owner = state,
      form = q('[data-history-form="replay"]');
    if (!owner || !form || owner.pending) return;
    const selected = ["baseline", "challenger", "counterfactual"]
        .map((n) => form.elements[n].value)
        .filter(Boolean),
      signature = packageSignature(form),
      edit = owner.editVersion;
    if (selected.length < 2) {
      message("请先选择基线和挑战方案");
      return;
    }
    message("正在校验冻结方案字段…");
    try {
      const manifests = await Promise.all(
        selected.map((hash) =>
          apiClient(`/api/reference-decision/packages/${path(hash)}`),
        ),
      );
      if (
        !current(owner) ||
        owner.editVersion !== edit ||
        packageSignature(form) !== signature
      )
        return;
      const nodes = new Set(
        manifests.map((r) => r.manifest.configuration.decision_node),
      );
      if (nodes.size !== 1) throw new Error("方案决策节点不同，请重新选择");
      const schemas = new Set(
        manifests.map((r) =>
          JSON.stringify(
            r.manifest.configuration.raw_schema.map((f) => f.name).sort(),
          ),
        ),
      );
      if (schemas.size !== 1)
        throw new Error("方案输入字段不同，无法在同一历史字段合同中比较");
      owner.featureNames = [
        ...new Set(
          manifests.flatMap((r) =>
            r.manifest.configuration.raw_schema.map((f) => f.name),
          ),
        ),
      ];
      owner.packageSignature = signature;
      const dataset = owner.datasets.find(
          (d) => d.id === form.elements.dataset_id.value,
        ),
        columns = (dataset?.columns || []).map((c) => ({
          id: typeof c === "string" ? c : c.name,
        }));
      form.querySelector("[data-history-feature-fields]").innerHTML =
        featureFieldsHtml(owner.featureNames, columns);
      form.querySelector("[data-history-package-message]").textContent =
        `已校验 ${manifests.length} 个方案 · ${owner.featureNames.length} 个输入字段。请明确绑定各字段时点。`;
      message("");
    } catch (e) {
      if (current(owner) && owner.editVersion === edit) message(errorText(e));
    }
  }
  function collect(form) {
    const values = formValues(form);
    if (form.dataset.historyForm === "reconciliation")
      return collectReconciliation(values, {
        taskId: state.taskId,
        datasets: state.datasets,
        artifactId: form.dataset.historyId,
      });
    if (packageSignature(form) !== state.packageSignature)
      throw new Error("方案已变化，请重新读取字段合同");
    return collectReplay(values, {
      taskId: state.taskId,
      datasets: state.datasets,
      packages: state.packages,
      features: Array.from(form.querySelectorAll("[data-history-feature]")).map(
        (row) => ({ name: row.dataset.historyFeature, ...formValues(row) }),
      ),
      constraints: Array.from(
        form.querySelectorAll("[data-history-constraint]"),
      ).map(formValues),
      windows: Array.from(form.querySelectorAll("[data-history-window]")).map(
        (row) => ({ kind: row.dataset.historyWindow, ...formValues(row) }),
      ),
      enabled: Array.from(
        form.querySelectorAll("[data-history-enable]:checked"),
      ).map((i) => i.dataset.historyEnable),
    });
  }
  async function propose(form) {
    const owner = state;
    if (owner.pending) return;
    let contract;
    try {
      contract = collect(form);
    } catch (e) {
      message(errorText(e));
      return;
    }
    const edit = owner.editVersion,
      kind = form.dataset.historyForm;
    owner.pending = true;
    message("正在校验本次数据、时点与来源…");
    try {
      const result = await post(
        `/api/tasks/${path(owner.taskId)}/decision-twin/${kind === "replay" ? "proposal" : "reconciliation-proposal"}`,
        contract,
      );
      if (!current(owner) || edit !== owner.editVersion) return;
      owner.proposal = { result, kind, edit };
      q("[data-history-proposal]").innerHTML = proposalHtml(result, kind);
      message("");
    } catch (e) {
      if (current(owner) && edit === owner.editVersion) message(errorText(e));
    } finally {
      if (current(owner)) owner.pending = false;
    }
  }
  async function plan() {
    const owner = state,
      proposal = owner?.proposal;
    if (!proposal || owner.pending || isBusy()) return;
    if (proposal.edit !== owner.editVersion) {
      message("表单已改变，请重新生成提案");
      return;
    }
    owner.pending = true;
    const activity = beginActivity("historical_plan", owner.taskId);
    message("正在创建受治理执行计划…");
    try {
      const key =
        proposal.kind === "replay"
          ? "replay_contract"
          : "reconciliation_contract";
      const result = await post(`/api/tasks/${path(owner.taskId)}/plans`, {
        goal:
          proposal.kind === "replay" ? "历史决策回放" : "历史决策现金流对账",
        slots: {
          [key]: proposal.result.contract,
          proposal_hash: proposal.result.proposal_hash,
        },
      });
      if (!current(owner)) return;
      owner.proposal = null;
      q("[data-history-proposal]").innerHTML = "";
      await onPlanCreated(result.plan, owner.view);
      if (current(owner))
        message("计划已生成，请在执行计划中开始，随后复核并授权本次回放。");
    } catch (e) {
      if (current(owner)) message(errorText(e));
    } finally {
      owner.pending = false;
      endActivity(activity);
    }
  }
  async function decision(action) {
    const owner = state,
      plan = getPlan(),
      step = plan?.steps?.find(
        (s) =>
          s.status === "awaiting_confirm" &&
          s.tool_ref?.plugin === "decision_twin",
      );
    if (!owner || !step) return;
    if (owner.pending || isBusy()) {
      message("上一步仍在收尾，完成后即可提交复核；已填写的理由会保留。");
      return;
    }
    const reason = q(
      "[data-history-gate] [name=decision_reason]",
    )?.value.trim();
    if (!reason) {
      message("请填写本次人工复核理由");
      return;
    }
    const snapshot = step.confirmation_snapshot;
    if (!snapshot?.expected_step_fingerprint) {
      message("计划快照缺失，请刷新后重试");
      return;
    }
    owner.pending = true;
    const activity = beginActivity("historical_decision", owner.taskId);
    message("正在提交本次复核决定…");
    try {
      await post(
        `/api/plans/${path(plan.id)}/steps/${path(step.id)}/decisions`,
        { decision: action, reason, ...snapshot },
      );
      if (current(owner)) {
        await refreshPlan(owner.taskId);
        if (current(owner)) {
          await receipts();
          if (current(owner))
            message(
              action === "approve"
                ? "授权已记录，执行结果将显示为冻结回执。"
                : "拒绝决定已记录。",
            );
        }
      }
    } catch (e) {
      if (current(owner)) {
        message(errorText(e));
        await refreshPlan(owner.taskId);
      }
    } finally {
      owner.pending = false;
      endActivity(activity);
    }
  }
  async function detail(id) {
    const owner = state,
      version = (owner.detailVersion = (owner.detailVersion || 0) + 1);
    message("");
    try {
      const result = await apiClient(
        `/api/tasks/${path(owner.taskId)}/decision-twin/${path(id)}`,
      );
      if (current(owner) && version === owner.detailVersion) {
        owner.receipt = result;
        q("[data-history-detail]").innerHTML = receiptHtml(result);
      }
    } catch (e) {
      if (current(owner) && version === owner.detailVersion)
        message(errorText(e));
    }
  }
  async function download(id, format) {
    const owner = state;
    if (!["json", "xlsx", "docx"].includes(format)) return;
    try {
      const blob = await apiClient(
        `/api/tasks/${path(owner.taskId)}/decision-twin/${path(id)}/export/${format}`,
        { responseType: "blob" },
      );
      if (current(owner))
        downloadBlob(blob, `historical-replay-${id.slice(0, 12)}.${format}`);
    } catch (e) {
      if (current(owner)) message(errorText(e));
    }
  }
  function handle(event) {
    const panel = getElement();
    if (!panel?.contains(event.target) || !state) return false;
    if (event.type === "submit") {
      const form = event.target.closest("[data-history-form]");
      if (!form) return false;
      event.preventDefault();
      void propose(form);
      return true;
    }
    if (event.type === "input" || event.type === "change") {
      const form = event.target.closest("[data-history-form]");
      if (!form) return false;
      invalidate();
      if (event.type === "change" && event.target.name === "dataset_id")
        datasetChanged(form);
      if (event.target.matches("[data-history-enable]")) {
        const name = event.target.dataset.historyEnable;
        form.querySelector(`[data-history-fields="${name}"]`).disabled =
          !event.target.checked;
        event.target.closest("details").open = event.target.checked;
      }
      return true;
    }
    const button = event.target.closest("[data-history-action]");
    if (!button) return false;
    event.preventDefault();
    if (state.pending) {
      message("操作正在提交，请等待完成");
      return true;
    }
    const action = button.dataset.historyAction;
    if (action === "load") void load();
    if (action === "identity") openIdentity();
    if (action === "refresh") void receipts();
    if (action === "features") void features();
    if (action === "plan") void plan();
    if (action === "add-constraint") {
      q("[data-history-constraints]").insertAdjacentHTML(
        "beforeend",
        constraintHtml(),
      );
      invalidate();
    }
    if (action === "add-window") {
      q("[data-history-windows]").insertAdjacentHTML(
        "beforeend",
        temporalWindowHtml("comparison"),
      );
      invalidate();
    }
    if (action === "remove-window") {
      button.closest("[data-history-window]").remove();
      invalidate();
    }
    if (action === "remove-constraint") {
      button.closest("[data-history-constraint]").remove();
      invalidate();
    }
    if (action === "receipt") void detail(button.dataset.historyId);
    if (action === "download")
      void download(button.dataset.historyId, button.dataset.historyFormat);
    if (action === "reconcile") {
      invalidate();
      q("[data-history-editor]").innerHTML = reconciliationFormHtml(
        state.datasets,
        button.dataset.historyId,
      );
    }
    if (action === "approve" || action === "reject") void decision(action);
    return true;
  }
  return { render, handle };
}
