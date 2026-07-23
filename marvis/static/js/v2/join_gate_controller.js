import { escapeHtml } from "../ui-utils.js";
import { datasetTableHtml } from "./artifact_view.js";

const DEDUP_STRATEGY_LABELS = { first: "保留首条（first）", last: "保留末条（last）" };
// UX-6: first/last follows raw file row order (not a business timestamp), so the
// picker states that plainly next to the strategy select instead of letting the user
// assume it means something like "most recent".
const DEDUP_STRATEGY_NOTE = "「首条/末条」按当前文件行序保留，行序无业务含义时建议改用聚合或先按时间列排序后再拼接。";

function joinGateContext(context = {}) {
  return {
    taskId: typeof context.getSelectedTaskId === "function"
      ? context.getSelectedTaskId()
      : context.selectedTaskId,
    api: context.api,
    acceptanceMode: typeof context.agentAcceptanceModeValue === "function"
      ? context.agentAcceptanceModeValue()
      : context.acceptanceMode,
    setActionStatus: context.setActionStatus || (() => {}),
    setAgentMessages: context.setAgentMessages || (() => {}),
    renderAgentConversation: context.renderAgentConversation || (() => {}),
    pollAgentMessagesUntilSettled: context.pollAgentMessagesUntilSettled || (() => Promise.resolve()),
    resetFetchThrottle: context.resetFetchThrottle || (() => {}),
    renderWorkflowStepper: context.renderWorkflowStepper || (() => {}),
    setDriverExecutionBusy: context.setDriverExecutionBusy || (() => {}),
  };
}

// UX-1: the driver turn triggered by these gate submissions now runs inside a
// task job (REL-1) and can take minutes (execute_join / retrain downstream of an
// adjust). Give immediate busy feedback, poll agent messages so intermediate
// step content streams in, and force the plan rail to re-fetch on a short
// interval so the running step doesn't look frozen.
function withDriverTurnBusyFeedback(taskId, context, run) {
  const {
    setActionStatus,
    pollAgentMessagesUntilSettled,
    resetFetchThrottle,
    renderWorkflowStepper,
    setDriverExecutionBusy,
  } = context;
  setDriverExecutionBusy(true, taskId);
  setActionStatus("正在执行下一步…", "busy");
  let planRailTimer = null;
  if (typeof setInterval === "function") {
    planRailTimer = setInterval(() => {
      resetFetchThrottle(taskId);
      renderWorkflowStepper({ force: true });
    }, 1500);
  }
  const stopPlanRailTicker = () => {
    if (planRailTimer !== null) clearInterval(planRailTimer);
    setDriverExecutionBusy(false, taskId);
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  };
  return run(pollAgentMessagesUntilSettled).finally(stopPlanRailTicker);
}

export function renderJoinC1Form(message, options = {}) {
  const c1 = message?.metadata?.join_c1;
  if (!c1 || !Array.isArray(c1.files) || !c1.files.length) return "";
  const messageId = message?.id ? String(message.id) : "";
  const gateStepId = message?.metadata?.step_id ? String(message.metadata.step_id) : "";
  // UX-2: earlier C1 forms (superseded by a later gate) render read-only so a
  // stale tab cannot re-submit role assignments against an already-advanced
  // step — mirrors the screen/modeling-setup readonly convention.
  const interactive = options.interactive !== false;
  const disabledAttr = interactive ? "" : " disabled aria-disabled=\"true\"";
  const roleSelect = (datasetId, selected) => {
    const opt = (value, label) =>
      `<option value="${value}"${selected === value ? " selected" : ""}>${label}</option>`;
    return (
      `<select class="c1-role" data-c1-dataset="${escapeHtml(datasetId)}"${disabledAttr}>`
      + opt("anchor", "样本主表")
      + opt("feature", "特征表")
      + opt("ignore", "忽略")
      + "</select>"
    );
  };
  const rows = c1.files
    .map(
      (file) => `<tr>
      <td class="c1-file"><button type="button" class="c1-file-preview" data-c1-preview-dataset="${escapeHtml(file.dataset_id || "")}" data-c1-preview-name="${escapeHtml(file.name || "")}" title="预览前 10 行">${escapeHtml(file.name || "")}</button></td>
      <td>${escapeHtml(String(file.row_count ?? ""))}</td>
      <td>${escapeHtml(String(file.n_cols ?? ""))}</td>
      <td>${file.has_target ? "✓" : ""}</td>
      <td>${roleSelect(file.dataset_id || "", file.proposed_role || "feature")}</td>
    </tr>`,
    )
    .join("");
  const columns = [];
  const seen = new Set();
  for (const file of c1.files) {
    for (const col of file.columns || []) {
      if (!seen.has(col)) {
        seen.add(col);
        columns.push(col);
      }
    }
  }
  const targetOptions = ['<option value="">（不指定）</option>']
    .concat(
      columns.map(
        (col) => `<option value="${escapeHtml(col)}"${col === c1.target_col ? " selected" : ""}>${escapeHtml(col)}</option>`,
      ),
    )
    .join("");
  return `<div class="c1-form" data-c1-form="${escapeHtml(messageId)}" data-c1-gate-step-id="${escapeHtml(gateStepId)}"${interactive ? "" : ' data-c1-readonly="true"'}>
    <table class="c1-form-table">
      <thead><tr><th>文件</th><th>行数</th><th>列数</th><th>含目标</th><th>角色</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="c1-form-foot gate-action-bar">
      <label class="c1-target-label">目标列 <select class="c1-target"${disabledAttr}>${targetOptions}</select></label>
      <button type="button" class="button compact primary c1-confirm"${interactive ? ` data-c1-confirm="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "确认角色" : "历史结果"}</button>
    </div>
  </div>`;
}

function c1PreviewDocument(context = {}) {
  if (context.document) return context.document;
  return typeof document !== "undefined" ? document : null;
}

export async function showC1DatasetPreview(button, context = {}) {
  const doc = c1PreviewDocument(context);
  const dialog = doc?.getElementById?.("c1DatasetPreviewDialog");
  const title = doc?.getElementById?.("c1DatasetPreviewTitle");
  const body = doc?.getElementById?.("c1DatasetPreviewBody");
  const datasetId = button?.dataset?.c1PreviewDataset || "";
  const name = button?.dataset?.c1PreviewName || "数据文件";
  if (!dialog || !title || !body || !datasetId || typeof context.api !== "function") return;

  title.textContent = `${name} · 前 10 行`;
  body.innerHTML = '<div class="c1-preview-loading" role="status">正在读取预览…</div>';
  if (!dialog.open) dialog.showModal();
  try {
    const preview = await context.api(`/api/datasets/${encodeURIComponent(datasetId)}/preview?rows=10`);
    body.innerHTML = datasetTableHtml(preview);
  } catch (error) {
    body.innerHTML = `<div class="c1-preview-error" role="alert">${escapeHtml(error?.message || "预览读取失败")}</div>`;
  }
}

export function closeC1DatasetPreview(context = {}) {
  const dialog = c1PreviewDocument(context)?.getElementById?.("c1DatasetPreviewDialog");
  if (dialog?.open) dialog.close();
}

export function handleC1PreviewClick(event, context = {}) {
  const closeButton = event.target?.closest?.("[data-c1-preview-close]");
  if (closeButton) {
    event.preventDefault();
    closeC1DatasetPreview(context);
    return true;
  }
  const previewButton = event.target?.closest?.("[data-c1-preview-dataset]");
  if (!previewButton) return false;
  event.preventDefault();
  void showC1DatasetPreview(previewButton, context);
  return true;
}

export async function submitC1Assignment(button, rawContext = {}) {
  const form = button.closest(".c1-form");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = joinGateContext(rawContext);
  if (!form || !taskId || typeof api !== "function") return;
  if (form.dataset.c1Readonly === "true") {
    setActionStatus("这是历史拼接角色结果，请使用最新待确认步骤确认。", "error");
    return;
  }
  const anchorIds = [];
  const featureIds = [];
  for (const select of form.querySelectorAll(".c1-role")) {
    const datasetId = select.getAttribute("data-c1-dataset");
    if (select.value === "anchor") anchorIds.push(datasetId);
    else if (select.value === "feature") featureIds.push(datasetId);
  }
  if (!anchorIds.length) {
    setActionStatus("请先把一张表选为「样本主表」。", "error");
    return;
  }
  if (anchorIds.length > 1) {
    setActionStatus("只能有一张样本主表，请把其余表改为「特征表」或「忽略」。", "error");
    return;
  }
  const targetCol = form.querySelector(".c1-target")?.value || "";
  const expectedStepId = form.dataset.c1GateStepId || "";
  button.disabled = true;
  const context = joinGateContext(rawContext);
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const body = {
        content: "[C1]" + JSON.stringify({ anchor_id: anchorIds[0], anchor_ids: anchorIds, feature_ids: featureIds, target_col: targetCol }),
        ui_action: "confirm_roles",
        acceptance_mode: acceptanceMode,
      };
      // C1 is a pre-plan role-assignment gate and therefore normally has no
      // PlanDriver step id.  Keep stale-step protection for any future / legacy
      // C1 message that does carry one, but do not reject the canonical
      // pre-plan contract used by data, feature and modeling workflows.
      if (expectedStepId) body.expected_step_id = expectedStepId;
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
      const result = await requestPromise;
      await streamPollPromise;
      setAgentMessages(result.messages);
      renderAgentConversation();
    });
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "确认角色失败", "error");
  }
}

export function handleC1ConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-c1-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitC1Assignment(button, context);
  return true;
}

function joinKeyRate(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : "n/a";
}

export function renderJoinKeyPicker(message, options = {}) {
  const payload = message?.metadata?.join_keys;
  if (!payload || !Array.isArray(payload.features) || !payload.features.length) return "";
  const interactive = options.interactive !== false;
  const disabledAttr = interactive ? "" : ' disabled aria-disabled="true"';
  const messageId = String(message?.id || "");
  const stepId = String(message?.metadata?.step_id || "");
  const cards = payload.features.map((feature) => {
    const featureId = String(feature.feature_id || "");
    const featureName = String(feature.feature_name || featureId);
    const selected = new Set((feature.selected_anchor_cols || []).map(String));
    const currentKeys = (feature.current_keys || []).map((pair) => {
      const anchorCol = String(pair.anchor_col || "");
      const featureCol = String(pair.feature_col || "");
      return `<label class="join-key-option">
        <input type="checkbox" data-join-key-feature="${escapeHtml(featureId)}" value="${escapeHtml(anchorCol)}"${selected.has(anchorCol) ? " checked" : ""}${disabledAttr}>
        <span><code>${escapeHtml(anchorCol)}</code><span aria-hidden="true"> = </span><code>${escapeHtml(featureCol)}</code></span>
      </label>`;
    }).join("");
    const alternatives = (feature.alternatives || []).map((alternative) => {
      const anchorCols = (alternative.anchor_cols || []).map(String);
      const pairLabel = (alternative.key_pairs || []).map((pair) => `${pair[0]}=${pair[1]}`).join(" + ");
      const risks = [
        alternative.feature_key_unique ? "键唯一" : "键不唯一",
        alternative.fan_out_detected ? "会膨胀" : "不膨胀",
      ].join(" · ");
      return `<button type="button" class="join-key-suggestion${alternative.fan_out_detected ? " is-risk" : ""}"
        data-join-key-suggestion="${escapeHtml(featureId)}"
        data-join-key-columns="${escapeHtml(JSON.stringify(anchorCols))}"${disabledAttr}>
        <span>${escapeHtml(pairLabel || anchorCols.join(" + "))}</span>
        <small>命中 ${escapeHtml(joinKeyRate(alternative.match_rate))} · ${escapeHtml(risks)}</small>
      </button>`;
    }).join("");
    return `<article class="join-key-card" data-join-key-card="${escapeHtml(featureId)}">
      <header><div><strong>${escapeHtml(featureName)}</strong>${featureName !== featureId ? `<small>${escapeHtml(featureId)}</small>` : ""}</div>
        <span>当前命中 ${escapeHtml(joinKeyRate(feature.current_match_rate))}</span></header>
      <div class="join-key-options" role="group" aria-label="${escapeHtml(featureName)} 拼接键">${currentKeys}</div>
      ${alternatives ? `<div class="join-key-alternatives"><span>Agent 候选方案（点击即可选中）</span>${alternatives}</div>` : ""}
    </article>`;
  }).join("");
  return `<section class="join-key-picker" data-join-key-form="${escapeHtml(messageId)}" data-join-key-gate-step-id="${escapeHtml(stepId)}"${interactive ? "" : ' data-join-key-readonly="true"'}>
    <div class="join-key-picker-intro"><strong>逐表确认拼接键</strong><p>这里仍是 ${payload.features.length} 张特征表；每张表只能生成一个最终拼接方案。选择后会先重新诊断，不会立即拼接。</p></div>
    <div class="join-key-card-list">${cards}</div>
    <div class="gate-action-bar">${interactive
    ? `<button type="button" class="button compact primary" data-join-key-confirm="${escapeHtml(messageId)}">应用拼接键并重新诊断</button>`
    : '<span class="gate-history-label" aria-label="历史拼接键结果">历史结果</span>'}</div>
  </section>`;
}

export async function submitJoinKeySelection(button, rawContext = {}) {
  const form = button.closest(".join-key-picker");
  const context = joinGateContext(rawContext);
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = context;
  if (!form || !taskId || typeof api !== "function") return;
  if (form.dataset.joinKeyReadonly === "true") {
    setActionStatus("这是历史拼接键结果，请使用最新待确认步骤。", "error");
    return;
  }
  const keyOverrides = {};
  for (const card of form.querySelectorAll("[data-join-key-card]")) {
    const featureId = card.getAttribute("data-join-key-card");
    const selected = [...card.querySelectorAll("[data-join-key-feature]:checked")].map((input) => input.value);
    if (!selected.length) {
      setActionStatus("每张特征表至少选择一个拼接键。", "error");
      return;
    }
    keyOverrides[featureId] = selected;
  }
  const expectedStepId = form.dataset.joinKeyGateStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息，请刷新后重试。", "error");
    return;
  }
  button.disabled = true;
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: "重新诊断拼接键",
          ui_action: "apply_join_keys",
          adjust_params: { key_overrides: keyOverrides },
          expected_step_id: expectedStepId,
          acceptance_mode: acceptanceMode,
        }),
      });
      const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
      const result = await requestPromise;
      await streamPollPromise;
      setAgentMessages(result.messages);
      renderAgentConversation();
    });
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "拼接键重诊断失败", "error");
  }
}

export function handleJoinKeyConfirmClick(event, context = {}) {
  const suggestion = event.target?.closest?.("[data-join-key-suggestion]");
  if (suggestion) {
    event.preventDefault();
    const form = suggestion.closest(".join-key-picker");
    if (form?.dataset.joinKeyReadonly === "true") return true;
    const featureId = suggestion.getAttribute("data-join-key-suggestion");
    let selected = [];
    try { selected = JSON.parse(suggestion.getAttribute("data-join-key-columns") || "[]"); } catch { selected = []; }
    for (const input of form?.querySelectorAll("[data-join-key-feature]") || []) {
      if (input.getAttribute("data-join-key-feature") === featureId) {
        input.checked = selected.includes(input.value);
      }
    }
    return true;
  }
  const button = event.target?.closest?.("[data-join-key-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitJoinKeySelection(button, context);
  return true;
}

// UX-6: cap the conflicting-column list shown per row so a wide table with many
// disagreeing columns doesn't blow out the picker layout.
const DEDUP_CONFLICT_COLUMNS_DISPLAY_CAP = 5;

// GAP-4: when the task has a registered data dictionary, each conflicting column
// name carries a title tooltip with its business meaning; falls back to the bare
// column name (unchanged behavior) when no dictionary entry exists.
function dedupColumnLabel(column, dictionary) {
  const meaning = dictionary && typeof dictionary === "object" ? dictionary[column] : "";
  return meaning
    ? `<span class="dedup-conflict-column" title="${escapeHtml(String(meaning))}">${escapeHtml(column)}</span>`
    : escapeHtml(column);
}

function dedupConflictColumnsHtml(feature, dictionary) {
  const columns = Array.isArray(feature?.conflict_columns) ? feature.conflict_columns : [];
  if (!columns.length) return "";
  const shown = columns.slice(0, DEDUP_CONFLICT_COLUMNS_DISPLAY_CAP);
  const more = columns.length > shown.length ? ` 等 ${columns.length} 列` : "";
  const labels = shown
    .map((column) => `<span class="dedup-conflict-chip">${dedupColumnLabel(column, dictionary)}</span>`)
    .join("");
  return `<div class="dedup-conflict-columns"><span class="dedup-evidence-label">冲突列</span><div class="dedup-conflict-chips">${labels}${more ? `<span class="dedup-conflict-more">${escapeHtml(more.trim())}</span>` : ""}</div></div>`;
}

// UX-6: one real conflicting-value example per feature (e.g. "k=138... 时 balance
// 两行分别为 0、999"), sourced from the backend's sample_conflicts — replaces the
// previous "conflict_keys number only" black box with a concrete case the user can
// reason about before picking first/last.
function dedupExampleHtml(feature) {
  const examples = Array.isArray(feature?.examples) ? feature.examples : [];
  if (!examples.length) return "";
  const example = examples[0];
  const values = example?.values && typeof example.values === "object" ? example.values : {};
  const valueRows = Object.entries(values).map(([col, rawValues]) => {
    const items = Array.isArray(rawValues) ? rawValues : [rawValues];
    const protectedValue = items.some((item) => /\[REDACTED(?:_[A-Z]+)?\]/.test(String(item)));
    const valueHtml = protectedValue
      ? '<span class="dedup-value-protected">敏感字段，示例值已保护；两行实际值不同</span>'
      : `<span class="dedup-value-pair">${items.map((item) => `<code>${escapeHtml(String(item ?? "空值"))}</code>`).join('<span aria-hidden="true">→</span>')}</span>`;
    return `<div class="dedup-value-row"><span class="dedup-value-column">${escapeHtml(col)}</span>${valueHtml}</div>`;
  }).join("");
  if (!valueRows) return "";
  return `<div class="dedup-example"><div class="dedup-example-key"><span class="dedup-evidence-label">冲突样例</span><code>${escapeHtml(String(example.key || ""))}</code></div><div class="dedup-value-list">${valueRows}</div></div>`;
}

export function renderDedupPicker(message, options = {}) {
  const dedup = message?.metadata?.dedup;
  if (!dedup || !Array.isArray(dedup.features) || !dedup.features.length) return "";
  const messageId = message?.id ? String(message.id) : "";
  const gateStepId = message?.metadata?.step_id ? String(message.metadata.step_id) : "";
  // UX-2: an earlier dedup gate (superseded by a later gate) renders read-only
  // so a stale tab cannot re-submit strategies against an already-advanced
  // step — mirrors the screen/modeling-setup readonly convention.
  const interactive = options.interactive !== false;
  const disabledAttr = interactive ? "" : " disabled aria-disabled=\"true\"";
  const strategies = Array.isArray(dedup.strategies) && dedup.strategies.length ? dedup.strategies : ["first", "last"];
  const cards = dedup.features
    .map((feature) => {
      const fid = String(feature.feature_id);
      const featureName = String(feature.feature_name || fid);
      const conflicts = feature.conflict_keys ? `${feature.conflict_keys} 个同键冲突` : "拼接键不唯一";
      const options = strategies
        .map((strategy) => {
          const value = String(strategy);
          return `<option value="${escapeHtml(value)}">${escapeHtml(DEDUP_STRATEGY_LABELS[value] || value)}</option>`;
        })
        .join("");
      const evidence = dedupConflictColumnsHtml(feature, dedup.dictionary) + dedupExampleHtml(feature);
      // UX-6: "排除该特征表" — an exit for a table whose conflicts are too dirty to
      // resolve with first/last. Submits the same free-text instruction channel the
      // driver already routes adjust/replan through (agent mode acts on it; manual
      // mode — no LLM — shows the existing canned "回复「确认」或调参" hint, which is
      // still an honest, non-broken response).
      return `<article class="dedup-feature-card">
      <header class="dedup-feature-head">
        <div><strong>${escapeHtml(featureName)}</strong>${featureName !== fid ? `<small>${escapeHtml(fid)}</small>` : ""}</div>
        <span class="dedup-conflict-count">${escapeHtml(conflicts)}</span>
      </header>
      <div class="dedup-feature-evidence">${evidence}</div>
      <div class="dedup-feature-actions">
        <label><span>去重策略</span>
        <select class="dedup-strategy" data-dedup-feature="${escapeHtml(fid)}"${disabledAttr}>${options}</select>
        </label>
        <button type="button" class="button compact secondary dedup-exclude" data-dedup-exclude="${escapeHtml(fid)}"${disabledAttr}>排除该特征表</button>
      </div>
    </article>`;
    })
    .join("");
  return `<div class="dedup-picker" data-dedup-form="${escapeHtml(messageId)}" data-dedup-gate-step-id="${escapeHtml(gateStepId)}"${interactive ? "" : ' data-dedup-readonly="true"'}>
    <p class="dedup-note">以下特征表的拼接键不唯一（同键多行），请选择去重策略后再拼接:</p>
    <p class="dedup-strategy-note">${escapeHtml(DEDUP_STRATEGY_NOTE)}</p>
    <div class="dedup-feature-list">${cards}</div>
    <div class="dedup-foot gate-action-bar">
      <button type="button" class="button compact primary dedup-confirm"${interactive ? ` data-dedup-confirm="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "应用去重并确认" : "历史结果"}</button>
    </div>
  </div>`;
}

export async function submitDedupStrategies(button, rawContext = {}) {
  const form = button.closest(".dedup-picker");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = joinGateContext(rawContext);
  if (!form || !taskId || typeof api !== "function") return;
  if (form.dataset.dedupReadonly === "true") {
    setActionStatus("这是历史去重结果，请使用最新待确认步骤确认。", "error");
    return;
  }
  const dedupStrategies = {};
  for (const select of form.querySelectorAll(".dedup-strategy")) {
    const featureId = select.getAttribute("data-dedup-feature");
    if (featureId) dedupStrategies[featureId] = select.value;
  }
  const expectedStepId = form.dataset.dedupGateStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息，请刷新后重试。", "error");
    return;
  }
  button.disabled = true;
  const context = joinGateContext(rawContext);
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: "确认",
          ui_action: "confirm_dedup",
          dedup_strategies: dedupStrategies,
          expected_step_id: expectedStepId,
          acceptance_mode: acceptanceMode,
        }),
      });
      const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
      const result = await requestPromise;
      await streamPollPromise;
      setAgentMessages(result.messages);
      renderAgentConversation();
    });
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "应用去重失败", "error");
  }
}

export function handleDedupConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-dedup-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitDedupStrategies(button, context);
  return true;
}

// UX-6: "排除该特征表" — sends the same free-text instruction channel a typed
// composer message would use (agent mode's instruction router treats it as a
// structural replan dropping the table; manual mode has no LLM router, so the
// driver responds with its existing canned adjust hint rather than applying it
// silently — never a broken request either way).
export async function submitDedupExclude(button, rawContext = {}) {
  const form = button.closest(".dedup-picker");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = joinGateContext(rawContext);
  if (!form || !taskId || typeof api !== "function") return;
  if (form.dataset.dedupReadonly === "true") {
    setActionStatus("这是历史去重结果，请使用最新待确认步骤确认。", "error");
    return;
  }
  const featureId = button.getAttribute("data-dedup-exclude") || "";
  if (!featureId) return;
  const expectedStepId = form.dataset.dedupGateStepId || "";
  button.disabled = true;
  const context = joinGateContext(rawContext);
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: `排除特征表 ${featureId}，其余按当前拼接方案继续`,
          expected_step_id: expectedStepId,
          acceptance_mode: acceptanceMode,
        }),
      });
      const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
      const result = await requestPromise;
      await streamPollPromise;
      setAgentMessages(result.messages);
      renderAgentConversation();
    });
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "排除特征表失败", "error");
  }
}

export function handleDedupExcludeClick(event, context = {}) {
  const button = event.target?.closest?.("[data-dedup-exclude]");
  if (!button) return false;
  event.preventDefault();
  void submitDedupExclude(button, context);
  return true;
}
