import { escapeHtml } from "../ui-utils.js";

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
  };
}

// UX-1: the driver turn triggered by these gate submissions now runs inside a
// task job (REL-1) and can take minutes (execute_join / retrain downstream of an
// adjust). Give immediate busy feedback, poll agent messages so intermediate
// step content streams in, and force the plan rail to re-fetch on a short
// interval so the running step doesn't look frozen.
function withDriverTurnBusyFeedback(taskId, context, run) {
  const { setActionStatus, pollAgentMessagesUntilSettled, resetFetchThrottle, renderWorkflowStepper } = context;
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
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  };
  return run(pollAgentMessagesUntilSettled).finally(stopPlanRailTicker);
}

export function renderJoinC1Form(message, options = {}) {
  const c1 = message?.metadata?.join_c1;
  if (!c1 || !Array.isArray(c1.files) || !c1.files.length) return "";
  const messageId = message?.id ? String(message.id) : "";
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
      <td class="c1-file">${escapeHtml(file.name || "")}</td>
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
  return `<div class="c1-form" data-c1-form="${escapeHtml(messageId)}"${interactive ? "" : ' data-c1-readonly="true"'}>
    <table class="c1-form-table">
      <thead><tr><th>文件</th><th>行数</th><th>列数</th><th>含目标</th><th>角色</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="c1-form-foot">
      <label class="c1-target-label">目标列 <select class="c1-target"${disabledAttr}>${targetOptions}</select></label>
      <button type="button" class="button compact primary c1-confirm"${interactive ? ` data-c1-confirm="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "确认角色" : "历史结果"}</button>
    </div>
  </div>`;
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
  button.disabled = true;
  const context = joinGateContext(rawContext);
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: "[C1]" + JSON.stringify({ anchor_id: anchorIds[0], anchor_ids: anchorIds, feature_ids: featureIds, target_col: targetCol }),
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
  const labels = shown.map((column) => dedupColumnLabel(column, dictionary)).join("、");
  return `<div class="dedup-conflict-columns">冲突列：${labels}${more}</div>`;
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
  const valueParts = Object.entries(values)
    .map(([col, vals]) => `${col} 两行分别为 ${(Array.isArray(vals) ? vals : [vals]).join("、")}`)
    .join("；");
  if (!valueParts) return "";
  return `<div class="dedup-example">示例：k=${escapeHtml(String(example.key || ""))} 时 ${escapeHtml(valueParts)}</div>`;
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
  const rows = dedup.features
    .map((feature) => {
      const fid = String(feature.feature_id);
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
      return `<tr>
      <td class="dedup-feat">${escapeHtml(fid)}${evidence}</td>
      <td>${escapeHtml(conflicts)}</td>
      <td>
        <select class="dedup-strategy" data-dedup-feature="${escapeHtml(fid)}"${disabledAttr}>${options}</select>
        <button type="button" class="button compact secondary dedup-exclude" data-dedup-exclude="${escapeHtml(fid)}"${disabledAttr}>排除该特征表</button>
      </td>
    </tr>`;
    })
    .join("");
  return `<div class="dedup-picker" data-dedup-form="${escapeHtml(messageId)}" data-dedup-gate-step-id="${escapeHtml(gateStepId)}"${interactive ? "" : ' data-dedup-readonly="true"'}>
    <p class="dedup-note">以下特征表的拼接键不唯一（同键多行），请选择去重策略后再拼接:</p>
    <p class="dedup-strategy-note">${escapeHtml(DEDUP_STRATEGY_NOTE)}</p>
    <table class="dedup-table">
      <thead><tr><th>特征表</th><th>冲突</th><th>去重策略</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="dedup-foot">
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
