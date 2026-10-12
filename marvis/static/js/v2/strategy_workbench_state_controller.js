/** Server drafts are user intent. Native execution checkpoints remain on the server. */
import { apiGet, apiPost, apiPut } from '../api.js';
import { escapeHtml } from '../ui-utils.js';
import { loadStrategyCandidateLabViewState } from './strategy_candidate_lab_state.js';

export const WORKBENCH_DRAFT_SCHEMA = 'strategy.workbench-draft.v1';

function kind(field) {
  if (field.readOnly) return 'reference';
  if (field.tagName?.toLowerCase() === 'select') return 'select';
  if (['checkbox', 'radio'].includes(field.type)) return 'checks';
  if (field.tagName?.toLowerCase() === 'textarea' || (field.type || 'text') === 'text') return 'narrative';
  return 'scalar';
}

export function captureWorkbenchDraft(root) {
  const grouped = new Map();
  const open = new Set();
  for (const form of root?.querySelectorAll?.('[data-candidate-lab-form]') || []) {
    const workflow = form.dataset.candidateLabWorkflow;
    if (form.closest?.('.candidate-lab-launcher')?.open) open.add(workflow);
    for (const field of form.querySelectorAll('[data-candidate-lab-field]')) {
      if (['file', 'password'].includes(field.type)) continue;
      const identity = `${workflow}:${field.dataset.candidateLabField}`;
      const entry = grouped.get(identity) || { workflow, field: field.dataset.candidateLabField, kind: kind(field), values: [] };
      const values = entry.kind === 'checks' ? (field.checked ? [field.value] : [])
        : entry.kind === 'select' && field.multiple ? Array.from(field.selectedOptions || []).map(o => o.value) : [field.value || ''];
      entry.values = Array.from(new Set([...entry.values, ...values]));
      grouped.set(identity, entry);
    }
  }
  return { schema_version: WORKBENCH_DRAFT_SCHEMA, fields: Array.from(grouped.values()), open_workflows: Array.from(open), scroll_top: Math.round(root?.scrollTop || 0) };
}

export function restoreWorkbenchDraft(root, draft) {
  if (draft?.schema_version !== WORKBENCH_DRAFT_SCHEMA) return false;
  const entries = new Map(draft.fields.map(e => [`${e.workflow}:${e.field}`, e]));
  for (const form of root?.querySelectorAll?.('[data-candidate-lab-form]') || []) {
    const workflow = form.dataset.candidateLabWorkflow;
    const details = form.closest?.('.candidate-lab-launcher');
    if (details) details.open = draft.open_workflows.includes(workflow);
    for (const field of form.querySelectorAll('[data-candidate-lab-field]')) {
      const entry = entries.get(`${workflow}:${field.dataset.candidateLabField}`);
      if (!entry || entry.kind !== kind(field) || field.readOnly || field.type === 'hidden') continue;
      if (entry.kind === 'checks') field.checked = entry.values.includes(field.value);
      else if (entry.kind === 'select') {
        const allowed = new Set(Array.from(field.options || []).map(o => o.value));
        const values = entry.values.filter(v => allowed.has(v));
        if (field.multiple) for (const option of field.options || []) option.selected = values.includes(option.value);
        else field.value = values[0] || '';
      } else field.value = entry.values[0] || '';
    }
  }
  if (root) root.scrollTop = draft.scroll_top || 0;
  return true;
}

function errorCode(error) {
  return error?.detail?.code || error?.payload?.detail?.code;
}

function publicError(error, fallback) {
  const code = errorCode(error);
  const copy = {
    workbench_evidence_changed: '输入或证据已经变化。请刷新证据，重新选择后启动新的搜索。',
    workbench_evidence_verification_failed: '当前证据未通过核验。请在证据区检查来源，再恢复草稿或启动搜索。',
    workbench_draft_sensitive_or_oversized: '草稿含敏感信息或超长内容。请移除客户明细、凭据和连接信息后再保存。',
    workbench_draft_choice_invalid: '选择不在当前表单的固定选项中。请重新选择后保存草稿。',
    workbench_draft_conflict: '另一页面已保存新版本。请选择读取服务端草稿，或保留本页并读取最新版本号。',
    workbench_search_conflict: '搜索进度已推进，请刷新后再次操作。',
    workbench_plan_not_ready: '搜索正在收尾。稍后刷新后再继续。',
    workbench_resume_requires_native_checkpoint: '检查点与当前运行实现不再匹配。请保留现有证据并启动新的搜索。',
    workbench_search_control_unavailable: '当前阶段无法执行该操作。正在发布证据时，请等待完成。',
    workbench_actor_inactive: '当前用户身份已停用，请重新建立有效会话。',
  };
  return copy[code] || fallback;
}

function invalidateDraftSelections(draft, staticChoices = []) {
  const invalidated = [];
  if (!draft) return { draft, invalidated };
  const installed = new Map(staticChoices.map(field => [`${field.workflow}:${field.field}`, new Set(field.values)]));
  const fields = draft.fields.map(field => {
    const meaningful = field.values.filter(Boolean);
    const choices = installed.get(`${field.workflow}:${field.field}`);
    if (['select', 'checks', 'reference'].includes(field.kind)
        && meaningful.length && !(choices && meaningful.every(value => choices.has(value)))
        && !(field.workflow === 'strategy_workbench_preferences' && field.field === 'field_display_mode'
          && meaningful.length === 1 && ['physical', 'business', 'both'].includes(meaningful[0]))) {
      invalidated.push({ workflow: field.workflow, field: field.field });
      return { ...field, values: [] };
    }
    return field;
  });
  return { draft: { ...draft, fields }, invalidated };
}

const STATE_COPY = { queued: '等待执行', running: '计算中', pause_requested: '正在等待当前候选完成后暂停', paused: '已暂停，检查点已保存', cancel_requested: '正在取消', cancelled: '已取消', resuming: '正在继续', publishing: '正在发布证据', published: '正在核验发布凭据', completed: '已完成', failed: '执行失败', uncertain: '发布结果待核对' };
const TOOL_COPY = { 'strategy.search_voting_candidates': 'Voting 组合搜索', 'strategy.search_cross_threshold_rules': 'Cross 阈值搜索', 'strategy.search_cross_matrix_candidates': 'Cross Matrix 搜索', 'strategy.search_interactive_tree_split_candidates': '交互树分裂搜索' };

export function createStrategyWorkbenchStateController({ getElementById, getPanel, getTaskId, onRestored, onExecutionSettled, get = apiGet, put = apiPut, post = apiPost, schedule = setTimeout, cancelSchedule = clearTimeout } = {}) {
  const slot = () => getElementById?.('strategyWorkbenchState');
  const enabled = !!slot();
  const entries = new Map();
  let current = '';
  let pollTimer = null;
  let message = '';
  let epoch = 0;
  const path = (task) => `/api/tasks/${encodeURIComponent(task)}/strategy-workbench`;

  function render() {
    const target = slot();
    if (!target) return;
    const entry = entries.get(current);
    const revisions = entry?.binding?.history || [];
    const note = message || (entry?.conflict === 'workbench_evidence_changed' ? '证据已更新，草稿未保存。本页输入仍保留；请读取最新版本号，清除失效选择后再保存。'
      : entry?.conflict ? '草稿已被另一页面修改。请先选择读取服务端草稿，或保留本页并读取最新版本号。'
      : entry?.binding ? `草稿版本 ${entry.binding.head_revision} · ${entry.dirty ? '本页有待保存修改' : '已绑定当前用户与任务'}` : '正在读取工作台草稿…');
    target.innerHTML = `<div class="candidate-lab-inline-controls"><span>${escapeHtml(note)}</span><button type="button" class="button compact secondary" data-workbench-save>保存草稿</button><button type="button" class="button compact secondary" data-workbench-read-head>保留本页，读取最新版本号</button><button type="button" class="button compact secondary" data-workbench-import-local>导入本机旧版视图（未绑定用户）</button>${revisions.length ? `<select data-workbench-version aria-label="草稿版本">${revisions.map(r => `<option value="${r.revision}">版本 ${r.revision}</option>`).join('')}</select><button type="button" class="button compact secondary" data-workbench-restore>恢复所选草稿</button>` : ''}</div><small>叙事是草稿，不作为测量证据。证据变化会清除失效选择；只支持列出的四类搜索内部暂停。</small>${(entry?.executions?.items || []).map(item => {
      const canControl = !entry.busy && (item.state !== 'paused' || item.evidence_hash === entry.executions.evidence_hash);
      const buttons = (item.state === 'running' ? ['pause', 'cancel'] : item.state === 'pause_requested' ? ['cancel'] : item.state === 'paused' ? ['resume'] : []);
      return `<div class="candidate-lab-inline-controls"><span>${escapeHtml(TOOL_COPY[item.tool_ref] || item.tool_ref)} · ${escapeHtml(STATE_COPY[item.state] || item.state)} · ${item.completed}/${item.total} 候选</span>${buttons.map(action => `<button type="button" class="button compact secondary" data-workbench-control="${action}" data-workbench-search="${escapeHtml(item.id)}" ${canControl ? '' : 'disabled'}>${{pause:'暂停', cancel:'取消', resume:'继续'}[action]}</button>`).join('')}${!canControl && buttons.length && !entry.busy ? '<span>证据已变化，不能继续旧搜索</span>' : ''}${item.state === 'uncertain' ? '<span>请在原步骤核对执行凭据后处理</span>' : item.state === 'failed' ? '<span>请在原步骤查看错误，核对输入后启动新的搜索</span>' : ''}</div>`;
    }).join('')}`;
  }

  async function refreshExecutions(task = current) {
    const entry = entries.get(task);
    if (!enabled || !entry || task !== current || task !== getTaskId?.()) return;
    try {
      const previous = new Map((entry.executions?.items || []).map(i => [i.id, i.state]));
      const executions = await get(`${path(task)}/executions`);
      if (entries.get(task) !== entry || task !== current || task !== getTaskId?.()) return;
      entry.executions = executions;
      if (task === current) {
        render();
        if (entry.executions.items.some(i => previous.has(i.id) && previous.get(i.id) !== i.state && ['completed','paused','cancelled','failed','uncertain'].includes(i.state))) onExecutionSettled?.(task);
      }
    }
    catch (error) { if (task === current) { message = publicError(error, '搜索进度读取失败，请刷新后重试。'); render(); } }
  }

  function poll() {
    if (pollTimer) cancelSchedule(pollTimer);
    if (!enabled || !current) return;
    pollTimer = schedule(async () => { await refreshExecutions(); poll(); }, 1500);
  }

  async function load(task) {
    if (!enabled) return null;
    const turn = ++epoch;
    current = task;
    const entry = entries.get(task) || { binding: null, dirty: false, timer: null, saving: false, conflict: false, draft: null, executions: null, busy: false };
    if (entry.timer) cancelSchedule(entry.timer);
    entry.timer = null;
    entries.set(task, entry);
    message = ''; render();
    try {
      // Read the head after our own in-flight save has settled. Otherwise an
      // older response can advance the head after this load, causing a false CAS conflict.
      while (entry.saving) await entry.saveComplete;
      if (turn !== epoch || current !== task || task !== getTaskId?.()) return null;
      const binding = await get(`${path(task)}/draft`);
      if (turn !== epoch || current !== task || task !== getTaskId?.()) return null;
      if (entry.dirty && entry.binding && binding.head_revision !== entry.binding.head_revision) {
        entry.conflict = 'workbench_draft_conflict';
      } else if (!entry.conflict) {
        if (entry.dirty && entry.binding?.evidence_hash !== binding.evidence_hash) {
          entry.draft = invalidateDraftSelections(entry.draft, binding.static_choices).draft;
        }
        entry.binding = binding;
      }
      if (!entry.dirty) entry.draft = binding.draft;
      else if (!entry.conflict) message = '读取期间本页已有修改；服务端草稿未覆盖本页。';
      if (!entry.conflict && binding.invalidated_selections?.length) message = `已有 ${binding.invalidated_selections.length} 项选择因证据变化失效，请重新选择。`;
      await refreshExecutions(task);
      poll(); render(); return entry.draft;
    } catch (error) { if (turn === epoch) { message = publicError(error, '工作台草稿读取失败，请刷新后重试。'); await refreshExecutions(task); poll(); render(); } return null; }
  }

  function snapshot(task) { return entries.get(task)?.draft || null; }

  async function save(task = current) {
    const entry = entries.get(task);
    if (!enabled || !entry?.binding || entry.saving || entry.conflict || !entry.dirty) return;
    const draft = entry.draft;
    entry.saving = true;
    let finishSave;
    entry.saveComplete = new Promise(resolve => { finishSave = resolve; });
    try {
      const response = await put(`${path(task)}/draft`, { expected_revision: entry.binding.head_revision, evidence_hash: entry.binding.evidence_hash, draft });
      entry.binding = response;
      if (entry.draft === draft) entry.dirty = false;
      if (task === current) message = '草稿已保存；叙事和测量证据分别管理。';
    } catch (error) {
      if (error.status === 409) entry.conflict = errorCode(error) === 'workbench_evidence_changed'
        ? 'workbench_evidence_changed' : 'workbench_draft_conflict';
      if (task === current) message = error.status === 409 ? '' : publicError(error, '草稿保存失败，请检查当前任务和内容后重试。');
    } finally {
      entry.saving = false;
      finishSave();
      if (task === current) render();
      if (entry.dirty && !entry.conflict && entry.draft !== draft) await save(task);
    }
  }

  function persist(task, root) {
    if (!enabled) return false;
    const entry = entries.get(task);
    if (!entry) return false;
    const draft = captureWorkbenchDraft(root);
    if (JSON.stringify(draft) === JSON.stringify(entry.draft)) return true;
    entry.draft = draft;
    entry.dirty = true;
    if (entry.timer) cancelSchedule(entry.timer);
    const timer = schedule(() => {
      if (entries.get(task) !== entry || entry.timer !== timer) return;
      entry.timer = null;
      void save(task);
    }, 600);
    entry.timer = timer;
    if (task === current) render();
    return true;
  }

  async function click(event) {
    const button = event.target?.closest?.('[data-workbench-save], [data-workbench-read-head], [data-workbench-restore], [data-workbench-import-local], [data-workbench-control]');
    if (!button || !slot()?.contains?.(button)) return;
    const task = current, entry = entries.get(task);
    if (!entry || (!entry.binding && !button.hasAttribute('data-workbench-control')) || entry.busy || getTaskId?.() !== task) return;
    entry.busy = true; message = ''; render();
    try {
      if (button.hasAttribute('data-workbench-save')) { persist(task, getPanel()); await save(task); }
      else if (button.hasAttribute('data-workbench-import-local')) {
        const legacy = loadStrategyCandidateLabViewState(task);
        if (!legacy) { message = '本机没有该任务的旧版视图。'; return; }
        const imported = captureWorkbenchDraft(getPanel());
        const byKey = new Map(legacy.fields.map(e => [`${e.workflow}:${e.field}`, e]));
        for (const field of imported.fields) {
          const old = byKey.get(`${field.workflow}:${field.field}`);
          if (old) field.values = old.values;
        }
        imported.open_workflows = legacy.open_workflows;
        restoreWorkbenchDraft(getPanel(), imported); onRestored?.(imported);
        persist(task, getPanel()); await save(task);
        if (!entry.conflict) message = '旧版参数和选择已导入并保存到当前用户的新版本；不在当前证据中的选项已清除。';
      }
      else if (button.hasAttribute('data-workbench-read-head')) {
        if (entry.timer) cancelSchedule(entry.timer);
        entry.timer = null;
        while (entry.saving) await entry.saveComplete;
        const binding = await get(`${path(task)}/draft`);
        if (task !== current || getTaskId?.() !== task) return;
        const changedEvidence = entry.binding.evidence_hash !== binding.evidence_hash;
        const invalidated = changedEvidence ? invalidateDraftSelections(entry.draft, binding.static_choices) : { draft: entry.draft, invalidated: [] };
        entry.draft = invalidated.draft;
        entry.binding = binding; entry.conflict = false;
        if (invalidated.invalidated.length) {
          entry.dirty = true;
          restoreWorkbenchDraft(getPanel(), entry.draft); onRestored?.(entry.draft);
        }
        message = changedEvidence ? `本页文字、参数和展示偏好已保留；${invalidated.invalidated.length} 项旧证据选择已清除。请重新选择后保存草稿。`
          : '本页修改已保留。点击保存草稿会创建新版本，旧版本仍可恢复。';
      } else if (button.hasAttribute('data-workbench-restore')) {
        if (entry.timer) cancelSchedule(entry.timer);
        entry.timer = null;
        while (entry.saving) await entry.saveComplete;
        const version = slot().querySelector('[data-workbench-version]')?.value;
        const binding = await get(`${path(task)}/draft?revision=${encodeURIComponent(version || '')}`);
        if (task !== current || getTaskId?.() !== task) return;
        entry.binding = binding; entry.conflict = false; entry.dirty = false; entry.draft = binding.draft;
        restoreWorkbenchDraft(getPanel(), binding.draft); onRestored?.(binding.draft);
        message = binding.invalidated_selections.length ? '草稿已恢复；失效选择已清除，请按当前证据重新选择。' : '所选草稿已恢复。';
      } else {
        const item = entry.executions.items.find(i => i.id === button.dataset.workbenchSearch);
        if (!item) return;
        await post(`${path(task)}/executions/${encodeURIComponent(item.id)}/${button.dataset.workbenchControl}`, { expected_revision: item.revision, evidence_hash: item.evidence_hash });
        await refreshExecutions(task);
      }
    } catch (error) { if (task === current) message = publicError(error, error.status === 409 ? '当前版本或执行状态已变化，请刷新后按提示处理。' : '操作失败，请刷新后重试。'); }
    finally { entry.busy = false; if (task === current) render(); }
  }

  slot()?.addEventListener?.('click', click);
  function clear() { if (pollTimer) cancelSchedule(pollTimer); current = ''; epoch += 1; message = ''; render(); }
  return { enabled, load, snapshot, persist, save, refreshExecutions, render, clear, restore: restoreWorkbenchDraft };
}
