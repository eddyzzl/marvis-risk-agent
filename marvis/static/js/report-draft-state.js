// Edits stay in memory until saved to this task's local database. Never put
// report narratives in browser storage or Agent memory.
export function createReportDraftState({ api, onChange = () => {}, delay = 650 } = {}) {
  const entries = new Map();
  const timers = new Map();
  const confirming = new Set();
  const snapshot = (message) => ({
    messageId: String(message.id),
    revision: Number(message.metadata?.report_revision || 0),
    editRevision: Number(message.metadata?.draft_edit_revision || 0),
    values: { ...message.metadata?.draft_values },
  });
  const sameVersion = (a, b) => a.messageId === b.messageId
    && a.revision === b.revision && a.editRevision === b.editRevision;

  function receive(taskId, message) {
    const incoming = snapshot(message);
    let entry = entries.get(taskId);
    // Polls started before our save may arrive after the user starts another edit.
    // A known older revision is never a competing version, even while dirty.
    if (entry?.messageId === incoming.messageId && entry.revision === incoming.revision
      && entry.editRevision > incoming.editRevision) return entry;
    if (!entry || (!entry.dirty && !entry.pending && !entry.conflict)) {
      entry = { ...incoming, dirty: false, confirming: confirming.has(taskId), status: "已保存草稿", error: "" };
      entries.set(taskId, entry);
    } else if (!sameVersion(entry, incoming) && !entry.pending) {
      entry.conflict = incoming;
      entry.status = "版本已变化 · 修改仍保留";
    }
    return entry;
  }

  function edit(taskId, values) {
    const entry = entries.get(taskId);
    if (!entry || confirming.has(taskId)) return;
    entry.values = { ...values };
    entry.dirty = true;
    entry.status = entry.conflict ? "版本冲突 · 修改仍保留" : "有修改 · 等待保存";
    entry.error = "";
    clearTimeout(timers.get(taskId));
    if (!entry.conflict) timers.set(taskId, setTimeout(() => { save(taskId).catch(() => {}); }, delay));
    onChange(taskId, entry);
  }

  function payload(taskId) {
    const e = entries.get(taskId);
    return e ? { revision: e.revision, draft_message_id: e.messageId,
      draft_edit_revision: e.editRevision, text_values: { ...e.values } } : null;
  }

  async function save(taskId) {
    clearTimeout(timers.get(taskId));
    const entry = entries.get(taskId);
    if (!entry) return;
    if (entry.pending) { await entry.pending; return save(taskId); }
    if (entry.conflict) throw new Error("请先比较并解决报告草稿版本冲突。所有修改仍保留。");
    if (!entry.dirty) return;
    const sent = payload(taskId);
    entry.status = "正在保存…";
    entry.pending = (async () => {
      try {
        const result = await api(`api/tasks/${encodeURIComponent(taskId)}/agent/report-draft`, {
          method: "PUT", body: JSON.stringify(sent),
        });
        entry.editRevision = Number(result.message.metadata.draft_edit_revision);
        entry.dirty = JSON.stringify(entry.values) !== JSON.stringify(sent.text_values);
        entry.status = entry.dirty ? "有修改 · 等待保存" : "已保存草稿";
        entry.error = "";
      } catch (error) {
        entry.error = error?.message || "保存失败";
        entry.status = "保存失败 · 修改仍保留";
        if (error?.status === 409) {
          try {
            const result = await api(`api/tasks/${encodeURIComponent(taskId)}/agent/messages`);
            const messages = result.messages || [];
            const latest = [...messages].reverse().find((m) =>
              ["word_conclusion_draft", "word_conclusion_confirmed"].includes(m.stage));
            if (latest?.stage === "word_conclusion_draft") entry.conflict = snapshot(latest);
            else entry.conflict = { unavailable: true };
          } catch (_) { /* Keep the edits and allow a deliberate retry. */ }
        }
        throw error;
      } finally {
        entry.pending = null;
        onChange(taskId, entry);
      }
    })();
    onChange(taskId, entry);
    await entry.pending;
    if (entry.dirty) await save(taskId);
  }

  function resolve(taskId, choice) {
    const entry = entries.get(taskId);
    if (!entry?.conflict || entry.conflict.unavailable || confirming.has(taskId)) return;
    const local = entry.values;
    Object.assign(entry, entry.conflict, { conflict: null, error: "" });
    entry.values = choice === "local" ? local : entry.values;
    entry.dirty = choice === "local";
    entry.status = entry.dirty ? "有修改 · 等待保存" : "已载入最新版本";
    onChange(taskId, entry);
  }

  function setConfirming(taskIds, active) {
    for (const taskId of taskIds) {
      if (active) confirming.add(taskId);
      else confirming.delete(taskId);
      const entry = entries.get(taskId);
      if (entry) {
        entry.confirming = Boolean(active);
        onChange(taskId, entry);
      }
    }
  }

  return { receive, edit, save, payload, resolve, setConfirming,
    isConfirming: (id) => confirming.has(id), get: (id) => entries.get(id),
    hasUnsaved: () => [...entries.values()].some((e) => e.dirty || e.pending),
    async flush(taskIds) { for (const id of taskIds) await save(id); },
    discard(taskId) { clearTimeout(timers.get(taskId)); entries.delete(taskId); },
  };
}
