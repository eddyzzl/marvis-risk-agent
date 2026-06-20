import { escapeHtml } from "../ui-utils.js";
import {
  getDraft as getDraftApi,
  listDrafts as listDraftsApi,
  promoteDraft as promoteDraftApi,
  rejectDraft as rejectDraftApi,
  runDraft as runDraftApi,
} from "./api_v2.js";

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function jsonText(value) {
  return escapeHtml(JSON.stringify(value || {}, null, 2));
}

function statusLabel(status) {
  return {
    draft: "Draft",
    tested: "Tested",
    promoted: "Promoted",
    rejected: "Rejected",
  }[status] || String(status || "unknown");
}

function sourceLabel(source) {
  return {
    web_learning: "Web learning",
    llm_generated: "LLM generated",
    hand_written: "Hand written",
  }[source] || String(source || "unknown");
}

function statusQuery(root) {
  const status = String(root?.querySelector?.("[data-draft-status]")?.value || "").trim();
  return status ? { status } : {};
}

function parseJsonField(root, selector, fallback) {
  const field = root?.querySelector?.(selector);
  if (!field || !String(field.value || "").trim()) {
    return fallback;
  }
  try {
    return JSON.parse(field.value);
  } catch (_error) {
    throw new Error("Invalid JSON");
  }
}

function showDetail(root, payload) {
  const slot = root?.querySelector?.("[data-draft-detail]");
  if (slot) {
    slot.innerHTML = draftDetailHtml(payload);
  }
}

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

function defaultConfirmReject() {
  if (typeof prompt === "function") {
    return prompt("Reject draft reason?", "");
  }
  return "";
}

export function draftManagerHtml(data = {}) {
  const drafts = data.drafts || [];
  const rows = drafts.length
    ? drafts.map(draftRowHtml).join("")
    : '<div class="v2-empty" data-v2-empty="drafts">No draft tools</div>';
  return `<section class="draft-manager">
    <header class="draft-manager-head">
      <label>
        Status
        <select data-draft-status>
          <option value="">All</option>
          <option value="draft">Draft</option>
          <option value="tested">Tested</option>
          <option value="promoted">Promoted</option>
          <option value="rejected">Rejected</option>
        </select>
      </label>
      <button type="button" data-refresh-drafts>Refresh</button>
    </header>
    <div class="draft-manager-layout">
      <div class="draft-list" data-draft-list>${rows}</div>
      <section class="draft-detail" data-draft-detail>
        <div class="v2-empty" data-v2-empty="draft-detail">Select a draft tool</div>
      </section>
    </div>
  </section>`;
}

export function draftRowHtml(draft) {
  const id = String(draft?.id || "");
  return `<article class="draft-row draft-${escapeHtml(draft?.status || "unknown")}" data-draft-id="${escapeHtml(id)}" role="button" tabindex="0">
    <header>
      <strong>${escapeHtml(draft?.name || "Unnamed draft")}</strong>
      <span>${escapeHtml(statusLabel(draft?.status))}</span>
    </header>
    ${draft?.summary ? `<p>${escapeHtml(draft.summary)}</p>` : ""}
    <small>${escapeHtml(sourceLabel(draft?.source))}${draft?.task_id ? ` · ${escapeHtml(draft.task_id)}` : ""}</small>
  </article>`;
}

export function draftDetailHtml(payload = {}) {
  const draft = payload.draft || null;
  if (!draft) {
    return '<div class="v2-empty" data-v2-empty="draft-detail">Select a draft tool</div>';
  }
  const id = String(draft.id || "");
  const terminal = ["promoted", "rejected"].includes(String(draft.status || ""));
  return `<article class="draft-detail-card">
    <header>
      <div>
        <h3>${escapeHtml(draft.name || "Unnamed draft")}</h3>
        ${draft.summary ? `<p>${escapeHtml(draft.summary)}</p>` : ""}
      </div>
      <span class="draft-status">${escapeHtml(statusLabel(draft.status))}</span>
    </header>
    ${learningNoteHtml(payload.learning_note)}
    <pre class="draft-code"><code>${escapeHtml(draft.code || "")}</code></pre>
    <div class="draft-schema-grid">
      <section><strong>Input schema</strong><pre><code>${jsonText(draft.input_schema)}</code></pre></section>
      <section><strong>Output schema</strong><pre><code>${jsonText(draft.output_schema)}</code></pre></section>
    </div>
    <section>
      <strong>Run inputs</strong>
      <textarea data-draft-run-inputs rows="4">{}</textarea>
      <button type="button" data-run-draft="${escapeHtml(id)}">Run draft</button>
    </section>
    <section>
      <strong>Promotion test cases</strong>
      <textarea data-draft-promotion-tests rows="4">[
  {"inputs": {}, "expect": {}}
]</textarea>
      <button type="button" data-promote-draft="${escapeHtml(id)}"${terminal ? " disabled" : ""}>Promote</button>
      <button type="button" data-reject-draft="${escapeHtml(id)}"${terminal ? " disabled" : ""}>Reject</button>
    </section>
    <section>
      <strong>Runs</strong>
      ${runHistoryHtml(payload.runs || [])}
    </section>
  </article>`;
}

function learningNoteHtml(note) {
  if (!note) {
    return "";
  }
  const sources = (note.sources || [])
    .map((source) => `<li>${escapeHtml(source)}</li>`)
    .join("");
  return `<section class="draft-learning-note">
    <strong>Learning note</strong>
    <p>${escapeHtml(note.distilled || "")}</p>
    ${sources ? `<ul>${sources}</ul>` : ""}
  </section>`;
}

function runHistoryHtml(runs) {
  if (!runs.length) {
    return '<div class="v2-empty" data-v2-empty="draft-runs">No runs</div>';
  }
  return runs.map((run) => (
    `<div class="draft-run ${run.ok ? "ok" : "failed"}">
      <span>${escapeHtml(run.ok ? "ok" : "failed")}</span>
      <code>${escapeHtml(run.error || JSON.stringify(run.output || {}))}</code>
      <small>${escapeHtml(run.at || "")}</small>
    </div>`
  )).join("");
}

export function renderDraftManagerShell(container, data = {}) {
  if (!container) {
    throw new Error("renderDraftManagerShell requires a container");
  }
  if (container.dataset) {
    container.dataset.v2DraftManager = "true";
  }
  container.innerHTML = draftManagerHtml(data);
  return () => {};
}

export async function renderDraftManager(container, deps = {}) {
  if (!container) {
    throw new Error("renderDraftManager requires a container");
  }
  if (container.dataset) {
    container.dataset.v2DraftManager = "true";
  }
  const actions = { listDrafts: listDraftsApi, ...deps };
  const data = await actions.listDrafts();
  container.innerHTML = draftManagerHtml(data);
  return data;
}

export function attachDraftHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachDraftHandlers requires a stable event root");
  }
  const actions = {
    confirmReject: defaultConfirmReject,
    getDraft: getDraftApi,
    promoteDraft: promoteDraftApi,
    refreshDrafts: async () => {},
    rejectDraft: rejectDraftApi,
    runDraft: runDraftApi,
    showError: defaultShowError,
    ...deps,
  };

  const refresh = () => actions.refreshDrafts(statusQuery(root));

  const clickHandler = async (event) => {
    const target = event.target;
    const draftItem = closest(target, "[data-draft-id]");
    if (draftItem?.dataset?.draftId) {
      event.preventDefault?.();
      try {
        showDetail(root, await actions.getDraft(draftItem.dataset.draftId));
      } catch (error) {
        actions.showError(error?.message || "draft detail failed");
      }
      return;
    }

    const runButton = closest(target, "[data-run-draft]");
    if (runButton?.dataset?.runDraft) {
      event.preventDefault?.();
      try {
        const inputs = parseJsonField(root, "[data-draft-run-inputs]", {});
        await actions.runDraft(runButton.dataset.runDraft, inputs);
        showDetail(root, await actions.getDraft(runButton.dataset.runDraft));
      } catch (error) {
        actions.showError(error?.message || "draft run failed");
      }
      return;
    }

    const promoteButton = closest(target, "[data-promote-draft]");
    if (promoteButton?.dataset?.promoteDraft) {
      event.preventDefault?.();
      try {
        const testCases = parseJsonField(root, "[data-draft-promotion-tests]", []);
        if (!Array.isArray(testCases) || !testCases.length) {
          throw new Error("Promotion test cases are required");
        }
        await actions.promoteDraft(promoteButton.dataset.promoteDraft, testCases);
        await refresh();
      } catch (error) {
        actions.showError(error?.message || "draft promotion failed");
      }
      return;
    }

    const rejectButton = closest(target, "[data-reject-draft]");
    if (rejectButton?.dataset?.rejectDraft) {
      event.preventDefault?.();
      const reason = actions.confirmReject(rejectButton.dataset.rejectDraft);
      if (reason === null || reason === false) {
        return;
      }
      try {
        await actions.rejectDraft(rejectButton.dataset.rejectDraft, String(reason || ""));
        await refresh();
      } catch (error) {
        actions.showError(error?.message || "draft reject failed");
      }
      return;
    }

    const refreshButton = closest(target, "[data-refresh-drafts]");
    if (refreshButton) {
      event.preventDefault?.();
      await refresh();
    }
  };

  const changeHandler = async (event) => {
    if (closest(event.target, "[data-draft-status]")) {
      await refresh();
    }
  };

  root.addEventListener("click", clickHandler);
  root.addEventListener("change", changeHandler);
  return () => {
    root.removeEventListener?.("click", clickHandler);
    root.removeEventListener?.("change", changeHandler);
  };
}
