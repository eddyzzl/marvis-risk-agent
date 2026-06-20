import { escapeHtml } from "../ui-utils.js";
import {
  getDraft as getDraftApi,
  listDrafts as listDraftsApi,
  promoteDraft as promoteDraftApi,
  rejectDraft as rejectDraftApi,
  runDraft as runDraftApi,
  searchDraftWeb as searchDraftWebApi,
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

function statusOptionHtml(value, label, selectedStatus) {
  const selected = String(value || "") === selectedStatus ? " selected" : "";
  return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(label)}</option>`;
}

function safeWebResultUrl(value) {
  const url = String(value || "").trim();
  return /^https?:\/\//i.test(url) ? url : "";
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

function showWebLearningResult(root, payload) {
  const slot = root?.querySelector?.("[data-draft-web-result]");
  if (slot) {
    slot.innerHTML = webLearningResultHtml(payload);
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

function defaultConfirmPromote(id) {
  if (typeof confirm === "function") {
    return confirm(`Promote draft ${id} into the trusted tool registry?`);
  }
  return true;
}

export function draftManagerHtml(data = {}, options = {}) {
  const drafts = data.drafts || [];
  const selectedStatus = String(options.status || "");
  const rows = drafts.length
    ? drafts.map(draftRowHtml).join("")
    : '<div class="v2-empty" data-v2-empty="drafts">No draft tools</div>';
  return `<section class="draft-manager">
    <header class="draft-manager-head">
      <label>
        Status
        <select data-draft-status>
          ${statusOptionHtml("", "All", selectedStatus)}
          ${statusOptionHtml("draft", "Draft", selectedStatus)}
          ${statusOptionHtml("tested", "Tested", selectedStatus)}
          ${statusOptionHtml("promoted", "Promoted", selectedStatus)}
          ${statusOptionHtml("rejected", "Rejected", selectedStatus)}
        </select>
      </label>
      <button type="button" data-refresh-drafts>Refresh</button>
    </header>
    <section class="draft-web-learning">
      <label>
        Web learning
        <input type="search" data-draft-web-query>
      </label>
      <button type="button" data-draft-web-search>Search</button>
      <div data-draft-web-result></div>
    </section>
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

export function webLearningResultHtml(payload = {}) {
  if (payload.offline) {
    return `<div class="draft-web-guidance offline">${escapeHtml(payload.guidance || "No network. Produce the tool externally, then upload it as a plugin.")}</div>`;
  }
  const results = payload.results || [];
  if (!results.length) {
    return '<div class="v2-empty" data-v2-empty="draft-web-results">No web results</div>';
  }
  const items = results.map((result) => {
    const safeUrl = safeWebResultUrl(result.url);
    const urlHtml = safeUrl
      ? `<a href="${escapeHtml(safeUrl)}" rel="noreferrer">${escapeHtml(safeUrl)}</a>`
      : escapeHtml(result.url || "");
    return `<li>
      <strong>${escapeHtml(result.title || result.url || "Result")}</strong>
      ${urlHtml}
      ${result.snippet ? `<p>${escapeHtml(result.snippet)}</p>` : ""}
    </li>`;
  }).join("");
  return `<ol class="draft-web-results">${items}</ol>`;
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
  const query = statusQuery(container);
  const data = await actions.listDrafts(query);
  container.innerHTML = draftManagerHtml(data, query);
  return data;
}

export function attachDraftHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachDraftHandlers requires a stable event root");
  }
  const actions = {
    confirmPromote: defaultConfirmPromote,
    confirmReject: defaultConfirmReject,
    getDraft: getDraftApi,
    promoteDraft: promoteDraftApi,
    refreshDrafts: async () => {},
    rejectDraft: rejectDraftApi,
    runDraft: runDraftApi,
    webSearch: searchDraftWebApi,
    showError: defaultShowError,
    ...deps,
  };

  const refresh = () => actions.refreshDrafts(statusQuery(root));
  const refreshWithFeedback = async () => {
    try {
      await refresh();
    } catch (error) {
      actions.showError(error?.message || "draft refresh failed");
    }
  };

  const clickHandler = async (event) => {
    const target = event.target;
    const webSearchButton = closest(target, "[data-draft-web-search]");
    if (webSearchButton) {
      event.preventDefault?.();
      const query = String(root.querySelector?.("[data-draft-web-query]")?.value || "").trim();
      if (!query) {
        actions.showError("web learning query is required");
        return;
      }
      try {
        showWebLearningResult(root, await actions.webSearch(query));
      } catch (error) {
        actions.showError(error?.message || "draft web learning failed");
      }
      return;
    }

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
        const draftId = promoteButton.dataset.promoteDraft;
        const testCases = parseJsonField(root, "[data-draft-promotion-tests]", []);
        if (!Array.isArray(testCases) || !testCases.length) {
          throw new Error("Promotion test cases are required");
        }
        if (!actions.confirmPromote(draftId, testCases)) {
          return;
        }
        await actions.promoteDraft(draftId, testCases);
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
      await refreshWithFeedback();
    }
  };

  const changeHandler = async (event) => {
    if (closest(event.target, "[data-draft-status]")) {
      await refreshWithFeedback();
    }
  };

  root.addEventListener("click", clickHandler);
  root.addEventListener("change", changeHandler);
  return () => {
    root.removeEventListener?.("click", clickHandler);
    root.removeEventListener?.("change", changeHandler);
  };
}
