import { escapeHtml } from "../ui-utils.js";
import {
  authorDraftTool as authorDraftToolApi,
  distillDraftLearning as distillDraftLearningApi,
  fetchDraftUrl as fetchDraftUrlApi,
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
    draft: "草稿",
    tested: "已测试",
    promoted: "已晋升",
    rejected: "已拒绝",
  }[status] || String(status || "未知");
}

function sourceLabel(source) {
  return {
    web_learning: "Web 学习",
    llm_generated: "LLM 生成",
    hand_written: "手写",
  }[source] || String(source || "未知来源");
}

function statusOptionHtml(value, label, selectedStatus) {
  const selected = String(value || "") === selectedStatus ? " selected" : "";
  return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(label)}</option>`;
}

function safeWebResultUrl(value) {
  const url = String(value || "").trim();
  return /^https?:\/\//i.test(url) ? url : "";
}

function fieldValue(root, selector) {
  return String(root?.querySelector?.(selector)?.value || "").trim();
}

function setFieldValue(root, selector, value) {
  const field = root?.querySelector?.(selector);
  if (field) {
    field.value = String(value || "");
  }
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
    throw new Error("JSON 格式无效");
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

function showLearningNote(root, note) {
  const slot = root?.querySelector?.("[data-draft-learning-note]");
  if (slot) {
    slot.innerHTML = learningNoteHtml(note);
  }
}

function showFetchedContent(root, payload) {
  setFieldValue(root, "[data-draft-learning-source]", payload?.url || "");
  setFieldValue(root, "[data-draft-learning-content]", payload?.content || "");
  const slot = root?.querySelector?.("[data-draft-learning-note]");
  if (!slot) return;
  if (payload?.offline) {
    slot.innerHTML = `<div class="draft-web-guidance offline">${escapeHtml(payload.guidance || "")}</div>`;
    return;
  }
  slot.innerHTML = `<div class="draft-web-guidance">${escapeHtml(payload?.content || "")}</div>`;
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
    return prompt("请输入拒绝草稿的原因", "");
  }
  return "";
}

function defaultConfirmPromote(id) {
  if (typeof confirm === "function") {
    return confirm(`确定将草稿 ${id} 晋升到可信工具注册表？`);
  }
  return true;
}

export function draftManagerHtml(data = {}, options = {}) {
  const drafts = data.drafts || [];
  const selectedStatus = String(options.status || "");
  const rows = drafts.length
    ? drafts.map(draftRowHtml).join("")
    : '<div class="v2-empty" data-v2-empty="drafts">暂无草稿工具</div>';
  return `<section class="draft-manager">
    <header class="draft-manager-head">
      <label>
        状态
        <select data-draft-status>
          ${statusOptionHtml("", "全部", selectedStatus)}
          ${statusOptionHtml("draft", "草稿", selectedStatus)}
          ${statusOptionHtml("tested", "已测试", selectedStatus)}
          ${statusOptionHtml("promoted", "已晋升", selectedStatus)}
          ${statusOptionHtml("rejected", "已拒绝", selectedStatus)}
        </select>
      </label>
      <button type="button" data-refresh-drafts>刷新</button>
    </header>
    <section class="draft-web-learning">
      <label>
        Web 学习
        <input type="search" data-draft-web-query>
      </label>
      <button type="button" data-draft-web-search>搜索</button>
      <label>
        任务
        <input type="text" data-draft-task-id>
      </label>
      <label>
        目标
        <textarea data-draft-goal rows="2"></textarea>
      </label>
      <label>
        模型
        <input type="text" data-draft-model-id>
      </label>
      <div data-draft-web-result></div>
      <label>
        来源
        <input type="text" data-draft-learning-source>
      </label>
      <label>
        内容
        <textarea data-draft-learning-content rows="4"></textarea>
      </label>
      <input type="hidden" data-draft-learning-note-id>
      <button type="button" data-draft-distill-learning>沉淀学习</button>
      <button type="button" data-draft-author>生成草稿</button>
      <div data-draft-learning-note></div>
    </section>
    <div class="draft-manager-layout">
      <div class="draft-list" data-draft-list>${rows}</div>
      <section class="draft-detail" data-draft-detail>
        <div class="v2-empty" data-v2-empty="draft-detail">请选择一个草稿工具</div>
      </section>
    </div>
  </section>`;
}

export function draftRowHtml(draft) {
  const id = String(draft?.id || "");
  return `<article class="draft-row draft-${escapeHtml(draft?.status || "unknown")}" data-draft-id="${escapeHtml(id)}" role="button" tabindex="0">
    <header>
      <strong>${escapeHtml(draft?.name || "未命名草稿")}</strong>
      <span>${escapeHtml(statusLabel(draft?.status))}</span>
    </header>
    ${draft?.summary ? `<p>${escapeHtml(draft.summary)}</p>` : ""}
    <small>${escapeHtml(sourceLabel(draft?.source))}${draft?.task_id ? ` · ${escapeHtml(draft.task_id)}` : ""}</small>
  </article>`;
}

export function webLearningResultHtml(payload = {}) {
  if (payload.offline) {
    return `<div class="draft-web-guidance offline">${escapeHtml(payload.guidance || "当前无网络。请在外部生成工具后，再通过插件上传导入。")}</div>`;
  }
  const results = payload.results || [];
  if (!results.length) {
    return '<div class="v2-empty" data-v2-empty="draft-web-results">暂无 Web 结果</div>';
  }
  const items = results.map((result) => {
    const safeUrl = safeWebResultUrl(result.url);
    const urlHtml = safeUrl
      ? `<a href="${escapeHtml(safeUrl)}" rel="noreferrer">${escapeHtml(safeUrl)}</a>`
      : escapeHtml(result.url || "");
    const fetchHtml = safeUrl
      ? `<button type="button" data-draft-fetch-url="${escapeHtml(safeUrl)}">抓取</button>`
      : "";
    return `<li>
      <strong>${escapeHtml(result.title || result.url || "结果")}</strong>
      ${urlHtml}
      ${result.snippet ? `<p>${escapeHtml(result.snippet)}</p>` : ""}
      ${fetchHtml}
    </li>`;
  }).join("");
  return `<ol class="draft-web-results">${items}</ol>`;
}

export function draftDetailHtml(payload = {}) {
  const draft = payload.draft || null;
  if (!draft) {
    return '<div class="v2-empty" data-v2-empty="draft-detail">请选择一个草稿工具</div>';
  }
  const id = String(draft.id || "");
  const terminal = ["promoted", "rejected"].includes(String(draft.status || ""));
  return `<article class="draft-detail-card">
    <header>
      <div>
        <h3>${escapeHtml(draft.name || "未命名草稿")}</h3>
        ${draft.summary ? `<p>${escapeHtml(draft.summary)}</p>` : ""}
      </div>
      <span class="draft-status">${escapeHtml(statusLabel(draft.status))}</span>
    </header>
    ${learningNoteHtml(payload.learning_note)}
    <pre class="draft-code"><code>${escapeHtml(draft.code || "")}</code></pre>
    <div class="draft-schema-grid">
      <section><strong>输入 schema</strong><pre><code>${jsonText(draft.input_schema)}</code></pre></section>
      <section><strong>输出 schema</strong><pre><code>${jsonText(draft.output_schema)}</code></pre></section>
    </div>
    <section>
      <strong>运行输入</strong>
      <textarea data-draft-run-inputs rows="4">{}</textarea>
      <button type="button" data-run-draft="${escapeHtml(id)}">运行草稿</button>
    </section>
    <section>
      <strong>晋升测试用例</strong>
      <textarea data-draft-promotion-tests rows="4">[
  {"inputs": {}, "expect": {}}
]</textarea>
      <button type="button" data-promote-draft="${escapeHtml(id)}"${terminal ? " disabled" : ""}>晋升</button>
      <button type="button" data-reject-draft="${escapeHtml(id)}"${terminal ? " disabled" : ""}>拒绝</button>
    </section>
    <section>
      <strong>运行记录</strong>
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
    <strong>学习沉淀</strong>
    <p>${escapeHtml(note.distilled || "")}</p>
    ${sources ? `<ul>${sources}</ul>` : ""}
  </section>`;
}

function runHistoryHtml(runs) {
  if (!runs.length) {
    return '<div class="v2-empty" data-v2-empty="draft-runs">暂无运行记录</div>';
  }
  return runs.map((run) => (
    `<div class="draft-run ${run.ok ? "ok" : "failed"}">
      <span>${escapeHtml(run.ok ? "成功" : "失败")}</span>
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
    authorDraft: authorDraftToolApi,
    confirmPromote: defaultConfirmPromote,
    confirmReject: defaultConfirmReject,
    distillLearning: distillDraftLearningApi,
    fetchUrl: fetchDraftUrlApi,
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
      actions.showError(error?.message || "草稿工具刷新失败");
    }
  };

  const clickHandler = async (event) => {
    const target = event.target;
    const webSearchButton = closest(target, "[data-draft-web-search]");
    if (webSearchButton) {
      event.preventDefault?.();
      const query = String(root.querySelector?.("[data-draft-web-query]")?.value || "").trim();
      if (!query) {
        actions.showError("请输入 Web 学习搜索词。");
        return;
      }
      try {
        showWebLearningResult(root, await actions.webSearch(query));
      } catch (error) {
        actions.showError(error?.message || "草稿 Web 学习失败");
      }
      return;
    }

    const fetchButton = closest(target, "[data-draft-fetch-url]");
    if (fetchButton?.dataset?.draftFetchUrl) {
      event.preventDefault?.();
      try {
        showFetchedContent(root, await actions.fetchUrl(fetchButton.dataset.draftFetchUrl));
      } catch (error) {
        actions.showError(error?.message || "Web 内容抓取失败");
      }
      return;
    }

    const distillButton = closest(target, "[data-draft-distill-learning]");
    if (distillButton) {
      event.preventDefault?.();
      const query = fieldValue(root, "[data-draft-web-query]");
      const content = fieldValue(root, "[data-draft-learning-content]");
      const source = fieldValue(root, "[data-draft-learning-source]");
      const modelId = fieldValue(root, "[data-draft-model-id]");
      if (!query || !content || !source) {
        actions.showError("搜索词、来源和内容不能为空。");
        return;
      }
      const payload = {
        query,
        contents: [content],
        sources: [source],
      };
      if (modelId) payload.model_id = modelId;
      try {
        const result = await actions.distillLearning(payload);
        const note = result?.learning_note || null;
        setFieldValue(root, "[data-draft-learning-note-id]", note?.id || "");
        showLearningNote(root, note);
      } catch (error) {
        actions.showError(error?.message || "学习沉淀失败");
      }
      return;
    }

    const authorButton = closest(target, "[data-draft-author]");
    if (authorButton) {
      event.preventDefault?.();
      const taskId = fieldValue(root, "[data-draft-task-id]");
      const goal = fieldValue(root, "[data-draft-goal]");
      const learningNoteId = fieldValue(root, "[data-draft-learning-note-id]");
      const modelId = fieldValue(root, "[data-draft-model-id]");
      if (!taskId || !goal || !learningNoteId) {
        actions.showError("任务、目标和学习沉淀不能为空。");
        return;
      }
      const payload = {
        task_id: taskId,
        goal,
        learning_note_id: learningNoteId,
      };
      if (modelId) payload.model_id = modelId;
      try {
        const result = await actions.authorDraft(payload);
        await refresh();
        if (result?.draft?.id) {
          showDetail(root, await actions.getDraft(result.draft.id));
        }
      } catch (error) {
        actions.showError(error?.message || "草稿生成失败");
      }
      return;
    }

    const draftItem = closest(target, "[data-draft-id]");
    if (draftItem?.dataset?.draftId) {
      event.preventDefault?.();
      try {
        showDetail(root, await actions.getDraft(draftItem.dataset.draftId));
      } catch (error) {
        actions.showError(error?.message || "草稿详情读取失败");
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
        actions.showError(error?.message || "草稿运行失败");
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
          throw new Error("晋升测试用例不能为空");
        }
        if (!(await actions.confirmPromote(draftId, testCases))) {
          return;
        }
        await actions.promoteDraft(draftId, testCases);
        await refresh();
      } catch (error) {
        actions.showError(error?.message || "草稿晋升失败");
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
        actions.showError(error?.message || "草稿拒绝失败");
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
