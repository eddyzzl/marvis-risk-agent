import { escapeHtml } from "./ui-utils.js";
import { schemaTableHtml } from "./v2/schema_table.js";

// The plugin-admin token replaces the old fixed "local-dev" magic header. The
// server embeds the per-workspace token into <body data-marvis-plugin-admin-
// token> for local clients only (see marvis/app.py index handler); promote /
// reject echo it back via X-MARVIS-Plugin-Admin. A remote client never receives
// it and is additionally blocked by the shared-host access guard.
function pluginAdminToken() {
  return typeof document !== "undefined"
    ? document.body?.dataset?.marvisPluginAdminToken || ""
    : "";
}

function draftStatusLabel(status) {
  return {
    draft: "草稿",
    tested: "已测试",
    promoted: "已转正",
    rejected: "已拒绝",
  }[status] || status || "未知";
}

function draftSourceLabel(source) {
  return {
    web_learning: "联网学习",
    llm_generated: "LLM 生成",
    hand_written: "人工编写",
  }[source] || source || "未知来源";
}

export function createDraftToolsPanelController({
  $,
  api,
  runAction,
  showPlatformConfirm,
} = {}) {
  let draftTools = [];
  let selectedDraftToolId = "";
  let selectedDraftToolDetail = null;
  let loaded = false;

  function setStatus(message = "", kind = "") {
    const status = $("draftToolsStatus");
    if (!status) return;
    status.textContent = message;
    status.className = ["status", kind].filter(Boolean).join(" ");
  }

  function query() {
    const params = new URLSearchParams();
    const status = String($("draftStatusFilter")?.value || "").trim();
    if (status) params.set("status", status);
    return params.toString();
  }

  function renderList() {
    const list = $("draftToolsList");
    if (!list) return;
    if (!draftTools.length) {
      list.innerHTML = '<div class="draft-tool-empty">暂无草稿工具。</div>';
      return;
    }
    list.innerHTML = draftTools.map((draft) => {
      const draftId = String(draft.id || "");
      const selected = draftId === selectedDraftToolId;
      const meta = [
        draftSourceLabel(draft.source),
        draftStatusLabel(draft.status),
        draft.task_id ? `任务 ${draft.task_id}` : "",
      ].filter(Boolean).join(" · ");
      return [
        `<article class="draft-tool-item${selected ? " selected" : ""}" data-draft-tool-id="${escapeHtml(draftId)}" role="button" tabindex="0">`,
        '<div class="draft-tool-item-main">',
        `<strong>${escapeHtml(draft.name || "未命名草稿")}</strong>`,
        `<span>${escapeHtml(meta)}</span>`,
        draft.summary ? `<p>${escapeHtml(draft.summary)}</p>` : "",
        "</div>",
        "</article>",
      ].join("");
    }).join("");
  }

  function renderLearningNote(note) {
    const target = $("draftLearningNote");
    if (!target) return;
    if (!note) {
      target.innerHTML = '<span>无学习笔记来源。</span>';
      return;
    }
    const sources = (note.sources || []).map((source) => `<li>${escapeHtml(source)}</li>`).join("");
    target.innerHTML = [
      '<strong>学习笔记</strong>',
      `<p>${escapeHtml(note.distilled || "")}</p>`,
      sources ? `<ul>${sources}</ul>` : "",
    ].join("");
  }

  function renderRunHistory(runs = []) {
    const target = $("draftRunHistory");
    if (!target) return;
    if (!runs.length) {
      target.innerHTML = '<div class="draft-tool-empty">暂无运行记录。</div>';
      return;
    }
    target.innerHTML = runs.map((run) => [
      '<div class="draft-run-item">',
      `<strong>${run.ok ? "通过" : "失败"}</strong>`,
      `<span>${escapeHtml(run.at || "")}</span>`,
      run.error ? `<code>${escapeHtml(run.error)}</code>` : "",
      "</div>",
    ].join("")).join("");
  }

  function renderDetail(payload = null) {
    selectedDraftToolDetail = payload;
    const draft = payload?.draft || null;
    const body = $("draftToolBody");
    const empty = $("draftToolEmpty");
    if (!body || !empty) return;
    if (!draft) {
      selectedDraftToolId = "";
      empty.classList.remove("hidden");
      body.classList.add("hidden");
      return;
    }
    selectedDraftToolId = String(draft.id || "");
    empty.classList.add("hidden");
    body.classList.remove("hidden");
    $("draftToolName").textContent = draft.name || "未命名草稿";
    $("draftToolSummary").textContent = draft.summary || "";
    $("draftToolStatus").textContent = draftStatusLabel(draft.status);
    $("draftToolStatus").dataset.status = draft.status || "";
    $("draftToolMeta").textContent = [
      draftSourceLabel(draft.source),
      draft.determinism,
      draft.task_id ? `任务 ${draft.task_id}` : "",
      draft.created_at || "",
    ].filter(Boolean).join(" · ");
    $("draftToolCode").textContent = draft.code || "";
    $("draftInputSchema").innerHTML = schemaTableHtml(draft.input_schema || {}, "输入");
    $("draftOutputSchema").innerHTML = schemaTableHtml(draft.output_schema || {}, "输出");
    renderLearningNote(payload.learning_note || null);
    renderRunHistory(payload.runs || []);
    $("draftRunInputs").value = "{}";
    $("draftPromotionTestCases").value = '[\n  {"inputs": {}, "expect": {}}\n]';
    const terminal = ["promoted", "rejected"].includes(String(draft.status || ""));
    $("runDraftButton").disabled = !selectedDraftToolId;
    $("promoteDraftButton").disabled = terminal || !selectedDraftToolId;
    $("rejectDraftButton").disabled = terminal || !selectedDraftToolId;
    renderList();
  }

  function parseJsonField(fieldId, fallback) {
    const raw = String($(fieldId)?.value || "").trim();
    if (!raw) return fallback;
    try {
      return JSON.parse(raw);
    } catch (_) {
      throw new Error(`${fieldId} 不是有效 JSON。`);
    }
  }

  async function load({ preserveSelection = false } = {}) {
    setStatus("正在读取草稿工具...");
    const filter = query();
    const payload = await api("/api/drafts" + (filter ? `?${filter}` : ""));
    loaded = true;
    draftTools = Array.isArray(payload?.drafts) ? payload.drafts : [];
    renderList();
    const selectedStillVisible = draftTools.some((draft) => String(draft.id || "") === selectedDraftToolId);
    if (preserveSelection && selectedStillVisible) {
      await inspect(selectedDraftToolId);
    } else {
      renderDetail(null);
    }
    setStatus(`已读取 ${draftTools.length} 个草稿工具。`, "success");
  }

  async function inspect(draftId) {
    if (!draftId) return;
    setStatus("正在读取草稿详情...");
    const payload = await api(`/api/drafts/${encodeURIComponent(draftId)}`);
    renderDetail(payload);
    setStatus("草稿详情已更新。", "success");
  }

  async function run() {
    const draftId = selectedDraftToolId;
    if (!draftId) return;
    const inputs = parseJsonField("draftRunInputs", {});
    setStatus("正在试运行草稿...");
    const runPayload = await api(`/api/drafts/${encodeURIComponent(draftId)}/run`, {
      method: "POST",
      body: JSON.stringify({ inputs }),
    });
    await inspect(draftId);
    setStatus(
      runPayload.ok ? "试运行通过。" : `试运行失败：${runPayload.error || "未知错误"}`,
      runPayload.ok ? "success" : "error"
    );
  }

  async function promote() {
    const draftId = selectedDraftToolId;
    if (!draftId) return;
    const testCases = parseJsonField("draftPromotionTestCases", []);
    if (!Array.isArray(testCases) || testCases.length === 0) {
      setStatus("请填写转正测试用例。", "error");
      return;
    }
    const confirmed = await showPlatformConfirm({
      title: "转正草稿工具",
      message: "转正后该工具会进入正式工具库并可被 Planner 选用，确定转正？",
      confirmText: "转正",
      cancelText: "取消",
      tone: "warning",
    });
    if (!confirmed) return;
    setStatus("正在执行转正闸门...");
    const payload = await api(`/api/drafts/${encodeURIComponent(draftId)}/promote`, {
      method: "POST",
      headers: { "X-MARVIS-Plugin-Admin": pluginAdminToken() },
      body: JSON.stringify({ test_cases: testCases }),
    });
    await load({ preserveSelection: true });
    setStatus(`已转正为 ${payload?.plugin?.name || "正式工具"}。`, "success");
  }

  async function reject() {
    const draftId = selectedDraftToolId;
    if (!draftId) return;
    const reason = window.prompt("拒绝原因", "") || "";
    setStatus("正在拒绝草稿...");
    await api(`/api/drafts/${encodeURIComponent(draftId)}/reject`, {
      method: "POST",
      headers: { "X-MARVIS-Plugin-Admin": pluginAdminToken() },
      body: JSON.stringify({ reason }),
    });
    await load({ preserveSelection: true });
    setStatus("草稿已拒绝。", "success");
  }

  function inspectFromEvent(event) {
    const item = event.target?.closest?.("[data-draft-tool-id]");
    if (!item) return false;
    event.preventDefault();
    runAction(() => inspect(item.dataset.draftToolId), {
      actionId: "draftTools",
      busyText: "正在读取草稿详情...",
    });
    return true;
  }

  function handleListClick(event) {
    inspectFromEvent(event);
  }

  function handleListKeydown(event) {
    if (event.key !== "Enter" && event.key !== " ") return;
    inspectFromEvent(event);
  }

  return {
    detail: () => selectedDraftToolDetail,
    handleListClick,
    handleListKeydown,
    hasLoaded: () => loaded,
    hasItems: () => draftTools.length > 0,
    inspect,
    load,
    promote,
    reject,
    renderDetail,
    renderList,
    run,
    setStatus,
  };
}
