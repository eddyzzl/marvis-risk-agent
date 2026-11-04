// Workbench chrome for the 2026-10-14 redesign (batch 2):
// - the Inspector popover that holds governance and extension panels, so the
//   conversation stays the only thing in the center column (decision D8);
// - the plan rail that collapses into a progress capsule in the task toolbar (D9).
// Panels keep their own ids and controllers (index.html places them inside the
// Inspector); this module only watches their `hidden` state, the business
// verdict attribute, the rail's step classes and its resolved status.

import { statusPresentation } from "./task-status.js";
import { localizeHtml, t } from "./i18n.js";

const RAIL_COLLAPSED_KEY = "marvis_rail_collapsed";
// V3 UI-9 (2026-10-23): the rail folds up instead of hiding to the right; the
// ring and status lines stay, only the step list collapses.
const RAIL_STEPS_COLLAPSED_KEY = "marvis_rail_steps_collapsed";
const RAIL_FOLD_MS = 220;
// V3 UI-15: the sidebar's top block and footer turn to glass only while task
// rows are scrolled under them (the rows' own top/bottom spacing first).
const SIDEBAR_UNDER_TOP_PX = 4;
const SIDEBAR_UNDER_BOTTOM_PX = 8;

function readFlag(storage, key) {
  try {
    return storage.getItem(key) === "1";
  } catch (_) {
    return false;
  }
}

function writeFlag(storage, key, value) {
  try {
    storage.setItem(key, value ? "1" : "0");
  } catch (_) {
    // Optional per-viewer convenience; restricted browsers keep the default.
  }
}

// Q3 (2026-10-14): an item "needs attention" when it is not configured yet and
// would affect the acceptance conclusion. Today that is an unconfigured
// business acceptance contract; the rule reads structured attributes only.
export function inspectorAttentionCount(panelRoot) {
  if (!panelRoot) return 0;
  return panelRoot.querySelectorAll('[data-business-verdict="not_configured"]').length;
}

export function inspectorVisiblePanels(panelRoot) {
  if (!panelRoot) return [];
  return [...panelRoot.querySelectorAll("[data-inspector-item]")].filter(
    (item) => !item.hidden && !item.classList.contains("hidden"),
  );
}

// Fallback when the rail carries no resolved status (isolated renders only).
function railStatusFromClasses(steps, done) {
  const has = (name) => steps.some((step) => step.classList.contains(name));
  if (has("failed")) return "failed";
  if (has("review")) return "awaiting_confirm";
  if (has("running")) return "running";
  if (steps.length && done === steps.length) return "done";
  return "not_started";
}

// D16 (batch 4): the capsule word is the rail record's resolved status, which
// app.js writes to #workflowStepper[data-rail-status]; the top-level step
// classes only give the position (n/m, and which step failed).
export function railProgressSummary(stepper, status = stepper?.dataset?.railStatus) {
  if (!stepper) return { total: 0, done: 0, failedAt: 0, status: status || "not_started" };
  const steps = [...stepper.querySelectorAll(".step")].filter(
    (step) => !step.parentElement?.closest(".step"),
  );
  const done = steps.filter((step) => step.classList.contains("succeeded") || step.classList.contains("skipped")).length;
  const failedAt = steps.findIndex((step) => step.classList.contains("failed")) + 1;
  return {
    total: steps.length,
    done,
    failedAt,
    status: status || railStatusFromClasses(steps, done),
  };
}

// "执行中 · 2/5" (the current step's position: steps run in order, so it is
// the first unfinished one) while unfinished, "失败 · 第 2 步" on failure,
// "已完成" when done.
export function railProgressLabel(summary) {
  const status = statusPresentation(summary?.status);
  const label = t(status.label);
  if (status.key === "done") return label;
  if (status.tone === "fail") return summary.failedAt ? `${label} · ${t("第 {0} 步", { 0: summary.failedAt })}` : label;
  if (!summary?.total) return label;
  return `${label} · ${Math.min(summary.done + 1, summary.total)}/${summary.total}`;
}

function stepTitle(step) {
  return String(step?.querySelector?.(".step-title, strong")?.textContent || "").trim();
}

// Mockup rail header (2026-10-19): a progress ring plus one line saying where
// the plan stands and one saying what happens next. Built only from the rail's
// own step classes and resolved status, so it never claims more than the rail.
export function railSummaryView(stepper, status = stepper?.dataset?.railStatus) {
  const summary = railProgressSummary(stepper, status);
  const presentation = statusPresentation(summary.status);
  const steps = stepper
    ? [...stepper.querySelectorAll(".step")].filter((step) => !step.parentElement?.closest(".step"))
    : [];
  const total = summary.total;
  const fraction = total ? summary.done / total : 0;
  const current = steps.find((step) => !step.classList.contains("succeeded") && !step.classList.contains("skipped"));
  const currentAt = current ? steps.indexOf(current) + 1 : total;
  const name = stepTitle(current);
  if (!total) return { tone: presentation.tone, fraction: 0, title: presentation.label, detail: "" };
  const skipped = steps.filter((step) => step.classList.contains("skipped")).length;
  if (presentation.key === "done") {
    return {
      tone: "ok",
      fraction: 1,
      title: "全部完成",
      detail: skipped ? `${total} 步完成（含 ${skipped} 步跳过）` : `${total} 步完成`,
    };
  }
  if (presentation.tone === "fail") {
    // Only a plain step failure resumes from that step; reconciliation and
    // partial failures say what the rail itself says.
    const resumable = presentation.key === "failed" && summary.failedAt > 0;
    return {
      tone: "fail",
      fraction: Math.max(fraction, 0.12),
      title: summary.failedAt ? `卡在第 ${summary.failedAt} 步` : presentation.label,
      detail: resumable ? "处理后从这一步接着跑" : summary.failedAt ? presentation.label : "",
    };
  }
  if (presentation.tone === "review") {
    const waitingOn = {
      needs_review: "等你复核结果",
      awaiting_confirm: "等你确认",
      awaiting_next: "等你继续",
    }[presentation.key] || presentation.label;
    if (summary.done >= total) {
      return { tone: "review", fraction: 1, title: `${total} 步完成`, detail: waitingOn };
    }
    return {
      tone: "review",
      fraction: Math.max(fraction, 0.04),
      title: `停在第 ${currentAt} 步`,
      detail: name ? `${name}，${waitingOn}` : waitingOn,
    };
  }
  if (presentation.tone === "run") {
    return {
      tone: "run",
      fraction: Math.max(fraction, 0.04),
      title: `已完成 ${summary.done}/${total} 步`,
      detail: name ? `正在：${name}` : presentation.label,
    };
  }
  return { tone: presentation.tone, fraction, title: presentation.label, detail: `共 ${total} 步` };
}

const RING_LENGTH = 94.25; // 2πr for r = 15

export function railSummaryHtml(view, escape = (value) => String(value ?? "")) {
  const offset = (RING_LENGTH * (1 - Math.min(Math.max(Number(view?.fraction) || 0, 0), 1))).toFixed(2);
  return [
    `<svg class="rail-ring" data-tone="${escape(view?.tone || "neutral")}" viewBox="0 0 38 38" aria-hidden="true" focusable="false">`,
    '<circle class="rail-ring-track" cx="19" cy="19" r="15"></circle>',
    `<circle class="rail-ring-progress" cx="19" cy="19" r="15" stroke-dasharray="${RING_LENGTH}" stroke-dashoffset="${offset}"></circle>`,
    "</svg>",
    '<span class="rail-summary-copy">',
    `<b title="${escape(view?.title || "")}">${escape(view?.title || "")}</b>`,
    view?.detail ? `<small title="${escape(view.detail)}">${escape(view.detail)}</small>` : "",
    "</span>",
  ].join("");
}

function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}

export function createWorkbenchChrome({
  doc = globalThis.document,
  storage = globalThis.localStorage,
  MutationObserverCtor = globalThis.MutationObserver,
  ResizeObserverCtor = globalThis.ResizeObserver,
} = {}) {
  // app.js creates this before boot; headless test harnesses may lack DOM APIs.
  const $ = (id) => doc?.getElementById?.(id) || null;
  const inspector = $("taskInspector");
  const inspectorBody = $("taskInspectorBody");
  const inspectorToggle = $("taskInspectorToggle");
  const inspectorCount = $("taskInspectorCount");
  const artifacts = $("taskArtifacts");
  const artifactsBody = $("taskArtifactsBody");
  const artifactsToggle = $("taskArtifactsToggle");
  const artifactsCount = $("taskArtifactsCount");
  const workspace = $("validationWorkspace");
  const stepper = $("workflowStepper");
  const railCollapse = $("progressRailCollapse");
  const railExpand = $("progressRailExpand");
  const railExpandLabel = $("progressRailExpandLabel");
  const railSummary = $("railSummary");
  const threadFilter = $("threadFilter");
  const resultWorkspace = $("resultWorkspace");
  const sidebar = $("taskSidebar");
  const taskList = $("taskList");
  let threadFilterTaskId = null;
  // The fold the user last asked for, and the fold animation in flight. A
  // click mid-animation reads the target, not the transient class (review C1).
  let railStepsTarget = null;
  let railFold = null;
  let artifactsTaskId = null;

  function setArtifactsOpen(open) {
    if (!artifacts || !artifactsToggle) return;
    artifacts.hidden = !open;
    artifactsToggle.setAttribute("aria-expanded", String(open));
    if (open) setInspectorOpen(false);
  }

  function syncArtifacts() {
    if (!artifactsToggle || !artifactsBody) return;
    const taskId = artifactsBody.dataset?.taskId || "";
    if (taskId !== artifactsTaskId) setArtifactsOpen(false);
    artifactsTaskId = taskId;
    const count = artifactsBody.querySelectorAll(".rail-artifact").length;
    artifactsToggle.hidden = count === 0;
    if (!count) setArtifactsOpen(false);
    if (artifactsCount) {
      artifactsCount.textContent = count ? String(count) : "";
      artifactsCount.hidden = !count;
    }
  }

  function setInspectorOpen(open) {
    if (!inspector || !inspectorToggle) return;
    inspector.hidden = !open;
    inspectorToggle.setAttribute("aria-expanded", String(open));
    if (open) setArtifactsOpen(false);
  }

  // V3 D-7: open the Inspector on one of its visible panels (a quick action in
  // the conversation). Only an existing, shown Inspector item opens.
  function openInspectorItem(id) {
    const item = id ? $(String(id)) : null;
    if (!item || !inspectorBody?.contains?.(item) || !item.hasAttribute?.("data-inspector-item") || item.hidden) return false;
    setInspectorOpen(true);
    const details = item.tagName === "DETAILS" ? item : item.querySelector?.("details");
    if (details) details.open = true;
    item.scrollIntoView?.({ block: "nearest" });
    (details?.querySelector?.("summary") || item).focus?.();
    return true;
  }

  function syncInspector() {
    if (!inspectorToggle) return;
    const visible = inspectorVisiblePanels(inspectorBody);
    inspectorToggle.hidden = visible.length === 0;
    if (!visible.length) setInspectorOpen(false);
    const attention = inspectorAttentionCount(inspectorBody);
    if (inspectorCount) {
      inspectorCount.textContent = attention ? String(attention) : "";
      inspectorCount.hidden = !attention;
    }
    inspectorToggle.setAttribute("aria-label", attention ? `检查器，${attention} 项需要处理` : "检查器");
  }

  function setRailCollapsed(collapsed, { persist = true } = {}) {
    if (!workspace) return;
    workspace.classList.toggle("rail-collapsed", collapsed);
    if (railCollapse) railCollapse.setAttribute("aria-expanded", String(!collapsed));
    if (railExpand) {
      railExpand.hidden = !collapsed;
      railExpand.setAttribute("aria-expanded", String(!collapsed));
    }
    if (persist) writeFlag(storage, RAIL_COLLAPSED_KEY, collapsed);
    syncRailCapsule();
  }

  function prefersReducedMotion() {
    try {
      return Boolean(globalThis.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches);
    } catch (_) {
      return false;
    }
  }

  // Folds the step list under the rail header with an up/down height animation.
  function setRailStepsCollapsed(collapsed, { persist = true, animate = true } = {}) {
    if (!workspace) return;
    if (railFold) {
      // Drop the in-flight fold: its finish must not land after this call.
      railFold.cancelled = true;
      clearTimeout(railFold.timer);
      try {
        railFold.animation.cancel();
      } catch (_) {
        // already finished
      }
      railFold.rail.classList.remove("is-steps-folding");
      workspace.classList.toggle("rail-steps-collapsed", railFold.collapsed);
      railFold = null;
    }
    railStepsTarget = Boolean(collapsed);
    const was = workspace.classList.contains("rail-steps-collapsed");
    if (railCollapse) {
      railCollapse.setAttribute("aria-expanded", String(!collapsed));
      const label = collapsed ? "展开步骤" : "收起步骤";
      railCollapse.setAttribute("aria-label", label);
      railCollapse.setAttribute("title", label);
    }
    if (persist) writeFlag(storage, RAIL_STEPS_COLLAPSED_KEY, collapsed);
    const canAnimate = animate && was !== collapsed && typeof stepper?.animate === "function" && !prefersReducedMotion();
    if (!canAnimate) {
      workspace.classList.toggle("rail-steps-collapsed", collapsed);
      return;
    }
    const rail = stepper.closest?.(".progress-rail");
    if (!rail || typeof rail.animate !== "function") {
      workspace.classList.toggle("rail-steps-collapsed", collapsed);
      return;
    }
    // V3 UI-19: animate the whole rail between its measured start and end
    // heights (FLIP). Animating only the step list left the rail's gap and
    // padding to jump when the class landed, which read as a stutter.
    const from = rail.getBoundingClientRect().height;
    workspace.classList.toggle("rail-steps-collapsed", collapsed);
    const to = rail.getBoundingClientRect().height;
    rail.classList.add("is-steps-folding");
    if (collapsed) workspace.classList.remove("rail-steps-collapsed");
    const animation = rail.animate(
      [{ height: `${from}px` }, { height: `${to}px` }],
      { duration: RAIL_FOLD_MS, easing: "cubic-bezier(0.4, 0, 0.2, 1)" },
    );
    const fold = { animation, rail, collapsed, timer: null, cancelled: false };
    railFold = fold;
    let done = false;
    const finish = () => {
      if (done || fold.cancelled) return;
      done = true;
      clearTimeout(fold.timer);
      if (collapsed) workspace.classList.add("rail-steps-collapsed");
      rail.classList.remove("is-steps-folding");
      if (railFold === fold) railFold = null;
    };
    animation.onfinish = finish;
    animation.oncancel = finish;
    // A page that is not painting (background tab) never finishes the
    // animation or fires its finish event; the fold still lands.
    fold.timer = setTimeout(() => {
      try {
        animation.finish();
      } catch (_) {
        // already finished or cancelled
      }
      finish();
    }, RAIL_FOLD_MS + 120);
  }

  function syncRailCapsule() {
    if (railSummary) {
      const view = railSummaryView(stepper);
      const html = railSummaryHtml(view, escapeText);
      if (railSummary.innerHTML !== localizeHtml(html, { context: railSummary })) railSummary.innerHTML = html;
      railSummary.hidden = !stepper?.querySelector?.(".step");
    }
    if (!railExpandLabel) return;
    const summary = railProgressSummary(stepper);
    railExpandLabel.textContent = railProgressLabel(summary);
    if (railExpand) railExpand.dataset.railState = statusPresentation(summary.status).tone;
  }

  // UI-15: the task list runs under the top block (logo, search, 新建任务) and
  // the footer. CSS pulls the list out by these two heights and pads it back,
  // so nothing moves at rest; only measured on resize or a list re-render.
  function measureSidebarOverlap() {
    if (!sidebar || !taskList?.parentElement) return;
    const box = sidebar.getBoundingClientRect();
    const topRows = [sidebar.querySelector(".list-toolbar"), sidebar.querySelector(".list-head")].filter(Boolean);
    const wrap = taskList.parentElement.getBoundingClientRect();
    const borderBottom = sidebar.offsetHeight - sidebar.clientHeight - sidebar.clientTop;
    const topEdge = topRows.length ? Math.max(...topRows.map((row) => row.getBoundingClientRect().bottom)) : wrap.top;
    const top = topEdge - box.top - sidebar.clientTop;
    const bottom = box.bottom - borderBottom - wrap.bottom;
    sidebar.style.setProperty("--sidebar-list-top", `${Math.max(0, Math.round(top))}px`);
    sidebar.style.setProperty("--sidebar-list-bottom", `${Math.max(0, Math.round(bottom))}px`);
    syncSidebarScrollGlass();
  }

  function syncSidebarScrollGlass() {
    if (!sidebar || !taskList) return;
    const remaining = taskList.scrollHeight - taskList.clientHeight - taskList.scrollTop;
    sidebar.classList.toggle("list-under-top", taskList.scrollTop > SIDEBAR_UNDER_TOP_PX);
    sidebar.classList.toggle("list-under-bottom", remaining > SIDEBAR_UNDER_BOTTOM_PX);
  }

  // "全部 / 只看结果": a view filter on the conversation, reset on task switch.
  function setThreadFilter(value) {
    const results = value === "results";
    if (resultWorkspace) {
      if (results) resultWorkspace.dataset.threadFilter = "results";
      else delete resultWorkspace.dataset.threadFilter;
    }
    threadFilter?.querySelectorAll("[data-thread-filter]").forEach((button) => {
      button.setAttribute("aria-checked", String((button.dataset.threadFilter === "results") === results));
    });
  }

  function setThreadFilterAvailable(available, taskId = null) {
    if (!threadFilter) return;
    if (taskId !== threadFilterTaskId) {
      threadFilterTaskId = taskId;
      setThreadFilter("all");
    }
    threadFilter.hidden = !available;
    if (!available) setThreadFilter("all");
  }

  function bind() {
    if (inspectorToggle) {
      inspectorToggle.addEventListener("click", () => setInspectorOpen(inspector?.hidden !== false));
    }
    artifactsToggle?.addEventListener("click", () => setArtifactsOpen(artifacts?.hidden !== false));
    doc?.addEventListener?.("keydown", (event) => {
      if (event.key === "Escape" && artifacts && !artifacts.hidden) {
        setArtifactsOpen(false);
        artifactsToggle?.focus();
      }
      if (event.key === "Escape" && inspector && !inspector.hidden) {
        setInspectorOpen(false);
        inspectorToggle?.focus();
      }
    });
    doc?.addEventListener?.("pointerdown", (event) => {
      if (artifacts && !artifacts.hidden
        && !artifacts.contains(event.target) && !artifactsToggle?.contains(event.target)
        && !event.target?.closest?.("dialog")) setArtifactsOpen(false);
      if (!inspector || inspector.hidden) return;
      if (inspector.contains(event.target) || inspectorToggle?.contains(event.target)) return;
      // Confirm dialogs opened from an Inspector panel keep the panel in view.
      if (event.target?.closest?.("dialog")) return;
      setInspectorOpen(false);
    });
    threadFilter?.addEventListener("click", (event) => {
      const button = event.target?.closest?.("[data-thread-filter]");
      if (button) setThreadFilter(button.dataset.threadFilter);
    });
    railCollapse?.addEventListener("click", () => {
      const current = railStepsTarget ?? workspace?.classList.contains("rail-steps-collapsed");
      setRailStepsCollapsed(!current);
    });
    // The top-bar plan capsule belonged to the old sideways collapse; it now
    // only reopens a rail hidden by an earlier version.
    railExpand?.addEventListener("click", () => setRailCollapsed(false));

    taskList?.addEventListener("scroll", syncSidebarScrollGlass, { passive: true });
    if (sidebar && ResizeObserverCtor) {
      const resize = new ResizeObserverCtor(measureSidebarOverlap);
      for (const node of [sidebar, sidebar.querySelector(".sidebar-head"), sidebar.querySelector(".list-head"),
        sidebar.querySelector(".list-toolbar"), sidebar.querySelector(".sidebar-footer")]) {
        if (node) resize.observe(node);
      }
    }
    if (taskList && MutationObserverCtor) {
      // One measurement per frame: a poll re-render mutates many rows at once.
      let measureQueued = false;
      const queueMeasure = () => {
        if (measureQueued) return;
        measureQueued = true;
        const run = () => {
          measureQueued = false;
          measureSidebarOverlap();
        };
        if (typeof globalThis.requestAnimationFrame === "function") globalThis.requestAnimationFrame(run);
        else run();
      };
      new MutationObserverCtor(queueMeasure).observe(taskList, { childList: true, subtree: true });
    }
    if (inspectorBody && MutationObserverCtor) {
      new MutationObserverCtor(syncInspector).observe(inspectorBody, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: ["hidden", "class", "data-business-verdict"],
      });
    }
    if (artifactsBody && MutationObserverCtor) {
      new MutationObserverCtor(syncArtifacts).observe(artifactsBody, { subtree: true, childList: true });
    }
    if (stepper && MutationObserverCtor) {
      new MutationObserverCtor(syncRailCapsule).observe(stepper, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: ["class", "data-rail-status"],
      });
    }
    // A rail hidden sideways by an earlier version comes back folded instead.
    const legacyHidden = readFlag(storage, RAIL_COLLAPSED_KEY);
    if (legacyHidden) writeFlag(storage, RAIL_COLLAPSED_KEY, false);
    setRailCollapsed(false, { persist: false });
    setRailStepsCollapsed(legacyHidden || readFlag(storage, RAIL_STEPS_COLLAPSED_KEY), {
      persist: legacyHidden,
      animate: false,
    });
    syncInspector();
    syncArtifacts();
    measureSidebarOverlap();
  }

  return {
    bind,
    setInspectorOpen,
    openInspectorItem,
    setArtifactsOpen,
    setRailCollapsed,
    setRailStepsCollapsed,
    setThreadFilter,
    setThreadFilterAvailable,
    syncInspector,
    syncArtifacts,
    syncSidebarScrollGlass,
    syncRailCapsule,
  };
}
