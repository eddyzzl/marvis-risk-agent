"""Workbench chrome (2026-10-14 redesign, batch 2): Inspector popover and collapsible plan rail."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"
PRELUDE = r'''
import assert from "node:assert/strict";
import {
  createWorkbenchChrome,
  inspectorAttentionCount,
  inspectorVisiblePanels,
  railProgressLabel,
  railProgressSummary,
} from "./marvis/static/js/workbench-chrome.js";

function makeEl(id, {classes = [], attrs = {}, hidden = false} = {}) {
  const listeners = {};
  const el = {
    id,
    hidden,
    attrs: {...attrs},
    dataset: {},
    children: [],
    parentElement: null,
    textContent: "",
    focused: false,
    classSet: new Set(classes),
    classList: {
      contains: (name) => el.classSet.has(name),
      toggle: (name, force) => {
        const on = force === undefined ? !el.classSet.has(name) : Boolean(force);
        if (on) el.classSet.add(name); else el.classSet.delete(name);
        return on;
      },
    },
    setAttribute(name, value) { el.attrs[name] = String(value); },
    getAttribute(name) { return el.attrs[name] ?? null; },
    addEventListener(name, fn) { (listeners[name] ||= []).push(fn); },
    fire(name, event = {}) { (listeners[name] || []).forEach((fn) => fn(event)); },
    contains(other) {
      for (let node = other; node; node = node.parentElement) if (node === el) return true;
      return false;
    },
    closest(selector) {
      const name = selector.replace(/^\./, "");
      for (let node = el; node; node = node.parentElement) if (node.classSet?.has(name)) return node;
      return null;
    },
    append(...nodes) { nodes.forEach((node) => { node.parentElement = el; el.children.push(node); }); return el; },
    descendants() { return el.children.flatMap((child) => [child, ...child.descendants()]); },
    querySelectorAll(selector) {
      const all = el.descendants();
      if (selector === "[data-inspector-item]") return all.filter((node) => "data-inspector-item" in node.attrs);
      if (selector === '[data-business-verdict="not_configured"]') return all.filter((node) => node.attrs["data-business-verdict"] === "not_configured");
      if (selector === ".step") return all.filter((node) => node.classSet.has("step"));
      throw new Error(`unsupported selector ${selector}`);
    },
    focus() { el.focused = true; },
  };
  return el;
}

function makeStorage(initial = {}) {
  const data = {...initial};
  return {data, getItem: (key) => data[key] ?? null, setItem: (key, value) => { data[key] = String(value); }};
}

const observers = [];
class FakeObserver {
  constructor(callback) { this.callback = callback; observers.push(this); }
  observe(target, options) { this.target = target; this.options = options; }
}

function makeDoc() {
  const docListeners = {};
  const ids = {};
  const add = (el) => { ids[el.id] = el; return el; };
  const body = add(makeEl("taskInspectorBody"));
  const acceptance = makeEl("businessAcceptancePanel", {attrs: {"data-inspector-item": ""}, hidden: true});
  const replay = makeEl("historicalReplayPanel", {attrs: {"data-inspector-item": ""}, hidden: true});
  body.append(acceptance, replay);
  const inspector = add(makeEl("taskInspector", {hidden: true}));
  inspector.append(body);
  add(makeEl("taskInspectorToggle", {hidden: true}));
  add(makeEl("taskInspectorCount", {hidden: true}));
  add(makeEl("validationWorkspace"));
  add(makeEl("workflowStepper"));
  add(makeEl("progressRailCollapse"));
  add(makeEl("progressRailExpand", {hidden: true}));
  add(makeEl("progressRailExpandLabel"));
  const filter = add(makeEl("threadFilter", {hidden: true}));
  const filterAll = makeEl("", {attrs: {"aria-checked": "true"}});
  filterAll.dataset.threadFilter = "all";
  const filterResults = makeEl("", {attrs: {"aria-checked": "false"}});
  filterResults.dataset.threadFilter = "results";
  filter.append(filterAll, filterResults);
  filter.querySelectorAll = (selector) => {
    if (selector === "[data-thread-filter]") return [filterAll, filterResults];
    throw new Error(`unsupported selector ${selector}`);
  };
  add(makeEl("resultWorkspace"));
  return {
    filterAll,
    filterResults,
    ids,
    acceptance,
    replay,
    getElementById: (id) => ids[id] || null,
    addEventListener(name, fn) { (docListeners[name] ||= []).push(fn); },
    fire(name, event) { (docListeners[name] || []).forEach((fn) => fn(event)); },
  };
}

function step(status, parent = null) {
  const el = makeEl("", {classes: ["step", "plan-rail-step", status]});
  if (parent) parent.append(el);
  return el;
}
'''


def node(body):
    result = subprocess.run(["node", "--input-type=module", "-e", PRELUDE + body], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_inspector_counts_only_visible_panels_and_unconfigured_acceptance():
    node(r'''
const doc = makeDoc();
const chrome = createWorkbenchChrome({doc, storage: makeStorage(), MutationObserverCtor: FakeObserver});
chrome.bind();
const toggle = doc.ids.taskInspectorToggle;
const count = doc.ids.taskInspectorCount;
// No panel applies to the task: the toolbar shows no Inspector at all.
assert.equal(toggle.hidden, true);
assert.equal(inspectorVisiblePanels(doc.ids.taskInspectorBody).length, 0);

// A panel renders an unconfigured business acceptance contract.
doc.acceptance.hidden = false;
const verdict = makeEl("", {attrs: {"data-business-verdict": "not_configured"}});
doc.acceptance.append(verdict);
observers.find((o) => o.target === doc.ids.taskInspectorBody).callback();
assert.equal(toggle.hidden, false);
assert.equal(inspectorAttentionCount(doc.ids.taskInspectorBody), 1);
assert.equal(count.hidden, false);
assert.equal(count.textContent, "1");
assert.equal(toggle.getAttribute("aria-label"), "检查器，1 项需要处理");

// Configured contracts no longer ask for attention.
verdict.attrs["data-business-verdict"] = "passed";
chrome.syncInspector();
assert.equal(count.hidden, true);
assert.equal(toggle.getAttribute("aria-label"), "检查器");
''')


def test_inspector_opens_closes_on_escape_outside_click_and_when_panels_vanish():
    node(r'''
const doc = makeDoc();
const chrome = createWorkbenchChrome({doc, storage: makeStorage(), MutationObserverCtor: FakeObserver});
chrome.bind();
doc.replay.hidden = false;
chrome.syncInspector();
const toggle = doc.ids.taskInspectorToggle;
const inspector = doc.ids.taskInspector;

toggle.fire("click");
assert.equal(inspector.hidden, false);
assert.equal(toggle.getAttribute("aria-expanded"), "true");

// Clicks inside the popover keep it open; Escape closes it and returns focus.
doc.fire("pointerdown", {target: doc.replay});
assert.equal(inspector.hidden, false);
doc.fire("keydown", {key: "Escape"});
assert.equal(inspector.hidden, true);
assert.equal(toggle.focused, true);

toggle.fire("click");
// A confirm dialog raised by an Inspector panel does not dismiss the popover.
const dialog = makeEl("confirm", {classes: ["dialog"]});
const dialogButton = makeEl("confirmButton");
dialog.append(dialogButton);
dialog.closest = (selector) => (selector === "dialog" ? dialog : null);
dialogButton.closest = (selector) => (selector === "dialog" ? dialog : null);
doc.fire("pointerdown", {target: dialogButton});
assert.equal(inspector.hidden, false);
doc.fire("pointerdown", {target: makeEl("elsewhere")});
assert.equal(inspector.hidden, true);
assert.equal(toggle.getAttribute("aria-expanded"), "false");

// Switching to a task without inspector panels closes and hides it.
toggle.fire("click");
doc.replay.hidden = true;
chrome.syncInspector();
assert.equal(inspector.hidden, true);
assert.equal(toggle.hidden, true);
''')


def test_rail_folds_vertically_persists_and_capsule_reports_top_level_step_state():
    node(r'''
const storage = makeStorage({marvis_rail_collapsed: "1"});
const doc = makeDoc();
const stepper = doc.ids.workflowStepper;
const first = step("succeeded", stepper);
step("running", stepper);
step("pending", stepper);
// Nested sub-steps do not count towards the plan total.
step("failed", first);

const chrome = createWorkbenchChrome({doc, storage, MutationObserverCtor: FakeObserver});
chrome.bind();
const workspace = doc.ids.validationWorkspace;
const capsule = doc.ids.progressRailExpand;
const fold = doc.ids.progressRailCollapse;
// V3 UI-9: a rail hidden sideways by an earlier version comes back folded,
// with its header in place; the old flag is cleared.
assert.equal(workspace.classList.contains("rail-collapsed"), false);
assert.equal(workspace.classList.contains("rail-steps-collapsed"), true, "restores the stored state as folded");
assert.equal(capsule.hidden, true);
assert.equal(storage.data.marvis_rail_collapsed, "0");
assert.equal(storage.data.marvis_rail_steps_collapsed, "1");
assert.equal(fold.getAttribute("aria-expanded"), "false");
assert.equal(fold.getAttribute("aria-label"), "展开步骤");
// Batch 4 (2026-10-14 redesign): canonical word plus position, no 计划 prefix; tone in the shared set.
// Batch 4 review B5: the position is the current (first unfinished) step.
assert.equal(doc.ids.progressRailExpandLabel.textContent, "执行中 · 2/3");
assert.equal(capsule.dataset.railState, "run");

// The header button folds the step list up and down, and remembers it.
fold.fire("click");
assert.equal(workspace.classList.contains("rail-steps-collapsed"), false);
assert.equal(storage.data.marvis_rail_steps_collapsed, "0");
assert.equal(fold.getAttribute("aria-expanded"), "true");
assert.equal(fold.getAttribute("aria-label"), "收起步骤");
fold.fire("click");
assert.equal(workspace.classList.contains("rail-steps-collapsed"), true);
assert.equal(workspace.classList.contains("rail-collapsed"), false, "the rail never hides sideways");
assert.equal(storage.data.marvis_rail_steps_collapsed, "1");
assert.equal(fold.getAttribute("aria-expanded"), "false");

// Batch 4 (2026-10-14 redesign): the summary carries the canonical status key and the failed position.
assert.deepEqual(railProgressSummary(stepper), {total: 3, done: 1, failedAt: 0, status: "running"});
// Batch 4: "<状态词> · n/m" while unfinished, "失败 · 第 k 步" on failure, "已完成" when done.
assert.equal(railProgressLabel({total: 0, done: 0, failedAt: 0, status: "not_started"}), "待开始");
assert.equal(railProgressLabel({total: 4, done: 2, failedAt: 3, status: "failed"}), "失败 · 第 3 步");
assert.equal(railProgressLabel({total: 4, done: 2, failedAt: 0, status: "awaiting_confirm"}), "待确认 · 3/4");
assert.equal(railProgressLabel({total: 4, done: 4, failedAt: 0, status: "needs_review"}), "需复核 · 4/4");
assert.equal(railProgressLabel({total: 4, done: 4, failedAt: 0, status: "done"}), "已完成");

// The resolved status written by app.js wins over the step classes.
stepper.dataset.railStatus = "needs_review";
chrome.syncRailCapsule();
assert.equal(doc.ids.progressRailExpandLabel.textContent, "需复核 · 2/3");
assert.equal(capsule.dataset.railState, "review");
stepper.dataset.railStatus = "failed";
step("failed", stepper);
chrome.syncRailCapsule();
assert.equal(doc.ids.progressRailExpandLabel.textContent, "失败 · 第 4 步");
assert.equal(capsule.dataset.railState, "fail");
''')


def test_storage_failures_keep_the_rail_open():
    node(r'''
const doc = makeDoc();
const storage = {getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }};
const chrome = createWorkbenchChrome({doc, storage, MutationObserverCtor: FakeObserver});
chrome.bind();
assert.equal(doc.ids.validationWorkspace.classList.contains("rail-steps-collapsed"), false);
doc.ids.progressRailCollapse.fire("click");
assert.equal(doc.ids.validationWorkspace.classList.contains("rail-steps-collapsed"), true);
''')


def test_governance_panels_live_in_the_inspector_with_their_ids():
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    inspector_start = html.index('<div id="taskInspector"')
    inspector_end = html.index('<section id="reportDraftWorkspace"', inspector_start)
    inspector = html[inspector_start:inspector_end]
    for panel_id in ('businessAcceptancePanel', 'blindOotPanel', 'historicalReplayPanel', 'riskContextPanel', 'institutionAdapterPanel', 'operationsExtensionsPanel', 'collectionWorkspacePanel'):
        assert f'<section id="{panel_id}" data-inspector-item' in inspector
        assert html.count(f'id="{panel_id}"') == 1
    assert '<details id="modelMonitoringPanel" data-inspector-item' in inspector
    assert 'id="dataWorkspacePanel"\n                  data-inspector-item' in inspector
    assert 'id="sqlImport' in inspector or 'sql-import' in inspector
    assert 'id="labelingSetupPanel"' not in inspector
    assert 'id="portfolioSetupPanel"' not in inspector
    assert html.count('id="modelMonitoringPanel"') == 1
    scroll_start = html.index('<div id="resultScrollContent"')
    assert 'data-inspector-item' not in html[scroll_start:]
    assert 'class="task-inspector ui-glass" role="dialog" aria-label="检查器" hidden' in inspector
    head_start = html.index('<div class="task-hero-top-right">')
    head = html[head_start:html.index('<div class="task-hero-details"', head_start)]
    assert 'id="progressRailExpand"' in head
    assert 'id="taskInspectorToggle"' in head
    assert 'aria-controls="taskInspector"' in head
    assert 'id="progressRailCollapse"' in html


def test_chrome_styles_keep_glass_on_floating_layers_only():
    css = (STATIC / "css" / "workbench-chrome.css").read_text(encoding="utf-8")
    assert ".task-inspector[hidden] {\n  display: none;\n}" in css
    assert "background: var(--glass-bg-dense);" in css
    assert ".validation-workspace.rail-collapsed .progress-rail" in css
    assert "grid-template-columns: minmax(0, 1fr) var(--workspace-main-gutter) 0;" in css
    assert "left: calc(var(--sidebar-width) - 6px);" in css
    assert ".app-shell.sidebar-collapsed .resize-handle-left" in css
    # Content inside the popover stays flat: no nested blur.
    assert css.count("backdrop-filter") == 0
    app_shell = (STATIC / "css" / "app-shell.css").read_text(encoding="utf-8")
    assert "body.is-resizing .resize-handle::after {\n  background: var(--accent);\n}" in app_shell


def test_thread_filter_toggles_results_view_and_resets_per_task():
    node(r'''
const doc = makeDoc();
const chrome = createWorkbenchChrome({doc, storage: makeStorage(), MutationObserverCtor: FakeObserver});
chrome.bind();
const filter = doc.ids.threadFilter;
const workspace = doc.ids.resultWorkspace;
chrome.setThreadFilterAvailable(true, "task-a");
assert.equal(filter.hidden, false);
const clickOn = (button) => filter.fire("click", {target: {closest: (sel) => (sel === "[data-thread-filter]" ? button : null)}});
clickOn(doc.filterResults);
assert.equal(workspace.dataset.threadFilter, "results");
assert.equal(doc.filterResults.getAttribute("aria-checked"), "true");
assert.equal(doc.filterAll.getAttribute("aria-checked"), "false");
// Re-rendering the same task keeps the choice; switching tasks resets it.
chrome.setThreadFilterAvailable(true, "task-a");
assert.equal(workspace.dataset.threadFilter, "results");
chrome.setThreadFilterAvailable(true, "task-b");
assert.equal(workspace.dataset.threadFilter, undefined);
clickOn(doc.filterResults);
chrome.setThreadFilterAvailable(false, "task-b");
assert.equal(filter.hidden, true);
assert.equal(workspace.dataset.threadFilter, undefined);
''')


def test_chrome_tolerates_missing_dom_globals():
    node(r'''
const chrome = createWorkbenchChrome({doc: undefined, storage: undefined, MutationObserverCtor: undefined});
chrome.bind();
chrome.setThreadFilterAvailable(true, "t");
chrome.syncInspector();
''')
