"""Strategy workbench fold (2026-10-14 redesign, batch 4).

The Candidate Lab renders above the conversation, so it folds into one summary
card: the head, the summary and the status line stay visible, and the body
(draft bar, preferences, launchers, results) opens on demand. The open state
is remembered per task in this viewer, defaults to collapsed, and deep links
into a launcher open the body first.
"""

import json
import re
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


def _run_node(script: str) -> object:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout or "null")


def _payload_js(task_id: str = "task-1") -> str:
    return f"""
        const payload = {{
          task_id: {json.dumps(task_id)},
          can_start: true,
          blocked_reason: null,
          workflow: {{
            stages: [
              {{ id: "current_context", label: "项目现状", status: "complete" }},
              {{ id: "history", label: "历史版本", status: "complete" }},
              {{ id: "sample_design", label: "样本设计", status: "missing" }},
              {{ id: "candidate_analysis", label: "单变量/模型", status: "complete" }},
              {{ id: "strategy_combination", label: "交叉组合/策略", status: "complete" }},
              {{ id: "impact", label: "影响测算", status: "stale" }},
              {{ id: "report", label: "形成报告", status: "missing" }},
            ],
            latest_evidence: {{
              pool_stability: {{ artifact: {{ created_at: "2026-10-14T08:00:00Z" }}, freshness: "current" }},
              pool_impact: null,
              impact_cube: null,
              pool_validation: {{ validation: null, oot: null }},
            }},
            report: null,
          }},
          candidates: {{
            univariate: {{
              latest: {{ artifact: {{ artifact_id: "u-2", created_at: "2026-10-14T09:30:00Z" }} }},
              all: [], total: 9, truncated: true,
            }},
            cross_matrix: {{ latest: null, all: [{{ artifact: {{ artifact_id: "c-1" }} }}], total: 2, truncated: false }},
          }},
          pool_add_sources: {{ latest: null, all: [], total: 3, truncated: false }},
          pools: {{
            latest: null,
            all: [
              {{ kind: "candidate_pool", pool_id: "p1", total: 5, entries: [], artifact: {{ created_at: "2026-10-13T10:00:00Z" }} }},
              {{ kind: "candidate_pool", pool_id: "p2", entries: [{{}}, {{}}], artifact: {{ created_at: "2026-10-12T10:00:00Z" }} }},
            ],
            total: 2,
            truncated: false,
          }},
          evidence_drawer: {{ red_flags: {{ all: [{{ code: "x" }}], total: 4, truncated: true }} }},
          strategies: {{}},
        }};
    """


def test_summary_facts_come_from_the_structured_payload():
    data = _run_node(
        """
        import {
          strategyCandidateLabSummaryFacts,
          strategyCandidateLabSummaryHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_fold.js";
        PAYLOAD
        const facts = strategyCandidateLabSummaryFacts(payload);
        const done = strategyCandidateLabSummaryFacts({
          ...payload,
          workflow: { stages: payload.workflow.stages.map((stage) => ({ ...stage, status: "complete" })) },
        });
        process.stdout.write(JSON.stringify({
          facts,
          html: strategyCandidateLabSummaryHtml(facts),
          doneHtml: strategyCandidateLabSummaryHtml(done),
          empty: strategyCandidateLabSummaryFacts(null),
          loading: strategyCandidateLabSummaryHtml(strategyCandidateLabSummaryFacts(null, { blockedReason: "loading", loading: true })),
          blocked: strategyCandidateLabSummaryFacts(payload, { blockedReason: "active_plan" }),
          error: strategyCandidateLabSummaryFacts(null, { error: "网络错误 <b>" }),
        }));
        """.replace("PAYLOAD", _payload_js())
    )
    facts = data["facts"]
    assert facts["state"] == "ready"
    assert facts["notice"] == ""
    assert facts["stage"] == {"number": 3, "label": "样本设计", "status": "missing", "statusLabel": "待补充"}
    assert (facts["completed"], facts["stageTotal"]) == (4, 7)
    # Totals, not the truncated page: univariate 9 + cross matrix 2.
    assert facts["candidates"] == 11
    assert facts["poolReady"] == 3
    assert (facts["pools"], facts["rules"]) == (2, 7)
    assert facts["latest"] == {"title": "单变量候选", "createdAt": "2026-10-14T09:30:00.000Z"}
    assert facts["redFlags"] == 4

    html = data["html"]
    assert "第 3 步 样本设计 · 待补充" in html
    assert "<dd>4/7</dd>" in html
    assert "11 个 · 可入池 3" in html
    assert "2 个 · 7 条规则" in html
    assert re.search(r"单变量候选 · \d{2}-\d{2} \d{2}:\d{2}", html)
    assert '<div class="candidate-lab-summary-fact" data-fact="red_flags" data-tone="warn">' in html
    assert "全部阶段已完成" in data["doneHtml"]

    assert data["empty"]["candidates"] is None and data["empty"]["state"] == "ready"
    assert data["loading"] == '<p class="candidate-lab-summary-state" data-state="loading">正在核验候选证据</p>'
    assert data["blocked"]["state"] == "blocked"
    assert data["blocked"]["notice"] == "策略计划执行中，暂不能启动新分析"
    assert data["blocked"]["candidates"] == 11
    # The raw error text stays in the status line; the card shows a fixed word.
    assert data["error"]["state"] == "error"
    assert data["error"]["notice"] == "候选证据读取失败"


def test_fold_defaults_collapsed_toggles_and_remembers_per_task():
    data = _run_node(
        """
        import { createStrategyCandidateLabController } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";
        PAYLOAD
        const store = new Map();
        const storage = {
          getItem: (key) => (store.has(key) ? store.get(key) : null),
          setItem: (key, value) => store.set(key, String(value)),
          removeItem: (key) => store.delete(key),
        };
        // Only the fold keys; the same storage also holds the view-state drafts.
        const foldKeys = () => Object.fromEntries(
          [...store].filter(([key]) => key.startsWith("marvis_strategy_candidate_lab_expanded_v1:")),
        );
        function makeIds() {
          const attrs = {};
          return {
            strategyCandidateLabPanel: {
              classList: { toggle() {} }, dataset: {}, setAttribute() {},
              querySelector() { return null; }, querySelectorAll() { return []; },
            },
            strategyCandidateLabResults: { innerHTML: "" },
            strategyCandidateLabStatus: { textContent: "", dataset: {} },
            strategyCandidateLabSummary: { innerHTML: "" },
            strategyCandidateLabBody: { hidden: true },
            strategyCandidateLabToggle: {
              textContent: "打开工作台",
              setAttribute(name, value) { attrs[name] = value; },
              attrs,
            },
          };
        }
        let current = { id: "task-1", task_type: "strategy" };
        function makeController(ids, fetch = async (taskId) => ({ ...payload, task_id: taskId })) {
          return createStrategyCandidateLabController({
            $: (id) => ids[id] || null,
            storage,
            getSelectedTask: () => current,
            getSelectedTaskId: () => current.id,
            getBlockedReason: () => "",
            getStrategyCandidateLab: fetch,
          });
        }
        const view = (ids) => ({
          hidden: ids.strategyCandidateLabBody.hidden,
          expanded: ids.strategyCandidateLabToggle.attrs["aria-expanded"],
          label: ids.strategyCandidateLabToggle.textContent,
        });
        const toggleClick = {
          target: { closest: (selector) => (selector === "[data-candidate-lab-fold-toggle]" ? {} : null) },
          preventDefault() {},
        };
        const ids = makeIds();
        const controller = makeController(ids);
        await controller.selectTask(current);
        const initial = view(ids);
        const summaryAfterLoad = ids.strategyCandidateLabSummary.innerHTML;
        const handled = controller.handleClick(toggleClick);
        const opened = view(ids);
        const storedAfterOpen = foldKeys();
        current = { id: "task-2", task_type: "strategy" };
        await controller.selectTask(current);
        const otherTask = view(ids);
        current = { id: "task-1", task_type: "strategy" };
        await controller.selectTask(current);
        const backToFirst = view(ids);
        // A fresh page (new controller) reads the remembered choice.
        const reloadIds = makeIds();
        await makeController(reloadIds).selectTask(current);
        const reloaded = view(reloadIds);
        controller.handleClick(toggleClick);
        const closed = view(ids);
        const storedAfterClose = foldKeys();
        // A load error without a payload opens the body for its error card,
        // without remembering it.
        current = { id: "task-3", task_type: "strategy" };
        const errorIds = makeIds();
        const errorController = makeController(errorIds, async () => { throw new Error("offline"); });
        await errorController.selectTask(current);
        const onError = view(errorIds);
        const storedAfterError = foldKeys();
        const errorStatus = errorIds.strategyCandidateLabStatus?.textContent || "";
        // Batch 4 review B8: the user can fold an error-opened body, and a
        // re-render does not force it open again.
        errorController.handleClick(toggleClick);
        errorController.renderAvailability?.();
        const errorClosed = view(errorIds);
        // Unreadable storage reads as collapsed.
        const brokenIds = makeIds();
        await createStrategyCandidateLabController({
          $: (id) => brokenIds[id] || null,
          storage: { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } },
          getSelectedTask: () => ({ id: "task-1", task_type: "strategy" }),
          getSelectedTaskId: () => "task-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => payload,
        }).selectTask({ id: "task-1", task_type: "strategy" });
        process.stdout.write(JSON.stringify({
          initial, summaryAfterLoad, handled, opened, storedAfterOpen, otherTask, backToFirst,
          reloaded, closed, storedAfterClose, onError, storedAfterError, errorClosed, errorStatus, broken: view(brokenIds),
        }));
        """.replace("PAYLOAD", _payload_js())
    )
    collapsed = {"hidden": True, "expanded": "false", "label": "打开工作台"}
    expanded = {"hidden": False, "expanded": "true", "label": "收起工作台"}
    key = "marvis_strategy_candidate_lab_expanded_v1:task-1"
    assert data["initial"] == collapsed
    assert "第 3 步 样本设计 · 待补充" in data["summaryAfterLoad"]
    assert data["handled"] is True
    assert data["opened"] == expanded
    assert data["storedAfterOpen"] == {key: "1"}
    assert data["otherTask"] == collapsed
    assert data["backToFirst"] == expanded
    assert data["reloaded"] == expanded
    assert data["closed"] == collapsed
    assert data["storedAfterClose"] == {}
    assert data["onError"] == expanded
    assert data["storedAfterError"] == {}
    # Batch 4 review B8/U3: folding an error-opened body sticks, and the status
    # line is a plain sentence instead of the raw error.
    assert data["errorClosed"] == collapsed
    assert data["errorStatus"] == "策略候选读取失败，可在工作台里重新读取。"
    assert "offline" not in data["errorStatus"]
    assert data["broken"] == collapsed


def test_deep_links_into_a_launcher_open_the_body_first():
    data = _run_node(
        """
        import { createStrategyCandidateLabController } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";
        const store = new Map();
        const storage = { getItem: (key) => store.get(key) ?? null, setItem: (key, value) => store.set(key, value), removeItem: (key) => store.delete(key) };
        const launcher = { open: false, hidden: false, scrolled: null, scrollIntoView(options) { this.scrolled = options; } };
        const form = { closest: (selector) => (selector === ".candidate-lab-launcher" ? launcher : null) };
        const selectors = [];
        const panel = {
          classList: { toggle() {} }, dataset: {}, setAttribute() {},
          querySelector(selector) { selectors.push(selector); return selector.includes('data-candidate-lab-workflow="strategy_pool_add"') ? form : null; },
          querySelectorAll() { return []; },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabBody: { hidden: true },
          strategyCandidateLabToggle: { textContent: "", setAttribute() {} },
        };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          storage,
          getSelectedTask: () => ({ id: "task-1", task_type: "strategy" }),
          getSelectedTaskId: () => "task-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => ({ task_id: "task-1", can_start: true, candidates: {}, workflow: {} }),
        });
        await controller.selectTask({ id: "task-1", task_type: "strategy" });
        const before = ids.strategyCandidateLabBody.hidden;
        const unknown = controller.reveal("missing_workflow");
        const unsafe = controller.reveal('x"] , *');
        const revealed = controller.reveal("strategy_pool_add");
        process.stdout.write(JSON.stringify({
          before, unknown, unsafe, revealed,
          after: ids.strategyCandidateLabBody.hidden,
          launcherOpen: launcher.open,
          scrolled: launcher.scrolled,
          stored: Object.fromEntries(store),
          selectors: selectors.filter((selector) => selector.includes("data-candidate-lab-workflow=\\"strategy_pool_add\\"")).length,
        }));
        """
    )
    assert data["before"] is True
    assert data["unknown"] is False
    assert data["unsafe"] is False
    assert data["revealed"] is True
    assert data["after"] is False
    assert data["launcherOpen"] is True
    assert data["scrolled"] == {"block": "nearest"}
    assert data["stored"] == {"marvis_strategy_candidate_lab_expanded_v1:task-1": "1"}

    controller_js = (STATIC / "js" / "v2" / "strategy_candidate_lab_controller.js").read_text(encoding="utf-8")
    # The five result-card deep links (frontier materialize, auto continuation,
    # split candidate, threshold adjust, prune) all reveal through the fold.
    assert controller_js.count("revealStrategyToolLauncher(panel(), launcher);") == 1
    reveal_helper = controller_js[controller_js.index("function revealLauncher(launcher) {"):]
    reveal_helper = reveal_helper[: reveal_helper.index("\n  }\n")]
    assert reveal_helper.index("setExpanded(true, { persist: true });") < reveal_helper.index(
        "revealStrategyToolLauncher(panel(), launcher);"
    )
    assert controller_js.count("      revealLauncher(launcher);\n") == 5
    # Fold elements are looked up by id only, never through the panel.
    for name in ("function applyFold()", "function renderSummary()"):
        body = controller_js[controller_js.index(name):]
        body = body[: body.index("\n  }\n")]
        assert "root." not in body and "panel()." not in body, name


def test_fold_markup_keeps_head_summary_and_status_outside_the_body():
    index_html = (STATIC / "index.html").read_text(encoding="utf-8")
    lab = index_html[index_html.index('id="strategyCandidateLabPanel"'):index_html.index('id="agentConversationPanel"')]
    body_start = lab.index('<div id="strategyCandidateLabBody" class="candidate-lab-body" hidden>')
    body_end = lab.index("</div><!-- /#strategyCandidateLabBody -->")
    body = lab[body_start:body_end]
    for inside in (
        'id="strategyWorkbenchState"',
        'data-candidate-lab-workflow="strategy_workbench_preferences"',
        'class="candidate-lab-layout"',
        'class="candidate-lab-launchers"',
        'id="strategyCandidateLabResults"',
        'id="strategyAssetStabilityPanel"',
    ):
        assert inside in body, inside
    head = lab[:body_start]
    for outside in (
        'class="candidate-lab-head"',
        'data-candidate-lab-retry="1"',
        'id="strategyCandidateLabSummary"',
        'id="strategyCandidateLabStatus"',
    ):
        assert outside in head, outside
    toggle = head[head.rindex("<button", 0, head.index('id="strategyCandidateLabToggle"')):]
    toggle = toggle[: toggle.index("</button>")]
    assert "data-candidate-lab-fold-toggle" in toggle
    assert 'aria-expanded="false"' in toggle
    assert 'aria-controls="strategyCandidateLabBody"' in toggle
    assert "打开工作台" in toggle
    assert "<details" not in head  # a button, not a details element (no toggle-persist side effect)

    css = (STATIC / "css" / "v2-strategy.css").read_text(encoding="utf-8")
    assert re.search(r"\.candidate-lab-body\[hidden\] \{\s*display: none;\s*\}", css)
    for selector in (".candidate-lab-summary-fact {", ".strategy-candidate-lab {"):
        rule = css[css.index(selector):]
        rule = rule[: rule.index("}")]
        # Flat card: no drawn border (border-radius is fine).
        assert not re.search(r"\bborder(-(top|right|bottom|left|color|width|style))?:[ ]*(?![ ]|0;|none;)", rule), (selector, rule)
    assert "background: var(--card);" in css[css.index(".strategy-candidate-lab {"):][:600]


def test_threshold_and_prune_deep_links_open_the_collapsed_body():
    """The real result-card deep links (threshold adjust, prune) open the fold
    before they reveal the interactive-tree launcher. Stubs mirror
    test_interactive_tree_threshold_controller_requires_explicit_pointer_and_refreshes."""
    _run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        class FakeSelect {
          constructor(fieldName, value = "") {
            this.dataset = { candidateLabField: fieldName };
            this.options = [];
            this._value = value;
            this.form = null;
          }
          set innerHTML(html) {
            this.options = Array.from(
              html.matchAll(/<option value="([^"]*)"([^>]*)>(.*?)<\\/option>/g),
            ).map((match) => {
              const attrs = match[2];
              const data = {};
              for (const [attribute, key] of [
                ["candidate-lab-projection", "candidateLabProjection"],
                ["source-tree-id", "sourceTreeId"],
                ["node-id", "nodeId"],
                ["operation", "operation"],
                ["feature", "feature"],
                ["current-threshold", "currentThreshold"],
              ]) {
                const value = attrs.match(
                  new RegExp(`data-${attribute}="([^"]*)"`),
                );
                if (value) data[key] = value[1];
              }
              return {
                value: match[1],
                selected: false,
                dataset: data,
              };
            });
            this.value = "";
          }
          set value(value) {
            this._value = String(value);
            for (const option of this.options) {
              option.selected = option.value === this._value;
            }
          }
          get value() { return this._value; }
          get selectedOptions() {
            return this.options.filter((option) => option.selected);
          }
          closest(selector) {
            if (selector === "[data-candidate-lab-field]") return this;
            return selector.includes("interactive_tree_revision")
              ? this.form
              : null;
          }
        }

        const sourceTreeId =
          "candidate-asset-0123456789abcdef0123456789abcdef";
        const nodeId = "node-0123456789abcdef0123";
        const operation = new FakeSelect(
          "interactive_tree_operation",
          "prune_subtree",
        );
        operation.options = [
          { value: "prune_subtree", selected: true, dataset: {} },
          {
            value: "adjust_split_threshold",
            selected: false,
            dataset: {},
          },
        ];
        const source = new FakeSelect("interactive_tree_source_id");
        const node = new FakeSelect("interactive_tree_node_id");
        const threshold = {
          value: "",
          dataset: { candidateLabField: "interactive_tree_threshold" },
          closest(selector) {
            if (selector === "[data-candidate-lab-field]") return this;
            return selector.includes("interactive_tree_revision")
              ? form
              : null;
          },
        };
        const reason = {
          value: "人工确认 score 新阈值",
          dataset: { candidateLabField: "interactive_tree_reason" },
          closest() { return null; },
        };
        const feature = { textContent: "" };
        const currentThreshold = { textContent: "" };
        const help = { textContent: "" };
        const errorTarget = { textContent: "" };
        const hiddenClasses = new Set(["hidden"]);
        const thresholdPanel = {
          classList: {
            contains: (name) => hiddenClasses.has(name),
            toggle(name, force) {
              if (force) hiddenClasses.add(name);
              else hiddenClasses.delete(name);
            },
          },
        };
        const launcher = { open: false };
        const fields = new Map([
          ["interactive_tree_operation", operation],
          ["interactive_tree_source_id", source],
          ["interactive_tree_node_id", node],
          ["interactive_tree_threshold", threshold],
          ["interactive_tree_reason", reason],
        ]);
        const form = {
          dataset: { candidateLabWorkflow: "interactive_tree_revision" },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-form-error]") {
              return errorTarget;
            }
            if (selector === "[data-candidate-lab-tree-threshold-panel]") {
              return thresholdPanel;
            }
            if (selector === "[data-candidate-lab-tree-threshold-feature]") {
              return feature;
            }
            if (selector === "[data-candidate-lab-tree-current-threshold]") {
              return currentThreshold;
            }
            if (selector === "[data-candidate-lab-tree-help]") return help;
            const match = selector.match(
              /data-candidate-lab-field="([^"]+)"/,
            );
            return match ? (fields.get(match[1]) || null) : null;
          },
          querySelectorAll() { return []; },
          closest(selector) {
            return selector === ".candidate-lab-launcher" ? launcher : null;
          },
          reset() {},
        };
        operation.form = form;
        source.form = form;
        node.form = form;
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector(selector) {
            return selector.includes("interactive_tree_revision")
              ? form
              : null;
          },
          querySelectorAll() { return []; },
        };
        const store = new Map();
        const storage = { getItem: (key) => store.get(key) ?? null, setItem: (key, value) => store.set(key, value), removeItem: (key) => store.delete(key) };
        const toggleAttrs = {};
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
          strategyCandidateLabBody: { hidden: true },
          strategyCandidateLabToggle: { textContent: "", setAttribute(name, value) { toggleAttrs[name] = value; } },
        };
        const tree = {
          kind: "automatic_tree",
          candidate_id: "candidate-tree",
          detail: {
            source_tree_id: sourceTreeId,
            asset_id: sourceTreeId,
          },
          pointers: {
            nodes: [{
              node_id: nodeId,
              kind: "split",
              feature: "score",
              threshold: 600,
              is_visible: true,
              is_frontier: false,
              can_prune: true,
            }],
            eligible_prunes: [{
              source_tree_id: sourceTreeId,
              node_id: nodeId,
              operation: "prune_subtree",
            }],
            eligible_threshold_adjustments: [{
              source_tree_id: sourceTreeId,
              node_id: nodeId,
              operation: "adjust_split_threshold",
              feature: "score",
              current_threshold: 600,
            }],
          },
        };
        const payload = {
          task_id: "task-1",
          can_start: true,
          blocked_reason: null,
          workflow: {},
          pools: {},
          strategies: {},
          candidates: {
            automatic_tree: {
              latest: null,
              total: 1,
              truncated: false,
              all: [tree],
            },
          },
        };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          storage,
          getSelectedTask: () => ({ id: "task-1", task_type: "strategy" }),
          getSelectedTaskId: () => "task-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => payload,
        });
        await controller.selectTask({ id: "task-1", task_type: "strategy" });
        assert.equal(ids.strategyCandidateLabBody.hidden, true);
        assert.equal(launcher.open, false);

        const thresholdButton = {
          dataset: { sourceTreeId, nodeId, feature: "score", currentThreshold: "600" },
          closest(selector) {
            return selector === "[data-candidate-lab-interactive-tree-threshold]" ? this : null;
          },
        };
        assert.equal(controller.handleClick({ target: thresholdButton, preventDefault() {} }), true);
        assert.equal(ids.strategyCandidateLabBody.hidden, false);
        assert.equal(toggleAttrs["aria-expanded"], "true");
        assert.equal(launcher.open, true);
        assert.equal(store.get("marvis_strategy_candidate_lab_expanded_v1:task-1"), "1");

        // Collapse again, then the prune deep link reopens it.
        controller.handleClick({
          target: { closest: (selector) => (selector === "[data-candidate-lab-fold-toggle]" ? {} : null) },
          preventDefault() {},
        });
        assert.equal(ids.strategyCandidateLabBody.hidden, true);
        launcher.open = false;
        const pruneButton = {
          dataset: { sourceTreeId, nodeId },
          closest(selector) {
            return selector === "[data-candidate-lab-interactive-tree-prune]" ? this : null;
          },
        };
        assert.equal(controller.handleClick({ target: pruneButton, preventDefault() {} }), true);
        assert.equal(ids.strategyCandidateLabBody.hidden, false);
        assert.equal(launcher.open, true);
        assert.equal(operation.value, "prune_subtree");
        process.stdout.write("null");
        """
    )
