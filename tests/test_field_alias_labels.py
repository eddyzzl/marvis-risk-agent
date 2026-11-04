"""V3 5.7: confirmed field business names label feature-analysis and modeling
result tables; the cells keep the physical codes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.data.field_aliases import attach_field_labels
from tests.javascript_source import slice_function

ROOT = Path(__file__).resolve().parents[1]


def _tables():
    return [
        {"title": "特征指标", "columns": ["特征", "IV", "KS"], "rows": [["dpd_30", 0.3, 0.2], ["age", 0.1, 0.05]]},
        {"title": "高相关特征对", "columns": ["特征A", "特征B", "相关系数"], "rows": [["dpd_30", "age", 0.8]]},
        {"title": "模型指标", "columns": ["指标", "值"], "rows": [["dpd_30", "0.4"]]},
    ]


def test_labels_name_only_codes_in_field_columns_and_cells_stay_codes():
    before = _tables()
    labelled = attach_field_labels(before, {"dpd_30": "近30天逾期次数", "age": "age", "unused": "未出现"})
    assert labelled[0]["field_labels"] == {"dpd_30": "近30天逾期次数"}
    assert labelled[1]["field_labels"] == {"dpd_30": "近30天逾期次数"}
    # A metric table's first column is not a field column.
    assert "field_labels" not in labelled[2]
    assert [table["rows"] for table in labelled] == [table["rows"] for table in _tables()]
    assert before == _tables()
    assert attach_field_labels(before, {}) == before


def test_composer_attaches_workspace_names_to_feature_and_modeling_tables_only(monkeypatch, tmp_path):
    names = {"dpd_30": "近30天逾期次数"}
    monkeypatch.setattr(
        "marvis.repositories.data_workspace.DataWorkspaceRepository.get_or_default",
        lambda self, task_id: SimpleNamespace(semantic_mapping=SimpleNamespace(business_names=names)),
    )
    composer = PlanMessageComposer(load_output=lambda *_: None, db_path=tmp_path / "marvis.db")
    monkeypatch.setattr(
        "marvis.agent.plan_message_composer.render_tool_output",
        lambda *args, **kwargs: ("text", _tables()),
    )
    plan = SimpleNamespace(task_id="t1")

    def step(tool):
        return SimpleNamespace(id="s", output_ref=None, tool_ref=SimpleNamespace(tool=tool))

    composer._trusted_step_evidence = lambda *a, **k: None
    composer._trusted_presenter_inputs = lambda *a, **k: None
    composer._trusted_terminal_artifacts = lambda *a, **k: None
    _, feature_tables = composer._render_step_output(plan, step("compute_feature_metrics"), {})
    _, model_tables = composer._render_step_output(plan, step("train_model"), {})
    _, vintage_tables = composer._render_step_output(plan, step("vintage_curve"), {})
    assert feature_tables[0]["field_labels"] == names and model_tables[0]["field_labels"] == names
    assert all("field_labels" not in table for table in vintage_tables)
    assert feature_tables[0]["rows"] == _tables()[0]["rows"]


def test_conversation_table_shows_the_name_under_the_code():
    app_js = (ROOT / "marvis" / "static" / "app.js").read_text(encoding="utf-8")
    start = app_js.index("const FIELD_LABEL_HEADERS")
    headers = app_js[start : app_js.index("\n", start)]
    tables = slice_function(app_js, "function agentMessageTablesHtml(")
    script = f"""
const escapeHtml = (v) => String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;");
const platformLabelSource = (v) => v;
const driverColumnKindFromHeader = () => "text";
const columnFractions = () => new Map();
const driverTableCellHtml = (cell) => ({{ cls: "", html: escapeHtml(cell), platform: false }});
const driverTableChartHtml = () => "";
const renderWorkflowDataWidget = ({{ tableHtml }}) => tableHtml;
{headers}
{tables}
const html = agentMessageTablesHtml({{ id: "m", metadata: {{ tables: [
  {{ title: "特征指标", columns: ["特征", "IV"], rows: [["dpd_30", "0.3"], ["age", "0.1"]], field_labels: {{ dpd_30: "近30天<逾期>" }} }},
] }} }});
process.stdout.write(JSON.stringify({{ html }}));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script], check=False, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    html = json.loads(result.stdout)["html"]
    assert 'translate="no">dpd_30<small class="field-alias">近30天&lt;逾期></small></td>' in html
    assert 'translate="no">age</td>' in html


def _node(script: str) -> dict:
    result = subprocess.run(["node", "--input-type=module", "-e", script], check=False, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_composer_labels_the_screen_widget_codes(monkeypatch, tmp_path):
    names = {"dpd_30": "近30天逾期次数", "age": "年龄", "other": "未出现"}
    monkeypatch.setattr(
        "marvis.repositories.data_workspace.DataWorkspaceRepository.get_or_default",
        lambda self, task_id: SimpleNamespace(semantic_mapping=SimpleNamespace(business_names=names)),
    )
    composer = PlanMessageComposer(load_output=lambda *_: None, db_path=tmp_path / "marvis.db")
    screen = {"selected": ["dpd_30"], "ranked": [["age", 0.1]], "leakage": [], "unusable": [["const_x", "常量"]], "scores": {}}
    labelled = composer._with_screen_field_labels("t1", screen)
    assert labelled["field_labels"] == {"dpd_30": "近30天逾期次数", "age": "年龄"}
    assert labelled["selected"] == ["dpd_30"] and "field_labels" not in screen


def test_screen_widget_and_summary_bars_show_names_and_keep_codes():
    data = _node(
        """
import { renderScreenGateTable } from "./marvis/static/js/v2/screen_gate_controller.js";
import { resultSummaryHtml } from "./marvis/static/js/result-summary.js";
const screen = { selected: ["dpd_30"], ranked: [["dpd_30", 0.3], ["age", 0.1]], leakage: [], suspected: [], unusable: [],
  scores: { dpd_30: { ks: 0.3, iv: 0.4 }, age: { ks: 0.1, iv: 0.05 } }, thresholds: {}, field_labels: { dpd_30: "近30天逾期次数" } };
const widget = renderScreenGateTable({ id: "m1", metadata: { kind: "gate", screen } });
const bars = resultSummaryHtml([{ title: "特征指标", columns: ["特征", "IV"], rows: [["dpd_30", "0.4"], ["age", "0.05"]],
  field_labels: { dpd_30: "近30天逾期次数" } }]);
process.stdout.write(JSON.stringify({ widget, bars }));
"""
    )
    widget, bars = data["widget"], data["bars"]
    assert "<th>业务含义</th>" in widget
    assert 'title="近30天逾期次数">近30天逾期次数</td>' in widget
    assert 'class="screen-pick" value="dpd_30"' in widget
    assert '<td class="screen-feat" translate="no">dpd_30</td>' in widget
    assert 'dpd_30<small class="field-alias">近30天逾期次数</small></span>' in bars
    assert 'title="age">age</span>' in bars


def test_collections_quick_action_only_opens_the_inspector_panel():
    data = _node(
        """
import { agentQuickReplies, agentQuickRepliesHtml } from "./marvis/static/js/agent-quick-replies.js";
import { createWorkbenchChrome } from "./marvis/static/js/workbench-chrome.js";
const replies = agentQuickReplies([{ role: "assistant", metadata: {
  inspector_actions: [{ label: "打开催收参考工作台", item: "collectionWorkspacePanel" }] } }]);
const html = agentQuickRepliesHtml(replies);
const attrs = (extra = {}) => ({ attributes: new Set(), hasAttribute(n) { return this.attributes.has(n); },
  setAttribute(n, v) { this[n] = v; }, ...extra });
const details = { open: false, tagName: "DETAILS", querySelector: () => ({ focus() { focused.push("summary"); } }) };
const focused = [];
const panel = attrs({ tagName: "SECTION", hidden: false, querySelector: () => details, scrollIntoView() {} });
panel.attributes.add("data-inspector-item");
const hiddenPanel = attrs({ tagName: "SECTION", hidden: true, querySelector: () => null });
hiddenPanel.attributes.add("data-inspector-item");
const body = { contains: (node) => node === panel || node === hiddenPanel, querySelectorAll: () => [] };
const inspector = { hidden: true }, toggle = attrs({});
const artifacts = { hidden: false }, artifactsToggle = attrs({});
const ids = { taskInspector: inspector, taskInspectorBody: body, taskInspectorToggle: toggle, taskArtifacts: artifacts,
  taskArtifactsToggle: artifactsToggle, collectionWorkspacePanel: panel, riskContextPanel: hiddenPanel };
const chrome = createWorkbenchChrome({ doc: { getElementById: (id) => ids[id] || null }, storage: null });
const hiddenOpened = chrome.openInspectorItem("riskContextPanel");
const unknownOpened = chrome.openInspectorItem("nope");
const stillClosed = inspector.hidden;
const opened = chrome.openInspectorItem("collectionWorkspacePanel");
process.stdout.write(JSON.stringify({ replies, html, hiddenOpened, unknownOpened, stillClosed, opened,
  inspectorHidden: inspector.hidden, expanded: toggle["aria-expanded"], detailsOpen: details.open, focused, artifactsHidden: artifacts.hidden }));
"""
    )
    assert data["replies"] == [{"label": "打开催收参考工作台", "command": "", "kind": "inspector", "item": "collectionWorkspacePanel"}]
    assert 'data-agent-inspector-item="collectionWorkspacePanel"' in data["html"]
    assert "data-agent-quick-reply=" not in data["html"]
    assert data["hiddenOpened"] is False and data["unknownOpened"] is False and data["stillClosed"] is True
    assert data["opened"] is True and data["inspectorHidden"] is False and data["expanded"] == "true"
    assert data["detailsOpen"] is True and data["focused"] == ["summary"] and data["artifactsHidden"] is True
