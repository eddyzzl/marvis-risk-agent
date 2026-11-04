"""V3 plan batch 4 frontend gaps: hover full names, first-paint brand lines and
a 按状态 group that does not move when its task is opened."""

import json
import subprocess
from pathlib import Path

from tests.javascript_source import slice_function

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


def _run_module(script: str):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_status_group_ignores_selection_only_status_inputs():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    group_key = slice_function(app, "function taskAttentionGroupKey(task)")
    data = _run_module(
        """
import { resolveTaskStatus, workflowStatusSnapshot } from "./marvis/static/js/task-status.js";
let selected = false;
// Opening a task adds the plan rail's live snapshot, an open gate and the
// loaded batch payload; none of them may move the row to another group.
function taskStatusContext(task) {
  return {
    usesPlanWorkflow: true,
    planStatus: selected ? { key: "awaiting_confirm" } : workflowStatusSnapshot(task.workflow_status),
    openGate: selected,
    batch: selected && task.task_type === "validation_batch" ? { status: "failed", items: [] } : null,
    batchItem: null,
    batchStatus: "",
  };
}
"""
        + group_key
        + """
const tasks = [
  { id: "plan", task_type: "modeling", status: "created", workflow_status: "running" },
  { id: "idle", task_type: "modeling", status: "created", workflow_status: null },
];
const before = tasks.map(taskAttentionGroupKey);
selected = true;
const after = tasks.map(taskAttentionGroupKey);
process.stdout.write(JSON.stringify({ before, after }));
"""
    )
    assert data["before"] == ["attention:running", "attention:recent"]
    assert data["after"] == data["before"]


def test_brand_name_ships_as_two_lines_in_static_markup():
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    assert '<h1 id="platformName"><b>MARVIS</b><small>全能风控智能体</small></h1>' in index


def test_truncating_headers_and_tiles_carry_the_full_text_on_hover():
    data = _run_module(
        """
import { railSummaryHtml } from "./marvis/static/js/workbench-chrome.js";
import { validationMetricCardHtml } from "./marvis/static/js/result-cards.js";
import { renderTaskSnapshot } from "./marvis/static/js/task-workspace-view.js";
const esc = (v) => String(v ?? "").replace(/"/g, "&quot;");
process.stdout.write(JSON.stringify({
  rail: railSummaryHtml({ title: "很长的步骤标题", detail: "第 3 步，共 6 步" }, esc),
  card: validationMetricCardHtml({ tiles: [] }, { title: "一个非常长的模型名称_v2" }),
}));
"""
    )
    assert '<b title="很长的步骤标题">' in data["rail"]
    assert '<small title="第 3 步，共 6 步">' in data["rail"]
    assert '<b title="一个非常长的模型名称_v2 的效果与稳定性">' in data["card"]
    summary = (STATIC / "js" / "result-summary.js").read_text(encoding="utf-8")
    assert '<b class="num" title="${escapeHtml(tile.value)}">' in summary
    view = (STATIC / "js" / "task-workspace-view.js").read_text(encoding="utf-8")
    assert '<strong title="${escapeHtml(value)}">' in view
