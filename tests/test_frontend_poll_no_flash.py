"""Polling must not flash the page (2026-10-20 user feedback).

A batch task polls about once a second. Each poll rendered the field
confirmation panel twice: first from the scan result's copy of the contract
(still 待确认 · revision 1), then from the contract endpoint (已就绪 ·
revision 2), so the panel swapped form <-> text every second. Forced renders
also rebuilt identical step rails, task title glyphs and scan summaries.
"""

import json
import subprocess
from pathlib import Path

from tests.javascript_source import slice_function

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_scan_copy_of_the_contract_never_replaces_the_endpoint_record():
    app_js = _read("app.js")
    render_scan_call = 'renderValidationInputContract(result.validation_input_contract || null, {\n    taskId: workbenchTaskId(),\n    source: "scan",\n  });'
    assert render_scan_call in app_js
    body = slice_function(app_js, "function renderValidationInputContract(")
    guard = body.index('options.source === "scan"')
    assert 'panel.dataset.renderSource === "api"' in body[guard:guard + 200]
    assert "panel.dataset.renderTaskId === taskId" in body[guard:guard + 300]
    # The guard runs before anything is hidden or rebuilt.
    assert guard < body.index("panel.classList.add(\"hidden\")")
    assert 'panel.dataset.renderSource = options.source === "scan" ? "scan" : "api";' in body


def test_unchanged_markup_is_not_written_again():
    module = (STATIC / "js" / "ui-utils.js").as_uri()
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        "let writes = 0;",
        "const fakeDocument = { createElement: () => ({ innerHTML: '' }) };",
        "const el = { ownerDocument: fakeDocument, _html: '', textContent: 'a',",
        "  get innerHTML() { return this._html; }, set innerHTML(v) { writes += 1; this._html = v; } };",
        "const first = m.setHtmlIfChanged(el, '<b>x</b>');",
        "const same = m.setHtmlIfChanged(el, '<b>x</b>');",
        "const changed = m.setHtmlIfChanged(el, '<b>y</b>');",
        "const textSame = m.setTextIfChanged(el, 'a');",
        "const textChanged = m.setTextIfChanged(el, 'b');",
        "process.stdout.write(JSON.stringify({ first, same, changed, writes, textSame, textChanged, text: el.textContent }));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {
        "first": True, "same": False, "changed": True, "writes": 2,
        "textSame": False, "textChanged": True, "text": "b",
    }


def test_forced_renders_keep_identical_steps_titles_and_summaries():
    app_js = _read("app.js")
    stepper = slice_function(app_js, "function renderWorkflowStepper(")
    assert 'stepper.innerHTML = ""' not in stepper
    assert "nextSteps.innerHTML !== stepper.innerHTML" in stepper
    assert "stepper.replaceChildren(...nextSteps.childNodes)" in stepper
    assert 'setTextIfChanged(railTitle, "验证步骤")' in stepper
    assert 'setHtmlIfChanged($("currentTaskGlyph")' in app_js
    assert 'setHtmlIfChanged($("scanSummary"), [' in app_js
    view = _read("js/task-workspace-view.js")
    assert "setHtmlIfChanged(snapshot, [" in view
    assert 'setTextIfChanged(title, taskDisplayName?.(selectedTask) || "")' in view


def test_batch_polls_slowly_while_every_model_waits_on_the_user():
    # Eval S5: ~49 requests per 10 s while the batch only waited on the user.
    from tests.test_frontend_validation_batch import _run_batch_module

    _run_batch_module(
        """
const panel = {
  innerHTML: "", classList: { toggle() {} }, setAttribute() {},
  addEventListener() {}, querySelectorAll() { return []; },
};
let itemStatus = "awaiting_confirmation";
let task = { id: "parent-1", task_type: "validation_batch", run_mode: "agent" };
const delays = [];
const controller = batch.createValidationBatchPanelController({
  api: async () => ({
    batch: { parent_task_id: "parent-1", status: "running", item_count: 1 },
    items: [{ child_task_id: "m1", status: itemStatus }],
  }),
  getElementById: () => panel,
  getSelectedTask: () => task,
  schedulePoll: (_callback, delay) => { delays.push(delay); return delays.length; },
  cancelPoll: () => {},
});
await controller.selectTask(task);
assert.equal(delays.at(-1), 5000, "every model waits on the user: slow poll");
itemStatus = "running";
await controller.selectTask(task, { force: true });
assert.equal(delays.at(-1), 1500, "a running model polls at the normal rate");
// The manual batch view keeps the normal rate even while models wait.
itemStatus = "awaiting_confirmation";
task = { ...task, run_mode: "manual" };
await controller.selectTask(task, { force: true });
assert.equal(delays.at(-1), 1500);
"""
    )
