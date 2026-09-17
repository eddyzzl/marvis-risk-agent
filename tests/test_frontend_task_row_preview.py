import json
import subprocess
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _run_preview_module(script_body: str) -> None:
    module_url = (STATIC_DIR / "js" / "task-row-preview.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
const preview = await import({json.dumps(module_url)});
{script_body}
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout or "node test failed")


def test_task_row_preview_lists_full_name_owner_and_created_time():
    _run_preview_module(
        r"""
const html = preview.taskRowPreviewHtml({
  name: "2026-08-20 模型验证批次 (2个模型)",
  typeLabel: "模型验证",
  ownerLabel: "验证人员",
  ownerName: "张三",
  createdAt: "2026-08-20T06:32:00+08:00",
});
assert.match(html, /task-row-preview-title/);
assert.match(html, /2026-08-20 模型验证批次 \(2个模型\)/);
assert.match(html, /模型验证/);
assert.match(html, /验证人员/);
assert.match(html, /张三/);
assert.match(html, /创建时间/);
assert.doesNotMatch(html, /task-row-meta/);
const escaped = preview.taskRowPreviewHtml({
  name: "<b>xss</b>",
  ownerName: "<img>",
});
assert.match(escaped, /&lt;b&gt;xss&lt;\/b&gt;/);
assert.match(escaped, /&lt;img&gt;/);
"""
    )


def test_task_row_preview_sits_to_the_right_of_the_sidebar():
    _run_preview_module(
        r"""
globalThis.window = { innerWidth: 1280, innerHeight: 800 };
const card = {
  style: {},
  getBoundingClientRect() {
    return { width: 320, height: 120, top: Number.parseFloat(this.style.top), left: Number.parseFloat(this.style.left) };
  },
};
const row = {
  getBoundingClientRect() {
    return { top: 96, right: 280, bottom: 136, left: 12, width: 268, height: 40 };
  },
};
const sidebar = {
  getBoundingClientRect() {
    return { top: 0, right: 280, bottom: 800, left: 0, width: 280, height: 800 };
  },
};
preview.positionTaskRowPreview(card, row, sidebar);
assert.equal(card.style.left, "288px");
assert.equal(card.style.top, "96px");
"""
    )


def test_app_wires_single_line_task_rows_to_preview_card():
    app_js = _read_static("app.js")
    row_html = app_js.split("function taskRowInnerHtml", 1)[1].split(
        "function createTaskRowShell",
        1,
    )[0]
    assert "task-row-name" in row_html
    assert "task-row-meta" not in row_html
    assert "task-row-date" not in row_html
    assert "bindTaskRowPreview" in app_js
    assert "pointerdown" in _read_static("js/task-row-preview.js")
    assert "suppressedShell" in _read_static("js/task-row-preview.js")
    assert "function dismiss(shell)" in _read_static("js/task-row-preview.js")
    assert 'id="taskRowPreview"' in _read_static("index.html")
    css = _read_static("css/task-shell.css")
    assert "text-overflow: ellipsis" in css
    assert "white-space: nowrap" in css.split(".task-row-name {", 1)[1].split("}", 1)[0]
