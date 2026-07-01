"""Smoke-only frontend tests.

These tests grep the static JS/HTML for v2 strings. They DO NOT load the
frontend or exercise any flow; they only guard against accidental deletion
of v2-shaped fields/endpoints. Real frontend behavior must be exercised in
a browser against the running FastAPI app.
"""

import json
import subprocess
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _css_rule(css: str, selector: str) -> str:
    start = css.index(f"{selector} {{")
    end = css.index("}", start)
    return css[start:end]


def _css_vars(rule: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in rule.splitlines():
        line = line.strip()
        if not line.startswith("--") or ":" not in line:
            continue
        name, value = line.rstrip(";").split(":", 1)
        values[name.strip()] = value.strip()
    return values


def _hex_rgb(value: str) -> tuple[int, int, int]:
    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"expected #RRGGBB color, got {value!r}")
    return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))


def _relative_luminance(value: str) -> float:
    def channel(component: int) -> float:
        normalized = component / 255
        if normalized <= 0.04045:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(component) for component in _hex_rgb(value))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(left: str, right: str) -> float:
    light = max(_relative_luminance(left), _relative_luminance(right))
    dark = min(_relative_luminance(left), _relative_luminance(right))
    return (light + 0.05) / (dark + 0.05)


def test_artifact_metrics_object_values_render_as_readable_key_values():
    module_url = (STATIC_DIR / "js" / "v2" / "artifact_view.js").as_uri()
    script = (
        f"import({json.dumps(module_url)}).then((module) => {{"
        "  const html = module.metricsHtml({ stats: { mean: 0.532, std: 0.042 } });"
        "  process.stdout.write(html);"
        "});"
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mean: 0.532；std: 0.042" in result.stdout
    assert '{"mean":0.532' not in result.stdout


def test_v2_artifact_preview_uses_tabular_numeric_metrics():
    css = _read_static("styles.css")

    assert '[data-v2-artifact-view="true"] :is(.metrics-preview, .dataset-preview table)' in css
    assert 'font-feature-settings: "tnum"' in css
    assert '[data-v2-artifact-view="true"] .metrics-preview td' in css
    assert "text-align: right" in css


def test_agent_message_polling_uses_incremental_cursor_when_safe():
    app_js = _read_static("app.js")

    assert "function agentMessageCanPollIncrementally" in app_js
    assert "message?.metadata?.optimistic || message?.metadata?.streaming" in app_js
    assert "function mergeIncrementalAgentMessages" in app_js
    assert "?after_id=${encodeURIComponent(lastMessageId)}" in app_js
    assert "if (payload.incremental)" in app_js


def test_v2_plan_rail_fetch_errors_are_visible_and_retryable():
    app_js = _read_static("app.js")
    plan_js = _read_static("js/v2/plan_rail_controller.js")

    assert "createPlanRailController" in app_js
    assert "planRailController.render({ force, renderSignatures })" in app_js
    assert "const v2PlanFetchErrors = new Map()" in plan_js
    assert "if (!response.ok) throw new Error(`HTTP ${response.status}`)" in plan_js
    assert "计划读取失败" in plan_js
    assert "当前显示的是上次缓存的计划" in plan_js
    assert "const fetchErrorBanner = fetchError" in plan_js
    assert "return fetchErrorBanner + phasesHtml + startControl;" in plan_js
    assert "data-plan-rail-retry" in plan_js
    assert "function retryFetch" in plan_js


def _agent_timeline_items_for(
    messages: list[dict],
    visible_stages: list[str],
    *,
    frozen_snapshots: list[dict] | None = None,
    selected_task_id: str | None = None,
) -> list[dict]:
    module_url = (STATIC_DIR / "js" / "agent-conversation-view.js").as_uri()
    snapshots = frozen_snapshots or []
    snapshot_task = selected_task_id or ("test-task" if snapshots else "")
    script = "\n".join(
        [
            f"import * as conversation from {json.dumps(module_url)};",
            (
                f"const selectedTaskId = {json.dumps(snapshot_task)};"
                if snapshot_task
                else "const selectedTaskId = null;"
            ),
            "const taskFrozenSectionSnapshots = new Map();",
            f"const messages = {json.dumps(messages, ensure_ascii=False)};",
            f"const visibleStages = {json.dumps(visible_stages, ensure_ascii=False)};",
            "const agentMessages = messages;",
            f"const __seedSnapshots = {json.dumps(snapshots, ensure_ascii=False)};",
            "if (__seedSnapshots.length && selectedTaskId) {",
            "  taskFrozenSectionSnapshots.set(selectedTaskId, __seedSnapshots);",
            "}",
            "const snapshotsByTrigger = conversation.agentFrozenSnapshotsByTriggerId({",
            "  selectedTaskId,",
            "  taskFrozenSectionSnapshots,",
            "  agentMessages: messages,",
            "});",
            "const items = conversation.agentTimelineItems(messages, visibleStages, { snapshotsByTrigger }).map((item) => {",
            "  if (item.type === 'stage') return { type: item.type, stage: item.stage };",
            "  if (item.type === 'frozen') return {",
            "    type: item.type,",
            "    triggerMessageId: item.snapshot?.triggerMessageId || '',",
            "    stage: item.snapshot?.stage || '',",
            "  };",
            "  return { type: item.type, contents: item.messages.map((message) => message.content) };",
            "});",
            "process.stdout.write(JSON.stringify(items));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _agent_messages_html_for(messages: list[dict], label_stage: str | None = None) -> str:
    module_url = (STATIC_DIR / "js" / "agent-conversation-view.js").as_uri()
    script = "\n".join(
        [
            f"import * as conversation from {json.dumps(module_url)};",
            "function agentStageLabel(stage) { return 'Agent'; }",
            "function agentMessageHtml(message, labelStage = message?.stage, options = {}) {",
            "  const label = message.role === 'user' ? '' : agentStageLabel(labelStage);",
            "  return `${options.hideMeta || !label ? '' : `<meta>${label}</meta>`}<body>${message.content}</body>`;",
            "}",
            f"const messages = {json.dumps(messages, ensure_ascii=False)};",
            "process.stdout.write(conversation.agentMessagesHtml(",
            f"  messages, {json.dumps(label_stage, ensure_ascii=False)},",
            "  { agentStageLabel, agentMessageHtml },",
            "));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _agent_report_messages_for_display(messages: list[dict]) -> list[dict]:
    module_url = (STATIC_DIR / "js" / "agent-conversation-view.js").as_uri()
    script = "\n".join(
        [
            f"import * as conversation from {json.dumps(module_url)};",
            f"const messages = {json.dumps(messages, ensure_ascii=False)};",
            "process.stdout.write(JSON.stringify(conversation.agentReportMessagesForDisplay(messages)));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _render_agent_markdown(markdown: str) -> str:
    script = "\n".join(
        [
            "import { renderAgentMarkdown } from './marvis/static/js/render-agent.js';",
            f"process.stdout.write(renderAgentMarkdown({json.dumps(markdown, ensure_ascii=False)}));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _workflow_step_statuses_for(task: dict, notebook_steps: list[dict]) -> list[str]:
    app_js = _read_static("app.js")
    failure_start = app_js.index("function taskFailedDuringScan")
    failure_end = app_js.index("function taskFailureActionStatusTitle", failure_start)
    workflow_start = app_js.index("function workflowIndex")
    workflow_end = app_js.index("function workflowStepStatusLabel", workflow_start)
    notebook_start = app_js.index("function notebookStepTone")
    notebook_end = app_js.index("function plannedReproducibilitySteps", notebook_start)
    script = "\n".join(
        [
            "const selectedTaskId = 'task-1';",
            f"const selectedTask = {json.dumps(task, ensure_ascii=False)};",
            f"let latestNotebookSteps = {json.dumps(notebook_steps, ensure_ascii=False)};",
            "const workflowSteps = [{id:'scan'}, {id:'notebook'}, {id:'metrics'}, {id:'report'}];",
            "function taskBusyAction() { return null; }",
            "function taskServerBusyAction() { return null; }",
            app_js[failure_start:failure_end],
            app_js[workflow_start:workflow_end],
            app_js[notebook_start:notebook_end],
            "const statuses = workflowSteps.map((_, index) => workflowStepStatus(index, workflowIndex(selectedTask.status)));",
            "process.stdout.write(JSON.stringify(statuses));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _notebook_step_tones_for(task: dict, notebook_steps: list[dict]) -> list[str]:
    app_js = _read_static("app.js")
    failure_start = app_js.index("function taskStopped")
    failure_end = app_js.index("function workflowIndex", failure_start)
    notebook_start = app_js.index("function notebookStepTone")
    notebook_end = app_js.index("function plannedReproducibilitySteps", notebook_start)
    script = "\n".join(
        [
            f"let selectedTask = {json.dumps(task, ensure_ascii=False)};",
            app_js[failure_start:failure_end],
            app_js[notebook_start:notebook_end],
            f"const steps = {json.dumps(notebook_steps, ensure_ascii=False)};",
            "process.stdout.write(JSON.stringify(steps.map((step) => notebookStepTone(step.status))));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _normalized_notebook_steps_for(notebook_steps: list[dict], notebook_cells: list[dict]) -> list[dict]:
    app_js = _read_static("app.js")
    normalize_start = app_js.index("function normalizeNotebookSteps")
    normalize_end = app_js.index("function renderNotebookSteps", normalize_start)
    script = "\n".join(
        [
            app_js[normalize_start:normalize_end],
            f"const steps = {json.dumps(notebook_steps, ensure_ascii=False)};",
            f"const cells = {json.dumps(notebook_cells, ensure_ascii=False)};",
            "process.stdout.write(JSON.stringify(normalizeNotebookSteps(steps, cells)));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _task_action_status_for(task: dict) -> dict | None:
    app_js = _read_static("app.js")
    stopped_start = app_js.index("function taskStopped")
    stopped_end = app_js.index("function taskBusyAction", stopped_start)
    status_start = app_js.index("function taskFailureActionStatusMessage")
    status_end = app_js.index("function actionFailureStatusTitle", status_start)
    script = "\n".join(
        [
            f"let selectedTask = {json.dumps(task, ensure_ascii=False)};",
            "let captured = null;",
            "function setActionStatus(title, kind, detail) { captured = { title, kind, detail }; }",
            "function setActionErrorDetail(detail) { captured = { title: '', kind: 'clear', detail }; }",
            app_js[stopped_start:stopped_end],
            app_js[status_start:status_end],
            "setTaskFailureActionStatus(selectedTask);",
            "process.stdout.write(JSON.stringify(captured));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _task_display_status_for(
    task: dict,
    *,
    action_message: str = "已停止当前动作。",
    action_kind: str = "stopped",
) -> dict:
    app_js = _read_static("app.js")
    stopped_start = app_js.index("function taskStopped")
    stopped_end = app_js.index("function taskBusyAction", stopped_start)
    pill_start = app_js.index("function actionStatusPill")
    pill_end = app_js.index("function describeActionStatus", pill_start)
    label_start = app_js.index("function statusLabel")
    label_end = app_js.index("function notebookReproducibilityComplete", label_start)
    script = "\n".join(
        [
            f"const statusLabels = {json.dumps({'scanned': '已扫描', 'failed': '失败'}, ensure_ascii=False)};",
            f"const task = {json.dumps(task, ensure_ascii=False)};",
            app_js[stopped_start:stopped_end],
            app_js[pill_start:pill_end],
            app_js[label_start:label_end],
            "statusLabels.review_required = '待复核';",
            "process.stdout.write(JSON.stringify({",
            "  rowLabel: taskStatusLabel(task),",
            "  rowTone: taskStatusTone(task),",
            f"  heroPill: actionStatusPill({json.dumps(action_message, ensure_ascii=False)}, {json.dumps(action_kind)}),",
            "}));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_frontend_uses_v2_task_actions_only():
    app_js = _read_static("app.js")

    assert "run-notebook" not in app_js
    assert "report-template" not in app_js
    assert "api/tasks/${taskId}/notebook" in app_js
    assert "api/tasks/${taskId}/metrics" in app_js
    assert "api/tasks/${taskId}/report" in app_js
    assert "api/tasks/${selectedTaskId}/report/download" in app_js
    assert "api/tasks/${selectedTaskId}/analysis/download" in app_js
    assert "api/tasks/${selectedTaskId}/report/preview" in app_js
    assert 'data-step-action="downloadWordReport"' in app_js
    assert 'data-step-action="downloadExcelAnalysis"' in app_js
    assert 'data-step-action="previewWordReport"' in app_js
    assert "下载Word" in app_js
    assert "下载Excel" in app_js
    assert "预览" in app_js
    assert "下载Word报告" not in app_js
    assert "下载Excel分析" not in app_js
    assert "预览Word报告" not in app_js
    assert 'data-step-action="downloadReport"' not in app_js


def test_word_report_preview_dialog_uses_task_dialog_backdrop():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")

    assert 'id="wordPreviewDialog"' in index_html
    assert 'class="task-dialog word-preview-dialog"' in index_html
    assert 'id="wordPreviewFrame"' in index_html
    assert ".task-dialog::backdrop" in styles_css
    assert ".word-preview-dialog" in styles_css
    assert ".word-preview-frame" in styles_css


def test_step_rail_narrow_layout_keeps_titles_horizontal_and_stacks_report_actions():
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")
    layout_resize_js = _read_static("js/layout-resize.js")

    title_start = styles_css.index(".step-title {")
    title_end = styles_css.index("}", title_start)
    title_rule = styles_css[title_start:title_end]
    assert "white-space: nowrap;" in title_rule
    assert "text-overflow: ellipsis;" in title_rule
    assert ".step-copy {" in styles_css
    head_start = styles_css.index(".step-head {")
    head_end = styles_css.index("}", head_start)
    head_rule = styles_css[head_start:head_end]
    assert "align-items: center;" in head_rule
    assert "align-items: flex-start;" not in head_rule

    copy_start = styles_css.index(".step-copy {")
    copy_end = styles_css.index("}", copy_start)
    copy_rule = styles_css[copy_start:copy_end]
    assert "padding-top" not in copy_rule

    assert "container: step-rail / inline-size;" in styles_css

    renderer_start = app_js.index("function renderWorkflowStepper")
    renderer_end = app_js.index("function formatDate", renderer_start)
    renderer = app_js[renderer_start:renderer_end]
    assert '<span class="step-copy">' in renderer
    assert '<div class="step-sub">' not in renderer

    assert "export const PROGRESS_WIDTH_MIN = 314;" in layout_resize_js
    assert "clamp(stored.progress, PROGRESS_WIDTH_MIN, PROGRESS_WIDTH_MAX)" in layout_resize_js
    assert "clamp(startProgress - deltaX, PROGRESS_WIDTH_MIN, PROGRESS_WIDTH_MAX)" in layout_resize_js
    assert 'from "./js/layout-resize.js"' in app_js


def test_plan_rail_matches_validation_stepper_with_nested_subtasks():
    plan_js = _read_static("js/v2/plan_rail_controller.js")
    v2_css = _read_static("css/v2-workbench.css")

    plan_start = plan_js.index("function planRailHtml")
    plan_end = plan_js.index("function render({", plan_start)
    plan_renderer = plan_js[plan_start:plan_end]
    substeps_start = plan_js.index("function planSubstepGroupHtml")
    substeps_end = plan_js.index("function driverHasBlockingError", substeps_start)
    substeps_renderer = plan_js[substeps_start:substeps_end]

    assert "function planPhaseStatus" in plan_js
    assert "function planPhaseHint" in plan_js
    assert "function planRetryControlHtml" in plan_js
    assert "function planSubstepGroupHtml" in plan_js
    assert "const phaseNumber = phaseIndex + 1;" in plan_renderer
    assert 'class="step plan-rail-step' in plan_renderer
    assert '<span class="step-number">${phaseNumber}</span>' in plan_renderer
    assert '<strong class="step-title">${escapeHtml(phase)}</strong>' in plan_renderer
    assert "planSubstepGroupHtml(phaseSteps, phaseNumber)" in plan_renderer
    assert '<section class="notebook-step-group plan-rail-substeps">' in substeps_renderer
    assert '<h4>子任务 · ${steps.length}</h4>' in substeps_renderer
    assert "const subNumber = parentNumber ? `${parentNumber}.${index + 1}` : `${index + 1}`;" in substeps_renderer
    assert '<span class="notebook-step-no">${escapeHtml(subNumber)}</span>' in substeps_renderer
    assert '<span class="plan-substep-copy">' in substeps_renderer
    assert "const description = step.description || step.summary || PLAN_STEP_HINTS" in substeps_renderer
    assert 'const descriptionHtml = description ? `<small>${escapeHtml(description)}</small>` : "";' in substeps_renderer
    assert 'const retry = status === "failed" ? planRetryControlHtml(step) : "";' in substeps_renderer
    assert "`<strong>${escapeHtml(step.title || \"未命名步骤\")}</strong>`" in substeps_renderer
    assert "descriptionHtml" in substeps_renderer
    assert "retry" in substeps_renderer
    assert 'data-plan-retry-step="${escapeHtml(stepId)}"' in plan_js
    assert 'class="plan-retry-inputs"' in plan_js
    assert ': "";' in substeps_renderer
    assert '<section class="plan-rail-phase"' not in plan_renderer
    assert "plan-rail-major-number" not in plan_renderer
    assert "plan-rail-phase-name" not in plan_renderer
    assert "plan-rail-substep" not in plan_renderer
    assert "let number = 0;" not in plan_renderer

    plan_step_rule = _css_rule(v2_css, ".plan-rail-step")
    assert "cursor: default" in plan_step_rule

    plan_substeps_rule = _css_rule(v2_css, ".plan-rail-substeps")
    assert "margin-top: 6px" in plan_substeps_rule
    assert ".plan-substep-copy" in v2_css
    assert "display: grid" in _css_rule(v2_css, ".plan-substep-copy")
    assert "white-space: nowrap" in _css_rule(v2_css, ".plan-substep-copy small")
    assert "grid-column: 1 / -1" in _css_rule(v2_css, ".plan-step-retry")
    assert "width: 100%" in _css_rule(v2_css, ".plan-step-retry")

    assert ".plan-rail-phase" not in v2_css
    assert ".plan-rail-major-number" not in v2_css


def test_layout_resize_controller_restores_drags_and_persists_widths():
    script = """
import assert from "node:assert/strict";
import {
  createLayoutResizeController,
  PROGRESS_WIDTH_MAX,
  SIDEBAR_WIDTH_MIN,
} from "./marvis/static/js/layout-resize.js";

const styleValues = new Map([
  ["--sidebar-width", "400px"],
  ["--progress-width", "400px"],
]);
const root = {
  style: {
    setProperty(name, value) {
      styleValues.set(name, value);
    },
  },
};
const bodyClasses = new Set();
const body = {
  classList: {
    add(value) {
      bodyClasses.add(value);
    },
    remove(value) {
      bodyClasses.delete(value);
    },
    contains(value) {
      return bodyClasses.has(value);
    },
  },
};
const storageData = {
  marvis_layout: JSON.stringify({ sidebar: 320, progress: 999 }),
};
const storage = {
  getItem(key) {
    return storageData[key] || null;
  },
  setItem(key, value) {
    storageData[key] = value;
  },
};
const listeners = {};
const removed = [];
const controller = createLayoutResizeController({
  body,
  clamp: (value, min, max) => Math.min(Math.max(value, min), max),
  getComputedStyleFn: () => ({
    getPropertyValue(name) {
      return styleValues.get(name) || "";
    },
  }),
  root,
  storage,
  windowObj: {
    addEventListener(name, fn) {
      listeners[name] = fn;
    },
    removeEventListener(name, fn) {
      removed.push([name, fn]);
      if (listeners[name] === fn) delete listeners[name];
    },
  },
});

controller.restoreLayoutWidths();
assert.equal(styleValues.get("--sidebar-width"), `${SIDEBAR_WIDTH_MIN}px`);
assert.equal(styleValues.get("--progress-width"), `${PROGRESS_WIDTH_MAX}px`);

let prevented = false;
controller.startResizeDrag("left", {
  clientX: 100,
  preventDefault() {
    prevented = true;
  },
});
assert.equal(prevented, true);
assert.equal(bodyClasses.has("is-resizing"), true);
listeners.pointermove({ clientX: 400 });
assert.equal(styleValues.get("--sidebar-width"), "520px");
listeners.pointerup();
assert.equal(bodyClasses.has("is-resizing"), false);
assert.equal(removed.length, 2);
assert.equal(JSON.parse(storageData.marvis_layout).sidebar, 520);

styleValues.set("--progress-width", "400px");
let keyPrevented = false;
controller.handleResizeKey("right", {
  key: "ArrowRight",
  shiftKey: false,
  preventDefault() {
    keyPrevented = true;
  },
});
assert.equal(keyPrevented, true);
assert.equal(styleValues.get("--progress-width"), "388px");
assert.equal(JSON.parse(storageData.marvis_layout).progress, 388);
process.stdout.write("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_plan_rail_retry_step_posts_edited_inputs():
    plan_js = _read_static("js/v2/plan_rail_controller.js")
    v2_css = _read_static("css/v2-workbench.css")

    retry_text_body = _slice_function(plan_js, "function planRetryInputsText")
    retry_fields_body = _slice_function(plan_js, "function planRetrySchemaFieldsHtml")
    retry_parse_structured_body = _slice_function(plan_js, "function collectPlanRetryStructuredInputs")
    retry_scope_body = _slice_function(plan_js, "function planRetryScopeHtml")
    retry_body = _slice_function(plan_js, "async function retryPlanStep")
    click_body = _slice_function(plan_js, "function handleClick")

    assert "step?.failure_envelope?.editable_input_schema" in retry_text_body
    assert 'Object.prototype.hasOwnProperty.call(spec, "default")' in retry_text_body
    assert 'data-plan-retry-input-key="${encodedKey}"' in retry_fields_body
    assert 'data-plan-retry-input-type="${typeLabel}"' in retry_fields_body
    assert "plan-retry-schema-fields" in retry_fields_body
    assert "collectPlanRetryStructuredInputs(form)" in _slice_function(plan_js, "function parsePlanRetryInputs")
    assert "[data-plan-retry-input-key]" in retry_parse_structured_body
    assert "step?.failure_envelope" in retry_scope_body
    assert "downstream_reset_steps" in retry_scope_body
    assert "plan-retry-scope" in retry_scope_body
    assert "grid-template-columns: repeat(auto-fit, minmax(160px, 1fr))" in _css_rule(v2_css, ".plan-retry-schema-fields")
    assert "color: var(--text-muted)" in _css_rule(v2_css, ".plan-retry-scope")
    assert 'button?.dataset?.planRetryStep || ""' in retry_body
    assert 'parsePlanRetryInputs(button.closest("[data-plan-step-retry]"))' in retry_body
    assert "JSON.stringify({ inputs })" in retry_body
    assert "v2PlanCache.delete(taskId)" in retry_body
    assert "window.setTimeout(() => retryFetch(taskId), 1000)" in retry_body
    assert "[data-plan-retry-step]" in click_body
    assert "void retryPlanStep(planRetryButton);" in click_body
    assert "[data-plan-rail-retry]" in click_body


def test_completed_report_actions_render_below_step_copy_with_office_colors():
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")

    assert "function completedReportReadyForDownloads" in app_js
    ready_start = app_js.index("function completedReportReadyForDownloads")
    ready_end = app_js.index("function stepDownloadActionsHtml", ready_start)
    ready_helper = app_js[ready_start:ready_end]
    assert 'step.action === "report"' in ready_helper
    assert "selectedTask?.report_available === true" in ready_helper
    assert '["succeeded", "review_required"].includes(selectedTask?.status)' in ready_helper

    assert "function stepDownloadActionsHtml" in app_js
    downloads_start = app_js.index("function stepDownloadActionsHtml")
    downloads_end = app_js.index("function stepActionButtonHtml", downloads_start)
    downloads_renderer = app_js[downloads_start:downloads_end]
    assert '<div class="step-download-actions">' in downloads_renderer
    assert 'data-step-action="previewWordReport"' in downloads_renderer
    assert 'data-step-action="downloadWordReport"' in downloads_renderer
    assert 'data-step-action="downloadExcelAnalysis"' in downloads_renderer
    assert "step-action-button primary word" in downloads_renderer
    assert "step-action-button excel" in downloads_renderer

    step_renderer_start = app_js.index("function renderWorkflowStepper")
    step_renderer_end = app_js.index("function formatDate", step_renderer_start)
    step_renderer = app_js[step_renderer_start:step_renderer_end]
    assert "stepActionButtonHtml(step)" in step_renderer
    assert "stepDownloadActionsHtml(step)" in step_renderer
    assert step_renderer.index("stepActionButtonHtml(step)") < step_renderer.index("stepDownloadActionsHtml(step)")

    action_start = styles_css.index(".step-download-actions {")
    action_end = styles_css.index("}", action_start)
    action_rule = styles_css[action_start:action_end]
    assert "margin-left: 48px;" in action_rule
    assert "flex-wrap: nowrap;" in action_rule
    assert "justify-content: stretch;" in action_rule

    word_start = styles_css.index(".step-action-button.word")
    word_end = styles_css.index("}", word_start)
    word_rule = styles_css[word_start:word_end]
    assert "background: var(--download-word-bg);" in word_rule
    assert "border-color: var(--download-word-border);" in word_rule
    assert "color: var(--action-on-solid);" in word_rule
    assert "box-shadow: var(--button-solid-shadow)" in word_rule
    assert "var(--brand-primary)" not in word_rule

    word_hover_start = styles_css.index(".step-action-button.word:hover")
    word_hover_end = styles_css.index("}", word_hover_start)
    word_hover_rule = styles_css[word_hover_start:word_hover_end]
    assert "background: var(--download-word-bg-hover);" in word_hover_rule
    assert "border-color: var(--download-word-border-hover);" in word_hover_rule
    assert "box-shadow: var(--button-solid-shadow-hover)" in word_hover_rule

    excel_start = styles_css.index(".step-action-button.excel")
    excel_end = styles_css.index("}", excel_start)
    excel_rule = styles_css[excel_start:excel_end]
    assert "background: var(--download-excel-bg);" in excel_rule
    assert "border-color: var(--download-excel-border);" in excel_rule
    assert "color: var(--action-on-solid);" in excel_rule
    assert "box-shadow: var(--button-solid-shadow)" in excel_rule
    assert "var(--brand-primary)" not in excel_rule

    excel_hover_start = styles_css.index(".step-action-button.excel:hover")
    excel_hover_end = styles_css.index("}", excel_hover_start)
    excel_hover_rule = styles_css[excel_hover_start:excel_hover_end]
    assert "background: var(--download-excel-bg-hover);" in excel_hover_rule
    assert "border-color: var(--download-excel-border-hover);" in excel_hover_rule
    assert "box-shadow: var(--button-solid-shadow-hover)" in excel_hover_rule


def test_report_download_readiness_requires_generated_report_flag():
    app_js = _read_static("app.js")
    busy_start = app_js.index("function taskStopped")
    busy_end = app_js.index("function currentTaskSignature", busy_start)
    ready_start = app_js.index("function completedReportReadyForDownloads")
    ready_end = app_js.index("function stepDownloadActionsHtml", ready_start)
    script = "\n".join(
        [
            "let selectedTaskId = 'task-1';",
            "let globalBusyAction = null;",
            "const taskBusyActions = new Map();",
            app_js[busy_start:busy_end],
            app_js[ready_start:ready_end],
            "const step = { action: 'report' };",
            "selectedTask = { status: 'review_required', report_available: false, active_job_kind: null };",
            "const withoutReport = completedReportReadyForDownloads(step);",
            "selectedTask = { status: 'review_required', report_available: true, active_job_kind: 'report' };",
            "const busyReport = completedReportReadyForDownloads(step);",
            "selectedTask = { status: 'review_required', report_available: true, active_job_kind: null };",
            "const readyReport = completedReportReadyForDownloads(step);",
            "process.stdout.write(JSON.stringify({ withoutReport, busyReport, readyReport }));",
        ]
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "withoutReport": False,
        "busyReport": False,
        "readyReport": True,
    }


def test_generic_actions_and_composer_use_semantic_visual_tokens():
    styles_css = _read_static("styles.css")
    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))

    for token in [
        "--focus-ring",
        "--button-disabled-bg",
        "--action-on-solid",
        "--download-word-bg",
        "--download-word-bg-hover",
        "--download-excel-bg",
        "--download-excel-bg-hover",
        "--agent-send-stop-bg",
        "--agent-send-stop-bg-hover",
        "--agent-send-stop-shadow",
        "--agent-user-message-bg",
        "--agent-user-message-shadow",
    ]:
        assert token in root_vars

    for token in [
        "--focus-ring",
        "--button-disabled-bg",
        "--agent-composer-chip-bg-hover",
        "--agent-send-stop-bg",
        "--agent-send-stop-bg-hover",
        "--agent-send-stop-shadow",
        "--agent-user-message-bg",
        "--agent-user-message-shadow",
    ]:
        assert token in dark_vars

    disabled_rule = _css_rule(styles_css, ".button:disabled")
    assert "background: var(--button-disabled-bg);" in disabled_rule

    focus_start = styles_css.index(".button:focus-visible,")
    focus_end = styles_css.index("}", focus_start)
    focus_rule = styles_css[focus_start:focus_end]
    assert "outline: 3px solid var(--focus-ring);" in focus_rule


def test_stage_actions_capture_task_id_before_polling():
    app_js = _read_static("app.js")

    poll_start = app_js.index("async function pollValidationProgress")
    poll_end = app_js.index("async function validateCurrentTask", poll_start)
    poll_renderer = app_js[poll_start:poll_end]
    assert "taskId = selectedTaskId" in poll_renderer
    assert "const polledTask = findTaskInCache(taskId)" in poll_renderer
    assert "selectedTaskId === taskId" in poll_renderer

    for function_name in [
        "scanCurrentTask",
        "validateCurrentTask",
        "generateMetrics",
        "generateReport",
    ]:
        start = app_js.index(f"async function {function_name}")
        end = app_js.index("\n}\n", start)
        body = app_js[start:end]
        assert "const taskId = selectedTaskId;" in body
        assert "pollValidationProgress(" not in body or "taskId" in body


def test_selected_running_task_auto_polls_progress_after_refresh_or_reselect():
    app_js = _read_static("app.js")
    polling_js = _read_static("js/polling.js")

    assert "const progressPolls = createProgressPollRegistry();" in app_js
    assert "export function createProgressPollRegistry" in polling_js
    assert "existing.cancelled = true;" in polling_js
    assert "function ensureActiveTaskProgressPolling" in app_js

    ensure_start = app_js.index("function ensureActiveTaskProgressPolling")
    ensure_end = app_js.index("async function refreshTasks", ensure_start)
    ensure_body = app_js[ensure_start:ensure_end]
    assert "taskServerBusyAction(task)" in ensure_body
    assert "progressPolls.has(taskId)" in ensure_body
    assert "pollValidationProgress(terminalTaskStatuses, taskId, { background: true })" in ensure_body

    refresh_start = app_js.index("async function refreshTasks")
    refresh_end = app_js.index("async function scanCurrentTask", refresh_start)
    refresh_body = app_js[refresh_start:refresh_end]
    assert "syncSelectedTaskFromCache();" in refresh_body
    assert "ensureActiveTaskProgressPolling();" in refresh_body

    select_start = app_js.index("function selectTask")
    select_end = app_js.index("function deselectCurrentTask", select_start)
    select_body = app_js[select_start:select_end]
    assert "ensureActiveTaskProgressPolling(task);" in select_body

    poll_start = app_js.index("async function pollValidationProgress")
    poll_end = app_js.index("async function validateCurrentTask", poll_start)
    poll_body = app_js[poll_start:poll_end]
    assert "{ stopping = false, background = false } = {}" in poll_body
    assert "const claim = claimProgressPoll(progressPolls, taskId, { background });" in poll_body
    assert "if (!claim.claimed) return claim.existing.promise;" in poll_body
    assert "releaseProgressPoll(progressPolls, taskId, pollState)" in poll_body
    assert "if (selectedTaskId === taskId && !background)" in poll_body
    assert "if (selectedTaskId === taskId && !background) {" in poll_body


def test_create_dialog_enter_does_not_submit_textareas():
    app_js = _read_static("app.js")

    handler_start = app_js.index('event.key === "Enter"')
    handler_end = app_js.index("runAction(createTaskAndScan", handler_start)
    handler = app_js[handler_start:handler_end]
    assert 'event.target.closest("#taskDialog")' in handler
    assert 'event.target.tagName !== "TEXTAREA"' in handler
    assert "!event.isComposing" in handler


def test_pointer_focus_ring_only_shows_when_clicking_inside_form_controls():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")
    focus_ring_js = _read_static("js/focus-ring.js")

    assert "static/app.js?v=__MARVIS_STATIC_VERSION__" in index_html
    assert '<script type="module" src="static/app.js?v=__MARVIS_STATIC_VERSION__"></script>' in index_html
    assert "static/app.js?v=20260613-task-entry-welcome" not in index_html
    assert "static/app.js?v=20260613-review-fixes" not in index_html
    assert "static/app.js?v=20260613-task-entry-upload" not in index_html
    assert 'static/app.js?v=20260613-task-entry"' not in index_html
    assert "static/app.js?v=20260605-create-task-error" not in index_html
    assert "static/app.js?v=20260603-zero-rail-collapse" not in index_html
    assert "static/app.js?v=20260603-task-validator-icon" not in index_html
    assert "static/app.js?v=20260603-field-focus-ring" not in index_html
    assert "static/app.js?v=20260603-dark-masks" not in index_html

    assert "export function formControlFocusTarget(target)" in focus_ring_js
    assert "export function installFormControlFocusRingGuard" in focus_ring_js
    assert 'target?.closest?.("input, textarea, select")' in focus_ring_js
    assert "let lastPointerDownControl = null;" in focus_ring_js
    assert "let lastPointerDownAt = 0;" in focus_ring_js
    assert 'root.addEventListener("pointerdown", handleFormControlPointerDown, true);' in focus_ring_js
    assert 'root.addEventListener("mousedown", handleFormControlPointerDown, true);' in focus_ring_js
    assert 'root.addEventListener("touchstart", handleFormControlPointerDown, true);' in focus_ring_js
    assert 'root.addEventListener("click", handleFormControlLabelClick, true);' in focus_ring_js
    assert 'root.addEventListener("focusin", handleFormControlFocusIn, true);' in focus_ring_js
    assert 'root.addEventListener("focusout", handleFormControlFocusOut, true);' in focus_ring_js
    assert "const pointerFocusPending = now() - lastPointerDownAt < suppressionWindowMs;" in focus_ring_js
    focus_in_start = focus_ring_js.index("function handleFormControlFocusIn")
    focus_in_end = focus_ring_js.index("function handleFormControlFocusOut", focus_in_start)
    focus_in_body = focus_ring_js[focus_in_start:focus_in_end]
    assert 'control.classList.toggle(' in focus_in_body
    assert '"suppress-pointer-focus-ring"' in focus_in_body
    assert "pointerFocusPending && lastPointerDownControl !== control" in focus_in_body
    label_click_start = focus_ring_js.index("function handleFormControlLabelClick")
    label_click_end = focus_ring_js.index("root.addEventListener", label_click_start)
    label_click_body = focus_ring_js[label_click_start:label_click_end]
    assert 'event.target.closest?.("label")' in label_click_body
    assert "label.contains(focused)" in label_click_body
    assert "focused.id === label.htmlFor" in label_click_body
    assert 'focused.classList.add("suppress-pointer-focus-ring")' in label_click_body
    assert 'if (control) control.classList.remove("suppress-pointer-focus-ring");' in focus_ring_js
    assert 'from "./js/focus-ring.js"' in app_js
    assert "function formControlFocusTarget(target)" not in app_js
    assert "function installFormControlFocusRingGuard" not in app_js
    assert "installFormControlFocusRingGuard();" in app_js

    suppress_start = styles_css.index("input.suppress-pointer-focus-ring:focus-visible,")
    suppress_end = styles_css.index("}", suppress_start)
    suppress_rule = styles_css[suppress_start:suppress_end]
    assert "textarea.suppress-pointer-focus-ring:focus-visible" in suppress_rule
    assert "select.suppress-pointer-focus-ring:focus-visible" in suppress_rule
    assert "outline: none" in suppress_rule
    assert "box-shadow: none" in suppress_rule


def test_form_control_focus_ring_guard_handles_pointer_and_label_focus():
    script = """
import assert from "node:assert/strict";
import { installFormControlFocusRingGuard } from "./marvis/static/js/focus-ring.js";

function classList() {
  return {
    values: new Set(),
    add(value) {
      this.values.add(value);
    },
    remove(value) {
      this.values.delete(value);
    },
    toggle(value, enabled) {
      if (enabled) this.add(value);
      else this.remove(value);
    },
    contains(value) {
      return this.values.has(value);
    },
  };
}

function control(id) {
  const item = {
    id,
    classList: classList(),
    closest(selector) {
      return selector === "input, textarea, select" ? item : null;
    },
  };
  return item;
}

const listeners = {};
let time = 1000;
let active = null;
const timeouts = [];
installFormControlFocusRingGuard({
  activeElement: () => active,
  now: () => time,
  root: {
    addEventListener(name, fn, capture) {
      listeners[name] = { fn, capture };
    },
  },
  setTimeoutFn: (fn, delay) => {
    timeouts.push({ fn, delay });
  },
});

for (const name of ["pointerdown", "mousedown", "touchstart", "click", "focusin", "focusout"]) {
  assert.equal(listeners[name].capture, true);
}

const first = control("first");
const second = control("second");
listeners.pointerdown.fn({ target: first });
time += 100;
listeners.focusin.fn({ target: second });
assert.equal(second.classList.contains("suppress-pointer-focus-ring"), true);
listeners.focusout.fn({ target: second });
assert.equal(second.classList.contains("suppress-pointer-focus-ring"), false);

listeners.pointerdown.fn({ target: first });
time += 100;
listeners.focusin.fn({ target: first });
assert.equal(first.classList.contains("suppress-pointer-focus-ring"), false);

const label = {
  htmlFor: "",
  contains(node) {
    return node === second;
  },
  closest(selector) {
    return selector === "label" ? label : null;
  },
};
active = second;
listeners.click.fn({ target: label });
assert.equal(timeouts[0].delay, 0);
timeouts[0].fn();
assert.equal(second.classList.contains("suppress-pointer-focus-ring"), true);
process.stdout.write("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_initialization_failure_shows_service_connection_error():
    app_js = _read_static("app.js")
    init_body = _slice_function(app_js, "async function initializeApp")

    assert "initializeApp();" in app_js
    assert "服务连接失败，请检查后端是否运行。" in init_body
    assert 'setActionStatus("服务连接失败，请检查后端是否运行。", "error", detail)' in init_body
    assert "runAction(async () => {\n  await refreshTasks();" not in app_js


def test_create_task_payload_omits_notebook_contract_fields():
    app_js = _read_static("app.js")
    create_dialog_js = _read_static("js/create-task-dialog.js")

    for field in ["report_values"]:
        assert f"{field}:" in create_dialog_js

    for removed_field in [
        "algorithm:",
        "notebook_path:",
        "sample_path:",
        "pmml_path:",
        "dictionary_path:",
        "target_col:",
        "score_col:",
        "split_col:",
        "time_col:",
        "feature_columns:",
    ]:
        assert removed_field not in app_js
        assert removed_field not in create_dialog_js


def test_create_task_uses_single_model_name_field():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    create_dialog_js = _read_static("js/create-task-dialog.js")

    assert 'id="modelName"' in index_html
    assert "模型名称" in index_html
    assert 'id="modelVersion"' not in index_html
    assert "模型版本" not in index_html
    assert 'model_version: ""' in create_dialog_js
    assert '$("modelVersion")' not in app_js
    assert '$("modelVersion")' not in create_dialog_js
    assert "请先填写模型名称、验证人员和材料目录。" in create_dialog_js
    assert "请先填写模型名称、版本" not in app_js
    assert "请先填写模型名称、版本" not in create_dialog_js


def test_task_display_does_not_require_model_version_separator():
    app_js = _read_static("app.js")
    workspace_view_js = _read_static("js/task-workspace-view.js")

    assert "function taskDisplayName" in app_js
    assert "taskDisplayName," in app_js
    assert "taskDisplayName?.(selectedTask)" in workspace_view_js
    assert "${selectedTask.model_name} · ${selectedTask.model_version}" not in app_js
    assert "${task.model_name} · ${task.model_version}" not in app_js


def test_create_dialog_hides_v2_config_controls():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")

    for removed_element_id in [
        "algorithm",
        "targetCol",
        "scoreCol",
        "splitCol",
        "timeCol",
        "featureColumns",
        "notebookPath",
        "samplePath",
        "pmmlPath",
        "dictionaryPath",
        "draftDate",
        "revisionVersion",
        "revisionDate",
        "revisionAuthor",
        "createDataSourceSummary",
        "createDatasetSplitSummary",
        "revisionDescription",
    ]:
        assert f'id="{removed_element_id}"' not in index_html

    assert "验证配置" not in index_html
    assert "显式材料路径" not in index_html
    assert "Notebook 路径" not in index_html
    assert "样本路径" not in index_html
    assert "PMML 路径" not in index_html
    assert "数据字典路径" not in index_html
    assert "起草日期" not in index_html
    assert "修订版本" not in index_html
    assert "修订日期" not in index_html
    assert "修订人" not in index_html
    assert "数据来源说明" not in index_html
    assert "样本划分说明" not in index_html
    assert "修订说明" not in index_html
    assert 'optionalInputValue("notebookPath")' not in app_js
    assert 'optionalInputValue("samplePath")' not in app_js
    assert 'optionalInputValue("pmmlPath")' not in app_js
    assert 'optionalInputValue("dictionaryPath")' not in app_js

    for element_id in ["runModeManual", "runModeAgent", "modelName", "validator", "sourceDir"]:
        assert f'id="{element_id}"' in index_html


def test_create_dialog_omits_legacy_model_training_description_autofill():
    create_dialog_js = _read_static("js/create-task-dialog.js")

    # The legacy auto-filled "model_training_description" report value stays removed —
    # the modeling algorithm is now a real user choice (the create-dialog picker),
    # not an auto-derived training blurb.
    defaults_start = create_dialog_js.index("function defaultCreateReportValues")
    defaults_end = create_dialog_js.index("function prefillCreateTaskReportFields", defaults_start)
    defaults = create_dialog_js[defaults_start:defaults_end]
    assert '"TEXT:model_training_description"' not in defaults
    assert "MODEL_TRAINING_DESCRIPTIONS" not in create_dialog_js


def test_create_dialog_auto_fills_removed_report_values():
    create_dialog_js = _read_static("js/create-task-dialog.js")

    defaults_start = create_dialog_js.index("function defaultCreateReportValues")
    defaults_end = create_dialog_js.index("function prefillCreateTaskReportFields", defaults_start)
    defaults = create_dialog_js[defaults_start:defaults_end]

    assert 'const today = formatDateInput();' in defaults
    assert '"TEXT:draft_date": today' in defaults
    assert '"TEXT:revision_date": today' in defaults
    assert '"TEXT:revision_version": "V1"' in defaults
    assert '"TEXT:revision_author": seed.validator' in defaults
    assert '"TEXT:revision_description": "初稿"' in defaults
    assert (
        '"TEXT:model_overview": `为了更好的对xx用户进行授信环节风险管控，现开发${seed.modelName}模型，对xx客群做前置风险拦截，从授信申请阶段做好风险防范。`'
        in defaults
    )
    assert '"TEXT:model_scope": "本模型适用于xx渠道用户。"' in defaults
    assert '"TEXT:bad_sample_definition": "xx逾期 >= xx天"' in defaults
    assert '"TEXT:good_sample_definition": "xx未逾期"' in defaults
    assert '"TEXT:data_source_summary"' not in defaults
    assert '"TEXT:dataset_split_summary"' not in defaults


def test_create_dialog_uses_visual_run_mode_cards():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")

    assert 'class="run-mode-cards"' in index_html
    assert 'class="run-mode-card selected-tone-amber"' in index_html
    assert 'class="run-mode-card selected-tone-green"' in index_html
    assert 'class="run-mode-icon"' in index_html
    assert 'class="manual-checks"' in index_html
    assert 'class="manual-check"' in index_html
    assert 'class="robot-eye"' in index_html
    assert 'class="robot-signal"' in index_html
    assert 'class="robot-wrench"' in index_html
    assert 'class="robot-tablet"' in index_html
    assert 'name="runMode"' in index_html
    assert 'value="manual"' in index_html
    assert 'value="agent"' in index_html
    assert 'name="runMode" type="radio" value="manual" checked' not in index_html
    assert 'name="runMode" type="radio" value="agent" disabled' not in index_html
    assert "预留" not in index_html
    assert "后续" not in index_html
    assert 'class="mode-choice"' not in index_html

    assert ".run-mode-cards {" in styles_css
    assert ".run-mode-card {" in styles_css
    run_mode_rule = _css_rule(styles_css, ".run-mode-card")
    assert "border-radius: var(--radius)" in run_mode_rule
    assert "border-radius: var(--radius-control)" not in run_mode_rule
    assert ".run-mode-card:hover:not(.disabled) {\n" not in styles_css
    hover_start = styles_css.index(".run-mode-card:hover:not(.disabled):not(:has(input:checked))")
    hover_end = styles_css.index("}", hover_start)
    hover_rule = styles_css[hover_start:hover_end]
    assert "transform: translateY(-1px)" in hover_rule
    assert "0 0 24px" not in hover_rule
    assert ".run-mode-card:has(input:checked)" in styles_css
    checked_start = styles_css.index(".run-mode-card:has(input:checked)")
    checked_end = styles_css.index("}", checked_start)
    checked_rule = styles_css[checked_start:checked_end]
    assert "border-color: var(--run-mode-tone)" in checked_rule
    assert "border-color: var(--border)" not in checked_rule
    assert "box-shadow:" in checked_rule
    assert "0 0 24px color-mix(in srgb, var(--run-mode-tone) 24%, transparent)" in checked_rule

    focus_start = styles_css.index(".run-mode-card:focus-within")
    focus_end = styles_css.index("}", focus_start)
    focus_rule = styles_css[focus_start:focus_end]
    assert ".run-mode-card:focus-within:not(:has(input:checked))" in focus_rule
    assert ".run-mode-card:focus-within {" not in focus_rule
    assert "@keyframes run-mode-check-draw" in styles_css
    assert "@keyframes run-mode-robot-blink" in styles_css
    assert "@keyframes run-mode-robot-signal" in styles_css
    assert "@keyframes run-mode-robot-crank" in styles_css
    assert "@keyframes run-mode-robot-float" not in styles_css
    assert ".mode-choice" not in styles_css


def test_create_dialog_updates_run_mode_copy_by_task_type():
    index_html = _read_static("index.html")
    create_dialog_js = _read_static("js/create-task-dialog.js")
    task_types_js = _read_static("js/task-types.js")

    assert 'data-run-mode-description="manual"' in index_html
    assert 'data-run-mode-description="agent"' in index_html
    assert "function setRunModeDescription" in create_dialog_js
    assert 'setRunModeDescription("manual", definition.manualModeDescription);' in create_dialog_js
    assert 'setRunModeDescription("agent", definition.agentModeDescription);' in create_dialog_js
    assert "由验证人员逐步执行材料扫描、Notebook 验证与报告生成" not in index_html
    assert "智能解析材料、规划验证步骤并辅助生成验证报告" not in index_html

    definitions_start = task_types_js.index("export const taskTypeDefinitions = {")
    definitions_end = task_types_js.index("export const taskTypeDisplayOrder", definitions_start)
    definitions = task_types_js[definitions_start:definitions_end]
    expected_copy = {
        "data_join": [
            "用结构化控件确认主表、目标列、join key、去重策略，再执行左连接",
            "Agent 先读 schema 提议角色和键，汇总命中率/膨胀风险，等你确认后执行",
        ],
        "feature_analysis": [
            "选择指标并查看 IV/KS/AUC/PSI/coverage/lift/共线结果，导出分析报告",
            "Agent 根据字段和字典建议补算指标、解释异常特征，并按你的反馈重跑",
        ],
        "modeling": [
            "确认目标列、train/test/OOT 切分和算法，执行泄漏筛选、调参、训练和报告",
            "Agent 组织读样本、切分确认、泄漏筛选、调参训练与结果解释",
        ],
        "validation": [
            "逐步完成材料扫描、Notebook 复现、分数一致性、效果稳定性和报告生成",
            "Agent 辅助扫描材料、解释验证证据、推进确认步骤并起草验证报告",
        ],
        "strategy": [
            "识别评分列和目标列，生成候选规则，在回测前确认并查看收益权衡",
            "Agent 根据评分、目标和客群起草规则，回测通过率、坏账、swap 和收益权衡",
        ],
        "vintage": [
            "识别 cohort、MOB 和坏账列，计算 Vintage 曲线并展示风险趋势",
            "Agent 识别 Vintage 字段，计算曲线并解释 cohort 风险变化",
        ],
    }
    for task_type, copy_items in expected_copy.items():
        task_start = definitions.index(f"  {task_type}: {{")
        task_end = definitions.index("\n  },", task_start)
        task_definition = definitions[task_start:task_end]
        assert "manualModeDescription:" in task_definition
        assert "agentModeDescription:" in task_definition
        for copy in copy_items:
            assert copy in task_definition


def test_create_dialog_does_not_preselect_modes_or_modeling_algorithms():
    index_html = _read_static("index.html")
    create_dialog_js = _read_static("js/create-task-dialog.js")
    task_types_js = _read_static("js/task-types.js")

    assert 'name="runMode" type="radio" value="manual" checked' not in index_html
    assert 'name="runMode" type="radio" value="agent" checked' not in index_html
    assert 'name="modelAlgorithm" value="lgb" checked' not in index_html
    assert 'name="modelAlgorithm" value="xgb" checked' not in index_html
    assert 'name="modelAlgorithm" value="lr" checked' not in index_html
    assert 'name="modelAlgorithm" value="scorecard" checked' not in index_html

    definitions_start = task_types_js.index("export const taskTypeDefinitions = {")
    definitions_end = task_types_js.index("export const taskTypeDisplayOrder", definitions_start)
    definitions = task_types_js[definitions_start:definitions_end]
    assert 'defaultRunMode: "manual"' not in definitions
    assert 'defaultRunMode: "agent"' not in definitions
    assert definitions.count('defaultRunMode: ""') == 6

    dialog_start = create_dialog_js.index("function openTaskDialog")
    dialog_end = create_dialog_js.index("function openTaskDialogFromCard", dialog_start)
    dialog_body = create_dialog_js[dialog_start:dialog_end]
    assert "input.checked = false;" in dialog_body
    assert "resetModelAlgorithmChoices();" in dialog_body
    assert "updateAlgorithmFieldVisibility();" in dialog_body
    assert "definition.defaultRunMode ===" not in dialog_body

    assert "function resetModelAlgorithmChoices" in create_dialog_js
    apply_start = create_dialog_js.index("function applyTaskTypeToDialog")
    apply_end = create_dialog_js.index("function updateAlgorithmFieldVisibility", apply_start)
    apply_body = create_dialog_js[apply_start:apply_end]
    assert "checked: false" in apply_body
    assert "checked: definition.defaultRunMode" not in apply_body


def test_create_task_requires_run_mode_and_allows_agent_mode():
    create_dialog_js = _read_static("js/create-task-dialog.js")

    create_start = create_dialog_js.index("async function createTask")
    create_end = create_dialog_js.index("function bindMaterialSourceControls", create_start)
    create_renderer = create_dialog_js[create_start:create_end]

    assert "const selectedRunMode" in create_renderer
    assert "请选择执行模式。" in create_renderer
    assert "Agent 模式当前暂不支持创建任务，请选择手动模式。" not in create_renderer
    assert "run_mode: selectedRunMode" in create_renderer
    assert '?.value || "manual"' not in create_renderer


def test_create_dialog_moves_material_source_to_bottom_segment():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    create_dialog_js = _read_static("js/create-task-dialog.js")
    dialogs_js = _read_static("js/dialogs.js")
    styles_css = _read_static("styles.css")

    assert "报告初始内容" not in index_html
    task_info_start = index_html.index('<h3>任务信息</h3>')
    report_start = index_html.index('id="createTaskReportFields"')
    material_start = index_html.index('id="createTaskMaterialSection"')
    create_button_start = index_html.index('id="createTaskButton"')
    assert task_info_start < report_start < material_start < create_button_start

    task_info_section = index_html[task_info_start:report_start]
    assert 'id="sourceDir"' not in task_info_section
    assert index_html.index('id="createGoodSampleDefinition"') < index_html.index('id="sourceDir"')

    material_section = index_html[material_start:create_button_start]
    assert 'role="tablist"' in material_section
    assert 'id="materialSourcePathTab"' in material_section
    assert 'id="materialSourceUploadTab"' in material_section
    assert "文件路径" in material_section
    assert "文件上传" in material_section
    assert 'aria-selected="true"' in material_section
    assert 'aria-selected="false"' in material_section
    assert 'id="materialSourcePathPanel"' in material_section
    assert 'id="materialSourceUploadPanel"' in material_section
    assert 'class="material-source-panel material-upload-panel hidden"' not in material_section
    assert 'class="material-source-panel material-upload-panel"' in material_section
    assert 'id="sourceDir"' in material_section
    assert "材料目录" in material_section
    assert 'id="materialUploadInput" class="visually-hidden" type="file" multiple />' in material_section
    assert 'id="materialFolderUploadInput"' not in material_section
    assert "webkitdirectory" not in material_section
    assert 'class="material-upload-dropzone"' in material_section
    assert 'role="button"' in material_section
    assert 'tabindex="0"' in material_section
    assert 'aria-describedby="materialUploadStatus"' in material_section
    assert 'class="material-upload-icon"' in material_section
    assert 'id="materialUploadStatus"' in material_section
    assert "点击或拖拽上传" in material_section
    assert 'id="materialUploadFileButton"' not in material_section
    assert 'id="materialUploadFolderButton"' not in material_section
    assert ">选择文件</button>" not in material_section
    assert ">选择文件夹</button>" not in material_section
    assert "material-upload-actions" not in material_section
    assert "暂未开放" not in material_section

    assert "export function createMaterialSourceController" in dialogs_js
    assert "export function materialUploadSelectionText" in dialogs_js
    assert "export function renderMaterialUploadSelection" in dialogs_js
    assert "function bindDropzone()" in dialogs_js
    assert "captureFiles(input.files)" in dialogs_js
    assert "file?.webkitRelativePath" in dialogs_js
    assert 'dropzone.addEventListener("click", openFilePicker)' in dialogs_js
    assert 'dropzone.addEventListener("keydown", (event) =>' in dialogs_js
    assert "function walkDroppedEntry" in dialogs_js
    assert "typeof item.webkitGetAsEntry" in dialogs_js
    assert "captureFileItems(await droppedFileItems(event.dataTransfer))" in dialogs_js
    assert 'dropzone.classList.add("is-dragover")' in dialogs_js
    assert 'pathPanel.classList.toggle("hidden", !isPath)' in dialogs_js
    assert 'uploadPanel.classList.toggle("hidden", isPath)' in dialogs_js
    assert "function renderMaterialUploadSelection" not in app_js
    assert "onFilesChanged: (files) => renderMaterialUploadSelection({ files, getElementById: $ })" in app_js
    assert "createTaskDialog.bindMaterialSourceControls();" in app_js
    assert "materialSourceController.bindDropzone();" in create_dialog_js

    assert ".material-source-section" in styles_css
    assert ".material-source-segment" in styles_css
    assert ".material-source-tab" in styles_css
    assert ".material-upload-dropzone" in styles_css
    assert ".material-upload-dropzone.is-dragover" in styles_css
    assert ".material-upload-actions" not in styles_css
    segment_rule = _css_rule(styles_css, ".material-source-segment")
    assert "width: 100%;" in segment_rule
    assert "border-radius: var(--radius-control)" in segment_rule
    source_tab_rule = _css_rule(styles_css, ".material-source-tab")
    assert "border-radius: var(--radius-control)" in source_tab_rule


def test_material_upload_selection_renderer_summarizes_files():
    script = """
import assert from "node:assert/strict";
import {
  materialUploadSelectionText,
  renderMaterialUploadSelection,
} from "./marvis/static/js/dialogs.js";

assert.equal(materialUploadSelectionText([]), "请选择文件或文件夹。");
assert.equal(
  materialUploadSelectionText([
    { name: "a.csv", relativePath: "raw/a.csv" },
    { name: "b.csv", relativePath: "raw/b.csv" },
    { name: "c.csv", relativePath: "features/c.csv" },
    { name: "d.csv", relativePath: "features/sub/d.csv" },
  ]),
  "已选择 a.csv、b.csv、c.csv 等 4 个文件，包含 3 个目录。",
);

const elements = {
  materialUploadStatus: { textContent: "" },
};
renderMaterialUploadSelection({
  files: [{ name: "sample.parquet", relativePath: "oot/sample.parquet" }],
  getElementById: (id) => elements[id],
});
assert.equal(elements.materialUploadStatus.textContent, "已选择 sample.parquet，包含 1 个目录。");

renderMaterialUploadSelection({
  files: [],
  getElementById: (id) => elements[id],
});
assert.equal(elements.materialUploadStatus.textContent, "请选择文件或文件夹。");
process.stdout.write("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_create_task_upload_mode_posts_materials_before_creating_task():
    create_dialog_js = _read_static("js/create-task-dialog.js")
    api_js = _read_static("js/api.js")

    api_start = api_js.index("export async function api")
    api_end = api_js.index("export function sleep", api_start)
    api_body = api_js[api_start:api_end]
    assert "body instanceof FormData" in api_body
    assert '"Content-Type": "application/json"' not in api_body

    upload_start = create_dialog_js.index("async function uploadMaterialFiles")
    upload_end = create_dialog_js.index("async function createTask", upload_start)
    upload_body = create_dialog_js[upload_start:upload_end]
    assert "new FormData()" in upload_body
    assert 'formData.append("files"' in upload_body
    assert 'formData.append("relative_paths"' in upload_body
    assert 'api("api/material-uploads"' in upload_body

    create_start = create_dialog_js.index("async function createTask")
    create_end = create_dialog_js.index("function bindMaterialSourceControls", create_start)
    create_body = create_dialog_js[create_start:create_end]
    assert "await uploadMaterialFiles" in create_body
    assert "payload.source_dir = upload.source_dir" in create_body
    assert "文件上传暂未开放" not in create_body


def test_run_mode_cards_can_be_deselected_by_clicking_selected_card():
    app_js = _read_static("app.js")
    create_dialog_js = _read_static("js/create-task-dialog.js")

    assert "function bindRunModeDeselectableCards" in create_dialog_js
    assert "handleRunModeCardPointerDown" in create_dialog_js
    assert "handleRunModeCardClick" in create_dialog_js
    assert 'card.dataset.wasChecked = input.checked ? "true" : "false";' in create_dialog_js
    assert 'if (card.dataset.wasChecked !== "true") return;' in create_dialog_js
    assert "event.preventDefault();" in create_dialog_js
    assert "input.checked = false;" in create_dialog_js
    assert 'input.dispatchEvent(new Event("change", { bubbles: true }));' in create_dialog_js
    assert "bindRunModeDeselectableCards();" in app_js


def test_create_dialog_sections_are_unframed():
    styles_css = _read_static("styles.css")

    dialog_start = styles_css.index(".task-dialog {")
    dialog_end = styles_css.index("}", dialog_start)
    dialog_rule = styles_css[dialog_start:dialog_end]
    assert "width: min(600px, calc(100vw - 32px))" in dialog_rule

    section_start = styles_css.index(".task-form-section {")
    section_end = styles_css.index("}", section_start)
    section_rule = styles_css[section_start:section_end]
    assert "border:" not in section_rule
    assert "background:" not in section_rule
    assert "padding: 0" in section_rule

    form_control_start = styles_css.index("\ninput,\ntextarea,\nselect {\n  width: 100%;")
    form_control_end = styles_css.index("}", form_control_start)
    form_control_rule = styles_css[form_control_start:form_control_end]
    assert "border-radius: var(--radius-control)" in form_control_rule
    assert "border-radius: var(--radius)" not in form_control_rule


def test_create_dialog_scrolls_only_when_content_exceeds_viewport():
    styles_css = _read_static("styles.css")

    create_dialog_start = styles_css.index(".task-dialog:not(.environment-dialog) {")
    create_dialog_end = styles_css.index("}", create_dialog_start)
    create_dialog_rule = styles_css[create_dialog_start:create_dialog_end]
    assert "max-height: calc(100dvh - 32px)" in create_dialog_rule
    assert "\n  height:" not in create_dialog_rule
    assert "overflow: hidden" in create_dialog_rule

    head_start = styles_css.index(".dialog-head {")
    head_end = styles_css.index("}", head_start)
    head_rule = styles_css[head_start:head_end]
    assert "height: 55px" in head_rule
    assert "padding: 11px 16px" in head_rule

    panel_start = styles_css.index(".task-dialog:not(.environment-dialog) .task-dialog-panel {")
    panel_end = styles_css.index("}", panel_start)
    panel_rule = styles_css[panel_start:panel_end]
    assert "max-height: calc(100dvh - 32px)" in panel_rule
    assert "height: 100%" not in panel_rule
    assert "min-height: 0" in panel_rule
    assert "display: flex" in panel_rule
    assert "flex-direction: column" in panel_rule

    form_start = styles_css.index(".task-dialog:not(.environment-dialog) .task-form {")
    form_end = styles_css.index("}", form_start)
    form_rule = styles_css[form_start:form_end]
    assert "flex: 0 1 auto" in form_rule
    assert "min-height: 0" in form_rule
    assert "max-height: none" in form_rule
    assert "overflow-x: hidden" in form_rule
    assert "overflow-y: auto" in form_rule
    assert "overscroll-behavior: contain" in form_rule
    assert "grid-template-rows: auto auto auto auto auto minmax(0, auto)" in form_rule
    assert "minmax(19px, auto)" not in form_rule

    environment_start = styles_css.index(".environment-dialog {")
    environment_end = styles_css.index("}", environment_start)
    environment_rule = styles_css[environment_start:environment_end]
    assert "width: min(520px, calc(100vw - 32px))" in environment_rule
    assert "height: min(839px" not in environment_rule


def test_workbench_uses_middle_output_and_right_step_rail_layout():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")
    plan_js = _read_static("js/v2/plan_rail_controller.js")

    assert 'id="progressRail"' in index_html
    assert 'aria-label="验证步骤"' in index_html
    assert 'progressRail?.setAttribute("aria-label", "计划步骤");' in plan_js
    assert 'progressRail?.setAttribute("aria-label", "验证步骤");' in app_js
    assert 'id="taskSnapshot"' in index_html
    assert index_html.index('id="scanSection"') < index_html.index('id="notebookSection"')
    assert index_html.index('id="notebookSection"') < index_html.index('id="metricSection"')
    assert index_html.index('id="metricSection"') < index_html.index('id="progressRail"')
    assert 'class="progress-panel task-snapshot-panel"' not in index_html
    assert "<h3>当前任务</h3>" not in index_html
    assert "<h3>Word 输出</h3>" not in index_html
    assert 'id="wordDocumentEditor"' not in index_html
    assert "<h3>操作</h3>" not in index_html
    assert "按步骤执行，也会显示 Notebook 标题进度。" not in index_html
    assert 'id="scanTaskButton"' not in index_html
    assert 'id="runNotebookButton"' not in index_html
    assert 'id="generateReportButton"' not in index_html
    assert "原始扫描结果" not in index_html
    assert 'id="scanResult"' not in index_html
    assert 'class="raw-details"' not in index_html
    assert ".raw-details" not in styles_css
    assert ".raw-json" not in styles_css
    assert ".supporting-evidence {\n  display: none;" not in styles_css
    assert "#reportSection[aria-hidden=\"true\"]" in styles_css
    assert "#reportSection {\n  display: none;" not in styles_css


def test_plan_rail_artifact_preview_is_wired_to_real_app_shell():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    plan_js = _read_static("js/v2/plan_rail_controller.js")
    styles_css = _read_static("styles.css")

    assert 'id="artifactPanel"' in index_html
    assert 'id="artifactPanelBody"' in index_html
    assert 'import { attachArtifactHandlers, renderArtifact } from "./artifact_view.js";' in plan_js
    assert 'function planOutputButtonHtml(step)' in plan_js
    assert 'data-artifact="${escapeHtml(outputRef)}"' in plan_js
    assert "attachArtifactHandlers(root, artifactPreviewContainer" in plan_js
    assert "artifactRenderer(target, artifactRef)" in plan_js
    assert "planRailController.installArtifactHandlers(document);" in app_js
    assert ".artifact-panel {" in styles_css
    assert ".artifact-panel-body" in styles_css


def test_report_editor_form_and_summary_are_removed_from_frontend():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    removed_fragments = [
        'id="reportSummary"',
        'id="wordReportTitle"',
        'id="reportFieldsForm"',
        "fields-form",
        "word-field-group",
        "data-report-key",
        "setReportEditorLocked",
        "renderWordDocument",
        "renderReportFields",
        "saveReportFields",
        "showReportFieldsLoading",
        "还没生成 Word 报告",
        "保存内容后在右侧步骤中点击",
        "Word 内容有未保存修改",
    ]
    combined = "\n".join([index_html, app_js, styles_css])
    for fragment in removed_fragments:
        assert fragment not in combined


def test_sidebar_empty_state_is_compact():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    empty_start = styles_css.index(".empty-state {\n  padding")
    empty_end = styles_css.index("}", empty_start)
    empty_rule = styles_css[empty_start:empty_end]
    assert "padding: 9px 12px" in empty_rule
    assert '<div class="empty-state">暂无任务</div>' in app_js
    assert '<div class="empty-state">暂无任务</div>' in index_html
    assert "暂无任务。先创建一个验证任务。" not in app_js

    task_empty_start = styles_css.index(".task-list > .empty-state {")
    task_empty_end = styles_css.index("}", task_empty_start)
    task_empty_rule = styles_css[task_empty_start:task_empty_end]
    assert "border: 0" in task_empty_rule
    assert "background: transparent" in task_empty_rule
    assert "text-align: center" in task_empty_rule


def test_shell_has_collapsible_compact_sidebar():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    assert root_vars["--collapsed-entry-size"] == "44px"
    assert root_vars["--collapsed-entry-gutter"] == "16px"
    assert root_vars["--collapsed-search-top"] == "79px"
    assert root_vars["--collapsed-popover-left"] == (
        "calc(var(--collapsed-entry-gutter) + var(--collapsed-entry-size) + 10px)"
    )

    assert 'id="sidebarCollapseButton"' in index_html
    assert 'id="sidebarBrandTrigger"' in index_html
    assert 'id="collapsedCreateTaskButton"' in index_html
    assert 'aria-label="收起侧栏"' in index_html
    assert "toggleSidebarCollapsed" in app_js
    assert "expandSidebarFromBrand" in app_js
    assert "handleSidebarBrandKeydown" in app_js
    assert "restoreSidebarCollapsed" in app_js
    assert "ensurePetWithinViewport({ persist: true });" in app_js
    assert "const shouldKeepPetOnLeftEdge = petIsPinnedToWorkspaceLeftEdge();" in app_js
    assert "pinPetToWorkspaceLeftEdge({ persist: true });" in app_js
    assert "window.setTimeout(() => {" in app_js
    assert '$("collapsedCreateTaskButton").onclick = openTaskTypeWelcome;' in app_js
    assert 'localStorage.setItem("sidebarCollapsed"' in app_js
    assert 'localStorage.getItem("marvis_layout")' in index_html
    assert 'style.setProperty("--sidebar-width"' in index_html
    assert 'style.setProperty("--progress-width"' in index_html
    assert 'localStorage.getItem("sidebarCollapsed") === "1"' in index_html
    assert 'classList.add("sidebar-collapsed")' in index_html
    assert index_html.index('id="appShell"') < index_html.index('localStorage.getItem("marvis_layout")')
    assert index_html.index('localStorage.getItem("marvis_layout")') < index_html.index('id="taskSidebar"')
    assert index_html.index('id="appShell"') < index_html.index('localStorage.getItem("sidebarCollapsed") === "1"')
    assert index_html.index('localStorage.getItem("sidebarCollapsed") === "1"') < index_html.index('id="taskSidebar"')
    assert ".app-shell.sidebar-collapsed" in styles_css
    assert "--sidebar-width: 314px" in styles_css
    assert "--sidebar-width: 0px" in styles_css
    assert ".app-shell.sidebar-collapsed .brand-logo h1" in styles_css
    assert "task-row-short" not in app_js
    assert ".app-shell.sidebar-collapsed .task-row-short" not in styles_css
    assert 'class="collapse-panel"' in index_html
    assert 'class="collapse-divider"' in index_html
    assert 'class="collapse-chevron"' in index_html
    collapse_start = styles_css.index(".sidebar-collapse-button {")
    collapse_end = styles_css.index("}", collapse_start)
    collapse_rule = styles_css[collapse_start:collapse_end]
    assert "top: 23px" in collapse_rule
    assert "width: var(--sidebar-control-size)" in collapse_rule
    assert "height: var(--sidebar-control-size)" in collapse_rule
    assert "border: 0" in collapse_rule
    assert "background: transparent" in collapse_rule
    assert "box-shadow:" not in collapse_rule
    collapsed_sidebar_rule = _css_rule(styles_css, ".app-shell.sidebar-collapsed .task-sidebar")
    assert "position: fixed" in collapsed_sidebar_rule
    assert "width: var(--collapsed-hit-area-width)" in collapsed_sidebar_rule
    assert "border-right: 0" in collapsed_sidebar_rule
    assert "background: transparent" in collapsed_sidebar_rule
    assert "pointer-events: none" in collapsed_sidebar_rule

    sidebar_start = styles_css.index(".task-sidebar {\n  display: flex")
    sidebar_end = styles_css.index("}", sidebar_start)
    sidebar_rule = styles_css[sidebar_start:sidebar_end]
    assert "grid-column: 1" in sidebar_rule
    workspace_rule = _css_rule(styles_css, ".validation-workspace")
    assert "grid-column: 2" in workspace_rule

    collapsed_head_start = styles_css.index(".app-shell.sidebar-collapsed .sidebar-head")
    collapsed_head_end = styles_css.index("}", collapsed_head_start)
    collapsed_head_rule = styles_css[collapsed_head_start:collapsed_head_end]
    assert "position: fixed" in collapsed_head_rule
    assert "top: 18px" in collapsed_head_rule
    assert "left: var(--collapsed-entry-gutter)" in collapsed_head_rule
    assert "width: var(--collapsed-entry-size)" in collapsed_head_rule
    assert "padding: 0" in collapsed_head_rule
    assert ".app-shell.sidebar-collapsed .brand-mark" in styles_css
    collapsed_logo_start = styles_css.index(".app-shell.sidebar-collapsed .brand-mark")
    collapsed_logo_end = styles_css.index("}", collapsed_logo_start)
    collapsed_logo_rule = styles_css[collapsed_logo_start:collapsed_logo_end]
    assert "width: var(--collapsed-entry-size)" in collapsed_logo_rule
    assert "height: var(--collapsed-entry-size)" in collapsed_logo_rule

    collapsed_brand_rule = _css_rule(styles_css, ".app-shell.sidebar-collapsed .brand-logo")
    assert "cursor: pointer" in collapsed_brand_rule
    assert "pointer-events: auto" in collapsed_brand_rule
    assert "border-radius: var(--radius)" in collapsed_brand_rule
    collapsed_brand_hover_rule = _css_rule(styles_css, ".app-shell.sidebar-collapsed .brand-logo:hover")
    assert "background: transparent" in collapsed_brand_hover_rule
    assert "box-shadow: none" in collapsed_brand_hover_rule
    assert "transform:" not in collapsed_brand_hover_rule

    collapsed_mark_hover_start = styles_css.index(
        ".app-shell.sidebar-collapsed .brand-logo:hover .brand-mark,"
    )
    collapsed_mark_hover_end = styles_css.index("}", collapsed_mark_hover_start)
    collapsed_mark_hover_rule = styles_css[collapsed_mark_hover_start:collapsed_mark_hover_end]
    assert ".app-shell.sidebar-collapsed .brand-logo:focus-visible .brand-mark" in collapsed_mark_hover_rule
    assert "opacity: 0" in collapsed_mark_hover_rule
    assert "transform: scale(0.88)" in collapsed_mark_hover_rule

    collapsed_button_rule = _css_rule(styles_css, ".app-shell.sidebar-collapsed .sidebar-collapse-button")
    assert "position: fixed" in collapsed_button_rule
    assert "top: 18px" in collapsed_button_rule
    assert "left: var(--collapsed-entry-gutter)" in collapsed_button_rule
    assert "width: var(--collapsed-entry-size)" in collapsed_button_rule
    assert "height: var(--collapsed-entry-size)" in collapsed_button_rule
    assert "color: var(--text)" in collapsed_button_rule
    assert "opacity: 0" in collapsed_button_rule
    assert "pointer-events: none" in collapsed_button_rule

    collapsed_button_reveal_start = styles_css.index(
        ".app-shell.sidebar-collapsed:has(.brand-logo:hover) .sidebar-collapse-button,"
    )
    collapsed_button_reveal_end = styles_css.index("}", collapsed_button_reveal_start)
    collapsed_button_reveal_rule = styles_css[collapsed_button_reveal_start:collapsed_button_reveal_end]
    assert ".app-shell.sidebar-collapsed:has(.brand-logo:focus-visible) .sidebar-collapse-button" in (
        collapsed_button_reveal_rule
    )
    assert "opacity: 1" in collapsed_button_reveal_rule
    assert "background: var(--option-hover)" in collapsed_button_reveal_rule
    assert "box-shadow: none" in collapsed_button_reveal_rule
    assert ".app-shell.sidebar-collapsed .brand-logo::before" not in styles_css

    collapsed_hidden_rule = _css_rule(
        styles_css,
        ".app-shell.sidebar-collapsed .list-toolbar,\n.app-shell.sidebar-collapsed .task-list",
    )
    assert "display: none" in collapsed_hidden_rule

    collapsed_list_head_rule = _css_rule(styles_css, ".app-shell.sidebar-collapsed .list-head")
    assert "top: var(--collapsed-search-top)" in collapsed_list_head_rule
    assert "flex-direction: column" in collapsed_list_head_rule
    assert "gap: 8px" in collapsed_list_head_rule

    collapsed_create_rule = _css_rule(styles_css, ".app-shell.sidebar-collapsed .collapsed-create-toggle")
    assert "display: inline-flex" in collapsed_create_rule


def test_sidebar_icon_controls_share_settings_sizing_and_interaction():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")

    assert "static/styles.css?v=__MARVIS_STATIC_VERSION__" in index_html
    assert "static/css/welcome.css?v=__MARVIS_STATIC_VERSION__" in index_html
    assert "static/styles.css?v=20260613-task-entry-upload" not in index_html
    assert 'static/styles.css?v=20260613-task-entry"' not in index_html
    assert "static/styles.css?v=20260605-create-dialog-button-gap" not in index_html
    assert "static/styles.css?v=20260605-create-dialog-scroll" not in index_html
    assert "static/styles.css?v=20260603-sidebar-icon-controls" not in index_html
    assert "static/styles.css?v=20260603-run-mode-selected-glow" not in index_html
    assert "static/styles.css?v=20260603-validator-icon-16" not in index_html
    assert "static/styles.css?v=20260603-settings-no-focus-frame" not in index_html
    assert "static/styles.css?v=20260603-brand-icon-neutral-fill" not in index_html
    assert "static/styles.css?v=20260603-task-validator-icon" not in index_html
    assert "static/styles.css?v=20260603-run-mode-border" not in index_html
    assert "static/styles.css?v=20260603-scan-env-add-style" not in index_html
    assert "static/styles.css?v=20260603-brand-icon-buttons" not in index_html
    assert "static/styles.css?v=20260603-run-mode-glow" not in index_html

    brand_token_rule = _css_rule(styles_css, ":root")
    assert "--sidebar-control-size: 34px" in brand_token_rule
    assert "--sidebar-control-icon-size: 17px" in brand_token_rule
    assert "--radius-control: 10px" in brand_token_rule
    assert "--brand-primary: #303034" in brand_token_rule
    assert "--brand-primary-hover: #3b3b42" in brand_token_rule
    assert "--brand-icon-color: color-mix(in srgb, var(--brand-primary)" in brand_token_rule
    assert "--brand-icon-hover-bg:" not in brand_token_rule
    assert "--brand-icon-ring: color-mix(in srgb, var(--brand-primary)" in brand_token_rule

    collapse_rule = _css_rule(styles_css, ".sidebar-collapse-button")
    assert "width: var(--sidebar-control-size)" in collapse_rule
    assert "height: var(--sidebar-control-size)" in collapse_rule
    assert "border-radius: var(--radius-control)" in collapse_rule
    assert "color: var(--text)" in collapse_rule

    search_toggle_start = styles_css.index("\n.list-search-toggle {")
    search_toggle_end = styles_css.index("}", search_toggle_start)
    search_toggle_rule = styles_css[search_toggle_start:search_toggle_end]
    assert "width: var(--sidebar-control-size)" in search_toggle_rule
    assert "height: var(--sidebar-control-size)" in search_toggle_rule
    assert "border-radius: var(--radius-control)" in search_toggle_rule
    assert "color: var(--text)" in search_toggle_rule

    collapsed_create_start = styles_css.index("\n.collapsed-create-toggle {")
    collapsed_create_end = styles_css.index("}", collapsed_create_start)
    collapsed_create_rule = styles_css[collapsed_create_start:collapsed_create_end]
    assert "display: none" in collapsed_create_rule
    assert "width: var(--collapsed-entry-size)" in collapsed_create_rule
    assert "height: var(--collapsed-entry-size)" in collapsed_create_rule
    assert "border-radius: var(--radius-control)" in collapsed_create_rule
    assert "color: var(--text)" in collapsed_create_rule

    settings_summary_start = styles_css.index("\n.sidebar-settings summary {")
    settings_summary_end = styles_css.index("}", settings_summary_start)
    settings_summary_rule = styles_css[settings_summary_start:settings_summary_end]
    assert "min-height: var(--sidebar-control-size)" in settings_summary_rule
    assert "border-radius: var(--radius-control)" in settings_summary_rule
    assert "color: var(--text)" in settings_summary_rule
    assert "font-size: 14px" in settings_summary_rule

    shared_icon_rule_start = styles_css.index(".nav-action svg,")
    shared_icon_rule_end = styles_css.index("}", shared_icon_rule_start)
    shared_icon_rule = styles_css[shared_icon_rule_start:shared_icon_rule_end]
    for selector in [
        ".sidebar-collapse-button svg",
        ".list-search-toggle svg",
        ".collapsed-create-toggle svg",
        ".sidebar-settings summary svg",
    ]:
        assert selector in shared_icon_rule
    assert "width: var(--sidebar-control-icon-size)" in shared_icon_rule
    assert "height: var(--sidebar-control-icon-size)" in shared_icon_rule
    assert "18px" not in shared_icon_rule

    collapse_hover_start = styles_css.index(
        ".sidebar-collapse-button:hover,\n.sidebar-collapse-button:focus-visible {"
    )
    collapse_hover_end = styles_css.index("}", collapse_hover_start)
    collapse_hover_rule = styles_css[collapse_hover_start:collapse_hover_end]
    assert "color: var(--text)" in collapse_hover_rule
    assert "background: var(--option-hover)" in collapse_hover_rule
    assert "var(--brand-icon-hover-bg)" not in collapse_hover_rule
    assert "var(--brand-icon-ring)" not in collapse_hover_rule
    assert "var(--accent" not in collapse_hover_rule

    search_input_rule = _css_rule(styles_css, "body.search-active .task-search input")
    assert "border-color: var(--button-outline-border)" in search_input_rule
    assert "box-shadow: 0 0 0 3px var(--brand-icon-ring)" in search_input_rule
    assert "var(--accent" not in search_input_rule

    search_close_hover_start = styles_css.index(
        ".task-search-close:hover,\n.task-search-close:focus-visible {"
    )
    search_close_hover_end = styles_css.index("}", search_close_hover_start)
    search_close_hover_rule = styles_css[search_close_hover_start:search_close_hover_end]
    assert "color: var(--brand-icon-color)" in search_close_hover_rule
    assert "background: var(--option-hover)" in search_close_hover_rule
    assert "var(--brand-icon-hover-bg)" not in search_close_hover_rule
    assert "var(--brand-icon-ring)" not in search_close_hover_rule
    assert "var(--accent" not in search_close_hover_rule

    search_toggle_hover_start = styles_css.index(
        ".list-search-toggle:hover,\n.list-search-toggle:focus-visible,"
    )
    search_toggle_hover_end = styles_css.index("}", search_toggle_hover_start)
    search_toggle_hover_rule = styles_css[search_toggle_hover_start:search_toggle_hover_end]
    assert "color: var(--text)" in search_toggle_hover_rule
    assert "background: var(--option-hover)" in search_toggle_hover_rule
    assert ".collapsed-create-toggle:hover" in search_toggle_hover_rule
    assert ".collapsed-create-toggle:focus-visible" in search_toggle_hover_rule
    assert "var(--brand-icon-hover-bg)" not in search_toggle_hover_rule
    assert "var(--brand-icon-ring)" not in search_toggle_hover_rule
    assert "var(--accent" not in search_toggle_hover_rule

    search_toggle_active_rule = _css_rule(styles_css, "body.search-active .list-search-toggle")
    assert "color: var(--text)" in search_toggle_active_rule
    assert "background: var(--option-hover)" in search_toggle_active_rule
    assert "var(--brand-icon-hover-bg)" not in search_toggle_active_rule
    assert "var(--brand-icon-ring)" not in search_toggle_active_rule
    assert "var(--accent" not in search_toggle_active_rule

    collapsed_search_toggle_rule = _css_rule(styles_css, ".app-shell.sidebar-collapsed .list-search-toggle")
    assert "width: var(--collapsed-entry-size)" in collapsed_search_toggle_rule
    assert "height: var(--collapsed-entry-size)" in collapsed_search_toggle_rule
    assert "margin: 0 auto" in collapsed_search_toggle_rule


def test_collapsed_sidebar_search_flyout_stays_above_scrim_and_aligns_with_search_button():
    styles_css = _read_static("styles.css")

    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    assert root_vars["--collapsed-search-top"] == "79px"

    scrim_rule = _css_rule(styles_css, ".search-scrim")
    assert "z-index: 50" in scrim_rule
    assert "backdrop-filter: var(--scrim-blur)" in scrim_rule

    active_sidebar_selector = "body.search-active .app-shell.sidebar-collapsed .task-sidebar"
    assert f"{active_sidebar_selector} {{" in styles_css
    active_sidebar_rule = _css_rule(styles_css, active_sidebar_selector)
    assert "position: static" in active_sidebar_rule
    assert "z-index: auto" in active_sidebar_rule

    active_brand_selector = "body.search-active .app-shell.sidebar-collapsed .sidebar-head"
    active_brand_rule = _css_rule(styles_css, active_brand_selector)
    assert "opacity: 0" not in active_brand_rule
    assert "pointer-events: none" in active_brand_rule

    flyout_rule = _css_rule(styles_css, "body.search-active .app-shell.sidebar-collapsed .task-list-wrap")
    assert "top: calc(var(--collapsed-search-top) - 10px)" in flyout_rule
    assert "top: 58px" not in flyout_rule
    assert "z-index: 60" in flyout_rule
    assert "pointer-events: auto" in flyout_rule
    assert "padding: 10px" in flyout_rule
    assert "background: var(--sidebar-bg)" in flyout_rule
    assert "border: 1px solid var(--sidebar-border)" in flyout_rule
    assert "background: var(--surface)" not in flyout_rule
    assert "border: 1px solid transparent" not in flyout_rule
    assert "body.search-active .task-row {\n  box-shadow:" not in styles_css

    active_collapsed_create_rule = _css_rule(
        styles_css, "body.search-active .app-shell.sidebar-collapsed .collapsed-create-toggle"
    )
    assert "display: none" in active_collapsed_create_rule

    collapsed_search_rule = _css_rule(styles_css, "body.search-active .app-shell.sidebar-collapsed .task-search")
    assert "width: 100%" in collapsed_search_rule
    assert "opacity: 1" in collapsed_search_rule
    assert "transition: none" in collapsed_search_rule

    collapsed_head_rule = _css_rule(styles_css, "body.search-active .app-shell.sidebar-collapsed .list-head")
    assert "padding: 0 0 10px" in collapsed_head_rule

    collapsed_task_list_rule = _css_rule(styles_css, "body.search-active .app-shell.sidebar-collapsed .task-list")
    assert "display: grid" in collapsed_task_list_rule
    assert "padding: 0" in collapsed_task_list_rule


def test_sidebar_brand_title_stays_on_one_line():
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")

    sidebar_head_start = styles_css.index(".sidebar-head {")
    sidebar_head_end = styles_css.index("}", sidebar_head_start)
    sidebar_head_rule = styles_css[sidebar_head_start:sidebar_head_end]
    assert "padding: 18px 10px 14px 16px" in sidebar_head_rule

    logo_start = styles_css.index(".brand-logo {")
    logo_end = styles_css.index("}", logo_start)
    logo_rule = styles_css[logo_start:logo_end]
    assert "display: grid" in logo_rule
    assert "grid-template-columns:" in logo_rule
    assert "minmax(0, 1fr)" in logo_rule
    assert "inline-size: 100%" in logo_rule
    assert "container-type: inline-size" in logo_rule

    brand_start = styles_css.index(".brand-logo h1 {")
    brand_end = styles_css.index("}", brand_start)
    brand_rule = styles_css[brand_start:brand_end]
    assert "inline-size: 100%" in brand_rule
    assert "font-size: clamp(16px, 7cqi, 22px)" in brand_rule
    assert "cqi" in brand_rule
    assert "vw" not in brand_rule
    assert "white-space: nowrap" in brand_rule
    assert "overflow-wrap: normal" in brand_rule
    assert "word-break: keep-all" in brand_rule
    assert "text-overflow" not in brand_rule
    assert "letter-spacing: 0" in brand_rule

    layout_resize_js = _read_static("js/layout-resize.js")
    assert "export const SIDEBAR_WIDTH_MIN = 314;" in layout_resize_js
    assert "export const SIDEBAR_WIDTH_MAX = 520;" in layout_resize_js
    assert "stored.sidebar === 320 ? SIDEBAR_WIDTH_MIN : stored.sidebar" in layout_resize_js
    assert "clamp(startSidebar + deltaX, SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX)" in layout_resize_js
    assert "clamp(current + direction * step, SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX)" in layout_resize_js
    assert 'from "./js/layout-resize.js"' in app_js
    assert "grid-template-columns: min(var(--sidebar-width), 314px) minmax(0, 1fr)" in styles_css


def test_sidebar_footer_and_create_action_match_brand_treatment():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")

    assert "<span>新建任务</span>" in index_html
    assert "<span>创建任务</span>" not in index_html
    assert 'aria-label="新建任务"' in index_html
    assert 'title="新建任务"' in index_html

    footer_start = styles_css.index(".sidebar-footer {")
    footer_end = styles_css.index("}", footer_start)
    footer_rule = styles_css[footer_start:footer_end]
    assert "border-top" not in footer_rule

    toolbar_start = styles_css.index(".list-toolbar {")
    toolbar_end = styles_css.index("}", toolbar_start)
    toolbar_rule = styles_css[toolbar_start:toolbar_end]
    assert "flex-shrink: 0" in toolbar_rule
    assert "padding: 2px 10px 16px" in toolbar_rule

    create_start = styles_css.index(".nav-action {")
    create_end = styles_css.index("}", create_start)
    create_rule = styles_css[create_start:create_end]
    assert "color: var(--button-primary-text)" in create_rule
    assert "background: var(--button-primary-bg)" in create_rule
    assert "border: 0" in create_rule
    assert "border-radius: var(--radius-control)" in create_rule
    assert "box-shadow: var(--button-solid-shadow)" in create_rule
    assert "0 7px 10px" not in create_rule
    assert "0 5px 10px" not in create_rule
    assert "0 10px 14px" not in create_rule
    assert "linear-gradient" not in create_rule
    assert "inset" not in create_rule
    assert "justify-content: center" in create_rule
    assert "transform 140ms ease" not in create_rule

    create_hover_start = styles_css.index(".nav-action:hover,")
    create_hover_end = styles_css.index("}", create_hover_start)
    create_hover_rule = styles_css[create_hover_start:create_hover_end]
    assert "color: var(--button-primary-text-hover)" in create_hover_rule
    assert "background: var(--button-primary-bg-hover)" in create_hover_rule
    assert "outline: none" in create_hover_rule
    assert "box-shadow: var(--button-solid-shadow-hover)" in create_hover_rule
    assert "0 8px 12px" not in create_hover_rule
    assert "0 7px 14px" not in create_hover_rule
    assert "0 12px 18px" not in create_hover_rule
    assert "transform:" not in create_hover_rule
    assert "linear-gradient" not in create_hover_rule
    assert "inset" not in create_hover_rule
    assert "border-color" not in create_hover_rule
    assert 'body[data-theme="dark"] .nav-action {' not in styles_css
    assert 'body[data-theme="dark"] .nav-action:hover,' not in styles_css


def test_empty_workspace_copy_is_shorter_and_direct():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    workspace_view_js = _read_static("js/task-workspace-view.js")

    assert "验证任务" in index_html
    assert "选择或创建验证任务" not in index_html
    assert "选择或创建验证任务" not in app_js
    assert "创建任务或从左侧选择已有任务" in index_html
    assert "先创建任务或从左侧选择已有任务" not in index_html
    assert "核心任务信息" in index_html
    assert "选择任务后显示核心任务信息" not in index_html
    assert "核心任务信息" in workspace_view_js
    assert "选择任务后显示核心任务信息" not in app_js
    assert "选择任务后显示核心任务信息" not in workspace_view_js
    assert '选择任务后点击\\"扫描材料\\"开始' in app_js
    assert '还没扫描材料。选择任务后点击\\"扫描材料\\"开始。' not in app_js


def test_selected_task_header_omits_local_validation_subtitle():
    app_js = _read_static("app.js")
    workspace_view_js = _read_static("js/task-workspace-view.js")
    styles_css = _read_static("styles.css")

    current_start = app_js.index("function renderCurrentTask")
    current_end = app_js.index("function workflowStepStatus", current_start)
    current_renderer = app_js[current_start:current_end]

    assert "本地验证任务" not in current_renderer
    assert "renderCurrentTaskWorkspace({" in current_renderer
    assert 'if (subtitle) subtitle.textContent = "";' in workspace_view_js
    assert ".workspace-subtitle:empty" in styles_css
    assert "display: none" in styles_css[
        styles_css.index(".workspace-subtitle:empty") : styles_css.index("}", styles_css.index(".workspace-subtitle:empty"))
    ]


def test_task_selection_keeps_same_task_active_and_refresh_restores_remembered_task():
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")

    assert 'export const selectedTaskStorageKey = "marvis_selected_task_id";' in state_js
    assert "function rememberSelectedTaskId" in app_js
    assert "function storedSelectedTaskId" in app_js

    sync_start = app_js.index("function syncSelectedTaskFromCache")
    sync_end = app_js.index("function runModeLabel", sync_start)
    sync_renderer = app_js[sync_start:sync_end]
    assert "taskCache[0]" not in sync_renderer
    assert "const storedTaskId = storedSelectedTaskId();" in sync_renderer
    assert "taskCache.find((task) => task.id === storedTaskId)" in sync_renderer
    assert "rememberSelectedTaskId(null);" in sync_renderer

    select_start = app_js.index("function selectTask")
    select_end = app_js.index("function renderMetricPreview", select_start)
    select_renderer = app_js[select_start:select_end]
    assert "if (selectedTaskId === task.id && selectedTask)" in select_renderer
    same_task_start = select_renderer.index("if (selectedTaskId === task.id && selectedTask)")
    same_task_end = select_renderer.index("resetAgentTypingState", same_task_start)
    same_task_branch = select_renderer[same_task_start:same_task_end]
    assert "selectedTask = task;" in same_task_branch
    assert "rememberSelectedTaskId(task.id);" in same_task_branch
    assert "deselectCurrentTask()" not in same_task_branch
    assert "rememberSelectedTaskId(task.id);" in select_renderer
    assert "rememberSelectedTaskId(null);" in select_renderer
    assert "renderMetricPreview({});" in select_renderer
    assert "function deselectCurrentTask" in app_js

    create_start = app_js.index("async function createTask")
    create_end = app_js.index("async function refreshTasks", create_start)
    create_renderer = app_js[create_start:create_end]
    assert "rememberSelectedTaskId(task.id);" in create_renderer

    delete_start = app_js.index("async function deleteTask")
    delete_end = app_js.index("async function runAction", delete_start)
    delete_renderer = app_js[delete_start:delete_end]
    assert "rememberSelectedTaskId(null);" in delete_renderer


def test_refresh_restores_selected_task_before_async_detail_loads():
    app_js = _read_static("app.js")
    index_html = _read_static("index.html")
    state_js = _read_static("js/state.js")
    styles_css = _read_static("styles.css")

    assert 'export const resultScrollPositionsStorageKey = "marvis_result_scroll_positions";' in state_js
    assert "function loadResultScrollPositions" in app_js
    assert "function restoreResultScrollPositionAfterRender" in app_js

    restore_body = _slice_function(app_js, "function restoreSelectedTaskPlaceholder")
    assert "const storedTaskId = storedSelectedTaskId();" in restore_body
    assert "selectedTaskId = storedTaskId;" in restore_body
    assert "selectedTask = null;" in restore_body

    current_body = _slice_function(app_js, "function renderCurrentTask")
    workspace_view_js = _read_static("js/task-workspace-view.js")
    assert "renderCurrentTaskWorkspace({" in current_body
    assert "const hasTaskContext = Boolean(selectedTask || selectedTaskId);" in workspace_view_js
    assert 'classList.toggle("is-empty", !hasTaskContext)' in workspace_view_js
    assert "正在恢复任务" in workspace_view_js

    load_scroll_call = app_js.index("loadResultScrollPositions();")
    restore_call = app_js.index("restoreSelectedTaskPlaceholder();")
    initialize_call = app_js.index("initializeApp();")
    assert load_scroll_call < restore_call
    assert restore_call < initialize_call

    init_body = _slice_function(app_js, "async function initializeApp")
    assert "await refreshTasks();" in init_body
    assert 'class="app-booting"' in index_html
    assert "function finishAppBoot" in app_js
    assert "finishAppBoot();" in init_body
    assert "document.body.classList.remove(\"app-booting\")" in app_js
    assert "body.app-booting .validation-workspace:not(.is-empty) :is(.workspace-head, .workspace-body, .progress-rail)" in styles_css
    assert "body.app-booting :is(.workspace-head, .workspace-body, .progress-rail)" in styles_css
    assert "function enableAppAnimationsAfterBoot" in app_js
    assert "document.body.classList.add(\"anim-ready\")" in _slice_function(app_js, "function enableAppAnimationsAfterBoot")
    assert "enableAppAnimationsAfterBoot();" in _slice_function(app_js, "function finishAppBoot")
    assert "requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.add(\"anim-ready\")));" not in app_js

    app_shell_rule = _css_rule(styles_css, ".app-shell")
    assert "transition:" not in app_shell_rule

    first_render = init_body.index("renderAll();")
    message_load = init_body.index("await loadAgentMessages();")
    restore_scroll = init_body.index("await restoreResultScrollPositionAfterRender(selectedTaskId);")
    finish_boot = init_body.index("finishAppBoot();")
    assert message_load < first_render
    assert first_render < restore_scroll < finish_boot
    assert "if (selectedTaskId) renderAll();" not in init_body


def test_workspace_cards_float_on_one_background_with_top_step_rail():
    styles_css = _read_static("styles.css")

    assert ".validation-workspace {" in styles_css
    assert "--workspace-main-gutter: 106px" in styles_css
    assert "--workspace-collapsed-main-gutter: 180px" in styles_css
    assert "--workspace-rail-gap: 70px" in styles_css
    assert "--workspace-collapsed-rail-gap: 78px" in styles_css
    assert "--pet-default-workspace-offset: -2px" in styles_css
    assert "--pet-min-workspace-offset: -2px" in styles_css
    assert "--pet-collapsed-workspace-offset: 72px" in styles_css
    assert "--progress-width: 314px" in styles_css
    assert "grid-template-columns: minmax(0, 1fr) var(--workspace-rail-gap) var(--progress-width)" in styles_css
    assert "grid-template-columns: minmax(340px, 1fr) var(--workspace-rail-gap) min(var(--progress-width), 340px)" in styles_css
    workspace_start = styles_css.index(".validation-workspace {")
    workspace_end = styles_css.index("}", workspace_start)
    workspace_rule = styles_css[workspace_start:workspace_end]
    assert "background: var(--surface)" in workspace_rule
    collapsed_shell_rule = _css_rule(styles_css, ".app-shell.sidebar-collapsed")
    assert "--sidebar-width: 0px" in collapsed_shell_rule
    assert "--workspace-main-gutter: var(--workspace-collapsed-main-gutter)" in collapsed_shell_rule
    assert "--workspace-rail-gap: var(--workspace-collapsed-rail-gap)" in collapsed_shell_rule
    assert "--pet-default-workspace-offset: var(--pet-collapsed-workspace-offset)" in collapsed_shell_rule
    assert "--pet-min-workspace-offset: var(--pet-collapsed-workspace-offset)" in collapsed_shell_rule
    assert ".workspace-body {" in styles_css
    assert "display: contents" in styles_css
    assert ".progress-rail {" in styles_css
    rail_start = styles_css.index(".progress-rail {")
    rail_end = styles_css.index("}", rail_start)
    rail_rule = styles_css[rail_start:rail_end]
    assert "align-self: start" in rail_rule
    assert "height: auto" in rail_rule
    assert "gap: 0" in rail_rule
    assert "max-height: calc(100dvh - 28px)" in rail_rule
    assert "min-width: 300px" in rail_rule
    assert "margin: 14px 14px 14px 0" in rail_rule
    assert ".progress-panel {" in styles_css
    assert "--shadow-floating: 0 2px 8px rgba(0, 0, 0, 0.035)" in styles_css
    assert ".result-scroll-content > .progress-panel" in styles_css
    assert "flex: 0 0 auto" in styles_css


def test_right_resize_handle_sits_on_step_rail_left_edge():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")

    assert 'id="rightResizeHandle"' in index_html
    assert 'class="resize-handle resize-handle-right"' in index_html

    handle_start = styles_css.index(".resize-handle-right {")
    handle_end = styles_css.index("}", handle_start)
    handle_rule = styles_css[handle_start:handle_end]
    assert "grid-column: 3;" in handle_rule
    assert "justify-self: start;" in handle_rule
    assert "margin-left: -6px;" in handle_rule
    assert "z-index: 2;" in handle_rule
    assert "grid-column: 2;" not in handle_rule

    handle_base_start = styles_css.index(".resize-handle {")
    handle_base_end = styles_css.index("}", handle_base_start)
    handle_base_rule = styles_css[handle_base_start:handle_base_end]
    assert "width: 12px;" in handle_base_rule
    assert "min-width: 12px;" in handle_base_rule

    handle_line_start = styles_css.index(".resize-handle::after {")
    handle_line_end = styles_css.index("}", handle_line_start)
    handle_line_rule = styles_css[handle_line_start:handle_line_end]
    assert "left: 5px;" in handle_line_rule


def test_middle_result_sections_are_unframed():
    styles_css = _read_static("styles.css")

    start = styles_css.index(".result-scroll-content > .progress-panel {")
    end = styles_css.index("}", start)
    rule = styles_css[start:end]
    assert "border: 0" in rule
    assert "border-radius: 0" in rule
    assert "background: transparent" in rule
    assert "box-shadow: none" in rule
    assert "padding: 0" in rule

    assert ".result-scroll-content > .progress-panel > .result-summary" in styles_css
    summary_start = styles_css.index(".result-scroll-content > .progress-panel > .result-summary")
    summary_end = styles_css.index("}", summary_start)
    summary_rule = styles_css[summary_start:summary_end]
    assert "border: 0" in summary_rule
    assert "border-radius: 0" in summary_rule
    assert "background: transparent" in summary_rule
    assert "box-shadow: none" in summary_rule
    assert "padding: 0" in summary_rule


def test_right_step_rail_uses_subtle_shadow():
    styles_css = _read_static("styles.css")

    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))
    assert root_vars["--progress-rail-shadow"] == "0 3px 12px rgba(0, 0, 0, 0.045)"
    assert dark_vars["--progress-rail-shadow"] == "0 4px 16px rgba(0, 0, 0, 0.22)"
    rail_rule = _css_rule(styles_css, ".progress-rail")
    assert "box-shadow: var(--progress-rail-shadow)" in rail_rule

    dark_rule = _css_rule(styles_css, 'body[data-theme="dark"] .progress-rail')
    assert "box-shadow:" not in dark_rule


def test_dark_theme_tokens_keep_panels_and_muted_text_readable():
    styles_css = _read_static("styles.css")
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))

    assert dark_vars["--bg"] == "#181818"
    assert dark_vars["--surface"] == "#2d2d2d"
    assert dark_vars["--surface-soft"] == "#242424"
    assert dark_vars["--sidebar-bg"] == "#1f1f1f"
    assert dark_vars["--sidebar-hover"] == "#313131"
    assert dark_vars["--border"] == "#363636"
    assert dark_vars["--border-strong"] == "#474747"
    assert _contrast_ratio(dark_vars["--surface"], dark_vars["--bg"]) >= 1.25
    assert _contrast_ratio(dark_vars["--surface"], dark_vars["--surface-soft"]) >= 1.12
    assert _contrast_ratio(dark_vars["--sidebar-bg"], dark_vars["--bg"]) >= 1.07
    assert _contrast_ratio(dark_vars["--text-muted"], dark_vars["--surface"]) >= 6.5
    assert _contrast_ratio(dark_vars["--text-secondary"], dark_vars["--surface"]) >= 8.0


def test_dark_theme_visible_scrollbars_match_dark_surfaces():
    styles_css = _read_static("styles.css")
    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))

    assert root_vars["--scrollbar-track"] == root_vars["--sidebar-bg"]
    assert root_vars["--scrollbar-thumb"] == "#b8b8bf"
    assert root_vars["--scrollbar-thumb-hover"] == "#9f9fa7"
    assert dark_vars["--scrollbar-track"] == dark_vars["--sidebar-bg"]
    assert dark_vars["--scrollbar-thumb"] == "#5f5f66"
    assert dark_vars["--scrollbar-thumb-hover"] == "#74747c"
    assert _contrast_ratio(dark_vars["--scrollbar-thumb"], dark_vars["--scrollbar-track"]) >= 2.0
    assert _contrast_ratio(dark_vars["--scrollbar-thumb-hover"], dark_vars["--scrollbar-track"]) >= 2.5

    visible_scrollbar_start = styles_css.index(".task-list,\n.task-list-wrap,")
    visible_scrollbar_end = styles_css.index("}", visible_scrollbar_start)
    visible_scrollbar_rule = styles_css[visible_scrollbar_start:visible_scrollbar_end]
    for selector in (
        ".task-list",
        ".task-list-wrap",
        ".task-form",
        ".metric-table-scroll",
    ):
        assert selector in visible_scrollbar_rule
    assert "scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track)" in visible_scrollbar_rule
    assert "scrollbar-width: thin" in visible_scrollbar_rule

    track_start = styles_css.index(".task-list::-webkit-scrollbar-track,")
    track_end = styles_css.index("}", track_start)
    track_rule = styles_css[track_start:track_end]
    assert "background: var(--scrollbar-track)" in track_rule

    thumb_start = styles_css.index(".task-list::-webkit-scrollbar-thumb,")
    thumb_end = styles_css.index("}", thumb_start)
    thumb_rule = styles_css[thumb_start:thumb_end]
    assert "background: var(--scrollbar-thumb)" in thumb_rule
    assert "border: 2px solid var(--scrollbar-track)" in thumb_rule

    thumb_hover_start = styles_css.index(".task-list::-webkit-scrollbar-thumb:hover,")
    thumb_hover_end = styles_css.index("}", thumb_hover_start)
    thumb_hover_rule = styles_css[thumb_hover_start:thumb_hover_end]
    assert "background: var(--scrollbar-thumb-hover)" in thumb_hover_rule

    hidden_result_rule = _css_rule(styles_css, ".result-scroll-content::-webkit-scrollbar")
    assert "display: none" in hidden_result_rule


def test_sidebar_task_and_settings_interactions_use_neutral_gray_states():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")
    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))

    assert root_vars["--option-hover"] == "#dedee3"
    assert root_vars["--option-selected"] == "#d4d4da"
    assert root_vars["--option-selected-border"] == root_vars["--option-selected"]
    assert dark_vars["--option-hover"] == "#46464a"
    assert dark_vars["--option-selected"] == "#525258"
    assert dark_vars["--option-selected-border"] == dark_vars["--option-selected"]

    assert _contrast_ratio(root_vars["--option-hover"], root_vars["--surface"]) >= 1.28
    assert _contrast_ratio(root_vars["--option-selected"], root_vars["--surface"]) >= 1.42
    assert _contrast_ratio(dark_vars["--option-hover"], dark_vars["--surface"]) >= 1.45
    assert _contrast_ratio(dark_vars["--option-selected"], dark_vars["--surface"]) >= 1.75

    for tokens in (root_vars, dark_vars):
        assert _contrast_ratio(tokens["--text"], tokens["--option-selected"]) >= 4.5
        assert _contrast_ratio(tokens["--text-secondary"], tokens["--option-selected"]) >= 4.5

    settings_summary_start = styles_css.index(".sidebar-settings summary:hover,")
    settings_summary_end = styles_css.index("}", settings_summary_start)
    settings_summary_rule = styles_css[settings_summary_start:settings_summary_end]
    assert "color: var(--text)" in settings_summary_rule
    assert "background: var(--option-hover)" in settings_summary_rule
    assert "var(--accent" not in settings_summary_rule

    settings_hover_start = styles_css.index(".settings-system-row:hover,")
    settings_hover_end = styles_css.index("}", settings_hover_start)
    settings_hover_rule = styles_css[settings_hover_start:settings_hover_end]
    assert ".settings-select:focus-visible" in settings_hover_rule
    assert "border-color: var(--option-hover)" in settings_hover_rule
    assert "background: var(--option-hover)" in settings_hover_rule
    assert "outline:" not in settings_hover_rule
    assert "#b9d7fb" not in settings_hover_rule
    assert "#f1f7ff" not in settings_hover_rule
    assert (
        "\n\n.settings-system-row:focus-visible,\n.settings-select:focus-visible {\n  outline:"
        not in styles_css
    )

    model_card_hover_start = styles_css.index(".llm-engine-item:hover,")
    model_card_hover_end = styles_css.index("}", model_card_hover_start)
    model_card_hover_rule = styles_css[model_card_hover_start:model_card_hover_end]
    model_card_rule = _css_rule(styles_css, ".llm-engine-item")
    assert "border-radius: var(--radius-control)" in model_card_rule
    engine_delete_rule = _css_rule(styles_css, ".engine-del-btn")
    assert "border-radius: var(--radius-control)" in engine_delete_rule
    engine_add_rule = _css_rule(styles_css, ".llm-engine-add")
    assert "border-radius: var(--radius-control)" in engine_add_rule
    assert ".llm-engine-item:focus-visible" in model_card_hover_rule
    assert "border-color: var(--option-hover)" in model_card_hover_rule
    assert "background: var(--option-hover)" in model_card_hover_rule
    assert "var(--accent" not in model_card_hover_rule
    model_card_focus_start = styles_css.index("\n.llm-engine-item:focus-visible {", model_card_hover_end)
    model_card_focus_end = styles_css.index("}", model_card_focus_start)
    model_card_focus_rule = styles_css[model_card_focus_start:model_card_focus_end]
    assert "outline: 3px solid var(--option-focus-ring)" in model_card_focus_rule
    assert "outline: none" not in model_card_focus_rule

    task_hover_start = styles_css.index(".task-row:hover,")
    task_hover_end = styles_css.index("}", task_hover_start)
    task_hover_rule = styles_css[task_hover_start:task_hover_end]
    assert ".task-row-shell:hover .task-row" in task_hover_rule
    assert "border-color" not in task_hover_rule
    assert "background: var(--option-hover)" in task_hover_rule
    assert "#9fbfe4" not in task_hover_rule
    assert "#fbfdff" not in task_hover_rule

    task_selected_rule = _css_rule(styles_css, ".task-row.selected")
    assert "border-color" not in task_selected_rule
    assert "background: var(--option-selected)" in task_selected_rule
    assert "box-shadow: none" in task_selected_rule
    assert "box-shadow: 0 0 0 2px var(--option-focus-ring)" not in task_selected_rule
    assert "var(--accent" not in task_selected_rule
    assert "#f8fbff" not in task_selected_rule

    assert ".app-shell.sidebar-collapsed .task-row-short" not in styles_css

    assert "#182a3f" not in styles_css
    assert "#345b8a" not in styles_css
    assert "#172032" not in styles_css
    assert "static/styles.css?v=__MARVIS_STATIC_VERSION__" in index_html
    assert "static/css/welcome.css?v=__MARVIS_STATIC_VERSION__" in index_html
    assert "static/styles.css?v=20260613-task-entry-upload" not in index_html
    assert 'static/styles.css?v=20260613-task-entry"' not in index_html
    assert "static/styles.css?v=20260605-create-dialog-button-gap" not in index_html
    assert "static/styles.css?v=20260605-create-dialog-scroll" not in index_html
    assert "static/styles.css?v=20260603-sidebar-icon-controls" not in index_html
    assert "static/styles.css?v=20260603-run-mode-selected-glow" not in index_html
    assert "static/styles.css?v=20260603-validator-icon-16" not in index_html
    assert "static/styles.css?v=20260603-settings-no-focus-frame" not in index_html
    assert "static/styles.css?v=20260603-brand-icon-neutral-fill" not in index_html
    assert "static/styles.css?v=20260603-task-validator-icon" not in index_html
    assert "static/styles.css?v=20260603-run-mode-border" not in index_html
    assert "static/styles.css?v=20260603-scan-env-add-style" not in index_html
    assert "static/styles.css?v=20260603-brand-icon-buttons" not in index_html
    assert "static/styles.css?v=20260603-run-mode-glow" not in index_html
    assert "static/styles.css?v=20260603-dark-scrollbar" not in index_html
    assert "static/styles.css?v=20260603-task-options" not in index_html
    assert "static/styles.css?v=20260603-neutral-options" not in index_html
    assert "static/styles.css?v=20260603-brand-buttons" not in index_html


def test_dark_theme_shell_columns_follow_reference_tones():
    styles_css = _read_static("styles.css")

    workspace_rule = _css_rule(styles_css, 'body[data-theme="dark"] .validation-workspace')
    assert "background: var(--bg)" in workspace_rule

    rail_rule = _css_rule(styles_css, 'body[data-theme="dark"] .progress-rail')
    assert "background: var(--surface)" in rail_rule
    assert "border-color: var(--border)" in rail_rule


def test_dark_workspace_masks_match_center_background():
    styles_css = _read_static("styles.css")

    base_workspace_rule = _css_rule(styles_css, ".validation-workspace")
    assert "--workspace-mask-bg: var(--surface)" in base_workspace_rule

    workspace_rule = _css_rule(styles_css, 'body[data-theme="dark"] .validation-workspace')
    assert "--workspace-mask-bg: var(--bg)" in workspace_rule

    head_rule = _css_rule(styles_css, ".workspace-head")
    assert "transparent calc(var(--radius) - 1px)" in head_rule
    assert "var(--workspace-mask-bg, var(--surface)) calc(var(--radius) - 0.5px)" in head_rule
    assert "background-position: left -1px top -1px, right -1px top -1px" in head_rule
    assert "background-size: calc(var(--radius) + 2px) calc(var(--radius) + 2px)" in head_rule
    assert "transparent 16px, var(--surface) 16.5px" not in head_rule

    composer_mask_rule = _css_rule(styles_css, ".agent-composer::before")
    composer_rule = _css_rule(styles_css, ".agent-composer")
    assert "--agent-composer-mask-bg: var(--workspace-mask-bg, var(--surface))" in composer_rule
    assert "--agent-composer-mask-bg: #ffffff" not in composer_rule
    assert "--agent-composer-mask-bg: var(--bg)" not in composer_rule
    dark_composer_rule = _css_rule(styles_css, 'body[data-theme="dark"] .agent-composer')
    assert "--agent-composer-mask-bg: var(--workspace-mask-bg, var(--surface))" in dark_composer_rule
    assert "left: -1px" in composer_mask_rule
    assert "right: -1px" in composer_mask_rule
    assert "bottom: -1px" in composer_mask_rule
    assert "height: calc(var(--radius) + 2px)" in composer_mask_rule
    assert "background: var(--agent-composer-mask-bg, var(--workspace-mask-bg, var(--surface)))" in composer_mask_rule
    assert "radial-gradient" not in composer_mask_rule
    assert "var(--workspace-mask-bg, var(--surface)) calc(var(--radius) + 0.5px)" not in composer_mask_rule
    assert "transparent 24px, var(--surface) 24.5px" not in composer_mask_rule


def test_primary_step_action_hover_keeps_button_text_readable():
    styles_css = _read_static("styles.css")
    root_rule = _css_rule(styles_css, ":root")

    assert "--button-solid-shadow:" in root_rule
    assert "0 1px 1px rgba(0, 0, 0, 0.10)" in root_rule
    assert "0 3px 6px rgba(0, 0, 0, 0.10)" in root_rule
    assert "0 6px 10px rgba(0, 0, 0, 0.07)" in root_rule
    assert "--button-solid-shadow-hover:" in root_rule
    assert "0 1px 1px rgba(0, 0, 0, 0.12)" in root_rule
    assert "0 4px 8px rgba(0, 0, 0, 0.12)" in root_rule
    assert "0 7px 12px rgba(0, 0, 0, 0.08)" in root_rule
    assert "--button-secondary-shadow:" in root_rule
    assert "0 1px 1px rgba(0, 0, 0, 0.06)" in root_rule
    assert "0 2px 4px rgba(0, 0, 0, 0.04)" in root_rule
    assert "0 5px 8px rgba(0, 0, 0, 0.035)" in root_rule
    assert "--button-secondary-shadow-hover:" in root_rule
    assert "0 3px 6px rgba(0, 0, 0, 0.06)" in root_rule
    assert "0 6px 10px rgba(0, 0, 0, 0.045)" in root_rule
    assert "--button-primary-bg: var(--brand-primary)" in root_rule
    assert "--button-primary-text: #ffffff" in root_rule
    assert "--button-outline-border: var(--brand-primary)" in root_rule
    assert "--button-outline-bg-hover: color-mix(in srgb, var(--brand-primary) 7%, transparent)" in root_rule

    dark_theme_rule = _css_rule(styles_css, 'body[data-theme="dark"]')
    assert "--button-primary-bg: #525258" in dark_theme_rule
    assert "--button-primary-bg-hover: #5d5d64" in dark_theme_rule
    assert "--button-primary-bg-active: #47474d" in dark_theme_rule
    assert "--button-primary-text: #f2f2f2" in dark_theme_rule
    assert "--button-outline-border: #96999f" in dark_theme_rule
    assert "--button-outline-bg-hover: color-mix(in srgb, #96999f 12%, transparent)" in dark_theme_rule
    assert "width:" not in dark_theme_rule
    assert "height:" not in dark_theme_rule
    assert "padding:" not in dark_theme_rule
    assert "transform:" not in dark_theme_rule

    primary_rule = _css_rule(styles_css, ".button.primary")
    assert "color: var(--button-primary-text)" in primary_rule
    assert "background: var(--button-primary-bg)" in primary_rule
    assert "border-color: var(--button-primary-border)" in primary_rule
    assert "box-shadow: var(--button-solid-shadow)" in primary_rule

    assert ".button.primary:hover:not(:disabled)" in styles_css
    hover_start = styles_css.index(".button.primary:hover:not(:disabled)")
    hover_end = styles_css.index("}", hover_start)
    hover_rule = styles_css[hover_start:hover_end]
    assert "color: var(--button-primary-text-hover)" in hover_rule
    assert "background: var(--button-primary-bg-hover)" in hover_rule
    assert "border-color: var(--button-primary-border-hover)" in hover_rule
    assert "box-shadow: var(--button-solid-shadow-hover)" in hover_rule
    assert "transform:" not in hover_rule

    secondary_rule = _css_rule(styles_css, ".button.secondary")
    assert "border-color: var(--border-strong)" in secondary_rule
    assert "background: var(--surface)" in secondary_rule
    assert "box-shadow: var(--button-secondary-shadow)" in secondary_rule

    secondary_hover_rule = _css_rule(
        styles_css, ".button.secondary:hover:not(:disabled),\n.button.secondary:focus-visible:not(:disabled)"
    )
    assert "border-color: var(--border-strong)" in secondary_hover_rule
    assert "color-mix(in srgb, var(--surface) 88%, var(--text) 12%)" in secondary_hover_rule
    assert "box-shadow: var(--button-secondary-shadow-hover)" in secondary_hover_rule


def test_theme_button_tokens_drive_create_environment_and_model_buttons():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")

    # The execution-environment panel auto-saves on row click (radiogroup), so
    # the #saveExecutionEnvironmentButton was retired and no longer carries the
    # solid brand styling.
    assert "#saveExecutionEnvironmentButton" not in styles_css
    for selector in [
        "#createTaskButton.button.primary",
        "#saveLLMEngineEditButton.button.primary",
        "#createTaskButton.button.primary:hover:not(:disabled)",
        "#saveLLMEngineEditButton.button.primary:hover:not(:disabled)",
    ]:
        assert selector not in styles_css

    nav_start = styles_css.index(".nav-action {")
    nav_end = styles_css.index("}", nav_start)
    nav_rule = styles_css[nav_start:nav_end]
    assert "color: var(--button-primary-text)" in nav_rule
    assert "background: var(--button-primary-bg)" in nav_rule
    assert "border: 0" in nav_rule
    assert "box-shadow: var(--button-solid-shadow)" in nav_rule
    assert "0 7px 10px" not in nav_rule
    assert "0 5px 10px" not in nav_rule
    assert "0 10px 14px" not in nav_rule
    assert "linear-gradient" not in nav_rule
    assert "inset" not in nav_rule

    send_start = styles_css.index(".agent-send {")
    send_end = styles_css.index("}", send_start)
    send_rule = styles_css[send_start:send_end]
    assert "color: var(--button-primary-text)" in send_rule
    assert "background: var(--button-primary-bg)" in send_rule

    assert 'id="refreshExecutionEnvironmentOptionsButton" class="button primary"' in index_html
    assert 'id="addLLMModelButton" class="button primary"' in index_html
    assert 'id="refreshExecutionEnvironmentOptionsButton" class="button secondary"' not in index_html
    assert 'id="addLLMModelButton" class="button secondary"' not in index_html

    llm_add_rule = _css_rule(styles_css, ".llm-engine-add")
    assert "color: var(--button-outline-text)" in llm_add_rule
    assert "border: 1px dashed var(--button-outline-border)" in llm_add_rule

    llm_add_hover_rule = _css_rule(styles_css, ".llm-engine-add:hover")
    assert "color: var(--button-outline-text-hover)" in llm_add_hover_rule
    assert "border-color: var(--button-outline-border-hover)" in llm_add_hover_rule
    assert "background: var(--button-outline-bg-hover)" in llm_add_hover_rule

    settings_action_width_start = styles_css.index(
        "#refreshExecutionEnvironmentOptionsButton.button.primary,"
    )
    settings_action_width_end = styles_css.index("}", settings_action_width_start)
    settings_action_width_rule = styles_css[settings_action_width_start:settings_action_width_end]
    assert "#addLLMModelButton.button.primary" in settings_action_width_rule
    assert "width: 84px" in settings_action_width_rule
    assert "min-height: 34px" in settings_action_width_rule
    assert "padding: 6px 12px" in settings_action_width_rule
    assert "font-size: 13px" in settings_action_width_rule
    assert "font-weight: 600" in settings_action_width_rule

    assert "#refreshExecutionEnvironmentOptionsButton.button.secondary {\n" not in styles_css
    assert "#addLLMModelButton.button.secondary {\n" not in styles_css
    assert "#refreshExecutionEnvironmentOptionsButton.button.secondary:hover:not(:disabled)" not in styles_css
    assert "#addLLMModelButton.button.secondary:hover:not(:disabled)" not in styles_css
    assert "#refreshExecutionEnvironmentOptionsButton.button.secondary:focus-visible:not(:disabled)" not in styles_css
    assert "#addLLMModelButton.button.secondary:focus-visible:not(:disabled)" not in styles_css
    assert 'body[data-theme="dark"] #refreshExecutionEnvironmentOptionsButton.button.secondary' not in styles_css
    assert 'body[data-theme="dark"] #addLLMModelButton.button.secondary' not in styles_css
    assert "border: 1px dashed var(--brand-primary)" not in settings_action_width_rule

    hover_start = styles_css.index(".button.primary:hover:not(:disabled)")
    hover_end = styles_css.index("}", hover_start)
    hover_rule = styles_css[hover_start:hover_end]
    assert "background: var(--button-primary-bg-hover)" in hover_rule
    assert "border-color: var(--button-primary-border-hover)" in hover_rule


def test_sidebar_task_card_is_two_line_compact_without_icon():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    append_start = app_js.index("function appendTaskRow")
    append_end = app_js.index("function renderTaskSnapshot", append_start)
    append_renderer = app_js[append_start:append_end]

    assert "task-row-icon" not in append_renderer
    assert "task-row-top" in append_renderer
    assert "task-row-meta" in append_renderer
    assert "task-row-validator" in append_renderer
    assert "task-row-validator-icon" in append_renderer
    assert "task-row-validator-text" in append_renderer
    assert "task-row-date" in append_renderer
    assert 'aria-label="验证人员：${validatorName}"' in append_renderer
    assert ">验证人员：" not in append_renderer
    assert "delete-task-button" in append_renderer
    assert "formatDate(task.updated_at)" in append_renderer
    assert 'const validatorName = escapeHtml(task.validator || "-");' in append_renderer
    delete_hover_rule = _css_rule(
        styles_css, ".delete-task-button:hover,\n.delete-task-button:focus-visible"
    )
    assert "border-color: var(--danger-border)" in delete_hover_rule

    tone_start = app_js.index("function statusTone")
    tone_end = app_js.index("function notebookReproducibilityComplete", tone_start)
    tone_renderer = app_js[tone_start:tone_end]
    assert 'if (status === "failed") return "danger";' in tone_renderer
    assert 'if (status === "review_required") return "success";' in tone_renderer
    assert 'status === "succeeded" || status === "executed"' in tone_renderer
    assert 'status === "running" || status === "computing_metrics") return "run";' in tone_renderer
    assert 'status === "running" || status === "computing_metrics") return "warning";' not in tone_renderer
    assert 'status === "failed" || status === "review_required"' not in tone_renderer
    # writing_artifacts is dual-meaning (idle vs. report-job-running) so its
    # tone must be resolved by taskStatusTone via active_job_kind, not by the
    # status-only statusTone fallback.
    assert '|| status === "writing_artifacts"' not in tone_renderer

    row_start = styles_css.index("\n.task-row {\n  --task-card-action-space")
    row_end = styles_css.index("}", row_start)
    row_rule = styles_css[row_start:row_end]
    assert "--task-card-action-space: 36px" in row_rule
    assert "padding: 11px 12px" in row_rule
    assert "border: 1px solid transparent" in row_rule
    assert "padding: 11px 42px" not in row_rule

    top_start = styles_css.index(".task-row-top {")
    top_end = styles_css.index("}", top_start)
    top_rule = styles_css[top_start:top_end]
    assert "grid-template-columns: minmax(0, 1fr) max-content" in top_rule
    assert "padding-right: var(--task-card-action-space)" in top_rule

    meta_start = styles_css.index("\n.task-row-meta {", top_end)
    meta_end = styles_css.index("}", meta_start)
    meta_rule = styles_css[meta_start:meta_end]
    assert "display: grid" in meta_rule
    assert "grid-template-columns: minmax(0, 1fr) max-content" in meta_rule
    assert "padding-right" not in meta_rule

    name_start = styles_css.index(".task-row-name {")
    name_end = styles_css.index("}", name_start)
    name_rule = styles_css[name_start:name_end]
    assert "min-width: 0" in name_rule

    validator_start = styles_css.index(".task-row .task-row-validator {")
    validator_end = styles_css.index("}", validator_start)
    validator_rule = styles_css[validator_start:validator_end]
    assert "display: inline-flex" in validator_rule
    assert "align-items: center" in validator_rule
    assert "gap: 4px" in validator_rule
    assert "text-overflow: ellipsis" in validator_rule
    assert "white-space: nowrap" in validator_rule

    validator_icon_start = styles_css.index(".task-row-validator-icon {")
    validator_icon_end = styles_css.index("}", validator_icon_start)
    validator_icon_rule = styles_css[validator_icon_start:validator_icon_end]
    assert "width: 16px" in validator_icon_rule
    assert "height: 16px" in validator_icon_rule
    assert "width: 14px" not in validator_icon_rule
    assert "height: 14px" not in validator_icon_rule
    assert "stroke: currentColor" in validator_icon_rule
    assert "flex: 0 0 auto" in validator_icon_rule

    validator_text_start = styles_css.index(".task-row-validator-text {")
    validator_text_end = styles_css.index("}", validator_text_start)
    validator_text_rule = styles_css[validator_text_start:validator_text_end]
    assert "overflow: hidden" in validator_text_rule
    assert "text-overflow: ellipsis" in validator_text_rule
    assert "white-space: nowrap" in validator_text_rule

    date_start = styles_css.index(".task-row .task-row-date {")
    date_end = styles_css.index("}", date_start)
    date_rule = styles_css[date_start:date_end]
    assert "font-size: 12px" in date_rule
    assert "white-space: nowrap" in date_rule

    pill_start = styles_css.index(".task-row-top .pill {")
    pill_end = styles_css.index("}", pill_start)
    pill_rule = styles_css[pill_start:pill_end]
    assert "flex: 0 0 auto" in pill_rule
    assert "white-space: nowrap" in pill_rule

    run_pill_start = styles_css.index(".pill.run {")
    run_pill_end = styles_css.index("}", run_pill_start)
    run_pill_rule = styles_css[run_pill_start:run_pill_end]
    assert "color: var(--accent)" in run_pill_rule
    assert "background: var(--accent-soft)" in run_pill_rule


def test_task_list_selected_card_has_no_visible_border_or_outline():
    styles_css = _read_static("styles.css")

    task_list_start = styles_css.index("\n.task-list {\n  display: grid")
    task_list_end = styles_css.index("}", task_list_start)
    task_list_rule = styles_css[task_list_start:task_list_end]
    assert "overflow-y: auto" in task_list_rule
    assert "padding: 3px 10px 14px" in task_list_rule

    selected_rule = _css_rule(styles_css, ".task-row.selected")
    assert "border-color" not in selected_rule
    assert "background: var(--option-selected)" in selected_rule
    assert "box-shadow: none" in selected_rule
    assert "box-shadow: 0 0 0 2px var(--option-focus-ring)" not in selected_rule


def test_header_task_meta_is_compact_and_not_duplicate_status_or_source():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    workspace_view_js = _read_static("js/task-workspace-view.js")
    styles_css = _read_static("styles.css")

    head_start = index_html.index('<header class="workspace-head"')
    head_end = index_html.index("</header>", head_start)
    head_markup = index_html[head_start:head_end]
    result_workspace_start = index_html.index('id="resultWorkspace"')
    result_workspace_end = index_html.index('id="scanSection"', result_workspace_start)
    result_workspace_markup = index_html[result_workspace_start:result_workspace_end]

    snapshot_start = workspace_view_js.index("function snapshotItem")
    snapshot_end = workspace_view_js.index("export function renderCurrentTaskWorkspace", snapshot_start)
    snapshot_renderer = workspace_view_js[snapshot_start:snapshot_end]

    # status lives in the header hero, in order: name+pill, description, meta
    assert head_markup.index('id="actionStatus"') < head_markup.index('id="actionErrorDetail"')
    assert head_markup.index('id="actionErrorDetail"') < head_markup.index('id="taskSnapshot"')
    assert 'class="task-pill"' in head_markup
    # the header is fixed outside the inner scrolling content, so it does not move
    # during trackpad boundary bounce.
    assert 'class="workspace-head"' in result_workspace_markup
    assert 'class="result-scroll-content"' in result_workspace_markup
    assert result_workspace_markup.index('class="workspace-head"') < result_workspace_markup.index('class="result-scroll-content"')
    head_css = styles_css[styles_css.index(".workspace-head {"):styles_css.index("}", styles_css.index(".workspace-head {"))]
    assert "position: relative" in head_css
    # status pill + status-keyed aurora, no "当前状态" tile label
    assert ".task-pill {" in styles_css
    assert '.task-hero[data-tone="fail"]' in styles_css
    assert 'content: "当前状态"' not in styles_css
    assert ".action-error-detail {" in styles_css
    assert ".action-error-detail.error" in styles_css
    # meta uses monochrome icons, not colored squares
    assert ".task-snapshot-item .meta-ico" in styles_css
    assert ".workspace-task-meta .task-snapshot-item::before" not in styles_css
    assert 'snapshotItem("状态"' not in snapshot_renderer
    assert 'snapshotItem("字段来源"' not in snapshot_renderer
    assert 'snapshotItem("显式材料"' not in snapshot_renderer
    assert "task-meta-tile" in snapshot_renderer
    # validator was dropped from the task-hero snapshot per user feedback;
    # only the execution mode and source directory remain.
    assert "验证人员" not in snapshot_renderer
    assert "执行模式" in snapshot_renderer
    assert "材料目录" in snapshot_renderer
    assert 'snapshotItem("folder", "材料目录", selectedTask.source_dir, null, {' in snapshot_renderer
    assert "copy: selectedTask.source_dir" in snapshot_renderer
    assert 'class="task-snapshot-copy"' in snapshot_renderer
    assert 'data-copy="${escapeHtml(options.copy)}"' in snapshot_renderer
    assert 'aria-label="复制${escapeHtml(label)}路径"' in snapshot_renderer
    assert "copyText(copyButton.dataset.copy)" in app_js


def test_header_task_meta_values_stay_on_one_line():
    styles_css = _read_static("styles.css")

    head_start = styles_css.index(".workspace-head {")
    head_end = styles_css.index("}", head_start)
    head_rule = styles_css[head_start:head_end]
    assert "min-width: 0" in head_rule
    # fixed header layer; the card keeps the frosted glass surface instead of an
    # opaque shell.
    assert "position: relative" in head_rule
    assert "margin: 0" in head_rule
    assert "padding: 0" in head_rule
    assert "background: transparent" in head_rule

    result_start = styles_css.index(".result-workspace {")
    result_end = styles_css.index("}", result_start)
    result_rule = styles_css[result_start:result_end]
    assert "margin: 14px 0 14px var(--workspace-main-gutter)" in result_rule
    assert "overflow: hidden" in result_rule

    scroll_start = styles_css.index(".result-scroll-content {")
    scroll_end = styles_css.index("}", scroll_start)
    scroll_rule = styles_css[scroll_start:scroll_end]
    assert "overflow-y: auto" in scroll_rule
    assert "overscroll-behavior-y: none" in scroll_rule
    assert "grid-row: 1 / -1" in scroll_rule
    assert "padding-top: calc(var(--workspace-head-space) + 12px)" in scroll_rule

    # meta is a single horizontal icon row, values truncate on one line
    list_start = styles_css.index(".task-snapshot-list {")
    list_end = styles_css.index("}", list_start)
    list_rule = styles_css[list_start:list_end]
    assert "display: flex" in list_rule
    assert "min-width: 0" in list_rule

    tile_start = styles_css.index(".task-snapshot-item {")
    tile_end = styles_css.index("}", tile_start)
    tile_rule = styles_css[tile_start:tile_end]
    assert "display: flex" in tile_rule
    assert "align-items: center" in tile_rule

    value_start = styles_css.index(".task-snapshot-item strong {")
    value_end = styles_css.index("}", value_start)
    value_rule = styles_css[value_start:value_end]
    assert "overflow: hidden" in value_rule
    assert "text-overflow: ellipsis" in value_rule
    assert "white-space: nowrap" in value_rule

    copy_rule = _css_rule(styles_css, ".task-snapshot-copy")
    assert "display: inline-flex" in copy_rule
    assert "max-width: 360px" in copy_rule
    assert "background: transparent" in copy_rule
    assert "cursor: copy" in copy_rule
    assert "appearance: none" in copy_rule

    copy_value_rule = _css_rule(styles_css, ".task-snapshot-copy strong")
    assert "max-width: inherit" in copy_value_rule
    last_copy_rule = _css_rule(styles_css, ".workspace-task-meta .task-snapshot-item:last-child .task-snapshot-copy")
    assert "max-width: 460px" in last_copy_rule


def test_step_rail_embeds_notebook_steps_inside_notebook_action_card():
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")

    renderer_start = app_js.index("function stepActionButtonHtml")
    renderer_end = app_js.index("function formatDate", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert "data-step-action" in renderer
    assert "step-action-button" in renderer
    assert "renderNotebookStepRail" in renderer
    assert 'step.id === "notebook"' in renderer
    assert 'insertAdjacentHTML("beforeend", renderNotebookStepRail' not in renderer
    assert "Notebook 标题步骤" not in app_js
    assert "分段进度" in renderer
    assert "扫描或执行后显示 Notebook 分段进度。" not in app_js
    assert "notebookSteps.map" in app_js
    assert "succeeded" in renderer
    assert "running" in renderer
    assert "failed" in renderer

    steps_start = state_js.index("export const workflowSteps = [")
    steps_end = state_js.index("];", steps_start)
    steps_config = state_js[steps_start:steps_end]
    assert 'title: "模型材料完备性验证"' in steps_config
    assert 'hint: "巡检材料内容"' in steps_config
    assert 'title: "模型可复现性验证"' in steps_config
    assert 'hint: "执行建模代码"' in steps_config
    assert 'title: "模型效果&稳定性验证"' in steps_config
    assert 'hint: "指标概览"' in steps_config
    assert 'title: "报告输出"' in steps_config
    assert 'hint: "Word报告与Excel分析"' in steps_config
    assert 'title: "材料识别"' not in steps_config
    assert 'title: "执行 Notebook"' not in steps_config
    assert 'title: "指标概览"' not in steps_config
    assert 'title: "Word 输出"' not in steps_config
    assert "模型验证报告自动编写" not in steps_config
    assert "文件与 RMC 字段检查" not in steps_config
    assert "按标题分段执行" not in steps_config
    assert "生成指标与 Excel" not in steps_config
    assert "填充模板并保存" not in steps_config

    assert "handleWorkflowStepperClick" in app_js
    for action_id in ["scan", "notebook", "metrics", "report"]:
        assert action_id in renderer


def test_step_rail_splits_notebook_and_metrics_progress_with_elapsed_time():
    app_js = _read_static("app.js")

    assert "function stepWorkflowStage" in app_js
    assert "function stepElapsedSeconds" in app_js
    assert "function formatStepElapsed" in app_js
    assert "function stepAfterInLatestNotebookSteps" in app_js
    assert "metricStepsForRail" in app_js
    assert "notebookStepsForRail" in app_js
    assert 'step.id === "notebook" ? renderNotebookStepRail(notebookStepsForRail(), "分段进度", index + 1, stepStatus, "notebook") : ""' in app_js
    assert 'step.id === "metrics" ? renderNotebookStepRail(metricStepsForRail(), "计算进度", index + 1, stepStatus, "metrics") : ""' in app_js
    assert "formatStepElapsed(step, notebookSteps[index + 1] || stepAfterInLatestNotebookSteps(step))" in app_js
    assert "elapsed_seconds" in app_js
    elapsed_start = app_js.index("function stepElapsedSeconds")
    elapsed_end = app_js.index("function formatStepElapsed", elapsed_start)
    elapsed_helper = app_js[elapsed_start:elapsed_end]
    assert "nextStep = null" in elapsed_helper
    assert "nextStartedAt" in elapsed_helper
    assert 'step?.status !== "running" && Number.isFinite(step?.elapsed_seconds)' in elapsed_helper
    assert "Date.now()" in elapsed_helper

    elapsed_start = app_js.index("function formatStepElapsed")
    elapsed_end = app_js.index("function renderNotebookStepRail", elapsed_start)
    elapsed_formatter = app_js[elapsed_start:elapsed_end]
    assert 'return "0s";' in elapsed_formatter
    assert "totalSeconds === 0" in elapsed_formatter


def test_validate_action_primes_reproducibility_system_steps_immediately():
    app_js = _read_static("app.js")

    assert "function plannedReproducibilitySteps" in app_js
    assert "function appendPendingReproducibilitySteps" in app_js
    assert 'id: "system-repro-pmml"' in app_js
    assert 'title: "PMML 打分"' in app_js
    assert 'id: "system-repro-compare"' in app_js
    assert 'title: "分数一致性对比"' in app_js

    validate_start = app_js.index("async function validateCurrentTask")
    validate_end = app_js.index("async function cancelCurrentNotebook", validate_start)
    validate_renderer = app_js[validate_start:validate_end]
    assert "appendPendingReproducibilitySteps();" in validate_renderer
    assert validate_renderer.index("renderValidationResult(result)") < validate_renderer.index(
        "appendPendingReproducibilitySteps();"
    )
    assert validate_renderer.index("appendPendingReproducibilitySteps();") < validate_renderer.index(
        "pollValidationProgress"
    )


def test_metrics_action_primes_metric_system_steps_immediately():
    app_js = _read_static("app.js")

    assert "function plannedMetricSteps" in app_js
    assert "function appendPendingMetricSteps" in app_js
    for step_id, title in [
        ("system-metrics-prepare", "指标数据准备"),
        ("system-metrics-score", "RMC_SCORE_FN 全量打分"),
        ("system-metrics-basic", "样本与变量概览"),
        ("system-metrics-ks", "KS 计算"),
        ("system-metrics-psi", "PSI 计算"),
        ("system-metrics-binning", "分箱计算"),
        ("system-metrics-stress", "压力测试"),
        ("system-metrics-output", "写入指标产物"),
    ]:
        assert f'id: "{step_id}"' in app_js
        assert f'title: "{title}"' in app_js

    metrics_start = app_js.index("async function generateMetrics")
    metrics_end = app_js.index("async function generateReport", metrics_start)
    metrics_renderer = app_js[metrics_start:metrics_end]
    assert "appendPendingMetricSteps();" in metrics_renderer
    assert metrics_renderer.index("appendPendingMetricSteps();") < metrics_renderer.index(
        "pollValidationProgress"
    )


def test_notebook_step_progress_items_are_single_line_and_compact():
    styles_css = _read_static("styles.css")

    step_start = styles_css.index(".notebook-step {")
    step_end = styles_css.index("}", step_start)
    step_rule = styles_css[step_start:step_end]
    assert "grid-template-columns: 16px auto minmax(0, 1fr) auto" in step_rule
    assert "min-height: 28px" in step_rule
    assert "padding: 4px 8px" in step_rule

    title_start = styles_css.index(".notebook-step strong {")
    title_end = styles_css.index("}", title_start)
    title_rule = styles_css[title_start:title_end]
    assert "white-space: nowrap" in title_rule
    assert "overflow: hidden" in title_rule
    assert "text-overflow: ellipsis" in title_rule
    assert "overflow-wrap: normal" in title_rule

    cells_start = styles_css.index(".notebook-step small {")
    cells_end = styles_css.index("}", cells_start)
    cells_rule = styles_css[cells_start:cells_end]
    assert "grid-column: auto" in cells_rule
    assert "white-space: nowrap" in cells_rule
    assert "tabular-nums" in cells_rule


def test_running_step_buttons_turn_into_cancel_buttons():
    app_js = _read_static("app.js")
    renderer_start = app_js.index("function stepStopAction")
    renderer_end = app_js.index("function notebookStepTone", renderer_start)
    renderer = app_js[renderer_start:renderer_end]
    handler_start = app_js.index("function workflowActionConfig")
    handler_end = app_js.index("function scrollStepTarget", handler_start)
    handler = app_js[handler_start:handler_end]

    assert "async function cancelCurrentNotebook" in app_js
    assert "async function cancelCurrentMetrics" in app_js
    assert "async function cancelCurrentReport" in app_js
    assert "async function cancelCurrentScan" not in app_js
    assert '`api/tasks/${taskId}/notebook/cancel`' in app_js
    assert '`api/tasks/${taskId}/metrics/cancel`' in app_js
    assert '`api/tasks/${taskId}/report/cancel`' in app_js
    assert 'function stepStopAction' in app_js
    assert '"cancelNotebook"' in renderer
    assert '"cancelMetrics"' in renderer
    assert '"cancelReport"' in renderer
    assert '"cancelScan"' not in renderer
    assert 'taskServerBusyAction() === "report"' in renderer
    assert '"停止"' in renderer
    assert "selectedBusy && !isStopAction" in renderer
    assert 'actionId === "cancelNotebook"' in handler
    assert 'actionId === "cancelMetrics"' in handler
    assert 'actionId === "cancelReport"' in handler
    assert 'actionId === "cancelScan"' not in handler
    assert "cancelCurrentNotebook" in handler
    assert "cancelCurrentMetrics" in handler
    assert "cancelCurrentReport" in handler


def test_running_visual_tone_uses_header_status_blue():
    styles_css = _read_static("styles.css")

    header_run_start = styles_css.index(".task-pill.run")
    header_run_end = styles_css.index("}", header_run_start)
    header_run_rule = styles_css[header_run_start:header_run_end]
    assert "color: var(--accent)" in header_run_rule
    assert "background: color-mix(in srgb, var(--accent)" in header_run_rule

    action_busy_start = styles_css.index(".action-status.busy")
    action_busy_end = styles_css.index("}", action_busy_start)
    action_busy_rule = styles_css[action_busy_start:action_busy_end]
    assert "color: var(--accent)" in action_busy_rule

    check_running_start = styles_css.index(".check-icon.running")
    check_running_end = styles_css.index("}", check_running_start)
    check_running_rule = styles_css[check_running_start:check_running_end]
    assert "border-top-color: var(--accent)" in check_running_rule

    step_running_start = styles_css.index(".step.running .step-number")
    step_running_end = styles_css.index("}", step_running_start)
    step_running_rule = styles_css[step_running_start:step_running_end]
    assert "color: var(--accent)" in step_running_rule


def test_busy_state_is_scoped_to_selected_task_for_parallel_tasks():
    app_js = _read_static("app.js")

    assert "const taskBusyActions = new Map();" in app_js
    assert "let isBusy" not in app_js
    assert "let busyAction" not in app_js
    assert "function taskBusyAction" in app_js
    assert "function taskServerBusyAction" in app_js
    assert "function selectedTaskIsBusy" in app_js
    assert "active_job_kind" in app_js
    server_busy_start = app_js.index("function taskServerBusyAction")
    server_busy_end = app_js.index("function selectedTaskIsBusy", server_busy_start)
    server_busy = app_js[server_busy_start:server_busy_end]
    assert server_busy.index('kind === "agent"') < server_busy.index("taskStopped(task)")
    assert 'kind === "plan"' in server_busy
    assert 'kind === "join"' in server_busy

    renderer_start = app_js.index("function stepActionButtonHtml")
    renderer_end = app_js.index("function notebookStepTone", renderer_start)
    renderer = app_js[renderer_start:renderer_end]
    assert "const selectedBusy = selectedTaskIsBusy();" in renderer
    assert "selectedBusy && !isStopAction" in renderer
    assert "current task is running" not in renderer

    downloads_ready_start = app_js.index("function completedReportReadyForDownloads")
    downloads_ready_end = app_js.index("function stepDownloadActionsHtml", downloads_ready_start)
    downloads_ready = app_js[downloads_ready_start:downloads_ready_end]
    assert "const selectedBusyAction = taskBusyAction();" in downloads_ready
    assert "selectedTask?.report_available === true" in downloads_ready

    status_start = app_js.index("function taskActionStatusSnapshot")
    status_end = app_js.index("function clearStatus", status_start)
    status_snapshot = app_js[status_start:status_end]
    assert 'task.active_job_kind === "join"' in status_snapshot
    assert "数据拼接进行中。" in status_snapshot
    assert 'task.active_job_kind === "plan"' in status_snapshot
    assert "计划执行进行中。" in status_snapshot

    action_start = app_js.index("async function runAction")
    action_end = app_js.index("function handleTaskListKeydown", action_start)
    action_runner = app_js[action_start:action_end]
    assert "const taskId = options.taskId || selectedTaskId;" in action_runner
    assert 'setBusy(actionId, options.busyText || "正在处理...", taskId)' in action_runner
    assert 'setBusy(null, "", taskId)' in action_runner


def test_workflow_actions_are_gated_by_completed_previous_steps():
    app_js = _read_static("app.js")

    assert "function taskFailedDuringScan" in app_js
    scan_start = app_js.index("function taskFailedDuringScan")
    scan_end = app_js.index("function taskFailureWasRestartReclaim", scan_start)
    scan_helper = app_js[scan_start:scan_end]
    assert "status_message" not in scan_helper
    assert 'normalizedFailureStage(task.failure_stage) === "scan"' in scan_helper
    recommended_start = app_js.index("function recommendedAction")
    recommended_end = app_js.index("function canRunStepAction", recommended_start)
    recommended = app_js[recommended_start:recommended_end]
    assert 'if (status === "created" || taskFailedDuringScan(selectedTask)) return "scan";' in recommended
    assert 'if (taskFailedDuringMetrics(selectedTask)) return "metrics";' in recommended
    assert 'if (taskFailedDuringReport(selectedTask)) return "report";' in recommended
    assert 'if (taskFailedDuringNotebook(selectedTask)) return "notebook";' in recommended

    can_run_start = app_js.index("function canRunStepAction")
    can_run_end = app_js.index("function stepActionButtonHtml", can_run_start)
    can_run = app_js[can_run_start:can_run_end]
    assert 'return ["created", "scanned", "failed", "executed", "writing_artifacts", "succeeded", "review_required"].includes(status);' in can_run
    assert 'case "notebook":' in can_run
    assert 'if (taskFailedDuringScan(selectedTask)) return false;' in can_run
    assert 'return ["scanned", "configured", "executed", "writing_artifacts", "succeeded", "review_required"].includes(status)' in can_run
    assert "|| taskFailedDuringNotebook(selectedTask);" in can_run
    assert 'case "metrics":' in can_run
    assert 'return status === "executed" || taskFailedDuringMetrics(selectedTask);' in can_run
    assert 'case "report":' in can_run
    assert 'return ["writing_artifacts", "review_required"].includes(status) || taskFailedDuringReport(selectedTask);' in can_run

    renderer_start = app_js.index("function stepActionButtonHtml")
    renderer_end = app_js.index("function notebookStepTone", renderer_start)
    renderer = app_js[renderer_start:renderer_end]
    assert "const canRunAction = isStopAction || canRunStepAction(step.action);" in renderer
    assert "|| !canRunAction" in renderer
    assert "请先完成上一步" in renderer


def test_workflow_step_status_separates_next_action_from_running_action():
    app_js = _read_static("app.js")

    status_start = app_js.index("function workflowStepStatus")
    status_end = app_js.index("function workflowStepStatusLabel", status_start)
    status_renderer = app_js[status_start:status_end]
    running_start = app_js.index("function taskRunningStepId")
    running_end = app_js.index("function workflowStepStatus", running_start)
    running_helper = app_js[running_start:running_end]

    assert "function taskRunningStepId" in app_js
    assert 'return "notebook";' in running_helper
    assert 'status === "executed"' in status_renderer
    assert 'return index < 2 ? "succeeded" : "pending";' in status_renderer
    assert 'status === "computing_metrics"' in status_renderer
    assert 'return index < 2 ? "succeeded" : index === 2 ? "running" : "pending";' in status_renderer
    assert 'taskServerBusyAction() === "report"' in status_renderer
    assert 'return index < 3 ? "succeeded" : index === 3 && taskServerBusyAction() === "report" ? "running" : "pending";' in status_renderer
    assert 'if (status === "review_required") return "succeeded";' in status_renderer
    assert 'if (runningStepId && step.id === runningStepId) return "running";' in status_renderer
    assert "if (index === activeIndex) return \"running\";" not in status_renderer


def test_modeling_create_dialog_has_algorithm_selector():
    """The create dialog exposes a manual-mode modeling algorithm multi-select
    (G2: 算法可选), gated to modeling tasks via the algorithmField flag, and
    posted as `payload.recipes` + `payload.target_type`."""
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    create_dialog_js = _read_static("js/create-task-dialog.js")
    task_types_js = _read_static("js/task-types.js")

    assert 'id="createTaskAlgorithmField"' in index_html
    assert 'id="modelAlgorithmChoices"' in index_html
    assert 'name="modelAlgorithm"' in index_html
    for recipe in ('value="lgb"', 'value="xgb"', 'value="catboost"', 'value="lr"', 'value="scorecard"', 'value="mlp"'):
        assert recipe in index_html
    assert 'id="modelSampleWeightCol"' in index_html
    assert 'data-recipe-family="binary"' in index_html
    # regression + multiclass target types are exposed too, so the UI can drive
    # §8.2/§8.3 tasks, not only binary
    assert 'value="lgb_regressor"' in index_html
    assert 'value="lgb_multiclass"' in index_html
    assert 'data-recipe-family="continuous"' in index_html
    assert 'data-recipe-family="multiclass"' in index_html
    assert 'id="modelSampleWeightPolicy"' in index_html
    assert 'value="none">不使用样本权重' in index_html
    assert 'value="explicit">指定权重列' in index_html
    assert "updateSampleWeightCreateState" in create_dialog_js
    assert "algorithmField: true" in task_types_js
    assert 'payload.recipes = [...document.querySelectorAll(\'input[name="modelAlgorithm"]:checked\')].map((box) => box.value);' in create_dialog_js
    assert 'payload.target_type = [...families][0] || "binary";' in create_dialog_js
    assert 'const sampleWeightPolicy = $("modelSampleWeightPolicy")?.value || "none";' in create_dialog_js
    assert "请填写样本权重列，或改选不使用样本权重。" in create_dialog_js
    assert "payload.sample_weight_col = sampleWeightCol;" in create_dialog_js
    assert "normalizeModelAlgorithmFamilies" in create_dialog_js
    assert "二分类、回归与多分类算法不能混选。" in create_dialog_js
    assert "请至少选择一个建模算法。" in create_dialog_js
    assert 'payload.algorithm = $("modelAlgorithm")' not in app_js


def test_strategy_and_vintage_welcome_cards_are_enabled():
    """风险分析(vintage) + 策略开发(strategy) are wired PlanDriver entries."""
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    create_dialog_js = _read_static("js/create-task-dialog.js")
    task_types_js = _read_static("js/task-types.js")
    toast_js = _read_static("js/toast.js")
    for card_id in ("welcomeVintageAnalysisCard", "welcomeStrategyDevelopmentCard"):
        start = index_html.index(f'id="{card_id}"')
        tag_end = index_html.index(">", start)
        tag = index_html[start:tag_end]
        assert "data-coming-soon" not in tag, card_id
        assert 'class="welcome-task-card available"' in tag, card_id
        assert 'aria-describedby="welcomeComingSoonHint"' not in tag, card_id
    definitions = task_types_js[
        task_types_js.index("export const taskTypeDefinitions = {"):
        task_types_js.index("export const taskTypeDisplayOrder")
    ]
    assert 'available: false' not in definitions
    assert 'manualEnabled: false' not in definitions
    plan_js = _read_static("js/v2/plan_rail_controller.js")
    assert 'export const PLAN_RAIL_TASK_TYPES = new Set(["data_join", "feature_analysis", "modeling", "strategy", "vintage"]);' in plan_js
    # The toast path remains available for future explicitly unavailable task definitions.
    assert "card.dataset.comingSoon" not in app_js
    assert "definition.available === false" in create_dialog_js
    assert "unavailableMessage" in create_dialog_js
    assert "export function createComingSoonToastController" in toast_js
    assert 'from "./js/toast.js"' in app_js
    assert "function showComingSoonToast" not in app_js
    assert "const { showComingSoonToast } = createComingSoonToastController" in app_js
    assert 'setActionStatus(message, "info"' in app_js
    assert "新功能开发中，敬请期待" in create_dialog_js


def test_coming_soon_toast_controller_reuses_node_and_resets_timer():
    script = """
import assert from "node:assert/strict";
import { createComingSoonToastController } from "./marvis/static/js/toast.js";

const elements = {};
const appended = [];
const timers = [];
const cleared = [];
function createElement(tagName) {
  assert.equal(tagName, "div");
  return {
    id: "",
    className: "",
    attributes: {},
    classList: {
      values: new Set(),
      add(value) {
        this.values.add(value);
      },
      remove(value) {
        this.values.delete(value);
      },
      contains(value) {
        return this.values.has(value);
      },
    },
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
    textContent: "",
  };
}

const controller = createComingSoonToastController({
  body: {
    appendChild(node) {
      appended.push(node);
      elements[node.id] = node;
    },
  },
  clearTimeoutFn: (timer) => cleared.push(timer),
  createElement,
  getElementById: (id) => elements[id] || null,
  setTimeoutFn: (fn, delay) => {
    const timer = { fn, delay };
    timers.push(timer);
    return timer;
  },
  visibleDurationMs: 1200,
});

controller.showComingSoonToast("敬请期待");
const toast = elements.comingSoonToast;
assert.equal(appended.length, 1);
assert.equal(toast.className, "coming-soon-toast");
assert.equal(toast.attributes.role, "status");
assert.equal(toast.attributes["aria-live"], "polite");
assert.equal(toast.textContent, "敬请期待");
assert.equal(toast.classList.contains("is-visible"), true);
assert.equal(timers[0].delay, 1200);

controller.showComingSoonToast("开发中");
assert.equal(appended.length, 1);
assert.equal(cleared.length, 1);
assert.equal(toast.textContent, "开发中");
timers[1].fn();
assert.equal(toast.classList.contains("is-visible"), false);
process.stdout.write("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_acceptance_chip_relabels_auto_accept_per_task_type():
    """The auto-accept chip label tracks the task type (自动拼接/分析/建模), not always
    自动审查 (ACCEPT-RELABEL)."""
    app_js = _read_static("app.js")
    assert "function autoAcceptLabel" in app_js
    for label in ("自动拼接", "自动分析", "自动建模", "自动审查"):
        assert label in app_js
    assert "autoOption.textContent = autoAcceptLabel(selectedTask?.task_type)" in app_js


def test_feature_create_dialog_has_optional_metric_selector():
    """The feature-analysis create dialog exposes a manual-mode optional-metric
    multi-select (spec §2: 选了才算), gated via the metricField flag and posted as
    `payload.metrics`. VIF is the first wired optional metric; empty is valid."""
    index_html = _read_static("index.html")
    create_dialog_js = _read_static("js/create-task-dialog.js")
    task_types_js = _read_static("js/task-types.js")

    assert 'id="createTaskMetricField"' in index_html
    assert 'id="featureMetricChoices"' in index_html
    assert 'name="featureMetric" value="vif"' in index_html
    assert 'name="featureMetric" value="head_tail_lift"' in index_html
    assert 'name="featureMetric" value="importance"' in index_html
    # no optional metric is pre-checked (base metrics always compute, optional opt-in)
    assert 'name="featureMetric" value="vif" checked' not in index_html
    assert 'name="featureMetric" value="head_tail_lift" checked' not in index_html
    assert 'name="featureMetric" value="importance" checked' not in index_html
    assert "metricField: true" in task_types_js
    assert 'payload.metrics = [...document.querySelectorAll(\'input[name="featureMetric"]:checked\')].map((box) => box.value);' in create_dialog_js


def test_stage_failures_keep_completed_previous_steps_green():
    app_js = _read_static("app.js")

    status_start = app_js.index("function workflowStepStatus")
    status_end = app_js.index("function workflowStepStatusLabel", status_start)
    status_renderer = app_js[status_start:status_end]
    helper_start = app_js.index("function taskFailureStage")
    helper_end = app_js.index("function taskFailureActionStatusTitle", helper_start)
    helper_renderer = app_js[helper_start:helper_end]

    assert "function taskFailureStage" in app_js
    assert "function normalizedFailureStage" in app_js
    assert "const structuredStage = normalizedFailureStage(task.failure_stage);" in helper_renderer
    assert "status_message" not in helper_renderer
    assert "if (structuredStage) return structuredStage;\n  return null;" in helper_renderer
    assert "const failedIndex = workflowSteps.findIndex((candidate) => candidate.id === failedStepId);" in status_renderer
    assert 'if (step.id === failedStepId) return "failed";' in status_renderer
    assert 'if (failedIndex >= 0) return index < failedIndex ? "succeeded" : "pending";' in status_renderer


def test_review_required_marks_all_workflow_steps_as_completed():
    assert _workflow_step_statuses_for(
        {
            "status": "review_required",
            "status_message": "验证已完成，需人工复核报告",
            "active_job_kind": None,
        },
        [],
    ) == ["succeeded", "succeeded", "succeeded", "succeeded"]


def test_structured_failure_stage_overrides_legacy_status_message():
    assert _workflow_step_statuses_for(
        {
            "status": "failed",
            "status_message": "模型可复现性验证失败：notebook failed",
            "active_job_kind": None,
            "failure_stage": "report",
        },
        [],
    ) == ["succeeded", "succeeded", "succeeded", "failed"]


def test_restart_reclaimed_task_keeps_completed_notebook_and_metrics_steps_green():
    app_js = _read_static("app.js")
    restart_body = _slice_function(app_js, "function taskFailureWasRestartReclaim")
    assert "failure_reason_code" in restart_body
    assert "status_message" not in restart_body

    notebook_steps = [
        {"id": "notebook-load", "status": "succeeded"},
        {"id": "system-repro-pmml", "status": "succeeded"},
        {"id": "system-repro-compare", "status": "succeeded"},
        {"id": "system-metrics-prepare", "status": "succeeded"},
        {"id": "system-metrics-score", "status": "succeeded"},
        {"id": "system-metrics-basic", "status": "succeeded"},
        {"id": "system-metrics-ks", "status": "succeeded"},
        {"id": "system-metrics-psi", "status": "succeeded"},
        {"id": "system-metrics-binning", "status": "succeeded"},
        {"id": "system-metrics-stress", "status": "succeeded"},
        {"id": "system-metrics-output", "status": "succeeded"},
    ]

    assert _workflow_step_statuses_for(
        {
            "status": "failed",
            "status_message": "普通失败文案",
            "failure_reason_code": "server_restart_while_running",
            "active_job_kind": None,
        },
        notebook_steps,
    ) == ["succeeded", "succeeded", "succeeded", "pending"]


def test_missing_structured_failure_stage_stays_unknown():
    app_js = _read_static("app.js")

    helper_start = app_js.index("function taskFailureStage")
    helper_end = app_js.index("function taskFailedDuringMetrics", helper_start)
    helper_renderer = app_js[helper_start:helper_end]
    index_start = app_js.index("function workflowIndex")
    index_end = app_js.index("function taskFailureStepId", index_start)
    index_renderer = app_js[index_start:index_end]

    assert "status_message" not in helper_renderer
    assert "return null;" in helper_renderer
    assert 'if (taskFailedDuringMetrics(selectedTask)) return 2;' in index_renderer
    assert 'if (taskFailedDuringReport(selectedTask)) return 3;' in index_renderer
    assert _workflow_step_statuses_for(
        {
            "status": "failed",
            "status_message": "unexpected pipeline failure",
            "active_job_kind": None,
            "failure_stage": None,
        },
        [],
    ) == ["pending", "pending", "pending", "pending"]


def test_workflow_stepper_preserves_scroll_position_during_poll_rerender():
    app_js = _read_static("app.js")

    renderer_start = app_js.index("function renderWorkflowStepper")
    renderer_end = app_js.index("function formatDate", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert "const renderTaskId = selectedTaskId || \"\";" in renderer
    assert "const previousScrollTop = stepper.dataset.taskId === renderTaskId ? stepper.scrollTop : 0;" in renderer
    assert "stepper.dataset.taskId = renderTaskId;" in renderer
    assert "stepper.scrollTop = previousScrollTop;" in renderer
    assert "stepper.scrollTop = 0;" not in renderer


def test_result_workspace_preserves_scroll_position_per_task_switch():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    assert "const resultScrollPositionsByTask = new Map();" in app_js
    assert "let resultScrollPersistFrame = null;" in app_js
    assert "function rememberResultScrollPosition" in app_js
    assert "function loadResultScrollPositions" in app_js
    assert "function persistResultScrollPositions" in app_js
    assert "function prepareResultScrollRestoreForTask" in app_js
    assert "function applyResultScrollPosition" in app_js
    assert "function restoreResultScrollPositionAfterRender" in app_js
    assert "function beginTaskContentLoad" in app_js
    assert "function finishTaskContentLoad" in app_js
    assert 'addEventListener("scroll", handleResultScroll' in app_js
    assert "scheduleResultScrollPositionsPersist();" in _slice_function(app_js, "function rememberResultScrollPosition")
    assert "persistResultScrollPositions();" in _slice_function(app_js, "async function deleteTask")
    assert "await restoreResultScrollPositionAfterRender(selectedTaskId);" in _slice_function(app_js, "async function initializeApp")

    select_start = app_js.index("function selectTask")
    select_end = app_js.index("function deselectCurrentTask", select_start)
    select_renderer = app_js[select_start:select_end]
    assert "rememberResultScrollPosition();" in select_renderer
    assert "beginTaskContentLoad(task.id);" in select_renderer
    assert "prepareResultScrollRestoreForTask(task.id);" in select_renderer
    assert "applyResultScrollPosition(task.id);" not in select_renderer
    assert "scheduleResultScrollRestore(task.id);" not in select_renderer
    assert "renderAll();" in select_renderer
    assert "await restoreResultScrollPositionAfterRender(task.id);" in select_renderer
    assert "finishTaskContentLoad(task.id);" in select_renderer
    assert "}, { renderAfter: false });" in select_renderer

    run_action = _slice_function(app_js, "async function runAction")
    assert "let shouldRenderAfter = options.renderAfter !== false;" in run_action
    assert "shouldRenderAfter = true;" in run_action
    assert "if (shouldRenderAfter) renderAll();" in run_action

    assert ".validation-workspace.is-task-content-loading :is(.workspace-head, .result-scroll-content, .agent-composer, .progress-rail)" in styles_css
    assert ".validation-workspace.is-task-content-loading :is(.result-workspace, .progress-rail)" not in styles_css
    assert "body.anim-ready .validation-workspace:not(.is-task-content-loading) :is(.workspace-head)" in styles_css
    assert "body.anim-ready .validation-workspace:not(.is-task-content-loading) :is(.result-scroll-content)" in styles_css
    assert "transition: opacity 150ms ease 90ms;" in styles_css
    assert ".validation-workspace.is-task-content-settling :is(.task-hero)" in styles_css

    agent_scroll_start = app_js.index("function requestAgentConversationScrollToLatest")
    agent_scroll_end = app_js.index("function renderAgentConversation", agent_scroll_start)
    agent_scroll_renderer = app_js[agent_scroll_start:agent_scroll_end]
    assert "if (suppressAgentAutoScrollTaskId === selectedTaskId) return;" in agent_scroll_renderer
    assert "scrollContent.scrollTo({ top: scrollContent.scrollHeight, behavior: \"auto\" });" in agent_scroll_renderer


def test_workspace_blank_area_wheel_scrolls_center_result_only():
    app_js = _read_static("app.js")

    assert "function routeWorkspaceWheelToResult" in app_js
    handler_start = app_js.index("function routeWorkspaceWheelToResult")
    handler_end = app_js.index("function scrollTargetIsWithin", handler_start)
    handler = app_js[handler_start:handler_end]

    assert "if (event.defaultPrevented || event.ctrlKey) return;" in handler
    assert 'scrollTargetIsWithin(target, "#taskSidebar, #progressRail, #workflowStepper")' in handler
    assert 'scrollTargetIsWithin(target, "#resultScrollContent")' in handler
    assert 'scrollTargetIsWithin(target, "dialog, textarea, select, input, .metric-table-scroll")' in handler
    assert "scrollContent.scrollTop += event.deltaY;" in handler
    assert "if (scrollContent.scrollTop !== previousTop" in handler
    assert "event.preventDefault();" in handler
    assert 'document.addEventListener("wheel", routeWorkspaceWheelToResult, { passive: false });' in app_js


def test_scan_failure_sets_top_status_instead_of_success_message():
    app_js = _read_static("app.js")

    scan_start = app_js.index("async function scanCurrentTask")
    scan_end = app_js.index("async function createTaskAndScan", scan_start)
    scanner = app_js[scan_start:scan_end]

    assert 'selectedTaskIsAgentMode(selectedTask) ? "材料完备性识别完成。" : "材料扫描完成。"' in scanner
    assert "if (selectedTask?.status === \"failed\")" in scanner
    assert "setTaskFailureActionStatus(selectedTask)" in scanner
    assert "return;" in scanner


def test_create_task_auto_scans_materials_after_creation():
    app_js = _read_static("app.js")

    create_start = app_js.index("async function createTask")
    create_end = app_js.index("async function refreshTasks", create_start)
    create_renderer = app_js[create_start:create_end]
    assert "return task" in create_renderer

    handler_start = app_js.index("function createTaskAndScan")
    handler_end = app_js.index("function handleTaskListKeydown", handler_start)
    handler_renderer = app_js[handler_start:handler_end]
    assert "await createTask()" in handler_renderer
    assert "await scanCurrentTask()" in handler_renderer
    assert "await loadTaskEvidence(task.id)" in handler_renderer


def test_create_task_submit_keeps_create_errors_in_dialog_before_task_exists():
    app_js = _read_static("app.js")
    create_dialog_js = _read_static("js/create-task-dialog.js")
    index_html = _read_static("index.html")

    click_start = app_js.index('$("createTaskButton").onclick')
    click_end = app_js.index('$("workflowStepper").onclick', click_start)
    click_handler = app_js[click_start:click_end]
    assert 'runAction(createTaskAndScan);' in click_handler
    assert 'actionId: "scan"' not in click_handler

    keydown_start = app_js.index('event.key === "Enter"')
    keydown_end = app_js.index('document.addEventListener("click"', keydown_start)
    keydown_handler = app_js[keydown_start:keydown_end]
    assert 'runAction(createTaskAndScan);' in keydown_handler
    assert 'actionId: "scan"' not in keydown_handler
    assert 'id="statusMessage" class="status" role="status" aria-live="polite"' in index_html
    assert 'setCreateStatus("请选择执行模式。", "error")' in create_dialog_js
    assert 'setCreateStatus("请先选择要上传的材料文件。", "error")' in create_dialog_js
    assert 'setCreateStatus(\n        definition.reportFields ? "请先填写模型名称、验证人员和材料目录。"' in create_dialog_js


def test_dialogs_close_when_clicking_backdrop():
    app_js = _read_static("app.js")
    dialogs_js = _read_static("js/dialogs.js")
    index_html = _read_static("index.html")

    for dialog_id in [
        "taskDialog",
        "llmEngineEditDialog",
        "governanceSettingsDialog",
        "wordPreviewDialog",
    ]:
        assert f'<dialog id="{dialog_id}"' in index_html

    assert "export function closeDialogOnBackdropClick" in dialogs_js
    assert "export function bindDialogBackdropDismissal" in dialogs_js
    assert 'root.querySelectorAll("dialog").forEach((dialog) =>' in dialogs_js
    assert 'dialog.addEventListener("click", closeDialogOnBackdropClick);' in dialogs_js
    assert "event.target !== dialog || !dialog.open" in dialogs_js
    assert 'from "./js/dialogs.js"' in app_js
    assert "function closeDialogOnBackdropClick" not in app_js
    assert "function bindDialogBackdropDismissal" not in app_js
    assert "bindDialogBackdropDismissal();" in app_js


def test_dialog_backdrop_dismissal_closes_only_open_backdrop_clicks():
    script = """
import assert from "node:assert/strict";
import {
  bindDialogBackdropDismissal,
  closeDialogOnBackdropClick,
} from "./marvis/static/js/dialogs.js";

function makeDialog(open = true) {
  return {
    open,
    closeCalls: 0,
    listeners: {},
    close() {
      this.closeCalls += 1;
      this.open = false;
    },
    addEventListener(name, fn) {
      this.listeners[name] = fn;
    },
  };
}

const dialog = makeDialog(true);
closeDialogOnBackdropClick({ currentTarget: dialog, target: dialog });
assert.equal(dialog.closeCalls, 1);
assert.equal(dialog.open, false);

const innerClickDialog = makeDialog(true);
closeDialogOnBackdropClick({ currentTarget: innerClickDialog, target: { role: "button" } });
assert.equal(innerClickDialog.closeCalls, 0);
assert.equal(innerClickDialog.open, true);

const closedDialog = makeDialog(false);
closeDialogOnBackdropClick({ currentTarget: closedDialog, target: closedDialog });
assert.equal(closedDialog.closeCalls, 0);

const boundDialog = makeDialog(true);
bindDialogBackdropDismissal({
  root: {
    querySelectorAll(selector) {
      assert.equal(selector, "dialog");
      return [boundDialog];
    },
  },
});
boundDialog.listeners.click({ currentTarget: boundDialog, target: boundDialog });
assert.equal(boundDialog.closeCalls, 1);
process.stdout.write("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_dialog_close_buttons_render_as_x_controls():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    for button_id in [
        "closeTaskDialogButton",
        "closeLLMEngineEditButton",
        "closeGovernanceSettingsButton",
        "closeWordPreviewButton",
    ]:
        button_start = index_html.index(f'id="{button_id}"')
        button_start = index_html.rfind("<button", 0, button_start)
        button_end = index_html.index("</button>", button_start)
        button_markup = index_html[button_start:button_end]
        assert "dialog-close-button" in button_markup
        assert 'aria-label="关闭"' in button_markup
        assert 'title="关闭"' in button_markup
        assert "<span aria-hidden=\"true\">&times;</span>" in button_markup
        assert ">关闭" not in button_markup

    close_button_rule = _css_rule(styles_css, ".button.dialog-close-button")
    assert "width: 30px" in close_button_rule
    assert "min-width: 30px" in close_button_rule
    assert "height: 30px" in close_button_rule
    assert "min-height: 30px" in close_button_rule
    assert "padding: 0" in close_button_rule
    assert "border: 0" in close_button_rule
    assert "background: transparent" in close_button_rule
    assert "color: var(--text-secondary)" in close_button_rule
    assert "font-size: 18px" in close_button_rule

    close_button_hover_rule = _css_rule(
        styles_css,
        ".button.dialog-close-button:hover:not(:disabled),\n.button.dialog-close-button:focus-visible:not(:disabled),\n.button.dialog-close-button:active:not(:disabled)",
    )
    assert "outline: none" in close_button_hover_rule
    assert "border-color: transparent" in close_button_hover_rule
    assert "background: var(--option-hover)" in close_button_hover_rule
    assert "box-shadow: none" in close_button_hover_rule
    assert "#f8fbff" not in close_button_hover_rule
    assert "#9fbfe4" not in close_button_hover_rule

    dark_close_button_rule = _css_rule(styles_css, 'body[data-theme="dark"] .button.dialog-close-button')
    assert "border-color: transparent" in dark_close_button_rule
    assert "background: transparent" in dark_close_button_rule
    assert "box-shadow: none" in dark_close_button_rule

    dark_close_button_hover_rule = _css_rule(
        styles_css,
        'body[data-theme="dark"] .button.dialog-close-button:hover:not(:disabled),\nbody[data-theme="dark"] .button.dialog-close-button:focus-visible:not(:disabled),\nbody[data-theme="dark"] .button.dialog-close-button:active:not(:disabled)',
    )
    assert "background: var(--option-hover)" in dark_close_button_hover_rule
    assert "box-shadow: none" in dark_close_button_hover_rule

    assert 'id="governanceRefreshButton" class="governance-icon-button is-unavailable"' in index_html
    assert 'aria-hidden="true" disabled' in index_html
    assert 'button.hidden = !governanceRefreshActions[navKey]' not in app_js
    assert 'button.classList.toggle("is-unavailable", unavailable);' in app_js
    assert "button.disabled = unavailable;" in app_js
    assert 'button.setAttribute("aria-hidden", unavailable ? "true" : "false");' in app_js

    head_rule = _css_rule(styles_css, ".governance-settings-head")
    assert "position: relative" in head_rule
    assert "grid-template-columns: minmax(0, 1fr)" in head_rule
    assert "padding: 20px 96px 16px 24px" in head_rule

    actions_rule = _css_rule(styles_css, ".governance-head-actions")
    assert "position: absolute" in actions_rule
    assert "top: 20px" in actions_rule
    assert "right: 24px" in actions_rule
    assert "justify-content: flex-end" in actions_rule
    assert "width: 68px" in actions_rule

    shared_start = styles_css.index(".governance-head-actions .governance-icon-button {")
    shared_end = styles_css.index("}", shared_start)
    shared_rule = styles_css[shared_start:shared_end]
    assert "width: 30px" in shared_rule
    assert "min-width: 30px" in shared_rule
    assert "height: 30px" in shared_rule
    assert "min-height: 30px" in shared_rule
    assert "border: 0" in shared_rule
    assert "background: transparent" in shared_rule
    assert "box-shadow: none" in shared_rule
    assert ".governance-head-actions .dialog-close-button {" not in styles_css
    unavailable_rule = _css_rule(styles_css, ".governance-icon-button.is-unavailable")
    assert "visibility: hidden" in unavailable_rule
    assert "pointer-events: none" in unavailable_rule

    shared_hover_start = styles_css.index(
        ".governance-head-actions .governance-icon-button:hover:not(:disabled),"
    )
    shared_hover_end = styles_css.index("}", shared_hover_start)
    shared_hover_rule = styles_css[shared_hover_start:shared_hover_end]
    assert ".governance-head-actions .dialog-close-button" not in shared_hover_rule
    assert "outline: none" in shared_hover_rule
    assert "border-color: transparent" in shared_hover_rule
    assert "background: var(--option-hover)" in shared_hover_rule
    assert "box-shadow: none" in shared_hover_rule
    assert ".governance-head-actions .governance-icon-button:active:not(:disabled)" in shared_hover_rule
    assert "#f8fbff" not in shared_hover_rule
    assert "#9fbfe4" not in shared_hover_rule

    dark_shared_start = styles_css.index('body[data-theme="dark"] .governance-head-actions .governance-icon-button {')
    dark_shared_end = styles_css.index("}", dark_shared_start)
    dark_shared_rule = styles_css[dark_shared_start:dark_shared_end]
    assert "border-color: transparent" in dark_shared_rule
    assert "background: transparent" in dark_shared_rule

    dark_hover_start = styles_css.index(
        'body[data-theme="dark"] .governance-head-actions .governance-icon-button:hover:not(:disabled),'
    )
    dark_hover_end = styles_css.index("}", dark_hover_start)
    dark_hover_rule = styles_css[dark_hover_start:dark_hover_end]
    assert 'body[data-theme="dark"] .governance-head-actions .dialog-close-button' not in dark_hover_rule
    assert "background: var(--option-hover)" in dark_hover_rule
    assert "box-shadow: none" in dark_hover_rule
    assert 'body[data-theme="dark"] .governance-head-actions .governance-icon-button:active:not(:disabled)' in dark_hover_rule

    refresh_icon_rule = _css_rule(styles_css, ".governance-icon-button svg")
    assert "width: 15px" in refresh_icon_rule
    assert "height: 15px" in refresh_icon_rule
    assert "stroke-width: 2.25" in refresh_icon_rule


def test_initial_load_restores_task_evidence_for_selected_task():
    app_js = _read_static("app.js")
    bootstrap_start = app_js.index("restoreTheme();")
    bootstrap = app_js[bootstrap_start:]

    assert "await refreshTasks()" in bootstrap
    assert "await loadReportFields()" in bootstrap
    assert "await loadTaskEvidence()" in bootstrap


def test_sort_group_and_theme_live_in_sidebar_settings():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    settings_start = index_html.index('id="sidebarSettings"')
    settings_end = index_html.index("</details>", settings_start)
    settings_markup = index_html[settings_start:settings_end]

    for row in ["sort", "group", "appearance", "pet", "system"]:
        assert f'data-settings-row="{row}"' in settings_markup

    assert 'id="settingsSortSelect"' in settings_markup
    assert 'id="settingsGroupSelect"' in settings_markup
    assert 'id="settingsThemeSelect"' in settings_markup

    for sort_value in ["created_desc", "created_asc", "name_asc", "name_desc"]:
        assert f'value="{sort_value}"' in settings_markup

    for group_value in ["none", "task_type", "validator", "created_month"]:
        assert f'value="{group_value}"' in settings_markup

    for theme_value in ["light", "dark", "system"]:
        assert f'value="{theme_value}"' in settings_markup

    assert 'id="openGovernanceSettingsButton"' in settings_markup
    assert "环境、模型、记忆与 Runtime" not in settings_markup
    assert 'class="settings-system-value"' not in settings_markup
    assert 'class="settings-system-row"' in settings_markup
    assert 'class="settings-environment-button"' not in settings_markup
    assert 'aria-label="系统设置"' not in settings_markup
    assert 'class="settings-section-label">系统设置' not in settings_markup
    assert 'data-settings-row="environment"' not in settings_markup
    assert 'data-settings-row="llm"' not in settings_markup
    assert 'data-settings-row="governance"' not in settings_markup

    assert 'id="taskSortSelect"' not in settings_markup
    assert 'id="taskGroupSelect"' not in settings_markup
    assert 'id="themeToggle"' not in settings_markup
    assert 'data-sort-value=' not in settings_markup
    assert 'data-group-value=' not in settings_markup
    assert 'data-theme-choice=' not in settings_markup
    assert 'data-settings-panel=' not in settings_markup
    assert 'class="task-controls"' not in index_html

    assert 'const taskSortModes = new Set(["created_desc", "created_asc", "name_asc", "name_desc"]);' in app_js
    assert 'const taskGroupModes = new Set(["none", "task_type", "validator", "created_month"]);' in app_js
    assert "function restoreTaskListSettings" in app_js
    assert "function saveTaskListSettings" in app_js
    assert 'localStorage.getItem("marvis_task_list_settings")' in app_js
    assert 'localStorage.setItem("marvis_task_list_settings"' in app_js
    assert "taskSortMode = normalizeTaskSortMode(stored.sort);" in app_js
    assert "taskGroupMode = normalizeTaskGroupMode(stored.group);" in app_js

    change_start = app_js.index("function handleSettingsMenuChange")
    change_end = app_js.index("async function loadExecutionEnvironmentSettings", change_start)
    change_renderer = app_js[change_start:change_end]
    assert "saveTaskListSettings();" in change_renderer

    bootstrap_start = app_js.index("restoreTheme();")
    bootstrap = app_js[bootstrap_start:]
    assert bootstrap.index("restoreTaskListSettings();") < bootstrap.index("renderSettingsState();")
    assert bootstrap.index("restoreTaskListSettings();") < bootstrap.index("await refreshTasks();")


def test_task_group_setting_supports_created_month_and_task_type():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    task_types_js = _read_static("js/task-types.js")
    settings_start = index_html.index('id="sidebarSettings"')
    settings_end = index_html.index("</details>", settings_start)
    settings_markup = index_html[settings_start:settings_end]

    assert '<option value="task_type">按任务类型</option>' in settings_markup
    assert '<option value="created_month">按创建月份</option>' in settings_markup
    assert 'taskGroupMode === "task_type"' in app_js
    assert 'taskGroupMode === "created_month"' in app_js
    assert 'export const taskTypeDisplayOrder = ["data_join", "feature_analysis", "vintage", "modeling", "validation", "strategy"];' in task_types_js
    assert "sortTaskTypeGroups" in app_js
    assert "appendTaskGroup(list, taskTypeLabel(taskType), groupTasks)" in app_js
    assert "function taskCreatedMonth" in app_js
    assert "task.created_at || task.updated_at" in app_js
    assert "未知创建月份" in app_js
    assert "sortMonthGroups" in app_js


def test_sidebar_settings_uses_dropdowns_and_stays_inside_sidebar():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")

    assert 'class="settings-tabs"' not in index_html
    assert 'class="settings-content"' not in index_html
    assert 'class="settings-panel"' not in index_html
    assert 'class="settings-row"' in index_html
    assert 'class="settings-row-title"' in index_html
    assert 'class="settings-select"' in index_html
    assert 'class="settings-system-row"' in index_html
    for row in ["sort", "group", "appearance", "pet"]:
        row_start = index_html.index(f'data-settings-row="{row}"')
        row_start = index_html.rfind('<div class="settings-row"', 0, row_start)
        row_end = index_html.index("</div>", index_html.index("</select>", row_start))
        row_markup = index_html[row_start:row_end]
        assert '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' in row_markup
    system_row_start = index_html.index('id="openGovernanceSettingsButton"')
    system_row_start = index_html.rfind("<button", 0, system_row_start)
    system_row_end = index_html.index("</button>", system_row_start)
    system_row_markup = index_html[system_row_start:system_row_end]
    assert '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' in system_row_markup
    assert 'class="settings-governance-card"' not in index_html
    assert 'class="settings-governance-description"' not in index_html
    assert 'class="settings-option"' not in index_html

    assert ".settings-menu {" in styles_css
    menu_start = styles_css.index(".settings-menu {")
    menu_end = styles_css.index("}", menu_start)
    menu_rule = styles_css[menu_start:menu_end]
    assert "left: 0" in menu_rule
    assert "right: 0" in menu_rule
    assert "gap: 12px" in menu_rule
    assert "width: auto" in menu_rule
    assert "max-width: 100%" in menu_rule
    assert "padding: 14px 16px 16px" in menu_rule
    assert "box-shadow: var(--settings-menu-shadow)" in menu_rule
    assert "width: min(520px" not in menu_rule
    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))
    assert root_vars["--settings-menu-shadow"] == "0 10px 24px rgba(0, 0, 0, 0.10)"
    assert dark_vars["--settings-menu-shadow"] == "0 10px 24px rgba(0, 0, 0, 0.10)"
    section_rule = _css_rule(styles_css, ".settings-menu-section")
    assert "padding: 0" in section_rule
    assert "border: 0" in section_rule
    assert "background: transparent" in section_rule
    assert "border-radius" not in section_rule
    assert "gap: 12px" in section_rule
    label_rule = _css_rule(styles_css, ".settings-section-label")
    assert "padding: 0 2px" in label_rule
    assert ".settings-row {" in styles_css
    row_start = styles_css.index(".settings-row {")
    row_end = styles_css.index("}", row_start)
    row_rule = styles_css[row_start:row_end]
    assert "grid-template-columns: 72px minmax(0, 1fr)" in row_rule
    assert "gap: 8px" in row_rule
    row_title_rule = _css_rule(styles_css, ".settings-row-title")
    assert "display: inline-flex" in row_title_rule
    assert "align-self: center" in row_title_rule
    assert "height: 34px" in row_title_rule
    assert "gap: 7px" in row_title_rule
    assert "margin: 0" in row_title_rule
    assert "font-size: 14px" in row_title_rule
    assert "line-height: 20px" in row_title_rule
    assert "pointer-events: none" in row_title_rule
    shared_icon_start = styles_css.index(".settings-row-title svg,")
    shared_icon_end = styles_css.index("}", shared_icon_start)
    shared_icon_rule = styles_css[shared_icon_start:shared_icon_end]
    assert ".settings-system-row > svg" in shared_icon_rule
    assert "width: 18px" in shared_icon_rule
    assert "height: 18px" in shared_icon_rule
    assert "stroke: currentColor" in shared_icon_rule
    settings_control_start = styles_css.index(".settings-select,\n.settings-system-row {")
    settings_control_end = styles_css.index("}", settings_control_start)
    settings_control_rule = styles_css[settings_control_start:settings_control_end]
    assert "border-radius: var(--radius-control)" in settings_control_rule
    assert "min-height: 34px" in settings_control_rule
    assert "font-size: 14px" in settings_control_rule
    assert "line-height: 20px" in settings_control_rule
    settings_select_rule = _css_rule(styles_css, ".settings-select")
    assert "min-width: 0" in settings_select_rule
    assert "padding: 6px 30px 6px 12px" in settings_select_rule
    assert "text-overflow: ellipsis" in settings_select_rule
    assert "font-size: 14px" in settings_select_rule
    settings_select_option_rule = _css_rule(styles_css, ".settings-select option")
    assert "font-size: 14px" in settings_select_option_rule
    assert ".settings-row:hover" not in styles_css
    system_row_start = styles_css.index(".settings-system-row {\n  display: flex")
    system_row_end = styles_css.index("}", system_row_start)
    system_row_rule = styles_css[system_row_start:system_row_end]
    assert "display: flex" in system_row_rule
    assert "justify-content: center" in system_row_rule
    assert "gap: 7px" in system_row_rule
    assert "height: 34px" in system_row_rule
    assert "min-height: 34px" in system_row_rule
    assert "padding: 6px 12px" in system_row_rule
    assert "text-align: center" in system_row_rule
    system_title_rule = _css_rule(styles_css, ".settings-system-row .settings-row-title")
    assert "height: 20px" in system_title_rule
    assert ".task-sidebar {" in styles_css
    sidebar_start = styles_css.index(".task-sidebar {\n  display: flex")
    sidebar_end = styles_css.index("}", sidebar_start)
    sidebar_rule = styles_css[sidebar_start:sidebar_end]
    # The rail clips by default (clean slide animation) but reveals popovers when open.
    assert "overflow: hidden" in sidebar_rule
    overflow_rule_start = styles_css.index(".task-sidebar:has(.sidebar-settings[open])")
    overflow_rule_end = styles_css.index("}", overflow_rule_start)
    assert "overflow: visible" in styles_css[overflow_rule_start:overflow_rule_end]

    collapsed_menu_start = styles_css.index(
        ".app-shell.sidebar-collapsed .sidebar-settings[open] .settings-menu"
    )
    collapsed_menu_end = styles_css.index("}", collapsed_menu_start)
    collapsed_menu_rule = styles_css[collapsed_menu_start:collapsed_menu_end]
    assert "position: fixed" in collapsed_menu_rule
    assert "left: var(--collapsed-popover-left)" in collapsed_menu_rule
    assert "right: auto" in collapsed_menu_rule
    assert "bottom: 12px" in collapsed_menu_rule
    assert (
        "width: min(calc(var(--rail-content-width, 314px) - 24px), "
        "calc(100vw - var(--collapsed-popover-left) - 16px))"
    ) in collapsed_menu_rule
    assert "max-width: none" in collapsed_menu_rule
    assert "width: 268px" not in collapsed_menu_rule

    assert "function renderSettingsState" in app_js
    assert "function handleSettingsMenuChange" in app_js
    assert "function setActiveSettingsSection" not in app_js
    assert "function markSettingsOptions" not in app_js
    assert '$("settingsMenu").onchange = handleSettingsMenuChange' in app_js
    assert "data-settings-section" not in app_js
    assert "data-settings-panel" not in app_js


def test_appearance_setting_supports_light_dark_and_system_modes():
    app_js = _read_static("app.js")
    theme_js = _read_static("js/theme.js")
    index_html = _read_static("index.html")

    assert 'from "./js/theme.js"' in app_js
    assert "const themeController = createThemeController" in app_js
    assert "function systemTheme" in theme_js
    assert "const watchSystemTheme = () =>" in theme_js
    assert "function syncBrowserChromeTheme" in theme_js
    assert 'const browserChromeThemeColors = {' in theme_js
    assert 'light: "#ffffff"' in theme_js
    assert 'dark: "#181818"' in theme_js
    assert '$("brandFaviconDark")?.setAttribute("media", isDark ? "all" : "not all");' in theme_js
    assert 'preference === "system"' in theme_js
    assert 'localStorage.setItem("marvis_theme", preference)' in theme_js
    assert 'id="settingsThemeSelect"' in index_html
    assert 'value="system"' in index_html
    assert "跟随系统" in index_html
    assert 'id="appThemeColor"' in index_html
    assert 'id="brandFaviconDark"' in index_html
    assert 'id="brandAppleTouchIconDark"' in index_html
    assert 'const syncBrowserChrome = (resolvedTheme) => {' in index_html
    appearance_start = index_html.index('data-settings-row="appearance"')
    appearance_start = index_html.rfind('<div class="settings-row"', 0, appearance_start)
    appearance_end = index_html.index("</div>", index_html.index("</select>", appearance_start))
    appearance_markup = index_html[appearance_start:appearance_end]
    assert '<circle cx="12" cy="12" r="3.4"></circle>' in appearance_markup
    assert '<path d="m18.4 5.6-1.55 1.55"></path>' in appearance_markup
    assert '<path d="m7.15 16.85-1.55 1.55"></path>' in appearance_markup
    assert 'localStorage.getItem("marvis_theme") || "light"' in index_html
    assert 'document.body.dataset.theme = resolvedTheme;' in index_html
    assert '<meta id="appThemeColor" name="theme-color" content="#ffffff" />' in index_html
    assert 'document.getElementById("appThemeColor")?.setAttribute("content", isDark ? "#181818" : "#ffffff");' in index_html
    assert index_html.index('<body class="app-booting" data-theme="light">') < index_html.index('localStorage.getItem("marvis_theme")')
    assert index_html.index('localStorage.getItem("marvis_theme")') < index_html.index('id="taskDialog"')


def test_sidebar_settings_closes_on_outside_click_only():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    assert "function closeSidebarSettingsOnOutsideClick" in app_js
    assert "function openGovernanceSettingsFromSidebar" in app_js
    assert 'document.addEventListener("click", closeSidebarSettingsOnOutsideClick)' in app_js

    handler_start = app_js.index("function closeSidebarSettingsOnOutsideClick")
    handler_end = app_js.index("function workflowActionConfig", handler_start)
    handler = app_js[handler_start:handler_end]
    assert '$("sidebarSettings")' in handler
    assert "settings.open" in handler
    assert 'target.closest("#sidebarSettings")' in handler
    assert "settings.open = false" in handler
    assert "function closeSidebarSettingsMenu" in app_js
    assert "function scheduleGovernanceSettingsFromSidebar" in app_js
    assert "function handleGovernanceSettingsPointerDown" in app_js
    assert "window.requestAnimationFrame" in app_js
    assert '$("openGovernanceSettingsButton").addEventListener("pointerdown", handleGovernanceSettingsPointerDown, true);' in app_js
    assert "function setSidebarSettingsSuppressed" not in app_js
    assert "modal-suppressed" not in app_js
    assert 'openGovernanceSettingsCenter("execution-environment")' in app_js

    assert ".sidebar-settings.modal-suppressed" not in styles_css
    closed_menu_rule = _css_rule(styles_css, ".sidebar-settings:not([open]) .settings-menu")
    assert "display: none" in closed_menu_rule


def test_system_settings_exposes_execution_environment_panel():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")
    settings_start = index_html.index('id="sidebarSettings"')
    settings_end = index_html.index("</details>", settings_start)
    settings_markup = index_html[settings_start:settings_end]

    assert 'id="openGovernanceSettingsButton"' in settings_markup
    assert "系统设置" in settings_markup
    assert "环境、模型、记忆与 Runtime" not in settings_markup
    assert 'id="openExecutionEnvironmentButton"' not in settings_markup
    assert 'id="settingsExecutionEnvironmentValue"' not in settings_markup
    assert 'id="executionEnvironmentDialog"' not in index_html
    assert 'data-governance-nav="execution-environment"' in index_html
    assert 'data-governance-panel-content="execution-environment"' in index_html

    for element_id in [
        "executionEnvironmentList",
        "refreshExecutionEnvironmentOptionsButton",
        "executionEnvironmentStatus",
    ]:
        assert f'id="{element_id}"' in index_html
    # The panel is now a click-to-select radiogroup that persists on click, so
    # the old <select> + explicit save button were retired.
    assert 'id="executionEnvironmentSelect"' not in index_html
    assert 'id="saveExecutionEnvironmentButton"' not in index_html

    env_settings_group_rule = _css_rule(
        styles_css, '.governance-panel[data-governance-panel-content="execution-environment"] > .settings-group'
    )
    assert "border: 0" in env_settings_group_rule
    assert "background: transparent" in env_settings_group_rule
    assert "overflow: visible" in env_settings_group_rule
    env_settings_row_rule = _css_rule(
        styles_css,
        '.governance-panel[data-governance-panel-content="execution-environment"] > .settings-group > .settings-row',
    )
    assert "padding: 0 2px 2px" in env_settings_row_rule

    env_list_rule = _css_rule(styles_css, ".exec-env-list")
    assert "display: flex" in env_list_rule
    assert "gap: 8px" in env_list_rule
    assert "border: 0" in env_list_rule
    assert "background: transparent" in env_list_rule
    env_row_rule = _css_rule(styles_css, ".exec-env-row")
    llm_card_rule = _css_rule(styles_css, ".llm-engine-item")
    assert "background: var(--surface-soft)" in env_row_rule
    assert "background: var(--surface-soft)" in llm_card_rule
    assert "border: 1px solid var(--border)" in env_row_rule
    assert "border: 1px solid var(--border)" in llm_card_rule
    assert "border-radius: var(--radius-control)" in env_row_rule
    assert "border-radius: var(--radius-control)" in llm_card_rule
    env_row_hover_rule = _css_rule(styles_css, ".exec-env-row:hover")
    assert "border-color: var(--option-hover)" in env_row_hover_rule
    assert "background: var(--option-hover)" in env_row_hover_rule


def test_pet_setting_includes_naitang_xiaojiu_auditbots_and_none():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")
    styles_css = _read_static("styles.css")
    settings_start = index_html.index('id="sidebarSettings"')
    settings_end = index_html.index("</details>", settings_start)
    settings_markup = index_html[settings_start:settings_end]

    assert 'data-settings-row="pet"' in settings_markup
    assert 'id="settingsPetSelect"' in settings_markup
    assert '<option value="none">不显示</option>' in settings_markup
    assert '<option value="naitang">蛋黄</option>' in settings_markup
    assert '<option value="xiaojiu">小九</option>' in settings_markup
    expected_auditbot_pets = {
        "auditbot": ("MARVIS", "3D 玩具审计机器人，青色护目镜眼睛和铜色耳机"),
        "auditbot-pro": ("MARVIS Pro", "专业风格 3D 审计机器人"),
        "auditbot-poly": ("MARVIS Poly", "低多边形硬表面审计机器人"),
        "auditbot-ink": ("MARVIS Ink", "技术线稿风格审计机器人"),
        "auditbot-clay": ("MARVIS Clay", "黏土与乙烯基质感审计机器人"),
        "auditbot-comic": ("MARVIS Comic", "漫画描边风格审计机器人"),
        "auditbot-pixel": ("MARVIS Pixel", "像素风审计机器人"),
    }
    for pet_id, (display_name, pet_label) in expected_auditbot_pets.items():
        assert f'<option value="{pet_id}">{display_name}</option>' in settings_markup
        key = f'"{pet_id}": {{' if "-" in pet_id else f"{pet_id}: {{"
        assert key in index_html
        assert f'name: "{display_name}"' in index_html
        assert f'label: "{pet_label}"' in index_html
        assert f'asset: "static/pets/{pet_id}/spritesheet.webp"' in index_html
    pet_row_start = settings_markup.index('data-settings-row="pet"')
    pet_row_start = settings_markup.rfind('<div class="settings-row"', 0, pet_row_start)
    pet_row_end = settings_markup.index("</div>", settings_markup.index("</select>", pet_row_start))
    pet_row_markup = settings_markup[pet_row_start:pet_row_end]
    assert "M8.2 10.4C8.7 6.7 10.2 3.8 12 3.8s3.3 2.9 3.8 6.6" in pet_row_markup
    assert "M6.4 12.2 8.2 10.4l1.65 2.05 2-2.75" in pet_row_markup
    assert "M6.8 12.4c-1 4.35 1.35 7.25 5.2 7.25" in pet_row_markup
    assert "M9.7 13.4c.4-1.15 1.18-1.78 2.3-1.78" in pet_row_markup
    assert '<circle cx="7.2" cy="9.2" r="1.65"></circle>' not in pet_row_markup
    assert 'id="petCompanion"' in index_html
    assert 'class="pet-companion"' in index_html
    assert 'data-pet-id="auditbot"' in index_html
    assert 'id="petSticker"' in index_html
    assert 'class="pet-sprite"' in index_html
    assert 'background-image: url("static/pets/auditbot/spritesheet.webp")' in index_html
    assert 'localStorage.getItem("marvis_pet")' in index_html
    assert 'localStorage.getItem("marvis_pet_none_explicit") === "1"' in index_html
    assert 'localStorage.getItem("marvis_pet_position")' in index_html
    assert 'pet.classList.add("hidden");' in index_html
    assert 'Number.isFinite(storedPosition.workspaceOffsetLeft)' in index_html
    assert 'const minWorkspaceOffset = petCssPx("--pet-min-workspace-offset", padding);' in index_html
    assert 'pet.style.setProperty("--pet-offset-left", `${Math.round(offsetLeft)}px`);' in index_html
    assert 'pet.style.left = "";' in index_html
    assert 'id="petCompanionLabel"' not in index_html
    assert 'class="pet-companion-label"' not in index_html
    assert 'aria-live="polite"' in index_html

    for removed_pet in [
        "buou",
        "danhuang",
        "pixel-talisman-cat",
        "ragdoll-cat",
        "viola",
        "布偶猫",
        "Pixel Talisman Cat",
        "Viola",
        "Naitang",
        "/static/pets/buou.svg",
        "/static/pets/danhuang.svg",
        "/static/pets/pixel-talisman-cat/spritesheet.webp",
        "/static/pets/ragdoll-cat/spritesheet.webp",
        "/static/pets/viola/spritesheet.webp",
    ]:
        assert removed_pet not in settings_markup

    for removed_asset in [
        "/static/pets/buou.svg",
        "/static/pets/danhuang.svg",
        "/static/pets/pixel-talisman-cat/spritesheet.webp",
        "/static/pets/ragdoll-cat/spritesheet.webp",
        "/static/pets/viola/spritesheet.webp",
        "布偶猫",
        "Pixel Talisman Cat",
        "Viola",
        "Naitang",
    ]:
        assert removed_asset not in app_js

    assert 'export const defaultPetPreference = "auditbot";' in state_js
    assert "let petPreference = defaultPetPreference" in app_js
    assert 'naitang: {' in app_js
    assert 'name: "蛋黄"' in app_js
    assert 'kind: "spritesheet"' in app_js
    assert 'asset: "static/pets/naitang/spritesheet.webp"' in app_js
    assert 'xiaojiu: {' in app_js
    assert 'name: "小九"' in app_js
    assert 'asset: "static/pets/xiaojiu/spritesheet.webp?v=c078ec6f"' in app_js
    for pet_id, (display_name, pet_label) in expected_auditbot_pets.items():
        key = f'"{pet_id}": {{' if "-" in pet_id else f"{pet_id}: {{"
        assert key in app_js
        assert f'name: "{display_name}"' in app_js
        assert f'label: "{pet_label}"' in app_js
        assert f'asset: "static/pets/{pet_id}/spritesheet.webp"' in app_js
    assert 'pet-sprite' in app_js
    assert "sprite.style.backgroundImage" in app_js
    assert "petCompanionLabel" not in app_js
    assert "label.textContent" not in app_js
    pet_definitions = app_js[app_js.index("const petDefinitions") : app_js.index("const legacyPetPreferences")]
    assert "svg:" not in pet_definitions
    assert "ragdoll-cat" not in pet_definitions
    assert "sticker.innerHTML = definition.svg" not in app_js
    assert "document.createElement(\"img\")" in app_js
    assert "image.className = \"pet-image\"" in app_js
    assert "function restorePetPreference" in app_js
    assert "function applyPetPreference" in app_js
    assert "function renderPetState" in app_js
    assert "marvis_pet" in app_js
    assert '$("settingsPetSelect").value = petPreference' in app_js

    assert ".pet-companion {" in styles_css
    pet_rule = _css_rule(styles_css, ".pet-companion")
    assert "left: calc(var(--sidebar-width) + var(--pet-offset-left, var(--pet-default-workspace-offset)))" in pet_rule
    assert "right: auto" in pet_rule
    assert "bottom: 28px" in pet_rule
    pet_follow_rule = _css_rule(styles_css, "body.anim-ready:not(.is-resizing) .pet-companion:not(.dragging)")
    assert "transition: left 300ms cubic-bezier(0.4, 0, 0.2, 1)" in pet_follow_rule
    pet_dragging_rule = _css_rule(styles_css, ".pet-companion.dragging")
    assert "transition: none" in pet_dragging_rule
    mobile_start = styles_css.index("@media (max-width: 860px)")
    mobile_pet_start = styles_css.index(".pet-companion {", mobile_start)
    mobile_pet_end = styles_css.index("}", mobile_pet_start)
    mobile_pet_rule = styles_css[mobile_pet_start:mobile_pet_end]
    assert "left: 16px" in mobile_pet_rule
    assert "right: auto" in mobile_pet_rule
    assert "bottom: 16px" in mobile_pet_rule
    assert ".pet-image" in styles_css
    assert ".pet-sprite" in styles_css
    assert "--pet-sheet-y" in styles_css
    assert "--pet-frame-count" in styles_css
    assert ".pet-companion-label" not in styles_css
    assert '[data-pet-id="pixel-talisman-cat"]' not in styles_css
    assert ".pet-svg" not in styles_css
    assert ".pet-buou" not in styles_css
    assert ".pet-danhuang" not in styles_css
    assert "@keyframes pet-float" not in styles_css
    assert "@keyframes pet-soft-breathe" not in styles_css
    assert "@keyframes pet-playful-bounce" in styles_css
    assert '[data-pet-mood="success"]' in styles_css
    assert '[data-pet-mood="failed"]' in styles_css
    assert '[data-pet-mood="running"]' in styles_css
    assert '[data-pet-mood="complete"]' in styles_css
    assert "@media (prefers-reduced-motion: reduce)" in styles_css


def test_pet_preference_restores_legacy_local_storage_ids():
    app_js = _read_static("app.js")

    assert "const legacyPetPreferences" in app_js
    assert 'danhuang: "naitang"' in app_js
    assert '"ragdoll-cat": "xiaojiu"' in app_js
    assert 'buou: "xiaojiu"' in app_js
    assert 'buou: "ragdoll-cat"' not in app_js
    assert "function normalizePetPreference" in app_js
    assert "const normalized = normalizePetPreference(value);" in app_js
    assert 'petPreference = normalized;' in app_js

    preference_start = app_js.index("const petDefinitions")
    preference_end = app_js.index("executionEnvironmentSettings =", preference_start)
    normalize_start = app_js.index("function normalizePetPreference")
    normalize_end = app_js.index("function persistPetPreference", normalize_start)
    script = "\n".join(
        [
            'const defaultPetPreference = "auditbot";',
            app_js[preference_start:preference_end],
            app_js[normalize_start:normalize_end],
            "const values = ['danhuang', 'buou', 'ragdoll-cat', 'unknown', 'none'].map(normalizePetPreference);",
            "process.stdout.write(JSON.stringify(values));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == ["naitang", "xiaojiu", "xiaojiu", "auditbot", "none"]

    restore_start = app_js.index("function restorePetPreference")
    restore_end = app_js.index("function applyPetPosition", restore_start)
    restore_renderer = app_js[restore_start:restore_end]
    assert "const stored = localStorage.getItem(\"marvis_pet\");" in restore_renderer
    assert "const normalized = normalizePetPreference(stored);" in restore_renderer
    assert "applyPetPreference(normalized, { persist: normalized !== stored });" in restore_renderer


def test_pet_preference_defaults_visible_and_preserves_explicit_hide():
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")

    assert 'export const defaultPetPreference = "auditbot";' in state_js
    assert 'export const explicitPetNoneStorageKey = "marvis_pet_none_explicit";' in state_js
    assert "function persistPetPreference" in app_js
    assert 'if (value === "none" && explicitNone) {' in app_js
    assert "localStorage.setItem(explicitPetNoneStorageKey, \"1\");" in app_js
    assert "localStorage.removeItem(explicitPetNoneStorageKey);" in app_js

    restore_start = app_js.index("function restorePetPreference")
    restore_end = app_js.index("function applyPetPosition", restore_start)
    restore_renderer = app_js[restore_start:restore_end]
    assert 'const explicitNone = localStorage.getItem(explicitPetNoneStorageKey) === "1";' in restore_renderer
    assert 'if (!stored || (stored === "none" && !explicitNone)) {' in restore_renderer
    assert 'applyPetPreference(defaultPetPreference, { persist: stored === "none" });' in restore_renderer
    assert 'if (stored === "none") {' in restore_renderer
    assert 'applyPetPreference("none", { persist: false });' in restore_renderer

    settings_start = app_js.index("function handleSettingsMenuChange")
    settings_end = app_js.index("async function loadExecutionEnvironmentSettings", settings_start)
    settings_renderer = app_js[settings_start:settings_end]
    assert 'applyPetPreference(target.value, { explicit: true });' in settings_renderer


def test_task_search_controller_toggles_search_state_and_resets_query():
    module_url = (STATIC_DIR / "js" / "task-search.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
import {{ createTaskSearchController }} from {json.dumps(module_url)};

const bodyClasses = new Set();
const elements = {{
  taskSearchToggle: {{
    attrs: {{}},
    focused: false,
    setAttribute(name, value) {{ this.attrs[name] = value; }},
    focus() {{ this.focused = true; }},
  }},
  taskSearchInput: {{
    value: "risk",
    focused: false,
    selected: false,
    focus() {{ this.focused = true; }},
    select() {{ this.selected = true; }},
  }},
}};
let query = "risk";
let renderCount = 0;
let frameCount = 0;
const controller = createTaskSearchController({{
  getElementById: (id) => elements[id],
  documentRef: {{
    body: {{
      classList: {{
        add: (name) => bodyClasses.add(name),
        remove: (name) => bodyClasses.delete(name),
        contains: (name) => bodyClasses.has(name),
      }},
    }},
  }},
  windowRef: {{
    requestAnimationFrame(callback) {{
      frameCount += 1;
      callback();
    }},
  }},
  getQuery: () => query,
  setQuery: (value) => {{ query = value; }},
  renderTaskList: () => {{ renderCount += 1; }},
}});

assert.equal(controller.isActive(), false);
controller.openTaskSearch();
assert.equal(controller.isActive(), true);
assert.equal(bodyClasses.has("search-active"), true);
assert.equal(elements.taskSearchToggle.attrs["aria-expanded"], "true");
assert.equal(elements.taskSearchInput.focused, true);
assert.equal(elements.taskSearchInput.selected, true);
assert.equal(frameCount, 1);

controller.closeTaskSearch({{ focusToggle: true }});
assert.equal(controller.isActive(), false);
assert.equal(bodyClasses.has("search-active"), false);
assert.equal(elements.taskSearchToggle.attrs["aria-expanded"], "false");
assert.equal(elements.taskSearchToggle.focused, true);
assert.equal(elements.taskSearchInput.value, "");
assert.equal(query, "");
assert.equal(renderCount, 1);

controller.toggleTaskSearch();
assert.equal(controller.isActive(), true);
controller.toggleTaskSearch();
assert.equal(controller.isActive(), false);
"""
    subprocess.run(["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True)


def test_pet_position_restore_clamps_stale_coordinates_to_viewport():
    app_js = _read_static("app.js")

    assert "function clampPetPosition" in app_js
    assert "function ensurePetWithinViewport" in app_js
    assert "ensurePetWithinViewport({ persist });" in app_js
    assert "function petCssPx" in app_js
    assert "function petIsPinnedToWorkspaceLeftEdge" in app_js
    assert "function pinPetToWorkspaceLeftEdge" in app_js
    sidebar_renderer = _slice_function(app_js, "function applySidebarCollapsed")
    assert "const shouldKeepPetOnLeftEdge = petIsPinnedToWorkspaceLeftEdge();" in sidebar_renderer
    assert "pinPetToWorkspaceLeftEdge({ persist: true });" in sidebar_renderer

    restore_start = app_js.index("function restorePetPosition")
    restore_end = app_js.index("function petDragBounds", restore_start)
    restore_renderer = app_js[restore_start:restore_end]
    assert "Number.isFinite(stored.workspaceOffsetLeft)" in restore_renderer
    assert "workspace.left + stored.workspaceOffsetLeft" in restore_renderer
    assert "const next = clampPetPosition(storedLeft, stored.top);" in restore_renderer
    assert "applyPetPosition(next.left, next.top);" in restore_renderer
    assert "!Number.isFinite(stored.workspaceOffsetLeft)" in restore_renderer
    assert "savePetPosition(next.left, next.top);" in restore_renderer

    apply_start = app_js.index("function applyPetPosition")
    apply_end = app_js.index("function savePetPosition", apply_start)
    apply_renderer = app_js[apply_start:apply_end]
    assert 'pet.style.setProperty("--pet-offset-left", `${Math.round(offsetLeft)}px`);' in apply_renderer
    assert 'pet.style.left = "";' in apply_renderer

    save_start = app_js.index("function savePetPosition")
    save_end = app_js.index("function restorePetPosition", save_start)
    save_renderer = app_js[save_start:save_end]
    assert "payload.workspaceOffsetLeft = left - workspace.left;" in save_renderer

    bounds_start = app_js.index("function petDragBounds")
    bounds_end = app_js.index("function clampPetPosition", bounds_start)
    bounds_renderer = app_js[bounds_start:bounds_end]
    assert 'petCssPx("--pet-min-workspace-offset", padding)' in bounds_renderer
    assert "workspace ? workspace.left + minWorkspaceOffset : minWorkspaceOffset" in bounds_renderer

    pinned_start = app_js.index("function petIsPinnedToWorkspaceLeftEdge")
    pinned_end = app_js.index("function pinPetToWorkspaceLeftEdge", pinned_start)
    pinned_renderer = app_js[pinned_start:pinned_end]
    assert "Math.abs(offset - minWorkspaceOffset) <= 2" in pinned_renderer

    pin_start = app_js.index("function pinPetToWorkspaceLeftEdge")
    pin_end = app_js.index("function clampPetPosition", pin_start)
    pin_renderer = app_js[pin_start:pin_end]
    assert "workspace.left + minWorkspaceOffset" in pin_renderer
    assert "applyPetPosition(next.left, next.top);" in pin_renderer
    assert "if (persist) savePetPosition(next.left, next.top);" in pin_renderer

    drag_start = app_js.index("function startPetDrag")
    drag_end = app_js.index("function renderSettingsState", drag_start)
    drag_renderer = app_js[drag_start:drag_end]
    assert "const next = clampPetPosition(" in drag_renderer
    assert "applyPetPosition(next.left, next.top);" in drag_renderer


def test_only_selected_pet_assets_are_bundled():
    pets_dir = STATIC_DIR / "pets"
    expected_pets = {
        "naitang": "蛋黄",
        "xiaojiu": "小九",
        "auditbot": "MARVIS",
        "auditbot-pro": "MARVIS Pro",
        "auditbot-poly": "MARVIS Poly",
        "auditbot-ink": "MARVIS Ink",
        "auditbot-clay": "MARVIS Clay",
        "auditbot-comic": "MARVIS Comic",
        "auditbot-pixel": "MARVIS Pixel",
    }
    for pet_id, display_name in expected_pets.items():
        assert (pets_dir / pet_id / "pet.json").exists()
        assert (pets_dir / pet_id / "spritesheet.webp").exists()
        assert f'"displayName": "{display_name}"' in (pets_dir / pet_id / "pet.json").read_text(encoding="utf-8")
    assert not (pets_dir / "ragdoll-cat").exists()
    bundled_files = sorted(path.relative_to(pets_dir).as_posix() for path in pets_dir.rglob("*") if path.is_file())
    assert bundled_files == sorted(
        [f"{pet_id}/pet.json" for pet_id in expected_pets]
        + [f"{pet_id}/spritesheet.webp" for pet_id in expected_pets]
    )


def test_naitang_uses_pet_atlas_rows_and_drag_directions():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")
    pyproject = (STATIC_DIR.parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    assert 'static/pets/naitang/*' in pyproject
    assert 'static/pets/xiaojiu/*' in pyproject
    for pet_id in [
        "auditbot",
        "auditbot-pro",
        "auditbot-poly",
        "auditbot-ink",
        "auditbot-clay",
        "auditbot-comic",
        "auditbot-pixel",
    ]:
        assert f"static/pets/{pet_id}/*" in pyproject
    assert 'static/pets/ragdoll-cat/*' not in pyproject

    assert 'return "success";' in app_js
    assert 'return "failed";' in app_js
    assert 'return "running";' in app_js
    assert 'return "complete";' in app_js
    assert 'return "review";' in app_js
    assert 'pet.dataset.petMood = next.left >= current.left ? "running-right" : "running-left";' in app_js
    assert "renderPetState();" in app_js[app_js.index("function startPetDrag") : app_js.index("function renderSettingsState")]

    expected_rows = {
        'data-pet-mood="idle"': ("0%", "6", "85.7143%"),
        'data-pet-mood="running-right"': ("12.5%", "8", "114.2857%"),
        'data-pet-mood="running-left"': ("25%", "8", "114.2857%"),
        'data-pet-mood="complete"': ("37.5%", "4", "57.1429%"),
        'data-pet-mood="success"': ("50%", "5", "71.4286%"),
        'data-pet-mood="failed"': ("62.5%", "8", "114.2857%"),
        'data-pet-mood="review"': ("100%", "6", "85.7143%"),
        'data-pet-mood="running"': ("87.5%", "6", "85.7143%"),
    }
    for selector, (row, frames, x_end) in expected_rows.items():
        start = styles_css.index(f'[{selector}] .pet-sprite')
        end = styles_css.index("}", start)
        rule = styles_css[start:end]
        assert f"--pet-sheet-y: {row}" in rule
        assert f"--pet-frame-count: {frames}" in rule
        assert f"--pet-sheet-x-end: {x_end}" in rule

    assert "animation: pet-sprite-frames" in styles_css
    assert "steps(var(--pet-frame-count))" in styles_css
    assert "var(--pet-sheet-x-end) var(--pet-sheet-y)" in styles_css
    assert "to { background-position: 100% var(--pet-sheet-y); }" not in styles_css


def test_naitang_sprite_animation_uses_slower_frame_timing():
    styles_css = _read_static("styles.css")

    def css_rule(selector: str) -> str:
        start = styles_css.index(selector)
        end = styles_css.index("}", start)
        return styles_css[start:end]

    sprite_rule = css_rule(".pet-sprite {")
    assert "animation: pet-sprite-frames 5s steps(var(--pet-frame-count)) infinite;" in sprite_rule

    expected_durations = {
        '[data-pet-mood="idle"] .pet-sprite': "5s",
        '[data-pet-mood="running"] .pet-sprite': "4.2s",
        '[data-pet-mood="running-right"] .pet-sprite': "4.4s",
        '[data-pet-mood="running-left"] .pet-sprite': "4.4s",
        '[data-pet-mood="complete"] .pet-sprite': "5s",
        '[data-pet-mood="success"] .pet-sprite': "5.5s",
        '[data-pet-mood="failed"] .pet-sprite': "7.5s",
        '[data-pet-mood="waiting"] .pet-sprite': "5s",
        '[data-pet-mood="review"] .pet-sprite': "5s",
    }
    for selector, duration in expected_durations.items():
        assert f"animation-duration: {duration}" in css_rule(selector)


def test_pet_companion_does_not_auto_float_vertically():
    styles_css = _read_static("styles.css")

    sticker_start = styles_css.index(".pet-sticker {")
    sticker_end = styles_css.index("}", sticker_start)
    sticker_rule = styles_css[sticker_start:sticker_end]
    assert "animation: none" in sticker_rule
    assert "pet-float" not in sticker_rule
    assert "@keyframes pet-float" not in styles_css
    assert "@keyframes pet-soft-breathe" not in styles_css

    pet_block_start = styles_css.index(".pet-companion {")
    pet_block_end = styles_css.index(".resize-handle", pet_block_start)
    pet_block = styles_css[pet_block_start:pet_block_end]
    assert "translateY" not in pet_block

    running_start = styles_css.index('[data-pet-mood="running"] .pet-image {')
    running_end = styles_css.index("}", running_start)
    running_rule = styles_css[running_start:running_end]
    assert '[data-pet-mood="running"] .pet-sticker' not in running_rule
    assert "pet-playful-bounce" in running_rule


def test_pet_reaction_moods_return_to_idle_after_feedback_window():
    app_js = _read_static("app.js")

    assert "const PET_REACTION_DURATION_MS = 6500;" in app_js
    assert 'const petReactionMoods = new Set(["success", "failed", "complete", "review"]);' in app_js
    assert "let petReactionMood = null;" in app_js
    assert "let petReactionKey = \"\";" in app_js
    assert "let petReactionTimer = null;" in app_js
    assert "function basePetMoodFromTask()" in app_js
    assert "function petReactionKeyForMood(mood)" in app_js
    assert "function schedulePetReactionReset(key)" in app_js
    assert "clearTimeout(petReactionTimer);" in app_js
    assert "petReactionMood = null;" in app_js
    assert 'return petReactionMood || "idle";' in app_js
    assert "petReactionMoods.has(mood)" in app_js
    assert "task?.updated_at" in app_js


def test_pet_companion_is_draggable_and_reacts_to_task_status():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")

    assert "function petMoodFromTask" in app_js
    assert "function startPetDrag" in app_js
    assert "function restorePetPosition" in app_js
    assert "function savePetPosition" in app_js
    assert "marvis_pet_position" in app_js
    assert 'selectedTask?.status || ""' in app_js
    assert 'if (selectedTaskIsBusy()) return "running";' in app_js
    assert 'if (status === "succeeded") return "success";' in app_js
    assert 'if (status === "failed") return "failed";' in app_js
    assert 'if (status === "review_required") return "review";' in app_js
    assert '["running", "computing_metrics"].includes(status)' in app_js
    assert '["scanned", "executed", "writing_artifacts"].includes(status)' in app_js
    assert 'target.id === "settingsPetSelect"' in app_js
    assert 'pet.addEventListener("pointerdown", startPetDrag)' in app_js
    assert 'window.addEventListener("pointermove", onPointerMove)' in app_js
    assert 'window.addEventListener("pointerup", onPointerUp)' in app_js
    assert "clamp(" in app_js
    assert "renderPetState();" in app_js

    for removed_element_id in [
        "executionModeJupyterKernel",
        "executionModeCondaEnv",
        "executionModePythonExecutable",
        "kernelName",
        "condaEnvName",
        "pythonExecutable",
    ]:
        assert f'id="{removed_element_id}"' not in index_html


def test_execution_environment_panel_is_in_system_settings():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")
    index_html = _read_static("index.html")

    assert "executionEnvironmentSettingsLabel" in app_js
    assert "renderExecutionEnvironmentSummary" in app_js
    assert "settingsExecutionEnvironmentValue" not in app_js
    assert 'data-governance-nav="execution-environment"' in index_html
    assert 'data-governance-panel-content="execution-environment"' in index_html

    section_start = styles_css.index(".environment-dialog .execution-environment-section {")
    section_end = styles_css.index("}", section_start)
    section_rule = styles_css[section_start:section_end]
    assert "grid-template-columns: 1fr" in section_rule
    assert ".settings-panel-form" in styles_css
    assert ".governance-panel-actions" in styles_css


def test_execution_environment_api_fields_are_wired():
    app_js = _read_static("app.js")

    assert "/api/settings/execution-environment/options" in app_js
    assert "loadExecutionEnvironmentSettings" in app_js
    assert "saveExecutionEnvironmentSettings" in app_js
    assert "renderExecutionEnvironmentOptions" in app_js

    for field in [
        "execution_mode",
        "jupyter_kernel",
        "conda_env",
        "python_executable",
        "kernel_name",
        "conda_env_name",
        "python_executable",
    ]:
        assert field in app_js


def test_realtime_panel_keeps_only_reproducibility_evidence_in_center():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")

    assert 'id="reproducibilitySummary"' in index_html
    assert 'id="notebookSummary"' not in index_html
    assert 'id="notebookStepsSummary"' not in index_html
    assert 'id="contractSummary"' not in index_html

    assert "api/tasks/${taskId}/evidence" in app_js
    assert "renderEvidence" in app_js
    assert "notebook_steps" in app_js
    assert "reproducibility" in app_js
    assert "renderNotebookSteps(result.notebook_steps || [], result.notebook_cells || notebookCells)" in app_js
    assert "暂无分数一致性证据，运行完建模代码后展示结果" in index_html
    assert "暂无分数一致性证据，运行完建模代码后展示结果" in app_js
    assert "暂无 Notebook 契约证据" not in index_html
    assert "还没运行验证。扫描材料后运行当前任务验证。" not in index_html


def test_reproducibility_panel_renders_score_rows_and_diff_visuals():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    renderer_start = app_js.index("function renderReproducibilityEvidence")
    renderer_end = app_js.index("function renderEvidence", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert "const rows = Array.isArray(reproducibility?.rows) ? reproducibility.rows : [];" in renderer
    assert "score-compare-list" in renderer
    assert "score-diff-bar" in renderer
    assert "score_code_model" in renderer
    assert "score_submitted_pmml" in renderer
    assert "abs_diff" in renderer
    assert "const rowLimit = 10;" in renderer
    assert '"<strong>分数一致性</strong>"' not in renderer
    assert ".score-compare-list" in styles_css
    assert ".score-diff-bar" in styles_css
    assert "border-color: var(--danger-border)" in _css_rule(
        styles_css, ".result-summary.error"
    )
    assert "border-color: var(--danger-border)" in _css_rule(
        styles_css, ".score-compare-row.mismatched"
    )


def test_reproducibility_summary_omits_six_decimal_match_count_and_keeps_status_tone():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    renderer_start = app_js.index("function renderReproducibilityEvidence")
    renderer_end = app_js.index("function renderEvidence", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert "<span>6位小数一致条数</span>" not in renderer
    assert "match_count: summary.match_count" not in app_js
    assert "<span>6位小数不一致条数</span>" in renderer
    assert "reproducibilityStatusClass(summary.status)" in renderer
    assert "function reproducibilityStatusClass" in app_js
    assert ".summary-item.repro-status-pass" in styles_css
    assert ".summary-item.repro-status-fail" in styles_css


def test_reproducibility_panel_renders_precision_consistency_chart():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    renderer_start = app_js.index("function renderReproducibilityEvidence")
    renderer_end = app_js.index("function renderEvidence", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert "function buildPrecisionConsistencyBars" in app_js
    assert "function renderPrecisionConsistencyChart" in app_js
    assert "for (let decimals = 1; decimals <= 6; decimals += 1)" in app_js
    assert "roundedScoresMatch(row.score_code_model, row.score_submitted_pmml, decimals)" in app_js
    assert "renderPrecisionConsistencyChart(rows, {" in renderer
    assert ".score-precision-chart" in styles_css
    assert ".score-precision-bars" in styles_css
    assert ".score-precision-bar" in styles_css
    assert '.score-precision-chart[data-animation="none"] .score-precision-bar' in styles_css


# Reproducibility chart structural / animation behavior is now covered by
# the renderSignatures-based tests further down in this file:
#   - test_reproducibility_guard_lives_in_render_signatures
#   - test_reproducibility_animation_replays_only_on_first_render_per_task
#   - test_reproducibility_render_skips_replay_and_disables_animation_on_rebuild
# Those replace an earlier suite that grepped for the now-removed
# element.dataset.reproducibility* fields and that recreated only narrow
# slices of app.js. The new suite runs the real renderReproducibilityEvidence
# under stubbed browser globals, so it catches behavioral regressions, not
# just text-pattern drift.


def test_reproducibility_panel_formats_null_scores_as_missing_values():
    app_js = _read_static("app.js")

    formatter_start = app_js.index("function formatScoreValue")
    formatter_end = app_js.index("function reproducibilityStatusLabel", formatter_start)
    formatter = app_js[formatter_start:formatter_end]
    renderer_start = app_js.index("function renderReproducibilityEvidence")
    renderer_end = app_js.index("function renderEvidence", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert "value === null || value === undefined || value === \"\"" in formatter
    assert "row.abs_diff === null || row.abs_diff === undefined" in renderer


def test_reproducibility_card_is_hidden_until_notebook_success():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")
    styles_css = _read_static("styles.css")

    assert 'id="notebookSection" class="progress-panel hidden"' in index_html
    assert ".progress-panel.hidden" in styles_css
    assert "function shouldShowReproducibilitySection" in app_js
    assert "export const notebookReproducibilityCompleteStatuses = new Set([" in state_js
    for status in ["executed", "computing_metrics", "writing_artifacts", "succeeded", "review_required"]:
        assert f'"{status}"' in state_js
    assert "notebookReproducibilityCompleteStatuses.has(task?.status || \"\")" in app_js
    assert '$("notebookSection")?.classList.toggle("hidden", !shouldShowReproducibilitySection())' in app_js


def test_metric_card_is_hidden_until_metric_validation_success():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")
    styles_css = _read_static("styles.css")

    assert 'id="metricSection" class="progress-panel hidden"' in index_html
    assert ".progress-panel.hidden" in styles_css
    assert "function shouldShowMetricSection" in app_js
    assert "export const metricOverviewCompleteStatuses = new Set([" in state_js
    for status in ["writing_artifacts", "succeeded"]:
        assert f'"{status}"' in state_js
    helper_start = state_js.index("export const metricOverviewCompleteStatuses = new Set([")
    helper_end = state_js.index("export const workflowSteps = [", helper_start)
    helper_block = state_js[helper_start:helper_end]
    assert '"computing_metrics"' not in helper_block
    assert '"executed"' not in helper_block
    assert "metricOverviewCompleteStatuses.has(task?.status || \"\")" in app_js
    assert '$("metricSection")?.classList.toggle("hidden", !shouldShowMetricSection())' in app_js


def test_metric_sparkline_html_requires_local_spec_marker():
    app_js = _read_static("app.js")
    renderer_start = app_js.index("function renderCellByKind")
    renderer_end = app_js.index("function renderMetricTable", renderer_start)
    renderer = app_js[renderer_start:renderer_end]
    trend_start = app_js.index("function renderTrendTable")
    trend_end = app_js.index("function renderSparklineSvg", trend_start)
    trend_renderer = app_js[trend_start:trend_end]

    assert 'kind === "trend-spark" && spec && spec.__localHtml === true' in renderer
    assert 'trendSpecs.splice(insertAt, 0, { kind: "trend-spark", __localHtml: true });' in trend_renderer


def test_metric_tooltip_uses_document_delegation_for_rebuilt_preview():
    app_js = _read_static("app.js")
    tooltip_start = app_js.index("function attachMetricTooltip")
    tooltip_end = app_js.index("function renderEnhancedTable", tooltip_start)
    tooltip = app_js[tooltip_start:tooltip_end]

    assert 'document.addEventListener("mouseover"' in tooltip
    assert 'event.target.closest("#metricPreview [data-tip]")' in tooltip
    assert 'rootEl.addEventListener("mouseover"' not in tooltip


def test_metric_kpi_footer_values_are_centered_in_each_column():
    styles_css = _read_static("styles.css")

    cell_start = styles_css.index(".kpi-card-footer-cell {")
    cell_end = styles_css.index("}", cell_start)
    cell_rule = styles_css[cell_start:cell_end]

    assert "align-items: center" in cell_rule
    assert "text-align: center" in cell_rule


def test_metric_tables_use_tabular_right_aligned_numeric_cells():
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")

    table_start = styles_css.index(".metric-table-section .metric-table {")
    table_end = styles_css.index("}", table_start)
    table_rule = styles_css[table_start:table_end]
    number_start = styles_css.index(".metric-table-section .metric-table td.cell-number {")
    number_end = styles_css.index("}", number_start)
    number_rule = styles_css[number_start:number_end]

    assert "font-variant-numeric: tabular-nums" in table_rule
    assert "text-align: right" in number_rule
    assert "font-variant-numeric: tabular-nums" in number_rule
    assert 'return { cls: "cell-number"' in app_js


def test_metric_overview_uses_semantic_visual_tokens():
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")
    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))
    metric_section = styles_css[styles_css.index("/* ====== Metric overview redesign ====== */") :]

    required_tokens = [
        "--report-tone-cool-blue",
        "--report-tone-cool-blue-soft",
        "--report-tone-warm-orange",
        "--report-tone-warm-orange-soft",
        "--report-tone-deep-purple",
        "--report-tone-deep-purple-soft",
        "--report-tone-warning-red",
        "--report-tone-warning-red-soft",
        "--report-tone-heatmap",
        "--report-tone-heatmap-soft",
        "--metric-border",
        "--metric-surface",
        "--metric-surface-soft",
        "--metric-control-surface",
        "--metric-text",
        "--metric-text-strong",
        "--metric-text-muted",
        "--metric-databar-accent",
        "--metric-psi-stable",
        "--metric-psi-warn",
        "--metric-psi-critical",
        "--chart-axis",
        "--chart-grid",
        "--chart-muted",
        "--chart-roc-tpr",
        "--chart-roc-baseline",
        "--chart-roc-ks",
    ]
    for token in required_tokens:
        assert token in root_vars
        assert token in dark_vars

    assert "--accent: var(--report-tone-cool-blue)" in _css_rule(styles_css, ".metric-table-section")
    assert "--accent-soft: var(--report-tone-warm-orange-soft)" in _css_rule(
        styles_css,
        '.metric-table-section[data-theme="warm-orange"]',
    )
    assert "background: var(--metric-control-surface)" in metric_section
    assert "stroke: var(--chart-roc-ks)" in metric_section
    for legacy_color in [
        "#3A6EA5",
        "#E4ECF6",
        "#0EA5E9",
        "#16A34A",
        "#DC2626",
        "#3B82F6",
        "#F3F4F6",
        "#E5E7EB",
        "#1F2937",
        "#6B7280",
    ]:
        assert legacy_color not in metric_section
    assert "#0EA5E9" not in app_js
    assert 'var(--metric-databar-accent)' in app_js


def test_metric_overview_dark_theme_keeps_hover_and_chart_text_readable():
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))

    for selector, accent_token, soft_token in [
        (
            'body[data-theme="dark"] .metric-table-section',
            "--report-tone-cool-blue",
            "--report-tone-cool-blue-soft",
        ),
        (
            'body[data-theme="dark"] .metric-table-section[data-theme="cool-blue"]',
            "--report-tone-cool-blue",
            "--report-tone-cool-blue-soft",
        ),
        (
            'body[data-theme="dark"] .metric-table-section[data-theme="warm-orange"]',
            "--report-tone-warm-orange",
            "--report-tone-warm-orange-soft",
        ),
        (
            'body[data-theme="dark"] .metric-table-section[data-theme="deep-purple"]',
            "--report-tone-deep-purple",
            "--report-tone-deep-purple-soft",
        ),
        (
            'body[data-theme="dark"] .metric-table-section[data-theme="warning-red"]',
            "--report-tone-warning-red",
            "--report-tone-warning-red-soft",
        ),
        (
            'body[data-theme="dark"] .metric-table-section[data-theme="heatmap"]',
            "--report-tone-heatmap",
            "--report-tone-heatmap-soft",
        ),
    ]:
        rule_vars = _css_vars(_css_rule(styles_css, selector))
        assert rule_vars["--accent"] == f"var({accent_token})"
        assert rule_vars["--accent-soft"] == f"var({soft_token})"
        assert _contrast_ratio(dark_vars[soft_token], dark_vars["--surface"]) >= 1.12

    hover_rule = _css_rule(
        styles_css,
        'body[data-theme="dark"] .metric-table.metric-table-hoverable tbody:has(tr:hover) tr:hover',
    )
    assert "background: color-mix(in srgb, var(--accent-soft) 76%, var(--surface))" in hover_rule
    assert "color: var(--text)" in hover_rule

    assert (
        'body[data-theme="dark"] .metric-table.metric-table-hoverable tbody:has(tr:hover) '
        "tr:hover :is(.databar-label, .period-text, .psi-value)"
    ) in styles_css
    assert 'body[data-theme="dark"] .roc-axis-label' in styles_css
    assert 'class="roc-axis-label"' in app_js
    assert 'fill="#6B7280"' not in app_js


def test_agent_progress_refreshes_metric_preview_before_streaming_analysis_messages():
    app_js = _read_static("app.js")
    poll_start = app_js.index("async function pollValidationProgress")
    poll_end = app_js.index("async function validateCurrentTask", poll_start)
    poll_body = app_js[poll_start:poll_end]

    assert "metricOverviewComplete(polledTask)" in poll_body
    assert "await loadReportFields(taskId);" in poll_body
    assert poll_body.index("await loadReportFields(taskId);") < poll_body.index("await loadAgentMessages(taskId);")


def test_evidence_restore_renders_persisted_scan_result():
    app_js = _read_static("app.js")
    renderer_start = app_js.index("function renderEvidence")
    renderer_end = app_js.index("async function loadTaskEvidence", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert "evidence.scan" in renderer
    assert "renderScanResult(evidence.scan, evidence.notebook_cells || [])" in renderer


def test_scan_result_renders_structured_preflight_checks():
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")
    renderer_start = app_js.index("function renderScanResult")
    renderer_end = app_js.index("function renderValidationResult", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert "result.checks" in renderer
    assert "preflight-check-list" in renderer
    assert "notebook_contract" in app_js
    assert "file-list" not in renderer
    assert "ambiguity-list" not in renderer
    for selector, token in [
        (".preflight-check-item.danger", "--danger-border"),
        (".preflight-check-item.warning", "--warning-border"),
        (".preflight-check-item.success", "--success-border"),
    ]:
        assert f"border-color: var({token})" in _css_rule(styles_css, selector)


def test_validate_action_polls_task_status_and_evidence_until_terminal():
    app_js = _read_static("app.js")

    assert "pollValidationProgress" in app_js
    assert "terminalTaskStatuses" in app_js
    assert "activeValidationStatuses" in app_js
    assert 'await pollValidationProgress(new Set(["executed", "failed", "scanned"]), taskId)' in app_js
    assert 'await pollValidationProgress(new Set(["executed", "writing_artifacts", "failed"]), taskId)' in app_js
    assert "await pollValidationProgress(terminalTaskStatuses, taskId)" in app_js
    assert "await loadTaskEvidence(taskId)" in app_js
    assert "验证进行中" in app_js


def test_validate_action_reloads_notebook_evidence_after_completion():
    app_js = _read_static("app.js")
    validate_start = app_js.index("async function validateCurrentTask")
    validate_end = app_js.index("async function cancelCurrentNotebook", validate_start)
    validate_renderer = app_js[validate_start:validate_end]

    assert "await loadReportFields(taskId);\n  await loadTaskEvidence(taskId);" in validate_renderer


def test_evidence_fetch_failure_preserves_completed_notebook_evidence():
    app_js = _read_static("app.js")
    loader_start = app_js.index("async function loadTaskEvidence")
    loader_end = app_js.index("function renderActionError", loader_start)
    loader = app_js[loader_start:loader_end]

    assert "notebookReproducibilityComplete(selectedTask)" in loader
    assert "resetEvidenceSummaries();" in loader


def test_notebook_step_renderer_does_not_cap_steps_at_eight():
    app_js = _read_static("app.js")
    renderer_start = app_js.index("function renderNotebookSteps")
    renderer_end = app_js.index("function renderReproducibilityEvidence", renderer_start)
    renderer = app_js[renderer_start:renderer_end]

    assert ".slice(0, 8)" not in renderer
    assert "notebookSteps.map" not in renderer
    assert "latestNotebookSteps = mergePendingSystemSteps(normalizeNotebookSteps(notebookSteps, notebookCells))" in renderer
    assert "renderWorkflowStepper()" in renderer
    assert "Notebook 步骤（共" not in app_js


def test_notebook_step_renderer_uses_latest_retried_system_cell_status():
    steps = [
        {
            "id": "system-metrics-prepare",
            "title": "指标数据准备",
            "status": "failed",
            "started_at": "2026-05-28T06:30:14+00:00",
            "ended_at": "2026-05-28T06:37:22+00:00",
            "elapsed_seconds": 428,
            "cell_count": 2,
            "cell_indexes": [40, 48],
            "source_previews": ["prepare_old()", "prepare_new()"],
            "system": True,
        },
        {
            "id": "system-metrics-score",
            "title": "RMC_SCORE_FN 全量打分",
            "status": "pending",
            "started_at": "2026-05-28T06:37:22+00:00",
            "ended_at": "2026-05-28T06:37:24+00:00",
            "elapsed_seconds": 2,
            "cell_count": 2,
            "cell_indexes": [41, 49],
            "source_previews": ["score_old()", "score_new()"],
            "system": True,
        },
    ]
    cells = [
        {"cell_index": 40, "step_id": "system-metrics-prepare", "status": "failed"},
        {
            "cell_index": 48,
            "step_id": "system-metrics-prepare",
            "status": "succeeded",
            "started_at": "2026-05-28T06:37:21+00:00",
            "ended_at": "2026-05-28T06:37:22+00:00",
        },
        {
            "cell_index": 49,
            "step_id": "system-metrics-score",
            "status": "succeeded",
            "started_at": "2026-05-28T06:37:22+00:00",
            "ended_at": "2026-05-28T06:37:24+00:00",
        },
    ]

    normalized = _normalized_notebook_steps_for(steps, cells)

    assert normalized[0]["status"] == "succeeded"
    assert normalized[0]["cell_indexes"] == [48]
    assert normalized[0]["source_previews"] == ["prepare_new()"]
    assert normalized[0]["started_at"] == "2026-05-28T06:37:21+00:00"
    assert normalized[1]["status"] == "succeeded"
    assert normalized[1]["cell_indexes"] == [49]


def test_failed_task_error_detail_moves_to_current_status_only():
    app_js = _read_static("app.js")
    workspace_view_js = _read_static("js/task-workspace-view.js")

    append_start = app_js.index("function appendTaskRow")
    append_end = app_js.index("function renderTaskSnapshot", append_start)
    append_renderer = app_js[append_start:append_end]
    assert "task.status_message" not in append_renderer

    snapshot_start = workspace_view_js.index("export function renderTaskSnapshot")
    snapshot_end = workspace_view_js.index("export function renderCurrentTaskWorkspace", snapshot_start)
    snapshot_renderer = workspace_view_js[snapshot_start:snapshot_end]
    assert "selectedTask.status_message" not in snapshot_renderer

    step_start = app_js.index("function renderWorkflowStepper")
    step_end = app_js.index("function formatDate", step_start)
    step_renderer = app_js[step_start:step_end]
    assert "selectedTask.status_message" not in step_renderer
    assert "step-error" not in step_renderer

    assert "function taskFailureActionStatusMessage" in app_js
    assert "function taskFailureActionStatusTitle" in app_js
    status_start = app_js.index("function taskFailureActionStatusMessage")
    status_end = app_js.index("function clearStatus", status_start)
    status_renderer = app_js[status_start:status_end]
    assert "task.status_message" in status_renderer
    assert 'const kind = task.status === "review_required" ? "success" : "error";' in status_renderer
    assert "setActionStatus(taskFailureActionStatusTitle(task), kind, message)" in status_renderer


def test_review_required_status_bar_uses_completed_green_copy_not_failure_detail():
    result = _task_action_status_for(
        {
            "status": "review_required",
            "status_message": "reproducibility failed; review required",
            "active_job_kind": None,
        }
    )

    assert result == {
        "title": "验证已完成，需复核报告。",
        "kind": "success",
        "detail": "全部流程已完成，请查看右侧报告并进行人工复核。",
    }
    rendered = json.dumps(result, ensure_ascii=False).lower()
    assert "failed" not in rendered
    assert "unresolved" not in rendered

    placeholder_result = _task_action_status_for(
        {
            "status": "review_required",
            "status_message": "report has unresolved placeholders",
            "active_job_kind": None,
        }
    )
    assert placeholder_result == result

    display = _task_display_status_for(
        {
            "status": "review_required",
            "status_message": "report has unresolved placeholders",
            "active_job_kind": None,
        },
        action_message="验证已完成，需复核报告。",
        action_kind="success",
    )
    assert display == {
        "rowLabel": "待复核",
        "rowTone": "success",
        "heroPill": {"label": "已完成", "tone": "ok"},
    }


def test_stopped_agent_task_status_copy_is_not_failure_or_busy():
    display = _task_display_status_for(
        {
            "status": "scanned",
            "status_message": "已停止当前动作",
            "active_job_kind": "agent",
            "stopped": True,
        }
    )
    result = _task_action_status_for(
        {
            "status": "scanned",
            "status_message": "已停止当前动作",
            "active_job_kind": "agent",
            "stopped": True,
        }
    )

    assert display == {
        "rowLabel": "停止",
        "rowTone": "",
        "heroPill": {"label": "停止", "tone": "neutral"},
    }
    assert result == {
        "title": "已停止当前动作。",
        "kind": "stopped",
        "detail": "已停止当前动作，请问有什么指示？",
    }


def test_stopped_agent_task_does_not_spin_running_substeps():
    task = {
        "status": "scanned",
        "status_message": "已停止当前动作",
        "active_job_kind": "agent",
        "stopped": True,
    }
    notebook_steps = [
        {"id": "system-repro-pmml", "status": "succeeded"},
        {"id": "system-repro-compare", "status": "running"},
    ]

    assert _workflow_step_statuses_for(task, notebook_steps) == [
        "succeeded",
        "pending",
        "pending",
        "pending",
    ]
    assert _notebook_step_tones_for(task, notebook_steps) == ["succeeded", "stopped"]


def test_stopped_step_checker_has_no_stop_square_mark():
    app_js = _read_static("app.js")
    start = app_js.index("function stepCheckerHtml")
    end = app_js.index("function notebookStepTone", start)
    source = app_js[start:end]
    stopped_start = source.index('if (state === "stopped")')
    stopped_end = source.index('if (state === "review")', stopped_start)
    stopped_branch = source[stopped_start:stopped_end]

    assert '<span class="check-icon stopped" aria-hidden="true"></span>' in stopped_branch
    assert "<svg" not in stopped_branch
    assert "<rect" not in stopped_branch


def test_current_status_error_detail_is_always_visible_and_turns_red_for_failures():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    assert 'id="actionErrorDetail"' in index_html
    assert 'class="action-error-detail"' in index_html
    assert "暂无报错" in index_html
    assert 'id="actionErrorDetail"\n                  class="action-error-detail"' in index_html
    assert 'role="alert"' not in index_html[index_html.index('id="actionErrorDetail"'):index_html.index('id="taskSnapshot"')]
    assert 'aria-live="assertive"' not in index_html[index_html.index('id="actionErrorDetail"'):index_html.index('id="taskSnapshot"')]

    status_start = app_js.index("function setActionErrorDetail")
    status_end = app_js.index("function taskFailureActionStatusMessage", status_start)
    status_renderer = app_js[status_start:status_end]
    assert "function setActionErrorDetail" in status_renderer
    assert 'detail.textContent = message || "";' in status_renderer
    assert 'detail.setAttribute("role", kind === "error" ? "alert" : "status");' in status_renderer
    assert 'detail.setAttribute("aria-live", kind === "error" ? "assertive" : "polite");' in status_renderer
    assert 'detail.className = `action-error-detail ${kind === "error" ? "error" : ""}`.trim();' in status_renderer
    assert "setActionErrorDetail(describeActionStatus(message, kind, detail), kind)" in status_renderer
    assert "actionErrorDetail" in status_renderer

    # failures are signalled by a red status pill, not a red box
    assert "function actionStatusPill" in app_js
    assert 'tone: "fail"' in app_js
    assert '? { label: "需复核", tone: "ok" }' in app_js
    assert '.task-pill.fail' in styles_css

    detail_start = styles_css.index(".action-error-detail {")
    detail_end = styles_css.index("}", detail_start)
    detail_rule = styles_css[detail_start:detail_end]
    assert "color: var(--text-secondary)" in detail_rule
    assert ".action-error-detail.error" in styles_css


def test_center_workspace_scroll_locks_status_card_and_lateral_overscroll():
    styles_css = _read_static("styles.css")
    index_html = _read_static("index.html")

    workspace_start = styles_css.index(".result-workspace {")
    workspace_end = styles_css.index("}", workspace_start)
    workspace_rule = styles_css[workspace_start:workspace_end]
    assert "overflow: hidden;" in workspace_rule
    assert "display: grid;" in workspace_rule
    assert "grid-template-rows: auto minmax(0, 1fr);" in workspace_rule
    assert "--workspace-head-space:" in workspace_rule

    scroll_start = styles_css.index(".result-scroll-content {")
    scroll_end = styles_css.index("}", scroll_start)
    scroll_rule = styles_css[scroll_start:scroll_end]
    assert "grid-row: 1 / -1;" in scroll_rule
    assert "position: relative;" in scroll_rule
    assert "z-index: 1;" in scroll_rule
    assert "height: 100%;" not in scroll_rule
    assert "padding-top: calc(var(--workspace-head-space) + 12px);" in scroll_rule
    assert "overflow-y: auto;" in scroll_rule
    assert "overflow-x: hidden;" in scroll_rule
    assert "scrollbar-width: none;" in scroll_rule
    assert "-ms-overflow-style: none;" in scroll_rule
    assert "scrollbar-gutter: auto;" in scroll_rule
    assert "scrollbar-gutter: stable;" not in scroll_rule
    assert "overscroll-behavior-y: none;" in scroll_rule
    assert "overscroll-behavior-x: none;" in scroll_rule
    target_offset_start = styles_css.index(".result-scroll-content > :is(")
    target_offset_end = styles_css.index("}", target_offset_start)
    target_offset_rule = styles_css[target_offset_start:target_offset_end]
    assert ".progress-panel" in target_offset_rule
    assert ".supporting-evidence" in target_offset_rule
    assert ".agent-conversation" in target_offset_rule
    assert "scroll-margin-top: calc(var(--workspace-head-space) + 12px);" in target_offset_rule
    assert ".result-scroll-content::-webkit-scrollbar" in styles_css
    webkit_scrollbar_start = styles_css.index(".result-scroll-content::-webkit-scrollbar {")
    webkit_scrollbar_end = styles_css.index("}", webkit_scrollbar_start)
    webkit_scrollbar_rule = styles_css[webkit_scrollbar_start:webkit_scrollbar_end]
    assert "display: none;" in webkit_scrollbar_rule

    head_start = styles_css.index(".workspace-head {")
    head_end = styles_css.index("}", head_start)
    head_rule = styles_css[head_start:head_end]
    assert "position: relative;" in head_rule
    assert "grid-column: 1;" in head_rule
    assert "grid-row: 1;" in head_rule
    assert "background: transparent;" in head_rule
    assert "padding: 0;" in head_rule
    assert "isolation: isolate;" in head_rule
    assert ".workspace-head::before" not in styles_css
    assert ".workspace-head::after" not in styles_css

    hero_start = styles_css.index(".task-hero {")
    hero_end = styles_css.index("}", hero_start)
    hero_rule = styles_css[hero_start:hero_end]
    assert "overflow: hidden;" in hero_rule
    assert "isolation: isolate;" in hero_rule
    assert "border: 1px solid color-mix(in srgb, var(--border) 54%, transparent);" in hero_rule
    assert "border: 1px solid transparent;" not in hero_rule
    assert "transform: translateZ(0);" in hero_rule
    assert "contain: paint;" in hero_rule
    assert "will-change: transform;" in hero_rule
    assert "backdrop-filter: blur(18px) saturate(1.55);" in hero_rule
    assert "background: linear-gradient" in hero_rule
    assert "transition: border-color 140ms ease;" in hero_rule
    assert "background 180ms" not in hero_rule
    assert "box-shadow 180ms" not in hero_rule
    assert "0 14px" not in hero_rule
    assert ".task-hero.is-glass-active" in styles_css
    assert ".task-hero.is-glass-active::after" in styles_css

    after_start = styles_css.index(".task-hero::after {")
    after_end = styles_css.index("}", after_start)
    after_rule = styles_css[after_start:after_end]
    assert "bottom: 0;" in after_rule
    assert "bottom: -18px;" not in after_rule
    assert "filter:" not in after_rule
    assert "rgba(0, 113, 227" not in after_rule
    assert "rgba(31, 122, 63" not in after_rule

    active_start = styles_css.index(".task-hero.is-glass-active {")
    active_end = styles_css.index("}", active_start)
    active_rule = styles_css[active_start:active_end]
    assert "border-color: color-mix(in srgb, var(--border) 62%, transparent);" in active_rule
    assert "border-color: transparent;" not in active_rule
    assert "var(--accent)" not in active_rule
    assert "rgba(0, 113, 227" not in active_rule
    assert "0 18px" not in active_rule
    assert "0 10px" not in active_rule

    dark_hero_start = styles_css.index('body[data-theme="dark"] .task-hero {')
    dark_hero_end = styles_css.index("}", dark_hero_start)
    dark_hero_rule = styles_css[dark_hero_start:dark_hero_end]
    assert "border-color: color-mix(in srgb, var(--border) 58%, transparent);" in dark_hero_rule
    assert "border-color: transparent;" not in dark_hero_rule
    assert "0 16px 42px" not in dark_hero_rule
    assert "rgba(0, 0, 0" not in dark_hero_rule
    assert "inset 0 1px 0 rgba(255, 255, 255, 0.08)" in dark_hero_rule

    assert 'id="resultScrollContent"' in index_html
    assert "static/styles.css?v=__MARVIS_STATIC_VERSION__" in index_html
    assert "static/css/welcome.css?v=__MARVIS_STATIC_VERSION__" in index_html
    assert "static/styles.css?v=20260613-task-entry-upload" not in index_html
    assert 'static/styles.css?v=20260613-task-entry"' not in index_html
    assert "static/styles.css?v=20260605-create-dialog-button-gap" not in index_html
    assert "static/styles.css?v=20260605-create-dialog-scroll" not in index_html
    assert "static/styles.css?v=20260603-sidebar-icon-controls" not in index_html
    assert "static/styles.css?v=20260603-run-mode-selected-glow" not in index_html
    assert "static/styles.css?v=20260603-validator-icon-16" not in index_html
    assert "static/styles.css?v=20260603-settings-no-focus-frame" not in index_html
    assert "static/styles.css?v=20260603-brand-icon-neutral-fill" not in index_html
    assert "static/styles.css?v=20260603-task-validator-icon" not in index_html
    assert "static/styles.css?v=20260603-run-mode-border" not in index_html
    assert "static/styles.css?v=20260603-scan-env-add-style" not in index_html
    assert "static/styles.css?v=20260603-brand-icon-buttons" not in index_html
    assert "static/styles.css?v=20260603-run-mode-glow" not in index_html
    assert "static/styles.css?v=20260603-dark-scrollbar" not in index_html
    assert "static/styles.css?v=20260603-task-options" not in index_html
    assert "static/styles.css?v=20260603-neutral-options" not in index_html
    assert "static/styles.css?v=20260603-dark-masks" not in index_html
    assert "static/app.js?v=__MARVIS_STATIC_VERSION__" in index_html
    assert "static/app.js?v=20260613-task-entry-welcome" not in index_html
    assert "static/app.js?v=20260613-review-fixes" not in index_html
    assert "static/app.js?v=20260613-task-entry-upload" not in index_html
    assert 'static/app.js?v=20260613-task-entry"' not in index_html
    assert "static/app.js?v=20260605-create-task-error" not in index_html
    assert "static/app.js?v=20260603-zero-rail-collapse" not in index_html
    assert "static/app.js?v=20260603-task-validator-icon" not in index_html
    assert "static/app.js?v=20260603-field-focus-ring" not in index_html
    assert "static/app.js?v=20260603-dark-masks" not in index_html


def test_status_card_glass_glow_tracks_inner_scroll_position():
    app_js = _read_static("app.js")

    assert "function updateTaskHeroGlassState" in app_js
    assert "function scheduleTaskHeroGlassState" in app_js
    assert "function syncTaskHeroGlassLayout" in app_js
    assert "let taskHeroGlassActive = null;" in app_js
    assert "let taskHeroCanScroll = false;" in app_js
    assert "function setTaskHeroGlassActive" in app_js
    assert "if (taskHeroGlassActive === glassActive) return;" in app_js
    assert "updateTaskHeroGlassState({ measureScroll: true });" in app_js
    state_start = app_js.index("function updateTaskHeroGlassState")
    state_end = app_js.index("function scheduleTaskHeroGlassState", state_start)
    state_body = app_js[state_start:state_end]
    assert "if (measureScroll)" in state_body
    assert "taskHeroCanScroll = scrollContent.scrollHeight > scrollContent.clientHeight + 1;" in state_body
    assert "const glassActive = taskHeroCanScroll && scrollContent.scrollTop > 6;" in state_body
    assert 'workspace.style.setProperty("--workspace-head-space"' in app_js
    assert 'hero.classList.toggle("is-glass-active", glassActive)' in app_js
    assert 'workspace.classList.toggle("is-glass-active", glassActive)' in app_js
    assert "function handleResultScroll" in app_js
    result_scroll_start = app_js.index("function handleResultScroll")
    result_scroll_end = app_js.index("function syncTaskHeroGlassLayout", result_scroll_start)
    result_scroll_body = app_js[result_scroll_start:result_scroll_end]
    assert "if (pendingResultScrollRestoreTaskId !== selectedTaskId)" in result_scroll_body
    assert "rememberResultScrollPosition();" in result_scroll_body
    assert "scheduleTaskHeroGlassState();" in result_scroll_body
    assert '$("resultScrollContent").addEventListener("scroll", handleResultScroll, { passive: true });' in app_js
    assert 'window.addEventListener("resize", syncTaskHeroGlassLayout);' in app_js
    assert "requestAnimationFrame(syncTaskHeroGlassLayout)" in app_js


def test_validation_failure_writes_error_detail_to_global_action_status():
    app_js = _read_static("app.js")

    validate_start = app_js.index("async function validateCurrentTask")
    validate_end = app_js.index("async function loadReportFields", validate_start)
    validate_renderer = app_js[validate_start:validate_end]
    assert "setTaskFailureActionStatus(selectedTask || finalTask)" in validate_renderer

    poll_start = app_js.index("async function pollValidationProgress")
    poll_end = app_js.index("async function validateCurrentTask", poll_start)
    poll_renderer = app_js[poll_start:poll_end]
    assert "setTaskFailureActionStatus(polledTask)" in poll_renderer
    assert "setActionStatus(\"\")" not in poll_renderer


def test_agent_mode_creation_and_stepper_hide_manual_buttons():
    app_js = _read_static("app.js")
    create_dialog_js = _read_static("js/create-task-dialog.js")
    driver_confirm_js = _read_static("js/v2/driver_gate_confirm.js")

    create_start = app_js.index("async function createTask")
    create_end = app_js.index("async function refreshTasks", create_start)
    create_body = app_js[create_start:create_end]
    assert "Agent 模式当前暂不支持创建任务" not in create_body
    assert "const task = await createTaskDialog.createTask();" in create_body
    assert "run_mode: selectedRunMode" in create_dialog_js

    assert "function selectedTaskIsAgentMode" in app_js
    assert "renderDriverGateButton(message, { isAgentMode: selectedTaskIsAgentMode })" in app_js
    assert 'message?.metadata?.kind !== "gate" || isAgentMode' in driver_confirm_js
    assert "startAgentValidation" in app_js


def test_agent_mode_creation_routes_non_validation_tasks_to_conversation_composer():
    app_js = _read_static("app.js")

    create_scan_start = app_js.index("async function createTaskAndScan")
    create_scan_end = app_js.index("async function pollValidationProgress", create_scan_start)
    create_scan_body = app_js[create_scan_start:create_scan_end]
    agent_branch_start = create_scan_body.index('if (task.run_mode === "agent")')
    agent_branch_end = create_scan_body.index('setBusy(null, "", null);', agent_branch_start)
    agent_branch = create_scan_body[agent_branch_start:agent_branch_end]

    assert "const activeDialogTaskType = createTaskDialog.activeTaskType();" in agent_branch
    assert "const definition = taskTypeDefinition(task.task_type || activeDialogTaskType);" in agent_branch
    assert 'const isValidationTask = (task.task_type || activeDialogTaskType || defaultTaskType) === "validation";' in agent_branch
    # Non-validation agent tasks route to the inline conversation composer:
    # createTask() already seeded it via prefillAgentTaskInstruction, so the
    # branch only focuses the composer (the V2 plan dialog is retired).
    assert "if (!isValidationTask && definition.initialGoal)" in agent_branch
    assert '$("agentComposerInput")?.focus?.();' in agent_branch
    assert "已填入建议目标，确认后发送即可。" in agent_branch
    assert 'setActionStatus("Agent 任务已创建，等待你的下一条指令。", "success");' in agent_branch
    assert "openV2WorkspaceWithGoal(" not in agent_branch
    assert "已打开 V2 Workflow 计划面板" not in agent_branch
    assert "await dispatchAgentValidation(taskId);" not in agent_branch
    assert "await scanCurrentTask();" not in agent_branch
    assert "正在自动识别材料" not in agent_branch
    assert "开始验证" not in agent_branch
    assert "const taskId = task.id || selectedTaskId;" in agent_branch

    assert "async function dispatchAgentValidation" in app_js
    assert 'api(`/api/tasks/${normalizedTaskId}/agent/start`' in app_js


def test_welcome_task_cards_share_the_same_visual_treatment():
    index_html = _read_static("index.html")
    welcome_css = _read_static("css/welcome.css")

    cards_start = index_html.index('id="welcomeTaskCards"')
    cards_end = index_html.index("</div>", cards_start)
    cards_markup = index_html[cards_start:cards_end]

    assert 'class="welcome-task-card available primary-task"' not in cards_markup
    assert ".welcome-task-card.primary-task" not in welcome_css
    for task_kind in [
        "feature_analysis",
        "data_join",
        "vintage",
        "modeling",
        "validation",
        "strategy",
    ]:
        task_index = cards_markup.index(f'data-task-kind="{task_kind}"')
        class_start = cards_markup.rfind('class="', 0, task_index)
        class_end = cards_markup.index('"', class_start + len('class="'))
        assert cards_markup[class_start:class_end + 1] == 'class="welcome-task-card available"'


def test_agent_task_creation_prefills_conversation_composer_with_goal():
    app_js = _read_static("app.js")
    task_types_js = _read_static("js/task-types.js")

    # The V2 plan-composer dialog is retired; agent tasks now prefill the inline
    # conversation composer with the task type's suggested goal.
    helper_start = app_js.index("function prefillAgentTaskInstruction")
    helper_end = app_js.index("async function createTask", helper_start)
    helper_body = app_js[helper_start:helper_end]

    assert 'if (task?.run_mode !== "agent") return;' in helper_body
    assert 'const input = $("agentComposerInput");' in helper_body
    # Only seed when the composer is empty, so a user's draft is never clobbered.
    assert "if (!input || input.value.trim()) return;" in helper_body
    assert "const definition = taskTypeDefinition(task.task_type || createTaskDialog.activeTaskType());" in helper_body
    assert "input.value = definition.initialGoal;" in helper_body
    assert "autoGrowComposerInput();" in helper_body
    assert "updateAgentSendDisabled();" in helper_body
    assert "上传资产Vintage&滚动率分析、FPD、入催回收率分析数据" in task_types_js
    assert "先识别 cohort、MOB 和坏账标签字段" in task_types_js
    assert "计算资产 Vintage 曲线并给出风险观察" in task_types_js
    assert "营利性测算" not in app_js

    # createTask() invokes the prefill once the task is created.
    assert "prefillAgentTaskInstruction(task);" in app_js
    # The retired V2 workspace composer helpers are gone.
    assert "function seedV2GoalComposer" not in app_js
    assert "function openV2WorkspaceWithGoal" not in app_js
    assert "function showV2WorkspaceDialog" not in app_js


def test_driver_manual_analysis_omits_plan_overview_messages():
    app_js = _read_static("app.js")
    module_js = _read_static("js/v2/driver_manual_analysis.js")
    body = _slice_function(module_js, "export function driverManualAnalysisHtml")

    assert 'meta.kind === "overview" || meta.kind === "plan_overview"' in body
    assert "driverManualAnalysisHtmlController(messages" in app_js


def test_api_paths_are_absolute_and_agent_start_rejects_missing_task_id():
    app_js = _read_static("app.js")
    api_js = _read_static("js/api.js")

    api_start = api_js.index("export async function api")
    api_end = api_js.index("export function sleep", api_start)
    api_body = api_js[api_start:api_end]
    assert 'endpoint.startsWith("/")' in api_body
    assert '`/${endpoint}`' in api_body
    assert "fetch(normalizedEndpoint" in api_body

    dispatch_start = app_js.index("async function dispatchAgentValidation")
    dispatch_end = app_js.index("async function waitForAgentValidation", dispatch_start)
    dispatch_body = app_js[dispatch_start:dispatch_end]
    assert 'requireTaskId(taskId || selectedTaskId, "Agent 初始化")' in dispatch_body
    assert 'api(`/api/tasks/${normalizedTaskId}/agent/start`' in dispatch_body
    assert "api(`api/tasks/${taskId}/agent/start`" not in app_js


def test_delete_task_blocks_active_jobs_instead_of_stale_running_status():
    app_js = _read_static("app.js")

    delete_start = app_js.index("async function deleteTask")
    delete_end = app_js.index("async function runAction", delete_start)
    delete_body = app_js[delete_start:delete_end]

    assert "taskServerBusyAction(task)" in delete_body
    assert 'task.status === "running"' not in delete_body


def test_delete_task_uses_platform_confirm_dialog_instead_of_browser_confirm():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")
    platform_confirm_js = _read_static("js/platform-confirm.js")

    assert 'id="platformConfirmDialog"' in index_html
    assert 'id="platformConfirmTitle"' in index_html
    assert 'id="platformConfirmMessage"' in index_html
    assert 'id="platformConfirmCancelButton"' in index_html
    assert 'id="platformConfirmConfirmButton"' in index_html
    assert '<svg viewBox="0 0 32 32" focusable="false">' in index_html
    assert 'class="platform-confirm-icon-plate"' in index_html
    assert 'class="platform-confirm-icon-triangle"' in index_html
    assert 'class="platform-confirm-icon-mark"' in index_html
    assert 'class="platform-confirm-icon-dot"' in index_html
    assert 'M10.35 4.8 2.9 17.6' not in index_html
    assert ".platform-confirm-dialog" in styles_css
    assert ".platform-confirm-panel" in styles_css
    confirm_icon_rule = _css_rule(styles_css, ".platform-confirm-icon")
    assert "border-radius: 50%" not in confirm_icon_rule
    assert "background: transparent" in confirm_icon_rule
    assert "box-shadow: none" in confirm_icon_rule
    confirm_icon_svg_rule = _css_rule(styles_css, ".platform-confirm-icon svg")
    assert "display: block" in confirm_icon_svg_rule
    assert "width: 32px" in confirm_icon_svg_rule
    assert "height: 32px" in confirm_icon_svg_rule
    assert "stroke-width: 1.8" in confirm_icon_svg_rule
    confirm_icon_plate_rule = _css_rule(styles_css, ".platform-confirm-icon-plate")
    assert "fill: color-mix(in srgb, currentColor 8%, transparent)" in confirm_icon_plate_rule
    assert "stroke: color-mix(in srgb, currentColor 28%, transparent)" in confirm_icon_plate_rule
    assert "stroke-width: 1.2" in confirm_icon_plate_rule
    confirm_icon_triangle_rule = _css_rule(styles_css, ".platform-confirm-icon-triangle")
    assert "fill: color-mix(in srgb, currentColor 7%, transparent)" in confirm_icon_triangle_rule
    assert "stroke-width: 1.7" in confirm_icon_triangle_rule
    confirm_icon_mark_rule = _css_rule(styles_css, ".platform-confirm-icon-mark")
    assert "stroke-width: 2.1" in confirm_icon_mark_rule
    confirm_icon_dot_rule = _css_rule(styles_css, ".platform-confirm-icon-dot")
    assert "stroke-width: 2.4" in confirm_icon_dot_rule
    confirm_icon_danger_rule = _css_rule(
        styles_css, '.platform-confirm-dialog[data-tone="danger"] .platform-confirm-icon'
    )
    assert "color: var(--danger)" in confirm_icon_danger_rule
    assert "background:" not in confirm_icon_danger_rule
    assert "box-shadow:" not in confirm_icon_danger_rule
    assert "export function createPlatformConfirmController" in platform_confirm_js
    assert 'from "./js/platform-confirm.js"' in app_js
    assert "const platformConfirm = createPlatformConfirmController({ getElementById: $ });" in app_js
    assert "const showPlatformConfirm = platformConfirm.showPlatformConfirm;" in app_js
    assert "const bindPlatformConfirmDialog = platformConfirm.bindPlatformConfirmDialog;" in app_js
    assert "function showPlatformConfirm" not in app_js
    assert "function bindPlatformConfirmDialog" not in app_js
    assert "bindPlatformConfirmDialog();" in app_js
    assert "window.confirm" not in app_js

    delete_start = app_js.index("async function deleteTask")
    delete_end = app_js.index("async function runAction", delete_start)
    delete_body = app_js[delete_start:delete_end]

    assert "await showPlatformConfirm({" in delete_body
    assert "window.confirm" not in delete_body
    assert 'title: "删除任务"' in delete_body
    assert 'confirmText: "删除"' in delete_body
    assert 'cancelText: "取消"' in delete_body
    assert 'tone: "danger"' in delete_body

    confirm_action_rule = _css_rule(
        styles_css, '.platform-confirm-dialog[data-tone="danger"] .platform-confirm-affirmative'
    )
    assert "background: var(--danger)" in confirm_action_rule
    assert "border-color: var(--danger)" in confirm_action_rule
    assert "box-shadow: var(--button-solid-shadow)" in confirm_action_rule

    confirm_action_hover_rule = _css_rule(
        styles_css,
        '.platform-confirm-dialog[data-tone="danger"] .platform-confirm-affirmative:hover:not(:disabled),\n'
        '.platform-confirm-dialog[data-tone="danger"] .platform-confirm-affirmative:focus-visible:not(:disabled)',
    )
    assert "box-shadow: var(--button-solid-shadow-hover)" in confirm_action_hover_rule


def test_platform_confirm_controller_resolves_confirm_and_cancel():
    script = """
import assert from "node:assert/strict";
import { createPlatformConfirmController } from "./marvis/static/js/platform-confirm.js";

const elements = {};
function button() {
  return {
    textContent: "",
    onclick: null,
    focusCalls: 0,
    focus() {
      this.focusCalls += 1;
    },
  };
}
const dialog = {
  open: false,
  dataset: {},
  listeners: {},
  closeValue: "",
  showModal() {
    this.open = true;
  },
  close(value) {
    this.open = false;
    this.closeValue = value;
    this.listeners.close?.({ type: "close" });
  },
  addEventListener(name, fn) {
    this.listeners[name] = fn;
  },
};
elements.platformConfirmDialog = dialog;
elements.platformConfirmTitle = { textContent: "" };
elements.platformConfirmMessage = { textContent: "" };
elements.platformConfirmConfirmButton = button();
elements.platformConfirmCancelButton = button();

const controller = createPlatformConfirmController({ getElementById: (id) => elements[id] });
controller.bindPlatformConfirmDialog();

const confirmedPromise = controller.showPlatformConfirm({
  title: "删除任务",
  message: "确认删除?",
  confirmText: "删除",
  cancelText: "取消",
  tone: "danger",
});
assert.equal(dialog.open, true);
assert.equal(elements.platformConfirmTitle.textContent, "删除任务");
assert.equal(elements.platformConfirmMessage.textContent, "确认删除?");
assert.equal(elements.platformConfirmConfirmButton.textContent, "删除");
assert.equal(elements.platformConfirmCancelButton.textContent, "取消");
assert.equal(elements.platformConfirmCancelButton.focusCalls, 1);
assert.equal(dialog.dataset.tone, "danger");
elements.platformConfirmConfirmButton.onclick();
assert.equal(await confirmedPromise, true);
assert.equal(dialog.closeValue, "confirm");

const cancelledPromise = controller.showPlatformConfirm({ title: "二次确认" });
elements.platformConfirmCancelButton.onclick();
assert.equal(await cancelledPromise, false);
assert.equal(dialog.closeValue, "cancel");

const escapePromise = controller.showPlatformConfirm({ title: "ESC" });
let prevented = false;
dialog.listeners.cancel({
  preventDefault() {
    prevented = true;
  },
});
assert.equal(prevented, true);
assert.equal(await escapePromise, false);
process.stdout.write("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_agent_stop_response_polls_until_active_agent_job_finishes():
    app_js = _read_static("app.js")

    start_agent_start = app_js.index("async function startAgentValidation")
    start_agent_end = app_js.index("async function dispatchAgentValidation", start_agent_start)
    start_agent_body = app_js[start_agent_start:start_agent_end]

    assert 'result.status === "cancel_requested"' in start_agent_body
    assert "await waitForAgentValidation(taskId, { stopping: true });" in start_agent_body

    wait_start = app_js.index("async function waitForAgentValidation")
    wait_end = app_js.index("function handleTaskListKeydown", wait_start)
    wait_body = app_js[wait_start:wait_end]
    assert '"scanned"' in wait_body
    assert '"executed"' in wait_body
    assert "{ stopping }" in wait_body
    assert "agentValidationStopped(finalTask" in wait_body

    poll_start = app_js.index("async function pollValidationProgress")
    poll_end = app_js.index("async function validateCurrentTask", poll_start)
    poll_body = app_js[poll_start:poll_end]
    assert "{ stopping = false, background = false } = {}" in poll_body
    assert "const serverBusyAction = taskServerBusyAction(polledTask);" in poll_body
    assert "doneStatuses.has(status) && !serverBusyAction" in poll_body


def test_agent_mode_hides_empty_scan_section_until_evidence_or_messages():
    app_js = _read_static("app.js")
    mount_js = _read_static("js/agent-conversation-mount.js")

    stored_start = app_js.index("function renderStoredStateSummaries")
    stored_end = app_js.index("function renderAll", stored_start)
    stored_body = app_js[stored_start:stored_end]

    assert "updateAgentScanSectionVisibility();" in stored_body
    assert '"等待你输入“开始验证”后执行材料识别"' not in stored_body
    assert '"材料完备性识别将自动开始"' not in stored_body

    visibility_start = app_js.index("function updateAgentScanSectionVisibility")
    visibility_end = app_js.index("function renderStoredStateSummaries", visibility_start)
    visibility_body = app_js[visibility_start:visibility_end]
    assert "selectedTaskIsAgentMode()" in visibility_body
    assert 'const hasScanResult = scanSummaryHasResult();' in visibility_body
    assert 'scanSection.classList.toggle("hidden", !hasScanResult);' in visibility_body

    scan_start = app_js.index("function renderScanResult")
    scan_end = app_js.index("function renderValidationResult", scan_start)
    scan_body = app_js[scan_start:scan_end]
    assert "updateAgentScanSectionVisibility();" in scan_body

    timeline_start = app_js.index("function renderAgentTimeline")
    timeline_end = app_js.index("function resetAgentTypingState", timeline_start)
    timeline_body = app_js[timeline_start:timeline_end]
    assert "agentTimelineVisibleStages()" in timeline_body
    assert "renderAgentTimelineDom(messages, {" in timeline_body
    assert "taskFrozenSectionSnapshots" in timeline_body
    assert "agentTimelineItems(messages, deps.visibleStages || [], {" in mount_js
    assert "snapshotsByTrigger: deps.snapshotsByTrigger || agentFrozenSnapshotsByTriggerId({" in mount_js

    report_visibility_start = app_js.index("function updateAgentReportSectionVisibility")
    report_visibility_end = app_js.index(
        "function renderStoredStateSummaries",
        report_visibility_start,
    )
    report_visibility_body = app_js[report_visibility_start:report_visibility_end]
    assert "selectedTaskIsAgentMode()" not in report_visibility_body
    assert "reportSummary" not in report_visibility_body
    assert "wordReportTitle" not in report_visibility_body
    assert "reportFieldsForm" not in report_visibility_body
    assert '"agentReportLeadMessages"' in report_visibility_body
    assert '"agentReportMessages"' in report_visibility_body
    assert 'reportSection.setAttribute("aria-hidden", hasReportMessages ? "false" : "true");' in report_visibility_body


def test_llm_settings_panel_and_agent_model_selector_exist():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    assert "模型引擎" in index_html
    assert 'id="llmSettingsDialog"' not in index_html
    assert 'id="openLLMSettingsButton"' not in index_html
    assert 'data-governance-nav="llm"' in index_html
    assert 'data-governance-panel-content="llm"' in index_html
    assert 'id="llmModelProfiles"' in index_html
    assert 'class="llm-model-profiles llm-engine-list"' in index_html
    assert 'id="settingsLLMValue"' not in index_html
    assert 'id="agentModelSelect"' in index_html

    # Connection details (incl. API key) are edited in a focused dialog; the
    # settings area no longer picks a model — that happens in the composer.
    assert 'id="llmEngineEditDialog"' in index_html
    assert 'id="llmEngineModelName"' in index_html
    assert 'id="llmEngineBaseUrl"' in index_html
    assert 'id="llmEngineApiKey"' in index_html
    assert 'id="llmEngineEnableThinking"' in index_html
    assert 'id="addLLMModelButton"' in index_html
    assert 'class="llm-engine-toolbar"' not in index_html
    assert 'id="llmDefaultModelSelect"' not in index_html

    assert "api/settings/llm" in app_js
    assert "loadLLMSettings" in app_js
    assert "saveLLMSettings" in app_js
    assert "renderAgentModelOptions" in app_js
    assert "openLLMEngineEdit" in app_js
    assert "saveLLMEngineEdit" in app_js
    assert 'openGovernanceSettingsCenter("llm")' in app_js
    assert "enabled: model.enabled !== false" in app_js
    assert "enable_thinking: Boolean(model.enable_thinking)" in app_js
    assert "$(\"llmEngineEnableThinking\").checked = Boolean(model.enable_thinking)" in app_js
    assert "llm-engine-item" in app_js
    assert ".llm-engine-item" in styles_css
    assert ".checkbox-field" in styles_css

    llm_settings_group_rule = _css_rule(
        styles_css, '.governance-panel[data-governance-panel-content="llm"] > .settings-group'
    )
    assert "border: 0" in llm_settings_group_rule
    assert "background: transparent" in llm_settings_group_rule
    assert "overflow: visible" in llm_settings_group_rule
    llm_settings_row_rule = _css_rule(
        styles_css, '.governance-panel[data-governance-panel-content="llm"] > .settings-group > .settings-row'
    )
    assert "padding: 0 2px 2px" in llm_settings_row_rule
    llm_settings_head_rule = _css_rule(
        styles_css, '.governance-panel[data-governance-panel-content="llm"] .settings-row-head'
    )
    assert "align-items: center" in llm_settings_head_rule

    llm_edit_actions_rule = _css_rule(styles_css, ".llm-engine-edit-actions")
    assert "justify-content: flex-end" in llm_edit_actions_rule

    llm_edit_button_rule = _css_rule(styles_css, ".llm-engine-edit-actions .button")
    assert "min-width: 76px" in llm_edit_button_rule
    assert "min-height: 34px" in llm_edit_button_rule
    assert "padding: 6px 15px" in llm_edit_button_rule
    assert "font-size: 13px" in llm_edit_button_rule
    assert "font-weight: 600" in llm_edit_button_rule

    assert ".llm-engine-edit-actions .button.secondary" not in styles_css
    assert ".llm-engine-edit-actions .button.secondary:hover" not in styles_css

    llm_cancel_rule = _css_rule(styles_css, ".button.secondary")
    assert "border-color: var(--border-strong)" in llm_cancel_rule
    assert "background: var(--surface)" in llm_cancel_rule
    assert "box-shadow: var(--button-secondary-shadow)" in llm_cancel_rule

    llm_cancel_hover_rule = _css_rule(
        styles_css, ".button.secondary:hover:not(:disabled),\n.button.secondary:focus-visible:not(:disabled)"
    )
    assert "#9fbfe4" not in llm_cancel_hover_rule
    assert "#f8fbff" not in llm_cancel_hover_rule
    assert "color-mix(in srgb, var(--surface) 88%, var(--text) 12%)" in llm_cancel_hover_rule


def test_system_settings_center_keeps_extensions_without_runtime_workbench():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    draft_tools_panel_js = _read_static("js/draft-tools-panel.js")
    styles_css = _read_static("styles.css")
    v2_css = _read_static("css/v2-workbench.css")

    settings_start = index_html.index('id="sidebarSettings"')
    settings_end = index_html.index("</details>", settings_start)
    settings_markup = index_html[settings_start:settings_end]

    assert 'data-settings-row="system"' in settings_markup
    assert 'id="openGovernanceSettingsButton"' in settings_markup
    assert 'class="settings-system-row"' in settings_markup
    assert "环境、模型、记忆与 Runtime" not in settings_markup
    assert "系统设置" in settings_markup
    assert "运行配置" not in settings_markup
    assert "治理与扩展" not in settings_markup
    assert "治理中心" not in settings_markup
    assert 'id="openExecutionEnvironmentButton"' not in settings_markup
    assert 'id="openLLMSettingsButton"' not in settings_markup
    assert 'id="openDraftToolsButton"' not in settings_markup
    assert 'id="openV2WorkspaceButton"' not in settings_markup
    assert 'id="openAgentMemoryButton"' not in settings_markup
    assert 'class="settings-governance-card"' not in settings_markup

    assert 'id="governanceSettingsDialog"' in index_html
    assert 'id="governanceSettingsNav"' in index_html
    assert 'id="governanceSettingsSearch"' in index_html
    search_field_start = index_html.index('class="governance-search-field"')
    search_field_end = index_html.index("</label>", search_field_start)
    search_field_markup = index_html[search_field_start:search_field_end]
    assert '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' in search_field_markup
    assert '<circle cx="10.5" cy="10.5" r="6"></circle>' in search_field_markup
    assert '<path d="m14.85 14.85 4.15 4.15"></path>' in search_field_markup
    assert 'id="closeGovernanceSettingsButton"' in index_html
    # The IA refactor consolidated 11 nav items into 6: memory records /
    # distillations moved inside the memory-policy panel, while the old
    # plan/audit runtime workbench was retired from settings.
    for nav in [
        "execution-environment",
        "llm",
        "capabilities",
        "memory-policy",
        "plugins",
        "workflows",
    ]:
        assert f'data-governance-nav="{nav}"' in index_html
    for retired_nav in [
        "memory-records",
        "memory-distillations",
        "drafts",
        "runtime-plan",
        "audit",
    ]:
        assert f'data-governance-nav="{retired_nav}"' not in index_html

    assert 'data-governance-panel-content="execution-environment"' in index_html
    assert 'data-governance-panel-content="llm"' in index_html
    assert 'id="executionEnvironmentList"' in index_html
    assert 'id="refreshExecutionEnvironmentOptionsButton"' in index_html
    # Click-to-select radiogroup auto-saves; the <select> + save button are gone.
    assert 'id="executionEnvironmentSelect"' not in index_html
    assert 'id="saveExecutionEnvironmentButton"' not in index_html
    assert 'id="llmModelProfiles"' in index_html
    assert 'id="addLLMModelButton"' in index_html
    assert "Notebook 和工具运行使用的 Python 环境" in index_html
    assert "配置本地会话可调用的大模型连接信息" in index_html
    assert "控制 Agent 记忆的引用范围" in app_js
    # Memory policies are checkbox switches now; only the platform-forced
    # (sensitive-material) row still renders a status badge.
    assert 'class="memory-policy-switch" data-memory-policy="reference_cross_task"' in index_html
    assert 'class="memory-policy-switch" data-memory-policy="auto_distill"' in index_html
    assert 'class="governance-status-badge required">平台强制</span>' in index_html
    assert 'class="governance-toggle"' not in index_html
    assert "复核 Agent 产出的工具草稿" in index_html
    assert "管理可调用工具包" in app_js
    assert 'id="draftToolsDialog"' not in index_html
    assert 'id="v2WorkspaceDialog"' not in index_html
    assert 'id="agentMemoryDialog"' not in index_html
    assert 'id="executionEnvironmentDialog"' not in index_html
    assert 'id="llmSettingsDialog"' not in index_html

    assert 'id="draftToolsList"' in index_html
    assert 'id="draftToolDetail"' in index_html
    assert 'id="draftStatusFilter"' in index_html
    # Drafts live in a collapsible <details> that lazy-loads on toggle; the
    # standalone refresh button was retired.
    assert 'id="draftManageDetails"' in index_html
    assert 'id="refreshDraftToolsButton"' not in index_html
    assert 'id="draftRunInputs"' in index_html
    assert 'id="draftPromotionTestCases"' in index_html
    assert 'id="runDraftButton"' in index_html
    assert 'id="promoteDraftButton"' in index_html
    assert 'id="rejectDraftButton"' in index_html
    assert "无网时请在外部产出工具后通过插件上传导入" in index_html
    assert 'class="draft-tools-sidebar"' in index_html
    assert 'class="draft-tool-overview-card"' in index_html
    assert 'class="draft-tool-section"' in index_html
    assert "工具代码" in index_html
    assert "输入 / 输出契约" in index_html
    assert "试运行" in index_html
    assert "转正闸门" in index_html

    assert 'id="agentMemoryList"' in index_html
    assert 'id="agentMemoryStatusFilter"' in index_html
    assert 'data-agent-memory-view="raw"' in index_html
    assert 'data-agent-memory-view="distillation"' in index_html
    assert 'data-agent-memory-mode=' not in index_html
    assert 'id="governanceExtensionMount"' in index_html
    assert 'id="v2RuntimeMount"' not in index_html
    assert '计划与执行' not in index_html
    assert 'data-governance-panel-content="extensions"' in index_html
    # Plugins / workflows / capabilities share the extension panel and are
    # selected via data-extension-view; their controls are rendered into the
    # extension mount by the settings modules rather than baked into index.html.
    assert 'data-governance-panel="extensions" data-extension-view="plugins"' in index_html
    assert 'data-governance-panel="extensions" data-extension-view="workflows"' in index_html
    assert 'data-governance-panel="extensions" data-extension-view="capabilities"' in index_html
    assert 'data-governance-panel="runtime"' not in index_html
    assert 'data-v2-view=' not in index_html
    capabilities_nav_start = index_html.index('data-governance-nav="capabilities"')
    capabilities_nav_end = index_html.index("</button>", capabilities_nav_start)
    capabilities_nav_markup = index_html[capabilities_nav_start:capabilities_nav_end]
    assert '<path d="M3.34 19a10 10 0 1 1 17.32 0"></path>' in capabilities_nav_markup
    assert '<path d="M12 14l4-4"></path>' in capabilities_nav_markup
    assert '<circle cx="12" cy="18" r="1.5"></circle>' not in capabilities_nav_markup
    memory_nav_start = index_html.index('data-governance-nav="memory-policy"')
    memory_nav_end = index_html.index("</button>", memory_nav_start)
    memory_nav_markup = index_html[memory_nav_start:memory_nav_end]
    assert '<circle cx="14" cy="6" r="2"></circle>' in memory_nav_markup
    assert '<circle cx="16" cy="18" r="2"></circle>' in memory_nav_markup

    assert "function openGovernanceSettingsCenter" in app_js
    assert "function openGovernanceSettingsFromSidebar" in app_js
    assert "function closeSidebarSettingsMenu" in app_js
    assert "function scheduleGovernanceSettingsFromSidebar" in app_js
    assert "function handleGovernanceSettingsPointerDown" in app_js
    assert "function setSidebarSettingsSuppressed" not in app_js
    assert "function handleGovernanceSettingsDialogClose" not in app_js
    assert "function setGovernanceSettingsPanel" in app_js
    assert "handleGovernanceSettingsSearch" in app_js
    assert '$("openGovernanceSettingsButton").onclick = openGovernanceSettingsFromSidebar;' in app_js
    assert '$("openGovernanceSettingsButton").addEventListener("pointerdown", handleGovernanceSettingsPointerDown, true);' in app_js
    assert 'openGovernanceSettingsCenter("execution-environment")' in app_js
    assert 'openGovernanceSettingsCenter("llm")' in app_js
    assert '$("governanceSettingsDialog").addEventListener("click", handleGovernanceSettingsNavClick);' in app_js
    assert 'import { createDraftToolsPanelController } from "./js/draft-tools-panel.js";' in app_js
    assert "const draftToolsPanel = createDraftToolsPanelController" in app_js
    assert "let draftTools = [];" not in app_js
    assert "let draftTools = [];" in draft_tools_panel_js
    # Drafts open via the plugins extension <details> toggle (lazy load) and the
    # status filter, not a dedicated dialog or nav key.
    assert '$("draftManageDetails").addEventListener("toggle"' in app_js
    assert "!draftToolsPanel.hasLoaded()" in app_js
    assert 'runAction(loadDraftTools, { actionId: "draftTools"' in app_js
    assert "function openDraftToolsDialog" not in app_js
    assert 'openGovernanceSettingsCenter("drafts")' not in app_js
    assert "async function loadDraftTools" in app_js
    assert "async function inspectDraftTool" in app_js
    assert "async function runDraftTool" in app_js
    assert "async function promoteDraftTool" in app_js
    assert "async function rejectDraftTool" in app_js
    assert "return draftToolsPanel.load({ preserveSelection });" in app_js
    assert "return draftToolsPanel.run();" in app_js
    assert "return draftToolsPanel.promote();" in app_js
    assert "return draftToolsPanel.reject();" in app_js
    assert 'api("/api/drafts' in draft_tools_panel_js
    assert 'api(`/api/drafts/${encodeURIComponent(draftId)}/run`' in draft_tools_panel_js
    assert 'api(`/api/drafts/${encodeURIComponent(draftId)}/promote`' in draft_tools_panel_js
    assert 'api(`/api/drafts/${encodeURIComponent(draftId)}/reject`' in draft_tools_panel_js
    assert '"X-MARVIS-Plugin-Admin": "local-dev"' in draft_tools_panel_js
    assert "function governanceExtensionActions" in app_js
    assert "pluginActions:" in app_js
    assert "skillActions:" in app_js
    assert "draftActions:" not in app_js
    assert "memoryActions:" not in app_js
    assert "function mountV2Runtime" not in app_js
    assert "function mountGovernanceExtensions" in app_js
    assert "function runV2WorkspaceAction" not in app_js
    assert "function runGovernanceExtensionAction" in app_js
    assert 'title: "移除插件"' in app_js
    assert "showError: showExtensionError" in app_js
    assert "confirmRemove: (name) => showPlatformConfirm({" in app_js
    assert "await renderPluginManager(mounted.panels.pluginPanel, actions.pluginActions)" in app_js
    assert "await renderSkillManager(mounted.panels.skillPanel, actions.skillActions)" in app_js
    assert "await renderTierSettings(mounted.panels.capabilityPanel, actions.capabilityActions)" in app_js
    assert "请填写转正测试用例。" in draft_tools_panel_js
    assert "转正后该工具会进入正式工具库并可被 Planner 选用，确定转正？" in draft_tools_panel_js

    assert "dialog.governance-settings-dialog" in styles_css
    assert ".governance-settings-dialog" in styles_css
    assert ".governance-settings-shell" in styles_css
    assert ".governance-nav-item.selected" in styles_css
    for selector in [
        ".governance-search-field input",
        ".governance-nav-item",
    ]:
        assert "border-radius: var(--radius-control)" in _css_rule(styles_css, selector)
    nav_icon_rule = _css_rule(styles_css, ".governance-nav-icon")
    assert "width: 20px" in nav_icon_rule
    assert "height: 20px" in nav_icon_rule
    nav_icon_svg_rule = _css_rule(styles_css, ".governance-nav-icon svg")
    assert "display: block" in nav_icon_svg_rule
    assert "width: 18px" in nav_icon_svg_rule
    assert "height: 18px" in nav_icon_svg_rule
    assert "stroke-width: 1.8" in nav_icon_svg_rule
    nav_label_rule = _css_rule(styles_css, ".governance-nav-item strong")
    assert "line-height: 20px" in nav_label_rule
    search_field_rule = _css_rule(styles_css, ".governance-search-field")
    assert "position: relative" in search_field_rule
    search_icon_rule = _css_rule(styles_css, ".governance-search-field svg")
    assert "left: 11px" in search_icon_rule
    assert "width: var(--sidebar-control-icon-size)" in search_icon_rule
    assert "stroke: currentColor" in search_icon_rule
    search_input_rule = _css_rule(styles_css, ".governance-search-field input")
    assert "padding: 0 12px 0 36px" in search_input_rule
    nav_hover_rule = _css_rule(styles_css, ".governance-nav-item:hover")
    assert "background: var(--option-hover)" in nav_hover_rule
    assert "background: var(--surface)" not in nav_hover_rule
    nav_focus_rule = _css_rule(styles_css, ".governance-nav-item:focus-visible")
    assert "background: var(--option-hover)" in nav_focus_rule
    assert "outline: 3px solid var(--option-focus-ring)" in nav_focus_rule
    assert "var(--accent" not in nav_focus_rule
    nav_selected_rule = _css_rule(styles_css, ".governance-nav-item.selected")
    assert "background: var(--option-selected)" in nav_selected_rule
    assert "box-shadow: none" in nav_selected_rule
    assert "border-color" not in nav_selected_rule
    assert "background: var(--surface)" not in nav_selected_rule
    governance_icon_button_rule = _css_rule(styles_css, ".governance-head-actions .governance-icon-button")
    assert "border-radius: var(--radius-control)" in governance_icon_button_rule
    assert ".governance-setting-row .governance-status-badge" in styles_css
    assert ".hidden,\n[hidden]" in styles_css
    assert "display: none !important;" in styles_css
    assert ".draft-tools-layout" in styles_css
    assert ".draft-tools-sidebar" in styles_css
    assert ".draft-tool-section" in styles_css
    assert ".draft-tool-detail" in styles_css
    assert ".draft-code-block" in styles_css

    assert ".v2-workspace-summary" in v2_css
    assert "grid-template-areas:" not in v2_css
    for selector in [
        ".v2-plugin-panel",
        ".v2-skill-panel",
        ".v2-capability-panel",
    ]:
        assert selector in v2_css
    for retired_selector in [
        ".v2-goal-panel",
        ".v2-plan-panel",
        ".v2-join-panel",
        ".v2-subagent-panel",
        ".v2-draft-panel",
        ".v2-memory-panel",
        ".v2-loop-panel",
        ".v2-artifact-panel",
    ]:
        assert retired_selector not in v2_css
    assert '.governance-settings-dialog .plugin-row input[type="checkbox"]' in v2_css
    assert '.governance-settings-dialog[data-extension-view="plugins"] .v2-plugin-panel' in v2_css
    assert '.governance-settings-dialog[data-extension-view="capabilities"] .v2-capability-panel' in v2_css
    runtime_panel_rule = _css_rule(v2_css, ".governance-settings-dialog .v2-panel")
    assert "padding: 0" in runtime_panel_rule
    assert "border: 0" in runtime_panel_rule
    assert "background: transparent" in runtime_panel_rule
    plugin_upload_rule = _css_rule(v2_css, ".governance-settings-dialog .plugin-upload")
    assert "grid-template-columns: minmax(0, 1fr) auto" in plugin_upload_rule
    plugin_upload_action_rule = _css_rule(v2_css, ".governance-settings-dialog .plugin-upload::after")
    assert 'content: "选择文件"' in plugin_upload_action_rule
    assert "border: 1px dashed var(--button-outline-border)" in plugin_upload_action_rule
    assert "color: var(--button-outline-text)" in plugin_upload_action_rule
    skill_reload_rule = _css_rule(v2_css, ".governance-settings-dialog .skill-manager > button")
    assert "background: var(--button-primary-bg)" in skill_reload_rule
    assert "color: var(--button-primary-text)" in skill_reload_rule
    assert "box-shadow: var(--button-solid-shadow)" in skill_reload_rule
    memory_toolbar_rule = _css_rule(v2_css, ".governance-settings-dialog .memory-manager-toolbar")
    assert "padding: 0 2px 2px" in memory_toolbar_rule
    memory_row_rule = _css_rule(
        v2_css,
        ".governance-settings-dialog .tier-row,\n"
        ".governance-settings-dialog .plugin-row,\n"
        ".governance-settings-dialog .skill,\n"
        ".governance-settings-dialog .memory-distillation-row,\n"
        ".governance-settings-dialog .memory-distillation-detail-inner",
    )
    assert "border: 1px solid var(--border)" in memory_row_rule
    assert "border-radius: var(--radius-control)" in memory_row_rule
    memory_rollback_rule = _css_rule(
        v2_css, ".governance-settings-dialog .memory-distillation-row button[data-rollback-memory-distillation]"
    )
    assert "color: var(--danger)" in memory_rollback_rule

    memory_policy_group_rule = _css_rule(
        styles_css, '.governance-panel[data-governance-panel-content="memory-policy"] > .settings-group'
    )
    assert "border: 0" in memory_policy_group_rule
    assert "background: transparent" in memory_policy_group_rule
    memory_policy_row_rule = _css_rule(
        styles_css, '.governance-panel[data-governance-panel-content="memory-policy"] > .settings-group > .settings-row'
    )
    assert "border-radius: var(--radius-control)" in memory_policy_row_rule
    assert "background: var(--surface-soft)" in memory_policy_row_rule
    memory_policy_row_gap_rule = _css_rule(
        styles_css, '.governance-panel[data-governance-panel-content="memory-policy"] .settings-row + .settings-row'
    )
    assert "border-top: 0" in memory_policy_row_gap_rule
    memory_manage_rule = _css_rule(styles_css, ".memory-manage")
    assert "border: 0" in memory_manage_rule
    assert "background: transparent" in memory_manage_rule
    memory_manage_summary_rule = _css_rule(styles_css, ".memory-manage-summary")
    assert "border-radius: var(--radius-control)" in memory_manage_summary_rule
    assert "background: var(--surface-soft)" in memory_manage_summary_rule


def test_agent_conversation_panel_layout_and_message_shapes():
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")
    app_js = _read_static("app.js")
    conversation_js = _read_static("js/agent-conversation-view.js")
    mount_js = _read_static("js/agent-conversation-mount.js")

    assert 'id="agentConversationPanel"' in index_html
    assert 'id="agentScanLeadMessages"' in index_html
    assert 'id="agentScanBeforeMessages"' in index_html
    assert 'id="agentScanMessages"' in index_html
    assert 'id="agentReproducibilityMessages"' in index_html
    assert 'id="agentMetricMessages"' in index_html
    assert 'id="agentReportMessages"' in index_html
    assert 'id="agentComposer"' in index_html
    panel_start = index_html.index('id="agentConversationPanel"')
    panel_end = index_html.index("</section>", panel_start)
    panel_markup = index_html[panel_start:panel_end]
    assert index_html.index('id="agentComposer"') > panel_end
    assert index_html.index('id="reportSection"') < index_html.index('id="agentConversationPanel"')
    assert index_html.index('id="agentScanLeadMessages"') < index_html.index("<h3>材料识别</h3>")
    assert index_html.index('id="agentScanBeforeMessages"') < index_html.index('id="scanSummary"')
    assert index_html.index('id="scanSummary"') < index_html.index('id="agentScanMessages"')
    assert "<h3>Agent</h3>" not in panel_markup
    assert 'id="agentMessages"' in index_html
    assert 'id="agentComposerInput"' in index_html
    assert 'id="sendAgentMessageButton"' in index_html
    assert 'placeholder="输入执行步骤或问题"' in index_html

    # model + effort selectors live inside the input box's toolbar; effort is sent
    assert 'class="agent-composer-toolbar"' in index_html
    assert "agent-composer-chip" in index_html
    assert 'id="agentEffortSelect"' in index_html
    bar_open = index_html.index('class="agent-composer-bar"')
    assert bar_open < index_html.index('id="agentModelSelect"')
    assert bar_open < index_html.index('id="agentEffortSelect"')
    assert "agentEffort" in app_js
    assert ".agent-composer-chip" in styles_css
    assert "agent-composer-active" in app_js
    assert ".result-workspace.agent-composer-active .result-scroll-content" in styles_css
    assert "--agent-composer-gap: 28px;" in styles_css
    assert "--agent-composer-clearance: 118px;" in styles_css
    assert "--agent-composer-clearance: 154px;" not in styles_css
    assert "function syncAgentComposerClearance" in app_js
    assert 'workspace.style.setProperty("--agent-composer-clearance"' in app_js
    assert "composerHeight + composerGap" in app_js
    assert ".agent-send:hover:not(:disabled)" in styles_css
    assert ".agent-send:hover {" not in styles_css
    send_start = styles_css.index(".agent-send {")
    send_end = styles_css.index("}", send_start)
    send_rule = styles_css[send_start:send_end]
    assert "color: var(--button-primary-text);" in send_rule
    assert "background: var(--button-primary-bg);" in send_rule
    assert "0 3px 10px rgba(0, 0, 0, 0.18)" in send_rule
    send_hover_start = styles_css.index(".agent-send:hover:not(:disabled) {")
    send_hover_end = styles_css.index("}", send_hover_start)
    send_hover_rule = styles_css[send_hover_start:send_hover_end]
    assert "color: var(--button-primary-text-hover);" in send_hover_rule
    assert "background: var(--button-primary-bg-hover);" in send_hover_rule
    assert "transform:" not in send_hover_rule
    assert "background: #a91017;" not in send_hover_rule
    send_active_rule = _css_rule(styles_css, ".agent-send:active:not(:disabled)")
    assert "background: var(--button-primary-bg-active);" in send_active_rule

    dark_send_rule = _css_rule(styles_css, 'body[data-theme="dark"] .agent-send')
    assert "background:" not in dark_send_rule
    assert 'body[data-theme="dark"] .agent-send:hover:not(:disabled)' not in styles_css
    dark_send_active_rule = _css_rule(styles_css, 'body[data-theme="dark"] .agent-send:active:not(:disabled)')
    assert "background: var(--button-primary-bg-active);" in dark_send_active_rule

    disabled_start = styles_css.index(".agent-send:disabled {")
    disabled_end = styles_css.index("}", disabled_start)
    disabled_rule = styles_css[disabled_start:disabled_end]
    assert "cursor: default;" in disabled_rule
    assert "cursor: not-allowed;" not in disabled_rule

    conversation_start = styles_css.index(".agent-conversation {")
    conversation_end = styles_css.index("}", conversation_start)
    conversation_rule = styles_css[conversation_start:conversation_end]
    assert "margin-top: 0;" in conversation_rule
    assert "padding: 8px 0 0;" in conversation_rule
    assert "min-height:" not in conversation_rule
    assert "border:" not in conversation_rule
    assert "background:" not in conversation_rule
    assert "box-shadow:" not in conversation_rule
    assert 'body[data-theme="dark"] .agent-conversation' not in styles_css

    messages_start = styles_css.index(".agent-messages {", styles_css.index(".agent-stage-messages.hidden"))
    messages_end = styles_css.index("}", messages_start)
    messages_rule = styles_css[messages_start:messages_end]
    assert "padding: 8px 2px 0;" in messages_rule
    assert "overflow-y:" not in messages_rule
    assert "max-height:" not in messages_rule
    assert ".agent-stage-messages" in styles_css
    assert ".agent-stage-lead-messages" in styles_css
    assert ".agent-stage-after-messages" in styles_css
    stage_start = styles_css.index(".agent-stage-messages {")
    stage_end = styles_css.index("}", stage_start)
    stage_rule = styles_css[stage_start:stage_end]
    assert "border-top" not in stage_rule
    lead_start = styles_css.index(".agent-stage-lead-messages {")
    lead_end = styles_css.index("}", lead_start)
    lead_rule = styles_css[lead_start:lead_end]
    assert "margin-bottom: 13px;" in lead_rule
    assert "padding-bottom: 13px;" in lead_rule
    assert "border-bottom" in lead_rule
    assert "border-top" not in lead_rule
    after_start = styles_css.index(".agent-stage-after-messages {")
    after_end = styles_css.index("}", after_start)
    after_rule = styles_css[after_start:after_end]
    assert "margin-top: 13px;" in after_rule
    assert "padding-top: 13px;" in after_rule
    assert "border-top" in after_rule
    adjacent_start = styles_css.index(".agent-stage-lead-messages:not(.hidden) + .agent-stage-after-messages:not(.hidden) {")
    adjacent_end = styles_css.index("}", adjacent_start)
    adjacent_rule = styles_css[adjacent_start:adjacent_end]
    assert "border-top: 0;" in adjacent_rule
    adjacent_lead_start = styles_css.index(".agent-stage-lead-messages:not(.hidden):has(+ .agent-stage-after-messages:not(.hidden)) {")
    adjacent_lead_end = styles_css.index("}", adjacent_lead_start)
    adjacent_lead_rule = styles_css[adjacent_lead_start:adjacent_lead_end]
    assert "border-bottom: 0;" in adjacent_lead_rule

    composer_start = styles_css.index(".agent-composer {")
    composer_end = styles_css.index("}", composer_start)
    composer_rule = styles_css[composer_start:composer_end]
    assert "position: absolute;" in composer_rule
    assert "bottom: 0;" in composer_rule
    assert "left: 0;" in composer_rule
    assert "right: 0;" in composer_rule
    assert "margin-top: -94px;" not in composer_rule
    assert ".agent-composer::before" in styles_css
    # The composer bar carries only an inset hairline highlight — no heavy
    # drop shadow at rest or on focus (the box-shadow was the visual
    # clutter the user explicitly asked to remove).
    bar_start = styles_css.index(".agent-composer-bar {")
    bar_end = styles_css.index("}", bar_start)
    bar_rule = styles_css[bar_start:bar_end]
    assert "inset 0 1px 0" in bar_rule
    assert "0 16px 44px" not in bar_rule
    assert "0 4px 14px" not in bar_rule

    focus_start = styles_css.index(".agent-composer-bar:focus-within {")
    focus_end = styles_css.index("}", focus_start)
    focus_rule = styles_css[focus_start:focus_end]
    assert "var(--accent)" not in focus_rule
    assert "0 0 0 3px" not in focus_rule
    assert "0 18px 48px" not in focus_rule
    assert "0 6px 18px" not in focus_rule

    assert ".agent-message.user" in styles_css
    assert ".agent-message.user .agent-message-content" in styles_css
    assert "margin-left: auto;" in styles_css
    user_start = styles_css.index(".agent-message.user .agent-message-content")
    user_end = styles_css.index("}", user_start)
    user_rule = styles_css[user_start:user_end]
    assert "max-width: min(300px, 86%)" in user_rule
    assert "background: var(--agent-user-message-bg)" in user_rule
    assert "box-shadow: var(--agent-user-message-shadow)" in user_rule
    assert "font-size: 14px" in user_rule
    assert "line-height: 1.58" in user_rule
    assert "border:" not in user_rule
    assert "backdrop-filter" not in user_rule
    assert "rgba(255, 255, 255, 0.62)" not in user_rule
    assert "rgba(255, 255, 255, 0.16) 34%" not in user_rule
    assert "0 8px 24px" not in user_rule
    assert "var(--accent)" not in user_rule

    dark_user_start = styles_css.index('body[data-theme="dark"] .agent-message.user .agent-message-content')
    dark_user_end = styles_css.index("}", dark_user_start)
    dark_user_rule = styles_css[dark_user_start:dark_user_end]
    assert "background: var(--agent-user-message-bg)" in dark_user_rule
    assert "box-shadow: var(--agent-user-message-shadow)" in dark_user_rule
    assert "border" not in dark_user_rule
    assert "backdrop-filter" not in dark_user_rule
    assert "rgba(255, 255, 255, 0.12)" not in dark_user_rule
    assert "rgba(255, 255, 255, 0.04) 36%" not in dark_user_rule
    assert "0 8px 24px" not in dark_user_rule
    assert "var(--accent)" not in dark_user_rule

    assert ".agent-message.assistant .agent-message-content" in styles_css
    assert ".agent-message:last-child" not in styles_css
    assert "agent-message-rise" not in styles_css
    assistant_start = styles_css.index(".agent-message.assistant .agent-message-content")
    assistant_end = styles_css.index("}", assistant_start)
    assistant_rule = styles_css[assistant_start:assistant_end]
    assert "background:" not in assistant_rule
    assert "border:" not in assistant_rule

    assert "renderAgentConversation" in app_js
    assert "agentTypingState" in app_js
    assert "function agentMessageIsStreaming" in app_js
    assert "function agentMessageIsThinking" in app_js
    assert "function agentThinkingHtml" in app_js
    assert "function agentVisibleContent" in app_js
    assert "function tickAgentTyping" in app_js
    assert "const AGENT_STREAM_POLL_INTERVAL_MS = 180;" in app_js
    assert "const AGENT_TYPEWRITER_INTERVAL_MS = 12;" in app_js
    assert "const AGENT_TYPEWRITER_CHARS_PER_TICK = 2;" in app_js
    assert "const AGENT_TYPEWRITER_CATCHUP_TICKS" in app_js
    assert "metadata.streaming === true" in app_js
    assert "data-agent-streaming" in app_js
    assert "data-agent-thinking" in app_js
    assert "正在思考" in app_js
    assert "agent-thinking" in app_js
    assert ".agent-thinking" in styles_css
    assert "agent-thinking-icon" not in app_js
    assert ".agent-thinking-icon" not in styles_css
    assert "agent-thinking-pulse" in styles_css
    assert "agent-thinking-dot" in styles_css
    assert "agent-typing-cursor" not in app_js
    assert ".agent-typing-cursor" not in styles_css
    assert "agent-typing-cursor-blink" not in styles_css
    assert "function requestAgentConversationScrollToLatest" in app_js
    assert "scrollContent.scrollTo({ top: scrollContent.scrollHeight, behavior: \"auto\" });" in app_js
    assert "agent/report-draft/confirm" not in app_js
    assert "confirmAgentDraft" not in app_js
    assert "confirmDraft" not in app_js
    assert "确认写入" not in app_js
    assert "agent-empty" not in app_js
    assert "Agent 将按流程自动执行" not in app_js
    assert "agent-message user" in app_js
    assert "agent-message assistant" in app_js
    assert "renderAgentTimeline" in app_js
    render_start = app_js.index("function renderAgentConversation")
    render_end = app_js.index("function agentStructuralSignature", render_start)
    render_body = app_js[render_start:render_end]
    # v2 wiring: renderAgentTimeline must be fed via agentReportMessagesForDisplay.
    assert "agentReportMessagesForDisplay(agentMessages)" in render_body
    assert "renderAgentTimeline(" in render_body
    assert "agentMessages.filter((message, index) => !agentMessageTargetId(message, index, agentMessages))" not in render_body
    assert "requestAgentConversationScrollToLatest();" in render_body
    assert "function agentTimelineItems" in conversation_js
    assert "function agentTimelineInsertionIndex" in conversation_js
    assert "function agentReportMessagesForDisplay" in conversation_js
    assert "function agentMessagesHtml" in conversation_js
    assert "function agentMessageHtml(message, labelStage = message?.stage, options = {})" in app_js
    assert "agentMessagesHtml(item.messages, undefined, {" in mount_js
    assert "export function renderAgentTimeline" in mount_js
    assert "agentMessageMetaLabel(message, labelStage)" in app_js
    alias_start = app_js.index("function agentValidatorAlias")
    stage_label_start = app_js.index("function agentStageLabel", alias_start)
    stage_label_end = app_js.index("function formatAgentMessageContent", stage_label_start)
    alias_and_stage_label_body = app_js[alias_start:stage_label_end]
    stage_label_body = app_js[stage_label_start:stage_label_end]
    assert "function agentValidatorAlias" in alias_and_stage_label_body
    assert 'return agentValidatorAlias(selectedTask?.validator) || "Agent";' in stage_label_body
    # Real validator names must not be hard-coded in the shipped bundle: the alias
    # map is sourced from the workspace brand.json via agentValidatorAliases.
    assert "agentValidatorAliases[String(validator" in alias_and_stage_label_body
    assert "于添" not in alias_and_stage_label_body
    assert "张雯萱" not in alias_and_stage_label_body
    assert "材料完备性" not in stage_label_body
    assert "分数一致性" not in stage_label_body
    assert "效果与稳定性" not in stage_label_body
    assert 'metadata.tool_call?.name === "scan_materials"' in conversation_js
    assert "function agentMessageIsAdvanceIntent" in conversation_js
    assert 'metadata.intent === "advance"' in conversation_js
    assert "agentTimelineVisibleStages" in app_js
    assert "restoreResultScrollDefaultOrder" in app_js
    assert 'if (message.role === "user" && nextStage) return agentTargetIdForStage(nextStage);' not in app_js
    assert 'if (message.stage === "chat" && nextStage) return agentTargetIdForStage(nextStage);' not in app_js


def test_agent_message_meta_label_includes_plan_step_context():
    app_js = _read_static("app.js")
    plan_js = _read_static("js/v2/plan_rail_controller.js")
    meta_start = app_js.index("function agentMessageMetaLabel")
    meta_end = app_js.index("function formatAgentMessageContent", meta_start)
    meta_body = app_js[meta_start:meta_end]

    assert "function agentMessagePlanStep" in meta_body
    assert "planRailController.planStep(metadata, selectedTaskId)" in meta_body
    assert "metadata.step_id" in plan_js
    assert "metadata.step_title || step?.title" in meta_body
    assert "metadata.phase || step?.phase" in meta_body
    assert "metadata.run_seq" in meta_body
    assert "第 ${runSeq} 轮" in meta_body


def test_agent_memory_has_no_permanent_task_top_block():
    """V1.1 memory may appear in the agent conversation and management UI,
    but must not become a fixed gray task-level panel above the workflow.
    """
    app_js = _read_static("app.js")
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")
    result_area = index_html[index_html.index('id="resultWorkspace"') : index_html.index('id="agentComposer"')]

    forbidden_names = [
        "memoryPanel",
        "memorySummary",
        "memoryBlock",
        "agentMemoryPanel",
        "agentMemorySummary",
        "agentMemoryBlock",
    ]
    for name in forbidden_names:
        assert f'id="{name}"' not in result_area
        assert f'class="{name}"' not in result_area
        assert f"#{name}" not in styles_css
        assert f".{name}" not in styles_css
        assert f'$("${name}")' not in app_js

    assert 'id="governanceSettingsDialog"' in index_html
    assert 'id="agentMemoryList"' in index_html


def test_agent_message_renderer_outputs_inline_memory_references():
    app_js = _read_static("app.js")
    assert "function agentMemoryReferencesHtml" in app_js
    assert "agentMemoryReferencesHtml(message?.metadata?.memory_references)" in app_js
    assert "memory_references" in _slice_function(app_js, "function agentStructuralSignature")

    references_start = app_js.index("function agentMemoryReferencesHtml")
    references_end = app_js.index("function agentMessageHtml", references_start)
    script = "\n".join(
        [
            "function escapeHtml(value) {",
            "  return String(value || '').replace(/[&<>\"']/g, (char) => ({",
            "    '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', \"'\": '&#39;',",
            "  }[char]));",
            "}",
            "function formatMemoryConfidence(value) { return `${Math.round(Number(value) * 100)}%`; }",
            app_js[references_start:references_end],
            "const html = agentMemoryReferencesHtml([{",
            "  id: 'mem-1',",
            "  memory_type: 'field_profile',",
            "  source_task_id: 'task-7',",
            "  confidence: 0.86,",
            "  use_reason: '沿用坏样本字段口径',",
            "}]);",
            "process.stdout.write(html);",
        ]
    )
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    html = result.stdout
    assert "<details" in html
    assert "agent-memory-references" in html
    assert "mem-1" in html
    assert "field_profile" in html
    assert "task-7" in html
    assert "86%" in html
    assert "沿用坏样本字段口径" in html


def test_agent_message_renderer_outputs_distillation_reference_audit_fields():
    app_js = _read_static("app.js")
    references_start = app_js.index("function agentMemoryReferencesHtml")
    references_end = app_js.index("function agentMessageHtml", references_start)
    script = "\n".join(
        [
            "function escapeHtml(value) {",
            "  return String(value || '').replace(/[&<>\"']/g, (char) => ({",
            "    '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', \"'\": '&#39;',",
            "  }[char]));",
            "}",
            "function formatMemoryConfidence(value) { return String(value); }",
            app_js[references_start:references_end],
            "const html = agentMemoryReferencesHtml([{",
            "  kind: 'distillation',",
            "  id: 'dist-1',",
            "  memory_type: 'field_convention',",
            "  confidence: 'high',",
            "  support_count: 4,",
            "  source_memory_ids: ['mem-1', 'mem-2'],",
            "}]);",
            "process.stdout.write(html);",
        ]
    )
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    html = result.stdout
    assert "进化沉淀" in html
    assert "支持 4" in html
    assert "来源记忆 2" in html
    assert 'data-agent-memory-inline-kind="distillation"' in html


def test_agent_memory_management_view_wires_actions_and_api_paths():
    app_js = _read_static("app.js")
    memory_panel_js = _read_static("js/agent-memory-panel.js")
    index_html = _read_static("index.html")
    styles_css = _read_static("styles.css")

    assert 'id="openGovernanceSettingsButton"' in index_html
    assert 'id="governanceSettingsDialog"' in index_html
    assert 'id="agentMemoryList"' in index_html
    assert 'data-agent-memory-view="raw"' in index_html
    assert 'data-agent-memory-view="distillation"' in index_html
    status_filter_start = index_html.index('id="agentMemoryStatusFilter"')
    status_filter_end = index_html.index("</select>", status_filter_start)
    status_filter = index_html[status_filter_start:status_filter_end]
    assert '<option value="">全部</option>' not in status_filter
    assert '<option value="active">启用</option>' in status_filter
    assert '<option value="deleted">已删除</option>' in index_html
    assert '<option value="rejected">已拒绝</option>' in index_html
    assert 'data-agent-memory-action="inspect"' in memory_panel_js
    assert 'data-agent-memory-action="disable"' in memory_panel_js
    assert 'data-agent-memory-action="enable"' in memory_panel_js
    assert 'data-agent-memory-action="delete"' in memory_panel_js
    assert 'data-agent-memory-action="rollback"' in memory_panel_js
    assert 'data-agent-memory-action="load_more"' in memory_panel_js
    assert 'params.set("limit", String(pageLimit));' in memory_panel_js
    assert 'params.set("offset", String(offset));' in memory_panel_js
    assert "hasMoreItems = Boolean(payload?.has_more)" in memory_panel_js
    assert ".agent-memory-load-more" in styles_css
    assert 'memoryStatus === "active" && !memory.superseded_by' in memory_panel_js

    assert '"api/agent-memory"' in memory_panel_js
    assert '"api/agent-memory/distillations"' in memory_panel_js
    assert '`api/agent-memory/distillations/${encodeURIComponent(memoryId)}`' in memory_panel_js
    assert 'api(`api/agent-memory/distillations/${encodeURIComponent(memoryId)}/rollback`, { method: "POST" })' in memory_panel_js
    assert '`api/agent-memory/${encodeURIComponent(memoryId)}`' in memory_panel_js
    assert 'api(`api/agent-memory/${encodeURIComponent(memoryId)}/disable`, { method: "POST" })' in memory_panel_js
    assert 'api(`api/agent-memory/${encodeURIComponent(memoryId)}/enable`, { method: "POST" })' in memory_panel_js
    assert 'api(`api/agent-memory/${encodeURIComponent(memoryId)}`, { method: "DELETE" })' in memory_panel_js
    assert 'api(`api/tasks/${encodeURIComponent(taskId)}/agent/messages/${encodeURIComponent(messageId)}/memory-references`)' in app_js
    assert 'if (actionId === "agentMemory") setAgentMemoryStatus(message, "error");' in app_js
    assert "function syncAgentMemoryViewControls" in app_js
    assert "function setAgentMemoryViewMode" in app_js
    assert "dialog.agent-memory-dialog" in styles_css
    assert ".governance-settings-dialog" in styles_css
    assert ".governance-settings-shell" in styles_css
    assert ".agent-memory-filter-card" in styles_css
    assert ".agent-memory-workspace" in styles_css
    assert ".agent-memory-detail:empty::before" in styles_css
    assert ".agent-memory-references" in styles_css
    memory_switch_rule = _css_rule(styles_css, ".agent-memory-view-switch")
    assert "border-radius: var(--radius-control)" in memory_switch_rule
    memory_tab_rule = _css_rule(styles_css, ".agent-memory-view-tab")
    assert "border-radius: var(--radius-control)" in memory_tab_rule


def test_agent_memory_delete_keeps_audit_detail_visible():
    memory_panel_js = _read_static("js/agent-memory-panel.js")
    delete_start = memory_panel_js.index("async function remove")
    delete_end = memory_panel_js.index("async function rollbackDistillation", delete_start)
    delete_body = memory_panel_js[delete_start:delete_end]

    assert "renderDetail(payload?.memory || null, payload?.events || [])" in delete_body
    assert "await loadItems()" not in delete_body
    assert "renderItems()" in delete_body


def test_agent_timeline_keeps_messages_in_occurrence_order_around_stage_outputs():
    messages = [
        {"role": "user", "stage": "chat", "content": "开始验证", "metadata": {"intent": "advance"}},
        {"role": "assistant", "stage": "chat", "content": "我将先检查本次验证材料的完备性。", "metadata": {}},
        {
            "role": "assistant",
            "stage": "scan",
            "content": "正在调用材料识别工具 scan_materials：读取材料目录。",
            "metadata": {"tool_call": {"name": "scan_materials", "stage": "scan"}},
        },
        {"role": "assistant", "stage": "scan", "content": "材料完备性检查已完成。", "metadata": {}},
        {
            "role": "assistant",
            "stage": "chat",
            "content": "是否继续执行【模型可复现性验证】？",
            "metadata": {"awaiting_next_stage": "reproducibility"},
        },
        {"role": "user", "stage": "chat", "content": "这些样本来自哪里？", "metadata": {}},
        {"role": "assistant", "stage": "chat", "content": "样本来自材料目录中的建模样本。", "metadata": {}},
        {"role": "user", "stage": "chat", "content": "先继续吧", "metadata": {}},
        {"role": "assistant", "stage": "chat", "content": "收到，我将继续执行模型可复现性验证。", "metadata": {}},
        {"role": "assistant", "stage": "reproducibility", "content": "正在执行 Notebook。", "metadata": {}},
    ]

    assert _agent_timeline_items_for(messages, ["scan", "reproducibility"]) == [
        {
            "type": "messages",
            "contents": [
                "开始验证",
                "我将先检查本次验证材料的完备性。",
                "正在调用材料识别工具 scan_materials：读取材料目录。",
            ],
        },
        {"type": "stage", "stage": "scan"},
        {
            "type": "messages",
            "contents": [
                "材料完备性检查已完成。",
                "是否继续执行【模型可复现性验证】？",
                "这些样本来自哪里？",
                "样本来自材料目录中的建模样本。",
                "先继续吧",
                "收到，我将继续执行模型可复现性验证。",
            ],
        },
        {"type": "stage", "stage": "reproducibility"},
        {"type": "messages", "contents": ["正在执行 Notebook。"]},
    ]


def test_agent_timeline_places_metric_output_before_metric_analysis_after_chat():
    messages = [
        {"role": "assistant", "stage": "reproducibility", "content": "分数一致性检查完成。", "metadata": {}},
        {
            "role": "assistant",
            "stage": "chat",
            "content": "是否继续执行【模型效果&稳定性验证】？",
            "metadata": {"awaiting_next_stage": "metrics"},
        },
        {"role": "user", "stage": "chat", "content": "这个问题严重吗？", "metadata": {}},
        {"role": "assistant", "stage": "chat", "content": "分数不一致会影响上线决策。", "metadata": {}},
        {"role": "user", "stage": "chat", "content": "先继续吧", "metadata": {}},
        {"role": "assistant", "stage": "chat", "content": "收到，我将继续执行模型效果与稳定性验证。", "metadata": {}},
        {"role": "assistant", "stage": "metrics", "content": "效果与稳定性验证已完成。", "metadata": {}},
    ]

    assert _agent_timeline_items_for(messages, ["reproducibility", "metrics"]) == [
        {"type": "stage", "stage": "reproducibility"},
        {
            "type": "messages",
            "contents": [
                "分数一致性检查完成。",
                "是否继续执行【模型效果&稳定性验证】？",
                "这个问题严重吗？",
                "分数不一致会影响上线决策。",
                "先继续吧",
                "收到，我将继续执行模型效果与稳定性验证。",
            ],
        },
        {"type": "stage", "stage": "metrics"},
        {"type": "messages", "contents": ["效果与稳定性验证已完成。"]},
    ]


def test_agent_timeline_keeps_rerun_and_later_outputs_at_the_rerun_position():
    messages = [
        {"role": "assistant", "stage": "word_report_ready", "content": "报告已生成。", "metadata": {}},
        {
            "role": "user",
            "stage": "chat",
            "content": "重新执行一下完备性验证",
            "metadata": {"intent": "rerun_stage", "target_stage": "scan"},
        },
        {
            "role": "assistant",
            "stage": "scan",
            "content": "正在调用材料识别工具 scan_materials：读取材料目录。",
            "metadata": {"tool_call": {"name": "scan_materials", "stage": "scan"}},
        },
        {"role": "assistant", "stage": "scan", "content": "材料完备性检查已完成。", "metadata": {}},
        {"role": "user", "stage": "chat", "content": "继续", "metadata": {"intent": "advance"}},
        {"role": "assistant", "stage": "reproducibility", "content": "正在执行 Notebook。", "metadata": {}},
    ]

    assert _agent_timeline_items_for(messages, ["scan", "reproducibility"]) == [
        {
            "type": "messages",
            "contents": [
                "报告已生成。",
                "重新执行一下完备性验证",
                "正在调用材料识别工具 scan_materials：读取材料目录。",
            ],
        },
        {"type": "stage", "stage": "scan"},
        {"type": "messages", "contents": ["材料完备性检查已完成。", "继续"]},
        {"type": "stage", "stage": "reproducibility"},
        {"type": "messages", "contents": ["正在执行 Notebook。"]},
    ]


def test_agent_timeline_inserts_frozen_snapshot_before_rerun_trigger_message():
    # A rerun captures the current preview as a frozen snapshot; that snapshot
    # must land right before the user message that triggered the rerun so the
    # chart history sits chronologically next to its narration.
    rerun_message_id = "user-rerun-metrics-1"
    messages = [
        {
            "id": "asst-metrics-old",
            "role": "assistant",
            "stage": "metrics",
            "content": "上一次的效果与稳定性分析。",
            "metadata": {},
        },
        {
            "id": rerun_message_id,
            "role": "user",
            "stage": "chat",
            "content": "重新执行第三步",
            "metadata": {"intent": "rerun_stage", "target_stage": "metrics"},
        },
        {
            "id": "asst-metrics-new",
            "role": "assistant",
            "stage": "metrics",
            "content": "新一轮效果与稳定性分析。",
            "metadata": {},
        },
    ]
    snapshots = [
        {
            "triggerMessageId": rerun_message_id,
            "triggerFingerprint": "rerun_stage|metrics|重新执行第三步",
            "stage": "metrics",
            "sectionId": "metricSection",
            "headingHtml": "<h3>指标概览</h3>",
            "label": "指标概览（历史）",
            "contentClassName": "metric-preview",
            "contentHtml": "<div>old chart</div>",
        }
    ]

    items = _agent_timeline_items_for(
        messages,
        ["metrics"],
        frozen_snapshots=snapshots,
        selected_task_id="task-1",
    )

    assert items == [
        {"type": "messages", "contents": ["上一次的效果与稳定性分析。"]},
        {"type": "frozen", "triggerMessageId": rerun_message_id, "stage": "metrics"},
        {"type": "messages", "contents": ["重新执行第三步"]},
        {"type": "stage", "stage": "metrics"},
        {"type": "messages", "contents": ["新一轮效果与稳定性分析。"]},
    ]


def test_agent_timeline_re_anchors_frozen_snapshot_when_optimistic_id_becomes_real():
    # The optimistic rerun message's transient id is replaced by the server id
    # on the next poll. The fingerprint-based fallback must re-anchor the
    # snapshot to the new id so it still renders at the right position.
    real_id = "server-assigned-rerun-id"
    messages = [
        {
            "id": real_id,
            "role": "user",
            "stage": "chat",
            "content": "重新执行第三步",
            "metadata": {"intent": "rerun_stage", "target_stage": "metrics"},
        },
        {
            "id": "asst-metrics-new",
            "role": "assistant",
            "stage": "metrics",
            "content": "新一轮效果与稳定性分析。",
            "metadata": {},
        },
    ]
    snapshots = [
        {
            "triggerMessageId": "optimistic-1234",
            "triggerFingerprint": "rerun_stage|metrics|重新执行第三步",
            "stage": "metrics",
            "sectionId": "metricSection",
            "headingHtml": "<h3>指标概览</h3>",
            "label": "指标概览（历史）",
            "contentClassName": "metric-preview",
            "contentHtml": "<div>old chart</div>",
        }
    ]

    items = _agent_timeline_items_for(
        messages,
        ["metrics"],
        frozen_snapshots=snapshots,
        selected_task_id="task-1",
    )

    # Snapshot is re-anchored to the real message id and inserted right
    # before the rerun message.
    assert items[0] == {"type": "frozen", "triggerMessageId": real_id, "stage": "metrics"}
    assert items[1] == {"type": "messages", "contents": ["重新执行第三步"]}


def test_agent_report_draft_messages_render_in_visible_report_section():
    messages = [
        {"role": "assistant", "stage": "metrics", "content": "效果与稳定性验证完成。", "metadata": {}},
        {
            "role": "assistant",
            "stage": "chat",
            "content": "是否继续执行【报告结论草稿生成】？",
            "metadata": {"awaiting_next_stage": "word_conclusion_draft"},
        },
        {"role": "user", "stage": "chat", "content": "先继续吧", "metadata": {}},
        {
            "role": "assistant",
            "stage": "chat",
            "content": "收到，我将基于已完成的验证结果起草 Word 报告中的三段结论，完成后会等你确认。",
            "metadata": {},
        },
        {
            "role": "assistant",
            "stage": "word_conclusion_draft",
            "content": "",
            "metadata": {"streaming": True},
        },
        {
            "role": "assistant",
            "stage": "word_conclusion_draft",
            "content": "压力测试总结\n压力测试显示模型整体稳定。",
            "metadata": {"draft_values": {"TEXT:pressure_test_summary": "压力测试显示模型整体稳定。"}},
        },
        {
            "role": "assistant",
            "stage": "chat",
            "content": "三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
            "metadata": {"awaiting_confirmation": True},
        },
    ]

    assert _agent_timeline_items_for(messages, ["metrics"]) == [
        {"type": "stage", "stage": "metrics"},
        {
            "type": "messages",
            "contents": [
                "效果与稳定性验证完成。",
                "是否继续执行【报告结论草稿生成】？",
                "先继续吧",
                "收到，我将基于已完成的验证结果起草 Word 报告中的三段结论，完成后会等你确认。",
                "",
                "压力测试总结\n压力测试显示模型整体稳定。",
                "三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
            ],
        },
    ]


def test_agent_report_confirmation_keeps_draft_visible_and_hides_stale_prompt():
    messages = [
        {
            "role": "assistant",
            "stage": "word_conclusion_draft",
            "content": "压力测试总结\n旧草稿。",
            "metadata": {"draft_values": {"TEXT:pressure_test_summary": "旧草稿。"}},
        },
        {
            "role": "assistant",
            "stage": "chat",
            "content": "三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
            "metadata": {"awaiting_confirmation": True},
        },
        {
            "role": "user",
            "stage": "chat",
            "content": "确认",
            "metadata": {"intent": "confirm_report"},
        },
        {
            "role": "assistant",
            "stage": "word_conclusion_confirmed",
            "content": "三段报告结论已确认，将开始生成最终 Word 报告。",
            "metadata": {},
        },
        {
            "role": "assistant",
            "stage": "word_report_ready",
            "content": "报告已生成。右侧步骤里的“预览”可以在线查看 Word，“下载Word”用于下载验证报告，“下载Excel”用于下载指标分析明细。",
            "metadata": {"report_ready": True},
        },
    ]

    visible_messages = _agent_report_messages_for_display(messages)
    assert [message["content"] for message in visible_messages] == [
        "压力测试总结\n旧草稿。",
        "确认",
        "三段报告结论已确认，将开始生成最终 Word 报告。",
        "报告已生成。右侧步骤里的“预览”可以在线查看 Word，“下载Word”用于下载验证报告，“下载Excel”用于下载指标分析明细。",
    ]


def test_agent_report_regenerated_draft_after_confirmation_keeps_history_visible():
    messages = [
        {
            "role": "assistant",
            "stage": "word_conclusion_draft",
            "content": "压力测试总结\n旧草稿。",
            "metadata": {"draft_values": {"TEXT:pressure_test_summary": "旧草稿。"}},
        },
        {
            "role": "assistant",
            "stage": "chat",
            "content": "三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
            "metadata": {"awaiting_confirmation": True},
        },
        {
            "role": "user",
            "stage": "chat",
            "content": "确认",
            "metadata": {"intent": "confirm_report"},
        },
        {
            "role": "assistant",
            "stage": "word_conclusion_confirmed",
            "content": "三段报告结论已确认，将开始生成最终 Word 报告。",
            "metadata": {},
        },
        {
            "role": "user",
            "stage": "chat",
            "content": "重新生成报告",
            "metadata": {"intent": "regenerate_report_draft"},
        },
        {
            "role": "assistant",
            "stage": "word_conclusion_draft",
            "content": "压力测试总结\n新草稿。",
            "metadata": {"draft_values": {"TEXT:pressure_test_summary": "新草稿。"}},
        },
        {
            "role": "assistant",
            "stage": "chat",
            "content": "三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
            "metadata": {"awaiting_confirmation": True},
        },
    ]

    visible_messages = _agent_report_messages_for_display(messages)

    assert [message["content"] for message in visible_messages] == [
        "压力测试总结\n旧草稿。",
        "确认",
        "三段报告结论已确认，将开始生成最终 Word 报告。",
        "重新生成报告",
        "压力测试总结\n新草稿。",
        "三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
    ]


def test_agent_stage_messages_merge_consecutive_duplicate_titles():
    html = _agent_messages_html_for(
        [
            {"role": "assistant", "stage": "chat", "content": "我将先检查材料。", "metadata": {}},
            {"role": "assistant", "stage": "scan", "content": "正在调用材料识别工具。", "metadata": {}},
            {"role": "assistant", "stage": "scan", "content": "材料检查完成。", "metadata": {}},
        ],
        "scan",
    )

    assert html.count("<meta>Agent</meta>") == 1
    assert "<body>我将先检查材料。</body>" in html
    assert "<body>正在调用材料识别工具。</body>" in html
    assert "<body>材料检查完成。</body>" in html


def test_agent_label_uses_validator_aliases_from_workspace_config():
    app_js = _read_static("app.js")
    alias_start = app_js.index("function agentValidatorAlias")
    alias_end = app_js.index("function formatAgentMessageContent", alias_start)
    script = "\n".join(
        [
            # Aliases are sourced from the workspace brand.json (loadBranding sets
            # agentValidatorAliases), not hard-coded in the bundle.
            "let agentValidatorAliases = { '于添': '蛋黄', '张雯萱': '小九' };",
            "let selectedTask = { validator: '于添' };",
            app_js[alias_start:alias_end],
            "const labels = [];",
            "labels.push(agentStageLabel('chat'));",
            "selectedTask = { validator: '张雯萱' };",
            "labels.push(agentStageLabel('chat'));",
            "selectedTask = { validator: '其他人' };",
            "labels.push(agentStageLabel('chat'));",
            "selectedTask = { validator: '  于添  ' };",
            "labels.push(agentStageLabel('chat'));",
            "process.stdout.write(JSON.stringify(labels));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["蛋黄", "小九", "Agent", "蛋黄"]


def test_agent_send_clears_composer_and_renders_user_message_before_network_wait():
    app_js = _read_static("app.js")

    assert "function appendOptimisticAgentUserMessage" in app_js
    assert "function appendOptimisticAgentThinkingMessage" in app_js
    assert "function removeOptimisticAgentMessage" in app_js

    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    send_body = app_js[send_start:send_end]
    post_call = 'api(`api/tasks/${taskId}/agent/messages`'
    assert "const originalValue = input.value;" in send_body
    assert "const optimisticMessage = appendOptimisticAgentUserMessage(content, modelId);" in send_body
    assert "const optimisticThinkingMessage = appendOptimisticAgentThinkingMessage(modelId);" in send_body
    assert "const requestPromise = api(" in send_body
    assert "pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true })" in send_body
    assert send_body.index('input.value = "";') < send_body.index(post_call)
    assert send_body.index("autoGrowComposerInput();") < send_body.index(post_call)
    assert send_body.index("updateAgentSendDisabled();") < send_body.index(post_call)
    assert send_body.index("appendOptimisticAgentUserMessage(content, modelId)") < send_body.index(post_call)
    assert send_body.index("appendOptimisticAgentThinkingMessage(modelId)") < send_body.index(post_call)
    assert "removeOptimisticAgentMessage(optimisticMessage.id);" in send_body
    assert "removeOptimisticAgentMessage(optimisticThinkingMessage.id);" in send_body
    assert "input.value = originalValue;" in send_body


def test_agent_send_without_enabled_model_shows_inline_guidance_before_post():
    app_js = _read_static("app.js")

    helpers_start = app_js.index("function setAgentComposerNotice")
    helpers_end = app_js.index("function renderAgentEffortPreference", helpers_start)
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    script = "\n".join(
        [
            "let selectedTaskId = 'task-1';",
            "let selectedTask = { task_type: 'validation' };",
            "function taskUsesPlanRail(t) { const type = t && t.task_type; return Boolean(type) && type !== 'validation'; }",
            "let llmSettings = { enabled_models: [] };",
            "let apiCalls = 0;",
            "let focusedModel = false;",
            "const AGENT_NO_ENABLED_MODEL_MESSAGE = '请先在设置中配置并启用大模型，再发送 Agent 消息。';",
            "const AGENT_NO_SELECTED_MODEL_MESSAGE = '请先选择一个可用大模型，再发送 Agent 消息。';",
            "const statuses = [];",
            "const input = { value: '开始验证', style: {} };",
            "const modelSelect = { value: '', focus() { focusedModel = true; } };",
            "const notice = { textContent: '', className: '', attrs: {}, setAttribute(name, value) { this.attrs[name] = value; } };",
            "function $(id) { if (id === 'agentComposerInput') return input; if (id === 'agentModelSelect') return modelSelect; if (id === 'agentComposerNotice') return notice; return null; }",
            "function requestAnimationFrame(fn) { fn(); }",
            "function syncAgentComposerClearance() {}",
            "function setActionStatus(message, kind = 'info', detail = '') { statuses.push({ message, kind, detail }); }",
            "function setActionStatusOverride(message, kind = 'info', detail = '') { setActionStatus(message, kind, detail); }",
            "function clearActionStatusOverride() {}",
            "function autoGrowComposerInput() { throw new Error('composer should not clear before model guidance'); }",
            "function updateAgentSendDisabled() { throw new Error('send state should not update before model guidance'); }",
            "async function api() { apiCalls += 1; throw new Error('network should not be called'); }",
            app_js[helpers_start:helpers_end],
            app_js[send_start:send_end],
            "await startAgentValidation();",
            "process.stdout.write(JSON.stringify({ apiCalls, focusedModel, inputValue: input.value, noticeText: notice.textContent, noticeClass: notice.className, status: statuses[statuses.length - 1] }));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["apiCalls"] == 0
    assert payload["inputValue"] == "开始验证"
    assert payload["focusedModel"] is True
    assert "配置并启用大模型" in payload["noticeText"]
    assert "error" in payload["noticeClass"]
    assert payload["status"]["kind"] == "error"
    assert "配置并启用大模型" in payload["status"]["message"]


def test_agent_send_always_requires_llm():
    """Agent mode IS "manual mode whose operator decisions are made by an LLM", so
    it always requires a configured model — no task type bypasses the gate, and
    there is no canned/default agent conversation. The deterministic no-LLM path is
    the separate manual mode."""
    app_js = _read_static("app.js")
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    body = app_js[send_start:send_end]
    assert "const unavailableModelMessage = agentModelUnavailableMessage();" in body
    assert "showAgentModelGuidance(unavailableModelMessage)" in body
    # no task-type bypass of the model-availability gate
    assert "requiresLlm" not in body
    assert "taskUsesPlanRail(selectedTask)" not in body


def test_agent_send_button_switches_to_stop_control_while_agent_is_executing():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    assert 'data-agent-send-state="send"' in index_html
    assert 'class="agent-send-icon agent-send-icon-send"' in index_html
    assert 'class="agent-send-icon agent-send-icon-stop"' in index_html
    assert '<rect x="4" y="4" width="16" height="16" rx="3.4" fill="currentColor" />' in index_html

    assert "function agentSendIsStopMode" in app_js
    assert "function renderAgentSendButtonState" in app_js
    assert "function stopAgentValidation" in app_js
    assert 'api(`api/tasks/${normalizedTaskId}/agent/stop`' in app_js
    assert 'button.dataset.agentSendState = stopMode ? "stop" : "send";' in app_js
    assert 'button.setAttribute("aria-label", stopMode ? "停止当前 Agent 动作" : "发送消息");' in app_js
    assert "button.disabled = stopMode ? false : !input.value.trim();" in app_js

    click_start = app_js.index('$("sendAgentMessageButton").onclick')
    click_end = app_js.index('$("agentComposerInput").addEventListener("keydown"', click_start)
    click_handler = app_js[click_start:click_end]
    assert "if (agentSendIsStopMode())" in click_handler
    assert 'runAction(stopAgentValidation, { actionId: "agent", busyText: "Agent 正在停止..." });' in click_handler
    assert 'runAction(startAgentValidation, { actionId: "agent", busyText: "Agent 正在处理..." });' in click_handler

    keydown_start = app_js.index('$("agentComposerInput").addEventListener("keydown"')
    keydown_end = app_js.index('$("agentComposerInput").addEventListener("input"', keydown_start)
    keydown_handler = app_js[keydown_start:keydown_end]
    assert "if (agentSendIsStopMode()) return;" in keydown_handler

    assert '.agent-send[data-agent-send-state="stop"]' in styles_css
    assert ".agent-send-icon-stop" in styles_css
    assert ".agent-send[data-agent-send-state=\"stop\"] svg" in styles_css
    stop_rule = _css_rule(styles_css, '.agent-send[data-agent-send-state="stop"]')
    assert "background: var(--agent-send-stop-bg)" in stop_rule
    assert "box-shadow: var(--agent-send-stop-shadow)" in stop_rule
    dark_stop_rule = _css_rule(styles_css, 'body[data-theme="dark"] .agent-send[data-agent-send-state="stop"]')
    assert "background: var(--agent-send-stop-bg)" in dark_stop_rule
    assert "box-shadow: var(--agent-send-stop-shadow)" in dark_stop_rule
    dark_stop_hover_rule = _css_rule(
        styles_css, 'body[data-theme="dark"] .agent-send[data-agent-send-state="stop"]:hover:not(:disabled)'
    )
    assert "background: var(--agent-send-stop-bg-hover)" in dark_stop_hover_rule


def test_agent_stop_polling_finishes_when_server_job_is_cancelled_even_if_status_is_mid_stage():
    app_js = _read_static("app.js")
    poll_start = app_js.index("async function pollValidationProgress")
    poll_end = app_js.index("async function validateCurrentTask", poll_start)
    poll_body = app_js[poll_start:poll_end]

    assert "{ stopping = false, background = false } = {}" in poll_body
    assert "if (stopping && !serverBusyAction)" in poll_body
    assert "return polledTask;" in poll_body


def test_agent_send_shows_thinking_message_before_network_wait():
    app_js = _read_static("app.js")
    module_url = (STATIC_DIR / "js" / "agent-conversation-view.js").as_uri()
    helpers_start = app_js.index("function appendOptimisticAgentUserMessage")
    helpers_end = app_js.index("function renderAgentTimeline", helpers_start)
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    script = "\n".join(
        [
            f"import {{ agentMessageIsAdvanceIntent }} from {json.dumps(module_url)};",
            "let selectedTaskId = 'task-1';",
            "let selectedTask = { task_type: 'validation' };",
            "function taskUsesPlanRail(t) { const type = t && t.task_type; return Boolean(type) && type !== 'validation'; }",
            "let agentMessages = [];",
            "let lastAgentRenderSignature = null;",
            "const input = { value: '开始', style: {}, classList: { toggle() {} } };",
            "const modelSelect = { value: 'model-1' };",
            "const renderSnapshots = [];",
            "function $(id) { return id === 'agentComposerInput' ? input : modelSelect; }",
            "function autoGrowComposerInput() {}",
            "function updateAgentSendDisabled() {}",
            "function setActionStatus() {}",
            "function renderAgentConversation() { renderSnapshots.push(agentMessages.map((message) => ({ role: message.role, content: message.content, metadata: message.metadata || {} }))); }",
            "function agentModelUnavailableMessage() { return ''; }",
            "function showAgentModelGuidance() { return false; }",
            "function setAgentComposerNotice() {}",
            "function agentModelConfigurationErrorMessage() { return ''; }",
            "function agentEffort() { return 'high'; }",
            "function agentAcceptanceModeValue() { return 'normal'; }",
            "function pollAgentMessagesUntilSettled() { return Promise.resolve(); }",
            "let resolveApi;",
            "async function api() { return await new Promise((resolve) => { resolveApi = resolve; }); }",
            app_js[helpers_start:helpers_end],
            app_js[send_start:send_end],
            "const sendPromise = startAgentValidation();",
            "await new Promise((resolve) => setTimeout(resolve, 0));",
            "const pendingMessages = agentMessages.map((message) => ({ role: message.role, content: message.content, metadata: message.metadata || {} }));",
            "resolveApi({ status: 'message_saved', messages: [{ role: 'user', stage: 'chat', content: '开始', metadata: {} }, { role: 'assistant', stage: 'chat', content: '收到。', metadata: { streaming: false } }] });",
            "await sendPromise;",
            "process.stdout.write(JSON.stringify({ inputValue: input.value, pendingMessages, finalMessages: agentMessages, renderSnapshots }));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["inputValue"] == ""
    assert payload["pendingMessages"][0]["role"] == "user"
    assert payload["pendingMessages"][0]["content"] == "开始"
    assert payload["pendingMessages"][1]["role"] == "assistant"
    assert payload["pendingMessages"][1]["content"] == ""
    assert payload["pendingMessages"][1]["metadata"]["streaming"] is True
    assert payload["pendingMessages"][1]["metadata"]["optimistic"] is True
    assert payload["finalMessages"][-1]["content"] == "收到。"


def test_agent_send_polls_streaming_messages_before_network_response_finishes():
    app_js = _read_static("app.js")
    module_url = (STATIC_DIR / "js" / "agent-conversation-view.js").as_uri()
    helpers_start = app_js.index("async function pollAgentMessagesUntilSettled")
    helpers_end = app_js.index("async function startAgentValidation", helpers_start)
    message_helpers_start = app_js.index("function appendOptimisticAgentUserMessage")
    message_helpers_end = app_js.index("function renderAgentTimeline", message_helpers_start)
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    script = "\n".join(
        [
            f"import {{ agentMessageIsAdvanceIntent }} from {json.dumps(module_url)};",
            "const AGENT_STREAM_POLL_INTERVAL_MS = 1;",
            "let selectedTaskId = 'task-1';",
            "let selectedTask = { task_type: 'validation' };",
            "function taskUsesPlanRail(t) { const type = t && t.task_type; return Boolean(type) && type !== 'validation'; }",
            "let agentMessages = [];",
            "let lastAgentRenderSignature = null;",
            "let pollCount = 0;",
            "let firstPollOptions = null;",
            "const input = { value: '解释一下', style: {}, classList: { toggle() {} } };",
            "const modelSelect = { value: 'model-1' };",
            "function $(id) { return id === 'agentComposerInput' ? input : modelSelect; }",
            "function autoGrowComposerInput() {}",
            "function updateAgentSendDisabled() {}",
            "function setActionStatus() {}",
            "function renderAgentConversation() {}",
            "function agentModelUnavailableMessage() { return ''; }",
            "function showAgentModelGuidance() { return false; }",
            "function setAgentComposerNotice() {}",
            "function agentModelConfigurationErrorMessage() { return ''; }",
            "function agentEffort() { return 'high'; }",
            "function agentAcceptanceModeValue() { return 'normal'; }",
            "function sleep() { return new Promise((resolve) => setTimeout(resolve, 0)); }",
            "async function loadAgentMessages(_taskId, options = {}) { pollCount += 1; firstPollOptions ||= options; }",
            "let resolveApi;",
            "async function api() { return await new Promise((resolve) => { resolveApi = resolve; }); }",
            app_js[message_helpers_start:message_helpers_end],
            app_js[helpers_start:helpers_end],
            app_js[send_start:send_end],
            "const sendPromise = startAgentValidation();",
            "await new Promise((resolve) => setTimeout(resolve, 10));",
            "const polledBeforeResponse = pollCount > 0;",
            "resolveApi({ status: 'message_saved', messages: [{ role: 'user', stage: 'chat', content: '解释一下', metadata: {} }, { role: 'assistant', stage: 'chat', content: '收到。', metadata: { streaming: false } }] });",
            "await sendPromise;",
            "process.stdout.write(JSON.stringify({ polledBeforeResponse, pollCount, firstPollOptions }));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["polledBeforeResponse"] is True
    assert payload["pollCount"] >= 1
    assert payload["firstPollOptions"] == {"preserveOptimistic": True}


def test_agent_composer_model_and_effort_preferences_survive_refresh_until_user_changes_them():
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")

    assert 'export const agentComposerPreferenceStorageKey = "marvis_agent_composer_preferences";' in state_js
    assert "const agentComposerPreferences = restoreAgentComposerPreferences();" in app_js
    assert 'let agentSelectedModelId = agentComposerPreferences.model_id || "";' in app_js
    assert 'let agentSelectedEffort = agentComposerPreferences.effort || "high";' in app_js
    assert 'localStorage.getItem(agentComposerPreferenceStorageKey)' in app_js
    assert 'localStorage.setItem(agentComposerPreferenceStorageKey' in app_js
    assert "function normalizeAgentEffort" in app_js
    assert "function renderAgentEffortPreference" in app_js

    model_options_start = app_js.index("function renderAgentModelOptions")
    model_options_end = app_js.index("function renderAgentConversation", model_options_start)
    model_options = app_js[model_options_start:model_options_end]
    assert "saveAgentComposerPreferences()" not in model_options
    assert 'agentSelectedModelId = "";' not in model_options
    assert "agentSelectedModelId = select.value;" not in model_options
    assert "renderAgentEffortPreference()" in app_js

    conversation_start = app_js.index("function renderAgentConversation")
    conversation_end = app_js.index("function agentStructuralSignature", conversation_start)
    conversation = app_js[conversation_start:conversation_end]
    assert conversation.index("renderAgentModelOptions();") < conversation.index("if (!showConversation)")
    assert conversation.index("renderAgentEffortPreference();") < conversation.index("if (!showConversation)")

    model_change_start = app_js.index('$("agentModelSelect").onchange')
    model_change_end = app_js.index('$("sendAgentMessageButton").onclick', model_change_start)
    model_change = app_js[model_change_start:model_change_end]
    assert "agentSelectedModelId = event.target.value;" in model_change
    # Changes are persisted via the per-task dispatcher; the global save call
    # is reserved for the fallback path when no task is selected.
    assert "persistCurrentAgentComposerPreference({ model_id: agentSelectedModelId });" in model_change

    assert '$("agentEffortSelect").onchange = (event) =>' in app_js
    effort_change_start = app_js.index('$("agentEffortSelect").onchange')
    effort_change_end = app_js.index('$("sendAgentMessageButton").onclick', effort_change_start)
    effort_change = app_js[effort_change_start:effort_change_end]
    assert "agentSelectedEffort = normalizeAgentEffort(event.target.value);" in effort_change
    assert "persistCurrentAgentComposerPreference({ effort: agentSelectedEffort });" in effort_change


def test_agent_composer_preferences_are_kept_per_task_in_local_storage():
    # Mode / model / effort must persist per task so that two tasks can
    # remember different configurations. The global preference acts as the
    # seed when a task is opened for the first time.
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")

    assert 'export const agentTaskComposerStorageKey = "marvis_agent_task_composer_preferences";' in state_js
    assert "function loadAgentTaskComposerOverrides" in app_js
    assert "function persistAgentTaskComposerOverrides" in app_js
    assert "function getAgentTaskComposerOverride" in app_js
    assert "function updateAgentTaskComposerOverride" in app_js
    assert "function applyAgentTaskComposerPreferences" in app_js
    assert "function resetAgentComposerToGlobalDefaults" in app_js

    # selectTask wires the per-task overrides into the live composer state.
    select_start = app_js.index("function selectTask(task)")
    select_end = app_js.index("function deselectCurrentTask", select_start)
    select_body = app_js[select_start:select_end]
    assert "applyAgentTaskComposerPreferences(task.id);" in select_body

    # Deselecting a task falls back to the global seed so the composer
    # state is coherent if a fresh task is selected next.
    deselect_start = app_js.index("function deselectCurrentTask()")
    deselect_end = app_js.index("function renderMetricPreview", deselect_start)
    deselect_body = app_js[deselect_start:deselect_end]
    assert "resetAgentComposerToGlobalDefaults();" in deselect_body

    # Dispatcher: persist to per-task override when a task is selected;
    # only fall back to the legacy global save when no task is active.
    dispatcher_start = app_js.index("function persistCurrentAgentComposerPreference")
    dispatcher_end = dispatcher_start + 400
    dispatcher = app_js[dispatcher_start:dispatcher_end]
    assert "updateAgentTaskComposerOverride(selectedTaskId, patch);" in dispatcher
    assert "saveAgentComposerPreferences();" in dispatcher

    # Acceptance mode change handler must also route through the per-task
    # dispatcher rather than the global-only saver.
    accept_change_start = app_js.index('$("agentAcceptanceModeSelect").onchange')
    accept_change_end = app_js.index("function blurChipSelectIfFocused", accept_change_start)
    accept_change = app_js[accept_change_start:accept_change_end]
    assert "persistCurrentAgentComposerPreference({ acceptance_mode: agentAcceptanceMode });" in accept_change


def test_agent_composer_acceptance_mode_selector_controls_auto_accept_payload():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_static("styles.css")

    assert 'id="agentAcceptanceModeSelect"' in index_html
    assert 'class="agent-composer-chip-icon agent-acceptance-icon-default"' in index_html
    assert 'class="agent-composer-chip-icon agent-acceptance-icon-auto"' in index_html
    assert 'value="normal">默认权限</option>' in index_html
    assert 'value="auto_accept">自动审查</option>' in index_html
    assert index_html.index('id="agentAcceptanceModeSelect"') < index_html.index('id="agentModelSelect"')

    assert "function normalizeAgentAcceptanceMode" in app_js
    assert 'let agentAcceptanceMode = agentComposerPreferences.acceptance_mode || "normal";' in app_js
    assert "acceptance_mode: normalizeAgentAcceptanceMode(agentAcceptanceMode)" in app_js
    assert "function renderAgentAcceptanceModePreference" in app_js
    assert "function agentAcceptanceModeValue" in app_js
    assert "renderAgentAcceptanceModePreference();" in app_js
    assert '$("agentAcceptanceModeSelect").onchange = (event) =>' in app_js
    assert '"agentAcceptanceModeSelect"' in app_js
    assert "acceptance_mode: agentAcceptanceModeValue()" in app_js

    mode_rule_start = styles_css.index(".agent-composer-acceptance")
    mode_rule_end = styles_css.index(".agent-composer-model", mode_rule_start)
    mode_rules = styles_css[mode_rule_start:mode_rule_end]
    assert "var(--danger)" in mode_rules
    assert '[data-acceptance-mode="normal"]' in mode_rules
    assert '[data-acceptance-mode="normal"] select' in mode_rules
    assert '[data-acceptance-mode="auto_accept"]' in mode_rules
    assert ".agent-acceptance-icon-default" in mode_rules
    assert ".agent-acceptance-icon-auto" in mode_rules

    toolbar_start = styles_css.index(".agent-composer-toolbar {")
    toolbar_end = styles_css.index("}", toolbar_start)
    toolbar_rules = styles_css[toolbar_start:toolbar_end]
    assert "flex-wrap: nowrap;" in toolbar_rules


def test_agent_composer_select_accent_does_not_stick_after_native_dropdown_closes():
    styles_css = _read_static("styles.css")

    accent_start = styles_css.index("/* Agent execution mode chip.")
    accent_end = styles_css.index("\n.agent-composer-chip-caret {", accent_start)
    accent_rules = styles_css[accent_start:accent_end]

    sticky_focus_selectors = [
        ".agent-composer-acceptance:focus-within",
        ".agent-composer-model:focus-within",
        ".agent-composer-effort:focus-within",
        'body[data-theme="dark"] .agent-composer-model:focus-within',
        'body[data-theme="dark"] .agent-composer-effort:focus-within',
    ]
    for selector in sticky_focus_selectors:
        assert selector not in accent_rules

    assert ".agent-composer-model:hover" in accent_rules
    assert ".agent-composer-effort:hover" in accent_rules
    assert ".agent-composer-acceptance:hover" in accent_rules
    focus_start = styles_css.index(".agent-composer-chip select:focus-visible {")
    focus_end = styles_css.index("}", focus_start)
    focus_rule = styles_css[focus_start:focus_end]
    assert "outline: none;" in focus_rule
    assert "outline-offset" not in focus_rule
    assert "#8b5cf6" not in focus_rule
    assert "#eab308" not in focus_rule
    assert "var(--danger)" not in focus_rule


def test_agent_composer_selects_are_not_rebuilt_on_every_poll_render():
    app_js = _read_static("app.js")

    model_options_start = app_js.index("function renderAgentModelOptions")
    model_options_end = app_js.index("function renderAgentEffortPreference", model_options_start)
    model_options = app_js[model_options_start:model_options_end]

    assert "agentModelOptionsSignature" in model_options
    assert "const preferred = agentPreferredModelId(enabledModels);" in model_options
    assert "agentSelectedModelId || llmSettings.default_model_id" not in model_options
    assert "preferredStillAvailable" in model_options
    assert "select.dataset.agentModelOptionsSignature === signature" in model_options
    assert "select.innerHTML = \"\";" in model_options
    assert (
        model_options.index("select.dataset.agentModelOptionsSignature === signature")
        < model_options.index("select.innerHTML = \"\";")
    )

    effort_start = app_js.index("function renderAgentEffortPreference")
    effort_end = app_js.index("function requestAgentConversationScrollToLatest", effort_start)
    effort_renderer = app_js[effort_start:effort_end]
    assert "if (select.value !== agentSelectedEffort)" in effort_renderer


def test_agent_model_preference_ignores_disabled_saved_model():
    app_js = _read_static("app.js")
    preference_start = app_js.index("function agentPreferredModelId")
    preference_end = app_js.index("function renderAgentEffortPreference", preference_start)
    script = "\n".join(
        [
            "let agentSelectedModelId = 'disabled-model';",
            "let llmSettings = { default_model_id: 'enabled-default', enabled_models: [] };",
            app_js[preference_start:preference_end],
            "const enabledModels = [{ model_id: 'enabled-default' }, { model_id: 'other-model' }];",
            "const first = agentPreferredModelId(enabledModels);",
            "llmSettings.default_model_id = 'also-disabled';",
            "const second = agentPreferredModelId(enabledModels);",
            "agentSelectedModelId = 'other-model';",
            "const third = agentPreferredModelId(enabledModels);",
            "process.stdout.write(JSON.stringify([first, second, third]));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["enabled-default", "enabled-default", "other-model"]


def test_agent_assistant_messages_render_markdown_safely():
    app_js = _read_static("app.js")
    render_agent_js = _read_static("js/render-agent.js")
    styles_css = _read_static("styles.css")

    assert 'import { renderAgentMarkdown } from "./js/render-agent.js";' in app_js
    assert "export function renderAgentMarkdown" in render_agent_js
    assert "export function renderMarkdownInline" in render_agent_js
    assert "export function renderMarkdownInlineText" in render_agent_js
    assert 'formatAgentMessageContent(agentVisibleContent(message), { markdown: role === "assistant" })' in app_js
    assert 'import { escapeHtml } from "./ui-utils.js";' in render_agent_js
    assert ".agent-markdown" in styles_css
    assert ".agent-markdown ul" in styles_css
    assert ".agent-markdown code" in styles_css
    code_start = styles_css.index(".agent-markdown code {")
    code_end = styles_css.index("}", code_start)
    code_rule = styles_css[code_start:code_end]
    assert "color: var(--text);" in code_rule
    assert "color: var(--accent);" not in code_rule
    em_start = styles_css.index(".agent-markdown em {")
    em_end = styles_css.index("}", em_start)
    em_rule = styles_css[em_start:em_end]
    assert "color: inherit;" in em_rule
    assert "color: var(--text-secondary);" not in em_rule
    assert ".agent-markdown a" in styles_css
    assert "renderMarkdownInlineText(segment)" in render_agent_js
    assert "export function isSafeMarkdownHref" in render_agent_js
    assert "export function markdownAnchorHtml" in render_agent_js
    assert ".replace(/\\[([^\\]\\n]+)\\]\\(" not in render_agent_js
    assert ".replace(/_([^_]+)_/g" not in render_agent_js
    assert "isMarkdownBoundary" in render_agent_js


def test_agent_markdown_renders_highlighted_code_blocks():
    styles_css = _read_static("styles.css")
    html = _render_agent_markdown(
        "\n".join(
            [
                "```python",
                "def score(df):",
                "    threshold = 0.5",
                "    return df[\"prob\"] > threshold  # pass",
                "```",
            ]
        )
    )

    assert '<code class="language-python">' in html
    assert '<span class="agent-code-token keyword">def</span>' in html
    assert '<span class="agent-code-token function">score</span>' in html
    assert '<span class="agent-code-token number">0.5</span>' in html
    assert '<span class="agent-code-token string">&quot;prob&quot;</span>' in html
    assert '<span class="agent-code-token comment"># pass</span>' in html
    assert "&gt; threshold" in html
    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))
    for token in [
        "--agent-code-token-keyword",
        "--agent-code-token-function",
        "--agent-code-token-string",
        "--agent-code-token-number",
    ]:
        assert token in root_vars
        assert token in dark_vars
    assert "color: var(--agent-code-token-keyword)" in _css_rule(
        styles_css, ".agent-code-token.keyword"
    )
    assert "color: var(--agent-code-token-function)" in _css_rule(
        styles_css, ".agent-code-token.function"
    )
    assert "color: var(--agent-code-token-string)" in _css_rule(
        styles_css, ".agent-code-token.string"
    )
    assert "color: var(--agent-code-token-number)" in _css_rule(
        styles_css, ".agent-code-token.number"
    )
    assert "color: var(--text-muted)" in _css_rule(styles_css, ".agent-code-token.comment")


def test_agent_markdown_preserves_ordered_section_numbers_after_blank_lines():
    render_agent_js = _read_static("js/render-agent.js")

    renderer_start = render_agent_js.index("export function renderAgentMarkdown")
    renderer_end = render_agent_js.index("export function normalizeMarkdownCodeLanguage", renderer_start)
    renderer = render_agent_js[renderer_start:renderer_end]

    assert r"^\s*(\d+)\.\s+(.+)$" in renderer
    assert 'openList("ol", ordered[1])' in renderer
    assert "startAttr" in renderer
    assert 'html.push(`<${type}${startAttr}>`);' in renderer
    assert "renderMarkdownInline(ordered[2])" in renderer


def test_agent_markdown_renders_pipe_tables():
    styles_css = _read_static("styles.css")
    html = _render_agent_markdown(
        "\n".join(
            [
                "| 文件类型 | 描述 | 信贷风控适用性 |",
                "|:---|:---|:---|",
                "| **PFA** | 基于 JSON/YAML 的模型交换标准 | 在部分场景中可被审计 |",
                "| Pickle (.pkl) | Python 原生序列化格式 | **不推荐上线部署** |",
            ]
        )
    )

    assert "<table>" in html
    assert "<thead><tr>" in html
    assert "<tbody>" in html
    assert "<th>文件类型</th>" in html
    assert "<td><strong>PFA</strong></td>" in html
    assert "<strong>不推荐上线部署</strong>" in html
    assert "|:---|:---|:---|" not in html
    assert ".agent-markdown table" in styles_css
    assert ".agent-markdown th" in styles_css
    assert ".agent-markdown td" in styles_css


def test_agent_markdown_rejects_unsafe_links_and_escapes_html():
    html = _render_agent_markdown(
        "[bad](javascript:alert(1)) [data](data:text/html,test) "
        "[phish](//evil.test/steal) "
        "[ok](https://example.test) <img src=x onerror=alert(1)>"
    )

    assert "javascript:" not in html
    assert "data:text/html" not in html
    # protocol-relative URLs ("//evil.test") resolve to an external https origin
    # in the browser, so they must not pass the same-origin "/" allowance: the URL
    # is dropped entirely while the link label survives as plain text.
    assert "evil.test" not in html
    assert "phish" in html
    assert '<a href="https://example.test"' in html
    assert "<img" not in html
    assert "&lt;img" in html


def test_branding_normalizer_rejects_unsafe_asset_urls():
    script = "\n".join(
        [
            "import { isSafeAssetUrl, normalizeBranding, normalizeValidatorAliases } from "
            "'./marvis/static/js/branding.js';",
            "const probes = {",
            "  absolute: isSafeAssetUrl('/branding/assets/logo.png'),",
            "  relative: isSafeAssetUrl('static/brand/logo.png'),",
            "  https: isSafeAssetUrl('https://cdn.test/logo.png'),",
            "  protocolRelative: isSafeAssetUrl('//evil.test/x'),",
            "  javascript: isSafeAssetUrl('javascript:alert(1)'),",
            "  data: isSafeAssetUrl('data:text/html,x'),",
            "};",
            "const fallback = normalizeBranding({ logoUrl: 'javascript:alert(1)', "
            "workspaceLogoUrl: 'data:image/png,x', faviconUrl: '//evil.test/f.ico' });",
            "const aliases = normalizeValidatorAliases({ '  A  ': '  a  ', B: '', C: 5, D: 'd' });",
            "process.stdout.write(JSON.stringify({ probes, logoUrl: fallback.logoUrl, "
            "workspaceLogoUrl: fallback.workspaceLogoUrl, faviconUrl: fallback.faviconUrl, "
            "aliases, brandingAliases: fallback.validatorAliases }));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)

    assert data["probes"] == {
        "absolute": True,
        "relative": True,
        "https": True,
        "protocolRelative": False,
        "javascript": False,
        "data": False,
    }
    # Unsafe URLs are dropped and the safe defaults are kept (never the injection).
    assert "javascript:" not in data["logoUrl"]
    assert "data:" not in data["workspaceLogoUrl"]
    assert "evil.test" not in data["faviconUrl"]
    # Validator aliases are trimmed; empty / non-string entries are dropped.
    assert data["aliases"] == {"A": "a", "D": "d"}
    assert data["brandingAliases"] == {}


def test_frozen_snapshot_sanitizer_strips_scripts_and_event_handlers():
    app_js = _read_static("app.js")
    start = app_js.index("function stripIdsFromHtml(")
    end = app_js.index("\n}", start)
    body = app_js[start:end]
    # Frozen snapshots must be inert: the sanitizer drops <script> and inline on*
    # handlers in addition to id attributes (defense-in-depth re-insertion guard).
    assert 'querySelectorAll("script")' in body
    assert ".remove()" in body
    assert 'removeAttribute("id")' in body
    assert "/^on/i" in body


def test_step_rail_bottom_padding_is_visually_balanced():
    styles_css = _read_static("styles.css")

    rail_start = styles_css.index(".progress-rail {")
    rail_end = styles_css.index("}", rail_start)
    rail_rule = styles_css[rail_start:rail_end]

    assert "padding: 12px 12px 8px;" in rail_rule


# ====== Polling render signature guards ======


def _slice_function(app_js: str, signature: str) -> str:
    """Return the source of a top-level function declared with `signature`.

    All top-level functions in app.js close with a `}` at column 0, so we
    find the next column-0 `}` line after the signature. This is robust
    against destructured default params (which would confuse naive brace
    counting started at the first `{`).
    """
    start = app_js.index(signature)
    needle = "\n}"
    end = app_js.index(needle, start)
    return app_js[start : end + len(needle)]


def test_poll_validation_progress_does_not_call_render_all_in_loop():
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "async function pollValidationProgress")
    assert "renderAll();" not in body, (
        "pollValidationProgress() must not call renderAll() in its steady-state "
        "loop; use renderChangedValidationViews() instead so unchanged regions "
        "are not repainted every second."
    )
    assert "renderChangedValidationViews" in body


def test_action_status_writer_is_idempotent():
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function setActionStatus")
    assert "renderSignatures.actionStatus" in body, (
        "setActionStatus() must short-circuit on identical (message, kind, "
        "detail) writes so the top status pill does not flicker during polling."
    )
    # The guard must be an early return before the DOM is touched.
    guard_index = body.index("renderSignatures.actionStatus")
    pill_write_index = body.index('$("actionStatus")')
    assert guard_index < pill_write_index


def test_metric_preview_skips_unchanged_payload():
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function renderMetricPreview")
    guard_index = body.index("renderSignatures.metricPreview")
    html_index = body.index('"metricPreview").innerHTML')
    assert guard_index < html_index, (
        "renderMetricPreview() must short-circuit on identical payloads before "
        "any innerHTML assignment so animated metric cards do not replay every "
        "polling tick."
    )


def test_workflow_stepper_guard_refreshes_elapsed_time():
    """The skip-path must still tick elapsed seconds on running steps."""
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function renderWorkflowStepper")
    assert "refreshWorkflowStepperElapsedTimes" in body, (
        "renderWorkflowStepper() must call refreshWorkflowStepperElapsedTimes() "
        "so running steps' elapsed seconds tick even when the structural "
        "signature guard skips a rebuild."
    )
    guard_idx = body.index("renderSignatures.workflowStepper === nextSignature")
    refresh_idx = body.index("refreshWorkflowStepperElapsedTimes")
    assert guard_idx < refresh_idx, (
        "The elapsed-time refresher must be reachable from the skip branch "
        "(i.e. positioned after the signature comparison)."
    )


def test_step_fingerprint_excludes_clock_time():
    """stepFingerprint must NOT include Date.now() or it defeats the guard."""
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function stepFingerprint")
    assert "Date.now()" not in body, (
        "stepFingerprint() must not bake clock time into the structural "
        "signature; otherwise every poll tick triggers a full rebuild and "
        "replays animations."
    )


def test_reproducibility_guard_lives_in_render_signatures():
    """The reproducibility chart's structural guard must be unified into the
    renderSignatures cache (not dropped onto element.dataset where evidenceEmpty
    would wipe it). This is the structural piece of the fix."""
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function renderReproducibilityEvidence")
    assert "renderSignatures.reproducibilityEvidence" in body
    assert "renderSignatures.reproducibilityTaskId" in body
    assert "renderSignatures.reproducibilityAnimatedTaskId" in body
    # The OLD dataset-based guard must be gone — it was wipe-prone by design.
    assert "element.dataset.reproducibilitySignature" not in body
    assert "element.dataset.reproducibilityTaskId" not in body
    # And the empty branch must not gate on notebookReproducibilityComplete,
    # which was the original regression vector during running notebook steps.
    empty_branch_start = body.index("Object.keys(summary).length === 0")
    empty_branch_end = body.index("evidenceEmpty(", empty_branch_start)
    empty_branch = body[empty_branch_start:empty_branch_end]
    assert "notebookReproducibilityComplete" not in empty_branch


def test_reproducibility_animation_replays_only_on_first_render_per_task():
    """Animation policy: shouldAnimatePrecisionChart is true only when the
    per-task animated marker doesn't match — never on every rebuild."""
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function renderReproducibilityEvidence")
    assert (
        "shouldAnimatePrecisionChart = renderSignatures.reproducibilityAnimatedTaskId !== taskId"
        in body
    )
    # And the marker must be set after a render that actually animated, so the
    # next rebuild for the same task uses data-animation="none".
    assert "renderSignatures.reproducibilityAnimatedTaskId = taskId" in body


def _run_node_capture_json(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


_BROWSER_STUBS = """
// ---- Browser environment stubs for Node-side execution ----
// app.js is written for the browser; under Node we stub the minimum surface
// the module-level code touches (localStorage, document, window, RAF, etc).
// The reproducibilitySummary element is wrapped in a Proxy so every
// innerHTML write is captured for assertion.

globalThis.__writes = [];
const __storageBacking = new Map();
globalThis.localStorage = {
  getItem(key) { return __storageBacking.has(key) ? __storageBacking.get(key) : null; },
  setItem(key, value) { __storageBacking.set(key, String(value)); },
  removeItem(key) { __storageBacking.delete(key); },
  clear() { __storageBacking.clear(); },
};

const __elements = new Map();
function __makeMockElement() {
  const inner = {
    dataset: {},
    className: '',
    innerHTML: '',
    textContent: '',
    value: '',
    checked: false,
    disabled: false,
    hidden: false,
    scrollTop: 0,
    scrollHeight: 0,
    clientHeight: 0,
    children: [],
    childNodes: [],
    style: new Proxy({}, { get: () => () => {}, set: () => true }),
    classList: { add() {}, remove() {}, toggle() { return false; }, contains() { return false; } },
  };
  return new Proxy(inner, {
    get(target, prop) {
      if (prop in target) return target[prop];
      // Any other access (querySelector, addEventListener, appendChild, ...)
      // returns a no-op function. Returning null for non-call sites would
      // crash callers that immediately .invoke() the result.
      return (...args) => {
        if (prop === 'querySelector' || prop === 'closest') return null;
        if (prop === 'querySelectorAll' || prop === 'getElementsByTagName') return [];
        if (prop === 'getBoundingClientRect') return { width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0 };
        return undefined;
      };
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });
}

function __getOrMakeElement(id) {
  if (!__elements.has(id)) {
    const base = __makeMockElement();
    if (id === 'reproducibilitySummary') {
      const wrapped = new Proxy(base, {
        get(target, prop) { return target[prop]; },
        set(target, prop, value) {
          if (prop === 'innerHTML') globalThis.__writes.push(value);
          target[prop] = value;
          return true;
        },
      });
      __elements.set(id, wrapped);
    } else {
      __elements.set(id, base);
    }
  }
  return __elements.get(id);
}

globalThis.document = new Proxy({
  getElementById: __getOrMakeElement,
  querySelector(sel) {
    if (sel && sel.startsWith('#')) return __getOrMakeElement(sel.slice(1));
    return null;
  },
  querySelectorAll() { return []; },
  addEventListener() {},
  removeEventListener() {},
  body: __makeMockElement(),
  activeElement: null,
}, {
  get(target, prop) {
    if (prop in target) return target[prop];
    return () => undefined;
  },
});

globalThis.window = new Proxy({
  addEventListener() {},
  removeEventListener() {},
  matchMedia() { return { matches: false, addEventListener() {}, removeEventListener() {} }; },
}, { get(target, prop) { return prop in target ? target[prop] : () => undefined; } });

globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });
globalThis.MutationObserver = class { observe() {} disconnect() {} };
globalThis.AbortController = globalThis.AbortController || class { constructor() { this.signal = {}; } abort() {} };
globalThis.fetch = async () => ({ ok: true, json: async () => ({}), text: async () => '' });
"""


def test_reproducibility_render_skips_replay_and_disables_animation_on_rebuild():
    """End-to-end behavioral test: drive populated_A → empty → populated_B
    through the real renderReproducibilityEvidence + signature cache and
    confirm:

    1. First populated render writes innerHTML with an animated chart.
    2. Empty payload is preserved — NO additional innerHTML write.
    3. Second populated render (different rows) rebuilds, but the chart now
       carries data-animation="none" so the bars don't visually replay.
    """
    app_js = _read_static("app.js")
    # Cut the module-level boot block (DOM listeners, restorePet*, etc.). The
    # marker is the first `document.addEventListener("mousedown", ...)` block
    # bound to the agent-composer-chip blur logic.
    boot_marker = 'document.addEventListener(\n  "mousedown"'
    boot_idx = app_js.index(boot_marker)
    app_js = app_js[:boot_idx].replace('from "./js/', 'from "./marvis/static/js/')

    populated_a = {
        "summary": {"status": "ok", "mismatch_count": 0, "max_abs_diff": 0.00001},
        "sample_size": 100,
        "seed": 42,
        "rows": [
            {
                "row_index": 1,
                "score_code_model": 0.5,
                "score_submitted_pmml": 0.5,
                "abs_diff": 0.0,
                "matched": True,
            },
        ],
    }
    populated_b = {
        "summary": {"status": "ok", "mismatch_count": 1, "max_abs_diff": 0.00009},
        "sample_size": 100,
        "seed": 42,
        "rows": [
            {
                "row_index": 1,
                "score_code_model": 0.6,
                "score_submitted_pmml": 0.6000001,
                "abs_diff": 0.0000001,
                "matched": False,
            },
        ],
    }
    empty = {"summary": {}, "rows": []}

    test_driver = "\n".join(
        [
            "selectedTaskId = 'task-A';",
            "selectedTask = { id: 'task-A', status: 'running', active_job_kind: 'pipeline' };",
            f"renderReproducibilityEvidence({json.dumps(populated_a)});",
            "const writesAfter1 = globalThis.__writes.length;",
            f"renderReproducibilityEvidence({json.dumps(empty)});",
            "const writesAfter2 = globalThis.__writes.length;",
            f"renderReproducibilityEvidence({json.dumps(populated_b)});",
            "const writesAfter3 = globalThis.__writes.length;",
            "const writes = globalThis.__writes;",
            "process.stdout.write(JSON.stringify({",
            "  writesAfter1,",
            "  writesAfter2,",
            "  writesAfter3,",
            "  firstWriteHadAnimation: writes[0] ? !writes[0].includes('data-animation=\"none\"') : null,",
            "  lastWriteHasNoAnimation: writes.length >= 2 ? writes[writes.length - 1].includes('data-animation=\"none\"') : null,",
            "  animatedTaskAfter: renderSignatures.reproducibilityAnimatedTaskId,",
            "  signatureTaskAfter: renderSignatures.reproducibilityTaskId,",
            "}));",
        ]
    )

    script = _BROWSER_STUBS + "\n" + app_js + "\n" + test_driver
    data = _run_node_capture_json(script)

    assert data["writesAfter1"] == 1, data
    assert data["firstWriteHadAnimation"] is True, data
    assert data["writesAfter2"] == 1, data
    assert data["writesAfter3"] == 2, data
    assert data["lastWriteHasNoAnimation"] is True, data
    assert data["animatedTaskAfter"] == "task-A", data
    assert data["signatureTaskAfter"] == "task-A", data


def test_reproducibility_render_handles_task_switch_animation_policy():
    """Task switching covers two doc-mandated policies in one flow:

    - Switching to a different task re-allows the entry animation.
    - Empty evidence on a NEW task (no prior populated render for that task)
      falls through to the placeholder — it does NOT preserve the previous
      task's chart.
    """
    app_js = _read_static("app.js")
    boot_marker = 'document.addEventListener(\n  "mousedown"'
    app_js = app_js[: app_js.index(boot_marker)].replace(
        'from "./js/', 'from "./marvis/static/js/'
    )

    populated = {
        "summary": {"status": "ok", "mismatch_count": 0, "max_abs_diff": 0.00001},
        "sample_size": 100,
        "seed": 42,
        "rows": [
            {
                "row_index": 1,
                "score_code_model": 0.5,
                "score_submitted_pmml": 0.5,
                "abs_diff": 0.0,
                "matched": True,
            },
        ],
    }
    empty = {"summary": {}, "rows": []}

    test_driver = "\n".join(
        [
            "selectedTaskId = 'task-A';",
            "selectedTask = { id: 'task-A', status: 'running' };",
            f"renderReproducibilityEvidence({json.dumps(populated)});",
            "const writesAfter_A_populated = globalThis.__writes.length;",
            "const animatedTaskAfter_A = renderSignatures.reproducibilityAnimatedTaskId;",
            # Switch to task B and render the SAME evidence shape — should
            # rebuild (different task) AND re-enable animation.
            "selectedTaskId = 'task-B';",
            "selectedTask = { id: 'task-B', status: 'running' };",
            f"renderReproducibilityEvidence({json.dumps(populated)});",
            "const writesAfter_B_populated = globalThis.__writes.length;",
            "const animatedTaskAfter_B = renderSignatures.reproducibilityAnimatedTaskId;",
            "const writeForB = globalThis.__writes[writesAfter_B_populated - 1];",
            # Switch to task C and feed empty evidence — placeholder should
            # appear; previous chart must not be preserved across tasks.
            "selectedTaskId = 'task-C';",
            "selectedTask = { id: 'task-C', status: 'running' };",
            f"renderReproducibilityEvidence({json.dumps(empty)});",
            "const reproElText = document.getElementById('reproducibilitySummary').textContent;",
            "const signatureAfter_C = renderSignatures.reproducibilityEvidence;",
            "process.stdout.write(JSON.stringify({",
            "  writesAfter_A_populated,",
            "  writesAfter_B_populated,",
            "  animatedTaskAfter_A,",
            "  animatedTaskAfter_B,",
            "  bWriteHadAnimation: writeForB ? !writeForB.includes('data-animation=\"none\"') : null,",
            "  reproElText,",
            "  signatureAfter_C,",
            "}));",
        ]
    )
    script = _BROWSER_STUBS + "\n" + app_js + "\n" + test_driver
    data = _run_node_capture_json(script)

    assert data["writesAfter_A_populated"] == 1, data
    assert data["animatedTaskAfter_A"] == "task-A", data
    # Switching to task-B with same payload still rebuilds (signature carries
    # taskId) AND plays the animation again for the new task.
    assert data["writesAfter_B_populated"] == 2, data
    assert data["animatedTaskAfter_B"] == "task-B", data
    assert data["bWriteHadAnimation"] is True, data
    # Empty evidence on a brand-new task clears to placeholder text.
    assert "暂无分数一致性证据" in data["reproElText"], data
    assert data["signatureAfter_C"] == "", data


def test_substep_elapsed_text_lives_in_its_own_span():
    """Substep elapsed text needs a stable key so the refresher can target it."""
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function renderNotebookStepRail")
    assert "data-step-elapsed-key" in body, (
        "Notebook substeps must render elapsed text in a span with a stable "
        "data-step-elapsed-key so refreshWorkflowStepperElapsedTimes() can "
        "update it in place without rebuilding the substep DOM."
    )


def test_computing_metrics_has_one_status_phrase():
    app_js = _read_static("app.js")
    poll_body = _slice_function(app_js, "async function pollValidationProgress")
    assert "验证进行中：" not in poll_body, (
        "pollValidationProgress() must not write its own 验证进行中：... copy; "
        "taskActionStatusSnapshot() owns the per-status phrase."
    )
    snapshot_body = _slice_function(app_js, "function taskActionStatusSnapshot")
    assert "指标概览进行中" in snapshot_body


def test_writing_artifacts_idle_shows_metrics_complete():
    """writing_artifacts is a dual-meaning state.

    Backend sets task.status=writing_artifacts the moment metrics finishes,
    BEFORE any report job is dispatched. The top status bar must therefore
    distinguish "metrics done, awaiting next step" from "report job running"
    using task.active_job_kind. Showing 报告输出进行中 while the user has not
    started the report yet misrepresents stage 3 as still in progress.
    """
    app_js = _read_static("app.js")
    stopped_body = _slice_function(app_js, "function taskStopped")
    snapshot_body = _slice_function(app_js, "function taskActionStatusSnapshot")
    script = "\n".join(
        [
            "let selectedTask = null;",
            stopped_body,
            snapshot_body,
            "const idle = taskActionStatusSnapshot({"
            " status: 'writing_artifacts', active_job_kind: null });",
            "const reportBusy = taskActionStatusSnapshot({"
            " status: 'writing_artifacts', active_job_kind: 'report' });",
            "const metricsRunning = taskActionStatusSnapshot({"
            " status: 'computing_metrics', active_job_kind: 'metrics' });",
            "process.stdout.write(JSON.stringify({ idle, reportBusy, metricsRunning }));",
        ]
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    snapshots = json.loads(result.stdout)
    assert snapshots["idle"]["kind"] == "success", (
        "writing_artifacts with no active report job means metrics just "
        "finished; the top status must read as a completion, not 进行中."
    )
    assert "完成" in snapshots["idle"]["message"]
    assert "进行中" not in snapshots["idle"]["message"]
    assert snapshots["reportBusy"] == {
        "message": "报告输出进行中。",
        "kind": "busy",
    }
    assert snapshots["metricsRunning"] == {
        "message": "指标概览进行中。",
        "kind": "busy",
    }


def test_task_stopped_uses_structured_stopped_field_only():
    app_js = _read_static("app.js")
    stopped_body = _slice_function(app_js, "function taskStopped")
    script = "\n".join(
        [
            "let selectedTask = null;",
            stopped_body,
            "const structured = taskStopped({ stopped: true, status_message: '普通状态' });",
            "const legacy = taskStopped({ status_message: '已停止当前动作' });",
            "const active = taskStopped({ stopped: false, status_message: '普通状态' });",
            "process.stdout.write(JSON.stringify({ structured, legacy, active }));",
        ]
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    states = json.loads(result.stdout)
    assert states == {"structured": True, "legacy": False, "active": False}


def test_writing_artifacts_status_tone_idle_vs_report_busy():
    """taskStatusTone must mirror taskActionStatusSnapshot's idle/busy split.

    The sidebar pill colors writing_artifacts as if it were running even when
    no report job is dispatched, repeating the same dual-meaning confusion.
    """
    app_js = _read_static("app.js")
    stopped_body = _slice_function(app_js, "function taskStopped")
    tone_body = _slice_function(app_js, "function statusTone")
    task_tone_body = _slice_function(app_js, "function taskStatusTone")
    script = "\n".join(
        [
            "let selectedTask = null;",
            stopped_body,
            tone_body,
            task_tone_body,
            "const idle = taskStatusTone({"
            " status: 'writing_artifacts', active_job_kind: null });",
            "const reportBusy = taskStatusTone({"
            " status: 'writing_artifacts', active_job_kind: 'report' });",
            "process.stdout.write(JSON.stringify({ idle, reportBusy }));",
        ]
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    tones = json.loads(result.stdout)
    assert tones["idle"] == "success", (
        "writing_artifacts with no active report job should color the pill "
        "as a completion, not as a running task."
    )
    assert tones["reportBusy"] == "run"


def test_manual_stage_finish_scrolls_to_section_top():
    """In manual mode each stage handler must jump to the top of the
    newly-rendered section once execution succeeds, matching the behavior
    of clicking the corresponding row in the right-side workflow stepper.
    The helper is gated to manual mode so agent-mode runs do not fight the
    agent auto-scroll-to-bottom behavior.
    """
    app_js = _read_static("app.js")

    helper_body = _slice_function(app_js, "function scrollToManualWorkflowSection")
    assert "selectedTaskIsAgentMode()" in helper_body, (
        "scrollToManualWorkflowSection must short-circuit in agent mode."
    )
    assert "workflowSteps.find" in helper_body, (
        "Helper must resolve the section id via workflowSteps so it stays "
        "in sync with the stepper definition."
    )
    assert "scrollStepTarget" in helper_body, (
        "Helper must reuse scrollStepTarget so manual-finish scroll uses the "
        "same scrollIntoView call as the stepper click."
    )

    scan_body = _slice_function(app_js, "async function scanCurrentTask")
    assert 'scrollToManualWorkflowSection("scan")' in scan_body

    notebook_body = _slice_function(app_js, "async function validateCurrentTask")
    assert 'scrollToManualWorkflowSection("notebook")' in notebook_body

    metrics_body = _slice_function(app_js, "async function generateMetrics")
    assert 'scrollToManualWorkflowSection("metrics")' in metrics_body

    report_body = _slice_function(app_js, "async function generateReport")
    assert 'scrollToManualWorkflowSection("report")' in report_body


def test_agent_auto_scroll_drops_distance_threshold():
    """The follow-mode rewrite must replace the old 120px sticky window with
    a strict at-bottom check. The threshold made tiny upward scrolls feel
    glued to the bottom because the typewriter kept yanking the viewport
    back down.
    """
    app_js = _read_static("app.js")
    assert "AGENT_AUTO_SCROLL_STICKY_PX" not in app_js, (
        "The 120px sticky threshold must be removed so any upward scroll "
        "disengages auto-follow."
    )
    body = _slice_function(app_js, "function requestAgentConversationScrollToLatest")
    assert "distanceFromBottom" not in body
    assert "agentAutoScrollFollows" in body, (
        "requestAgentConversationScrollToLatest must consult the follow-mode "
        "state instead of a distance threshold."
    )


def test_task_switch_resets_agent_follow_and_typing_state():
    """Both agentAutoScrollFollows and agentTypingState are module-level
    globals. Without an explicit reset on task switch, a stale `false` flag
    from task A would stop task B's typewriter from auto-following, and a
    lingering typing entry from A could re-reveal (or contaminate via a
    shared messageId) on B's panel.
    """
    app_js = _read_static("app.js")
    prep = _slice_function(app_js, "function prepareResultScrollRestoreForTask")
    assert "agentAutoScrollFollows = true;" in prep, (
        "Task switch must reset agentAutoScrollFollows so a stale `false` "
        "from the previous task does not freeze the next task's follow-mode."
    )
    select = _slice_function(app_js, "function selectTask")
    assert "resetAgentTypingState();" in select, (
        "Task identity change must wipe typewriter state so a still-revealing "
        "message from the previous task can't re-reveal on the new task."
    )


def test_agent_typewriter_resume_after_completion_seeds_visible():
    """If the server flips a message id back to streaming=true after it had
    already finished, the next render must NOT clear the visible content and
    re-type from byte 0. The seeded-visible path must engage instead.
    """
    app_js = _read_static("app.js")
    streaming_body = _slice_function(app_js, "function agentMessageIsStreaming")
    visible_body = _slice_function(app_js, "function agentVisibleContent")
    sched_body = _slice_function(app_js, "function scheduleAgentTyping")

    script = "\n".join(
        [
            "const window = { setTimeout: () => 1, clearTimeout: () => {} };",
            "const agentTypingState = new Map();",
            "const agentTypingCompleted = new Map();",
            "let agentTypingTimer = null;",
            "let lastAgentRenderSignature = null;",
            "const AGENT_TYPEWRITER_INTERVAL_MS = 12;",
            "function renderAgentConversation() {}",
            "function tickAgentTyping() {}",
            streaming_body,
            sched_body,
            visible_body,
            # Phase 1: streaming completes naturally.
            "agentTypingState.set('m1', { visible: 'hello world', target: 'hello world' });",
            "const settled = { id: 'm1', role: 'assistant',"
            " content: 'hello world', metadata: { streaming: false } };",
            "agentVisibleContent(settled);",
            "const completedAfterSettle = agentTypingCompleted.get('m1');",
            # Phase 2: same id resumes streaming with more content.
            "const resumed = { id: 'm1', role: 'assistant',"
            " content: 'hello world and then some', metadata: { streaming: true } };",
            "const visibleOnResume = agentVisibleContent(resumed);",
            "const typingOnResume = agentTypingState.get('m1');",
            "process.stdout.write(JSON.stringify({"
            " completedAfterSettle,"
            " visibleOnResume,"
            " seededVisible: typingOnResume ? typingOnResume.visible : null,"
            " targetOnResume: typingOnResume ? typingOnResume.target : null,"
            "}));",
        ]
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    out = json.loads(result.stdout)
    assert out["completedAfterSettle"] == "hello world", (
        "A naturally-settled streaming message must store the content it "
        "settled with so a later resume can seed visible with the already-"
        "shown bytes."
    )
    assert out["seededVisible"] == "hello world", (
        "Resume-after-completion must seed visible with the already-shown "
        "content; otherwise the rendered text clears to empty and re-types "
        "from byte 0."
    )
    assert out["targetOnResume"] == "hello world and then some"
    assert out["visibleOnResume"] == "hello world", (
        "First render after resume returns the seeded visible (not empty), "
        "so the user sees the existing text plus a typewriter tail."
    )


def test_user_scroll_input_required_to_flip_follow_mode():
    """Without a recent wheel/touch input, recomputeAgentAutoScrollFollow must
    be a no-op. This is what stops the typewriter's own scrollTo from flipping
    follow-mode back on the frame after the user disengaged it.
    """
    app_js = _read_static("app.js")
    recompute_body = _slice_function(app_js, "function recomputeAgentAutoScrollFollow")
    note_body = _slice_function(app_js, "function noteAgentUserScrollInput")
    assert "lastUserScrollInputAt" in recompute_body, (
        "recomputeAgentAutoScrollFollow must consult the last user-input "
        "timestamp so it ignores programmatic typewriter/restore scrolls."
    )
    assert "AGENT_USER_SCROLL_INPUT_WINDOW_MS" in recompute_body
    assert "lastUserScrollInputAt = performance.now();" in note_body
    assert 'document.addEventListener("wheel", noteAgentUserScrollInput' in app_js
    assert 'document.addEventListener("touchstart", noteAgentUserScrollInput' in app_js
    assert 'document.addEventListener("touchmove", noteAgentUserScrollInput' in app_js


def test_recompute_agent_auto_scroll_follow_ignores_negative_distance():
    """Overscroll bounce and mid-mount panels can produce a negative or zero
    distance value. `distance <= 2` alone is trivially true in those cases
    and would spuriously re-engage follow-mode.
    """
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function recomputeAgentAutoScrollFollow")
    assert "scrollHeight <= scrollContent.clientHeight" in body, (
        "When the panel cannot scroll yet (content fits within viewport), "
        "recompute must short-circuit instead of flipping follow-mode."
    )
    assert "if (distance < 0) return;" in body, (
        "Negative distance (overscroll bounce, sub-pixel rounding past max) "
        "must not flip follow-mode to true."
    )


def test_scroll_to_manual_workflow_section_captures_task_id():
    """The rAF inside scrollToManualWorkflowSection runs one frame after
    scheduling; the user may have switched tasks in between. Capturing
    selectedTaskId at scheduling time and bailing on mismatch prevents the
    panel from jumping to the wrong section.
    """
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function scrollToManualWorkflowSection")
    assert "const targetTaskId = selectedTaskId;" in body
    assert "if (selectedTaskId !== targetTaskId) return;" in body


def test_recompute_agent_auto_scroll_follow_toggles_on_position():
    """Any wheel that leaves the bottom must drop follow-mode to false, even
    if the user only nudged up by a few pixels. Returning to the bottom (or
    within sub-pixel rounding of it) must re-engage follow-mode so the next
    typewriter tick resumes streaming.
    """
    app_js = _read_static("app.js")
    recompute_body = _slice_function(app_js, "function recomputeAgentAutoScrollFollow")

    script = "\n".join(
        [
            "let scrollContent = { scrollHeight: 1000, clientHeight: 400, scrollTop: 0 };",
            "function $(id) { return id === 'resultScrollContent' ? scrollContent : null; }",
            "let isAgent = true;",
            "function selectedTaskIsAgentMode() { return isAgent; }",
            "let agentAutoScrollFollows = true;",
            "const AGENT_AUTO_SCROLL_BOTTOM_TOLERANCE_PX = 2;",
            "const AGENT_USER_SCROLL_INPUT_WINDOW_MS = 250;",
            "let lastUserScrollInputAt = 0;",
            "const performance = { now: () => 100 };",
            recompute_body,
            "lastUserScrollInputAt = 100;",  # pretend wheel just fired
            "scrollContent.scrollTop = 600;",
            "recomputeAgentAutoScrollFollow();",
            "const atBottom = agentAutoScrollFollows;",
            "scrollContent.scrollTop = 590;",
            "recomputeAgentAutoScrollFollow();",
            "const slightlyAbove = agentAutoScrollFollows;",
            "scrollContent.scrollTop = 200;",
            "recomputeAgentAutoScrollFollow();",
            "const farAbove = agentAutoScrollFollows;",
            "scrollContent.scrollTop = 599.4;",
            "recomputeAgentAutoScrollFollow();",
            "const subpixelRounding = agentAutoScrollFollows;",
            "isAgent = false;",
            "scrollContent.scrollTop = 0;",
            "recomputeAgentAutoScrollFollow();",
            "const nonAgentLeft = agentAutoScrollFollows;",
            "process.stdout.write(JSON.stringify({"
            " atBottom, slightlyAbove, farAbove, subpixelRounding, nonAgentLeft"
            " }));",
        ]
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    out = json.loads(result.stdout)
    assert out["atBottom"] is True
    assert out["slightlyAbove"] is False, (
        "Even a small upward scroll must disengage follow-mode — the user "
        "explicitly asked for 'even a tiny bit up' to stop the auto-pull."
    )
    assert out["farAbove"] is False
    assert out["subpixelRounding"] is True, (
        "Sub-pixel rounding (<= 2px) must still count as at-bottom so the "
        "typewriter's own scroll-to-bottom does not flip follow-mode off."
    )
    assert out["nonAgentLeft"] is True, (
        "recompute must be a no-op outside agent mode; otherwise switching "
        "tasks would wipe agent follow state."
    )


def test_handle_result_scroll_recomputes_follow_mode():
    """The follow-mode toggle must hang off the existing scroll listener so
    both user wheels and the typewriter's own scrollTo land in the same
    recompute path.
    """
    app_js = _read_static("app.js")
    body = _slice_function(app_js, "function handleResultScroll")
    assert "recomputeAgentAutoScrollFollow();" in body


def test_agent_typewriter_continues_after_stream_ends():
    """The agent typewriter must not dump remaining content when the server
    flips metadata.streaming to false. Short messages caught up before the
    flag flipped, so the dump only showed on long replies — exactly the
    "一开始一个字一个字浮现，后面一大段突然全部蹦出来" symptom.
    """
    app_js = _read_static("app.js")
    streaming_body = _slice_function(app_js, "function agentMessageIsStreaming")
    visible_body = _slice_function(app_js, "function agentVisibleContent")
    sched_body = _slice_function(app_js, "function scheduleAgentTyping")
    tick_body = _slice_function(app_js, "function tickAgentTyping")

    script = "\n".join(
        [
            "const window = { setTimeout: () => 1, clearTimeout: () => {} };",
            "const agentTypingState = new Map();",
            "const agentTypingCompleted = new Map();",
            "let agentTypingTimer = null;",
            "let lastAgentRenderSignature = null;",
            "const AGENT_TYPEWRITER_INTERVAL_MS = 12;",
            "const AGENT_TYPEWRITER_CHARS_PER_TICK = 2;",
            "const AGENT_TYPEWRITER_CATCHUP_TICKS = 15;",
            "function renderAgentConversation() {}",
            streaming_body,
            sched_body,
            tick_body,
            visible_body,
            "const longContent = 'A'.repeat(200);",
            "const streaming = { id: 'm1', role: 'assistant',"
            " content: longContent.slice(0, 10),"
            " metadata: { streaming: true } };",
            "const first = agentVisibleContent(streaming);",
            "agentTypingState.get('m1').visible = longContent.slice(0, 5);",
            "const settled = { id: 'm1', role: 'assistant',"
            " content: longContent,"
            " metadata: { streaming: false } };",
            "const afterStreamEnd = agentVisibleContent(settled);",
            "const stateAfterStreamEnd = agentTypingState.has('m1');",
            "tickAgentTyping();",
            "const afterTick = agentTypingState.get('m1');",
            "process.stdout.write(JSON.stringify({"
            " first,"
            " afterStreamEnd,"
            " stateAfterStreamEnd,"
            " visibleAfterTick: afterTick ? afterTick.visible : null,"
            " targetAfterTick: afterTick ? afterTick.target : null,"
            "}));",
        ]
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    out = json.loads(result.stdout)
    assert out["stateAfterStreamEnd"] is True, (
        "Typing state must survive metadata.streaming flipping to false "
        "until the visible cursor catches up to the full content."
    )
    assert out["afterStreamEnd"] == "A" * 5, (
        "When streaming ends while the typewriter still has content left "
        "to reveal, agentVisibleContent must return the partial visible "
        "string, not the full content."
    )
    assert out["targetAfterTick"] == "A" * 200
    assert out["visibleAfterTick"] is not None
    assert 5 < len(out["visibleAfterTick"]) < 200, (
        "Tick after stream end must advance the typewriter — not jump "
        "straight to the end and not stay frozen."
    )


def test_agent_typewriter_catches_up_long_backlog_quickly():
    """The typewriter must accelerate when far behind so a multi-thousand
    character backlog still finishes in a fraction of a second instead of
    taking 30+ seconds at one-char-per-tick.
    """
    app_js = _read_static("app.js")
    streaming_body = _slice_function(app_js, "function agentMessageIsStreaming")
    visible_body = _slice_function(app_js, "function agentVisibleContent")
    sched_body = _slice_function(app_js, "function scheduleAgentTyping")
    tick_body = _slice_function(app_js, "function tickAgentTyping")

    script = "\n".join(
        [
            "const window = { setTimeout: () => 1, clearTimeout: () => {} };",
            "const agentTypingState = new Map();",
            "const agentTypingCompleted = new Map();",
            "let agentTypingTimer = null;",
            "let lastAgentRenderSignature = null;",
            "const AGENT_TYPEWRITER_INTERVAL_MS = 12;",
            "const AGENT_TYPEWRITER_CHARS_PER_TICK = 2;",
            "const AGENT_TYPEWRITER_CATCHUP_TICKS = 15;",
            "function renderAgentConversation() {}",
            streaming_body,
            sched_body,
            tick_body,
            visible_body,
            "const bigContent = 'B'.repeat(3000);",
            "const message = { id: 'm1', role: 'assistant',"
            " content: bigContent,"
            " metadata: { streaming: true } };",
            "agentVisibleContent(message);",
            "let ticks = 0;",
            "while (agentTypingState.get('m1').visible.length"
            " < agentTypingState.get('m1').target.length) {",
            "  tickAgentTyping();",
            "  ticks += 1;",
            "  if (ticks > 500) break;",
            "}",
            "process.stdout.write(JSON.stringify({ ticks }));",
        ]
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    out = json.loads(result.stdout)
    assert out["ticks"] <= 100, (
        "A 3000-char backlog must drain in roughly a second; the catch-up "
        "branch should consume a fraction of the backlog per tick instead "
        "of advancing by AGENT_TYPEWRITER_CHARS_PER_TICK alone."
    )
    assert out["ticks"] >= 30, (
        "The catch-up must not collapse to a single-tick dump — that would "
        "reproduce the original bug."
    )


def test_base_pet_mood_writing_artifacts_idle_is_complete():
    """basePetMoodFromTask reads writing_artifacts as running even when the
    report job has not started yet, leaving the pet in the running animation
    after stage 3 finishes.
    """
    app_js = _read_static("app.js")
    mood_body = _slice_function(app_js, "function basePetMoodFromTask")

    # Drop the writing_artifacts entry from the running fallback so the test
    # cleanly captures the regression we want to prevent.
    assert (
        '["running", "computing_metrics", "writing_artifacts"].includes(status)'
        not in mood_body
    ), (
        "basePetMoodFromTask must not classify writing_artifacts as running "
        "based on status alone; the busy check on active_job_kind already "
        "covers the genuinely-running case."
    )
    assert (
        '["scanned", "executed", "writing_artifacts"].includes(status)'
        in mood_body
    ), (
        "writing_artifacts without an active job must fall into the same "
        "complete bucket as scanned and executed."
    )
