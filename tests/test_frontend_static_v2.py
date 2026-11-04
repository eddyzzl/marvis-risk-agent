"""Frontend compatibility contracts and isolated production-function behavior.

CSS/markup assertions pin visual and packaging constraints, not running flows.
Node tests execute the actual JavaScript; real creation DOM event coverage lives
in test_frontend_creation_dom_behavior.py. Reviewed source-only behavior proxies
and their replacements are recorded in the F13 disposition document.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.javascript_source import session_projection_fixture
from tests.javascript_source import slice_function as _slice_function
from tests.static_stylesheets import read_browser_stylesheets

STATIC_DIR = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _with_workbench_context(script: str, app_js: str) -> str:
    """Supply the real workbench helpers to explicitly isolated Node fixtures."""
    batch_js = _read_static("js/validation-batch.js")
    states = {
        "selectedTask": "null",
        "agentMessages": "[]",
        "projectedValidationChildTask": "null",
        "projectedValidationChildTaskId": "''",
        "agentBatchAutoRunGeneration": "0",
        "validationBatchTaskType": "'validation_batch'",
        # These fixtures do not open a report editor. Draft-save lifecycle is
        # exercised separately with populated state in the behavior regressions.
        "reportDraftState": "{ get: () => null }",
        "historicalReplayController": "{ render: () => {} }",
        "riskContextPanel": "{ render: () => {} }",
        "institutionAdapterPanel": "{ render: () => {} }",
        "operationsExtensionsPanel": "{ render: () => {} }",
        "collectionWorkspaceController": "{ render: () => {} }",
        "strategyHistoryMappingPanel": "{ selectTask: async () => {} }",
    }
    prefix = [
        f"let {name} = {value};"
        for name, value in states.items()
        if not re.search(rf"\b(?:let|const|var)\s+{name}\b", script)
    ]
    scope_url = (STATIC_DIR / "js/task-request-scope.js").as_uri()
    if not re.search(r"\b(?:let|const|var)\s+taskRequests\b", script):
        prefix.insert(0, f"import {{ createTaskRequestScope }} from {json.dumps(scope_url)};")
        prefix.append("const taskRequests = createTaskRequestScope();")
    activity_url = (STATIC_DIR / "js/task-activity.js").as_uri()
    prefix.insert(0, f"import {{ createTaskActivityOwner }} from {json.dumps(activity_url)};")
    if not re.search(r"\b(?:let|const|var)\s+taskActivities\b", script):
        prefix.append("const taskActivities = createTaskActivityOwner();")
    for name in ("renderBusyActivity", "claimBusy", "releaseBusy"):
        if not re.search(rf"\bfunction\s+{name}\b", script):
            prefix.append(_slice_function(app_js, f"function {name}("))
    for name in ("renderWorkflowStepper", "renderPetState", "updateAgentSendDisabled"):
        if not re.search(rf"\bfunction\s+{name}\b", script):
            prefix.append(f"function {name}() {{}}")
    for source, names in (
        (
            app_js,
            ("setSelectedTask", "workbenchTask", "workbenchTaskId", "isWorkbenchTaskId", "invalidateAgentBatchAutoRun"),
        ),
        (batch_js, ("textValue", "isValidationBatchTask", "usesAgentValidationWorkbench")),
    ):
        for name in names:
            if not re.search(rf"\bfunction\s+{name}\b", script):
                prefix.append(_slice_function(source, f"function {name}("))
    return session_projection_fixture("\n".join([*prefix, script]))


def _read_browser_css() -> str:
    return read_browser_stylesheets(STATIC_DIR)


def _css_rule(css: str, selector: str) -> str:
    marker = f"{selector} {{"
    cursor = 0
    rules: list[str] = []
    while (start := css.find(marker, cursor)) >= 0:
        line_start = css.rfind("\n", 0, start) + 1
        if css[line_start:start].strip():
            cursor = start + len(marker)
            continue
        end = css.index("}", start)
        rules.append(css[start:end])
        cursor = end + 1
    if not rules:
        raise ValueError(f"CSS selector not found: {selector}")
    return "\n".join(rules)


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
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mean: 0.532；std: 0.042" in result.stdout
    assert '{"mean":0.532' not in result.stdout


def test_v2_artifact_preview_uses_tabular_numeric_metrics():
    css = _read_browser_css()

    assert '[data-v2-artifact-view="true"] :is(.metrics-preview, .dataset-preview table)' in css
    assert 'font-feature-settings: "tnum"' in css
    assert '[data-v2-artifact-view="true"] .metrics-preview td' in css
    assert "text-align: right" in css


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
    script = session_projection_fixture(script)
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
    script = session_projection_fixture(script)
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
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_button_confirmation_audit_message_is_hidden_from_visible_conversation():
    messages = [
        {"id": "gate", "role": "assistant", "content": "请确认"},
        {
            "id": "audit-confirm",
            "role": "user",
            "content": "确认",
            "metadata": {"display_in_timeline": False, "ui_action": "confirm_gate"},
        },
        {"id": "ack", "role": "assistant", "content": "收到确认，开始执行拼接。"},
    ]

    displayed = _agent_report_messages_for_display(messages)

    assert [message["id"] for message in displayed] == ["gate", "ack"]


def _render_agent_markdown(markdown: str) -> str:
    script = "\n".join(
        [
            "import { renderAgentMarkdown } from './marvis/static/js/render-agent.js';",
            f"process.stdout.write(renderAgentMarkdown({json.dumps(markdown, ensure_ascii=False)}));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _safe_markdown_hrefs(hrefs: list[str]) -> list[bool]:
    script = "\n".join(
        [
            "import { isSafeMarkdownHref } from './marvis/static/js/render-agent.js';",
            f"const hrefs = {json.dumps(hrefs, ensure_ascii=False)};",
            "process.stdout.write(JSON.stringify(hrefs.map(isSafeMarkdownHref)));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _safe_api_hrefs(hrefs: list[str]) -> list[str]:
    script = "\n".join(
        [
            "import { safeSameOriginApiHref } from './marvis/static/js/url-safety.js';",
            f"const hrefs = {json.dumps(hrefs, ensure_ascii=False)};",
            "process.stdout.write(JSON.stringify(hrefs.map(safeSameOriginApiHref)));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _agent_report_download_html(reports: list[dict]) -> str:
    app_js = _read_static("app.js")
    start = app_js.index("function agentMessageReportDownloadHtml")
    end = app_js.index("\nif (typeof document", start)
    presenter = app_js[start:end]
    script = "\n".join(
        [
            "import { safeSameOriginApiHref } from './marvis/static/js/url-safety.js';",
            "import { escapeHtml } from './marvis/static/js/ui-utils.js';",
            "const selectedTask = { task_type: 'modeling' };",
            presenter,
            f"const reports = {json.dumps(reports, ensure_ascii=False)};",
            "process.stdout.write(agentMessageReportDownloadHtml({ metadata: { report_downloads: reports } }));",
        ]
    )
    script = session_projection_fixture(script)
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
    notebook_start = app_js.index("function notebookStepTone")
    notebook_end = app_js.index("function plannedReproducibilitySteps", notebook_start)
    script = "\n".join(
        [
            "const selectedTaskId = 'task-1';",
            f"const selectedTask = {json.dumps(task, ensure_ascii=False)};",
            f"taskSession.panels.notebook.steps = {json.dumps(notebook_steps, ensure_ascii=False)};",
            "const workflowSteps = [{id:'scan'}, {id:'notebook'}, {id:'metrics'}, {id:'report'}];",
            "function taskBusyAction() { return null; }",
            "function taskServerBusyAction() { return null; }",
            app_js[failure_start:failure_end],
            *(_slice_function(app_js, f"function {name}(") for name in (
                # Batch 4 (2026-10-14 redesign): workflowStepStatus parks a stopped task's step.
                "taskStopped",
                "workflowIndex", "taskFailureStepId", "taskRunningStepId", "workflowStepStatus",
            )),
            app_js[notebook_start:notebook_end],
            "const statuses = workflowSteps.map((_, index) => workflowStepStatus(index, workflowIndex(selectedTask.status)));",
            "process.stdout.write(JSON.stringify(statuses));",
        ]
    )
    script = session_projection_fixture(script)
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
    script = session_projection_fixture(script)
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
    script = session_projection_fixture(script)
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
    script = session_projection_fixture(script)
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
    # Batch 4 (2026-10-14 redesign): the row/title helpers resolve through js/task-status.js.
    label_start = app_js.index("function taskPlanWorkflowStatusSnapshot")
    label_end = app_js.index("function notebookReproducibilityComplete", label_start)
    script = "\n".join(
        [
            'import { actionStatusPresentation, resolveTaskStatus } from "./marvis/static/js/task-status.js";',
            f"const task = {json.dumps(task, ensure_ascii=False)};",
            "function taskUsesPlanRail(task) { return ['data_join', 'feature_analysis', 'modeling', 'strategy', 'vintage', 'portfolio'].includes(task?.task_type); }",
            "const planRailController = { statusSnapshot() { return null; } };",
            "function workflowStatusSnapshot() { return null; }",
            app_js[stopped_start:stopped_end],
            app_js[pill_start:pill_end],
            app_js[label_start:label_end],
            f"const pill = actionStatusPill({json.dumps(action_message, ensure_ascii=False)}, {json.dumps(action_kind)}, task);",
            "process.stdout.write(JSON.stringify({",
            "  rowLabel: taskStatusLabel(task),",
            "  rowTone: taskStatusTone(task),",
            # Batch 4: the presentation also carries key/face/dot/checker; the pins compare the word and tone.
            "  heroPill: pill && { label: pill.label, tone: pill.tone },",
            "}));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)




def test_word_report_preview_dialog_uses_task_dialog_backdrop():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    assert 'id="wordPreviewDialog"' in index_html
    assert 'class="task-dialog word-preview-dialog"' in index_html
    assert 'id="wordPreviewFrame"' in index_html
    assert ".task-dialog::backdrop" in styles_css
    assert ".word-preview-dialog" in styles_css
    assert ".word-preview-frame" in styles_css


def test_step_rail_narrow_layout_keeps_titles_horizontal_and_stacks_report_actions():
    styles_css = _read_browser_css()

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




def test_plan_rail_preserves_parent_steps_with_pure_status_subtasks():
    v2_css = _read_browser_css()

    # No tags, events, sub-agent rows or action entries can leak into the rail.

    plan_step_rule = _css_rule(v2_css, ".plan-rail-step")
    assert "cursor: default" in plan_step_rule
    # Middle-workspace retry card is roomy (full width, glass card language).
    assert "width: 100%" in _css_rule(v2_css, ".plan-retry-card")
    # Batch 1 (2026-10-14 redesign D1-D5): content cards are flat; glass is only for floating controls.
    assert "backdrop-filter" not in _css_rule(v2_css, ".plan-retry-card")

    assert ".plan-rail-phase" not in v2_css
    assert ".plan-rail-major-number" not in v2_css


def test_plan_rail_phase_rows_keep_modeling_spec_before_feature_screen():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ planRailPhaseRows }} from {json.dumps(module_url)};",
            "const rows = planRailPhaseRows({ steps: [",
            "  { id: 's6', index: 5, phase: '特征', title: '特征筛选', status: 'failed', tool_ref: { tool: 'screen_features' } },",
            "  { id: 's5', index: 4, phase: '建模', title: '选择建模规格', status: 'done', tool_ref: { tool: 'choose_modeling_spec' } },",
            "  { id: 's4', index: 3, phase: '特征', title: '切分样本', status: 'done', tool_ref: { tool: 'make_split' } },",
            "] });",
            "process.stdout.write(JSON.stringify(rows.map((row) => [row.number, row.phase, row.steps.map((step) => step.title), row.checkerStatus])));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        [1, "特征", ["切分样本", "选择建模规格", "特征筛选"], "failed"],
    ]


def test_layout_resize_controller_restores_drags_and_persists_widths():
    script = """
import assert from "node:assert/strict";
import {
  createLayoutResizeController,
  PROGRESS_WIDTH_MAX,
  SIDEBAR_WIDTH_DEFAULT,
  SIDEBAR_WIDTH_MAX,
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
// The legacy 320px default migrates to the new default width.
assert.equal(styleValues.get("--sidebar-width"), `${SIDEBAR_WIDTH_DEFAULT}px`);
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
assert.equal(styleValues.get("--sidebar-width"), `${SIDEBAR_WIDTH_MAX}px`);
listeners.pointerup();
assert.equal(bodyClasses.has("is-resizing"), false);
assert.equal(removed.length, 2);
assert.equal(JSON.parse(storageData.marvis_layout).sidebar, SIDEBAR_WIDTH_MAX);

controller.resetSidebarWidth();
assert.equal(styleValues.get("--sidebar-width"), `${SIDEBAR_WIDTH_DEFAULT}px`);
assert.equal(JSON.parse(storageData.marvis_layout).sidebar, SIDEBAR_WIDTH_DEFAULT);

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
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_plan_rail_retry_step_posts_edited_inputs():
    v2_css = _read_browser_css()


    # Roomier middle-region grid (wider min column than the old narrow rail).
    assert "grid-template-columns: repeat(auto-fit, minmax(220px, 1fr))" in _css_rule(v2_css, ".plan-retry-schema-fields")
    assert "color: var(--text-muted)" in _css_rule(v2_css, ".plan-retry-scope")


def test_plan_retry_uses_manually_edited_json_when_structured_fields_are_present():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
import {{ createPlanRailController }} from {json.dumps(module_url)};

globalThis.window = {{ setTimeout: () => {{}} }};
globalThis.fetch = async () => ({{
  ok: true,
  json: async () => ({{ plans: [{{ id: "plan-1", steps: [{{ id: "step-1", status: "failed" }}] }}] }}),
}});
const calls = [];
const controller = createPlanRailController({{
  getSelectedTaskId: () => "task-1",
  getSelectedTask: () => ({{ id: "task-1" }}),
  getAgentMessages: () => [],
  isAgentMode: () => false,
  renderWorkflowStepper: () => {{}},
  setActionStatus: () => {{}},
  refreshTasks: async () => {{}},
  loadAgentMessages: async () => {{}},
  renderAll: () => {{}},
  apiClient: async (url, options) => {{
    if (!options.method) return (await globalThis.fetch(url)).json();
    calls.push({{ url, body: JSON.parse(options.body) }});
    return {{ ok: true }};
  }},
}});
controller.maybeFetchPlan("task-1");
await new Promise((resolve) => setTimeout(resolve, 0));
await new Promise((resolve) => setTimeout(resolve, 0));

const structuredField = {{
  dataset: {{ planRetryInputKey: "foo", planRetryInputType: "string" }},
  value: "old",
}};
const jsonField = {{
  value: '{{"foo":"edited","extra":1}}',
  defaultValue: '{{"foo":"old"}}',
}};
const form = {{
  querySelectorAll: (selector) => selector === "[data-plan-retry-input-key]" ? [structuredField] : [],
  querySelector: (selector) => selector === ".plan-retry-inputs" ? jsonField : null,
}};
const button = {{
  dataset: {{ planRetryStep: "step-1" }},
  disabled: false,
  closest: () => form,
}};
controller.handleClick({{
  target: {{ closest: (selector) => selector === "[data-plan-retry-step]" ? button : null }},
  preventDefault: () => {{}},
  stopPropagation: () => {{}},
}});
await new Promise((resolve) => setTimeout(resolve, 0));

assert.equal(calls.length, 1);
assert.deepEqual(calls[0].body.inputs, {{ foo: "edited", extra: 1 }});
process.stdout.write("ok");
"""
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"


def test_plan_retry_preserves_nullable_integer_as_null():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
import {{ createPlanRailController }} from {json.dumps(module_url)};

globalThis.window = {{ setTimeout: () => {{}} }};
globalThis.fetch = async () => ({{
  ok: true,
  json: async () => ({{ plans: [{{ id: "plan-1", steps: [{{ id: "step-1", status: "failed" }}] }}] }}),
}});
const calls = [];
const controller = createPlanRailController({{
  getSelectedTaskId: () => "task-1",
  getSelectedTask: () => ({{ id: "task-1" }}),
  getAgentMessages: () => [],
  isAgentMode: () => false,
  renderWorkflowStepper: () => {{}},
  setActionStatus: () => {{}},
  refreshTasks: async () => {{}},
  loadAgentMessages: async () => {{}},
  renderAll: () => {{}},
  apiClient: async (url, options) => {{
    if (!options.method) return (await globalThis.fetch(url)).json();
    calls.push({{ url, body: JSON.parse(options.body) }});
    return {{ ok: true }};
  }},
}});
controller.maybeFetchPlan("task-1");
await new Promise((resolve) => setTimeout(resolve, 0));
await new Promise((resolve) => setTimeout(resolve, 0));

const structuredField = {{
  dataset: {{
    planRetryInputKey: "early_stopping_rounds",
    planRetryInputType: "integer",
    planRetryInputNullable: "true",
  }},
  value: "",
}};
const jsonField = {{
  value: '{{"early_stopping_rounds":null}}',
  defaultValue: '{{"early_stopping_rounds":null}}',
}};
const form = {{
  querySelectorAll: (selector) => selector === "[data-plan-retry-input-key]" ? [structuredField] : [],
  querySelector: (selector) => selector === ".plan-retry-inputs" ? jsonField : null,
}};
const button = {{
  dataset: {{ planRetryStep: "step-1" }},
  disabled: false,
  closest: () => form,
}};
controller.handleClick({{
  target: {{ closest: (selector) => selector === "[data-plan-retry-step]" ? button : null }},
  preventDefault: () => {{}},
  stopPropagation: () => {{}},
}});
await new Promise((resolve) => setTimeout(resolve, 0));

assert.equal(calls.length, 1);
assert.deepEqual(calls[0].body.inputs, {{ early_stopping_rounds: null }});
process.stdout.write("ok");
"""
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"


def test_plan_retry_refreshes_until_the_retried_plan_is_terminal():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
import {{ createPlanRailController }} from {json.dumps(module_url)};

const timers = [];
globalThis.window = {{ setTimeout: (callback) => timers.push(callback) }};
let planFetchCount = 0;
const planStates = [
  {{ id: "plan-1", status: "failed", steps: [{{ id: "step-1", status: "failed" }}] }},
  {{ id: "plan-1", status: "running", steps: [{{ id: "step-1", status: "running" }}] }},
  {{ id: "plan-1", status: "done", steps: [{{ id: "step-1", status: "done" }}] }},
];
globalThis.fetch = async () => {{
  const plan = planStates[Math.min(planFetchCount, planStates.length - 1)];
  planFetchCount += 1;
  return {{ ok: true, json: async () => ({{ plans: [plan] }}) }};
}};
let refreshCount = 0;
let messageRefreshCount = 0;
const controller = createPlanRailController({{
  getSelectedTaskId: () => "task-1",
  getSelectedTask: () => ({{ id: "task-1" }}),
  getAgentMessages: () => [],
  isAgentMode: () => false,
  renderWorkflowStepper: () => {{}},
  setActionStatus: () => {{}},
  setDriverExecutionBusy: () => {{}},
  refreshTasks: async () => {{ refreshCount += 1; }},
  loadAgentMessages: async () => {{ messageRefreshCount += 1; }},
  renderAll: () => {{}},
  apiClient: async (url, options) => options.method ? ({{ ok: true }}) : (await globalThis.fetch(url)).json(),
}});
controller.maybeFetchPlan("task-1");
await new Promise((resolve) => setTimeout(resolve, 0));
await new Promise((resolve) => setTimeout(resolve, 0));

const structuredField = {{
  dataset: {{ planRetryInputKey: "seed", planRetryInputType: "integer" }},
  value: "23",
}};
const jsonField = {{
  value: '{{"seed":23}}',
  defaultValue: '{{"seed":23}}',
}};
const form = {{
  querySelectorAll: (selector) => selector === "[data-plan-retry-input-key]" ? [structuredField] : [],
  querySelector: (selector) => selector === ".plan-retry-inputs" ? jsonField : null,
}};
const button = {{
  dataset: {{ planRetryStep: "step-1" }},
  disabled: false,
  closest: () => form,
}};
controller.handleClick({{
  target: {{ closest: (selector) => selector === "[data-plan-retry-step]" ? button : null }},
  preventDefault: () => {{}},
  stopPropagation: () => {{}},
}});
await new Promise((resolve) => setTimeout(resolve, 0));
await new Promise((resolve) => setTimeout(resolve, 0));

assert.equal(planFetchCount, 2);
assert.equal(timers.length, 1);
timers.shift()();
await new Promise((resolve) => setTimeout(resolve, 0));
await new Promise((resolve) => setTimeout(resolve, 0));

assert.equal(planFetchCount, 3);
assert.equal(refreshCount, 2);
assert.equal(messageRefreshCount, 2);
assert.equal(timers.length, 0);
process.stdout.write("ok");
"""
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"


def test_plan_retry_replace_semantics_warning_is_always_present():
    """LT-4: a smoke pass on the retry flow found /api/plans/{id}/steps/{id}/retry
    fully REPLACES the step's inputs_json (marvis/repositories/plans.py
    retry_failed_step, UPDATE ... SET inputs_json = ?) rather than merging with
    the step's existing inputs. Before any schema form existed this was invisible
    footgun risk on the JSON editor -- assert the warning copy renders next to it
    unconditionally, and that the JSON editor keeps pre-filling current values
    (the mitigation for a user who edits only part of the object)."""
    v2_css = _read_browser_css()

    # The form (with the warning + JSON editor) now renders in the middle
    # workspace via planRetryCardHtml, not the old rail planRetryControlHtml.

    assert "color: var(--warning-strong)" in _css_rule(v2_css, ".plan-retry-warning")
    assert "background: var(--warning-soft)" in _css_rule(v2_css, ".plan-retry-warning")


def test_plan_retry_schema_form_marks_required_fields_and_falls_back_to_inferred_schema():
    """LT-4: the schema form (planRetrySchemaFieldsHtml) previously only ever
    saw the failure_envelope's *inferred* editable_input_schema (value-typed,
    no `required`/`enum`/`title` -- see marvis/agent/gates/contracts.py
    _editable_input_schema). It now accepts an optional real tool input_schema
    (sourced from the already-existing GET /api/plugins/{name}/tools) and
    marks required fields with a red `*`. When no real schema resolves, the
    form must render exactly as before (defensive fallback, no behavior
    regression)."""
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ createPlanRailController }} from {json.dumps(module_url)};",
            "function fakeEl() {",
            "  return {",
            "    _html: '',",
            "    get innerHTML() { return this._html; },",
            "    set innerHTML(value) { this._html = value; },",
            "    dataset: {},",
            "    classList: { add() {}, remove() {}, toggle() {} },",
            "    setAttribute() {},",
            "    querySelector() { return null; },",
            "  };",
            "}",
            "const elements = {",
            "  progressRail: { setAttribute() {} },",
            "  workflowStepper: { innerHTML: '' },",
            "  planRetryPanel: fakeEl(),",
            "};",
            "function $(id) { return elements[id] || null; }",
            "globalThis.document = { querySelector() { return { textContent: '' }; } };",
            "const step = {",
            "  id: 'step-1',",
            "  index: 0,",
            "  title: 'Propose join',",
            "  status: 'failed',",
            "  tool_ref: { plugin: 'data_ops', tool: 'propose_join' },",
            "  depends_on: [],",
            "  inputs: { dataset_id: 'ds-1', strategy: 'left' },",
            "  failure_envelope: {",
            "    editable_input_schema: {",
            "      type: 'object',",
            "      properties: {",
            "        dataset_id: { type: 'string', default: 'ds-1' },",
            "        strategy: { type: 'string', default: 'left' },",
            "      },",
            "    },",
            "    downstream_reset_steps: [],",
            "  },",
            "};",
            "const plan = { id: 'plan-1', status: 'failed', steps: [step] };",
            "const toolsResponses = {",
            "  'data_ops': { tools: [{ name: 'propose_join', input_schema: {",
            "    type: 'object',",
            "    properties: {",
            "      dataset_id: { type: 'string', title: '数据集' },",
            "      strategy: { type: 'string', enum: ['left', 'inner'], title: '策略' },",
            "    },",
            "    required: ['dataset_id', 'strategy'],",
            "  } }] },",
            "};",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "let renders = 0;",
            "function makeController(listPluginToolsClient) {",
            "  return createPlanRailController({",
            "    $,",
            "    getSelectedTask: () => ({ task_type: 'data_join' }),",
            "    getSelectedTaskId: () => 'task-A',",
            "    getAgentMessages: () => [],",
            "    isAgentMode: () => false,",
            "    renderWorkflowStepper: () => { renders += 1; controller.render({ force: true, renderSignatures: {} }); },",
            "    setActionStatus: () => {},",
            "    listPluginToolsClient,",
            "  });",
            "}",
            "let controller = makeController((name) => Promise.resolve(toolsResponses[name] || { tools: [] }));",
            "controller.render({ force: true, renderSignatures: {} });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: {} });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "const withRealSchema = elements.planRetryPanel.innerHTML;",
            "",
            "elements.workflowStepper.innerHTML = '';",
            "elements.planRetryPanel = fakeEl();",
            "controller = makeController((name) => Promise.reject(new Error('network down')));",
            "controller.render({ force: true, renderSignatures: {} });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: {} });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "const withFailedFetch = elements.planRetryPanel.innerHTML;",
            "process.stdout.write(JSON.stringify({ withRealSchema, withFailedFetch }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    with_real_schema = payload["withRealSchema"]
    with_failed_fetch = payload["withFailedFetch"]

    # Real schema resolved: both fields are required (red `*`), strategy
    # renders as a <select> from its real enum, and titles come from the
    # real schema instead of the raw key.
    assert with_real_schema.count('<span class="plan-retry-required">*</span>') == 2
    assert '<option value="left" selected>left</option>' in with_real_schema
    assert '<option value="inner">inner</option>' in with_real_schema
    assert "数据集" in with_real_schema
    assert "策略" in with_real_schema
    assert 'plan-retry-schema-field required' in with_real_schema

    # Defensive fallback: schema fetch rejected, so the form still renders
    # from the inferred failure_envelope schema alone -- flat text inputs,
    # no required marks, no crash (behavior does not regress).
    assert "plan-retry-required" not in with_failed_fetch
    assert 'data-plan-retry-input-key="dataset_id"' in with_failed_fetch
    assert 'data-plan-retry-input-key="strategy"' in with_failed_fetch
    assert "整体替换" in with_failed_fetch


def test_completed_report_actions_render_below_step_copy_with_office_colors():
    styles_css = _read_browser_css()




    action_start = styles_css.index(".step-download-actions {")
    action_end = styles_css.index("}", action_start)
    action_rule = styles_css[action_start:action_end]
    assert "margin-left: 48px;" in action_rule
    # The three download buttons wrap instead of being clipped in the narrow rail.
    assert "flex-wrap: wrap;" in action_rule
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

    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
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
    styles_css = _read_browser_css()
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




def test_pointer_focus_ring_only_shows_when_clicking_inside_form_controls():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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
    script = session_projection_fixture(script)
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
    script = r'''
import assert from 'node:assert/strict';
const taskSession = { taskId: null, task: null };
let failure;
const statuses = [], createStatuses = [], finished = [];
const refreshTasks = async () => { throw failure; };
const setActionStatus = (...args) => statuses.push(args);
const setCreateStatus = (...args) => createStatuses.push(args);
const renderAll = () => finished.push('render');
const restoreResultScrollPositionAfterRender = async () => finished.push('restore');
const validationBatchPanelController = { focusRequestedItem: () => finished.push('focus') };
const finishAppBoot = () => finished.push('boot');
const maybeResumeAgentValidationBatch = () => { throw Error('failed initialization cannot resume'); };
''' + init_body + r'''
failure = new Error('backend unreachable');
await initializeApp();
assert.deepEqual(statuses[0], ['服务连接失败，请检查后端是否运行。', 'error', 'backend unreachable']);
assert.deepEqual(createStatuses[0], ['backend unreachable', 'error']);
failure = Object.assign(new Error('login required'), { status: 401 });
await initializeApp();
assert.deepEqual(statuses[1], ['请在设置中的机构身份页面登录，登录后读取已授权任务。', 'info', '']);
assert.deepEqual(createStatuses[1], [statuses[1][0], 'info']);
assert.deepEqual(finished, ['render', 'restore', 'focus', 'boot', 'render', 'restore', 'focus', 'boot']);
'''
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True,
    check=False)
    assert result.returncode == 0, result.stderr
    assert "runAction(async () => {\n  await refreshTasks();" not in app_js


def test_create_dialog_hides_v2_config_controls():
    index_html = _read_static("index.html")

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

    for element_id in ["runModeManual", "runModeAgent", "modelName", "validator", "sourceDir"]:
        assert f'id="{element_id}"' in index_html


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
    assert "...validationNarrativeDefaults(seed.modelName)" in defaults
    narrative = _slice_function(create_dialog_js, "function validationNarrativeDefaults(")
    payload = _run_node_capture_json(f"let modelName = '模型A';\nfunction taskTextSeed() {{ return {{ modelName, validator: 'qa', reportTitle: '测试报告' }}; }}\nfunction formatDateInput() {{ return '2026-09-17'; }}\n{narrative}\n{defaults}\nconst generic = defaultCreateReportValues();\nmodelName = '渠道甲T卡 MOB6'; const t = defaultCreateReportValues();\nmodelName = '渠道甲A卡 MOB3'; const a = defaultCreateReportValues();\nprocess.stdout.write(JSON.stringify({{ generic, t, a }}));")
    for variant in payload.values():
        for key in ("model_overview", "model_scope", "sample_audience", "bad_sample_definition", "good_sample_definition"):
            assert variant[f"TEXT:{key}"] == "", "Model names must not invent business facts"
    assert '"TEXT:data_source_summary"' not in defaults
    assert '"TEXT:dataset_split_summary"' not in defaults


def test_create_dialog_uses_visual_run_mode_cards():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()
    run_mode_start = index_html.index('class="run-mode-cards"')
    run_mode_end = index_html.index("</section>", run_mode_start)
    run_mode_markup = index_html[run_mode_start:run_mode_end]

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
    assert "预留" not in run_mode_markup
    assert "后续" not in run_mode_markup
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

    assert 'data-run-mode-description="manual"' in index_html
    assert 'data-run-mode-description="agent"' in index_html
    assert "由验证人员逐步执行材料扫描、Notebook 验证与报告生成" not in index_html
    assert "智能解析材料、规划验证步骤并辅助生成验证报告" not in index_html



def test_create_dialog_defaults_available_mode_without_preselecting_algorithms():
    index_html = _read_static("index.html")

    assert 'name="runMode" type="radio" value="manual" checked' not in index_html
    assert 'name="runMode" type="radio" value="agent" checked' not in index_html
    assert 'name="modelAlgorithm" value="lgb" checked' not in index_html
    assert 'name="modelAlgorithm" value="xgb" checked' not in index_html
    assert 'name="modelAlgorithm" value="lr" checked' not in index_html
    assert 'name="modelAlgorithm" value="scorecard" checked' not in index_html


    # Batch 5: the reset lives in prepareTaskDialog, shared by openTaskDialog
    # and the home card's prefillTaskDialog.



def test_create_dialog_reopens_with_clean_task_specific_inputs_and_defaults():
    create_dialog_js = _read_static("js/create-task-dialog.js")
    module_url = (STATIC_DIR / "js" / "create-task-dialog.js").as_uri()
    # Batch 5: the reset lives in prepareTaskDialog, shared by openTaskDialog
    # and the home card's prefillTaskDialog.
    dialog_start = create_dialog_js.index("function prepareTaskDialog")
    dialog_end = create_dialog_js.index("function openTaskDialogFromCard", dialog_start)
    dialog_body = create_dialog_js[dialog_start:dialog_end]
    assert "prepareTaskDialog(taskType);" in dialog_body

    assert "resetCreateTaskSpecificInputs" in dialog_body
    assert 'document.querySelectorAll(\'input[name="runMode"]\')' in dialog_body
    assert "input.checked = input.value === defaultMode;" in dialog_body
    assert "materialSourceController.reset();" in dialog_body
    assert dialog_body.index("resetCreateTaskSpecificInputs") < dialog_body.index(
        "prefillCreateTaskReportFields"
    )

    script = f"""
import assert from "node:assert/strict";
import {{ resetCreateTaskSpecificInputs }} from {json.dumps(module_url)};

const elements = {{
  modelName: {{ value: "MODEL-XGB" }},
  validator: {{ value: "上一负责人" }},
  sourceDir: {{ value: "/tmp/previous-materials" }},
  modelOotKsMin: {{ value: "0.42" }},
  materialUploadInput: {{ value: "previous.csv" }},
}};
const reportInputs = [
  {{ value: "上一任务模型概述" }},
  {{ value: "上一任务适用范围" }},
];
const root = {{
  querySelectorAll(selector) {{
    assert.equal(selector, "[data-create-report-key]");
    return reportInputs;
  }},
}};

resetCreateTaskSpecificInputs({{
  $: (id) => elements[id] || null,
  root,
}});

for (const id of ["modelName", "validator", "sourceDir", "modelOotKsMin", "materialUploadInput"]) {{
  assert.equal(elements[id].value, "", id);
}}
assert.deepEqual(reportInputs.map((input) => input.value), ["", ""]);
"""
    script = session_projection_fixture(script)
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_create_dialog_prioritizes_materials_before_optional_report_background():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    assert "报告初始内容" not in index_html
    task_info_start = index_html.index('id="createTaskPrimaryModelHeading"')
    report_start = index_html.index('id="createTaskReportFields"')
    material_start = index_html.index('id="createTaskMaterialSection"')
    create_button_start = index_html.index('id="createTaskButton"')
    assert task_info_start < material_start < report_start < create_button_start

    task_info_section = index_html[task_info_start:material_start]
    assert 'id="sourceDir"' not in task_info_section
    assert index_html.index('id="sourceDir"') < index_html.index('id="createGoodSampleDefinition"')

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
  materialUploadStatus: { textContent: "", title: "" },
};
renderMaterialUploadSelection({
  files: [{ name: "sample.parquet", relativePath: "oot/sample.parquet" }],
  getElementById: (id) => elements[id],
});
assert.equal(elements.materialUploadStatus.textContent, "已选择 sample.parquet，包含 1 个目录。");
assert.equal(elements.materialUploadStatus.title, "已选择 sample.parquet，包含 1 个目录。");

renderMaterialUploadSelection({
  files: [],
  getElementById: (id) => elements[id],
});
assert.equal(elements.materialUploadStatus.textContent, "请选择文件或文件夹。");
assert.equal(elements.materialUploadStatus.title, "");
process.stdout.write("ok");
"""
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_material_upload_long_filename_cannot_expand_task_dialog():
    styles_css = _read_browser_css()
    containment_rule = _css_rule(
        styles_css,
        ".material-source-section,\n.material-source-panel",
    )
    assert "min-width: 0" in containment_rule
    assert "max-width: 100%" in containment_rule

    dropzone_rule = _css_rule(styles_css, ".material-upload-dropzone")
    assert "overflow: hidden" in dropzone_rule

    status_rule = _css_rule(styles_css, "#materialUploadStatus")
    assert "min-width: 0" in status_rule
    assert "max-width: 100%" in status_rule
    assert "overflow: hidden" in status_rule
    assert "text-overflow: ellipsis" in status_rule
    assert "white-space: nowrap" in status_rule


def test_validation_material_binding_dialog_is_wired_before_scan():
    index_html = _read_static("index.html")

    assert 'id="materialBindingDialog"' in index_html
    assert 'id="materialBindingRows"' in index_html
    assert 'id="materialBindingConfirmButton"' in index_html
    # Role labels and their accessible names are exercised in the rendered
    # material dialog by test_frontend_conversation_actions_browser.py.


def test_validation_input_candidates_use_native_tables_with_legacy_row_styling():
    styles_css = _read_browser_css()

    assert '.agent-message[data-agent-stage="input_confirmation"] .agent-message-content ul' in styles_css
    assert '.agent-message[data-agent-stage="input_confirmation"] .agent-message-content li' in styles_css
    legacy_list_rule = _css_rule(
        styles_css,
        '.agent-message[data-agent-stage="input_confirmation"] .agent-message-content ul',
    )
    assert "list-style: none" in legacy_list_rule
    assert "border: 1px solid var(--border)" in legacy_list_rule


def test_validation_material_binding_dialog_keeps_content_inside_viewport():
    styles_css = _read_browser_css()

    dialog_rule = _css_rule(styles_css, ".material-binding-dialog")
    assert "width: min(760px, calc(100vw - 32px))" in dialog_rule

    panel_rule = _css_rule(styles_css, ".material-binding-dialog .task-dialog-panel")
    assert "width: 100%" in panel_rule

    body_rule = _css_rule(styles_css, ".material-binding-body")
    assert "flex: 1 1 auto" in body_rule
    assert "min-height: 0" in body_rule
    assert "overflow: hidden" in body_rule

    rows_rule = _css_rule(styles_css, ".material-binding-rows")
    assert "overflow-y: auto" in rows_rule
    assert "overscroll-behavior: contain" in rows_rule






def test_risk_analysis_can_defer_materials_and_upload_from_composer():
    index_html = _read_static("index.html")

    assert 'id="riskMaterialUploadInput"' in index_html
    assert 'id="riskMaterialUploadButton"' in index_html
    assert 'accept=".csv,.xlsx,.xlsm,.parquet"' in index_html


def test_manual_risk_analysis_intake_exposes_deterministic_composer_without_llm():
    app_js = _read_static("app.js")
    styles_css = _read_browser_css()

    helper_start = app_js.index("function latestRiskAnalysisIntakePhase")
    helper_end = app_js.index("function syncRiskMaterialUploadControl", helper_start)
    helper_body = app_js[helper_start:helper_end]
    script = f"let selectedTask = {{ id: 'risk-1', task_type: 'vintage', run_mode: 'manual' }};\nlet agentMessages = [{{ role: 'assistant', metadata: {{ risk_analysis_intake: {{ phase: 'ask_goal' }} }} }}];\n{helper_body}\nconst phases = [];\nphases.push(selectedTaskNeedsManualRiskIntake());\nagentMessages = [{{ role: 'assistant', metadata: {{ risk_analysis_intake: {{ phase: 'request_materials' }} }} }}];\nphases.push(selectedTaskNeedsManualRiskIntake());\nagentMessages = [{{ role: 'assistant', metadata: {{ risk_analysis_intake: {{ phase: 'ready' }} }} }}];\nphases.push(selectedTaskNeedsManualRiskIntake());\nselectedTask = {{ id: 'risk-2', task_type: 'vintage', run_mode: 'agent' }};\nagentMessages = [{{ role: 'assistant', metadata: {{ risk_analysis_intake: {{ phase: 'ask_goal' }} }} }}];\nphases.push(selectedTaskNeedsManualRiskIntake());\nprocess.stdout.write(JSON.stringify(phases));"
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [True, True, False, False]

    render = _slice_function(app_js, "function renderAgentConversation")
    assert "selectedTaskNeedsManualRiskIntake()" in render
    assert 'classList.toggle("manual-risk-intake"' in render
    assert "isAgent || showManualRiskIntake" in render
    assert ".agent-composer.manual-risk-intake .agent-composer-mode-controls" in styles_css
    assert ".agent-composer.manual-risk-intake .agent-composer-right-controls > .agent-composer-controls" in styles_css

    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    send_body = app_js[send_start:send_end]
    assert "const deterministicRiskIntake = selectedTaskNeedsManualRiskIntake();" in send_body
    assert "const deterministicTurn = deterministicRiskIntake || deterministicPortfolioTurn;" in send_body
    assert "if (!deterministicTurn)" in send_body
    assert "const requestBody = { content };" in send_body
    assert "requestBody.model_id = modelId || null;" in send_body


def test_manual_risk_analysis_intake_posts_content_without_model_configuration():
    app_js = _read_static("app.js")
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    send_body = app_js[send_start:send_end]
    script = f"let selectedTaskId = 'risk-1';\nlet selectedTask = {{ id: 'risk-1', task_type: 'vintage', run_mode: 'manual' }};\nlet agentMessages = [];\nconst agentRequestAbortControllers = new Map();\nconst input = {{ value: '标准 Vintage', style: {{}} }};\nconst posted = [];\nfunction $(id) {{ if (id === 'agentComposerInput') return input; if (id === 'agentModelSelect') return {{ value: '' }}; return null; }}\nfunction selectedTaskNeedsManualRiskIntake() {{ return true; }}\nfunction agentModelUnavailableMessage() {{ throw new Error('manual intake must not require an LLM'); }}\nfunction showAgentModelGuidance() {{ throw new Error('manual intake must not show LLM guidance'); }}\nfunction setAgentComposerNotice() {{}}\nfunction autoGrowComposerInput() {{}}\nfunction updateAgentSendDisabled() {{}}\nfunction appendOptimisticAgentUserMessage() {{ return {{ id: 'optimistic-user' }}; }}\nfunction appendOptimisticAgentThinkingMessage() {{ return {{ id: 'optimistic-thinking' }}; }}\nfunction removeOptimisticAgentMessage() {{}}\nfunction agentEffort() {{ throw new Error('manual intake must not send effort'); }}\nfunction agentAcceptanceModeValue() {{ throw new Error('manual intake must not send acceptance mode'); }}\nasync function api(url, options) {{ posted.push({{ url, body: JSON.parse(options.body) }}); return {{ messages: [], status: 'ok' }}; }}\nasync function pollAgentMessagesUntilSettled(_taskId, pending) {{ await pending; }}\nfunction renderAgentConversation() {{}}\nasync function waitForAgentValidation() {{}}\nasync function handleAgentMaterialSelectionRequest() {{}}\nfunction setActionStatus() {{}}\n{send_body}\nawait startAgentValidation();\nprocess.stdout.write(JSON.stringify(posted));"
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        {
            "url": "api/tasks/risk-1/agent/messages",
            "body": {"content": "标准 Vintage"},
        }
    ]


def test_manual_vintage_material_upload_control_runs_deterministic_intake():
    app_js = _read_static("app.js")
    helpers_start = app_js.index("function selectedTaskIsRiskAnalysisAgent")
    helpers_end = app_js.index("function updateWorkspaceGreeting", helpers_start)
    helpers_body = app_js[helpers_start:helpers_end]
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function uploadRiskAnalysisMaterials", send_start)
    send_body = app_js[send_start:send_end]
    upload_start = send_end
    upload_end = app_js.index("async function dispatchAgentValidation", upload_start)
    upload_body = app_js[upload_start:upload_end]
    handlers_start = app_js.index('$("riskMaterialUploadButton").onclick')
    handlers_end = app_js.index('$("sendAgentMessageButton").onclick', handlers_start)
    handlers_body = app_js[handlers_start:handlers_end]
    script = "\n".join(
        [
            'import assert from "node:assert/strict";',
            "let selectedTaskId = 'risk-manual';",
            "let selectedTask = { id: 'risk-manual', task_type: 'vintage', run_mode: 'manual' };",
            "let agentMessages = [{ role: 'assistant', metadata: { risk_analysis_intake: { phase: 'request_materials' } } }];",
            "const agentRequestAbortControllers = new Map();",
            "const uploads = [];",
            "const posts = [];",
            "let pendingAction = null;",
            "const button = { disabled: true, hidden: true, onclick: null, classList: { toggle(name, force) { if (name === 'hidden') button.hidden = force; } } };",
            "const picker = { value: 'selected', clicks: 0, changeHandler: null, click() { this.clicks += 1; }, addEventListener(name, handler) { if (name === 'change') this.changeHandler = handler; } };",
            "const composer = { value: '', style: {} };",
            "function $(id) {",
            "  if (id === 'riskMaterialUploadButton') return button;",
            "  if (id === 'riskMaterialUploadInput') return picker;",
            "  if (id === 'agentComposerInput') return composer;",
            "  if (id === 'agentModelSelect') return { value: '' };",
            "  return null;",
            "}",
            "function taskBusyAction() { return null; }",
            "function requireTaskId(value) { return value; }",
            "async function uploadDataset(taskId, file, options) { uploads.push({ taskId, name: file.name, role: options.role }); }",
            "async function reloadDataWorkspace() {}",
            "function setAgentComposerNotice() {}",
            "function autoGrowComposerInput() {}",
            "function updateAgentSendDisabled() {}",
            "function appendOptimisticAgentUserMessage() { return { id: 'optimistic-user' }; }",
            "function appendOptimisticAgentThinkingMessage() { return { id: 'optimistic-thinking' }; }",
            "function removeOptimisticAgentMessage() {}",
            "function agentModelUnavailableMessage() { throw new Error('manual intake must not require an LLM'); }",
            "function showAgentModelGuidance() { throw new Error('manual intake must not show LLM guidance'); }",
            "function agentModelConfigurationErrorMessage(error) { return error.message; }",
            "function agentEffort() { throw new Error('manual intake must not send effort'); }",
            "function agentAcceptanceModeValue() { throw new Error('manual intake must not send acceptance mode'); }",
            "async function api(url, options) { posts.push({ url, body: JSON.parse(options.body) }); return { messages: [], status: 'ok' }; }",
            "async function pollAgentMessagesUntilSettled(_taskId, pending) { await pending; }",
            "function renderAgentConversation() {}",
            "async function waitForAgentValidation() {}",
            "async function handleAgentMaterialSelectionRequest() {}",
            "function setActionStatus() {}",
            "function runAction(action) { pendingAction = action(); return pendingAction; }",
            helpers_body,
            send_body,
            upload_body,
            handlers_body,
            "syncRiskMaterialUploadControl();",
            "assert.equal(button.hidden, false);",
            "assert.equal(button.disabled, false);",
            "button.onclick();",
            "assert.equal(picker.clicks, 1);",
            "picker.changeHandler({ currentTarget: { files: [{ name: 'vintage.csv' }] } });",
            "await pendingAction;",
            "assert.deepEqual(uploads, [{ taskId: 'risk-manual', name: 'vintage.csv', role: 'sample' }]);",
            "assert.deepEqual(posts, [{ url: 'api/tasks/risk-manual/agent/messages', body: { content: '已上传材料：vintage.csv。请检查表结构和字段是否满足要求，并继续分析。' } }]);",
            "assert.equal(button.disabled, false);",
            "assert.equal(picker.value, '');",
            "selectedTask = { id: 'risk-agent', task_type: 'vintage', run_mode: 'agent' };",
            "selectedTaskId = 'risk-agent';",
            "button.hidden = true;",
            "button.disabled = true;",
            "syncRiskMaterialUploadControl();",
            "assert.equal(button.hidden, false);",
            "assert.equal(button.disabled, false);",
            'process.stdout.write("ok");',
        ]
    )
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"




def test_create_dialog_sections_are_unframed():
    styles_css = _read_browser_css()

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
    styles_css = _read_browser_css()

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
    styles_css = _read_browser_css()

    assert 'id="progressRail"' in index_html
    assert 'aria-label="验证步骤"' in index_html
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


def test_plan_rail_contains_only_steps_and_progress_not_output_preview():
    index_html = _read_static("index.html")

    assert 'id="artifactPanel"' not in index_html
    assert 'id="artifactPanelBody"' not in index_html


def test_plan_workflow_status_drives_task_header_from_real_plan_state():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    data = _run_node_capture_json(
        "\n".join([
            f"const {{ planWorkflowStatus }} = await import({json.dumps(module_url)});",
            "const failed = planWorkflowStatus({ status: 'failed', steps: [{ title: '拼接诊断', status: 'failed', error: 'boom' }] });",
            "const failedAwaitingRetry = planWorkflowStatus({ status: 'awaiting_confirm', steps: [",
            "  { title: '调参', status: 'failed', error: 'resource limit' },",
            "  { title: '训练模型', status: 'pending' },",
            "] });",
            "const waiting = planWorkflowStatus({ status: 'awaiting_confirm', steps: [{ title: '执行拼接', status: 'awaiting_confirm' }] });",
            "const done = planWorkflowStatus({ status: 'done', steps: [] });",
            "console.log(JSON.stringify({ failed, failedAwaitingRetry, waiting, done }));",
        ])
    )

    assert data["failed"]["label"] == "失败"
    assert data["failed"]["kind"] == "error"
    assert "拼接诊断" in data["failed"]["detail"]
    assert data["failedAwaitingRetry"]["label"] == "失败"
    assert data["failedAwaitingRetry"]["kind"] == "error"
    # Batch 4 (2026-10-14 redesign): snapshots use the shared tone set (danger -> fail).
    assert data["failedAwaitingRetry"]["tone"] == "fail"
    assert "调参" in data["failedAwaitingRetry"]["detail"]
    assert data["waiting"]["label"] == "待确认"
    assert "执行拼接" in data["waiting"]["detail"]
    assert data["done"]["label"] == "已完成"


def test_failed_retry_confirmation_keeps_rail_failure_and_downstream_pending():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    data = _run_node_capture_json(
        "\n".join([
            f"const {{ planRailPhaseRows, planWorkflowStatus }} = await import({json.dumps(module_url)});",
            "const plan = { status: 'awaiting_confirm', steps: [",
            "  { id: 'tune', index: 5, phase: '建模', title: '调参', status: 'failed' },",
            "  { id: 'train', index: 6, phase: '建模', title: '训练模型', status: 'pending' },",
            "] };",
            "const snapshot = planWorkflowStatus(plan);",
            "const rows = planRailPhaseRows(plan);",
            "console.log(JSON.stringify({",
            "  snapshot,",
            "  parent: rows[0].checkerStatus,",
            "  children: rows[0].steps.map((step) => step.status),",
            "}));",
        ])
    )

    # Batch 4 (2026-10-14 redesign): the snapshot is the shared presentation (key/face/dot/checker, tone fail).
    assert data["snapshot"] == {
        "key": "failed",
        "label": "失败",
        "tone": "fail",
        "face": "fail",
        "dot": "bad",
        "checker": "failed",
        "message": "计划执行失败。",
        "kind": "error",
        "detail": "失败步骤：调参。请在中间信息流中查看诊断与恢复方案。",
    }
    assert data["parent"] == "failed"
    assert data["children"] == ["failed", "pending"]


def test_waiting_confirmation_uses_matching_task_header_pill():
    display = _task_display_status_for(
        {"status": "scanned", "status_message": "等待你的确认。"},
        action_message="等待你的确认。",
        action_kind="info",
    )

    # Batch 4 (2026-10-14 redesign): the word comes from structured state, not message text; scanned is 待继续.
    assert display["heroPill"] == {"label": "待继续", "tone": "review"}




def test_sidebar_empty_state_is_compact():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    empty_start = styles_css.index(".empty-state {\n  padding")
    empty_end = styles_css.index("}", empty_start)
    empty_rule = styles_css[empty_start:empty_end]
    assert "padding: 9px 12px" in empty_rule
    assert '<div class="empty-state">暂无任务</div>' in index_html

    task_empty_start = styles_css.index(".task-list > .empty-state {")
    task_empty_end = styles_css.index("}", task_empty_start)
    task_empty_rule = styles_css[task_empty_start:task_empty_end]
    assert "border: 0" in task_empty_rule
    assert "background: transparent" in task_empty_rule
    assert "text-align: center" in task_empty_rule


def test_shell_has_collapsible_compact_sidebar():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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
    # Batch 2 (2026-10-14 redesign D7-D10): the sidebar defaults to 248px.
    assert "--sidebar-width: 248px" in styles_css
    assert "--sidebar-width: 0px" in styles_css
    assert ".app-shell.sidebar-collapsed .brand-logo h1" in styles_css
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
    styles_css = _read_browser_css()

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
    assert "--radius-control: 8px" in brand_token_rule
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
    styles_css = _read_browser_css()

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
    # The collapsed search flyout is a floating layer (D2): white surface,
    # hairline edge and the single popover elevation.
    assert "background: var(--surface)" in flyout_rule
    assert "border: 1px solid var(--sep)" in flyout_rule
    assert "box-shadow: var(--shadow-pop)" in flyout_rule
    assert "background: var(--sidebar-bg)" not in flyout_rule
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
    styles_css = _read_browser_css()

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

    # Batch 2 (2026-10-14 redesign D7-D10): drag 200-380px, double-click resets to 248px.
    assert "grid-template-columns: min(var(--sidebar-width), 314px) minmax(0, 1fr)" in styles_css


def test_sidebar_footer_and_create_action_match_brand_treatment():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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
    assert "border-radius: var(--radius-pill)" in create_rule
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

    assert "验证任务" in index_html
    assert "选择或创建验证任务" not in index_html
    assert "创建任务或从左侧选择已有任务" in index_html
    assert "先创建任务或从左侧选择已有任务" not in index_html
    assert "核心任务信息" in index_html
    assert "选择任务后显示核心任务信息" not in index_html


def test_selected_task_header_omits_local_validation_subtitle():
    styles_css = _read_browser_css()


    assert ".workspace-subtitle:empty" in styles_css
    assert "display: none" in styles_css[
        styles_css.index(".workspace-subtitle:empty") : styles_css.index("}", styles_css.index(".workspace-subtitle:empty"))
    ]




def test_refresh_restores_selected_task_before_async_detail_loads():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()


    # Placeholder identity and previous-visit invalidation have behavior coverage
    # in test_frontend_task_request_scope.py.



    assert 'class="app-booting"' in index_html
    assert "body.app-booting .validation-workspace:not(.is-empty) :is(.workspace-head, .workspace-body, .progress-rail)" in styles_css
    assert "body.app-booting :is(.workspace-head, .workspace-body, .progress-rail)" in styles_css

    app_shell_rule = _css_rule(styles_css, ".app-shell")
    assert "transition:" not in app_shell_rule





def test_workspace_cards_float_on_one_background_with_top_step_rail():
    styles_css = _read_browser_css()

    assert ".validation-workspace {" in styles_css
    assert "--workspace-main-gutter: 24px" in styles_css
    assert "--workspace-collapsed-main-gutter: 76px" in styles_css
    assert "--workspace-rail-gap: 24px" in styles_css
    assert "--workspace-collapsed-rail-gap: 24px" in styles_css
    assert "--pet-default-workspace-offset: -2px" in styles_css
    assert "--pet-min-workspace-offset: -2px" in styles_css
    assert "--pet-collapsed-workspace-offset: 72px" in styles_css
    assert "--progress-width: 278px" in styles_css
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
    assert "min-width: 0" in rail_rule
    assert "margin: 14px 14px 14px 0" in rail_rule
    assert ".progress-panel {" in styles_css
    # Batch 1 (2026-10-14 redesign D1-D5): content surfaces carry no shadow.
    assert "--shadow-floating: none" in styles_css
    assert ".result-scroll-content > .progress-panel" in styles_css
    assert "flex: 0 0 auto" in styles_css


def test_right_resize_handle_sits_on_step_rail_left_edge():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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
    styles_css = _read_browser_css()

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
    styles_css = _read_browser_css()

    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))
    # Batch 2 (2026-10-14 redesign D7-D10): the rail is a floating glass card and carries the glass elevation.
    assert root_vars["--progress-rail-shadow"] == "var(--glass-shadow)"
    assert dark_vars["--progress-rail-shadow"] == "var(--glass-shadow)"
    rail_rule = _css_rule(styles_css, ".progress-rail")
    assert "box-shadow: var(--progress-rail-shadow)" in rail_rule

    dark_rule = _css_rule(styles_css, 'body[data-theme="dark"] .progress-rail')
    assert "box-shadow:" not in dark_rule


def test_dark_theme_tokens_keep_panels_and_muted_text_readable():
    styles_css = _read_browser_css()
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
    styles_css = _read_browser_css()
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
    styles_css = _read_browser_css()
    root_vars = _css_vars(_css_rule(styles_css, ":root"))
    dark_vars = _css_vars(_css_rule(styles_css, 'body[data-theme="dark"]'))

    # Batch 1 (2026-10-14 redesign D1-D5): neutral greys tuned to the contrast floors below.
    assert root_vars["--option-hover"] == "#e0e0e5"
    assert root_vars["--option-selected"] == "#d8d8dd"
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
    # D15 (batch 6b): engines are rows of one grouped list; the group is rounded.
    assert "border-radius: 0" in model_card_rule
    assert "border-radius: var(--radius-md)" in _css_rule(styles_css, ".llm-model-profiles")
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
    styles_css = _read_browser_css()

    workspace_rule = _css_rule(styles_css, 'body[data-theme="dark"] .validation-workspace')
    assert "background: var(--bg)" in workspace_rule

    rail_rule = _css_rule(styles_css, 'body[data-theme="dark"] .progress-rail')
    # Batch 2 (2026-10-14 redesign D7-D10): the dark rail uses the same glass tokens as the light one.
    assert "background: var(--glass-bg)" in rail_rule
    assert "border-color: var(--glass-rim)" in rail_rule


def test_dark_workspace_masks_match_center_background():
    styles_css = _read_browser_css()

    base_workspace_rule = _css_rule(styles_css, ".validation-workspace")
    assert "--workspace-mask-bg: var(--surface)" in base_workspace_rule

    workspace_rule = _css_rule(styles_css, 'body[data-theme="dark"] .validation-workspace')
    assert "--workspace-mask-bg: var(--bg)" in workspace_rule

    head_rule = _css_rule(styles_css, ".workspace-head")
    assert "transparent calc(var(--radius) - 1px)" in head_rule
    assert "var(--workspace-mask-bg, var(--surface)) calc(var(--radius) - 0.5px)" in head_rule
    # Batch 1 (2026-10-14 redesign D1-D5): the status card is grey on a white canvas, so all four corners are masked.
    assert "background-position: left 0 top 0, right 0 top 0, left 0 bottom 0, right 0 bottom 0" in head_rule
    assert "calc(var(--radius) + 2px) calc(var(--radius) + 2px)" in head_rule
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
    # The composer bar is now translucent glass, so the matte only masks the two
    # bottom-corner slivers (mirroring .workspace-head's top-corner masks) instead
    # of painting an opaque block behind the whole bar.
    assert "radial-gradient" in composer_mask_rule
    assert "transparent calc(var(--radius) - 1px)" in composer_mask_rule
    assert (
        "var(--agent-composer-mask-bg, var(--workspace-mask-bg, var(--surface))) calc(var(--radius) - 0.5px)"
        in composer_mask_rule
    )
    assert "background-position: left -1px bottom -1px, right -1px bottom -1px" in composer_mask_rule
    assert "background-size: calc(var(--radius) + 2px) calc(var(--radius) + 2px)" in composer_mask_rule
    assert "var(--workspace-mask-bg, var(--surface)) calc(var(--radius) + 0.5px)" not in composer_mask_rule
    assert "transparent 24px, var(--surface) 24.5px" not in composer_mask_rule


def test_primary_step_action_hover_keeps_button_text_readable():
    styles_css = _read_browser_css()
    root_rule = _css_rule(styles_css, ":root")

    for key in ("solid", "solid-shadow-hover", "secondary", "secondary-shadow-hover"):
        name = key if "shadow" in key else f"{key}-shadow"
        # Batch 1 (2026-10-14 redesign D1-D5): buttons are flat capsules.
        assert f"--button-{name}: none" in root_rule
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

    # Secondary buttons are flat neutral-fill capsules (R3): no visible border.
    secondary_rule = _css_rule(styles_css, ".button.secondary")
    assert "border-color: transparent" in secondary_rule
    assert "background: var(--fill)" in secondary_rule
    assert "box-shadow: var(--button-secondary-shadow)" in secondary_rule

    secondary_hover_rule = _css_rule(
        styles_css, ".button.secondary:hover:not(:disabled),\n.button.secondary:focus-visible:not(:disabled)"
    )
    assert "border-color: transparent" in secondary_hover_rule
    assert "background: var(--fill-2)" in secondary_hover_rule
    assert "box-shadow: var(--button-secondary-shadow-hover)" in secondary_hover_rule


def test_theme_button_tokens_drive_create_environment_and_model_buttons():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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


def test_sidebar_task_card_is_single_line_with_hover_preview():
    styles_css = _read_browser_css()
    index_html = _read_static("index.html")


    assert 'id="taskRowPreview"' in index_html
    delete_hover_rule = _css_rule(
        styles_css, ".delete-task-button:hover,\n.delete-task-button:focus-visible"
    )
    # Batch 1 (2026-10-14 redesign D1-D5): no decorative coloured borders; the danger tint and red glyph carry the state.
    assert "border-color: transparent" in delete_hover_rule

    # Batch 4 (2026-10-14 redesign): app.js keeps no local tone map; the row resolves
    # through js/task-status.js, whose V1 table is pinned by test_frontend_status_vocabulary.
    # writing_artifacts is dual-meaning (idle vs. report-job-running): the table
    # says 待继续 and the resolver splits the report job off via active_job_kind.

    row_start = styles_css.index("\n.task-row {\n  --task-card-action-space")
    row_end = styles_css.index("}", row_start)
    row_rule = styles_css[row_start:row_end]
    assert "--task-card-action-space: 24px" in row_rule
    assert "padding: 9px 10px" in row_rule
    assert "border: 1px solid transparent" in row_rule
    assert "padding: 11px 42px" not in row_rule

    top_start = styles_css.index("\n.task-row-top {")
    top_end = styles_css.index("}", top_start)
    top_rule = styles_css[top_start:top_end]
    assert "grid-template-columns: minmax(0, 1fr) max-content" in top_rule
    assert "padding-right: var(--task-card-action-space)" in top_rule

    name_start = styles_css.index(".task-row-name {")
    name_end = styles_css.index("}", name_start)
    name_rule = styles_css[name_start:name_end]
    assert "min-width: 0" in name_rule
    assert "overflow: hidden" in name_rule
    assert "font-size: 14px" in name_rule
    assert "font-weight: 500" in name_rule
    assert "text-overflow: ellipsis" in name_rule
    assert "white-space: nowrap" in name_rule

    preview_rule = _css_rule(styles_css, ".task-row-preview")
    assert "position: fixed" in preview_rule
    assert "pointer-events: none" in preview_rule
    # Batch 1: the hover card is a floating layer (radius 12, shared popover elevation).
    assert "border-radius: var(--radius-md)" in preview_rule
    assert "box-shadow: var(--shadow-pop)" in preview_rule

    # Batch 4 (2026-10-14 redesign, D6): the row shows a status dot, not a word pill;
    # running is the blue accent, as the run pill was.
    assert ".task-row-top .pill {" not in styles_css
    assert ".pill.run {" not in styles_css
    dot_rule = _css_rule(styles_css, ".task-status-dot")
    assert "flex: none" in dot_rule
    assert "border-radius: 50%" in dot_rule
    run_dot_rule = _css_rule(styles_css, ".task-status-dot.run")
    assert "background: var(--accent)" in run_dot_rule


def test_task_list_signature_tracks_plan_derived_status():
    """A recovered driver plan must invalidate the sidebar's render guard.

    Driver task rows display the plan-derived label/tone, while the task record
    itself commonly remains ``created``.  The list signature therefore has to
    change when only the cached plan state changes (failed -> running).
    """
    app_js = _read_static("app.js")
    start = app_js.index("function taskListSignature")
    end = app_js.index("function metricPreviewSignature", start)
    signature_source = app_js[start:end]
    script = f"function signatureFromParts(parts) {{ return JSON.stringify(parts); }}\nlet taskSearchQuery = '';\nlet taskSortMode = 'created_desc';\nlet taskGroupMode = 'none';\nlet selectedTaskId = 'task-1';\nlet derivedLabel = '失败';\nlet derivedTone = 'danger';\nfunction taskStatusLabel() {{ return derivedLabel; }}\nfunction taskStatusTone() {{ return derivedTone; }}\n{signature_source}\nconst task = {{ id: 'task-1', name: '恢复任务', task_type: 'modeling', status: 'created', updated_at: '2026-07-19T00:00:00Z', active_job_kind: 'driver', validator: 'QA' }};\nconst failed = taskListSignature([task], 1);\nderivedLabel = '执行中';\nderivedTone = 'run';\nconst running = taskListSignature([task], 1);\nconsole.log(JSON.stringify({{ failed, running }}));"

    data = _run_node_capture_json(script)
    assert data["failed"] != data["running"], data


def test_task_list_selected_card_has_no_visible_border_or_outline():
    styles_css = _read_browser_css()

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
    styles_css = _read_browser_css()

    head_start = index_html.index('<header class="workspace-head"')
    head_end = index_html.index("</header>", head_start)
    head_markup = index_html[head_start:head_end]
    result_workspace_start = index_html.index('id="resultWorkspace"')
    result_workspace_end = index_html.index('id="scanSection"', result_workspace_start)
    result_workspace_markup = index_html[result_workspace_start:result_workspace_end]


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
    # validator was dropped from the task-hero snapshot per user feedback;
    # only the execution mode and source directory remain.


def test_header_task_meta_values_stay_on_one_line():
    styles_css = _read_browser_css()

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








def test_notebook_step_progress_items_are_single_line_and_compact():
    styles_css = _read_browser_css()

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




def test_running_visual_tone_uses_header_status_blue():
    styles_css = _read_browser_css()

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


def test_running_indicators_stop_animating_when_motion_is_reduced():
    execution_css = (STATIC_DIR / "css" / "execution-workspace.css").read_text(
        encoding="utf-8"
    )
    task_shell_css = (STATIC_DIR / "css" / "task-shell.css").read_text(
        encoding="utf-8"
    )

    assert re.search(
        r"@media \(prefers-reduced-motion: reduce\)\s*\{\s*"
        r"\.check-icon\.running\s*\{\s*animation: none;",
        execution_css,
    )
    assert re.search(
        r"@media \(prefers-reduced-motion: reduce\)\s*\{\s*"
        r"\.task-pill\.run::before\s*\{\s*animation: none;",
        task_shell_css,
    )




def test_global_settings_actions_do_not_mark_selected_task_busy():
    app_js = _read_static("app.js")
    set_busy_start = app_js.index("function renderBusyActivity")
    set_busy_end = app_js.index("function setAgentMemoryStatus", set_busy_start)
    run_action_start = app_js.index("async function runAction")
    run_action_end = app_js.index("function handleTaskListKeydown", run_action_start)
    script = "\n".join(
        [
            "const taskBusyActions = new Map();",
            "let globalBusyAction = null;",
            "let selectedTaskId = 'task-done';",
            "const events = [];",
            "function setActionStatus(...args) { events.push(['status', ...args]); }",
            "function renderWorkflowStepper() { events.push(['stepper']); }",
            "function renderPetState() { events.push(['pet']); }",
            "function updateAgentSendDisabled() { events.push(['send']); }",
            "function renderAll() { events.push(['render']); }",
            "function setAgentMemoryStatus(...args) { events.push(['memory', ...args]); }",
            "function setDraftToolsStatus(...args) { events.push(['draft', ...args]); }",
            "function renderActionError(...args) { events.push(['action-error', ...args]); }",
            "function setCreateStatus(...args) { events.push(['create', ...args]); }",
            "function actionCancelledStatusTitle(actionId) { return `${actionId} cancelled`; }",
            "function actionFailureStatusTitle(actionId) { return `${actionId} failed`; }",
            "async function refreshTasks() { events.push(['refresh']); }",
            app_js[set_busy_start:set_busy_end],
            app_js[run_action_start:run_action_end],
            "await runAction(async () => { events.push(['global-action', taskActivities.action('task-done')]); }, {",
            "  actionId: 'executionEnvironment',",
            "  taskScoped: false,",
            "});",
            "let taskScopedSizeDuringAction = null;",
            "await runAction(async () => {",
            "  taskScopedSizeDuringAction = taskActivities.action('task-done');",
            "}, { actionId: 'scan' });",
            "process.stdout.write(JSON.stringify({",
            "  globalActionDidNotMarkTask: events.some((event) => event[0] === 'global-action' && event[1] === null),",
            "  taskScopedSizeDuringAction,",
            "  finalTaskBusySize: taskActivities.action('task-done'),",
            "  statusEvents: events.filter((event) => event[0] === 'status').length,",
            "}));",
        ]
    )

    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "globalActionDidNotMarkTask": True,
        "taskScopedSizeDuringAction": "scan",
        "finalTaskBusySize": None,
        "statusEvents": 1,
    }




@pytest.mark.parametrize("status,expected", [
    ("executed", ["succeeded", "succeeded", "pending", "pending"]),
    ("computing_metrics", ["succeeded", "succeeded", "running", "pending"]),
    ("writing_artifacts", ["succeeded", "succeeded", "succeeded", "pending"]),
    ("review_required", ["succeeded"] * 4),
])
def test_workflow_step_status_separates_next_action_from_running_action(status, expected):
    assert _workflow_step_statuses_for({"status": status}, []) == expected


def test_modeling_create_dialog_has_algorithm_selector():
    """The create dialog exposes a manual-mode modeling algorithm multi-select
    (G2: 算法可选), gated to modeling tasks via the algorithmField flag, and
    posted as `payload.recipes` + `payload.target_type`."""
    index_html = _read_static("index.html")

    assert 'id="createTaskAlgorithmField"' in index_html
    assert 'id="modelAlgorithmChoices"' in index_html
    assert 'name="modelAlgorithm"' in index_html
    for recipe in ('value="lgb"', 'value="xgb"', 'value="catboost"', 'value="lr"', 'value="scorecard"', 'value="mlp"'):
        assert recipe in index_html
    assert 'id="modelSampleWeightCol"' in index_html
    assert 'data-recipe-family="binary"' in index_html
    # regression + multiclass target types are exposed too, so the UI can drive
    # §8.2/§8.3 tasks, not only binary
    for recipe in (
        "lgb_regressor",
        "xgb_regressor",
        "lr_regressor",
        "mlp_regressor",
        "lgb_multiclass",
        "xgb_multiclass",
        "lr_multiclass",
        "mlp_multiclass",
    ):
        assert f'value="{recipe}"' in index_html
    assert 'data-recipe-family="continuous"' in index_html
    assert 'data-recipe-family="multiclass"' in index_html
    assert 'id="modelSampleWeightPolicy"' in index_html
    assert 'value="none">不使用样本权重' in index_html
    assert 'value="explicit">指定权重列' in index_html


def test_modeling_create_dialog_maps_every_non_lgb_recipe_to_payload_target_type():
    module_url = (STATIC_DIR / "js" / "create-task-dialog.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
import {{ modelTargetTypeForRecipes }} from {json.dumps(module_url)};

const expected = {{
  xgb_regressor: "continuous",
  lr_regressor: "continuous",
  mlp_regressor: "continuous",
  xgb_multiclass: "multiclass",
  lr_multiclass: "multiclass",
  mlp_multiclass: "multiclass",
}};
for (const [recipe, targetType] of Object.entries(expected)) {{
  const payload = {{ recipes: [recipe] }};
  payload.target_type = modelTargetTypeForRecipes(payload.recipes);
  assert.equal(payload.target_type, targetType, recipe);
}}
assert.equal(modelTargetTypeForRecipes(["xgb_regressor", "lr_regressor"]), "continuous");
assert.equal(modelTargetTypeForRecipes(["xgb_regressor", "xgb_multiclass"]), null);
"""
    script = session_projection_fixture(script)
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_strategy_vintage_and_portfolio_welcome_cards_are_enabled():
    """风险分析、策略开发和组合分析都是已接通的 PlanDriver 入口。"""
    index_html = _read_static("index.html")
    for card_id in (
        "welcomeVintageAnalysisCard",
        "welcomeStrategyDevelopmentCard",
        "welcomePortfolioAnalysisCard",
    ):
        start = index_html.index(f'id="{card_id}"')
        tag_end = index_html.index(">", start)
        tag = index_html[start:tag_end]
        assert "data-coming-soon" not in tag, card_id
        assert 'class="welcome-task-card available"' in tag, card_id
        assert 'aria-describedby="welcomeComingSoonHint"' not in tag, card_id
    # The toast path remains available for future explicitly unavailable task definitions.


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
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_feature_create_dialog_has_optional_metric_selector():
    """Every feature metric is an explicit manual-mode choice (spec §2: 选了才算).

    Only IV/KS/AUC/coverage default on; PSI/VIF/lift/importance remain opt-in and
    meaning consistency clearly advertises its Agent + dictionary gate.
    """
    index_html = _read_static("index.html")

    assert 'id="createTaskMetricField"' in index_html
    assert 'id="featureMetricChoices"' in index_html
    for metric in ("iv", "ks", "auc", "coverage"):
        assert f'name="featureMetric" value="{metric}" checked' in index_html
    for metric in (
        "psi_month_first",
        "psi_month_last",
        "psi_month_previous",
        "psi_split",
        "vif",
        "head_tail_lift",
        "importance",
    ):
        assert f'name="featureMetric" value="{metric}"' in index_html
        assert f'name="featureMetric" value="{metric}" checked' not in index_html
    assert 'name="featureMetric" value="meaning_consistency" disabled' in index_html


@pytest.mark.parametrize("stage", ["scan", "notebook", "metrics", "report"])
def test_stage_failures_keep_completed_previous_steps_green(stage):
    steps = ["scan", "notebook", "metrics", "report"]
    index = steps.index(stage)
    expected = ["succeeded"] * index + ["failed"] + ["pending"] * (3 - index)
    assert _workflow_step_statuses_for({"status": "failed", "failure_stage": stage,
                                        "status_message": "不同的失败文案"}, []) == expected


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
    assert _workflow_step_statuses_for(
        {
            "status": "failed",
            "status_message": "unexpected pipeline failure",
            "active_job_kind": None,
            "failure_stage": None,
        },
        [],
    ) == ["pending", "pending", "pending", "pending"]


def test_result_workspace_preserves_scroll_position_per_task_switch():
    styles_css = _read_browser_css()



    # Late action failures/finalizers are exercised with real visit identities.

    assert ".validation-workspace.is-task-content-loading :is(.workspace-head, .result-scroll-content, .report-draft-workspace, .agent-composer, .progress-rail)" in styles_css
    assert ".validation-workspace.is-task-content-loading :is(.result-workspace, .progress-rail)" not in styles_css
    assert ".validation-workspace.is-task-content-loading:has(#validationBatchSwitcher:not([hidden])) :is(.workspace-head)" in styles_css
    assert "body.anim-ready .validation-workspace:not(.is-task-content-loading) :is(.workspace-head)" in styles_css
    assert "body.anim-ready .validation-workspace:not(.is-task-content-loading) :is(.result-scroll-content)" in styles_css
    assert "transition: opacity 150ms ease 90ms;" in styles_css
    assert ".validation-workspace.is-task-content-settling :is(.task-hero)" in styles_css
    assert ".validation-workspace.is-projected-child-loading" not in styles_css

    # Suppression while a scroll restores: test_frontend_task_panels.py (F13).






def test_create_task_submit_keeps_create_errors_in_dialog_before_task_exists():
    index_html = _read_static("index.html")

    # Batch 5: the dialog creates through createTaskFromDialog, which only adds
    # the home sentence (when opened from 高级设置) before createTaskAndScan.

    assert 'id="statusMessage" class="status" role="status" aria-live="polite"' in index_html


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
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_dialog_close_buttons_render_as_x_controls():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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
    assert "font-size: 17px" in close_button_rule

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




def test_sort_group_and_theme_live_in_sidebar_settings():
    index_html = _read_static("index.html")
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





def test_task_group_setting_supports_created_month_and_task_type():
    index_html = _read_static("index.html")
    settings_start = index_html.index('id="sidebarSettings"')
    settings_end = index_html.index("</details>", settings_start)
    settings_markup = index_html[settings_start:settings_end]

    assert '<option value="task_type">按任务类型</option>' in settings_markup
    assert '<option value="created_month">按创建月份</option>' in settings_markup


def test_sidebar_settings_uses_dropdowns_and_stays_inside_sidebar():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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
    # Batch 1 (2026-10-14 redesign D1-D5): floating layers share one elevation.
    assert root_vars["--settings-menu-shadow"] == "var(--shadow-pop)"
    assert dark_vars["--settings-menu-shadow"] == "var(--shadow-pop)"
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



def test_appearance_setting_supports_light_dark_and_system_modes():
    index_html = _read_static("index.html")

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
    assert index_html.index('<body class="app-booting" data-theme="light"') < index_html.index('localStorage.getItem("marvis_theme")')
    assert index_html.index('localStorage.getItem("marvis_theme")') < index_html.index('id="taskDialog"')


def test_sidebar_settings_closes_on_outside_click_only():
    styles_css = _read_browser_css()



    assert ".sidebar-settings.modal-suppressed" not in styles_css
    closed_menu_rule = _css_rule(styles_css, ".sidebar-settings:not([open]) .settings-menu")
    assert "display: none" in closed_menu_rule


def test_system_settings_exposes_execution_environment_panel():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()
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
    assert 'class="settings-row settings-row-block exec-env-picker-row"' in index_html
    assert 'class="settings-row exec-env-memory-row"' in index_html
    assert '<p class="exec-env-section-label">Python 环境</p>' not in index_html
    assert "选择后用于 Notebook、验证脚本和工具运行" in index_html
    assert "留空或填 0 表示不限制" in index_html

    # The settings group is now a unified macOS-style card (the per-panel
    # card-stripping was removed); rows sit inside it, split by hairlines, and
    # each carries a leading monochrome icon.
    settings_group_rule = _css_rule(styles_css, ".governance-panel .settings-group")
    assert "border: 1px solid var(--border)" in settings_group_rule
    assert "border-radius: var(--radius-lg)" in settings_group_rule
    assert "background: var(--card)" in settings_group_rule
    assert "overflow: hidden" in settings_group_rule
    settings_row_rule = _css_rule(styles_css, ".governance-panel .settings-row")
    assert "padding: 13px 16px" in settings_row_rule
    assert 'class="settings-row-ico"' in index_html[
        index_html.index('data-governance-panel-content="execution-environment"'):
        index_html.index('data-governance-panel-content="llm"')
    ]

    # D15 (batch 6b): 执行环境 and 模型引擎 are grouped lists (one rounded
    # white group, hairline rows, no per-row box); the checkmark marks the
    # current environment.
    env_list_rule = _css_rule(styles_css, ".exec-env-list")
    llm_list_rule = _css_rule(styles_css, ".llm-model-profiles")
    for rule in (env_list_rule, llm_list_rule):
        assert "display: flex" in rule
        assert "gap: 0" in rule
        assert "border-radius: var(--radius-md)" in rule
        assert "background: var(--surface)" in rule
        assert "overflow: hidden" in rule
    env_row_rule = _css_rule(styles_css, ".exec-env-row")
    llm_card_rule = _css_rule(styles_css, ".llm-engine-item")
    for rule in (env_row_rule, llm_card_rule):
        assert "border: 0" in rule
        assert "border-radius: 0" in rule
        assert "background: transparent" in rule
    assert "box-shadow: inset 0 1px 0 var(--sep)" in _css_rule(styles_css, ".exec-env-row + .exec-env-row")
    assert "box-shadow: inset 0 1px 0 var(--sep)" in _css_rule(styles_css, ".llm-engine-item + .llm-engine-item")
    env_row_hover_rule = _css_rule(styles_css, ".exec-env-row:hover")
    assert "background: var(--option-hover)" in env_row_hover_rule
    env_row_selected_rule = _css_rule(styles_css, ".exec-env-row.selected")
    assert "background: transparent" in env_row_selected_rule
    assert "opacity: 1" in _css_rule(styles_css, ".exec-env-row.selected .exec-env-check")


def test_capability_tier_rows_match_execution_environment_density():
    styles_css = _read_browser_css()
    v2_css = _read_browser_css()

    env_list_rule = _css_rule(styles_css, ".exec-env-list")
    tier_settings_rule = _css_rule(v2_css, ".tier-settings")
    # D15 (batch 6b): both single-select lists are grouped lists with no gap.
    assert "gap: 0" in env_list_rule
    assert "gap: 8px" in tier_settings_rule
    tier_settings_head_rule = _css_rule(v2_css, ".tier-settings-head")
    assert "padding: 0 2px 2px" in tier_settings_head_rule
    tier_settings_list_rule = _css_rule(v2_css, ".tier-settings-list")
    assert "display: grid" in tier_settings_list_rule
    assert "gap: 0" in tier_settings_list_rule
    assert "background: var(--surface)" in tier_settings_list_rule

    capability_tier_start = v2_css.index("/* Capability tiers:")
    tier_row_start = v2_css.index(".tier-row {", capability_tier_start)
    tier_row_rule = v2_css[tier_row_start : v2_css.index("}", tier_row_start)]
    assert "grid-template-columns: 22px minmax(0, 1fr)" in tier_row_rule
    assert "padding: 9px 12px" in tier_row_rule
    assert "align-items: center" in tier_row_rule
    assert "margin: 0" in tier_row_rule

    governance_tier_row_rule = _css_rule(v2_css, ".governance-settings-dialog .tier-row")
    assert "padding: 9px 12px" in governance_tier_row_rule
    assert "padding: 11px 13px" not in governance_tier_row_rule

    tier_body_rule = _css_rule(v2_css, ".tier-row-body")
    assert "gap: 3px" in tier_body_rule

    tier_summary_rule = _css_rule(v2_css, ".tier-row-body p")
    assert "line-height: 1.35" in tier_summary_rule

    tier_limits_rule = _css_rule(v2_css, ".tier-limits")
    assert "gap: 4px 10px" in tier_limits_rule


def test_governance_settings_text_buttons_match_scan_environment_size():
    styles_css = _read_browser_css()
    v2_css = _read_browser_css()

    scan_rule = _css_rule(
        styles_css,
        "#refreshExecutionEnvironmentOptionsButton.button.primary,\n#addLLMModelButton.button.primary",
    )
    for expected in [
        "width: 84px",
        "min-height: 34px",
        "padding: 6px 12px",
        "font-size: 13px",
        "font-weight: 600",
    ]:
        assert expected in scan_rule

    settings_button_rule = _css_rule(
        styles_css,
        ".governance-settings-dialog :is(\n"
        "  .settings-row-control > .button,\n"
        "  .agent-memory-toolbar > .button,\n"
        "  .agent-memory-load-more .button,\n"
        "  .agent-memory-actions .button,\n"
        "  .agent-memory-view-tab,\n"
        "  .draft-tool-run-section > .button,\n"
        "  .draft-governance-actions .button\n"
        ")",
    )
    for expected in [
        "display: inline-flex",
        "align-items: center",
        "justify-content: center",
        "min-width: 84px",
        "min-height: 34px",
        "padding: 6px 12px",
        "font-size: 13px",
        "font-weight: 600",
        "line-height: 1.2",
    ]:
        assert expected in settings_button_rule

    extension_button_rule = _css_rule(
        v2_css,
        ".governance-settings-dialog :is(\n"
        "  .plugin-row-actions .button,\n"
        "  .plugin-upload-button,\n"
        "  .skill-toolbar .button,\n"
        "  .memory-manager-toolbar button,\n"
        "  .memory-distillation-row button[data-rollback-memory-distillation]\n"
        ")",
    )
    for expected in [
        "display: inline-flex",
        "align-items: center",
        "justify-content: center",
        "min-width: 84px",
        "min-height: 34px",
        "padding: 6px 12px",
        "font-size: 13px",
        "font-weight: 600",
        "line-height: 1.2",
    ]:
        assert expected in extension_button_rule


def test_governance_settings_uses_shared_typography_scale():
    v2_css = _read_browser_css()

    token_vars = _css_vars(_css_rule(v2_css, ".governance-settings-dialog"))
    assert token_vars["--settings-row-title-size"] == "14px"
    # Batch 1 (2026-10-14 redesign D1-D5): weights are 400 / 500 / 600 / 700 only.
    assert token_vars["--settings-row-title-weight"] == "600"
    assert token_vars["--settings-body-size"] == "12.5px"
    assert token_vars["--settings-body-line-height"] == "1.45"
    assert token_vars["--settings-label-size"] == "12px"
    assert token_vars["--settings-control-size"] == "13px"
    assert token_vars["--settings-control-weight"] == "600"

    title_rule = _css_rule(
        v2_css,
        ".governance-settings-dialog :is(\n"
        "  .governance-panel .settings-row-text strong,\n"
        "  .governance-setting-row strong,\n"
        "  .exec-env-row-title,\n"
        "  .llm-engine-item-name,\n"
        "  .memory-manage-head strong,\n"
        "  .draft-manage-summary-text strong,\n"
        "  .agent-memory-pane-head strong,\n"
        "  .agent-memory-item-main strong,\n"
        "  .plugin-upload-text strong,\n"
        "  .plugin-row-id strong,\n"
        "  .plugin-tool-head strong,\n"
        "  .tier-row-body h4,\n"
        "  .skill-section-label,\n"
        "  .skill-row-head strong,\n"
        "  .skill-validator-summary-text strong,\n"
        "  .memory-distillation-row button:first-child strong,\n"
        "  .extension-format-guide > summary strong,\n"
        "  .skill-builtin-workflow-title strong,\n"
        "  .draft-tool-head h3,\n"
        "  .draft-tool-section h4\n"
        ")",
    )
    assert "font-size: var(--settings-row-title-size)" in title_rule
    assert "font-weight: var(--settings-row-title-weight)" in title_rule
    assert "line-height: var(--settings-row-line-height)" in title_rule

    body_rule = _css_rule(
        v2_css,
        ".governance-settings-dialog :is(\n"
        "  .governance-panel .settings-row-text span,\n"
        "  .governance-setting-row span,\n"
        "  .exec-env-row-sub,\n"
        "  .exec-env-hint,\n"
        "  .exec-env-empty,\n"
        "  .llm-engine-empty,\n"
        "  .llm-engine-item-url,\n"
        "  .memory-manage-head span,\n"
        "  .draft-manage-summary-text span,\n"
        "  .agent-memory-view-hint,\n"
        "  .agent-memory-pane-head span,\n"
        "  .agent-memory-item-main span,\n"
        "  .agent-memory-item-main p,\n"
        "  .agent-memory-detail-meta,\n"
        "  .agent-memory-empty,\n"
        "  .plugin-upload-text > span,\n"
        "  .plugin-row-desc,\n"
        "  .plugin-tool-head span,\n"
        "  .plugin-tool-impl dd,\n"
        "  .plugin-schema-empty,\n"
        "  .tier-row-body p,\n"
        "  .tier-guardrail-note,\n"
        "  .skill-section-hint,\n"
        "  .skill-builtin-workflow-title span,\n"
        "  .skill-workflow-goals,\n"
        "  .skill-validator-summary-text span,\n"
        "  .skill-problems,\n"
        "  .skill-validation,\n"
        "  .draft-tools-hint,\n"
        "  .draft-tool-meta,\n"
        "  .draft-learning-note,\n"
        "  .draft-run-history,\n"
        "  .extension-format-guide > summary span span\n"
        ")",
    )
    assert "font-size: var(--settings-body-size)" in body_rule
    assert "font-weight: var(--settings-body-weight)" in body_rule
    assert "line-height: var(--settings-body-line-height)" in body_rule

    label_rule = _css_rule(
        v2_css,
        ".governance-settings-dialog :is(\n"
        "  .exec-env-section-label,\n"
        "  .agent-memory-filter-grid label > span,\n"
        "  .memory-manager-toolbar label,\n"
        "  .plugin-tool-impl dt,\n"
        "  .plugin-schema-table caption,\n"
        "  .plugin-schema-table th,\n"
        "  .extension-format-guide-body strong,\n"
        "  .skill-builtin-workflow-body h4,\n"
        "  .skill-workflow-slots dt,\n"
        "  .skill-workflow-step-inputs > strong,\n"
        "  .skill-workflow-checks > strong,\n"
        "  .agent-memory-detail-eyebrow,\n"
        "  .agent-memory-summary-card strong,\n"
        "  .agent-memory-evidence-card h4,\n"
        "  .agent-memory-audit-panel h4,\n"
        "  .agent-memory-source-list strong,\n"
        "  .draft-status-filter span,\n"
        "  .draft-schema-grid strong\n"
        ")",
    )
    assert "font-size: var(--settings-label-size)" in label_rule
    assert "font-weight: var(--settings-label-weight)" in label_rule

    control_rule = _css_rule(
        v2_css,
        ".governance-settings-dialog :is(\n"
        "  .settings-row-control > .button,\n"
        "  .agent-memory-toolbar > .button,\n"
        "  .agent-memory-load-more .button,\n"
        "  .agent-memory-actions .button,\n"
        "  .agent-memory-view-tab,\n"
        "  .draft-tool-run-section > .button,\n"
        "  .draft-governance-actions .button,\n"
        "  .plugin-row-actions .button,\n"
        "  .plugin-upload-button,\n"
        "  .skill-toolbar .button,\n"
        "  .memory-manager-toolbar button,\n"
        "  .memory-distillation-row button[data-rollback-memory-distillation]\n"
        ")",
    )
    assert "font-size: var(--settings-control-size)" in control_rule
    assert "font-weight: var(--settings-control-weight)" in control_rule


def test_pet_catalog_runtime_contract_drives_aliases_defaults_explicit_none_and_options():
    catalog_path = STATIC_DIR / "js" / "pet_catalog.js"
    script = f"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const context = {{}};
context.globalThis = context;
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync({json.dumps(str(catalog_path))}, "utf8"), context);

const catalog = context.MarvisPetCatalog;
assert.equal(catalog.schemaVersion, 1);
assert.equal(catalog.defaultId, "auditbot");
assert.equal(catalog.noneId, "none");
assert.deepEqual(
  JSON.parse(JSON.stringify(catalog.storageKeys)),
  {{ preference: "marvis_pet", explicitNone: "marvis_pet_none_explicit", position: "marvis_pet_position" }},
);
assert.deepEqual(
  ["danhuang", "buou", "ragdoll-cat", "unknown", "none"].map(catalog.normalizePreference),
  ["naitang", "xiaojiu", "xiaojiu", "auditbot", "none"],
);
// Batch 4 (2026-10-14 redesign): the pet is opt-in, so nothing stored (or a stored none) shows no pet.
assert.deepEqual(
  [
    catalog.resolveStoredPreference(null, false),
    catalog.resolveStoredPreference("none", false),
    catalog.resolveStoredPreference("none", true),
    catalog.resolveStoredPreference("danhuang", false),
  ],
  ["none", "none", "none", "naitang"],
);

const options = [];
const select = {{ replaceChildren: (...children) => options.push(...children) }};
const documentRef = {{ createElement: () => ({{ value: "", textContent: "" }}) }};
catalog.populateSelect(select, documentRef);
assert.deepEqual(
  options.map((option) => [option.value, option.textContent]),
  JSON.parse(JSON.stringify(catalog.order.map((petId) => [petId, catalog.definitions[petId].name]))),
);
assert.deepEqual(
  options.map((option) => option.value),
  ["none", "naitang", "xiaojiu", "auditbot", "auditbot-pro", "auditbot-poly", "auditbot-ink", "auditbot-clay", "auditbot-comic", "auditbot-pixel"],
);
"""
    script = session_projection_fixture(script)
    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)


def test_pet_catalog_assets_bootstrap_and_package_data_stay_in_sync():
    catalog_path = STATIC_DIR / "js" / "pet_catalog.js"
    script = f"""
const fs = require("node:fs");
const vm = require("node:vm");
const context = {{}};
context.globalThis = context;
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync({json.dumps(str(catalog_path))}, "utf8"), context);
process.stdout.write(JSON.stringify(context.MarvisPetCatalog));
"""
    script = session_projection_fixture(script)
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    catalog = json.loads(result.stdout)

    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    state_js = _read_static("js/state.js")
    pyproject = (STATIC_DIR.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    pets_dir = STATIC_DIR / "pets"

    catalog_script = '<script src="static/js/pet_catalog.js?v=__MARVIS_STATIC_VERSION__"></script>'
    assert index_html.count(catalog_script) == 1
    assert index_html.index(catalog_script) < index_html.index('id="petCompanion"')
    assert index_html.index(catalog_script) < index_html.index('type="module" src="static/app.js')
    settings_start = index_html.index('id="settingsPetSelect"')
    settings_end = index_html.index("</select>", settings_start)
    assert "<option" not in index_html[settings_start:settings_end]
    pet_start = index_html.index('id="petCompanion"')
    pet_end = index_html.index("</div>", index_html.index('id="petSticker"', pet_start)) + len("</div>")
    initial_pet_markup = index_html[pet_start:pet_end]
    assert 'data-pet-id=""' in initial_pet_markup
    assert "static/pets/" not in initial_pet_markup
    assert catalog["defaultId"] not in initial_pet_markup
    assert "window.MarvisPetCatalog" in index_html
    assert "globalThis.MarvisPetCatalog" in app_js
    assert "globalThis.MarvisPetCatalog" in state_js
    assert "catalog.resolveStoredPreference(stored, explicitNone)" in index_html
    assert "petCatalog.resolveStoredPreference(stored, explicitNone)" in app_js
    assert 'petCatalog.populateSelect($("settingsPetSelect"), document)' in app_js
    assert "petCatalog.storageKeys.explicitNone" in state_js
    assert 'import "./pet_catalog.js"' not in state_js
    assert "static/pets/" not in app_js
    assert '"auditbot"' not in state_js
    assert '"marvis_pet_none_explicit"' not in state_js

    asset_definitions = {
        pet_id: definition
        for pet_id, definition in catalog["definitions"].items()
        if definition["asset"] is not None
    }
    assert list(asset_definitions) == catalog["order"][1:]
    for pet_id, definition in asset_definitions.items():
        asset_path = definition["asset"].split("?", 1)[0]
        assert asset_path == f"static/pets/{pet_id}/spritesheet.webp"
        assert (STATIC_DIR / asset_path.removeprefix("static/")).is_file()
        assert not (pets_dir / pet_id / "pet.json").exists()
    # Each pet ships its spritesheet plus one avatar head per Agent face
    # (css/agent-face.css), cropped from that spritesheet.
    agent_faces = ("idle", "busy", "wait", "done", "fail")
    assert sorted(path.relative_to(pets_dir).as_posix() for path in pets_dir.rglob("*") if path.is_file()) == sorted(
        [f"{pet_id}/spritesheet.webp" for pet_id in asset_definitions]
        + [f"{pet_id}/faces/{face}.webp" for pet_id in asset_definitions for face in agent_faces]
    )

    assert '"static/pets/*/*"' in pyproject
    assert '"static/pets/*/faces/*"' in pyproject
    for pet_id in asset_definitions:
        assert f'"static/pets/{pet_id}/*"' not in pyproject


def test_pet_companion_visual_contract_stays_intact():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()
    settings_start = index_html.index('id="sidebarSettings"')
    settings_end = index_html.index("</details>", settings_start)
    settings_markup = index_html[settings_start:settings_end]

    assert 'data-settings-row="pet"' in settings_markup
    assert 'id="settingsPetSelect"' in settings_markup
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
    assert 'data-pet-id=""' in index_html
    assert 'id="petSticker"' in index_html
    assert 'pet.classList.add("hidden");' in index_html
    assert 'Number.isFinite(storedPosition.workspaceOffsetLeft)' in index_html
    assert 'const minWorkspaceOffset = petCssPx("--pet-min-workspace-offset", padding);' in index_html
    assert 'pet.style.setProperty("--pet-offset-left", `${Math.round(offsetLeft)}px`);' in index_html
    assert 'pet.style.left = "";' in index_html
    assert 'id="petCompanionLabel"' not in index_html
    assert 'class="pet-companion-label"' not in index_html
    assert 'aria-live="polite"' in index_html

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
    script = session_projection_fixture(script)
    subprocess.run(["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True)


def test_naitang_uses_pet_atlas_rows_and_drag_directions():
    styles_css = _read_browser_css()


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
    styles_css = _read_browser_css()

    def css_rule(selector: str) -> str:
        start = styles_css.index(selector)
        end = styles_css.index("}", start)
        return styles_css[start:end]

    sprite_rule = css_rule("\n.pet-sprite {")
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
    styles_css = _read_browser_css()

    # Match the base selector, not a task-workspace size override.
    sticker_start = re.search(r"(?m)^\.pet-sticker \{", styles_css).start()
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
    # 2026-10-19 user feedback: moods change the face only, never the position.
    assert "pet-playful-bounce" not in running_rule
    assert "animation: none" in running_rule


def test_pet_companion_is_draggable_and_reacts_to_task_status():
    index_html = _read_static("index.html")


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
    styles_css = _read_browser_css()
    index_html = _read_static("index.html")

    assert 'data-governance-nav="execution-environment"' in index_html
    assert 'data-governance-panel-content="execution-environment"' in index_html

    section_start = styles_css.index(".environment-dialog .execution-environment-section {")
    section_end = styles_css.index("}", section_start)
    section_rule = styles_css[section_start:section_end]
    assert "grid-template-columns: 1fr" in section_rule
    assert ".settings-panel-form" in styles_css
    assert ".governance-panel-actions" in styles_css




def test_realtime_panel_keeps_only_reproducibility_evidence_in_center():
    index_html = _read_static("index.html")

    assert 'id="reproducibilitySummary"' in index_html
    assert 'id="notebookSummary"' not in index_html
    assert 'id="notebookStepsSummary"' not in index_html
    assert 'id="contractSummary"' not in index_html

    assert "暂无 PMML 打分证据，完成 PMML 全量评分后展示结果" in index_html
    assert "暂无 Notebook 契约证据" not in index_html
    assert "还没运行验证。扫描材料后运行当前任务验证。" not in index_html


def test_reproducibility_panel_renders_score_rows_and_diff_visuals():
    styles_css = _read_browser_css()


    assert ".score-compare-list" in styles_css
    assert ".score-diff-bar" in styles_css
    assert "border-color: var(--danger-border)" in _css_rule(
        styles_css, ".result-summary.error"
    )
    assert "border-color: var(--danger-border)" in _css_rule(
        styles_css, ".score-compare-row.mismatched"
    )


def test_reproducibility_summary_omits_six_decimal_match_count_and_keeps_status_tone():
    styles_css = _read_browser_css()


    summary_grid_rule = _css_rule(styles_css, ".summary-grid")
    summary_item_rule = _css_rule(styles_css, ".summary-item")
    assert "grid-template-columns: repeat(4, minmax(176px, 1fr))" in summary_grid_rule
    assert "repeat(auto-fit" not in summary_grid_rule
    assert "overflow-x: auto" in summary_grid_rule
    assert "overflow-y: hidden" in summary_grid_rule
    assert "overscroll-behavior-x: contain" in summary_grid_rule
    assert "min-width: 176px" in summary_item_rule
    assert ".summary-item.repro-status-pass" in styles_css
    assert ".summary-item.repro-status-fail" in styles_css


def test_reproducibility_panel_renders_precision_consistency_chart():
    styles_css = _read_browser_css()



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


def test_reproducibility_card_is_hidden_until_notebook_success():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    assert 'id="notebookSection" class="progress-panel hidden"' in index_html
    assert ".progress-panel.hidden" in styles_css


def test_metric_card_is_hidden_until_metric_validation_success():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    assert 'id="metricSection" class="progress-panel hidden"' in index_html
    assert ".progress-panel.hidden" in styles_css






def test_metric_kpi_footer_values_are_centered_in_each_column():
    styles_css = _read_browser_css()

    cell_start = styles_css.index(".kpi-card-footer-cell {")
    cell_end = styles_css.index("}", cell_start)
    cell_rule = styles_css[cell_start:cell_end]

    assert "align-items: center" in cell_rule
    assert "text-align: center" in cell_rule


def test_metric_tables_use_tabular_right_aligned_numeric_cells():
    styles_css = _read_browser_css()

    table_start = styles_css.index(".metric-table-section .metric-table {")
    table_end = styles_css.index("}", table_start)
    table_rule = styles_css[table_start:table_end]
    number_start = styles_css.index(".metric-table-section .metric-table td.cell-number {")
    number_end = styles_css.index("}", number_start)
    number_rule = styles_css[number_start:number_end]

    assert "font-variant-numeric: tabular-nums" in table_rule
    assert "text-align: right" in number_rule
    assert "font-variant-numeric: tabular-nums" in number_rule


def test_metric_overview_uses_semantic_visual_tokens():
    styles_css = _read_browser_css()
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


def test_metric_overview_dark_theme_keeps_hover_and_chart_text_readable():
    styles_css = _read_browser_css()
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




def test_scan_result_renders_structured_preflight_checks():
    styles_css = _read_browser_css()

    # Errors keep a semantic red edge; warnings keep a tinted fill; passed checks are
    # neutral rows whose green label carries the state (no decorative coloured borders).
    assert "border-color: var(--danger-border)" in _css_rule(styles_css, ".preflight-check-item.danger")
    warning_rule = _css_rule(styles_css, ".preflight-check-item.warning")
    assert "border-color: transparent" in warning_rule
    assert "background: var(--warning-soft)" in warning_rule
    success_rule = _css_rule(styles_css, ".preflight-check-item.success")
    assert "border-color: transparent" in success_rule
    assert "background: var(--card)" in success_rule


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




def test_review_required_status_bar_uses_completed_green_copy_not_failure_detail():
    result = _task_action_status_for(
        {
            "status": "review_required",
            "status_message": "reproducibility failed; review required",
            "active_job_kind": None,
        }
    )

    # Batch 4 (2026-10-14 redesign): review_required is sent with its own kind, not success.
    assert result == {
        "title": "验证已完成，需复核报告。",
        "kind": "review",
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
    # Batch 4 (2026-10-14 redesign): sidebar and title share one word and tone (待复核/green + 已完成 -> 需复核/review).
    assert display == {
        "rowLabel": "需复核",
        "rowTone": "review",
        "heroPill": {"label": "需复核", "tone": "review"},
    }


def test_intermediate_stage_success_never_claims_the_whole_task_is_complete():
    scanned = _task_display_status_for(
        {
            "id": "validation-scanned",
            "task_type": "validation",
            "status": "scanned",
            "active_job_kind": None,
        },
        action_message="材料扫描完成。",
        action_kind="success",
    )
    portfolio_setup = _task_display_status_for(
        {
            "id": "portfolio-setup",
            "task_type": "portfolio",
            # A shared task record may be terminal-looking while the governed
            # portfolio workflow is still waiting for its first typed gate.
            "status": "succeeded",
            "active_job_kind": None,
        },
        action_message="组合口径已校验，请确认逾期桶由好到坏的顺序。",
        action_kind="success",
    )

    assert scanned["heroPill"] == {"label": "待继续", "tone": "review"}
    assert portfolio_setup["heroPill"] == {"label": "待确认", "tone": "review"}
    assert portfolio_setup["rowLabel"] == "待确认"
    # Batch 4 (2026-10-14 redesign): the row tone joins the shared set ('' -> review, matching the title).
    assert portfolio_setup["rowTone"] == "review"


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

    # Batch 4 (2026-10-14 redesign): 停止 -> the canonical 已停止, neutral on both surfaces.
    assert display == {
        "rowLabel": "已停止",
        "rowTone": "neutral",
        "heroPill": {"label": "已停止", "tone": "neutral"},
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

    # Batch 4 (2026-10-14 redesign): the step the task stopped on reads stopped, like the title.
    assert _workflow_step_statuses_for(task, notebook_steps) == [
        "succeeded",
        "stopped",
        "pending",
        "pending",
    ]
    assert _notebook_step_tones_for(task, notebook_steps) == ["succeeded", "stopped"]


def test_completed_parent_stage_does_not_spin_stale_running_substeps():
    app_js = _read_static("app.js")
    notebook_start = app_js.index("function notebookStepTone")
    notebook_end = app_js.index("function stepWorkflowStage", notebook_start)
    script = "\n".join(
        [
            "let selectedTask = { status: 'succeeded' };",
            "function taskStopped() { return false; }",
            app_js[notebook_start:notebook_end],
            "const tones = [",
            "  notebookStepToneForRail({ status: 'running' }, 'succeeded'),",
            "  notebookStepToneForRail({ status: 'running' }, 'running'),",
            "  notebookStepToneForRail({ status: 'failed' }, 'succeeded'),",
            "];",
            "process.stdout.write(JSON.stringify(tones));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["succeeded", "running", "failed"]


def test_later_started_substep_clears_stale_running_rail_tone():
    app_js = _read_static("app.js")
    notebook_start = app_js.index("function notebookStepTone")
    notebook_end = app_js.index("function stepWorkflowStage", notebook_start)
    script = "\n".join(
        [
            "let selectedTask = { status: 'running' };",
            "function taskStopped() { return false; }",
            app_js[notebook_start:notebook_end],
            "const steps = [",
            "  { status: 'running', started_at: '2026-07-06T10:00:00+00:00' },",
            "  { status: 'succeeded', started_at: '2026-07-06T10:00:05+00:00' },",
            "  { status: 'pending' },",
            "];",
            "const tones = steps.map((step, index) => notebookStepToneForRail(step, 'running', steps[index + 1]));",
            "process.stdout.write(JSON.stringify(tones));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["succeeded", "succeeded", "pending"]


def test_skipped_step_uses_dash_and_only_skipped_titles_are_struck_out():
    styles_css = _read_browser_css()

    assert "text-decoration: line-through" in _css_rule(styles_css, ".step.skipped .step-title")
    assert "text-decoration" not in _css_rule(styles_css, ".step.succeeded .step-title")


def test_current_status_error_detail_is_compact_accessible_and_collapsible():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    styles_css = _read_browser_css()

    assert 'id="actionErrorDetail"' in index_html
    assert 'class="action-error-detail"' in index_html
    assert "暂无报错" in index_html
    hero_start = index_html.index('id="taskHero"')
    details_start = index_html.index('id="taskHeroDetails"', hero_start)
    details_end = index_html.index("</header>", details_start)
    assert 'id="actionErrorDetail"' in index_html[details_start:details_end]
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
    assert "setActionErrorDetail(describeActionStatus(message, kind, detail), kind, detailParts)" in status_renderer
    assert "actionErrorDetail" in status_renderer
    describe_body = _slice_function(app_js, "function describeActionStatus")
    compact_body = _slice_function(app_js, "function compactActionStatusDetail")
    assert "compactActionStatusDetail(detail)" in describe_body
    assert 'kind === "error"' not in describe_body

    # The task hero owns only a concise first-line summary. The raw trace stays
    # in the middle timeline/error card and cannot expand this sticky header.
    long_detail = "No match for C0: " + "schema-field " * 80 + "\nsecond line"
    script = "\n".join(
        [
            compact_body,
            describe_body,
            f"const raw = {json.dumps(long_detail)};",
            "process.stdout.write(JSON.stringify({",
            "  compact: compactActionStatusDetail(raw),",
            "  described: describeActionStatus('任务执行失败。', 'error', raw),",
            "}));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    status_text = json.loads(result.stdout)
    assert "second line" not in status_text["compact"]
    assert len(status_text["compact"]) == 180
    assert status_text["compact"].endswith("…")
    assert status_text["described"].startswith("任务执行失败。 · No match for C0:")

    # failures are signalled by a red status pill, not a red box
    assert "function actionStatusPill" in app_js
    # Batch 4 (2026-10-14 redesign): one failure word (验证失败 -> 失败); stage copy stays in the detail line.
    assert _task_display_status_for(
        {"task_type": "validation", "status": "failed"},
        action_message="复核前检查失败。", action_kind="error",
    )["heroPill"] == {"label": "失败", "tone": "fail"}
    # Batch 4 (2026-10-14 redesign): 需复核 waits on the user, so it takes the review tone, not ok.
    assert _task_display_status_for(
        {"task_type": "validation", "status": "review_required"},
        action_message="报告需要复核。", action_kind="error",
    )["heroPill"] == {"label": "需复核", "tone": "review"}
    assert '.task-pill.fail' in styles_css

    detail_start = styles_css.index(".action-error-detail {")
    detail_end = styles_css.index("}", detail_start)
    detail_rule = styles_css[detail_start:detail_end]
    assert "color: var(--text-secondary)" in detail_rule
    assert "-webkit-line-clamp: 2" in detail_rule
    assert ".action-error-detail.error" in styles_css


def test_center_workspace_scroll_locks_status_card_and_lateral_overscroll():
    styles_css = _read_browser_css()
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
    # Batch 1 (2026-10-14 redesign D1-D5): the status card is a flat grey card.
    assert "border: 1px solid transparent;" in hero_rule
    assert "background: var(--card);" in hero_rule
    assert "transform: translateZ(0);" in hero_rule
    assert "contain: paint;" not in hero_rule
    assert "will-change: transform;" in hero_rule
    # Batch 1 (2026-10-14 redesign D1-D5): no blur on the status card; glass toolbar capsules arrive in batch 2.
    assert "backdrop-filter" not in hero_rule
    assert "background: linear-gradient" not in hero_rule
    assert "transition: border-color 140ms ease;" in hero_rule
    assert "background 180ms" not in hero_rule
    assert "box-shadow 180ms" not in hero_rule
    assert "0 14px" not in hero_rule
    assert ".task-hero.is-glass-active" in styles_css
    # Batch 2 (2026-10-14 redesign D7-D10): the glass rim replaces the dark-theme highlight line.
    assert "body[data-theme=\"dark\"] .task-hero.is-glass-active::after {" not in styles_css

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
    # Batch 2 (2026-10-14 redesign D7-D10): once content scrolls beneath it, the header becomes toolbar glass.
    # Batch 6a removed the validation tabs, so every task header floats as glass.
    assert "#resultWorkspace .task-hero.is-glass-active {" in styles_css
    assert "data-validation-view" not in styles_css
    assert "border-color: var(--glass-rim);" in active_rule
    assert "background: var(--glass-bg);" in active_rule
    assert "backdrop-filter: blur(var(--glass-blur)) saturate(180%);" in active_rule
    assert "var(--accent)" not in active_rule
    assert "rgba(0, 113, 227" not in active_rule
    assert "0 18px" not in active_rule
    assert "0 10px" not in active_rule

    dark_hero_start = styles_css.index('body[data-theme="dark"] .task-hero {')
    dark_hero_end = styles_css.index("}", dark_hero_start)
    dark_hero_rule = styles_css[dark_hero_start:dark_hero_end]
    # Batch 1 (2026-10-14 redesign): light and dark share one style, only colours differ,
    # so the dark status card is the same flat card as the light one.
    assert "border-color: transparent;" in dark_hero_rule
    assert "background: var(--card);" in dark_hero_rule
    assert "backdrop-filter: none;" in dark_hero_rule
    assert "0 16px 42px" not in dark_hero_rule

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




def test_task_hero_keeps_unclipped_top_glass_edge():
    """Status card must keep a visible top border; paint containment and a
    flush overflow parent clip the 1px edge so glassmorphism disappears."""
    styles_css = _read_browser_css()
    hero_rule = _css_rule(styles_css, ".task-hero")
    assert "border: 1px solid" in hero_rule
    assert "contain: paint;" not in hero_rule
    # Batch 1 (2026-10-14 redesign D1-D5): no blur on the status card; glass toolbar capsules arrive in batch 2.
    assert "backdrop-filter" not in hero_rule
    workspace_rule = _css_rule(styles_css, ".result-workspace")
    assert "overflow: hidden;" in workspace_rule
    assert "padding: 1px 0 0;" in workspace_rule
    assert "padding: 0;" not in workspace_rule


def test_workbench_has_no_full_width_top_hairline():
    """Every module shares the workbench chrome; do not paint a 1px strip
    above the hero via negative background offset or a workspace top border."""
    styles_css = _read_browser_css()
    for selector in (".app-shell", ".validation-workspace", ".region", ".result-workspace"):
        rule = _css_rule(styles_css, selector)
        assert "border-top:" not in rule
        assert "border: 1px solid" not in rule
    head_rule = _css_rule(styles_css, ".workspace-head")
    assert "top -1px" not in head_rule
    assert "background-position: left 0 top 0, right 0 top 0" in head_rule








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






def test_agent_driver_timeline_hides_recovered_failures_and_tracks_plan_status():
    module_url = (STATIC_DIR / "js" / "v2" / "driver_manual_analysis.js").as_uri()
    script = "\n".join(
        [
            f"import {{ filterRecoveredWorkflowFailures }} from {json.dumps(module_url)};",
            "const messages = [",
            "  { id: 'u1', role: 'user', content: '开始' },",
            "  { id: 'failure', role: 'assistant', metadata: { error: true, step_id: 'step-1' } },",
            "  { id: 'done', role: 'assistant', content: '完成' },",
            "];",
            "const ids = (status) => filterRecoveredWorkflowFailures(messages, () => status).map((message) => message.id);",
            "process.stdout.write(JSON.stringify({ done: ids('done'), failed: ids('failed'), historical: ids('historical'), unknown: ids('') }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["done"] == ["u1", "done"]
    assert payload["failed"] == ["u1", "failure", "done"]
    assert payload["historical"] == ["u1", "failure", "done"]
    assert payload["unknown"] == ["u1", "failure", "done"]

    app_js = _read_static("app.js")
    render = _slice_function(app_js, "function renderAgentConversation")
    assert "filterRecoveredWorkflowFailures(reportMessages, driverManualMessageStepStatus)" in render
    assert "driverPlanSignature" in render
    assert "driverManualMessageStepStatus(message)" in render


def test_modeling_gate_only_mounts_screen_picker_for_feature_selection():
    manual_url = (STATIC_DIR / "js" / "v2" / "driver_manual_analysis.js").as_uri()
    confirm_url = (STATIC_DIR / "js" / "v2" / "driver_gate_confirm.js").as_uri()
    script = "\n".join(
        [
            f"import {{ driverManualAnalysisHtml }} from {json.dumps(manual_url)};",
            f"import {{ renderDriverGateButton }} from {json.dumps(confirm_url)};",
            "const base = { id: 'gate', role: 'assistant', content: '结果已生成', metadata: { kind: 'gate', step_id: 'step-7', screen: { selected: ['sig1'] } } };",
            "const renderers = {",
            "  renderAgentMarkdown: (value) => value,",
            "  renderScreenTable: () => '<div data-screen-picker=\"1\"></div>',",
            "  renderGateConfirm: (message) => renderDriverGateButton(message, { gateStepTool: message.metadata.gate_source_tool }),",
            "};",
            "const tuning = driverManualAnalysisHtml([{ ...base, metadata: { ...base.metadata, gate_source_tool: 'tune_hyperparameters' } }], renderers);",
            "const selection = driverManualAnalysisHtml([{ ...base, metadata: { ...base.metadata, gate_source_tool: 'select_features' } }], renderers);",
            "process.stdout.write(JSON.stringify({ tuning, selection }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert 'data-screen-picker="1"' not in payload["tuning"]
    assert "确认并开始调参" in payload["tuning"]
    assert 'data-screen-picker="1"' in payload["selection"]
    assert "确认并开始调参" not in payload["selection"]

    app_js = _read_static("app.js")
    assert "message = gateMessageForCurrentTool(message);" in app_js


def test_select_experiment_gate_does_not_resurrect_modeling_setup_controls():
    manual_url = (STATIC_DIR / "js" / "v2" / "driver_manual_analysis.js").as_uri()
    confirm_url = (STATIC_DIR / "js" / "v2" / "driver_gate_confirm.js").as_uri()
    script = "\n".join(
        [
            f"import {{ driverManualAnalysisHtml }} from {json.dumps(manual_url)};",
            f"import {{ renderDriverGateButton }} from {json.dumps(confirm_url)};",
            "const message = {",
            "  id: 'select-gate', role: 'assistant', content: '实验对比完成',",
            "  metadata: {",
            "    kind: 'gate', step_id: 'step-10', gate_source_tool: 'select_experiment',",
            "    modeling_setup: { recipe: ['lgb'] },",
            "    model_delivery: { selected_experiment_id: 'exp-lgb' },",
            "  },",
            "};",
            "const html = driverManualAnalysisHtml([message], {",
            "  renderAgentMarkdown: (value) => value,",
            "  renderModelingSetup: () => '<div data-modeling-setup=\"1\"></div>',",
            "  renderModelDelivery: () => '<div data-model-delivery=\"1\"></div>',",
            "  renderGateConfirm: (current) => renderDriverGateButton(current, { gateStepTool: current.metadata.gate_source_tool }),",
            "});",
            "process.stdout.write(html);",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'data-modeling-setup="1"' not in result.stdout
    assert 'data-model-delivery="1"' in result.stdout
    assert "确认所选实验" in result.stdout


def test_task_creation_clicks_are_serialized_while_create_request_is_pending():
    app_js = _read_static("app.js")

    create_start = app_js.index("async function createTask")
    create_end = app_js.index("async function refreshTasks", create_start)
    create_submit_start = app_js.index("function setCreateTaskSubmitting")
    create_submit_end = app_js.index("async function createTask", create_submit_start)
    create_scan_start = app_js.index("async function createTaskAndScan")
    create_scan_end = app_js.index("async function pollValidationProgress", create_scan_start)
    script = "\n".join(
        [
            "let selectedTaskId = null;",
            "let selectedTask = null;",
            "let createTaskInFlight = false;",
            "let createApiCalls = 0;",
            "let renderAllCalls = 0;",
            "const statuses = [];",
            "const elements = {",
            "  modelName: { value: '模型A' },",
            "  validator: { value: '验证员' },",
            "  sourceDir: { value: '/tmp/materials' },",
            "  createTaskButton: { disabled: false, dataset: {} },",
            "};",
            "const checkedRunMode = { value: 'agent' };",
            "function $(id) { return elements[id] || null; }",
            "const document = { querySelector(selector) { return selector === 'input[name=\"runMode\"]:checked' ? checkedRunMode : null; } };",
            "const materialSourceController = { mode() { return 'path'; }, selectedFiles() { return []; } };",
            "const createTaskDialog = {",
            "  activeTaskType() { return 'modeling'; },",
            "  async createTask() {",
            "    createApiCalls += 1;",
            "    await new Promise((resolve) => setTimeout(resolve, 20));",
            "    return { id: `task-${createApiCalls}`, run_mode: 'agent' };",
            "  },",
            "};",
            "const strategyCandidateLabController = { selectTask() { return Promise.resolve(); } };",
            "function taskTypeDefinition() { return { initialGoal: '', label: '建模' }; }",
            "const defaultTaskType = 'modeling';",
            "function prefillAgentTaskInstruction() {}",
            # Batch 5: createTask resets the home card and remembers the owner.
            "const homeComposer = { reset() {} };",
            "const LAST_VALIDATOR_KEY = 'marvis_last_validator';",
            "const localStorage = { setItem() {} };",
            "function setCreateStatus(message, kind = '') { statuses.push({ message, kind }); }",
            "function collectCreateTaskReportValues() { return {}; }",
            "async function uploadMaterialFiles() { throw new Error('upload should not run'); }",
            "function rememberSelectedTaskId() {}",
            "function renderStoredStateSummaries() {}",
            "async function refreshTasks() {}",
            "async function loadReportFields() {}",
            "async function reloadDataWorkspace() {}",
            "function closeTaskDialog() {}",
            "function setActionStatus() {}",
            "function setBusy() {}",
            "async function loadAgentMessages() {}",
            "function renderAll() { renderAllCalls += 1; }",
            "async function scanCurrentTask() { throw new Error('agent mode should not scan'); }",
            "async function loadTaskEvidence() {}",
            "async function ensureValidationMaterialSelection(task) { return task; }",
            "async function api(endpoint) {",
            "  throw new Error(`unexpected endpoint ${endpoint}`);",
            "}",
            app_js[create_submit_start:create_submit_end],
            app_js[create_start:create_end],
            app_js[create_scan_start:create_scan_end],
            "await Promise.all([createTaskAndScan(), createTaskAndScan()]);",
            "process.stdout.write(JSON.stringify({ createApiCalls, disabled: elements.createTaskButton.disabled, busy: elements.createTaskButton.dataset.createBusy || '', statuses, renderAllCalls }));",
        ]
    )
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["createApiCalls"] == 1
    assert payload["disabled"] is False
    assert payload["busy"] == "false"
    assert any(status["message"] == "任务正在创建，请稍候。" for status in payload["statuses"])
    assert payload["renderAllCalls"] == 1




def test_delete_task_uses_platform_confirm_dialog_instead_of_browser_confirm():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    assert 'id="platformConfirmDialog"' in index_html
    assert 'id="platformConfirmTitle"' in index_html
    assert 'id="platformConfirmMessage"' in index_html
    assert 'id="platformConfirmCancelButton"' in index_html
    assert 'id="platformConfirmConfirmButton"' in index_html
    assert 'id="platformConfirmPurge"' in index_html
    # Glyph follows the welcome task-card icon language: solid silhouette, no
    # tile, layered depth shades. Both tone glyphs live in the DOM; CSS toggles.
    assert 'class="platform-confirm-glyph"' in index_html
    assert 'class="platform-confirm-glyph-trash"' in index_html
    assert 'class="platform-confirm-glyph-warn"' in index_html
    assert 'class="platform-confirm-trash-lid"' in index_html
    assert 'class="platform-confirm-sheen"' in index_html
    # The old flat plate/triangle warning icon must be gone.
    assert 'class="platform-confirm-icon-plate"' not in index_html
    assert 'class="platform-confirm-icon-triangle"' not in index_html
    assert '<svg viewBox="0 0 32 32" focusable="false">' not in index_html
    assert ".platform-confirm-dialog" in styles_css
    assert ".platform-confirm-panel" in styles_css
    # Glyph tile is bare (no background/tile), tone-colored, welcome-icon sized.
    confirm_glyph_rule = _css_rule(styles_css, ".platform-confirm-glyph")
    assert "color: var(--tone-confirm-warn)" in confirm_glyph_rule
    assert "background:" not in confirm_glyph_rule
    confirm_glyph_svg_rule = _css_rule(styles_css, ".platform-confirm-glyph svg")
    assert "fill: currentColor" in confirm_glyph_svg_rule
    assert "stroke: none" in confirm_glyph_svg_rule
    # Depth layers reuse the welcome back/mid opaque-shade recipe.
    confirm_glyph_back_rule = _css_rule(styles_css, ".platform-confirm-glyph svg .back")
    assert "fill: color-mix(in srgb, currentColor 30%, var(--surface))" in confirm_glyph_back_rule
    confirm_glyph_mid_rule = _css_rule(styles_css, ".platform-confirm-glyph svg .mid")
    assert "fill: color-mix(in srgb, currentColor 62%, var(--surface))" in confirm_glyph_mid_rule
    confirm_glyph_cs_rule = _css_rule(styles_css, ".platform-confirm-glyph svg .cs")
    assert "stroke: var(--surface)" in confirm_glyph_cs_rule
    # Danger tone recolors to --danger and swaps warn glyph for the trash glyph.
    confirm_glyph_danger_rule = _css_rule(
        styles_css, '.platform-confirm-dialog[data-tone="danger"] .platform-confirm-glyph'
    )
    assert "color: var(--danger)" in confirm_glyph_danger_rule
    assert (
        '.platform-confirm-dialog[data-tone="danger"] .platform-confirm-glyph-trash {'
        in styles_css
    )
    assert (
        '.platform-confirm-dialog[data-tone="danger"] .platform-confirm-glyph-warn {'
        in styles_css
    )
    # New tone token must exist in both light (default :root) and dark themes.
    assert styles_css.count("--tone-confirm-warn:") >= 2
    # Entrance animation + reduced-motion escape hatch.
    assert "@keyframes platform-confirm-glyph-pop" in styles_css
    assert "@keyframes platform-confirm-sheen-sweep" in styles_css
    assert "@keyframes platform-confirm-lid-peek" in styles_css
    assert "prefers-reduced-motion: reduce" in styles_css

    # Controller renders the danger-list as structured chips and supports an
    # emphasized (strong) task name, without ever using innerHTML for user text.


    # Purge counts now flow through as structured chip items, not appended text.

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
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "ok"


def test_delete_task_cannot_clear_a_pending_local_operation_from_server_idle():
    app_js = _read_static("app.js")

    busy_start = app_js.index("function taskBusyAction")
    busy_end = app_js.index("function selectedTaskIsBusy", busy_start)
    set_busy_start = app_js.index("function renderBusyActivity")
    set_busy_end = app_js.index("function setAgentMemoryStatus", set_busy_start)
    refresh_start = app_js.index("async function refreshTasks")
    refresh_end = app_js.index("async function scanCurrentTask", refresh_start)
    reconcile_start = app_js.index("async function reconcileTaskBeforeDelete")
    reconcile_end = app_js.index("async function deleteTask", reconcile_start)
    delete_start = app_js.index("async function deleteTask")
    delete_end = app_js.index("async function runAction", delete_start)
    script = "\n".join(
        [
            "let selectedTaskId = 'task-1';",
            "let selectedTask = { id: 'task-1', model_name: '模型A', status: 'created', active_job_kind: null };",
            "let globalBusyAction = null;",
            "let taskCache = [selectedTask];",
            "const taskBusyActions = new Map([['task-1', 'agent']]);",
            "const statuses = [];",
            "const apiCalls = [];",
            "let refreshed = false;",
            "function taskStopped() { return false; }",
            "function syncSelectedTaskFromCache() { selectedTask = taskCache.find((task) => task.id === selectedTaskId) || null; }",
            "function findTaskInCache(taskId) { return taskCache.find((task) => task.id === taskId) || null; }",
            "function ensureActiveTaskProgressPolling() {}",
            "function setActionStatus(message, kind = '', detail = '') { statuses.push({ message, kind, detail }); }",
            "function renderWorkflowStepper() {}",
            "function renderPetState() {}",
            "function updateAgentSendDisabled() {}",
                "function taskDisplayName(task) { return task.model_name; }",
                "function isValidationBatchTask() { return false; }",
                "const window = { confirm() { return true; } };",
            "function rememberSelectedTaskId() {}",
            "const resultScrollPositionsByTask = new Map();",
            "function renderStoredStateSummaries() {}",
            "async function loadReportFields() {}",
            "function renderAll() {}",
            "async function loadTaskPurgeSummary() { return []; }",
            "function taskPurgeApiBase(task) { return `api/tasks/${task.id}`; }",
            "function showPlatformConfirm() { return true; }",
            "function persistResultScrollPositions() {}",
            "async function api(endpoint, options = {}) {",
            "  apiCalls.push({ endpoint, method: options.method || 'GET' });",
            "  if (endpoint === 'api/tasks') {",
            "    refreshed = true;",
            "    return [{ id: 'task-1', model_name: '模型A', status: 'created', active_job_kind: null }];",
            "  }",
            "  if (endpoint === 'api/tasks/task-1' && options.method === 'DELETE') return null;",
            "  throw new Error(`unexpected ${endpoint}`);",
            "}",
            app_js[busy_start:busy_end],
            app_js[set_busy_start:set_busy_end],
            app_js[refresh_start:refresh_end],
            app_js[reconcile_start:reconcile_end],
            app_js[delete_start:delete_end],
            "const lease = taskActivities.claim('task-1', 'agent');",
            "await deleteTask(selectedTask);",
            "process.stdout.write(JSON.stringify({ refreshed, apiCalls, hasBusy: Boolean(taskActivities.action('task-1')), statuses }));",
        ]
    )
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["refreshed"] is False
    assert payload["hasBusy"] is True
    assert payload["apiCalls"] == []
    assert "正在处理" in payload["statuses"][-1]["message"]




def test_agent_wait_settles_when_failed_job_leaves_task_in_created_state():
    app_js = _read_static("app.js")
    wait_body = _slice_function(app_js, "async function waitForAgentValidation")
    poll_body = _slice_function(app_js, "async function pollValidationProgress")

    assert "settleWhenServerIdle: true" in wait_body

    script = f"const progressPolls = new Map();\nlet selectedTaskId = 'task-1';\nlet selectedTask = {{ id: 'task-1', status: 'created', active_job_kind: null }};\nlet projectedValidationChildTaskId = '';\nlet projectedValidationChildTask = null;\nfunction isWorkbenchTaskId() {{ return true; }}\nfunction usesAgentValidationWorkbench() {{ return false; }}\nconst taskCache = [selectedTask];\nconst statuses = [];\nlet refreshCount = 0;\nlet nowCall = 0;\nDate.now = () => nowCall++ === 0 ? 0 : 3600001;\nfunction claimProgressPoll(registry, taskId) {{\n  const pollState = {{ cancelled: false, promise: null }};\n  registry.set(taskId, pollState);\n  return {{ claimed: true, pollState }};\n}}\nfunction releaseProgressPoll(registry, taskId, pollState) {{\n  if (registry.get(taskId) === pollState) registry.delete(taskId);\n}}\nasync function sleep() {{}}\nasync function refreshTasks() {{ refreshCount += 1; }}\nfunction findTaskInCache(taskId) {{ return taskCache.find((task) => task.id === taskId) || null; }}\nasync function loadTaskEvidence() {{}}\nfunction metricOverviewComplete() {{ return false; }}\nfunction currentMetricPreviewHasValues() {{ return false; }}\nasync function loadReportFields() {{}}\nfunction selectedTaskIsAgentMode() {{ return true; }}\nasync function loadAgentMessages() {{}}\nfunction renderChangedValidationViews() {{}}\nfunction renderTaskList() {{}}\nfunction taskServerBusyAction(task) {{ return task.active_job_kind || null; }}\nfunction setActionStatus(message, kind) {{ statuses.push({{ message, kind }}); }}\nfunction setTaskFailureActionStatus() {{}}\n{poll_body}\nconst result = await pollValidationProgress(new Set(['scanned', 'failed']), 'task-1', {{ settleWhenServerIdle: true }});\nprocess.stdout.write(JSON.stringify({{ result, refreshCount, statuses }}));"
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["result"]["status"] == "created"
    assert payload["refreshCount"] == 1
    assert payload["statuses"] == []




def test_llm_settings_panel_and_agent_model_selector_exist():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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

    assert ".llm-engine-item" in styles_css
    assert ".checkbox-field" in styles_css

    # The LLM group shares the unified macOS-style card (no per-panel
    # card-stripping), and its heading carries a leading monochrome icon.
    settings_group_rule = _css_rule(styles_css, ".governance-panel .settings-group")
    assert "border: 1px solid var(--border)" in settings_group_rule
    assert "background: var(--card)" in settings_group_rule
    assert "overflow: hidden" in settings_group_rule
    llm_markup = index_html[
        index_html.index('data-governance-panel-content="llm"'):
        index_html.index('data-governance-panel-content="memory-policy"')
    ]
    assert 'class="settings-row-ico"' in llm_markup
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
    assert "border-color: transparent" in llm_cancel_rule
    assert "background: var(--fill)" in llm_cancel_rule
    assert "box-shadow: var(--button-secondary-shadow)" in llm_cancel_rule

    llm_cancel_hover_rule = _css_rule(
        styles_css, ".button.secondary:hover:not(:disabled),\n.button.secondary:focus-visible:not(:disabled)"
    )
    assert "#9fbfe4" not in llm_cancel_hover_rule
    assert "#f8fbff" not in llm_cancel_hover_rule
    assert "background: var(--fill-2)" in llm_cancel_hover_rule



# --- GAP-8: LLM connection preflight (test-connection button) ---------------


def test_llm_engine_edit_dialog_has_test_connection_button():
    index_html = _read_static("index.html")

    assert 'id="testLLMEngineConnectionButton"' in index_html
    assert 'id="llmEngineTestResult"' in index_html



def test_system_settings_center_keeps_extensions_without_runtime_workbench():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()
    v2_css = _read_browser_css()

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
    # Memory policies are checkbox switches now; only the platform-forced
    # (sensitive-material) row still renders a status badge.
    assert 'class="memory-policy-switch" data-memory-policy="reference_cross_task"' in index_html
    assert 'class="memory-policy-switch" data-memory-policy="auto_distill"' in index_html
    assert 'class="settings-row memory-policy-protection-row"' in index_html
    assert 'class="governance-status-badge required">平台强制</span>' in index_html
    assert 'class="governance-toggle"' not in index_html
    assert "复核 Agent 产出的工具草稿" in index_html
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
    assert 'class="draft-manage-summary"' in index_html
    assert 'class="draft-manage-summary-text"' in index_html
    assert 'class="draft-manage-chevron"' in index_html
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

    # Drafts open via the plugins extension <details> toggle (lazy load) and the
    # status filter, not a dedicated dialog or nav key.
    # The hardcoded magic-header *value* is gone (a mention in an explanatory
    # comment is fine); the header is now sourced from the injected token.

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
    draft_summary_rule = _css_rule(styles_css, ".draft-manage-summary")
    assert "display: flex" in draft_summary_rule
    assert "list-style: none" in draft_summary_rule
    assert "cursor: pointer" in draft_summary_rule
    draft_summary_marker_rule = _css_rule(styles_css, ".draft-manage-summary::marker")
    assert 'content: ""' in draft_summary_marker_rule
    assert "font-size: 0" in draft_summary_marker_rule
    draft_summary_webkit_marker_rule = _css_rule(
        styles_css, ".draft-manage-summary::-webkit-details-marker"
    )
    assert "display: none" in draft_summary_webkit_marker_rule
    draft_chevron_rule = _css_rule(styles_css, ".draft-manage-chevron")
    assert "width: 20px" in draft_chevron_rule
    assert "height: 20px" in draft_chevron_rule
    draft_open_rule = _css_rule(styles_css, ".draft-manage[open] .draft-manage-chevron")
    assert "transform: rotate(90deg)" in draft_open_rule

    workspace_summary_rule = _css_rule(v2_css, ".v2-workspace-summary")
    assert "grid-template-areas:" not in workspace_summary_rule
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
    # Upload row is a flat prompt + a real shared-.button file picker (was a
    # bespoke dashed ::after pseudo-button). The picker's native input is hidden.
    plugin_upload_rule = _css_rule(v2_css, ".governance-settings-dialog .plugin-upload")
    assert "display: flex" in plugin_upload_rule
    assert "justify-content: space-between" in plugin_upload_rule
    assert ".governance-settings-dialog .plugin-upload::after" not in v2_css
    assert 'content: "选择文件"' not in v2_css
    upload_input_rule = _css_rule(v2_css, ".plugin-upload-button input[type=\"file\"]")
    assert "opacity: 0" in upload_input_rule
    for selector in [
        ".plugin-tool-impl",
        ".plugin-schema-table-wrap",
        ".plugin-schema-table",
        ".plugin-required",
        ".plugin-schema-empty",
    ]:
        assert selector in v2_css
    assert ".plugin-tool-schemas" not in v2_css
    assert ".plugin-tools pre" not in v2_css
    # In-dialog buttons all use the shared .button classes now — no bespoke
    # .plugin-action / reload-skills button rules left to drift out of sync.
    assert ".plugin-action {" not in v2_css
    assert ".governance-settings-dialog .skill-manager button[data-reload-skills]" not in v2_css
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
    # Batch 1 (2026-10-14 redesign D1-D5): settings rows are flat grey tiles.
    assert "border: none" in memory_row_rule
    assert "background: var(--card)" in memory_row_rule
    assert "border-radius: var(--radius-control)" in memory_row_rule
    memory_rollback_rule = _css_rule(
        v2_css, ".governance-settings-dialog .memory-distillation-row button[data-rollback-memory-distillation]"
    )
    assert "color: var(--danger)" in memory_rollback_rule

    # Memory policy now uses the unified grouped card (macOS System-Settings
    # pattern): the rows sit inside ONE card split by hairlines, each with a
    # leading monochrome icon and a switch/badge pinned right.
    memory_group_rule = _css_rule(styles_css, ".governance-panel .settings-group")
    assert "border: 1px solid var(--border)" in memory_group_rule
    assert "background: var(--card)" in memory_group_rule
    assert "overflow: hidden" in memory_group_rule
    divider_rule = _css_rule(
        styles_css, ".governance-panel .settings-group > .settings-row + .settings-row"
    )
    assert "border-top" in divider_rule
    ico_rule = _css_rule(styles_css, ".settings-row-ico")
    assert "border-radius" in ico_rule
    memory_markup = index_html[
        index_html.index('data-governance-panel-content="memory-policy"'):
        index_html.index('id="memoryPolicyStatus"')
    ]
    assert memory_markup.count('class="settings-row-ico"') == 3
    memory_policy_selected_rule = _css_rule(
        styles_css, '.governance-panel[data-governance-panel-content="memory-policy"].selected'
    )
    assert "gap: 8px" in memory_policy_selected_rule
    assert (
        '.governance-panel[data-governance-panel-content="memory-policy"] .settings-row + .settings-row'
        not in styles_css
    )
    # Empty status lines collapse so panels don't open with a blank band on top.
    status_empty_rule = _css_rule(styles_css, ".governance-settings-dialog .status:empty")
    assert "display: none" in status_empty_rule
    # 查看与管理记忆 is an always-visible section now, not a collapsible fold:
    # the <details>/<summary>/chevron are gone, replaced by a plain header with a
    # top divider, and the raw-vs-distillation explainer is rendered inline.
    memory_manage_rule = _css_rule(styles_css, ".memory-manage")
    assert "background: transparent" in memory_manage_rule
    assert "border-top: 1px solid var(--border)" in memory_manage_rule
    assert "padding-top: 12px" in memory_manage_rule
    assert 'id="memoryManageSection"' in index_html
    assert 'id="memoryManageDetails"' not in index_html
    assert 'class="memory-manage-summary"' not in index_html
    assert 'class="memory-manage-chevron"' not in index_html
    assert ".memory-manage-summary" not in styles_css
    assert ".memory-manage-chevron" not in styles_css
    memory_manage_head_rule = _css_rule(styles_css, ".memory-manage-head strong")
    assert "font-weight: 600" in memory_manage_head_rule
    assert 'class="agent-memory-view-hint"' in index_html
    assert "进化沉淀</strong>" in index_html
    memory_hint_rule = _css_rule(styles_css, ".agent-memory-view-hint")
    assert "background: var(--surface-soft)" in memory_hint_rule


def test_execution_environment_exposes_notebook_memory_limit_field():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    assert 'id="notebookMemoryLimitInput"' in index_html
    assert 'placeholder="不限制"' in index_html
    assert "Notebook 内存上限" in index_html
    # Default settings shape carries the field so PUT payloads round-trip it.
    # Save-on-change wiring exists and is bound.
    # Selecting a kernel must preserve the configured limit, not reset it.
    assert ".exec-env-memory-field" in styles_css


def test_agent_conversation_panel_layout_and_message_shapes():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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
    assert ".agent-composer-chip" in styles_css
    assert ".result-workspace.agent-composer-active .result-scroll-content" in styles_css
    assert "--agent-composer-gap: 28px;" in styles_css
    assert "--agent-composer-clearance: 118px;" in styles_css
    assert "--agent-composer-clearance: 154px;" not in styles_css
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
    # Batch 1 (2026-10-14 redesign D1-D5): the composer floats above the conversation and uses the glass tokens.
    assert "box-shadow: var(--glass-shadow)" in bar_rule
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

    assert ".agent-thinking" in styles_css
    assert ".agent-thinking-icon" not in styles_css
    assert "agent-thinking-pulse" in styles_css
    assert "agent-thinking-dot" in styles_css
    assert ".agent-typing-cursor" not in styles_css
    assert "agent-typing-cursor-blink" not in styles_css
    # v2 wiring: renderAgentTimeline must be fed via agentReportMessagesForDisplay.
    # Real validator names must not be hard-coded in the shipped bundle: the alias
    # map is sourced from the workspace brand.json via agentValidatorAliases.


def test_agent_memory_has_no_permanent_task_top_block():
    """V1.1 memory may appear in the agent conversation and management UI,
    but must not become a fixed gray task-level panel above the workflow.
    """
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()
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
    script = session_projection_fixture(script)
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
    script = session_projection_fixture(script)
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    html = result.stdout
    assert "进化沉淀" in html
    assert "支持 4" in html
    assert "来源记忆 2" in html
    assert 'data-agent-memory-inline-kind="distillation"' in html


def test_agent_memory_management_view_wires_actions_and_api_paths():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

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
    for memory_type in (
        "join_experience",
        "strategy_experience",
        "risk_analysis_experience",
    ):
        assert f'<option value="{memory_type}">' in index_html
    assert ".agent-memory-load-more" in styles_css
    assert ".agent-memory-item.merged" in styles_css

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


def test_agent_memory_item_actions_do_not_overlap_long_text():
    styles_css = _read_browser_css()

    workspace_rule = _css_rule(styles_css, ".agent-memory-workspace")
    assert "height: clamp(420px, 58dvh, 620px);" in workspace_rule
    assert "max-height: 620px;" in workspace_rule
    assert "min-height: 0;" in workspace_rule

    list_rule = _css_rule(styles_css, ".agent-memory-list")
    assert "align-content: start;" in list_rule
    assert "grid-auto-rows: max-content;" in list_rule

    list_and_detail_rule = _css_rule(styles_css, ".agent-memory-list,\n.agent-memory-detail")
    assert "overflow: auto;" in list_and_detail_rule
    assert "min-height: 0;" in list_and_detail_rule

    pane_rule = _css_rule(styles_css, ".agent-memory-list-pane,\n.agent-memory-detail-pane")
    assert "overflow: hidden;" in pane_rule

    item_rule = _css_rule(styles_css, ".agent-memory-item")
    assert "grid-template-columns: minmax(0, 1fr);" in item_rule
    assert "grid-template-columns: minmax(0, 1fr) auto;" not in item_rule

    main_rule = _css_rule(styles_css, ".agent-memory-item-main")
    assert "overflow-wrap: anywhere;" in main_rule

    actions_rule = _css_rule(styles_css, ".agent-memory-actions")
    assert "justify-content: flex-start;" in actions_rule
    assert "justify-content: flex-end;" not in actions_rule


def test_agent_memory_detail_view_uses_structured_cards():
    styles_css = _read_browser_css()

    for fragment in [
        "agent-memory-detail-header",
        "agent-memory-title-block",
        "agent-memory-badges",
        "agent-memory-summary-card",
        "agent-memory-detail-grid",
        "agent-memory-evidence-card",
        "agent-memory-audit-panel",
        "agent-memory-audit-timeline",
        "agent-memory-audit-event",
    ]:
        assert f".{fragment}" in styles_css


    detail_start = styles_css.index("\n.agent-memory-detail-inner {")
    detail_rule = styles_css[detail_start : styles_css.index("}", detail_start)]
    assert "grid-template-rows: auto auto minmax(0, 1fr);" in detail_rule

    header_rule = _css_rule(styles_css, ".agent-memory-detail-header")
    assert "align-items: start;" in header_rule

    title_rule = _css_rule(styles_css, ".agent-memory-title-block h3")
    assert "overflow-wrap: anywhere;" in title_rule

    grid_rule = _css_rule(styles_css, ".agent-memory-detail-grid")
    assert "grid-template-columns:" in grid_rule


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


def test_agent_advance_intent_accepts_task_start_shortcuts():
    module_url = (STATIC_DIR / "js" / "agent-conversation-view.js").as_uri()
    script = "\n".join(
        [
            f"import {{ agentMessageIsAdvanceIntent }} from {json.dumps(module_url)};",
            "const positive = [",
            "  '开始模型验证',",
            "  '请开始模型验证吧',",
            "  '开始特征分析',",
            "  '开始数据处理',",
            "  '开始建模',",
            "  '开始策略开发',",
            "  '开始风险分析',",
            "];",
            "const negative = ['不要继续', '先不继续', '这是开始模型验证吗？'];",
            "const result = {",
            "  positive: positive.map((content) => agentMessageIsAdvanceIntent({ role: 'user', content })),",
            "  negative: negative.map((content) => agentMessageIsAdvanceIntent({ role: 'user', content })),",
            "};",
            "process.stdout.write(JSON.stringify(result));",
        ]
    )

    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["positive"] == [True] * 7
    assert payload["negative"] == [False] * 3


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
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["蛋黄", "小九", "Agent", "蛋黄"]


@pytest.mark.parametrize("task_type", ["validation", "modeling", "feature_analysis", "strategy", "vintage", "portfolio", "data_join"])
def test_agent_send_without_enabled_model_shows_inline_guidance_before_post(task_type):
    app_js = _read_static("app.js")

    helpers_start = app_js.index("function setAgentComposerNotice")
    helpers_end = app_js.index("function renderAgentEffortPreference", helpers_start)
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    script = "\n".join(
        [
            "let selectedTaskId = 'task-1';",
            f"let selectedTask = {{ task_type: {json.dumps(task_type)} }};",
            "function taskUsesPlanRail(t) { const type = t && t.task_type; return Boolean(type) && type !== 'validation'; }",
            "function selectedTaskNeedsManualRiskIntake() { return false; }",
            "function selectedTaskNeedsDeterministicPortfolioTurn() { return false; }",
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
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
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




def test_agent_send_button_switches_to_stop_control_while_agent_is_executing():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    assert 'data-agent-send-state="send"' in index_html
    assert 'class="agent-send-icon agent-send-icon-send"' in index_html
    assert 'class="agent-send-icon agent-send-icon-stop"' in index_html
    assert '<rect x="4" y="4" width="16" height="16" rx="3.4" fill="currentColor" />' in index_html




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


def test_agent_running_composer_only_submits_explicit_stop_intent():
    app_js = _read_static("app.js")
    helper = _slice_function(app_js, "function agentComposerStopIntent")

    script = f"{helper}\nprocess.stdout.write(JSON.stringify([\n  agentComposerStopIntent('停止当前调参'),\n  agentComposerStopIntent('cancel'),\n  agentComposerStopIntent('不要停止'),\n  agentComposerStopIntent('继续执行'),\n  agentComposerStopIntent('为什么停止了？'),\n  agentComposerStopIntent('开始执行这个计划，先到数据划分确认处停下。'),\n  agentComposerStopIntent('确认开始执行这个计划，并在数据划分确认处停下。')\n]));"
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == [True, True, False, False, False, False, False]


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
            "function selectedTaskNeedsManualRiskIntake() { return false; }",
            "function selectedTaskNeedsDeterministicPortfolioTurn() { return false; }",
            "let agentMessages = [];",
            "let lastAgentRenderSignature = null;",
            "const agentRequestAbortControllers = new Map();",
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
            "const pendingInputValue = input.value;",
            "resolveApi({ status: 'message_saved', messages: [{ role: 'user', stage: 'chat', content: '开始', metadata: {} }, { role: 'assistant', stage: 'chat', content: '收到。', metadata: { streaming: false } }] });",
            "await sendPromise;",
            "process.stdout.write(JSON.stringify({ inputValue: input.value, pendingInputValue, pendingMessages, finalMessages: agentMessages, renderSnapshots }));",
        ]
    )
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["inputValue"] == ""
    assert payload["pendingInputValue"] == ""
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
    helpers_start = app_js.index("function agentMessagePollSignature")
    helpers_end = app_js.index("async function startAgentValidation", helpers_start)
    message_helpers_start = app_js.index("function appendOptimisticAgentUserMessage")
    message_helpers_end = app_js.index("function renderAgentTimeline", message_helpers_start)
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    script = "\n".join(
        [
            f"import {{ agentMessageIsAdvanceIntent }} from {json.dumps(module_url)};",
            "const AGENT_STREAM_POLL_INTERVAL_MS = 1;",
            "const AGENT_STREAM_POLL_IDLE_INTERVAL_MS = 2;",
            "const AGENT_STREAM_POLL_LONG_INTERVAL_MS = 3;",
            "const AGENT_STREAM_POLL_IDLE_AFTER_MS = 2;",
            "const AGENT_STREAM_POLL_LONG_AFTER_MS = 5;",
            "let selectedTaskId = 'task-1';",
            "let selectedTask = { task_type: 'validation' };",
            "function taskUsesPlanRail(t) { const type = t && t.task_type; return Boolean(type) && type !== 'validation'; }",
            "function selectedTaskNeedsManualRiskIntake() { return false; }",
            "function selectedTaskNeedsDeterministicPortfolioTurn() { return false; }",
            "let agentMessages = [];",
            "let lastAgentRenderSignature = null;",
            "const agentRequestAbortControllers = new Map();",
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
            "async function reloadDataWorkspace() {}",
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
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
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


def test_agent_stop_aborts_in_flight_message_request_and_clears_optimistic_state():
    app_js = _read_static("app.js")
    message_helpers_start = app_js.index("function appendOptimisticAgentUserMessage")
    message_helpers_end = app_js.index("function clearAgentStageMessages", message_helpers_start)
    send_start = app_js.index("async function startAgentValidation")
    send_end = app_js.index("async function dispatchAgentValidation", send_start)
    stop_start = app_js.index("async function stopAgentValidation")
    stop_end = app_js.index("async function waitForAgentValidation", stop_start)
    script = "\n".join(
        [
            "let selectedTaskId = 'task-1';",
            "let agentMessages = [];",
            "let lastAgentRenderSignature = null;",
            "const agentRequestAbortControllers = new Map();",
            "const statuses = [];",
            "let messageSignal = null;",
            "let stopCalled = false;",
            "function selectedTaskNeedsManualRiskIntake() { return false; }",
            "function selectedTaskNeedsDeterministicPortfolioTurn() { return false; }",
            "const input = { value: '开始', style: {}, classList: { toggle() {} } };",
            "const modelSelect = { value: 'model-1' };",
            "function $(id) { return id === 'agentComposerInput' ? input : modelSelect; }",
            "function autoGrowComposerInput() {}",
            "function updateAgentSendDisabled() {}",
            "function setActionStatus(message, kind = '') { statuses.push({ message, kind }); }",
            "function renderAgentConversation() {}",
            "function agentMessageIsAdvanceIntent() { return false; }",
            "function agentModelUnavailableMessage() { return ''; }",
            "function showAgentModelGuidance() { return false; }",
            "function setAgentComposerNotice() {}",
            "function agentModelConfigurationErrorMessage() { return ''; }",
            "function agentEffort() { return 'high'; }",
            "function agentAcceptanceModeValue() { return 'normal'; }",
            "function pollAgentMessagesUntilSettled(_taskId, pendingPromise) { return pendingPromise.catch(() => {}); }",
            "function requireTaskId(taskId) { return taskId; }",
            "async function waitForAgentValidation() { throw new Error('wait should not run'); }",
            "async function api(endpoint, options = {}) {",
            "  if (endpoint === 'api/tasks/task-1/agent/messages') {",
            "    messageSignal = options.signal;",
            "    return await new Promise((_resolve, reject) => {",
            "      options.signal.addEventListener('abort', () => {",
            "        const error = new Error('aborted');",
            "        error.name = 'AbortError';",
            "        reject(error);",
            "      });",
            "    });",
            "  }",
            "  if (endpoint === 'api/tasks/task-1/agent/stop') {",
            "    stopCalled = true;",
            "    return { status: 'message_saved', message: '已停止当前动作，请问有什么指示？', messages: [] };",
            "  }",
            "  throw new Error(`unexpected ${endpoint}`);",
            "}",
            app_js[message_helpers_start:message_helpers_end],
            app_js[send_start:send_end],
            app_js[stop_start:stop_end],
            "const sendPromise = startAgentValidation();",
            "await new Promise((resolve) => setTimeout(resolve, 0));",
            "const optimisticCount = agentMessages.length;",
            "await Promise.all([sendPromise, stopAgentValidation()]);",
            "process.stdout.write(JSON.stringify({ stopCalled, signalAborted: messageSignal?.aborted === true, optimisticCount, finalMessages: agentMessages.length, hasController: agentRequestAbortControllers.has('task-1'), statuses }));",
        ]
    )
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["stopCalled"] is True
    assert payload["signalAborted"] is True
    assert payload["optimisticCount"] == 2
    assert payload["finalMessages"] == 0
    assert payload["hasController"] is False
    assert payload["statuses"][-1]["message"] == "已停止当前动作，请问有什么指示？"






def test_agent_composer_acceptance_mode_selector_controls_auto_accept_payload():
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    assert 'id="agentAcceptanceModeSelect"' in index_html
    assert 'class="agent-composer-chip-icon agent-acceptance-icon-default"' in index_html
    assert 'class="agent-composer-chip-icon agent-acceptance-icon-auto"' in index_html
    assert 'value="normal">默认权限</option>' in index_html
    assert 'value="auto_accept">自动审查</option>' in index_html
    assert index_html.index('id="agentAcceptanceModeSelect"') < index_html.index('id="agentModelSelect"')


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
    styles_css = _read_browser_css()

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
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["enabled-default", "enabled-default", "other-model"]


def test_agent_assistant_messages_render_markdown_safely():
    styles_css = _read_browser_css()

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


def test_agent_markdown_rejects_browser_normalization_origin_bypasses():
    unsafe_hrefs = [
        "/\\\\evil.example/steal",
        "\\\\evil.example/steal",
        "///evil.example/steal",
        "/api/tasks/task-1/report\\download",
        "/api/tasks/task-1/report\ndownload",
    ]

    assert _safe_markdown_hrefs(unsafe_hrefs) == [False] * len(unsafe_hrefs)


def test_agent_markdown_preserves_explicit_http_and_same_origin_links():
    safe_hrefs = [
        "https://docs.example.com/guide",
        "http://docs.example.com/guide",
        "/api/tasks/task-1/report/download",
        "#validation-result",
    ]

    assert _safe_markdown_hrefs(safe_hrefs) == [True] * len(safe_hrefs)


def test_agent_markdown_keeps_safe_local_links_below_proxy_mount():
    html = _render_agent_markdown("[下载报告](/api/tasks/task-1/report/download)")

    assert 'href="api/tasks/task-1/report/download"' in html
    assert 'href="/api/tasks/task-1/report/download"' not in html


def test_persisted_download_urls_are_limited_to_same_origin_api_paths():
    unsafe_hrefs = [
        "javascript:alert(1)",
        "data:text/html,boom",
        "//evil.example/report.xlsx",
        "/\\\\evil.example/report.xlsx",
        "/api/tasks/task-1/report\\download",
        "/api/../admin",
        "/api/plugins",
        "https://marvis.example/api/tasks/task-1/report/download",
    ]
    safe_hrefs = [
        "/api/tasks/task-1/report/download",
        "/api/tasks/task-1/report/download?expected_content_hash=abc",
    ]

    assert _safe_api_hrefs(unsafe_hrefs) == [""] * len(unsafe_hrefs)
    assert _safe_api_hrefs(safe_hrefs) == [href.removeprefix("/") for href in safe_hrefs]




def test_report_download_presenter_drops_unsafe_persisted_urls():
    html = _agent_report_download_html(
        [
            {"label": "安全报告", "download_url": "/api/tasks/task-1/report/download"},
            {"label": "脚本", "download_url": "javascript:alert(1)"},
            {"label": "跨站", "download_url": "//evil.example/report.xlsx"},
            {"label": "反斜杠", "download_url": "/\\evil.example/report.xlsx"},
        ]
    )

    assert "安全报告" in html
    assert "api/tasks/task-1/report/download" in html
    assert "javascript:" not in html
    assert "evil.example" not in html
    assert "脚本" not in html
    assert "跨站" not in html
    assert "反斜杠" not in html


def test_agent_markdown_renders_highlighted_code_blocks():
    styles_css = _read_browser_css()
    html = _render_agent_markdown(
        '```python\ndef score(df):\n    threshold = 0.5\n    return df["prob"] > threshold  # pass\n```'
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


def test_agent_markdown_renders_pipe_tables():
    styles_css = _read_browser_css()
    html = _render_agent_markdown(
        "| 文件类型 | 描述 | 信贷风控适用性 |\n|:---|:---|:---|\n| **PFA** | 基于 JSON/YAML 的模型交换标准 | 在部分场景中可被审计 |\n| Pickle (.pkl) | Python 原生序列化格式 | **不推荐上线部署** |"
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


def test_agent_markdown_renders_finding_fail_markup_as_red_strong():
    styles_css = _read_browser_css()
    html = _render_agent_markdown("总体判断：!!过拟合检查未通过（相对差 18%）!!，其余稳定性可接受。")

    assert '<strong class="agent-finding-fail">过拟合检查未通过（相对差 18%）</strong>' in html
    assert "!!" not in html
    assert ".agent-markdown .agent-finding-fail" in styles_css
    assert "color: var(--danger-strong)" in _css_rule(
        styles_css, ".agent-markdown .agent-finding-fail"
    )


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
    script = "await import('./marvis/static/js/pet_catalog.js');\nconst { isSafeAssetUrl, normalizeBranding, normalizeValidatorAliases } = await import('./marvis/static/js/branding.js');\nconst probes = {\n  absolute: isSafeAssetUrl('/branding/assets/logo.png'),\n  relative: isSafeAssetUrl('static/brand/logo.png'),\n  https: isSafeAssetUrl('https://cdn.test/logo.png'),\n  protocolRelative: isSafeAssetUrl('//evil.test/x'),\n  javascript: isSafeAssetUrl('javascript:alert(1)'),\n  data: isSafeAssetUrl('data:text/html,x'),\n};\nconst fallback = normalizeBranding({ logoUrl: 'javascript:alert(1)', workspaceLogoUrl: 'data:image/png,x', faviconUrl: '//evil.test/f.ico' });\nconst aliases = normalizeValidatorAliases({ '  A  ': '  a  ', B: '', C: 5, D: 'd' });\nprocess.stdout.write(JSON.stringify({ probes, logoUrl: fallback.logoUrl, workspaceLogoUrl: fallback.workspaceLogoUrl, faviconUrl: fallback.faviconUrl, aliases, brandingAliases: fallback.validatorAliases }));"
    script = session_projection_fixture(script)
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


def test_step_rail_bottom_padding_is_visually_balanced():
    styles_css = _read_browser_css()

    rail_start = styles_css.index(".progress-rail {")
    rail_end = styles_css.index("}", rail_start)
    rail_rule = styles_css[rail_start:rail_end]

    assert "padding: 12px 12px 8px;" in rail_rule


# ====== Polling render signature guards ======


def _gate_controller_context_region(app_js: str) -> str:
    """Return the executable controller-context declarations as one seam.

    The modeling and JOIN factories may share an implementation; callers only
    depend on the contexts those named factories return.  Keeping the whole
    declaration region makes the behavioral tests insensitive to that internal
    factoring choice.
    """

    shared_start = app_js.find("function agentAcceptanceControllerContext")
    modeling_start = app_js.index("function modelingSetupControllerContext")
    start = modeling_start if shared_start < 0 else min(shared_start, modeling_start)
    end = app_js.index("function handleDriverReportDownloadClick", start)
    return app_js[start:end]


def _run_node_capture_json(script: str) -> dict:
    # These harnesses inline the whole app.js source, which exceeds Linux's
    # 128KB per-argument limit (MAX_ARG_STRLEN) — feed the program via stdin
    # instead of `-e` so the script size is unbounded on every platform.
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module"], input=script, capture_output=True, text=True
    , check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _prepare_app_module_for_node(app_js: str) -> str:
    """Mirror the browser's classic-catalog-before-module startup contract."""

    rewritten = app_js.replace('from "./js/', 'from "./marvis/static/js/')
    return 'import "./marvis/static/js/pet_catalog.js";\n' + _MOUNTED_PANEL_DOM_FIXTURE + rewritten


_MOUNTED_PANEL_DOM_FIXTURE = r'''
// These whole-module fixtures exercise validation renderers and keyed task rows.
// The other mounted panels need concrete DOM controls during module startup;
// their interactions are tested by their own module and real-browser suites.
const fixturePanelIds = new Set([
  'sqlImportPanel', 'governedCapabilitiesPanel', 'workerIsolationPanel',
  'institutionPanel', 'instanceSynchronizationPanel', 'strategyHistoryMappingPanel',
  'dataHistoryMappingPanel',
]);
const fixturePanelDocument = { createElement: tag => fixturePanelNode(tag) };
function fixturePanelNode(tag = 'div') {
  const selectors = new Map();
  const node = {
    tagName: tag.toUpperCase(), ownerDocument: fixturePanelDocument,
    dataset: {}, style: {}, value: '', checked: false, disabled: false, hidden: false,
    textContent: '', innerHTML: '', children: [], files: [],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    get options() { return this.children; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = [...children]; },
    setAttribute() {}, addEventListener() {}, removeEventListener() {}, contains() { return false; },
    querySelector(selector) {
      if (!selectors.has(selector)) {
        const child = fixturePanelNode();
        child.parentElement = this;
        selectors.set(selector, child);
      }
      return selectors.get(selector);
    },
    querySelectorAll() { return []; },
  };
  return node;
}
const fixturePanels = new Map();
const fixtureOriginalGetElement = document.getElementById.bind(document);
document.getElementById = id => {
  if (!fixturePanelIds.has(id)) return fixtureOriginalGetElement(id);
  if (!fixturePanels.has(id)) fixturePanels.set(id, fixturePanelNode());
  return fixturePanels.get(id);
};
'''


def test_validation_evidence_sections_reveal_only_after_their_stage_completes():
    app_js = _read_static("app.js")
    visibility_start = app_js.index("function notebookReproducibilityComplete")
    visibility_end = app_js.index("function workflowIndex", visibility_start)
    visibility_functions = app_js[visibility_start:visibility_end]
    script = f"let selectedTaskId = 'task';\nlet selectedTask = null;\nconst notebookReproducibilityCompleteStatuses = new Set([\n  'executed', 'computing_metrics', 'writing_artifacts', 'succeeded', 'review_required',\n]);\nconst metricOverviewCompleteStatuses = new Set([\n  'writing_artifacts', 'succeeded', 'review_required',\n]);\nfunction taskFailedDuringNotebook(task) {{\n  return task?.status === 'failed' && task?.failure_stage === 'notebook';\n}}\nfunction taskFailedDuringMetrics(task) {{\n  return task?.status === 'failed' && task?.failure_stage === 'metrics';\n}}\nfunction taskFailedDuringReport(task) {{\n  return task?.status === 'failed' && task?.failure_stage === 'report';\n}}\nfunction taskFailureWasRestartReclaim() {{ return false; }}\nfunction workflowStageCompleteFromEvidence() {{ return false; }}\n{visibility_functions}\nfunction visible(task) {{\n  selectedTask = task;\n  selectedTaskId = task.id;\n  return {{\n    reproducibility: shouldShowReproducibilitySection(),\n    metrics: shouldShowMetricSection(),\n  }};\n}}\nprocess.stdout.write(JSON.stringify({{\n  failedNotebook: visible({{ id: 'n', status: 'failed', failure_stage: 'notebook' }}),\n  failedMetrics: visible({{ id: 'm', status: 'failed', failure_stage: 'metrics' }}),\n  failedReport: visible({{ id: 'r', status: 'failed', failure_stage: 'report' }}),\n  notebookComplete: visible({{ id: 'e', status: 'executed' }}),\n  metricsComplete: visible({{ id: 'w', status: 'writing_artifacts' }}),\n}}));"

    data = _run_node_capture_json(_with_workbench_context(script, app_js))

    assert data == {
        "failedNotebook": {"reproducibility": False, "metrics": False},
        "failedMetrics": {"reproducibility": True, "metrics": False},
        "failedReport": {"reproducibility": True, "metrics": True},
        "notebookComplete": {"reproducibility": True, "metrics": False},
        "metricsComplete": {"reproducibility": True, "metrics": True},
    }


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


def test_reproducibility_pass_status_hides_score_compare_rows():
    app_js = _read_static("app.js")
    boot_marker = 'document.addEventListener(\n  "mousedown"'
    boot_idx = app_js.index(boot_marker)
    app_js = _prepare_app_module_for_node(app_js[:boot_idx])

    rows = [
        {
            "row_index": 1,
            "score_code_model": 0.5,
            "score_submitted_pmml": 0.50005,
            "abs_diff": 0.00005,
            "matched": False,
        },
        {
            "row_index": 2,
            "score_code_model": 0.6,
            "score_submitted_pmml": 0.6,
            "abs_diff": 0.0,
            "matched": True,
        },
    ]
    payload = {
        "summary": {"status": "pass", "mismatch_count": 1, "max_abs_diff": 0.00005},
        "sample_size": 1000,
        "seed": 42,
        "rows": rows,
    }
    test_driver = "\n".join(
        [
            "taskSession.selectTask(null, 'task-pass');",
            "taskSession.selectTask({ id: 'task-pass', status: 'executed' });",
            f"renderReproducibilityEvidence({json.dumps(payload)});",
            "const html = globalThis.__writes.at(-1) || '';",
            "process.stdout.write(JSON.stringify({",
            "  hasRows: html.includes('score-compare-list'),",
            "  hasHeader: html.includes('行号'),",
            "  hasStatus: html.includes('一致'),",
            "  hasRandomSeed: html.includes('随机种子'),",
            "  summaryItemCount: (html.match(/class=\"summary-item/g) || []).length,",
            "}));",
        ]
    )

    data = _run_node_capture_json(_BROWSER_STUBS + "\n" + app_js + "\n" + test_driver)

    assert data == {
        "hasRows": False,
        "hasHeader": False,
        "hasStatus": True,
        "hasRandomSeed": False,
        "summaryItemCount": 4,
    }


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
    app_js = _prepare_app_module_for_node(app_js[:boot_idx])

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
            "taskSession.selectTask(null, 'task-A');",
            "taskSession.selectTask({ id: 'task-A', status: 'running', active_job_kind: 'pipeline' });",
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
    app_js = _prepare_app_module_for_node(app_js[: app_js.index(boot_marker)])

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
            "taskSession.selectTask(null, 'task-A');",
            "taskSession.selectTask({ id: 'task-A', status: 'running' });",
            f"renderReproducibilityEvidence({json.dumps(populated)});",
            "const writesAfter_A_populated = globalThis.__writes.length;",
            "const animatedTaskAfter_A = renderSignatures.reproducibilityAnimatedTaskId;",
            # Switch to task B and render the SAME evidence shape — should
            # rebuild (different task) AND re-enable animation.
            "taskSession.selectTask(null, 'task-B');",
            "taskSession.selectTask({ id: 'task-B', status: 'running' });",
            f"renderReproducibilityEvidence({json.dumps(populated)});",
            "const writesAfter_B_populated = globalThis.__writes.length;",
            "const animatedTaskAfter_B = renderSignatures.reproducibilityAnimatedTaskId;",
            "const writeForB = globalThis.__writes[writesAfter_B_populated - 1];",
            # Switch to task C and feed empty evidence — placeholder should
            # appear; previous chart must not be preserved across tasks.
            "taskSession.selectTask(null, 'task-C');",
            "taskSession.selectTask({ id: 'task-C', status: 'running' });",
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
    script = f"let selectedTask = null;\nfunction taskPlanWorkflowStatusSnapshot() {{ return null; }}\nfunction usesPmmlScoringWorkflow() {{ return false; }}\n{stopped_body}\n{snapshot_body}\nconst idle = taskActionStatusSnapshot({{ status: 'writing_artifacts', active_job_kind: null }});\nconst reportBusy = taskActionStatusSnapshot({{ status: 'writing_artifacts', active_job_kind: 'report' }});\nconst metricsRunning = taskActionStatusSnapshot({{ status: 'computing_metrics', active_job_kind: 'metrics' }});\nprocess.stdout.write(JSON.stringify({{ idle, reportBusy, metricsRunning }}));"

    script = session_projection_fixture(script)
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
    assert snapshots["reportBusy"]["message"] == "报告输出进行中。"
    assert snapshots["reportBusy"]["kind"] == "busy"
    assert "Word" in snapshots["reportBusy"]["detail"]
    assert snapshots["metricsRunning"]["message"] == "指标概览进行中。"
    assert snapshots["metricsRunning"]["kind"] == "busy"
    assert "压力测试" in snapshots["metricsRunning"]["detail"]


def test_task_stopped_uses_structured_stopped_field_only():
    app_js = _read_static("app.js")
    stopped_body = _slice_function(app_js, "function taskStopped")
    script = f"let selectedTask = null;\n{stopped_body}\nconst structured = taskStopped({{ stopped: true, status_message: '普通状态' }});\nconst legacy = taskStopped({{ status_message: '已停止当前动作' }});\nconst active = taskStopped({{ stopped: false, status_message: '普通状态' }});\nprocess.stdout.write(JSON.stringify({{ structured, legacy, active }}));"

    script = session_projection_fixture(script)
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
    # Batch 4 (2026-10-14 redesign): taskStatusTone resolves through js/task-status.js.
    context_body = _slice_function(app_js, "function taskStatusContext")
    resolved_body = _slice_function(app_js, "function taskResolvedStatus")
    task_tone_body = _slice_function(app_js, "function taskStatusTone")
    script = "\n".join(
        [
            'import { resolveTaskStatus } from "./marvis/static/js/task-status.js";',
            "let selectedTask = null;",
            "function taskPlanWorkflowStatusSnapshot() { return null; }",
            stopped_body,
            context_body,
            resolved_body,
            task_tone_body,
            ("const idle = taskStatusTone({"
            " status: 'writing_artifacts', active_job_kind: null });"),
            ("const reportBusy = taskStatusTone({"
            " status: 'writing_artifacts', active_job_kind: 'report' });"),
            "process.stdout.write(JSON.stringify({ idle, reportBusy }));",
        ]
    )

    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    tones = json.loads(result.stdout)
    # Batch 4 (2026-10-14 redesign): metrics done and the report not started is 待继续 (review tone), as in the title.
    assert tones["idle"] == "review", (
        "writing_artifacts with no active report job means the user starts "
        "the report next, not a running task."
    )
    assert tones["reportBusy"] == "run"







def test_agent_typewriter_resume_after_completion_seeds_visible():
    """If the server flips a message id back to streaming=true after it had
    already finished, the next render must NOT clear the visible content and
    re-type from byte 0. The seeded-visible path must engage instead.
    """
    app_js = _read_static("app.js")
    streaming_body = _slice_function(app_js, "function agentMessageIsStreaming")
    visible_body = _slice_function(app_js, "function agentVisibleContent")
    sched_body = _slice_function(app_js, "function scheduleAgentTyping")

    script = f"const window = {{ setTimeout: () => 1, clearTimeout: () => {{}} }};\nconst agentTypingState = new Map();\nconst agentTypingCompleted = new Map();\nlet agentTypingTimer = null;\nlet lastAgentRenderSignature = null;\nconst AGENT_TYPEWRITER_INTERVAL_MS = 12;\nfunction renderAgentConversation() {{}}\nfunction tickAgentTyping() {{}}\n{streaming_body}\n{sched_body}\n{visible_body}\nagentTypingState.set('m1', {{ visible: 'hello world', target: 'hello world' }});\nconst settled = {{ id: 'm1', role: 'assistant', content: 'hello world', metadata: {{ streaming: false }} }};\nagentVisibleContent(settled);\nconst completedAfterSettle = agentTypingCompleted.get('m1');\nconst resumed = {{ id: 'm1', role: 'assistant', content: 'hello world and then some', metadata: {{ streaming: true }} }};\nconst visibleOnResume = agentVisibleContent(resumed);\nconst typingOnResume = agentTypingState.get('m1');\nprocess.stdout.write(JSON.stringify({{ completedAfterSettle, visibleOnResume, seededVisible: typingOnResume ? typingOnResume.visible : null, targetOnResume: typingOnResume ? typingOnResume.target : null,}}));"

    script = session_projection_fixture(script)
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




def test_modeling_and_join_controller_contexts_expose_the_same_behavior():
    """The two named controller seams expose the same accepted dependencies.

    This characterizes the caller-visible contract while allowing their shared
    implementation to move behind either factory.
    """
    app_js = _read_static("app.js")
    script = "\n".join(
        [
            "let selectedTaskId = 'task-A';",
            "let agentMessages = [];",
            "function api() {}",
            "function agentAcceptanceModeValue() { return 'manual'; }",
            "function setActionStatus() {}",
            "function renderAgentConversation() {}",
            "function pollAgentMessagesUntilSettled() { return Promise.resolve(); }",
            "function renderWorkflowStepper() {}",
            "function setBusy() {}",
            "const planRailController = { resetFetchThrottle() {} };",
            _gate_controller_context_region(app_js),
            "const modeling = modelingSetupControllerContext();",
            "const join = joinGateControllerContext();",
            "process.stdout.write(JSON.stringify({",
            "  modelingKeys: Object.keys(modeling).sort(),",
            "  joinKeys: Object.keys(join).sort(),",
            "  modelingTaskId: modeling.getSelectedTaskId(),",
            "  joinTaskId: join.getSelectedTaskId(),",
            "  modelingAcceptance: modeling.agentAcceptanceModeValue(),",
            "  joinAcceptance: join.agentAcceptanceModeValue(),",
            "}));",
        ]
    )
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["modelingKeys"] == payload["joinKeys"]
    assert payload["modelingTaskId"] == payload["joinTaskId"] == "task-A"
    assert payload["modelingAcceptance"] == payload["joinAcceptance"] == "manual"


def test_gate_controller_context_setter_drops_stale_task_messages_behaviorally():
    """Late task-A responses must not overwrite task B's live conversation."""
    app_js = _read_static("app.js")
    script = "\n".join(
        [
            "let selectedTaskId = 'task-A';",
            "let agentMessages = [];",
            "function api() {}",
            "function agentAcceptanceModeValue() { return 'manual'; }",
            "function setActionStatus() {}",
            "function renderAgentConversation() {}",
            # UX-1: the context factories now also carry busy-state plumbing
            # (REL-1 job-wrapped driver turns) — stub it so the sliced source
            # executes; this test's own scope is only the stale-message guard.
            "function pollAgentMessagesUntilSettled() { return Promise.resolve(); }",
            "function renderWorkflowStepper() {}",
            "const planRailController = { resetFetchThrottle() {} };",
            _gate_controller_context_region(app_js),
            "const results = {};",
            ("for (const [name, factory] of Object.entries({"
            " modelingSetupControllerContext,"
            " joinGateControllerContext,"
            " screenGateControllerContext,"
            " driverConfirmControllerContext,"
            "})) {"),
            "  selectedTaskId = 'task-A'; taskRequests.select('task-A');",
            "  agentMessages = ['seed'];",
            "  const ctx = factory();",
            "  selectedTaskId = 'task-B'; taskRequests.select('task-B');",
            "  ctx.setAgentMessages(['A-turn-finished-late']);",
            "  results[name] = agentMessages.slice();",
            "}",
            "process.stdout.write(JSON.stringify(results));",
        ]
    )
    script = _with_workbench_context(script, app_js)
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(result.stdout)
    for name, agent_messages in results.items():
        assert agent_messages == ["seed"], (
            f"{name}: a task-A response resolving after switching to task B "
            f"must not overwrite agentMessages (got {agent_messages!r})."
        )


def test_submit_driver_confirm_shows_busy_state_and_polls_before_response_resolves():
    """UX-1 step 1: clicking a driver gate's confirm button must show the busy
    pill and start polling agent messages *immediately* (before the POST
    resolves), since the backend now runs the whole turn inside a job (REL-1)
    and can take minutes. Also guards that the pre-existing "task switched
    while pending" no-op guard on setAgentMessages (UX-3) still holds once the
    busy-state polling plumbing is wired through the same context.
    """
    module_url = (STATIC_DIR / "js" / "v2" / "driver_gate_confirm.js").as_uri()
    script = "\n".join(
        [
            f"import {{ submitDriverConfirm }} from {json.dumps(module_url)};",
            "const statusCalls = [];",
            "function setActionStatus(message, kind) { statusCalls.push({ message, kind }); }",
            "const pollCalls = [];",
            "function pollAgentMessagesUntilSettled(taskId, pendingPromise) {",
            "  pollCalls.push(taskId);",
            "  return pendingPromise.then(() => {}, () => {});",
            "}",
            "const railTicks = [];",
            "function resetFetchThrottle(taskId) { railTicks.push(['reset', taskId]); }",
            "function renderWorkflowStepper(opts) { railTicks.push(['render', opts && opts.force]); }",
            "let selectedTaskId = 'task-A';",
            "let agentMessages = ['seed'];",
            "let resolveApi;",
            "function api() { return new Promise((resolve) => { resolveApi = resolve; }); }",
            "const attrs = {",
            "  'data-expected-plan-id': 'plan-A',",
            "  'data-expected-step-id': 'gate-A',",
            "  'data-expected-plan-status': 'awaiting_confirm',",
            "  'data-expected-plan-revision': '0',",
            "  'data-expected-plan-fingerprint': 'a'.repeat(64),",
            "  'data-expected-step-fingerprint': 'b'.repeat(64),",
            "};",
            "const button = { disabled: false, getAttribute: (name) => attrs[name] || '' };",
            "const context = {",
            "  getSelectedTaskId: () => selectedTaskId,",
            "  api,",
            "  setActionStatus,",
            "  setAgentMessages: (messages) => {",
            "    if (selectedTaskId !== 'task-A') return;",
            "    agentMessages = messages || agentMessages;",
            "  },",
            "  renderAgentConversation: () => {},",
            "  pollAgentMessagesUntilSettled,",
            "  resetFetchThrottle,",
            "  renderWorkflowStepper,",
            "};",
            "const confirmPromise = submitDriverConfirm(button, context);",
            "await new Promise((resolve) => setTimeout(resolve, 0));",
            "const busyCallBeforeResolve = statusCalls[0];",
            "const buttonDisabledDuring = button.disabled;",
            "const polledBeforeResolve = pollCalls.slice();",
            # Simulate switching to task B while the confirm's turn is still
            # running server-side, then the (stale) request finally resolving.
            "selectedTaskId = 'task-B';",
            "resolveApi({ messages: ['A-turn-finished-late'] });",
            "await confirmPromise;",
            "process.stdout.write(JSON.stringify({",
            "  busyCallBeforeResolve, buttonDisabledDuring, polledBeforeResolve,",
            "  railTicks, agentMessagesAfter: agentMessages,",
            "}));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["busyCallBeforeResolve"] == {"message": "正在执行下一步…", "kind": "busy"}
    assert payload["buttonDisabledDuring"] is True
    assert payload["polledBeforeResolve"] == ["task-A"]
    # the plan-rail ticker fired at least once while the request was pending,
    # and the final finally-block tick still runs after the response resolves.
    assert ["render", True] in payload["railTicks"]
    # UX-3: task switched away before the stale response resolved, so the
    # setAgentMessages guard inside this call's own context must still drop it.
    assert payload["agentMessagesAfter"] == ["seed"]


def test_recompute_agent_auto_scroll_follow_toggles_on_position():
    """Any wheel that leaves the bottom must drop follow-mode to false, even
    if the user only nudged up by a few pixels. Returning to the bottom (or
    within sub-pixel rounding of it) must re-engage follow-mode so the next
    typewriter tick resumes streaming.
    """
    app_js = _read_static("app.js")
    recompute_body = _slice_function(app_js, "function recomputeAgentAutoScrollFollow")

    script = f"let scrollContent = {{ scrollHeight: 1000, clientHeight: 400, scrollTop: 0 }};\nfunction $(id) {{ return id === 'resultScrollContent' ? scrollContent : null; }}\nlet isAgent = true;\nfunction selectedTaskIsAgentMode() {{ return isAgent; }}\nlet agentAutoScrollFollows = true;\nconst AGENT_AUTO_SCROLL_BOTTOM_TOLERANCE_PX = 2;\nconst AGENT_USER_SCROLL_INPUT_WINDOW_MS = 250;\nlet lastUserScrollInputAt = 0;\nconst performance = {{ now: () => 100 }};\n{recompute_body}\nlastUserScrollInputAt = 100;\nscrollContent.scrollTop = 600;\nrecomputeAgentAutoScrollFollow();\nconst atBottom = agentAutoScrollFollows;\nscrollContent.scrollTop = 590;\nrecomputeAgentAutoScrollFollow();\nconst slightlyAbove = agentAutoScrollFollows;\nscrollContent.scrollTop = 200;\nrecomputeAgentAutoScrollFollow();\nconst farAbove = agentAutoScrollFollows;\nscrollContent.scrollTop = 599.4;\nrecomputeAgentAutoScrollFollow();\nconst subpixelRounding = agentAutoScrollFollows;\nisAgent = false;\nscrollContent.scrollTop = 0;\nrecomputeAgentAutoScrollFollow();\nconst nonAgentLeft = agentAutoScrollFollows;\nprocess.stdout.write(JSON.stringify({{ atBottom, slightlyAbove, farAbove, subpixelRounding, nonAgentLeft }}));"

    script = session_projection_fixture(script)
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

    script = f"const window = {{ setTimeout: () => 1, clearTimeout: () => {{}} }};\nconst agentTypingState = new Map();\nconst agentTypingCompleted = new Map();\nlet agentTypingTimer = null;\nlet lastAgentRenderSignature = null;\nconst AGENT_TYPEWRITER_INTERVAL_MS = 12;\nconst AGENT_TYPEWRITER_CHARS_PER_TICK = 2;\nconst AGENT_TYPEWRITER_CATCHUP_TICKS = 15;\nfunction renderAgentConversation() {{}}\n{streaming_body}\n{sched_body}\n{tick_body}\n{visible_body}\nconst longContent = 'A'.repeat(200);\nconst streaming = {{ id: 'm1', role: 'assistant', content: longContent.slice(0, 10), metadata: {{ streaming: true }} }};\nconst first = agentVisibleContent(streaming);\nagentTypingState.get('m1').visible = longContent.slice(0, 5);\nconst settled = {{ id: 'm1', role: 'assistant', content: longContent, metadata: {{ streaming: false }} }};\nconst afterStreamEnd = agentVisibleContent(settled);\nconst stateAfterStreamEnd = agentTypingState.has('m1');\ntickAgentTyping();\nconst afterTick = agentTypingState.get('m1');\nprocess.stdout.write(JSON.stringify({{ first, afterStreamEnd, stateAfterStreamEnd, visibleAfterTick: afterTick ? afterTick.visible : null, targetAfterTick: afterTick ? afterTick.target : null,}}));"

    script = session_projection_fixture(script)
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

    script = f"const window = {{ setTimeout: () => 1, clearTimeout: () => {{}} }};\nconst agentTypingState = new Map();\nconst agentTypingCompleted = new Map();\nlet agentTypingTimer = null;\nlet lastAgentRenderSignature = null;\nconst AGENT_TYPEWRITER_INTERVAL_MS = 12;\nconst AGENT_TYPEWRITER_CHARS_PER_TICK = 2;\nconst AGENT_TYPEWRITER_CATCHUP_TICKS = 15;\nfunction renderAgentConversation() {{}}\n{streaming_body}\n{sched_body}\n{tick_body}\n{visible_body}\nconst bigContent = 'B'.repeat(3000);\nconst message = {{ id: 'm1', role: 'assistant', content: bigContent, metadata: {{ streaming: true }} }};\nagentVisibleContent(message);\nlet ticks = 0;\nwhile (agentTypingState.get('m1').visible.length < agentTypingState.get('m1').target.length) {{\n  tickAgentTyping();\n  ticks += 1;\n  if (ticks > 500) break;\n}}\nprocess.stdout.write(JSON.stringify({{ ticks }}));"

    script = session_projection_fixture(script)
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


def test_driver_gate_tables_render_databar_psi_and_champion_row():
    """VD-1: agentMessageTablesHtml (the generic plan-driver inline table
    renderer used by JOIN diagnostics / feature metrics / model comparison
    gate tables) must sink the same rich-cell language the validation metric
    preview already has — databar for match/hit-rate columns, PSI three-band
    cells, tabular-nums right-aligned numeric columns, and a highlighted
    champion row for backend-flagged (" \u2605" suffixed) winning candidates —
    instead of leaving every cell as bare escaped text.
    """
    app_js = _read_static("app.js")
    metric_tables_js = _read_static("js/metric-tables.js")
    kind_body = _slice_function(app_js, "function driverColumnKindFromHeader")
    cell_body = _slice_function(app_js, "function driverTableCellHtml")
    align_body = _slice_function(metric_tables_js, "export function metricHeaderShouldRightAlign")
    chart_html_body = _slice_function(app_js, "function driverTableChartHtml")
    tables_body = _slice_function(app_js, "function agentMessageTablesHtml")

    ui_utils_url = (STATIC_DIR / "js" / "ui-utils.js").as_uri()
    render_metrics_url = (STATIC_DIR / "js" / "render-metrics.js").as_uri()
    workflow_widgets_url = (STATIC_DIR / "js" / "v2" / "workflow_widgets.js").as_uri()

    script = "\n".join(
        [
            f"import {{ escapeHtml }} from {json.dumps(ui_utils_url)};",
            f"import {{ renderWorkflowDataWidget }} from {json.dumps(workflow_widgets_url)};",
            f"import {{ platformLabelSource }} from {json.dumps((STATIC_DIR / 'js/platform-labels.js').as_uri())};",
            ("import {"
            " columnFractions, parseNumeric, psiTier, psiTooltipText,"
            f" }} from {json.dumps(render_metrics_url)};"),
            kind_body,
            cell_body,
            align_body,
            chart_html_body,
            tables_body,
            "const message = {",
            "  metadata: {",
            "    tables: [",
            "      {",
            "        title: '拼接诊断(逐特征表)',",
            "        columns: ['特征表', '匹配键', '命中率', '键唯一'],",
            "        rows: [",
            "          ['features.parquet', 'id=id', '92.30%', '是'],",
            "          ['bureau.parquet', 'id=id', '41.10%', '否'],",
            "        ],",
            "      },",
            "      {",
            "        title: '特征指标',",
            "        columns: ['特征', 'IV', 'KS', 'PSI', '样本量'],",
            "        rows: [",
            "          ['age', '0.31', '0.42', '0.015', '12000'],",
            "          ['income', '0.12', '0.20', '0.25', '12000'],",
            "        ],",
            "      },",
            "      {",
            "        title: '候选模型对比',",
            "        columns: ['算法', 'train_ks', 'test_ks', 'oot_ks'],",
            "        rows: [",
            "          ['lightgbm ★', '0.55', '0.50', '0.48'],",
            "          ['xgboost', '0.50', '0.45', '0.40'],",
            "        ],",
            "      },",
            "    ],",
            "  },",
            "};",
            "const html = agentMessageTablesHtml(message);",
            "process.stdout.write(JSON.stringify({ html }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    html = json.loads(result.stdout)["html"]

    # 命中率 (match/hit-rate) gets a databar, not a bare cell.
    assert 'class="databar"' in html
    assert 'data-tip="命中率 92.30%"' in html
    # PSI gets the three-band tiered cell, not a bare 4-decimal string.
    assert 'class="psi-cell"' in html
    assert 'data-tier=' in html
    # 样本量 (row count) is tabular-nums right-aligned, not left-aligned text.
    assert '<td class="cell-number" translate="no">12000</td>' in html
    # The backend's " ★"-flagged champion candidate renders as a highlighted
    # badge, not a literal star character glued onto plain text.
    assert 'class="champion-badge"' in html
    assert "lightgbm ★" not in html
    assert ">lightgbm<" in html



def test_agent_gate_reply_is_not_wrapped_in_a_large_coloured_card():
    """Agent prose stays conversational; only nested functional widgets own cards."""
    styles_css = _read_browser_css()

    assert ".gate-card {" not in styles_css




def test_agent_mode_latest_gate_keeps_typed_controls_interactive_with_chat():
    """UX-2: the agent-mode chat timeline must mount the SAME structured gate
    widgets manual mode uses (screening table / dedup picker / modeling setup
    panel / C1 role form) instead of a bare text bubble + confirm button.

    This drives the real agentMessageHtml() (via the full app.js module, same
    harness as the reproducibility tests) so the assertions exercise actual
    production wiring, not a hand-rolled stand-in. Three things are pinned:
    1. The latest gate reuses the same evidence widget as an interactive typed
       control while keeping the natural-language reply visible.
    2. An OLDER (non-latest) gate message's widget renders read-only
       (disabled inputs, data-screen-readonly="true") — the stale-gate guard.
    3. The free-text composer contract is untouched: agentMessageHtml keeps
       rendering the message content bubble regardless of the widget, so free
       text remains a second channel alongside the structured controls.
    """
    app_js = _read_static("app.js")
    boot_marker = 'document.addEventListener(\n  "mousedown"'
    app_js = _prepare_app_module_for_node(app_js[: app_js.index(boot_marker)])

    messages = [
        {
            "id": "old-screen",
            "role": "assistant",
            "stage": "chat",
            "content": "第一次筛选完成。",
            "metadata": {
                "kind": "gate",
                "step_id": "gate-old",
                "screen": {"selected": ["x1"], "thresholds": {"leakage_ks": 0.4, "max_missing_rate": 0.95}},
            },
        },
        {
            "id": "latest-screen",
            "role": "assistant",
            "stage": "chat",
                "content": "阈值调整后重新筛选完成。确认请回复「确认」继续；可直接操作下方控件，或用文字说明要调整的参数。",
            "metadata": {
                "kind": "gate",
                "plan_id": "plan-new",
                "step_id": "gate-new",
                "confirmation_snapshot": {
                    "expected_plan_status": "awaiting_confirm",
                    "expected_plan_revision": 2,
                    "expected_plan_fingerprint": "a" * 64,
                    "expected_step_fingerprint": "b" * 64,
                },
                "screen": {"selected": ["x2"], "thresholds": {"leakage_ks": 0.35, "max_missing_rate": 0.9}},
            },
        },
    ]

    test_driver = "\n".join(
        [
            f"const messages = {json.dumps(messages)};",
            "taskSession.selectTask(null, 'task-gate');",
            "taskSession.selectTask({ id: 'task-gate', task_type: 'modeling', run_mode: 'agent', active_job_kind: null });",
            "taskCache = [taskSession.task];",
            "globalThis.fetch = async (url) => ({ ok: true, json: async () => ({ plans: [{",
            "  id: 'plan-new', task_id: 'task-gate', status: 'awaiting_confirm', steps: [{",
            "    id: 'gate-new', status: 'awaiting_confirm',",
            "    confirmation_snapshot: messages[1].metadata.confirmation_snapshot,",
            "  }],",
            "}] }) });",
            "planRailController.resetFetchThrottle('task-gate');",
            "await planRailController.maybeFetchPlan('task-gate');",
            "const oldHtml = agentMessageHtml(messages[0], 'chat', { isLatestGate: false, conversationOnly: true });",
            "const latestHtml = agentMessageHtml(messages[1], 'chat', { isLatestGate: true, conversationOnly: true });",
            "process.stdout.write(JSON.stringify({ oldHtml, latestHtml }));",
        ]
    )
    script = _BROWSER_STUBS + "\n" + app_js + "\n" + test_driver
    payload = _run_node_capture_json(script)
    old_html = payload["oldHtml"]
    latest_html = payload["latestHtml"]

    # Both render the SAME structured screen-table control classes manual mode
    # uses (screen_gate_controller.js) — no separate agent-only markup.
    assert 'class="screen-table-wrap"' in old_html
    assert 'class="screen-table-wrap"' in latest_html
    assert 'class="screen-pick"' in old_html
    assert 'class="screen-pick"' in latest_html
    assert 'class="gate-card"' not in old_html
    assert 'class="gate-card"' not in latest_html

    # The free-text bubble is still rendered alongside the widget in both
    # cases — agent mode's free-text channel is not replaced by the widget.
    assert "agent-message-content" in old_html
    assert "agent-message-content" in latest_html
    assert "第一次筛选完成" in old_html
    assert "阈值调整后重新筛选完成" in latest_html
    assert "可直接操作下方控件" in latest_html
    assert "自然语言表达同意并继续" in latest_html
    assert "回复「确认」" not in latest_html

    # Only historical Agent-mode gates are evidence-only. The latest gate keeps
    # its typed action path, alongside the unchanged natural-language channel.
    assert 'data-screen-readonly="true"' in old_html
    assert 'data-screen-readonly="true"' not in latest_html
    assert old_html.count(" disabled") > 0
    assert 'data-screen-step-id="gate-new"' in latest_html
    assert 'data-screen-plan-id="plan-new"' in latest_html
    assert 'data-screen-confirm="latest-screen"' in latest_html
    assert 'data-expected-plan-revision="2"' in latest_html
    assert 'data-expected-step-fingerprint="' + ("b" * 64) + '"' in latest_html
    assert "历史筛选结果" in old_html

    # Gates with a structured widget do NOT also render the plain
    # driver-gate-actions confirm button (the widget owns the primary action).
    assert "driver-gate-actions" not in old_html
    assert "driver-gate-actions" not in latest_html


def test_agent_mode_plain_gate_uses_natural_language_without_confirm_button():
    """UX-2: a gate message with no structured payload (no screen/dedup/
    modeling_setup/join_c1 — e.g. a plain "上一步已完成，确认继续" step) must
    use the conversation channel and never offer a parallel click path.
    """
    app_js = _read_static("app.js")
    boot_marker = 'document.addEventListener(\n  "mousedown"'
    app_js = _prepare_app_module_for_node(app_js[: app_js.index(boot_marker)])

    message = {
        "id": "plain-gate",
        "role": "assistant",
        "stage": "chat",
        "content": "上一步已完成。请回复「继续」或说明需要调整的内容。",
        "metadata": {"kind": "gate", "step_id": "gate-plain"},
    }
    test_driver = "\n".join(
        [
            f"const message = {json.dumps(message)};",
            "const html = agentMessageHtml(message, 'chat', { isLatestGate: true, conversationOnly: true });",
            "process.stdout.write(JSON.stringify({ html }));",
        ]
    )
    script = _BROWSER_STUBS + "\n" + app_js + "\n" + test_driver
    html = _run_node_capture_json(script)["html"]

    assert "driver-gate-actions" not in html
    assert 'data-driver-confirm="1"' not in html
    assert "自然语言表达同意并继续" in html
    assert "请回复" not in html


def test_agent_mode_plan_overview_explains_semantic_authorization_without_keywords():
    app_js = _read_static("app.js")
    boot_marker = 'document.addEventListener(\n  "mousedown"'
    app_js = _prepare_app_module_for_node(app_js[: app_js.index(boot_marker)])

    message = {
        "id": "plan-overview",
        "role": "assistant",
        "stage": "plan",
        "content": "计划已生成。手动模式请点击「开始执行」；Agent 模式请回复「开始」或「继续」。",
        "metadata": {"kind": "plan_overview"},
    }
    test_driver = "\n".join(
        [
            f"const message = {json.dumps(message)};",
            "const html = agentMessageHtml(message, 'plan', { conversationOnly: true });",
            "process.stdout.write(JSON.stringify({ html }));",
        ]
    )
    script = _BROWSER_STUBS + "\n" + app_js + "\n" + test_driver
    html = _run_node_capture_json(script)["html"]

    assert "手动模式可点击「开始执行」" in html
    assert "Agent 模式可直接用自然语言表达同意并开始执行" in html
    assert "请回复「开始」" not in html


def test_manual_mode_gate_strips_chat_instruction_and_keeps_button():
    module_url = (STATIC_DIR / "js" / "v2" / "driver_manual_analysis.js").as_uri()
    script = "\n".join(
        [
            f'import {{ driverManualAnalysisHtml }} from {json.dumps(module_url)};',
            "const message = { id: 'm1', role: 'assistant', content: '结果已生成。\\n\\nAgent 模式请回复「继续」或「确认」；如需调整，直接用自然语言说明。', metadata: { kind: 'gate', step_id: 's1' } };",
            "const html = driverManualAnalysisHtml([message], { renderAgentMarkdown: (x) => x, renderGateConfirm: () => '<button>确认并继续</button>' });",
            "process.stdout.write(html);",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Agent 模式请回复" not in result.stdout
    assert "确认并继续" in result.stdout


def test_manual_gate_interactivity_requires_awaiting_step_and_no_driver_busy():
    module_url = (STATIC_DIR / "js" / "v2" / "driver_manual_analysis.js").as_uri()
    script = "\n".join(
        [
            f'import {{ driverManualAnalysisHtml }} from {json.dumps(module_url)};',
            "const message = { id: 'm1', role: 'assistant', content: '确认筛选。', metadata: { kind: 'gate', plan_id: 'p1', step_id: 's1', screen: { selected: ['x1'] } } };",
            "const render = (status, busy) => driverManualAnalysisHtml([message], {",
            "  renderAgentMarkdown: (x) => x,",
            "  stepStatus: () => status,",
            "  isGateActionable: () => !busy,",
            "  renderScreenTable: (_message, options) => `<div data-screen-interactive=\"${options.interactive}\"></div>`,",
            "});",
            "process.stdout.write(JSON.stringify({",
            "  done: render('done', false),",
            "  busy: render('awaiting_confirm', true),",
            "  ready: render('awaiting_confirm', false),",
            "}));",
        ]
    )
    payload = _run_node_capture_json(script)
    assert 'data-screen-interactive="false"' in payload["done"]
    assert 'is-gate-pending' not in payload["done"]
    assert 'data-screen-interactive="false"' in payload["busy"]
    assert 'is-gate-pending' not in payload["busy"]
    assert 'data-screen-interactive="true"' in payload["ready"]
    assert 'is-gate-pending' in payload["ready"]


def test_agent_mode_widget_submit_payload_matches_manual_mode_controller():
    """UX-2: the agent-mode-mounted widget must post through the exact same
    controller function manual mode uses (screenGateControllerContext /
    submitScreenSelection), so the request payload shape (selection +
    expected_step_id, mode-independent /agent/messages POST) is identical
    regardless of which mode mounted the control — there is no agent-only
    branch that could drift from the manual-mode contract.
    """
    module_url = (STATIC_DIR / "js" / "v2" / "screen_gate_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ renderScreenGateTable, submitScreenSelection }} from {json.dumps(module_url)};",
            "const message = {",
            "  id: 'agent-screen-msg',",
            "  metadata: {",
            "    kind: 'gate',",
            "    step_id: 'gate-agent',",
            "    screen: { selected: ['x1', 'x2'], thresholds: { leakage_ks: 0.35, max_missing_rate: 0.9 } },",
            "  },",
            "};",
            # This is the exact renderer app.js's agentMessageGateBodyHtml calls
            # for a screen gate — proves the DOM the widget produces is postable
            # through the same submit function manual mode uses.
            "const html = renderScreenGateTable(message, { interactive: true });",
            "const calls = [];",
            "const wrap = {",
            "  dataset: {",
            "    screenReadonly: 'false',",
            "    screenPlanId: 'plan-agent',",
            "    screenStepId: 'gate-agent',",
            "    expectedPlanStatus: 'awaiting_confirm',",
            "    expectedPlanRevision: '0',",
            "    expectedPlanFingerprint: 'a'.repeat(64),",
            "    expectedStepFingerprint: 'b'.repeat(64),",
            "  },",
            "  querySelectorAll: (selector) => selector === '.screen-pick:checked' ? [",
            "    { value: 'x1', disabled: false, closest: () => ({ classList: { contains: () => false } }) },",
            "  ] : [],",
            "};",
            "const button = { disabled: false, closest: () => wrap };",
            "const context = {",
            "  selectedTaskId: 'task-agent',",
            "  agentAcceptanceModeValue: () => 'manual',",
            "  setActionStatus: () => {},",
            "  setAgentMessages: () => {},",
            "  renderAgentConversation: () => {},",
            "  api: async (url, options) => { calls.push([url, JSON.parse(options.body)]); return { messages: [] }; },",
            "};",
            "await submitScreenSelection(button, context);",
            "process.stdout.write(JSON.stringify({ html, calls }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert 'class="screen-table-wrap"' in payload["html"]
    assert payload["calls"], "submitScreenSelection did not POST"
    url, body = payload["calls"][0]
    assert url == "/api/tasks/task-agent/agent/messages"
    # Same structured fields the review's fix step 2 calls out: selection +
    # expected_step_id (the mode-independent contract validation_agent.py and
    # gate_response_adapter.py already accept).
    assert body["selection"] == ["x1"]
    assert body["expected_plan_id"] == "plan-agent"
    assert body["expected_step_id"] == "gate-agent"
    assert body["expected_plan_status"] == "awaiting_confirm"
    assert body["expected_plan_revision"] == 0
    assert body["expected_plan_fingerprint"] == "a" * 64
    assert body["expected_step_fingerprint"] == "b" * 64
    assert "content" in body


def test_skeleton_templates_render_block_rows_and_table_shapes():
    """VD-3: the shared skeleton.js template helpers must emit the three
    documented shapes (block / rows / table), each using the generic
    `.skeleton` shimmer primitive so a single CSS rule drives all of them.
    """
    module_url = (STATIC_DIR / "js" / "skeleton.js").as_uri()
    script = "\n".join(
        [
            ("import {"
            " skeletonBlockHtml, skeletonRowsHtml, skeletonTableHtml,"
            f" }} from {json.dumps(module_url)};"),
            "process.stdout.write(JSON.stringify({",
            "  block: skeletonBlockHtml(),",
            "  rows: skeletonRowsHtml({ rows: 3 }),",
            "  table: skeletonTableHtml({ rows: 2, columns: 3 }),",
            "}));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert 'class="skeleton skeleton-block"' in payload["block"]
    assert payload["rows"].count('class="skeleton skeleton-row"') == 3
    assert payload["table"].count("skeleton-table-cell-head") == 3  # header row, 3 columns
    assert payload["table"].count('class="skeleton-table-row"') == 2  # 2 data rows
    assert payload["table"].count('class="skeleton skeleton-table-cell"') == 3 * 2  # 2 rows x 3 cols


def test_plan_rail_shows_skeleton_only_on_genuine_first_load():
    """VD-3: the plan rail's first fetch for a task (nothing cached yet) must
    render a table/row skeleton instead of the old blank-then-"计划生成中…"
    text, matching the busy-pill coexistence requirement (different DOM
    region, no visual fight). Once a response has landed at least once, later
    still-empty renders fall back to the plain text — the skeleton must not
    re-flash on every poll tick.
    """
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ createPlanRailController }} from {json.dumps(module_url)};",
            "const elements = {",
            "  progressRail: { setAttribute() {} },",
            "  workflowStepper: { innerHTML: '' },",
            "};",
            "function $(id) { return elements[id] || null; }",
            "globalThis.document = { querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [] }) });",
            "const controller = createPlanRailController({",
            "  $,",
            "  getSelectedTask: () => ({ task_type: 'data_join' }),",
            "  getSelectedTaskId: () => 'task-A',",
            "  getAgentMessages: () => [],",
            "  isAgentMode: () => false,",
            "  renderWorkflowStepper: () => {},",
            "  setActionStatus: () => {},",
            "});",
            "const renderSignatures = {};",
            "controller.render({ force: true, renderSignatures });",
            "const firstHtml = elements.workflowStepper.innerHTML;",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures });",
            "const secondHtml = elements.workflowStepper.innerHTML;",
            "process.stdout.write(JSON.stringify({ firstHtml, secondHtml }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert 'data-skeleton="plan-rail"' in payload["firstHtml"]
    assert "skeleton-row" in payload["firstHtml"]
    assert "计划生成中" not in payload["firstHtml"]

    assert 'data-skeleton="plan-rail"' not in payload["secondHtml"]
    # Batch 4 (2026-10-14 redesign): the empty rail uses the canonical not-started word.
    assert "待开始" in payload["secondHtml"]





def test_plan_rail_omits_replan_events_and_subagent_rows():
    """The non-validation rail is steps only, even when the plan payload also
    carries replan history and active sub-agent metadata."""
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ createPlanRailController }} from {json.dumps(module_url)};",
            "const elements = {",
            "  progressRail: { setAttribute() {} },",
            "  workflowStepper: { innerHTML: '' },",
            "};",
            "function $(id) { return elements[id] || null; }",
            "globalThis.document = { querySelector() { return { textContent: '' }; } };",
            "const plan = {",
            "  id: 'plan-1',",
            "  status: 'running',",
            "  replan_count: 2,",
            "  loop_events: [",
            "    { type: 'replan', reason: 'failure', at: 't1', trigger_step_id: 'step-1', tool_ref: 'data_ops.propose_join' },",
            "    { type: 'no_progress', reason: 'failure', at: 't2', trigger_step_id: 'step-2' },",
            "  ],",
            "  sub_agents: [",
            "    { id: 'sub-1', scope: 'feature <scan>', status: 'running', granted_tools: [] },",
            "    { id: 'sub-2', scope: 'done scope', status: 'returned', granted_tools: [] },",
            "  ],",
            "  steps: [",
            "    { id: 'step-1', index: 0, title: 'Propose join', status: 'done', tool_ref: { plugin: 'data_ops', tool: 'propose_join' }, depends_on: [] },",
            "  ],",
            "};",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "const controller = createPlanRailController({",
            "  $,",
            "  getSelectedTask: () => ({ task_type: 'data_join' }),",
            "  getSelectedTaskId: () => 'task-A',",
            "  getAgentMessages: () => [],",
            "  isAgentMode: () => false,",
            "  renderWorkflowStepper: () => {},",
            "  setActionStatus: () => {},",
            "});",
            "controller.render({ force: true, renderSignatures: {} });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: {} });",
            "process.stdout.write(elements.workflowStepper.innerHTML);",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    html = result.stdout

    assert "Propose join" in html
    assert "已重规划" not in html
    assert "暂无进展" not in html
    assert "发消息介入" not in html
    assert "子任务运行中" not in html
    assert "feature &lt;scan&gt;" not in html
    assert "done scope" not in html


def test_plan_rail_shows_waiting_for_confirmation_not_generating():
    """UX-10: at the C1 role-assignment gate (no plan built yet by design), the
    rail must say "等待确认：<步骤名>" — the system is waiting on the USER, not
    generating a plan. The old unconditional "计划生成中…" text misattributed
    the wait for this exact scenario.
    """
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ createPlanRailController }} from {json.dumps(module_url)};",
            "const elements = {",
            "  progressRail: { setAttribute() {} },",
            "  workflowStepper: { innerHTML: '' },",
            "};",
            "function $(id) { return elements[id] || null; }",
            "globalThis.document = { querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [] }) });",
            "const gateMessage = { role: 'assistant', metadata: { join_c1: { target_col: 'bad' } } };",
            "const controller = createPlanRailController({",
            "  $,",
            "  getSelectedTask: () => ({ task_type: 'data_join' }),",
            "  getSelectedTaskId: () => 'task-A',",
            "  getAgentMessages: () => [gateMessage],",
            "  isAgentMode: () => false,",
            "  renderWorkflowStepper: () => {},",
            "  setActionStatus: () => {},",
            "});",
            "controller.render({ force: true, renderSignatures: {} });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: {} });",
            "process.stdout.write(elements.workflowStepper.innerHTML);",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    html = result.stdout

    assert "等待确认：文件角色与目标列" in html
    assert "计划生成中" not in html


def test_plan_rail_reports_not_started_without_job_or_gate():
    """UX-10: a genuine still-generating wait (no plan yet, no open gate message)
    must keep showing "计划生成中…" — only the gate-waiting case changes.
    """
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ createPlanRailController }} from {json.dumps(module_url)};",
            "const elements = {",
            "  progressRail: { setAttribute() {} },",
            "  workflowStepper: { innerHTML: '' },",
            "};",
            "function $(id) { return elements[id] || null; }",
            "globalThis.document = { querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [] }) });",
            "const controller = createPlanRailController({",
            "  $,",
            "  getSelectedTask: () => ({ task_type: 'data_join' }),",
            "  getSelectedTaskId: () => 'task-A',",
            "  getAgentMessages: () => [],",
            "  isAgentMode: () => false,",
            "  renderWorkflowStepper: () => {},",
            "  setActionStatus: () => {},",
            "});",
            "controller.render({ force: true, renderSignatures: {} });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: {} });",
            "process.stdout.write(elements.workflowStepper.innerHTML);",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    html = result.stdout

    # Batch 4 (2026-10-14 redesign): the empty rail uses the canonical not-started word.
    assert "待开始" in html
    assert "等待确认" not in html


def test_gate_confirm_button_states_consequence_by_tool():
    """UX-10: gate confirm-button copy maps to the gate step's own tool
    (execute_join/screen_features/train_model/...) instead of a bare "确认"
    for every gate — the button looks identical today whether it just accepts
    a read-only screening result or writes artifacts to disk.
    """
    driver_gate_js = _read_static("js/v2/driver_gate_confirm.js")

    assert "export function gateConfirmLabel(toolName)" in driver_gate_js
    assert 'execute_join: "确认并执行拼接"' in driver_gate_js
    assert 'screen_features: "确认所选特征"' in driver_gate_js
    assert 'train_model: "确认并开始训练"' in driver_gate_js
    assert 'configure_tuning: "确认调参配置并开始调参"' in driver_gate_js
    assert 'generate_model_reports: "确认并生成报告"' in driver_gate_js
    assert 'post_training_action: "确认模型交付"' in driver_gate_js

    module_url = (STATIC_DIR / "js" / "v2" / "driver_gate_confirm.js").as_uri()
    script = "\n".join(
        [
            f"import {{ renderDriverGateButton }} from {json.dumps(module_url)};",
            "const message = { metadata: { kind: 'gate', step_id: 'gate-1' } };",
            "const joinHtml = renderDriverGateButton(message, { gateStepTool: 'execute_join' });",
            "const reportHtml = renderDriverGateButton(message, { gateStepTool: 'generate_model_reports' });",
            "const deliveryHtml = renderDriverGateButton(message, { gateStepTool: 'post_training_action' });",
            "const genericHtml = renderDriverGateButton(message, {});",
            "process.stdout.write(JSON.stringify({ joinHtml, reportHtml, deliveryHtml, genericHtml }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert "确认并执行拼接" in payload["joinHtml"]
    assert "确认并生成报告" in payload["reportHtml"]
    assert "确认模型交付" in payload["deliveryHtml"]
    assert ">确认<" in payload["genericHtml"]


def test_structured_confirmation_buttons_submit_ui_action_and_share_action_bar():
    workbench_css = _read_browser_css()

    assert ".gate-action-bar" in workbench_css




def test_strategy_create_dialog_captures_governed_business_input():
    """Phase 0A: strategy development must not invent an operating objective."""
    index_html = _read_static("index.html")

    assert 'id="createTaskStrategyField"' in index_html
    assert 'id="strategyEntryMode"' in index_html
    assert 'id="strategyObjective"' in index_html
    assert 'id="strategyMaxBadRate"' in index_html
    assert 'id="strategyMinApprovalRate"' in index_html
    assert 'id="strategyEadCol"' in index_html
    assert 'id="strategyPdCol"' in index_html


def test_manual_analysis_mounts_structured_adoption_gate_in_middle_workspace():
    module_url = (
        STATIC_DIR / "js" / "v2" / "driver_manual_analysis.js"
    ).as_uri()
    script = "\n".join(
        [
            (
                "import { driverManualAnalysisHtml } from "
                f"{json.dumps(module_url)};"
            ),
            "const message = { id: 'adopt-gate', role: 'assistant', content: '回测完成。', metadata: {",
            "  kind: 'gate', step_id: 'adopt-step', gate_source_tool: 'adopt_strategy',",
            "  editable_input_schema: { type: 'object', properties: { adoption_reason: { type: 'string', minLength: 2 } }, required: ['adoption_reason'] },",
            "} };",
            "const html = driverManualAnalysisHtml([message], {",
            "  renderAgentMarkdown: (value) => value,",
            "  renderAdoptionGate: (_message, options) => `<div data-adoption-widget data-interactive=\"${options.interactive}\"></div>`,",
            "  renderGateConfirm: () => '<button data-generic-confirm></button>',",
            "});",
            "process.stdout.write(html);",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'data-adoption-widget' in result.stdout
    assert 'data-interactive="true"' in result.stdout
    assert "data-generic-confirm" not in result.stdout


def test_adoption_gate_requires_reason_and_submits_gate_bound_payload():
    """Phase 0A: one request binds a real reason to the current adopt gate."""
    module_url = (STATIC_DIR / "js" / "v2" / "adoption_gate_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ renderAdoptionGate, submitAdoption }} from {json.dumps(module_url)};",
            "const message = { metadata: {",
            "  kind: 'gate', plan_id: 'plan-1', step_id: 'adopt-step',",
            "  confirmation_snapshot: { expected_plan_status: 'awaiting_confirm', expected_plan_revision: 0, expected_plan_fingerprint: 'a'.repeat(64), expected_step_fingerprint: 'b'.repeat(64) },",
            "  editable_input_schema: { type: 'object', properties: { adoption_reason: { type: 'string', minLength: 2 } }, required: ['adoption_reason'] },",
            "} };",
            "const html = renderAdoptionGate(message, { interactive: true });",
            "const calls = [];",
            "const wrap = {",
            "  dataset: { adoptionPlanId: 'plan-1', adoptionStepId: 'adopt-step', expectedPlanStatus: 'awaiting_confirm', expectedPlanRevision: '0', expectedPlanFingerprint: 'a'.repeat(64), expectedStepFingerprint: 'b'.repeat(64) },",
            "  querySelector: (selector) => selector === '[data-adoption-reason]' ? { value: '委员会批准 Q3 本地采纳' } : null,",
            "};",
            "const button = { disabled: false, closest: () => wrap };",
            "const context = {",
            "  selectedTaskId: 'task-1',",
            "  setActionStatus: () => {}, setAgentMessages: () => {}, renderAgentConversation: () => {},",
            "  resetFetchThrottle: () => {}, renderWorkflowStepper: () => {},",
            "  pollAgentMessagesUntilSettled: async (_taskId, promise) => promise,",
            "  api: async (url, options) => { calls.push([url, JSON.parse(options.body)]); return { messages: [] }; },",
            "};",
            "await submitAdoption(button, context);",
            "process.stdout.write(JSON.stringify({ html, calls }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert "采纳理由" in payload["html"]
    assert "填写理由并采纳" in payload["html"]
    assert payload["calls"] == [[
        "/api/tasks/task-1/agent/messages",
            {
                "content": "确认采纳",
                    "ui_action": "confirm_adoption",
                    "adjust_params": {"adoption_reason": "委员会批准 Q3 本地采纳"},
                "expected_plan_id": "plan-1",
                "expected_step_id": "adopt-step",
                "expected_plan_status": "awaiting_confirm",
                "expected_plan_revision": 0,
                "expected_plan_fingerprint": "a" * 64,
                "expected_step_fingerprint": "b" * 64,
        },
    ]]


def test_adoption_conflict_refreshes_latest_gate_without_reviving_stale_button():
    module_url = (STATIC_DIR / "js" / "v2" / "adoption_gate_controller.js").as_uri()
    script = "\n".join(
        [
            "import assert from 'node:assert/strict';",
            f"import {{ submitAdoption }} from {json.dumps(module_url)};",
            "const wrap = {",
            "  dataset: { adoptionPlanId: 'plan-1', adoptionStepId: 'adopt-step', expectedPlanStatus: 'awaiting_confirm', expectedPlanRevision: '0', expectedPlanFingerprint: 'a'.repeat(64), expectedStepFingerprint: 'b'.repeat(64) },",
            "  querySelector: (selector) => selector === '[data-adoption-reason]' ? { value: '委员会批准 Q3 本地采纳' } : null,",
            "};",
            "const button = { disabled: false, closest: () => wrap };",
            "const refreshed = []; const statuses = [];",
            "await submitAdoption(button, {",
            "  selectedTaskId: 'task-1',",
            "  api: async () => { throw Object.assign(new Error('该操作对应的计划已变化，请刷新页面后重试。'), { status: 409 }); },",
            "  pollAgentMessagesUntilSettled: async () => undefined,",
            "  refreshAgentMessages: async (taskId) => { refreshed.push(taskId); },",
            "  setActionStatus: (message, kind) => statuses.push([message, kind]),",
            "});",
            "assert.deepEqual(refreshed, ['task-1']);",
            "assert.equal(button.disabled, true);",
            "assert.deepEqual(statuses.at(-1), ['计划已更新，已加载最新待确认步骤，请重新检查后操作。', 'info']);",
        ]
    )
    script = session_projection_fixture(script)
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_active_driver_conflict_is_busy_duplicate_not_snapshot_or_validation_failure():
    module_url = (STATIC_DIR / "js" / "v2" / "driver_gate_confirm.js").as_uri()
    script = "\n".join(
        [
            "import assert from 'node:assert/strict';",
            f"import {{ refreshAfterConfirmationConflict }} from {json.dumps(module_url)};",
            "const refreshed = []; const statuses = [];",
            "const handled = await refreshAfterConfirmationConflict(",
            "  Object.assign(new Error('该任务正在执行上一步，请等待完成'), { status: 409, detail: '该任务正在执行上一步，请等待完成' }),",
            "  { taskId: 'task-1', refreshAgentMessages: async (taskId) => refreshed.push(taskId), setActionStatus: (message, kind) => statuses.push([message, kind]) },",
            ");",
            "assert.equal(handled, true);",
            "assert.deepEqual(refreshed, ['task-1']);",
            "assert.deepEqual(statuses, [['上一步仍在执行，本次重复操作未提交。', 'busy']]);",
            "assert.ok(!statuses[0][0].includes('验证失败'));",
        ]
    )
    script = session_projection_fixture(script)
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_strategy_clarification_controller_renders_exact_contract_and_readonly_history():
    """Phase 0A: only the locked strategy clarification code gets a governed form.

    Historical clarification messages keep the same evidence-visible form but
    cannot submit stale business inputs; the newest message is interactive.
    """
    module_url = (
        STATIC_DIR / "js" / "v2" / "strategy_clarification_controller.js"
    ).as_uri()
    script = "\n".join(
        [
            f"import {{ renderStrategyClarification }} from {json.dumps(module_url)};",
            "const message = { id: 'clarify-2', role: 'assistant', metadata: {",
            "  kind: 'clarification', clarification: {",
            "    code: 'strategy_business_inputs_required',",
            "    missing_fields: ['max_bad_rate_or_min_approval_rate'],",
            "    current_input: { objective: 'max_approval', max_bad_rate: null, min_approval_rate: null, baseline_strategy_id: 'champion-v2', profit: null },",
            "  },",
            "} };",
            "const profitMessage = { id: 'clarify-profit', role: 'assistant', metadata: {",
            "  kind: 'clarification', clarification: { code: 'strategy_business_inputs_required',",
            "    missing_fields: ['profit.pd_col'], current_input: { objective: 'max_profit',",
            "      max_bad_rate: 0.07, min_approval_rate: null, baseline_strategy_id: 'profit-v1',",
            "      profit: { ead_col: 'ead_balance', pd_col: '', annual_rate: 0.18, funding_rate: 0.04, lgd: 0.6, operating_cost_per_loan: 20, term_months: 12 } } } } };",
            "const wrong = { metadata: { kind: 'clarification', clarification: { code: 'other' } } };",
            "process.stdout.write(JSON.stringify({",
            "  interactive: renderStrategyClarification(message, { interactive: true }),",
            "  historical: renderStrategyClarification(message, { interactive: false }),",
            "  profit: renderStrategyClarification(profitMessage, { interactive: true }),",
            "  wrong: renderStrategyClarification(wrong, { interactive: true }),",
            "}));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert 'data-strategy-clarification="1"' in payload["interactive"]
    assert 'data-strategy-objective' in payload["interactive"]
    assert 'data-strategy-max-bad-rate' in payload["interactive"]
    assert 'data-strategy-min-approval-rate' in payload["interactive"]
    assert 'data-strategy-profit-fields' in payload["interactive"]
    assert "缺少：至少一项策略约束" in payload["interactive"]
    assert '<option value="max_approval" selected>' in payload["interactive"]
    assert 'data-strategy-baseline-id' in payload["interactive"]
    assert 'value="champion-v2"' in payload["interactive"]
    assert 'data-strategy-clarification-readonly="false"' in payload["interactive"]
    assert 'data-strategy-clarification-submit="1"' in payload["interactive"]
    assert 'data-strategy-clarification-readonly="true"' in payload["historical"]
    assert "disabled" in payload["historical"]
    assert 'title="此为历史澄清请求，只能查看，不能再次提交"' in payload["historical"]
    assert '<option value="max_profit" selected>' in payload["profit"]
    assert 'value="0.07"' in payload["profit"]
    assert 'value="ead_balance"' in payload["profit"]
    assert 'value="0.18"' in payload["profit"]
    assert 'value="12"' in payload["profit"]
    assert payload["wrong"] == ""


def test_strategy_clarification_submits_exact_http_payload_and_refreshes_workspace():
    """The continuation form must post the exact locked backend contract."""
    module_url = (
        STATIC_DIR / "js" / "v2" / "strategy_clarification_controller.js"
    ).as_uri()
    script = "\n".join(
        [
            f"import {{ submitStrategyClarification }} from {json.dumps(module_url)};",
            "const controls = new Map([",
            "  ['[data-strategy-objective]', { value: 'max_profit' }],",
            "  ['[data-strategy-max-bad-rate]', { value: '0.05' }],",
            "  ['[data-strategy-min-approval-rate]', { value: '0.62' }],",
            "  ['[data-strategy-baseline-id]', { value: '   ' }],",
            "  ['[data-strategy-ead-col]', { value: 'ead' }],",
            "  ['[data-strategy-pd-col]', { value: 'pd' }],",
            "  ['[data-strategy-annual-rate]', { value: '0.18' }],",
            "  ['[data-strategy-funding-rate]', { value: '0.04' }],",
            "  ['[data-strategy-lgd]', { value: '0.6' }],",
            "  ['[data-strategy-operating-cost]', { value: '20' }],",
            "  ['[data-strategy-term-months]', { value: '12' }],",
            "  ['[data-strategy-clarification-error]', { textContent: '' }],",
            "]);",
            "const wrap = { querySelector: (selector) => controls.get(selector) || null };",
            "const button = { disabled: false, closest: () => wrap };",
            "const calls = []; const statuses = []; const refreshes = []; const rails = [];",
            "let rendered = 0; let messages = null;",
            "const context = {",
            "  selectedTaskId: 'task-1',",
            "  setActionStatus: (...args) => statuses.push(args),",
            "  setAgentMessages: (value) => { messages = value; },",
            "  renderAgentConversation: () => { rendered += 1; },",
            "  pollAgentMessagesUntilSettled: async (_taskId, promise) => promise,",
            "  refreshAgentMessages: async (taskId) => { refreshes.push(taskId); },",
            "  resetFetchThrottle: (taskId) => rails.push(['reset', taskId]),",
            "  renderWorkflowStepper: (options) => rails.push(['render', options]),",
            "  api: async (url, options) => {",
            "    calls.push([url, options.method, JSON.parse(options.body)]);",
            "    return { messages: [{ id: 'plan-overview' }] };",
            "  },",
            "};",
            "await submitStrategyClarification(button, context);",
            "process.stdout.write(JSON.stringify({ calls, statuses, refreshes, rails, rendered, messages }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["calls"] == [[
        "/api/tasks/task-1/agent/messages",
        "POST",
        {
            "content": "补充策略业务口径",
            "strategy_input": {
                "entry_mode": "strategy_development",
                "objective": "max_profit",
                "max_bad_rate": 0.05,
                "min_approval_rate": 0.62,
                "baseline_strategy_id": None,
                "profit": {
                    "ead_col": "ead",
                    "pd_col": "pd",
                    "annual_rate": 0.18,
                    "funding_rate": 0.04,
                    "lgd": 0.6,
                    "operating_cost_per_loan": 20,
                    "term_months": 12,
                },
            },
        },
    ]]
    assert payload["refreshes"] == ["task-1"]
    assert payload["messages"] == [{"id": "plan-overview"}]
    assert payload["rendered"] == 1
    assert payload["rails"][-2:] == [
        ["reset", "task-1"],
        ["render", {"force": True}],
    ]


def test_strategy_clarification_validation_and_request_error_keep_form_editable():
    module_url = (
        STATIC_DIR / "js" / "v2" / "strategy_clarification_controller.js"
    ).as_uri()
    script = "\n".join(
        [
            f"import {{ submitStrategyClarification }} from {json.dumps(module_url)};",
            "function makeForm(values) {",
            "  const error = { textContent: '' };",
            "  const controls = new Map(Object.entries(values).map(([key, value]) => [key, { value }]));",
            "  controls.set('[data-strategy-clarification-error]', error);",
            "  return { error, wrap: { querySelector: (selector) => controls.get(selector) || null }, controls };",
            "}",
            "const calls = []; const statuses = [];",
            "const noConstraint = makeForm({",
            "  '[data-strategy-objective]': 'max_approval',",
            "  '[data-strategy-max-bad-rate]': '', '[data-strategy-min-approval-rate]': '',",
            "  '[data-strategy-baseline-id]': '',",
            "});",
            "const invalidButton = { disabled: false, closest: () => noConstraint.wrap };",
            "const base = { selectedTaskId: 'task-1', setActionStatus: (...args) => statuses.push(args),",
            "  api: async (...args) => { calls.push(args); return { messages: [] }; } };",
            "await submitStrategyClarification(invalidButton, base);",
            "const partialProfit = makeForm({",
            "  '[data-strategy-objective]': 'max_profit',",
            "  '[data-strategy-max-bad-rate]': '0.08', '[data-strategy-min-approval-rate]': '',",
            "  '[data-strategy-baseline-id]': '', '[data-strategy-ead-col]': 'ead',",
            "  '[data-strategy-pd-col]': '', '[data-strategy-annual-rate]': '0.18',",
            "});",
            "const partialButton = { disabled: false, closest: () => partialProfit.wrap };",
            "await submitStrategyClarification(partialButton, base);",
            "const backendFailure = makeForm({",
            "  '[data-strategy-objective]': 'max_approval',",
            "  '[data-strategy-max-bad-rate]': '0.08', '[data-strategy-min-approval-rate]': '',",
            "  '[data-strategy-baseline-id]': 'base-v1',",
            "});",
            "const failureButton = { disabled: false, closest: () => backendFailure.wrap };",
            "await submitStrategyClarification(failureButton, { ...base, api: async () => { throw new Error('后端拒绝'); } });",
            "process.stdout.write(JSON.stringify({",
            "  calls: calls.length, invalidError: noConstraint.error.textContent, partialError: partialProfit.error.textContent,",
            "  failureError: backendFailure.error.textContent, failureDisabled: failureButton.disabled,",
            "  preservedObjective: backendFailure.controls.get('[data-strategy-objective]').value,",
            "  preservedConstraint: backendFailure.controls.get('[data-strategy-max-bad-rate]').value,",
            "}));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["calls"] == 0
    assert "至少填写一个" in payload["invalidError"]
    assert "EAD/PD 列和完整收益参数" in payload["partialError"]
    assert payload["failureError"] == "后端拒绝"
    assert payload["failureDisabled"] is False
    assert payload["preservedObjective"] == "max_approval"
    assert payload["preservedConstraint"] == "0.08"


def test_strategy_clarification_uses_one_renderer_in_agent_and_manual_modes():
    manual_url = (STATIC_DIR / "js" / "v2" / "driver_manual_analysis.js").as_uri()
    script = "\n".join(
        [
            f"import {{ driverManualAnalysisHtml }} from {json.dumps(manual_url)};",
            "const messages = [",
            "  { id: 'old', role: 'assistant', content: '旧请求', metadata: { kind: 'clarification', clarification: { code: 'strategy_business_inputs_required', missing_fields: ['objective'] } } },",
            "  { id: 'new', role: 'assistant', content: '新请求', metadata: { kind: 'clarification', clarification: { code: 'strategy_business_inputs_required', missing_fields: ['constraints'] } } },",
            "];",
            "const calls = [];",
            "const html = driverManualAnalysisHtml(messages, {",
            "  renderAgentMarkdown: (value) => value,",
            "  renderStrategyClarification: (message, options) => {",
            "    calls.push([message.id, options.interactive]);",
            "    return `<form data-id=\"${message.id}\" data-interactive=\"${options.interactive}\"></form>`;",
            "  },",
            "});",
            "process.stdout.write(JSON.stringify({ html, calls }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    app_js = _read_static("app.js")

    assert payload["calls"] == [["old", False], ["new", True]]
    assert 'data-id="old" data-interactive="false"' in payload["html"]
    assert 'data-id="new" data-interactive="true"' in payload["html"]
    assert 'from "./js/v2/strategy_clarification_controller.js"' in app_js
    assert "renderStrategyClarification: agentMessageStrategyClarificationHtml" in app_js
    assert (
        "agentMessageStrategyClarificationBodyHtml(message, clarificationInteractive)"
        in app_js
    )
    assert "return renderStrategyClarification(message, options);" in app_js
    assert "handleStrategyClarificationSubmit" in app_js
    clarification_css = _css_rule(
        _read_browser_css(), ".strategy-clarification-card"
    )
    assert "position: relative" in clarification_css
    assert "z-index: 25" in clarification_css


def test_all_rail_interactions_move_to_middle_workspace():
    """所有交互（确认/开始执行/下载报告）都在中间主区进行，右侧 rail 只保留
    状态与进度，不再提供会把页面拉回顶部的定位入口。

    - 开始执行 (plan validated, manual mode) and 下载报告 (a report step done)
      render as real buttons in the MIDDLE #planDriverActions panel, driven by
      the same document-level handlers (data-driver-confirm /
      data-driver-report-download).
    - The rail's start slot and report step row carry NO such button and NO
      data-plan-*-locate control.
    """
    plan_js = _read_static("js/v2/plan_rail_controller.js")
    driver_analysis_js = _read_static("js/v2/driver_manual_analysis.js")
    app_js = _read_static("app.js")
    v2_css = _read_browser_css()

    # These plan-rail helpers are nested inside createPlanRailController, so their
    # closing brace is indented — _slice_function (which stops at a column-0 `}`)
    # would over-capture. Use a brace-counting slice for nested functions.
    def _nested(source: str, signature: str) -> str:
        start = source.index(signature)
        depth = 0
        seen = False
        for i in range(start, len(source)):
            if source[i] == "{":
                depth += 1
                seen = True
            elif source[i] == "}":
                depth -= 1
                if seen and depth == 0:
                    return source[start : i + 1]
        return source[start:]

    # The rail never emits an interactive confirm/download button or text badge.
    # The middle
    # planDriverActionsHtml / planRetryCardHtml own the real buttons; the plain
    # gate confirm is rendered by driverManualAnalysisHtml via renderGateConfirm.)
    rail_html_builders = "".join(
        _nested(plan_js, sig)
        for sig in ("function planSubstepHtml", "function planPhaseHtml", "function planRailHtml", "function reconcilePlanRail")
    )
    assert "data-driver-confirm" not in rail_html_builders
    assert "data-driver-report-download" not in rail_html_builders
    assert "data-plan-gate-locate" not in rail_html_builders
    assert "data-plan-start-locate" not in rail_html_builders
    assert "data-plan-report-locate" not in rail_html_builders

    # The middle driver-actions panel owns the real buttons.
    actions_body = _nested(plan_js, "function planDriverActionsHtml")
    assert 'data-driver-confirm="1"' in actions_body
    assert "开始执行" in actions_body
    assert 'data-driver-report-download="1"' in actions_body
    assert "下载报告" in actions_body

    # driverManualAnalysisHtml renders the plain-gate confirm control in the
    # middle section (renderGateConfirm) and anchors the pending gate section so
    # the rail locate entry can scroll to it.
    assert "renderGateConfirm" in driver_analysis_js
    assert "data-driver-gate-section=" in driver_analysis_js
    assert "renderGateConfirm: agentMessageGateButtonHtml" in app_js

    # The .is-open dead class now has a meaningful rule (open-state left edge).
    is_open_rule = _css_rule(v2_css, ".plan-retry-panel.is-open,\n.plan-driver-actions.is-open")
    assert "border-left" in is_open_rule
    assert ".plan-driver-action-card" in v2_css

    # Executable proof: drive the controller against a mini-DOM and read the
    # middle panel's produced markup for both the start gate and a done report.
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = _MINI_DOM_JS + "\n" + "\n".join(
        [
            f"const {{ createPlanRailController }} = await import({json.dumps(module_url)});",
            "const __doc = makeDocument();",
            "const workflowStepper = __doc.createElement('div');",
            "const planDriverActions = __doc.createElement('section');",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper, planDriverActions };",
            "function $(id) { return elements[id] || null; }",
            "let plan;",
            "let messages = [];",
            "globalThis.document = { createElement: (t) => __doc.createElement(t), querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "let agentMode = false;",
            "let selectedTask = { task_type: 'modeling', active_job_kind: null };",
            "const controller = createPlanRailController({ $, getSelectedTask: () => selectedTask, getSelectedTaskId: () => 'task-A', getAgentMessages: () => messages, isAgentMode: () => agentMode, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "const rs = {};",
            "// Plan built but not started -> middle panel shows 开始执行.",
            "plan = { id: 'plan-1', status: 'validated', steps: [",
            "  { id: 's1', index: 0, phase: '建模', title: 'Train', status: 'pending', tool_ref: { plugin: 'modeling', tool: 'train_model' }, depends_on: [] },",
            "] };",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "// The mini-DOM serializer only re-emits class (drops data-*), so read",
            "// live nodes: query by class, then confirm the parsed dataset flag.",
            "const startBtn = planDriverActions.querySelector('.driver-confirm');",
            "const startRailBtn = workflowStepper.querySelector('.driver-confirm');",
            "const startLocate = workflowStepper.querySelector('.plan-step-locate');",
            "// A report step completed -> middle panel shows 下载报告.",
            "plan = { id: 'plan-1', status: 'running', steps: [",
            "  { id: 's1', index: 0, phase: '建模', title: 'Train', status: 'done', tool_ref: { plugin: 'modeling', tool: 'train_model' }, depends_on: [] },",
            "  { id: 's2', index: 1, phase: '报告', title: 'Report', status: 'done', tool_ref: { plugin: 'modeling', tool: 'generate_model_report' }, depends_on: ['s1'] },",
            "] };",
            "controller.resetFetchThrottle('task-A');",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const dlBtn = planDriverActions.querySelector('.plan-step-download');",
            "const railDlBtn = workflowStepper.querySelector('.plan-step-download');",
            "const railReady = workflowStepper.querySelector('.plan-step-ready');",
            "// The completion message owns the primary report card. Once it is",
            "// present, the generic plan action must disappear instead of showing",
            "// a second download entry for the same file.",
            "messages = [{ role: 'assistant', metadata: { report_download: { download_url: '/api/tasks/task-A/driver-report/download' } } }];",
            "controller.render({ force: true, renderSignatures: rs });",
            "const duplicateDlBtn = planDriverActions.querySelector('.plan-step-download');",
            "// Agent mode: a plain authoritative awaiting-confirm step exposes a",
            "// visible plan+step-bound authorization button only after the matching",
            "// gate message has arrived. Structured gates must never get this",
            "// generic button because it cannot supply their required inputs.",
            "agentMode = true;",
            "selectedTask = { task_type: 'modeling', active_job_kind: 'driver' };",
            "messages = [];",
            "plan = { id: 'plan-agent', status: 'running', steps: [",
                "  { id: 'report-gate', index: 0, phase: '报告', title: '生成模型开发报告', status: 'awaiting_confirm', needs_confirmation: true, tool_ref: { plugin: 'modeling', tool: 'generate_model_reports' }, depends_on: [], confirmation_snapshot: { expected_plan_status: 'awaiting_confirm', expected_plan_revision: 1, expected_plan_fingerprint: 'a'.repeat(64), expected_step_fingerprint: 'b'.repeat(64) } },",
            "  { id: 'delivery', index: 1, phase: '交付', title: '模型交付动作', status: 'pending', tool_ref: { plugin: 'modeling', tool: 'post_training_action' }, depends_on: ['report-gate'] },",
            "] };",
            "controller.resetFetchThrottle('task-A');",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const pendingMessageGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "messages = [{ id: 'report-message', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-agent', step_id: 'report-gate', gate_source_tool: 'generate_model_reports' } }];",
            "controller.render({ renderSignatures: rs });",
            "const agentGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "const agentGateDisabledWhileBusy = agentGateBtn ? agentGateBtn.getAttribute('disabled') != null : null;",
            "// Once task polling reports the prior driver job cleared, the same",
            "// gate must be rebuilt as enabled even though its plan step is unchanged.",
            "selectedTask = { task_type: 'modeling', active_job_kind: null };",
            "controller.render({ renderSignatures: rs });",
            "const readyAgentGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "// Special-value, adoption, and red-monitoring gates require",
            "// structured decisions. Their matching messages suppress the",
            "// misleading generic confirm.",
            "plan = { id: 'plan-special', status: 'awaiting_confirm', steps: [",
            "  { id: 'special-gate', index: 0, phase: '特征', title: '治理特殊值', status: 'awaiting_confirm', needs_confirmation: true, tool_ref: { plugin: 'modeling', tool: 'resolve_special_values' }, depends_on: [] },",
            "] };",
            "messages = [{ id: 'special-message', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-special', step_id: 'special-gate', gate_source_tool: 'resolve_special_values', special_values: { columns: [{ column: 'x1', values: [{ value: -999 }] }] } } }];",
            "controller.resetFetchThrottle('task-A');",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const structuredSpecialGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "plan = { id: 'plan-adopt', status: 'awaiting_confirm', steps: [",
            "  { id: 'adopt-gate', index: 0, phase: '采纳', title: '采纳策略', status: 'awaiting_confirm', needs_confirmation: true, tool_ref: { plugin: 'strategy', tool: 'adopt_strategy' }, depends_on: [] },",
            "] };",
            "messages = [{ id: 'adopt-message', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-adopt', step_id: 'adopt-gate', gate_source_tool: 'adopt_strategy', editable_input_schema: { properties: { adoption_reason: { type: 'string' } } } } }];",
            "controller.resetFetchThrottle('task-A');",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const structuredAdoptionGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "plan = { id: 'plan-monitoring', status: 'awaiting_confirm', steps: [",
            "  { id: 'monitoring-gate', index: 0, phase: '处置', title: '处置监控结果', status: 'awaiting_confirm', needs_confirmation: true, tool_ref: { plugin: 'strategy', tool: 'apply_monitoring_disposition' }, depends_on: [] },",
            "] };",
            "const monitoringSchema = { properties: { disposition: { type: 'string', enum: ['observe', 'adjust_threshold', 'new_version'] }, reason: { type: 'string' }, threshold_patch: { type: 'object' } } };",
            "messages = [{ id: 'monitoring-green', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-monitoring', step_id: 'monitoring-gate', gate_source_tool: 'apply_monitoring_disposition', monitoring_disposition: { overall_level: 'green', requires_structured_input: false }, editable_input_schema: monitoringSchema } }];",
            "controller.resetFetchThrottle('task-A');",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const greenMonitoringGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "messages = [{ id: 'monitoring-amber', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-monitoring', step_id: 'monitoring-gate', gate_source_tool: 'apply_monitoring_disposition', monitoring_disposition: { overall_level: 'amber', requires_structured_input: false }, editable_input_schema: monitoringSchema } }];",
            "controller.render({ renderSignatures: rs });",
            "const amberMonitoringGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "messages = [{ id: 'monitoring-red', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-monitoring', step_id: 'monitoring-gate', gate_source_tool: 'apply_monitoring_disposition', monitoring_disposition: { overall_level: 'red', requires_structured_input: true }, editable_input_schema: monitoringSchema } }];",
            "controller.render({ renderSignatures: rs });",
            "const redMonitoringGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "messages = [{ id: 'monitoring-unknown', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-monitoring', step_id: 'monitoring-gate', gate_source_tool: 'apply_monitoring_disposition', monitoring_disposition: { overall_level: 'unknown', requires_structured_input: false }, editable_input_schema: monitoringSchema } }];",
            "controller.render({ renderSignatures: rs });",
            "const unknownMonitoringGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "messages = [{ id: 'monitoring-conflict', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-monitoring', step_id: 'monitoring-gate', gate_source_tool: 'apply_monitoring_disposition', monitoring_disposition: { overall_level: 'green', requires_structured_input: true }, editable_input_schema: monitoringSchema } }];",
            "controller.render({ renderSignatures: rs });",
            "const conflictingMonitoringGateBtn = planDriverActions.querySelector('.driver-confirm');",
            "process.stdout.write(JSON.stringify({",
            "  startPanelConfirm: startBtn ? (startBtn.dataset.driverConfirm || '') : null,",
            "  startPanelExpectedPlan: startBtn ? (startBtn.dataset.expectedPlanId || '') : null,",
            "  startPanelLabel: startBtn ? startBtn.textContent : '',",
            "  railHasConfirm: startRailBtn != null,",
            "  railHasLocate: startLocate != null,",
            "  reportPanelDownload: dlBtn ? (dlBtn.dataset.driverReportDownload || '') : null,",
            "  reportPanelLabel: dlBtn ? dlBtn.textContent : '',",
            "  railHasDownloadBtn: railDlBtn != null,",
            "  railHasReadyBadge: railReady != null,",
            "  railReadyText: railReady ? railReady.textContent : '',",
            "  duplicateReportDownload: duplicateDlBtn != null,",
            "  pendingMessageHasConfirm: pendingMessageGateBtn != null,",
            "  agentGateConfirm: agentGateBtn ? (agentGateBtn.dataset.driverConfirm || '') : null,",
            "  agentGateExpectedPlan: agentGateBtn ? (agentGateBtn.dataset.expectedPlanId || '') : null,",
            "  agentGateExpectedStep: agentGateBtn ? (agentGateBtn.dataset.expectedStepId || '') : null,",
            "  agentGateLabel: agentGateBtn ? agentGateBtn.textContent : '',",
            "  agentGateDisabledWhileBusy,",
            "  agentGateEnabledAfterIdle: readyAgentGateBtn ? readyAgentGateBtn.getAttribute('disabled') == null : null,",
            "  structuredSpecialHasConfirm: structuredSpecialGateBtn != null,",
            "  structuredAdoptionHasConfirm: structuredAdoptionGateBtn != null,",
            "  greenMonitoringHasConfirm: greenMonitoringGateBtn != null,",
            "  greenMonitoringLabel: greenMonitoringGateBtn ? greenMonitoringGateBtn.textContent : '',",
            "  amberMonitoringHasConfirm: amberMonitoringGateBtn != null,",
            "  redMonitoringHasConfirm: redMonitoringGateBtn != null,",
            "  unknownMonitoringHasConfirm: unknownMonitoringGateBtn != null,",
            "  conflictingMonitoringHasConfirm: conflictingMonitoringGateBtn != null,",
            "}));",
        ]
    )
    data = _run_node_capture_json(script)

    # Middle panel: real actionable buttons wired to the document-level handlers.
    assert data["startPanelConfirm"] == "1", data
    assert data["startPanelExpectedPlan"] == "plan-1", data
    assert "开始执行" in data["startPanelLabel"]
    assert data["reportPanelDownload"] == "1", data
    assert "下载报告" in data["reportPanelLabel"]

    # Rail: status only, never an interactive or locate button.
    assert data["railHasConfirm"] is False, data
    assert data["railHasLocate"] is False, data
    assert data["railHasDownloadBtn"] is False, data
    assert data["railHasReadyBadge"] is False, data
    assert data["railReadyText"] == "", data
    assert data["duplicateReportDownload"] is False, data
    assert data["pendingMessageHasConfirm"] is False, data
    assert data["agentGateConfirm"] == "1", data
    assert data["agentGateExpectedPlan"] == "plan-agent", data
    assert data["agentGateExpectedStep"] == "report-gate", data
    assert "确认并生成报告" in data["agentGateLabel"], data
    assert data["agentGateDisabledWhileBusy"] is True, data
    assert data["agentGateEnabledAfterIdle"] is True, data
    assert data["structuredSpecialHasConfirm"] is False, data
    assert data["structuredAdoptionHasConfirm"] is False, data
    assert data["greenMonitoringHasConfirm"] is True, data
    assert "确认知悉" in data["greenMonitoringLabel"], data
    assert data["amberMonitoringHasConfirm"] is True, data
    assert data["redMonitoringHasConfirm"] is False, data
    assert data["unknownMonitoringHasConfirm"] is False, data
    assert data["conflictingMonitoringHasConfirm"] is False, data


def test_plan_start_action_refreshes_when_confirmation_snapshot_changes():
    """Polling a newer CAS snapshot must replace the actionable start button."""
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = _MINI_DOM_JS + "\n" + "\n".join(
        [
            f"const {{ createPlanRailController }} = await import({json.dumps(module_url)});",
            "const __doc = makeDocument();",
            "const workflowStepper = __doc.createElement('div');",
            "const planDriverActions = __doc.createElement('section');",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper, planDriverActions };",
            "function $(id) { return elements[id] || null; }",
            "let plan = { id: 'plan-refresh', status: 'validated', confirmation_snapshot: {",
            "  expected_plan_status: 'validated', expected_plan_revision: 1,",
            "  expected_plan_fingerprint: 'a'.repeat(64),",
            "}, steps: [] };",
            "globalThis.document = { createElement: (t) => __doc.createElement(t), querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "const controller = createPlanRailController({ $, getSelectedTask: () => ({ task_type: 'modeling', active_job_kind: null }), getSelectedTaskId: () => 'task-A', getAgentMessages: () => [], isAgentMode: () => false, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "const rs = {};",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const first = planDriverActions.querySelector('.driver-confirm');",
            "plan = { ...plan, confirmation_snapshot: {",
            "  expected_plan_status: 'validated', expected_plan_revision: 2,",
            "  expected_plan_fingerprint: 'c'.repeat(64),",
            "} };",
            "controller.resetFetchThrottle('task-A');",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const refreshed = planDriverActions.querySelector('.driver-confirm');",
            "process.stdout.write(JSON.stringify({",
            "  firstRevision: first?.dataset.expectedPlanRevision || '',",
            "  firstFingerprint: first?.dataset.expectedPlanFingerprint || '',",
            "  refreshedRevision: refreshed?.dataset.expectedPlanRevision || '',",
            "  refreshedFingerprint: refreshed?.dataset.expectedPlanFingerprint || '',",
            "}));",
        ]
    )
    data = _run_node_capture_json(script)

    assert data["firstRevision"] == "1", data
    assert data["firstFingerprint"] == "a" * 64, data
    assert data["refreshedRevision"] == "2", data
    assert data["refreshedFingerprint"] == "c" * 64, data


def test_agent_gate_action_refreshes_when_step_confirmation_snapshot_changes():
    """A same-plan step CAS change must replace the actionable gate button."""
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = _MINI_DOM_JS + "\n" + "\n".join(
        [
            f"const {{ createPlanRailController }} = await import({json.dumps(module_url)});",
            "const __doc = makeDocument();",
            "const workflowStepper = __doc.createElement('div');",
            "const planDriverActions = __doc.createElement('section');",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper, planDriverActions };",
            "function $(id) { return elements[id] || null; }",
            "const baseSnapshot = { expected_plan_status: 'awaiting_confirm', expected_plan_revision: 1, expected_plan_fingerprint: 'a'.repeat(64) };",
            "let plan = { id: 'plan-gate-refresh', status: 'awaiting_confirm', confirmation_snapshot: baseSnapshot, steps: [{",
            "  id: 'report-gate', index: 0, phase: '报告', title: '生成报告', status: 'awaiting_confirm',",
            "  tool_ref: { plugin: 'modeling', tool: 'generate_model_reports' }, depends_on: [],",
            "  confirmation_snapshot: { ...baseSnapshot, expected_step_fingerprint: 'b'.repeat(64) },",
            "}] };",
            "const messages = [{ id: 'gate-message', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-gate-refresh', step_id: 'report-gate', gate_source_tool: 'generate_model_reports' } }];",
            "globalThis.document = { createElement: (t) => __doc.createElement(t), querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "const controller = createPlanRailController({ $, getSelectedTask: () => ({ task_type: 'modeling', active_job_kind: null }), getSelectedTaskId: () => 'task-A', getAgentMessages: () => messages, isAgentMode: () => true, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "const rs = {};",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const first = planDriverActions.querySelector('.driver-confirm');",
            "plan = { ...plan, steps: [{ ...plan.steps[0], confirmation_snapshot: { ...baseSnapshot, expected_step_fingerprint: 'd'.repeat(64) } }] };",
            "controller.resetFetchThrottle('task-A');",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const refreshed = planDriverActions.querySelector('.driver-confirm');",
            "process.stdout.write(JSON.stringify({",
            "  firstStepFingerprint: first?.dataset.expectedStepFingerprint || '',",
            "  refreshedStepFingerprint: refreshed?.dataset.expectedStepFingerprint || '',",
            "}));",
        ]
    )
    data = _run_node_capture_json(script)

    assert data["firstStepFingerprint"] == "b" * 64, data
    assert data["refreshedStepFingerprint"] == "d" * 64, data


def test_plan_gate_stays_disabled_across_rerender_for_local_busy_and_pending_claim():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    gate_url = (STATIC_DIR / "js" / "v2" / "driver_gate_confirm.js").as_uri()
    script = _MINI_DOM_JS + "\n" + "\n".join(
        [
            f"const {{ createPlanRailController }} = await import({json.dumps(module_url)});",
            f"const {{ claimDriverGateSubmission, releaseDriverGateSubmission }} = await import({json.dumps(gate_url)});",
            "const __doc = makeDocument();",
            "const workflowStepper = __doc.createElement('div');",
            "const planDriverActions = __doc.createElement('section');",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper, planDriverActions };",
            "function $(id) { return elements[id] || null; }",
            "const snapshot = { expected_plan_status: 'awaiting_confirm', expected_plan_revision: 1, expected_plan_fingerprint: 'a'.repeat(64), expected_step_fingerprint: 'b'.repeat(64) };",
            "const plan = { id: 'plan-1', status: 'awaiting_confirm', steps: [{",
            "  id: 'step-1', index: 0, phase: '报告', title: '生成报告', status: 'awaiting_confirm',",
            "  tool_ref: { plugin: 'modeling', tool: 'generate_model_reports' }, depends_on: [], confirmation_snapshot: snapshot,",
            "}] };",
            "const messages = [{ id: 'gate-1', role: 'assistant', metadata: { kind: 'gate', plan_id: 'plan-1', step_id: 'step-1', gate_source_tool: 'generate_model_reports' } }];",
            "let localBusy = null;",
            "globalThis.document = { createElement: (t) => __doc.createElement(t), querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "const controller = createPlanRailController({ $, getSelectedTask: () => ({ task_type: 'modeling', active_job_kind: null }), getTaskBusyAction: () => localBusy, getSelectedTaskId: () => 'task-1', getAgentMessages: () => messages, isAgentMode: () => true, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "const rs = {};",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const ready = planDriverActions.querySelector('.driver-confirm');",
            "claimDriverGateSubmission({ taskId: 'task-1', planId: 'plan-1', stepId: 'step-1', snapshot });",
            "controller.render({ force: true, renderSignatures: rs });",
            "const claimed = planDriverActions.querySelector('.driver-confirm');",
            "releaseDriverGateSubmission({ taskId: 'task-1', planId: 'plan-1', stepId: 'step-1' });",
            "localBusy = 'driver_execute';",
            "controller.render({ force: true, renderSignatures: rs });",
            "const busy = planDriverActions.querySelector('.driver-confirm');",
            "localBusy = null;",
            "controller.render({ force: true, renderSignatures: rs });",
            "const idleAgain = planDriverActions.querySelector('.driver-confirm');",
            "process.stdout.write(JSON.stringify({",
            "  ready: ready?.getAttribute('disabled') == null,",
            "  claimed: claimed?.getAttribute('disabled') != null,",
            "  busy: busy?.getAttribute('disabled') != null,",
            "  idleAgain: idleAgain?.getAttribute('disabled') == null,",
            "}));",
        ]
    )
    data = _run_node_capture_json(script)
    assert data == {"ready": True, "claimed": True, "busy": True, "idleAgain": True}


def test_acceptance_mode_chip_explains_auto_mode_scope():
    """AUTO is limited to low-risk nodes and never represents human approval."""
    index_html = _read_static("index.html")

    assert 'id="agentAcceptanceModeSelect"' in index_html
    assert "自动模式仅处理低风险节点" in index_html
    assert "强制业务决策与副作用授权始终需要人工操作" in index_html
    assert "替你确认全部关键节点" not in index_html
    # fires the composer notice once on switching INTO auto mode.


def test_calibration_and_score_band_chart_functions_exist_in_metric_tables():
    """VD-4: metric-tables.js must export the two new chart renderers and
    interaction attacher, mirroring the existing ROC card export shape."""
    metric_tables_js = _read_static("js/metric-tables.js")
    assert "export function renderCalibrationCard(chart)" in metric_tables_js
    assert "export function renderScoreBandCard(chart)" in metric_tables_js
    assert "export function attachCalibrationInteractions(rootEl)" in metric_tables_js
    node_check = subprocess.run(
        ["node", "--check", str(STATIC_DIR / "js" / "metric-tables.js")],
        capture_output=True,
        text=True,
    check=False)
    assert node_check.returncode == 0, node_check.stderr


def test_app_js_wires_calibration_and_score_band_charts_into_driver_tables():
    """VD-4: app.js must import the new renderers and mount them above their
    driver table via driverTableChartHtml, keyed off table.chart.kind (see
    marvis/agent/renderers.py::_calibration_table / _score_band_table)."""
    app_js = _read_static("app.js")
    assert "renderCalibrationCard," in app_js
    assert "renderScoreBandCard," in app_js
    assert "attachCalibrationInteractions," in app_js
    chart_fn = _slice_function(app_js, "function driverTableChartHtml")
    assert 'chart.kind === "calibration_curve"' in chart_fn
    assert 'chart.kind === "score_band_bars"' in chart_fn
    tables_fn = _slice_function(app_js, "function agentMessageTablesHtml")
    assert "driverTableChartHtml(table?.chart)" in tables_fn
    node_check = subprocess.run(
        ["node", "--check", str(STATIC_DIR / "app.js")],
        capture_output=True,
        text=True,
    check=False)
    assert node_check.returncode == 0, node_check.stderr


def test_calibration_reliability_curve_renders_diagonal_and_tiered_points():
    """VD-4 acceptance: the calibration chart must draw a diagonal reference
    (perfect-calibration) line plus one point per reliability-curve bin, with
    points that deviate from the diagonal tagged into the warn/critical PSI
    tier so they read as visually distinct from well-calibrated points."""
    metric_tables_js = _read_static("js/metric-tables.js")
    renderer_body = _slice_function(metric_tables_js, "export function renderCalibrationCard(chart)")

    ui_utils_url = (STATIC_DIR / "js" / "ui-utils.js").as_uri()
    script = "\n".join(
        [
            f"import {{ escapeHtml }} from {json.dumps(ui_utils_url)};",
            renderer_body,
            "const chart = {",
            "  kind: 'calibration_curve',",
            "  points: [",
            "    { avg_predicted_pd: 0.05, observed_bad_rate: 0.04, sample_count: 120, bin: 1 },",
            "    { avg_predicted_pd: 0.20, observed_bad_rate: 0.50, sample_count: 80, bin: 2 },",
            "    { avg_predicted_pd: 0.60, observed_bad_rate: 0.58, sample_count: 40, bin: 3 },",
            "  ],",
            "  brier_raw: 0.1234,",
            "  ece_raw: 0.0567,",
            "};",
            "const html = renderCalibrationCard(chart);",
            "process.stdout.write(JSON.stringify({ html }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True
    )
    html = json.loads(result.stdout)["html"]

    # Diagonal perfect-calibration reference line is present.
    assert 'class="roc-curve roc-curve-baseline"' in html
    # One scatter point per reliability-curve bin.
    assert html.count('class="calibration-point"') == 3
    # The far-off-diagonal bin (0.20 predicted vs 0.50 actual, gap=0.30) is
    # tagged critical; the near-diagonal bins are not.
    assert 'data-tier="critical"' in html
    assert 'data-tier="stable"' in html
    # Brier/ECE summary surfaces in the header, not silently dropped.
    assert "0.1234" in html
    assert "0.0567" in html
    # Hover tip carries predicted/actual/sample count, matching the ROC card's
    # data-tip convention (no separate tooltip system needed on this path).
    assert "预测 20.0%" in html
    assert "实际 50.0%" in html
    assert "n=80" in html


def test_calibration_reliability_curve_renders_nothing_for_empty_points():
    """Empty/missing calibration data must not crash or draw an empty frame —
    driverTableChartHtml skips the chart entirely when points is empty."""
    metric_tables_js = _read_static("js/metric-tables.js")
    renderer_body = _slice_function(metric_tables_js, "export function renderCalibrationCard(chart)")
    script = f"{renderer_body}\nconst html = renderCalibrationCard({{ kind: 'calibration_curve', points: [] }});\nprocess.stdout.write(JSON.stringify({{ html }}));"
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True
    )
    html = json.loads(result.stdout)["html"]
    assert "暂无校准数据" in html
    assert 'class="calibration-point"' not in html

    app_js = _read_static("app.js")
    chart_fn = _slice_function(app_js, "function driverTableChartHtml")
    assert "chart.points.length === 0) return" in chart_fn


def test_score_band_chart_renders_bars_and_bad_rate_line():
    """VD-4 acceptance: the score-band chart must draw one sample-count bar
    plus one bad-rate line point per band, labelled with the bin index on the
    x-axis, matching the payload's data_field mapping (bin/score bounds/
    sample_count/bad_rate — see marvis/agent/renderers.py::_score_band_table)."""
    metric_tables_js = _read_static("js/metric-tables.js")
    renderer_body = _slice_function(metric_tables_js, "export function renderScoreBandCard(chart)")

    ui_utils_url = (STATIC_DIR / "js" / "ui-utils.js").as_uri()
    script = "\n".join(
        [
            f"import {{ escapeHtml }} from {json.dumps(ui_utils_url)};",
            renderer_body,
            "const chart = {",
            "  kind: 'score_band_bars',",
            "  split: 'oot',",
            "  bands: [",
            "    { bin: 1, score_lower: 0.0, score_upper: 0.1, sample_count: 500, bad_rate: 0.02 },",
            "    { bin: 2, score_lower: 0.1, score_upper: 0.2, sample_count: 480, bad_rate: 0.05 },",
            "    { bin: 3, score_lower: 0.2, score_upper: 0.3, sample_count: 300, bad_rate: 0.12 },",
            "  ],",
            "};",
            "const html = renderScoreBandCard(chart);",
            "process.stdout.write(JSON.stringify({ html }));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True
    )
    html = json.loads(result.stdout)["html"]

    assert html.count('class="score-band-bar"') == 3
    assert html.count('class="score-band-rate-point"') == 3
    assert 'data-split="oot"' in html
    assert "oot 分段分布" in html
    # Bin index labels on the x-axis (Chinese axis labelling requirement).
    for bin_label in [">1<", ">2<", ">3<"]:
        assert bin_label in html
    assert "样本量" in html
    assert "坏率" in html
    assert "分箱1 · 样本量 500 · 坏率 2.00%" in html
    assert "分箱3 · 样本量 300 · 坏率 12.00%" in html


def test_score_band_chart_renders_nothing_for_empty_bands():
    metric_tables_js = _read_static("js/metric-tables.js")
    renderer_body = _slice_function(metric_tables_js, "export function renderScoreBandCard(chart)")
    script = f"{renderer_body}\nconst html = renderScoreBandCard({{ kind: 'score_band_bars', split: 'oot', bands: [] }});\nprocess.stdout.write(JSON.stringify({{ html }}));"
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True
    )
    html = json.loads(result.stdout)["html"]
    assert "暂无分段数据" in html
    assert 'class="score-band-bar"' not in html


def test_driver_wide_tables_keep_a_local_horizontal_scroll_viewport():
    """A wide driver table must shrink with the middle timeline and scroll
    inside its own wrapper instead of expanding the message and being clipped
    by result-scroll-content's intentional x-axis boundary."""
    styles_css = _read_browser_css()
    v2_css = _read_browser_css()

    assistant_rule = _css_rule(styles_css, ".agent-message.assistant")
    content_rule = _css_rule(styles_css, ".agent-message.assistant .agent-message-content")
    tables_rule = _css_rule(v2_css, ".agent-message-tables")
    inline_rule = _css_rule(v2_css, ".agent-inline-table")
    scroll_rule = _css_rule(v2_css, ".agent-inline-table-scroll")

    for rule in (assistant_rule, content_rule, tables_rule, inline_rule, scroll_rule):
        assert "min-width: 0" in rule
        assert "width: 100%" in rule
    assert "overflow-x: auto" in scroll_rule
    assert ".agent-inline-table-scroll::-webkit-scrollbar" in styles_css


def test_driver_table_chart_html_mounts_above_table_and_skips_when_absent():
    """Mounting-point contract: driverTableChartHtml must be called with
    table.chart and its output placed before the table markup (chart is an
    enhancement above the table, not a replacement — the table always stays),
    and calling agentMessageTablesHtml with no chart field on a table must
    not error or inject an empty chart wrapper."""
    app_js = _read_static("app.js")
    metric_tables_js = _read_static("js/metric-tables.js")
    tables_fn = _slice_function(app_js, "function agentMessageTablesHtml")
    widget_call = tables_fn[tables_fn.index("renderWorkflowDataWidget"):]
    chart_before_table = widget_call.index("chartHtml,") < widget_call.index("tableHtml:")
    assert chart_before_table

    kind_body = _slice_function(app_js, "function driverColumnKindFromHeader")
    cell_body = _slice_function(app_js, "function driverTableCellHtml")
    align_body = _slice_function(metric_tables_js, "export function metricHeaderShouldRightAlign")
    chart_html_body = _slice_function(app_js, "function driverTableChartHtml")
    cal_body = _slice_function(metric_tables_js, "export function renderCalibrationCard(chart)")
    band_body = _slice_function(metric_tables_js, "export function renderScoreBandCard(chart)")
    tables_body = tables_fn

    ui_utils_url = (STATIC_DIR / "js" / "ui-utils.js").as_uri()
    render_metrics_url = (STATIC_DIR / "js" / "render-metrics.js").as_uri()
    workflow_widgets_url = (STATIC_DIR / "js" / "v2" / "workflow_widgets.js").as_uri()

    script = "\n".join(
        [
            f"import {{ escapeHtml }} from {json.dumps(ui_utils_url)};",
            f"import {{ renderWorkflowDataWidget }} from {json.dumps(workflow_widgets_url)};",
            f"import {{ platformLabelSource }} from {json.dumps((STATIC_DIR / 'js/platform-labels.js').as_uri())};",
            ("import {"
            " columnFractions, parseNumeric, psiTier, psiTooltipText,"
            f" }} from {json.dumps(render_metrics_url)};"),
            kind_body,
            cell_body,
            align_body,
            cal_body,
            band_body,
            chart_html_body,
            tables_body,
            "const messageNoChart = {",
            "  metadata: { tables: [ { title: '报告章节状态', columns: ['章节', '状态'], rows: [['汇总', '可生成']] } ] },",
            "};",
            "const messageWithChart = {",
            "  metadata: { tables: [ {",
            "    title: '概率校准（可靠性曲线）',",
            "    columns: ['类型', '分箱'],",
            "    rows: [['原始', '1']],",
            "    chart: { kind: 'calibration_curve', points: [",
            "      { avg_predicted_pd: 0.1, observed_bad_rate: 0.1, sample_count: 10, bin: 1 },",
            "    ] },",
            "  } ] },",
            "};",
            "process.stdout.write(JSON.stringify({",
            "  noChartHtml: agentMessageTablesHtml(messageNoChart),",
            "  withChartHtml: agentMessageTablesHtml(messageWithChart),",
            "}));",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True
    )
    parsed = json.loads(result.stdout)

    # No chart field -> no chart wrapper injected, table still renders.
    assert "agent-inline-table-chart" not in parsed["noChartHtml"]
    assert "报告章节状态" in parsed["noChartHtml"]

    # Chart present -> chart wrapper appears before the table's own <table>.
    with_chart = parsed["withChartHtml"]
    assert "agent-inline-table-chart" in with_chart
    assert 'class="calibration-point"' in with_chart
    assert with_chart.index("agent-inline-table-chart") < with_chart.index("<table>")


# ---------------------------------------------------------------------------
# Hover-flicker regression: keyed reconciliation must preserve the DOM node
# under the cursor across poll ticks. Both the task list and the plan-rail
# stepper rebuilt their whole innerHTML every second (updated_at churn),
# destroying the :hover'd node and making its grey hover background flash.
# These two tests drive the REAL renderers against a minimal DOM and assert the
# first-render node object still IS the node after a second render whose data
# differs only by a churny timestamp field.
# ---------------------------------------------------------------------------

_MINI_DOM_JS = r"""
// Minimal DOM good enough for the task-list + plan-rail keyed reconcilers.
function makeDocument() {
  class ClassList {
    constructor(el) { this._el = el; }
    _list() { return (this._el._className || "").split(/\s+/).filter(Boolean); }
    _set(list) { this._el._className = list.join(" "); }
    contains(c) { return this._list().includes(c); }
    add(c) { const l = this._list(); if (!l.includes(c)) { l.push(c); this._set(l); } }
    remove(c) { this._set(this._list().filter((x) => x !== c)); }
    toggle(c, force) {
      const has = this.contains(c);
      const want = force === undefined ? !has : force;
      if (want) this.add(c); else this.remove(c);
      return want;
    }
  }

  class El {
    constructor(tag) {
      this.tagName = String(tag || "div").toUpperCase();
      this._className = "";
      this.dataset = {};
      this.attributes = {};
      this.childNodes = [];
      this.parentNode = null;
      this.classList = new ClassList(this);
      this.onclick = null;
      this._textContent = "";
    }
    get className() { return this._className; }
    set className(v) { this._className = String(v == null ? "" : v); }
    get children() { return this.childNodes.filter((n) => n instanceof El); }
    get firstChild() { return this.childNodes[0] || null; }
    get firstElementChild() { return this.children[0] || null; }
    get nextSibling() {
      if (!this.parentNode) return null;
      const sibs = this.parentNode.childNodes;
      const i = sibs.indexOf(this);
      return i >= 0 ? sibs[i + 1] || null : null;
    }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
    appendChild(node) {
      if (node.parentNode) node.parentNode.removeChild(node);
      node.parentNode = this;
      this.childNodes.push(node);
      return node;
    }
    insertBefore(node, ref) {
      if (node.parentNode) node.parentNode.removeChild(node);
      node.parentNode = this;
      if (ref == null) { this.childNodes.push(node); return node; }
      const i = this.childNodes.indexOf(ref);
      if (i < 0) this.childNodes.push(node);
      else this.childNodes.splice(i, 0, node);
      return node;
    }
    removeChild(node) {
      const i = this.childNodes.indexOf(node);
      if (i >= 0) this.childNodes.splice(i, 1);
      node.parentNode = null;
      return node;
    }
    remove() { if (this.parentNode) this.parentNode.removeChild(this); }
    addEventListener() {}
    removeEventListener() {}
    contains() { return false; }
    getBoundingClientRect() { return { width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0 }; }
    focus() {}
    scrollIntoView() {}
    get textContent() { return this._textContent || this.children.map((c) => c.textContent).join(""); }
    set textContent(v) { this._textContent = String(v == null ? "" : v); this.childNodes = []; }
    set innerHTML(html) { this._textContent = ""; this.childNodes = parseHtml(String(html), this); }
    get innerHTML() { return serialize(this); }
    _match(sel) {
      sel = sel.trim();
      if (sel.startsWith(".")) return this.classList.contains(sel.slice(1));
      return this.tagName === sel.toUpperCase();
    }
    _descendants() {
      const out = [];
      for (const c of this.children) { out.push(c); out.push(...c._descendants()); }
      return out;
    }
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
    querySelectorAll(sel) {
      sel = sel.trim();
      if (sel.startsWith(":scope >")) {
        const rest = sel.slice(":scope >".length).trim();
        return this.children.filter((c) => c._match(rest));
      }
      return this._descendants().filter((c) => c._match(sel));
    }
  }

  function parseHtml(html, ownerParent) {
    const nodes = [];
    let i = 0;
    const stack = [];
    let current = null;
    const push = (node) => {
      node.parentNode = current || ownerParent;
      if (current) current.childNodes.push(node);
      else nodes.push(node);
    };
    const voidTags = new Set(["br", "img", "input", "hr", "path", "circle", "rect"]);
    while (i < html.length) {
      if (html[i] === "<") {
        const close = html[i + 1] === "/";
        const end = html.indexOf(">", i);
        if (end < 0) break;
        const selfClose = html[end - 1] === "/";
        const tagContent = html.slice(i + (close ? 2 : 1), end).replace(/\/$/, "").trim();
        i = end + 1;
        if (close) { current = stack.pop() || null; continue; }
        const spaceIdx = tagContent.search(/\s/);
        const tag = (spaceIdx < 0 ? tagContent : tagContent.slice(0, spaceIdx)).toLowerCase();
        const el = new El(tag);
        const attrStr = spaceIdx < 0 ? "" : tagContent.slice(spaceIdx);
        const attrRe = /([a-zA-Z0-9_-]+)(?:="([^"]*)")?/g;
        let m;
        while ((m = attrRe.exec(attrStr))) {
          const name = m[1];
          const val = m[2] == null ? "" : m[2];
          if (name === "class") el.className = val;
          else if (name.startsWith("data-")) {
            const key = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
            el.dataset[key] = val;
          } else el.setAttribute(name, val);
        }
        push(el);
        if (!voidTags.has(tag) && !selfClose) { stack.push(current); current = el; }
      } else {
        const next = html.indexOf("<", i);
        const text = html.slice(i, next < 0 ? html.length : next);
        i = next < 0 ? html.length : next;
        if (text && current) current._textContent += text;
      }
    }
    return nodes;
  }

  function serialize(el) {
    let out = el._textContent || "";
    for (const n of el.childNodes) {
      if (!(n instanceof El)) continue;
      const cls = n._className ? ` class="${n._className}"` : "";
      out += `<${n.tagName.toLowerCase()}${cls}>${serialize(n)}</${n.tagName.toLowerCase()}>`;
    }
    return out;
  }

  return { createElement: (t) => new El(t), El };
}

"""


def test_task_list_reconciliation_preserves_hovered_row_node_across_poll_ticks():
    """Two renderTaskList calls with tasks that differ ONLY by updated_at must
    keep the same .task-row-shell / .task-row node objects (so a :hover under
    the cursor never drops), not wipe the list via innerHTML="" and rebuild."""
    app_js = _read_static("app.js")
    boot_marker = 'document.addEventListener(\n  "mousedown"'
    app_js = _prepare_app_module_for_node(app_js[: app_js.index(boot_marker)])

    stubs = _MINI_DOM_JS + r"""
const __doc = makeDocument();
const __taskList = __doc.createElement("div");
const __storage = new Map();
globalThis.localStorage = { getItem: (k) => (__storage.has(k) ? __storage.get(k) : null), setItem: (k, v) => __storage.set(k, String(v)), removeItem: (k) => __storage.delete(k), clear: () => __storage.clear() };
function __mockEl() { return new Proxy({ dataset: {}, className: '', innerHTML: '', textContent: '', value: '', checked: false, scrollTop: 0, scrollHeight: 0, clientHeight: 0, children: [], childNodes: [], style: new Proxy({}, { get: () => () => {}, set: () => true }), classList: { add() {}, remove() {}, toggle() { return false; }, contains() { return false; } } }, { get(t, p) { if (p in t) return t[p]; return (...a) => { if (p === 'querySelector' || p === 'closest') return null; if (p === 'querySelectorAll') return []; if (p === 'getBoundingClientRect') return { width: 0, height: 0, top: 0, left: 0 }; return undefined; }; }, set(t, p, v) { t[p] = v; return true; } }); }
const __els = new Map();
function __get(id) { if (id === 'taskList') return __taskList; if (!__els.has(id)) __els.set(id, __mockEl()); return __els.get(id); }
globalThis.document = new Proxy({ getElementById: __get, createElement: (t) => __doc.createElement(t), querySelector(s) { if (s && s.startsWith('#')) return __get(s.slice(1)); return null; }, querySelectorAll() { return []; }, addEventListener() {}, removeEventListener() {}, body: __mockEl(), activeElement: null }, { get(t, p) { if (p in t) return t[p]; return () => undefined; } });
globalThis.window = new Proxy({ addEventListener() {}, removeEventListener() {}, matchMedia() { return { matches: false, addEventListener() {}, removeEventListener() {} }; } }, { get(t, p) { return p in t ? t[p] : () => undefined; } });
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });
globalThis.MutationObserver = class { observe() {} disconnect() {} };
globalThis.AbortController = globalThis.AbortController || class { constructor() { this.signal = {}; } abort() {} };
globalThis.fetch = async () => ({ ok: true, json: async () => ({}), text: async () => '' });
globalThis.__taskList = __taskList;
"""

    driver = r"""
taskGroupMode = "none";
taskSession.selectTask(null, "t1");
function mk(u) { return [
  { id: "t1", model_name: "模型A", task_type: "modeling", status: "running", validator: "张三", updated_at: u },
  { id: "t2", model_name: "模型B", task_type: "validation", status: "scanned", validator: "李四", updated_at: u },
]; }
taskCache = mk("2026-07-05T10:00:00Z");
renderTaskList(applyTaskFilters(taskCache), { force: true });
const shellsA = globalThis.__taskList.querySelectorAll(".task-row-shell");
const firstShellA = shellsA[0];
const firstRowA = firstShellA.querySelector(".task-row");
// Second render: identical shape, only the churny updated_at differs (same
// minute -> even the formatted date text is unchanged).
taskCache = mk("2026-07-05T10:00:30Z");
renderTaskList(applyTaskFilters(taskCache), { force: true });
const shellsB = globalThis.__taskList.querySelectorAll(".task-row-shell");
const firstShellB = shellsB[0];
const firstRowB = firstShellB.querySelector(".task-row");
process.stdout.write(JSON.stringify({
  countA: shellsA.length,
  countB: shellsB.length,
  sameShellNode: firstShellA === firstShellB,
  sameRowNode: firstRowA === firstRowB,
  shellStillAttached: firstShellB.parentNode === globalThis.__taskList,
}));
"""

    data = _run_node_capture_json(stubs + "\n" + app_js + "\n" + driver)
    assert data["countA"] == 2, data
    assert data["countB"] == 2, data
    # The core flicker guarantee: the hovered card node objects survive the tick.
    assert data["sameShellNode"] is True, data
    assert data["sameRowNode"] is True, data
    assert data["shellStillAttached"] is True, data


def test_plan_rail_reconciliation_preserves_parent_and_subtask_nodes_across_poll_ticks():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = _MINI_DOM_JS + "\n" + "\n".join(
        [
            f"const {{ createPlanRailController }} = await import({json.dumps(module_url)});",
            "const __doc = makeDocument();",
            "const workflowStepper = __doc.createElement('div');",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper };",
            "function $(id) { return elements[id] || null; }",
            "let plan;",
            "globalThis.document = { createElement: (t) => __doc.createElement(t), querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "function mk(st) { return { id: 'plan-1', status: 'running', steps: [",
            "  { id: 's1', index: 0, phase: '拼接', title: 'Propose join', status: 'done', tool_ref: { plugin: 'data_ops', tool: 'propose_join' }, depends_on: [] },",
            "  { id: 's2', index: 1, phase: '拼接', title: 'Execute join', status: st, tool_ref: { plugin: 'data_ops', tool: 'execute_join' }, depends_on: ['s1'] },",
            "  { id: 's3', index: 2, phase: '建模', title: 'Train', status: 'pending', tool_ref: { plugin: 'modeling', tool: 'train_model' }, depends_on: ['s2'] },",
            "] }; }",
            "plan = mk('running');",
            "const controller = createPlanRailController({ $, getSelectedTask: () => ({ task_type: 'data_join' }), getSelectedTaskId: () => 'task-A', getAgentMessages: () => [], isAgentMode: () => false, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "const rs = {};",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const phasesA = workflowStepper.querySelectorAll(':scope > .plan-rail-step');",
            "const sectionA = phasesA[0].querySelector(':scope > .plan-rail-substeps');",
            "const s2NodeA = sectionA.querySelectorAll(':scope > .notebook-step')[1];",
            "const s2ClassA = s2NodeA.className;",
            "// Advance the plan: s2 running -> done. Clear the fetch throttle and",
            "// await the fetch so the new plan actually lands in the controller cache.",
            "plan = mk('done');",
            "controller.resetFetchThrottle('task-A');",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const phasesB = workflowStepper.querySelectorAll(':scope > .plan-rail-step');",
            "const sectionB = phasesB[0].querySelector(':scope > .plan-rail-substeps');",
            "const s2NodeB = sectionB.querySelectorAll(':scope > .notebook-step')[1];",
            "process.stdout.write(JSON.stringify({",
            "  phaseCount: phasesB.length,",
            "  samePhaseNode: phasesA[0] === phasesB[0],",
            "  sameStepNode: s2NodeA === s2NodeB,",
            "  stepStillAttached: s2NodeB.parentNode === sectionB,",
            "  classBefore: s2ClassA,",
            "  classAfter: s2NodeB.className,",
            "  contentAdvancedInPlace: s2ClassA !== s2NodeB.className,",
            "}));",
        ]
    )

    data = _run_node_capture_json(script)
    assert data["phaseCount"] == 2, data
    assert data["samePhaseNode"] is True, data
    assert data["sameStepNode"] is True, data
    assert data["stepStillAttached"] is True, data
    assert data["classBefore"] == "notebook-step running", data
    assert data["classAfter"] == "notebook-step succeeded", data
    assert data["contentAdvancedInPlace"] is True, data


def test_agent_answer_busy_does_not_promote_a_plan_step_to_running():
    """A normal Agent question/diagnosis is generation work, not plan execution.

    The generic local ``agent`` busy flag and even the server's generic
    ``active_job_kind=driver`` lock cover both kinds of turn, so neither may
    fabricate a running step while the authoritative plan is still pending.
    """
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = _MINI_DOM_JS + "\n" + "\n".join(
        [
            f"const {{ createPlanRailController }} = await import({json.dumps(module_url)});",
            "const __doc = makeDocument();",
            "const workflowStepper = __doc.createElement('div');",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper };",
            "function $(id) { return elements[id] || null; }",
            "const plan = { id: 'plan-1', status: 'validated', steps: [",
            "  { id: 's1', index: 0, phase: '数据准备', title: '拼接诊断', status: 'pending', tool_ref: { plugin: 'data_ops', tool: 'propose_join' }, depends_on: [] },",
            "  { id: 's2', index: 1, phase: '数据准备', title: '确认拼接', status: 'pending', tool_ref: { plugin: 'data_ops', tool: 'confirm_join' }, depends_on: ['s1'] },",
            "  { id: 's3', index: 2, phase: '特征分析', title: '特征指标', status: 'pending', tool_ref: { plugin: 'modeling', tool: 'screen_features' }, depends_on: ['s2'] },",
            "] };",
            "globalThis.document = { createElement: (t) => __doc.createElement(t), querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "const controller = createPlanRailController({ $, stepCheckerHtml: (s) => `<i class=\"check-icon ${s}\"></i>`, getSelectedTask: () => ({ task_type: 'feature_analysis', active_job_kind: 'driver' }), getTaskBusyAction: () => 'agent', getSelectedTaskId: () => 'task-A', getAgentMessages: () => [], isAgentMode: () => true, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "const rs = {};",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const steps = workflowStepper.querySelectorAll(':scope > .plan-rail-step');",
            "process.stdout.write(JSON.stringify({",
            "  stepClasses: Array.from(steps).map((node) => node.className),",
            "  runningIcons: Array.from(workflowStepper.querySelectorAll('.check-icon')).filter((node) => node.className.includes('running')).length,",
            "}));",
        ]
    )
    data = _run_node_capture_json(script)
    assert data["stepClasses"] == [
        "step plan-rail-step pending",
        "step plan-rail-step pending",
    ], data
    assert data["runningIcons"] == 0, data


def test_explicit_driver_execute_promotes_only_the_current_plan_step():
    """A structured execute/confirm/retry action owns a distinct local hint, so
    it can provide immediate progress feedback before the next plan poll lands."""
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = _MINI_DOM_JS + "\n" + "\n".join(
        [
            f"const {{ createPlanRailController }} = await import({json.dumps(module_url)});",
            "const __doc = makeDocument();",
            "const workflowStepper = __doc.createElement('div');",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper };",
            "function $(id) { return elements[id] || null; }",
            "const plan = { id: 'plan-1', status: 'awaiting_confirm', steps: [",
            "  { id: 's1', index: 0, phase: '数据准备', title: '拼接诊断', status: 'done', tool_ref: { plugin: 'data_ops', tool: 'propose_join' }, depends_on: [] },",
            "  { id: 's2', index: 1, phase: '数据准备', title: '确认拼接', status: 'awaiting_confirm', tool_ref: { plugin: 'data_ops', tool: 'confirm_join' }, depends_on: ['s1'] },",
            "  { id: 's3', index: 2, phase: '特征分析', title: '特征指标', status: 'pending', tool_ref: { plugin: 'modeling', tool: 'screen_features' }, depends_on: ['s2'] },",
            "] };",
            "globalThis.document = { createElement: (t) => __doc.createElement(t), querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "const controller = createPlanRailController({ $, stepCheckerHtml: (s) => `<i class=\"check-icon ${s}\"></i>`, getSelectedTask: () => ({ task_type: 'feature_analysis', active_job_kind: null }), getTaskBusyAction: () => 'driver_execute', getSelectedTaskId: () => 'task-A', getAgentMessages: () => [], isAgentMode: () => false, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "const rs = {};",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const steps = workflowStepper.querySelectorAll(':scope > .plan-rail-step');",
            "process.stdout.write(JSON.stringify({",
            "  stepClasses: Array.from(steps).map((node) => node.className),",
            "  runningIcons: Array.from(workflowStepper.querySelectorAll('.check-icon')).filter((node) => node.className.includes('running')).length,",
            "}));",
        ]
    )
    data = _run_node_capture_json(script)
    assert data["stepClasses"] == [
        "step plan-rail-step running",
        "step plan-rail-step pending",
    ], data
    # One running parent checker plus one running current-subtask checker.
    assert data["runningIcons"] == 2, data


def test_task_hero_click_collapses_to_title_and_status_only():
    """The top card starts folded and remains manually expandable/collapsible."""
    index_html = _read_static("index.html")
    styles_css = _read_browser_css()

    # DOM: failure text, subtitle and meta stay inside the collapsible wrapper.
    # The multi-model switcher sits between the title row and details so it
    # remains clickable whether the card is expanded or collapsed.
    hero_start = index_html.index('id="taskHero"')
    hero_markup = index_html[hero_start:index_html.index("</header>", hero_start)]
    assert 'class="task-hero-top-right"' in hero_markup
    assert 'class="task-hero is-collapsed"' in index_html[
        max(0, hero_start - 80):hero_start + 40
    ]
    assert 'id="taskHeroToggle"' in hero_markup
    assert 'aria-controls="taskHeroDetails"' in hero_markup
    assert 'aria-expanded="false"' in hero_markup
    assert 'aria-label="展开任务详情"' in hero_markup
    assert 'class="task-hero-details"' in hero_markup
    assert 'id="taskHeroDetails"' in hero_markup
    details_tag = hero_markup[
        hero_markup.index('id="taskHeroDetails"'):
        hero_markup.index(">", hero_markup.index('id="taskHeroDetails"'))
    ]
    assert 'aria-hidden="true"' in details_tag
    assert "inert" in details_tag
    assert 'class="task-hero-details-inner"' in hero_markup
    # The always-visible title row and model switcher precede the collapsible details.
    assert hero_markup.index('class="task-hero-top"') < hero_markup.index('id="validationBatchSwitcher"')
    assert hero_markup.index('id="validationBatchSwitcher"') < hero_markup.index('id="taskHeroDetails"')
    # The status pill stays visible; failure detail, subtitle and meta fold away.
    details_at = hero_markup.index('id="taskHeroDetails"')
    for hidden_id in ("actionErrorDetail", "currentTaskSubtitle", "taskSnapshot"):
        assert f'id="{hidden_id}"' in hero_markup[details_at:]
    assert 'id="actionStatus"' in hero_markup[:details_at]
    assert 'id="validationBatchSwitcher"' not in hero_markup[details_at:]

    # CSS: grid-rows 1fr<->0fr animates the height; the inner wrapper clips.
    details_rule = _css_rule(styles_css, ".task-hero-details")
    assert "grid-template-rows: 1fr" in details_rule
    assert "transition: grid-template-rows" in details_rule
    inner_rule = _css_rule(styles_css, ".task-hero-details-inner")
    assert "max-height:" in inner_rule
    assert "overflow-y: auto" in inner_rule
    assert "min-height: 0" in inner_rule
    collapsed_rule = _css_rule(styles_css, ".task-hero.is-collapsed .task-hero-details")
    assert "grid-template-rows: 0fr" in collapsed_rule
    # The whole card advertises itself as clickable, and the chevron flips.
    assert ".task-hero { cursor: pointer; }" in styles_css
    assert ".task-hero.is-collapsed .task-hero-toggle-icon" in styles_css
    # Reduced-motion users get an instant fold, not an animation.
    rm_group = styles_css.index(".task-hero-details,")
    rm_block = styles_css[rm_group:rm_group + 140]
    assert ".task-hero-toggle-icon" in rm_block
    assert "transition: none" in rm_block

    # JS: one hero-level click handler toggles .is-collapsed, guards interactive
    # descendants (the path copy button) and text selection, and syncs layout.


def test_plan_failure_status_keeps_raw_trace_in_timeline_not_task_hero():
    styles_css = _read_browser_css()

    # Batch 4 (2026-10-14 redesign): plan status copy moved to js/task-status.js (re-exported by the controller).
    detail_rule = _css_rule(styles_css, ".action-error-detail")
    assert "-webkit-line-clamp: 2" in detail_rule


def test_agent_composer_bar_is_translucent_glass_like_the_task_hero():
    """The bottom input bar uses the same frosted-glass material as the top status
    card: a translucent surface gradient plus a backdrop blur, not an opaque fill."""
    styles_css = _read_browser_css()

    bar_rule = _css_rule(styles_css, ".agent-composer-bar")
    assert "backdrop-filter: blur(var(--glass-blur)) saturate(180%)" in bar_rule
    assert "-webkit-backdrop-filter: blur(var(--glass-blur)) saturate(180%)" in bar_rule
    assert "background: var(--glass-bg)" in bar_rule
    assert "border: 1px solid var(--glass-rim)" in bar_rule
    # No longer a flat opaque surface fill.
    assert "background: var(--surface);" not in bar_rule

    # Dark theme uses the same glass tokens (values come from the dark token block).
    dark_bar_rule = _css_rule(styles_css, 'body[data-theme="dark"] .agent-composer-bar')
    assert "background: var(--glass-bg)" in dark_bar_rule
    assert "box-shadow: var(--glass-shadow)" in dark_bar_rule


def test_settings_panels_use_macos_grouped_cards_with_monochrome_icons():
    """All row-based settings panels share ONE rounded card per group, with
    hairline dividers between rows and a faint monochrome leading icon —
    the macOS System Settings pattern, replacing the old loose flat rows."""
    styles_css = _read_browser_css()

    # Unified card (no per-panel card-stripping overrides survive).
    group_rule = _css_rule(styles_css, ".governance-panel .settings-group")
    assert "border: 1px solid var(--border)" in group_rule
    assert "border-radius: var(--radius-lg)" in group_rule
    assert "background: var(--card)" in group_rule
    assert "overflow: hidden" in group_rule
    assert "box-shadow" in group_rule
    for panel in ("execution-environment", "llm", "memory-policy"):
        assert (
            f'.governance-panel[data-governance-panel-content="{panel}"] > .settings-group {{'
            not in styles_css
        ), f"{panel} must not strip the shared card"

    # Hairline divider between stacked rows.
    divider_rule = _css_rule(
        styles_css, ".governance-panel .settings-group > .settings-row + .settings-row"
    )
    assert "border-top:" in divider_rule

    # Monochrome leading-icon tile: faint neutral fill, no colour.
    ico_rule = _css_rule(styles_css, ".settings-row-ico")
    assert "border-radius: var(--radius-control)" in ico_rule
    assert "color: var(--text-secondary)" in ico_rule
    assert "color-mix(in srgb, var(--text)" in ico_rule
    # Rows that opt into an icon get a 3-column grid (icon | text | control).
    assert ".governance-panel .settings-row:has(> .settings-row-ico)" in styles_css


def test_tool_detail_rows_render_as_grouped_list_with_right_aligned_values():
    """The plugin tool detail is a divided row-list (icon | label | value) inside
    the tool card, not a loose grid of boxed chips."""
    v2_css = _read_browser_css()

    # JS renders a leading monochrome icon per implementation field.

    # CSS: a divided row-list with hairlines and right-aligned values, no longer
    # a per-chip boxed grid.
    impl_rule = _css_rule(v2_css, ".plugin-tool-impl")
    assert "flex-direction: column" in impl_rule
    assert "repeat(auto-fit" not in impl_rule
    row_rule = _css_rule(v2_css, ".plugin-tool-impl div")
    assert "grid-template-columns: auto minmax(0, 1fr) auto" in row_rule
    dd_rule = _css_rule(v2_css, ".plugin-tool-impl dd")
    assert "text-align: right" in dd_rule
    assert "justify-self: end" in dd_rule


def test_v2_validation_ui_uses_pmml_scoring_copy_and_evidence():
    index_html = _read_static("index.html")

    assert 'id="scoringSectionTitle">PMML打分测试</h3>' in index_html
    assert "PMML打分、效果稳定性验证，模型压力测试" in index_html


def test_v2_pending_input_contract_has_an_interactive_confirmation_panel():
    index_html = _read_static("index.html")

    assert 'id="validationContractPanel"' in index_html
