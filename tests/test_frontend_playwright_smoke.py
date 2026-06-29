"""Optional browser smoke tests for the V2 frontend panels.

These tests are skipped by default because CI currently does not install
Playwright browsers. Run locally with:

    MARVIS_RUN_PLAYWRIGHT_SMOKE=1 python -m pytest tests/test_frontend_playwright_smoke.py -q
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
    reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 to run browser smoke tests.",
)


ROOT = Path(__file__).resolve().parents[1]


class _SmokeHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        smoke_html: str,
        tasks: list[dict] | None = None,
        messages_by_task: dict[str, list[dict]] | None = None,
        plans_by_task: dict[str, list[dict]] | None = None,
        **kwargs,
    ):
        self._smoke_html = smoke_html
        self._tasks = tasks or []
        self._messages_by_task = messages_by_task or {}
        self._plans_by_task = plans_by_task or {}
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._send_file(ROOT / "marvis/static/index.html", content_type="text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            self._send_file(ROOT / "marvis/static" / path.removeprefix("/static/"))
            return
        if path == "/api/branding":
            self._send_json({})
            return
        if path == "/api/tasks":
            self._send_json(self._tasks)
            return
        task_id = _task_api_id(path, "/api/tasks/", "/agent/messages")
        if task_id is not None:
            self._send_json({"messages": self._messages_by_task.get(task_id, []), "incremental": False})
            return
        task_id = _task_api_id(path, "/api/tasks/", "/plans")
        if task_id is not None:
            self._send_json({"plans": self._plans_by_task.get(task_id, [])})
            return
        task_id = _task_api_id(path, "/api/tasks/", "/evidence")
        if task_id is not None:
            self._send_json({})
            return
        task_id = _task_api_id(path, "/api/tasks/", "/report-fields")
        if task_id is not None:
            self._send_json({"metric_values": {}, "workbook_source": "", "metric_table_sections": []})
            return
        if path == "/api/settings/execution-environment/options":
            self._send_json({"settings": {}, "options": [], "validation": None})
            return
        if path == "/api/settings/llm":
            self._send_json({"default_model_id": "", "models": [], "enabled_models": []})
            return
        if path == "/api/settings/memory-policy":
            self._send_json({})
            return
        if self.path == "/smoke.html":
            body = self._smoke_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, *, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        guessed_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def _serve_smoke_page(
    smoke_html: str,
    *,
    tasks: list[dict] | None = None,
    messages_by_task: dict[str, list[dict]] | None = None,
    plans_by_task: dict[str, list[dict]] | None = None,
):
    handler = partial(
        _SmokeHandler,
        smoke_html=smoke_html,
        tasks=tasks,
        messages_by_task=messages_by_task,
        plans_by_task=plans_by_task,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/smoke.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _task_api_id(path: str, prefix: str, suffix: str) -> str | None:
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    task_id = path[len(prefix):-len(suffix)]
    return task_id or None


def _smoke_html() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="/marvis/static/css/styles.css" />
  <link rel="stylesheet" href="/marvis/static/css/v2-workbench.css" />
  <style>
    body { margin: 0; padding: 18px; background: var(--app-bg, #f6f7f9); }
    #root { max-width: 1080px; margin: 0 auto; display: grid; gap: 14px; }
  </style>
</head>
<body>
  <main id="root"></main>
  <script type="module">
    import { renderModelingSetupPanel } from "/marvis/static/js/v2/modeling_setup_panel.js";
    import { renderModelDeliveryPanel } from "/marvis/static/js/v2/model_delivery_panel.js";

    const longPath = "/tmp/" + "very-long-artifact-name-".repeat(12) + "model.pmml";
    const setupHtml = renderModelingSetupPanel({
      id: "setup-smoke",
      metadata: {
        step_id: "gate-setup",
        modeling_setup: {
          target_type: "binary",
          recipe: "lgb",
          recipes: ["lgb", "xgb"],
          feature_count: 128,
          n_trials: 32,
          metric_policy: "oot_ks",
          eligible_algorithms: ["lgb", "xgb", "lr"],
          disabled_algorithms: [
            { recipe: "lgb_regressor", reason: "recipe target family does not match `binary`" },
            { recipe: "lgb_multiclass", reason: "recipe target family does not match `binary`" },
          ],
          pmml_supported_algorithms: ["lgb", "xgb", "lr"],
          override_guidance: [
            { id: "target_type", label: "目标类型", level: "info", message: "二分类适合 0/1 风控标签。" },
            { id: "n_trials", label: "调参预算", level: "warning", message: "当前调参轮数适合作为常规搜索。" },
            { id: "sample_weight", label: "样本权重", level: "review", message: "权重列会改变拟合目标。" },
          ],
          split_summary: {
            split_col: "split",
            split_counts: { train: 800, test: 120, oot: 80 },
            total_rows: 1000,
            warnings: [],
          },
          sample_weight_col: "",
          sample_weight_candidates: ["weight"],
          sample_weight_diagnostics: [
            { column: "weight", valid: true, missing_rate: 0, min: 0.5, max: 2.0, mean: 1.0 },
          ],
        },
      },
    });
    const deliveryHtml = renderModelDeliveryPanel({
      metadata: {
        model_delivery: {
          source_tool: "select_experiment",
          selected_experiment_id: "exp-lgb",
          artifact_id: "art-lgb",
          recipe: "lgb",
          target_type: "binary",
          selection_metric: "oot_ks",
          business_signals: { stability: "关注", feature_count: 128, calibration: "需说明", delivery: "可移交" },
          policy_signals: { scorecard: "非评分卡", monotonicity: "未声明", approval: "仅实验候选" },
          readiness: [
            { id: "native_model", label: "原生模型", status: "ready", artifact: "/tmp/model.pkl" },
            { id: "pmml", label: "PMML", status: "succeeded", artifact: longPath },
            { id: "approval_policy", label: "审批策略", status: "warning", reason: "仅实验候选" },
          ],
          metrics: { oot_ks: 0.31, test_ks: 0.29, psi_oot_vs_train: 0.12 },
          candidates: [
            {
              id: "exp-lgb",
              recipe: "lgb",
              selected: true,
              metrics: { oot_ks: 0.31, test_ks: 0.29, psi_oot_vs_train: 0.12 },
              business_signals: { stability: "关注", feature_count: 128, calibration: "需说明", delivery: "可移交" },
              policy_signals: { scorecard: "非评分卡", monotonicity: "未声明", approval: "仅实验候选" },
              capabilities: { pmml_supported: true, handoff_supported: true, native_model_supported: true },
            },
          ],
          pmml_path: longPath,
        },
      },
    });
    document.querySelector("#root").innerHTML = setupHtml + deliveryHtml;
  </script>
</body>
</html>
"""


def _workspace_smoke_html() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="/static/styles.css" />
  <link rel="stylesheet" href="/static/css/v2-workbench.css" />
  <style>
    body { margin: 0; padding: 16px; background: var(--app-bg, #f6f7f9); }
    #root { max-width: 1120px; margin: 0 auto; display: grid; gap: 14px; }
    .smoke-section { min-width: 0; background: var(--surface, #fff); }
  </style>
</head>
<body data-theme="dark">
  <main id="root">
    <section id="planMount" class="smoke-section"></section>
    <section class="smoke-section">
      <div class="screen-table-wrap" data-screen-form="screen-smoke" data-screen-step-id="screen-step">
        <div class="screen-threshold-controls">
          <label>泄漏KS <input class="screen-threshold-input" data-screen-threshold="leakage_ks" value="0.40" /></label>
          <label>最大缺失率 <input class="screen-threshold-input" data-screen-threshold="max_missing_rate" value="0.95" /></label>
          <button type="button" class="button compact secondary screen-adjust">重算</button>
        </div>
        <div class="screen-table-scroll">
          <table class="screen-table">
            <thead><tr><th>选</th><th>特征</th><th>KS</th><th>IV</th><th>缺失率</th><th>类别</th></tr></thead>
            <tbody>
              <tr class="screen-row screen-keep">
                <td class="screen-pick-cell"><input type="checkbox" class="screen-pick" checked /></td>
                <td class="screen-feat">good_feature_long_name_for_layout</td>
                <td class="screen-num">0.3120</td><td class="screen-num">0.1820</td><td class="screen-num">2.0%</td>
                <td><span class="screen-badge keep">入选</span></td>
              </tr>
              <tr class="screen-row screen-leakage">
                <td class="screen-pick-cell"><input type="checkbox" class="screen-pick" /></td>
                <td class="screen-feat">post_loan_result_leakage_signal</td>
                <td class="screen-num">0.8120</td><td class="screen-num">0.6400</td><td class="screen-num">0.0%</td>
                <td><span class="screen-badge leak">泄漏</span></td>
              </tr>
              <tr class="screen-row screen-suspected">
                <td class="screen-pick-cell"><input type="checkbox" class="screen-pick" /></td>
                <td class="screen-feat">suspected_policy_flag</td>
                <td class="screen-num">0.4210</td><td class="screen-num">0.2100</td><td class="screen-num">5.5%</td>
                <td><span class="screen-badge susp">疑似</span></td>
              </tr>
              <tr class="screen-row screen-unusable">
                <td class="screen-pick-cell"><input type="checkbox" class="screen-pick" disabled /></td>
                <td class="screen-feat">constant_column</td>
                <td class="screen-num">n/a</td><td class="screen-num">n/a</td><td class="screen-num">100.0%</td>
                <td><span class="screen-badge unusable">不可用</span></td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="screen-table-foot">
          <span class="screen-note">共筛 128 列;泄漏阈值 KS≥0.4。勾选=入选,可硬选泄漏/疑似列。</span>
          <button type="button" class="button compact primary screen-confirm">确认所选特征</button>
        </div>
      </div>
    </section>
  </main>
  <script type="module">
    import { renderPlanView } from "/static/js/v2/plan_view.js";
    import { setPlan } from "/static/js/v2/state_v2.js";
    renderPlanView(document.querySelector("#planMount"));
    setPlan({
      id: "plan-smoke",
      goal: "建模任务浏览器 smoke",
      status: "awaiting_confirm",
      tier: "guarded",
      novel_mode: "plan_ahead",
      steps: [
        { id: "spec", index: 0, title: "确认建模规格", status: "done", tool_ref: { plugin: "modeling", tool: "choose_modeling_spec" } },
        { id: "screen", index: 1, title: "筛选特征", status: "awaiting_confirm", tool_ref: { plugin: "modeling", tool: "screen_features" }, depends_on: ["spec"], decision_point: true },
        { id: "train", index: 2, title: "训练候选模型", status: "pending", tool_ref: { plugin: "modeling", tool: "train_models" }, depends_on: ["screen"] },
        { id: "report", index: 3, title: "生成模型报告", status: "failed", tool_ref: { plugin: "modeling", tool: "generate_model_report" }, output_ref: "artifact://model/report", failure_envelope: { editable_input_schema: { properties: { reason: { default: "补充业务字段后重试" } } }, downstream_reset_steps: ["report"] } },
      ],
    });
  </script>
</body>
</html>
"""


def _real_modeling_task() -> dict:
    return {
        "id": "task-modeling-smoke",
        "task_type": "modeling",
        "model_name": "A卡模型开发",
        "model_version": "v1",
        "validator": "MARVIS",
        "source_dir": "/tmp/modeling-smoke",
        "algorithm": "lgb",
        "recipes": ["lgb", "xgb"],
        "run_mode": "manual",
        "target_type": "binary",
        "target_col": "bad_flag",
        "score_col": "score",
        "split_col": "split",
        "sample_path": "sample.csv",
        "pmml_path": "model.pmml",
        "status": "review_required",
        "status_message": "",
        "created_at": "2026-06-30T00:00:00Z",
        "updated_at": "2026-06-30T00:01:00Z",
        "status_reason_code": "",
        "sample_weight_col": "",
        "metrics": [],
        "capability_tier": "guarded",
        "report_available": True,
    }


def _real_modeling_messages() -> list[dict]:
    return [
        {
            "id": "msg-delivery",
            "role": "assistant",
            "stage": "done",
            "content": "训练后交付动作完成，已生成最终模型卡、审批包和 Champion 对比。",
            "created_at": "2026-06-30T00:01:00Z",
            "metadata": {
                "kind": "result",
                "step_id": "post_training_action",
                "phase": "G5 交付",
                "step_title": "训练后交付动作",
                "model_delivery": {
                    "source_tool": "post_training_action",
                    "selected_experiment_id": "exp-lgb",
                    "artifact_id": "art-lgb",
                    "recipe": "lgb",
                    "target_type": "binary",
                    "selection_metric": "oot_ks",
                    "pmml_path": "/tmp/modeling-smoke/art-lgb.pmml",
                    "model_report_path": "/tmp/modeling-smoke/model_report.xlsx",
                    "validation_task_id": "task-validation",
                    "approval_package_path": "/tmp/modeling-smoke/art-lgb.approval_package.md",
                    "monitoring_policy_path": "/tmp/modeling-smoke/art-lgb.monitoring_policy.md",
                    "champion_comparison_path": "/tmp/modeling-smoke/art-lgb.champion_comparison.md",
                    "model_card_path": "/tmp/modeling-smoke/art-lgb.model_card.json",
                    "model_card_markdown_path": "/tmp/modeling-smoke/art-lgb.model_card.md",
                    "model_card": {
                        "version": "model_card_v1",
                        "artifact_id": "art-lgb",
                        "recipe": "lgb",
                        "target_type": "binary",
                        "selection_metric": "oot_ks",
                        "key_metrics": {"oot_ks": 0.31, "test_ks": 0.29, "psi_oot_vs_train": 0.12},
                        "limitations": ["需业务复核校准与稳定性。"],
                        "next_review_actions": ["审批前复核 Champion 差异。"],
                    },
                    "metrics": {"oot_ks": 0.31, "test_ks": 0.29, "psi_oot_vs_train": 0.12},
                    "business_signals": {
                        "stability": "关注",
                        "feature_count": 128,
                        "calibration": "需说明",
                        "delivery": "可移交",
                    },
                    "policy_signals": {
                        "scorecard": "非评分卡",
                        "monotonicity": "未声明",
                        "approval": "仅实验候选",
                    },
                    "readiness": [
                        {"id": "native_model", "label": "原生模型", "status": "ready", "artifact": "/tmp/modeling-smoke/art-lgb.pkl"},
                        {"id": "pmml", "label": "PMML", "status": "succeeded", "artifact": "/tmp/modeling-smoke/art-lgb.pmml"},
                        {"id": "approval_package", "label": "审批包", "status": "ready", "artifact": "/tmp/modeling-smoke/art-lgb.approval_package.md"},
                        {"id": "model_card", "label": "模型卡", "status": "ready", "artifact": "/tmp/modeling-smoke/art-lgb.model_card.md"},
                        {"id": "monitoring_policy", "label": "监控策略", "status": "pass", "artifact": "/tmp/modeling-smoke/art-lgb.monitoring_policy.md"},
                        {"id": "challenger_comparison", "label": "Champion对比", "status": "warn", "artifact": "/tmp/modeling-smoke/art-lgb.champion_comparison.md"},
                    ],
                    "actions": {
                        "export_pmml": {"status": "succeeded", "path": "/tmp/modeling-smoke/art-lgb.pmml"},
                        "handoff_validation": {"status": "succeeded", "task_id": "task-validation"},
                        "approval_package": {"status": "succeeded", "path": "/tmp/modeling-smoke/art-lgb.approval_package.md"},
                    },
                    "candidates": [
                        {
                            "id": "exp-lgb",
                            "recipe": "lgb",
                            "selected": True,
                            "metrics": {"oot_ks": 0.31, "test_ks": 0.29, "psi_oot_vs_train": 0.12},
                            "business_signals": {
                                "stability": "关注",
                                "feature_count": 128,
                                "calibration": "需说明",
                                "delivery": "可移交",
                            },
                            "policy_signals": {
                                "scorecard": "非评分卡",
                                "monotonicity": "未声明",
                                "approval": "仅实验候选",
                            },
                            "capabilities": {
                                "pmml_supported": True,
                                "handoff_supported": True,
                                "native_model_supported": True,
                            },
                        }
                    ],
                },
            },
        }
    ]


def _real_modeling_plan() -> dict:
    return {
        "id": "plan-real-smoke",
        "task_id": "task-modeling-smoke",
        "goal": "模型开发 smoke",
        "status": "done",
        "tier": "guarded",
        "novel_mode": "plan_ahead",
        "created_at": "2026-06-30T00:00:00Z",
        "updated_at": "2026-06-30T00:01:00Z",
        "steps": [
            {
                "id": "choose_modeling_spec",
                "index": 0,
                "phase": "G1 规格",
                "title": "确认建模规格",
                "status": "done",
                "tool_ref": {"plugin": "modeling", "tool": "choose_modeling_spec"},
            },
            {
                "id": "train_models",
                "index": 1,
                "phase": "G4 训练",
                "title": "训练候选模型",
                "status": "done",
                "tool_ref": {"plugin": "modeling", "tool": "train_models"},
                "depends_on": ["choose_modeling_spec"],
            },
            {
                "id": "post_training_action",
                "index": 2,
                "phase": "G5 交付",
                "title": "训练后交付动作",
                "status": "done",
                "tool_ref": {"plugin": "modeling", "tool": "post_training_action"},
                "depends_on": ["train_models"],
                "output_ref": "artifact://modeling/art-lgb/model_card",
            },
        ],
    }


def test_modeling_setup_and_delivery_panels_render_in_real_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with _serve_smoke_page(_smoke_html()) as url, playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1366, "height": 900})
            _assert_panel_smoke(page, url)

            mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
            _assert_panel_smoke(mobile, url)
        finally:
            browser.close()


def test_app_shell_welcome_and_create_dialog_render_in_real_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with _serve_smoke_page(_smoke_html()) as url, playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for viewport, theme in (
                ({"width": 1366, "height": 900}, "light"),
                ({"width": 390, "height": 844}, "dark"),
            ):
                page = browser.new_page(viewport=viewport, is_mobile=viewport["width"] < 600)
                page.add_init_script(
                    f"localStorage.clear(); localStorage.setItem('marvis_theme', {json.dumps(theme)});",
                )
                _assert_app_shell_smoke(page, url.rsplit("/", 1)[0] + "/", expected_theme=theme)
        finally:
            browser.close()


def test_real_modeling_task_delivery_workspace_renders_in_real_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    task = _real_modeling_task()
    with _serve_smoke_page(
        _smoke_html(),
        tasks=[task],
        messages_by_task={task["id"]: _real_modeling_messages()},
        plans_by_task={task["id"]: [_real_modeling_plan()]},
    ) as url, playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1366, "height": 900})
            page.add_init_script(
                """
                localStorage.clear();
                localStorage.setItem("marvis_selected_task_id", "task-modeling-smoke");
                """
            )
            _assert_real_modeling_workspace_smoke(page, url.rsplit("/", 1)[0] + "/")
        finally:
            browser.close()


def test_plan_rail_and_screen_table_render_in_real_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with _serve_smoke_page(_workspace_smoke_html()) as url, playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for viewport in (
                {"width": 1366, "height": 900},
                {"width": 390, "height": 844},
            ):
                page = browser.new_page(viewport=viewport, is_mobile=viewport["width"] < 600)
                _assert_workspace_smoke(page, url)
        finally:
            browser.close()


def _assert_panel_smoke(page, url: str) -> None:
    page.goto(url)
    page.wait_for_selector(".modeling-setup-panel")
    page.wait_for_selector(".model-delivery-panel")
    metrics = page.evaluate(
        """
        () => {
          const setup = document.querySelector(".modeling-setup-panel").getBoundingClientRect();
          const delivery = document.querySelector(".model-delivery-panel").getBoundingClientRect();
          const badText = document.body.innerText.includes("undefined") || document.body.innerText.includes("NaN");
          const overflow = document.documentElement.scrollWidth - window.innerWidth;
          return {
            setupWidth: setup.width,
            setupHeight: setup.height,
            deliveryWidth: delivery.width,
            deliveryHeight: delivery.height,
            guidance: document.querySelectorAll(".modeling-guidance-item").length,
            readiness: document.querySelectorAll(".model-delivery-readiness-grid > *").length,
            policy: document.querySelectorAll(".model-delivery-policy-card").length,
            badText,
            overflow,
          };
        }
        """
    )
    assert metrics["setupWidth"] > 260
    assert metrics["setupHeight"] > 240
    assert metrics["deliveryWidth"] > 260
    assert metrics["deliveryHeight"] > 180
    assert metrics["guidance"] >= 3
    assert metrics["readiness"] >= 2
    assert metrics["policy"] >= 3
    assert metrics["badText"] is False
    assert metrics["overflow"] <= 1


def _assert_app_shell_smoke(page, url: str, *, expected_theme: str) -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error" and not message.text.startswith("Failed to load resource:")
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(url)
    page.wait_for_selector("body:not(.app-booting)", timeout=10_000)
    page.wait_for_selector("#welcomeTaskCards .welcome-task-card.available")
    shell_metrics = page.evaluate(
        """
        () => {
          const welcome = document.querySelector("#workspaceWelcome").getBoundingClientRect();
          const cards = [...document.querySelectorAll("#welcomeTaskCards .welcome-task-card.available")]
            .map((card) => card.getBoundingClientRect());
          return {
            theme: document.body.dataset.theme,
            welcomeWidth: welcome.width,
            welcomeHeight: welcome.height,
            cards: cards.length,
            minCardWidth: Math.min(...cards.map((card) => card.width)),
            minCardHeight: Math.min(...cards.map((card) => card.height)),
            badText: document.body.innerText.includes("undefined") || document.body.innerText.includes("NaN"),
            overflow: document.documentElement.scrollWidth - window.innerWidth,
          };
        }
        """
    )
    assert shell_metrics["theme"] == expected_theme
    assert shell_metrics["welcomeWidth"] > 260
    assert shell_metrics["welcomeHeight"] > 260
    assert shell_metrics["cards"] >= 6
    assert shell_metrics["minCardWidth"] > 120
    assert shell_metrics["minCardHeight"] > 80
    assert shell_metrics["badText"] is False
    assert shell_metrics["overflow"] <= 1

    page.click("#welcomeModelDevelopmentCard")
    page.wait_for_selector("#taskDialog[open]")
    dialog_metrics = page.evaluate(
        """
        () => {
          const dialog = document.querySelector("#taskDialog");
          const panel = document.querySelector("#taskDialog .task-dialog-panel").getBoundingClientRect();
          document.querySelector("#runModeManual").click();
          const manualAlgorithmVisible = !document.querySelector("#createTaskAlgorithmField").hidden;
          const manualTierVisible = !document.querySelector("#createTaskTierField").hidden;
          document.querySelector("#runModeAgent").click();
          const agentAlgorithmVisible = !document.querySelector("#createTaskAlgorithmField").hidden;
          const agentTierVisible = !document.querySelector("#createTaskTierField").hidden;
          return {
            open: dialog.open,
            title: document.querySelector("#taskDialogTitle").textContent,
            taskType: document.querySelector("#taskType").value,
            panelWidth: panel.width,
            panelHeight: panel.height,
            manualAlgorithmVisible,
            manualTierVisible,
            agentAlgorithmVisible,
            agentTierVisible,
            tierValue: document.querySelector("#createTaskTier").value,
            overflow: document.documentElement.scrollWidth - window.innerWidth,
          };
        }
        """
    )
    assert dialog_metrics["open"] is True
    assert "建模" in dialog_metrics["title"]
    assert dialog_metrics["taskType"] == "modeling"
    assert dialog_metrics["panelWidth"] > 260
    assert dialog_metrics["panelHeight"] > 320
    assert dialog_metrics["manualAlgorithmVisible"] is True
    assert dialog_metrics["manualTierVisible"] is False
    assert dialog_metrics["agentAlgorithmVisible"] is False
    assert dialog_metrics["agentTierVisible"] is True
    assert dialog_metrics["tierValue"] in {"", "guarded", "balanced", "explorer", "autonomous"}
    assert dialog_metrics["overflow"] <= 1
    assert not page_errors
    assert not console_errors


def _assert_real_modeling_workspace_smoke(page, url: str) -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error" and not message.text.startswith("Failed to load resource:")
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(url)
    page.wait_for_selector("body:not(.app-booting)", timeout=10_000)
    page.wait_for_selector("#taskList .task-row.selected")
    page.wait_for_selector("#workflowStepper .plan-rail-step")
    page.wait_for_selector(".driver-analysis-section .model-delivery-panel")
    metrics = page.evaluate(
        """
        () => {
          const delivery = document.querySelector(".model-delivery-panel").getBoundingClientRect();
          const plan = document.querySelector("#workflowStepper").getBoundingClientRect();
          const analysis = document.querySelector("#agentConversationPanel").getBoundingClientRect();
          const text = document.body.innerText;
          return {
            selectedRows: document.querySelectorAll("#taskList .task-row.selected").length,
            planSteps: document.querySelectorAll("#workflowStepper .plan-rail-step").length,
            outputButtons: document.querySelectorAll(".step-output-button, .plan-step-output").length,
            deliveryWidth: delivery.width,
            deliveryHeight: delivery.height,
            analysisWidth: analysis.width,
            planWidth: plan.width,
            hasModelCard: text.includes("模型卡") && text.includes("art-lgb.model_card.md"),
            hasChampionComparison: text.includes("Champion对比") && text.includes("art-lgb.champion_comparison.md"),
            hasApprovalPackage: text.includes("审批包") && text.includes("art-lgb.approval_package.md"),
            hasTaskTitle: text.includes("A卡模型开发"),
            badText: text.includes("undefined") || text.includes("NaN"),
            overflow: document.documentElement.scrollWidth - window.innerWidth,
          };
        }
        """
    )
    assert metrics["selectedRows"] == 1
    assert metrics["planSteps"] == 3
    assert metrics["outputButtons"] >= 1
    assert metrics["deliveryWidth"] > 320
    assert metrics["deliveryHeight"] > 180
    assert metrics["analysisWidth"] > 320
    assert metrics["planWidth"] > 240
    assert metrics["hasModelCard"] is True
    assert metrics["hasChampionComparison"] is True
    assert metrics["hasApprovalPackage"] is True
    assert metrics["hasTaskTitle"] is True
    assert metrics["badText"] is False
    assert metrics["overflow"] <= 1
    assert not page_errors
    assert not console_errors


def _assert_workspace_smoke(page, url: str) -> None:
    page.goto(url)
    page.wait_for_selector(".v2-plan")
    page.wait_for_selector(".screen-table-wrap")
    metrics = page.evaluate(
        """
        () => {
          const plan = document.querySelector(".v2-plan").getBoundingClientRect();
          const screen = document.querySelector(".screen-table-wrap").getBoundingClientRect();
          const retry = document.querySelector(".retry-step-panel");
          return {
            planWidth: plan.width,
            planHeight: plan.height,
            screenWidth: screen.width,
            screenHeight: screen.height,
            steps: document.querySelectorAll(".plan-step").length,
            awaitingConfirm: document.querySelectorAll(".step-status-awaiting_confirm").length,
            failed: document.querySelectorAll(".step-status-failed").length,
            outputButtons: document.querySelectorAll(".step-output-button").length,
            retryPanel: Boolean(retry),
            screenRows: document.querySelectorAll(".screen-row").length,
            leakageRows: document.querySelectorAll(".screen-row.screen-leakage").length,
            thresholdInputs: document.querySelectorAll(".screen-threshold-input").length,
            badText: document.body.innerText.includes("undefined") || document.body.innerText.includes("NaN"),
            overflow: document.documentElement.scrollWidth - window.innerWidth,
          };
        }
        """
    )
    assert metrics["planWidth"] > 260
    assert metrics["planHeight"] > 180
    assert metrics["screenWidth"] > 260
    assert metrics["screenHeight"] > 180
    assert metrics["steps"] == 4
    assert metrics["awaitingConfirm"] >= 1
    assert metrics["failed"] >= 1
    assert metrics["outputButtons"] == 1
    assert metrics["retryPanel"] is True
    assert metrics["screenRows"] == 4
    assert metrics["leakageRows"] == 1
    assert metrics["thresholdInputs"] == 2
    assert metrics["badText"] is False
    assert metrics["overflow"] <= 1
