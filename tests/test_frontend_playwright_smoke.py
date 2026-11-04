"""Browser smoke tests for the V2 frontend panels.

The explicit full-release CI lane installs Chromium and enables this suite.
Run locally with:

    MARVIS_RUN_PLAYWRIGHT_SMOKE=1 python -m pytest tests/test_frontend_playwright_smoke.py -q
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import threading
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

from marvis.app import _static_asset_version, _static_import_map
from marvis.execution_environment import ExecutionEnvironmentSettings
from marvis.routers.plans import list_capability_tiers
from tests.static_stylesheets import browser_stylesheet_hrefs


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
        reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 to run browser smoke tests.",
    ),
]


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
            self._send_index_html()
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
        if path == "/api/plugins":
            self._send_json({"plugins": []})
            return
        if path == "/api/skills":
            self._send_json({"skills": [], "builtin": [],
                             "counts": {"active": 0, "disabled": 0, "rejected": 0}})
            return
        if path == "/api/capability-tiers":
            self._send_json(list_capability_tiers())
            return
        if path == "/api/plugins/governance":
            self._send_json({
                "settings": {"mode": "trusted_local", "revision": 0},
                "publishers": [], "versions": [],
                "signature_schema": "marvis.capability-signature.v1",
                "signature_verification_available": False,
                "boundaries": {
                    "identity": "workspace_server_session_and_admin_confirmation",
                    "immutable_versions": "read_only_material_with_signature_and_hash_verification",
                    "hostile_same_uid_os_isolation": False,
                    "independent_skill_runtime": False,
                },
            })
            return
        if path == "/api/institution/status":
            self._send_json({
                "singleton": 1, "mode": "trusted_local", "audience": "marvis-local", "revision": 0,
                "protocol": "marvis.identity-assertion.v1", "algorithm": "Ed25519",
                "external_identity_verified": False,
                "resource_policy": "canonical owner and expiring explicit grants",
                "unsupported": ["arbitrary Notebook", "unregistered local path intake", "unknown HTTP surfaces"],
            })
            return
        if path == "/api/production-governance/me":
            # This UI fixture has no registered production principal. Preserve
            # the real endpoint's denial instead of granting a synthetic role.
            self._send_json({"detail": "production principal is not registered"}, status=403)
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
        if path in {"/api/settings/execution-environment", "/api/settings/execution-environment/options"}:
            self._send_json({
                "settings": asdict(ExecutionEnvironmentSettings()), "options": [],
                "validation": {"ok": False, "message": "UI fixture does not launch kernels",
                               "kernel_name": "python3", "python_version": "",
                               "network_capability": {}, "worker_capability": {}},
            })
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

    def _send_index_html(self) -> None:
        # Mirrors marvis.app's "/" route: the real app replaces
        # __MARVIS_STATIC_VERSION__ / __MARVIS_STATIC_IMPORT_MAP__ before
        # serving index.html, so this smoke harness must do the same or the
        # <script type="importmap"> tag ships the literal placeholder text,
        # which is not valid JSON and fails to parse in a real browser.
        static_dir = ROOT / "marvis/static"
        index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        static_version = _static_asset_version(static_dir)
        index_html = index_html.replace("__MARVIS_STATIC_VERSION__", static_version)
        index_html = index_html.replace(
            "__MARVIS_STATIC_IMPORT_MAP__", _static_import_map(static_dir, static_version)
        )
        body = index_html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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


class _SmokeHTTPServer(ThreadingHTTPServer):
    # Chromium's module graph opens a burst of connections. The stdlib default
    # queue of five can reset module requests while other test workers run.
    request_queue_size = 128


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
    server = _SmokeHTTPServer(("127.0.0.1", 0), handler)
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
    stylesheet_links = "\n".join(
        f'<link rel="stylesheet" href="/{urlparse(href).path}" />'
        for href in browser_stylesheet_hrefs(ROOT / "marvis/static")
    )
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  __MARVIS_SMOKE_STYLESHEETS__
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
    return html.replace("__MARVIS_SMOKE_STYLESHEETS__", stylesheet_links)


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
            "task_id": "task-modeling-smoke",
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
                    "pmml_path": "tasks/task-modeling-smoke/art-lgb.pmml",
                    "model_report_path": "/tmp/modeling-smoke/model_report.xlsx",
                    "validation_task_id": "task-validation",
                    "approval_package_path": "tasks/task-modeling-smoke/art-lgb.approval_package.md",
                    "monitoring_policy_path": "/tmp/modeling-smoke/art-lgb.monitoring_policy.md",
                    "champion_comparison_path": "/tmp/modeling-smoke/art-lgb.champion_comparison.md",
                    "model_card_path": "/tmp/modeling-smoke/art-lgb.model_card.json",
                    "model_card_markdown_path": "tasks/task-modeling-smoke/art-lgb.model_card.md",
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


@pytest.mark.parametrize(
    ("theme", "collapsed", "width"),
    [
        ("light", False, 1366),
        ("dark", False, 1366),
        ("light", True, 1366),
        ("dark", True, 1366),
        ("light", True, 390),
        ("dark", True, 390),
    ],
)
def test_task_search_stays_above_scrim_and_remains_interactive(theme, collapsed, width):
    playwright = pytest.importorskip("playwright.sync_api")
    tasks = [
        {**_real_modeling_task(), "id": "search-alpha", "model_name": "Search layer alpha"},
        {**_real_modeling_task(), "id": "search-beta", "model_name": "Search layer beta"},
    ]
    with _serve_smoke_page(_smoke_html(), tasks=tasks) as url, playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.set_default_timeout(5000)
            page.add_init_script(
                "localStorage.clear();"
                f"localStorage.setItem('marvis_theme', {json.dumps(theme)});"
                f"localStorage.setItem('sidebarCollapsed', {'1' if collapsed else '0'});"
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            page.wait_for_selector("#taskList .task-row", state="attached")
            page.locator("#taskSearchToggle").click()
            page.wait_for_function("document.body.classList.contains('search-active')")
            page.wait_for_timeout(350)  # Let the search flyout and scrim finish their transitions.

            def hit_test(selector):
                return page.evaluate(
                    """selector => {
                      const target = document.querySelector(selector);
                      const box = target.getBoundingClientRect();
                      const hit = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
                      return {
                        selector,
                        receivesPointer: target === hit || target.contains(hit),
                        hitId: hit?.id || "",
                        hitClass: hit?.className || "",
                      };
                    }""",
                    selector,
                )

            input_hit = hit_test("#taskSearchInput")
            assert input_hit["receivesPointer"], input_hit
            result_hit = hit_test("#taskList .task-row")
            assert result_hit["receivesPointer"], result_hit
            page.locator("#taskSearchInput").fill("alpha")
            assert page.locator("#taskList .task-row").count() == 1
            assert page.locator("#taskList .task-row-name").inner_text() == "Search layer alpha · v1"
            page.keyboard.press("Escape")
            assert not page.locator("body").evaluate("el => el.classList.contains('search-active')")
            assert page.locator("#taskList .task-row").count() == 2
            assert page.locator("#taskSearchInput").input_value() == ""
            page.locator("#taskSearchToggle").click()
            page.locator("#searchScrim").click(position={"x": width - 8, "y": 892})
            assert not page.locator("body").evaluate("el => el.classList.contains('search-active')")
            assert not errors
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
    import_map_warnings: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error" and not message.text.startswith("Failed to load resource:")
        else None,
    )
    page.on(
        "console",
        lambda message: import_map_warnings.append(message.text)
        if message.type == "warning" and "Ignored an import map value" in message.text
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
            moduleUrls: performance.getEntriesByType("resource")
              .map((entry) => entry.name)
              .filter((resourceUrl) => resourceUrl.includes("/static/js/")),
          };
        }
        """
    )
    assert shell_metrics["theme"] == expected_theme
    assert shell_metrics["welcomeWidth"] > 260
    assert shell_metrics["welcomeHeight"] > 260
    assert shell_metrics["cards"] >= 6
    # Batch 5 (D7): icon + name buttons in one row (four per row on phones).
    assert shell_metrics["minCardWidth"] > 56
    assert shell_metrics["minCardHeight"] > 50
    assert shell_metrics["badText"] is False
    assert shell_metrics["overflow"] <= 1
    assert shell_metrics["moduleUrls"]
    assert all("?v=" in module_url for module_url in shell_metrics["moduleUrls"])

    # A work icon only fills the home composer; the sentence opens the creation
    # card, and 高级设置 opens the create dialog prefilled with it.
    page.click("#welcomeModelDevelopmentCard")
    home_fill = page.evaluate(
        """
        () => ({
          dialogOpen: document.querySelector("#taskDialog").open,
          value: document.querySelector("#homeComposerInput").value,
          sendDisabled: document.querySelector("#homeComposerSend").disabled,
        })
        """
    )
    assert home_fill["dialogOpen"] is False
    # U8 (2026-10-19): the icon fills a fill-in sentence with editable slots.
    assert home_fill["value"].startswith("根据这个Y样本和这几个X数据，给我做一个MOB3_30的A卡模型")
    assert home_fill["sendDisabled"] is False
    page.press("#homeComposerTemplate", "Enter")
    page.wait_for_selector("#homeDraft .home-draft-card")
    draft_metrics = page.evaluate(
        """
        () => ({
          composerVisible: !!document.querySelector("#homeComposer").offsetParent,
          type: document.querySelector("#homeDraft [data-home-draft-type][aria-pressed='true']")?.dataset.homeDraftType,
          name: document.querySelector("#homeDraft [data-home-draft-field='name']").value,
          lead: document.querySelector("#homeDraft .home-draft-lead").textContent,
          overflow: document.documentElement.scrollWidth - window.innerWidth,
        })
        """
    )
    assert draft_metrics["composerVisible"] is False
    assert draft_metrics["type"] == "modeling"
    assert draft_metrics["name"] == "MOB3_30 A卡模型"
    assert "模型开发" in draft_metrics["lead"]
    assert draft_metrics["overflow"] <= 1
    page.click("#homeDraft [data-home-draft-advanced]")
    page.wait_for_selector("#taskDialog[open]")
    assert page.input_value("#modelName") == "MOB3_30 A卡模型"
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
    # The card's strategy limits reach the dialog as its 0-1 fractions.
    page.click("#closeTaskDialogButton")
    page.click("#homeDraft [data-home-draft-back]")
    # Examples are a random pick (V3 UI-7); type the strategy example instead.
    # Returning from a work-icon draft keeps its editable template visible.
    editor = "#homeComposerTemplate" if page.locator("#homeComposerTemplate").is_visible() else "#homeComposerInput"
    page.fill(editor, "通过率 60% 以上，坏账 5% 以内的准入策略")
    page.set_input_files(
        "#homeComposerFileInput",
        files=[{"name": "score_sample.csv", "mimeType": "text/csv", "buffer": b"apply_id,score,bad\n1,600,0\n"}],
    )
    page.press(editor, "Enter")
    page.wait_for_selector('#homeDraft [data-home-draft-limit="minApprovalRate"]')
    page.click("#homeDraft [data-home-draft-advanced]")
    page.wait_for_selector("#taskDialog[open]")
    strategy_prefill = page.evaluate(
        """
        () => ({
          taskType: document.querySelector("#taskType").value,
          name: document.querySelector("#modelName").value,
          entry: document.querySelector("#strategyEntryMode").value,
          objective: document.querySelector("#strategyObjective").value,
          minApproval: document.querySelector("#strategyMinApprovalRate").value,
          maxBad: document.querySelector("#strategyMaxBadRate").value,
          files: document.querySelector("#materialUploadStatus").textContent,
        })
        """
    )
    assert strategy_prefill == {
        "taskType": "strategy",
        "name": "准入策略",
        "entry": "strategy_development",
        "objective": "max_approval",
        "minApproval": "0.6",
        "maxBad": "0.05",
        "files": "已选择 score_sample.csv。",
    }
    assert not page_errors
    assert not console_errors
    assert not import_map_warnings


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
            railOutputButtons: document.querySelectorAll("#workflowStepper .step-output-button, #workflowStepper .plan-step-output").length,
            deliveryArtifactItems: document.querySelectorAll(".model-delivery-panel .model-delivery-artifacts > div").length,
            deliveryDownloads: document.querySelectorAll(".model-delivery-panel [data-model-delivery-download]").length,
            deliveryWidth: delivery.width,
            deliveryHeight: delivery.height,
            analysisWidth: analysis.width,
            planWidth: plan.width,
            // Deliverables read by human labels; the stored file name only
            // rides in the tooltip, and each file stays downloadable.
            hasModelCard: text.includes("模型卡")
              && Boolean(document.querySelector('.model-delivery-panel [data-model-delivery-download="model_card"]'))
              && Boolean(document.querySelector('.model-delivery-panel [title="art-lgb.model_card.md"]')),
            hasChampionComparison: text.includes("Champion对比")
              && Boolean(document.querySelector('[data-readiness-id="challenger_comparison"] [title="art-lgb.champion_comparison.md"]')),
            hasApprovalPackage: text.includes("审批材料包")
              && Boolean(document.querySelector('.model-delivery-panel [data-model-delivery-download="approval_package"]'))
              && Boolean(document.querySelector('.model-delivery-panel [title="art-lgb.approval_package.md"]')),
            showsInternalFileNames: /art-lgb\.(model_card|approval_package|champion_comparison)\.md/.test(text),
            hasTaskTitle: text.includes("A卡模型开发"),
            badText: text.includes("undefined") || text.includes("NaN"),
            overflow: document.documentElement.scrollWidth - window.innerWidth,
          };
        }
        """
    )
    assert metrics["selectedRows"] == 1
    assert metrics["planSteps"] == 3
    assert metrics["railOutputButtons"] == 0
    assert metrics["deliveryArtifactItems"] >= 3
    assert metrics["deliveryDownloads"] >= 3
    assert metrics["deliveryWidth"] > 320
    assert metrics["deliveryHeight"] > 180
    assert metrics["analysisWidth"] > 320
    # The compact 278px track leaves 238px inside the rail's border/padding.
    assert metrics["planWidth"] >= 220
    assert metrics["hasModelCard"] is True
    assert metrics["hasChampionComparison"] is True
    assert metrics["hasApprovalPackage"] is True
    assert metrics["showsInternalFileNames"] is False
    assert metrics["hasTaskTitle"] is True
    assert metrics["badText"] is False
    assert metrics["overflow"] <= 1
    assert not page_errors
    assert not console_errors


# D-10 (batch 1b): English UI guard. The fixture payloads go through the same
# serve-time localization an English request gets from UiLocaleMiddleware.
_ENGLISH_GUARD_JS = r"""
() => {
  const cjk = /[㐀-䶿一-鿿]/;
  const userData = [
    '[translate="no"]', '[data-i18n-skip]', 'script', 'style', 'textarea', 'pre', 'code',
    '.agent-message.user', '.agent-markdown',
    '.agent-message-content > :is(p, ul, ol, h1, h2, h3, h4, h5, h6, blockquote, table, hr)',
    '.agent-message-tables td', 'td', '.task-row-name', '.task-hero-name', '#currentTaskTitle', '.home-continue-name',
    '[data-field-name]', '[data-file-name]',
  ].join(', ');
  const left = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const parent = node.parentElement;
    if (!parent || !cjk.test(node.data)) continue;
    if (parent.closest(userData)) continue;
    if (!parent.getClientRects().length) continue;  // not rendered
    left.push(node.data.trim());
  }
  return [...new Set(left)];
}
"""


def test_english_mode_shows_no_platform_chinese_outside_user_data():
    playwright = pytest.importorskip("playwright.sync_api")
    from marvis.ui_locale import localize_payload

    task = _real_modeling_task()
    messages = localize_payload(_real_modeling_messages(), locale="en")
    plan = _real_modeling_plan()
    for step, phase in zip(plan["steps"], ("建模", "建模", "交付")):
        step["phase"] = phase  # phases as the real modeling template names them
    plan = localize_payload(plan, locale="en")
    with _serve_smoke_page(
        _smoke_html(),
        tasks=[task],
        messages_by_task={task["id"]: messages},
        plans_by_task={task["id"]: [plan]},
    ) as url, playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            base = url.rsplit("/", 1)[0] + "/"
            home = browser.new_page(viewport={"width": 1366, "height": 900})
            home.add_init_script("localStorage.clear(); localStorage.setItem('marvis_ui_locale', 'en');")
            home.goto(base)
            home.wait_for_function("() => !document.body.classList.contains('app-booting')")
            assert home.evaluate("document.documentElement.lang") == "en"
            assert home.evaluate(_ENGLISH_GUARD_JS) == []

            workspace = browser.new_page(viewport={"width": 1366, "height": 900})
            workspace.add_init_script(
                "localStorage.clear(); localStorage.setItem('marvis_ui_locale', 'en');"
                f"localStorage.setItem('marvis_selected_task_id', {json.dumps(task['id'])});"
            )
            workspace.goto(base)
            workspace.wait_for_function("() => !document.body.classList.contains('app-booting')")
            workspace.wait_for_timeout(800)
            assert workspace.evaluate(_ENGLISH_GUARD_JS) == []

            # Switching back restores Chinese without a reload.
            workspace.evaluate("document.querySelector(\"[data-ui-locale-option='zh']\").click()")
            assert workspace.evaluate("document.documentElement.lang") == "zh-CN"
            assert workspace.evaluate("document.querySelector('#createTaskOpenButton span').textContent") == "新建任务"
        finally:
            browser.close()
