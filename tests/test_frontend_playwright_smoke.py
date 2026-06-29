"""Optional browser smoke tests for the V2 frontend panels.

These tests are skipped by default because CI currently does not install
Playwright browsers. Run locally with:

    MARVIS_RUN_PLAYWRIGHT_SMOKE=1 python -m pytest tests/test_frontend_playwright_smoke.py -q
"""

from __future__ import annotations

import contextlib
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
    reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 to run browser smoke tests.",
)


ROOT = Path(__file__).resolve().parents[1]


class _SmokeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, smoke_html: str, **kwargs):
        self._smoke_html = smoke_html
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/smoke.html":
            body = self._smoke_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


@contextlib.contextmanager
def _serve_smoke_page(smoke_html: str):
    handler = partial(_SmokeHandler, smoke_html=smoke_html)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/smoke.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


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
          readiness: [
            { id: "native_model", label: "原生模型", status: "ready", artifact: "/tmp/model.pkl" },
            { id: "pmml", label: "PMML", status: "succeeded", artifact: longPath },
          ],
          metrics: { oot_ks: 0.31, test_ks: 0.29, psi_oot_vs_train: 0.12 },
          candidates: [
            {
              id: "exp-lgb",
              recipe: "lgb",
              selected: true,
              metrics: { oot_ks: 0.31, test_ks: 0.29, psi_oot_vs_train: 0.12 },
              business_signals: { stability: "关注", feature_count: 128, calibration: "需说明", delivery: "可移交" },
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
    assert metrics["badText"] is False
    assert metrics["overflow"] <= 1
