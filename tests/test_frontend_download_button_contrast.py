"""Real app CSS/renderer contrast checks using synthetic report metadata.

This is a rendering fixture, not evidence that a report workflow executed.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.test_frontend_playwright_smoke import (
    _real_modeling_task,
    _serve_smoke_page,
    _smoke_html,
)


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
        reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 for Chromium contrast checks.",
    ),
]


_RENDERED_COLORS = """node => {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 1;
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  const rgba = color => {
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, 1, 1);
    const p = ctx.getImageData(0, 0, 1, 1).data;
    return [p[0], p[1], p[2], p[3] / 255];
  };
  const over = (fg, bg) => [0, 1, 2].map(i => fg[i] * fg[3] + bg[i] * (1 - fg[3]));
  // Composite translucent glass over its actual ancestor surfaces. Using
  // Canvas accepts Chromium's color(srgb ...) serialization as well as rgba.
  const layers = [];
  for (let parent = node; parent; parent = parent.parentElement) {
    layers.push(rgba(getComputedStyle(parent).backgroundColor));
  }
  let background = [255, 255, 255];
  for (const layer of layers.reverse()) background = over(layer, background);
  const style = getComputedStyle(node);
  const foreground = over(rgba(style.color), background);
  const luminance = rgb => {
    const channels = rgb.map(value => {
      const s = value / 255;
      return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  const a = luminance(foreground), b = luminance(background);
  return {
    foreground, background, color: style.color,
    backgroundColor: style.backgroundColor,
    ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05),
    decoration: style.textDecorationLine,
    focused: node === document.activeElement,
    focusVisible: node.matches(':focus-visible'),
  };
}"""


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("locale", ["zh", "en"])
@pytest.mark.parametrize("run_mode", ["manual", "agent"])
def test_report_download_button_contrast_and_body_links(theme, locale, run_mode):
    playwright = pytest.importorskip("playwright.sync_api")
    task = {
        **_real_modeling_task(), "task_type": "feature_analysis",
        "model_name": "Synthetic report rendering", "run_mode": run_mode,
        "status": "succeeded",
    }
    messages = [{
        "id": "synthetic-report-result", "task_id": task["id"],
        "role": "assistant", "stage": "chat",
        "content": "Report fixture. [Source details](/api/tasks/fixture/evidence)",
        "created_at": "2026-11-04T00:00:00Z",
        "metadata": {
            "kind": "result",
            "report_download": {
                "label": "下载特征分析报告",
                "download_url": f"/api/tasks/{task['id']}/driver-report/download",
            },
        },
    }]
    with _serve_smoke_page(
        _smoke_html(), tasks=[task], messages_by_task={task["id"]: messages},
    ) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.set_default_timeout(8000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.add_init_script(
                "localStorage.clear();"
                f"localStorage.setItem('marvis_selected_task_id', {json.dumps(task['id'])});"
                f"localStorage.setItem('marvis_theme', {json.dumps(theme)});"
                f"localStorage.setItem('marvis_ui_locale', {json.dumps(locale)});",
            )
            page.goto(address.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            button = page.locator(
                ".result-dataset-download.report-result-download a.button.compact.primary"
            )
            button.wait_for(state="visible")
            assert button.count() == 1
            assert button.get_attribute("download") is not None
            page.wait_for_function(
                "prefix => document.querySelector('[data-agent-report-download]')?.textContent.includes(prefix)",
                arg="Download" if locale == "en" else "下载",
            )
            assert page.locator("body").get_attribute("data-theme") == theme
            assert page.locator("html").get_attribute("lang") == ("en" if locale == "en" else "zh-CN")
            plain_link = page.get_by_role("link", name="Source details", exact=True)
            plain_link.wait_for(state="visible")
            observations = {}
            for state in ("default", "hover", "focus"):
                if state == "hover":
                    button.hover()
                elif state == "focus":
                    page.mouse.move(1500, 900)
                    button.focus()
                    page.keyboard.press("Tab")
                    page.keyboard.press("Shift+Tab")
                    assert button.evaluate("node => node === document.activeElement")
                    assert button.evaluate("node => node.matches(':focus-visible')")
                else:
                    page.mouse.move(1500, 900)
                # Let the real hover/focus CSS transitions reach their endpoint.
                page.wait_for_timeout(300)
                observations[state] = button.evaluate(_RENDERED_COLORS)
                plain = plain_link.evaluate("""node => {
                  const style = getComputedStyle(node);
                  return {decoration: style.textDecorationLine, color: style.color,
                          parentColor: getComputedStyle(node.parentElement).color};
                }""")
                assert "underline" in plain["decoration"], plain
                assert plain["color"] == plain["parentColor"], plain
            assert errors == []
            context = {"theme": theme, "locale": locale, "run_mode": run_mode}
            details = json.dumps({**context, "states": observations}, ensure_ascii=False)
            assert all(value["ratio"] >= 4.5 for value in observations.values()), details
            assert all("underline" not in value["decoration"] for value in observations.values()), details
        finally:
            browser.close()
