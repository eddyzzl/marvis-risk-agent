"""Header output popover keeps the real preview/download actions available."""

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
        reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 to run browser smoke tests.",
    ),
]


@pytest.mark.parametrize("theme,width,locale", [("light", 1440, "zh"), ("dark", 1440, "en"), ("light", 390, "zh"), ("light", 390, "en")])
def test_header_artifacts_keep_preview_downloads_and_readiness(theme, width, locale):
    playwright = pytest.importorskip("playwright.sync_api")
    task = {
        **_real_modeling_task(),
        "task_type": "validation",
        "validation_workflow_version": 2,
        "model_name": "Output controls fixture",
        "run_mode": "manual",
    }
    with _serve_smoke_page(_smoke_html(), tasks=[task]) as url, playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.set_default_timeout(5000)
            page.add_init_script(
                "localStorage.clear();"
                f"localStorage.setItem('marvis_theme', {json.dumps(theme)});"
                f"localStorage.setItem('marvis_ui_locale', {json.dumps(locale)});"
                f"localStorage.setItem('marvis_selected_task_id', {json.dumps(task['id'])});"
            )
            page.route("**/report/preview?**", lambda route: route.fulfill(
                content_type="text/html", body="<html><body>Preview fixture</body></html>",
            ))
            for endpoint, filename in (("report", "validation.docx"), ("analysis", "metrics.xlsx")):
                page.route(f"**/{endpoint}/download", lambda route, request, filename=filename: route.fulfill(
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                    content_type="application/octet-stream", body=b"download fixture",
                ))
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            toggle = page.locator("#taskArtifactsToggle")
            toggle.wait_for(state="visible")
            if width >= 600:
                # Toolbar shadows must paint into the grid gutter beside the
                # progress rail, rather than stop at the results column edge.
                clipping = toggle.evaluate("""button => {
                  const right = button.getBoundingClientRect().right + 16;
                  const clipped = [];
                  for (let node = button.parentElement; node; node = node.parentElement) {
                    const style = getComputedStyle(node);
                    if (["hidden", "clip", "auto", "scroll"].includes(style.overflowX)
                        && node.getBoundingClientRect().right < right) {
                      clipped.push(node.id || node.className);
                    }
                  }
                  return clipped;
                }""")
                assert not clipping, clipping
            assert page.locator("#workflowStepper .step-download-actions").count() == 0
            toggle.click()
            panel = page.locator("#taskArtifacts")
            assert panel.is_visible()
            assert panel.locator(".rail-artifact").count() == 2
            assert panel.locator("[data-step-action]").count() == 3
            bounds = panel.bounding_box()
            assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
            layout = panel.evaluate(
                """panel => {
                  const number = value => Number.parseFloat(value) || 0;
                  const rect = element => element.getBoundingClientRect();
                  const horizontal = (style, prefix, suffix = "") =>
                    number(style[`${prefix}Left${suffix}`]) + number(style[`${prefix}Right${suffix}`]);
                  const textWidth = element => {
                    const range = document.createRange();
                    range.selectNodeContents(element);
                    return range.getBoundingClientRect().width;
                  };
                  const body = panel.querySelector(".task-artifacts-body");
                  const list = panel.querySelector(".step-download-actions");
                  const bodyStyle = getComputedStyle(body);
                  const panelStyle = getComputedStyle(panel);
                  const rows = [...panel.querySelectorAll(".rail-artifact")].map(row => {
                    const style = getComputedStyle(row);
                    const names = [...row.querySelectorAll(".rail-artifact-name b, .rail-artifact-name small")];
                    const buttons = [...row.querySelectorAll(".step-action-button")];
                    const icon = row.querySelector(".rail-artifact-icon");
                    const needed = rect(icon).width + Math.max(...names.map(textWidth))
                      + buttons.reduce((sum, button) => sum + rect(button).width, 0)
                      + number(style.columnGap) * (buttons.length + 1)
                      + horizontal(style, "padding");
                    return {
                      width: rect(row).width,
                      lastButtonInset: rect(row).right - rect(buttons.at(-1)).right,
                      clippedNames: names.filter(name => name.scrollWidth > name.clientWidth + 1)
                        .map(name => ({text: name.textContent, available: name.clientWidth, required: name.scrollWidth})),
                      needed,
                    };
                  });
                  return {
                    panelWidth: rect(panel).width,
                    availableBodyWidth: body.clientWidth - horizontal(bodyStyle, "padding"),
                    listWidth: rect(list).width,
                    excessSingleRowWidth: rect(panel).width - Math.max(...rows.map(row => row.needed))
                      - horizontal(bodyStyle, "padding") - horizontal(panelStyle, "border", "Width"),
                    rows,
                  };
                }""",
            )
            assert abs(layout["listWidth"] - layout["availableBodyWidth"]) <= 1, layout
            for row in layout["rows"]:
                assert abs(row["width"] - layout["listWidth"]) <= 1, layout
                assert 0 <= row["lastButtonInset"] <= 12, layout
                assert not row["clippedNames"], layout
            if width >= 600:
                # A short Chinese label must not inherit the English panel's
                # fixed width and leave a large unused name column.
                assert layout["excessSingleRowWidth"] <= 48, layout

            panel.locator('[data-step-action="previewWordReport"]').click()
            page.wait_for_selector("#wordPreviewDialog[open]")
            assert f"/report/preview?ui_locale={locale}" in page.locator("#wordPreviewFrame").get_attribute("src")
            page.locator("#closeWordPreviewButton").click()
            page.keyboard.press("Escape")
            assert not panel.is_visible()
            assert toggle.evaluate("e => e === document.activeElement")

            toggle.click()
            for action, filename in (("downloadWordReport", "validation.docx"), ("downloadExcelAnalysis", "metrics.xlsx")):
                with page.expect_download() as download:
                    panel.locator(f'[data-step-action="{action}"]').click()
                assert download.value.suggested_filename == filename
            page.mouse.click(width - 5, 990)
            assert not panel.is_visible()

            # Reusing the same task after readiness changes must remove stale actions.
            task["report_available"] = False
            page.reload()
            page.wait_for_selector("body:not(.app-booting)")
            assert not toggle.is_visible()
            assert page.locator("#taskArtifacts [data-step-action]").count() == 0
            assert not errors
        finally:
            browser.close()
