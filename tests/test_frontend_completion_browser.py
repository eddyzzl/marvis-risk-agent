"""Real wide-desktop boot bindings for F13's retained DOM/style contracts."""

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.test_frontend_playwright_smoke import _serve_smoke_page, _smoke_html


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
                       reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 for the Chromium app binding check"),
]


def _expected_principal_denial(url, status):
    # The fixture deliberately has no registered production principal; this
    # exact denial is the real local-mode contract, not a successful login.
    return urlparse(url).path == "/api/production-governance/me" and status == 403


def test_wide_app_boot_binds_language_welcome_and_task_preview(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    tasks = [{"id": "preview-task", "model_name": "用户原始名称", "validator": "QA",
              "model_version": "v1", "task_type": "validation", "run_mode": "manual",
              "status": "created", "created_at": "2026-11-04T00:00:00Z"}]
    with _serve_smoke_page(_smoke_html(), tasks=tasks) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for width in (1440, 1600, 1920):
                page = browser.new_page(viewport={"width": width, "height": 1000})
                errors = []
                console_issues = []
                failed_requests = []
                bad_responses = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda message: console_issues.append({
                    "text": message.text, "url": message.location.get("url", "")})
                        if message.type in {"error", "warning"} else None)
                page.on("requestfailed", lambda request: failed_requests.append(
                    {"url": request.url, "failure": request.failure}))
                page.on("response", lambda response: bad_responses.append({
                    "url": response.url, "status": response.status})
                    if response.status >= 400 and not _expected_principal_denial(
                        response.url, response.status) else None)
                page.add_init_script("localStorage.clear();")
                page.goto(address.rsplit("/", 1)[0] + "/")
                try:
                    page.wait_for_selector("body:not(.app-booting)")
                except Exception as exc:
                    raise AssertionError({
                        "width": width, "page_errors": errors, "console": console_issues,
                        "requests": failed_requests, "responses": bad_responses,
                        "body_class": page.locator("body").get_attribute("class"),
                    }) from exc
                assert page.url == address.rsplit("/", 1)[0] + "/"
                assert page.title() == "MARVIS-全能风控智能体"
                assert page.locator("#workspaceWelcome").is_visible()
                page.locator("#sidebarSettings > summary").click()
                page.locator('[data-ui-locale-option="en"]').click()
                page.wait_for_function("document.documentElement.lang === 'en'")
                assert page.locator('.task-row-name').text_content() == "用户原始名称 · v1"
                page.locator('[data-ui-locale-option="zh"]').click()
                page.wait_for_function("document.documentElement.lang === 'zh-CN'")
                page.locator("#sidebarSettings > summary").click()
                shell = page.locator('[data-task-id="preview-task"]').first
                shell.hover()
                page.wait_for_function("!document.getElementById('taskRowPreview').hidden")
                assert "用户原始名称" in page.locator("#taskRowPreview").text_content()
                shell.dispatch_event("pointerdown")
                page.wait_for_function("document.getElementById('taskRowPreview').hidden")
                page.locator("#welcomeModelValidationCard").click()
                assert page.locator("#homeComposerInput").input_value()
                assert not page.locator("#taskDialog").evaluate("node => node.open")
                assert errors == []
                assert not [issue for issue in console_issues if not (
                    _expected_principal_denial(issue["url"], 403)
                    and "403 (Forbidden)" in issue["text"]
                )], console_issues
                assert failed_requests == []
                assert bad_responses == []
                screenshot_dir = Path(os.environ.get("MARVIS_FRONTEND_QA_DIR", str(tmp_path)))
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_dir / f"f13-app-bindings-{width}.png"))
                page.close()
        finally:
            browser.close()


def test_parallel_app_boot_serves_every_module_without_connection_resets():
    playwright = pytest.importorskip("playwright.async_api")

    async def exercise(address):
        async with playwright.async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                async def boot(index):
                    context = await browser.new_context(viewport={"width": 1440, "height": 1000})
                    page = await context.new_page()
                    issues = {"page": index, "page_errors": [], "requests": [], "responses": []}
                    page.on("pageerror", lambda error: issues["page_errors"].append(str(error)))
                    page.on("requestfailed", lambda request: issues["requests"].append(
                        {"url": request.url, "failure": request.failure}))
                    page.on("response", lambda response: issues["responses"].append(
                        {"url": response.url, "status": response.status})
                        if response.status >= 400 and not _expected_principal_denial(
                            response.url, response.status) else None)
                    try:
                        await page.goto(address.rsplit("/", 1)[0] + "/")
                        await page.wait_for_selector("body:not(.app-booting)")
                        assert await page.locator("#workspaceWelcome").is_visible()
                        assert issues["page_errors"] == [], issues
                        assert issues["requests"] == [], issues
                        assert issues["responses"] == [], issues
                    except Exception as exc:
                        raise AssertionError(issues) from exc
                    finally:
                        await context.close()

                outcomes = await asyncio.gather(*(boot(index) for index in range(6)),
                                                return_exceptions=True)
                failures = [str(outcome) for outcome in outcomes if isinstance(outcome, Exception)]
                assert not failures, json.dumps(failures, ensure_ascii=False)
            finally:
                await browser.close()

    with _serve_smoke_page(_smoke_html()) as address:
        asyncio.run(exercise(address))
