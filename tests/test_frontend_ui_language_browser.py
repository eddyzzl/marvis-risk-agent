"""Rendered locale boundaries for fixed widgets, raw data and Agent prose."""

from __future__ import annotations

import json
import os
import re

import pytest

from marvis.ui_locale import localize_payload
from tests.test_frontend_playwright_smoke import (
    _real_modeling_plan,
    _real_modeling_task,
    _serve_smoke_page,
    _smoke_html,
)


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
        reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 to run browser locale tests.",
    ),
]


def _set_locale(page, locale):
    settings = page.locator("#sidebarSettings")
    if not settings.evaluate("element => element.open"):
        page.locator("#sidebarSettings summary").click()
    page.locator(f'[data-ui-locale-option="{locale}"]').click()
    if settings.evaluate("element => element.open"):
        page.locator("#sidebarSettings summary").click()


def _read_tasks_through_app_client(page):
    # Use the exact module instance selected by the app's real import map.
    # This exercises the two application headers instead of manually faking
    # them with page.request or fetch options.
    page.evaluate(
        """async () => {
          const { api } = await import('/static/js/api.js');
          await api('api/tasks');
        }""",
    )


@pytest.mark.parametrize("status,workflow_version", [
    ("created", 2), ("scanned", 2), ("running", 2), ("executed", 2),
    ("computing_metrics", 2), ("writing_artifacts", 2), ("succeeded", 2),
    ("review_required", 2), ("succeeded", 1),
])
def test_task_detail_system_status_round_trips_without_translating_task_name(status, workflow_version):
    playwright = pytest.importorskip("playwright.sync_api")
    task = {
        **_real_modeling_task(), "task_type": "validation", "status": status,
        "validation_workflow_version": workflow_version, "run_mode": "manual",
        "model_name": "用户自己的中文任务", "source_dir": "/fixture/用户材料",
    }
    with _serve_smoke_page(_smoke_html(), tasks=[task]) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(5000)
            page.add_init_script(
                "localStorage.clear();localStorage.setItem('marvis_ui_locale','en');"
                f"localStorage.setItem('marvis_selected_task_id',{json.dumps(task['id'])});",
            )
            page.goto(address.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            page.locator("#taskHeroToggle").click()
            detail = page.locator("#actionErrorDetail")
            detail.wait_for(state="visible")
            page.wait_for_timeout(50)
            english = detail.inner_text()
            assert english and not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", english), english
            assert "用户自己的中文任务" in page.locator("#currentTaskTitle").inner_text()
            assert "/fixture/用户材料" in page.locator("#taskSnapshot").inner_text()
            _set_locale(page, "zh")
            page.wait_for_function("() => /[\\u3400-\\u4dbf\\u4e00-\\u9fff]/.test(document.querySelector('#actionErrorDetail').textContent)")
            if status == "succeeded":
                page.wait_for_function("() => document.querySelector('#taskList .task-row')?.getAttribute('aria-label').includes('已完成')")
            _set_locale(page, "en")
            page.wait_for_function(
                "expected => document.querySelector('#actionErrorDetail').textContent === expected",
                arg=english,
            )
            if not detail.is_visible():
                page.locator("#taskHeroToggle").click()
            assert detail.inner_text() == english
            if status == "succeeded":
                assert "Done" in page.locator("#taskList .task-row").first.get_attribute("aria-label")
            assert "用户自己的中文任务" in page.locator("#currentTaskTitle").inner_text()
        finally:
            browser.close()


@pytest.mark.parametrize("initial_locale", ["zh", "en"])
def test_fixed_widgets_round_trip_while_agent_prose_and_raw_values_stay(initial_locale):
    playwright = pytest.importorskip("playwright.sync_api")
    task = {**_real_modeling_task(), "run_mode": "agent", "model_name": "Locale boundary task"}
    messages = [{
        "id": "locale-fixed-widget",
        "task_id": task["id"],
        "role": "assistant",
        "stage": "chat",
        "content": "**建议**：继续。",
        "created_at": "2026-06-30T00:01:00Z",
        "metadata": {
            "kind": "result",
            "phase": "特征",
            "step_title": "筛选特征",
            "run_seq": 2,
            "memory_references": [{
                "id": "通过", "source_task_id": "raw-task", "confidence": 0.86,
                "support_count": 4, "use_reason": "验证完成。",
            }],
            "tables": [{
                "title": "材料识别",
                "columns": ["特征", "状态", "文件名", "IV"],
                "rows": [["通过", "推荐", "数据字典.csv", 0.31]],
            }],
        },
    }]
    task_reads = []
    message_reads = []
    with _serve_smoke_page(_smoke_html(), tasks=[task]) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(8000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.add_init_script(
                "localStorage.clear();"
                f"localStorage.setItem('marvis_selected_task_id', {json.dumps(task['id'])});"
                f"localStorage.setItem('marvis_ui_locale', {json.dumps(initial_locale)});",
            )

            def task_route(route):
                task_reads.append(route.request.all_headers())
                route.fulfill(json=[task])

            def message_route(route):
                headers = route.request.all_headers()
                message_reads.append(headers)
                payload = {"messages": messages, "incremental": False}
                route.fulfill(json=payload if headers.get("x-marvis-ui-canonical") == "1" else localize_payload(
                    payload,
                    locale=headers.get("x-marvis-ui-locale", "zh"),
                ))

            def plan_route(route):
                headers = route.request.all_headers()
                payload = {"plans": [_real_modeling_plan()]}
                route.fulfill(json=payload if headers.get("x-marvis-ui-canonical") == "1" else localize_payload(
                    payload,
                    locale=headers.get("x-marvis-ui-locale", "zh"),
                ))

            page.route("**/api/tasks", task_route)
            page.route(f"**/api/tasks/{task['id']}/agent/messages*", message_route)
            page.route(f"**/api/tasks/{task['id']}/plans*", plan_route)
            page.goto(address.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            article = page.locator('[data-agent-message-id="locale-fixed-widget"]')
            page.wait_for_selector('[data-agent-message-id="locale-fixed-widget"] .workflow-data-widget')

            def assert_locale(locale):
                expected_title = "Material recognition" if locale == "en" else "材料识别"
                expected_status = "Recommended" if locale == "en" else "推荐"
                expected_columns = ["Feature", "Status", "File name", "IV"] if locale == "en" else ["特征", "状态", "文件名", "IV"]
                page.wait_for_function(
                    """expected => {
                      const article = document.querySelector('[data-agent-message-id="locale-fixed-widget"]');
                      return article?.querySelector('.workflow-widget-title strong')?.textContent === expected;
                    }""",
                    arg=expected_title,
                )
                assert article.locator(".workflow-widget-title strong").inner_text() == expected_title
                assert article.locator("th").all_text_contents() == expected_columns
                cells = article.locator("tbody tr").first.locator("td")
                assert cells.nth(0).inner_text() == "通过"
                assert cells.nth(1).inner_text() == expected_status
                assert cells.nth(2).inner_text() == "数据字典.csv"
                assert cells.nth(3).inner_text() == "0.31"
                assert article.locator(".agent-message-content").inner_text() == "建议：继续。"
                expected_meta = "Feature · Screen features · Round 2" if locale == "en" else "特征 · 筛选特征 · 第 2 轮"
                meta = article.locator(".agent-message-meta")
                assert meta.text_content() == expected_meta
                assert meta.locator("span").evaluate_all(
                    "nodes => new Set(nodes.map(node => Math.round(node.getBoundingClientRect().top))).size === 1",
                )
                references = article.locator(".agent-memory-references")
                if not references.evaluate("element => element.open"):
                    references.locator("summary").click()
                memory = article.locator(".agent-memory-reference")
                assert memory.locator("strong").inner_text() == "通过"
                assert memory.locator(".agent-memory-reference-main > span").inner_text() == "验证完成。"
                expected_support = "Supports 4" if locale == "en" else "支持 4"
                assert expected_support in memory.locator("small").inner_text()

            assert_locale(initial_locale)
            current_locale = initial_locale
            for locale in ("zh", "en", "zh"):
                if locale == current_locale:
                    _set_locale(page, locale)
                else:
                    with page.expect_response(
                        lambda response: "/agent/messages" in response.url
                        and response.request.all_headers().get("x-marvis-ui-locale") == locale,
                    ):
                        _set_locale(page, locale)
                assert_locale(locale)
                assert any(read.get("x-marvis-ui-locale") == locale for read in message_reads)
                current_locale = locale

            _set_locale(page, "en")
            assert_locale("en")
            page.locator("#agentComposerInput").fill("请解释这个结果")
            _read_tasks_through_app_client(page)
            assert task_reads[-1]["x-marvis-ui-locale"] == "en"
            assert task_reads[-1]["x-marvis-reply-language"] == "zh"
            assert task_reads[-1]["x-marvis-ui-canonical"] == "1"
            _set_locale(page, "zh")
            assert_locale("zh")
            page.locator("#agentComposerInput").fill("Please explain this result")
            _read_tasks_through_app_client(page)
            assert task_reads[-1]["x-marvis-ui-locale"] == "zh"
            assert task_reads[-1]["x-marvis-reply-language"] == "en"
            assert task_reads[-1]["x-marvis-ui-canonical"] == "1"
            assert not errors
        finally:
            browser.close()


@pytest.mark.parametrize("status", ["created", "review_required"])
def test_all_validation_stage_tasks_start_open_and_keep_manual_collapse(status):
    playwright = pytest.importorskip("playwright.sync_api")
    task = {
        **_real_modeling_task(),
        "id": "validation-locale-boundary",
        "task_type": "validation",
        "validation_workflow_version": 2,
        "model_name": "Validation phase visibility",
        "run_mode": "manual",
        "status": status,
        "report_available": False,
    }
    with _serve_smoke_page(_smoke_html(), tasks=[task]) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(8000)
            page.add_init_script(
                "localStorage.clear();"
                f"localStorage.setItem('marvis_selected_task_id', {json.dumps(task['id'])});",
            )
            page.goto(address.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            groups = page.locator("#workflowStepper details[data-validation-stage-expansion]")
            assert groups.count() == 4
            assert groups.evaluate_all("elements => elements.every(element => element.open)")
            for group in groups.all():
                assert group.is_visible()
                assert group.locator(".notebook-step").first.is_visible()
            first_key = groups.first.get_attribute("data-validation-stage-expansion")
            groups.first.locator("summary").click()
            assert not groups.first.evaluate("element => element.open")
            # A task-state update plus locale refresh forces real stepper
            # reconstruction, rather than merely inspecting the same node.
            task["status"] = "scanned" if status == "created" else "succeeded"
            _set_locale(page, "en")
            page.wait_for_function(
                """() => document.querySelector('#workflowStepper details > summary')?.textContent.includes('Stage tasks')""",
            )
            restored = page.locator(f'[data-validation-stage-expansion="{first_key}"]')
            assert not restored.evaluate("element => element.open")
            assert groups.evaluate_all("elements => elements.slice(1).every(element => element.open)")
            page.reload()
            page.wait_for_selector("body:not(.app-booting)")
            assert groups.count() == 4
            assert groups.evaluate_all("elements => elements.every(element => element.open)")
        finally:
            browser.close()
