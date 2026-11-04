"""Rendered conversation forms: changing data cannot keep stale column choices."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

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
        reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 to run browser form tests.",
    ),
]


def _schedule_message(task_id):
    return {
        "id": "schedule-form", "task_id": task_id, "role": "assistant", "stage": "chat",
        "content": "核对后确认创建。", "created_at": "2026-06-30T00:01:00Z",
        "metadata": {"conversation_action": {
            "kind": "monitoring_schedule_form", "missing": [],
            "form": {
                "target_id": "exp-1", "dataset_id": "ds-a", "time_col": "申请日",
                "frequency": "daily", "interval": 1, "start_date": "2027-01-02",
                "notify": "local_inbox", "advanced": {
                    "schedule_id": "用户调度", "timezone": "Asia/Shanghai",
                    "complete_through": "2026-06-30T00:00:00+00:00",
                    "target_col": "bad", "label_maturity_days": 30,
                    "catch_up_budget": 1, "max_attempts": 1, "lease_seconds": 600,
                },
            },
            "options": {
                "targets": [{"id": "exp-1", "kind": "model", "label": "用户模型"}],
                "datasets": [
                    {"id": "ds-a", "row_count": 3,
                     "column_names": ["申请日", "bad", "shared_date"]},
                    {"id": "ds-b", "row_count": 4,
                     "column_names": ["观察日", "label", "shared_date"]},
                    {"id": "ds-empty", "row_count": 0, "column_names": []},
                ],
            },
        }},
    }


def _route_app_fixture(page, task, messages, posts):
    """Stub surrounding services; the real app renders and handles the form."""
    available = {
        "/api/branding", "/api/tasks", "/api/settings/execution-environment/options",
        "/api/settings/llm", "/api/settings/memory-policy",
        *(f"/api/tasks/{task['id']}/{suffix}" for suffix in ("agent/messages", "plans", "evidence", "report-fields")),
    }

    def route(request):
        path = urlparse(request.request.url).path
        if path == "/api/operations/conversation/schedules" and request.request.method == "POST":
            posts.append(request.request.post_data_json)
            messages.append({
                "id": "schedule-created", "task_id": task["id"], "role": "assistant",
                "stage": "chat", "content": "定时监控已创建。", "created_at": "2026-06-30T00:02:00Z",
                "metadata": {"conversation_action": {
                    "kind": "monitoring_schedule_created", "source_message_id": "schedule-form",
                    "schedule_id": "用户调度", "summary": {},
                }},
            })
            request.fulfill(json={"message": messages[-1]})
        elif path in available:
            request.continue_()
        else:
            request.fulfill(json=task if path == f"/api/tasks/{task['id']}" else {})

    page.route("**/api/**", route)


@pytest.mark.parametrize("width,theme,locale", [
    (1440, "light", "zh"), (1600, "light", "en"), (1920, "light", "zh"),
    (1440, "dark", "en"), (1600, "dark", "zh"), (1920, "dark", "en"),
])
def test_monitoring_dataset_change_refreshes_columns_without_losing_edits(width, theme, locale):
    playwright = pytest.importorskip("playwright.sync_api")
    task = {**_real_modeling_task(), "run_mode": "agent", "status": "succeeded"}
    messages = [_schedule_message(task["id"])]
    with _serve_smoke_page(_smoke_html(), tasks=[task], messages_by_task={task["id"]: messages}) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.set_default_timeout(8000)
            errors, posts = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            _route_app_fixture(page, task, messages, posts)
            page.add_init_script(
                "localStorage.clear();"
                f"localStorage.setItem('marvis_selected_task_id',{json.dumps(task['id'])});"
                f"localStorage.setItem('marvis_theme',{json.dumps(theme)});"
                f"localStorage.setItem('marvis_ui_locale',{json.dumps(locale)});",
            )
            page.goto(address.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            assert "MARVIS" in page.title()
            card = page.locator('[data-conversation-action="monitoring_schedule_form"]')
            card.wait_for(state="visible")
            def field(name):
                return card.locator(f'[data-schedule-field="{name}"]')
            field("interval").fill("3")
            field("frequency").select_option("weekly")
            card.locator("details summary").click()
            field("advanced.schedule_id").fill("用户改过的名称")
            field("dataset_id").select_option("ds-b")
            assert field("time_col").locator("option").evaluate_all("nodes => nodes.map(n => n.value)") == ["", "观察日", "label", "shared_date"]
            assert field("time_col").input_value() == ""
            assert field("advanced.target_col").input_value() == ""
            assert field("advanced.label_maturity_days").input_value() == ""
            assert field("interval").input_value() == "3"
            assert field("frequency").input_value() == "weekly"
            assert field("advanced.schedule_id").input_value() == "用户改过的名称"
            assert field("time_col").locator('option[value="观察日"]').inner_text() == "观察日"
            assert not posts  # Editing a draft does not create a schedule.
            field("time_col").select_option("shared_date")
            field("dataset_id").select_option("ds-a")
            assert field("time_col").input_value() == "shared_date"
            field("dataset_id").select_option("ds-empty")
            assert field("time_col").locator("option").count() == 1
            assert field("time_col").input_value() == ""
            field("dataset_id").select_option("ds-b")
            field("time_col").select_option("观察日")
            assert field("time_col").evaluate("node => !node.closest('.conversation-form-field').hasAttribute('data-invalid')")
            assert card.bounding_box()["width"] > 500
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            evidence_dir = os.environ.get("MARVIS_UI_EVIDENCE_DIR")
            if evidence_dir:
                Path(evidence_dir).mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(Path(evidence_dir) / f"monitoring-{width}-{theme}-{locale}.png"))
            card.locator("[data-schedule-conversation-confirm]").click()
            page.wait_for_function("() => document.querySelector('[data-schedule-conversation-confirm]')?.disabled")
            page.locator('[data-conversation-action="monitoring_schedule_created"]').wait_for()
            assert len(posts) == 1
            assert posts[0]["form"]["dataset_id"] == "ds-b"
            assert posts[0]["form"]["time_col"] == "观察日"
            assert posts[0]["form"]["interval"] == 3
            assert posts[0]["form"]["advanced"]["target_col"] is None
            assert field("dataset_id").is_disabled()
            assert not errors, errors
        finally:
            browser.close()


def test_redundant_captions_are_removed_but_creation_and_material_guidance_remain():
    playwright = pytest.importorskip("playwright.sync_api")
    with _serve_smoke_page(_smoke_html()) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(address)
            page.wait_for_selector(".modeling-setup-panel")
            assert page.locator(".modeling-journey-node").count() == 4
            assert page.locator(".modeling-agent-note").count() == 0
            assert page.locator(".modeling-setup-title small").count() == 0
            assert "OOT" in page.locator('[data-modeling-stage="split"]').inner_text()
            page.evaluate("""async () => {
              const { strategyCandidateLabResultsHtml } = await import('/marvis/static/js/v2/strategy_candidate_lab_presenters.js');
              const panel = document.createElement('section');
              panel.id = 'candidate-caption-check';
              panel.innerHTML = strategyCandidateLabResultsHtml({});
              document.querySelector('#root').append(panel);
            }""")
            assert page.locator("#candidate-caption-check .candidate-lab-result-head h4").count() > 0
            assert page.locator("#candidate-caption-check .candidate-lab-result-head p").count() == 0
            page.goto(address.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            page.locator("#welcomeModelDevelopmentCard").click()
            page.locator("#homeComposerTemplate").press("Enter")
            page.locator("#homeDraft [data-home-draft-advanced]").click()
            page.locator("#taskDialog[open]").wait_for()
            assert page.locator("#taskDialogSubtitle").count() == 1
            assert not page.locator("#taskDialogSubtitle").is_visible()
            assert page.locator("#modelName").is_visible()
            page.evaluate("document.querySelector('#taskDialog').close()")
            page.evaluate("""async () => {
              const { createMaterialBindingDialogController } = await import('/static/js/material-binding-dialog.js');
              const controller = createMaterialBindingDialogController({
                $: id => document.getElementById(id), api: async () => ({ candidates: {} })
              });
              controller.bind();
              void controller.ensureMaterialSelection({ id: 'caption-task' }, { force: true });
            }""")
            page.locator("#materialBindingDialog[open]").wait_for()
            assert page.locator(".material-binding-role-text").count() == 4
            assert page.locator(".material-binding-role-text small").count() == 0
            assert page.locator(".material-binding-role-text strong").all_text_contents() == ["Notebook", "样本数据", "PMML 模型", "数据字典"]
            assert page.locator('[data-material-binding-field="dictionary_path"]').get_attribute("aria-label") == "选择数据字典 / 特征元数据"
            assert page.locator("#materialBindingSupplementHint").is_visible()
        finally:
            browser.close()
