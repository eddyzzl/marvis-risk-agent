"""Composed platform labels translate and restore independently of raw names."""

from __future__ import annotations

import os

import pytest

from tests.test_frontend_playwright_smoke import _serve_smoke_page, _smoke_html


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
        reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 to run browser locale tests.",
    ),
]


def test_composed_scope_readiness_and_material_summaries_round_trip():
    playwright = pytest.importorskip("playwright.sync_api")
    with _serve_smoke_page(_smoke_html()) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(6000)
            page.add_init_script("localStorage.clear();localStorage.setItem('marvis_ui_locale','en');")
            page.goto(address.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            page.evaluate("""async () => {
              const { renderValidationBatchSwitcher } = await import('/static/js/validation-batch.js');
              const { renderModelDeliveryPanel } = await import('/static/js/v2/model_delivery_panel.js');
              const { createCreateTaskDialogController } = await import('/static/js/create-task-dialog.js');
              const fixture = document.createElement('section');
              fixture.id = 'composed-locale-fixture';
              fixture.innerHTML = renderValidationBatchSwitcher({
                batch: { parent_task_id: 'fixture-parent', status: 'running', item_count: 5 },
                items: [
                  { child_task_id: 'a', status: 'writing_artifacts', pending_report_draft: true },
                  { child_task_id: 'b', status: 'writing_artifacts', pending_report_draft: true },
                  { child_task_id: 'c', status: 'failed' },
                  { child_task_id: 'd', status: 'cancelled' },
                  { child_task_id: 'e', status: 'created' },
                ],
              }) + renderModelDeliveryPanel({ metadata: { model_delivery: {
                readiness: [
                  { id: 'a', label: '原生模型', status: 'unknown' },
                  { id: 'b', label: '模型卡', status: 'unknown' },
                  { id: 'c', label: '模型报告', status: 'failed' },
                ],
              } } });
              document.body.appendChild(fixture);
              const controller = createCreateTaskDialogController({
                $: id => document.getElementById(id),
                materialSourceController: { reset() {}, mode: () => 'folder', selectedFiles: () => [] },
                getSelectedTier: () => 'high', selectedTierStorageKey: 'composed-fixture-tier',
              });
              controller.prefillTaskDialog({ taskType: 'validation', name: '通过' });
            }""")
            expected = {
                "en": [
                    "2 to confirm · 1 failed · 1 cancelled · 1 not ready",
                    "2 evidence items unknown；1 items need action / unsupported；Business acceptance is not confirmed here",
                    "通过 · Materials not chosen",
                    "1 models · 1 materials to choose",
                ],
                "zh": [
                    "2 份待确认 · 1 个失败 · 1 个已取消 · 1 个未就绪",
                    "2 项证据未知；1 项需处理/不支持；业务验收未在此确认",
                    "通过 · 待选择材料",
                    "1 个模型 · 1 个待选择材料",
                ],
            }
            for locale in ("en", "zh", "en", "zh"):
                page.evaluate("""async locale => {
                  const { setUiLocale } = await import('/static/js/i18n.js');
                  setUiLocale(locale);
                }""", locale)
                actual = page.evaluate("""async () => {
                  await new Promise(requestAnimationFrame);
                  const selectors = [
                    '#composed-locale-fixture .validation-batch-confirm-scope',
                    '#composed-locale-fixture .model-delivery-head small',
                    '#primaryModelSummary', '#createModelCount',
                  ];
                  return selectors.map(selector => document.querySelector(selector)?.textContent);
                }""")
                assert actual == expected[locale]
        finally:
            browser.close()
