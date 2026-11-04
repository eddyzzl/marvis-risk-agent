"""Real workflow diagnostics localize their UI while preserving raw context."""

from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess

import pytest

from marvis.agent.workflow_error_diagnostics import (
    build_workflow_error_diagnostic,
    enrich_workflow_error_diagnostic,
)
from marvis.agent.workflow_recovery import (
    _diagnostic_from_failure_envelope,
    _legacy_csv_diagnostic,
)
from marvis.data.errors import CsvParseError, DataIngestError
from marvis.recovery import _restart_failure_payload
from tests.test_frontend_playwright_smoke import _serve_smoke_page, _smoke_html


ROOT = Path(__file__).resolve().parents[1]


def _diagnostics(filename="fixture.csv"):
    cases = [
        build_workflow_error_diagnostic(workflow="modeling", exc=exc, setup_error=setup)
        for exc, setup in [
            (FileNotFoundError(2, "No such file", filename), False),
            (DataIngestError("Unable to read file"), False),
            (ValueError("Missing target column"), True),
            (RuntimeError("Failed safely"), False),
            (TypeError("Object of type Timestamp is not JSON serializable"), False),
            (CsvParseError(path=Path(filename), technical_message="parse failure",
                           line_number=10, expected_fields=7, actual_fields=8), False),
            (CsvParseError(path=Path(filename), technical_message="parse failure"), False),
        ]
    ]
    cases.extend([
        _diagnostic_from_failure_envelope("modeling", {
            "suggested_actions": ["retry", "adjust", "replan", "halt"],
        }),
        _legacy_csv_diagnostic("modeling", "Expected 7 fields in line 10, saw 8"),
        enrich_workflow_error_diagnostic({
            "code": "workflow_execution_failed", "workflow": "modeling",
            "technical_detail": "No match for FieldRef.Name(C0)",
        }),
    ])
    step = SimpleNamespace(title="切分样本", id="fixture-step", error="ServerRestart")
    for reconciliation in (None, "Needs reconciliation"):
        cases.append(_restart_failure_payload(
            plan_id="fixture-plan", workflow="modeling", failed_step=step,
            reconciliation_error=reconciliation,
        )[0])
    assert all(isinstance(item, dict) for item in cases)
    return cases


def _ui_fields(diagnostic):
    fields = [diagnostic.get(key, "") for key in (
        "title", "summary", "cause", "impact", "location", "agent_prompt",
    )]
    fields.extend(diagnostic.get("actions", []))
    for item in diagnostic.get("recovery_actions", []):
        fields.extend(item.get(key, "") for key in ("label", "consequence", "effect"))
    for item in diagnostic.get("evidence", []):
        fields.extend([item.get("label", ""), item.get("value", "")])
    return [str(value) for value in fields if value]


def test_real_diagnostic_fields_have_english_translations_and_keep_raw_file_names():
    diagnostics = _diagnostics()
    fields = [field for diagnostic in diagnostics for field in _ui_fields(diagnostic)]
    script = '''
import assert from "node:assert/strict";
import { translateText } from "./marvis/static/js/i18n.js";
const fields = FIELDS;
const left = fields.filter(value => /[\u3400-\u4dbf\u4e00-\u9fff]/.test(value))
  .map(source => ({source, text: translateText(source)}))
  .filter(item => !item.text || /[\u3400-\u4dbf\u4e00-\u9fff]/.test(item.text));
assert.deepEqual(left, []);
const source = "找不到需要读取的材料文件 `通过.csv`。";
assert.equal(translateText(source), "The required material file `通过.csv` could not be found.");
assert.equal(translateText("通过.csv · 第 10 行"), "通过.csv · Line 10");
'''.replace("FIELDS", json.dumps(fields, ensure_ascii=False))
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.e2e
@pytest.mark.skipif(os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
                    reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 for browser checks.")
def test_rendered_error_cards_round_trip_and_leave_agent_prose_and_technical_detail_raw():
    playwright = pytest.importorskip("playwright.sync_api")
    diagnostics = _diagnostics("通过.csv")
    diagnostics[0]["technical_detail"] = "verbatim 中文技术说明"
    with _serve_smoke_page(_smoke_html()) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(6000)
            page.add_init_script("localStorage.clear();localStorage.setItem('marvis_ui_locale','en');")
            page.goto(address.rsplit("/", 1)[0] + "/")
            page.wait_for_selector("body:not(.app-booting)")
            page.evaluate("""async diagnostics => {
              const {workflowErrorDiagnosticHtml} = await import('/static/js/v2/workflow_error_card.js');
              const fixture = document.createElement('div');
              fixture.id = 'diagnostic-locale-fixture';
              fixture.innerHTML = diagnostics.map(workflowErrorDiagnosticHtml).join('')
                + '<div class="agent-message-content"><p>验证完成。</p></div>';
              document.body.appendChild(fixture);
            }""", diagnostics)
            for locale in ("en", "zh", "en"):
                page.evaluate("""async locale => {
                  const {setUiLocale} = await import('/static/js/i18n.js');
                  setUiLocale(locale);
                }""", locale)
                if locale == "en":
                    page.wait_for_function("""() => {
                      const root = document.querySelector('#diagnostic-locale-fixture');
                      return [...root.querySelectorAll('.workflow-error-card')].every(card => {
                        const clone = card.cloneNode(true);
                        clone.querySelectorAll('pre,code').forEach(node=>node.remove());
                        return !/[\u3400-\u4dbf\u4e00-\u9fff]/.test(clone.textContent.replaceAll('通过.csv',''));
                      });
                    }""")
                    assert page.locator("#diagnostic-locale-fixture h3").first.text_content() == "Model development not started"
                else:
                    assert page.locator("#diagnostic-locale-fixture h3").first.text_content() == "模型开发未开始"
                assert "通过.csv" in page.locator(".workflow-error-card__summary").first.text_content()
                assert page.locator("#diagnostic-locale-fixture pre code").first.text_content() == "verbatim 中文技术说明"
                assert page.locator("#diagnostic-locale-fixture .agent-message-content p").text_content() == "验证完成。"
        finally:
            browser.close()
