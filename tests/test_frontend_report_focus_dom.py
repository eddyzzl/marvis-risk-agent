"""Report-card relocation preserves real DOM focus and caret in Chromium."""
import os
from pathlib import Path

import pytest

from tests.javascript_source import slice_function

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(
    os.environ.get('MARVIS_RUN_PLAYWRIGHT_SMOKE') != '1',
    reason='Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 for real DOM behavior.')]


@pytest.mark.parametrize('fallback', [False, True])
def test_report_card_relocation_preserves_field_selection_and_editing_row(fallback):
    playwright = pytest.importorskip('playwright.sync_api')
    app = (Path(__file__).resolve().parents[1] / 'marvis/static/app.js').read_text()
    functions = '\n'.join(slice_function(app, f'function {name}(') for name in (
        'rememberReportCardFocus', 'moveReportCard', 'restoreReportCardFocus',
        'parkReportDraftCard', 'placeReportDraftCard'))
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={'width': 1600, 'height': 1000})
            page.set_content('''<main id="resultWorkspace"><section id="resultScrollContent">
              <div data-report-draft-slot><div id="reportDraftWorkspace">
              <div class="report-card-row is-editing" data-report-card-row="TEXT:summary">
              <textarea data-report-draft-key="TEXT:summary">用户填写的结论原文</textarea>
              </div></div></div></section></main>''')
            result = page.evaluate("""(fallback) => {
              if (fallback) Element.prototype.moveBefore = undefined;
              const $ = id => document.getElementById(id);
              let reportCardFocusMemo = null, reportCardRelocating = false;
              const taskSession = {panels:{reportDraft:{expanded:false}}};
            """ + functions + """
              const panel=$('reportDraftWorkspace');
              const input=panel.querySelector('textarea');
              let nativeMoves=0;
              if (!fallback) {
                const native=Element.prototype.moveBefore;
                if (typeof native !== 'function') throw Error('Native moveBefore unavailable');
                Element.prototype.moveBefore=function(...args){nativeMoves++;return native.apply(this,args);};
              }
              input.focus();input.setSelectionRange(2,6);
              parkReportDraftCard();
              const parked=panel.parentElement.id==='resultWorkspace';
              placeReportDraftCard();
              return {parked,slotted:panel.parentElement.hasAttribute('data-report-draft-slot'),
                focused:document.activeElement===input,selection:[input.selectionStart,input.selectionEnd],
                editing:input.closest('.report-card-row').classList.contains('is-editing'),
                value:input.value,nativeMoves,relocating:reportCardRelocating};
            }""", fallback)
            assert result['parked'] and result['slotted'] and result['focused'] and result['editing']
            assert result['selection'] == [2, 6]
            assert result['value'] == '用户填写的结论原文'
            assert result['relocating'] is False
            assert result['nativeMoves'] == (0 if fallback else 2)
        finally:
            browser.close()
