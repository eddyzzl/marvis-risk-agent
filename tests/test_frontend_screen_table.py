"""Static checks for the §4 interactive feature-screening selection table (FEAT-1).

The table is a thin consumer of the backend ``metadata.screen`` contract: it renders
the screened features with metric columns + checkboxes (pre-checked = the screen's
proposed set) and, on confirm, posts ``{content:"确认", selection:[...]}`` so the
backend overrides the screen step's selected set. These assertions pin the wiring so
a frontend rewrite can't silently drop it. (Kept in its own file, away from the
brand-treatment tests, to stay decoupled from unrelated frontend churn.)
"""

from __future__ import annotations

from pathlib import Path

_STATIC = Path(__file__).resolve().parent.parent / "marvis" / "static"


def _read(rel: str) -> str:
    return (_STATIC / rel).read_text(encoding="utf-8")


def test_screen_table_renderer_and_manual_branch_are_wired():
    app_js = _read("app.js")
    # the interactive renderer exists and is dispatched for screen gate messages
    assert "function agentMessageScreenTableHtml(message)" in app_js
    assert "if (meta.screen)" in app_js
    assert "agentMessageScreenTableHtml(message)" in app_js
    # it reads the structured screen payload the backend attaches
    assert "message?.metadata?.screen" in app_js
    # checkbox per feature, pre-checked from the proposed selected set
    assert 'class="screen-pick"' in app_js
    assert "screen.selected" in app_js


def test_screen_confirm_posts_edited_selection():
    app_js = _read("app.js")
    assert "function submitScreenSelection(button)" in app_js
    assert "data-screen-confirm" in app_js
    # collects checked, non-disabled features and posts them as `selection`
    assert ".screen-pick:checked" in app_js
    assert '"content": "确认"' in app_js or 'content: "确认"' in app_js
    assert "selection" in app_js
    # a delegated document click handler drives it (mirrors the C1 form pattern)
    assert "handleScreenConfirmClick" in app_js


def test_screen_table_has_hardcut_coloring_styles():
    css = _read("css/v2-workbench.css")
    assert ".screen-table" in css
    # hard-cut buckets are visually distinguished (leakage / suspected / unusable)
    assert ".screen-row.screen-leakage" in css
    assert ".screen-row.screen-suspected" in css
    assert ".screen-badge" in css


def test_dedup_picker_renderer_and_branch_are_wired():
    app_js = _read("app.js")
    assert "function agentMessageDedupPickerHtml(message)" in app_js
    assert "if (meta.dedup)" in app_js
    assert "message?.metadata?.dedup" in app_js
    # a first/last strategy <select> per conflicting feature
    assert 'class="dedup-strategy"' in app_js
    assert "data-dedup-feature" in app_js


def test_dedup_picker_posts_strategies():
    app_js = _read("app.js")
    assert "function submitDedupStrategies(button)" in app_js
    assert "data-dedup-confirm" in app_js
    assert "dedup_strategies" in app_js
    assert "handleDedupConfirmClick" in app_js
    css = _read("css/v2-workbench.css")
    assert ".dedup-picker" in css and ".dedup-table" in css


def test_capability_tier_picker_is_wired():
    """TIER-IA (spec §5.1): the create dialog exposes a per-task capability-tier
    selector (the previously-missing entry point), collected into payload."""
    index_html = _read("index.html")
    app_js = _read("app.js")
    assert 'id="createTaskTier"' in index_html
    for tier in ("conservative", "balanced", "aggressive"):
        assert f'value="{tier}"' in index_html
    assert "createTaskTierField" in app_js
    assert "tierField" in app_js  # gated to driver task types
    assert "payload.capability_tier" in app_js
