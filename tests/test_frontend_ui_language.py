"""D-10 (batch 1b): UI language switch — i18n.js runtime, dictionary and static coverage."""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

PRELUDE = r'''
import assert from "node:assert/strict";
import * as i18n from "./marvis/static/js/i18n.js";
import { EN_UI } from "./marvis/static/js/i18n-en.js";
'''


def _node(body: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", PRELUDE + body],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_locale_persistence_defaults_to_chinese_and_survives_broken_storage():
    _node(r'''
assert.equal(typeof i18n.readStoredUiLocale, "function");
const store = {data: {}, getItem(k) { return this.data[k] ?? null; }, setItem(k, v) { this.data[k] = String(v); }};
assert.equal(i18n.readStoredUiLocale(store), "zh");
assert.equal(i18n.writeStoredUiLocale("en", store), true);
assert.equal(store.data[i18n.UI_LOCALE_STORAGE_KEY], "en");
assert.equal(i18n.readStoredUiLocale(store), "en");
store.data[i18n.UI_LOCALE_STORAGE_KEY] = "klingon";
assert.equal(i18n.readStoredUiLocale(store), "zh");
const broken = {getItem() { throw new Error("denied"); }, setItem() { throw new Error("denied"); }};
assert.equal(i18n.readStoredUiLocale(broken), "zh");
assert.equal(i18n.writeStoredUiLocale("en", broken), false);
assert.equal(i18n.readStoredUiLocale(null), "zh");
assert.equal(i18n.normalizeUiLocale("en-GB"), "en");
assert.equal(i18n.normalizeUiLocale(undefined), "zh");
''')


def test_t_translates_interpolates_and_falls_back_to_chinese_with_missing_collection():
    _node(r'''
assert.equal(typeof i18n.t, "function");
i18n.__setI18nStateForTests({
  locale: "zh",
  entries: {"新建任务": "New task", "第 {n} 步": "Step {n}", "状态：{s|t}": "Status: {s|t}", "运行中": "Running"},
});
assert.equal(i18n.t("新建任务"), "新建任务");
assert.equal(i18n.t("第 {n} 步", {n: 3}), "第 3 步");
i18n.__setI18nStateForTests({locale: "en"});
assert.equal(i18n.getUiLocale(), "en");
assert.equal(i18n.t("新建任务"), "New task");
assert.equal(i18n.t("第 {n} 步", {n: 3}), "Step 3");
assert.equal(i18n.t("状态：{s|t}", {s: "运行中"}), "Status: Running");
assert.equal(i18n.t("一个没有译文的句子"), "一个没有译文的句子");
assert.deepEqual(i18n.uiMissingStrings(), ["一个没有译文的句子"]);
assert.equal(i18n.t("plain English"), "plain English");
''')


def test_translate_text_keeps_whitespace_raw_captures_and_rejects_unknown_text():
    _node(r'''
i18n.__setI18nStateForTests({
  locale: "en",
  entries: {"文件路径": "File path", "删除任务「{0}」？": "Delete task “{0}”?", "状态：{0|t}": "Status: {0|t}", "失败": "Failed"},
});
assert.equal(i18n.translateText("\n    文件路径\n  "), "\n    File path\n  ");
// Raw capture: a task name that contains a platform word is never rewritten.
assert.equal(i18n.translateText("删除任务「失败的模型」？"), "Delete task “失败的模型”?");
assert.equal(i18n.translateText("状态：失败"), "Status: Failed");
assert.equal(i18n.translateText("年龄"), null);
assert.equal(i18n.translateText("KS 0.31"), null);
assert.equal(i18n.localizeText("文件路径"), "File path");
assert.equal(i18n.localizeText("年龄"), "年龄");
i18n.__setI18nStateForTests({locale: "zh"});
assert.equal(i18n.localizeText("文件路径"), "文件路径");
''')


def test_reply_language_follows_what_the_user_typed_not_the_toggle():
    _node(r'''
i18n.__setI18nStateForTests({locale: "en"});
assert.equal(i18n.detectTextLanguage("帮我看下 KS"), "zh");
assert.equal(i18n.detectTextLanguage("check PSI"), "en");
assert.equal(i18n.detectTextLanguage("0.30"), "");
// Unknown before typing: no hint header, so the backend reads the body.
assert.deepEqual(i18n.uiRequestHeaders(), {"X-MARVIS-UI-Locale": "en"});
i18n.rememberTypedLanguage("用这份样本建模");
assert.deepEqual(i18n.uiRequestHeaders(), {"X-MARVIS-UI-Locale": "en", "X-MARVIS-Reply-Language": "zh"});
i18n.rememberTypedLanguage("42");
assert.equal(i18n.replyLanguage(), "zh");
''')


def test_reply_hint_is_seeded_per_task_and_quick_reply_sends_keep_it():
    _node(r'''
i18n.__setI18nStateForTests({locale: "en"});
i18n.rememberTypedLanguage("用这份样本建模");
i18n.setQuickReplyCommands(["确认", "继续"]);
// Switching to a task whose latest typed message is English re-seeds the hint;
// a later chip command ("确认") is a platform phrase and does not count.
const englishTask = [
  {role: "user", content: "Build a scorecard"},
  {role: "assistant", content: "x"},
  {role: "user", content: "确认"},
];
assert.equal(i18n.seedReplyLanguage(englishTask), "en");
assert.equal(i18n.uiRequestHeaders()["X-MARVIS-Reply-Language"], "en");
// A Chinese task switches it back; a task with no typed text clears it.
assert.equal(i18n.seedReplyLanguage([{role: "user", content: "帮我看下 KS"}]), "zh");
assert.equal(i18n.seedReplyLanguage([{role: "assistant", content: "hi"}]), "");
assert.equal("X-MARVIS-Reply-Language" in i18n.uiRequestHeaders(), false);
// A send that is exactly a chip command is marked so the backend keeps the hint.
i18n.seedReplyLanguage(englishTask);
const chip = i18n.uiRequestHeaders(JSON.stringify({content: " 确认 "}));
assert.equal(chip["X-MARVIS-Reply-Source"], "quick-reply");
assert.equal(chip["X-MARVIS-Reply-Language"], "en");
const typed = i18n.uiRequestHeaders(JSON.stringify({content: "确认一下这个口径"}));
assert.equal("X-MARVIS-Reply-Source" in typed, false);
assert.equal("X-MARVIS-Reply-Source" in i18n.uiRequestHeaders("not json"), false);
''')


def test_dictionary_values_are_english_and_keep_every_placeholder():
    raw = _node('process.stdout.write(JSON.stringify(EN_UI));')
    entries = json.loads(raw)
    assert len(entries) > 5000
    placeholder = re.compile(r"\{([A-Za-z0-9_]+)(?:\|t)?\}")
    for key, value in entries.items():
        assert value.strip(), key
        assert not CJK.search(value), (key, value)
        assert sorted(placeholder.findall(key)) == sorted(placeholder.findall(value)), key


class _StaticText(HTMLParser):
    SKIP_TAGS = {"script", "style", "textarea", "pre", "code", "kbd", "iframe"}
    VOID = {"br", "img", "input", "meta", "link", "hr", "source", "path", "circle", "rect",
            "ellipse", "line", "polyline", "polygon", "col", "wbr", "area", "base", "embed", "param", "track"}
    ATTRS = {"placeholder", "title", "aria-label", "aria-description", "alt", "data-tooltip",
             "data-panel-title", "data-panel-description"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.texts: list[str] = []
        self.attrs: list[str] = []

    def _skipped(self) -> bool:
        return any(skip for _, skip in self.stack)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        skip = tag in self.SKIP_TAGS or values.get("translate") == "no" or "data-i18n-skip" in values
        if not (self._skipped() or skip) or (tag == "textarea" and not self._skipped()):
            self.attrs.extend(v for k, v in attrs if k in self.ATTRS and v and CJK.search(v))
        if tag not in self.VOID:
            self.stack.append((tag, skip))

    def handle_startendtag(self, tag, attrs):
        if not self._skipped():
            self.attrs.extend(v for k, v in attrs if k in self.ATTRS and v and CJK.search(v))

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop()[0] == tag:
                break

    def handle_data(self, data):
        if not self._skipped() and CJK.search(data):
            self.texts.append(data)


def test_english_mode_leaves_no_chinese_in_static_index_html():
    parser = _StaticText()
    parser.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    samples = parser.texts + parser.attrs
    assert len(samples) > 500
    raw = _node(
        "i18n.__setI18nStateForTests({locale: 'en'});"
        f"const samples = {json.dumps(samples, ensure_ascii=False)};"
        "const left = samples.filter((text) => { const out = i18n.translateText(text) ?? text; "
        "return /[\\u3400-\\u4dbf\\u4e00-\\u9fff]/.test(out); });"
        "process.stdout.write(JSON.stringify(left));"
    )
    assert json.loads(raw) == []


def test_language_switch_lives_inside_settings_for_both_sidebar_states():
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    footer = html[html.index('<footer class="sidebar-footer">'):html.index('</footer>', html.index('<footer class="sidebar-footer">'))]
    assert 'role="radiogroup"' in footer
    assert 'data-ui-locale-option="zh"' in footer and '>中文<' in footer
    assert 'data-ui-locale-option="en"' in footer and '>English<' in footer
    settings = footer[footer.index('<details id="sidebarSettings"'):]
    assert 'data-settings-row="language"' in settings
    assert 'role="radiogroup"' in settings
    assert 'class="ui-locale-switch"' not in footer[:footer.index('<details id="sidebarSettings"')]
    assert 'data-ui-locale-toggle' not in footer
    assert 'href="static/css/ui-language.css' in html
    css = (STATIC / 'css' / 'ui-language.css').read_text(encoding='utf-8')
    assert '.app-shell.sidebar-collapsed .ui-locale-switch' not in css


def test_quick_reply_bubble_shows_the_chip_label_but_keeps_the_command():
    _node(r'''
import { quickReplyBubbleText } from "./marvis/static/js/agent-quick-replies.js";
i18n.__setI18nStateForTests({ locale: "en", entries: EN_UI });
const messages = [
  {role: "assistant", content: "x", metadata: {
    quick_replies: [{label: "确认并开始训练", command: "确认训练"}],
    error_diagnostic: {recovery_actions: [{label: "Retry the current step", command: "重试当前步骤"}]},
  }},
];
const shown = (content) => quickReplyBubbleText({role: "user", content}, messages, i18n.t);
assert.equal(shown("继续"), EN_UI["继续"]);
assert.equal(shown("重试当前步骤"), "Retry the current step");
assert.equal(shown(" 确认训练 "), i18n.t("确认并开始训练"));
// Free text (and a command nobody offered) is never rewritten.
assert.equal(shown("确认训练吧"), "确认训练吧");
assert.equal(shown("帮我看看"), "帮我看看");
// Assistant turns are not touched.
assert.equal(quickReplyBubbleText({role: "assistant", content: "继续"}, messages, i18n.t), "继续");
''')


def test_user_data_controls_and_cells_are_never_translated():
    _node(r'''
import { conversationActionCardHtml } from "./marvis/static/js/conversation-action-cards.js";
import { isRawColumnHeader } from "./marvis/static/js/raw-columns.js";
const message = {id: "m1", role: "assistant", content: "", metadata: {conversation_action: {
  kind: "monitoring_schedule_form",
  form: {dataset_id: "d1", time_col: "状态", advanced: {target_col: "通过"}},
  options: {datasets: [{id: "d1", row_count: 3, column_names: ["状态", "通过"]}]},
}}};
const html = conversationActionCardHtml(message, [message]);
assert.match(html, /<option value="状态"( selected)? translate="no">状态<\/option>/);
assert.match(html, /<option value="通过"( selected)? translate="no">通过<\/option>/);
assert.match(html, /<option value=""[^>]*>请选择<\/option>/);  // the placeholder still translates
for (const header of ["特征", "特征名", "特征名称", "字段名", "验证字段", "Feature Name", "column"]) {
  assert.equal(isRawColumnHeader(header), true, header);
}
assert.equal(isRawColumnHeader("状态"), false);
''')
    screen = (STATIC / "js" / "v2" / "screen_gate_controller.js").read_text(encoding="utf-8")
    assert '<td class="screen-feat" translate="no">' in screen
    assert '<td class="screen-meaning" translate="no" title=' in screen
    i18n_js = (STATIC / "js" / "i18n.js").read_text(encoding="utf-8")
    mutation = i18n_js[i18n_js.index("function handleMutations"):i18n_js.index("function startObserver")]
    assert 'record.attributeName === "value"' in mutation and "VALUE_INPUT_TYPES.has" in mutation
    confirm = (STATIC / "js" / "platform-confirm.js").read_text(encoding="utf-8")
    assert 'if (!PLATFORM_NAME_KEYS.has(item.key)) names.setAttribute("translate", "no");' in confirm
    assert 'new Set(["plans", "experiments"])' in confirm


def test_draft_reply_language_survives_redraw_but_resets_for_new_task_or_message():
    _node(r'''
i18n.__setI18nStateForTests({locale: "en"});
const messages = [{id: "u", role: "user", content: "Build a model"}];
assert.equal(i18n.seedReplyLanguage(messages, {taskId: "a"}), "en");
i18n.rememberTypedLanguage("请解释这个结果");
assert.equal(i18n.seedReplyLanguage([...messages, {id: "answer", role: "assistant", content: "poll"}], {taskId: "a"}), "zh");
assert.equal(i18n.uiRequestHeaders()["X-MARVIS-Reply-Language"], "zh");
assert.equal(i18n.seedReplyLanguage(messages, {taskId: "b"}), "en");
assert.equal(i18n.seedReplyLanguage([], {taskId: "c"}), "");
i18n.setQuickReplyCommands(["确认"]);
i18n.rememberTypedLanguage("Please explain");
assert.equal(i18n.seedReplyLanguage([{id: "quick", role: "user", content: "确认"}], {taskId: "c"}), "en");
assert.equal(i18n.seedReplyLanguage([{id: "new", role: "user", content: "新问题"}], {taskId: "c"}), "zh");
''')
