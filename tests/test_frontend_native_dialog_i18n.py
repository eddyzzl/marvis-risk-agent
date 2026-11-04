"""Native dialogs use the UI locale and keep names and typed input untouched."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _node(script: str) -> None:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout


PRELUDE = r'''
import assert from "node:assert/strict";
import { __setI18nStateForTests, localizeText, t } from "./marvis/static/js/i18n.js";
function eventRoot() {
  return {
    listeners: {},
    addEventListener(type, fn) { this.listeners[type] = fn; },
    removeEventListener() {},
    querySelector() { return null; },
  };
}
'''


def test_native_confirmation_uses_ui_locale_without_translating_plugin_name():
    _node(PRELUDE + r'''
import { attachPluginHandlers } from "./marvis/static/js/v2/plugin_manager.js";
import { handleScreenBulkClick } from "./marvis/static/js/v2/screen_gate_controller.js";
const name = "失败的模型插件";
const calls = [];
globalThis.confirm = message => { calls.push(message); return false; };
globalThis.window = {confirm: globalThis.confirm};
for (const locale of ["zh", "en", "zh"]) {
  __setI18nStateForTests({locale});
  const root = eventRoot();
  let removed = false;
  attachPluginHandlers(root, {removePlugin: async () => { removed = true; }});
  const plugin = {dataset: {removePlugin: name}, closest(query) {
    return query === "[data-remove-plugin]" ? this : null;
  }};
  await root.listeners.click({target: plugin, preventDefault() {}});
  assert.equal(calls.at(-1), locale === "en" ? `Remove plugin ${name}?` : `确定移除插件 ${name}？`);
  assert.equal(removed, false);
  const wrap = {querySelectorAll() { return [{}]; }};
  const bulk = {
    closest(query) { return query === ".screen-table-wrap" ? wrap : null; },
    getAttribute() { return "select_visible"; },
  };
  assert.equal(handleScreenBulkClick({
    target: {closest: query => query === "[data-screen-bulk]" ? bulk : null},
    preventDefault() {},
  }), true);
  assert.equal(calls.at(-1), t("当前可见范围包含泄漏/疑似列，全选将把它们一并选中，是否继续？"));
  if (locale === "en") assert.doesNotMatch(calls.at(-1), /[\u3400-\u9fff]/);
}
''')


def test_default_native_alerts_translate_platform_errors_in_all_three_panels():
    _node(PRELUDE + r'''
import { attachPluginHandlers } from "./marvis/static/js/v2/plugin_manager.js";
import { attachSkillHandlers } from "./marvis/static/js/v2/skill_manager.js";
import { attachArtifactHandlers } from "./marvis/static/js/v2/artifact_view.js";
const calls = [];
globalThis.alert = message => calls.push(message);
for (const locale of ["zh", "en", "zh"]) {
  __setI18nStateForTests({locale});
  const pluginRoot = eventRoot();
  attachPluginHandlers(pluginRoot, {uploadPlugin: async () => { throw {status: 403}; }});
  const upload = {files: [{name: "用户插件.zip"}], closest(query) {
    return query === "[data-upload-plugin]" ? this : null;
  }};
  await pluginRoot.listeners.change({target: upload});
  assert.equal(calls.at(-1), localizeText("插件变更需要本地插件管理员确认。"));
  const skillRoot = eventRoot();
  attachSkillHandlers(skillRoot, {reloadSkills: async () => { throw {status: 403}; }});
  const reload = {closest(query) { return query === "#reloadSkills" ? this : null; }};
  await skillRoot.listeners.click({target: reload, preventDefault() {}});
  assert.equal(calls.at(-1), localizeText("Workflow 模板管理仅支持在本地工作区使用。"));
  const artifactRoot = eventRoot();
  attachArtifactHandlers(artifactRoot);
  const artifact = {dataset: {artifact: "用户产物"}, closest(query) {
    return query === "[data-artifact]" ? this : null;
  }};
  await artifactRoot.listeners.click({target: artifact, preventDefault() {}});
  assert.equal(calls.at(-1), localizeText("工件预览面板不可用"));
  if (locale === "en") assert.ok(calls.slice(-3).every(message => !/[\u3400-\u9fff]/.test(message)));
}
''')


def test_draft_rejection_prompt_translates_label_but_preserves_typed_reason():
    _node(PRELUDE + r'''
import { createDraftToolsPanelController } from "./marvis/static/js/draft-tools-panel.js";
const elements = new Map();
const $ = id => {
  if (!elements.has(id)) elements.set(id, {
    value: "", textContent: "", innerHTML: "", disabled: false, dataset: {},
    classList: {add() {}, remove() {}, toggle() {}},
  });
  return elements.get(id);
};
const prompts = [];
const sent = [];
const reason = "这个中文理由是用户输入";
globalThis.window = {prompt(label, initial) { prompts.push([label, initial]); return reason; }};
for (const locale of ["zh", "en", "zh"]) {
  __setI18nStateForTests({locale});
  const controller = createDraftToolsPanelController({
    $, runAction: action => action(), showPlatformConfirm: async () => true,
    api: async (url, options = {}) => {
      if (url.endsWith("/reject")) { sent.push(JSON.parse(options.body)); return {}; }
      return {drafts: []};
    },
  });
  controller.renderDetail({draft: {id: "draft-fixture", name: "用户工具", status: "draft"}});
  await controller.reject();
  assert.deepEqual(prompts.at(-1), [locale === "en" ? "Rejection reason" : "拒绝原因", ""]);
  assert.deepEqual(sent.at(-1), {reason});
}
''')
