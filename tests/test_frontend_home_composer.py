"""Home page (2026-10-14 redesign, batch 5: D7 + 5B conversational creation).

One sentence on the home page reaches each of the seven workflows: the
sentence is read deterministically (``js/home-intent.js``) into a creation card
that asks only for what the create contract still needs, then creates through
the existing create dialog. "创建并开始" leads the first Agent message with the
workflow's own start command, because the governed router never starts a plan
from a bare goal.
"""

import json
import subprocess
from pathlib import Path

from tests.javascript_source import slice_function

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


def _run_module(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_every_workflow_is_reachable_from_one_home_sentence():
    data = _run_module(
        """
import { taskTypeDisplayOrder } from "./marvis/static/js/task-types.js";
import { HOME_EXAMPLES, homeStarterText, inferTaskType } from "./marvis/static/js/home-intent.js";
const starters = Object.fromEntries(taskTypeDisplayOrder.map((type) => {
  const text = homeStarterText(type);
  return [type, { text, ...inferTaskType(text) }];
}));
const examples = HOME_EXAMPLES.map((example) => ({ ...example, inferred: inferTaskType(example.text) }));
const sentences = {
  modeling: inferTaskType("用这份样本做一个贷前 A 卡，OOT 取最近两个月，KS 不低于 0.30"),
  validation: inferTaskType("复核这份 PMML 和 Notebook 的分数一致性"),
  data_join: inferTaskType("把主表和三方数据按 apply_id 拼接"),
  portfolio: inferTaskType("看一下在贷组合的集中度和迁徙"),
  vague: inferTaskType("帮我看看"),
  multi: inferTaskType("验证这三个模型并出报告"),
  exportPmml: inferTaskType("建模并导出PMML"),
  splits: inferTaskType("用训练集和验证集建模"),
};
process.stdout.write(JSON.stringify({ order: taskTypeDisplayOrder, starters, examples, sentences }));
"""
    )
    assert len(data["order"]) == 7
    for task_type in data["order"]:
        starter = data["starters"][task_type]
        assert starter["text"], task_type
        assert starter["taskType"] == task_type, starter
        assert starter["confidence"] == "high", starter
    for example in data["examples"]:
        assert example["inferred"]["taskType"] == example["taskType"], example
    for task_type in ("modeling", "validation", "data_join", "portfolio"):
        assert data["sentences"][task_type]["taskType"] == task_type
    assert data["sentences"]["vague"] == {
        "taskType": None,
        "confidence": "none",
        "candidates": [],
        "multiModel": False,
    }
    assert data["sentences"]["multi"]["multiModel"] is True
    # Data splits and PMML export are modeling words, not a validation task.
    assert (data["sentences"]["exportPmml"]["taskType"], data["sentences"]["exportPmml"]["confidence"]) == ("modeling", "high")
    assert (data["sentences"]["splits"]["taskType"], data["sentences"]["splits"]["confidence"]) == ("modeling", "high")


def test_home_recommendations_cover_every_workflow_and_vary_without_duplicates():
    data = _run_module(
        '''
import { HOME_EXAMPLES, HOME_EXAMPLE_COUNT, pickHomeExamples } from "./marvis/static/js/home-intent.js";
import { localizeText, __setI18nStateForTests } from "./marvis/static/js/i18n.js";
__setI18nStateForTests({locale: "en"});
const counts = {};
HOME_EXAMPLES.forEach(({ taskType }) => counts[taskType] = (counts[taskType] || 0) + 1);
const first = pickHomeExamples(undefined, () => 0.1);
const second = pickHomeExamples(undefined, () => 0.9);
process.stdout.write(JSON.stringify({counts, count: HOME_EXAMPLE_COUNT, first, second,
  types: first.map((i) => HOME_EXAMPLES[i].taskType), english: HOME_EXAMPLES.map(({text}) => localizeText(text))}));
'''
    )
    assert len(data["counts"]) == 7
    assert all(count >= 4 for count in data["counts"].values())
    assert len(data["first"]) == data["count"] == 5
    assert len(set(data["types"])) == 5
    assert data["first"] != data["second"]
    assert all(not any("\u3400" <= char <= "\u9fff" for char in text) for text in data["english"])


def test_drafted_task_names_name_the_work_not_the_targets():
    data = _run_module(
        """
import { draftTaskName, homeStarterText } from "./marvis/static/js/home-intent.js";
const cases = [
  ["通过率 60% 以上，坏账 5% 以内的准入策略", "strategy"],
  ["用这份样本做一个贷前 A 卡，OOT 取最近两个月，KS 不低于 0.30", "modeling"],
  ["反欺诈 FPD7 模型验证并出报告", "validation"],
  ["这批新特征值不值得用？", "feature_analysis"],
  ["验证这个模型并出报告", "validation"],
  ["KS 不低于 0.3", "modeling"],
  ["", "portfolio"],
  ["用户分层准入策略", "strategy"],
  ["建模，OOT KS 不低于 0.3", "modeling"],
  ["建行客群的准入策略", "strategy"],
  ["根据数据字典拼接主表", "data_join"],
  ["来源数据合并", "data_join"],
  ["基于这批数据做特征分析", "feature_analysis"],
];
const starters = ["data_join", "vintage", "strategy"].map((type) => draftTaskName(homeStarterText(type), type));
process.stdout.write(JSON.stringify({ names: cases.map(([text, type]) => draftTaskName(text, type)), starters }));
"""
    )
    assert data["names"] == [
        "准入策略",
        "贷前 A 卡 建模",
        "反欺诈 FPD7 模型验证",
        "新特征分析",
        "模型验证",
        "新建模任务",
        "新组合分析任务",
        # Prefixes are stripped only as whole phrases, never a word's first character.
        "用户分层准入策略",
        "建模",
        "建行客群的准入策略",
        "根据数据字典拼接主表",
        "来源数据合并",
        "特征分析",
    ]
    assert data["starters"] == ["主表和特征表拼成建模宽表", "标准 Vintage", "准入策略"]


def test_card_reads_strategy_limits_algorithms_and_one_risk_analysis():
    data = _run_module(
        """
import {
  parseModelAlgorithms,
  parseRiskAnalysis,
  parseStrategyLimits,
  strategyLimitValue,
} from "./marvis/static/js/home-intent.js";
process.stdout.write(JSON.stringify({
  limits: parseStrategyLimits("通过率 60% 以上，坏账 5% 以内的准入策略"),
  reordered: parseStrategyLimits("坏率不超过3.5%，通过率不低于55%"),
  none: parseStrategyLimits("做一版准入策略并回测"),
  values: ["60", "", "160", "abc"].map((value) => strategyLimitValue(value)),
  algorithms: parseModelAlgorithms("用 LightGBM 和逻辑回归做评分卡"),
  analyses: [
    parseRiskAnalysis("做一次标准 Vintage，看各月放款的坏账曲线"),
    parseRiskAnalysis("测算 VTG 终值和年化不良"),
    parseRiskAnalysis("做一次三季度 Vintage 与收益测算"),
    parseRiskAnalysis("看看风险"),
  ],
}));
"""
    )
    assert data["limits"] == {"minApprovalRate": "60", "maxBadRate": "5"}
    assert data["reordered"] == {"minApprovalRate": "55", "maxBadRate": "3.5"}
    assert data["none"] == {"minApprovalRate": "", "maxBadRate": ""}
    # JSON turns NaN into null; out-of-range and non-numbers are both invalid.
    assert data["values"] == [0.6, None, None, None]
    assert data["algorithms"] == ["lgb", "lr", "scorecard"]
    # Two analyses in one sentence are not guessed: the card asks.
    assert data["analyses"] == ["标准 Vintage", "VTG终值与年化不良", "", ""]


def test_card_asks_only_for_what_the_create_contract_still_needs():
    data = _run_module(
        """
import { draftMissingFields } from "./marvis/static/js/home-intent.js";
const ready = { taskType: "feature_analysis", name: "新特征分析", validator: "张三", runMode: "agent", files: [{}] };
process.stdout.write(JSON.stringify({
  ready: draftMissingFields(ready),
  empty: draftMissingFields({}),
  noOwner: draftMissingFields({ ...ready, validator: "  " }),
  noFiles: draftMissingFields({ ...ready, files: [] }),
  vintageAgent: draftMissingFields({ ...ready, taskType: "vintage", files: [], analysis: "标准 Vintage" }, { deferredMaterials: true }),
  vintageManual: draftMissingFields({ ...ready, taskType: "vintage", runMode: "manual", files: [], analysis: "" }, { deferredMaterials: true }),
  strategyNone: draftMissingFields({ ...ready, taskType: "strategy", strategy: { minApprovalRate: "", maxBadRate: "" } }),
  strategyBad: draftMissingFields({ ...ready, taskType: "strategy", strategy: { minApprovalRate: "160", maxBadRate: "" } }),
  strategyOne: draftMissingFields({ ...ready, taskType: "strategy", strategy: { minApprovalRate: "", maxBadRate: "5" } }),
  manualModeling: draftMissingFields({ ...ready, taskType: "modeling", runMode: "manual", algorithms: [] }),
  agentModeling: draftMissingFields({ ...ready, taskType: "modeling", runMode: "agent", algorithms: [] }),
  multiModel: draftMissingFields({ ...ready, taskType: "validation", multiModel: true }),
}));
"""
    )
    assert data["ready"] == []
    assert data["empty"] == ["taskType", "name", "validator", "files"]
    assert data["noOwner"] == ["validator"]
    assert data["noFiles"] == ["files"]
    # Only vintage in Agent mode may start without materials (deferredMaterials).
    assert data["vintageAgent"] == []
    assert data["vintageManual"] == ["files", "riskAnalysis"]
    assert data["strategyNone"] == ["strategyLimits"]
    assert data["strategyBad"] == ["strategyLimits"]
    assert data["strategyOne"] == []
    assert data["manualModeling"] == ["algorithms"]
    assert data["agentModeling"] == []
    # A sentence that asks for several models waits until the card holds them.
    assert data["multiModel"] == ["models"]


def test_continue_list_puts_waiting_work_first_then_running_then_failed():
    data = _run_module(
        """
import { homeContinueTasks } from "./marvis/static/js/home-composer.js";
const tasks = ["a-bad", "b-done", "c-wait", "d-run", "e-wait", "f-bad"].map((id) => ({ id }));
const dots = { "a-bad": "bad", "b-done": "none", "c-wait": "wait", "d-run": "run", "e-wait": "wait", "f-bad": "bad" };
const rows = homeContinueTasks(tasks, (task) => ({ dot: dots[task.id], label: task.id }));
process.stdout.write(JSON.stringify(rows.map((row) => row.task.id)));
"""
    )
    assert data == ["c-wait", "e-wait", "d-run"]



def test_continue_rows_say_which_step_the_task_stopped_on():
    # Batch 4: 接着做 names the step from the structured current step, never
    # from free text; without one the row keeps the status word.
    data = _run_module(
        """
import { createHomeComposer, homeContinueProgress } from "./marvis/static/js/home-composer.js";
const els = { homeContinue: { hidden: true }, homeContinueList: { innerHTML: "" } };
const tasks = [
  { id: "w", name: "很长很长的任务名", current_step: { number: 3, title: "PMML打分测试" } },
  { id: "r", name: "r", current_step: { number: 2, title: "训练模型" } },
  { id: "x", name: "x", current_step: null },
];
const dots = { w: "wait", r: "run", x: "bad" };
const composer = createHomeComposer({
  $: (id) => els[id] || null,
  getTasks: () => tasks,
  statusOf: (task) => ({ dot: dots[task.id], label: "失败" }),
  taskName: (task) => task.name,
});
composer.renderContinue();
process.stdout.write(JSON.stringify({
  html: els.homeContinueList.innerHTML,
  bad: homeContinueProgress({ number: 4, title: "报告输出" }, "bad"),
  invalid: [homeContinueProgress({ number: 0, title: "a" }), homeContinueProgress({ number: 2, title: "" }), homeContinueProgress(null)],
}));
"""
    )
    html = data["html"]
    assert "停在第 3 步：PMML打分测试" in html
    assert "正在第 2 步：训练模型" in html
    assert '<span class="home-continue-progress">失败</span>' in html
    assert 'class="home-continue-name" title="很长很长的任务名"' in html
    assert data["bad"] == "第 4 步失败：报告输出"
    assert data["invalid"] == ["", "", ""]


def test_continue_progress_strings_are_in_the_english_dictionary():
    data = _run_module("""
import { localizeText, __setI18nStateForTests } from "./marvis/static/js/i18n.js";
__setI18nStateForTests({locale: "en"});
process.stdout.write(JSON.stringify(["停在第 3 步：训练模型", "正在第 3 步：训练模型", "第 3 步失败：训练模型"].map(localizeText)));
""")
    assert len(set(data)) == 3
    for text in data:
        assert "3" in text
        assert not any("\u4e00" <= char <= "\u9fff" for char in text), text


_CONTROLLER_HARNESS = r"""
import { createHomeComposer } from "./marvis/static/js/home-composer.js";
import { taskTypeDefinitions, taskTypeDisplayOrder } from "./marvis/static/js/task-types.js";
function element(id) {
  const listeners = {};
  return {
    id, value: "", hidden: false, disabled: false, innerHTML: "", dataset: {},
    focus() {}, setSelectionRange() {}, querySelector() { return null; },
    addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
    async fire(type, event = {}) {
      for (const fn of listeners[type] || []) await fn({ preventDefault() {}, target: this, ...event });
    },
  };
}
const ids = ["homeComposer", "homeComposerInput", "homeComposerSend", "homeComposerFiles",
  "homeComposerFileInput", "homeExamples", "homeContinue", "homeContinueList", "homeDraft", "workspaceWelcome"];
const els = Object.fromEntries(ids.map((id) => [id, element(id)]));
const $ = (id) => els[id] || null;
const store = new Map([["marvis_last_validator", "张三"]]);
const created = [];
const advanced = [];
let failNext = "";
const composer = createHomeComposer({
  $, taskTypeDefinitions, taskTypeDisplayOrder,
  hasEnabledAgent: () => true,
  storage: { getItem: (key) => store.get(key) ?? null },
  onCreate: async (draft) => {
    if (failNext) { const message = failNext; failNext = ""; throw new Error(message); }
    created.push(draft);
    return true;
  },
  onAdvanced: (draft) => advanced.push(draft),
});
composer.bind();
const target = (selector, node) => ({ closest: (wanted) => (wanted === selector ? node : null) });
const draftField = (field, value) => target("[data-home-draft-field]", { dataset: { homeDraftField: field }, value });
"""


def test_icons_and_examples_only_fill_the_composer():
    data = _run_module(
        _CONTROLLER_HARNESS
        + """
composer.pickTaskType({ target: target("[data-task-kind]", { dataset: { taskKind: "data_join" } }) });
const afterIcon = { value: els.homeComposerInput.value, send: els.homeComposerSend.disabled, draft: composer.getDraft() };
// A sentence the user wrote is kept when an icon is clicked afterwards.
els.homeComposerInput.value = "我自己写的目标";
composer.pickTaskType({ target: target("[data-task-kind]", { dataset: { taskKind: "modeling" } }) });
const keptOwn = els.homeComposerInput.value;
await els.homeExamples.fire("click", { target: target("[data-home-example]", { dataset: { homeExample: "1" } }) });
process.stdout.write(JSON.stringify({ afterIcon, keptOwn, example: els.homeComposerInput.value, examplesHtml: els.homeExamples.innerHTML, created, advanced }));
"""
    )
    assert data["afterIcon"] == {"value": "把主表和特征表拼成建模宽表", "send": False, "draft": None}
    assert data["keptOwn"] == "我自己写的目标"
    assert data["example"] == "这批新特征值不值得用？"
    assert data["examplesHtml"].count("data-home-example=") == 5
    assert data["created"] == []
    assert data["advanced"] == []


def test_english_defaults_fill_in_english_and_locale_changes_keep_user_text():
    data = _run_module(
        _CONTROLLER_HARNESS.replace(
            'const composer = createHomeComposer({',
            '''const localeListeners = new Map();
const localeDoc = { addEventListener(name, listener) { localeListeners.set(name, listener); } };
const composer = createHomeComposer({ doc: localeDoc,''',
        )
        + '''
const { __setI18nStateForTests, UI_LOCALE_CHANGE_EVENT } = await import("./marvis/static/js/i18n.js");
__setI18nStateForTests({ locale: "en" });
composer.pickTaskType({ target: target("[data-task-kind]", { dataset: { taskKind: "data_join" } }) });
const starter = els.homeComposerInput.value;
await els.homeExamples.fire("click", { target: target("[data-home-example]", { dataset: { homeExample: "1" } }) });
const example = els.homeComposerInput.value;
__setI18nStateForTests({ locale: "zh" });
localeListeners.get(UI_LOCALE_CHANGE_EVENT)({ detail: { locale: "zh" } });
const switched = els.homeComposerInput.value;
els.homeComposerInput.value = "My own edited goal";
__setI18nStateForTests({ locale: "en" });
localeListeners.get(UI_LOCALE_CHANGE_EVENT)({ detail: { locale: "en" } });
const edited = els.homeComposerInput.value;
process.stdout.write(JSON.stringify({ starter, example, switched, edited }));
'''
    )
    assert data["starter"] == "Join the main table and feature tables into a modeling dataset"
    assert data["example"] == "Are these new features worth using?"
    assert data["switched"] == "这批新特征值不值得用？"
    assert data["edited"] == "My own edited goal"


def test_english_work_templates_keep_file_roles_and_parse_default_constraints():
    data = _run_module(
        '''
import { taskTypeDisplayOrder } from "./marvis/static/js/task-types.js";
import { homeStarterText, homeTemplateTokens, homeTemplateName, parseStrategyLimits,
  parseOotKsGoal, parseValidatorName, parseRiskAnalysis } from "./marvis/static/js/home-intent.js";
const sentence = (type) => homeTemplateTokens(type, "en").map((token) =>
  typeof token === "string" ? token : token.slot ?? token.label).join("");
const templates = taskTypeDisplayOrder.map((type) => {
  const english = homeTemplateTokens(type, "en");
  const sourceRoles = homeTemplateTokens(type).filter((token) => token?.file).map((token) => token.label);
  return {
    type, sentence: sentence(type), starter: homeStarterText(type, "", "en"),
    name: homeTemplateName(type, english.filter((token) => token?.slot).map((token) => token.slot), "en"),
    sourceRoles, englishRoles: english.filter((token) => token?.file).map((token) => token.role || token.label),
  };
});
process.stdout.write(JSON.stringify({ templates,
  limits: parseStrategyLimits(sentence("strategy")), ks: parseOotKsGoal(sentence("modeling")),
  validator: parseValidatorName(sentence("validation")), analysis: parseRiskAnalysis(sentence("vintage")),
}));
'''
    )
    for row in data["templates"]:
        assert row["sourceRoles"] == row["englishRoles"], row["type"]
        assert row["sentence"] and row["name"] and row["starter"]
        assert not any("\u3400" <= char <= "\u9fff" for char in row["sentence"] + row["name"] + row["starter"])
    assert data["limits"] == {"minApprovalRate": "60", "maxBadRate": "5"}
    assert data["ks"] == "0.3"
    assert data["validator"] == "Jane Doe"
    assert data["analysis"] == "标准 Vintage"


def test_sentence_opens_a_card_that_asks_then_creates_through_the_dialog_contract():
    data = _run_module(
        _CONTROLLER_HARNESS
        + """
store.delete("marvis_last_validator");
await els.homeExamples.fire("click", { target: target("[data-home-example]", { dataset: { homeExample: "2" } }) });
await els.homeComposer.fire("submit");
const draft = { ...composer.getDraft() };
const mode = els.workspaceWelcome.dataset.homeMode;
const firstHtml = els.homeDraft.innerHTML;
await els.homeDraft.fire("submit");
const blocked = { created: created.length, html: els.homeDraft.innerHTML };
await els.homeDraft.fire("input", { target: draftField("validator", "  李四 ") });
const file = { name: "score.csv", size: 2048, lastModified: 1 };
await els.homeDraft.fire("change", { target: { matches: (selector) => selector === "[data-home-draft-files]", files: [file] } });
failNext = "上传失败";
await els.homeDraft.fire("submit");
const failedHtml = els.homeDraft.innerHTML;
await els.homeDraft.fire("submit");
process.stdout.write(JSON.stringify({
  draft, mode, firstHtml, blocked, failedHtml,
  created: created.map(({ taskType, name, validator, runMode, strategy, files, text }) => ({ taskType, name, validator, runMode, strategy, files: files.length, text })),
  after: { mode: els.workspaceWelcome.dataset.homeMode, draft: composer.getDraft(), input: els.homeComposerInput.value, hidden: els.homeDraft.hidden },
}));
"""
    )
    draft = data["draft"]
    assert draft["taskType"] == "strategy"
    assert draft["typeSource"] == "picked"
    assert draft["showTypes"] is False
    assert draft["name"] == "准入策略"
    assert draft["validator"] == ""
    assert draft["runMode"] == "agent"
    assert draft["strategy"] == {"minApprovalRate": "60", "maxBadRate": "5"}
    assert data["mode"] == "draft"
    assert "我理解为「策略开发」任务。开始前还需要：负责人、材料。" in data["firstHtml"]
    assert 'data-home-draft-advanced' in data["firstHtml"]
    assert 'data-home-draft-limit="minApprovalRate"' in data["firstHtml"]
    # Missing items block creation and are named on the card.
    assert data["blocked"]["created"] == 0
    assert '<p class="home-draft-error" role="alert">还需要：负责人、材料。</p>' in data["blocked"]["html"]
    # A failed create stays on the card with the reason.
    assert "上传失败" in data["failedHtml"]
    assert data["created"] == [
        {
            "taskType": "strategy",
            "name": "准入策略",
            "validator": "李四",
            "runMode": "agent",
            "strategy": {"minApprovalRate": "60", "maxBadRate": "5"},
            "files": 1,
            "text": "通过率 60% 以上，坏账 5% 以内的准入策略",
        }
    ]
    assert data["after"] == {"mode": "home", "draft": None, "input": "", "hidden": True}


def test_unclear_sentence_lists_every_type_and_advanced_settings_waits_for_one():
    data = _run_module(
        _CONTROLLER_HARNESS
        + """
els.homeComposerInput.value = "帮我看看";
await els.homeComposer.fire("submit");
const before = { ...composer.getDraft() };
const html = els.homeDraft.innerHTML;
await els.homeDraft.fire("click", { target: target("[data-home-draft-type]", { dataset: { homeDraftType: "vintage" } }) });
const picked = { ...composer.getDraft() };
const pickedHtml = els.homeDraft.innerHTML;
await els.homeDraft.fire("click", { target: target("[data-home-draft-advanced]", {}) });
process.stdout.write(JSON.stringify({ before, html, picked, pickedHtml, advanced }));
"""
    )
    assert data["before"]["taskType"] == ""
    assert data["before"]["showTypes"] is True
    assert "这是哪一类工作？选一下，我按它准备。" in data["html"]
    assert data["html"].count("data-home-draft-type=") == 7
    assert "data-home-draft-advanced disabled" in data["html"]
    assert data["picked"]["taskType"] == "vintage"
    assert data["picked"]["validator"] == "张三"
    assert data["picked"]["name"] == "新风险分析任务"
    # Vintage asks which single analysis the intake should start.
    assert data["pickedHtml"].count("data-home-draft-analysis=") == 3
    assert "开始前还需要：分析类型。" in data["pickedHtml"]
    assert [item["taskType"] for item in data["advanced"]] == ["vintage"]


def _first_message_harness(cases: str) -> list:
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    return _run_module(
        "import { taskTypeDefinitions } from './marvis/static/js/task-types.js';\n"
        "function taskTypeDefinition(taskType) { return taskTypeDefinitions[taskType] || {}; }\n"
        + slice_function(app_js, "function homeFirstAgentMessage(")
        + "\n"
        + cases
    )


def test_first_agent_message_leads_with_the_workflow_start_command():
    data = _first_message_harness(
        """
const goal = "看看这批新特征的 IV 和 PSI，哪些值得用";
process.stdout.write(JSON.stringify([
  homeFirstAgentMessage({ task_type: "feature_analysis" }, goal),
  homeFirstAgentMessage({ task_type: "modeling" }, "用这份样本做一个贷前 A 卡"),
  homeFirstAgentMessage({ task_type: "data_join" }, "开始数据处理"),
  homeFirstAgentMessage({ task_type: "validation" }, "这个模型能不能上线？"),
  homeFirstAgentMessage({ task_type: "vintage" }, "标准 Vintage"),
  homeFirstAgentMessage({ task_type: "strategy" }, "  "),
]));
"""
    )
    assert data == [
        "开始特征分析。我的目标：看看这批新特征的 IV 和 PSI，哪些值得用",
        "开始建模。我的目标：用这份样本做一个贷前 A 卡",
        "开始数据处理",
        # Validation's start check rejects questions and takes no free-text goal.
        "开始模型验证",
        # The risk intake takes its own analysis command as-is.
        "标准 Vintage",
        "",
    ]




def test_a_create_that_finishes_late_never_touches_a_newer_draft():
    data = _run_module(
        _CONTROLLER_HARNESS.replace(
            "const composer = createHomeComposer({",
            "let release = null;\nconst errors = [];\nconst composer = createHomeComposer({\n  onError: (error) => errors.push(error.message),",
        ).replace(
            "    created.push(draft);\n    return true;",
            "    created.push(draft);\n    // Like app.js: createTask resets the home page once the task exists.\n    composer.reset();\n    await new Promise((resolve) => { release = resolve; });\n    if (failNext) throw new Error(failNext);\n    return true;",
        )
        + """
const file = { name: "a.csv", size: 10, lastModified: 1 };
els.homeComposerInput.value = "看看这批新特征值不值得用";
await els.homeComposerFileInput.fire("change", { target: { files: [file], value: "" } });
await els.homeComposer.fire("submit");
const pending = els.homeDraft.fire("submit");
await new Promise((resolve) => setTimeout(resolve, 0));
const afterCreate = { mode: els.workspaceWelcome.dataset.homeMode, draft: composer.getDraft() };
// The user starts the next task while the first one is still starting.
els.homeComposerInput.value = "把主表和特征表拼成建模宽表";
await els.homeComposer.fire("submit");
failNext = "扫描失败";
release();
await pending;
// The card's submit handler does not await the create; let it settle.
await new Promise((resolve) => setTimeout(resolve, 0));
process.stdout.write(JSON.stringify({
  afterCreate,
  second: composer.getDraft() && { taskType: composer.getDraft().taskType, text: composer.getDraft().text },
  mode: els.workspaceWelcome.dataset.homeMode,
  secondHtml: els.homeDraft.innerHTML.includes("扫描失败"),
  errors,
  created: created.length,
}));
"""
    )
    assert data["afterCreate"] == {"mode": "home", "draft": None}
    assert data["second"] == {"taskType": "data_join", "text": "把主表和特征表拼成建模宽表"}
    assert data["mode"] == "draft"
    # The late failure is reported elsewhere, not written into the new card.
    assert data["secondHtml"] is False
    assert data["errors"] == ["扫描失败"]
    assert data["created"] == 1


def test_rewritten_sentence_drops_a_stale_icon_pick():
    data = _run_module(
        _CONTROLLER_HARNESS
        + """
composer.pickTaskType({ target: target("[data-task-kind]", { dataset: { taskKind: "strategy" } }) });
await els.homeComposer.fire("submit");
const picked = { ...composer.getDraft() };
await els.homeDraft.fire("click", { target: target("[data-home-draft-back]", {}) });
els.homeComposerInput.value = "用这份样本建一个评分卡";
await els.homeComposer.fire("submit");
const rewritten = { ...composer.getDraft() };
await els.homeDraft.fire("click", { target: target("[data-home-draft-back]", {}) });
els.homeComposerInput.value = "帮我看看";
await els.homeComposer.fire("submit");
const vague = { ...composer.getDraft() };
process.stdout.write(JSON.stringify({ picked, rewritten, vague }));
"""
    )
    assert (data["picked"]["taskType"], data["picked"]["confidence"], data["picked"]["showTypes"]) == ("strategy", "high", False)
    assert (data["rewritten"]["taskType"], data["rewritten"]["typeSource"], data["rewritten"]["confidence"]) == ("modeling", "inferred", "high")
    assert data["rewritten"]["name"] == "评分卡 建模"
    # Nothing in the new sentence contradicts the pick: kept, but only as a guess.
    assert (data["vague"]["taskType"], data["vague"]["confidence"], data["vague"]["showTypes"]) == ("strategy", "low", True)


def test_multi_model_sentence_lists_models_and_waits_for_their_materials():
    # 2026-11-10: several models are added on the card itself (添加模型), not
    # through 高级设置; 只验证一个模型 still falls back to one model.
    data = _run_module(
        _CONTROLLER_HARNESS
        + """
await els.homeExamples.fire("click", { target: target("[data-home-example]", { dataset: { homeExample: "0" } }) });
await els.homeComposerFileInput.fire("change", { target: { files: [{ name: "m.pmml", size: 5, lastModified: 2 }], value: "" } });
await els.homeComposer.fire("submit");
const html = els.homeDraft.innerHTML;
const models = composer.getDraft().models.map((model) => model.items.map((item) => item.file.name));
await els.homeDraft.fire("submit");
const blocked = created.length;
await els.homeDraft.fire("click", { target: target("[data-home-draft-single]", {}) });
const single = { models: composer.getDraft().models, files: composer.getDraft().files.map((item) => item.file.name) };
await els.homeDraft.fire("submit");
process.stdout.write(JSON.stringify({ html, models, blocked, single, created: created.map((draft) => [draft.multiModel, draft.models]) }));
"""
    )
    assert "共 2 个模型" in data["html"]
    assert "开始前还需要：每个模型的名称和四份材料。" in data["html"]
    assert 'data-home-model-name="1"' in data["html"]
    assert "data-home-model-add" in data["html"]
    assert "data-home-draft-single" in data["html"]
    assert "高级设置」里逐个添加" not in data["html"]
    assert data["models"] == [["m.pmml"], []]
    assert data["blocked"] == 0
    assert data["single"] == {"models": None, "files": ["m.pmml"]}
    assert data["created"] == [[False, None]]


def test_dropped_materials_of_several_models_create_one_batch():
    data = _run_module(
        _CONTROLLER_HARNESS
        + """
const file = (name) => ({ name, size: 5, lastModified: 3 });
els.homeComposerInput.value = "帮我验证模型并出报告";
await els.homeComposerFileInput.fire("change", { target: { files: [
  file("A卡_notebook.ipynb"), file("A卡_model.pmml"), file("A卡_sample.csv"),
  file("B卡.ipynb"), file("B卡.pmml"), file("B卡_样本.csv"), file("数据字典.xlsx"),
], value: "" } });
await els.homeComposer.fire("submit");
const draft = composer.getDraft();
const html = els.homeDraft.innerHTML;
const before = { runMode: draft.runMode, models: draft.models.map((model) => [model.name, model.items.length]) };
await els.homeDraft.fire("submit");
const sent = created[0];
process.stdout.write(JSON.stringify({ html, before, created: created.length,
  sent: sent && { models: sent.models.map((model) => [model.name, model.items.map((item) => item.file.name)]), runMode: sent.runMode } }));
"""
    )
    assert data["before"] == {"runMode": "agent", "models": [["A卡", 4], ["B卡", 4]]}
    assert "模型（2）" in data["html"]
    # A batch runs under Agent auto-review: the manual chip is disabled.
    assert 'data-home-draft-mode="manual" aria-pressed="false" disabled' in data["html"]
    # No single-model name field: the batch is named by date and model count.
    assert 'data-home-draft-field="name"' not in data["html"]
    assert data["created"] == 1
    assert data["sent"]["runMode"] == "agent"
    assert data["sent"]["models"] == [
        ["A卡", ["A卡_notebook.ipynb", "A卡_model.pmml", "A卡_sample.csv", "数据字典.xlsx"]],
        ["B卡", ["B卡.ipynb", "B卡.pmml", "B卡_样本.csv", "数据字典.xlsx"]],
    ]


def test_card_adds_names_and_removes_models():
    data = _run_module(
        _CONTROLLER_HARNESS
        + """
const file = (name) => ({ name, size: 5, lastModified: 4 });
els.homeComposerInput.value = "验证这个模型";
await els.homeComposerFileInput.fire("change", { target: { files: [file("a.ipynb"), file("a.pmml"), file("a.csv"), file("dict.csv")], value: "" } });
await els.homeComposer.fire("submit");
const single = composer.getDraft().models;
const name = composer.getDraft().name;
await els.homeDraft.fire("click", { target: target("[data-home-model-add]", { disabled: false }) });
const afterAdd = composer.getDraft().models.map((model) => model.items.length);
await els.homeDraft.fire("input", { target: target("[data-home-model-name]", { dataset: { homeModelName: "1" }, value: "B卡" }) });
await els.homeDraft.fire("change", { target: { matches: (selector) => selector === "[data-home-model-files]", dataset: { homeModelFiles: "1" },
  files: [file("b.ipynb"), file("b.pmml"), file("b.csv"), file("b_dict.csv")] } });
const filled = composer.getDraft().models.map((model) => [model.name, model.items.length]);
const lead = els.homeDraft.innerHTML.includes("都齐了");
await els.homeDraft.fire("click", { target: target("[data-home-model-remove]", { dataset: { homeModelRemove: "1" } }) });
const after = composer.getDraft();
process.stdout.write(JSON.stringify({ single, name, afterAdd, filled, lead,
  collapsed: { models: after.models, files: after.files.map((item) => item.file.name) } }));
"""
    )
    assert data["single"] is None
    assert data["afterAdd"] == [4, 0]
    # Model 1 keeps the card's name and files; model 2 gets its own.
    assert data["filled"] == [[data["name"], 4], ["B卡", 4]]
    assert data["name"]
    assert data["lead"] is True
    assert data["collapsed"] == {"models": None, "files": ["a.ipynb", "a.pmml", "a.csv", "dict.csv"]}


def test_home_templates_name_tasks_from_their_slots_and_strip_owner_phrases():
    # 2026-10-19 user feedback (U8) and eval S2: template sentences are named
    # from their value slots; "这是我们新上的行为评分卡" names the model.
    import json
    import subprocess
    from pathlib import Path

    module = (Path(__file__).resolve().parents[1] / "marvis" / "static" / "js" / "home-intent.js").as_uri()
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        "const out = {",
        "  modeling: m.homeTemplateName('modeling', ['MOB3_30', 'B卡']),",
        "  strategy: m.homeTemplateName('strategy', ['贷前准入']),",
        "  emptySlot: m.homeTemplateName('modeling', ['MOB3_30', ' ']),",
        "  owner: m.draftTaskName('这是我们新上的行为评分卡，帮我做一下模型验证', 'validation'),",
        "  kinds: Object.keys(m.HOME_TEMPLATES).sort(),",
        "  fileSlots: m.HOME_TEMPLATES.modeling.filter((t) => t && t.file).map((t) => t.label),",
        "};",
        "process.stdout.write(JSON.stringify(out));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data["modeling"] == "MOB3_30 B卡模型"
    assert data["strategy"] == "贷前准入策略"
    assert data["emptySlot"] == ""
    assert data["owner"] == "行为评分卡"
    assert data["kinds"] == sorted(["modeling", "validation", "feature_analysis", "data_join", "vintage", "portfolio", "strategy"])
    assert data["fileSlots"] == ["Y样本", "X数据"]


def test_home_sentence_ks_goal_prefills_the_modeling_oot_ks_floor():
    # Eval S4: "OOT 的 KS 尽量不低于 0.3" never reached the task's oot_ks_min,
    # so the modeling Agent could not say whether the goal was met.
    import json
    import subprocess
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "marvis" / "static"
    module = (static / "js" / "home-intent.js").as_uri()
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        "const texts = ['OOT 的 KS 尽量不低于 0.3', 'KS 至少 0.30', 'KS≥30%', 'ks不低于35%', 'KS 0.3 左右', 'AUC 不低于 0.7'];",
        "process.stdout.write(JSON.stringify(texts.map((t) => m.parseOotKsGoal(t))));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["0.3", "0.3", "0.3", "0.35", "", ""]
    dialog = (static / "js" / "create-task-dialog.js").read_text(encoding="utf-8")
    assert '$("modelOotKsMin").value = draft.ootKsMin;' in dialog
    composer = (static / "js" / "home-composer.js").read_text(encoding="utf-8")
    assert "ootKsMin: parseOotKsGoal(text)," in composer


def test_home_sentence_names_the_validator_for_the_card():
    # Eval round 1: the validation template says "验证人员是 张三" but the
    # card's 验证人员 field stayed empty, so the user typed the name twice.
    import json
    import subprocess
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "marvis" / "static"
    module = (static / "js" / "home-intent.js").as_uri()
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        "const texts = ['验证人员是 张三，出一份报告', '负责人：李四。', '验证人 Alice Wang, 出报告', '帮我验证模型'];",
        "process.stdout.write(JSON.stringify(texts.map((t) => m.parseValidatorName(t))));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["张三", "李四", "Alice Wang", ""]
    composer = (static / "js" / "home-composer.js").read_text(encoding="utf-8")
    assert "validator: parseValidatorName(text) || lastValidator()," in composer
    # The template's person slot starts from the last name used.
    assert 'token.key === "validator" ? lastValidator() || token.slot : token.slot' in composer


def test_joining_tables_named_after_features_reads_as_data_processing():
    # Eval S6: 「征信特征表」 pulled a join request into feature analysis; an
    # unrelated request got the type list without saying what the platform does.
    import json
    import subprocess
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "marvis" / "static"
    module = (static / "js" / "home-intent.js").as_uri()
    texts = [
        "把这两张表拼起来，申请表是主表，征信特征表按客户关联过来",
        "这批特征哪些能用？帮我看看区分度和稳定性",
        "帮我订一张去上海的机票",
    ]
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        f"const texts = {json.dumps(texts, ensure_ascii=False)};",
        "process.stdout.write(JSON.stringify(texts.map((t) => { const r = m.inferTaskType(t); return [r.taskType, r.confidence]; })));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [["data_join", "high"], ["feature_analysis", "high"], [None, "none"]]
    composer = (static / "js" / "home-composer.js").read_text(encoding="utf-8")
    assert "订票、写作这类别的事情我做不了" in composer


def test_same_named_attachments_are_called_out_before_creating():
    # Eval S5: two sets of model.ipynb/model.pmml/... silently became one.
    import json
    import subprocess
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "marvis" / "static"
    module = (static / "js" / "home-composer.js").as_uri()
    script = "\n".join([
        "globalThis.document = undefined;",
        f"const m = await import({json.dumps(module)});",
        "const items = ['a/model.ipynb', 'b/model.ipynb', 'sample.csv'].map((relativePath) => ({ relativePath, file: { name: relativePath.split('/').pop() } }));",
        "process.stdout.write(JSON.stringify(m.duplicateFileNames(items)));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["model.ipynb"]
    composer = (static / "js" / "home-composer.js").read_text(encoding="utf-8")
    assert "同名文件上传后只会保留一份" in composer


def test_home_sentence_asking_for_stability_adds_monthly_psi():
    # Eval S3: 「帮我看看区分度和稳定性」 ran without PSI.
    import json
    import subprocess
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "marvis" / "static"
    module = (static / "js" / "home-intent.js").as_uri()
    texts = ["这批特征哪些能用？帮我看看区分度和稳定性，bad_flag 是标签", "看看共线和头尾 lift", "看看 IV"]
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        f"process.stdout.write(JSON.stringify({json.dumps(texts, ensure_ascii=False)}.map((t) => m.parseFeatureMetrics(t))));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [["psi_month_first", "psi_month_previous"], ["vif", "head_tail_lift"], []]
    dialog = (static / "js" / "create-task-dialog.js").read_text(encoding="utf-8")
    assert 'draft.taskType === "feature_analysis" && Array.isArray(draft.featureMetrics)' in dialog
    composer = (static / "js" / "home-composer.js").read_text(encoding="utf-8")
    assert "featureMetrics: parseFeatureMetrics(text)," in composer


def test_collections_and_anti_fraud_sentences_route_with_plain_scope_notes():
    data = _run_module(
        """
import { draftMissingFields, inferTaskType, parseScopeNotes } from "./marvis/static/js/home-intent.js";
const sentences = ["给逾期客户做一版催收分案规则", "做一套催收策略，自动外呼并预测回收率", "Set up a collections queue",
  "收益测算，催收成本率 3%", "做一个催收评分卡模型", "做一个反欺诈模型", "反欺诈规则"];
const base = { name: "催收", validator: "qa", runMode: "agent", files: [{ name: "a.csv" }], strategy: { minApprovalRate: "", maxBadRate: "" } };
process.stdout.write(JSON.stringify({
  inferred: Object.fromEntries(sentences.map((text) => [text, inferTaskType(text)])),
  notes: Object.fromEntries(sentences.map((text) => [text, parseScopeNotes(text)])),
  collectionsGaps: draftMissingFields({ ...base, taskType: "strategy", collections: true }),
  strategyGaps: draftMissingFields({ ...base, taskType: "strategy", collections: false }),
}));
"""
    )
    inferred, notes = data["inferred"], data["notes"]
    for text in ("给逾期客户做一版催收分案规则", "做一套催收策略，自动外呼并预测回收率", "Set up a collections queue"):
        assert (inferred[text]["taskType"], inferred[text]["confidence"]) == ("strategy", "high"), text
        assert notes[text]["collections"] is True
    assert notes["做一套催收策略，自动外呼并预测回收率"]["unsupported"] == [
        "联系客户（外呼、短信、话术）",
        "自动优化催收动作或预测、保证回收效果",
    ]
    # A profit field and a C-card model are not collections requests.
    assert inferred["收益测算，催收成本率 3%"]["taskType"] == "vintage"
    assert notes["收益测算，催收成本率 3%"]["collections"] is False
    assert inferred["做一个催收评分卡模型"]["taskType"] == "modeling"
    # Anti-fraud has no task type: the sentence's own work decides it.
    assert inferred["做一个反欺诈模型"]["taskType"] == "modeling"
    assert inferred["反欺诈规则"]["taskType"] == "strategy"
    assert notes["反欺诈规则"]["antiFraud"] is True and notes["反欺诈规则"]["collections"] is False
    # Collections opens the strategy task's workspace: no approval/bad-rate limits needed.
    assert data["collectionsGaps"] == []
    assert data["strategyGaps"] == ["strategyLimits"]


def test_collections_card_points_to_the_workspace_and_asks_for_no_limits():
    data = _run_module(
        _CONTROLLER_HARNESS
        + """
els.homeComposerInput.value = "做一套催收分案规则，自动外呼";
await els.homeComposer.fire("submit");
const collections = { draft: { ...composer.getDraft() }, html: els.homeDraft.innerHTML };
els.homeComposerInput.value = "反欺诈规则";
await els.homeDraft.fire("click", { target: target("[data-home-draft-back]", {}) });
await els.homeComposer.fire("submit");
process.stdout.write(JSON.stringify({ collections, fraudHtml: els.homeDraft.innerHTML }));
"""
    )
    draft, html = data["collections"]["draft"], data["collections"]["html"]
    assert draft["taskType"] == "strategy" and draft["collections"] is True
    assert "催收参考工作台" in html
    assert "现在做不到：联系客户（外呼、短信、话术）。" in html
    assert 'data-home-draft-limit="minApprovalRate"' not in html
    assert "通过率下限或坏率上限" not in html
    assert "外部数据与事件特征" in data["fraudHtml"]


def test_home_collections_needs_a_strong_intent():
    data = _run_module(
        """
import { inferTaskType, parseScopeNotes } from "./marvis/static/js/home-intent.js";
const texts = ["训练一个预测催收回款率的模型", "贷后管理口径下做vintage分析", "feature collection for the model", "贷后管理怎么分案"];
process.stdout.write(JSON.stringify(texts.map((text) => ({ text, type: inferTaskType(text).taskType, notes: parseScopeNotes(text) }))));
"""
    )
    assert [(row["type"], row["notes"]["collections"]) for row in data] == [
        ("modeling", False), ("vintage", False), (None, False), (None, False),
    ]
