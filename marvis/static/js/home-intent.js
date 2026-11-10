// Home page intent (2026-10-14 redesign, batch 5): read one sentence and guess
// which of the seven workflows it starts. Deterministic and offline on purpose:
// the guess only pre-selects the task type in the creation card, where the user
// confirms or changes it; it never runs anything. Ambiguous sentences come back
// with low confidence so the card asks instead of guessing.

import { validationModelsReady } from "./home-validation-models.js";

// Only a strong collections intent counts (same rule as the backend's
// turn_handlers/collections_guidance.py): 催收 next to collections work, or a
// verb asking for it; in English "collections <work>" or "debt collection".
// A model, a recovery-rate target or a vintage/feature reading is other work.
const COLLECTIONS_WORDS = new RegExp([
  "催收.{0,4}(?:策略|分案|分配|排队|批次|案件|名单|队列|动作|计划|方案|工作台|流程|规则|外呼|话术|频次)",
  "(?:做|设计|制定|优化|搭建|安排|开展|处理)(?:一套|一版|一下|个)?.{0,6}催收(?!成本|费用|费率|评分|模型|卡|回款率)",
  "(?<![A-Za-z])(?:debt\\s+collections?|collections\\s+(?:strateg(?:y|ies)|polic(?:y|ies)|queues?|batch(?:es)?|cases?"
    + "|workflows?|campaigns?|actions?|assignments?|plans?|workspace))(?![A-Za-z])",
].join("|"), "i");
const COLLECTIONS_OTHER_WORK = /模型|评分卡|[Cc]\s?卡|回款率|催收成本|费率|vintage|vtg|账龄|特征|变量|(?<![A-Za-z])(?:model|models|scorecard|features?|vintage)(?![A-Za-z])/i;
const isCollections = (value) => COLLECTIONS_WORDS.test(value) && !COLLECTIONS_OTHER_WORK.test(value);

const TYPE_PATTERNS = {
  validation: [
    // 验证集 / 验证样本 are modeling data splits, not a validation task.
    [/验证(?!集|样本)|复核模型|审查模型/, 3],
    [/pmml|notebook|复现|分数一致/i, 3],
    [/验证报告|出报告/, 1],
  ],
  modeling: [
    [/建模|训练|模型开发|开发.{0,6}模型|调参/, 3],
    [/(做|建|搭|开发|训练)(一个|个|一版|一张)?.{0,8}(模型|评分卡|[ABCT] ?卡)/i, 3],
    [/lightgbm|xgboost|lgb|xgb|逻辑回归/i, 2],
    // Exporting a PMML is the last modeling step; reviewing one is validation.
    [/(导出|生成|输出) ?pmml/i, 3],
    [/oot|ks ?(不低于|≥|>=|大于)/i, 1],
  ],
  feature_analysis: [
    // "征信特征表" names a table to join, not a feature analysis (eval S6).
    [/特征(?!表)|变量|数据源/, 3],
    [/\biv\b|分箱|psi|覆盖率|共线/i, 2],
    [/值不值得用|有没有用/, 1],
  ],
  data_join: [
    [/拼接|拼成|拼起来|合并|关联|join|多表|主表/i, 3],
    // Two tables put together is the join itself (eval S6).
    [/[两几多]张表|主表|副表|表.{0,12}(拼|关联|合并|join)|(拼|关联|合并|join).{0,12}表/i, 3],
    [/数据处理|清洗|去重|宽表/, 3],
  ],
  strategy: [
    [/策略|规则|准入|cutoff|阈值/i, 3],
    [/通过率|拒绝率|额度|决策树/, 2],
    // D-7: collections lives under the strategy task. 催收成本率 is a profit
    // field and 催收评分卡 (C 卡) is a model, so those stay out.
    [{ test: isCollections }, 3],
  ],
  vintage: [
    [/vintage|vtg|账龄|收益测算|风险分析/i, 3],
    [/不良|坏账曲线|年化/, 2],
  ],
  portfolio: [
    [/组合|在贷|存量|集中度|预期损失|\becl\b|迁徙/i, 3],
  ],
};

// Requests the collections workspace or anti-fraud tooling cannot meet; the
// card says so plainly instead of promising them (D-7, 5.7 anti-fraud).
const COLLECTIONS_UNSUPPORTED = [
  [/外呼|打电话|拨打|短信|联系客户|触达客户|话术/, "联系客户（外呼、短信、话术）"],
  [/实时|对接.{0,6}(系统|平台)|上线催收/, "对接外部催收系统或实时调度"],
  [/(优化|最优|自动).{0,6}催收|(预测|提升|保证).{0,4}回收|回收(率)?预测|催收效果/, "自动优化催收动作或预测、保证回收效果"],
];
const ANTI_FRAUD_WORDS = /反欺诈|欺诈|骗贷|\bfraud\b|设备指纹|关系图谱|团伙/i;

export function parseScopeNotes(text) {
  const value = String(text || "");
  const collections = isCollections(value);
  return {
    collections,
    antiFraud: ANTI_FRAUD_WORDS.test(value),
    unsupported: collections
      ? COLLECTIONS_UNSUPPORTED.filter(([pattern]) => pattern.test(value)).map(([, label]) => label)
      : [],
  };
}

const MULTI_MODEL = /([2-9两三四五六七八九十]|多)\s*个模型|这些模型|几个模型|批次|批量|一批模型|全部模型|所有模型|各个模型/;

export function inferTaskType(text) {
  const value = String(text || "");
  const scores = Object.entries(TYPE_PATTERNS)
    .map(([taskType, patterns]) => [
      taskType,
      patterns.reduce((total, [pattern, weight]) => total + (pattern.test(value) ? weight : 0), 0),
    ])
    .filter(([, score]) => score > 0)
    .sort((left, right) => right[1] - left[1]);
  if (!scores.length) return { taskType: null, confidence: "none", candidates: [], multiModel: false };
  const [[taskType, top], second] = scores;
  // A clear winner leads by at least 2; otherwise the card asks.
  const confidence = !second || top - second[1] >= 2 ? "high" : "low";
  return {
    taskType,
    confidence,
    candidates: scores.map(([type]) => type),
    multiModel: taskType === "validation" && MULTI_MODEL.test(value),
  };
}

const NAME_SUFFIX = {
  validation: "模型验证",
  modeling: "建模",
  feature_analysis: "特征分析",
  data_join: "数据处理",
  strategy: "策略",
  vintage: "风险分析",
  portfolio: "组合分析",
};
const LEADING_FILLERS = /^(请|帮我|帮忙|麻烦|我想|我要|想要|想|能不能|可以)+/;
// Prefixes are stripped only as whole phrases: 用/拿/基于/根据 need a following
// 这/那/上面 ("用这份样本", not "用户分层"), and the one-character verbs that
// start ordinary words (建模, 建行, 来源, 搭配) need a measure word after them.
const LEADING_SOURCE = /^(?:把|(?:用|拿|基于|根据)(?=这|那|上面))(?:(?:这|那|上面的)(?:几张|几份|几批|几个|份|批|些|个|张)?(?:样本|数据|材料|文件|表)?)?(?:来)?/;
const QUANTIFIER = "(?:一个|个|一下|一次|一版|一张)";
const LEADING_VERB = new RegExp(
  `^(?:(?:做|跑)${QUANTIFIER}?|(?:建|搭|来)${QUANTIFIER}|(?:开发|训练|验证|分析|看看)${QUANTIFIER}?)`,
);
const LEADING_DEMONSTRATIVE = /^(这|那)(几个|几张|几份|几批|[两三四五六七八九十多]个|个|些|批|份|张)?/;
// "这是我们新上的行为评分卡" names the model after the copula and the owner.
const LEADING_COPULA = /^是?(?:我们|我司|咱们|公司)?(?:的)?(?:新上的|新做的|刚上的|刚上线的|上线的|新开发的|最新的)?/;
const TRAILING_REPORT = /(并|再|然后)?(出|生成|写)(一份)?(验证)?报告$|(并|再|然后)(回测|上线|部署)$/;
const TRAILING_QUESTION = /(值不值得用|值不值得接入|有没有用|怎么样|行不行|好不好)$/;

// Target phrases ("通过率 60% 以上的", "KS 不低于 0.30") say how good, not what;
// they are dropped before naming.
const TARGET_PHRASE = /(通过率|坏账率?|坏率|不良率?|逾期率|ks|auc|psi|iv)\s*(要|需|在|控制在|不低于|不高于|不超过|大于|小于|高于|低于|≥|≤|>=|<=|>|<)?\s*\d+(?:\.\d+)?\s*%?\s*(以上|以内|以下|及以上|左右)?的?/gi;

// A short, editable task name drafted from the sentence's first clause that
// names something; the type suffix is added only when the name does not
// already say the workflow.
export function draftTaskName(text, taskType = "") {
  const clause = String(text || "")
    .replace(/\s+/g, " ")
    .trim()
    .split(/[，,。；;！!？?\n]/)
    .map((part) => part.replace(TARGET_PHRASE, "").trim())
    .find(Boolean) || "";
  let name = clause
    .replace(LEADING_FILLERS, "")
    .replace(LEADING_SOURCE, "")
    .replace(LEADING_VERB, "")
    .replace(LEADING_DEMONSTRATIVE, "")
    .replace(LEADING_COPULA, "")
    .replace(TRAILING_REPORT, "")
    .replace(TRAILING_QUESTION, "")
    .trim();
  if (name.length > 24) name = name.slice(0, 24).trim();
  const suffix = NAME_SUFFIX[taskType] || "";
  if (!name) return suffix ? `新${suffix}任务` : "新任务";
  if (!suffix) return name;
  // A validation task's name field is the model's name ("行为评分卡").
  // A bare "模型" still reads as 模型验证.
  if (taskType === "validation" && !/验证/.test(name) && (/卡|评分/.test(name) || (/模型/.test(name) && name.length > 2))) {
    return name;
  }
  // "…模型" + 模型验证 → "…模型验证"; "新特征" + 特征分析 → "新特征分析".
  if (name.length >= 2 && suffix.startsWith(name.slice(-2))) return `${name}${suffix.slice(2)}`;
  if (inferTaskType(name).taskType === taskType) return name;
  return `${name} ${suffix}`.slice(0, 32);
}

// Example sentences above the home composer (D7): clicking one only fills the
// input; the task type is set from the example, not inferred. V3 UI-7: a
// larger pool, of which the home page shows HOME_EXAMPLE_COUNT at random.
export const HOME_EXAMPLES = Object.freeze([
  { text: "验证这几个模型并出报告", taskType: "validation" },
  { text: "这批新特征值不值得用？", taskType: "feature_analysis" },
  { text: "通过率 60% 以上，坏账 5% 以内的准入策略", taskType: "strategy" },
  { text: "用这份样本做一个贷前 A 卡，KS 不低于 0.30", taskType: "modeling" },
  { text: "建一个 LGB 评分模型，OOT 取最近两个月", taskType: "modeling" },
  { text: "把主表和三方数据按 apply_id 拼接", taskType: "data_join" },
  { text: "看看各月放款的 Vintage 坏账曲线", taskType: "vintage" },
  { text: "按放款月看 MOB3 的 Vintage 逾期表现", taskType: "vintage" },
  { text: "看一下在贷组合的集中度和迁徙", taskType: "portfolio" },
  { text: "复核这份 PMML 和 Notebook 的分数一致性", taskType: "validation" },
  { text: "算一下这批特征的 IV 和 PSI", taskType: "feature_analysis" },
  { text: "给这个评分定一个审批阈值并回测策略", taskType: "strategy" },
  { text: "给逾期客户做一版催收分案规则", taskType: "strategy" },
  { text: "验证这个贷前模型的效果和稳定性", taskType: "validation" },
  { text: "复核这个模型的 OOT 表现并出验证报告", taskType: "validation" },
  { text: "验证这个 PMML 模型并生成 Word 报告", taskType: "validation" },
  { text: "分析这批特征的缺失率和覆盖率", taskType: "feature_analysis" },
  { text: "检查特征分箱的单调性和区分能力", taskType: "feature_analysis" },
  { text: "筛选稳定、有效且低共线的入模特征", taskType: "feature_analysis" },
  { text: "回测不同审批阈值下的通过率和坏账率策略", taskType: "strategy" },
  { text: "做一版控制坏账率的准入策略", taskType: "strategy" },
  { text: "比较几套准入规则并选出候选策略", taskType: "strategy" },
  { text: "训练一个 LightGBM 模型并比较训练集和 OOT", taskType: "modeling" },
  { text: "用逻辑回归开发一个评分卡模型", taskType: "modeling" },
  { text: "建模时比较 LGB 和逻辑回归的稳定性", taskType: "modeling" },
  { text: "检查主表和特征表的拼接命中率", taskType: "data_join" },
  { text: "按客户号合并多张表并检查重复键", taskType: "data_join" },
  { text: "清洗主表并拼成建模宽表", taskType: "data_join" },
  { text: "把申请表和表现表按申请编号关联", taskType: "data_join" },
  { text: "做一次标准 Vintage 风险分析", taskType: "vintage" },
  { text: "比较不同放款月份的 Vintage 表现", taskType: "vintage" },
  { text: "分析这份放款面板的账龄坏账曲线", taskType: "vintage" },
  { text: "分析存量组合的逾期迁徙率", taskType: "portfolio" },
  { text: "检查在贷组合的客户和渠道集中度", taskType: "portfolio" },
  { text: "测算存量组合的预期损失", taskType: "portfolio" },
  { text: "比较不同月份的存量组合风险分布", taskType: "portfolio" },
  { text: "批量验证这 3 个模型并出汇总报告", taskType: "validation" },
]);

export const HOME_EXAMPLE_COUNT = 5;

// Indices into HOME_EXAMPLES: a random pick of `count`, one per task type.
export function pickHomeExamples(count = HOME_EXAMPLE_COUNT, random = Math.random) {
  const order = HOME_EXAMPLES.map((_, index) => index);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const seen = new Set();
  const picked = [];
  for (const index of order) {
    const type = HOME_EXAMPLES[index].taskType;
    if (seen.has(type)) continue;
    seen.add(type);
    picked.push(index);
    if (picked.length >= count) break;
  }
  return picked;
}

// Starter sentence for a work icon: fills the input without opening a form.
export function homeStarterText(taskType, label = "", locale = "zh") {
  if (locale === "en") {
    const starters = {
      data_join: "Join the main table and feature tables into a modeling dataset",
      feature_analysis: "Check whether these new features are worth using",
      vintage: "Run a standard Vintage analysis of bad-rate curves by origination month",
      portfolio: "Analyze migration and concentration in the existing portfolio",
      modeling: "Build a scoring model using this sample",
      validation: "Validate this model and write a report",
      strategy: "Develop an approval strategy and backtest it",
    };
    return starters[taskType] || (label ? `Start a ${label} task` : "");
  }
  const starters = {
    data_join: "把主表和特征表拼成建模宽表",
    feature_analysis: "看看这批新特征值不值得用",
    vintage: "做一次标准 Vintage，看各月放款的坏账曲线",
    portfolio: "分析一下存量组合的迁徙和集中度",
    modeling: "用这份样本做一个评分模型",
    validation: "验证这个模型并出报告",
    strategy: "做一版准入策略并回测",
  };
  return starters[taskType] || (label ? `开始一个${label}任务` : "");
}

// Fill-in templates for the seven work icons (2026-10-19 user feedback): a
// fuller sentence with the values a user usually changes marked as slots and
// the materials as file slots, so the user edits a few words or attaches a
// file right where the sentence names it. V3 UI-20: no 「」 around values;
// they ended up in task names and requests. Tokens: plain strings, {slot}
// (editable value) and {file, label} (a file slot; the serialised sentence
// names the attached files next to their role so the Agent knows which file
// is which).
export const HOME_TEMPLATES = Object.freeze({
  modeling: [
    "根据这个", { file: "y_sample", label: "Y样本" }, "和这几个", { file: "x_data", label: "X数据", multiple: true },
    "，给我做一个", { slot: "MOB3_30" }, "的", { slot: "A卡" }, "模型，KS 至少 ", { slot: "0.30" },
    "，选 ", { slot: "2026年3月" }, " 之后的样本做 OOT，train/test 按 ", { slot: "7:3" }, " 分，算法用 ", { slot: "LGB" }, "。",
  ],
  validation: [
    "帮我验证", { slot: "贷前A卡" }, "模型，材料是 ", { file: "notebook", label: "Notebook", multiple: true }, "、",
    { file: "pmml", label: "PMML", multiple: true }, "、", { file: "sample", label: "样本", multiple: true }, "和",
    { file: "dictionary", label: "数据字典", multiple: true },
    "，验证人员是 ", { slot: "张三", key: "validator" }, "，出一份 Word 验证报告。",
  ],
  feature_analysis: [
    "分析这份", { file: "sample", label: "样本" }, "（附", { file: "dictionary", label: "数据字典" }, "）里的特征，目标列是 ",
    { slot: "bad_flag" }, "，看 IV、KS，再按 ", { slot: "apply_month" }, " 看 PSI 稳定性，筛出可以入模的特征。",
  ],
  data_join: [
    "把", { file: "main_table", label: "样本主表" }, "和这些", { file: "feature_tables", label: "特征表", multiple: true },
    "按 ", { slot: "客户号" }, " 拼成建模宽表，目标列是 ", { slot: "bad_flag" }, "，检查命中率和重复键。",
  ],
  vintage: [
    "用这份", { file: "panel", label: "放款面板" }, "做", { slot: "标准 Vintage" }, "，cohort 按 ", { slot: "放款月" },
    "，坏账口径是 ", { slot: "当期新增" }, "，看看哪个月放款的资产质量最差。",
  ],
  portfolio: [
    "分析这份", { file: "snapshot", label: "存量快照" }, "的迁徙和集中度，快照月是 ", { slot: "snapshot_month" },
    "，逾期桶是 ", { slot: "bucket" }, "，余额是 ", { slot: "balance" }, "，损失态是 ", { slot: "M3+" }, "。",
  ],
  strategy: [
    "用这份", { file: "sample", label: "样本" }, "做一套", { slot: "贷前准入" }, "策略，", { slot: "score" },
    " 越高风险越高，通过率不低于 ", { slot: "60%" }, "，通过客群坏账不超过 ", { slot: "5%" }, "。",
  ],
});

// Full sentences preserve natural English word order. File roles keep their
// source identifier so attachments remain bound when the UI language changes.
const HOME_TEMPLATES_EN = Object.freeze({
  modeling: [
    "Using this ", { file: "y_sample", label: "Y sample", role: "Y样本" }, " and these ",
    { file: "x_data", label: "X data", role: "X数据", multiple: true },
    ", build a ", { slot: "MOB3_30" }, " ", { slot: "A-card" }, " model with KS at least ", { slot: "0.30" },
    ". Use samples after ", { slot: "March 2026" }, " for OOT, a train/test split of ", { slot: "7:3" },
    ", and the ", { slot: "LGB" }, " algorithm.",
  ],
  validation: [
    "Validate the ", { slot: "Application A-card" }, " model using ", { file: "notebook", label: "Notebook", multiple: true }, ", ",
    { file: "pmml", label: "PMML", multiple: true }, ", ", { file: "sample", label: "Sample", role: "样本", multiple: true }, " and ",
    { file: "dictionary", label: "Data dictionary", role: "数据字典", multiple: true }, ". The validator is ",
    { slot: "Jane Doe", key: "validator" }, ". Write a Word validation report.",
  ],
  feature_analysis: [
    "Analyze features in this ", { file: "sample", label: "Sample", role: "样本" }, " with its ",
    { file: "dictionary", label: "Data dictionary", role: "数据字典" }, ". The target column is ", { slot: "bad_flag" },
    ". Check IV and KS, assess PSI stability by ", { slot: "apply_month" }, ", and select features for modeling.",
  ],
  data_join: [
    "Join the ", { file: "main_table", label: "Main sample table", role: "样本主表" }, " and these ",
    { file: "feature_tables", label: "Feature tables", role: "特征表", multiple: true }, " on ", { slot: "customer_id" },
    " into a modeling dataset. The target column is ", { slot: "bad_flag" }, ". Check match rates and duplicate keys.",
  ],
  vintage: [
    "Use this ", { file: "panel", label: "Origination panel", role: "放款面板" }, " for ", { slot: "Standard Vintage" },
    " analysis, with cohorts by ", { slot: "origination_month" }, " and a bad-debt definition of ",
    { slot: "new defaults in the current period" }, ". Identify the origination month with the worst asset quality.",
  ],
  portfolio: [
    "Analyze migration and concentration in this ", { file: "snapshot", label: "Portfolio snapshot", role: "存量快照" },
    ". The snapshot month is ", { slot: "snapshot_month" }, ", the delinquency bucket is ", { slot: "bucket" },
    ", the balance is ", { slot: "balance" }, ", and the loss state is ", { slot: "M3+" }, ".",
  ],
  strategy: [
    "Use this ", { file: "sample", label: "Sample", role: "样本" }, " to develop an ", { slot: "application approval" },
    " strategy. Higher ", { slot: "score" }, " means higher risk. Keep the approval rate at least ", { slot: "60%" },
    " and the approved population's bad rate at most ", { slot: "5%" }, ".",
  ],
});

export function homeTemplateTokens(taskType, locale = "zh") {
  return (locale === "en" ? HOME_TEMPLATES_EN : HOME_TEMPLATES)[taskType] || null;
}

// Task name for a template sentence, from its value slots in order (the
// first-clause heuristic would name the file slot instead).
const HOME_TEMPLATE_NAMES = Object.freeze({
  modeling: ([target, card]) => `${target} ${card}模型`,
  validation: ([model]) => model,
  feature_analysis: ([target]) => `${target} 特征分析`,
  data_join: ([key]) => `按${key}拼接宽表`,
  vintage: ([analysis]) => analysis,
  portfolio: () => "存量组合迁徙与集中度",
  strategy: ([kind]) => `${kind}策略`,
});

export function homeTemplateName(taskType, slots = [], locale = "zh") {
  const englishNames = {
    modeling: ([target, card]) => `${target} ${card} model`,
    validation: ([model]) => model,
    feature_analysis: ([target]) => `${target} feature analysis`,
    data_join: ([key]) => `Join dataset on ${key}`,
    vintage: ([analysis]) => analysis,
    portfolio: () => "Portfolio migration and concentration",
    strategy: ([kind]) => `${kind} strategy`,
  };
  const builder = (locale === "en" ? englishNames : HOME_TEMPLATE_NAMES)[taskType];
  const values = slots.map((value) => String(value || "").trim());
  if (!builder || values.some((value) => !value)) return "";
  return builder(values).replace(/\s+/g, " ").trim().slice(0, 32);
}

// Percent limits the user stated for a strategy ("通过率 60% 以上，坏账 5% 以内").
// Only explicit "<metric> … N%" pairs count; the card shows them for review and
// asks when neither is present (full strategy development needs one).
const APPROVAL_RATE = /(?:通过率|approval rate)[^\d%，,。；;.]{0,24}?(\d+(?:\.\d+)?)\s*%/i;
const BAD_RATE = /(?:坏账率?|坏率|不良率?|逾期率|bad rate)[^\d%，,。；;.]{0,24}?(\d+(?:\.\d+)?)\s*%/i;

export function parseStrategyLimits(text) {
  const value = String(text || "");
  const approval = value.match(APPROVAL_RATE);
  const bad = value.match(BAD_RATE);
  return {
    minApprovalRate: approval ? approval[1] : "",
    maxBadRate: bad ? bad[1] : "",
  };
}

export function strategyLimitValue(percent) {
  const raw = String(percent ?? "").trim();
  if (!raw) return null;
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 && value <= 100 ? value / 100 : Number.NaN;
}

// Binary-target recipes the card offers for manual modeling; regression and
// multiclass stay in the create dialog (高级设置).
export const HOME_MODEL_ALGORITHMS = Object.freeze([
  { id: "lgb", label: "LightGBM", pattern: /lightgbm|\blgb\b/i },
  { id: "xgb", label: "XGBoost", pattern: /xgboost|\bxgb\b/i },
  { id: "catboost", label: "CatBoost", pattern: /catboost/i },
  { id: "lr", label: "逻辑回归", pattern: /逻辑回归|\blr\b|logistic/i },
  { id: "scorecard", label: "评分卡", pattern: /评分卡|scorecard/i },
  { id: "mlp", label: "神经网络", pattern: /神经网络|\bmlp\b|\bdnn\b/i },
]);

// "OOT 的 KS 尽量不低于 0.3", "KS 至少 0.30", "KS≥30%": the modeling goal the
// create contract stores as oot_ks_min (a 0-1 fraction). Only a KS stated
// with a lower-bound word counts; anything else stays unset.
const KS_GOAL = /ks\s*(?:尽量|最好|要|需要|需)?\s*(?:不低于|至少|不小于|大于等于|高于|大于|at least|no less than|≥|>=|>)\s*(\d+(?:\.\d+)?)\s*(%?)/i;

export function parseOotKsGoal(text) {
  const match = String(text || "").match(KS_GOAL);
  if (!match) return "";
  let value = Number(match[1]);
  if (match[2] === "%" || value > 1) value /= 100;
  if (!Number.isFinite(value) || value <= 0 || value >= 1) return "";
  return String(Number(value.toFixed(4)));
}

// "验证人员是 张三" / "负责人：李四": the person the sentence names, for the
// card's 验证人员 field. Only a name right after such a role word counts.
const VALIDATOR_NAME = /(?:验证人员|验证人|负责人)\s*(?:是|为|[:：])?\s*([\u4e00-\u9fa5·]{2,4}|[A-Za-z][A-Za-z .'-]{1,30}[A-Za-z])/;
const EN_VALIDATOR_NAME = /\bvalidator\s*(?:is\b|:)?\s*([^,;.\n]{2,40})/i;

export function parseValidatorName(text) {
  const value = String(text || "");
  const match = value.match(VALIDATOR_NAME) || value.match(EN_VALIDATOR_NAME);
  return match ? match[1].trim() : "";
}

// Extra feature metrics a sentence asks for (eval S3: 「看看区分度和稳定性」
// ran without PSI). Only explicit words count; the defaults stay as they are.
const FEATURE_METRIC_WORDS = [
  [/稳定性|稳定度|psi/i, ["psi_month_first", "psi_month_previous"]],
  [/(按|分)\s*(split|样本集|训练集|切分)[^，。]{0,6}psi|psi[^，。]{0,6}(split|样本集)/i, ["psi_split"]],
  [/共线|vif/i, ["vif"]],
  [/头尾|lift/i, ["head_tail_lift"]],
];

export function parseFeatureMetrics(text) {
  const value = String(text || "");
  const metrics = [];
  for (const [pattern, names] of FEATURE_METRIC_WORDS) {
    if (!pattern.test(value)) continue;
    names.forEach((name) => { if (!metrics.includes(name)) metrics.push(name); });
  }
  return metrics;
}

export function parseModelAlgorithms(text) {
  const value = String(text || "");
  return HOME_MODEL_ALGORITHMS.filter((item) => item.pattern.test(value)).map((item) => item.id);
}

// The risk-analysis intake starts exactly one analysis; these are the commands
// its own option buttons send (marvis/agent/risk_analysis_setup.py). A
// sentence that names one is preselected; naming none or several asks.
export const HOME_RISK_ANALYSES = Object.freeze([
  { command: "VTG终值与年化不良", pattern: /vtg|终值|年化不良|周转/i },
  { command: "收益测算", pattern: /收益|利润|盈利/ },
  { command: "标准 Vintage", pattern: /vintage|坏账曲线|账龄|\bmob\b/i },
]);

export function parseRiskAnalysis(text) {
  const value = String(text || "");
  const matched = HOME_RISK_ANALYSES.filter((item) => item.pattern.test(value));
  return matched.length === 1 ? matched[0].command : "";
}

// What each workflow needs before it can be created (mirrors the create
// dialog's contract); the card asks only for what is still missing.
export function draftMissingFields(draft, { deferredMaterials = false } = {}) {
  const missing = [];
  // Two or more models are one batch: each model brings its own name and
  // materials, and the batch is named by date and model count.
  const batch = draft?.taskType === "validation" && Array.isArray(draft?.models) && draft.models.length >= 2;
  if (!draft?.taskType) missing.push("taskType");
  if (!batch && !String(draft?.name || "").trim()) missing.push("name");
  if (!String(draft?.validator || "").trim()) missing.push("validator");
  const needsMaterials = !batch && !(deferredMaterials && draft?.runMode === "agent");
  if (needsMaterials && !(Array.isArray(draft?.files) && draft.files.length)) missing.push("files");
  // Collections runs in the strategy task's own workspace, not a strategy
  // development, so it needs no approval/bad-rate limits.
  if (draft?.taskType === "strategy" && !draft?.collections) {
    const limits = [draft?.strategy?.minApprovalRate, draft?.strategy?.maxBadRate].map(strategyLimitValue);
    if (limits.some((value) => Number.isNaN(value)) || limits.every((value) => value === null)) {
      missing.push("strategyLimits");
    }
  }
  if (draft?.taskType === "vintage" && !draft?.analysis) missing.push("riskAnalysis");
  // Several models need a name and four materials each; a sentence that asks
  // for several models waits until the card holds them.
  if (draft?.taskType === "validation" && (batch ? !validationModelsReady(draft.models) : draft?.multiModel)) {
    missing.push("models");
  }
  if (draft?.taskType === "modeling" && draft?.runMode === "manual"
    && !(Array.isArray(draft?.algorithms) && draft.algorithms.length)) {
    missing.push("algorithms");
  }
  return missing;
}
