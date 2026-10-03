import { escapeHtml as esc } from "./ui-utils.js";
import { field, select, button } from "./collection-workspace-view.js";
export const channelChoices = (name) =>
  [
    ["sms", "短信"],
    ["phone", "电话"],
    ["in_app", "应用内"],
    ["letter", "信函"],
  ]
    .map(
      ([v, l]) =>
        `<label class="collection-check"><input type="checkbox" name="${name}" value="${v}">${l}</label>`,
    )
    .join("");
const kinds = [
  ["number", "数值"],
  ["text", "文本"],
  ["boolean", "布尔"],
];
const actionKinds = [
  ["contact", "本地参考触达"],
  ["review", "转人工队列"],
  ["hold", "暂缓"],
];
const need = "required";
export function featureHtml() {
  return `<div data-collection-feature class="collection-grid">${field("feature_name", "字段名", "text", 'maxlength="128"')}${select("feature_type", "字段值类型", [...kinds, ["unknown", "未知（空值）"]], "")}${field("feature_value", "明确字段值（布尔填 true / false；未知留空）")}${button("remove-row", "移除此字段")}</div>`;
}
export function featureSection() {
  return `<details class="collection-rule-section"><summary>本案分层字段声明</summary><p class="collection-note">逐项声明用于规则的业务字段与数值，未知须明确选择。逾期天数等业务口径不由期初余额自动推断；本次时点和来源声明与批次一同冻结。</p><div data-collection-features></div>${button("add-feature", "添加分层字段")}</details>`;
}
export function conditionHtml() {
  return `<div data-collection-condition class="collection-grid">${field("condition_field", "业务字段名", "text", 'required maxlength="128"')}${select("condition_type", "比较字段类型", kinds)}${select(
    "condition_operator",
    "比较方式",
    [
      ["<", "小于"],
      ["<=", "小于等于"],
      [">", "大于"],
      [">=", "大于等于"],
      ["==", "等于"],
      ["!=", "不等于"],
      ["is_null", "未知 / 空值"],
      ["is_not_null", "已知 / 非空"],
    ],
  )}${field("condition_value", "比较值（布尔填 true / false；空值判断留空）")}${select(
    "condition_missing",
    "字段为空时（空值判断不适用）",
    [
      ["no_match", "不命中该条件"],
      ["match", "命中该条件"],
      ["error", "求值到此条件且缺失时，阻止批次"],
    ],
  )}${button("remove-row", "移除此条件")}</div>`;
}
export function ruleHtml() {
  return `<fieldset data-collection-rule><legend>分层规则</legend><div class="collection-grid">${field("rule_id", "规则编号", "text", 'required pattern="[A-Za-z0-9_.:-]{1,128}"')}${select(
    "rule_combination",
    "条件组合",
    [
      ["all", "全部满足（且）"],
      ["any", "任一满足（或）"],
    ],
  )}</div><div data-collection-conditions>${conditionHtml()}</div>${button("add-condition", "添加条件")}<div class="collection-grid">${select("rule_kind", "命中后的动作", actionKinds)}${field("rule_queue", "目标政策队列（暂缓不填）")}${field("rule_priority", "队列优先级（0—1000，暂缓不填）", "number", 'min="0" max="1000" step="1"')}${select(
    "rule_channel",
    "参考触达渠道",
    [
      ["sms", "短信"],
      ["phone", "电话"],
      ["in_app", "应用内"],
      ["letter", "信函"],
    ],
    "",
  )}${field("rule_cost", "单次预计成本（触达必填，最小单位整数）", "number", 'min="0" max="1000000000000000" step="1"')}</div><div class="collection-actions">${button("rule-up", "上移")}${button("rule-down", "下移")}${button("remove-row", "移除此规则")}</div></fieldset>`;
}
export function queueHtml() {
  return `<fieldset data-collection-extra-queue><legend>附加政策队列</legend><div class="collection-grid">${field("extra_id", "队列编号", "text", need)}${field("extra_batch", "单批容量", "number", 'required min="1" max="10000" step="1"')}${field("extra_active", "活动容量上限", "number", 'required min="1" max="10000" step="1"')}</div><div class="collection-actions">${channelChoices("extra_channel")}</div>${button("remove-row", "移除此队列")}</fieldset>`;
}
export function layeredHtml() {
  return `<details class="collection-rule-section"><summary>分层规则与附加队列（可选）</summary><p class="collection-note">按界面从上到下首命中，命中后不再继续匹配；未命中采用上方默认动作。每条规则支持“且 / 或”条件组；“且”遇到不满足、“或”遇到满足后不再求值后续条件，缺失时报错仅对实际求值的条件生效。所有案件须声明所引用字段，可明确为未知。预览只生成待审批提案；审批后可执行本地参考流程。</p><div data-collection-rules></div>${button("add-rule", "添加分层规则")}<h4>附加政策队列</h4><div data-collection-extra-queues></div>${button("add-queue", "添加政策队列")}</details>`;
}
const required = (value, label) => {
  const s = String(value ?? "").trim();
  if (!s) throw Error(`请填写${label}`);
  return s;
};
const integer = (value, label, min, max) => {
  const s = required(value, label),
    n = Number(s);
  if (!Number.isSafeInteger(n) || n < min || n > max)
    throw Error(`${label}须为 ${min}—${max} 的整数`);
  return n;
};
function typed(type, value) {
  if (type === "unknown") {
    if (String(value ?? "").trim()) throw Error("未知字段不能同时声明值");
    return null;
  }
  const s = required(value, "字段值");
  if (type === "text") return String(value);
  if (type === "boolean") {
    if (!["true", "false"].includes(s))
      throw Error("布尔字段请明确填写 true 或 false");
    return s === "true";
  }
  if (type === "number") {
    const n = Number(s);
    if (!Number.isFinite(n) || Math.abs(n) > Number.MAX_SAFE_INTEGER)
      throw Error("字段须为有限数值，绝对值不得超过安全数值上限");
    return n;
  }
  throw Error("请明确字段类型");
}
export function collectFeatures(rows = []) {
  const result = {};
  for (const v of rows) {
    const name = required(v.feature_name, "字段名");
    if (Object.hasOwn(result, name)) throw Error(`分层字段 ${name} 重复`);
    Object.defineProperty(result, name, {
      value: typed(v.feature_type, v.feature_value),
      enumerable: true,
      writable: true,
      configurable: true,
    });
  }
  return result;
}
export function collectExtraQueues(rows = []) {
  return rows.map(({ values: v, channels }) => ({
    queue_id: required(v.extra_id, "附加队列编号"),
    channels,
    max_batch_actions: integer(v.extra_batch, "附加单批容量", 1, 10000),
    max_active_actions: integer(v.extra_active, "附加活动容量", 1, 10000),
  }));
}
export function collectRules(rows = [], cases = [], policy) {
  if (rows.length > 50) throw Error("每批最多 50 条业务分层规则");
  const fieldTypes = new Map();
  return rows.map(({ values: v, conditions }, index) => {
    if (!["all", "any"].includes(v.rule_combination))
      throw Error(`规则 ${index + 1} 请明确条件组合`);
    if (!conditions.length || conditions.length > 20)
      throw Error("每条规则须有 1—20 个条件");
    const actionKind = required(v.rule_kind, "规则动作");
    if (!actionKinds.some(([k]) => k === actionKind))
      throw Error("请明确规则动作");
    const action = {
      kind: actionKind,
      queue_id:
        actionKind === "hold" ? null : required(v.rule_queue, "目标队列"),
      priority:
        actionKind === "hold"
          ? 0
          : integer(v.rule_priority, "动作优先级", 0, 1000),
      channel:
        actionKind === "contact" ? required(v.rule_channel, "触达渠道") : null,
      estimated_cost_minor:
        actionKind === "contact"
          ? integer(v.rule_cost, "预计成本", 0, 1e15)
          : null,
    };
    const queue = policy.queues.find((q) => q.queue_id === action.queue_id);
    if (actionKind !== "hold" && !queue)
      throw Error("规则目标队列须在业务政策中声明");
    if (actionKind === "contact" && !queue.channels.includes(action.channel))
      throw Error("规则渠道须在目标政策队列中明确允许");
    return {
      rule_id: required(v.rule_id, "规则编号"),
      combination: v.rule_combination,
      action,
      conditions: conditions.map((c) => {
        const fieldName = required(c.condition_field, "条件字段名"),
          operator = required(c.condition_operator, "比较方式"),
          valueType = required(c.condition_type, "字段类型");
        if (!kinds.some(([k]) => k === valueType))
          throw Error("请选择条件字段类型");
        if (
          ![
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
            "is_null",
            "is_not_null",
          ].includes(operator)
        )
          throw Error("请选择比较方式");
        if (["<", "<=", ">", ">="].includes(operator) && valueType !== "number")
          throw Error("大小比较只适用于数值字段");
        if (fieldTypes.has(fieldName) && fieldTypes.get(fieldName) !== valueType)
          throw Error(`字段 ${fieldName} 不可同时声明为不同类型`);
        fieldTypes.set(fieldName, valueType);
        const nullOp = ["is_null", "is_not_null"].includes(operator);
        if (nullOp) {
          if (String(c.condition_value ?? "").trim() || c.condition_missing)
            throw Error("空值判断不适用比较值和缺失处理方式");
        } else if (!["no_match", "match", "error"].includes(c.condition_missing))
          throw Error("请明确字段缺失时的处理方式");
        for (const record of cases) {
          if (!Object.hasOwn(record.features, fieldName))
            throw Error(`${record.case_id} 尚未声明分层字段 ${fieldName}`);
          const value = record.features[fieldName];
          if (
            value !== null &&
            typeof value !==
              { text: "string", number: "number", boolean: "boolean" }[
                valueType
              ]
          )
            throw Error(`${record.case_id} 的 ${fieldName} 类型与规则不一致`);
        }
        return {
          field: fieldName,
          value_type: valueType,
          operator,
          value: nullOp ? null : typed(valueType, c.condition_value),
          missing: nullOp ? null : c.condition_missing,
        };
      }),
    };
  });
}
