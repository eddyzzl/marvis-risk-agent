import { escapeHtml as esc } from "./ui-utils.js";
import {
  input,
  select,
  button,
  jsonDetails,
} from "./production-governance-view.js";
export const selectOptions = (items) =>
  '<option value="">请选择</option>' +
  items
    .map((i) => `<option value="${esc(i.id)}">${esc(i.label || i.id)}</option>`)
    .join("");
export function packageBuildHtml(tasks) {
  const choices = tasks.map((t) => ({ id: t.id, label: t.model_name || t.id }));
  return `<form data-production-form="build"><h3>构建冻结决策包</h3><p class="production-note">先选择模型决策或纯规则决策，再选择对应的物化策略。模型决策另需认证训练产物；原始字段的请求类型和空值规则由你明确声明，不从样本类型或缺失率推定。</p>${select(
    "package_kind",
    "决策包类型",
    [
      { id: "model", label: "模型与规则" },
      { id: "rule_only", label: "纯规则（无模型分数）" },
    ],
    "required",
  )}<div class="production-grid" data-production-model-source hidden>${select("model_task_id", "模型来源任务", choices, "disabled")}${select("model_artifact_id", "已完成训练的当前模型产物", [], "disabled")}</div><div class="production-grid">${select("strategy_task_id", "策略来源任务", choices, "required")}${select("strategy_id", "已物化策略版本", [], "required")}</div>${button("readiness", "校验构包来源与原始字段")}<div data-production-readiness></div></form>`;
}
export function readinessHtml(r, kind = "model") {
  const ruleOnly = kind === "rule_only";
  if (r.state !== "authenticated")
    return `<p class="production-error">当前来源未通过认证，不能构包：${esc((r.reason_codes || []).join("、"))}</p>${jsonDetails("查看来源缺口", r)}`;
  return `<p class="production-note">${ruleOnly ? "策略表达式来源已认证。纯规则包不含模型和分数；请声明请求字段、失败动作与超时。" : "原生训练来源已认证。请复核以下字段、分数产品、请求失败动作和超时口径。"}</p>${jsonDetails(ruleOnly ? "策略来源认证证据" : "来源与预处理认证证据", r)}<h4>原始请求字段</h4><div class="production-grid">${r.raw_requirements.length ? rawFieldsHtml(r.raw_requirements) : '<p class="production-note">此策略不引用请求字段；请求合同为空。</p>'}</div><div class="production-grid">${
    ruleOnly
      ? ""
      : `${select(
          "score_product",
          "分数产品",
          r.score_products
            .filter((s) => s.available)
            .map((s) => ({
              id: s.value,
              label:
                {
                  raw_pd: "原始 PD",
                  calibrated_pd: "校准 PD",
                  scorecard_points: "评分卡分数",
                }[s.value] || s.value,
            })),
          "required",
        )}${input("score_field", "策略中的模型分数字段（明确绑定）", "text", 'required maxlength="160" list="productionScoreFields"')}<datalist id="productionScoreFields">${r.score_field_candidates.map((name) => `<option value="${esc(name)}"></option>`).join("")}</datalist>`
  }${input("decision_node", "决策节点标识（英文）", "text", 'required pattern="[a-z][a-z0-9_-]{0,63}"')}${input("timeout_seconds", "执行超时（秒）", "number", 'required min="1" max="60" step="1"')}${select(
    "failure_action",
    "请求失败时动作",
    [
      { id: "review", label: "转人工" },
      { id: "reject", label: "拒绝" },
    ],
    "required",
  )}</div>${ruleOnly ? "" : `<p class="production-note">分数字段候选仅表示策略尚未绑定的字段。请明确哪一个接收模型分数；其余字段须声明外部输入合同。</p><div class="production-grid" data-production-extra-fields></div>`}<button type="submit" class="button compact primary">确认合同并构建冻结包</button>`;
}
export function collectPackage(
  values,
  { readiness, strategies, checkedSignature },
) {
  if (readiness?.state !== "authenticated") throw new Error("请先校验构包来源");
  if (!["model", "rule_only"].includes(values.package_kind))
    throw new Error("请选择决策包类型");
  const ruleOnly = values.package_kind === "rule_only";
  const strategy = strategies.find((s) => s.strategy_id === values.strategy_id);
  if (!strategy || packageSignature(values, strategy) !== checkedSignature)
    throw new Error("来源已变化，请重新校验");
  const timeout = Number(values.timeout_seconds);
  if (
    !values.timeout_seconds ||
    !Number.isInteger(timeout) ||
    timeout < 1 ||
    timeout > 60
  )
    throw new Error("超时须为 1 至 60 秒");
  if (
    !ruleOnly &&
    (!readiness.score_products.some(
      (s) => s.available && s.value === values.score_product,
    ) ||
      !values.score_field?.trim() ||
      readiness.raw_requirements.some((f) => f.name === values.score_field))
  )
    throw new Error("请选择已认证的分数产品和策略分数字段");
  if (
    !/^[a-z][a-z0-9_-]{0,63}$/.test(values.decision_node) ||
    !["review", "reject"].includes(values.failure_action)
  )
    throw new Error("请明确决策节点与失败动作");
  return {
    ...(ruleOnly
      ? { package_kind: "rule_only" }
      : {
          model_artifact_id: values.model_artifact_id,
          score_product: values.score_product,
          score_field: values.score_field,
        }),
    strategy_id: strategy.strategy_id,
    strategy_version: strategy.version,
    decision_node: values.decision_node,
    timeout_seconds: timeout,
    failure_action: values.failure_action,
    raw_schema: packageRawFields(
      readiness,
      ruleOnly ? null : values.score_field,
    ).map((f, i) => {
      const type = values[`raw_type_${i}`],
        nullable = values[`raw_nullable_${i}`];
      if (
        !["number", "integer", "string", "boolean"].includes(type) ||
        !["true", "false"].includes(nullable)
      )
        throw new Error(`请明确 ${f.name} 的请求类型和空值规则`);
      return { name: f.name, type, nullable: nullable === "true" };
    }),
  };
}

export const packageSignature = (values, strategy) =>
  JSON.stringify([
    values.package_kind,
    values.package_kind === "rule_only" ? null : values.model_artifact_id,
    strategy.strategy_id,
    strategy.version,
  ]);

export function packageRawFields(readiness, scoreField) {
  const base = readiness.raw_requirements;
  return [
    ...base,
    ...(readiness.unbound_strategy_fields || [])
      .filter(
        (name) => name !== scoreField && !base.some((f) => f.name === name),
      )
      .map((name) => ({ name })),
  ];
}
export function rawFieldsHtml(fields, offset = 0) {
  return fields
    .map(
      (f, i) =>
        `<fieldset class="production-field"><legend>${esc(f.name)}</legend>${select(
          `raw_type_${i + offset}`,
          "请求数据类型",
          [
            { id: "number", label: "数值" },
            { id: "integer", label: "整数" },
            { id: "string", label: "文本" },
            { id: "boolean", label: "布尔值" },
          ],
          "required",
        )}${select(
          `raw_nullable_${i + offset}`,
          "允许 null 空值",
          [
            { id: "false", label: "不允许" },
            { id: "true", label: "允许" },
          ],
          "required",
        )}</fieldset>`,
    )
    .join("");
}
