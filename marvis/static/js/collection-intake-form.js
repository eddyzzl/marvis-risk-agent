import { escapeHtml as esc } from "./ui-utils.js";
import { button, money } from "./collection-workspace-view.js";
export const field = (name, label, type = "text", extra = "", value = "") =>
  `<label>${esc(label)}<input name="${name}" type="${type}" value="${esc(value)}" ${extra}></label>`;
export const select = (name, label, options, extra = "required") =>
  `<label>${esc(label)}<select name="${name}" ${extra}><option value="">请选择</option>${options.map(([value, text]) => `<option value="${value}">${esc(text)}</option>`).join("")}</select></label>`;
const identity = (name, label) =>
  field(name, label, "text", 'required pattern="[A-Za-z0-9_.:-]{1,128}"');
const integer = (name, label, min = 0, max = 1e15) =>
  field(name, label, "number", `required step="1" min="${min}" max="${max}"`);
const time = (name, label, required = true) =>
  field(
    name,
    `${label}（带时区的 ISO 时间）`,
    "text",
    `${required ? "required" : ""} placeholder="2026-09-28T12:00:00+08:00"`,
  );
const source = () =>
  field(
    "source_description",
    "资料来源与口径说明",
    "text",
    'required maxlength="1000"',
  );
const caseSelect = (cases) =>
  select(
    "case_id",
    "已登记案件",
    cases.map((c) => [
      c.case_id,
      `${c.case_id} · ${money(c.opening_balance_minor, c.unit)}`,
    ]),
  );
const unit = () =>
  `${field("currency", "币种（三位代码）", "text", 'required pattern="[A-Z]{3}" maxlength="3"')}${select(
    "minor_unit_exponent",
    "金额小数位（录入金额为最小货币单位）",
    [
      ["0", "0 位"],
      ["1", "1 位"],
      ["2", "2 位"],
      ["3", "3 位"],
      ["4", "4 位"],
    ],
  )}${field("definition_source", "货币单位定义来源", "text", "required")}`;
const commonNote =
  '<p class="collection-note">此处录入脱敏历史声明。平台登记内容及来源指纹，不能据此独立证明外部事实。金额均为最小货币单位整数；例如明确为 CNY / 2 位小数后，100 表示 1 元。时点须带明确时区；留空的可用时间保持未知。</p>';
const submit = (label) =>
  `<button type="submit" class="button compact primary">${label}</button>`;
export function installmentHtml() {
  return `<div data-collection-installment class="collection-grid">${identity("installment_id", "期次编号")}${time("due_at", "应还时间")}${integer("amount_minor", "该期应还金额（最小单位）", 1)}${button("remove-row", "移除此期")}</div>`;
}
export function coverageHtml() {
  return `<div data-collection-coverage class="collection-grid">${identity("source_id", "预期资金来源编号")}${select(
    "status",
    "来源声明的覆盖状态",
    [
      ["complete", "完整"],
      ["partial", "部分"],
      ["unknown", "未知"],
    ],
  )}${time("from_at", "覆盖开始")}${time("through_at", "覆盖结束")}${time("available_at", "覆盖声明可用时间")}${button("remove-row", "移除此覆盖")}</div>`;
}
export function windowHtml() {
  return `<div data-collection-window class="collection-grid">${field("weekdays", "允许星期（1=周一，逗号分隔）", "text", 'required placeholder="1,2,3,4,5"')}${field("start", "每日开始（含）", "time", "required")}${field("end", "每日结束（不含）", "text", 'required placeholder="18:00 或 24:00"')}${button("remove-row", "移除此时段")}</div>`;
}
export function attemptHtml() {
  return `<div data-collection-attempt class="collection-grid">${identity("attempt_id", "历史动作编号")}${time("attempted_at", "历史动作时间")}${time("available_at", "历史动作可用时间", false)}${select(
    "state",
    "历史动作状态",
    [
      ["reserved", "已预留"],
      ["dispatched", "已发出"],
      ["completed", "已完成"],
      ["unknown_effect", "效果未知"],
      ["cancelled_before_dispatch", "发出前取消"],
      ["failed_before_dispatch", "发出前失败"],
    ],
  )}${button("remove-row", "移除此动作")}</div>`;
}
function batchCases(cases) {
  return cases
    .map(
      (c) =>
        `<fieldset data-collection-case="${esc(c.case_id)}"><legend><label class="collection-check"><input type="checkbox" name="include_case">纳入 ${esc(c.case_id)} · ${esc(c.unit.currency)} / ${c.unit.minor_unit_exponent} 位小数</label></legend><div class="collection-grid">${select(
          "contact_permission",
          "业务声明的触达许可",
          [
            ["allowed", "允许（仅声明）"],
            ["prohibited", "禁止"],
            ["unknown", "未知"],
          ],
          "",
        )}${select(
          "history_coverage",
          "历史覆盖声明",
          [
            ["complete", "完整"],
            ["partial", "部分"],
            ["unknown", "未知"],
          ],
          "",
        )}${time("history_from", "历史覆盖开始", false)}${time("history_through", "历史覆盖结束", false)}${time("history_available", "历史声明可用时间", false)}</div><p class="collection-note">未知覆盖保留证据不足。声明完整且没有填写历史动作，表示明确声明该窗口内没有历史触达。</p><div data-collection-attempts></div>${button("add-attempt", "添加历史动作")}</fieldset>`,
    )
    .join("");
}
export function intakeHtml(kind, cases = []) {
  let body = "",
    label = "";
  if (kind === "case") {
    label = "登记历史案件";
    body = `<div class="collection-grid">${identity("case_id", "案件编号")}${identity("subject_namespace", "主体标识命名空间")}${field("subject_token", "主体脱敏摘要（64 位十六进制）", "text", 'required pattern="[0-9a-f]{64}"')}${unit()}${integer("opening_balance_minor", "期初余额（最小货币单位）")}${time("opened_at", "开户时间")}</div>`;
  }
  if (kind === "schedule") {
    label = "登记还款排期";
    body = `<div class="collection-grid">${caseSelect(cases)}${identity("schedule_id", "排期编号")}</div><p data-collection-unit class="collection-note"></p><div data-collection-installments>${installmentHtml()}</div>${button("add-installment", "添加期次")}`;
  }
  if (kind === "cashflow") {
    label = "登记历史资金观测";
    body = `<div class="collection-grid">${caseSelect(cases)}${identity("source_id", "资金来源编号")}${identity("event_id", "流水唯一编号")}${select(
      "kind",
      "流水类型",
      [
        ["payment", "回款"],
        ["payment_reversal", "回款冲正"],
        ["cost", "成本"],
        ["cost_reversal", "成本冲正"],
      ],
    )}${integer("amount_minor", "流水金额（最小货币单位）", 1)}${time("event_at", "发生时间")}${time("available_at", "可用时间", false)}${field("reversal_source", "冲正：原资金来源编号")}${field("reversal_event", "冲正：原流水编号")}${field("schedule_id", "回款分配：排期编号（可选）")}${field("installment_id", "回款分配：期次编号（可选）")}</div><p data-collection-unit class="collection-note"></p>`;
  }
  if (kind === "reconcile") {
    label = "核对案件资金与成熟度";
    body = `<div class="collection-grid">${caseSelect(cases)}${time("as_of", "观察截止")}${time("knowledge_cutoff", "知识截止")}${field("expected_sources", "预期资金来源（逗号分隔）", "text", "required")}${field("schedule_id", "关联排期编号（可选）")}${field("maturity_days", "成熟观察天数（不填写则未声明）", "number", 'min="1" max="36500" step="1"')}</div><p class="collection-note">缺少完整资金来源覆盖时，总回款/成本/余额保持未知；成熟口径不推定因果回收。</p><div data-collection-coverages></div>${button("add-coverage", "添加来源覆盖声明")}`;
  }
  if (kind === "batch") {
    label = "校验并冻结批次提案";
    body = `<p class="collection-note">为所选案件明确同一批次动作。平台计算政策指纹并生成规则合同；复杂分层规则可用高级合同导入。</p><p data-collection-unit class="collection-note">请在下方选择案件；预算和单次成本采用案件的最小货币单位。</p><div class="collection-grid">${identity("batch_id", "批次编号")}${time("as_of", "参考决策时点")}${time("knowledge_cutoff", "知识截止")}${identity("policy_id", "政策编号")}${identity("revision", "政策版本")}${time("valid_from", "政策有效期开始")}${time("valid_until", "政策有效期结束")}${field("timezone", "政策时区（IANA）", "text", 'required placeholder="Asia/Shanghai"')}${select(
      "action_kind",
      "本批统一动作",
      [
        ["contact", "本地参考触达"],
        ["review", "转人工队列"],
        ["hold", "暂缓"],
      ],
    )}${identity("queue_id", "政策队列编号（暂缓不入队）")}${integer("priority", "优先级（0—1000）", 0, 1000)}${select(
      "channel",
      "参考触达渠道",
      [
        ["sms", "短信"],
        ["phone", "电话"],
        ["in_app", "应用内"],
        ["letter", "信函"],
      ],
      "",
    )}${field("estimated_cost_minor", "单次预计成本（触达必填，最小单位）", "number", 'min="0" max="1000000000000000" step="1"')}${integer("max_batch_actions", "单批队列容量", 1, 10000)}${integer("max_active_actions", "当前活动队列容量上限", 1, 10000)}${integer("frequency_window_seconds", "主体频次窗口（秒）", 1, 31622400)}${integer("max_contacts_per_subject_window", "窗口内每主体最多次数", 1, 10000)}${integer("min_contact_interval_seconds", "同主体最小间隔（秒）", 0, 31622400)}${integer("max_estimated_batch_cost_minor", "单批预计成本上限（最小单位）")}${integer("max_estimated_active_cost_minor", "当前活动预计成本上限（最小单位）")}</div><h4>允许联系时段</h4><p class="collection-note">采用政策时区，时间区间含开始、不含结束；跨夜分两段声明。</p><div data-collection-windows>${windowHtml()}</div>${button("add-window", "添加允许时段")}<h4>案件与历史声明</h4>${batchCases(cases)}`;
  }
  return `<form data-collection-form="${kind}"><h4>${label}</h4>${commonNote}${body}${source()}<div class="collection-actions">${submit(label)}${button("close-intake", "收起表单")}</div></form>`;
}
const text = (v, label) => {
  const s = String(v ?? "").trim();
  if (!s) throw new Error(`请填写${label}`);
  return s;
};
const number = (v, label, min = 0, max = 1e15) => {
  const n = Number(v);
  if (v === "" || v == null || !Number.isSafeInteger(n) || n < min || n > max)
    throw new Error(`${label}须为${min}至${max}的整数`);
  return n;
};
const date = (v, label) => {
  const s = text(v, label);
  if (!/(?:Z|[+-]\d{2}:\d{2})$/i.test(s) || !Number.isFinite(Date.parse(s)))
    throw new Error(`${label}须包含明确时区`);
  return s;
};
const items = (v) =>
  String(v || "")
    .split(/[,，\n]/)
    .map((s) => s.trim())
    .filter(Boolean);
export const valuesOf = (form) =>
  Object.fromEntries(
    [...form.querySelectorAll("[name]")].map((e) => [e.name, e.value]),
  );
export function collectIntake(
  kind,
  v,
  { record, refs, rows = [], coverage = [], windows = [], cases = [] } = {},
) {
  if (kind !== "case" && kind !== "batch" && !record)
    throw new Error("请选择已登记案件");
  const unit =
    kind === "case"
      ? {
          currency: text(v.currency, "币种"),
          minor_unit_exponent: number(
            v.minor_unit_exponent,
            "金额小数位",
            0,
            4,
          ),
          definition_source: text(v.definition_source, "货币单位来源"),
        }
      : record?.unit;
  if (kind === "case")
    return {
      case_id: text(v.case_id, "案件编号"),
      subject_namespace: text(v.subject_namespace, "命名空间"),
      subject_token: text(v.subject_token, "主体脱敏摘要"),
      unit,
      opening_balance_minor: number(v.opening_balance_minor, "期初余额"),
      opened_at: date(v.opened_at, "开户时间"),
      source_assurance: "historical_import_unverified",
      ...refs,
    };
  if (kind === "schedule")
    return {
      schedule_id: text(v.schedule_id, "排期编号"),
      case_id: record.case_id,
      unit,
      installments: rows.map((r) => ({
        installment_id: text(r.installment_id, "期次编号"),
        due_at: date(r.due_at, "应还时间"),
        amount_minor: number(r.amount_minor, "应还金额", 1),
      })),
      terms_artifact_id: refs.source_artifact_id,
      terms_artifact_hash: refs.source_artifact_hash,
    };
  if (kind === "cashflow") {
    if (
      !["payment", "payment_reversal", "cost", "cost_reversal"].includes(v.kind)
    )
      throw new Error("请选择流水类型");
    const event = {
      source_id: text(v.source_id, "资金来源编号"),
      event_id: text(v.event_id, "流水编号"),
      case_id: record.case_id,
      kind: v.kind,
      amount_minor: number(v.amount_minor, "流水金额", 1),
      unit,
      event_at: date(v.event_at, "发生时间"),
      available_at: v.available_at ? date(v.available_at, "可用时间") : null,
      source_assurance: "historical_import_unverified",
      ...refs,
    };
    if (v.kind.endsWith("_reversal"))
      event.reverses = {
        source_id: text(v.reversal_source, "原资金来源编号"),
        event_id: text(v.reversal_event, "原流水编号"),
      };
    if (v.schedule_id || v.installment_id)
      event.installment = {
        schedule_id: text(v.schedule_id, "排期编号"),
        installment_id: text(v.installment_id, "期次编号"),
      };
    return { events: [event] };
  }
  if (kind === "reconcile") {
    const result = {
      case_id: record.case_id,
      as_of: date(v.as_of, "观察截止"),
      knowledge_cutoff: date(v.knowledge_cutoff, "知识截止"),
      expected_sources: items(v.expected_sources),
      coverage: coverage.map((r) => ({
        source_id: text(r.source_id, "覆盖资金来源"),
        status: text(r.status, "覆盖状态"),
        from_at: date(r.from_at, "覆盖开始"),
        through_at: date(r.through_at, "覆盖结束"),
        available_at: date(r.available_at, "覆盖可用时间"),
        ...refs,
      })),
    };
    if (!result.expected_sources.length) throw new Error("请声明预期资金来源");
    if (v.schedule_id) result.schedule_id = v.schedule_id;
    if (v.maturity_days)
      result.maturity = {
        anchor: "case_opened_at",
        observation_days: number(v.maturity_days, "成熟天数", 1, 36500),
        policy_artifact_id: refs.source_artifact_id,
        policy_artifact_hash: refs.source_artifact_hash,
      };
    return result;
  }
  if (kind === "batch") {
    if (!cases.length) throw new Error("请纳入至少一个已登记案件");
    const unit = cases[0].record.unit;
    if (
      cases.some((c) => JSON.stringify(c.record.unit) !== JSON.stringify(unit))
    )
      throw new Error("本批案件须使用完全相同的货币单位定义");
    const actionKind = text(v.action_kind, "统一动作");
    if (!["contact", "review", "hold"].includes(actionKind))
      throw new Error("请选择批次统一动作");
    const contact = actionKind === "contact",
      hold = actionKind === "hold";
    const policy = {
      policy_id: text(v.policy_id, "政策编号"),
      revision: text(v.revision, "政策版本"),
      unit,
      valid_from: date(v.valid_from, "政策开始"),
      valid_until: date(v.valid_until, "政策结束"),
      timezone: text(v.timezone, "政策时区"),
      contact_windows: windows.map((w) => ({
        weekdays: items(w.weekdays).map((s) => number(s, "星期", 1, 7)),
        start: text(w.start, "每日开始"),
        end: text(w.end, "每日结束"),
      })),
      queues: [
        {
          queue_id: text(v.queue_id, "政策队列编号"),
          channels: contact ? [text(v.channel, "参考渠道")] : [],
          max_batch_actions: number(v.max_batch_actions, "单批容量", 1, 10000),
          max_active_actions: number(
            v.max_active_actions,
            "活动容量",
            1,
            10000,
          ),
        },
      ],
      frequency_window_seconds: number(
        v.frequency_window_seconds,
        "频次窗口",
        1,
        31622400,
      ),
      max_contacts_per_subject_window: number(
        v.max_contacts_per_subject_window,
        "主体频次",
        1,
        10000,
      ),
      min_contact_interval_seconds: number(
        v.min_contact_interval_seconds,
        "最小间隔",
        0,
        31622400,
      ),
      max_estimated_batch_cost_minor: number(
        v.max_estimated_batch_cost_minor,
        "批次成本上限",
      ),
      max_estimated_active_cost_minor: number(
        v.max_estimated_active_cost_minor,
        "活动成本上限",
      ),
      basis_artifact_id: refs.source_artifact_id,
      basis_artifact_hash: refs.source_artifact_hash,
    };
    const histories = [];
    const inputs = cases.map(({ record: r, values: c, attempts }) => {
      if (
        !["allowed", "prohibited", "unknown"].includes(c.contact_permission) ||
        !["complete", "partial", "unknown"].includes(c.history_coverage)
      )
        throw new Error(`请明确 ${r.case_id} 的许可和历史覆盖状态`);
      if (c.history_coverage !== "unknown") {
        histories.push({
          subject_namespace: r.subject_namespace,
          subject_token: r.subject_token,
          from_at: date(c.history_from, "历史覆盖开始"),
          through_at: date(c.history_through, "历史覆盖结束"),
          available_at: date(c.history_available, "历史声明可用时间"),
          coverage: c.history_coverage,
          attempts: attempts.map((a) => ({
            attempt_id: text(a.attempt_id, "历史动作编号"),
            case_id: r.case_id,
            attempted_at: date(a.attempted_at, "历史动作时间"),
            available_at: a.available_at
              ? date(a.available_at, "历史动作可用时间")
              : null,
            state: text(a.state, "历史动作状态"),
          })),
          ...refs,
        });
      } else if (attempts.length)
        throw new Error("已有历史动作时请明确覆盖区间和状态");
      return {
        case_id: r.case_id,
        subject_namespace: r.subject_namespace,
        subject_token: r.subject_token,
        contact_permission: c.contact_permission,
        features: {},
        ...refs,
      };
    });
    return {
      batch_id: text(v.batch_id, "批次编号"),
      policy,
      action: {
        kind: actionKind,
        queue_id: hold ? null : v.queue_id,
        priority: hold ? 0 : number(v.priority, "优先级", 0, 1000),
        channel: contact ? text(v.channel, "参考渠道") : null,
        estimated_cost_minor: contact
          ? number(v.estimated_cost_minor, "预计单位成本")
          : null,
      },
      cases: inputs,
      histories: [
        ...new Map(
          histories.map((h) => {
            const key = `${h.subject_namespace}:${h.subject_token}`;
            const conflicting = histories.find(
              (other) =>
                other !== h &&
                other.subject_namespace === h.subject_namespace &&
                other.subject_token === h.subject_token &&
                JSON.stringify(other) !== JSON.stringify(h),
            );
            if (conflicting)
              throw new Error(
                "同一主体的多个案件必须声明一致的完整历史；不能叠加冲突历史",
              );
            return [key, h];
          }),
        ).values(),
      ],
      as_of: date(v.as_of, "参考时点"),
      knowledge_cutoff: date(v.knowledge_cutoff, "知识截止"),
    };
  }
  throw new Error("不支持的资料类型");
}
