import { escapeHtml as esc } from "./ui-utils.js";
import { collectIntake } from "./collection-intake-form.js";
import { button, details } from "./collection-workspace-view.js";
const f = (name, label, required = true, choices = null) => ({
  name,
  label,
  required,
  choices,
});
export const csvFields = {
  case: [
    f("case_id", "案件编号"),
    f("subject_namespace", "主体命名空间"),
    f("subject_token", "主体脱敏摘要（64 位小写十六进制）"),
    f("currency", "币种（三位大写代码）"),
    f("minor_unit_exponent", "金额小数位", true, [
      ["0", "0"],
      ["1", "1"],
      ["2", "2"],
      ["3", "3"],
      ["4", "4"],
    ]),
    f("definition_source", "货币单位定义来源"),
    f("opening_balance_minor", "期初余额（最小单位整数）"),
    f("opened_at", "开户时间（带时区）"),
  ],
  schedule: [
    f("case_id", "已登记案件编号"),
    f("schedule_id", "排期编号"),
    f("installment_id", "期次编号"),
    f("due_at", "应还时间（带时区）"),
    f("amount_minor", "应还金额（最小单位整数）"),
  ],
  cashflow: [
    f("case_id", "已登记案件编号"),
    f("source_id", "资金来源编号"),
    f("event_id", "流水唯一编号"),
    f("kind", "流水类型", true, [
      ["payment", "回款"],
      ["payment_reversal", "回款冲正"],
      ["cost", "成本"],
      ["cost_reversal", "成本冲正"],
    ]),
    f("amount_minor", "流水金额（最小单位整数）"),
    f("event_at", "发生时间（带时区）"),
    f("available_at", "可用时间（空值保持未知）", false),
    f("reversal_source", "冲正的原来源编号", false),
    f("reversal_event", "冲正的原流水编号", false),
    f("schedule_id", "回款分配的排期编号", false),
    f("installment_id", "回款分配的期次编号", false),
  ],
};
export function parseCsv(text) {
  if (typeof text !== "string" || text.length > 2_000_000)
    throw Error("CSV 文件不得超过 2 MB");
  text = text.replace(/^\uFEFF/, "");
  const records = [];
  let record = [],
    value = "",
    quoted = false,
    closed = false;
  const cell = () => {
    record.push(value);
    value = "";
    closed = false;
    if (record.length > 100) throw Error("CSV 不得超过 100 列");
  };
  const row = () => {
    cell();
    if (record.length !== 1 || record[0] !== "") records.push(record);
    record = [];
    if (records.length > 1001) throw Error("每次最多导入 1000 行，请分批");
  };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          value += '"';
          i++;
        } else {
          quoted = false;
          closed = true;
        }
      } else value += c;
      continue;
    }
    if (c === '"') {
      if (value || closed) throw Error("CSV 引号位置无效");
      quoted = true;
    } else if (c === ",") cell();
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row();
    } else {
      if (closed) throw Error("CSV 闭合引号后只能有逗号或换行");
      value += c;
    }
  }
  if (quoted) throw Error("CSV 引号未闭合");
  if (value || record.length || closed) row();
  if (records.length < 2) throw Error("CSV 需要表头及至少一条数据");
  const headers = records.shift().map((h) => h.trim());
  if (headers.some((h) => !h) || new Set(headers).size !== headers.length)
    throw Error("CSV 表头不能为空或重复");
  records.forEach((r, i) => {
    if (r.length !== headers.length)
      throw Error(`第 ${i + 1} 条数据的列数与表头不一致`);
  });
  return { headers, rows: records };
}
export function csvShell() {
  return `<form data-collection-form="csv" data-collection-bulk><h4>批量导入历史资料</h4><p class="collection-note">UTF-8 CSV，最多 2 MB / 1000 行。明确每列的业务含义，或声明本批固定值；金额不自动换算，时点必须带时区。平台冻结文件与映射，不独立验证外部事实。</p><div class="collection-grid"><label>资料角色<select name="csv_kind" required><option value="">请选择</option><option value="case">历史案件</option><option value="schedule">还款排期（每行一期）</option><option value="cashflow">资金观测（每行一笔）</option></select></label><label>历史资料文件<input name="csv_file" type="file" accept="text/csv,.csv" required></label><label>本次导入唯一编号<input name="csv_import_id" required pattern="[A-Za-z0-9_.:-]{1,100}" maxlength="100"></label><label>资料来源与口径说明<input name="csv_description" required maxlength="1000"></label></div><p class="collection-note">相同导入编号必须保持文件、映射与来源说明一致，便于重试；更正资料请用新的编号。案件与排期逐项登记，若中断会保留已完成项。资金观测在同一次请求中校验登记。</p><div data-collection-csv-mapping></div><div class="collection-actions"><button type="submit" class="button compact primary">校验映射并预览</button>${button("csv-close", "收起导入")}</div><div data-collection-csv-preview></div></form>`;
}
export function mappingHtml(kind, data) {
  const fields = csvFields[kind];
  if (!fields || !data) return "";
  return `<h4>业务字段映射</h4><p class="collection-note">${data.rows.length} 行 · ${data.headers.length} 列。${kind === "schedule" ? "同案件、同排期的行合并为一张排期；请按应还时间排序。" : kind === "cashflow" ? "流水类型列须使用 payment / payment_reversal / cost / cost_reversal，也可选择固定值。" : ""}${kind !== "case" ? "金额采用每个已登记案件的货币单位；无需重复声明。" : ""}</p><div class="collection-csv-mapping">${fields.map((d) => `<label>${esc(d.label)}${d.required ? " *" : ""}<select data-csv-field="${d.name}"><option value="">${d.required ? "请选择来源" : "不声明 / 未知"}</option>${data.headers.map((h, i) => `<option value="column:${i}">${esc(h)}</option>`).join("")}<option value="constant">本批固定值</option></select></label><label data-csv-constant-wrap="${d.name}" hidden>固定值 · ${esc(d.label)}${d.choices ? `<select data-csv-constant="${d.name}"><option value="">请选择</option>${d.choices.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join("")}</select>` : `<input data-csv-constant="${d.name}" type="text">`}</label>`).join("")}</div>${details("原始文件前 5 行（值保持原文）", { columns: data.headers, rows: data.rows.slice(0, 5) })}`;
}
export function mapCsv(kind, data, mapping) {
  const fields = csvFields[kind];
  if (!fields) throw Error("请选择资料角色");
  return data.rows.map((row, i) =>
    Object.fromEntries(
      fields.map((f) => {
        const m = mapping[f.name];
        let value = "";
        if (m?.source === "constant") value = m.value;
        else if (m?.source?.startsWith("column:")) {
          const n = Number(m.source.slice(7));
          if (!Number.isInteger(n) || n < 0 || n >= data.headers.length)
            throw Error("列映射已失效");
          value = row[n];
        }
        if (f.required && !String(value ?? "").trim())
          throw Error(`第 ${i + 1} 条数据：请填写${f.label}`);
        return [f.name, String(value ?? "")];
      }),
    ),
  );
}
export const pendingRefs = {
  source_artifact_id: "pending",
  source_artifact_hash: "0".repeat(64),
};
export function csvOperations(
  kind,
  rows,
  cases = new Map(),
  refs = pendingRefs,
) {
  const seen = new Set(),
    operations = [],
    groups = new Map();
  for (const [i, v] of rows.entries()) {
    try {
      for (const field of [
        "case_id",
        "subject_namespace",
        "schedule_id",
        "installment_id",
        "source_id",
        "event_id",
        "reversal_source",
        "reversal_event",
      ])
        if (v[field] && !/^[A-Za-z0-9_.:-]{1,128}$/.test(v[field].trim()))
          throw Error(
            `编号 ${field} 只能使用英文字母、数字、点、下划线、冒号和短横线`,
          );
      if (kind === "case") {
        if (!/^[0-9a-f]{64}$/.test(v.subject_token.trim()))
          throw Error("主体脱敏摘要须为 64 位小写十六进制");
        if (!/^[A-Z]{3}$/.test(v.currency.trim()))
          throw Error("币种须为三位大写代码");
      }
      const record = cases.get(v.case_id.trim());
      const value = collectIntake(kind, v, {
        record,
        refs,
        rows: kind === "schedule" ? [v] : [],
      });
      const identity =
        kind === "case"
          ? value.case_id
          : kind === "schedule"
            ? JSON.stringify([
                value.case_id,
                value.schedule_id,
                v.installment_id.trim(),
              ])
            : JSON.stringify([v.source_id.trim(), v.event_id.trim()]);
      if (seen.has(identity)) throw Error("文件内存在重复业务编号，请先核对");
      seen.add(identity);
      if (kind === "schedule") {
        const key = JSON.stringify([value.case_id, value.schedule_id]);
        if (!groups.has(key))
          groups.set(key, {
            label: `${value.case_id} / ${value.schedule_id}`,
            route: "schedules",
            value,
          });
        else groups.get(key).value.installments.push(...value.installments);
      } else if (kind === "cashflow") {
        const e = value.events[0];
        if (
          e.available_at &&
          Date.parse(e.available_at) < Date.parse(e.event_at)
        )
          throw Error("可用时间不得早于发生时间");
        if (e.installment && e.kind !== "payment")
          throw Error("仅原始回款可分配期次，冲正继承原流水");
        operations.push(value.events[0]);
      } else operations.push({ label: value.case_id, route: "cases", value });
    } catch (e) {
      throw Error(`第 ${i + 1} 条数据：${e.message}`);
    }
  }
  if (kind === "cashflow")
    return [
      {
        label: `${operations.length} 笔资金观测`,
        route: "cashflows",
        value: { events: operations },
      },
    ];
  if (kind === "schedule") {
    for (const g of groups.values()) {
      if (g.value.installments.length > 600)
        throw Error(`${g.label} 超过 600 期`);
      if (
        g.value.installments.some(
          (p, i, a) => i && Date.parse(p.due_at) < Date.parse(a[i - 1].due_at),
        )
      )
        throw Error(`${g.label} 需要按应还时间排序`);
    }
    return [...groups.values()];
  }
  return operations;
}
