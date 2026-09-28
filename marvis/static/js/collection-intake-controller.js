import { escapeHtml as esc } from "./ui-utils.js";
import {
  intakeHtml,
  installmentHtml,
  coverageHtml,
  windowHtml,
  attemptHtml,
  valuesOf,
  collectIntake,
} from "./collection-intake-form.js";
import { button, details, money } from "./collection-workspace-view.js";
const path = encodeURIComponent;
const routes = {
  case: "cases",
  schedule: "schedules",
  cashflow: "cashflows",
  reconcile: "reconciliations",
  batch: "batch-proposals",
};
const err = (e) => e?.detail?.code || e?.message || "提交失败，请核对本次资料";
export function createCollectionIntakeController({
  getRoot,
  getOwner,
  isCurrent,
  apiClient,
  message,
  setPending,
  refresh,
  onBatch,
  downloadBlob,
}) {
  let draft = null;
  const q = (s) => getRoot()?.querySelector(s),
    base = (o) => `/api/tasks/${path(o.taskId)}/collection`;
  const identity = (o) =>
    JSON.stringify([o?.principal?.id, o?.principal?.role]);
  const valid = (d) =>
    !!d &&
    d === draft &&
    isCurrent(d.owner) &&
    identity(d.owner) === d.identity;
  const post = (url, value) =>
    apiClient(url, { method: "POST", body: JSON.stringify(value) });
  function open(kind) {
    const owner = getOwner();
    if (owner?.principal?.role !== "maker") {
      message("请先读取工作区，并使用案件所属配置人员的会话");
      return;
    }
    draft = {
      owner,
      identity: identity(owner),
      kind,
      edit: 0,
      material: null,
      document: null,
    };
    q("[data-collection-editor]").innerHTML =
      kind === "import-batch"
        ? `<form data-collection-form="import-batch"><h4>高级批次合同导入</h4><p class="collection-note">用于已冻结的复杂分层规则。导入完整业务政策、规则、案件及历史来源合同（JSON，最大 16 MB）；平台验证政策指纹与来源归属，导入不会排队或执行。</p><label>批次合同文件<input name="batch_file" type="file" accept="application/json,.json" required></label><div data-collection-file-preview></div><div class="collection-actions"><button type="submit" class="button compact primary">校验并冻结导入合同</button>${button("close-intake", "收起表单")}</div></form>`
        : intakeHtml(kind, owner.cases || []);
    message("");
  }
  async function chooseFile(input) {
    const d = draft,
      edit = ++d.edit;
    d.document = null;
    const file = input.files?.[0];
    if (!file) return;
    if (file.size > 16_000_000) {
      message("批次合同不得超过 16 MB");
      return;
    }
    try {
      const value = JSON.parse(await file.text());
      if (!valid(d) || edit !== d.edit) return;
      if (
        !value ||
        Array.isArray(value) ||
        value.execution_mode !== "local_reference" ||
        !Array.isArray(value.cases) ||
        !value.policy ||
        !value.strategy
      )
        throw new Error("文件须包含完整本地参考批次合同");
      d.document = value;
      q("[data-collection-file-preview]").innerHTML =
        `<p class="collection-note">${esc(file.name)} · ${value.cases.length} 个案件 · 政策 ${esc(value.policy.policy_id)} / ${esc(value.policy.revision)}</p>${details("复核完整导入合同", value)}`;
      message("");
    } catch (e) {
      if (valid(d) && edit === d.edit) message(err(e));
    }
  }
  function collectRows(form, name) {
    return [...form.querySelectorAll(`[data-collection-${name}]`)].map(
      valuesOf,
    );
  }
  async function submit(form) {
    const d = draft;
    if (!d || !valid(d) || d.owner.pending) return;
    const owner = d.owner,
      edit = d.edit,
      values = valuesOf(form);
    setPending(owner, true);
    message("正在校验本次资料与来源…");
    try {
      let result;
      if (d.kind === "import-batch") {
        if (!d.document) throw new Error("请选择并复核完整批次合同");
        result = await post(base(owner) + "/batches", d.document);
      } else {
        if (!values.source_description?.trim())
          throw new Error("请填写资料来源与口径说明");
        const selected = [
          ...form.querySelectorAll("[data-collection-case]"),
        ].filter((row) => row.querySelector("[name=include_case]").checked);
        const [record, cases] = await Promise.all([
          values.case_id && d.kind !== "case"
            ? apiClient(base(owner) + "/cases/" + path(values.case_id))
            : null,
          Promise.all(
            selected.map(async (row) => ({
              record: await apiClient(
                base(owner) + "/cases/" + path(row.dataset.collectionCase),
              ),
              values: valuesOf(row),
              attempts: collectRows(row, "attempt"),
            })),
          ),
        ]);
        if (!valid(d) || edit !== d.edit) return;
        const context = {
          record,
          cases,
          rows: collectRows(form, "installment"),
          coverage: collectRows(form, "coverage"),
          windows: collectRows(form, "window"),
        };
        collectIntake(d.kind, values, {
          ...context,
          refs: {
            source_artifact_id: "pending",
            source_artifact_hash: "0".repeat(64),
          },
        });
        if (!d.material)
          d.material = await post(base(owner) + "/materials", {
            material_id: "ui-" + crypto.randomUUID(),
            description: values.source_description.trim(),
            declarations: {
              kind: d.kind,
              values,
              installments: context.rows,
              coverage: context.coverage,
              windows: context.windows,
              cases: cases.map((c) => ({
                case_id: c.record.case_id,
                values: c.values,
                attempts: c.attempts,
              })),
            },
          });
        if (!valid(d) || edit !== d.edit) return;
        const refs = {
          source_artifact_id: d.material.source_artifact_id,
          source_artifact_hash: d.material.source_artifact_hash,
        };
        result = await post(
          base(owner) + "/" + routes[d.kind],
          collectIntake(d.kind, values, { ...context, refs }),
        );
      }
      if (!valid(d) || edit !== d.edit) return;
      await refresh();
      if (!valid(d)) return;
      if (["batch", "import-batch"].includes(d.kind)) {
        await onBatch(result.batch_id);
        if (!valid(d)) return;
      } else if (d.kind === "reconcile") {
        d.report = result;
        q("[data-collection-detail]").innerHTML =
          reconciliationHtml(result) +
          button("export-reconciliation", "重新读回并导出本次对账");
      } else
        q("[data-collection-detail]").innerHTML =
          `<h4>本次资料已登记</h4><p class="collection-note">历史来源声明未独立验证；登记不代表客户已被联系或产生因果回收。</p>${details("查看本次登记内容", result)}`;
      message("已登记并重新读取工作区。现有历史记录及执行回执保持不变。");
    } catch (e) {
      if (valid(d) && edit === d.edit)
        message(
          `${err(e)}${d.material ? "。来源声明已登记；业务记录未确认，请保留输入并核对后重试。" : ""}`,
        );
    } finally {
      setPending(owner, false);
    }
  }
  async function exportReport() {
    const d = draft;
    if (!valid(d) || !d.report) return;
    try {
      const r = d.report,
        result = await apiClient(
          base(d.owner) +
            `/cases/${path(r.case_id)}/reconciliations/${path(r.receipt_hash)}`,
        );
      if (valid(d))
        downloadBlob(
          new Blob([JSON.stringify(result, null, 2)], {
            type: "application/json",
          }),
          "collection-cashflow-reconciliation.json",
        );
    } catch (e) {
      if (valid(d)) message(err(e));
    }
  }
  function handle(event) {
    const form = event.target.closest("[data-collection-form]");
    if (event.type === "submit" && form) {
      event.preventDefault();
      void submit(form);
      return true;
    }
    if ((event.type === "input" || event.type === "change") && form && draft) {
      if (draft.owner.pending) return true;
      if (event.target.name === "batch_file" && event.type === "change") {
        void chooseFile(event.target);
        return true;
      }
      draft.edit++;
      draft.material = null;
      if (event.target.name === "case_id") {
        const c = draft.owner.cases?.find(
          (c) => c.case_id === event.target.value,
        );
        const hint = form.querySelector("[data-collection-unit]");
        if (hint)
          hint.textContent = c
            ? `采用案件单位：${c.unit.currency} / ${c.unit.minor_unit_exponent} 位小数 · ${c.unit.definition_source}`
            : "";
      }
      if (event.target.name === "include_case") {
        const selected = [...form.querySelectorAll("[data-collection-case]")]
          .filter((row) => row.querySelector("[name=include_case]").checked)
          .map((row) =>
            draft.owner.cases.find(
              (c) => c.case_id === row.dataset.collectionCase,
            ),
          );
        const hint = form.querySelector("[data-collection-unit]");
        if (hint)
          hint.textContent = !selected.length
            ? "请在下方选择案件；预算和单次成本采用案件的最小货币单位。"
            : selected.some(
                  (c) =>
                    JSON.stringify(c.unit) !== JSON.stringify(selected[0].unit),
                )
              ? "所选案件的货币单位定义不一致，请分批配置。"
              : `本批金额单位：${selected[0].unit.currency} / ${selected[0].unit.minor_unit_exponent} 位小数 · ${selected[0].unit.definition_source}`;
      }
      if (event.target.name === "action_kind") {
        const hold = event.target.value === "hold",
          contact = event.target.value === "contact";
        for (const name of ["channel", "estimated_cost_minor"]) {
          form.elements[name].disabled = !contact;
          form.elements[name].required = contact;
        }
        form.elements.priority.disabled = hold;
        form.elements.priority.required = !hold;
      }
      return true;
    }
    if (event.type !== "click") return false;
    const control = event.target.closest("[data-collection-action]");
    if (!control) return false;
    const action = control.dataset.collectionAction;
    if (
      ![
        "new-case",
        "new-schedule",
        "new-cashflow",
        "new-reconcile",
        "new-batch",
        "import-batch",
        "add-installment",
        "add-coverage",
        "add-window",
        "add-attempt",
        "remove-row",
        "close-intake",
        "export-reconciliation",
      ].includes(action)
    )
      return false;
    event.preventDefault();
    if (getOwner()?.pending) {
      message("正在提交，请等待完成");
      return true;
    }
    if (action.startsWith("new-")) open(action.slice(4));
    else if (action === "import-batch") open(action);
    else if (action === "close-intake") {
      draft = null;
      q("[data-collection-editor]").innerHTML = "";
    } else if (action === "export-reconciliation") void exportReport();
    else {
      if (!draft || !valid(draft)) return true;
      draft.edit++;
      draft.material = null;
      if (action === "remove-row")
        control
          .closest(
            "[data-collection-installment],[data-collection-coverage],[data-collection-window],[data-collection-attempt]",
          )
          ?.remove();
      else {
        const [selector, html] = {
          "add-installment": [
            "[data-collection-installments]",
            installmentHtml,
          ],
          "add-coverage": ["[data-collection-coverages]", coverageHtml],
          "add-window": ["[data-collection-windows]", windowHtml],
          "add-attempt": ["[data-collection-attempts]", attemptHtml],
        }[action];
        const target =
          action === "add-attempt"
            ? control.closest("[data-collection-case]").querySelector(selector)
            : form?.querySelector(selector);
        target?.insertAdjacentHTML("beforeend", html());
      }
    }
    return true;
  }
  return { handle, open };
}
export function reconciliationHtml(r) {
  const maturity =
    {
      not_declared: "未声明",
      insufficient_evidence: "证据不足",
      mature_under_declared_policy: "在声明口径下已成熟",
      immature: "未成熟",
    }[r.maturity?.status] || "未知";
  const labels = {
    net_payments_minor: "净回款",
    net_cost_minor: "净成本",
    net_recovery_minor: "净回收",
    remaining_balance_minor: "剩余余额",
    overpayment_minor: "超额回款",
  };
  return `<h4>案件资金对账 · ${r.status === "reconciled_under_declared_coverage" ? "声明覆盖下已对账" : "证据不足"}</h4><p class="collection-note">${esc(r.case_id)} · 来源真实性未独立验证；增量回收因果未识别。成熟状态 ${esc(maturity)}。</p><div class="collection-grid">${Object.entries(
    labels,
  )
    .map(
      ([key, label]) => `<p>${label}：${money(r.amounts?.[key], r.unit)}</p>`,
    )
    .join("")}</div>${details("冻结对账、可见观测小计与未知原因", r)}`;
}
