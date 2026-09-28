import { escapeHtml as esc } from "./ui-utils.js";
import {
  csvShell,
  mappingHtml,
  parseCsv,
  mapCsv,
  csvOperations,
} from "./collection-csv-form.js";
import {
  button,
  details,
  collectionError,
} from "./collection-workspace-view.js";
const error = collectionError;
export function createCollectionCsvController({
  getRoot,
  getOwner,
  isCurrent,
  apiClient,
  message,
  setPending,
  refresh,
  onOpen,
}) {
  let draft = null;
  const q = (s) => getRoot()?.querySelector(s),
    identity = (o) => JSON.stringify([o?.principal?.id, o?.principal?.role]);
  const valid = (d) =>
    !!d &&
    d === draft &&
    isCurrent(d.owner) &&
    identity(d.owner) === d.identity;
  const base = (d) =>
    `/api/tasks/${encodeURIComponent(d.owner.taskId)}/collection`;
  const post = (url, value) =>
    apiClient(url, { method: "POST", body: JSON.stringify(value) });
  function open() {
    const owner = getOwner();
    if (owner?.principal?.role !== "maker") {
      message("请先读取工作区，并绑定案件所属配置人员身份");
      return;
    }
    onOpen();
    draft = {
      owner,
      identity: identity(owner),
      version: 0,
      kind: "",
      file: null,
      preview: null,
      sealed: false,
      results: [],
      material: null,
    };
    q("[data-collection-editor]").innerHTML = csvShell();
    message("");
  }
  const invalidate = (d) => {
    d.version++;
    d.preview = null;
    q("[data-collection-csv-preview]").innerHTML = "";
  };
  async function chooseFile(input) {
    const d = draft;
    invalidate(d);
    const version = d.version;
    d.file = null;
    q("[data-collection-csv-mapping]").innerHTML = "";
    const file = input.files?.[0];
    if (!file) return;
    try {
      if (file.size > 2_000_000) throw Error("CSV 文件不得超过 2 MB");
      const text = new TextDecoder("utf-8", { fatal: true }).decode(
          await file.arrayBuffer(),
        ),
        data = parseCsv(text);
      if (!valid(d) || version !== d.version) return;
      d.file = { name: file.name, text, data };
      q("[data-collection-csv-mapping]").innerHTML = mappingHtml(d.kind, data);
      message("文件已读取，请明确每个业务字段的来源");
    } catch (e) {
      if (valid(d) && version === d.version) message(error(e));
    }
  }
  async function preview(form) {
    const d = draft;
    if (!valid(d) || d.sealed || d.owner.pending) return;
    const version = ++d.version;
    d.preview = null;
    q("[data-collection-csv-preview]").innerHTML = "";
    setPending(d.owner, true);
    try {
      if (!d.file || !d.kind) throw Error("请选择资料角色和 CSV 文件");
      const importId = form.elements.csv_import_id.value.trim(),
        description = form.elements.csv_description.value.trim();
      if (!/^[A-Za-z0-9_.:-]{1,100}$/.test(importId))
        throw Error(
          "导入编号须为 1—100 位英文字母、数字、点、下划线、冒号或短横线",
        );
      if (!description) throw Error("请填写资料来源与口径说明");
      const mapping = Object.fromEntries(
        [...form.querySelectorAll("[data-csv-field]")].map((el) => [
          el.dataset.csvField,
          {
            source: el.value,
            value: form.querySelector(
              `[data-csv-constant="${el.dataset.csvField}"]`,
            ).value,
          },
        ]),
      );
      const rows = mapCsv(d.kind, d.file.data, mapping),
        cases = new Map();
      if (d.kind !== "case") {
        const ids = [...new Set(rows.map((r) => r.case_id.trim()))];
        for (let offset = 0; offset < ids.length; offset += 10) {
          const items = await Promise.all(
            ids
              .slice(offset, offset + 10)
              .map((id) =>
                apiClient(base(d) + "/cases/" + encodeURIComponent(id)),
              ),
          );
          if (!valid(d) || version !== d.version) return;
          items.forEach((item) => cases.set(item.case_id, item));
        }
      }
      if (!valid(d) || version !== d.version) return;
      const operations = csvOperations(d.kind, rows, cases);
      d.preview = {
        kind: d.kind,
        file: d.file,
        mapping,
        rows,
        cases,
        operations,
        importId,
        description,
      };
      q("[data-collection-csv-preview]").innerHTML =
        `<h4>待登记预览</h4><p>${rows.length} 条资料 → ${operations.length} 次业务登记</p><p class="collection-note">案件与排期按下列顺序登记，发生错误时停止；已登记记录保留。金额单位来自明确声明或现有案件。空白可用时间保持未知；此次导入不产生催收动作或资金效果。</p>${details("逐项核对业务合同（来源指纹将在登记时生成）", operations)}${button("csv-commit", "确认并登记这些资料")}<div data-collection-csv-results></div>`;
      message("映射预览已就绪；确认后才会登记来源和业务记录");
    } catch (e) {
      if (valid(d) && version === d.version) message(error(e));
    } finally {
      setPending(d.owner, false);
    }
  }
  function renderResults(d) {
    if (!valid(d)) return;
    const el = q("[data-collection-csv-results]");
    if (!el) return;
    const operations = d.preview.operations;
    el.innerHTML = `<h4>登记进度 · ${d.results.filter((r) => r?.ok).length} / ${operations.length}</h4><ol>${operations.map((op, i) => `<li>${esc(op.label)} · ${d.results[i]?.ok ? "已登记" : d.results[i]?.error ? `未确认：${esc(d.results[i].error)}` : "尚未登记"}</li>`).join("")}</ol>${d.material ? details("本次文件与映射的来源指纹", d.material) : ""}${details("逐项登记响应", d.results)}`;
  }
  async function commit() {
    const d = draft;
    if (!valid(d) || !d.preview || d.owner.pending) return;
    d.sealed = true;
    const p = d.preview;
    setPending(d.owner, true);
    message("正在登记冻结来源与业务记录…");
    try {
      if (!d.material)
        d.material = await post(base(d) + "/materials", {
          material_id: "csv-" + p.importId,
          description: p.description,
          declarations: {
            kind: p.kind,
            file_name: p.file.name,
            file_text: p.file.text,
            mapping: p.mapping,
          },
        });
      if (!valid(d)) return;
      const refs = {
        source_artifact_id: d.material.source_artifact_id,
        source_artifact_hash: d.material.source_artifact_hash,
      };
      const operations = csvOperations(p.kind, p.rows, p.cases, refs);
      for (const [index, op] of operations.entries()) {
        if (!valid(d)) return;
        if (d.results[index]?.ok) continue;
        try {
          const result = await post(base(d) + "/" + op.route, op.value);
          if (!valid(d)) return;
          d.results[index] = { label: op.label, ok: true, result };
          renderResults(d);
        } catch (e) {
          if (valid(d)) {
            d.results[index] = { label: op.label, ok: false, error: error(e) };
            renderResults(d);
          }
          throw e;
        }
      }
      await refresh();
      if (!valid(d)) return;
      message("本次资料已全部登记。历史事实仍为来源声明，未新增催收动作。");
    } catch (e) {
      if (valid(d))
        message(
          `${error(e)}。已登记 ${d.results.filter((r) => r?.ok).length} / ${p.operations.length} 项；已登记记录不会回滚。保留冻结文件与映射，核对后可重试未确认项。`,
        );
    } finally {
      setPending(d.owner, false);
      if (valid(d)) {
        const form = q("[data-collection-bulk]");
        form
          ?.querySelectorAll('input,select,button[type="submit"]')
          .forEach((el) => (el.disabled = true));
        const retry = form?.querySelector(
          '[data-collection-action="csv-commit"]',
        );
        if (retry) {
          retry.disabled =
            d.results.filter((r) => r?.ok).length === p.operations.length;
          retry.textContent = retry.disabled ? "全部登记完成" : "重试未确认项";
        }
        renderResults(d);
      }
    }
  }
  function handle(event) {
    const form = event.target.closest("[data-collection-bulk]");
    if (form && event.type === "submit") {
      event.preventDefault();
      void preview(form);
      return true;
    }
    if (form && ["input", "change"].includes(event.type)) {
      const d = draft;
      if (!valid(d) || d.owner.pending || d.sealed) return true;
      const el = event.target;
      if (el.name === "csv_file") {
        if (event.type === "change") void chooseFile(el);
        return true;
      }
      invalidate(d);
      if (el.name === "csv_kind") {
        d.kind = el.value;
        q("[data-collection-csv-mapping]").innerHTML = mappingHtml(
          d.kind,
          d.file?.data,
        );
      }
      if (el.dataset.csvField) {
        const wrap = form.querySelector(
          `[data-csv-constant-wrap="${el.dataset.csvField}"]`,
        );
        wrap.hidden = el.value !== "constant";
      }
      return true;
    }
    if (event.type !== "click") return false;
    const control = event.target.closest("[data-collection-action]"),
      action = control?.dataset.collectionAction;
    if (!["import-csv", "csv-close", "csv-commit"].includes(action))
      return false;
    event.preventDefault();
    if (getOwner()?.pending) {
      message("正在处理，请等待完成");
      return true;
    }
    if (action === "import-csv") open();
    else if (action === "csv-commit") void commit();
    else {
      draft = null;
      q("[data-collection-editor]").innerHTML = "";
    }
    return true;
  }
  return {
    handle,
    reset: () => {
      draft = null;
    },
  };
}
