/** Exact source selection, immutable freeze, and independent evidence display. */
import { escapeHtml } from "../ui-utils.js";

const panels = new WeakMap();
const label = (source) => `${source.source_kind} · ${source.ref.asset_id}`;
const pct = (value) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;

export function mountStrategyAssetStabilityPanel(root, taskId) {
  if (!root || !taskId) return;
  let state = panels.get(root);
  if (state?.taskId === taskId) return;
  state?.abort?.abort();
  const abort = new AbortController();
  state = { taskId, abort, catalogue: null, freeze: null, measured: null, busy: false, revision: 0 };
  panels.set(root, state);
  root.innerHTML = '<details><summary>独立候选稳定性 · validation / OOT</summary><div data-asset-stability-body>正在核验候选来源…</div></details>';
  const body = root.querySelector("[data-asset-stability-body]");
  const api = async (suffix = "", payload) => {
    const revision = state.revision;
    const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/strategy-asset-stability${suffix}`, {
      ...(payload ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) } : {}),
      signal: abort.signal,
    });
    const result = await response.json();
    if (panels.get(root) !== state || state.revision !== revision) throw new DOMException("Selection changed", "AbortError");
    if (!response.ok) throw new Error(result.detail?.message || result.detail || "来源核验或测量失败");
    return result;
  };
  const render = () => {
    if (panels.get(root) !== state) return;
    const data = state.catalogue;
    body.innerHTML = `<p>先选择明确的候选和样本，冻结版本后测量。development 分桶保持固定；结果不改变候选状态。</p>
      <label>候选资产<select data-asset-source><option value="">选择资产</option>${data.sources.map((s, i) => `<option value="${i}">${escapeHtml(label(s))}</option>`).join("")}</select></label>
      <label>样本设计<select data-asset-sample><option value="">选择样本</option>${data.samples.map((s, i) => `<option value="${i}">${escapeHtml(s.ref.expected_sample_design_id)} · dev ${s.counts.development} / validation ${s.counts.validation} / OOT ${s.counts.oot}</option>`).join("")}</select></label>
      <label><input type="checkbox" data-asset-partition="validation" checked> validation</label>
      <label><input type="checkbox" data-asset-partition="oot" checked> OOT</label>
      <button type="button" data-asset-freeze>冻结选定来源</button>
      <button type="button" data-asset-measure disabled>测量冻结资产</button>
      <button type="button" data-asset-report disabled>生成含此证据的报告</button>
      <button type="button" data-asset-refresh>刷新候选</button>
      <label>已有冻结证据<select data-asset-existing-freeze><option value="">选择冻结版本</option>${data.evidence.filter(x => x.kind === "strategy_asset_stability_freeze_json").map(x => `<option value="${escapeHtml(x.ref.artifact_id)}">${escapeHtml(x.evidence.source_kind)} · ${escapeHtml(x.ref.stability_id)}</option>`).join("")}</select></label><button type="button" data-asset-resume>使用选定冻结证据</button>
      <details><summary>已测量证据记录</summary>${data.evidence.filter(x => x.kind === "strategy_asset_independent_stability_json").map(x => `<p>${escapeHtml(x.evidence.freeze.source_kind)} · ${escapeHtml(x.ref.stability_id)} <a href="/api/tasks/${encodeURIComponent(taskId)}/task-artifacts/${encodeURIComponent(x.ref.artifact_id)}/download?expected_content_hash=${encodeURIComponent(x.ref.artifact_content_hash)}">查看确切证据</a></p>`).join("") || "暂无独立测量证据"}</details>
      <div role="status" data-asset-status></div><div data-asset-results></div>`;
  };
  const status = (text) => { if (panels.get(root) !== state) return; const el = root.querySelector("[data-asset-status]"); if (el) el.textContent = text; };
  const refresh = async () => { state.catalogue = await api(); state.freeze = null; state.measured = null; render(); };
  const invalidate = () => {
    state.freeze = null;
    state.measured = null;
    state.revision += 1;
    const measure = root.querySelector("[data-asset-measure]"); if (measure) measure.disabled = true;
    const report = root.querySelector("[data-asset-report]"); if (report) report.disabled = true;
    const results = root.querySelector("[data-asset-results]"); if (results) results.innerHTML = "";
    status("选择已变化，请重新冻结来源。");
  };
  root.addEventListener("change", invalidate, { signal: abort.signal });
  root.addEventListener("click", async (event) => {
    const button = event.target.closest?.("button");
    if (!button || state.busy) return;
    if (!button.hasAttribute("data-asset-freeze") && !button.hasAttribute("data-asset-measure") && !button.hasAttribute("data-asset-refresh") && !button.hasAttribute("data-asset-report") && !button.hasAttribute("data-asset-resume")) return;
    state.busy = true; button.disabled = true;
    status(button.hasAttribute("data-asset-measure") ? "正在校验冻结版本并测量独立分区；超出完整证据预算会明确拒绝。" : "正在核验来源并处理请求…");
    try {
      if (button.hasAttribute("data-asset-refresh")) { await refresh(); return; }
      if (button.hasAttribute("data-asset-resume")) {
        const id = root.querySelector("[data-asset-existing-freeze]").value;
        const frozen = state.catalogue.evidence.find(x => x.kind === "strategy_asset_stability_freeze_json" && x.ref.artifact_id === id);
        if (!frozen) throw new Error("请选择明确的已有冻结证据。");
        state.freeze = { ref: frozen.ref, evidence: frozen.evidence };
        root.querySelector("[data-asset-measure]").disabled = false;
        status(`已选择冻结版本 ${frozen.ref.stability_id}；已有测量会直接恢复确切证据。`);
        return;
      }
      if (button.hasAttribute("data-asset-report")) {
        if (!state.measured) throw new Error("请先测量冻结资产。");
        const proposal = await api("/report-proposal", { title: "独立候选稳定性评审报告", status: "partial", asset_stability_refs: [state.measured.ref] });
        const report = await api("/report", proposal);
        status("报告已生成，四种格式使用同一份来源和确定性指标。");
        const links = report.artifacts.map(a => `<a href="${escapeHtml(a.url || a.download_url)}">${escapeHtml(a.format)}</a>`).join(" · ");
        root.querySelector("[data-asset-results]").insertAdjacentHTML("beforeend", `<p>${links}</p>`);
        return;
      }
      if (button.hasAttribute("data-asset-freeze")) {
        const sourceValue = root.querySelector("[data-asset-source]").value;
        const sampleValue = root.querySelector("[data-asset-sample]").value;
        if (sourceValue === "" || sampleValue === "") throw new Error("请选择候选资产和样本设计。");
        const partitions = [...root.querySelectorAll("[data-asset-partition]:checked")].map(el => el.dataset.assetPartition);
        state.freeze = await api("/freeze", { source_ref: state.catalogue.sources[Number(sourceValue)].ref, sample_design_ref: state.catalogue.samples[Number(sampleValue)].ref, partitions });
        root.querySelector("[data-asset-measure]").disabled = false;
        status(`已冻结 ${state.freeze.ref.stability_id}。来源变化将拒绝测量。`);
      } else {
        const measured = await api("/measure", { frozen_ref: state.freeze.ref });
        state.measured = measured;
        root.querySelector("[data-asset-report]").disabled = false;
        const evidence = measured.evidence;
        status(`测量完成 · ${measured.ref.stability_id} · development ${evidence.baseline.sample_count} 条`);
        const rows = evidence.partitions.flatMap(p => p.buckets.map(b => `<tr><td>${escapeHtml(p.partition)}</td><td>${escapeHtml(b.bucket_id)}</td><td>${b.count}</td><td>${pct(b.share)}</td><td>${pct(b.bad_rate)}</td><td>${p.psi_vs_development.toFixed(6)}</td></tr>`)).join("");
        root.querySelector("[data-asset-results]").innerHTML = `<table><thead><tr><th>分区</th><th>固定分桶</th><th>样本数</th><th>占比</th><th>坏率</th><th>相对开发集 PSI</th></tr></thead><tbody>${rows}</tbody></table><p>${evidence.partitions.some(p => p.small_sample) ? "分区样本较少，需谨慎解读。" : ""}</p><a href="${escapeHtml(measured.artifacts[0].url)}">下载确切证据</a><details><summary>查看报告证据来源</summary><code>${escapeHtml(JSON.stringify(measured.ref))}</code></details>`;
      }
    } catch (error) { if (error.name !== "AbortError") status(String(error.message || error)); }
    finally {
      state.busy = false;
      if (button.isConnected) button.disabled = false;
      if (panels.get(root) === state) {
        const measure = root.querySelector("[data-asset-measure]"); if (measure) measure.disabled = !state.freeze;
        const report = root.querySelector("[data-asset-report]"); if (report) report.disabled = !state.measured;
      }
    }
  }, { signal: abort.signal });
  refresh().catch(error => { if (error.name !== "AbortError") body.textContent = `读取失败：${error.message}`; });
}
