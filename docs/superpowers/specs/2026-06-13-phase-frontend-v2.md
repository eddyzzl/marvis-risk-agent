# Phase 前端 V2 — 编排/插件/工件 前端（函数级 spec，含内部伪代码）

## 文档状态

- 状态：已实现并验证
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 11 节）
- 依赖后端 API：Phase 1（plugins）、Phase 2（plans）、Phase 2B（capability_tier/replan/explore）、Phase 3（datasets/joins）
- 可与后端**并行**开发（按 API 契约 mock）
- 目标：把 V2 的编排能力做成可视、可操作的前端——计划 DAG、步骤确认门、**join 诊断确认**、子 agent 状态、plugin 管理、工件预览、能力档位、自适应 loop 进度。

## 约束（沿用 Phase 0 决定）

- **无构建、无 npm、ES Modules**，`<script type="module">` 加载；离线可部署。
- **只渲染 API 结构化 payload，不解析后端自由文本作为业务事实**（AGENTS.md）。
- **一切插入 DOM 的动态文本过 `escapeHtml`**；事件委托挂稳定祖先节点。
- 轮询统一走 `core/poll.js`（去重，Phase 0 已建）。
- 复用 Phase 0 抽出的 `core/{state,bus,api,poll}.js`、`render/markdown.js`、`ui-utils.js`。

## 模块布局

```text
static/js/v2/
  api_v2.js          v2 端点封装（plans/plugins/datasets/joins/capability）
  state_v2.js        v2 视图状态（当前 plan、选中 step、plugin 列表…）
  plan_view.js       计划 DAG + 步骤状态 + 进度
  plan_confirm.js    计划确认 / 决策点 / 步骤确认门
  join_review.js     JoinPlan 诊断 + 逐表确认（INV-3 关键）
  subagent_view.js   子 agent 状态树
  plugin_manager.js  plugin 列表 / 上传 / 启停 / tools
  skill_manager.js   用户 skill 模板：列表 / 状态(active/rejected) / 校验 / reload
  workflow_create.js 目标→计划（意图路由 + 模板/生成）
  artifact_view.js   数据集 / 报告 / 图表 工件预览
  capability.js      capability_tier 选择器
  loop_progress.js   自适应 loop：replan / explore 事件流
  main_v2.js         v2 装配入口（挂载点 + 路由）
```

---

## Part A — 共享约定与装配（`v2/main_v2.js`、`v2/api_v2.js`、`v2/state_v2.js`）

### A-1 v2 API 封装（`api_v2.js`）

```js
import { apiGet, apiPost, apiDelete } from "../core/api.js";

// plans
export const createPlan = (taskId, body) => apiPost(`/api/tasks/${taskId}/plans`, body);
export const getPlan = (planId) => apiGet(`/api/plans/${planId}`);
export const confirmPlan = (planId) => apiPost(`/api/plans/${planId}/confirm`, {});
export const runPlan = (planId) => apiPost(`/api/plans/${planId}/run`, {});
export const confirmStep = (planId, stepId) => apiPost(`/api/plans/${planId}/steps/${stepId}/confirm`, {});
export const cancelPlan = (planId) => apiPost(`/api/plans/${planId}/cancel`, {});
// plugins
export const listPlugins = (incDisabled=false) => apiGet(`/api/plugins?include_disabled=${incDisabled}`);
export const uploadPlugin = (file) => {
  const fd = new FormData();
  fd.append("file", file);
  return apiPost("/api/plugins", fd);   // Phase 0 apiPost: FormData 不手动设置 Content-Type
};
export const setPluginEnabled = (name, on) => apiPost(`/api/plugins/${name}/${on?"enable":"disable"}`, {});
export const removePlugin = (name) => apiDelete(`/api/plugins/${name}`);
export const listPluginTools = (name) => apiGet(`/api/plugins/${name}/tools`);
// skills（用户可编写 Workflow 模板，Phase 2 Part C2）
export const listSkills = () => apiGet("/api/skills");
export const reloadSkills = () => apiPost("/api/skills/reload", {});
export const validateSkill = (skill) => apiPost("/api/skills/validate", {skill});
// datasets / joins
export const listDatasets = (taskId) => apiGet(`/api/tasks/${taskId}/datasets`);
export const uploadDataset = (taskId, file, opts={}) => {
  const fd = new FormData();
  fd.append("file", file);
  if (opts.role) fd.append("role", opts.role);
  if (opts.sheet) fd.append("sheet", opts.sheet);
  return apiPost(`/api/tasks/${taskId}/datasets/upload`, fd);
};
export const previewDataset = (datasetId, rows=50) => apiGet(`/api/datasets/${datasetId}/preview?rows=${rows}`);
export const proposeJoin = (taskId, body) => apiPost(`/api/tasks/${taskId}/joins/propose`, body);
export const getJoinPlan = (joinId) => apiGet(`/api/joins/${joinId}`);
export const confirmJoinSpec = (joinId, body) => apiPost(`/api/joins/${joinId}/confirm`, body);
export const executeJoin = (joinId) => apiPost(`/api/joins/${joinId}/execute`, {});
// capability
export const listCapabilityTiers = () => apiGet("/api/capability-tiers");
```

> 注：上述端点契约对应各 Phase spec；后端未到时前端用 `api_v2.mock.js` 提供同形 fixture，保证前端可独立开发测试。任何上传接口必须走 Phase 0 `apiPost(FormData)` 分支，不能在 v2 模块里手写 fetch 或覆盖 multipart boundary。

### A-2 v2 状态（`state_v2.js`）

```js
import { setState, getState, subscribe } from "../core/state.js";
// v2 状态键（集中，避免散落全局）：
//   v2.currentPlan, v2.selectedStepId, v2.plugins, v2.datasets, v2.currentJoin,
//   v2.capabilityTiers, v2.selectedTier, v2.loopEvents
export const setPlan = (plan) => setState("v2.currentPlan", plan);
export const getPlan = () => getState("v2.currentPlan");
export const onPlanChange = (fn) => subscribe("v2.currentPlan", fn);
// ...各键同构 setter/getter/subscribe
```

### A-3 装配（`main_v2.js`）

```js
export function mountV2(root) {
  /** 在主页面挂载 v2 视图容器 + 注册轮询/事件委托。
   * 伪代码:
   *   renderPlanView(root.querySelector("#planPanel"));
   *   renderSubAgentView(root.querySelector("#subAgentPanel"));
   *   renderPluginManager(root.querySelector("#pluginPanel"));
   *   renderArtifactView(root.querySelector("#artifactPanel"));
   *   attachV2Delegation(root);   // 事件委托挂到 root（稳定节点）
   */
}
```

- **测试要点**：api_v2 端点 URL 正确；`uploadPlugin`/`uploadDataset` 传入 FormData 且不设置 `Content-Type`；state_v2 setter/subscribe 联动；mountV2 不报错、挂载点存在。

---

## Part B — 计划 DAG 视图（`v2/plan_view.js`，中心视图）

```js
export function renderPlanView(container) {
  /** 渲染当前 plan：步骤列表（拓扑序）+ 每步状态徽标 + DAG 连线 + 整体进度。
   * 不变量: 只读 plan payload 的结构化字段（status/steps/depends_on），不解析文本。
   * 伪代码:
   *   onPlanChange((plan) => container.innerHTML = plan ? planHtml(plan) : emptyHtml());
   */
}

function planHtml(plan) {
  /** plan → HTML。
   * 伪代码:
   *   header = `<div class="plan-header">${escapeHtml(plan.goal)} · ${statusBadge(plan.status)}
   *             · 档位 ${escapeHtml(plan.tier)}${plan.novel_mode==="explore"?" · 探索中":""}</div>`;
   *   steps = plan.steps.map(stepRowHtml).join("");
   *   return header + `<ol class="plan-steps">${steps}</ol>` + progressBarHtml(plan);
   */
}

function stepRowHtml(step) {
  /** 单步行：序号 · 标题 · 工具 · 状态徽标 · 决策点标记 · 确认按钮(若 AWAITING_CONFIRM)。
   * 不变量: 工具/状态来自 step 字段；确认按钮只在 status==="awaiting_confirm" 显示。
   * 伪代码:
   *   tool = escapeHtml(`${step.tool_ref.plugin}.${step.tool_ref.tool}`);
   *   badge = stepStatusBadge(step.status);            // pending/running/checking/done/failed/skipped
   *   dp = step.decision_point ? `<span class="dp-mark" data-tip="决策点：完成后按结果重规划">◆</span>` : "";
   *   confirm = step.status==="awaiting_confirm"
   *           ? `<button data-confirm-step="${escapeHtml(step.id)}">确认执行</button>` : "";
   *   verdict = step.review_verdicts?.length ? reviewVerdictHtml(step.review_verdicts) : "";
   *   return `<li data-step="${escapeHtml(step.id)}" class="step ${step.status}">
   *             <span class="idx">${step.index+1}</span>
   *             <span class="title">${escapeHtml(step.title)}</span>
   *             <span class="tool">${tool}</span> ${dp} ${badge} ${confirm} ${verdict}
   *           </li>`;
   */
}

function reviewVerdictHtml(verdicts) {
  /** 展示确定性检查/LLM critic 结论（可展开）。
   * 不变量: deterministic 失败用红色硬标记；llm_critic 只作提示不覆盖。
   */
}

export function startPlanPolling(planId) {
  /** 用 core/poll.js 轮询 plan 进度（去重），更新 state。
   * 伪代码:
   *   startPoll(`plan:${planId}`, async () => { const p = await getPlanApi(planId); setPlan(p);
   *     if (TERMINAL.has(p.status)) stopPoll(`plan:${planId}`); }, 1000);
   */
}
```

- **测试要点**（`test_frontend_static_v2` 风格 + node import）：plan payload → 步骤行正确；状态徽标映射；决策点标记；AWAITING_CONFIRM 才出确认按钮；deterministic 失败红标；轮询去重且终态停。

---

## Part C — 计划确认 / 决策点 / 步骤确认门（`v2/plan_confirm.js`）

```js
export function attachPlanConfirmHandlers(root) {
  /** 事件委托处理计划级/步骤级确认动作。
   * 伪代码:
   *   root.addEventListener("click", async (e) => {
   *     const planBtn = e.target.closest("[data-confirm-plan]");
   *     if (planBtn) { await confirmPlan(planBtn.dataset.confirmPlan);
   *                    await runPlan(planBtn.dataset.confirmPlan); startPlanPolling(...); return; }
   *     const stepBtn = e.target.closest("[data-confirm-step]");
   *     if (stepBtn) { const plan=getPlan(); await confirmStep(plan.id, stepBtn.dataset.confirmStep);
   *                    startPlanPolling(plan.id); return; }   // 确认后续跑（executor 恢复）
   *     const cancelBtn = e.target.closest("[data-cancel-plan]");
   *     if (cancelBtn) { await cancelPlan(cancelBtn.dataset.cancelPlan); ... }
   *   });
   */
}

export function renderPlanValidationProblems(container, problems) {
  /** 当 createPlan 返回 422（plan 不合法）时，展示 PlanValidator 的 problems 列表。
   * 不变量: problems 来自后端结构化（工具缺失/schema 错/join 缺确认门/指标缺区间检查）。
   * 伪代码: container.innerHTML = `<ul class="plan-problems">${problems.map(p=>`<li>${escapeHtml(p)}</li>`).join("")}</ul>`;
   */
}
```

- **不变量**：决策点的"看结果再决定"在后端 executor 自动重规划；前端只展示重规划事件（Part J），用户在确认门处介入。
- **测试要点**：确认计划→运行→轮询；确认步骤→续跑；取消；422 problems 渲染。

---

## Part D — JoinPlan 诊断确认（`v2/join_review.js`，INV-3 关键）

> 这是整个前端最关键的安全交互——用户最容易出大错的地方（错 join = 错样本）。**逐表展示诊断 + 强制确认 + 膨胀/低命中显著告警**。

```js
export function renderJoinReview(container, joinPlan) {
  /** 渲染 JoinPlan：锚定样本表 + 每个特征表的诊断卡 + 确认控件。
   * 不变量: 只读 join payload 的结构化诊断；fan_out/shrink 必须显著告警；执行按钮在全部确认前禁用。
   * 伪代码:
   *   const cards = joinPlan.joins.map(joinSpecCardHtml).join("");
   *   const canExec = joinPlan.joins.every(j => j.confirmed);
   *   container.innerHTML = `<div class="join-anchor">样本表(锚定): ${escapeHtml(joinPlan.anchor_dataset_id)}</div>
   *     ${cards}
   *     <button data-exec-join="${escapeHtml(joinPlan.id)}" ${canExec?"":"disabled"}>执行拼接</button>`;
   */
}

function joinSpecCardHtml(spec) {
  /** 单特征表的诊断卡：键对 + 命中率 + 行数 + 唯一性 + 告警 + 去重选择 + 确认。
   * 不变量: fan_out_detected / shrink_detected 用红色告警条；键不唯一时强制 dedup 下拉（未选不能确认）。
   * 伪代码:
   *   const d = spec.diagnostics;
   *   const warns = [];
   *   if (d.fan_out_detected) warns.push(warnBar("⚠ 拼接会膨胀（笛卡尔积）：拼后约 "+d.joined_rows_preview+" 行 > 样本表 "+d.anchor_rows+" 行"));
   *   if (d.shrink_detected) warns.push(warnBar("⚠ 命中率过低："+pct(d.match_rate)+"，拼完大量缺失"));
   *   const keyPairs = spec.key_pairs.map(k =>
   *     `${escapeHtml(k.anchor_col)} ↔ ${escapeHtml(k.feature_col)} <span class="method">${escapeHtml(k.match_method)}</span>
   *      <span class="rate">实测命中 ${pct(k.match_rate)}</span>`).join("、");
   *   const dedup = d.feature_key_unique ? `<span>键唯一</span>`
   *     : `<select data-dedup="${escapeHtml(spec.feature_dataset_id)}">
   *          <option value="">⚠ 键不唯一，请选去重策略…</option>
   *          <option value="first">保留首行</option><option value="last">保留末行</option>
   *          <option value="agg_mean">聚合均值</option><option value="agg_max">聚合最大</option>
   *          <option value="abort">中止该表</option></select>`;
   *   const diag = `命中 ${d.matched_rows}/${d.anchor_rows}（${pct(d.match_rate)}） · 新增列 ${d.new_columns}（缺失率 ${pct(d.new_columns_null_rate)}）`;
   *   const confirmed = spec.confirmed
   *     ? `<span class="confirmed">✓ 已确认</span>`
   *     : `<button data-confirm-join="${escapeHtml(spec.feature_dataset_id)}">确认该表</button>`;
   *   return `<div class="join-card ${d.fan_out_detected||d.shrink_detected?'has-warn':''}">
   *             <div class="feat">${escapeHtml(spec.feature_dataset_id)}</div>
   *             <div class="keys">${keyPairs}</div><div class="diag">${diag}</div>
   *             ${warns.join("")}${dedup}${confirmed}</div>`;
   */
}

export function attachJoinHandlers(root) {
  /** 逐表确认 + 执行。确认时把所选 dedup 一并提交；键不唯一未选 dedup 不允许确认。
   * 伪代码:
   *   root.addEventListener("click", async (e) => {
   *     const cb = e.target.closest("[data-confirm-join]");
   *     if (cb) { const fid=cb.dataset.confirmJoin;
   *       const sel=root.querySelector(`[data-dedup="${cssEsc(fid)}"]`);
   *       const dedup = sel ? sel.value : null;
   *       if (sel && !dedup) { alert("该表键不唯一，请先选择去重策略"); return; }
   *       await confirmJoinSpec(getCurrentJoinId(), {feature_dataset_id:fid, dedup_strategy:dedup});
   *       await refreshJoin(); return; }
   *     const xb = e.target.closest("[data-exec-join]");
   *     if (xb) { const r = await executeJoin(xb.dataset.execJoin);
   *               if (r.fan_out) showError("拼接产生膨胀，已中止"); else showResult(r); }
   *   });
   */
}
```

- **不变量**：执行按钮在所有 spec 确认前 `disabled`；键不唯一未选 dedup 不能确认；fan_out/shrink 红色显著告警；执行后若后端返回 fan_out 错误，显著报错（INV-3 最后防线）。
- **测试要点**（重点）：fan_out 诊断→红告警 + 显示拼后行数 vs 样本行数；shrink→低命中告警；键不唯一→强制 dedup 下拉、未选不能确认；全部确认才能点执行；raw-vs-md5 的 match_method 正确展示（如 `hash:md5 实测命中 98%`）；执行膨胀错误显著提示。

---

## Part E — 子 agent 状态树（`v2/subagent_view.js`）

```js
export function renderSubAgentView(container) {
  /** 渲染当前 plan 派生的子 agent 树：scope · 授予工具 · 状态 · 结果引用。
   * 不变量: 只读 sub_agents payload；展示最小授权工具集（让用户看到子 agent 权限边界）。
   * 伪代码:
   *   onPlanChange((plan) => {
   *     const subs = plan?.sub_agents || [];
   *     container.innerHTML = subs.length ? subs.map(subAgentRowHtml).join("") : "<div class='empty'>无子 agent</div>";
   *   });
   */
}

function subAgentRowHtml(sub) {
  /** scope · 状态徽标 · 授予工具（只读展示权限边界）· 结果链接。
   * 伪代码:
   *   const tools = sub.granted_tools.map(t=>escapeHtml(`${t.plugin}.${t.tool}`)).join("、");
   *   return `<div class="subagent ${sub.status}"><span class="scope">${escapeHtml(sub.scope)}</span>
   *           ${agentStatusBadge(sub.status)}<span class="grant" data-tip="子 agent 最小授权">${tools}</span>
   *           ${sub.result_ref?`<a data-artifact="${escapeHtml(sub.result_ref)}">查看结果</a>`:""}</div>`;
   */
}
```

- **测试要点**：子 agent 行展示 scope/状态/授予工具；无子 agent 空态；结果链接打开工件预览。

---

## Part F — Plugin 管理（`v2/plugin_manager.js`）

```js
export function renderPluginManager(container) {
  /** plugin 列表 + 上传 + 启停 + 删除 + tools 展开。
   * 伪代码:
   *   const data = await listPlugins(true);
   *   container.innerHTML = uploadBoxHtml() + data.plugins.map(pluginRowHtml).join("");
   */
}

function pluginRowHtml(p) {
  /** 名称 · 版本 · builtin 标记 · 启停开关 · tool 数 · 删除(非 builtin) · tools 展开。
   * 伪代码:
   *   const toggle = `<input type="checkbox" data-toggle-plugin="${escapeHtml(p.name)}" ${p.enabled?"checked":""}>`;
   *   const del = p.builtin ? "" : `<button data-remove-plugin="${escapeHtml(p.name)}">删除</button>`;
   *   return `<div class="plugin"><b>${escapeHtml(p.display_name)}</b> v${escapeHtml(p.version)}
   *           ${p.builtin?'<span class="builtin">内置</span>':''} ${toggle}
   *           <span>${p.tool_count} tools</span>${del}
   *           <button data-show-tools="${escapeHtml(p.name)}">查看工具</button></div>`;
   */
}

export function attachPluginHandlers(root) {
  /** 上传 / 启停 / 删除 / 展开 tools。
   * 伪代码:
   *   - 上传: file input change → uploadPlugin(file) → 处理 201/409/422 → refresh
   *   - 启停: checkbox change → setPluginEnabled(name, checked) → refresh
   *   - 删除: confirm → removePlugin(name) → refresh（builtin 后端会 400，前端也不出删除按钮）
   *   - 查看工具: listPluginTools(name) → 展示 input/output schema
   */
}
```

- **不变量**：builtin 包不出删除按钮（后端也拒）；上传失败按状态码给清晰提示（409 重复/422 manifest 错）。
- **测试要点**：plugin 行渲染；启停 checkbox 联动；上传错误码提示；builtin 无删除；tools 展开显示 schema。

---

## Part F2 — Skill 模板管理（`v2/skill_manager.js`）

用户可编写 Workflow 模板（= skill，Phase 2 Part C2）的可视治理面：看哪些 skill 生效/被拒、为什么被拒、改完后 reload，以及编辑时实时校验。

```js
export function renderSkillManager(container) {
  /** skill 列表（active / disabled / rejected + 问题）+ reload + 校验编辑框。
   * 伪代码:
   *   const report = await listSkills();   // {active:[], disabled:[], rejected:[[id,problems]]}
   *   container.innerHTML =
   *     `<button id="reloadSkills">重新加载 skills</button>`
   *     + report.active.map(id => skillRowHtml(id, "active", [])).join("")
   *     + report.disabled.map(id => skillRowHtml(id, "disabled", [])).join("")
   *     + report.rejected.map(([id, ps]) => skillRowHtml(id, "rejected", ps)).join("")
   *     + skillValidateBoxHtml();          // 粘贴/编辑 JSON 实时校验
   */
}

function skillRowHtml(id, status, problems) {
  /** id · 状态徽章 · rejected 时列出 problems（每条转义）。
   * 伪代码:
   *   const badge = {active:"生效", disabled:"已停用", rejected:"被拒"}[status];
   *   const probs = problems.length
   *     ? `<ul class="skill-problems">${problems.map(p=>`<li>${escapeHtml(p)}</li>`).join("")}</ul>` : "";
   *   return `<div class="skill skill-${status}"><b>${escapeHtml(id)}</b>
   *           <span class="skill-badge ${status}">${badge}</span>${probs}</div>`;
   */
}

export function attachSkillHandlers(root) {
  /** reload + 编辑框实时校验。
   * 伪代码:
   *   - reload: #reloadSkills → reloadSkills() → 重渲染（active/rejected 变化即时可见）
   *   - 校验: 编辑框 input 防抖 → validateSkill(JSON.parse(text)) → 显示 problems（空=合法）；
   *           JSON 解析失败本地直接提示，不打后端。
   */
}
```

- **不变量**：rejected skill 只读展示问题、不可"强制启用"（启用唯一路径是改文件让它过校验，与后端 INV-7 一致）；reload/validate 是本机配置操作，远程被后端守卫拦（前端按 403 给"仅本机可用"提示）。
- **测试要点**：active/disabled/rejected 三态渲染；rejected 的 problems 列表转义显示；reload 触发重取；校验编辑框合法/非法回显；JSON 语法错本地提示；问题文本含 `<img onerror>` 只显示文本。

---

## Part G — 目标→计划（`v2/workflow_create.js`）

```js
export function renderGoalComposer(container) {
  /** 自然语言目标输入 + 档位选择 + novel_mode 选择 → 创建 plan。
   * 伪代码:
   *   container.innerHTML = `
   *     <textarea id="goalInput" placeholder="描述你的目标，如：把这几张表拼起来做一个贷前A卡"></textarea>
   *     ${capabilitySelectHtml()}      // Part I
   *     <label>novel 模式 <select id="novelMode"><option value="">自动(按档位)</option>
   *        <option value="plan_ahead">先出完整计划</option>
   *        <option value="explore">边探索边规划</option></select></label>
   *     <button id="createPlanBtn">生成计划</button>`;
   */
}

export function attachGoalHandlers(root, taskId) {
  /** 提交目标 → createPlan → 展示计划(或 422 problems) → 用户确认。
   * 伪代码:
   *   root.querySelector("#createPlanBtn").onclick = async () => {
   *     const body = {goal: val("#goalInput"), tier: val("#tierSelect")||undefined,
   *                   novel_mode: val("#novelMode")||undefined};
   *     try { const plan = await createPlan(taskId, body); setPlan(plan); renderPlanView(...); }
   *     catch (e) { if (e.status===422) renderPlanValidationProblems(..., e.detail.problems);
   *                 else showError(e); }
   *   };
   */
}
```

- **不变量**：意图路由/模板命中由后端 IntentRouter 决定；前端只提交目标 + 展示返回的 plan（含 source=template|generated）。
- **测试要点**：目标提交→plan 展示；422→problems；档位/novel_mode 传参。

---

## Part H — 工件预览（`v2/artifact_view.js`）

```js
export function renderArtifact(container, artifactRef) {
  /** 按 output_ref 类型预览：dataset(表预览) / metrics(结构化指标) / artifact(文件:报告/图) / value。
   * 不变量: dataset 只预览前 N 行（后端分页）；report 用现有 word_preview / excel 下载；图直接 <img>。
   * 伪代码:
   *   const [kind, id] = artifactRef.split(":");
   *   if (kind==="dataset") { const d = await apiGet(`/api/datasets/${id}/preview?rows=50`);
   *                           container.innerHTML = datasetTableHtml(d); }    // 列 profile + 前 50 行
   *   else if (kind==="metrics") { const m = await apiGet(`/api/step-outputs/${id}`);
   *                                container.innerHTML = metricsHtml(m); }    // 复用 render-metrics
   *   else if (kind==="artifact") { container.innerHTML = artifactFileHtml(id); }  // 报告下载/预览/图片
   */
}

function datasetTableHtml(preview) {
  /** 列名 + 语义角色 + null率 + 前 N 行（脱敏样例）。
   * 不变量: 展示 ColumnProfile（语义角色/null率），让用户确认列识别对不对；样例已脱敏。
   */
}
```

- **测试要点**：dataset 预览展示列 profile + 前 N 行；metrics 复用指标渲染；报告工件给下载/预览入口；图片直接显示。

---

## Part I — 能力档位选择器（`v2/capability.js`）

```js
export async function capabilitySelectHtml() {
  /** 渲染档位下拉 + 当前档说明（conservative/balanced/autonomous）。
   * 伪代码:
   *   const data = await listCapabilityTiers();   // 后端返回三档 + 默认值
   *   const opts = data.tiers.map(t=>`<option value="${escapeHtml(t.name)}">${escapeHtml(t.name)} — ${escapeHtml(t.summary)}</option>`).join("");
   *   return `<label>能力档位 <select id="tierSelect">${opts}</select></label>`;
   */
}

export function renderTierSettings(container) {
  /** 设置页：选择默认档位 + 展示各档的自由度（autonomy/explore/replan 上限），持久化。
   * 不变量: 改档位只改"给模型多少规划自由"，UI 文案说明领域护栏恒定不受档位影响。
   */
}
```

- **测试要点**：档位下拉渲染；选择持久化；设置页展示各档差异 + "护栏恒定"说明。

---

## Part J — 自适应 loop 进度（`v2/loop_progress.js`）

```js
export function renderLoopEvents(container) {
  /** 展示 replan / explore 续段 / 无进展中断 等自适应事件流（来自 plan.replan_count + hook 事件）。
   * 不变量: 只读结构化事件（plan.replanned/explore_segment）；让用户看到"agent 看了结果改了计划"。
   * 伪代码:
   *   subscribe("v2.loopEvents", (events) => {
   *     container.innerHTML = events.map(evtHtml).join("");
   *   });
   */
}

function evtHtml(evt) {
  /** 事件行：类型 · 原因 · 时间。
   * 伪代码（类型）:
   *   replan(decision_point): "◆ 决策点：根据「X」结果重规划了后续步骤"
   *   replan(failure):        "↻ 失败修复：检测到「fan-out」，自动插入去重步重试"
   *   explore_segment:        "→ 探索续段：规划了下一段 N 步"
   *   no_progress:            "⏸ 检测到无进展，已暂停等你介入"
   *   return `<div class="loop-evt ${evt.type}">${escapeHtml(evtLabel(evt))}<time>${escapeHtml(evt.at)}</time></div>`;
   */
}
```

- **不变量**：loop 事件让自适应"可见"——用户能看到 agent 为什么改了计划，符合"可治理"。无进展中断要醒目（提示用户介入）。
- **测试要点**：replan/explore/no_progress 事件渲染；无进展醒目；事件按时间排序。

---

## Part K — 测试计划

| 文件 | 覆盖 |
|------|------|
| `tests/test_frontend_v2_plan.py` | plan DAG 渲染、步骤状态、决策点标记、确认门、deterministic 失败红标 |
| `tests/test_frontend_v2_join.py` | **JoinPlan 诊断：fan_out/shrink 告警、键不唯一强制 dedup、全确认才能执行**（核心） |
| `tests/test_frontend_v2_plugin.py` | plugin 列表/启停/上传错误码/FormData/builtin 无删除/tools schema |
| `tests/test_frontend_v2_skill.py` | skill active/disabled/rejected 三态渲染、problems 转义、reload、编辑框校验、远程 403 提示 |
| `tests/test_frontend_v2_subagent.py` | 子 agent 树、最小授权展示 |
| `tests/test_frontend_v2_artifact.py` | dataset 上传 FormData/预览/metrics/报告工件 |
| `tests/test_frontend_v2_capability.py` | 档位下拉/持久化/护栏恒定说明 |
| `tests/test_frontend_v2_loop.py` | replan/explore/no_progress 事件流 |
| `tests/test_frontend_v2_smoke`（浏览器/Playwright，可选） | 无 module 404、目标→计划→确认→进度全流程可达 |

- 纯函数（html 构造）用 node import 测；交互用静态断言 + 可选浏览器 smoke。
- **安全**：所有动态文本走 escapeHtml 的断言（恶意 plugin 名/数据集名/scope 输入 `<img onerror>` 只显示文本）。

---

## Part L — 任务执行顺序

```text
1. A api_v2 + state_v2 + main_v2（装配骨架，可 mock 后端）
2. B plan_view（中心视图）
3. C plan_confirm（确认门）
4. D join_review（INV-3 关键，重点测试）
5. F plugin_manager
6. F2 skill_manager（用户 skill 模板治理面）
7. G workflow_create（目标→计划）
8. H artifact_view
9. I capability
10. E subagent_view
11. J loop_progress
12. K 测试（含 join 重点 + 安全 escapeHtml + 可选浏览器 smoke）
```

每项 atomic commit。前端 V2 完成标志：用户能用自然语言提目标→看到生成/模板计划→逐步确认（含决策点/join 强制确认诊断）→看子 agent 与自适应 loop 进度→管理 plugin 与用户 skill 模板（看生效/被拒原因、reload）→预览工件→选能力档位；全部只读结构化 payload、无文本解析、动态文本全转义；JoinPlan 的膨胀/低命中/键不唯一有显著告警和强制确认（INV-3 在前端落地）。

---

*前端 V2 把"看不见的编排"变成"可治理的操作面"：计划是可见的 DAG、决策点和 join 是强制确认的门、自适应重规划是看得见的事件、子 agent 权限是透明的。这正是风控场景要的——agent 很能干，但每一步都在用户的注视和确认之下。无构建 ES Modules 保证它和整个平台一样能离线私有化部署。*
