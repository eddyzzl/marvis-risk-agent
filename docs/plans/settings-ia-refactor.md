# 设置弹窗信息架构重构方案

> 目标：让"系统设置"只装真正的设置；把混进来的运行/观测/数据/资产视图各归其位。
> 触发：用户指出"计划与执行不是设置，不该放这里"。审计发现这是系统性问题——整个 V2 运行层都借住在设置弹窗里。

## 0. 决议（本轮已定）

- **D1 已定：对话流为主，V2 结构化面板退役。** agent 任务的主体验是工作区里现成的**对话面板 + 输入框 + 右侧步骤栏**（后端 `/api/tasks/{id}/agent/start` + `/agent/messages`，选中 agent 任务即显示）。V2 的"计划与执行/运行审计"（goal/plan/join/subAgent/loop/artifact）是旧的结构化路线，**退役**——不搬到工作区、不重建面板。
  - **D1 豁免（2026-06-25 补，被 [v2-completion-plan.md](v2-completion-plan.md) / [v2-frontend-layout-spec.md](v2-frontend-layout-spec.md) 取代此一处）**：D1 退役的是**旧结构化创建流**（goal composer 编排器 + 设置壳里的多面板"运行审计"workbench），"不搬工作区"针对的是**重建那整套 workbench**。它**不**禁止把 `plan_view.js` + `loop_progress.js` 这**一对模块**作为**右侧步骤栏的"计划步骤"视图**复用——这正是本文件 §5「可选(增强):右侧步骤栏对 agent 任务改显计划步骤」(line 112)预告的方向。即:goal composer / 设置 workbench 仍退役；`plan_view` 标记为**可迁移**到 `#progressRail`。后端 `PlanExecutor` 从未退役(见 §1.3),仍是执行真源。
- **重大简化**：原"阶段2 把 V2 面板搬工作区 + 补后端持久化"**取消**。整个重构变成"设置做减法 + 退役旧面板 + 数据/资产归位 + 修一致性"。
- 因此 §3.3 / §4 / §5 中"搬运行层、补 `GET /tasks/{id}/plans`、selectTask 持久化、挂载迁移"等大工程**不再需要**；保留下面文字仅作历史/备选记录，实际按本节执行。
- 仍需处理的硬约束：`createTaskAndScan`（app.js:6240）新建 agent 任务时打开的是 V2 计划弹窗（`openV2WorkspaceWithGoal`），与"选中即显示对话"不一致——**改为直接进对话工作区**（把目标作为首条消息/起手）。

## 0.1 参考 Claude 设置后的最终 IA（本节为准）

用户提供 Claude.ai 设置弹窗为参考：设置/配置/记忆/插件/技能都收在**一个组织良好的设置弹窗**里（记忆页＝开关 + "View and manage memory" 下钻一页搞定），**显示偏好留在弹窗外**。据此：**不再另起"数据/资产区"**（推翻 §3.2 方案 A/B），数据/资产留在设置弹窗内、组织好即可。

**最终设置弹窗结构（11 项 → ~6 项）：**

分组「设置」
1. 执行环境
2. 模型引擎
3. 能力档位（Capabilities）
4. **记忆**（三合一）：顶部放记忆策略两个开关（引用跨任务记忆 / 自动沉淀）+ 平台强制徽章；下方一行"查看与管理记忆 ›"下钻，打开记忆浏览器（原始记忆 / 进化沉淀，用内部切换）。← 合并原 记忆策略 + 记忆库 + 进化沉淀。

分组「扩展 / 自定义」
5. 插件
6. Workflow 模板（Skills）
7. 草稿工具：作为「插件」页内的"待转正草稿 ›"下钻（草稿最终转正为插件，归属自然），或独立项 — 见 D5。

**保留在弹窗外、不动**：侧栏「显示偏好」popover（排序/分组等列表展示偏好，`#sidebarSettings`/`#settingsMenu`）。

**退役（D1）**：计划与执行、运行审计。

> 与 §2 表的差异：记忆/模板/插件不再"移出设置"，而是留在设置弹窗内重组；记忆三项合一为一页（带下钻）。§2 表的"数据/资产区"目标作废，以本节为准。

## 1. 现状关键事实（来自架构调查）

1. **任务工作区已存在**：`#validationWorkspace`（index.html:722）在选中任务时由 `is-empty` 类切到 `resultWorkspace`（中栏，index.html:950）+ `progressRail`（右栏，含 `workflowStepper`，index.html:1086）。**运行类视图有现成的家**。
2. **V2 运行层可迁移**：`mountV2(root, {taskId})`（main_v2.js:66）把 11 个面板挂到任意 root，按元素 `__marvisV2MountState` 守卫（非模块单例），面板模块全部 container/事件委托驱动、零弹窗硬引用。唯一绑死弹窗的是：(a) `mountV2Runtime()`（app.js:1841）只挂 `#v2RuntimeMount`；(b) 该元素只存在于设置弹窗（index.html:552）；(c) 视图切换是 CSS 限定在 `.governance-settings-dialog[data-v2-view]`（v2-workbench.css:144-174）。基础 CSS 其实是把所有面板平铺成 grid，弹窗 override 才隐藏并按视图露出。
3. **计划与执行是真功能且强绑任务**：后端 `POST /api/tasks/{task_id}/plans`（plans.py:38）+ 真实 `PlanExecutor.run()` 循环（executor.py:51，含子Agent、确定性+LLM 复核、replan/explore 循环事件、finalize）。前端目标编排器无任务时硬阻断（workflow_create.js:160）。
4. **但计划是会话态、未做按任务持久**：没有"列出某任务计划"的端点（只有 `GET /plans/{plan_id}`）；`selectTask`（app.js:3906）不加载也不重置 v2 计划状态，`v2.currentPlan` 跨任务共享。→ 关弹窗 / 切任务，计划就从视图消失。
5. **运行审计是两种东西被 CSS 拼一起**：`data-v2-view="audit"` 同时露出 memory/loop/artifact 三个面板。loop/子Agent/工件读 `currentPlan`（按任务）；但 memory 面板是 `listMemoryDistillations`（**全局**，无 task_id），和记忆库/进化沉淀重复。**必须拆**。
6. **各项作用域**：
   - 能力档位：**双重**。全局默认存 `settings/llm.json:capability_tier`（llm_settings.py:81）+ localStorage；按计划覆盖已在目标编排器里（workflow_create.js:94）。
   - 草稿工具：**按任务**的审批队列，`draft_tools` 表按 `(task_id,status)` 索引（db.py:547），由 Agent 跑任务时产出；转正/拒绝管理员门控、写入全局插件注册表。
   - 记忆库/进化沉淀：**全局**跨任务数据存储（单 `db_path`），`source_task_id` 只是过滤不是作用域。
   - Workflow 模板：**全局**用户自编资产（`workspace/skills/*.json`）。
7. **Nav 机制数据驱动、删除成本低**：移除项 = 删 nav 按钮 + 面板 section（若非共享）+ `governanceSettingsCopy` 项 + `governanceRefreshActions` 项（若有）+ runtime 项还要删对应 `data-v2-view` CSS 块。有安全 fallback，删 key 不会崩。
8. **两个外部入口是硬约束**：
   - `createTaskAndScan`（app.js:6240）创建 agent 任务时调 `openV2WorkspaceWithGoal` → 进入计划视图。**这是进入计划的主流程，迁移后必须重定向到新家**，否则建任务即坏。
   - 记忆的内联"查看"按钮（`handleAgentMemoryInlineInspect`，app.js:2138）打开记忆库/进化沉淀，迁移后要重指向。
9. **记忆有两套 UI**：独立 `#memory-browser` 面板（nav 项 4/5）vs mountV2 的 `memoryPanel`（audit 视图露出的那块）。迁移时要选一套为准，避免重复。

## 2. 目标信息架构（三个家）〔部分已被 §0.1 修订，以 §0.1 为准〕
> ⚠️ 本表的"数据/资产区"目标已作废（见 §0.1 与 §0 line 10/34）；记忆/模板/插件改为留在设置弹窗内重组，不再"移出设置"。本节保留作历史对照。

| 项 | 现属 | 判定 | 目标家 |
|---|---|---|---|
| 执行环境 | 设置 | 配置 | **设置** ✅ |
| 模型引擎 | 设置 | 配置 | **设置** ✅ |
| 记忆策略 | 设置 | 配置 | **设置** ✅ |
| 能力档位 | 运行组 | 全局默认(配置)+按计划覆盖 | **设置**（只留全局默认）；按计划选择已在工作区编排器里 |
| 插件 | 扩展组 | 扩展管理(配置邻接) | **设置** ✅ |
| 记忆库 | 记忆组 | 全局数据 | **数据/资产区** |
| 进化沉淀 | 记忆组 | 全局数据 | **数据/资产区**（与记忆库合并为一套"Agent 记忆"） |
| Workflow 模板 | 扩展组 | 全局资产 | **数据/资产区** |
| 计划与执行 | 运行组 | 按任务·运行操作 | **任务工作区** |
| 运行审计(loop/子Agent/工件) | 运行组 | 按任务·观测 | **任务工作区**（跟计划走） |
| 运行审计·记忆审计 | 运行组 | 全局数据(重复) | **拆除**，并入数据区的"Agent 记忆" |
| 草稿工具 | 扩展组 | 按任务·审批队列 | **任务工作区**（按任务复核） |

**最终设置弹窗只剩 5 项**：执行环境、模型引擎、记忆策略、能力档位、插件。

## 3. 三个家怎么落地

### 3.1 设置弹窗（瘦身）
- 移除 nav 项：记忆库、进化沉淀、Workflow 模板、草稿工具、计划与执行、运行审计。
- 能力档位**保留但回归"默认设置"语义**：它写的是全局默认；删掉它作为 runtime 面板的框架感，放到"环境/代理"组里作为一个普通设置项（下拉选默认档位，写 `settings/llm.json`）。按计划的临时覆盖维持在目标编排器（已有）。
- 分组从 4 组（环境/记忆/扩展/运行）收成 2-3 组（如：环境、记忆、扩展），不再有"运行"组。

### 3.2 数据/资产管理区（全局，非设置）
全局数据/资产不该混进"设置"，但它们确实是全局管理面。两个选项：
- **方案 A（推荐，轻）**：仍在同一弹窗内，但**重命名/重组**为清晰的两段——"设置"与"数据与资产"，视觉上分开（不同分组标题/分隔）。弹窗标题从"系统设置"改为更中性的（如"系统中心"/"管理"）。记忆库+进化沉淀合并成一个"Agent 记忆"项（一套 UI，原始/沉淀用内部切换或两子项），Workflow 模板作为"模板库"项。
- **方案 B（重）**：在侧栏新增独立入口"Agent 管理 / 知识库"，把数据/资产移出设置弹窗到独立视图。
> 倾向 A：数据/资产是全局管理，和设置同处一个"中心"弹窗、但分区清晰，成本低、心智清楚。

### 3.3 任务工作区（按任务·运行/观测/复核）
把 V2 运行层从设置弹窗搬到工作区。落点选项（`resultWorkspace` 内新增一个"代理运行"标签页 / 或 `progressRail` 区 / 或 `resultScrollContent` 新 section）。推荐：**在 `resultWorkspace` 头部增加一个视图切换（验证结果 ↔ 代理运行），代理运行下挂 V2 运行面板**，仅 agent 模式任务可见。
- 计划与执行：goal/plan/join/子Agent 面板。
- 运行审计：loop/子Agent/工件面板（**去掉记忆面板**）。
- 草稿工具：该任务的草稿复核（按 task_id 列出 + 试运行 + 转正/拒绝）。
- 实现：把 `#v2RuntimeMount` 物理移到工作区新容器，或在工作区新建第二个 `mountV2` root；视图切换 CSS 从 `.governance-settings-dialog[data-v2-view]` 改写为工作区容器作用域。

## 4. 必须配套的新建工作（gaps）

搬"计划与执行/运行审计"到工作区要成为**真正按任务**的体验，必须补：
1. **后端**：新增 `GET /api/tasks/{task_id}/plans`（或 `/active-plan`），用于按任务列出/重载计划。
2. **前端**：`selectTask` 增加 hook——切任务时加载该任务的活动计划进 `v2.currentPlan`，并在切任务时 `resetV2State`（避免串台）。
3. **挂载迁移**：`#v2RuntimeMount` 移入工作区容器 / 或第二 root；重写视图切换 CSS（脱离弹窗作用域）。
4. **重定向主入口**：`createTaskAndScan`（app.js:6240）的 `openV2WorkspaceWithGoal` 改为打开工作区的代理运行视图，而非设置弹窗。
5. **拆审计**：从 audit 视图移除全局 memory 面板，只留 loop/子Agent/工件。
6. **草稿迁移**：`loadDraftTools`/`#draftToolsList` 整套自包含逻辑移到工作区按任务调用；转正/拒绝仍调管理员门控端点。
7. **记忆合一**：选定一套记忆 UI 为准（建议独立 `#memory-browser` 那套，功能更全），删除/复用 mountV2 的 memoryPanel。
8. **清理外部入口**：`openAgentMemoryDialog`、记忆内联"查看"、`openDraftToolsDialog`、`showV2WorkspaceDialog` 等重指向或移除。

## 5. 分阶段实施（参考 Claude 设置后的最终版，以 §0.1 为准）

- **阶段 1（退役 V2 操作面板 + 修一致性，对症核心）**：移除 nav「计划与执行」「运行审计」；删 `data-v2-view="plan"`/`"audit"` 的 CSS 露出块 + 对应 `governanceSettingsCopy` 项；mountV2 的 goal/plan/join/subAgent/loop/artifact 面板不再被露出（可从 panelDefinitions 裁掉）。**关键：`createTaskAndScan`（app.js:6240）的 `openV2WorkspaceWithGoal` 改为进对话工作区**；清理 `showV2WorkspaceDialog`/`openV2WorkspaceWithGoal`/`seedV2GoalComposer` 等旧入口。→ 改后用预览实测新建 agent 任务走对话流正常。
- **阶段 2（记忆三合一）**：把 记忆策略 + 记忆库 + 进化沉淀 合并为一页「记忆」——顶部开关，下方"查看与管理记忆 ›"下钻打开浏览器（原始/沉淀内部切换）。记忆 UI 两套择一为准（建议独立 `#memory-browser`）。移除多余两个 nav 项。
- **阶段 3（分组重排）**：nav 收成「设置」（执行环境/模型引擎/能力档位/记忆）+「扩展」（插件/Workflow模板/草稿）。能力档位回归"全局默认"语义。
- **阶段 4（草稿归位）**：草稿改为「插件」页内"待转正草稿 ›"下钻，或独立项（见 D5）。
- **阶段 5（清理）**：删死 CSS、未用的 `data-governance-jump`、退役面板残留模块、外部入口残链。
- **可选（增强）**：右侧步骤栏对 agent 任务从"验证步骤"改显示**计划步骤**。

> 不动：侧栏「显示偏好」popover（排序/分组）保持现状。
> 取消：原"搬 V2 面板进工作区 + 补后端持久化 + 另起数据区"——因 D1 与 §0.1 修订作废。

## 6. 决策点

- **D1（已定）**：对话流为主，V2 结构化面板退役。见 §0。
- **D2**：数据/资产用方案 A（同弹窗重组分区）还是方案 B（侧栏独立入口）？建议 A。
- **D3（已定向）**：能力档位保留为设置项=全局默认；按运行的自治控制由对话输入框的"审查模式"chip 承担；V2 编排器里的 tier 选择随 V2 退役。
- **D4**：记忆两套 UI 以哪套为准？建议独立 `#memory-browser`。
- **D5**：草稿改为按任务复核，还是做成全局"待审批"汇总？（对话流为主后，草稿是 agent 按任务产出，倾向：全局"待审批草稿"入口更简单，或先保留可访问、只是移出"设置"。）
