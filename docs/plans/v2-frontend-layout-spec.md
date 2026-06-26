# 三区前端布局 Spec

> V2 完善总计划第 5 份 spec（见 [v2-completion-plan.md](v2-completion-plan.md)）。
> 决策 #7:顶栏只状态/报错;中间对话流放所有富表(append);右栏只看流程 + 下载预览。
> **三区骨架已存在**——本 spec 是把新 3 类任务接进去 + 验证共存,不是重搭。

---

## 0. 地面真值（已核对 index.html / app.js）
- 左:`#taskSidebar`(任务列表/导航)。
- 中:`#resultWorkspace` = `#taskHero`(标题 + `#actionStatus` 状态 pill + `#actionErrorDetail` 报错 + `#taskSnapshot` 元信息)+ `#resultScrollContent`(验证写死 stage 区:`#scanSection/#notebookSection/#metricSection/#reportSection` + `#agentConversationPanel`>`#agentMessages`)+ `#agentComposer`。
- 右:`#progressRail.step-rail` > `#workflowStepper`(验证流程步骤,`renderWorkflowStepper`)。
- composer 已含 `#agentAcceptanceModeSelect`(`默认权限`/`自动审查`)+ 模型/effort chip。
- V2 计划渲染 `static/js/v2/plan_view.js`(planHtml/stepRowHtml/startPlanPolling)现挂在**设置壳 `#v2RuntimeMount`**,需改挂到 `#progressRail`。
- 富表渲染 `renderMetricTableSection` 家族已有;Excel 下载管道(验证报告)已有。

---

## 1. 三区映射（按 task_type 切换内容,容器复用）
| 区 | 容器 | 新 3 类任务(拼接/特征/建模) | 验证(不改) |
|---|---|---|---|
| **顶** | `#taskHero` | `#actionStatus`(整体状态)+`#actionErrorDetail`(报错);**`#taskSnapshot` 暂保留**(用户之后再定怎么改) | 原样 |
| **中** | `#resultScrollContent` | **只用 `#agentConversationPanel`**:append-only 对话 + 内联富表(`renderMetricTableSection`);**隐藏 4 个验证 stage 区** | 原样(stage 区) |
| **右** | `#progressRail` | 挂 **`plan_view`**(V2 计划步骤/进度/确认按钮 + loop 事件)+ **下载/预览区** | `#workflowStepper` 原样 |

## 2. 右栏(progressRail)改造
> **后端前置（C4，须先于本节做）**:① `PlanStep/StepTemplate` 当前**无 `phase` 字段**(`contracts.py`/`templates/__init__.py`)——phase 分组依赖它,先加并串序列化;② `plan_id 来自任务的计划`需要 `GET /tasks/{task_id}/plans`(现仅 `GET /plans/{plan_id}`,`PlanRepository` 无 list-by-task)。**首建任务** create 接口已返回 plan_id 可直接 poll;**重开/恢复任务**才必需该查询接口。见总计划 §八步骤1。
- 新任务:`renderPlanView(#progressRail)` + `startPlanPolling(plan_id)`(plan_id 来自任务的计划)。复用 plan_view 的步骤行/状态徽章/进度条/确认按钮 + loop_progress。
- **phase 分组展示**(决策 #6):按 `phase` 字段把 PlanStep 折叠成大步骤标题(数据准备/特征/建模/报告),小步骤挂下面。
- **下载/预览 = 挂在对应步骤上**(非侧栏底部):每个产出步骤在右栏其**行内带步骤动作按钮**(复用验证 `data-step-action` 模式,如 `downloadExcelAnalysis`)。例:建模过程中做了特征分析 → "下载特征分析报告"按钮出现在右栏**特征分析那个大步骤**的合适位置;建模步骤带"下载模型报告/模型文件"。预览复用 `artifact_view`、下载复用验证管道。
- 验证任务:仍渲 `workflowStepper`。`#progressRail` 按 task_type 二选一渲染。

## 3. 中间对话流改造
- 新任务:`#agentConversationPanel` 承载**全部**——每个门/分析/报告由驱动 **append** 一条 assistant 消息,带内联富表(`message.metadata.tables → renderMetricTableSection`)。
- **隐藏**验证专属 `#scanSection/#notebookSection/#metricSection/#reportSection`(按 task_type)。
- **append-only**:重跑追加在下,旧内容保留可回看(决策:历史不覆盖)。
- 手动模式:门里渲**结构化控件**(滑杆/勾选/下拉/按钮),嵌在对话流消息中;agent 模式:LLM 话术 + 自由文本输入。

## 4. 顶栏(taskHero)
- 标题、`#actionStatus`(running/awaiting/done/failed 整体状态)、`#actionErrorDetail`(报错)。
- `#taskSnapshot` **暂保留原样**(用户之后再定怎么精简,不在本轮动)。

## 5. composer / 模式
- `#agentAcceptanceModeSelect` **复用**,按 task_type **改名**:自动审查→`自动拼接/自动分析/自动建模`;默认权限不变(仅 agent 模式)。
- **run_mode=手动(参考验证手动模式)**:**完全隐藏自由文本 composer**,纯**控件 +「继续/暂停」**驱动——右栏步骤行内带动作按钮(同验证 `step-action-button`),门里控件(滑杆/勾选/下拉)+「继续」推进、「暂停」停;**无 LLM、无自由输入**。
- **run_mode=agent**:自由文本 composer + 模式选择器。

### 5.1 能力档位选择（补 IA 重构遗留缺口）
- **现状**:设置里的"能力档位"面板是**只读**展示三档(稳健/均衡/自治);真正的选择器 `#tierSelect` 在已退役的 `workflow_create.js` 里 → **当前无入口可选**,后端回退 `tier_from_settings`。
- **改造**:
  1. **设置面板改可选**:三档点选,存为**全局默认档**(`marvis_v2_selected_tier` / settings)。
  2. **创建任务弹窗加档位选择**:per-task,**默认 = 设置里选的那档**;随计划创建传 `tier`(`plans.py` 已支持 `body.tier`)。
- **备注(防误解,写进 UI 文案)**:**档位只影响自治程度(最多重规划几次 `max_replan_iterations`、能否加模板外探索步 `allow_explore_mode`),不影响模型效果/确定性/确认门/安全护栏**。默认「均衡」,常规无需动。
- 与本计划的关系:仅约束决策 #5「调整→replan」的**重规划次数上限**;模板流程形状固定,`max_plan_depth`/explore 基本不触发。

## 6. 切换矩阵
- `task_type ∈ {data_join, feature_analysis, modeling}` → 新三区行为(plan_view 右栏 + 对话流中 + 极简顶)。
- `task_type == validation` → 完全原样(workflowStepper + stage 区),**零改动**。
- `task_type ∈ {strategy, vintage}` → 入口保留,暂不接(本轮不开发)。
- 渲染分发集中在一处(按 task_type 选右栏渲染器 + 中间区可见性),避免散落。

## 7. 缓存破除
- 改 `app.js`/`index.html`/css → 静态版本串含 mtime 自动变(`app.py _STATIC_VERSION_FILES`);开发时硬刷新 Cmd+Shift+R。

## 8. 新建 vs 复用
- **复用**:三区 DOM 骨架、`plan_view/loop_progress/state_v2`、`renderMetricTableSection`、`artifact_view` 预览、Excel 下载、acceptance 选择器、composer。
- **新建**:`#progressRail` 按 task_type 渲 plan_view(替换/并存 workflowStepper)、右栏下载区、中间区 append 富表 + 隐藏验证 stage、顶栏精简、acceptance 改名、手动模式门控件组件、渲染分发开关。

## 9. 已锁小项
1. **顶栏 taskSnapshot**(已定):**暂保留原样**,用户之后再思考怎么改,本轮不动。
2. **手动模式 composer**(已定):**完全隐藏自由文本输入**,纯控件 +「继续/暂停」,**照模型验证手动模式**(右栏 `step-action-button` + 步骤推进)。
3. **下载位置**(已定):**挂在右栏对应大步骤的合适位置**(步骤行内动作按钮),**不**放侧栏底部。
