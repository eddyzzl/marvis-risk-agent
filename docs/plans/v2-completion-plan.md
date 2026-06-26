# V2 完善总计划（整体改造）

> 本文是一份**整体架构 + 路线主轴**规格，综合自一次结构化盘问（12 条决策）。细节 spec 按"一份一份审"逐个钻（先 JOIN 阶段）。
> 目标状态（用户原话）：欢迎页 6 个入口都能直接进对应任务；agent 模式下走 V2 DAG，通过对话让 LLM 编排+执行并在右栏展示流程；顶栏只展示报错/整体状态；拆出的每个步骤可在中间对话流可视化确认并按用户输入调整；类似模型验证 agent 的"大步骤套小步骤"。先做 特征分析/数据拼接/模型开发；模型验证不改；策略/vintage 不开发。

---

## 一、敲定的 12 条架构决策

1. **LLM 与引擎分工**：LLM 负责"编排 + 决定调哪个工具/传什么参数/下一步做什么"；**确定性引擎执行工具**（`runner.invoke`）。LLM 不亲手算数值（守住"确定性指标"不变量）。
2. **计划来源**：**模板优先的混合**。有模板从骨架起步、LLM 对话式调整；纯 LLM 生成（`Planner.generate`）作非常规兜底，永远过 `PlanValidator`。
3. **补模板**：给 特征分析 / 数据拼接 补 V2 模板（数据拼接带**强制确认门**）。
4. **执行所有权**：**真·`PlanExecutor` 拥有执行**；本会话临时建的 `ModelingSession` 收编进 V2 计划（一套引擎）。
5. **确认门语义**：每个门 = **确认 / 提指令**。指令交 LLM 路由——改本步参数→**重跑本步**；结构改动→**`Planner.replan`**（用户指令当约束）。
6. **大小步骤**：**扁平 DAG + 展示用 `phase` 分组**（执行语义不变）。确认门只放有意义的复核点；小步骤连续跑不停，大步骤跑完一次性摊效果给用户。
7. **三区布局**：顶栏=只报错/整体状态；中间对话流=**所有**分析/产出/表**内联**（复用 `metric_tables.py → renderMetricTableSection`）；右栏=只看流程(步骤/进度)+文件下载/预览。
8. **双层确认**：第一条对话先**填槽**（`detect_setup` 识别样本/目标/切分）+**亮计划**→**计划级确认/调整**→跑→**步骤级确认门**。复用 create→confirm→run。
9. **一个通用「计划对话驱动」**：盯执行器状态→每门把步骤产出经"工具→表格转换登记表"渲成内联富表→组织门话术→解析回复(确认恢复/指令路由)。任务差异全落在**模板 + 表格转换**，不写多套驱动。
10. **两个一等模式（共享底座）**：
    - **手动模式**：固定 workflow、**无 LLM**、不可改步骤结构；系统自动决策(默认阈值/自动筛选)，用户**看/点/选**(阈值框、特征勾选、算法下拉)覆盖。
    - **agent 模式**：LLM 提议+解释、**自由文本调整**、可重规划增删步骤。
    - 共用：模板 + 执行器 + 确认门 + 内联富表 + **"重跑本步换参数"原语**；只差"谁提议(自动 vs LLM)""怎么改(控件 vs 自然语言)""能否改结构"。
11. **可组合阶段模块**：三个阶段 `JOIN / FEATURE / MODELING`；模板=模块堆叠（拼接=[JOIN]、特征分析=[JOIN,FEATURE]、模型开发=[JOIN,FEATURE,MODELING]）；**输入已满足则跳过该阶段**（给拼好样本→JOIN 跳过并确认一次；甩零散 csv→JOIN 启动）。一处改进惠及上层。
12. **构建主轴（2026-06-25 修正）**：原定"先全手动再叠 agent"已与现实相反——建模原型 `marvis/agent/modeling_agent.py` 已是 **agent 模式**、三新类型当前 `manualEnabled=false`；且评审确认 `task_type` 路由是硬前置（详见 §八）。故修正为 **agent 先行**：先补 `task_type` 路由骨架 → 把建模原型泛化为通用 agent 驱动 → JOIN/FEATURE 挂同一驱动（agent 模式）→ 手动模式作为后续**控件皮肤**叠加。两序皆可（用户确认），取 agent 先行以匹配现状、最小返工；模块仍按 JOIN→FEATURE→MODELING 自底向上组合，最终交付"甩 csv 出模型"一条龙。

---

## 二、共享底座（两模式都用）

```
模板(可组合 phase 模块)  ─►  PlanExecutor 逐步执行 pack 工具
        │                          │
        │                    每步产出落 plan_step_outputs
        │                          │
   phase 分组(展示)          needs_confirmation 门 → AWAITING_CONFIRM 暂停
        │                          │
   右栏流程视图            通用计划对话驱动：产出→内联富表 + 门交互
                                    │
                       确认→恢复 / 调整→重跑本步 or replan
```

复用的现成件（来自调研，均已存在、能用）：
- `marvis/orchestrator/`：IntentRouter / Planner(from_template + generate) / PlanValidator / **PlanExecutor**（遇 `needs_confirmation` 自动停、`confirm_step` 恢复、failure/decision_point 可 replan）。
- `marvis/orchestrator/templates/`：`StepTemplate`(含 `needs_confirmation`/`decision_point`) + `WorkflowTemplate`；已有 `STANDARD_MODELING`。
- `marvis/db.py PlanRepository`：plan/steps/outputs/sub_agents/loop_events 全持久化。
- `marvis/routers/plans.py`：create/get/confirm/run/cancel + 单步 confirm。
- 前端 `marvis/static/js/v2/`：`plan_view.js`(步骤行/状态/进度条/确认按钮) + `loop_progress.js` + `state_v2.js`。
- `marvis/metric_tables.py` + `app.js renderMetricTableSection 家族`：内联富表渲染（数据条/热力/PSI 条/KPI 卡/ROC-KS 曲线）。
- pack 工具：`feature`(compute_feature_metrics/correlation_analysis/bin_feature/woe_encode/cross_features…)、`data_ops`(infer_schema/align_columns/propose_join/execute_join/dedup_rows…)、`modeling`(screen_features/tune_hyperparameters/train_model 多 recipe…)。

新建的件（全计划仅这些是"新"）：
- **3 个 phase 模块 + 3 个组合模板**（JOIN/FEATURE/MODELING）。
- `StepTemplate/PlanStep` 加**展示用 `phase` 字段** + **skip 预判**。
- **通用计划对话驱动**（agent 模式）+ **手动模式结构化门控件**。
- "工具产出 → metric_tables 段schema"**转换登记表**。
- "调整指令"端点 + `Planner.replan` 接受用户约束。
- 前端：把 `plan_view`/内联富表挂到**对话工作区**（非设置壳 `#v2RuntimeMount`）；三区布局；按 `task_type` 与 `run_mode` 切换（验证保持旧 UI 不动）。

---

## 三、三个阶段模块（自底向上构建）

### 阶段 1：JOIN（数据拼接）—— 最先做、含最高风险门
- 步骤骨架：`infer_schema` → `align_columns` → **`propose_join`（decision_point + needs_confirmation 强制确认）** → `execute_join` → `dedup_rows`。
- 确认门内联展示：命中率 / 行数膨胀 / 键唯一性 / 列值指纹(raw vs md5)；**join 执行前必须强制确认**（不可违反的 join 安全不变量：样本锚定、左连接、键字典、强制确认）。
- skip 预判：单一拼好样本→整阶段跳过(确认一次)；多文件→启动。
- 手动模式门控件：键选择下拉 / 转换方式 / 确认执行按钮。

### 阶段 2：FEATURE（特征分析+筛选）—— 复用 [JOIN]
- 步骤骨架：(读样本/`infer_schema`) → `compute_feature_metrics`(批量 IV/KS/AUC/PSI/coverage，**小步骤连续跑不停**) → `correlation_analysis`(共线/VIF) → **筛选确认门**〔可选 `bin_feature`/`woe_encode`/`cross_features`〕。
- 确认门一次性摊：全特征效果表 + agent 选了/弃了哪些(及原因)。
- 调整(走"重跑本步"原语)：IV 阈值 / PSI 阈值 / 去掉某些 / 强制加回某些。
- 手动模式门控件：阈值输入框 + 特征去留勾选。

### 阶段 3：MODELING（建模）—— 复用 [JOIN, FEATURE]
- 收编本会话已建：`screen_features` / `tune_hyperparameters` / `train_model` + `detect_setup`(→填槽)。
- 步骤骨架：(筛后特征) → **算法/任务类型 SELECT 门** → `tune_hyperparameters` → `train_model` → 指标确认门 → 〔compare_experiments / export / handoff〕。
- 任意模型(已确认范围)：多算法 xgb/lr/scorecard、回归、多分类、DNN(sklearn MLP 先行)。多分类/DNN 需新指标与配方，排该阶段后段。
- 确认门内联：train/test/OOT KS/AUC/PSI、lift/gains、ROC-KS、调参 trials 排行、特征重要性、对比基准。

> 每个阶段都**先手动模式**(结构化控件，无 LLM)，三阶段手动跑通 = **离线"甩 csv 出模型"一条龙**。

---

## 四、agent 模式层（手动底座之上统一叠）

通用计划对话驱动（一个，服务全部任务类型）：
1. 第一条对话：填槽(`detect_setup`) + 亮计划(右栏) + 计划级确认/调整。
2. 跑计划；每到确认门：步骤产出→内联富表 + LLM 组织门话术(无 LLM 时 canned 兜底)。
3. 解析回复：确认→`confirm_step` 恢复；自由文本指令→LLM 决定"重跑本步换参数"或"`replan` 带约束"。
4. 顶栏只反映整体状态/报错。

agent 模式 = 手动模式同一底座 + 这一层 LLM 编排/解析/重规划。任务差异零新驱动。

---

## 五、范围与不动项
- **做**：数据拼接 / 特征分析 / 模型开发（各自手动→后续 agent）。
- **不改**：模型验证（保留其现有 手动/agent 与写死步骤、旧 `#progressPanel`；按 task_type 与新 UI 共存）。
- **不开发**：策略 / vintage（欢迎页入口保留，暂不接 V2）。
- 6 入口直达对应任务为目标；本轮落地其中 3 个。

### 周边模块定位（2026-06-25 澄清）
- **记忆**：现仅验证管道自动沉淀（`pipeline.py`，受 auto_distill 门控）+ 聊天偏好捕获。**新 3 流程本轮不接**（后续增强：接则记各门复发决定/偏好做默认建议，须门控，注意"两条写记忆路径"坑）。
- **插件运行时**（`ToolRunner`+packs）：**命脉**——计划里每个工具都是 pack 插件，必需。**插件管理 UI**：扩展自定义工具用，次要，本轮不重点。
- **Workflow 模板**：**核心**——新写 JOIN/FEATURE/MODELING 三模板，任务=模板堆叠；模板管理区是它们的家。
- **能力档位**（第三轴，与 run_mode/acceptance_mode 正交）：**只管自治程度**（`max_replan_iterations`/explore），**不影响效果/确定性/门/安全**；补选择器入口（设置可选 + 创建弹窗 per-task，默认取设置），见 [前端 spec §5.1](v2-frontend-layout-spec.md)。

---

## 六、主要风险
- **双状态所有者**：手动/agent 都不得与 PlanExecutor 争写步骤状态——执行器是唯一写方，驱动只读+发指令。
- **join 强制门**：必须由模板确定性钉死，不能交 LLM 自由裁量。
- **phase skip 语义**：跳过要显式可见+确认一次，避免"静默跳过"误导。
- **新旧 UI 共存**：验证旧面板 vs 新 V2 计划面板，按 task_type 切换，别双进度。
- **多分类/DNN 耦合**：触碰最耦合的指标/打分/报告表，排建模阶段后段、严格门控不破坏二分类序列化。
- **纯 LLM 兜底脆弱**：核心流程不赌纯生成；模板把工具/$ref 接线钉对。

---

## 七、分阶段 spec（均已出,逐份已审）
1. ✅ [JOIN 阶段 spec](v2-join-phase-spec.md)（强制确认门数据契约 + 动态键识别 + 两级去重 + 手动控件 + skip）。
2. ✅ [FEATURE 阶段 spec](v2-feature-phase-spec.md)（指标全勾选 + 宽表筛选门 + 独立=出报告/被调用=出选中集 + lift按风险定向）。
3. ✅ [MODELING 阶段 spec](v2-modeling-phase-spec.md)（5 门:切分/算法类型/调参/选模报告/动作 + 任意模型 + 固定格式报告）。
4. ✅ [通用计划对话驱动 spec](v2-plan-driver-spec.md)（checkpoint 门 + 主循环 + 路由 + 登记表 + append + 两受控度）。
5. ✅ [三区前端布局 spec](v2-frontend-layout-spec.md)（顶状态/中对话富表/右 plan_view+下载 + 验证共存）。

## 八、构建顺序（2026-06-25 修正：agent 先行 + 评审硬前置）
> 经外部评审核实后调整（C3 为 confirmed blocker，C4 为 confirmed should-fix）。

- **步骤0 路由骨架（C3 blocker，最先做）**：`api.py` import `TASK_TYPE_FEATURE_ANALYSIS / TASK_TYPE_DATA_JOIN`，在 `/agent/start` + `/agent/messages` 按 `task_type` **分发表**路由；尚未接通的类型**显式报错**（绝不静默落验证 agent——现状 `data_join/feature_analysis` 会静默走验证）。dispatch 表防止未来新类型因遗漏默认落验证。
- **步骤1 后端契约（C4）**：`StepTemplate/PlanStep` 加展示用 `phase` 字段并串 `_step_to_dict/_from_dict` + 模板→步骤构建；`GET /tasks/{task_id}/plans` + `PlanRepository.list_plans_for_task`（**恢复/重开任务用**；create 已返回 plan_id，首建不依赖此接口）。排在前端布局工作之前。
- **步骤2 通用 agent 驱动**：把 `modeling_agent.py` 原型泛化为通用计划对话驱动（agent 模式），先服务 MODELING（其阶段机已跑通）。
- **步骤3 JOIN → 步骤4 FEATURE**：各自 phase 模块/模板 + pack 补缺，挂同一驱动；MODELING 由 [JOIN,FEATURE,MODELING] 组合成型。
- **步骤5 手动模式**：作为控件皮肤（无 LLM、纯控件 +「继续/暂停」）叠加到三模块。
- 每模块:模板/步骤 + pack 补缺 + 门（agent 话术 / 手动控件）→ 单测/离线验证 → 下一模块。
