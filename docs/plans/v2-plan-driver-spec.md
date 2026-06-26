# 通用计划对话驱动 Spec（agent 层）

> V2 完善总计划第 4 份 spec（见 [v2-completion-plan.md](v2-completion-plan.md)）。
> **一个驱动服务所有任务类型**(决策 #9);任务差异全在 模板 + tool→表格转换登记表,不写多套驱动。
> 本会话临建的 `marvis/agent/modeling_agent.py` = 原型,**泛化为本驱动后收编/退役**。

---

## 0. 地面真值（已核对）
- `PlanExecutor.run`:命中未确认的 `needs_confirmation` 步骤 → 置 `AWAITING_CONFIRM` 并**返回**(后台 job 结束,执行器是 pre-step 暂停)。`confirm_step` 标确认后**重新 run** 即续跑。
- `failure/decision_point` → `Planner.replan`(已有)。
- `HookDispatcher`:执行器发 `step.completed/workflow.completed/plan.replanned` 等事件。
- 验证 agent 已有 **acceptance_mode**(`AGENT_ACCEPTANCE_AUTO/NORMAL` + `_agent_auto_accept`)——**直接复用**。
- 富表渲染 `metric_tables.py→renderMetricTableSection`、消息持久化 `agent_messages`(metadata_json)均已有。

---

## 1. Checkpoint 门模式（决策,不改引擎)
- 每个大步骤:**分析工具步**(screen/compute_metrics/propose_join/train…)正常跑 → 紧跟**轻量 checkpoint 步**(`needs_confirmation=True`,依赖分析步)。
- 执行器跑完分析步、**停在 checkpoint 前** → 驱动取**前一步(分析)产出**摊给用户。
- 确认 → checkpoint 放行(转交结果)→ 续;调整 → 重跑**分析步**换参(checkpoint 留 pending)→ 刷新再停同门。
- **零引擎改动**,且"你确认的就是刚跑出的结果"。

## 2. 驱动主循环
```
开任务/首条消息
  → 填槽(detect_setup:样本/目标/所选指标/算法类型…缺则问)
  → 从模板实例化计划(Planner.from_template)+ 右栏亮计划
  → **计划级总览门(两种模式都先亮一眼整体编排)**:默认权限+自动审查都先让用户看计划、
     用户确认"开始"后再跑(自动审查 = 看一眼+确认开始→之后各步全自动;默认权限 = 之后每大步还停)
循环:
  PlanExecutor.run  →  返回 {AWAITING_CONFIRM | DONE | FAILED}
   ├ AWAITING_CONFIRM: 取前一分析步产出 → §4 tool→表格 → 组织话术(LLM/canned)
   │                    → **append** 一条 assistant 消息(带内联富表)→ 等用户
   ├ DONE: append 完成消息 + 报告/下载
   └ FAILED: append 报错(顶栏也反映整体状态)
用户回复(§3 解析)→ 续跑 / 重跑 / replan → 各自 **append** 新轮
```

## 3. 回复解析与路由（决策 #5）
- **确认** → `confirm_step` → 重新 `run`。
- **提指令**:
  - 改本步参数(换阈值/算法/调参范围/去留特征…)→ **新「调整」端点**:LLM 算新工具参数 → **重跑该分析步**(append 新产出)。
  - 结构改动(加/删/换步骤、加对比实验)→ `Planner.replan` **带用户指令为约束**。
  - 路由由 LLM 决定(决策 #1:LLM 编排)。
  - **replan 次数受能力档位约束**:`tier.max_replan_iterations`(稳健 2/均衡 4/自治 8)。档位只管自治程度(重规划次数 + 能否探索),**不碰确认门/证据/安全护栏/确定性**——与本驱动的门和路由正交。档位 per-task 选(默认取设置),见前端 spec §5.1。
- 手动模式:控件(滑杆/勾选/下拉/按钮)直接映射到 确认 / 调整端点,无需 LLM。

## 4. tool→表格转换登记表
- 一张登记表:`{tool_name → transform(output)→ metric_tables 段schema}`。
- 例:`screen_features→特征宽表`、`compute_feature_metrics→IV/KS/AUC表`、`tune→trials排行`、`train→指标网格+ROC-KS`、`propose_join→join诊断表`。
- 驱动据 awaiting 门的前置工具查表渲染;**任务差异全落这里**,驱动本体不分任务。
- 缺转换 → 退化为 markdown 表/文字(兜底)。

## 5. 追加不覆盖（历史可回看）
- 每次 分析/重跑/调整 → **append 新 assistant 消息**(新表 + 新话术),**旧消息全保留**,用户可往回翻。
- **不**更新/覆盖旧占位消息(纠正原型 modeling_agent 的"更新流式占位"做法)。
- 每条消息 metadata 记 `{plan_id, step_id, run_seq}` 便于定位"这是第几次跑哪步"。

## 6. 两种受控度（复用 acceptance_mode）
| 模式 | 行为 | checkpoint |
|---|---|---|
| **默认权限**(NORMAL) | 每个大步骤后停,**用户确认才续** | 命中即停、等确认 |
| **自动审查**(AUTO,按任务改名 自动拼接/分析/建模) | agent **全程自动**,各节点按 agent 选择执行 | 自动确认(`confirm_step`),不停 |
- 对话框顶部一个选择器(同验证 agent 现状)。AUTO 下 agent 在每个本该停的门**用其推荐值自动确认**并 append 一条"已自动决定 X(理由)"留痕。
- 与"手动 vs agent"正交:手动模式=控件驱动(可"用默认一路跑"=类 AUTO,但无 LLM 决策);agent 模式才有 AAUTO/NORMAL 这层。

## 7. 状态与持久化
- **计划状态唯一真源 = PlanRepository**(决策:执行器唯一写步骤状态);驱动**只读计划 + 步骤产出 + 会话**,不另存状态机(纠正原型把状态塞 message metadata 的做法)。
- 会话 = append-only 日志;槽位/选择存计划或任务记录。

## 8. 新建 vs 复用
- **复用**:PlanExecutor 暂停/恢复、confirm_step、Planner.replan、acceptance_mode、HookDispatcher、metric_tables 渲染、agent_messages 持久化、detect_setup(填槽)。
- **新建**:通用驱动主循环(包 run + 门消息组织 + append)、**「调整」端点**(带指令→重跑步换参)、replan 扩接受用户约束、**tool→表格登记表**、checkpoint 步在各模板的编排、AUTO 下"自动确认 + 留痕"。
- **退役**:`modeling_agent.py` 的任务专属逻辑并入通用驱动(其 detect_setup/screen/tune/session 编排迁为 MODELING 模板的步骤)。

## 9. 已锁小项
1. **AUTO 留痕**(已定):自动审查下**每个**自动确认的门都 append 一条"自动决定 X + 理由",便于回溯。
2. **计划级总览**(已定):**两种模式都先亮一眼整体计划编排**,用户确认"开始"后再跑;之后自动审查全自动、默认权限每大步停。
