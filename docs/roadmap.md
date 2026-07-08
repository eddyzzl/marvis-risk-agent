# MARVIS 路线图

本文档是 MARVIS 产品阶段、运行时术语和能力边界的统一来源。历史实施文档可以继续使用 P1/P2/P3、Phase 1/2/3 或批次编号；发布、演示、开源和对外说明统一使用 V1/V2/V3/V4。

## 当前状态

- 当前主线：**MARVIS-Agent V2.x**，本地优先、可治理的多工作流信贷风控 Agent 平台。
- 当前产品面：数据处理、特征分析、模型开发、模型验证、策略开发、Vintage/风险分析、监控、组合分析、额度/定价和即席问数等信贷风控 workflow。
- 模型验证是稳定兼容工作流之一：继续保留 V1.1 手动模式和 Agent 辅助验证能力，Notebook 契约、PMML 对比、确定性验证指标和 Excel/Word 产出必须保持兼容。
- V2 不是只有 Plugin/Tool runtime 外壳；欢迎页展示的入口必须对应真实可用的端到端 workflow：人在环确认、受控工具执行、结构化结果、下载/报告或可审计产物。
- Portfolio / 组合分析能力已有后端工具、模板和测试覆盖；具体是否作为首屏入口或 Agent start allowlist 暴露，以当前代码和 UI 为准。

## 术语

- **Plugin（插件）**：可安装或内置的能力包。包含 manifest、代码、权限、版本、测试、展示声明、tools 和 hooks。
- **Tool（工具）**：Plugin 内 Agent 可以主动调用的具体动作，有输入 schema、输出 schema、权限、失败策略和审计记录。
- **Hook（钩子）**：Plugin 内由平台事件自动触发的动作，例如任务扫描完成、验证完成、报告生成前后。
- **Workflow（流程）**：Agent 生成、平台内置、或用户可编写的一组 Tool/Hook 编排计划（模板），用来完成端到端任务。
- **Skill（技能）**：SOP / Playbook / 方法论型知识，落地为「用户可编写的 Workflow 模板」——声明式、只编排已信任工具、过 `PlanValidator` 校验，不直接执行 Python 代码、不另立 runtime。历史文档里的 skill runtime 是旧称；MARVIS 执行能力统一是 Plugin / Tool / Hook / Workflow。

示例：

```text
Plugin: credit_modeling
  Tool: check_data_quality
  Tool: train_lgb_model
  Tool: export_pmml
  Hook: validation.completed -> summarize_historical_comparison

Workflow: "做一个 LGB A 卡模型"
  check_data_quality -> train_lgb_model -> evaluate_model -> export_pmml -> validate_model
```

## V1.0.x：上一条稳定模型验证线

V1.0.x 是上一条稳定线，目标是让 MARVIS 第一个完整工作流可靠、可演示、可回滚。

已实现范围：

- 创建模型验证任务。
- 扫描提交的 Notebook、样本、PMML、数据字典等材料。
- 执行 Notebook，并保留 live kernel 供下游验证复用。
- 对比 Notebook 内存模型分和提交 PMML 分。
- 计算 KS、AUC、PSI、分箱、稳定性、压力测试等验证证据。
- 生成 Excel 和 Word 报告。
- 支持手动模式和 Agent P1 模式。
- 支持从本地 workspace 配置运行时 branding。

V1.0.x 只接受必要缺陷修复、兼容性修复、文档修正和发布流程修正。

## V1.1：Agent Memory Foundation

V1.1 给模型验证工作流加入长期、可审计的 Agent 记忆。它现在作为 V2 平台里的兼容基础能力继续保留，而不是当前产品中心。

参考原则：

- 参考 OpenClaw 的本地优先、可查看文件、短期工作层到长期紧凑层的蒸馏、action-sensitive memory 和人工可复核的记忆管理。
- 参考 Hermes 的用户画像 / Agent 经验分层、紧凑上下文注入、会话搜索、外部 provider 预留和写入前安全扫描。
- MARVIS 不照搬通用 agent 记忆。记忆必须适配信贷风控：确定性指标隔离、敏感材料禁存、来源 task_id 可审计、历史模型效果可比性置信度。

目标：

- 让 Agent 能跨任务记住验证、建模、数据处理、策略和分析相关经验，而不是只记住当前对话。
- 记忆只辅助解释、参数建议、风险提醒、历史对比、报告口径和后续 workflow 编排，不直接改变确定性结果。
- 所有记忆必须可查看、可禁用、可删除、可审计。
- 记忆应自然体现在 Agent 对话、阶段分析、报告草稿建议和 workflow 选择中，不新增常驻前端灰块展示匹配记忆。
- 验证完成、建模完成、JOIN 执行、策略采纳、任务失败、用户纠正和字段识别时，系统可以提取候选记忆；候选记忆必须经过分类、安全过滤、压缩和来源记录。

允许保存：

- **用户偏好**：报告措辞、解释详细程度、常用输出风格、用户明确纠正过的表达禁忌。
- **字段口径**：常见字段别名、渠道字段、时间字段、样本分组字段、目标字段和分数字段习惯。
- **验证/建模/策略坑点**：某类 Notebook、PMML、字段、执行环境、数据字典、训练配置、策略口径或报告问题的摘要和修复建议。
- **任务经验**：历史任务的非敏感摘要、失败原因、复核提醒、报告确认口径和人工复核结论摘要。
- **模型经验**：KS、AUC、PSI、月份、渠道、模型名称、模型版本、适用范围、来源 task_id、重要特征的数据源；可对比多个模型、多个版本、多个月份、多个渠道和多个指标。
- **Workflow 经验**：用户编写或系统内置 Workflow 模板的非敏感执行经验摘要。

禁止保存：

- 原始样本数据、客户明细、完整 Notebook 源码。
- PMML 文件内容、模型文件内容、API key、数据库连接。
- 未脱敏报告全文、机构敏感信息、私有 branding 内容。
- 会直接改变 KS/PSI/AUC/分数一致性等确定性指标的内容。
- 无来源、无置信度或无法审计的自动推断结论。

记忆生命周期：

- **候选提取**：从结构化工具结果、Agent 消息、任务失败原因、用户明确偏好和报告确认中生成候选记忆。
- **安全过滤**：拒绝敏感内容、过长内容、源码/数据/密钥、提示注入和无来源结论。
- **压缩保存**：保存短摘要、结构化字段、来源、置信度、创建/更新时间、禁用状态和审计事件。
- **检索使用**：Agent 在阶段分析或聊天前按任务上下文检索相关记忆，生成 bounded memory context，不把全部记忆塞进提示词。
- **审计管理**：用户可以查看、禁用、删除记忆；系统记录读、写、禁用、删除和用于回复的引用。

前端体验：

- 不新增任务顶部固定记忆区域，不用灰块列出“匹配到的记忆”。
- 对话中自然体现记忆价值，例如：“上一版分润 A 卡模型在 2026 年 2 月样本上的 KS 高于当前模型，需要关注。”
- Agent 消息可带可展开的“记忆引用”，展示来源 task_id、类别、置信度和用途。
- 记忆管理入口放在设置或审计管理视图中，用于查看、禁用、删除和导出审计，不作为每个任务的常驻内容。

## V2：当前主线 Agent 平台

V2 是当前主线。它把信贷风控任务纳入统一 Plugin / Tool / Hook / Workflow 运行时，并用受控工具、计划校验、人在环确认、审计证据和结构化产物来约束 Agent 行为。

运行时职责：

- Agent 理解用户目标，补齐任务上下文，选择可用 Plugin/Tool，生成或实例化 Workflow。
- `PlanValidator` 校验工具存在性、输入 schema、DAG、post-check、确认门、指标范围和权限边界。
- `PlanExecutor` 执行步骤、维护状态、暂停确认、重试、replan、loop events、hooks 和 evidence envelope。
- `ToolRunner` 做权限校验、参数校验、确定性 seed、子进程隔离、超时、资源限制、输出 schema 和审计记录。
- AUTO 模式只能在 gate envelope 里选择有限动作，并受 schema、confidence、预算和安全规则约束。

当前产品入口：

- **数据处理 / Data Join**：识别主表/特征表/键，诊断命中率、膨胀、去重和键格式风险，确认后执行 join 并产出拼接数据。
- **特征分析**：计算 IV/KS/AUC/PSI/coverage/lift/共线等指标，输出可下载特征分析报告；被建模或策略调用时可进入筛选确认门。
- **模型开发**：读样本、确认目标和切分、做泄漏感知筛选、调参训练、比较实验并输出模型开发报告、打分产物和交接材料。
- **模型验证**：保持 V1.1 既有手动/Agent 验证能力可用，并可通过 `v1_compat` 作为 Workflow 里的稳定工具包调用。
- **策略开发**：构造规则、回测策略，计算通过率、坏账率、swap、利润或收益权衡，关键上线类动作保留人工确认。
- **Vintage / 风险分析**：计算 vintage、roll-rate、稳定性观察和相关分析，输出可复核图表、表格和报告材料。
- **监控与组合分析**：围绕评分、策略、组合表现、迁移矩阵、Expected Loss、限额/定价和 ad-hoc slice analytics 提供工具、模板和报告能力；首屏暴露范围以当前代码和产品选择为准。

不可用或未接通的入口不得作为“可用”入口展示；如果保留占位，必须明确标记为未开放或实验中。

最小平台 Hook：

```text
task.created
task.scanned
notebook.completed
validation.completed
report.before_generate
report.after_generate
memory.before_save
memory.after_save
workflow.completed
feature.computed
step.completed
plan.replanned
```

## V3：平台治理与扩展深化

V3 不再是“才开始交付建模能力”。V3 基于 V2 已有闭环，重点深化治理、扩展和更重的生产化能力。

候选范围：

- 第三方或机构自定义 Plugin/Workflow 的更完整安装、签名、权限和回滚治理。
- 多用户 / maker-checker / 审批留痕 / 权限分层。
- 调度器、监控任务、推送告警和批量运行。
- 实时评分 API、影子运行和决策引擎对接。
- 更完整的训练平台、实验追踪、模型注册、模型交接和复核流。
- 更强执行隔离，例如容器、系统沙箱或远程 worker。

## V4：Strategy And Portfolio Pack 深化

V4 把已存在的策略、监控和组合分析能力继续向经营闭环深化。

候选范围：

- 额度、定价、准入、拒绝、分群策略工具的更完整策略库。
- Vintage、portfolio、迁移矩阵和收益风险监控的周期化运行。
- 风险、收益、通过率、坏账率、利润、资本占用等权衡视图。
- 策略版本、上线建议、挑战者策略、监控计划和复盘报告的完整闭环。

V4 仍应使用 V2 Plugin/Tool/Workflow 底座，并为关键流程保留手动模式和审计证据。

## 文档模型

- `README.md` / `README.zh-CN.md`：公开入口。
- `docs/roadmap.md`：产品阶段、术语和能力边界。
- `docs/versioning.md`：版本号、tag、发布、forward-port 规则。
- `DESIGN.md`：产品体验、信息架构、视觉和交互约束。
- `docs/notebook_contract.md`：Notebook 运行契约。
- `docs/对notebook的要求.md`：给模型开发人员看的 Notebook 提交要求。
