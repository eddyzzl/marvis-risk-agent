# MARVIS 路线图

本文档是 MARVIS 产品阶段、运行时术语和后续能力边界的统一来源。历史实施文档可以继续使用 P1/P2/P3 叫法；发布、演示、开源和对外说明统一使用 V1/V2/V3/V4。

## 当前状态

- 当前公开稳定线：**V1.1.x**。
- 当前已实现产品面：本地模型验证工作流、手动模式、Agent P1 解释和报告草稿、Notebook 执行、确定性验证证据、Excel/Word 产出、Agent Memory Foundation、运行时 branding、发布工具。
- 下一条计划功能线：**V2 可用的 Agent 工作台**（Plugin/Tool Runtime 为其技术底座；见下「V2」一节）。

## 术语

- **Plugin（插件）**：可安装或内置的能力包。包含 manifest、代码、权限、版本、测试、展示声明、tools 和 hooks。
- **Tool（工具）**：Plugin 内 Agent 可以主动调用的具体动作，有输入 schema、输出 schema、权限、失败策略和审计记录。
- **Hook（钩子）**：Plugin 内由平台事件自动触发的动作，例如任务扫描完成、验证完成、报告生成前后。
- **Workflow（流程）**：Agent 生成、平台内置、或用户可编写的一组 Tool/Hook 编排计划（模板），用来完成端到端任务。
- **Skill（技能）**：SOP / Playbook / 方法论型知识，**落地为「用户可编写的 Workflow 模板」**（Phase 2 Part C2）——声明式、只编排已信任工具、过 `PlanValidator` 校验，不直接执行 Python 代码、不另立 runtime。历史文档里的 skill runtime 是旧称；MARVIS 执行能力统一是 Plugin / Tool / Hook / Workflow。

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

V1.1.x 是当前稳定线，给模型验证工作流加入长期、可审计的 Agent 记忆。它仍属于 V1 产品线，因为它增强的是当前验证流程，不引入新的插件运行时架构。

参考原则：

- 参考 OpenClaw 的本地优先、可查看文件、短期工作层到长期紧凑层的蒸馏、action-sensitive memory 和人工可复核的记忆管理。
- 参考 Hermes 的用户画像 / Agent 经验分层、紧凑上下文注入、会话搜索、外部 provider 预留和写入前安全扫描。
- MARVIS 不照搬通用 agent 记忆。V1.1 必须适配信贷风控验证：确定性指标隔离、敏感材料禁存、来源 task_id 可审计、历史模型效果可比性置信度。

目标：

- 让 Agent 能跨任务记住验证相关经验，而不是只记住当前对话。
- 记忆只辅助解释、参数建议、风险提醒、历史对比、报告口径和后续 workflow 编排，不直接改变确定性验证结果。
- 所有记忆必须可查看、可禁用、可删除、可审计。
- 记忆应自然体现在 Agent 对话、阶段分析、报告草稿建议和未来 Tool/Workflow 选择中，不新增常驻前端灰块展示匹配记忆。
- 验证完成、验证失败、报告确认、用户纠正和字段识别时，系统可以提取候选记忆；候选记忆必须经过分类、安全过滤、压缩和来源记录。

允许保存：

- **用户偏好**：报告措辞、解释详细程度、常用输出风格、用户明确纠正过的表达禁忌。
- **字段口径**：常见字段别名、渠道字段、时间字段、样本分组字段、目标字段和分数字段习惯。
- **验证坑点**：某类 Notebook、PMML、字段、执行环境、数据字典或报告问题的摘要和修复建议。
- **任务经验**：历史任务的非敏感摘要、失败原因、复核提醒、报告确认口径和人工复核结论摘要。
- **模型经验**：KS、AUC、PSI、月份、渠道、模型名称、模型版本、适用范围、来源 task_id、重要特征的数据源；可对比多个模型、多个版本、多个月份、多个渠道和多个指标，并可总结 A 卡、B 卡、额度、利率、前筛、C 卡等场景经验。
- **skill 经验预留**：未来 V2 skill（= 用户可编写 Workflow 模板）/ SOP / playbook 的执行经验摘要。V1.1 只预留 schema，不落地 V2 能力。

禁止保存：

- 原始样本数据、客户明细、完整 Notebook 源码。
- PMML 文件内容、模型文件内容、API key、数据库连接。
- 未脱敏报告全文、机构敏感信息、私有 branding 内容。
- 会直接改变 KS/PSI/AUC/分数一致性等确定性指标的内容。
- 无来源、无置信度或无法审计的自动推断结论。

模型经验第一版字段：

- `ks`
- `auc`
- `psi`
- `month`
- `channel`
- `model_name`
- `model_version`
- `scope`
- `source_task_id`
- `important_feature_sources`

历史匹配优先级：

```text
模型名称关键词 > 适用范围 > 模型场景 > 渠道 > 月份 > 模型版本
```

对比置信度：

- 高置信：Agent 可以明确说明当前模型相比历史基线效果提升或下降。
- 中置信：Agent 只能说明“可能可比”，并提示需要人工确认。
- 低置信：不用于历史对比。

记忆生命周期：

- **候选提取**：从结构化验证结果、Agent 消息、任务失败原因、用户明确偏好和报告确认中生成候选记忆。
- **安全过滤**：拒绝敏感内容、过长内容、源码/数据/密钥、提示注入和无来源结论。
- **压缩保存**：保存短摘要、结构化字段、来源、置信度、创建/更新时间、禁用状态和审计事件。
- **检索使用**：Agent 在阶段分析或聊天前按任务上下文检索相关记忆，生成 bounded memory context，不把全部记忆塞进提示词。
- **审计管理**：用户可以查看、禁用、删除记忆；系统记录读、写、禁用、删除和用于回复的引用。

前端体验：

- 不新增任务顶部固定记忆区域，不用灰块列出“匹配到的记忆”。
- 对话中自然体现记忆价值，例如：“上一版分润 A 卡模型在 2026 年 2 月样本上的 KS 高于当前模型，需要关注。”
- Agent 消息可带可展开的“记忆引用”，展示来源 task_id、类别、置信度和用途。
- 记忆管理入口放在设置或审计管理视图中，用于查看、禁用、删除和导出审计，不作为每个任务的常驻内容。

非目标：

- 不保存原始样本、客户明细、完整 Notebook、PMML 内容、模型文件、API key、数据库连接或敏感报告全文。
- 不让记忆改变 KS/AUC/PSI/分数一致性等确定性计算结果。
- 不在 V1.1 实现训练建模、策略仿真或插件执行。

## V2：可用的 Agent 工作台

V2 不只是通用 Agent 运行时底座。V2 的完成标准是：欢迎页露出的每个入口都必须是真实可用的端到端功能，而不是只有入口、demo plugin 或半成品 runtime。模型验证沿用并保持 V1.1 稳定能力；数据拼接、特征分析、模型开发、策略开发和 Vintage 分析都要能从创建任务进入对应流程，完成必要的人在环确认、工具执行、结构化结果展示、下载/报告或可审计产物。

运行时仍是 V2 的技术底座：Agent 理解用户目标，选择可用 Plugin/Tool，生成 Workflow，执行受控 Python 能力，展示结构化结果，生成报告内容，并留下审计记录。但这些底座能力必须服务于真实入口的可用闭环。

典型用户目标：

- “我要做一个模型验证。”
- “我要做一个 LGB A 卡模型。”
- “做一个额度策略。”
- “分析一下最近的 vintage。”

V2 第一版范围：

- Plugin manifest/schema。
- Plugin 上传、安装、启用、禁用、版本管理和审计元数据。
- Tool 输入/输出 schema。
- Python Tool runner：参数校验、执行、结果校验、超时和失败策略。
- Hook 声明和平台事件触发。
- Workflow 计划和执行状态。
- Agent planner：基于用户意图、任务上下文、V1.1 记忆、Plugin manifest 和可用 Tool 生成计划。
- 结构化输出进入前端预览、Excel/Word 报告区、记忆或审计证据。
- 欢迎页入口的真实功能闭环：
  - **模型验证**：保持 V1.1 既有手动/Agent 验证能力可用。
  - **数据拼接**：识别主表/特征表/键，诊断命中率、膨胀、去重和键格式风险，确认后执行 join 并产出拼接数据。
  - **特征分析**：计算所选 IV/KS/AUC/PSI/coverage/lift/共线等指标，输出可下载特征分析报告；被建模或策略调用时可进入筛选确认门。
  - **模型开发**：读样本、确认目标和切分、做泄漏感知筛选、调参训练、比较实验并输出模型开发报告和可交接产物。
  - **策略开发**：构造规则、回测策略，计算通过率、坏账率、swap、利润或收益权衡，关键上线类动作保留人工确认。
  - **Vintage 分析**：计算 vintage / roll-rate / 稳定性观察，输出可复核图表、表格和报告材料。
- 不可用或未接通的入口不得作为“可用”入口展示；如果保留占位，必须明确标记为未开放或实验中。

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
```

V2 是验证扩展、建模工具、策略工具、vintage 分析工具的共同底座，也是这些入口对用户可用的第一条完整产品线。V2 第一版可以控制深度，但不能只交付 runtime；每个已展示入口都必须至少有一条可演示、可恢复、可审计的真实闭环。

## V3：Model Development 深化

V3 基于 V2 已可用的模型开发入口继续深化，而不是才开始交付第一版建模能力。核心工作是把 V2 的建模闭环扩展成更完整的训练平台：更丰富的建模 Plugin、训练 Workflow、实验记录、模型产物交接和前端模型展示。

计划范围：

- 数据质量和建模准备度工具。
- 特征加工和特征筛选工具。
- 常见信贷模型训练工具，例如 LGB、XGB、LR、评分卡等 recipe。
- 实验记录、指标、特征清单、模型产物和审计引用。
- 模型展示 UI，用于比较候选模型并把模型交给验证流程。

训练产物必须进入验证流程后才可视为可复核产物。训练工具应使用独立训练上下文和结果契约，不要复用当前验证契约。

## V4：Strategy And Portfolio Pack

V4 把 MARVIS 从建模和验证扩展到策略与组合管理。

计划范围：

- 额度、定价、准入、拒绝、分群策略工具。
- Vintage 和 portfolio 监控分析。
- 风险、收益、通过率、坏账率等权衡视图。
- 策略报告段落和复核证据。

V4 仍应使用 V2 Plugin/Tool/Workflow 底座，并为关键流程保留手动模式。

## 文档模型

- `README.md` / `README.zh-CN.md`：公开入口。
- `docs/roadmap.md`：产品阶段、术语和能力边界。
- `docs/versioning.md`：版本号、tag、发布、forward-port 规则。
- `DESIGN.md`：产品体验、信息架构、视觉和交互约束。
- `docs/notebook_contract.md`：Notebook 运行契约。
- `docs/对notebook的要求.md`：给模型开发人员看的 Notebook 提交要求。
