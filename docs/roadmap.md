# MARVIS 路线图

本文档是 MARVIS 产品阶段、运行时术语和后续能力边界的统一来源。历史实施文档可以继续使用 P1/P2/P3 叫法；发布、演示、开源和对外说明统一使用 V1/V2/V3/V4。

## 当前状态

- 当前公开稳定线：**V1.0.x**。
- 当前已实现产品面：本地模型验证工作流、手动模式、Agent P1 解释和报告草稿、Notebook 执行、确定性验证证据、Excel/Word 产出、运行时 branding、发布工具。
- 下一条计划功能线：**V1.1 Agent Memory Foundation**。

## 术语

- **Plugin（插件）**：可安装或内置的能力包。包含 manifest、代码、权限、版本、测试、展示声明、tools 和 hooks。
- **Tool（工具）**：Plugin 内 Agent 可以主动调用的具体动作，有输入 schema、输出 schema、权限、失败策略和审计记录。
- **Hook（钩子）**：Plugin 内由平台事件自动触发的动作，例如任务扫描完成、验证完成、报告生成前后。
- **Workflow（流程）**：Agent 生成或平台内置的一组 Tool/Hook 编排计划，用来完成端到端任务。
- **Skill（技能）**：预留给未来 SOP / Playbook / 方法论型知识，不直接执行 Python 代码。历史文档里的 skill runtime 是旧称；新 MARVIS 执行能力统一使用 Plugin / Tool / Hook / Workflow。

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

## V1.0.x：稳定模型验证

V1.0.x 是当前稳定线，目标是让 MARVIS 第一个完整工作流可靠、可演示、可回滚。

已实现范围：

- 创建模型验证任务。
- 扫描提交的 Notebook、样本、PMML、数据字典等材料。
- 执行 Notebook，并保留 live kernel 供下游验证复用。
- 对比 Notebook 内存模型分和提交 PMML 分。
- 计算 KS、AUC、PSI、分箱、稳定性、压力测试等验证证据。
- 生成 Excel 和 Word 报告。
- 支持手动模式和 Agent P1 模式。
- 支持从本地 workspace 配置运行时 branding。

V1.0.x 只接受缺陷修复、兼容性修复、文档修正、发布流程修正，以及不改变主架构和产品边界的小体验改进。

## V1.1：Agent Memory Foundation

V1.1 给当前模型验证工作流加入长期、可审计的 Agent 记忆。它仍属于 V1 产品线，因为它增强的是当前验证流程，不引入新的插件运行时架构。

目标：

- 验证完成后自动把历史验证指标快照保存为 active memory。
- 新验证时，Agent 能把当前模型和历史可比模型版本、月份、渠道、适用范围进行对比。
- 支持一次对比多个历史模型、多个版本、多个月份、多个渠道和多个指标。
- 用户可以查看、禁用、删除记忆。
- Agent 回复或建议中必须记录使用了哪些记忆。

第一版记忆字段：

- KS
- AUC
- PSI
- 月份
- 渠道
- 模型名称
- 模型版本
- 适用范围
- 来源 task_id
- 重要特征的数据源

历史匹配优先级：

```text
模型名称 > 适用范围 > 渠道 > 月份 > 模型版本
```

对比置信度：

- 高置信：Agent 可以明确说明当前模型相比历史基线效果提升或下降。
- 中置信：Agent 只能说明“可能可比”，并提示需要人工确认。
- 低置信：不用于历史对比。

非目标：

- 不保存原始样本、客户明细、完整 Notebook、PMML 内容、模型文件、API key、数据库连接或敏感报告全文。
- 不让记忆改变 KS/AUC/PSI/分数一致性等确定性计算结果。
- 不在 V1.1 实现训练建模、策略仿真或插件执行。

## V2：Agent Plugin/Tool Runtime

V2 是通用 Agent 运行时底座。目标是让 Agent 理解用户目标，选择可用 Plugin/Tool，生成 Workflow，执行受控 Python 能力，展示结构化结果，生成报告内容，并留下审计记录。

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
- 少量内置和 demo Plugin，用于证明外部 Python 能力可以接入。

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

V2 是验证扩展、建模工具、策略工具、vintage 分析工具的共同底座。V2 第一版不需要一次性完成所有业务能力。

## V3：Model Development Pack

V3 基于 V2 运行时构建，不再从零做一个庞大的训练平台。核心工作是提供建模 Plugin、训练 Workflow、实验记录、模型产物交接和前端模型展示。

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
- `AGENTS.md`：Codex/Claude 共享的工程约定。
- `CLAUDE.md`：Claude 专用短入口，指向 `AGENTS.md`。
- `docs/superpowers/specs/`：未来具体功能设计时生成的实施级 specs。
- `docs/superpowers/plans/`：未来从已确认 spec 生成的实施计划。
