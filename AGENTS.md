# AGENTS.md

本项目是 MARVIS Risk Agent：本地优先的信贷风控 Agent 平台。当前稳定产品线是 V1 模型验证；长期方向是用可治理的 Agent workflows 覆盖验证、建模、分析、策略和监控。

## 优先阅读

- `docs/roadmap.md`：产品阶段、术语和能力边界。
- `docs/versioning.md`：版本命名、发布 helper、tag 和 forward-port 规则。
- `DESIGN.md`：产品体验、视觉语言和 UI/UX 约束。
- `docs/notebook_contract.md`：当前 Notebook 运行契约。
- `docs/对notebook的要求.md`：给模型开发人员看的 Notebook 提交要求。

## 当前版本边界

- **V1.0.x**：当前稳定模型验证工作流。
- **V1.1**：计划中的 Agent Memory Foundation，用于历史验证指标对比。
- **V2**：计划中的 Agent Plugin/Tool Runtime。
- **V3**：计划中的 Model Development Pack。
- **V4**：计划中的 Strategy and Portfolio Pack。

除非任务明确要求改变，当前 V1 行为必须保持稳定：

- Notebook 契约使用 `RMC_SAMPLE_DF`、`RMC_TARGET_COL`、`RMC_ALGORITHM`、`RMC_SCORE_FN`。
- 主分数一致性比较是 Notebook 内存模型分 vs 提交 PMML 分。
- 手动模式和 Agent P1 模式都必须可用。
- 确定性指标由平台代码计算，不由 LLM 计算。
- Agent 可以解释、总结、起草、规划和请求确认，但不能编造指标或绕过平台证据。

## 运行时术语

新文档和新代码讨论统一使用：

- **Plugin**：可安装或内置的能力包。
- **Tool**：Plugin 内可调用的具体动作。
- **Hook**：Plugin 内由平台事件触发的动作。
- **Workflow**：Agent 生成或内置的 Tool/Hook 工作序列。
- **Skill**：预留给未来 SOP / Playbook / 方法论型知识；历史文档可能把它作为 plugin/tool runtime 的旧称。

不要在 MARVIS 执行代码的新文档里继续使用 “skill runtime” 作为主术语，除非任务明确讨论 SOP/playbook 型 skill。

## 模块边界

```text
validation/     确定性验证算法；不拥有 DB、FastAPI、Agent 或任务生命周期
output/         从结构化 payload 渲染 Excel/Word/chart；不执行上传插件
pipeline.py     任务状态、文件系统、validation、output 的主编排层
api.py          HTTP、任务创建、active job、前端 payload；不承载验证算法
static/         根据 API payload 渲染前端状态；不要解析后端自由文本作为业务事实
agent/          意图、解释、总结、报告草稿和未来 planner
agent_memory/   V1.1 计划边界；存储和检索可审计记忆，不直接改变任务指标
plugins/        V2 计划边界；Plugin/Tool/Hook runtime
training/       V3 计划边界；训练工具和训练上下文
```

## 文档规则

- `docs/roadmap.md` 是唯一产品路线和术语来源。
- `docs/versioning.md` 只维护发布、tag、版本号和 forward-port 规则。
- `DESIGN.md` 只维护产品体验、信息架构、视觉和交互约束。
- `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 是未来具体功能设计和实施计划的输出位置；不要因为旧内部 specs 从公开 baseline 删除就废弃这个工作流。
- 不要在 `README.md`、`AGENTS.md`、`CLAUDE.md` 中复制完整路线；链接到 `docs/roadmap.md`。

## Python 环境

在本机 workspace 中运行 Python 命令时，使用开发者 conda 环境：

```bash
conda run -n py_313 python ...
conda run -n py_313 python -m pytest ...
conda run -n py_313 python -m ruff check ...
```

只有写给公开用户的 README/runbook 示例才使用普通 `python`。

## 验证

代码改动先跑最小相关测试，再根据影响面扩大验证。

常用验证命令：

```bash
conda run -n py_313 python -m pytest -q
conda run -n py_313 python -m ruff check riskmodel_checker tests --extend-exclude '*.ipynb'
node --check riskmodel_checker/static/app.js
git diff --check
```

文档-only 改动至少运行：

```bash
git diff --check
```

## 发布和提交说明

公开发布版本和 tag 使用 `scripts/release_push.py`，不要手动移动已发布 tag。

Commit message 应说明为什么改。承载产品或工作流决策的 commit，优先使用简洁 decision trailers：

```text
Constraint: ...
Rejected: ... | ...
Confidence: high
Scope-risk: narrow
Tested: ...
Not-tested: ...
```
