# MARVIS 项目、Agent 能力与架构健康综合审计

> 日期：2026-07-31
> 审计对象：`risk_manager` 当前 `main` 工作区
> 基线：`9845e907c702f65edc77a9d0121f0967f52488c2`（V2.1.19）
> 审计性质：原始审计为只读评估；用户随后授权整改。1.3 保留首轮整改快照，
> 2026-08-01 的完整结果见
> [综合审计整改记录](2026-08-01-comprehensive-audit-remediation.md)
> 结论置信度：产品与代码结构为高；三个月失败预测和产品建议为中高

## 一、结论先行

### 1.1 六个问题的直接答案

| 问题 | 直接答案 |
|---|---|
| 我最不自信的是什么？ | 不是代码能否在合成样本上运行，而是换成真实机构数据、真实业务口径和真实模型服务后，语义是否仍正确、结果是否能复现人工基线、Agent 是否仍能稳定地失败关闭。当前两个项目自定义的公开数据集参考门、真实材料对账和真实模型评测都没有形成完成证据。 |
| 你最大的盲点是什么？ | 把“实现了很多 Tool/Workflow、测试数量很多、合成 E2E 能跑”与“用户可达、真实材料正确、生产可运营、有人愿意签字”混成一个完成状态。当前最强的是工程实现，最弱的是外部地面真值和生产责任闭环。 |
| 三个月后最可能怎样失败？ | 不太可能是突然完全跑不起来，更可能是能力扩张速度继续超过证据收敛速度：演示越来越丰富、代码越来越重，但首个真实业务试点仍需要开发者大量解释、修数据和人工兜底，业务方最终不敢把业务责任交给平台。 |
| 最值得添加的行业领先功能？ | “可治理的信用决策数字孪生”：冻结任意时点的数据、特征、模型、策略和人工覆盖，逐笔重放 Champion/Challenger，做反事实与风险—收益—公平性联合约束，再通过 shadow、maker-checker、晋级、回滚和真实结果回流形成闭环。 |
| 有僵尸代码和屎山吗？ | 有明确的孤儿/测试专用生产包候选、精确重复实现、超大高复杂模块和一个很大的 Python 依赖强连通分量；但没有证据支持“整个项目已经不可维护”。当前更准确的描述是：核心能力很深，但局部编排层和兼容门面正在形成维护重力。不能仅凭静态扫描直接删除。 |
| Agent 已能做信贷风控全流程吗？ | 必须拆成两个答案：本地文件驱动的“分析与开发主链”已大体覆盖，但标签入口、portfolio 和真实材料验收仍未闭环；“生产经营主链”中的部署、决策、持续监控、权限、合规和回滚尚未完成。一个人可以成为业务 owner 和主要操作人，不能同时成为开发、独立验证、审批、合规、生产和审计的唯一责任人。 |

### 1.2 最终定位

当前 MARVIS 最准确的定位是：

> **本地优先、可治理的信贷风控分析与开发工作台 + Agent 工作流执行器。**

它已经明显超过“聊天机器人套工具”的阶段：确定性计算、人在环门禁、证据、审计、模型验证兼容、数据/特征/建模/策略/风险分析等主干均有真实实现和大量测试。
但它尚未证明自己是：

> **能由一个人独立承担整条生产信贷业务线责任的 Risk Operating System。**

决定这一差距的不是再增加几个算法或页面，而是五条尚未闭合的纵向链路：

1. 真实数据与业务口径地面真值；
2. 真实 LLM 的稳定性与安全评测；
3. 真实浏览器、真实服务、真实材料的全旅程；
4. 评分/决策、shadow、晋级、回滚、调度和告警；
5. 多角色职责分离、独立有效质疑、合规和责任签字。

### 1.3 2026-07-31 首轮整改快照（历史）

以下是审计后的同日 dirty-worktree 整改，不改变原始审计对当时证据的描述，也不构成
发布或生产验收：

| 审计问题 | 当前整改 | 本快照证据 |
|---|---|---|
| `_invents_numbers` 恒假、非指标 Tool 可夹带指标字面量 | `PlanValidator` 现在拒绝 `ks=0.42` / exact metric numeric literal 等无工具输出支撑的结果；eval 复用同一检查 | Eval/validator/touchpoint 定向测试合计 `59 passed` |
| 通用 PlanningError 被冒充指定 guardrail | 只接受明确的 INV-1/INV-3 诊断，不再有“任意非 JSON 失败即命中”的 fallback | unrelated ghost-tool 公开 seam 现在得分失败 |
| 空 case 得到 100%、`all([])` 被视为安全 | 空 pass rate/guardrail rate 为 `0.0`，无 guardrail case 时 `guardrail_intact=false`、不推荐 tier | calibration 空集与 normal-only 回归测试通过 |
| 触点账本漂移与三处结构化字段/理由矛盾 | planner fence/prose/think 改为合法非空计划并单轮通过；truncated/key-casing 失败关闭；gate/router/reviewer 的明确否定理由不再允许正向动作，reviewer 缺 exact-case boolean 也失败关闭 | 触点矩阵 `28 passed`，expected-failure ledger 为 0 |
| 六份精确重复审计写入 | 五个 repository 改为保留模块局部 alias、复用 `repositories.audit` 唯一实现 | 六个事务回滚 seam 全部通过；生产定义从 6 份降为 1 份 |
| `db_schema → agent_memory.store → marvis.db` 反向依赖 | DDL 移到中性 `agent_memory_schema.py`；store 直接依赖 `db_schema.connect`，旧 public re-export 保持 | fresh subprocess 初始化不再加载 `marvis.db`/store；migration/store 13 个测试通过 |
| 高置信私有死代码 | 删除 11 个无 Name/Attribute/string/decorator/manifest/route/CLI 动态入口的私有函数；避开正在大改的 compiler 两个候选 | 第一批相关测试全过；第二批领域回归 `74 passed` |
| Portfolio 入口与内核契约混乱 | 产品明确保持 internal / experimental；策略 Agent 不再建议用户跳转到不存在的正式入口；完整报告 setup/template 强制余额/EAD 与业务分群，缺失时失败关闭 | setup/template/routing 定向测试通过 |
| 状态真相源缺失 | 新增 [capability-status.md](../capability-status.md)，逐层区分 implementation、unit、API、Agent、browser、fresh workspace、real data、sign-off、production | README 与 roadmap 已指向唯一状态矩阵 |

最终交叉回归覆盖 Agent auto-drive、instruction router、eval、validator、
agent-memory import/store/retrieval、Portfolio setup/template/API/routing，共
`248 passed`；本轮变更文件的定向 Ruff 与 `git diff --check` 均通过。这些是
dirty worktree 的定向工程证据，不等于同一 immutable SHA 的 full CI、浏览器、
真实材料或生产验收。

仍未关闭：

- T4-2 公开数据参考门、T4-3 真实材料对账和责任签字；
- 当前 dirty worktree 的 immutable SHA、远端 full CI 与六入口真实浏览器 + 真实 LLM 旅程；
- authenticated snapshot 五路去重：其中 scorecard 路径与当前用户改动重叠，必须作为独立安全批次处理；
- Canonical `StrategyWorkflowSpec` / handler adapter：compiler 和 turn handler 都在高频大改，不能在本批强行迁移；
- `strategy_request_compiler.py` 的两个静态死函数：确认无动态入口，但与当前 `+1044/-36` 大改重叠，暂不删除；
- 信用决策数字孪生、生产评分/决策、持续监控、多用户治理与合规闭环。

### 1.4 2026-08-01 整改续报

首轮快照之后，已继续完成 Portfolio 与 Labeling 正式入口、认证数据快照、
多模型分数比较、生产治理 fail-closed 基础、运营调度基础、信用决策数字孪生
认证 replay 入口，以及第二批僵尸代码与重复实现清理。Portfolio 与 Labeling
均已有 real-process HTTP + Playwright 的合成全旅程和下载证据。

这些改动关闭了本审计中“后端存在但用户不可达”、若干 TOCTOU/自报证据边界和
高置信架构债；没有关闭真实机构数据、人工基线、业务签字、真实身份、外部部署
verifier、生产评分/决策和同一 immutable SHA 的远端 full CI。逐项实现、测试、
浏览器证据、外部门和最终结论统一见独立整改记录；不要用本历史段落推断当前状态。

## 二、审计范围与证据边界

### 2.1 当前工作区

审计开始时：

- 分支为 `main`，HEAD 为 `9845e907c702`，与 `origin/main` 无领先或落后；
- 版本为 V2.1.19；
- 工作区已有 123 个 tracked 文件变化，合计约 `+10820/-580`，另有 8 个 untracked 条目；
- 这些既有变化属于用户工作，本审计没有回滚、覆盖或整理它们；
- 因而已发布 SHA 的 CI 只能证明发布基线，不能证明当前 dirty 工作区。

V2.1.19 对应的远端手工 full CI 为成功，GitHub 页面显示 SHA `9845e90`、`full_checks` 成功、总耗时约 3 小时 38 分钟；详细 pytest 数量来自该次任务日志/既有审查记录。这证明源码/tag 基线，不证明 V2.1.19 安装包或完整分发；本次未发现该 SHA 对应的 Windows installer / GitHub Release 验收。当前 dirty 工作区的完整交付必须绑定新的 immutable SHA，并优先由远端门禁重新证明；本地 dirty check 只能作为诊断证据。

### 2.2 证据等级

本审计不把“有代码”直接等同于“能力已交付”，采用以下阶梯：

| 层级 | 含义 | 当前总体状态 |
|---|---|---|
| E0 | 文档或产品声明 | 广泛 |
| E1 | 确定性实现存在 | 强 |
| E2 | 单元、静态、契约、集成测试 | 强 |
| E3 | Agent 可路由和编排 | 主入口较强，部分能力缺入口 |
| E4 | 用户有正式 UI 入口 | 六个主入口成立，部分后端能力无入口 |
| E5 | Fresh workspace | 数据、特征、建模等部分成立；主要使用进程内 `create_app(tmp_path)` |
| E6 | Real-process HTTP | 仅合成 JOIN → LR 建模旅程较明确 |
| E7 | 真实浏览器行为 | 薄弱；Playwright 默认跳过且使用 canned API |
| E8 | 真实 LLM + 真实材料 + 项目参考门 | 阻断或薄弱 |
| E9 | 生产运行、责任治理和业务签字 | 未完成 |

这也是后文所有“已覆盖”“部分覆盖”“缺失”的判定口径。

### 2.3 行业对照

这里的外部标准用于能力对照，不构成任何司法辖区的法律意见：

- 2025 年 BCBS 信用风险原则把完整信用风险管理分为风险环境、授信过程、管理/计量/监控和控制四个区域；仅完成模型开发并不等于全流程信用风险管理。
- 2026 年美国联邦银行机构修订的模型风险管理指引强调有效质疑、结果分析、持续监控、清晰角色和模型清单。该指引主要面向总资产超过 300 亿美元的受监管银行，不是可执行标准，并明确排除 generative/agentic AI；本文只将其类比用于 MARVIS 内传统统计模型的治理，不把它当作 Agent 合规标准。
- NIST AI RMF 强调 Govern、Map、Measure、Manage 应贯穿整个 AI 生命周期并持续执行。
- 美国现行 Regulation B §1002.9 要求不利行动理由具体并指出主要原因，且应与实际考虑或评分因素相关；这里只作为国际能力对照，说明“模型能预测”与“决策能被解释、复核和申诉”是两套能力。曾更直接讨论复杂算法的 CFPB Circular 2022-03 已于 2025-05-12 撤回，本审计不把它当作当前指引。

## 三、问题一：我现在最不自信的是什么？

### 3.1 第一名：真实材料上的业务正确性

当前最不确定的不是 KS/AUC 函数有没有实现，而是当数据来自另一家机构、另一种产品、另一套账务和表现口径时：

- 标签定义、观察窗、表现窗和样本成熟度能否被正确确认；
- 特征时间点是否真的早于决策和表现；
- 人工基线、平台指标、财务结果和历史报告能否独立对账；
- 缺失、特殊值、拒绝样本、重组/核销/展期等业务语义是否被一致处理；
- 模型、预处理、规则和报告能否从原始材料完整重放。

仓库已有很成熟的真实材料 checklist，但当前完成证据仍明确不足：

- `docs/ks_baseline/baselines.json:20-47` 中 GiveMeSomeCredit 和 Home Credit 两个项目自定义参考门均仍缺数据集和人工精调 baseline；
- 本审计实时运行 `conda run --no-capture-output -n py_313 python scripts/ks_baseline.py --status`，退出码为 2；两项均返回 `file_present=false`、`baseline_recorded=false`；
- `docs/plans/v2-master-backlog.md:383-384` 的 T4-2、T4-3 仍未完成；
- `docs/reviews/closure-real-materials-machine-check-2026-07-24.md:1-56` 是历史失效快照，不能证明当前代码仍为 `FAIL / BLOCKED_MACHINE`，只能证明它不是当前 PASS 证据；
- `docs/plans/v2-real-materials-reconciliation-checklist.md:76-83` 是空白签字模板；在本次 repo/workspace 搜索范围内未发现当前机器预检 PASS 或已完成 sign-off artifact，因此 T4-3 判定为 **NOT PROVEN**，而不是据此断言绝对不存在外部签字。

因此，现在可以说“工程主干很强”，不能说“已在代表性真实业务上证明与专家地面真值等价”。

### 3.2 第二名：真实模型下 Agent 的安全性证据

当前 Agent 的确定性工具边界是正确方向，但 eval 本身存在一个实质性证据漏洞：

- `marvis/orchestrator/eval/runner.py:328-338` 的 `_invents_numbers()` 要求普通步骤同时满足 `is_safety_step(step)` 且不存在 range check；
- `marvis/orchestrator/safety.py:6-11` 对这些普通步骤又以“存在 range check”来定义 `is_safety_step`；
- 排除 join/draft 后，两者逻辑互斥，使该检测对普通步骤恒为 False；
- `marvis/orchestrator/eval/cases.py:174-187` 已明确承认：模型可选非 metric schema 工具，把伪造指标塞进 inputs/reasoning，目前只有 prompt 文案防守；
- `marvis/orchestrator/eval/cli.py:7-15` 说明真实模型 eval 不进入 CI；当前只有 7 个 initial offline case，`tests/test_orch_eval.py:517-539` 用 `_ScriptedLLM` 执行。本次未发现可比较的 `workspace/eval/*.json` 真实模型报告。

这不表示确定性工具计算出的指标是错的，而是表示“Agent 不会绕开工具夹带数字”这一安全主张尚未被当前 eval 有效证明。

另有两个会高估 eval 安全性的实现：

- `marvis/orchestrator/eval/runner.py:341-352` 会把多数一般性 planning failure 回退记为该 case 期望的 guardrail 命中；
- `marvis/orchestrator/eval/scoring.py:134-138` 在空集合分母时返回 100%，应改为 N/A/阻断或显式空样本状态。

### 3.3 触点安全账本自身已经漂移

`tests/test_orch_eval_touchpoints.py:174-180` 锁定 8 条 `expected_failure`。审计进一步复核发现：

- 其中 gate、instruction router、reviewer 的 3 条“结构化 confirm/pass 与自然语言否定理由互相矛盾”仍是真实不安全路径；
- 另外 5 条属于 planner，其中多条说明仍声称 planner 使用裸 `json.loads`；
- 当前 `marvis/orchestrator/planner.py:390-405,719-754` 已改用 `load_json_object`，可容忍 Markdown fence、前后说明和 `<think>` 前缀；
- 本审计用生产 `Planner` 和非空合法计划做动态探针，plain、fence、prose、think 四种形式均一次解析成功；
- 现有触点测试仍给 planner 输入 `{"steps":[]}`，最终失败来自“计划不得为空”，却继续被记为解析器不安全。

因此，“8 条均为当前真实 unsafe”也是不准确的。更可靠的当前表述是：

1. 3 条结构化动作与否定理由冲突仍被账本动态复现；
2. 指标夹带及其 eval 检测恒假是另一条真实安全缺口；
3. planner 的若干 expected-failure 标签已经陈旧，应重写为非空有效载荷并重新分类。

这同时说明：项目不仅需要更多 eval，还需要验证 eval 本身的检测力。

### 3.4 第三名：后端能力与用户可达性

`portfolio` 是合法 task type，并有模板和后端工具，但：

- `marvis/domain.py:24-40` 包含 `TASK_TYPE_PORTFOLIO`；
- `marvis/agent/validation_app_service.py:169-182` 的 Agent allowlist 不包含 portfolio；
- `marvis/static/js/task-types.js:115` 的首页六类任务也不包含 portfolio；
- `README.md:91-108` 明确承认组合能力已有后端工具/模板，但不是当前受支持的对话任务入口。

更具体的问题是，策略 Agent 还会建议用户转到 portfolio 任务，而首页没有对应入口。
标签构造也类似：labeling pack、模板和测试存在，但缺少独立 task type、正式首页入口和自然语言全旅程。

这类“后端存在但用户进不去”的能力，最容易在路线图或代码统计中被误算成已交付。

## 四、问题二：你现在最大的盲点是什么？

### 4.1 最大盲点：横向覆盖被误认为纵向闭环

MARVIS 的横向能力非常广：

- 文件摄取、清洗、JOIN；
- 标签与样本设计；
- 特征分析、筛选和衍生；
- 多算法建模、调参、比较、校准和交付；
- 模型验证；
- 规则、评分卡、树、Voting、Pool 和策略报告；
- Vintage、流率、迁徙、Expected Loss、利润和组合分析；
- 监控、记忆、审计、恢复和结构化产物。

但一个能力真正可以交给业务使用，还需要同时满足：

```text
实现
  → 确定性测试
  → Agent 路由
  → UI 可达
  → 真服务/真浏览器
  → 真实 LLM
  → 真实材料与人工基线
  → 生产运行与责任签字
```

当前前两层很强，主流程部分到达中间层，最后两层明显不足。
因此最大的管理风险不是“不知道还缺什么工具”，而是没有用同一张能力矩阵持续区分这些验收层。

### 4.2 第二盲点：状态真相源不再可信

`docs/plans/v2-master-backlog.md:1-7` 宣称自己是 V2 唯一权威来源，但：

- `docs/plans/v2-master-backlog.md:15-24` 仍把 FSP-0A 到 FSP-8 全部标成未完成；
- `docs/roadmap.md:7-14` 已声明 Phase 0A/0B/1 和策略七步主干完成；
- 主清单顶部 `PREP-1/2/3`、`FS-1` 等已标记完成，但附录仍保留修复前的“当前缺口”叙述；
- Agent 触点账本仍保留已被当前解析器修复的 expected-failure 解释。

这不是纯文档美观问题，而是决策问题：如果唯一真相源不能回答“已实现、已接线、已验证、已业务验收、已生产”分别是什么状态，就会持续投资到错误的下一步。

### 4.3 第三盲点：一个人更高效，不等于一个人应承担全部责任

Agent 可以显著减少分析、编码、重复报告和证据整理的工作量，但以下职责不能因为自动化而自然消失：

- 数据拥有者确认字段和账务口径；
- 模型风险人员进行独立、有效质疑；
- 业务审批人确认阈值、额度、定价和策略采纳；
- 合规人员确认公平性、拒绝原因、通知和申诉；
- 生产工程保障实时/批量服务、发布、监控和回滚；
- 财务/风险责任人对真实结果对账并签字。

“一个业务专家 + Agent”可以成为高杠杆的小团队核心，但不应被产品文案描述成取消职责分离。

## 五、问题三：三个月后最可能的失败原因

### 5.1 首要失败模式：能力扩张速度超过证据收敛速度

最可能的因果链是：

1. 新 Tool、Workflow、策略场景和 UI 纵切继续快速增加；
2. 代码和测试继续主要证明合成数据、Fake LLM、TestClient 和局部契约；
3. roadmap、master backlog、review 和 eval 账本继续出现状态分叉；
4. 首个真实业务试点暴露字段语义、模型泛化、LLM 理解或外部对账差异；
5. 因缺少同一 SHA 上的项目公开数据参考门、真实材料签字和全旅程证据，团队无法迅速定位错在数据、算法、Agent、UI、口径还是生产集成；
6. 平台继续依赖开发者现场兜底，业务方拒绝承担结果责任。

一句话：

> **项目最可能败在可信闭环，而不是功能清单。**

### 5.2 架构会成为放大器，但不是第一原因

当前 37 万行以上 Python、多个万行级模块、大依赖环和 123 文件级并行变化，会让上述证据问题更难收敛：

- 修改一个策略请求可能穿过编译、turn handler、renderer、repository、前端 controller 多层；
- 兼容门面与包级 re-export 让依赖方向不清晰；
- 重复的审计/证据绑定 helper 可能在不同路径上逐渐产生语义偏差；
- 超大回合处理器和请求编译器让精确定位、局部测试和审查成本上升。

所以架构债不是“项目三个月后失败”的最初诱因，但会放大修复时长和认知负担。

### 5.3 三个月预警指标

若出现以下任意三项，应认为正在进入失败轨迹：

- 新功能数持续增长，但 T4-2/T4-3 仍未关闭；
- 真实模型黄金集仍只有个位数且不进入周期性门禁；
- portfolio、labeling 等继续停留在后端存在、产品不可达；
- master backlog 与 roadmap 状态仍无法自动对账；
- 单次变更长期维持 100+ 文件；
- 每次发布仍只能引用旧 SHA 的 full CI；
- 新增 E2E 仍以 canned API/Fake LLM 为最终证据；
- 真实业务负责人仍没有对一份完整材料签字；
- 监控仍是“可单次运行”，而不是有日历、重试、告警、处置和回滚的持续系统。

## 六、问题四：一个真正可能形成行业差异的功能

### 6.1 可治理的信用决策数字孪生

建议名称：

> **Governed Credit Decision Twin / 可治理的信用决策数字孪生**

它不是再增加一个报表或算法，而是把 MARVIS 已有的数据、模型、策略、组合、监控、证据和门禁组装成一条难以复制的闭环：

```text
历史申请、授信、表现和账务结果
    ↓
冻结某个历史时点的业务语义与版本
    ↓
逐笔重放数据 → 特征 → 模型 → 策略 → 额度/定价 → 人工覆盖
    ↓
Champion / Challenger / Counterfactual 比较
    ↓
审批率、坏账、利润、资本、稳定性、公平性联合约束
    ↓
Shadow 运行与预测差异归因
    ↓
maker-checker 晋级 / 单环境回滚
    ↓
真实结果回流、持续校准和完整审计
```

### 6.2 核心能力

1. **任意时点重放**：按当时可见数据和当时版本重建逐笔决策，防止时间穿越。
2. **完整 lineage**：数据快照、字典、标签、预处理、模型、规则、阈值、价格、额度、人工覆盖均可追溯。
3. **反事实与离线策略评估**：不仅比较“模型分数”，还比较“若采用另一策略，审批、风险和收益会怎样”；对不可识别的反事实明确给出不确定性和适用边界。
4. **联合约束**：同时管理坏账、审批率、利润、资本、稳定性、公平性和运营容量，而不是单目标追 KS。
5. **Shadow / Promotion / Rollback**：开发、验证、shadow、生产状态分离；任何晋级都需要门禁、责任人和可回滚版本。
6. **结果对账**：预测风险/收益与实际账务、表现、人工处置持续 reconciliation。
7. **决策解释**：输出基于实际使用特征和规则的具体原因，保留人工复核和申诉链。
8. **Agent 的正确角色**：Agent 负责提出实验、解释差异、识别风险和整理证据；确定性引擎负责计算、重放和执行；人负责高风险授权。

### 6.3 为什么它比“再加一个工作流”更领先

- 它把现有能力从功能集合变成可证明的经营闭环；
- 它直接解决真实材料、生产落地、模型风险和业务签字四个最弱点；
- 它的壁垒来自长期积累的版本化证据与结果回流，而不是单个 LLM 或算法；
- 它能把“一个业务专家做更多事”升级为“一个小团队可安全运营更多策略”，同时不破坏职责分离。

### 6.4 前置条件

不建议立刻以新一轮大规模功能开发启动。至少先完成：

1. Agent eval 检测力修复；
2. T4-2 项目公开数据参考门和 T4-3 真实材料签字；
3. 唯一能力状态矩阵；
4. 模型/策略/数据版本重放契约；
5. FSP-7/8 中的调度、环境、RBAC、maker-checker、shadow 和 rollback 基础。

否则“数字孪生”会变成另一层漂亮 UI，而不是可信系统。

## 七、问题五：僵尸代码、重复代码与架构健康

本轮没有发现可以仅凭现有证据判定为“正在造成运行失效或业务数据损坏”的 P0 架构缺陷。项目整体也不应被概括为“屎山”：确定性计算、Plugin/Tool、Repository、validation、output 等边界仍有扎实基础。更准确的判断是，`strategy / agent / static frontend` 已形成局部高维护熵区，增长方式需要收敛。

### 7.1 量化快照

以下是当前 dirty 工作区的静态快照，不代表发布基线：

| 指标 | 结果 | 如何解释 |
|---|---:|---|
| Python 文件 | 504 | 业务域和工作流面已经很大 |
| Python 代码行 | 约 375,460 | 包含大量策略、报告和兼容实现 |
| 函数/方法 | 10,486 | AST 统计 |
| 函数长度中位数 | 15 行 | 多数函数仍较小 |
| 函数长度 P95 | 103 行 | 尾部复杂度明显 |
| 超过 100 行的函数 | 555 | 需要按认知边界治理 |
| 超过 200 行的函数 | 105 | 高维护成本候选 |
| 超过 500 行的函数 | 8 | 应优先重构或证明其深模块价值 |
| Ruff 额外复杂度启发式 | 1,947 条 | 非仓库强制规则，只是密度信号；其中 561 个 C901、741 个参数过多 |
| Python 强连通依赖分量 | 最大约 72 个模块 | 包含局部 import、包级 re-export 和 `TYPE_CHECKING` edge 的静态图信号；不等于 72 个模块形成 import-time runtime cycle |
| 生产 JS 可达性 | 53 个模块中 52 个可从 `app.js` 到达 | 前端模块图整体比 Python 清晰 |

### 7.2 超大模块与超长函数

最显著的文件：

| 文件 | 约行数 | 风险 |
|---|---:|---|
| `marvis/agent/strategy_request_compiler.py` | 19,901 | 请求识别、grounding、确认文案和多策略变体聚合，变化轴过多 |
| `marvis/agent/turn_handlers.py` | 15,059 | 多任务回合编排集中，内部依赖 fan-out 约 96 |
| `marvis/static/js/v2/strategy_candidate_lab_controller.js` | 8,895 | 单一前端 controller 承担过多状态与交互 |
| `marvis/agent/renderers.py` | 8,853 | 多领域输出渲染聚合，变化理由不一致 |
| `marvis/static/styles.css` | 8,670 | 全局样式维护和回归定位成本高 |
| `marvis/packs/strategy/report_bundle_adapters.py` | 8,400 | 报告适配和领域证据拼装过重 |
| `marvis/static/app.js` | 7,821 | 应用入口仍有较多协调职责 |
| `tests/test_frontend_static_v2.py` | 13,486 | 317 个测试中大量为源码字符串/元素 id 断言，体量已开始阻碍重构 |

最长函数包括：

- `strategy_request_compiler._standard_workflow_confirmation_text`：约 901 行；
- `turn_handlers._run_validated_strategy_request`：约 892 行；
- `risk_analysis.calculations.calculate_vtg_terminal`：约 812 行；
- `db_schema._migration_001_baseline`：约 670 行；
- `strategy.project_context_tools._build_state`：约 648 行。

长并不自动等于坏。数据库基线 migration 长但变化频率低，可以接受；请求编译、回合处理和渲染属于高频变化面，长函数会直接增加修改和审查风险，应优先治理。

Strategy workflow 的事实还分散在至少这些独立表面：

- `strategy_request_compiler.py:66` 的 42 个 fresh workflow；
- `strategy_request_compiler.py:3623` 的请求 JSON schema；
- `strategy_request_compiler.py:3984` 的编译入口；
- `turn_handlers.py:3801` 的手工执行分派；
- `orchestrator/templates/sample.py:91` 注册的 76 个内置模板；
- `packs/strategy/manifest.json:8` 起的 60 个 strategy tools；
- `renderers.py:8663` 的 85 个 renderer 注册；
- Candidate Lab HTML 中约 35 个 workflow 表单，以及各自的 JS 收集、同步和提交逻辑。

这些数量不需要机械相等，问题是目前没有一个可自动验证它们关系的单一 workflow 契约。一个新纵切可能在 compiler、confirmation、template、execution、renderer、HTML、JS 中只接通一部分，而源码形状测试仍然通过。

### 7.3 三个生产未接线候选与一个 eval 资产归位问题

“候选”表示静态生产引用为零或只有测试引用，不表示允许直接删除。

| 候选 | 当前证据 | 置信度 | 建议 |
|---|---|---:|---|
| `marvis/static/js/v2/data_workspace_controller.js` | 约 376 行；不在 `app.js` 生产依赖图，仓库引用主要来自前端测试和 package-data 检查 | 高 | 产品决定二选一：正式挂载 Data Workspace，或连同只验证存在性的测试一起移出生产包 |
| `marvis/packs/strategy/model_score_evidence_adapter.py` | 约 265 行；核心函数只在专门测试中出现，未发现 manifest、模板或内部生产调用；但它是带 `__all__` 的治理比较 API | “无内部调用”为高；“可安全删除”为中低 | 先查外部 API/semver 和动态入口；优先接入真实 evidence flow 或 deprecate，不能直接删除 |
| `marvis/data/sampler.py` | 约 89 行；public `sample_dataset` 只被 sampler/profiler 测试调用，当前 profiler 直接使用 `backend.sample_rows` | “无内部调用”为高；“可安全删除”为中 | 先查外部 API/semver；若无外部支持承诺，deprecate 后移入测试 helper 或删除 |

`marvis/orchestrator/eval/touchpoint_cases.py` 不属于僵尸代码。它被 `tests/test_orch_eval_touchpoints.py` 主动消费，是有价值的可执行安全回归语料。问题是它与生产真实模型 eval CLI 使用两套 case 集，且部分标签已陈旧；应把它归类为“测试/eval 资产的位置与接线问题”，选择移到 `tests/eval`，或正式接入生产 eval CLI 并统一状态。

另外发现 13 个高可信未调用 private definition：在当前 `marvis + tests` 和 HEAD 中均只有定义自身一次，也未出现在 manifest、route、`__all__` 或字符串动态入口。合计约 294 行：

- `agent_memory/retrieval.py:295-300` `_is_usable_model_experience`
- `agent/strategy_request_compiler.py:12346-12510` `_ground_strategy_sample_design_request`
- `agent/strategy_request_compiler.py:16316-16330` `_automatic_tree_column_spans`
- `agent/join_setup.py:227-228` `_data_datasets`
- `agent/gate_payloads.py:970-971` `_has_monotonic_policy`
- `data/backend.py:1731-1732` `_iter_strings`
- `validation/stress_test.py:43-46` `_model_features`
- `validation/stress_test.py:49-62` `_filter_feature_categories`
- `packs/modeling/tune.py:490-493` `_sample_params`
- `packs/strategy/sample_design_v2_native_tools.py:1408-1441` `_native_source_binding_from_provenance`
- `packs/strategy/pool_impact_tools.py:1682-1697` `_require_pool_measurement_target`
- `packs/strategy/pool_impact_tools.py:1700-1713` `_require_pool_sample_design_ref`
- `packs/strategy/pool_requirement_resolver.py:541-556` `_outer_requirement`

这批比模块级零引用更适合小批清理，但仍应按业务 family 分批删除并运行相关测试，不应把 294 行做成一个无差别提交。

必须保留的反例：

- `notebook_worker`、`subprocess_worker` 等可能以字符串模块名启动，静态零引用不代表死代码；
- `packs/_sample/tools.py`、`packs/labeling/tools.py`、`packs/risk_analysis/tools.py` 由 manifest 动态加载；
- `strategy/tools.py` 中的短 `tool_*` wrapper 是 manifest `(inputs, ctx)` 到 typed runtime 的 adapter seam，不是普通 pass-through 垃圾；
- `api.py` 的少量无内部引用 wrapper 被定义为兼容 re-export，外部 extension 不能通过仓库静态分析排除，应先 deprecate；
- Plugin manifest、CLI entry point、前端动态 import 和打包资源都需要单独检查；
- 因此任何删除前必须做“静态引用 + 动态入口 + 打包 + 测试检测力”四联验证。

本审计没有删除任何候选。

### 7.4 精确重复与可合并 seam

AST 规范化函数体发现的高价值重复包括：

| 重复 | 位置/规模 | 建议 |
|---|---|---|
| `_write_audit_row` | `repositories/audit.py` 已有共享实现，但 datasets、drafts、modeling、plugins、strategy 等 repository 仍有约 29 行的精确副本 | 统一到 repository 层的一个事务化 AuditWriter；在审计 schema 边界补契约测试 |
| `_validate_pool_ref` | `impact_cube_tools.py` 与 `pool_validation_tools.py` 有约 32 行精确副本 | 抽成 PoolEvidenceVerifier，统一 revision/hash/lineage fail-closed 语义 |
| `_strict_json_object_from_text` | 多个 strategy 模块存在同类/重复实现 | 统一到治理 artifact 的 fail-closed JSON reader；fence/prefix/`<think>`/截断/重复 key/NaN 均应拒绝，不能复用 LLM reply 的宽容提取器 |
| Sample Design / evidence binding helper | impact、pool validation、model evidence 等模块有相似 loader/validator | 只抽取认证 I/O、hash、identity 等底层 `EvidenceBindingVerifier`；sample/pool/model 的业务语义仍按 family 校验 |
| `_output_artifact`、`_amount_metrics`、`_population` | 分布在 modeling/strategy/feature 的小型重复 | 只在业务语义完全一致时合并；先建立输入/输出口径测试，避免为了去重误合并不同领域含义 |
| 认证数据快照读取 | scorecard candidate、impact cube、DSL delivery、pool validation、candidate stability 有五套 `nofollow → fstat → copy/read → hash/size/stable-stat → parquet` 流程 | 抽成 `AuthenticatedDatasetSnapshotReader`，领域工具只保留错误翻译和业务 schema |
| Excel 公式注入识别 | dataset export、automatic-tree report、candidate report、strategy bundle 有四份完全相同逻辑 | 统一安全实现并做恶意单元格 golden tests |
| canonical JSON | repository、modeling、strategy tool 中广泛重复 | 先定义带 profile/version 的 `canonical_json_bytes()` 并锁定历史字节，再迁移；不能直接全局替换 |

重复清理的原则不是追求零重复，而是消除会独立演化的治理语义。审计、hash、lineage、样本绑定、授权和 fail-closed 属于必须单一来源的内容。

### 7.5 依赖环与门面问题

静态 import 图发现一个约 72 模块的强连通分量，主要跨越：

- `marvis.db` / `db_schema` / repositories；
- Plugin registry/runner/SDK；
- modeling 和 strategy packs；
- Agent 与应用协调层。

该数字包含局部 import、包级 re-export 和 `TYPE_CHECKING` edge，只能作为静态依赖方向混杂的信号，不能解释成 72 个模块会形成 import-time runtime cycle。后面的具体反向 import 路径才是可操作证据。

两个典型原因：

1. `marvis/db.py` 同时作为兼容 re-export 门面和内部依赖入口；内部模块通过门面导入 repository，而门面又导回大量具体模块。
2. `marvis/packs/strategy/__init__.py` 大量 eager re-export，部分包内模块再从包顶层导入兄弟模块。

其中两条可以直接读出的反向依赖是：

```text
marvis.db
  → marvis.db_schema
  → marvis.agent_memory.store
  → marvis.db
```

- `db.py:3` 导入 `db_schema`；
- `db_schema.py:865` 在 migration 中导入 `agent_memory.store`；
- `agent_memory/store.py:24` 又从 `marvis.db` 导入 `_now, connect`。

以及：

```text
db façade
  → repositories.strategy
  → typed_backtest
  → materialized_runtime_requirements
  → pool_requirement_resolver / pool_tools
  → marvis.db
```

另有 `strategy/tools.py:180` 与 `monitor_tools.py:1785-1786` 通过 lazy import 形成的显式双向依赖。这是工具包内部的职责纠缠；schema/repository 的层次反转由前述两条大依赖环证明。它们目前都不一定直接触发 import error，但会模糊变更边界。

建议：

- 保留 `marvis.db` 和包顶层 re-export 作为外部兼容 seam；
- 内部代码改为直接依赖 `db_schema`、具体 repository 或具体 strategy 子模块；
- schema 层不再导入 store 实现，将 Agent Memory DDL 移到纯 schema installer；
- repository 不再导入可执行 pack/tool，把 persistence DTO、序列化和 contract 下沉到中性领域模块；
- monitoring 共用的 artifact persistence helper 从 `strategy/tools.py` 移到中性模块；
- 用 import-linter/自定义架构测试明确 `domain → repositories/services → app/agent/ui` 的允许方向；
- 每一批只打破一个小环，并用导入时间、测试和 public import 兼容性证明没有回归。

其他静态小分量要分别解释：

- `pipeline` / `pipeline_cellgen` / `pipeline_io` 的反向边只存在于 `TYPE_CHECKING`，不是运行时循环；
- `interactive_tree_revision` / `interactive_tree_revision_v2` 是显式 lazy compatibility adapter，应作为兼容边界管理，不与普通架构债并列；
- strategy `monitor_tools` / `tools` 才是实际 lazy 双向依赖，应拆出共用 persistence seam。

### 7.6 不应采取的“整理”

- 不要仅按文件行数机械拆文件；拆完仍共享同一巨大上下文，只会增加跳转。
- 不要为消除重复而合并业务口径不同的函数。
- 不要先删所有静态零引用；动态入口在该项目中是真实存在的。
- 不要在 123 文件 dirty diff 上同时做大规模架构迁移。
- 不要把兼容 public import 和内部依赖治理一次性完成。

### 7.7 推荐的架构收敛顺序

#### P0：先修证据，不先大拆文件

1. 修复 `_invents_numbers` 检测恒假；
2. 重写触点 eval 的陈旧 planner case；
3. 让通用 planning failure 不再被自动算成目标 guardrail 命中；
4. 建立当前能力矩阵和状态真相源；
5. 让当前工作按可审查边界落到独立 SHA，并由远端 full gate 形成不可变交付证据。

#### P1：建立四个小而深的接口

1. **Canonical StrategyWorkflowSpec / Registry**：每个策略 family 在唯一 registry 中提供 compile/ground/validate/confirm、template/slot、handler adapter 和 presenter reference；主引擎只面对这一个注册表，不能再建立独立的 request registry 与 handler registry。
2. **Turn handler adapter**：作为 `StrategyWorkflowSpec` 的一部分，`turn_handlers` 只保留生命周期协调，不再拥有第二份 workflow 事实源。
3. **EvidenceBindingVerifier**：只统一 authenticated I/O、artifact/dataset identity、hash 和 lineage 机制；sample/pool/model 的业务语义仍由 family verifier 负责。
4. **AuditWriter**：统一 repository 审计写入、事务和 detail schema。

#### P2：清理孤儿和兼容依赖

1. 对三个生产未接线候选给出“接线 / deprecate / 删除”决定，并统一 touchpoint eval 资产的位置与 CLI 接线；
2. 内部模块停止通过 `marvis.db` 和 strategy 包顶层互相导入；
3. 合并 exact duplicate；
4. 给 import 边界加自动化架构测试。

#### P3：按变化理由拆 UI 和渲染层

- Candidate Lab controller 按“状态投影、表格/选择、提交、持久化”真实变化轴拆分；
- renderer 按领域 presenter 注册，不按任意行数切片；
- CSS 以 token、layout primitive、workflow component 和页面特例分层；
- 每次拆分都要求行为测试或真实浏览器截图/DOM 证据，而不是只做 source-shape 测试。

前端还存在两类事实源重复，应在 P3 一并治理：

- Pet id、资源路径和 alias 同时维护在 `app.js`、`index.html`、`pyproject.toml`、`test_frontend_static_v2.py` 和各 pet 的 `pet.json`；运行时又没有消费 `pet.json`。应选一个 canonical catalog 并生成/嵌入其他投影。
- `test_frontend_static_v2.py` 的源码字符串断言只适合保留为一层很薄的 package/source contract；workflow 字段矩阵应从 registry 生成，主要行为迁移到 controller interface 和真实浏览器测试。新测试覆盖后必须同步删除旧 source-shape 断言。

## 八、问题六：Agent 是否已能完成信贷风控全流程？

必须先拆开“全流程开发”和“全流程生产经营”：

- **本地文件驱动的分析/开发主链：SUPPORTED。** 数据 → 特征 → 建模 → 验证 → 策略 → Vintage/风险分析已大体覆盖，标签独立入口、portfolio 产品入口和真实材料验收仍有缺口。
- **持续生产经营主链：NOT READY。** 数据源运营 → 评分/决策 → 发布/shadow/rollback → 持续监控/告警 → 权限/合规/审计没有闭合。

### 8.1 能力矩阵

| 生命周期 | 实现/测试 | Agent 路由 | UI/真实浏览器 | 真实材料/生产 | 自动化边界 | 当前判断 |
|---|---|---|---|---|---|---|
| 文件数据摄取、清洗、JOIN | 较强 | 已接线 | 有正式入口；E2E 以合成/API 为主 | 缺生产 SQL/数仓连接和长期数据运营 | profile/计算可自动；字段语义、主表、JOIN/去重须确认 | 本地开发可用 |
| 标签与样本设计 | pack、模板、成熟度门存在 | 作为模板能力存在，缺独立任务路由 | 无独立正式入口/对话全旅程 | 真实观察窗、表现窗和账务口径未签署 | 规则执行可自动；标签定义和成熟度必须人工确认 | 后端能力存在，产品化不足 |
| 特征分析与筛选 | 指标、筛选、衍生、报告较完整 | 已接线 | 有正式入口；fresh-workspace 进程内证据较强 | 缺跨机构真实材料签字 | 指标自动；特征保留、override 和业务含义须确认 | 较强本地能力 |
| 模型开发 | split、select、tune、train、compare、calibrate、PMML/handoff 丰富 | 已接线 | 有正式入口；real-process HTTP 仅证明合成 JOIN→LR 主旅程 | 两个项目公开数据参考门与真实模型对账未完成 | 有界搜索和训练可自动；最终选择、阈值和交付须确认 | 强研发工作台 |
| 模型验证 | V1 兼容、手工与 Agent、确定性报告成熟 | 已接线 | 正式入口较成熟 | 独立模型风险有效质疑和责任签字仍是组织职责 | 指标自动；验证结论、限制接受和使用授权须人工 | 当前最成熟环节之一 |
| 策略开发 | 七步、Pool、回测、影响、报告和本地采纳能力丰富 | 已接线 | 正式入口，测试面广 | 本地 adoption 不等于生产 deployment | 计算/有界搜索可自动；入池、采纳、生产动作须确认 | 强本地策略工作台 |
| Vintage / roll-rate / 利润风险分析 | 有确定性计算和报告 | 经 `vintage` 正式任务路由 | 有正式入口/部分 E2E | 真实账务口径和签字未完成 | 字段、单位、截止日、场景确认后自动计算；结论须人工 | 本地分析可用 |
| 模型/策略监控 | 可单次运行、产出状态/草稿 | 策略监控集成在 strategy task；模型监控主要通过 modeling handoff | 无独立 monitoring 首屏任务；浏览器持续旅程未证明 | 缺 scheduler、告警、重试、值班和持续处置 | 单次诊断可自动；红灯处置和版本动作须确认 | 不是生产监控系统 |
| Portfolio / 集中度 / EL / 组合趋势 | 后端工具、模板和报告存在 | 不在正式 Agent allowlist | 无首屏/受支持对话入口；技术用户可经后端/API 使用 | 外部账务对账未完成 | 计算可自动；经营解释、限额和资本决策须确认 | 后端较强、产品未接线 |
| 实时/批量评分与决策 | 有本地打分和交付物 | 可辅助交付，非生产执行路由 | 无完整生产操作面 | 缺服务、环境晋级、shadow、rollback 和决策引擎 | Agent 可准备/解释；生产执行未覆盖 | 明显缺口 |
| 多用户与组织治理 | 有本地 token、审计和门禁基础 | 局部 | 无完整企业操作面 | 缺真实 identity、RBAC、maker-checker、多环境权限 | 高风险状态必须人工；组织执行能力未覆盖 | 明显缺口 |
| 合规与模型风险 | 有 provenance、报告和人工 gate 基础 | 局部辅助 | 无独立完整工作台 | 缺系统化公平性、拒绝原因、申诉、模型清单/例外处置 | Agent 仅解释/起草；确定性验证和责任审批不可替代 | 生产级缺口 |
| 催收 | 无完整工作流 | 无 | 无 | 未覆盖 | 未覆盖 | 缺失 |
| 反欺诈 | 通用建模可训练欺诈标签，但无完整反欺诈系统 | 无 | 无 | 产品明确留口 | 未覆盖 | 缺失 |
| 征信/三方报文 | 无完整解析与生产接入 | 无 | 无 | 被显式推迟 | 未覆盖 | 缺失 |

### 8.2 当前一个人可以做到什么？

对于“懂业务、具备基础数据能力、在本地工作、关键口径愿意确认”的用户，MARVIS 已有潜力让其独立完成：

- 文件数据检查、JOIN 和质量诊断；
- 样本/标签方案讨论与受控执行；
- 特征分析、筛选、分箱和报告；
- 多模型实验、比较、验证和交付材料；
- 规则/评分/树/Voting 等策略开发与本地回测；
- Vintage、迁徙、流率和利润分析；技术用户还可通过后端/API 使用部分 portfolio 能力，但普通 UI/Agent 用户没有正式入口；
- 生成结构化证据和可审计报告；
- 在高风险节点保留人工确认。

### 8.3 为什么还不能独立负责生产业务线？

因为“负责一条业务线”还包括：

- 对生产数据源、延迟、回补、权限和质量负责；
- 对实时/批量决策服务及故障负责；
- 对模型和策略发布、shadow、回滚负责；
- 对持续监控、告警、值班和处置时效负责；
- 对公平性、拒绝原因、申诉和监管证据负责；
- 对模型开发与独立验证之间的职责分离负责；
- 对财务/风险真实结果和业务阈值签字负责；
- 对催收、反欺诈、征信和运营反馈负责。

这些不是再提高 Agent 自主级别就能消失的职责。
一个人可以成为唯一的风控业务 owner 和主要操作人，但不能同时成为开发、独立验证、审批、合规、生产和审计的唯一责任人。更现实的目标是：

> **让一个业务专家成为一支小型风控团队的高杠杆核心，而不是让一个人同时成为开发、验证、审批、合规、生产和审计。**

## 九、30 / 60 / 90 天建议

### 9.1 30 天：收敛证据债和架构入口

1. 修复 Agent eval 的恒假检测与陈旧 touchpoint case；
2. 将 7 个 initial offline case 扩展为覆盖六主入口的黄金集：歧义、否定、缺材料、越权、伪造指标、多轮修订、stale evidence、超长上下文；
3. 修复 offline/touchpoint eval 的检测力并保持其为 blocking fast CI；真实模型 eval 定期运行并保存可比较基线；
4. 建立唯一能力矩阵，字段至少包括 implemented、unit、API、Agent、browser、fresh workspace、real data、sign-off、production；
5. 对齐 roadmap、master backlog 和 eval ledger；
6. 将当前 123 文件变更按能力边界收敛到可审查 SHA，并让该 SHA 通过 full CI；
7. 给三个生产未接线候选做保留/接线/deprecate/删除决定，并统一 touchpoint eval 资产；
8. 合并 AuditWriter 和 EvidenceBindingVerifier 这两类治理重复。

**30 天验收：**至少形成四个有限交付物：新 immutable SHA 的远端 full CI、修正后的 eval 报告、唯一能力矩阵、roadmap/backlog/eval 无冲突状态源；任何“完成”声明都能追溯到 SHA、命令、结果和明确证据层。

### 9.2 60 天：证明真实世界

1. 提供 GiveMeSomeCredit、Home Credit 数据和可追溯人工基线，预注册 split、seed、特征范围、调参预算、容差和独立复核人，关闭 T4-2；该门只能证明在预设协议下不弱于记录参考值，不能单独证明专家等价或行业领先；
2. 至少一套真实业务材料完成最新机器预检、B1-B5 外部对账和责任签字，关闭 T4-3；
3. 六个主要入口在真实 FastAPI + 真实浏览器中完成创建、上传、配置、门禁、执行、下载、刷新和重启恢复；
4. portfolio 与 labeling 做明确产品决定：正式接线或从当前交付声明中降级；
5. 当前发布在干净 Windows/Linux 环境完成安装/部署、备份恢复和基础故障 smoke；
6. 建立模型清单、使用目的、限制、验证、监控、例外和责任人最小治理对象。

**60 天验收：**两个项目公开数据参考门按预注册协议通过、真实材料签字、浏览器旅程和发布 SHA 绑定一致。

### 9.3 90 天：用真实业务线试点决定产品称谓

1. 选择一条真实业务线连续运行至少四个周周期；
2. 预先登记 Go/No-Go、风险/收益口径和人工介入边界；
3. 记录人工介入率、失败恢复、重跑一致性、对账差异、缺陷逃逸和交付时间；
4. 做任务卡死、进程中断、磁盘不足、模型服务异常和数据漂移故障注入；
5. 若要称生产业务线系统，补齐 maker-checker/RBAC、环境晋级/回滚、调度告警和生产评分/决策集成；
6. 否则明确将产品称谓稳定在“本地风控开发与分析工作台”。

**90 天验收：**形成已签署的 Go/No-Go 记录，预登记阈值、实际结果、例外和处置全部可追溯，异常时有明确停止、回滚、追责和恢复路径。四周只构成有限试点证据，不自动等同生产就绪。

## 十、优先级总表

| 优先级 | 事项 | 原因 |
|---|---|---|
| P0 | 修复 Agent eval 检测力与陈旧安全账本 | 当前安全主张的证据本身不可靠 |
| P0 | 关闭 T4-2/T4-3 | 这是“工程可跑”升级为“业务可信”的最短路径 |
| P0 | 建立唯一能力状态矩阵 | 防止继续把实现、接线、验收和生产混为一谈 |
| P1 | 当前 dirty 大变更绑定新 SHA 跑远端 full gate | 旧发布 CI 不覆盖当前状态，本地 dirty check 不能替代不可变交付证据 |
| P1 | 真实浏览器 + 真实 LLM 六入口旅程 | 补足当前 Fake/canned/synthetic 证据的盲区 |
| P1 | Portfolio/labeling 入口决策 | 消除后端存在但用户不可达 |
| P1 | Canonical StrategyWorkflowSpec + handler adapter | 用一个 registry 降低两个万行级高频变化模块的认知负担，避免重建第二真相源 |
| P1 | 底层 EvidenceBindingVerifier / AuditWriter | 消除会产生治理分叉的认证 I/O 与审计重复；业务语义仍按 family 保留 |
| P2 | 打破 `marvis.db` 与 strategy 包级依赖环 | 恢复内部依赖方向，同时保留外部兼容 |
| P2 | 治理三个生产未接线候选与 eval 资产 | 减少生产包噪声、虚假能力面和评测状态分叉 |
| P2 | FSP-7/8 生产治理基础 | 为持续经营和决策数字孪生铺路 |
| P3 | 按变化轴拆前端 controller、renderer 和 CSS | 在真实行为测试保护下逐步降低 UI 维护成本 |

## 十一、验证记录与限制

### 11.1 本次执行的主要检查

- Git 分支、HEAD、upstream、版本、dirty/untracked 和 diff 规模；
- `scripts/check --profile` 的 `git diff --check`、Ruff 和 JavaScript syntax 阶段通过；pytest 在 26% 时为 `2810 passed, 3 skipped`、尚无失败，因当前 dirty 工作区完整门禁需数小时且不能形成 immutable 交付证据而主动停止，**不得记为 full PASS**；
- Agent touchpoint 相关定向测试由子审计执行，结果为 `80 passed`；其中 expected-failure 测试通过代表复现账本状态，不代表风险已修复；
- `scripts/ks_baseline.py --status` 实时退出码 2，两项项目参考门均缺数据文件和人工 baseline；
- 生产 `Planner` 非空合法计划探针中，plain、Markdown fence、prose prefix/suffix、`<think>` 四种载荷均一次解析成功；
- Python AST 文件/函数/长度、精确重复函数体、import 图 SCC 和 fan-in/fan-out；
- Ruff 额外复杂度启发式：`C901`、`PLR0911/12/13/15` 等；
- 前端 ES module 从 `marvis/static/app.js` 的生产可达性和循环依赖；
- 产品 task type、Agent allowlist、首页入口和 workflow 模板交叉核对；
- roadmap、master backlog、真实材料 closure、KS baseline、CI、E2E 和 Playwright 证据交叉核对；
- Agent eval 初始 case、touchpoint expected-failure、实际 parser 和动态非空计划探针；
- 三路子审计分别覆盖产品/Agent、验证/失败模式、架构健康。

### 11.2 重要限制

- 静态零引用不能识别所有 Plugin、CLI、字符串子进程和打包动态入口；
- 本次没有真实业务数据、真实机构人工基线、财务对账人或责任签字；
- 本次没有调用外部真实 LLM 执行完整黄金集；
- 本次没有执行生产部署、实时评分、故障演练或多用户权限测试；
- 当前工作区存在大量用户未提交变化，审计结论描述的是这一快照；
- 复杂度规则不是当前仓库强制 lint，数字只能作为重构优先级信号；
- 本文提出的“信用决策数字孪生”是产品建议，不是已实现能力。

### 11.3 最终 verdict

| 维度 | 判定 |
|---|---|
| V2.1.19 源码/tag 基线工程验证 | **PASS；该判定只覆盖发布 SHA 的 full CI，不证明安装包/完整分发，也不外推为当前 dirty 工作区整体工程质量** |
| 当前 dirty 工作区完整交付 | **NOT PROVEN，需对新 SHA 重跑完整门禁** |
| 本地风控分析/开发工作台 | **E1–E3 较强；E4–E7 PARTIAL；E8–E9 NOT PROVEN** |
| 僵尸与重复代码 | **存在可验证候选，可有序清理；禁止直接批量删除** |
| 架构健康 | **可维护但已出现明显维护重力，需要优先治理编排层、证据 seam 和依赖方向** |
| 全流程生产风控 Agent | **NOT DELIVERED，尚缺真实业务与生产闭环** |
| 一人独立负责整条生产业务线 | **NOT PROVEN / 不宜单人全责；可以显著赋能，但不能替代职责分离与生产责任** |
| 行业领先潜力 | **中高；前提是从功能扩张转向证据、结果回流和治理闭环** |

## 十二、外部参考

- [BCBS — Principles for the Management of Credit Risk（2025，Current）](https://www.bis.org/bcbs/publ/d595.htm)
- [Federal Reserve / OCC / FDIC — Revised Guidance on Model Risk Management（SR 26-2，2026；含适用范围与 Agentic AI 排除）](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf)
- [NIST — AI Risk Management Framework Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
- [CFPB — Current Regulation B §1002.9 / Notifications](https://www.consumerfinance.gov/rules-policy/regulations/1002/9/)
- [CFPB — Withdrawn guidance list（Circular 2022-03 已于 2025-05-12 撤回）](https://www.consumerfinance.gov/compliance/guidance/withdrawn-guidance/)
- [V2.1.19 SHA 的 GitHub Actions CI #88](https://github.com/eddyzzl/marvis-risk-agent/actions/runs/30325298518)
