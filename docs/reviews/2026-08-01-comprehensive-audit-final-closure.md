# MARVIS 综合审计最终收口

> 日期：2026-08-01
>
> 最后更新：2026-08-03
>
> 审计基线：`9845e907c702f65edc77a9d0121f0967f52488c2`（V2.1.19）之上的当前 dirty worktree
>
> 原始审计：[项目、Agent 能力与架构健康综合审计](2026-07-31-project-agent-architecture-comprehensive-audit.md)
>
> 整改记录：[综合审计整改记录](2026-08-01-comprehensive-audit-remediation.md)
>
> 信任边界复审：[Presenter Trust Review](2026-08-01-presenter-trust-code-review.md)
>
> 最终本地复审：[Final local closure](2026-08-01-final-local-closure-code-review.md)
>
> 分层状态真相源：[能力验收状态](../capability-status.md)；产品范围真相源：[路线图](../roadmap.md)

## 一、先回答“前面的问题全部解决完了吗”

**如果“全部”包含真实业务、生产运营和发布验收，答案是否。** 当前更准确的三层判定是：

| 层 | 当前判定 | 含义 |
|---|---|---|
| 六个审计问题 | `ANSWERED` | 六个问题均已有代码、测试、产品和能力矩阵证据支撑的明确答案 |
| 本地可操作整改 | `NO_KNOWN_OPEN_LOCAL_BLOCKER / FULL_LOCAL_GATE_PASS` | 已识别的本地高优先级缺陷均已修复，并通过定向、独立攻击复核及最新代码状态的完整本地 fast gate |
| 项目生产交付 | `NOT_PROVEN_FOR_PRODUCTION` | 真实材料、人工基线、业务签字、真实 LLM 准入、原生 Windows、不可变 SHA 远端 CI 和生产系统闭环仍未完成 |

因此，“代码整改已基本收口”不能被改写成“整个项目已经完成”或“已经可以单人承担生产业务线”。

## 二、六个问题的最终答案

### 1. 关于项目和代码，现在最不自信的是什么？

最不自信的已经不是合成样本上的函数正确性，而是**换成真实机构材料后，业务语义、人工基线和生产行为是否仍能一致重放并得到责任人签字**。

具体有三层：

1. **真实业务地面真值**：标签、观察窗、表现窗、成熟度、核销/展期/重组、财务损失和收益等口径，尚未用一套当前真实材料完成 B1-B5 对账与签字。
2. **真实模型稳定性**：2026-08-01 生成的两份最新可用真实 LLM 评测均为 `INCOMPLETE`，分别出现 7 和 12 个 LLM error，三档 guardrail 均不完整，`recommended_tier=None`。这证明系统会失败关闭；当前 worktree 之后已有变化且尚未形成新的同源报告，因此更不能据此宣称已有可推荐自主等级。
3. **生产运行正确性**：真实身份源、外部 deployment verifier、批量/实时决策服务、长期调度告警、值班和故障演练尚未接入。

代码层面的不确定性已经明显下降：exact step-result 回执、结果数据集绑定、authenticated snapshot、下载竞态、报告新鲜度、恢复收敛及 Decision Twin 的本地信任边界均经过对抗复核。但这些证据不能替代真实账务、业务和生产验收。

### 2. 关于当前情况，最大的盲点是什么？忽略了什么？

最大的盲点是把下面这些不同层级合并成一个“完成”：

```text
有文档 → 有实现 → 有单测 → Agent 可路由 → API/UI 可达
→ 浏览器能完成合成旅程 → fresh workspace 可复现
→ 真实材料对账 → 责任人签字 → 生产可运营
```

MARVIS 当前左半段很强，右半段仍弱。容易被忽略的不是又少一个算法，而是：

- 用户是否能从正式入口完成整条旅程，而不是开发者调用内部 Tool；
- 最新失败、损坏或未完成证据是否会阻断，而不是静默回退旧结果；
- 合成 E2E 是否被误当成真实材料等价性；
- 本地 maker-checker 对象是否被误当成企业身份与职责分离；
- dirty worktree 的大量测试是否被误当成 immutable SHA 的发布证明；
- Agent 自主性是否被误当成可以取消独立验证、审批、合规和审计责任。

本轮通过能力矩阵、Portfolio/Labeling 正式旅程、报告 freshness barrier 和 exact binding，已经关闭了一部分“后端有、用户不可达”与“证据串错”的盲点；外部地面真值仍不能由代码补出来。

### 3. 如果项目三个月后失败，最可能的原因是什么？

最可能的原因是：**能力扩张速度继续快于证据收敛和真实试点速度。**

失败形态大概率不是程序突然完全不能运行，而是：

1. workflow、页面、Tool 和测试持续增加；
2. 真实业务线仍没有按预登记口径完成多周期试点；
3. 每次换数据、换机构或换产品仍需要开发者修数据、解释口径和人工兜底；
4. 真实 LLM 仍无安全准入档位，生产治理对象仍未接真实系统；
5. 业务、模型风险、合规和运营因此不愿签字，也不敢把责任交给平台。

三个月内最重要的成功指标不应是新增功能数，而应是：至少一条真实业务线完成可追溯的多周期试点，人工介入、对账差异、例外、故障恢复、回滚和责任人全部有记录。

### 4. 如果只能增加一个行业领先功能，会是什么？

建议仍是：**可治理的信用决策数字孪生（Governed Credit Decision Twin）**。

它的目标不是生成一份更漂亮的报告，而是冻结某一时点的数据、字段语义、特征、模型、策略、人工覆盖和版本证据，逐笔重放 Champion/Challenger，执行有界反事实和风险—收益—公平性联合约束，再经过 shadow、maker-checker、晋级、回滚和真实结果回流形成闭环。

当前已经完成的是**领先功能的可信基础**：

- 本地 content-addressed audit store；
- 冻结 manifest、facts、字段来源、adapter 及可达 helper 的认证；
- point-in-time replay；
- Champion/Challenger/反事实联合约束；
- 只生成 proposal、不能直接采纳或改变生产状态的晋级桥。

当前**没有完成**正式 API/UI、外部签名、远端不可篡改存储、真实结果回流、shadow 运行、生产决策接入和生产晋级闭环。因此它应被描述为“行业领先功能的基础已落地”，不能宣传为“生产级 Decision Twin 已交付”。

### 5. 有没有僵尸代码、屎山代码和可清理冗余？

有局部维护重力和历史兼容债，但没有证据支持“整个仓库已经不可维护”。本轮采取的是高置信、可验证、渐进式清理，而不是按文件行数大批删除。

已经收敛的部分包括：

- 44/44 Strategy family 已迁入 canonical spec、validator、confirmation 和 plan preparer，compiler 侧静态 shadow 分支归零；
- 8 类 governed result 统一进入 presenter registry，并绑定 exact plan/step/run/output/tool-version/manifest 回执；
- 多份重复 `AuditWriter`、`$ref` 解析、表格安全、artifact id 和 authenticated snapshot 实现收敛到共享 seam；
- `memory.before_save` 已真实接线并有顺序、拒绝和审计测试；所有 memory-aware driver 保存路径现在必须传入完整 runtime，11 条真实保存旁路已统一收敛，ad-hoc 保存被明确标记为 context-free，不再以缺 runtime 的隐式方式绕过治理；
- SampleDesign V2 的 selector 与 fresh shape 已收敛到单一权威实现，避免不同消费者各自解释“当前样本”；
- app controller context、DSL JSON 和 GC renderer 的重复实现已统一，减少前后端状态与治理输出漂移；
- pet 资源改为单一 catalog，移除 9 份重复 `pet.json`，并且只保留一个 classic 入口；真实 Chromium 已证明页面只发起一次 classic 资源请求；
- `marvis` 内部对 DB façade 的反向 import 清零；
- renderer、Candidate Lab 和 CSS 完成第一轮按职责拆分；
- 数据集源文件和任务目录增加持久化 GC、重试、隔离和共享 Windows handle 适配层；
- Feature rollback 的旧测试夹具已升级到 exact run/evidence/hash 生命周期，回滚测试不再依赖已经退役的宽松证据形状；
- 人工编辑筛选结果不再覆写已完成的 `screen_features` Tool 输出；受约束的特征集合现在与 gate 确认在同一事务中写入具体 inputs，原始 output ref、hash 和 succeeded run receipt 保持不可变；
- 一年期 `immutable` 静态资源 URL 不再依赖最大 mtime，而是绑定全部 JS/CSS 的相对路径和文件字节 SHA-256；保留时间戳的内容变化也会生成新版本；
- 两批只删除有静态与动态双重证据的私有死代码，没有批量删除可能被 Plugin、CLI、字符串入口、打包资源或兼容面动态使用的对象。
- 最终独立健康复核扫描生产私有函数和同文件函数体：未发现新的“生产、测试、脚本均无引用且不少于 8 行”的私有函数，也未发现同文件不少于 8 行的完全重复函数体。

仍然存在的结构债：

| 区域 | 当前问题 | 正确处理方式 |
|---|---|---|
| `strategy_request_compiler.py`、`turn_handlers.py` | 约 1.45 万和 1.55 万行，认知负担高 | 按 request compiler、presenter、domain service 的变化理由渐进拆分，并保持 golden contract |
| `report_bundle_adapters.py`、部分 Strategy tool、`static/app.js` | 聚合职责仍重 | 在真实变更发生时抽稳定 seam，不做一次性大爆炸重构 |
| canonical 兼容 facade | 迁移完成后仍保留兼容边界 | 至少经过兼容发布周期和消费者证据后再退役 |
| `sample_dataset()` | 已弃用但 stratified 尚无完整替代 | 当前不是普通僵尸，替代能力闭合前不能直接删除 |
| 文件 GC / Windows 删除 | 有平台适配和确定性测试 | 仍需真实 Windows/NTFS sharing violation 原生验收 |
| 跨文件重复与 lazy import 耦合 | 仍有 55 组完全相同函数体候选；加入 lazy import 后有 6 个 SCC，最大 41 模块 | 先确定共享模块所有权和兼容边界，再按业务 seam 收敛；不能据扫描结果直接删除 |

结论：**高置信僵尸和最危险的重复信任边界已清理；代码库仍有明确技术债，但适合继续定向治理，不适合宣称“已经没有屎山”或进行无差别清仓。**

### 6. Agent 已经能做信贷风控全流程开发吗？能否让一个懂业务的人独立负责整条业务线？

必须区分“风控内容生产”与“生产业务线责任”。

在本地文件、口径明确、关键门有人确认且用户具备基础数据能力的前提下，一个懂业务的人现在可以主要依靠 MARVIS 完成：

- 数据检查、JOIN、质量诊断、标签和样本构造；
- 特征分析、筛选、分箱、模型开发、模型比较与模型验证材料；
- 规则、树、评分卡、Voting、Strategy Pool、回测、独立分区验证和本地受控采纳；
- Vintage、迁徙、流率、利润和 Portfolio 分析；
- 结构化证据、审计记录、Excel/Word/JSON/Markdown 报告和受控下载；
- 在样本、口径、策略、结论、采纳等高风险位置保留人工确认门。

但当前还不能让同一个人独立承担：

- 生产数据接入、权限、延迟、回补和质量 SLA；
- 实时/批量评分决策、shadow、逐环境发布、回滚和灾难恢复；
- 独立模型验证、maker-checker、合规、公平性、拒绝原因、申诉与审计；
- 长期监控、值班、财务对账、真实风险收益结果和最终责任签字；
- 催收、反欺诈、征信及三方报文等尚未交付的专业系统职责。

所以答案是：

- 若“独立负责风控内容”指**本地开发、分析、报告与方案迭代的主要 owner**：已经具备很强的单人赋能能力，但仍受真实材料和 Agent 安全准入门限制。
- 若指**独立承担整条生产业务线的全部职责和最终责任**：**不可以，也不应把职责分离设计成由一个 Agent 或一个人消除。**

当前准确定位仍是：

> **本地优先、可治理的信贷风控开发与分析工作台；让一个业务专家成为小型风控团队的高杠杆核心，而不是单人生产 Risk Operating System。**

## 三、本地整改关闭清单

| 整改面 | 当前状态 | 关键证据边界 |
|---|---|---|
| P0 Agent/eval 证据 | 已关闭本地缺陷 | 伪造指标、空样本、错误归因、case-level LLM error、provenance 与 fail-closed 均有回归；真实模型准入仍失败 |
| Portfolio 与 Labeling 可达性 | 已关闭本地纵切 | real-process HTTP + Playwright 合成旅程通过；真实机构组合、标签口径和周期材料未验收 |
| authenticated I/O 与 C1 CAS | 已关闭本地缺陷 | source/CAS swap、pin/GC 锁序、共享引用和失败清理有对抗测试；Windows 原生环境仍待验收 |
| Decision Twin 本地信任边界 | 已关闭已知本地攻击面 | adapter/helper、manifest/data/facts、DSL/cache 和 replay 错配均失败关闭；生产功能仍未交付 |
| canonical Presenter 与结果下载 | 独立复审 `PASS` | exact step binding、私有单 descriptor 快照、registry/文件同步替换、Range 206/416 均有回归 |
| 报告新鲜度与恢复 | 独立复审 `PASS` | 最新失败/运行状态阻断旧报告；父绑定冲突收敛为 step/plan `FAILED` |
| Strategy 重复事实源 | 已关闭迁移项 | 44/44 canonical migration、8 个 presenter；大文件物理拆分仍是后续渐进债 |
| Memory 保存治理 | 已关闭已知旁路 | 所有 memory-aware driver 路径强制完整 runtime；11 条真实保存旁路统一；ad-hoc 显式 context-free |
| SampleDesign 与共享展示 seam | 已关闭重复实现 | SampleDesign V2 selector/fresh shape 单一权威；app controller context、DSL JSON、GC renderer 统一 |
| Feature rollback 证据生命周期 | 已关闭旧夹具漂移 | rollback 夹具使用 exact run/evidence/hash，不再证明已退役的宽松路径 |
| 人工筛选与 Tool 结果边界 | 独立复审 `PASS` | 人工 selection 原子绑定 gate inputs，不改写 screen Tool 输出/receipt；此前失败的真实 Modeling API E2E 已转绿 |
| 静态资源 immutable 缓存 | 独立复审 `PASS` | 版本键按排序相对路径与字节生成 SHA-256；保留 mtime 回归、完整 app security 与真实 Chromium 均通过 |
| 高置信死代码和资源重复 | 已完成本轮清理 | 仅删除可证明无消费者的私有代码与 9 份重复 manifest；pet 只保留一个 classic 入口，真实 Chromium 证明单次请求；动态入口和兼容面保留 |
| 最终僵尸/重复健康复核 | 独立复审 `PASS` | 定向 `61 passed`；生产私有函数无引用候选 0、同文件完全重复函数体 0；跨文件候选和 lazy-import SCC 只登记为非阻塞结构债 |

截至当前独立复审和最终全量门禁，没有已知仍开放的本地 Critical/Blocker。

## 四、仍然开放且不能由本地代码替代的门禁

| 门禁 | 当前状态 | 缺什么 | 为什么不能伪造关闭 |
|---|---|---|---|
| T4-2 公开数据参考门 | `BLOCKED_EXTERNAL_INPUT` | GiveMeSomeCredit、Home Credit 数据文件和可追溯人工 baseline | 合成 KS 或 LLM 生成 baseline 不能证明与人工基线等价 |
| T4-3 真实材料对账 | `NOT_PROVEN` | 当前真实任务、B1-B5 对账、业务/财务/独立验证/合规责任签字 | 代码测试不能替代业务地面真值和责任接受 |
| 真实 LLM 安全准入 | `FAILED / INCOMPLETE` | 当前 worktree 重新生成同 schema/corpus 的报告，并达到零 LLM error、完整 guardrail 和批准阈值 | 最新可用的两份 2026-08-01 报告均无可推荐 tier；它们不能证明当前 worktree 已通过 |
| 七入口同 SHA 浏览器验收 | `PARTIAL` | 同一 immutable SHA 上七入口的创建、上传、配置、门禁、执行、下载、刷新和重启恢复 | 当前完整浏览器证据集中在 Portfolio、Labeling 及部分既有旅程 |
| 原生 Windows/NTFS | `NOT_PROVEN` | 真实 Windows handle sharing、文件删除、任务树删除和失败恢复验收 | POSIX/macOS 测试与适配器单测不能证明 Windows 内核行为 |
| 当前代码远端 full CI | `NOT_PROVEN` | 收敛为可审查 immutable SHA，并由该 SHA 通过远端 full gate | 基础 SHA `9845e90` 的成功不覆盖当前 dirty worktree |
| 生产治理与运营 | `NOT_DELIVERED` | 企业身份、可信部署 verifier、生产决策服务、长期调度告警、值班和故障演练 | 本地对象和模拟 verifier 不是生产系统 |

## 五、当前验证状态

已有定向与独立证据包括 Strategy `5360 passed`、Presenter/下载/恢复组合回归、真实浏览器 CSS 与 Portfolio/Labeling 旅程、Decision Twin/C1 原始攻击重放、Ruff、JavaScript 语法和 diff whitespace 检查。最新定向关闭还覆盖 memory-aware driver 的 11 条真实保存旁路、SampleDesign V2 单一 selector/fresh shape、app controller context/DSL JSON/GC renderer 去重、Feature rollback 的 exact run/evidence/hash 生命周期，以及真实 Chromium 中 classic pet 资源只请求一次。详细命令与分组结果见[整改记录](2026-08-01-comprehensive-audit-remediation.md)和[最终本地复审](2026-08-01-final-local-closure-code-review.md)。

最新完整门禁曾在以下测试暴露测试时钟耦合：

```text
tests/test_dataset_source_gc.py::
test_gc_quarantines_malformed_timestamp_without_blocking_valid_due_entry
```

该问题的根因是队列入队时间与 collector 的固定测试时钟没有绑定到同一时点，导致全量运行时合法 due row 被误排除；精确同步测试时钟后，**22 个 GC 测试已通过**。这不是忽略失败，而是保留了可复现根因和最小修复。

后续全量门禁和独立复审还暴露并关闭了先前定向测试没有证明的边界：

- 历史 schema 测试只建目标表，却把残缺数据库标成完整 v1+；共享夹具现保留自 v1 已存在的 `plan_step_runs` 前置表，未修改或重排已发布 migration。覆盖全部 `PRAGMA user_version` 成功夹具的迁移/schema 组合为 `362 passed`，完整 `test_db.py` 为 `86 passed`。
- Modeling 的人工筛选曾生成没有 succeeded run 回执的 screen v2 输出，严格 parent binding 会使真实 API 旅程中断。现在人工选择只原子更新 gate inputs，Tool 输出保持不可变；PlanDriver `111 passed`，此前失败的真实 Modeling API E2E 已通过。
- special-value gate 测试曾伪造未登记的 `result_dataset`，从而掩盖 exact dataset binding；夹具改为真实 task-owned 数据集与真实 screen execution，整文件 `18 passed`，相关回归 `232 passed`。
- artifact identity guard 原来只扫描一组手写 façade 列表，存在新增消费者逃逸的假绿风险；现在对全部 `marvis/**/*.py` 做 AST 扫描并只允许 canonical owner，受影响组合 `138 passed`。
- Notebook worker 在源码 checkout 中依赖环境安装且可能被 notebook 目录同名包影子导入；现在使用 `sys.executable -I -c` 的可信 bootstrap，只注入仓库根并从隔离解释器导入 worker，non-slow Notebook 集为 `43 passed, 1 deselected`。
- Strategy migration 测试仍传入已移除的 `_TurnHandlerSpec.pass_memory_kwargs`，会鼓励恢复 runtime/memory 旁路；测试改为当前 runtime contract，相关 migration/memory 组合 `64 passed`。
- automatic-tree Parquet reader 在 schema/preflight 失败时可延迟释放 native descriptor；最终实现使用单一 exact-once 关闭边界与显式 `xb` 输出所有权，覆盖普通异常、`KeyboardInterrupt`、`close()` 失败、并发方抢先创建文件及 cleanup 失败不掩盖主异常。整文件 `26 passed`，automatic-tree/weighted-tree 扩大回归 `692 passed`，独立复审 `PASS`。

静态资源复审另发现最大 mtime 不能作为一年期 immutable cache key；改为内容 SHA-256 后，`test_app_security.py` 为 `38 passed`，真实 Chromium shell smoke 通过。

**最终全量门禁已通过。** 在上述最终代码状态执行：

```text
conda run --no-capture-output -n py_313 scripts/check --fast -- -q --maxfail=1
```

结果为 exit `0`：`11140 passed, 235 deselected, 32 warnings`，耗时 `4778.77s（1:19:38）`；同一命令中的 `git diff --check`、全仓 Ruff 与前端 JavaScript 语法检查均通过。32 条 warning 来自 LightGBM 参数弃用、固定预算 MLP 未收敛和 `sample_dataset()` 兼容弃用提示，均已显式保留为依赖/兼容债，而不是测试失败。

## 六、最终结论

1. 六个问题均已回答，并形成了实现、复审、能力矩阵和外部门禁的独立文档链。
2. 截至当前，没有已知仍开放的本地 Critical/Blocker；最新完整本地 fast gate 已退出 0，因此本地可操作整改正式关闭。
3. 高置信僵尸、重复事实源和关键竞态已定向清理；memory 保存旁路、SampleDesign V2 形状、共享前端/renderer seam、pet 入口和 Feature rollback 旧生命周期也已收敛；剩余主要是可管理的大模块认知债与兼容退役债。
4. Decision Twin 的可信基础已经建立，但不是生产功能完成证明。
5. Agent 已能让业务专家高杠杆地主导本地风控开发与分析，尚不能让一个人独立承担整条生产业务线的全部职责。
6. 项目整体结论保持 `NOT_PROVEN_FOR_PRODUCTION`，直到真实材料、真实 LLM、Windows、immutable SHA 远端 CI、生产运营和责任签字门分别关闭。
