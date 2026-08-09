# MARVIS 综合审计整改记录

> 日期：2026-08-01
>
> 审计基线：`9845e907c702f65edc77a9d0121f0967f52488c2`（V2.1.19）
>
> 整改对象：该基线之上的当前 dirty worktree
>
> 原始审计：[MARVIS 项目、Agent 能力与架构健康综合审计](2026-07-31-project-agent-architecture-comprehensive-audit.md)
>
> Presenter 信任复审：[Governed Tool Presenter Trust Review](2026-08-01-presenter-trust-code-review.md)
>
> 最终本地代码复审：[Final local closure](2026-08-01-final-local-closure-code-review.md)
>
> 分层状态真相源：[MARVIS 能力验收状态](../capability-status.md)

## 一、最终判定

本轮不是只“把审计文档改成已完成”，而是按审计优先级修实现、补对抗测试、做真实
浏览器旅程，并把无法由本地代码替代的外部证据继续保留为阻断门。

当前应使用两层判定：

- **本地整改边界：`LOCAL_FINDINGS_CLOSED_WITH_OPEN_ACCEPTANCE_GATES`。** P0 eval 证据、Portfolio、
  Labeling、authenticated I/O、生产治理与运营基础、Decision Twin、报告新鲜度、
  C1 内容寻址快照、exact step-result 回执、私有下载快照、恢复收敛和高置信清理均已落地；
  最终信任边界攻击已由独立复核重放通过。
- **项目整体：`NOT_PROVEN_FOR_PRODUCTION`。** T4-2、T4-3、可接受的真实 LLM
  基线、同一 immutable SHA 的远端 full CI、真实身份、真实部署 verifier、生产
  评分/决策、故障演练和责任签字仍没有完成证据。

换句话说：本轮显著提高了“本地风控开发与分析工作台”的完整性与可治理性，
没有把 MARVIS 证明成生产 Risk Operating System。

## 二、六个问题的整改后答案

| 原问题 | 2026-08-01 的答案 |
|---|---|
| 1. 现在最不自信的是什么？ | 仍是真实机构材料上的业务语义、人工基线等价性与真实 LLM 长链路稳定性。两个已配置模型在 2026-08-01 生成的最新可用 13-case 评测均已落盘，分别有 7 和 12 个 LLM error，三档均未保持 guardrail，系统正确拒绝推荐自主等级；当前 worktree 之后已有变化，仍需重新生成同源报告。代码和合成旅程能证明工程纵切，不能证明账务、标签、收益、公平性和模型风险结论会被业务、财务、独立验证与合规共同签字。 |
| 2. 你最大的盲点是什么？ | “后端有能力但普通用户不可达”的盲点已在 Portfolio 与 Labeling 上部分关闭；更大的盲点仍是把 implementation、unit、API、Agent、browser、real data、sign-off、production 合并成一个“完成”。能力矩阵现在强制分层，但外部地面真值仍为空。 |
| 3. 三个月后最可能为什么失败？ | 仍是证据收敛速度慢于能力扩张速度。最危险的形态不是系统完全不能跑，而是演示、模块和测试继续增加，首个真实试点却仍需开发者修口径、解释失败和人工托底，业务方因此不敢交付责任。 |
| 4. 最值得增加的行业领先功能是什么？ | 仍是“可治理的信用决策数字孪生”。本轮只完成了本地 CAS、认证重放、Champion/Challenger/反事实比较和 proposal-only 桥的基础；API、UI、外部签名、真实结果回流、shadow 与生产晋级闭环尚未交付。 |
| 5. 有僵尸代码和屎山吗？ | 有局部维护重力，但不是全仓不可维护。两批高置信私有死代码、重复审计、`$ref` 解析、表格安全、artifact id、authenticated snapshot 与下载快照 seam 已收敛；44/44 Strategy family 已迁入 canonical spec/validator/confirmation/preparer，8 个 canonical result presenter 已统一；`memory.before_save` 已真实分派；9 份重复 `pet.json` 已由单一 catalog 替代；数据集与任务目录已有持久化 GC/重试及共享 Windows handle 适配器。剩余问题主要是 `strategy_request_compiler.py`、`turn_handlers.py` 和若干 report/tool 模块仍过大，应继续按变化理由渐进拆分，不能只看行数批量删除。 |
| 6. Agent 已能覆盖信贷风控全流程吗？ | 本地文件驱动的开发/分析主链已更完整，新增 Labeling 与 Portfolio 正式旅程；持续生产经营仍未闭合，且两份最新可用真实模型评测都没有可推荐 tier，当前 worktree 也尚无新报告。业务专家可成为一条业务线的主要 owner 和高杠杆操作人，但不能同时替代独立验证、审批、合规、生产、值班和审计责任。 |

## 三、已完成的整改

### 3.1 P0：先修证据，而不是先拆大文件

1. 修复 Planner/eval 对伪造指标的恒假检测；确定性 `PlanValidator` 与 eval 共用约束。
2. 通用 planning failure 不再冒充指定 guardrail 命中；空 case 不再得到虚假满分。
3. touchpoint 账本覆盖 planner、gate、router、reviewer 的退化输入与否定理由冲突。
4. 初始 offline corpus 从 7 个扩展到 13 个，覆盖六类入口及歧义、否定、缺材料、
  越权、伪造指标、多轮、stale evidence 和长上下文。
5. 真实模型 eval 的单个 `LLMClientError` 现在记录为 case-level 失败并继续后续 case，
   不再丢弃此前结果或阻止 JSON 报告落盘；报告按 tier 保存每个 case 的 pass、metrics、
   final status、transcript ref 与脱敏 error kind。
6. eval 报告新增 schema、corpus hash/case ids、18 份提示词版本快照、错误计数和
   `INCOMPLETE` fail-closed 判定；任一 LLM error 都禁止推荐 tier，expected-failure case
   不得混入通过率，历史 baseline 只有完整同源 schema 才能比较且永不覆盖。
7. 新建 [capability-status.md](../capability-status.md)，将路线范围与验收状态拆开；
   master backlog 降级为历史实施追踪，不再充当产品完成证明。

### 3.2 产品可达性：Portfolio 与 Labeling

#### Portfolio

- 进入正式 task type、HTTP Agent allowlist、首屏入口和 workflow 模板。
- 计划前由人确认数据集版本、余额/EAD、分群、状态顺序、坏状态、LGD 与期限。
- 对输入使用 content hash 和 authenticated snapshot；执行前再次认证。
- 支持组合分析报告登记和受控下载；下载只接受任务拥有且哈希匹配的报告。
- 已完成 real-process HTTP + Playwright 合成 `no-trend` 旅程；当前机器的 ignored
  报告工件为 `output/playwright/portfolio-browser-workspace/tasks/ee273135ef64409ea58b5344e90ff608/portfolio/portfolio_report_56f21a692ac5ee9e.xlsx`，
  SHA-256 为 `56f21a692ac5ee9ef6a619b8581a943e3e8bdf6cd1d155e3951f4f456fdfb6fd`。
- 已证明的是合成单期组合；真实机构组合、跨期 trend、财务对账和生产调度未证明。

#### Labeling

- 作为数据处理任务中的受治理 workflow 接入，不新增顶层 task type。
- 正式对话覆盖字段/窗口/坏定义提案、标签定义确认、成熟度确认、执行、派生数据集
  登记与下载；结果不会静默替换 active dataset。
- proposal、maturity 与 define 均从同一次 authenticated private frame 计算，提交事务再次
  认证 dataset/workspace 绑定；已有 swap/restore、path drift 和 workspace drift 对抗测试。
- 已完成 real-process HTTP + Playwright 合成旅程：输入 16 行、4 个成熟账户，输出列
  `loan_id/cohort/bad_m3`，4 行中 `bad_m3=1` 为 2 行；浏览器 console 为 0 error。
- 当前机器的 ignored 证据截图为
  `output/playwright/evidence/labeling-browser-completion-2026-08-01.png`，
  SHA-256 为 `814786925dc21f59e13ee87cd4ba7fba2a50d5cce61d0cc677544caac4fd1e8d`。

### 3.3 认证 I/O、审计与编排 seam

- 五类 Parquet 读取迁移到 authenticated snapshot；读前、持有描述符读取和读后均核对
  身份与 SHA-256，避免 path verify 与实际读取分离。
- `AuditWriter` 生产实现从 6 份收敛为 1 份，保留兼容 alias 与事务回滚语义。
- 表格公式注入清洗和 governed JSON 解析归到共享 seam。
- 三处 `$ref` 解析统一为 `marvis.orchestrator.references`，validator、executor 与治理层
  保留各自错误类型。
- `TaskArtifact` stable id 的重复实现统一到公开 helper。
- `db_schema → agent_memory.store → marvis.db` 反向环被切断；Strategy 包改为 lazy export，
  fresh DB 初始化加载的 Strategy 模块由 34 个降为 0，同时保留既有导出身份。
- 单表 data-join 的人工确认现在会把已确认 source/version/target 绑定到
  `DataWorkspace`，不再停留在只有聊天状态的假完成。

### 3.4 Strategy、模型分数比较与架构收敛

- 建立 canonical `StrategyWorkflowSpec`/catalog，统一承载 workflow identity、exposure、
  requirements 和 template identity；迁移完成的 family 还统一承载 validator、confirmation
  与 plan preparer。
- 当前 44/44 Strategy spec 均已标记为 `migrated`，并统一具备 validator、confirmation
  workflow 与 plan preparer；compiler 侧静态 shadow validator/confirmation 分支均为 0，
  canonical turn entry 保持单一入口。8 类 governed result 也进入统一 presenter registry。
  这关闭了重复事实源，但没有自动消除 compiler/turn handler 的体积与认知复杂度。
- 多模型分数比较完成 tool、Agent、API 的受治理纵切：认证最新 SampleDesign 与模型分数证据，
  至少两个模型才计算，只输出比较证据，不自动选冠军、采纳或部署。
- 第一、二批只删除有高置信静态与动态证据的私有死代码；第二批另删除 9 个私有函数、
  1 个无消费者正则和 1 个无用 import，共减少 114 行生产代码。
- 没有删除可能由 Plugin、CLI、字符串入口、打包资源或兼容面动态使用的候选。

### 3.5 生产治理、运营与信用决策数字孪生基础

- 生产治理提供本地 maker/checker/admin、promotion、逐环境 deployment、activation、rollback
  和 immutable audit 的基础对象。
- deployment manifest 由服务端生成；activation 默认无可信 verifier 时 fail closed，不能由
  请求体自报“部署成功/健康”形成生产事实。
- 运营基础提供持久化 run、scheduler、notification、retry 与 API；它尚未绑定真实生产执行器、
  长期 cadence、值班和故障演练。
- Decision Twin 提供本地 content-addressed audit store、冻结 manifest/facts/字段来源/adapter
  材料、point-in-time replay、Champion/Challenger/反事实联合约束，以及 FSP-8 proposal-only
  桥；不允许它直接采纳、发布或改变生产状态。

## 四、独立代码复审与对抗整改

深度复审阶段先后累计发现 5 个 Critical、10 个 Warning、1 个 Info。此后的完整
门禁与最终资源生命周期复审又暴露出历史 schema、测试证据真实性、全树静态 guard、
隔离 worker 启动和 descriptor/output ownership 等缺口。普通 happy-path 测试通过并不
代表这些信任边界安全，因此均以可复现的真实执行、恶意/并发输入或全树扫描作为关闭条件。

| 编号 | 问题 | 当前状态 | 关闭证据 |
|---|---|---|---|
| CR-01 | Portfolio 确认未绑定 immutable dataset | 已修复 | gate 带 dataset/hash；执行认证 snapshot；race/drift 测试 |
| CR-02 | Labeling 在 verify 与 read 间可 swap/restore | 已修复 | proposal/maturity/define 共用认证 frame；对抗测试不产生派生数据或审计 |
| CR-03 | 生产 activation 接受调用方自报 manifest/健康证据 | 已修复 | 服务端 manifest、allowlisted verifier、事务内重认证；默认 fail closed |
| WR-01 | queued→running 失败泄漏 active job | 已修复 | typed conflict 且 rejected claim 被同步终结 |
| WR-02 | Decision Twin 接受自报 hash/adapter | 已修复并独立复核 PASS | 生产入口只从 CAS 与冻结 allowlist 解析；更深的 data/object/transitive-code binding 见下两项 |
| FR-CR-01 | Strategy DSL 内容变化但缓存 hash 不变仍可激活 | 已修复并独立复核 PASS | 事务内从 canonical DSL 重算 hash，同时核对 cache、审批快照与 manifest；双改 DSL/cache 仍 409 |
| FR-WR-01 | replay record/facts 未绑定 `manifest.data` | 已修复并独立复核 PASS | 新 `replay_data_binding.v1` CAS 物料精确绑定 record 与 facts；跨 manifest/record/facts 错配失败关闭 |
| FR-WR-02 | allowlist 字节 hash 未绑定实际执行 Python object | 已修复并独立复核 PASS | 只注册可信 exact adapter type、内部实例化，执行前后复核 type/identity/implementation；伪装和 monkeypatch 均拒绝 |
| FR-WR-02b | adapter 可经 global/helper/closure 间接执行未冻结代码 | 已修复并独立攻击 PASS | 递归指纹并冻结函数、defaults、closure 与可达 globals；module namespace 和不支持对象失败关闭；恶意替换 helper `__code__` 在执行前被拒绝，副作用计数保持 0 |
| FR-WR-03 | 单表 source path 认证与 workspace 提交之间仍可竞态 | 已修复并独立攻击 PASS | 认证 bytes 先原子发布到 `_cas/<sha>/<sha>.parquet`，dataset row CAS 更新后 workspace 才提交；原源在最后认证后变化不影响绑定快照，覆写 CAS 失败且 revision 保持 0 |
| FR-WR-03b | 删除只读 CAS 会留僵尸，且并发 pin 可形成 DB→missing-file | 已修复并独立复核 PASS | 最后引用 GC 在 `BEGIN IMMEDIATE` 内重查引用并持锁清理；仓储 pin 强制要求认证 callback，并在同一写锁内再认证 CAS，GC 先行时有限重建；两种锁序、共享首删/末删、瞬时/持续锁失败均有回归测试，post-commit cleanup 不再返回 500 |
| FR-WR-04 | 单表重复确认忽略 target 语义变化 | 已修复并独立复核 PASS | 只有完整 `DataSemanticMapping` 与 C1 映射一致才允许幂等；target/business name 漂移拒绝 |
| FR-WR-05 | 最新报告损坏/未完成时静默回退旧 plan | 已修复并独立攻击 PASS | 任一更新的 report-producing attempt 都是 freshness barrier；最新为 failed/pending/running/skipped 或认证失败时返回空/404，不再泄露旧报告 |
| FR-WR-06 | Portfolio 列结构从路径二读产生 TOCTOU | 已修复并独立复核 PASS | schema 与 bucket states 从同一次 authenticated snapshot frame 推导 |
| FR-I-01 | verifier `verified_at` 只校验非空 | 已修复并独立复核 PASS | 仅接受合法日历时间的严格 UTC ISO-8601 `Z/+00:00`；新鲜度 SLA 仍由 verifier 负责 |
| FR-CR-02 | Agent normal 草稿门可被通用 `POST /report` 绕过 | 已修复并独立复核 PASS | generic report job、preview 与 download 恢复 confirmed-conclusions gate；失败同步终结 active job；normal/AUTO/V2 正反路径均通过 |
| PR-CR-01 | 同任务另一份合法 canonical envelope 可替换当前步骤展示 | 已修复并独立复核 PASS | ToolRunner 在成功前认证实时领域对象与 resolved inputs，并签出 run/output/tool-version/manifest 回执；Repository 原子绑定不可变 output version，Presenter 只接受 exact presentation binding |
| PR-CR-02 | result dataset 的旧消息、旧 plan 或下载路径可漂移 | 已修复并独立复核 PASS | message 按自身 plan 重建；URL 绑定 plan/step/output/hash；下载重载 exact evidence，并从单 descriptor 复制校验到私有快照后响应；registry+文件一致替换和 verify-to-open 竞态均返回 409；快照上的 Range/206/416 语义已恢复 |
| PR-CR-03 | 人工筛选通过无 run receipt 的新 output version 覆写已完成 Tool 结果，导致 strict parent binding 中断 Modeling API 旅程 | 已修复并独立复核 PASS | selection 经过原 screen evidence 约束后，与 gate confirmation 在同一事务中写入 concrete inputs；原 output ref/hash/succeeded run receipt 不变，真实 Modeling E2E 转绿 |
| FE-P2-01 | 静态缓存版本只取最大 mtime，保留时间戳的旧文件内容变化不会刷新一年期 immutable URL | 已修复并独立复核 PASS | 对全部 JS/CSS 的排序相对路径和字节生成 SHA-256；改名、增删和内容变化均换 key，保留 mtime 回归通过 |
| RR-BL-01 | 父输出绑定损坏时恢复抛出 `ConflictError`，计划永久停在 RUNNING | 已修复并独立复核 PASS | recovery 将完整性冲突收敛为 step/plan FAILED；direct 与 startup reclaim 两条回归均通过 |
| FG-01 | 历史 schema 夹具把缺失 v1 前置表的残缺数据库标成完整版本 | 已修复并独立复核 PASS | 单一共享 fixture 安装 `plan_step_runs` predecessor；覆盖全部成功夹具的 schema/migration 组合 `362 passed`；已发布 migration 未修改或重排 |
| FG-02 | special-value gate 夹具伪造未登记的结果数据集和 screen output | 已修复并独立复核 PASS | task-owned exact result dataset + 真实 screen executor/receipt；整文件 `18 passed`，相关组合 `232 passed` |
| FG-03 | artifact identity guard 只扫描手写 façade 列表，可对新增消费者假绿 | 已修复并独立复核 PASS | 对全部 `marvis/**/*.py` 做 AST 扫描；私有 alias/attribute/import 与 namespace literal 只允许 canonical owner；相关组合 `138 passed` |
| FG-04 | Notebook worker 在 source checkout 中依赖环境安装，并可能被 notebook 目录同名 `marvis` 影子导入 | 已修复并独立攻击 PASS | `sys.executable -I -c` 可信 bootstrap 只注入仓库根；真实 shadow 攻击被拒绝；non-slow Notebook 集 `43 passed, 1 deselected` |
| FG-05 | Strategy migration 测试仍传入已移除的 memory 绕过开关 | 已修复并独立复核 PASS | 删除 stale `pass_memory_kwargs=False`，按当前 runtime contract 构造；migration/memory 组合 `64 passed` |
| FG-06 | automatic-tree schema/preflight 异常可延迟释放 Parquet reader；初版关闭修复又会误删竞争方输出或让 cleanup error 掩盖主异常 | 已修复并独立复核 PASS | reader 单一 exact-once finally；显式 `xb` 输出所有权；覆盖 `KeyboardInterrupt`、close failure、竞争创建和 cleanup failure；整文件 `26 passed`，扩大回归 `692 passed` |

最终独立复核重新执行 Decision Twin helper 替换、C1 source/CAS race 和最新报告
failed/running 回退攻击，三组均 PASS；原始攻击集 9 条测试通过。C1 随后的 pin/GC
正确性复核又覆盖 7 个定向场景和完整 task-purge 文件，结论 PASS；对应 Ruff/diff
检查通过。
这关闭的是本地实现边界，不等于关闭真实数据、生产运营或业务签字门。

## 五、架构健康与残余债

本轮应描述为“把最危险的重复信任边界和明确僵尸先收敛”，不是一次大爆炸重构。

本轮已关闭的结构项：44/44 Strategy canonical 迁移、8 个 presenter、`memory.before_save`
分派、pet catalog 单一事实源、9 份重复 pet manifest、renderer/候选实验室/CSS 的第一轮
拆分、持久化文件 GC 与共享 Windows handle adapter、artifact identity 全树 guard、隔离的
Notebook worker bootstrap，以及 automatic-tree reader/output 单一所有权边界。删除的 pet
manifest 在提交前仍可由 Git 恢复；没有执行其它广泛物理删除。

仍需保留在 backlog 的结构债：

1. `strategy_request_compiler.py`（约 1.45 万行）、`turn_handlers.py`（约 1.55 万行）、
   `report_bundle_adapters.py`、`static/app.js` 和若干 Strategy tool 模块仍是高认知负担；
   应继续按 presenter、request compiler、domain service 的变化理由拆分。
2. canonical catalog 已消除 Strategy family 的 schema/validator/confirmation 重复事实源，
   但迁移后的兼容 facade 与超大编排文件仍需在后续发布周期内渐进退役，不能立刻删除。
3. `memory.before_save` 已接线并有顺序、拒绝和审计测试；后续债是建立生产级 retention、
   脱敏策略演练和真实机构数据治理验收，而不是再增加 hook 名称。
4. pet 资源现在由 `static/js/pet_catalog.js` 统一；后续只需保持打包/缓存测试，不再恢复
   每个目录一份重复 JSON 的事实源。
5. 数据集 CAS 与任务目录删除已有持久化 GC 队列、退避重试、管理员重入队和共享 Windows
   handle adapter；真实 Windows/NTFS sharing violation 的原生验收仍是外部环境门。
6. `sample_dataset()` 已进入一个兼容发布周期的弃用期；head/random 有明确替代，
   stratified 尚无受支持生产替代，因而当前不能把这层兼容代码当普通僵尸直接删除。
7. 当前工作区共有大量用户与整改并行变化；必须先收敛边界、绑定新 SHA，再做远端 full CI，
   不能把 dirty worktree 的本地测试描述成可发布工件。
8. 最终独立健康扫描未发现新的高置信生产私有僵尸或同文件大段完全重复；跨文件仍有
   55 组完全相同函数体候选，加入 lazy import 后仍有 6 个 SCC（最大 41 模块）。这些属于
   需要先明确共享模块所有权与兼容边界的 P2/P3 收敛债，不是可以凭扫描结果直接删除的代码。

## 六、验证记录

### 6.1 已通过

| 层 | 命令或行为 | 结果 |
|---|---|---|
| 本地 fast 总门禁 | `conda run --no-capture-output -n py_313 scripts/check --fast -- -q --maxfail=1` | exit `0`；`11140 passed, 235 deselected, 32 warnings`，耗时 `4778.77s（1:19:38）`；同一命令内 diff check、全仓 Ruff、前端 JS 语法均通过 |
| Strategy 全量 | `pytest -q tests/test_strategy*.py` | `5360 passed`，exit `0`，耗时 `1:14:57`；44/44 canonical migration |
| Presenter / 下载 / 恢复闭环 | PlanDriver、canonical presenters、Labeling、data、artifact、recovery 定向组合 | `520 + 303 + 54 + 215 passed`；精确回执、下载竞态与恢复冲突复审最终 PASS |
| CSS 拆分真实浏览器 | 1920×1080 Chromium，禁缓存硬刷新和策略任务壳交互 | 15/15 stylesheet HTTP 200；console/page error 0；`327 passed` |
| Decision Twin 全套 | `pytest tests/test_decision_twin*.py` | `46 passed`；恶意 transitive helper 在执行前被拒绝 |
| C1 / JOIN / 任务清理 | 11 个 authenticated snapshot、workspace、JOIN、purge 相关测试文件 | `121 passed`；含共享 CAS、GC/pin 两种并发锁序和 cleanup lock 重试/降级 |
| 报告新鲜度 / Portfolio | report helper、driver report 与 Portfolio 定向集 | `25 passed`；latest failed/running 攻击返回 404 |
| Strategy 受影响边界 | DSL activation、normal report gate 与相关策略回归 | `50 passed` |
| Eval runner/score | 本地核心 eval 集；独立 provenance/baseline 复核集 | `112 passed`；独立合并复核 `140 passed` |
| Strategy compiler 提示词锁 | 15 个 request-compiler 测试文件 | `1026 passed`；统一引用 `STRATEGY_REQUEST_COMPILER_SYS.version=54` |
| 独立最终攻击复核 | Decision Twin、C1、report freshness 三个原始攻击 harness | `9 passed`，对应 Ruff/diff PASS |
| 独立 C1 GC 复核 | pin-first、GC-first、共享引用、强制认证、持续锁失败等场景 | `7 passed`；当前完整 `test_task_purge.py` 为 `20 passed` |
| 最终僵尸/重复健康复核 | memory 保存路径、Strategy 语义、app/DSL/GC/pet 重复及静态探针 | `61 passed`；相关 Ruff 与 JS 语法 PASS；高置信私有僵尸 0、同文件完全重复函数体 0；结论 PASS |
| 人工筛选 exact binding | PlanDriver selection/stale/governance 与真实 Modeling API E2E | screen Tool output/receipt 保持不变；selection 原子绑定 gate inputs；PlanDriver `111 passed`，原失败 E2E 转绿；独立复审 PASS |
| 静态 immutable cache key | preserved-mtime 回归、完整 app security、真实 Chromium shell | 内容 SHA-256 版本键；`38 passed`；Chromium `1 passed`；独立复审 PASS |
| 历史 schema fixture | 全部 `PRAGMA user_version` 成功夹具与 migration 31-33 | 共享 v1 predecessor；`362 passed`；已发布 migration 保持 append-only |
| special-value exact binding | 真实 screen execution、task-owned result dataset 与相关 driver/frontend/template | `18 passed`；扩大组合 `232 passed` |
| artifact identity 全树 guard | `marvis/**/*.py` AST 消费者/私有 alias/namespace 扫描与受影响回归 | `138 passed`；canonical owner 之外无私有 identity 定义 |
| Notebook worker 隔离启动 | source checkout discovery、真实 shadow attack 与 non-slow Notebook 集 | `43 passed, 1 deselected`；`-I` bootstrap 拒绝 notebook 目录影子包 |
| Strategy migration/runtime contract | sample plan migration 与 V2 memory 组合 | `64 passed`；stale memory 绕过开关已删除 |
| automatic-tree reader/output ownership | success/schema/execution/interrupt/close/竞争创建/cleanup denial | 整文件 `26 passed`；扩大 automatic-tree/weighted-tree `692 passed`；独立复审 PASS |
| 前端测试 | 前端 Python/Node 测试集合 | `525 passed, 3 skipped` |
| Word 草稿/报告确认回归 | `test_agent_api.py -k 'word_conclusion or regenerate_report or continue_generate_report'` | `23 passed, 116 deselected` |
| Agent polling | `tests/test_agent_poll_backoff_frontend.py` | `3 passed` |
| API v2 | `tests/test_api_v2.py` | `94 passed` |
| Python 静态检查 | `python -m ruff check marvis tests --extend-exclude '*.ipynb'` | PASS |
| JavaScript 语法 | `node --check marvis/static/app.js` | PASS |
| diff whitespace | `git diff --check` | PASS |
| Labeling 浏览器 | 创建、绑定、配置、双确认、执行、下载、刷新；console 检查 | PASS，0 console error |
| Portfolio 浏览器 | 创建、配置、人工门、执行、报告登记、受控下载 | PASS，报告 SHA 匹配 |

fast gate 的 32 条 warning 来自 LightGBM 参数弃用、测试用 MLP 在固定迭代预算内未收敛，
以及上述 `sample_dataset()` 兼容弃用提示；它们未转化为测试失败，但已作为依赖升级和
兼容清理债保留，不能在结果中静默忽略。

### 6.2 尚未通过或不能由本地代码替代

| 门 | 当前证据 | 判定 |
|---|---|---|
| 真实 LLM eval（默认模型） | 2026-08-01 最新可用报告：13 cases、18 个提示词版本快照、7 个 LLM error；autonomous/balanced/conservative pass rate 为 `0.46/0.38/0.38`，guardrail pass rate 为 `0.67/0.33/0.33`，三档 `guardrail_intact=false`；当前 worktree 尚未重跑 | `INCOMPLETE`，`recommended_tier=None` |
| 真实 LLM eval（第二模型） | 同一 corpus/prompt 报告：12 个 LLM error；autonomous/balanced/conservative pass rate 为 `0.31/0.38/0.38`，三档 guardrail pass rate 均为 `0.33` 且 `guardrail_intact=false` | `INCOMPLETE`，`recommended_tier=None` |
| T4-2 | `scripts/ks_baseline.py --status` 退出 2；两个公开数据文件与人工 baseline 均缺失 | BLOCKED_EXTERNAL_INPUT |
| T4-3 | 无当前真实任务 id、B1-B5 外部对账和责任签字 | NOT_PROVEN |
| 当前工作树远端 full CI | 远端成功只绑定基础 SHA `9845e90`；当前仍为 dirty worktree | NOT_PROVEN |
| 生产验证 | 无真实身份源、部署 verifier、评分/决策服务、长期调度告警、值班和故障演练 | NOT_DELIVERED |

### 6.3 T4-2 当前机器证据

2026-08-01 运行：

```bash
conda run --no-capture-output -n py_313 python scripts/ks_baseline.py --status
```

退出码为 `2`；GiveMeSomeCredit 与 Home Credit 均为：

- `file_present=false`
- `baseline_recorded=false`
- 缺 `public_dataset_file`
- 缺 `human_tuned_baseline`

这不是代码失败，也不能通过生成合成 KS 或让 LLM 编一个 baseline 来关闭。

### 6.4 真实 LLM 当前机器证据

两次有效评测均从当前工作树运行 `python -m marvis`，共同 provenance 为：

- `schema_version=marvis.eval.report.v1`
- `corpus_version=sha256:07f5a32b95ca8b02563a7ca7f6bfcadaa763e37fc85073aba6cd263544e64533`
- `case_ids=13`，`prompt_version_snapshot=18`，其中
  `STRATEGY_REQUEST_COMPILER_SYS=54`

默认模型的当前机器 ignored 报告：

- 路径：`workspace/eval/model-452c506756-20260801T060256668245Z-cba3bff87b90.json`
- `run_id=cba3bff87b9047a38b4e3fd06826fad1`
- SHA-256：`28c26b54311792014ab21eed001e3ea7794b91d4fc5e002183f03a4ae9e8b15f`
- `status=INCOMPLETE`，`error_count=7`，`recommended_tier=null`

第二模型的当前机器 ignored 报告：

- 路径：`workspace/eval/model-55a028c09d-20260801T073307723043Z-c4e1fed454b4.json`
- `run_id=c4e1fed454b44a20a5b0f135cf95dac7`
- SHA-256：`3ab17be6932ea710e22582260f71246034a12b151bbb069c2b87f86a2cfba3ed`
- `status=INCOMPLETE`，`error_count=12`，`recommended_tier=null`

环境中的旧 `marvis` console entrypoint 曾生成一个 7-case、无 schema/corpus/prompt
快照的日级文件，并错误显示可推荐 autonomous；它不是当前工作树结果，已明确排除，
也没有用作 baseline。这个实测说明本地验收必须使用
`conda run --no-capture-output -n py_313 python -m marvis ...`，并校验报告 provenance，
不能只相信同名 console 命令。

两份有效报告成功落盘证明了 case-level 错误隔离、provenance 与 fail-closed gate；
它们证明这两个模型在 2026-08-01 的最新可用运行中都没有通过自主级别准入；当前
worktree 后续已有变化且尚无新同源报告，因此状态只能保持失败关闭。旧报告缺少同源
schema/corpus/prompt 快照，不能作为可比较 baseline，系统没有覆盖或伪造历史基线。

## 七、一个业务专家现在能做到什么

在本地文件、已确认口径和关键门禁有人负责的前提下，一个懂业务且具备基础数据能力的
用户，现在可以主要依靠 MARVIS 完成：

- 数据检查、JOIN、质量诊断与受治理的标签构造；
- 特征分析、筛选、分箱、建模、比较、验证与交付材料；
- 规则/评分/树/Voting 策略的本地开发、回测和受控采纳；
- Vintage、迁徙、流率、利润和 Portfolio 分析；
- 生成结构化证据、审计记录与报告，并在口径、样本、策略和报告结论处保留人工门。

仍不能据此让一个人独自承担：

- 生产数据源、延迟、回补、权限与质量责任；
- 实时/批量评分决策、发布、shadow、回滚与故障恢复；
- 独立模型验证、maker-checker、合规、公平性、拒绝原因、申诉和审计；
- 监控告警、值班、财务对账、实际风险收益结果和最终责任签字；
- 催收、反欺诈、征信/三方报文的完整专业系统职责。

因此当前产品定位不变：

> **本地优先、可治理的信贷风控开发与分析工作台；让一个业务专家成为小型风控团队的高杠杆核心。**

## 八、三个月内必须观察的预警指标

如果以下现象持续，项目最可能按原审计预测失败：

1. 新增 workflow/页面数量继续增长，但 T4-2、T4-3 仍为空，真实 LLM 仍无可推荐 tier
   与可比较通过基线；
2. 浏览器验收只覆盖合成 happy path，没有重启恢复、失败注入和真实材料；
3. Strategy canonical 迁移虽已 44/44 完成，但兼容 façade 长期不退役、编排大模块继续同步膨胀；
4. 生产治理对象增加，但仍没有真实 identity/verifier/decision service 消费它们；
5. 业务试点仍依赖开发者逐次修数据、解释口径和手工改数据库；
6. “测试数量”被用来替代 business sign-off、财务对账和独立有效质疑。

真正的 90 天成功标准不是功能清单继续变长，而是至少一条真实业务线按预登记协议完成
多周期试点，人工介入、对账差异、故障恢复、例外、回滚和责任人均可追溯。
