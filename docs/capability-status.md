# MARVIS 能力验收状态

路线范围见 [roadmap.md](roadmap.md)。V3 软件与验收边界核对日期：2026-11-04。

- **升级基线**：`V2.7.1` / `a8bf7a6f`。历史发布记录见 [V2.7.1](releases/V2.7.1.md)。
- **V3 软件交付范围**：`V3.0.0` 工作台改版的软件与本地验证已完成，本次交付范围包含分支整合、`main` 同步和公开软件发布。用户可见变化与兼容边界见 [V3.0.0 发布说明](releases/V3.0.0.md)。
- **发布证据单独核对**：tag、发行文件及 SHA-256 以 [GitHub Releases](https://github.com/eddyzzl/marvis-risk-agent/releases) 为准；远端验证查看 [GitHub Actions](https://github.com/eddyzzl/marvis-risk-agent/actions/workflows/ci.yml) 中与发布 tag 同一提交的运行。本文不由本地结果推定远端 CI 已成功。
- **外部验收仍独立**：软件公开发布不自动关闭正式软件验收 S、历史业务效果 H 或机构生产 P。真实模型付费运行、脱敏历史数据、批准的性能口径、正式人工验收与机构生产接入分别保留外部条件。
- **本地软件收尾记录**：2026-11-04 的 17,442 个测试身份去重闭合为 17,434 通过、8 项真实模型测试按范围跳过；初始失败和记录器中断原件保留，由完整续跑和修正文件回归闭合。静态检查通过；这不是正式 S 验收或单次全量命令零退出的声明。
- 以下矩阵描述 V3 软件范围与现有证据。源码、工具产物、本地浏览器、发布 CI、正式软件验收、历史业务效果和机构生产各自判定；下方带日期的历史记录保持原样。

## 正式验收

正式软件 S、历史业务 H、机构生产 P 均未由本轮建立。

| 层 | 状态 | 当前事实与待办 |
| --- | --- | --- |
| 软件正式验收 S | 待外部批准与复核 | 隐藏场景清单已冻结为 133 个案例模板、193 个执行槽位，已跑协议替身 dry run；独立持有的实际材料/预期答案、真实模型配置、费用上限、批准的硬件/负载/SLO 和独立签字仍需分别冻结，不能把模板或 dry run 当作正式运行。54 项手动验收表已生成，签字仍为空。 |
| 历史业务效果 H | 待外部数据 | 旧历史比较未达到“新模型更好”门槛；需要新的独立 OOT 期间、标签可得时间及脱敏历史原件。旧材料不恢复为未暴露的盲测。 |
| 机构生产 P | 待机构接入与批准 | 缺真实接口、身份源、批准流量、SLO、业务负责人和成熟结果。参考执行及本机测试不替代生产证据。 |

审计 finding 的本地证据索引与独立复核记录分别维护；没有外部信任配置、审核签字的条目保持未关闭。完整运行结果见 [本轮验证记录](reviews/2026-11-04-v3-local-validation.md)。

## V3 软件能力与边界

| 范围 | 软件和本地验证 | 保留边界 |
| --- | --- | --- |
| 工作台与语言 | 工作台改版、中文/English 切换、任务状态/错误/下一步、长标题、主题、监控表单和冗余副标题收尾已实现；桌面 1440/1600/1920 浏览器检查记录独立保留。 | 用户名称、文件名、原始字段与数值保持原样；Agent 叙事不按界面语言强制翻译。 |
| 出站与规划 | 全部 `.complete(` 入口经结构化适配器，规划/业务复核保留原目标，策略编译与搜索敏感检查接通；零引用旧提示函数已清理。 | 没有用本地协议替身证明真实模型表现或出站业务接受率。 |
| 诊断和恢复 | V1/V2 诊断、明确确认、预算、任务环境覆盖、回滚和批次处理记录已接通。 | 安装包/Java 仍只提供说明；改标签、阈值、删行等动作要求明确选择并保留审计。 |
| macOS 原生执行 | 读不了设置时失败关闭；loopback API 令牌、AF_UNIX、系统解析器、子进程检查及 Agent V1/双模型 V2 旅程有本地测试；CI 增加独立 macOS 检查且跳过不能算通过。 | V1 Notebook 由原生 worker 运行；标准 V2 静态读取 Notebook、PMML 在宿主 JVM 评分，不能据此声称整个服务受 Notebook 网络策略隔离。远端执行结果见与发布 tag 同一提交的 [GitHub Actions](https://github.com/eddyzzl/marvis-risk-agent/actions/workflows/ci.yml)。 |
| Windows / Linux 隔离 | 不可用状态和支持边界明确保留。 | 需要对应设备/backend 原生验证。 |
| 盲 OOT 与时间事实 | 冻结、一次评估与幂等读回、隔离读取、标签可得时间/历史参数/候选来源采集及结论依赖门控已实现。 | 未知事实不自动补真；同 UID 恶意进程、root 或 trusted_local 宿主不在强隔离保证内。 |
| 验证独立复算 | 标准 V2 原始材料、评分、指标、报告确认和下载绑定可重检；受支持 PMML 子集的独立评分与 KS/AUC/PSI/lift 等参考计算已接入。 | 压力类别缺失值语义仍依赖 PMML 引擎复现；不称任意 PMML、报告叙事和业务标签均已独立证明。 |
| 特征独立复算 | PSI 多视角、VIF、重要性、非空分箱、头尾 lift、字典含义一致性和真实材料上传已接入。 | 重要性训练依赖同类第三方 learner，明确标注 library_dependent；不支持的输入保留 unavailable。 |
| 组合与 Vintage | 组合 score PSI 趋势与冻结基线、报表原件及标准 Vintage 复算接通正式分发；显式剔除缺失标签后的重试记录保留。 | 基线训练真实性、数据可得时间及任意自定义趋势计划不是算法复算能证明的事实。 |
| 建模独立复算 | 校准、集成、CatBoost、MLP、评分卡、权重、提前停止及测试集参与模型选择的语义已扩展；保留受限反序列化。 | 外部预处理、类别 CatBoost、非二元加权指标等超出安装参考范围时失败关闭；第三方 learner 拟合可复现不等于独立证明拟合最优。 |
| 边界和恢复证据 | 原生澄清、拒绝、过期 CAS、审批重放及原清单 16 个故障槽均有本地执行和独立原件核对；建模恢复后完整选择/报告/交付和验证报告丢响应均有专门路径。 | 只能证明绑定到原件的声明场景；机构授权、实际外部业务效果和缺失的历史原件不会由测试补齐。 |
| 数据、SQL 与字段语义 | SQL 对话提案复用受治理确认；字段别名进入建模/特征；历史材料映射扩至数据处理。 | SQL 白名单、来源权限、行数与超时限制保持；原始值不翻译。 |
| Strategy Workbench | 月度样本、分数方向、双人群、额度/定价/分群、回放、报告和交付链路保留；有价值的孤立诊断接为 Tool。 | 策略输出和历史比较不承诺真实经营效果。 |
| 监控、反欺诈与催收 | 对话调度业务表单和确认、数据集切换字段刷新、入口意图引导及提升列表数据库分页已实现。 | 通知/催收仍区分本地参考和真实送达；反欺诈按现有建模/策略入口引导，不新造无真实数据可验证的独立工作流。 |
| 代码健康 | 文件副作用移出 validation、取消检查、按任务前端状态、兼容符号登记、重复校验共享归属及行为测试替代已完成本轮约定项。 | 细小独立哈希合同和真实变体保留；静态 CSS/构建安全契约不冒充行为测试。 |
| 性能及验收材料 | 冻结性能/隐藏基准入口、全尝试预算预留、同进程预热、真实并发及故障计时、签名观测格式、空白手动验收清单已具备。 | 本机观察值不自动成为批准 SLO；独立签名由验收责任人提供。 |

V1 兼容边界保持：`RMC_SAMPLE_DF`、`RMC_TARGET_COL`、`RMC_ALGORITHM`、`RMC_SCORE_FN`；手动和 Agent 两种模式；主分数比较 Notebook 内存模型分与 PMML 分；确定性指标由平台计算。

<details>
<summary>2026-10-12 V2.7.0 候选记录（未发布）</summary>

> 以下为 2026-10-12 写入的 V2.7.0 候选记录，原样保留为历史。V2.7.0 没有公开发布，原文指向的 GitHub 发布页不存在，链接已去掉；实际发布的是 V2.7.1。


路线范围见 [roadmap.md](roadmap.md)。本文更新于 2026-10-12，记录 V2.7.0 的能力范围与验收口径；版本更新前开发候选以 V2.6.3 为基线。版本、tag、工件摘要和远端 CI 的实际结果以 GitHub 发布记录为准。下方原日期记录保留为历史；软件实现、本地检查、用户正式验收、业务效果与机构生产分别裁定。

### 本轮候选与验收层

| 范围 | 当前软件事实 | 验收边界 |
| --- | --- | --- |
| 开发候选 | 固定环境完整 V6 回归为 16,191 通过、0 失败、0 错误、11 跳过，实际退出 0；16,202 案例身份闭合，源码前后一致。固定运行输入未变；运行生成的字体缓存导致原严格目录整体检查未通过，其失败和补充分类保留 | 这是版本更新前开发候选的本地记录；3 项浏览器 smoke 和 8 项真实模型诊断为按需跳过。V2.7.0 版本更新后的完整发布检查、远端 CI 与工件结果以 GitHub 发布记录为准，不能由开发候选结果继承为已通过 |
| Agent 出站 | 分族结构化请求、主动指令敏感检查、非法材料零请求、调用预算和错误回退；聊天认证指标与报告修订/重生成接线 | 用户主动指令仍须敏感检查；原样本、文件正文、凭据和报告全文不发模型。模型解释不改变确定性指标，澄清或失败不记作工具成功 |
| Notebook / worker | V1 手动与 Agent 流程、Notebook 内存分与 PMML 比较保持兼容；第三方 input-only OS worker 已接线 | 强隔离限已验证 macOS backend 的非内置 input-only 计算；内置工具保留可信宿主执行，Linux/Windows 此隔离 backend 不可用 |
| 时间与盲 OOT | 独立封存、开发数据隔离、原生模型冻结、一次批次评估、所选候选或开发集重训子模型的监控基线 | 同主机恶意 UID 不在保证内；真实材料的可得时间、标签成熟与业务口径须独立证明。已消费 OOT 不能重新称为盲验收 |
| 独立复核机制 | 建模、策略、监控及扩展领域复算；原生工具身份、权限、PIT、幂等、固定故障和产物绑定 | 保留候选的真实留出指标与完整开发集重训分开；精简指标和实际交付终态可复核。不补造事前记录，不把本地一致性说成机构权威 |
| 软件正式验收 S | [54 项操作手册](development/手动验收手册.md)及[空白记录表](development/手动验收记录表.csv)已提供；正式冻结方法保留至少 133 个不同案例、193 个执行槽位及原门槛 | 用户使用独立材料、责任安排和原件另行验收；开发者不代签，不用测试数量自动关闭正式台账 |
| 业务效果 H | 提供历史材料映射、模型比较、现金流回放和证据核对机制 | 以用户自行冻结的材料和独立业务门槛判定；本版本不承诺新模型普遍优于已有模型，也不由模拟推断真实收益 |
| 机构生产 P | 通用协议、固定参考服务、短期签名登录/授权、模型登记、规则部署及失响应恢复 | 本地参考流程不代表真实机构上线；目标身份源、接口、SLO、批准流量和成熟业务结果另行验收 |

### 当前可用用户流程

| 范围 | 已接通的用户结果 | 明确支持边界 |
| --- | --- | --- |
| 数据与语义 | 受治理 SQL 的 Agent/API/UI、字段别名与显式方向、隔离派生、导出和历史映射 | 有界 SELECT 和 typed AST；不开放任意代码；未知方向不按列名猜测 |
| Strategy Workbench | 双人群消费者、当前快照、历史映射、任务级持久草稿、暂停/恢复与宽桌面 | 四类具有内部检查点的搜索可控制；不把所有同步 Tool 声称可抢占 |
| 策略分析与交付 | 额度/定价/分群、分群×月/金额/swap、独立 asset/OOT 稳定性、四格式报告与三种逐行等价产物 | 样本、金额和成熟事实各自绑定；验证、采纳、部署分开 |
| 在线决策与部署 | 原生模型/规则参考包、安装/读回/激活/影子/幂等/逐环境回滚；通用机构失响应恢复 | 本地原生二分类 PD 支持显式模型族；通用机构模型登记仅不透明存储 `storage_only`，远端参考部署仅 `rule_only` |
| 持续监控 | 自然语言调度、冻结日历/CAS、实际周期执行/通知、月度组合、红灯处置和反馈指标 | 未配置外部通知不称真实送达；缺 PSI、经济值或标签成熟时保持未知 |
| 决策孪生 | 历史/现金流窗口回放、原生事件引用跨源 task 授权、决策日志→监控→反馈 | 事实、假设、观察差异和因果效应分开 |
| 连接器/反欺诈 | 来源/依据/授权/撤销、事件/覆盖导入、主体/时点绑定、领域复算及原生 UI 路径 | 参考与真实来源分开；没有真实设备/关系历史不编造效果 |
| 催收 | 原生候选政策/材料映射、独立审批、队列/取消/参考执行、幂等及动作成本回收对账 | 参考 channel，不声明真实触达、回收收益或经营批准 |
| 平台治理 | 机构签名主体/RBAC/资源授权、多级复核、注册交接、双实例签名同步/撤销、签名 Plugin/Workflow 安装/权限/回滚及 OS worker | 跨实例模型传输仅二分类原生 LR 的 median-imputer→StandardScaler→LR；额外预处理及其他模型族明确拒绝；Skill 为用户编写的 Workflow 模板来源 |

本轮新增能力与兼容要求见 [V2.7.0 发布说明](releases/V2.7.0.md)。各模块使用对应运行说明和支持合同，正式验收按[手册](development/手动验收手册.md)独立记录；本地参考检查和源码存在不自动构成业务或生产通过。

`VERIFIED` 仅适用于有原件的明确层；`FAILED` 保留原失败；`NOT_PROVEN` 是事实不足；实现、API、界面、业务、签字与生产相互独立。

</details>

<details>
<summary>2026-08-09 至 2026-08-14 历史验收记录</summary>


> 唯一状态矩阵；路线范围见 [roadmap.md](roadmap.md)
>
> 当前候选状态更新：2026-08-09。V2.3.0 候选已完成代码复审、真实界面验收、
> 真实模型安全评测和完整本地 release gate；证据汇总见
> [V2.3.0 发布就绪复审](reviews/2026-08-09-v2-3-0-release-readiness.md)。
> 候选尚未在本文件写入时完成 commit、tag、release push 或对应 SHA 的远端 CI，
> 因而这些交付层仍保持 `NOT_PROVEN`，不得由本地通过结果外推。
>
> 2026-08-13/14 增量更新（90 天计划执行）：SQL 只读接入、策略历史回溯、泄漏与
> 选择偏差检测、额度/定价 typed 影响、公平性证据模块、反事实重放核心（均为
> 核心+单测纵切，未接 Agent/UI 的层保持 NOT_PROVEN）；两个超大编排文件物理
> 拆分完成（见本地架构表）。新增行的 API/Agent/Browser 层在接线并跑通对应
> 旅程前保持 NOT_PROVEN，不因单测通过而升级。

## 状态口径

- `VERIFIED`：该能力层已有可重复证据。
- `IN_PROGRESS`：当前候选正在集成或验证，尚未形成最终 verdict。
- `PARTIAL`：已有可用纵切，但未覆盖该层的完整用户旅程。
- `IMPLEMENTED`：内核存在；不能据此推断 API、Agent、浏览器或生产可用。
- `NOT_PROVEN`：当前没有足够的候选验收证据。
- `FAILED`：已有当前可重复证据，但没有达到该层验收门槛。
- `BLOCKED`：关闭条件依赖当前工作区之外的必要输入或责任人证据。
- `NOT_DELIVERED`：产品明确尚未交付。
- `N/A`：该层不适用于此能力。

下表各列彼此独立。`implemented=VERIFIED` 不会自动把右侧任何一列升级。
其中 `VERIFIED` / `PARTIAL` 记录能力层已有证据，不表示当前 dirty candidate 已通过
发布门；候选交付状态以紧随其后的交付门为准。

## 2026-08-09 候选交付门

| 交付层 | 状态 | 当前证据 | 完成条件 |
|---|---|---|---|
| 代码集成与 code review | VERIFIED | 最终集成复审未发现可复现 P0/P1/P2；LLM/Planner/Eval 专项复审未发现剩余 P0-P3 | 提交时纳入全部实现与回归文件，并冻结可审查 SHA |
| 聚焦测试与静态检查 | VERIFIED | 集成复审 378 passed、1 skipped；LLM/Planner/Eval 复审 317 passed；Ruff、核心 JS 语法与 diff check 均通过 | 后续改动必须重跑受影响门禁 |
| 完整本地 `scripts/check` | VERIFIED | `MARVIS_RUN_PLAYWRIGHT_SMOKE=1` 下退出码 0：12135 passed、8 skipped、32 warnings；Ruff、Node 和 diff check 同步通过 | 发布后用远端 full CI 复核同一发布 SHA |
| 真实浏览器验收 | VERIFIED | 真实 in-app browser 覆盖八入口、批量验证及主要正负分支；状态栏默认收起、下载、刷新、typed gate、错误阻断和 Candidate Lab 响应式布局均可见通过 | 产品新增或改变用户路径时重跑对应旅程 |
| Commit / release push | NOT_PROVEN | 当前候选尚未 commit、tag 或 push | 审查和本地门关闭后，按 `scripts/release_push.py` 冻结并发布同一 SHA |
| GitHub 远端 CI | NOT_PROVEN | 当前候选尚无对应远端 workflow 结果 | 发布 SHA 的完整 GitHub CI 全绿，并核对本地 HEAD、远端分支与 tag 指向一致 |

## 2026-08-09 能力矩阵

| 能力 | Implemented | Unit | API | Agent | Browser | Fresh workspace | Real data | Sign-off | Production | 当前产品结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| 数据处理 / JOIN | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 八个正式桌面入口之一；双表上传、角色/键/目标确认、诊断、执行与下载已走通 |
| SQL 只读数据接入 | VERIFIED | VERIFIED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 核心与单测完成（`data/sql_ingest.py`：DuckDB/SQLite/PostgreSQL 只读、单条 SELECT 白名单、行预算/超时、凭据脱敏、与文件导入同权同责注册；22 tests）；尚未接 Agent/UI，方向按 90 天计划 A-6 待用户确认 |
| 特征分析 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 全指标、选定分箱与跳过分箱分支均已走通；指标与报告由确定性工具负责 |
| 模型开发 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | LR 无 OOT、XGB 随机 OOT、LGB 时间 OOT 均以单轮调参产出结果；T4-2 未关闭 |
| 模型验证 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 手动、Agent、失败阻断及报告下载均已走通；不替代独立验证责任 |
| 模型验证（多模型） | VERIFIED | VERIFIED | VERIFIED | PARTIAL | PARTIAL | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 不再作为独立首屏入口；N≥2 从「模型验证」创建流编排，接通 Agent 对话确认合同；旧终态批次仍可下载。新工作台旅程需重跑浏览器验收，不能沿用旧独立入口的 VERIFIED 含义 |
| 策略开发 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 七阶段 Candidate Lab 与主要策略方法、稳定性/重放、编译/应用、ProjectContext 和四格式报告已走通；本地采纳不等于生产部署 |
| 策略历史回溯打分 | VERIFIED | VERIFIED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 逐月口径由 ImpactCube `month`/`segment_month` 切片与 `measure_candidate_monthly_stability` 提供；原 `historical_backtest.py` 只是同一冻结快照按月过滤，与之重复，2026-11-02 删除 |
| 泄漏与选择偏差检测 | VERIFIED | VERIFIED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 时间泄漏红旗 + 双人群分布对比（`packs/strategy/leakage_diagnostics.py`；12 tests），纯证据、不自动阻断；2026-11-02 接成 strategy Tool `diagnose_time_leakage` / `diagnose_selection_bias`（`tests/test_strategy_diagnostics_tools.py`）；未接 UI |
| 额度/定价 typed 影响 | VERIFIED | VERIFIED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 敞口/EL、定价收益-成本与 segment×month 由 ImpactCube `limit`/`pricing` family 与 `limit_pricing_matrix` Tool 提供；重复的 `limit_pricing_impact.py` 于 2026-11-02 删除 |
| Vintage / roll-rate / 利润分析 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | Standard Vintage、VTG、base-only VTG、利润分析与不兼容混合场景阻断均已走通 |
| 标签与样本定义 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 作为数据处理内的受治理 workflow；已有口径提案、成熟度门、双确认、派生数据集与下载的聚焦证据，不静默替换 active dataset；当前候选仍需重跑真实浏览器旅程 |
| 多模型分数比较 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 认证最新 SampleDesign 与每个模型分数证据，至少两个模型才执行；输出比较证据但不自动选冠军、采纳或部署 |
| 模型 / 策略监控 | IMPLEMENTED | VERIFIED | PARTIAL | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 只支持单次诊断/处置纵切，不是生产监控系统 |
| Portfolio / 集中度 / EL | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | no-trend 合成旅程已完成余额/EAD、分群、损失态、LGD、期限、报告与下载；真实机构组合仍未验收 |
| 持续运营调度 / 通知 | PARTIAL | VERIFIED | VERIFIED | NOT_DELIVERED | NOT_DELIVERED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 有持久化运行、scheduler、通知、重试与 API 基础；尚无绑定生产执行器、长期 cadence、值班与故障演练 |
| 实时 / 批量评分决策服务 | PARTIAL | PARTIAL | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 本地打分与交付物存在，生产决策服务未交付 |
| 多用户 RBAC / maker-checker | PARTIAL | VERIFIED | VERIFIED | NOT_DELIVERED | NOT_DELIVERED | PARTIAL | N/A | NOT_PROVEN | NOT_DELIVERED | 本地 session principal、maker/checker/admin、晋级与回滚 API 已有；不是企业身份源、SSO 或组织权限治理 |
| 逐环境晋级 / 激活 / 回滚 | PARTIAL | VERIFIED | VERIFIED | NOT_DELIVERED | NOT_DELIVERED | PARTIAL | N/A | NOT_PROVEN | NOT_DELIVERED | 服务端生成不可变 deployment manifest，激活只接受 allowlisted verifier 的内容寻址证据；默认无 verifier 因而不能宣称真实激活 |
| 信用决策数字孪生 | VERIFIED | VERIFIED | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 已有本地 CAS、冻结 manifest/facts/字段来源/adapter 及其 transitive helper、Champion/Challenger/反事实联合约束与 proposal-only FSP-8 桥；2026-08-13 新增有界反事实重放核心（`decision_twin/counterfactual.py`：白名单 delta、行预算 fail-closed、counterfactual_only 标记，10 tests）；尚无正式 API、UI、外部签名或远端不可篡改存储 |
| 公平性 / 拒绝原因 / 申诉 | PARTIAL | VERIFIED | NOT_DELIVERED | PARTIAL | NOT_DELIVERED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 2026-08-13 新增分群通过率/坏率/拒绝原因分布证据模块（`packs/strategy/fairness_evidence.py`，纯证据、无判定词、措辞模板待合规审阅，14 tests）；2026-11-02 接成 strategy Tool `segment_disparity_evidence`；仍不是完整合规工作台 |
| 催收 | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 不在当前已交付工作流内 |
| 反欺诈 | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 通用建模能力不等于完整反欺诈系统 |
| 征信 / 三方报文接入 | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_DELIVERED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 未交付生产解析与接入 |

## 本地架构与可信链状态

| 边界 | 状态 | 当前证据 | 不能外推的结论 |
|---|---|---|---|
| Strategy workflow 事实源 | VERIFIED | 44/44 spec 已迁移，validator/confirmation/preparer 齐全；compiler shadow 分支为 0；单一 canonical turn entry | 不代表两个超大编排文件已经完成物理拆分 |
| 超大编排文件物理拆分 | VERIFIED | 2026-08-14：`turn_handlers.py`（17,868 行）拆为 17 车道包、`strategy_request_compiler.py`（14,449 行）拆为 10 车道包，均采用 exec-merge 命名空间设计（运行时语义与原单体一致，含 monkeypatch）；turn_handlers 相关 865 tests、compiler 相关 1,993 tests（1 个 `inspect.getsource` 测试迁移到合并车道源码，语义不变）通过；两包 ruff 干净；`__all__` 逐名一致 | 不代表剩余大车道文件（strategy_candidates 约 4.5k 行等）与 app.js 单体已继续细分 |
| Canonical result 展示 | VERIFIED | 8 个 presenter；ToolRunner 在成功前做 live authentication，并绑定 invocation/output/tool version/manifest；Repository 与展示端重验 exact binding | 不代表外部存储或主机管理员不可篡改 |
| 结果数据集下载 | VERIFIED | URL 绑定 plan/step/output/hash；registry、文件与 evidence 重验；单 descriptor 私有快照关闭 verify-to-open 竞态；冻结快照支持 Range/206/416 | 不代表远端对象存储、CDN 或跨区域传输已验收 |
| 崩溃恢复 | VERIFIED | 父绑定完整性冲突会把 CHECKING step 与 RUNNING plan 收敛为 FAILED；startup reclaim 有回归 | 不代表生产级进程编排与故障演练已完成 |
| 人工筛选证据边界 | VERIFIED | 人工 selection 原子绑定 gate inputs，不覆写已完成 screen Tool 的 output ref/hash/succeeded run receipt；真实 Modeling API E2E 与独立复审通过 | 不代表人工审批可替代独立验证或生产职责分离 |
| 隔离执行与资源所有权 | VERIFIED | Notebook worker 使用 `-I` 可信 bootstrap 并拒绝 notebook 目录影子包；automatic-tree Parquet reader exact-once 关闭，`xb` 输出所有权约束 cleanup，覆盖中断、close failure 与并发创建 | 不代表外部 sandbox、容器隔离或原生 Windows 文件语义已验收 |
| 高置信清理 | VERIFIED | `marvis` 内部 DB façade import 清零；`memory.before_save` 所有真实保存路径已接线；pet catalog 单一化；持久化文件 GC；artifact identity 全树 AST guard；JS/CSS/renderer 第一轮拆分；静态 immutable URL 绑定 JS/CSS 内容 SHA-256 | 不代表仓库已无技术债或可以批量删除兼容代码 |

## 2026-08-09 阻断门

| 门 | 状态 | 当前证据 | 关闭条件 |
|---|---|---|---|
| Agent fast safety eval | VERIFIED | 语义授权、Planner/Validator、typed provenance、replan、Explore 与 LLM error fail-closed 的集成/专项回归分别 378 passed、1 skipped 和 317 passed | 发布后在远端同一 SHA 重跑 blocking safety eval |
| 当前候选本地完整 gate | VERIFIED | 完整 `scripts/check` 退出码 0：12135 passed、8 skipped、32 warnings；Ruff、Node 与 diff check 同步通过 | 后续源码变化需重新运行完整 gate |
| T4-2 公开数据参考门 | BLOCKED | `scripts/ks_baseline.py --status` 返回 2；GiveMeSomeCredit/Home Credit 文件与可追溯人工基线未齐 | 预登记 split/seed/特征/预算/容差，两个数据集均通过 |
| T4-3 真实材料对账 | NOT_PROVEN | checklist 和机器预检框架存在；当前没有已完成 B1-B5 与责任签字的 repo/workspace 证据 | 至少一套真实材料完成机器预检、外部对账和签字 |
| 当前候选远端 CI | NOT_PROVEN | 候选尚未冻结、发布或触发对应 GitHub workflow；既有远端成功不能覆盖当前 dirty worktree | 收敛为可审查 SHA，并让该 SHA 通过完整远端 CI |
| 八入口真实浏览器 | VERIFIED | 2026-08-08 至 2026-08-09 使用真实 in-app browser 覆盖八入口、批量验证、主要方法/数据集划分/按钮/Agent 话术和正负分支，并保留截图、下载、任务与日志 | 后续入口或交互契约变化时重跑 |
| 当前候选发布 | NOT_PROVEN | 尚未 commit、tag 或 release push，远端也没有当前候选 SHA | 本地完整 gate 与真实浏览器验收关闭后，使用 release helper 发布并核对远端引用 |
| 真实 LLM 安全评测 | VERIFIED | 当前候选生成 `marvis.eval.report.v2` 完整报告：0 LLM/harness error；`autonomous` 13/13、guardrail 3/3，pass rate 与 guardrail pass rate 均 1.00，成为唯一可推荐 tier；`balanced` 与 `conservative` 因 guardrail 不完整而继续不可推荐 | 模型、prompt、corpus 或治理阈值变化时必须重跑；不可把推荐 tier 外推为生产授权 |
| 原生 Windows / NTFS | NOT_PROVEN | macOS/POSIX 行为与共享 handle adapter 单测通过；尚无真实 Windows sharing violation、删除/重试/恢复证据 | 在原生 Windows/NTFS 上执行文件与任务树 GC、被占用文件、退避重试和恢复验收 |
| 生产运营 | NOT_DELIVERED | 只有本地调度/通知和 maker-checker/晋级对象；缺真实身份源、生产执行器与决策集成、长期告警值班和故障演练 | 按已批准的生产架构完成并通过故障演练与运营签字 |

## Portfolio 与 Labeling 决定

1. Portfolio 已完成正式 HTTP Agent、pre-plan 手动确认、首屏入口和报告下载纵切，
   当前只把 no-trend 作为已验证公开旅程。完整报告仍必须有余额/EAD 和业务分群列，
   缺失时 setup 失败关闭；真实机构组合、周期 trend 与生产调度仍未验收。
2. Labeling 暂不新增顶层 task type。它应先作为数据处理任务中的高风险 workflow：
   标签定义、观察窗、表现窗、坏样本阈值、成熟度 override 和派生数据集写出均需
   结构化确认、版本、lineage 和审计；结果不得静默替换 active dataset。
3. 上述决定是产品可达性边界，不是删除内核代码的理由。

## 2026-08-09 业务负责人能力判断

当前平台可以让业务专家独立完成大量本地开发与分析工作，但结论仍是
`NOT_PROVEN`。当前真实模型评测已得到可推荐的 `autonomous` tier，但这只证明当前
13-case 评测合同下的规划与安全表现，不能称为“单人生产业务线系统”。
生产数据、独立验证、审批、合规、
发布、回滚、值班和审计责任不能由提高 Agent 自主级别消除。当前准确定位是：

> 本地风控开发与分析工作台；让一个业务专家成为小型风控团队的高杠杆核心。

</details>
