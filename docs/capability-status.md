# MARVIS 能力验收状态

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

## 当前候选交付门

| 交付层 | 状态 | 当前证据 | 完成条件 |
|---|---|---|---|
| 代码集成与 code review | VERIFIED | 最终集成复审未发现可复现 P0/P1/P2；LLM/Planner/Eval 专项复审未发现剩余 P0-P3 | 提交时纳入全部实现与回归文件，并冻结可审查 SHA |
| 聚焦测试与静态检查 | VERIFIED | 集成复审 378 passed、1 skipped；LLM/Planner/Eval 复审 317 passed；Ruff、核心 JS 语法与 diff check 均通过 | 后续改动必须重跑受影响门禁 |
| 完整本地 `scripts/check` | VERIFIED | `MARVIS_RUN_PLAYWRIGHT_SMOKE=1` 下退出码 0：12135 passed、8 skipped、32 warnings；Ruff、Node 和 diff check 同步通过 | 发布后用远端 full CI 复核同一发布 SHA |
| 真实浏览器验收 | VERIFIED | 真实 in-app browser 覆盖八入口、批量验证及主要正负分支；状态栏默认收起、下载、刷新、typed gate、错误阻断和 Candidate Lab 响应式布局均可见通过 | 产品新增或改变用户路径时重跑对应旅程 |
| Commit / release push | NOT_PROVEN | 当前候选尚未 commit、tag 或 push | 审查和本地门关闭后，按 `scripts/release_push.py` 冻结并发布同一 SHA |
| GitHub 远端 CI | NOT_PROVEN | 当前候选尚无对应远端 workflow 结果 | 发布 SHA 的完整 GitHub CI 全绿，并核对本地 HEAD、远端分支与 tag 指向一致 |

## 当前矩阵

| 能力 | Implemented | Unit | API | Agent | Browser | Fresh workspace | Real data | Sign-off | Production | 当前产品结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| 数据处理 / JOIN | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 八个正式桌面入口之一；双表上传、角色/键/目标确认、诊断、执行与下载已走通 |
| SQL 只读数据接入 | VERIFIED | VERIFIED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 核心与单测完成（`data/sql_ingest.py`：DuckDB/SQLite/PostgreSQL 只读、单条 SELECT 白名单、行预算/超时、凭据脱敏、与文件导入同权同责注册；22 tests）；尚未接 Agent/UI，方向按 90 天计划 A-6 待用户确认 |
| 特征分析 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 全指标、选定分箱与跳过分箱分支均已走通；指标与报告由确定性工具负责 |
| 模型开发 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | LR 无 OOT、XGB 随机 OOT、LGB 时间 OOT 均以单轮调参产出结果；T4-2 未关闭 |
| 模型验证 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 手动、Agent、失败阻断及报告下载均已走通；不替代独立验证责任 |
| 模型验证（多模型） | VERIFIED | VERIFIED | VERIFIED | PARTIAL | PARTIAL | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 不再作为独立首屏入口；N≥2 从「模型验证」创建流编排，接通 Agent 对话确认合同；旧终态批次仍可下载。新工作台旅程需重跑浏览器验收，不能沿用旧独立入口的 VERIFIED 含义 |
| 策略开发 | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 七阶段 Candidate Lab 与主要策略方法、稳定性/重放、编译/应用、ProjectContext 和四格式报告已走通；本地采纳不等于生产部署 |
| 策略历史回溯打分 | VERIFIED | VERIFIED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | as-of 逐月重放核心与单测完成（`packs/strategy/historical_backtest.py`：缺月 fail-closed、backtested/unvalidated 标记；15 tests）；未接 Agent/UI |
| 泄漏与选择偏差检测 | VERIFIED | VERIFIED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 时间泄漏红旗 + 双人群分布对比核心与单测完成（`packs/strategy/leakage_diagnostics.py`；12 tests），纯证据、不自动阻断；未接 Agent/UI |
| 额度/定价 typed 影响 | VERIFIED | VERIFIED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | PARTIAL | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 敞口/EL 增量、定价收益-坏账成本、segment×month 矩阵核心与单测完成（`packs/strategy/limit_pricing_impact.py`；20 tests），缺失口径=unavailable；未接 Agent/UI |
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
| 公平性 / 拒绝原因 / 申诉 | PARTIAL | VERIFIED | NOT_DELIVERED | PARTIAL | NOT_DELIVERED | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | NOT_DELIVERED | 2026-08-13 新增分群通过率/坏率/拒绝原因分布证据模块（`packs/strategy/fairness_evidence.py`，纯证据、无判定词、措辞模板待合规审阅，14 tests）；仍不是完整合规工作台 |
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

## 当前阻断门

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

## “一个业务专家能否独立负责业务线”的当前答案

当前平台可以让业务专家独立完成大量本地开发与分析工作，但结论仍是
`NOT_PROVEN`。当前真实模型评测已得到可推荐的 `autonomous` tier，但这只证明当前
13-case 评测合同下的规划与安全表现，不能称为“单人生产业务线系统”。
生产数据、独立验证、审批、合规、
发布、回滚、值班和审计责任不能由提高 Agent 自主级别消除。当前准确定位是：

> 本地风控开发与分析工作台；让一个业务专家成为小型风控团队的高杠杆核心。
