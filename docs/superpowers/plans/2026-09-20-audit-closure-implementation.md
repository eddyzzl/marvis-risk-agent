# 审计整改实施记录：可信评测与代码减法

开始日期：2026-09-20；续作至 2026-09-21。用户已授权开始开发并继续完成全部开发和检查。

- 分支：`codex/audit-closure-foundation-20260920`。
- 开始时的代码：`60d4b1b45fd8161b14be6c7a02c740e16343c0b4`。
- 本记录覆盖 WP00、WP01、WP02 的第一批实现。整套计划、这些工作包及 22 个问题均未整体关闭。
- 保留开始时存在的临时 Python 文件、`.workbuddy/` 和前端审查产物；未修改业务数据、部署系统或发布版本。
- 用户当前只能提供脱敏历史数据。已询问数据目录/数据集位置，尚未完成真实数据盘点。
- 首批可信评测/代码减法已本地提交 `ea100cf6`，未推送或发布；后续 WP03 改动在同一分支继续。

## 已实现

### 评测隔离和真实口径

`EvalOrchestrator` 默认使用 `blind`；CLI 新增 `--evaluation-mode blind|contract_regression`。只有显式契约回归才允许把 expected 转成规划约束。盲测的路由、生成、重试、上下文裁剪、重规划、探索与安全检查输入不再受 expected 影响。

fixture harness 使用 `planned`、`simulated_done`、`simulation_unmodeled` 等状态，不生成真实执行的 DONE 凭证。缺失、空或非对象的工具输出不能算模拟成功。探索预算只能收紧，不能超过能力层级上限；预算耗尽不能算收敛。

报告升级至 schema v3，明确评测模式、模型来源、工具执行方式及 Executor 是否实际调用。默认 harness 的真实执行完成数为 0。能力推荐仅适用于盲测规划，标记为 `planning_only`；契约回归仅保留比较层级。未知来源的旧报告及跨模式 baseline 不能直接比较。

盲测分母保留所有 case，包括已知缺口、harness/runner/scorer 异常；错误分别归类，不记录可能含敏感信息的异常正文。契约回归中的显式已知缺口排除单独列出，不能带入盲测完成率。

### 验收证据校验

新增 `marvis/orchestrator/eval/acceptance.py` 和 `scripts/check_audit_closure.py`：

- 校验 finding、工作包、口径版本、commit、源文件快照及输入/数据/产物 SHA-256。
- 校验相对路径边界，拒绝越界、缺文件、哈希不符及未绑定原始凭证。
- 区分 C/R/A/H/P；参考环境不能满足 P，合成 fixture 不能满足 H，真实联合验收必须有真实模型和 Executor 声明及原始回执。
- 证据文件完整不代表执行或身份真实。必须由验收环境提供可信回执验证器与独立复核验证器；不能从证据 JSON 中接收 `passed=true` 作为证明。
- 关闭还要求独立冻结的 acceptance spec，锁定全部 finding ID、证据层级、口径、源码及工作包。待验台账不能通过删除问题或降低要求来完成。
- spec 和证据 JSON 都只读取一次 bytes，对同一份 bytes 校验哈希并解析，避免校验后换文件。
- 校验器只读，不修改状态。默认 CLI 未装配真实回执/复核验证器，因此能检查台账结构和绑定，不能认证问题关闭。

冻结要求文件：`2026-09-20-audit-closure-requirements.json`。

独立摘要：`05587bd02aec550400a9a6d4ad726766a3ddbf67fe204a2de28c974e26f8cc6b`。

此摘要必须从独立审核的版本交给验收环境，不能自动读取待验台账中自述的摘要作为信任依据。改变验收要求须形成新版本并保留旧要求与结果。

### KS 修复及十组辅助函数收敛

KS 参考核在常量分数、同时含好坏样本时曾误算出 KS=1。现在分别计算好坏样本的经验 CDF，在完整同分组边界比较；保留独立参考路径，没有调用生产 KS 函数。覆盖同分、部分同分、行排列、空集、单类及无效值边界。生产指标计算函数没有修改。

| 组 | 唯一责任位置及迁移 |
| --- | --- |
| C01 | `validation/field_transformations.py` 的转换闭包；输入确认、平台指标、PMML 压力测试、PMML 分数产物四处共用。第四份是在独立复核中发现并补入范围。 |
| C02 | Excel 直接使用 `validation/stress_risk.py` 的风险标签。 |
| C03 | 执行与旧结果解码共用 `validation/stress_status.py`，只接收状态字符串，不反向依赖结果类。 |
| C04/C05 | `agent/setup_context.py` 负责字段名称识别和带显式 registry 的数据集名称。 |
| C06/C08 | `agent/strategy_workflows/_input_validation.py` 共用字段拒绝检查，显式保留携带/不携带 `exception.fields` 的两个既有错误契约。 |
| C07/C09 | 同一输入校验模块复用有界数值、字段及必填文本检查；不同语义的其他校验仍保留。 |
| C10 | `feature/amount_metrics.py` 负责配对金额、覆盖率和逾期金额率，供分析和策略候选资产共用。 |

原消费者需要的局部私有名称使用导入别名，未增加转发函数或新的执行层。不会因为名称相似而合并具有不同业务语义的校验。

### 前端残留清理

已删除确认没有生产调用的六个内部函数：`closeLLMSettingsDialog`、`syncAgentMemoryViewControls`、`clearStatus`、`loadAgentMessageMemoryReferences`、`inspectDraftTool`、`workflowStepStatusLabel`。

移除保护这些名称的断言。涉及状态和批次的既有行为测试改为选择实际被测函数，不再用残留函数作为切片边界；复用草稿 controller 行为测试，并增加实际记忆 controller 的视图切换、列表加载、详情入口测试。

## 验证与独立复核

下面是各次检查的实际结果，存在重复覆盖，不能相加成独立测试总数。

| 检查 | 当前结果 |
| --- | --- |
| 前端静态/Node 行为、验证批次、草稿及报告草稿 | 357 passed |
| KS、压力结果、压力运行、Excel 和输出链路 | 171 passed |
| setup、策略输入、workflow、金额和候选资产 | 381 passed |
| 建模三文件提案和 Vintage HTTP 集成 | 2 passed |
| 第四处 PMML 闭包迁移及字段转换 | 81 passed |
| CLI 参数传递 | 23 passed；记忆 controller 另 1 passed |
| 评测隔离/异常分母修复 | 179 passed；后续预算边界改动的受影响三文件 82 passed |
| 独立复核评测隔离 | 25 passed，分母及预算两项发现已修复 |
| 证据校验器 | 32 passed，独立复核发现的降级要求、异常 JSON、校验后换文件问题已修复 |
| Ruff、JS 语法、差异空白 | 已通过；最终归档前再次检查本批新增文件 |
| `scripts/check --affected-full` | 首轮因 `.gitignore` 映射不确定扩大到全量；真实 PMML 交付失败后中止定位，不能计为通过。原始记录为 1 failed、1967 passed 后中断。环境对齐后全量为 12543 passed、2 failed、11 skipped，失败定位见下文，不能将该轮命令写为通过。 |

独立复核覆盖了 expected 输入隔离、预算/失败分母、关闭要求绑定与类型边界，以及 KS/转换闭包/压力状态/六个前端残留的行为等价性。复核者没有实现其所复核的对应改动。

原始本地验证日志保留于 `.pytest_cache/audit-closure-20260920/`。该目录不提交业务数据或被当成正式 H/P 凭证；最终阶段结果及日志摘要在本记录续写。

### 全量门发现的依赖不兼容

首轮全量门中的真实交付测试失败后，单独重跑仍复现。检查合成实验的原模型、导出调用和 JVM 输出，确认 `sklearn2pmml 0.131.0` 的转换器拒绝本机 `scikit-learn 1.9.1`，最高支持到 `1.9.0`；Java 已启动并进入 `SimpleImputer` 转换。原始诊断为 `pmml-export-repro.log`，没有修改模型 pickle 中的版本或绕过导出检查。

处理：

1. `pyproject.toml` 将 sklearn 范围从 `>=1.4,<2` 收紧到 `>=1.4,<1.9.1`。既有 `uv.lock` 本来就是 sklearn `1.9.0` / sklearn2pmml `0.131.0`；重新锁定只更新此 specifier，没有升级其他依赖。
2. E2E 服务输出从未消费的 PIPE 改为测试临时目录里的 `server.log`，保留转换器原始错误并避免管道背压。
3. conda dry-run 确认只更换 sklearn 一个包后，保存 `py313-before-explicit.txt`，离线将共享 `py_313` 的 sklearn 从 `1.9.1` 对齐到项目锁定的 `1.9.0`。安装动作记录为 `conda-compatible-install.json`；没有放宽 worker 环境白名单或注入 PYTHONPATH。
4. 后续验证重新训练合成模型，使用真实 worker、PMML 导出及 HTTP 交付。源码快照先绑定 1468 个文件，记录在 `source-before.json`，SHA-256 为 `df1c5877518fd807c9c366c6f4a5694d30595dff7d9623814fd39e3380fad881`；冻结后不再并行修改被测源码。

环境对齐后的其余实测版本为 numpy `2.5.3`、pandas `3.0.5`。这是本机开发环境验证，不等于全量依赖严格按 lock 重建、跨平台安装验证或发布验收。

### 冻结源码后的全量回归

环境对齐后，真实 HTTP → worker → JOIN → 训练 → 报告 → PMML 交付重新运行通过：`test_real_server_join_and_modeling_journey`，1 passed / 33.86 秒，日志 `e2e-compatible.log`。

随后全量 `scripts/check --affected-full` 在 6374.68 秒内完成，12543 passed、2 failed、11 skipped。Ruff、JS 语法及 Bandit 的 medium/high 门通过（Bandit 保留 23 个 low 记录）。源码前后校验 `source-after.json` 确认 1468 个文件没有变化。原始日志及 JUnit 为 `full-final.log`、`full-final.xml`。

两项失败为 `test_data_ops_ingest_excel_rejects_paths_outside_material_roots` 与 `test_tool_runner_denies_adhoc_file_read_outside_allowed_roots_at_runtime`。原因已确认：本轮将 `--basetemp` 放在仓库 `.pytest_cache` 内，而 `plugins/subprocess_worker.py` 的现行只读根包含 `_project_root()`，使负例文件实际位于允许读取范围。将同样两项测试放到系统临时目录重新执行，2 passed / 3.30 秒，日志及 JUnit 为 `boundary-external-temp.log/xml`；未修改源码、断言或隔离配置。全量门原始退出仍为失败，记录不抹去；后续全量使用仓库外临时目录。

### 真实浏览器交互

在独立临时工作区启动真实 MARVIS HTTP 服务，以既有 store/repository API 写入明确标注的合成记忆、沉淀和工具草稿。没有拦截/伪造 HTTP 响应，没有调用 LLM、执行草稿或转正插件。

Chromium 在 1440、1600、1920 × 1000 下均通过：页面身份与非空内容、模型引擎设置关闭/重新打开、原始记忆列表/详情、沉淀视图/过滤项/详情、返回原始视图、工具草稿列表/代码详情、Escape 关闭；对话框边界位于视口内，API 无 4xx/5xx，console/pageerror 为空。实际查看了 1440 记忆详情、1920 草稿详情、1600 关闭后截图，内容和交互与本次整理保持一致。

证据在仓库外 `/var/folders/z1/yls0t5ds7zj16tk472xr8kpc0000gn/T/marvis-audit-ui-20260920-dvgtp0l4/`：`browser-qa.py`、`browser-qa-result.json`、`seed-manifest.json`、`memory-detail-*.png`、`draft-detail-*.png`、`closed-*.png`。Browser plugin not available，按所用前端验收技能采用本机 Playwright。临时服务已停止。此结果不覆盖全部任务/批次 UI、真实业务数据、真实模型自主能力或生产环境。

## 尚未完成，不能隐含关闭

- WP00：完整入口/产物基线、兼容符号归属、业务预算及独立封存基准；历史数据目录和字段/时间/成熟度盘点。
- WP01：真实 API → PlanDriver → PlanExecutor → ToolRunner 的联合 benchmark runner、实际模型运行、成本/时间/token 记录、可信回执和复核验证器装配。当前新增 validator 的测试替身不是这些适配器的实现。
- WP02：剩余仅测试引用/兼容符号的逐项处置，以及全任务/批次 UI 验收。本批设置、记忆和工具草稿入口已完成真实浏览器检查；十组 helper 整理不能代替整个架构健康验收。
- WP03 及以后：正常执行/恢复与效果验证编排、共享 exec 模块、前端状态所有权、业务验收与时间契约、在线生产生命周期、调度告警、历史回放闭环、连接器/反欺诈、催收及最终独立验收。
- H/P：未提供和未实际运行的历史/机构证据保持待完成；历史回放、参考环境、单元测试都不能证明真实增量收益或生产通过。

本批交付不会将整套整改称为完成，也不构成“一个业务人员已经能独立负责完整业务线风控”的能力声明。

## WP03 续作：验证阶段与可靠完成协议

### 计算与应用编排

`EffectivenessComputation` 提供 KS、PSI、分箱阶段，普通调用方与生成 Notebook cell 共用计算，保留 live DataFrame、评分函数和全部八个进度节点。`validation_services/{metrics,confirmation}.py` 接管文件/Excel 产物和仓储确认，`validation/platform_metrics.py` 只接收内存数据和已验证契约；只读 PMML 文件身份/窄列读取放在共享 `validation/pmml_analysis.py`，仍保留原哈希及读后文件身份检查。

两项非等价修复独立留证：

1. Notebook 的 KS、PSI、分箱阶段原先没有收到取消回调，三个中途取消反例先失败、修复后通过。
2. 首版分阶段对象禁止重复运行 PSI，独立审查在真实 Notebook 内核复现了兼容回归。现允许阶段重算、清除下游失效结果；重复执行 PSI/KS/分箱不重复调用评分函数。取消或失败不能继续使用旧 `_rmc_effectiveness`。

验证侧 232 项集成检查通过；保持错误优先顺序的后续 22 项通过；真实内核重入修复后的 47 项通过。原始记录 `wp03-validation-final/ordering`、`wp03-cancellation-before`、`wp03-reentry-before/final` 均在上述本地验证目录。确认仓储的 revision CAS、PMML 身份、取消和产物回滚仍有实际行为覆盖。

### 执行、恢复和 Hook

正常执行与恢复共用 `orchestrator/completion.py`：相同复核条件、同一持久化复核快照、绑定原执行与输出身份的完成事件。新增数据库 migration 36 保存完成快照、事件义务和 delivery claim。必需 Hook 失败或结果不确定时不能记 DONE；已保存的成功回执可复用，无法确定是否发生的外部动作不自动重发。默认可选 Hook 省略 `required:false` 的 canonical 字段，保持旧 manifest hash；没有修改历史证据哈希。

独立审查实证并修复了复核快照前后、feature/step 义务分开冻结、workflow summary 重建以及 DONE 后写 warning 等窗口。写入、网络、进程或副作用未知的 Tool 失败不再自动 retry/skip/replan；明确只读 Tool 保留原重试行为。

第一轮执行联合检查 894 项通过后，继续审查发现直接 retry 之外的上游 rollback 可以清空未核对状态。当前补齐同事务的重置/修订闭包检查，并统一 Agent 消息、GET plan、启动通知、诊断增强和步骤栏的不可重试状态；对应零写入、并发写锁及真实 Driver/API 负例正在集成。894 项只代表该次快照，不能替代这些后续修改的最终检查。

**WP03 仍未关闭：**真正 unknown 结果还需要可信的只读核对适配器，以及只恢复完成协议、不重跑原 Tool 的应用入口。调用方不能传一个 success 布尔值作为证明，不能手工改 DB 解除状态。该剩余项继续属于本计划范围。

## WP01 联合 runner 续作

已按实际调用路径重新定位：六类 Driver 任务与专门 validation pipeline 分开记录；原 manual E2E 不能作为 Agent 能力成绩。新 runner 在独立分支 `codex/audit-runtime-eval-20260921` 实施，以隔离 workspace 的真实 HTTP → Agent/Driver → Executor → ToolRunner 为执行边界。案例输入与 expected 评分答案分离，fixture 模型只证明 C/R；全部模型构造路径都需由共同 client/transport 观测点记录调用与重试，未知 token/成本不能当 0。当前没有真实模型联合成绩或历史数据验收结果。

## WP00 源码入口基线

本节由独立只读源码盘点形成，确认注册位置和产物契约，不把库函数或注册项当作 UI/真实业务已验证。

共同链路是 CLI `serve` → `app.create_app` → task/Agent/Plan 路由 → 内置 Workflow 模板 → manifest 注册的 Tool。`domain.py:24` 的八个 TaskType 包含 `validation_batch`，与八个业务族并非一一对应；监控使用模板与 operations 入口，没有独立 TaskType。

| 范围 | 入口和注册依据 | 产物/证据与当前限制 |
| --- | --- | --- |
| 数据处理、JOIN、标签 | `routers/data.py:495/895`；`orchestrator/templates/join.py:23`、`labeling.py:21`；`packs/data_ops/manifest.json`、`packs/labeling/manifest.json` | 数据集 ID/hash、行数、fan-out、变换和标签质量证据；未成熟标签保持空值。文件/数据集入口不能证明已接通外部征信、核心账务。 |
| 特征 | `orchestrator/templates/feature.py:17/168`；`packs/feature/manifest.json` | 统计、分箱、筛选理由、衍生数据集与报告；不代表在线特征服务或跨源时点一致性已通过。 |
| 建模 | `orchestrator/templates/modeling.py:17`；`routers/modeling.py:101`；`packs/modeling/manifest.json` | 实验、模型、训练/评分证据、确定性指标、PMML、验证移交；`score_dataset` 是批量评分，不是请求级在线评分。 |
| 验证 | `routers/validation_stages.py:83`、`validation_batches.py:176`；`orchestrator/templates/validation.py:13`；`packs/v1_compat/manifest.json` | Notebook/runtime model、验证结果、Word/Excel；批次走专用创建入口，旧验证与 Plan/Tool 路径并存。 |
| 策略 | `routers/strategy_candidate_lab.py:18`；`orchestrator/templates/strategy.py`；`packs/strategy/manifest.json` | DSL、候选、回测、采用版本、监控计划和交付等价性证据；domain 的策略类型比确定性候选设计支持类型更宽，不可混用能力范围。 |
| Vintage/风险分析 | `orchestrator/templates/strategy.py:4495`；`packs/risk_analysis/manifest.json`、`tools.py:71` | 风险 Excel、headline metrics、red flags、assumptions、column map；报告不代表账务对账和贷后动作已接通。 |
| 监控 | `orchestrator/templates/monitoring.py:18/75`；`operations/router.py:168/245`；modeling/strategy manifests | 漂移、监控版本/hash、处置记录和报告；`operations/integration.py:134/144` 默认没有实际执行器、通知适配器或后台节拍。 |
| 组合 | `orchestrator/templates/portfolio.py:184/208`；`packs/analysis/manifest.json` | 迁徙/流转矩阵、集中度、EL/假设、风险和 Excel；趋势要求真实模型产物及基线分布，缺失时拒绝并提供 no_trend 模板。 |
| 生产生命周期 | `production_governance/router.py:132/177`、`evidence.py:117`；`app.py:462` | 有批准、激活、回滚、审计 API 与匹配回执门；默认 verifier 为空。治理记录不能证明线上部署器已装配。 |
| 连接器与反欺诈 | `app.py:746/764`；`packs/modeling/scenarios.py:79` | 内置包未注册专用 KYC/征信/核心账务连接器；允许外部插件动态加载，本次未检查用户安装。transaction 建模场景不等于设备/关系图/流式拦截服务。 |
| 催收 | `domain.py:46`、`packs/strategy/candidate_design.py:970` | 无 collection 策略枚举，候选设计显式报 unsupported；迁徙/EL 不能代替催收动作、成本、回收和案件管理。 |
| 决策回放 | `decision_twin/authenticated_entry.py:264`、`replay.py:490` | 已有认证回放、比较、反事实和结果对账库；本次代码引用检查未见 app/router/template/pack 接入，产品入口仍待实现。 |

下列边界必须保留或另批迁移，不能按静态引用少直接删除：

1. `plugins/subprocess_worker.py:184/663` 按 manifest 的 module/entrypoint 动态调用 Tool；无静态 import 不代表无调用。
2. 初始盘点时 `agent/turn_handlers/__init__.py:35` 与 `strategy_request_compiler/__init__.py:27` 均使用共享 exec；compiler 已在后续 WP04 批次迁移，turn handlers 仍待处理。
3. `api.py:96/127` 的兼容别名和旧 `repo=` 参数转换仍承载扩展导入责任；需要逐项迁移与兼容测试。
4. `orchestrator/templates/sample.py:173` 的动态注册及 `templates/skills.py:77` 用户模板装载继续使用既有 ToolRegistry/PlanValidator，不能另立 runtime。
5. `reconcile.naive_ks` 保留独立数值交叉核验责任。消费者的 `_dataset_name`、`_resolve_named_col`、`_amount_metrics` 等导入别名仍被业务调用，是单一实现的引用，不是重复算法。


### 执行契约、取消和重启窗口

新增 migration 37：在现有 step-run 账本中保存不可变 `invocation_contract_json` 和只允许首次写入的 `dispatch_started_at`。真实 ToolRunner 在准备阶段解析权限和声明，在任何副作用预约/worker 启动前再次核验声明并持久化派发边界。执行失败、取消和启动恢复共用冻结记录判断；旧记录 NULL 或损坏契约保持未知，不用当前 manifest 倒推历史只读。准备后未派发可证明没有执行，已经派发的写入且无回执继续停在待核对。

独立新增的真实 ToolRunner/worker 故障集 18 项通过，覆盖声明漂移、升级、禁用、写入后失败/取消/进程退出/host 崩溃、可信只读、未派发、数据库不可变约束和 36→37 迁移。上游 rollback、参数修订、reset、replace、append 共用事务内未核对检查，历史说明文字不能覆盖底层执行记录。

主树冻结后的 34 文件联合门：**1748 passed、1 failed / 293.30 秒**。1480 个源码/测试/脚本文件前后 hash 未变；记录为 `wp03-combined.log/xml` 与 `wp03-source-before/after.json`。唯一失败是上传迁移测试只创建一个 upload 表却自称完整 v34，后续 migration 37 无法更新不存在的 step-run 表。修复测试为真实 v34 基线再注入残缺 upload 账本；生产迁移不静默跳过缺失的核心表。修正后的上传 API、DB、step-run 与真实 invocation 迁移联合回归 **192 passed / 68.80 秒**，日志 `wp03-migration-followup.log/xml`；仅修正测试前提，没有改变运行时代码，不改写原始失败结果。全量 Ruff、JS 语法与差异检查通过。

### WP03 浏览器呈现检查

真实独立 HTTP 服务，任务由 API 创建、失败计划由真实 repository 写入明确标注的合成状态；未 mock 网络。Chromium 1440/1600/1920 × 1000 均验证：后端 `retryable:false`、待核对卡片不含输入/重试按钮、步骤栏显示待核对、安全失败仍显示重试表单、切换任务后旧控件移除。API 无 4xx/5xx，console/pageerror 为空。页面动画完成后重新截图，实际查看 1440 待核对与 1920 安全重试页面。

最终检查目录：`/var/folders/z1/yls0t5ds7zj16tk472xr8kpc0000gn/T/marvis-wp03-browser-settled-20260921-glfwqdjc/`，含 `browser-qa.py`、`browser-qa-result.json`、`seed-manifest.json`、`unknown-*.png`、`safe-*.png`，临时服务已停止。首轮动画中间态截图保存在另一个 `marvis-wp03-browser-20260921-diokcv48` 目录，不作为静态呈现证明。

检查同时发现顶部通用状态仍写“验证失败”，与侧栏待核对以及 JOIN 任务类型不一致；归 WP05 状态所有权收敛继续修复。本轮仅证明重试保护和任务切换行为，没有宣称所有状态呈现已统一。

## WP04 独立开发分支

`codex/audit-module-refactor-20260921` 已开始编译器真实模块化。初步去除 shared exec 的版本通过 1635 项编译器回归与 11 项模块身份/类型解析检查；仍有普通 imports 循环，不能以删除 exec 即宣告完成。已根据符号依赖制定 contracts、共享文本规则、标识正则与各语法家族的归属搬迁，继续实现无环依赖并保持公开 facade。此时尚未合入主树，turn handlers 和真实 Agent 全流程验收仍待完成。


## WP04 编译器模块化合入

`5c9b5fa8` 已合入主线：十个语法 lane 使用普通 imports；contracts、grounding、identifiers 持有公共结果、文本规则及标识定义，语法家族不反向依赖 facade/core。保留已有公共导出、兼容入口和 logger 名称；删除 exec/shared-globals 及 TYPE_CHECKING 假接线。模块依赖无环，函数定义归属和类型解析由实际导入检查。

隔离分支最终 1640 项编译器、特征化和模块归属回归通过（12.85 秒），scoped Ruff 与差异检查通过；主代理检查模块分层和接口后合入。原始记录在模块 worktree `.pytest_cache/wp04/compiler-dag-tests.log/xml`。这批未运行外部模型，不能代替 A 层；`turn_handlers` 仍需迁移，WP04/F16 保持进行中。

## WP05 状态呈现首批合入

`e2f7922d` 将任务顶部状态移到 `task-status.js`，直接使用现有 plan 的结构化状态。待核对显示待核对，通用失败提示不再把 JOIN 任务解释为模型验证失败；自由文本中的“复核”也不能改变确定失败的状态。

最终两文件 336 项 Python/Node 检查通过（7.03 秒），JS 语法、scoped Ruff 和差异检查通过。真实临时 HTTP 服务下，1440/1600/1920 × 1000 均检查顶部、侧栏和步骤栏一致，unknown 无重试表单、安全失败保留重试入口，切换任务没有残留控件；实际查看了 1440 unknown 和 1920 safe 的静止截图。Browser plugin not available，沿用本机 Playwright。证据目录为 `/var/folders/z1/yls0t5ds7zj16tk472xr8kpc0000gn/T/marvis-wp05-status-browser-20260921-onojce71/`，临时服务已停止。

这批只关闭具体状态显示缺陷，不将 WP05/F17 标为完成。任务/模型访问身份、请求取消、乱序响应、消息和 busy 状态所有权继续开发；历史数据与生产证据尚未取得。


## WP04 Agent 流程模块合入

`bdf47290` 合入 `turn_handlers` 的普通模块边界；contracts、C1 state、data context、semantic authorization、strategy evidence 持有下层职责，dispatch/recovery/registry 使用明确依赖，facade 保留全部 307 个既有导出。产品定义归一化 AST 等价，250 个注解可解析、992 个跨模块符号查找可解析，依赖图无环。

73 文件分四片运行，首次为 1255 passed、4 failed、1 skipped；4 个失败均来自同一个参数化测试仍通过变量名 patch 旧 facade。迁移到实际 strategy_candidates owner 后，受影响文件和架构检查 17 passed，去重后的相关案例为 1259 passed、1 个既有 live DeepSeek opt-in skip。产品源码 29 文件在回归期间哈希未变。没有将这批称为真实模型验收，也未宣称大领域函数全部完成细分。原始记录在 `/tmp/marvis-turn-lanes-20260921/.pytest_cache/`。

## WP05 访问身份与晚到读取

`8876ec66` 合入 task/model 访问代际、按资源 latest-request 身份和 AbortController。任务与批次子模型切换使旧读取失效；旧消息、输入契约、报告字段及动作提示不能写入新访问。377 个相关 Python/Node 检查通过（8.29 秒），JS/Ruff/diff 检查通过。

真实 HTTP 浏览器在 1440/1600/1920 下延迟真实后端消息响应，执行 A→B→A 切换；旧请求取消、最新服务器消息保留，页面无错误。实际查看 1440 截图。证据 `/var/folders/z1/yls0t5ds7zj16tk472xr8kpc0000gn/T/marvis-wp05-request-browser-20260921-uehgty84/`。此前脚本误用 update_agent_message 签名的失败记录保留，不计为通过。写请求、busy lease 与完整 session 所有权继续由后续批次完成。

## WP01 真实 Agent 联合运行及失败记录

`156baa3c` 引入隔离应用副本的真实 HTTP Agent/Driver/Executor/ToolRunner runner；评分 expected 在父进程、应用与 worker 的模型配置均经过父进程代理，真实模型 key 不进入应用快照。服务调用、重试、工具回执和输出认证均留存；未知 usage/费用不按 0 处理。88 个相关检查通过，另有真实慢 headers/JSON trickle/无换行 SSE 的有界终止反例。总体 token 与费用硬上限尚未实现，仍标记 not_enforced。

使用已有本机 DeepSeek v4 flash 配置，只发送开发用合成 JOIN/feature 数据与聚合，不读取用户历史数据。两例在首轮运行前冻结预算：每例 180 秒、8 次模型 transport attempts、120 HTTP、单次输出上限 4096；expected 与案例均未在重跑时变更。

- 首轮 `20260920T182834930534Z-3084d3bb7af5`：2/2 失败；JOIN 及 feature 都因评测器 50ms 轮询耗尽 HTTP 限额，不能因为部分工具 DONE 而计为通过。
- `8193c8fe` 修正为状态不变时逐步退避至 2 秒，实际状态变化恢复 250ms；两项 clock-driven 反例先失败，修复后 18 项 runtime 检查通过。
- 原预算重跑 `20260920T183556260801Z-448e6690a648`：JOIN 通过（44.489 秒、23 HTTP、8 attempts）；feature 失败（107.433 秒、50 HTTP、8 attempts，后续 retry 被拒）。三 Tool/产物断言通过不能覆盖预算失败，总成绩仍是 1/2。baseline comparison 未发现倒退也不等于整组验收通过。
- feature 第 6 次 critic 和第 8 次 summary 的 completion/reasoning 均为 4096；可证实 critic 重试及 summary 后续请求耗尽预算。旧记录缺 finish_reason，reasoning 耗尽正文额度只是推测，不补写历史结论，也不提高预算掩盖失败。

原始案例、expected、两轮 report/attempt/HTTP/原始评分档案位于 `/var/folders/z1/yls0t5ds7zj16tk472xr8kpc0000gn/T/marvis-runtime-acceptance-20260921-fkiev2tv/evidence/`。这些是公开合成开发例的真实模型证据，不是隐藏集、全部工作流、H 或 P 验收；WP01 保持进行中。


## WP03 原生产者核对与明确续执行

`f39e981a` 合入 migration 38、只读 OutcomeVerifier 注册、不可变核对证据和原始策略采用凭据。策略采用在同一领域事务中保存完整输出、审批/调用身份和产物哈希；核对仅接受服务器推导的 target_id，不能提交 success/proof/verifier。缺核对器、旧 NULL 契约、缺完整输出、身份漂移或产物改动继续未知。已确认未生效且原生产者被 fence 后，最多消费一次新 Hook generation；Tool 后续重试仍经正常审批。

恢复只调用共享 completion 协议，不重发原 Tool。取消后核对不自动继续，需明确 resume-completion；复核不通过仍保持失败。root 独立审查补出非末步骤恢复后的死路：原步骤完成但计划 RUNNING/无活动 job。现在 GET 给出带当前 fingerprint 的 continuation，UI 显示待继续，POST 现有 run 入口执行剩余步骤；入队前、拿到执行 lease 后再次检查快照，后续确认门不改变。三个新反例在修复前全部失败、修复后通过。

最终验证：API/controller/reconciliation 90 passed；producer/governance/runner/recovery/module 226 passed（64.93 秒）；completion/Driver/reconciliation/controller/module 联合 583 passed（63.25 秒）。这些集合重叠，不相加为独立测试数量。相关 Ruff、JS 语法与 diff 检查通过。根额外检查了原子 producer receipt、绑定不可变约束和领域输出认证路径。

真实独立 HTTP 服务、Chromium 1440/1600/1920 均通过“待继续 → 继续剩余步骤 → 原审批门”；首步 run 账本未变化，第二步没有提前生成 run。UI 数据由明确标注的合成 Hook/runner 夹具生成，服务与 API 未 mock；此证据证明界面与审批门，不是机构效果核对。实际查看 1440 继续页与 1920 确认页。证据 `/var/folders/z1/yls0t5ds7zj16tk472xr8kpc0000gn/T/marvis-wp03-continuation-browser-20260921-u_eb3dnj/`；首版脚本误把说明定位在步骤栏的失败留在 `_15nfitu`，修正定位后新目录通过。Browser plugin not available，使用本机 Playwright。

正式实现的原生产者核对器当前只有 strategy.adopt_strategy；合成 Hook adapter 仅用于测试。不声称外部支付、通知、机构部署已有真实核对能力，WP03 整体仍等待范围内独立验收。

## WP05 写请求与操作租约

`94ffecb0` 引入 task-activity 唯一活动账本，旧 finally 只能释放自己的 lease；请求访问和 operation 代际拒绝过期 start/POST/poll 回复，stop 使用独立操作，重复提交不重复派发。真实浏览器暴露 manual Vintage 入口 POST 在途时没有 stop 投影，已修正为这些已支持风险入口也能停止。

372 个相关前端检查通过，最后提示清理后 17 个受影响检查再次通过。三个桌面宽度的真实后端浏览器证明 POST 恰一次、stop 独立、旧响应不能覆盖停止结果，A→B→A 保留最新消息/未发送草稿，批次 M1→M2→M1 的真实 GET 乱序不串模型。实际查看 model-revisit-1440 和 stop-owner-1920。证据 `/var/folders/z1/yls0t5ds7zj16tk472xr8kpc0000gn/T/marvis-wp05-activity-browser-20260921-bsiv4s0k/`。完整 session、plan 读取与修订身份继续下一批，未以删除若干全局变量冒充完成。

## WP01 安全的响应观测

`e0c738e9` 增加 allowlist finish_reason、首 choice 正文/思考字符计数与 attempt_outcome；不保存正文、prompt、思考内容、业务明细或密钥。HTTP 200 与正文缺失、格式损坏、输出耗尽分别记录，content_present 只表示 transport envelope，不意味着业务成功。修正无末尾换行 SSE 行的 usage 漏记。

98 项真实 loopback response/runtime/LLM/CLI 检查通过（87.94 秒），原两轮 16 个归档文件 hash 保持不变。未额外调用真实模型、未改变预算或判分。总体 token/费用硬保证仍 not_enforced：事后 usage 门只能阻止后续调用，不能保证供应商已在途计费上限；未知价格/usage 仍未知。
