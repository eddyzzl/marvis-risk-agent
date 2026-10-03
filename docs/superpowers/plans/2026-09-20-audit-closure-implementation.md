# 审计整改实施记录：可信评测与代码减法

开始日期：2026-09-20；续作至 2026-10-03。用户已授权开始开发并继续完成全部开发和检查。

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


## 2026-09-27 恢复开发与已合入批次

截至主线 `614afb94`，已合入以下经过独立阅读的批次；这些是软件进展，完整 S/H/P 仍未完成。

- `305fed25`：明确声明的主体分组不能因缺列而退化为逐行切分；缺失主体身份、跨分区冲突停止执行。218 项建模相关检查通过。
- `d1088b40`：WOE、统计变换、聚合共享训练成员选择；未知/空分区拒绝，省略 test 的 holdout 名单不能让 test 进入拟合；独热编码只从指定训练集学习类别，评分复用同一映射。16 个反例修复前失败，最终 102 项特征检查通过。
- `3e367923` / `614afb94`：时点 JOIN 用事件、可得、决策时间及版本选择生成不可变矩阵与双来源证据，并通过正式 `data_ops.asof_join` 和 `dataset_asof_join` 模板运行。模式及 available_at（列契约或明确 null）必须显式给出，复用原有输入绑定审批；155 项相关检查通过，包括真实 Driver→Executor→ToolRunner 子进程。verified 仅代表记录的时间约束通过，外部来源真实性未独立验证；训练仍需显式选择产物。
- `6c620d41`：TaskSession 收敛任务、模型、消息、计划及请求/操作生命周期；计划读取和重试同样受访问代际约束。446 项相关检查和 78 项额外消费者检查通过；真实后端三个桌面宽度包含计划 GET 的 A→B→A 乱序场景。
- `e90a4b84`：业务合同、确定性采用对象验收、经济假设和 API/Agent/DOCX/XLSX 统一投影。宽门首次 968 passed / 2 个新增 null 字段断言失败，修复后相关 169 passed；解释链 29 passed。尚缺认证测量上下文时结果仍为证据不足。
- `d46a6496`：整合检查发现核对恢复仍会把新业务 goal_doubt 错当作执行未完，两个反例先失败；修复后 completion/reconciliation/business acceptance/frontend session 联合 106 passed。旧 execution_completed=None 保留原兼容语义。

恢复时主线 tracked 文件干净，用户未跟踪文件保持原样。多个 `/tmp` 隔离工作目录已仅剩空目录，未提交草稿和部分测试日志不可再读；不把历史日志位置当作当前可访问的正式证据。仍存在的前端八文件草稿已保留并迁移到持久工作区。后续修改和测试证据改在 `/Users/eddyz/.codex/worktrees/audit-{fit-provenance,business-producers,business-ui,operations-runtime}/` 下保存；检查目录位于各 checkout 外，避免源码读取许可干扰隔离负例。

上述测试次数记录已发生的开发验证，不重新解释为当前全量回归或正式关闭证明。最终验收仍须使用当前冻结源码、实际可读取的原始证据与独立检查器。当前继续实现预处理参数/拟合成员证据到训练的完整消费、通用业务合同与真实采用上下文、业务验收界面、默认监控运行和本地通知。数据清单与机构条件的缺失不阻止这些独立软件工作，也不使 H/P 自动通过。


## 2026-09-28 预处理证据与默认监控合入

`27cc4c70` 将拟合参数、每次使用的训练行集合及父数据 hash 绑定到同一数据/产物事务。参数 sidecar 存在但损坏时明确失败，不再按“无预处理”继续；重复变换使用独立文件，避免覆盖旧数据。建模准备保留证据；单模型、批量训练、调参在训练前核验，新切分不能把过去拟合过的行移入评估集。参数证据写入失败回滚数据与文件。模型参数保留证据摘要，平台元数据不进入估计器参数。

相关宽门首次为 357 passed / 6 failed（510.90 秒）；六项均为调参检查点单元测试使用不具备 repository 的旧测试替身。将该单元测试的证据边界显式替代，同时保留真实 ToolRunner 的训练、批量训练、调参阻断负例后，最终 31 passed（41.28 秒），scoped Ruff 与 diff 检查通过。日志在持久 checkout 外 `/Users/eddyz/.codex/worktrees/audit-fit-provenance/{combined-valid,followup,final}.log`。这只证明已记录变换的拟合成员；完整 PIT、标签成熟度、外部来源真实性及所有数据变换的传递仍有工作，不将 `training_only` 解释为这些更强结论。

`419c75c8` 在普通应用启动时绑定真实 model/strategy monitor 并管理后台生命周期。调度保留原 SQLite lease、幂等、补跑预算，长任务续租；禁用或改版后旧快照不能重新领取。监控只读取指定周期的已认证数据窗口；无新数据、不完整水位、未来水位、未成熟/缺失标签分别给出诊断和下一步。本地收件箱提交后才返回投递成功，用户已读另有身份/时间；显式重新评估生成关联单周期任务，原结果保留。`fb200d07` 使停用调度仍能由管理接口查到，执行器默认继续只选启用项。

最终 operations/default-runtime 52 passed（60.89 秒）；app/CLI/既有监控核 126 passed（147.96 秒，发生在最后一次显式成熟时长输入收紧之前）；管理接口 followup 20 passed。使用最终配置重跑实际 `python -m marvis serve`：全新 workspace、HTTP 上传 400 行合成数据、配置日历，后台自行调用真实模型监控并投递本地通知，再经 HTTP 记录已读；没有手工 tick、注入监控执行器或外部发送。证据 `/Users/eddyz/.codex/work-artifacts/operations-runtime-20260927/` 保存 HANDOFF、proof、server/configuration、CLI probe、测试日志和文件摘要；服务正常 shutdown。

WP11 的应用内配置、通知、诊断、已读、重新评估浏览器完整旅程仍在接入。软件运行不能认证发布者声明的数据完整水位及真实标签事实；机构渠道与 P 验收仍未取得。主线最新软件批次是 `fb200d07`；业务合同/采用 producer 和 UI 的独立分支正在复核，申请级参考服务已经开始开发。冻结要求不变，全部工作包和问题均未借本次进展整体关闭。


## WP07 派生特征的训练与评分一致性

`1c0bcd8c` 补齐交叉、比值、日期和 log 派生的评分规则；分位排名和组统计保存训练集拟合参数，而非在新批次重新计算。平台评分器与移交 Notebook 继续调用原 `apply_preprocessing_steps`，新增规则进入同一链。每项统计变换保留自己的拟合成员，混合 recipe 只要一项用了评估成员就不能作为独立评估。纯逐行派生显示 `row_local`，不冒充时点来源已验证；目标列不能直接作为派生输入。

抽出组统计拟合/回放的唯一纯计算实现，保留行顺序及 index；大整数分组 ID 不经 float 取整，避免将两个不同身份合并。未见组用固定训练总体回退值；缺失值保持缺失。已有纯探索 helper 的行为保留，正式 cross_features Tool 的 rank 则要求明确训练分区或全量探索声明。

特征/证据回归 92 passed（190.85 秒）；拟合/回放 27 passed（8.54 秒）；评分/Notebook 门 67 passed / 1 个新增测试注册参数误用失败，修正为实际 public API 后最终 30 passed（10.12 秒）。真实 ToolRunner 派生→训练后，已变换批量评分、原始批量回放评分和逐行评分一致。scoped Ruff、diff 检查通过。持久日志 `/Users/eddyz/.codex/worktrees/audit-fit-provenance/{derived-first,derived-replay,derived-scoring,derived-final}.log`；没有覆盖原失败档案。普通清洗/JOIN 的完整传递、特征可得时间、业务标签及真实线上环境仍分别验收，不因这一批通过而关闭 WP07。

## WP06 通用业务合同、采用对象与业务验收界面

`2187333f` 将任务业务目标独立持久化到 schema39，明确更新 lease 和计划生成后的合同边界；旧嵌套目标仅按原值迁移，不推断新目标。`d4210195` 用真实 SampleDesign、最终模型及策略测量上下文绑定采用对象，人审确认来源口径；`b4f7d2bc` 核验标签 producer、源/结果 hash、实际成员、目标定义和观察成熟度。`8edf09ce` 阻止旧候选 max 上界被偷偷换成采用对象单值上界，非等价口径必须明确澄清。

`227ac10e` 接入业务合同编辑、业务五态、证据和同一验收引用的 DOCX/XLSX 导出，同时修复空计划身份判定。UI 分支 416 项相关检查通过；真实浏览器三个桌面宽度的业务状态是明确合成夹具，不代表业务目标实际达标。主线 `227ac10e` 的 DB/合同/producer/标签/UI/operations 联合检查 209 passed（105.63 秒），仅既有 Starlette 弃用提醒；日志 `/Users/eddyz/.codex/work-artifacts/audit-integration-20260928/merged.log`。原生证明只保证平台登记内容一致，外部业务真实与 H 验收继续保留。

## WP09/10 本地参考在线决策与受治理生命周期

`07b15738` 冻结 canonical DSL、模型、ensemble 成员、校准、预处理来源及输入/业务节点契约，评分与规则继续复用既有计算核。`43645f1e` 把可信来源前移至原生训练/校准产出事务；打包阶段只核验原生字节收据，禁止读取旧文件后补签为可信来源，主进程不新增不可信 pickle 反序列化。已选中模型需要精确匹配最终产物；过大的数字输入返回受控错误。`5d9b6afd` 阻止校准期间的并发模型元数据变化被重新认证。

`4d12919f` 提供真实 worker 安装探针、只读发现、受审批激活、状态读回及回滚，唯一运行指针继续属于原生产治理 head。客户端超时后查询已经提交的安装记录，不能盲目重复改变运行状态。`a6c3b9a1` 将崩溃恢复与新申请统一纳入四个并发槽；120/min 只指不同新申请 ID，不冒充全部执行尝试限额。完成请求的幂等读回不占新槽。

模型/认证包相关 182 passed，部署治理 18 passed，校准并发反例 6 passed，过期恢复并发 5 passed。真实标准 CLI→HTTP 探针覆盖两包、三次审批安装、shadow/production 分离、决定幂等、安装客户端超时后的读回、应用重启及回滚；训练是实际 ToolRunner 的 400 行合成数据，策略 validated 状态是明确夹具。证据 `/Users/eddyz/.codex/work-artifacts/operations-runtime-20260927/reference-decision/{cli-proof.json,cli-server.log,cli_governance_probe.py,recovery-admission.log}`。这些检查不证明业务批准的 SLO、真实策略业务效果或机构生产验收。

## WP11 监控工作台完整参考旅程

`5ae9534e` 增加应用内任务/数据/目标选择、配置修订、诊断与确定性证据、本地通知、明确已读、停用恢复及关联重新评估。`59d376ce` 给后台轮询补身份与访问代际约束；旧 403 不得清空新会话身份，旧能力响应不得覆盖新视图，关闭面板后仍可在同一身份下刷新未读。

394 项相关前端回归通过（53.71 秒），轮询反例修复前失败、修复后 20 项通过。标准 CLI 真实训练 LR 并运行后台监控，浏览器实际验证 1440/1600/1920、投递与已读分离、停用→恢复、陈旧版本 409 保留输入、重新评估保留原周期、真实上传晚响应和独立 checker 只读；无 pageerror，只有有意触发的 403/409。最新证据 `/Users/eddyz/.codex/work-artifacts/operations-ui-20260928/run-1790578954/proof.json`；root 查看实际 1440 指标证据截图。没有真实外部消息发送。

## WP12 原生历史批量回放与现金流对账

`117b204e` 增加独立 v2 批量合同，保留 v1 严格模型。认证同任务数据、逐字段事件/可得/决策时点和冻结包，在同一支持人群执行实际评分和规则；业务人审绑定来源、字段及可选经济/群体/产能假设。缺失维度保留 unknown，原生签名回执作为 API、Agent、JSON/Excel/Word 的同一证据。外部动作明确为 historical import，不冒充平台实际执行；成熟现金流与有可比预测的成员对账，不推断因果收益。

新增 native/Agent/manual Workflow→人审→ToolRunner/API/导出检查 20 passed；v1/reference 联合 80 passed。插件/模板联合 136 passed / 1 个旧迁移夹具失败；该夹具曾把最新库版本号改成 35，却留有新索引，不能模拟历史升级。`89300c6a` 改为真实安装 schema35 后升级，`3fb9b21b` 修复相邻 schema37 夹具使用新增列的问题，均只改测试、保留原断言，联合 180 passed（34.42 秒）。记录 `/Users/eddyz/.codex/work-artifacts/historical-replay-20260928/{proof.json,migration-proof.json}`。独立时间窗口稳定性、完整工作台、历史包可用性及 H/P 仍未完成。

上述在线/治理/历史/UI/调度/迁移在主线 `3fb9b21b` 联合 188 passed（124.20 秒），仅既有 Starlette 弃用提醒。首次命令误写不存在的测试文件，未执行测试；修正文件名后运行通过，两个日志分别保留为 `/Users/eddyz/.codex/work-artifacts/audit-integration-20260928/runtime-joined.log` 和 `runtime-joined-final.log`。这是一组相关软件回归，不是全量项目或正式关闭验收。

## WP01 Agent 总 token 与费用准入预算

增加可选任务总 token、费用及币种合同，每个重试/子调用在统一父进程锁内先预留上下文与输出的上界，再发出请求。已确认的最终完整 usage 可以释放未用预留；未知、断流、残缺、回退或自相矛盾的流式 usage 保留额度。即使 usage 不完整，已知字段越界也立即使后续准入停止。拒绝多个 completion 与可绕过 max_tokens 的替代字段；费用价格表在父进程预先冻结并核对模型/币种，缺失不能当成零。旧 case 未声明新字段时 wire/hash 保持不变。

独立审阅先后发现流式提前释放、reasoning 矛盾、输出倍率、缺失字段掩盖已知超限及跨 chunk 拼接问题，均补实际 loopback 反例后修复。最终 budget/runtime/response 联合 54 passed（102.75 秒）；scoped Ruff/diff 通过。日志 `/Users/eddyz/.codex/worktrees/audit-fit-provenance/aggregate-budget-{first,final,review,complete}.log` 保留每轮结果。本次没有追加真实模型调用或修改旧评分档案。该预算保证的是声明的模型上限与指定价格下的准入控制；不宣称独立供应商账单保证。全工作流真实模型及封存验收、完整受信关闭适配器仍待完成。


## WP12 时间窗口稳定性补齐

`5e194018` 增加显式时区、独立参考/比较窗口、样本量与阈值合同，参考窗口独立拟合 bins，复用确定性 PSI 计算。常量、不足、不可比较的窗口返回 unknown，不能通过跳过坏窗口报告整体正常；同一结果进入已签名 JSON/Excel/Word。195 项相关检查与真实人审→ToolRunner→导出结构验证通过，证据 `/Users/eddyz/.codex/work-artifacts/historical-replay-20260928/temporal-proof.json`。没有把这些检查称为历史包当时可用、训练外独立性或因果收益证明；导出视觉检查仍待完成。

## WP13 来源授权与事件特征基础

`e46955ab` 提供来源协议、真实本地身份角色和有范围/期限的来源授权基础。`22aab8d7` 将事件及覆盖声明从认证 DatasetRegistry 导入，分别记录事件时间、来源可得时间和平台接收时间，冻结版本成员、授权与窗口契约。窗口和关系特征使用同一纯计算核；命名空间不同的同值 token 不自动合并，缺少覆盖不能计算为零；历史修正不能重写已生成的决策快照。74 项事件/认证快照和 70 项直接认证调用方回归通过，证据 `/Users/eddyz/.codex/work-artifacts/event-foundation-20260928/proof.json`。HTTP、Workflow、ToolRunner、决策包和界面接线继续进行，未宣称完整反欺诈工作流已验收。

真实并发反例发现重复认证已硬化文件时再次 chmod 会改变 ctime，导致持有原 FD 的合法读取被误报为发生变化。`66e37b2c` 仅在权限仍超出允许范围时收紧权限，保留首次硬化和更严格的已有权限；两项原反例先失败，修复后 9 项相关测试通过，没有吞掉完整性错误或用重试掩盖问题。

## WP14 催收资金对账与动作约束

`1438e2ee` 新增同任务的不可变案件、付款/成本/冲销与显式分期分配；金额只用整数最小货币单位。重复回传幂等，金额冲突和缺失原始事件拒绝整批入账，冲销不得超过原金额。完整覆盖声明不足时金额结论为 insufficient_evidence；成熟度绑定业务观察政策，但不把声明认证成外部事实。对账文件、登记记录和不可变回执在同一事务内完成。首次 19 项、增量 20 项通过；修正一次不存在的测试路径后，最终现金流/产物事务检查 51 passed。日志 `/Users/eddyz/.codex/worktrees/audit-fit-provenance/collection-cashflow-final-corrected.log`。

`50632c6b` 通过显式 DSL v2 加入 contact/review/hold 结构化动作；默认 v1 及既有规范化结果不变，继续复用同一行/批量策略执行器。业务政策提供币种单位、队列/渠道、时区时段、主体窗口频次、最小间隔及批次预算。联系许可未知、历史覆盖不足、迟到记录或效果未知都不能静默当成可联系或零历史；同一主体的多个案件共享预览频次，优先级和案件 ID 决定稳定分配顺序。预览明确不代表实际授权或实时容量预留。110 项动作与 DSL 检查通过，扩大至现金流/候选/既有 DSL 交付后 203 passed（9.53 秒），Ruff/diff 通过。日志 `/Users/eddyz/.codex/worktrees/audit-fit-provenance/collection-planning-regression.log`，首次误写的不存在测试路径日志仍保留。

审批、实际受治理执行、取消/重启与资金反馈、Agent/界面和独立复核继续开发。通用候选设计仍拒绝 collection，而不是把分群改名充当催收；完成专属候选和闭环后再开放入口。没有真实联系客户、发布或机构部署。冻结 17 个工作包和 22 项发现的要求保持不变，以上均为部分软件进展。


## WP12 历史回放工作台与交互验证

`ab87745d` 接入任务内历史回放、明确字段时点与可选业务假设、独立时间窗、提案、原有 manual/Agent 计划、带理由人审、冻结结果和成熟资金对账。修复手动“开始执行”误发 Agent 接口、刷新并行计划 GET 的伪失败以及上一动作收尾时误点 gate。138 项相关测试通过；8 个提取 app 函数的旧测试夹具缺新增 controller stub，补齐后 401 项 driver/UI 回归通过，生产代码未加测试绕过。实际标准 CLI + Python Playwright 在 1440/1600/1920 验证手动和 Agent 两种模式，用实际原生 LR 与两套冻结策略回放；策略 validated 状态仍是明确合成夹具。三种导出读取数值正确、对账不改原回执。证据 `/Users/eddyz/.codex/work-artifacts/historical-ui-20260928/run-1790603565/proof.json` 与 `run-1790603958/proof.json`。root 查看实际 results-1600 截图后要求率使用百分比、差值使用百分点、来源 ID 折叠展示，已窄修。

## WP13 来源查询和事件工作流实际接线

`e38e20e2` 接入独立本地 HTTP reference provider 和历史导入；查询意图先冻结，来源授权和真实用户角色由服务端复核。超时/进程丢失后只对同请求 GET 读回，不能重复 POST；全部 HTTP 尝试统一限流/熔断，原始响应与规范响应分别校验字节指纹。23 项 source 检查、134 项既有 CLI/插件/模板检查和标准 CLI→HTTP→受限 ToolRunner 通过。root 审阅发现大表读完才限行数，以及撤销后仍能返回完整业务数据，`d0fc4fb0` 改为绑定行数读前拒绝、仅投影两列，并在业务证据返回前再次复核授权；失效授权只保留审计收据和 unauthorized 状态。4 项相关反例通过。证据 `/Users/eddyz/.codex/work-artifacts/operations-runtime-20260927/risk-sources/`。

`b1b86aa2`、`00ccdee3`、`10b8a193`、`b341f90e` 将事件请求接入真实 HTTP、原有必需人工 gate、ToolRunner、签名产物及授权回放/export。`4367c53d` 还修复直接删活任务事件导致完整覆盖下假零的问题：6 个删除反例先失败，再验证仅允许整个任务级联删除。95 项事件及 2 项原来源 worker 兼容检查通过；证据 `/Users/eddyz/.codex/work-artifacts/event-runtime-20260928/proof.json`。事件特征与冻结包/在线评分器、界面和 H/P 继续完成。

## WP14 提案认证、载体修正和独立审阅

`c00635ab` 统一工作区相对产物路径，使对账可以走共享 listing/download；保持活任务证据不可直接删除，但整任务删除可以级联；时段 end 支持 24:00，能够明确拆分跨午夜政策。真实 API 下载和午夜反例先失败，修复后的 collection/artifact/task 联合 201 passed（52.09 秒），日志 `collection-carrier-{red,green,regression}.log` 位于 `/Users/eddyz/.codex/worktrees/audit-fit-provenance/`。

独立审阅复现了历史 attempts.case_id 与本批已知主体矛盾时，频次被计给错误主体的问题；`ebae19ae` 直接拒绝冲突，保留批外案件的明确来源声明，审阅者独立确认反例被拒绝。`33b9da67`、`9fc4ded6` 冻结本地参考批次提案，核对真实 maker、同任务案件主体、金额单位、开户时点和源文件字节，HMAC 绑定不可变请求和同一个预览指纹。9 项新提案测试及 collection/DSL 联合 124 项通过；其 scope 只到提案，当前没有审批即执行的入口。受治理队列/参考动作执行、取消/重启及原生效果收据接入继续开发，禁止真实对客发送。

主线 `9fc4ded6` 对来源、事件、collection、历史 UI、插件与并发快照做联合检查：248 passed（101.90 秒），仅既有 Starlette 弃用提醒；证明 `/Users/eddyz/.codex/work-artifacts/audit-integration-20260928/event-collection-ui-proof.json`，所有检查均以明确合成数据运行。未把局部通过计为全量 readiness 或独立真实 Agent/历史/生产验收；17 个工作包和 22 项发现的冻结要求没有删减。


## WP01 多任务运行基准与真实模型失败定位（2026-09-28）

`13cad66d` 将组合设置合同和结果拒绝接入同一真实 HTTP 旅程，增加“最新助手消息”与“指定工具确未执行”的严格判据；缺少计划、缺少工具或未知输出不能证明未执行。覆盖表按实际 task_type 列出缺失场景，默认原两用例生成器保持兼容。`bd257425` 使存在多个目标候选的建模/特征任务保留 C1 选择问题，禁止自动“确认”覆盖为错误。`a69b65dc` 修正旧测试把业务证据不足等同执行失败的断言，保留原阈值和业务失败检查。

8 项重点测试通过；扩大回归是 213 passed / 1 个旧断言失败，修正后该项单独通过，不能写成同一次 214 项全过。实际 deepseek-v4-flash、冻结 9 用例、源版本 `1ed8df96` 的运行通过 7/9：Vintage 未创建计划，组合拒绝用例在开始授权前停止；失败记录保持。两用例诊断为 1/2，之后 Vintage 响应形态诊断为 0/1，均不替换 9 用例分母。公开合成响应证实部分语义结果缺必需 reason，第二条材料确认的两遍置信度也不一致。JSON 传输成功不等于平台语义校验通过；正在修复 schema 传递及补充脱敏决策观测，未降低授权门。

证据根 `/Users/eddyz/.codex/work-artifacts/runtime-family-20260928`；原始 9 用例报告为 `real-model-runs/20260928T151133954431Z-6bcbf70f7cfa/report.json`。历史两用例曾记录的系统临时目录当前不存在，旧报告结果继续保留，但不能把旧原始档案称为当前可核验证据。新日志和报告均放在持久工作产物目录。完整工作流正常/澄清/拒绝/恢复、独立隐藏集和受信关闭适配器仍待完成。

## WP09 构包输入发现

`f2972f72` 从原生签名收据核验模型、成员、校准和预处理字节，提供构包所需原始字段与可用分数产品；发现接口不执行 pickle。字段类型与可空性仍需明确声明，旧无来源证据模型要求重新训练。6 项测试在分支和主线分别通过；不把发现成功等同构包、部署或生产 readiness。

## WP14 实际参考执行、历史保留与请求限流

`031d1dcf`、`ca074628` 把签名批次接入既有 Workflow、人审、Approval、Invocation/Effect 和实际 SQLite 预留/分配，完成或取消走独立批准，原始 producer 可读回及导出，后续资金结果独立导入对账。跨任务主体频次和队列容量使用同一工作区资源；未知结果不盲重试。review 未提供费用时保留 None 并计入未估算项，实际费用不能用预估替代。所有动作明确为 local_reference，不发送短信、电话或任何真实客户接触。

独立审阅发现整任务删除可抹掉历史并重置频次/容量，`b2fff15b` 对已有队列记录的任务禁止删除，并只将该精确保留错误映射为 409；完成/取消后仍需保留历史。仅提案的任务可照常删除。此前段落中“整任务可级联”的描述只代表当时未执行提案边界，不适用于现在已有执行历史的任务。`06a35bef` 补真实 HTTP→人审→ToolRunner→资金对账和重启读回，还将 16 MiB 请求限制前移到流式读取过程，缺失或伪造 Content-Length 均不能绕过。

分支 300 项相关检查及最后 8 项 HTTP/流式检查通过；主线 `06a35bef` 的 collection/task/readiness/plugin 联合 111 passed（97.50 秒），仅既有 Starlette 弃用提醒。证据 `/Users/eddyz/.codex/work-artifacts/collection-execution-20260928/proof.json` 和 `/Users/eddyz/.codex/work-artifacts/runtime-family-20260928/main-joined.log`。界面、专属候选和完整业务闭环继续开发，H/P 没有据此通过。17 个工作包及 22 项发现要求不变。


## WP01 语义传输、失败观测与材料确认上下文（2026-09-29）

`74143c94` 把严格 JSON schema 同时传递给仅支持 JSON object 的 provider；独立审阅发现 plain-text profile 兼容回归，`128b1d0a` 保持原 response_format 行为并仅在提示中附 schema，67 项传输检查通过。`909e9d68` 将当前用户授权语义的置信度与业务质量、后续成功率分开，70 项离线检查通过；固定公开 10 项授权诊断由 9/10 到 10/10，七个否定/询问/条件/注入反例仍阻断。该小诊断不是整条 Agent 验收。

`6f386a86` 仅持久化两遍语义判断的白名单枚举、严格布尔值和有界解析位置，不保存原始响应、理由、引用或业务参数；`839366b4` 让基准从助手 metadata 再次净化后读取这份观测，不改变评分或执行判据。603 passed / 1 个可选真实模型检查跳过，联合 152 passed，基准回执 21 passed。固定九用例在 schema 修复快照 `0f7f07e1` 为 8/9，在提示/观测快照 `f3032362` 仍为 8/9；前者组合开始授权失败，后者 Vintage 材料确认失败，所有失败档案保留。

Vintage 平台原本明确要求回复“材料已上传”，但路由只收到 phase，没有收到正在等待的问题。`9b8cfad2` 传递固定平台问题、已选择的合法分析类型及标签语义，不传用户原始范围/样本；63 项检查通过，四类独立复核危险标记仍阻断计划创建。固定九用例的新真实模型运行进行中，预算、期望和分母均未改变。

证据根为 `/Users/eddyz/.codex/work-artifacts/runtime-family-20260928`，新增报告分别在 `schema-fixed-real-model-runs/20260928T153204895840Z-c7c5130f0110`、`semantic-fixed-real-model-runs/20260928T154804487908Z-d6e189ef5692`。真实模型通过不自动等于隐藏集或业务验收；缺失价格表时费用仍未知。

## WP09/10 纯规则包及模型部署工作台

`28978311` 引入明确的 rule_only v2 包，模型/分数产品为 None，文件和模型成员为空，不执行反序列化；沿用现有策略 DSL、restricted worker、账本、治理和 CAS。原模型 v1 manifest 与哈希契约保持。`0abffd6a` 从经过认证的策略版本发现纯规则原始字段，不要求模型 readiness。37 项规则/服务/治理检查与 38 项兼容检查通过；storage 检查为 141 passed / 1 个旧迁移夹具失败，`39b2c7df` 修正为实际 pre-18 数据库后单项通过。

`f6c24780` 增加模型包原生字段发现、明确类型/可空性、构包、maker/checker/admin、安装探针和回读、激活、幂等决定、旧 head 冲突保留输入、生产回滚及 shadow 入口。364 项前端、51 项后端/UI、9 项重点检查通过；真实浏览器在 1440/1600/1920 完成八条旅程，无页面错误。证据 `/Users/eddyz/.codex/work-artifacts/production-ui-20260928/run-1790610032/proof.json`；实际查看两个宽度的构包与治理截图。审阅发现发布拒绝和 shadow 回滚在后端也缺失，已经继续实现，不能据已有界面关闭整个 WP09/10。

## WP12 纯规则时间窗口与 WP13/14 原生证据

`4ded3db1` 使纯规则包的历史窗口严格校验明确为空的 score/score_product、成员与包身份，真实计算动作 PSI 和通过率变化；分数 PSI 记录为不适用，部分证据不能判为通过，动作漂移仍失败。原模型缺分数仍拒绝。88 项联合检查和 61 项最终时间窗口检查通过，覆盖真实 HTTP→人审→ToolRunner→签名回执及 JSON/XLSX/DOCX 导出。格式化造成的单个 Ruff fixture 注释位置问题由 `c8721c07` 修正后检查通过。

`c326cd6e` 新增轻量事件读取授权，仍检查实际主体、来源和当前 grant，原生事件到包/worker 的完整绑定继续开发。`5407a443` 统一 collection 结构化动作与 repository 核验。所有执行仍限定本地参考环境，没有机构或客户接触。17 个工作包、22 项发现与 S/H/P 要求保持不变。


## WP01 固定九例通过与正常建模预算失败（2026-09-29）

材料上下文修复快照 `5eb4ce76` 的第四次固定九例运行通过 9/9，报告 `runtime-family-20260928/intake-context-real-model-runs/20260928T160127635130Z-4c5798dca411/report.json`。原预算、期望、分母没有变化；四个无需模型调用的澄清用例不计 A 证据，公开合成例整体仍标记 acceptance=not_established。

`35f68b5b` 增加独立正常建模案例，当前展示中唯一推荐实验通过现有人审快照选择，不能任意传 ID、route 或 selector。46 项 runtime 和 9 项传输复核通过；原两例/九例所有案例、期望、数据字节与基线相同。新 cases SHA 为 `e65ab52e191d0a9b5ce9d03cd2c05032ee771cf6c8f1346ddc30972c14cc91b8`，expected SHA 为 `5f02beff0c886ee6bb22f3e78e1f6bf7a1cd7ec612ee8ccf425a91b9292cc485`，在 `normal-modeling-20260928/frozen-public-suite`。

新案例真实模型首轮源 `f3b9c423` 仍为 0/1：12 步 done、13 条功能断言全过，但第 17 次 critic 请求/转发上限 2048，provider usage 回报 completion_tokens=2049。因此严格预算门判失败，未调整预算、覆盖使用量或重复运行直到通过。242.822 秒、28 次模型请求、162 次 HTTP 未超各自上限；13/28 响应在输出限制时无 content，是另一个效率诊断。原报告及诊断在 `normal-modeling-20260928/real-model-runs/20260928T161426771028Z-d2852a3faa2f/report.json` 和 `real-model-first-diagnostic.json`。所有证据根均为 `/Users/eddyz/.codex/work-artifacts`。

## WP07 字段级时点证据传到建模产物

`04673717` 复用既有 as-of、preprocessing 和产物登记，只给原生 as-of 真正选择的带前缀字段认证，决策锚表自带字段继续未知。逐行确定性派生保留输入证据，未记录参数可得时点的拟合变换继续未知；train-only 是拟合成员证明，不能冒充历史时点证明。建模投影比较实际列值及行序，原始时点链无预处理步骤时也能保留；tune/train 的证据经过平台参数剥离后附回模型产物，模型卡 JSON 和 Markdown 如实呈现证据或缺口。

独立审阅在真实 ToolRunner 复现 WOE 覆盖已有同名列却仍 verified 的问题；先保留红例，再按输出写集覆盖已有字段证据，numeric/categorical WOE 反例均通过。相关 71 项、交付/runtime 31 项、最后 as-of/字段证据 29 项通过，Ruff/diff 通过。初轮 5 passed/1 错写测试工具名失败和 WOE 红例日志保留。证据 `feature-time-20260929/{initial,collision-red,related,delivery,final}.log`，独立复现目录 `feature-time-review-20260929`。普通清洗/JOIN、拟合参数时点、业务标签和真实历史数据仍未整体关闭。

## WP13 原生事件进入同一个受限决策 worker

`254c305d` 将冻结事件 recipe 接入模型/纯规则包，逐次申请提交完整原生证据引用，主体身份来自实际 session。主 HTTP 在幂等读前检查当前来源授权，准入后 worker 校验签名收据、来源 hash、完整主体/时点合同，并重放原生事件；不能外填派生事件值。未知覆盖按包内 review/reject 降级，安装审计只暴露执行状态/hash，避免给无来源权限的人展示主体评估。

94 项联合检查和最后 2 项真实 HTTP→人工门→ToolRunner→安装激活→决定→CLI 重启→撤权检查通过，日志 `operations-runtime-20260927/reference-decision/event-binding-{final,last}-tests.log`。旧无事件包/请求哈希契约保持。历史批量逐行事件绑定继续开发，未把本地决定称为机构部署或生产 SLO。17 个工作包和 22 项发现未删减，整体保持进行中。

## WP07 清洗和受治理训练共享真实成员边界

`9fa64ed4` 认证既有清洗 producer 的不可变 run、产物、源表和结果绑定，重命名、筛选、转换、去重和逐行派生保留适用的字段时点；填充值没有参数可得时点则保持未知。46 项联合检查通过。`c25441a0` 修复另一训练入口遗漏：受治理 Strategy 训练也用原生 V2 全表成员掩码校验历史预处理拟合成员，不能用源表的 train 标签代替。全量拟合和“源 train 实际属于 V2 验证集”两例先复现成功训练，再验证修复后拒绝；无原生证据的输入不升级为已认证。

新增证据读取曾重新绑定 source_path，导致已冻结样本设计失效；改为保留原文件绑定的认证快照。独立审阅指出临时整表读取会破坏宽表内存边界，最终复用 retained-descriptor 校验读取 Parquet schema，仅投影比对时加载相关列。136 项快照、清洗、时点、拟合成员和原生训练检查通过，Ruff/diff 通过。日志 `feature-time-20260929/{cleaning-joined,governed-red-fixed-fixture,governed-fixed,governed-joined,governed-schema-final}.log` 保留真实失败和修复。普通 JOIN、拟合参数历史时点及真实历史业务验收仍开放。

## WP09–10 纯规则界面、拒绝和影子回滚

`82811b19` 补齐 checker/admin 的终止拒绝与理由审计；拒绝后不能安装、激活。影子与正式回滚绑定各自当前指针和前驱，不互相改变服务。36 项相关检查和主线事件/治理 45 项通过，真实 CLI、原生模型和浏览器十项旅程通过。`0276fd99` 增加纯规则构包，明确字段类型与空值政策，默认动作策略可用空请求合同；全程无需发现模型或请求模型 readiness，score=null 显示不适用。26 项相关检查、真实纯规则构包→审批→worker 安装→激活→决定通过，晚到旧 readiness 不污染切换后的类型。

实际 1440/1600/1920 截图及证据在 `production-ui-20260928/run-1790612756/proof.json`、`rule-run-1790613639/proof.json`，模型兼容复核在 `run-1790613786/proof.json`。使用本机 Python Playwright 承载真实浏览器，未使用不可用的 Browser plugin；仅证明本地参考运行。

## WP12–13 历史事件逐行回放与发现的下载旁路

`3dcc7f83` 将每行原生事件引用与独立主体命名空间/token、决策时点和冻结包 recipe 对齐。HTTP 真实 session actor 写入签名提案，工具仅执行已审核提案；准备、执行、详情和导出重查当前授权。缺引用、主体/时点/hash 错误、撤权及篡改阻断整批；只有认证后的未知覆盖走包内 review/reject，保持原分母，时间稳定性明确未知。97 项联合检查通过，JSON/XLSX/DOCX 实际文件解析通过。日志在 `operations-runtime-20260927/reference-decision/event-batch-20260929/pytest-run1.log`。

来源受限的批量回执由 domain 路由检查权限，两个通用下载路由对无角色、跨 actor、撤权均拒绝。继续沿源链检查却复现既有 `risk_event_features` 原生收据经通用产物路由撤权后仍返回 200；已分配独立权限修复，不能将批量回执私有目录视为该旁路已经关闭。

## WP01 正常策略真实模型首轮失败保留

`5004a46e` 增加独立正常策略 case，通过明确人工样本选择/语义输入、原生 workspace CAS 和既有采纳审批；旧两例、九例、正常建模的 case/expected/数据/预算字节全部保持。70 项 runtime 检查通过，真实 HTTP fixture 两计划九工具和文档交付完成，拒绝采纳时采纳/文档工具均不执行。冻结 cases SHA 为 `1ec965a101a39a82f8673a89307815be50415ff1b50d8a64e48e390fcdba45a3`，expected SHA 为 `65298e9bce23f83906f91ebfeeae68f9ab3371e894790af3df3ef81043d6328e`。

真实模型源 `4de23581` 首轮 0/1，20.979 秒、4 次模型请求；两条明确的策略请求均在独立 reviewer 被标记 requests_change=true，router 则为 false，安全门阻断，零 Plan。报告在 `normal-strategy-20260929/real-model-runs/20260928T165503740738Z-a46a47ac1af9/report.json`。独立固定十例语义诊断同源 7/10，三个肯定请求全失败，七个否定/条件/修改/注入反例均拒绝；该诊断不替换正式 case 分母。正在修复意图定义，失败记录与预算保持原样。全部证据根为 `/Users/eddyz/.codex/work-artifacts`。

## WP01 正常策略复跑通过；WP09 分数方向与 WP13 权限修复

`1df30c5b` 将两遍独立意图判断的标志定义统一到同一文本合同，区分首次声明工作流约束与修改既有口径；问题、未满足条件、拒绝和真正修改的标志仍逐项拦截。53 项离线语义/JOIN/提示词版本检查通过。相同十例真实模型诊断在 `a1854755` 从 7/10 提升至 10/10；相同正常策略完整 case 随后通过 1/1，146.993 秒、18 次模型尝试、43 次 HTTP、九工具 done、13 条断言全通过。报告为 `normal-strategy-20260929/intent-corrected-real-model-runs/20260928T170733449072Z-b43d73fdb111/report.json`。7 次输出限制空响应和 4 次重试仍是效率问题；未配置有日期来源的价格表，费用未知。原九例跨流程回归正在运行，公开合成案例不等同隐藏或业务验收。

`7371a761` 让原生样本的历史分方向进入 tradeoff/cutoff 消费端，显式相反方向即使携带确认标志也不能覆盖冻结合同，旧 V1 默认保持。91 项策略检查、原低风险方向 HTTP 旅程通过；高分高风险真实 HTTP 链九工具与 15 断言通过。证据 `normal-strategy-20260929/direction-{regression,low-risk-compat}.log`。主线建模证据后续消费及治理界面另通过 30 项联合检查，见 `feature-time-20260929/native-delivery-joined.log`。

`f3f93d51` 封堵刚复现的通用产物下载旁路：按原生产物 kind+origin 识别来源权限，所有同文件登记别名同样受限，通用下载/预览拒绝并指向既有 domain 证据入口。两项真实 HTTP 红例先确认 200 旁路；后续 43 项相关检查通过，最终两项权限例通过，共验证 44 个不同用例。SourceService 原 maker 撤权后保留不含 normalized_source/assessment 的审计回执，沿用已有 unauthorized 语义；没有把这种审计响应改成泄露敏感字段。日志在 `operations-runtime-20260927/reference-decision/event-batch-20260929/artifact-acl-{before,after,final}.log`。未声称全应用 task ACL 已完成。

## WP01 固定九例回归完成与审查输出契约修复

源 `a1854755` 的第五次固定九例运行通过 9/9，案例、期望、预算和分母均未变。5 例有真实模型与工具联合证据，4 个零模型调用澄清例不计 A；仍为公开开发集，acceptance=not_established。报告 `runtime-family-20260928/intent-v3-real-model-runs/20260928T171200197862Z-1034e5eadb15/report.json` 保留完整运行事实。

代码检查发现步骤审查与最终总结共用仅要求 passed/reasons 的提示词，首轮请求未提供两者各自的结构，错误类型还可能被拆成字符列表。`e9109ecb` 从首轮开始传独立 typed schema，并在解析端拒绝类型错误；最多重试一次，保留旧最小 summary 返回与未配置模型的手动模式。CRITIC_SYS 升至 v2，LLM 仍只能提供解释，不能改写确定性业务验收。17 项新增测试先红，修复后 228 项审查、完成协议、执行器、业务验收及 LLM 传输检查通过，Ruff/diff 通过。首次扩大回归因 basetemp 父目录缺失产生 109 项 setup error；修正运行环境后全过，未把环境失败隐去。日志 `/Users/eddyz/.codex/work-artifacts/reviewer-contract-red.log`、`reviewer-contract-green.log` 和 `reviewer-contract-20260929/green-run2.log`。

独立正常建模案例正在源 `e473f53a` 复跑，未改原案例、预算或 expected；此前 provider 上报 2049/2048 token 的严格预算失败继续保留。主线权限修复另通过 20 项事件/批量回放/来源权限检查，日志 `operations-runtime-20260927/reference-decision/event-batch-20260929/mainline-acl.log`。

正常建模原案例在 `e473f53a` 通过 1/1，12 步 done、13 条断言过，178.146 秒、24 次模型尝试和 127 次 HTTP 均符合原预算。6 次输出上限空响应仍记录；两次运行不能证明普遍性能提升，首轮 0/1 不覆盖。报告 `normal-modeling-20260928/reviewer-schema-real-model-runs/20260928T172915986322Z-53029500cbe3/report.json`，两轮绑定比对 `reviewer-schema-diagnostic.json`。该源码正在执行原九例跨流程回归。

实际 attempt receipt 未带提示词 name/version 的既有缺口由 `1b786c6c` 补齐，首轮和重试均引用注册值，42 项审查/提示词检查通过；这项 telemetry 补充没有合入正在运行的冻结源码，未倒填旧回执。

## WP07 普通 JOIN 来源证据与 WP14 执行界面集成

`246a548f` 修复空表资料重复反序列化时 NaN 不等引起的身份误判，仅双方对应 NaN 视为同一未知值，完整字段和文件认证仍校验。`6fc5154d` 从同一物化 JOIN 记录实际左/右成员、双亲 hash 与碰撞后列名，在同一产物事务中登记；左字段须核验复制值才能继承时点证据，右字段及整表 PIT 状态保持 unknown。复用原去重、规范化、1:1 行数和 source_path 绑定，未改变拟合或标签证据。旧注入式 registry/repository 兼容路径保留，但不生成原生时点证明。96 项执行、53 项时点/快照、32 项原生/身份以及最终 5 项事务负例通过；失败原记录保留于 `/Users/eddyz/.codex/work-artifacts/wp07-join-*.log`。

`e0887ad9` 增加催收任务工作台的批次发现、冻结预览、现有 Plan 人审、独立 checker、排队/本地参考执行/取消及原生回执 JSON 导出。真实 CLI 浏览器在 1440/1600/1920 验证完整链；根代理回看完成态截图，实际成本未知、未联系客户表达清楚。344 项前端与 8 项专项检查通过；主线 JOIN/训练证据/催收 UI 联合 111 项通过，改动 JS 的 node 检查通过。证据 `collection-ui-20260929/run-1790616614/proof.json` 与 `reviewer-contract-20260929/main-joined.log`。数据准备和常用政策业务表单继续开发，WP14 未闭合。

审查输出契约两个提交经独立只读代码审阅，未发现本次引入的可复现回归或确定性验收绕过。`e473f53a` 的第六次固定九例再次通过 9/9，5 个真实模型候选与 4 个零模型澄清例分别记录，报告 `runtime-family-20260928/reviewer-schema-real-model-runs/20260928T173258834913Z-196fd869b4d6/report.json`。正常验证将分别覆盖标准 V2 Agent 与兼容 Workflow，不能用不同入口的成功替代标准入口验收。

## WP02 特征输入重复实现收口

源码 AST 扫描确认 Feature 与 Modeling 中三类 helper 共六份函数体完全相同。`d471b1a3` 将它们收口到共同的 pack 输入适配层：特征列表去重展开、被自动筛除的类别列提示、数值类别编码提示；保留旧导入名，算法内核仍在 feature 层，未合并两包不同的指标/错误语义。63 行新增、129 行移除，净减 66 行。6 对函数体排除文档字符串后 AST 一致，证据 `reviewer-contract-20260929/feature-dedup-body-equivalence.json`；190 项真实能力包、候选列和筛选检查通过（398.63 秒），scoped Ruff/diff 通过，日志 `feature-dedup.log`。未将仅扫描命中或动态入口误判为僵尸代码，剩余大模块和兼容边界仍需复核。


## 2026-10-03：验证报告、原始回执权限与催收配置

本节补齐上一批实现至当前主线 `29b46a6d` 的实施事实。下面均为开发检查，17 个工作包和 22 个发现仍未整体关闭；冻结要求 SHA 保持 `05587bd02aec550400a9a6d4ad726766a3ddbf67fe204a2de28c974e26f8cc6b`。WP08 以及已有实现的 F06–F11 状态由 planned 校正为 in_progress，未降低任何验收要求。

| 实现 | 结果与边界 |
| --- | --- |
| `95919b83` 样本设计重复函数 | 三份同义读取逻辑回到现有 sample-design owner，旧私有名称保留；净减 51 行，144 项相关检查通过。 |
| `447317cc`、`c6962853`、`02fd5d70` 原生结果恢复 | 事件、历史批次、现金流对账的原调用身份与产物同事务记录；只有原始回执可恢复，旧敏感结果缺授权不能补造成功。真实旧 schema 迁移和缺回执负例保留。主线联合 64 项通过。 |
| `8e678577` 当前读取权限 | 计划、历史步骤、消息、业务报告等载体复用来源授权检查；异步返回前重查撤权，原计划完成不回退、不重复执行。独立 10 项真实 HTTP 检查通过；不代表全应用 ACL 已验收。 |
| `f9250525` 时点证据消费 | 包准备保留 unknown / row-local / training-only 等原证据，不把未知预处理写成不需要；只认证 schema，不改写冻结数据集路径。原生 JOIN/训练/包准备/评分等 47 项通过。 |
| `c2bc227a`、`bc995da1`、`e6b138c4` 催收业务表单 | 来源、案件、排期、现金流、覆盖与成熟度、对账、常用政策、CSV 映射、类型化有序规则接入已有审批和执行链。原件映射明确、前导零保留、未知时间不填导入时间；分批失败保留成功项，只重试未确认项。复杂嵌套规则继续使用高级 JSON。 |
| `5bc1f299` 报告分段与版本保护 | 必需段落分开按 schema 生成，全部有效后才合成；V1 生成路径保持。可选字段省略时保留原草稿，显式空值与缺失分开。同步生成前捕获报告/草稿/任务状态，事务内比较并发布；并发保存、确认、生成和新任务均使旧结果返回 409。 |
| `29b46a6d` 未知业务定义 | V2 prompt 升至 v2，移除默认 DPD30 和从模型名推断 MOB 的旧规则。明确缺少口径保持未知，不能用假设填业务定义；这不是对模型永不幻觉的保证。 |

分段生成初次广泛检查 100 项通过；独立审查复现了可选字段被清空和同步生成覆盖新编辑，已修复。新 provider 适配测试曾因错误要求 JSON 对象键顺序导致 2 失败/55 通过，修正为字段集合后 43 项通过，没有放松实际字段/类型/schema 检查。同步竞态测试修复前 9 失败，最终 23 项及一项旧接口检查通过。最终联合 validation/transport/Agent API/draft 回归 **281 passed**；合并权限、催收规则后的联合 **88 passed**；提示词 v2 与分段检查 **21 passed**。这些存在重叠，不能相加为独立测试数量。

主线日志：`/Users/eddyz/.codex/work-artifacts/report-sections-20261003/{regression,revision-regression,transport-regression,joint-regression,final-integration}.log`。同步竞态红/绿及最终证据为 `review-v2-draft-race-{red,green,final,api}.log`；独立访问权限日志为 `review-access-20261003/native-read-scope.log`。

催收规则独立审查指出 missing=error 在 AND/OR/first-match 短路下并非无条件阻断，已修正文案并增加真实 HTTP 回归。最终 38 项专项检查通过；早先联合运行 91 通过、1 条测试错误断言已修复并重新覆盖，原失败不删除。原生 CLI 与 Playwright 完成 10 条业务旅程、三种桌面宽度、零 page error；root 查看了 1600 规则编辑及 CSV 部分失败截图。证据为 `collection-ui-20261003/rules-run-1790985645/proof.json`、`collection-ui-20260929/csv-run-1790622070/proof.json`；独立短路复现为 `collection-rules-review-cck2evnq/short-circuit-review.json`。这些是公开合成、本地参考执行，没有机构触达或真实回款。

### 正常验证的两条真实模型入口

兼容 Workflow（`26bb7888`）与标准 V2 Agent（`a83a9e9f`）分别计量。兼容入口实跑 Notebook 模型分与 PMML 分一致性；V2 走原生任务 job、静态 Notebook 字段识别、PMML 全量评分和报告草稿确认，没有伪称经过 Plan/ToolRunner。两个入口都验证实际 Word/Excel 文件与下载 hash。

兼容 r2 固定公开用例真实模型 **1/1**，报告 `normal-validation-20260929/compatibility-r2-real-model-runs/20260928T182720934263Z-fb9c1583cb71/report.json`。标准 V2 r2 首次 **0/1**，报告 `v2-agent-r2-real-model-runs/20260928T185903844319Z-32fde5fbb90b/report.json`：PMML/指标完成，报告草稿失败，provider 上报一次 2049 token 超过 2048 冻结上限，严格算失败，没有截去 token 或放宽预算。此时实际存在长报告单次 JSON 请求截断。该失败记录保留，分段修复后的同例实跑待追加。

V2 r2 cases/expected SHA 分别为 `730b2846377ff4603eb5c566704b7edc4e6a48ee72956303ddc303baf2bc4613` / `5eeaef634993bf31830104df983cdfaad0cb4879d0243801cfd1f7fe76c1ed51`，本次再次读回一致。公开开发集不是最终隐藏验收，所有报告 acceptance_claim 保持 not_established。当前仍未收到真实脱敏数据目录，也未具备机构生产验收条件。
