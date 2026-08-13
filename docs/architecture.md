# MARVIS-Agent 架构文档（代码级）

> 本文是 90 天计划条目 **B-13**（降 bus factor）的产物。目的：让新读者不看人肉解释、只读代码就能理解系统如何组装、各模块的职责边界、一条 workflow 如何从自然语言走到报告、以及哪些兼容边界与信任机制是不可破坏的。
>
> 写作原则：**所有结论来自对代码的实际阅读**，不写营销语言；代码里没有明确依据的地方标注「未核实」。范围清单、产品阶段、术语口径以 [roadmap.md](./roadmap.md) 为准，本文不复制。执行排序以 [2026-08-13-next-90-days-development-plan.md](./superpowers/plans/2026-08-13-next-90-days-development-plan.md) 为准。

### 0.1 运行时术语（先对齐词汇，否则下面的边界读不懂）

本项目统一使用（AGENTS.md 定义，代码内一致沿用）：

- **Plugin**：可安装或内置的能力包（`marvis/packs/*` 即内置包，每个有 `manifest.json`）。
- **Tool**：Plugin 内可调用的具体动作（`tools.py` 里以 `tool_*` 命名的函数）。
- **Hook**：Plugin 内由平台事件触发的动作（`plugins/hooks.py` 的 `HookDispatcher`，事件如 `CONSOLIDATION_TRIGGERS`）。
- **Workflow**：Agent 生成、平台内置、或用户可编写的 Tool/Hook 工作序列（模板）。`orchestrator/templates/*.py` 是内置模板，`orchestrator/templates/skills.py` 加载用户可编写模板。
- **Skill**：SOP/Playbook/方法论型知识，落地为「用户可编写的 Workflow 模板」——声明式、只编排已信任工具、实例化后过 `PlanValidator`，**不另立 runtime**。

执行面统一是 Plugin / Tool / Hook / Workflow 四者；没有独立的「skill runtime」（历史文档里的旧称已被上述口径取代）。

### 0.2 关键环境变量（本地访问与隔离的开关）

这些环境变量在 `app.py` / `pipeline.py` 中硬编码读入，是理解「本地优先」与「隔离」边界的钥匙：

| 变量 | 作用 | 来源 |
|---|---|---|
| `MARVIS_ALLOW_REMOTE_READ` | 允许远程只读 API（默认关，非安全方法始终 403） | `app.py:_remote_read_enabled` |
| `MARVIS_TRUSTED_PROXY_HOSTS` | 信任的反向代理地址（转发头只信任显式代理） | `app.py:_trusted_proxy_hosts` |
| `MARVIS_LOCAL_TOKEN` | 共享主机加固：本地读写需 Basic / `X-Marvis-Token` | `app.py:_configured_local_token` |
| `MARVIS_ALLOW_LEGACY_LIVE_NOTEBOOK_EXECUTION` | legacy live kernel 三重 opt-in 之一 | `pipeline.py:LEGACY_LIVE_NOTEBOOK_ENV_VAR` |

---

## 1. 运行时栈（入口与启动）

运行时栈是「FastAPI 组合根 + 两份持久化（SQLite / DuckDB）+ 隔离子进程 + 前端单体」。启动链如下：

```
python -m marvis            → marvis/__main__.py 的 main()
  └─ serve（默认命令）       → _serve() → configure_logging() → create_app() → uvicorn.run()
```

### 1.1 入口与启动文件

- `marvis/__main__.py`（644 行）——CLI 入口。子命令：`serve`（默认，也是无子命令时的行为）、`validate`（直接跑 V1 验证流水线）、`update`、`version`、`eval-llm`、`backup`、`restore`。`--profile` 通过 `_profile_defaults()` 映射到 host/port/workspace（默认 `127.0.0.1:8000` + `./workspace`；`dev`→`:8001`；`vN-M`→按版本号算端口）。当检测到当前是 conda base 时，`serve`/`validate` 会委托到专用 conda 环境（`_should_delegate_to_dedicated_conda_env`）。`_serve()` 最终 `uvicorn.run(app, ...)`。
- `marvis/app.py`（903 行）——`create_app()` 是全局组合根：`init_db`、各类启动 GC/恢复、`_configure_plugin_runtime()`（组装 PluginRegistry / ToolRegistry / ToolRunner / HookDispatcher / AgentMemory 调度器）、`_configure_orchestrator()`（组装 PlanValidator / Planner / Reviewer / IntentRouter / SubAgentDispatcher / PlanExecutor）、`reclaim_running_plans`、`sweep_heartbeat_lost_jobs`、`JobHeartbeatWatchdog` 等看门狗，注册 27 个 `include_router`，挂载 `/static`，定义 `/api/health`、`/`、`/branding/assets`，以及静态资源内容哈希版本化（`_static_asset_version` / `_static_import_map`，缓存键覆盖全部 `*.js`/`*.css`）。两道 `http` 中间件：`_local_access_guard`（本地/远程访问边界 + 共享主机 token）与 `_static_cache_control`（`?v=` 命中则一年 immutable，否则 no-cache）。
- `marvis/api.py`（186 行）——旧 `/api` 组合根。文档串明确说明：validation-agent 的 HTTP 组合逻辑已按 ARCH-1 迁到 `marvis/agent/validation_app_service.py`，本文件保留的只是兼容 re-export（大量 `_xxx = _vas.xxx`），供旧测试/扩展点导入。新代码应直接 import 叶子模块。
- `marvis/routers/`——24 个 router 模块（见 §2.4）。另有 `marvis/operations/router.py` 与 `marvis/production_governance/router.py` 两个不在 `routers/` 下的 router，加上 `api.py` 的 `/api` 组合根，`app.py` 共 `include_router` 27 次。

### 1.2 持久化

- `marvis/db.py`（47 行）——纯 re-export 门面：从 `db_schema` 与 `repositories/*` 转出 `init_db`、`connect`、`sqlite_health`、各 `*Repository`，保持 `from marvis.db import ...` 的历史导入面。
- `marvis/db_schema.py`（4,609 行）——SQLite schema 与迁移的实际实现（`SCHEMA_VERSION`、`_ensure_column`、`_MIGRATION_TABLES` 等）。表覆盖 `tasks`/`jobs`/`plans`/`plan_steps`/`datasets`/`strategies`/`llm_calls`/`audit` 等。
- `marvis/agent_memory_schema.py`——Agent 记忆的独立 schema，由 `db_schema` 在初始化时一并创建。
- `marvis/data/backend.py`——DuckDB 的 `DataBackend`（工作区级数据集后端，`duckdb_health` 等），每次操作新建连接而非共享默认连接。

### 1.3 前端

`marvis/static/`：`index.html` + 样式（`css/`、`styles.css`）+ 前端 JS。JS 分三层：单体 `app.js`（8,382 行）、第一轮拆出的 `js/`（31 个模块，6,409 行）、V2 拆出的 `js/v2/`（32 个模块，18,937 行）。`app.py` 的 `/` 路由把静态版本号与 import-map 注入 `index.html`，实现 cache-busting。

---

## 2. 模块边界

每节给出：**职责** / **关键文件** / **不该拥有的东西**（“不该拥有”多为 AGENTS.md 明确约束或代码文档串的自述，非我臆测）。

### 2.1 `validation/` —— 确定性验证算法

- 职责：模型验证的确定性计算内核：分箱（`binning.py`）、KS/AUC/PSI 等平台指标（`platform_metrics.py`，入口 `compute_platform_validation_results`）、PMML 打分（`pmml_scoring.py` / `pmml_score_artifacts.py`）、可复现性（`reproducibility.py`）、内存模型分（`in_memory_scores.py`）、压力测试（`stress_test.py` / `pmml_stress.py`）、Vintage（`vintage.py`）、结果类型（`results.py` 的 `ConsistencyStatus` 等）。
- 不该拥有：DB、FastAPI、Agent、任务生命周期（AGENTS.md 原话）。

### 2.2 `output/` —— 从结构化 payload 渲染文档

- 职责：把结构化 payload 渲染为 Excel/Word/图表。`excel.py`、`word.py`、`strategy_report_bundle.py`（策略报告四格式投影，见 §3）、`xlsx_safety.py`、`styles.py`。
- 关键约束：`strategy_report_bundle.py` 文档串自称「renderer，不是 analysis layer」，它校验自认证 bundle 后把**同一份事实**投影到 JSON/Markdown/XLSX/DOCX。
- 不该拥有：执行上传插件（AGENTS.md 原话）；不重算指标。

### 2.3 `pipeline.py` + `pipeline_*` —— V1 验证流水线编排

- 职责：V1 模型验证三阶段编排 `notebook → metrics → report`。入口：`run_notebook_stage`、`run_metrics_stage`、`run_report_stage`、`run_staged_pipeline`、`run_pipeline`。`PipelineSettings` 是 frozen dataclass（含 `notebook_isolated_execution`、`allow_legacy_live_notebook_execution`，见 §4.3）。
- 文件结构（文档串自述的 ARCH-6 拆分）：`pipeline.py`（2,324 行，阶段入口 + 被 monkeypatch 的名字必须留在这里）、`pipeline_errors.py`（`PipelineError`/`PipelineCancelled`）、`pipeline_cellgen.py`（注入到 Notebook 的 cell 源码生成）、`pipeline_io.py`（路径解析、样本加载、生成物卫生）、`pipeline_memory.py`（按 `auto_distill` 策略捕获成功/失败的 agent 记忆）。
- 不该拥有：本身不承载验证算法（调 `validation/`），不承载 HTTP。

### 2.4 `api.py` + `routers/` —— HTTP 层

- 职责：HTTP 路由、任务创建、active job、前端 payload。`routers/` 按域拆分：`data.py`、`tasks.py`、`plans.py`、`modeling.py`、`plugins.py`、`artifacts.py`、`agent_memory.py`、`validation_agent.py`、`validation_batches.py`、`validation_contracts.py`、`validation_stages.py`、`scans.py`、`reports.py`、`evidence.py`、`drafts.py`、`audit.py`、`llm.py`、`skills.py`、`materials.py`、`stage_controls.py`、`strategy_candidate_lab.py`、`branding.py`、`data_analysis.py`、`report_fields.py`。`app.py` 里与 router 并排的还有一批 `api_*_helpers.py` / `api_schemas.py`（78,995 字节，请求/响应 schema）支撑层。
- 不该拥有：验证算法（AGENTS.md 原话）；validation-agent 的服务逻辑应落在 `agent/validation_app_service.py`（ARCH-1）。

### 2.5 `static/` —— 前端

- 职责：根据 API payload 渲染前端状态。`js/` 是通用/第一轮拆分模块（`api.js`、`state.js`、`polling.js`、`metric-tables.js`、`render-agent.js` 等），`js/v2/` 是 V2 工作台模块（`plan_rail_controller.js`、`strategy_candidate_lab_*`、`data_workspace_*`、各 gate controller 等）。
- 不该拥有：不解析后端自由文本作为业务事实（AGENTS.md 原话）；状态渲染以 payload 为准。

### 2.6 `agent/` —— 意图、计划驱动与任务级 turn loop

- 职责：意图识别、解释、总结、报告草稿、PlanDriver 与任务级 Agent turn loop。
- `plan_driver.py` —— 「一个 driver 服务所有 V2 任务类型」的通用计划对话驱动：按模板+已填 slot 建计划，在真实 `PlanExecutor` 上跑，遇到 `needs_confirmation` gate 时把上一步刚算出的输出转成 append-only assistant 消息，**在 gate 步骤前暂停**，用户确认的正是刚跑完的结果。文档串自称「pure-ish」：改计划状态走 repo/executor，但**返回**消息而非持久化，便于离线单测。
- `turn_handlers/`（包，原 17,868 行单体的物理拆分，见 §6）——各任务类型的 turn 处理入口，按 workflow lane 拆为 17 个车道文件（`_shared`/`join`/`feature`/`modeling`/`vintage`/`portfolio`/`strategy_*`/`c1`/`typed_ui` 等）+ `_registry`，由 `__init__.py` 以 exec-merge 方式顺序注入单一命名空间（运行时语义与原单体一致，含 monkeypatch）。`DRIVER_AGENT_TASK_TYPES`（data_join / feature_analysis / modeling / strategy / vintage / portfolio）走 driver 化计划流，validation 走独立 scan/notebook 流。
- `strategy_request_compiler/`（包，原 14,449 行单体的物理拆分，见 §6）——策略自然语言编译器，按请求家族拆为 10 个车道文件（`core`/`sample_design`/`tree`/`cross`/`voting`/`pool`/`impact`/`scorecard`/`model_evidence`/`refinement_report`），同样 exec-merge 注入；公开面（`__all__` 29 名）与原文件逐名一致，见 §3。
- `strategy_workflows/`——标准策略 workflow 目录（`_catalog.py`、`_univariate_scorecard.py`、`_tree_workflows.py`、`_pool_workflows.py`、`_foundation_delivery.py` 等），提供 `resolve_strategy_request` / `prepare_strategy_plan`。
- `gates/`——gate 适配层（`adapters.py`、`contracts.py`），配合 `gate_execution_adapter.py`、`gate_param_schema.py`、`gate_response_adapter.py`。
- 其他：`service.py`、`validation_app_service.py`（ARCH-1 服务层）、`validation_service.py`、`validation_runner.py`、`validation_stages.py`、`semantic_authorization.py`、`instruction_router.py`、`renderers.py`、`presenters/`、各 `*_setup.py`（`strategy_setup.py`/`modeling_setup.py`/`feature_setup.py` 等）。
- 不该拥有：不执行确定性指标计算（指标归平台代码）；不绕过平台证据（AGENTS.md）。

### 2.7 `agent_memory/` —— 记忆的存储/检索/压缩/审计

- 职责：`store.py`（`AgentMemoryStore`）、`capture.py`（`save_memory_candidate`）、`retrieval.py`、`consolidation.py`（`ConsolidationScheduler`，由 Hook 事件触发）、`distillation.py`（`DistillationEngine`）、`evolution.py`（`EvolutionManager`）、`extractors.py`、`models.py`、`policy.py`（红线分类，见 §5.3）、`api_support.py`、`prompting.py`。
- 不该拥有：不直接改变任务指标（AGENTS.md 原话）；记忆只能辅助解释/建议/风险提醒/报告口径。

### 2.8 `plugins/` + `packs/` —— Plugin/Tool/Hook runtime 与能力包

- `plugins/`（runtime）：`loader.py`（`load_builtin_packs`/`sync_builtin_packs`）、`registry.py`（`PluginRegistry`/`ToolRegistry`）、`runner.py`（`ToolRunner`，2,175 行，工具执行 + 结果哈希/receipt）、`manifest.py`、`schema_validation.py`、`contracts.py`、`hooks.py`（`HookDispatcher`）、`sdk.py`、`subprocess_worker.py`（工具子进程隔离）。
- `packs/`（内置能力包，每个包有 `manifest.json` + `tools.py`）：`data_ops`、`feature`、`modeling`（最大，含 `train_tools.py`/`prepare_tools.py`/`score_evidence_tools.py`/`tune_*.py`/`recipes/` 等）、`strategy`（最大且最细，含 `sample_design*.py`、`candidate_evidence.py`、`backtest.py`、`pool_*.py`、`interactive_tree_*.py`、`report_bundle.py`、`dsl.py` 等）、`analysis`、`v1_compat`、`risk_analysis`、`labeling`、`drafts`、`_sample`。
- 不该拥有：Plugin 不自己持 DB 生命周期；`packs/strategy` 的证据契约明确「只描述开发证据，不能用于声称验证/采纳，不拥有持久化」。

### 2.9 `data/` —— DuckDB 数据后端与数据集处理

- 职责：`backend.py`（DuckDB `DataBackend`）、导入（`csv_ingest.py`/`excel_ingest.py`/`schema_infer.py`）、`join_engine.py`、`transforms.py`/`transform_semantics.py`、`sampler.py`、`profiler.py`、`registry.py`（数据集注册与 `content_hash`）、`fingerprint.py`、`workspace.py`、`data_dictionary.py`、`dedup.py`/`align.py`、`label_construction.py`/`labels.py`，以及 GC（`dataset_source_gc.py`/`task_filesystem_gc.py`/`validation_batch_upload_gc.py`）。
- 不该拥有：不承载工作流编排（那是 orchestrator/agent 的事）。

### 2.10 `repositories/` —— 按域的数据访问层

- 职责：SQLite 数据访问，每个域一个 Repository：`tasks.py`（`TaskRepository`）、`datasets.py`、`plans.py`、`plugins.py`、`strategy.py`、`drafts.py`、`modeling.py`、`audit.py`、`llm_calls.py`、`validation_contracts.py`、`validation_batches.py`、`data_workspace.py`、`task_artifacts.py`、`strategy_*.py`（多个策略子域）等。`db.py` 转出主要入口。
- 不该拥有：业务算法与 HTTP（只是持久化）。

### 2.11 `governance/` —— maker/checker 治理

- 职责：`repository.py`（`GovernanceRepository`，含本地 session/TTL）、`service.py`（`GovernanceService`，工具绑定的授权解析）、`contracts.py`、`errors.py`（`AuthorizationError`）。`app.py` 在启动时 reconcile，并把 `governance_service` 注入 `ToolRunner`/`PlanExecutor` 作为 `authorizer`/`binding_resolver`。

### 2.12 `production_governance/` —— 生产激活证据

- 职责：`evidence.py`（`ActivationEvidenceVerifier`）、`repository.py`、`router.py`、`errors.py`。`create_app` 接受 `production_activation_verifiers` 注入（`app.state.production_activation_verifiers`）。路径 `/api/production-governance` 被 `_is_local_only_path` 列为本地专属。

### 2.13 `decision_twin/` —— 反事实/重放沙箱

- 职责：`replay.py`、`comparison.py`、`reconciliation.py`、`_canonical.py`、`artifacts.py`、`authenticated_entry.py`、`contracts.py`。计划 B-11 将其产品化（见 §6 引用的计划文档），本文不展开其细节。

### 2.14 `operations/` —— 监控/调度运行时

- 职责：`router.py`、`scheduler.py`（`MonitoringExecutor`）、`integration.py`（`build_operations_runtime`，注入 `app.state.operations_runtime`）、`notifications.py`、`repository.py`、`contracts.py`、`schema.py`。`create_app` 接受 `operations_executor_allowlist` 白名单。

### 2.15 `orchestrator/` —— 计划运行时

- 职责：把「声明式 Workflow 模板」实例化为 `Plan` 并逐步执行。
  - `contracts.py`——`Plan`/`PlanStep`/`StepStatus`/`PlanStatus`/`plan_fingerprint` 等核心类型。
  - `templates/`——内置模板（`strategy.py`、`validation.py`、`modeling.py`、`feature.py`、`join.py`、`portfolio.py`、`monitoring.py`、`sample.py` 等）+ `skills.py`（`load_user_skill_templates`，用户可编写 Workflow 模板来源）。
  - `validator.py`——`PlanValidator`：校验 tool 引用、`{slot:...}` 占位符、`post_checks`（`schema`/`range`/`rowcount`/`invariant`/`nonempty`/`match_rate`/`one_of`）与指标安全（`orchestrator/safety.py` 的 `METRIC_FIELDS`）。
  - `executor.py`——`PlanExecutor`：逐步执行、引用解析、`needs_confirmation` gate 暂停、canonical 结果认证（`canonical_results.py`）、重试与失败恢复（`plan_recovery.py`）。
  - 其他：`planner.py`（`Planner`，LLM 生成计划）、`reviewer.py`（`Reviewer`）、`intent.py`（`IntentRouter`）、`subagent.py`（`SubAgentDispatcher`）、`harness_state.py`、`evidence.py`（`artifact_refs`/`payload_hash`）、`references.py`、`capability.py`（能力分层）、`context/`（预算/账本/观测）、`eval/`（`eval-llm` 评测 harness）。

### 2.16 `notebooks.py` / `notebook_worker.py` —— 隔离执行

- 职责：Jupyter Notebook 的会话与隔离执行。`notebooks.py`（1,810 行）管理 `NotebookExecutionSession`、`register/get/close_live_notebook_session`、`prepare_execution_notebook_v3`、`run_notebook(isolated=...)`、`_run_notebook_in_subprocess`（`subprocess.Popen` 启动 `marvis.notebook_worker`，`start_new_session` 脱离进程组），结果用 `NOTEBOOK_RESULT_SENTINEL` 标记。`notebook_worker.py`（84 行）是子进程 worker：从 stdin 读 job JSON → `run_notebook(... isolated=False)` → 以 sentinel+UTF-8 原始字节写回 stdout。契约层在 `notebook_contract.py`（747 行，见 §4.1）。
- 关键点：隔离的物理边界是**子进程**（worker 自身再起 kernel）；`notebook_isolated_execution` 开关决定走子进程隔离还是「legacy live kernel」（§4.3）。

### 2.17 相邻模块（本任务未单列，但新读者常撞见）

- `marvis/artifacts/`——受治理的 task artifact 抽象：`__init__.py`（`ArtifactUnitOfWork`）、`model_score_vector.py`（分数向量 + 哈希前后重验）、`recovery.py`（`reconcile_workspace_artifacts`，启动时对账）、`transactional.py`。
- `marvis/drafts/`——报告草稿子系统：`registry.py`（`DraftRegistry`）、`sandbox.py`（`DraftSandbox`）、`authoring.py`、`promotion.py`、`learning.py`、`language.py`、`tools.py`、`web_search.py`。`app.py` 组装 `DraftRegistry`/`DraftSandbox`。
- `marvis/feature/`——特征工程纯算法（`binning.py`、`iv.py`、`correlation.py`、`univariate.py`、`weighted_rule_tree.py`、`preprocessing.py`、`screen.py` 等），独立于 `packs/feature`（后者是 Tool 包装，前者是算法本体）。
- `marvis/workspace/`——运行期工作区目录（`datasets/`、`tasks/`、`plugins/`、`logs/`、`marvis.sqlite`、`plugin_admin_token`），由 `settings.py` 的 `Settings` 定位，非代码。
- 顶层支撑模块：`domain.py`（`TaskStatus`/`TaskRecord`/`FileArtifact` 等域类型）、`state_machine.py`（`IllegalTransition`，`app.py` 捕获为 409）、`files.py`（`sha256_file`/`write_json_atomic`/`scan_source_dir`）、`llm_client.py`（`OpenAICompatibleLLMClient`）、`llm_prompts.py`（77,089 字节 prompt 文本）、`llm_settings.py`（模型解析）、`api_schemas.py`（请求/响应 schema）、`safe_paths.py`（路径逃逸防护）、`redaction.py`、`reconcile.py`。

---

## 3. 一条 workflow 的端到端数据流（策略开发为例）

以「自然语言策略开发请求」为例，链路如下（每一环都有对应代码文件）：

1. **自然语言请求** → 任务级 Agent turn loop（`agent/turn_handlers/` 包的分发 + `agent/plan_driver.py` 的 driver）。策略任务属于 `DRIVER_AGENT_TASK_TYPES` 之一，走 driver 化计划流（manual 与 agent 两种模式都经过 agent 端点；agent 模式由 LLM 操作控制，manual 模式由用户操作）。
2. **意图编译** → `agent/strategy_request_compiler/` 包的 `compile_strategy_request`：LLM 只把一句话翻译成 draft，随后本模块用固定词表（`STRATEGY_OPERATIONS` / `STRATEGY_TYPES` / `STRATEGY_REQUEST_KINDS`）、Strategy DSL（`packs/strategy/dsl.py` 的 `parse_strategy_spec`）、数据集列白名单来校验 draft。文档串明确：**它不执行任何 tool，也绝不接受模型算出来的指标**。
3. **workflow 解析与计划生成** → `agent/strategy_workflows/__init__.py` 的 `resolve_strategy_request` / `prepare_strategy_plan`：把编译结果解析为标准 workflow（fresh/replay/legacy 三种），再由 `orchestrator/templates/strategy.py` 的 `WorkflowTemplate` 实例化为带 `post_checks` 的 `Plan`。
4. **计划校验** → `orchestrator/validator.py` 的 `PlanValidator`：校验 tool 引用、slot 占位符、每个步骤的 `post_checks`（范围/非空/一致率等）与指标安全规则。
5. **计划执行** → `orchestrator/executor.py` 的 `PlanExecutor`：逐步执行，通过 `plugins/runner.py` 的 `ToolRunner` 调用 `packs/strategy/*` 工具；在 `needs_confirmation` gate 前暂停，把上一步刚算出的输出转成待确认消息。
6. **证据产出** → `packs/strategy` 的工具产出内容寻址、确定性的证据（如 `candidate_evidence.py` 的 `StrategyCandidateEvidence`、`sample_design_v2.py` 的 `StrategySampleDesign`），由平台代码计算指标，LLM 只解释/总结/起草。
7. **四格式报告** → `packs/strategy/report_bundle.py` 生成**自认证**的 `StrategyReportBundle`（`report_id` 与 `content_sha256` 由规范 JSON 重算校验），再由 `output/strategy_report_bundle.py`（renderer）投影到四种格式：**canonical JSON、Markdown、无公式 XLSX、无宏 DOCX**。缺失值保持空白，其类型化可用性与原因留在证据索引里。

关于第 5 步的确认语义，再补一层细节（它决定了「LLM 不能替你拍板」）：`PlanExecutor` 在标记为 `needs_confirmation` 的 gate 步骤**之前**暂停，`plan_driver.py` 把上一步刚刚算出的输出转成 append-only assistant 消息（含内联富表格）。用户看到并确认的，正是刚刚真实执行过的结果；确认才 resume。manual 模式（无 LLM，用户操作控件）与 agent 模式（LLM 操作控件）都经过同一 driver 计划流——validation 例外，其 manual 模式走独立的 scan/notebook 流（见 `api.py` 中 `_DRIVER_AGENT_TASK_TYPES` 的注释）。

数据流小结：

```
自然语言 → strategy_request_compiler（编译+校验，不收 LLM 指标）
        → strategy_workflows.resolve（fresh/replay/legacy）
        → templates/strategy.py（WorkflowTemplate → Plan）
        → PlanValidator（tool 引用 + post_checks + 指标安全）
        → PlanExecutor（ToolRunner 调 packs/strategy 工具，gate 前暂停）
        → packs/strategy 证据（内容寻址、平台算指标）
        → StrategyReportBundle（自认证）→ output/strategy_report_bundle（四格式）
```

### 3.1 模型验证 V1-compat 路径（并列简述）

- `orchestrator/templates/validation.py` 定义 `MODEL_VALIDATION` 模板，步骤依次是 `scan_materials` → `run_notebook` → `compute_validation_metrics` → `render_reports`，tool 全部指向 `v1_compat` 包，并带 `post_checks`（`status` 值域、`ks`/`auc` 的 `0..1` 范围、`psi` 可空等）。
- `packs/v1_compat/tools.py` 是这些 tool 的实现：`tool_run_notebook` 调 `pipeline.run_notebook_stage`，`tool_compute_validation_metrics` 调 `pipeline.run_metrics_stage`，`tool_render_reports` 调 `pipeline.run_report_stage` —— 也就是把 `pipeline.py` 的 V1 三阶段包装成可被 orchestrator 编排的 tool。
- 因此 V1-compat 路径 = **orchestrator 模板 → v1_compat 工具 → pipeline 三阶段 → validation/ 确定性算法 → output/ 渲染**，与 native 策略路径共用同一套 PlanValidator/PlanExecutor 运行时。

---

## 4. 兼容边界

### 4.1 Notebook 契约（`RMC_*`）

Notebook 执行结束前必须在顶层作用域定义四个名字（`notebook_contract.py` 用 AST 逐项校验，缺失即报错）：

- `RMC_SAMPLE_DF`：pandas DataFrame，平台用于分数一致性、KS、PSI、分箱、压力测试的原始样本；必须在顶层可见，不能只存在于函数局部。
- `RMC_TARGET_COL`：目标列名字符串。
- `RMC_ALGORITHM`：算法名（如 `"lgb"`）。
- `RMC_SCORE_FN(df)`：可调用，接收 DataFrame、返回一维数值分数、长度等于行数。

权威文档：[notebook_contract.md](./notebook_contract.md)（平台运行契约摘要）与 [对notebook的要求.md](./对notebook的要求.md)（给建模人员的提交要求）。`RMC_FEATURES` 已废弃；代码模型与 PMML 都应从同一份原始样本自行取数转换。

### 4.2 Notebook 内存分 vs PMML 分

主分数一致性比较是 **Notebook 内存里的模型分 `RMC_SCORE_FN(sample_df)` vs 提交目录中拟投产 PMML 的打分结果**（`validation/in_memory_scores.py` 与 `validation/pmml_scoring.py` / `pmml_score_artifacts.py` 的 `build_pmml_scoring_identity` / `validate_pmml_score_artifact`）。Notebook 可额外导出 PMML 作为审计材料，但平台不再把新导出的 PMML 作为主对比对象。

### 4.3 legacy live kernel 的三重 opt-in

`pipeline.py` 的 `legacy_live_notebook_execution_allowed()` 要求**三项同时满足**才允许 legacy live kernel 执行：

1. `PipelineSettings.notebook_isolated_execution == False`（默认 `True`）；
2. `PipelineSettings.allow_legacy_live_notebook_execution == True`（默认 `False`）；
3. 环境变量 `MARVIS_ALLOW_LEGACY_LIVE_NOTEBOOK_EXECUTION=1`。

任一不满足则 `_require_legacy_live_notebook_execution` 抛 `PipelineError`，错误文案 `LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE` 精确列出三项要求。默认路径走子进程隔离（§2.16）。

### 4.4 legacy vs native sample design 双轨

- `packs/strategy/sample_design.py`——V1 的 `StrategySampleDesign`：单一边界、内容寻址，冻结「已物化的 active dataset 边界」并给出带版本的指标定义与观测。
- `packs/strategy/sample_design_v2.py`——V2 契约：在同一个精确分析 universe 上冻结 **approval 与 risk 两个人群**，各自持有 development/validation/OOT 成员引用；只接受「已认证的 membership header + 已解析的布尔 mask + 已计算的类型化统计」，不拥有 DB/DataFrame 过滤/Tool/Agent。
- 执行边界：`sample_design_v2_native_tools.py` 文档串标注为「Native active-dataset execution boundary」——即 native 路径直接对 active dataset 计算；`sample_design_v2_tools.py` 是共享实现。二者并存构成「legacy（物化边界）vs native（active-dataset 执行）」双轨。

---

## 5. 信任机制

### 5.1 hash / provenance / 落盘前重验

- `provenance.py`：`NumberProvenance` 是最小溯源四元组 `(dataset_fingerprint, code_version, params_digest, seed)`，让每个摆在人面前的决策数字都能回答「你从哪来」。`dataset_fingerprint` 复用 `data/registry.py` 的 sha256 `content_hash`，`params_digest` 复用 `orchestrator/executor.py:_payload_hash` 的规范 JSON 哈希约定。
- **落盘前重验**（先重算再接受，而不是信任自带字段）：`packs/strategy/report_bundle.py` 的 `validate_strategy_report_bundle` 会从规范 JSON **重算** `report_id`（`strategy-report-` + body 哈希前 24 位）与 `content_sha256`，再用 `hmac.compare_digest` 比较，不匹配即抛 `StrategyReportBundleError`。同样的「重哈希 + `hmac.compare_digest`」出现在 `routers/artifacts.py`（下发行 artifact 前比对 `content_hash`）、`download_snapshot.py`（拷贝快照时边拷边验）、`artifacts/model_score_vector.py`（哈希前后都验，防「哈希期间被改」）、`plugins/runner.py`（`_payload_hash` / `_manifest_receipt_hash` / `result_hash`）。
- **证据信封（evidence envelope）**：`orchestrator/evidence.py` 提供 `payload_hash`、`artifact_refs`、`dataset_refs`、`artifact_bindings`、`result_dataset_ids`，`executor.py` 用它给每个步骤输出打 `input_hash` 与引用清单。`provenance.py` 的 `params_digest` 与 `executor.py:_payload_hash` 保持同一规范哈希约定，二者可比对（`provenance.py` 文档串明确「matches marvis/orchestrator/executor.py:_payload_hash」）。

### 5.2 fail-closed 惯例

- `canonical_results.py`：`authenticate_canonical_result` 是受治理 canonical tool 结果的信任闸口；`CANONICAL_RESULT_TOOLS` 之外的返回 `False`，canonical 不匹配统一归一成**单一 fail-closed 异常**，调用方无法把它降级成「泛化成功渲染」。
- `app.py` 的 `_local_access_guard`：非本地且未过代理认证的请求，非安全方法直接 403；`_is_local_only_path`（含 `/api/settings`、`/api/operations`、`/api/production-governance` 等）对远程一律 403；转发头只信任显式配置的 `MARVIS_TRUSTED_PROXY_HOSTS`，不可信 loopback 对端的转发头 fail-closed。
- `strategy_adoption.py` 的 `normalize_adoption_reason`：遇到待定占位符时 fail-closed（文档串「fail closed on pending placeholders」）。

### 5.3 agent_memory 红线（禁存什么）

`agent_memory/policy.py` 的 `classify_memory_candidate` / `classify_distillation_payload` 用正则逐项拦截，命中即 `allowed=False`：客户明细（客户号/身份证/手机号）、原始样本行、Notebook 源码、PMML/模型文件内容、密钥（api_key/secret/token、`sk-...`）、数据库连接串、长报告全文、绝对本地路径；另有单条记忆文本上限 `MEMORY_CANDIDATE_TEXT_MAX_CHARS = 12000` 与 `PAYLOAD_FIELD_ALLOWLISTS` 的逐类型字段白名单（多出的字段即拒绝）。这与 AGENTS.md 的「禁止保存原始样本、客户明细、完整 Notebook 源码、PMML/模型文件内容、API key、数据库连接、未脱敏报告全文或机构敏感信息」一致。

### 5.4 确定性指标归平台代码

KS/AUC/PSI、分数一致性等确定性指标由平台代码计算（`validation/platform_metrics.py` 的 `compute_platform_validation_results` 等），**不由 LLM 计算**。Agent 只能解释、总结、起草、规划和请求确认，不能编造指标或绕过平台证据（AGENTS.md 铁律；`strategy_request_compiler.py` 文档串重申「never accepts calculated metrics from the model」）。

---

## 6. 结构性债务与拆分（现状与方向）

**2026-08-14 更新（B-4/B-6 已落地）**：本节写作时的两个巨型文件已完成物理拆分：

- `marvis/agent/turn_handlers.py`（17,868 行）→ `marvis/agent/turn_handlers/` 包：17 个车道文件 + `_registry`，`__init__.py` 以 exec-merge 设计把车道源码按依赖序注入单一命名空间——运行时语义与原单体完全一致（名字解析、monkeypatch 行为不变），相关 865 个测试通过、ruff 干净。
- `marvis/agent/strategy_request_compiler.py`（14,449 行）→ `marvis/agent/strategy_request_compiler/` 包：10 个请求家族车道，同样 exec-merge；`__all__` 与原文件逐名一致（29 名），相关 1,993 个测试通过（1 个测试因 `inspect.getsource` 读模块源码而迁移到读合并车道源码，语义不变）。
- `marvis/static/app.js`：8,465 → 8,382 行。B-6 核查结论：`static/js/` 各模块是**刻意的薄适配层**，与单体不存在真正的重复实现；本轮只删除确认死代码（14 个函数、5 个未用 import），507 个前端静态测试通过。

**剩余结构性债务（如实）**：

- `marvis/static/app.js` 仍是约 8.4k 行的 JS 单体，是当前最大的单文件；继续收敛需要按功能迁移而非删除。
- 拆分后最大的车道文件：`turn_handlers/strategy_turns.py`（约 3.0k 行）、`turn_handlers/strategy_candidates.py`（约 4.5k 行）、`strategy_request_compiler/core.py`（约 2.7k 行）、`pool.py`（约 2.5k 行）、`tree.py`（约 2.4k 行）——仍可在车道内继续细分，但已不再是跨 workflow 的单体。
- `.bandit-baseline.json` 承载 83 条已接受 finding（主体为 B608 f-string SQL），B-5 已重建基线并加入增量门，专项复审见 `docs/reviews/2026-08-13-bandit-baseline-review.md`。

拆分方向与执行记录以 [2026-08-13-next-90-days-development-plan.md](./superpowers/plans/2026-08-13-next-90-days-development-plan.md) 为准，本文只陈述现状。

## 7. 读码入口指引

想理解某个主题，按下列顺序先读这些文件（路径相对仓库根，`marvis/` 前缀省略处即 `marvis/` 下）：

| 主题 | 先读（顺序即推荐顺序） |
|---|---|
| 进程如何启动、如何组装 | `marvis/__main__.py` → `marvis/app.py` → `marvis/settings.py` |
| HTTP 层与路由归属 | `marvis/api.py` → `marvis/routers/tasks.py` → `marvis/routers/validation_agent.py` |
| 持久化与 schema | `marvis/db.py` → `marvis/db_schema.py` → `marvis/agent_memory_schema.py` |
| V1 验证流水线 | `marvis/pipeline.py`（先读模块 docstring）→ `marvis/pipeline_errors.py` → `marvis/pipeline_cellgen.py` → `marvis/pipeline_io.py` |
| 确定性验证算法 | `marvis/validation/platform_metrics.py` → `marvis/validation/results.py` → `marvis/validation/pmml_scoring.py` |
| Notebook 执行与契约 | `marvis/notebook_contract.py` → `marvis/notebooks.py` → `marvis/notebook_worker.py` → `docs/notebook_contract.md` |
| 计划运行时 | `marvis/orchestrator/contracts.py` → `marvis/orchestrator/templates/strategy.py` → `marvis/orchestrator/validator.py` → `marvis/orchestrator/executor.py` |
| Plugin/Tool runtime | `marvis/plugins/registry.py` → `marvis/plugins/runner.py` → `marvis/plugins/manifest.py` → `marvis/packs/strategy/manifest.json` |
| 策略端到端 | `marvis/agent/strategy_request_compiler.py`（读头部 docstring）→ `marvis/agent/strategy_workflows/__init__.py` → `marvis/packs/strategy/report_bundle.py` → `marvis/output/strategy_report_bundle.py` |
| Agent turn loop 与 driver | `marvis/agent/plan_driver.py` → `marvis/agent/driver_turn.py` → `marvis/agent/turn_handlers.py` |
| 记忆与红线 | `marvis/agent_memory/store.py` → `marvis/agent_memory/policy.py` → `marvis/agent_memory/consolidation.py` |
| 治理与授权 | `marvis/governance/service.py` → `marvis/governance/repository.py` → `marvis/production_governance/evidence.py` |
| 信任与溯源 | `marvis/provenance.py` → `marvis/canonical_results.py` → `marvis/packs/strategy/report_bundle.py`（`validate_strategy_report_bundle`） |
| 兼容边界 | `marvis/notebook_contract.py` → `marvis/pipeline.py`（`legacy_live_notebook_execution_allowed`）→ `marvis/packs/strategy/sample_design_v2_native_tools.py` → `marvis/packs/v1_compat/tools.py` |
| 前端 | `marvis/static/app.js` → `marvis/static/js/api.js` → `marvis/static/js/state.js` → `marvis/static/js/v2/plan_rail_controller.js` |
