# MARVIS V2 分支深度 Code Review（2026-06-28）

- **分支**：`codex/v2-plugin-tool-runtime`（vs `main`：369 文件、+78,212 / −1,150）
- **范围**：V2 平台全量后端 + 前端 —— orchestrator、plugins、packs(modeling/strategy/feature/data_ops/v1_compat)、agent、agent_memory、data(JOIN 引擎)、validation、output、routers、`api.py`/`db.py`/`pipeline.py`、`static/js/v2/*`。Python ~44K LOC / 195 文件 + 前端 ~12K LOC。
- **方法**：17 路分领域 reviewer 各自读真实源码出结构化发现 → 每条发现派独立 skeptic 做对抗式验证（confirm/refute）→ 对被限流丢弃的发现二次验证 → 人工直读核实头部高危项。**所有结论均经"验证后"过滤**，误报单列。
- **测试基线**：canonical env（`/opt/miniconda3/envs/py_313/bin/python`）**1568 passed**；`.venv` 缺必需依赖 `sklearn2pmml`/`pypmml` 会有 5 个 PMML 用例假失败（非代码 bug，已确认）。

> 验证产能说明：对抗验证阶段两次触发服务端限流/会话上限，**31→17 条二次验证已回收**，余 9 条标注为「待最终验证」（附原始证据与置信度）。`orchestrator-support`、`api-core` 两个子系统的复核 reviewer 未能跑完，单列「未覆盖」。

## 2026-06-28 修复状态更新

下面的 High/Medium/Low 列表是本轮修复前的原始审查快照，用于追溯证据和修复意图；不应直接当作当前未修复清单。

当前复核验证：`CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest -q` 通过，`1752 passed, 2 warnings`（MLP 收敛 warning）。

本轮追加专项验证：`tests/test_data_repository_registry.py tests/test_modeling_prepare.py tests/test_modeling_db.py tests/test_modeling_experiment.py tests/test_modeling_handoff.py tests/test_modeling_pack.py tests/test_modeling_reject_inference.py` 通过，`58 passed`；`tests/test_db.py tests/test_modeling_api.py tests/test_modeling_report.py` 通过，`57 passed`。

后续追加验证：`tests/test_plugin_runner.py tests/test_plugin_hooks.py` 通过，`36 passed`；`tests/test_drafts_sandbox.py tests/test_drafts_tools.py tests/test_drafts_registry.py tests/test_orch_executor.py tests/test_orch_subagent.py tests/test_plugin_db.py tests/test_plugin_registry.py` 通过，`72 passed`。

已修复并已有回归覆盖的重点项：

- H1/H2：LightGBM 默认配方与调参路径已接入共享 NaN 标签确认门，避免静默缩样或污染选参。
- H3/M3/P4：JOIN match-rate/唯一性/去重改为按与实际 JOIN 一致的变换键空间处理，特征键扫描下推 DuckDB，并保留 hash/date fallback 测试。
- H4/L14：确认门与去重指令已处理否定语义，避免“不要/不可以”被当作确认。
- H5：子 agent 内层计划只有 `DONE` 才会向父计划返回成功；确认暂停/评审暂停会作为 paused/failure 暴露，不再把 `summary_ref=None` 当成功输出。
- Sub-agent 审计完整性：`PlanRepository.upsert_sub_agent_with_audit` 和 `set_sub_agent_status_with_audit` 已把 sub-agent spawn、returned/failed 状态与 `subagent.spawn` / `subagent.run` 审计写入放进同一 SQLite 事务；审计失败不会留下未审计的授权子 agent 或把子 agent 状态推进到 returned/failed。
- 记忆沉淀演进完整性：`AgentMemoryStore.replace_active_distillation_with_audit` 和 `rollback_active_distillation_with_audit` 已把新 head 创建、旧 head supersede、rollback 恢复旧 head、当前 head rolled_back 及对应审计事件放进同一事务；审计失败不会产生两个 active head 或半回滚状态。
- H6/M1：插件 worker 环境改为 allowlist，保留非敏感网络降级配置，stdout/stderr tail 脱敏，资源限制扩展到内存/CPU/文件大小并写回协议与审计。
- M2/M5/M6/M7/M8/M9/M10 及多项 Low：已补齐场景参数、策略 dtype、NaN-key 去重、manual WOE 分箱、auto_distill 门控、记忆排序确定性、WOE train-only 拟合等修复与测试。
- M4/L7/L8/L10/L11/L13/L15-L24/P1-P9：已补 roll-rate 时间排序校验、vintage NaN 标签门、backtest swap 复用、缺列统一错误、复现性索引对齐、退化 AUC=0.5、目标探测跳过 split/id/meta/weight/date、记忆 use 状态守卫、蒸馏非数值指标逐值跳过、feature AUC/PSI/IV 降级、并发更新守卫、恢复范围化、loop_event 容错、plan 轮询清理、pickle 中转、join 大表冲突样例、execute_join plan 级 post-check。
- L5/L6/L26-L30：final review 的 `goal_doubt` 会阻断 `goal_met`，安全步骤判断已收敛到共享模块，vintage 报告长表改为 `melt` 向量化；前端确认/取消按钮防重复提交、记忆回滚二次确认、draft run 输入必须是 JSON 对象、JOIN fan-out/409 显示专用提示。
- L4：`python_requires` 已在插件注册/工具目录里强制执行；`permissions`/`side_effects` 已加词表、manifest 子集校验和 runner 二次校验；worker 默认禁用未授权外部网络访问，但允许本机 loopback/Unix socket 供 notebook kernel、PMML/Java 等本地运行时使用；draft run 复用静态安全扫描。
- 插件输出路径：`ToolRunner` 已对工具返回的 `path`/`*_path` 和 `artifact:*` 引用做后置边界校验；绝对路径只允许落在 workspace 或 datasets root，下游相对路径和 artifact ref 会拒绝绝对、`..`、`~` 等逃逸形态。
- Runtime 恢复：`CHECKING`/`RUNNING` 仅在当前步骤行持有 `output_ref` 时按该版本恢复 review，不再用历史输出版本表猜最新输出；`RUNNING` 无当前输出时标失败并要求显式 retry，且 adaptive replan 不会自动绕过；`CHECKING` 恢复 DONE 时会补发完成 hooks。
- 显式 retry：已新增失败步骤 retry endpoint 与 V2 计划视图“重试步骤”入口；retry 会在事务内重置失败步及下游依赖，清空旧输出/评审/确认/子 agent 关联，并把失败 plan 重新置为 `running` 后启动 executor。endpoint 支持可选 `inputs` 覆盖目标失败步参数；模块化 V2 计划视图和主工作台右侧 plan rail 均已提供 JSON 参数编辑后重试入口，前端会拒绝非对象 JSON。
- 材料路径与健康检查：`tool_ingest_excel` 已收敛到 workspace/home/`RMC_MATERIAL_ROOTS` 材料根；`/api/health` 已暴露 SQLite journal mode / WAL degraded / busy timeout。
- Notebook RSS 软监控：pipeline 执行路径默认给 Notebook kernel 配置 4096MB RSS soft limit；监控器通过 jupyter-client provisioner PID + `psutil` 采样 kernel 进程树，超限时关闭 kernel、终止进程树兜底，并在 `NotebookRunResult.resource_usage` 与失败日志中保留 peak RSS / limit / pid。底层 `run_notebook` 仍支持显式传入 `memory_limit_mb`，未配置时不制造监控噪声。
- JOIN 审计完整性：`JoinEngine` 不再接受缺少 `write_audit` 的 repo，避免确认/执行审计静默丢失；真实 `DatasetRepository` 的 `update_join_spec_with_audit` 和 `set_join_plan_executed_with_audit` 已把 JOIN spec/状态更新与审计写入放进同一 SQLite 事务，审计失败会回滚业务状态。JOIN 执行结果也已收敛到 `record_join_result_with_audit`，新 derived dataset row、join plan executed 状态和 `join.executed` 审计同事务落库；审计失败时会删除本次生成的 joined parquet，避免留下无审计的数据谱系。
- Draft/Plugin 审计完整性：`DraftRepository.set_status_with_audit` 已把 draft reject 状态与审计写入事务化；`PluginRepository` 的 register/enable/disable/remove 已新增 `*_with_audit` 事务路径，`PluginRegistry` 默认使用这些路径，审计失败会回滚插件表/工具表/启停状态/删除状态。draft promote 现在走 `promote_draft_with_plugin_audits` 同连接事务，插件注册、`plugin.register` 审计、draft promoted 状态、`draft.promote` 审计任一失败都会整体回滚；插件文件写入也改为 staging+backup 流程，DB 失败会恢复文件系统。
- Draft 学习/作者ing 审计完整性：`DraftRepository.save_learning_note_with_audit` 和 `save_draft_with_audit` 已把 learning note / draft tool 创建与 `draft.learning_note.create` / `draft.author` 审计写入放进同一 SQLite 事务；真实 draft tools 现在在保存时写审计，router 不再事后补写，审计失败不会留下未审计的学习笔记或草稿工具。
- Draft sandbox 审计完整性：`DraftRepository.save_draft_run_with_status_audit` 已把 draft run 记录、可选 tested 状态推进和 `draft.run.record` 审计写入放进同一 SQLite 事务；审计失败不会留下 promotion-eligible 的未审计测试运行。`draft.invoke` 仍保留为子进程执行审计，`draft.run.record` 记录本地治理状态。
- Strategy 审计完整性：`StrategyRepository.create_strategy_with_audit` 和 `save_backtest_with_audit` 已把策略规则创建、回测结果落库与 `strategy.create` / `strategy.backtest` 审计写入放进同一 SQLite 事务；审计失败不会留下未审计的业务决策规则或回测结果。
- Report override 审计完整性：`TaskRepository.update_report_values_with_audit` 和 `update_agent_report_conclusions_with_audit` 已把报告文本覆盖、revision 推进与 `report.values.update` / `report.agent_conclusions.confirm` 审计写入放进同一 SQLite 事务；审计失败不会留下未审计的最终报告文本变更。
- Modeling 审计完整性：`ExperimentStore` 的 experiment create/trained/status/PMML/calibration 已改为业务状态与 audit 同事务写入；PMML/calibration 会先确保 model meta 写盘成功，再写 DB+audit，失败时删除新产物并尽量恢复旧 meta。validation handoff 已把 validation task 创建、experiment `handed_off` 状态和 `modeling.validation_handoff.create` 审计放入同一事务，审计失败会清理本次 material 目录并恢复旧 material 目录。`prepare_modeling_frame` / `make_split` / `reject_inference` 的派生数据集注册已新增 `modeling.dataset.derived` / `modeling.reject_inference.created` 审计，审计失败会回滚 dataset row 并删除本次 parquet。模型报告生成已新增 `modeling.report.generated` 成功审计，渲染或审计失败不会留下成功审计，审计失败会删除本次报告和 scored 中间 parquet。
- Tool/Hook invocation checkpoint：`ToolRunner.invoke` / `invoke_adhoc` 已在启动 worker 前写 `tool.invoke.started` / `<mode>.invoke.started`；started audit 写失败时不会启动 worker。执行后的最终 audit 写失败会返回 `error_kind="audit"`，保留 started checkpoint，避免计划继续把不可追溯副作用当作成功。`HookDispatcher` 已同样为插件 hook 写 `hook.dispatch.started`，并为内置 listener 写 `hook.listener.started` / `hook.listener`；started audit 失败时跳过对应 hook/listener。
- 半成品项：`reject_inference` 已落地为受确认/审计约束的建模 tool；strategy/vintage 也已接入 PlanDriver，不再是 501/coming-soon 路径。

仍需要后续专项的风险：

- Notebook RSS 软监控已完成单测与全量回归；真实 OOM 慢测、OS 级 Notebook 沙箱/进程命名空间隔离仍未完成。
- 失败后从失败步显式 retry 的用户/agent 契约已补 endpoint、前端入口和 retry 前参数编辑面板；后续还可按产品需要增加按工具 schema 生成的结构化参数表单。
- `api.py` / `db.py` 拆分、仍未逐一事务化的其它跨仓库业务写入、完整视觉 token 系统仍未做。
- validation 的 head/tail lift 暂未按相关方向自动翻转：当前验证层保留“声明的分数方向”语义，已有测试固定反向模型应暴露为 `auc=0/head_lift=0/tail_lift>0`，避免把方向错误隐藏成好模型。
- `agg_max/agg_mean` 去重逐列聚合仍按当前设计保留，依赖冲突报告在 C2 门显式提示；如果业务要求“必须保留真实行”，应另增 row-level 聚合策略。
- 插件进程级文件系统根、Notebook 沙箱仍是后续架构项；当前已先补输出路径后置校验、Notebook RSS soft limit 和 tool/hook started checkpoint，但还不是 chroot/OS 级文件系统沙箱。DAG `CHECKING`/`RUNNING` 崩溃恢复和显式 retry 已补基础 checkpoint 契约，后续如需更强保证应把 started/final invocation audit 接入可视化恢复队列。

---

## 0. 结论速览

| 严重度 | 已确认（可直接修） | 待最终验证 | 说明 |
|---|---|---|---|
| **Critical** | 0 | 0 | 无"已确认的数据损坏/安全击穿/必崩主路径"。头条疑似 critical（vintage cum_bad_rate）经双重验证为**误报**。 |
| **High** | 6 | 1 | NaN 标签门两处绕过、JOIN 键空间不一致、确认门否定误判、子 agent 暂停误报成功、插件继承宿主密钥 |
| **Medium** | 10 | 2 | 确定性/泄漏/健壮性 |
| **Low** | 22 | 6 | 一致性、健壮性、性能、死代码 |
| 未实现/半成品 | 3 | — | reject_inference 空壳、插件能力声明未强制、资源限额不全 |
| 误报（已验证驳回） | 20+ | — | 见 §6，含 vintage cum_bad_rate ×2 |

**最该先修的 6 个（High）**：

1. `train_lgb` 默认建模配方**绕过 NaN 标签门**（INV-5）—— 默认路径直接把 NaN 标签喂给训练 / 静默缩样上报指标。
2. `tune_hyperparameters` 用 `lgb.Dataset` **静默在 NaN 标签上调参**（INV-5）—— 选参被污染。
3. JOIN 唯一性/去重在**原始键**上算，实际 JOIN 在**变换键**（lower/hash/date）上做 —— 大小写/md5/日期键场景漏判，用户确认后 execute 期硬失败。
4. `is_confirm()` 把**否定回复**当确认（"不可以"→True）—— 可触发被拒绝的确认门，含破坏性 JOIN。
5. 子 agent 在确认门**暂停**被当作**成功返回**（`result_ref=None`），父步骤误标 DONE。
6. 插件/草稿子进程**继承父进程完整环境变量** —— 不可信代码可读 `OPENAI_API_KEY` 等宿主密钥。

---

## 1. High（已确认，6）

### H1. `train_lgb` 绕过 NaN 标签门（INV-5）— 默认建模配方
- **位置**：[`marvis/packs/modeling/recipes/lgb.py:21-47`](marvis/packs/modeling/recipes/lgb.py:21)
- **问题**：`lgb` 是 `pre_screen/loan_in/loan_post/marketing/credit_limit/pricing` 等场景的默认配方，也是唯一消费调参结果的配方。其它 7 个配方（lr/xgb/catboost/mlp/scorecard/lgb_regressor/lgb_multiclass）在 `split_modeling_frame` 后都调用了 `resolve_modeling_splits(...)` 做 NaN 标签门，**唯独 `train_lgb` 直接 `LGBMClassifier.fit`**。结果：(a) NaN 标签行进入 `fit`（`LGBMClassifier` 会直接报 `Input contains NaN`，即便 `drop_nan_labels=True` 也救不了，崩在训练而非走确认门）；(b) 指标侧 `compute_model_metrics` 经 `_finite_binary_pairs` **静默剔除 NaN 标签行**，train/test/oot KS/AUC 在悄悄缩小的样本上算，`nan_labels_dropped` 仍为默认 0。**既违反"NaN 强制确认"，又静默缩样上报指标。**
- **验证**：对抗验证 confirmed/high + 人工直读核实（对照 `lr.py:26-28`）。
- **修复**：在 `split_modeling_frame` 后照搬 `lr.py` —— `train, test, oot, oot_has_labels, audit = resolve_modeling_splits(train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels)`，把 `oot_has_labels` 传入 `compute_model_metrics`，`TrainResult(... nan_labels_dropped=audit["total_dropped"])`。**根治建议**：把所有配方收敛到一个"已门控"的共享 helper，新配方无法忘记门。

### H2. `tune_hyperparameters` 在 NaN 标签上静默调参（INV-5）
- **位置**：[`marvis/packs/modeling/tune.py:98-128`](marvis/packs/modeling/tune.py:98)（具体 `ytr = train[target_col].to_numpy(dtype=float)` @ L100）
- **问题**：`tune_hyperparameters` 读帧、`_split` 后直接 `lgb.Dataset(label=ytr)`，从不调用 `resolve_modeling_splits/require_labels_confirmed`，`tool_tune_hyperparameters` 也不传 `drop_nan_labels`。与高层 `LGBMClassifier` 不同，底层 `lgb.Dataset` **静默接受 NaN 标签并训练**（NaN 当作 ~class 0）。同时每 trial 的 `train_ks/test_ks` 由 `feature_ks` 计算，会 isfinite-drop 掉 NaN 行 —— **选参指标和训练样本是不同子集**，`best_params` 选自被污染的模型，全程无确认门。
- **验证**：对抗验证（限流丢弃）+ 人工直读核实（L100/L108/L117 无任何 NaN 门）。
- **修复**：构建 `dtrain/dvalid` 前跑 `resolve_modeling_splits`（从 `tool_tune_hyperparameters` 透传 `drop_nan_labels`），有 NaN 且未确认时抛 `NanLabelNotConfirmedError`，确认后丢弃（绝不训练）。

### H3. JOIN 唯一性/去重在原始键上算，但 JOIN 在变换键上做
- **位置**：[`marvis/data/backend.py:147-148`](marvis/data/backend.py:147)、`308-330`、`396-407`
- **问题**：`is_key_unique()/distinct_count()` 与 `_dedup_feature_rel()` 按**原始**特征键值计算/分组；而真正 LEFT JOIN 经 `_join_condition/_sql_transform` 在**变换键**上匹配：`exact_lower` 套 `lower()`，`hash:*` 套 `lower(hash(...))`，`date` 套 `strftime(try_strptime(...))`。对任何非 `exact` 方法，原始键上不同、变换后相同的两行（`'ABC'` vs `'abc'`；md5 大小写；`2024-01-01` vs `20240101`）会塌成同一 JOIN 键。后果：(1) `diagnose_join` 报 `feature_key_unique=True`，`confirm_join_spec` 不要求去重策略，用户以 `dedup_strategy=None` 确认；(2) 即便选了去重，first/last/agg 按原始键分区也留下两行 → JOIN 仍 fan-out。**数据不会被静默污染**（execute 期 1:1 断言会抛 `FanOutError`），但用户合法确认的 JOIN 会在 execute 期硬失败，且这些恰是信贷常见场景（md5 手机号、混合大小写 ID、多格式日期）。现有 fan-out 测试只覆盖 `exact`。
- **验证**：对抗验证 confirmed/high（"两个后果均执行复现"）。
- **修复**：把 uniqueness/distinct_count/去重的 PARTITION BY/GROUP BY 改用与 JOIN **同一套变换键表达式**（抽出 `_sql_transform` 复用），并补 `exact_lower`/`hash:md5`/`date` 的"原始相异、变换相同"测试。

### H4. `is_confirm()` 把否定回复当确认（确认门可被否定语句触发）
- **位置**：[`marvis/agent/plan_driver.py:29-36`](marvis/agent/plan_driver.py:29)；调用点 `plan_driver.py:117/128`、`api.py:3217`
- **问题**：`_CONFIRM` 正则在全串里子串搜索肯定词（可以/确认/继续/对/ok/yes…），**无否定处理**。`is_confirm('不可以')`、`'不确认'`、`'不可以继续'`、`'先不要继续'` 全部返回 True。`PlanDriver.resume()` 在 LLM 路由**之前**先查 `is_confirm()`，否定语句直接短路进 `confirm_step(gate.id)+_run_and_handle` 执行下一步；同一函数还守着 `api.py:3217` 的 JOIN 角色确认与 plan "开始"门。用户在 `execute_join` 确认处打"不可以"会触发其正要拒绝的破坏性 JOIN，违反 JOIN-SAFETY"破坏性 JOIN 需显式确认"。对照 `service.py` 的 V1 意图匹配器是有 `negation_markers` 守卫的，属内部不一致。
- **验证**：对抗验证 confirmed/high + 人工直读核实（`is_confirm('不可以')==True`）。
- **修复**：在肯定匹配前加否定守卫 —— 紧邻/前置出现 `不/别/勿/无需/暂不/取消` 即返回 False；并把短确认锚定为"去标点/ack 前缀后近似整串肯定"而非子串搜索。`api.py:3217` 同源一并修。

### H5. 子 agent 在确认门暂停被当作成功返回
- **位置**：[`marvis/orchestrator/subagent.py:136-155`](marvis/orchestrator/subagent.py:136)
- **问题**：`SubAgentDispatcher.run()` 把 mini-plan 强制 `CONFIRMED` 后跑内层 executor。若 mini-plan 含 `needs_confirmation` 步（validator 对 `execute_join`/`run_draft` **强制**要求），内层 `executor.run()` 返回 `AWAITING_CONFIRM` 且 `summary_ref=None`。但 `run()` **从不检查 `execution.status`**，无条件取 `result_ref=execution.summary_ref`（None），置 `RETURNED`，返回 `ToolResult(ok=True, output={'result_ref': None})`。父步 `_execute_step` 见 `ok=True` 即标 DONE —— **实际停在等用户确认的子 agent 被当成已完成**，下游 `$ref` 拿到 null，且无机制暴露/恢复挂起确认。
- **验证**：对抗验证（限流丢弃）+ 人工直读核实（L136-137 后直接 set RETURNED + 返回 ok=True，无 status 分支）。
- **修复**：跑完内层 plan 后按 `execution.status` 分支：仅 `PlanStatus.DONE` 返回 ok=True；`AWAITING_CONFIRM/REVIEW/FAILED` 返回 ok=False（或独立 paused 结果）带 `error_kind`，使父步不前进、确认门可暴露/恢复。

### H6. 插件/草稿子进程继承父进程完整环境变量（宿主密钥外泄）
- **位置**：[`marvis/plugins/runner.py:315-325`](marvis/plugins/runner.py:315)（`Popen` 无 `env=`）；加重于 `subprocess_worker.py:66-90`、`llm_settings.py:206`
- **问题**：`_run_worker` 以 `subprocess.Popen(...)` 起 worker 但**不传 `env=`**，worker（运行任意不可信插件/草稿代码 `func(job['inputs'], ctx)`）继承父进程整个环境。MARVIS 服务进程里的任何 API key/LLM token/DB 口令（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 等）都能被不可信代码经 `os.environ` 读到，违反 INV-4「不可访问宿主密钥」。`drafts/sandbox.py run_draft → invoke_adhoc` 也走这条路，攻击面不止 admin 安装的插件。
- **验证**：对抗验证 confirmed/high（"代码逐字核对，全仓 grep 无任何环境清洗"）。
- **修复**：给 `Popen` 传显式最小 `env`（仅 `PATH`/`PYTHONPATH`/`LANG` 等 + `PYTHONHASHSEED=0` 利于确定性），剥离一切密钥变量；或在 `subprocess_worker.worker_main()` 进入 `_run_tool` 前按白名单清 `os.environ`。
- **备注**：与之相关的"插件文件系统无限制访问""checksum 不校验"经验证为**有意接受的信任边界**（单用户离线模型，见 §6），不在必修之列；但**环境密钥**是真问题，因为 LLM/DB 密钥即使在单用户模型下也不该暴露给 web-learned 草稿代码。

---

## 2. Medium（已确认，10）

### M1. 插件资源限额仅 Linux 且不全（macOS 上内存上限静默失效）
- **位置**：[`marvis/plugins/subprocess_worker.py:108-117`](marvis/plugins/subprocess_worker.py:108)
- **问题**：`_apply_resource_limits` 只设 `RLIMIT_AS` 且吞掉 `ImportError/OSError/ValueError`。macOS 上 `setrlimit(RLIMIT_AS)` 常失败/被忽略 → except 静默返回 → **无内存上限**；无 `RLIMIT_CPU`（紧 CPU 循环只能等 wall-clock 超时，烧满一核）、无 `RLIMIT_FSIZE`（可写满磁盘）。项目主开发平台是 Darwin，内存界基本形同虚设。超时路径本身正确（进程组 SIGKILL）。
- **修复**：加 `RLIMIT_CPU`（略高于超时）+`RLIMIT_FSIZE`，优先 `RLIMIT_DATA`+`RLIMIT_AS` 并用；`setrlimit` 失败时把告警写进 stderr 协议而非静默续跑。

### M2. 场景模板注入 `scale_pos_weight='auto'` 直接崩 lgb
- **位置**：[`marvis/packs/modeling/scenarios.py:85`](marvis/packs/modeling/scenarios.py:85)
- **问题**：`transaction`（反欺诈）场景 `param_overrides={'scale_pos_weight':'auto'}` 且默认 `recipe=lgb`，`'auto'` 在建模包里从未被解析成数值比，直接进 `LGBMClassifier`。实证 `LGBMClassifier(scale_pos_weight='auto').fit()` → `LightGBMError: Unknown token auto`。选该场景训练即硬崩。
- **验证**：对抗验证 confirmed/medium。
- **修复**：在 `apply_scenario`/lgb 配方里把 `'auto'` 解析成 neg/pos 比，或在接通不平衡处理前移除该 override。

### M3. `match_rate_for_method` 读全特征帧 + Python 逐行循环、无大小护栏（提 JOIN 主路径 OOM/慢）
- **位置**：[`marvis/data/backend.py:236-273`](marvis/data/backend.py:236)
- **问题**：读**整张**特征帧（`read_frame(feature_path, columns=feature_keys)` 无 nrows/sample），`feature_frame.iterrows()` 在 Python 里建 set，再逐行扫 anchor。只对 anchor 抽样，特征侧全量物化。该函数在提 JOIN 主路径被反复调用（`ColumnAligner._resolve_by_data` 每候选列、多键诊断、每个松弛备选）。子系统其它全帧读都受 `LARGE_ROW_THRESHOLD=200_000` 护栏，**唯独这里没有**。百万行特征表上是交互主流程的 OOM/严重延迟风险。
- **验证**：对抗验证 confirmed/medium。
- **修复**：特征侧也抽样/封顶，或把 normalize+match 下推 DuckDB SQL 用 semi-join 计数；至少像 `sample_rows()` 做蓄水池抽样并说明 match_rate 变估计值。

### M4. roll-rate 转移按 `time_col` 的字典序字符串排序（未校验）
- **位置**：[`marvis/packs/strategy/roll_rate.py:46-53`](marvis/packs/strategy/roll_rate.py:46)
- **问题**：`_adjacent_pairs` 用 `sort_values([id_col, time_col])` 后取相邻 (status_i, status_{i+1}) 当月度转移，`time_col` 从不 parse 成 datetime 也不校验格式。ISO `YYYY-MM` 字典序恰好等于时序，但非零填充/非 ISO（`2026-1/2026-10/2026-2`、`M/D/YYYY`）排序错乱，**静默产出错的 roll-rate 矩阵**且不报错。period 还硬编码 `'month'`。
- **验证**：对抗验证 confirmed/medium。
- **修复**：排序前 `pd.to_datetime(errors='raise')`（或校验已是 datetime/ISO），让非时序格式 fail-loud。

### M5. 策略条件等值比较未做 dtype 协调（字符串列上 `==`/`isin` 静默全 False）
- **位置**：[`marvis/packs/strategy/strategy.py:145-165`](marvis/packs/strategy/strategy.py:145)
- **问题**：`_eval_comparison` 把字面值直接对原始列 dtype 比较。数值分数列被读成 object/string 时（混合 CSV 常见），`< <= > >=` 会 TypeError（被捕获重抛 StrategyError，可接受），但 `==`/`!=`/`isin` 把 `'0'`(str) 比 `0`(int) **静默全 False 不报错**，该规则对所有行不命中 → 落 `default_decision`，回测里产出**貌似合理但错的**决策分布且无告警。
- **验证**：对抗验证 confirmed/medium（"经真实公开 API 端到端复现"）。
- **修复**：字面值为数值时先 `pd.to_numeric(series, errors='coerce')` 再比；或在 build/apply 期校验列 dtype 兼容并对不匹配抛 StrategyError。

### M6. `tool_dedup_rows` 策略路径静默塌缩不同的 NaN-键行
- **位置**：[`marvis/packs/data_ops/tools.py:222-224`](marvis/packs/data_ops/tools.py:222)（另见 `data/dedup.py:49`、`contracts.py:94`）
- **问题**：给定显式策略（first/last）且有冲突时跑 `drop_duplicates(subset=keys, keep=keep)`。pandas 在 subset 上把 NaN 视作彼此相等，**多行 NaN-键塌成一行**，即便它们不是同一实体。这与 `two_level_dedup`（`dropna(subset=keys)`、明确把 NaN-键行排除冲突检测）的设计相悖，且损失被埋进 `removed_rows`，用户看不出 NaN-键行被丢。
- **验证**：对抗验证 confirmed/medium（"实证复现"）。
- **修复**：策略塌缩只作用于完整键行：拆出 `df[keys].isna().any(axis=1)` 的 NaN-键行，仅对有键子集 `drop_duplicates`，再 concat 回未动的 NaN-键行。

### M7. `tool_woe_encode` 多特征共享一份手动分箱断点
- **位置**：[`marvis/packs/feature/tools.py:287-294`](marvis/packs/feature/tools.py:287)、`456-457`
- **问题**：`tool_woe_encode` 遍历每个特征调 `_edges_for`，`method='manual'` 时读**单个全局** `inputs['breakpoints']`，把同一断点套到所有特征。不同特征量纲/分布不同 → 其余特征 WOE 分箱无意义（多数行落一箱、IV/WOE 失真），多特征 manual WOE 请求被**静默错分**而非拒绝。
- **验证**：对抗验证 confirmed/medium。
- **修复**：`len(features)>1` 且 `method='manual'` 时报 FeatureError，或要求 per-feature 断点映射 `breakpoints_by_feature[feature]`。

### M8. 第三条自动写记忆路径（ConsolidationScheduler）绕过 `auto_distill` 开关
- **位置**：[`marvis/app.py:250-251`](marvis/app.py:250)（监听器装配）；`agent_memory/consolidation.py:28-55`、经 `evolution.py:10-18` 写入
- **问题**：`pipeline.py`/`api.py` 两条自动捕获都按 `load_memory_policy(...).auto_distill` 门控。但 `app.py` 把 `ConsolidationScheduler.on_event` 注册为 `validation.completed`/`report.after_generate`/`memory.after_save` 监听器，`on_event→_consolidate→DistillationEngine.distill_category + EvolutionManager.upsert_with_evolution` **会自动新建 memory_distillations 行，且全程不查 `auto_distill`**。用户即使关了 `auto_distill`，每次验证完成/报告生成仍静默跑蒸馏/进化写记忆 —— 一条用户以为已禁用的自动写记忆面。（与既有记忆"双面门控"记录相关：现在是**三面**。）
- **验证**：对抗验证 confirmed/medium。
- **修复**：把 `on_event` 自动路径按 `auto_distill` 门控（手动 `POST /agent-memory/consolidate` 保持不门控），在 `ConsolidationScheduler.on_event` 注入 policy 检查或在 `app.py` 注册处包一层门控回调。

### M9. 蒸馏检索排序在分数相等时不确定（无 ORDER BY / 无 tie-breaker）
- **位置**：[`marvis/agent_memory/store.py:502-516`](marvis/agent_memory/store.py:502)
- **问题**：`search_distillations` 跑 `SELECT * FROM memory_distillations {where_sql}` **无 ORDER BY**，再在 Python 里 `scored.sort(key=...score, reverse=True)`（稳定排序）。最终顺序取决于 SQLite 行返回顺序（无 ORDER BY 不保证、随版本/计划/vacuum 变）。同分（同 confidence+support 很常见）时 top-N 截断在相同输入下可能返回不同集合，记忆注入提示词影响下游 LLM，**把顺序泄进输出**，违反确定性不变量。
- **验证**：对抗验证 confirmed/medium。
- **修复**：SELECT 加确定性 `ORDER BY updated_at DESC, id DESC`，并把 Python sort key 改成含稳定次键的元组。

### M10. WOE 编码在全量数据（含 holdout/OOT）上拟合 —— 目标泄漏进编码特征
- **位置**：[`marvis/packs/feature/tools.py:273-306`](marvis/packs/feature/tools.py:273)；拟合在 `feature/iv.py:compute_woe_iv`
- **问题**：`tool_woe_encode` 经 `_read_frame` 读**整张**数据集，无 `split_col/holdout` 过滤，`compute_woe_iv` 在全部行上推 WOE 分箱/值并把 `*_woe` 列写回派生数据集（横跨 train+test+OOT）。WOE map 用到了 holdout 行的标签，污染列被持久化给下游建模消费。与同子系统 `screen_features`（`_dev_mask` 明确排除 holdout）相悖。WOE 正是"在 test/OOT 上拟合会抬高 KS/AUC"的监督变换，违反 INV-6。
- **验证**：对抗验证 confirmed/medium（"核心事实准确"）。
- **修复**：把 `split_col/holdout_values` 接入 `tool_woe_encode`，`compute_woe_iv`（edges+WOE）仅在 dev 行拟合，再 apply 到全部行（镜像 `screen.py` 的 `_dev_mask`）。

---

## 3. Low（已确认，22）

> 一致性/健壮性/性能/死代码。逐条含位置与修法，可批量清理。

1. **分层抽样可低于/高于请求量 n 且无信号** — [`data/sampler.py:61-86`](marvis/data/sampler.py:61)。`_allocate_strata` 缩减循环只减 >1 的层、加层循环满即停，返回行数不保证 == n。修：分配后按稳定键钳制 total，或文档化"近似 n"。
2. **`numeric_columns` 按子串判类型，误报** — [`data/backend.py:63-74`](marvis/data/backend.py:63)。`INTEGER[]`/`STRUCT(x INTEGER)` 被判数值后 `avg()` 出错。修：按基类型 token 精确白名单匹配。
3. **worker 失败用 `os._exit(0)` 掩盖退出码** — [`plugins/subprocess_worker.py:40,55`](marvis/plugins/subprocess_worker.py:40)。协议截断时父进程失去独立失败信号。修：失败路径用 `_hard_exit(1)`，协议 JSON 走专用 fd。
4. **`python_requires`/`permissions`/`side_effects` 校验了但从不强制** — [`plugins/manifest.py:77-80`](marvis/plugins/manifest.py:77)。半成品能力元数据给出虚假保护感。修：要么强制（运行前比 `sys.version`、把 permissions 接入运行期能力门），要么删字段。（见 §5）
5. **final-review LLM 判定被结构性忽略，`goal_met` 仅由步骤完成度决定** — [`orchestrator/reviewer.py:77-89`](marvis/orchestrator/reviewer.py:77)。所有步 DONE 即 goal-met，reviewer 无法以质量否决。修：`goal_met = 全步完成 AND llm goal_met != false`，或负判定走 `goal_doubt` 且给出解析路径。
6. **`_is_safety_step` 在 validator 与 executor 两处定义、语义分叉** — [`orchestrator/executor.py:534-537`](marvis/orchestrator/executor.py:534) vs `validator.py:277-282`。当前被上游遮蔽但是 latent bug 磁铁。修：抽到共享模块统一引用。
7. **vintage/roll_rate/profit 工具绕过 NaN 确认门、改为硬崩** — [`packs/strategy/tools.py:26-67`](marvis/packs/strategy/tools.py:26)。`tool_vintage_curve` 的 NaN 落到 `_parse_target` 抛裸 `ValueError`，且 manifest 只给 backtest/tradeoff 暴露 `drop_nan_labels`。修：vintage 的 `bad_col` 走 `resolve_labeled_frame` 并在 schema 暴露 `drop_nan_labels`。
8. **backtest swap 分析重复跑 `apply_strategy`** — [`packs/strategy/backtest.py:32-35,64`](marvis/packs/strategy/backtest.py:32)。大帧上把主策略算两遍。修：把已算 `decision/approved` 传进 `_swap_analysis`。
9. **`tool_ingest_excel` 读任意宿主路径、无 workspace 约束** — [`packs/data_ops/tools.py:21-28`](marvis/packs/data_ops/tools.py:21)。与 v1_compat adapters 的 `safe_relative_path` 约束不一致（单用户低危，但破坏 INV-4 纪律）。修：解析并约束在允许的 ingest root 下。
10. **缺列错误一会儿裸 KeyError 一会儿 FeatureError** — [`packs/feature/tools.py:477-479`](marvis/packs/feature/tools.py:477) 及 `tool_normalize/impute/cap`。修：入口统一 `_assert_columns`。
11. **复现性 harness 在非位置索引下错位** — [`validation/reproducibility.py:55-65`](marvis/validation/reproducibility.py:55) + `engine.py:35-38`。`sample.reset_index(drop=True)` 后用 `.loc[drawn.index]` 查带原始索引的 code_scores，过滤/JOIN 帧会 KeyError 或静默错配。修：两侧对齐同一索引空间并加行数/索引断言。
12. **validation head/tail lift 忽略风险方向** — [`validation/effectiveness.py:387-404`](marvis/validation/effectiveness.py:387)。反向模型 head/tail 互换，与 feature 层 `head_tail_lift`（按相关符号翻转）不一致。修：用 `risk_sign=sign(corr(scores,labels))` 排序。
13. **退化 AUC 返回 0.0 而非 0.5** — [`validation/effectiveness.py:376-377`](marvis/validation/effectiveness.py:376)。单类/空时 `compute_auc` 返 0.0，feature 层 `feature_auc` 返 0.5，跨层不一致。修：统一返 0.5 或 None。
14. **`_parse_dedup_instruction` 不处理否定** — [`agent/plan_driver.py:39-52`](marvis/agent/plan_driver.py:39)。"别用 first 去重"→`first`。修：出现 `别/不要/勿/不用/无需` 否定标记返回 None。（与 H4 同源）
15. **`_detect_continuous_target` 不排除 split/id 列、返回首个 token 命中** — [`agent/sample_setup.py:146-163`](marvis/agent/sample_setup.py:146)。`limit_flag` 等可能被当回归目标。修：镜像 multiclass 检测器跳 `_looks_like_split_name`/id/meta 列。
16. **`record_use` 接受 disabled 条目、记录检索永不会命中的"使用"事件** — [`agent_memory/store.py:545-548`](marvis/agent_memory/store.py:545)。修：`entry['status'] != 'active'` 时拒绝/no-op。
17. **model_experience 指标合并遇非数值会崩掉整组并静默丢蒸馏** — [`agent_memory/distillation.py:248-261`](marvis/agent_memory/distillation.py:248)。`float('N/A')` 抛错被 `except Exception: continue`（L116-119）吞掉整组。修：逐值 try/except 跳过不可解析项。
18. **per-feature 指标循环对 `compute_woe_iv` 硬抛无保护** — [`packs/feature/tools.py:63-72`](marvis/packs/feature/tools.py:63)。单个退化目标/单类列让整批指标 abort（`screen.py:142-145` 已有同款 IV 保护，此处缺）。修：循环前校验目标或 per-feature 包 try/except 降级。
19. **`feature_auc` 对单变量特征报原始（可 <0.5）AUC** — [`feature/metrics.py:38-39`](marvis/feature/metrics.py:38)。保护性特征显示 AUC<0.5，与同模块 sign-agnostic KS 不一致、混淆排序表。修：单变量读数报 `max(auc, 1-auc)`。
20. **PSI 对空质量箱地板平滑但不重归一化，虚高 PSI** — [`feature/metrics.py:110-112`](marvis/feature/metrics.py:110)。空箱贡献由 1e-6 地板量级驱动而非真实漂移。修：地板后重归一化，或对计数做 add-k 平滑。
21. **`_period_text` 在 start==end 时两套实现不一致** — [`report_texts.py:208-215`](marvis/report_texts.py:208) vs `excel.py:485-492`/`image_render.py:392-399`/`metric_tables.py:442-451`。Word 叙述显示 `2023-01-2023-01`，表格塌成单端点。修：抽一个共享 `_period_text` 并统一塌缩。
22. **死代码 `_run_at_offset` 从不被调用** — [`template_reports.py:187-194`](marvis/template_reports.py:187)。修：删除（解析由 `_run_for_placeholder` 负责）。

### 另含已确认 Low（来自二次验证回收）

23. **`update_join_spec` 跨两个连接 read-modify-write（JOIN spec 丢更新）** — [`db.py:1699-1721`](marvis/db.py:1699)。两个标签页/agent+UI 并发编辑同一 join plan，后写覆盖前写，可丢掉用户对破坏性 JOIN 的确认（spec.confirmed 存在同一 JSON blob）。修：load+update 放进单个 `BEGIN IMMEDIATE` 事务，或加 revision 乐观并发。
24. **`set_plan_status`/draft `set_status`/`confirm_step` 跨连接 TOCTOU** — [`db.py:1242-1257`](marvis/db.py:1242)。读状态断言后另一连接无条件 UPDATE，非法转移可落地。修：单连接 `BEGIN IMMEDIATE` + `AND status=<expected>` 守卫（镜像 `TaskRepository.update_status`）。
25. **WAL 降级仅 warning，并发保证可能静默丢失** — [`db.py:3109-3120`](marvis/db.py:3109)。网络/只读 FS 上 SQLite 静默回退到 rollback journal，单用户桌面 app 日志告警易被忽略。修：启动期健康检查显著标记有效 journal mode。
26. **`compute_vintage_report` 用逐行 `iterrows` 建长表（性能）** — [`packs/modeling/report_compute.py:124-135`](marvis/packs/modeling/report_compute.py:124)。O(rows×mob_cols) Python 行对象。修：`pandas.melt` 向量化。
27. **前端 plan 确认/运行、步骤确认无防重复提交** — [`static/js/v2/plan_confirm.js:58-86`](marvis/static/js/v2/plan_confirm.js:58)。快速双击重入 async；后端 job 锁挡住但抛原始 409 alert。修：await 期禁用按钮/置 in-flight 标志，或把 409 ACTIVE_JOB 译成"已在运行"。
28. **前端死 fan-out 守卫遮蔽后端消息** — [`static/js/v2/join_review.js:378-387`](marvis/static/js/v2/join_review.js:378)。后端 execute 恒返 `fan_out:False`、fan-out 走 409 FanOutError 被通用 catch 接住，curated 文案不可达。修：catch 里识别 409 渲染专用 fan-out 告警，或后端返结构化 body。
29. **前端破坏性记忆回滚无确认弹窗** — [`static/js/v2/memory_manager.js:203-213`](marvis/static/js/v2/memory_manager.js:203)。单击即回滚一版蒸馏，与 draft promote/plugin remove 的确认纪律不一致。修：加 `confirm()`。
30. **前端 draft run JSON 输入未校验为对象** — [`static/js/v2/draft_manager.js:454-475`](marvis/static/js/v2/draft_manager.js:454)。标量/数组/null 直接转发后端报模糊错。修：断言 plain object 否则抛"运行输入必须是 JSON 对象"。
31. **agg_max/agg_mean 去重逐列取极值、为冲突键造"不存在的行"** — [`data/backend.py:369-394`](marvis/data/backend.py:369)。level-2 冲突键被合成 Frankenstein 行。**按设计可接受**（ConflictReport 在 C2 门提示），仅提示风险。修：保持现状但确保 UI 在选 agg 且有 level-2 冲突时显著展示 ConflictReport。

---

## 4. 待最终验证（session-limited，9）

> 对抗验证因会话上限丢弃、且我未逐一直读核实。**附原始证据与置信度**，建议下一轮先验证再处置。

| # | 严重度/置信度 | 位置 | 摘要 |
|---|---|---|---|
| P1 | High / 0.72 | [`pipeline.py:1317-1354`](marvis/pipeline.py:1317) | 跨 Python 读 arrow 样本时经 `to_json/read_json(orient='table')` 中转，高精度浮点/int64+NaN/datetime/大整数键可能静默失真，而该样本正是后续 RMC_SAMPLE_DF 打分对象 → 影响 KS/PSI/复现性、违反确定性。修：用 parquet/feather/pickle5 无损中转，或子进程直接出分数；必走 JSON 则校验 dtype+逐列 hash。 |
| P2 | Medium / 0.70 | [`orchestrator/executor.py:256-261`](marvis/orchestrator/executor.py:256) | `goal_doubt=True` 时返回 `REVIEW` 但不置终态，re-run 经 L57-62 守卫早退、丢失原 summary_ref/FinalReview，plan 卡死无 resume 契约。修：定义 goal_doubt 的显式 resume 路径或持久化 review。 |
| P3 | Medium / 0.65 | [`orchestrator/executor.py:269-274`](marvis/orchestrator/executor.py:269) | 崩溃恢复把 `CHECKING` 步重置回 `PENDING`，但 CHECKING 意味着工具已 ok 执行（含 join 物化/写 artifact/训练），恢复会**二次执行**有副作用工具，违反确定性/数据完整。修：CHECKING 与 RUNNING 分别处理，有产出则进 review 阶段不重跑工具。 |
| P4 | Medium / 0.78 | [`data/backend.py:537-545`](marvis/data/backend.py:537) | 日期规范化 Python（`_canonical_date` 有 `pd.to_datetime` 兜底）与 SQL（仅 `try_strptime` over DATE_FORMATS 无兜底）不对称，Python 侧 match_rate 高估、可能让本应少匹配的键过 `MIN_KEY_MATCH_RATE`。修：两侧用同一日期文法。 |
| P5 | Low / 0.70 | [`data/join_engine.py:153-156`](marvis/data/join_engine.py:153) | 大特征表（>LARGE_ROW_THRESHOLD）`conflict_report` 置 None，C2 门无法展示 same-key 冲突（硬安全网仍在）。修：大表上用有界采样/`GROUP BY HAVING count>1` 出代表性冲突。 |
| P6 | Low / 0.60 | [`recovery.py:74-84`](marvis/recovery.py:74) | 回收陈旧任务后 `UPDATE jobs SET status='failed' WHERE status IN('queued','running')` **无 task_id/cutoff 范围**，启动期（stale=0）语义正确，但若以非零 cutoff 在运行中调用会误杀新 job。修：把 jobs UPDATE 按 reclaimed task ids/cutoff 范围化。 |
| P7 | Low / 0.50 | [`db.py:2885-2900`](marvis/db.py:2885) | `_normalize_loop_event` 用 `payload['type']` 硬索引（其余字段都 `.get`），缺 `type` 时 KeyError 致整个 replan 写事务回滚。修：`.get('type')` + 默认或边界校验。 |
| P8 | Low / 0.50 | [`static/js/v2/plan_view.js:238-289`](marvis/static/js/v2/plan_view.js:238) | `renderPlanView` 清理只退订 store，不 `stopPlanPolling`，离开未终态 plan 视图后 1s 轮询持续后台发请求（去重+最终自停，低危）。修：卸载时对 active plan id 调 `stopPlanPolling`。 |
| P9 | Low / 0.60 | [`orchestrator/validator.py:130-135`](marvis/orchestrator/validator.py:130) | `_check_join_gates` 只强制 `execute_join` 需确认（INV-3），不强制 fan-out/match_rate post_check（INV-2），全靠工具内部兜底，无 plan 级后备。修：对 execute_join 步强制要求 rowcount/match_rate post_check。 |

---

## 5. 未实现 / 半成品

1. **`reject_inference` 永远抛 `NotImplementedError`** — [`packs/modeling/reject_inference.py:4-10`](marvis/packs/modeling/reject_inference.py:4)。显式空壳（"需方法论评审，见 blueprint 15.1"）。**已正确门控**（调用即抛），属已知未实现，非偷偷半成品。处置：保持门控或在 UI/能力清单标注"未提供"。
2. **插件能力声明（`python_requires`/`permissions`/`side_effects`）只校验不强制** — 见 §3.4。给出虚假保护感，建议要么强制要么删。
3. **资源限额不全（仅 Linux RLIMIT_AS）** — 见 §2.M1。主开发平台 macOS 上内存界形同虚设。

---

## 6. 误报 / 已验证驳回（20+，不要去"修"）

> 这些经独立 skeptic（部分双人）或人工直读**驳回**。误改会引入回归。

- **【头条】vintage `cum_bad_rate` 不是累计 → 误报 ×2**：[`validation/vintage.py:86-95`](marvis/validation/vintage.py:86) 设 `cum_bad_rate=bad_rate`。两个独立 verifier + 人工核实 `report_compute.py:113-151`：生产路径输入是 `mob_observe_cols`（**每列即"到该 MOB 为止的累计坏标"，cumulative-by-construction 快照面板**），故每格 `bad_rate` 本就是累计率，`cum_bad_rate=bad_rate` 正确。与 4V 既定设计一致。**强行"累加边际"会把正确的累计语义改坏。**
- **`select_features` 在 train+test+OOT 上选特征（holdout 泄漏）→ 误报**：finding 引用的 `marvis/packs/modeling/select.py` **文件不存在**（reviewer 臆造路径），真实逻辑在 `tools.py:214 tool_select_features`。该具体发现无效。（注：真实的 WOE-encode 泄漏 = §2.M10 是真问题，勿混淆。）
- **handoff 生成 notebook 引用未定义 `RMC_SAMPLE_PATH` → 误报**：`RMC_SAMPLE_PATH` 是验证流水线在运行 notebook 时**注入的单元**，非 NameError。
- **绝对路径绕过 datasets-root 约束（path traversal）→ 误报**：`_resolve_path`(`backend.py:275-277`) 确实不校验，但所有真实调用点都喂受约束路径，无逃逸。
- **目标列检测只看 1000 行样本 → 误报**：检测主分支按名匹配语义角色，且下游有守卫。
- **`detect_header_rows` 短表 off-by-one → 误报**：前提被实证证伪（`range(1,limit)` 含最后一行）。
- **插件无文件系统约束 / checksum 不校验 → 误报（有意信任边界）**：单用户离线模型下文档化接受；checksum 仅 write 路径用于完整性记录。（但**环境密钥继承 = §1.H6 是真问题**，因为 LLM/DB 密钥不该暴露给 web-learned 草稿。）
- **默认异步 consolidation 共享 SQLite 致 'database is locked' → 误报**：代码无该并发写争用模式。
- **蒸馏 `source_task_ids` 泄进提示词 → 误报**：前提（"禁止 task id 进记忆输出"策略）不成立。
- **`search_distillations` 空查询返回全部 → 误报**：有意且有测试固定的行为。
- **`equal_frequency_edges` 塌成单箱无信号 → 误报**：`unique_count` 独立返回提供了信号。
- **roll-rate 丢自转移 → 误报**：转移矩阵分母定义如此。
- **income 场景 objective=regression 串到分类路径 → 误报**：`_assert_recipe_matches_target` 守卫，冗余但安全。
- **consolidate-memory 计数假设 object-of-numbers → 误报**：后端契约即此形状。
- **报告层 NaN/Inf 渲染成字面 'nan'/'inf' → 误报**：生产者无法产出该输入（上游已守卫）。
- **vintage/dict-table 表头只取首行丢列 → 误报**：行同构 by construction。
- **压测 slot 图>7 无占位 → 误报**：有无界 list 占位接收全部。

---

## 7. 未覆盖（需补审）

- **`orchestrator-support`**（context/budget/ledger、eval/scoring、templates）：复核 reviewer 因会话上限未跑完。预算/账本 token 计费数学、eval 评分聚合、模板 goal 路由完整性**尚未审**。
- **`api-core`**（`api.py` 4381 行 + routers）：复核 reviewer 未跑完。逐端点输入校验、`safe_paths`、任务 payload 校验、**反向代理伪造 127.0.0.1 绕过 local-only 守卫**（历史 S1 类问题）等**尚未在本轮复核**。
> 注：这两块在 6-13/6-14 历史多轮 review 已被深审过（见 [[marvis-codereview-debt]] 与 docs/reviews/ 旧报告），本轮只是 V2 增量未单独复跑。建议下一轮补上。

---

## 8. 建议修复顺序

1. **先修 6 个 High**（H1–H6）：NaN 门两处（H1/H2）→ JOIN 键空间（H3）→ 确认门否定（H4，连带 §3.14 与 `api.py:3217`）→ 子 agent 暂停（H5）→ 子进程环境密钥（H6）。这些直接关乎"上报错指标/破坏性 JOIN/密钥外泄"。
2. **再修 10 个 Medium**：M8（第三条记忆写面，用户预期之外）、M10/M5/M6/M7（泄漏与静默错值）、M2/M3/M4（崩溃与主路径性能）、M1（资源界）、M9（确定性 ties）。
3. **批量清扫 22+ Low**：确定性/一致性优先（#5/#6/#9/#13/#19/#20/#24/#25），其余健壮性与死代码随手清。
4. **验证 §4 的 9 条**（先验证再改，尤其 P1 arrow-json 与 P3 CHECKING 重跑，涉及确定性/数据完整）。
5. **补审 §7 两块**（`orchestrator-support`、`api-core`，含反代 local-only 守卫）。

### 修复纪律（来自本仓历史教训）
- 涉及 pandas/sqlite/lgb/duckdb **运行时行为**的发现，**实测再改**（本轮已挡掉 vintage、select.py、handoff、path-traversal、PSI 空箱等一批会"白改/改坏"的误报）。
- 每修一条补/改对应回归测试；全量 pytest 用 **miniconda env**（`.venv` 缺 PMML 依赖）。
- 数值/序列化层改动后跑两次随机种子顺序，确认确定性。

---

## 附：本轮统计
- 17 路 reviewer 产 77 条发现；对抗验证 confirmed 31 + 二次回收 confirmed 14（去重后净增 ~13）+ 人工直读确认 2（tune/subagent）。
- 误报驳回 20+（含头条 vintage ×2、臆造文件 select.py、注入式 RMC_SAMPLE_PATH 等）。
- 测试：canonical env 1568 passed（`.venv` 5 个 PMML 假失败已解释）。
- 限流/会话上限导致 9 条待最终验证、2 个子系统复核未跑完。
