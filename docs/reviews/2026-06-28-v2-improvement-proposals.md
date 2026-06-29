# MARVIS V2 前瞻性改进提案（多专家审查）

- **日期**：2026-06-28
- **分支**：`codex/v2-plugin-tool-runtime`
- **方法**：7 个专家镜头并行深读源码（架构 / Agent编排 / 产品UX / 前端视觉 / 信贷领域 / 性能 / 鲁棒性）→ 跨镜头综合排序 → 对抗式 critic 抽查证据并补盲区。
- **定位**：这一轮**不是挖 bug**（前两轮 `2026-06-21`、`2026-06-28-plan-code-review` 已深挖正确性缺陷），而是"怎样更高效 / 结果更好 / Agent 更聪明 / 体验更好 / 更专业 / 更解决真实业务"的升级建议。每条带 `file:line` 证据。
- **约束基线**：所有改动须守不变量 INV-1（确定性指标只由平台 tool 算）/ INV-3（JOIN 不静默）/ INV-4（记忆不改确定性）/ INV-6（子进程隔离）/ INV-8（审计完整）。

## 2026-06-28 落地状态更新

已落地并通过全量测试的条目：

- Agent JSON 鲁棒性：`decide_gate` / instruction router / reviewer 已改为 JSON 抽取与重试，避免弱模型输出围栏或前后缀文本时直接降级。
- Gate 安全：确认词识别已处理否定语义；screen gate 前端会回传 `expected_step_id`，后端对 stale screen 调参/选择返回 409；机器红旗 checklist 已注入 join/screen gate prompt。
- Reviewer / planner：软评审失败不再伪装 passed；`success_criteria` 已进计划与 final review；planner prompt 改为瘦 catalog + 示例。
- 调参方法：已纠正为不使用 OOT 做搜索目标，OOT 只报告不参与选择，避免间接窥视 OOT。
- 建模专业交付物：scorecard 已输出 base row、分箱 points、分数分段；scorecard PD 与 points 分离；默认单调分箱；CatBoost、sample weight、pkl+PMML 路径已接通；`select_features(space="woe")` 已支持训练 split 限定、WOE 空间相关/VIF、单调分箱和系数符号告警；LightGBM/XGBoost 已支持 top-level `monotone_constraints` 并记录规范化约束。
- 概率校准：新增 `calibrate_model`，支持 sigmoid / isotonic，记录 Brier、ECE、可靠性曲线和校准器，报告新增“概率校准”sheet。
- PMML 边界：lr / scorecard PMML 经 `pypmml` 校验；native LightGBM/XGBoost Booster 会明确拒绝 PMML 导出。
- Reject inference：新增建模 `reject_inference` tool，包含人工确认口径、报告标注和回归测试；不再是 `NotImplementedError` 空壳。
- INV-5 纵深防御：新增 `marvis.redaction`，active memory、memory distillation、tool stdout/stderr tail 已接入脱敏，并记录 `redacted_count` 或审计证据。
- INV-6 可观测性第一步：插件 worker 会把资源限制施加状态带回 protocol，runner 写入审计 `resource_limits`，不再完全静默。
- 性能/体验细节：JOIN match-rate 已将特征键扫描下推 DuckDB 并保留 hash fallback；Agent 消息轮询已支持安全场景下的 `after_id` 增量拉取，streaming/optimistic 场景保留全量拉取；artifact preview 的对象值改为键值摘要；artifact metrics / dataset preview 数字启用 tabular-nums；strategy / vintage 入口已接入真实 PlanDriver，旧 `data-coming-soon` 残留已清理。

仍未落地或只完成第一步的条目：

- Notebook kernel 隔离、RSS 软监控、真实 OOM 慢测仍未完成；当前只完成插件 worker 资源限制可观测化。
- DAG/checkpoint 续跑、失败后“从失败步继续”的后端指纹与前端 UX 尚未实现。
- 记忆历史 KS 锚点尚未接入 gate；WOE 空间特征选择、显式 LightGBM/XGB 单调约束 plumbing 和机器红旗 checklist 已实现。
- `api.py` / `db.py` 架构拆分、完整视觉 token 系统仍是后续大项。

---

## 0. 一句话诊断

**确定性骨架很扎实（真护城河），但卡在三个断层 + 三个可靠性缺口。**

骨架优点（已核实）：模板→DAG→post_check→强制确认门主干清晰；JOIN 引擎双重防线（`join_engine.py:317` 强制 `joined_rows==anchor_rows`，异常分支 unlink 产物）；validator 静态校验是真护城河；插件/pack 是真正的深模块（加一个 tool 只动 2 个文件）；SQLite 已正确开 WAL+busy_timeout。

三大断层：
1. **智能层偏薄** —— 所有 LLM 触点（`decide_gate` / `route_instruction` / `critic`）都是单次 JSON、无重试、无 few-shot、无历史锚点；记忆系统完整却对 V2 规划/门决策**零接入**（`plan_driver.py`/`templates.py` 中 `memory_context = 0` 引用）。在本地 32–72B 弱模型上直接表现为"agent 变笨 / 莫名停下 / 听不懂"。
2. **信贷专业交付物缺位** —— 评分卡是"伪评分卡"（算了 factor/offset 却从不折算 points，scorer 直接返回 `predict_proba`）；无强制单调分箱；无概率校准；reject inference 是 `NotImplementedError` 桩。直接挡住"能定价 / 可上线 / 可过审"。
3. **视觉系统化缺位** —— 门面（welcome / task-hero）已建立玻璃+色调高级语言，但 V2 业务证据面板几乎全是 flat 同质行；设计 token 名存实亡（`--radius` 语义档塌缩、`--space`/`--fs-` 各 0 命中、无 skeleton）。

三个可靠性缺口：**DAG 无断点续跑**（MODELING 崩溃→前功尽弃）、**seed 未落盘且默认值三处分叉**（0/42/23，违反 INV-1 可复现）、**Notebook kernel 隔离最弱**（只有 timeout，是 INV-6 最易被击穿处）。

ROI 顺序：先做一批 S 级速赢（救活弱模型体验 + 直接服务模型效果目标），再战略投入建模深度与执行护栏，架构/视觉系统化作为长线。

---

## 1. 速赢（S 成本、高 ROI，建议第一批）

| # | 提案 | 镜头 | 影响/成本 | 证据 |
|---|------|------|-----------|------|
| 1 | `decide_gate`/`route_instruction` 加 JSON 抽取 + 一次重试（抄 planner 已有范式） | agent | high/S | `auto_drive.py:35-42`、`instruction_router.py:38-45` 单次 `complete` 无重试；`planner.py:124-156` 已有 `max_retries+1` 回灌 `last_error` |
| 2 | `is_confirm` 收紧：转折/否定词不判确认 | agent | **high**/S | `plan_driver.py:29-36` `_CONFIRM` 用 `.search` 且含"好的/可以"；"好的但先别执行"→命中"好的"→在 `AWAITING_CONFIRM`(L337/349) 真左连接。critic 已确认这是 INV-3 直接击穿、数据正确性事故，不是体感问题 |
| 3 | ~~`tune.py` 纳入 OOT 稳定性惩罚~~ **（误报，已撤回）** | domain | — | 复核确认：`tune.py` 本就**正确地把 OOT 排除在调参之外**（docstring 明写 OOT 仅报告不参与选择；`_trial_score` 只用 train/test；按 `score` 选 best）。这是正确方法学（防间接窥视 OOT），无需改。审查把正确行为误读成缺陷 |
| 4 | seed 端到端贯通：统一默认值 + 落盘 `model_meta.json` | arch | high/S | 默认值三处分叉 `pipeline.py:65=42`/`modeling_setup.py:43=23`/`api.py:1018=0`；`_write_model_meta_from_contract`(`pipeline.py:1372`) 不写 seed，`notebook_contract.py` 无 seed 字段→artifact 不可复现（INV-1/2/8） |
| 5 | KS/AUC/IV 补"带阈值解读 + 基准线"，对齐 PSI 标杆 | ux | high/S | PSI 有三阶段色条+tooltip(`render-metrics.js:84-98`)；KS 无基准参考线；**IV 最差**：`app.js:6880-6910` 纯 4 位小数无条形/排名/tooltip |
| 6 | 风控数字全面 `tabular-nums` + 指标列右对齐 | visual | med/S | `tabular-nums` 仅 6 处命中；metric-table/join-card 的 KS/PSI/IV/行数未启用，逐行对比易看错档位 |
| 7 | `_apply_adjust` 未匹配参数不再伪报"已调整" | agent | med/S | `plan_driver.py:261-291` overrides 为空时仍回显"已按指令调整参数 {params} 并重算"，制造"说了等于做了"幻觉 |
| 8 | Strategy/Vintage 占位入口改真禁用 + 需求收集 | ux | high/S | `app.js:596-620` 卡片未禁用点击、只闪 2.4s toast，反复可点=读作"按钮坏了"；改 `cursor:not-allowed`+锁图标+常驻面板+"上线通知我" |

---

## 2. 按主题的改进建议

### A. 弱模型鲁棒性：让 32–72B 的 LLM 触点不再一抖就降级

- **门决策/路由重试 + JSON 抽取**（见速赢 #1）：剥离 ```json 围栏、取第一个 `{...}` 块再 `json.loads`，失败回灌一次，最终仍 fallback 到安全态。
- **`reviewer` 软闸恢复真实价值** [med/M]：`reviewer.py:68-69` except→`passed=True`、`_parse_soft_verdict` 默认 `passed=True`，`executor.py:161` 只 append verdicts 从不影响 step——整条第二道复核闸**空转**。改：失败明确标"软评审不可用"而非伪 passed；`not-passed` 且 reasons 非空时以 ⚠️ 上浮到 done/gate 消息（不阻断）。弱模型最需要"第二双眼睛"。
- **planner few-shot + catalog 瘦身** [med/M]：`planner.py:299-325` 直接灌每个工具完整 input/output schema 原文且零示例，`$ref` 接线规则无样例。改：给 1–2 个合法 step 范例 + 只给工具 summary/入参名/required，完整 schema 留给 validator。提升自由规划/重规划成功率、省 token。

### B. 把记忆 + 机器红旗 + KS 基准接进规划与门决策（直击模型效果目标）

- **门决策注入机器红旗 checklist + 历史 KS 锚点** [high/M]：`auto_drive.py:19-56` 只把 gate content/tables 拼文本喂 LLM，无 `plan.goal`/阈值/历史基准，让弱模型自行判"命中率极低/指标异常"却无任何数值锚点。改：把平台已算出的红旗（`execute_join` 的 `anchor_rows==joined_rows`、`propose_join` 的 match_rate/fan_out/needs_dedup、screen 的 leakage 计数）显式列成 checklist 让 LLM **复核**而非重新发明；附记忆里同类模型 KS 区间。红旗须来自平台确定性输出（守 INV-1），摘要限长防超窗。
- **`final_review` 对标可配置的 `success_criteria`，未达触发自我重规划** [high/M]：`reviewer.py:77-89` `goal_met = not incomplete`（全 step DONE 即达标），KS 很低也判 DONE，模型效果目标从未进系统。改：模板携带**由任务/数据集传入、不写死任何数值**的 `success_criteria`（如 `{metric:'oot_ks', min:<按数据集配置>}`），从终态 step 取指标比较写进 `goal_met`/summary；未达标配 `decision_point_replan` 触发一次重规划（受 `max_replan_iterations` 约束防刷指标）。
- **modeling/feature driver 起点注入历史同类实验作只读锚点** [med/M]：`retrieval.py:101 retrieve_with_distillations` 完善，但 `plan_driver.py`/`templates.py` 中 `memory_context = 0` 引用。改：driver turn 起点用 `compare_model_experience` 拉同 scope/family 历史指标作门决策锚点/调参参照/final 对比。严守 INV-4（只读不改确定性）+ confidence 过滤 + 来源标注。

### C. 信贷专业交付物：从"能排序"到"能定价 / 可上线 / 可过审"

- **评分卡补 points 分数刻度 + 分段表（当前是伪评分卡）** [high/M]：`scorecard.py:61-62` 算了 factor/offset 但 `_save_scorecard_model` 只存 LR+woe_maps，`tools.py:1149-1154` scorer 对 scorecard 直接返回 `predict_proba`，无 points/score-band。改：训练后用 `points_i = factor*(-woe_i*beta_i)` 折算每分箱分数，写 artifact 一张 `scorecard_table(feature,bin_range,woe,coef,points,bin_bad_rate,bin_count)`；scorer 增 scorecard 分支按 points 求和+offset；report 增"评分卡明细表"+"分数分段排序性"，清楚标注"分数"与"PD"两套刻度。
- **分箱单调性：从仅打标升级为可强制 `enforce_monotonic`** [high/M]：`iv.py:90` 仅把 `_is_monotonic` 作只读标志；`binning.py:51` chimerge/tree 都不保证 bad_rate 单调，非单调箱被静默放进模型。改：新增 `monotonic_binning(values,target,direction='auto'|...)` 对 chimerge/tree 结果做"违反单调则与相邻箱合并"后处理；`tool_bin_feature` 增 `enforce_monotonic`；scorecard 默认强制单调（方向由 corr 自判），回传单调前/后 IV 供权衡。同时利于 OOT 稳定。
- **概率校准（isotonic/sigmoid）+ Brier/可靠性曲线** [high/M]：全 packs 无 calibrat/isotonic/brier，`common.py compute_model_metrics` 只算 KS/AUC/PSI。未校准的 LGB/XGB 概率不可用于定价/拨备/IFRS9。改：新增 `tool_calibrate_model` 在**独立校准集**上拟合，产校准后 scorer + Brier/ECE/10 段可靠性曲线；report 增"概率校准"节；artifact 记录校准器。需独立校准集或交叉拟合避免与早停串用，isotonic 设最小样本门。
- **特征选择移到 WOE 空间 + 系数符号校验** [high/M]：`select.py:39-67` 在**原始列**算 IV/相关/VIF，而 scorecard 在 `_fit_woe_maps` 后用 WOE 入模——选择空间≠建模空间，raw-VIF 在偏态/含缺失的信贷特征上严重失真、系数符号可能反向。改：`select_features(space='woe')` 先拟合 WOE 再在 WOE 矩阵算 IV/相关/VIF + "LR 系数与 WOE 单调方向相悖则告警"；raw 空间保留给树模型。
- **LightGBM/XGB 单调约束 plumbing** [med/S]：`lgb.py:24-38` 无 `monotone_constraints`，原生支持却未接出。改：`TrainConfig` 增可选 `monotone_constraints`（可由 WOE 方向自动推断），recipe 透传，report 校验分数分段坏账率单调性。
- **Reject Inference 从桩落地为 parcelling** [med/L · critic 建议排在评分卡/单调/校准之后]：`reject_inference.py:4-12` 直接 raise。落地 parcelling：KGB 模型对拒绝样本打分→按分段推断标签/权重→并入重训，强制人工确认 + 留审计 + 结果标注"含推断标签"。
- **治理三件（M 级，长线专业度）**：模型卡自动生成（聚合 config/metrics/保留剔除原因/PSI/校准/单调，渲染 Markdown+Excel）；群体公平性分群诊断（按渠道/地区/产品拆 KS/坏账/通过率 + disparate-impact 比率，只输出聚合守 INV-5）；打分原因码（评分卡按 points 贡献给 top-N 负向原因，满足 adverse-action 合规）。

### D. 执行护栏系统化：把点状护栏做成系统

- **Notebook kernel 复用插件的资源/OOM/编码护栏** [high/M]：`notebooks.py` 只有 timeout，grep 无 setrlimit/start_new_session/MALLOC_ARENA/OMP；而建模（用户核心目标）恰跑在此最弱路径，一次失控 self-join/读 10M 行就吃光内存拖垮 FastAPI 主服务=单机产品不可用。kill 仅 `shutdown_kernel(now=True)` 未对进程组 SIGKILL。改：抽统一"执行护栏契约"让 kernel 复用——KernelManager preexec 调 setrlimit+start_new_session、设线程/MALLOC/编码 env、close/cancel 对 pgid SIGTERM→SIGKILL，内存上限与 manifest `memory_limit_mb` 同源；Windows 软监控降级（INV-9）。
- **内存护栏 RLIMIT_AS→RSS 软监控，失败不再静默** [high/M]：`subprocess_worker.py:108-117` 只设 RLIMIT_AS 且 `except(...):return` 静默放弃；RLIMIT_AS 限虚拟地址空间，numpy/OpenBLAS 预留大块虚存会在远未用满物理内存时误杀，或真 OOM 时以非 MemoryError 崩→分类失真；Windows=零隔离。改：RLIMIT_DATA + 父进程 psutil RSS 软监控（超阈值→SIGKILL 标 `error_kind='resource'`）；施加失败把"未施加内存限制"写进 protocol meta + 审计使无隔离可观测。
- **审计去 hasattr 软降级 + 与业务写入同事务原子化** [med→critic 上调/M]：`join_engine.py:363-366` `_write_audit` 先 `if not hasattr(repo,'write_audit'):return` 静默不留审计；审计与业务写入不原子（写完产物审计失败→有结果无凭证）。直接违反 INV-8。改：构造期强制注入支持 write_audit 的 repo（缺失即启动失败）；审计与 `set_join_plan_executed` 同 SQLite 事务原子提交；加 audit 完整性自检校验关键 kind 齐全。
- **INV-5 敏感数据脱敏 choke point** [med/M]：`agent_memory/store.py:75/119` 直接 INSERT 原始 content/metadata，无任何脱敏；`runner.py` stdout/stderr_tail 截 4000 字符可能含打印的明细行；`auto_distill` 默认 True 把用户每条消息当候选直存。改：建 `marvis/redaction.py`（身份证/手机/邮箱/银行卡正则 + 列名黑名单）作三个写入 choke point 强制过滤，留 `redacted_count` 使"确实脱敏"可观测；定位为纵深防御只脱值保结构。
- **隔离/不变量集成测试：消除"测了路径未断言不变量"伪绿** [med/M]：`test_plugin_runner.py:152` 只断言 `result.ok` 未用 psutil 查残留 PID；`:162` monkeypatch `os.killpg` 成 append 只验"调了"不验"真杀"；`test_drafts_sandbox.py` grep RLIMIT/MemoryError=0。改：建 robustness-invariant 慢测组纳入 CI——起真子进程 fork 孙进程超时后 psutil 断言整树死、真分配大数组断言 `error_kind=='resource'` 且主进程存活、join 构造 fan-out/row-loss fixture 断言抛错且产物 unlink、任一 tool/JOIN/memory 写入后断言 audit 表有对应行。
- **统一可观测面** [med/L · 分阶段]：错误 kind 字符串散落手写、隔离降级静默 return、无回合级 trace。改：建 `marvis/observability.py` 集中 `ErrorKind` 枚举 + `IsolationLevel(full/degraded/none)` + 结构化事件 `emit(trace_id=task_id+turn)`，所有静默降级点改 emit 一条 degraded 事件并计数；前端给轻量"执行健康"玻璃质感卡片（兼顾用户口味）。先落枚举+降级事件，面板后做。
- **路径安全贯穿写入** [med/M]：`assert_within` 仅守 API 边界（`safe_paths.py`/`api.py`），join 产物/notebook 输出/插件 workspace 写入无统一沙箱根校验。改：定义 task `sandbox_root`，所有产物写入经 `SafeWriter.open` 强制 `assert_within`；子进程 worker 内提供受限 `ctx.write_path` 二次校验。面向未来插件生态必备。

### E. 架构与可维护性：拆 god-module、补续跑、收口双编排

- **从 `api.py` 抽出 Agent 编排服务层（拆 god-module 第一刀）** [high/L]：`api.py` 4381 行、177 个 `def`、仅 47 个 `@router`，约 40 个是 agent 编排且**并存两条互不复用的编排主轴**（V1.1 `_run_agent_*_stage` 家族 `api.py:2453-2729` 与 V2 `_run_*_driver_turn` 家族 `api.py:3044-3460`），还直接 import `pipeline.py` 私有 `_clear_generated_artifacts`(`api.py:128`)，三层焊死、无法脱 HTTP 单测编排。改（分批可验证）：先抽 V2 驱动家族→`agent/driver_service.py`（签名换 `DriverDeps` dataclass），再抽 V1.1 验证自动机→`agent/validation_pilot.py`，最后 memory 编排→复用 `agent_memory/`。路由层只剩薄封装。
- **修 DAG 无断点续跑：MODELING 崩溃强制重跑 JOIN+FEATURE** [high/M]：`run_staged_pipeline`(`pipeline.py:1008`) 每次从头全量重跑，入口 `_clear_generated_artifacts(stage="notebook")`(`pipeline.py:1399`) 连带删下游 metrics 产物；plan-step 状态机与 pipeline stage 是两套独立机制，前者续跑不覆盖后者。改：先做"收窄删除范围只删 notebook 自身产物"止血；再加 stage 完成指纹（含源 hash+slot+seed）幂等门，命中则 skip。指纹须含 seed 守 INV-1。
- **seed 落盘 + 统一默认值**（见速赢 #4）：收成单一 `DEFAULT_RANDOM_SEED` 常量；`model_meta.json`/contract/reproducibility 都带实际 seed。
- **Agent 消息轮询无游标的全表重读** [med/M]：`app.js:246` 每 180ms 轮询、`db.py:1023` `list_agent_messages` 每次全表 SELECT 无游标、前端 `app.js:5947` full-rebuild 渲染。改：加 `after_id`/`since` keyset 游标只拉增量 append，建复合索引；注意 `update_agent_message` 原地更新无新 id，对可变的最后一条 thinking/draft 单独全量刷新。
- **拆 `db.py` 按领域** [med/L]：3135 行含 7 个 Repository + 40 个序列化助手 + 514 行 schema/migration。改：建 `marvis/db/` 包按**领域**垂直切（tasks/plans/datasets/modeling/strategy/drafts/plugins + connection），`__init__.py` re-export 保持调用方零改动，先抽 connection。
- **统一错误处理与日志** [med/M]：`api.py` 85 处手写 `HTTPException`、`except` 类型散乱（`except Exception` 13 次、裸 `except KeyError` 15 次混淆语义）；`pipeline.py` 0 处 logger。改：建 `marvis/errors.py` 领域异常基类 + FastAPI exception_handler 集中映射；pipeline 每个 stage 边界加 `logger.exception`（先做日志增量、零行为变更）。
- **收敛 orchestrator/agent 双编排** [med/L · 留到最后]：明确单一执行权威=`orchestrator/executor.PlanExecutor`，agent 只负责"对话↔slot↔gate"，让 V1.1 验证也走 plan-step DAG 与 V2 共脊，消除"加一个 stage 改 3 处契约"税。

### F. 端到端体验：把"出问题之后"和"看不懂之后"收口

- **失败后给"从哪一步续跑"的明确出路** [high/M]：`recovery.py:21-37` 仅服务重启时按磁盘推断阶段，运行内无 step checkpoint；`api.py:1871` 只有 metrics 允许一次重试，notebook/JOIN 失败必须从 SCANNED 整体重跑；`app.js:1846-1896` 失败卡片没有"从失败步继续"档。改：失败卡片分三档 [从失败步重试]/[改参后重试]/[从头重建] 默认高亮第一档；附"已保留的中间产物"清单；后端先在每个 gate 确认点持久化已完成 step 输出指针。（依赖 E 的续跑能力，需先定产品边界）
- **KS/AUC/IV 阈值解读**（见速赢 #5）：抽 PSI 的分级色条为通用组件套到 KS/AUC/IV，KS 把任务配置的基准（如有）画成参考线，每指标加一句"衡量什么"tooltip。
- **占位入口改真禁用 + 需求收集**（见速赢 #8）。
- **无障碍系统补课** [high/M]：状态全靠红绿色编码（`styles.css:19-24/97-102`，`.action-error-detail.error` 只换红无图标）；`app.js:2454,3030` 等 `role="button"` 的 div 只有 onclick 无 keydown→键盘激活不了；dialog 无焦点陷阱。改：状态色一律配 ✓/△/✕ 图标+文字（不靠色相单维度，**这也是引入 tonal 状态色后色弱可辨的前提**）；全局给 role=button 加 keydown 委托或换原生 `<button>`；dialog 加焦点陷阱+关闭还焦点；校验 dark 模式 danger/success 对比度达 AA。
- **Loop 进度从"事件日志"升级为"我在哪一步"** [med/M]：`loop_progress.js:20-48` 只渲事件流无当前步/进度/ETA；长任务黑盒等待易让人误判挂了去强杀。改：顶部加"第 N/共 M 步:正在做 X"当前态条（与 `notebookStepRail` 统一成一个 stepper），`no_progress` 累计达阈值给"可介入/调参/终止"提示，给粗略 ETA。
- **gate 强制选择缺默认推荐 + Manual 模式拒绝是死路** [med/S]：`join_review.js:160-169` 去重下拉首项"需要选择"无默认无"first/last 差别说明"；`plan_driver.py:300-313` Manual 模式非"确认"的自由文本只回"请回复确认"，不能改参/编辑/重规划（adjust/replan 仅 Agent 模式）。改：去重给带依据的默认推荐（如按时间列默认 last 标"按 apply_time 取最新"）；Manual 拒绝时列出 [调参重算]/[编辑特征集]/[转 Agent 协助] 按钮（`screen_payload` 已含可编辑特征/阈值）。
- **产物表格 JSON.stringify 违反"人类可读"** [med/S]：`artifact_view.js:14-19` 对 object 直接 `JSON.stringify` 包进 `<pre>`，`{mean:0.532,std:0.042}` 直曝花括号。改：对 stats dict/区间/列表键值化人类可读，无法识别的给折叠"查看原始值"。
- **agent 陈述无证据回链** [med/M]：`render-agent.js:250-260` 只支持外链无内部引用；gate metadata 已带 `step_id`/`plan_id`/`tables` 具备回链锚点。改：陈述支持内部锚点 `[查看筛选明细](#step:screen_features)` 点击高亮对应 step 证据，落实 DESIGN.md"每个陈述可追溯"。

### G. 性能：去掉高频热点固定开销（守 INV-1/INV-6）

- **JOIN 匹配率改 DuckDB 向量化** [high/M]：`backend.py:236-273` `match_rate_for_method` 对 anchor/feature 各跑一遍 `for _,row in frame.iterrows()` 逐行归一化，`read_frame(feature_keys)` 全量读入键列无 LIMIT；同文件 `_sql_transform` 已有等价 SQL 归一化。改：重写为一条 DuckDB 查询（anchor 用 `USING SAMPLE reservoir REPEATABLE(seed)`、feature `SELECT DISTINCT` 归一化键做 semi-join COUNT），保留 Python 仅作 `.feather` fallback，加断言/测试保证两路径逐位一致（否则 match_rate 漂移会改 C2 门人工决策）。
- **Agent 消息轮询增量游标**（见 E）。
- **复用 SQLite 连接，去掉每次 PRAGMA-WAL 重握手** [med/M]：`db.py:3109-3135` 每次 connect 都跑 journal_mode=WAL+4 条 PRAGMA，几十处 `with connect()` 叠加 1s+180ms 高频轮询。改：threading.local 缓存连接首次跑 PRAGMA 之后复用（sqlite3 连接不能跨线程），保留 with 用于事务边界。⚠️ critic 盲区：需验证高频轮询读 + 长 JOIN 写事务下 busy_timeout 是否把读饿死。
- **backend 元数据按路径+mtime 缓存** [med/S]：`column_names`/`row_count`/`numeric_columns` 在一次 JOIN 流程被反复 DESCRIBE（CSV 的 read_csv_auto 推断尤贵）。改：以 `(path, mtime, size)` 为键缓存只读派生信息，文件指纹变即失效（不影响确定性）。
- **DuckDB 全局连接配置 threads/memory_limit** [med/S]：全仓未配置，默认吃满核数+按系统内存比例，重 JOIN 可能饿死 UI 轮询或与训练子进程争内存。改：集中配置可调上限连接并复用；`left_join` 的 `anchor_rows` 从上层已知值传入省一次 count（保留对 out_path 实际 count 守 INV-3）。
- **插件子进程预热常驻 worker 池** [high/L · critic 降级为"可选、排在正确性/隔离/交付物之后"]：`runner.py:315-332` 每调用 Popen 新进程、`tools.py:9-10` 顶层 import numpy/pandas 付冷启动。改（最低增量）：只做"解释器+pandas 预热"forkserver 风格。⚠️ 与 INV-1/INV-6 双张力（job 间 seed/全局态/已 import 插件污染），第三方插件仍走一次性进程。

### H. 视觉系统化：从"门面高级、干活区简陋"到一致状态语言

> ⚠️ **用户口味优先**：用户明确偏好玻璃/立体/动画/吉祥物（高于 DESIGN.md "避免玻璃效果/重阴影"旧基调）。以下提案在尊重该口味前提下做高级化；与 DESIGN.md 的张力已标注。

- **V2 证据面板继承 hero 色调状态语言** [high/M]：`v2-workbench.css:191-204` plan-step/join-card/loop-evt 全 flat 1px border+10px；`styles.css:3684` task-hero[data-tone] 已有 tone-glow、`welcome.css:95-108` 已有 6 类 --tone，两套语言互不相通；`join-card.has-warn` 只染红 border 无 tonal 晕染。改：为 plan-step/loop-evt/join-card/subagent-row 引入 `[data-state=running/ok/warn/fail/review]`（左侧 3px 状态色条 + 极淡 tonal 背景 color-mix 8-12% + 状态点），合并 welcome/hero 调色板成一套 status token。把 INV-3 告警从"文字+红框"升级成可扫读视觉信号。**必须配图标做颜色+形状双编码**（见 F 无障碍），并校验 dark 对比度。
- **补齐 skeleton/shimmer 加载骨架** [high/M]：全仓 grep skeleton/shimmer=0（`app.js:5960` 的 skeleton 变量是消息 diff 签名，与加载无关）；长任务加载只有 `styles.css:177` opacity:0 淡入→证据区变空白无法判断"加载还是卡死"。改：新增 `.skeleton` 工具类（shimmer + 尊重 prefers-reduced-motion 降级静态），为 metric-table/join-card/plan-steps/thinking 区渲染形状骨架；只首屏用骨架、轮询更新就地 diff 不每帧重建。
- **吉祥物从静态 logo 升级为情境化品牌资产** [med/M · critic 点名补回的口味盲区]：`styles.css:1338-1481` 已有完整 sprite 系统（7 mood + 8 方向帧）、`pets/` 7 套皮肤，但只被动随任务状态切。改：用现有 sprite-sheet 机制叠（零新资源）——idle 超 N 秒触发 peek/stretch 彩蛋；JOIN 膨胀告警切 worry 并朝告警卡偏头；KS 超基准 proud/celebrate；皮肤切换做成缩略图网格选择器。与证据面板 `[data-state]` 状态语言联动。worry 表情要克制不喜感，尊重 prefers-reduced-motion + 关闭开关。
- **风控数字 tabular-nums**（见速赢 #6）。
- **深色模式补齐玻璃/状态质感对等** [med/M]：`styles.css:4029` dark task-hero inset 高光从 0.72 砍到 0.08，玻璃顶光几乎消失；governance 面板 dark 纯 flat。改：dark 用克制双层 inset 重建玻璃边缘、tone-glow 适度提亮。与 DESIGN.md 张力，以用户口味优先。
- **重建 radius/spacing/type 三套真 token** [high/L · 需先出对比视觉稿]：`styles.css:34-37` `--radius` 语义档塌缩（注意 `--radius-control=10` 是工作的，别误删）、`--space`/`--fs-` 各 0、font-size 散落 13 种字面值含 12.5px。改：radius 恢复语义梯度、引入 `--space-1..6`(4/8/12/16/20/24)、`--fs-xs..2xl` 对齐 DESIGN.md 五档。⚠️ **radius 收紧会让圆角变小、可能动到用户偏好的圆润玻璃口味——必须先出对比稿让用户拍板，绝不擅自收紧。**
- **focus 环 token 化 + 响应式断点收敛** [low-med/M]：54 处 focus-visible 颜色来源不一（有的硬编码蓝、有的用 token），14 个离散断点彼此不对齐。改：抽 `--focus-ring` token 跟随品牌色、收敛断点到 3-4 个语义档。

---

## 3. 需要你拍板的取舍

1. **视觉口味 vs DESIGN.md**：tonal 状态语言 / dark 玻璃高光 / 吉祥物情境动效都踩 DESIGN.md "避免玻璃/重阴影 + compact 密度"旧基调。**默认以你口味优先**，但 **radius 收紧必须先出对比稿**——别让"设计系统规范"的审查直觉碾压你的圆润玻璃口味。
2. **新增 GUARDED 中间确认档？** agent 镜头提议"平台无红旗自动过、有红旗/安全步才停"。critic 反对**现在**做：自动放行非安全步完全依赖机器红旗覆盖完整，漏判一类异常就静默过，与 INV-3 正面冲突。建议**等机器红旗 checklist 成熟并验证后再议**。
3. ~~**调参纳入 OOT 的方法学张力**~~ **（已澄清，无张力，条目作废）**：审查曾建议把 OOT 纳入选参，但复核确认 `tune.py` 本就正确地**把 OOT 排除在调参之外**（仅报告、不参与选择），符合"OOT 应盲测"的方法学。无需取舍。
4. **记忆接入 vs INV-4**：注入历史 KS 补弱模型能力，但旧 KS 被误当本次结果会违反 INV-4。须严格只读 + 来源标注 + confidence 过滤。
5. **失败续跑边界**：只"从失败 stage 重跑"（便宜）还是真"从失败 step 续跑保留 kernel/中间表"（贵但体验质变）？决定 F-1 与 E-2 的改造范围。
6. **Manual 模式是否应获得与 Agent 对等的编辑/调参能力？** 还是刻意保持 Manual=只读确认以区分两档？这关系 F-gate 提案是产品定位还是缺陷。
7. **先补专业交付物（直接业务价值）还是先补隔离护栏（防灾）？** 两块都是 M 级战略投入。

---

## 4. critic 标出的盲区（值得补审）

- **可复现性/seed 端到端贯通**（已被架构镜头覆盖为速赢 #4——seed 默认值三处分叉且不落盘 artifact，动摇 INV-1 与模型结果可验证性）。
- **数据切分泄漏**：未系统审 train/test/oot 是否按**时间**正确切、`woe_maps` 是否只 fit 在 train、early-stopping 集与 OOT 是否串用。**WOE 在全量 fit 再切分是评分卡最经典泄漏源**，值得专项一条。
- **并发/单写者**：SQLite 连接复用后，高频轮询读 + 写 + 长 JOIN 事务下是否 `database is locked` 或读饿死——性能改造的反作用面。
- **错误恢复/断点续跑**（已被架构+UX 覆盖）：长链路 DAG 中途 kernel OOM 后会话能否续跑。
- **前端无障碍/对比度**：dark 模式下 tonal 极淡背景 + 状态色的 WCAG 对比度，色弱/低对比下"扫一眼红绿"若不可辨反而是专业度倒退（已被 UX 无障碍提案覆盖，但 dark 对比度需专门验证）。
- **成本/延迟**：给弱模型加重试 + JSON 抽取 + 记忆注入 + 红旗 checklist 会显著增长每次门决策 prompt 与调用次数，需估算对本地 32–72B 推理延迟与上下文窗口的累积压力（记忆+红旗+历史 KS 同时注入易超窗）。

---

## 5. 建议落地顺序

- **第一批（止血 + 速赢，几乎全 S）**：速赢 #1、#2、#4–#8（#3 已撤回为误报；含 seed 统一+落盘、artifact 删除收窄止血）。一两周内救活弱模型体验 + 消除信任杀手。
- **第二批（建模专业深度）**：评分卡 points → 强制单调 → WOE 空间选择 → `final_review` 对标 → 概率校准。把模型从"能排序"升到"能定价/可上线/可过审"。**先补一条数据切分泄漏专项审计**。
- **第三批（护栏与可靠性）**：kernel 隔离护栏 → 内存 RSS 软监控 → 审计原子化 → 脱敏 choke point → DAG 断点续跑 → 失败续跑 UX。
- **第四批（智能增强）**：机器红旗注入 → 记忆接进 driver → reviewer 软闸恢复 → planner few-shot → 增量轮询 → JOIN 向量化。⚠️ 先估算 prompt 膨胀对 32–72B 的延迟/上下文压力。
- **第五批（架构 + 视觉系统化，大投入长线）**：抽 api.py 服务层 → 拆 db.py → 收口双编排 → 证据面板状态语言 + 吉祥物情境动效 + skeleton → 重建设计 token（先出对比稿）→ 统一可观测面板。

---

*生成方式：7 镜头并行深读 + 综合排序 + 对抗式 critic（抽查 8 处关键证据均属实）。总 ~73 万 token、191 次工具调用。critic 对 2 处证据措辞做了校正（`--radius` 是"语义档塌缩"非"全塌"、skeleton grep 命中的是无关变量），结论不变。*
