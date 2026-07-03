# MARVIS V2 全面 Code Review

- **日期**：2026-06-21
- **分支**：`codex/v2-plugin-tool-runtime` @ `8a3b313`
- **范围**：对比 `main`，+170 commits / 281 files / +43,243 行
- **方法**：13 个子系统逐函数对照 function-level spec + 2 个横切扫描（全局不变量 INV-1~10 / 模块边界 INV-10）；每条 critical/high 经独立对抗式复核（仅 `real=True` 保留），复核会主动否决/降级误报。
- **基线**：全量测试通过（~1251+）、ruff clean。"测试通过"不作为正确性证据——本审计专门找测试覆盖不到的逻辑缺陷、不变量被违反、伪绿（测试通过但不断言关键不变量）。

> 不变量速查（蓝图 `docs/superpowers/specs/2026-06-13-marvis-platform-blueprint.md`）：
> INV-1 确定性指标只由平台 tool 算，LLM 不算 · INV-2 业务数字可追溯 · INV-3 join 不静默、须诊断+人工确认、`joined_rows<=anchor_rows`、膨胀/低命中须告警 · INV-4 记忆不改确定性结果 · INV-5 禁存敏感明细 · INV-6 子进程隔离，超时/崩溃/OOM 不拖垮主服务 · INV-7 草稿转正前 Planner 不可选 · INV-8 工具/记忆/计划决策/join 确认须留审计 · INV-9 跨平台路径与子进程编码 · INV-10 模块边界。

---

## 1. 总体结论

开发计划**结构上基本完整**（各子系统 85–92%），但**"全部完成"不成立**：

- **2 个 Critical**：产出错误业务数字，且已上线到报告/策略面，被伪绿测试长期掩盖。
- **约 14 个已验证 High**：崩溃、错误数字、审计/隔离/敏感数据缺口、状态损坏、前端竞态。
- **若干关键能力未落地**：主力评分卡无法交接验证、vintage 真累计口径根本没实现、scan 误判失败。
- 本轮提交**确实修掉了不少早前问题**（见 §2），但也**引入了新 bug**（modeling-3、strategy-1、memory-4、data 异类键等）。

复核同时**否决了 2 个被提报为 Critical 的误报**（memory task_id、memory 置信度降级）和数个降级项，避免误修（见 §4）。

| 子系统 | 完成度 | Critical | High（已验证） | 早前问题 fixed / open+partial |
|---|---|---|---|---|
| Phase 1 Plugins | 90% | 0 | 1 | 1 / 4 |
| Phase 2 Orchestration | 90% | 0 | 1 | 1 / 2 |
| Phase 2B Adaptive Loop | 85% | 0 | 0（均降级 med/low） | 0 / 6 |
| Phase 2C v1_compat | 90% | 0 | 1 | 0 / 1 |
| Phase 3 Data + data_ops | 88% | 0 | 3 | 0 / 4 |
| Phase 4 Feature | 90% | 1 | 1 | 0 / 6 |
| Phase 4V Vintage | 75% | 1 | 1 | 0 / 3 |
| Phase 5 Memory | 92% | 0（2 误报） | 2 | 0 / 5 |
| Phase 6 Modeling | 85% | 0（1 降 high） | 3 | 3 / 3 |
| Phase 7 Strategy | 92% | 0 | 1 | 0 / 2 |
| Phase 8 Drafts | 90% | 0（1 降 high） | 1 | 1 / 2 |
| Frontend V2 | ~92% | 0 | 1 | 1 / 4 |
| 横切（不变量/边界） | — | 1（join 审计） | 2 | — |

---

## 2. 本轮已确认修复（credit）

| 早前问题 | 现状 | 证据 |
|---|---|---|
| 模型报告用原始特征列冒充模型分 | **已修** | `_ModelArtifactScorer` 真 `load_model`+`predict_proba`；`_report_scored_dataset` 产出真实 score 列（`packs/modeling/tools.py`） |
| 特征重要性硬编码 0.0 | **已修** | 改用 `artifact.feature_importance` 真 gain（`tools.py:492-502`） |
| stochastic 工具不强制 seed | **已修** | `manifest.py:175,243` `_validate_stochastic_seed` 在 parse 期强制 `properties.seed:integer`（探针验证） |
| `create_plan` 计划决策无审计 | **已修** | 横切复核确认 |
| 联网学习闭环未接通 | **已修** | web_search→fetch_url→distill→save_learning_note 已接线 |
| `api.py` 直接执行 overfitting 算法（INV-10） | **已修** | 迁至 `validation/overfitting.py` |
| memory 敏感内容 scrubber + allowlist | **已修** | 横切复核确认 |
| 前端 join 死代码 / 新面板转义 | **部分修** | join 已加 try/catch；`draft_manager.js`/`memory_manager.js` escapeHtml 到位 |
| income 回归场景缺 lgb_regressor | **部分修** | recipe 已加；overfit 判定 / 报告连续 target 适配仍缺 |

---

## 3. 必须修的 BUG（已验证 `real=True`，去重）

### 3.1 Critical — 产出错误业务数字

#### C1 · feature 包在 NaN 守卫前把目标列强转 int，污染 IV/KS/AUC
- **位置**：`marvis/packs/feature/tools.py:54,73,131,271,273`
- **不变量**：INV-1 / INV-2 · 复核 `real=True / high conf / critical`
- **机制**：`frame[target_col].to_numpy(dtype=int)` 在核心 `iv.py:30` 的 `np.isfinite(tgt)` 守卫**之前**执行。pandas/numpy 对 `float NaN → int` 静默给 `0`（=good，仅 RuntimeWarning，不抛错）。核心的 NaN 过滤拿到的已是无 NaN 的 int 数组，**永不触发**。
- **影响**：探针 8 行含 2 个 NaN 标签 → int 路径 IV/KS 与正确浮点路径不一致，类别分布被污染（3/3 → 5/3，2 个未标注样本被当成 good）。**任何含未标注/待定账户的真实信贷数据都会产出错误且不可追溯的 IV/KS/AUC**。该反模式在 modeling/validation 也有出现，属系统性。
- **修法**：以 float 读取 → `mask = np.isfinite` → 丢弃 NaN 标签行后再 cast；或在入口拒绝 NaN 标签。

#### C2 · vintage `cum_bad_rate` 是跨 MOB 池化平均，不是真累计坏率，会翻转 cohort 风险排序
- **位置**：`marvis/validation/vintage.py:90-92`
- **不变量**：INV-1 / INV-2 · 复核 `real=True / high conf / critical`
- **机制**：实现成 `cum = Σ(per-MOB bad) / Σ(per-MOB sample)`（跨 MOB 求和的池化均值），再 `max(prev, ratio)` 强行单调。但 `compute_vintage_report`（`packs/modeling/report_compute.py:118-137`）喂进来的是**同一批贷款每个 MOB 一行的重复快照**（cohort 大小恒为 N），正确口径应是 `bad_by_MOB_k / cohort_size`。
- **影响**：探针 100 笔 cohort，MOB6 正确累计 0.12，实现给 0.085；**更糟会翻转排序**——A 真 0.42 / B 真 0.50（B 更危险），实现报 A=0.41 / B=0.255（把 A 报成更危险）。错误曲线渲染进 Phase 6 报告 Vintage sheet（`output/model_report.py:156-165`），并驱动 Phase 7 策略 trend（探针确认从 improving 翻成 deteriorating）。balance 分母同样错（真 1.0、实现 0.25）。
- **伪绿**：`tests/validation/test_vintage.py:69-71` 只断言 `rates==sorted(rates)`（被 `max()` 钳成恒真）；`tests/test_modeling_report.py:125` 甚至把错误值 `0.75`（真 1.0）钉为期望，**主动保护 bug**。
- **注**：4V spec line 98 字面就写"累计坏/累计分母"，所以核心函数符合自身 spec；真正缺陷是 Phase 6/7 集成边界处契约口径错误。`bad_rate`（边际）才是该快照形态下的正确累计值——强证据表明接错了指标。
- **修法**：对重复快照形态按 `bad_by_MOB_k / cohort_size` 计算；去掉 `max()` 钳制，让单调性作为正确数学的自然结果；补"钉死正确累计值"的测试。

### 3.2 High — 确定性数字 / join 安全

#### modeling-3（新）· 报告"OOT 十分箱评估"对全量 train+test+oot 计算而非仅 OOT
- **位置**：`marvis/packs/modeling/tools.py:242-248,695-706`；`report_compute.py:149-203` · 复核 `real=True / high`
- `_report_scored_dataset`（`tools.py:599-605`）`read_frame` 不过滤 split、给所有行打分写 `model_report_scored.parquet`，再传给 `_report_bin_table`，bin 表/edges 全程无 split 过滤。同模块 `_stress_product_removal:563-564`、`_dataset_split_profile`、单变量页都正确按 `split==oot` 过滤，**唯独 bin 表漏了**。OOT 十分箱的样本数/逾期率/lift/金额 lift 混入 train+test 行，OOT 区分能力被系统性稀释（INV-2）。
- **修法**：算 edges 与 bin table 前按 `config.split_values['oot']` 过滤。

#### feature-2 · `equal_frequency_edges` 把不平衡二值特征塌成单 bin，IV=0
- **位置**：`marvis/feature/binning.py:18` · 复核 `real=True / high`（复核纠正了原报告的 worked example，但确认 bug 真实）
- 极端 prevalence（探针确认 p=0.05 / 0.95）时无分位点落在两个值之间，`edges` 被 `-inf/+inf` 覆盖成 `[-inf, inf]` 单 bin → `total_iv=0`、`lift_top_bin=1.0`（无 lift），而 KS=0.29/AUC=0.65 仍有信号——内部自相矛盾，强预测力 flag 会被 IV 选择误删。无 min-bin 守卫；测试只用多值输入（伪绿）。
- **修法**：唯一边缘恰为 2 个时保留中间切点（如 `[-inf, mid, inf]`）。

#### data-3 · 复合键诊断对所有键列套用同一个 match_method
- **位置**：`marvis/data/join_engine.py:120-132`；`backend.py:198-254` · 复核 `real=True / high`
- 复合键（`len(key_pairs)>1`）只取 `key_pairs[0].match_method` 并应用到**每个**键列。spec 核心场景"手机号(md5)+日期"会把 md5 也套到日期列：anchor `yyyymmdd` 与 feature `yyyy-mm-dd` 哈希不等 → 诊断报 `match_rate=0`、误报 `shrink_detected`。**执行本身用 per-pair 方法是对的**、行数断言仍成立，但在人工确认门给出假阴性诊断（侵蚀 INV-3，用户可能因此放弃好 join 或对告警脱敏）。该 mixed-method 分支完全无测试。

#### boundaries-F1（新）· 数值键 vs 字符串键：诊断报高命中、实际 SQL join 全 null
- **位置**：`marvis/data/backend.py:389 (_value_text)` vs `:399 (_sql_transform)`；`join_engine.py:117,132` · 横切发现（未走 verify 管线，但证据充分、与 data-3 同族）
- 诊断侧 `_value_text` 把整数值规范化（`5.0 → '5'`），执行侧 DuckDB `trim(CAST(col AS VARCHAR))` 把 DOUBLE `5.0` 渲染为 `'5.0'`。Excel 摄取 + schema-infer 后一边数值一边文本极常见 → 诊断报高 `match_rate`/低 null 率，用户确认，**真实 left join 匹配≈0、特征列全 null**；`joined_rows<=anchor_rows` 仍成立故**不触发 FanOutError，静默落库坏数据**（INV-3）。测试均用同类型键，未覆盖。

### 3.3 High — 审计缺口（INV-8）

#### data-1 / 横切-F1 · join 确认/执行/落库零审计，`join.confirmed` hook 无订阅者
- **位置**：`join_engine.py:165,182`；`db.py:1614,1638`；`api.py:885,967,982` · 复核 `real=True / high`
- `confirm_join_spec`/`execute_join_plan` 与 `update_join_spec`/`set_join_plan_executed` 全部不写审计；`HookDispatcher.dispatch` 仅对**有订阅的 plugin** 写审计，默认安装无 plugin 订阅，且根本没有 `join.executed` 事件。前端直接驱动这些 HTTP 端点 → **每次 join 确认/执行零留痕**。蓝图 line 660 明确"join 确认全留痕"——这恰是"join 错配静默产生错误样本"的头号风险。

#### drafts-2（新）· 联网学习治理端点 `/learning-notes`、`/author` 无审计
- **位置**：`routers/drafts.py:86-141`；`drafts/tools.py:43-78`；`learning.py:18` · 复核 `real=True / high`
- 两端点 in-process 调 `tool_distill_learning`/`tool_draft_script`（绕过 ToolRunner），router/tool/learning/authoring/`db.save_learning_note` 全无 `write_audit`。把外部网页内容蒸馏入库却零审计；同名工具走 ToolRunner 时有审计、HTTP 治理面没有；兄弟端点 promote/reject 都有审计——属不一致缺口。

### 3.4 High — 隔离（INV-6）

#### plugins-1 · 超时只杀直接 worker，tool 派生的孙进程泄漏
- **位置**：`marvis/plugins/runner.py:93-101,198-206` · 复核 `real=True / high`
- `subprocess.run(timeout=...)` 无 `start_new_session=True`/`killpg`；`TimeoutExpired` 时只 SIGKILL worker 本身，tool 派生的子进程（duckdb/spark helper、multiprocessing、shell-out）被孤儿化继续吃 CPU/内存。探针：worker 被杀后孙进程跑完（marker=29）。测试只断言直接 worker 超时（伪绿）。
- **修法**：`Popen(start_new_session=True)` + 超时 `os.killpg(os.getpgid(pid), SIGKILL)`（POSIX），Windows job-object/taskkill 兜底。
- **相关（low/residual）**：`subprocess_worker.py:103-112` `RLIMIT_AS` 在 macOS 是静默 no-op（OOM 防护只剩 timeout）；native lib 写 OS fd1 可污染单行 JSON 协议（worker 只重定向 Python `sys.stdout`），探针确认成功的 tool 被误报 `error_kind='protocol'`。

### 3.5 High — 敏感数据（INV-5）

#### data-2 · 姓名/银行卡原值落库且被 preview API 原样返回
- **位置**：`marvis/data/schema_infer.py:142-149`；`db.py:2398`；`api.py:316,342` · 复核 `real=True / high`
- `_desensitize` 只对 role∈{phone,idcard,id} 脱敏。中文姓名→categorical、19 位银行卡→numeric（非 18 字符漏 IDCARD、非 1+10 位漏 PHONE）→全部绕过。探针确认 `sample_values=('张三丰',...)` 与完整卡号原值存入 `datasets.columns_json`，且 `/api/datasets/{id}/preview` 对非 phone/idcard/id 列**原样返回整表**。违反 INV-5"禁存客户明细"。
- **修法**：对所有非数值/非日期/非明确公开列统一最小化/哈希 `sample_values`，或只存分布摘要；preview 同样脱敏。

### 3.6 High — 崩溃 / 状态损坏 / 流程 / 体验

#### modeling-1 · `loan_pre_a` 把 `max_depth` 注入评分卡(LogisticRegression)，端到端 TypeError
- **位置**：`scenarios.py:30` + `recipes/scorecard.py:99-104` · 复核 `real=True / high`（原报 critical，因"响亮失败而非静默错数"降为 high）
- `param_overrides={'max_depth':3}` 经 `apply_scenario` 并入 `config.params`，`_lr_params` 仅排除 `scorecard_max_bins`，`max_depth=3` 直达 `LogisticRegression(**params)` → `TypeError`。spec 头号"贷前 A 卡评分卡"场景无法训练。测试伪绿（只断言元数据/裸 recipe）。
- **修法**：`_lr_params` 白名单 LR 合法超参，或评分卡 recipe 忽略/转换树参数。

#### modeling-2 · handoff 评分 notebook 模型文件名双重编码 → 运行期 FileNotFoundError
- **位置**：`handoff.py:208,235` · 复核 `real=True / high`
- `model_json=json.dumps('model.joblib')` → `'"model.joblib"'`，再 `{model_json!r}` 套一层 → 生成 `joblib.load(Path('"model.joblib"'))`（文件名带字面引号）。V1 真执行 scoring notebook（`nbclient` 经 `run_notebook` tool）时 `FileNotFoundError`，模型加载失败 → 分数一致性比较崩。测试只做静态 AST 扫描（`precheck_notebook_contract`），伪绿。
- **修法**：`joblib.load(Path(__file__).parent / {model_filename!r})` 或传裸字符串。

#### memory-4（新）· 回滚非 head 节点产生两个 active 蒸馏
- **位置**：`evolution.py:20-25`；`store.py:303-317` · 复核 `real=True / high`
- `rollback` 无条件 `clear_superseded(predecessor)` 并把目标置 rolled_back，不校验目标是否当前 head。探针：同 scope `A<-B<-C`（C 现行），对中间已被取代的 B 回滚 → A 恢复 active 而 C 仍 active，**同 scope 出现 2 个 `superseded_by IS NULL`**。`get_active_distillation` 用 `LIMIT 1` 静默取一，留孤儿污染检索；可经 `api.py:1344` 任意 id 触发。
- **修法**：仅允许回滚当前 head，或回滚时清理整链。

#### v1compat-1（新）· scan 误判失败，把 V1 能正常扫描的任务置 FAILED
- **位置**：`v1_compat/adapters.py:72-95` · 复核 `real=True / high`
- `material_checks` 纯按 FileRole 分组，任一 role 命中 >1 文件即 `ambiguous`→tool `failed`→DB `FAILED`，**从不读** `task.notebook_path/sample_path/pmml_path/dictionary_path`。V1 `_resolve_scan_material` 会尊重显式字段并 SCAN 成功。探针：两个 .pmml + 显式 `pmml_path='a.pmml'` → v1_compat 判 FAILED，V1 判 success。`model_validation` 模板 scan 步 post_check 要求 `status in ['scanned']`，故误中止合法验证任务。V2 自己的 pipeline（`_required_path`）反而能跑该任务。

#### frontend-1 · plan 轮询取消竞态：在途请求把 cancelled 覆盖回 running 并复活轮询
- **位置**：`marvis/static/js/v2/plan_view.js:241-261` · 复核 `real=True / high`
- `tick()` 在 `await fetchPlan()` 后不复检 `pollState.stopped` 就 `setPlan(plan)` 并重排 timer。取消发生在 fetch 在途时：`stopPlanPolling` 设 `stopped=true`/清旧 timer，但无法撤销在途请求；其 resolve 后用过期 `running` 覆盖已 `cancelled` 状态、并排新 timer 复活轮询。Node 探针复现。`skill_manager.js` 用 `validationVersion` 守卫了同类竞态，说明是疏漏。
- **修法**：`await` 后 `if (pollState.stopped) return null;` 再 setPlan/重排。

### 3.7 High — INV-1 强制网有洞（防御纵深缺失，当前内置工具数值自洽未算错）

#### orch-1 / loop-4 / 横切-F2 · validator 的 range 闸门只认顶层属性 + 6 字段白名单
- **位置**：`marvis/orchestrator/validator.py:15,141-157,236` · 复核 `real=True`（severity 由 critical 调为 high）
- `_metric_fields_in = _schema_fields(顶层 properties) ∩ {ks,auc,psi,iv,lift,gini}`，是唯一的 plan 期 INV-1 闸门。逃逸项：`train_model` 的 KS/AUC 嵌 `metrics:{}`；`compute_feature_metrics` 嵌 `metrics:[]`；`backtest_strategy` 的 `approved_bad_rate/expected_profit` 等顶层但不在白名单；`bin_feature.total_iv`（名字非 `iv`）。运行期 `reviewer._dig` 只走 dict 路径、无法索引 list，故连显式 `metrics.0.ks` range 也失效（探针 KS=1.7 通过）。
- **现状**：核心 INV-1（数值由 tool 算、不由 LLM 算）成立，当前内置工具内部有界故"没算错"；但任何越界/回归/promoted draft 的越界值会静默过审——这是防御纵深的洞。
- **修法**：对 array/nested 指标按 JSON Pointer 递归识别；或对 `determinism=deterministic` 且输出含已知指标名的 tool 强制 range/invariant。

### 3.8 降级但仍需关注（medium）

- **loop-2 · NO_PROGRESS 检测功能性失效**（`db.py:1273` / `executor.py:414`）：`replace_remaining_steps` 删除 FAILED 步，`recent_failed_tool_refs` 永远累计不到阈值 2 → 同一失败工具被重试满 `max_replan_iterations`。复核降 medium（仍被 replan 预算兜底，非无界），但专门的"无进展"防御纵深为零，且测试预置 FAILED 行（伪绿）。
- **memory-3 · 蒸馏 `distill.skip` / 敏感丢弃无审计**（`distillation.py:118-119,168-169`）：spec Part B 伪码明确要 `write_audit(distill.skip)`；当前 `except: continue` 与 `return None` 静默。复核降 medium（不碰确定性指标）。
- **横切-F3 · `pipeline.py:1341` 子进程 `text=True` 缺 `encoding`（INV-9，Windows mojibake 风险）。**
- **横切-F2(边界) · 金额分箱报告 NaN/边缘分数 population 不一致**（`report_compute.py:169-202`，`np.digitize` vs `assign_bins`）：业务指标与 bin 自身计数对不上。

---

## 4. 误报澄清（不要去"修"）

| 提报 | 复核裁决 | 理由 |
|---|---|---|
| memory-1：蒸馏 `source_task_ids` 注入 prompt 违反 INV-5 | **real=False / low** | INV-5 敏感集**不含** task_id；`source_task_id` 是设计要展示的溯源字段（raw 包早就带它），`DISTILL_SYS`"不输出任务ID"只约束 LLM 摘要文本（有测试）。 |
| memory-2：再蒸馏 support 缩小→置信度降级违反 INV-4 | **real=False / low** | INV-4 只管确定性指标；蒸馏置信度是检索优先级建议，spec 明确允许双向变化、`_is_meaningful_update` 即如此设计；证据被撤回时降置信度反而正确。 |
| orch-2：backtest 决策点可被 LLM replan | **real=False / low** | replan 只动后续步，已完成步的 tool 计算值不可改/不可删，INV-1 未违反；仅 `_is_safety_step` 两处定义有 cosmetic 分叉。 |
| drafts-1：子进程无 fs/网络牢笼违反 INV-6 | **降为 high** | 蓝图 line 437 **明确选择不做** seccomp/容器（性价比），INV-6 只保证超时/崩溃/OOM 不拖垮主服务（worker 做到了）。真正该修：静态扫描可被 `getattr`/字符串拼接绕过、且 `promote_draft` 不重扫（`authoring.py:34-52`、`promotion.py:12-78`）。 |
| loop-1：autonomy_level 未生效削弱护栏 | **降为 low** | 执行器确实不按档位调确认强度，但**与 function-level spec 伪码一致**；join/不可逆步的确认由 validator 独立强制（档位不放宽）。属文档/语义不一致。 |

---

## 5. 完成度缺口（"全部开发完成"的反例）

- **Modeling — 主力评分卡无法交接验证**：`export_pmml`/`handoff_to_validation` 只支持 `lr`，scorecard/lgb/xgb 走不通"训练→验证"闭环；评分卡标准分（base_score/PDO/odds）未产出（artifact 存 factor/offset 但训练/报告只用 `predict_proba` 概率）；income 回归无 overfit 判定（`overfit_flag` 硬编码 False）、report 对连续 target 未适配。
- **Vintage — 真累计口径根本没实现**（即 C2）；缺失 MOB 行静默丢弃、`warnings` 恒 `[]`（`vintage.py:177`）。
- **v1_compat** — scan 期不调用 `precheck_notebook_contract`（V1 会在 scan 失败契约破损 notebook）；一次性子进程无法复用 V1 live kernel，`_ensure_metrics_session` 重跑 notebook（正确但重复计算）。
- **Strategy** — 无 go-live/activate 工具（V4 手动模式只是空满足）；业务数字无任何 range/sanity post_check；`vintage_profit` 按 raw cohort 值分组而 `vintage_curve` 归一化为 YYYY-MM，两视图无法对齐。
- **前端** — `index.html` 仍以 `app.js`(6329 行) 为 type=module 入口（V2 经 `mountV2` 挂载达成，但 V1 主面未拆，Phase 0 地基债未清）；`dedup=abort` 前端可选但后端 `DEDUP_STRATEGIES` 无此项，永远 422。
- **Drafts** — 默认 `MARVIS_SEARCH_ENDPOINT`（DuckDuckGo Instant-Answer）响应形态 `RelatedTopics` 未被 `_parse_search_results` 解析，实时默认搜索静默返回 `[]`；`network_available()` 探 `example.com` 与实际 endpoint 解耦，内网代理部署会误判 offline。

---

## 6. 系统性问题

1. **伪绿测试反复掩盖真 bug**：feature-1、vintage-1/2、modeling-1/2/3、data-3、frontend-1、loop-2 全部"测试通过但从不断言关键不变量"，甚至 `test_modeling_report.py:125` 把错误值钉为期望。建议补：喂 NaN 目标列、钉死正确累计值、复合/异类键 join、取消竞态、no-progress 跨轮、nested 指标越界应被拦截等用例。
2. **模块边界倒置环仍在（INV-10，still-present）**：`db.py:34,40` 顶层 import `packs.*.contracts`（把 modeling/strategy 业务代码 eager 加载进 DB 层），而 packs 反向 import `db.py` Repository；`output/model_report.py:10` import `packs.modeling.report_compute` 而 `packs.modeling.tools:19` 反向 import output——仅靠 import 顺序未成硬循环。且 `tests/` 无任何 import-graph/边界测试守护（建议加 ast 静态扫描，把 INV-10 变成可执行不变量）。
3. **INV-1 强制网覆盖面窄**（见 §3.7）：嵌套/业务数字逃逸，reviewer 无法索引 list。

---

## 7. 建议修复优先级

1. **2 个 Critical**：feature-1（int 转换污染指标）、vintage-1（累计口径错+排序翻转）——已上线到报告/策略，最高优先。
2. **join 安全簇**：data-3 复合键诊断、boundaries-F1 异类键全 null、data-1 join 审计、execute_join 告警可见性——join 是"真金白银"风险点。
3. **崩溃/错误数字 High**：modeling-1（评分卡崩）、modeling-2（handoff 加载崩）、modeling-3（OOT 全量）、feature-2（二值 IV=0）、strategy-1（vintage MOB off-by-one，见下注）、memory-4（双 active head）、v1compat-1（scan 误判）。
4. **隔离/敏感/审计**：plugins-1（孙进程泄漏）、data-2（明细落库）、drafts-2（学习审计）、INV-1 白名单。
5. **系统性**：补伪绿测试 + 架构 import 测试。

> 注：strategy-1（`packs/strategy/vintage.py:43-57`，`vintage_summary`/`mob_max` 按位置索引 `curves[c][ref_mob-1]`，当 MOB 从 0 起或非连续时取错 MOB、可翻转 trend）为已验证 High，归在确定性数字簇。

---

## 附：审计方法与可信度

- 13 子系统 + 2 横切共 39 个 agent，逐文件读当前 HEAD 真实代码，关键怀疑用一次性探针（`/opt/miniconda3/envs/py_313/bin/python`）复现，**未修改任何仓库文件**。
- 每条 critical/high 经独立"对抗式 skeptic"复核，默认怀疑——本轮据此**否决 3 条、降级 5 条**，避免误修。
- 仍有 31 medium / 26 low 未在本文逐条展开（多为健壮性/边界/质量），如需可另出清单。
