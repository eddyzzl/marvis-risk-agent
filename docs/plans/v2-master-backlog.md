# V2 Master Backlog（唯一主待办清单）

> **权威声明**：本文档是 V2 全部待办的**唯一权威来源**（single source of truth）。审查报告、策略计划、旧追踪器等文档保留为背景与证据，**不再各自维护待办状态**；进度只在本文档更新。
> **生成**：2026-07-02。合并来源：① 2026-07-02 全方位审查（115 条，[报告](../reviews/2026-07-02-v2-comprehensive-improvement-review.md)）；② 建模方法学极致专项审计（37 条，四镜头：预处理/筛选/调参/选择评估，本文 §3）；③ [策略与风险分析计划](v2-strategy-risk-analysis-plan.md)（S1a–S6，6 项已拍板）；④ 旧执行追踪器 [v2-comprehensive-improvement-plan.md](v2-comprehensive-improvement-plan.md) 提取的 24 条余项；⑤ modeling-agent-roadmap / 6-28 报告残留（经代码核实）；⑥ PR 前流程清单。
> **图例**：状态 ⬜ 待做 · 🔄 进行中 · ✅ 完成 · ⏸️ 显式搁置（记录为产品选择）。验证列：✅ 对抗验证确认 · ⚠️ 部分确认（细节见源报告） · — 未独立验证。
> **快速统计**：待办共 **179 项** = 审查 115 + 极致审计 37 + 策略批次 S1a–S6（6 个批次，含被吸收的 7 条审查项）+ NEW 2 + 长线区 19（与前述部分重叠已去重）。

---

## 0. 完全体 v1 验收定义（Definition of Done）

前 7 条继承自旧追踪器（原文见其 L1022-1032），8–10 为本轮新增：

1. 所有核心任务类型都经由 typed gate 与 evidence envelope 运行。
2. AUTO 能执行有界结构化动作，并在声明权限之外安全停车。
3. 建模覆盖 G2–G5：用户或 AUTO 显式决策、选定实验、PMML/PKL/报告/交接闭环。
4. 运行时副作用可经 step-run ledger 与 artifact staging 恢复；剩余非原子边界显式写入文档。
5. API/DB/前端控制器拆分到"新增领域不再撑大单体文件"的程度。
6. CI、聚焦建模 API smoke、真实业务材料 smoke 从**干净的已提交树**全绿。
7. 最终 review 把剩余限制记录为产品选择，而非未完成的核心架构。
8. **建模方法学四镜头（预处理/筛选/调参/选择评估）审计缺口清零**——兑现"同样本同标签下，特征预处理与筛选、模型选择、调参到位则 KS 封顶"的用户核心判据（§3）。
9. **策略开发与风险分析 S1a–S6 全部交付**，六入口全功能可用（§4/§6/§8）。
10. 本清单全部条目 ✅ 或 ⏸️（⏸️ 必须写明"为什么不做"并进最终 review 的产品选择清单）；长线区（§12）不阻塞完全体 v1。
11. **收官双 review 循环（2026-07-03 用户新增）**：全部条目完成后 → (a) **落地核验 review**：逐项对照本清单验证每个 ✅ 是否真实按计划落地（抽真代码证据，不信 commit message）；(b) **全量 code review**：对最终代码做新一轮多镜头审查（bug+可提升点）；(c) 发现项修复后**再循环**，直至一轮 review 无新 critical/high 发现为止。

---

## 1. 阶段〇：PR 前收口（0.5–1 天）

| 状态 | ID | 事项 | 来源 |
|---|---|---|---|
| ✅ | PR-1 | 未跟踪文件已收口（2026-07-02）：task-search.js 已随早前提交入库；glow 脚本/assets 已被用户从磁盘移除（VD-5 实施时需重新生成素材）；审查报告已入 `30877a4c`；master backlog + 策略计划 + 指针改动已入 `5f6bb17c`；`git status` 干净 | 追踪器 L499-502 |
| ✅ | PR-2 | 已提交树 `5f6bb17c` 全量门禁通过（2026-07-02）：diff/ruff/node 全过，pytest **1988 passed, 4 skipped**（8m37s，py_313） | 追踪器 L497/1015 |
| ✅ | PR-3 | 六旅程 smoke 全 PASS（2026-07-02，真服务+合成数据+manual 模式）：JOIN 2000==2000、特征报告落盘、G2–G5 走通选定实验、PMML4.4+PKL 落盘、交接任务+5 材料、强制失败经 retry 端点真实恢复。两个产品发现：time_col 不触发时间切分（佐证 SEL-1）；retry inputs 整体替换语义（备注入 LT-4） | 追踪器 Phase H |
| ✅ | PR-4 | PR 描述完稿：docs/releases/2026-07-02-intermediate-pr.md（含全量门禁 1988 passed 与六旅程 smoke 结果、剩余风险清单、v1 拍板记录） | 用户清单 |
| ✅ | PR-5 | Open Decisions 4 项已拍板记录（见下方"PR-5 拍板记录"） | 追踪器 L1002-1007 |
| ✅ | PR-6 | 决议：选"随 AGT-1 一并修"（阶段一第 1 项将同时收紧手动文本确认面）；PR 描述中列为已知限制待修 | 追踪器 L851 |

**PR-5 拍板记录（2026-07-02，依据既有代码行为与用户既往决策）**：
1. **AUTO 自治级别**：v1 正式定为"仅有界低风险调整"（现有实现即此行为）；更高自治待阶段六 AGT-7/AGT-9（门预算+建模门红旗）与 AGT-4（成功标准）落地后再评估。
2. **沙箱机制**：subprocess + env allowlist + 路径后置校验 + RSS 软监控为 v1 终态；OS 级 containerization 归长线 LT-9 评估，不阻塞完全体 v1。
3. **PMML 承诺口径**：`.pkl` 为源、`.pmml` 为兼容交付件（代码现状一致：native Booster 明确拒绝 PMML；校准层不入 PMML 已在模型卡标注）。
4. **视觉重设计深度**：走 token 收口（VD-11）而非全面重设计；radius/间距等口味项按用户既往约束**先出对比稿拍板再实施**。

## 2. 阶段一：审查 Batch 1 —— 复发债与小正确性（约 3–5 天）

> 三条"上轮号称已修未修净"+ 一条三轮审查漏网的旧 critical。出门标准：**兄弟路径全枚举回归**（train/tune/select、propose/execute、manual/agent）。

| 状态 | ID | 事项 | 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | AGT-1 | is_confirm 已加疑问句守卫 + 整串锚定（`bfd075b2`，4 个误判串回归覆盖；PR-6 的手动文本确认面同步收紧） | High/S | ✅ |
| ✅ | DOM-1 | tune 已接 NaN 门（`60e48bb9`）：resolve_modeling_splits + typed error + nan_labels_dropped 上报 + 全无标签 OOT 转 scoring-only | High/S | ✅ |
| ✅ | PERF-2 | 唯一性/去重已改在变换键空间计算（`57d25038`，DuckDB+pandas 双路径，exact_lower/date/hash 三场景回归；复现脚本验证修复） | High/M | ✅ |
| ✅ | NEW-1 | vintage 内核已真累计（`95863f0b`）：cohort 固定基数（max sample_count/balance_sum）+ 单调不减 + 超 1 clip 并出 data_quality_warnings；report_compute 快照口径已识别防双重累计；S3/B1/B5 前置解除 | High/S | ✅ |
| ✅ | ARCH-3 | 全部软探测已收口（`2785fa0a`，11 文件 -184 行）：直调审计方法+删 fallback；仅剩 JoinEngine 构造器 3 处有意硬失败守卫；测试替身补齐 *_with_audit | High/S | ⚠️ |
| ✅ | UX-3 | 四个门控件 context 工厂已捕获发起时 taskId，写回前校验未切换（`36097f54`，含行为级回归） | High/S | ✅ |
| ✅ | UX-7 | C1 双主表前后端双重校验：前端即时拦截+后端 typed JoinSetupError 点名多余表（`12801be5`） | Med/S | — |
| ✅ | DOM-9 | 冠军改按防过拟合 test KS（`c92fdf00`，test_ks−0.5·max(0,train−test)），OOT 只报告；selection_policy 显式配置优先（TUNE-5/SEL-7 的加权/阈值维度留 §3） | Med/S | — |
| ✅ | DOM-10 | 报告评分列缺失改抛 ReportScoreMissingError（`ce15d0ab`），删除首特征列冒充回退 | Med/S | — |
| ✅ | REL-9 | execute_join_plan 同步分支已加与 async 对称的 job 守卫（`4ed8e44e`，并发二次调用 409） | Low/S | — |
| ✅ | REL-8 | execution_environment.json 已改原子写（write_json_atomic）+ 损坏自愈回退默认（`e71cf3c0`） | Low/S | — |
| ✅ | NEW-3 | smoke 发现的新 bug 当场修复（`70716027`）：join+无切分列建模在 C1 确认后 409（modeling_setup 留空 split_col 但模板 schema 要求非空）——改为 split_config 自动切分（含 group_cols 防泄漏）贯通 make_split，端到端回归覆盖 | High/S | ✅ |

## 3. 阶段二：建模方法学极致专项（EXC）——高影响 8 条约 2–3 周；全部 37 条约 4–6 周

> **这是用户定义的完全体核心判据**：同样本同标签下，预处理+筛选、模型选择、调参到位则 KS 封顶（谁做差距 ≤1-2 点）。四镜头审计结论：**地基扎实（53 项 done_well 经核实），但默认路径离极致还差一个台阶**——高影响缺口每条都在 0.5–2 个 KS 点量级。8 条 High 全部对抗验证 CONFIRMED，零驳回。完整证据、改法与 53 项 done_well 见本文档**附录 A**。
> 排序说明：按用户判据将本批插在最前（复发债之后）；与其他批次可穿插。
> 已核实到位（抽样）：KS/AUC 逐样本精确计算且 tie 处理正确；scorecard 配方 WOE 严格 train-only 且 woe_maps 随 artifact 落盘、打分/PMML/handoff 三面可精确重放；MLP 用 sklearn Pipeline 打包 impute+scale；切分器具备时间 OOT/分组防泄漏能力；单调分箱链完整；泄漏单变量硬门；seed 贯通；OOT 不参与调参选择；NaN 标签门贯穿训练配方。

#### 特征预处理
| 状态 | ID | 事项 | KS 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | PREP-1 | 四工具 train-only 拟合已落地（`bc537dba`）：woe/impute/normalize/cap 默认排除 ("test","oot")、无 split 抛 FitRequiresSplitError（allow_full_fit 逃生口）、fit_rows/fit_split 口径回显 | High/M | ✅ |
| ✅ | PREP-2 | 预处理链已落盘可重放（`30b72ccb`）：`.preprocessing.json` sidecar（impute/cap/normalize/onehot）随派生数据集累计→训练时进 ModelArtifact→scorer/handoff notebook 重放；PMML 不含预处理已在 model card 诚实标注。尾巴已闭环：woe/categorical_woe 入 sidecar 且 scorer/handoff 统一重放（`c2d95fd9`） | High/L | ✅ |
| ✅ | PREP-3 | 类别链路三层落地（`3ee2146d`/`982ae470`/`ca47fef4`）：excluded_categorical 显性化进 screen 门文案；categorical_woe_encode 工具（train-only+Laplace+rare 归并+未见类别先验 fallback）；CatBoost 原生 cat_features（含调参路径）；setup 提示不改默认 | High/L | ✅ |
| ✅ | PREP-4 | detect_sentinel_values 落地并打通 impute/cap/normalize/bin/woe（`9fc22a46`），screen 门带 sentinel 提示 | Med/M | — |
| ✅ | PREP-6 | LR Pipeline（impute median+standardize）落地（`56827495`） | Med/S | — |
| ✅ | PREP-7 | derive_date_features 工具落地（`9f8220d1`）：days_since/month/months_on_book，opt-in | Med/M | — |
| ✅ | PREP-8 | impute_missing 支持 add_indicators（`1dfffdcb`），指示列入重放链 | Med/S | — |
| ✅ | PREP-5 | suspected_categorical 启发式落地并进 screen 注记（`b2d266fd`），不改默认行为 | Med/S | — |
| ✅ | PREP-9 | min_bin_pct=0.05 贯通等频/卡方分箱与 scorecard/select/tune 路径（`750a55f7`） | Low/S | — |
| ✅ | PREP-10 | fit_mask+min_group_size=30+target_col 硬拒绝（`bd26b000`，FS-11 泄漏通道一半随之关闭） | Low/S | — |

#### 特征筛选深度
| 状态 | ID | 事项 | KS 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | FS-1 | 两模板已插入"精选特征"漏斗步（`c00032fe`）：IV≥0.02 → 相关去冗余 0.95（高 IV 胜出）→ 可选 VIF；门内 adjust 可调阈值；screen 加 top_k=200 兜底；含 2 强+3 噪声+1 冗余端到端回归（迭代剪枝/null-importance 归 roadmap-1d 长线备注） | High/M | ✅ |
| ✅ | FS-2 | select_features 默认排除 test+oot、自动识别 SPLIT_COLUMN、typed error 兜底（`bc537dba`）；legacy 模板筛选步已接 split_col | Med/S | — |
| ✅ | FS-3 | （随 PREP-3 落地）excluded_categorical 上报 + woe_encode_categorical 防泄漏编码 | Med/M | — |
| ✅ | FS-4 | split_shift（|Δks|>0.15）+ leakage_watch 软区间双线落地（`b93ea353`），进门文案 | Med/M | — |
| ✅ | FS-5 | 衍生模板筛选步改指衍生数据集+新列并集（`d927fa4c`，validator 支持容器内 $ref） | Med/S | — |
| ✅ | FS-6 | ks_train/ks_test/ks_decay 进 scores + 可选 max_ks_decay 观察阈值（`502a22d9`） | Med/S | — |
| ✅ | FS-7 | coverage 列+缺失即信息候选 note（`21647039`），排序不变 | Low/S | — |
| ✅ | FS-8 | 样本不足返回 None+聚合告警，VIF 门显式跳过（`c4b9a6b5`） | Low/S | — |
| ✅ | FS-9 | iv_binning 口径字段全路径记录+DEFAULT_IV_BINS 统一（`1bc6c9bf`） | Low/S | — |
| ✅ | FS-10 | continuous→|Spearman|、multiclass→OvR macro-AUC 排名（`cfeb328b`） | Low/S | — |
| ✅ | FS-11 | log1p/rank 变换算子（`a35c05f3`）；目标列泄漏通道已随 PREP-10 关闭；真时序 diff 因平台无 panel 概念明确出界 | Low/S | — |

#### 调参与训练方法学
| 状态 | ID | 事项 | KS 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | TUNE-1 | 两阶段搜索已泛化到全配方家族（`5e30ab4e`）：lgb/xgb/catboost 各 40 trial 带早停，lr/scorecard/mlp 各 12 trial 小空间；per-recipe sha256 确定性 seed；lgb 单配方路径字节级向后兼容 | High/L | ✅ |
| ✅ | TUNE-2 | 确定性两阶段搜索已落地（`5909edba`）：60/40 粗搜+邻域细搜、lambda log-uniform、lr 0.01–0.3 与轮数反比联动、默认 40 轮、gate 文案改按规模建议；无新依赖；全量 2012 passed | High/M | ✅ |
| ✅ | TUNE-3 | 早停改用 train 内切 15% 折（`cf924d49`），test 只做选择；可选 grouped cv_folds+稳健惩罚（`8a009dd7`） | Med/M | — |
| ✅ | TUNE-4 | refit_on_train_plus_test 默认开启（`4fe808d9`）：冻结参数+轮数缩放、OOT 前后对比进报告 | Med/S | — |
| ✅ | TUNE-5 | 有权重时 trial/冠军统一按 weighted KS（`aed0f8af`）；顺修 compare/select 从不呈现加权指标的真 bug | Med/S | — |
| ✅ | TUNE-6 | 线程数收敛到 defaults 常量+force_row_wise+trial 记录 deterministic 标志（`1b724747`） | Low/S | — |
| ✅ | TUNE-7 | dict/str/list 约束经共享纯函数归一化，tune 与训练路径一致（`014d1bd3`） | Low/S | — |
| ✅ | TUNE-8 | 单配方失败记为 failed candidate 不连坐（`56827495`） | Low/S | — |

#### 模型选择与评估口径
| 状态 | ID | 事项 | KS 影响/工作量 | 验证 |
|---|---|---|---|---|
| ⬜ | SEL-1 | 默认切分不建 OOT，时间外推 OOT（oot_by_time）是全仓从未被调用的死代码 | High/M | ✅ |
| ✅ | SEL-2 | 公平竞技场已落地（`babd61fe`）：每配方先调参再参赛、树模型统一早停、同切分同特征断言进回归；门文案含"总预算=Σ配方预算"与耗时提示 | High/M | ✅ |
| ✅ | SEL-3 | LR 重建为 impute→scale→LR Pipeline（`56827495`），随 artifact 可重放 | Med/S | — |
| ✅ | SEL-4 | 早停折从 train 内切出（`cf924d49`），test 职责单一化 | Med/M | — |
| ✅ | SEL-5 | bootstrap_ks_ci 落地（`95b49ab4`）：分层重采样 n_boot=200/大样本自动降 100、确定性 seed；对比/选择输出 CI 并在重叠时标"差异在抽样误差内" | Med/M | — |
| ✅ | SEL-6 | ensemble 配方落地（`e0b527e2`）：seed-bagging N=5 全家族、opt-in 不进默认、scorer 重放、PMML 显式拒绝带原因 | Med/L | — |
| ✅ | SEL-7 | 默认护栏（gap>0.10 overfit_warning、<3 特征 sanity_warning，可关）+ 排除候选就地标注原因（`d6f3217a`） | Med/S | — |
| ✅ | SEL-8 | 按预授权降级交付：segment_value_evaluation 诊断工具（整体 vs 分群 KS/AUC+小群归并；`404b4797`），完整多模型路由的设计残章在 commit message | Low/L | — |

## 4. 阶段三：S1a/S1b 策略与打分底座（6–10 天）

> 详细 spec 见[策略与风险分析计划](v2-strategy-risk-analysis-plan.md) §七；**第一份函数级 spec = S1a 方向迁移**。

| 状态 | ID | 事项 | 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | S1a | 按 spec 四 commit 落地（`f4452461`/`d9e0399c`/`1caa5f26`/`a61e54dd`）：direction 原语+自检门、ModelArtifact 双方向字段（9 配方全接线+修 SELECT 漏列）、tradeoff/reject_inference 可选参数、build_strategy 决策感知算子自检；spec §6 五项开放问题落地选择已记录 | — | 163 tests |
| ✅ | DOM-2 | （随 S1a 落地 `1caa5f26`）双工具带方向参数+corr 自检门+渲染层方向标注 | High/M | ✅ |
| ✅ | NEW-2 | （随 S1a 落地 `a61e54dd`）改为委托 feature/metrics 方向自适应实现（validation→feature 依赖已有先例） | Low/S | — |
| ✅ | S1b | 三 commit 落地（`724c0f26`/`0b7742ce`/`aebc6cf6`）：训练期基准快照（分数等频分位+特征分位）随 artifact 落盘、score_dataset（预处理链重放+方向元数据+审计+衍生数据集登记）、monitor_run（PSI/CSI/有标签对比+green/amber/red 判级）+MONITORING_RUN 模板；六项回归含手算基准分箱、逐字节重放、漂移注入→red、无标签→n/a | — /2–4天 | ✅ |
| ✅ | DOM-3 | （随 S1b 落地）打分与监控执行闭环完成，监控阈值不再是纸面 JSON | High/M | ✅ |

## 5. 阶段四：审查 Batch 2 —— 运行时脊柱（1.5–2.5 周）

| 状态 | ID | 事项 | 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | PERF-1 | 四个重端点（上传/propose/confirm/execute）改线程池执行（`d26c1714`），附事件循环响应性回归（revert 即红）；driver messages 链路核实本就是 def | Critical/S | ⚠️ |
| ✅ | REL-1 | driver 回合已 job 化（`232d3b35`）：start_job/finish_job 包裹全回合、并发二次确认 409（中文提示）、异常路径 finally 释放；五任务类型全覆盖+竞态回归 | Critical/M | ✅ |
| ✅ | UX-1 | 五个 submit 路径即时 busy+消息轮询+计划栏 1.5s 刷新（`8babda7f`）；taskServerBusyAction 接 driver job，刷新后忙碌可见；停止能力归 REL-5 的 cancel（docstring 注明） | Critical/M | ✅ |
| ✅ | REL-6 | job 化+running 步骤 started_at 进 plan payload+前端耗时显示（随 REL-1/UX-1 落地） | Med/M | — |
| ✅ | REL-4 | startup 已 reclaim RUNNING plan（`7cf1bb0e`，plan_recovery 模块）：步骤按 ledger 语义收敛、plan 暂停、driver 任务收到中文重启通知 | High/M | ✅ |
| ✅ | REL-2 | 重启 reclaim 识别 metrics 可续场景改发 METRICS_STAGE_FAILURE 前缀消息（`bfba3983`），is_metrics_failure 命中→metrics-only 重试；last_completed_step 接通 | High/S | ✅ |
| ✅ | REL-5 | jobs 心跳列+看门狗（`c9a9be7b`，超时默认可配、/api/health 暴露 stuck_jobs）+ join/plan job cancel 端点（协作式） | Med/M | — |
| ✅ | REL-3 | ProcessTreeResourceMonitor 已泛化接入 ToolRunner（`a278a0ff`）：默认 4096MB 可配、超限杀进程树、error_kind=resource_limit+peak 审计；真杀路径实测验证 | High/M | ✅ |
| ✅ | REL-7 | reconcile 已泛化到 datasets/tasks 根（`6dac2349`）：.bak 恢复、.tmp 孤儿清理、幂等、7 个回归 | Med/M | — |
| ✅ | PERF-5 | worker 入口依赖链已切断（`c1313514`，ToolContext 抽到无依赖 contracts 模块）：入口 import 实测 1.085s→0.021s，sys.modules 无 pandas/sklearn/marvis.db（回归断言守住） | High/S | ✅ |
| ✅ | PERF-3 | 隔离模式复现+metrics cells 合并进单次 notebook 子进程（`0e832a99`）：端到端断言执行次数==1；metrics 重试与非隔离路径不受影响 | High/M | ✅ |
| ✅ | PERF-4 | 诊断请求级 memoization + 多方法单扫描批量化（`90667a7b`）：200k 行 propose 15.9s→10.1s、relaxation 1.02s→0.63s | High/M | ✅ |
| ✅ | PERF-8 | DuckDB 统一配置（`3842eb21`）：memory_limit 4GB/threads cpu·½/temp_directory spill，均可 env 覆盖，状态进 /api/health | Med/S | — |

追踪器补充细则（实施本阶段时参照）：统一 job policy（同步/后台/子进程三型，覆盖 join/validation/train/tune/report）→ PERF-1+REL-6+UX-1 的"怎么做"；interrupted step-runs 在 plan rail 呈现 retry/repair 状态 → REL-4 的 UI 侧；主 app workspace 完整接入 V2 join 组件 + join 的 task/job 状态 UX 标准化到其他长任务。

## 6. 阶段五：S2/S3 策略主线与组合分析（10–16 天）

| 状态 | ID | 事项 | 影响/工作量 |
|---|---|---|---|
| ✅ | S2 | 按 spec 四 commit 落地（Commit2=`24578b3f`/Commit3=`cf7f47de`/Commit4=`1dbf91b1`+版本化持久化 commit）：strategies 版本/状态/谱系列+strategy_artifacts 表（原子采纳 CAS+自动退役）、design_cutoff_bands 五类红旗、tradeoff 可行域、compare 2×2、adopt 强制门+决策表 CSV+监控计划 JSON、STRATEGY_DEVELOPMENT 七步四门模板、strategy_experience 记忆 kind+ARCH-4 发现的 MEM-1 kwargs 缺口修复；133 验证于合并树 | 5–8天 |
| ⬜ | DOM-12 | （并入 S2）运营点约束不可行时静默回退；fuzzy 拒绝推断忽略逐件分数 | Low/S |
| ✅ | S3 | 四 commit 落地（`f5715373`/`6d82bdb2`/`d269d512`/`2d852674`）：表现期契约+马尔可夫合成快照、packs/analysis 新包（flow_rate/bucket_migration/segment_profile/expected_loss_estimate 吸收链手算断言）、PSI/CSI 趋势与 monitor_run 同内核、PORTFOLIO_ANALYSIS 并行模板+states 人工确认门+组合报告 xlsx | 5–8天 |
| ⬜ | DOM-8 | （并入 S3）Roll-rate 缺月盲视、无余额口径、period 硬编码 | Med/M |
| ⬜ | DOM-11 | （并入 S2/S3）swap/bad rate 空集显示 0.0%、缺标签覆盖率口径 | Low/S |
| ✅ | UX-9 | 一键示例数据首跑（`bd274e71`）：sample_data.py + POST /api/sample-data + 欢迎页入口；S3 的合成表现数据脚本仍随 S3 批次补充 | Med/M |

## 7. 阶段六：审查 Batch 3 —— 智能闭环（2–3 周）

| 状态 | ID | 事项 | 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | MEM-1 | 双向接线落地（`8de582ec`/`80ccb858`，memory_bridge 模块）：实验/JOIN 结果写入记忆、门注入【历史同类实验(只读参照)】锚点+use 审计；政策双门控、无历史时字节级等同现状（INV-4）；**四条 critical 至此全清** | Critical/M | ✅ |
| ✅ | AGT-3 | metric-aware 摘要（深度2/20键/600字符）+goal 带 slots 摘要+LLM 意见降权为 goal_doubt→REVIEW（`5ada60a2`） | High/M | ✅ |
| ✅ | AGT-4 | build_plan 支持 success_criteria+任务级 oot_ks_min 可选控件（不写死数值）→未达标走既有 replan 环（`a0f5c19c`） | High/M | ✅ |
| ✅ | MEM-2 | 两处构造点接通 llm_factory（`70eded8d`），失败回退模板句 | High/S | ✅ |
| ✅ | MEM-3 | (type,task,指纹) 幂等去重+support 按独立 task 计（`db996264`） | High/S | ✅ |
| ✅ | MEM-4 | detect_setup 接 field_hints 决胜（`80ccb858`）：按本任务数据文件名限定来源、只影响候选排序与提示文案 | High/M | ✅ |
| ✅ | LLM-1 | json_schema 约束解码+能力探测+json_object 回退（`ed09e819`），gate/router/planner/reviewer/intent 全接 schema，抽取重试兜底保留 | High/M | ✅ |
| ✅ | LLM-2 | run_eval_case 生产实现+`marvis eval-llm` CLI（`aa745e7a`）：真 IntentRouter/Planner/Validator + 注入 LLM，7 个初始 case 可执行 | High/M | ✅ |
| ✅ | LLM-3 | llm_calls 表+7 类 caller 标签+GET /api/llm/usage 报表（`399d3b8d`） | High/M | ✅ |
| ✅ | TST-1 | 24 个退化 fixtures（6 形态×4 触点）+回归门（`40bac66f`）；<think> 三例经 LLM-6 修复翻转为回归守卫；仍不安全路径 9 条已 expected_failure 记录（负反馈候选：action/negation 交叉校验、planner 围栏=AGT-10 在修、指标伪造结构检查） | High/M | ⚠️ |
| ✅ | MEM-5 | recency 加减分+age_days+"(N 天前)"标注（`ef56b2e7`） | Med/S | — |
| ✅ | MEM-6 | 按 kind 定向查询+各类上限（`42731506`） | Med/S | — |
| ✅ | MEM-7 | 负反馈闭环落地（`0158ea0d`）：任务失败自动降档+面板"没用/有误"按钮+API、蒸馏 support 计净值、全程审计 | Med/M | — |
| ✅ | MEM-8 | raw 侧 low 过滤+raw_quota 保底名额（`d3729dd0`） | Med/S | — |
| ✅ | AGT-5 | 路由 prompt 注入门参数 schema 摘要+提示词 v2 约束键名（`3c5f069c`） | Med/S | — |
| ✅ | AGT-6 | 无 LLM→status=skipped 不渲染警告；触发面收窄到 decision/confirm/带指标步骤（`b53254cd`） | Med/M | — |
| ✅ | AGT-7 | 预算按门数动态（门数+2 可配）+耗尽显式消息+confidence<0.6 降级人工（`a9ad8eca`） | Med/S | — |
| ✅ | AGT-8 | AUTO replan 改走结构化 replan 路径带 constraint（`4aa1b8ba`） | Med/S | — |
| ✅ | AGT-9 | 调参门与选实验门确定性红旗落地（`20dc685d`，样本量/特征数/不平衡/OOT 缺失/gap/CI 重叠/failed candidate） | Med/S | — |
| ✅ | AGT-10 | 三条解析路径改 load_json_object（`f8360fa0`），评测集 planner 围栏 case 随之转安全 | Low/S | — |
| ✅ | MEM-9 | 触发词表拓宽+reserved 主题需主题级判定+被拒给回执不再静默（`19381de`） | Low/S | — |
| ✅ | MEM-10 | 触发器接通+吞错改记录+consolidate 返回错误计数（`bdc10ede`） | Low/S | — |
| ✅ | MEM-11 | ids 截断计数+3 样本、3000 字符总预算、审计保全量（`96d4f409`） | Low/S | — |

## 8. 阶段七：S4/S5/S6（12–19 天）

| 状态 | ID | 事项 | 工作量 |
|---|---|---|---|
| ✅ | S4 | 两 commit 落地（`8a6a5ed6`/`a4ce4162`）：双通道挖掘（树路径+单变量，确定性）、瀑布评估+重叠矩阵、三方共享条件求值（往返锁）、RULE_STRATEGY 模板+「选 1,3,5」规则集门+采纳面复用 S2；e2e 含门覆盖重跑 | 4–6天 |
| ✅ | S5 | 两 commit 落地（`d7cd5292`/`d9e551d6`）：monitoring_plan.py 单一来源+run_strategy_monitoring（委托 monitor_run 内核+策略面 ±5pp/±10pp 漂移分级带 IEEE-754 边界 eps、无标签 n/a、纯规则跳 PSI/CSI、last_run_at 写回+审计）、STRATEGY_MONITORING 告警门（挂报告步——executor 先暂停后执行的机制现实，语义等价；红灯三选项处置→next_action 不自动建任务）、逾期可见（health 计数+/api/strategies/monitoring-due）；e2e 红灯全旅程 | 3–5天 |
| ✅ | S6 | 三 commit 落地（`6cee5d5c`/`4c8d239d`/`8c6aece9`）：slice_aggregate 白名单算子（标识符白名单防注入+确定性 ORDER BY+截断/空结果旗+审计）、adhoc_analysis 模块（LLM 出 spec→平台校验→口径确认门先行→幻觉列中文澄清，新 PromptSpec 入注册表）、limit_pricing_matrix（2×2×2 手算锁+PD 代理旗+矩阵门后落 csv artifact）、challenger 报告（matrix-heat+数字跟随工具输出+无基线优雅降级）；问数分支已接线（`1749fdb1`：分派前保守守卫三条件、pending spec 走消息 metadata 零新状态表、AUTO 永不误触、6 项 turn 级测试含手算出表） | 5–8天 |

## 9. 阶段八：审查 Batch 4 —— 领域剩余（约 1 周）

| 状态 | ID | 事项 | 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | DOM-4 | 校准默认在 train 内切折拟合、test/OOT 出样评估（`128cf6fc`）；显式 fit_split 向后兼容并标注 in-sample | Med/S | — |
| ✅ | DOM-5 | 分段表重建（`20a64468`）：train 定界三 split 共享分箱、累计坏账率/通过率、方向感知、cutoff 算例注解进 sheet 说明 | Med/M | — |
| ✅ | DOM-6 | eval_metric 接入冠军选择全链路（`a86c25ae`）：lift head/tail 5/10 进 ModelMetrics、response_lift 场景按 lift 选优；顺修 DB 回读静默丢字段 bug | Med/M | — |
| ✅ | DOM-7 | psi_split/psi_watch 进 screen（可选 max_feature_psi 观察阈值）+ 报告单变量 sheet 加 psi_vs_train 列（`fac87c9e`） | Med/S | — |

## 10. 阶段九：审查 Batch 5 —— 看得见的体验（约 2 周）

| 状态 | ID | 事项 | 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | UX-2 | agent 时间线挂载同款结构化控件（`e4384807`）：共享 driverGateBodyHtml、旧门只读快照、控件与文本双通道并存、payload 与 manual 一致 | High/M | ✅ |
| ✅ | UX-4 | 筛选表全面升级（`cdd5237d`）：搜索/七列排序/分类 chips/批量可见操作/50 行分页/已选计数/泄漏与嫌疑列强制 override 理由随确认提交 | High/M | ✅ |
| ✅ | VD-1 | 图表语言下沉（`9a9eb2c8`）：数值列 databar、PSI 三段徽、KS/AUC 色阶、match_rate 条形、冠军行高亮+failed 灰化带原因、tabular-nums | High/M | ⚠️ |
| ✅ | VD-2 | 门卡片形态落地（`5d9e7563`）：强调竖条+盾形图标+玻璃底+红旗警示区+"确认后将执行:<下一步>"后果文案 | High/M | ⚠️ |
| ✅ | VD-3 | 骨架系统落地（`5e4189f7`）：块/行/表基元+reduced-motion 降级，接入任务切换/门表格/计划栏 | High/M | ⚠️ |
| ⬜ | VD-4 | 校准可靠性曲线/分数分段数据已产出但前端无图表 | High/M | ✅ |
| ✅ | VD-5 | 发光资产已用真 logo alpha 蒙版重建（`cfed1796`，PIL 高斯 halo 不重绘像素）+吉祥物接 V2 门/红旗/完成事件；**设置项默认关闭，待用户视觉验收后再默认开** | Med/M | — |
| ✅ | UX-5 | plan rail 渲染 loop_events（replan/no_progress+介入捷径）+重规划计数徽+活跃子 agent 行（`89452d8e`） | Med/M | — |
| ✅ | UX-6 | 冲突列+真实冲突值样例+first/last 语义说明+"排除该特征表"出口（`9d9cb996`；"按时间列取最新"策略记为后续） | Med/M | — |
| ✅ | UX-10 | "等待确认：<步骤>"与真生成态区分+按工具定制后果文案（`67412a57`） | Med/S | — |
| ✅ | UX-11 | 22 文件用户可见文案全角化（`893f8404`，LLM prompt/代码/正则不动） | Low/S | — |
| ✅ | UX-12 | XHR upload.onprogress 实时百分比（>10MB 显示，`1959915d`） | Low/S | — |
| ✅ | VD-6 | 状态三通道（底色+语义 SVG 图标+边框，`4137593a`），色弱可辨 | Med/S | — |
| ✅ | VD-7 | KS/IV databar+IV 四档徽标+泄漏 watch 行警示边框（`c22da798`），全 token 化 | Med/S | — |
| ✅ | VD-8 | dark 玻璃立体感恢复（`bbd38165`）：双 inset 高光/阴影 token 化、hero-glow 提到 0.16-0.18 | Med/S | — |
| ✅ | VD-9 | databar scaleX 生长+逐行 stagger、KPI 淡入、门卡片入场（`4811af6c`），全部尊重 reduced-motion | Med/S | — |
| ✅ | PERF-7 | **经穷尽验证为非缺陷**（审查证据已陈旧）：LLM-9 节流落库+轮询并行已实现真增量，实测单次 POST 期捕获 5 个中间快照——无代码改动，结案记录在案 | Med/M | — |

## 11. 阶段十：审查 Batch 6 —— 产品化运营（3–4 周）

| 状态 | ID | 事项 | 影响/工作量 | 验证 |
|---|---|---|---|---|
| ✅ | GAP-1 | 编码 try 链（utf-8→sig→gbk→gb18030）+长数字列防截断读为字符串+摄取报告含 warnings（`ce39ec67`） | High/M | — |
| ✅ | GAP-2 | purge_task 同事务清理全资产+引用计数保护+task.delete 审计+purge-preview 端点+删除框摘要（`caf21fe7`/`2988c155`） | High/M | — |
| ✅ | GAP-3 | 审计读取面落地（`b9dfcd3f`）：GET /api/audit 多维过滤+分页、CSV 流式导出、任务审计时间线端点、新索引 idx_audit_target_ref_at；15 个新测试 | High/M | — |
| ✅ | GAP-4 | 数据字典全链路（`fbda4883`）：agent/data_dictionary.py 统一识别/注册/查询，四个 setup 流通用注册、筛选门与 JOIN C1/去重门带业务含义标注、LLM 门决策 prompt 注入紧凑字典上下文、handoff dictionary.csv 用真实业务名；端到端测试 | High/M | ✅ |
| ✅ | TST-2 | 流式上传+护栏（`c0e4b593`：8MB 分块、CSV 2GB/Excel 500MB/200万行可配、Content-Length+流式累计双保险、Excel read_only 预检）+ 本地路径注册端点（`12792c1c`：loopback 守卫/防遍历/白名单/复制入 workspace/content_hash 幂等/硬审计）；其分支全量 2488 通过 | High/M | ✅ |
| ✅ | TST-3 | 真 e2e 落地（`1978319c`）：真 `marvis serve` 子进程全旅程（JOIN→standard_modeling 全门→PMML），三连跑确定性验证（~30s/次，e2e marker） | High/M | ✅ |
| ✅ | TST-4 | 真进程验证落地（`677064bb`）：真 OOM 进程树杀+审计 peak、含孙进程的进程组杀、子进程 env 白名单真回显、notebook kernel 真杀；检测力经"故意弄坏"自证；无生产 bug | High/M | ✅ |
| ✅ | ARCH-1 | legacy shim 拆除（`e3a7d6f5`/`4f6b845e`）：36 处私有 seam 迁入新 agent/validation_app_service.py 公有面，routers 直连、_agent_api()/legacy_api 服务定位器删除、api.py 955→184 行、DriverTurnRuntime 全类型化；routers/ 内 legacy_api 引用归零 | High/M | ✅ |
| ✅ | ARCH-2 | modeling tools 拆 13 子模块（`5103dede`：prepare/feature/train/select/calibrate/delivery/monitor/report_tools+_common/_runtime/scoring 等），tools.py 保留聚合门面、对外 import 面不变；B6h 字典代码已随合并迁至 feature_tools | High/M | ✅ |
| ✅ | GAP-5 | 可选 MARVIS_LOCAL_TOKEN 写操作令牌（`668ddb73`）：非安全方法须带 X-Marvis-Token（恒时比较），token 只注入本地 index 页由 api.js 回带、远端读永不下发；未配置=行为不变 | Med/S | ✅ |
| ✅ | GAP-6 | 只读注册表 API（跨任务列表/详情/分页，`fc4e737f`）；前端面板按预授权降级（进 UX 后续） | Med/M | — |
| ✅ | GAP-7 | content_hash 指纹复用（同内容跨任务共享 parquet+profile，dedup 审计，`da47ae0e`），删除引用计数联动 | Med/M | — |
| ✅ | GAP-8 | 测试连接端点+设置按钮+health.llm_configured（`efbc8880`） | Med/S | — |
| ✅ | GAP-9 | marvis backup/restore 命令（SQLite backup API 一致性快照+tar，`4522a628`） | Med/S | — |
| ✅ | GAP-10 | logging_setup+RotatingFileHandler 落 workspace/logs+关键模块事件（`96ecdf4c`） | Med/S | — |
| ✅ | TST-5 | slow/e2e/llm markers+strict+scripts/check --fast（`8bcbd6d3`）：39 慢测打标（真 durations 数据），fast 层 2374 用例 6m14s（全量 14m，2.25×——审查的 3 分钟目标经实测不改测试逻辑达不到，如实记录） | Med/S | — |
| ✅ | TST-6 | redaction 缺口修复+会话转录脱敏（`f1fc7f71`，B6d） | Med/M | ✅ |
| ✅ | TST-7 | 交互式 kernel 接入同一 env allowlist（`b8b43bae`，B6d），与 worker 路径对称 | Med/M | ✅ |
| ✅ | TST-8 | CI security job（pip-audit+bandit -ll，continue-on-error 观察期，`22299d71`）；首扫：pip-audit 零漏洞、bandit 42 中危（40×B608 f-string SQL 等）**留待 FIN-2 审查裁决** | Med/S | — |
| ✅ | TST-9 | 并发测试落地（`ecb2506e`）挖出 3 个真 bug 并已全修（`cabe409c` confirm_step 单发 CAS、`0b849f6e` start_job UNIQUE→ConflictError 只剩 202/409、`91ffc3da` DuckDB 每操作独立连接×12 调用点+PERF-8 配置保持）；xfail 全部转正为回归守卫，HTTP 双确认五连跑稳定 | Med/M | ✅ |
| ✅ | ARCH-4 | 五 handler 收敛为 _TurnHandlerSpec 参数化核心（`ca0c156c`），差异轴显式表格化，共享尾部 diff 字节等价验证；**发现真缺口：feature/strategy/vintage 从不传 settings/task→MEM-1 捕获只对 join/modeling 生效，修复归入 S2 记忆接线 commit** | Med/M | ✅ |
| ✅ | ARCH-5 | worker 协议 protocol_version 握手（`74fa5c7`）：不匹配→typed error+审计；入口 import 轻量护栏保持绿 | Med/M | — |
| ✅ | ARCH-6 | pipeline 拆四模块（`6a411a3`：errors/cellgen/io/memory，门面保 monkeypatch 命名空间语义）+ 阶段边界/重试点/异常路径结构化日志；auto_distill 门控（INV-4）原样 | Med/M | ✅ |
| ⬜ | ARCH-7 | 错误分类学散乱：132 处手写 HTTPException、error_kind 裸字符串 | Med/M | — |
| ⬜ | ARCH-8 | 四个 pack 的 _Runtime 复制粘贴，pack SDK 公共层缺位 | Med/S | — |
| ✅ | ARCH-9 | 模板按域拆八模块（`aa064cce`：_shared/sample_echo/validation/modeling/join/feature/strategy/monitoring），sample.py 缩为 36 行注册门面；拆分前后 WorkflowTemplate 输出 SHA256 字节等价验证 | Med/S | ✅ |
| ✅ | ARCH-10 | schema_version 落 PRAGMA user_version（`51a3130`）：编号迁移清单、旧库无损升级测试、重复 init 幂等 | Low/S | ✅ |
| ✅ | ARCH-11 | 纯函数簇抽出（`ea96e8b5`：metric-tables 609 行/precision-consistency 86/step-checker 38，app.js 6970→6254）+ 状态所有权地图交付（40 全局态归属定界）；深拆的硬边界=Node harness 内联整文件直呼顶层符号（编排函数被测试架构钉住），继续拆需 harness 重构——记录为显式架构决策非欠账；真浏览器 smoke 3 通过 | Med/L | ✅ |
| ✅ | PERF-6 | 轮询热路径收敛（`fe5180c4`）：GET /api/tasks 11 连接→2（与任务数无关）、evidence mtime/size 缓存、journal_mode PRAGMA 每库一次；查询计数断言守护 | Med/M | — |
| ✅ | PERF-9 | 全量静态版本化（`a752660d`）：_static_asset_version rglob 全部 js/css、importmap 把裸相对导入重写为 ?v= URL、/static Cache-Control 分层（?v= immutable / 否则 no-cache）；真 Chromium smoke 验证 | Med/S | — |
| ✅ | PERF-10 | read_frame 深拷贝改 CoW 视图（`d3336acf`）：内存峰值随配方数从线性变常量；六配方训练字节等价验证 | Med/S | ✅ |
| ✅ | UX-8 | 8 个死代码模块删除（`caee3d9f`）：join_review/plan_view/plan_confirm/workflow_create/subagent_view/loop_progress/memory_manager/draft_manager，全部先复核零挂载；7 个专属测试文件同步清理 | Med/M | — |
| ✅ | VD-10 | （随 UX-8 落地）零挂载模块已清除；join_review 的"中止去重"选项无在线等价物，按纯死代码删除不移植 | Med/S | — |
| ✅ | VD-11 | 零歧义重复模式已收敛（`9bdaac9e`：--radius-pill 37 处，逐字节等价验证）+ 全类别 token 清单（docs/reviews/2026-07-03-vd11-design-token-inventory.md：radius 二档/颜色/spacing/shadow/type scale 值×次数×归类）；**取值调整按红线待用户对比稿拍板**（清单即底稿）；hex 核实无重复 legacy | Med/L | ✅ |
| ✅ | LLM-4 | role_overrides 按 caller 分级路由（`20f5da11`）：planner/critic/router/gate/distill 可各指模型，默认不配=现状 | Med/M | — |
| ✅ | LLM-5 | context_window/max_tokens 进 profile+预检预算+typed error（`049c6164`）；gate 内容/planner catalog/记忆注入三触点接截断 | Med/M | — |
| ✅ | LLM-6 | 终文剥离 <think> 段后再抽 JSON（`572c6e33`，流式期直通、落库剥离） | Med/M | — |
| ✅ | LLM-7 | 超时/连接错误指数退避重试一次（`367aa13f`），审计记 retry | Med/S | — |
| ✅ | LLM-8 | draft 授权接 JSON 抽取+错误回灌重试（`b642a308`） | Med/S | — |
| ✅ | LLM-9 | 落库节流 ≥500ms/≥512 字符+终稿完整落库（`91d3b1e1`） | Med/S | — |
| ✅ | LLM-10 | llm_prompts 注册表收敛 14 个提示词带版本（`049c6164`），usage 记录 prompt_name/version | Low/M | — |

## 11.5 收官双 review 循环（用户 2026-07-03 指令，DoD-11）

| 状态 | ID | 事项 |
|---|---|---|
| ⬜ | FIN-1 | 落地核验 review：逐项抽查全部 ✅ 条目的真实代码证据（plan-vs-code），发现"勾了没落地/落偏"即开修复项 |
| ⬜ | FIN-2 | 全量 code review 第 N 轮：多镜头审查最终代码（bug/可提升），高影响发现对抗验证 |
| ⬜ | FIN-3 | 修复循环：FIN-1/FIN-2 发现项修复→回归→再审，直至一轮无新 critical/high |

## 12. 长线追踪（不阻塞完全体 v1；⏸️ 需写明理由）

| 状态 | ID | 事项 | 来源 |
|---|---|---|---|
| ✅ | LT-1 | fixtures 扩面（`7222b2d8`）：三步预处理链导出一致性、单特征 PMML、policy 部分满足端到端、SEL-7 模型卡呈现；**挖出 2 真 bug**（_apply_cap 只读数组崩溃、模型卡吞 warnings）已修复（`99e6ca8b`/`f9f15378`）xfail 转正 | 追踪器 |
| ✅ | LT-2 | AUTO 安全矩阵（`26f3d01d`）：五门 bare-confirm 阻断矩阵、stale token AUTO 路径、GAP-4 字典注入×阻断组合；**挖出严重 bug：risk_flags 结构性死代码**（envelope 从不设旗、composer 从不写 meta，真实交付门裸 confirm 放行）已修复（`7541db3d`：信封四路推导 risk_flags+composer 落 meta，纯信息门不设旗防过阻断）xfail 转正；另确认 stale-control 唯一机制=expected_step_id 比对 | 追踪器 |
| ⬜ | LT-3 | PlanDriver 收尾：per-tool gate adapters + schema-driven adjust specs（验收=PlanDriver 不再 import 任务特定渲染细节） | 追踪器 |
| ✅ | LT-4 | 重试 UX 收口（`d6d17ff`）：整体替换语义中文警示常显（后端确证 UPDATE 全列覆盖非合并）、真 schema（required 红星/enum 下拉）经既有 GET /api/plugins/{name}/tools 懒取合并到推断 stub、array/object 与取 schema 失败均回落 JSON 编辑器；364 前端+插件测试绿 | 追踪器 |
| ✅ | LT-5 | UoW 收尾（`ee020ce`）：全建模多写工具盘点表；两处真修（champion refit attach 失败零清理→快照回滚+重试回归、报告 xlsx 直写终path→stage/promote+审计单事务）；train 路径 file-then-DB 缺口如实文档化（recipe uow 穿线判定过侵入，DB 侧本已原子+既有补偿器）；三类写路径文档进 transactional.py docstring；223 测试绿 | 追踪器 |
| ✅ | LT-6 | 证据驱动列投影（`cf36f01a`）：9 recipe 训练读 82MB→18MB（-78%）、monitor scored 分支 -93%、champion 重训同改；全帧回写类路径（score/reject/report）复核后不改（会丢输出列）；442 测试+LT-8 计数守卫全绿 | 追踪器 |
| ✅ | LT-7 | 证据裁决（`c5ad8499`，文档 commit）：盘点后无路径在不冒精度漂移/输出保真风险下过 30% 收益线（筛选已列批读、profile 已采样有界、dedup 已有 SQL 路径），不硬改 | 追踪器 |
| ✅ | LT-8 | 三件套落地（`435bb86c`）：筛选列批读计数（20万行×80列，7 次调用严格匹配批算术）、多配方端到端单次加载守卫（真实 backend 非 mock）、join 连接数与行数无关+双实例确定性；六连跑零 flake，未发现回归点 | 追踪器 |
| ✅ | LT-9 | 已裁决：subprocess+护栏为 V2 终态（威胁模型=单机单用户；TST-4 真进程证据链）；OS 沙箱升级条件=V3 多用户，见 v2-longtail-adjudications.md | 6-28+roadmap |
| ✅ | LT-10 | 已裁决：triple-opt-in legacy-only 为终态（settings 双开关+env var 三重门槛既有）；不做 RPC 化，只收安全修复，见 v2-longtail-adjudications.md | 追踪器 |
| ⬜ | LT-11 | agent 推荐产品化：引用 evidence refs、给 tradeoff、AUTO 解释 bounded action 为何安全 | 追踪器 |
| ✅ | LT-12 | 已裁决：触发条件不存在，关闭；未来出现行级保真要求时按 join 引擎设计重开新 spec，见 v2-longtail-adjudications.md | 6-28 |
| ✅ | LT-13 | 分页审计（`a4accccb`）：全端点盘点表，任务内 experiments/plans 两个线性增长端点补 opt-in limit/offset/total（默认全量兼容），datasets/plugins/skills 复核判不需要 | 追踪器 |
| ✅ | LT-14 | 指南+真诊断（`0f26b4e4`）：docs/sample_weight_guide.md；**发现 leakage_risk 一直硬编码 low 从未计算**，改为真算权重×目标相关（|corr|≥0.3→high+target_correlation 字段），门提示升级 warning 点名泄漏 | 追踪器 |
| ✅ | LT-15 | 全部批次 spec 已出：S1a（已实现）、S2-S6（docs/plans/specs/v2-s2..s6-*.md，函数级）；S5 范围校准注明趋势/EL/报告项由 S3 吸收 | 策略计划 |
| ✅ | LT-16 | 已裁决：EXC 清零=仓内载体完成；真实数据对照转显式外部输入门（用户提供脱敏样本即跑，承接件就绪），见 v2-longtail-adjudications.md | roadmap |
| ✅ | LT-17 | 已裁决：机制定型（每 2-3 阶段一轮聚焦审查），已执行三轮+收官 FIN 循环为第四轮，见 v2-longtail-adjudications.md | 本轮教训 |
| ✅ | LT-18 | 蓄水池已建：五类 V3+ 显式不做项与准入门槛记录于 v2-longtail-adjudications.md | 产品选择 |
| ⬜ | LT-19 | 每阶段完成后同步更新本清单与记忆索引 | 流程 |

## 13. 映射、去重与已收尾文档记录

**合并去重**：agent 镜头 AGT-2 = MEM-1（双向断链，保留 MEM-1 为主）；FS-3 = PREP-3（类别特征，合并执行）；TUNE-8 ≈ SEL-3 ≈ PREP-6（LR 预处理+失败隔离，一份 spec）；roadmap-1c/1d 分别被 TUNE-1/FS-1 吸收；roadmap-1e 被 TST-2 吸收；DOM-2/3/8/11/12、UX-9 被 S 批次吸收（行内已标注）；追踪器 8 条同源项已在对应阶段行内标注补充细则。

**已收尾、不再携带待办的文档**（2026-07-02 代码核实）：`v2-completion-plan.md` §八 构建顺序步骤 0–5 全部落地；`settings-ia-refactor.md` 除死代码清理（并入 UX-8/VD-10）外全部完成。`v2-comprehensive-improvement-plan.md` 的执行追踪职能由本文档接替（其验证记录保留为证据）。

**审查已确认修复项**（不在本清单）：见审查报告 Appendix C。

---

## 附录 A：建模方法学极致审计完整证据（2026-07-02，四镜头）

> 对应 §3 的 37 条缺口：每条含行业对照、file:line 证据、步骤化改法与对抗验证记录；各镜头开头附总体判断与 done_well 清单（共 53 项）。缺失类断言均经独立验证员穷举同义实现与兜底后判定。行号为 2026-07-02 工作区（分支 codex/v2-plugin-tool-runtime）。

### 特征预处理

**总体判断**：预处理这块"评分卡主干"是扎实的：WOE/IV 单一约定+拉普拉斯平滑+缺失单独箱、scorecard 配方内部 train-only 拟合 WOE 且 woe_maps 随 artifact 落盘并能在打分/PMML 精确重放、MLP 用 sklearn Pipeline 把 impute+scale 打包进模型，这些都做对了。但离"极致"还有明显距离，核心断层有三个：(1) FEATURE 阶段的 fit 类工具（impute/normalize/cap/woe_encode）没有 train-only 纪律，WOE 默认甚至用 test/OOT 的标签参与拟合，评估指标系统性虚高；(2) 除 scorecard/MLP 外，预处理参数只活在工具响应 JSON 里，不进 artifact，且变换原地覆盖同名列，新数据打分会静默拿原始值喂给按变换后分布训练的模型；(3) 类别特征、日期列、哨兵值三类信贷数据的常规信息源在链路上基本被丢弃或污染。前两条影响的是"测得的 KS 可信度"和上线一致性，第三条直接压低同样本同标签下可达到的 KS 上限。

**已做到位（done_well，经核实）**：

- WOE/IV 全平台单一约定（WOE=ln(good/bad) 分布比）+拉普拉斯平滑，缺失默认单独 NA 箱并产出 na_woe（marvis/feature/iv.py:10-92，na_as_bin 默认 True）
- woe_encode 打分时 NaN/越界值统一落 na_woe（无 na_woe 时 0.0），不会崩（marvis/feature/encode.py:41-54）；assign_bins 边界用 ±inf 且越界 clip 进端箱（marvis/feature/binning.py:174-187）
- scorecard 配方 WOE 严格 train-only 拟合（_fit_woe_maps 只吃 train），test/OOT 用 train 映射 transform 后评估（marvis/packs/modeling/recipes/scorecard.py:39-69,114-136）
- scorecard artifact 完整落盘 woe_maps+scorecard_table+factor/offset，打分路径（_ModelArtifactScorer tools.py:4134-4160）、handoff 打分 notebook（handoff.py:588-596）、PMML（WOE ExpressionTransformer 含 NaN→na_woe 默认分支，artifact.py:269-337）三面都能精确重放 WOE
- MLP 配方把 SimpleImputer(median)+StandardScaler+MLP 做成单一 sklearn Pipeline 在 train 上拟合并整体 joblib 落盘，打分自动重放（marvis/packs/modeling/recipes/mlp.py:42-51,74）
- 单调分箱基础设施完整：chimerge/tree/equal_freq/manual 多方法、auto 方向解析、违反单调的相邻箱按卡方最小合并（marvis/feature/binning.py:51-171）；tree 分箱 seed 固定且 min_samples_leaf=5%
- 切分反泄漏做得细：OOT 优先按时间截断、group_cols 整组同侧防近重复样本跨集、规则切分先冻结、固定 seed 可复现、空集合硬报错（marvis/packs/modeling/prepare.py:132-185,293-305）
- NaN 标签强制确认门贯穿 feature 工具与全部训练配方，标签从不被静默 coerce（marvis/data/labels.py:28-135；feature/tools.py:53-57,222-226,308-313）
- 入模前质量检查覆盖：>95% 缺失 block、常量列 block、重复列 block、与 target 相关>0.95 泄漏嫌疑 block、高基数类别 warn、accept-only 样本 warn（marvis/packs/modeling/readiness.py:46-128,155-181）
- screen_features 有泄漏 KS 硬门（默认0.40）+模型输出命名嫌疑+不可用列剔除，非二分类目标自动跳过 KS 筛选（marvis/packs/feature/tools.py:108-195）
- onehot 类别顺序确定性、max_categories 防爆护栏、未见类别 transform 为全零（handle_unknown=ignore）（marvis/feature/encode.py:11-31,57-66）
- 派生特征安全性：除零→NaN 而非 inf（derive.py:49）、agg join 行数膨胀硬报错（derive.py:86-87）、派生列名冲突/重复硬报错（derive.py:109-111,238-241）；LLM 只推荐交叉对、指标一律平台计算（derive.py:17-21）
- impute/normalize/cap 工具都把拟合出的参数（fill_values/scaler_params/bounds）回显给调用方，至少在对话内可审计（marvis/packs/feature/tools.py:385,404,424）

**缺口**：


#### PREP-1 · FEATURE 阶段 fit 类变换无 train-only 纪律：WOE 默认用 test（无 split 时连 OOT）标签参与拟合，impute/normalize/cap 全量池化拟合

**KS 影响：High · 工作量：M · ✅ 对抗验证 CONFIRMED**

**缺口**：行业极致做法是所有 fit 类变换（WOE 映射、分箱边界、填充值、标准化参数、截断边界）只在 train 上拟合、对 test/OOT 只 transform。现在 WOE 编码用了评估集的标签算每箱 good/bad 分布——这是直接的目标泄漏，会让下游模型的 test KS（默认配置）乃至 OOT KS（无 split 时）系统性虚高，模型选择被污染；统计类变换池化拟合也让 OOT 分布信息渗入训练表征。注意 scorecard 配方内部路径是干净的，此缺口只打击 woe_encode/impute/normalize/cap 工具组合出的流程（如 woe→LR、impute→LR）。

**证据**：marvis/packs/feature/tools.py:338-347 _woe_fit_frame：无 split_col 时 fit_frame=整个 frame（含未来的 test/OOT 行及其标签）；有 split_col 时 holdout_values 默认只有 ("oot",)，test 仍参与 WOE 拟合。tools.py:364-424 tool_normalize/tool_impute_missing/tool_cap_outliers 直接在全量 frame 上拟合 min/max、mean/std、median、分位边界；manifest.json 确认这三个工具的 input schema 完全没有 split_col/holdout 参数。而 split 列通常在 MODELING 阶段才由 prepare_modeling_frame 生成（prepare.py:66），所以可组合流程里 FEATURE 阶段根本无 split 可传。

**改法**：1) 给 impute_missing/normalize/cap_outliers 三个工具加 split_col+holdout_values（或 fit_on=train）参数，拟合帧只取 train 行，transform 应用全表；2) woe_encode 的 holdout 默认从 ("oot",) 改为 ("test","oot")，且无 split_col 时要求显式确认或直接报错（typed error，同 NaN 标签门风格）；3) 流程层面把"先切分后做 fit 类变换"写进 FEATURE→MODELING 编排约束：prepare_modeling_frame/make_split 先行，FEATURE fit 类工具检测到无 split 列时给出警告字段；4) 加回归测试：同一数据 train-only 拟合 vs 全量拟合的 test KS 差异应可复现。

**验证说明**：逐条核实全部属实。(1) marvis/packs/feature/tools.py:338-347 _woe_fit_frame：无 split_col 时直接 return frame（整帧含 test/OOT 行及标签参与 WOE 拟合）；有 split_col 时 line 342 holdout_values 默认 ("oot",)，即默认只排 OOT、test 标签仍进 compute_woe_iv（tools.py:317-323 用 fit_frame 算 edges 和每箱 good/bad）。(2) tools.py:364-385/388-404/407-424 三个工具 tool_normalize/tool_impute_missing/tool_cap_outliers 均在全量 frame 上拟合 min-max/mean-std、mean/median/mode 填充值、IQR/分位截断边界，函数体无任何 split 逻辑。(3) manifest.json 中 normalize(300-328)、impute_missing(330-358)、cap_outliers(360-389) 的 input_schema 无 split_col/holdout_values 且 additionalProperties:false——参数想传都传不进去；全 manifest 只有 bin_feature(85-86) 和 woe_encode(244-245) 有这两个参数。(4) split 列确实在 MODELING 阶段才由 prepare_modeling_frame 生成：marvis/packs/modeling/prepare.py:63 `if split_col:` 否则 line 66 `_make_split(frame, split_config, seed=seed)` 造 "split" 列，因此可组合流程中 FEATURE 阶段通常无 split 可用。(5) 无兜底：marvis/feature/transform.py 是这四个统计原语的唯一实现，仅被 feature pack tools 消费；其他 pack（data_ops/modeling/strategy/v1_compat）manifest 均无 impute/normalize/winsor/cap 工具；全仓搜 SimpleImputer/StandardScaler/MinMaxScaler/fit_transform/winsor/clip 只命中 recipes/mlp.py（sklearn Pipeline 只在 train 上 fit，干净）。(6) 断言中"scorecard 配方内部路径干净"也属实：recipes/scorecard.py:39-53 先 split_modeling_frame 再 _fit_woe_maps(train,...) 只用 train 拟合，woe_encode 只做 transform；modeling/tools.py 与 handoff.py 中的 woe_encode 调用均是应用已存 woe_maps 的打分路径。(7) 库层原语其实支持 fit/transform 分离（apply_scaler 存在、params/fill_values/bounds 均返回），但 feature pack 没有任何 "apply" 类工具暴露，agent 无法通过工具组合出 train-fit → 全表-transform 的流程——缺口真实存在于工具层。附带发现：screen_features 与 bin_feature 也用同样的 ("oot",) 默认（tools.py:142/182），与 WOE 一致。

**验证员核实证据**：marvis/packs/feature/tools.py:338-347 _woe_fit_frame：无 split_col 时 fit_frame=整个 frame（test/OOT 行及标签全部参与 WOE 拟合）；有 split_col 时 line 342 holdout_values 默认 ("oot",)，test 默认参与 WOE 拟合（tools.py:307-323 用 fit_frame 计算 edges 与每箱 good/bad 分布）。tools.py:364-385 tool_normalize、388-404 tool_impute_missing、407-424 tool_cap_outliers 直接在全量 frame 上拟合 min/max、mean/std、median/mode 填充值、IQR/分位边界，无任何 split 逻辑。marvis/packs/feature/manifest.json：normalize(L300-328)、impute_missing(L330-358)、cap_outliers(L360-389) 的 input_schema 无 split_col/holdout_values 且 additionalProperties:false（参数无法传入）；全 manifest 仅 bin_feature(L85-86) 与 woe_encode(L244-245) 具备这两个参数。split 列由 MODELING 阶段 marvis/packs/modeling/prepare.py:63-66 生成（无 split_col 时 _make_split 造 "split" 列），故可组合流程中 FEATURE 阶段通常无 split 可传。无其他模块兜底：marvis/feature/transform.py 是四个统计原语唯一实现（且提供 apply_scaler 等 transform-only 原语，但 feature pack 未暴露任何 apply 类工具）；data_ops/strategy/v1_compat 等 pack manifest 无 impute/normalize/winsor/cap 工具；全仓 SimpleImputer/StandardScaler/fit_transform/winsor 仅命中 recipes/mlp.py（Pipeline 只在 train 上 fit，干净）。scorecard 配方内部路径干净：recipes/scorecard.py:39-53 _fit_woe_maps 只用 train 拟合、woe_encode 仅 transform；modeling/tools.py:4138/4154 与 handoff.py:592 均为应用已存 woe_maps 的打分路径。附带：screen_features/bin_feature 同样使用 ("oot",) 默认（tools.py:142/182）。影响面与断言一致：woe_encode/impute/normalize/cap 工具组合流程存在目标泄漏（WOE）与分布泄漏（统计变换），test KS（默认配置）乃至无 split 时 OOT KS 系统性虚高。


#### PREP-2 · 预处理器不随模型 artifact 落盘、打分期无法重放：填充值/截断边界/scaler/onehot 映射只活在工具响应 JSON 里，且变换原地覆盖同名列导致新数据打分静默错误

**KS 影响：High · 工作量：L · ✅ 对抗验证 CONFIRMED**

**缺口**：行业极致做法是预处理器与模型绑定为单一可序列化 pipeline（或 artifact 附 preprocessing spec），打分/PMML/handoff 精确重放。现在若 agent 走了 impute/cap/normalize 再训 LR/LGB 的路径，这些变换不改列名（out[col]=values 原地覆盖），模型 feature_list 与原始列同名——对新数据打分时 notebook/PMML/scorer 会拿未变换的原始值直接喂模型，不报任何错，分数静默错误；走 woe_encode 路径则新列名 *_woe 在新原始数据里不存在，打分直接 KeyError。平台内报告的 KS 不受影响（同一派生数据集上算），但交付一致性/生产稳定性是硬伤。与已知 DOM-3（缺 score_dataset 工具）不同：这是即使补了打分工具，也没有可重放的预处理链。

**证据**：marvis/packs/feature/tools.py:385,404,424 三个工具只把 scaler_params/fill_values/bounds 放进返回值，_register_frame(tools.py:481-505) 注册派生数据集时仅存 role/anchor_target/seed，不存变换参数；marvis/packs/modeling/artifact.py:96-120 persist_model_meta 的 meta 字典没有任何 preprocessing 字段（只有 scorecard 的 woe_maps/scorecard_table 例外）；handoff.py:588-596 打分 notebook 的 RMC_SCORE_FN 只对 dict+woe_maps 的 scorecard 重放 WOE，其余算法直接 predict_proba(dataframe[RMC_FEATURES])；artifact.py:240-244 LR/LGB/XGB 的 PMML 用 make_pmml_pipeline(model) 不含任何预处理层。全仓搜 "preprocess" 无预处理链持久化代码。

**改法**：1) 在派生数据集注册时把变换 spec（工具名+列+参数，如 {op:impute, col, strategy, value}）写入 dataset lineage（registry detail JSON）；2) 训练时沿 anchor_target 链把从"原始拼接表"到"建模表"的全部变换 spec 收集进 model_meta.json 新增 preprocessing 段；3) handoff 打分 notebook 生成时按 preprocessing 段生成对应 apply 代码（apply_scaler/fillna/clip/woe_encode 已有纯函数可直接调用）；4) LR/MLP 的 PMML 导出把 impute/scale 编进 pipeline（sklearn2pmml 原生支持 SimpleImputer/StandardScaler）；5) 加一致性测试：训练集上重放预处理链打分 == 训练时 train_scores。

**验证说明**：断言核心完全成立：特征包的 impute/cap/normalize/onehot 变换参数只出现在工具返回 JSON 中（marvis/packs/feature/tools.py:385/404/424/361），变换均为 out[col]=values 原地覆盖同名列（382/401/421）；_register_frame(481-505) 与 registry.register_existing(marvis/data/registry.py:72-90) 只存 task_id/role/anchor_target/seed，DB schema 无任何变换参数列（仅 db_schema.py:508 的 woe_maps_json，属 scorecard 模型内部 WOE）；persist_model_meta(artifact.py:96-120) 的 meta 键中无任何 preprocessing/imputer/scaler 字段；打分三条路径（handoff.py:588-596 notebook、modeling/tools.py:4127-4142 _ModelArtifactScorer、artifact.py:240-244 PMML make_pmml_pipeline）除 scorecard WOE 外均直接对原始列 predict_proba，无预处理重放；handoff 物料清单(handoff.py:57-78)不含预处理 spec。全仓搜 preprocess/imputer/winsor/transform_spec/lineage 等确认 data_ops/strategy/repositories 均无兜底。失败场景成立：impute/cap/normalize 后训 lr/lgb 对新数据打分静默错误；feature 包 woe_encode 生成 *_woe 新列则打分 KeyError；平台内 KS 不受影响。两处微小不精确：(1) scaler_params/fill_values 字面 grep 实际命中 4 行(385,392,402,404)而非 3 处，断言引用的 424 行返回的是 bounds 键（cap_outliers），属断言宽泛表述"scaler_params/fill_values/bounds"范围内；(2) mlp recipe(recipes/mlp.py:42-46) 将 SimpleImputer+StandardScaler 打进 pickled sklearn Pipeline，是模型内部预处理随 artifact 落盘并自动重放的窄例外——但它不覆盖特征包变换链，也不适用于 lr/lgb/xgb/catboost/PMML，不动摇断言主体。

**验证员核实证据**：marvis/packs/feature/tools.py:385(scaler_params)/404(fill_values)/424(bounds)/361(onehot mapping) 四个工具只把变换参数放进返回值；字面 grep scaler_params|fill_values 命中 4 行(385,392,402,404)全在该文件（断言原文"三处含424行"应更正为：424 行是 bounds 键）。tools.py:382/401/421 为 out[col]=values 原地覆盖同名列。_register_frame(tools.py:481-505) 与 register_existing(marvis/data/registry.py:72-90) 仅存 task_id/role/anchor_target/seed；marvis/db_schema.py 与 marvis/repositories/ 无 fill_value/bounds/onehot/mapping 任何列，仅 db_schema.py:508 woe_maps_json（scorecard 模型内部 WOE）。marvis/packs/modeling/artifact.py:104-120 persist_model_meta 的 meta 键为 artifact_id/algorithm/model_path/pmml_path/feature_list/params/seed/dataset_id/target_col/split_col/split_values/target_type/recipe_id/scorecard_table/created_at，无 preprocessing 字段。打分无重放：handoff.py:588-596 RMC_SCORE_FN 仅对 dict+woe_maps 的 scorecard 重放 WOE，其余 predict_proba(dataframe[RMC_FEATURES])；modeling/tools.py:4127-4142 _ModelArtifactScorer.raw_score 同构；artifact.py:240-244 make_pmml_pipeline 无预处理层（scorecard 例外在 254-285）；handoff.py:57-78 物料含 sample/model/pmml/calibration/dictionary/notebook，无预处理 spec。全仓兜底排查为空：data_ops/strategy 包无任何 scaler/impute/cap/woe 代码；无 transform_spec/lineage/provenance 持久化。唯一窄例外需补充：recipes/mlp.py:42-46 的 MLP 用 Pipeline(SimpleImputer median→StandardScaler→MLP) 整体 joblib 落盘，模型内部预处理可自动重放——但这是 recipe 内部标准化，不重放特征包变换，且 lr(recipes/lr.py:35-40 裸 LogisticRegression)/lgb/xgb/catboost 均无此结构。失败场景验证成立：特征包 woe_encode(tools.py:330-335) 生成 *_woe 新列并只在返回值给出 woe_maps，新原始数据打分 KeyError；impute/cap/normalize 路径新数据打分静默错误；平台内 KS 在同一派生数据集上计算不受影响。


#### PREP-3 · 类别特征全链路缺位：候选只取数值列、所有配方强转 float、无类别 WOE/target encoding/罕见类归并，CatBoost 未传 cat_features

**KS 影响：High · 工作量：L · ✅ 对抗验证 CONFIRMED**

**缺口**：行业极致的评分卡/机器学习流程里，省份/城市/渠道/职业/设备等类别列是重要信息源：类别 WOE（按类别聚 good/bad+罕见类归并+未见类回退全局 WOE）、高基数用 target encoding（平滑+train-only）、CatBoost 直接吃原生类别。现在这些列在候选推断阶段就被静默丢掉（或手动指定后直接崩溃），唯一出路是 ≤50 类的 onehot——高基数类别信息整体损失，同样本同标签下可达 KS 上限被压低。

**证据**：marvis/feature/candidates.py:28-34 candidate_numeric_features 只取 select_dtypes("number")；scorecard.py:127 train[feature].to_numpy(dtype=float)、lr.py:37、mlp/lgb/xgb 同样直接喂数值帧，object 列必崩；marvis/feature/encode.py:41-54 woe_encode 走数值分箱 edges，无类别→WOE 映射；encode.py:24-25 onehot 超 max_categories(默认50) 直接 raise 无归并降级；catboost.py:38-46 CatBoostClassifier.fit 未传 cat_features。全仓 grep "target_encod|rare" 无实现（readiness.py 只有 high_cardinality 警告）。

**改法**：1) woe_encode 支持类别列：按类别值聚合 good/bad 算 WOE，占比 < min_pct 的类别归并为 __rare__，未见类别回退 na_woe/全局 WOE，映射并入 woe_maps 持久化（与 PREP-2 联动）；2) catboost 配方增加 cat_features 参数（从 dataset 列 semantic_role==categorical 自动推断+允许显式指定），并让 candidate 推断为 catboost 路径保留类别列；3) 补 target_encoding 工具（train-only 拟合+m-estimate 平滑+映射落盘）；4) onehot 超限时降级为 top-k+其余归 other 而非报错。

**验证说明**：断言完全成立。两条 grep 复现：全仓 marvis/ 无 "target_encod" 命中；"cat_features" 全仓零命中（含 catboost.py）。同义词扫描（mean/leave-one-out encoding、TargetEncoder、OrdinalEncoder、sklearn OneHotEncoder、category_encoders、get_dummies、factorize、.cat.codes、frequency/count encoding、rare-category 归并）除 screen.py:85 一句注释外全部为空，确认无其他模块兜底。类别特征链路逐环验证：(1) 候选推断 candidates.py:30-33 只取 select_dtypes("number")，且 agent/sample_setup.py:120、packs/modeling/prepare.py:42、packs/modeling/tools.py:3218、packs/feature/tools.py:460 四条推断路径全部走它，object 列静默丢弃；(2) 所有配方直接喂原始帧或强转 float——scorecard.py:127 chimerge 只能吃 float，lr.py:36-38、mlp.py:48、lgb.py:54、xgb.py:52 直接 fit(train[features])，手动指定类别列必崩；(3) woe_encode（encode.py:44-47）只支持数值分箱 edges，无类别→WOE 映射；(4) onehot（encode.py:24-25）超 max_categories(工具默认50) 直接 raise，无罕见类归并降级；(5) catboost.py:40-46 fit 未传 cat_features，CatBoost 原生类别能力被浪费（喂 object 列同样报错）；(6) readiness.py:116-127 对高基数类别只发 warn 不提供出路；(7) data_ops 清洗仅 strip/lower/upper/to_numeric/to_datetime，notebook/handoff 代码生成复用同一数值 woe_encode。仅有两处不改变结论的细微出入见 corrected_evidence。

**验证员核实证据**：核实后证据（全部行号已对照当前工作区代码）：marvis/feature/candidates.py:30-33 candidate_numeric_features 仅 probe.select_dtypes("number")（断言写 28-34，实际选择逻辑在 30-33）；四个调用点 agent/sample_setup.py:120、packs/modeling/prepare.py:42、packs/modeling/tools.py:3218（_resolve_feature_cols）、packs/feature/tools.py:460 全走该推断，类别列在候选阶段静默丢失。scorecard.py:127 values = train[feature].to_numpy(dtype=float)；lr.py:36-38、mlp.py:48、xgb.py:52、lgb.py:54 直接 fit 原始帧。encode.py:44-47 woe_encode 强制 float edges + to_numpy(dtype=float)，无类别 WOE/未见类回退全局 WOE 机制；encode.py:24-25 onehot 超限 raise FeatureError，packs/feature/tools.py:358 默认 max_categories=50，无归并降级。catboost.py:40-46 CatBoostClassifier.fit 未传 cat_features。readiness.py:116-127 high_cardinality 仅 severity="warn"。全仓无 target/mean/frequency encoding、无 rare 归并实现（screen.py:85 仅注释；screen.py:47/108 还把非数值列 to_numeric(errors="coerce") 后按缺失剔除）。两处补充修正：(a) encode.py:34-38 存在 label_encode（普通序号编码，未见类=-1），但全仓无任何调用点，属死代码，非 target-based，不构成兜底；(b) feature/derive.py:65-89 aggregate_feature（经 packs/feature/tools.py:427 tool_cross_features 的 kind="agg" 暴露）允许按类别列 groupby 聚合另一数值列的 mean/max/min/std/sum/count，是类别列进入数值特征的唯一窄通道——但它是全帧拟合（无 train-only 纪律，若 value 传 target 即泄漏式 target encoding）、非类别 WOE、无罕见类归并，且不在默认候选流程内，不改变"类别特征全链路缺位"的结论。


#### PREP-4 · 哨兵值/特殊值（-999/-1/9999 类）无任何识别与处理机制，会污染填充/标准化/截断/分箱

**KS 影响：Medium · 工作量：M · — 未独立验证（中低影响）**

**缺口**：信贷数据（尤其征信/三方分）大量用 -999/-1/9999 表示"查无此人/未覆盖/超限"，行业极致做法是每特征维护 special_values 清单（自动检测极端值处的频次尖峰+人工确认），特殊值当缺失处理或单独成箱。现在：mean/median 填充和 zscore 的 mean/std 被 -999 严重拉偏；cap_outliers 的 IQR/分位边界被污染；equal_frequency/chimerge 分箱中 -999 混进最左箱与真实低值同箱，扭曲该箱 WOE 和单调合并方向。对 LR/MLP 路径伤害最大，WOE 路径中度受损，树模型能自行切开受损最小。

**证据**：全仓 grep -i "sentinel|special_value|-999|9999|特殊值|哨兵"（marvis/ --include=*.py）仅命中：validation/stress_test.py:24 的压测注入常量 STRESS_MISSING_VALUE=-9999、scorecard.py:229 的 bin_index=-999 占位、report 文案两处——没有任何检测/处理代码。marvis/data/profiler.py:13-21 列画像只出 dtype/null_rate/cardinality/样本值，无 min/max/极值频次尖峰统计；transform.py:66-117 impute/cap 对 -999 一视同仁当真值。

**改法**：1) profiler/字段画像增加数值列的 min/max/top-5 高频值统计，检测"极端位置+高频尖峰"模式自动提示疑似特殊值；2) 给 impute/cap/normalize/bin/woe 工具统一加 special_values 参数：这些值先置 NaN（或在 WOE 中强制单独箱）再拟合；3) 走一次强制确认门（与 NaN 标签门同风格）：检测到疑似哨兵值未声明处理策略时提示用户确认，而非静默当真值。


#### PREP-6 · LR 配方裸训：无填充/标准化/截断（对比 MLP 有 Pipeline、scorecard 有 WOE），含 NaN 即崩、未标准化下 L2 正则失衡；全平台也无 log/偏态变换

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业做法：LR 要么全链路 WOE 化（scorecard 已覆盖），要么 impute+winsorize+standardize 后再进 L2 LR。现在裸 LR 路径：特征含 NaN 直接 sklearn 报错（agent 被迫用全量拟合的 impute 工具救场，又踩 PREP-1/2）；未标准化时 L2 惩罚被大尺度特征（如金额 vs 比率）主导，系数排序力打折；收入/金额类右偏特征无 log 变换可用。影响限于 LR/MLP 家族（MLP 缺 capping，重尾极值拉偏 StandardScaler 的 mean/std），树模型不受影响。

**证据**：marvis/packs/modeling/recipes/lr.py:35-40 LogisticRegression 直接 fit(train[features])，无 SimpleImputer/StandardScaler（对比 mlp.py:42-46 的三段 Pipeline）；recipes/__init__.py:57-66 lr 默认参数只有 max_iter/solver，sklearn LR 默认 penalty=l2、C=1。全仓 grep "log1p|boxcox|yeo|skew"（marvis/ --include=*.py）零命中——无任何偏态变换工具。

**改法**：1) lr 配方仿照 mlp 包成 Pipeline([impute(median), scale(standard), lr])——顺带解决其 artifact 打分一致性（pipeline 整体落盘）；2) 或最少：训练前检测 NaN 给出"建议走 scorecard 或先 WOE 化"的 typed 提示；3) mlp 的 Pipeline 在 scale 前插入分位截断步（sklearn 无内置，可用 FunctionTransformer+落盘边界或 RobustScaler 替代 StandardScaler）；4) FEATURE 包补 log1p/yeo-johnson 变换工具（无状态或参数落盘）。


#### PREP-7 · 日期列信息全部丢弃：识别出 date 角色后没有任何派生路径（无 datediff/账龄/间隔/近期性/趋势派生）

**KS 影响：Medium · 工作量：M · — 未独立验证（中低影响）**

**缺口**：行业极致的信贷特征工程里，日期差类派生常在 IV 排行前列：申请日-开户日（账龄）、申请日-最近逾期日（近期性）、两次申请间隔、按月聚合的近 3/6/12 月趋势斜率。现在这些只能依赖用户在入库前自己算好数值列，平台内日期列等于死重——同样本下可达 KS 上限被压低（幅度取决于数据里日期字段的多少）。

**证据**：marvis/data/schema_infer.py:114 识别 date 角色、data_ops/tools.py:359 有 to_datetime 清洗，但之后无消费者；marvis/feature/derive.py:14-15 ALLOWED_CROSS_OPS={add,sub,mul,div,ratio}、ALLOWED_AGGS={mean,max,min,std,sum,count}，derive_batch 只支持 kind∈{cross,agg,ratio}，无任何日期运算；candidates.py:10-14 META_TOKENS 把 date/time/month/day/dt/ts 命名列从候选特征中全部排除。全仓 grep -i "datediff|date_diff|days_since" 无特征派生实现（只有 vintage 分析的 MOB 消费）。

**改法**：1) derive_batch 增加 kind=datediff：{a: date_col, b: date_col|anchor_date, unit: days|months}，产出数值列并进特征字典；2) 增加 kind=recency（参考日-最近事件日）与 kind=trend（按时间列聚合某数值列的近 N 期斜率/环比）；3) candidate_numeric_features 对派生出的数值列正常纳入（命名避开 META_TOKENS，如 age_days_x）；4) 与 cross 推荐一样让 LLM 基于字典推荐日期对，平台算指标。


#### PREP-8 · 缺失指示变量（missing indicator）全平台缺失：填充后缺失信号丢失，"缺失即信息"只在 WOE 链路成立

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：征信/三方数据的缺失高度 MNAR（查无此人≈高风险或低风险人群系统性缺失），行业做法是填充的同时生成 col_missing 0/1 指示列，或至少给 LR/MLP 路径保留缺失信号。现在 WOE 路径有 NA 单独箱（做对了），但 raw LR（手动 impute 后）和 MLP（配方内静默 median 填充）把缺失信息完全抹掉——这两条路径上缺失率高且缺失有信号的特征会损失真实排序力。

**证据**：marvis/feature/transform.py:66-87 impute_missing 只有 mean/median/mode/constant 四种策略、无 add_indicator 选项；mlp.py:43 SimpleImputer(strategy="median") 未开 add_indicator；全仓 grep "missing_indicator|add_indicator|isna_flag"（marvis/ --include=*.py）零命中；derive.py 也无 kind=isna 之类可以手工造指示列的工具。

**改法**：1) impute_missing 工具加 add_indicator: bool，为每列追加 {col}__isna 0/1 列并回显在 new_columns；2) mlp 配方的 SimpleImputer 开 add_indicator=True（sklearn 原生支持，Pipeline 序列化自动带上）；3) 指示列纳入特征字典并声明来源，防止后续筛选误删。


#### PREP-5 · 数值编码类别（邮编/行业代码/区域码）被当连续值：>90% 可数值解析即判 numeric 进入连续分箱

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业做法是对整数编码列做"名义变量"启发式识别（列名含 code/zip/邮编、整数+高基数+取值不连续、位数固定等），转类别处理（类别 WOE/target encoding/分组归并）。现在 6 位邮编、4 位行业码会进 equal_frequency/chimerge 的区间分箱——编码的数字顺序无业务含义，区间箱把无关类别捆在一起，WOE 单调合并进一步破坏信号；树模型可以反复切分部分自救，评分卡路径损失更明显。整体影响中等偏低，但属于"静默错误处理"而非报错。

**证据**：marvis/data/fingerprint.py:73-74 _frac_numeric>0.9 → value_kind="numeric"；schema_infer.py:124 非特殊角色时 numeric→semantic_role="numeric"；candidates.py:10-14 META_TOKENS 只挡 id/uid/phone/date 等命名，不含 zip/code/postal/邮编/行业/区县类词；readiness.py:117-118 高基数警告只对 semantic_role=="categorical" 生效，数值化的编码列不触发。

**改法**：1) fingerprint/schema_infer 增加名义编码启发式：整数、固定位数、基数高于阈值、列名命中 code/zip/区划 词表 → semantic_role=categorical_code；2) 此类列走 PREP-3 的类别 WOE/归并路径而非连续分箱；3) 在字段画像/readiness 中对疑似编码列给 warn，让用户确认按类别还是连续处理。


#### PREP-9 · 分箱缺最小箱占比约束：chimerge/equal_frequency 可留极小箱，小箱 WOE 时点间不稳

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业评分卡惯例是每箱最小占比 5%（且好坏样本各有下限），保证 WOE 估计稳定和上线后 PSI 可控。现在 chimerge 出的箱如果统计显著即可幸存，极小箱在 train 上 IV 好看、OOT 上 WOE 漂移，轻微侵蚀稳定性；有单调合并兜底时影响进一步缩小，故评 low。

**证据**：marvis/feature/binning.py:51-86 chimerge_edges 只按卡方/p 值合并（min_pvalue=0.05），无 min_bin_pct/min_samples 参数；equal_frequency_edges(8-24 行) 分位去重后可能留下占比极小的箱；只有 tree_edges(144-171 行) 有 min_samples_leaf=0.05。iv.py 的 0.5 平滑只缓解零频不稳，不解决 1-2% 小箱的跨期波动。

**改法**：1) chimerge_edges/equal_frequency_edges 增加 min_bin_pct（默认 0.05）后处理：占比不足的箱与卡方最小的相邻箱合并；2) bin_feature/woe_encode 工具透出该参数并在结果里报告各箱占比，占比 <5% 的箱标黄提示。


#### PREP-10 · aggregate_feature 组统计在全量数据（含 test/OOT 行）上池化计算，且无最小组样本约束

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业做法是组级聚合统计（如 城市均收入）在 train 上拟合成映射、test/OOT 查表，未见组回退全局值；且小组（<N 样本）归并防过拟合。现在聚合值混入了 test/OOT 行的分布信息（非标签，泄漏温和），单样本组的聚合特征等价于把该行自身值复制一份，轻微过拟合倾向。因不涉及标签，对 KS 的真实影响有限，评 low。

**证据**：marvis/feature/derive.py:65-88 aggregate_feature 直接 df.groupby(group_col)[value_col].agg(...) 后 merge 回原表——df 是 FEATURE 阶段的全量帧（此时通常尚未切分）；无 fit/transform 分离，无 min_group_size 参数，单行组的 mean=自身值。

**改法**：1) aggregate_feature 增加 fit_mask/split 参数：groupby 只在 train 行上算，映射 merge 到全表，未见组填全局统计；2) 加 min_group_size（默认 30），不足的组统一用全局值；3) 组映射作为变换 spec 落盘（与 PREP-2 联动）。


### 特征筛选深度

**总体判断**：特征筛选这一块的"底座卫生"已经相当扎实：泄漏感知的单变量 screen（排除 OOT、KS 硬门 + 模型输出名软标记）、完整的单调分箱/WOE/IV 链、WOE 空间的 IV+相关+VIF+系数符号检查、以及就绪度层的重复列/近似标签检测都到位了。但离"极致"还差一个关键台阶：V2 默认对话式建模流里根本没有多变量精筛环节（screen 把所有干净列全量放进模型，无 IV 底线、无相关性去冗余、无模型迭代剪枝），而带精筛的 select_features 又默认在含 test+OOT 的全量数据上算统计量；类别型特征被静默排除在候选之外；泄漏检测停留在"合并样本单变量 KS 一条线"，无按 split 突变/时间维度检测。这些缺口对最终 OOT KS 的影响估计在 0.5~2 个点量级（数据含强类别变量或宽表噪声列多时更大），且对稳定性影响明显。补齐 FS-1/FS-2/FS-3 后，"同样本同标签 KS 封顶"的判断在这个平台上才基本成立。

**已做到位（done_well，经核实）**：

- 泄漏感知单变量筛选且严格排除 OOT：screen_features 用 _dev_mask 把 holdout(默认 oot)排出筛选统计，单列 KS>=0.40 硬性标记疑似泄漏、pred/score/pmml 命名正则软标记模型输出列，交用户确认而非静默删除 (marvis/feature/screen.py:53-65, 121-126, 30-33)
- 缺失率/常量列剔除阈值可配（max_missing_rate、min_unique、leakage_ks、top_k 均为参数）(marvis/feature/screen.py:68-90)
- 分箱工具链完整：等频/等宽/手工/卡方 ChiMerge(带 0.5 平滑的 chi2 合并 + p 值二阶段)/决策树分箱，且有自动方向判定的单调分箱合并后处理 (marvis/feature/binning.py:51-171, 105-141)
- WOE/IV 全平台单一口径 + Laplace 平滑 + 缺失单独成箱(na_as_bin)并给缺失箱独立 WOE，woe_encode 对 NaN 回填 na_woe (marvis/feature/iv.py:10-92, marvis/feature/encode.py:41-54)
- 评分卡训练 WOE 严格 train-only 拟合后再 transform test/OOT (marvis/packs/modeling/recipes/scorecard.py:39-53, 114-147)；tool_woe_encode 的拟合帧默认排除 holdout('oot') (marvis/packs/feature/tools.py:338-347)
- WOE 空间多变量筛选一条龙：chimerge+单调分箱 -> IV 底线 -> WOE 相关性去冗余(保高 IV) -> WOE VIF -> top_k，并附 LR 系数符号检查(WOE 系数为正即告警) (marvis/packs/modeling/select.py:121-175, 238-268)
- select_features 已接 NaN 标签强制确认门 (marvis/packs/modeling/select.py:54-56)
- 就绪度层有独立泄漏防线：重复列检测(block) + 与目标 |corr|>0.95 检测(block) + 高基数类别告警 (marvis/packs/modeling/readiness.py:131-181, 12-15)
- 多变量 gain importance 可选指标：单个 seed-pinned、确定性 LightGBM，内部 7:3 分层切分，补充单变量 IV/KS 排名 (marvis/feature/importance.py:23-76)
- 风险方向感知的头尾 lift、加权 KS/AUC/PSI 等辅助指标齐备且确定性实现 (marvis/feature/metrics.py:109-145, 30-89)
- KS 计算正确处理并列值(change_points)，相关性计算 pairwise 剔 NaN，VIF 对全 NaN 列有防护并 cap 到有限值 (marvis/feature/metrics.py:13-27, marvis/feature/correlation.py:34-61)
- lgb 配方支持单调约束(monotone_constraints 归一化后传入) (marvis/packs/modeling/recipes/lgb.py:39-43)
- 衍生特征防重名/防行数膨胀断言 (marvis/feature/derive.py:61, 84-88)

**缺口**：


#### FS-1 · V2 默认建模流没有任何多变量精筛环节：screen 把全部干净列直通模型，无 IV 底线/去冗余/迭代剪枝

**KS 影响：High · 工作量：M · ✅ 对抗验证 CONFIRMED**

**缺口**：行业极致做法是多级漏斗：单变量底线(IV/KS) -> 相关性聚类去冗余 -> 模型迭代精筛（零增益剪枝/null importance/逐步剔除后重训验证）-> 稳定性复核，最终收敛到几十个精选特征。现状：V2 对话式主流程只有 sanity 级 screen（缺失率/常量/泄漏 KS 门），凡是'干净'的列全部进入 tune/train——宽表几百上千列噪声特征直通 LightGBM（默认 num_boost_round=20），稀释分裂增益、放大过拟合与 OOT 不稳定。带 IV/corr/VIF 的 select_features 只挂在 legacy standard_modeling 模板里，主推的 MODELING/MODELING_WITH_JOIN 流程完全不经过它。

**证据**：marvis/orchestrator/templates/sample.py:402-421（MODELING 模板"特征筛选"步骤只调 modeling.screen_features，不传 top_k，且全流程无 select_features 步骤）；marvis/feature/screen.py:130（top_k=None -> selected=全部 clean 列）；后续 调参/训练 直接引用 $ref:特征筛选.output.selected (sample.py:445, 477)

**改法**：1) 在 MODELING 模板"特征筛选"与"配置调参"之间加一步多变量精筛：先 IV/KS 底线(iv_min 可配)，再相关性去冗余（建议按 |corr| 降序贪心、保留高 IV/KS 者，并暴露 spearman 选项），可选 VIF；2) 增加模型迭代剪枝工具：用初训 LGB 的 gain==0 列剔除 + null-importance（打乱标签的 importance 分布做显著性）二选一，重训对比 test KS 决定收敛；3) screen 的 top_k 在模板中给一个宽松默认（如 200）作为兜底护栏；4) 报告中记录每级漏斗的进/出特征数与理由。

**验证说明**：断言全部核实成立。(1) V2 模板 MODELING（sample.py:325 起，id="modeling"）与 MODELING_WITH_JOIN（sample.py:559 起）的"特征筛选"步骤均为 ToolRef("modeling","screen_features")（sample.py:404、662），两模板步骤列表中不存在 ToolRef("modeling","select_features")——该工具全仓模板层唯一引用在 legacy STANDARD_MODELING（sample.py:166）。(2) 两处 inputs_template 均不含 top_k（sample.py:405-414、663-672 只有 leakage_ks=0.4 与 max_missing_rate=0.95 两个门），模板 slots 也无 top_k 通道；虽然 tool_screen_features 包装器支持 top_k（packs/modeling/tools.py:458），默认流程从不传。(3) marvis/feature/screen.py:130 确认 top_k=None 时 selected=全部 clean 列；该 screen 仅做缺失率/常量/单变量泄漏 KS 三重 sanity 过滤，IV 只作为展示性 enrichment（screen.py:132-146）不做筛选。(4) 下游调参/训练直接消费 $ref:特征筛选.output.selected（sample.py:445、477；WITH_JOIN 版 701 等）。(5) 全仓负向 grep（null_importance/permutation/boruta/RFE/stepwise/forward-backward selection 及同义词）确认无任何迭代精筛实现；各 recipe 的 feature_importance 仅用于报告展示（tools.py:3860+），从不反馈剪枝。(6) 带 IV 底线(0.02)/相关性/VIF(10.0) 的 select_features 实现存在于 packs/modeling/select.py，但只挂在 legacy standard_modeling；correlation.py 的 VIF 另只用于特征分析报告 sheet。(7) lgb 默认 num_boost_round=20 属实（recipes/lgb.py:44）。唯一细微缓解是 G1 人工确认门允许用户手动增删 screen 结果（agent/gate_response_adapter.py:48 screen_adjust），属人工干预而非自动化多变量漏斗，不构成兜底，不影响判定。

**验证员核实证据**：marvis/orchestrator/templates/sample.py:402-421（MODELING 模板"特征筛选"步骤：ToolRef("modeling","screen_features")，inputs_template 无 top_k，仅 leakage_ks=0.4/max_missing_rate=0.95）；sample.py:660-677（MODELING_WITH_JOIN 同构步骤，同样无 top_k）；sample.py:166（select_features 全仓模板层唯一引用，在 legacy STANDARD_MODELING）；marvis/feature/screen.py:130（selected = clean[:top_k] if top_k else clean，top_k=None → 全部干净列直通）及 screen.py:104-126（筛选逻辑仅为缺失率≥0.95/唯一值<2/单变量KS≥0.4 三个 sanity 门，IV 仅作展示 enrichment 于 132-146 行）；sample.py:445、477（调参/训练直接引用 $ref:特征筛选.output.selected；WITH_JOIN 版对应 701 行等）；marvis/packs/modeling/tools.py:458（tool_screen_features 包装器支持 top_k 但无模板/slot 传入）；marvis/packs/modeling/select.py:33-36（iv_min=0.02/vif_max=10.0/相关性去冗余的 select_features 实现，仅被 standard_modeling 使用）；marvis/packs/modeling/recipes/lgb.py:44（num_boost_round 默认 20）；全仓负向 grep（marvis/ 下 *.py，关键词 null_importance/permutation/boruta/RFE/recursive feature/stepwise/forward select/backward elim）无任何迭代精筛实现，recipe feature_importance（tools.py:3860-3876）仅用于报告，不反馈剪枝。补充：G1 门允许人工 screen_adjust（marvis/agent/gate_response_adapter.py:48），属人工干预非自动化兜底。


#### FS-2 · select_features 的 IV/相关/VIF/WOE 拟合默认在含 test+OOT 的全量数据上计算——筛选统计量偷看样本外

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业极致做法是所有监督性筛选统计（IV、单变量 KS、目标相关的分箱）只在 train（至多 train+valid）上拟合，OOT 绝不参与特征取舍——同库的 screen_features 和 tool_woe_encode 都遵守了这一契约，唯独 select_features 例外。现状默认口径下：选择偏差使 OOT 评估被污染（入选特征在 OOT 上'天然'表现好），WOE 空间还把 OOT 标签直接喂进了卡方分箱。

**证据**：marvis/packs/modeling/select.py:48-53（read_frame 全量，仅当 split_col 与 split_value 同时给出才过滤单一 split）；tool_select_features 的 split_value 为可选输入且无默认 (marvis/packs/modeling/tools.py:418-419)；legacy standard_modeling 模板"筛选特征"步骤只传 dataset_id/features/target_col (marvis/orchestrator/templates/sample.py:164-174)；WOE 空间下 chimerge/WOE/KS 全部在该全量帧上拟合 (select.py:137-165)

**改法**：1) 给 select_features 增加与 screen_features 一致的 holdout_values 语义：默认排除 split_col 中的 oot（以及可选 test），而不是要求调用方显式传 split_value；2) standard_modeling 模板把 split_col/split_value(train) 接入"筛选特征"步骤；3) 在返回的 scores 里标注统计口径（fit_rows/split），报告可审计。


#### FS-3 · 类别型（字符串）特征被静默排除在建模候选之外，且全仓无防泄漏的类别编码方案（train-only 拟合/平滑/罕见类归并）

**KS 影响：Medium · 工作量：M · — 未独立验证（中低影响）**

**缺口**：行业极致做法：类别变量（渠道/省份/职业/设备等在信贷场景常有 2-5 个点 KS 贡献）要么走带平滑与罕见类归并的类别 WOE/target encoding（train-only 拟合 + out-of-fold），要么原生喂给 LGB categorical_feature。现状：数值推断静默丢弃所有字符串列，用户不会收到'这些列没用上'的提示；即便手工先 onehot，也没有罕见类归并（onehot 超 50 类直接报错）与类别 WOE；woe_encode 只支持数值分箱。

**证据**：marvis/feature/candidates.py:32（candidate_numeric_features 只取 select_dtypes("number")，字符串列直接不进候选）；prepare_modeling_frame 用同一函数推断 (marvis/packs/modeling/prepare.py:42-47)；MODELING 模板无任何编码步骤（sample.py:362-557）；lgb 配方不设 categorical_feature (marvis/packs/modeling/recipes/lgb.py:48-60)；全仓 grep "target_encod"、"rare"（罕见类归并）无实现，仅有 onehot_encode/label_encode/数值 woe_encode (marvis/feature/encode.py)

**改法**：1) candidate 推断时把被排除的类别列作为 excluded_categorical 显式回报给用户/agent，而非静默丢弃；2) 增加类别 WOE 编码：按类别聚合 good/bad + Laplace 平滑 + 频次低于阈值归并为 __rare__，拟合帧沿用 _woe_fit_frame 的排除 holdout 契约；3) lgb/catboost 路径打通原生类别特征（category dtype + categorical_feature）。


#### FS-4 · 泄漏检测只有'合并样本单变量强度'一条线：无按 split 的 KS/AUC 突变检测、无时间维度审计，条件性/部分泄漏（KS 0.30-0.40）静默通过

**KS 影响：Medium · 工作量：M · — 未独立验证（中低影响）**

**缺口**：行业极致做法在单变量强度之外还有：a) 按 split/月份分别算单变量 KS/AUC，识别'训练期弱、近期异常强'的迁移型泄漏（字段随时间被回填/口径变更）；b) 与 y 同源字段识别（对 target 的非线性近似复制，如分桶后的标签，Pearson corr 检不出而 KS 也可能落在 0.35）；c) 表现期字段的时间审计（特征时间戳 vs 观察点）。现状 0.40 的 pooled KS 门对'只在部分子群/时段泄漏'的字段无感，泄漏字段稀释后 KS 0.30-0.38 会作为高排名特征直接入选。

**证据**：现有三道防线：命名正则 (marvis/feature/screen.py:30-33)、dev 行合并单变量 KS>=0.40 硬门 (screen.py:121-122)、就绪度 |corr|>0.95 (marvis/packs/modeling/readiness.py:155-181, 阈值 L15)。搜索过 marvis/ 全仓关键词 time_travel/时间穿越/temporal/observation、以及 screen/readiness/report_compute 中按 split 分别计算特征 KS 的代码——均不存在；feature_ks 只在 pooled dev 行上算一次 (screen.py:119)

**改法**：1) screen_features 对每个候选补充按 split（train/test，各月可选）的 KS 明细，任一子样本 KS>=leakage_ks 或跨 split KS 差>0.15 即升级为 suspected；2) 加与目标的秩相关/单调近似检测（Spearman 或 2 箱 AUC 接近 1）补 corr 的非线性盲区；3) 若样本表带日期列（META_TOKENS 已能识别 date/month），提供'按月单变量 KS 曲线'诊断供确认门展示。


#### FS-5 · 特征衍生工作流的最终筛选步骤筛的是原始数据集+原始特征列——刚衍生出来的新列从未进入筛选

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：衍生->评估->筛选的闭环在最后一步断了：衍生列既没有和基础列合并参与泄漏/冗余筛选，筛选产出的 selected 也不含任何衍生列——下游拿到的'筛选后特征集'等于白做了衍生。行业做法是基础+衍生合并后统一过泄漏门与冗余筛（衍生的比值/交叉列与父列高度相关，恰恰最需要去冗余），且两两比值类衍生列若父列泄漏会继承泄漏。

**证据**：marvis/orchestrator/templates/sample.py:973-986：FEATURE_DERIVATION 模板"特征筛选"步骤 inputs_template 为 dataset_id="{slot:dataset_id}"（原始数据集而非 $ref:衍生特征.output.result_dataset_id）、features="{slot:feature_cols}"（基础列而非 new_columns），而上一步"分析衍生特征"明明已经拿到了衍生数据集和新列 (L961-972)

**改法**：把该步骤改为 dataset_id=$ref:衍生特征.output.result_dataset_id、features=基础列+$ref:衍生特征.output.new_columns 的并集；同时 evaluate_crosses/分析衍生特征 的指标计算建议沿用 screen 的 holdout 排除契约（当前 feature_metrics 在全量帧上算, marvis/feature/derive.py:142-167）。


#### FS-6 · 无按 split 的特征区分力衰减筛选：没有任何工具计算单特征 train vs test/OOT 的 KS 衰减（与 DOM-7 的无标签 PSI 缺口互补）

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：DOM-7 记录的是分布迁移（PSI/CSI，不用标签）；这里缺的是有标签的效果衰减维度：单特征在 train KS 0.25 / OOT KS 0.08 这类'训练期有效、样本外失效'的特征（数据源覆盖变化、口径漂移的典型症状），现有筛选完全识别不了，只能等模型整体 OOT KS 掉下来再倒查。行业极致做法把'跨期 KS 保持率'作为与 IV 并列的准入指标（如 OOT KS/train KS >= 0.6）。

**证据**：grep 全仓 decay/衰减 仅命中 agent 提示词与场景备注 (marvis/agent/prompts.py:7, marvis/packs/modeling/scenarios.py:33)；feature_ks 的调用点只有 screen（pooled dev 行, marvis/feature/screen.py:119）、select（全量帧, select.py:161）、report_compute（模型分数而非特征, report_compute.py:242）——没有任何按 split 分组的特征级 KS 对比

**改法**：在 screen_features 的 scores 里对每个入选特征补 ks_train/ks_test/ks_oot（OOT 有标签时）与保持率，跨期衰减超阈值的列降入 suspected 交确认门；确认门 UI 已存在（G1 特征确认），只需扩展展示列。注意与 DOM-7 的 PSI 修复合并实现可共享分箱。


#### FS-7 · 筛选排名用的 KS 完全忽略缺失行，'缺失本身有信息'的特征被系统性低估

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：征信查询类字段'无记录'（缺失）往往与风险强相关。行业做法是排名指标把缺失作为独立箱参与（IV 的 na_as_bin 已支持，但只用于事后 enrich 而非排名），或用'缺失指示器+KS'的组合口径。现状：一个 60% 缺失但缺失即高风险的特征，其 KS 只在 40% 非缺失行上计算，排名/top_k 截断时被低估。LGB 训练时原生用 NaN，所以主要影响发生在 top_k 截断与用户按 ranked 做取舍的环节。

**证据**：marvis/feature/metrics.py:258-268（_finite_binary_pairs 直接丢掉特征为 NaN 的行）；screen 按此 KS 排名并做 top_k 截断 (marvis/feature/screen.py:119, 128-130)；带 NA 箱的 IV 只对已入选特征事后补算 (screen.py:132-146)

**改法**：screen 的 ranked 增加一列带 NA 箱的 IV（把现在的 enrich 提前到全量候选，成本可控：已按 batch 读列），或提供 rank_by=iv_with_na 选项；文档标注 ks 列的'仅非缺失行'口径。


#### FS-8 · VIF 在高缺失数据上静默失效：listwise dropna 清空后全部返回 0，VIF 门变成空转且无警告

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：信贷宽表几十个特征各缺 20-30% 时，完整行交集经常为空或只剩几十行——此时 VIF 要么全 0（门失效）要么在极小样本上估计（不稳定），调用方完全无感。行业做法：成对删除或先插补再算 VIF，样本不足时显式报'VIF 不可用'。raw 空间受影响最大；WOE 空间因 na_woe 回填无缺失，不受影响。

**证据**：marvis/feature/correlation.py:54-56（clean = df[usable].dropna(); clean.empty 时直接返回全 0 的 result）；_drop_high_vif 拿到全 0 自然一个都不剔 (marvis/packs/modeling/select.py:201-219)，无任何 warning 字段

**改法**：clean 行数低于阈值（如 max(200, 20*特征数)）时在 SelectionResult.warnings 里报'VIF 基于 N 行完整样本/已跳过'，并提供 median 插补后计算的选项。


#### FS-9 · IV 的分箱口径跨工具不一致（等频10箱 vs 卡方6箱）且无分箱数敏感性检查，同一特征在不同路径的 IV 结论可能互相矛盾

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：IV 是分箱依赖的统计量：等频 10 箱与单调卡方 6 箱对同一特征差 30-50% 很常见，0.02 的底线在两个口径下含义不同；行业极致做法会固定单一'筛选口径'（通常最优/单调分箱）并对边界特征做分箱数敏感性复核（bins∈{5,10,20} IV 稳定才保留）。现状影响主要在 iv_min 边界特征的取舍一致性，非系统性偏差。

**证据**：screen 的 IV 与 raw-space select 用 feature_metrics -> equal_frequency_edges(values, 10) (marvis/feature/metrics.py:211, marvis/packs/modeling/select.py:104-111)；woe-space select 用 chimerge_edges(max_bins=6)+单调合并 (select.py:145-155)；iv_min=0.02 底线对两种口径同值套用 (tools.py:411)；全仓无分箱数敏感性/IV 稳定性检查

**改法**：1) 文档与 scores 里标注 IV 口径（method+bins）；2) iv_min 按口径分别给默认（等频10箱可略高）；3) 可选提供 iv_sensitivity 诊断（多分箱数 IV 极差/均值），供确认门展示边界特征。


#### FS-10 · 非二分类目标的筛选没有任何排序力指标：top_k 按输入列顺序截断，等于随机选特征

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：回归/多分类场景（收入预测 income 场景已在 scenarios.py 注册）下，行业做法至少用 |Spearman| 或分箱后目标均值方差比做单变量排序；现状只剔常量/高缺失列，若用户传 top_k 则前 k 个'碰巧排前面'的列入选。主镜头是二分类，故影响评级低，但 income 场景一旦启用 top_k 就是实际伤害。

**证据**：marvis/feature/screen.py:186-206（screen_features_non_binary 的 clean 全部 ks=None、不排序，selected=clean[:top_k] 即按 features 入参顺序切片）；无 Spearman/互信息等回归目标单变量指标（grep mutual_info/spearman 仅 correlation.py 的特征间相关）

**改法**：非二分类路径按 |Spearman(feature, target)| 排序 ranked 并作为 top_k 依据；多分类可用一对多 AUC 的最大值。确定性、无额外依赖。


#### FS-11 · 派生算子缺趋势/变换类，且 aggregate_feature 允许 value_col=目标列——无护栏的组内目标编码泄漏通道

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：a) 覆盖面：行业常用的 log/幂变换、时间窗口趋势(斜率/环比)类派生不存在——在单样本表锚定的架构下窗口趋势可暂缓，但 log/分位变换成本极低；b) 护栏：recipe 若写 {kind:'agg', group:'channel', value:'<target_col>'} 就是全量数据上的组内目标均值编码（教科书级泄漏），只能靠下游 screen 的 KS>=0.40 门兜底，而组基数大时该特征 KS 可能落在门内。

**证据**：marvis/feature/derive.py:14-15（ALLOWED_CROSS_OPS 仅 add/sub/mul/div/ratio；ALLOWED_AGGS 仅 mean/max/min/std/sum/count）；aggregate_feature (L65-88) 对 value_col 无'不得为目标列'校验，且 groupby 在含 OOT 的全量帧上计算；tool_cross_features 直接透传用户/LLM recipe (marvis/packs/feature/tools.py:427-432)

**改法**：1) derive_batch/aggregate_feature 增加断言：value_col/col_a/col_b 不得等于 target_col（需要把 target_col 传入 tool_cross_features）；2) agg 类派生的 groupby 统计沿用 holdout 排除契约（fit 于 dev 行、transform 全量）；3) 低成本补 log1p/rank 单列变换算子。


### 调参与训练方法学

**总体判断**：调参主线（LGB）的工程骨架是对的：早停+最优轮数回填、过拟合惩罚目标、OOT 不参与选择、seed 贯通、样本权重从调参到指标全链路打通，这些都达到了正规水准。但离"调参到位、模型选择到位"的极致标准还有两个结构性距离：其一，调参器只为 LGB 一家服务——xgb/catboost/lr/scorecard/mlp 全部以硬编码弱默认参数"参赛"（xgb 仅 20 棵树、catboost 仅 50 轮），多算法对比选优实际是"调过参的 LGB vs 没调参的其他"，且 A 卡场景默认主算法 scorecard 全程零调参；其二，默认搜索预算只有 12 轮随机搜索（gate 还劝退 >50），无 CV、无两阶段搜索、无选参后全量重训。这些缺口在典型信贷样本上合计可能吃掉 0.5~2 个 KS 点（视样本量与算法家族匹配度），即：用户"同样本同标签 KS 封顶"的判断本身成立，但当前默认路径并没有把"调参到位"这个前提完全兑现。

**已做到位（done_well，经核实）**：

- 早停与最优轮数回填闭环正确：每个 trial 用 early_stopping(100) 训练至 max 3000 轮，best_iteration 写回 best_params['num_boost_round']（marvis/packs/modeling/tune.py:121-128,155），最终训练直接消费该轮数（marvis/packs/modeling/recipes/lgb.py:44-51），learning_rate 与 n_estimators 联动到位
- 调参目标带过拟合惩罚且 OOT 纪律正确：score = test_ks - 0.5*max(0, train_ks-test_ks)（tune.py:184-190），OOT 指标仅报告、明确不参与超参选择（tune.py:6-8 文档与实现一致），每 trial 还记录 train-oot gap 与 oot 稳定性 gap（tune.py:145-161）
- 搜索空间对信贷数据是内行的：num_leaves 8-63 偏小叶、min_child_samples 50-500、feature_fraction/bagging、lambda_l1/l2、min_gain_to_split、learning_rate 对数均匀 0.01-0.08（tune.py:57-73），偏保守正合信贷小中样本
- 类别不平衡进了搜索空间：scale_pos_weight 候选含数据推导的 neg/pos hint（tune.py:72,106-108），且 'auto' 在训练路径按样本权重口径解析（recipes/common.py:61-77）
- 样本权重全链路贯通：调参时进 lgb.Dataset 的 train 和 valid（tune.py:104-111）并做非负/非空校验（tune.py:193-205）；训练时 lgb/xgb 连 eval_sample_weight 都带上（recipes/lgb.py:52-60、recipes/xgb.py:50-58）；指标层有 weighted KS/AUC/PSI 全套（recipes/common.py:216-267）；setup 阶段自动探测权重候选列并强制移出入模特征（modeling_setup.py:412-465、tools.py:533-535）
- 随机性控制基本到位：空间采样用 seeded RNG（tune.py:112），每 trial 固定 seed+deterministic（tune.py:120），全部配方 pin random_state=config.seed（lgb.py:34-37、xgb.py:34-36、catboost.py:33-35、lr.py:33、mlp.py:38），seed 有统一 fallback 链（tools.py:3657-3662）
- 多配方对比至少在数据层面公平：train_models 对每个 recipe 用同一数据集、同一特征集、同一切分、同一 seed（tools.py:700-715）
- 切分方法学正确：OOT 按时间分位数先切（信贷标准做法）、随机 OOT 需显式声明不伪造（prepare.py:155-173）、按身份列分组防近重复样本跨集泄漏（prepare.py:266-290、modeling_setup.py:502-511）、规则切分引擎支持渠道/时间条件（prepare.py:188-236）
- 单调约束在训练路径有完整契约：dict/list/string 三种形式归一化并校验长度与取值（recipes/common.py:80-115），scorecard 默认单调分箱+WOE 系数符号检查（select.py:238-268）
- 调参过程透明可审：每 trial 持久化 params/train/test/oot KS/AUC/lift@5/10/双 gap（tune.py:154-163），G4 门展示 trials 排行榜（orchestrator/templates/sample.py:524-526）；n_trials 可对话式调整（agent/adjust_specs.py:8-12）并有范围护栏（agent/gate_payloads.py:338-350）
- MLP 配方正确地做了 impute→scale→MLP 管道（recipes/mlp.py:42-46），没有把带缺失的原始特征直接喂神经网络

**缺口**：


#### TUNE-1 · 调参器只为 LGB 一家服务：全部挑战者算法以硬编码弱默认参数参赛，多算法对比选优结构性失真；A 卡场景默认主算法 scorecard 全程零调参

**KS 影响：High · 工作量：L · ✅ 对抗验证 CONFIRMED**

**缺口**：行业极致做法是每个候选算法各自有（哪怕小预算的）调参空间，或至少给挑战者一套经验强默认（xgb/catboost 1000+ 轮+早停+合理 lr），保证'冠军-挑战者'对比是调参后 vs 调参后。现状是调参后的 LGB 对阵近乎裸奔的其他算法：如果某数据集上 XGB/CatBoost 家族本可赢 0.5-1 个 KS，平台永远发现不了；对比实验表还会误导用户得出'xgb 不如 lgb'的无效结论。configure_tuning 自己也承认'暂不执行随机搜索'（tools.py:585）。

**证据**：tools.py:575 `tune_enabled = recipe == "lgb"`；tools.py:598-599 非 lgb 直接返回空 best_params；tools.py:708-709 train_models 只让 lgb 消费调参结果、其余用默认。默认参数极弱：xgb 仅 20 棵树（recipes/xgb.py:43 `num_boost_round` 默认 20，XGB 默认 lr=0.3/depth=6）；catboost 仅 50 轮 lr=0.05 depth=4（recipes/catboost.py:37、recipes/__init__.py:44-56），远低于其正常上千轮的收敛点；scorecard/lr 的 C、max_bins（scorecard.py:43 固定 6）、单调方向均无搜索；mlp 固定 (32,16)。scenarios.py:24-34 A 卡场景 default_recipe='scorecard'，modeling_setup.py:187-189 primary_recipe 取 lgb 不在列时的第一个 → A 卡默认流程零调参。lgb 自身在无调参直训路径也退化为 20 轮（recipes/lgb.py:44）。

**改法**：1) 把 tune.py 抽象为 per-recipe 搜索空间注册表（ModelRecipe.param_space 字段已存在但全部为空 {}，recipes/__init__.py:31,41,54），先补 xgb/catboost 两个空间（关键参数与 lgb 同构：树数上限+早停、depth/leaves、min_child_weight、subsample/colsample、lambda/alpha、scale_pos_weight）；2) scorecard 至少搜 C（对数均匀 0.01-10）与 max_bins（4-10）；3) 短期兜底：把挑战者默认参数改为'强默认'（xgb/catboost num_boost_round 上限 2000+early_stopping_rounds=100，早停轮数回填），train_models 别再传 early_stopping_rounds=None（tools.py:711）；4) A 卡场景 primary_recipe=scorecard 时给出'scorecard 未调参'的明确提示或走其专属搜索。

**验证说明**：断言完全成立，且部分证据比断言更强。(1) 全仓唯一搜索实现是 marvis/packs/modeling/tune.py，且明确是 LightGBM 专用随机搜索（导入 lightgbm、_sample_params 只采样 LGB 参数空间）；对 marvis/ 全量 grep optuna/hyperopt/skopt/bayes/tpe/GridSearchCV/RandomizedSearchCV/ParameterSampler 均无命中，无任何其他模块兜底。(2) 入口硬编码 lgb-only：tools.py:575 `tune_enabled = recipe == "lgb"`；tools.py:598-599 非 lgb 直接返回空 best_params/0 trials；tools.py:585 自认"暂不执行随机搜索,使用算法默认参数"；tools.py:708-709 train_models 只让 lgb 消费调参结果（`if recipe == "lgb" else dict(control_params)`），并有注释直言。编排模板 orchestrator/templates/sample.py:459-461 注释再次确认"only lgb runs the random search; every other recipe (lr/xgb/scorecard/mlp/regressor/multiclass) skips it and trains with its own defaults"。(3) 挑战者默认参数确实极弱：xgb.py:43 默认 20 棵树（且 train_models 路径 tools.py:711 硬编码 early_stopping_rounds=None，连早停都没有，XGB 库默认 lr=0.3/depth=6）；catboost.py:37 默认 50 轮，recipe 默认 lr=0.05/depth=4（recipes/__init__.py:50-51）；scorecard.py:43 max_bins 固定默认 6、C 仅是 config 透传（scorecard.py:150-152）无搜索；mlp 固定 [32,16]（recipes/__init__.py:92）；lgb 无调参直训路径同样退化 20 轮（lgb.py:44）。tune.py 模块 docstring 自己承认 20 轮默认"badly underperformed a hand-tuned reference"。(4) A 卡零调参成立：scenarios.py loan_pre_a default_recipe='scorecard'（tools.py:642-644 apply_scenario 消费），modeling_setup.py:189 primary_recipe 逻辑使 scorecard-only 流程的主算法为 scorecard，configure_tuning 对其完全关闭调参；且门槛是严格 == "lgb"，lgb_regressor/lgb_multiclass（回归/多分类场景默认算法）同样全程零调参，比断言范围更广。(5) 唯一小误差：断言称 param_space "三处注册"，实际 recipes/__init__.py 有 8 处注册（行 31/41/54/64/74/85/97/109），全部为空 dict {}，且该字段在生产代码中无任何消费点（仅 contracts.py:12 定义 + 一个序列化测试），是死字段——实质比断言更强。方法学缺口本身（冠军调参 vs 挑战者裸奔、对比实验结构性失真、A卡主算法零调参）经代码逐行核实为真。

**验证员核实证据**：核实后的证据链（全部为当前工作区真实行号）：[入口硬编码] marvis/packs/modeling/tools.py:575 `tune_enabled = recipe == "lgb"`；tools.py:581 非 lgb n_trials 归零；tools.py:585 reason 文案自认"{recipe} 暂不执行随机搜索,使用算法默认参数"；tools.py:589-599 tool_tune_hyperparameters 对 recipe != "lgb" 直接 `return {"best_params": _jsonable(base_params), "best_metrics": {}, "n_trials": 0, "trials": []}`（注释：xgb tuning is a later slice）；tools.py:708-709 train_models `params={**tuned_params, **control_params} if recipe == "lgb" else dict(control_params)`；tools.py:711 `early_stopping_rounds=None`（多算法对比路径连早停都关闭）。[唯一搜索实现] marvis/packs/modeling/tune.py 全文 LGB 专用（import lightgbm，_sample_params 行 57-73 只采样 LGB 空间，搜索空间 lr~0.01-0.08/max_boost_round 3000/早停 100）；全仓 grep optuna|hyperopt|skopt|bayes|tpe|GridSearchCV|RandomizedSearchCV|ParameterSampler|ParameterGrid 零命中。[死字段] param_space 注册处为 recipes/__init__.py 行 31/41/54/64/74/85/97/109 共 8 处（断言说"三处"不准确），全部 `param_space={}`；字段定义在 contracts.py:12；生产代码无任何读取点。[弱默认] recipes/xgb.py:43 `num_boost_round = int(params.pop("num_boost_round", 20))`（recipe default_params 仅 objective/eval_metric，lr/depth 用 XGB 库默认 0.3/6）；recipes/catboost.py:37 `iterations = int(params.pop("iterations", params.pop("num_boost_round", 50)))`，recipes/__init__.py:50-51 learning_rate=0.05/depth=4；recipes/scorecard.py:43 `max_bins = int(config.params.get("scorecard_max_bins", 6))`，C 仅透传（scorecard.py:150-152 _lr_params）无搜索；recipes/__init__.py:92 mlp hidden_layer_sizes=[32,16]（mlp.py:40 兜底同值）；recipes/lgb.py:44 无调参直训默认 20 轮。tune.py:3-5 docstring 自认历史 20 轮配置"badly underperformed a hand-tuned reference"。[A卡零调参] scenarios.py:23-34 loan_pre_a default_recipe="scorecard"（param_overrides 仅 {"max_depth": 3}）；scenario 仅经 tools.py:642-644 tool_train_model 的 apply_scenario 消费；marvis/agent/modeling_setup.py:187-189 `primary_recipe = "lgb" if "lgb" in recipe_list else recipe_list[0]`（含注释 "The tuner is lgb-specific ... tuning is skipped for it"）→ scorecard 主算法流程调参全关。[范围更广] 门槛严格 == "lgb"，故 income 场景默认 lgb_regressor、多分类 lgb_multiclass 也零调参。[结构确认] orchestrator/templates/sample.py:422-489 流水线为 配置调参→调参→训练模型(train_models)→对比实验，sample.py:459-461 注释明示只有 lgb 跑搜索、其余算法以自身默认参数参加对比——即"调参后 LGB vs 裸默认挑战者"的对比失真由模板层固化。


#### TUNE-2 · 默认搜索预算仅 12 轮纯随机搜索，无 TPE/贝叶斯、无粗搜+细搜两阶段，gate 文案还劝退用户加大预算；空间细节有两处失真（lambda 线性均匀、lr 上限 0.08）

**KS 影响：High · 工作量：M · ✅ 对抗验证 CONFIRMED**

**缺口**：行业极致是 Optuna TPE + median pruner 跑 100-300 trial（有早停时单 trial 便宜），或至少随机搜索 50-100 轮起步；两阶段（粗搜定区域→细搜收敛）是评分卡团队标配。12 轮随机采样在 ~10 维空间的覆盖极其稀疏，选中组合的期望质量对 seed 敏感，典型数据集上相对充分搜索会留下 0.5-1.5 个 KS 点在桌上——这恰是用户'调参到位'前提中最直接没兑现的一环。

**证据**：编排模板把 n_trials 钉死为 12（orchestrator/templates/sample.py:391,649），choose_modeling_spec/configure_tuning 默认也是 12（tools.py:536,570）；工具层默认 40（tools.py:609）被模板覆盖不生效。gate 提示 >50 就警告'增加运行成本'（agent/gate_payloads.py:343-347）。搜索算法是均匀随机采样（tune.py:57-73,112,118-119）；全仓无 optuna/hyperopt/skopt（grep 'optuna|hyperopt|bayes' 于 *.py/*.toml/*.txt 零命中，requirements 无此依赖）。空间细节：lambda_l1/l2 线性均匀 0-20（tune.py:69-70，半数 trial 落在 >10 的重正则区，业界惯例对数均匀 1e-3~10）；learning_rate 上限 ~0.08（tune.py:64）配 max 3000 轮对大样本偏慢、对小样本无 0.1+ 快速档；缺 max_bin/min_sum_hessian_in_leaf。

**改法**：1) 立即项（S）：模板默认 n_trials 提到 48-64（早停下成本可控），lambda 改对数均匀，gate 文案改为鼓励 50-100；2) 中期（M）：引入 Optuna TPE（保 seed 确定性：TPESampler(seed=seed)），加 MedianPruner 用 valid AUC 剪枝；3) 加两阶段模式：前 2/3 预算粗搜全空间，后 1/3 在 top-3 trial 邻域细搜 learning_rate/num_leaves/min_child_samples。

**验证说明**：每一条证据都在真实代码中核实成立。(1) 缺失类断言经多关键词对抗验证:精确复现声称的 grep('optuna|hyperopt|skopt|bayes' 于 *.py/*.toml/*.txt,排除 .venv)零命中;扩展同义词扫描(tpe、gaussian process、hyperband、pruner、successive halving、grid search、surrogate、acquisition、两阶段/粗搜/细搜、GridSearchCV/RandomizedSearch)在 marvis 包内同样零命中;pyproject.toml 依赖仅有 lightgbm/xgboost/catboost/sklearn 等,无任何调参库;全仓唯一调参入口就是 marvis/packs/modeling/tune.py 的 tune_hyperparameters,无其他模块兜底。(2) tune.py:118-119 为 for 循环内每 trial 独立从 seeded RandomState 采样,无任何基于历史 trial 的自适应逻辑、无跨 trial 剪枝(仅 :126 有每 trial 的 LightGBM 早停)。(3) n_trials 钉死 12:sample.py:391 与 :649 两处模板字面量 12,经 配置调参(:429/:685)与 调参(:456/:709)的 $ref 链传递,tools.py:609 的默认 40 因 key 恒被模板提供而永不生效;tools.py:536/:570 默认 `or 12`。(4) gate 文案劝退属实:gate_payloads.py:343-345 对 >50 警告"会显著增加运行成本…AUTO 不应直接放大该预算",:347 的 info 文案也劝阻大幅上调(次要平衡点::340-341 对 <5 有"仅适合烟测"的下限提示)。(5) 空间失真属实:tune.py:69-70 lambda_l1/l2 线性均匀 0-20(半数质量落在 >10 重正则区);:64 learning_rate=10^uniform(-2.0,-1.1) 即 [0.01, ~0.0794],上限 ~0.08,配 max_boost_round=3000(:87);_sample_params 中确无 max_bin 与 min_sum_hessian_in_leaf(仅可经 base_params 固定透传,不参与搜索)。唯一出口是人工:gate 表单可编辑 n_trials(gates/contracts.py:334)、adjust 流支持"n_trials 调到 20"(adjust_specs.py:8-12),但无任何自动扩大预算的机制,且文案方向与断言描述一致。缺口定性(12 轮随机采样在 ~10 维空间覆盖稀疏、对 seed 敏感、无 TPE/两阶段)与代码事实相符。

**验证员核实证据**：编排模板把 n_trials 钉死为字面量 12(marvis/orchestrator/templates/sample.py:391 与 :649,均在"选择建模规格"步),经"配置调参"(:429/:685)与"调参"(:456/:709)的 $ref 链原样传递;choose_modeling_spec/configure_tuning 默认同为 12(marvis/packs/modeling/tools.py:536、:570,写法 `int(inputs.get("n_trials") or 12)`);tool_tune_hyperparameters 的默认 40(tools.py:609,`int(inputs.get("n_trials", 40))`)仅在 n_trials key 缺失时生效,模板运行中 key 恒存在故永不触发。gate 提示:agent/gate_payloads.py:340-341 对 <5 警告"适合快速烟测",:343-345 对 >50 警告"会显著增加运行成本并扩大后续重算范围;AUTO 不应直接放大该预算",:347 常规区间 info 文案亦称"大幅上调会增加运行成本并触发更宽的下游重算"。搜索算法为纯随机采样:tune.py:112 seeded RandomState,:118 `for trial in range(max(1, n_trials))`,:119 每 trial 独立调用 _sample_params(:57-73),无历史感知采样、无跨 trial pruner(仅 :126 每 trial 内 LightGBM early_stopping(100));签名默认 n_trials=40(:84)、max_boost_round=3000(:87)。缺失验证:grep -rniE 'optuna|hyperopt|skopt|bayes' 于 *.py/*.toml/*.txt(排除 .venv)零命中;扩展扫描 tpe/hyperband/pruner/successive halving/gaussian process/grid search/surrogate/两阶段/粗搜/细搜 在 marvis 包内零命中;pyproject.toml(第 9-40 行 dependencies)无任何调参库;tune_hyperparameters 为全仓唯一搜索实现(tools.py:602 唯一调用点)。空间细节:lambda_l1/l2 线性均匀 uniform(0.0, 20.0)(tune.py:69-70,半数采样落在 >10);learning_rate=10**uniform(-2.0,-1.1)≈[0.01, 0.0794](:64);_sample_params 采样空间不含 max_bin 与 min_sum_hessian_in_leaf(仓内 max_bin 相关命中均为分箱/评分卡的 max_bins,与 LightGBM 超参无关)。唯一放大预算途径为人工:gate 可编辑字段 n_trials(agent/gates/contracts.py:334-337)与 adjust 指令(agent/adjust_specs.py:8-12,如"n_trials 调到 20"),无自动扩大机制。


#### TUNE-3 · 调参选择建立在单一 train/test 切分上：无任何交叉验证/重复切分，且早停集、超参选择集、（下游校准集）三者共用同一个 test，选择噪声与乐观偏差无对冲

**KS 影响：Medium · 工作量：M · — 未独立验证（中低影响）**

**缺口**：行业极致做法：在 train 内做 grouped/stratified 5-fold CV（信贷截面数据）或按月 blocked CV（时间序敏感场景），trial 得分用折均 KS±std，test 完全不参与调参只做一次终评；或至少 train/valid/test 三切。现状 12-40 个 trial 都在同一个含噪 test 上比大小：选择本身不稳（换 seed 换切分可能选出不同参数），报告的 test KS 系统性乐观 0.2-0.5 个点（这个乐观偏差还会传导进 DOM-4 已记录的校准问题）。

**证据**：tune.py:101-111 只切一次 train/test，dvalid 既做早停 valid_sets（tune.py:125-126）又做选择指标集（tune.py:132,149-153）。全仓无 CV：grep 'KFold|StratifiedKFold|TimeSeriesSplit|cross_val|cv=' 于 marvis/ 零命中。切分角色只有 train/test/oot 三种（prepare.py:20 VALID_SPLIT_ASSIGNMENTS），没有独立 valid 的概念。test 默认只占 30%（prepare.py:16），几万行样本下单切分 KS 的抽样波动可达 ±1-2 个点。

**改法**：1) tune 内实现折内评估：对 train 做 group-aware（复用 prepare.py 的 group_cols 逻辑）5-fold，trial score = mean(fold_ks) - penalty*std(fold_ks)，早停用各折自己的 valid；test 只在最优参数定型后评一次；2) 样本量小（<2 万）时默认启用 CV、大样本可保留单切分但至少提供开关；3) 有月份列时提供 blocked-by-month CV 选项，与平台'OOT 不参与选择'的既有纪律一致。


#### TUNE-4 · 超参定型后没有'全量在时样本重训'步骤：交付的冠军模型只见过 ~50-70% 的标注样本，test（默认 30%）的信息被永久浪费

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业标准收尾动作：超参（含最优轮数）在 train/test 上定型后，把 train+test 合并、按比例缩放轮数（或用固定轮数）重训一次，OOT 保持纯净做终评；上线工件甚至用全量重训。对几万行的典型信贷样本，多用 43% 的数据重训通常能在 OOT/线上多拿 0.3-1 个 KS 点，样本越小收益越大。现状连可选项都没有。

**证据**：train_models 每个配方只 fit train 切分（recipes/lgb.py:53-60、xgb.py:51-58、catboost.py:40-46、lr.py:36-40、scorecard.py:56-60 均为 fit(train[...])）；全仓无重训逻辑（grep 'refit|retrain|全量重训|full_train' 仅命中两处无关的错误提示文案 tools.py:1230,3368）。默认切分 test=30%、OOT 时间切 20%（prepare.py:16-17），冠军实际训练样本约为全量的 56%。

**改法**：1) 在 select_experiment 之后加可选步骤 `refit_champion`：用冠军的 best_params（num_boost_round 按 (n_train+n_test)/n_train 缩放或直接复用），在 train+test 上重训，产出新 artifact 并只报 OOT 指标；2) 报告中并排展示 refit 前后 OOT KS，让用户确认收益后选用；3) OOT 缺失时（随机切分场景）默认不启用并说明原因，避免无终评集的重训。


#### TUNE-5 · 样本权重在'训练-选择'链路断裂：调参用加权样本训练却用未加权 KS 选参，champion 选择同样只看未加权指标，weighted_* 指标沦为展示品

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业做法：一旦启用业务权重/拒绝推断权重，评估口径必须与训练口径一致——加权训练就应加权选参、加权选冠军，否则优化目标（加权总体）与选择目标（未加权总体）不同，选出的参数对'真实加权总体'不是最优。现状权重越重要（如拒绝推断放大件），错位越大；weighted_* 全套指标算了却不进任何决策。

**证据**：tune.py:110-111 把权重传入 lgb.Dataset（训练与早停都加权），但 trial 评分用 feature_ks(preds, y) 未加权（tune.py:131-132,149-153），全程不调用 weighted_feature_ks（该函数存在于 marvis/feature/metrics 且被 recipes/common.py:235-247 使用）；_pick_best_experiment 只读 oot_ks/test_ks 键（tools.py:789-795），从不读 weighted_oot_ks/weighted_test_ks。

**改法**：1) tune.py 在 weight_col 存在时改用 weighted_feature_ks/weighted_feature_auc 计算 train/test/oot 指标并进 _trial_score；2) _pick_best_experiment 在权重存在时优先读 weighted_oot_ks/weighted_test_ks；3) 在 trials 记录里同时保留两套口径便于审计；4) 顺带审视 scale_pos_weight 采样与样本权重并存时的双重加权问题（tune.py:72，可在有权重时把候选收敛到 {1.0, hint}）。


#### TUNE-6 · 确定性声明与实现不符：tune 文档称'fixed seed + single thread'，实际 num_threads=0（全核并行），与直训路径 n_jobs=1 线程数不一致，跨机/跨路径复现无保障

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：平台有'确定性指标'全局不变量（也是本仓的核心卖点之一）。现状同一 seed 在不同核数机器上、或'调参后重训 vs 直训'两条路径间，可能长出浮点意义上不同的树，KS 第三四位小数漂移，破坏跨环境回归测试与审计复现。对最终 KS 数值影响微小，但属于声明级契约违背。

**证据**：tune.py:11-12 docstring 声称 'LightGBM is trained with fixed seed + single thread'，但 tune.py:120 实际设 `"num_threads": 0`（LightGBM 语义为使用全部核心）；best_params 携带该键流入最终训练，覆盖 recipes/lgb.py:36 的 n_jobs=1（LightGBM 参数解析中 num_threads 主名优先于 n_jobs 别名）；而不经调参的 train_model 直训路径则真是单线程。LightGBM 官方仅保证同数据+同参数（含线程数）+force_row/col_wise 下的确定性，deterministic=True 未配 force_*（tune.py:120、lgb.py:37 均无）。

**改法**：1) 二选一并统一：要么全链路 num_threads=1（可复现优先，调参慢可用 n_trials 换）、要么显式接受多线程并把 docstring 与文档改为'同机同核数可复现'；2) 无论选哪个，补 force_row_wise=true 配合 deterministic=true；3) 加一条回归测试：同 seed 两次 tune_hyperparameters 的 trials 序列逐字段相等。


#### TUNE-7 · 单调约束在调参路径不归一化：dict 形式约束直接透传给 lgb.train 会崩溃，list 形式仅靠位置巧合生效——'带约束调参'与训练路径契约不一致

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：A 卡等强解释场景要求'约束下调参'（约束改变最优 num_leaves/min_child_samples 的位置，无约束调出的参数在约束下未必最优）。现状要么崩（dict）、要么静默依赖特征顺序巧合（list），且调参与训练两条路径对同一输入的解释可能不同——违背'同配置同结果'的平台纪律。

**证据**：训练路径有完整归一化：recipes/common.py:80-115 normalized_monotone_constraints 支持 dict/str/list 三种形式并按 config.features 顺序展开、校验取值。调参路径没有：tune.py:208-217 _lgb_base_params 只剔除权重键，monotone_constraints 以原始形态混入 fixed_params（tools.py:596-597 _training_control_params 会把它从 inputs/params 提进来），tune.py:119 直接进 lgb.train——dict 形式在 LightGBM 参数序列化时抛 TypeError，字符串/列表形式则不校验长度与特征对齐。

**改法**：1) 在 tune_hyperparameters 入口复用 normalized_monotone_constraints（构造临时 TrainConfig 或抽出纯函数），归一化后以 list 形式进 fixed_params；2) 补一条测试：dict 形式约束下 tune 与 train 的约束向量一致；3) 特征筛选步骤改变特征集后（模板中 tune 用的是筛选后特征），确保约束 dict 按新特征集重展开。


#### TUNE-8 · 多算法训练无失败隔离且 lr 配方无缺失值处理：数据含 NaN 时 lr 挑战者必崩并拖垮整个 train_models 步骤

**KS 影响：Low · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业做法：线性类配方要么走 WOE（scorecard 已如此）要么内置 impute+scale 管道；多算法对比要有 per-recipe 失败隔离（失败者记 failed 并继续，最后从成活者里选优）。现状用户勾选 lr 做挑战者时，大概率整个'训练模型'步骤报错，只能手动剔除 lr 重跑——多算法对比的可用性被单点脆弱性拖累。

**证据**：recipes/lr.py:35-40 用 sklearn LogisticRegression 直接 fit 原始特征，无 SimpleImputer/StandardScaler（对比 mlp.py:42-46 有完整 impute→scale 管道；grep 'impute' 在 recipes/ 下仅命中 mlp.py）；sklearn LR 遇 NaN 抛 ValueError。tools.py:719-733 train_models 循环中任一配方异常直接 `raise`，前面已训成功的实验虽已入库但整个工具调用失败、计划步骤中断。信贷特征常态性含缺失，lgb/xgb/catboost 原生兜底而 lr 必崩。

**改法**：1) lr 配方加 Pipeline(SimpleImputer(median) → StandardScaler → LR)，与 mlp 同构（顺带让 C 的正则几何有意义）；2) train_models 把逐配方 try/except 改为记录失败原因后 continue，全部失败才抛错，返回值中带 failed 列表供 compare/report 展示；3) 补测试：含 NaN 特征 + recipes=[lgb,lr] 时 lgb 实验存活且被选为冠军。


### 模型选择与评估口径

**总体判断**：评估口径的"地基"质量很高：KS/AUC 是逐样本精确计算且 tie 处理正确、加权指标齐全、切分器具备规则/分组防泄漏/时间 OOT 能力、调参明确不用 OOT 选择，这些都达到或超过行业常规水平。但离"极致"有一段清晰距离，且缺口集中在默认路径：默认切分是随机 75/25 且完全不建 OOT（时间外推 OOT 是从未被调用的死代码）；多算法对比只有 lgb 被调参，xgb/catboost 用 20 棵树/50 次迭代的裸默认参数参赛，比较结论被预先决定；LR 配方没有任何预处理（有缺失值就直接崩溃）；KS 的抽样噪声完全没有量化手段（无 bootstrap/多 seed），用户"差 1-2 个点"的判断在平台上无法被证实或证伪；融合与分群建模全仓为零。结论：lgb 单模路径可以接近封顶，但"模型选择与评估口径"整体尚不是完全体——尤其默认无 OOT 这一条，会让报告出的 KS 本身就不是行业认可的口径。

**已做到位（done_well，经核实）**：

- 精确 KS（非分箱）：逐样本累计分布 + 分数 tie 用 change_points 正确处理（marvis/feature/metrics.py:13-27），加权版同样处理 tie（metrics.py:30-47）
- AUC 用秩统计量（rankdata 平均秩）实现，tie 处理正确；加权 AUC 对同分组做 0.5 交叉项修正（metrics.py:50-89）
- 样本权重全链路一致：加权 KS/AUC/PSI 在训练指标中与非加权并行输出，权重列校验严格（非正/缺失即报错）（marvis/packs/modeling/recipes/common.py:118-267）
- 切分器能力完备：有序规则集（first-match-wins）、时间分位数 OOT、分组随机（整组同侧防近重复泄漏）、空集守卫，全部确定性（marvis/packs/modeling/prepare.py:132-305）
- 自动切分默认带防泄漏分组：按身份类列（cust_id/mobile/身份证等）整组切分（marvis/agent/modeling_setup.py:502-533）
- 调参纪律正确：选择目标=test_ks−0.5×max(0,train−test) 过拟合罚项，OOT 明确只报告不参与选择；seeded RNG+固定种子+单线程确定性（marvis/packs/modeling/tune.py:92-127）
- 调参搜索空间合理（num_leaves/max_depth/lr 0.01-0.08/min_child_samples/L1/L2/bagging/scale_pos_weight），带 3000 轮上限+100 轮早停（tune.py:57-73,110-127）
- 多配方对比在数据轴上公平：同特征集、同切分列、同种子（marvis/packs/modeling/tools.py:700-715）
- 选择指标族与目标类型匹配：binary 用 KS、continuous 用 RMSE、multiclass 用 macro-AUC→logloss 回退（tools.py:752-797,915-929）
- selection_policy 机制完善：PMML/handoff/评分卡/单调性/特征数上限/OOT PSI 上限/任意指标阈值 + 违规 override 必须留 override_reason（tools.py:969-1232）
- 单调性支持到位：lgb/xgb 单调约束归一化校验、评分卡 chimerge 分箱+单调方向自动判定（common.py:80-115, marvis/packs/modeling/recipes/scorecard.py:114-136）
- 过拟合双口径检查（train/test 相对 10% + train/OOT 绝对 0.05）并入训练指标与验证（marvis/validation/overfitting.py:6-18）
- 切分确认门有 OOT 缺失/占比<5% 的明确警告文案（marvis/agent/gate_payloads.py:259-266）
- MLP 配方预处理正确：确定性 impute(median)→scale→MLP 管道（marvis/packs/modeling/recipes/mlp.py:27-45）
- 类不平衡处理：scale_pos_weight='auto' 按（加权）好坏比解析，训练与调参两侧一致（common.py:61-77, tune.py:106-108,215-216）
- 分数 PSI 用 train 分数十分位边界统一计算 test/OOT 漂移（common.py:185-208）

**缺口**：


#### SEL-1 · 默认建模切分不建 OOT，时间外推 OOT（oot_by_time）是全仓从未被调用的死代码

**KS 影响：High · 工作量：M · ✅ 对抗验证 CONFIRMED**

**缺口**：行业极致做法：信贷模型时间外推 OOT 是强制口径——按放款月/申请月留出最近 1-3 个月做 holdout，随机切分对时间相关特征（征信查询数、额度使用率等随宏观漂移）会系统性高估 KS 且掩盖衰减。现状：默认路径产出的是纯随机 75/25 train/test、无任何 OOT；prepare.py 里已实现的时间分位数 OOT 能力没有任何调用方，用户只有在上传数据自带切分列时才可能有 OOT。champion 选择的 oot_ks 全部回退到 test_ks，monitoring 的 psi_oot 检查全为 n/a

**证据**：marvis/agent/modeling_setup.py:518-548 _generate_split 固定 split_config={"test_size":0.25,"group_cols":...}（无 oot_by_time/random_oot），只在 note 里说"未设 OOT"；modeling_setup.py:74,99 模板 slots 固定 split_config={}；grep -rn 'oot_by_time|random_oot' marvis --include=*.py 除定义处 prepare.py:157-173 外零命中。同时 modeling_setup.py:119-121 已经能识别 loan_month/放款月列（用于 vintage 报告）却不用于切分

**改法**：1) _generate_split 在检测到 loan_month_col/日期列（业务列推断已有此能力）时默认走 oot_by_time 切分（如最近 20% 时间做 OOT），并在 G1 切分门里展示 OOT 时间窗供确认；2) 无日期列时保留现状但把 gate 警告升级为需要显式确认的决策项；3) 把 split_config 的 oot_by_time/rules 暴露进切分门的 adjust 词表，让用户能一句话改成时间切分

**验证说明**：断言实质成立,所有引用证据核实无误。(1) grep 验证:oot_by_time 在 marvis/**/*.py 中仅命中 marvis/packs/modeling/prepare.py:156,318;random_oot 仅 prepare.py:136,168;全仓其余命中只有 tests/test_modeling_prepare.py 和一份 spec 文档,无任何 agent/模板/工具生产调用点。(2) 默认切分:modeling_setup.py:533 自动切分固定 split_config={"test_size":0.25,"group_cols":group_cols},无 oot_by_time/random_oot;74/99 行模板 slots 固定 split_config={};且模块 docstring(9-10 行)与 _generate_split docstring(519-520 行)明确写明"No OOT is fabricated; downstream OOT metrics degrade to n/a"——即无 OOT 是有意设计而非疏漏,但方法学缺口成立:即使 modeling_setup.py:120 已能检测 loan_month/apply_month 等放款月列,该检测只喂给 vintage/样本分析报告(report_compute.py、tools.py:3013-3027),从不用于时间外推切分。(3) 兜底路径逐一排除:adjust_specs.py 类型化调整参数不含任何 split/OOT 项;gate_response_adapter 结构化控件只覆盖 screen/dedup/modeling_setup/tuning;instruction_router 与 auto_drive 的 LLM prompt 均未暴露 split_config schema;orchestrator 模板 slot 默认 {}。唯一理论可达路径是 G1 门自由文本 adjust(apply_adjust 会应用 dep.inputs 已有键,split_config 在 make_split 步骤 inputs 中),但需 LLM 凭空猜出从未文档化的嵌套 schema,不构成实际调用方。(4) 下游退化属实:冠军选择 tools.py:929 用 _score_first(("oot_ks","test_ks")) 即 oot 缺失回退 test_ks(回归/多分类同理 923-928);gate_payloads.py:478-485,521 psi_oot 为 None 时回退 psi_test;report_texts.py:272,276 显示"暂无OOT…待复核"。唯一措辞修正:oot_by_time 并非"不可达死代码"——它是 tool_make_split/tool_prepare_modeling_frame(tools.py:285,334)原样透传的配置键且有单测覆盖;准确表述为"已实现、可配置,但全仓没有任何生产路径填入该键",用户只有自带切分列(modeling_setup.py:226-229 passthrough)才有 OOT。该修正不影响断言实质,故判 CONFIRMED。

**验证员核实证据**：核实后的证据链(文件均为仓库相对路径):(1) grep -rn 'oot_by_time' marvis --include='*.py' 仅命中 marvis/packs/modeling/prepare.py:156(_make_split 读 config.get("oot_by_time"),按时间分位数 cutoff 切 OOT)与 prepare.py:318(_requested_columns 收集该列);random_oot 仅 prepare.py:136,168。全仓(含非 .py)另有 tests/test_modeling_prepare.py 与 docs/superpowers/specs/2026-06-13-phase-6-modeling-pack.md,即该能力只被单元测试调用。(2) marvis/agent/modeling_setup.py:518-548 _generate_split:split_config 固定 {"test_size":0.25,"group_cols":group_cols},note 明说"未设 OOT…OOT 相关指标将显示 n/a";docstring(519-520)及模块头(9-10 行)写明"No OOT is fabricated"为有意设计。74/99 行两个模板 slots 固定 "split_config": {}。modeling_setup.py:119-126 _BUSINESS_COLUMN_ALIASES 检测 loan_month/apply_month/book_month/放款月等列,但仅用于 vintage/样本分析报告(marvis/packs/modeling/report_compute.py:45-46、tools.py:3013-3027),不用于切分。(3) 透传链:orchestrator/templates/sample.py:158,348,371,573,629 的 split_config slot 默认 {};tools.py:285(tool_prepare_modeling_frame)与 334(tool_make_split)原样透传 inputs.get("split_config") or {} 给 prepare_modeling_frame——即 oot_by_time 是可达配置键但从无生产填充方。兜底排除:agent/adjust_specs.py:7-12 类型化调整参数集合不含任何 split/OOT 键;agent/gate_response_adapter.py 结构化控件仅限 screen_features/confirm_join/choose_modeling_spec/tune_hyperparameters;agent/instruction_router.py 与 agent/auto_drive.py 的 LLM prompt 未暴露 split_config schema。理论残余路径:plan_driver.py:195-196 + gate_execution_adapter.py:109-133 的自由文本 adjust 会应用 dep.inputs 中已存在的键(make_split 步骤 inputs 含 split_config),但需 LLM 自行猜出未文档化的嵌套 schema,实践上不可达。(4) 下游退化:tools.py:929 冠军选择 max(rows, key=_score_first(("oot_ks","test_ks")))——无 OOT 时全体回退 test_ks(回归 923 行 ("oot_rmse","test_rmse")、多分类 925-928 同理);agent/gate_payloads.py:478-485,521 psi_oot(psi_oot_vs_train)为 None 时 stability 回退 psi_test;report_texts.py:272,276 OOT KS/PSI 显示"暂无OOT…数据,待复核"。(5) 用户自带切分列时 modeling_setup.py:226-229,265-273 passthrough 且 holdout_values=["oot"],与断言"仅上传自带切分列才可能有 OOT"一致。措辞修正:标题中"死代码"应改为"已实现但零生产调用方的配置能力"(有单测覆盖、工具层可透传),其余断言全部成立。


#### SEL-2 · 多算法对比系统性不公平：只有 lgb 被调参，xgb 默认 20 棵树、catboost 默认 50 次迭代、全部无早停参赛

**KS 影响：High · 工作量：M · ✅ 对抗验证 CONFIRMED**

**缺口**：行业极致做法：算法横向比较必须"同等调参预算"，否则结论只反映调参投入而非算法优劣。现状：lgb 拿 12 轮随机搜索+3000 轮早停训练，xgb 拿 20 棵 0.3 学习率的树、catboost 拿 50 次迭代——两者严重欠训练，KS 距其调参后的潜力可差 3-10 个点。后果双重：a) "多算法取最优"的比较结论被预先决定为 lgb 胜出，比较仪式化；b) 用户单选 xgb/catboost 建模时直接拿到欠训练模型，违背用户"调参到位则 KS 封顶"的前提

**证据**：marvis/packs/modeling/tools.py:575 tune_enabled = recipe == "lgb"；tools.py:598-599 非 lgb 直接返回空调参；tools.py:709-711 train_models 中非 lgb 的 params=dict(control_params)（仅权重列/单调约束）且 early_stopping_rounds=None；recipes/xgb.py:43 num_boost_round 默认 20（xgb 库默认 eta=0.3/max_depth=6）；recipes/catboost.py:37 iterations 默认 50；recipes/lgb.py:44 lgb 兜底也是 20 棵但实际吃到调参结果（最多 3000 轮+早停）

**改法**：1) 把 tune.py 的随机搜索泛化为按 recipe 的参数空间（xgb/catboost 空间与 lgb 高度同构：树数/深度/学习率/正则/子采样），至少做到"每个参赛算法同等 n_trials"；2) 短期兜底：给 xgb/catboost 换上可辩护的静态默认（如 lr=0.05 + 早停定树数，early_stopping_rounds 在 train_models 里打开）；3) 在对比结果里标注每个算法的调参预算（n_trials/轮数），让不公平至少可见

**验证说明**：断言的每一处代码证据都逐行核实为真，且"缺失"部分经多关键词全仓搜索确认无兜底实现。

核实结果：
1. tools.py:575 `tune_enabled = recipe == "lgb"` 原文存在；:581 非 lgb n_trials 归零；:585 reason 明说"暂不执行随机搜索,使用算法默认参数"。
2. tools.py:594-599 tool_tune_hyperparameters 对 recipe != "lgb" 直接 return，n_trials=0、trials=[]，best_params 只是入参透传（configured+control），无任何搜索。:590-593 注释自认"xgb tuning is a later slice"。
3. tools.py:680-715 tool_train_models：:709 `params={**tuned_params, **control_params} if recipe == "lgb" else dict(control_params)`，而 _training_control_params（tools.py:3670-3687）只含 sample_weight_col 和 monotone_constraints，不含任何树数/学习率；:711 `early_stopping_rounds=None` 对所有 recipe。lgb 不受此害——tune.py:128/:155 把早停后的 best_iteration 写回 best_params["num_boost_round"]（上限 3000 轮、早停 100 轮），xgb/catboost 则纯裸跑。
4. recipes/xgb.py:43 `num_boost_round = int(params.pop("num_boost_round", 20))`，xgb recipe default_params（recipes/__init__.py:37-40）只有 objective/eval_metric，故对比路径必然落到 20 棵；:44-45 因 early_stopping_rounds=None 不设早停；XGBClassifier 库默认 eta≈0.3、max_depth=6。
5. recipes/catboost.py:37 iterations 兜底 50（default_params 给 learning_rate=0.05、depth=4，lr=0.05 配 50 轮严重欠训练）；无早停参数（fit 带 eval_set，CatBoost 默认 use_best_model 只能在 50 轮内截断，不能加轮数）。
6. recipes/lgb.py:44 兜底同样是 20，但流水线 lgb 吃到调参结果。
7. 编排确认非纸面问题：orchestrator/templates/sample.py:422-489 的"配置调参→调参→训练模型"链路把 调参.best_params（lgb-only）喂给 train_models，recipes 来自建模规格（BINARY_MODELING_RECIPES=lgb/xgb/catboost/lr/scorecard/mlp，tools.py:85），随后 compare_experiments/select_experiment 按 OOT KS 选优——即"多算法对比"确实是 tuned-lgb vs 默认-20棵-xgb vs 默认-50轮-catboost。
8. 缺失确认：grep optuna/hyperopt/skopt/bayes 全仓零命中；唯一调参模块 tune.py 文档自述"Hyperparameter search for the LightGBM modeling recipe"；recipes/__init__.py 所有 recipe 的 param_space={}（钩子存在但全空）；scenarios.py 的 param_overrides 无任何 xgb/catboost 树数覆盖；defaults.py 只有随机种子。

仅两处精度修正（不影响结论）：a) :598-599 返回的 best_params 是入参透传而非空 dict（效果等同零调参）；b) "KS 差 3-10 个点"是经验估计，代码本身无法证实具体幅度，但方向性成立（lr=0.05×50 轮 / eta=0.3×20 棵 vs 12 trials×3000 轮早停搜索的不对称是结构性的）。n_trials 默认 12（tools.py:536/:570）也与断言一致。

**验证员核实证据**：- marvis/packs/modeling/tools.py:575 `tune_enabled = recipe == "lgb"`；:581 非 lgb n_trials=0；:585 reason="…暂不执行随机搜索,使用算法默认参数"
- marvis/packs/modeling/tools.py:598-599 recipe != "lgb" 时 tool_tune_hyperparameters 直接返回 {"best_params": 入参透传, "n_trials": 0, "trials": []}（注意：非空 dict，是 configured+control 参数原样返回，但零搜索）；:590-593 注释明言 "xgb tuning is a later slice"
- marvis/packs/modeling/tools.py:709 train_models 中非 lgb 的 params=dict(control_params)，control_params 仅含 sample_weight_col/monotone_constraints（tools.py:3670-3687）；:711 所有 recipe early_stopping_rounds=None（lgb 不受害：tune.py:128/:155 将早停后的 best_iteration 写入 best_params["num_boost_round"]，搜索上限 max_boost_round=3000、early_stopping_rounds=100，tools.py:611-612）
- marvis/packs/modeling/recipes/xgb.py:43 num_boost_round 兜底 20；xgb recipe default_params 仅 objective/eval_metric（recipes/__init__.py:37-40），无树数覆盖来源 → 对比路径固定 20 棵、无早停（xgb.py:44-45 因 None 跳过）、XGBClassifier 库默认 eta≈0.3/max_depth=6
- marvis/packs/modeling/recipes/catboost.py:37 iterations 兜底 50，default_params 含 learning_rate=0.05/depth=4（recipes/__init__.py:47-53）→ lr=0.05 配 50 轮；无早停参数（fit 带 eval_set，CatBoost 默认 use_best_model 至多截断到 ≤50 轮，不能补训练量）
- marvis/packs/modeling/recipes/lgb.py:44 兜底同为 20，但编排链路使其吃到调参 num_boost_round
- 编排落地证据：marvis/orchestrator/templates/sample.py:422-489（配置调参→调参→训练模型，params=$ref:调参.output.best_params，recipes=$ref:选择建模规格.output.recipes）；对比池 BINARY_MODELING_RECIPES={lgb,xgb,catboost,lr,scorecard,mlp}（tools.py:85）；train_models 按 OOT KS(test 兜底)选 best（tools.py:680-683 docstring）
- 缺失确认：全仓 grep optuna/hyperopt/skopt/bayes 零命中；唯一调参实现 marvis/packs/modeling/tune.py 自述 LightGBM 专用随机搜索；recipes/__init__.py 所有 recipe param_space={}；scenarios.py param_overrides 无 xgb/catboost 树数/轮数覆盖
- n_trials 默认 12：tools.py:536 与 :570 `int(inputs.get("n_trials") or 12)`
- 幅度修正："KS 可差 3-10 个点"为经验估计而非代码可证事实；结构性不公平（tuned lgb vs 默认欠训练 xgb/catboost）本身完全成立


#### SEL-3 · LR 配方零预处理：有缺失值直接崩溃、无标准化/WOE，且一个配方失败会连坐终止整批多算法训练

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业极致做法：LR 参赛必须配 WOE 编码或 缺失填充+标准化 管道，否则不是"LR 的实力"——未标准化时 L2 罚项(C=1.0)按原始量纲扭曲系数，缺失值则直接不能跑。现状：典型信贷数据（普遍含缺失）下 recipes 含 lr 时整批训练失败（连 lgb 的结果也拿不到）；即便无缺失，LR 的 KS 也被量纲+正则扭曲压低，家族间比较失真

**证据**：marvis/packs/modeling/recipes/lr.py:24-40 直接 LogisticRegression().fit(train[features])，无 SimpleImputer/StandardScaler/WOE（对照 mlp.py:43 有 impute→scale 管道，scorecard.py:46-57 有 WOE）；sklearn LR 遇 NaN 抛 ValueError；tools.py:729-733 train_models 循环内任一 recipe 异常即 raise，无 per-recipe 隔离。搜索确认：grep -n 'Imputer|Scaler|woe' recipes/lr.py 零命中

**改法**：1) lr.py 改为 Pipeline(SimpleImputer(median)→StandardScaler→LogisticRegression)，与 mlp 同构（都是确定性变换，可进 PMML）；2) train_models 循环改为 per-recipe try/except：失败的记为 failed 并附原因，其余算法继续，最后在对比里展示失败项；3) 若想保留"裸 LR"语义，至少在 choose_modeling_spec 的 disabled_algorithms/警告里声明 lr 需要无缺失数据


#### SEL-4 · 没有独立验证集：test 同时承担早停、调参选择、对比口径三重职责，test_ks 被选择过程污染

**KS 影响：Medium · 工作量：M · — 未独立验证（中低影响）**

**缺口**：行业极致做法：train/valid/test(/OOT) 四分——valid 供早停与调参选择，test 只做一次性无偏评估，OOT 只报告。现状三分口径下，test_ks 是"在 12-40 个候选里挑出来的最大值"，天然向上偏（几十个 trial 下偏差可达 0.5-1.5 个 KS 点），而 DOM-9 若按其建议改回"用 test 选 champion"，污染会进一步叠加：同一个 test 既选参数又选算法。用户想验证"谁做差距不超 1-2 个点"，需要一个真正无偏的口径集

**证据**：marvis/packs/modeling/prepare.py:20 VALID_SPLIT_ASSIGNMENTS=("train","test","oot") 无 valid；tune.py:110-127 早停 valid_sets 用 test 且 12-40 个 trial 均按 test_ks 择优；recipes/lgb.py:57 最终训练 eval_set 也是 test；实验对比行的 test_ks 即这同一个集合（experiment.py:117）。全仓 grep 'valid' 无第四切分。（DOM-4 只修校准自评、DOM-9 只修 champion 用 OOT，均未触及缺 valid 集本身）

**改法**：1) _make_split/VALID_SPLIT_ASSIGNMENTS 增加 valid（如 60/15/25±OOT），tune 与早停全部指向 valid；2) test 只在 train_models 后算一次并冻结；3) 兼容旧三分数据：无 valid 列时在 train 内部再切（按 group_cols 防泄漏），并在指标里标注 test 是否被调参使用过


#### SEL-5 · KS 的抽样误差零量化：无 bootstrap 置信区间、无多 seed 重复，冠军由千分位 KS 差决出

**KS 影响：Medium · 工作量：M · — 未独立验证（中低影响）**

**缺口**：行业极致做法：报告 KS 时附 bootstrap 95% CI（几千次重抽样，确定性种子即可保持平台不变量），冠军差距落在彼此 CI 内时提示"统计不可分"；同配方跑 3-5 个 seed 看 KS 方差，把"稳定地好"与"抽中好"分开。现状：用户的核心判断——"谁做差距不超 1-2 个点"——在典型 OOT 规模（几千-几万样本，KS 标准误约 0.01-0.02）下恰好是噪声量级，而平台既不能量化也不提示；两个 KS 差 0.003 的候选会被斩钉截铁地分出胜负

**证据**：grep -rniE 'bootstrap|confidence|conf_int' marvis/packs/modeling marvis/validation 零命中（仅 feature/derive.py 的 LLM confidence 字段）；grep -rniE 'multi.?seed|seed_list|seeds\b' modeling 包零命中；tools.py:797/929 用 max() 对单点 KS 排序定冠军，无并列/区间概念；seed 全链路单值（DEFAULT_RANDOM_SEED）

**改法**：1) compute_model_metrics 增加确定性 bootstrap（固定种子重抽样 1000 次）输出 test/oot KS 的 std 与 95% CI，进实验对比行与报告；2) train_models 增加可选 seeds 参数（如 [seed, seed+1, seed+2]）输出同配方 KS 均值±方差，作为 selection_policy 可引用的稳定性证据；3) 选择理由文案在差距 < 2×标准误时明确写"统计上不可分，按次级标准（特征数/稳定性）选择"


#### SEL-6 · 模型融合能力全仓为零：无 seed-bagging、无 stacking/blending，追求极致 KS 缺最后一层

**KS 影响：Medium · 工作量：L · — 未独立验证（中低影响）**

**缺口**：行业极致做法：竞赛级与头部机构的信贷模型常规操作——多 seed 同配方概率平均（seed-bagging，稳定 +0.3-0.8 KS 且降低方差）、lgb+xgb+catboost 概率/秩平均、或 OOF stacking 一层 LR。这正是"特征和调参都到位后"仍能再挤出 0.5-1.5 个 KS 点的主要手段，也直接检验用户"KS 封顶"假设的边界。现状平台完全没有此能力，多算法训练的产物只用来单选

**证据**：grep -rniE 'ensemble|stacking|blend|VotingClassifier|StackingClassifier' marvis --include='*.py' 全仓零命中；SUPPORTED_MODELING_RECIPES（tools.py:75-84）全部是单模型配方；train_models 训练 N 个模型后只做 argmax 选择（tools.py:741），从不组合

**改法**：1) 最低成本切入：加一个 'blend' 配方——对 train_models 已产出的 2-3 个成员模型做概率平均（或秩平均），作为额外一行参赛（部署侧可导出成员模型+权重）；2) 进阶：seed-bagging 版 lgb（同参数 5 seeds 平均）作为独立 recipe；3) 明示代价：blend 不支持 PMML/评分卡，默认 selection_policy(require_pmml) 下仅作 benchmark 上界展示，让用户看到"距离封顶还有多少"


#### SEL-7 · 默认 selection_policy 只查 PMML/handoff 两项交付位，不含任何稳定性/过拟合/最小 KS 默认阈值；delivery-ready 预过滤还会静默把 catboost/mlp 排除出冠军竞争

**KS 影响：Medium · 工作量：S · — 未独立验证（中低影响）**

**缺口**：行业极致做法：champion 选择默认就带稳定性护栏——OOT PSI≤0.1/0.25、train-test KS gap 上限、最小可接受 KS 下限，且"因不可交付被跳过的更优候选"必须显式披露。现状：默认策略下一个 oot_ks 略高但 PSI 0.30、gap 0.15 的候选会直接当选（monitoring 阈值 DEFAULT_MONITORING_THRESHOLDS 在选择之后才生成，是 DOM-3 已知的纸面 JSON）；catboost 参赛胜出却被无声让位给 lgb 时，用户在 selection_reason 里看不到任何解释

**证据**：marvis/agent/modeling_setup.py:406-409 _default_selection_policy 二分类默认仅 {require_pmml:True, require_handoff:True}；max_oot_psi/metric_thresholds/max_feature_count 全部 opt-in（tools.py:969-989）；tools.py:915-920 _pick_best_comparison_row 先按 delivery_ready(pmml+handoff) 过滤再排序——catboost/mlp（不在 PMML_SUPPORTED_ALGORITHMS，tools.py:71）即使 KS 最高也被静默跳过，选择理由不提及

**改法**：1) _default_selection_policy 二分类默认加 max_oot_psi=0.25 与 metric_thresholds（如 overfit_train_test_gap max=0.12，与 DEFAULT_MONITORING_THRESHOLDS 的 fail 线对齐，把监控阈值前置为选择约束）；2) _pick_best_comparison_row 在 delivery-ready 过滤淘汰了指标更优的行时，把"更优但不可交付"的候选与差值写进 selection_reason/policy_decision；3) 文档化：默认参赛列表里 catboost/mlp 在默认策略下只能当 benchmark


#### SEL-8 · 分群建模（segmented models）不支持：无按客群分别建模再合并的任何路径

**KS 影响：Low · 工作量：L · — 未独立验证（中低影响）**

**缺口**：行业极致做法：客群显著异质时（新客/老客、有征信/白户、不同渠道）分群建模 + 分数对齐合并，常见收益 1-3 个 KS 点，评分卡时代即是标配（segmentation analysis 决定几张卡）。现状平台已能检测渠道/月份列并展示分布，但止步于展示；用户想做只能手工拆文件跑多个 task，交叉对比与合并口径全部自理

**证据**：grep -rniE 'segment' marvis/packs/modeling marvis/feature marvis/agent/modeling_setup.py --include='*.py' 零命中（业务语义上仅有渠道/月份列的分布展示 tools.py:364-392 _GROUP_COLUMN_HINTS，用于切分确认表，不用于建模）；split 规则集只能把渠道分到 train/test，不能按渠道各建一模

**改法**：1) 先补诊断：在切分/筛选阶段加"分群价值评估"——按候选分群列分别算基线模型 KS 与合并 KS 差值，量化分群收益后再决定是否投入；2) 若收益显著，扩展 plan 模板支持 segment_col：按 segment 循环 train_models、各段独立调参、报告并排展示分段 KS 与总体加权 KS；3) 合并口径用分段内校准（各段 PD 校准后天然可比）衔接现有 calibrate_model

