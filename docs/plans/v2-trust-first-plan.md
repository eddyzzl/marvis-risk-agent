# V2 信任优先开发计划（Trust-First Plan）

> **依据**：`docs/reviews/2026-07-04-full-read-and-owner-qa.md`（两轮全量读码 + 所有者五问）。
> **生成**：2026-07-04，经盘问式拍板会敲定。
> **与 master backlog 的关系**：`v2-master-backlog.md` 仍是唯一进度权威——本计划批准后在 backlog 追加 Phase T / Phase C 两个 section 与本文档指针，进度只在 backlog 更新；本文档冻结为决策记录与 spec 总纲。
> **交付节奏**：沿用既定偏好——每份 spec 单独出、单独审（函数级 + 内部伪代码颗粒度），审过再实现。

---

## 0. 拍板记录（2026-07-04，共 6 项）

| # | 决策点 | 拍板结果 |
|---|---|---|
| 1 | 计划总体定位 | **信任优先，两阶段**：Phase T（信任层）全绿后再进 Phase C（能力层） |
| 2 | 出口判据数据源 | **三层混合**：对抗形状注入器（进 CI）+ 公开数据集端到端/KS 基准 + 真实业务材料人工对账 |
| 3 | 可信数字层范围 | **对账 MVP + 轻血缘**：5 个最高危 gate 数字双路对账+阻断红旗，数字附最小溯源元组；一键重放与全链血缘图推迟 |
| 4 | Phase C 范围 | **只做①标签构造+成熟度工具**；评分 API+影子、监控调度+推送、征信报文解析全部显式推迟 |
| 5 | DoD-11 改写 | **数据判据为主，review 限范围**：收口 = 三层数据判据全绿 + 一轮仅限本期 diff 的范围化 review（只修 critical/high，不循环）；全量 review 降为大重构后按需 |
| 6 | 反欺诈定位 | **不做但留口**：本期及可预见未来不做，记入长线区候选（"视业务需要再评估"），不写永久排除 |

随拍板落地的推导性决定（沿平台既有先例，未单独征询）：

- **label_semantics 门**跟随 NaN 标签门先例：快照/增量语义强制确认（typed error + 显式 flag），不做静默默认。
- **口径变化不做兼容**：单机单用户，bug 修复直接修正数字，发布说明列明每处口径变化（清单见 T1）。
- **KS 基准达标定义**在 T4 spec 内敲定（提案：公开数据集上 agent 全流程 KS ≥ 一次性人工精调基线 − 0.005，基线实验落盘为地面真值）。

---

## 1. 总体结构与依赖

```
Phase T（信任层）                                Phase C（能力层，收缩版）
T0 清障 PR ──┐
T1 语义正确性修复包 ──┬─→ T3 可信数字层 MVP ──→ T4 三层数据判据 harness ──→ 范围化 review ──→ 收口 ──→ C1 标签构造与成熟度工具
T2 重复实现收敛 ──────┘        （T2 是 T3 前置：对账要求每个概念先有唯一权威实现）
```

- T0 与 T1 可并行启动；T2 依赖 T1 中的 labels 收敛决定；T3 依赖 T2（对账的"权威路"必须先唯一化）；T4 最后，因为它验收前面全部。
- 每份 spec 独立成文、独立审、独立 PR；发版走 `scripts/release_push.py`。

---

## 2. Phase T — 信任层

### T0：清障 PR（无 spec，直接一个 PR，可与 T1 并行）

低风险机械清理，全部来自报告 Q4.3 CONFIRMED 清单：

1. 删除 `run_validation` + `EngineInputs`（先把 `_filter_feature_categories`/`_model_features` 迁至活模块并更新 `pipeline_cellgen.py` 两处字符串 import；同步删除/改写 `tests/validation/test_engine.py`、`tests/output/test_e2e_results_to_outputs.py` 的 engine 分支）。
2. 删除 `LLMClient` Protocol（`orchestrator/eval/runner.py:39`）。
3. 删除 `plugins/errors.py` 的 `ToolResourceError`、`WorkerProtocolError`（保留 spec 规定的 `ToolTimeoutError`）。
4. 删除 `ModelingRepository.set_model_artifact_baseline_distributions`（含私有 helper）与 `attach_experiment_result`。
5. 删除 glow 帧图 `marvis-glow-00..09.png`、`readErrorMessage`、`governanceExtensionPanelDefinitions`。
6. `data_dictionary` 从 `marvis/agent/` 挪到共享层，消除 3 个 pack 的向上 import。
7. CSS 死选择器 **token 级**手术清理（严格按报告警告：不整块删，逐 token 核对）。

**不进 T0**（列出以免误删）：高置信未验证候选（repo 层 no-audit 变体、api_v2 包装等——T0 之后另开小 PR 逐条自查）；`_sample` 包与 `join/cancel` 端点（见"待产品判断"）。

**待产品判断（T0 期间顺手拍板即可）**：① V2 join gate 是否补"取消"按钮（`join/cancel` 端点已存在、前端未接线）；② `_sample` 诊断包保留还是让 loader 跳过下划线目录。

### T1 spec：语义正确性修复包（本计划核心，第一份出审的 spec）

范围 = 报告 A2 全部 8 条确认 bug + join 二审 2 条 high + 方法学残留 4 条 + 语义契约门。每条修复必须附"失败形状测试"（喂脏形状断言正确行为，而非只测快乐路径）。

**A. 数据语义类（6 条）**

| # | 问题 | 修法要点 |
|---|---|---|
| 1 | vintage 快照标签重复计数（`validation/vintage.py:111` ← `strategy/vintage.py:26`） | 引入 `label_semantics ∈ {incremental, snapshot}` 参数：incremental 走现有累计；snapshot 按每 MOB 边际口径直读（等价 `report_compute` 的自我保护）。策略路径强制确认门（无声明→typed error，gate 询问"你的 bad 列是快照还是增量"，附两种语义的示例说明）。`VintageCurve` 契约补 `warnings` 字段，`data_quality_warnings` 贯通 tool 输出与 renderer |
| 2 | EL 跨快照月求和虚高（`analysis/loss.py:198`） | `total_el` 改为参考快照月口径（默认最新快照），逐月表保留；gate/xlsx 的 highlights 标注口径；求和基础写进 assumptions |
| 3 | sentinel 掩码不进重放链（`feature/tools.py:616`） | 预处理链新增 `sentinel` step kind，`apply_preprocessing_steps` 先行掩码；补 sentinel×重放链组合测试 |
| 4 | slice bad_rate/approval_rate NULL 当好客户（`data_ops/tools.py:527`） | SQL 改为 NULL 剔除分母（与全平台 isfinite 口径对齐）；输出附 `unlabeled_count` |
| 5 | join 空白键错配（`backend.py:1135`） | `_sql_transform`/`_sql_value_text` 统一 `nullif('')` 语义，与诊断、pandas fallback 三方一致；空白键行为测试 |
| 6 | float64 证件号科学计数法（`backend.py:1136`） | 键规范化统一处理 float 存储整数键（含 >15 位精度警告红旗）；SQL 与 Python 双路渲染一致性测试 |

**B. join 读取一致性（2 条，join 二审 high）**

| # | 问题 | 修法要点 |
|---|---|---|
| 7 | 诊断 all_varchar vs 执行 typed 读取分裂（`backend.py:829` vs `:820`） | 匹配率诊断与执行 join 使用同一读取器/同一键变换（读一致性原则：诊断读到什么，执行就 join 什么） |
| 8 | 长 ID 守卫阈值门控跨文件不一致（`csv_ingest.py:67`） | 键列 dtype 决策提升为跨文件一致性检查：同一 join 键在两侧 dtype 不一致→红旗+强制确认；阈值放宽评估（零填充短码保护） |

**C. 证据/交互失真类（2 条）**

| # | 问题 | 修法要点 |
|---|---|---|
| 9 | 冠军证据错指标+方向词写死（`renderers.py:383`） | renderer 读取工具输出的 `selection_metric` 字段渲染真实选择依据；比较句方向由符号决定；修正 `test_strategy_development.py:786` 锁错标签的测试 |
| 10 | 意图识别无否定（`service.py:113`） | `is_start_validation_intent` 对齐 continue/stop 的否定标记表 + 疑问句护栏；参数化否定用例测试 |

**D. 方法学残留（4 条）**

| # | 问题 | 修法要点 |
|---|---|---|
| 11 | 默认模板 holdout=['oot'] 令精选特征在 train+test 拟合（`templates/modeling.py:293/:581`） | select_features 的 holdout 恢复安全默认（test+oot）；模板显式传参处修正 |
| 12 | `task.time_col` 从不触发时间 OOT（`turn_handlers.py:659`） | build_modeling_proposal 接收并传递 task.time_col；非别名列名也能触发 oot_by_time |
| 13 | screen 无 NaN 标签门（`feature/screen.py:197`） | screen 接入与建模同款 NaN 标签确认（typed error + drop flag），两条自动写记忆路径若受影响一并门控 |
| 14 | refit 5% carve 的 test_ks 进 headline（`train_tools.py:672` / `select_tools.py:272`） | `__refit_holdout__` 指标从 headline metrics 剔除（保留内部诊断），兑现"never reported"注释 |

**发布说明附录**：列明全部口径变化（EL total、slice bad_rate、vintage 快照口径、精选特征拟合范围）——数字与历史报告不一致是修正，不是回归。

### T2 spec：重复实现收敛（4 条 CONFIRMED + 3 条低危顺手）

1. **标签 0/1 强转收敛**：全部 target 读取统一过 `marvis/data/labels.py` + `validation/checks.binary_target_series`；以 grep 清单驱动（~30 处），禁止裸 `pd.to_numeric` 读标签（加 lint/测试守卫）。与 T1-#4/#13 同一动作的两面。
2. **分数分段累计表统一内核**：参数化 `(edges, direction, denominator=count|labeled|amount)`，5 处调用点迁移（`bin_table`、`effectiveness._recompute_*`、`report_tools._score_band_rows`、`report_compute.compute_amount_bin_table`、`metric_tables._ranking_rows`）。
3. **策略分段边界统一**：`strategy/bands.py:131` 默认路径改用 `equal_frequency_edges`，保留显式 `band_edges` 覆盖分支；cutoff 推荐回归测试。
4. **AUC 收敛**：删 `effectiveness.compute_auc`，调用 `feature_auc`；`_roc_ks_curve` 曲线自算、KS 标量取自 `feature_ks`。
5. 顺手（低危）：WOE/IV 平滑公式提取共享 helper；`metric_tables`/`excel`/`effectiveness` 三份格式化 helper 合并至一个模块；MOB 双正则统一为单一 resolver。

**验收**：每个概念收敛后加"双面一致性测试"（同输入喂两个原调用面，断言数字相同）。

### T3 spec：可信数字层 MVP（对账 + 轻血缘）

- **双路对账**：5 个最高危 gate 数字——join 匹配统计、vintage 曲线、EL、KS、bad_rate——每个由权威路（T2 收敛后的唯一内核）+ 独立路（DuckDB SQL vs pandas，取平台现成的另一条路）各算一遍；相对偏差超阈值（默认 1e-9，浮点场景 1e-6）→ **阻断性红旗 gate**（不是警告），payload 展示两路数值与计算路径。
- **轻血缘**：gate 上述数字附最小溯源元组 `(dataset_fingerprint, code_version, params_digest, seed)`，渲染为可展开详情；基于现有 evidence envelope 扩展，不建血缘图。
- **显式不做**（防膨胀线）：一键重放、跨报告血缘查询、5 个数字之外的对账覆盖——全部记入推迟清单。

### T4 spec：三层数据判据 harness + DoD 修订

- **第一层：对抗形状注入器**（进 CI fast tier）：可复现生成器覆盖全部已知脏形状（sentinel、快照面板、NULL/非法标签、float64 长 ID、空白键、零填充键、自定义 split 词表、重复 anchor 键），每个 T1 修复绑定至少一个注入器用例；生成器 API 设计为可扩展（新脏形状=新注册项）。
- **第二层：公开数据集端到端**：GiveMeSomeCredit + Home Credit（各一份入库脚本），agent 全流程（接入→特征→建模→策略）跑通 + KS 基准判定；**基线地面真值实验**：一次性人工精调（对照记忆中"数据/基准地面真值"目标），KS 达标线在本 spec 敲定并落盘。
- **第三层：真实材料对账清单**：扩充现有真实业务材料 smoke 为人工验收 checklist——vintage 对财务口径、EL 对拨备口径、bad_rate 对已知报表，逐项签字；不进 CI，作为收口的人工步骤。
- **DoD-11 修订落文档**：按拍板 #5 改写 `v2-master-backlog.md` DoD 第 11 条；DoD 第 7 条产品选择清单追加：反欺诈（不做但留口）、评分 API/影子/监控调度/征信解析（显式推迟）、一键重放与全链血缘（推迟）。

### Phase T 收口

1. 三层数据判据全绿（第一层进 CI、第二层脚本化可复跑、第三层人工签字）。
2. 一轮**范围化 review**（仅 T0–T4 diff），critical/high 修复后即收口，不循环。
3. 走 `release_push.py` 发版。

---

## 3. Phase C — 能力层（收缩版）

### C1 spec：标签构造与成熟度工具（Phase C 唯一项）

覆盖报告 12 阶段图中"样本与标签设计"的 blocker：

- **tool_define_label**：从还款/DPD 长表构造 0/1 目标——参数化观察期、表现期、逾期阈值（FPD/30+/60+/90+@mob）；输出带定坏口径元数据的衍生数据集（进 T3 血缘元组）。
- **成熟度检查**：按 vintage 判定各 cohort 表现期是否闭合，不成熟 cohort 强制确认门（"这些 cohort 尚未成熟，纳入将低估坏率"——正是快照语义门的姊妹）。
- **roll_rate→定坏桥接**：从既有 roll_rate_matrix 输出生成定坏口径建议（如"60+ 在 mob6 后回滚率 <10%，建议 60+@mob6"），作为 tool_define_label 的推荐默认值。两端组件均已存在，本工具是连线。
- 模板接入：标签构造作为建模前置 stage 进 workflow 模板，与 NaN 门、label_semantics 门形成完整标签防线。

**显式推迟清单**（决策记录，非承诺）：批量评分 API+影子运行；监控调度+推送告警；征信报文解析（另立探索 spec 后评估）；反欺诈（留口）；maker-checker 留痕导出；一键重放与全链血缘；import 环拆解与 `app.js`/`renderers.py` 拆分（执行"新功能进新文件+搬一个算一个"的增量策略，`renderers.py` 因安全部件属性优先）。

---

## 4. 风险与对策

| 风险 | 对策 |
|---|---|
| T1 修复改变数字口径引发"哪个才对"的困惑 | 发布说明口径变化清单；T3 血缘元组让每个数字可追来源 |
| T2 收敛引入回归（5 处调用面行为差异是故意的业务口径） | 每处迁移前先写"双面一致性测试"锁现状，差异逐条判定是口径还是 bug |
| 对账层误报（浮点噪声）淹没真信号 | 阈值分级（精确路 1e-9 / 浮点路 1e-6），红旗 payload 必须含两路差值，误报即调阈值并记录 |
| 公开数据集 KS 基准立不住（基线本身没调好） | 基线实验独立成文（数据、seed、参数、过程全落盘），作为地面真值可复审 |
| Phase T 又膨胀成还债循环 | 范围硬边界=本计划列出的项；新发现一律进 backlog 不进本期，除非 critical 且在三层判据路径上 |

---

## 5. 下一步

1. 本计划审定后：backlog 追加 Phase T/C section + 本文档指针。
2. 第一份出审：**T1 spec**（语义正确性修复包，函数级+伪代码颗粒度）。
3. T0 清障 PR 可在 T1 spec 评审期间并行执行。
