# V2 策略开发与风险分析完善计划

> **定位**：把「策略开发」与「风险分析」从 demo 级（6 个工具、2 个模板、共 4 步）升级为与 MODELING 同级深度的一等能力，补齐"六入口全功能可用的完整信贷 Agent 平台"的最后两块。本文是**能力地图 + 架构决策 + 工具/模板清单 + 构建批次**级别的路线主轴；函数级 spec 按惯例"一份一份审"逐个出。
> **状态**：规划中，未开发（2026-07-02）。前置依赖对齐 `docs/reviews/2026-07-02-v2-comprehensive-improvement-review.md`（下称"7-02 审查"）。
> **评审记录**：2026-07-02 已过独立评审（信贷专业 + 架构一致性双视角）：ACCEPT-WITH-RESERVATIONS、零 critical；4 条 major（S1 爆炸半径、分数方向消费方分型、口径钉子、可复用内核盘点）已全部回填至本版。
> **拍板记录**：2026-07-02 用户已拍板全部 6 项（见 §八）：analysis 独立 pack、A3 本期做、即席分析对话 turn 先行、示例数据合成生成、批次交替穿插、监控基准=训练期快照。
> **业务闭环目标**：数据 → 模型 → **策略（把分数变成决策）** → 上线 → **监控与组合分析（把表现数据变成洞察）** → 迭代。前四阶段已有骨架，本计划补后两环。

---

## 一、现状盘点（2026-07-02）

**已有（可复用的地基）**：

- `packs/strategy` 6 个工具：`vintage_curve` / `roll_rate_matrix` / `profit_calc` / `build_strategy` / `backtest_strategy` / `tradeoff_view`，全部确定性、走 ToolRunner 子进程、有 contracts/errors。
- 2 个模板：`STRATEGY_ANALYSIS`（构造→回测(门)→权衡视图，3 步）、`VINTAGE_ANALYSIS`（1 步）；均已接入 PlanDriver，入口不再是 coming-soon。
- 策略创建/回测已走 `*_with_audit` 事务（6-28 修复）。
- 共享底座全部现成：模板→validator→executor→确认门→PlanDriver→内联富表（`metric_tables.py` 已有曲线/热力/PSI 条/KPI 卡语言）。

**可复用内核（评审盘点，防止重复造轮子 / 偏离确定性内核）**：

- **余额加权 vintage 已在内核**：`marvis/validation/vintage.py` 的 `compute_vintage_curve(..., balance_col, denominator="balance")` 已实现；strategy 包装层 `vintage_curve` 只是没把参数透传。B1 的"余额加权"主体是**接线活**，不是新方法学。
- **swap / baseline 对比已在内核**：`backtest_strategy` 已接受 `baseline` 策略并计算 swap-in/out（`backtest.py:26,35,56-73`），`BacktestResult` 已携带 `swap_*` 字段与 `by_segment`（`contracts.py:62-66`）。A4 的 `compare_strategies` 主体是**把既有输出重组呈现**（2×2 矩阵卡 + 差异摘要），不是新计算。
- **打分路径已有全套零件**：`load_model`（`packs/modeling/artifact.py:176`）、`predict_proba` 打分适配器（`packs/modeling/tools.py:4129-4141`）、handoff 的重载打分代码生成（`handoff.py:583-595`）。S-4 的 `score_dataset` 是把它们组装成正式工具 + 数据集登记。

**7-02 审查确认的缺口 + 评审补充（本计划要消化的存量债）**：

- **DOM-2**：分数方向约定自相矛盾（`tradeoff_view` 假设高分=好，`reject_inference` 假设高分=坏，模型原生分是 PD），无方向参数——策略开发的第一硬前置。
- **DOM-3**：无任何"把已训模型应用到新数据"的打分工具；监控策略是纸面 JSON 无执行路径——监控/稳定性分析的硬前置。
- **`cum_bad_rate` 仍是假累计**（评审复核确认，本轮 7-02 审查漏网）：`marvis/validation/vintage.py:93` `cum_bad_rate=bad_rate`（逐 MOB 值冒充累计值）。6-21 审查即列为 confirmed critical，至今未修。**B1/B5 的硬前置**——损失外推建在假累计曲线上必然错。
- **DOM-8**：roll-rate 对观察缺月盲视（跨月断档当单期迁移）、只支持按件数不支持按余额；`period` 硬编码 `"month"`（`roll_rate.py:35`），多期链式需参数化。
- **DOM-11**：swap set / bad rate 空集显示 0.0%、回测缺标签覆盖率口径。
- **DOM-12**：运营点推荐在约束不可行时静默回退最低坏账点。
- **结构性缺失**：无规则策略（挖掘/评估/瀑布）、无分段 cutoff、无额度/定价、无策略版本与 champion/challenger（`StrategyRepository` 无版本/父子字段，需 DB 加列迁移）、无组合迁徙/流量分析、无分数与特征稳定性趋势、无损失估算、无定期组合报告、无即席对话分析、无策略文档交付物。

---

## 二、架构决策（新增 S-1 ~ S-8，沿用总计划 12 条）

1. **S-1 同一底座，不建新驱动**（决策 9 的延伸）：策略/分析全部走「模板 + PlanExecutor + 确认门 + PlanDriver + 转换登记表」；任务差异只落在模板与工具，**不写策略专用驱动**。
2. **S-2 第四个可组合阶段模块 `STRATEGY` + `ANALYSIS` 模板族**：`STRATEGY` 与 JOIN/FEATURE/MODELING 同级（组合示例：策略开发=[SCORE?, STRATEGY]，模型到策略一条龙=[JOIN, FEATURE, MODELING, STRATEGY]）；风险分析类是**独立可跑的 ANALYSIS 模板族**（组合分析、监控、规则评估），输入已满足则跳过前置阶段（沿用决策 11 的 skip 语义）。
3. **S-3 平台级分数方向制度化——渐进式，消费方分两型**（DOM-2 的制度解，按评审意见修订）：
   - 平台常量 `score_direction ∈ {higher_is_riskier, higher_is_better}`；`score_dataset` 出分时把 direction 元数据随 experiment/衍生数据集落盘（需扩展 `ModelArtifact` 契约 + `persist_model_meta` + `.model_meta.json` 读取方，列入 S1 迁移面清单）。
   - **裸分数消费方**（tradeoff cutoff 扫描、`score_dataset` 输出、`monitor_run` PSI 对基准、`reject_inference` 的风险排序——注意后者在 **modeling pack**，跨包）：新增**可选** `score_direction` 参数，未传时默认=该工具当前隐含行为（保持向后兼容，不做 big-bang 破坏性迁移）；**强制层是确定性方向自检**——有标签时算 corr(score,target)，与声明/默认方向矛盾且 |corr| 超阈值 → typed error 走强制确认门。
   - **规则求值消费方**（`backtest_strategy`/`build_strategy`：方向已编码在规则算子 `score >= x` / `score <= x` 里）：**不加冗余参数**（避免双重方向逻辑引入新的反接类 bug），改做**一致性自检**——声明方向 vs 规则算子使用方向矛盾时告警/门。
4. **S-4 打分底座先行**：`score_dataset`（复用 `load_model` + predict 适配器组装，见 §一 可复用内核）是策略开发、监控、稳定性分析的共同前置，最先做。评分卡输出 points 与 PD 双列。
5. **S-5 数据契约扩展**：组合分析需要**表现期长表**契约（`cohort/mob/status/balance` 角色列；`balance/EAD` 列同时是 B5 损失估算的必需输入）；策略回测需要申请层表（含拒绝件时启用 swap/RI 口径）。沿用 JOIN 阶段"数据契约 + typed error + 引导补列"的模式，缺列不静默降级。
6. **S-6 策略采纳是副作用门**：策略"采纳/发布"（生成决策表、写入策略版本库、产出上线包）单独一个强制确认门，与"构造/回测"分离；采纳走 `*_with_audit` 事务（沿用现有范式）；版本库需要 `StrategyRepository` 加列迁移（version/parent/status），一并入该 spec。
7. **S-7 即席分析不许 LLM 算数**（INV-1 红线）：对话式"按渠道看近三个月通过率"类问题，路由到确定性 `slice_aggregate` 白名单算子（**白名单类目**：分组=登记过角色/字典的列；聚合=count/sum/mean/rate/badrate/approval_rate；过滤=等值/区间/时间窗；禁止自定义表达式与跨数据集 join），LLM 只负责把自然语言翻成算子参数并解读结果；翻译结果作为单步计划展示给用户确认口径。
8. **S-8 门注入确定性红旗**（对齐 7-02 审查 AGT-9）：策略/分析每个门的 prompt/控件都附平台算好的红旗 checklist——swap 空集、标签覆盖率低、方向自检矛盾、回测期与开发期重叠、**cohort 未成熟（最大 MOB < 表现窗，坏账被低估）**、缺月断档计数等，LLM 复核而非发明。

---

## 三、能力地图（目标态）

### A. 策略开发（STRATEGY 阶段模块）

| # | 能力 | 说明 | 现状 |
|---|------|------|------|
| A1 | **Cutoff 策略** | 单点 → 三段（通过/人审/拒绝）分数带设计；tradeoff 扫描、swap set、利润联动；运营点推荐带约束不可行显式报告（修 DOM-12，并把不可行契约延伸到三段设计器） | tradeoff_view 雏形 |
| A2 | **规则策略** | 规则挖掘（单变量/双变量组合，限深度 2 + top-k + 确定性排序）；规则集评估（hit rate / bad capture / overlap / **瀑布图**逐条拒绝归因）；与分数带叠加成完整准入策略 | 无 |
| A3 | **额度与定价** | 风险等级 × 额度矩阵；PD 分band 定价曲线；EL/利润模拟（复用 profit_calc 扩展） | profit_calc 雏形 |
| A4 | **策略版本与 champion/challenger** | 策略登记带版本/父子关系（DB 加列）；**把既有 baseline/swap 回测输出重组为对比呈现**（swap 2×2、通过率/坏账率/利润差异）；采纳门 | swap 内核已有（见 §一），缺版本与呈现 |
| A5 | **策略交付物** | 策略说明文档（渲染）、决策表导出（CSV/JSON，可给决策引擎）、策略上线监控计划（阈值随策略落盘） | 无 |

### B. 风险分析（ANALYSIS 模板族）

| # | 能力 | 说明 | 现状 |
|---|------|------|------|
| B1 | **组合分析套件** | vintage 升级（**先修 `cum_bad_rate` 假累计**；余额加权=把内核 `balance_col/denominator` 透传；分 segment 曲线族对比）；roll-rate 升级（观察缺月处理、余额口径、`period` 参数化、多期迁徙链）（修 DOM-8） | 内核比包装层新（见 §一） |
| B2 | **流量/迁徙分析** | flow rate、逾期 bucket migration 热力、入催/出催率 | 无 |
| B3 | **稳定性趋势分析** | score PSI 趋势、特征 CSI 趋势、KS/AUC 时序衰减（依赖 S-4 打分底座；把 DOM-3 的纸面监控策略变成可执行） | 无 |
| B4 | **细分风险分析** | segment risk profile：渠道/地区/额度段/客群 的 通过率×坏账率×利润 矩阵 + 集中度 | 无 |
| B5 | **损失估算** | roll-rate 链式外推的预期损失近似：**EL ≈ Σ_segment EAD × PD_chain(roll-rate 链) × LGD(参数输入)**；EAD/balance 来自 S-5 表现期契约、LGD 为用户参数；输出概率口径与金额口径分列，明确标注"近似口径"，不冒充 ECL 计提 | 无 |
| B6 | **即席对话分析** | 自然语言 → `slice_aggregate` 确定性算子（S-7）；结果内联富表 + LLM 解读 | driver_manual_analysis.js 有面板雏形 |
| B7 | **定期组合报告** | 月度组合风险报告模板：拼装 B1–B5 产出，达到风控会汇报级（复用 report 渲染管线新增 sheets） | 无 |

---

## 四、工具清单（函数级 spec 后续逐份出）

**升级（strategy pack）**：

| 工具 | 升级点 | 门 |
|------|--------|----|
| `tradeoff_view` | +`score_direction`（S-3 裸分数型，可选参数+自检）；+三段带扫描模式；+约束不可行显式返回 `infeasible`（DOM-12） | 运营点/分段选择门 |
| `roll_rate_matrix` | +观察缺月识别与口径（断档剔除/标注）；+`weight_col`（余额）；+`period` 参数化 + 多期链式模式（DOM-8） | decision_point |
| `vintage_curve` | 透传内核 `balance_col/denominator`；+`segment_col` 曲线族；**前置：修内核 `cum_bad_rate`** | decision_point |
| `backtest_strategy` | +标签覆盖率口径与空集显式 N/A（DOM-11）；+**规则算子方向一致性自检**（S-3 规则型，不加冗余参数）；+成熟度红旗（S-8） | 回测确认门（已有） |
| `build_strategy` | +版本/父子字段（`StrategyRepository` DB 加列迁移）；+与分数带/规则集的组合表示 | — |
| `profit_calc` | +定价参数（利率/资金成本/期限）；+额度维度 | — |

**新增（strategy pack）**：

| 工具 | 摘要 | 门 |
|------|------|----|
| `design_cutoff_bands` | 给定约束（目标通过率/最大坏账率/人审容量）产出三段分数带 + 各段量/质预估；约束不可满足时按**显式优先序**报告不可行（口径见 §八） | 分段确认门 |
| `mine_rules` | 从特征集挖掘候选规则（深度≤2、top-k、确定性排序；输出 hit/bad_capture/lift；排序键/最小支持度/去重口径见 §八） | — |
| `evaluate_rule_set` | 规则集顺序评估：瀑布拒绝归因、overlap 矩阵、增量贡献 | 规则集确认门 |
| `compare_strategies` | champion vs challenger：**重组既有 baseline 回测的 swap 输出**为 2×2 矩阵 + 通过率/坏账率/利润差摘要 | decision_point |
| `adopt_strategy` | 采纳/发布：版本定稿 + 决策表导出 + 监控计划落盘（S-6，`*_with_audit`） | **强制确认门** |
| `render_strategy_doc` | 策略说明文档（Markdown/报告 sheet） | — |
| `limit_pricing_matrix` | 风险等级×额度矩阵 / PD band 定价曲线 + EL 模拟（A3，**已拍板本期做**，落 S6） | 矩阵确认门 |

**新增（已拍板：独立 `packs/analysis`；`slice_aggregate` 放 `data_ops`）**：

| 工具 | 摘要 |
|------|------|
| `flow_rate` / `bucket_migration` | 流量与逾期桶迁徙（热力矩阵输出） |
| `segment_profile` | 细分 通过率×坏账率×利润 矩阵 + 集中度 |
| `score_stability_trend` / `feature_csi_trend` | 按月 PSI/CSI/KS 趋势（依赖 score_dataset） |
| `expected_loss_estimate` | roll-rate 链式损失近似（公式与输入见 B5） |
| `portfolio_report` | 组合报告拼装（B7） |
| `slice_aggregate` | 即席分析白名单算子（S-7；**放 `data_ops`，已拍板**） |

**前置（modeling pack，即 7-02 审查 DOM-3）**：

| 工具 | 摘要 |
|------|------|
| `score_dataset` | 组装 `load_model` + predict 适配器（见 §一 可复用内核）对新数据出分（PD + points 双列 + direction 元数据），登记衍生数据集 |
| `monitor_run` | 按已落盘监控策略执行一次：PSI/CSI/KS 对基准 + 阈值判定 + 告警级别报告（基准分布来源见 §八-6） |

---

## 五、模板与门设计

| 模板 | 步骤骨架（门加粗） | 替代/关系 |
|------|--------------------|-----------|
| `STRATEGY_DEVELOPMENT`（**新 id**） | 输入检查(方向自检) → tradeoff 扫描 → **运营点/分段门** → `design_cutoff_bands` → **分段确认门** → `build_strategy` → `backtest_strategy` → **回测 decision_point** → (可选 `compare_strategies`) → **采纳门** `adopt_strategy` → 交付物 | 新增模板用新 id；**现 `strategy_analysis` id 原样保留**为轻量入口（`strategy_setup.py:51` 与 goal_patterns 路由引用该 id，不得改写原地） |
| `RULE_STRATEGY` | `mine_rules` → `evaluate_rule_set` → **规则集门** → 与分数带组合 `build_strategy` → 回测 → **采纳门** | 新 |
| `PORTFOLIO_ANALYSIS` | vintage(+segment) ∥ roll_rate ∥ flow/migration ∥ segment_profile → 汇总 **decision_point** → `portfolio_report` | 新 id；单步 `vintage_analysis` 保留为轻量入口 |
| `MONITORING_RUN` | `score_dataset` → `monitor_run` → **告警确认门**（红旗 checklist 注入，S-8） | 新；把 DOM-3 监控闭环落地 |
| 即席分析 | 不建模板：driver 单步计划（`slice_aggregate`）+ 口径确认（S-7）；**已拍板：对话 turn 先行（S6），manual 面板扩展作为后续控件皮肤叠加** | — |

两模式覆盖：以上全部同时支持 agent（LLM 话术门）与手动（结构化控件门：分数带滑块、规则勾选表、约束输入框），沿用决策 10 的"控件皮肤"路线。

---

## 六、前端呈现与记忆接入

- **证据面板**（复用 `metric_tables.py` 语言，对齐 7-02 审查 VD-1 的"下沉"方向）：tradeoff 双轴曲线（通过率×坏账率+利润）、分数带色带图、swap 2×2 矩阵卡、规则瀑布条形、vintage 曲线族、迁徙热力、PSI/CSI 趋势 sparkline。
- **报告**：新增"策略"与"组合分析"sheets，进现有 report 渲染管线与导出路径。
- **记忆**（守 INV-4 只读）：新增 `strategy_experience` 记忆 kind（采纳策略的通过率/坏账率/cutoff 区间作历史锚点）；策略门与监控门注入同 scope 历史锚点（复用 7-02 审查 MEM-1 的接线方案）。
- **成功标准**：策略计划支持确定性 `success_criteria`（如 `approved_bad_rate ≤ x`、`approval_rate ≥ y`，值由任务传入不写死），接 AGT-4 已就绪机制。

---

## 七、构建批次与估算

> 前置：7-02 审查 Batch 1 复发债（AGT-1/DOM-1/PERF-2）与 REL-1/UX-1（策略回测也是长任务，同样吃"零反馈/并发"的亏）应先行或并行完成。

| 批次 | 内容 | 验收标准 | 估算 |
|------|------|----------|------|
| **S1a 方向制度化（跨包迁移，独立回归门）** | S-3 渐进式落地。**迁移面清单（全枚举，吸取"三个未修净"教训）**：`strategy.tradeoff_view`、`modeling.reject_inference`（manifest + tool + core，跨包）、`score_dataset`/`monitor_run` 契约预留、`ModelArtifact` + `persist_model_meta` + `.model_meta.json` 读取方（direction 元数据落盘）、`backtest_strategy`/`build_strategy` 规则算子一致性自检（不加参数）；train/tune/select、propose/execute、manual/agent 兄弟路径回归 | 方向反接必触发确认门；规则算子矛盾必告警；全量回归绿 | 4–6 天 |
| **S1b 打分与监控骨架** | `score_dataset`（组装可复用内核）+ `monitor_run` 骨架 + **训练期分数/特征分布快照随 experiment 落盘（基准来源已拍板，契约变更并入 S1a 的 ModelArtifact 迁移面）** | 已训模型可对新数据出分并登记衍生数据集；新训 experiment 携带基准快照 | 2–4 天 |
| **S2 策略开发主线** | tradeoff 升级 + `design_cutoff_bands` + `STRATEGY_DEVELOPMENT` 模板（新 id）+ 采纳门（含 `StrategyRepository` 版本加列）+ 决策表/策略文档交付物 | agent/手动双模式跑通"分数→分段→回测→采纳→导出"全程，每门有红旗 | 5–8 天 |
| **S3 组合分析套件** | **前置：修内核 `cum_bad_rate` 假累计**；新建 `packs/analysis`（已拍板）；vintage/roll-rate 升级（DOM-8 + 内核透传）+ flow/migration + segment_profile + `PORTFOLIO_ANALYSIS` + 表现期数据契约（S-5，含 balance/EAD 列——B5 依赖它）+ **合成表现期数据生成脚本（示例数据已拍板合成，兼解 UX-9 首跑体验）** | 用合成表现数据出一份风控会级组合报告；vintage 累计口径有回归测试 | 5–8 天 |
| **S4 规则策略** | `mine_rules` + `evaluate_rule_set`（瀑布）+ `RULE_STRATEGY` + 规则控件（口径先过 §八-7 拍板） | 规则集与分数带组合成策略并可回测采纳 | 4–6 天 |
| **S5 监控闭环与定期报告** | `MONITORING_RUN` 模板 + score/feature 稳定性趋势 + `expected_loss_estimate`（依赖 S3 契约）+ `portfolio_report` + 告警门 | 对基准跑一次监控出告警报告；监控计划不再是纸面 JSON | 3–5 天 |
| **S6 即席分析 + 额度/定价 + challenger 呈现** | `slice_aggregate` + 口径确认 UX + `limit_pricing_matrix` + `compare_strategies` | 自然语言问数走确定性算子；两策略对比可出 swap 报告 | 5–8 天 |

批次合计 28–45 人日，**日历上约 6–9 周**（与 7-02 审查 Batch 3–6 穿插时更长；穿插建议见 §八-5）。每批次出门前置：该批函数级 spec 先审后写（惯例）。**第一份 spec 建议就是 S1a 方向迁移 spec**（最跨切、风险最高，评审同判）。

---

## 八、拍板结果 + spec 期口径钉子

**已拍板（2026-07-02，用户确认）**：

1. **analysis 工具归属**：✅ **独立 `packs/analysis`**；`slice_aggregate` 放 `data_ops`。
2. **A3 额度/定价**：✅ **本期做**（落 S6，目标是完整全功能平台）。
3. **即席分析入口形态**：✅ **对话 turn 先行**（S6），manual 面板扩展作为后续控件皮肤叠加。
4. **示例表现数据**：✅ **合成生成**（确定性生成脚本，可控 vintage 形状/迁移率/坏账率；进回归测试与首跑体验，兼解 7-02 审查 UX-9）。
5. **与 7-02 审查六批次穿插**：✅ **交替穿插**——审查Batch1 → S1a/S1b → 审查Batch2 → S2/S3 → 审查Batch3 → S4/S5/S6。
6. **`monitor_run` 基准分布**：✅ **训练期快照**（分数/特征分布随 experiment 落盘；契约变更并入 S1a 的 ModelArtifact 迁移面）。

**spec 期口径钉子**（函数级 spec 必须先钉死，防"两个开发者写出两个矿工"）：

7. `mine_rules`：排序键（建议 lift×bad_capture 复合 + 置信下界）、最小支持度下限、重叠规则去重口径、平局的确定性 tie-break。
8. `design_cutoff_bands`：约束不可行时的**显式优先序**（建议：最大坏账率 > 人审容量 > 目标通过率，即先保风险再保容量最后保量），沿 DOM-12 的 `infeasible` 契约。
9. `expected_loss_estimate`：公式固定为 EL ≈ Σ EAD × PD_chain × LGD；EAD 取数口径（期末余额 vs 授信额度）、LGD 参数默认值与来源标注、概率口径与金额口径分列输出。
10. `slice_aggregate` 白名单的具体算子枚举（§二 S-7 已给类目级）。

---

## 九、风险与红线

- **数据可得性**是最大外部风险：没有含状态序列/余额的表现期长表，B1/B2/B5 只能靠合成数据验收（→拍板项 4）。
- **规则挖掘组合爆炸**：硬限深度 ≤2、候选 top-k、确定性排序与随机种子落盘（守 INV-1 可复现）。
- **即席分析滑向"LLM 算数"**：`slice_aggregate` 白名单之外一律拒绝，拒绝时引导用户改问法；LLM 输出只做解读并标注"解读基于上表"。
- **S-3 迁移面**：已改为渐进式（可选参数 + 确定性自检兜底），降低 big-bang 风险；但 S1a 仍须一次枚举全部消费方与兄弟路径回归（train/tune/select、propose/execute、manual/agent），清单见 §七——这是对 7-02 审查"三个未修净"教训的直接应用。
- **不变量不动**：确定性指标只由平台工具算（INV-1）、策略采纳强制确认（INV-3 精神）、记忆只读（INV-4）、子进程隔离（INV-6）、采纳/回测审计同事务（INV-8）。
