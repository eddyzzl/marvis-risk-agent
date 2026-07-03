# S2 策略开发主线 — 函数级 Spec

> 状态：待实现。依赖：**必须排在 ARCH-9（templates/sample.py 拆分）与 ARCH-10（db_schema 迁移版本化）合并之后**——两者正在重构本 spec 要改的文件。
> 上游拍板（策略计划 §七）：新模板用新 id `STRATEGY_DEVELOPMENT`，现 `strategy_analysis` 原样保留为轻量入口；采纳门为强制确认门；本 spec 遵循 S1a 已落地的 direction 机制。
> 地面真值基线：strategies 表无 version 列（db_schema.py:584-593）；StrategyRepository 已有 `*_with_audit` 先例（repositories/strategy.py:34-44/92-111）；tradeoff_view 已带 score_direction+direction_diagnostics（packs/strategy/tools.py:171-212）；MONITORING_RUN 是告警确认门参照（templates sample.py:1158-1210）。

## 一、验收标准（照抄计划，不降级）

agent/手动双模式跑通「分数→分段→回测→采纳→导出」全程；每个门有红旗（red_flags 字段驱动，非文案装饰）；全量回归绿。

## 二、Commit 1：策略版本化与采纳持久化

### db_schema（走 ARCH-10 的新迁移清单，新增一个版本号迁移）
- `strategies` 加列：`version INTEGER NOT NULL DEFAULT 1`、`status TEXT NOT NULL DEFAULT 'draft'`（draft|adopted|retired）、`adopted_at TEXT`、`adoption_reason TEXT`、`parent_strategy_id TEXT`（迭代谱系，可 NULL）。
- 新表 `strategy_artifacts`：`id TEXT PK, strategy_id TEXT NOT NULL REFERENCES strategies(id) ON DELETE CASCADE, kind TEXT NOT NULL`（decision_table_csv|strategy_doc_md|monitoring_plan_json）`, path TEXT NOT NULL, created_at TEXT NOT NULL`。
- 旧库升级测试：造旧 schema 库 → init_db → 断言新列默认值就位、既有策略行 status='draft'。

### StrategyRepository 新方法（全部带 audit 版本，复用 _write_audit_row）
```
adopt_strategy_with_audit(strategy_id, *, reason, audit, adopted_at=None) -> Strategy
  # 原子一次性转移：UPDATE strategies SET status='adopted', adopted_at=?, adoption_reason=?
  #   WHERE id=? AND status='draft'；rowcount==0 -> ConflictError（吸取 confirm_step 双确认教训）
  # 同事务内：把同 (task_id, strategy_type) 下其他 status='adopted' 行置 'retired'
  #   （每任务每类型仅一个在役策略；被退役的写独立 audit 行 kind='strategy.retire'）
new_version_from(strategy_id, *, rules=None, description=None) -> Strategy
  # 克隆为 version=max(version)+1 的 draft，parent_strategy_id=源 id；rules 可覆盖
save_strategy_artifact(strategy_id, *, kind, path, created_at=None) -> artifact_id
list_strategy_artifacts(strategy_id) -> list[dict]
```
- `get_strategy`/`list_for_task` 返回体补 version/status/adopted_at/parent_strategy_id。
- 测试：双采纳→第二次 ConflictError；采纳 B 后 A 自动 retired 且有 retire 审计行；new_version_from 谱系与版本号；旧库升级。

## 三、Commit 2：分析工具 —— design_cutoff_bands + tradeoff 升级 + compare_strategies

### `tool_design_cutoff_bands`（新，manifest 权限沿用 read:dataset）
```
入: dataset_id, score_col, target_col, score_direction?, n_bands=5, band_edges?(手动覆盖),
    objective(max_profit|max_approval), max_bad_rate?, min_approval_rate?,
    profit_params?, ead_col?, pd_col?, drop_nan_labels(接 NaN 标签门既有旗)
出: bands=[{lo, hi, count, pop_pct, bad_rate, cum_approval_rate, cum_bad_rate, expected_profit,
           decision(approve|review|decline)}],
    recommended_rules(可直接喂 build_strategy 的 rules 数组), band_edges,
    red_flags=[{code, level(red|amber), message}], score_direction, nan_labels_dropped
```
- 内部：`normalize_score_direction` + `check_score_direction`（S1a 原语，方向冲突→red flag + 需 confirm_direction_conflict 才继续，复用 tradeoff_view 的既有模式 tools.py:199-211）；无手动 edges 时按分位数取候选边界再按 objective+约束扫描；决策分配 = 按方向单调切 approve/review/decline 三段（review 段可空）。
- red_flags 枚举（写死 code，测试逐一触发）：`direction_conflict`、`nonmonotonic_bad_rate`（相邻带坏率逆序）、`sparse_band`（pop_pct<2%）、`infeasible_constraints`（无满足 max_bad_rate/min_approval_rate 的切法——此时仍返回最接近解+red）、`nan_labels_dropped`。
- 确定性（INV-1）：同输入同输出，边界用分位数确定式计算，禁止随机。

### `tool_tradeoff_view` 升级（不破坏既有出参）
- points 逐点补 `feasible: bool`（对 max_bad_rate/min_approval_rate 约束）；新增可选入参 `min_approval_rate`；`recommended` 只从 feasible 集选，全不可行→recommended=null + red_flags=[infeasible_constraints]。
- 出参新增 `red_flags`（沿用上面 code 枚举）；渲染器同步（见 Commit 3）。

### `tool_compare_strategies`（新）
```
入: dataset_id, target_col, strategy_id, baseline_strategy_id（二者都必须已有该 dataset 的回测，
    没有则先内部跑 backtest 核心复用 _backtest 实现，不落新 backtest 行）
出: matrix_2x2={both_approve:{count,bad_rate}, only_new, only_baseline, both_decline},
    deltas={approval_rate, approved_bad_rate, expected_profit}, summary_text, red_flags
```
- red_flags：`swap_in_worse`（swap-in 坏率 > swap-out 坏率）、`profit_negative_delta`。
- 模板里挂 decision_point（非强制门）。

## 四、Commit 3：采纳与交付物 —— adopt_strategy + render_strategy_doc + 渲染器

### `tool_adopt_strategy`（**强制确认门后执行**）
```
入: strategy_id, backtest_id(必须属于该 strategy), adoption_reason
出: strategy_id, version, status='adopted', retired_strategy_ids, artifacts=[{kind, path}]
```
- 流程：校验 backtest 归属（不符→typed error）→ `adopt_strategy_with_audit`（kind='strategy.adopt'，detail 含 backtest 关键指标）→ 生成两件交付物并 `save_strategy_artifact` 登记 + 各写一条 audit：
  - 决策表 CSV（`decision_table_csv`）：列 = 序号/条件/决策/取值/band 区间/样本占比/坏率/预期利润（band 统计来自最近一次 design_cutoff_bands 输出，经 inputs 传入或从 rules 反推——**取 inputs 传入**：模板用 $ref 接线，见 §五）。
  - 监控计划 JSON（`monitoring_plan_json`）：复用 MONITOR_RUN_THRESHOLDS 结构，S5 闭环消费。
- 文件落 `workspace/tasks/<task_id>/strategy/` 下，文件名带 strategy_id+version。

### `tool_render_strategy_doc`
```
入: strategy_id（读 strategy+其 backtests+artifacts）
出: doc_path(strategy_doc_md 交付物，登记 strategy_artifacts), sections=[...]
```
- Markdown 章节：策略概览（类型/版本/状态/谱系）、规则清单表、回测摘要（含 swap）、band 表、红旗与处置记录、监控计划摘要。中文文案，指标数值全部来自持久化结果（INV-1：文档不重算）。

### 渲染器（agent/renderers.py，对照 _render_tradeoff_view:601-630 风格）
- `_render_design_cutoff_bands`：文案首行=推荐切法+红旗计数；表1=分数带（databar/percent-heat 语言对齐 metric_tables COLUMN_SPEC）；表2=红旗清单（等级/说明）。红旗非空时文案显式列 red 项——这是「每门有红旗」的落点。
- `_render_compare_strategies`：2×2 矩阵表 + delta 摘要行。
- `_render_adopt_strategy`：版本/状态/退役清单/交付物路径表。
- `_render_strategy_doc`：路径+章节清单。
- registry 注册四条 tool→renderer 映射；tradeoff 渲染器补 red_flags/feasible 呈现。

## 五、Commit 4：模板、驱动接线、记忆与端到端

### `STRATEGY_DEVELOPMENT` 模板（新 id；放入 ARCH-9 拆分后的 strategy 模板模块）
```
slots: dataset_id/target_col/score_col(task_context, required)
       score_direction(task_context, optional)   # 有模型工件时上游注入
       objective/max_bad_rate/min_approval_rate/profit_params(user, optional)
       strategy_type(user, default approval), adoption_reason(user, required)
steps（门加粗）:
 1 权衡扫描        tradeoff_view(带约束)          decision_point   # 方向自检在此触发（S1a 机制）
 2 设计分数带      design_cutoff_bands            **needs_confirmation**  post: nonempty bands
 3 构造策略        build_strategy(rules=$ref:2.output.recommended_rules)  post: nonempty strategy_id
 4 回测策略        backtest_strategy              **needs_confirmation** + decision_point  post: range 检查（沿用 strategy_analysis L1091-1108）
 5 对比基线(可选)   compare_strategies             decision_point   # baseline_strategy_id slot 缺省时 planner 剪掉该步
 6 采纳策略        adopt_strategy(backtest_id=$ref:4.output.backtest_id, band 统计=$ref:2)  **needs_confirmation（强制，auto-accept 不放行，遵循交付门先例）**
 7 策略文档        render_strategy_doc            post: nonempty doc_path
goal_patterns: ("策略开发","开发策略","设计cutoff","分数带策略","strategy development")  # 与 strategy_analysis 的模式集不相交
```
- `strategy_setup.py:51` 不改默认；新增 `StrategyProposal.template_id` 可为 "strategy_development"（由意图路由选择，轻量入口继续走 strategy_analysis）。
- success_criteria：turn_handlers 加 `_strategy_success_criteria(task)`（镜像 _modeling_success_criteria:832-848）：task 可选字段 `strategy_bad_rate_max`/`strategy_approval_min` → `[{"metric":"approved_bad_rate","max":..},{"metric":"approval_rate","min":..}]`。

### 记忆（INV-4 只读注入不变）
- `agent_memory/models.py` MEMORY_TYPES 加 `strategy_experience`；必填字段：`strategy_type, cutoff_summary, approval_rate, approved_bad_rate, expected_profit, scope, source_task_id`。
- `policy.py` PAYLOAD_FIELD_ALLOWLISTS 加同名白名单；`extractors.py` 加 `extract_strategy_experience()`（从 adopt 输出+回测组装）；`memory_bridge.py` 加 `_capture_strategy_experience()`，仅在采纳成功后触发，走既有两条 auto_distill 门控（**api.py 与 pipeline.py 两面都查**，见记忆策略既有坑）。
- 门注入：分数带门与采纳门注入同 scope 历史锚点（MEM-1 bridge 既有接线面）。

### 手动模式
- 分段确认门支持结构化覆盖：门回复解析 `band_edges=[...]`（参照 _parse_dedup_instruction:72-88 的解析先例）→ 以覆盖边界重跑 design_cutoff_bands 后再过门。滑块类控件皮肤按拍板留后续批次，本期不做前端新控件。

### 测试清单（新文件 tests/test_strategy_development.py + 分散补齐）
1. repository：双采纳 Conflict、自动退役+审计、版本谱系、旧库升级（并入 ARCH-10 迁移测试面）。
2. design_cutoff_bands：6 行手算小数据集边界/坏率/利润逐值断言；五类 red_flags 各一条触发用例；band_edges 手动覆盖；方向冲突门。
3. tradeoff 升级：feasible 过滤、全不可行→recommended null+red。
4. compare：2×2 计数手算断言、swap_in_worse 触发。
5. adopt：backtest 归属校验、交付物落盘+登记+audit 三连、决策表 CSV 内容抽查。
6. 模板：instantiate+validate 零错、步骤/门/$ref 接线断言（进 test_orch_templates）、goal_patterns 路由不串 strategy_analysis。
7. 端到端（test_agent_task_routing 风格）：agent 模式全程「扫描→分段门→回测门→采纳门→文档」，含一次手动 band_edges 覆盖回合；success_criteria 通过/不通过两路径。
8. 记忆：采纳后 strategy_experience 落库、白名单外字段被拒、未采纳不写。
9. 渲染：四个新 renderer 的表结构断言。

## 六、非目标（本批不做）
- 规则挖掘（S4 RULE_STRATEGY）、组合分析（S3）、监控闭环消费 monitoring_plan（S5）、slice_aggregate/limit_pricing（S6）、前端滑块控件皮肤、PORTFOLIO 报表。
