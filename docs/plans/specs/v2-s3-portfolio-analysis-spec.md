# S3 组合分析套件 — 函数级 Spec

> 状态：待实现。依赖排序：**ARCH-9（模板拆分）、ARCH-10（迁移版本化）、ARCH-4（turn handler 参数化）合并之后**；ARCH-8（pack SDK 公共 _Runtime）若先落地则新包直接用公共层，否则按现有 _Runtime 模式复制（strategy/tools.py:215-226 先例）。
> 上游拍板：独立 `packs/analysis` 新包；`slice_aggregate` 归 data_ops（S6，非本批）；NEW-1 真累计已落地（validation/vintage.py:110-138），前置解除。
> 地面真值：vintage_curve/roll_rate_matrix 契约（strategy/tools.py:27-74）；PSI/CSI 确定性内核 compute_psi（modeling monitor_run 复用，S1b）；执行器天然支持无依赖步骤就绪即跑（executor.py:150-158）；sample_data.py 固定 seed=20260701 可扩展；报告拼装先例 = generate_model_report → render_model_report → TransactionalArtifactStore（output/model_report.py:47-84）。

## 一、验收标准

表现期数据从注册到组合报告全链路可跑（agent/手动双模式）；全部指标确定性（INV-1，趋势类与 monitor_run 共用同一 compute_psi 内核）；迁徙/流量矩阵、细分矩阵、趋势、EL 估计、组合报告各有独立回归与手算断言；合成表现数据一键生成。

## 二、Commit 1：表现期数据契约 + 合成数据

### 契约（marvis/data/ 层，不新增表）
- Dataset `role` 词表新增 `"performance"`（registry.py:45 默认词表处集中定义成常量元组，替换散落字符串——只加不改）。
- 表现期表最小列契约（写进 packs/analysis manifest 的 $defs 与校验函数 `validate_performance_frame(df, *, id_col, snapshot_col, bucket_col, balance_col=None)`）：贷款 id、快照月（YYYY-MM 可解析）、逾期桶状态（字符串枚举，桶序由输入 `states` 显式给定，不猜）、余额（可选，缺省时 count 口径）。校验失败→typed error（中文说明缺哪列/哪行不可解析，示例值截断展示）。

### 合成数据（sample_data.py 扩展，UX-9 已有骨架）
```
generate_performance_frame(sample_df, *, n_months=12, seed=20260701) -> DataFrame
  # 对样本表每笔贷款生成逐月快照：余额线性摊还+噪声；逾期桶马尔可夫链
  #（转移矩阵写死在常量，坏样本(y=1)用高恶化矩阵）；确定性：同 seed 字节一致
PERFORMANCE_TABLE_NAME = "表现期快照.csv"  # create_sample_task 同步落此文件并按 role=performance 注册
```
- 测试：字节级确定性；y=1 群体的末月坏桶占比显著高于 y=0（手算阈值断言）；桶序合法。

## 三、Commit 2：packs/analysis 新包 + 四个矩阵/画像工具

### 脚手架
- `marvis/packs/analysis/{manifest.json, tools.py, flow.py, segment.py, trend.py, loss.py, report.py}`；manifest 按 data_ops 词表（name/version/tools/permissions/$defs），permissions=`["read:dataset","write:dataset"]`（趋势工具落衍生数据集）+`"read:experiment"`（读基准快照）。loader.load_builtin_packs 自动发现，无需注册代码。
- `_Runtime`：settings/datasets_root/repo/backend/registry（+experiments 只读）。worker 入口轻量约束（PERF-5）自动满足——entrypoint 走既有 runner。

### `tool_flow_rate`（flow.py）
```
入: dataset_id(role=performance), id_col, snapshot_col, bucket_col, states(有序), balance_col?
出: months(有序), matrix_by_month=[{month, from_to_matrix(NxN 占比), base(count|balance)}],
    net_flows=[{month, into_bad, out_of_bad}], red_flags
```
- 内核 `flow_rate(df,...) -> FlowRateResult`（纯函数，逐相邻月对 id 对齐后统计桶间转移；缺失下月快照的贷款计入 `exited` 伪状态，显式列出不静默丢）。red_flags：`sparse_month`（某月对齐对 <100）、`unknown_bucket`（出现 states 外取值→typed error 而非旗）。

### `tool_bucket_migration`（flow.py）
```
入: 同上 + window(months, 默认全窗口)
出: states, avg_matrix(NxN 平均迁徙率), worst_matrix(逐单元格最差月), heat_table(渲染用行列表), red_flags
```
- 与 flow_rate 共用对齐内核，只聚合口径不同。roll_rate_matrix（strategy 包）保留不动——它是"状态×时间长表"口径，本工具是"相邻快照对齐"口径，文档注明差异。

### `tool_segment_profile`（segment.py）
```
入: dataset_id(样本或打分衍生集), segment_col, target_col?, score_col?, approved_col?,
    profit_params?, ead_col?, pd_col?, top_k=20
出: segments=[{segment, count, pop_pct, approval_rate?, bad_rate?, avg_score?, net_profit?}],
    concentration={top1_pct, top5_pct, hhi}, red_flags
```
- 利润列复用 strategy 包 profit 内核同款公式（不 import 跨包——公式简单，本地实现+两包各自测试锁同一手算值，注释互指）。red_flags：`high_concentration`（top1>40% 或 HHI>0.25）、`sparse_segment`（并入 top_k 外归并为「其他」）。

### `tool_expected_loss_estimate`（loss.py）
```
入: dataset_id(performance), id_col/snapshot_col/bucket_col/states/balance_col(必),
    loss_state(默认 states 末位), lgd=0.6, horizon_months=12
出: chain=[{from_state, p_to_loss(链式乘积)}], el_by_month=[{month, balance, expected_loss}],
    total_el, assumptions{lgd, horizon, matrix_window}, red_flags
```
- 链式近似：用 bucket_migration 的 avg_matrix 求各状态到 loss_state 的 horizon 步吸收概率（马尔可夫吸收链，确定性线性代数，无迭代随机）。red_flags：`matrix_not_absorbing`（loss 态非吸收→按吸收强制处理并示警）、`short_history`（可用月 < 3）。
- 测试：3 状态手算吸收概率逐值断言。

## 四、Commit 3：趋势工具（依赖 score_dataset 衍生集）

### `tool_score_stability_trend` / `tool_feature_csi_trend`（trend.py）
```
入: experiment_id(读基准快照 baseline_distributions), dataset_ids=[打分衍生集] 或
    dataset_id+month_col(单表按月切), score_col, feature_cols?(csi 用), thresholds?
出: trend=[{month, psi|max_csi, level(green|amber|red), sample_count}],
    per_feature_trend?(csi: 前 10 特征逐月), red_flags
```
- **INV-1 硬约束**：PSI/CSI 复用 monitor_run 的同一 `compute_psi`/`bin_distribution` 内核（实现时定位其模块；若内核私有在 modeling tools 内，则把纯函数上提到共享模块 `marvis/metrics_kernels.py` 之类，modeling 与 analysis 都 import——上提是纯搬家，S1b 的六项回归保护）。阈值默认沿用 MONITOR_RUN_THRESHOLDS。
- red_flags：`missing_baseline`（experiment 无快照→typed error）、`month_gap`（趋势月份不连续）。

## 五、Commit 4：PORTFOLIO_ANALYSIS 模板 + portfolio_report + 接线

### `tool_portfolio_report`（report.py）
```
入: task_id 语境 + 各前序步骤输出（$ref 注入：flow/migration/segment/trend/el 的结果 dict）+ project_meta?
出: report_path(xlsx), sheets=[...], artifact 登记
```
- 拼装完全对照 model_report 先例：`PortfolioReportPayload` dataclass → `render_portfolio_report(payload, path)`（marvis/output/portfolio_report.py 新模块，Workbook + TransactionalArtifactStore stage/promote/commit）。Sheet：组合概览、桶迁徙（矩阵）、逐月流量、细分画像、稳定性趋势、预期损失、数据质量红旗汇总。**报告只搬运前序步骤已持久化的数字，不重算**（INV-1）。

### `PORTFOLIO_ANALYSIS` 模板（新 id；vintage_analysis 原样保留为轻量入口）
```
slots: performance_dataset_id/id_col/snapshot_col/bucket_col/states(task_context)
       segment_col?/score_col?/experiment_id?(user, optional)
steps: 1 流量分析 flow_rate ∥ 2 迁徙热力 bucket_migration ∥ 3 细分画像 segment_profile
       ∥ 4 稳定性趋势 score_stability_trend（experiment_id 缺省→planner 剪步）
       5 损失估计 expected_loss_estimate（depends: 迁徙热力）
       6 组合分析汇总 —— depends 全部前步，decision_point + needs_confirmation
         （门文案聚合各步 red_flags，红旗清单=门 checklist）
       7 生成组合报告 portfolio_report（depends: 汇总）post: nonempty report_path
```
- 步骤 1-4 无互依赖，executor 既有就绪即跑语义天然并行（executor.py:150-158，无需新机制）。
- 汇总门用一个轻量 `tool_portfolio_gate_summary`（analysis 包内，纯拼装各 $ref 的 red_flags 与关键数字为 gate payload）——先例：MONITORING_RUN 的告警门。
- goal_patterns: ("组合分析","组合报告","资产质量","portfolio analysis")；与 vintage/strategy 模式集不相交。

### 任务接线（ARCH-4 参数化后的 turn spec 新条目）
- 新任务类型 `portfolio`（run_mode/agent 双模式），setup=`portfolio_setup.py`：探测 performance 表（role 或列命中），推断 id/snapshot/bucket 列（hints 常量），bucket states 从数据枚举+按恶化度排序提示用户在 C1 式确认门核对（**states 顺序必须人工确认**——语义顺序机器不可猜，红旗门）。
- 渲染器：`_render_flow_rate`/`_render_bucket_migration`（NxN 热力表——metric_tables 新增 `matrix-heat` kind，前端按单元格值染色，复用 percent-heat 色标）/`_render_segment_profile`/`_render_el_estimate`/`_render_portfolio_report`；vintage renderer 不动。
- `_portfolio_success_criteria(task)`：可选 `portfolio_el_max`。
- 记忆：不新增 kind（组合分析非"经验"型产出；策略采纳的 strategy_experience 已覆盖），门注入沿用 scope 锚点。

### 测试清单（tests/test_analysis_pack.py + test_portfolio_api.py）
1. 契约校验：缺列/坏月格式/未知桶 typed error 文案。
2. flow/migration：4 贷款×3 月手算矩阵逐值断言；exited 伪状态计数；sparse red flag。
3. segment_profile：手算 HHI/top1；「其他」归并。
4. EL：3 状态吸收链手算；非吸收示警。
5. 趋势：与 monitor_run 同输入同 PSI 断言（内核一致性回归）；缺基准 typed error。
6. 模板：instantiate+validate 零错；1-4 步无依赖断言；剪步（无 experiment_id）；汇总门聚合 red_flags。
7. 端到端：合成表现数据→portfolio 任务→states 确认→并行分析→汇总门→报告落盘+artifact 登记+审计行全查。
8. 报告：xlsx sheet 清单与数字搬运一致性（不重算断言：改 payload 数字报告跟着变）。

## 六、非目标
规则挖掘（S4）、监控闭环消费（S5）、slice_aggregate/limit_pricing（S6）、前端图表组件（矩阵热力用表格染色，曲线图皮肤留 VD 后续）、跨包 import profit 内核（本地实现+双向手算锁）。
