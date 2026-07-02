# S4 规则策略（RULE_STRATEGY）— 函数级 Spec

> 状态：待实现。依赖：**S2 先落地**（复用其 adopt_strategy 强制门、策略版本化、决策表交付物、STRATEGY 模板模块）。
> 地面真值（S2 探查）：build_strategy 的 rules 形状 = `[{condition, decision, value}]`、按顺序命中、default_decision 兜底（packs/strategy/tools.py:92-124）；S1a 规则算子方向自检已内建于 build_strategy；门回复结构化解析先例 = dedup 指令 / S2 band_edges。

## 一、验收标准

从"给我挖拒绝规则"到"规则集采纳+文档导出"全程可跑（agent/手动双模式）；挖掘确定性（同 seed 同输出，INV-1）；规则集门支持人工选择子集；每门有 red_flags。

## 二、Commit 1：挖掘与评估内核 + 两个工具（packs/strategy/rules.py + tools.py）

### 内核（纯函数，rules.py）
```
mine_rules(df, *, feature_cols, target_col, max_depth=3, min_support=0.02,
           min_lift=1.5, top_k=20, seed=20260701) -> list[CandidateRule]
  # 双通道：
  # A. 浅决策树路径：DecisionTreeClassifier(max_depth, min_samples_leaf=min_support*n,
  #    random_state=seed) -> 提取通往高坏率叶的路径为合取条件
  # B. 单变量最优切点：逐特征按坏率 lift 扫分位点候选（确定式分位集，无随机）
  # 合并去重（条件等价判定：同特征集+同阈值容差）→ 按 lift 降序稳定排序取 top_k
CandidateRule = {rule_id, clauses:[{feature, op(<,<=,>,>=,==), value}],
                 condition(渲染字符串), support, hit_count, hit_bad_rate, lift,
                 source(tree|univariate)}
evaluate_rule_set(df, rules_ordered, *, target_col, decision="reject")
  -> {waterfall:[{rule_id, incremental_hits, incremental_bad_rate, cum_reject_rate,
                  cum_reject_bad_rate}],
      overlap_matrix(共同命中占比 NxN), residual:{approval_rate, bad_rate},
      combined:{reject_rate, rejected_bad_rate, approved_bad_rate}}
  # 按顺序命中语义与 build_strategy 执行器完全一致（同一条件求值函数，不许重写第二份）
```
- **条件求值统一**：把 build_strategy 现有 rule 条件求值抽为共享函数（若已是独立函数则直接复用），mine/evaluate/build 三方同源——测试锁三方对同一 df 同一规则命中集完全一致。
- clauses→condition 字符串的生成必须能被 build_strategy 的 condition 解析原样接受（往返测试）。

### 工具
```
tool_mine_rules:  入 dataset_id, target_col, feature_cols?(缺省取数值列), max_depth?,
                     min_support?, min_lift?, top_k?, drop_nan_labels
                  出 candidate_rules, n_rows, red_flags, nan_labels_dropped
tool_evaluate_rule_set: 入 dataset_id, target_col, rules(用户选定子集，有序), decision?
                  出 evaluate_rule_set 内核全量输出 + red_flags
```
- red_flags 枚举（测试逐一触发）：`suspect_leakage`（单条规则 lift>10 或 hit_bad_rate>0.9——疑似泄漏特征进规则）、`low_support`（support<min_support 的入选规则）、`rule_shadowed`（waterfall 增量命中=0）、`high_overlap`（两规则重叠>80%）、`nan_labels_dropped`。
- manifest 同步两工具 schema；permissions 不变（read:dataset 已有）。
- 测试：8 行手算数据集——树通道与单变量通道各至少一条规则逐字段断言；确定性（同 seed 两跑 dict 全等）；往返（mine 出的 condition 喂 build_strategy 命中集一致）；五类 red_flags。

## 三、Commit 2：模板、门与接线

### `RULE_STRATEGY` 模板（templates/strategy.py 模块内新增）
```
slots: dataset_id/target_col(task_context) feature_cols?/score_col?(user, optional)
       adoption_reason(user, required)
steps: 1 挖掘规则 mine_rules                       post: nonempty candidate_rules
       2 规则集确认 —— **needs_confirmation**（规则集门：门回复解析「选 1,3,5」/「全选」/
         「去掉 2」式指令→有序子集；解析函数参照 dedup/band_edges 先例）
         实现为轻量 tool_select_rule_set（纯拼装选定子集为 gate payload+透传）
       3 评估规则集 evaluate_rule_set               decision_point  post: nonempty waterfall
       4 构造策略 build_strategy(rules=$ref:3 选定子集 + score_col 存在时可携带分数带规则，
         方向自检自动生效)                          post: nonempty strategy_id
       5 回测策略 backtest_strategy                 **needs_confirmation** + decision_point
       6 采纳策略 adopt_strategy（S2 强制门复用）    **needs_confirmation（强制）**
       7 策略文档 render_strategy_doc               post: nonempty doc_path
goal_patterns: ("规则挖掘","拒绝规则","规则策略","rule mining","rule strategy")
```
- 任务接线：strategy 任务类型的意图路由多认一组 goal patterns（_TurnHandlerSpec 不新增类型；strategy_setup 的 proposal 增加 template_id 选择分支，S2 已开先例）。
- `_render_mine_rules`：候选规则表（规则/支持度/命中坏率/lift/来源）+ red_flags 清单；`_render_evaluate_rule_set`：瀑布表 + 残余通过率摘要 + 重叠告警；规则集门 payload 带勾选语义说明。
- 记忆：采纳路径复用 S2 strategy_experience（cutoff_summary 字段放规则摘要），不新增 kind。
- 测试：模板 instantiate+validate 零错；门回复解析（选/去掉/全选/非法序号 typed 提示）；端到端 agent 旅程（挖掘→选集→评估→构造→回测→采纳→文档）；瀑布与 build_strategy 命中一致性回归。

## 四、非目标
规则热更新/生产规则引擎导出（决策表 CSV 已含规则清单）、跨数据集规则迁移评估（S5 监控面）、前端勾选控件皮肤（门回复文本先行，按拍板留后续）。
