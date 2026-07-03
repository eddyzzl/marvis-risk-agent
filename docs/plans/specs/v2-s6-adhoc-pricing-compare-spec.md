# S6 即席分析 + 额度/定价 + challenger 呈现 — 函数级 Spec

> 状态：待实现。依赖：S2（compare_strategies 工具与采纳面已有）、S3（segment 口径先例）。拍板：`slice_aggregate` 归 **data_ops**；即席分析**对话 turn 先行**，manual 面板皮肤后续；A3 额度/定价本期做。

## 一、验收标准

自然语言问数（"按渠道看 5 月坏率"）走确定性白名单算子出表+口径确认，LLM 只做意图解析不算数（INV-1）；额度×定价矩阵含 EL 模拟可出确认门；两策略对比出 swap 报告呈现（复用 S2 compare_strategies，本批补呈现面）。

## 二、Commit 1：`slice_aggregate`（data_ops 包）+ 口径确认

### 工具
```
tool_slice_aggregate:
入: dataset_id, group_by=[col…]（≤3）, metrics=[{op, col?}…]（op 白名单:
    count|sum|mean|min|max|bad_rate(col=target)|approval_rate(col=decision)|distinct）,
    filters=[{col, op(==|!=|>|>=|<|<=|in|between), value}]（≤8）, month_col?+months?,
    top_k=50, sort_by?
出: rows(聚合表), spec_echo(全部口径原样回显), n_rows_scanned, red_flags
```
- 实现：全部编译成单条 DuckDB SQL（参数化绑定，列名经 quote_identifier 白名单校验——列必须在数据集 profile 里存在，杜绝注入；op 映射写死字典）。空结果/超 top_k 截断→red_flags（`empty_result`/`truncated`）。
- 确定性：ORDER BY 显式（sort_by 缺省→group_by 字典序），同输入同输出。
- 测试：手算聚合断言；注入尝试（列名带 `; DROP`）→typed error；截断旗；between/in 边界。

### 对话 turn 接线（即席分析无模板，driver 单步计划）
- `agent/adhoc_analysis.py`：`build_slice_spec_from_utterance(utterance, dataset_profile, llm) -> SliceSpec`——LLM 产出结构化 spec（新 PromptSpec 进 llm_prompts 注册表，新增不动 hash 锁），平台校验列存在/op 白名单后**回显口径确认门**（"将按〔渠道〕分组统计〔坏率〕，范围〔2026-05〕，确认？"），确认后才执行 slice_aggregate 单步计划。解析失败/列不存在→中文澄清问句，不猜。
- 路由：数据集就绪的任务里，问数类 utterance（"看一下/统计/分布/多少"启发词 + LLM 意图）走此支线；现有 turn 流不受影响（新增意图分支，防御式默认走原路）。
- 测试：spec 校验拒绝幻觉列；确认门先行（未确认不执行）；审计行（kind='data.slice_aggregate'）。

## 三、Commit 2：`limit_pricing_matrix`（strategy 包，A3）

```
入: dataset_id, score_col, target_col?, pd_col?（无则用分数带经验坏率作 PD 代理并示警）,
    ead_col?, band_edges?（缺省调 design_cutoff_bands 内核取带）,
    limit_grid=[额度档], rate_grid=[年化档], lgd=0.6, funding_rate, term_months, cost_per_loan
出: matrix=[{band, limit, rate, expected_profit, el, roa, feasible}],
    recommended=[{band, limit, rate}](每带利润最大可行档),
    assumptions{…全部入参回显…}, red_flags
```
- EL/利润公式与 profit 内核同源（同一函数或双向手算锁，S3 同款约定）；`feasible` = roa≥0 且 el/ead≤阈值。red_flags：`pd_proxy_used`、`negative_profit_band`（整带无可行档）。
- 模板：不新增——作为 STRATEGY_DEVELOPMENT 的可选后置步（slot 触发）+ 独立单步 driver 计划（即席同款：**矩阵确认门**后落 strategy_artifacts(kind='limit_pricing_csv')）。
- 渲染 `_render_limit_pricing_matrix`：band×limit×rate 透视表（行=band，列=limit/rate 组合，值=利润，负值红染），recommended 行置顶。
- 测试：2 带×2 额×2 价手算矩阵逐值；PD 代理旗；确认门后才落 artifact。

## 四、Commit 3：challenger 对比呈现

- `_render_compare_strategies` 升级（S2 已有基础版）：swap 2×2 矩阵卡（metric_tables `matrix-heat`，S3 引入的 kind 复用）、双策略关键指标并排表、结论行（"挑战者在通过率 +x pp 下坏率 −y pp"模板化中文，数值全来自工具输出）。
- `tool_render_challenger_report`：compare 输出 + 双方 backtest + 采纳状态拼 Markdown 报告，登记 strategy_artifacts(kind='challenger_report_md')；审计。
- STRATEGY_DEVELOPMENT 的可选步 5（compare）后接可选报告步（champion slot 存在时 planner 保留）。
- 测试：报告数字与 compare 输出一致性（改输出报告跟变）；无 champion 时剪步不出报告。

## 五、非目标
manual 面板问数控件（皮肤后续）、定价优化求解器（网格枚举即可）、多期动态定价、外部利率数据接入。
