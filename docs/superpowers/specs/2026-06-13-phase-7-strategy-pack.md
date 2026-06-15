# Phase 7 — Vintage / 盈利测算 / 策略回测包（函数级 spec，含内部伪代码）

## 文档状态

- 状态：待实施
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 15.2 节）
- 前置依赖：Phase 1（Tool 契约）、Phase 3（数据层）、Phase 4（指标核心）、Phase 4V（共享 vintage/roll-rate 核心 `validation/vintage.py`，策略包**复用不重写**）、Phase 6（模型分可作策略输入）
- 目标：交付策略与组合管理能力包——vintage 曲线、滚动率、盈利测算、策略生成（额度/定价/准入/拒绝/分群）、策略回测、风险收益权衡视图。

## 捍卫的不变量

- **INV-1/INV-2**：vintage/滚动率/盈利/回测的每个数字由 tool 代码算，LLM 不算（LLM 可解释策略、起草说明，但通过率/坏账率/利润是平台算的）。
- **INV-5**：策略报告不落客户明细，只落分群统计。
- **roadmap V4 约束**：关键流程保留手动模式；策略上线类动作需人工确认。
- 确定性：盈利/回测给定输入和参数，结果可复现（无随机）。

## 模块布局

```text
marvis/packs/strategy/
  __init__.py
  manifest.json
  contracts.py     VintageCurve / RollRateMatrix / ProfitResult / Strategy / BacktestResult / TradeoffPoint
  errors.py
  vintage.py       vintage_curve（薄适配：调 Phase 4V compute_vintage_curve + vintage_curve_wide，包成策略视图）
  roll_rate.py     roll_rate_matrix（薄包装：从客户状态时序推 from/to bucket，再调 Phase 4V compute_roll_rate）
  profit.py        profit_calc（收入-损失-成本）
  strategy.py      build_strategy（规则构造）+ 策略类型
  backtest.py      backtest_strategy（历史回放）
  tradeoff.py      tradeoff_view（风险收益前沿）
  tools.py
marvis/db.py   新增 strategies / backtests 表
```

新增依赖：Phase 4V 共享核心 `validation/vintage.py`（`compute_vintage_curve` / `vintage_curve_wide` / `compute_roll_rate`）。vintage/roll-rate 的确定性计算**不在本包重写**，只做策略侧适配与解读。

---

## Part A — 契约（`packs/strategy/contracts.py`）

```python
# VintageCurve / RollRateMatrix 是策略侧的"视图"类型，由 Phase 4V 核心输出包装而来
# （vintage_curve_wide / compute_roll_rate），不承载独立的计算口径。
@dataclass(frozen=True)
class VintageCurve:
    cohort_col: str                 # 放款月份/起始账期列
    mob_max: int                    # 最大账龄（months on book）
    cohorts: tuple[str, ...]        # 各 cohort 标识（如 "202401"）
    curves: dict                    # cohort -> [每个 MOB 的累计坏账率]（来自 vintage_curve_wide）
    counts: dict                    # cohort -> 该 cohort 样本量

@dataclass(frozen=True)
class RollRateMatrix:
    states: tuple[str, ...]         # 如 ("C","M1","M2","M3","M4+")
    matrix: tuple[tuple[float, ...], ...]   # 转移概率方阵 P[from][to]
    period: str                     # 观察周期（如 "month"）
    base_counts: dict               # 各起始状态样本量

@dataclass(frozen=True)
class ProfitResult:
    segment: str
    count: int
    revenue: float                  # 利息+费用
    expected_loss: float            # PD*LGD*EAD
    funding_cost: float
    operating_cost: float
    net_profit: float
    roa: float                      # net_profit / EAD

@dataclass(frozen=True)
class StrategyRule:
    condition: str                  # 表达式，如 "score < 600" / "feature_x in [a,b]"
    decision: str                   # approve|reject|limit|price|segment
    value: object                   # 决策值（额度/利率/分群名；approve/reject 为 None）

@dataclass(frozen=True)
class Strategy:
    id: str
    strategy_type: str              # approval|limit|pricing|reject|segmentation
    rules: tuple[StrategyRule, ...]
    score_col: str | None
    default_decision: object        # 未命中任何规则时的兜底
    description: str

@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    approval_rate: float
    approved_count: int
    approved_bad_rate: float
    rejected_bad_rate: float        # 被拒群体的坏账率（评估误拒）
    expected_profit: float
    swap_in_count: int              # 相对 baseline 新通过的
    swap_out_count: int             # 相对 baseline 新拒绝的
    swap_in_bad_rate: float
    swap_out_bad_rate: float
    by_segment: tuple[dict, ...]

@dataclass(frozen=True)
class TradeoffPoint:
    cutoff: float
    approval_rate: float
    bad_rate: float                 # 通过群体坏账率
    expected_profit: float
```

- **测试要点**：dataclass 往返。

---

## Part B — Vintage 曲线（`packs/strategy/vintage.py`）

```python
def vintage_curve(df: pd.DataFrame, *, cohort_col: str, mob_col: str,
                  bad_col: str, mob_max: int = 12) -> VintageCurve:
    """账龄分析（**薄适配**）：委托 Phase 4V 共享核心算累计坏账率，再包成策略侧 VintageCurve。
    **不在此重算**——累计坏账率口径、单调性、分母安全全由 `validation/vintage.py` 保证。
    入参: df; cohort_col 放款月份; mob_col 账龄（月）; bad_col 0/1 是否坏; mob_max（按需截断）。
    出参: VintageCurve（各 cohort 的累计坏账率序列）。
    伪代码:
      from marvis.validation.vintage import compute_vintage_curve, vintage_curve_wide
      points = compute_vintage_curve(df, cohort_col=cohort_col, mob_col=mob_col, target_col=bad_col)
      wide = vintage_curve_wide(points, metric="cum_bad_rate")   # {cohort: [按 MOB 升序累计坏率]}
      curves = {c: _truncate_or_pad(vals, mob_max) for c, vals in wide.items()}
      counts = {p.cohort: p.sample_count for p in points if p.mob == min_mob_of(p.cohort, points)}
      return VintageCurve(cohort_col, mob_max, tuple(sorted(curves)), curves, counts)
    """

def vintage_summary(curve: VintageCurve, *, ref_mob: int = 6) -> dict:
    """vintage 对比摘要：各 cohort 在参考账龄(如 MOB6)的坏账率，识别恶化趋势。
    出参: {cohort: bad_rate_at_ref}, {trend: "deteriorating"|"stable"|"improving"}。
    伪代码:
      at_ref = {c: curve.curves[c][ref_mob-1] for c in curve.cohorts if len(curve.curves[c])>=ref_mob}
      trend = _trend(list(at_ref.values()))   # 按 cohort 时间序看斜率
      return {"at_ref": at_ref, "trend": trend}
    """
```

- **测试要点**：`vintage_curve` 正确委托 4V 并包成 VintageCurve（曲线值与 `vintage_curve_wide` 一致、按 mob_max 截断/补齐）；`vintage_summary` 恶化趋势识别（新 cohort 坏账率上行）；空 cohort 兜底。（累计单调、分母安全由 Phase 4V 核心测试覆盖，不在此重测。）

---

## Part C — 滚动率（`packs/strategy/roll_rate.py`）

```python
def roll_rate_matrix(df: pd.DataFrame, *, id_col: str, time_col: str,
                     status_col: str, states: list[str]) -> RollRateMatrix:
    """逾期状态迁移矩阵：相邻周期 P(下期状态 | 本期状态)。
    **策略侧只负责"从客户状态时序推出相邻周期的 (from, to) bucket 对"这一步**（本包独有），
    迁移计数/行归一化复用 Phase 4V `compute_roll_rate`，不在此重写矩阵口径。
    入参: df; id_col 客户; time_col 周期; status_col 逾期状态; states 状态顺序（如 C/M1/M2/M3/M4+）。
    出参: RollRateMatrix（转移概率方阵）。
    不变量: 每行转移概率和≈1（由 4V 的 per-from-bucket 归一化保证）。
    伪代码:
      from marvis.validation.vintage import compute_roll_rate
      df = df.sort_values([id_col, time_col])
      # 本包独有：构造每个客户相邻周期 (from_state, to_state) 明细
      rows = []
      for cid, g in df.groupby(id_col):
          s = g[status_col].tolist()
          rows += [{"from": a, "to": b} for a, b in zip(s[:-1], s[1:])]
      pairs_df = pd.DataFrame(rows)
      # 迁移率由共享核心算（buckets=states，覆盖完整矩阵、每行归一）
      points = compute_roll_rate(pairs_df, from_bucket_col="from", to_bucket_col="to", buckets=tuple(states))
      matrix, base = _points_to_matrix(points, states)   # 把 tidy RollRatePoint 摆回 len×len 方阵 + base_counts
      return RollRateMatrix(tuple(states), matrix, "month", base)
    """
```

- **测试要点**：时序→相邻 (from,to) 对构造正确（按 id+time 排序、跨客户不串期）；`_points_to_matrix` 把 4V tidy 点摆回 len×len 方阵且状态顺序保持；吸收态自转移=1。（每行归一≈1、空行全 0 由 Phase 4V 核心测试覆盖。）

---

## Part D — 盈利测算（`packs/strategy/profit.py`）

```python
@dataclass(frozen=True)
class ProfitParams:
    annual_rate: float          # 年化利率（收入）
    funding_rate: float         # 资金成本率
    lgd: float                  # 违约损失率
    operating_cost_per_loan: float
    term_months: int            # 期限

def profit_calc(df: pd.DataFrame, *, segment_col: str | None, ead_col: str,
               pd_col: str, params: ProfitParams) -> list[ProfitResult]:
    """分群盈利测算：收入 - 预期损失 - 资金成本 - 运营成本。
    入参: df; segment_col 分群列（None=整体）; ead_col 敞口; pd_col 违约概率; params。
    出参: 各分群 ProfitResult。
    不变量: INV-1——盈利公式确定性；PD 来自模型分/标签，不由 LLM 估。
    伪代码:
      groups = df.groupby(segment_col) if segment_col else [("all", df)]
      out = []
      for seg, g in groups:
          ead = g[ead_col].sum()
          revenue = (g[ead_col] * params.annual_rate * params.term_months/12).sum()
          expected_loss = (g[ead_col] * g[pd_col] * params.lgd).sum()
          funding = (g[ead_col] * params.funding_rate * params.term_months/12).sum()
          opcost = len(g) * params.operating_cost_per_loan
          net = revenue - expected_loss - funding - opcost
          out.append(ProfitResult(str(seg), len(g), revenue, expected_loss, funding, opcost,
                                  net, roa=(net/ead if ead else 0.0)))
      return out
    """

def vintage_profit(df, *, cohort_col, ead_col, pd_col, params) -> dict:
    """按 vintage cohort 维度的盈利（结合账龄看 cohort 盈利演化）。
    出参: {cohort: ProfitResult}。
    """
```

- **测试要点**：盈利公式手算对账；分群/整体；vintage 维度；ROA 计算；零敞口兜底。

---

## Part E — 策略构造（`packs/strategy/strategy.py`）

```python
def build_strategy(strategy_type: str, rules: list[dict], *, score_col: str | None,
                  default_decision, description: str = "") -> Strategy:
    """构造策略对象，校验规则合法性。
    入参: strategy_type（approval|limit|pricing|reject|segmentation）; rules（条件→决策）;
          score_col; default_decision 兜底; description。
    出参: Strategy。
    异常: StrategyError（规则条件非法 / 决策类型与策略类型不符）。
    不变量: 规则条件是受限表达式（白名单字段+运算符），不执行任意代码。
    伪代码:
      parsed = []
      for r in rules:
          _validate_condition(r["condition"])   # 只允许 field op value / and / or / in，白名单字段
          _validate_decision(strategy_type, r["decision"], r.get("value"))
          parsed.append(StrategyRule(r["condition"], r["decision"], r.get("value")))
      return Strategy(id=_new_id(), strategy_type=strategy_type, rules=tuple(parsed),
                      score_col=score_col, default_decision=default_decision, description=description)

def apply_strategy(df: pd.DataFrame, strategy: Strategy) -> pd.Series:
    """把策略应用到数据，返回每行的决策。
    出参: pd.Series（每行 decision/value）。
    不变量: 规则按顺序匹配，首个命中生效；都不命中用 default_decision。条件经安全求值（非 eval 任意代码）。
    伪代码:
      decisions = pd.Series([strategy.default_decision]*len(df), index=df.index)
      assigned = pd.Series(False, index=df.index)
      for rule in strategy.rules:
          mask = _safe_eval_condition(df, rule.condition) & (~assigned)
          decisions[mask] = rule.value if rule.value is not None else rule.decision
          assigned |= mask
      return decisions
    """
```

辅助 `_safe_eval_condition(df, condition)`：把受限表达式解析成 pandas 布尔掩码（白名单字段 + `< > <= >= == != in and or`），**不用 `eval`**（防注入）。

- **测试要点**：规则构造校验（非法条件/决策不符抛错）；apply 首个命中生效；兜底决策；**安全求值不执行任意代码**（注入测试）。

---

## Part F — 策略回测（`packs/strategy/backtest.py`）

```python
def backtest_strategy(df: pd.DataFrame, strategy: Strategy, *, target_col: str,
                     baseline: Strategy | None = None, profit_params: ProfitParams | None = None,
                     ead_col: str | None = None, pd_col: str | None = None) -> BacktestResult:
    """历史回放策略：算通过率、通过群坏账率、被拒群坏账率、利润、相对 baseline 的 swap。
    入参: df; strategy; target_col 真实标签; baseline 对比策略; 盈利参数（可选）。
    出参: BacktestResult。
    不变量: INV-1——所有率/利润由平台算；swap 分析用真实标签评估决策质量。
    伪代码:
      decision = apply_strategy(df, strategy)
      approved = decision != "reject"
      approval_rate = approved.mean()
      approved_bad = df.loc[approved, target_col].mean()
      rejected_bad = df.loc[~approved, target_col].mean() if (~approved).any() else 0.0
      profit = _strategy_profit(df[approved], profit_params, ead_col, pd_col) if profit_params else 0.0
      # swap 分析
      swap = _swap_analysis(df, strategy, baseline, target_col) if baseline else _zero_swap()
      by_seg = _segment_breakdown(df, decision, target_col)
      return BacktestResult(strategy.id, approval_rate, int(approved.sum()), approved_bad,
                            rejected_bad, profit, swap.in_count, swap.out_count,
                            swap.in_bad, swap.out_bad, by_seg)

def _swap_analysis(df, strategy, baseline, target_col) -> SwapStats:
    """相对 baseline 的换入换出分析：新通过的（swap-in）和新拒绝的（swap-out）及其坏账率。
    不变量: swap-in 坏账率高=新策略放进了坏客户；swap-out 坏账率低=误拒好客户。
    伪代码:
      new = apply_strategy(df, strategy) != "reject"
      old = apply_strategy(df, baseline) != "reject"
      swap_in = new & ~old; swap_out = ~new & old
      return SwapStats(swap_in.sum(), swap_out.sum(),
                       df.loc[swap_in, target_col].mean(), df.loc[swap_out, target_col].mean())
    """
```

- **测试要点**：通过率/坏账率手算对账；被拒群坏账率（误拒评估）；swap-in/out 计数与坏账率；无 baseline 时 swap 归零；盈利结合。

---

## Part G — 权衡视图（`packs/strategy/tradeoff.py`）

```python
def tradeoff_view(df: pd.DataFrame, *, score_col: str, target_col: str,
                 cutoffs: list[float] | None = None, profit_params: ProfitParams | None = None,
                 ead_col: str | None = None, pd_col: str | None = None) -> list[TradeoffPoint]:
    """风险收益前沿：扫描不同分数 cutoff，给出 (通过率, 坏账率, 利润) 曲线。
    入参: df; score_col 决策分; target_col; cutoffs（None=自动分位）; 盈利参数。
    出参: TradeoffPoint 列表（按 cutoff 排序），供选运营点。
    不变量: INV-1——每个点的率/利润平台算；帮助人工/Agent 在风险与收益间选点。
    伪代码:
      cuts = cutoffs or list(np.quantile(df[score_col].dropna(), np.linspace(0.05,0.95,19)))
      points = []
      for c in cuts:
          approved = df[score_col] >= c    # 约定分高=好（方向由调用方保证）
          ar = approved.mean()
          br = df.loc[approved, target_col].mean() if approved.any() else 0.0
          pf = _strategy_profit(df[approved], profit_params, ead_col, pd_col) if profit_params else 0.0
          points.append(TradeoffPoint(float(c), float(ar), float(br), float(pf)))
      return points

def recommend_operating_point(points: list[TradeoffPoint], *, objective: str = "max_profit",
                             max_bad_rate: float | None = None) -> TradeoffPoint:
    """在权衡曲线上推荐运营点（约束下最优）。
    入参: points; objective（max_profit|max_approval）; max_bad_rate 约束。
    出参: 满足约束的最优 TradeoffPoint。
    伪代码:
      feasible = [p for p in points if max_bad_rate is None or p.bad_rate <= max_bad_rate]
      if not feasible: return min(points, key=lambda p: p.bad_rate)   # 退化：选最低坏账
      key = (lambda p: p.expected_profit) if objective=="max_profit" else (lambda p: p.approval_rate)
      return max(feasible, key=key)
    """
```

- **测试要点**：cutoff 扫描产单调权衡（通过率↑则坏账率倾向↑）；推荐点满足坏账约束；max_profit/max_approval 目标；分数方向约定明确。

---

## Part H — strategy 能力包（`packs/strategy/tools.py`）

```python
def tool_vintage_curve(inputs, ctx) -> dict:
    """inputs:{dataset_id, cohort_col, mob_col, bad_col, mob_max?}。
       output:{cohorts, curves:{cohort:[...]}, counts, summary:{trend}}。determinism=deterministic。"""

def tool_roll_rate(inputs, ctx) -> dict:
    """inputs:{dataset_id, id_col, time_col, status_col, states:[str]}。
       output:{states, matrix:[[...]], base_counts}。"""

def tool_profit_calc(inputs, ctx) -> dict:
    """inputs:{dataset_id, segment_col?, ead_col, pd_col, params:{annual_rate,funding_rate,lgd,...}}。
       output:{results:[{segment,revenue,expected_loss,net_profit,roa}]}。"""

def tool_build_strategy(inputs, ctx) -> dict:
    """inputs:{strategy_type, rules:[{condition,decision,value}], score_col?, default_decision}。
       output:{strategy_id}。determinism=deterministic。"""

def tool_backtest_strategy(inputs, ctx) -> dict:
    """inputs:{dataset_id, strategy_id, target_col, baseline_strategy_id?, profit_params?, ead_col?, pd_col?}。
       output:{approval_rate, approved_bad_rate, rejected_bad_rate, expected_profit, swap_in/out...}。
       不变量: INV-1——回测数字平台算。"""

def tool_tradeoff_view(inputs, ctx) -> dict:
    """inputs:{dataset_id, score_col, target_col, cutoffs?, profit_params?, ...}。
       output:{points:[{cutoff,approval_rate,bad_rate,expected_profit}], recommended:{...}}。"""
```

`manifest.json`：策略**上线/执行**类动作（如把策略标为生效）若有，需 `needs_confirmation=True`（roadmap V4 手动模式）。回测/分析类是只读 deterministic。

- **测试要点**：每个 tool 经 runner 子进程往返；vintage/roll/profit/backtest/tradeoff 数字结构化；策略安全求值。

---

## Part I — 持久层

```sql
CREATE TABLE IF NOT EXISTS strategies (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, strategy_type TEXT NOT NULL,
  rules_json TEXT NOT NULL, score_col TEXT, default_decision_json TEXT,
  description TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS backtests (
  id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
  result_json TEXT NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE CASCADE
);
```

`StrategyRepository`：`create_strategy`、`get_strategy`、`list_for_task`、`save_backtest`、`get_backtest`、`list_backtests`。

- **测试要点**：策略/回测往返；FK CASCADE。

---

## Part J — 测试计划汇总

| 文件 | 覆盖 |
|------|------|
| `tests/test_strategy_contracts.py` | dataclass 往返 |
| `tests/test_strategy_vintage.py` | 累计坏账单调、恶化趋势、空 cohort |
| `tests/test_strategy_roll_rate.py` | 行概率和≈1、吸收态、空状态 |
| `tests/test_strategy_profit.py` | 盈利公式对账、分群、vintage 维度、ROA |
| `tests/test_strategy_build.py` | 规则校验、apply 首命中、**安全求值防注入** |
| `tests/test_strategy_backtest.py` | 通过率/坏账率/swap/盈利 |
| `tests/test_strategy_tradeoff.py` | cutoff 扫描、运营点推荐、约束 |
| `tests/test_strategy_pack.py` | 6 个 tool 经 runner 往返 |
| `tests/test_strategy_db.py` | 策略/回测往返、CASCADE |

---

## Part K — 任务执行顺序

```text
1. A 契约
2. B vintage
3. C roll_rate
4. D profit
5. E strategy（含安全求值）
6. F backtest（依赖 E）
7. G tradeoff
8. I DB + StrategyRepository
9. H strategy pack tools（依赖全部 + Phase 1/3）
10. J 测试 + 回归
```

每项 atomic commit。Phase 7 完成标志：能算 vintage 曲线/滚动率/分群盈利、构造并回测策略（通过率/坏账率/swap/利润全平台算）、产出风险收益权衡曲线并推荐运营点；策略规则安全求值不执行任意代码；6 个 tool 经子进程 runner 可用；策略上线类动作走人工确认（roadmap V4）。

---

*Phase 7 把策略从"拍脑袋定 cutoff"变成"数据驱动的风险收益权衡"。每个通过率、坏账率、利润数字都由平台算、可回测、可审计——Agent 负责解释和起草策略说明，但不编造任何一个业务数字。*
