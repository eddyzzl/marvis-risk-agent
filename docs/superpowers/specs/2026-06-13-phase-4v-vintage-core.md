# Phase 4V — Vintage / Roll Rate Deterministic Core（函数级 spec）

## 文档状态

- 状态：待实施
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 13 节）
- 前置依赖：Phase 3 数据层可登记/读取数据集；Phase 4 特征包可并行推进
- 目标：在 `validation/` 中交付共享的 vintage / roll rate 确定性核心，供 Phase 6 模型开发报告和 Phase 7 策略包复用，避免 Phase 6 反向依赖 Phase 7。

## 边界

Phase 4V 只交付确定性计算核心和轻量 payload，不做策略生成、不做前端 V2 图形、不做 LLM 文案。

- 计算函数放在 `riskmodel_checker/validation/vintage.py`。
- 只依赖 pandas/numpy 标准数据结构，不依赖 DB/FastAPI/Agent/Plugin runtime。
- packs 和报告层可以 import 本模块；本模块不 import packs/output/api。

## 捍卫的不变量

- **INV-1/INV-2**：vintage、roll rate、坏账率、MOB 指标全部由平台代码计算，LLM 不算。
- **INV-5**：输出只包含聚合指标，不包含客户明细。
- **INV-9**：月份/时间列解析显式，时区按 V1 月度分析口径保留本地钟面。
- **INV-10**：`validation/` 保持纯算法边界。

## 模块布局

```text
riskmodel_checker/validation/vintage.py
tests/validation/test_vintage.py
```

## Part A — 契约

```python
@dataclass(frozen=True)
class VintagePoint:
    cohort: str              # YYYY-MM
    mob: int                 # 0,1,2...
    sample_count: int
    bad_count: int
    bad_rate: float          # 单 MOB 边际坏率（该 MOB 口径）
    cum_bad_rate: float      # cohort 内按 MOB 升序的累计坏率（单调非减）——报告累计曲线用
    balance_sum: float | None
    denominator: str         # "count"|"balance"

@dataclass(frozen=True)
class RollRatePoint:
    from_bucket: str
    to_bucket: str
    count: int
    rate: float
```

辅助常量：

```python
DEFAULT_OVERDUE_BUCKETS = ("current", "1-30", "31-60", "61-90", "90+")
```

## Part B — `compute_vintage_curve`

```python
def compute_vintage_curve(
    dataframe: pd.DataFrame,
    *,
    cohort_col: str,
    mob_col: str,
    target_col: str,
    balance_col: str | None = None,
    denominator: str = "count",
) -> list[VintagePoint]:
    """按 cohort x MOB 计算 vintage 曲线。

    入参:
      dataframe: 明细或已预处理样本。
      cohort_col: cohort 月份列，可为 YYYYMM / YYYY-MM / datetime。
      mob_col: MOB 整数列。
      target_col: 0/1 坏样本列，1=bad。
      balance_col: 可选余额/金额列，用于 balance denominator。
      denominator: "count" 或 "balance"。

    出参:
      按 cohort、mob 排序的 VintagePoint 列表。

    异常:
      ValueError: 缺列、target 非 0/1、MOB 不能解析为非负整数、denominator 非法。
    """
```

实现要点：

1. cohort 统一成 `YYYY-MM` 字符串，保留本地钟面时间，不做 UTC 归一。
2. MOB 转非负整数；缺失 MOB 行不参与计算并在 summary warning 中记录。
3. count denominator：`bad_rate = bad_count / sample_count`。
4. balance denominator：要求 `balance_col` 非空；`bad_rate = bad_balance / total_balance`，`sample_count/bad_count` 仍保留。
5. 所有 rate 遇到分母 0 返回 `0.0`，不返回 NaN。
6. `cum_bad_rate`：cohort 内按 MOB 升序累计——累计坏 / 累计分母（count 或 balance 口径与 `bad_rate` 一致），**单调非减**；这是 Phase 6 报告 Vintage sheet 要的"累计曲线"，也是 Phase 7 vintage 视图的来源。

测试要点：

- YYYYMM / YYYY-MM / datetime cohort 等价。
- MOB 排序按整数，不按字符串。
- bad_count=0、sample_count=0 分母安全。
- balance denominator 与 count denominator 结果可区分。
- **`cum_bad_rate` 在每个 cohort 内随 MOB 单调非减**；末 MOB 的 cum_bad_rate ≥ 任一更早 MOB。
- 输出不包含明细字段。

## Part B-2 — `vintage_curve_wide`（透视为宽表，供报告/前端直接渲染）

```python
def vintage_curve_wide(
    points: Sequence[VintagePoint],
    *,
    metric: str = "cum_bad_rate",
) -> dict[str, list[float]]:
    """把 tidy VintagePoint 列表透视成 {cohort: [按 MOB 升序的 metric 值]} 宽表。
    metric ∈ {"cum_bad_rate","bad_rate"}（默认累计，供 Phase 6 报告累计曲线；Phase 7 同源）。
    各 cohort 的 MOB 轴对齐到全体 cohort 的 MOB 并集，缺失位用 None 占位（不编造 0）。

    异常: ValueError（metric 非法）。
    """
```

- **用途**：Phase 6 报告 Vintage sheet 和 Phase 7 策略 vintage 视图都用它把核心输出转成宽表，**不各自重算**。
- **测试要点**：宽表 MOB 轴对齐；缺失 cohort×MOB 为 None；`metric="cum_bad_rate"` 各行单调非减；与 `compute_vintage_curve` 数值一致。

## Part C — `compute_roll_rate`

```python
def compute_roll_rate(
    dataframe: pd.DataFrame,
    *,
    from_bucket_col: str,
    to_bucket_col: str,
    buckets: Sequence[str] = DEFAULT_OVERDUE_BUCKETS,
) -> list[RollRatePoint]:
    """计算逾期状态迁移矩阵。

    入参:
      dataframe: 含起点/终点逾期 bucket 的明细。
      from_bucket_col / to_bucket_col: 起点和终点 bucket 列。
      buckets: 合法 bucket 顺序。

    出参:
      所有 from x to 组合的 RollRatePoint；缺失组合 count=0/rate=0。

    异常:
      ValueError: 缺列或出现未知 bucket。
    """
```

实现要点：

1. 每个 from_bucket 内 rate 归一，总和约等于 1；该 from_bucket 无样本时全 0。
2. 输出覆盖完整矩阵，便于前端和报告渲染。
3. 不在本函数里解释“恶化/好转”，解释留给策略/报告层。

测试要点：

- 完整矩阵输出。
- 每个 from_bucket rate sum 在容差内等于 1。
- 未知 bucket 报错。
- 空数据返回全 0 矩阵或抛错的口径必须固定；推荐返回全 0 并带 warning helper。

## Part D — `vintage_summary_payload`

```python
def vintage_summary_payload(
    vintage_points: Sequence[VintagePoint],
    roll_rate_points: Sequence[RollRatePoint] | None = None,
) -> dict:
    """把 dataclass 列表转成报告/Tool output 可用的 JSON payload。

    出参:
      {
        "vintage": [asdict(point), ...],
        "roll_rate": [asdict(point), ...],
        "warnings": [str, ...],
      }
    """
```

- **用途**：Phase 6 model report 和 Phase 7 strategy pack 复用同一 payload。
- **测试要点**：JSON 可序列化；无 NaN/Inf；字段名稳定。

## Part E — 未来 Tool 包装

Phase 4V 本身不要求注册 Tool。后续两个入口复用本核心（**都 import 这里，不自建 vintage/roll-rate 实现**）：

- Phase 6 `compute_vintage_report`：调 `compute_vintage_curve` 拿 tidy 点、`vintage_curve_wide(..., metric="cum_bad_rate")` 拿累计曲线，再补放款金额/件均/利率等 cohort 业务列。
- Phase 7 `strategy` pack：vintage 视图直接复用 `compute_vintage_curve` + `vintage_curve_wide`；`roll_rate_matrix` 是**薄包装**——先从每客户逾期状态时序推出 from/to bucket，再调本核心 `compute_roll_rate`；策略层只加"恶化/好转"解读和盈利/回测，不重算迁移。

如果 Phase 4V 需要提前暴露给前端，可在 Phase 4 或 Phase 7 中新增薄 Tool 包装；不要把 Tool runtime 依赖加进 `validation/vintage.py`。

## Part F — 测试计划

| 文件 | 覆盖 |
|------|------|
| `tests/validation/test_vintage.py` | cohort 解析、MOB、count/balance denominator、bad_count=0、**cum_bad_rate 单调非减**、**vintage_curve_wide 透视/MOB 对齐**、roll rate 矩阵、JSON payload |
| `tests/test_report_texts_v2.py` 或 Phase 6 报告测试 | 报告层只消费 payload，不重算数字 |

Phase 4V 完成标志：`validation/vintage.py` 提供稳定、纯函数、聚合输出的 vintage/roll-rate 核心，Phase 6/7 可以 import 它而不产生阶段倒置。
