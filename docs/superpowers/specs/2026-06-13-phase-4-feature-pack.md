# Phase 4 — 特征分析 + 处理包（函数级 spec，含内部伪代码）

## 文档状态

- 状态：待实施
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 9 节能力包目录）
- 前置依赖：Phase 1（Tool 契约）、Phase 3（数据层 / Dataset / DataBackend）
- 目标：交付特征工程能力包——分析（IV/KS/AUC/Lift/PSI/相关性）、处理（分箱/WOE/onehot/归一化/缺失/异常）、衍生（交叉/比率/聚合）。**同时把这套确定性算法核心做干净，修掉 CODE_REVIEW 暴露的 PSI 双实现、KS O(n²)、分箱 NaN、相关系数 NaN 等算法债，validation/ 改为复用这个干净核心。**

## 捍卫的不变量

- **INV-1/INV-2**：所有特征指标（IV/KS/AUC/Lift/PSI/WOE）由 tool 代码计算，结构化产出，LLM 不碰。
- **INV-5**：特征报告里不落原始客户明细，只落统计量和脱敏分箱。
- 确定性核心唯一化：PSI/KS/分箱只有一份实现，`validation/` 与 `feature/` 共用，杜绝同报告内数值不一致（CODE_REVIEW P1-11）。

## 模块布局

```text
marvis/feature/
  __init__.py
  contracts.py      Bin / BinningResult / FeatureMetrics / WOEResult / CorrelationReport
  errors.py
  binning.py        分箱：等频/等距/ChiMerge/决策树/手动（修 NaN 边界，CODE_REVIEW P2-1/P0-2/P0-3）
  iv.py             IV / WOE 计算（统一口径）
  metrics.py        单特征 KS/AUC/Lift/PSI（修 KS O(n²) P2-2；统一 PSI P1-11）
  correlation.py    相关矩阵 / VIF 共线性（修相关系数 NaN P1-12）
  encode.py         onehot / label / woe 编码
  transform.py      归一化 / 标准化 / 缺失填充 / 异常截断
  derive.py         特征衍生 / 交叉
marvis/packs/feature/
  manifest.json     10 个 tool 声明
  tools.py
```

不新增重依赖（numpy/pandas/scipy/sklearn 已可用；ChiMerge/tree 分箱用 scipy/sklearn）。

> 复用与修债策略：`feature/binning.py` 和 `feature/metrics.py` 成为**唯一**的分箱/KS/PSI 实现；`validation/binning.py`、`validation/effectiveness.py` 改为从 `feature/` import（删除 `_psi_component` 重复实现、对齐 smoothing、KS 改线性），现有 validation 测试作为回归护栏。

---

## Part A — 契约（`feature/contracts.py`）

```python
@dataclass(frozen=True)
class Bin:
    index: int
    lower: float            # 左边界（-inf 表示开区间起点）
    upper: float            # 右边界（+inf 表示开区间终点）
    count: int
    bad_count: int
    good_count: int
    bad_rate: float
    woe: float
    iv_contribution: float

@dataclass(frozen=True)
class BinningResult:
    feature: str
    method: str             # equal_freq|equal_width|chimerge|tree|manual
    bins: tuple[Bin, ...]
    edges: tuple[float, ...]  # 长度 = len(bins)+1
    total_iv: float
    monotonic: bool         # bad_rate 是否单调（评分卡常要求）
    na_bin: Bin | None      # 缺失值单独成箱（若有）

@dataclass(frozen=True)
class FeatureMetrics:
    feature: str
    iv: float
    ks: float
    auc: float
    psi: float | None       # 需提供基准/对比期才有
    missing_rate: float
    unique_count: int
    lift_top_bin: float     # 头部箱 lift

@dataclass(frozen=True)
class WOEResult:
    feature: str
    edges: tuple[float, ...]
    woe_by_bin: tuple[float, ...]   # 与 bins 同序
    na_woe: float | None

@dataclass(frozen=True)
class CorrelationReport:
    features: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]   # 方阵
    collinear_pairs: tuple[tuple[str, str, float], ...]  # |corr|>=阈值
    vif: dict                                # feature -> VIF
```

- **测试要点**：dataclass 往返；`edges` 长度 = bins+1。

---

## Part B — 分箱（`feature/binning.py`，修 NaN/边界债）

```python
def equal_frequency_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    """等频分箱边界。修 CODE_REVIEW P2-1：先过滤 NaN/Inf，空数组兜底。
    入参: values 数值数组（可能含 NaN）; bin_count 目标箱数。
    出参: 单调递增边界数组（含 ±inf 端点），长度 ≤ bin_count+1（去重后）。
    不变量: NaN/Inf 不进分位计算（CODE_REVIEW P2-1/P0-2）。
    伪代码:
      arr = np.asarray(values, dtype=float)
      arr = arr[np.isfinite(arr)]
      if arr.size == 0: return np.array([-np.inf, np.inf])
      quantiles = np.linspace(0, 1, bin_count + 1)
      edges = np.unique(np.quantile(arr, quantiles))   # unique 去重并列分位
      edges[0], edges[-1] = -np.inf, np.inf            # 开端点，覆盖未来越界值
      return edges
    """

def equal_width_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    """等距分箱边界。同样过滤 NaN/Inf。
    伪代码:
      arr = finite(values); if empty: [-inf, inf]
      lo, hi = arr.min(), arr.max()
      if lo == hi: return np.array([-inf, inf])
      edges = np.linspace(lo, hi, bin_count + 1); edges[0], edges[-1] = -inf, inf
      return edges
    """

def chimerge_edges(values: np.ndarray, target: np.ndarray, *,
                   max_bins: int, min_pvalue: float = 0.05, init_bins: int = 100) -> np.ndarray:
    """ChiMerge 有监督分箱：初始细分→相邻箱卡方最小者合并，直到箱数/显著性满足。
    入参: values; target 0/1; max_bins 上限; min_pvalue 合并停止阈值。
    出参: 单调边界（±inf 端点）。
    不变量: 用 target 指导，得到对 y 有区分力的边界。
    伪代码:
      arr, tgt = _finite_pairs(values, target)
      edges = equal_frequency_edges(arr, init_bins)       # 初始细分
      bins = _bin_stats(arr, tgt, edges)                  # 每箱 (good, bad)
      while len(bins) > max_bins or _max_merge_pvalue(bins) > min_pvalue:
          i = _argmin_chi2(bins)                          # 卡方最小的相邻对
          bins = _merge(bins, i, i+1)
          if len(bins) <= 2: break
      return _edges_from_bins(bins)
    """

def tree_edges(values: np.ndarray, target: np.ndarray, *,
               max_bins: int, min_samples_leaf: float = 0.05, seed: int = 0) -> np.ndarray:
    """决策树分箱：单特征 DecisionTree 拟合 y，取分裂点作边界。
    入参: values; target; max_bins(=max_leaf_nodes); min_samples_leaf 比例; seed（stochastic）。
    出参: 单调边界（±inf 端点）。
    伪代码:
      from sklearn.tree import DecisionTreeClassifier
      arr, tgt = _finite_pairs(values, target)
      tree = DecisionTreeClassifier(max_leaf_nodes=max_bins,
                 min_samples_leaf=min_samples_leaf, random_state=seed)
      tree.fit(arr.reshape(-1,1), tgt)
      thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != _TREE_UNDEFINED])
      return np.array([-inf, *thresholds, inf])
    """

def manual_edges(breakpoints: list[float]) -> np.ndarray:
    """手动分箱：用户给定切点，补 ±inf 端点、排序去重。
    异常: BinningError（切点为空）。
    """

def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """把值分配到箱索引。NaN → -1（缺失箱标记）。
    出参: int 数组，[0, len(edges)-2] 或 -1（NaN）。
    不变量: 用统一边界（避免 CODE_REVIEW P0-3 各 split 各自分箱）。
    伪代码:
      arr = np.asarray(values, dtype=float)
      out = np.full(arr.shape, -1, dtype=int)
      mask = np.isfinite(arr)
      out[mask] = np.clip(np.searchsorted(edges[1:-1], arr[mask], side="right"), 0, len(edges)-2)
      return out
    """
```

辅助：`_finite_pairs(values, target)` 对齐过滤两边 NaN；`_bin_stats`、`_argmin_chi2`、`_merge`、`_edges_from_bins`。

- **测试要点**：含 NaN 列分箱不崩（修 P2-1）；并列值 `np.unique` 后边界不重复；`assign_bins` NaN→-1；ChiMerge/tree 分箱对 y 单调；统一边界跨 split 一致（修 P0-3）；空/单值列兜底 `[-inf, inf]`。

---

## Part C — IV / WOE（`feature/iv.py`，统一口径）

```python
def compute_woe_iv(values: np.ndarray, target: np.ndarray, edges: np.ndarray, *,
                   feature: str, smoothing: float = 0.5, na_as_bin: bool = True) -> BinningResult:
    """给定边界，算每箱 WOE/IV，组装 BinningResult。
    入参: values; target 0/1（1=bad）; edges; feature 名; smoothing 拉普拉斯平滑; na_as_bin 缺失单独成箱。
    出参: BinningResult（含每箱 woe/iv_contribution、total_iv、单调性）。
    异常: FeatureError（target 非二值 / 全同类）。
    不变量: WOE/IV 口径统一（下方公式），smoothing 单一常数，避免 CODE_REVIEW P1-11 双实现。
    """
    # 伪代码:
    if set(np.unique(target[np.isfinite(values) | True])) - {0, 1}: raise FeatureError("target must be 0/1")
    bin_idx = assign_bins(values, edges)
    total_bad = max((target == 1).sum(), 1)
    total_good = max((target == 0).sum(), 1)
    bins = []
    iv_total = 0.0
    groups = list(range(len(edges) - 1)) + ([-1] if na_as_bin and (bin_idx == -1).any() else [])
    for b in groups:
        mask = bin_idx == b
        count = int(mask.sum())
        bad = int((target[mask] == 1).sum()); good = count - bad
        # 拉普拉斯平滑分布占比
        bad_dist = (bad + smoothing) / (total_bad + smoothing * len(groups))
        good_dist = (good + smoothing) / (total_good + smoothing * len(groups))
        woe = float(np.log(good_dist / bad_dist))           # 统一口径: ln(%good / %bad)
        iv_c = float((good_dist - bad_dist) * woe)
        iv_total += iv_c
        bins.append(Bin(index=b, lower=_lo(edges,b), upper=_hi(edges,b), count=count,
                        bad_count=bad, good_count=good,
                        bad_rate=(bad / count if count else 0.0), woe=woe, iv_contribution=iv_c))
    na_bin = next((bb for bb in bins if bb.index == -1), None)
    real_bins = tuple(bb for bb in bins if bb.index != -1)
    return BinningResult(feature=feature, method="given", bins=real_bins, edges=tuple(edges),
                         total_iv=round(iv_total, 6), monotonic=_is_monotonic([bb.bad_rate for bb in real_bins]),
                         na_bin=na_bin)

def woe_result_from_binning(binning: BinningResult) -> WOEResult:
    """从 BinningResult 抽出 WOE 映射（编码时用）。
    伪代码:
      return WOEResult(feature=binning.feature, edges=binning.edges,
                       woe_by_bin=tuple(b.woe for b in binning.bins),
                       na_woe=(binning.na_bin.woe if binning.na_bin else None))
    """
```

公式注释（写进 docstring，固定口径）：

```text
WOE_i = ln( (good_i_dist) / (bad_i_dist) )           # %good / %bad，含 smoothing
IV_i  = ( good_i_dist - bad_i_dist ) * WOE_i
IV    = Σ IV_i
分数方向约定：WOE 越大 = 该箱越偏好客户（good 占比高）。全平台统一此口径。
```

- **测试要点**：已知分布手算 WOE/IV 对得上；某箱全 good/全 bad 时 smoothing 防 ±inf；缺失成箱；单调性判断正确；target 非 0/1 抛错；IV 与 validation 旧实现回归一致（修债后）。

---

## Part D — 单特征指标（`feature/metrics.py`，修 KS O(n²) / 统一 PSI）

```python
def feature_ks(scores: np.ndarray, target: np.ndarray) -> float:
    """单特征/分数 KS，线性实现（修 CODE_REVIEW P2-2 的 O(n²)）。
    入参: scores; target 0/1。
    出参: KS ∈ [0,1]。
    不变量: O(n log n)；与 validation 的 _roc_ks_curve 同源。
    伪代码:
      arr, tgt = _finite_pairs(scores, target)
      order = np.argsort(arr, kind="mergesort")
      s, t = arr[order], tgt[order]
      total_bad = max(t.sum(), 1); total_good = max((1 - t).sum(), 1)
      cum_bad = np.cumsum(t) / total_bad
      cum_good = np.cumsum(1 - t) / total_good
      # 只在分数变化点取 KS
      change = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
      return float(np.max(np.abs(cum_bad[change] - cum_good[change]))) if len(s) else 0.0
    """

def feature_auc(scores: np.ndarray, target: np.ndarray) -> float:
    """AUC（Mann-Whitney U / 秩和，避免逐阈值）。
    出参: AUC ∈ [0,1]。
    伪代码:
      arr, tgt = _finite_pairs(scores, target)
      pos = arr[tgt == 1]; neg = arr[tgt == 0]
      if len(pos)==0 or len(neg)==0: return 0.5
      ranks = rankdata(arr)                         # scipy
      auc = (ranks[tgt==1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
      return float(auc)
    """

def feature_lift(scores: np.ndarray, target: np.ndarray, *, bins: int = 10) -> list[float]:
    """分箱 lift：每箱 bad_rate / 总体 bad_rate（降序，头部高 lift = 区分力强）。
    出参: 各箱 lift 列表（按分数降序）。
    伪代码:
      edges = equal_frequency_edges(scores, bins); idx = assign_bins(scores, edges)
      base = target.mean()
      return [ (target[idx==b].mean()/base if (idx==b).any() and base>0 else 0.0)
               for b in reversed(range(len(edges)-1)) ]
    """

def compute_psi(expected_dist: np.ndarray, actual_dist: np.ndarray, *,
                smoothing: float = 1e-6) -> float:
    """PSI 唯一实现（修 CODE_REVIEW P1-11 双实现 + smoothing 不一致）。
    入参: expected_dist/actual_dist 两个分布占比数组（同长，和≈1）; smoothing。
    出参: PSI ≥ 0。
    不变量: 全平台唯一 PSI 实现；validation 改为 import 此函数。
    伪代码:
      e = np.where(np.asarray(expected_dist)==0, smoothing, expected_dist)
      a = np.where(np.asarray(actual_dist)==0, smoothing, actual_dist)
      return float(np.sum((a - e) * np.log(a / e)))
    """

def feature_psi(base_values: np.ndarray, compare_values: np.ndarray, edges: np.ndarray, *,
                smoothing: float = 1e-6) -> float:
    """特征稳定性 PSI：用同一 edges 对基准期/对比期分箱算占比，再 compute_psi。
    不变量: 两期必须用同一 edges（CODE_REVIEW P0-3 教训）。
    伪代码:
      be = _bin_distribution(base_values, edges)      # 各箱占比
      ce = _bin_distribution(compare_values, edges)
      return compute_psi(be, ce, smoothing=smoothing)
    """

def feature_metrics(values: np.ndarray, target: np.ndarray, *, feature: str,
                    bins: int = 10, compare_values: np.ndarray | None = None) -> FeatureMetrics:
    """综合单特征指标：IV/KS/AUC/Lift/PSI/缺失率。
    伪代码:
      edges = equal_frequency_edges(values, bins)
      binning = compute_woe_iv(values, target, edges, feature=feature)
      psi = feature_psi(values, compare_values, edges) if compare_values is not None else None
      return FeatureMetrics(feature=feature, iv=binning.total_iv,
          ks=feature_ks(values, target), auc=feature_auc(values, target), psi=psi,
          missing_rate=float(np.isnan(values).mean()), unique_count=int(_nunique(values)),
          lift_top_bin=feature_lift(values, target, bins=bins)[0])
    """
```

- **测试要点**：KS 线性结果与逐阈值旧实现一致但复杂度降（修 P2-2）；AUC 秩和与 sklearn `roc_auc_score` 一致；PSI 单实现、与 validation 一致（修 P1-11）；两期同 edges；全同分/单类别兜底；KS/AUC ∈ [0,1]、PSI ≥ 0（这些区间是 Phase 2 post_check 的依据）。

---

## Part E — 相关性 / 共线性（`feature/correlation.py`，修 NaN 债）

```python
def correlation_matrix(df: pd.DataFrame, features: list[str], *, method: str = "pearson") -> np.ndarray:
    """特征相关矩阵。method: pearson|spearman。
    伪代码: return df[features].corr(method=method).to_numpy()
    """

def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """两列相关系数，零方差/NaN 安全（修 CODE_REVIEW P1-12）。
    出参: 相关系数；任一列零方差或结果非有限 → 0.0（不反转/不崩）。
    伪代码:
      xa, ya = _finite_pairs(x, y)
      if xa.size < 2 or np.std(xa)==0 or np.std(ya)==0: return 0.0
      c = np.corrcoef(xa, ya)[0,1]
      return float(c) if np.isfinite(c) else 0.0
    """

def vif(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """方差膨胀因子（共线性诊断）。VIF_i = 1/(1-R²_i)，R² 来自 feature_i ~ 其它特征。
    伪代码:
      out = {}
      X = df[features].dropna()
      for f in features:
          others = [c for c in features if c != f]
          r2 = _ols_r2(X[others], X[f])
          out[f] = float(1.0 / (1.0 - r2)) if r2 < 1 else float("inf")
      return out
    """

def find_collinear_pairs(matrix: np.ndarray, features: list[str], *,
                         threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """|corr| ≥ threshold 的特征对。
    伪代码: 上三角扫描，收 |matrix[i,j]|>=threshold 的 (features[i], features[j], corr)。
    """

def correlation_report(df, features, *, method="pearson", threshold=0.8) -> CorrelationReport:
    """组装 CorrelationReport。"""
```

- **测试要点**：零方差列 `safe_correlation` 返 0 不崩（修 P1-12）；VIF 对完全共线列→inf；collinear_pairs 阈值生效。

---

## Part F — 编码（`feature/encode.py`）

```python
def onehot_encode(df: pd.DataFrame, columns: list[str], *, max_categories: int = 50,
                  handle_unknown: str = "ignore") -> tuple[pd.DataFrame, dict]:
    """onehot 编码。高基数列（>max_categories）拒绝并报错，避免列爆炸。
    出参: (编码后 df, mapping{col: [categories]})。
    异常: FeatureError（某列基数 > max_categories）。
    伪代码:
      for c in columns:
          if df[c].nunique() > max_categories: raise FeatureError(f"{c} too many categories")
      dummies = pd.get_dummies(df[columns], prefix=columns, dummy_na=False)
      return pd.concat([df.drop(columns=columns), dummies], axis=1), {c: list(df[c].dropna().unique()) for c in columns}
    """

def label_encode(series: pd.Series) -> tuple[pd.Series, dict]:
    """类别→整数。出参: (编码序列, {类别: 码})。未知/NaN→-1。"""

def woe_encode(df: pd.DataFrame, feature: str, woe: WOEResult) -> pd.Series:
    """用 WOEResult 把特征值映射成 WOE。
    伪代码:
      idx = assign_bins(df[feature].to_numpy(float), np.array(woe.edges))
      out = np.array([woe.woe_by_bin[i] if i>=0 else (woe.na_woe or 0.0) for i in idx])
      return pd.Series(out, index=df.index, name=f"{feature}_woe")
    """
```

- **测试要点**：onehot 高基数拒绝；woe_encode 用训练边界映射（不重新分箱）；NaN→na_woe；label encode 未知→-1。

---

## Part G — 变换（`feature/transform.py`）

```python
def minmax_normalize(values: np.ndarray, *, feature_range=(0,1)) -> tuple[np.ndarray, dict]:
    """min-max 归一化。出参: (归一值, {min,max} 训练参数，供应用到测试集)。
    伪代码: lo,hi=nanmin,nanmax; scaled=(v-lo)/(hi-lo) if hi>lo else 0; 映射到 feature_range; 返回参数
    """

def zscore_standardize(values: np.ndarray) -> tuple[np.ndarray, dict]:
    """z-score 标准化。出参: (标准化值, {mean,std})。std=0 → 全 0。"""

def apply_scaler(values: np.ndarray, params: dict, *, kind: str) -> np.ndarray:
    """用训练集参数变换新数据（防训练/测试口径不一致）。"""

def impute_missing(series: pd.Series, *, strategy: str = "median",
                   fill_value=None) -> tuple[pd.Series, object]:
    """缺失填充。strategy: mean|median|mode|constant。出参: (填充后, 填充值)。
    不变量: 返回填充值，供测试集用同一值（不在测试集重算）。
    """

def cap_outliers(values: np.ndarray, *, method: str = "iqr", lower_q=0.01, upper_q=0.99) -> tuple[np.ndarray, dict]:
    """异常截断。method: iqr|quantile。出参: (截断值, {lower,upper})。"""
```

- **测试要点**：归一化/标准化往返参数可复用到新数据；零方差兜底；缺失填充返回值可应用测试集；截断边界正确。

---

## Part H — 衍生 / 交叉（`feature/derive.py`）

```python
def cross_arithmetic(df: pd.DataFrame, col_a: str, col_b: str,
                     ops: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """两列算术交叉。ops ⊆ {add,sub,mul,div,ratio}。div/ratio 防除零。
    出参: (新增列的 df, 新列名)。
    伪代码:
      out = {}
      if "add" in ops: out[f"{a}_add_{b}"] = df[a]+df[b]
      if "sub" in ops: out[f"{a}_sub_{b}"] = df[a]-df[b]
      if "mul" in ops: out[f"{a}_mul_{b}"] = df[a]*df[b]
      if "div" in ops: out[f"{a}_div_{b}"] = df[a]/df[b].replace(0,np.nan)
      ...
      return df.assign(**out), list(out.keys())
    """

def aggregate_feature(df: pd.DataFrame, group_col: str, value_col: str,
                      aggs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """分组聚合衍生（如按用户聚合交易额 mean/max/std）。aggs ⊆ {mean,max,min,std,sum,count}。
    伪代码:
      g = df.groupby(group_col)[value_col].agg(aggs).add_prefix(f"{value_col}_by_{group_col}_")
      return df.merge(g, on=group_col, how="left"), list(g.columns)
    """

def derive_batch(df: pd.DataFrame, recipe: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """按 recipe 批量衍生。recipe: [{kind:"cross",a,b,ops}|{kind:"agg",group,value,aggs}|{kind:"ratio",num,den}]。
    出参: (df, 所有新列名)。
    不变量: 衍生是确定性的；交叉爆炸由调用方/Planner 控制规模。
    """
```

### H-2 LLM 推荐交叉（语义推荐 + 用户选 + 平台算，用户确认的形式）

> 形式（用户确认）：**LLM 看特征含义/产品/厂商 → 推荐值得交叉的特征对 + 理由（不算数）→ 用户选/确认（高档位可自动选 top-N）→ 平台算选中的交叉 + IV（确定性）**。避免无脑生成 C(n,2) 组合爆炸。

```python
@dataclass(frozen=True)
class CrossRecommendation:
    col_a: str
    col_b: str
    ops: tuple[str, ...]          # 推荐的运算（ratio/div/mul/...）
    rationale: str                # LLM 给的语义理由（如"额度使用率=支用/授信，业务含义强"）
    confidence: str               # high|medium|low

def recommend_feature_crosses(feature_dictionary: dict, existing_metrics: dict, *,
                              llm_factory, max_candidates: int = 30) -> list[CrossRecommendation]:
    """LLM 基于特征语义（含义/产品/厂商）+ 已有 IV，推荐值得交叉的特征对。**只推荐，不算数**。
    入参: feature_dictionary{feat:{含义,产品,厂商}}; existing_metrics{feat:{iv,ks}}; llm_factory; 上限。
    出参: CrossRecommendation 列表（带理由+置信度）。
    异常: 无（LLM 失败回退空列表，降级为用户手动指定）。
    不变量: INV-1/INV-2——LLM 只产"哪些对值得交叉"的语义建议，**绝不产 IV/KS 数字**；
            推荐的 col_a/col_b 必须是已存在特征（校验，防 LLM 编列名）。
    伪代码:
      prompt = build_cross_prompt(feature_dictionary, existing_metrics, max_candidates)
      raw = llm_factory().complete(system_prompt=CROSS_SYS, user_prompt=prompt,
                                   response_format={"type":"json_object"}, stream=False)
      recs = _parse_recommendations(raw)
      valid = [r for r in recs if r.col_a in feature_dictionary and r.col_b in feature_dictionary]
      return valid[:max_candidates]
    """

def evaluate_crosses(df, target, recommendations: list[CrossRecommendation], *,
                     selected_pairs: list[tuple[str, str]] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """对【用户选中的】交叉对，平台算交叉特征 + 其 IV/KS（确定性）。
    入参: df; target; recommendations; selected_pairs（用户选的子集，None=全部推荐）。
    出参: (含新交叉列的 df, [{new_col, iv, ks, from:(a,b,op)}])。
    不变量: INV-1——交叉值和 IV/KS 由平台算（cross_arithmetic + feature_metrics），LLM 不碰。
    伪代码:
      pick = selected_pairs or [(r.col_a, r.col_b) for r in recommendations]
      new_cols, results = [], []
      for (a, b) in pick:
          ops = _ops_for(a, b, recommendations)
          df, cols = cross_arithmetic(df, a, b, ops)
          for c in cols:
              m = feature_metrics(df[c].to_numpy(float), target, feature=c)
              results.append({"new_col": c, "iv": m.iv, "ks": m.ks, "from": (a, b)})
          new_cols += cols
      return df, results
    """
```

`CROSS_SYS`（写死 INV-1）：「你基于特征的业务含义推荐值得交叉的特征对和运算，给出理由。**你不计算任何 IV/KS/指标**——那些由平台算。只输出特征对+运算+理由的 JSON。」

- **测试要点**：`recommend_feature_crosses` 产合法推荐（col 必须存在、不编列名）、LLM 不产数字；`evaluate_crosses` 只算用户选中的对、IV/KS 来自平台核心；LLM 失败回退手动；决策点：推荐→用户选→算（编排层 `decision_point`）。

- **测试要点**：算术交叉除零→NaN 不崩；聚合衍生 left join 不膨胀（行数不变）；recipe 批量衍生新列名无冲突。

---

## Part I — feature 能力包（`packs/feature/`）

`manifest.json` 声明 10 个 tool（determinism 多为 deterministic，tree 分箱 stochastic 带 seed）：

```python
# tools.py —— 每个包装 feature/ 能力为 Tool 契约，结构化产出（INV-1）

def tool_compute_feature_metrics(inputs, ctx) -> dict:
    """批量算特征指标。
    inputs: {dataset_id, features:[str], target_col, bins?:int, compare_dataset_id?}。
    output: {metrics:[{feature, iv, ks, auc, psi, missing_rate, unique_count, lift_top_bin}]}。
    不变量: INV-1，区间字段供 Phase 2 post_check（ks/auc∈[0,1]、psi≥0）。
    """

def tool_bin_feature(inputs, ctx) -> dict:
    """分箱 + WOE/IV。
    inputs: {dataset_id, feature, target_col, method, max_bins?, breakpoints?, seed?}。
    output: {edges, bins:[{lower,upper,count,bad_rate,woe,iv_contribution}], total_iv, monotonic}。
    """

def tool_compute_psi(inputs, ctx) -> dict:
    """特征/分数 PSI（跨时间或跨样本）。
    inputs: {dataset_id, feature, base_filter, compare_filter, bins?}。
    output: {psi, bin_distributions}。
    """

def tool_correlation_analysis(inputs, ctx) -> dict:
    """相关 + 共线性。
    inputs: {dataset_id, features:[str], method?, threshold?}。
    output: {matrix, collinear_pairs:[[f1,f2,corr]], vif:{f:val}}。
    """

def tool_woe_encode(inputs, ctx) -> dict:
    """WOE 编码（产出新 dataset）。
    inputs: {dataset_id, features:[str], target_col, method?}。
    output: {result_dataset_id, woe_maps}。
    """

def tool_onehot_encode(inputs, ctx) -> dict:
    """onehot（新 dataset）。inputs:{dataset_id, columns, max_categories?}。output:{result_dataset_id, mapping}。"""

def tool_normalize(inputs, ctx) -> dict:
    """归一化/标准化（新 dataset + 参数）。inputs:{dataset_id, columns, method}。output:{result_dataset_id, scaler_params}。"""

def tool_impute_missing(inputs, ctx) -> dict:
    """缺失填充（新 dataset + 填充值）。inputs:{dataset_id, columns, strategy, fill_value?}。output:{result_dataset_id, fill_values}。"""

def tool_cap_outliers(inputs, ctx) -> dict:
    """异常截断（新 dataset + 边界）。inputs:{dataset_id, columns, method, lower_q?, upper_q?}。output:{result_dataset_id, bounds}。"""

def tool_cross_features(inputs, ctx) -> dict:
    """特征交叉/衍生（新 dataset）。
    inputs: {dataset_id, recipe:[{kind,...}]}。
    output: {result_dataset_id, new_columns:[str]}。
    不变量: 衍生数量受 recipe 控制；输出行数 = 原行数（聚合衍生 left join 不膨胀）。
    """
```

所有产出新 dataset 的 tool 都经 `DatasetRegistry.register_existing` 登记，返回 `result_dataset_id`，供 Plan 下游引用。

- **测试要点**：每个 tool 经 `ToolRunner.invoke` 子进程往返；指标 tool 输出区间合法；编码/变换 tool 产出可被下游读取；`scaler_params`/`fill_values`/`bounds`/`woe_maps` 回传（供应用到测试集，口径一致）。

---

## Part J — validation/ 算法债消除（回归护栏）

把 CODE_REVIEW 的算法债在本阶段一次清掉，让 `validation/` 与 `feature/` 共用干净核心：

| CODE_REVIEW 项 | 动作 |
|---------------|------|
| P1-11 PSI 双实现 + smoothing 不一致 | 删 `validation/effectiveness.py:_psi_component`，改 import `feature.metrics.compute_psi` |
| P2-2 KS O(n²) | `validation/binning.py:compute_ks` 改 import `feature.metrics.feature_ks`（线性） |
| P2-1 `equal_frequency_bin_edges` NaN | 改 import `feature.binning.equal_frequency_edges`（已过滤 NaN） |
| P0-3 `compute_bin_tables` 各 split 各自分箱 | 统一传 `context.edges`（训练集边界）到所有 split |
| P1-12 相关系数 NaN 反转 | `_should_reverse_eval_bins` 改用 `feature.correlation.safe_correlation` |

- **不变量**：删除重复实现后，现有 `tests/test_model_algorithms.py`、`tests/test_metric_tables.py`、`tests/test_pipeline_v2.py` 必须全绿（回归护栏），数值差异在浮点容差内。
- **测试要点**：迁移前后同一份样本的 KS/PSI/IV 数值一致（容差 1e-9）；validation 与 feature 对同输入返回同结果。

---

## Part K — 测试计划汇总

| 文件 | 覆盖 |
|------|------|
| `tests/test_feature_contracts.py` | dataclass 往返 |
| `tests/test_feature_binning.py` | 等频/等距/ChiMerge/tree/手动；**NaN 过滤、并列值、统一边界** |
| `tests/test_feature_iv.py` | WOE/IV 手算对账、smoothing、缺失成箱、单调性 |
| `tests/test_feature_metrics.py` | **KS 线性=旧值、AUC=sklearn、PSI 单实现、区间合法** |
| `tests/test_feature_correlation.py` | **零方差安全、VIF、collinear** |
| `tests/test_feature_encode.py` | onehot 高基数拒绝、woe 用训练边界、label 未知 |
| `tests/test_feature_transform.py` | 归一/标准化参数复用、缺失填充、截断 |
| `tests/test_feature_derive.py` | 交叉除零、聚合不膨胀、recipe 批量 |
| `tests/test_feature_pack.py` | 10 个 tool 经 runner 往返、产新 dataset |
| `tests/test_validation_debt.py` | **validation↔feature 数值一致回归** |

---

## Part L — 任务执行顺序

```text
1. A 契约              （无依赖）
2. B binning           （依赖 A；修 NaN 债）
3. C iv/woe            （依赖 A,B）
4. D metrics           （依赖 A,B；修 KS/PSI 债）
5. E correlation       （依赖 A；修相关 NaN 债）
6. F encode            （依赖 B,C）
7. G transform         （依赖 A）
8. H derive            （依赖 A）
9. J validation 债消除  （依赖 B,D,E；回归护栏）
10. I feature pack      （依赖全部 + Phase 1 Tool 契约 + Phase 3 registry）
11. K 测试 + 全量回归
```

每项 atomic commit。Phase 4 完成标志：10 个 feature tool 经子进程 runner 可用、产出结构化指标（区间被 Phase 2 post_check 守住）；分箱/WOE/IV/KS/AUC/PSI/相关全套确定性核心唯一化；validation 旧测试在复用新核心后全绿（算法债清零）。

---

*Phase 4 把确定性算法核心做干净并暴露成工具。它既交付特征工程能力，又顺手还清了 CODE_REVIEW 的算法债——之后 validation 和 feature 共用一份 KS/PSI/分箱实现，不会再有"同报告内 PSI 不一致"这类问题。*
