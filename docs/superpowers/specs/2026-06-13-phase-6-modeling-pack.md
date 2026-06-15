# Phase 6 — 模型开发包（函数级 spec，含内部伪代码）

## 文档状态

- 状态：待实施
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 15.1 节）
- 前置依赖：Phase 1（Tool 契约）、Phase 3（数据层/Dataset）、Phase 4（特征工程/确定性指标核心）、Phase 4V（共享 vintage/roll-rate 核心 `validation/vintage.py`）
- 目标：交付信贷建模能力包——数据质量与建模就绪检查、建模样本准备、特征筛选、常见模型 recipe（LGB/XGB/LR/评分卡）、实验记录、模型产物、PMML 导出、交接到验证流程。覆盖贷前/贷中/贷后/前筛/营销/交易/捞回/收入/额度/定价等场景（recipe + 场景参数模板，不各写一套）。

## 捍卫的不变量

- **INV-1/INV-2**：模型指标（train/test/oot 的 KS/AUC/PSI、过拟合差值）由 tool 代码 + Phase 4 特征核心计算，LLM 不算。
- **INV-5**：实验记录/产物元数据不落原始样本明细；模型文件单独存盘，不进记忆。
- **roadmap V3 约束**：训练产物必须进验证流程后才算可复核产物；训练上下文与验证契约**独立**（不复用 RMC_* notebook 契约）。
- **stochastic 可复现**：所有训练 tool `determinism="stochastic"` 且必须传 `seed`，同 seed 结果一致。

## 模块布局

```text
marvis/packs/modeling/
  __init__.py
  manifest.json
  contracts.py      ModelRecipe / TrainConfig / ModelMetrics / ModelArtifact / TrainResult / Experiment
  errors.py
  readiness.py      check_data_quality / modeling_readiness
  prepare.py        prepare_modeling_frame（split + target 准备）
  select.py         select_features（IV/相关/重要性筛选）
  recipes/
    __init__.py     recipe 注册表
    lgb.py / xgb.py / lr.py / scorecard.py
  experiment.py     ExperimentStore（实验记录）
  artifact.py       模型存取 / export_pmml
  handoff.py        handoff_to_validation
  scenarios.py      场景参数模板（贷前/前筛/营销/...）
  reject_inference.py   拒绝推断（桩，标注需方法论评审，V1 不实现）
  tools.py          tool_* 包装
marvis/db.py   新增 experiments / model_artifacts 表
marvis/validation/overfitting.py   迁入过拟合检测（CODE_REVIEW P2-27）
```

新增依赖：`lightgbm>=4`、`xgboost>=2`、`scikit-learn>=1.3`、`sklearn2pmml` 或 `nyoka`（PMML 导出，二选一在实施时定）。

---

## Part A — 契约（`packs/modeling/contracts.py`）

```python
@dataclass(frozen=True)
class ModelRecipe:
    id: str                 # lgb|xgb|lr|scorecard
    algorithm: str
    default_params: dict
    param_space: dict       # 调参空间（可选网格/范围）
    requires_woe: bool      # 评分卡=True（需先 WOE 编码）

@dataclass(frozen=True)
class TrainConfig:
    dataset_id: str
    features: tuple[str, ...]
    target_col: str
    split_col: str          # train/test/oot 划分列
    split_values: dict      # {"train":..,"test":..,"oot":..}
    params: dict            # 覆盖 recipe 默认参数
    seed: int               # 必填，stochastic 可复现
    early_stopping_rounds: int | None

@dataclass(frozen=True)
class ModelMetrics:
    train_ks: float; test_ks: float; oot_ks: float | None
    train_auc: float; test_auc: float; oot_auc: float | None
    psi_test_vs_train: float | None
    psi_oot_vs_train: float | None
    overfit_train_test_gap: float   # |train_ks - test_ks| / train_ks（相对）
    overfit_train_oot_gap: float | None
    overfit_flag: bool              # 超阈值标记（来自 validation/overfitting）

@dataclass(frozen=True)
class ModelArtifact:
    id: str
    experiment_id: str
    algorithm: str
    model_path: str         # 相对 task_dir（pickle/txt/json，不进记忆）
    pmml_path: str | None
    feature_list: tuple[str, ...]
    params: dict
    woe_maps: dict | None   # 评分卡/WOE 模型的编码映射
    created_at: str

@dataclass(frozen=True)
class TrainResult:
    artifact: ModelArtifact
    metrics: ModelMetrics
    feature_importance: tuple[tuple[str, float], ...]   # 降序
    experiment_id: str

@dataclass(frozen=True)
class Experiment:
    id: str
    task_id: str
    recipe_id: str
    config: TrainConfig
    metrics: ModelMetrics | None
    artifact_id: str | None
    status: str             # created|trained|failed|handed_off|validated
    created_at: str
```

- **测试要点**：dataclass 往返；`overfit_*_gap` 字段存在（供过拟合判断）。

---

## Part B — 数据质量与建模就绪（`packs/modeling/readiness.py`）

```python
@dataclass(frozen=True)
class QualityIssue:
    column: str
    kind: str               # missing|constant|near_constant|duplicate_col|high_cardinality|leakage_suspect
    detail: str
    severity: str           # block|warn

def check_data_quality(backend, dataset: Dataset, dataset_path: Path, *,
                       target_col: str | None = None) -> list[QualityIssue]:
    """扫描建模数据质量问题。
    入参: backend; dataset; path; target_col（用于泄漏检测）。
    出参: QualityIssue 列表（block 级阻断建模，warn 级提示）。
    不变量: 确定性检查，不靠 LLM 判断。
    伪代码:
      issues = []
      profiles = dataset.columns
      for p in profiles:
          if p.null_rate > 0.95: issues.append(QualityIssue(p.name, "missing", f"null {p.null_rate:.0%}", "block"))
          elif p.null_rate > 0.5: issues.append(QualityIssue(p.name, "missing", ..., "warn"))
          if p.cardinality <= 1: issues.append(QualityIssue(p.name, "constant", "single value", "block"))
          if p.semantic_role == "categorical" and p.cardinality > 1000:
              issues.append(QualityIssue(p.name, "high_cardinality", f"{p.cardinality}", "warn"))
      # 重复列（值完全相同的两列）
      issues += _detect_duplicate_columns(backend, dataset_path, profiles)
      # 泄漏嫌疑：某特征与 target 相关性畸高（如 >0.95）
      if target_col:
          issues += _detect_leakage(backend, dataset_path, profiles, target_col)
      return issues
    """

def modeling_readiness(backend, dataset: Dataset, dataset_path: Path, *,
                      target_col: str, split_col: str | None) -> dict:
    """建模就绪评估：target 合法性、样本量、正负比、split 完整性、质量阻断项。
    出参: {ready: bool, blockers: [str], warnings: [str], stats: {...}}。
    伪代码:
      blockers, warnings = [], []
      df = backend.sample_rows(dataset_path, 50000, seed=0)
      # target 检查
      tvals = set(df[target_col].dropna().unique())
      if tvals - {0,1}: blockers.append("target must be binary 0/1")
      bad_rate = df[target_col].mean()
      if bad_rate < 0.005 or bad_rate > 0.5: warnings.append(f"imbalanced bad_rate {bad_rate:.2%}")
      if dataset.row_count < 1000: blockers.append("too few samples (<1000)")
      # split 检查
      if split_col and split_col in df.columns:
          present = set(df[split_col].unique())
          if "train" related values missing: blockers.append("missing train split")
      quality = check_data_quality(backend, dataset, dataset_path, target_col=target_col)
      blockers += [f"{i.column}: {i.detail}" for i in quality if i.severity == "block"]
      # 拒绝推断在 V1 不实现（Part J 桩）。贷前/前筛场景如果样本只含"通过"客群，模型在
      # 已批准人群上训练，对全申请人群有已知的样本偏差——这里给**告警**（不阻断），提醒
      # 用户：在拒绝推断落地前，效果指标只代表已批准客群。
      if _looks_accept_only(df):   # 无拒绝/未批标识列，或该列单一取值
          warnings.append("样本疑似仅含已批准客群；拒绝推断未实现，效果指标存在接受偏差，"
                          "解读时注意只代表已批准人群")
      return {"ready": not blockers, "blockers": blockers, "warnings": warnings,
              "stats": {"rows": dataset.row_count, "bad_rate": round(bad_rate,4)}}
    """


def _looks_accept_only(df) -> bool:
    """启发式：样本里没有"拒绝/未批/declined"标识列，或有但只剩单一取值（全是 approved）→
    判为疑似仅含已批准客群。只用于 warning，不阻断；列名走常见别名（approve_flag/decision/
    审批结果/是否通过 等）。"""
```

- **测试要点**：高缺失/常量列→block；重复列检出；泄漏嫌疑（与 y 畸高相关）检出；target 非二值→block；样本过少→block；就绪数据→ready=True。

---

## Part C — 建模样本准备（`packs/modeling/prepare.py`）

```python
def prepare_modeling_frame(registry, backend, dataset_id: str, *,
                          target_col: str, feature_cols: list[str],
                          split_col: str | None, split_config: dict | None,
                          seed: int = 0) -> Dataset:
    """准备建模样本：选列、处理 target、生成/校验 train/test/oot 划分，登记为新 Dataset。
    入参: dataset_id; target_col; feature_cols; split_col（已有划分列，None=自动划分）;
          split_config（自动划分时如 {"test_size":0.3,"oot_by_time":"date_col"}）; seed。
    出参: 建模就绪 Dataset（role="derived"，含 split 列）。
    异常: ModelingError（target 缺失/特征不存在）。
    不变量: 划分确定性（seed 固定）；时间外样本(OOT)按时间切，不随机混入。
    伪代码:
      df = backend.read_frame(registry.resolve_path(dataset_id),
                              columns=feature_cols + [target_col] + ([split_col] if split_col else []))
      _assert_columns_exist(df, feature_cols + [target_col])
      if split_col and split_col in df.columns:
          frame = df                              # 用已有划分
      else:
          frame = _make_split(df, split_config, seed)   # 自动：随机 train/test + 按时间 OOT
      out_path = _write_parquet(frame)
      return registry.register_existing(out_path, task_id=..., role="derived")
    """

def _make_split(df, split_config, seed) -> pd.DataFrame:
    """自动划分：先按时间切 OOT（若指定时间列），剩余随机分 train/test。
    伪代码:
      df = df.copy(); df["split"] = "train"
      if split_config.get("oot_by_time"):
          cutoff = df[time_col].quantile(1 - split_config.get("oot_size",0.2))
          df.loc[df[time_col] >= cutoff, "split"] = "oot"
      remaining = df[df["split"]=="train"].index
      test_idx = np.random.RandomState(seed).choice(remaining, int(len(remaining)*test_size), replace=False)
      df.loc[test_idx, "split"] = "test"
      return df
    """
```

- **测试要点**：用已有 split 列；自动划分 train/test 比例正确、可复现（seed）；OOT 按时间切不混入；缺特征/target 抛错。

---

## Part D — 特征筛选（`packs/modeling/select.py`）

```python
@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]   # (feature, reason)
    scores: dict                            # feature -> {iv, ks, vif}

def select_features(backend, dataset_path: Path, *, features: list[str], target_col: str,
                   iv_min: float = 0.02, corr_max: float = 0.8, vif_max: float = 10.0,
                   top_k: int | None = None, seed: int = 0) -> SelectionResult:
    """多准则特征筛选：低 IV 剔除 → 共线性剔除（保留 IV 高者）→ 高 VIF 剔除 → 可选 top_k。
    入参: features; target_col; 各阈值; top_k 上限; seed。
    出参: SelectionResult（保留/剔除/打分）。
    不变量: 复用 Phase 4 的 IV/相关/VIF（确定性核心）；筛选过程可解释、留剔除原因。
    伪代码:
      df = backend.read_frame(dataset_path, columns=features+[target_col])
      scores = {}
      kept = []
      dropped = []
      # 1) IV 门槛
      for f in features:
          m = feature_metrics(df[f].to_numpy(float), df[target_col].to_numpy(int), feature=f)
          scores[f] = {"iv": m.iv, "ks": m.ks}
          if m.iv < iv_min: dropped.append((f, f"low IV {m.iv:.3f}"))
          else: kept.append(f)
      # 2) 共线性：相关 > corr_max 的对里剔除 IV 低者
      pairs = find_collinear_pairs(correlation_matrix(df, kept), kept, threshold=corr_max)
      for f1, f2, c in pairs:
          loser = f1 if scores[f1]["iv"] < scores[f2]["iv"] else f2
          if loser in kept: kept.remove(loser); dropped.append((loser, f"collinear with {f1 if loser==f2 else f2} ({c:.2f})"))
      # 3) VIF
      vifs = vif(df, kept)
      for f, v in vifs.items():
          scores.setdefault(f,{})["vif"] = v
          if v > vif_max and f in kept: kept.remove(f); dropped.append((f, f"high VIF {v:.1f}"))
      # 4) top_k by IV
      if top_k: kept = sorted(kept, key=lambda x: -scores[x]["iv"])[:top_k]
      return SelectionResult(tuple(kept), tuple(dropped), scores)
    """
```

- **测试要点**：低 IV 剔除；共线对保留 IV 高者；高 VIF 剔除；top_k 截断；剔除原因可读；全程复用 Phase 4 确定性算法。

---

## Part E — 模型 recipe（`packs/modeling/recipes/`）

### E-1 recipe 注册表（`recipes/__init__.py`）

```python
_RECIPES: dict[str, ModelRecipe] = {}
def register_recipe(r: ModelRecipe) -> None: ...
def get_recipe(recipe_id: str) -> ModelRecipe: ...    # 异常 KeyError
def list_recipes() -> list[ModelRecipe]: ...
```

### E-2 LGB（`recipes/lgb.py`）

```python
def train_lgb(backend, dataset_path: Path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    """训练 LightGBM 模型，算 train/test/oot 指标，存产物。
    入参: backend; dataset_path（含 split 列）; config; out_dir 产物目录。
    出参: TrainResult（artifact + metrics + importance）。
    异常: ModelingError（split 缺失/训练失败）。
    不变量: seed 固定可复现；指标用 Phase 4 feature_ks/auc（INV-1）；OOT 指标算但不参与早停。
    伪代码:
      import lightgbm as lgb
      df = backend.read_frame(dataset_path)
      tr = df[df[config.split_col]==config.split_values["train"]]
      te = df[df[config.split_col]==config.split_values["test"]]
      oot = df[df[config.split_col]==config.split_values.get("oot")] if "oot" in config.split_values else None
      params = {**get_recipe("lgb").default_params, **config.params, "seed": config.seed}
      dtrain = lgb.Dataset(tr[list(config.features)], tr[config.target_col])
      dvalid = lgb.Dataset(te[list(config.features)], te[config.target_col])
      model = lgb.train(params, dtrain, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(config.early_stopping_rounds)] if config.early_stopping_rounds else [])
      # 预测分 + 指标（Phase 4 确定性核心）
      metrics = _compute_model_metrics(model, tr, te, oot, config)
      importance = _lgb_importance(model, config.features)
      artifact = _save_artifact(model, "lgb", config, out_dir, importance)
      return TrainResult(artifact=artifact, metrics=metrics,
                         feature_importance=importance, experiment_id="")
    """

def _compute_model_metrics(model, tr, te, oot, config) -> ModelMetrics:
    """对三个数据集预测打分，用 Phase 4 feature_ks/feature_auc 算指标 + 过拟合差值。
    不变量: INV-1——指标来自平台确定性核心；过拟合判断用 validation/overfitting。
    伪代码:
      def score(d): return model.predict(d[list(config.features)])
      tr_ks = feature_ks(score(tr), tr[config.target_col].to_numpy())
      te_ks = feature_ks(score(te), te[config.target_col].to_numpy())
      oot_ks = feature_ks(score(oot), oot[config.target_col].to_numpy()) if oot is not None else None
      ... auc 同理 ...
      gap_tt, gap_to, flag = overfitting_check(tr_ks, te_ks, oot_ks)   # validation/overfitting.py
      return ModelMetrics(tr_ks, te_ks, oot_ks, tr_auc, te_auc, oot_auc,
                          psi_test_vs_train=..., psi_oot_vs_train=...,
                          overfit_train_test_gap=gap_tt, overfit_train_oot_gap=gap_to, overfit_flag=flag)
    """
```

### E-3 XGB / LR（`recipes/xgb.py` / `recipes/lr.py`）

```python
def train_xgb(backend, dataset_path, config, *, out_dir) -> TrainResult:
    """同 LGB 结构，换 xgboost.train；共用 _compute_model_metrics / _save_artifact。"""

def train_lr(backend, dataset_path, config, *, out_dir) -> TrainResult:
    """逻辑回归（sklearn LogisticRegression）。建议先标准化/WOE（由 Plan 上游 tool 处理）。
    伪代码: 同结构，model=LogisticRegression(**params).fit(tr[features], tr[target])；
            score=model.predict_proba(...)[:,1]；指标同。
    """
```

### E-4 评分卡（`recipes/scorecard.py`）

```python
def train_scorecard(backend, dataset_path, config, *, out_dir,
                    base_score: int = 600, pdo: int = 50, base_odds: float = 50) -> TrainResult:
    """标准信贷评分卡：WOE 编码 → LR → 转标准分（base_score/PDO/odds）。
    入参: 同上 + base_score/pdo/base_odds 评分卡刻度参数。
    出参: TrainResult（artifact 含 woe_maps + 评分映射）。
    不变量: requires_woe=True；WOE 用 Phase 4；评分公式确定性。
    伪代码:
      df = backend.read_frame(dataset_path)
      tr, te, oot = _split(df, config)
      # 1) 每个特征在 train 上分箱 + WOE（Phase 4）
      woe_maps = {}; tr_woe = pd.DataFrame(index=tr.index)
      for f in config.features:
          edges = chimerge_edges(tr[f].to_numpy(float), tr[config.target_col].to_numpy(int), max_bins=6)
          binning = compute_woe_iv(tr[f].to_numpy(float), tr[config.target_col].to_numpy(int), edges, feature=f)
          woe = woe_result_from_binning(binning); woe_maps[f] = woe
          tr_woe[f] = woe_encode(tr, f, woe)
      # 2) LR on WOE
      lr = LogisticRegression(random_state=config.seed).fit(tr_woe, tr[config.target_col])
      # 3) 标准分映射（确定性公式）
      factor = pdo / np.log(2); offset = base_score - factor * np.log(base_odds)
      # 4) test/oot 用 train 的 woe_maps 编码（不重新分箱）→ 指标
      metrics = _scorecard_metrics(lr, woe_maps, tr, te, oot, config, factor, offset)
      artifact = _save_scorecard(lr, woe_maps, factor, offset, config, out_dir)
      return TrainResult(artifact, metrics, _lr_importance(lr, config.features), "")
    """
```

- **测试要点**：四种 recipe 各训出模型、产 metrics；同 seed 复现；OOT 指标算但不参与早停；评分卡 test/oot 用 train 的 WOE（不重新分箱）；过拟合差值/flag 正确；指标来自 Phase 4 核心（与单独算一致）。

---

## Part F — 实验记录（`packs/modeling/experiment.py`）

```python
class ExperimentStore:
    def __init__(self, db_path): self._db_path = db_path
    def create(self, task_id, recipe_id, config: TrainConfig) -> str:
        """建实验（status=created），返回 experiment_id。"""
    def attach_result(self, experiment_id, result: TrainResult) -> None:
        """挂训练结果（metrics + artifact_id，status=trained）。"""
    def set_status(self, experiment_id, status) -> None: ...
    def get(self, experiment_id) -> Experiment: ...
    def list_for_task(self, task_id) -> list[Experiment]: ...
    def compare(self, experiment_ids: list[str]) -> dict:
        """多实验对比表（各 metrics 并排），供模型选型。
        出参: {experiments:[{id,recipe,train_ks,test_ks,oot_ks,...,overfit_flag}]}。
        """
```

- **测试要点**：实验 CRUD；attach 后 status=trained；compare 并排多模型指标。

---

## Part G — 产物与 PMML（`packs/modeling/artifact.py`）

```python
def save_model(model, algorithm: str, out_dir: Path, *, feature_list, params, woe_maps=None) -> ModelArtifact:
    """存模型文件（按算法选格式：lgb→txt, xgb→json, lr/scorecard→pickle）+ 元数据。
    不变量: INV-5——模型文件存盘，不进记忆；路径相对 task_dir。
    """

def load_model(artifact: ModelArtifact):
    """按 algorithm 加载模型对象。异常: ModelingError（文件缺失/格式错）。"""

def export_pmml(artifact: ModelArtifact, dataset_path: Path, out_path: Path) -> Path:
    """导出 PMML（供验证流程的分数一致性比较）。
    入参: artifact; dataset_path（取 schema）; out_path。
    出参: pmml 路径。
    异常: ModelingError（算法不支持 PMML 导出）。
    不变量: PMML 是训练产物交接到 V1 验证流程的桥梁（roadmap V3）。
    伪代码:
      model = load_model(artifact)
      pipeline = _build_sklearn_pipeline(model, artifact)   # sklearn2pmml PMMLPipeline
      sklearn2pmml(pipeline, out_path)
      return out_path
    """
```

- **测试要点**：四种算法存取往返；export_pmml 产出可被 pypmml 读取；不支持的算法给清晰错误。

---

## Part H — 交接验证（`packs/modeling/handoff.py`）

```python
def handoff_to_validation(experiment_store, artifact: ModelArtifact, *,
                         sample_dataset_id: str, settings) -> str:
    """把训练好的模型交接到 V1 验证流程：导出 PMML + 准备样本 → 创建验证任务。
    入参: artifact; sample_dataset_id（建模样本）; settings。
    出参: 新建的验证 task_id。
    异常: ModelingError。
    不变量: roadmap V3——训练产物经验证才算可复核；训练上下文与验证契约独立。
    伪代码:
      pmml_path = export_pmml(artifact, ...)
      # 用现有 V1 任务创建逻辑：把 PMML + 样本 + 模型分作为验证材料
      validation_task_id = _create_validation_task(settings, pmml_path, sample_dataset_id, artifact)
      experiment_store.set_status(artifact.experiment_id, "handed_off")
      return validation_task_id
    """
```

- **不变量**：复用 V1 验证流程（不改 V1 行为，AGENTS.md V1 稳定约束）。交接后由 Phase 2 编排或手动模式跑验证。
- **测试要点**：交接产 PMML + 创建验证任务；实验状态→handed_off；验证跑完→validated。

---

## Part I — 场景参数模板（`packs/modeling/scenarios.py`）

```python
@dataclass(frozen=True)
class ScenarioTemplate:
    id: str
    description: str
    target_type: str        # binary | continuous —— 收入是回归(continuous)，其余多为 binary
    default_recipe: str     # 默认 recipe（lgb/xgb/lr/scorecard/lgb_regressor）
    target_hint: str        # target 口径提示
    param_overrides: dict    # 该场景推荐的训练参数
    feature_guidance: str    # 特征选取建议（解释性，不强制）
    eval_metric: str        # 该场景主指标：ks_auc(分类) | rmse_mae(回归) | response_lift(营销)
    notes: str              # 方法论提示/坑

# 全部场景显式定义（不再"同构扩展"）。不同场景=参数+口径+目标类型差异，共用同一训练/指标核心。
SCENARIO_TEMPLATES = {
    "loan_pre_a":   ScenarioTemplate("loan_pre_a", "贷前A卡（准入评分）", "binary", "scorecard",
                       "首逾FPD/30+ @ mob", {"max_depth": 3}, "强解释性、单调分箱、跨渠道稳定",
                       "ks_auc", "评分卡优先；OOT KS 衰减需关注"),
    "pre_screen":   ScenarioTemplate("pre_screen", "前筛（粗筛坏客户）", "binary", "lgb",
                       "坏客户/拒绝后表现", {}, "外部数据源覆盖率优先", "ks_auc",
                       "高拒绝率场景，注意样本偏差（拒绝推断另议）"),
    "loan_in":      ScenarioTemplate("loan_in", "贷中（行为评分B卡）", "binary", "lgb",
                       "未来N期逾期", {}, "行为/还款/支用类特征为主", "ks_auc",
                       "观察期+表现期定义要清晰；账龄影响"),
    "loan_post":    ScenarioTemplate("loan_post", "贷后（早期预警/催收前）", "binary", "lgb",
                       "滚动恶化/进入M2+", {}, "近期还款行为+逾期状态迁移", "ks_auc",
                       "目标常是状态迁移，配合 roll_rate"),
    "marketing":    ScenarioTemplate("marketing", "营销响应", "binary", "lgb",
                       "是否响应/转化", {}, "触达/活跃/历史响应特征", "response_lift",
                       "目标是响应非风险；用 lift/响应率评估，不混淆 KS"),
    "transaction":  ScenarioTemplate("transaction", "交易（反欺诈/异常）", "binary", "lgb",
                       "欺诈/异常交易", {"scale_pos_weight": "auto"}, "交易序列/设备/关系网特征",
                       "ks_auc", "极度不平衡，关注 recall/精确率而非只看 KS"),
    "recall":       ScenarioTemplate("recall", "捞回（流失/沉睡唤醒）", "binary", "lgb",
                       "唤醒后用信/复借", {}, "沉睡时长/历史用信/营销触达", "response_lift",
                       "目标是被唤醒后的正向行为，类营销口径"),
    "income":       ScenarioTemplate("income", "收入预测", "continuous", "lgb_regressor",
                       "月收入/可支配收入(连续值)", {"objective": "regression"},
                       "工资/公积金/交易流水/消费类特征", "rmse_mae",
                       "回归任务：指标用 RMSE/MAE/R²，不是 KS/AUC；需 continuous recipe"),
    "credit_limit": ScenarioTemplate("credit_limit", "额度", "binary", "lgb",
                       "提额后风险/用信意愿", {}, "额度使用率/收入/风险分", "ks_auc",
                       "常与额度策略联动（Phase 7）"),
    "pricing":      ScenarioTemplate("pricing", "定价（风险定价）", "binary", "lgb",
                       "风险分层支撑定价", {}, "风险+价格敏感度特征", "ks_auc",
                       "输出风险分供定价策略用（Phase 7）"),
}

def get_scenario(scenario_id: str) -> ScenarioTemplate: ...
def list_scenarios() -> list[ScenarioTemplate]: ...
def apply_scenario(config: TrainConfig, scenario_id: str) -> TrainConfig:
    """把场景推荐参数 + 目标类型合并进 TrainConfig（用户可覆盖）。
    不变量: 回归场景（income）强制 target_type=continuous + 回归指标，分类场景用 KS/AUC；
            apply 时校验 recipe 与 target_type 匹配（continuous 必须用回归 recipe）。
    伪代码:
      tpl = get_scenario(scenario_id)
      cfg = replace(config, params={**tpl.param_overrides, **config.params})
      _assert_recipe_matches_target(tpl.default_recipe if not config.recipe else config.recipe, tpl.target_type)
      return cfg
    """
```

- **不变量**：场景模板只提供推荐参数、目标类型和口径提示（解释性），**不绕过确定性指标计算**；不同场景=参数+目标类型差异，共用同一训练/指标核心。**收入是回归任务（连续目标），指标走 RMSE/MAE/R² 而非 KS/AUC**——这是与其它场景的关键差异，需 `lgb_regressor` recipe 和回归指标分支（E 部分 recipes 增加回归 recipe；Phase 4 指标核心增加回归指标）。
- **营销/捞回**用 lift/响应率评估，不混 KS。**交易**极度不平衡，关注 recall/精确率。
- **测试要点**：9 个场景全部可取/可列；`apply_scenario` 合并参数 + 用户覆盖优先；**回归场景强制 continuous recipe + 回归指标**；分类场景用 KS/AUC；recipe 与 target_type 不匹配时报错。

---

## Part J — 拒绝推断（`packs/modeling/reject_inference.py`，桩）

```python
def reject_inference(*args, **kwargs):
    """拒绝推断（Heckman/平行分配/增广/模糊增广）。
    **V1 不实现——标注需方法论评审。** 直接调用抛 NotImplementedError 并提示。
    不变量: 方法论复杂，做错会系统性偏置模型；必须先经方法论确认再实现。
    伪代码:
      raise NotImplementedError(
        "reject inference requires methodology review before implementation; "
        "see blueprint 15.1. Candidate methods: Heckman / parceling / augmentation / fuzzy augmentation.")
    """
```

- **不变量**：manifest **不**把 reject_inference 注册为可用 tool（Planner 选不到），仅留模块占位 + 文档说明。
- **测试要点**：调用抛 `NotImplementedError` 带方法论提示；不出现在 tool registry。

---

## Part K — modeling 能力包（`packs/modeling/tools.py`）

```python
def tool_check_data_quality(inputs, ctx) -> dict:
    """inputs:{dataset_id, target_col?}。output:{issues:[{column,kind,detail,severity}]}。determinism=deterministic。"""

def tool_modeling_readiness(inputs, ctx) -> dict:
    """inputs:{dataset_id, target_col, split_col?}。output:{ready, blockers, warnings, stats}。"""

def tool_prepare_modeling_frame(inputs, ctx) -> dict:
    """inputs:{dataset_id, target_col, feature_cols, split_col?, split_config?, seed?}。
       output:{result_dataset_id, split_counts}。determinism=stochastic（seed）。"""

def tool_select_features(inputs, ctx) -> dict:
    """inputs:{dataset_id, features, target_col, iv_min?, corr_max?, vif_max?, top_k?}。
       output:{selected, dropped:[[f,reason]], scores}。"""

def tool_train_model(inputs, ctx) -> dict:
    """统一训练入口，按 recipe 分派 lgb/xgb/lr/scorecard。
    inputs:{dataset_id, recipe, features, target_col, split_col, split_values, params?, seed, scenario?}。
    output:{experiment_id, artifact_id, metrics:{train_ks,test_ks,oot_ks,...,overfit_flag}, feature_importance}。
    determinism=stochastic（seed 必填）。
    不变量: INV-1——metrics 来自 Phase 4 核心；overfit_flag 来自 validation/overfitting。
    """

def tool_compare_experiments(inputs, ctx) -> dict:
    """inputs:{experiment_ids:[str]}。output:{experiments:[{id,recipe,metrics...}]}。"""

def tool_export_pmml(inputs, ctx) -> dict:
    """inputs:{artifact_id}。output:{pmml_path}。"""

def tool_handoff_to_validation(inputs, ctx) -> dict:
    """inputs:{experiment_id, sample_dataset_id}。output:{validation_task_id}。
    不变量: roadmap V3——产物经验证才可复核。"""
```

`manifest.json`：`tool_train_model` 标 `determinism="stochastic"`、`side_effects=["write:model","write:dataset"]`、较长 `timeout_seconds`；`tool_handoff_to_validation` 在 Plan 里建议 `needs_confirmation=True`（产生新验证任务，人工确认）。

- **测试要点**：每个 tool 经 runner 子进程往返；`tool_train_model` 四 recipe 都能训、产结构化 metrics；seed 复现；handoff 产验证任务。

---

## Part L — 持久层 + 过拟合迁移

```sql
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, recipe_id TEXT NOT NULL,
  config_json TEXT NOT NULL, metrics_json TEXT, artifact_id TEXT,
  status TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_artifacts (
  id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, algorithm TEXT NOT NULL,
  model_path TEXT NOT NULL, pmml_path TEXT, feature_list_json TEXT NOT NULL,
  params_json TEXT NOT NULL, woe_maps_json TEXT, created_at TEXT NOT NULL,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);
```

**过拟合检测迁移**（CODE_REVIEW P2-27）：新建 `validation/overfitting.py`，把 `api.py` 的 `_agent_overfitting_check`/阈值常量迁入：

```python
# validation/overfitting.py
OVERFIT_TRAIN_TEST_REL = 0.10
OVERFIT_TRAIN_OOT_ABS = 0.05
def overfitting_check(train_ks: float, test_ks: float, oot_ks: float | None) -> tuple[float, float|None, bool]:
    """返回 (train_test 相对差, train_oot 绝对差, 是否过拟合 flag)。
    伪代码:
      gap_tt = abs(train_ks - test_ks) / train_ks if train_ks else 0.0
      gap_to = abs(train_ks - oot_ks) if oot_ks is not None else None
      flag = gap_tt > OVERFIT_TRAIN_TEST_REL or (gap_to is not None and gap_to > OVERFIT_TRAIN_OOT_ABS)
      return gap_tt, gap_to, flag
    """
```

`api.py` 改为 import 此函数（修 CODE_REVIEW P2-27 模块边界）。

- **测试要点**：experiments/artifacts 往返、FK CASCADE；`overfitting_check` 阈值正确；api.py 复用后行为不变（回归）。

---

## Part M — 测试计划汇总

| 文件 | 覆盖 |
|------|------|
| `tests/test_modeling_contracts.py` | dataclass 往返 |
| `tests/test_modeling_readiness.py` | 质量问题/就绪评估/泄漏检测 |
| `tests/test_modeling_prepare.py` | split 复现、OOT 按时间、缺列报错 |
| `tests/test_modeling_select.py` | IV/共线/VIF/top_k 筛选、剔除原因 |
| `tests/test_modeling_recipes.py` | **四 recipe 训练、seed 复现、指标来自 Phase 4、评分卡 WOE 复用** |
| `tests/test_modeling_overfitting.py` | 过拟合差值/flag、api 迁移回归 |
| `tests/test_modeling_experiment.py` | 实验 CRUD、compare |
| `tests/test_modeling_artifact.py` | 存取、export_pmml 可被 pypmml 读 |
| `tests/test_modeling_handoff.py` | 交接产验证任务、状态流转 |
| `tests/test_modeling_scenarios.py` | 场景模板、参数合并 |
| `tests/test_modeling_pack.py` | 8 个 tool 经 runner 往返 |
| `tests/test_modeling_reject_inference.py` | 抛 NotImplementedError、不在 registry |

---

## Part N — 任务执行顺序

```text
1. A 契约
2. L DB + overfitting 迁移（含 api.py 回归）
3. B readiness
4. C prepare
5. D select（依赖 Phase 4）
6. E recipes（依赖 Phase 4 + lgb/xgb/sklearn；核心）
7. F experiment
8. G artifact + PMML
9. H handoff（依赖 V1 验证流程）
10. I scenarios
11. J reject_inference 桩
12. K modeling pack tools（依赖全部 + Phase 1/3）
13. M 测试 + 回归
```

每项 atomic commit。Phase 6 完成标志：能对建模样本做质量/就绪检查、筛特征、用四种 recipe 训练（seed 复现、指标走 Phase 4 确定性核心、过拟合 flag 来自 validation/overfitting）、记录对比实验、导出 PMML 交接到 V1 验证流程；拒绝推断仅留桩 + 方法论评审标注；8 个 tool 经子进程 runner 可用。

---

## Part O — 模型开发报告自动生成（7 页分析报告契约）

> 目标：模型训练完成（`train_model` → `log_experiment`）后**自动**产出一份与标准模型开发报告**格式 & 内容一致**的 Excel（7 个 sheet）。模板基于真实样本 `复借T卡_多维度融合风险模型`（2026-06-13 拆解）。
> **铁律 INV-1**：报告里每一个数字（KS/AUC/PSI/IV/lift/vintage/压测）由平台 tool 算；「汇总」叙事页的章节文字可由 LLM 起草，但 LLM 不碰任何数字。

### O-1 报告结构与数据来源映射

```text
汇总（叙事页，6 章）：
  一、建模背景  1.1项目概述(项目元数据+指标表) 1.2数据集划分 1.3稳定性指标 1.4建模目标
  二、样本分析结论  2.1样本时间分布 2.2关键结论
  三、Vintage分析结论  3.1Vintage表现摘要 3.2风险走势分析
  四、模型结论  4.1区分能力 4.2OOT十分箱表现 4.3Top10特征重要性 4.4应用建议
  五、使用产品清单
  六、压力测试  6.1产品缺失 6.2低定价人群占比提升
明细页：样本分析 · Vintage · 特征重要性 · oot分箱评估_十分箱 · 单变量分析 · 压力测试
```

| 报告位置 | 数据来源 | 现状 |
|---------|---------|------|
| 汇总 1.2 数据集划分（split/窗口/样本量/KS/AUC/bad_rate） | validation effectiveness overall | ✅ 现成 |
| 汇总 1.3 稳定性（Stability gap/PSI + 约束 + 达标） | `validation/overfitting` + validation PSI | ✅ 现成 |
| 汇总 1.1 指标表 / 1.4 目标 / 项目概述文字 | `project_meta`（用户填）+ dataset row_counts | 需用户元数据 |
| 汇总 2/3/4.4 叙事结论 | LLM 起草（基于对应明细页结构化数据，不编数字） | LLM 文案 |
| 汇总 4.1 区分能力 / 4.2 十分箱 / 4.3 Top10 | validation KS/AUC + bin table + feature_importance | ✅ 现成 |
| 汇总 五 使用产品清单 | feature dictionary 去重「产品/厂商」 | 需特征字典 |
| 单变量分析页（coverage/IV/KS×split） | Phase 4 `compute_feature_metrics` | ✅ 现成 |
| 特征重要性页（gain/百分比/累计 + 含义/产品/厂商） | train_model importance + **feature dictionary** | gain 现成；含义/产品/厂商需字典 |
| oot分箱评估页（样本/逾期率/lift + **金额维度/额度/利率/件均**） | bin table + **业务列扩展** | 人头维度现成；金额维度需业务列 |
| 样本分析页（放款月 × **利率/金额/期数/支用/Mob3/Mob6逾期率**） | **`compute_sample_analysis`（新）** + 业务列 | 新增 |
| Vintage 页（放款月 × MOB 曲线 fpd1/fpd30/mob2-9） | **`compute_vintage_report`（新，复用 Phase 4V 核心）** | 新增 |
| 压测页 6.1 产品缺失 | `validation/stress_test`（按数据源移除） | ✅ 现成 |
| 压测页 6.2 低定价人群占比提升 | **`stress_low_pricing`（新）** | 新增 |

### O-2 业务列契约 + 缺失策略（用户已确认）

```python
@dataclass(frozen=True)
class BusinessColumns:
    loan_month_col: str | None        # 放款月（样本分析/Vintage 的 cohort）
    interest_rate_col: str | None     # 利率（低定价/样本分析）
    loan_amount_col: str | None       # 放款金额
    term_col: str | None              # 期数
    drawdown_amount_col: str | None   # 支用金额
    credit_limit_col: str | None      # 授信额度（额度使用率）
    mob_observe_cols: tuple[str, ...] = ()   # mob1..mob9 逾期观察列（Vintage）

@dataclass(frozen=True)
class ReportSectionStatus:
    section: str                      # sample_analysis|vintage|amount_bin|low_pricing|product_list
    available: bool
    reason: str | None                # 不可用原因，如 "缺少业务列: loan_month_col, mob_observe_cols"

def resolve_report_sections(business: BusinessColumns | None,
                            dictionary_id: str | None) -> list[ReportSectionStatus]:
    """根据可用业务列/字典，判定每个需业务数据的章节是否可生成。
    出参: 各章节可用性 + 缺失原因。
    策略（用户确认）: 缺业务列 → 该章节标记 unavailable；report 生成时该 sheet 留空 + 写"无业务数据（缺 X 列）"，
                     不阻断整份报告；可用章节正常出。
    伪代码:
      out = []
      need = {
        "sample_analysis": ["loan_month_col"],
        "vintage": ["loan_month_col", "mob_observe_cols"],
        "amount_bin": ["loan_amount_col"],   # 金额维度分箱
        "low_pricing": ["interest_rate_col"],
        "product_list": [],                   # 需字典
      }
      for sec, cols in need.items():
          missing = [c for c in cols if not _has(business, c)]
          if sec == "product_list" and not dictionary_id: missing.append("feature_dictionary")
          out.append(ReportSectionStatus(sec, not missing,
                     None if not missing else f"缺少业务列/字典: {missing}"))
      return out
    """
```

> **缺失交互（用户确认）**：生成报告前，若检测到 `available=False` 的章节，Agent 先**问用户**是否补充业务列；用户不补 → 对应 sheet 留空并标 `无业务数据（缺 X）`，其余照常生成。

### O-3 新增计算工具

```python
# packs/modeling/report_compute.py

def compute_sample_analysis(backend, dataset_path: Path, *, loan_month_col: str,
                            target_col: str, business: BusinessColumns,
                            mob_cols: tuple[str, ...]) -> list[dict]:
    """按放款月聚合业务+风险指标（样本分析 sheet）。
    出参: 每月一行 {放款月, 放款笔数, 平均利率, 平均放款金额, 平均期数, 平均支用金额,
                   未逾期/逾期/逾期率, Mob3逾期率, Mob6逾期率}。
    不变量: 纯聚合确定性；缺某业务列则该列省略（不编值）。
    伪代码:
      df = backend.read_frame(dataset_path, columns=_needed(business, target_col, loan_month_col, mob_cols))
      rows = []
      for month, g in df.groupby(loan_month_col):
          row = {"放款月": month, "放款笔数": len(g),
                 "逾期率": g[target_col].mean(), ...}
          if business.interest_rate_col: row["平均利率"] = g[business.interest_rate_col].mean()
          if business.loan_amount_col:   row["平均放款金额"] = g[business.loan_amount_col].mean()
          # Mob3/Mob6 逾期率：用对应 mob 观察列
          ...
          rows.append(row)
      return rows
    """

def compute_vintage_report(backend, dataset_path: Path, *, loan_month_col: str,
                           mob_observe_cols: tuple[str, ...], amount_col: str | None) -> dict:
    """Vintage：按放款月 cohort × MOB 的累计逾期率曲线（Vintage sheet）。
    出参: {cohorts, curves:{放款月: [各 mob 累计逾期率]}, headers:[fpd1,fpd30,mob2..mob9], counts, amounts}。
    不变量: 复用共享确定性核心 `validation/vintage.py`（Phase 4V，先于 Phase 6/7 交付，
            三处 import 同一份，避免依赖倒置）；累计单调。
    伪代码: points = validation.vintage.compute_vintage_curve(...)；
            curves = validation.vintage.vintage_curve_wide(points, metric="cum_bad_rate")；
            再补放款金额/件均/利率等 cohort 业务列。
    """

def compute_amount_bin_table(backend, dataset_path: Path, *, score_col: str, target_col: str,
                             edges, business: BusinessColumns) -> list[dict]:
    """OOT 十分箱评估扩展业务维度（oot分箱评估 sheet）。
    出参: 每箱 {分箱区间, 样本数, 累积样本占比, 好/坏人数, 人头逾期率/累积/lift,
              金额逾期率/累积/lift, 平均利率, 平均期数, 平均授信金额, 平均放款金额, 额度使用率}。
    不变量: 人头维度复用 Phase 4 bin_table；金额维度/额度使用率需 business 列，缺则省略对应列。
    伪代码: 在 Phase 4 bin_table 基础上，按箱聚合 business 列的金额加权逾期率与额度使用率。
    """

def stress_low_pricing(backend, dataset_path: Path, *, score_col: str, target_col: str,
                       interest_rate_col: str, low_pricing_threshold: float | None,
                       ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)) -> dict:
    """低定价人群占比提升压测（压测 sheet 6.2）。
    入参: interest_rate_col; low_pricing_threshold（None=用利率中位数）; ratios 提升占比序列。
    出参: {threshold, bins_by_ratio:{ratio: 十分箱累计占比}, ks_by_ratio, psi_by_ratio, conclusion_data}。
    不变量: INV-1——KS/PSI 由 Phase 4 核心算；只重采样人群占比，不改模型分。
    伪代码:
      thr = low_pricing_threshold or df[interest_rate_col].median()
      base = _bin_dist(df, score_col, edges)
      out = {}
      for r in ratios:
          resampled = _oversample_low_pricing(df, interest_rate_col, thr, target_ratio=r)
          out[r] = {"ks": feature_ks(resampled[score_col], resampled[target_col]),
                    "psi": compute_psi(base, _bin_dist(resampled, score_col, edges)),
                    "bins": _bin_cumpct(resampled, score_col, edges)}
      return {"threshold": thr, "by_ratio": out}
    """

def build_feature_dictionary(backend, dict_dataset_id, registry) -> dict:
    """从数据字典构建特征元数据（含义/产品/厂商），供特征重要性页和使用产品清单。
    出参: {feature_name: {含义, 产品名称, 厂商名称}}。
    不变量: 字典缺失则含义/产品/厂商列留空，不阻断报告。
    """
```

### O-4 报告渲染器（`output/model_report.py`）

```python
@dataclass(frozen=True)
class ModelReportPayload:
    project_meta: dict                 # 项目概述/标签定义/建模目标（用户填）
    dataset_split: list[dict]          # 数据集划分（含 KS/AUC/bad_rate）
    stability: list[dict]              # 稳定性指标 + 达标
    sample_analysis: list[dict] | None
    vintage: dict | None
    feature_importance: list[dict]     # gain + 含义/产品/厂商
    univariate: list[dict]             # IV/KS×split
    oot_bin_table: list[dict]
    stress_product_removal: dict
    stress_low_pricing: dict | None
    narratives: dict                   # LLM 起草的章节文字（建模背景/结论/建议）
    section_status: list[ReportSectionStatus]

def render_model_report(payload: ModelReportPayload, out_path: Path) -> Path:
    """渲染 7-sheet Excel，匹配标准模板（合并单元格、分章节布局、列顺序一致）。
    入参: payload（全部结构化数据 + LLM 文案）; out_path。
    出参: xlsx 路径。
    不变量: 复用 output/ 的 openpyxl 样式工具；数字来自 payload（平台算），文案来自 narratives；
            section_status 中 unavailable 的 sheet 写"无业务数据（缺 X）"。
    伪代码:
      wb = Workbook()
      _write_summary_sheet(wb, payload)          # 汇总 6 章（含指标表/划分表/稳定性表 + 叙事）
      _write_sample_analysis_sheet(wb, payload)  # 缺则空表 + 无业务数据标注
      _write_vintage_sheet(wb, payload)
      _write_feature_importance_sheet(wb, payload)
      _write_oot_bin_sheet(wb, payload)
      _write_univariate_sheet(wb, payload)
      _write_stress_sheet(wb, payload)           # 6.1 产品缺失 + 6.2 低定价
      wb.save(out_path); return out_path
    """
```

### O-5 编排工具 + Hook

```python
# packs/modeling/tools.py 新增
def tool_generate_model_report(inputs, ctx) -> dict:
    """模型开发报告自动生成入口。
    inputs: {experiment_id, dataset_id, business_columns?, feature_dictionary_id?, project_meta}。
    output: {report_path, section_status:[{section,available,reason}]}。
    determinism=stochastic（low_pricing 重采样带 seed）；数字部分确定性。
    不变量: INV-1——数字平台算，narratives 由 LLM 起草且过 _guard（不得出现 payload 外的数字）。
    伪代码:
      exp = experiment_store.get(inputs["experiment_id"]); artifact = ...
      business = BusinessColumns(**inputs.get("business_columns", {}))
      status = resolve_report_sections(business, inputs.get("feature_dictionary_id"))
      # 1) 平台算的部分（全部确定性）
      validation = _load_or_run_validation(exp)          # split KS/AUC/bad_rate/PSI/stability/bin/stress
      univariate = compute_feature_metrics_batch(...)    # Phase 4
      sample = compute_sample_analysis(...) if _ok(status,"sample_analysis") else None
      vintage = compute_vintage_report(...) if _ok(status,"vintage") else None
      low_pricing = stress_low_pricing(...) if _ok(status,"low_pricing") else None
      fdict = build_feature_dictionary(...) if inputs.get("feature_dictionary_id") else {}
      # 2) LLM 起草叙事（仅文字，过 INV-1 守卫）
      narratives = _draft_report_narratives(structured_summary, ctx)   # 建模背景/2.2/3.2/4.4 文字
      narratives = _guard_no_invented_numbers(narratives, structured_summary)
      # 3) 渲染
      payload = ModelReportPayload(project_meta=inputs["project_meta"], dataset_split=validation.overall,
                                   stability=..., sample_analysis=sample, vintage=vintage,
                                   feature_importance=_join_importance_dict(exp.importance, fdict),
                                   univariate=univariate, oot_bin_table=compute_amount_bin_table(...),
                                   stress_product_removal=validation.stress, stress_low_pricing=low_pricing,
                                   narratives=narratives, section_status=status)
      report_path = render_model_report(payload, out_dir / "model_report.xlsx")
      return {"report_path": str(report_path), "section_status": [asdict(s) for s in status]}
    """
```

- **Hook**：在 modeling 包注册 `feature.computed`/自定义 `experiment.trained` 事件后触发，或编排把 `train_model → generate_model_report` 编进模型开发 Workflow 模板（report step 标 `needs_confirmation=True`，与现有报告生成一致）。
- **INV-1 守卫**：`_guard_no_invented_numbers` 扫描 narratives，凡出现 structured_summary 之外的数字（KS/AUC/百分比等）即替换为占位或回退模板句，防 LLM 编数。

### O-6 测试 + 任务顺序（并入 Phase 6 的 M/N）

- 测试（新增）：
  - `compute_sample_analysis` 按月聚合正确、缺业务列省略对应列。
  - `compute_vintage_report` 累计单调、与 Phase 7 `vintage_curve` 数值一致。
  - `compute_amount_bin_table` 金额维度加权逾期率正确、缺列降级。
  - `stress_low_pricing` KS/PSI 来自 Phase 4 核心、重采样可复现（seed）。
  - `resolve_report_sections` 缺列→unavailable + 原因。
  - `render_model_report` 产出 7 sheet、列顺序与模板一致、unavailable sheet 标"无业务数据"。
  - `tool_generate_model_report` 经 runner 往返；**INV-1 守卫**：narratives 不含 payload 外数字。
  - 缺业务列端到端：报告仍生成、对应章节空白标注、可用章节正常。
- 任务顺序：在 Part N 之后追加 → `report_compute`（新计算工具）→ `output/model_report`（渲染器）→ `tool_generate_model_report`（编排）→ 模型开发 Workflow 模板挂 report step → 测试。
- 依赖：Phase 4（指标核心）、**Phase 4V**（共享 vintage 核心 `validation/vintage.py`，`compute_vintage_curve` + `vintage_curve_wide`；先于 Phase 6/7 交付，三处 import 同一份）、Phase 3（业务列来自 dataset/字典）。

---

*Phase 6 把建模能力工具化，但守住两条线：指标永远由平台确定性核心算（不信任训练框架自报的指标），产物必须经 V1 验证流程才算可复核。拒绝推断这类高风险方法论留到方法论确认后再做。模型开发报告（Part O）让"训练完自动出一份标准 7 页分析报告"成为契约：数字平台算、文案 LLM 起草、缺业务列优雅降级。*
