# S1a 分数方向制度化 —— 函数级 Spec

> 依据：[v2-strategy-risk-analysis-plan.md](../v2-strategy-risk-analysis-plan.md) §二 S-3、§七 S1a 行；[v2-master-backlog.md](../v2-master-backlog.md) §4（S1a/DOM-2/NEW-2）；[2026-07-02 综合改进审查](../../reviews/2026-07-02-v2-comprehensive-improvement-review.md) DOM-2 小节。
> 状态：待审（草稿），未开发。分支 `codex/v2-plugin-tool-runtime`，行号对应本分支当前工作区。
> 范围声明：本 spec **只做 S1a**（方向制度化本体，含 NEW-2 的 validation lift 方向裁决）。`score_dataset`/`monitor_run`（S1b）与训练期分布快照契约变更**只在本 spec 预留字段位置**，不展开实现——两者仍是独立 spec。

---

## 0. 地面真值核对（写 spec 前实读代码，供审阅者复核）

逐文件核对，行号为当前工作区实测（不是审查报告的历史行号，个别已漂移）：

| 文件 | 现状 | 与计划假设的出入 |
|---|---|---|
| `marvis/packs/strategy/tradeoff.py:11-20,38` | `tradeoff_view(df,*,score_col,target_col,cutoffs=None,profit_params=None,ead_col=None,pd_col=None)`；`for approved in [scores >= cutoff]`（L38）硬编码"高分=通过" | 与计划一致 |
| `marvis/packs/strategy/manifest.json` | `tradeoff_view`/`build_strategy`/`backtest_strategy` 三个 input_schema 均 `additionalProperties: false`，且**都没有**任何方向类参数 | 与计划一致 |
| `marvis/packs/modeling/reject_inference.py:29-40,199-204` | `_risk_order(frame, score_col) -> np.ndarray`；L204 `np.argsort(-safe_scores, kind="mergesort")` 硬编码"高分=高风险" | 与计划一致；`_fuzzy_augment_rejected`（L173-180）**不吃 score_col**，只用全局 bad_rate 加权（DOM-12 (b) 的证据，本 spec 不修，仅记录不受影响） |
| `marvis/packs/strategy/strategy.py` 全文 | **没有 `build.py` 这个文件**；`build_strategy`/`apply_strategy` 都在 `strategy.py`。规则条件是**原始 Python 表达式字符串**（`StrategyRule.condition: str`），经 `ast.parse` 解析，比较运算符是 AST 节点类型（`ast.Lt/LtE/Gt/GtE/Eq/NotEq/In/NotIn`），**不是**独立的 `operator` 字段/枚举；分数列与其他特征列在 schema 层**不做区分**——`"score>=650"` 与 `"income>=5000"` 结构完全一样 | **计划术语"规则算子"需澄清**：不存在可枚举的算子字段，一致性自检必须从 `condition` 字符串里**解析**出引用 `strategy.score_col` 的比较子句及其运算符方向 |
| `marvis/packs/strategy/contracts.py:37-49` | `StrategyRule(condition, decision, value)`；`Strategy(id, strategy_type, rules, score_col: str\|None, default_decision, description)`——`score_col` 是可选字段 | `build_strategy` manifest 的 `score_col` 不在 `required` 里——一致性自检只在 `score_col` 非空时才可能触发 |
| `marvis/packs/modeling/tools.py:4414-4461`（`_ModelArtifactScorer`） | `raw_score()`：xgb 走 `predict.(matrix)` 原始值；scorecard 走 WOE→logit→`predict_proba[:,1]`；其余凡有 `predict_proba` 一律取 `[:,1]`；`scorecard_points()`（L4463-4480）走 `offset - factor*logits`（**独立的、方向相反的**评分卡习惯：高分=低风险） | **审查未点出的额外发现**：`raw_score()` 与 `scorecard_points()` 在同一个类里已经是两套互相矛盾的方向（前者高分=坏，后者高分=好），本 spec 必须把两者都纳入"打分产物携带 direction 元数据"的范围，而不能只当 `raw_score` 一种口径 |
| `marvis/packs/modeling/tools.py:997-1038,1156-1172`（原审查称 L789/929，行号已漂移） | `_pick_best_experiment`/`_pick_best_comparison_row` 只按 KS/AUC（方向无关的对称指标）挑冠军，不涉及分数方向 | 确认**不需要改动**——本 spec 不动这两个函数 |
| `marvis/packs/modeling/artifact.py:96-145`（`persist_model_meta`） | meta dict 用显式键值对构造（不是 `**kwargs`），当前键：`artifact_id/algorithm/model_path/pmml_path/feature_list/params/seed/dataset_id/target_col/split_col/split_values/target_type/recipe_id/scorecard_table/created_at` | 加 `score_direction` 键是纯 additive，**读取方无需容错改造**（因为读取方本来就是显式 `.get(key)`/具名解包，见下一行） |
| `marvis/repositories/modeling.py:355-373`（`_model_artifact_from_row`） | 显式具名字段构造 `ModelArtifact(id=..., ...)`，**不是** `**row` 解包 | 加字段是安全的，但**运行时打分路径读的是 SQLite `model_artifacts` 表，不是 `.model_meta.json` 文件**——`.model_meta.json` 只是 handoff/审计侧信道（`marvis/recovery.py:32` 只 `.exists()` 判定，不解析内容）。**这与计划"读取方兼容"的表述有出入**：真正需要改的读取方是 `_model_artifact_from_row` + DB schema 迁移，不是文件解析器 |
| `marvis/db_schema.py:10-18,619-632`（`_MIGRATION_TABLES`/`_ensure_column`） | 已有"加列不加版本号"的迁移机制（`ALTER TABLE ... ADD COLUMN`），`model_artifacts` **已在**允许迁移的表名单里（此前 `feature_importance_json`/`scorecard_table_json` 就是这么加的） | 本 spec 可直接复用该机制，无需新造迁移框架 |
| `marvis/data/errors.py:28-66`、`marvis/plugins/subprocess_worker.py:53-68,603-622` | `NanLabelNotConfirmedError.to_detail()` 返回 `{"kind": "nan_label_not_confirmed", ...}`；子进程 worker 在捕获异常时若其对象有可调用的 `to_detail()`，直接把 `detail["kind"]` 提升为顶层 `error_kind`——**这是通用机制，不是 NaN 门专属** | `score_direction_conflict` 只需照此模式定义一个新 typed error 类，**不需要触碰 `subprocess_worker.py`** |
| `marvis/feature/metrics.py:109-145`（`head_tail_lift`） | **已经**是方向自适应实现：`risk_sign = sign(safe_correlation(scores,target)) or 1.0`，按 `risk_sign*scores` 排序 | 这是"做对的参照系"，NEW-2 要把 `validation/effectiveness.py` 那份改成与它一致的口径（不是发明新算法） |
| `marvis/validation/effectiveness.py:413-430`（`compute_head_tail_lift`） | **无方向感知**：`order = np.argsort(scores)[::-1]` 硬编码降序（假设高分=高风险）；被 L97、L261 两处调用（`compute_overall_ks` 内部，产出 `OverallRow.head_lift_5pct`/`tail_lift_5pct`） | 与计划 NEW-2 描述一致，这就是要修的点 |
| `marvis/feature/correlation.py:34-39`（`safe_correlation`） | Pearson 相关，`std=0` 或样本 `<2` 时返回 `0.0`（不抛错）；Spearman 需调用方先 `.rank()` 再传入 | 确定性自检直接复用它，无需新写相关系数函数 |
| `marvis/orchestrator/templates/sample.py:1062-1121`（`STRATEGY_ANALYSIS`） | `tradeoff_view` 步骤 `inputs_template` 只有 `dataset_id/score_col/target_col` 三个映射（L1109-1113），无方向相关字段 | 与计划一致；模板改动属于 S1a 迁移面第 3 类（见 §2.3） |

**结论**：计划 §二 S-3 与 §七 S1a 行的方向性判断全部成立；本 spec 在其基础上补三处代码级细化：(a) 规则一致性自检必须做**字符串解析**而非"读一个 operator 字段"；(b) `_ModelArtifactScorer` 存在 `raw_score` vs `scorecard_points` 两套已矛盾的方向，direction 元数据要覆盖两者；(c) `ModelArtifact` 的运行时真源是 SQLite 表，`.model_meta.json` 只是旁路副本，两处都要写但语义不同。

---

## 1. 平台常量与类型

### 1.1 放置位置

新增枚举放 `marvis/data/direction.py`（新文件，比照 `marvis/data/labels.py` 的粒度——两者都是"NaN 标签门"级别的横切原语，理应同级）。不放进任何单一 pack（strategy/modeling 都要用，跨包共享）。

```python
# marvis/data/direction.py
from __future__ import annotations

from typing import Literal

ScoreDirection = Literal["higher_is_riskier", "higher_is_better"]

SCORE_DIRECTIONS: tuple[ScoreDirection, ...] = ("higher_is_riskier", "higher_is_better")


def normalize_score_direction(value: str | None) -> ScoreDirection | None:
    """Validate + normalize a caller-supplied direction string.

    Returns None when value is falsy (caller must supply their own default —
    this module never guesses; see §2 per-consumer default semantics).
    Raises ValueError for any non-empty value outside SCORE_DIRECTIONS
    (typed as plain ValueError, not a DataLayerError subclass — this is a
    schema-shape problem, not a data-quality gate; manifest enum validation
    should already catch it before this runs, this is defense in depth).
    """
    if not value:
        return None
    normalized = str(value)
    if normalized not in SCORE_DIRECTIONS:
        raise ValueError(f"invalid score_direction: {normalized!r}; expected one of {SCORE_DIRECTIONS}")
    return normalized  # type: ignore[return-value]
```

### 1.2 序列化形态

- JSON Schema（manifest 里新参数 / `ModelArtifact` meta 字段）：字符串枚举 `["higher_is_riskier", "higher_is_better"]`，**不加第三态**（`unknown`/`null` 用"字段整体缺失"表达，而非枚举值——见 §2.5 的 `None` 语义）。
- Python 侧：`Literal["higher_is_riskier", "higher_is_better"] | None`，`None` = "未声明，落到该消费方自己的默认值"（各消费方默认值不同，§2 逐条给出）。
- 不新建 dataclass 包装这个枚举——它总是作为某个更大 dataclass/dict 的一个字段出现，单独包装徒增拆装箱代码。

---

## 2. 逐消费方变更清单

### 2.1 `tradeoff_view`（strategy pack，裸分数消费方）

**现签名**（`marvis/packs/strategy/tradeoff.py:11-20`）：
```python
def tradeoff_view(
    df: pd.DataFrame, *, score_col: str, target_col: str,
    cutoffs: list[float] | None = None, profit_params: ProfitParams | None = None,
    ead_col: str | None = None, pd_col: str | None = None,
) -> list[TradeoffPoint]:
```

**新签名**：
```python
def tradeoff_view(
    df: pd.DataFrame, *, score_col: str, target_col: str,
    cutoffs: list[float] | None = None, profit_params: ProfitParams | None = None,
    ead_col: str | None = None, pd_col: str | None = None,
    score_direction: ScoreDirection | None = None,
) -> list[TradeoffPoint]:
```

**默认值语义**：`score_direction=None` → 内部按**当前隐含行为**处理，即 `higher_is_better`（因为现状 L38 `scores >= cutoff` 即"approved"，approved 语义上等价于"分数越高越该批")。**不静默改变现有测试/现有行为**——`tests/test_strategy_tradeoff.py` 用 500-760 分制且高分=好客户，必须继续通过。

**内部伪代码**：
```
def tradeoff_view(df, *, score_col, target_col, cutoffs, profit_params, ead_col, pd_col, score_direction=None):
    effective_direction = score_direction or "higher_is_better"   # 向后兼容默认
    labeled = df 里 target_col 非 NaN 的行（调用方已经过 resolve_labeled_frame，这里不重复过滤）

    # 确定性方向自检（仅当有标签样本时）：
    if len(labeled) >= MIN_CORR_SAMPLE_SIZE:  # 见 §3 阈值
        corr = safe_correlation(labeled[score_col].to_numpy(float), labeled[target_col].to_numpy(float))
        implied_direction = "higher_is_riskier" if corr > 0 else "higher_is_better" if corr < 0 else None
        if implied_direction is not None and abs(corr) >= CORR_CONFLICT_THRESHOLD and implied_direction != effective_direction:
            raise ScoreDirectionConflictError(
                tool="tradeoff_view", score_col=score_col, target_col=target_col,
                declared_direction=effective_direction, implied_direction=implied_direction,
                corr=corr, n_labeled=len(labeled),
            )

    for cutoff in sorted(cutoffs or _default_cutoffs(df[score_col])):
        if effective_direction == "higher_is_better":
            approved_mask = df[score_col] >= cutoff      # 现状行为，未改变
        else:  # higher_is_riskier
            approved_mask = df[score_col] < cutoff        # 新分支：低分（低风险）才批
        ... 其余计算不变 ...
```

**错误路径**：`ScoreDirectionConflictError`（见 §3 typed error 定义），走强制确认门；用户可 (a) 改 `score_direction` 参数重跑，或 (b) 确认按声明方向继续（`drop_nan_labels` 式的显式覆盖参数，见下）。

**新增可选参数** `confirm_direction_conflict: bool = False`（比照 `drop_nan_labels` 的"确认后继续"范式）：为 `True` 时即使方向自检矛盾也不阻断，只把 `direction_conflict` 诊断信息塞进返回值的 `diagnostics` 字段（不再 raise）。

**返回值新增字段**：`TradeoffPoint` **不改**（保持 `cutoff/approval_rate/bad_rate/expected_profit` 四元组，向后兼容）；`tool_tradeoff_view` 顶层返回 dict 新增 `"score_direction": effective_direction` 与可选 `"direction_diagnostics": {...}`（仅当算过自检时出现）。

**manifest.json 变更**（`marvis/packs/strategy/manifest.json`，`tradeoff_view` 的 `input_schema.properties`）：
```json
"score_direction": {"type": "string", "enum": ["higher_is_riskier", "higher_is_better"]},
"confirm_direction_conflict": {"type": "boolean"}
```
（保持 `additionalProperties: false`，两个新键显式加进 `properties`，不加进 `required`。）

**`tool_tradeoff_view`**（`marvis/packs/strategy/tools.py:169-195`）改动：透传 `score_direction=_optional_str(inputs.get("score_direction"))`、`confirm_direction_conflict=bool(inputs.get("confirm_direction_conflict"))`；返回值合入 `"score_direction"`/`"direction_diagnostics"`。

**渲染层**（`marvis/agent/renderers.py:601-629` `_render_tradeoff_view`）：`text` 前置一句方向标注，例如：
```python
direction_label = "分数越高风险越低" if o.get("score_direction") == "higher_is_better" else "分数越高风险越高"
text = f"**策略权衡视图完成**（{direction_label}）: ..."
```

### 2.2 `reject_inference`（modeling pack，裸分数消费方，跨包）

**现签名**（`marvis/packs/modeling/reject_inference.py:29-40`）：
```python
def reject_inference(
    frame: pd.DataFrame, *, target_col: str, decision_col: str,
    method: str = "parceling", score_col: str | None = None,
    reject_bad_rate: float | None = None, reject_weight: float = 1.0,
    output_target_col: str = INFERRED_TARGET_COL, output_weight_col: str = SAMPLE_WEIGHT_COL,
) -> RejectInferenceResult:
```

**新签名**：
```python
def reject_inference(
    frame: pd.DataFrame, *, target_col: str, decision_col: str,
    method: str = "parceling", score_col: str | None = None,
    reject_bad_rate: float | None = None, reject_weight: float = 1.0,
    output_target_col: str = INFERRED_TARGET_COL, output_weight_col: str = SAMPLE_WEIGHT_COL,
    score_direction: ScoreDirection | None = None,
) -> RejectInferenceResult:
```

**默认值语义**：`score_direction=None` → 按当前隐含行为 `higher_is_riskier`（`_risk_order` 现状 L204 `argsort(-safe_scores)`，即"高分排前面=优先标 bad"，等价于"高分=高风险"）。`tests/test_modeling_reject_inference.py` 现有断言（高分件标 bad）必须继续通过。

**内部伪代码**（`_risk_order`，仅在 `method="parceling"` 且 `score_col` 提供时才有意义）：
```
def _risk_order(frame, score_col, score_direction=None):
    if score_col is None:
        return 原有的"无 score_col 退化路径"（现状不变，通常是按索引或均匀处理）
    effective_direction = score_direction or "higher_is_riskier"
    safe_scores = frame[score_col].fillna(frame[score_col].median())  # 现状缺失值处理不变
    if effective_direction == "higher_is_riskier":
        return np.argsort(-safe_scores, kind="mergesort")   # 现状行为，未改变
    else:  # higher_is_better -> 低分=高风险，优先标 bad 的应是低分
        return np.argsort(safe_scores, kind="mergesort")
```

**确定性自检**：与 tradeoff_view 相同模式，但**样本前提不同**——`reject_inference` 的调用场景通常是"拒绝件无标签、通过件有标签"，所以自检 corr 只能在**通过件子集**（`decision_col` 表示批准的行）上算 `corr(score_col, target_col)`，且要求这部分标签非空。若通过件里也没有可信标签样本（< 阈值），自检跳过（不是报错，是"数据不支持自检"，与 §3 的样本量下限规则一致）。

```
def _direction_self_check(frame, score_col, target_col, decision_col, declared_direction):
    approved_labeled = frame[(frame[decision_col] == "approve") & frame[target_col].notna()]
    if len(approved_labeled) < MIN_CORR_SAMPLE_SIZE:
        return None  # 自检不可用，不阻断，diagnostics 里记 "insufficient_labeled_approved_sample"
    corr = safe_correlation(approved_labeled[score_col].to_numpy(float), approved_labeled[target_col].to_numpy(float))
    implied = "higher_is_riskier" if corr > 0 else "higher_is_better" if corr < 0 else None
    if implied is not None and abs(corr) >= CORR_CONFLICT_THRESHOLD and implied != declared_direction:
        raise ScoreDirectionConflictError(tool="reject_inference", ...)
    return {"corr": corr, "n_labeled_approved": len(approved_labeled), "implied_direction": implied}
```

**错误路径**：同 `ScoreDirectionConflictError`，`confirm_direction_conflict` 覆盖参数同款。

**manifest.json 变更**（`marvis/packs/modeling/manifest.json`，`reject_inference` 的 `input_schema.properties`，当前 `additionalProperties: false`）：新增 `score_direction`、`confirm_direction_conflict`，与 2.1 相同枚举。

**`tool_reject_inference`**（`marvis/packs/modeling/tools.py:218` 起）：透传新参数，返回值合入 `score_direction`/`direction_diagnostics`。

**不修 `_fuzzy_augment_rejected`**：DOM-12(b) 记录的"全局 bad_rate 加权、不吃 score_col"是独立缺口，本 spec 明确不动它（避免范围蔓延；该函数目前根本不读 score_col，方向参数对它无意义）。

### 2.3 `backtest_strategy` / `build_strategy`（strategy pack，规则求值消费方——一致性自检，不加参数）

**不改签名**（S-3 决策：规则求值消费方方向已编码在 `condition` 字符串的比较运算符里，加冗余方向参数会造成双重方向逻辑源）。

**一致性自检位置**：`build_strategy`（`marvis/packs/strategy/strategy.py:23-62`）内部，在 `parsed_rules` 构造完成后、返回 `Strategy` 前追加。

**自检触发条件**：`score_col` 非空 **且** 至少一条规则的 `condition` 是形如 `<score_col> <op> <literal>` 的简单比较（`ast.Compare` 且 `node.left.id == score_col`）。复合条件（`AND`/`OR` 嵌套多个字段）里，只检查那些**顶层直接引用 `score_col`** 的比较子句，不递归展开去猜测复合逻辑的整体方向（避免过度工程——复杂布尔组合的"方向"本身是病态定义）。

**内部伪代码**：
```
def build_strategy(strategy_type, rules, *, score_col, default_decision, description=""):
    ... 现有解析逻辑不变 ...
    parsed_rules = [...]  # 现状

    direction_flags = []
    if score_col:
        for rule in parsed_rules:
            op_direction = _infer_condition_direction(rule.condition, score_col)
            # op_direction ∈ {"gte_style"(>=,>), "lte_style"(<=,<), None(不引用 score_col 或非简单比较)}
            if op_direction is not None:
                direction_flags.append((rule, op_direction))

    if direction_flags:
        distinct_directions = {flag for _, flag in direction_flags}
        if len(distinct_directions) > 1:
            # 同一策略内，一部分规则"高分批准"、另一部分"低分批准"——矛盾
            raise ScoreDirectionConflictError(
                tool="build_strategy", score_col=score_col,
                declared_direction=None,  # 规则求值型没有"声明方向"这个概念，字段留空
                implied_direction=None,
                conflicting_rules=[r.condition for r, _ in direction_flags],
                reason="rules reference score_col with inconsistent comparison direction",
            )
        # 单一方向：不阻断，只把推断方向记入 strategy 返回值供下游（如 compare_strategies）参考
        inferred_direction = "higher_is_better" if distinct_directions == {"gte_style"} else "higher_is_riskier"
    else:
        inferred_direction = None

    return Strategy(..., 现有字段不变 ...)  # Strategy dataclass 本身不加字段（见下）
```

```
def _infer_condition_direction(condition: str, score_col: str) -> str | None:
    # 复用 strategy.py 已有的 _parse_condition，只多做一步：定位顶层直接比较 score_col 的子句
    expr = _parse_condition(condition).body
    clauses = _flatten_top_level_compares(expr)  # 新增小工具函数：展开 BoolOp 的直接子节点里的 Compare
    for clause in clauses:
        if isinstance(clause, ast.Compare) and isinstance(clause.left, ast.Name) and clause.left.id == score_col:
            op = clause.ops[0]
            if isinstance(op, (ast.GtE, ast.Gt)):
                return "gte_style"
            if isinstance(op, (ast.LtE, ast.Lt)):
                return "lte_style"
            return None  # Eq/NotEq/In/NotIn 无方向含义，不计入判断
    return None
```

**为什么不把 `inferred_direction` 塞进 `Strategy`/`StrategyRule` dataclass**：这是一个自检副产品，不是持久化契约的一部分（`StrategyRepository` 表结构不变，见 §2.6 的"S1a 不做"边界）；下游若要用，从 `build_strategy` 工具返回的 dict 里读 `"inferred_score_direction"` 字段即可（新增在 `tool_build_strategy` 的返回 dict，不在 `Strategy` dataclass 上）。

**`backtest_strategy` 侧**：不新增自检（避免重复计算——一份策略只在 `build_strategy` 时检查一次，`backtest_strategy` 消费的是已构建好的 `Strategy`，信任其已经过 `build_strategy` 的门）。但如果策略是通过旧数据/直接写库绕过 `build_strategy` 产生的（历史数据、测试 fixture），`backtest_strategy` 不重复校验——这是显式的"只在创建时校验一次"设计决策，写入 §6 开放问题供用户确认是否要在 backtest 时也补一次防御性检查。

**manifest.json 变更**：`build_strategy`/`backtest_strategy` **不加任何新 schema 字段**（符合"不加参数"决策）。

**`tool_build_strategy`** 返回 dict 新增（`marvis/packs/strategy/tools.py:91-122`）：`"inferred_score_direction": inferred_direction`（可能为 `None`）。

**错误路径**：`ScoreDirectionConflictError` 从 `build_strategy` 抛出，`tool_build_strategy` 不吞掉（当前该工具没有 try/except 包裹 `build_strategy` 调用，异常自然沿子进程 worker 的 `to_detail()` 机制上抛为 `error_kind=score_direction_conflict`，走强制确认门）。**没有 `confirm_direction_conflict` 覆盖参数**——规则内部自相矛盾是配置错误而非"方向声明与数据矛盾"的统计判断，不应该有"确认后继续"的旁路（用户应该去改规则，而不是无视矛盾硬建策略）。此点与 2.1/2.2 的确认门语义不同，**§6 留作开放问题**请用户确认是否接受这个不对称设计。

### 2.4 `validation` 头尾 lift 方向翻转裁决（NEW-2，并入本 spec 统一裁决）

**问题**：`marvis/feature/metrics.py:109-145` 的 `head_tail_lift` 已经是方向自适应（按 `corr` 符号翻转）；`marvis/validation/effectiveness.py:413-430` 的 `compute_head_tail_lift` 硬编码降序、无方向感知，被 `effectiveness.py:97,261` 两处调用产出 `OverallRow.head_lift_5pct`/`tail_lift_5pct`。

**裁决**：**统一到 `feature/metrics.py:head_tail_lift` 的算法**（相关系数符号自适应），而不是引入 `score_direction` 参数——理由：validation 报告口径此前完全不知道模型的 `score_direction`（`effectiveness.py` 不吃 `ModelArtifact`，只吃裸 `scores`/`labels` 数组），若要求它接受方向参数就要求调用方（`marvis/validation/scorer.py`/`engine.py`）额外传参，改动面更大；而相关系数自适应对两套实现都已经在用（`feature/metrics.py` 珠玉在前），直接复用同一份判定逻辑，**验证包不需要知道也不需要传入 score_direction**，是更小的改动。

**具体改法——两个选项，本 spec 建议选项 A**：

- **选项 A（建议）**：`marvis/validation/effectiveness.py` 直接 `import` 并调用 `marvis/feature/metrics.py` 的 `head_tail_lift`，删除 `compute_head_tail_lift` 自己的排序逻辑，只做返回值形状适配（`feature/metrics.py` 返回 `dict[str, float|None]`（`lift_head_5/lift_tail_5/...`），`effectiveness.py` 现签名要 `tuple[float|None, float|None]`）：
  ```python
  def compute_head_tail_lift(scores, labels, fraction: float = 0.05) -> tuple[float | None, float | None]:
      result = head_tail_lift(  # from marvis.feature.metrics
          np.asarray(scores, dtype=float), np.asarray(labels, dtype=int),
          fractions=(fraction,), min_rows=20,
      )
      pct = int(round(fraction * 100))
      return result.get(f"lift_head_{pct}"), result.get(f"lift_tail_{pct}")
  ```
  **优点**：单一实现源，消灭"两套算法会漂移"的风险；签名不变，`effectiveness.py:97,261` 两处调用点零改动。
  **代价**：`feature/metrics.py` 与 `validation/` 目前是两个独立子系统（`feature/` 服务于特征工程，`validation/` 服务于模型验证报告），引入跨子系统 import——需确认这不违反现有分层约定（本 spec §6 开放问题请架构层确认）。

- **选项 B（备选，若选项 A 的跨层 import 不被接受）**：在 `validation/effectiveness.py` 内部复刻同款 `risk_sign` 逻辑（复制 `feature/metrics.py:137-138` 的两行核心算法，而不是复制整个函数），保持两个子系统解耦但算法口径一致：
  ```python
  def compute_head_tail_lift(scores, labels, fraction: float = 0.05) -> tuple[float | None, float | None]:
      scores = np.asarray(scores, dtype=float); labels = np.asarray(labels, dtype=int)
      finite_mask = np.isfinite(scores); scores, labels = scores[finite_mask], labels[finite_mask]
      if len(scores) == 0: return None, None
      bad_rate = float(labels.mean())
      bucket_size = int(len(scores) * fraction)
      if bucket_size <= 0 or bad_rate == 0.0: return None, None
      risk_sign = float(np.sign(safe_correlation(scores, labels.astype(float)))) or 1.0  # 新增，来自 feature/correlation.py
      order = np.argsort(risk_sign * scores, kind="mergesort")  # 改行：原为 np.argsort(scores)[::-1]
      sorted_labels = labels[order]
      head_lift = float(sorted_labels[-bucket_size:].mean() / bad_rate)  # 高风险端在数组尾部（ascending order）
      tail_lift = float(sorted_labels[:bucket_size].mean() / bad_rate)
      return head_lift, tail_lift
  ```

**回归测试要求（两个选项都需要）**：构造一个 `higher_is_better`（分数与 target 负相关）的合成样本，断言 `head_lift`（高风险端）落在**低分**那一侧，而非机械的"数组末尾"——现有测试若只用高分=高风险的样本会掩盖这个 bug，必须新增反向样本用例。

### 2.5 `score_dataset` / `monitor_run` 的契约预留（S1b 占位，本 spec 只定字段位）

**不实现**这两个工具（S1b 独立 spec）。本 spec 只确保 S1a 落地的 `ModelArtifact`/`.model_meta.json` 字段位置，S1b 直接读得到，不需要再改一轮 schema。

**预留字段**：`ModelArtifact.score_direction`（见 §2.6）在训练完成时**必须**被写入（不是可选留空）——因为 `score_dataset`（S1b）出分时要照抄这个字段到派生数据集的 direction 元数据，如果 S1a 阶段就漏填，S1b 就没有数据源。

**`score_dataset`/`monitor_run` 将来的输入 schema 占位**（仅记录设计意图，不在本 spec 创建 manifest 条目）：`score_dataset` 输出的派生数据集登记信息应含 `score_direction`（从 `artifact.score_direction` 直接复制，不重新推断）；`monitor_run` 比较 PSI 基准分布时不需要额外知道方向（PSI 是对称的分布差异度量），但告警文案要能读到 direction 以生成"分数上升/下降代表风险上升/下降"的人话解释——这依赖同一个 `artifact.score_direction` 字段，无需新字段。

### 2.6 `ModelArtifact` / `persist_model_meta` / `.model_meta.json` 的 direction 元数据

**`ModelArtifact` dataclass**（`marvis/packs/modeling/contracts.py:75-86`）新增尾部可选字段（遵循现有 `feature_importance`/`scorecard_table` 的"尾部默认值"惯例，frozen dataclass 加字段是安全的）：
```python
@dataclass(frozen=True)
class ModelArtifact:
    id: str
    experiment_id: str
    algorithm: str
    model_path: str
    pmml_path: str | None
    feature_list: tuple[str, ...]
    params: dict[str, Any]
    woe_maps: dict[str, Any] | None
    created_at: str
    feature_importance: tuple[tuple[str, float], ...] = ()
    scorecard_table: tuple[dict[str, Any], ...] = ()
    score_direction: str | None = None       # 新增；raw_score() 产出的方向
    points_direction: str | None = None      # 新增；scorecard_points() 产出的方向（仅 scorecard 非 None）
```

**为什么拆两个字段而不是一个**：§0 地面真值已核实 `_ModelArtifactScorer.raw_score()`（永远 `predict_proba[:,1]`，`higher_is_riskier`）与 `scorecard_points()`（`offset - factor*logits`，`higher_is_better`）是**同一个 artifact 上两种并存的、方向相反的打分产物**。只写一个字段会让调用方不知道该字段描述的是哪一路输出。约定：
- `score_direction`：描述 `raw_score()`/`score()` 方法的输出（PD 概率或树模型原始分），**所有算法固定为 `"higher_is_riskier"`**（因为 `raw_score` 的实现本身就是全平台统一取 `predict_proba[:,1]`，没有可配置的余地——这不是"检测"出来的，是实现方式决定的常量）。
- `points_direction`：只有 `algorithm == "scorecard"` 时非空，固定为 `"higher_is_better"`（评分卡分数惯例）；非 scorecard 算法此字段恒为 `None`。

**训练配方写入时机**：`create_artifact`（`marvis/packs/modeling/artifact.py:50-93`，即审查报告称的"artifact 构造函数"）在构造 `ModelArtifact(...)` 时按 `algorithm` 参数直接赋常量，**不做任何统计推断**（方向由实现决定，不是数据决定，符合 INV-1"确定性指标只由平台工具算"——这里甚至比"算"更强，是"编译期常量"）：
```python
score_direction = "higher_is_riskier"  # raw_score() 全算法统一如此，见 §0 地面真值
points_direction = "higher_is_better" if algorithm == "scorecard" else None
artifact = ModelArtifact(..., score_direction=score_direction, points_direction=points_direction)
```

**`persist_model_meta`**（`marvis/packs/modeling/artifact.py:96-145`）meta dict 新增两键：
```python
meta = {
    ... 现有 14 个键不变 ...
    "score_direction": artifact.score_direction,
    "points_direction": artifact.points_direction,
}
```

**DB schema 迁移**（`marvis/db_schema.py`）：`model_artifacts` 表已在 `_MIGRATION_TABLES` 允许名单里（§0 已核实），新增两列：
```python
_ensure_column(conn, "model_artifacts", "score_direction", "TEXT")
_ensure_column(conn, "model_artifacts", "points_direction", "TEXT")
```
放在 `init_db`（`marvis/db_schema.py:611-612` 紧邻现有两条 `_ensure_column` 调用之后）。

**读取方兼容——老 meta 无字段 → None → 按工具默认**：

1. **SQLite 读取路径**（真正的运行时源）：`_model_artifact_from_row`（`marvis/repositories/modeling.py:355-373`）新增：
   ```python
   score_direction=_optional_str(row["score_direction"]) if "score_direction" in row.keys() else None,
   points_direction=_optional_str(row["points_direction"]) if "points_direction" in row.keys() else None,
   ```
   老行迁移后该列存在但值为 `NULL`（`_ensure_column` 加列不回填历史行），`_optional_str(None) -> None`——自然落到 `None`，不需要特殊分支。**下游任何读 `artifact.score_direction` 的代码都必须把 `None` 当作"未知，回退调用方自己的默认值"处理**，不能假设它总非空（老 experiment 训练出的 artifact 永远是 `None`）。

2. **`.model_meta.json` 读取路径**（旁路信道，`marvis/recovery.py:32` 只判断存在性，不解析字段——**无需改动**；`marvis/pipeline.py:638` 的 V1 兼容 handoff 写入路径 `_write_model_meta_from_contract` 是独立写入器，**本 spec 不强制它也写这两个键**，因为它服务的是遗留 notebook 交接流程而非 V2 打分路径——若未来 handoff notebook 要读方向字段，用 `.get("score_direction")` 式访问，缺失时得 `None`，同样安全）。

3. **`_ModelArtifactScorer` 消费方**：不需要改动内部逻辑（`raw_score`/`scorecard_points` 的实现不依赖 `artifact.score_direction` 字段——方向是实现决定的，字段只是把这个事实暴露给外部消费方读取）。`score()`/`raw_score()` 方法签名不变。

**其余读取方枚举**（§0 已核实的调用点，逐一确认兼容性）：
- `marvis/packs/modeling/tools.py:3892`（`_snapshot_latest_model_meta`）：按字节快照/回滚，不解析字段——不受影响。
- `marvis/packs/modeling/tools.py:3903,3910`（`_cleanup_unattached_artifact`）：只按文件名删除——不受影响。

---

## 3. 确定性方向自检算法

### 3.1 相关系数口径

**Pearson，不是 Spearman**——理由：直接复用已有的 `marvis/feature/correlation.py:safe_correlation`（Pearson 实现），避免为这一处自检新写/新测一个 Spearman 原语；打分场景下 score 与 target(0/1) 的关系近似单调，Pearson 与 point-biserial 相关系数在数学上等价（二元 target 时 Pearson 相关就是 point-biserial correlation），**不存在信息损失**——这与 `head_tail_lift` 的既有实现（`marvis/feature/metrics.py:137` 同样用 `safe_correlation`，非 Spearman）保持算法口径一致。

若某个消费方将来需要对非二元/非近似单调关系做自检（当前 S1a 范围内没有这种消费方），届时可传入预排序（`.rank()`）值来达到 Spearman 效果——`safe_correlation` 本身对输入不挑，`correlation_matrix`（`correlation.py:9-31`）已经示范了这个"rank 后调用同一函数"的模式，不需要新函数。

### 3.2 样本量下限

`MIN_CORR_SAMPLE_SIZE = 30`（新增常量，放 `marvis/data/direction.py`）。理由：与 `readiness.py` 等现有质量门槛在同一数量级（"统计量在 <30 样本上不稳定"是通用统计常识阈值，非本平台已有硬编码值，此处需要显式标注为**建议值待用户拍板**，见 §6）。低于该阈值时自检**跳过**（不是通过、不是失败），diagnostics 里显式记 `"skipped: insufficient_labeled_sample (n=<n>, min=30)"`，避免小样本假阳性矛盾误伤门。

### 3.3 |corr| 阈值

`CORR_CONFLICT_THRESHOLD = 0.05`（新增常量，同上放置）。理由：DOM-2 原文"How to fix" 第 2 条即建议此值；0.05 是"方向判断有起码统计意义"的宽松下限（不是"强相关"阈值，只是排除"corr 几乎为零、方向判断本身没有意义"的噪声区间）。**同样标注为建议值待用户拍板**（§6）。

### 3.4 完整判定伪代码（跨 2.1/2.2 复用的共享函数）

```python
# marvis/data/direction.py（延续 §1.1 的同一文件）

def check_score_direction(
    scores: np.ndarray, target: np.ndarray, *,
    declared_direction: ScoreDirection,
    min_sample_size: int = MIN_CORR_SAMPLE_SIZE,
    corr_threshold: float = CORR_CONFLICT_THRESHOLD,
) -> DirectionCheckResult:
    """确定性方向自检，供 tradeoff_view/reject_inference 复用。

    不修改 scores/target；纯函数，可重复调用同一结果（INV-1 确定性指标要求）。
    """
    finite_mask = np.isfinite(scores) & np.isfinite(target)
    scores_f, target_f = scores[finite_mask], target[finite_mask]
    n = len(scores_f)
    if n < min_sample_size:
        return DirectionCheckResult(status="skipped", reason="insufficient_labeled_sample", n=n, corr=None)
    corr = safe_correlation(scores_f, target_f.astype(float))
    if abs(corr) < corr_threshold:
        return DirectionCheckResult(status="inconclusive", reason="corr_below_threshold", n=n, corr=corr)
    implied = "higher_is_riskier" if corr > 0 else "higher_is_better"
    if implied != declared_direction:
        return DirectionCheckResult(status="conflict", n=n, corr=corr, implied_direction=implied)
    return DirectionCheckResult(status="consistent", n=n, corr=corr, implied_direction=implied)
```

`DirectionCheckResult` 是一个小 frozen dataclass（`status: Literal["skipped","inconclusive","conflict","consistent"]`, `n: int`, `corr: float|None`, `reason: str|None = None`, `implied_direction: ScoreDirection|None = None`）——调用方（`tradeoff_view`/`reject_inference`）拿到 `status="conflict"` 时才抛 `ScoreDirectionConflictError`；其余三种状态都不阻断，只把 `DirectionCheckResult` 塞进各自的 `direction_diagnostics` 输出字段。

### 3.5 `ScoreDirectionConflictError` typed error 定义

放 `marvis/data/errors.py`（与 `NanLabelNotConfirmedError` 同级，因为这也是跨包共享的数据层确认门——不放进某个 pack 的 `errors.py`，避免 strategy/modeling 两边各定义一份重复类）：

```python
class ScoreDirectionConflictError(DataLayerError):
    """Raised when a declared/default score_direction contradicts the empirical
    corr(score, target) sign beyond the configured threshold (S1a determinism gate).

    Mirrors NanLabelNotConfirmedError's pattern: typed error + to_detail() payload,
    default is to stop and hand structured diagnostics to the user, who confirms
    (confirm_direction_conflict=True) to proceed with the declared direction anyway,
    or fixes score_direction and retries.
    """

    def __init__(
        self, *, tool: str, score_col: str, target_col: str | None,
        declared_direction: str | None, implied_direction: str | None,
        corr: float | None, n_labeled: int,
        conflicting_rules: list[str] | None = None,
        reason: str | None = None,
    ) -> None:
        self.tool = str(tool)
        self.score_col = str(score_col)
        self.target_col = str(target_col) if target_col else None
        self.declared_direction = declared_direction
        self.implied_direction = implied_direction
        self.corr = float(corr) if corr is not None else None
        self.n_labeled = int(n_labeled)
        self.conflicting_rules = list(conflicting_rules or [])
        self.reason = reason
        super().__init__(
            f"{tool}: score_direction conflict on {score_col!r}"
            + (f" (declared={declared_direction}, data implies={implied_direction}, corr={corr:.3f}, n={n_labeled})"
               if declared_direction else f" ({reason or 'rules disagree on direction'})")
            + "; pass confirm_direction_conflict=true to proceed anyway, or fix score_direction/rules and retry"
        )

    def to_detail(self) -> dict:
        return {
            "kind": "score_direction_conflict",
            "tool": self.tool,
            "score_col": self.score_col,
            "target_col": self.target_col,
            "declared_direction": self.declared_direction,
            "implied_direction": self.implied_direction,
            "corr": self.corr,
            "n_labeled": self.n_labeled,
            "conflicting_rules": self.conflicting_rules,
            "reason": self.reason,
        }
```

**门 payload 形态**：无需改动 `marvis/plugins/subprocess_worker.py`（§0 已核实，`_structured_error_detail` 是通用机制，任何带 `to_detail()` 的异常自动生效）；`error_kind` 会自动变成 `"score_direction_conflict"`，`error_detail` 会自动携带上述完整 dict。前端/门渲染层（`marvis/agent/gates/contracts.py` 的 `FailureEnvelope`）不需要新增分支——`error_kind`/`editable_input_schema` 机制已是通用的（`build_failure_envelope` 从 `step_inputs` 自动生成可编辑 schema，`score_direction`/`confirm_direction_conflict` 一旦加入 manifest 的 `properties`，就会自动出现在门的可编辑字段里，**无需为这个新 error_kind 专门写一个 `infer_gate_envelope` 分支**——现有 `elif kind == "gate": allowed = [...]` 兜底分支已覆盖）。

---

## 4. 兄弟路径回归矩阵

> 吸取审查报告反复强调的"三个未修净"教训（DOM-1 的 NaN 门只接了训练路径漏了调参路径）——本节要求逐条勾选，不是抽样验证。

### 4.1 维度枚举

- **train/tune/select**：本 spec 改动的两个函数（`tradeoff_view`/`reject_inference`）都不在 train/tune/select 路径上（它们是 strategy 阶段与 reject-inference 专用路径），**该维度对本 spec 不适用**——但 `ModelArtifact.score_direction` 字段的写入点在 `create_artifact`（`artifact.py:50-93`），这个函数被 train 路径调用；tune/select 路径**不**直接构造 `ModelArtifact`（tune 只产出 trial 参数，select 只挑选已有 experiment），所以只需验证 **train** 侧写入正确，tune/select 侧"透传不动"即可，不需要新增改动。

| 路径 | 验证点 | 状态 |
|---|---|---|
| 训练（`tool_train_model`/`tool_train_models`，所有 recipe：lgb/xgb/lr/scorecard/catboost/mlp/lgb_regressor） | 每个 recipe 训出的 `ModelArtifact.score_direction == "higher_is_riskier"`；`algorithm=="scorecard"` 时额外 `points_direction == "higher_is_better"`，其余算法 `points_direction is None` | 需新增回归测试，逐 recipe 参数化 |
| 调参（`tune_hyperparameters`） | 不产出 `ModelArtifact`，不涉及；确认 tune.py 不需要改动（读代码确认，不新增测试） | 只需静态确认，已在 §0 核实 |
| 选模（`compare_experiments`/`_pick_best_experiment`） | 不涉及方向字段，选择逻辑（KS/AUC）不变；确认新字段不影响选择结果（现有测试不应因加字段而失败） | 跑现有 `_pick_best_experiment` 测试确认无回归 |

### 4.2 propose/execute × manual/agent（四象限，对 `tradeoff_view`/`reject_inference`/`build_strategy` 三个改动工具分别过一遍）

| 象限 | `tradeoff_view` | `reject_inference` | `build_strategy` |
|---|---|---|---|
| **agent 模式 propose**（LLM 生成计划，含 `score_direction` slot 填充） | LLM 若不填 `score_direction`，走默认值路径（`higher_is_better`），确认自检不因"未声明"而误报——需回归测试：LLM 风格调用（不传该参数）在方向与数据一致时静默通过 | 同左，默认 `higher_is_riskier` | LLM 生成的规则若方向自相矛盾，`build_strategy` 门必须能在 agent 模式下正确渲染 `error_kind=score_direction_conflict`（复用现有门文本渲染管线，不新增 agent 专属分支） |
| **agent 模式 execute**（确认门后重跑同一步，带 `confirm_direction_conflict=true`） | 需回归测试：第一次因矛盾被拦，agent 补 `confirm_direction_conflict=true` 重跑后放行且 `direction_diagnostics` 里保留矛盾记录（不是静默清空） | 同左 | 不适用（`build_strategy` 无确认覆盖参数，§2.3 已注明该不对称设计） |
| **manual 模式 propose**（用户在控件里手填 `score_direction` 下拉框） | 控件默认值需与"未声明"的运行时默认值一致（`higher_is_better`），避免"控件默认选中的选项"与"代码里 `None` 落到的默认值"产生第二套隐性默认值分裂——**这是最容易埋雷的一格**，必须显式检查前端控件的默认选中项 | 控件默认 `higher_is_riskier` | 规则构建器 UI 若允许自由输入 condition 字符串，矛盾检测在提交时（而非仅在最终确认时）就能触发，避免用户填了 5 条规则才在最后一步收到矛盾报错 |
| **manual 模式 execute**（用户手动确认覆盖） | 需回归测试：手动模式下点击"仍要继续"对应到 `confirm_direction_conflict=true` 的正确传参（不是漏传导致门死循环） | 同左 | 同上，无覆盖路径 |

### 4.3 回归测试文件命名（沿用现有 `test_<pack或模块>_<行为>` 惯例，无测试类，模块级 import + 多个 `test_<动词短语>` 函数）

- `tests/test_strategy_tradeoff.py`：追加 `test_tradeoff_view_score_direction_default_matches_legacy_behavior`、`test_tradeoff_view_score_direction_higher_is_riskier_flips_approval`、`test_tradeoff_view_raises_on_direction_conflict`、`test_tradeoff_view_confirm_direction_conflict_bypasses_gate`。
- `tests/test_modeling_reject_inference.py`：追加 `test_reject_inference_score_direction_default_matches_legacy_behavior`、`test_reject_inference_score_direction_higher_is_better_flips_risk_order`、`test_reject_inference_raises_on_direction_conflict`。
- `tests/test_strategy_pack.py`（`build_strategy` 已有测试大概率在此文件——若不在，新建 `tests/test_strategy_build_direction_check.py`）：追加 `test_build_strategy_raises_on_inconsistent_rule_directions`、`test_build_strategy_single_direction_rules_pass_and_report_inferred_direction`。
- `tests/validation/test_effectiveness.py`：追加 `test_head_tail_lift_flips_for_higher_is_better_score`（§2.4 的核心回归点：构造负相关样本，断言高风险端落在低分侧）。
- 新文件 `tests/test_modeling_artifact_score_direction.py`：`test_create_artifact_sets_score_direction_per_algorithm`（参数化跑 lgb/xgb/lr/scorecard/catboost/mlp/lgb_regressor，断言 `score_direction`/`points_direction` 组合）、`test_model_artifact_from_row_tolerates_missing_direction_columns`（模拟老行缺列场景）。

---

## 5. 迁移与验收

### 5.1 一次 PR 的提交切分建议

按依赖顺序拆 4 个提交（每个提交独立可跑测试，避免"半成品提交导致 CI 红一段时间"）：

1. **提交 1：平台原语**——新增 `marvis/data/direction.py`（枚举 + `check_score_direction`）、`marvis/data/errors.py` 新增 `ScoreDirectionConflictError`。无消费方改动，纯新增文件，零回归风险。附单元测试（`tests/test_data_direction.py`，覆盖 `check_score_direction` 四种 status 分支 + `normalize_score_direction` 校验）。
2. **提交 2：`ModelArtifact` 元数据**——§2.6 全部改动（dataclass 字段、`create_artifact`、`persist_model_meta`、DB 迁移、`_model_artifact_from_row`）。附 `tests/test_modeling_artifact_score_direction.py`。此提交不依赖提交 3/4，可独立验收。
3. **提交 3：`tradeoff_view` + `reject_inference`**（§2.1、§2.2）——两个裸分数消费方一起做（复用同一个 `check_score_direction`，独立测试但共享被测原语，放一个提交里减少"半个方向自检"的中间态）。附对应回归测试。
4. **提交 4：`build_strategy` 一致性自检 + NEW-2 lift 方向裁决**（§2.3、§2.4）——两者互不依赖但都是"收尾类"改动，一起提交降低 PR 数量。附对应回归测试 + §4 的兄弟路径回归矩阵全跑一遍。

### 5.2 全量回归口径

- 每个提交后：`pytest tests/ -x`（沿用现有惯例，Python 环境 `/opt/miniconda3/envs/py_313/bin/python`，参照既往阶段记录）全绿，且新增测试数量与 §4.3 清单一一对应（不能少测）。
- 提交 4 完成后：额外跑一次 §4.2 四象限矩阵里标"需回归测试"的全部用例（manual/agent 控件默认值一致性那一格尤其不能跳过，历史上这类"两套默认值分裂"的 bug 最难在纯后端测试里发现，需要前端/门渲染层联动检查，若前端控件当前还没有 `score_direction` 下拉框，此格降级为"记录为已知空白，控件本身留给 S2 做"）。
- PR 描述需包含：受影响 manifest 文件 diff（`marvis/packs/strategy/manifest.json`、`marvis/packs/modeling/manifest.json`）、DB 迁移新增列清单、`ModelArtifact` 字段变更前后对比。

### 5.3 门文案变化

- `tradeoff_view`/`reject_inference` 的确认门新增红旗类型 `score_direction_conflict`，门文案模板（中文，供渲染层/agent 话术复用）：
  > "分数方向自检矛盾：声明方向为『{declared_direction_label}』，但样本数据显示分数与坏账的相关方向是『{implied_direction_label}』（相关系数 {corr:.3f}，基于 {n_labeled} 条有标签样本）。请确认分数方向是否声明有误，或数据本身存在异常。"
- `build_strategy` 规则矛盾门文案：
  > "策略规则内部方向矛盾：以下规则对同一分数列 `{score_col}` 使用了相反方向的比较运算符，无法确定该分数是"越高越该批准"还是"越高越该拒绝"：{conflicting_rules}。请检查规则设置。"
- 两类门文案都遵循 S-8 决策（"门注入确定性红旗，LLM 复核而非发明"）——文案由平台渲染层拼装好完整数值，LLM/agent 只做话术包装，不重新计算或转述数字。

---

## 6. 开放问题区（留给用户审）

1. **`MIN_CORR_SAMPLE_SIZE=30` 与 `CORR_CONFLICT_THRESHOLD=0.05` 是否合适？** §3.2/3.3 标注为"建议值"，非平台已有硬编码惯例值——需要用户依据实际信贷数据规模（尤其小样本策略场景，如某些细分渠道回测样本可能就是几十条量级）拍板，过严会导致小样本场景自检永远 `skipped`（形同虚设），过松会导致噪声样本误报。
2. **`build_strategy` 的规则矛盾门不设 `confirm_direction_conflict` 覆盖参数，是否接受这个不对称设计？**（§2.3 末尾）——裸分数型两个工具允许"确认后继续"，规则型不允许、必须改规则重来。这是本 spec 的判断（规则矛盾是逻辑错误而非统计判断），但如果用户认为存在合理场景（例如故意设计"高分段特批 + 低分段特批"的双向规则），需要放开覆盖参数，请明确指出。
3. **NEW-2 选项 A vs 选项 B（§2.4）**：选项 A 让 `marvis/validation/` 反向 import `marvis/feature/`，需要确认这不违反现有分层（`marvis/feature/` 是否被设计为只能被 `packs/` 消费、不能被 `validation/` 反向依赖？本 spec 未在代码里找到显式的分层强制机制，只是观察到当前两个子系统互不 import，可能只是历史演进结果而非架构约束）。若分层有强约束，选一律选项 B。
4. **`backtest_strategy` 是否需要对"绕过 `build_strategy` 产生的历史 Strategy"补一次防御性方向自检？**（§2.3 末尾）——本 spec 当前设计是"只在创建时查一次"，如果历史数据/测试 fixture 里存在未经校验的策略，`backtest_strategy` 会静默放行。需要用户确认这个边界是否可接受，或要求补一道运行时防线。
5. **`points_direction` 字段是否需要在 `tradeoff_view`/其他裸分数消费方的 `score_direction` 参数校验时做交叉提示？**——例如用户对一个 scorecard 模型的 `points` 列跑 `tradeoff_view` 却传了 `score_direction="higher_is_riskier"`（与 `points_direction="higher_is_better"` 矛盾），当前 spec 的确定性自检会通过 corr 符号自然抓到这个矛盾（不需要额外硬编码"scorecard points 必须 higher_is_better"的业务规则），但如果用户希望有更早、更直接的"字段级"交叉校验（不依赖跑完自检），需要额外设计，本 spec 未纳入。
