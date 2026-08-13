# 测试真实性审计（计划条目 B-1）

- 日期：2026-08-13
- 范围：确定性指标内核（KS / AUC / PSI / 分箱 / 稳定性 / 压力 / 坏率 / 通过率 / 影响 / waterfall / 分数一致性）
- 结论性质：只读审计。未改动任何代码；唯一产物是本文件。

## 1. 方法与口径

### 1.1 关注面

按任务要求，聚焦以下确定性指标内核及其测试：

| 域 | 内核实现 | 主要测试 |
|---|---|---|
| KS / AUC / PSI 平台参考实现 | `marvis/feature/metrics.py`（`feature_ks` / `feature_auc` / `compute_psi` / 加权变体） | `tests/test_feature_metrics.py` |
| 验证层 KS/AUC/PSI/分箱 | `marvis/validation/effectiveness.py`、`marvis/validation/binning.py` | `tests/validation/test_effectiveness.py`、`tests/validation/test_binning.py` |
| 建模指标层 | `marvis/…/modeling`（`compute_model_metrics`） | `tests/test_modeling_recipes.py` |
| 坏率 / 通过率 / 影响 | `marvis/packs/strategy/backtest.py`、`tradeoff.py`、`typed_backtest.py` | `tests/test_strategy_tradeoff.py`、`tests/test_strategy_typed_backtest.py` |
| 稳定性 / waterfall / PSI 归一化 | `marvis/packs/strategy/pool_stability.py`、`candidate_stability.py` | `tests/test_strategy_pool_stability.py` |
| 分数一致性 | `marvis/validation/reproducibility.py` | `tests/validation/test_reproducibility.py` |
| 样本统计（坏率/分布） | `marvis/validation/sample_stats.py` | `tests/validation/test_sample_stats.py` |
| 压力测试 / 风险分档 | `marvis/validation/stress_test.py`、`stress_risk.py` | `tests/validation/test_stress_test.py` |
| 数据 join / dedup / profiling | `marvis/data/join_engine.py`、`dedup.py`、`descriptive.py`、`profiler.py` | `tests/test_join_engine.py`、`tests/test_data_dedup.py`、`tests/test_data_descriptive.py`、`tests/test_sampler_profiler.py` |
| 指标表渲染 | `marvis/metric_tables.py` | `tests/test_metric_tables.py` |

### 1.2 分级定义（任务给定）

- **A**：断言手算/外部黄金常量——测试里硬编码的、不是用实现公式推导出来的数字。
- **B**：只测内部一致性——用另一模块/算法复算，或只测结构性质（范围、方向、非负、确定性）。
- **C**：期望值通过导入/复制被测函数公式计算（自证 / 镜像实现）。

### 1.3 方法说明

1. 对每个关键指标函数，定位其唯一/主要测试，逐条判断期望值的来源：硬编码常量（A）、独立复算或结构断言（B）、调用同一实现或手写同一公式（C）。
2. 对 C 类测试记录 `文件:行号`。
3. 核对五个关键指标（KS、AUC、PSI、坏率、分数一致性）是否至少有一个 A 类黄金锚点。
4. 变异抽查是**阅读实现后的推演**（不实际注入变异、不改代码），判断「类别顺序搞反 / 符号取反」这类典型错误是否会让现有测试变红。

> 口径说明：本审计按「测试函数」而非「12,000+ 条测试用例」计数；只对确定性内核的测试函数逐一分级，未对全仓库做穷举。表内数字是本次审计**实际分级过**的测试函数数，不是全量统计。

## 2. 分级统计表（按指标域，A/B/C 测试函数计数）

| 指标域 | A（黄金常量） | B（一致性/结构） | C（自证/镜像） | 是否有非退化黄金锚 |
|---|---|---|---|---|
| KS（`feature_ks` 及封装） | 3（含 1 个非退化加权黄金 `4/7`） | 3 | 1 | 部分：非加权 KS 只有 0/1 极端锚 + 1 处加权 4/7 |
| AUC（`feature_auc` 及封装） | 5 | 1（sklearn 对拍） | 1 | 是（0.0/1.0/0.5/12⁄14 + sklearn） |
| PSI（`compute_psi`/`feature_psi`） | 1（仅退化 0.0） | 3 | 4 | **否**（无非退化黄金常量） |
| 坏率 / 通过率 | 6 | 2 | 0 | 是 |
| 分箱 / `bin_table` / 累积核 | 5（含 `bin_table` 黄金锁定） | 3 | 1（被黄金测试钉住） | 是 |
| 稳定性 / waterfall / PSI 归一化 | 3 | 4 | 0（实现内校验） | 是（waterfall counts + `max_abs_share_delta`） |
| 压力测试 / 风险分档 | 0（结构方向） | 6 | 0 | 否（压力 KS/PSI 分档阈值部分由 batch runner 覆盖，见 §4） |
| 分数一致性（`scores_match_at_precision` + 状态阈值） | 4 | 5 | 0 | 是 |
| 样本统计（坏率/分布/周期） | 3 | 2 | 0 | 是 |
| 数据 join / dedup / profiling | 6 | 4 | 0 | 是 |
| 指标表渲染（`metric_tables.py`） | 0（纯渲染，不重算指标） | 8 | 0 | 不适用（渲染层不计算指标） |

> 说明：表中计数以本次审计逐条阅读的测试函数为准。多个断言出现在同一测试函数时，按该函数中最强的等级计入一次。

## 3. C 类测试清单（文件:行号）

这些测试的期望值通过**导入或复制被测函数公式**得到。实现与测试若被同一错误公式驱动，会「一起改错仍全绿」。

1. `tests/test_feature_metrics.py:117` — `test_compute_psi_and_feature_psi_use_shared_edges_and_smoothing`：期望 PSI 直接按公式手写 `(0.75-0.5)*log(0.75/0.5) + (0.25-0.5)*log(0.25/0.5)`，是 `compute_psi` 公式的逐字复制。
2. `tests/test_feature_metrics.py:130` — `test_compute_psi_renormalizes_after_zero_bucket_smoothing`：期望按同一「零桶平滑 + 重归一化」公式内联计算。
3. `tests/validation/test_effectiveness.py:303-308` — `test_psi_stability_table_uses_shared_compute_psi_smoothing`：`sum(row.psi) == compute_psi([expected_pct...], [actual_pct...])`，用被测 `compute_psi` 复算期望。
4. `tests/validation/test_effectiveness.py:170-172` — `test_compute_auc_matches_canonical_feature_auc`：`compute_auc(...) == feature_auc(...)`。`compute_auc` 本就是 `feature_auc` 的薄封装（直接 `return _feature_auc(...)`），属于同义反复，只能证明「封装不漂移」，不能证明 AUC 数值正确。
5. `tests/validation/test_effectiveness.py:191-193` — `test_roc_ks_curve_scalar_matches_canonical_feature_ks`：`curve.ks == feature_ks(...)`。这是 ROC 曲线 KS 与累积分布 KS 两个平台实现的互证（比 4 强，因为算法路径不同），但仍是「平台代码对平台代码」，无外部/手算锚。
6. `tests/test_validation_debt.py:28-29` — `compute_ks == feature_ks`、`compute_psi == feature_compute_psi`：两个 `validation.binning` 封装与 `feature.metrics` 原函数的同义反复（封装直接委托）。
7. `tests/test_feature_iv.py:10-33` — `test_smoothed_woe_iv_kernel_matches_inline_formula`：`_inline_woe_iv` 是被测公式的逐行镜像（docstring 明言「kept here as the ground truth the shared kernel must reproduce」）。
8. `tests/test_feature_encode.py:85-88` — `test_categorical_woe_encode_matches_hand_computed_smoothed_woe`：期望 WOE 按公式内联写出（`np.log(((5+0.5)/(total_good+0.5*n_groups)) / ...)`）。
9. `tests/validation/test_binning.py:97-149` — `test_accumulate_bin_metrics_matches_legacy_loop_both_directions`：`_legacy_accumulate` 是累积核逻辑的逐行复制，作为期望。**注**：该测试本身是镜像，但被 `test_bin_table_golden_values_preserved`（第 152-174 行，硬编码 lift/ks/cum 黄金数字）独立钉住，风险被缓解。
10. `tests/test_modeling_recipes.py:157-158` — `test_compute_model_metrics_uses_platform_feature_metrics_and_overfitting`：`metrics.train_ks == feature_ks(...)`、`metrics.train_auc == feature_auc(...)`。建模层的指标与平台参考函数互证，非外部锚。

> 对照：`tests/test_feature_metrics.py:31-35`（`_naive_ks` 阈值扫描）**不列入 C**——它是独立算法（逐阈值扫描）而非公式复制，属 B 类中较有价值的一档。

## 4. 关键指标黄金锚点核对

| 指标 | 是否有 A 类黄金锚 | 锚点位置与性质 |
|---|---|---|
| KS | **是（但偏极端）** | `test_binning.py:56`（`compute_ks` = 1.0 完美分离）；`test_effectiveness.py:211`（ROC KS = 1.0）；`test_modeling_recipes.py:228`（加权 KS = `4/7`，唯一非退化手算锚）；退化 0.0（`test_binning.py:62`、`test_feature_metrics.py:36`） |
| AUC | **是** | `test_feature_metrics.py:103-104`（方向敏感 0.0 / 方向无关 1.0）；`test_effectiveness.py:116/142/151`（1.0 / 0.0 / 0.5）；`test_modeling_recipes.py:227/229`（加权 `12/14`）；外加 sklearn 对拍（`test_feature_metrics.py:95`） |
| PSI | **否（无非退化黄金）** | 唯一黄金是「同分布 → 0.0」退化锚（`test_binning.py:68`、`test_effectiveness.py:96`、`test_modeling_recipes.py:230`）；非退化值只由 C 类公式镜像和 `>0` 结构断言覆盖 |
| 坏率 | **是** | `test_sample_stats.py:42/52`（0.5）；`test_strategy_tradeoff.py:42-46`（1/3、0.5、1.0）；`test_strategy_typed_backtest.py:135-140`（0.5） |
| 分数一致性 | **是** | `test_reproducibility.py:347-349`（`scores_match_at_precision` 手算四舍五入对拍）；状态阈值（98%/95%/max_abs_diff 1e-4）逐档手算（`test_reproducibility.py:86-190`） |

### 4.1 无黄金锚点指标清单（需补锚）

1. **PSI（最重要缺口）**——五个关键指标中唯一缺乏非退化 A 类黄金常量的。当前数值只被 C 类镜像 + `>=0`/`>0` 结构断言覆盖，系统性公式错误（错误约定/归一化/平滑）会「实现+镜像测试一起绿」。
2. **非加权 KS 的非退化值**——只有 0.0/1.0 极端锚 + 一处加权 `4/7`；`feature_ks` 的 change-points/cumsum 中间逻辑未被一个硬编码的「中等 KS 数值」钉住（`_naive_ks` 阈值扫描是独立复算，但无硬编码数字）。
3. **压力测试 KS 衰减 / PSI 分档阈值**（`stress_risk.py::ks_drop_ratio` / `stress_ks_risk` / `stress_psi_risk`）——`ks_drop_ratio` 的算术本身未见直接黄金测试；`stress_psi_risk` 的 0.10/0.25 分档仅在 `test_validation_batch_runner.py:651-675` 经 batch runner 间接覆盖（0.15→medium、0.30→high），`stress_ks_risk` 的 0.10/0.20 阈值未见直接锚。
4. **指标表渲染层（不适用）**——`test_metric_tables.py` 用夹具喂入指标值、只断言渲染/布局/格式化，属 B 类；它本就不重算指标，因此「无黄金锚」不是缺陷，但意味着**渲染层不会兜住上游指标算错**。

## 5. 变异抽查结论（推演，未实际注入）

### 5.1 `feature_auc`（AUC）——「类别顺序搞反」

实现用 `pos = scores[target==1]` / `neg = scores[target==0]` 做 Mann-Whitney。若把 pos/neg 对调（相当于 target 0/1 反转），非方向无关路径的 AUC 会翻转为 `1 - auc`。

- 会不会红：**会**。`test_feature_metrics_reports_direction_agnostic_auc_for_single_features`（断言 0.0 与 1.0）、`test_overall_auc_keeps_declared_positive_score_direction`（断言 0.0）、`test_feature_auc_matches_sklearn_rank_auc`（sklearn 对拍）都会失败。
- 结论：AUC 的类别顺序错误会被**独立锚**（手算黄金 + sklearn）抓住，保护强度高。

### 5.2 `feature_ks`（KS）——「符号取反」与「类别顺序搞反」

- 符号取反（去掉 `abs` 或加负号）：**会红**。`test_compute_ks_known_values` 断言 +1.0（负值不匹配），`test_overall_metrics_cover_all_three_splits` 断言 `0 <= ks <= 1`，`test_roc_ks_curve_scalar_matches_canonical_feature_ks` 对拍非负 ROC KS。
- 类别顺序搞反（swap bad/good 计数）：**不会红**，但这不是 bug——KS 用 `abs(cum_bad - cum_good)`，对 bad/good 对称。
- 更隐蔽的 change-points/cumsum 中间逻辑错误：只有 `_naive_ks` 阈值扫描（B 类独立复算）能兜住非退化中间值；0/1 极端黄金对此无能为力。
- 结论：符号错误会被抓；「非退化 KS 数值」仍缺硬编码黄金锚（唯一例外是加权 KS `4/7`）。建议补一个非退化 KS 黄金锚（见 §6 P1）。

### 5.3 `compute_psi`（PSI）——「符号取反」与「公式约定系统性错误」

- 加负号（`max(0, -sum(...))` → 恒 0）：**会被 C 类测试抓住**（`test_compute_psi_and_feature_psi_use_shared_edges_and_smoothing` 期望一个正数）。但抓住它的正是 C 类镜像测试——保护强度取决于「镜像测试作者是否把约定写对」。
- 系统性约定错误（如平滑/归一化步骤、零桶处理约定写错，且实现与 C 类镜像同时按错误约定写）：**不会红**。因为 PSI 没有任何独立 A 类黄金常量，实现与镜像测试会一起错、一起绿。
- 结论：**PSI 是五个关键指标中变异保护最弱的一个**。这是本次审计最需要补锚的地方（见 §6 P1）。

## 6. 补锚点队列（按优先级）

优先级判定标准：确定性铁律下影响面 × 当前锚点强度缺口 × 补锚成本。

### P1 — PSI 非退化黄金常量（最高优先）

- 理由：唯一缺乏非退化黄金锚的关键指标；`compute_psi` 是 PSI 稳定性、pool/candidate stability、压力 PSI 的共同底座，错误会被放大到多张报告。
- 怎么补：在 `tests/test_feature_metrics.py` 增加一个测试，硬编码一组手算分布的手算 PSI 数值。例如 `expected=[0.5, 0.5]`、`actual=[0.6, 0.4]` → 手算 `(0.6-0.5)*ln(0.6/0.5) + (0.4-0.5)*ln(0.4/0.5)` 并**直接写死数值**（≈ `0.040547`），不要用 `np.log` 表达式、不要调 `compute_psi`。同时补一个含零桶的手算数值锚，锁定「零桶平滑 + 重归一化」两步约定。

### P2 — 非加权 KS 非退化黄金常量

- 理由：`feature_ks` 是 KS 全链路（overall / monthly / ROC / 压力衰减）的底座；现有黄金只在 0/1 极端。
- 怎么补：构造一个「不完全分离」的小样本（如 8-12 行、好坏交错），用纸笔/独立脚本算出 KS 的中间值（如 0.4~0.6），在 `tests/validation/test_binning.py` 或 `tests/test_feature_metrics.py` 硬编码该数字（保留 `_naive_ks` 扫描作 B 类佐证，但新增 A 类硬编码）。

### P3 — 压力分档阈值黄金（`stress_risk.py`）

- 理由：`ks_drop_ratio` 算术与 `stress_ks_risk` 的 0.10/0.20、`stress_psi_risk` 的 0.10/0.25 阈值缺乏直接锚（目前只有 psi 经 batch runner 间接覆盖）。
- 怎么补：新增 `tests/validation/test_stress_risk.py`，对 `ks_drop_ratio`（如 `(0.5, 0.4) -> 0.2`、基线为 0/NaN -> None）、`stress_ks_risk`/`stress_psi_risk` 的阈值边界（0.10、0.20、0.25 及 `+1e-12` 容差）逐档硬编码断言。

### P4 — C 类镜像的「黄金化」（次要，防漂移加固）

- 理由：`test_binning.py` 的 `_legacy_accumulate` 镜像已被 `test_bin_table_golden_values_preserved` 钉住；其余镜像（WOE inline、PSI inline）应至少各配一条硬编码数字锚。
- 怎么补：`test_feature_iv.py` 已有一条 `total_iv == 2.145917` 黄金（良好）；给 `test_feature_encode.py` 的 WOE 内联期望补一条写死数值，给 §3 第 1/2 条 PSI 内联测试补写死数值。

## 7. 我无法完成的检查

1. **全量 12,000+ 测试的穷举分级**——本次只对确定性内核的测试函数逐一分级，未穷举全仓库（尤其 strategy 大量 e2e/渲染/编排测试）。其余域的 C 类测试可能存在但未被枚举。
2. **实际变异注入验证**——铁律「不改动任何代码」约束下，未做真实的变异测试（mutation testing）；§5 全部为阅读实现后的推演结论，非实测。
3. **`test_modeling_recipes.py:157-158` 的 C/B 定级**——未深入 `compute_model_metrics` 内部确认其是否直接调用 `feature_ks/feature_auc`；若内部直接委托则为同义反复（C），若独立路径则为 B 类互证。当前按「平台函数互证」保守记为 C（列入清单），定级可能偏严。
4. **`marvis/metric_tables.py` 渲染层**——已确认不重算指标（纯渲染），但未逐一核对渲染层的格式化/舍入是否可能对指标值做二次变换。
5. **strategy 域 impact cube / economics 的更细锚点**——`test_strategy_impact_cube.py`、`test_strategy_economics.py` 等未逐条分级；本报告只覆盖了坏率/通过率（tradeoff/backtest）与 waterfall 稳定性。
