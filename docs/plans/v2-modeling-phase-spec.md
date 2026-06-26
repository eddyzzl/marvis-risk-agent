# MODELING 阶段详细 Spec（模型开发）

> V2 完善总计划第 3 份阶段 spec（见 [v2-completion-plan.md](v2-completion-plan.md)）。
> MODELING = 组合 **[JOIN, FEATURE, MODELING]**——甩几个 csv 进来即可"拼→析→筛→训"一条龙;复用 JOIN/FEATURE 两模块。
> 任意模型(已确认范围):二分类/多分类/回归 × lgb/xgb/lr/scorecard/dnn(sklearn MLP 先行)。

---

## 0. 地面真值（已核对）
- recipe 注册表(`recipes/__init__.py`):`register_recipe`/`get_recipe`;已有 **lgb / xgb / lr / scorecard**(二分类)、**lgb_regressor**(回归)。多分类/DNN **缺**,需新建。
- `TrainConfig` 已有 `target_type`(默认 binary)、`eval_metric`、`recipe_id` 字段——**对话流没接,需接**。
- 切分:`split_modeling_frame` **只会按已有 split 列切**;`resolve_modeling_splits` 处理 NaN 标签门 + scoring-only OOT(oot 缺失/无标签时报 n/a 不崩)。**无时间切/随机切生成器,需建 `make_split`**。
- 本会话已建并验证:`screen_features`(泄漏感知筛选)、`tune_hyperparameters`(随机搜索 + 过拟合惩罚,OOT 无偏)、`session.py`(将收编为本阶段步骤)。
- `report_compute.py` 已算:gains/lift 分箱、vintage、单变量 IV/KS/AUC/coverage、特征重要性、数据源移除压力测试。
- 多分类硬缺口:`feature/metrics.py _finite_binary_pairs` 对非二分类硬抛错;`modeling_readiness` 硬挡非二分类 → 需 target_type 感知。

---

## 1. 阶段总览（步骤序列与门）
```
[JOIN 跳过/完成] → [FEATURE 筛选完成,得选中特征集]
 → G1 切分门(必需)  → G2 算法/任务类型门  → G3 调参门(可跳过)
 → 训练  → G4 trials 排行 + 选模 + 详细报告门  → G5 训练后动作(对比/导出/移交)
```
- 任意门都遵守总计划 #5:确认 / 提指令(改参=重跑本步、结构=replan)。
- **不变量**:**oot 缺失时所有门/指标优雅降级**(报 n/a,不崩、不挡后续)——满足"不分 oot 也不影响后续"。

---

## 2. G1 切分门（必需;最重要,高度可定制）
- **先问有没有**:agent 问"有没有现成切分列";有 → 用用户说的。
- **没有 → 先样本分析**(target-type 感知):**分月 / 分渠道 / 分渠道×月** 的 样本量、y 统计(二分类=坏率;回归=均值/分位/分布;多分类=各类占比)、缺失率、分布 → 内联富表发给用户看 → 用户据时间/渠道分布**说怎么切**。
- **`make_split`：任意规则集(条件→train/test/oot)确定性生成切分列**:
  - 支持 已有列 / 时间 cutoff / 随机比例 / **任意分渠道分时段**(如"渠道A全train、渠道B 10月前test/10月后oot、渠道C全oot")。
  - 支持**只 train/test 不要 oot**。
- **切分不变量(护栏)**:
  1. **分组随机**:随机指派以 **`身份要素(人)+日期` 整组同侧**(组键=JOIN 身份键去渠道/订单),杜绝同人同日近重样本跨 train/test → 防过拟合。
  2. **固定 seed**,可复现。
  3. **非空校验**:规则切空 train/test → 阻断 + 告警。
  4. **oot 缺失安全**:下游全程优雅降级。
  5. **target-type 感知样本统计**;**同人跨段**(渠道规则导致)仅温和提醒不阻断。
- 手动:规则构建器(条件行:列+运算+值→指派)+ 预设(按列/时间/比例);agent:把自然语言规则翻成规则集应用。

## 3. G2 算法 / 任务类型门（无默认,每次必选）
- **任务类型单选**:二分类 / 多分类 / 回归。**进任务后脚本校验 y 是否符合**(0/1→二分类、连续→回归、小基数整数→多分类;不符报错纠正)。
- **算法 ≥1 选**:按类型列可用(二分类 lgb/xgb/lr/scorecard;回归 lgb_regressor(+后续 xgb_reg/dnn_reg);多分类 lgb/xgb/lr(后续));**选多个→多算法对比取最优**。多分类/DNN 建好前置灰。
- **手动**:创建弹窗勾选(类型单选 + 算法多选);**agent**:不展示选项,按**检测类型 + 数据/特征量级推荐**,用户确认/改。
- 接 `TrainConfig.target_type/recipe_id`(已有字段),驱动 recipe 分发 + 指标分发。

## 4. G3 调参门（可跳过）
- **可跳过**:「用默认快训」一键基线。
- **设搜索空间 + 轮数**:手动=超参范围控件(num_leaves/max_depth/lr/min_child/feature_frac/lambda 的 min-max + 轮数);agent=按量级提议空间+轮数,用户确认/改。
- **调参包 + 损失**:按各算法/场景选业界最优(树→贝叶斯搜索如 optuna + 对应 objective;回归/多分类各自最佳 loss)。
- **搜索内部选优用 in-time 指标**(test:二分类 KS 减惩罚、回归 RMSE、多分类 logloss/macro-AUC),**不偷看 OOT**。
- lr/scorecard 无随机搜索→只给各自旋钮(lr 正则;scorecard 分箱/PDO)。

## 5. G4 trials 排行 + 选模 + 详细报告门
- **trials 排行富表**:每 trial 列 train/test/oot × {KS、AUC、头部 lift5/10%、尾部 lift5/10%} + 过拟合 gap(train-test、train-oot)。→ 需扩 `tune.py` 把每 trial 指标算全。
- **选模归用户**:在排行表上**按任意列排序点选**(最高 OOT KS / OOT AUC / OOT 头部 lift / 最小过拟合 gap……);auto 高亮 in-time 最优,用户可覆盖。
- **不满意**:用户让 agent **换调参方式/范围重调重训**(重跑本步/replan),再回排行。
- **满意**:**支持同时选多版参数模型,各出一份报告**。
- **详细报告(固定格式 Excel,右栏下载,样式同模型验证)**:见 §6。选中即定模型 → 出报告 → 确认。

## 6. 详细报告格式（固定,可增不可删;参考用户两份样本）
> 参考:`复借T卡_多头裂变风险模型_20260605.xlsx`、`basic_all_分析报告_20250305.xlsx`。
固定 sheet（缺数据的指标留空/标 n/a,不删 sheet）:
1. **汇总/总结**:建模背景、项目概述、样本范围、关键结论。**手动模式无文字总结(仅表);agent 模式 agent 写**。
2. **样本分析**:① 分数据集(train/test/oot:样本量/占比/坏样本/逾期率/平均额度);② 分月(放款笔数/利率/金额/期数/逾期笔数…)。
3. **Vintage(可选)**:分月 fpd/mob 曲线——用户可关。
4. **特征重要性**:特征名/**含义/产品/厂商**(读数据字典)/gain/百分比/累计百分比。
5. **单变量分析**:coverage / train·test·oot IV / train·test·oot KS(来自 FEATURE)。
6. **OOT 分箱评估_十分箱**:区间/样本数/累积占比/好坏人数/人头逾期率/累积逾期率/**人头 lift**。
7. **压力测试**:逐数据源(产品/厂商)移除后 OOT 十分箱稳定性(需字典的特征→数据源映射)。
- **我可加我认为重要的内容(不删)**:如 ROC-KS 曲线 sheet、调参 trials sheet、混淆矩阵(多分类)、残差图(回归)。
- 多模型 → 文件名带版本,各一份。
- 复用 `report_compute.py` 算好的表 + xlsx 写出(对齐模型验证下载管道)。

## 7. G5 训练后动作
- `compare_experiments`(多版/多算法对比排行)、`export_pmml`(**lr/scorecard/lgb/xgb 均导 PMML**——扩现"仅 lr"实现,树走 nyoka/jpmml;**仅 DNN(MLP/torch)不走 PMML→导原生模型+打分脚本**)、`handoff_to_validation`(交模型验证,同样扩到树模型)。
- 以**选项按钮**出现(总计划 #5 的"下一步"选项);agent 模式 LLM 提议、手动模式按钮。

## 8. 任意模型构建子序（阶段内自底向上）
1. **二分类**(lgb/xgb/lr/scorecard):接 target_type/recipe 分发 + 现有指标/报告——最快通。
2. **回归**(lgb_regressor 已有 + 后续 xgb_reg / sklearn 线性):`compute_regression_metrics` 已有;readiness 放开非二分类;报告回归段(RMSE/MAE/R²/残差),关掉二分类专属(vintage/lift/分箱)或改回归口径。
3. **多分类**(新):`compute_multiclass_metrics`(macro/weighted AUC、logloss、accuracy、混淆/各类 recall)、`ModelMetrics` 加 Optional 多分类字段、打分器支持 2D proba、报告多分类段;`feature/metrics.py` 二分类硬断言要绕开。
4. **DNN**:**sklearn MLP 先行**(复用 lr 模式 + 现打分器,无新依赖、random_state 可复现);torch 作明确独立后续(提依赖 + 放宽 RLIMIT/超时 + 确定性 + 线程清理)。

## 9. 手动 vs agent（同底座换脸）
| 门 | 手动 | agent |
|---|---|---|
| G1 切分 | 样本分析表 + 规则构建器/预设 | 样本分析表 + 自然语言规则→规则集 |
| G2 算法/类型 | 创建弹窗勾选(类型单选+算法多选) | 不展示,按检测+量级推荐,用户确认 |
| G3 调参 | 超参范围控件 + 跳过按钮 | LLM 提议空间/轮数,用户确认 |
| G4 选模/报告 | 排行表排序点选 + 报告下载 | LLM 解读排行 + 建议 + 你拍板 + 报告(含叙述) |
| G5 动作 | 按钮 | LLM 提议 |

## 10. 映射 V2 原语 / 新建 vs 复用
- PlanSteps(phase=`建模`):make_split(G1,needs_confirmation)→ 选算法/类型(G2,needs_confirmation)→ tune(G3,可跳过)→ train(每算法)→ 选模+报告(G4,needs_confirmation)→ compare/export/handoff(G5)。
- **复用**:recipe 注册表 + TrainConfig 字段 + screen/tune(已建)+ report_compute 全表 + resolve_modeling_splits(oot 安全)+ xlsx 下载管道。
- **新建**:`make_split`(规则集 + 分组随机 + 样本分析)、target_type/recipe 全程接线、tune 每 trial 全指标、固定格式报告写出、回归/多分类/DNN recipe 与指标、export/handoff 扩非 lr。

## 11. 已锁小项
1. **导出/移交**(已定):**lr/scorecard/lgb/xgb 均导 PMML**(扩现"仅 lr",树走 nyoka/jpmml);**仅 DNN(MLP/torch)走原生模型+打分脚本**(PMML 不实际)。
2. **多分类报告**(已定):lift/分箱/vintage 等二分类专属 sheet **本版标 n/a,下版迭代**再做"风险最高类 vs 其余"。
3. **压力测试**(已定):靠字典"特征→数据源"映射;无字典则该 sheet 留空标"缺字典"。
