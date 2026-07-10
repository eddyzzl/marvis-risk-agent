# 建模 Notebook 与平台的契约

平台会执行开发人员提交的 Jupyter Notebook，并用 Notebook 运行后留在内存里的模型对象打分，再与提交目录中的拟投产 PMML 打分结果做一致性比较。
模型效果与稳定性验证也会追加到同一个 Notebook kernel 中执行，直接复用 Notebook 内存里的样本 DataFrame。

完整提交要求见 [对notebook的要求.md](./对notebook的要求.md)。本文件是平台运行契约的开发版摘要。

## 核心原则

- 验证人员不需要填写模型变量名、模型类型、特征字段或评分表达式。
- Notebook 必须显式暴露平台契约，平台不自动猜模型对象。
- 主一致性验证比较的是 `RMC_SCORE_FN(sample_df)` 与提交 PMML 的打分结果。
- Notebook 可以额外导出 PMML 作为审计材料，但平台不再把新导出的 PMML 作为主对比对象。
- `RMC_FEATURES` 不再需要。代码模型和 PMML 都应能从同一份原始样本 DataFrame 中自行取数和转换。

## 必填契约

Notebook 执行结束前，必须在顶层作用域定义：

```python
RMC_SAMPLE_DF = modeling_sample
RMC_TARGET_COL = "target"
RMC_ALGORITHM = "lgb"

def RMC_SCORE_FN(df):
    return final_model.predict_proba(df)[:, 1]
```

`RMC_SAMPLE_DF` 要求：

- 类型为 pandas DataFrame。
- 是平台用于分数一致性、KS、PSI、分箱、压力测试的原始样本。
- 包含目标列、分组列、时间列，以及 PMML scorer 和 `RMC_SCORE_FN` 需要的字段。
- 必须在 Notebook 顶层作用域可见，不能只存在于函数局部变量中。

`RMC_SCORE_FN(df)` 要求：

- 接收 pandas DataFrame。
- 返回一维数值分数。
- 返回长度等于输入行数。
- 不返回空值或无穷值。
- 函数内部自行处理特征选择、字段顺序、缺失值、WOE/标准化等评分逻辑。

`RMC_TARGET_COL` 要求：

- 类型为字符串。
- 对应字段必须存在于样本数据中。

`RMC_ALGORITHM` 要求：

- 类型为字符串。
- 平台会归一化为内部枚举：`lgb`、`xgb`、`lr`、`catboost`、`scorecard`、`dnn`。
- 可接受常见别名，例如 `lightgbm`、`lgbm`、`LGBMClassifier` -> `lgb`，`xgboost`、`XGBClassifier` -> `xgb`，`LogisticRegression`、`逻辑回归` -> `lr`，`CatBoostClassifier` -> `catboost`，`评分卡`、`score_card` -> `scorecard`，`deep neural network`、`神经网络` -> `dnn`。
- 未知算法会在静态契约检查阶段失败，开发人员需要修正 Notebook，不由验证人员在平台新建任务时选择。

## 可选契约

```python
RMC_SPLIT_COL = "sample_type"
RMC_TIME_COL = "apply_month"
RMC_PMML_OUTPUT_FIELD = "probability_1"
RMC_SCORE_DECIMAL_PLACES = 6
```

- `RMC_SPLIT_COL`：样本分组字段，用于 train/test/OOT 指标。
- `RMC_TIME_COL`：时间字段，用于逐月效果和稳定性。
- `RMC_PMML_OUTPUT_FIELD`：PMML 正类输出字段，默认 `probability_1`。
- `RMC_SCORE_DECIMAL_PLACES`：代码模型分与 PMML 模型分的一致性比较精度，默认 6 位。

## 特征重要性

如报告需要特征重要性，Notebook 应定义：

```python
RMC_FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": feature_names,
    "类别": feature_categories,  # 可选，也可命名为 "category"
    "importance": importance_values,
})
```

要求：

- 类型为 pandas DataFrame。
- 必须包含 `feature`、`importance` 两列。
- 可选包含 `类别` 或 `category` 列；平台会统一写入 Web、Excel 和 Word 的“类别”列。
- 压力测试将非空的 `类别` 或 `category` 视为该最终 `feature` 名称的权威分类。
- 数据字典只会按完整特征名精确补充空类别，不会猜测前缀、后缀或派生特征名。
- 如果仍有入模特征无法分类，压力测试整体状态为 `partial`；没有任何入模特征可分类时为 `failed`，不会显示为完整完成。
- `importance` 必须是数值。
- 平台按 `importance` 降序展示。
- 平台不会自行决定是否取绝对值，LR 系数等场景由开发人员在 Notebook 中决定。

过渡期兼容旧变量名：

```python
FEATURE_IMPORTANCE = RMC_FEATURE_IMPORTANCE
```

## 模型参数

如报告需要模型参数，Notebook 应定义：

```python
RMC_MODEL_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
}
```

要求：

- 类型为 dict。
- key 必须是字符串。
- value 建议使用 JSON 可序列化的简单值；平台只负责展示，不解释模型语义。

过渡期兼容旧变量名：

```python
MODEL_HYPERPARAMETERS = RMC_MODEL_PARAMS
```

## 推荐完整契约 Cell

建议放在 Notebook 最后一个普通代码 cell。

```python
import pandas as pd

RMC_TARGET_COL = "target"
RMC_ALGORITHM = "lgb"
RMC_SPLIT_COL = "sample_type"
RMC_TIME_COL = "apply_month"
RMC_PMML_OUTPUT_FIELD = "probability_1"
RMC_SCORE_DECIMAL_PLACES = 6
RMC_SAMPLE_DF = modeling_sample

def RMC_SCORE_FN(df):
    return final_model.predict_proba(df)[:, 1]

RMC_FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values,
})

RMC_MODEL_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 300,
}
```

## 平台检查

执行前静态检查：

- 扫描代码 cell 文本。
- 缺少 `RMC_SAMPLE_DF`、`RMC_SCORE_FN`、`RMC_TARGET_COL` 或 `RMC_ALGORITHM` 时，不执行 Notebook，直接失败。
- 过渡期可以从 `RMC_MODEL_PARAMS["algorithm"]` 或 `MODEL_HYPERPARAMETERS["algorithm"]` 读取算法，但新 Notebook 应使用 `RMC_ALGORITHM`。

执行后运行时检查：

- 在同一个 kernel 中运行平台尾部检查 cell。
- 验证 `RMC_SAMPLE_DF` 是 pandas DataFrame。
- 验证 `RMC_SCORE_FN` 可调用。
- 验证目标列存在。
- 调用 `RMC_SCORE_FN(RMC_SAMPLE_DF.copy())` 并写出代码模型分。
- 验证分数长度、数值类型、空值、无穷值。
- 验证可选的特征重要性和模型参数格式。

## PMML 要求

提交目录中的 PMML 是拟投产模型文件。平台会把同一份原始样本 DataFrame 传给提交 PMML scorer。

PMML 必须包含 DataFrame mapper、pipeline、derived fields 或等价结构，能够自行完成特征选择和转换。平台不要求验证人员提供特征清单。

主对比路径：

```text
Notebook 内存样本 -> RMC_SAMPLE_DF
Notebook 内存模型 -> RMC_SCORE_FN(RMC_SAMPLE_DF.copy()) -> code_model_scores
提交 PMML -> PMML scorer(RMC_SAMPLE_DF.copy()) -> submitted_pmml_scores
code_model_scores vs submitted_pmml_scores -> 一致性结论
```

默认完整流水线中，第 3 步“模型效果&稳定性验证”不会重新执行 Notebook，也不会由平台进程重新读取样本文件；它会在同一次隔离 Notebook 执行末尾追加验证 cell，直接使用 `RMC_SAMPLE_DF`、`RMC_SCORE_FN` 和 `RMC_FEATURE_IMPORTANCE`。

如果用户单独重新执行第 3 步，平台会重新执行原 Notebook，以确定性重建上述契约对象，再在本次执行末尾追加指标 cell。平台不会复用已经退出或可能污染的旧 kernel，也不会绕过 Notebook 后重新解释原始特征分类。

## 常见失败

- `Notebook contract check failed before execution: missing RMC_SCORE_FN`
  - Notebook 没有定义评分函数。
- `Notebook contract check failed before execution: missing RMC_SAMPLE_DF`
  - Notebook 没有在顶层暴露样本 DataFrame。
- `RMC_SCORE_FN returned N scores for M rows`
  - 评分函数返回长度不等于样本行数。
- `RMC_SCORE_FN returned null scores`
  - 评分函数返回空值。
- `RMC_FEATURE_IMPORTANCE must be a pandas DataFrame`
  - 特征重要性格式不符合要求。
- `RMC_MODEL_PARAMS must be a dict`
  - 模型参数格式不符合要求。
- `unsupported model algorithm`
  - `RMC_ALGORITHM` 或过渡期模型参数里的算法写法无法归一化。
- `submitted PMML scorer returned N scores for M rows`
  - PMML 打分输出长度异常。

## 最小合格示例

```python
RMC_SAMPLE_DF = modeling_sample
RMC_TARGET_COL = "target"
RMC_ALGORITHM = "lgb"

def RMC_SCORE_FN(df):
    return final_model.predict_proba(df)[:, 1]
```
