# 建模 Notebook 提交要求

## 1. 目的

本文件只描述 MARVIS V1.1.8 当前内置的模型验证工作流，不代表平台只能做模型验证。该工作流会执行开发人员提交的 Jupyter Notebook，并使用 Notebook 运行后留在内存里的模型对象进行打分，再与开发人员提交目录中的 PMML 模型打分结果进行一致性比较。

本要求的目标是：

- 验证人员不需要理解 Notebook 代码、模型变量名、模型类型、特征字段或评分方式。
- 开发人员通过固定契约暴露模型评分函数、目标字段、可选的报告材料。
- 平台可以自动检查 Notebook 是否满足要求，并在缺少契约或运行失败时给出明确错误。
- 平台按 Notebook 标题展示执行进度，但执行顺序仍保持 Notebook 原始顺序。

## 2. 文件要求

开发人员提交的 Notebook 必须满足：

- 文件格式为 `.ipynb`。
- Notebook 能从头到尾完整执行。
- 不依赖人工点击、手动输入、临时交互确认。
- 不要求验证人员修改 Notebook 代码。
- 不要求验证人员填写模型变量名、模型类型、特征字段或评分表达式。
- Notebook 最后必须定义平台契约变量和函数。

开发人员同时提交的 PMML 必须满足：

- PMML 是最终拟投产模型文件。
- PMML 内包含 DataFrame mapper、pipeline 或等价结构，能够直接对平台传入的原始样本 DataFrame 打分。
- PMML 不要求验证人员提供特征清单。
- PMML 的输出字段名默认使用 `probability_1`，如不一致，需在 Notebook 契约中指定。

## 3. Notebook 标题要求

平台会解析 Notebook 中的 Markdown 标题，并按标题展示执行步骤。

支持的标题层级：

```markdown
# 数据准备
## 特征处理
### 模型训练
```

要求：

- 建议 Notebook 的主要步骤都用 Markdown 标题分隔。
- 标题下面的代码 cell 会归属到该标题步骤。
- 第一个标题前的代码 cell 会归为 `Notebook 初始化`。
- 平台只按标题展示进度，不会改变 Notebook 执行顺序。
- 平台不会跳过 cell，也不会按标题拆成多个独立进程。
- 整个 Notebook 会在同一个 kernel 中从上到下执行。
- 模型效果与稳定性验证会追加到第 2 步刚执行完的同一个 kernel，直接复用内存里的 `RMC_SAMPLE_DF`。

推荐标题结构：

```markdown
# 1. 环境与参数
# 2. 数据读取
# 3. 样本处理
# 4. 特征处理
# 5. 模型训练
# 6. 模型评估
# 7. 平台验证契约
```

## 4. 必填契约

Notebook 执行结束前，必须在顶层作用域定义以下内容：

```python
RMC_SAMPLE_DF = modeling_sample
RMC_TARGET_COL = "target"
RMC_ALGORITHM = "lgb"

def RMC_SCORE_FN(df):
    return final_model.predict_proba(df)[:, 1]
```

### 4.1 `RMC_SAMPLE_DF`

`RMC_SAMPLE_DF` 是平台验证使用的样本 DataFrame。

要求：

- 必须是 pandas DataFrame。
- 必须在 Notebook 顶层作用域可见。
- 必须包含目标列、分组列、时间列，以及 `RMC_SCORE_FN` 和 PMML scorer 需要的字段。
- 第 2 步分数一致性和第 3 步模型效果&稳定性验证都会直接使用它，不会让平台进程重新读取样本文件再分析。

### 4.2 `RMC_SCORE_FN`

`RMC_SCORE_FN` 是平台调用内存模型打分的唯一入口。

格式要求：

```python
def RMC_SCORE_FN(df):
    ...
    return scores
```

要求：

- 必须是可调用函数。
- 入参必须能接收一个 pandas DataFrame。
- 平台会把同一份抽样原始 DataFrame 传给该函数。
- 函数内部自行完成特征选择、字段顺序处理、缺失值处理、WOE/标准化等模型打分所需逻辑。
- 返回值必须是一维分数，长度等于输入 DataFrame 行数。
- 返回值必须是数值。
- 返回值不能包含空值、无穷值。
- 返回值应为正类概率或平台报告所需的模型分。

不要让平台猜模型对象。Notebook 必须明确提供 `RMC_SCORE_FN`。

示例：

```python
def RMC_SCORE_FN(df):
    return final_model.predict_proba(df)[:, 1]
```

如果模型需要特定字段或转换，在函数内部处理：

```python
def RMC_SCORE_FN(df):
    x = df[["age", "income", "loan_cnt"]].copy()
    x["income"] = x["income"].fillna(0)
    return final_model.predict_proba(x)[:, 1]
```

### 4.3 `RMC_TARGET_COL`

`RMC_TARGET_COL` 是样本中的目标标签列，用于 KS、AUC、分箱效果等验证指标。

格式要求：

```python
RMC_TARGET_COL = "target"
```

要求：

- 必须是字符串。
- 该字段必须存在于平台抽样数据中。
- 字段值应能表达好坏样本标签。

### 4.4 `RMC_ALGORITHM`

`RMC_ALGORITHM` 是模型算法类型，由开发人员在 Notebook 契约中填写，验证人员不再在平台新建任务时选择。

格式要求：

```python
RMC_ALGORITHM = "lgb"
```

平台内部只保存以下规范值：

- `lgb`：LightGBM。
- `xgb`：XGBoost。
- `lr`：逻辑回归。
- `catboost`：CatBoost。
- `scorecard`：评分卡。
- `dnn`：深度神经网络。

平台会接受常见别名并归一化，例如：

- `lgb`、`lgbm`、`lightgbm`、`Lightgbm`、`LGBMClassifier` -> `lgb`。
- `xgb`、`xgboost`、`XGBClassifier`、`xgboost.XGBClassifier` -> `xgb`。
- `lr`、`logistic`、`LogisticRegression`、`逻辑回归` -> `lr`。
- `catboost`、`CatBoostClassifier` -> `catboost`。
- `scorecard`、`score_card`、`评分卡`、`记分卡` -> `scorecard`。
- `dnn`、`deep neural network`、`neural network`、`神经网络`、`深度神经网络` -> `dnn`。

未知算法会在静态契约检查阶段失败。开发人员应修改 Notebook 契约，不应要求验证人员在平台界面兜底选择。

## 5. 不要求 `RMC_FEATURES`

平台不要求 Notebook 定义 `RMC_FEATURES`。

原因：

- 代码模型打分由 `RMC_SCORE_FN(df)` 自行决定使用哪些字段。
- PMML 侧要求开发人员提交的 PMML 内包含 DataFrame mapper 或等价 pipeline，能自行从原始 DataFrame 取字段。
- 平台只保证把同一份抽样原始 DataFrame 分别传给 `RMC_SCORE_FN` 和 PMML scorer。

因此，验证人员不需要填写特征清单，平台也不会把特征清单作为主一致性验证的必填项。

## 6. 可选契约

以下变量不是所有任务都必填。平台在变量存在时读取并写入报告；不存在时跳过对应内容或使用默认值。

### 6.1 `RMC_SPLIT_COL`

样本分组字段，例如训练集、测试集、OOT。

```python
RMC_SPLIT_COL = "sample_type"
```

要求：

- 可选。
- 如果报告需要分样本集展示效果，则建议提供。
- 必须是字符串。
- 字段存在时，平台会按该字段做分组指标。

### 6.2 `RMC_TIME_COL`

时间字段，例如申请月份、观察月份。

```python
RMC_TIME_COL = "apply_month"
```

要求：

- 可选。
- 如果报告需要 PSI、稳定性、跨期表现，则建议提供。
- 必须是字符串。

### 6.3 `RMC_PMML_OUTPUT_FIELD`

PMML 输出字段名。

```python
RMC_PMML_OUTPUT_FIELD = "probability_1"
```

要求：

- 可选。
- 默认值为 `probability_1`。
- 如果 PMML 输出字段不是 `probability_1`，必须指定。

### 6.4 `RMC_SCORE_DECIMAL_PLACES`

模型分一致性比较精度。

```python
RMC_SCORE_DECIMAL_PLACES = 6
```

要求：

- 可选。
- 默认值为 `6`。
- 平台会在统一的比较函数中按该配置判断代码模型分和 PMML 模型分是否一致。

## 7. 特征重要性要求

如果报告需要特征重要性，Notebook 应定义：

```python
RMC_FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": feature_names,
    "类别": feature_categories,  # 可选，也可命名为 "category"
    "importance": importance_values,
})
```

格式要求：

- 类型必须是 pandas DataFrame。
- 必须包含两列：`feature`、`importance`。
- 可选包含 `类别` 或 `category` 列；平台会统一写入 Web、Excel 和 Word 的“类别”列。
- `feature` 为特征名，建议为字符串。
- `importance` 为重要性数值。
- 平台会按 `importance` 降序展示和写入报告。
- 平台不会自行决定是否取绝对值。
- 如果是 LR 系数，开发人员应在 Notebook 中自行决定传原始系数还是 `abs(coef)`。

LR 示例：

```python
RMC_FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": feature_names,
    "importance": [abs(v) for v in final_model.coef_[0]],
})
```

LightGBM 示例：

```python
RMC_FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": feature_names,
    "importance": final_model.feature_importances_,
})
```

兼容旧变量：

```python
FEATURE_IMPORTANCE = RMC_FEATURE_IMPORTANCE
```

平台可以在过渡期兼容 `FEATURE_IMPORTANCE`，但新 Notebook 推荐使用 `RMC_FEATURE_IMPORTANCE`。

## 8. 模型参数要求

如果报告需要模型参数，Notebook 应定义：

```python
RMC_MODEL_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
```

格式要求：

- 类型必须是 dict。
- key 必须是字符串。
- value 建议使用 JSON 可序列化的简单类型：`str`、`int`、`float`、`bool`、`None`。
- value 可以是 list 或 dict，但平台只负责展示，不解释模型语义。
- 平台会在报告中渲染为两列表：参数名、参数值。

兼容旧变量：

```python
MODEL_HYPERPARAMETERS = RMC_MODEL_PARAMS
```

平台可以在过渡期兼容 `MODEL_HYPERPARAMETERS`，但新 Notebook 推荐使用 `RMC_MODEL_PARAMS`。

过渡期内，如果旧 Notebook 尚未补充 `RMC_ALGORITHM`，平台可以从 `RMC_MODEL_PARAMS["algorithm"]` 或 `MODEL_HYPERPARAMETERS["algorithm"]` 读取算法；新 Notebook 必须优先使用第 4.4 节的 `RMC_ALGORITHM`。

## 9. 推荐完整契约 cell

建议放在 Notebook 最后一个普通代码 cell。

```python
import pandas as pd

RMC_TARGET_COL = "target"
RMC_SAMPLE_DF = modeling_sample
RMC_ALGORITHM = "lgb"
RMC_SPLIT_COL = "sample_type"
RMC_TIME_COL = "apply_month"
RMC_PMML_OUTPUT_FIELD = "probability_1"
RMC_SCORE_DECIMAL_PLACES = 6

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

# 过渡期兼容旧字段，可选
FEATURE_IMPORTANCE = RMC_FEATURE_IMPORTANCE
MODEL_HYPERPARAMETERS = RMC_MODEL_PARAMS
```

## 10. PMML 要求

开发人员提交的 PMML 是拟投产模型文件。平台会把 PMML 打分结果与 Notebook 内存模型打分结果对比。

PMML 必须满足：

- 能使用平台传入的原始样本 DataFrame 打分。
- 内部包含 DataFrame mapper、pipeline、derived fields 或等价转换逻辑。
- 不依赖验证人员额外配置特征清单。
- 输出字段可通过 `RMC_PMML_OUTPUT_FIELD` 指定。

平台不会将 Notebook 重新导出的 PMML 作为主对比对象。

主对比路径：

```text
Notebook 内存样本 -> RMC_SAMPLE_DF
Notebook 执行后内存模型 -> RMC_SCORE_FN(RMC_SAMPLE_DF.copy()) -> code_model_scores
提交目录中的 PMML -> PMML scorer(RMC_SAMPLE_DF.copy()) -> submitted_pmml_scores
code_model_scores vs submitted_pmml_scores -> 一致性结论
```

## 11. 平台检查逻辑

### 11.1 执行前静态检查

平台会在执行 Notebook 前扫描代码文本，检查是否出现：

- `RMC_SCORE_FN`
- `RMC_SAMPLE_DF`
- `RMC_TARGET_COL`
- `RMC_ALGORITHM`（或过渡期模型参数里的 `algorithm`）

如果缺少，任务会在执行前失败。

示例错误：

```text
Notebook contract check failed before execution:
missing RMC_SCORE_FN

Please add the MARVIS contract cell at the end of the notebook.
```

静态检查只用于提前发现明显缺失，不证明函数一定正确。

### 11.2 执行后运行时检查

Notebook 原始 cell 全部执行完成后，平台会在同一个 kernel 中执行系统检查 cell。

平台会检查：

- `RMC_SCORE_FN` 是否存在。
- `RMC_SCORE_FN` 是否可调用。
- `RMC_SAMPLE_DF` 是否存在且为 pandas DataFrame。
- `RMC_TARGET_COL` 是否存在且为字符串。
- `RMC_ALGORITHM` 是否存在且能归一化为支持的算法。
- `RMC_SCORE_FN(RMC_SAMPLE_DF.copy())` 是否能正常运行。
- 输出长度是否等于样本行数。
- 输出是否为数值。
- 输出是否包含空值或无穷值。
- `RMC_FEATURE_IMPORTANCE` 存在时格式是否正确。
- `RMC_MODEL_PARAMS` 存在时格式是否正确。

如果任何检查失败，平台会展示具体错误，并保留执行日志和失败 Notebook。

## 12. 不允许的写法

不要只在函数内部定义契约变量：

```python
def main():
    RMC_TARGET_COL = "target"
    def RMC_SCORE_FN(df):
        return model.predict_proba(df)[:, 1]

main()
```

原因：平台尾部检查 cell 在 Notebook 顶层作用域读取变量，函数内部局部变量不可见。

不要依赖手动输入：

```python
threshold = input("请输入阈值")
```

不要在 `RMC_SCORE_FN` 中修改外部数据文件：

```python
def RMC_SCORE_FN(df):
    df.to_csv("debug.csv")
    return final_model.predict_proba(df)[:, 1]
```

不要让 `RMC_SCORE_FN` 返回二维数组：

```python
def RMC_SCORE_FN(df):
    return final_model.predict_proba(df)
```

应返回一维正类概率：

```python
def RMC_SCORE_FN(df):
    return final_model.predict_proba(df)[:, 1]
```

## 13. 运行环境要求

平台会在设置中选择 Python、conda 或 Jupyter kernel 环境。

Notebook 必须能在所选环境中运行。开发人员应确保该环境包含 Notebook 依赖，例如：

- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- catboost
- pypmml 或 PMML 相关依赖
- tensorflow、pytorch 或其他 DNN 推理依赖（如模型使用）
- 训练代码中使用的其他内部包

如果 Notebook 依赖特定 conda 环境，应在提交说明中写清楚环境名称。

平台执行前会检查所选 kernel 是否可用；依赖包缺失导致的运行失败会展示在执行日志中。

## 14. 失败处理

平台可能给出的失败类型包括：

- 未找到 Notebook。
- Notebook 静态契约检查失败。
- Notebook 某个 cell 执行失败。
- `RMC_SCORE_FN` 不存在。
- `RMC_SCORE_FN` 输出长度不正确。
- `RMC_SCORE_FN` 输出包含非数值、空值或无穷值。
- `RMC_TARGET_COL` 不存在或样本中缺少该列。
- PMML 打分失败。
- PMML 输出字段不存在。
- 代码模型分与 PMML 模型分不一致。

失败后平台会保留：

- 原始 Notebook 哈希。
- 执行副本。
- 执行日志。
- 失败 cell 信息。
- stdout / stderr 摘要。
- 合约检查结果。
- 分数差异明细，若已生成。

## 15. 最小合格示例

下面是最小可通过契约检查的 Notebook 末尾代码。

```python
RMC_SAMPLE_DF = modeling_sample
RMC_TARGET_COL = "target"
RMC_ALGORITHM = "lgb"

def RMC_SCORE_FN(df):
    return final_model.predict_proba(df)[:, 1]
```

如果报告需要更完整内容，使用第 9 节的完整契约 cell。

## 16. 提交前自检清单

开发人员提交前应确认：

- Notebook 能从头到尾运行完成。
- Notebook 末尾有 `RMC_SAMPLE_DF`。
- Notebook 末尾有 `RMC_SCORE_FN`。
- Notebook 末尾有 `RMC_TARGET_COL`。
- Notebook 末尾有 `RMC_ALGORITHM`，且属于平台支持的算法或常见别名。
- `RMC_SCORE_FN(df)` 返回一维数值分数。
- `RMC_SCORE_FN(df)` 返回长度等于输入样本行数。
- 提交的 PMML 能直接对原始样本 DataFrame 打分。
- PMML 输出字段与 `RMC_PMML_OUTPUT_FIELD` 一致，或使用默认 `probability_1`。
- 如需特征重要性，已定义 `RMC_FEATURE_IMPORTANCE`。
- 如需模型参数，已定义 `RMC_MODEL_PARAMS`。
- Notebook 不依赖人工输入。
- Notebook 不要求验证人员理解模型变量名或代码细节。
