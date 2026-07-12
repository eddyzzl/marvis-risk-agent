# 模型验证 PMML 打分迁移与零退化设计

## 状态

- 日期：2026-07-12
- 状态：已完成对话设计确认，待书面规格复核
- 适用范围：MARVIS V2.x 主线中的模型验证手动模式与 Agent 模式
- 迁移原则：只替换代码模型与 PMML 的一致性验证流程；其余现有能力不得退化

## 背景

当前模型验证会执行用户提交的 Notebook，取得内存模型和
`RMC_SCORE_FN`，再将代码模型分与提交 PMML 分进行一致性比较。真实模型项目普遍依赖
旧版 Python、XGBoost、LightGBM、私有包、外部路径、数据库或历史中间数据。平台无法仅靠
安装时固定一个 Python 环境，稳定复刻所有模型开发环境；为每个项目构建容器也不能补齐
私有依赖、外部数据和运行时状态。

因此，新流程不再把“重现开发环境”作为模型验证的前置条件，也不再执行 Notebook 或加载
PKL。提交 PMML 成为验证任务唯一的模型评分实现：平台对验证样本全量评分，后续效果、
稳定性、分箱和压力测试统一使用该分数。

这次迁移不是模型验证能力重做。当前 Word、Excel、Agent 对话、图表、指标、报告确认、
手动模式和 Agent 模式均属于稳定产品能力，必须原样保留。允许变化的范围只包括旧的
“模型可复现性／分数一致性”阶段及其直接证据、文案和持久化结构。

## 目标

1. 不执行 Notebook，不依赖模型开发时的 Python、包版本、私有环境或容器。
2. 将提交 PMML 作为唯一评分模型，对完整验证样本执行可审计的全量打分。
3. 使用同一 PMML 分数完成现有 KS、AUC、PSI、分箱、逐月效果和稳定性分析。
4. 在完整 OOT 上使用同一 PMML 完成必做的模型压力测试。
5. 保留现有 Word 模板、Excel 明细、Agent 图表和结论流程，不减少任何非一致性能力。
6. 将 Word 最终内容、图表及底层数据完整保存到 Excel，形成一份可检索、可计算的工作底稿。
7. 对百万级数据采用分块批量 PMML 评分，消除当前逐行 Python/JVM 调用瓶颈。

## 非目标

- 不重现 Notebook 的训练过程。
- 不验证代码模型、PKL、样本原有分数列与 PMML 的数值一致性。
- 不执行 Notebook 中的任意 Python、Shell、SQL、网络或文件写入代码。
- 不为不同客户或模型项目构建开发环境容器。
- 不从 PMML 猜测训练时的真实 feature importance。
- 不改写现有 Word 模板版式，不重新设计现有 Agent 工作区或验证指标。
- 不以本次迁移为由删除历史任务、历史报告或旧结果的只读展示能力。

## 核心决策

### 1. 四类材料全部必填

每个新模型验证任务必须由用户明确选择以下四类材料：

1. Notebook（`.ipynb`）；
2. 验证样本（CSV、Parquet、Feather 或平台已支持的表格格式）；
3. 拟验证 PMML；
4. 数据字典／特征元数据。

Notebook 必传但只读。平台只分析源码、Markdown、保存的有限输出和静态赋值关系，不启动
kernel，不选择 conda 环境，也不检查训练依赖是否安装。

### 2. PMML 是唯一评分真源

新任务不再生成代码模型分，不读取 PKL，也不信任样本中已有的 `score`、`predict`、
`pred_pmml` 等分数列。所有确定性验证指标只消费本次任务由提交 PMML 生成的分数。

### 3. importance 与压力类别均为硬契约

规范化特征元数据必须至少包含：

- `feature`：最终特征名；
- `category`：压力测试类别、数据源、产品或厂商；
- `importance`：训练方提供的数值重要性，允许为 `0`。

PMML 的全部可施压模型输入必须能精确匹配非空类别和数值 importance。平台先解析 PMML
外部必需输入与派生字段依赖，再建立模型特征到原始样本字段的可审计映射；缺失、重复
冲突、模糊匹配或无法解释的派生依赖均阻断任务。PMML 之外的额外元数据行保留在 Excel，
并标记为“非当前 PMML 入模字段”，但不进入压力场景。

### 4. 模型压力测试必做

压力测试不是可选报告章节。新任务必须存在可识别且非空的 OOT，并对每个特征类别完成
全量 OOT 重评分。执行错误、类别覆盖不全或缺少 OOT 会阻断任务完成；压测结果表现较差
属于模型风险结论，不等同于技术执行失败，仍应生成完整报告供人工判断。

### 5. 输出零退化

除旧一致性内容外，现有 Word、Excel 和 Agent 输出均为兼容基线。迁移采用“允许变化
白名单”，而不是重新挑选需要保留的内容。

## 用户可见流程

新任务使用以下阶段名称和顺序：

1. **材料与字段识别**
2. **PMML打分测试**
3. **模型效果与稳定性验证**
4. **模型压力测试**
5. **验证报告生成**

“PMML全量评分”统一改称“PMML打分测试”；“PMML压力测试”统一改称“模型压力测试”。
内部为了兼容历史任务可以暂时保留旧状态枚举或数据库状态，但 API、前端、Agent 消息、
失败提示和新报告不得继续使用旧阶段名。

以上是用户可见的逻辑阶段，不意味着增加新的人工确认次数。`模型压力测试`继续作为现有
指标执行中的独立可见子阶段，并进入现有 Agent 效果／稳定性总结和报告结论确认；除旧
一致性确认被 PMML打分测试替换外，手动模式和 Agent 模式的确认节奏保持不变。

## 材料与字段识别

### 1. Notebook 静态识别范围

Notebook 用于识别和解释以下信息：

- 目标字段及正负样本取值；
- train/test/OOT 划分字段及取值映射；
- 时间字段及粒度；
- 模型名称、版本和训练参数；
- PMML 输出字段提示；
- 字段别名、重命名和必要的派生字段逻辑；
- 数据字典／importance 在 Notebook 中的使用线索。

静态证据优先级固定为：

1. 明确的 `RMC_*` 字符串或字面量赋值；
2. 普通变量、DataFrame 赋值、重命名和模型调用的数据流；
3. Markdown、代码注释和保存的结构化输出；
4. Agent 基于前三类证据提出的候选；
5. 用户最终确认。

`RMC_SCORE_FN`、`RMC_SAMPLE_DF` 和 `RMC_ALGORITHM` 不再是新流程的执行契约。
`RMC_ALGORITHM` 存在时只作为提示，算法类型以 PMML 模型结构为准。
`RMC_FEATURE_IMPORTANCE_PATH` 存在时只帮助解释用户已选择的元数据文件，不能替代第四类
必传材料。

### 2. 结构化识别结果

字段识别必须输出确定性结构，而不是只保存一段 Agent 自由文本：

```text
FieldRecognitionResult
├── target_col
├── positive_label / negative_label
├── split_col
├── split_value_mapping
├── time_col / time_granularity
├── model_name / model_version
├── algorithm
├── model_params
├── pmml_output_field
├── per_field_evidence
│   ├── source_kind
│   ├── notebook_cell
│   ├── source_excerpt
│   └── confidence
└── confirmations
```

确定性解析器负责提取候选和证据。Agent 可以排序、解释和指出冲突，但不能在没有证据或
用户确认时直接写入最终字段契约。

模型参数优先从 `RMC_MODEL_PARAMS`、普通参数字典、估计器构造和训练调用中静态提取。
如果最终 Word 模板需要模型参数而静态证据仍不完整，报告准备阶段必须请用户确认或补充，
不能执行 Notebook 取得运行时对象，也不能用 PMML 能观察到的树数反推训练超参数。

### 3. Notebook 中生成、样本中不存在的字段

平台只允许重建明确、无副作用、可审计的派生字段，例如：

- 字段复制或重命名；
- 日期截取到月；
- 基于常量阈值的条件标签；
- 基于已有字段的固定 train/test/OOT 映射；
- 对来源文件追加固定来源标签。

支持的操作转换为声明式 transformation spec，展示样例值和分布后由用户确认。平台禁止
`eval`、执行任意 cell、数据库查询、网络访问、加载 Notebook 指向的外部模型或调用私有
函数。如果逻辑超出允许集合，任务暂停并要求用户上传已物化该字段的样本，不能静默猜测。

### 4. 算法和 PMML 输出字段

算法由 PMML 的模型节点和 mining function 确定。正类输出按以下顺序选择：

1. PMML 中明确的 target value 和 probability output；
2. Notebook 的 `RMC_PMML_OUTPUT_FIELD` 静态提示；
3. 唯一可识别的二分类正类概率字段；
4. 多个候选时由用户确认。

输出字段一经确认即写入任务契约，PMML打分测试和全部压力场景必须使用同一个字段。

## 特征元数据规范化

### 1. PMML 输入清单

平台在读取元数据前生成结构化输入清单：

```text
PmmlInputManifest
├── raw_required_fields
├── derived_fields
├── model_features
├── stress_units
│   ├── model_feature
│   ├── raw_input_fields
│   └── derivation_evidence
└── unsupported_derivations
```

`raw_required_fields`用于检查样本能否直接交给 PMML；`model_features`用于 importance 和
报告；`stress_units`用于将模型特征的类别映射到实际要置 `-9999` 的原始样本字段。
直接入模字段是一对一映射。派生字段只有在依赖关系明确、有限且不会产生类别冲突时才能
形成 stress unit；否则任务在压力测试前阻断，不能把派生字段名误当作样本列。

### 2. 输入兼容

元数据读取器继续支持当前数据字典的实际差异：

- CSV、XLS、XLSX；
- Excel 多 sheet；
- UTF-8、UTF-8-SIG、GB18030 等常见编码；
- `feature`、`特征名`、`特征名称`、`指标英文`、`feature_name` 等特征列别名；
- `category`、`类别`、`分类`、`数据源`、`来源`、`厂商名称`、`供应商` 等类别列别名；
- `importance`、`feature_importance`、`gain`、`权重` 等重要性列别名。

确定性别名映射优先。多个 sheet 或多列都可能匹配时，Agent 展示候选、覆盖率和冲突，
由用户确认，不能按文件顺序任意选择。

### 3. 精确覆盖规则

规范化后按 `PmmlInputManifest.model_features` 完整字段名精确匹配，不做前缀、后缀或
相似度自动匹配。
同一特征的重复行只有在 category 和 importance 一致时才能合并；任一冲突均阻断。

评分前必须满足：

- `raw_required_fields` 在样本中的字段覆盖率为 100%；
- 所有 `model_features` 都能形成无冲突的 `stress_units`；
- category 覆盖率为 100%；
- importance 覆盖率为 100%；
- importance 全部为有限数值；
- 不存在重复冲突。

完整规范化结果、来源 sheet、原列名、覆盖状态和额外特征均进入 Excel 与 Agent 证据。

## PMML打分测试

### 1. 评分语义

PMML打分测试对验证样本的每一行评分，不再抽样比较。通过条件为：

- 输入行数与输出行数完全一致；
- 每一行都成功取得指定输出字段；
- 分数全部为有限数值；
- 分数产物与任务、样本和 PMML 哈希绑定；
- 评分过程中没有被忽略的失败行。

任意一行评分失败、输出为空或非有限值，阶段失败并停止后续验证。这里检查的是 PMML 在
本次验证数据上的完整可执行性，不是与另一套分数的一致率。

### 2. 评分产物

```text
PmmlScoringResult
├── schema_version
├── pmml_sha256 / sample_sha256
├── engine / engine_version
├── output_field / score_direction
├── input_row_count / success_count / failure_count
├── null_count / non_finite_count
├── elapsed_seconds / rows_per_second / chunk_size
├── required_input_count / missing_inputs
├── score_artifact_path / score_artifact_sha256
├── status
└── bounded_errors
```

分数按平台生成的稳定位置行号写入 Parquet sidecar。后续逻辑通过该位置行号与原样本对齐，
不依赖用户 DataFrame index，也不把百万级逐行分数写入 JSON。

### 3. 百万级性能设计

当前 `PmmlScorer.score` 将 DataFrame 转为 records 后逐行调用 Java，绕过了 pypmml 已有的
DataFrame 批量入口。新默认后端使用现有 pypmml/PMML4S 的 DataFrame 分块批量评分：

- PMML 每个任务只加载一次；
- Python/JVM 每个 chunk 只跨边界一次，禁止逐行调用；
- 只投影 PMML 评分必需字段；
- chunk 大小按内存上限和列宽选择并记录；
- 输出直接追加到 Parquet sidecar；
- 同一任务的全量打分和所有压力场景复用同一 engine 与已加载模型；
- 评分缓存键为 PMML 哈希、样本哈希、输出字段、评分器版本和 transformation spec 哈希。

本地只读基准已经证明批量入口相对当前逐行实现有数量级提升，因此它作为首个稳定默认
后端，不引入容器和新的系统 Java 依赖。Windows 安装包继续使用已内置的私有 OpenJDK 17。

评分器边界保持后端可替换。编译型 JPMML/Transpiler 可以作为后续可选后端，但只有在
许可证审查、全部黄金模型语义回归、Windows 打包和百万级基准均通过后才能启用；本次迁移
不得为了追求未经验证的峰值速度，直接更换 PMML 语义引擎。

## 模型效果与稳定性验证

该阶段只改变分数来源，不改变现有计算口径或展示内容：

- 样本基本信息和逐月分布；
- train/test/OOT 的 KS、AUC、坏样本量、坏样本率和 lift；
- PSI 稳定性表；
- train/test/OOT 独立分箱；
- 逐月 KS、AUC、lift 和 PSI；
- train/test/OOT ROC-KS 曲线；
- 模型超参数；
- 全量特征重要性及 Top20。

所有指标继续由平台确定性代码计算。Agent 只解释结构化结果，不重算或编造指标。

## 模型压力测试

### 1. 基线与场景

- 基线使用 PMML打分测试已经保存的完整 OOT 分数，不重复计算。
- 每个 category 通过 `stress_units`取得其全部原始样本输入字段，在完整 OOT 中统一置为
  `-9999`。
- 每个类别使用与基线相同的 PMML、输出字段、评分后端和 chunk 策略重新全量打分。
- 继续计算现有 KS 变化、相对基线 PSI 和分类分箱。
- 不同类别的场景产物独立保存，便于失败重试和审计。

### 2. 必做门禁

新任务只有在以下条件全部满足时，模型压力测试才是 `completed`：

- PMML `model_features` 的类别与 stress unit 覆盖率均为 100%；
- OOT 非空；
- 每个非空类别都执行成功；
- 每个场景的输出行数与 OOT 完全一致；
- 每个场景分数全部为有限数值；
- 现有压力汇总和每类分箱产物均已生成。

旧任务的历史 `partial` 或 `failed` 结果继续只读展示。新任务不允许以 `partial` 作为成功
终态。需要区分“压力执行失败”和“压力结果暴露高风险”：前者阻断，后者完成计算并进入
Word、Excel 和 Agent 的审慎结论。

## Word、Excel 与 Agent 的零退化契约

### 1. 单一结果真源

Word、Excel、Web/Agent 不得分别重算或分别生成同一内容。报告阶段先形成：

```text
FinalValidationPresentation
├── validation_results
├── pmml_scoring_result
├── field_recognition_result
├── feature_metadata_resolution
├── final_report_values
├── metric_tables
└── rendered_images
```

三条输出链共同消费这份结构。Agent 确认或用户编辑后的最终报告文字必须先合并到
`final_report_values`，再原子地重新生成 Word 和最终 Excel，避免目前“最终文字只进入
Word、Excel仍是旧内容”的分叉。

### 2. Word

当前模板填充模式保持不变：

- 封面、修订信息、模型概述、适用范围和好坏样本定义保留；
- 样本周期、样本拆分、模型训练说明和超参数保留；
- Top20 importance、PSI、OOT KS 保留；
- train/test/OOT 的 ROC-KS 和分箱图表保留；
- 整体效果、逐月效果、模型压力测试及分类分箱保留；
- 压力建议、处置建议、监控建议和最终验证结论保留；
- 占位符替换继续继承原 run 的字体、字号、加粗、颜色等 `rPr`。

只允许将旧的 `reproducibility_summary` 语义替换为 PMML打分测试摘要，并删除最终结论中
“Notebook可复现／代码模型与PMML分数一致”的虚假陈述。为兼容客户旧模板，
`TEXT:reproducibility_summary` 可以作为 `TEXT:pmml_scoring_summary` 的 legacy alias，
但新模板和新 API 使用新名称。

### 3. Excel

现有 Excel 工作表、列、图表和条件格式全部保留，除“验证总览”中的旧一致性字段按新
语义替换。当前固定明细包括：

- 验证总览；
- 样本基本信息；
- 样本逐月分布；
- 模型超参；
- 特征重要性；
- 模型效果；
- PSI稳定性；
- ROC_KS曲线；
- train/test/OOT 三张分箱；
- 逐月效果；
- 压力测试汇总；
- 每个压力类别的分箱 sheet。

新增以下工作表，不替换现有明细：

- `PMML打分测试`：完整评分审计、性能、哈希、输出字段和错误统计；
- `字段识别`：Notebook 静态证据、候选、置信度和用户确认；
- `特征元数据覆盖`：PMML字段、类别、importance、来源和覆盖状态；
- `报告全文`：按最终 Word 顺序保存固定正文、填充文本、表格文本和图题；
- `报告图表`：嵌入 Word 中全部最终图表／图片；
- `报告内容索引`：将 Word 章节、TEXT/IMAGE/TABLE 占位符映射到 Excel sheet 和单元格区域。

每张 Word 图表在 Excel 中必须同时存在可查看图形和可计算底层数据。`报告全文`不是把
Word 页面截图粘贴进 Excel；它保存最终可检索文本。Excel 在 Agent 结论确认后重新生成，
确保其中的正文与最终 Word 完全一致。

### 4. Agent 对话与 Web 证据

以下现有能力全部保留：

- 材料识别、字段解释、失败诊断和自由追问；
- 样本总体与逐月分布图表；
- 整体与逐月效果／稳定性图表；
- train/test/OOT 三套分箱；
- Top20 importance；
- 模型压力测试 KS/PSI 与分类覆盖；
- train/test/OOT ROC-KS；
- 效果、稳定性和压力风险的阶段分析；
- 高／中／低风险分层与建议；
- Word 结论草稿、改写、确认和报告就绪消息；
- 手动模式与 Agent 模式；
- 当前任务证据、记忆引用和审计 metadata。

旧的“模型可复现性”Agent 阶段替换为“PMML打分测试”。其阶段证据包括全量评分完成率、
输入覆盖、输出字段、空值／非有限值、耗时、吞吐和评分器信息，不再展示两套分数差异。
效果稳定性和模型压力测试的 Agent 分析不得因新流程合并或缩短。

## 状态机、产物和兼容性

### 1. 新产物

新任务写入：

```text
field_recognition_result.json
feature_metadata_resolution.json
pmml_scoring_result.json
pmml_scores.parquet
validation_results.json
stress_scenario_scores/<category>.parquet
validation_metrics.xlsx
validation_report.docx
```

`validation_results.json` 的 canonical 字段由 `reproducibility` 改为 `pmml_scoring`。
新 reader 能读取历史 `reproducibility_result.json` 和旧 payload 以展示历史任务；新任务
不再写旧对比分数行，也不把新结果伪装成一致性结果。

### 2. 生命周期兼容

为减少数据库与任务恢复风险，现有通用 `TaskStatus` 可以在首版继续使用；内部
`EXECUTED` 对新任务表示“PMML打分测试完成”，不再表示 Notebook 已执行。所有用户可见
文案必须采用新语义。

新任务的终态门禁为：

- 四类材料和字段契约通过；
- PMML打分测试通过；
- 效果与稳定性产物完整；
- 模型压力测试完整执行；
- Word 和最终 Excel 原子生成；
- Word 无未解析的必填占位符。

模型指标较差或压力风险较高不会导致技术失败，而是进入报告结论和人工复核。技术阶段
未执行完整则不得显示验证成功。

### 3. 历史任务

- 已完成任务的原始产物不迁移、不重算。
- 历史任务继续显示原来的可复现性／一致性结论。
- 旧下载链接、报告文件名和只读 API 继续可用。
- 新前端根据结果 schema version 决定展示旧一致性或新 PMML打分测试。
- 新任务创建入口不再提供旧 Notebook 执行流程。

## 错误处理

必须提供结构化、可定位的错误：

- 四类材料缺失或用户未完成唯一选择；
- Notebook 字段识别存在多个冲突候选；
- 必需派生字段无法安全物化；
- PMML `raw_required_fields` 在样本中缺失；
- 派生模型特征无法形成无冲突的 stress unit；
- PMML 输出字段不明确或不存在；
- 元数据缺 category、importance 或存在冲突；
- PMML 任一行评分失败、为空或非有限值；
- OOT 缺失或为空；
- 任一压力类别评分失败或产物不完整；
- Word、Excel 或图表镜像生成失败。

错误详情保留有限行数的样例、总数、字段名、来源和修复建议，不能把百万级明细写进任务
消息。Agent 可以解释错误，但结构化失败状态和计数由平台产生。

## 安全与资源边界

- Notebook 全程只读解析，禁止 kernel 和任意代码执行。
- PMML 评分在本地受控 Java runtime 中执行，不访问网络。
- 输入路径经过 workspace 边界检查，产物原子写入任务目录。
- 大样本与压力场景使用分块读取和分块输出，内存上限不随总行数线性增长。
- PMML、样本、元数据和 transformation spec 全部记录哈希。
- 取消、超时或失败后清理临时 chunk，已提交产物不被半成品覆盖。
- Agent Memory 继续禁止保存原始样本、完整 Notebook、PMML 内容或未脱敏报告全文。

## 非退化测试策略

### 1. 允许变化白名单

新旧黄金任务的结构化快照比较中，只允许以下变化：

- `reproducibility`／分数一致性替换为 `pmml_scoring`／PMML打分测试；
- Notebook 执行步骤替换为静态字段识别步骤；
- 一致性相关的文案、图表、失败门和产物名；
- 新增 Excel 镜像和审计 sheet。

其余变化均视为回归，必须单独说明并取得新的产品决定。

### 2. 单元测试

- Notebook RMC literal、普通赋值、重命名和参数字典的静态识别。
- 不执行 Notebook、不访问 Notebook 引用的外部文件。
- 派生字段 allowlist 与拒绝任意代码执行。
- 字典列别名、编码、多 sheet、冲突和 100% 覆盖门。
- PMML input/output 识别和输出字段确认。
- pypmml DataFrame 批量评分与当前单行语义在现有 PMML fixtures 上一致。
- 分块前后行序、行数、空值和异常证据正确。
- 基线复用与每类完整 OOT 压力重评分。
- 新旧 validation result schema 的兼容读取。

### 3. 输出黄金测试

- Word 现有占位符、表格和图表清单除一致性项外保持不变。
- Word 替换后 run 级 `rPr` 完整继承。
- Excel 现有 sheet、表头、图表和条件格式全部存在。
- `报告全文`覆盖最终 Word 的全部正文和表格文本。
- 每个 Word IMAGE/TABLE 都能在`报告内容索引`中定位到 Excel 图表与底层数据。
- Agent metric section、table key、ROC-KS、importance、压力图表和结论确认流程保持不变。
- 手动模式与 Agent 模式输出同一确定性指标。
- 历史任务仍能展示和下载旧报告。

### 4. 端到端与性能测试

- LGB、XGB 各至少一个真实复杂 PMML 的完整新流程。
- CSV、Parquet、Feather 和不同编码元数据。
- 字段已落盘与允许派生字段两类样本。
- 百万级样本的分块评分、取消、失败恢复和缓存命中。
- 多类别完整 OOT 压力测试。
- 对比逐行基线，确认不存在逐行 Python/JVM 调用，记录加载耗时、热评分吞吐、峰值内存和
  各压力场景耗时。
- Windows 私有 OpenJDK 安装包 smoke 和实际评分测试。

## 文档迁移

实现时必须同步更新以下当前真源，不能只改代码：

- `docs/roadmap.md`：模型验证稳定能力改为 PMML-only 新口径；
- `docs/notebook_contract.md`：从运行契约改为只读字段识别契约；
- `docs/对notebook的要求.md`：删除执行环境和 `RMC_SCORE_FN` 必填要求；
- `docs/runbook.md`：更新任务流程、产物和失败处理；
- `AGENTS.md`：更新 V1 兼容边界，明确本设计是经批准的行为迁移；
- Agent system prompt、阶段消息和最终报告提示；
- API schema、前端标签、测试 fixture 和发布说明。

这是用户可见工作流和 Notebook 契约变化，应按 `docs/versioning.md` 形成明确的 V2 minor
发布节点；不得作为无说明的 patch 偷渡，也不得手工移动已发布 tag。

## 分阶段实施边界

实施计划应按以下依赖顺序拆分，任何中间阶段都不能让主线新任务处于半新半旧状态：

1. 建立新结果 schema、兼容 reader 和黄金非退化清单；
2. 实现 Notebook 静态字段识别和用户确认契约；
3. 实现特征元数据规范化与 100% 覆盖门；
4. 实现 pypmml 分块批量评分、Parquet sidecar 和缓存；
5. 将效果稳定性迁移到 PMML 分数；
6. 将必做模型压力测试迁移到同一 PMML scorer；
7. 迁移 Agent 阶段、前端证据和历史兼容展示；
8. 完成 Word/Excel 共享 presentation、报告全文和图表索引；
9. 更新文档并跑完整回归、百万级基准和 Windows 打包验证；
10. 以新 V2 minor 版本发布。

新任务切换到新流程时必须一次性具备上述完整能力。可以在开发分支分步提交，但不能在
稳定发布中暴露缺少压力测试、报告镜像或 Agent 图表的中间形态。

## 被拒绝的方案

### 为每个项目构建容器并继续执行 Notebook

容器只能固定公开包，无法自动补齐私有依赖、外部数据、数据库、绝对路径和历史状态，且
会把平台安装和客户交付复杂度推高，不能解决普遍复现问题。

### 使用 PKL 或样本已有分数做一致性验证

PKL 继续受 Python 和包版本约束；样本已有分数无法证明来源。两者都会重新引入环境或
可信度问题，与“PMML为唯一评分真源”的简化目标冲突。

### 执行 Notebook 只为了识别字段

字段识别不应获得任意代码执行权限。静态证据、声明式派生和用户确认足以建立审计契约，
复杂字段应由用户物化后重新提交。

### 让 Agent 猜 category 或 importance

压力测试和报告 importance 是确定性材料。Agent 可以解释候选列，不能创造训练重要性或
按字段名称猜供应商分类。

### 继续逐行 PMML 评分

百万级全量评分和多类别压力测试会将跨语言调用放大到不可接受的时间。默认实现必须使用
DataFrame 分块批量评分。

### 为简化流程删除现有报告或 Agent 图表

本次改造只解决一致性验证的环境不可复现问题。删除现有图表、结论、Excel 明细、Word
占位符或确认流程属于产品退化，不在授权范围内。

## 验收定义

新流程只有同时满足以下条件才算完成：

1. 用户仍选择 Notebook、样本、PMML、数据字典／特征元数据四类材料；
2. Notebook 未被执行，字段识别结果有来源证据和必要确认；
3. PMML 对完整样本 100% 成功评分；
4. 全部现有效果、稳定性、分箱、ROC-KS 和 importance 产物存在；
5. 全部类别在完整 OOT 上完成模型压力测试；
6. Word 保持原模板填充、图表和样式，仅一致性语义被替换；
7. Excel 保留全部原有内容，并完整镜像最终 Word 文本、图表和底层数据；
8. Agent 保留全部原有图表、阶段分析、风险分层、结论确认和自由追问；
9. 历史任务仍能按旧语义只读展示和下载；
10. 非退化黄金测试、完整验证测试、ruff、前端语法检查、`git diff --check`、百万级评分
    基准和 Windows PMML smoke 全部通过。
