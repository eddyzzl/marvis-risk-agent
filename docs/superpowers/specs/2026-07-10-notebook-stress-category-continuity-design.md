# Notebook 压测分类连续性设计

**日期：** 2026-07-10

**状态：** 待用户复核

**范围：** V2.x 模型验证工作流第 3 步“模型效果&稳定性验证”

## 背景

当前模型验证工作流会在 Notebook 执行完成后追加平台 cell，使用内存中的
`RMC_SAMPLE_DF` 和 `RMC_SCORE_FN` 计算模型效果、稳定性和分类压力测试。
默认连续流水线已经在同一次隔离 Notebook 执行中完成这些工作，但压力测试的
分类生产端和平台结果消费端仍会各自重新读取磁盘中的原始特征字典。

任务 `afa268ab72c841babcf6a12ed5e34668` 暴露了这个断点：Notebook 将原始字典中的
`BH_A044` 转换为模型实际使用的 `BH_A044_C0580`，并通过
`RMC_FEATURE_IMPORTANCE` 正确输出类别“睿智”；压力测试随后重新读取未转换的原始
字典并进行精确匹配，因而在执行前静默删除了整个“睿智”类别。最终任务仍显示
`completed`，但没有评估模型最重要的数据源。

## 目标

- 第 3 步承接 Notebook 已形成的特征名和类别语义，不绕过 Notebook 契约重新推断。
- 压测场景生产端与结果消费端使用同一份规范化“模型特征—类别”映射。
- 保留隔离执行、资源回收和可重复重试能力，不恢复跨任务长期保活 kernel。
- 类别覆盖不完整时给出结构化证据，禁止静默标记为完整完成。
- 不引入针对 `_C0580` 或 `BH_` 的项目特例和模糊前缀匹配。

## 非目标

- 不改变 `-9999` 分类剔除压力场景。
- 不改变 KS、PSI、分箱和模型分计算方法。
- 不要求验证人员理解或手工修正 Notebook 内部变量。
- 不把 Notebook 中任意命名的 `var_dict` 等局部实现变量纳入平台接口。
- 不恢复已禁用的 legacy live Notebook 执行模式。

## 设计决策

### 1. Notebook 契约结果优先

平台以 Notebook 顶层公开的 `RMC_FEATURE_IMPORTANCE` 作为模型特征分类的第一来源。
当该 DataFrame 包含 `category` 或 `类别` 时，非空类别直接对应其同一行的最终
`feature` 名称。这些名称已经经历 Notebook 自身的重命名、后缀补充或特征工程，
因此不得再被原始磁盘字典覆盖。

磁盘特征字典仅补充 `RMC_FEATURE_IMPORTANCE` 中类别为空的特征，并且只允许按完整
特征名精确匹配。平台不猜测后缀、不进行前缀匹配，也不读取未声明的 Notebook
变量。

解析优先级固定为：

1. `RMC_FEATURE_IMPORTANCE.feature + category/类别`；
2. 对类别为空的模型特征，用磁盘特征字典精确补充；
3. 仍无法匹配的特征进入 `unclassified_features`。

如果同一特征在同一来源内出现相互冲突的非空类别，解析失败并输出冲突明细，
不能按行顺序任意覆盖。

### 2. 建立唯一分类解析模块

新增一个深模块负责分类解析，外部接口只接收 Notebook 特征重要性和可选磁盘字典，
返回一个结构化解析结果：

```text
FeatureCategoryResolution
├── per_category: {category: [feature, ...]}
├── unclassified_features: [feature, ...]
├── conflicts: [{feature, categories, source}]
└── source_counts: {notebook: N, dictionary: N, unresolved: N}
```

Notebook 追加 cell 和平台结果生成代码都必须通过这个接口解析分类，不能各自复制
分组逻辑。这样分类规则的修改、测试和审计只集中在一个 seam。

### 3. 场景产物携带解析证据

`stress_scenario_scores.json` 升级 schema，除每个类别的置缺特征和模型分外，同时保存：

- 最终 `per_category` 映射；
- `unclassified_features`；
- 分类来源计数；
- 分类冲突或解析错误；
- 用于关联 Notebook 本次运行的必要 schema/version 信息。

平台结果阶段直接消费并验证这份映射，不再重新从磁盘字典生成另一份类别列表。
如产物中的类别、置缺特征或模型元数据互相矛盾，第三步失败并明确报告产物不一致。

### 4. 覆盖状态不可静默

压力测试整体状态按分类覆盖情况确定：

- 所有模型特征均已分类，且所有类别场景成功：`completed`；
- 至少一个类别成功，但仍有未分类特征或部分场景失败：`partial`；
- 没有任何模型特征可分类、分类存在冲突，或场景产物不一致：`failed`。

`validation_results.json`、Web、Excel 和 Word 必须展示未分类特征数量。可展开证据中
保留完整特征名；摘要展示受控数量并给出总数。不能仅因为已解析的三个类别执行成功，
就将整体状态标记为 `completed`。

### 5. 执行与重跑语义

默认完整流水线继续采用当前隔离执行方式：原 Notebook cell 执行完成后，在同一次
Notebook 执行中追加契约检查、复现性和第 3 步指标 cell，直接使用
`RMC_SAMPLE_DF`、`RMC_SCORE_FN` 和 `RMC_FEATURE_IMPORTANCE`。

用户单独选择“重新执行第三步”时，不依赖可能已经退出或污染的旧 kernel。平台重新
执行原 Notebook 以确定性重建这些契约对象，然后在同一次重建执行末尾追加第 3 步。
这属于重建 Notebook 状态，不是由平台绕过 Notebook 后重新解释原始样本或分类语义。

## 数据流

```text
原 Notebook
  -> RMC_SAMPLE_DF
  -> RMC_SCORE_FN
  -> RMC_FEATURE_IMPORTANCE（最终 feature，可含 category/类别）
       |
       v
统一分类解析模块 <- 磁盘特征字典（仅补充空类别）
       |
       +-> FeatureCategoryResolution
       |
       v
同一 Notebook 执行中的压测追加 cell
  -> stress_scenario_scores.json（分数 + 分类解析证据）
       |
       v
平台确定性指标计算
  -> validation_results.json / Excel / Word / Web
```

## 兼容性

- 现有只包含 `feature`、`importance` 的 `RMC_FEATURE_IMPORTANCE` 继续可用，类别由磁盘
  字典精确补充。
- 现有包含 `category` 或 `类别` 的 Notebook 无需修改，并自动获得最高优先级。
- 旧版 `stress_scenario_scores.json` 仅允许在能证明分类覆盖完整时读取；否则重新生成
  第 3 步产物，不能默默降级。
- `RMC_SAMPLE_DF`、`RMC_TARGET_COL`、`RMC_ALGORITHM`、`RMC_SCORE_FN` 等现有 V1
  兼容契约保持不变。

## 错误与审计

新增错误信息必须直接指出失败位置和证据，例如：

- `stress category conflict for BH_A044_C0580: 睿智, 其他`；
- `stress category coverage is partial: 6/39 model features are unclassified`；
- `stress scenario artifact category mapping does not match model metadata`。

分类解析结果和产物验证结果属于确定性平台证据，不由 LLM 推断。Agent 只能基于这些
字段解释覆盖情况，不能把缺少类别描述为“自然结果”或猜测前缀规则。

## 测试与验收

### 单元测试

- Notebook 类别优先于磁盘字典。
- Notebook 空类别可由磁盘精确补充。
- `BH_A044_C0580` 不会与磁盘 `BH_A044` 模糊匹配。
- 同一来源内的类别冲突会失败。
- 未分类特征被结构化保留并使整体状态为 `partial` 或 `failed`。

### 流水线测试

- 回归任务形态包含 `BH_A044_C0580 -> 睿智` 时，场景产物和最终结果都包含“睿智”。
- 默认完整流水线只执行一次原 Notebook，并在其后追加第三步。
- 单独重跑第三步会重建 Notebook 状态，再追加第三步，不读取旧 kernel 状态。
- 平台消费场景产物时不会用磁盘字典重建并覆盖分类映射。
- Web、Excel、Word 与 JSON 对 `completed`、`partial`、`failed` 的表达一致。

### 验收标准

使用本次问题任务的等价材料运行后：

- “睿智”出现在 `stress_test.per_category`；
- `BH_A044_C0580`、`BH_A055_C0580` 等睿智模型特征进入对应置缺列表；
- 任务不再因原始字典缺少 `_C0580` 后缀而漏测；
- 若人为移除一个模型特征的类别，整体状态不得显示 `completed`；
- 最小相关测试、完整验证测试、ruff、`git diff --check` 均通过。

## 被拒绝的方案

### 为 `BH_` 自动追加 `_C0580`

该方案只修复一个供应商命名规则，会把业务特例写入平台确定性逻辑，未来遇到其他
后缀或重命名仍会失败。

### 前缀或模糊匹配

模糊匹配可能把一个原始字段错误映射到多个派生特征，无法满足可审计和确定性要求。

### 恢复跨步骤长期保活 kernel

长期保活会重新引入进程生命周期、应用重启、取消、内存上限和旧状态污染问题。
当前隔离模式已经支持在一次执行中追加第三步；本次缺陷属于数据 seam 错位，不需要
用持久 kernel 修复。
