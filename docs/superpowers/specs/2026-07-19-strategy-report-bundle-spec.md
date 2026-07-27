# Strategy Report Bundle 契约（V2.x）

## 文档状态

- 日期：2026-07-19
- 状态：V2.x 实施契约
- 适用范围：策略开发、策略回测、独立验证、策略影响测算和最终策略报告
- 上级计划：`../plans/2026-07-17-strategy-platform-gap-analysis-and-roadmap.md`
- 版本边界：本文全部能力在 V2.x 内交付，不迁移到 V3/V4

## 一、目标

MARVIS 需要把策略人员通常执行的完整过程固化为一条由自然语言驱动、由确定性 Tool 计算、由 Agent 编排的七步 Workflow：

1. 当前项目状况，包括风险、通过率和其他经营指标；
2. 历史版本策略及其效果；
3. 本次样本设计；
4. 单变量、单模型分析和效果评估；
5. 策略交叉组合、自建决策树、评分卡、Voting 等效果评估；
6. 策略影响测算，包括逐月效果、通过率影响和风险影响；
7. 形成可审计、可复现的最终文档。

最终报告不是独立填表功能，而是前六步结构化 evidence 的投影。所有数值、规则逻辑、代码和状态必须来自受信任 Tool、持久化 repository 或用户明确提供的信息；LLM 只负责自然语言理解、追问、编排、证据引用和总结，不计算或补造指标。

## 二、产品原则

### 2.1 Strategy-first：先把策略做对，再组装报告

- 报告渲染器不得成为数据处理、候选生成、回测、比较或验证的前置依赖。
- 缺少报告标题、背景文案、负责人、上线计划、外部历史说明等可选资料时，Agent 继续执行可逆的策略分析，并在报告阶段按本文空白规则处理。
- 缺少会改变策略决策语义或确定性结果的信息时必须 fail closed。Agent 不能为了让报告完整而使用技术默认值替代业务决定。
- 只有最终采纳、监控处置和生产变更需要相应人工决策或副作用授权；生成报告和导出本地产物属于可逆操作，不额外增加人工门。
- 报告生成失败不得回滚已经完成且可验证的策略 evidence。修复渲染问题后，应能从同一 evidence 幂等重建报告。

### 2.2 同一事实只有一个结构化来源

- 同一个 evaluator 驱动策略应用、回测、验证和代码交付。
- 报告不重新解释自由文本条件，不在 Excel 公式、前端或 LLM 中复制策略规则解释器。
- 每个数值均绑定 dataset/version/hash、strategy/artifact version、Tool run 和指标定义。
- 用户提供的报告补充信息保存在 task-scoped report context 和 audit 中，不把敏感项目资料或完整报告写入 Agent 长期记忆。

### 2.3 空白不等于零，不涉及不等于暂缺

报告字段的内部状态必须显式区分：

- `present`：已有可信值；
- `unavailable`：用户明确表示暂时没有，或经允许确认当前无法提供；
- `not_applicable`：本次策略客观上不涉及；
- `not_matured`：表现窗尚未成熟，当前不能形成该指标。

渲染规则：

- `unavailable` 的值单元格保持空白，不能显示 `0`、猜测值或技术默认值；
- `not_applicable` 可以显示“本次不涉及”；
- `not_matured` 的数值单元格保持空白，但证据/完整度区域必须保留成熟度状态和预计可用时间；
- Tool 运行错误、证据 hash 不一致或来源缺失不是 `unavailable`，报告生成必须失败并返回 typed error；
- 内部 completeness metadata 保留缺失原因，即使最终业务单元格按用户要求留空。

## 三、七步 Workflow 契约

七步是用户可理解的固定输出顺序，不要求底层严格串行。当前状况、历史版本和不依赖的描述分析可以并行执行；只要策略正确性依赖已经满足，报告可选信息不得阻塞策略主链。

| 步骤 | 标准输出 | 主要依赖 | 缺失处理 |
|---|---|---|---|
| 1. 当前项目状况 | `CurrentProjectSnapshot` | 当前数据、字段语义、指标定义 | 可选经营文案缺失不阻塞；缺目标或适用范围时阻塞完整策略开发 |
| 2. 历史版本效果 | `HistoricalStrategyReview[]` | MARVIS 策略版本、外部历史资料、监控 evidence | 外部历史资料可标记 `unavailable`；需要相对当前策略测算时 baseline 缺失会阻塞影响步骤 |
| 3. 样本设计 | `StrategySampleDesign` | dataset、标签、表现窗、纳排规则、切分规则 | 标签语义、样本边界等策略正确性信息缺失时阻塞 |
| 4. 单变量/模型 | `UnivariateEvidence[]`、`ModelEvidence[]` | 样本设计、feature/model Tool | 未使用模型时 `not_applicable`；没有 OOT 时不得生成 OOT 指标 |
| 5. 候选组合 | `CandidateEvidence[]`、`SelectedStrategyDesign` | Candidate Lab、Strategy DSL | 未采用的方法进入附件或 `not_applicable`；最终规则 DSL 不可为空 |
| 6. 影响测算 | `StrategyImpactAssessment` | baseline、候选策略、回测/验证数据、经济参数 | 不影响策略逻辑的经济参数缺失只阻塞相应影响表，不阻塞候选策略生成 |
| 7. 文档组装 | `StrategyReportBundle` | 前六步结构化 evidence | 可选字段空白；证据不一致或确定性结果缺失时 fail closed |

每一步必须记录：`status`、`input_refs`、`output_refs`、`producer_version`、`started_at`、`completed_at`、`red_flags` 和 `missing_information_refs`。任务刷新、进程重启和 Agent replan 后继续消费同一 task-scoped 状态，不能靠对话自由文本猜测上一步是否完成。

## 四、核心数据契约

以下是逻辑 schema；实现可使用 dataclass、Pydantic 或等价的版本化 JSON schema，但字段语义不可分叉。

### 4.1 `StrategyReportBundle`

```text
StrategyReportBundle
  schema_version
  report_id
  report_revision
  task_id
  strategy_id
  strategy_version
  strategy_type
  title: ReportField[str]
  status: draft | partial | final
  effect_stages[]
  sections[]
  dataset_refs[]
  strategy_artifact_refs[]
  tool_run_refs[]
  missing_information_refs[]
  completeness_summary
  generated_at
  producer_version
  content_sha256
  previous_report_id?
```

约束：

- `report_id` 对应不可变报告 artifact；补充资料后生成新 revision，不覆盖旧报告。
- 相同规范化输入、producer version 和 evidence 必须产生相同内容 hash 与同一幂等结果。
- `status=final` 只表示当前允许范围内的报告已完成，不自动表示策略已 OOT 验证、已采纳或已部署；这些状态必须独立展示。
- 当策略正确性 blocker 仍存在时，不得生成宣称策略完成的 `final` bundle；可以生成明确标记为探索性的 `partial` bundle。

### 4.2 `ReportField<T>`

```text
ReportField<T>
  value: T | null
  availability: present | unavailable | not_applicable | not_matured
  origin: tool_output | repository | uploaded_file | user
  source_refs[]
  as_of?
  blocking: none | strategy | impact | validation
  note?
```

约束：

- `availability=present` 时 `value` 不得为 `null`，并且至少有一个可信 `source_ref`；纯展示标题等平台字段可引用 task/report context。
- 其他三种状态的 `value` 必须为 `null`。
- 数值字段不得由 `origin=user` 覆盖已有确定性 Tool 结果。用户可提供原始参数或外部证据，平台仍需验证或明确标记为外部输入。
- LLM 生成的总结不是新的事实来源；总结中的数字必须反向引用对应 `MetricObservation`。

### 4.3 `MetricDefinition`

参考报告同时使用 `#FPD7`、`$FPD7`、`FPD30`、`MOB3`、`MOB6`、额度使用率、通过率、放款定价和年化风险，不能把风险口径写死为单个 `bad_rate`。

```text
MetricDefinition
  metric_definition_id
  metric_key
  display_name
  metric_family: volume | approval | drawdown | risk | pricing | exposure | cost | profit | stability
  basis: count | amount | balance
  numerator_definition
  denominator_definition
  label_semantics?
  performance_window?
  maturity_rule?
  aggregation: ratio | sum | mean | median | quantile | count
  direction: higher_is_worse | higher_is_better | neutral
  unit
  precision
  schema_version
```

约束：

- `basis=count`、`amount`、`balance` 必须是独立口径，不能从名称前缀猜测。
- 风险指标必须声明表现窗、成熟度规则、分子和分母。
- 无标签模式不创建坏率、KS、AUC 等定义；标签覆盖不足时显式红旗。
- 额度/定价和利润指标的期限、EAD、余额、资金/运营/数据成本口径必须版本化。

### 4.4 `MetricObservation`

```text
MetricObservation
  observation_id
  metric_definition_ref
  availability
  effect_stage
  dataset_ref
  strategy_ref?
  baseline_strategy_ref?
  period?
  segment?
  channel?
  decision_bucket?
  rule_id?
  numerator?
  denominator?
  value?
  sample_count
  label_coverage?
  amount_coverage?
  matured_count?
  scenario_ref?
  assumption_refs[]
  component_observation_refs[]
  calculation_definition_ref?
  tool_run_ref
  inputs_hash
```

同一个观察模型支持整体、逐月、分群、渠道、分群×月、waterfall、swap、树叶、评分带、Voting 命中数和 Cross group，避免每张表定义一套不可复用 payload。

派生指标不得只留下最终值：年化风险、利差、利润、加权风险和预计放款影响等 observation 必须引用场景、原始假设、组件 observation 和平台注册的确定性计算定义。`calculation_definition_ref` 只能指向版本化 Tool/公式定义，不能保存或执行 LLM 生成的自由公式。

### 4.5 `MissingInformationRecord`

```text
MissingInformationRecord
  missing_information_id
  task_id
  field_path
  reason
  blocking: strategy | impact | validation | report_optional
  question
  status: pending | provided | unavailable
  asked_count
  asked_at?
  answered_at?
  answer_source_ref?
  dependency_hash
```

行为约束：

1. Agent 先从 task input、dataset semantics、repository、上传材料和 Tool evidence 查找，不重复询问已经存在的信息。
2. `report_optional` 信息最多主动询问一次。
3. 用户回答“暂时没有”“无法提供”或等价表达后，记录 `status=unavailable`；同一 task 和同一 `dependency_hash` 下不得再次主动询问，报告对应值留空。
4. 用户之后主动补充时，可以把记录更新为 `provided`，重新生成新的报告 revision。
5. `strategy` blocker 即使用户回答暂缺也不能静默继续完整策略开发；只能保持阻塞，或由用户明确选择降级为不作业务结论的探索性分析。
6. `impact` blocker 只阻塞依赖该参数的影响表。例如缺单位数据成本不阻塞规则搜索，但数据成本测算留空。
7. `validation` blocker 不阻塞 development evidence，但禁止把策略标成 OOT validated。

### 4.6 七步核心对象公共契约

七步 Workflow 中的对象不能只作为文档名称存在。以下对象均必须有独立 `schema_version`、稳定 id、自认证内容 hash、task/dataset/strategy/tool refs、`producer_version`、availability/red flags 和生成状态；各 Tool 不得自行发明同名但不同义的 payload。

```text
CurrentProjectSnapshot
  schema_version / snapshot_id / content_hash
  task_id / as_of / scope
  dataset_refs[] / workspace_ref / champion_strategy_ref?
  metric_definition_refs[] / metric_observation_refs[]
  monthly_observation_refs[] / segment_observation_refs[]
  maturity_summary / user_context_fields[] / red_flags[]
  tool_run_refs[] / producer_version

HistoricalStrategyReview
  schema_version / review_id / content_hash
  strategy_ref / version / effective_period / asset_status
  scope / traffic_allocation?
  change_set: added_rule_refs[] / modified_rule_refs[] / removed_rule_refs[]
  observation_refs_by_effect_stage
  external_source_refs[] / decision_context_fields[]
  availability / red_flags[] / tool_run_refs[] / producer_version

StrategySampleDesign
  schema_version / sample_design_id / content_hash
  task_id / dataset_refs[] / workspace_ref
  risk_sample_definition / approval_sample_definition
  target_definition_ref? / performance_window? / observation_window?
  inclusion_rules[] / exclusion_rules[] / time_range / scope
  split_definition? / weight_definition? / month_field?
  amount_field_refs[] / maturity_rule?
  red_flags[]
  tool_run_refs[] / producer_version

StrategySampleDesignBundle
  schema_version / bundle_id / content_hash / producer_version
  sample_design / metric_definitions[] / metric_observations[]
  每个 MetricObservation 反向引用 sample_design_id/content_hash 与 dataset ref

CandidateEvidence
  schema_version / candidate_id / evidence_hash / candidate_type
  sample_design_ref / dataset_refs[] / workspace_ref
  generation_parameters / seed / search_budget / tie_break / truncated
  fragment_refs[] / metric_observation_refs[] / requirement_refs[]
  candidate_stage / observation_stage / validation_status
  artifact_refs[] / red_flags[] / tool_run_refs[] / producer_version

SelectedStrategyDesign
  schema_version / design_id / design_hash / strategy_type
  pool_ref / source_fragment_refs[] / ordered_rule_refs[]
  default_action / requirements[] / scope / traffic_allocation?
  candidate_stage / validation_status / producer_version

StrategyImpactAssessment
  schema_version / assessment_id / content_hash
  baseline_strategy_ref? / challenger_strategy_ref / scenario_refs[]
  sample_design_ref / metric_observation_refs[]
  waterfall_ref? / swap_ref? / monthly_refs[] / segment_refs[]
  conservation_checks[] / maturity_summary / red_flags[]
  tool_run_refs[] / producer_version
```

`StrategySampleDesignBundle` 是样本设计与其指标对象的 canonical 聚合边界。这里刻意不让
`StrategySampleDesign` 再正向持有 observation hash 列表：observation 已反向引用 sample design，
若双方 content hash 互相包含会形成不可计算的循环。Bundle 的 content hash 同时覆盖 sample
design、全部 definition 和 observation；下游通过 bundle artifact 的 registry hash 验证整组证据，
再使用其中的稳定对象 id/hash 引用。

`CandidateEvidence` 是公共 envelope；单变量、自动树、交互树、评分卡、Voting 和 Cross 仍各自拥有严格的类型专属 asset/validator，不能把公共 envelope 变成允许任意 JSON 的宽松候选格式。

### 4.7 场景、实验和数据成本契约

```text
ScenarioDefinition
  schema_version / scenario_id / content_hash
  name / purpose / effect_stage: estimated
  baseline_strategy_ref? / challenger_strategy_ref?
  assumptions[]: ReportField[typed value + unit + valid range]
  component_observation_refs[]
  calculation_definition_refs[]
  producer_version

ExperimentDesign
  schema_version / experiment_id / content_hash
  eligibility_definition_ref / randomization_unit
  allocation: treatment_ratio / control_ratio
  seed / assignment_artifact_ref
  treatment_policy_refs[] / control_policy_refs[]
  analysis_window / effect_estimator / guardrail_metric_refs[]
  status / producer_version

DataCostAssessment
  schema_version / assessment_id / content_hash
  scenario_ref? / currency / effective_period
  nodes[]
    node_id / provider / tool_or_data_source
    trigger_condition_ref / eligible_count / query_count / query_rate
    hit_or_funnel_rate? / unit_cost / cache_or_reuse_policy?
    total_cost / cost_per_application / cost_per_approval?
    cost_per_drawdown? / cost_per_loan_amount_unit?
    source_refs[]
  total_cost / conservation_checks[] / producer_version
```

实验分组必须由确定性 Tool 生成 assignment artifact；报告不能只写“实验组/对照组人数”而不记录随机单元、比例、seed 和分配证据。数据成本必须能表达“节点漏斗率 × 条件调用 × 数据源单价”，单价及有效期作为输入，不能隐藏在 Excel 公式或代码常量中。

## 五、策略正确性阻塞矩阵

以下字段在对应意图下属于策略正确性输入，不能作为报告可选资料留空：

| 信息 | 何时阻塞 | 允许的降级 |
|---|---|---|
| 业务目标和策略动作类型 | 完整策略开发 | 用户明确选择快速探索，只输出候选、不采纳 |
| 适用渠道、客群和策略范围 | 规则可能影响不同人群时 | 仅对已明确的数据集总体做探索 |
| target/好坏标签语义和表现窗 | 需要风险效果、分箱、模型或规则优选时 | 无标签模式只计算触发率，不产生风险指标 |
| 样本纳排边界和时间范围 | 需要形成可复现策略结论时 | 只做数据概览 |
| 分数/风险方向 | 使用分数阈值或评分带时 | 只输出方向诊断，不生成最终 cutoff |
| baseline 策略语义 | 请求策略调整、swap 或相对影响时 | 仍可分析候选绝对表现，但不输出相对影响 |
| 额度/定价动作、单位和边界 | limit/pricing 策略 | 只做风险分层，不生成额度/定价动作 |
| EAD/PD/期限/成本 | 利润或风险收益最优化时 | 只输出通过率/风险，不输出利润 |
| JOIN 键、基数和冲突语义 | 分析需要多数据源 JOIN 时 | 停止 JOIN，不能猜键或去重规则 |

## 六、报告 section 契约

### 6.1 当前项目状况

至少支持：

- 统计时点、产品、渠道、客群和当前 champion；
- 申请量、通过率、支用/放款量、件均额度、定价、额度使用率；
- 件数、金额和余额口径风险指标；
- 整体、逐月、渠道和重点客群趋势；
- 标签成熟度、样本数、坏样本数和覆盖率；
- 用户提供的业务背景、监管要求和调整目标。

### 6.2 历史版本策略效果

至少支持：

- 策略版本、有效期、资产状态、流量比例和适用范围；
- 新增、修改、下线规则 diff；
- 开发回测、OOT、上线后观察和监控结果分别展示；
- 版本切换、采纳、退役和 rollback evidence；
- 外部历史版本作为上传 artifact 导入，不把用户口述的历史数字伪装成平台实测。

### 6.3 样本设计

至少支持：

- 风险样本和通过率样本分别定义；
- 数据源、版本/hash、时间范围、渠道/客群、纳排规则；
- 标签、表现窗、观察窗、成熟度、金额/月字段和历史回溯打分；
- train/development/validation/OOT 切分和权重；
- 总样本、好坏样本、成熟样本、标签覆盖和金额覆盖；
- 数据泄漏、选择偏差、样本不足和未成熟红旗。

### 6.4 单变量和模型

单变量至少支持：分箱方法和切点、缺失箱、total/good/bad、占比、坏率、WOE、IV、Lift、KS/AUC、件数/金额口径、风险方向、逐月箱占比、均值、标准差、P25/P50/P75、PSI 和选择理由。

模型至少支持：model artifact/version、算法、特征、训练/验证/OOT 样本、KS/AUC/IV/系数或重要性、分数分布、校准、Lift、PSI、阈值、限制和 strategy bridge。未使用模型时整个模块标记 `not_applicable`。

### 6.5 候选组合和最终策略

统一支持：

- 单规则、自动树、交互树、标准 WOE-LR 评分卡、Voting/n-of-k、2D/3D 自动 Cross、2D matrix/cell 和人工规则；
- candidate artifact id/version、生成参数、搜索预算、随机种子、tie-break、截断状态和 lineage；
- 树节点/叶路径、评分卡 points、Voting 阈值/命中数、Cross cell/group；
- development、OOT 和逐月稳定性证据；
- 最终采用和未采用候选的理由；
- 策略级别、范围、流量、稳定 rule id、priority/first-match、默认动作，以及规则新增/修改/下线明细。

### 6.6 策略影响

至少支持：

- baseline/challenger 和场景假设；
- 逐规则级联 waterfall：命中数、命中率、成熟样本、坏样本、坏率、Lift；
- old×new swap：通过/拒绝/审核/额度/定价/分群换入换出；
- 整体、逐月、分群、渠道、分群×月；
- 件数、金额和余额口径；
- 通过率、风险、放款、件均、额度、定价、数据成本、收益和利润；
- 估计值、回测值、OOT 值和上线后观察值不能混在同一无标签列中。

## 七、效果证据阶段

canonical storage 和报告展示必须区分以下阶段：

| 存储值 | 报告标签 | 含义 |
|---|---|---|
| `estimated` | estimated / 预估 | 基于用户给定转化、成本或经营假设的情景测算，不是历史回放 |
| `backtested` | backtested / 开发回测 | 在 development 或历史样本上回放策略 |
| `oot_validated` | OOT | 在独立 validation/OOT 样本上应用冻结 artifact |
| `post_launch_observed` | post-launch / 上线后观察 | 来自有效版本上线后的真实监控或经营观测 |

约束：

- 同一指标可有多个阶段的 observation，但必须分别标记和对齐时间/样本。
- `estimated` 不得用“实际提升”“真实下降”等措辞。
- `backtested` 不得自动晋级 `oot_validated`。
- `post_launch_observed` 必须绑定环境、deployment ref、有效期和监控 evidence；本地采纳不等于上线。

## 八、模块化 Excel 和 artifact 结构

Sheet key 固定、显示名可配置。未使用的高级分析可以不生成附件 Sheet，但核心目录和证据索引必须稳定。

| Sheet key | 默认显示名 | 内容 |
|---|---|---|
| `00_summary` | 结论与变更摘要 | 背景、目标、核心变化、影响摘要、红旗和状态 |
| `01_current_state` | 项目现状 | 当前规模、通过率、风险和逐月/分群趋势 |
| `02_history` | 历史策略 | 版本、diff、历史效果和监控 |
| `03_sample` | 样本与口径 | 样本设计、字段、标签、成熟度和覆盖率 |
| `04_univariate_model` | 单变量与模型 | 单变量、模型和稳定性摘要 |
| `05_candidates` | 候选组合 | 树、评分卡、Voting、Cross 和候选比较 |
| `06_strategy` | 最终策略明细 | rule id、优先级、逻辑、动作、默认动作和变更类型 |
| `07_waterfall_swap` | Waterfall 与 Swap | 级联命中、换入换出和一致性 |
| `08_impact` | 逐月与分群影响 | 通过率、风险、金额、分群和逐月效果 |
| `09_economics` | 收益与数据成本 | 数据成本、收益、利润和假设 |
| `10_validation` | 验证与稳定性 | OOT、PSI、覆盖率、限制和红旗 |
| `11_evidence` | 证据与版本 | dataset/artifact/tool/hash、缺失信息状态和完整度 |
| `appendix_*` | 分析附件 | 变量趋势、树图、评分卡、Cross、Cap、代码和 equivalence |

交付 artifact 至少包括：

- 一份多 Sheet Excel 主报告；
- 一份机器可读 `StrategyReportBundle` JSON manifest；
- 一份 Markdown 执行摘要；
- 适用时附 Python/SQL/JSON 策略代码和逐行 equivalence report。

Excel 中的数值可以用公式链接同工作簿内的结构化明细，但不能把 LLM 输出写成计算公式。每个 Sheet 的 totals、waterfall 守恒、swap 守恒和指标覆盖率必须在导出前校验。

## 九、额度和定价扩展

`limit`、`pricing` 策略在公共 section 之外增加 `LimitPricingReportExtension`：

```text
LimitPricingReportExtension
  action_policies[]
    policy_id
    segment_or_rule_refs[]
    action_type: temporary_limit | permanent_limit | decrease_limit | price
    effective_days?
    calculation_definition_ref
    cap_definition_refs[]
    minimum_change? / maximum_change? / rounding_rule?
  current_limit_col
  proposed_limit_col
  price_col?
  cap_definition_refs[]
  segment_or_grade_refs[]
  eligible_count
  experiment_ref?
  average_change
  total_exposure_change
  utilization_observations[]
  application_or_drawdown_observations[]
  pricing_observations[]
  risk_observations[]
  annualized_risk_observations[]
  spread_or_margin_observations[]
```

必须支持：同一策略按不同客群同时使用临额和固额、有效期、提降额人数、户均变化、总敞口、T30 等申请/通过/支用表现、额度使用率、职业或资质 Cap、层级风险、年化风险和利差。没有 EAD、期限或成本口径时，相应经济表留空，不影响风险分层和规则候选生成。年化风险、利差和放款影响必须通过 `ScenarioDefinition`、组件 observation 和确定性计算定义保留完整依赖链。

## 十、Agent 追问和报告生成行为

### 10.1 追问优先级

1. 先询问 `strategy` blocker；一次最多聚合三个高相关问题。
2. 策略主链可运行后自动继续分析，不因报告可选字段停机。
3. 在报告组装前统一询问尚未提供的 `report_optional` 字段，一次问题中列清楚用途。
4. 用户明确说暂缺后写入 `unavailable` 并继续，不再在同一任务重复询问。
5. `impact` 或 `validation` blocker 只停止依赖部分，并明确报告哪些表会留空或哪些结论不能声明。

### 10.2 用户信息的使用边界

- 用户提供背景、业务目标、成本、期限、外部版本说明时，保存原始回答、来源 turn 和时间。
- 用户提供的原始参数进入确定性 Tool 前必须通过类型、单位和范围校验。
- 用户直接给出的历史结果如果无法复算，标记为外部提供，不和平台实测 observation 合并。
- 用户说暂缺时不生成替代措辞填入对应业务字段；报告值保持空白。

## 十一、验收测试

### 11.1 Contract 测试

- `StrategyReportBundle`、`ReportField`、`MetricDefinition`、`MetricObservation` 和 `MissingInformationRecord` schema 往返稳定，未知 enum 和旧 schema 漂移 fail closed。
- `present` 必须有非空值和来源；其他 availability 必须是 `null` 值。
- count/amount/balance、表现窗和成熟度定义不能通过显示名猜测或静默转换。
- 相同 evidence 重建得到相同 content hash；新增资料生成新 revision，不覆盖旧 artifact。

### 11.2 对话测试

- 可选信息只主动询问一次；用户回答“暂时没有”后，重启任务和 replan 均不再询问，对应报告单元格为空。
- 用户之后主动补充信息时，字段从 `unavailable` 变为 `present`，生成新报告 revision。
- 标签语义、样本边界或策略动作缺失时完整策略开发停止，不能用默认值绕过。
- 缺数据成本只使成本表留空，候选生成和风险回测继续。
- 缺 OOT 数据时 development evidence 可完成，但报告和生命周期都不能宣称 OOT validated。

### 11.3 确定性和证据测试

- 报告中的每个数字都能解析到 `MetricObservation` 和 Tool run；删除或篡改 evidence 后报告 fail closed。
- 无标签模式不产生坏率、KS、AUC 或 Lift。
- 未成熟月份的风险值为空，成熟度状态正确，不能假报 0 或绿灯。
- waterfall 每层与剩余样本守恒，swap 四象限/多动作矩阵与总体守恒，件数和金额 totals 与源数据对账。
- 页面、Excel、Markdown 和 JSON 对同一字段展示相同值和 effect stage。

### 11.4 报告渲染测试

- 核心 Sheet 顺序和 key 稳定，按需附件不会改变核心索引。
- `unavailable` 留空、`not_applicable` 显示“不涉及”、`not_matured` 值留空且证据区展示成熟度。
- 长规则、合并标题、表头、百分比、金额、日期和超链接在正常缩放下可读；不得裁切关键值。
- 公式错误扫描无 `#REF!`、`#DIV/0!`、`#VALUE!`、`#NAME?` 或意外 `#N/A`。
- 报告不包含客户级明细、API key、数据库凭证、原始 Notebook 源码或未脱敏敏感信息。

### 11.5 七步 E2E

至少覆盖以下旅程：

1. 用户用自然语言提出策略目标；
2. Agent 只澄清策略正确性 blocker；
3. 平台完成现状、样本、单变量/模型、候选组合和影响计算；
4. 用户对可选历史说明回答“暂时没有”；
5. Agent 不重复追问并继续到本地采纳门；
6. 人工确认采纳后生成 Excel、JSON、Markdown 和适用代码；
7. 报告中的历史说明为空，数值 evidence 完整，策略采纳状态与部署状态明确分离。

另外覆盖额度/定价旅程：缺成本时完成分层、Cap、风险和规则策略，经济表留空；补充成本后只重算依赖 observation 并生成新报告 revision。

## 十二、分阶段落地

- Phase 0A：`MissingInformationRecord`、追问一次、阻塞级别和 task-scoped report context。
- Phase 2：`MetricDefinition`、`MetricObservation`、当前项目快照、样本/指标口径和外部历史资料映射。
- Phase 3：候选 artifact 统一输出 report-ready evidence，不在报告层复制算法。
- Phase 4：`StrategyReportBundle`、模块化 Excel/JSON/Markdown、impact/waterfall/swap 和幂等 artifact。
- Phase 5：OOT evidence、成熟度、PSI 和验证 Sheet 追加，严格区分 effect stage。
- Phase 6：Manual/Agent 共用追问、状态、证据和报告生成 Workflow，完成浏览器/API E2E。
- Phase 7：把 post-launch monitoring observation 追加到新 revision，不能回写或覆盖原开发报告。

**实施进度（2026-07-19）**：Phase 4 已先交付可供报告复用的 `strategy.impact-assessment.v1` approval/reject Pool 影响证据，包含 first-match waterfall、总体/逐月动作与风险、标签/金额覆盖、可选基线件数/风险/金额 delta 和不可变 TaskArtifact。持久化 evidence 的真实性以 TaskArtifact registry 中的 expected content hash 为可信锚点；artifact 内 hash 只做 canonical 内容对账，不是离线签名。它完成的是第 6 步的首个确定性 evidence vertical，不等于 `StrategyReportBundle` 或最终 Excel/Markdown 已完成；分群×月、swap、OOT、limit/pricing/segmentation 专属影响表和最终七步组装继续留在 V2.x。

**实施进度（2026-07-22）**：Phase 2 已交付 `strategy.sample-design-bundle.v1` 首个确定性 evidence vertical。自然语言 Workflow 只抽取用户拥有的目标坏样本值（0/1）、表现窗、观察窗、成熟度、切分和可选业务字段；dataset/hash/workspace/semantic/target column 由平台绑定。Bundle 内含严格、版本化并逐对象 content-addressed 的 `StrategySampleDesign`、`MetricDefinition[]` 与 `MetricObservation[]`，覆盖 overall 及可选 development/validation/OOT 的件数、好坏样本、坏率、标签覆盖、金额覆盖/汇总与权重观测，并注册不可变 task-owned JSON。未声明的 validation/OOT 不生成假 0 observation；已绑定但全缺失的金额/权重汇总返回 `insufficient_data/null`，不把空白解释成零。该纵切不执行自由文本过滤；纳排必须先由 DataWorkspace 物化为派生数据集。表现窗或观察窗缺失、成熟度未确认时结果强制 `exploration_only / development / unvalidated`，依赖成熟度的风险指标返回 `unavailable/not_matured`，不能声称 OOT 或独立验证。当前风险样本与通过率样本仍共用活动数据集边界；双样本定义、渠道/客群纳排、历史回溯打分、泄漏/选择偏差/样本不足检测、下游 Candidate/Impact 的强引用以及最终报告渲染仍待后续纵切，因此本进度不等于第 3 步或七步报告已经完整完成。

## 十三、参考工作簿

以下工作簿仅用于内容模块、指标口径和业务表达参考。MARVIS 不复制其全局状态、固定布局或无法审计的公式；新实现必须使用统一契约、确定性 Tool 和结构化 provenance。

1. `/Users/eddyz/Downloads/业务学习/20260630-风险策略迭代评审文档模板.xlsx`
   - 参考内容：背景、设计思路、样本、策略级别、规则变更、waterfall、swap、通过率/风险影响和数据成本。
2. `/Users/eddyz/Downloads/业务学习/云闪付&存量经营复借策略调整-20260422(1).xlsx`
   - 参考内容：当前项目逐月表现、件数/金额风险口径、多头变量分箱占比、均值/分位数趋势、长表现分箱和规则效果。
3. `/Users/eddyz/Downloads/业务学习/20260609-自营中信借钱贷中提额方案.xlsx`
   - 参考内容：额度策略目标、当前规模和 T30 行为、分层、策略瀑布、同一方案内的临额/固额组合、随机实验/对照分配、额度 Cap、年化风险、利差和放款影响。

参考工作簿中不得照抄的做法：示例数字或权重写死在公式中、把多项数据费用合并成不可追溯常量、用 `#/$` 前缀猜件数/金额口径、混列 estimated/backtested/OOT/post-launch、超宽千行附件、重复总结入口，以及把 Excel serial date 直接作为图表横轴。MARVIS 应以结构化明细附件、稳定 Sheet key、`YYYY-MM` 时间标签、业务名+原字段双标签和 evidence drill-down 替代这些设计。
