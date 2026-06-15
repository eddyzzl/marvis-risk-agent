# MARVIS 全能信贷风控 Agent 平台 — 总架构蓝图

## 文档状态

- 状态：待实施规划（governing blueprint，统辖后续所有 Phase 函数级 spec）
- 日期：2026-06-13
- 适用版本线：V2 → V4（V1.1.x 为当前稳定基线）
- 关联文档：`docs/roadmap.md`（产品阶段与术语唯一来源）、`AGENTS.md`（工程约定）、`DESIGN.md`（体验约束）、`CODE_REVIEW_2026-06-13.md`（地基阶段债务审计附件；Phase 0 已内联必须修复项）
- 本蓝图本身不含实现代码；每个 Phase 的函数级 spec 是独立文件。

## 0. 文档地图与阅读顺序

```text
docs/superpowers/specs/
  2026-06-13-marvis-platform-blueprint.md      ← 本文件（先读）
  2026-06-13-phase-0-foundation.md             ← 函数级：地基（修债 + 前端拆分）
  2026-06-13-phase-1-plugin-runtime.md         ← 函数级：Plugin/Tool Runtime
  2026-06-13-phase-2-orchestration.md          ← 函数级：编排 Harness
  2026-06-13-phase-2b-adaptive-loop.md         ← 函数级：自适应 replan / explore loop
  2026-06-13-phase-2c-v1-compat.md             ← 函数级：V1 验证能力包装成 v1_compat
  2026-06-13-phase-3-data-layer.md             ← 函数级：数据层 + 数据文件处理包
  2026-06-13-phase-4-feature-pack.md           ← 函数级：特征分析 + 处理包
  2026-06-13-phase-4v-vintage-core.md           ← 函数级：vintage / roll-rate 确定性核心
  2026-06-13-phase-5-memory-evolution.md       ← 函数级：记忆自进化
  2026-06-13-phase-6-modeling-pack.md          ← 函数级：建模能力包
  2026-06-13-phase-7-strategy-pack.md          ← 函数级：策略与组合能力包
  2026-06-13-phase-8-draft-zone.md             ← 函数级：联网学习草稿区
  2026-06-13-phase-frontend-v2.md              ← 函数级：V2 前端操作面
```

阅读顺序：本蓝图第 1~14 节 → 对应 Phase spec。任何实现者动手前必须读完本蓝图的「全局不变量」（第 2 节）与「全局工程约定」（第 14 节）。

---

## 1. 愿景、能力边界与离线/联网契约

### 1.1 愿景

把 MARVIS 从「模型验证工具」升级为「**本地优先、可治理、可自进化的全能信贷风控 Agent 平台**」：用户用自然语言描述目标（"帮我把这几个表拼起来做一个贷前 A 卡"），平台理解意图、拆分任务、编排预置工具、必要时派子 Agent、自我审查，产出可审计的结果与报告。

### 1.2 五大能力与落地方式

| 用户诉求 | 落地子系统 | 关键原则 |
|---------|-----------|---------|
| 1. 有记忆、能自我总结进化 | `agent_memory/` + 自进化层（Phase 5） | 记忆只辅助，不改确定性指标 |
| 2. 可插拔 plugin/tool/skill | `plugins/` Runtime（Phase 1）+ 用户可编写 Workflow 模板（Phase 2，= skill） | 安装即信任 + 子进程隔离；skill 是声明式模板，只编排已信任工具，不另立 runtime |
| 3. 编排/拆任务/派子 agent/自审 | `orchestrator/` Harness（Phase 2） | 模板优先 + 受约束规划 + 确定性闸门 |
| 4. 信贷全场景预置能力 | `packs/*` 能力包（Phase 3~7） | 工具预置注册，不临场生成 |
| 5. 不会的联网学习写脚本 | `drafts/` 草稿区（Phase 8） | 草稿区运行 + 人工转正才入库 |

### 1.3 离线/联网契约（硬边界）

> **离线自洽原则**：平台在**完全无网**环境下，能力 = 已注册工具库的并集。缺一个工具就是真的缺一个，平台**不会**临场上网补，也**不会**让 LLM 现场生成业务计算代码冒充工具。

- **离线下可用**：所有已注册 Plugin/Tool、所有 Workflow 模板、记忆检索、报告产出。
- **仅联网可用（能力扩展通道）**：`drafts/` 草稿区的"联网搜索 → 学习 → 写脚本"。产出的脚本进草稿区，**人在有网环境产出、上传到平台、人工确认后转正**为正式 tool。
- **部署现实**：大部分时间无网。因此工具库的**预置完备度**是产品的核心竞争力，联网学习是补充而非依赖。

### 1.4 非目标（V2 首发不做）

- 不做拒绝推断（reject inference）—— 方法论复杂，放 Phase 6 蓝图并标注「需方法论评审」。
- 不做实时在线评分服务 / 模型部署上线。
- 不做多租户登录鉴权（数据模型预留 `owner` 字段，但 V2 单机单用户）。
- 不做分布式 / 集群训练。
- 不引入外部向量库或云记忆 provider。

---

## 2. 设计哲学与全局不变量

### 2.1 设计哲学：强结构化、弱自由发挥

目标 LLM 档位为 **32B~72B**（如 Qwen3-32B/72B、DeepSeek 蒸馏版）。该档位模型「意图理解、文案、解释」可用，但「长链路自由规划」不可靠。因此：

- **harness 扛重活**：任务拆分有结构化 `Plan` DAG 约束，不是放任模型自由 loop。
- **高频走模板**：常见任务（模型验证、拼表、特征分析）走预置 Workflow 模板，模型只填参数。
- **新任务受约束生成**：模型生成 Plan 必须过 `PlanValidator` schema 校验 + 工具可用性检查 + 人工确认。
- **确定性优先于 LLM 自评**：每步执行后先过确定性后置检查（硬闸门），LLM critic 是第二层软审查。
- **能力可伸缩**：换更强模型时调高 `autonomy_level` 即可放宽约束，harness 不重写。

### 2.2 全局不变量（任何代码不得违反）

```text
INV-1  确定性指标只由平台 tool 代码计算，LLM 永不计算。
       KS/AUC/PSI/IV/Lift/WOE/vintage/回测/分数一致性等所有数字，
       必须来自 tool 的结构化 output；LLM 拿到的是算好的结果，只做编排和解释。

INV-2  LLM 不得编造平台未提供的数据。Agent 可解释、总结、起草、规划、请求确认，
       但产出里的每个业务数字都必须可追溯到某个 tool output 或平台证据。

INV-3  数据集拼接（join）不得静默执行。任何 join 必须产出结构化 JoinPlan +
       诊断（行数/命中率/fan-out/唯一性）+ 人工确认，膨胀或命中率异常必须告警。

INV-4  记忆不得改变确定性验证结果。记忆只辅助解释、参数建议、风险提醒、历史对比、报告口径。

INV-5  禁存敏感内容：原始样本、客户明细、完整 Notebook 源码、PMML/模型文件内容、
       API key、数据库连接、未脱敏报告全文、机构敏感信息。

INV-6  Plugin/Tool 执行在子进程隔离，超时/崩溃/资源超限不得拖垮主服务。

INV-7  联网学习产出的脚本默认进草稿区，人工转正前不得作为正式 tool 被 Planner 自动选用。

INV-8  所有 Tool 执行、记忆读写、计划决策、子 agent 派发必须留审计记录（who/when/inputs hash/outcome）。

INV-9  跨平台：所有路径用 pathlib + as_posix() 注入，子进程用显式编码，
       不假设 POSIX-only（参考 codex/windows-create-task 教训）。

INV-10 模块边界（第 3 节）是硬约束：validation/ 不依赖 DB/FastAPI；
       output/ 不执行上传；packs/ 通过 Tool 契约暴露，不被 api.py 直接 import 业务逻辑。
```

每个 Phase spec 的每个函数都要标注它**捍卫了哪条不变量**或**受哪条约束**。

---

## 3. 模块架构与边界

```text
┌────────────────────────────────────────────────────────────────────┐
│ 前端  marvis/static/  (ES Modules, 无构建, 离线)          │
│   js/core/      状态、事件总线、API client、轮询管理                  │
│   js/views/     任务树、子agent状态、plugin管理、workflow图、工件预览  │
│   js/render/    指标表、Agent对话、Markdown                          │
├────────────────────────────────────────────────────────────────────┤
│ API 层  api.py + routers/                                            │
│   不承载验证算法、不承载编排逻辑细节；只做 HTTP/任务生命周期/SSE       │
├────────────────────────────────────────────────────────────────────┤
│ 编排层  orchestrator/        ◄══ V2 核心                              │
│   intent.py      IntentRouter：意图分类、模板命中                    │
│   planner.py     Planner：模板填参 / 受约束 DAG 生成                  │
│   plan.py        Plan/PlanStep 数据契约 + PlanValidator              │
│   executor.py    PlanExecutor：DAG 执行、依赖调度                    │
│   subagent.py    SubAgentDispatcher：派发 scoped 子 agent            │
│   reviewer.py    Reviewer：确定性后置检查 + LLM critic               │
│   harness_state.py  HarnessState：可持久化状态机                     │
├──────────────┬──────────────┬───────────────┬──────────────────────┤
│ plugins/     │ agent_memory/ │ validation/   │ output/              │
│ Tool Runtime │ 记忆+自进化   │ V1 确定性算法  │ Excel/Word/图        │
│ (子进程)     │              │ (复用)        │ (复用)               │
├──────────────┴──────────────┴───────────────┴──────────────────────┤
│ 数据层  data/                ◄══ 用户重点                            │
│   registry.py    DatasetRegistry：数据集登记、版本、schema 缓存      │
│   schema_infer.py  列类型 + 语义角色推断                            │
│   fingerprint.py   列值指纹（解决 raw vs md5）                       │
│   align.py       ColumnAligner：键字典 + 模糊匹配兜底               │
│   join_engine.py JoinEngine：JoinPlan 生成、诊断、执行              │
│   excel_ingest.py  多 sheet + 合并表头拍平                          │
│   sampler.py     采样；profiler.py 列画像                          │
│   backend.py     DuckDB/pandas 后端抽象（十万~千万行）              │
├──────────────┬──────────────────────────────────────────────────────┤
│ packs/       │ 能力包（每个是一个内置 Plugin）                       │
│   data_ops/  │ 数据文件处理包（Phase 3）                            │
│   feature/   │ 特征分析+处理包（Phase 4）                           │
│   modeling/  │ 模型开发包（Phase 6 蓝图）                           │
│   strategy/  │ vintage/盈利/策略回测包（Phase 7 蓝图）              │
├──────────────┴──────────────────────────────────────────────────────┤
│ drafts/      │ 联网学习草稿区 + 转正治理（Phase 8 蓝图）             │
├─────────────────────────────────────────────────────────────────────┤
│ 持久层  db.py  SQLite: tasks·jobs·plans·plan_steps·agents·plugins·   │
│                tools·memory·datasets·joins·audit·drafts              │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 模块边界规则（扩展 INV-10）

- `validation/`：纯确定性算法，不依赖 DB/FastAPI/Agent/Plugin runtime。能力包通过 import validation 函数复用，但 validation 不反向依赖。
- `data/`：数据集/schema/join 的基础设施，不依赖 orchestrator/plugins；packs 和 orchestrator 依赖 data。
- `packs/*`：每个能力包是一个**内置 Plugin**，通过 Tool 契约（input/output schema）暴露能力。api.py 和 orchestrator **只通过 Tool 契约调用**，不 import 包内部实现细节。
- `orchestrator/`：依赖 plugins（拿 tool registry）、data、agent_memory、validation；被 api.py 调用。
- `api.py`：只做 HTTP/任务/SSE，业务编排委托给 orchestrator，验证算法委托给 validation/packs。**过拟合检测等现存于 api.py 的算法逐步迁出**（见 CODE_REVIEW P2-27）。

---

## 4. 核心数据契约

所有契约用 `@dataclass`（与现有 `validation/results.py` 风格一致），持久化映射到 SQLite。字段标注 `[必填]/[可选]`。

### 4.1 编排契约

```text
Plan
  id: str                      [必填] uuid
  task_id: str                 [必填] 关联任务
  goal: str                    [必填] 用户目标自然语言
  source: "template"|"generated"  [必填] 来源
  template_id: str|None        [可选] 命中的模板
  steps: list[PlanStep]        [必填]
  autonomy_level: int          [必填] 0=全人工确认 1=关键步确认 2=仅终审确认
  status: PlanStatus           [必填] DRAFT|VALIDATED|CONFIRMED|RUNNING|REVIEW|DONE|FAILED|CANCELLED
  created_at/updated_at: str   [必填]

PlanStep
  id: str                      [必填]
  plan_id: str                 [必填]
  index: int                   [必填] 拓扑序展示用
  title: str                   [必填] 人类可读
  tool_ref: ToolRef            [必填] 该步调用的工具
  inputs: dict                 [必填] 已校验的入参（引用上游 output_ref）
  depends_on: list[str]        [必填] 前置 step id
  post_checks: list[PostCheck] [必填] 确定性后置检查（硬闸门）
  needs_confirmation: bool     [必填] 是否需人工确认才执行
  sub_agent_id: str|None       [可选] 若派子 agent
  status: StepStatus           [必填] PENDING|BLOCKED|AWAITING_CONFIRM|RUNNING|CHECKING|DONE|FAILED|SKIPPED
  output_ref: str|None         [可选] 产物句柄（指向 artifact/dataset/metrics）
  review_verdict: ReviewVerdict|None [可选]
  error: str|None              [可选]

ToolRef       { plugin: str, tool: str, version: str }

PostCheck                     ◄ INV-1/INV-3 的执行点
  kind: "schema"|"range"|"rowcount"|"invariant"|"nonempty"|"match_rate"
  spec: dict                   例：{"field":"ks","min":0,"max":1}
                                   {"rule":"joined_rows<=anchor_rows"}

ReviewVerdict
  reviewer: "deterministic"|"llm_critic"
  passed: bool
  reasons: list[str]
  at: str

SubAgent
  id: str
  parent_task_id: str
  parent_step_id: str|None
  scope: str                   子 agent 的受限目标描述
  granted_tools: list[ToolRef] 只授予完成 scope 所需的工具
  context_budget: int          注入上下文 token 上限
  status: AgentStatus          SPAWNED|RUNNING|RETURNED|FAILED|KILLED
  result_ref: str|None
```

### 4.2 Plugin/Tool 契约

```text
PluginManifest
  name: str                    [必填] 唯一
  version: str                 [必填] semver
  display_name: str            [必填]
  description: str             [必填]
  builtin: bool                [必填] 内置包 vs 上传插件
  tools: list[ToolSpec]        [必填]
  hooks: list[HookSpec]        [可选]
  permissions: list[str]       [必填] 仅审计展示（INV-6 下不强制跨进程拦截）
  entrypoint: str              [必填] 模块路径
  python_requires: str         [可选]
  checksum: str                [平台安装时计算并持久化；上传 manifest 中的 checksum 不可信]

ToolSpec
  name: str                    [必填]
  summary: str                 [必填] 给 Planner 选工具用的一句话
  input_schema: dict           [必填] JSON Schema
  output_schema: dict          [必填] JSON Schema（确定性结构化）
  determinism: "deterministic"|"stochastic"  [必填] stochastic 须声明 seed 入参
  timeout_seconds: int         [必填]
  failure_policy: "fail"|"retry"|"skip"  [必填]
  side_effects: list[str]      [必填] 读/写哪些资源，审计用

HookSpec
  event: str                   平台事件名（见第 5.5 节）
  tool: str                    触发的本插件 tool
```

### 4.3 数据层契约

```text
Dataset
  id: str                      [必填] uuid
  task_id: str                 [必填]
  role: "sample"|"feature"|"derived"|"unknown"  [必填] 样本表/特征表/派生
  source_path: str             [必填] 相对 task_dir
  format: "csv"|"feather"|"parquet"|"xlsx"  [必填]
  sheet: str|None              [可选] xlsx sheet 名
  row_count: int               [必填]
  columns: list[ColumnProfile] [必填]
  has_target: bool             [必填] 是否含 y
  target_col: str|None         [可选]
  created_at: str              [必填]

ColumnProfile
  name: str                    [必填] 拍平后的列名
  dtype: str                   [必填]
  semantic_role: str           [必填] id|phone|idcard|date|amount|target|score|categorical|numeric|unknown
  fingerprint: ColumnFingerprint  [必填]
  null_rate: float             [必填]
  cardinality: int             [必填]
  sample_values: list          [必填] 脱敏后前 N 个

ColumnFingerprint             ◄ 解决 raw vs md5 的核心
  value_kind: str              raw_phone|md5|raw_idcard|date|numeric|categorical|unknown
  length_mode: int|None        最常见长度
  regex_pattern: str|None      命中的模式
  is_hashed: bool              是否疑似哈希值

JoinPlan                      ◄ INV-3
  id: str
  task_id: str
  anchor_dataset_id: str       样本表（左表锚定）
  joins: list[JoinSpec]
  status: "draft"|"confirmed"|"executed"|"rejected"
  result_dataset_id: str|None

JoinSpec
  feature_dataset_id: str
  key_pairs: list[KeyPair]     [(anchor_col, feature_col, transform)]
  diagnostics: JoinDiagnostics
  dedup_strategy: "first"|"last"|"agg_mean"|"agg_max"|"abort"|None
  confirmed: bool

KeyPair
  anchor_col: str
  feature_col: str
  transform: "none"|"md5"|"lower"|"strip"|"to_date"  对齐两侧的变换

JoinDiagnostics
  anchor_rows: int
  feature_rows: int
  feature_key_unique: bool     特征表键是否唯一（不唯一=fan-out 源）
  matched_rows: int            命中行数
  match_rate: float            命中率
  joined_rows_preview: int     试拼后行数（小样本外推）
  fan_out_detected: bool       joined > anchor → True，告警
  shrink_detected: bool        match_rate 过低 → True，告警
  new_columns: int
  new_columns_null_rate: float
```

### 4.4 记忆契约

沿用 V1.1 `agent_memory` schema，自进化层新增（详见 Phase 5 spec）：

```text
MemoryDistillation            自进化产物
  id: str
  source_memory_ids: list[str] 被蒸馏的原始记忆
  category: str                user_preference|field_convention|validation_pitfall|task_experience|model_experience
  distilled_summary: str       压缩后的经验
  support_count: int           支持该结论的任务数
  confidence: "high"|"medium"|"low"
  superseded: bool             是否被更新的蒸馏取代
```

---

## 5. 编排 Harness 设计（方案 C：模板优先 + 受约束规划）

### 5.1 执行总环

```text
用户目标
  │
  ▼
IntentRouter.route(goal, task_context)
  │   ├─ 命中 Workflow 模板 → Planner.from_template(template, slots)
  │   └─ 未命中 → Planner.generate(goal, available_tools, memory_context)
  ▼
Plan (DRAFT)
  │
  ▼
PlanValidator.validate(plan)
  │   schema 校验 · 工具可用性 · 权限 · 数据契约兼容 · 环检测 · 不变量预检
  ▼
Plan (VALIDATED)
  │
  ▼
[autonomy_level 决定] 人工确认计划 → Plan (CONFIRMED)
  │
  ▼
PlanExecutor.run(plan)
  │   按拓扑序，对每个 ready 的 step：
  │     1. needs_confirmation? → AWAITING_CONFIRM → 等用户
  │     2. sub_agent? → SubAgentDispatcher.spawn(step) → 执行
  │        else → ToolRunner.invoke(step.tool_ref, step.inputs)
  │     3. Reviewer.deterministic_check(step, output, post_checks)  ◄ 硬闸门
  │        失败 → 按 failure_policy 处理（fail/retry/skip）
  │     4. Reviewer.llm_critique(step, output)  ◄ 软审查（可配置开关）
  │     5. step.output_ref 登记，解锁下游
  ▼
Reviewer.final_review(plan, all_outputs, goal)  ◄ 终审：结果是否达成 goal
  │
  ▼
产出（artifacts / metrics / 报告 / 记忆候选）
```

### 5.2 IntentRouter

- 输入：用户目标自然语言 + 任务上下文（已有数据集、已完成步骤、记忆）。
- 职责：把目标映射到 (a) 已知 Workflow 模板 id + 槽位，或 (b) 标记为 novel 走生成路径。
- 实现：优先**规则 + 关键词 + 模板签名匹配**（确定性、可审计），LLM 仅做"这个目标最接近哪个模板"的兜底分类，且分类结果要落到结构化模板 id，不让 LLM 自由发挥流程。
- **模板来源**：候选模板 = 内置模板 + 用户可编写的 Workflow 模板（= skill，声明式、过 PlanValidator 校验后才进候选，见 Phase 2 Part C2）。两者匹配平权，内置 id 权威不可被用户 skill 遮蔽。

### 5.3 Planner

- `from_template(template, slots)`：模板是预定义的 PlanStep 序列骨架，Planner 只填 `inputs` 槽位（数据集 id、列名、参数）。**这是高频任务的主路径**，确定性最高。
- `generate(goal, tools, memory)`：novel 任务。LLM 在「可用工具清单 + 输入输出 schema + 记忆上下文」约束下生成 Plan JSON。生成后**强制**过 PlanValidator。生成失败或校验不过 → 降级为「请用户手动选工具」或「拆成更小目标重试」。

### 5.4 PlanValidator（受约束规划的关键）

逐项校验，任一失败则 Plan 不可执行：

```text
- 每个 step.tool_ref 在 registry 中存在且 enabled
- step.inputs 符合 tool.input_schema
- depends_on 无环、无悬挂引用
- 上游 output_schema 与下游 input_schema 类型兼容
- 涉及 join 的 step 必带 JoinPlan 确认门（INV-3）
- 涉及确定性指标的 step，其 post_checks 覆盖 INV-1 区间检查
- 权限/side_effects 在审计可接受范围
```

### 5.5 平台 Hook 事件（与 roadmap 对齐）

```text
task.created · task.scanned · dataset.registered · join.confirmed
notebook.completed · validation.completed · feature.computed
plan.confirmed · step.completed · report.before_generate · report.after_generate
memory.before_save · memory.after_save · workflow.completed
```

### 5.6 SubAgentDispatcher

- 子 agent = 受限上下文 + 受限工具集的一次性执行单元，用于拆分大任务（如"并行对 5 张特征表分别 profiling"）。
- 关键约束：子 agent 只拿完成 `scope` 所需的 `granted_tools` 和 `context_budget`，不继承父 agent 全部权限；产出 `result_ref` 回父计划。
- 失败隔离：子 agent 崩溃只影响其 step，按 failure_policy 处理，不拖垮整个 Plan。

### 5.7 Reviewer（双层自审）

- `deterministic_check`：硬闸门。跑 step.post_checks——schema 符合？数值区间合法（KS∈[0,1]）？行数不变量（INV-3）？非空？匹配率达标？**任一不过则 step 失败**。
- `llm_critique`：软审查。LLM 判断"这步输出是否合理地服务于目标"，输出 verdict 但**不能覆盖** deterministic 的失败结论。
- `final_review`：终审。对照原始 goal 检查整体产出完整性，生成"待人工复核项"。

---

## 6. Plugin/Tool Runtime 设计

### 6.1 安装即信任 + 子进程隔离

- **信任模型**：管理员安装的 Plugin 即受信任（单机私有化场景）。manifest 的 `permissions` 仅做审计展示，不在子进程层面强制拦截文件/网络（避免 seccomp/容器复杂度，性价比考量）。
- **隔离模型**：每次 Tool 执行 fork 一个子进程（`multiprocessing` 或 `subprocess`），传入校验过的 inputs（序列化），回收校验过的 outputs。子进程负责：超时杀进程、内存上限、异常捕获、stdout/stderr 收集。**崩溃/超时/OOM 不影响主服务**（INV-6）。
- **数据传递**：大数据集**不**走进程间序列化，而是传 **dataset 句柄/文件路径**，子进程自己从 data 层读取（避免 pickle 大 DataFrame，参考 CODE_REVIEW P2-7 教训，中间格式用 parquet/feather）。

### 6.2 ToolRunner 执行流

```text
invoke(tool_ref, inputs):
  1. registry 查 ToolSpec，校验 inputs ⊢ input_schema
  2. 起子进程，注入 inputs + dataset 句柄 + seed（若 stochastic）
  3. 子进程内 import entrypoint，调用 tool 函数
  4. 超时/OOM 守护
  5. 收集 output，校验 output ⊢ output_schema
  6. 写审计（inputs hash, duration, outcome, side_effects）
  7. 返回结构化 output 或结构化错误
```

### 6.3 内置包 vs 上传插件

- 内置包（`packs/*`）随平台发布，`builtin=True`，免上传，启动时自动注册。
- 上传插件走 `POST /plugins` → 校验 manifest 结构 → 平台计算 checksum → 落盘 → 注册 → enable/disable。

---

## 7. 数据层设计（含 Join 引擎方法论）

> 这是用户最强调、风险最高的子系统。设计目标：**把人工拼表的隐性判断显性化为可机检的诊断 + 强制确认**。

### 7.1 Excel 多 sheet + 合并表头摄取（`excel_ingest.py`）

- 枚举所有 sheet，每个 sheet 作为候选 Dataset。
- **合并表头拍平**：读前 N 行，检测多行表头（合并单元格在 pandas 读取时表现为 NaN + ffill 模式），把多级表头拍平成 `父_子` 单行列名，自动定位数据起始行。
- 输出标准化的单层表头 DataFrame + 摄取报告（原始 sheet、表头层数、数据起始行）。

### 7.2 Schema 推断与列指纹（`schema_infer.py` + `fingerprint.py`）

- 列类型推断 + **语义角色**推断：id/phone/idcard/date/amount/target/score/categorical/numeric。
- **列值指纹**（关键）：对每列采样算指纹——
  - 11 位纯数字 → `raw_phone`
  - 18 位（17 数字+X）→ `raw_idcard`
  - 32 位 hex → `md5`（`is_hashed=True`）
  - 日期格式 → `date`
  - 据此识别"名字不同但都是手机号"的列，并标记 raw vs hashed 差异。

### 7.3 列对齐（`align.py`）

- **维护的信贷键字典**（主路径）：
  ```text
  phone 族: phone, mobile, tel, phone_no, phone_md5, mobile_md5, ...
  id    族: idcard, idnumber, id_no, cert_no, id_md5, idcard_md5, ...
  date  族: date, applydate, apply_date, huisudate, data_date, dt, ...
  ```
- **模糊匹配兜底**：字典外的列用名称相似度 + 指纹相似度匹配，但置信度标低、必须人工确认。
- 输出候选 `KeyPair` 列表（含 transform 建议：若一侧 raw 一侧 md5 → transform=md5）。

### 7.4 Join 引擎（`join_engine.py`）

```text
propose_join_plan(anchor_dataset, feature_datasets):
  对每个 feature_dataset：
    1. align 出候选 key_pairs（含 raw→md5 transform）
    2. 小样本命中验证：从 anchor 取样本键，查 feature，算 match_rate
    3. 试拼诊断（小样本外推）：
         - feature_key_unique?  否 → fan_out 风险
         - matched_rows / match_rate
         - joined_rows_preview vs anchor_rows → fan_out_detected
         - match_rate 过低 → shrink_detected
    4. 生成 JoinSpec（含 diagnostics、dedup_strategy 待选）
  返回 JoinPlan(status="draft")

execute_join_plan(join_plan):   # 仅当所有 JoinSpec.confirmed
  逐表 LEFT JOIN 锚定到样本表
  执行后断言 INV-3：joined_rows <= anchor_rows（否则 abort + 告警）
  产出 result Dataset
```

- **不变量**：左连接锚定 → 结果行数 ≤ 样本表行数。违反即 fan-out，硬告警。
- **去重**：feature 键不唯一时，强制用户选 dedup_strategy，绝不静默挑行。
- **后端选择**：十万行内 pandas；百万~千万行用 DuckDB 做 join/聚合（`backend.py` 抽象），建模环节再降采样转 pandas。

### 7.5 采样与画像（`sampler.py` / `profiler.py`）

- 分层/随机/按时间采样，供小样本验证、试拼、特征预览。
- 列画像：null 率、基数、分布摘要、Top 值，喂给前端预览和记忆字段口径。

---

## 8. 记忆自进化设计（Phase 5 概要，详见对应 spec）

在 V1.1 记忆基础上加「蒸馏 / 经验固化」层：

- **触发**：验证完成、报告确认、用户纠正、多任务积累到阈值时，后台触发蒸馏。
- **蒸馏**：把同类多条原始记忆压缩成一条高置信 `MemoryDistillation`（如"该机构 idcard 字段常叫 id_md5"出现 5 次 → 固化为高置信字段口径）。
- **进化**：新证据出现时，旧蒸馏 `superseded=True`，保留审计链。
- **使用**：检索时优先返回高置信蒸馏，降低 prompt 体积。
- **铁律**：蒸馏只影响解释/建议/口径，不碰确定性指标（INV-4）。

---

## 9. 能力包目录

| 包 | Phase | 关键 Tool（示意，详见 spec） |
|----|-------|---------------------------|
| `data_ops` | 3 | ingest_excel, infer_schema, align_columns, propose_join, execute_join, clean_format, dedup_rows |
| `feature` | 4 | compute_feature_metrics, compute_psi, bin_feature, woe_encode, onehot_encode, normalize, impute_missing, cap_outliers, cross_features, correlation_analysis |
| `modeling` | 6 | check_data_quality, modeling_readiness, prepare_modeling_frame, select_features, train_model, compare_experiments, export_pmml, handoff_to_validation |
| `strategy` | 7 | vintage_curve, roll_rate, profit_calc, build_strategy, backtest_strategy, tradeoff_view |
| `v1_compat` | 2C | scan_materials, run_notebook, compute_validation_metrics, render_reports —— 把现有 V1 `pipeline.py`/`notebooks.py`/`validation/`/`output/` 能力包装成 Tool（plugin=`v1_compat`），激活 `model_validation` 旗舰模板（见 Phase 2 模板注解 + §13 Phase 2C） |
| `drafts` | 8 | web_search, draft_script, run_draft（转正后才进正式 registry，INV-7） |

每个 Tool 都是 `determinism` 明确、input/output schema 完整、子进程可执行的单元。`v1_compat` 是把 V1 验证流程接入 V2 编排的桥接包：在它落地前，Phase 2 编排用 `_sample` 桩工具验证，`model_validation` 模板的真实运行需等它就绪。

> **Phase 4V 不在本表**：它是 `validation/vintage.py` 里的**共享确定性核心**（`compute_vintage_curve`/`vintage_curve_wide`/`compute_roll_rate`），不是 pack、不单独注册 Tool；由 `modeling`（Phase 6 报告）和 `strategy`（Phase 7）**import 复用**——`strategy` 表里的 `vintage_curve`/`roll_rate` Tool 是对它的薄包装，不重写计算口径。这样避免 Phase 6 反向依赖 Phase 7（见 §13 Phase 4V）。

---

## 10. 联网学习草稿区与治理（Phase 8 概要）

```text
缺工具 → (有网) Agent 联网搜索 → 学习 → 写脚本 → 注册为 DraftTool
DraftTool 只能在当前任务"临时运行"（有审计标记），不进 registry，Planner 不自动选
人工评审 → "转正" → 校验 schema/审计/测试 → 注册为正式 Tool
```

- 离线环境下草稿区不可用（INV 离线契约），改为"人在外部环境产出工具 → 上传"。
- 治理闸门：草稿转正前必须有 input/output schema、确定性声明、最小测试。

---

## 11. 前端架构（ES Modules 拆分）

把现有 6300 行 `app.js` 拆为无构建 ES Modules：

```text
static/js/
  core/
    state.js        集中可变状态 + 订阅
    bus.js          事件总线
    api.js          fetch 封装、错误处理
    poll.js         统一轮询管理（去重，修 CODE_REVIEW P2-21）
  views/
    task_tree.js    多 agent 任务树
    subagent.js     子 agent 实时状态
    plugins.js      plugin 管理
    workflow.js     workflow DAG 可视化
    artifacts.js    工件预览
  render/
    metrics.js      指标表（迁移现有逻辑）
    agent_chat.js   Agent 对话
    markdown.js     Markdown（修 P2-20 链接安全）
  main.js           装配入口
```

- 保持「无 npm、无构建、离线可部署」。`index.html` 用 `<script type="module">` 加载。
- 迁移时一并修 CODE_REVIEW 报告的前端问题（P1-22 结构化字段、P2-19~24）。

---

## 12. 持久层 schema 总览

SQLite 新增表（沿用 `db.py` 的 `connect()` 封装 + WAL + busy_timeout，修复 CODE_REVIEW P0/P1 后）：

```text
plans(id, task_id, goal, source, template_id, autonomy_level, status, created_at, updated_at)
plan_steps(id, plan_id, idx, title, tool_plugin, tool_name, tool_version,
           inputs_json, depends_on_json, post_checks_json, needs_confirmation,
           sub_agent_id, status, output_ref, review_json, error, ...)
sub_agents(id, parent_task_id, parent_step_id, scope, granted_tools_json,
           context_budget, status, result_ref, created_at)
plugins(name, version, display_name, description, builtin, manifest_json,
        checksum, enabled, installed_at)
tools(plugin, name, summary, input_schema_json, output_schema_json,
      determinism, timeout_seconds, failure_policy, side_effects_json)
datasets(id, task_id, role, source_path, format, sheet, row_count,
         columns_json, has_target, target_col, created_at)
joins(id, task_id, anchor_dataset_id, joins_json, status, result_dataset_id, created_at)
memory_distillations(id, source_memory_ids_json, category, distilled_summary,
                     support_count, confidence, superseded, created_at)
draft_tools(id, task_id, name, source, code_ref, input_schema_json,
            output_schema_json, status, created_at)
audit(id, kind, actor, target_ref, inputs_hash, outcome, detail_json, at)
```

---

## 13. 阶段计划与依赖

```text
Phase 0  地基       ─┐ 修 CODE_REVIEW P0/P1 + 前端 ES Module 拆分 + 补边界测试
                     │ （无新功能依赖，但后续都站在它上面）
Phase 1  Plugin     ─┤ 依赖 0；Tool Runtime + registry + 子进程 runner + hooks
Phase 2  编排       ─┤ 依赖 1；IntentRouter/Planner/Validator/Executor/SubAgent/Reviewer
                     │ + 用户可编写 Workflow 模板（= skill，声明式、过 PlanValidator，Part C2）
Phase 2B 自适应Loop ─┤ 依赖 2；决策点重规划 + 失败驱动重规划 + explore 模式 +
                     │ 上下文工程层 + capability_tier（按模型能力伸缩，换模型只改配置）
Phase 2C v1_compat  ─┤ 依赖 1 + 现有 pipeline/notebooks/validation/output；把 V1 扫描/
                     │ Notebook/指标/报告封成 Tool（scan_materials/run_notebook/
                     │ compute_validation_metrics/render_reports），**激活 model_validation
                     │ 旗舰模板**（纯包装、保持 V1 行为；可与 2B 并行）
Phase 3  数据层+包  ─┤ 依赖 1（作为内置包）；data/ + packs/data_ops（join 引擎）
Phase 4  特征包     ─┤ 依赖 1,3；packs/feature（复用 validation/）
Phase 4V vintage核  ─┤ 依赖 3；validation/vintage.py（vintage_curve/roll_rate 确定性核心，
                     │ 被 Phase 6 报告和 Phase 7 策略共用，先于两者交付）
Phase 5  记忆自进化 ─┘ 依赖现有 agent_memory；蒸馏层
Phase 6  模型包     依赖 1,3,4,4V（报告 Part O 用 vintage）
Phase 7  策略包     （蓝图）依赖 1,3,4,4V,6
Phase 8  草稿区治理 （蓝图）依赖 1,2
前端 V2  函数级 spec：2026-06-13-phase-frontend-v2.md（依赖 1,2,2B,3 的 API；可与后端并行）
```

关键路径：0 → 1 → 2 → 2B 是 harness 主干；0 → 1 → 3 → 4 是能力主干。两条可在 1 完成后并行。
**vintage 顺序修正**：`vintage_curve`/`roll_rate` 的确定性实现放在共享的 `validation/vintage.py`（Phase 4V，紧跟 Phase 3/4），Phase 6 模型报告和 Phase 7 策略包都 import 同一份，避免"Phase 6 报告依赖更靠后的 Phase 7"的倒置。

---

## 14. 全局工程约定

- **语言/风格**：Python 3.11+，`@dataclass` 契约，类型注解完整。中文叙述 + 英文标识符。
- **错误处理**：结构化异常（每层定义自己的 `*Error`），不裸 `except Exception: pass`（修 CODE_REVIEW 教训）。
- **跨平台**（INV-9）：路径 `pathlib` + `as_posix()` 注入；子进程显式 `encoding="utf-8"`；不依赖 POSIX-only。
- **DB**（INV-8）：一律走 `db.py` 的 `connect()` 封装；DDL 在单事务；审计必写。
- **确定性**（INV-1）：所有指标 tool 标 `determinism`；stochastic 必须有 `seed` 入参且记录。
- **测试**：每个函数级 spec 标测试要点；重点覆盖边界（NULL 列、tz-aware 时间、零方差、bad_count=0、非 DeepSeek provider、fan-out join、raw-vs-md5）——这些正是 CODE_REVIEW 暴露的盲区。
- **审计**：Tool 执行、记忆读写、计划决策、子 agent 派发、join 确认全留痕。
- **ruff**：开启 F811（重定义检查），CI 拦截 CODE_REVIEW P1-1 类重复定义。
- **提交**：遵循 AGENTS.md decision trailers；发布走 `scripts/release_push.py`。

---

## 15. Phase 6~8 概要设计

> Phase 6~8 已拆出独立函数级 spec；本节保留产品和依赖关系概要，具体函数、测试和任务顺序以对应 spec 为准。

### 15.1 Phase 6：模型开发包（`packs/modeling`）

- **范围**：数据质量/建模准备度检查、特征加工与筛选、常见信贷模型 recipe（LGB/XGB/LR/评分卡）、实验记录、模型产物交接到验证流程。
- **关键 Tool（蓝图）**：`check_data_quality`、`prepare_modeling_frame`、`select_features`、`train_lgb/xgb/lr/scorecard`、`log_experiment`、`export_pmml`、`handoff_to_validation`。
- **拒绝推断**：单列为 `reject_inference`（Heckman/平行分配/增广/模糊增广），**标注「需方法论评审，第一版不实现」**。
- **铁律**：训练产物必须进验证流程后才算可复核产物；训练上下文与验证契约独立（roadmap V3 约束）。
- **场景覆盖**：贷前/贷中/贷后、前筛、营销、交易、捞回、收入、额度、定价——以 recipe + 场景参数模板实现，不是各写一套。

### 15.2 Phase 7：Vintage / 盈利测算 / 策略回测包（`packs/strategy`）

- **范围**：vintage 曲线、滚动率（roll rate）、盈利测算（含 vintage 维度）、策略生成（额度/定价/准入/拒绝/分群）、策略回测。
- **关键 Tool（蓝图）**：`vintage_curve`、`roll_rate_matrix`、`profit_calc`、`build_strategy`、`backtest_strategy`、`tradeoff_view`（风险/收益/通过率/坏账率权衡）。
- **铁律**：所有测算数字由 tool 算（INV-1）；关键流程保留手动模式（roadmap V4 约束）。

### 15.3 Phase 8：联网学习草稿区 + 治理（`drafts/`）

- **范围**：联网搜索学习、草稿脚本编写、DraftTool 临时运行、人工转正闸门。
- **关键 Tool/流程（蓝图）**：`web_search`（仅联网）、`draft_script`、`run_draft`（带审计的临时执行）、`promote_draft`（转正校验：schema/审计/测试）。
- **治理**：INV-7 —— 草稿转正前不得被 Planner 自动选用；离线环境改为外部产出+上传路径。

---

## 16. 风险登记册

| 风险 | 影响 | 缓解 |
|-----|------|------|
| 32B~72B 模型规划质量不稳 | 任务执行出错 | 模板优先 + PlanValidator + 确定性闸门 + 人工确认 |
| Join 错配静默产生错误样本 | 错误模型，真金白银损失 | INV-3：JoinPlan 诊断 + 行数不变量 + 强制确认（第 7 节） |
| raw vs md5 拼出 0 命中 | 浪费人力、误判无数据 | 列值指纹自动检测 + transform 建议（第 7.2 节） |
| 千万行数据 pandas 撑爆内存 | 任务失败 | DuckDB 后端 + 建模环节降采样（第 7.4 节） |
| Plugin 崩溃拖垮主服务 | 平台不可用 | 子进程隔离 + 超时/OOM 守护（INV-6） |
| LLM 自评放过错误结果 | 错误产出流向报告 | 确定性闸门为主、LLM critic 为辅（第 5.7 节） |
| 联网学习引入未审计代码 | 合规风险 | 草稿区 + 人工转正闸门（INV-7） |
| 现有 6300 行前端继续膨胀 | 不可维护 | Phase 0 强制 ES Module 拆分（第 11 节） |
| 地基债务（CODE_REVIEW P0/P1）带入新功能 | 缺陷放大 | Phase 0 先还债再盖楼（第 13 节） |

---

*本蓝图是 governing document。后续每个 Phase 函数级 spec 必须引用本蓝图的不变量编号（INV-x）和数据契约。蓝图变更需同步更新受影响的 Phase spec。*
