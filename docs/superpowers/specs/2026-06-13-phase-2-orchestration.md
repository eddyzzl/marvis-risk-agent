# Phase 2 — 编排 Harness（函数级 spec，含内部伪代码）

## 文档状态

- 状态：待实施
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 5 节，方案 C）
- 前置依赖：Phase 1（Tool Runtime / ToolRegistry / ToolRunner / HookDispatcher）完成
- 目标：交付「模板优先 + 受约束规划」的编排核心——把用户目标变成可校验、可确认、可执行、可自审、可恢复的 `Plan`。

## 设计哲学（必读）

目标模型 32B~72B，**不信任长链路自由规划**。因此：

- **高频任务走预置模板**（`WorkflowTemplate`），模型只填槽位，确定性最高。
- **新任务受约束生成**：LLM 在「可用工具目录 + 输入输出 schema + 记忆」约束下产 `Plan` JSON，**强制过 `PlanValidator`**，失败则重试或降级人工选工具。
- **确定性闸门 > LLM 自评**：每步 `post_checks`（schema/区间/行数不变量）是硬闸门；`llm_critique` 是软审查，不能覆盖确定性失败。
- **可恢复**：执行器无状态、由 DB 持久状态驱动，进程重启或等待人工确认后可从断点续跑。

## 捍卫的不变量

- **INV-1/INV-2**：Planner 只选工具、接数据流；不让 LLM 算指标。`post_checks` 在运行时强制区间/结构。
- **INV-3**：含 join 的 step 必带确认门，`PlanValidator` 强制检查。
- **INV-6**：子 agent 失败隔离，不拖垮整个 Plan。
- **INV-8**：规划决策、确认、执行、子 agent 派发全审计。

## 模块布局

```text
riskmodel_checker/orchestrator/
  __init__.py
  contracts.py        Plan / PlanStep / PostCheck / ToolRef / ReviewVerdict / SubAgent + 枚举
  errors.py           编排异常层级
  harness_state.py    HarnessState：状态机转移权威
  templates/
    __init__.py       WorkflowTemplate / SlotSpec / StepTemplate + 注册表（内置 + 用户）
    model_validation.py   内置模板（镜像 V1 验证流程）
    skills.py         用户可编写 Workflow 模板（= skill）加载/校验/治理（Part C2）
  intent.py           IntentRouter
  planner.py          Planner（from_template / generate）
  validator.py        PlanValidator
  reviewer.py         Reviewer（deterministic / llm_critique / final）
  subagent.py         SubAgentDispatcher
  executor.py         PlanExecutor（可恢复 DAG 执行）
riskmodel_checker/db.py    新增 plans / plan_steps / sub_agents 表 + PlanRepository
riskmodel_checker/routers/plans.py   HTTP 端点
```

---

## Part A — 契约与枚举（`orchestrator/contracts.py`）

### A-1 枚举

```python
class PlanStatus(str, Enum):
    DRAFT = "draft"            # 刚生成，未校验
    VALIDATED = "validated"   # 过了 PlanValidator
    CONFIRMED = "confirmed"   # 用户确认计划
    RUNNING = "running"
    AWAITING_CONFIRM = "awaiting_confirm"  # 卡在某步人工确认
    REVIEW = "review"         # 执行完，终审中
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"               # 上游未完成
    AWAITING_CONFIRM = "awaiting_confirm"
    RUNNING = "running"
    CHECKING = "checking"            # 跑 post_checks
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"

class AgentStatus(str, Enum):
    SPAWNED = "spawned"; RUNNING = "running"
    RETURNED = "returned"; FAILED = "failed"; KILLED = "killed"
```

### A-2 dataclasses（与蓝图 4.1 一致，补默认值与方法）

> **ToolRef 复用 Phase 1 定义**：`from riskmodel_checker.plugins.manifest import ToolRef`，不在 orchestrator 重复定义（见 Phase 1 Part B-1）。下面只列出编排层新增的契约。

```python
# ToolRef 见 Phase 1 plugins/manifest.py —— { plugin, tool, version="" }, .label()

@dataclass(frozen=True)
class PostCheck:
    kind: str                # schema|range|rowcount|invariant|nonempty|match_rate
    spec: dict               # 例 {"field":"ks","min":0.0,"max":1.0}
                             #    {"rule":"joined_rows<=anchor_rows"}
                             #    {"field":"match_rate","min":0.5}

@dataclass
class ReviewVerdict:
    reviewer: str            # "deterministic" | "llm_critic"
    passed: bool
    reasons: list[str]
    at: str

@dataclass
class PlanStep:
    id: str
    plan_id: str
    index: int
    title: str
    tool_ref: ToolRef
    inputs: dict
    depends_on: list[str]
    post_checks: list[PostCheck]
    needs_confirmation: bool = False
    sub_agent_scope: str | None = None   # 非空=该步派子 agent
    granted_tools: list[ToolRef] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    sub_agent_id: str | None = None
    output_ref: str | None = None
    review_verdicts: list[ReviewVerdict] = field(default_factory=list)
    error: str | None = None

@dataclass
class Plan:
    id: str
    task_id: str
    goal: str
    source: str              # "template" | "generated"
    template_id: str | None
    steps: list[PlanStep]
    autonomy_level: int      # 0=全人工确认 1=关键步确认 2=仅终审确认
    status: PlanStatus = PlanStatus.DRAFT
    created_at: str = ""
    updated_at: str = ""

@dataclass
class SubAgent:
    id: str
    parent_task_id: str
    parent_step_id: str | None
    scope: str
    granted_tools: list[ToolRef]
    context_budget: int
    status: AgentStatus = AgentStatus.SPAWNED
    result_ref: str | None = None
```

### A-3 输出引用（`output_ref` 约定）

```text
output_ref 是产物句柄字符串，格式 "<kind>:<id>"，kind ∈ {dataset, metrics, artifact, value}:
  "dataset:<dataset_id>"   指向 data 层 Dataset
  "metrics:<step_id>"      指向 plan_step_outputs 表里存的结构化 metrics JSON
  "artifact:<path>"        指向 task_dir 下文件
下游 step.inputs 用 "$ref:<step_id>.output.<field>" 语法引用上游 output，
PlanValidator 校验引用可解析、类型兼容，PlanExecutor 执行前做实参替换。
```

- **测试要点**：dataclass 序列化往返；`ToolRef.label()`；`output_ref` 解析辅助函数正/反例。

---

## Part B — 状态机（`orchestrator/harness_state.py`）

集中管理 Plan/Step 合法转移（参考现有 `db.py` 的 task `assert_transition` 模式）。

```python
PLAN_TRANSITIONS = {
    PlanStatus.DRAFT: {PlanStatus.VALIDATED, PlanStatus.FAILED, PlanStatus.CANCELLED},
    PlanStatus.VALIDATED: {PlanStatus.CONFIRMED, PlanStatus.FAILED, PlanStatus.CANCELLED},
    PlanStatus.CONFIRMED: {PlanStatus.RUNNING, PlanStatus.CANCELLED},
    PlanStatus.RUNNING: {PlanStatus.AWAITING_CONFIRM, PlanStatus.REVIEW,
                         PlanStatus.FAILED, PlanStatus.CANCELLED},
    PlanStatus.AWAITING_CONFIRM: {PlanStatus.RUNNING, PlanStatus.CANCELLED},
    PlanStatus.REVIEW: {PlanStatus.DONE, PlanStatus.FAILED},
    PlanStatus.DONE: set(), PlanStatus.FAILED: set(), PlanStatus.CANCELLED: set(),
}

def assert_plan_transition(current: PlanStatus, target: PlanStatus) -> None:
    """非法转移抛 IllegalPlanTransition。
    伪代码:
      if target not in PLAN_TRANSITIONS.get(current, set()):
          raise IllegalPlanTransition(f"{current} -> {target}")
    """

STEP_TRANSITIONS = {
    StepStatus.PENDING: {StepStatus.BLOCKED, StepStatus.AWAITING_CONFIRM,
                         StepStatus.RUNNING, StepStatus.SKIPPED},
    StepStatus.BLOCKED: {StepStatus.PENDING, StepStatus.AWAITING_CONFIRM,
                         StepStatus.RUNNING, StepStatus.SKIPPED},
    StepStatus.AWAITING_CONFIRM: {StepStatus.RUNNING, StepStatus.SKIPPED},
    StepStatus.RUNNING: {StepStatus.CHECKING, StepStatus.FAILED},
    StepStatus.CHECKING: {StepStatus.DONE, StepStatus.FAILED},
    StepStatus.DONE: set(), StepStatus.FAILED: {StepStatus.PENDING},  # retry 回 PENDING
    StepStatus.SKIPPED: set(),
}

def assert_step_transition(current: StepStatus, target: StepStatus) -> None:
    """非法转移抛 IllegalStepTransition。"""
```

- **测试要点**：合法/非法转移分别通过/抛错；retry 路径 FAILED→PENDING 合法。

---

## Part C — Workflow 模板（`orchestrator/templates/`）

### C-1 模板契约

```python
@dataclass(frozen=True)
class SlotSpec:
    name: str                # 槽位名，如 "sample_dataset_id"
    required: bool
    source: str              # "task_context" | "user" | "infer"
    description: str

@dataclass(frozen=True)
class StepTemplate:
    title: str
    tool_ref: ToolRef
    inputs_template: dict    # 含 "{slot:xxx}" 或 "$ref:..." 占位
    depends_on_titles: tuple[str, ...]   # 用 title 表达依赖，实例化时转 step id
    post_checks: tuple[PostCheck, ...]
    needs_confirmation: bool = False
    sub_agent_scope: str | None = None
    # 派子 agent 的步骤必须随附最小工具授权；否则 from_template 出来的 PlanStep
    # granted_tools=[] → SubAgentDispatcher 拿到空注册表 → mini_planner 必失败。
    # sub_agent_scope 非空时 granted_tools 必须非空（PlanValidator 强制，见 Part F）。
    granted_tools: tuple[ToolRef, ...] = ()

@dataclass(frozen=True)
class WorkflowTemplate:
    id: str
    title: str
    goal_patterns: tuple[str, ...]   # 给 IntentRouter 的关键词/正则签名
    slots: tuple[SlotSpec, ...]
    steps: tuple[StepTemplate, ...]
    default_autonomy: int = 1
    source: str = "builtin"          # "builtin"（随包代码）| "user"（workspace skill 文件，见 Part C2）
```

### C-2 模板注册表

```python
_TEMPLATES: dict[str, WorkflowTemplate] = {}

def register_template(template: WorkflowTemplate) -> None:
    """注册内置模板。重复 id 抛 ValueError。"""

def get_template(template_id: str) -> WorkflowTemplate:
    """取模板。不存在抛 KeyError。"""

def list_templates() -> list[WorkflowTemplate]:
    """列出全部。"""

def load_builtin_templates() -> None:
    """import 各 templates/*.py 触发 register_template（启动时调）。"""
```

### C-3 内置模板示例（`templates/model_validation.py`）

```python
MODEL_VALIDATION = WorkflowTemplate(
    id="model_validation",
    title="模型验证（镜像 V1 流程）",
    goal_patterns=("模型验证", "验证模型", "validate model", "跑验证"),
    slots=(
        SlotSpec("task_id", True, "task_context", "当前任务"),
    ),
    steps=(
        # 四个 tool 同属 v1_compat 包（Phase 2C），把 V1 扫描/Notebook/指标/报告封成 Tool。
        # 它们都是 task_id-stateful：通过任务目录串接（scan 识别 Notebook/样本、run 写产物、
        # metrics 读产物…），所以每步只接 {"task_id"}（与 Phase 2C Part A 的 input 契约一致），
        # 不走 $ref 数据流占位；步序由 depends_on 保证。
        StepTemplate("扫描材料", ToolRef("v1_compat", "scan_materials"),
                     {"task_id": "{slot:task_id}"}, (),
                     (PostCheck("nonempty", {"field": "materials"}),)),
        StepTemplate("执行 Notebook", ToolRef("v1_compat", "run_notebook"),
                     {"task_id": "{slot:task_id}"}, ("扫描材料",),
                     (PostCheck("invariant", {"rule": "notebook_executed"}),)),
        StepTemplate("计算验证指标", ToolRef("v1_compat", "compute_validation_metrics"),
                     {"task_id": "{slot:task_id}"}, ("执行 Notebook",),
                     (PostCheck("range", {"field": "ks", "min": 0.0, "max": 1.0}),
                      PostCheck("range", {"field": "auc", "min": 0.0, "max": 1.0}))),
        StepTemplate("生成报告", ToolRef("v1_compat", "render_reports"),
                     {"task_id": "{slot:task_id}"}, ("计算验证指标",),
                     (PostCheck("nonempty", {"field": "artifacts"}),),
                     needs_confirmation=True),   # 报告生成前人工确认
    ),
    default_autonomy=1,
)
register_template(MODEL_VALIDATION)
```

> **这四个 tool 同属 `v1_compat` 包（已排期 Phase 2C，见蓝图 §13）**：`scan_materials`/`run_notebook`/`compute_validation_metrics`/`render_reports` 是把现有 V1 能力（`pipeline.py` 扫描/Notebook 执行、`validation/` 指标、`output/` 报告）**包装成 Tool 契约**的产物，**不**复用 Phase 3 `data_ops` / Phase 4 `feature`（feature 的指标 tool 叫 `compute_feature_metrics`，语义也不同）。
> - `compute_validation_metrics` 的 `output_schema` 必须声明 `ks`/`auc`（否则 Part F `_check_determinism_checks` 无从对它强制 INV-1 区间门）。
> - **Phase 2 自身**用 `_sample` 桩工具 + mock 验证编排逻辑（不依赖真实 V1 tool），所以 Phase 2 可独立实施和测试；`model_validation` 模板的**真实运行**等 Phase 2C `v1_compat` 就绪（纯包装、保持 V1 行为，可与 2B 并行）。

- **测试要点**：注册/取/列模板；`goal_patterns` 可被 IntentRouter 命中；模板 slots 完整。

> 用户 skill 模板的注册口 `register_user_template` / 防遮蔽 `builtin_template_ids` 见 **Part C2**。

---

## Part C2 — 用户可编写 Workflow 模板（= skill）

### 设计立场（为什么 skill 不单独立 runtime）

"skill" 在历史文档里指 SOP / Playbook / 方法论型知识（如「A 卡标准建模 SOP」）。本平台**不**为它建第 4 套抽象（Plugin/Tool/Hook/Workflow 之外的独立 skill runtime）。**一个 skill 就是用户可编写、可版本化的 Workflow 模板**：与 Part C 的内置 `WorkflowTemplate` 同构，只是由用户以**声明式文件**编写，而非随包发布的 `templates/*.py`。

**信任边界（关键）**：用户 skill 模板是**纯声明式数据**（槽位 + 步骤 + `tool_ref` + `post_checks`），**不含可执行代码**。它只能编排**已注册、已信任**的工具（工具信任在 Phase 1 安装期 + 子进程隔离时已建立）。模板实例化成 `Plan` 后**仍旧过 `PlanValidator`（Part F）**，因此 INV-1（指标 step 必带区间 post_check）、INV-3（join step 必带确认门）对用户模板**一视同仁**——用户无法编写出"绕过确认门的 join"或"让 LLM 算指标"的 skill。这就是 skill 不需要独立沙箱 runtime 的根据：**它复用工具的信任边界与 Plan 的校验闸门，不引入新的执行面**。

### C2-1 声明式文件格式（`workspace/skills/<skill_id>.json`）

用户在工作区 `skills/` 目录下每个 skill 一个 JSON 文件（与 branding `brand.json` 同属"工作区可编写配置"，天然可 git 版本化；**离线可用**）。结构镜像 `WorkflowTemplate`（C-1）：

```json
{
  "id": "a_card_sop",
  "title": "A 卡标准建模 SOP",
  "goal_patterns": ["A卡", "申请评分卡", "做一张A卡"],
  "default_autonomy": 1,
  "enabled": true,
  "slots": [
    {"name": "sample_dataset_id", "required": true, "source": "user", "description": "建模样本数据集"}
  ],
  "steps": [
    {"title": "特征分析", "tool": {"plugin": "feature", "tool": "compute_feature_metrics"},
     "inputs": {"dataset": "{slot:sample_dataset_id}"}, "depends_on": [],
     "post_checks": [{"kind": "range", "spec": {"field": "iv", "min": 0.0, "max": 10.0}}],
     "needs_confirmation": false}
  ]
}
```

字段与 C-1 一一对应：`tool` = `ToolRef` 的 `{plugin, tool, version?}`；`post_checks[].{kind, spec}` = `PostCheck`；`depends_on` 用上游 step 的 `title` 表达（与 `StepTemplate.depends_on_titles` 一致）。step 还可带可选 `needs_confirmation`、`sub_agent_scope` + `granted_tools`（要派子 agent 的步骤**必须**给非空 `granted_tools`，否则 `validate_skill_template` 拦截，与内置模板同规则——见 Part F `_check_subagent_grants`）。格式由 `SKILL_TEMPLATE_JSON_SCHEMA` 约束。`enabled`（默认 `true`）置 `false` 时该 skill 加载时跳过注册。

### C2-2 加载器与校验（`orchestrator/templates/skills.py`）

```python
def parse_skill_template(data: dict) -> WorkflowTemplate:
    """声明式 JSON → WorkflowTemplate（source="user"）。
    异常: SkillTemplateError（schema 不符 / tool 字段缺失 / 枚举非法）。
    伪代码:
      validate_against_schema(data, SKILL_TEMPLATE_JSON_SCHEMA, label="skill")   # 复用 Phase 1
      slots = tuple(SlotSpec(**s) for s in data["slots"])
      steps = tuple(_step_template_from_json(s) for s in data["steps"])  # tool→ToolRef, post_checks→PostCheck
      return WorkflowTemplate(id=data["id"], title=data["title"],
                              goal_patterns=tuple(data.get("goal_patterns", ())),
                              slots=slots, steps=steps,
                              default_autonomy=data.get("default_autonomy", 1),
                              source="user")

def validate_skill_template(template, tool_registry, plan_validator) -> list[str]:
    """作者期校验：把模板用占位槽位"干跑"实例化，再过 PlanValidator，返回问题列表（空=合法）。
    这一步在"注册前"就强制 INV-1/INV-3，坏 skill 不会进 IntentRouter 候选。
    不变量: 与运行期同一套 PlanValidator；作者期校验"结构"，槽位实值的 schema 留运行时
            （见 _dry_instantiate 的延迟占位），不因未知槽位值误拒合法 skill。
    伪代码:
      problems = []
      if template.id in builtin_template_ids():            # 不许遮蔽内置（内置权威）
          problems.append(f"skill id '{template.id}' shadows a builtin template")
      try:
          plan = _dry_instantiate(template, tool_registry)  # 见下：用类型合法占位填槽
          problems += plan_validator.validate(plan)          # 工具存在/DAG/$ref/join 门/指标 post_check
      except (PlanningError, SkillTemplateError) as e:
          problems.append(str(e))
      return problems

def _dry_instantiate(template, tool_registry) -> Plan:
    """仿 Planner.from_template 实例化模板用于作者期校验，但**不**给 "{slot:x}" 填实值——
    保留为延迟占位（PlanValidator 的 _is_deferred_input 跳过它的字面量 schema，见 Part F）。
    "$ref:Title.output.f" 照常转真实 step id（结构/依赖/上游字段仍校验）。这样作者期只校验
    "结构"（工具存在、DAG、$ref 兼容、join 确认门、指标 post_check、子 agent 授权），把"槽位
    实值的 schema"留到用户真正运行 skill 时由 from_template→validate 校验——既不会因未知槽位值
    误拒合法 skill，作者期与运行期对"结构"的判定又完全一致。task_id 用占位（如 "<dry>"）。
    """

def load_user_skill_templates(workspace, tool_registry, plan_validator) -> "SkillLoadReport":
    """扫描 workspace/skills/*.json，逐个 parse+validate，合法的 register_user_template，
    非法的记进 report.rejected（带 reasons）。启动时与 /skills/reload 端点调用。
    不变量: 非法 skill 绝不注册（不进 IntentRouter）；INV-7 治理（未过闸的内容不入正式可执行候选）。
    伪代码:
      report = SkillLoadReport()
      for path in sorted(_skill_dir(workspace).glob("*.json")):
          try: data = _safe_json_loads(path.read_text("utf-8"))
          except Exception as e: report.rejected.append((path.stem, [f"unreadable: {e}"])); continue
          if data.get("enabled", True) is False:
              report.disabled.append(data.get("id", path.stem)); continue
          try: tpl = parse_skill_template(data)
          except SkillTemplateError as e: report.rejected.append((data.get("id", path.stem), [str(e)])); continue
          problems = validate_skill_template(tpl, tool_registry, plan_validator)
          if problems: report.rejected.append((tpl.id, problems)); continue
          register_user_template(tpl)                        # 覆盖旧 user 同 id；撞内置上面已挡
          report.active.append(tpl.id)
      return report
```

```python
@dataclass
class SkillLoadReport:
    active: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    rejected: list[tuple[str, list[str]]] = field(default_factory=list)   # (skill_id, problems)
```

### C2-3 治理生命周期（INV-7 对齐）

- **文件是唯一真相源**；skill 状态在加载时**派生**，不另存 DB（可重算）：
  - `active`：parse + validate 全过 → `register_user_template` → 进 IntentRouter 候选。
  - `rejected`：解析或校验失败 → **不注册** → 原因经 `/api/skills` 暴露给用户改文件。
  - `disabled`：文件里 `"enabled": false` → 加载时跳过注册。
- 与 Phase 8 草稿区一致的治理直觉：**未通过校验的东西不进可执行候选**。区别：skill 是声明式数据（只编排已信任工具），不像 Phase 8 草稿是**新代码**，所以 skill 无需草稿沙箱执行，只需 `schema` + `PlanValidator` 干跑两道闸——这正是把"skill"折叠进 Workflow 模板、而非另立 runtime 的收益。

### C2-4 注册表整合与优先级（扩展 C-2）

C-2 注册表加 `source` 维度（`WorkflowTemplate.source` 字段，见 C-1）与用户注册口：

```python
def register_user_template(template: WorkflowTemplate) -> None:
    """注册用户 skill 模板（source="user"）。
    规则: 撞内置 id 抛 ValueError（内置权威，不可遮蔽）；撞已有 user id 则覆盖（reload 语义）。"""

def builtin_template_ids() -> set[str]:
    """当前已注册的内置模板 id 集合（给 skill 校验防遮蔽用）。"""

def clear_user_templates() -> None:
    """reload 前清掉所有 source=="user" 的模板，内置保留。"""
```

`list_templates()` / IntentRouter `_match_templates`（Part D）**自动**把 active 用户 skill 纳入候选——**无需改 IntentRouter/Planner 逻辑**。内置与用户模板在匹配上平权，靠 `goal_patterns` 命中；id 唯一性由注册规则保证（内置 > 用户，禁遮蔽）。

### C2-5 HTTP 端点（`routers/plans.py` 增）

```python
@router.get("/skills")
def list_skills(request) -> dict:
    """列出用户 skill 模板及状态（active / disabled / rejected + 问题）。
    读 app.state 最近一次 SkillLoadReport + 注册表——**反映最近一次启动/reload 的结果，
    不实时扫盘**；用户改了 workspace/skills/ 下的文件后需调 /skills/reload 才生效（前端在
    skill 面板提供"重新加载"按钮，见前端 Part F2）。"""

@router.post("/skills/reload")
def reload_skills(request) -> dict:
    """重扫 workspace/skills/ 重新 parse+validate+register（先 clear_user_templates）。返回新的 SkillLoadReport。
    伪代码: clear_user_templates(); report = load_user_skill_templates(ws, registry, validator);
            app.state.skill_report = report; return _report_payload(report)"""

@router.post("/skills/validate")
def validate_skill(request, body: SkillDraftRequest) -> dict:
    """对一段未落盘的 skill JSON（body.skill）干跑 parse+validate，返回 problems（不注册、不落盘）。
    给前端 skill 编辑器实时校验用。"""
```

> 写/配置类端点受 `app.py` 本机访问守卫（`_local_access_guard` / `_is_local_only_path`）：远程只读**不**放 `/api/skills/reload`、`/api/skills/validate`，与 `/api/settings`、`/branding/` 同级（实现期把这两个前缀加进 `_is_local_only_path`）。

### 测试要点

- `parse_skill_template`：合法 JSON→`WorkflowTemplate(source="user")`；缺字段 / 坏 `tool` / 坏 `post_check.kind` 抛 `SkillTemplateError`。
- `validate_skill_template`：**引用不存在工具被拦**；**join step 缺确认门被拦（INV-3）**；**指标 step 缺区间 post_check 被拦（INV-1）**；**子 agent step 空 granted_tools 被拦**；撞内置 id 被拦；未填槽位作延迟占位、跳过字面量 schema（enum/pattern/min/max **不**误报）；合法返回空。
- `load_user_skill_templates`：合法 skill 进 `active` 且可被 `get_template` 取到；非法进 `rejected` 且**不**在 `list_templates()` 里；`enabled:false` 进 `disabled` 不注册；坏 JSON 文件不拖垮整体加载。
- 注册表：`register_user_template` 撞内置 id 抛错；撞 user id 覆盖；`clear_user_templates` 后旧 user 模板清掉、内置保留。
- IntentRouter（复用 Part D 测试）：active 用户 skill 的 `goal_patterns` 能被命中并走 `from_template`。
- 端点：list 反映 active/disabled/rejected；validate 干跑不落盘；reload 生效；reload/validate 远程被本机守卫拦。

---

## Part D — IntentRouter（`orchestrator/intent.py`）

```python
@dataclass
class IntentResult:
    kind: str                # "template" | "novel"
    template_id: str | None
    slots: dict              # 已抽取的槽位值（template 时）
    confidence: float        # 0-1
    rationale: str

class IntentRouter:
    def __init__(self, llm_factory, tool_registry):
        """入参:
             llm_factory: () -> LLM client（按当前 profile）；
             tool_registry: 给 novel 分类时参考可用工具。
        """
        self._llm_factory = llm_factory
        self._tools = tool_registry

    def route(self, goal: str, task_context: dict) -> IntentResult:
        """把用户目标路由到模板或 novel。确定性匹配优先，LLM 仅做受限分类。
        入参: goal 自然语言目标; task_context（已有数据集/已完成步骤等）。
        出参: IntentResult。
        异常: 无（兜底返回 novel）。
        不变量: LLM 输出必须落到结构化 template_id 或 "novel"，不自由发挥流程。
        """
        # 伪代码:
        # 1) 确定性签名匹配
        hit = self._match_templates(goal)
        if hit and hit[1] >= STRONG_MATCH_THRESHOLD:   # 关键词强命中
            slots = self._extract_slots(get_template(hit[0]), goal, task_context)
            return IntentResult("template", hit[0], slots, hit[1],
                                rationale=f"keyword match: {hit[0]}")
        # 2) LLM 受限分类（候选 = 所有模板 id + "novel"）
        choice = self._llm_classify(goal, task_context, candidates=list_template_ids() + ["novel"])
        if choice != "novel":
            slots = self._extract_slots(get_template(choice), goal, task_context)
            return IntentResult("template", choice, slots, 0.6, rationale="llm classified")
        return IntentResult("novel", None, {}, 0.5, rationale="no template matched")

    def _match_templates(self, goal: str) -> tuple[str, float] | None:
        """对每个模板的 goal_patterns 做关键词/正则匹配，返回 (best_template_id, score) 或 None。
        伪代码:
          best = None
          for tpl in list_templates():
              score = max(_pattern_score(p, goal) for p in tpl.goal_patterns)
              if best is None or score > best[1]: best = (tpl.id, score)
          return best if best and best[1] > 0 else None
        """

    def _llm_classify(self, goal, task_context, candidates) -> str:
        """让 LLM 从 candidates 里选一个 id（受限分类，不自由生成）。
        伪代码:
          prompt = build_classify_prompt(goal, task_context, candidates)
          raw = self._llm_factory().complete(system_prompt=CLASSIFY_SYS, user_prompt=prompt,
                                             stream=False)   # 短输出非流式
          choice = _extract_choice(raw, candidates)   # 解析回 candidates 之一
          return choice if choice in candidates else "novel"   # 兜底 novel
        """

    def _extract_slots(self, template, goal, task_context) -> dict:
        """按 SlotSpec.source 填槽：task_context 直接取；infer 从 task 材料推断；user 留空待问。
        出参: {slot_name: value | None}。缺 required 槽位的留 None，由 Planner 决定是否追问。
        """
```

- **测试要点**：强关键词命中走模板不调 LLM；无命中走 LLM 分类；LLM 返回非候选值兜底 novel；slots 抽取 task_context 来源正确。

---

## Part E — Planner（`orchestrator/planner.py`）

```python
class Planner:
    def __init__(self, tool_registry, llm_factory, validator: "PlanValidator"):
        self._tools = tool_registry
        self._llm_factory = llm_factory
        self._validator = validator

    def from_template(self, template: WorkflowTemplate, slots: dict,
                      task_id: str, *, autonomy: int | None = None) -> Plan:
        """模板实例化为 Plan（高频主路径，确定性）。
        入参: template; slots（已抽取，可能含 None）; task_id; autonomy 覆盖。
        出参: Plan（status=DRAFT，待 validate）。
        异常: PlanningError（required slot 缺失且无法推断）。
        不变量: 不调 LLM（纯填槽），最可靠。
        """
        # 伪代码:
        missing = [s.name for s in template.slots if s.required and not slots.get(s.name)]
        if missing:
            raise PlanningError(f"missing required slots: {missing}")
        title_to_id = {st.title: _new_id() for st in template.steps}
        steps = []
        for idx, st in enumerate(template.steps):
            steps.append(PlanStep(
                id=title_to_id[st.title], plan_id="",  # 填充于下
                index=idx, title=st.title, tool_ref=st.tool_ref,
                inputs=_fill_inputs(st.inputs_template, slots, title_to_id),
                depends_on=[title_to_id[t] for t in st.depends_on_titles],
                post_checks=list(st.post_checks),
                needs_confirmation=st.needs_confirmation,
                sub_agent_scope=st.sub_agent_scope,
                granted_tools=list(st.granted_tools),   # 子 agent 步骤的最小授权随模板带下来
            ))
        plan = Plan(id=_new_id(), task_id=task_id, goal=template.title,
                    source="template", template_id=template.id, steps=steps,
                    autonomy_level=autonomy if autonomy is not None else template.default_autonomy)
        for s in plan.steps: s.plan_id = plan.id
        return plan

    def generate(self, goal: str, task_id: str, *,
                 memory_context: dict, task_context: dict, max_retries: int = 2) -> Plan:
        """novel 任务：LLM 受约束生成 Plan DAG，强制过 validator，失败重试后降级。
        入参: goal; task_id; memory_context（Phase 5 记忆）; task_context; max_retries。
        出参: Plan（status=DRAFT，已自洽但仍待外层 validate+confirm）。
        异常: PlanningError（重试耗尽仍不合法）。
        不变量: INV-1/INV-2（LLM 只选工具接数据流，inputs 引用上游 output，不算指标）。
        """
        # 伪代码:
        catalog = self._tools.catalog_for_planner()      # 紧凑工具目录（Phase 1）
        last_error = None
        for attempt in range(max_retries + 1):
            prompt = build_plan_prompt(goal, catalog, memory_context, task_context, last_error)
            raw = self._llm_factory().complete(system_prompt=PLAN_SYS, user_prompt=prompt,
                                               response_format={"type": "json_object"}, stream=False)
            try:
                plan = self._parse_plan_json(raw, goal, task_id)   # 抛 PlanningError
                problems = self._validator.validate(plan)          # 见 Part F
                if not problems:
                    return plan
                last_error = "; ".join(problems)                   # 反馈给下一轮
            except PlanningError as exc:
                last_error = str(exc)
        # 降级：重试耗尽
        raise PlanningError(f"could not generate valid plan after retries: {last_error}")

    def _parse_plan_json(self, raw: str, goal: str, task_id: str) -> Plan:
        """把 LLM 的 JSON 解析成 Plan（严格 schema），id 补齐。
        异常: PlanningError（非 JSON / 缺字段 / tool_ref 格式错）。
        伪代码:
          data = _safe_json_loads(raw)            # 失败抛 PlanningError
          validate_against_schema(data, PLAN_JSON_SCHEMA, label="plan")  # 复用 Phase 1
          steps = [_step_from_json(s, ...) for s in data["steps"]]
          ... 构造 Plan（source="generated"）...
        """
```

辅助：

```python
def _fill_inputs(template_inputs: dict, slots: dict, title_to_id: dict) -> dict:
    """把 inputs_template 里的 "{slot:x}" 替换为 slots[x]，"$ref:Title.output.f" 里的 Title 替换为 step id。"""

PLAN_SYS = """你是 MARVIS 的规划器。只能从给定工具目录选工具、把它们连成 DAG。
铁律：你不计算任何指标；指标由工具产出。你只决定调用哪些工具、参数怎么接、依赖顺序。
输出严格 JSON，匹配给定 schema。"""   # INV-1/INV-2 写进 system prompt
```

- **测试要点**：
  - `from_template` 填槽正确、缺 required 槽抛错、依赖 title→id 正确。
  - `generate` happy path 产出可过 validator 的 Plan。
  - LLM 返回非法 JSON → 重试 → 耗尽抛 `PlanningError`。
  - LLM 产出引用不存在工具 → validator 拦截 → 重试反馈。
  - PLAN_SYS 含"不计算指标"约束（INV-1）。

---

## Part F — PlanValidator（`orchestrator/validator.py`，受约束规划关键）

```python
class PlanValidator:
    def __init__(self, tool_registry):
        self._tools = tool_registry

    def validate(self, plan: Plan) -> list[str]:
        """全面校验 Plan，返回问题列表（空=合法）。不抛异常（problems 供 Planner 反馈重试）。
        入参: plan。 出参: list[str] 问题描述。
        不变量: INV-1（确定性 step 须有区间 post_check）、INV-3（join step 须确认门）。
        """
        # 伪代码:
        problems = []
        problems += self._check_tools_exist(plan)
        problems += self._check_inputs_schema(plan)
        problems += self._check_dag(plan)
        problems += self._check_ref_compatibility(plan)
        problems += self._check_join_gates(plan)
        problems += self._check_determinism_checks(plan)
        problems += self._check_subagent_grants(plan)
        return problems

    def _check_tools_exist(self, plan) -> list[str]:
        """每个 step.tool_ref 能 resolve 且 enabled。
        伪代码:
          out = []
          for s in plan.steps:
              try: self._tools.resolve(s.tool_ref)
              except (ToolNotFoundError, PluginNotFoundError) as e: out.append(f"step {s.title}: {e}")
          return out
        """

    def _check_inputs_schema(self, plan) -> list[str]:
        """step.inputs 中的字面量部分符合 tool.input_schema；**延迟值跳过，留运行时**。
        "延迟值" = `$ref:...`（上游产出，运行时才有）**或** `{slot:...}`（模板槽位，
        from_template 运行期才填实值）。后者是 Part C2 作者期 _dry_instantiate 干跑时的
        关键：未填的槽位占位**不能**拿去撞 enum/pattern/min/max（那是用户真正运行 skill 时
        才校验的实值），否则合法 skill 会被误判非法。
        伪代码:
          for s in plan.steps:
              tool = resolve(s.tool_ref)
              literal = {k:v for k,v in s.inputs.items() if not _is_deferred_input(v)}
              try: validate_against_schema(literal, _relax_required(tool.input_schema, s.inputs), label=...)
              except SchemaValidationError as e: out.append(...)
        """

    def _check_dag(self, plan) -> list[str]:
        """depends_on 无环、无悬挂引用。
        伪代码:
          ids = {s.id for s in plan.steps}
          for s in plan.steps:
              for d in s.depends_on:
                  if d not in ids: out.append(f"step {s.title} dangling dep {d}")
          if _has_cycle(plan.steps): out.append("dependency cycle detected")
        """

    def _check_ref_compatibility(self, plan) -> list[str]:
        """每个 "$ref:StepX.output.field" 指向的上游存在、field 在上游 output_schema 里。
        伪代码:
          by_id = {s.id: s for s in plan.steps}
          for s in plan.steps:
              for v in s.inputs.values():
                  if _is_ref(v):
                      up_id, field = _parse_ref(v)
                      if up_id not in by_id: out.append("ref to unknown step")
                      elif up_id not in s.depends_on: out.append("ref without dependency edge")
                      else:
                          up_tool = resolve(by_id[up_id].tool_ref)
                          if field and field not in _schema_fields(up_tool.output_schema):
                              out.append(f"ref field {field} not in upstream output")
        """

    def _check_join_gates(self, plan) -> list[str]:
        """INV-3: 任何调用 join 工具（execute_join）的 step 必须 needs_confirmation=True。
        伪代码:
          for s in plan.steps:
              if s.tool_ref.tool == "execute_join" and not s.needs_confirmation:
                  out.append(f"join step {s.title} must require confirmation (INV-3)")
        """

    def _check_determinism_checks(self, plan) -> list[str]:
        """INV-1: 产出确定性指标的 step（output_schema 含 ks/auc/psi/iv 等）必须有对应 range post_check。
        伪代码:
          for s in plan.steps:
              tool = resolve(s.tool_ref)
              metric_fields = _metric_fields_in(tool.output_schema)  # 命中 {ks,auc,psi,iv,lift,...}
              checked = {pc.spec.get("field") for pc in s.post_checks if pc.kind=="range"}
              for f in metric_fields:
                  if f not in checked:
                      out.append(f"step {s.title}: metric {f} lacks range post_check (INV-1)")
        """

    def _check_subagent_grants(self, plan) -> list[str]:
        """INV-6 配套：派子 agent 的 step（sub_agent_scope 非空）必须有非空 granted_tools，
        否则 SubAgentDispatcher 拿到空注册表、子规划必失败。每个授权工具还要能 resolve。
        伪代码:
          for s in plan.steps:
              if s.sub_agent_scope:
                  if not s.granted_tools:
                      out.append(f"sub-agent step {s.title} has empty granted_tools")
                  for t in s.granted_tools:
                      try: resolve(t)
                      except (ToolNotFoundError, PluginNotFoundError) as e:
                          out.append(f"sub-agent step {s.title}: granted tool {t.label()} {e}")
        """
```

辅助（**延迟值判定 + required 放松**，给 `_check_inputs_schema` 与 C2 作者期校验共用）：

```python
def _is_slot_placeholder(value) -> bool:
    """字符串形如 "{slot:xxx}"（模板未填槽位）。"""

def _is_deferred_input(value) -> bool:
    """运行时才确定的输入：$ref 上游产出 或 {slot:} 模板槽位。"""
    return _is_ref(value) or _is_slot_placeholder(value)

def _relax_required(input_schema: dict, step_inputs: dict) -> dict:
    """返回 input_schema 的浅拷贝，把"由延迟值（$ref/{slot:}）提供的字段"从 required 里去掉，
    使校验只看现有字面量，不为运行时才填的字段报 missing。
    不动 properties / additionalProperties（字面量字段仍按原 schema 校验类型/enum/pattern）。
    伪代码:
      deferred_keys = {k for k,v in step_inputs.items() if _is_deferred_input(v)}
      relaxed = dict(input_schema)
      relaxed["required"] = [r for r in input_schema.get("required", []) if r not in deferred_keys]
      return relaxed
    """
```

- **测试要点**（每个子检查正/反例）：未知工具/禁用工具被拦；inputs 类型错被拦；**延迟值（$ref/{slot:}）跳过字面量 schema、不误报 enum/pattern/min/max**；`_relax_required` 把延迟字段移出 required；环/悬挂依赖被拦；$ref 指向不存在步骤或缺依赖边被拦；**join step 无确认门被拦（INV-3）**；**指标 step 缺区间 post_check 被拦（INV-1）**；**sub_agent_scope step 空 granted_tools 被拦**；合法 plan 返回空列表。

---

## Part G — Reviewer（`orchestrator/reviewer.py`，双层自审）

```python
@dataclass
class FinalReview:
    goal_met: bool           # 结构性完整：所有非 SKIPPED step 都 DONE + 关键产物齐（确定性，不被 LLM 否决）
    summary: str
    open_items: list[str]    # 待人工复核项
    goal_doubt: bool = False # LLM 对"是否真正达成 goal"的存疑（仅建议；为真→计划落 REVIEW 交人工）

class Reviewer:
    def __init__(self, llm_factory):
        self._llm_factory = llm_factory

    def deterministic_check(self, step: PlanStep, output: dict) -> ReviewVerdict:
        """硬闸门：逐条跑 step.post_checks。任一不过则 passed=False。
        入参: step; output（tool 的结构化产出）。
        出参: ReviewVerdict(reviewer="deterministic")。
        异常: 无（检查失败体现在 passed=False）。
        不变量: INV-1/INV-3 在此强制。
        """
        # 伪代码:
        reasons = []
        for pc in step.post_checks:
            ok, why = _run_post_check(pc, output, step)
            if not ok: reasons.append(why)
        return ReviewVerdict("deterministic", passed=not reasons, reasons=reasons, at=_now_iso())

    def llm_critique(self, step: PlanStep, output: dict, goal: str) -> ReviewVerdict:
        """软审查：LLM 判断这步产出是否合理服务目标。不能覆盖 deterministic 失败。
        出参: ReviewVerdict(reviewer="llm_critic")。
        不变量: 仅参考，executor 不因 llm_critic.passed=False 而判失败（只记录/提示）。
        伪代码:
          prompt = build_critique_prompt(step.title, _summ(output), goal)
          raw = self._llm_factory().complete(system_prompt=CRITIC_SYS, user_prompt=prompt, stream=False)
          verdict = _parse_verdict(raw)   # {passed, reasons}
          return ReviewVerdict("llm_critic", verdict.passed, verdict.reasons, _now_iso())
        """

    def final_review(self, plan: Plan, outputs: dict[str, dict], goal: str) -> FinalReview:
        """终审：对照原始 goal 检查整体产出完整性，生成待复核项。
        入参: plan; outputs（step_id->output）; goal。
        出参: FinalReview。
        不变量: goal_met 是**确定性的结构完整判定**，不被 LLM 否决（守"确定性闸门 > LLM 自评"）；
                LLM 对达成度的存疑落到 goal_doubt（建议），由 executor 路由到 REVIEW_REQUIRED 交人工，
                而不是把 goal_met 翻成 False（避免 LLM 一句话凭空判一个结构上已完成的计划为失败）。
        伪代码:
          # 确定性部分：所有非 SKIPPED step 都 DONE 吗？关键产物在吗？
          incomplete = [s.title for s in plan.steps if s.status not in (DONE, SKIPPED)]
          # LLM 部分：总结 + 复核项 + 对"是否真正达成 goal"的存疑（解释性，不改数据/不否决 goal_met）
          summary, llm_items, goal_doubt = self._llm_summarize(goal, plan, outputs)
          return FinalReview(goal_met=not incomplete, summary=summary,
                             open_items=incomplete + llm_items, goal_doubt=goal_doubt)
        # executor 终审：goal_met=False 或 goal_doubt=True → 计划落 REVIEW_REQUIRED（人工复核），
        # 二者皆否才 DONE。
        """


def _run_post_check(pc: PostCheck, output: dict, step: PlanStep) -> tuple[bool, str]:
    """执行单条 post_check。返回 (通过?, 失败原因)。
    伪代码:
      if pc.kind == "schema":
          try: validate_against_schema(output, pc.spec["schema"], label=step.title); return True, ""
          except SchemaValidationError as e: return False, str(e)
      if pc.kind == "range":
          v = _dig(output, pc.spec["field"])
          lo, hi = pc.spec.get("min"), pc.spec.get("max")
          if v is None: return False, f"{pc.spec['field']} missing"
          if lo is not None and v < lo: return False, f"{pc.spec['field']}={v} < {lo}"
          if hi is not None and v > hi: return False, f"{pc.spec['field']}={v} > {hi}"
          return True, ""
      if pc.kind == "rowcount":
          ... 比较 output 的行数字段与 spec ...
      if pc.kind == "invariant":
          return _eval_invariant(pc.spec["rule"], output)   # 例 joined_rows<=anchor_rows (INV-3)
      if pc.kind == "nonempty":
          v = _dig(output, pc.spec["field"]); return (bool(v), f"{pc.spec['field']} empty" if not v else "")
      if pc.kind == "match_rate":
          v = _dig(output, pc.spec["field"]); lo = pc.spec["min"]
          return (v >= lo, f"match_rate {v} < {lo}" if v < lo else "")
      return False, f"unknown post_check kind {pc.kind}"
    """
```

- **测试要点**：每种 post_check kind 正/反例；`range` 对 KS=1.2 拦截（INV-1）；`invariant` 对 `joined_rows>anchor_rows` 拦截（INV-3）；`llm_critique` 失败不导致 step 失败（只记录）；`final_review` 对未完成 step 标 open_items。

---

## Part H — SubAgentDispatcher（`orchestrator/subagent.py`）

```python
class SubAgentDispatcher:
    def __init__(self, plan_repo, planner, executor_factory, tool_registry, intent_router):
        """executor_factory: 造一个受限工具视图的 executor。
        intent_router: 让子 agent 也走"模板优先"——确定性 scope（如逐表画像）命中模板就
        from_template（不调 LLM、可复现），不必每次让 LLM 现编 mini-plan。"""
        self._repo = plan_repo
        self._planner = planner
        self._executor_factory = executor_factory
        self._tools = tool_registry
        self._intent_router = intent_router

    def spawn(self, step: PlanStep, *, parent_task_id: str) -> SubAgent:
        """为某 step 派一个受限子 agent。
        入参: step（sub_agent_scope 非空）; parent_task_id。
        出参: SubAgent（已落库，status=SPAWNED）。
        不变量: 只授予 step.granted_tools，不继承父全部权限（最小授权）。
        伪代码:
          sub = SubAgent(id=_new_id(), parent_task_id=parent_task_id, parent_step_id=step.id,
                         scope=step.sub_agent_scope, granted_tools=step.granted_tools,
                         context_budget=DEFAULT_SUBAGENT_BUDGET)
          self._repo.upsert_sub_agent(sub)
          self._repo.write_audit(kind="subagent.spawn", target_ref=sub.id,
                                 detail={"scope": sub.scope, "tools": [t.label() for t in sub.granted_tools]})
          return sub
        """

    def run(self, sub: SubAgent, *, goal_inputs: dict) -> ToolResult:
        """运行子 agent：在受限工具视图下，为 scope 生成并执行一个 mini-plan。
        出参: ToolResult（汇总产物 result_ref；失败 ok=False）。
        异常: 不抛（失败收进 ToolResult，INV-6 隔离）。
        伪代码:
          try:
              restricted = _restricted_tool_registry(self._tools, sub.granted_tools)
              mini_planner = Planner(restricted, ...); mini_validator = PlanValidator(restricted)
              # 决定论优先：scope 命中模板就 from_template（不调 LLM、可复现，适合"逐表画像"
              # 这类同构 fan-out）；否则才退回 LLM generate。两条路出来的 mini_plan 都过
              # mini_validator（受限注册表，越权工具/无授权子步会被拦）。
              intent = self._intent_router.route(sub.scope, goal_inputs)
              if intent.kind == "template":
                  mini_plan = mini_planner.from_template(get_template(intent.template_id),
                                                         intent.slots, sub.parent_task_id)
              else:
                  mini_plan = mini_planner.generate(sub.scope, sub.parent_task_id,
                                                    memory_context={}, task_context=goal_inputs)
              executor = self._executor_factory(restricted)
              result = executor.run(mini_plan.id)   # 复用主执行器逻辑
              self._repo.set_sub_agent_status(sub.id, AgentStatus.RETURNED, result_ref=result.summary_ref)
              return ToolResult(ok=True, output={"result_ref": result.summary_ref}, ...)
          except Exception as exc:
              self._repo.set_sub_agent_status(sub.id, AgentStatus.FAILED)
              return ToolResult(ok=False, error=str(exc), error_kind="execution", ...)
        """
```

- **测试要点**：spawn 落库 + 审计；run 用受限工具视图（越权工具不可见）；**scope 命中模板走确定性 from_template（不调 LLM）、未命中才 generate**；子 agent 失败不抛、不影响父；最小授权验证。

---

## Part I — PlanExecutor（`orchestrator/executor.py`，可恢复 DAG 执行）

```python
@dataclass
class ExecutionResult:
    plan_id: str
    status: PlanStatus
    summary_ref: str | None
    final_review: FinalReview | None

class PlanExecutor:
    def __init__(self, plan_repo, tool_runner, reviewer, subagent_dispatcher,
                 hook_dispatcher, harness_state):
        self._repo = plan_repo
        self._runner = tool_runner
        self._reviewer = reviewer
        self._subagents = subagent_dispatcher
        self._hooks = hook_dispatcher
        self._state = harness_state

    def run(self, plan_id: str) -> ExecutionResult:
        """可恢复地推进一个 Plan：处理所有 ready 的 step，直到卡在确认门或完成。
        入参: plan_id（已 CONFIRMED 或从 AWAITING_CONFIRM 恢复）。
        出参: ExecutionResult（status 可能是 AWAITING_CONFIRM / REVIEW→DONE / FAILED）。
        异常: 不抛业务异常（失败收进 plan/step 状态）；仅 IllegalPlanTransition 表示编程错。
        不变量: INV-1/INV-3（每步过 deterministic_check）、INV-6（子 agent 隔离）、INV-8（审计）、可恢复。
        """
        # 伪代码:
        plan = self._repo.load_plan(plan_id)
        if plan.status in (PlanStatus.CONFIRMED,):
            self._state.set_plan_status(plan, PlanStatus.RUNNING)
        elif plan.status == PlanStatus.AWAITING_CONFIRM:
            self._state.set_plan_status(plan, PlanStatus.RUNNING)   # 恢复
        while True:
            step = self._next_ready_step(plan)        # 拓扑序下一个可跑的
            if step is None:
                break                                  # 没有可跑的了
            # 确认门
            if step.needs_confirmation and not self._is_confirmed(step):
                self._state.set_step_status(step, StepStatus.AWAITING_CONFIRM)
                self._state.set_plan_status(plan, PlanStatus.AWAITING_CONFIRM)
                return ExecutionResult(plan_id, PlanStatus.AWAITING_CONFIRM, None, None)
            self._execute_step(plan, step)
            plan = self._repo.load_plan(plan_id)       # 重载最新状态
            if any(s.status == StepStatus.FAILED for s in plan.steps if s.tool_ref):
                if self._is_fatal_failure(plan):
                    self._state.set_plan_status(plan, PlanStatus.FAILED)
                    return ExecutionResult(plan_id, PlanStatus.FAILED, None, None)
        # 全部 step 终态 → 终审
        return self._finalize(plan)

    def _next_ready_step(self, plan: Plan) -> PlanStep | None:
        """返回第一个 PENDING 且所有 depends_on 都 DONE 的 step；否则把它标 BLOCKED。
        伪代码:
          done = {s.id for s in plan.steps if s.status == StepStatus.DONE}
          for s in sorted(plan.steps, key=lambda x: x.index):
              if s.status in (StepStatus.PENDING, StepStatus.BLOCKED):
                  if all(d in done for d in s.depends_on): return s
          return None
        """

    def _execute_step(self, plan: Plan, step: PlanStep) -> None:
        """执行单步：实参替换 → (子agent | tool) → deterministic_check → llm_critique → 落产物。
        不变量: INV-1/INV-3 在 deterministic_check 强制；失败按 failure_policy。
        伪代码:
          self._state.set_step_status(step, StepStatus.RUNNING)
          resolved_inputs = self._resolve_refs(plan, step)   # 把 $ref 换成上游真实产物
          if step.sub_agent_scope:
              sub = self._subagents.spawn(step, parent_task_id=plan.task_id)
              result = self._subagents.run(sub, goal_inputs=resolved_inputs)
              step.sub_agent_id = sub.id
          else:
              result = self._runner.invoke(step.tool_ref, resolved_inputs, task_id=plan.task_id)
          if not result.ok:
              return self._handle_step_failure(plan, step, result)
          # 确定性硬闸门
          self._state.set_step_status(step, StepStatus.CHECKING)
          det = self._reviewer.deterministic_check(step, result.output)
          step.review_verdicts.append(det)
          if not det.passed:
              return self._handle_step_failure(plan, step,
                       ToolResult(ok=False, error="; ".join(det.reasons), error_kind="postcheck", ...))
          # 软审查（记录，不阻断）
          crit = self._reviewer.llm_critique(step, result.output, plan.goal)
          step.review_verdicts.append(crit)
          # 落产物
          output_ref = self._repo.store_step_output(step.id, result.output)
          step.output_ref = output_ref
          self._state.set_step_status(step, StepStatus.DONE)
          self._repo.update_step(step)
          self._hooks.dispatch("step.completed", {"plan_id": plan.id, "step_id": step.id},
                               task_id=plan.task_id)

    def _handle_step_failure(self, plan, step, result) -> None:
        """按 tool 的 failure_policy 处理：fail/retry/skip。
        伪代码:
          policy = resolve(step.tool_ref).failure_policy   # fail|retry|skip
          step.error = result.error
          if policy == "retry" and _retry_count(step) < MAX_STEP_RETRY:
              self._state.set_step_status(step, StepStatus.PENDING)   # 回 PENDING 重试
          elif policy == "skip":
              self._state.set_step_status(step, StepStatus.SKIPPED)
          else:
              self._state.set_step_status(step, StepStatus.FAILED)
          self._repo.update_step(step)
        """

    def _resolve_refs(self, plan: Plan, step: PlanStep) -> dict:
        """把 step.inputs 里的 "$ref:StepX.output.field" 替换为上游真实产物值。
        伪代码:
          out = {}
          for k, v in step.inputs.items():
              if _is_ref(v):
                  up_id, field = _parse_ref(v)
                  up_output = self._repo.load_step_output(up_id)
                  out[k] = _dig(up_output, field) if field else up_output
              else: out[k] = v
          return out
        """

    def _finalize(self, plan: Plan) -> ExecutionResult:
        """终审：收集 outputs → Reviewer.final_review → plan DONE/FAILED。
        伪代码:
          self._state.set_plan_status(plan, PlanStatus.REVIEW)
          outputs = {s.id: self._repo.load_step_output(s.id) for s in plan.steps if s.output_ref}
          review = self._reviewer.final_review(plan, outputs, plan.goal)
          self._state.set_plan_status(plan, PlanStatus.DONE if review.goal_met else PlanStatus.FAILED)
          self._hooks.dispatch("workflow.completed", {"plan_id": plan.id}, task_id=plan.task_id)
          summary_ref = self._repo.store_plan_summary(plan.id, review)
          return ExecutionResult(plan.id, plan.status, summary_ref, review)
        """
```

- **测试要点**（核心）：
  - 线性 plan 全 DONE → 终审 DONE。
  - 带确认门的 step → 卡 AWAITING_CONFIRM → 确认后 `run` 恢复续跑（**可恢复性**）。
  - tool 失败 + failure_policy=retry → 重试；=skip → SKIPPED 继续；=fail → plan FAILED。
  - **deterministic_check 失败（KS=1.2）→ step FAILED，不被 llm_critique 救回（INV-1）**。
  - `$ref` 实参替换正确取上游产物。
  - 子 agent step 正常派发与隔离。
  - 进程"重启"模拟：从 DB 重载 plan，`run` 从断点续跑不重复已 DONE 步骤。
  - DAG 并行依赖：A→C, B→C，A/B 都 DONE 后 C 才 ready。

---

## Part J — 持久层（`db.py` 新增 + `PlanRepository`）

### J-1 表（DDL，`connect()` 封装，单事务）

```sql
CREATE TABLE IF NOT EXISTS plans (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, goal TEXT NOT NULL,
  source TEXT NOT NULL, template_id TEXT, autonomy_level INTEGER NOT NULL,
  status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plan_steps (
  id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, idx INTEGER NOT NULL, title TEXT NOT NULL,
  tool_plugin TEXT NOT NULL, tool_name TEXT NOT NULL, tool_version TEXT,
  inputs_json TEXT NOT NULL, depends_on_json TEXT NOT NULL, post_checks_json TEXT NOT NULL,
  needs_confirmation INTEGER NOT NULL, sub_agent_scope TEXT, granted_tools_json TEXT,
  status TEXT NOT NULL, sub_agent_id TEXT, output_ref TEXT, review_json TEXT, error TEXT,
  confirmed INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS plan_step_outputs (
  step_id TEXT PRIMARY KEY, output_json TEXT NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY (step_id) REFERENCES plan_steps(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS sub_agents (
  id TEXT PRIMARY KEY, parent_task_id TEXT NOT NULL, parent_step_id TEXT,
  scope TEXT NOT NULL, granted_tools_json TEXT, context_budget INTEGER,
  status TEXT NOT NULL, result_ref TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps(plan_id);
```

### J-2 `PlanRepository`（方法逐个，均走 `connect()`）

```python
class PlanRepository:
    def __init__(self, db_path): self._db_path = db_path

    def create_plan(self, plan: Plan) -> None:
        """插入 plan + 所有 steps（单事务）。不变量: INV-8 审计；DDL/写单事务。"""

    def load_plan(self, plan_id: str) -> Plan:
        """读 plan + steps 组装成 Plan。异常: PlanNotFoundError。"""

    def update_step(self, step: PlanStep) -> None:
        """更新单个 step（status/output_ref/sub_agent_id/review/error/confirmed）。"""

    def set_plan_status(self, plan_id, status) -> None:
        """更新 plan.status + updated_at。"""

    def confirm_plan(self, plan_id) -> None:
        """plan 从 VALIDATED→CONFIRMED（经 assert_plan_transition）。"""

    def confirm_step(self, step_id) -> None:
        """标 step.confirmed=1（解开确认门）。"""

    def store_step_output(self, step_id, output: dict) -> str:
        """存结构化 output 到 plan_step_outputs，返回 output_ref "metrics:<step_id>"。"""

    def load_step_output(self, step_id) -> dict:
        """读回 step output。异常: KeyError 若无。"""

    def store_plan_summary(self, plan_id, review: FinalReview) -> str:
        """存终审摘要，返回 summary_ref。"""

    def upsert_sub_agent(self, sub: SubAgent) -> None: ...
    def set_sub_agent_status(self, sub_id, status, *, result_ref=None) -> None: ...
    def write_audit(self, **kw) -> None:
        """复用 Phase 1 audit 表。"""
```

- **测试要点**：plan 创建/读取往返；step 更新；output 存取；FK CASCADE 删 steps/outputs；状态转移经 `assert_*_transition`；可恢复（重新 `load_plan` 状态一致）。

---

## Part K — HTTP 端点（`routers/plans.py`）

```python
router = APIRouter(prefix="/api", tags=["plans"])

@router.post("/tasks/{task_id}/plans", status_code=201)
def create_plan(request, task_id: str, body: CreatePlanRequest) -> dict:
    """从用户目标创建 Plan（路由意图→规划→校验）。
    body: {goal: str, autonomy_level?: int}。
    出参: 201 + plan payload（含 steps、status=validated 或 draft、待确认项）。
    异常→HTTP: PlanningError→422; 工具缺失→409。
    伪代码:
      router_ = request.app.state.intent_router; planner = request.app.state.planner
      validator = request.app.state.plan_validator; repo = request.app.state.plan_repo
      intent = router_.route(body.goal, _task_context(request, task_id))
      if intent.kind == "template":
          plan = planner.from_template(get_template(intent.template_id), intent.slots, task_id,
                                       autonomy=body.autonomy_level)
      else:
          plan = planner.generate(body.goal, task_id,
                                  memory_context=_memory_ctx(request, task_id),
                                  task_context=_task_context(request, task_id))
      problems = validator.validate(plan)
      if problems: raise HTTPException(422, {"problems": problems})
      plan.status = PlanStatus.VALIDATED
      repo.create_plan(plan)
      return _plan_payload(plan)
    """

@router.get("/plans/{plan_id}")
def get_plan(request, plan_id: str) -> dict:
    """读 plan + steps + 各步状态/产物引用/审查结论。"""

@router.post("/plans/{plan_id}/confirm")
def confirm_plan(request, plan_id: str) -> dict:
    """用户确认整个计划（VALIDATED→CONFIRMED）。"""

@router.post("/plans/{plan_id}/run", status_code=202)
def run_plan(request, plan_id: str) -> dict:
    """后台启动执行（CONFIRMED→RUNNING）。202 + 立即返回，前端轮询进度。
    伪代码: 起后台 job 调 executor.run(plan_id)（复用现有 active job 机制）。
    """

@router.post("/plans/{plan_id}/steps/{step_id}/confirm", status_code=202)
def confirm_step(request, plan_id: str, step_id: str) -> dict:
    """确认一个卡住的 step（解确认门），并恢复执行（re-invoke executor.run）。
    伪代码: repo.confirm_step(step_id); 起后台 job 调 executor.run(plan_id) 续跑。
    """

@router.post("/plans/{plan_id}/cancel")
def cancel_plan(request, plan_id: str) -> dict:
    """取消（→CANCELLED）。运行中的子进程 tool 由 runner 超时/取消机制处理。"""
```

- **测试要点**：模板目标→201 VALIDATED；novel 目标→规划→校验；不合法→422 带 problems；confirm→run→轮询到 DONE；卡确认门→confirm_step→续跑；cancel 生效。

---

## Part L — 装配（`app.state`）

```python
# 伪代码（接 Phase 1 的 app.state）:
load_builtin_templates()
plan_repo = PlanRepository(settings.db_path)
plan_validator = PlanValidator(app.state.tool_registry)
# 用户 skill 模板：内置模板 + validator 就绪后加载（Part C2）；报告挂 app.state 供 /api/skills
app.state.skill_report = load_user_skill_templates(settings.workspace, app.state.tool_registry, plan_validator)
intent_router = IntentRouter(llm_factory=_llm_factory(settings), tool_registry=app.state.tool_registry)
planner = Planner(app.state.tool_registry, _llm_factory(settings), plan_validator)
reviewer = Reviewer(_llm_factory(settings))
harness_state = HarnessState(plan_repo)
subagent_dispatcher = SubAgentDispatcher(plan_repo, planner, _executor_factory,
                                         app.state.tool_registry, intent_router)
executor = PlanExecutor(plan_repo, app.state.tool_runner, reviewer,
                        subagent_dispatcher, app.state.hook_dispatcher, harness_state)
app.state.update(plan_repo=plan_repo, plan_validator=plan_validator, intent_router=intent_router,
                 planner=planner, reviewer=reviewer, plan_executor=executor)
app.include_router(plans_router)
```

---

## Part M — 测试计划汇总

| 文件 | 覆盖 |
|------|------|
| `tests/test_orch_contracts.py` | dataclass 往返、ref 解析、状态机转移 |
| `tests/test_orch_templates.py` | 模板注册/取/命中 |
| `tests/test_orch_skills.py` | 用户 skill 模板 parse/校验（INV-1/INV-3 拦截）/加载治理/注册防遮蔽/端点（Part C2） |
| `tests/test_orch_intent.py` | 关键词命中 / LLM 分类兜底 / 槽位抽取（LLM mock） |
| `tests/test_orch_planner.py` | from_template 填槽 / generate happy+重试+降级（LLM mock） |
| `tests/test_orch_validator.py` | 6 个子检查正反例，**INV-1/INV-3 拦截** |
| `tests/test_orch_reviewer.py` | 各 post_check kind、**KS 越界拦截**、llm_critic 不阻断 |
| `tests/test_orch_executor.py` | 线性/DAG/确认门可恢复/失败策略/确定性闸门/子 agent/重启续跑 |
| `tests/test_orch_subagent.py` | 最小授权、失败隔离 |
| `tests/test_orch_db.py` | PlanRepository 往返、CASCADE、状态机 |
| `tests/test_orch_api.py` | 端点状态码、确认→运行→续跑全流程（用 `_sample` 桩工具） |

LLM 全部用 mock（注入假 `llm_factory`），不依赖真实模型；编排逻辑的确定性必须可单测。

---

## Part N — 任务执行顺序

```text
1. A 契约+枚举         （无依赖）
2. B 状态机            （依赖 A）
3. J DB + PlanRepository（依赖 A,B；Phase 0 connect 封装）
4. C 模板             （依赖 A）
5. F PlanValidator     （依赖 A + Phase 1 ToolRegistry）
6. C2 用户 skill 模板  （依赖 C,F；声明式加载/校验/治理，复用 PlanValidator 干跑）
7. D IntentRouter      （依赖 C + LLM；自动纳入 active 用户 skill）
8. E Planner           （依赖 C,F + LLM）
9. G Reviewer          （依赖 A + LLM）
10. H SubAgentDispatcher （依赖 E,F + 占位 executor）
11. I PlanExecutor      （依赖 G,H,J + Phase 1 runner/hooks；核心，最花时间）
12. L 装配             （依赖全部；含 load_user_skill_templates）
13. K HTTP router       （依赖 D,E,F,I,L；含 /api/skills 端点）
14. M 测试补齐 + 回归
```

每项 atomic commit。Phase 2 完成标志：用 `_sample` 桩工具能跑通「目标→意图路由→（模板/生成）→校验→确认→执行（含确认门可恢复、失败策略、确定性闸门、子 agent）→终审」全链路；LLM 全 mock 下编排逻辑单测全绿；INV-1/INV-3 在 validator 和 reviewer 双处被强制。

---

*Phase 2 是平台的大脑。它把"一堆工具"变成"能理解目标、自己拆解、自我把关、可被人类叫停和确认"的 Agent。后续能力包（Phase 3+）只要遵守 Tool 契约，就能被这套编排自动调度。*
