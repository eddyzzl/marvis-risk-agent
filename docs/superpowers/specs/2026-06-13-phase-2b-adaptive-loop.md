# Phase 2B — 自适应 Loop + 上下文工程 + 能力档位（函数级 spec，含内部伪代码）

## 文档状态

- 状态：已实现并验证
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 5 节）
- 扩展对象：`2026-06-13-phase-2-orchestration.md`（在其上加自适应能力，不推翻方案 C）
- 前置依赖：Phase 2 编排核心（IntentRouter/Planner/PlanValidator/PlanExecutor/Reviewer）已实现
- 目标：给 harness 加"看结果再决定"的能力（有界 ReAct）+ 上下文工程层 + 按模型能力伸缩的档位配置。

## 设计决策（经用户确认）

- **大部分工作流是固定的**（建模/验证走纯 DAG，模板足够）；**策略分析、特征衍生**才需要"看结果再调整" → 决策点集中在这两类，不是所有 step 都放权。
- **novel 无模板任务两种模式都要、用户可选**：`plan_ahead`（一次出完整计划+确认）与 `explore`（边探索边规划）；默认值由能力档位决定。
- **失败驱动重规划默认开**（低风险高价值，有界）。
- **能力档位 `capability_tier` 决定自由度**，换模型只改配置不改架构。

## 捍卫的不变量（自适应不豁免任何一条）

- **INV-1/INV-2**：重规划/探索产出的每个 step 仍走 `PlanValidator`；指标仍由 tool 算。
- **INV-3**：动态插入的 join step 仍强制 `needs_confirmation`。
- **确定性闸门 > LLM 自评**：重规划不绕过 `Reviewer.deterministic_check`。
- **领域护栏恒定**：无论档位多高，join 确认、不可逆动作人工确认、审计、bounded 上下文都在。
- **有界**：重规划有迭代上限、无进展检测、目标锚定，防漂移/死循环。

## 模块布局（在 Phase 2 基础上新增）

```text
marvis/orchestrator/
  capability.py     CapabilityTier + 档位表 + 解析（新增）
  context/
    __init__.py
    observation.py  ObservationSummarizer：结构化产物压缩
    ledger.py       ProgressLedger：进度紧凑摘要
    budget.py       InjectionBudget：注入预算裁剪（新增）
  replanner.py      Planner.replan 的承载（或并入 planner.py）（新增）
  （扩展）contracts.py   PlanStep 加 decision_point；Plan 加 novel_mode/tier
  （扩展）executor.py    PlanExecutor 自适应主环
  （扩展）planner.py     generate 支持 explore 分段；新增 replan
```

---

## Part A — 能力档位（`orchestrator/capability.py`）

```python
@dataclass(frozen=True)
class CapabilityTier:
    name: str                    # conservative | balanced | autonomous
    default_autonomy_level: int  # 0 全确认 / 1 关键步确认 / 2 仅终审+不可逆确认
    failure_driven_replan: bool  # 失败时重规划（默认全档 True，靠迭代上限兜底）
    allow_explore_mode: bool     # novel 任务是否允许 explore（边探索边规划）
    decision_point_replan: bool  # 决策点是否触发重规划
    max_replan_iterations: int   # 单个 plan 累计重规划次数上限（防死循环）
    max_plan_depth: int          # 单次（重）规划产出的 step 上限
    explore_segment_size: int    # explore 模式每段最多规划几步

# 档位表（初始标定，基于对模型的了解；上线前需用真实 eval 校准——见文末）
TIERS: dict[str, CapabilityTier] = {
    # Flash/轻量档（如 DeepSeek V4 Flash 这类速度优先的小 MoE）：
    # 能稳定走模板主路径，但不放任自由 loop。novel 只走 plan_ahead + 人工确认。
    "conservative": CapabilityTier(
        name="conservative", default_autonomy_level=0,
        failure_driven_replan=True, allow_explore_mode=False,
        decision_point_replan=True, max_replan_iterations=2,
        max_plan_depth=8, explore_segment_size=1),
    # 强 agentic 档（如 GLM-4.5/4.6 这类 agentic 大 MoE）：决策点混合 + 有界 explore。
    "balanced": CapabilityTier(
        name="balanced", default_autonomy_level=1,
        failure_driven_replan=True, allow_explore_mode=True,
        decision_point_replan=True, max_replan_iterations=4,
        max_plan_depth=16, explore_segment_size=3),
    # 旗舰档（GPT-5 / Claude Opus 级）：novel 走完整 explore，自由重规划，仅终审+不可逆确认。
    "autonomous": CapabilityTier(
        name="autonomous", default_autonomy_level=2,
        failure_driven_replan=True, allow_explore_mode=True,
        decision_point_replan=True, max_replan_iterations=8,
        max_plan_depth=24, explore_segment_size=5),
}

DEFAULT_TIER = "balanced"

def resolve_tier(name: str | None) -> CapabilityTier:
    """解析档位，未知名回退默认档。
    出参: CapabilityTier。 异常: 无（兜底 DEFAULT_TIER）。
    伪代码: return TIERS.get(str(name or "").strip().lower(), TIERS[DEFAULT_TIER])
    """

def tier_from_settings(settings) -> CapabilityTier:
    """从平台设置读 capability_tier（与 LLM 模型配置同处持久化）。
    伪代码: return resolve_tier(getattr(settings, "capability_tier", None) or load_llm_settings(...).get("capability_tier"))
    """
```

- **不变量**：档位只调"给模型多少规划自由"，**不触碰**领域护栏（join 确认/不可逆确认/审计/INV-1）。
- **测试要点**：三档解析、未知名回退、字段单调性（autonomous 的上限 ≥ balanced ≥ conservative）。
- **标定说明**：表中数值是基于模型档位常识的初始值；`conservative` 对应 Flash 类、`balanced` 对应 GLM-4.5/4.6 类、`autonomous` 对应旗舰。**生产前应跑 eval 集**（见 Part K）校准每档的 `max_replan_iterations`/`max_plan_depth`，并把具体模型（GLM-5.1、DeepSeek V4 Flash 等）映射到档位。

---

## Part B — 契约扩展（`orchestrator/contracts.py`）

```python
# PlanStep 新增字段
@dataclass
class PlanStep:
    ...                          # 原有字段
    decision_point: bool = False # 该步完成后是否触发重规划（看结果再决定下一段）

# Plan 新增字段
@dataclass
class Plan:
    ...                          # 原有字段
    novel_mode: str = "plan_ahead"   # plan_ahead | explore（仅 source="generated" 有意义）
    tier: str = "balanced"           # 生成时锁定的能力档位名
    replan_count: int = 0            # 已累计重规划次数（防死循环）

# 探索游标（explore 模式：记录已规划到哪、下一段从哪续）
@dataclass
class ExploreCursor:
    plan_id: str
    segment_index: int           # 当前是第几段
    open_goal: str               # 尚未完成的目标描述（每段重注入，锚定）
    done: bool                   # 探索是否判定完成
```

- **测试要点**：dataclass 往返；`decision_point` 默认 False（不破坏现有纯 DAG 行为）。

---

## Part C — 上下文工程层（`orchestrator/context/`）

> 核心优势（再次强调）：MARVIS 的产物是**结构化**的，压缩天然简单——注入"算好的结构化摘要"而非原始输出。

### C-1 观察压缩（`context/observation.py`）

```python
def summarize_output(output: dict, tool_spec: "ToolSpec", *, max_chars: int = 600) -> dict:
    """把一个 tool 的结构化 output 压成可注入 LLM 的紧凑摘要（用于重规划/explore/critique）。
    入参: output（完整结构化产物）; tool_spec（拿 output_schema 知道哪些是关键字段）; max_chars。
    出参: 紧凑 dict（只留关键数值字段 + 形状信息，丢大数组/原始明细）。
    不变量: 注入的是摘要，完整 output 仍在 plan_step_outputs（output_ref 可下钻）；INV-5 不带明细。
    伪代码:
      summary = {}
      # 1) 标量/小字段直接保留（KS/AUC/PSI/行数/match_rate/fan_out 等）
      for field, spec in _scalar_fields(tool_spec.output_schema).items():
          if field in output: summary[field] = output[field]
      # 2) 大数组只留形状 + 统计（如 1000 行分箱表 → {"rows": 1000, "head": output[...][:2]}）
      for field in _array_fields(tool_spec.output_schema):
          if field in output:
              arr = output[field]
              summary[field] = {"len": len(arr), "head": arr[:2]}
      # 3) 整体裁剪到 max_chars
      return _truncate_json(summary, max_chars)
    """

def summarize_failure(error: str, error_kind: str, *, max_chars: int = 300) -> dict:
    """把失败收成紧凑摘要（供失败驱动重规划用）。
    出参: {"error_kind": ..., "error": <截断>}。
    """
```

### C-2 进度账本（`context/ledger.py`）

```python
def build_progress_ledger(plan: "Plan", step_summaries: dict[str, dict], *,
                          max_chars: int = 2000) -> str:
    """把"到目前为止做了什么"压成紧凑文本，供重规划/explore prompt 注入。
    入参: plan; step_summaries（step_id -> summarize_output 结果）; max_chars。
    出参: 紧凑进度文本（已完成步 + 关键产出 + 失败/跳过）。
    不变量: bounded；不含明细；只放对"下一步决策"有用的信息。
    伪代码:
      lines = [f"目标: {plan.goal}"]                 # 目标锚定，每次重注入
      for s in sorted(plan.steps, key=lambda x: x.index):
          if s.status == DONE:
              lines.append(f"[done] {s.title} -> {json.dumps(step_summaries.get(s.id, {}), ensure_ascii=False)}")
          elif s.status == FAILED:
              lines.append(f"[failed] {s.title}: {s.error}")
          elif s.status == SKIPPED:
              lines.append(f"[skipped] {s.title}")
      return _truncate("\n".join(lines), max_chars)
    """
```

### C-3 注入预算（`context/budget.py`）

```python
def fit_to_budget(items: list[dict], *, max_chars: int) -> list[dict]:
    """按优先级把待注入项裁到预算内（重规划时控制 prompt 体积）。
    入参: items（每个含 priority + 序列化大小）; max_chars。
    出参: 不超预算的子集（高优先先保）。
    伪代码:
      kept, used = [], 0
      for it in sorted(items, key=lambda x: -x.get("priority", 0)):
          size = len(json.dumps(it, ensure_ascii=False))
          if used + size > max_chars: continue
          kept.append(it); used += size
      return kept
    """
```

- **测试要点**：大数组 output 压成形状+head；摘要 bounded；进度账本含目标锚定 + 各步状态；预算裁剪按优先级保高优。

---

## Part D — 动态重规划（`orchestrator/planner.py` 扩展）

```python
class Planner:
    def replan(self, plan: "Plan", *, completed_summaries: dict[str, dict],
               observation: dict, reason: str, tier: "CapabilityTier") -> "Plan":
        """根据已观察结果，修订 plan 的【剩余未完成】步骤（已完成步不动）。
        入参:
          plan; completed_summaries（已完成步的压缩摘要）;
          observation（触发重规划的那步/失败的压缩摘要）;
          reason（"decision_point" | "failure" | "explore_segment"）;
          tier（控制深度/迭代上限）。
        出参: 新 Plan（已完成步保留，剩余步被替换为修订后的步；过 PlanValidator）。
        异常: ReplanError（超迭代上限 / 重试耗尽仍不合法）。
        不变量: INV-1/INV-3（修订步仍过 validator + join 确认门）；目标锚定（prompt 带原 goal）；有界。
        """
        # 伪代码:
        if plan.replan_count >= tier.max_replan_iterations:
            raise ReplanError(f"replan budget exhausted ({tier.max_replan_iterations})")
        catalog = self._tools.catalog_for_planner()
        ledger = build_progress_ledger(plan, completed_summaries)
        budget_items = fit_to_budget(
            [{"priority": 3, "ledger": ledger}, {"priority": 2, "observation": observation}], max_chars=4000)
        for attempt in range(MAX_REPLAN_PARSE_RETRY + 1):
            prompt = build_replan_prompt(plan.goal, ledger, observation, reason, catalog,
                                         remaining_titles=_remaining_titles(plan), last_error=...)
            raw = self._llm_factory().complete(system_prompt=REPLAN_SYS, user_prompt=prompt,
                                               response_format={"type": "json_object"}, stream=False)
            try:
                revised_remaining = self._parse_steps_json(raw, plan)   # 只解析"剩余步"
                if len(revised_remaining) > tier.max_plan_depth:
                    revised_remaining = revised_remaining[:tier.max_plan_depth]
                new_plan = _splice_remaining(plan, revised_remaining)    # 已完成步 + 修订剩余步
                problems = self._validator.validate(new_plan)            # INV-1/INV-3 仍强制
                if not problems:
                    new_plan.replan_count = plan.replan_count + 1
                    return new_plan
                last_error = "; ".join(problems)
            except PlanningError as exc:
                last_error = str(exc)
        raise ReplanError(f"replan could not produce valid plan: {last_error}")
```

`REPLAN_SYS`（写死约束）：

```text
你在修订一个执行计划的【剩余步骤】。已完成步骤和它们的结果在"进度"里，不要重做。
只能从工具目录选工具。你不计算任何指标。基于已观察结果，决定接下来调用哪些工具、怎么接。
不要偏离原始目标。输出严格 JSON（剩余步骤数组）。
```

- **测试要点**：决策点结果改变 → replan 产出不同剩余步；失败 → replan 插修复步（如 join fan-out → 插 dedup）；超 `max_replan_iterations` → `ReplanError`；修订步过 validator（join 仍带确认门）；目标锚定（prompt 含原 goal，LLM mock 校验）。

---

## Part E — 失败驱动重规划（默认开，有界）

逻辑并入 `PlanExecutor._handle_step_failure`（Part G），此处定义策略：

```python
def should_failure_replan(tier: "CapabilityTier", plan: "Plan", step: "PlanStep") -> bool:
    """失败时是否走重规划（而非只 retry/skip/fail）。
    伪代码:
      if not tier.failure_driven_replan: return False
      if plan.replan_count >= tier.max_replan_iterations: return False
      # 工具自身声明 failure_policy="fail" 的硬失败仍尊重（如不可恢复的契约错误）
      if resolve(step.tool_ref).failure_policy == "fail" and _is_fatal(step.error): return False
      return True
    """
```

- **不变量**：失败重规划目标不变（纯修复）；仍受 `max_replan_iterations` 约束；无进展检测（Part H）防止"修了又失败又修"死循环。
- **价值场景**：join fan-out → 自动插 dedup 步重试；缺列 → 插特征补齐步；schema 不符 → 调整上游参数。

---

## Part F — Novel 探索模式（`explore`，用户可选）

```python
class Planner:
    def generate(self, goal, task_id, *, memory_context, task_context,
                 tier: "CapabilityTier", novel_mode: str = "plan_ahead", max_retries=2) -> "Plan":
        """扩展：novel_mode="explore" 时只规划第一段（explore_segment_size 步），其余边走边规划。
        入参: 增加 tier、novel_mode。
        出参: Plan —— plan_ahead 出完整 DAG（原行为）；explore 出第一段 + novel_mode="explore"。
        不变量: explore 只在 tier.allow_explore_mode 时可用，否则回退 plan_ahead。
        伪代码:
          if novel_mode == "explore" and tier.allow_explore_mode:
              first_segment = self._generate_segment(goal, catalog, memory_context, task_context,
                                                      max_steps=tier.explore_segment_size)
              plan = _plan_from_steps(first_segment, goal, task_id, source="generated",
                                      novel_mode="explore", tier=tier.name)
              return plan
          # 否则原 plan_ahead 全量生成（含重试降级）
          return self._generate_full(goal, task_id, memory_context, task_context, max_retries)

    def next_explore_segment(self, plan: "Plan", *, completed_summaries, tier) -> tuple[list["PlanStep"], bool]:
        """explore 模式：已执行完当前段后，根据观察规划下一段，并判断是否完成。
        出参: (下一段 steps, done)。done=True 表示 LLM 判定目标已达成，无需再规划。
        不变量: 每段过 validator；段大小 ≤ explore_segment_size；累计受 max_replan_iterations 约束。
        伪代码:
          if plan.replan_count >= tier.max_replan_iterations:
              return [], True   # 探索预算耗尽，收尾
          ledger = build_progress_ledger(plan, completed_summaries)
          raw = self._llm_factory().complete(system_prompt=EXPLORE_SYS,
                    user_prompt=build_explore_prompt(plan.goal, ledger, catalog, tier.explore_segment_size),
                    response_format={"type":"json_object"}, stream=False)
          parsed = self._parse_explore_response(raw)   # {"done": bool, "steps": [...]}
          if parsed["done"]: return [], True
          steps = self._validate_segment(parsed["steps"], plan)[:tier.explore_segment_size]
          return steps, False
        """
```

- **用户可选**：API 创建 plan 时可传 `novel_mode`；缺省由 `tier` 决定（conservative→只 plan_ahead；balanced/autonomous→默认 explore 对开放任务、plan_ahead 对半结构化）。
- **测试要点**：explore 第一段生成；`next_explore_segment` 续段 + done 判定；conservative 档 explore 回退 plan_ahead；段大小受限；预算耗尽收尾。

---

## Part G — 执行器自适应主环（`orchestrator/executor.py` 扩展）

```python
class PlanExecutor:
    def run(self, plan_id: str) -> "ExecutionResult":
        """扩展：在原拓扑执行基础上加 决策点重规划 / 失败驱动重规划 / explore 续段 / 有界守护。
        不变量: 自适应不绕过确认门、确定性闸门、join 确认、迭代上限、无进展检测、目标锚定。
        """
        # 伪代码（在 Phase 2 主环上扩展）:
        plan = self._repo.load_plan(plan_id)
        tier = resolve_tier(plan.tier)
        self._state.set_plan_status(plan, RUNNING)
        while True:
            step = self._next_ready_step(plan)
            if step is None:
                # explore 模式：当前已规划步跑完但目标可能未达 → 续段
                if plan.novel_mode == "explore":
                    seg, done = self._planner.next_explore_segment(
                        plan, completed_summaries=self._summaries(plan), tier=tier)
                    if not done and seg:
                        self._append_steps(plan, seg); plan = self._repo.load_plan(plan_id); continue
                break
            if step.needs_confirmation and not self._is_confirmed(step):
                self._state.set_step_status(step, AWAITING_CONFIRM)
                self._state.set_plan_status(plan, AWAITING_CONFIRM)
                return ExecutionResult(plan_id, AWAITING_CONFIRM, None, None)
            self._execute_step(plan, step)
            plan = self._repo.load_plan(plan_id)
            last = _find(plan, step.id)
            # 决策点：成功完成后按观察重规划剩余。安全步（join 确认门 / 指标确定性门）
            # 上的 decision_point 一律忽略——这些步的下游义务是固定的（确认、区间校验），
            # 不能"看结果自由改计划"绕过守门（PlanValidator 也应在构造期就拒绝该组合）。
            if (last.status == DONE and last.decision_point and tier.decision_point_replan
                    and not _is_safety_step(last)):
                if not self._try_replan(plan, last, reason="decision_point", tier=tier):
                    pass  # 重规划失败/超预算：保持原剩余计划继续（降级）
                plan = self._repo.load_plan(plan_id); continue
            # 失败：失败驱动重规划（默认开）
            if last.status == FAILED:
                if should_failure_replan(tier, plan, last) and not self._no_progress(plan, last):
                    if self._try_replan(plan, last, reason="failure", tier=tier):
                        plan = self._repo.load_plan(plan_id); continue
                if self._is_fatal_failure(plan):
                    self._state.set_plan_status(plan, FAILED)
                    return ExecutionResult(plan_id, FAILED, None, None)
        return self._finalize(plan)

    def _try_replan(self, plan, trigger_step, *, reason, tier) -> bool:
        """调用 Planner.replan 并把修订后的剩余步落库。成功 True，失败/超预算 False。
        伪代码:
          try:
              observation = summarize_output(self._repo.load_step_output(trigger_step.id), resolve(trigger_step.tool_ref)) \
                            if reason != "failure" else summarize_failure(trigger_step.error, "...")
              new_plan = self._planner.replan(plan, completed_summaries=self._summaries(plan),
                                              observation=observation, reason=reason, tier=tier)
              self._repo.replace_remaining_steps(plan.id, new_plan)   # 原子替换未完成步 + replan_count++
              self._hooks.dispatch("plan.replanned", {"plan_id": plan.id, "reason": reason}, task_id=plan.task_id)
              return True
          except ReplanError:
              return False
        """

    def _no_progress(self, plan, failed_step) -> bool:
        """无进展检测：同一 tool_ref 连续失败 N 次（重规划又绕回同一步）→ 判定无进展，停。
        伪代码:
          recent = self._repo.recent_failed_tool_refs(plan.id, limit=NO_PROGRESS_WINDOW)
          return recent.count(failed_step.tool_ref.label()) >= NO_PROGRESS_THRESHOLD
        """


def _is_safety_step(step) -> bool:
    """安全守门步：join（INV-3 确认门）或带确定性指标 range post_check（INV-1 指标门）。
    这类步的下游义务固定，不应做"看结果再自由改计划"的决策点。
    伪代码:
      if step.tool_ref.tool == "execute_join": return True
      return any(pc.kind == "range" for pc in step.post_checks)
    """
```

> **构造期一并拦**：PlanValidator（Phase 2 Part F）的 2B 扩展应在校验时就拒绝
> `decision_point=True` 落在安全步上（`_is_safety_step` 为真），而不仅靠执行期忽略——
> 双保险，让"安全步不当决策点"在计划被确认前就暴露。

- **不变量**：
  - 决策点重规划只在 `decision_point=True` 的步后触发（其余仍纯 DAG，①的诉求）。
  - 失败重规划受 `should_failure_replan` + `_no_progress` 双重约束。
  - explore 续段受 `max_replan_iterations` 约束。
  - 重规划失败一律**降级**（保持原计划/判失败），不崩。
- **测试要点**（核心）：
  - 决策点步成功 → 触发 replan，剩余步按观察变化；非决策点步不触发。
  - **安全步（join / 指标 range 门）即使 decision_point=True 也不触发 replan；validator 构造期就拦该组合**。
  - 失败 → 失败驱动 replan 插修复步成功续跑；无进展（同步连失 N 次）→ 停。
  - explore 模式：跑完一段 → 续段 → done 收尾；超预算收尾。
  - 重规划超 `max_replan_iterations` → 降级不崩。
  - 自适应路径下确认门/确定性闸门/join 确认仍生效。
  - conservative 档：decision_point_replan 仍开但 explore 关、autonomy=0（全确认）。
  - 可恢复性：replan 后从 DB 重载续跑不重复已 DONE 步。

---

## Part H — 安全守护汇总（贯穿 D~G）

| 守护 | 机制 | 位置 |
|------|------|------|
| 迭代上限 | `replan_count >= tier.max_replan_iterations` → 停/降级 | replan / explore |
| 规划深度上限 | 修订剩余步 / 段 截到 `max_plan_depth` / `explore_segment_size` | replan / generate |
| 无进展检测 | 同 tool 连续失败 ≥ 阈值 → 停 | `_no_progress` |
| 目标锚定 | 每次 replan/explore prompt 重注入原 `plan.goal` | ledger / prompt |
| 确定性闸门不豁免 | 修订步仍过 `deterministic_check` + `PlanValidator` | replan → validator |
| 不可逆动作确认 | join/上线类 step 仍 `needs_confirmation`（档位不放宽这条） | validator + executor |
| 降级不崩 | 重规划任何失败 → 回退原计划或判失败 | `_try_replan` |

常量：`NO_PROGRESS_WINDOW=4`、`NO_PROGRESS_THRESHOLD=2`、`MAX_REPLAN_PARSE_RETRY=1`。

---

## Part I — 模板声明决策点（集中在策略/特征）

按①，决策点集中在策略分析、特征衍生类模板；建模/验证模板保持纯 DAG。

```python
# templates/feature_derivation.py（示例）
FEATURE_DERIVATION = WorkflowTemplate(
    id="feature_derivation", title="特征衍生与筛选",
    goal_patterns=("特征衍生", "特征交叉", "衍生变量"),
    steps=(
        StepTemplate("数据质量检查", ToolRef("modeling", "check_data_quality"), {...}, (),
                     (PostCheck("nonempty", {"field": "issues"}),)),
        # ↓ 决策点：看了质量报告再决定衍生策略
        StepTemplate("特征衍生", ToolRef("feature", "cross_features"),
                     {"recipe": "$ref:数据质量检查.output"}, ("数据质量检查",),
                     (PostCheck("invariant", {"rule": "row_count_unchanged"}),),
                     decision_point=True),   # ← 衍生结果决定后续筛选/编码
        StepTemplate("特征分析", ToolRef("feature", "compute_feature_metrics"),
                     {...}, ("特征衍生",),
                     (PostCheck("range", {"field": "iv", "min": 0.0}),)),
    ),
    default_autonomy=1,
)
```

- **不变量**：建模/验证模板**不**标 `decision_point`（纯 DAG，最确定）；只有策略/特征工作流标。
- **测试要点**：feature/strategy 模板含决策点；validation/modeling 模板无决策点；决策点步执行后触发 replan。

---

## Part J — DB + API 扩展

```sql
ALTER TABLE plan_steps ADD COLUMN decision_point INTEGER NOT NULL DEFAULT 0;
ALTER TABLE plans ADD COLUMN novel_mode TEXT NOT NULL DEFAULT 'plan_ahead';
ALTER TABLE plans ADD COLUMN tier TEXT NOT NULL DEFAULT 'balanced';
ALTER TABLE plans ADD COLUMN replan_count INTEGER NOT NULL DEFAULT 0;
```

`PlanRepository` 新增：
```python
def replace_remaining_steps(self, plan_id, new_plan) -> None:
    """原子替换 plan 的未完成步（DONE/SKIPPED 保留），replan_count++。单事务。"""
def append_steps(self, plan_id, steps) -> None:
    """explore 续段：追加 steps。"""
def recent_failed_tool_refs(self, plan_id, *, limit) -> list[str]:
    """最近失败 step 的 tool label 列表（无进展检测用）。"""
```

API（`routers/plans.py`）扩展：
```python
# POST /tasks/{task_id}/plans 的 body 增加可选项：
#   novel_mode?: "plan_ahead"|"explore"   缺省由 tier 决定
#   tier?: "conservative"|"balanced"|"autonomous"   缺省读平台设置
@router.get("/capability-tiers")
def list_capability_tiers(request) -> dict:
    """列出三档及其默认值（前端创建任务时供选择/展示）。"""
# 设置端点增加 capability_tier 持久化（与 LLM 模型配置同处）。
```

- **测试要点**：迁移加列默认值不破坏现有 plan；`replace_remaining_steps` 原子性 + replan_count++；append_steps；novel_mode/tier 经 API 落库。

---

## Part K — 测试计划 + 档位校准

| 文件 | 覆盖 |
|------|------|
| `tests/test_orch_capability.py` | 三档解析、回退、单调性 |
| `tests/test_orch_context.py` | 观察压缩（大数组→形状）、进度账本（目标锚定）、预算裁剪 |
| `tests/test_orch_replan.py` | 决策点 replan、失败 replan、超上限 ReplanError、修订步过 validator、目标锚定（LLM mock） |
| `tests/test_orch_explore.py` | 首段生成、续段、done 收尾、conservative 回退、段大小受限 |
| `tests/test_orch_executor_adaptive.py` | 决策点触发/非决策点不触发、失败 replan 插修复步、无进展停、explore 续段、降级不崩、确认门/确定性闸门不豁免、可恢复 |
| `tests/test_orch_plans_db_adaptive.py` | 迁移、replace_remaining_steps 原子、append_steps、recent_failed |

**档位校准（上线前，eval 驱动）**：
- 建一个 **eval 集**：覆盖固定工作流（验证/建模）+ 需适应的工作流（策略/特征衍生）+ novel 开放任务。
- 对每个候选模型（GLM-5.1、DeepSeek V4 Flash、Qwen 等）跑：模板命中率、生成 plan 合法率（过 validator）、replan 收敛率、无进展触发率、任务完成率、确定性闸门拦截数。
- 据此把每个模型**映射到档位**，并微调档位的 `max_replan_iterations`/`max_plan_depth`。
- LLM 全 mock 可单测编排逻辑；档位校准需真实模型（属 Phase 2B 验收的独立环节）。

---

## Part L — 任务执行顺序

```text
1. A capability（无依赖）
2. B 契约扩展（PlanStep.decision_point / Plan.novel_mode,tier,replan_count）
3. J DB 迁移 + Repository 扩展
4. C context（observation/ledger/budget）
5. D replan（依赖 A,C + Phase 2 Planner/Validator）
6. F explore（依赖 D）
7. E 失败重规划策略（依赖 A）
8. G executor 自适应主环（依赖 D,E,F + Phase 2 executor；核心）
9. I 模板标决策点（feature/strategy）
10. K 测试 + 档位 eval 校准
```

每项 atomic commit。Phase 2B 完成标志：决策点工作流（策略/特征）能"看结果再决定"、失败能自动重规划修复、novel 任务可选 plan_ahead/explore、全程有界（迭代上限/无进展/目标锚定）且降级不崩；`capability_tier` 配置可把同一套 harness 在 Flash 档与旗舰档间伸缩；领域护栏（INV-1/INV-3/确认/审计）在所有档位恒定；LLM mock 下编排逻辑单测全绿，档位用 eval 集校准。

---

## Part M — Agent Eval Harness（能力档位校准 + 行为回归基础设施）

> 为什么是基础设施：(1) `capability_tier` 要用真实模型 eval 才能标定到具体模型（GLM-5.1/DeepSeek V4 Flash…）；(2)"自进化 agent"长期需要回归集防退化（改了 prompt/记忆/工具后，编排质量不能掉）。这是离线、可重复、不依赖外网的 eval 设施。

### M-1 契约（`orchestrator/eval/contracts.py`）

```python
@dataclass(frozen=True)
class EvalCase:
    id: str
    goal: str                    # 用户目标自然语言
    task_context: dict           # 预置数据集/已完成步等
    kind: str                    # template_hit | plan_gen | replan | explore | guardrail
    expected: dict               # 期望：命中模板id / 必含工具 / 必触发的闸门 / 不可出现的越界
    fixtures: dict               # 离线 fixture（假数据集/假工具产出），保证可重复

@dataclass(frozen=True)
class EvalResult:
    case_id: str
    model_id: str
    tier: str
    passed: bool
    metrics: dict                # template_hit_rate/plan_valid/replan_converged/guardrail_blocked/...
    transcript_ref: str          # 完整决策轨迹（审计/调试）
```

### M-2 评测维度（确定性判分，不靠 LLM 评 LLM）

```python
def score_case(case: EvalCase, run: "PlanRunTrace") -> EvalResult:
    """对一个 eval case 的实际运行轨迹确定性判分。
    不变量: 判分是确定性规则（命中/合法/闸门触发），**不用 LLM 当裁判**（避免 LLM 评 LLM 不可靠）。
    伪代码（按 case.kind）:
      template_hit: run.plan.template_id == case.expected["template_id"]
      plan_gen:     PlanValidator.validate(run.plan) == [] and required_tools ⊆ run.tools
      replan:       run.replan_count <= cap and run.final_status == DONE   # 收敛
      explore:      run.segments <= cap and run.final_status == DONE
      guardrail:    case.expected["must_block"] in run.guardrail_hits      # 闸门确实拦住了越界
                    and not run.invented_numbers                          # INV-1 未被绕过
    """

def run_eval_suite(model_id: str, tier: str, cases: list[EvalCase], *,
                   orchestrator) -> list[EvalResult]:
    """对某模型×某档位跑整个 eval 集（离线 fixture，可重复）。
    出参: 每 case 的 EvalResult。
    不变量: 用 fixture 假工具产出（不真训模型/不真拼大数据），只测编排决策质量；可离线重复。
    """

def calibrate_tier_for_model(model_id: str, cases: list[EvalCase], *, orchestrator) -> dict:
    """对一个模型在三档各跑一遍，给出推荐档位 + 各档通过率。
    出参: {model_id, recommended_tier, per_tier:{tier: {pass_rate, guardrail_intact, ...}}}。
    用途: 把 GLM-5.1/DeepSeek V4 Flash 等映射到 conservative/balanced/autonomous。
    伪代码:
      out = {}
      for tier in ("conservative","balanced","autonomous"):
          results = run_eval_suite(model_id, tier, cases, orchestrator=orchestrator)
          out[tier] = {"pass_rate": _rate(results),
                       "guardrail_intact": all(r.passed for r in results if _is_guardrail(r))}
      # 推荐：guardrail 必须 100% intact 的前提下，pass_rate 最高的档
      recommended = _pick_recommended(out)
      return {"model_id": model_id, "recommended_tier": recommended, "per_tier": out}
    """
```

### M-3 回归用法

```python
def regression_gate(baseline: dict, current: dict, *, max_drop: float = 0.05) -> tuple[bool, list[str]]:
    """改了 prompt/记忆/工具后，对比基线 eval 通过率，跌幅超阈值则拦截（CI 用）。
    不变量: guardrail 维度（INV-1/INV-3 是否被绕过）**零容忍**，任何下降直接 fail。
    伪代码:
      regressions = []
      if current["guardrail_pass_rate"] < baseline["guardrail_pass_rate"]:
          regressions.append("GUARDRAIL REGRESSION (zero tolerance)")
      if baseline["overall_pass_rate"] - current["overall_pass_rate"] > max_drop:
          regressions.append(f"pass_rate dropped > {max_drop}")
      return (not regressions, regressions)
    """
```

- **eval 集内容**（覆盖你最初设想的工作流）：固定流（模型验证/标准建模）、需适应流（策略分析/特征衍生决策点）、novel 开放任务、**护栏用例**（故意诱导 LLM 算指标/静默 join → 必须被闸门拦住）。
- **离线 + 可重复**：用 fixture 假工具产出，不真训模型、不真拼大数据，只测编排决策；契合本地优先。
- **测试要点**：`score_case` 各 kind 判分正确；`calibrate_tier_for_model` 在 guardrail 100% 前提下选最高 pass_rate 档；`regression_gate` 对 guardrail 下降零容忍。
- **任务顺序**：并入 Part L 之后（M1 契约 → M2 判分/跑集 → M3 回归门 → 建初始 eval 集）。

---

## 与模型选择的关系（总结）

- **架构对模型无关**：换 GLM-5.1 / DeepSeek V4 Flash / Qwen，只改 `capability_tier`，不改 harness。
- **Flash 类（conservative）**：模板主路径 + 决策点 replan，关 explore，autonomy=0。便宜、快、稳，契合本地优先私有化。
- **GLM-4.5/4.6 类（balanced，默认）**：决策点混合 + 有界 explore + 失败重规划。
- **旗舰类（autonomous）**：novel 走完整 explore，自由重规划，仅终审+不可逆确认。
- **三档共守领域护栏**：INV-1 指标平台算、INV-3 join 确认、不可逆动作确认、审计、bounded 上下文——这些是风控合规要求，与模型多强无关。
- **档位数值是初始标定，需用 eval 集对具体模型校准后再上生产。**

---

*Phase 2B 把"loop 要不要、给多少"从一次性架构赌注，变成一个可按模型伸缩、可被 eval 校准、且永远守得住风控护栏的旋钮。这样你选 GLM-5.1 还是 DeepSeek V4 Flash，都不影响这套 harness 能用——只影响旋钮拧到哪一档。*
