# Phase 5 — 记忆自进化（函数级 spec，含内部伪代码）

## 文档状态

- 状态：待实施
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 8 节）
- 前置依赖：现有 V1.1 `agent_memory`（store/retrieval/extractors/policy/prompting）、Phase 1（Hook 系统可触发后台蒸馏）
- 目标：在 V1.1 记忆基础上加「**蒸馏 / 经验固化 / 置信度演进**」层——让 Agent 跨任务把零散记忆压缩成高置信经验，自动总结、自我进化，且全程可审计、可回滚。**铁律：自进化只影响解释/建议/口径，绝不改确定性指标。**

## 捍卫的不变量

- **INV-4**：蒸馏记忆只辅助解释、参数建议、风险提醒、历史对比、报告口径；不改 KS/AUC/PSI/分数一致性等确定性结果。
- **INV-5**：蒸馏过程不引入禁存内容（原始样本/明细/源码/密钥/未脱敏报告）；蒸馏输出过同样的安全过滤。
- **INV-8**：蒸馏的提取、固化、取代、使用全留审计链（source_memory_ids、support_count、superseded 关系）。
- 自进化是**后台离线**过程，不阻塞任务主流程；失败不影响验证。

## 设计哲学

V1.1 记忆是「单条原始记忆」（用户偏好、字段口径、验证坑点、任务经验、模型经验）。问题：同一事实在多个任务里重复记录、置信度无法累积、prompt 注入时体积膨胀。

自进化层做三件事：
1. **蒸馏（distill）**：把同类多条原始记忆压缩成一条高置信 `MemoryDistillation`（如"该机构 idcard 字段常叫 id_md5"出现 5 次 → 固化为高置信字段口径）。
2. **演进（evolve）**：新证据出现时，旧蒸馏 `superseded=True`，保留审计链，不物理删除。
3. **优先检索（prefer）**：检索时优先返回高置信蒸馏，降低 prompt 体积，原始记忆作为下钻证据。

## 模块布局

```text
marvis/agent_memory/
  （现有）store.py / retrieval.py / extractors.py / policy.py / prompting.py
  distillation.py    ← 新增：蒸馏引擎
  evolution.py       ← 新增：置信度演进 + 取代关系
  consolidation.py   ← 新增：后台固化调度（Hook 触发）
marvis/db.py   新增 memory_distillations 表 + 扩展 AgentMemoryStore
marvis/routers/memory.py   扩展：蒸馏查看/审计/回滚端点
```

无新依赖（复用现有 agent_memory 基础设施 + LLM client 做摘要）。

---

## Part A — 契约（`agent_memory/distillation.py` 顶部）

```python
@dataclass(frozen=True)
class MemoryDistillation:
    id: str
    category: str            # user_preference|field_convention|validation_pitfall|task_experience|model_experience
    scope_key: str           # 蒸馏归并键（如 "field_convention:idcard:机构A"）——同 key 的原始记忆被蒸馏到一起
    distilled_summary: str    # 压缩后的经验（脱敏、bounded 长度）
    structured: dict          # 结构化字段（如 {field_role:"idcard", aliases:["id_md5"], support:5}）
    source_memory_ids: tuple[str, ...]   # 被蒸馏的原始记忆 id（审计链）
    support_count: int        # 支持该结论的任务/记忆数
    confidence: str           # high|medium|low
    superseded_by: str | None # 被哪条更新蒸馏取代（None=当前有效）
    created_at: str
    updated_at: str

# 置信度阈值（support_count → confidence）
CONFIDENCE_THRESHOLDS = {"high": 4, "medium": 2}   # >=4 高, >=2 中, else 低
MAX_DISTILLED_SUMMARY_CHARS = 400   # bounded（防 prompt 膨胀，呼应 CODE_REVIEW P1-19）
```

- **测试要点**：dataclass 往返；`scope_key` 唯一归并；置信度阈值映射。

---

## Part B — 蒸馏引擎（`agent_memory/distillation.py`）

```python
class DistillationEngine:
    def __init__(self, store: "AgentMemoryStore", llm_factory, policy):
        self._store = store
        self._llm_factory = llm_factory
        self._policy = policy          # 复用 V1.1 安全过滤

    def distill_category(self, category: str) -> list[MemoryDistillation]:
        """对某类原始记忆按 scope_key 分组，每组蒸馏成一条 MemoryDistillation。
        入参: category（五类之一）。
        出参: 新建/更新的蒸馏列表。
        异常: 不抛（单组失败跳过并记录）。
        不变量: INV-4（只压缩解释性经验）、INV-5（输出过 policy 安全过滤）。
        """
        # 伪代码:
        entries = self._store.list_entries(category=category, status="active", limit=2000)
        groups = self._group_by_scope(entries)           # scope_key -> [原始记忆]
        results = []
        for scope_key, members in groups.items():
            if len(members) < 1: continue
            try:
                distilled = self._distill_group(category, scope_key, members)
                if distilled is not None:
                    results.append(self._upsert_distillation(distilled))
            except Exception as exc:
                self._store.write_audit(kind="distill.skip", target_ref=scope_key, detail={"error": str(exc)})
        return results

    def _group_by_scope(self, entries: list[dict]) -> dict[str, list[dict]]:
        """按 scope_key 归并。scope_key 由 category + 结构化字段派生。
        伪代码:
          groups = {}
          for e in entries:
              key = self._scope_key_for(e)        # 如 "field_convention:idcard:" + e.scope
              groups.setdefault(key, []).append(e)
          return groups
        """

    def _distill_group(self, category, scope_key, members) -> MemoryDistillation | None:
        """把一组同 scope 的原始记忆压缩成一条经验。
        - 结构化字段：对可聚合字段做确定性合并（如 aliases 取并集、support=len）。
        - 摘要文本：用 LLM 压缩成一句话（仅措辞，不引入新事实）；LLM 不可用则用模板兜底。
        不变量: INV-1/INV-4——结构化合并是确定性的；LLM 只润色摘要不造事实。
        伪代码:
          structured = self._merge_structured(category, members)    # 确定性合并
          support = len(members)
          confidence = self._confidence_from_support(support)
          summary = self._summarize(category, scope_key, members, structured)  # LLM 润色 + 模板兜底
          summary = summary[:MAX_DISTILLED_SUMMARY_CHARS]
          candidate = MemoryDistillation(id=_new_id(), category=category, scope_key=scope_key,
                        distilled_summary=summary, structured=structured,
                        source_memory_ids=tuple(m["id"] for m in members),
                        support_count=support, confidence=confidence,
                        superseded_by=None, created_at=_now(), updated_at=_now())
          # INV-5: 输出过安全过滤，命中敏感则丢弃
          verdict = self._policy.classify_distillation(candidate)
          return candidate if verdict.allowed else None
        """

    def _merge_structured(self, category, members) -> dict:
        """确定性合并结构化字段（不经 LLM）。
        伪代码（按 category）:
          field_convention → {field_role, aliases: 并集, scope}
          model_experience → {model_name, scopes: 并集, metric_ranges: min/max(ks/auc/psi), source_task_ids}
          validation_pitfall → {pitfall_type, fixes: 去重列表}
          user_preference → {preference_kind, statements: 去重}
          task_experience → {outcome_tags: 计数}
        """

    def _summarize(self, category, scope_key, members, structured) -> str:
        """LLM 把结构化经验润色成一句话；LLM 不可用 → 模板兜底。
        不变量: prompt 明确"只润色措辞，不得引入 members 之外的事实"（INV-2）。
        伪代码:
          try:
              raw = self._llm_factory().complete(system_prompt=DISTILL_SYS,
                        user_prompt=build_distill_prompt(structured, members), stream=False)
              return _sanitize(raw)         # 复用 V1.1 task_id 脱敏
          except LLMClientError:
              return _template_summary(category, structured)    # 确定性模板兜底
        """

    def _confidence_from_support(self, support: int) -> str:
        if support >= CONFIDENCE_THRESHOLDS["high"]: return "high"
        if support >= CONFIDENCE_THRESHOLDS["medium"]: return "medium"
        return "low"
```

`DISTILL_SYS` 系统提示（写死 INV-2 约束）：

```text
你在压缩 MARVIS 的历史记忆。只能基于给定的结构化字段和原始记忆措辞，输出一句话经验。
禁止引入任何未在输入中出现的事实、数字或结论。不要输出任务 ID。
```

- **测试要点**：同 scope 多条记忆蒸馏成一条；结构化合并确定性（aliases 并集正确）；support→confidence 映射；LLM 不可用走模板兜底；敏感内容被 policy 拦截丢弃（INV-5）；摘要长度 bounded。

---

## Part C — 置信度演进与取代（`agent_memory/evolution.py`）

```python
class EvolutionManager:
    def __init__(self, store: "AgentMemoryStore"):
        self._store = store

    def upsert_with_evolution(self, candidate: MemoryDistillation) -> MemoryDistillation:
        """落库新蒸馏，若同 scope_key 已有有效蒸馏则建立取代关系（旧的 superseded）。
        入参: candidate 新蒸馏。
        出参: 落库后的蒸馏（可能 id 复用或新建）。
        不变量: INV-8——不物理删旧蒸馏，标 superseded_by 保留审计链。
        伪代码:
          existing = self._store.get_active_distillation(candidate.scope_key)
          if existing is None:
              self._store.create_distillation(candidate)
              self._store.write_audit(kind="distill.create", target_ref=candidate.id,
                                      detail={"scope": candidate.scope_key, "support": candidate.support_count})
              return candidate
          # 已有：仅当新证据更强（support 增加或字段变化）才取代
          if self._is_meaningful_update(existing, candidate):
              self._store.set_superseded(existing.id, by=candidate.id)
              self._store.create_distillation(candidate)
              self._store.write_audit(kind="distill.supersede",
                  target_ref=candidate.id, detail={"supersedes": existing.id,
                  "old_support": existing.support_count, "new_support": candidate.support_count})
              return candidate
          # 无实质变化：只更新 support_count/updated_at
          self._store.update_distillation_support(existing.id, candidate.support_count)
          return existing

    def _is_meaningful_update(self, old, new) -> bool:
        """是否值得取代：support 跨置信档 / 结构化字段变化（新别名、指标区间扩大）。
        伪代码:
          if old.confidence != new.confidence: return True
          if set(new.structured.get("aliases",[])) != set(old.structured.get("aliases",[])): return True
          return False
        """

    def rollback(self, distillation_id: str) -> None:
        """回滚一条蒸馏：标 superseded 撤销，恢复其前驱为 active（人工审计用）。
        异常: MemoryError（找不到）。
        不变量: INV-8——回滚也留审计。
        伪代码:
          d = self._store.get_distillation(distillation_id)
          predecessor = self._store.find_superseded_by(distillation_id)
          if predecessor: self._store.clear_superseded(predecessor.id)
          self._store.set_status_distillation(distillation_id, "rolled_back")
          self._store.write_audit(kind="distill.rollback", target_ref=distillation_id)
        """
```

- **测试要点**：首次蒸馏直接建；同 scope 更强证据建立 supersede 链；无实质变化只更 support；旧蒸馏不被物理删；rollback 恢复前驱；审计链完整。

---

## Part D — 后台固化调度（`agent_memory/consolidation.py`）

```python
class ConsolidationScheduler:
    def __init__(self, distillation_engine, evolution_manager, store):
        self._distill = distillation_engine
        self._evolve = evolution_manager
        self._store = store

    def on_event(self, event: str, payload: dict) -> None:
        """Hook 回调：在合适的平台事件后触发后台蒸馏（不阻塞主流程）。
        入参: event（validation.completed|report.after_generate|memory.after_save 等）; payload。
        不变量: 后台异步；失败不影响任务（INV-6 隔离思想）。
        伪代码:
          if event not in CONSOLIDATION_TRIGGERS: return
          category = _category_for_event(event)
          # 节流：同 category 距上次蒸馏不足 N 分钟则跳过（避免频繁触发）
          if self._recently_consolidated(category): return
          self._run_async(lambda: self._consolidate(category))

    def _consolidate(self, category: str) -> None:
        """跑一轮蒸馏 + 演进。
        伪代码:
          candidates = self._distill.distill_category(category)
          for c in candidates:
              self._evolve.upsert_with_evolution(c)
          self._store.mark_consolidated(category, at=_now())
        """

    def consolidate_all(self) -> dict:
        """手动全量蒸馏（管理端点/定时触发）。出参: {category: count}。"""

CONSOLIDATION_TRIGGERS = {"validation.completed", "report.after_generate", "memory.after_save"}
```

接线：在 Phase 1 的 `HookDispatcher` 注册一个内置 hook，把这些事件转给 `ConsolidationScheduler.on_event`。后台执行复用现有 active job 机制或线程池（失败隔离）。

- **测试要点**：触发事件→蒸馏跑；节流生效（短时间内不重复）；后台失败不抛到主流程；`consolidate_all` 返回各类计数。

---

## Part E — 检索集成（扩展 `agent_memory/retrieval.py`）

```python
def retrieve_with_distillations(store, query_context: dict, *, limit: int = 6) -> list[dict]:
    """检索时优先返回高置信蒸馏，原始记忆作为下钻证据。
    入参: store; query_context（任务上下文、字段、模型场景）; limit。
    出参: bounded memory context packets（蒸馏优先 + 必要原始记忆）。
    不变量: INV-4（注入的记忆只辅助）；bounded 体积（呼应 CODE_REVIEW P1-19 prompt 膨胀）。
    伪代码:
      distillations = store.search_distillations(query_context, active_only=True, limit=limit)
      packets = []
      for d in distillations:
          packets.append(_distillation_packet(d))     # 含 id/category/confidence/support/source_memory_ids
          if len(packets) >= limit: break
      # 蒸馏不足时用原始记忆补，但总量 bounded
      if len(packets) < limit:
          raw = retrieve_relevant_memories(store.list_entries(...), query_context, limit=limit-len(packets))
          packets += [_raw_packet(r) for r in raw]
      return packets[:limit]
    """

def _distillation_packet(d: MemoryDistillation) -> dict:
    """蒸馏→prompt packet（带审计可下钻字段）。
    伪代码:
      return {"kind":"distillation", "id": d.id, "category": d.category,
              "summary": d.distilled_summary, "confidence": d.confidence,
              "support_count": d.support_count, "source_task_ids": _tasks_of(d.source_memory_ids)}
    """
```

- **不变量**：注入 prompt 的蒸馏摘要已 bounded（≤400 字符），summary 经 task_id 脱敏（呼应 CODE_REVIEW P2-3）。
- **测试要点**：高置信蒸馏优先；蒸馏不足用原始补；总量 bounded；packet 带 source 审计字段；低置信蒸馏不用于历史对比（INV-4，roadmap 对比置信度规则）。

---

## Part F — 持久层（`db.py` + `AgentMemoryStore` 扩展）

```sql
CREATE TABLE IF NOT EXISTS memory_distillations (
  id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  distilled_summary TEXT NOT NULL,
  structured_json TEXT NOT NULL,
  source_memory_ids_json TEXT NOT NULL,
  support_count INTEGER NOT NULL,
  confidence TEXT NOT NULL,
  superseded_by TEXT,
  status TEXT NOT NULL DEFAULT 'active',   -- active|rolled_back
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_distill_scope ON memory_distillations(scope_key, status);
CREATE INDEX IF NOT EXISTS idx_distill_category ON memory_distillations(category, status);
```

`AgentMemoryStore` 新增方法（均走 `connect()` 封装）：

```python
def create_distillation(self, d: MemoryDistillation) -> None: ...
def get_distillation(self, distillation_id) -> MemoryDistillation: ...     # 异常 MemoryNotFound
def get_active_distillation(self, scope_key) -> MemoryDistillation | None: ...
def set_superseded(self, distillation_id, *, by: str) -> None: ...
def clear_superseded(self, distillation_id) -> None: ...
def update_distillation_support(self, distillation_id, support_count) -> None: ...
def set_status_distillation(self, distillation_id, status) -> None: ...
def search_distillations(self, query_context, *, active_only=True, limit=6) -> list[MemoryDistillation]:
    """按 category + scope 关键词 + 置信度排序检索（高置信优先）。"""
def find_superseded_by(self, distillation_id) -> MemoryDistillation | None: ...
def mark_consolidated(self, category, *, at) -> None: ...
```

- **测试要点**：蒸馏 CRUD 往返；supersede 链查询；search 高置信优先；structured_json 反序列化；status 流转。

---

## Part G — HTTP 端点（扩展 `routers/memory.py`）

```python
@router.get("/api/memory/distillations")
def list_distillations(request, category: str | None = None, include_superseded: bool = False) -> dict:
    """列蒸馏（管理/审计视图）。出参: {distillations:[{id,category,summary,confidence,support_count,superseded_by}]}。"""

@router.get("/api/memory/distillations/{distillation_id}")
def get_distillation_detail(request, distillation_id: str) -> dict:
    """蒸馏详情 + 下钻：source 原始记忆、supersede 链、审计事件。"""

@router.post("/api/memory/distillations/{distillation_id}/rollback")
def rollback_distillation(request, distillation_id: str) -> dict:
    """人工回滚一条蒸馏（恢复前驱）。"""

@router.post("/api/memory/consolidate")
def trigger_consolidation(request, category: str | None = None) -> dict:
    """手动触发蒸馏（全部或指定类）。出参: {consolidated: {category: count}}。"""
```

- **不变量**：与 roadmap V1.1 前端规则一致——记忆管理入口在设置/审计视图，不在任务顶部常驻灰块。
- **测试要点**：列表/详情/回滚/手动触发端点；详情可下钻 source + 审计；回滚改变 active 状态。

---

## Part H — 前端（记忆审计视图，`static/js/views/memory.js`）

- 在设置/审计区（**非任务顶部常驻**，遵守 roadmap V1.1 前端规则）加"记忆进化"面板：
  - 蒸馏列表（按 category，显示置信度、support、是否被取代）。
  - 点开下钻：source 原始记忆、supersede 链、审计事件。
  - 回滚按钮（人工干预）。
- Agent 对话里，引用蒸馏的消息带可展开"记忆引用"，显示 distillation id、category、confidence、source task_ids（复用 V1.1 引用 UI）。
- **测试要点**（`test_frontend_static_v2.py` 风格）：面板渲染、下钻、回滚交互；对话引用展开显示审计字段。

---

## Part I — 测试计划汇总

| 文件 | 覆盖 |
|------|------|
| `tests/test_memory_distillation.py` | 分组、结构化确定性合并、support→confidence、LLM 兜底、**敏感拦截** |
| `tests/test_memory_evolution.py` | supersede 链、无实质更新只更 support、rollback、**不物理删** |
| `tests/test_memory_consolidation.py` | 事件触发、节流、后台失败隔离、consolidate_all |
| `tests/test_memory_retrieval_distill.py` | 高置信优先、bounded、下钻字段、**低置信不用于对比** |
| `tests/test_memory_distill_db.py` | 蒸馏 CRUD、supersede 查询、search 排序 |
| `tests/test_memory_distill_api.py` | 列表/详情/回滚/手动触发端点 |
| `tests/test_memory_determinism_guard.py` | **蒸馏不改任何确定性指标（INV-4）—— 跑验证前后指标一致** |

`test_memory_determinism_guard.py` 是关键护栏：构造带蒸馏记忆的任务，断言验证指标（KS/AUC/PSI）与无记忆时**逐位一致**，证明记忆只进 prompt 解释、不进计算。

---

## Part J — 任务执行顺序

```text
1. A 契约              （无依赖）
2. F DB + store 扩展    （依赖 A + 现有 agent_memory）
3. B 蒸馏引擎          （依赖 A,F + 现有 policy/LLM）
4. C 演进/取代          （依赖 A,F）
5. D 后台固化调度       （依赖 B,C + Phase 1 Hook）
6. E 检索集成          （依赖 F + 现有 retrieval）
7. G HTTP 端点          （依赖 B,C,F）
8. H 前端审计视图       （依赖 G + Phase 0 前端模块化）
9. I 测试 + 回归（含确定性护栏）
```

每项 atomic commit。Phase 5 完成标志：验证完成等事件后台触发蒸馏，同类记忆压缩成高置信经验、新证据演进取代旧经验并留审计链，检索优先高置信蒸馏且 bounded，全程可在审计视图查看/回滚；**确定性护栏测试证明蒸馏不改任何验证指标（INV-4）**。

---

*Phase 5 让 Agent 真正"越用越聪明"——但聪明只体现在解释、建议、历史对比和报告口径上，确定性指标永远由平台代码说了算。这正是 MARVIS 区别于通用 Agent 的地方：可治理、可审计、可回滚的进化。*
