# Phase 8 — 联网学习草稿区 + 治理（函数级 spec，含内部伪代码）

## 文档状态

- 状态：已实现并验证
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 10、15.3 节）
- 前置依赖：Phase 1（Tool Runtime / 子进程 runner / PluginRegistry）、Phase 2（编排，理解为什么草稿不能被 Planner 自动选）
- 目标：交付「不会的 → 联网学习 → 写脚本 → 草稿区临时运行 → 人工确认转正」闭环，且治理严密：草稿默认不进正式工具库、不被 Planner 自动选用。

## 捍卫的不变量

- **INV-7**：联网学习/现场编写的脚本默认进草稿区；**人工转正前 Planner 不得自动选用**（不进 `catalog_for_planner`）。
- **离线契约**：联网学习仅在有网时可用；离线环境下 `web_search`/`fetch_url` 优雅失败，引导"人在外部产出工具→上传"路径（Phase 1 的 install_plugin）。
- **INV-6**：草稿脚本执行同样走子进程隔离（不因为是草稿就放松隔离）。
- **INV-8**：草稿编写、运行、转正、拒绝全留审计。
- **转正闸门**：草稿转正必须有 input/output schema、determinism 声明、最小测试且测试通过（合规要求，风控场景）。

## 模块布局

```text
marvis/drafts/
  __init__.py
  contracts.py     DraftTool / DraftRun / PromotionCheck / LearningNote
  errors.py
  web_search.py    web_search / fetch_url（仅联网，离线优雅降级）
  learning.py      distill_learning（把网页内容蒸馏成结构化学习笔记）
  authoring.py     draft_script（LLM 按 Tool 契约模板写脚本）
  sandbox.py       run_draft（子进程隔离临时运行 + 审计）
  promotion.py     validate_for_promotion / promote_draft（转正闸门）
  registry.py      DraftRegistry（与正式 registry 隔离，Planner 不可见）
  tools.py
marvis/db.py   新增 draft_tools / draft_runs 表
marvis/routers/drafts.py
marvis/static/js/views/drafts.js
```

新增依赖：`httpx>=0.24`（联网，可选）；离线部署可不装，相关 tool 自检缺失即降级。

---

## Part A — 契约（`drafts/contracts.py`）

```python
@dataclass(frozen=True)
class LearningNote:
    id: str
    query: str
    sources: tuple[str, ...]        # 来源 URL（审计）
    distilled: str                  # 结构化学习笔记（脱敏、bounded）
    created_at: str

@dataclass(frozen=True)
class DraftTool:
    id: str
    task_id: str                    # 草稿绑定到产生它的任务
    name: str                       # 草稿工具名
    summary: str
    code: str                       # 脚本源码（单个 tool 函数）
    input_schema: dict
    output_schema: dict
    determinism: str                # deterministic|stochastic
    source: str                     # web_learning|hand_written|llm_generated
    learning_note_id: str | None
    status: str                     # draft|tested|promoted|rejected
    created_at: str

@dataclass(frozen=True)
class DraftRun:
    id: str
    draft_id: str
    task_id: str
    inputs_hash: str
    ok: bool
    output: dict | None
    error: str | None
    at: str

@dataclass(frozen=True)
class PromotionCheck:
    passed: bool
    problems: tuple[str, ...]       # 未通过原因
    test_result: dict | None        # 最小测试运行结果
```

- **测试要点**：dataclass 往返；草稿状态机 draft→tested→promoted/rejected。

---

## Part B — 联网搜索（`drafts/web_search.py`，仅联网，离线降级）

```python
def network_available() -> bool:
    """探测是否有网（短超时连接探测）。
    伪代码:
      try: httpx.head(PROBE_URL, timeout=2); return True
      except Exception: return False
    """

def web_search(query: str, *, max_results: int = 5) -> list[dict]:
    """联网搜索（仅有网时）。离线优雅失败。
    入参: query; max_results。
    出参: [{title, url, snippet}]。
    异常: OfflineError（无网时，message 引导"外部产出工具→上传"）。
    不变量: 离线契约——无网即明确报错，不假装能搜。
    伪代码:
      if not network_available():
          raise OfflineError("无网络：请在有网环境产出工具后，通过 插件上传 导入（见 Phase 1 install_plugin）")
      if httpx is None:
          raise OfflineError("httpx 未安装（离线部署）；改用外部产出+上传路径")
      resp = httpx.get(SEARCH_ENDPOINT, params={"q": query, "n": max_results}, timeout=15)
      return _parse_search_results(resp.json())[:max_results]
    """

def fetch_url(url: str, *, max_bytes: int = 500_000) -> str:
    """抓取网页正文（仅联网）。
    异常: OfflineError; FetchError（4xx/5xx/超大）。
    不变量: 限制大小防滥用；只抓文本。
    伪代码:
      if not network_available(): raise OfflineError(...)
      resp = httpx.get(url, timeout=20, follow_redirects=True)
      if resp.status_code >= 400: raise FetchError(f"HTTP {resp.status_code}")
      text = _extract_main_text(resp.text)[:max_bytes]   # 去 HTML 标签，留正文
      return text
    """
```

`SEARCH_ENDPOINT`/`PROBE_URL` 可配置（私有化环境可指向内网搜索代理）。

- **测试要点**：mock 无网→`OfflineError` 带引导文案；mock 有网→返回结果；httpx 缺失→降级报错；fetch 超大/4xx→错误；离线部署不装 httpx 也能 import 模块（lazy）。

---

## Part C — 学习蒸馏（`drafts/learning.py`）

```python
def distill_learning(query: str, contents: list[str], sources: list[str], *,
                    llm_factory) -> LearningNote:
    """把抓取的网页内容蒸馏成结构化学习笔记（供 authoring 写脚本参考）。
    入参: query; contents 网页正文列表; sources URL; llm_factory。
    出参: LearningNote（bounded、记来源）。
    不变量: INV-8——记来源 URL；INV-5——蒸馏输出不落敏感/超长原文。
    伪代码:
      joined = "\n---\n".join(c[:5000] for c in contents)[:20000]
      raw = llm_factory().complete(system_prompt=LEARN_SYS,
                user_prompt=build_learn_prompt(query, joined), stream=False)
      distilled = _sanitize(raw)[:MAX_NOTE_CHARS]
      return LearningNote(id=_new_id(), query=query, sources=tuple(sources),
                          distilled=distilled, created_at=_now())
    """
```

`LEARN_SYS`：「把资料压成可操作的实现要点（步骤/公式/库用法），不要复制大段原文，标注关键 API。」

- **测试要点**：蒸馏产 bounded 笔记；记来源；LLM mock；超长内容截断。

---

## Part D — 脚本编写（`drafts/authoring.py`）

```python
TOOL_TEMPLATE = '''
def {entrypoint}(inputs: dict, ctx) -> dict:
    """{summary}"""
    # inputs 已过 input_schema；返回必须符合 output_schema
    {body}
    return {return_expr}
'''

def draft_script(task_id: str, goal: str, *, learning_note: LearningNote | None,
                llm_factory) -> DraftTool:
    """LLM 按 Tool 契约模板编写一个草稿工具（含 input/output schema）。
    入参: task_id; goal 要实现的能力; learning_note 联网学习笔记（可选）; llm_factory。
    出参: DraftTool（status="draft"，未测试未注册）。
    异常: AuthoringError（LLM 输出无法解析成合法脚本+schema）。
    不变量: 产物只进草稿区；强制要求声明 input/output schema 和 determinism（转正闸门前置）。
    伪代码:
      prompt = build_authoring_prompt(goal, learning_note, TOOL_TEMPLATE)
      raw = llm_factory().complete(system_prompt=AUTHOR_SYS, user_prompt=prompt,
                response_format={"type":"json_object"}, stream=False)
      spec = _safe_json_loads(raw)   # {name, summary, code, input_schema, output_schema, determinism}
      _assert_keys(spec, REQUIRED_DRAFT_KEYS)
      _assert_valid_jsonschema(spec["input_schema"]); _assert_valid_jsonschema(spec["output_schema"])
      _static_safety_scan(spec["code"])   # 禁危险调用（见下）
      return DraftTool(id=_new_id(), task_id=task_id, name=spec["name"], summary=spec["summary"],
                       code=spec["code"], input_schema=spec["input_schema"],
                       output_schema=spec["output_schema"], determinism=spec["determinism"],
                       source=("web_learning" if learning_note else "llm_generated"),
                       learning_note_id=(learning_note.id if learning_note else None),
                       status="draft", created_at=_now())

def _static_safety_scan(code: str) -> None:
    """静态扫描草稿代码，拦截明显危险调用（防意外破坏；非完整沙箱，靠子进程隔离兜底）。
    不变量: 拒绝 os.system/subprocess/eval/exec/open(写)/网络 等；提示人工复核。
    伪代码:
      banned = ["os.system", "subprocess", "eval(", "exec(", "__import__", "socket", "shutil.rmtree"]
      hits = [b for b in banned if b in code]
      if hits: raise AuthoringError(f"draft code contains banned calls: {hits}")
    """
```

`AUTHOR_SYS`：「你在为 MARVIS 写一个数据/特征/分析工具。只用 pandas/numpy/标准库做纯计算；不读写任意文件、不联网、不执行系统命令。必须声明 input_schema/output_schema/determinism。」

- **测试要点**：产合法 DraftTool（含 schema）；LLM 输出缺 schema→`AuthoringError`；危险调用被静态扫描拦截；学习笔记影响来源标记。

---

## Part E — 草稿沙箱运行（`drafts/sandbox.py`，子进程隔离）

```python
class DraftSandbox:
    def __init__(self, tool_runner: "ToolRunner", draft_registry: "DraftRegistry", repo):
        self._runner = tool_runner
        self._drafts = draft_registry
        self._repo = repo

    def run_draft(self, draft_id: str, inputs: dict, *, task_id: str) -> DraftRun:
        """在子进程隔离下临时运行草稿工具（仅当前任务，带审计标记）。不进正式 registry。
        入参: draft_id; inputs; task_id。
        出参: DraftRun（ok/output/error）。
        异常: 不抛（失败收进 DraftRun）。
        不变量: INV-6（子进程隔离同正式 tool）；INV-7（草稿运行不等于转正）；INV-8（审计标 draft）。
        伪代码:
          draft = self._drafts.get(draft_id)
          # 把草稿落成临时模块 + 临时单 tool manifest，交给 Phase 1 runner 的"draft 执行"路径
          module_path = self._materialize_draft_module(draft)   # 写临时 .py
          result = self._runner.invoke_adhoc(
              module=module_path, entrypoint=draft.name, inputs=inputs,
              input_schema=draft.input_schema, output_schema=draft.output_schema,
              timeout=DRAFT_TIMEOUT, task_id=task_id, mode="draft")
          run = DraftRun(id=_new_id(), draft_id=draft_id, task_id=task_id,
                         inputs_hash=_hash(inputs), ok=result.ok, output=result.output,
                         error=result.error, at=_now())
          self._repo.save_draft_run(run)
          self._repo.write_audit(kind="draft.run", target_ref=draft_id,
                                 detail={"task_id": task_id, "ok": result.ok})
          if result.ok: self._drafts.set_status(draft_id, "tested")
          return run
```

> Phase 1 的 `ToolRunner` 需补一个 `invoke_adhoc(module, entrypoint, inputs, input_schema, output_schema, timeout, task_id, mode)` 方法：与 `invoke` 同样的子进程 + schema 校验 + 审计，但工具来自显式传入的 module/schema 而非 registry 解析。`mode="draft"` 标记审计。这是 Phase 1 的小扩展，在本阶段补。

- **测试要点**：草稿经子进程运行往返；草稿失败收进 DraftRun 不抛；运行成功标 tested；审计带 draft 标记；草稿超时被杀（复用 Phase 1 隔离）；**草稿不出现在 `ToolRegistry.catalog_for_planner`（INV-7）**。

---

## Part F — 转正闸门（`drafts/promotion.py`）

```python
def validate_for_promotion(draft: DraftTool, *, sandbox: DraftSandbox,
                          test_cases: list[dict]) -> PromotionCheck:
    """转正前校验：schema 完整 + determinism 声明 + 至少一个测试用例通过。
    入参: draft; sandbox; test_cases（[{inputs, expect?}]，至少 1 个）。
    出参: PromotionCheck（passed + problems + 测试结果）。
    不变量: 风控合规——无 schema/无测试/测试不过 不得转正。
    伪代码:
      problems = []
      if not draft.input_schema or not draft.output_schema: problems.append("missing schema")
      if draft.determinism not in ("deterministic","stochastic"): problems.append("determinism not declared")
      if not test_cases: problems.append("at least one test case required")
      test_result = None
      if not problems:
          # 跑测试：每个用例经 sandbox 运行，output 必须过 output_schema；有 expect 则比对
          results = []
          for tc in test_cases:
              run = sandbox.run_draft(draft.id, tc["inputs"], task_id=draft.task_id)
              ok = run.ok and (("expect" not in tc) or _matches(run.output, tc["expect"]))
              results.append(ok)
          test_result = {"passed": all(results), "n": len(results)}
          if not all(results): problems.append("test cases failed")
      return PromotionCheck(passed=not problems, problems=tuple(problems), test_result=test_result)

def promote_draft(draft: DraftTool, *, registry: "PluginRegistry", drafts: "DraftRegistry",
                 plugins_dir: Path, check: PromotionCheck) -> "PluginManifest":
    """人工确认 + 校验通过后，把草稿转正为正式 Plugin/Tool（进 registry，Planner 可见）。
    入参: draft; 正式 registry; draft registry; plugins_dir; check（必须 passed）。
    出参: 注册后的 PluginManifest。
    异常: PromotionError（check 未通过 / 落盘失败）。
    不变量: INV-7——只有走过这个闸门的草稿才进正式 registry 被 Planner 选用；INV-8 审计。
    伪代码:
      if not check.passed: raise PromotionError(f"cannot promote: {check.problems}")
      # 用草稿构造一个单 tool 的 builtin-style plugin（或并入"promoted_drafts"插件）
      manifest = _build_manifest_from_draft(draft)     # name/version/module/tool spec
      dest = plugins_dir / manifest.name
      _write_plugin_files(dest, draft)                 # 落盘脚本 + manifest.json
      checksum = compute_checksum(dest)
      manifest = replace(manifest, checksum=checksum)
      registry.register(manifest, enabled=True)        # 进正式 registry → Planner 可见
      drafts.set_status(draft.id, "promoted")
      registry._repo.write_audit(kind="draft.promote", target_ref=draft.id,
                                 detail={"plugin": manifest.name, "tests": check.test_result})
      return manifest

def reject_draft(draft: DraftTool, *, drafts: "DraftRegistry", reason: str) -> None:
    """人工拒绝草稿（不转正）。
    伪代码: drafts.set_status(draft.id, "rejected"); 审计 draft.reject + reason。
    """
```

- **测试要点**：无 schema/无测试/测试不过→`passed=False`；通过后 promote 进正式 registry 且 Planner 可见；未通过 promote→`PromotionError`；拒绝标 rejected；转正全程审计；**转正前草稿对 Planner 不可见、转正后可见**（INV-7 闭环测试）。

---

## Part G — 草稿注册表（`drafts/registry.py`）

```python
class DraftRegistry:
    """草稿工具索引，与正式 PluginRegistry 隔离。关键：不暴露给 ToolRegistry.catalog_for_planner。"""
    def __init__(self, repo): self._repo = repo
    def add(self, draft: DraftTool) -> None: ...
    def get(self, draft_id) -> DraftTool: ...        # 异常 DraftNotFound
    def list_for_task(self, task_id, *, status=None) -> list[DraftTool]: ...
    def set_status(self, draft_id, status) -> None: ...
```

- **不变量**：`DraftRegistry` 与 `ToolRegistry` 无任何连接；Planner 通过 `ToolRegistry.catalog_for_planner` 取工具，草稿天然不可见（INV-7 的结构性保证）。
- **测试要点**：草稿 CRUD；按任务/状态列；与正式 registry 隔离。

---

## Part H — tools / 端点（`drafts/tools.py` + `routers/drafts.py`）

部分能力作为 **tool**（Agent 在编排中可调，但产物进草稿区）：

```python
def tool_web_search(inputs, ctx) -> dict:
    """inputs:{query, max_results?}。output:{results:[{title,url,snippet}]} 或 {offline:true,guidance}。
    不变量: 离线时返回 offline 标记 + 引导，不报致命错（让编排优雅处理）。"""

def tool_draft_script(inputs, ctx) -> dict:
    """inputs:{goal, learning_note_id?}。output:{draft_id, name, has_schema:true}。
    不变量: INV-7——产物进草稿区，不可被 Planner 直接当工具调。"""

def tool_run_draft(inputs, ctx) -> dict:
    """inputs:{draft_id, inputs}。output:{ok, output, error}。子进程隔离。"""
```

HTTP 端点（人工治理，**不**走 Agent 自动）：

```python
@router.get("/api/drafts")              # 列草稿（按任务/状态）
@router.get("/api/drafts/{id}")         # 草稿详情（源码、schema、运行历史、来源学习笔记）
@router.post("/api/drafts/{id}/run")    # 手动试运行
@router.post("/api/drafts/{id}/promote")# 转正（body: test_cases）→ 校验闸门 → 注册
@router.post("/api/drafts/{id}/reject") # 拒绝（body: reason）
```

- **不变量**：`promote` 是**人工动作**（HTTP 端点，非 Agent 自动调用）；转正闸门在此强制（INV-7 + 合规）。
- **测试要点**：tool 经 runner 往返；离线 web_search 返回 offline 标记；promote 端点跑闸门；reject 标状态；**Agent 无法绕过人工 promote 端点把草稿变正式工具**。

---

## Part I — 持久层

```sql
CREATE TABLE IF NOT EXISTS draft_tools (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, name TEXT NOT NULL, summary TEXT,
  code TEXT NOT NULL, input_schema_json TEXT NOT NULL, output_schema_json TEXT NOT NULL,
  determinism TEXT NOT NULL, source TEXT NOT NULL, learning_note_id TEXT,
  status TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS draft_runs (
  id TEXT PRIMARY KEY, draft_id TEXT NOT NULL, task_id TEXT NOT NULL,
  inputs_hash TEXT, ok INTEGER NOT NULL, output_json TEXT, error TEXT, at TEXT NOT NULL,
  FOREIGN KEY (draft_id) REFERENCES draft_tools(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS learning_notes (
  id TEXT PRIMARY KEY, query TEXT, sources_json TEXT, distilled TEXT, created_at TEXT NOT NULL
);
```

`DraftRepository`：`save_draft`、`get_draft`、`list_drafts`、`set_status`、`save_draft_run`、`list_runs`、`save_learning_note`、`get_learning_note`。

- **测试要点**：草稿/运行/笔记往返；FK CASCADE；状态流转。

---

## Part J — 前端（草稿区视图，`static/js/views/drafts.js`）

- 设置/工具管理区加"草稿工具"面板（非任务常驻）：
  - 列草稿（来源 web_learning/llm_generated/hand_written、状态 draft/tested/promoted/rejected）。
  - 详情：源码（只读高亮）、input/output schema、运行历史、来源学习笔记 + URL。
  - 操作：试运行（填 inputs）、**转正**（填测试用例 → 触发闸门）、拒绝。
- 转正按钮明确是人工动作，二次确认（风控合规感）。
- **测试要点**（`test_frontend_static_v2.py` 风格）：草稿列表/详情渲染；转正需填测试用例；离线时 web_search 入口提示"无网，请外部产出后上传"。

---

## Part K — 测试计划汇总

| 文件 | 覆盖 |
|------|------|
| `tests/test_drafts_contracts.py` | dataclass 往返、状态机 |
| `tests/test_drafts_web_search.py` | **离线降级、有网搜索、httpx 缺失** |
| `tests/test_drafts_learning.py` | 蒸馏 bounded、记来源 |
| `tests/test_drafts_authoring.py` | 产合法草稿、缺 schema 报错、**危险调用静态拦截** |
| `tests/test_drafts_sandbox.py` | 子进程运行、失败收集、超时、**不进 planner catalog** |
| `tests/test_drafts_promotion.py` | **闸门：无 schema/无测试/测试不过拒绝；通过后进 registry 可见** |
| `tests/test_drafts_registry.py` | 与正式 registry 隔离 |
| `tests/test_drafts_api.py` | 端点；**Agent 无法绕过人工 promote** |
| `tests/test_drafts_db.py` | 往返、CASCADE |
| `tests/test_drafts_governance.py` | **INV-7 闭环：草稿全程对 Planner 不可见，仅人工转正后可见** |

`test_drafts_governance.py` 是治理护栏：构造草稿→确认它不在 `catalog_for_planner`→人工转正→确认它现在在 catalog 里。证明 INV-7 结构性成立。

---

## Part L — 任务执行顺序

```text
1. A 契约
2. I DB + DraftRepository
3. G DraftRegistry（与正式 registry 隔离）
4. B web_search（离线降级）
5. C learning
6. D authoring（含静态安全扫描）
7. （Phase 1 扩展）ToolRunner.invoke_adhoc
8. E sandbox（依赖 7）
9. F promotion（转正闸门）
10. H tools + HTTP 端点
11. J 前端草稿区
12. K 测试 + 治理护栏
```

每项 atomic commit。Phase 8 完成标志：有网时能搜索→蒸馏→LLM 写草稿→子进程隔离试运行→人工填测试用例→闸门校验→转正进正式 registry 被 Planner 选用；离线时 web_search 优雅降级引导外部上传；**治理护栏证明草稿转正前对 Planner 完全不可见（INV-7）**；危险调用被静态扫描 + 子进程隔离双重拦截。

---

*Phase 8 给 Agent "学新东西"的能力，但用一道人工闸门守住合规底线：联网学的、LLM 写的代码可以临时跑、可以审，但要变成正式工具被自动调用，必须有 schema、有测试、人点头。这正是风控场景需要的"可成长但可控"。*
