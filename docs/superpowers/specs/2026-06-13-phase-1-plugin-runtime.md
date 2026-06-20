# Phase 1 — Plugin/Tool Runtime（函数级 spec，含内部伪代码）

## 文档状态

- 状态：已实现并验证
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 4.2、6 节）
- 前置依赖：Phase 0 完成（地基修复 + db.py `connect()` 封装可靠）
- 目标：交付 V2 的能力底座——Plugin/Tool 的**声明（manifest）、注册（registry）、子进程执行（runner）、事件钩子（hooks）、上传安装（loader）、HTTP 管理（router）**。本阶段不含任何业务能力包（那是 Phase 3+），但要交付一个 trivial 内置 echo 包证明 runtime 全链路打通。

## 捍卫的不变量

- **INV-6**：Tool 执行子进程隔离，超时/崩溃/OOM 不拖垮主服务。
- **INV-1/INV-2**：Tool output 强制过 output_schema 校验，保证结构化、可追溯。
- **INV-8**：每次 Tool 执行、Plugin 安装/启停留审计。
- **INV-9**：子进程用 spawn + 显式 encoding，`setrlimit` 仅 POSIX、Windows 优雅降级。

## 模块布局

```text
marvis/plugins/
  __init__.py
  errors.py              异常层级
  manifest.py            PluginManifest / ToolSpec / HookSpec + 解析校验
  schema_validation.py   JSON Schema 校验封装
  registry.py            PluginRegistry / ToolRegistry
  loader.py              内置包发现、上传安装、checksum、entrypoint 解析
  subprocess_worker.py   子进程内执行入口（被 runner 起）
  runner.py              ToolRunner / ToolResult / ToolContext
  hooks.py               HookDispatcher
marvis/packs/
  __init__.py
  _sample/               trivial 内置包（echo），证明链路
    __init__.py
    manifest.json
    tools.py
marvis/routers/
  __init__.py
  plugins.py             HTTP 管理端点
marvis/db.py   新增 plugins/tools/audit 表 + PluginRepository
```

新增依赖（`pyproject.toml`）：`jsonschema>=4.0`。

---

## Part A — 异常层级（`plugins/errors.py`）

```python
class PluginError(Exception):
    """所有 plugin runtime 异常基类。"""

class ManifestError(PluginError):
    """manifest 缺字段/类型错/版本号非法。"""

class SchemaValidationError(PluginError):
    """Tool inputs 或 output 不符合声明的 JSON Schema。"""
    def __init__(self, label: str, detail: str):
        # label: "inputs" | "output:<tool>"，detail: 具体哪个字段错
        super().__init__(f"{label} schema validation failed: {detail}")
        self.label = label
        self.detail = detail

class PluginNotFoundError(PluginError): ...
class ToolNotFoundError(PluginError): ...
class DuplicatePluginError(PluginError): ...

class ToolExecutionError(PluginError):
    """Tool 函数在子进程内主动抛错（业务异常）。"""
    def __init__(self, message: str, traceback_text: str):
        super().__init__(message)
        self.traceback_text = traceback_text  # 子进程回传的 traceback，审计用

class ToolTimeoutError(PluginError):
    """Tool 执行超过 timeout_seconds，子进程被杀。"""

class ToolResourceError(PluginError):
    """子进程 OOM 或触发 setrlimit。"""

class WorkerProtocolError(PluginError):
    """子进程返回的不是合法 result JSON（worker 崩溃在协议层）。"""
```

- **测试要点**：`SchemaValidationError` 携带 `label`/`detail`；`ToolExecutionError` 携带 `traceback_text`。
- **不变量**：所有对外抛出的异常都是 `PluginError` 子类，调用方（runner/router）可统一 `except PluginError` 兜底，绝不裸 `except Exception: pass`。

---

## Part B — 数据契约与解析（`plugins/manifest.py`）

### B-1 dataclasses

> **ToolRef 的规范定义在此（Phase 1）**——runner/registry 都依赖它。Phase 2 `orchestrator/contracts.py` 复用本定义（`from marvis.plugins.manifest import ToolRef`），不重复定义。

```python
@dataclass(frozen=True)
class ToolRef:
    plugin: str
    tool: str
    version: str = ""          # 空=不锁版本，用当前注册版本
    def label(self) -> str: return f"{self.plugin}.{self.tool}"

@dataclass(frozen=True)
class ToolSpec:
    name: str                  # 工具名，plugin 内唯一
    summary: str               # 一句话，给 Planner 选工具用
    input_schema: dict         # JSON Schema
    output_schema: dict        # JSON Schema（确定性结构化）
    determinism: str           # "deterministic" | "stochastic"
    timeout_seconds: int       # 执行超时
    failure_policy: str        # "fail" | "retry" | "skip"
    side_effects: tuple[str, ...]  # 读/写的资源标识，审计用
    entrypoint: str            # 该 tool 的可调用名（tools.py 内函数名）
    memory_limit_mb: int = 2048  # 子进程内存上限（POSIX 强制，Windows best-effort）

@dataclass(frozen=True)
class HookSpec:
    event: str                 # 平台事件名（蓝图 5.5）
    tool: str                  # 触发的本插件 tool name

@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str               # semver
    display_name: str
    description: str
    builtin: bool
    module: str                # 包根模块路径，如 "marvis.packs.feature"
    tools: tuple[ToolSpec, ...]
    hooks: tuple[HookSpec, ...]
    permissions: tuple[str, ...]   # 仅审计展示
    python_requires: str | None
    checksum: str | None       # 安装时由平台计算；manifest 输入里不要求上传方提供，内置包可为 None
```

### B-2 `parse_manifest`

```python
def parse_manifest(data: dict, *, builtin: bool = False) -> PluginManifest:
    """把 manifest.json 解析的 dict 转成 PluginManifest 并做结构校验。

    入参:
      data: 从 manifest.json json.load 出来的 dict。
      builtin: 是否内置包（内置包 checksum 可缺省；上传包 checksum 由 install_plugin 计算）。
    出参:
      PluginManifest（已通过结构校验）。
    异常:
      ManifestError: 缺必填字段 / 类型错 / 版本号非法 / tool 名重复。
    不变量: INV-1（output_schema 必填，保证结构化）。
    """
    # 伪代码:
    require_keys = {"name", "version", "display_name", "description", "module", "tools"}
    missing = require_keys - data.keys()
    if missing:
        raise ManifestError(f"manifest missing keys: {sorted(missing)}")

    if not _is_semver(data["version"]):
        raise ManifestError(f"invalid version: {data['version']!r}")

    tools = []
    seen_tool_names = set()
    for raw in data["tools"]:
        for k in ("name", "summary", "input_schema", "output_schema",
                  "determinism", "timeout_seconds", "failure_policy", "entrypoint"):
            if k not in raw:
                raise ManifestError(f"tool missing key {k!r}: {raw.get('name')}")
        if raw["name"] in seen_tool_names:
            raise ManifestError(f"duplicate tool name: {raw['name']}")
        seen_tool_names.add(raw["name"])
        if raw["determinism"] not in ("deterministic", "stochastic"):
            raise ManifestError(f"bad determinism: {raw['determinism']}")
        if raw["failure_policy"] not in ("fail", "retry", "skip"):
            raise ManifestError(f"bad failure_policy: {raw['failure_policy']}")
        if not isinstance(raw["timeout_seconds"], int) or raw["timeout_seconds"] <= 0:
            raise ManifestError(f"bad timeout_seconds for {raw['name']}")
        # output_schema 必须是非空 dict（INV-1：结构化产出）
        if not isinstance(raw["output_schema"], dict) or not raw["output_schema"]:
            raise ManifestError(f"tool {raw['name']} must declare non-empty output_schema")
        tools.append(ToolSpec(
            name=raw["name"], summary=raw["summary"],
            input_schema=raw["input_schema"], output_schema=raw["output_schema"],
            determinism=raw["determinism"], timeout_seconds=raw["timeout_seconds"],
            failure_policy=raw["failure_policy"],
            side_effects=tuple(raw.get("side_effects", [])),
            entrypoint=raw["entrypoint"],
            memory_limit_mb=int(raw.get("memory_limit_mb", 2048)),
        ))

    hooks = tuple(
        HookSpec(event=h["event"], tool=h["tool"])
        for h in data.get("hooks", [])
    )
    # 校验 hook.tool 指向存在的 tool
    for h in hooks:
        if h.tool not in seen_tool_names:
            raise ManifestError(f"hook references unknown tool: {h.tool}")
    # 校验 hook.event 在允许集合内
    for h in hooks:
        if h.event not in PLATFORM_EVENTS:   # 见蓝图 5.5 常量
            raise ManifestError(f"unknown hook event: {h.event}")

    return PluginManifest(
        name=data["name"], version=data["version"],
        display_name=data["display_name"], description=data["description"],
        builtin=builtin, module=data["module"],
        tools=tuple(tools), hooks=hooks,
        permissions=tuple(data.get("permissions", [])),
        python_requires=data.get("python_requires"),
        # 上传方 manifest 里的 checksum 只作为非信任输入保留；安装时必须被平台计算值覆盖。
        checksum=data.get("checksum"),
    )
```

辅助：

```python
def _is_semver(v: str) -> bool:
    """semver 校验: 返回是否匹配 ^\d+\.\d+\.\d+([-+].*)?$"""
    return bool(re.match(r"^\d+\.\d+\.\d+([-+].*)?$", v))

PLATFORM_EVENTS = frozenset({
    "task.created", "task.scanned", "dataset.registered", "join.confirmed",
    "notebook.completed", "validation.completed", "feature.computed",
    "plan.confirmed", "step.completed", "report.before_generate",
    "report.after_generate", "memory.before_save", "memory.after_save",
    "workflow.completed",
})
```

### B-3 `manifest_to_dict`

```python
def manifest_to_dict(manifest: PluginManifest) -> dict:
    """序列化回 dict（持久化到 plugins.manifest_json 用）。
    入参: manifest。 出参: 可 json.dumps 的 dict。 异常: 无。
    伪代码: dataclasses.asdict + tuple→list 归一。
    """
    d = dataclasses.asdict(manifest)
    # asdict 已递归；确保 tuple 变 list（json 友好）
    return json.loads(json.dumps(d, default=list))
```

- **测试要点**：缺 key/坏 version/坏 determinism/重复 tool 名/空 output_schema/未知 hook event 各抛 `ManifestError`；`parse → to_dict → parse` 往返一致；内置包免 checksum；上传包不要求自带 checksum，安装后持久化平台计算 checksum。

---

## Part C — JSON Schema 校验（`plugins/schema_validation.py`）

```python
def validate_against_schema(value: object, schema: dict, *, label: str) -> None:
    """用 jsonschema 校验 value 是否符合 schema，不符合抛 SchemaValidationError。

    入参:
      value: 待校验对象（tool inputs 或 output，必须是 JSON 可序列化结构）。
      schema: JSON Schema dict。
      label: 错误标签，如 "inputs" 或 "output:compute_iv"。
    出参: None（通过即返回）。
    异常: SchemaValidationError（携带 label 和首个错误的 path+message）。
    不变量: INV-1（保证 tool 产出结构化）。
    """
    # 伪代码:
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        path = "/".join(str(p) for p in first.path) or "<root>"
        raise SchemaValidationError(label, f"at {path}: {first.message}")
```

- **测试要点**：合法对象通过；缺必填字段/类型错抛 `SchemaValidationError` 且 `label` 正确、`detail` 指出 path；schema 本身非法时 jsonschema 抛 `SchemaError`（让它冒泡，安装时已校验过 schema 合法性，见 loader）。

---

## Part D — 注册表（`plugins/registry.py`）

### D-1 `PluginRegistry`

内存索引 + DB 持久化的双写视图。进程启动时从 DB 重建。

```python
class PluginRegistry:
    def __init__(self, repo: "PluginRepository"):
        """入参: repo（DB 持久层）。内存缓存 name -> (manifest, enabled)。"""
        self._repo = repo
        self._plugins: dict[str, tuple[PluginManifest, bool]] = {}

    def load_from_db(self) -> None:
        """启动时从 DB 重建内存索引。
        出参: None。 异常: 无（DB 空则索引空）。
        伪代码:
          for row in self._repo.list_plugins(include_disabled=True):
              manifest = parse_manifest(json.loads(row['manifest_json']),
                                        builtin=bool(row['builtin']))
              self._plugins[manifest.name] = (manifest, bool(row['enabled']))
        """

    def register(self, manifest: PluginManifest, *, enabled: bool = True) -> None:
        """注册（或覆盖升级）一个 plugin，双写 DB + 内存。
        入参: manifest；enabled 初始启用态。
        异常: DuplicatePluginError（同名同版本已存在且非升级场景）。
        不变量: INV-8（写审计）。
        伪代码:
          existing = self._plugins.get(manifest.name)
          if existing and existing[0].version == manifest.version and existing[0].builtin == manifest.builtin:
              raise DuplicatePluginError(f"{manifest.name}@{manifest.version} already registered")
          self._repo.upsert_plugin(manifest, enabled=enabled)
          self._plugins[manifest.name] = (manifest, enabled)
          self._repo.write_audit(kind="plugin.register", target=manifest.name,
                                 detail={"version": manifest.version})
        """

    def unregister(self, name: str) -> None:
        """删除 plugin（内置包禁止删，抛 PluginError）。
        异常: PluginNotFoundError；删内置包抛 PluginError。
        伪代码:
          entry = self._require(name)
          if entry[0].builtin: raise PluginError("cannot remove builtin plugin")
          self._repo.delete_plugin(name); del self._plugins[name]
          self._repo.write_audit(kind="plugin.unregister", target=name)
        """

    def set_enabled(self, name: str, enabled: bool) -> None:
        """启用/禁用。 异常: PluginNotFoundError。
        伪代码:
          manifest, _ = self._require(name)
          self._repo.set_enabled(name, enabled)
          self._plugins[name] = (manifest, enabled)
          self._repo.write_audit(kind="plugin.enable" if enabled else "plugin.disable", target=name)
        """

    def get(self, name: str) -> PluginManifest:
        """取 manifest。 异常: PluginNotFoundError。"""
        return self._require(name)[0]

    def is_enabled(self, name: str) -> bool:
        """是否启用；不存在返回 False。"""
        entry = self._plugins.get(name)
        return bool(entry and entry[1])

    def list(self, *, include_disabled: bool = False) -> list[PluginManifest]:
        """列出 manifest。 include_disabled=False 时只列启用的。"""
        return [m for m, en in self._plugins.values() if include_disabled or en]

    def _require(self, name: str) -> tuple[PluginManifest, bool]:
        entry = self._plugins.get(name)
        if entry is None:
            raise PluginNotFoundError(name)
        return entry
```

### D-2 `ToolRegistry`

```python
class ToolRegistry:
    def __init__(self, plugin_registry: PluginRegistry):
        self._plugins = plugin_registry

    def resolve(self, tool_ref: ToolRef) -> ToolSpec:
        """把 ToolRef(plugin,tool,version) 解析成 ToolSpec。
        入参: tool_ref。
        出参: ToolSpec。
        异常:
          PluginNotFoundError: plugin 不存在；
          ToolNotFoundError: plugin 未启用 / tool 不存在 / 版本不匹配。
        不变量: 只解析 enabled plugin 的 tool（禁用即不可用）。
        伪代码:
          if not self._plugins.is_enabled(tool_ref.plugin):
              raise ToolNotFoundError(f"plugin disabled or missing: {tool_ref.plugin}")
          manifest = self._plugins.get(tool_ref.plugin)
          if tool_ref.version and tool_ref.version != manifest.version:
              raise ToolNotFoundError(f"version mismatch: want {tool_ref.version}, have {manifest.version}")
          for tool in manifest.tools:
              if tool.name == tool_ref.tool: return tool
          raise ToolNotFoundError(f"{tool_ref.plugin}.{tool_ref.tool}")
        """

    def list_tools(self, *, enabled_only: bool = True) -> list[tuple[str, ToolSpec]]:
        """列出 (plugin_name, ToolSpec)。"""
        out = []
        for manifest in self._plugins.list(include_disabled=not enabled_only):
            for tool in manifest.tools:
                out.append((manifest.name, tool))
        return out

    def catalog_for_planner(self) -> list[dict]:
        """给 Planner（Phase 2）的紧凑工具目录：只含选工具所需信息。
        出参: list of {plugin, tool, summary, input_schema, output_schema, determinism}。
        不变量: 不泄露 entrypoint/side_effects 等内部细节给 LLM。
        伪代码:
          return [
            {"plugin": p, "tool": t.name, "summary": t.summary,
             "input_schema": t.input_schema, "output_schema": t.output_schema,
             "determinism": t.determinism}
            for p, t in self.list_tools(enabled_only=True)
          ]
        """
```

- **测试要点**：注册/启停/删除往返；删内置包被拒；`resolve` 对禁用 plugin / 版本不匹配 / 缺 tool 各抛对应异常；`catalog_for_planner` 不含 entrypoint。

---

## Part E — 子进程执行（核心，INV-6）

### E-1 `runner.py` 的数据契约

```python
@dataclass
class ToolContext:
    """传给 tool 函数的运行上下文（在子进程内构造）。"""
    task_id: str
    seed: int | None              # stochastic tool 用
    datasets_root: Path           # 数据集根目录
    workspace: Path               # 任务 workspace
    def load_dataset_path(self, dataset_id: str) -> Path:
        """返回某 dataset 的物理文件路径（tool 自己用 pandas/duckdb 读，避免跨进程传大 DataFrame）。"""

@dataclass
class ToolResult:
    ok: bool
    output: dict | None           # 成功时的结构化 output（已过 output_schema）
    error: str | None             # 失败时的人类可读错误
    error_kind: str | None        # "schema"|"execution"|"timeout"|"resource"|"protocol"
    duration_ms: int
    stdout_tail: str              # 子进程 stdout 末尾（调试）
    stderr_tail: str              # 子进程 stderr 末尾（调试）
```

### E-2 Job/Result 协议（runner ↔ worker 的 JSON）

```text
Job (runner → worker, stdin):
  {
    "module": "marvis.packs.feature",   # plugin.module
    "entrypoint": "tool_compute_iv",               # ToolSpec.entrypoint
    "inputs": {...},                               # 已过 input_schema
    "task_id": "...", "seed": 42 | null,
    "datasets_root": "/abs/path", "workspace": "/abs/path",
    "memory_limit_mb": 2048
  }

Result (worker → runner, stdout, 单行 JSON):
  成功: {"ok": true, "output": {...}}
  失败: {"ok": false, "error_kind": "execution", "error": "...", "traceback": "..."}
```

### E-3 `subprocess_worker.py`

```python
def worker_main() -> None:
    """子进程入口：读 stdin 的 Job JSON，执行 tool，写 stdout 的 Result JSON。
    被 `python -m marvis.plugins.subprocess_worker` 调起。
    出参: None（结果走 stdout）。 进程退出码: 成功 0，协议级失败 1。
    不变量: INV-6（所有异常都转成 Result JSON，不让子进程裸崩）；INV-9（显式 utf-8）。
    """
    # 伪代码:
    raw = sys.stdin.buffer.read().decode("utf-8")
    try:
        job = json.loads(raw)
    except Exception as exc:
        _emit({"ok": False, "error_kind": "protocol", "error": f"bad job json: {exc}"})
        sys.exit(1)

    _apply_resource_limits(job.get("memory_limit_mb"))   # POSIX setrlimit

    try:
        output = _run_tool(job)
        _emit({"ok": True, "output": output})
    except MemoryError as exc:
        _emit({"ok": False, "error_kind": "resource", "error": str(exc)})
    except Exception as exc:
        _emit({"ok": False, "error_kind": "execution",
               "error": str(exc), "traceback": traceback.format_exc()})


def _run_tool(job: dict) -> dict:
    """import module、取 entrypoint callable、构造 ctx、调用。
    出参: tool 返回的 dict（未校验，校验在 runner 侧做）。
    伪代码:
      module = importlib.import_module(job["module"])
      func = getattr(module, job["entrypoint"])
      ctx = ToolContext(task_id=job["task_id"], seed=job.get("seed"),
                        datasets_root=Path(job["datasets_root"]),
                        workspace=Path(job["workspace"]))
      if job.get("seed") is not None:
          random.seed(job["seed"]); np.random.seed(job["seed"])   # stochastic 可复现
      result = func(job["inputs"], ctx)
      if not isinstance(result, dict):
          raise TypeError(f"tool must return dict, got {type(result)}")
      return result
    """


def _apply_resource_limits(memory_mb: int | None) -> None:
    """POSIX 下用 setrlimit 限制地址空间；Windows 优雅降级（no-op + 注释）。
    不变量: INV-9（跨平台）。
    伪代码:
      if memory_mb is None: return
      try:
          import resource   # 仅 POSIX
          soft = memory_mb * 1024 * 1024
          resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
      except (ImportError, ValueError, OSError):
          pass   # Windows 或受限环境：退化为仅靠 runner 侧 timeout 守护
    """


def _emit(obj: dict) -> None:
    """单行 JSON 写 stdout 并 flush。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    worker_main()
```

### E-4 `ToolRunner`

```python
class ToolRunner:
    def __init__(self, tool_registry: ToolRegistry, repo: "PluginRepository",
                 *, python_executable: str, datasets_root: Path, workspace: Path):
        """入参:
             tool_registry: 解析 ToolRef；
             repo: 写审计；
             python_executable: 跑 worker 的 python（来自 execution_environment）;
             datasets_root/workspace: 传给子进程。
        """
        self._tools = tool_registry
        self._repo = repo
        self._python = python_executable
        self._datasets_root = datasets_root
        self._workspace = workspace

    def invoke(self, tool_ref: ToolRef, inputs: dict, *,
               task_id: str, seed: int | None = None) -> ToolResult:
        """执行一个 Tool（子进程隔离），返回结构化 ToolResult。
        入参:
          tool_ref: 要执行的工具；
          inputs: 入参 dict（执行前会过 input_schema）；
          task_id: 归属任务；
          seed: stochastic 工具的随机种子（deterministic 工具忽略）。
        出参: ToolResult（ok/output/error/error_kind/duration/stdout_tail/stderr_tail）。
        异常: 不抛——所有失败都收敛进 ToolResult.ok=False（INV-6，便于 executor 按 failure_policy 处理）。
              仅 ToolNotFoundError/SchemaValidationError（inputs 侧）作为编程错误可冒泡。
        不变量: INV-1（output 过 output_schema）、INV-6（子进程+超时）、INV-8（审计）。
        """
        # 伪代码:
        started = _now_ms()
        tool = self._tools.resolve(tool_ref)              # 可能抛 ToolNotFoundError
        # stochastic 工具必须给 seed（可复现）
        if tool.determinism == "stochastic" and seed is None:
            seed = _derive_seed(task_id, tool_ref)        # 任务+工具派生稳定 seed
        try:
            self._validate_inputs(tool, inputs)            # 抛 SchemaValidationError
        except SchemaValidationError as exc:
            return self._fail(tool_ref, "schema", str(exc), started, "", "", task_id, inputs)

        manifest = self._tools._plugins.get(tool_ref.plugin)
        job = self._build_job(manifest, tool, inputs, task_id, seed)
        try:
            raw_result, out_tail, err_tail = self._run_subprocess(job, tool.timeout_seconds)
        except ToolTimeoutError as exc:
            return self._fail(tool_ref, "timeout", str(exc), started, "", "", task_id, inputs)
        except WorkerProtocolError as exc:
            return self._fail(tool_ref, "protocol", str(exc), started, "", str(exc), task_id, inputs)

        if not raw_result.get("ok"):
            kind = raw_result.get("error_kind", "execution")
            return self._fail(tool_ref, kind, raw_result.get("error", "tool failed"),
                              started, out_tail, err_tail, task_id, inputs,
                              traceback_text=raw_result.get("traceback"))

        output = raw_result["output"]
        try:
            validate_against_schema(output, tool.output_schema, label=f"output:{tool.name}")
        except SchemaValidationError as exc:
            # INV-1: 工具号称成功但产出不符合 schema，按失败处理
            return self._fail(tool_ref, "schema", str(exc), started, out_tail, err_tail, task_id, inputs)

        duration = _now_ms() - started
        self._write_audit(tool_ref, task_id, inputs, ok=True, duration=duration,
                          error=None, side_effects=tool.side_effects)
        return ToolResult(ok=True, output=output, error=None, error_kind=None,
                          duration_ms=duration, stdout_tail=out_tail, stderr_tail=err_tail)

    def _validate_inputs(self, tool: ToolSpec, inputs: dict) -> None:
        validate_against_schema(inputs, tool.input_schema, label="inputs")

    def _build_job(self, manifest, tool, inputs, task_id, seed) -> dict:
        return {
            "module": manifest.module, "entrypoint": tool.entrypoint,
            "inputs": inputs, "task_id": task_id, "seed": seed,
            "datasets_root": str(self._datasets_root), "workspace": str(self._workspace),
            "memory_limit_mb": tool.memory_limit_mb,
        }

    def _run_subprocess(self, job: dict, timeout: int) -> tuple[dict, str, str]:
        """起子进程跑 worker，喂 job、收 result，超时杀进程。
        出参: (result_dict, stdout_tail, stderr_tail)。
        异常: ToolTimeoutError（超时）；WorkerProtocolError（stdout 非合法 result JSON）。
        不变量: INV-6（超时强杀）、INV-9（spawn + utf-8）。
        伪代码:
          proc = subprocess.Popen(
              [self._python, "-m", "marvis.plugins.subprocess_worker"],
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
          try:
              out, err = proc.communicate(
                  input=json.dumps(job).encode("utf-8"), timeout=timeout)
          except subprocess.TimeoutExpired:
              proc.kill(); proc.communicate()        # 回收僵尸
              raise ToolTimeoutError(f"tool exceeded {timeout}s")
          out_text = out.decode("utf-8", errors="replace")
          err_text = err.decode("utf-8", errors="replace")
          # worker 的 result 是 stdout 最后一行 JSON
          last_line = out_text.strip().splitlines()[-1] if out_text.strip() else ""
          try:
              result = json.loads(last_line)
          except Exception:
              raise WorkerProtocolError(f"worker stdout not JSON; stderr tail: {err_text[-500:]}")
          return result, out_text[-2000:], err_text[-2000:]
        """

    def _fail(self, tool_ref, kind, error, started, out_tail, err_tail,
              task_id, inputs, *, traceback_text=None) -> ToolResult:
        duration = _now_ms() - started
        self._write_audit(tool_ref, task_id, inputs, ok=False, duration=duration,
                          error=error, side_effects=())
        return ToolResult(ok=False, output=None, error=error, error_kind=kind,
                          duration_ms=duration, stdout_tail=out_tail, stderr_tail=err_tail)

    def _write_audit(self, tool_ref, task_id, inputs, *, ok, duration, error, side_effects):
        """INV-8: 写 tool 执行审计。inputs 只存 hash（可能含敏感参数，INV-5）。"""
        self._repo.write_audit(
            kind="tool.invoke", actor="agent", target_ref=f"{tool_ref.plugin}.{tool_ref.tool}",
            inputs_hash=_hash_json(inputs), outcome="ok" if ok else "fail",
            detail={"task_id": task_id, "duration_ms": duration,
                    "error": error, "side_effects": list(side_effects)})

    def invoke_adhoc(self, *, module: str, entrypoint: str, inputs: dict,
                     input_schema: dict, output_schema: dict, timeout: int,
                     task_id: str, mode: str = "draft", seed: int | None = None) -> ToolResult:
        """执行一个【未注册】的临时工具（Phase 8 草稿沙箱用）。与 invoke 同样的子进程+schema+审计，
           但 tool 来自显式传入的 module/schema，而非 registry 解析。
        入参: module 临时模块路径; entrypoint 函数名; inputs; input/output_schema; timeout; task_id;
              mode 审计标记（draft）; seed。
        出参: ToolResult。
        异常: 不抛（失败收进 ToolResult）。
        不变量: INV-6（子进程隔离同 invoke）；INV-7（adhoc 不经 registry，草稿不因此进正式工具库）；
                INV-8（审计 kind="tool.invoke_adhoc"，detail 标 mode）。
        伪代码:
          start=_now_ms()
          try: validate_against_schema(inputs, input_schema, label="inputs")
          except SchemaValidationError as e: return _fail_adhoc("schema", str(e), ...)
          job = {"module": module, "entrypoint": entrypoint, "inputs": inputs, "task_id": task_id,
                 "seed": seed, "datasets_root": str(self._datasets_root),
                 "workspace": str(self._workspace), "memory_limit_mb": DRAFT_MEMORY_LIMIT_MB}
          try: raw, out_tail, err_tail = self._run_subprocess(job, timeout)
          except (ToolTimeoutError, WorkerProtocolError) as e: return _fail_adhoc(...)
          if not raw.get("ok"): return _fail_adhoc(raw.get("error_kind","execution"), raw.get("error"), ...)
          try: validate_against_schema(raw["output"], output_schema, label="output:adhoc")
          except SchemaValidationError as e: return _fail_adhoc("schema", str(e), ...)
          self._repo.write_audit(kind="tool.invoke_adhoc", actor="agent",
              target_ref=f"{mode}:{entrypoint}", inputs_hash=_hash_json(inputs),
              outcome="ok", detail={"task_id": task_id, "mode": mode})
          return ToolResult(ok=True, output=raw["output"], error=None, error_kind=None,
                            duration_ms=_now_ms()-start, stdout_tail=out_tail, stderr_tail=err_tail)
        """
```

> `invoke_adhoc` 是为 Phase 8 草稿沙箱预留的扩展。Phase 1 实施时一并交付（与 `invoke` 共用 `_run_subprocess`/schema 校验/审计）。它**不**做 registry 解析——这是 INV-7 的关键：草稿能跑但不进正式工具目录。

- **测试要点**（重点，CODE_REVIEW 教训：边界 + 跨平台）：
  - echo 工具正常往返（`_sample` 包）。
  - inputs 不符合 input_schema → `ok=False, error_kind="schema"`，不起子进程。
  - tool 内 `raise ValueError` → `ok=False, error_kind="execution"`，`traceback` 进审计。
  - tool 死循环 → `timeout` 内被杀，`error_kind="timeout"`，无僵尸进程残留。
  - tool 返回不符合 output_schema 的 dict → `ok=False, error_kind="schema"`（INV-1）。
  - stochastic 工具同 seed 两次 output 一致。
  - worker stdout 非 JSON（模拟崩溃）→ `error_kind="protocol"`。
  - 跨平台：`_apply_resource_limits` 在无 `resource` 模块时不抛（Windows 路径）。
  - 审计：每次 invoke 都写一条，inputs 只存 hash。

---

## Part F — 事件钩子（`plugins/hooks.py`）

```python
class HookDispatcher:
    def __init__(self, plugin_registry: PluginRegistry, tool_runner: ToolRunner):
        self._plugins = plugin_registry
        self._runner = tool_runner
        self._index: dict[str, list[ToolRef]] = {}   # event -> [ToolRef]

    def rebuild_index(self) -> None:
        """从所有启用 plugin 的 hooks 重建 event→ToolRef 索引。
        伪代码:
          self._index.clear()
          for manifest in self._plugins.list(include_disabled=False):
              for hook in manifest.hooks:
                  self._index.setdefault(hook.event, []).append(
                      ToolRef(manifest.name, hook.tool, manifest.version))
        """

    def dispatch(self, event: str, payload: dict, *, task_id: str) -> list[ToolResult]:
        """触发某事件的所有 hook tool，失败隔离（一个 hook 失败不影响其它）。
        入参: event 平台事件名；payload 传给 hook 的 inputs；task_id。
        出参: list[ToolResult]（每个 hook 一个，含失败的）。
        异常: 不抛（单个 hook 失败收进 ToolResult.ok=False）。
        不变量: INV-6（失败隔离）、INV-8（dispatch 审计）。
        伪代码:
          results = []
          for tool_ref in self._index.get(event, []):
              try:
                  results.append(self._runner.invoke(tool_ref, payload, task_id=task_id))
              except PluginError as exc:
                  results.append(ToolResult(ok=False, output=None, error=str(exc),
                                 error_kind="execution", duration_ms=0,
                                 stdout_tail="", stderr_tail=""))
          self._plugins._repo.write_audit(kind="hook.dispatch", target_ref=event,
                                          detail={"task_id": task_id, "count": len(results)})
          return results
        """
```

- **测试要点**：注册带 hook 的 plugin → `rebuild_index` 后 `dispatch` 触发；一个 hook 抛错不影响同事件其它 hook；禁用 plugin 后其 hook 不再触发。

---

## Part G — 加载与安装（`plugins/loader.py`）

### G-0 上传安装信任边界

- 上传 plugin 是**本地管理员显式安装的代码包**，本质上等同于在本机运行第三方 Python 代码；Phase 1 只做结构校验、子进程隔离、资源限制和审计，不承诺安全沙箱。
- 默认服务必须绑定 `127.0.0.1`。如果未来允许远程访问，必须先禁用上传安装接口，或增加鉴权、CSRF 防护和管理员确认。
- `permissions` 在 Phase 1 只用于展示和审计，不是强制权限隔离。任何敏感能力都必须通过平台内置 API/Tool 暴露，不允许 plugin 自行获得数据库连接字符串或密钥。
- 前端上传、启用、禁用、删除 plugin 都必须有明确的人为操作和审计事件；不得由 Agent 自动安装或启用未知 plugin。
- `_extract_to_temp` 必须限制上传 zip 大小，并拒绝 zip slip/path traversal：绝对路径、`..` 路径、解压后不在临时目录内的文件全部报错。
- 上传 manifest 中的 `checksum` 是不可信输入。安装时先解析结构，再由平台对解压后的实际目录计算 checksum，并用平台计算值覆盖 manifest 后注册/持久化。

```python
def load_manifest(plugin_dir: Path, *, builtin: bool) -> PluginManifest:
    """从 plugin 目录读 manifest.json 并解析。
    异常: ManifestError（文件缺失/非法 JSON/结构错）。
    伪代码:
      path = plugin_dir / "manifest.json"
      if not path.is_file(): raise ManifestError(f"no manifest.json in {plugin_dir}")
      data = json.loads(path.read_text(encoding="utf-8"))
      return parse_manifest(data, builtin=builtin)
    """

def compute_checksum(plugin_dir: Path) -> str:
    """对 plugin 目录所有文件算稳定 checksum（sha256 of sorted file contents）。
    出参: hex digest。 不变量: INV-9（排序保证跨平台稳定）。
    伪代码:
      h = hashlib.sha256()
      for f in sorted(plugin_dir.rglob("*")):
          if f.is_file():
              h.update(f.relative_to(plugin_dir).as_posix().encode())
              h.update(f.read_bytes())
      return h.hexdigest()
    """

def install_plugin(upload_path: Path, plugins_dir: Path,
                   registry: PluginRegistry) -> PluginManifest:
    """安装上传的 plugin（zip 或目录）：解包→校验 manifest→校验所有 tool schema 合法性
       →算 checksum→落盘→注册。
    入参: upload_path 上传文件; plugins_dir 安装根目录; registry。
    出参: 已注册的 PluginManifest。
    异常: ManifestError; DuplicatePluginError; PluginError（schema 非法 / 解包失败）。
    不变量: INV-8（安装审计）；安装即信任但仍校验结构；上传方 checksum 不可信。
    伪代码:
      tmp = _extract_to_temp(upload_path)              # zip→解包；目录→拷贝；限制大小并防 zip slip
      manifest = load_manifest(tmp, builtin=False)
      # 校验每个 tool 的 input/output schema 本身是合法 JSON Schema
      for tool in manifest.tools:
          _assert_valid_jsonschema(tool.input_schema)
          _assert_valid_jsonschema(tool.output_schema)
      checksum = compute_checksum(tmp)
      manifest = dataclasses.replace(manifest, checksum=checksum)
      dest = plugins_dir / manifest.name
      if dest.exists() and registry.get(manifest.name).version == manifest.version:
          raise DuplicatePluginError(f"{manifest.name}@{manifest.version} exists")
      _atomic_move(tmp, dest)                          # 落盘
      registry.register(manifest, enabled=True)
      return manifest
    """

def load_builtin_packs(registry: PluginRegistry, packs_root: Path) -> None:
    """启动时发现 packs/*/manifest.json 并注册为内置 plugin。
    伪代码:
      for child in sorted(packs_root.iterdir()):
          mf = child / "manifest.json"
          if mf.is_file():
              manifest = load_manifest(child, builtin=True)
              if not registry._plugins.get(manifest.name):  # 幂等
                  registry.register(manifest, enabled=True)
    """

def _assert_valid_jsonschema(schema: dict) -> None:
    """校验 schema 本身是合法 Draft 2020-12 Schema。异常: PluginError。
    伪代码:
      from jsonschema import Draft202012Validator
      try: Draft202012Validator.check_schema(schema)
      except Exception as exc: raise PluginError(f"invalid json schema: {exc}")
    """
```

- **测试要点**：装合法包→可注册可执行；坏 manifest 包→`ManifestError`；tool schema 非法→`PluginError`；同名同版本重装→`DuplicatePluginError`；`compute_checksum` 对同内容稳定、对改动敏感；上传包自带 checksum 时也被平台计算值覆盖；zip 绝对路径/`..` 路径/超限大小被拒；`load_builtin_packs` 幂等。

---

## Part H — 持久层（`db.py` 新增）

### H-1 表结构（DDL，在 `init_db` 内，走 `connect()` 封装）

```sql
CREATE TABLE IF NOT EXISTS plugins (
  name TEXT PRIMARY KEY,
  version TEXT NOT NULL,
  display_name TEXT NOT NULL,
  description TEXT,
  builtin INTEGER NOT NULL DEFAULT 0,
  manifest_json TEXT NOT NULL,
  checksum TEXT,
  enabled INTEGER NOT NULL DEFAULT 1,
  installed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tools (
  plugin TEXT NOT NULL,
  name TEXT NOT NULL,
  summary TEXT,
  input_schema_json TEXT NOT NULL,
  output_schema_json TEXT NOT NULL,
  determinism TEXT NOT NULL,
  timeout_seconds INTEGER NOT NULL,
  failure_policy TEXT NOT NULL,
  side_effects_json TEXT,
  PRIMARY KEY (plugin, name),
  FOREIGN KEY (plugin) REFERENCES plugins(name) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS audit (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  actor TEXT,
  target_ref TEXT,
  inputs_hash TEXT,
  outcome TEXT,
  detail_json TEXT,
  at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_kind_at ON audit(kind, at);
```

### H-2 `PluginRepository`（方法逐个）

```python
class PluginRepository:
    def __init__(self, db_path: Path):
        self._db_path = db_path

    def upsert_plugin(self, manifest: PluginManifest, *, enabled: bool) -> None:
        """插入或升级 plugin + 其 tools（同事务）。
        不变量: INV-8；DDL/写在单事务（CODE_REVIEW P1-5 教训）。
        伪代码:
          with connect(self._db_path) as conn:
              conn.execute("INSERT INTO plugins(...) VALUES(...) "
                           "ON CONFLICT(name) DO UPDATE SET version=...,manifest_json=...,enabled=...",
                           (manifest.name, manifest.version, ..., int(enabled), _now_iso()))
              conn.execute("DELETE FROM tools WHERE plugin=?", (manifest.name,))
              for t in manifest.tools:
                  conn.execute("INSERT INTO tools(...) VALUES(...)",
                               (manifest.name, t.name, t.summary,
                                json.dumps(t.input_schema), json.dumps(t.output_schema),
                                t.determinism, t.timeout_seconds, t.failure_policy,
                                json.dumps(list(t.side_effects))))
        """

    def set_enabled(self, name: str, enabled: bool) -> None:
        """伪代码: UPDATE plugins SET enabled=? WHERE name=?；rowcount=0 抛 PluginNotFoundError。"""

    def delete_plugin(self, name: str) -> None:
        """伪代码: DELETE FROM plugins WHERE name=?（tools 经 FK CASCADE 删）。"""

    def get_plugin(self, name: str) -> dict | None:
        """伪代码: SELECT * 单行→ dict 或 None。"""

    def list_plugins(self, *, include_disabled: bool = False) -> list[dict]:
        """伪代码: SELECT *；include_disabled=False 加 WHERE enabled=1。"""

    def list_tools(self) -> list[dict]:
        """伪代码: SELECT * FROM tools。"""

    def write_audit(self, *, kind: str, target_ref: str, actor: str = "system",
                    inputs_hash: str | None = None, outcome: str | None = None,
                    detail: dict | None = None) -> None:
        """INV-8 审计写入。
        伪代码:
          with connect(self._db_path) as conn:
              conn.execute("INSERT INTO audit(id,kind,actor,target_ref,inputs_hash,outcome,detail_json,at) "
                           "VALUES(?,?,?,?,?,?,?,?)",
                           (uuid4hex(), kind, actor, target_ref, inputs_hash, outcome,
                            json.dumps(detail or {}), _now_iso()))
        """
```

- **测试要点**（`tests/test_db.py` 风格）：upsert 往返；升级覆盖旧 tools；FK CASCADE 删 tools；`set_enabled` 对不存在抛错；审计可查询；并发连接不锁死（Phase 0 已保证 `connect()` 封装）。

---

## Part I — HTTP 管理端点（`routers/plugins.py`）

```python
router = APIRouter(prefix="/api/plugins", tags=["plugins"])

@router.get("")
def list_plugins(request: Request, include_disabled: bool = False) -> dict:
    """列出 plugins。
    出参: {"plugins": [{name,version,display_name,enabled,builtin,tool_count}, ...]}。
    伪代码: registry = request.app.state.plugin_registry
            return {"plugins": [_public_plugin(m, registry.is_enabled(m.name))
                                for m in registry.list(include_disabled=include_disabled)]}
    """

@router.post("", status_code=201)
async def upload_plugin(request: Request, file: UploadFile) -> dict:
    """上传安装 plugin（zip）。
    出参: 201 + {name, version, tool_count}。
    异常→HTTP: ManifestError→422; DuplicatePluginError→409; PluginError→400。
    安全边界: 仅本机管理员显式上传；远程部署必须禁用或增加鉴权。
    伪代码:
      tmp = await _save_upload(file)
      try:
          manifest = install_plugin(tmp, settings.plugins_dir, request.app.state.plugin_registry)
          request.app.state.hook_dispatcher.rebuild_index()
          return {"name": manifest.name, "version": manifest.version,
                  "tool_count": len(manifest.tools)}
      except DuplicatePluginError as exc: raise HTTPException(409, str(exc))
      except ManifestError as exc: raise HTTPException(422, str(exc))
      except PluginError as exc: raise HTTPException(400, str(exc))
    """

@router.post("/{name}/enable")
def enable_plugin(request: Request, name: str) -> dict:
    """启用。异常→HTTP: PluginNotFoundError→404。
    伪代码: registry.set_enabled(name, True); hook_dispatcher.rebuild_index(); return {"ok": True}
    """

@router.post("/{name}/disable")
def disable_plugin(request: Request, name: str) -> dict:
    """禁用。内置包可禁用但不可删。"""

@router.delete("/{name}")
def remove_plugin(request: Request, name: str) -> dict:
    """删除（内置包→400）。异常→HTTP: PluginNotFoundError→404; 删内置→400。"""

@router.get("/{name}/tools")
def list_plugin_tools(request: Request, name: str) -> dict:
    """列出某 plugin 的 tools（含 input/output schema，供前端 plugin 管理页展示）。"""
```

辅助 `_public_plugin(manifest, enabled) -> dict`：只暴露展示字段，不含 checksum/module/entrypoint。

- **测试要点**（`test_api_v2.py` 风格）：上传合法 zip→201→可在 list 看到；坏 manifest→422；重复→409；enable/disable 改变 list 结果；删内置→400；删上传包→204/200；`/tools` 返回 schema。

---

## Part J — Trivial 内置包（`packs/_sample/`，证明链路）

`manifest.json`：

```json
{
  "name": "_sample",
  "version": "0.1.0",
  "display_name": "Sample Echo Pack",
  "description": "Runtime smoke-test pack",
  "module": "marvis.packs._sample.tools",
  "tools": [{
    "name": "echo",
    "summary": "Echo back the given message",
    "input_schema": {"type": "object", "properties": {"message": {"type": "string"}},
                     "required": ["message"]},
    "output_schema": {"type": "object", "properties": {"echoed": {"type": "string"}},
                      "required": ["echoed"]},
    "determinism": "deterministic", "timeout_seconds": 10,
    "failure_policy": "fail", "entrypoint": "tool_echo"
  }],
  "hooks": [], "permissions": []
}
```

`tools.py`：

```python
def tool_echo(inputs: dict, ctx) -> dict:
    """trivial tool: 回显 message，证明 runner→worker→schema 全链路。
    入参: inputs={"message": str}; ctx: ToolContext。
    出参: {"echoed": str}。
    """
    return {"echoed": inputs["message"]}
```

- **用途**：集成测试用它验证 `ToolRunner.invoke` 端到端（含子进程、schema 校验、审计）。

---

## Part K — 装配（`app.state` 接线）

在 FastAPI 启动（`api.py` 的 lifespan/startup）：

```python
# 伪代码:
repo = PluginRepository(settings.db_path)
plugin_registry = PluginRegistry(repo); plugin_registry.load_from_db()
load_builtin_packs(plugin_registry, settings.packs_root)   # 注册内置包
tool_registry = ToolRegistry(plugin_registry)
tool_runner = ToolRunner(tool_registry, repo,
                         python_executable=load_execution_environment(settings.workspace).python_path,
                         datasets_root=settings.datasets_root, workspace=settings.workspace)
hook_dispatcher = HookDispatcher(plugin_registry, tool_runner); hook_dispatcher.rebuild_index()
app.state.plugin_registry = plugin_registry
app.state.tool_registry = tool_registry
app.state.tool_runner = tool_runner
app.state.hook_dispatcher = hook_dispatcher
app.include_router(plugins_router)
```

- **测试要点**：启动后 `_sample` 包已注册；`app.state.tool_runner.invoke(ToolRef("_sample","echo","0.1.0"), {"message":"hi"}, task_id="t")` 返回 `{"echoed":"hi"}`。

---

## Part L — 测试计划汇总

| 文件 | 覆盖 |
|------|------|
| `tests/test_plugin_manifest.py` | parse/validate manifest 边界、往返 |
| `tests/test_plugin_schema.py` | `validate_against_schema` 正/反例 |
| `tests/test_plugin_registry.py` | 注册/启停/删除/resolve/catalog |
| `tests/test_plugin_runner.py` | echo 往返、schema 失败、execution 失败、**timeout 无僵尸**、output schema 失败、stochastic seed 复现、**Windows setrlimit 降级** |
| `tests/test_plugin_hooks.py` | dispatch、失败隔离、禁用不触发 |
| `tests/test_plugin_loader.py` | install/checksum/builtin 发现/重复 |
| `tests/test_plugin_db.py` | repository 往返、FK CASCADE、审计 |
| `tests/test_plugin_api.py` | 上传/列出/启停/删除 HTTP 状态码 |

跨平台：runner 的 timeout/资源限制测试要在 CI 的 Linux 与（若有）Windows 都跑；`subprocess_worker` 用 `python -m` 方式起，避免路径问题（INV-9）。

---

## Part M — 任务执行顺序

```text
1. A 异常层级          （无依赖）
2. B manifest 契约+解析 （依赖 A）
3. C schema 校验        （依赖 A，加 jsonschema 依赖）
4. H DB 表+Repository   （依赖 Phase 0 的 connect 封装）
5. D registry           （依赖 B,H）
6. G loader             （依赖 B,C,D）
7. E worker + runner    （依赖 B,C,D,H；核心，最花时间）
8. F hooks              （依赖 D,E）
9. J _sample 内置包      （依赖 B；给 E/K 测试用）
10. K app.state 装配     （依赖全部）
11. I HTTP router        （依赖 D,G,F,K）
12. L 测试补齐 + 全量回归
```

每项一个 atomic commit。Phase 1 完成标志：`_sample.echo` 能经 `ToolRunner` 子进程往返、schema 校验生效、timeout 能杀进程无残留、plugin 可经 HTTP 上传/启停、审计有记录、跨平台测试绿。

---

*Phase 1 是 V2 的承重墙。它本身不做业务，但 Phase 2（编排）和 Phase 3+（能力包）全部站在这套 Tool 契约 + 子进程 runner 上。*
