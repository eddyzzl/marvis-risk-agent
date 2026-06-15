# Phase 0 — 地基（函数级 spec）

## 文档状态

- 状态：待实施
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`
- 债务来源：`CODE_REVIEW_2026-06-13.md`。本 spec 已内联必须修复的接口/测试契约；提交实现时，审计附件可随同提交或移入 `docs/superpowers/reviews/`。
- 目标：在盖 V2 楼之前还清地基债务——修 P0/P1 缺陷、拆分前端、补边界测试、上 ruff F811。**不引入任何新业务功能**。

## 颗粒度说明（给所有 Phase spec 的实现者）

每个工作项格式固定：

```
### 任务编号 简述
- 文件:行号 / 不变量
- 现状问题（证据）
- 目标函数签名（入参类型+语义 / 出参类型+语义 / 异常）
- 实现要点（一步步）
- 测试要点（必须覆盖的用例，尤其边界）
- 验收标准（怎么算做完）
```

一个任务一个 commit。改完跑该任务的测试要点 + 相关回归。

---

## Part A — P0 修复（3 项）

### A-1 修复 `_roc_ks_curve` 的 `population_at_ks` 语义错误

- **文件**：`marvis/validation/effectiveness.py:443-465`（`_roc_ks_curve`）
- **不变量**：INV-1（指标正确性）
- **现状问题**：`population_at_ks=float(fpr[ks_index])` 返回的是累计 good 占比（FPR），而非"达到最大 KS 时的累计人口占比"。信贷场景 bad rate 低，二者差异显著，KS 曲线分位标记点位置错误。

- **目标函数签名**（签名不变，行为修正）：
  ```python
  def _roc_ks_curve(*, split: str, scores: np.ndarray, labels: np.ndarray) -> RocKsCurve
  ```
  - 入参：
    - `split: str` — 样本分组名（"train"/"test"/"oot"），仅透传到结果。
    - `scores: np.ndarray[float]` — 该 split 的模型分，已保证 finite。
    - `labels: np.ndarray[int]` — 0/1 标签，1=bad。
  - 出参：`RocKsCurve` dataclass，其中：
    - `population_at_ks: float` — **修正为**：达到最大 |KS| 的阈值点处，累计样本数 / 总样本数 ∈ [0,1]。
  - 异常：样本为空或单类别时由上游 `run_effectiveness` 拦截；本函数假设输入非空、双类别。

- **实现要点**：
  1. 已有 `threshold_indexes = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(sorted_scores) - 1]`。
  2. 新增累计人口占比：`population = np.r_[0.0, (threshold_indexes + 1) / len(sorted_scores)]`。
  3. `population` 与 `tpr`/`fpr` 同长度（前置 0.0 对齐）。
  4. `population_at_ks = float(population[ks_index])`。

- **测试要点**（`tests/test_model_algorithms.py`）：
  - 构造 bad rate=5% 的不平衡样本，断言 `population_at_ks` ≈ 排序位置占比，**不等于** `fpr[ks_index]`。
  - 平衡样本（50/50）下 `population_at_ks` 与 `fpr[ks_index]` 接近但不必相等。
  - `population_at_ks ∈ [0, 1]`。
  - 回归：`ks` 值本身不变。

- **验收**：新增不平衡样本用例通过；现有 KS 数值回归不变；前端消费该字段的 KS 曲线渲染同步核对（见 C-7）。

---

### A-2 修复 `_timestamp_without_timezone` 对 tz-aware 时间崩溃

- **文件**：`marvis/validation/time_periods.py:106-111`
- **不变量**：INV-1（月度分析可用性）
- **现状问题**：对带时区的 `Timestamp` 直接 `tz_localize(None)` 在部分 pandas 版本抛 `TypeError`，含 `+08:00` 的时间列导致月度 KS/PSI 失败。
  > 实测修正（pandas 3.0.3）：`Timestamp.tz_localize(None)` 在该版本**不抛错**，直接返回钟面时间——原 CODE_REVIEW P0-2 在此版本是误报。但跨版本不保证，仍应显式处理。

- **目标函数签名**（签名不变）：
  ```python
  def _timestamp_without_timezone(value) -> pd.Timestamp
  ```
  - 入参：`value` — 任意可被 `pd.Timestamp()` 解析的值（字符串/datetime/Timestamp），可能 tz-aware 或 naive。
  - 出参：`pd.Timestamp` — 去除时区信息后的 naive timestamp。
  - 异常：不可解析时返回 `pd.NaT`（由上游 `_coerce_timestamp` 处理）；本函数不抛 `TypeError`。

- **实现要点**（保留钟面时间，**不**做 UTC 归一）：
  ```python
  if timestamp.tzinfo is not None:
      return timestamp.tz_convert(timestamp.tz).tz_localize(None)  # 保留本地钟面，不抛错
  return timestamp
  ```
  - **口径决策（已实测确认）**：保留钟面时间，**禁止** `tz_convert("UTC")`。
    UTC 归一会把 `2025-01-01 00:30+08:00` 挪到 `2024-12-31 16:30`，边界样本**跨月/跨年**，污染月度分组。
    月度分析要的是"申请发生在哪个月"的本地口径，钟面时间才对。
    （本条已在实现中采用 `tz_convert(timestamp.tz).tz_localize(None)`。）

- **测试要点**（`tests/` 新增或 `test_metric_tables.py`）：
  - `"2025-01-01T08:00:00+08:00"` 能正常解析为 naive，不抛异常。
  - naive 时间 `"2025-01-01"` 行为不变。
  - 混合 tz-aware 和 naive 的时间列，`parse_time_series` 不崩溃，按月正确分组。
  - 不可解析值返回 `NaT`。

- **验收**：含时区样本的月度分析端到端跑通。

---

### A-3 修复 LLM 客户端无条件发送 DeepSeek 专有字段

- **文件**：`marvis/llm_client.py:33-43`；`marvis/llm_settings.py`（save 字段）
- **不变量**：roadmap "手动模式和 Agent P1 模式都必须可用"
- **现状问题**：`reasoning_effort` 和 `thinking: {"type": "enabled"}` 无条件注入所有请求，非 DeepSeek 提供商返回 400，P1 模式整体不可用。

- **目标函数签名**（`complete` 增加可选行为，靠 profile 控制）：
  ```python
  def complete(self, *, model_name, system_prompt, user_prompt,
               temperature=..., response_format=None,
               on_delta=None, stream: bool = True) -> str
  ```
  - 新增 `stream: bool = True` 入参（同时服务 A-3 与 P1-17）：是否流式。JSON 场景传 `False`。
  - profile 新增字段：
    - `enable_thinking: bool`（默认 `False`）— 是否发送 `thinking`/`reasoning_effort`。
    - `reasoning_effort: str`（默认 `"high"`，仅 `enable_thinking=True` 时生效）。

- **实现要点**：
  ```python
  payload = {"model": model_name, "messages": [...], "stream": stream, "temperature": temperature}
  if self.profile.get("enable_thinking"):
      payload["reasoning_effort"] = str(self.profile.get("reasoning_effort") or "high")
      payload["thinking"] = {"type": "enabled"}
  if response_format:
      payload["response_format"] = response_format
  ```
  - `llm_settings.save_llm_settings` 增加 `enable_thinking` 字段持久化。
  - 前端 LLM 设置表单加"启用思考模式（仅 DeepSeek 等支持）"开关（见 C 部分前端，或先后端默认关、前端后补）。

- **测试要点**（`tests/test_llm_client.py`、`tests/test_llm_settings.py`）：
  - 默认 profile（无 `enable_thinking`）的 payload **不含** `thinking`/`reasoning_effort`。
  - `enable_thinking=True` 时 payload 含两字段。
  - `stream=False` 时 payload `stream` 为 `False`。
  - `save_llm_settings` 往返保存 `enable_thinking`。

- **验收**：模拟非 DeepSeek provider（拒绝未知字段）时默认 profile 调用成功。

---

## Part B — P1 修复（后端，11 项）

### B-1 删除重复函数定义（`_write_metrics_cancel_marker` ×2、`METRICS_CANCEL_MARKER` ×2、`_metrics_cancel_marker_path` ×2）

- **文件**：`api.py:3009`（删）、`pipeline.py:80`（删常量副本）、`pipeline.py:1447-1448`（删函数副本）
- **不变量**：INV（可维护性）
- **现状问题**：当前分支合并事故，同名重复定义，静默保留最后一个。
- **实现要点**：保留首处定义，删除后处副本。确认两处实现一致后再删（已核验一致）。
- **测试要点**：`node --check` 无关；`python -c "import marvis.api, marvis.pipeline"` 无错；ruff F811 不再报（见 D-1）。
- **验收**：ruff F811 清零；现有 cancel 相关测试（`test_notebook_cancellation.py`）通过。

### B-2 修复 `_agent_has_stop_ack_message` 只检查最后一条消息

- **文件**：`api.py:2348-2358`
- **现状问题**：遇到第一条 assistant/user 消息即 return，应是"历史中是否存在 stop ack"。
- **目标函数签名**：
  ```python
  def _agent_has_stop_ack_message(repo: TaskRepository, task_id: str) -> bool
  ```
  - 入参：`repo` 任务仓库；`task_id` 任务 id。
  - 出参：历史消息中**是否存在**一条 `role=assistant, intent=stop, cancel_requested=True` 的 ack。
- **实现要点**：正向遍历，命中即 `return True`，遍历完 `return False`（去掉 `reversed` + 提前 return 的逻辑）。
- **测试要点**：stop ack 后再追加普通用户消息，仍返回 True（不重复插 ack）；无 ack 时 False。
- **验收**：取消后对话不出现重复 stop ack。

### B-3 修复 `delete_task` 孤儿目录（Windows）

- **文件**：`api.py:563-580`
- **不变量**：INV-9（跨平台）
- **现状问题**：先删 DB 后 `rmtree`，Windows 上文件占用导致 DB 已删、目录残留。
- **目标行为**：调整顺序——先 `close_live_notebook_session(task_id)` → 再 `repo.delete_task(task_id)` → 最后 `rmtree` 容错。
  ```python
  close_live_notebook_session(task_id)
  repo.delete_task(task_id)
  try:
      if task_dir.exists():
          shutil.rmtree(task_dir)
  except OSError as exc:
      logger.warning("task dir cleanup failed for %s: %s", task_id, exc)
  ```
- **测试要点**：模拟 `rmtree` 抛 `OSError`，断言 DB 记录仍被删、不抛异常、记 warning。
- **验收**：删除流程在文件占用下不留 DB 孤儿、不崩。

### B-4 修复 `update_report_fields` 参数 `payload` 被遮蔽

- **文件**：`api.py:1081-1116`
- **现状问题**：参数 `payload: ReportFieldsUpdateRequest` 被局部 `dict|None` 重新赋值。
- **目标行为**：局部变量改名 `results_payload`，参数 `payload` 不被覆盖。
- **测试要点**：现有 `test_api_v2.py` 报告字段更新用例回归通过。
- **验收**：函数内 `payload` 全程保持 `ReportFieldsUpdateRequest` 类型。

### B-5 统一 `_resolve_scan_material` 路径校验走 `assert_within`

- **文件**：`api.py:3146-3158`；`marvis/safe_paths.py`
- **不变量**：INV-9 / 安全
- **现状问题**：用 `relative_to` 自行校验，与项目统一入口 `assert_within` 不一致。
- **目标行为**：改用 `assert_within(source_dir, resolved)`，捕获 `PermissionError` 返回友好错误 `(None, "配置的 {label} 必须位于材料目录内")`。
- **测试要点**：`test_safe_paths.py` 风格——目录外路径被拒、目录内通过、符号链接逃逸被拒。
- **验收**：路径校验逻辑全平台统一。

### B-6 修复 `_capture_user_preference_memory` 静默吞异常

- **文件**：`api.py:2524-2547`
- **不变量**：INV-8（审计）
- **现状问题**：`except Exception: return` 吞掉所有异常，记忆保存失败用户零反馈。
- **目标行为**：捕获后记 warning 日志；在对应 assistant 回复 metadata 写 `memory_save_failed: True`（前端可后续展示软提醒）。
- **测试要点**：模拟 `store.create` 抛异常，断言记日志、metadata 标记、主流程不中断。
- **验收**：记忆保存失败可观测、可审计。

### B-7 修复 `run_notebook_stage` session 注册顺序

- **文件**：`pipeline.py:143-157`
- **不变量**：INV-6（隔离）
- **现状问题**：先 `register_live_notebook_session` 后 `update_status`，后者失败时 registry 残留 + 状态失控。
- **目标行为**：调换顺序——先 `update_status(EXECUTED, expected=RUNNING)` 成功，再 `register_live_notebook_session`。失败路径只需 close 未注册 session。
- **测试要点**：模拟 `update_status` 抛异常，断言 session 未进 registry、被 close、无残留。
- **验收**：kernel 生命周期与任务状态不再脱钩。

### B-8 修复 `NotebookExecutionSession.close()` kernel 泄漏

- **文件**：`notebooks.py:174-183`
- **不变量**：INV-6
- **现状问题**：依赖 nbclient 私有 `_cleanup_kernel`，方法不存在时静默不关，kernel 进程泄漏。
- **目标函数签名**：
  ```python
  def close(self) -> None
  ```
  - 出参：无；幂等（`self.closed` 守卫）。
- **实现要点**：加公共 API 回退链：
  ```python
  cleanup = getattr(self.client, "_cleanup_kernel", None)
  if callable(cleanup):
      cleanup()
  elif getattr(self.client, "km", None) is not None:
      try: self.client.km.shutdown_kernel(now=True)
      except Exception: pass
  ```
- **测试要点**：mock client 无 `_cleanup_kernel` 但有 `km`，断言走 `shutdown_kernel`；二次 close 幂等。
- **验收**：私有 API 缺失时仍能关 kernel；`pyproject.toml` 给 nbclient 加版本约束（见 D-2）。

### B-9 修复 `split_col`/`time_col` 为 NULL 的 `KeyError: None`

- **文件**：`pipeline.py:743-744`（payload）、`pipeline.py:827-831`（注入代码列检查）
- **现状问题**：历史 DB 行 `split_col`/`time_col` 为 NULL 时传 `None` 进注入代码，`_rmc_sample[None]` 抛 `KeyError: None`。
- **目标行为**：
  - payload 构建兜底空串：`"split_col": contract.split_col or task.split_col or ""`。
  - 注入代码列检查跳过空值：`if column and column not in _rmc_sample.columns`。
- **测试要点**：构造 `split_col=None` 的任务，断言报错信息可读（不是 `KeyError: None`），或正确跳过。
- **验收**：NULL 列任务不产生不可调试错误。

### B-10 修复 `recovery.py` 绕过 `connect()` 封装

- **文件**：`recovery.py:47`；`db.py:30-31,183-185`（`init_db` 同问题）
- **不变量**：INV-8
- **现状问题**：裸 `sqlite3.connect`，无 `busy_timeout`/`foreign_keys`/`row_factory`，启动恢复期锁冲突立即报错。
- **目标行为**：`recovery.reclaim_stale_running_tasks` 改用 `from marvis.db import connect`，删手动 `conn.commit()`（封装自动提交）。`init_db` 统一走封装或显式事务包裹 DDL。
- **测试要点**：`test_recovery.py` 回归；并发连接下恢复不抛 `database is locked`。
- **验收**：恢复路径与主连接行为一致。

### B-11 修复流式读取阶段网络异常逃逸 + JSON 场景非流式

- **文件**：`llm_client.py:57-95`；`agent/service.py:653-657`（`generate_word_conclusions`）
- **不变量**：降级路径完整性
- **现状问题**：异常捕获只覆盖建连阶段，流式循环中 `RemoteDisconnected`/`IncompleteRead`/`OSError` 穿透，绕过 fallback，后台 job 崩溃。
- **目标行为**：
  - try 块包住整个 `with urlopen(...)` 体（含流式读取），新增：
    ```python
    except (TimeoutError, OSError, http.client.HTTPException) as exc:
        raise LLMClientError(f"LLM stream interrupted: {exc}") from exc
    ```
  - `generate_word_conclusions` 调 `complete(..., stream=False)`（依赖 A-3 的 `stream` 参数）；`json.JSONDecodeError` 单独捕获，给"LLM 返回非有效 JSON"明确错误。
- **测试要点**：mock 流式中断抛 `RemoteDisconnected`，断言被包成 `LLMClientError`、走 fallback；JSON 解析失败给区分错误。
- **验收**：流式中断不再崩后台 job。

---

## Part C — 前端 ES Module 拆分（核心结构 + 关键函数）

> 目标：把 6300 行 `app.js` 拆成无构建 ES Modules（蓝图第 11 节），迁移中一并修 CODE_REVIEW 前端问题。**行为不变**，纯结构重构 + 已知缺陷修复。

### C-0 模块骨架与装配

- **新增文件**：`static/js/core/{state,bus,api,poll}.js`、`static/js/views/*.js`、`static/js/render/*.js`、`static/js/main.js`。
- **`index.html`**：`<script src="static/app.js">` 改为 `<script type="module" src="static/js/main.js">`。
- **迁移策略**：按功能区逐块搬迁，每搬一块跑 `tests/test_frontend_static_v2.py` 回归（该测试约 6000 行，是行为护栏）。
- **验收**：所有现有前端测试通过；`node --check` 每个模块通过。

### C-1 `core/state.js` — 集中状态

- **导出**：
  ```js
  export const state = { tasks, selectedTaskId, ... }   // 集中可变状态
  export function getState(key)
  export function setState(key, value)   // 触发订阅
  export function subscribe(key, fn)     // 返回 unsubscribe
  ```
  - 把散落的全局变量（`selectedTaskId`、`metricTooltipAttached`、`agentTypingTimer` 等）收敛进 `state`。
- **测试要点**：`setState` 触发对应 `subscribe` 回调；`unsubscribe` 后不再触发。

### C-2 `core/api.js` — fetch 封装

- **导出**：
  ```js
  export async function apiGet(path)
  export async function apiPost(path, body, options = {})
  export async function apiDelete(path)
  ```
  - 出参：解析后的 JSON；非 2xx 抛结构化 `ApiError {status, detail}`。
  - `apiPost` 分两类 body：
    - `body instanceof FormData`：直接传给 `fetch`，不得手动设置 `Content-Type`，让浏览器生成 multipart boundary。
    - 普通 object：`JSON.stringify(body)`，并设置 `Content-Type: application/json`。
  - `options` 仅承载 `signal`、额外 header 等 fetch 安全参数；调用方不得绕过统一错误处理。
- **测试要点**：4xx/5xx 抛 `ApiError`；网络失败有明确错误；JSON body 设置 `application/json`；FormData body 不设置 `Content-Type` 且不被 stringify；422 detail 能原样进入 `ApiError.detail`。

### C-3 `core/poll.js` — 统一轮询管理（修 P2-21）

- **现状问题**：背景轮询与主动操作轮询并行，同任务双倍请求。
- **导出**：
  ```js
  export function startPoll(key, fn, intervalMs)   // 同 key 去重，已存在则不重复起
  export function stopPoll(key)
  export function stopAllPolls()
  ```
  - 不变量：同一 `key`（如 `progress:{taskId}`）只有一个活跃轮询。
- **测试要点**：同 key 二次 `startPoll` 不新增定时器；`stopPoll` 后停止。

### C-4 `render/markdown.js` — Markdown 渲染（修 P2-20）

- **现状问题**：链接渲染在已转义文本上正则替换 `$2`，模式脆弱。
- **目标函数**：
  ```js
  export function renderMarkdownInlineText(content)   // 返回安全 HTML 字符串
  ```
  - 实现要点：先解析 markdown 结构再构造 DOM/转义 href，协议白名单保持 `https?://` `/` `#`（已挡 `javascript:`/`data:`）。href 值独立转义。
- **测试要点**：含特殊字符 URL 的链接正确转义；`javascript:`/`data:` 被拒；普通文本不受影响。

### C-5 `views/` — 新增 V2 视图骨架（占位，Phase 1+ 填充）

- 本阶段只建**空骨架 + 路由挂载点**：`task_tree.js`、`subagent.js`、`plugins.js`、`workflow.js`、`artifacts.js` 各导出一个 `render{X}(container, data)` 空实现 + 注释 TODO 指向对应 Phase。
- **验收**：骨架可加载、不报错、不影响现有页面。

### C-6 修复 `taskStopped`/`taskFailureStage` 文本匹配（修 P1-22，前端侧）

- **文件**：迁移到 `render/` 或 `core/` 时一并改。
- **现状问题**：靠 `status_message` 文本匹配判断业务状态，违反 AGENTS.md。
- **目标行为**：
  - `taskStopped(task)`：优先读结构化 `task.stopped`（后端 B-12 提供）；缺失时返回 `false`（不再猜文本）。
  - `taskFailureStage(task)`：优先 `task.failure_stage`；缺失返回 `null` 显示"未知阶段"。
  - 保留的兼容兜底加注释 + TODO 标删除条件。
- **依赖**：后端 B-12（结构化字段）。
- **测试要点**：有结构化字段时用字段；缺失时不靠文本猜。

### C-7 核对 KS 曲线渲染消费 `population_at_ks`（配合 A-1）

- **现状**：A-1 修正了 `population_at_ks` 语义。
- **目标行为**：前端 KS 曲线分位标记点读 `population_at_ks` 作为横轴人口占比；核对修正后位置正确。
- **测试要点**：用 A-1 的不平衡样本 fixture，断言标记点横坐标用人口占比。

---

## Part D — 后端配套（结构化字段 + 工程约定）

### D-1 上 ruff F811（重定义检查）

- **文件**：`pyproject.toml`（ruff 配置）；CI/验证命令。
- **目标**：ruff 规则集加 `F811`，CI 拦截重复定义（防 B-1 类问题复发）。
- **验收**：`ruff check` 对当前代码（B-1 修复后）F811 清零。

### D-2 `pyproject.toml` 依赖版本约束

- **文件**：`pyproject.toml:8-20`
- **现状问题**：依赖无版本下限，pydantic v1/v2、pandas 1/2 破坏性差异。
- **目标**：加 `pydantic>=2`、`pandas>=2`、`nbclient>=0.7,<0.11`（配合 B-8）、`nbformat>=5`、`duckdb>=0.9`（为 Phase 3 预备）。
- **验收**：干净环境 `pip install` 拿到兼容版本。

### D-3 后端结构化 `stopped` / `failure_stage` 字段（支撑 P1-22）

- **文件**：`api.py`（任务 payload 构建）；`pipeline.py`（取消/失败路径写入）；`db.py`（字段持久化）。
- **不变量**：AGENTS.md「不解析后端自由文本作为业务事实」
- **现状问题**：前端靠 `status_message` 文本判断停止/失败阶段；后端 `_stage_returned_cancelled_task`（`api.py:308-315`）也在做字符串匹配。
- **目标行为**：
  - 任务表新增 `stopped: bool`、`failure_stage: str|None`（"scan"/"notebook"/"metrics"/"report"）。
  - 取消路径显式写 `stopped=True`；失败路径显式写 `failure_stage`。
  - 任务 payload 输出这两个结构化字段。
  - `_stage_returned_cancelled_task` 改读结构化标志，不再字符串匹配。
- **测试要点**（`test_api_v2.py`）：取消任务 payload `stopped=True`；各阶段失败 `failure_stage` 正确；不依赖 `status_message` 文案。
- **验收**：前端 C-6 可完全依赖结构化字段。

---

## Part E — 验收与回归

### E-1 全量回归

- 运行：
  ```bash
  conda run -n py_313 python -m pytest -q
  conda run -n py_313 python -m ruff check marvis tests --extend-exclude '*.ipynb'
  node --check marvis/static/js/main.js   # 及各模块
  git diff --check
  ```
- **验收**：全绿；F811 清零；前端模块化后 `test_frontend_static_v2.py` 通过。

### E-2 新增边界测试清单（CODE_REVIEW 暴露的盲区）

确保以下边界各有用例：
- 不平衡样本的 `population_at_ks`（A-1）
- tz-aware 时间列月度分析（A-2）
- 非 DeepSeek provider 默认 payload（A-3）
- stop ack 后追加消息不重复（B-2）
- `rmtree` 失败不留 DB 孤儿（B-3）
- NULL split_col 任务（B-9）
- 流式中断走 fallback（B-11）
- 结构化 stopped/failure_stage（D-3）
- `core/api.js` FormData 上传不覆盖 multipart boundary（C-2，供 plugin/dataset 上传共用）

---

## Part F — Phase 0 任务清单（执行顺序）

```text
1. A-1, A-2, A-3        P0 修复（独立，可并行）
2. B-1                  删重复定义（先做，解锁 F811）
3. D-1                  上 F811
4. B-2~B-11             P1 后端修复（多数独立）
5. D-2                  依赖约束
6. D-3                  结构化字段（前端 C-6 依赖它）
7. C-0~C-5              前端拆分骨架 + 迁移
8. C-6, C-7             前端配合后端字段 + A-1
9. E-1, E-2             全量回归 + 边界测试
```

每项一个 atomic commit，commit message 带 `Tested:` trailer。Phase 0 完成的标志：CODE_REVIEW 的 P0 全清、P1 后端全清、前端进入模块化结构、边界测试补齐、CI 有 F811 护栏。

---

*Phase 0 是后续所有 Phase 的地基。它不交付新功能，但它决定了 V2 楼盖在干净还是带病的地基上。*
