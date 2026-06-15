# MARVIS Risk Agent 全面代码与设计审查报告

- 审查日期：2026-06-13
- 审查对象：当前工作树（分支 `codex/windows-create-task-422`，含未提交修改）
- 审查范围：`marvis/` 全部后端模块、`static/` 前端、`validation/` 算法层、`agent/` 与 `agent_memory/`、`output/` 渲染层、CLI 与发布脚本、全部设计文档
- 方法：按模块并行深度审查（8 个独立审查通道），所有 P0 级发现已由人工二次核验代码确认

## 总览

| 级别 | 数量 | 含义 |
|------|------|------|
| P0 | 3 | 错误结果 / 核心功能不可用，应立即修复 |
| P1 | 22 | 真实功能缺陷、架构约束违反、数据一致性风险 |
| P2 | 18 | 改进建议、健壮性加固、可维护性 |

**最值得优先处理的三件事：**

1. `population_at_ks` 返回的是 FPR 而非人口占比 —— KS 曲线标记点数值错误（P0-1）
2. 带时区的时间字段会让整个月度分析崩溃（P0-2）
3. LLM 客户端无条件发送 DeepSeek 专有字段，配置其他 OpenAI 兼容模型时 Agent P1 模式整体不可用（P0-3）

---

## P0 — 错误结果 / 核心功能不可用

### P0-1 `population_at_ks` 语义错误：返回 FPR 而非累计人口占比

- **位置**：`marvis/validation/effectiveness.py:464`
- **置信度**：高（已人工核验）

```python
ks_index = int(np.argmax(np.abs(ks_curve)))
return RocKsCurve(
    ...
    ks=float(abs(ks_curve[ks_index])),
    population_at_ks=float(fpr[ks_index]),   # ← 这是累计好样本占比（FPR），不是人口占比
)
```

**问题**：字段名为 `population_at_ks`（达到最大 KS 时的累计样本占比），实际赋值是 `fpr[ks_index]`（累计 good 样本占比）。当好坏样本比例失衡时（信贷场景 bad rate 通常 <10%，两者差异显著），前端/报告中 KS 曲线的分位标记点会落在错误位置。

**修复方案**：基于排序后样本位置计算真实人口占比：

```python
threshold_indexes = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(sorted_scores) - 1]
population = np.r_[0.0, (threshold_indexes + 1) / len(sorted_scores)]
...
population_at_ks=float(population[ks_index]),
```

修复后需同步检查前端 `app.js` 中消费该字段的 KS 曲线渲染逻辑，并补一条单测：构造 bad rate=5% 的不平衡样本，断言 `population_at_ks` 等于排序位置占比而非 FPR。

---

### P0-2 带时区的时间字段导致月度分析整体崩溃

- **位置**：`marvis/validation/time_periods.py:107-111`
- **置信度**：高（已人工核验）

```python
def _timestamp_without_timezone(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)   # ← pandas 对 tz-aware Timestamp 调 tz_localize 直接抛 TypeError
    return timestamp
```

**问题**：pandas 规定对已带时区的 `Timestamp` 必须先 `tz_convert` 再 `tz_localize(None)`，直接 `tz_localize(None)` 抛 `TypeError: Cannot localize tz-aware Timestamp, use tz_convert for conversions`。任何样本里 `time_col` 含 `2025-01-01T08:00:00+08:00` 这类 ISO 带时区字符串，月度 KS/PSI 全部失败，且报错信息与真实原因（时区）无关，用户无法自查。

**修复方案**：

```python
def _timestamp_without_timezone(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp
```

注意：是否先转 UTC 再去时区，还是直接保留本地钟面时间（`tz_localize(None)` 的钟面语义），取决于业务口径——建议保留钟面时间用 `timestamp.tz_localize(None)` 改为正确写法 `pd.Timestamp(timestamp.tz_localize(None) if ... )` 不可行，正确的"保留钟面"写法是 `timestamp.tz_convert(timestamp.tz).tz_localize(None)` 等价于直接 `timestamp.tz_localize(None)` 的意图。最简单且确定的方案就是上面的 UTC 归一。补充单测：带 `+08:00` 后缀的时间列样本应能正常按月分组。

---

### P0-3 LLM 客户端无条件发送 DeepSeek 专有字段，非 DeepSeek 提供商下 Agent P1 模式不可用

- **位置**：`marvis/llm_client.py:33-43`
- **置信度**：高（已人工核验）

```python
payload = {
    "model": model_name,
    ...
    "stream": True,
    "reasoning_effort": reasoning_effort,     # DeepSeek/o系列扩展字段
    "thinking": {"type": "enabled"},          # 私有扩展字段
    "temperature": temperature,
}
```

**问题**：`reasoning_effort` 和 `thinking: {"type": "enabled"}` 被无条件注入所有请求。多数 OpenAI 兼容提供商（Qwen/Moonshot/各类网关）对未知字段或不支持的字段返回 `400 Bad Request`，所有 LLM 调用全部失败。`AGENTS.md` 明确要求"手动模式和 Agent P1 模式都必须可用"——配置非 DeepSeek 模型时该承诺被打破。

**修复方案**：把扩展字段做成 profile 配置项，默认不发送：

```python
extra = self.profile.get("extra_request_fields") or {}
if self.profile.get("enable_thinking"):           # 默认 False
    payload["reasoning_effort"] = reasoning_effort
    payload["thinking"] = {"type": "enabled"}
payload.update(extra)
```

同时在 `llm_settings.py` 的保存字段中加入 `enable_thinking`（并在前端 LLM 设置表单加开关），在 `tests/test_llm_client.py` 补"默认 payload 不含 thinking/reasoning_effort"的断言。

---

## P1 — 功能缺陷 / 架构约束违反 / 数据一致性

### 后端 API（api.py）

#### P1-1 `_write_metrics_cancel_marker` 在 api.py 中重复定义两次

- **位置**：`api.py:1218` 和 `api.py:3009`（已人工核验，两处实现相同）
- 同类问题：`pipeline.py:76/80` 重复定义 `METRICS_CANCEL_MARKER`，`pipeline.py:83/1447` 重复定义 `_metrics_cancel_marker_path`（已核验）。

**问题**：Python 静默保留最后一个定义，当前不产生行为差异，但这是合并/粘贴事故的明确信号（当前分支未提交修改引入）。未来只改其中一处会导致静默分叉。

**修复**：删除 `api.py:3009` 的副本、`pipeline.py:80` 的常量副本、`pipeline.py:1447-1448` 的函数副本。建议给 ruff 配置开启 `F811`（redefinition）检查，CI 直接拦截这类问题。

#### P1-2 `_agent_has_stop_ack_message` 只检查最后一条消息，导致重复 stop ack

- **位置**：`api.py:2348-2358`（已人工核验）

```python
for message in reversed(repo.list_agent_messages(task_id)):
    if message.get("role") not in {"assistant", "user"}:
        continue
    metadata = message.get("metadata") or {}
    return (...)        # ← 遇到第一条 assistant/user 消息就 return，不再继续遍历
```

**问题**：函数语义应是"历史中是否已存在 stop ack"，实际实现是"最后一条 assistant/user 消息是否是 stop ack"。取消发生时若最后一条是用户普通消息，会误判为 False，`_run_agent_validation_job` 的取消处理器（约 1605 行）会再插一条 stop ack，对话中出现重复消息。

**修复**：

```python
def _agent_has_stop_ack_message(repo: TaskRepository, task_id: str) -> bool:
    for message in repo.list_agent_messages(task_id):
        metadata = message.get("metadata") or {}
        if (
            message.get("role") == "assistant"
            and metadata.get("intent") == "stop"
            and metadata.get("cancel_requested") is True
        ):
            return True
    return False
```

#### P1-3 `delete_task`：DB 删除成功后文件删除失败会留下孤儿目录（Windows 上更易触发）

- **位置**：`api.py:563-580`

**问题**：顺序是 先删 DB 记录 → 关 kernel session → `shutil.rmtree(task_dir)`。Windows 上若 kernel 进程仍持有文件句柄，`rmtree` 抛 `PermissionError`，此时 DB 记录已删、目录永久残留且无法通过产品清理。另外"检查 active job → 删除"之间存在 TOCTOU 窗口。

**修复（最小改动）**：调整顺序为 先 `close_live_notebook_session` → 再删 DB → 最后 `rmtree` 并容错：

```python
close_live_notebook_session(task_id)
repo.delete_task(task_id)
try:
    if task_dir.exists():
        shutil.rmtree(task_dir)
except OSError as exc:
    logger.warning("task dir cleanup failed for %s: %s", task_id, exc)
```

更彻底的方案：DB 软删除标志 + 启动时清扫孤儿目录（可挂在 `recovery.py` 的启动恢复流程里）。

#### P1-4 `update_report_fields` 中参数 `payload` 被局部变量遮蔽

- **位置**：`api.py:1081-1116`（约 1110 行处 `payload = _validation_results_payload_for_task(...)`）

**问题**：函数参数 `payload: ReportFieldsUpdateRequest` 在函数后半段被重新赋值为 `dict | None`。当前无行为 bug，但后续维护者在该行之后引用 `payload` 会拿到错误类型且无任何报错。

**修复**：改名为 `results_payload`。

#### P1-5 `_resolve_scan_material` 路径校验不走统一的 `assert_within`

- **位置**：`api.py:3146-3158`

**问题**：用 `resolved.relative_to(source_dir)` 校验路径包含关系，而项目已有 `safe_paths.assert_within()` 统一入口。两套校验逻辑并存，未来一处加固另一处不会跟进。

**修复**：统一改为 `assert_within(source_dir, resolved)`，捕获 `PermissionError` 返回友好错误。

#### P1-6 `_capture_user_preference_memory` 静默吞掉所有异常

- **位置**：`api.py:2524-2547`

**问题**：用户明确说"请记住……"之后，磁盘满/DB 锁/schema 错误等任何异常都被 `except Exception: return` 吞掉，记忆悄悄保存失败，用户零反馈，也无日志可查。

**修复**：至少记录 warning 日志；更好的做法是在 assistant 回复的 metadata 中加 `memory_save_failed: true`，前端展示一条软提醒。

### 任务编排与 Notebook 执行（pipeline.py / notebooks.py / recovery.py）

#### P1-7 `run_notebook_stage`：session 注册与状态更新的顺序导致失败时状态失控

- **位置**：`pipeline.py:143-157`

**问题**：先 `register_live_notebook_session(task_id, live_session)` 再 `repo.update_status(...)`。若 `update_status` 抛异常，外层 except 会 `live_session.close()`，但 registry 里仍残留指向已关闭 session 的条目，且任务状态停在 `RUNNING` 等待崩溃恢复——kernel 生命周期与任务状态在这个窗口内完全脱钩。

**修复**：调换顺序——先 `update_status` 成功后再注册 session；这样失败路径只需 close 一个未注册的 session，无残留。

#### P1-8 `NotebookExecutionSession.close()` 依赖 nbclient 私有 API，假设不成立时 kernel 进程泄漏

- **位置**：`notebooks.py:174-183`

**问题**：通过 `getattr(self.client, "_cleanup_kernel", None)` 反射调用 nbclient 内部方法。该私有方法在 nbclient 升级后可能消失/改名，届时 `callable(None)` 为 False，close 静默什么都不做，kernel 进程永久泄漏（每个任务一个僵尸 Python 进程）。

**修复**：增加公共 API 回退链：

```python
cleanup = getattr(self.client, "_cleanup_kernel", None)
if callable(cleanup):
    cleanup()
elif getattr(self.client, "km", None) is not None:
    try:
        self.client.km.shutdown_kernel(now=True)
    except Exception:
        pass
```

并在 `pyproject.toml` 给 `nbclient` 加版本上下限（见 P2-14），让私有 API 假设至少有版本约束兜底。

#### P1-9 `split_col`/`time_col` 为 NULL 的历史任务在 metrics 注入 cell 中产生 `KeyError: None`

- **位置**：`pipeline.py:743-744`（payload 构建）与 `pipeline.py:827-831`（注入代码列检查）

**问题**：`contract.split_col or task.split_col` 在两者皆为 None（历史 DB 行可能为 NULL）时把 `None` 传进注入代码，列存在性检查中 `None not in df.columns` 为 True，报错文案显示 `split_col='None'`；若绕过检查则 `_rmc_sample[None]` 抛 `KeyError: None`，完全不可调试。

**修复**：payload 构建时兜底为空字符串并在注入代码里 `if column and column not in ...` 跳过空值；或在任务加载入口对 NULL 值统一回填默认列名。

#### P1-10 `recovery.py` 绕过 `connect()` 封装直接裸连 SQLite

- **位置**：`recovery.py:47`

**问题**：`sqlite3.connect(db_path)` 不带 `busy_timeout`/`foreign_keys`/`row_factory`，与 `db.py` 的 `connect()` 封装行为不一致。FastAPI 启动恢复期与其他连接并发时，锁冲突会立即抛 `database is locked` 而不是等待 5 秒。同时启动恢复对 `agent_memory_entries` 的半写入状态完全无感知。

**修复**：改用 `from marvis.db import connect`，删除手动 `conn.commit()`。`init_db`（`db.py:30-31`）同样建议统一（见 P2-9）。

### 验证算法层（validation/）

#### P1-11 两套 PSI 实现并存且 smoothing 常数不同（1e-6 vs 1e-7）

- **位置**：`validation/binning.py:51`（`compute_psi`，smoothing=1e-6）与 `validation/effectiveness.py:424`（`_psi_component`，smoothing=1e-7）（已人工核验）

**问题**：月度 PSI/总体 PSI 走 `compute_psi`，稳定性分箱表走 `_psi_component`。两者数学等价（`(a-b)ln(a/b) = (b-a)ln(b/a)`），但**当某个分箱为空时** smoothing 不同会导致同一份报告里 PSI 数值不一致，审计时难以解释。

**修复**：删除 `_psi_component`，`compute_psi_stability_table` 改为调用向量化的 `compute_psi`，统一 smoothing 常数并在 docstring 写明。

#### P1-12 `_should_reverse_eval_bins`：分数零方差时相关系数为 NaN，错误触发分箱反转

- **位置**：`validation/effectiveness.py:261-274`

**问题**：所有样本同分时 `np.corrcoef` 返回 NaN，`not bool(nan > 0)` 求值为 True，分箱顺序被错误反转。

**修复**：

```python
if not np.isfinite(correlation):
    return False
return not bool(correlation > 0)
```

#### P1-13 `reproducibility` 一致性状态机三态退化为二态，`REVIEW` 永远不会出现

- **位置**：`validation/reproducibility.py:103`

**问题**：`status = PASS if mismatch_count == 0 else FAIL`。`ConsistencyStatus.REVIEW`（部分不匹配但在容忍范围内）永远不会被产出，与领域语义和枚举设计不符。

**修复**：在 config 中增加 review 容忍阈值（如 `max_abs_diff` 上限或 mismatch 比例上限），mismatch>0 但在容忍内时输出 `REVIEW`。这是业务口径决策，建议先和模型验证负责人确认阈值再实现。

#### P1-14 月份存在缺口时"环比 PSI"实为隔月 PSI，无任何提示

- **位置**：`validation/effectiveness.py:326-341`

**问题**：样本月份不连续（如缺 202502）时，`psi_mom` 计算的是 202503 vs 202501，但报表标签仍是"环比"。

**修复**：检测相邻月份间隔，>1 个月时在该行 payload 加 `gap: true` 标记，报告/前端展示为"较上一有样本月份"或加脚注。

#### P1-15 `_bad_count` 的 falsy 陷阱：真实坏样本数为 0 时走估算分支

- **位置**：`output/excel.py:474-475`（已人工核验）

```python
def _bad_count(row) -> int:
    return int(row.bad_count or round(row.sample_count * row.bad_rate))
```

**问题**：`bad_count=0` 是合法值（某 split 无坏样本），`or` 把它当 False 落入 `round(sample_count * bad_rate)` 估算。若 `bad_rate` 因序列化精度为微小非零值，报表会显示非零坏样本数，与平台计算值矛盾——违反"确定性指标由平台代码计算"的产品承诺。

**修复**：`bad_count` 字段类型是 `int` 非 Optional，直接 `return int(row.bad_count)`；若历史 payload 确有缺失场景，用 `row.bad_count if row.bad_count is not None else round(...)`。

### Agent / LLM 层

#### P1-16 流式读取阶段的网络异常逃逸 `LLMClientError` 包装，后台 job 整体崩溃

- **位置**：`llm_client.py:57-95`

**问题**：异常捕获只覆盖 `urlopen` 建连阶段（`HTTPError`/`URLError`/`TimeoutError`）。流式循环 `for raw_line in response` 中连接中断抛出的 `http.client.RemoteDisconnected`/`IncompleteRead`/`OSError` 不是 `URLError` 子类，会直接穿透到 `_run_agent_validation_job`，绕过所有 fallback 降级路径，后台 job 崩溃。

**修复**：

```python
except (TimeoutError, OSError, http.client.HTTPException) as exc:
    raise LLMClientError(f"LLM stream interrupted: {exc}") from exc
```

注意 try 块要包住整个 `with urlopen(...)` 体（含流式读取），不只是 urlopen 调用。

#### P1-17 `generate_word_conclusions` 强制 `stream=True` + `response_format=json_object` 组合脆弱

- **位置**：`agent/service.py:653-657`；`llm_client.py`（`stream` 硬编码 True）

**问题**：依赖完整 JSON 的场景走流式拼接，流中断/截断时 `json.loads` 失败被归并为泛化的"LLM 离线"提示，用户无法区分"连不上"和"返回格式坏了"。

**修复**：`complete()` 增加 `stream: bool = True` 参数，JSON 场景传 `stream=False`；`json.JSONDecodeError` 单独捕获并给出"LLM 返回不是有效 JSON"的明确错误。

#### P1-18 Agent 可整体覆盖平台计算的压力测试摘要

- **位置**：`report_texts.py:34-38`（`AGENT_CONFIRMED_REPORT_TEXT_KEYS` 含 `TEXT:pressure_test_summary`）

**问题**：`pressure_test_summary` 由平台从真实压测结果生成，属于确定性证据，但被列入 Agent 可确认覆盖的 key 集合。按 AGENTS.md"Agent 不能编造指标或绕过平台证据"，该字段不应允许全量替换。

**修复**：将该 key 移出 `AGENT_CONFIRMED_REPORT_TEXT_KEYS`；若产品上需要 Agent 润色，改为"平台摘要 + Agent 补充说明"两段式，平台段不可改。

#### P1-19 记忆 summary 无长度上限，可无限膨胀 prompt

- **位置**：`agent_memory/prompting.py:83-95`（`_memory_packet`）；`agent_memory/extractors.py`（`extract_user_preference` 不截断）

**问题**：用户通过"请记住：…"可写入任意长度偏好文本，多条命中检索后累加进 prompt，最终触发上下文长度报错；且 `policy.py` 的长文本拦截只看带特定关键词的文本（见 P1-20）。

**修复**：`_memory_packet` 对 summary 截断（建议 400 字符）；`extract_user_preference` 入库前截断（建议 200 字符）。

#### P1-20 记忆策略的"报告全文"拦截只检查 summary，payload 可绕过

- **位置**：`agent_memory/policy.py:84-89`

**问题**：`_looks_like_long_report_text` 只检查 `candidate.summary`，把报告全文放进 `payload` 某字段即可绕过 AGENTS.md"禁止保存未脱敏报告全文"的规则。

**修复**：改为对 `_candidate_text(candidate)`（summary + payload JSON）做长度与标记检查；同时增加与关键词无关的通用长度上限（超限直接拒绝）。

#### P1-21 `record_use` 可对 `rejected` 状态的记忆条目写入"已使用"审计事件

- **位置**：`agent_memory/store.py:218`（`_select_entry` 的 WHERE 只排除 `deleted`）

**问题**：被策略拒绝存储的条目（如含敏感内容）仍可通过 `record_use` 留下 `use` 审计记录，审计语义被污染。`set_status` 对 rejected 有终态保护，`record_use` 没有。

**修复**：`record_use` 中显式拒绝 `status == "rejected"` 的条目（抛 `ValueError`），与 `set_status` 的终态保护对齐。

### 前端（static/app.js）

#### P1-22 前端用后端自由文本推断业务状态，违反 AGENTS.md 模块边界约定

- **位置**：`app.js:215-222`（`taskStopped`）、`app.js:1231-1241`（`taskFailureStage` fallback）、`app.js:4731/4777`（`agentMessageIsContinuePrompt`/`agentMessageIsScanLead` 的 content 匹配分支）

**问题**：AGENTS.md 明确规定 static/ "不要解析后端自由文本作为业务事实"。`taskStopped` 完全靠 `status_message` 包含"已取消/cancelled"判断停止态；只要某条失败消息碰巧含这些词，工作流步骤状态、操作按钮可用性全部错乱。后端其实已支持结构化字段（`failure_stage`、`metadata.awaiting_next_stage`、`metadata.tool_call.name`），文本匹配是并列路径而非过渡兜底。

**修复**（需前后端配合）：
1. 后端在任务 payload 中补充结构化 `stopped: bool` 字段（取消路径写入，而不是靠 status_message 文案）；与 api.py 侧的 `_stage_returned_cancelled_task`（`api.py:308-315`，同样在做字符串匹配）一起替换为结构化标志。
2. 前端删除文本匹配分支；`failure_stage` 缺失时返回 null 显示"未知阶段"，而不是猜。
3. 保留的兼容兜底加注释 + TODO 标明删除条件。

---

## P2 — 改进建议 / 健壮性加固

### 算法与数据

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| P2-1 | `validation/binning.py:7-10` | `equal_frequency_bin_edges` 不过滤 NaN/Inf（当前靠上游 `finite_score_series` 隐性保证） | 函数内自卫：`arr = arr[np.isfinite(arr)]`，空数组时返回 `[-inf, inf]` |
| P2-2 | `validation/binning.py:32-48` | `compute_ks` O(unique×n) 复杂度，10 万+样本明显变慢 | 对齐 `_roc_ks_curve` 的排序+cumsum 线性实现 |
| P2-3 | `validation/effectiveness.py:152` | 分箱排序表按各 split 独立等频分箱，与 PSI 使用的训练集边界（`context.edges`）不一致，跨 split 不可直接横比 | 这可能是有意的排序性分析口径；建议明确决策——若要跨表可比则统一传 `context.edges`，若保持现状则在文档/表头注明"各样本独立等频分箱" |
| P2-4 | `pipeline.py:880-884` | `_RmcNotebookScorer` 用 `list(index) ==` 判断缓存命中，类型不严格 | 改用 `dataframe.index.equals(_rmc_sample.index)` |
| P2-5 | `pipeline.py:537` | `json.dumps(str(package_root))` 与 head cell 的 `as_posix()` 不一致，Windows 上 sys.path 出现同一目录的两种写法（无功能性失败，但去重检查失效） | 统一用 `Path(package_root).as_posix()!r` |
| P2-6 | `pipeline.py:74,988` | `validation_results.pkl` 写入后全库无人读取，pickle 跨版本脆弱 | 确认无外部消费方后删除写入逻辑 |
| P2-7 | `pipeline.py:1308-1344` | 子进程转 pickle 再 `pd.read_pickle` 加载，反序列化攻击面 | 中间格式改 parquet/feather 原生加载或 JSON |
| P2-8 | `metric_tables.py:310-317` | `by_month.setdefault(...).update(row)` 键冲突时静默覆盖 | 合并前断言键集不相交，或显式字段映射 |

### 持久化与并发

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| P2-9 | `db.py:30-31,183-185` | `init_db` 裸连接 + 隐式 autocommit，schema 迁移无整体事务，中途失败留下半迁移状态 | 统一走 `connect()` 封装或显式事务包裹全部 DDL |
| P2-10 | `db.py:303-334` | `update_status` 先 SELECT 校验再 UPDATE，DEFERRED 事务下存在并发窗口（最终有 `WHERE status IN (...)` CAS 兜底，实际危害有限） | 简化为单条条件 UPDATE + rowcount 判断，或改 IMMEDIATE 事务 |
| P2-11 | `db.py:788-791` | `_ensure_column` f-string 拼表名/列名进 SQL（当前调用点全是常量） | 加表名白名单 + 列名正则断言，防未来误用 |
| P2-12 | `api.py:2607-2621` | 记忆检索后逐条 `get_entry(audit=True)`，每条一次独立连接 | 单连接批量写 retrieve 审计事件 |
| P2-13 | `agent_memory/store.py:398` | 审计事件 FK `ON DELETE SET NULL` 在纯软删模型下永不触发，形同虚设 | 改 CASCADE 或移除 FK 并在未来物理清理任务中显式处理 |

### 工程与配置

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| P2-14 | `pyproject.toml:8-20` | 全部依赖无版本下限；pydantic v1/v2、pandas 1/2 均有破坏性差异；nbclient 私有 API 假设（P1-8）无版本约束兜底 | 至少加 `pydantic>=2`、`pandas>=2`、`nbclient>=0.7,<0.11`、`nbformat>=5` |
| P2-15 | `__main__.py:156-164` | CLI `validate` 不读取 UI 配置的 execution environment，kernel 固定为默认 `python3` | 构建 `PipelineSettings` 时传入 `load_execution_environment(settings.workspace).kernel_name` |
| P2-16 | `scripts/release_push.py:162-183` | `--dry-run` 跳过 `ensure_clean_worktree`/`ensure_on_branch`，干跑结论与真实执行前置条件不一致 | dry-run 也执行只读检查并输出确认信息 |
| P2-17 | `files.py:70-74` | `_is_hidden_or_checkpoint_path` 用 `parts[:-1]` 漏掉顶层隐藏文件（`.hidden.csv` 会被扫描分类为样本） | 去掉 `[:-1]`，检查全部 parts |
| P2-18 | `execution_environment.py:327-331` | Windows 下 conda env 检测含永不命中的 `bin/python` 候选 | 按平台分支生成候选列表 |

### 前端

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| P2-19 | `app.js:3348-3349` | `trend-spark` 单元格按原始 HTML 注入。经核验，sparkline 实际由前端 `renderSparklineSvg` 自己生成（3549 行），当前无 XSS；但 `column_specs` 来自后端 payload，后端若声明 `kind: "trend-spark"` 列，行内字符串会被原样注入 DOM | 不要让"原样 HTML"这个渲染能力由后端可控的 `kind` 字段开启：前端插入 sparkline 列时用内部标记（如 `spec.__localHtml === true`），`renderCellByKind` 只信任该内部标记 |
| P2-20 | `app.js:5459` | Markdown 链接渲染：协议白名单（`https?://`、`/`、`#`）已挡住 `javascript:`/`data:`，整体安全；但在已转义文本上做正则替换的模式较绕，href 中实体处理依赖浏览器还原 | 重构为先解析后构造 DOM（`document.createElement('a')` + `href` 赋值），消除模式脆弱性 |
| P2-21 | `app.js:2226-2236` | 背景轮询与主动操作轮询并行，同任务双倍请求频率 | 轮询管理器统一去重：同 taskId 只允许一个 progress poll 存活 |
| P2-22 | `app.js:3723-3725` | `metricTooltipAttached` 一次性全局标志，`metricPreview` 节点若被重建则 tooltip 永久失效 | 事件委托挂到稳定祖先节点（document 或 workspace 容器） |
| P2-23 | `app.js` 整体 | 6300 行单文件、全局可变状态散布，变更影响面难以评估 | 按现有功能区拆分 ES Module（state / api / render-metrics / render-agent / polling / dialogs），无需引入框架 |
| P2-24 | `app.js:6331-6338` | 初始化失败（后端未启动）无任何用户可见提示，只有空任务列表 | 初始化 catch 中在状态区展示"服务连接失败，请检查后端是否运行" |

### 文档与设计一致性

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| P2-25 | `AGENTS.md:15-16`、`docs/roadmap.md:7`、`docs/versioning.md:7` | 版本边界过时：V1.1 仍标"计划中"，实际已发布 V1.1.1 且 agent_memory 已落地。AGENTS.md 是 AI 协作的事实来源，过时描述会让 Claude/Codex 对功能状态作出错误判断 | 三处统一改为"当前稳定线 V1.1.x（含 Agent Memory Foundation）" |
| P2-26 | `docs/对notebook的要求.md` 第 9 节（约 338-365 行） | "推荐完整契约 cell"缺少必填变量 `RMC_SAMPLE_DF`，建模人员照抄会直接静态检查失败 | 在该代码块补 `RMC_SAMPLE_DF = modeling_sample`，与 `docs/notebook_contract.md` 对齐 |
| P2-27 | `api.py:91-92, 2762-2817` | 过拟合检测算法（含业务阈值 0.10/0.05）实现在 HTTP 路由层，违反 AGENTS.md 模块边界（"api.py 不承载验证算法"），且 `validation/` 中无对应实现 | 迁移到 `validation/overfitting.py`，api.py 只调用 |
| P2-28 | `docs/runbook.md:128-131` | CLI 示例只给 `python -m marvis validate`，未展示首选入口 `marvis validate` | 首选 `marvis validate`，模块路径作为兼容备注 |
| P2-29 | `output/word_preview.py:326` | 内联 CSS 用 `backdrop-filter: blur(14px)`，违反 DESIGN.md 性能约束 | 改用不透明背景色 |

### 输出渲染（Word/Excel/图表）

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| P2-30 | `template_reports.py:375-378` | `_clear_paragraph` 只移除 `<w:r>`，段落内书签/域代码/超链接残留，可能与新 run 混排导致 Word 结构异常 | 遍历 `paragraph._p` 全部子元素，保留 `w:pPr`，其余移除 |
| P2-31 | `template_reports.py:72-83` | 占位符替换只遍历顶层表格，嵌套表格内占位符原样残留；合并单元格会被重复遍历 | 递归遍历 + 以 `id(cell._tc)` 去重 |
| P2-32 | `output/image_render.py:201-227, 410-477` | `fig.savefig` 抛异常时 `plt.close(fig)` 不执行，批量渲染时 figure 泄漏 | `try/finally: plt.close(fig)` |
| P2-33 | `output/image_render.py:627-643` | `findfont` 某些版本返回非 CJK 默认字体路径时被误认为命中，中文变方块 | 校验返回路径包含候选字体名关键词 |
| P2-34 | `output/excel.py:163-178` | KS/AUC 单元格存 ×100 后的浮点（32.2），列头无单位说明；下游若用 `data_only=True` 读值直接差 100 倍 | 列头改"KS(%)"，或存 0-1 小数 + `0.00%` 格式 |
| P2-35 | `report_texts.py:125-139` | `merge_report_text_values` 用 `*manual_value_sets` 位置参数区分权限语义（index 0 可覆盖 Agent 确认 key，index 1 不可），极易误改 | 改具名参数 `report_values=`, `manual_values=` 并写明权限规则 |

---

## 待确认事项（低置信度，建议人工判断）

1. **`agent_next_stage` 在 `REVIEW_REQUIRED` 且结论已确认时返回 None**（`agent/orchestrator.py:36-41`）：若用户手动确认流程（`_confirm_agent_report_conclusions`）直接调用 `run_report_stage`，则无问题；否则存在"已确认但永远不生成 Word"的死局。建议跑一次 auto_accept=False 的全流程验证。
2. **Agent rerun "从头" 不清理 outputs/ 旧产物**（`agent/service.py:276-298` + `api.py:1659-1675`）：需确认 `reset_status_for_agent_rerun` 是否清理 `task_dir/outputs/`，否则重跑后 LLM 可能引用过期验证结果。
3. **`compute_bin_tables` 按 split 独立分箱**（P2-3）：是 bug 还是有意的口径，需要模型验证负责人定夺。
4. **`reproducibility REVIEW` 阈值**（P1-13）：修复需要业务定义"可接受偏差"口径。

---

## 建议修复顺序

**第一批（立即，半天内）**——纯代码修复、无业务决策依赖：
- P0-1（population_at_ks）、P0-2（tz_localize）、P1-1（重复定义×3 处）、P1-2（stop ack）、P1-4（payload 遮蔽）、P1-12（NaN 反转）、P1-15（_bad_count）、P1-16（流式异常捕获）

**第二批（本周）**——需要小范围设计决策：
- P0-3（LLM 扩展字段开关，需加配置项与 UI）、P1-3（delete_task 顺序）、P1-7（session 注册顺序）、P1-8（kernel 关闭回退）、P1-9（split_col NULL）、P1-10（recovery 连接封装）、P1-19/20/21（记忆层三项）、P2-25/26（文档修正）

**第三批（下个迭代）**——跨前后端 / 需要业务口径：
- P1-22（结构化 stopped/failure_stage 字段，前后端联动）、P1-11（PSI 实现统一）、P1-13（REVIEW 阈值）、P1-14（月份缺口标记）、P1-18（压测摘要权限）、P2-27（过拟合逻辑迁移到 validation/）、P2-23（前端模块化）

每一项修复都应附带回归测试；本项目测试基建完善（tests/ 共 3.5 万行），上述多数问题恰恰说明缺的是针对边界条件（NULL 列、tz-aware 时间、零方差分数、bad_count=0、非 DeepSeek provider）的用例，建议修复时一并补齐。

---

*报告由 8 个模块级深度审查通道产出、关键发现经人工核验后汇总。行号基于 2026-06-13 工作树（分支 `codex/windows-create-task-422`，含未提交修改），后续提交可能造成偏移。*
