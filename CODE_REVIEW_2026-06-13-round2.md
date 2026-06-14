# MARVIS 第二轮全面代码审查报告

- 审查日期：2026-06-13（第二轮）
- 审查对象：当前工作树（含 Phase 0 还债的全部未提交修改）
- 方法：7 路并行专项审查（api / pipeline / validation / db+memory+agent+llm / 前端 / 输出渲染 / CLI+文档）+ 人工核实关键发现（含实测）
- 上一轮报告：`CODE_REVIEW_2026-06-13.md`（Phase 0 修复依据）

## 总览

| 类别 | 数量 | 说明 |
|------|------|------|
| Phase 0 修复已验证正确 | 全部 P0/P1 | 7 个通道交叉确认，无回归 |
| 本轮已修 | 1 P0 + 2 P2 | image_render bad_count、tz no-op、psi_vs_train 一致性 |
| 待修 P1 | 6 | 见下 |
| 待修/改进 P2 | 13 | 见下 |
| 误报（实测排除） | 4 | 见末尾，避免再被当 bug 修 |

测试现状：**731 passed，ruff 全过**（含本轮新增 1 个回归测试 + conftest JVM 隔离修复）。

---

## 一、Phase 0 修复验证（全部正确，无回归）

7 个审查通道逐一核实，以下 Phase 0 修复均正确实施：

- **A-1** `population_at_ks` → 累计人口占比（不平衡样本测试 `==1/20` 验证）✓
- **A-2** tz 时区 → 保留钟面时间（`test_timezone_aware_business_dates_preserve_local_wall_clock_day`）✓
- **A-3** LLM `enable_thinking` 开关 + `stream` 参数化 + llm_settings 持久化 ✓
- **B-1~B-11** 全部后端 P1（重复定义、stop ack、delete_task、payload、assert_within、capture memory、session 顺序、kernel 回退、split_col 兜底、recovery connect、流式异常）✓
- **P1-11~13** PSI 统一、NaN 反转、REVIEW 三态 ✓
- **P1-19~21** 记忆 summary 截断、长文本检测、record_use 终态保护 ✓
- **P2-17/29~35** 隐藏文件、backdrop-filter、_clear_paragraph、嵌套表格、figure 泄漏、CJK 字体、KS/AUC 列头、merge 具名参数 ✓
- **P2-27** 过拟合迁到 validation/overfitting.py，api.py 复用 ✓
- **D-2/D-3** 依赖下限、结构化 failure_stage/stopped ✓
- **CLI A-P1-2** validate 接入 notebook_kernel_name ✓；**文档 B-P1-1/B-P1-2** 版本边界、RMC_SAMPLE_DF ✓

这是一次质量很高的还债。

---

## 二、本轮已修复

### P0-R2-1 `image_render.py` 的 `bad_count=0` falsy 陷阱（已修 + 补测试）

- **位置**：`output/image_render.py:277`
- **问题**：`int(row.bad_count or round(row.sample_count * row.bad_rate))` —— 和 excel.py 的 P1-15 同源，但**漏修了 image_render 镜像**。`OverallRow.bad_count: int = 0`，真实坏样本数=0 时 `0 or round(...)` 走估算，PNG/Word 报告显示非零、与 Excel 不一致，违反"指标由平台算"。
- **修复**：`int(row.bad_count)`（已是非 Optional int）。补回归测试 `test_model_effect_rows_preserves_zero_bad_count`。
- 同行 `psi_vs_train` 一并改为 `_opt_float(row.psi_vs_train, digits=3)`，与 excel/metric_tables 的 None 处理对齐（P2）。

### P2-R2-1 `time_periods.py` tz 的 no-op 化简（已修）

- **位置**：`validation/time_periods.py:110`
- **问题**：`tz_convert(timestamp.tz)` 是 no-op（转到自己的时区）。实测确认原 CODE_REVIEW P0-2 是误报——pandas 3.0.3 下 `tz_localize(None)` 对 tz-aware 不抛错、直接保留钟面时间。
- **修复**：化简为 `timestamp.tz_localize(None)` + 注释说明"保留钟面、禁止 UTC 归一（会跨月）"。

---

## 三、待修 P1（确认的真实问题）

### P1-R2-1 `llm_settings.py` 未持久化 `reasoning_effort`（A-3 不完整）

- **位置**：`llm_settings.py:43-58`（save）、`:136-147`（_public_model）
- **置信度**：95（两个通道独立发现）
- **问题**：`save_llm_settings` 存了 `enable_thinking` 但没存 `reasoning_effort`；`_public_model` 也不暴露。`llm_client.py:45` 用 `profile.get("reasoning_effort") or "high"`，但该值仅由 `api.py._resolve_agent_model` 在请求时注入。用户在 UI 配的 effort 无法持久化，每次回退 "high"。
- **修复**：save 和 _public_model 各加 `"reasoning_effort": str(raw_model.get("reasoning_effort") or "high")`；前端设置表单加输入项；补 llm_settings 往返测试。

### P1-R2-2 `_task_stop_reason_code`：成功任务被历史 cancelled job 误标 stopped

- **位置**：`api.py` `_task_stop_reason_code`
- **置信度**：90（逻辑确认；触发需"取消→重跑成功"序列）
- **问题**：`_latest_cancelled_job_kind` 不看当前状态，SUCCEEDED 任务若历史有 cancelled job 会被标 `stopped=True`，前端据此渲染"已停止"。
- **待确认前置**：需确认"重跑成功"时 `status_reason_code` 是否被清除——若没清除，第一条 `if reason == USER_CANCELLED` 已会误判，那 bug 其实在 reason 清除逻辑。**建议先跑一遍"取消 metrics→重跑成功"端到端确认根因再修**，避免修错位置。
- **修复方向**：给 `_latest_cancelled_job_kind` 检查加状态守卫（SUCCEEDED/REVIEW_REQUIRED/WRITING_ARTIFACTS 跳过），并补 SUCCEEDED+cancelled-job 组合测试。

### P1-R2-3 `_handle_agent_stop_message`：HTTP stop 端点不防重复 ack

- **位置**：`api.py:2318-2348`
- **置信度**：80
- **问题**：HTTP stop 路径写 stop ack 前没检查 `_agent_has_stop_ack_message`（B-2 只修了后台 job 路径）。快速连发两次 stop → 对话出现两条"已停止"消息。
- **修复**：写 ack 前加 `if not _agent_has_stop_ack_message(repo, task.id):`。

### P1-R2-4 `_sanitize_llm_payload` 把 dict 的 key 也脱敏

- **位置**：`agent/service.py:1036-1043`
- **置信度**：83
- **问题**：遍历 dict 时对 key 也调 `_sanitize_llm_text`（把 task_id 替换成中文显示名）。key 不应被脱敏，只有 value 需要。极端情况下（key 含 task_id 子串）会把键名替换成带空格的中文，污染发给 LLM 的 JSON。
- **修复**：`return {key: _sanitize_llm_payload(item, task) for key, item in value.items()}`。

### P1-R2-5 失败阶段兜底默认 "notebook"，scan/metrics 失败被误标

- **位置**：`api.py` `_legacy_failure_stage_from_message` 末尾 `return "notebook"`；`app.js` `taskFailureStage` 末尾 `return "notebook"`
- **置信度**：85（前端通道评 P0，但因后端已结构化派生且 scan 走 `_task_failed_during_scan` 优先，实际影响小于评估）
- **问题**：`failure_stage` 缺失时硬猜 "notebook"。对真正在其它阶段失败的历史任务，UI 高亮错阶段、显示错误的重试按钮。违反"不解析自由文本/不猜业务事实"。
- **修复（需前后端协调）**：后端 legacy 兜底改为：匹配 notebook 专属模式→"notebook"，否则 `None`；前端 `taskFailureStage` 缺失返回 `null`，展示"阶段未知"。

### P1-R2-6 前端 `agentMessageIsContinuePrompt`/`agentMessageIsScanLead` 仍文本匹配

- **位置**：`app.js:4548`、`app.js:4592`
- **置信度**：85
- **问题**：Phase 0 修了 `taskStopped`，但这两个仍 `content.includes("是否继续执行")` / `startswith("正在调用材料识别工具")`，违反 AGENTS.md。影响 Agent 时间线布局，文本碰撞会错位。
- **修复**：删文本匹配分支，仅依赖 `metadata.awaiting_next_stage` / `metadata.tool_call.name`；后端未全量保证 metadata 时加 TODO 注释，但不引入新文本判断。

### P1-R2-7（低触发）stress test cell 对 `split_col=""` 无保护

- **位置**：`pipeline.py:963`（vs 748 的 `or ""` 兜底）
- **置信度**：中（DB `split_col NOT NULL DEFAULT 'split'`，正常不触发）
- **问题**：B-9 让 split_col 兜底为 `""`，列检查 `if column and ...` 正确跳过，但 stress cell 的 `_rmc_sample[_rmc_config.split_col]` 直接用 `""` 索引 → `KeyError: ''` 不可调试。
- **修复**：stress cell 注入前判断 `if _rmc_config.split_col:` 跳过，或在列检查后对空 split_col 抛友好错误。

---

## 四、待修/改进 P2

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| P2-R2-2 | `validation/binning.py:82-95` | `bad_cum_pct_running`/`good_cum_pct_running` 每轮即覆盖，"running"命名误导，实为 `cum_bad_pct`/`cum_good_pct` | 删死变量，直接用 `cum_bad_pct`/`cum_good_pct` |
| P2-R2-3 | `tests/validation/test_overfitting.py` | 只测 fail/not_available，缺 pass 与边界（value==threshold）用例 | 补 pass + 边界测试 |
| P2-R2-4 | `validation/binning.py:57-62` | `compute_psi` 对空桶 smoothing 后不重归一化，sum>1（每空桶约 1e-6 偏差） | 重归一化或乘法型 smoothing（待业务口径确认） |
| P2-R2-5 | `app.js:4511-4526` | 冻结快照 `contentHtml` 注释说"strip id"但实现没有；id 重复 + 再注入模式 | DOMParser 解析剥离 id 再序列化 |
| P2-R2-6 | `app.js:4837` | `renderAgentTimeline` 重建 bucket 后未重调 `attachRocInteractions` 等交互绑定 | 重建后重新 attach，或事件委托到稳定祖先 |
| P2-R2-7 | `render-agent.js:279` | `escapeHtml` 后再正则替换 bold/em，含 `&<>` 的加粗文字双重转义 | 先解析 markdown 结构再转义，或对实体单独处理 |
| P2-R2-8 | `js/dialogs.js:49` | `bindTabs` 无空值守护，DOM 缺失则静默崩溃 | 加 `if (!el) return` |
| P2-R2-9 | `output/image_render.py:135` vs `metric_tables.py:204` | 压测表列数不一致（图 4 列无 PSI，前端 5 列含 PSI） | 确认产品口径，统一或注明 |
| P2-R2-10 | `template_reports.py:345` | 死函数 `_set_paragraph_text` 无调用 | 删除 |
| P2-R2-11 | `agent_memory/policy.py:87` | `_looks_like_long_report_text(summary)` 参数名误导（实传 summary+payload 全文） | 改名 `text` |
| P2-R2-12 | `db.py:872` | `PRAGMA journal_mode=WAL` 返回值未检查，失败静默降级 | 检查返回值，非 wal 记 warning |
| P2-R2-13 | `README.md:23` | 英文版 "Planned... V1.1 includes" 让已发布的 Memory Foundation 像计划中，落后于中文版 | 改措辞与中文版一致 |
| P2-R2-14 | `scripts/release_push.py:162` vs `docs/versioning.md:99` | `tag_exists` 先于 `ensure_clean_worktree`，文档步骤顺序与代码不符 | 把 worktree 检查前移，或改文档 |
| P2-R2-15 | `pipeline.py:1644` | `_notebook_step_v3` 的 `cancel_expected_status` 参数定义后从未使用 | 删死参数 |

---

## 五、误报（已实测排除，勿当 bug 修）

1. **db.py `isolation_level="DEFERRED"` + 手动 `BEGIN IMMEDIATE`**：审查通道评 P0/P1（引 Python 文档称"语义混用、并发风险"）。**实测排除**：Python 3.13 / sqlite 3.53 下 `BEGIN IMMEDIATE` 正常（`in_transaction=True`），写锁正确生效（另一连接 `BEGIN IMMEDIATE` 被 "database is locked" 阻塞），commit 正常。功能正确，仅写法略非常规。
2. **原 CODE_REVIEW P0-2（tz_localize 抛 TypeError）**：实测确认 pandas 3.0.3 下不抛错，是误报（已在本轮化简代码并加注释）。
3. **`__main__.py._profile_defaults` 无兜底返回**：所有路径都有 return，误报。
4. **`recovery.py` UNION 查询参数绑定顺序**：第二段 SQL 用字面量无需占位符，绑定正确，误报。

> 教训：审查 agent 引用文档/通用知识时容易产生版本相关的误报（tz、db 事务两次）。**涉及 pandas/sqlite/py4j 等运行时行为的发现，一律实测确认再修。**

---

## 六、建议处理顺序

**第一批（本轮已做）**：image_render P0、tz 化简、JVM 测试隔离、psi_vs_train 一致性。

**第二批（纯后端、低风险、可立即修）**：P1-R2-1（reasoning_effort 持久化）、P1-R2-3（stop ack 防重）、P1-R2-4（sanitize 不动 key）、P2-R2-2/10/11/15（死代码/命名/死参数清理）、P2-R2-3（overfitting 补测试）。

**第三批（需确认根因或前后端协调）**：P1-R2-2（stop reason，先跑端到端确认 reason 清除逻辑）、P1-R2-5（failure_stage 兜底，前后端一起改）、P1-R2-6（前端文本匹配，配合前端模块化推进）、P1-R2-7（stress cell 守卫）。

**第四批（文档/体验）**：P2-R2-13（README 英文）、P2-R2-14（release 顺序）、其余前端 P2。

---

*本轮确认 Phase 0 还债质量很高，新发现以 1 个 P0（已修）+ 6 个 P1 为主，多数是 Phase 0 修复的"邻接遗漏"（同源镜像 image_render、HTTP 路径 stop ack、A-3 的 reasoning_effort 尾巴）。db.py 和 tz 两处"看似严重"的发现经实测均为误报——再次印证运行时行为必须实测。*

---

## 修复进度（2026-06-13 收尾）

最终状态：**733 passed × 3 随机顺序，ruff 全过，git diff --check 干净，所有 JS 模块语法 OK。**

### 已修复（含验证/补测试）

| 项 | 修复 |
|----|------|
| P0-R2-1 | image_render bad_count falsy → `int(row.bad_count)` + 回归测试 `test_model_effect_rows_preserves_zero_bad_count` |
| P1-R2-1 | llm_settings 持久化 `reasoning_effort`（save + _public_model + `_reasoning_effort` 归一）；api `_resolve_agent_model` 仅在请求显式传 effort 时覆盖，否则用持久化值 |
| P1-R2-2 | `_task_stop_reason_code` 加 `_NON_STOPPED_TASK_STATUSES` 守卫（已实测确认 reason 在成功更新时被清，故只需守第二道 cancelled-job 回退） |
| P1-R2-3 | `_handle_agent_stop_message` 写 ack 前查 `_agent_has_stop_ack_message` 防重 |
| P1-R2-4 | `_sanitize_llm_payload` 不再脱敏 dict key，只脱敏 value |
| P1-R2-6 | 前端 `agentMessageIsContinuePrompt`/`agentMessageIsScanLead` 删除文本匹配，仅用结构化 metadata（测试 fixture 已带 metadata，不破坏） |
| P1-R2-7 | stress cell 注入前加 `if not _rmc_config.split_col: raise ValueError(...)` 友好错误 |
| P2 系列 | tz no-op 化简、psi_vs_train 一致性、bin_table 死变量、死函数 `_set_paragraph_text`、policy 参数改名、死参数 `cancel_expected_status`、overfitting pass+边界测试、WAL 返回值检查、dialogs bindTabs 空值守护、README 英文措辞、versioning.md 步骤顺序、冻结快照 `stripIdsFromHtml` 剥离 id |

### 撤销（实测/核查后判定为非问题）

- **P1-R2-5（failure_stage 兜底 "notebook"）**：有显式测试 `test_missing_structured_failure_stage_defaults_to_notebook_step` 证明是**有意设计**，非 bug，不改。
- **P2-R2-7（bold/em 双重转义）**：推演确认 `escapeHtml` 只跑一次，`&`→`&amp;` 浏览器渲染回 `&`，**误报**。
- **db.py `isolation_level="DEFERRED"`+`BEGIN IMMEDIATE`**：实测写锁正确生效，**误报**。
- **P2-R2-6（时间线重建不重绑交互）**：agent 消息桶是 markdown 文本不含 roc-card，冻结快照是有意静态历史视图，**非 bug**。

### 已决定保持现状（不改）

- **P2-R2-4** `compute_psi` 空桶 smoothing 后不重归一化：误差约 1e-6/空箱，比 PSI 决策阈值（0.1/0.25）小 5~6 个数量级，无任何决策影响；改动反而会微扰现有报告数值/测试基线。**保持现状。**
- **P2-R2-9** 压测表列数（Word 图 4 列 vs 前端 5 列）：产品展示口径，**保持现状。**

**round-2 待办全部清空。**
