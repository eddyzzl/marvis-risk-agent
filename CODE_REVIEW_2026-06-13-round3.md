# MARVIS 第三轮代码 review（修复后当前工作树）

- 日期：2026-06-13（第三轮）
- 范围：上一轮大修复（CODEBASE_DESIGN_AUDIT 的 P1-01~P3-03）落地后的当前代码
- 方法：5 路并行专项审查 + 人工实测核实关键发现
- 客观信号：**750 passed，ruff 全过**

## 一句话结论

上一轮修复（package-data、stopped、远程只读、扫描上限、压测 status、api_key_env、memory 白名单、task_type、模块拆分…）**经核实基本都对**。但这轮修复**引入/暴露了几个新问题**，其中 2 个安全、3 个正确性值得优先处理。另有 1 个被反复误报的"P1"已第三次实测证伪。

---

## 必修：安全（2 个真问题）

### S1【P1】反向代理绕过本机访问控制 —— 远程客户端获得完整权限
- 位置：`app.py:26-29,49-67`（`_is_local_client` 用 `request.client.host`）
- **实测分析确认**：P1-03 的 local-only 守卫靠 `request.client.host` 判断本机。**MARVIS 部署在同机反向代理（nginx/Caddy/JupyterHub）后，所有远程请求的 client.host 变成代理的 loopback IP 127.0.0.1 → is_local=True → 远程客户端通过代理获得完整访问（含 POST/DELETE 等写操作）**，整个守卫失效。`__main__.py` 明确提示 JupyterHub `/proxy/` 用法，代理部署是已知路径，不是假设。
- 修复：用 uvicorn `--proxy-headers --forwarded-allow-ips=127.0.0.1`（让 `request.client` 解析为真实客户端），或加 `MARVIS_TRUSTED_PROXY_HOSTS` 显式信任名单 + 读 X-Forwarded-For 首跳。不要在无配置时信任转发头。

### S2【P1】Markdown 链接放行协议相对 URL `//evil.com` —— 钓鱼
- 位置：`static/js/render-agent.js:299-301`（`isSafeMarkdownHref`）
- **实测确认**：`//evil.com/steal` 被 `href.startsWith("/")` 判为安全 → 生成 `<a href="//evil.com">`，浏览器按协议相对解析成 `https://evil.com`。Agent 消息里 `[点这](//evil.com)` 即可生成外链钓鱼。
- 修复：`if (href.startsWith("//")) return false;` 放在 `startsWith("/")` 之前。

---

## 必修：正确性（3 个真问题）

### C1【P1】KS 标记竖线画在 ROC 的 FPR 轴上，位置错误（不平衡数据）
- 位置：`output/image_render.py:208` `ax.axvline(x=curve.population_at_ks)`；`static/app.js:3381` `xOf(curve.population_at_ks)`
- **实测确认**：ROC 图横轴是 `fpr`，但 KS 标记竖线画在 `x=population_at_ks`。信贷坏率约 7-9%，`population_at_ks ≠ fpr_at_ks`，竖线画错位置（report 图表 + 前端 ROC 卡都有）。文字"at pop=X"本身对，但画在 FPR 轴上不对。
- 修复：渲染层用 `fpr[argmax(|ks_curve|)]`（= fpr_at_ks）作为竖线 x；或在 `RocKsCurve` 加 `fpr_at_ks` 字段。`population_at_ks` 只用于文字 tooltip。

### C2【P1】scan 上限触发的 ValueError 未捕获 → HTTP 500 + 失败阶段误标 notebook
- 位置：`files.py:50-56`（超 max_files/max_depth 抛 ValueError）；`api.py` scan handler 只 catch FileNotFoundError/NotADirectoryError → 穿透成 500；`pipeline.py:1189` `_scan_artifacts` 同样不 catch ValueError → 被 notebook 阶段 `except Exception` 捕获，打 `NOTEBOOK_STAGE_FAILURE_PREFIX` 前缀 → `task_failed_during_scan` 失效，前端把扫描失败标成 notebook 失败。
- 这是 P2-03 修复引入的回归路径。
- 修复：scan handler 和 `_scan_artifacts` 都 catch `ValueError`，前者转 422，后者转 `PipelineError` 用"材料扫描失败："前缀。

### C3【P1】记忆 `delete()` 绕过 rejected 终态保护 —— 审计链可被覆写
- 位置：`agent_memory/store.py:304-335`（`delete()` 用 `_select_entry(include_deleted=False)`，WHERE 只排 `deleted` 不排 `rejected`）
- `set_status(id,"deleted")` 直接进 `delete()`，rejected 记录（含敏感候选的拒绝审计）被 `UPDATE SET status='deleted'` 覆写。`record_use` 已有 rejected 终态检查，`delete` 漏了。
- 修复：`delete()` 开头加 `if current["status"]=="rejected": raise ValueError("rejected entries are terminal")`。

---

## 建议修：P2

| # | 位置 | 问题 |
|---|------|------|
| P2-1 | `app.py:36-37,88-93` + `api_settings.py` | `MARVIS_ALLOW_REMOTE_READ=1` 时 `/branding/assets/*` 和 `/api/settings/*`（含本机 python 路径/conda/模型列表）被远程读。远程只读应只放数据 API，不放私有 branding + 系统配置 |
| P2-2 | `app.py:22,29` | `_is_local_client` 不认 IPv6-mapped loopback `::ffff:127.0.0.1` → 本机 IPv6 客户端被误判为远程（errs safe，但 UX 问题）。建议用 `ipaddress` 判 `is_loopback` |
| P2-3 | `static/app.js:4477`（`stripIdsFromHtml`） | 冻结快照只剥 id，不剥 `onerror`/`onclick` 等事件属性和 `<script>`。当前数据源已转义无活 XSS，但是潜在后门。建议 stripIds 一并移除事件属性 + script 标签 |
| P2-4 | `agent_memory/prompting.py:99-111`（`_bounded_payload`） | 只放行 model_experience 字段，导致 user_preference/field_convention/validation_pitfall/task_experience 的 payload 注入 LLM 时全变 `{}`，非模型类记忆丢失结构化上下文。应按 memory_type 用 `PAYLOAD_FIELD_ALLOWLISTS` 过滤 |
| P2-5 | `api_task_payloads.py:84-92,134-138` | `legacy_stop_reason_code_from_message` 用 `"cancelled"/"已停止"` 模糊子串匹配且对 SUCCEEDED/REVIEW_REQUIRED 终态无守卫 → message 含这些词的成功任务被标 stopped。建议精确匹配历史固定文案 + 终态跳过 |
| P2-6 | `api_task_payloads.py`（`failure_stage_from_job_kind`） | `pipeline`/`agent` job kind 不在映射 → 回退 legacy 文本默认 "notebook"。建议补映射或写结构化 failure_stage |
| P2-7 | `static/js/branding.js:43-49` | `logoUrl`/`faviconUrl` 来自后端直接赋 `img.src`/`link.href`，无协议白名单（data:/javascript: 隐患）。建议加 isSafeAssetUrl |
| P2-8 | `metric_tables.py` + `app.js` | 压测整体 status（partial/failed）Web 端无显著横幅/徽章，只靠展开看状态列。P2-05 的 Web 显示这点未完整落地 |
| P2-9 | `static/app.js:4954-4960` | `agentValidatorAlias` 把真实人名（于添/张雯萱）硬编码进公开静态 JS。建议移到 branding/workspace 配置 |
| P2-10 | `llm_client.py:66-67` → `service.py` | HTTP 错误响应 body 前 500 字节拼进 `LLMClientError` 并写入 `agent_messages.metadata`，可能间接持久化被拒 prompt 片段。建议只存状态码 |
| P2-11 | `db.py:791`（`_normalize_task_type`） | 不拒绝未知 task_type，任意字符串原样入库。建议白名单 |
| P2-12 | `db.py:554` | `get_latest_cancelled_job_kind` 已是死代码（P1-02 修复后无人调用），FakeTaskRepository 还实现它造成"已覆盖"错觉。建议删 |
| P2-13 | `output/excel.py` + `image_render.py` `_compact_number` | 未处理 ±inf（与 effectiveness.py 同名函数不一致）。当前路径不触发，建议统一 |
| P2-14 | `recovery.py:103-108` | UNION 后半段（活跃 agent job）无 cutoff 过滤，`stale_after_seconds>0` 时活跃任务会被提前插重启通知。默认 0 无害，逻辑仍错 |

---

## 已实测证伪（勿当 bug 修）

- **db.py `isolation_level="DEFERRED"` + `BEGIN IMMEDIATE`**：pipeline agent 报 P1（引 Python 文档称 3.12+ 抛 ProgrammingError）。**第三次实测证伪**：Python 3.13 下 `BEGIN IMMEDIATE` 正常（in_transaction=True、不抛错），750 测试全过（含 update_status 路径）。这是反复出现的文档型误报，已记入记忆。
- **renderMarkdownEmphasisText 双重转义**：agent 自我修正为非 bug（escapeHtml 只跑一次）。

---

## 建议处理顺序

1. **S1 反向代理绕过**（安全最高，部署即裸奔）→ uvicorn proxy-headers 或信任名单。
2. **S2 协议相对链接**（一行修）。
3. **C2 scan ValueError**（P2-03 引入的回归，500 + 误标阶段）。
4. **C1 ROC 标记线**、**C3 记忆 delete 终态**。
5. P2-1/P2-4/P2-5（远程暴露面、记忆字段、stopped 误报）。
6. 其余 P2 清理。

---

*教训重申：涉及 pandas/sqlite/py4j 等运行时行为的审查发现，一律实测再修——这轮又靠实测挡掉一个会白改的 db.py "P1"。*
