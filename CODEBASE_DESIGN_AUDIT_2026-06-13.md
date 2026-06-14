# MARVIS Risk Agent 代码与设计全面审计

审计日期：2026-06-13
审计分支：`codex/windows-create-task-422`
审计范围：后端 API、任务生命周期、Notebook/pipeline、验证算法、Agent 与 Agent Memory、静态前端、打包发布、测试结构、设计文档一致性。

> 说明：当前工作区存在大量未提交改动和未跟踪文件，本报告只记录审计结论，不修改业务代码。已有的 `CODE_REVIEW_2026-06-13.md` / `CODE_REVIEW_2026-06-13-round2.md` 未被覆盖。

## 已执行的自动化检查

```bash
conda run -n py_313 python -m pytest -q
# 750 passed in 26.08s

conda run -n py_313 python -m ruff check riskmodel_checker tests --extend-exclude '*.ipynb'
# All checks passed!

node --check riskmodel_checker/static/app.js
node --check riskmodel_checker/static/js/*.js

git diff --check
```

补充打包验证：

```bash
conda run -n py_313 python -m pip wheel . --no-deps -w /tmp/rmc-wheel-test
# Successfully built riskmodel-checker
```

已用 zipfile 确认 wheel 包含：

- `riskmodel_checker/static/js/branding.js`
- `riskmodel_checker/static/css/welcome.css`
- `riskmodel_checker/static/js/api.js`
- `riskmodel_checker/static/app.js`
- `riskmodel_checker/static/styles.css`

补充浏览器 smoke：

- 临时启动 `http://127.0.0.1:8765/`。
- Playwright 打开首页，确认欢迎页可见。
- 点击“模型验证”任务卡，确认创建验证任务弹窗可打开。
- 浏览器控制台错误/警告为 0。

结论：常规单测、ruff、JS 语法、wheel 构建、diff 空白检查和浏览器 smoke 当前均通过。下面的问题主要是自动化覆盖不到的安装包、运行边界、任务生命周期和架构设计风险。

## 本轮修复状态

已在本轮落地：

- P1-01：`static/js/*` 已加入 package-data，并新增 package-data 测试；wheel 构建后已确认模块存在。
- P1-02：`stopped` 不再由历史 cancelled job 推断，只由当前结构化 reason code 或旧版停止文案兼容判断。
- P1-03：远程客户端默认只能读 `/`、`/static/*`、`/api/health`；数据 API 默认 local-only，可用 `MARVIS_ALLOW_REMOTE_READ=1` 显式打开远程只读。
- P2-03：`scan_source_dir()` 已增加文件数、目录深度、累计 hash 大小上限。
- P2-04：平台验证结果不再生成新的 `validation_results.pkl`，保留旧文件清理逻辑。
- P2-05：压力测试新增类别级/整体 `status`，Excel/Web/报告文本会显式展示 skipped/error/partial 状态。
- P2-06：LLM 设置支持 `api_key_env`，可不把真实 key 写入 workspace 配置文件。
- P2-07：Agent markdown 不再保留不安全链接 URL；不安全链接只展示文字，HTML 继续 escape。
- P2-09：Agent Memory policy 增加按 memory type 的 payload 字段白名单。
- P2-10：任务 payload/DB 增加 `task_type`，当前默认 `validation`，为后续模型开发/策略开发分流预留。
- P2-01：已补第一轮低风险模块化拆分：settings 路由拆到 `api_settings.py`，请求 schema 拆到 `api_schemas.py`，任务 payload/状态派生拆到 `api_task_payloads.py`；前端 branding 拆到 `static/js/branding.js`，欢迎页样式拆到 `static/css/welcome.css`，shell 静态测试拆到 `tests/test_frontend_shell_static.py`。
- P2-08：已新增 `tests/test_frontend_smoke.py`，通过 FastAPI TestClient 加载 `/`、递归校验本地 ES module import、校验 CSS 链接资源，并覆盖欢迎页任务卡静态入口；package-data 也补充 `static/css/*`。
- P3-01：`index.html` 的 app.js cache-bust 改为 `__MARVIS_STATIC_VERSION__` 占位符，由 FastAPI 渲染时替换为包版本。
- P3-03：`docs/versioning.md` 已补充本机 conda 环境与公开命令的边界说明。

保留的后续重构边界：

- P2-01 中 Agent workflow 主体、任务 stage router、报告 router 仍不建议一次性大搬迁；本轮已把低耦合 settings/schema/payload/static shell 层先拆出，后续可继续按 stage 或 view 逐步迁移。

## 总体结论

- 未发现会在当前测试集内立即打红的 P0 级问题。
- 发现 3 个建议优先修复的 P1：前端 ES module 打包遗漏、取消后重跑可能被错误标记为 stopped、远程只读接口默认暴露本地敏感结果。
- 发现多项 P2 设计债务：`api.py` / `app.js` / `styles.css` / 静态测试文件过大，状态判断仍有自由文本兼容路径，材料目录默认允许扫描整个 home，pickle 中间件继续存在，压力测试的局部异常缺少整体状态，LLM key 明文存储需要边界说明或更安全后端。
- 设计方向与 `docs/roadmap.md`、`DESIGN.md` 基本一致：V1.1 保持模型验证稳定，Agent Memory 只能辅助解释和报告口径，未来模型开发/策略开发应走 V2 Plugin/Tool/Workflow 底座。

## P1-01 前端 ES module 文件未进入 package-data，安装包运行会 404

**证据**

- `riskmodel_checker/static/index.html` 使用 `<script type="module" src="static/app.js?...">`。
- `riskmodel_checker/static/app.js` 第 1 行开始导入 `./js/api.js`、`./js/dialogs.js`、`./js/polling.js`、`./js/render-agent.js`、`./js/render-metrics.js` 等模块。
- 当前存在 `riskmodel_checker/static/js/*.js`。
- `pyproject.toml` 的 package-data 只有：

```toml
riskmodel_checker = ["static/*", "static/brand/*", "static/pets/*", "static/pets/naitang/*", "static/pets/xiaojiu/*"]
```

没有包含 `static/js/*`。

**影响**

源码目录直接运行时没问题，因为文件实际在本地目录里；但构建 wheel/sdist 后，`static/js/*.js` 很可能不会被打进包里。用户通过 `pip install` 安装后打开页面，浏览器请求 `/static/js/api.js` 等模块会 404，主界面无法启动。

**修复方案**

1. 修改 `pyproject.toml`：

```toml
[tool.setuptools.package-data]
riskmodel_checker = [
  "static/*",
  "static/js/*",
  "static/brand/*",
  "static/pets/*",
  "static/pets/naitang/*",
  "static/pets/xiaojiu/*",
]
```

2. 更稳妥的做法是后续把静态资源收敛成可递归包含的结构，避免每新增一层目录都漏 package-data。
3. 增加安装包 smoke test：
   - `python -m build`
   - 在临时 venv 安装 wheel。
   - 断言 `importlib.resources.files("riskmodel_checker").joinpath("static/js/api.js").is_file()`。
   - 启动 FastAPI TestClient，GET `/`、`/static/app.js`、`/static/js/api.js` 都应返回 200。

**建议测试**

- 新增 `tests/test_package_data.py`，直接检查 `riskmodel_checker/static/js/api.js` 在包资源中可见。
- CI 增加最小 wheel 安装验证，避免源码路径掩盖打包遗漏。

## P1-02 取消过的任务重跑后可能继续显示为 stopped

**证据**

`riskmodel_checker/api.py` 当前用 `_task_stop_reason_code()` 生成 payload 里的 `stopped` / `stop_reason_code`：

```python
if task.status not in _NON_STOPPED_TASK_STATUSES:
    if _latest_cancelled_job_kind(repo, task.id):
        return TASK_STATUS_REASON_USER_CANCELLED
```

`_NON_STOPPED_TASK_STATUSES` 只包含：

```python
SUCCEEDED, REVIEW_REQUIRED, WRITING_ARTIFACTS
```

但任务取消后重新执行时，任务可能处在 `RUNNING`、`EXECUTED`、`COMPUTING_METRICS` 等状态。这些状态已经代表当前任务在继续推进，不应被历史 cancelled job 标记为 stopped。`TaskRepository.reset_status_for_agent_rerun()` 会清空 `status_reason_code`，但旧的 cancelled job 记录仍存在。

**影响**

用户可能看到：

- 任务正在运行，但左侧/中心状态显示“已停止”。
- Agent stop 状态和主动执行状态冲突。
- 前端 action bar 判断错误，导致按钮状态、轮询和提示不一致。

**修复方案**

优先改成“当前任务状态字段是唯一权威”，历史 cancelled job 只用于 job 审计，不参与当前 stopped 判断：

1. `_task_stop_reason_code()` 只在以下情况返回 `user_cancelled`：
   - `task.status_reason_code == TASK_STATUS_REASON_USER_CANCELLED`
   - 或 legacy 迁移期间，当前 status 是可停留的中间态/复核态，并且 status_message 明确是旧版停止文案。
2. 不再用 `_latest_cancelled_job_kind()` 推断当前 stopped；如果必须兼容旧数据，也应只对没有任何后续 job、且 `updated_at <= cancelled_job.finished_at` 的记录生效。
3. 将 `_NON_STOPPED_TASK_STATUSES` 扩展为更明确的策略函数，例如：

```python
def _task_can_be_currently_stopped(task: TaskRecord) -> bool:
    return task.status in {
        TaskStatus.SCANNED,
        TaskStatus.EXECUTED,
        TaskStatus.REVIEW_REQUIRED,
    } and not repo.task_has_active_job(task.id)
```

但更推荐完全依赖 `status_reason_code`。

**建议测试**

- 取消 agent/notebook job 后，调用 rerun 将任务重置到 `RUNNING`，断言 `/api/tasks/{id}` 中 `stopped == false`。
- 取消后重跑到 `EXECUTED`、`COMPUTING_METRICS`，断言仍为 false。
- 老数据只有 `status_message="已停止当前动作"` 且没有 reason code 时，仍能被兼容识别。

## P1-03 非本机客户端仍可读取所有 GET 接口

**证据**

`riskmodel_checker/app.py` 的中间件只阻止非本机客户端的非安全方法：

```python
if request.method.upper() not in _SAFE_METHODS and not _is_local_client(...):
    return JSONResponse(status_code=403, ...)
```

GET/HEAD/OPTIONS 不受限制。当前 GET 接口包括：

- `/api/tasks`
- `/api/tasks/{task_id}`
- `/api/tasks/{task_id}/evidence`
- `/api/tasks/{task_id}/agent/messages`
- `/api/agent-memory`
- `/api/agent-memory/{memory_id}`
- `/api/tasks/{task_id}/report/download`
- `/api/tasks/{task_id}/report/preview`
- `/api/settings/llm`

CLI 默认 host 是 `127.0.0.1`，但用户可以显式传 `--host 0.0.0.0` 或通过代理暴露服务。

**影响**

MARVIS 是本地优先工具，任务证据、Agent 消息、记忆、报告预览、分析下载都可能包含模型验证材料、报告段落、字段口径和历史经验。当前保护只防写，不防读；一旦服务暴露在局域网，远端用户可直接读取敏感结果。

**修复方案**

1. 默认所有 API 都 local-only，而不是只限制 unsafe methods。
2. 如果确实需要远程读取，增加显式开关和审计，例如：
   - `MARVIS_ALLOW_REMOTE_READ=1`
   - 或 `--allow-remote-read`
   - 或 token header：`Authorization: Bearer ...`
3. `/api/health` 可以例外公开；`/` 和 `/static/*` 是否公开取决于产品决策，但数据 API 应默认受限。
4. 对 `/branding/assets/*` 也建议按同一策略处理，避免私有 branding 资产被远端枚举。

**建议测试**

- 构造 `request.client.host = "192.168.1.20"` 的 TestClient/middleware 测试。
- 默认 GET `/api/tasks`、`/api/agent-memory`、报告下载均返回 403。
- 开启显式 allow 或 token 后返回 200。

## P2-01 `api.py`、`app.js`、`styles.css` 和前端静态测试过大，回归风险高

**证据**

```text
riskmodel_checker/api.py                 3246 lines
riskmodel_checker/pipeline.py            1781 lines
riskmodel_checker/static/app.js          5907 lines
riskmodel_checker/static/styles.css      4955 lines
tests/test_frontend_static_v2.py         6241 lines
```

`api.py` 同时承担任务 API、Agent orchestration、memory context、报告字段、stage rerun 等职责；settings/schema/payload 已在本轮拆出。`app.js` 虽然已经拆出部分 `static/js/*`，但主文件仍承担任务列表、材料上传、指标渲染、Agent 对话、报告区、轮询、冻结快照等大量职责；branding 已拆出，欢迎样式已从主 CSS 拆出。

**本轮已修**

- `riskmodel_checker/api_settings.py` 承载 execution environment 与 LLM settings 路由，`api.py` 通过主 router include，URL 不变。
- `riskmodel_checker/api_schemas.py` 承载 FastAPI request schema，避免 `api.py` 继续膨胀。
- `riskmodel_checker/api_task_payloads.py` 承载 task payload、失败阶段、停止状态、报告下载文件名等 UI payload 逻辑，并由 `api.py` 保持兼容导入。
- `riskmodel_checker/static/js/branding.js` 承载 branding normalize/apply helper。
- `riskmodel_checker/static/css/welcome.css` 承载欢迎页任务卡/问候样式，`index.html` 用版本占位符链接。
- `tests/test_frontend_shell_static.py` 从大静态测试里拆出 shell/branding/welcome 契约测试。

**影响**

- 修改一个局部 UI 容易引发远处状态回归。
- 字符串断言型测试很多，重构成本高，且无法模拟真实浏览器打包/加载问题。
- 模块边界与 `AGENTS.md` 中的职责边界开始偏离：API 层承载了过多 workflow 和 Agent 逻辑。

**修复方案**

后续继续分阶段拆，不建议一次性大重构：

1. 后端先拆 router：
   - `api/tasks.py`
   - `api/stages.py`
   - `api/agent.py`
   - `api/memory.py`
   - `api/settings.py`
2. 将 Agent workflow 编排从 `api.py` 移到 `riskmodel_checker/agent/workflows.py`，API 只做 HTTP 入参、权限和返回 payload。
3. 前端继续拆 ES modules：
   - `static/js/views/welcome.js`
   - `static/js/views/task-list.js`
   - `static/js/views/task-dialog.js`
   - `static/js/views/agent-chat.js`
   - `static/js/views/report.js`
   - `static/js/render/metrics.js`
4. CSS 拆成 token/base/layout/components/views，但构建仍可保持无前端依赖，使用多个 `<link>` 或一个简单拼接脚本。
5. 测试从“大文件字符串索引”逐步迁移到：
   - 纯函数模块用 Node import 测。
   - 关键流程用浏览器/Playwright smoke 测。
   - 字符串断言只保留 API/DOM id 的稳定契约。

**建议测试**

- 每拆一个模块，先用现有测试锁行为，再迁移对应测试到模块级。
- 增加“页面加载后无 module 404、欢迎页可见、创建任务入口可达”的浏览器 smoke。

## P2-02 后端业务判断仍混用 `status_message` 自由文本

**证据**

已经引入了 `status_reason_code`、`failure_stage`、job kind 等结构化字段，但仍有这些兼容路径：

- `_task_failed_during_scan(task)` 通过 `status_message.startswith("材料扫描失败：")` 判断。
- `_legacy_failure_stage_from_message()` 从中文失败前缀推断阶段。
- `_legacy_stop_reason_code_from_message()` 从“cancelled / 已停止 / 已取消”推断停止原因。
- `_agent_rerun_stage_reached()` 用 `task.status_message.startswith(NOTEBOOK_STAGE_FAILURE_PREFIX)` / `REPORT_STAGE_FAILURE_PREFIX` 判断阶段可重跑。

**影响**

- 新状态字段与旧文字不一致时，容易出现 UI 状态错误。
- 文案改动会影响业务逻辑。
- Agent 生成或人工写入的 status message 如果包含关键字，可能造成误判。

**修复方案**

1. 增加结构化字段，例如：
   - `failure_stage`
   - `failure_reason_code`
   - `last_completed_stage`
   - `stop_reason_code`
2. pipeline/API 更新状态时同步写结构化字段。
3. legacy parser 收敛到单独模块 `legacy_status_parser.py`，只用于迁移旧数据，并标注删除窗口。
4. 前端完全消费结构化字段；不要再需要 `status_message` 判断状态，只展示文字。

**建议测试**

- 构造 `status_message` 含有“报告输出失败”但 `failure_stage="metrics"`，断言后端和前端都按 structured stage 处理。
- 构造结构化字段为空的旧数据，断言 legacy 兼容仍生效。

## P2-03 材料目录默认允许整个 home，且扫描缺少总量/深度上限

**证据**

`riskmodel_checker/api.py`：

```python
def _allowed_material_roots(settings) -> tuple[Path, ...]:
    roots = [settings.workspace, Path.home()]
```

`riskmodel_checker/files.py`：

```python
for path in sorted(source_dir.rglob("*")):
    ...
    digest = sha256_file(path) if size_bytes <= hash_limit_bytes else None
```

会递归扫描所选目录下所有文件，单文件小于 50MB 时计算 sha256。

**影响**

- 用户误选 home 或上层目录时，可能扫到大量 CSV/Notebook/PMML，性能变差。
- 本地工具虽然由用户操作，但误扫私人目录会增加隐私和体验风险。
- 上传/路径两种入口合并后，这个风险更容易被普通用户触发。

**修复方案**

1. 默认允许 `settings.workspace` 和一个明确的 `workspace/materials`，不要默认允许整个 home。
2. 如果保留 home，需要前端二次确认：“将递归扫描该目录，可能包含大量文件”。
3. `scan_source_dir()` 增加：
   - 最大文件数，例如 2000。
   - 最大目录深度，例如 6。
   - 最大累计扫描大小。
   - 超限时返回结构化错误和已扫描摘要。
4. 把 `hash_limit_bytes` 拆成单文件 hash 上限和累计 hash 上限。

**建议测试**

- 构造超过最大文件数的目录，断言返回可理解错误。
- 构造深层目录，断言超过深度不继续扫描或明确报错。
- API 测试覆盖不在允许根目录下的路径仍返回 422。

## P2-04 验证结果仍写 `validation_results.pkl`

**证据**

`riskmodel_checker/pipeline.py`：

```python
VALIDATION_RESULTS_PICKLE = "validation_results.pkl"
...
import pickle as _rmc_pickle
...
with _rmc_results_pickle_path.open('wb') as _rmc_handle:
    _rmc_pickle.dump(_rmc_results, _rmc_handle)
```

同时也已经写 `validation_results.json`。

**影响**

- pickle 不是安全、稳定、跨版本友好的交换格式。
- 当前看主要是平台自己写，不一定存在立即安全漏洞；但未来 Plugin/Tool/Workflow 接入后，容易被误用为可读取的工作流输入。
- 用户之前确认建模脚本可能会读 `.pkl`，但那是 Notebook 内部模型/WOE 字典的业务材料；平台验证结果不一定需要 pickle。

**修复方案**

1. 若没有消费者依赖，删除 `validation_results.pkl` 输出，只保留 JSON 和 Excel。
2. 若仍需 Python 对象恢复，改为 dataclass JSON loader 或 parquet/feather 表格文件。
3. 如果短期不能删：
   - 文件名改为内部语义，如 `_internal_validation_results.pkl`。
   - 文档声明平台不会读取用户提供的 pickle。
   - 增加测试禁止对 workspace 中非平台生成 pkl 做 `pickle.load`。

**建议测试**

- pipeline 运行后断言 JSON 是主契约。
- 如果删除 pickle，更新依赖 `validation_results.pkl` 的测试。
- 对未来插件输入 schema 明确禁止 pickle 作为通用数据交换格式。

## P2-05 压力测试按类别吞异常，但缺少整体告警状态

**证据**

`riskmodel_checker/validation/stress_test.py` 在每个特征类别里捕获所有异常：

```python
except Exception as exc:
    per_category.append(StressCategoryResult(..., error=f"{type(exc).__name__}: {exc}"))
```

返回的 `StressTestResult` 只有 `baseline` 和 `per_category`。

**影响**

这对“一个类别失败不影响其他类别展示”是友好的，但没有整体 `status` 时，报告/前端很容易把压力测试显示成已完成。用户如果没展开明细，可能错过某些类别完全失败。

**修复方案**

1. 给 `StressTestResult` 增加派生状态：
   - `pass`：所有类别有结果。
   - `partial`：部分类别有 `error`。
   - `failed`：所有类别都失败或 baseline 失败。
2. Excel/Word/Web 都展示“压力测试部分失败”的显著提示。
3. 对 `no features from this category` 区分为 `skipped`，不要和 scorer 异常混在同一种 error。

**建议测试**

- 一个类别 scorer 抛异常，断言整体状态为 `partial`，报告中有告警。
- 所有类别都没有模型特征，断言状态为 `skipped` 或 `partial`，不显示成正常完成。

## P2-06 LLM API key 明文写入 workspace，需要产品边界和替代存储

**证据**

`riskmodel_checker/llm_settings.py`：

```python
path = workspace / "settings" / "llm.json"
...
"api_key": api_key
path.write_text(json.dumps(private_settings, ensure_ascii=False, indent=2))
```

`.gitignore` 默认忽略 `workspace/` 和 `workspace-*/`，常规默认路径不会提交。但用户可通过 `--workspace` 指到任意目录。

**影响**

- 本地工具可以接受明文配置，但必须对用户透明。
- 如果 workspace 被放在同步盘或 git 管理目录里，key 可能被同步或提交。
- GET `/api/settings/llm` 虽然只返回 `has_api_key`，但 P1-03 的远程只读问题仍会暴露模型配置元数据。

**修复方案**

1. 设置界面明确提示：API key 存在本地 workspace，默认不上传。
2. 支持环境变量引用，例如 `api_key_env="OPENAI_API_KEY"`，落盘不保存真实 key。
3. macOS 可选接入 Keychain；跨平台可后续抽象 `SecretStore`。
4. 保存前检查 workspace 是否在 git repo 内；如果是，提示确认或自动写入 `.git/info/exclude`。

**建议测试**

- 保存设置后 public API 不返回 `api_key`。
- 使用 env 引用时，`resolve_llm_model()` 能读取环境变量，配置文件不含真实 key。
- 指向 git 管理目录时返回 warning metadata。

## P2-07 前端 `innerHTML` 面积大，需要强制安全护栏

**证据**

`riskmodel_checker/static/app.js` 有大量 `innerHTML`、`outerHTML`、`insertAdjacentHTML`。大部分路径已经使用 `escapeHtml()` 或 `renderAgentMarkdown()`，例如任务列表、记忆列表、指标表格、Agent 消息、LLM 设置列表。

需要特别关注的路径：

- Agent markdown 渲染允许 http(s)、`/`、`#` 链接。
- frozen snapshot 通过保存现有 DOM 的 `contentNode.innerHTML` 再插回页面。
- 后续模块拆分后，新代码很容易绕过 `escapeHtml()`。

**影响**

当前没有确认到一个明确的未转义 XSS 点，但这是高风险区域。Agent 内容、任务字段、模型名、报告字段、记忆摘要都是用户或模型可控文本；一旦新增渲染路径漏掉 escape，就会变成持久注入。

**修复方案**

1. 建立统一 API：
   - `html(strings, ...values)` 默认 escape。
   - `unsafeHtml()` 只能在少数经过审计的内部模板使用。
2. 禁止直接写 `element.innerHTML = userText`；通过 eslint 不现实的话，用静态测试检查新增 `innerHTML` 调用是否在白名单函数内。
3. `renderAgentMarkdown()` 增加更严格的链接协议和属性测试，尤其检查 `javascript:`、`data:`、实体绕过、括号截断。
4. frozen snapshot 限定只能来自平台生成的 metric/scan/reproducibility 容器；不要允许从 Agent message 或 report editor 复制 HTML。

**建议测试**

- 用恶意模型名、任务名、memory summary、Agent markdown 输入 `<img onerror=...>`，断言页面只显示文本。
- Agent markdown 链接输入 `[x](javascript:alert(1))`、`[x](data:text/html,...)` 不生成 anchor。
- frozen snapshot 中如果出现 script/event handler，应被剥离或不进入 snapshot。

## P2-08 静态字符串测试很多，但缺少真实打包/浏览器 smoke

**证据**

`tests/test_frontend_static_v2.py` 超过 6000 行，包含大量 `app_js.index(...)`、精确字符串片段断言。已有 Node 级测试覆盖 `renderAgentMarkdown` 等函数。本轮已新增服务级静态 smoke，覆盖安装包/服务路径下的 CSS 与 ES module 资源加载；如后续引入 Playwright，再补真实浏览器点击 smoke。

**本轮已修**

- 新增 `tests/test_frontend_smoke.py`：打开 `/`，确认欢迎页任务卡文案，提取 module script，递归校验本地 `import` 的 ES modules 都能由 `/static/...` 返回 200，并校验 CSS 链接资源。
- `tests/test_package_data.py` 同时覆盖 `static/js/*` 与 `static/css/*` package-data 和声明文件存在性。
- `tests/test_frontend_shell_static.py` 从大文件拆出 shell/branding/welcome 契约，降低后续 UI 改动定位成本。

**影响**

- 字符串测试能防止某些回归，但对模块化重构阻力很大。
- 它无法发现 P1-01 这种“源码存在、安装包缺文件”的问题。
- 它也无法发现 CSS 重叠、移动端布局、module import 404 等真实浏览器问题。

**修复方案**

1. 保留少量静态契约测试，例如 DOM id、关键 API 调用、禁止解析 status_message。
2. 把纯函数移入 ES module 后用 Node import 测。
3. 增加 Playwright 或轻量浏览器 smoke：
   - 打开 `/`。
   - 等待欢迎页/任务卡出现。
   - 点击“模型验证”打开创建任务框。
   - 验证控制台没有 module 404。
4. 增加 wheel 安装后 smoke，专门覆盖 package-data。

**建议测试**

- `tests/test_static_assets_package.py`
- `tests/test_frontend_smoke.py` 或独立 `npm`/Playwright 脚本。

## P2-09 Agent Memory 策略方向正确，但安全过滤仍是启发式

**证据**

`riskmodel_checker/agent_memory/policy.py` 已经拒绝：

- 客户号/身份证/手机号等模式。
- 原始样本行。
- Notebook 源码片段。
- PMML/模型内容。
- API key/token。
- DB connection string。
- 长报告文本。

这是合理的第一层防线，但仍是 regex heuristic。

**影响**

- 脱敏不完整、表格型文本、中文自由描述、base64/JSON 包裹内容可能绕过。
- 未来 V2 Plugin/Tool 输出更丰富后，候选记忆来源更多，误存敏感摘要的风险会增加。

**修复方案**

1. 增加 memory candidate schema allowlist：不同 memory_type 只能保存允许字段。
2. 对 `payload` 做字段级过滤，而不是只拼接成 text 后 regex。
3. 增加“来源类型”与“提取器版本”，方便未来批量重扫/清理。
4. 对拒绝原因做用户可查看的审计，但不要保存原始敏感候选全文。

**建议测试**

- fuzz 一组手机号、身份证、连接串、PMML XML、Notebook 代码、长报告段落。
- 对嵌套 JSON payload 做字段级拒绝测试。

## P2-10 V1 与 V2/V3/V4 扩展边界需要继续守住

**证据**

`docs/roadmap.md` 已明确：

- V1.1 当前稳定线是模型验证 + Agent Memory。
- V2 才是 Plugin/Tool Runtime。
- V3/V4 分别是建模和策略能力包。

前端欢迎页已经开始为“模型开发 / 模型验证 / 策略开发”布局，这是正确方向；但除了模型验证，其他任务现在尚未有 runtime 和 workflow。

**影响**

如果在 V1.1 中直接把模型开发、策略开发塞进当前 validation pipeline，会导致：

- Notebook 契约混乱。
- 任务状态机继续膨胀。
- Agent 能力边界过早承诺，后续 V2 runtime 迁移成本变高。

**修复方案**

1. 欢迎页可继续展示未来任务卡，但未实现卡片保持 disabled 或显示“即将开放”。
2. 新任务类型先只建数据模型和 UI 壳，不复用验证 pipeline。
3. V2 runtime 落地前，不让“模型开发/策略开发”执行任意代码。
4. 任务类型建议从现在开始显式入库：`task_type = validation | modeling | strategy`，避免未来从 model_name/run_mode 推断。

**建议测试**

- 点击未开放任务卡不会打开模型验证创建框。
- 模型验证卡仍打开当前 validation dialog。
- 任务 payload 中未来新增 `task_type` 时，旧任务默认 `validation`。

## P3-01 `static/index.html` 的 cache-bust 版本手工维护，容易遗漏

**现象**

当前脚本引用带 `?v=20260613-task-entry-welcome`。每次静态资源改动都要手动改。

**建议**

短期保留手工 cache-bust；中期用应用启动时注入版本号或 git sha；发布包可用 `riskmodel_checker.__version__` 生成静态资源版本。

## P3-02 文件分类规则偏窄，需要在 V2 材料体系前统一设计

**现象**

`classify_file()` 当前只识别：

- `.ipynb`
- `.xlsx/.csv` 且名称含字典/dictionary 的数据字典
- `.feather/.csv/.parquet` 样本
- `.pmml`

`.pkl`、`.joblib`、`.txt`、`.json` 模型或 WOE 字典会被忽略。当前 V1 Notebook 契约允许建模脚本内部自己读取 `.pkl`，平台不一定需要把它作为验证材料。但未来模型开发 pack 需要更完整的 artifact taxonomy。

**建议**

V1 维持当前稳定契约；V2/V3 设计统一的 artifact role：

- `model_artifact`
- `transform_artifact`
- `woe_dictionary`
- `training_sample`
- `validation_sample`
- `deployment_pmml`

并明确哪些 artifact 可记忆、可上传、可审计、可进入报告。

## P3-03 发布前检查文档与本机 conda 命令不完全一致

**现象**

`docs/versioning.md` 的公开发布前检查示例使用普通 `python -m pytest`，而 `AGENTS.md` 指定本机 workspace 使用：

```bash
conda run -n py_313 python -m pytest ...
```

**建议**

公开文档继续保留普通 `python` 没问题；内部开发 runbook 可以补一句“本机开发使用 `conda run -n py_313 ...`”。避免贡献者把本机专属 conda 命令误认为公开安装要求。

## 推荐修复顺序

1. 先修 P1-01：加入 `static/js/*` package-data，并补 package smoke test。
2. 再修 P1-02：停止用历史 cancelled job 推断当前 stopped，补取消后重跑测试。
3. 再修 P1-03：默认限制远程 GET 数据 API，补 middleware 测试。
4. 接着修 P2-03：材料目录扫描加上限和二次确认。
5. 接着修 P2-05/P2-07：压力测试整体状态、前端 HTML 安全护栏。
6. 最后做 P2-01/P2-08：按模块逐步拆分 API/frontend/tests，配合浏览器 smoke。

## 建议新增验证命令

常规开发继续保留：

```bash
conda run -n py_313 python -m pytest -q
conda run -n py_313 python -m ruff check riskmodel_checker tests --extend-exclude '*.ipynb'
node --check riskmodel_checker/static/app.js
git diff --check
```

新增发布/前端 smoke：

```bash
conda run -n py_313 python -m build
conda run -n py_313 python -m pytest tests/test_package_data.py -q
node --check riskmodel_checker/static/js/api.js
node --check riskmodel_checker/static/js/dialogs.js
node --check riskmodel_checker/static/js/polling.js
node --check riskmodel_checker/static/js/render-agent.js
node --check riskmodel_checker/static/js/render-metrics.js
node --check riskmodel_checker/static/js/state.js
node --check riskmodel_checker/static/js/ui-utils.js
```

如果引入浏览器 smoke：

```bash
conda run -n py_313 python -m pytest tests/test_frontend_browser_smoke.py -q
```

## 不建议现在做的事

- 不建议把“模型开发/策略开发”直接塞进当前 validation pipeline。应等 V2 Plugin/Tool/Workflow runtime 或至少先建独立 task_type。
- 不建议一次性重写整个 `api.py` 或 `app.js`。当前测试很多，功能面复杂，应该按 router/view 分块迁移。
- 不建议让 Agent 或 Memory 影响 KS/AUC/PSI/分数一致性等确定性指标。当前文档边界是对的，应继续保持。
- 不建议把 pickle 作为未来插件间通用数据交换格式。业务 Notebook 可以自己读 pkl，但平台输出和 plugin contract 应优先 JSON/parquet/feather。
