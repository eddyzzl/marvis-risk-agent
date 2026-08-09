# Final local closure 代码质量复审（闭环）

**日期：** 2026-08-01

**最后更新：** 2026-08-03

**方式：** 整改由主流程完成，独立 subagent 复审保持只读；未暂存或提交。

**闭环范围：** 原 BL-01（恢复完整性）、IN-01（Candidate Lab 重复键）、`result_dataset` 精确下载绑定，以及 2026-08-03 最终门禁暴露的 fixture、worker 和资源所有权边界。

**最终结论：PASS**。

## 原 BLOCKER 已关闭：恢复绑定冲突

**原问题：** 父步骤结果绑定无效时，`PlanRepository` 会抛出 `ConflictError`；原来的恢复层没有将其转换为终止性恢复失败，启动回收会跳过该计划。

**复审证据：**

- `marvis/orchestrator/plan_recovery.py:150-156` 现已捕获 `ConflictError`，保存可审计完整性错误并将 `CHECKING` 步骤收敛为 `FAILED`。
- `tests/test_recovery.py:749-793` 直接注入父绑定冲突，断言步骤为 `FAILED` 且错误包含完整性原因。
- `tests/test_recovery.py:796-828` 通过启动回收注入同一冲突，断言 `reclaim_running_plans()` 回收该计划，并将计划和步骤均置为 `FAILED`。
- 已运行 `conda run -n py_313 python -m pytest -q tests/test_recovery.py`：**25 passed**。

**结论：** BL-01 已修复。未发现异常被吞没、回退到 latest 输出，或计划继续停在 `RUNNING` 的问题。

## 原 INFO 已关闭：Candidate Lab `artifact_id` 重复定义

- `marvis/static/js/v2/strategy_candidate_lab_contracts.js:143-156` 的 `FIELD_LABELS` 仅保留一项 `artifact_id: "产物 ID"`。
- `tests/test_frontend_strategy_candidate_lab.py:91-101` 新增唯一性断言，要求源文件中该标签定义恰好一次。
- `node --check marvis/static/js/v2/strategy_candidate_lab_contracts.js` 通过。

**结论：** IN-01 已清理且有直接回归保护。

## `result_dataset` 精确下载绑定快速复审

- `marvis/agent/plan_message_composer.py:313-425` 仅为已完成、已验证步骤生成下载元数据，URL 固定携带 `plan_id`、`step_id`、`output_ref` 和内容哈希。
- `marvis/routers/data.py:807-861` 要求绑定参数成组提供，重新认证持久化 presentation binding，逐项匹配任务、计划、步骤、输出引用、结果数据集和哈希；之后再验证当前数据集归属、哈希和实际文件。
- `tests/test_data_join_api.py:169-222` 覆盖成功下载及“数据库记录和文件同步替换后仍返回 `409`”的 fail-closed 情形。

**结论：** 未发现明显的跨任务、输出版本替换或内容哈希漂移绕过。无绑定入口只保留任务拥有权兼容行为，不削弱新的 completion-message 精确链接。

## 已关闭的 HTTP Range 兼容性问题

- `marvis/download_snapshot.py:85-163` 在已经认证并冻结的 snapshot 上解析单一 byte range；它只对 snapshot 执行 `seek/read`，不重新打开原始路径。
- 正常下载含 `Accept-Ranges: bytes`；有效范围产生 `206`、精确 `Content-Length` 和 `Content-Range`；不可满足范围关闭 snapshot 后产生 `416` 和 `Content-Range: bytes */N`。
- `marvis/routers/data.py:887-893` 及 `marvis/routers/artifacts.py:137-143,230-236,269-275` 均传入请求的 `Range` 头。
- `tests/test_artifact_api.py:776-812` 覆盖 task、strategy、generic artifact 的分段成功和 `416`；`tests/test_data_join_api.py:188-197` 覆盖 result dataset 分段成功。

**结论：** 原 Range WARNING 已关闭。快照生成器和 `BackgroundTask` 都会关闭同一私有 snapshot；重复关闭是安全的，未发现源文件重新打开或资源泄漏路径。

## 已关闭的 Ruff WARNING：`Response` 重复导入

`marvis/routers/data.py` 已移除 `from fastapi.responses import Response`，保留原有的 `fastapi.Response` 导入。复跑以下检查返回 `All checks passed!`：

```text
conda run -n py_313 python -m ruff check marvis/download_snapshot.py marvis/routers/artifacts.py marvis/routers/data.py
```

## 验证

- `conda run -n py_313 python -m pytest -q tests/test_recovery.py`：**25 passed**。
- `conda run -n py_313 python -m pytest -q tests/test_artifact_api.py::test_verified_snapshot_downloads_preserve_single_range_semantics tests/test_data_join_api.py::test_data_join_conversation_end_to_end`：**2 passed**。
- 受影响 Python 文件 Ruff 检查：通过（`All checks passed!`）。
- `node --check marvis/static/js/v2/strategy_candidate_lab_contracts.js`：通过。
- `git diff --check`：通过。

## 验收边界

真实外部数据、原生 Windows 删除和生产签字仍需对应环境验收；它们不是本次代码复审中的缺陷，也不能替代相应验收证据。

## 最终判定

**PASS。** 原 BL-01、IN-01、`result_dataset` 精确绑定、verify-to-open 竞态、HTTP Range 兼容性和 Ruff F811 均已关闭；本次最终只读复核未发现剩余可证明缺陷。

## 2026-08-03 补充回归复审

综合审计后续收口又发现并关闭多项此前测试未覆盖的真实边界；修复由主流程完成，
补充复审保持只读。

### 人工筛选不再冒充第二次 Tool 执行

原 `GateExecutionAdapter` 会为人工编辑后的特征集合写入新的 screen output version，
但该版本没有匹配的 succeeded run receipt。严格 parent binding 启用后，真实 Modeling
API 旅程会以 `dependency ... result binding is invalid` 中断。

当前实现把受原 screen evidence 约束后的 selection 与 gate confirmation 原子写入
concrete inputs；原 screen output ref、hash 和 succeeded run receipt 均保持不可变。
此前失败的 `tests/test_modeling_api.py::test_modeling_end_to_end` 已转绿；PlanDriver
完整文件为 `111 passed`，selection/stale-control 与治理 fail-closed 定向集也通过。

### 一年期 immutable 静态缓存键改为内容寻址

原版本键只取全部 JS/CSS 的最大 mtime；修改较旧文件并保留时间戳时，URL 不变但
响应仍宣告一年期 `immutable`。当前 `_static_asset_version` 对排序后的相对路径和文件
字节生成 SHA-256；改名、增删或内容变化都会换 key。保留 mtime 回归及完整
`tests/test_app_security.py` 为 `38 passed`，真实 Chromium shell smoke 为 `1 passed`。

### 迁移测试夹具

新 receipt migration 首次进入全仓门禁后，暴露若干历史测试只创建目标表、却把残缺
数据库标成完整 v1+。夹具现保留自 schema v1 就存在的 `plan_step_runs` 前置表；没有
修改、重排或回填已发布 migration。共享 fixture 已覆盖全部 `PRAGMA user_version`
成功夹具；schema/migration 组合 `362 passed`，完整 `tests/test_db.py` 为 `86 passed`。

### 门禁夹具和静态 guard 不再假绿

- special-value gate 改为登记 task-owned exact result dataset，并通过真实 screen executor
  生成 output/receipt，不再向 store 直接塞入不存在的数据集引用。整文件 `18 passed`，
  相关 PlanDriver/frontend/template 组合 `232 passed`。
- artifact identity guard 不再维护手写 façade 列表，而是对全部 `marvis/**/*.py` 做 AST
  扫描；私有 identity 定义、alias/attribute/import 和 namespace literal 只允许 canonical
  owner。受影响组合 `138 passed`。
- Strategy sample-plan migration 测试删除已从产品 contract 移除的
  `pass_memory_kwargs=False`，按当前 runtime/memory contract 构造；相关组合 `64 passed`。

### Notebook worker 使用隔离可信 bootstrap

原 subprocess 直接以 notebook 目录为 cwd 运行 `-m marvis.notebook_worker`，在 source
checkout 未安装包时会找不到模块，并允许 notebook 目录里的同名 `marvis` 影子包抢先
执行。现在使用同一 `sys.executable` 的 `-I -c` 启动，只把解析后的仓库根加入
`sys.path` 后导入可信 worker，同时仍保留 notebook cwd 语义。真实 shadow attack 与
命令形状均有回归；non-slow Notebook 集为 `43 passed, 1 deselected`。

### automatic-tree Parquet 生命周期与输出所有权

全量门禁首先证明 schema/preflight 失败会让 native Parquet reader 的 descriptor 延迟
释放。初版 close guard 虽修复 descriptor，却在独立复审中暴露两项回归：竞争方在
`xb` 前抢先创建的文件可能被误删，cleanup 的 `PermissionError` 可能掩盖主异常。

最终实现由单一 `finally` exact-once 关闭 reader，并由 `_ExclusiveOutputOwnership` 只在
本次 `xb` 成功后取得清理权。普通异常、`KeyboardInterrupt`、`close()` 失败、竞争创建、
cleanup denial 和成功路径均有直接测试；关闭失败只作为主异常 note，或在成功路径删除
本次拥有的输出后 fail closed。整文件 `26 passed`，automatic-tree/weighted-tree 扩大
回归 `692 passed`，最终独立复审 `PASS`。

### 最新完整本地门禁

在最终代码状态执行：

```text
conda run --no-capture-output -n py_313 scripts/check --fast -- -q --maxfail=1
```

exit `0`：`11140 passed, 235 deselected, 32 warnings`，耗时
`4778.77s（1:19:38）`；同一命令中的 `git diff --check`、全仓 Ruff 和前端 JavaScript
语法检查全部通过。warning 已明确归类为 LightGBM 弃用、固定迭代预算 MLP 未收敛与
`sample_dataset()` 兼容弃用，不作为静默失败处理。

### 补充结论

Runtime/memory、SampleDesign V2、controller/pet/DSL/GC、四类 fixture、人工筛选和静态
缓存，以及后续 schema/special-value/artifact/Notebook/automatic-tree 边界均经定向或
独立只读复核；当前未发现未关闭的 P0/P1/P2，最终全仓 fast gate 已退出 0。真实材料、
真实 LLM、原生 Windows、远端 immutable SHA 和生产责任签字继续是独立外部门禁。
