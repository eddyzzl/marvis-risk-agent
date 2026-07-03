# Phase 2C — V1 Compatibility Pack（函数级 spec）

## 文档状态

- 状态：已实现并验证
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 9、13 节）
- 前置依赖：Phase 1（Plugin/Tool Runtime）；现有 V1 `pipeline.py` / `notebooks.py` / `validation/` / `output/`
- 目标：把当前稳定 V1 模型验证流程包装成内置 Plugin `v1_compat`，让 Phase 2 的 `model_validation` Workflow 可以真实调用 V1 扫描、Notebook、指标和报告能力。

## 边界

Phase 2C 只做 **V1 能力包装**，不改 V1 验证口径，不新增建模、拼表或策略能力。

- 保持现有 `RMC_SAMPLE_DF`、`RMC_TARGET_COL`、`RMC_ALGORITHM`、`RMC_SCORE_FN` Notebook 契约不变。
- 保持 Notebook 内存模型分 vs PMML 分一致性验证主口径不变。
- 保持手动模式和 Agent P1 模式可用；旧 API 不下线。
- Tool output 只暴露结构化摘要和 artifact ref，不把原始样本、Notebook 源码或 PMML 内容写进 output/audit。

## 捍卫的不变量

- **INV-1/INV-2**：所有 KS/AUC/PSI/一致性指标仍由现有确定性 validation/pipeline 代码计算，LLM 只读取结果。
- **INV-5**：Tool output 和 audit 不保存原始样本、客户明细、完整 Notebook、PMML/模型文件内容。
- **INV-6**：通过 Phase 1 ToolRunner 子进程执行，失败收敛为结构化 ToolResult。
- **INV-8**：每个 V1 compatibility tool 调用都有审计。
- **INV-10**：`packs/v1_compat` 只通过 pipeline/domain/repository 等稳定边界调用 V1 能力，不从 `api.py` import 业务函数。

## 模块布局

```text
marvis/packs/v1_compat/
  __init__.py
  manifest.json
  contracts.py       V1 tool input/output dataclasses 和 payload helpers
  adapters.py        task/repo/settings/pipeline 适配层
  tools.py           tool_scan_materials / tool_run_notebook / tool_compute_validation_metrics / tool_render_reports
tests/test_v1_compat_pack.py
```

## Part A — Tool 契约

### A-1 `tool_scan_materials`

```python
def tool_scan_materials(inputs: dict, ctx: ToolContext) -> dict:
    """扫描任务材料目录并写入现有任务证据。

    inputs:
      {"task_id": str}

    output:
      {
        "task_id": str,
        "status": "scanned"|"failed",
        "materials": [{"role": str, "path": str, "name": str}],
        "checks": [{"name": str, "status": str, "detail": str}],
      }

    异常:
      不主动抛给主进程；材料缺失、路径错误等由 ToolRunner 收敛为 ok=False 或 output.status="failed"。
    """
```

- **实现要点**：复用现有 `scan_source_dir` 和任务材料识别逻辑；路径输出用相对材料目录或相对 task/workspace 的安全路径。
- **测试要点**：完整材料识别成功；缺 Notebook/样本/PMML 时输出 structured checks；不存在 `task_id` 失败；output 不含原始文件内容。

### A-2 `tool_run_notebook`

```python
def tool_run_notebook(inputs: dict, ctx: ToolContext) -> dict:
    """执行现有 V1 Notebook 阶段。

    inputs:
      {"task_id": str}

    output:
      {
        "task_id": str,
        "status": "executed"|"failed",
        "notebook_cells": int,
        "sample_ref": str,
        "runtime_model_ref": str,
        "evidence_ref": str,
      }
    """
```

- **实现要点**：复用 `run_notebook_stage` / `run_staged_pipeline` 的 Notebook 阶段能力；保留现有 execution environment 行为。
- **测试要点**：Notebook 契约注入不变；执行失败返回失败摘要；输出只有 ref/摘要，不包含完整 Notebook 源码。

### A-3 `tool_compute_validation_metrics`

```python
def tool_compute_validation_metrics(inputs: dict, ctx: ToolContext) -> dict:
    """执行现有 V1 模型效果、稳定性和一致性验证。

    inputs:
      {"task_id": str}

    output:
      {
        "task_id": str,
        "status": "writing_artifacts"|"review_required"|"failed",
        "ks": float,
        "auc": float,
        "psi": float | None,
        "score_consistency_passed": bool,
        "validation_results_ref": str,
      }
    """
```

- **契约要求**：`ks`、`auc`、`psi` 必须作为 top-level 字段出现在 `output_schema`，供 Phase 2 `PlanValidator._check_determinism_checks` 强制 range post_check。
- **实现要点**：复用现有 metrics 阶段和 `validation_results.json`；如果分组没有 PSI，`psi=null` 且 output_schema 允许 null。
- **测试要点**：KS/AUC/PSI 与现有 V1 API 输出一致；过拟合/一致性复核状态不变；缺 PSI 场景合法；指标区间 post_check 可通过。

### A-4 `tool_render_reports`

```python
def tool_render_reports(inputs: dict, ctx: ToolContext) -> dict:
    """生成现有 V1 Excel/Word 报告。

    inputs:
      {"task_id": str}

    output:
      {
        "task_id": str,
        "status": "succeeded"|"review_required"|"failed",
        "artifacts": [
          {"kind": "excel"|"word", "path": str, "download_url": str | None}
        ],
      }
    """
```

- **实现要点**：复用现有 output 渲染和下载路径；报告生成前是否需要人工确认由 Phase 2 PlanStep `needs_confirmation=True` 控制，不在 tool 内绕过。
- **测试要点**：生成 Excel/Word 成功；报告失败保留失败阶段；artifact path 不越界。

## Part B — manifest

`manifest.json`：

```json
{
  "name": "v1_compat",
  "version": "0.1.0",
  "display_name": "V1 Validation Compatibility Pack",
  "description": "Wraps stable V1 model validation stages as MARVIS tools.",
  "module": "marvis.packs.v1_compat.tools",
  "tools": [
    {"name": "scan_materials", "entrypoint": "tool_scan_materials", "determinism": "deterministic", "timeout_seconds": 60, "failure_policy": "fail"},
    {"name": "run_notebook", "entrypoint": "tool_run_notebook", "determinism": "deterministic", "timeout_seconds": 3600, "failure_policy": "fail"},
    {"name": "compute_validation_metrics", "entrypoint": "tool_compute_validation_metrics", "determinism": "deterministic", "timeout_seconds": 1800, "failure_policy": "fail"},
    {"name": "render_reports", "entrypoint": "tool_render_reports", "determinism": "deterministic", "timeout_seconds": 600, "failure_policy": "fail"}
  ],
  "hooks": [],
  "permissions": ["read:task", "write:task", "read:materials", "write:artifacts"]
}
```

实现时必须补全每个 tool 的 `input_schema` / `output_schema`，上方 JSON 只表达关键字段。

## Part C — 适配层

```python
def load_v1_task_context(ctx: ToolContext, task_id: str) -> V1TaskContext:
    """从 ctx.workspace 构造 Settings / TaskRepository / PipelineSettings，读取 task。

    异常:
      TaskNotFoundError：task_id 不存在。
      PermissionError：task artifact/material path 越界。
    """
```

- `packs/v1_compat` 不 import `api.py`。
- 对现有 pipeline 函数缺少的可复用边界，优先在 `pipeline.py` / `recovery.py` 提取小函数，再由旧 API 和 v1_compat 共用。
- 不复制 V1 验证逻辑；包装层只做参数转换和结构化 output。

## Part D — HTTP/前端关系

Phase 2C 不新增用户入口。它只让 Phase 2 Workflow 能调用 V1 能力。

旧的手动按钮、Agent P1 流程和 API 仍可直接走现有 V1 路径。等前端 V2 落地后，同一个“模型验证”任务可以通过 Workflow 调用 `v1_compat`；这属于新增入口，不替代旧入口。

## Part E — 测试计划

| 文件 | 覆盖 |
|------|------|
| `tests/test_v1_compat_pack.py` | 四个 tool 经 ToolRunner 子进程往返、schema、错误收敛 |
| `tests/test_pipeline_v2.py` | V1 阶段结果回归，确保包装后不改口径 |
| `tests/test_agent_api.py` / `tests/test_api_v2.py` | 旧手动/Agent P1 入口仍可用 |
| `tests/test_plugin_manifest.py` | `v1_compat` manifest 被内置包发现并注册 |

必须覆盖：

- 完整 happy path：扫描 → Notebook → 指标 → 报告。
- Notebook 执行失败、PMML 一致性失败、报告生成失败。
- 输出 schema 不含敏感文件内容。
- `model_validation` 模板真实运行时四个 step 都能 resolve。

## Part F — 任务执行顺序

```text
1. A/B manifest + output_schema 固定
2. C adapters：读取 task/settings/repo/pipeline context
3. A-1 scan_materials
4. A-2 run_notebook
5. A-3 compute_validation_metrics
6. A-4 render_reports
7. Phase 2 model_validation 模板接入真实 v1_compat
8. 回归旧 V1 手动/Agent P1 API
```

Phase 2C 完成标志：`model_validation` Workflow 不再依赖 `_sample` 桩工具，能通过 `v1_compat` 执行当前 V1 验证流程，同时旧 V1 入口行为不变。
