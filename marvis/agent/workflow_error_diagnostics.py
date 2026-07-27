from __future__ import annotations

from pathlib import Path

from marvis.agent.gates.contracts import FailureEnvelope
from marvis.data.errors import CsvParseError, DataIngestError
from marvis.error_kinds import ErrorKind


_WORKFLOW_NAMES = {
    "data_join": "数据处理",
    "feature_analysis": "特征分析",
    "modeling": "模型开发",
    "strategy": "策略分析",
    "vintage": "Vintage 风险分析",
    "portfolio": "组合分析",
}


def build_workflow_error_diagnostic(
    *,
    workflow: str,
    exc: Exception,
    task=None,
    setup_error: bool = False,
) -> dict:
    workflow_name = _WORKFLOW_NAMES.get(workflow, "工作流")
    if isinstance(exc, CsvParseError):
        diagnostic = _csv_parse_diagnostic(workflow, workflow_name, exc)
    elif isinstance(exc, FileNotFoundError):
        filename = Path(str(getattr(exc, "filename", "") or "")).name or "材料文件"
        diagnostic = {
            "code": "material_file_missing",
            "phase": "material_ingest",
            "title": f"{workflow_name}未开始",
            "summary": f"找不到需要读取的材料文件 `{filename}`。",
            "cause": "文件可能已被移动、重命名，或任务保存的材料路径已失效。",
            "location": filename,
            "evidence": [{"label": "缺失文件", "value": filename}],
            "actions": [
                "确认材料仍在任务目录中，且文件名没有变化。",
                "重新上传或重新绑定材料后再次发起任务。",
            ],
            "retryable": True,
            "error_kind": "file_missing",
        }
    elif isinstance(exc, DataIngestError):
        diagnostic = {
            "code": "material_ingest_failed",
            "phase": "material_ingest",
            "title": f"{workflow_name}未开始",
            "summary": "平台未能把材料读取为可分析的数据集。",
            "cause": _safe_message(exc),
            "location": "材料读取阶段",
            "evidence": [],
            "actions": [
                "确认文件格式、扩展名和文件内容一致。",
                "用 Excel/数据工具重新导出材料后再次发起任务。",
            ],
            "retryable": True,
            "error_kind": "data_ingest",
        }
    elif setup_error:
        diagnostic = {
            "code": "workflow_setup_incomplete",
            "phase": "prepare",
            "title": f"{workflow_name}尚未就绪",
            "summary": _safe_message(exc),
            "cause": "当前材料或任务配置不足以生成可执行计划。",
            "location": "计划准备阶段",
            "evidence": [],
            "actions": [
                "按上面的缺失项补充或调整材料。",
                f"完成后重新发起{workflow_name}。",
            ],
            "retryable": True,
            "error_kind": "setup",
        }
    else:
        diagnostic = {
            "code": "workflow_execution_failed",
            "phase": "prepare",
            "title": f"{workflow_name}未完成",
            "summary": f"{workflow_name}在准备或执行阶段停止，后续步骤没有继续。",
            "cause": "平台遇到了未能自动恢复的执行异常。",
            "location": "当前工作流步骤",
            "evidence": [],
            "actions": [
                "展开技术信息并核对失败位置。",
                "修正材料或参数后重试；若重复出现，请保留任务编号排查。",
            ],
            "retryable": True,
            "error_kind": "execution",
        }

    technical = _technical_detail(exc, task=task)
    return {
        "schema_version": "workflow_error.v1",
        "workflow": workflow,
        "exception_type": exc.__class__.__name__,
        "technical_detail": technical,
        **diagnostic,
    }


def failure_envelope_for_diagnostic(diagnostic: dict) -> dict:
    return FailureEnvelope(
        failed_step_id=None,
        error_kind=str(diagnostic.get("error_kind") or "execution"),
        message=str(diagnostic.get("summary") or ""),
        retryable=bool(diagnostic.get("retryable", True)),
        suggested_actions=tuple(str(item) for item in diagnostic.get("actions") or []),
        downstream_reset="none",
    ).to_dict()


def workflow_error_content(diagnostic: dict) -> str:
    lines = [f"**{diagnostic['title']}**", str(diagnostic["summary"])]
    cause = str(diagnostic.get("cause") or "").strip()
    if cause:
        lines.extend(["", f"**原因**：{cause}"])
    location = str(diagnostic.get("location") or "").strip()
    if location:
        lines.append(f"**问题位置**：{location}")
    evidence = diagnostic.get("evidence") or []
    if evidence:
        lines.extend(["", "**已确认**："])
        lines.extend(
            f"- {item.get('label', '证据')}：{item.get('value', '')}"
            for item in evidence
        )
    actions = diagnostic.get("actions") or []
    if actions:
        lines.extend(["", "**处理建议**："])
        lines.extend(f"{index}. {action}" for index, action in enumerate(actions, start=1))
    return "\n".join(lines)


def _csv_parse_diagnostic(workflow: str, workflow_name: str, exc: CsvParseError) -> dict:
    filename = Path(exc.path).name
    line_number = exc.line_number
    expected = exc.expected_fields
    actual = exc.actual_fields
    if line_number is not None and expected is not None and actual is not None:
        summary = (
            f"CSV `{filename}` 第 {line_number} 行字段数不一致："
            f"预期 {expected} 列，实际 {actual} 列。"
        )
        location = f"{filename} · 第 {line_number} 行"
    else:
        summary = f"CSV `{filename}` 的行结构无法按同一组字段解析。"
        location = filename
    evidence = [{"label": "文件", "value": filename}]
    if line_number is not None:
        evidence.append({"label": "行号", "value": str(line_number)})
    if expected is not None:
        evidence.append({"label": "预期列数", "value": str(expected)})
    if actual is not None:
        evidence.append({"label": "实际列数", "value": str(actual)})
    line_hint = f"第 {line_number} 行及前一行" if line_number is not None else "报错行附近"
    return {
        "code": "csv_field_count_mismatch",
        "phase": "material_ingest",
        "title": f"{workflow_name}未开始",
        "summary": summary,
        "cause": (
            "已确认是 CSV 行字段数不一致。常见可能原因包括分隔符混用、字段内分隔符未加引号，"
            "或文件内容与扩展名不一致；平台没有跳过坏行，以免静默改变样本。"
        ),
        "location": location,
        "evidence": evidence,
        "actions": [
            f"用 Excel 或文本工具检查 `{filename}` 的{line_hint}，修正分隔符或引号。",
            "确认文件是真正的 CSV；若实际是 Excel，请另存为 `.xlsx` 或重新导出 UTF-8 CSV。",
            f"保持每行列数与表头一致后，重新发起{workflow_name}。",
        ],
        "retryable": True,
        "error_kind": ErrorKind.CSV_PARSE,
        "impact": "数据集未注册，执行计划未生成，后续步骤均未运行。",
        "workflow": workflow,
        "line_number": line_number,
        "expected_fields": expected,
        "actual_fields": actual,
    }


def _technical_detail(exc: Exception, *, task=None) -> str:
    if isinstance(exc, CsvParseError):
        detail = f"ParserError: {exc.technical_message}"
    else:
        detail = f"{exc.__class__.__name__}: {_safe_message(exc)}"
    source_dir = str(getattr(task, "source_dir", "") or "")
    if source_dir:
        detail = detail.replace(source_dir, "<材料目录>")
    return detail[:2000]


def _safe_message(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


__all__ = [
    "build_workflow_error_diagnostic",
    "failure_envelope_for_diagnostic",
    "workflow_error_content",
]
