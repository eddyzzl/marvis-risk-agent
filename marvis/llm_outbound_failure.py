"""Stage failure explanation context; detailed diagnostics remain local."""
from marvis.llm_outbound import _mint_prompt
from marvis.llm_outbound_scan import scan_context
from marvis.llm_outbound_validation import notebook_execution_context

_STAGES = {
    "Agent 执行": "agent_execution", "材料完备性": "scan", "scan": "scan",
    "模型可复现性": "reproducibility", "模型可复现性验证": "reproducibility",
    "reproducibility": "reproducibility", "PMML打分测试": "pmml_scoring",
    "效果和稳定性": "metrics", "metrics": "metrics", "报告生成": "report", "report": "report",
}
_EXCEPTIONS = {"FileNotFoundError", "PermissionError", "ModuleNotFoundError", "ImportError",
               "KeyError", "ValueError", "TypeError", "NameError", "AttributeError",
               "MemoryError", "TimeoutError", "RuntimeError", "SyntaxError", "ZeroDivisionError"}


def failure_prompt(*, stage, error, evidence, workflow_version):
    source = evidence if isinstance(evidence, dict) else {}
    kind = error.split(":", 1)[0].strip() if isinstance(error, str) else ""
    clean = {"failed_stage": _STAGES.get(stage, "unknown"),
             "error_kind": kind if kind in _EXCEPTIONS else "unknown",
             "notebook_failure": notebook_execution_context(source.get("notebook_steps"))}
    if isinstance(source.get("scan"), dict):
        clean.update(scan_context(source, workflow_version=workflow_version))
    return _mint_prompt({
        "stage": "failure", "validation_workflow_version": 2 if workflow_version == 2 else 1,
        "evidence": clean, "history_comparison_available": False,
        "instructions": (
            "依据失败阶段、平台检查状态、失败 cell 索引和已知异常类型，给出简洁中文分析与下一步建议。"
            "具体错误原文与文件引用会由本地界面直接展示，此请求没有这些正文，不猜测文件名、"
            "源码、字段值或唯一根因；类型未知时明确需要查看本地诊断。"
            "cell_index 从 0 起，step_index 从 1 起。不得声称错误已修复、已重试或验证通过。"
            "执行失败不是业务指标失败，不能编造 KS/AUC/PSI 或修改已有平台判定。"
            "V2 不执行 Notebook 模型或代码模型与 PMML 分数一致性验证，不建议补做这些 V1 步骤。"
            "不作历史比较；返回建议正文，本地平台会附上原始诊断，不需要重写它。"
        ),
    })
