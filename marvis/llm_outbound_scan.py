"""Material scan state for explanations, without file or Notebook contents."""
from marvis.llm_outbound import _DROP, _identifier, _mint_prompt
from marvis.llm_outbound_reports import _labels, _numeric, _rows
from marvis.llm_outbound_validation import validation_metrics

_MATERIALS = ("material_notebook", "material_sample", "material_model_pmml", "material_data_dictionary")
_LABEL_IDS = {
    "Notebook 文件": "material_notebook", "样本数据": "material_sample",
    "PMML 模型": "material_model_pmml", "数据字典": "material_data_dictionary",
    "数据字典/特征元数据": "material_data_dictionary",
    "Notebook RMC 契约": "notebook_contract", "验证输入契约": "validation_input_contract",
}
_STATUS = {"success": "success", "passed": "success", "pass": "success", "ok": "success",
           "通过": "success", "已通过": "success", "error": "error", "failed": "error",
           "fail": "error", "pending": "pending", "warning": "warning", "skipped": "skipped"}
_CONTRACT_NAMES = {"RMC_SAMPLE_DF", "RMC_SCORE_FN", "RMC_TARGET_COL", "RMC_ALGORITHM"}


def scan_context(evidence, *, workflow_version):
    source = evidence if isinstance(evidence, dict) else {}
    raw = source.get("scan") if isinstance(source.get("scan"), dict) else source
    checks = []
    by_id = {}
    for index, check in enumerate(_rows(raw.get("checks")), 1):
        name = check.get("id")
        # A known legacy label is usable only when no id is provided. Unknown
        # extension ids cannot impersonate a required platform check by label.
        if name is None:
            name = _LABEL_IDS.get(check.get("label")) if isinstance(check.get("label"), str) else None
        status = check.get("status")
        status = _STATUS.get(status.strip().lower(), "unknown") if isinstance(status, str) else "unknown"
        item = {"check_index": index, "status": status}
        if isinstance(name, str) and name in _LABEL_IDS.values():
            item["id"] = name
            by_id.setdefault(name, []).append(status)
        checks.append(item)
    required = [*_MATERIALS, "validation_input_contract" if workflow_version == 2 else "notebook_contract"]
    satisfied = [name for name in required if by_id.get(name) and all(status == "success" for status in by_id[name])]
    interpretation = {
        "required_checks": required, "satisfied_checks": satisfied,
        "missing_or_unsatisfied_checks": [name for name in required if name not in satisfied],
        "required_materials_complete": len(satisfied) == len(required),
        "failed_check_count": sum(item["status"] == "error" for item in checks),
        "unknown_check_count": sum("id" not in item for item in checks),
    }
    clean = {"checks": checks, **_labels(raw, {"status": {"scanned", "failed"}})}
    contract = raw.get("notebook_contract")
    if isinstance(contract, dict):
        typed = validation_metrics({"algorithm": contract.get("algorithm")})
        for key in ("read_only", "sample_df_defined", "score_fn_defined", "target_col_defined", "algorithm_defined", "algorithm_valid"):
            if type(contract.get(key)) is bool:
                typed[key] = contract[key]
        target = _identifier(contract.get("target_col"))
        if target is not _DROP:
            typed["target_col"] = target
        for key in ("missing_names", "invalid_names"):
            if isinstance(contract.get(key), list):
                typed[key] = [name for name in contract[key] if isinstance(name, str) and name in _CONTRACT_NAMES]
        typed["diagnostic_present"] = bool(contract.get("error") or contract.get("algorithm_error"))
        clean["notebook_contract"] = typed
    pending = raw.get("validation_input_contract")
    if isinstance(pending, dict):
        clean["validation_input_contract"] = {
            **_labels(pending, {"status": {"pending_confirmation", "confirmed", "invalid", "stale"}}),
            **_numeric(pending, ("revision",)),
            **{key: pending[key] for key in ("read_only", "needs_confirmation") if type(pending.get(key)) is bool},
        }
    return {"scan": clean, "scan_interpretation": interpretation}


def scan_prompt(evidence, *, workflow_version):
    return _mint_prompt({
        "stage": "scan", "validation_workflow_version": 2 if workflow_version == 2 else 1,
        "evidence": scan_context(evidence, workflow_version=workflow_version),
        "history_comparison_available": False,
        "instructions": (
            "只解释本次材料完备性与检查状态，不分析评分、KS/AUC/PSI 或报告结论。"
            "required_checks 为本版本所需检查，satisfied_checks 表示已满足，不能声称这些材料缺失；"
            "missing_or_unsatisfied_checks 表示未提供或未通过检查，不等同于一定缺文件。"
            "其他检查失败和平台 failed 状态仍须说明，不能仅凭必需项齐全就宣布扫描通过。"
            "material_notebook 是 Notebook，material_sample 是样本，material_model_pmml 是 PMML，"
            "material_data_dictionary 是数据字典/特征元数据。"
            "V1 检查 Notebook RMC 契约；V2 检查验证输入契约候选，不能要求 V1 RMC 运行契约。"
            "pending_confirmation 或 needs_confirmation=true 表示候选仍需确认，不能声称已确认。"
            "源码、文件名、错误原文、原始字段值及样本未发送，具体诊断应查看本地检查项。"
            "pickle/pkl 和后续评分、效果产物不是扫描必需材料；不要猜测其他材料缺失。"
        ),
    })
