"""Recovery questions with diagnostic codes and platform status only."""
from marvis.llm_outbound import OutboundContextError, _mint_prompt
from marvis.llm_outbound_instructions import user_instruction


CODES = frozenset({"material_file_missing", "material_ingest_failed", "workflow_setup_incomplete",
    "workflow_execution_failed", "workflow_reconciliation_required", "platform_parquet_column_alias_failed",
    "platform_metadata_serialization_failed", "csv_field_count_mismatch", "workflow_step_failed"})
WORKFLOWS = frozenset({"data_join", "feature_analysis", "modeling", "strategy", "vintage", "portfolio", "monitoring", "validation"})


def recovery_question_prompt(question, diagnostic):
    if not isinstance(diagnostic, dict) or diagnostic.get("code") not in CODES:
        raise OutboundContextError("当前故障缺少受支持的结构化诊断类别，请在本地核对技术信息后继续；尚未启动重试。")
    evidence = {"code": diagnostic["code"], "execution_started": False}
    for key, choices in (("workflow", WORKFLOWS), ("phase", {"material_ingest", "prepare", "execution"})):
        if diagnostic.get(key) in choices:
            evidence[key] = diagnostic[key]
    for key in ("retryable", "auto_recoverable"):
        value = diagnostic.get(key, key == "retryable")
        if type(value) is not bool:
            raise OutboundContextError("故障恢复状态无效，请在本地核对；尚未启动重试。")
        evidence[key] = value
    if evidence["code"] == "workflow_reconciliation_required":
        evidence.update(retryable=False, auto_recoverable=False, reconciliation_required=True)
    for key in ("line_number", "expected_fields", "actual_fields"):
        value = diagnostic.get(key)
        if type(value) is int and 0 <= value <= 10**12:
            evidence[key] = value
    return _mint_prompt({"stage": "workflow_recovery_chat", "current_question": user_instruction(question),
        "failure_evidence": evidence,
        "instruction": "根据故障类别和状态回答完整问题。未提供文件名、位置正文、异常或诊断叙事，不能补写未提供的细节；需要具体材料时请用户在本地核对。当前尚未启动重试或配置修改；普通提问不是授权。auto_recoverable=true 可在用户明确授权后使用当前材料恢复；reconciliation_required=true 必须先核对执行凭据，禁止重试。"})
