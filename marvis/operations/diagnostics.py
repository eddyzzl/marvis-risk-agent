"""Stable UI-facing states and next actions for deterministic operations."""

_MESSAGES = {
    "no_new_data": (
        "本期没有新数据",
        "publish_source",
        "绑定覆盖本期的新数据后重新评估",
    ),
    "window_incomplete": (
        "数据尚未完整覆盖本期",
        "publish_source",
        "补齐数据和覆盖水位后重新评估",
    ),
    "future_source_watermark": (
        "来源覆盖水位晚于当前时间",
        "correct_source",
        "核对来源时区和覆盖水位",
    ),
    "labels_immature": (
        "结果标签尚未成熟",
        "wait_for_labels",
        "等待观察窗口成熟后重新评估",
    ),
    "labels_incomplete": (
        "本期结果标签不完整",
        "publish_source",
        "补齐结果标签后重新评估",
    ),
    "monitoring_source_error": (
        "监控数据来源不可读取或校验失败",
        "correct_source",
        "核对数据身份、字段、时间格式与文件完整性",
    ),
    "monitoring_tool_failed": (
        "确定性监控执行失败",
        "review_failure",
        "检查目标模型或策略及工具失败证据后重新评估",
    ),
    "monitoring_execution_failed": (
        "监控执行失败",
        "review_failure",
        "检查运行事件并修复后重新评估",
    ),
    "monitoring_green": (
        "本期监控检查通过",
        "continue_monitoring",
        "继续观察后续数据窗口",
    ),
    "monitoring_amber": (
        "本期监控需要关注",
        "review_evidence",
        "查看证据并决定是否调整监控或发起新策略",
    ),
    "monitoring_red": (
        "本期监控发现风险异常",
        "review_evidence",
        "由负责人审查证据并通过正常审批流程处置",
    ),
    "monitoring_not_available": (
        "本期无法形成监控判断",
        "review_evidence",
        "查看证据并补齐所需数据",
    ),
    "running": ("监控正在执行", "wait", "等待本次执行完成"),
    "operations_stopped": (
        "应用停止导致监控中断",
        "restart_runtime",
        "恢复应用后按重试策略继续",
    ),
}


def diagnostic(code: str) -> dict:
    summary, action, label = _MESSAGES.get(
        code, ("监控状态需要核对", "review_failure", "检查运行事件与证据")
    )
    return {
        "code": code,
        "summary": summary,
        "next_action": {"code": action, "label": label},
        "automatic_action_permitted": False,
    }


def period_diagnostic(period, outcome, events) -> dict:
    if outcome is not None:
        reason = (
            outcome.outcome.escalation.reason_code
            if outcome.outcome.escalation
            else f"monitoring_{outcome.outcome.level}"
        )
    else:
        reason = "running"
        for event in reversed(events):
            if event.event_type == "run_failed":
                reason = event.payload.get("error_code", "monitoring_execution_failed")
                break
    result = diagnostic(reason)
    result["execution_state"] = period.state
    result["retry_scheduled"] = period.state == "retry_wait"
    return result
