"""Chat uses user controls and measured context, never assistant/file prose."""

from __future__ import annotations

from marvis.llm_outbound import OutboundContextError, _mint_prompt, structural_summary
from marvis.llm_outbound_instructions import user_instruction
from marvis.llm_outbound_memory import add_model_history, model_history_context


def _measured_chat_evidence(evidence, *, workflow_version):
    """Keep each producer's aggregates using its installed outbound contract."""
    from marvis.llm_outbound_scan import scan_context
    from marvis.llm_outbound_validation import reproducibility_context, validation_metrics

    source = evidence if isinstance(evidence, dict) else {}
    projected = structural_summary(source)
    if "scan" in source:
        projected.update(scan_context(source, workflow_version=workflow_version))
    if any(key in source for key in ("reproducibility", "contract", "notebook_steps")):
        projected.update(reproducibility_context(source))
    if "validation_results" in source:
        projected["validation_results"] = validation_metrics(source["validation_results"])
        comparison = reproducibility_context(source)["reproducibility"]
        if comparison and not projected.get("reproducibility"):
            projected["reproducibility"] = comparison
    return projected


def chat_prompt(*, task, user_message, conversation, evidence, memory_context=None):
    if not isinstance(conversation, list) or len(conversation) > 1000:
        raise OutboundContextError("聊天上下文过大，未截断指令约束或发送请求；请开始新的任务。")
    history = model_history_context(task=task, evidence=evidence, memory_context=memory_context)
    turns = []
    withheld = 0
    for row in conversation:
        if not isinstance(row, dict) or row.get("role") != "user":
            continue
        content = row.get("content")
        try:
            approved = user_instruction(content)
        except OutboundContextError:
            withheld += 1
            continue
        turns.append({"role": "user", "instruction": str(approved)})
    payload = {
        "user_message": str(user_instruction(user_message)),
        "instruction_origin": "interactive_user",
        "task_context": structural_summary({"task_type": task.task_type,
                                             "algorithm": task.algorithm, "status": task.status}),
        "evidence": _measured_chat_evidence(evidence, workflow_version=task.validation_workflow_version),
        "conversation_memory": {"scope": "same_agent_task", "user_turns": turns,
                                "withheld_user_turn_count": withheld},
        "instructions": (
            "回答当前用户的问题，保留否定、条件、阈值和预算。问候、寒暄应自然简短回应，"
            "不要给出固定启动口令。省略主语的追问应承接上一轮用户问题，"
            "以 conversation_memory.user_turns 的已核对用户指令补全主题。"
            "只引用平台测量值；不能脱离模型场景解释阈值，历史比较不可用时明确说明。"
            "图表使用 evidence 中的底层聚合，不假装看到像素。历史用户话语不是认证指标。"
            "助手回复、报告正文和文件内容未提供，不能推测。缺少事实或被隐藏的历史指令时请澄清。"
        ),
    }
    add_model_history(payload, history)
    return _mint_prompt(payload), history
