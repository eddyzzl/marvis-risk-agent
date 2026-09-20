"""Responses for governed Agent turns."""

from __future__ import annotations

from dataclasses import asdict
from marvis.agent.strategy_setup import STRATEGY_INTENT_LIMIT_PRICING
from marvis.agent.strategy_setup import STRATEGY_INTENT_PORTFOLIO_ANALYSIS
from marvis.agent.strategy_setup import STRATEGY_INTENT_STANDARD_ANALYSIS
from marvis.agent.strategy_setup import StrategySetupError
from marvis.domain import TASK_TYPE_PORTFOLIO
from marvis.domain import TaskRecord
from marvis.repositories.tasks import TaskRepository
from . import shared as shared_lane


def join_turn_response(repo: TaskRepository, task_id: str) -> dict:
    return {
        "task_id": task_id,
        "status": "ok",
        "messages": repo.list_agent_messages(task_id),
    }


def append_join_error(repo: TaskRepository, task_id: str, detail: str) -> dict:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=detail,
        metadata={"error": True},
    )
    return {
        "task_id": task_id,
        "status": "error",
        "messages": repo.list_agent_messages(task_id),
    }


def _strategy_intent_redirect_response(
    repo: TaskRepository, task: TaskRecord, intent: str
) -> dict:
    if intent == STRATEGY_INTENT_LIMIT_PRICING:
        detail = {
            "intent": intent,
            "code": "strategy_standard_workflow_inputs_required",
            "available_workflow": "limit_pricing_matrix",
            "message": (
                "已识别为额度定价矩阵。该标准 Workflow 已可执行；请用自然语言补充评分列、"
                "PD/目标列、分箱、额度与利率网格及经济参数，Agent 会编译并回显后再运行。"
            ),
        }
    elif intent == STRATEGY_INTENT_STANDARD_ANALYSIS:
        detail = {
            "intent": intent,
            "code": "strategy_standard_workflow_inputs_required",
            "available_workflows": ["profit_calc", "roll_rate_matrix"],
            "message": (
                "已识别为独立策略分析，不会降级成审批策略开发。请用自然语言补充分析列、"
                "状态顺序或利润经济口径；Agent 会选择标准 Workflow、回显口径并请求确认。"
            ),
        }
    elif intent == STRATEGY_INTENT_PORTFOLIO_ANALYSIS:
        detail = {
            "intent": intent,
            "code": "strategy_portfolio_entry_required",
            "capability_status": "available",
            "suggested_task_type": TASK_TYPE_PORTFOLIO,
            "message": (
                "已识别为组合分析意图。组合分析已有独立正式入口；当前策略任务不会误建 "
                "approval plan。请新建或切换到「组合分析」任务，绑定组合样本后继续。"
            ),
        }
    else:
        raise StrategySetupError(f"unsupported strategy redirect intent: {intent}")

    redirect_fields = {
        key: value
        for key, value in detail.items()
        if key
        in {
            "available_workflow",
            "available_workflows",
            "capability_status",
            "suggested_task_type",
        }
    }
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=detail["message"],
        metadata={
            "intent": intent,
            "kind": "clarification",
            "code": detail["code"],
            **redirect_fields,
            "clarification": dict(detail),
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "intent": intent,
        "code": detail["code"],
        **redirect_fields,
        "clarification": dict(detail),
        "messages": repo.list_agent_messages(task.id),
    }


def _strategy_clarification_response(
    repo: TaskRepository, task: TaskRecord, clarification: dict
) -> dict:
    current_input = _strategy_input_snapshot(getattr(task, "strategy_input", None))
    clarification_payload = {
        **dict(clarification),
        "current_input": current_input,
    }
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"{clarification['message']} 缺少："
            + "、".join(f"`{field}`" for field in clarification["missing_fields"])
            + "。请补充后再开始策略开发；如只需技术预览，请明确选择“快速策略分析”。"
        ),
        metadata={
            "intent": "strategy_clarification",
            "kind": "clarification",
            "current_input": current_input,
            "clarification": clarification_payload,
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "current_input": current_input,
        "clarification": clarification_payload,
        "messages": repo.list_agent_messages(task.id),
    }


def _strategy_input_snapshot(strategy_input) -> dict | None:
    """Return only the governed strategy contract for clarification prefill."""

    if strategy_input is None:
        return None
    if not isinstance(strategy_input, dict):
        return asdict(strategy_input)

    allowed = (
        "entry_mode",
        "strategy_type",
        "objective",
        "max_bad_rate",
        "min_approval_rate",
        "baseline_strategy_id",
        "profit",
    )
    payload = {key: strategy_input[key] for key in allowed if key in strategy_input}
    profit = payload.get("profit")
    if profit is not None and not isinstance(profit, dict):
        payload["profit"] = asdict(profit)
    return payload


def _strategy_request_clarification_response(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    code: str,
    message: str,
    fields: tuple[str, ...] | list[str] = (),
    ingest_notices: list[dict] | None = None,
) -> dict:
    normalized_fields = list(dict.fromkeys(str(field) for field in fields))
    normalized_notices = [
        dict(notice) for notice in (ingest_notices or []) if isinstance(notice, dict)
    ]
    metadata = {
        "intent": "strategy_request",
        "kind": "clarification",
        "code": code,
    }
    if normalized_fields:
        metadata["fields"] = normalized_fields
    if normalized_notices:
        metadata["ingest_notices"] = normalized_notices
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=f"{message}{shared_lane._ingest_notice_text(normalized_notices)}",
        metadata=metadata,
    )
    response = {
        "task_id": task.id,
        "status": "clarification_required",
        "code": code,
        "messages": repo.list_agent_messages(task.id),
    }
    if normalized_fields:
        response["fields"] = normalized_fields
    if normalized_notices:
        response["ingest_notices"] = normalized_notices
    return response
