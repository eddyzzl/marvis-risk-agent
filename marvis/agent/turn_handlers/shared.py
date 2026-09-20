"""Shared for governed Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from marvis.agent.memory_bridge import build_workflow_memory_context
from marvis.agent.memory_bridge import capture_agent_memory_for_driver_done
from marvis.agent.plan_driver import PlanDriver
from marvis.agent.workflow_error_diagnostics import build_workflow_error_diagnostic
from marvis.agent.workflow_error_diagnostics import failure_envelope_for_diagnostic
from marvis.agent.workflow_error_diagnostics import workflow_error_content
from marvis.agent.workflow_insights import build_workflow_insight_context
from marvis.agent.workflow_insights import render_workflow_insight
from marvis.agent_memory.api_support import audit_agent_memory_use_from_store
from marvis.agent_memory.store import AgentMemoryStore
from marvis.domain import TaskRecord
from marvis.memory_policy import load_memory_policy
from marvis.orchestrator.capability import auto_gate_budget
from marvis.orchestrator.capability import resolve_tier
from marvis.orchestrator.contracts import plan_fingerprint
from marvis.repositories.tasks import TaskRepository
from . import contracts as contracts_lane


def append_driver_messages(
    repo: TaskRepository,
    task: TaskRecord,
    turn,
    *,
    runtime: contracts_lane.DriverTurnRuntime,
) -> None:
    """Persist driver output with the complete governed runtime context."""

    _append_driver_messages_core(
        repo,
        task.id,
        turn,
        settings=runtime.settings,
        task=task,
        llm_client=getattr(runtime, "llm_client", None),
        hook_dispatcher=getattr(runtime, "hook_dispatcher", None),
    )


def _append_context_free_driver_messages(
    repo: TaskRepository,
    task_id: str,
    turn,
) -> None:
    """Persist intentionally context-free output such as an ad-hoc slice."""

    _append_driver_messages_core(repo, task_id, turn)


def _append_driver_messages_core(
    repo: TaskRepository,
    task_id: str,
    turn,
    *,
    settings=None,
    task: TaskRecord | None = None,
    llm_client=None,
    hook_dispatcher=None,
) -> None:
    for message in turn.messages:
        metadata = dict(message.metadata)
        content = message.content
        memory_context = None
        if (
            settings is not None
            and task is not None
            and message.stage in {"gate", "done"}
        ):
            memory_context = build_workflow_memory_context(settings, task)
            if memory_context is not None:
                metadata["memory_references"] = memory_context["references"]
                metadata["memory_context"] = {
                    "category": memory_context["category"],
                    "count": len(memory_context["memories"]),
                    "lines": memory_context["lines"],
                }
                content += "\n\n**本次参考的历史记忆**\n" + "\n".join(
                    f"- {line}" for line in memory_context["lines"]
                )
        insight_context = None
        if task is not None and message.stage in {"gate", "done"}:
            insight_context = build_workflow_insight_context(
                task.task_type,
                stage=message.stage,
                metadata=metadata,
                content=content,
            )
        if insight_context is not None:
            insight = render_workflow_insight(
                insight_context,
                client=llm_client,
                memory_context=(memory_context or {}).get("memories") or [],
            )
            content += f"\n\n{insight['content']}"
            metadata["agent_insight"] = {
                key: value for key, value in insight.items() if key != "content"
            }
        receipts: list[dict] = []
        if settings is not None and task is not None and message.stage == "done":
            receipts = capture_agent_memory_for_driver_done(
                settings,
                task,
                done_message_content=content,
                done_message_metadata=metadata,
                hook_dispatcher=hook_dispatcher,
            )
            metadata["memory_capture"] = {
                "enabled": load_memory_policy(settings.workspace).auto_distill,
                "saved": receipts,
            }
            if receipts:
                content += "\n\n**本次记忆沉淀**\n" + "\n".join(
                    f"- 已保存 `{item['memory_type']}`：{item['summary']}"
                    for item in receipts
                )
                content += "\n- 已触发记忆归并，后续同类任务可引用。"
            elif not load_memory_policy(settings.workspace).auto_distill:
                content += "\n\n**本次记忆沉淀**\n- 自动沉淀已在设置中关闭，本次未保存跨任务记忆。"
        stored_message = repo.add_agent_message(
            task_id,
            role="assistant",
            stage="chat",
            content=content,
            metadata=metadata,
        )
        if memory_context is not None and settings is not None and task is not None:
            try:
                audit_agent_memory_use_from_store(
                    AgentMemoryStore(settings.db_path),
                    stored_message,
                    task_id=task.id,
                )
            except Exception:
                pass


def append_workflow_error(
    repo: TaskRepository,
    task: TaskRecord,
    spec: contracts_lane._TurnHandlerSpec,
    exc: Exception,
    *,
    setup_error: bool = False,
    diagnostic_overrides: Mapping[str, object] | None = None,
) -> dict:
    diagnostic = build_workflow_error_diagnostic(
        workflow=spec.intent,
        exc=exc,
        task=task,
        setup_error=setup_error,
    )
    if diagnostic_overrides:
        diagnostic.update(dict(diagnostic_overrides))
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=workflow_error_content(diagnostic),
        metadata={
            "error": True,
            "intent": spec.intent,
            "error_diagnostic": diagnostic,
            "failure_envelope": failure_envelope_for_diagnostic(diagnostic),
        },
    )
    return {
        "task_id": task.id,
        "status": "error",
        "messages": repo.list_agent_messages(task.id),
    }


def latest_open_gate(messages: list[dict]) -> dict | None:
    last_assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"), None
    )
    if last_assistant is None:
        return None
    meta = last_assistant.get("metadata") or {}
    if meta.get("error") or meta.get("join_skip"):
        return None
    if meta.get("kind") in ("gate", "plan_overview") or "join_c1" in meta:
        return last_assistant
    return None


def _driver(runtime: contracts_lane.DriverTurnRuntime) -> PlanDriver:
    return PlanDriver(
        runtime.plan_repo,
        runtime.plan_executor,
        planner=runtime.planner,
        validator=runtime.plan_validator,
        llm_client=runtime.llm_client,
        allow_manual_gate_adapters=runtime.allow_manual_gate_adapters,
        require_semantic_text_authorization=(
            runtime.require_semantic_text_authorization
        ),
        governance_service=runtime.governance_service,
        local_principal=runtime.local_principal,
        cancellation_check=runtime.cancellation_check,
    )


def _resume_new_routed_plan(
    runtime: contracts_lane.DriverTurnRuntime,
    driver: PlanDriver,
    *,
    plan_id: str,
    semantic_reason: str,
):
    """Confirm an auto-runnable plan against the post-start immutable snapshot."""

    kwargs: dict[str, object] = {
        "plan_id": plan_id,
        "user_text": "确认",
    }
    if runtime.require_semantic_text_authorization and runtime.semantic_intent:
        plan = runtime.plan_repo.load_plan(plan_id)
        kwargs.update(
            {
                "_confirmation_reason": semantic_reason,
                "_expected_plan_revision": int(plan.replan_count),
                "_expected_plan_status": plan.status.value,
                "_expected_plan_fingerprint": plan_fingerprint(plan),
            }
        )
    return driver.resume(**kwargs)


def _active_plan(plan_repo, task_id: str):
    for plan in reversed(plan_repo.list_plans_for_task(task_id)):
        status = getattr(plan.status, "value", plan.status)
        if status not in contracts_lane._TERMINAL_PLAN_STATUS_VALUES:
            return plan
    return None


def _auto_gate_budget(runtime: contracts_lane.DriverTurnRuntime, task_id: str) -> int:
    """AGT-7: size the AUTO auto-drive loop's per-turn gate budget off the active
    plan's own gate count (needs_confirmation steps + the plan-overview gate),
    capped by the task's capability tier — instead of the fixed AGENT_MAX_GATES=8
    that silently exhausted on any plan with >=9 gates (the modeling_with_join
    template alone has 7 needs_confirmation steps plus the overview + C1 gates).
    Falls back to AGENT_MAX_GATES when no plan has been built yet (e.g. before the
    first C1 file-role gate) or the plan repo is unavailable, while still honoring
    the selected tier's hard ceiling."""
    tier = resolve_tier(getattr(runtime, "tier", None))
    plan_repo = getattr(runtime, "plan_repo", None)
    plan = _active_plan(plan_repo, task_id) if plan_repo is not None else None
    if plan is None:
        return min(contracts_lane.AGENT_MAX_GATES, tier.max_auto_gates)
    gate_count = sum(1 for step in plan.steps if step.needs_confirmation)
    # +1 for the plan-overview gate every driver plan pauses at before running.
    return auto_gate_budget(tier, gate_count + 1)


def _ingest_notice_text(notices: list[dict]) -> str:
    messages = [str(item.get("message") or "").strip() for item in notices]
    messages = [message for message in messages if message]
    if not messages:
        return ""
    return "\n\n已自动处理：\n" + "\n".join(f"- {message}" for message in messages)


def _merge_ingest_notices(*groups) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for notice in group or []:
            key = (str(notice.get("code") or ""), str(notice.get("file") or ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(notice))
    return merged
