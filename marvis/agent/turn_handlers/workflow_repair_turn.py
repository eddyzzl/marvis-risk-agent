"""Turn integration for V2 workflow self-repair (B-line extension).

Two entry points:

- :func:`prepare_turn` runs on every persisted driver turn. It replaces a
  classified step failure's generic card with the deterministic diagnosis and,
  in Agent mode, hands back the ``auto`` option to run (within budget).
- :func:`maybe_handle_command` answers the exact command of an offered option,
  the undo command, or an expired command, before free-text recovery routing.

Spec: docs/superpowers/specs/2026-10-23-self-repair-extension-spec.md §5.2.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
import logging

from marvis.agent import workflow_repair as repair
from marvis.agent.driver_turn import DriverMessage, DriverTurn
from marvis.agent.validation_repair import repair_report_content, repair_undo_content
from marvis.agent.workflow_error_diagnostics import workflow_error_content


logger = logging.getLogger(__name__)

MAX_AUTO_REPAIR_DEPTH = 2
_depth: ContextVar[int] = ContextVar("workflow_auto_repair_depth", default=0)


@dataclass
class AutoRepair:
    message: DriverMessage
    option: dict
    context: repair.FailureContext


def _agent_mode(task) -> bool:
    return str(getattr(task, "run_mode", "") or "") == "agent"


def prepare_turn(runtime, repo, task, turn) -> tuple[object, AutoRepair | None]:
    """Diagnose a classified failure in ``turn``; maybe defer it to an auto repair."""

    if str(getattr(task, "task_type", "")) not in repair.V2_WORKFLOWS or not getattr(turn, "messages", None):
        return turn, None
    settings = getattr(runtime, "settings", None)
    plan_repo = getattr(runtime, "plan_repo", None)
    if settings is None or plan_repo is None:
        return turn, None
    messages = list(turn.messages)
    auto = None
    for index, message in enumerate(messages):
        metadata = dict(message.metadata or {})
        if not metadata.get("error") or not isinstance(metadata.get("failure_envelope"), dict):
            continue
        context = repair.failure_context(settings, task, plan_repo, metadata)
        if context is None:
            continue
        diagnostic = repair.diagnose_workflow_failure(context, agent_mode=_agent_mode(task))
        if diagnostic is None:
            continue
        history = repo.list_agent_messages(task.id)
        option = None
        if _agent_mode(task) and _depth.get() < MAX_AUTO_REPAIR_DEPTH:
            option = repair.plan_auto_repair(settings, task, diagnostic, history)
        if option is None and _agent_mode(task):
            diagnostic = repair.budget_note(settings, history, task, diagnostic)
        envelope = dict(metadata["failure_envelope"])
        envelope["retryable"] = bool(diagnostic.get("retryable", True)) and bool(envelope.get("retryable", True))
        metadata.update({"error_diagnostic": diagnostic, "failure_envelope": envelope})
        enriched = DriverMessage(message.stage, workflow_error_content(diagnostic), metadata)
        messages[index] = enriched
        if option is not None:
            auto = AutoRepair(enriched, option, context)
            messages.pop(index)
        break
    return replace(turn, messages=messages) if isinstance(turn, DriverTurn) else _with_messages(turn, messages), auto


def _with_messages(turn, messages):
    turn.messages = messages
    return turn


def _driver(runtime):
    # Same construction as shared._driver; this lane must not import shared
    # (shared calls into it), see tests/test_turn_handler_modules.py.
    from marvis.agent.plan_driver import PlanDriver

    return PlanDriver(
        runtime.plan_repo,
        runtime.plan_executor,
        planner=runtime.planner,
        validator=runtime.plan_validator,
        llm_client=runtime.llm_client,
        allow_manual_gate_adapters=runtime.allow_manual_gate_adapters,
        require_semantic_text_authorization=runtime.require_semantic_text_authorization,
        governance_service=runtime.governance_service,
        local_principal=runtime.local_principal,
        cancellation_check=runtime.cancellation_check,
    )


def _response(repo, task) -> dict:
    return {"task_id": task.id, "status": "ok", "messages": repo.list_agent_messages(task.id)}


def _add_report(repo, task, report: dict) -> None:
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=repair_report_content(report),
        metadata={"intent": "workflow_repair", "repair_report": report},
    )


def _add_card(repo, task, message: DriverMessage, diagnostic: dict) -> None:
    metadata = {**dict(message.metadata or {}), "error_diagnostic": diagnostic}
    repo.add_agent_message(
        task.id, role="assistant", stage="chat",
        content=workflow_error_content(diagnostic), metadata=metadata,
    )


def run_auto_repair(runtime, repo, task, auto: AutoRepair, *, append) -> None:
    """``append`` persists a follow-up driver turn (shared.append_driver_messages)."""

    diagnostic = auto.message.metadata["error_diagnostic"]
    token = _depth.set(_depth.get() + 1)
    try:
        try:
            outcome = repair.apply_workflow_repair(
                runtime, _driver(runtime), auto.context, auto.option,
                finding=str(diagnostic.get("summary") or ""), trigger="auto",
            )
        except Exception as exc:  # noqa: BLE001 - the original failure card must still appear
            logger.exception("workflow auto repair failed task_id=%s", task.id)
            if isinstance(exc, repair.RepairRolledBackError):
                diagnostic = repair.repair_card_without(
                    diagnostic, str(auto.option.get("id")),
                    {"label": "我试过", "value": _rolled_back_text(exc)},
                )
            _add_card(repo, task, auto.message, diagnostic)
            return
        # The deferred failure card was never shown; keep it with the report so
        # an undo can show it again.
        outcome.report["original_card"] = {
            "content": auto.message.content,
            "metadata": dict(auto.message.metadata or {}),
        }
        _add_report(repo, task, outcome.report)
        if outcome.report["verified"] and outcome.turn is not None:
            append(repo, task, outcome.turn, runtime=runtime)
            return
        _add_card(repo, task, auto.message, repair.repair_card_without(
            diagnostic, str(auto.option.get("id")),
            {"label": "我试过", "value": str(outcome.report["verification"])},
        ))
    finally:
        _depth.reset(token)


def maybe_handle_command(runtime, repo, task, *, user_text: str | None, append) -> dict | None:
    """Exact repair / undo commands for V2 workflows in Agent mode."""

    text = str(user_text or "").strip()
    if (
        not text
        or not _agent_mode(task)
        or str(getattr(task, "task_type", "")) not in repair.V2_WORKFLOWS
    ):
        return None
    settings, plan_repo = runtime.settings, runtime.plan_repo
    messages = repo.list_agent_messages(task.id)
    matched = repair.match_workflow_repair_command(settings, plan_repo, task.id, messages, text)
    if matched is None:
        return None
    kind = matched[0]
    repo.add_agent_message(
        task.id, role="user", stage="chat", content=text,
        metadata={"intent": "workflow_repair_undo" if kind == "undo" else "workflow_repair"},
    )

    def reply(content: str) -> dict:
        repo.add_agent_message(task.id, role="assistant", stage="chat", content=content,
                               metadata={"intent": "workflow_repair"})
        return {"task_id": task.id, "status": "message_saved", "messages": repo.list_agent_messages(task.id)}

    if kind == "stale":
        return reply(repair.UNDO_UNAVAILABLE_REPLY if matched[1] == "undo" else repair.STALE_COMMAND_REPLY)
    if kind == "undo":
        report = matched[1]
        try:
            undo = repair.undo_workflow_repair(settings, plan_repo, task, report)
        except Exception as exc:  # noqa: BLE001 - nothing changed; say so
            return reply(f"撤销没能执行，平台没有做任何改动：{str(exc)[:200]}")
        repo.add_agent_message(task.id, role="assistant", stage="chat",
                               content=repair_undo_content(undo),
                               metadata={"intent": "workflow_repair", "repair_undo": undo})
        source = report.get("original_card") or next(
            (item for item in messages if item.get("id") == report.get("source_message_id")), None
        )
        if source is None and report.get("option_id") != "use_task_environment":
            source = _original_card(messages, report)
        if source is not None and report.get("option_id") != "use_task_environment":
            metadata = source.get("metadata") or {}
            repo.add_agent_message(task.id, role="assistant", stage="chat",
                                   content=str(source.get("content") or ""), metadata=dict(metadata))
        return _response(repo, task)

    option, card = matched[1], matched[2]
    metadata = card.get("metadata") or {}
    context = repair.failure_context(settings, task, plan_repo, metadata)
    if context is None:
        return reply(repair.STALE_COMMAND_REPLY)
    diagnostic = metadata.get("error_diagnostic") or {}
    try:
        outcome = repair.apply_workflow_repair(
            runtime, _driver(runtime), context, option,
            finding=str(diagnostic.get("summary") or ""), trigger="user",
            source_message_id=str(card.get("id") or "") or None,
        )
    except repair.RepairRolledBackError as exc:
        logger.exception("workflow repair failed task_id=%s", task.id)
        return reply(_rolled_back_text(exc))
    except Exception as exc:  # noqa: BLE001 - raised before the first write
        logger.exception("workflow repair failed task_id=%s", task.id)
        return reply(f"这个办法没能执行，平台没有做任何改动：{str(exc)[:200]}")
    _add_report(repo, task, outcome.report)
    if outcome.report["verified"] and outcome.turn is not None:
        append(repo, task, outcome.turn, runtime=runtime)
    elif not outcome.report["verified"]:
        _add_card(repo, task, DriverMessage("chat", "", dict(metadata)), repair.repair_card_without(
            diagnostic, str(option.get("id")),
            {"label": "我试过", "value": str(outcome.report["verification"])},
        ))
    return _response(repo, task)


def _rolled_back_text(exc) -> str:
    """Only a complete rollback may claim nothing changed."""

    reason = str(exc)[:200]
    if not exc.restored:
        return f"这个办法执行中出错，而且没能完全还原：{'；'.join(exc.remaining)}。出错原因：{reason}"
    if exc.derived_dataset_id:
        return (f"这个办法没能执行，计划已全部还原，原样本没有改动；派生数据集 {exc.derived_dataset_id} "
                f"保留作审计证据，计划没有使用它。出错原因：{reason}")
    return f"这个办法没能执行，已全部还原，平台没有做任何改动：{reason}"


def _original_card(messages, report: dict) -> dict | None:
    """The failure card the repaired step last showed (auto repairs have no source id)."""

    for message in reversed(messages):
        metadata = message.get("metadata") or {}
        diagnostic = metadata.get("error_diagnostic")
        if (
            isinstance(diagnostic, dict)
            and diagnostic.get("diagnosed_by") == repair.DIAGNOSED_BY
            and diagnostic.get("step_id") == report.get("step_id")
        ):
            return message
    return None


__all__ = ["maybe_handle_command", "prepare_turn", "run_auto_repair"]
