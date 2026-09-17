"""adhoc driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping
import hmac
from marvis.agent.adhoc_analysis import build_slice_spec_from_utterance, detect_question_intent
from marvis.agent.plan_driver import DriverError, is_confirm
from marvis.agent.semantic_intent import INTENT_ADHOC_CONFIRM, INTENT_ADHOC_REJECT, INTENT_ADHOC_REVISE
from marvis.agent.risk_analysis_setup import latest_risk_analysis_intake
from marvis.data.errors import DatasetContentDriftError
from marvis.data.registry import AuthenticatedDatasetBinding
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TASK_TYPE_VINTAGE, TaskRecord

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _active_plan
    from . import _append_context_free_driver_messages
    from . import _driver
    from . import _modeling_data_runtime
    from . import _resume_new_routed_plan
    from . import append_join_error
    from . import join_turn_response
    from . import latest_open_gate

_ADHOC_SPEC_META_KEY = "adhoc_spec"

_ADHOC_DATA_ROLES = frozenset({"sample", "feature", "strategy_sample", "derived"})

def _maybe_handle_adhoc_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    force_intent: bool = False,
    pending_decision: str | None = None,
) -> dict | None:
    """Return a turn response when this turn is an ad-hoc 问数 interaction, else
    None so the caller falls through to the normal type dispatch."""
    conversation = repo.list_agent_messages(task.id)
    # Risk analysis owns the conversation while its ask-goal/material contract
    # is open. A perfectly natural answer such as “可以做收益测算吗？” contains a
    # question mark, but is an intake selection rather than an ad-hoc slice
    # query. Let the vintage/risk setup state machine consume it.
    if task.task_type == TASK_TYPE_VINTAGE:
        risk_intake = latest_risk_analysis_intake(conversation)
        if risk_intake is not None and risk_intake.get("phase") != "ready":
            return None
    pending = _latest_adhoc_pending(conversation)
    if pending is not None:
        # Round B: Agent free text is classified twice before reaching this
        # branch. Manual-mode compatibility keeps the legacy exact-confirm path.
        decision = pending_decision
        if decision is None and is_confirm(user_text or ""):
            decision = INTENT_ADHOC_CONFIRM
        if decision == INTENT_ADHOC_CONFIRM:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=user_text or "",
                metadata={
                    "intent": "adhoc_query",
                    "semantic_decision": INTENT_ADHOC_CONFIRM,
                },
            )
            resolved = _resolve_adhoc_dataset(runtime.settings, task.id)
            if (
                resolved is None
                or not _adhoc_pending_matches_binding(pending, resolved[0])
                or _active_plan(runtime.plan_repo, task.id) is not None
                or latest_open_gate(repo.list_agent_messages(task.id)) is not None
            ):
                repo.add_agent_message(
                    task.id,
                    role="assistant",
                    stage="chat",
                    content=(
                        "待确认期间数据集或任务状态已变化，旧问数口径已作废，"
                        "且没有执行工具。请基于当前数据重新提出问题。"
                    ),
                    metadata={
                        "intent": "adhoc_query",
                        "kind": "clarification",
                        "code": "adhoc_pending_stale",
                    },
                )
                return join_turn_response(repo, task.id)
            return _run_adhoc_slice_plan(
                runtime,
                repo,
                task,
                pending,
                confirmation_reason=user_text or "",
            )
        if decision == INTENT_ADHOC_REJECT:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=user_text or "",
                metadata={
                    "intent": "adhoc_query",
                    "semantic_decision": INTENT_ADHOC_REJECT,
                },
            )
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="已取消这份问数口径，没有创建计划或执行聚合工具。",
                metadata={
                    "intent": "adhoc_query",
                    "kind": "cancelled",
                    "code": "adhoc_pending_cancelled",
                },
            )
            return join_turn_response(repo, task.id)
        if decision == INTENT_ADHOC_REVISE:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=user_text or "",
                metadata={
                    "intent": "adhoc_query",
                    "semantic_decision": INTENT_ADHOC_REVISE,
                },
            )
            resolved = _resolve_adhoc_dataset(runtime.settings, task.id)
            if resolved is None or not _adhoc_pending_matches_binding(
                pending,
                resolved[0],
            ):
                repo.add_agent_message(
                    task.id,
                    role="assistant",
                    stage="chat",
                    content=(
                        "待确认期间数据集已变化，旧问数口径已作废。"
                        "请基于当前数据重新提出完整问题。"
                    ),
                    metadata={
                        "intent": "adhoc_query",
                        "kind": "clarification",
                        "code": "adhoc_pending_stale",
                    },
                )
                return join_turn_response(repo, task.id)
            binding, columns = resolved
            revised = build_slice_spec_from_utterance(
                user_text or "",
                columns,
                runtime.llm_client,
                caller="adhoc_analysis_revision",
            )
            if revised.needs_clarification:
                repo.add_agent_message(
                    task.id,
                    role="assistant",
                    stage="chat",
                    content=(
                        f"{revised.clarify or '请重新说明修改后的完整问数口径。'}"
                        " 原口径仍未执行并继续等待你的决定。"
                    ),
                    metadata={
                        "intent": "adhoc_query",
                        "kind": "clarification",
                        "code": "adhoc_revision_clarification",
                        _ADHOC_SPEC_META_KEY: dict(pending),
                    },
                )
                return join_turn_response(repo, task.id)
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=revised.confirmation_text or "",
                metadata={
                    _ADHOC_SPEC_META_KEY: _adhoc_tool_inputs(revised.spec, binding)
                },
            )
            return join_turn_response(repo, task.id)
        return None
    # Round A: no pending spec. Enter only when the guards all hold — conservative
    # by design (窄不触发优于劫持).
    if not force_intent and not detect_question_intent(user_text):
        return None
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None
    resolved = _resolve_adhoc_dataset(runtime.settings, task.id)
    if resolved is None:
        return None
    binding, columns = resolved
    result = build_slice_spec_from_utterance(
        user_text or "", columns, runtime.llm_client
    )
    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=user_text or "",
        metadata={"intent": "adhoc_query"},
    )
    if result.needs_clarification:
        # A Chinese clarification (never a guess, INV-1). No pending state is
        # stored — the user simply rephrases and round A runs again.
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=result.clarify or "没能理解这个问题，请换一种说法。",
            metadata={"intent": "adhoc_query"},
        )
        return join_turn_response(repo, task.id)
    # A validated spec: show the 口径确认门 and stash the exact tool inputs on it.
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=result.confirmation_text or "",
        metadata={_ADHOC_SPEC_META_KEY: _adhoc_tool_inputs(result.spec, binding)},
    )
    return join_turn_response(repo, task.id)

def _run_adhoc_slice_plan(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    tool_inputs: dict,
    *,
    confirmation_reason: str = "确认",
) -> dict:
    """Build + run the single-step slice_aggregate plan for a confirmed 口径.

    vintage's lightweight single-step entry is the precedent: one non-gated step
    that runs straight to DONE and renders its own table. Because the 口径 was just
    confirmed turn-side, the plan-overview 开始 gate is auto-confirmed here so the
    aggregate runs in the same turn instead of pausing again."""
    driver = _driver(runtime)
    try:
        start = driver.start(
            task_id=task.id,
            template_id="slice_aggregate",
            slots=dict(tool_inputs),
            tier=runtime.tier,
        )
        turn = _resume_new_routed_plan(
            runtime,
            driver,
            plan_id=start.plan_id,
            semantic_reason=confirmation_reason,
        )
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"即席问数出错：{exc}")
    _append_context_free_driver_messages(repo, task.id, turn)
    return join_turn_response(repo, task.id)

def _latest_adhoc_pending(conversation: list[dict]) -> dict | None:
    """The pending ad-hoc tool inputs, only when the LAST assistant message is the
    口径确认门 (mirrors latest_open_gate's last-assistant anchoring). Once the
    aggregate result/error is appended, this stops matching, so a confirmed spec is
    never re-run."""
    last_assistant = next(
        (m for m in reversed(conversation) if m.get("role") == "assistant"), None
    )
    if last_assistant is None:
        return None
    spec = (last_assistant.get("metadata") or {}).get(_ADHOC_SPEC_META_KEY)
    return spec if isinstance(spec, dict) else None

def _adhoc_tool_inputs(spec, binding: AuthenticatedDatasetBinding) -> dict:
    inputs = spec.tool_inputs(binding.dataset_id)
    inputs["expected_content_hash"] = binding.content_hash
    return inputs

def _adhoc_pending_matches_binding(
    pending: Mapping[str, object],
    binding: AuthenticatedDatasetBinding,
) -> bool:
    pending_dataset_id = str(pending.get("dataset_id") or "")
    pending_content_hash = str(pending.get("expected_content_hash") or "")
    return (
        pending_dataset_id == binding.dataset_id
        and bool(pending_content_hash)
        and hmac.compare_digest(pending_content_hash, binding.content_hash)
    )

def _resolve_adhoc_dataset(
    settings,
    task_id: str,
) -> tuple[AuthenticatedDatasetBinding, list[str]] | None:
    """A task's ready dataset id + its column whitelist, or None when the task has
    no already-registered dataset (guard (a) — this branch never scans/ingests
    from source_dir; that is the setup flow's job). Prefers a target-carrying
    dataset, else the largest — same ranking feature/vintage setup use."""
    _backend, registry = _modeling_data_runtime(settings)
    datasets = [
        d for d in registry.list_for_task(task_id) if d.role in _ADHOC_DATA_ROLES
    ]
    if not datasets:
        return None
    dataset = sorted(
        datasets,
        key=lambda d: (
            not bool(getattr(d, "has_target", False)),
            -int(getattr(d, "row_count", 0) or 0),
        ),
    )[0]
    try:
        binding = registry.authenticate_dataset_binding(
            dataset.id,
            expected_task_id=task_id,
            expected_content_hash=str(dataset.content_hash or ""),
        )
        columns = list(registry.authenticated_binding_column_names(binding))
    except (DatasetContentDriftError, KeyError, OSError, ValueError):
        return None
    if not columns:
        return None
    return binding, columns

def _has_adhoc_dataset(settings, task_id: str) -> bool:
    """Cheap, non-mutating routing hint; execution performs full authentication."""

    _backend, registry = _modeling_data_runtime(settings)
    try:
        return any(
            dataset.role in _ADHOC_DATA_ROLES
            for dataset in registry.list_for_task(task_id)
        )
    except Exception:  # noqa: BLE001 - routing hints fail closed
        return False
