"""Recovery for governed Agent turns."""

from __future__ import annotations

from marvis.agent.plan_driver import DriverError
from marvis.agent.workflow_recovery import WorkflowFailureContext
from marvis.agent.workflow_recovery import deterministic_workflow_recovery_reply
from marvis.agent.workflow_recovery import is_explicit_cancelled_workflow_resume
from marvis.agent.workflow_recovery import is_explicit_workflow_retry
from marvis.agent.workflow_recovery import is_workflow_repair_request
from marvis.agent.workflow_recovery import latest_unresolved_workflow_failure
from marvis.agent.workflow_recovery import parse_champion_refit_revision_intent
from marvis.agent.workflow_recovery import parse_tuning_budget_revision_intent
from marvis.agent.workflow_recovery import parse_workflow_rollback_intent
from marvis.domain import TASK_TYPE_FEATURE_ANALYSIS
from marvis.domain import TaskRecord
from marvis.orchestrator.contracts import PlanStatus
from marvis.orchestrator.contracts import StepStatus
from marvis.repositories.tasks import TaskRepository
import hashlib
import hmac
from . import c1_state as c1_state_lane
from . import contracts as contracts_lane
from . import feature as feature_lane
from . import responses as responses_lane
from . import shared as shared_lane


def _maybe_handle_workflow_recovery_turn(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None,
    dedup_strategies: dict | None,
    adjust_params: dict | None,
    expected_step_id: str | None,
    recovery_bypass: bool,
) -> dict | None:
    """Keep failed Agent tasks conversational until retry is explicit."""

    text = str(user_text or "").strip()
    if recovery_bypass or task.run_mode != "agent" or not text:
        return None
    # Structured UI actions are already explicit execution input. They must keep
    # their existing gate/setup path instead of being reclassified as chat.
    if any(
        value is not None
        for value in (selection, dedup_strategies, adjust_params, expected_step_id)
    ):
        return None
    conversation = repo.list_agent_messages(task.id)
    if shared_lane._active_plan(runtime.plan_repo, task.id) is not None:
        return None
    cancelled_recovery = _maybe_resume_cancelled_plan(
        runtime,
        repo,
        task,
        text=text,
        conversation=conversation,
    )
    if cancelled_recovery is not None:
        return cancelled_recovery
    rollback_intent = parse_workflow_rollback_intent(text)
    tuning_budget_intent = parse_tuning_budget_revision_intent(text)
    champion_refit_intent = parse_champion_refit_revision_intent(text)
    explicit_retry = is_explicit_workflow_retry(text)
    failure = latest_unresolved_workflow_failure(
        conversation,
        workflow=task.task_type,
    )
    legacy_restart_failure = _legacy_restart_failure_context(
        conversation,
        runtime.plan_repo,
        task_id=task.id,
        workflow=task.task_type,
    )
    # A failed plan is authoritative over a stale setup gate accidentally
    # appended after an old restart notice.  For all other utterances, retain
    # the normal gate routing (notably a bare “继续” must not execute a retry).
    if (
        shared_lane.latest_open_gate(conversation) is not None
        and legacy_restart_failure is None
    ):
        return None
    if failure is None:
        failure = legacy_restart_failure
    if failure is None:
        return None
    # A setup transaction can fail after a target choice was semantically
    # reviewed but before plan/target/receipt commit.  The prior target card is
    # still the authoritative open gate in that case.  Repeating an exact
    # candidate-bearing instruction must re-enter the C1 two-pass review,
    # rather than being swallowed as generic failure chat (which would make a
    # second authorization impossible without a magic recovery phrase).
    retryable = bool(failure.diagnostic.get("retryable", True))
    if failure.failure_envelope is not None:
        retryable = bool(failure.failure_envelope.get("retryable", retryable))
    expected_retry_hash = str(failure.diagnostic.get("retry_instruction_sha256") or "")
    current_retry_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    if (
        retryable
        and task.task_type == TASK_TYPE_FEATURE_ANALYSIS
        and failure.diagnostic.get("code") == "feature_target_plan_start_rolled_back"
        and bool(expected_retry_hash)
        and hmac.compare_digest(expected_retry_hash, current_retry_hash)
    ):
        feature_target_state = feature_lane._latest_feature_target_state(conversation)
        if (
            feature_target_state is not None
            and c1_state_lane._natural_language_c1_target(text, feature_target_state)
        ):
            return None
    repair_authorized = bool(
        failure.diagnostic.get("auto_recoverable")
    ) and is_workflow_repair_request(text)
    if retryable and champion_refit_intent is not None:
        envelope = failure.failure_envelope or {}
        failed_step_id = str(envelope.get("failed_step_id") or "")
        failed_plan = next(
            (
                plan
                for plan in reversed(runtime.plan_repo.list_plans_for_task(task.id))
                if getattr(plan.status, "value", plan.status) == PlanStatus.FAILED.value
                and any(step.id == failed_step_id for step in plan.steps)
            ),
            None,
        )
        failed_step = next(
            (
                step
                for step in (failed_plan.steps if failed_plan is not None else [])
                if step.id == failed_step_id
            ),
            None,
        )
        if (
            failed_plan is None
            or failed_step is None
            or failed_step.tool_ref.tool != "select_experiment"
        ):
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=(
                    "当前失败位置不是“选择实验”，为避免把重训设置改到错误步骤，"
                    "本次未修改计划。"
                ),
                metadata={
                    "intent": "workflow_recovery_champion_refit_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "reason": "failed_select_step_not_found",
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        input_updates: dict[str, object] = {"refit_on_train_plus_test": False}
        if champion_refit_intent.selected_experiment_id:
            input_updates["selected_experiment_id"] = (
                champion_refit_intent.selected_experiment_id
            )
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_recovery_champion_refit_revision",
                "recovery_of_message_id": failure.message_id,
                "plan_id": failed_plan.id,
                "step_id": failed_step_id,
                **input_updates,
            },
        )
        turn = shared_lane._driver(runtime).retry_failed_step(
            failed_plan.id,
            failed_step_id,
            run_seq=int(envelope.get("run_seq") or 0) + 1,
            inputs=input_updates,
            # Changing refit semantics requires a fresh governed confirmation;
            # never carry the previous decision over to the revised input hash.
            preserve_target_confirmation=False,
        )
        shared_lane.append_driver_messages(repo, task, turn, runtime=runtime)
        return responses_lane.join_turn_response(repo, task.id)
    if retryable and tuning_budget_intent is not None:
        envelope = failure.failure_envelope or {}
        failed_step_id = str(envelope.get("failed_step_id") or "")
        failed_plan = next(
            (
                plan
                for plan in reversed(runtime.plan_repo.list_plans_for_task(task.id))
                if getattr(plan.status, "value", plan.status) == PlanStatus.FAILED.value
                and any(step.id == failed_step_id for step in plan.steps)
            ),
            None,
        )
        requested_budgets = dict(tuning_budget_intent.n_trials_by_recipe)
        if failed_plan is None or not failed_step_id:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="没有找到与当前故障证据一致的失败计划；为避免修改错误计划，本次未调整调参预算。",
                metadata={
                    "intent": "workflow_recovery_tuning_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "reason": "failed_plan_not_found",
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        try:
            turn = shared_lane._driver(runtime).rollback_failed_plan_to_tuning_config(
                failed_plan.id,
                failed_step_id,
                default_n_trials=tuning_budget_intent.default_n_trials,
                n_trials_by_recipe=requested_budgets,
                run_seq=int(envelope.get("run_seq") or 0) + 1,
            )
        except DriverError as exc:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={
                    "intent": "workflow_recovery_tuning_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "plan_id": failed_plan.id,
                    "step_id": failed_step_id,
                    "default_n_trials": tuning_budget_intent.default_n_trials,
                    "n_trials_by_recipe": requested_budgets,
                },
            )
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"未修改当前计划：{exc}",
                metadata={
                    "intent": "workflow_recovery_tuning_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "plan_id": failed_plan.id,
                    "reason": str(exc),
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_recovery_tuning_revision",
                "recovery_of_message_id": failure.message_id,
                "plan_id": failed_plan.id,
                "step_id": failed_step_id,
                "root_step": "configure_tuning",
                "default_n_trials": tuning_budget_intent.default_n_trials,
                "n_trials_by_recipe": requested_budgets,
            },
        )
        shared_lane.append_driver_messages(repo, task, turn, runtime=runtime)
        return responses_lane.join_turn_response(repo, task.id)
    if retryable and rollback_intent is not None:
        envelope = failure.failure_envelope or {}
        failed_step_id = str(envelope.get("failed_step_id") or "")
        failed_plan = next(
            (
                plan
                for plan in reversed(runtime.plan_repo.list_plans_for_task(task.id))
                if getattr(plan.status, "value", plan.status) == PlanStatus.FAILED.value
                and any(step.id == failed_step_id for step in plan.steps)
            ),
            None,
        )
        if failed_plan is None or not failed_step_id:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="没有找到与当前故障证据一致的失败计划；为避免修改错误计划，本次未执行回退。",
                metadata={
                    "intent": "workflow_recovery_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "reason": "failed_plan_not_found",
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        try:
            turn = shared_lane._driver(runtime).rollback_failed_plan_to_feature_screen(
                failed_plan.id,
                failed_step_id,
                excluded_features=list(rollback_intent.excluded_features),
                run_seq=int(envelope.get("run_seq") or 0) + 1,
            )
        except DriverError as exc:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={
                    "intent": "workflow_recovery_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "plan_id": failed_plan.id,
                    "step_id": failed_step_id,
                    "root_step": rollback_intent.root_step,
                    "excluded_features": list(rollback_intent.excluded_features),
                },
            )
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"未修改当前计划：{exc}",
                metadata={
                    "intent": "workflow_recovery_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "plan_id": failed_plan.id,
                    "reason": str(exc),
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_recovery_revision",
                "recovery_of_message_id": failure.message_id,
                "plan_id": failed_plan.id,
                "step_id": failed_step_id,
                "root_step": rollback_intent.root_step,
                "excluded_features": list(rollback_intent.excluded_features),
            },
        )
        shared_lane.append_driver_messages(repo, task, turn, runtime=runtime)
        return responses_lane.join_turn_response(repo, task.id)
    if retryable and (explicit_retry or repair_authorized):
        envelope = failure.failure_envelope or {}
        failed_step_id = str(envelope.get("failed_step_id") or "")
        failed_plan = next(
            (
                plan
                for plan in reversed(runtime.plan_repo.list_plans_for_task(task.id))
                if getattr(plan.status, "value", plan.status) == PlanStatus.FAILED.value
                and any(step.id == failed_step_id for step in plan.steps)
            ),
            None,
        )
        if failed_plan is None or not failed_step_id:
            return None
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_recovery_retry",
                "recovery_of_message_id": failure.message_id,
                "plan_id": failed_plan.id,
                "step_id": failed_step_id,
            },
        )
        turn = shared_lane._driver(runtime).retry_failed_step(
            failed_plan.id,
            failed_step_id,
            run_seq=int(envelope.get("run_seq") or 0) + 1,
            # This branch is reached only from an explicit human Agent-mode
            # recovery command.  Reuse the confirmation only if this exact
            # failed step had already been confirmed; the repository keeps an
            # unconfirmed governed gate unconfirmed and clears all downstream
            # confirmations.
            preserve_target_confirmation=True,
        )
        shared_lane.append_driver_messages(repo, task, turn, runtime=runtime)
        return responses_lane.join_turn_response(repo, task.id)

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "workflow_recovery_chat",
            "recovery_of_message_id": failure.message_id,
        },
    )
    if runtime.recovery_responder is None:
        content = deterministic_workflow_recovery_reply(failure.diagnostic)
        response_metadata = {
            "fallback": True,
            "fallback_reason": "recovery_responder_unavailable",
        }
    else:
        content, response_metadata = runtime.recovery_responder(
            task=task,
            user_message=text,
            diagnostic=failure.diagnostic,
        )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=content,
        metadata={
            "intent": "workflow_recovery_chat",
            "recovery_of_message_id": failure.message_id,
            "recovery_code": failure.diagnostic.get("code"),
            **dict(response_metadata or {}),
        },
    )
    return {
        "task_id": task.id,
        "status": "message_saved",
        "messages": repo.list_agent_messages(task.id),
    }


def _maybe_resume_cancelled_plan(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    text: str,
    conversation: list[dict],
) -> dict | None:
    """Keep a stopped plan recoverable without rebuilding its setup.

    Cancellation is terminal for the executor invocation, but not destructive:
    its current step is persisted as failed and prior outputs/runs remain.  A
    fresh Agent turn can therefore reopen that exact step with a fresh
    cancellation token.  Non-command chat never starts a new plan implicitly.
    """

    plans = runtime.plan_repo.list_plans_for_task(task.id)
    if not plans:
        return None
    plan = plans[-1]
    if getattr(plan.status, "value", plan.status) != PlanStatus.CANCELLED.value:
        return None
    cancelled_message = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant"
            and (message.get("metadata") or {}).get("intent") == "execution_cancelled"
            and str((message.get("metadata") or {}).get("plan_id") or plan.id)
            == plan.id
        ),
        None,
    )
    metadata = (cancelled_message or {}).get("metadata") or {}
    step_id = str(metadata.get("step_id") or "")
    interrupted = next(
        (
            step
            for step in plan.steps
            if step.status == StepStatus.FAILED and (not step_id or step.id == step_id)
        ),
        None,
    )
    if interrupted is None:
        return None

    recovery_of_message_id = (cancelled_message or {}).get("id")
    if is_explicit_cancelled_workflow_resume(text):
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_cancelled_resume",
                "recovery_of_message_id": recovery_of_message_id,
                "plan_id": plan.id,
                "step_id": interrupted.id,
            },
        )
        try:
            turn = shared_lane._driver(runtime).retry_failed_step(
                plan.id,
                interrupted.id,
                run_seq=int(metadata.get("run_seq") or 0) + 1,
                preserve_target_confirmation=True,
            )
        except DriverError as exc:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"当前停止点未能恢复：{exc}",
                metadata={
                    "intent": "workflow_cancelled_resume_rejected",
                    "recovery_of_message_id": recovery_of_message_id,
                    "plan_id": plan.id,
                    "step_id": interrupted.id,
                    "reason": str(exc),
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        shared_lane.append_driver_messages(repo, task, turn, runtime=runtime)
        return responses_lane.join_turn_response(repo, task.id)

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "workflow_cancelled_chat",
            "recovery_of_message_id": recovery_of_message_id,
            "plan_id": plan.id,
            "step_id": interrupted.id,
        },
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"当前计划仍停在“{interrupted.title}”，本轮没有执行新步骤。"
            "已完成结果和检查点都在；要从这里恢复，请明确输入“继续当前步骤”。"
        ),
        metadata={
            "intent": "workflow_cancelled_chat",
            "recovery_of_message_id": recovery_of_message_id,
            "plan_id": plan.id,
            "step_id": interrupted.id,
        },
    )
    return {
        "task_id": task.id,
        "status": "message_saved",
        "messages": repo.list_agent_messages(task.id),
    }


def _legacy_restart_failure_context(
    messages: list[dict],
    plan_repo,
    *,
    task_id: str,
    workflow: str,
) -> WorkflowFailureContext | None:
    """Recover the target encoded by pre-envelope restart notifications.

    Older startup recovery notices only persisted ``plan_id``.  Because a
    plan-scoped assistant message is normally a progress boundary, those
    notices intentionally cannot be treated as a generic unresolved failure.
    It accepts only the latest task plan with exactly one failed step.  The
    caller still requires an explicit retry command before executing; ordinary
    conversation remains in recovery chat instead of falling back to setup.
    """

    plans = plan_repo.list_plans_for_task(task_id)
    if not plans:
        return None
    latest_plan = plans[-1]
    if (
        getattr(latest_plan.status, "value", latest_plan.status)
        != PlanStatus.FAILED.value
    ):
        return None
    failed_steps = [
        step
        for step in latest_plan.steps
        if getattr(step.status, "value", step.status) == "failed"
    ]
    if len(failed_steps) != 1:
        return None
    failed_step = failed_steps[0]

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        metadata = message.get("metadata") or {}
        if not metadata.get("plan_interrupted_by_restart"):
            continue
        if metadata.get("plan_resumed_at_confirmation"):
            continue
        if str(metadata.get("plan_id") or "") != latest_plan.id:
            continue
        summary = (
            f"服务重启中断了“{failed_step.title}”步骤，已完成步骤和中间产物均已保留。"
        )
        return WorkflowFailureContext(
            message_id=str(message.get("id") or "") or None,
            diagnostic={
                "schema_version": "workflow_error.v1",
                "workflow": workflow,
                "code": "server_restart_interrupted",
                "phase": "execution",
                "title": "工作流执行中断",
                "summary": summary,
                "cause": "MARVIS 服务在该步骤执行期间重启；源材料没有被修改。",
                "location": failed_step.title,
                "evidence": [
                    {"label": "计划", "value": latest_plan.id},
                    {"label": "失败步骤", "value": failed_step.id},
                ],
                "actions": ["回复“重试当前步骤”，从该失败步骤继续。"],
                "retryable": True,
                "auto_recoverable": True,
                "impact": "失败步骤之后的依赖步骤尚未执行。",
                "exception_type": "ServerRestart",
                "technical_detail": str(failed_step.error or "ServerRestart"),
            },
            failure_envelope={
                "schema_version": "failure.v1",
                "failed_step_id": failed_step.id,
                "error_kind": "ServerRestart",
                "message": summary,
                "retryable": True,
                "editable_input_schema": {},
                "suggested_actions": ["retry"],
                "downstream_reset": "dependent_steps",
                "downstream_reset_steps": [],
            },
        )
    return None
