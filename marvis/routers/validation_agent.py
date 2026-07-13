from fastapi import APIRouter, BackgroundTasks, Request
from marvis.errors import unprocessable

from marvis.agent.service import (
    agent_rerun_stage,
    answer_chat_message,
    is_agent_advance_intent,
    is_agent_material_reselection_intent,
    is_agent_report_revision_intent,
    is_continue_validation_intent,
    is_stop_validation_intent,
    summarize_stage,
)
from marvis.agent.turn_handlers import DRIVER_AGENT_TASK_TYPES
from marvis.agent.validation_app_service import (
    WIRED_AGENT_TASK_TYPES,
    add_and_stream_agent_message,
    agent_chat_evidence,
    agent_evidence,
    agent_memory_context,
    audit_agent_memory_use,
    capture_user_preference_memory,
    confirm_agent_report_conclusions,
    dispatch_agent_validation_job,
    dispatch_driver_turn,
    generate_word_conclusions,
    handle_agent_stop_message,
    is_agent_report_confirm_intent,
    is_agent_report_regenerate_intent,
    latest_pending_agent_report_draft,
    model_metadata,
    repo as agent_repo,
    require_agent_task,
    require_wired_agent_task_type,
    resolve_agent_model,
    resolve_driver_agent_client,
)
from marvis.agent.validation_messages import format_conclusion_values
from marvis.agent.validation_service import (
    require_agent_rerun_stage_reached,
    reset_agent_task_for_rerun,
)
from marvis.api_schemas import (
    AgentMessageRequest,
    AgentModelRequest,
    AgentReportDraftConfirmRequest,
)
from marvis.api_task_helpers import get_task_or_404, reject_if_task_has_active_job


router = APIRouter(prefix="/api", tags=["validation-agent"])
REPORT_DIRECTIVE_LIMIT = 6
REPORT_DIRECTIVE_CHARS = 1_600


def _report_revision_instruction_context(
    conversation: list[dict],
    current_instruction: str,
) -> tuple[str, int]:
    prior: list[str] = []
    for message in conversation:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        metadata = message.get("metadata") or {}
        is_word_rerun = (
            metadata.get("target_stage") == "word_conclusion_draft"
            and metadata.get("intent")
            in {"rerun_stage", "regenerate_report_draft"}
        )
        if not content or not (
            is_word_rerun or is_agent_report_revision_intent(content)
        ):
            continue
        prior.append(content[:REPORT_DIRECTIVE_CHARS])
    prior = prior[-(REPORT_DIRECTIVE_LIMIT - 1) :]
    if not prior:
        return current_instruction, 1
    directives = [*prior, current_instruction[:REPORT_DIRECTIVE_CHARS]]
    lines = "\n".join(
        f"{index}. {directive}" for index, directive in enumerate(directives, 1)
    )
    return (
        "以下是本任务按时间顺序记录的报告修改指令；后出现的指令覆盖与其冲突的旧指令，"
        "其余要求继续有效：\n"
        f"{lines}",
        len(directives),
    )


@router.get("/tasks/{task_id}/agent/messages")
def get_agent_messages(
    task_id: str,
    request: Request,
    after_id: str | None = None,
    limit: int | None = None,
) -> dict:
    repo = agent_repo(request)
    get_task_or_404(repo, task_id)
    bounded_limit = None if limit is None else max(1, min(int(limit), 500))
    cursor_found = bool(after_id and repo.has_agent_message(task_id, after_id))
    query_limit = bounded_limit + 1 if bounded_limit is not None else None
    messages = repo.list_agent_messages(task_id, after_id=after_id, limit=query_limit)
    has_more = False
    if bounded_limit is not None and len(messages) > bounded_limit:
        has_more = True
        messages = messages[:bounded_limit]
    return {
        "messages": messages,
        "incremental": cursor_found,
        "has_more": has_more,
        "limit": bounded_limit,
    }


@router.post("/tasks/{task_id}/agent/start", status_code=202)
def start_agent_task(
    task_id: str,
    payload: AgentModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = agent_repo(request)
    task = get_task_or_404(repo, task_id)
    require_agent_task(task, DRIVER_AGENT_TASK_TYPES)
    require_wired_agent_task_type(task, WIRED_AGENT_TASK_TYPES)
    if task.task_type in DRIVER_AGENT_TASK_TYPES:
        agent_client = resolve_driver_agent_client(request, task, payload)
        return dispatch_driver_turn(
            request,
            repo,
            task,
            user_text=None,
            agent_client=agent_client,
            acceptance_mode=payload.acceptance_mode,
        )
    model_profile = resolve_agent_model(request, payload.model_id, payload.effort)
    return dispatch_agent_validation_job(
        repo_=repo,
        task=task,
        settings=request.app.state.settings,
        model_profile=model_profile,
        acceptance_mode=payload.acceptance_mode,
        background_tasks=background_tasks,
    )


@router.post("/tasks/{task_id}/agent/messages", status_code=202)
def post_agent_message(
    task_id: str,
    payload: AgentMessageRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = agent_repo(request)
    task = get_task_or_404(repo, task_id)
    require_agent_task(task, DRIVER_AGENT_TASK_TYPES)
    require_wired_agent_task_type(task, WIRED_AGENT_TASK_TYPES)
    content = payload.content.strip()
    if not content:
        raise unprocessable("message content is required")
    if is_stop_validation_intent(content):
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={"intent": "stop"},
        )
        capture_user_preference_memory(request, task_id, user_message)
        return handle_agent_stop_message(repo, task)
    if task.task_type in DRIVER_AGENT_TASK_TYPES:
        agent_client = resolve_driver_agent_client(request, task, payload)
        return dispatch_driver_turn(
            request,
            repo,
            task,
            user_text=content,
            agent_client=agent_client,
            acceptance_mode=payload.acceptance_mode,
            selection=payload.selection,
            dedup_strategies=payload.dedup_strategies,
            adjust_params=payload.adjust_params,
            expected_step_id=payload.expected_step_id,
        )
    if is_agent_material_reselection_intent(content):
        reject_if_task_has_active_job(repo, task_id)
        ui_action = {"type": "select_validation_materials"}
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={"intent": "select_materials"},
        )
        capture_user_preference_memory(request, task_id, user_message)
        repo.add_agent_message(
            task_id,
            role="assistant",
            stage="chat",
            content=(
                "请在随后弹出的“重新选择验证材料”窗口中重新绑定 Notebook、样本数据、"
                "PMML 和数据字典；保存后再重新扫描材料。"
            ),
            metadata={"intent": "select_materials", "ui_action": ui_action},
        )
        return {
            "task_id": task_id,
            "status": "awaiting_material_selection",
            "ui_action": ui_action,
            "messages": repo.list_agent_messages(task_id),
        }
    conversation = repo.list_agent_messages(task_id)
    pending_report_draft = latest_pending_agent_report_draft(conversation)
    if is_agent_report_confirm_intent(content) and pending_report_draft:
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={"intent": "confirm_report"},
        )
        capture_user_preference_memory(request, task_id, user_message)
        return confirm_agent_report_conclusions(
            repo_=repo,
            task=task,
            task_id=task_id,
            settings=request.app.state.settings,
            text_values=pending_report_draft["values"],
            expected_revision=pending_report_draft["report_revision"],
            background_tasks=background_tasks,
            hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        )
    model_profile = resolve_agent_model(request, payload.model_id, payload.effort)
    rerun_stage = agent_rerun_stage(content)
    if rerun_stage:
        reject_if_task_has_active_job(repo, task_id)
        require_agent_rerun_stage_reached(task, rerun_stage)
        continue_after_rerun = (
            rerun_stage in {"scan", "reproducibility", "metrics"}
            and is_continue_validation_intent(content)
        )
        rerun_intent = (
            "regenerate_report_draft"
            if rerun_stage == "word_conclusion_draft"
            and is_agent_report_regenerate_intent(content)
            else "rerun_stage"
        )
        stage_instruction = content
        directive_count = 0
        if rerun_stage == "word_conclusion_draft":
            stage_instruction, directive_count = _report_revision_instruction_context(
                conversation,
                content,
            )
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={
                **model_metadata(model_profile),
                "intent": rerun_intent,
                "target_stage": rerun_stage,
                "continue_after_rerun": continue_after_rerun,
                **(
                    {"active_report_directive_count": directive_count}
                    if directive_count
                    else {}
                ),
            },
        )
        capture_user_preference_memory(request, task_id, user_message)
        task = reset_agent_task_for_rerun(repo, task_id, rerun_stage)
        return dispatch_agent_validation_job(
            repo_=repo,
            task=task,
            settings=request.app.state.settings,
            model_profile=model_profile,
            acceptance_mode=(
                "auto_accept" if continue_after_rerun else payload.acceptance_mode
            ),
            background_tasks=background_tasks,
            forced_stage=rerun_stage,
            stage_instruction=stage_instruction,
        )
    if is_agent_report_regenerate_intent(content) and pending_report_draft:
        reject_if_task_has_active_job(repo, task_id)
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={**model_metadata(model_profile), "intent": "regenerate_report_draft"},
        )
        capture_user_preference_memory(request, task_id, user_message)
        return dispatch_agent_validation_job(
            repo_=repo,
            task=task,
            settings=request.app.state.settings,
            model_profile=model_profile,
            acceptance_mode=payload.acceptance_mode,
            background_tasks=background_tasks,
        )
    if not is_agent_advance_intent(content):
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata=model_metadata(model_profile),
        )
        capture_user_preference_memory(request, task_id, user_message)
        conversation = repo.list_agent_messages(task_id)
        evidence = agent_chat_evidence(request, repo, task, conversation)
        memory_context = agent_memory_context(
            request,
            task,
            stage="chat",
            user_message=content,
            evidence=evidence,
        )
        message = add_and_stream_agent_message(
            repo,
            task_id,
            stage="chat",
            model_profile=model_profile,
            producer=lambda on_delta: answer_chat_message(
                task=task,
                user_message=content,
                conversation=conversation,
                evidence=evidence,
                memory_context=memory_context,
                model_profile=model_profile,
                on_delta=on_delta,
            ),
        )
        audit_agent_memory_use(request, message, task_id=task_id)
        return {
            "task_id": task_id,
            "status": "message_saved",
            "messages": repo.list_agent_messages(task_id),
        }
    user_message = repo.add_agent_message(
        task_id,
        role="user",
        stage="chat",
        content=content,
        metadata={**model_metadata(model_profile), "intent": "advance"},
    )
    capture_user_preference_memory(request, task_id, user_message)
    return dispatch_agent_validation_job(
        repo_=repo,
        task=task,
        settings=request.app.state.settings,
        model_profile=model_profile,
        acceptance_mode=payload.acceptance_mode,
        background_tasks=background_tasks,
    )


@router.post("/tasks/{task_id}/agent/stop", status_code=202)
def stop_agent_action(task_id: str, request: Request) -> dict:
    repo = agent_repo(request)
    task = get_task_or_404(repo, task_id)
    require_agent_task(task, DRIVER_AGENT_TASK_TYPES)
    return handle_agent_stop_message(repo, task)


@router.post("/tasks/{task_id}/agent/summarize")
def summarize_agent_task(
    task_id: str,
    payload: AgentModelRequest,
    request: Request,
) -> dict:
    repo = agent_repo(request)
    task = get_task_or_404(repo, task_id)
    require_agent_task(task, DRIVER_AGENT_TASK_TYPES)
    model_profile = resolve_agent_model(request, payload.model_id, payload.effort)
    evidence = agent_evidence(request, task_id)
    memory_context = agent_memory_context(
        request,
        task,
        stage="metrics",
        evidence=evidence,
    )
    content, metadata = summarize_stage(
        task=task,
        stage="metrics",
        evidence=evidence,
        memory_context=memory_context,
        model_profile=model_profile,
        fallback="已读取当前验证证据。请结合分数一致性、效果稳定性和压力测试明细复核模型表现。",
    )
    metadata.update(model_metadata(model_profile))
    message = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="summary",
        content=content,
        metadata=metadata,
    )
    audit_agent_memory_use(request, message, task_id=task_id)
    return {"message": message, "messages": repo.list_agent_messages(task_id)}


@router.post("/tasks/{task_id}/agent/report-draft")
def draft_agent_report_conclusions(
    task_id: str,
    payload: AgentModelRequest,
    request: Request,
) -> dict:
    repo = agent_repo(request)
    task = get_task_or_404(repo, task_id)
    require_agent_task(task, DRIVER_AGENT_TASK_TYPES)
    model_profile = resolve_agent_model(request, payload.model_id, payload.effort)
    evidence = agent_evidence(request, task_id)
    memory_context = agent_memory_context(
        request,
        task,
        stage="word_conclusion_draft",
        evidence=evidence,
    )
    values, metadata = generate_word_conclusions(
        task=task,
        evidence=evidence,
        memory_context=memory_context,
        model_profile=model_profile,
    )
    metadata.update(model_metadata(model_profile))
    _, report_revision = repo.get_report_values(task_id)
    message = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content=format_conclusion_values(values),
        metadata={**metadata, "draft_values": values, "report_revision": report_revision},
    )
    audit_agent_memory_use(request, message, task_id=task_id)
    return {
        "message": message,
        "text_values": values,
        "messages": repo.list_agent_messages(task_id),
    }


@router.post("/tasks/{task_id}/agent/report-draft/confirm", status_code=202)
def confirm_agent_report_conclusions_route(
    task_id: str,
    payload: AgentReportDraftConfirmRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = agent_repo(request)
    task = get_task_or_404(repo, task_id)
    require_agent_task(task, DRIVER_AGENT_TASK_TYPES)
    return confirm_agent_report_conclusions(
        repo_=repo,
        task=task,
        task_id=task_id,
        settings=request.app.state.settings,
        text_values=payload.text_values,
        expected_revision=payload.revision,
        background_tasks=background_tasks,
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
    )
