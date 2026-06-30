from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from marvis.api_schemas import (
    AgentMessageRequest,
    AgentModelRequest,
    AgentReportDraftConfirmRequest,
)


router = APIRouter(prefix="/api", tags=["validation-agent"])


def _agent_api():
    # Keep the HTTP adapter thin while the validation-agent implementation is
    # still exposed from marvis.api for compatibility with existing tests and
    # extension imports.
    from marvis import api as legacy_api

    return legacy_api


@router.get("/tasks/{task_id}/agent/messages")
def get_agent_messages(
    task_id: str,
    request: Request,
    after_id: str | None = None,
    limit: int | None = None,
) -> dict:
    api = _agent_api()
    repo = api._repo(request)
    api._get_task_or_404(repo, task_id)
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
    api = _agent_api()
    repo = api._repo(request)
    task = api._get_task_or_404(repo, task_id)
    api._require_agent_task(task)
    api._require_wired_agent_task_type(task)
    if task.task_type in api._DRIVER_AGENT_TASK_TYPES:
        agent_client = api._resolve_driver_agent_client(request, task, payload)
        return api._dispatch_driver_turn(
            request,
            repo,
            task,
            user_text=None,
            agent_client=agent_client,
            acceptance_mode=payload.acceptance_mode,
        )
    model_profile = api._resolve_agent_model(request, payload.model_id, payload.effort)
    return api._dispatch_agent_validation_job(
        repo=repo,
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
    api = _agent_api()
    repo = api._repo(request)
    task = api._get_task_or_404(repo, task_id)
    api._require_agent_task(task)
    api._require_wired_agent_task_type(task)
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="message content is required")
    if api.is_stop_validation_intent(content):
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={"intent": "stop"},
        )
        api._capture_user_preference_memory(request, task_id, user_message)
        return api._handle_agent_stop_message(repo, task)
    if task.task_type in api._DRIVER_AGENT_TASK_TYPES:
        agent_client = api._resolve_driver_agent_client(request, task, payload)
        return api._dispatch_driver_turn(
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
    conversation = repo.list_agent_messages(task_id)
    pending_report_draft = api._latest_pending_agent_report_draft(conversation)
    if api._is_agent_report_confirm_intent(content) and pending_report_draft:
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={"intent": "confirm_report"},
        )
        api._capture_user_preference_memory(request, task_id, user_message)
        return api._confirm_agent_report_conclusions(
            repo=repo,
            task=task,
            task_id=task_id,
            settings=request.app.state.settings,
            text_values=pending_report_draft["values"],
            expected_revision=pending_report_draft["report_revision"],
            background_tasks=background_tasks,
            hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        )
    model_profile = api._resolve_agent_model(request, payload.model_id, payload.effort)
    rerun_stage = api.agent_rerun_stage(content)
    if rerun_stage:
        api._reject_if_task_has_active_job(repo, task_id)
        api._require_agent_rerun_stage_reached(task, rerun_stage)
        rerun_intent = (
            "regenerate_report_draft"
            if rerun_stage == "word_conclusion_draft"
            and api._is_agent_report_regenerate_intent(content)
            else "rerun_stage"
        )
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={
                **api._model_metadata(model_profile),
                "intent": rerun_intent,
                "target_stage": rerun_stage,
            },
        )
        api._capture_user_preference_memory(request, task_id, user_message)
        task = api._reset_agent_task_for_rerun(repo, task_id, rerun_stage)
        return api._dispatch_agent_validation_job(
            repo=repo,
            task=task,
            settings=request.app.state.settings,
            model_profile=model_profile,
            acceptance_mode=payload.acceptance_mode,
            background_tasks=background_tasks,
            forced_stage=rerun_stage,
            stage_instruction=content,
        )
    if api._is_agent_report_regenerate_intent(content) and pending_report_draft:
        api._reject_if_task_has_active_job(repo, task_id)
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={**api._model_metadata(model_profile), "intent": "regenerate_report_draft"},
        )
        api._capture_user_preference_memory(request, task_id, user_message)
        return api._dispatch_agent_validation_job(
            repo=repo,
            task=task,
            settings=request.app.state.settings,
            model_profile=model_profile,
            acceptance_mode=payload.acceptance_mode,
            background_tasks=background_tasks,
        )
    if not api.is_agent_advance_intent(content):
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata=api._model_metadata(model_profile),
        )
        api._capture_user_preference_memory(request, task_id, user_message)
        conversation = repo.list_agent_messages(task_id)
        evidence = api._agent_chat_evidence(request, repo, task, conversation)
        memory_context = api._agent_memory_context(
            request,
            task,
            stage="chat",
            user_message=content,
            evidence=evidence,
        )
        message = api._add_and_stream_agent_message(
            repo,
            task_id,
            stage="chat",
            model_profile=model_profile,
            producer=lambda on_delta: api.answer_chat_message(
                task=task,
                user_message=content,
                conversation=conversation,
                evidence=evidence,
                memory_context=memory_context,
                model_profile=model_profile,
                on_delta=on_delta,
            ),
        )
        api._audit_agent_memory_use(request, message, task_id=task_id)
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
        metadata={**api._model_metadata(model_profile), "intent": "advance"},
    )
    api._capture_user_preference_memory(request, task_id, user_message)
    return api._dispatch_agent_validation_job(
        repo=repo,
        task=task,
        settings=request.app.state.settings,
        model_profile=model_profile,
        acceptance_mode=payload.acceptance_mode,
        background_tasks=background_tasks,
    )


@router.post("/tasks/{task_id}/agent/stop", status_code=202)
def stop_agent_action(task_id: str, request: Request) -> dict:
    api = _agent_api()
    repo = api._repo(request)
    task = api._get_task_or_404(repo, task_id)
    api._require_agent_task(task)
    return api._handle_agent_stop_message(repo, task)


@router.post("/tasks/{task_id}/agent/summarize")
def summarize_agent_task(
    task_id: str,
    payload: AgentModelRequest,
    request: Request,
) -> dict:
    api = _agent_api()
    repo = api._repo(request)
    task = api._get_task_or_404(repo, task_id)
    api._require_agent_task(task)
    model_profile = api._resolve_agent_model(request, payload.model_id, payload.effort)
    evidence = api._agent_evidence(request, task_id)
    memory_context = api._agent_memory_context(
        request,
        task,
        stage="metrics",
        evidence=evidence,
    )
    content, metadata = api.summarize_stage(
        task=task,
        stage="metrics",
        evidence=evidence,
        memory_context=memory_context,
        model_profile=model_profile,
        fallback="已读取当前验证证据。请结合分数一致性、效果稳定性和压力测试明细复核模型表现。",
    )
    metadata.update(api._model_metadata(model_profile))
    message = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="summary",
        content=content,
        metadata=metadata,
    )
    api._audit_agent_memory_use(request, message, task_id=task_id)
    return {"message": message, "messages": repo.list_agent_messages(task_id)}


@router.post("/tasks/{task_id}/agent/report-draft")
def draft_agent_report_conclusions(
    task_id: str,
    payload: AgentModelRequest,
    request: Request,
) -> dict:
    api = _agent_api()
    repo = api._repo(request)
    task = api._get_task_or_404(repo, task_id)
    api._require_agent_task(task)
    model_profile = api._resolve_agent_model(request, payload.model_id, payload.effort)
    evidence = api._agent_evidence(request, task_id)
    memory_context = api._agent_memory_context(
        request,
        task,
        stage="word_conclusion_draft",
        evidence=evidence,
    )
    values, metadata = api.generate_word_conclusions(
        task=task,
        evidence=evidence,
        memory_context=memory_context,
        model_profile=model_profile,
    )
    metadata.update(api._model_metadata(model_profile))
    _, report_revision = repo.get_report_values(task_id)
    message = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content=api._format_conclusion_values(values),
        metadata={**metadata, "draft_values": values, "report_revision": report_revision},
    )
    api._audit_agent_memory_use(request, message, task_id=task_id)
    return {
        "message": message,
        "text_values": values,
        "messages": repo.list_agent_messages(task_id),
    }


@router.post("/tasks/{task_id}/agent/report-draft/confirm", status_code=202)
def confirm_agent_report_conclusions(
    task_id: str,
    payload: AgentReportDraftConfirmRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    api = _agent_api()
    repo = api._repo(request)
    task = api._get_task_or_404(repo, task_id)
    api._require_agent_task(task)
    return api._confirm_agent_report_conclusions(
        repo=repo,
        task=task,
        task_id=task_id,
        settings=request.app.state.settings,
        text_values=payload.text_values,
        expected_revision=payload.revision,
        background_tasks=background_tasks,
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
    )
