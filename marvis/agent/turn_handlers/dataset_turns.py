"""dataset_turns driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from marvis.agent.dataset_analysis import build_dataset_analysis_request, detect_dataset_analysis_intent
from marvis.agent.dataset_export import build_dataset_export_request, detect_dataset_export_intent
from marvis.agent.dataset_transform import build_dataset_transform_request, detect_dataset_transform_intent
from marvis.agent.plan_driver import DriverError, is_confirm
from marvis.agent.risk_analysis_setup import latest_risk_analysis_intake
from marvis.data.transform_semantics import effective_transform_semantic_mapping
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TASK_TYPE_VINTAGE, TaskRecord
from marvis.repositories.data_workspace import DataWorkspaceDataError, DataWorkspaceDatasetNotFound, DataWorkspaceRepository

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _active_plan
    from . import _driver
    from . import _modeling_data_runtime
    from . import _resume_new_routed_plan
    from . import append_driver_messages
    from . import append_join_error
    from . import join_turn_response
    from . import latest_open_gate

_DATASET_TRANSFORM_PROTECTED_DROP_META_KEY = "pending_protected_drop"

def _maybe_handle_dataset_transform_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    force_intent: bool = False,
) -> dict | None:
    """Compile a natural-language data change into the closed transform AST."""

    text = str(user_text or "").strip()
    if not text:
        return None
    conversation = repo.list_agent_messages(task.id)
    # Risk-analysis material confirmations often describe scope by saying that
    # no extra filtering or other row processing is required.  Those phrases
    # match the generic transform detector, but while typed risk intake is
    # still open they are contract details, not a request to mutate a bound
    # DataWorkspace sample.
    if task.task_type == TASK_TYPE_VINTAGE:
        risk_intake = latest_risk_analysis_intake(conversation)
        if risk_intake is not None and risk_intake.get("phase") != "ready":
            return None
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    pending = _latest_pending_transform_protected_drop(conversation)
    confirming_pending = pending is not None and is_confirm(text)
    if (
        not confirming_pending
        and not force_intent
        and not detect_dataset_transform_intent(text)
    ):
        return None

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "dataset_transform",
            **({"confirmation": "protected_drop"} if confirming_pending else {}),
        },
    )
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, DataWorkspaceDatasetNotFound, KeyError) as exc:
        return _dataset_transform_clarification(
            repo,
            task.id,
            code="workspace_unavailable",
            message=f"数据工作区当前不可用：{exc}",
        )
    if snapshot.active_dataset_id is None:
        return _dataset_transform_clarification(
            repo,
            task.id,
            code="active_dataset_required",
            message="请先在数据工作区选择并保存本次要加工的样本。",
        )

    backend, registry = _modeling_data_runtime(runtime.settings)
    try:
        dataset = registry.get(snapshot.active_dataset_id)
        if dataset.task_id != task.id:
            raise PermissionError("active dataset does not belong to this task")
        path = registry.resolve_verified_path(dataset.id)
        columns = tuple(backend.column_names(path))
    except Exception as exc:  # noqa: BLE001 - converted to a typed chat boundary
        return _dataset_transform_clarification(
            repo,
            task.id,
            code="dataset_unavailable",
            message=f"当前活动样本无法安全读取：{exc}",
        )

    semantic_hash = data_semantic_mapping_hash(snapshot.semantic_mapping)
    effective_semantics = effective_transform_semantic_mapping(
        dataset,
        snapshot.semantic_mapping,
        source_columns=columns,
    )
    if confirming_pending:
        if not _pending_transform_matches_workspace(
            pending,
            dataset_id=dataset.id,
            content_hash=dataset.content_hash,
            analysis_generation=snapshot.analysis_generation,
            semantic_mapping_hash=semantic_hash,
        ):
            return _dataset_transform_clarification(
                repo,
                task.id,
                code="protected_drop_source_changed",
                message=(
                    "待确认期间活动数据或字段语义已经变化。请重新说明要删除的字段，"
                    "我会基于当前版本重新确认。"
                ),
            )
        pending_operations = pending.get("operations")
        pending_protected = pending.get("protected_fields")
        if (
            not isinstance(pending_operations, list)
            or not pending_operations
            or not all(isinstance(item, dict) for item in pending_operations)
            or not isinstance(pending_protected, list)
            or not pending_protected
            or not all(isinstance(item, str) for item in pending_protected)
        ):
            return _dataset_transform_clarification(
                repo,
                task.id,
                code="protected_drop_state_invalid",
                message="待确认的删列请求不完整，请重新说明要删除的字段。",
            )
        operations = list(pending_operations)
        confirm_protected_drop = True
    else:
        parsed = build_dataset_transform_request(
            text,
            columns=columns,
            business_names=effective_semantics.business_names,
            semantic_mapping=effective_semantics,
        )
        if parsed.request is None:
            extra_metadata: dict = {}
            if parsed.operations and parsed.protected_fields:
                extra_metadata[_DATASET_TRANSFORM_PROTECTED_DROP_META_KEY] = {
                    "request_text": text,
                    "operations": [dict(item) for item in parsed.operations],
                    "protected_fields": list(parsed.protected_fields),
                    "dataset_id": dataset.id,
                    "dataset_content_hash": dataset.content_hash,
                    "analysis_generation": snapshot.analysis_generation,
                    "semantic_mapping_hash": semantic_hash,
                }
            return _dataset_transform_clarification(
                repo,
                task.id,
                code="transform_request_clarification",
                message=parsed.clarification or "请说明要如何加工当前数据。",
                extra_metadata=extra_metadata,
            )
        request = parsed.request
        operations = list(request.operations)
        confirm_protected_drop = request.confirm_protected_drop

    slots = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
        "operations": operations,
        "confirm_protected_drop": confirm_protected_drop,
    }
    driver = _driver(runtime)
    try:
        started = driver.start(
            task_id=task.id,
            template_id="dataset_transform",
            slots=slots,
            tier=runtime.tier,
        )
        # A normal transform is reversible by selecting the immutable parent;
        # protected drops reached this point only after the explicit dialogue
        # acknowledgement above, so no second generic gate is needed.
        turn = _resume_new_routed_plan(
            runtime,
            driver,
            plan_id=started.plan_id,
            semantic_reason=text,
        )
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"数据加工出错：{exc}")
    append_driver_messages(repo, task, turn, runtime=runtime)
    return join_turn_response(repo, task.id)

def _latest_pending_transform_protected_drop(
    conversation: list[dict],
) -> dict | None:
    last_assistant = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant"
        ),
        None,
    )
    if last_assistant is None:
        return None
    pending = (last_assistant.get("metadata") or {}).get(
        _DATASET_TRANSFORM_PROTECTED_DROP_META_KEY
    )
    return pending if isinstance(pending, dict) else None

def _pending_transform_matches_workspace(
    pending: dict,
    *,
    dataset_id: str,
    content_hash: str,
    analysis_generation: int,
    semantic_mapping_hash: str,
) -> bool:
    request_text = pending.get("request_text")
    return (
        isinstance(request_text, str)
        and bool(request_text.strip())
        and pending.get("dataset_id") == dataset_id
        and pending.get("dataset_content_hash") == content_hash
        and pending.get("analysis_generation") == analysis_generation
        and pending.get("semantic_mapping_hash") == semantic_mapping_hash
    )

def _dataset_transform_clarification(
    repo: TaskRepository,
    task_id: str,
    *,
    code: str,
    message: str,
    extra_metadata: dict | None = None,
) -> dict:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "dataset_transform",
            "kind": "clarification",
            "code": code,
            **dict(extra_metadata or {}),
        },
    )
    return join_turn_response(repo, task_id)

def _maybe_handle_dataset_export_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    force_intent: bool = False,
) -> dict | None:
    """Run a bound CSV/XLSX export when the dataset object is explicit."""

    if not force_intent and not detect_dataset_export_intent(user_text):
        return None
    conversation = repo.list_agent_messages(task.id)
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=user_text or "",
        metadata={"intent": "dataset_export"},
    )
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, DataWorkspaceDatasetNotFound, KeyError) as exc:
        return _dataset_export_clarification(
            repo,
            task.id,
            code="workspace_unavailable",
            message=f"数据工作区当前不可用：{exc}",
        )
    if snapshot.active_dataset_id is None:
        return _dataset_export_clarification(
            repo,
            task.id,
            code="active_dataset_required",
            message="请先在数据工作区选择并保存本次要导出的样本。",
        )

    backend, registry = _modeling_data_runtime(runtime.settings)
    try:
        dataset = registry.get(snapshot.active_dataset_id)
        if dataset.task_id != task.id:
            raise PermissionError("active dataset does not belong to this task")
        path = registry.resolve_verified_path(dataset.id)
        columns = tuple(backend.column_names(path))
    except Exception as exc:  # noqa: BLE001 - converted to a typed chat boundary
        return _dataset_export_clarification(
            repo,
            task.id,
            code="dataset_unavailable",
            message=f"当前活动样本无法安全读取：{exc}",
        )

    parsed = build_dataset_export_request(
        user_text or "",
        columns=columns,
        business_names=snapshot.semantic_mapping.business_names,
    )
    if parsed.request is None:
        return _dataset_export_clarification(
            repo,
            task.id,
            code="export_request_clarification",
            message=parsed.clarification or "请选择 CSV 或 Excel 导出格式。",
        )
    request = parsed.request
    slots = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(snapshot.semantic_mapping),
        "format": request.format,
        "text_columns": list(request.text_columns),
    }
    driver = _driver(runtime)
    try:
        started = driver.start(
            task_id=task.id,
            template_id="dataset_export",
            slots=slots,
            tier=runtime.tier,
        )
        turn = _resume_new_routed_plan(
            runtime,
            driver,
            plan_id=started.plan_id,
            semantic_reason=user_text or "",
        )
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"数据导出出错：{exc}")
    append_driver_messages(repo, task, turn, runtime=runtime)
    return join_turn_response(repo, task.id)

def _dataset_export_clarification(
    repo: TaskRepository,
    task_id: str,
    *,
    code: str,
    message: str,
) -> dict:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "dataset_export",
            "kind": "clarification",
            "code": code,
        },
    )
    return join_turn_response(repo, task_id)

def _maybe_handle_dataset_analysis_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    force_intent: bool = False,
) -> dict | None:
    """Run the bound descriptive-analysis Workflow for an explicit request."""

    if not force_intent and not detect_dataset_analysis_intent(user_text):
        return None
    conversation = repo.list_agent_messages(task.id)
    # Risk-analysis material confirmations naturally mention phrases such as
    # “无缺失” and “字段分布”.  While that typed intake is still open, those
    # words describe its input contract rather than a new generic
    # profile_dataset request.  Let the risk state machine consume the turn;
    # after it reaches ready, explicit dataset diagnostics remain available.
    if task.task_type == TASK_TYPE_VINTAGE:
        risk_intake = latest_risk_analysis_intake(conversation)
        if risk_intake is not None and risk_intake.get("phase") != "ready":
            return None
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=user_text or "",
        metadata={"intent": "dataset_analysis"},
    )
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, DataWorkspaceDatasetNotFound, KeyError) as exc:
        return _dataset_analysis_clarification(
            repo,
            task.id,
            code="workspace_unavailable",
            message=f"数据工作区当前不可用：{exc}",
        )
    if snapshot.active_dataset_id is None:
        return _dataset_analysis_clarification(
            repo,
            task.id,
            code="active_dataset_required",
            message="请先在数据工作区选择并保存本次要分析的样本，再让我开始分析。",
        )

    backend, registry = _modeling_data_runtime(runtime.settings)
    try:
        dataset = registry.get(snapshot.active_dataset_id)
        if dataset.task_id != task.id:
            raise PermissionError("active dataset does not belong to this task")
        path = registry.resolve_verified_path(dataset.id)
        columns = tuple(backend.column_names(path))
    except Exception as exc:  # noqa: BLE001 - converted to a typed user-facing boundary
        return _dataset_analysis_clarification(
            repo,
            task.id,
            code="dataset_unavailable",
            message=f"当前活动样本无法安全读取：{exc}",
        )

    parsed = build_dataset_analysis_request(
        user_text or "",
        columns=columns,
        target_col=snapshot.semantic_mapping.target_col,
        business_names=snapshot.semantic_mapping.business_names,
    )
    if parsed.request is None:
        return _dataset_analysis_clarification(
            repo,
            task.id,
            code="analysis_request_clarification",
            message=parsed.clarification or "请说明要分析哪些数据内容。",
        )

    request = parsed.request
    slots: dict = {
        "dataset_id": dataset.id,
        "expected_content_hash": snapshot.active_dataset_content_hash,
        "workspace_revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(snapshot.semantic_mapping),
        "sections": list(request.sections),
    }
    if request.columns is not None:
        slots["columns"] = list(request.columns)
    if request.target_col is not None:
        slots["target_col"] = request.target_col

    driver = _driver(runtime)
    try:
        started = driver.start(
            task_id=task.id,
            template_id="dataset_descriptive_analysis",
            slots=slots,
            tier=runtime.tier,
        )
        turn = _resume_new_routed_plan(
            runtime,
            driver,
            plan_id=started.plan_id,
            semantic_reason=user_text or "",
        )
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"样本描述分析出错：{exc}")
    append_driver_messages(repo, task, turn, runtime=runtime)
    return join_turn_response(repo, task.id)

def _dataset_analysis_clarification(
    repo: TaskRepository,
    task_id: str,
    *,
    code: str,
    message: str,
) -> dict:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "dataset_analysis",
            "kind": "clarification",
            "code": code,
        },
    )
    return join_turn_response(repo, task_id)
