"""join driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
import json
import sqlite3
from typing import Callable
from marvis.agent.join_setup import C1TargetValidationError, JoinSetupError, authenticate_join_selection, build_join_proposal
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_AUTO, CONFIRMATION_SOURCE_HUMAN
from marvis.data.errors import DatasetContentDriftError
from marvis.data.registry import AuthenticatedDatasetBinding, DatasetRegistry
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft, data_semantic_mapping_from_dict
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskRecord
from marvis.repositories.data_workspace import DataWorkspaceDataError, DataWorkspaceDatasetNotFound, DataWorkspaceRepository, DataWorkspaceRevisionConflict

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _TurnHandlerSpec
    from . import _append_c1_message
    from . import _append_successful_ui_action_messages
    from . import _c1_display_text
    from . import _c1_expected_content_hashes
    from . import _c1_semantic_snapshot_matches
    from . import _c1_snapshot
    from . import _c1_state_from_proposal
    from . import _c1_table
    from . import _has_c1_semantic_authorization
    from . import _latest_c1_state
    from . import _modeling_data_runtime
    from . import _parse_c1_reply
    from . import _record_c1_semantic_authorization
    from . import _run_driver_turn
    from . import _validated_authenticated_c1_target

def run_join_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    expected_plan_id: str | None = None,
    expected_plan_status: str | None = None,
    expected_plan_revision: int | None = None,
    expected_plan_fingerprint: str | None = None,
    expected_step_fingerprint: str | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
    ui_action: str | None = None,
) -> dict:
    return _run_driver_turn(
        _JOIN_SPEC,
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        expected_plan_id=expected_plan_id,
        expected_plan_status=expected_plan_status,
        expected_plan_revision=expected_plan_revision,
        expected_plan_fingerprint=expected_plan_fingerprint,
        expected_step_fingerprint=expected_step_fingerprint,
        confirmation_source=confirmation_source,
        ui_action=ui_action,
    )

def _run_join_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict | tuple:
    conversation = repo.list_agent_messages(task.id)
    c1_state = _latest_c1_state(conversation)
    _, registry = _modeling_data_runtime(runtime.settings)
    proposal = build_join_proposal(registry, task.id, task.source_dir)
    fresh_c1_state = _c1_state_from_proposal(proposal)
    c1_was_already_applied = any(
        bool((message.get("metadata") or {}).get("join_skip"))
        for message in conversation
    )
    if c1_state is None or _c1_snapshot(c1_state) != _c1_snapshot(fresh_c1_state):
        if c1_state is not None and c1_was_already_applied:
            raise JoinSetupError(
                "文件角色、数据内容或 DataWorkspace 语义映射在上次确认后已变化，"
                "拒绝重复确认；请新建任务或恢复一致的语义绑定。"
            )
        _append_c1_message(repo, task.id, proposal)
        return join_turn_response(repo, task.id)
    assignment = _parse_c1_reply(
        user_text,
        c1_state,
        llm_client=runtime.llm_client,
        trusted_ui_action=(
            runtime.ui_action == "confirm_roles"
            or confirmation_source == CONFIRMATION_SOURCE_AUTO
        ),
        require_semantic_authorization=(
            runtime.require_semantic_text_authorization
        ),
    )
    if assignment is None:
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content="请确认文件角色与目标列:无误就回复「确认」，或用下方控件调整后点「确认角色」。",
            metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
        )
        return join_turn_response(repo, task.id)
    if _has_c1_semantic_authorization(assignment):
        current_proposal = build_join_proposal(registry, task.id, task.source_dir)
        current_c1_state = _c1_state_from_proposal(current_proposal)
        if not _c1_semantic_snapshot_matches(assignment, current_c1_state):
            _append_c1_message(repo, task.id, current_proposal)
            return join_turn_response(repo, task.id)
    if not assignment["anchor_id"]:
        return append_join_error(
            repo, task.id, "请先指定样本锚表（通常是含目标列的那张），再确认。"
        )
    authenticated_selection = authenticate_join_selection(
        registry,
        task.id,
        anchor_id=assignment["anchor_id"],
        feature_ids=assignment["feature_ids"],
        expected_content_hashes=_c1_expected_content_hashes(c1_state),
    )
    try:
        assignment["target_col"] = _validated_authenticated_c1_target(
            registry,
            authenticated_selection,
            assignment.get("target_col"),
        )
    except C1TargetValidationError as exc:
        if runtime.ui_action == "confirm_roles":
            raise
        return append_join_error(repo, task.id, str(exc))
    if not assignment["feature_ids"]:
        def persist_single_table_confirmation(conn: sqlite3.Connection) -> None:
            registry.persist_authenticated_target_on_connection(
                conn,
                authenticated_selection.anchor,
                assignment.get("target_col"),
            )
            repo.update_target_col_on_connection(
                conn,
                task.id,
                assignment.get("target_col"),
            )
            _append_successful_ui_action_messages(
                _JOIN_SPEC,
                repo,
                task,
                user_text=user_text,
                ui_action=runtime.ui_action,
                expected_plan_id=None,
                expected_step_id=None,
                conn=conn,
            )
            _record_c1_semantic_authorization(
                repo,
                task.id,
                assignment,
                conn=conn,
            )
            repo.add_agent_message_on_connection(
                conn,
                task.id,
                role="assistant",
                stage="chat",
                content=(
                    "已确认样本表与目标列。只有一张表，无需拼接"
                    "（数据拼接阶段已跳过）。"
                ),
                metadata={"join_skip": True},
            )

        _bind_single_join_dataset(
            runtime,
            task,
            registry=registry,
            dataset_id=assignment["anchor_id"],
            target_col=assignment.get("target_col"),
            authenticated_binding=authenticated_selection.anchor,
            on_connection=persist_single_table_confirmation,
            confirmation_already_persisted=c1_was_already_applied,
        )
        return join_turn_response(repo, task.id)
    start_kwargs = {
        "_post_start_c1_target_binding": (
            registry,
            authenticated_selection.anchor,
            assignment.get("target_col"),
        ),
        **(
            {"_post_start_c1_assignment": assignment}
            if _has_c1_semantic_authorization(assignment)
            else {}
        ),
    }
    return (
        "data_join",
        {
            "anchor_id": assignment["anchor_id"],
            "feature_ids": assignment["feature_ids"],
        },
        start_kwargs,
    )

def _bind_single_join_dataset(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    *,
    registry: DatasetRegistry,
    dataset_id: str,
    target_col: str | None,
    authenticated_binding: AuthenticatedDatasetBinding,
    on_connection: Callable[[sqlite3.Connection], None] | None = None,
    confirmation_already_persisted: bool = False,
) -> None:
    """Persist the human-confirmed single-table input as the active workspace.

    A one-table data-join intentionally creates no derived dataset. Without
    this binding, the UI says the join was complete while downstream Labeling,
    Feature and Strategy journeys have no selectable active data at all.
    """

    try:
        binding = registry.verify_dataset_binding(authenticated_binding)
        dataset = registry.get(dataset_id)
        if (
            dataset.task_id != task.id
            or binding.task_id != task.id
            or binding.dataset_id != dataset.id
            or dataset.content_hash != binding.content_hash
        ):
            raise JoinSetupError("确认的数据集不属于当前任务。")
        repository = DataWorkspaceRepository(runtime.settings.db_path)
        snapshot = repository.get_or_default(task.id)
        normalized_target = str(target_col or "").strip() or None
        confirmed_mapping = DataSemanticMapping(
            target_col=normalized_target,
            field_roles=(
                {normalized_target: "target"}
                if normalized_target is not None
                else {}
            ),
            business_names={},
        )
        if snapshot.active_dataset_id is not None:
            if (
                snapshot.active_dataset_id != dataset.id
                or snapshot.active_dataset_content_hash != dataset.content_hash
            ):
                raise JoinSetupError(
                    "当前 DataWorkspace 已绑定另一份数据，请刷新并重新确认。"
                )
            if snapshot.semantic_mapping != confirmed_mapping:
                raise JoinSetupError(
                    "当前 DataWorkspace 的语义映射已变化，请刷新并重新确认。"
                )
            if confirmation_already_persisted:
                return
            if on_connection is not None:
                with registry.transaction() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        """
                        SELECT active_dataset_id, active_dataset_content_hash,
                               semantic_mapping_json
                          FROM data_workspaces
                         WHERE task_id = ?
                        """,
                        (task.id,),
                    ).fetchone()
                    if row is None:
                        raise JoinSetupError(
                            "当前 DataWorkspace 绑定已变化，请刷新并重新确认。"
                        )
                    persisted_mapping = data_semantic_mapping_from_dict(
                        json.loads(str(row["semantic_mapping_json"]))
                    )
                    if (
                        str(row["active_dataset_id"] or "") != dataset.id
                        or str(row["active_dataset_content_hash"] or "")
                        != dataset.content_hash
                        or persisted_mapping != confirmed_mapping
                    ):
                        raise JoinSetupError(
                            "当前 DataWorkspace 绑定已变化，请刷新并重新确认。"
                        )
                    registry.verify_authenticated_binding_snapshot(binding)
                    on_connection(conn)
                    registry.verify_authenticated_binding_snapshot(binding)
            return
        if confirmation_already_persisted:
            raise JoinSetupError(
                "当前 DataWorkspace 绑定已变化，请刷新并重新确认。"
            )
        repository.save_initial_binding(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=dataset.id,
                active_dataset_content_hash=dataset.content_hash,
                page="overview",
                selected_field=normalized_target,
                semantic_mapping=confirmed_mapping,
            ),
            expected_revision=snapshot.revision,
            audit={
                "actor": "user:data-join-c1",
                "detail": {
                    "reason": "bind human-confirmed single-table input",
                    "dataset_id": dataset.id,
                    "dataset_content_hash": dataset.content_hash,
                    "target_col": normalized_target,
                },
            },
            dataset_authenticator=lambda: (
                registry.verify_authenticated_binding_snapshot(binding)
            ),
            on_connection=on_connection,
        )
    except JoinSetupError:
        raise
    except (
        DataWorkspaceDataError,
        DataWorkspaceDatasetNotFound,
        DataWorkspaceRevisionConflict,
        DatasetContentDriftError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise JoinSetupError(
            "单表确认期间数据或 DataWorkspace 已变化，请刷新后重新确认。"
        ) from exc

_JOIN_SPEC = _TurnHandlerSpec(
    intent="data_join",
    setup_error_types=(JoinSetupError,),
    error_label="数据拼接出错",
    run_setup=_run_join_setup,
    format_user_display=_c1_display_text,
)

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
