"""feature driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from marvis.agent.feature_setup import FeatureSetupError, FeatureTargetChoiceRequired, build_feature_proposal, infer_meaning_directions
from marvis.agent.join_setup import AuthenticatedJoinSelection, C1TargetValidationError, JoinSetupError, authenticate_join_selection
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_AUTO, CONFIRMATION_SOURCE_HUMAN
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskRecord

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _TurnHandlerSpec
    from . import _c1_expected_content_hashes
    from . import _c1_semantic_snapshot_matches
    from . import _has_c1_semantic_authorization
    from . import _identity_display_text
    from . import _ingest_notice_text
    from . import _modeling_data_runtime
    from . import _parse_c1_reply
    from . import _run_driver_turn
    from . import _validated_authenticated_c1_target
    from . import append_join_error
    from . import join_turn_response

def run_feature_driver_turn(
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
        _FEATURE_SPEC,
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

def _run_feature_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict | tuple:
    backend, registry = _modeling_data_runtime(runtime.settings)
    target_state = _latest_feature_target_state(repo.list_agent_messages(task.id))
    configured_target = str(getattr(task, "target_col", "") or "").strip()
    target_candidates = [
        str(item).strip()
        for item in ((target_state or {}).get("target_candidates") or [])
        if str(item).strip()
    ]
    # Conversation metadata is append-only. Once a choice has been persisted on
    # the task, an older target-choice card must not capture every later turn
    # and ask the user to choose the same target again.
    if configured_target and configured_target in target_candidates:
        target_state = None
    target_assignment: dict | None = None
    feature_c1_selection: AuthenticatedJoinSelection | None = None
    if target_state is not None:
        target_assignment = _parse_c1_reply(
            user_text,
            target_state,
            llm_client=runtime.llm_client,
            trusted_ui_action=(
                runtime.ui_action == "confirm_roles"
                or confirmation_source == CONFIRMATION_SOURCE_AUTO
            ),
            require_semantic_authorization=(
                runtime.require_semantic_text_authorization
            ),
        )
        configured_target = str(
            (target_assignment or {}).get("target_col") or ""
        ).strip()
        if not configured_target:
            candidates = "、".join(
                f"`{item}`" for item in target_state.get("target_candidates") or []
            )
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"不能只确认：请选择一个目标列{f'（{candidates}）' if candidates else ''}。",
                metadata={
                    "join_c1": target_state,
                    "feature_target_choice": {
                        "candidates": list(target_state.get("target_candidates") or []),
                    },
                },
            )
            return join_turn_response(repo, task.id)
        if _has_c1_semantic_authorization(target_assignment):
            current_dataset = registry.get(str(target_assignment["anchor_id"]))
            current_state = {
                **target_state,
                "files": [
                    {
                        **item,
                        "content_hash": current_dataset.content_hash,
                    }
                    for item in target_state.get("files") or []
                    if isinstance(item, dict)
                ],
            }
            if not _c1_semantic_snapshot_matches(target_assignment, current_state):
                raise FeatureSetupError(
                    "目标列复核期间数据快照已变化，请刷新后重新选择。"
                )
        feature_c1_selection = authenticate_join_selection(
            registry,
            task.id,
            anchor_id=str(target_assignment["anchor_id"]),
            feature_ids=[],
            expected_content_hashes=_c1_expected_content_hashes(target_state),
        )
        try:
            configured_target = _validated_authenticated_c1_target(
                registry,
                feature_c1_selection,
                configured_target,
            ) or ""
            target_assignment["target_col"] = configured_target or None
        except C1TargetValidationError as exc:
            if runtime.ui_action == "confirm_roles":
                raise
            return append_join_error(repo, task.id, str(exc))
    try:
        proposal = build_feature_proposal(
            registry,
            backend,
            task.id,
            task.source_dir,
            metrics=_feature_metrics(task),
            configured_target=configured_target,
            configured_features=list(getattr(task, "feature_columns", None) or []),
        )
    except FeatureTargetChoiceRequired as exc:
        dataset = registry.get(exc.dataset_id)
        state = _feature_target_choice_state(
            exc,
            content_hash=str(dataset.content_hash or ""),
        )
        candidates = "、".join(f"`{item}`" for item in exc.candidates)
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"数据集 `{exc.dataset_name}` 检测到多个合法目标列：{candidates}。"
                "请回复要使用的目标列名；手动模式也可在下方选择后确认。"
            ),
            metadata={
                "join_c1": state,
                "feature_target_choice": {"candidates": exc.candidates},
            },
        )
        return join_turn_response(repo, task.id)
    persist_target_col = (
        configured_target
        if configured_target
        and configured_target != str(getattr(task, "target_col", "") or "")
        else None
    )
    if str(getattr(task, "run_mode", "") or "") == "agent":
        proposal.meaning_directions = infer_meaning_directions(
            runtime.llm_client,
            backend,
            registry,
            proposal,
        )
    notices = list(proposal.ingest_notices or [])
    setup_message = {
        "role": "assistant",
        "stage": "chat",
        "content": (
            f"分析数据集 `{proposal.dataset_name}`（目标列 `{proposal.target_col}`，"
            f"{len(proposal.features)} 个候选特征）:"
            f"{_ingest_notice_text(notices)}"
        ),
        "metadata": {"intent": "feature_analysis", "ingest_notices": notices},
    }
    return (
        proposal.template_id,
        proposal.template_slots(),
        {
            **(
                {"_post_start_c1_assignment": target_assignment}
                if _has_c1_semantic_authorization(target_assignment)
                else {}
            ),
            **(
                {"_post_start_feature_target_col": persist_target_col}
                if persist_target_col is not None
                else {}
            ),
            **(
                {
                    "_post_start_c1_target_binding": (
                        registry,
                        feature_c1_selection.anchor,
                        configured_target or None,
                    )
                }
                if feature_c1_selection is not None
                else {}
            ),
            "_post_start_messages": [setup_message],
        },
    )

_FEATURE_SPEC = _TurnHandlerSpec(
    intent="feature_analysis",
    setup_error_types=(FeatureSetupError, JoinSetupError),
    error_label="特征分析出错",
    run_setup=_run_feature_setup,
    format_user_display=_identity_display_text,
)

def _feature_metrics(task: TaskRecord) -> list[str] | None:
    raw = getattr(task, "metrics", None)
    if raw is None:
        return None
    return [str(item).strip() for item in raw if str(item).strip()]

def _latest_feature_target_state(conversation: list[dict]) -> dict | None:
    for message in reversed(conversation):
        metadata = message.get("metadata") or {}
        if metadata.get("feature_target_choice") and isinstance(metadata.get("join_c1"), dict):
            return dict(metadata["join_c1"])
    return None

def _feature_target_choice_state(
    exc: FeatureTargetChoiceRequired,
    *,
    content_hash: str,
) -> dict:
    return {
        "files": [{
            "dataset_id": exc.dataset_id,
            "content_hash": content_hash,
            "name": exc.dataset_name,
            "row_count": "",
            "n_cols": "",
            "has_target": True,
            "candidate_target": None,
            "proposed_role": "anchor",
            "columns": list(exc.candidates),
            "target_candidates": list(exc.candidates),
        }],
        "anchor_id": exc.dataset_id,
        "feature_ids": [],
        "target_col": None,
        "target_candidates": list(exc.candidates),
        "skip": True,
    }

